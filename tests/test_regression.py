"""GT 기반 회귀 방지 테스트.

output/ 폴더의 최신 JSON 리포트를 Ground Truth와 비교하여
투구 감지 수가 올바른지 자동으로 검증합니다.

실행 방법:
    .venv/bin/python -m pytest tests/test_regression.py -v
"""

import json
import os
import glob
import unittest

import yaml  # type: ignore


class TestRegressionDetection(unittest.TestCase):
    """GT 기반 투구 감지 수 회귀 테스트.

    tests/ground_truth.yaml의 기대 투구 횟수와
    output/ 폴더의 최신 JSON 리포트를 비교합니다.
    """

    @classmethod
    def setUpClass(cls) -> None:
        """GT 라벨과 최신 리포트 로드."""
        # GT 라벨 로드
        gt_path = os.path.join(os.path.dirname(__file__), "ground_truth.yaml")
        with open(gt_path, "r") as f:
            cls.gt = yaml.safe_load(f)["labels"]

        # output 폴더에서 각 샘플의 최신 리포트 찾기
        cls.reports: dict[str, dict] = {}
        output_dir = os.path.join(os.path.dirname(__file__), "..", "output")

        for sample_name in cls.gt:
            # 해당 샘플의 모든 리포트 찾기 (예: sample_10_report_*.json)
            pattern = os.path.join(output_dir, f"{sample_name}_report_*.json")
            matches = glob.glob(pattern)

            if matches:
                # 파일명의 타임스탬프 부분으로 정렬하여 최신 것 선택
                latest = max(matches, key=os.path.getmtime)
                with open(latest, "r") as f:
                    cls.reports[sample_name] = {
                        "data": json.load(f),
                        "path": latest,
                    }

    def testAllSamplesHaveReports(self) -> None:
        """모든 GT 샘플에 대해 리포트가 존재하는지 확인."""
        missing = [s for s in self.gt if s not in self.reports]
        self.assertEqual(
            len(missing), 0,
            f"리포트 없는 샘플: {missing}. `python src/main.py --all` 실행 필요."
        )

    def testThrowCountAccuracy(self) -> None:
        """투구 감지 수가 GT와 일치하는 비율을 측정합니다.

        현재는 정보 제공 목적이며, 최소 통과 기준을 설정합니다.
        """
        correct = 0
        total = 0
        failures: list[str] = []

        for sample_name, expected_count in self.gt.items():
            if sample_name not in self.reports:
                continue

            total += 1
            actual_count = self.reports[sample_name]["data"]["total_throws_detected"]

            if actual_count == expected_count:
                correct += 1
            else:
                failures.append(
                    f"  {sample_name}: 기대={expected_count}, 실제={actual_count}"
                )

        accuracy = correct / total * 100 if total > 0 else 0

        print(f"\n{'='*50}")
        print(f"📊 투구 감지 정확도: {correct}/{total} ({accuracy:.1f}%)")
        if failures:
            print(f"❌ 실패 케이스:")
            for f in failures:
                print(f)
        print(f"{'='*50}")

        # 최소 통과 기준: 25% (현재 수준에서 회귀 감지용)
        # Phase 4 이후 이 기준을 점진적으로 올릴 것
        self.assertGreaterEqual(
            accuracy, 25.0,
            f"투구 감지 정확도 {accuracy:.1f}%가 최소 기준 25%에 미달."
        )

    def testMetricsNotAllZero(self) -> None:
        """성공한 리포트의 메트릭이 전부 0.0이 아닌지 확인.

        Phase 1에서 해결한 메트릭 0.0 회귀를 방지합니다.
        """
        zero_metric_reports: list[str] = []

        for sample_name, report_info in self.reports.items():
            data = report_info["data"]
            throws = data.get("throws", [])

            for throw in throws:
                metrics = throw.get("metrics", {})
                # 핵심 각도 메트릭이 모두 0이면 문제
                key_metrics = [
                    metrics.get("takeback_angle_deg", 0),
                    metrics.get("max_elbow_velocity_deg_s", 0),
                ]
                if all(abs(m) < 0.01 for m in key_metrics):
                    zero_metric_reports.append(
                        f"  {sample_name} throw {throw['throw_index']}: "
                        f"takeback={metrics.get('takeback_angle_deg', 0):.1f}°, "
                        f"velocity={metrics.get('max_elbow_velocity_deg_s', 0):.1f}°/s"
                    )

        if zero_metric_reports:
            print(f"\n⚠ 메트릭 0.0 감지:")
            for r in zero_metric_reports:
                print(r)

        # 전체 투구 중 50% 이상이 0.0 메트릭이면 회귀로 판단
        total_throws = sum(
            len(r["data"].get("throws", []))
            for r in self.reports.values()
        )
        if total_throws > 0:
            zero_ratio = len(zero_metric_reports) / total_throws
            self.assertLess(
                zero_ratio, 0.5,
                f"메트릭 0.0 비율 {zero_ratio:.0%}가 50% 이상 — Phase 1 회귀 의심."
            )

    def testReleaseTimingNotThirtyThreeMs(self) -> None:
        """릴리즈 타이밍이 항상 33ms(1프레임)인 회귀를 방지합니다."""
        always_33ms: list[str] = []

        for sample_name, report_info in self.reports.items():
            throws = report_info["data"].get("throws", [])
            for throw in throws:
                timing = throw.get("metrics", {}).get("release_timing_ms", 0)
                # 30fps에서 1프레임 = 33.3ms. 35ms 이하면 문제.
                if 0 < timing < 35:
                    always_33ms.append(
                        f"  {sample_name} throw {throw['throw_index']}: "
                        f"release_timing={timing:.1f}ms"
                    )

        # 전체 투구의 25% 이상이 33ms면 회귀
        total_throws = sum(
            len(r["data"].get("throws", []))
            for r in self.reports.values()
        )
        if total_throws > 0:
            ratio = len(always_33ms) / total_throws
            self.assertLess(
                ratio, 0.25,
                f"릴리즈 타이밍 1프레임 비율 {ratio:.0%}가 25% 이상 — Phase Detection 회귀 의심."
            )


if __name__ == "__main__":
    unittest.main()
