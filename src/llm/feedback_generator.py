"""LLM 기반 코칭 피드백 생성 모듈.

분석 결과를 자연어 코칭 피드백으로 변환합니다.
Ollama API 연결 시 LLM 응답, 미연결 시 템플릿 기반 피드백을 제공합니다.
"""

import json
from src.models import SessionResult, ThrowAnalysis
from src.config import OLLAMA_BASE_URL, OLLAMA_MODEL, LLM_TIMEOUT


# ─── 이슈 → 한국어 피드백 템플릿 ──────────────────────────────────

_ISSUE_TEMPLATES: dict[str, str] = {
    "elbow_unstable_y": (
        "투구 동안 팔꿈치의 높이가 불안정합니다. "
        "팔꿈치를 고정하고 전완(forearm)만 움직이도록 의식해보세요."
    ),
    "takeback_too_deep": (
        "테이크백(뒤로 당기기)이 너무 깊습니다. "
        "다트를 귀 옆까지만 당기되, 팔꿈치가 얼굴 앞으로 나오지 않도록 해보세요."
    ),
    "takeback_too_shallow": (
        "테이크백이 너무 얕습니다. "
        "좀 더 뒤로 당겨 충분한 가속 구간을 확보하면 정확도가 올라갑니다."
    ),
    "slow_elbow_extension": (
        "릴리즈 시 팔꿈치를 펴는 속도가 느립니다. "
        "팔꿈치를 스냅하듯 빠르게 이어서 목표 지점을 향해 쭉 뻗어주세요."
    ),
    "body_sway_detected": (
        "투구 시 상체가 좌우로 흔들리고 있습니다. "
        "양 발에 체중을 고르게 분배하고, 던지는 동안 상체를 고정해보세요."
    ),
    "shoulder_unstable": (
        "어깨 높이가 투구 중에 변합니다. "
        "어깨를 일정한 높이로 유지하며 전완만으로 던지는 연습을 해보세요."
    ),
    "inconsistent_takeback": (
        "투구마다 테이크백(뒤로 당기는) 각도가 다릅니다. "
        "매번 같은 위치까지 당기는 연습을 통해 일관성을 높여보세요."
    ),
    "inconsistent_elbow_speed": (
        "투구마다 팔꿈치를 펴는 속도가 일정하지 않습니다. "
        "같은 리듬과 템포로 던지는 연습이 정확도 향상에 도움됩니다."
    ),
}


class FeedbackGenerator:
    """코칭 피드백 생성기."""

    def __init__(self):
        self._ollama_available: bool | None = None

    def generate(self, session: SessionResult) -> str:
        """세션 분석 결과를 코칭 피드백 문자열로 변환합니다."""
        if not session.throws:
            return "⚠ 분석된 투구가 없습니다. 영상에서 다트 투구 동작이 인식되지 않았습니다."

        # TODO: 향후 LLM 연동 활성화 (현재는 템플릿 기반 피드백만 사용)
        # if self._is_ollama_available():
        #     llm_feedback = self._generate_with_llm(session)
        #     if llm_feedback:
        #         return llm_feedback

        # 현재는 템플릿 기반 피드백만 제공
        return self._generate_template_feedback(session)

    # ─── Template-based Feedback ──────────────────────────────────

    def _generate_template_feedback(self, session: SessionResult) -> str:
        """템플릿 기반으로 피드백을 생성합니다."""
        lines = []
        lines.append(f"📊 다트 투구 분석 리포트 ({session.total_throws_detected}회 투구)")
        lines.append("=" * 50)

        for throw in session.throws:
            lines.append(f"\n🎯 투구 #{throw.throw_index} ({throw.throwing_arm}손)")
            lines.append(f"   프레임 범위: {throw.frame_range[0]} ~ {throw.frame_range[1]}")

            m = throw.metrics
            lines.append(f"   · 팔꿈치 안정성: {m.elbow_stability_variance:.4f}")
            lines.append(f"   · 테이크백 최소 각도: {m.takeback_min_angle_deg:.1f}°")
            lines.append(f"   · 팔꿈치 펴는 속도: {m.elbow_extension_velocity_deg_s:.1f}°/s")
            lines.append(f"   · 손목 스냅 속도: {m.wrist_snap_velocity_deg_s:.1f}°/s")
            lines.append(f"   · 상체 흔들림: {m.body_sway_x_norm:.4f}")

            if throw.issues:
                lines.append(f"\n   💡 개선 포인트:")
                for issue in throw.issues:
                    feedback = _ISSUE_TEMPLATES.get(issue, f"  (알 수 없는 이슈: {issue})")
                    lines.append(f"     → {feedback}")
            else:
                lines.append("   ✅ 특별한 이슈 없음 — 좋은 폼입니다!")

        # 전체 요약
        all_issues = []
        for t in session.throws:
            all_issues.extend(t.issues)

        if all_issues:
            from collections import Counter
            common = Counter(all_issues).most_common(3)
            lines.append(f"\n{'=' * 50}")
            lines.append("📝 전체 요약 (가장 빈번한 이슈):")
            for issue, count in common:
                lines.append(f"   · {issue}: {count}회 / {session.total_throws_detected}회 투구")
        else:
            lines.append(f"\n{'=' * 50}")
            lines.append("✅ 전체적으로 안정적인 폼입니다. 잘하고 있어요!")

        return "\n".join(lines)

    # ─── LLM-based Feedback ───────────────────────────────────────

    def _is_ollama_available(self) -> bool:
        """Ollama 서버가 실행 중인지 확인합니다."""
        if self._ollama_available is not None:
            return self._ollama_available

        try:
            import urllib.request
            req = urllib.request.Request(f"{OLLAMA_BASE_URL}/api/tags", method="GET")
            with urllib.request.urlopen(req, timeout=3):
                self._ollama_available = True
                return True
        except Exception:
            self._ollama_available = False
            return False

    def _generate_with_llm(self, session: SessionResult) -> str | None:
        """Ollama API를 통해 LLM 피드백을 생성합니다."""
        try:
            import urllib.request

            summary_data = session.to_dict()
            # LLM에게 전달할 간결한 데이터
            compact = {
                "total_throws": summary_data["total_throws_detected"],
                "throws": [
                    {
                        "index": t["throw_index"],
                        "arm": t["throwing_arm"],
                        "metrics": t["metrics"],
                        "issues": t["issues"],
                    }
                    for t in summary_data["throws"]
                ],
            }

            prompt = (
                "당신은 프로 다트 코치입니다. 아래 생체역학 분석 데이터를 보고 "
                "초보 다트 선수에게 구체적이고 친절한 한국어 코칭 피드백을 작성해주세요.\n\n"
                "각 투구마다 잘한 점과 개선할 점을 구분해서 설명하고, "
                "마지막에 전체 요약과 연습 방법을 제안해주세요.\n\n"
                f"분석 데이터:\n{json.dumps(compact, indent=2, ensure_ascii=False)}"
            )

            payload = json.dumps({
                "model": OLLAMA_MODEL,
                "prompt": prompt,
                "stream": False,
            }).encode("utf-8")

            req = urllib.request.Request(
                f"{OLLAMA_BASE_URL}/api/generate",
                data=payload,
                headers={"Content-Type": "application/json"},
                method="POST",
            )

            with urllib.request.urlopen(req, timeout=LLM_TIMEOUT) as resp:
                result = json.loads(resp.read().decode("utf-8"))
                return result.get("response", "")

        except Exception as e:
            print(f"⚠ LLM 피드백 생성 실패 (템플릿으로 대체): {e}")
            return None
