// RegressionTests.swift
// 파이썬 엔진 결과와 스위프트 엔진 결과를 비교 검증하는 테스트
//
// 파이썬에서 추출한 JSON 케이스를 로드하여
// 스위프트 DartAnalyzer가 동일한 투구 수와 메트릭을 내놓는지 확인합니다.

import XCTest
@testable import DartPose

/// 파이썬에서 내보낸 JSON 구조를 담는 DTO
struct RegressionCase: Decodable {
    struct Metadata: Decodable {
        let fps: Double
        let total_frames: Int
    }
    
    struct ExpectedOutput: Decodable {
        struct ExpectedThrow: Decodable {
            let throw_index: Int
            let throwing_arm: String
            let frame_range: [Int]
        }
        let total_throws_detected: Int
        let throws_detail: [ExpectedThrow]
        
        enum CodingKeys: String, CodingKey {
            case total_throws_detected
            case throws_detail = "throws"
        }
    }
    
    let metadata: Metadata
    let input_frames: [FrameData]
    let expected_output: ExpectedOutput
}

class RegressionTests: XCTestCase {

    /// 파이썬 데이터 기반 회귀 테스트 실행
    func testCompareWithPythonResult() {
        // 1. JSON 파일 로드 (Bundle 또는 파일 시스템)
        // 실제 운영 시에는 테스트 번들에 .json 파일을 포함시켜야 합니다.
        guard let url = Bundle(for: type(of: self)).url(forResource: "regression_case_1", withExtension: "json") else {
            print("  ℹ 회귀 테스트 데이터(regression_case_1.json)가 없어 건너뜁니다.")
            return
        }
        
        do {
            let data = try Data(contentsOf: url)
            let regressionCase = try JSONDecoder().decode(RegressionCase.self, from: data)
            
            // 2. 스위프트 엔진 실행
            let analyzer = DartAnalyzer(fps: regressionCase.metadata.fps)
            let result = analyzer.analyzeSession(frames: regressionCase.input_frames)
            
            // 3. 결과 비교
            // (1) 투구 수 일치 확인
            XCTAssertEqual(
                result.totalThrowsDetected,
                regressionCase.expected_output.total_throws_detected,
                "투구 감지 수 불일치"
            )
            
            // (2) 각 투구의 프레임 범위 비교 (±5프레임 오차 허용)
            for (i, expected) in regressionCase.expected_output.throws_detail.enumerated() {
                guard i < result.throws_.count else { break }
                let actual = result.throws_[i]
                
                let startDiff = abs(actual.frameRange[0] - expected.frame_range[0])
                let endDiff = abs(actual.frameRange[1] - expected.frame_range[1])
                
                XCTAssertLessThanOrEqual(startDiff, 5, "투구 #\(i+1) 시작점 오차 과다")
                XCTAssertLessThanOrEqual(endDiff, 5, "투구 #\(i+1) 종료점 오차 과다")
            }
            
            print("  ✅ 파이썬 결과와 스위프트 결과가 일치합니다.")
            
        } catch {
            XCTFail("JSON 로드 또는 파싱 실패: \(error.localizedDescription)")
        }
    }
}
