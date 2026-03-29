// Config.swift
// 자동 생성된 파일입니다. 수동으로 수정하지 마세요.
// 생성 도구: tools/sync_config.py

import Foundation

enum DartConfig {
    static let mediapipeModelComplexity: Int = 2
    static let mediapipeMinDetectionConf: Double = 0.5
    static let mediapipeMinTrackingConf: Double = 0.5
    static let normalizerSmoothingWindow: Int = 5
    static let segmenterSmoothingSigma: Double = 0.06
    static let segmenterAngleDropThreshold: Double = 7.0
    static let segmenterReleaseAngleRise: Double = 10.0
    static let segmenterIdleStabilityFrames: Int = 8
    static let segmenterMinCockingAngle: Double = 6.0
    static let segmenterMinSegmentFrames: Int = 10
    static let segmenterMinThrowIntervalS: Double = 0.7
    static let segmenterMaxSegmentDurationS: Double = 4.0
    static let segmenterSegmentPadS: Double = 0.5
    static let segmenterMergeGapFrames: Int = -1
    static let segmenterLegacyMinPeakProminence: Double = 0.010
    static let segmenterLegacyMinPeakDistance: Double = 1.5
    static let segmenterLegacySegmentExpandFrames: Double = 1.2
    static let metricsMinForearmVecNorm: Double = 1e-5
    static let segmenterMinWristDisplacement: Double = 0.04
    static let validationMinWristDisplacement: Double = 0.04
    static let validationMinTakebackAngle: Double = 5.0
    static let validationMaxTakebackAngle: Double = 165.0
    static let validationMinElbowVelocity: Double = 0.0
    static let validationMaxElbowVelocity: Double = 1500.0
    static let validationMinReleaseTimingMs: Double = 100.0
    static let validationMaxReleaseTimingMs: Double = 1200.0
    static let validationMinRomAngle: Double = 30.0
    static let elbowStabilityThreshold: Double = 0.005
    static let takebackMinAngle: Int = 30
    static let takebackMaxAngle: Int = 110
    static let elbowExtensionVelMin: Int = 150
    static let bodySwayThreshold: Double = 0.05
    static let shoulderStabilityThreshold: Double = 0.003
    static let throwMinFrames: Int = 15
    static let ollamaBaseUrl: String = "http://localhost:11434"
    static let ollamaModel: String = "llama3"
    static let llmTimeout: Int = 30
    static let velocitySmoothingWindow: Int = 7
    static let throwIdleFrames: Int = 10
}
