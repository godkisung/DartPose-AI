# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

AI Dart Coach — a computer vision system that analyzes dart throw posture and provides biomechanical feedback. Currently a Python prototype; the long-term target is iOS on-device deployment.

## Commands

### Setup
```bash
uv sync
```

### Run Analysis

Single video (auto-selects latest file in `data/`):
```bash
uv run python src/main.py
```

Specific video with debug plots:
```bash
uv run python src/main.py --input data/sample_10.mp4 --debug-plot
```

Batch process all videos:
```bash
uv run python src/main.py --all --debug-plot
```

### Testing
```bash
# All tests
uv run pytest tests/ -v

# Regression tests only (compares detected throw count vs ground truth)
uv run pytest tests/test_regression.py -v
```

## Architecture

### 2-Pass Pipeline

**Pass 1 (Analysis)**:
```
Video → [video_utils: codec conversion] → [PoseExtractor] → FrameData[]
                                                              ↓
                                                        [DartAnalyzer]
                                                         ├─ PoseNormalizer
                                                         ├─ ThrowSegmenter (FSM)
                                                         ├─ PhaseDetector
                                                         └─ MetricsCalculator
                                                              ↓
                                                        SessionResult → JSON
```

**Pass 2 (Visualization)**:
```
SessionResult + skeleton_video → [TrajectoryRenderer] → trajectory_video.mp4
```

### Key Design Decisions

- **Typed data contracts**: All inter-module communication uses dataclasses defined in `src/models.py`. Never pass raw dicts between pipeline stages.
- **Threshold centralization**: All biomechanical thresholds live in `src/config.py`. No hardcoded magic numbers in vision modules.
- **Angles vs positions**: Metrics use raw (un-normalized) keypoints for angle computations, but normalized coordinates for position/displacement metrics.
- **FSM throw detection**: `ThrowSegmenter` uses a Finite State Machine (IDLE → COCKING → RELEASING → FOLLOW_THROUGH) with hysteresis to handle noisy angle signals. Gaussian smoothing (σ=0.06s) is applied before state transitions.
- **iOS portability**: Analysis logic in `src/vision/` is intentionally decoupled from MediaPipe. The plan is to replace `PoseExtractor` with Apple Vision framework while keeping the rest unchanged.

### Module Responsibilities

| Module | Role |
|--------|------|
| `src/main.py` | CLI entry point, orchestrates 2-pass execution |
| `src/models.py` | Dataclass schemas (`FrameData`, `ThrowAnalysis`, `SessionResult`, etc.) |
| `src/config.py` | All thresholds and hyperparameters |
| `src/vision/mediapipe_pose.py` | Joint extraction (MediaPipe Pose + Hands), outputs skeleton video |
| `src/vision/dart_analyzer.py` | Master orchestrator for the analysis pipeline |
| `src/vision/throw_segmenter.py` | FSM-based throw boundary detection |
| `src/vision/phase_detector.py` | 4-phase detection within a throw (Address → Takeback → Release → Follow-through) |
| `src/vision/metrics_calculator.py` | Computes 10+ biomechanical metrics per throw |
| `src/vision/pose_normalizer.py` | Camera-motion correction + coordinate normalization |
| `src/vision/trajectory_renderer.py` | Overlay throw trajectories on skeleton video |
| `src/llm/feedback_generator.py` | Template-based Korean coaching feedback (Ollama LLM integration is stubbed out) |
| `src/utils/video_utils.py` | AV1/WebM → H.264 conversion via ffmpeg |

### Testing Infrastructure

Regression tests in `tests/test_regression.py` compare detected throw counts against `tests/ground_truth.yaml`. Ground truth videos are in `data_ver2/` (samples 3–14) and `data/` (samples 15–20); all currently expect 3 throws per video.

Legacy modules `src/vision/rule_engine.py` and `src/vision/advanced_engine.py` are superseded by the current pipeline but kept for reference.
