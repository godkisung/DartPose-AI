# GEMINI.md - Project Context: AI Dart Coach 🎯

This file provides essential context for Gemini to understand and interact with the AI Dart Coach project.

## 🎯 Project Overview

**AI Dart Coach** is a computer vision-based system designed to analyze dart throw posture and provide real-time biomechanical feedback. 

- **Primary Goal:** Verify biomechanical analysis algorithms and rule engines using a Python prototype.
- **Final Target:** Implementation as an iOS on-device application (using Vision/CoreML).
- **Core Concept:** Extract joint coordinates (Pose), segment individual throws from continuous video, calculate biomechanical metrics, and generate coaching feedback.

### 🛠 Main Technologies
- **Language:** Python 3.11+
- **Pose Estimation:** [MediaPipe Pose](https://developers.google.com/mediapipe/solutions/vision/pose_landmarker)
- **Computer Vision:** OpenCV
- **Package Manager:** `uv`
- **Video Processing:** `ffmpeg` (for AV1 to H.264 conversion via `src/utils/video_utils.py`)
- **Analysis Foundation:** Biomechanical metrics based on *Huang et al. (2024)*.

---

## 📂 Project Structure

- `src/main.py`: The system entry point and integration pipeline.
- `src/models.py`: Defines data schemas using Python `dataclass` (FrameData, ThrowAnalysis, etc.).
- `src/config.py`: Centralized management of rule engine thresholds and constants.
- `src/vision/`:
    - `mediapipe_pose.py`: Extracts 3D joint coordinates using MediaPipe.
    - `rule_engine.py`: Implements throw segmentation, phase detection, and metric calculation.
- `src/llm/`:
    - `feedback_generator.py`: Generates coaching feedback using templates (LLM skeleton exists but template-based is prioritized).
- `src/utils/`:
    - `video_utils.py`: Handles codec compatibility (converts AV1/other formats to H.264).
- `tests/`: Contains unit tests for the rule engine and other components.
- `paper/`: Reference research papers for biomechanical analysis.

---

## 🚀 Building and Running

### Dependency Management
The project uses `uv` for fast and reliable dependency management.
```bash
# Sync dependencies
uv sync
```

### Running Analysis
Analyze a video file to get biomechanical feedback and a JSON report.
```bash
# Basic run
uv run python src/main.py --input data/sample_1.mp4

# Run with skeleton overlay video and custom JSON output
uv run python src/main.py --input data/sample_1.mp4 \
  --output-video output/result.mp4 \
  --output-json output/report.json
```

### Running Tests
```bash
uv run pytest tests/ -v
```

---

## 📐 Development Conventions & Architecture

1. **Data Driven:** All communication between modules should use the data classes defined in `src/models.py`.
2. **Threshold Management:** Avoid hardcoding constants in logic files. Use `src/config.py` for all biomechanical thresholds (e.g., `ELBOW_STABILITY_THRESHOLD`).
3. **Modular for Migration:** Keep analysis logic in `rule_engine.py` decoupled from MediaPipe-specific code to facilitate future migration to Swift (iOS Vision Framework).
4. **Throw Segmentation:** The system automatically detects and segments multiple throws from a single video based on wrist velocity.
5. **Phase Detection:** A single throw is analyzed through 4 phases: *Address* → *Takeback* → *Release* → *Follow-through*.

### Key Biomechanical Metrics
- **Elbow Stability:** Variance of elbow Y-coordinate during the throw.
- **Takeback Angle:** Minimum angle between shoulder, elbow, and wrist.
- **Release Velocity:** Angular velocity of elbow extension at the moment of release.
- **Body Sway:** X-axis displacement of the shoulder.
- **Consistency:** Variation of metrics across multiple throws in a single session.

---

## 📋 TODOs / Future Work
- [ ] Migrate core rule engine logic to Swift for iOS prototype.
- [ ] Enhance phase detection using peak detection algorithms (e.g., `scipy.signal.find_peaks`).
- [ ] Integrate Ollama/LLM for more natural language feedback generation.
