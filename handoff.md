# Handoff Notes

## Completed Today

### L/R Detection Rewrite (`src/vision/dart_analyzer.py`)

Rewrote `_detectThrowingSide` to leverage the strict 90-degree side-profile camera constraint.

The old logic used hand landmark detection counts + wrist variance heuristics, which failed frequently on diagonal views. The new implementation uses a **3-pillar majority vote** across all valid frames:

1. **Visibility** — arm joints (`shoulder`, `elbow`, `wrist`) on the throwing side have significantly higher MediaPipe `visibility` values when filmed from the correct side
2. **Depth (Z-axis)** — throwing arm is closer to the camera → lower (more negative) mean Z
3. **Variance** — throwing wrist XY coordinate variance is drastically higher than the non-throwing wrist

Each pillar casts ±1 vote. Final result: `right` if votes ≥ 0, `left` if votes < 0. The log line now shows all three decisions for easy debugging.

**Related changes required to support storing `visibility` as `coords[3]`:**
- `mediapipe_pose.py`: pose landmarks now stored as `[x, y, z, visibility]`
- All 5 joint extractors (`throw_segmenter`, `phase_detector`, `metrics_calculator`, `pose_normalizer`, `debug_plotter`) updated to slice `val[:3]` so the extra element is ignored downstream

---

## Next Steps (Tomorrow)

### 1. Run Batch Test

```bash
uv run python src/main.py --all --debug-plot
```

Compare detected throw counts against ground truth in `tests/ground_truth.yaml`.

### 2. Fix Remaining False Negatives

From the last batch run, known issues to resolve:

| Sample | Problem | Likely fix |
|--------|---------|-----------|
| sample_17 | Only 2/3 throws detected | Check FSM debug output — likely `SEGMENTER_ANGLE_DROP_THRESHOLD` (10°) or `SEGMENTER_MIN_COCKING_ANGLE` (12°) is too tight for the 3rd throw |
| sample_18 | Still 0 throws (0.0° takeback) | Phase detector `argmin` fix was applied — verify it now picks a valid frame; if not, check raw elbow angle signal in debug plot |

**Tuning knobs in `config.py`:**
- `SEGMENTER_ANGLE_DROP_THRESHOLD` (currently `10.0°`) — lower to `7.0°` if FSM misses a slow cocking motion
- `SEGMENTER_MIN_COCKING_ANGLE` (currently `12.0°`) — lower to `8.0°` if a valid cycle is being discarded
- `VALIDATION_MAX_ELBOW_VELOCITY` (currently `2000.0°/s`) — verify this doesn't accidentally filter any legitimate fast throws

### 3. Regression Test

```bash
uv run pytest tests/test_regression.py -v
```

All samples (3–20) should pass with exactly 3 throws detected.
