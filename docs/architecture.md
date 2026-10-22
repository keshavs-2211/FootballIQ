# FootballIQ architecture (one active system)

`run.py` is the single terminal launcher; `python run.py web` runs the only Flask user interface (`src/footballiq_web.py`). Both use `src/web_pipeline_entry.py` as the shared adapter for match paths and processing stages. Every stage runs in a new Python process for clean clip-specific settings and reproducible outputs.

1. **Upload/import or offline input** — `sample_data/input_videos/<clip>.mp4`.
2. **References** — 3 crops each of Team A, Team B, Ball stored in `references/<clip>/`; optional GK crops.
3. **Calibration** — points selected from a chosen video frame in the web app and converted to frame-0 homography using camera alignment. Offline OpenCV calibration selects frame 0.
4. **Detection** — `src/process_video.py` loads `models/best.pt`, three saved jerseys per team, tracks players using ByteTrack, saves `*_improved_coordinates.csv` and `*_improved.mp4`.
5. **Refinement** — `src/refine_coordinates.py` projects to a 105 × 68 m tactical pitch, applies camera compensation, smooths paths, exports reviewable ID-merge proposals without auto-merging by default.
6. **Analytics** — `src/run_analytics.py` builds player/team distances, heatmaps, team shapes, and dashboard images.
7. **Optional ball tracking** — `src/track_ball.py` ranks YOLO ball candidates against all three ball references with motion validation, exports pitch positions plus *measured* original video pixels and audit report.
8. **Ball overlay** — `src/annotate_ball.py` draws only measured/reference-verified ball candidates over the player-annotated MP4. Estimated-only positions are not represented as detections.
9. **Tactical render** — `src/render_refined_birdeye.py` generates bird's-eye and side-by-side videos, using ball data when available.
10. **Optional possession** — `src/run_possession.py` processes confidence-gated ball evidence; `src/render_possession_birdeye.py` creates a possession replay if run explicitly.
11. **Browser media** — the Flask backend transcodes source videos using FFmpeg H.264 into parallel `*_web.mp4` files to improve Chrome/Firefox playback.

## Integrity principles

Original videos and generated native-codec MP4s are never overwritten by the browser converter. Ball references are a *post-YOLO candidate filter*, not new YOLO training. Analytics statistics are visibility-based estimates, not verified full-match performance. Camera cuts, long occlusions, and poor references may invalidate tracking and calibration.
