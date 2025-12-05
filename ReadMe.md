## Method Overview
This solution implements a stabilize–then–detect-and-track pipeline tailored to a challenging top-down 
drone-surveillance video with strong compression artifacts, rolling-shutter “jello,” fast ego-motion, and partial occlusions. 
Each incoming frame is first motion-stabilized relative to the previous one using sparse Lucas–Kanade optical flow followed by an affine transform. 
Tracking is then performed on the stabilized frames with a CSRT tracker, while the resulting bounding box is mapped back to the original (non-stabilized) frame for visualization and video export.
The tracker maintains a continuous bounding box for the target drone and enters a LOST state after a configurable number of missed updates. 
A template-matching–based re-detection module periodically scans either a local neighborhood around the last known position or the full frame to reacquire the drone once tracking confidence drops below a threshold. 
In practice, this re-detection component is the weakest part of the pipeline and would require substantial improvement (e.g., a learned detector such as YOLO) for a better preformance.

Brief Survey of Alternative Approaches:

1. Detector + Tracker (classical hybrid)-
✔ Simple to implement, works without training data.
✖ Redetection reliability is limited for very small, low-contrast objects.

2. Color / Shape Segmentation-
✔ Fast and lightweight; easy to deploy on constrained hardware.
✖ Unstable here due to severe compression, lighting variation, and non-distinctive drone appearance.

3. Motion-based (optical flow / frame differencing)-
✔ Useful when background is static.
✖ Less effective with fast camera pans, jello artifacts, and ambiguous moving textures.

4. Stabilize-then-Track-
✔ Potentially improves motion consistency.
✖ Requires robust homography estimation, which is unreliable in this video due to blur and rapid motion.

Given the video’s characteristics, and considering both the hardware limitations and the short development time available (no time for creating a training dataset), 
a classical hybrid of stabilization, lightweight detection cues, and tracking logic represents the most practical approach.


## Data Plan
This approach does not rely on external datasets or model training.
All parameters were tuned empirically on the provided video, and tracking performance was evaluated directly on the final output.


## How to Run
```python
python run.py --video path/to/drone_video.mp4 --out out.mp4
# Optional:
#   --debug        (Enable visualization and diagnostics)
```
Expected output:
out.mp4 containing per-frame bounding box and track ID overlay, as required.


## KPIs and Runtime / Jetson Feasibility Estimate
Results from the full run:
- Total frames: 14,229
- Total time: 2770.79 s
- FPS: 5.14
-Lost frames: 3610 (25.4%)
- Redetect calls: 3776
- Redetect success rate: 4.4% (166 successful reacquisitions)
- Max LOST streak: 3270 frames
- Avg LOST streak: 97.6 frames
- Mean bbox area: 2048 px
- Mean center motion: 2.92 px/frame
- Jetson feasibility (estimate):
The current classical pipeline is CPU-bound and not optimized for low-power embedded hardware.
However, due to its lightweight nature, a Jetson Xavier NX / Orin should run this method in real time (≥15–20 FPS) 
after modest optimization (vectorization, reduced search windows, stricter re-detection triggers).

## Key Trade-offs and What I’d Do With More Time
1. Hardware limitations & workflow efficiency:
My development machine is relatively slow, and each full run on the video took significant time.
With more time, I would split the long video into short scenario-specific segments (fast motion, occlusions, small appearance, jello distortion).
Tune and validate each component on targeted segments to drastically reduce iteration time.
Maintain structured version control to track parameter changes and experiment results more clearly.

2. Improving the detector for reliable re-detection:
The primary weakness of the classical approach is poor reacquisition when the drone becomes too small or blends into the background.
With more time, I would Collect a small labeled dataset extracted from the video.
Train a lightweight YOLO-based drone detector on this custom dataset.
Run the detector + tracker pipeline on Colab GPU to achieve faster experimentation and significantly more robust re-detection.

3. Project-level open questions requiring clarification:
To fully adapt the solution, several key requirements must be clarified-
- Should initial detection be fully automatic or is a manual first-frame annotation acceptable?
- Does this video cover all operational scenarios expected in the real system?
- Must the algorithm run in real time? If so, what is the FPS requirement?
- What hardware will the final system run on (CPU / GPU / Jetson class / other)?
- Will there always be exactly one drone to track?
- What are the memory constraints?
- How is success defined (IoU threshold, percentage of tracked time, max allowed LOST streak, etc.)?


Clarifying these questions would directly guide whether the system should prioritize runtime, accuracy, robustness, or scalability.


