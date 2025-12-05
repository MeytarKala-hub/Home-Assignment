import cv2
import argparse
from drone_tracker import DroneTracker
import time

def stabilize_frame(prev_gray, frame):
    """
    Stabilize the current frame relative to the previous frame using
    optical flow and an affine transform.
    Returns: (frame_stab, gray_stab, M)
    M is the 2x3 matrix that maps frame -> frame_stab.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    pts_prev = cv2.goodFeaturesToTrack(prev_gray,
                                       maxCorners=200,
                                       qualityLevel=0.01,
                                       minDistance=30)
    if pts_prev is None:
        return frame, gray, None

    pts_curr, st, err = cv2.calcOpticalFlowPyrLK(prev_gray, gray, pts_prev, None)
    if pts_curr is None:
        return frame, gray, None

    good_prev = pts_prev[st == 1]
    good_curr = pts_curr[st == 1]

    if len(good_prev) < 10:
        return frame, gray, None

    M_est, _ = cv2.estimateAffinePartial2D(good_prev, good_curr)
    
    if M_est is None:
        return frame, gray, None
    
    # Now M truly maps frame -> frame_stab
    M = cv2.invertAffineTransform(M_est)

    h, w = frame.shape[:2]
    frame_stab = cv2.warpAffine(frame, M, (w, h))
    gray_stab = cv2.cvtColor(frame_stab, cv2.COLOR_BGR2GRAY)
    return frame_stab, gray_stab, M


def transform_bbox_back(bbox_stab, M):
    """
    Map a bounding box from the stabilized frame back to the original frame.
    bbox_stab: (x, y, w, h) in stabilized coordinates.
    M: 2x3 matrix mapping frame -> frame_stab.
    """
    if M is None:
        # No matrix available – return the bbox as-is (approximation)
        return bbox_stab

    x, y, w, h = bbox_stab
    # Four corners of the bbox in stabilized coordinates
    pts_stab = cv2.transform(
        np.array([[[x, y],
                   [x + w, y],
                   [x, y + h],
                   [x + w, y + h]]], dtype=np.float32),
        cv2.invertAffineTransform(M)
    )
    pts = pts_stab[0]

    x_min = int(pts[:, 0].min())
    x_max = int(pts[:, 0].max())
    y_min = int(pts[:, 1].min())
    y_max = int(pts[:, 1].max())

    return (x_min, y_min, x_max - x_min, y_max - y_min)


def main(args):
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print("Could not open video:", args.video)
        return

    # Video writer – saves the original, non-stabilized video
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(args.out, fourcc, fps, (width, height))

    tracker = DroneTracker(max_missed=20, reinit_threshold=0.7)

    # First frame reading
    ret, frame = cap.read()
    if not ret:
        print("Empty video")
        return

    # First frame – no stabilization (no previous frame)
    stabilized_frame = frame.copy()
    prev_gray = cv2.cvtColor(stabilized_frame, cv2.COLOR_BGR2GRAY)

    # Manually selecting ROI on the stabilized frame (which equals the first frame)
    if args.manual_init:
        bbox = cv2.selectROI("Select drone", stabilized_frame,
                             fromCenter=False, showCrosshair=True)
        cv2.destroyWindow("Select drone")
        print("INIT BBOX:", bbox)
    else:
        bbox = (954, 401, 40, 46)

    ok = tracker.init(stabilized_frame, bbox)
    if not ok:
        print("Failed to initialize tracker")
        return

    frame_idx = 0

    # --- METRICS init ---
    total_frames = 0
    tracked_frames = 0
    lost_frames = 0

    current_lost_streak = 0
    lost_streaks = []

    bbox_area_sum = 0.0
    bbox_area_count = 0

    center_motion_sum = 0.0
    center_motion_count = 0
    prev_center = None

    start_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        total_frames += 1

        # Keep a copy of the original frame for drawing and writing
        original_frame = frame.copy()

        # STABILIZATION – run the tracker on the stabilized frame
        stabilized_frame, prev_gray, M = stabilize_frame(prev_gray, frame)

        # CSRT tracking on the stabilized frame
        success, tbbox_stab = tracker.update(stabilized_frame)

        if success:
            # Map the bbox back to the original frame
            x, y, w, h = transform_bbox_back(tbbox_stab, M)

            # Draw bbox on the original frame
            cv2.rectangle(original_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(original_frame, f"ID: {tracker.track_id}",
                        (x, max(0, y - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                        # --- METRICS: successful frame ---
            tracked_frames += 1

            # If we were in a LOST streak – close it
            if current_lost_streak > 0:
                lost_streaks.append(current_lost_streak)
                current_lost_streak = 0

            # Bounding box area
            area = max(1, w * h)
            bbox_area_sum += area
            bbox_area_count += 1

            # Bounding box center motion
            cx = x + w / 2.0
            cy = y + h / 2.0
            if prev_center is not None:
                dx = cx - prev_center[0]
                dy = cy - prev_center[1]
                dist = (dx**2 + dy**2) ** 0.5
                center_motion_sum += dist
                center_motion_count += 1
            prev_center = (cx, cy)

        else:
            cv2.putText(original_frame, "Lost", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
            
            # --- METRICS: lost frame ---
            lost_frames += 1
            current_lost_streak += 1
            # Do not connect motion across holes
            prev_center = None  

        # Write the original frame (with annotations) to the output video
        out.write(original_frame)

        # If no live display is needed – simply avoid passing --debug
        if args.debug:
            cv2.imshow("tracking", original_frame)
            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC
                break

        frame_idx += 1

    # If the video ended during a LOST streak – close it as well
    if current_lost_streak > 0:
        lost_streaks.append(current_lost_streak)

    end_time = time.time()
    total_time = end_time - start_time

    cap.release()
    out.release()
    cv2.destroyAllWindows()

    # --- Metrics computation ---
    if total_frames > 0 and total_time > 0:
        fps = total_frames / total_time
        ms_per_frame = (total_time / total_frames) * 1000.0
        lost_ratio = lost_frames / total_frames
    else:
        fps = 0.0
        ms_per_frame = 0.0
        lost_ratio = 0.0

    max_lost_streak = max(lost_streaks) if lost_streaks else 0
    avg_lost_streak = (sum(lost_streaks) / len(lost_streaks)) if lost_streaks else 0.0
    mean_area = (bbox_area_sum / bbox_area_count) if bbox_area_count > 0 else 0.0
    mean_center_motion = (center_motion_sum / center_motion_count) if center_motion_count > 0 else 0.0

    # Counters from DroneTracker (RE-DETECTION)
    redetect_calls = getattr(tracker, "redetect_calls", 0)
    redetect_success = getattr(tracker, "redetect_success", 0)
    redetect_success_rate = (redetect_success / redetect_calls) if redetect_calls > 0 else 0.0

    print("=== METRICS ===")
    print(f"Total frames      : {total_frames}")
    print(f"Total time [s]    : {total_time:.2f}")
    print(f"FPS               : {fps:.2f}")
    print(f"Lost frames       : {lost_frames} ({lost_ratio*100:.1f}%)")
    print(f"Redetect calls    : {redetect_calls}")
    print(f"Redetect success  : {redetect_success} ({redetect_success_rate*100:.1f}%)")
    print(f"Max LOST streak   : {max_lost_streak}")
    print(f"Avg LOST streak   : {avg_lost_streak:.1f}")
    print(f"Mean bbox area    : {mean_area:.1f}")
    print(f"Mean center motion: {mean_center_motion:.2f} px/frame")

if __name__ == "__main__":
    import numpy as np
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=str, required=True,
                        help="Path to input video (mp4)")
    parser.add_argument("--out", type=str, default="out.mp4",
                        help="Path to output annotated video")
    parser.add_argument("--debug", action="store_true",
                        help="Show debug window")
    parser.add_argument("--manual_init", action="store_true",
                        help="Select initial bbox manually")
    args = parser.parse_args()
    main(args)
