import cv2
import numpy as np

class DroneTracker:
    def __init__(self, max_missed=10, reinit_threshold=0.6):
        # CSRT tracker
        self.tracker = cv2.TrackerCSRT_create()
        self.initialized = False

        # Template updated during runtime
        self.template = None
        self.template_size = None

        # Reference template from the initial initialization (does not change)
        self.ref_template = None
        # Kept for information only, no longer critical for size
        self.ref_template_size = None  

        # After N consecutive failures -> global search
        self.max_missed = max_missed
        self.missed_count = 0
        self.reinit_threshold = reinit_threshold

        self.track_id = 1  # Only one drone
        # Used for local re-detection
        self.last_bbox = None  
        
        # Counters for RE-DETECTION statistics
        self.redetect_calls = 0
        self.redetect_success = 0

    def init(self, frame, bbox):
        """bbox: (x, y, w, h)"""
        x, y, w, h = map(int, bbox)

        # CSRT initialization
        self.tracker = cv2.TrackerCSRT_create()
        ok = self.tracker.init(frame, (x, y, w, h))

        # Explicit failure
        if ok is False:
            return False

        # Success (or None in some newer OpenCV versions)
        roi = frame[y:y+h, x:x+w].copy()
        if roi.size == 0:
            return False

        self.initialized = True
        self.template = roi
        self.template_size = (w, h)

        # Reference template – fixed, used only for RE-DETECTION
        self.ref_template = roi.copy()
        # Kept for information only, not used directly for size
        self.ref_template_size = (w, h)  

        self.missed_count = 0
        self.last_bbox = (x, y, w, h)

        return True

    def update(self, frame):
        """
        Returns:
          success (bool),
          bbox (x, y, w, h) or None
        """
        if not self.initialized:
            return False, None

        ok, bbox = self.tracker.update(frame)

        if ok:
            x, y, w, h = map(int, bbox)
            self.last_bbox = (x, y, w, h)

            # --- Check tracking quality against the REFERENCE TEMPLATE ---
            if self.ref_template is not None:
                roi = frame[y:y+h, x:x+w]
                if roi.size == 0:
                    ok = False
                else:
                    # Actual template size from the image itself (no width/height confusion)
                    tmpl_h, tmpl_w = self.ref_template.shape[:2]
                    roi_resized = cv2.resize(roi, (tmpl_w, tmpl_h))
                    frame_gray = cv2.cvtColor(roi_resized, cv2.COLOR_BGR2GRAY)
                    tmpl_gray = cv2.cvtColor(self.ref_template, cv2.COLOR_BGR2GRAY)

                    res = cv2.matchTemplate(frame_gray, tmpl_gray, cv2.TM_CCOEFF_NORMED)
                    _, max_val, _, _ = cv2.minMaxLoc(res)

                    if max_val < self.reinit_threshold:
                        ok = False

            # If after the quality check ok is still True → normal update
            if ok:
                self.missed_count = 0

                # Update TEMPLATE only if the box is still large enough
                min_side = min(w, h)
                if min_side >= 20:
                    roi = frame[y:y+h, x:x+w]
                    if roi.size > 0:
                        self.template = roi.copy()
                        self.template_size = (w, h)

                return True, (x, y, w, h)

        # --- If we reach here → tracking failure (ok=False or low quality) ---
        self.missed_count += 1

        if self.ref_template is None:
            return False, None

        # Local RE-DETECTION as long as we have not exceeded the threshold
        if self.missed_count < self.max_missed:
            re_ok, re_bbox = self.redetect(frame, local=True)
        else:
            # After N failed frames – global search
            re_ok, re_bbox = self.redetect(frame, local=False)

        if re_ok:
            self.init(frame, re_bbox)
            return True, re_bbox

        return False, None

    def redetect(self, frame, local=True):
        """
        Template matching using the REFERENCE TEMPLATE.
        local=True  -> search around the last known location
        local=False -> global search over the entire frame
        Returns (success, bbox).
        """
        if self.ref_template is None:
            return False, None
            
        # Count how many times we tried RE-DETECTION
        self.redetect_calls += 1
        
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        template_gray = cv2.cvtColor(self.ref_template, cv2.COLOR_BGR2GRAY)

        # Actual template size from the image (height, width)
        tmpl_h, tmpl_w = template_gray.shape[:2]

        if local and self.last_bbox is not None:
            # Local search around the last known bbox
            lx, ly, lw, lh = map(int, self.last_bbox)
            pad = 80  # How far around to search; can be tuned
            H, W = frame_gray.shape

            x0 = max(0, lx - pad)
            y0 = max(0, ly - pad)
            x1 = min(W, lx + lw + pad)
            y1 = min(H, ly + lh + pad)

            search_region = frame_gray[y0:y1, x0:x1]

            # If the search region is smaller than the template – nothing to search
            if search_region.shape[0] < tmpl_h or search_region.shape[1] < tmpl_w:
                return False, None

            res = cv2.matchTemplate(search_region, template_gray, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, max_loc = cv2.minMaxLoc(res)

            if max_val < self.reinit_threshold:
                return False, None

            px, py = max_loc
            x = x0 + px
            y = y0 + py
            
            self.redetect_success += 1
            bbox = (x, y, tmpl_w, tmpl_h)  # width, height
            return True, bbox

        else:
            # Global search over the entire frame
            if frame_gray.shape[0] < tmpl_h or frame_gray.shape[1] < tmpl_w:
                return False, None

            res = cv2.matchTemplate(frame_gray, template_gray, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, max_loc = cv2.minMaxLoc(res)

            if max_val < self.reinit_threshold:
                return False, None

            x, y = max_loc
            
            self.redetect_success += 1
            bbox = (x, y, tmpl_w, tmpl_h)  # width, height
            return True, bbox
