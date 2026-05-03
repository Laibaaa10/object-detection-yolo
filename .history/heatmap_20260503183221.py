import cv2
import numpy as np


class Heatmap:
    """
    Accumulates object positions over time and renders
    a colour heatmap overlay on the video frame.
    """

    def __init__(self, frame_width=1280, frame_height=720,
                 decay=0.98, intensity=200):
        self.w        = frame_width
        self.h        = frame_height
        self.decay    = decay       # 0.95-0.99: how fast heat fades
        self.intensity = intensity  # how much each detection adds

        # Heat accumulation map (float32 for precision)
        self.heat_map = np.zeros((self.h, self.w), dtype=np.float32)

    def update(self, boxes):
        """
        Add heat at each detected object's center.
        boxes = list of [x1, y1, x2, y2]
        """
        # Decay existing heat slightly each frame
        self.heat_map *= self.decay

        for box in boxes:
            x1, y1, x2, y2 = map(int, box)
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2

            # Clamp to frame bounds
            cx = max(0, min(cx, self.w - 1))
            cy = max(0, min(cy, self.h - 1))

            # Add a gaussian blob at object center
            radius = max((x2 - x1), (y2 - y1)) // 2
            radius = max(radius, 30)

            # Draw filled circle on heat map
            cv2.circle(
                self.heat_map,
                (cx, cy),
                radius,
                self.intensity,
                -1
            )

        # Smooth the heatmap for nicer look
        self.heat_map = cv2.GaussianBlur(
            self.heat_map, (51, 51), 0)

        # Cap values
        self.heat_map = np.clip(self.heat_map, 0, 255)

    def draw(self, frame, alpha=0.45):
        """
        Overlay the heatmap on the frame.
        alpha = transparency (0.0 fully transparent, 1.0 opaque)
        """
        if self.heat_map.max() < 1:
            return frame   # nothing to draw yet

        # Normalize to 0-255
        norm = cv2.normalize(
            self.heat_map, None,
            0, 255,
            cv2.NORM_MINMAX
        ).astype(np.uint8)

        # Apply colormap: JET = blue→green→yellow→red
        colored = cv2.applyColorMap(norm, cv2.COLORMAP_JET)

        # Mask out near-zero areas (keep background clean)
        mask = norm > 15
        mask_3ch = np.stack([mask, mask, mask], axis=2)

        # Blend only where heat exists
        blended = frame.copy()
        blended[mask_3ch] = cv2.addWeighted(
            frame, 1 - alpha,
            colored, alpha, 0
        )[mask_3ch]

        return blended

    def draw_legend(self, frame):
        """Draw a colour scale legend on the frame."""
        h, w = frame.shape[:2]

        # Legend bar position (bottom right)
        bar_x  = w - 160
        bar_y  = h - 30
        bar_w  = 120
        bar_h  = 12

        # Draw gradient bar
        for i in range(bar_w):
            val   = int(i * 255 / bar_w)
            color = cv2.applyColorMap(
                np.array([[val]], dtype=np.uint8),
                cv2.COLORMAP_JET
            )[0][0].tolist()
            cv2.line(frame,
                (bar_x + i, bar_y),
                (bar_x + i, bar_y + bar_h),
                color, 1)

        # Border around bar
        cv2.rectangle(frame,
            (bar_x, bar_y),
            (bar_x + bar_w, bar_y + bar_h),
            (200, 200, 200), 1)

        # Labels
        cv2.putText(frame, "cool",
            (bar_x, bar_y - 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42, (150, 150, 255), 1)
        cv2.putText(frame, "hot",
            (bar_x + bar_w - 22, bar_y - 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42, (0, 60, 255), 1)
        cv2.putText(frame, "HEATMAP",
            (bar_x, bar_y + bar_h + 14),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4, (180, 180, 180), 1)

        return frame

    def reset(self):
        """Clear the heatmap."""
        self.heat_map = np.zeros(
            (self.h, self.w), dtype=np.float32)