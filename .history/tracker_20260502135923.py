import cv2


class LineCounter:
    """Counts objects crossing a virtual line on the frame."""

    def __init__(self, start_point, end_point):
        self.start    = start_point   # (x1, y1)
        self.end      = end_point     # (x2, y2)
        self.count    = 0
        self.track_history = {}       # track_id → list of center points
        self.crossed  = set()         # IDs that already crossed

    def _get_center(self, box):
        """Return bottom-center of bounding box."""
        x1, y1, x2, y2 = map(int, box)
        return ((x1 + x2) // 2, y2)

    def _is_crossing(self, prev, curr):
        """
        Check if movement from prev → curr crosses the counter line.
        Uses 2D cross-product sign change to detect crossing.
        """
        x1, y1 = self.start
        x2, y2 = self.end

        def side(px, py):
            return (x2 - x1) * (py - y1) - (y2 - y1) * (px - x1)

        return side(*prev) * side(*curr) < 0

    def update(self, track_id, box):
        """Update tracker history and check for line crossing."""
        center = self._get_center(box)

        if track_id not in self.track_history:
            self.track_history[track_id] = []

        history = self.track_history[track_id]
        history.append(center)

        # Keep only last 5 positions
        if len(history) > 5:
            history.pop(0)

        # Check crossing only if we have 2+ points and not already counted
        if len(history) >= 2 and track_id not in self.crossed:
            if self._is_crossing(history[-2], history[-1]):
                self.count += 1
                self.crossed.add(track_id)

    def draw(self, frame):
        """Draw the counter line and count on the frame."""
        # Draw line
        cv2.line(frame, self.start, self.end, (0, 255, 255), 3)

        # Draw endpoints
        cv2.circle(frame, self.start, 6, (0, 200, 255), -1)
        cv2.circle(frame, self.end,   6, (0, 200, 255), -1)

        # Draw count label at line midpoint
        mid_x = (self.start[0] + self.end[0]) // 2
        mid_y = (self.start[1] + self.end[1]) // 2

        label = f"Count: {self.count}"
        (w, h), _ = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
        cv2.rectangle(frame,
                      (mid_x - 4, mid_y - h - 8),
                      (mid_x + w + 4, mid_y + 4),
                      (0, 0, 0), -1)
        cv2.putText(frame, label,
                    (mid_x, mid_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (0, 255, 255), 2)
        return frame