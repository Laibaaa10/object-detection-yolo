from collections import defaultdict, deque
import numpy as np


class SpeedEstimator:
    """
    Estimates real-world speed of tracked objects.

    pixel_per_meter : how many pixels = 1 metre in your scene.
                      Calibrate by measuring a known object in frame.
    smoothing       : number of recent frames to average speed over.
    """

    def __init__(self, pixel_per_meter=8.0, fps=30, smoothing=10):
        self.ppm         = pixel_per_meter
        self.fps         = fps
        self.smoothing   = smoothing

        self.positions   = defaultdict(lambda: deque(maxlen=2))
        self.speed_hist  = defaultdict(lambda: deque(maxlen=smoothing))
        self.speeds      = {}          # track_id → current speed (km/h)

    def update(self, track_id, box):
        """
        Call once per frame per tracked object.
        box = [x1, y1, x2, y2]
        """
        x1, y1, x2, y2 = map(int, box)
        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2
        center = (cx, cy)

        self.positions[track_id].append(center)

        if len(self.positions[track_id]) == 2:
            p1, p2    = self.positions[track_id]
            px_dist   = np.hypot(p2[0]-p1[0], p2[1]-p1[1])
            metres    = px_dist / self.ppm
            mps       = metres * self.fps        # metres per second
            kmh       = mps * 3.6               # km/h

            self.speed_hist[track_id].append(kmh)
            self.speeds[track_id] = np.mean(self.speed_hist[track_id])

        return self.speeds.get(track_id, 0.0)

    def get_speed(self, track_id):
        return self.speeds.get(track_id, 0.0)

    def get_color(self, speed_kmh):
        """
        Returns BGR color based on speed:
        green = slow, yellow = medium, red = fast
        """
        if speed_kmh < 20:
            return (50, 200, 50)     # green
        elif speed_kmh < 60:
            return (0, 200, 255)     # amber/yellow
        else:
            return (0, 60, 220)      # red