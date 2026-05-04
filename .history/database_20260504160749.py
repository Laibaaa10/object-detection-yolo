from pymongo import MongoClient, ASCENDING, DESCENDING
from datetime import datetime
from collections import defaultdict
import time


class DetectionDB:
    """
    Handles all MongoDB operations for SENTINEL.
    Stores detections, sessions, and analytics.
    """

    def __init__(
        self,
        uri="mongodb://localhost:27017",
        db_name="sentinel_db"
    ):
        try:
            self.client  = MongoClient(uri, serverSelectionTimeoutMS=3000)
            self.client.server_info()          # test connection
            self.db      = self.client[db_name]

            # ── Collections ───────────────────────────────────────
            self.detections = self.db["detections"]
            self.sessions   = self.db["sessions"]
            self.analytics  = self.db["analytics"]

            # ── Indexes for fast queries ──────────────────────────
            self.detections.create_index([("timestamp", DESCENDING)])
            self.detections.create_index([("class_name", ASCENDING)])
            self.detections.create_index([("session_id", ASCENDING)])
            self.sessions.create_index(  [("start_time", DESCENDING)])

            self.session_id  = None
            self.batch_buffer = []
            self.batch_size   = 20     # insert every 20 detections

            print("✅ MongoDB connected — sentinel_db ready")

        except Exception as e:
            print(f"❌ MongoDB connection failed: {e}")
            print("   Make sure MongoDB Community Server is running")
            self.client = None

    @property
    def connected(self):
        return self.client is not None

    # ── Session management ────────────────────────────────────────
    def start_session(self, source="webcam", model="yolov8n",
                      classes=None):
        """Call once when detection starts."""
        if not self.connected:
            return None

        session = {
            "start_time":   datetime.now(),
            "end_time":     None,
            "source":       source,
            "model":        model,
            "classes":      classes or ["all"],
            "total_detections": 0,
            "unique_classes":   [],
            "avg_fps":      0.0,
            "status":       "active"
        }
        result = self.sessions.insert_one(session)
        self.session_id = result.inserted_id
        print(f"📂 Session started: {self.session_id}")
        return self.session_id

    def end_session(self, avg_fps=0.0):
        """Call when detection stops."""
        if not self.connected or not self.session_id:
            return

        # Flush remaining buffer
        self._flush_buffer()

        # Get session stats
        total = self.detections.count_documents(
            {"session_id": self.session_id})
        classes = self.detections.distinct(
            "class_name", {"session_id": self.session_id})

        self.sessions.update_one(
            {"_id": self.session_id},
            {"$set": {
                "end_time":         datetime.now(),
                "total_detections": total,
                "unique_classes":   classes,
                "avg_fps":          round(avg_fps, 2),
                "status":           "completed"
            }}
        )
        print(f"📂 Session ended — {total} detections saved")

    # ── Detection logging ─────────────────────────────────────────
    def log_detection(self, class_name, confidence,
                      track_id, bbox, speed=0.0, alert=False):
        """
        Log a single detection.
        Uses batching for performance — flushes every 20 records.
        """
        if not self.connected:
            return

        doc = {
            "session_id":  self.session_id,
            "timestamp":   datetime.now(),
            "class_name":  class_name,
            "confidence":  round(float(confidence), 3),
            "track_id":    int(track_id),
            "bbox": {
                "x1": int(bbox[0]),
                "y1": int(bbox[1]),
                "x2": int(bbox[2]),
                "y2": int(bbox[3])
            },
            "speed_kmh":   round(float(speed), 2),
            "is_alert":    bool(alert)
        }

        self.batch_buffer.append(doc)

        if len(self.batch_buffer) >= self.batch_size:
            self._flush_buffer()

    def _flush_buffer(self):
        """Insert buffered detections into MongoDB."""
        if self.batch_buffer and self.connected:
            try:
                self.detections.insert_many(
                    self.batch_buffer, ordered=False)
                self.batch_buffer.clear()
            except Exception as e:
                print(f"DB write error: {e}")

    def log_crossing(self, track_id, class_name):
        """Log a line-crossing event."""
        if not self.connected:
            return
        self.analytics.insert_one({
            "session_id": self.session_id,
            "timestamp":  datetime.now(),
            "event":      "line_crossing",
            "track_id":   int(track_id),
            "class_name": class_name
        })

    # ── Query helpers ─────────────────────────────────────────────
    def get_recent_detections(self, limit=50):
        """Fetch most recent detections."""
        if not self.connected:
            return []
        return list(
            self.detections
            .find({}, {"_id": 0})
            .sort("timestamp", DESCENDING)
            .limit(limit)
        )

    def get_class_summary(self, session_id=None):
        """Get detection counts grouped by class."""
        if not self.connected:
            return {}
        match = {"session_id": session_id} if session_id else {}
        pipeline = [
            {"$match": match},
            {"$group": {
                "_id":   "$class_name",
                "count": {"$sum": 1},
                "avg_conf": {"$avg": "$confidence"},
                "avg_speed": {"$avg": "$speed_kmh"}
            }},
            {"$sort": {"count": -1}}
        ]
        results = self.analytics.database["detections"].aggregate(
            pipeline)
        return {r["_id"]: {
            "count":     r["count"],
            "avg_conf":  round(r["avg_conf"], 3),
            "avg_speed": round(r["avg_speed"], 2)
        } for r in results}

    def get_sessions(self, limit=10):
        """Fetch recent sessions."""
        if not self.connected:
            return []
        return list(
            self.sessions
            .find({}, {"_id": 0})
            .sort("start_time", DESCENDING)
            .limit(limit)
        )

    def get_alert_count(self, session_id=None):
        """Count alert detections."""
        if not self.connected:
            return 0
        query = {"is_alert": True}
        if session_id:
            query["session_id"] = session_id
        return self.detections.count_documents(query)

    def get_hourly_stats(self):
        """Detections grouped by hour — for charts."""
        if not self.connected:
            return []
        pipeline = [
            {"$group": {
                "_id": {
                    "hour":  {"$hour":  "$timestamp"},
                    "class": "$class_name"
                },
                "count": {"$sum": 1}
            }},
            {"$sort": {"_id.hour": 1}}
        ]
        return list(self.detections.aggregate(pipeline))

    def close(self):
        if self.connected:
            self._flush_buffer()
            self.client.close()
            print("MongoDB connection closed")