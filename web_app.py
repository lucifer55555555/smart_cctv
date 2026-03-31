from typing import List, Dict

import os
import csv
from collections import Counter
from datetime import datetime
import threading
import time

import cv2
from flask import Flask, render_template, Response, request

from config import CONFIG
from app.pipeline import CampusSafetyPipeline

app = Flask(__name__)

# Single shared pipeline instance for the campus camera
pipeline = CampusSafetyPipeline()

# Global variables to safely share the current frame with web clients
latest_frame = None
latest_frame_lock = threading.Lock()

def pipeline_thread():
    """
    Runs the campus safety pipeline continuously in the background,
    recording incidents regardless of whether a web client is viewing.
    """
    global latest_frame
    print("[pipeline_thread] Starting background processing...")
    
    while True:
        # Run the generator. It will yield (False, None, None) if stopped.
        try:
            for ok, frame, risk in pipeline.frames():
                if not ok or frame is None:
                    # Not running or camera error; just wait and retry.
                    time.sleep(0.5)
                    with latest_frame_lock:
                        latest_frame = None
                    continue
                    
                with latest_frame_lock:
                    latest_frame = frame.copy()
        except Exception as e:
            print(f"[pipeline_thread] ERROR: {e}")
            time.sleep(1.0)



# Start the background thread immediately
thread = threading.Thread(target=pipeline_thread, daemon=True)
thread.start()


def gen_frames():
    """
    MJPEG frame generator using the latest frame from the background thread.
    """
    global latest_frame
    while True:
        with latest_frame_lock:
            frame_to_yield = latest_frame
            
        if frame_to_yield is None:
            time.sleep(0.1)
            continue

        ret, buffer = cv2.imencode(".jpg", frame_to_yield)
        if not ret:
            time.sleep(0.1)
            continue

        frame_bytes = buffer.tobytes()
        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n"
        )
        
        # Smooth streaming rate (approx 100 fps max, throttled by camera)
        time.sleep(0.01)



@app.route("/")
def index():
    """
    Home page: shows live campus stream and link to incidents.
    """
    location_name = CONFIG["camera"]["location_name"]
    return render_template("index.html", location_name=location_name)


@app.route("/video_feed")
def video_feed():
    """
    Video streaming route. Put this in the src of an img tag.
    """
    return Response(
        gen_frames(),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )


@app.route("/incidents")
def incidents():
    """
    Simple page listing recent campus incidents from the CSV log.
    """
    incidents = _load_incidents()

    location_name = CONFIG["camera"]["location_name"]
    return render_template(
        "incidents.html",
        location_name=location_name,
        incidents=incidents,
    )


def _load_incidents() -> List[Dict[str, str]]:
    """
    Helper to load all incidents from CSV into a list of dicts.
    """
    log_csv = CONFIG["incidents"]["log_csv"]
    incidents: List[Dict[str, str]] = []

    if os.path.exists(log_csv):
        with open(log_csv, mode="r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                incidents.append(row)

    return incidents


@app.route("/dashboard")
def dashboard():
    """
    Analytical dashboard for campus incidents: summary cards + simple charts.
    """
    incidents = _load_incidents()
    location_name = CONFIG["camera"]["location_name"]

    total_incidents = len(incidents)

    # Count by risk level
    risk_counts = Counter(row.get("risk_level", "UNKNOWN") for row in incidents)

    # Count by event type
    event_counts = Counter(row.get("event_type", "UNKNOWN") for row in incidents)

    # Count incidents per day (YYYY-MM-DD) for a simple trend chart
    day_counts: Counter[str] = Counter()
    for row in incidents:
        ts = row.get("timestamp", "")
        try:
            dt = datetime.strptime(ts, "%Y-%m-%d %H:%M:%S")
            day_key = dt.strftime("%Y-%m-%d")
            day_counts[day_key] += 1
        except Exception:
            continue

    # Prepare data for charts (use a stable, meaningful order)
    preferred_risk_order = ["CRITICAL", "HIGH_RISK", "MEDIUM_RISK", "NO_RISK", "UNKNOWN"]
    risk_labels = [k for k in preferred_risk_order if k in risk_counts] + [
        k for k in risk_counts.keys() if k not in preferred_risk_order
    ]
    risk_values = [risk_counts[label] for label in risk_labels]

    day_labels = sorted(day_counts.keys())
    day_values = [day_counts[d] for d in day_labels]

    # Latest few incidents to show in a small table
    recent_incidents = list(reversed(incidents))[:10]

    last_incident_time = incidents[-1].get("timestamp") if incidents else None
    critical_count = risk_counts.get("CRITICAL", 0)
    critical_pct = (critical_count / total_incidents * 100.0) if total_incidents else 0.0

    top_event_types = event_counts.most_common(5)
    event_labels = [k for k, _ in top_event_types]
    event_values = [v for _, v in top_event_types]

    return render_template(
        "dashboard.html",
        location_name=location_name,
        total_incidents=total_incidents,
        last_incident_time=last_incident_time,
        critical_count=critical_count,
        critical_pct=critical_pct,
        risk_labels=risk_labels,
        risk_values=risk_values,
        day_labels=day_labels,
        day_values=day_values,
        event_labels=event_labels,
        event_values=event_values,
        recent_incidents=recent_incidents,
    )


def _shutdown_server() -> None:
    """
    Ask the Werkzeug development server to shut down.
    """
    func = request.environ.get("werkzeug.server.shutdown")
    if func is None:
        # Not running with Werkzeug (e.g., production WSGI); just return.
        return
    func()


@app.route("/start_camera", methods=["POST"])
def start_camera():
    """Start the campus camera monitoring."""
    pipeline.start()
    return {"status": "started"}, 200


@app.route("/stop_camera", methods=["POST"])
def stop_camera():
    """Stop/Pause the campus camera monitoring."""
    pipeline.stop()
    return {"status": "stopped"}, 200


@app.route("/camera_status", methods=["GET"])
def camera_status():
    """Check if the camera is currently monitoring."""
    return {"running": pipeline.is_running()}, 200


@app.route("/get_latest_alert", methods=["GET"])
def get_latest_alert():
    """
    Returns the most recent incident to the frontend for real-time notifications.
    Clears it after reading to ensure a single notification per alert.
    """
    alert = pipeline.latest_incident
    # Optional: only return if it's "fresh" (last 10 seconds)
    if alert and (time.time() - alert["timestamp"] < 10.0):
        # Clear it so it won't be alerted again on next poll
        pipeline.latest_incident = None
        return alert, 200
    
    return {}, 200


@app.route("/test_alert", methods=["POST"])
def test_alert():
    """Trigger a dummy incident for notification testing."""
    pipeline.latest_incident = {
        "timestamp": time.time(),
        "level": "CRITICAL",
        "event_type": "test_alert_simulation",
        "details": "This is a simulated smart alert for testing notifications."
    }
    return {"status": "triggered"}, 200


@app.route("/shutdown", methods=["POST"])
def shutdown():
    """
    Endpoint triggered from the dashboard to stop monitoring and shut down the app.
    """
    # Release camera and other resources
    pipeline.release()
    _shutdown_server()
    return "Campus safety server is stopping. You can close this tab.", 200



if __name__ == "__main__":
    # For development only. In production, use a proper WSGI server.
    app.run(host="0.0.0.0", port=5000, debug=True, use_reloader=False)

