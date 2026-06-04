"""
Flask web application for the AMR Warehouse Robot Dashboard.

This module is imported by dashboard_node.py which sets `ros_bridge`
before starting the Flask/SocketIO server.
"""

import base64
import io
import json
import os
import threading
import time
from typing import Optional

import numpy as np
from flask import Flask, Response, jsonify, render_template, request
from flask_socketio import SocketIO

# ---------------------------------------------------------------------------
# App configuration
# ---------------------------------------------------------------------------

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_TEMPLATE_DIR = os.path.join(_THIS_DIR, "web", "templates")
_STATIC_DIR = os.path.join(_THIS_DIR, "web", "static")

app = Flask(
    __name__,
    template_folder=_TEMPLATE_DIR,
    static_folder=_STATIC_DIR,
)
app.config["SECRET_KEY"] = os.environ.get("DASHBOARD_SECRET", "amr-dashboard-secret")

socketio = SocketIO(app, cors_allowed_origins="*", async_mode="threading")

# Set by dashboard_node before the server starts
ros_bridge = None  # type: Optional[object]

# ---------------------------------------------------------------------------
# HMM voice recognizer (lazy-loaded on first /api/voice call)
# ---------------------------------------------------------------------------

_hmm_recognizer = None
_hmm_lock = threading.Lock()


def _get_hmm_recognizer():
    global _hmm_recognizer
    if _hmm_recognizer is not None:
        return _hmm_recognizer
    with _hmm_lock:
        if _hmm_recognizer is not None:
            return _hmm_recognizer
        try:
            from voice_control.hmm_recognizer import HMMRecognizer  # type: ignore[import]
            r = HMMRecognizer()
            if r.load():
                _hmm_recognizer = r
        except Exception as exc:  # noqa: BLE001
            print(f'[dashboard] HMM recognizer load failed: {exc}')
    return _hmm_recognizer

# ---------------------------------------------------------------------------
# Mission validation — types accepted by mission_control's parser.
# ---------------------------------------------------------------------------

VALID_MISSION_TYPES = {
    "ROLLER_TO_TRUCK", "RACK_TO_TRUCK", "CUSTOM", "PICK_ONLY",
    "SEARCH_ROLLERS", "SEARCH_RACKS",
}

# ---------------------------------------------------------------------------
# Voice-command → robot action map.
#
# The recognizer publishes a single lowercased word on /voice_command (still
# done below for the SM's pause/stop handling + the voice history). On top of
# that, these words drive the robot directly from the dashboard:
#   forward/left/right → a short timed teleop nudge (only while the SM is IDLE)
#   racks/rollers      → a SEARCH-only mission (drive the zone, stop at the pallet)
# map/one/two/continue/stop are intentionally NOT mapped yet (passthrough only).
# ---------------------------------------------------------------------------

_VOICE_TELEOP_LINEAR = 0.10   # m/s  (matches the teleop default speed)
_VOICE_TELEOP_ANGULAR = 0.21  # rad/s (matches the teleop default turn)
_VOICE_TELEOP_DURATION = 2.0  # s

VOICE_ACTIONS = {
    "forward": {"kind": "teleop", "linear":  _VOICE_TELEOP_LINEAR, "angular": 0.0},
    "left":    {"kind": "teleop", "linear": 0.0, "angular":  _VOICE_TELEOP_ANGULAR},
    "right":   {"kind": "teleop", "linear": 0.0, "angular": -_VOICE_TELEOP_ANGULAR},
    "racks":   {"kind": "mission", "type": "SEARCH_RACKS"},
    "rollers": {"kind": "mission", "type": "SEARCH_ROLLERS"},
}

# ---------------------------------------------------------------------------
# Teleop watchdog state
# ---------------------------------------------------------------------------

_last_teleop_time: float = 0.0
_teleop_active: bool = False
_teleop_lock = threading.Lock()


def _teleop_watchdog() -> None:
    """Stop the robot if no teleop command received for 400ms."""
    global _teleop_active
    while True:
        time.sleep(0.05)
        with _teleop_lock:
            active = _teleop_active
            last = _last_teleop_time
        if active and ros_bridge is not None:
            if time.monotonic() - last > 0.35:
                try:
                    ros_bridge.publish_cmd_vel(0.0, 0.0)
                except Exception:  # noqa: BLE001
                    pass
                with _teleop_lock:
                    _teleop_active = False


# Start watchdog thread eagerly (daemon, won't block shutdown)
_watchdog_thread = threading.Thread(target=_teleop_watchdog, daemon=True)
_watchdog_thread.start()


def _drive_timed(linear: float, angular: float, duration: float) -> None:
    """Drive at (linear, angular) for `duration` seconds, then stop.

    The velocity smoother (twist_relay) holds the last command forever, so a
    single publish would never stop on its own. We re-publish at 20 Hz and keep
    feeding the teleop watchdog (_last_teleop_time / _teleop_active) so it acts
    as the safety net: if this thread dies mid-move, the watchdog stops the
    robot 0.35 s later. At the end we send an explicit 0,0.
    """
    def _run() -> None:
        global _last_teleop_time, _teleop_active
        end = time.monotonic() + duration
        while time.monotonic() < end:
            if ros_bridge is None:
                break
            try:
                ros_bridge.publish_cmd_vel(linear, angular)
            except Exception:  # noqa: BLE001
                break
            with _teleop_lock:
                _last_teleop_time = time.monotonic()
                _teleop_active = True
            time.sleep(0.05)
        if ros_bridge is not None:
            try:
                ros_bridge.publish_cmd_vel(0.0, 0.0)
            except Exception:  # noqa: BLE001
                pass
        with _teleop_lock:
            _teleop_active = False

    threading.Thread(target=_run, daemon=True).start()


def _dispatch_voice_action(word: str) -> Optional[dict]:
    """Run the robot action bound to a recognized voice word, if any.

    Returns a small dict describing what happened (for the UI), or None when
    the word has no bound action (map/one/two/continue/stop → passthrough).
    """
    action = VOICE_ACTIONS.get(word)
    if action is None or ros_bridge is None:
        return None

    if action["kind"] == "teleop":
        # Only nudge the robot when it isn't running an autonomous mission,
        # otherwise voice teleop and the SM fight over /cmd_vel_in.
        try:
            sm_state = (ros_bridge.get_state().state or "").upper()
        except Exception:  # noqa: BLE001
            sm_state = ""
        if sm_state not in ("", "IDLE", "UNKNOWN"):
            return {"kind": "ignored", "reason": f"busy: {sm_state}"}
        _drive_timed(action["linear"], action["angular"], _VOICE_TELEOP_DURATION)
        return {"kind": "teleop", "duration": _VOICE_TELEOP_DURATION}

    if action["kind"] == "mission":
        # SEARCH-only mission; the SM ignores it if one is already running.
        try:
            ros_bridge.publish_mission(json.dumps({"type": action["type"]}))
        except Exception:  # noqa: BLE001
            return None
        return {"kind": "mission", "type": action["type"]}

    return None

# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/state")
def get_state():
    """Return current robot state as JSON."""
    if ros_bridge is None:
        return jsonify({"error": "ROS bridge not initialised"}), 503

    state = ros_bridge.get_state()
    return jsonify(
        {
            "pose": state.pose,
            "state": state.state,
            "nav_status": state.nav_status,
            "mission": state.mission,
            "velocity": state.velocity,
            "lifter_level": state.lifter_level,
            "system_mode": state.system_mode,
        }
    )


# ---------------------------------------------------------------------------
# mission_control debugger surface
# ---------------------------------------------------------------------------

# Outcomes per state — used by the front-end to populate the Force-outcome
# dropdown and by the server to reject malformed requests.
_SM_OUTCOMES = {
    "IDLE":           ["mission_received"],
    "SEARCH":         ["found", "done", "not_found", "stop"],
    "PICK":           ["picked", "done", "failed", "stop"],
    "NAV_TO_TRUCK":   ["arrived", "failed", "stop"],
    "PLACE":          ["placed", "failed", "stop"],
    "MISSION_DONE":   ["ok"],
    "MISSION_FAILED": ["ok"],
}

_SM_VALID_ACTIONS = {
    "pause", "resume", "step", "abort", "clear_abort", "set_step_mode", "force_outcome",
}


@app.route("/api/sm/outcomes")
def sm_outcomes():
    """Return the per-state outcome map for the Force-outcome dropdown."""
    return jsonify(_SM_OUTCOMES)


@app.route("/api/sm/snapshot")
def sm_snapshot():
    """Return the latest /sm/blackboard payload + recent transitions."""
    if ros_bridge is None:
        return jsonify({"error": "ROS bridge not initialised"}), 503
    return jsonify({
        "snapshot":    ros_bridge.get_sm_snapshot(),
        "transitions": ros_bridge.get_sm_transitions(),
    })


@app.route("/api/sm/control", methods=["POST"])
def sm_control():
    """Forward a debugger command to /sm/control."""
    if ros_bridge is None:
        return jsonify({"error": "ROS bridge not initialised"}), 503
    body = request.get_json(silent=True) or {}
    action = body.get("action")
    if action not in _SM_VALID_ACTIONS:
        return jsonify({"error": f"invalid action {action!r}"}), 400
    payload = {"action": action}
    if action == "set_step_mode":
        payload["value"] = bool(body.get("value", False))
    elif action == "force_outcome":
        outcome = body.get("value")
        if not isinstance(outcome, str) or not outcome:
            return jsonify({"error": "force_outcome requires non-empty 'value'"}), 400
        payload["value"] = outcome
    ros_bridge.publish_sm_control(payload)
    return jsonify({"ok": True, "sent": payload})


@app.route("/api/mode", methods=["GET"])
def get_mode():
    """Return the current system mode."""
    if ros_bridge is None:
        return jsonify({"error": "ROS bridge not initialised"}), 503
    return jsonify({"mode": ros_bridge.get_system_mode()})


@app.route("/api/mode", methods=["POST"])
def set_mode():
    """Switch system mode.

    Body:
      {"mode": "MAPPING" | "NAVIGATION",
       "reset_map": false  # only honoured when mode=MAPPING}

    Behaviour:
      - MAPPING → NAVIGATION: auto-saves the map (calls /map_saver/save_map)
        before publishing the new mode.
      - NAVIGATION → MAPPING: if reset_map=true, clears the SLAM grid first.
    """
    if ros_bridge is None:
        return jsonify({"error": "ROS bridge not initialised"}), 503

    body = request.get_json(silent=True) or {}
    target = str(body.get("mode", "")).strip().upper()
    if target not in ("MAPPING", "NAVIGATION"):
        return jsonify({"error": "mode must be MAPPING or NAVIGATION"}), 400

    current = ros_bridge.get_system_mode()
    notes: list[str] = []

    if current == "MAPPING" and target == "NAVIGATION":
        ok, msg = ros_bridge.call_save_map()
        notes.append(f"save_map: {'ok' if ok else 'failed'} ({msg})")
        # We continue regardless — the user can manually save later if the
        # service was missing.

    if current == "NAVIGATION" and target == "MAPPING":
        if bool(body.get("reset_map", False)):
            ok, msg = ros_bridge.call_reset_map()
            notes.append(f"reset_map: {'ok' if ok else 'failed'} ({msg})")

    ros_bridge.publish_system_mode(target)
    return jsonify({"mode": target, "previous": current, "notes": notes})


@app.route("/api/map/save", methods=["POST"])
def save_map():
    """Trigger the map_saver service to persist the current SLAM map."""
    if ros_bridge is None:
        return jsonify({"error": "ROS bridge not initialised"}), 503
    ok, msg = ros_bridge.call_save_map()
    code = 200 if ok else 500
    return jsonify({"success": ok, "message": msg}), code


@app.route("/api/map/reset", methods=["POST"])
def reset_map():
    """Clear the SLAM grid (start mapping from scratch)."""
    if ros_bridge is None:
        return jsonify({"error": "ROS bridge not initialised"}), 503
    ok, msg = ros_bridge.call_reset_map()
    code = 200 if ok else 500
    return jsonify({"success": ok, "message": msg}), code


@app.route("/api/mission", methods=["POST"])
def send_mission():
    """Accept a mission JSON and forward it to /mission.

    The full schema (per type) is validated by mission_control's parser; this
    endpoint only enforces the type tag so obviously-wrong payloads don't
    reach the topic. See src/mission_control/mission_control/mission_parser.py.
    """
    if ros_bridge is None:
        return jsonify({"error": "ROS bridge not initialised"}), 503

    body = request.get_json(silent=True)
    if not isinstance(body, dict):
        return jsonify({"error": "Invalid JSON body"}), 400

    mtype = body.get("type")
    if mtype not in VALID_MISSION_TYPES:
        return jsonify({
            "error": f"type must be one of {sorted(VALID_MISSION_TYPES)}",
        }), 400

    body.setdefault("id", f"ui_{int(time.time())}")
    ros_bridge.publish_mission(json.dumps(body))
    return jsonify({"status": "ok", "mission": body})


@app.route("/api/sm/zones")
def sm_zones():
    """Return mission_control's zones.yaml so the UI can populate dropdowns."""
    import yaml
    from ament_index_python.packages import (
        get_package_share_directory, PackageNotFoundError,
    )
    try:
        zones_path = os.path.join(
            get_package_share_directory("mission_control"),
            "config",
            "zones.yaml",
        )
        with open(zones_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        return jsonify(data)
    except PackageNotFoundError:
        return jsonify({"error": "mission_control package not installed"}), 503
    except FileNotFoundError:
        return jsonify({"error": "zones.yaml not found"}), 503
    except Exception as exc:  # noqa: BLE001
        return jsonify({"error": str(exc)}), 500


@app.route("/api/map")
def get_map():
    """Return the latest occupancy grid map as a PNG image."""
    if ros_bridge is None:
        return jsonify({"error": "ROS bridge not initialised"}), 503

    png = ros_bridge.get_latest_map()
    if png is None:
        return jsonify({"error": "No map available yet"}), 404

    return Response(png, mimetype="image/png")


@app.route("/api/map/info")
def get_map_info():
    """Return map metadata: origin (m), resolution (m/px), size (px), source."""
    if ros_bridge is None:
        return jsonify({"error": "ROS bridge not initialised"}), 503

    info = ros_bridge.get_map_info()
    if not info:
        return jsonify({"error": "No map metadata available yet"}), 404

    return jsonify(info)


@app.route("/video_feed")
def video_feed():
    """MJPEG stream from the camera."""
    return Response(
        _mjpeg_generator(),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )


@app.route("/api/waypoints", methods=["GET"])
def get_waypoints():
    """Return waypoints with full position data {name: {x, y, theta}}."""
    if ros_bridge is None:
        return jsonify({"waypoints": {}})
    wps = ros_bridge.get_waypoints()
    result = {}
    for name, data in wps.items():
        if isinstance(data, dict):
            result[name] = {
                "x": float(data.get("x", 0.0)),
                "y": float(data.get("y", 0.0)),
                "theta": float(data.get("theta", 0.0)),
            }
    return jsonify({"waypoints": result})


VALID_WAYPOINT_TYPES = {"roller", "rack", "truck"}


@app.route("/api/waypoints", methods=["POST"])
def add_waypoint():
    """Add a waypoint and persist to YAML.

    Two request shapes are accepted:
      - {"type": "rack"|"truck"|"roller", x, y, theta}  → the server assigns
        the next free name for that type (rack_1, truck_2, …) and returns it.
        This is what the map modal sends.
      - {"name": "...", x, y, theta}                     → legacy explicit name.
    """
    if ros_bridge is None:
        return jsonify({"error": "ROS bridge not initialised"}), 503
    body = request.get_json(silent=True) or {}
    try:
        x = float(body.get("x", 0.0))
        y = float(body.get("y", 0.0))
        theta = float(body.get("theta", 0.0))
    except (TypeError, ValueError):
        return jsonify({"error": "x, y, theta must be numbers"}), 400

    wtype = (body.get("type") or "").strip().lower()
    if wtype:
        if wtype not in VALID_WAYPOINT_TYPES:
            return jsonify({
                "error": f"type must be one of {sorted(VALID_WAYPOINT_TYPES)}",
            }), 400
        name = ros_bridge.add_typed_waypoint(wtype, x, y, theta)
        return jsonify({"ok": True, "name": name})

    name = (body.get("name") or "").strip()
    if not name:
        return jsonify({"error": "type or name is required"}), 400
    ros_bridge.add_waypoint(name, x, y, theta)
    return jsonify({"ok": True, "name": name})


@app.route("/api/waypoints/<name>", methods=["DELETE"])
def delete_waypoint(name: str):
    """Remove a waypoint by name."""
    if ros_bridge is None:
        return jsonify({"error": "ROS bridge not initialised"}), 503
    removed = ros_bridge.delete_waypoint(name)
    if not removed:
        return jsonify({"error": f"waypoint {name} not found"}), 404
    return jsonify({"ok": True})


@app.route("/api/goal", methods=["POST"])
def send_goal():
    """
    Send a navigation goal waypoint.

    Body: {"waypoint": "truck_1"} or {"waypoint": ""} to cancel.
    """
    if ros_bridge is None:
        return jsonify({"error": "ROS bridge not initialised"}), 503

    body = request.get_json(silent=True) or {}
    waypoint = body.get("waypoint", "")

    ros_bridge.publish_goal(waypoint)
    return jsonify({"ok": True})


@app.route("/api/lifter", methods=["POST"])
def set_lifter():
    """Set the lifter target level. Body: {"level": 0..3}.

    The robot's SPI HAL (Tang Nano) only reaches levels 0-3, so the UI and this
    endpoint cap there even though the message field allows 0-7.
    """
    if ros_bridge is None:
        return jsonify({"error": "ROS bridge not initialised"}), 503
    body = request.get_json(silent=True) or {}
    try:
        level = int(body.get("level"))
    except (TypeError, ValueError):
        return jsonify({"error": "level must be an integer 0-3"}), 400
    if not (0 <= level <= 3):
        return jsonify({"error": "level must be 0-3"}), 400
    ros_bridge.publish_lifter_level(level)
    return jsonify({"ok": True, "level": level})


@app.route("/api/teleop", methods=["POST"])
def send_teleop():
    """
    Send a teleop velocity command.

    Body: {"linear": 0.15, "angular": 0.0}
    """
    global _last_teleop_time, _teleop_active

    if ros_bridge is None:
        return jsonify({"error": "ROS bridge not initialised"}), 503

    body = request.get_json(silent=True) or {}
    try:
        linear = float(body.get("linear", 0.0))
        angular = float(body.get("angular", 0.0))
    except (TypeError, ValueError):
        return jsonify({"error": "linear and angular must be numbers"}), 400

    ros_bridge.publish_cmd_vel(linear, angular)

    with _teleop_lock:
        _last_teleop_time = time.monotonic()
        _teleop_active = True

    return jsonify({"ok": True})


@app.route("/api/voice", methods=["POST"])
def voice_command():
    """
    Receive an audio blob from the browser, run HMM recognition,
    publish the result to /voice_command, and return the recognised word.

    The browser should POST multipart/form-data with field 'audio'
    containing a WebM/OGG audio blob, or raw WAV bytes with
    Content-Type: audio/wav.
    """
    recognizer = _get_hmm_recognizer()
    if recognizer is None:
        return jsonify({"error": "HMM models not loaded"}), 503

    # ── Decode incoming audio ──────────────────────────────────────
    raw = None
    content_type = request.content_type or ""

    if "multipart/form-data" in content_type:
        f = request.files.get("audio")
        if f is None:
            return jsonify({"error": "No audio field in form"}), 400
        raw = f.read()
        fname = f.filename or ""
    else:
        raw = request.get_data()
        fname = ""

    if not raw:
        return jsonify({"error": "Empty audio data"}), 400

    # ── Convert to float32 numpy array at 16 kHz ──────────────────
    signal = None
    fs_in  = 16000

    # Try WAV first (scipy — no extra deps)
    try:
        import scipy.io.wavfile as wav_io
        fs_in, data = wav_io.read(io.BytesIO(raw))
        data = data.astype(np.float32)
        if data.ndim > 1:
            data = data.mean(axis=1)
        if data.max() > 1.5:
            data /= 32768.0
        signal = data
    except Exception:  # noqa: BLE001
        pass

    # Fallback: pydub (handles webm/ogg/mp3 via ffmpeg). The browser normally
    # decodes to a 16 kHz WAV client-side, so this only runs for raw/direct
    # posts or browsers without WebAudio. We surface the real reason because a
    # swallowed exception here ("Could not decode audio") was impossible to
    # debug — the usual culprit is a missing ffmpeg or pydub at runtime.
    if signal is None:
        try:
            import shutil
            from pydub import AudioSegment
            if shutil.which("ffmpeg") is None and shutil.which("avconv") is None:
                raise RuntimeError(
                    "ffmpeg not found on PATH (apt install ffmpeg). The dashboard "
                    "normally decodes audio in the browser, so update the web UI too."
                )
            seg = (
                AudioSegment.from_file(io.BytesIO(raw))
                .set_channels(1)
                .set_frame_rate(16000)
            )
            arr = np.array(seg.get_array_of_samples(), dtype=np.float32)
            arr /= 2 ** (seg.sample_width * 8 - 1)
            signal = arr
            fs_in  = 16000
        except Exception as exc:  # noqa: BLE001
            reason = f"{type(exc).__name__}: {exc}"
            print(
                f"[dashboard] /api/voice decode failed "
                f"({len(raw)} bytes, ct={content_type!r}, name={fname!r}): {reason}"
            )
            return jsonify({
                "error": "Could not decode audio",
                "reason": reason,
            }), 400

    # Resample if needed
    if fs_in != 16000:
        import scipy.signal as sp_signal
        signal = sp_signal.resample(signal, int(len(signal) * 16000 / fs_in)).astype(np.float32)

    if len(signal) < 800:
        return jsonify({"error": "Audio too short"}), 400

    # ── Recognise ─────────────────────────────────────────────────
    word = recognizer.recognize(signal, fs=16000)
    if word is None:
        return jsonify({"error": "Recognition failed"}), 500

    # ── Publish to ROS2 ───────────────────────────────────────────
    # Always publish the raw word (voice history + the SM's pause/stop handling).
    if ros_bridge is not None:
        try:
            ros_bridge.publish_voice_command(word)
        except Exception:  # noqa: BLE001
            pass

    # ── Drive the robot from the word (teleop nudge / SEARCH mission) ──
    action = _dispatch_voice_action(word)

    socketio.emit("voice_result", {"word": word, "action": action})
    return jsonify({"word": word, "action": action})


@app.route("/api/voice/status")
def voice_status():
    """Return whether the HMM recognizer is loaded and which words it knows."""
    r = _get_hmm_recognizer()
    if r is None:
        return jsonify({"ready": False, "vocabulary": []})
    return jsonify({"ready": True, "vocabulary": r.vocabulary})


@app.route("/api/teleop/stop", methods=["POST"])
def stop_teleop():
    """Immediately stop the robot."""
    global _teleop_active

    if ros_bridge is not None:
        ros_bridge.publish_cmd_vel(0.0, 0.0)

    with _teleop_lock:
        _teleop_active = False

    return jsonify({"ok": True})


# ---------------------------------------------------------------------------
# MJPEG generator
# ---------------------------------------------------------------------------

_PLACEHOLDER_JPEG: Optional[bytes] = None


def _get_placeholder_jpeg() -> bytes:
    """Return a small gray 320×240 JPEG used when no camera frame is available."""
    global _PLACEHOLDER_JPEG
    if _PLACEHOLDER_JPEG is not None:
        return _PLACEHOLDER_JPEG

    try:
        import numpy as np

        gray = (128 * np.ones((240, 320, 3), dtype=np.uint8))
        # Add "No Signal" text via OpenCV if available
        try:
            import cv2

            cv2.putText(
                gray, "No Signal", (90, 130),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 136), 2,
            )
            _, buf = cv2.imencode(".jpg", gray)
            _PLACEHOLDER_JPEG = bytes(buf)
        except ImportError:
            from PIL import Image as PILImage, ImageDraw
            import io

            img = PILImage.fromarray(gray.astype("uint8"), "RGB")
            draw = ImageDraw.Draw(img)
            draw.text((90, 110), "No Signal", fill=(0, 255, 136))
            buf = io.BytesIO()
            img.save(buf, format="JPEG")
            _PLACEHOLDER_JPEG = buf.getvalue()
    except Exception:  # noqa: BLE001
        # Absolute fallback
        from dashboard.ros_bridge import RosBridge
        _PLACEHOLDER_JPEG = RosBridge._minimal_jpeg()

    return _PLACEHOLDER_JPEG


def _mjpeg_generator():
    """Yield MJPEG frames at ~15 fps."""
    while True:
        if ros_bridge is not None:
            frame = ros_bridge.get_latest_frame()
        else:
            frame = None

        if frame is None:
            frame = _get_placeholder_jpeg()

        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n"
        )
        time.sleep(1.0 / 15.0)


# ---------------------------------------------------------------------------
# SocketIO background emitter
# ---------------------------------------------------------------------------

def _start_background_emitter() -> None:
    """Spawn the SocketIO emitter thread. Called once by dashboard_node."""
    t = threading.Thread(target=_background_socketio_emitter, daemon=True)
    t.start()


def _background_socketio_emitter() -> None:
    """
    Emit SocketIO events to all connected clients.

    - robot_pose   → 5 Hz
    - robot_state  → on change (polled at 5 Hz)
    - telemetry    → 1 Hz
    - scan_data    → 5 Hz (every other 200ms tick)
    - nav_plan     → on change
    """
    last_state_str = ""
    last_telemetry_time = 0.0
    last_nav_plan_str = ""
    last_sm_snapshot_str = ""
    last_transition_count = 0
    tick = 0
    interval = 0.2  # 5 Hz base tick

    while True:
        time.sleep(interval)
        tick += 1

        if ros_bridge is None:
            continue

        try:
            state = ros_bridge.get_state()
        except Exception:  # noqa: BLE001
            continue

        # 5 Hz — pose
        socketio.emit(
            "robot_pose",
            {
                "x": state.pose["x"],
                "y": state.pose["y"],
                "theta": state.pose["theta"],
            },
        )

        # On change — robot state + mission
        state_key = f"{state.state}|{json.dumps(state.mission, sort_keys=True)}"
        if state_key != last_state_str:
            last_state_str = state_key
            socketio.emit(
                "robot_state",
                {"state": state.state, "mission": state.mission},
            )

        # 5 Hz (every other tick ~2.5 Hz effectively, but emit every tick for simplicity)
        # Emit scan_data at 5 Hz when available
        if tick % 2 == 0 and state.scan is not None:
            socketio.emit("scan_data", state.scan)

        # Nav plan — on change
        nav_plan_str = json.dumps(state.nav_plan, sort_keys=True)
        if nav_plan_str != last_nav_plan_str:
            last_nav_plan_str = nav_plan_str
            socketio.emit(
                "nav_plan",
                {"points": state.nav_plan or []},
            )

        # mission_control SM snapshot — emit on change (~2 Hz upstream).
        sm_snapshot = ros_bridge.get_sm_snapshot()
        if sm_snapshot is not None:
            snap_str = json.dumps(sm_snapshot, sort_keys=True)
            if snap_str != last_sm_snapshot_str:
                last_sm_snapshot_str = snap_str
                socketio.emit("sm_snapshot", sm_snapshot)

        # SM transitions — emit only newly arrived ones.
        transitions = ros_bridge.get_sm_transitions()
        if len(transitions) > last_transition_count:
            new_transitions = transitions[last_transition_count:]
            last_transition_count = len(transitions)
            for tr in new_transitions:
                socketio.emit("sm_transition", tr)

        # 1 Hz — telemetry
        now = time.monotonic()
        if now - last_telemetry_time >= 1.0:
            last_telemetry_time = now

            # Map as base64-encoded PNG (send only when updated)
            map_png = ros_bridge.get_latest_map()
            map_b64 = (
                base64.b64encode(map_png).decode("ascii") if map_png else None
            )
            map_info = ros_bridge.get_map_info() or None

            socketio.emit(
                "telemetry",
                {
                    "linear_vel": state.velocity["linear"],
                    "angular_vel": state.velocity["angular"],
                    "lifter_level": state.lifter_level,
                    "map_png": map_b64,
                    "map_info": map_info,
                    "system_mode": state.system_mode,
                },
            )


# ---------------------------------------------------------------------------
# SocketIO connect/disconnect
# ---------------------------------------------------------------------------


@socketio.on("connect")
def on_connect():
    """Push full state to a freshly connected client."""
    if ros_bridge is None:
        return
    state = ros_bridge.get_state()
    socketio.emit(
        "robot_state",
        {"state": state.state, "mission": state.mission},
    )
    socketio.emit(
        "robot_pose",
        {"x": state.pose["x"], "y": state.pose["y"], "theta": state.pose["theta"]},
    )
    socketio.emit(
        "telemetry",
        {
            "linear_vel": state.velocity["linear"],
            "angular_vel": state.velocity["angular"],
            "lifter_level": state.lifter_level,
            "map_png": None,
        },
    )
    snap = ros_bridge.get_sm_snapshot()
    if snap is not None:
        socketio.emit("sm_snapshot", snap)
    transitions = ros_bridge.get_sm_transitions()
    if transitions:
        # Send all buffered transitions so the log isn't empty on reconnect.
        socketio.emit("sm_transition_bulk", transitions)
