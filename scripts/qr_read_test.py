#!/usr/bin/env python3
"""QR read diagnostic — figure out *why* /qr_detected stays empty.

It isolates each layer so you can see exactly where the read breaks:

  1. env      — is pyzbar/ZBar importable, what cv2 backends exist.
  2. synthetic — generate a QR with a known payload, decode it with pyzbar AND
                 cv2, and run the project's QRPoseDetector on it end-to-end.
                 (Also tested downscaled/blurred to mimic a far/again camera.)
  3. image    — decode a saved frame:        qr_read_test.py --image foo.png
  4. topic    — decode the LIVE camera topic: qr_read_test.py --topic /cam_img
                (this is the real on-robot test — needs ROS sourced & running)

Run on the machine that runs qr_quad_alignment (the one with the camera).

    python3 scripts/qr_read_test.py                 # env + synthetic
    python3 scripts/qr_read_test.py --image f.png
    python3 scripts/qr_read_test.py --topic /cam_img --frames 60
"""
from __future__ import annotations

import argparse
import sys

import numpy as np

try:
    import cv2
except Exception as exc:  # noqa: BLE001
    print(f"[FATAL] cannot import cv2: {exc}")
    sys.exit(1)


# --------------------------------------------------------------------------
# decoders
# --------------------------------------------------------------------------
def decode_pyzbar(gray: np.ndarray) -> list[str]:
    """Decode every QR string with ZBar via pyzbar (.data attribute)."""
    try:
        from pyzbar import pyzbar
    except Exception as exc:  # noqa: BLE001
        print(f"   pyzbar import FAILED: {exc}")
        return []
    out = []
    for r in pyzbar.decode(gray):
        raw = getattr(r, "data", b"")
        txt = raw.decode("utf-8", "replace") if isinstance(raw, (bytes, bytearray)) else str(raw)
        if txt:
            poly = getattr(r, "polygon", None)
            n = len(poly) if poly else 0
            out.append(txt)
            print(f"   [pyzbar] type={getattr(r,'type','?')} polygon_pts={n} data={txt!r}")
    return out


def decode_cv2(gray: np.ndarray) -> list[str]:
    """Decode with both cv2 backends (classic + Aruco) so we can compare."""
    out = []
    backends = [("classic", cv2.QRCodeDetector)]
    if hasattr(cv2, "QRCodeDetectorAruco"):
        backends.append(("aruco", cv2.QRCodeDetectorAruco))
    for name, ctor in backends:
        det = ctor()
        got = ""
        try:
            ok, decoded, points, _ = det.detectAndDecodeMulti(gray)
            if ok and decoded:
                got = " | ".join(repr(d) for d in decoded)
        except cv2.error:
            ok = False
        if not got:
            try:
                data, points, _ = det.detectAndDecode(gray)
                got = repr(data)
            except cv2.error:
                got = "(cv2.error)"
        # 'detected box but empty string' is the exact failure we are chasing.
        # Report the box SIZE in px too: on a 320x240 cam a QR under ~40 px is
        # usually too small for ANY decoder — that means "get closer", not a bug.
        detected = points is not None and len(np.asarray(points)) > 0
        size_px = ""
        if detected:
            pts = np.asarray(points, dtype=np.float32).reshape(-1, 2)
            wpx = float(pts[:, 0].max() - pts[:, 0].min())
            hpx = float(pts[:, 1].max() - pts[:, 1].min())
            size_px = f" box~{wpx:.0f}x{hpx:.0f}px"
        print(f"   [cv2:{name:7s}] detected_box={detected}{size_px} decoded={got}")
        if got and got not in ("''", '""', "(cv2.error)"):
            out.append(got)
    return out


# --------------------------------------------------------------------------
# QR generation (for the synthetic self-test)
# --------------------------------------------------------------------------
def make_qr(text: str, px: int = 400) -> np.ndarray | None:
    """Return a BGR image with `text` encoded as a QR, or None if no encoder."""
    if hasattr(cv2, "QRCodeEncoder_create") or hasattr(cv2, "QRCodeEncoder"):
        try:
            enc = (cv2.QRCodeEncoder_create() if hasattr(cv2, "QRCodeEncoder_create")
                   else cv2.QRCodeEncoder.create())
            qr = enc.encode(text)                       # single-channel 0/255
            qr = cv2.resize(qr, (px, px), interpolation=cv2.INTER_NEAREST)
            img = cv2.cvtColor(qr, cv2.COLOR_GRAY2BGR)
            # pad with a quiet zone (ZBar/cv2 need white border around the code)
            return cv2.copyMakeBorder(img, 60, 60, 60, 60, cv2.BORDER_CONSTANT,
                                      value=(255, 255, 255))
        except Exception as exc:  # noqa: BLE001
            print(f"   cv2.QRCodeEncoder failed: {exc}")
    try:
        import qrcode
        q = qrcode.QRCode(border=4, box_size=10)
        q.add_data(text); q.make(fit=True)
        pil = q.make_image(fill_color="black", back_color="white").convert("RGB")
        return cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
    except Exception:  # noqa: BLE001
        return None


def report(tag: str, gray: np.ndarray) -> None:
    print(f"-- {tag}  ({gray.shape[1]}x{gray.shape[0]}) --")
    pz = decode_pyzbar(gray)
    cv = decode_cv2(gray)
    print(f"   => pyzbar read {len(pz)} | cv2 read {len(cv)}")


# --------------------------------------------------------------------------
# modes
# --------------------------------------------------------------------------
def run_env() -> None:
    print("=" * 70)
    print("ENV")
    print(f"   cv2 {cv2.__version__}  QRCodeDetectorAruco={hasattr(cv2,'QRCodeDetectorAruco')}"
          f"  QRCodeEncoder={hasattr(cv2,'QRCodeEncoder_create') or hasattr(cv2,'QRCodeEncoder')}")
    try:
        from pyzbar import pyzbar  # noqa: F401
        print("   pyzbar: OK (ZBar shared lib loaded)")
    except Exception as exc:  # noqa: BLE001
        print(f"   pyzbar: NOT USABLE -> {exc}")
        print("   fix: sudo apt install -y libzbar0 && pip install pyzbar")


def run_synthetic() -> bool:
    print("=" * 70)
    print("SYNTHETIC QR  (payload='T1')")
    img = make_qr("T1")
    if img is None:
        print("   [skip] no QR encoder available (cv2.QRCodeEncoder & qrcode both missing).")
        print("   install one:  pip install qrcode[pil]")
        return False
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    report("clean", gray)
    report("downscaled x0.4", cv2.resize(gray, None, fx=0.4, fy=0.4))
    report("blurred", cv2.GaussianBlur(gray, (5, 5), 0))

    # End-to-end through the project's detector (needs camera intrinsics; any
    # plausible K works for the read — pose accuracy is irrelevant here).
    print("-- QRPoseDetector.detect() end-to-end --")
    try:
        from perception.qr_pose_detector import QRPoseDetector
    except Exception as exc:  # noqa: BLE001
        print(f"   [skip] cannot import QRPoseDetector (source workspace?): {exc}")
        return bool(decode_pyzbar(gray))
    h, w = gray.shape[:2]
    K = np.array([[w, 0, w / 2], [0, w, h / 2], [0, 0, 1]], dtype=np.float64)
    det = QRPoseDetector(K, np.zeros(5), marker_length=0.05, refine=True, backend="auto")
    print(f"   backend={det.backend!r}  pyzbar_available={det.pyzbar_available}")
    dets = det.detect(img)
    ids = [d["id"] for d in dets]
    print(f"   detect() -> {len(dets)} det(s), ids={ids}")
    ok = any(i == "T1" for i in ids)
    print(f"   RESULT: {'READ OK ✅' if ok else 'NOT READ ❌'}")
    return ok


def run_image(path: str) -> None:
    print("=" * 70)
    print(f"IMAGE  {path}")
    img = cv2.imread(path)
    if img is None:
        print("   [error] could not read image.")
        return
    report("image", cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))


def run_topic(topic: str, frames: int) -> None:
    print("=" * 70)
    print(f"LIVE TOPIC  {topic}  (decoding up to {frames} frames)")
    try:
        import rclpy
        from rclpy.node import Node
        from rclpy.qos import qos_profile_sensor_data
        from sensor_msgs.msg import Image
        from cv_bridge import CvBridge
    except Exception as exc:  # noqa: BLE001
        print(f"   [error] ROS not available: {exc}  (did you 'source install/setup.bash'?)")
        return

    bridge = CvBridge()
    state = {"n": 0, "hits": 0, "logged_shape": False}

    class _Probe(Node):
        def __init__(self):
            super().__init__("qr_read_probe")
            self.create_subscription(Image, topic, self._cb, qos_profile_sensor_data)

        def _cb(self, msg: Image):
            state["n"] += 1
            try:
                frame = bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
            except Exception:
                frame = bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
            if not state["logged_shape"]:
                print(f"   first frame {gray.shape[1]}x{gray.shape[0]} encoding={msg.encoding!r}")
                state["logged_shape"] = True
            pz = decode_pyzbar(gray)
            if pz:
                state["hits"] += 1
                print(f"   frame {state['n']}: pyzbar -> {pz}")
            elif state["n"] % 15 == 0:
                decode_cv2(gray)  # periodic cv2 comparison

    rclpy.init()
    node = _Probe()
    try:
        while rclpy.ok() and state["n"] < frames:
            rclpy.spin_once(node, timeout_sec=1.0)
            if state["n"] == 0:
                print("   ...waiting for frames (is the camera/qr topic publishing?)")
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
    print(f"   RESULT: {state['n']} frames, pyzbar decoded a QR in {state['hits']} of them.")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--image", help="decode a saved image file")
    ap.add_argument("--topic", help="decode a live ROS camera topic (e.g. /cam_img)")
    ap.add_argument("--frames", type=int, default=60, help="frames to sample in --topic mode")
    args = ap.parse_args()

    run_env()
    if args.image:
        run_image(args.image)
    elif args.topic:
        run_topic(args.topic, args.frames)
    else:
        run_synthetic()


if __name__ == "__main__":
    main()
