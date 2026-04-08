#!/usr/bin/env python3
"""
CueCatcher Face Calibration Tool

Standard facial expression models fail for children with 9p deletion due to:
  - Hypertelorism (wide-set eyes)
  - Midface hypoplasia (flattened midface)
  - Micrognathia (small jaw)
  - Facial hypotonia (reduced muscle tone → smaller expression range)

This tool builds a personalized expression baseline:
  1. Captures 50-100 labeled images across emotional states
  2. Computes the child's neutral face geometry
  3. Builds expression deltas relative to THEIR neutral (not a generic face)
  4. Saves calibration data for the inference pipeline

Usage:
  python calibrate.py --mode capture   # Interactive capture session
  python calibrate.py --mode build     # Build calibration from saved images
  python calibrate.py --mode test      # Test calibration on live video
"""

import argparse
import json
import time
from pathlib import Path
from datetime import datetime

import cv2
import numpy as np

try:
    import mediapipe as mp
    MP_AVAILABLE = True
except ImportError:
    MP_AVAILABLE = False


CALIBRATION_DIR = Path("data/calibration")
STATES = [
    ("neutral", "Child is calm and relaxed — no strong emotion"),
    ("happy", "Child is smiling, laughing, or showing pleasure"),
    ("distress", "Child is crying, grimacing, or in discomfort"),
    ("interest", "Child is focused on something — eyes wide, leaning forward"),
    ("discomfort", "Child is squirming, frowning, or showing mild displeasure"),
    ("excitement", "Child is animated — wide eyes, open mouth, maybe vocalizing"),
    ("sleepy", "Child is drowsy — heavy lids, relaxed face"),
]

MIN_IMAGES_PER_STATE = 5
TARGET_IMAGES_PER_STATE = 15


def capture_session(camera_index: int = 0):
    """Interactive guided capture session with the caregiver."""
    CALIBRATION_DIR.mkdir(parents=True, exist_ok=True)

    if not MP_AVAILABLE:
        print("ERROR: MediaPipe not installed. Run: pip install mediapipe")
        return

    face_mesh = mp.solutions.face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
    )

    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print(f"ERROR: Cannot open camera {camera_index}")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    print("\n🧭 CueCatcher Face Calibration")
    print("═" * 50)
    print()
    print("This tool captures your child's facial expressions to build")
    print("a personalized expression model. Standard AI face models")
    print("don't work well for children with craniofacial differences.")
    print()
    print("Instructions:")
    print("  - Position the camera so your child's face is clearly visible")
    print("  - We'll go through each emotional state")
    print("  - Press SPACE to capture a frame when the expression is clear")
    print("  - Press 'n' to skip to the next state")
    print("  - Press 'q' to quit")
    print()
    print(f"We need at least {MIN_IMAGES_PER_STATE} images per state,")
    print(f"ideally {TARGET_IMAGES_PER_STATE}.")
    print()
    input("Press Enter to begin...")

    all_captures = {}

    for state_name, state_desc in STATES:
        print(f"\n── {state_name.upper()} ──")
        print(f"   {state_desc}")
        print(f"   Press SPACE to capture, 'n' for next state, 'q' to quit")
        print()

        captures = []
        state_dir = CALIBRATION_DIR / state_name
        state_dir.mkdir(exist_ok=True)

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Detect face
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = face_mesh.process(rgb)

            display = frame.copy()

            if result.multi_face_landmarks:
                face = result.multi_face_landmarks[0]
                h, w = frame.shape[:2]

                # Draw face mesh outline
                for idx in [10, 338, 297, 332, 284, 251, 389, 356, 454,
                           323, 361, 288, 397, 365, 379, 378, 400, 377,
                           152, 148, 176, 149, 150, 136, 172, 58, 132,
                           93, 234, 127, 162, 21, 54, 103, 67, 109]:
                    lm = face.landmark[idx]
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    cv2.circle(display, (cx, cy), 2, (0, 255, 0), -1)

                # Show face quality indicator
                cv2.putText(display, "Face detected ✓", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            else:
                cv2.putText(display, "No face detected", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            # Status bar
            cv2.putText(display, f"State: {state_name} | Captured: {len(captures)}/{TARGET_IMAGES_PER_STATE}",
                       (10, display.shape[0] - 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            cv2.imshow("CueCatcher Calibration", display)
            key = cv2.waitKey(1) & 0xFF

            if key == ord(" ") and result.multi_face_landmarks:
                # Capture this frame
                timestamp = int(time.time() * 1000)
                filename = f"{state_name}_{timestamp}.jpg"
                filepath = state_dir / filename

                cv2.imwrite(str(filepath), frame)

                # Extract and save landmarks
                landmarks = extract_landmarks(face, frame.shape[1], frame.shape[0])
                captures.append({
                    "file": filename,
                    "landmarks": landmarks,
                    "timestamp": timestamp,
                })

                print(f"  📸 Captured ({len(captures)}/{TARGET_IMAGES_PER_STATE})")

                if len(captures) >= TARGET_IMAGES_PER_STATE:
                    print(f"  ✅ Got enough for '{state_name}'!")
                    break

            elif key == ord("n"):
                if len(captures) >= MIN_IMAGES_PER_STATE:
                    print(f"  ✅ Moving on ({len(captures)} captured)")
                    break
                else:
                    print(f"  ⚠️  Need at least {MIN_IMAGES_PER_STATE} images ({len(captures)} so far)")

            elif key == ord("q"):
                print("\nCalibration ended early.")
                cap.release()
                cv2.destroyAllWindows()
                _save_calibration(all_captures)
                return

        all_captures[state_name] = captures

    cap.release()
    cv2.destroyAllWindows()
    face_mesh.close()

    _save_calibration(all_captures)


def extract_landmarks(face, img_w: int, img_h: int) -> list:
    """Extract key facial landmark coordinates."""
    key_indices = {
        "nose_tip": 1,
        "chin": 152,
        "left_eye_outer": 33,
        "left_eye_inner": 133,
        "right_eye_outer": 263,
        "right_eye_inner": 362,
        "left_eyebrow_outer": 46,
        "left_eyebrow_inner": 105,
        "right_eyebrow_outer": 276,
        "right_eyebrow_inner": 334,
        "upper_lip": 13,
        "lower_lip": 14,
        "left_mouth": 61,
        "right_mouth": 291,
        "left_cheek": 234,
        "right_cheek": 454,
        "forehead": 10,
        "left_iris": 468,   # requires refine_landmarks=True
        "right_iris": 473,
    }

    result = {}
    for name, idx in key_indices.items():
        try:
            lm = face.landmark[idx]
            result[name] = {
                "x": round(lm.x * img_w, 1),
                "y": round(lm.y * img_h, 1),
                "z": round(lm.z * img_w, 1),  # depth, scaled to image width
            }
        except (IndexError, AttributeError):
            pass

    return result


def build_calibration():
    """Build calibration model from captured images."""
    print("\n🧭 Building calibration model...")

    calib_file = CALIBRATION_DIR / "calibration.json"
    if calib_file.exists():
        with open(calib_file) as f:
            data = json.load(f)
    else:
        print("ERROR: No calibration data found. Run --mode capture first.")
        return

    # Compute per-state average landmarks
    state_profiles = {}
    for state_name, captures in data["states"].items():
        if not captures:
            continue

        all_landmarks = [c["landmarks"] for c in captures if c.get("landmarks")]
        if not all_landmarks:
            continue

        # Average each landmark across all captures
        avg = {}
        keys = all_landmarks[0].keys()
        for key in keys:
            xs = [lm[key]["x"] for lm in all_landmarks if key in lm]
            ys = [lm[key]["y"] for lm in all_landmarks if key in lm]
            if xs and ys:
                avg[key] = {
                    "x": round(np.mean(xs), 1),
                    "y": round(np.mean(ys), 1),
                    "std_x": round(np.std(xs), 2),
                    "std_y": round(np.std(ys), 2),
                }

        state_profiles[state_name] = {
            "average_landmarks": avg,
            "sample_count": len(all_landmarks),
        }

    # Compute expression deltas (relative to neutral)
    if "neutral" not in state_profiles:
        print("ERROR: No neutral state captured. Re-run calibration.")
        return

    neutral = state_profiles["neutral"]["average_landmarks"]
    expression_deltas = {}

    for state_name, profile in state_profiles.items():
        if state_name == "neutral":
            continue

        deltas = {}
        for key in neutral:
            if key in profile["average_landmarks"]:
                deltas[key] = {
                    "dx": round(profile["average_landmarks"][key]["x"] - neutral[key]["x"], 2),
                    "dy": round(profile["average_landmarks"][key]["y"] - neutral[key]["y"], 2),
                }

        expression_deltas[state_name] = deltas

    # Compute key facial metrics for this child
    face_metrics = compute_face_metrics(neutral)

    # Save
    calibration = {
        "child_id": "default",
        "created_at": datetime.now().isoformat(),
        "state_profiles": state_profiles,
        "expression_deltas": expression_deltas,
        "face_metrics": face_metrics,
        "neutral_baseline": neutral,
    }

    output = CALIBRATION_DIR / "model.json"
    with open(output, "w") as f:
        json.dump(calibration, f, indent=2)

    print(f"✅ Calibration model saved: {output}")
    print(f"   States: {list(state_profiles.keys())}")
    print(f"   Face metrics: {face_metrics}")

    # Report
    print("\n── Calibration Report ──")
    for state, profile in state_profiles.items():
        print(f"  {state:15s}: {profile['sample_count']} samples")

    if face_metrics.get("eye_distance_ratio", 0) > 0.35:
        print("\n  ⚠️  Detected hypertelorism (wide eye spacing)")
        print("     Standard AU models will underperform.")
        print("     The personalized deltas will compensate.")


def compute_face_metrics(neutral_landmarks: dict) -> dict:
    """Compute key facial geometry metrics specific to this child."""
    metrics = {}

    try:
        # Inter-eye distance relative to face width
        l_eye = neutral_landmarks.get("left_eye_outer", {})
        r_eye = neutral_landmarks.get("right_eye_outer", {})
        l_cheek = neutral_landmarks.get("left_cheek", {})
        r_cheek = neutral_landmarks.get("right_cheek", {})

        if all([l_eye, r_eye, l_cheek, r_cheek]):
            eye_dist = abs(r_eye.get("x", 0) - l_eye.get("x", 0))
            face_width = abs(r_cheek.get("x", 0) - l_cheek.get("x", 0))
            if face_width > 0:
                metrics["eye_distance_ratio"] = round(eye_dist / face_width, 3)

        # Mouth width relative to face width
        l_mouth = neutral_landmarks.get("left_mouth", {})
        r_mouth = neutral_landmarks.get("right_mouth", {})
        if l_mouth and r_mouth and face_width > 0:
            mouth_width = abs(r_mouth.get("x", 0) - l_mouth.get("x", 0))
            metrics["mouth_width_ratio"] = round(mouth_width / face_width, 3)

        # Nose-to-chin relative to forehead-to-chin (midface proportion)
        nose = neutral_landmarks.get("nose_tip", {})
        chin = neutral_landmarks.get("chin", {})
        forehead = neutral_landmarks.get("forehead", {})
        if nose and chin and forehead:
            nose_chin = abs(chin.get("y", 0) - nose.get("y", 0))
            forehead_chin = abs(chin.get("y", 0) - forehead.get("y", 0))
            if forehead_chin > 0:
                metrics["midface_ratio"] = round(nose_chin / forehead_chin, 3)

        # Lip distance at neutral (baseline for mouth openness)
        upper = neutral_landmarks.get("upper_lip", {})
        lower = neutral_landmarks.get("lower_lip", {})
        if upper and lower:
            metrics["neutral_lip_gap"] = round(
                abs(lower.get("y", 0) - upper.get("y", 0)), 1
            )

    except Exception as e:
        print(f"Warning: metrics computation error: {e}")

    return metrics


def _save_calibration(all_captures: dict):
    """Save raw calibration data."""
    calib_file = CALIBRATION_DIR / "calibration.json"
    data = {
        "created_at": datetime.now().isoformat(),
        "states": all_captures,
    }
    with open(calib_file, "w") as f:
        json.dump(data, f, indent=2)
    print(f"\n✅ Raw data saved: {calib_file}")
    print(f"   Run 'python calibrate.py --mode build' to create the model.")


def main():
    parser = argparse.ArgumentParser(description="CueCatcher Face Calibration")
    parser.add_argument("--mode", choices=["capture", "build", "test"], default="capture")
    parser.add_argument("--camera", type=int, default=0, help="Camera index")
    args = parser.parse_args()

    if args.mode == "capture":
        capture_session(args.camera)
    elif args.mode == "build":
        build_calibration()
    elif args.mode == "test":
        print("Test mode not yet implemented. Coming in Phase 3.")


if __name__ == "__main__":
    main()
