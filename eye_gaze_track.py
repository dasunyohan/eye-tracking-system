import time
from collections import deque

import cv2
import numpy as np
from sklearn.linear_model import Ridge

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


# Configurations
MODEL_PATH = "models/face_landmarker.task"  
SCREEN_W, SCREEN_H = 1280, 720              # virtual screen canvas size for mapping
SAMPLES_PER_POINT = 35
SETTLE_TIME_S = 0.35
POINT_TIME_LIMIT_S = 3.0

# Smoothing
SMOOTH_ALPHA = 0.30
TRAIL_LEN = 200

# Landmark indices (MediaPipe FaceMesh topology)
LEFT_EYE_OUTER = 33
LEFT_EYE_INNER = 133
RIGHT_EYE_INNER = 362
RIGHT_EYE_OUTER = 263

LEFT_EYE_TOP = 159
LEFT_EYE_BOTTOM = 145
RIGHT_EYE_TOP = 386
RIGHT_EYE_BOTTOM = 374

# Iris indices exist ONLY if the model provides them (typically total landmarks >= 478)
LEFT_IRIS = [468, 469, 470, 471, 472]
RIGHT_IRIS = [473, 474, 475, 476, 477]


# helpers for robust statistics and feature extraction
def robust_center(points: np.ndarray) -> np.ndarray:
    return np.median(points, axis=0)


def robust_filter(points: np.ndarray, z_thresh: float = 3.5) -> np.ndarray:
    if len(points) < 10:
        return points
    med = np.median(points, axis=0)
    abs_dev = np.abs(points - med)
    mad = np.median(abs_dev, axis=0) + 1e-9
    z = 0.6745 * abs_dev / mad
    keep = np.all(z < z_thresh, axis=1)
    if np.sum(keep) < max(10, len(points) // 3):
        return points
    return points[keep]


class ExpSmoother:
    def __init__(self, alpha=0.25):
        self.alpha = float(alpha)
        self.v = None

    def reset(self):
        self.v = None

    def update(self, x):
        x = np.array(x, dtype=np.float64)
        if self.v is None:
            self.v = x
        else:
            self.v = self.alpha * x + (1.0 - self.alpha) * self.v
        return self.v


def make_calibration_points(w, h):
    mx, my = int(0.15 * w), int(0.15 * h)
    xs = [mx, w // 2, w - mx]
    ys = [my, h // 2, h - my]
    return [(x, y) for y in ys for x in xs]


def put_text(canvas, text, y=40):
    cv2.putText(canvas, text, (30, y), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)


def draw_dot(canvas, pt, radius=10):
    cv2.circle(canvas, pt, radius, (0, 255, 0), -1, lineType=cv2.LINE_AA)
    cv2.circle(canvas, pt, radius + 8, (0, 255, 0), 2, lineType=cv2.LINE_AA)


def lm_to_px(lm, w, h):
    return np.array([lm.x * w, lm.y * h], dtype=np.float64)


def rotation_matrix_to_yaw_pitch_roll(R: np.ndarray):
    """
    Convert rotation matrix to yaw, pitch, roll (radians).
    Convention is approximate; used only as a stabilizing feature for regression.
    """
    sy = np.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])
    singular = sy < 1e-6
    if not singular:
        pitch = np.arctan2(R[2, 1], R[2, 2])
        yaw = np.arctan2(-R[2, 0], sy)
        roll = np.arctan2(R[1, 0], R[0, 0])
    else:
        pitch = np.arctan2(-R[1, 2], R[1, 1])
        yaw = np.arctan2(-R[2, 0], sy)
        roll = 0.0
    return float(yaw), float(pitch), float(roll)


def extract_head_pose_from_transform(result):
    """
    If FaceLandmarkerOptions has output_facial_transformation_matrixes=True,
    result.facial_transformation_matrixes may exist (list of 4x4 matrices).
    We'll extract yaw/pitch/roll from the rotation component.
    """
    try:
        mats = result.facial_transformation_matrixes
        if not mats:
            return 0.0, 0.0, 0.0
        M = np.array(mats[0].data, dtype=np.float64).reshape(4, 4)
        R = M[:3, :3]
        return rotation_matrix_to_yaw_pitch_roll(R)
    except Exception:
        return 0.0, 0.0, 0.0


def iris_center_or_fallback(lms, w, h):
    """
    Returns (left_center_px, right_center_px, iris_available_bool)
    If iris landmarks exist (len>=478), use mean of iris points.
    Else fallback to midpoint between eye corners.
    """
    n = len(lms)
    # Eye corners
    l_outer = lm_to_px(lms[LEFT_EYE_OUTER], w, h)
    l_inner = lm_to_px(lms[LEFT_EYE_INNER], w, h)
    r_inner = lm_to_px(lms[RIGHT_EYE_INNER], w, h)
    r_outer = lm_to_px(lms[RIGHT_EYE_OUTER], w, h)

    if n >= 478:
        left_pts = np.array([lm_to_px(lms[i], w, h) for i in LEFT_IRIS], dtype=np.float64)
        right_pts = np.array([lm_to_px(lms[i], w, h) for i in RIGHT_IRIS], dtype=np.float64)
        return left_pts.mean(axis=0), right_pts.mean(axis=0), True

    # fallback: center between corners
    return (l_outer + l_inner) / 2.0, (r_outer + r_inner) / 2.0, False


def extract_features(lms, w, h, yaw, pitch, roll):
    """
    Feature vector:
      [l_x, l_y, r_x, r_y, yaw, pitch, roll, interocular_px]
    where l_x/l_y/r_x/r_y are normalized positions in eye coordinates.
    """
    # Eye anchors
    l_outer = lm_to_px(lms[LEFT_EYE_OUTER], w, h)
    l_inner = lm_to_px(lms[LEFT_EYE_INNER], w, h)
    r_inner = lm_to_px(lms[RIGHT_EYE_INNER], w, h)
    r_outer = lm_to_px(lms[RIGHT_EYE_OUTER], w, h)

    l_top = lm_to_px(lms[LEFT_EYE_TOP], w, h)
    l_bot = lm_to_px(lms[LEFT_EYE_BOTTOM], w, h)
    r_top = lm_to_px(lms[RIGHT_EYE_TOP], w, h)
    r_bot = lm_to_px(lms[RIGHT_EYE_BOTTOM], w, h)

    left_c, right_c, _ = iris_center_or_fallback(lms, w, h)

    # Normalize within each eye
    l_dx = np.linalg.norm(l_outer - l_inner) + 1e-6
    r_dx = np.linalg.norm(r_outer - r_inner) + 1e-6
    l_dy = np.linalg.norm(l_bot - l_top) + 1e-6
    r_dy = np.linalg.norm(r_bot - r_top) + 1e-6

    # x increases towards outer corner on both eyes
    l_x = np.dot(left_c - l_inner, (l_outer - l_inner) / l_dx) / 1.0
    r_x = np.dot(right_c - r_inner, (r_outer - r_inner) / r_dx) / 1.0

    l_y = np.dot(left_c - l_top, (l_bot - l_top) / l_dy) / 1.0
    r_y = np.dot(right_c - r_top, (r_bot - r_top) / r_dy) / 1.0

    # then normalize by dividing ONCE:
    l_x = l_x / l_dx
    r_x = r_x / r_dx
    l_y = l_y / l_dy
    r_y = r_y / r_dy

    l_x, l_y = np.clip([l_x, l_y], -0.5, 1.5)
    r_x, r_y = np.clip([r_x, r_y], -0.5, 1.5)

    interocular = np.linalg.norm(l_outer - r_outer)

    return np.array([l_x, l_y, r_x, r_y, yaw, pitch, roll, interocular], dtype=np.float64)


def build_landmarker():
    BaseOptions = python.BaseOptions
    FaceLandmarker = vision.FaceLandmarker
    FaceLandmarkerOptions = vision.FaceLandmarkerOptions
    VisionRunningMode = vision.RunningMode

    options = FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=VisionRunningMode.VIDEO,
        num_faces=1,
        output_face_blendshapes=False,
        output_facial_transformation_matrixes=True,  # gives head transform matrix
    )
    return FaceLandmarker.create_from_options(options)



def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise SystemExit("Could not open webcam.")

    # Fullscreen window
    win = "Gaze Path (Tasks API) — Calibrate then Live"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(win, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    landmarker = build_landmarker()

    # Models
    model_x = Ridge(alpha=1.0)
    model_y = Ridge(alpha=1.0)
    smoother = ExpSmoother(alpha=SMOOTH_ALPHA)
    trail = deque(maxlen=TRAIL_LEN)

    def reset_all():
        nonlocal model_x, model_y
        model_x = Ridge(alpha=1.0)
        model_y = Ridge(alpha=1.0)
        smoother.reset()
        trail.clear()

    calib_pts = make_calibration_points(SCREEN_W, SCREEN_H)
    X_feats = []
    Y_targets = []

    # Calibration loop
    for idx, pt in enumerate(calib_pts):
        t0 = time.time()
        collected = []

        while True:
            ok, frame = cap.read()
            if not ok:
                continue

            frame = cv2.flip(frame, 1)
            h_cam, w_cam = frame.shape[:2]

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            ts_ms = int(time.time() * 1000)

            result = landmarker.detect_for_video(mp_image, ts_ms)

            canvas = np.zeros((SCREEN_H, SCREEN_W, 3), dtype=np.uint8)
            put_text(canvas, f"Calibration {idx+1}/{len(calib_pts)}: Look at the dot", 40)
            put_text(canvas, "Keep head steady. ESC quit. R restart.", 75)
            draw_dot(canvas, pt)

            # camera preview
            preview = cv2.resize(frame, (320, 180))
            canvas[10:10+180, SCREEN_W-10-320:SCREEN_W-10] = preview

            if result.face_landmarks:
                lms = result.face_landmarks[0]
                yaw, pitch, roll = extract_head_pose_from_transform(result)

                feat = extract_features(lms, w_cam, h_cam, yaw, pitch, roll)

                if (time.time() - t0) > SETTLE_TIME_S and len(collected) < SAMPLES_PER_POINT:
                    collected.append(feat)

                # show if iris landmarks exist
                iris_ok = "YES" if len(lms) >= 478 else "NO"
                cv2.putText(canvas, f"Iris landmarks: {iris_ok} (N={len(lms)})",
                            (30, SCREEN_H - 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                            (200, 200, 200), 2, cv2.LINE_AA)

            cv2.putText(canvas, f"Samples: {len(collected)}/{SAMPLES_PER_POINT}",
                        (30, SCREEN_H - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                        (200, 200, 200), 2, cv2.LINE_AA)

            cv2.imshow(win, canvas)
            key = cv2.waitKey(1) & 0xFF
            if key == 27:
                cap.release()
                cv2.destroyAllWindows()
                return
            if key in (ord('r'), ord('R')):
                # restart everything
                reset_all()
                X_feats = []
                Y_targets = []
                idx = -1
                break

            # done collecting?
            if len(collected) >= SAMPLES_PER_POINT:
                break

            # time limit fallback
            if (time.time() - t0) > POINT_TIME_LIMIT_S and len(collected) >= max(10, SAMPLES_PER_POINT // 3):
                break

        if idx == -1:
            calib_pts = make_calibration_points(SCREEN_W, SCREEN_H)
            continue

        if len(collected) < 10:
            continue

        collected = np.array(collected, dtype=np.float64)
        collected = robust_filter(collected)
        center_feat = robust_center(collected)

        X_feats.append(center_feat)
        Y_targets.append(np.array(pt, dtype=np.float64))

    if len(X_feats) < 6:
        cap.release()
        cv2.destroyAllWindows()
        raise SystemExit(
            "Calibration failed (not enough points). Try:\n"
            "- better lighting\n- face centered\n- avoid backlight\n- keep head steady during calibration"
        )

    X = np.vstack(X_feats)
    Y = np.vstack(Y_targets)

    model_x.fit(X, Y[:, 0])
    model_y.fit(X, Y[:, 1])

    # Live tracking loop
    while True:
        ok, frame = cap.read()
        if not ok:
            continue

        frame = cv2.flip(frame, 1)
        h_cam, w_cam = frame.shape[:2]

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        ts_ms = int(time.time() * 1000)

        result = landmarker.detect_for_video(mp_image, ts_ms)

        canvas = np.zeros((SCREEN_H, SCREEN_W, 3), dtype=np.uint8)
        put_text(canvas, "LIVE: gaze dot + path   (R recalibrate, ESC quit)", 40)

        # camera preview
        preview = cv2.resize(frame, (320, 180))
        canvas[10:10+180, SCREEN_W-10-320:SCREEN_W-10] = preview

        if result.face_landmarks:
            lms = result.face_landmarks[0]
            yaw, pitch, roll = extract_head_pose_from_transform(result)

            feat = extract_features(lms, w_cam, h_cam, yaw, pitch, roll).reshape(1, -1)

            gx = float(model_x.predict(feat)[0])
            gy = float(model_y.predict(feat)[0])

            gx = float(np.clip(gx, 0, SCREEN_W - 1))
            gy = float(np.clip(gy, 0, SCREEN_H - 1))

            sx, sy = smoother.update([gx, gy])
            sx, sy = int(sx), int(sy)

            trail.append((sx, sy))

            if len(trail) >= 2:
                pts = np.array(trail, dtype=np.int32).reshape((-1, 1, 2))
                cv2.polylines(canvas, [pts], False, (255, 255, 255), 2, cv2.LINE_AA)

            cv2.circle(canvas, (sx, sy), 8, (0, 255, 0), -1, cv2.LINE_AA)
            cv2.circle(canvas, (sx, sy), 16, (0, 255, 0), 2, cv2.LINE_AA)

            iris_ok = "YES" if len(lms) >= 478 else "NO"
            cv2.putText(canvas, f"Iris landmarks: {iris_ok} (N={len(lms)})",
                        (30, SCREEN_H - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                        (200, 200, 200), 2, cv2.LINE_AA)
        else:
            put_text(canvas, "No face detected. Center your face and improve lighting.", 85)

        cv2.imshow(win, canvas)
        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            break
        if key in (ord('r'), ord('R')):
            # restart by re-running main (simple + reliable)
            cap.release()
            cv2.destroyAllWindows()
            main()
            return

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()