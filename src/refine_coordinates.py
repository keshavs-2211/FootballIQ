"""Offline FootballIQ refinement: camera-aware pitch projection, appearance-based
ID continuation proposals, verified/manual merges and RTS Kalman smoothing.

This DOES NOT change source ByteTrack IDs or invent actual detections.
Run: python refine_coordinates.py
"""
import csv
from collections import Counter, defaultdict, deque
from pathlib import Path
import json

import cv2
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

import offline_settings as S
from camera.camera_movement import CameraMotionCompensator

cv2.setNumThreads(2)


def check_inputs():
    for path in (S.VIDEO, S.FIRST_PASS_CSV, S.CALIBRATION_JSON):
        if not Path(path).is_file():
            raise FileNotFoundError(f"Missing {path}")


def team_name(x):
    v = str(x).strip()
    return v if v in ("Team A", "Team B") else "Unknown"


def stable_teams(data):
    """Backfill same-ID Unknown frames; do not force mixed-label tracks."""
    data = data.copy()
    data["team"] = data["team"].map(team_name)
    data["refined_team"] = data["team"]
    for tid, group in data.groupby("track_id"):
        known = group.loc[group.team != "Unknown", "team"].value_counts()
        if known.empty or known.iloc[0] / known.sum() < 0.80:
            continue
        winner = known.index[0]
        data.loc[group.index[group.team == "Unknown"], "refined_team"] = winner
        if known.iloc[0] / known.sum() > .97 and known.iloc[0] >= 20:
            data.loc[group.index, "refined_team"] = winner
    return data


def appearance_signature(frame, box):
    """Lightweight player signature: jersey HSV/Lab, shorts HSV and spatial splits.
    Distinguishes some players, but same-kit teammates can still be ambiguous.
    """
    H, W = frame.shape[:2]
    x1, y1, x2, y2 = map(float, box)
    width, height = x2 - x1, y2 - y1
    if width < 7 or height < 15:
        return None
    regions = (
        (.27, .73, .20, .47),  # shirt
        (.27, .73, .47, .69),  # shorts
        (.27, .50, .22, .48),  # left shirt pattern
        (.50, .73, .22, .48),  # right shirt pattern
    )
    parts = []
    for xa, xb, ya, yb in regions:
        a = max(0, int(x1 + xa * width)); b = min(W, int(x1 + xb * width))
        c = max(0, int(y1 + ya * height)); d = min(H, int(y1 + yb * height))
        if b - a < 3 or d - c < 3:
            return None
        crop = cv2.resize(frame[c:d, a:b], (24, 24), interpolation=cv2.INTER_AREA)
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        lab = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB)
        h1 = cv2.calcHist([hsv], [0, 1], None, [18, 6], [0, 180, 0, 256])
        h2 = cv2.calcHist([lab], [1, 2], None, [12, 12], [0, 256, 0, 256])
        cv2.normalize(h1, h1, alpha=1, norm_type=cv2.NORM_L1)
        cv2.normalize(h2, h2, alpha=1, norm_type=cv2.NORM_L1)
        parts.extend((h1, h2))
    return parts


def signature_similarity(a, b):
    if a is None or b is None:
        return 0.0
    # Shirt gets strongest weight, but left/right pattern and shorts also count.
    weights = [0.23, 0.18, 0.12, 0.12, 0.08, 0.07, 0.10, 0.10]
    scores = [max(0.0, 1 - cv2.compareHist(x, y, cv2.HISTCMP_BHATTACHARYYA))
              for x, y in zip(a, b)]
    return float(np.dot(weights, scores))


def project_and_collect(data):
    """Single video pass does camera compensation, projection and cheap appearance samples."""
    info = json.loads(Path(S.CALIBRATION_JSON).read_text(encoding="utf-8"))
    H = np.asarray(info["homography"], dtype=np.float64)
    if H.shape != (3, 3) or not np.isfinite(H).all() or abs(np.linalg.det(H)) < 1e-12:
        raise ValueError("Invalid calibration homography")
    grouped = {int(f): g for f, g in data.groupby("frame", sort=False)}
    video = cv2.VideoCapture(S.VIDEO)
    if not video.isOpened():
        raise RuntimeError(f"Cannot read {S.VIDEO}")
    fps = float(video.get(cv2.CAP_PROP_FPS) or 25.0)
    observations = []
    # Up to 3 first and last descriptors per ID, not all video frames in RAM.
    samples = defaultdict(lambda: {"first": [], "last": deque(maxlen=3)})
    camera = None
    idx = 0
    try:
        while True:
            ok, frame = video.read()
            if not ok:
                break
            if camera is None:
                camera = CameraMotionCompensator(frame)
                frame_to_first = np.eye(3, dtype=np.float64)
            else:
                frame_to_first = camera.update(frame)
            group = grouped.get(idx)
            if group is not None:
                img_pts = group[["foot_x", "foot_y"]].to_numpy(np.float64).reshape(-1, 1, 2)
                calibrated = cv2.perspectiveTransform(
                    cv2.perspectiveTransform(img_pts, frame_to_first), H).reshape(-1, 2)
                for r, xy in zip(group.itertuples(index=False), calibrated):
                    x, y = map(float, xy)
                    if not np.isfinite([x, y]).all():
                        continue
                    if not (-S.POSITION_MARGIN_METRES <= x <= 105 + S.POSITION_MARGIN_METRES and
                            -S.POSITION_MARGIN_METRES <= y <= 68 + S.POSITION_MARGIN_METRES):
                        continue
                    tid = int(r.track_id)
                    row = {"frame": idx, "track_id": tid, "original_track_id": tid,
                           "class_name": str(r.class_name), "team": str(r.refined_team),
                           "original_team": str(r.team), "confidence": float(r.confidence),
                           "raw_x_m": x, "raw_y_m": y,
                           "x1": float(r.x1), "y1": float(r.y1),
                           "x2": float(r.x2), "y2": float(r.y2)}
                    observations.append(row)
                    sig = appearance_signature(frame, (r.x1, r.y1, r.x2, r.y2))
                    if sig is not None and (idx % 4 == 0 or not samples[tid]["first"]):
                        if len(samples[tid]["first"]) < 3:
                            samples[tid]["first"].append(sig)
                        samples[tid]["last"].append(sig)
            idx += 1
            if idx % 150 == 0:
                print(f"Projected {idx} frames")
    finally:
        video.release()
    return pd.DataFrame(observations), samples, fps, idx


def compare_sample_groups(left, right):
    if not left or not right:
        return 0.0
    vals = [signature_similarity(a, b) for a in left for b in right]
    return float(np.median(vals))


def reid_proposals(observed, samples, fps):
    """One-to-one Hungarian proposals. Only *review* is automatic; no silent merges."""
    tracks = {}
    for tid, group in observed.groupby("track_id"):
        g = group.sort_values("frame")
        if len(g) < S.MIN_TRACK_LENGTH_FOR_REID:
            continue
        tracks[int(tid)] = g
    ids = sorted(tracks)
    n = len(ids)
    if n == 0:
        return pd.DataFrame()
    scores = np.full((n, n), -1.0, dtype=float)
    evidence = {}
    for i, old in enumerate(ids):
        a = tracks[old]
        last = a.iloc[-1]
        av = a.tail(min(5, len(a)))[["raw_x_m", "raw_y_m"]].to_numpy()
        aft = a.tail(min(5, len(a)))["frame"].to_numpy()
        vel = (av[-1] - av[0]) / max((aft[-1] - aft[0]) / fps, 1 / fps)
        speed = np.linalg.norm(vel)
        if speed > S.MAX_REID_SPEED_MPS * 1.5:
            vel *= (S.MAX_REID_SPEED_MPS * 1.5 / speed)
        for j, new in enumerate(ids):
            if old == new:
                continue
            b = tracks[new]
            first = b.iloc[0]
            gap = int(first.frame - last.frame)
            if gap < 1 or gap > S.MAX_REID_GAP_FRAMES:
                continue
            if last.team == "Unknown" or first.team == "Unknown" or last.team != first.team:
                continue
            dt = gap / fps
            departure = np.array([last.raw_x_m, last.raw_y_m])
            arrival = np.array([first.raw_x_m, first.raw_y_m])
            straight = float(np.linalg.norm(arrival - departure))
            maximum = min(S.MAX_REID_DISTANCE_METRES, 1.3 + S.MAX_REID_SPEED_MPS * dt)
            if straight > maximum:
                continue
            predicted = departure + vel * dt
            residual = float(np.linalg.norm(arrival - predicted))
            if residual > maximum:
                continue
            app = compare_sample_groups(samples[old]["last"], samples[new]["first"])
            if app < S.MIN_REID_APPEARANCE:
                continue
            # Score combines unique motion/appearance; avoid exact-team-color-only matches.
            distance_score = max(0., 1. - residual / max(maximum, 1e-6))
            time_score = max(0., 1. - gap / (S.MAX_REID_GAP_FRAMES + 1))
            score = 0.54 * app + 0.38 * distance_score + 0.08 * time_score
            scores[i, j] = score
            evidence[i, j] = (old, new, gap, straight, residual, app, score, last, first)
    # Unmatched dummies stop assignment from forcing bad pairings.
    size = n * 2
    costs = np.full((size, size), 1.0 - S.MIN_REID_SCORE, dtype=float)
    costs[:n, :n] = np.where(scores >= 0, 1 - scores, 1000)
    ri, cj = linear_sum_assignment(costs)
    output = []
    for i, j in zip(ri, cj):
        if i >= n or j >= n or (i, j) not in evidence:
            continue
        old, new, gap, distance, residual, app, score, last, first = evidence[i, j]
        if score < S.MIN_REID_SCORE:
            continue
        rivals = [float(v) for k, v in enumerate(scores[i]) if k != j and v >= 0]
        rivals += [float(v) for k, v in enumerate(scores[:, j]) if k != i and v >= 0]
        next_score = max(rivals) if rivals else 0
        margin = score - next_score
        output.append({
            "old_track_id": old, "new_track_id": new,
            "old_last_frame": int(last.frame), "new_first_frame": int(first.frame),
            "gap_frames": gap, "distance_m": round(distance, 3),
            "predicted_residual_m": round(residual, 3),
            "appearance_similarity": round(app, 4),
            "score": round(score, 4), "margin_to_competitor": round(margin, 4),
            "team": first.team, "review_status": "suggested" if margin >= S.MIN_REID_MARGIN else "ambiguous",
            "old_bbox": json.dumps([last.x1, last.y1, last.x2, last.y2]),
            "new_bbox": json.dumps([first.x1, first.y1, first.x2, first.y2]),
        })
    names = ["old_track_id", "new_track_id", "old_last_frame", "new_first_frame",
             "gap_frames", "distance_m", "predicted_residual_m", "appearance_similarity",
             "score", "margin_to_competitor", "team", "review_status", "old_bbox", "new_bbox"]
    return pd.DataFrame(output, columns=names).sort_values("score", ascending=False)


def load_approvals(proposals):
    """Only user-reviewed approved rows are actionable. Automatic option disabled."""
    path = Path(S.APPROVALS_CSV)
    approvals = {}
    if path.is_file():
        approved = pd.read_csv(path)
        if not {"old_track_id", "new_track_id", "approved"}.issubset(approved.columns):
            raise ValueError("Approvals CSV must contain old_track_id,new_track_id,approved")
        valid = {(int(r.old_track_id), int(r.new_track_id))
                 for r in proposals.itertuples(index=False)}
        for row in approved.itertuples(index=False):
            old, new = int(row.old_track_id), int(row.new_track_id)
            if str(row.approved).strip().lower() in ("1", "true", "yes", "y"):
                if (old, new) not in valid:
                    raise ValueError(f"Approved IDs {old}->{new} are not in this run's proposals")
                approvals[new] = old
    elif S.AUTO_APPROVE_ID_MERGES:
        # Explicit opt-in only, and still constrained by competing matches.
        for row in proposals.itertuples(index=False):
            if row.review_status == "suggested" and row.score >= .90:
                approvals[int(row.new_track_id)] = int(row.old_track_id)
    return approvals


def apply_approvals(observed, mapping):
    if not mapping:
        return observed
    output = observed.copy()
    def root(tid):
        seen = set()
        while tid in mapping:
            if tid in seen:
                raise ValueError("Cycle in ID approvals")
            seen.add(tid)
            tid = mapping[tid]
        return tid
    output["track_id"] = output.track_id.map(root)
    if output.duplicated(["frame", "track_id"]).any():
        raise ValueError("Approved ID merge produces simultaneous duplicate IDs. Revoke approval.")
    return output


def rts_segment(rows, fps):
    """Constant-velocity Kalman filtering then Rauch–Tung–Striebel smoothing.
    Full-clip offline smoother uses *future* observations, unlike live Kalman.
    """
    by_frame = {int(r["frame"]): r for r in rows}
    start, end = int(rows[0]["frame"]), int(rows[-1]["frame"])
    frames = list(range(start, end + 1))
    dt = 1.0 / fps
    F = np.array([[1, 0, dt, 0], [0, 1, 0, dt], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=float)
    # Discrete white acceleration noise.
    g = np.array([[dt * dt / 2, 0], [0, dt * dt / 2], [dt, 0], [0, dt]])
    Q = (g @ g.T) * S.RTS_PROCESS_ACCEL_NOISE ** 2 + 1e-8 * np.eye(4)
    M = np.array([[1., 0, 0, 0], [0, 1., 0, 0]])
    I = np.eye(4)
    x = np.array([rows[0]["raw_x_m"], rows[0]["raw_y_m"], 0., 0.])
    P = np.diag([1., 1., 16., 16.])
    filtered, covariance, prior_state, prior_cov, statuses = [], [], [], [], []
    last_accepted = start
    for f in frames:
        if f != start:
            x = F @ x; P = F @ P @ F.T + Q
        xp, pp = x.copy(), P.copy()
        row = by_frame.get(f)
        accepted = False
        if row is not None:
            z = np.array([row["raw_x_m"], row["raw_y_m"]])
            # Reject only extreme coordinate spikes, not normal rapid turns.
            if f == start or np.linalg.norm(z - xp[:2]) <= max(3.5, S.MAX_PLAUSIBLE_PLAYER_MPS * (f - last_accepted) / fps + 1.5):
                noise = S.MEASUREMENT_NOISE_METRES / max(.4, float(row["confidence"]))
                R = np.eye(2) * noise ** 2
                innovation = z - M @ x
                V = M @ P @ M.T + R
                K = P @ M.T @ np.linalg.inv(V)
                x = x + K @ innovation
                P = (I - K @ M) @ P @ (I - K @ M).T + K @ R @ K.T
                accepted = True
                last_accepted = f
        statuses.append("observed_smoothed" if accepted else "interpolated_same_id" if row is None else "rejected_outlier_predicted")
        filtered.append(x.copy()); covariance.append(P.copy())
        prior_state.append(xp); prior_cov.append(pp)
    smooth = np.stack(filtered)
    for i in range(len(frames) - 2, -1, -1):
        J = covariance[i] @ F.T @ np.linalg.pinv(prior_cov[i+1])
        smooth[i] = filtered[i] + J @ (smooth[i+1] - prior_state[i+1])
    return frames, smooth, statuses, by_frame


def refine_trajectories(observed, fps):
    results = []
    for tid, g in observed.groupby("track_id", sort=True):
        rows = g.sort_values("frame").drop_duplicates("frame").to_dict("records")
        if not rows:
            continue
        # Prevent unconstrained interpolation over long camera absences.
        segments, start = [], 0
        for i in range(1, len(rows)):
            if rows[i]["frame"] - rows[i-1]["frame"] - 1 > S.MAX_BRIDGE_MISSING_FRAMES:
                segments.append(rows[start:i]); start = i
        segments.append(rows[start:])
        dominant = Counter(r["team"] for r in rows if r["team"] != "Unknown")
        fallback_team = dominant.most_common(1)[0][0] if dominant else "Unknown"
        for seg in segments:
            fnums, estimates, statuses, source = rts_segment(seg, fps)
            for frame, state, status in zip(fnums, estimates, statuses):
                r = source.get(frame)
                reference = r if r is not None else seg[0]
                team = (r["team"] if r is not None else fallback_team)
                if team == "Unknown" and fallback_team != "Unknown":
                    team = fallback_team
                x, y = float(state[0]), float(state[1])
                if not (-S.POSITION_MARGIN_METRES <= x <= 105 + S.POSITION_MARGIN_METRES and
                        -S.POSITION_MARGIN_METRES <= y <= 68 + S.POSITION_MARGIN_METRES):
                    continue
                real = status == "observed_smoothed"
                results.append({
                    "frame": frame, "track_id": int(tid),
                    "original_track_id": int(r["original_track_id"] if r else reference["original_track_id"]),
                    "class_name": reference["class_name"], "team": team,
                    "pitch_x_m": x, "pitch_y_m": y,
                    "raw_x_m": float(r["raw_x_m"]) if r else np.nan,
                    "raw_y_m": float(r["raw_y_m"]) if r else np.nan,
                    "confidence": float(r["confidence"]) if r else np.nan,
                    "velocity_x_mps": float(state[2]), "velocity_y_mps": float(state[3]),
                    "position_source": status, "is_observed": int(real),
                })
        # Frame 0 backfill only if first observation is at frame 1.
        if rows[0]["frame"] == 1:
            base = next((r for r in results if r["track_id"] == int(tid) and r["frame"] == 1), None)
            if base:
                copied = dict(base); copied["frame"] = 0
                copied["raw_x_m"] = np.nan; copied["raw_y_m"] = np.nan
                copied["is_observed"] = 0; copied["position_source"] = "held_first_frame"
                results.append(copied)
    out = pd.DataFrame(results)
    if out.empty:
        return out
    out["timestamp_seconds"] = out.frame / fps
    return out.sort_values(["frame", "track_id"]).reset_index(drop=True)


def run():
    check_inputs()
    data = pd.read_csv(S.FIRST_PASS_CSV)
    needed = {"frame", "track_id", "foot_x", "foot_y", "x1", "y1", "x2", "y2", "confidence", "team", "class_name"}
    if not needed.issubset(data.columns):
        raise ValueError(f"Source CSV is missing columns: {sorted(needed - set(data.columns))}")
    data = stable_teams(data)
    print("Pass 1: projection, movement compensation and appearance signatures")
    observed, samples, fps, frames = project_and_collect(data)
    if observed.empty:
        raise RuntimeError("No usable projected observations; re-check pitch calibration")
    print("Pass 2: finding candidate tracklet associations")
    proposals = reid_proposals(observed, samples, fps)
    Path(S.PROPOSALS_CSV).parent.mkdir(parents=True, exist_ok=True)
    proposals.to_csv(S.PROPOSALS_CSV, index=False)
    mapping = load_approvals(proposals)
    print(f"Candidate pairs: {len(proposals)} | Approved merges applied: {len(mapping)}")
    observed = apply_approvals(observed, mapping)
    print("Pass 3: forward/backward Kalman (RTS) smoothing")
    out = refine_trajectories(observed, fps)
    Path(S.OUT_CSV).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(S.OUT_CSV, index=False, float_format="%.4f")
    print(f"Video: {frames} frames at {fps:.2f} FPS")
    print(f"Observed pitch positions: {len(observed)}")
    print(f"Output positions: {len(out)} | estimated: {int((out.is_observed == 0).sum())}")
    print(f"ID suggestions: {S.PROPOSALS_CSV}")
    print(f"Coordinates: {S.OUT_CSV}")
    print("Estimated positions are labeled, and should not be used as real detections.")


if __name__ == "__main__":
    run()
