"""
Person detection + tracking (YOLOv8 + ByteTrack).

Usage:
    python track.py --source my_video.mp4
    python track.py --source my_video.mp4 --tracker botsort.yaml

Outputs (in ./output):
    tracked.mp4            video with boxes + ID labels
    tracks.csv             frame, id, x1, y1, x2, y2
    summary.txt            counts + ID-switch candidates
    switch_candidates/     before/after images for each candidate switch
"""
import argparse
import csv
from pathlib import Path

import cv2
from ultralytics import YOLO


def centre(b):
    return ((b[0] + b[2]) / 2, (b[1] + b[3]) / 2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True, help="video path (or 0 for webcam)")
    ap.add_argument("--model", default="yolov8n.pt")
    ap.add_argument("--tracker", default="bytetrack.yaml",
                    help="bytetrack.yaml or botsort.yaml")
    ap.add_argument("--conf", type=float, default=0.3)
    ap.add_argument("--out", default="output")
    ap.add_argument("--max-gap", type=int, default=45,
                    help="frames: how far back to look for a lost ID when a new one appears")
    ap.add_argument("--min-frames", type=int, default=15,
                    help="IDs seen fewer frames than this count as 'short/unstable'")
    args = ap.parse_args()

    out_dir = Path(args.out)
    (out_dir / "switch_candidates").mkdir(parents=True, exist_ok=True)

    src = 0 if args.source == "0" else args.source
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        raise SystemExit(f"Could not open {args.source}")
    w, h = int(cap.get(3)), int(cap.get(4))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    video_path = out_dir / "tracked.mp4"
    writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    model = YOLO(args.model)

    stats = {}            # id -> dict(first, last, first_box, last_box, n)
    candidates = []       # (frame, new_id, old_id)
    claimed_old = set()
    rows = []
    frame_idx = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        res = model.track(frame, persist=True, classes=[0], conf=args.conf,
                          tracker=args.tracker, verbose=False)[0]

        if res.boxes.id is not None:
            ids = res.boxes.id.int().tolist()
            boxes = res.boxes.xyxy.int().tolist()
            visible_now = set(ids)

            for tid, box in zip(ids, boxes):
                if tid not in stats:
                    # A brand-new ID. Was some other ID lost very recently, nearby?
                    if frame_idx > 0:
                        for oid, s in stats.items():
                            if oid in visible_now or oid in claimed_old:
                                continue
                            gap = frame_idx - s["last"]
                            if not (1 <= gap <= args.max_gap):
                                continue
                            ox, oy = centre(s["last_box"])
                            nx, ny = centre(box)
                            dist = ((ox - nx) ** 2 + (oy - ny) ** 2) ** 0.5
                            size = max(s["last_box"][2] - s["last_box"][0],
                                       s["last_box"][3] - s["last_box"][1])
                            if dist < 1.5 * size:
                                candidates.append((frame_idx, tid, oid))
                                claimed_old.add(oid)
                                break
                    stats[tid] = dict(first=frame_idx, last=frame_idx,
                                      first_box=box, last_box=box, n=1)
                else:
                    s = stats[tid]
                    s["last"], s["last_box"] = frame_idx, box
                    s["n"] += 1

                rows.append([frame_idx, tid, *box])
                x1, y1, x2, y2 = box
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"ID {tid}", (x1, max(y1 - 8, 15)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        cv2.putText(frame, f"frame {frame_idx} | unique IDs so far: {len(stats)}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        writer.write(frame)
        frame_idx += 1

    cap.release()
    writer.release()

    with open(out_dir / "tracks.csv", "w", newline="") as f:
        csv.writer(f).writerows([["frame", "id", "x1", "y1", "x2", "y2"], *rows])

    # Save before/after images for each candidate switch
    cap = cv2.VideoCapture(str(video_path))
    for fr, new_id, old_id in candidates:
        pair = []
        for idx in (max(fr - 3, 0), fr):
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ok, img = cap.read()
            if ok:
                pair.append(img)
        if len(pair) == 2:
            cv2.imwrite(str(out_dir / "switch_candidates" /
                            f"frame{fr}_ID{old_id}_to_ID{new_id}.jpg"),
                        cv2.hconcat(pair))
    cap.release()

    stable = [i for i, s in stats.items() if s["n"] >= args.min_frames]
    lines = [
        f"Video: {args.source}  ({frame_idx} frames, {fps:.1f} fps)",
        f"Tracker: {args.tracker}  | model: {args.model}  | conf: {args.conf}",
        "",
        f"Unique IDs assigned by tracker: {len(stats)}",
        f"IDs seen >= {args.min_frames} frames (stable): {len(stable)}",
        f"Short-lived IDs (likely fragments/false positives): {len(stats) - len(stable)}",
        "",
        "Possible ID switches (old ID lost, new ID appeared nearby shortly after):",
    ]
    if candidates:
        for fr, new_id, old_id in candidates:
            lines.append(f"  frame {fr}: ID {old_id} -> ID {new_id}")
    else:
        lines.append("  none auto-detected; scrub the video manually")
    summary = "\n".join(lines)
    (out_dir / "summary.txt").write_text(summary)
    print(summary)
    print(f"\nSaved video to {video_path}")


if __name__ == "__main__":
    main()
