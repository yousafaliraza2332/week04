# Person Tracking: YOLOv8 + ByteTrack

Detects people in a video, tracks them across frames, and draws each person's ID on the output video.

## Run

```bash
pip install -r requirements.txt
python track.py --source my_video.mp4
```

Outputs go to `output/` (`tracked.mp4`, `tracks.csv`, `summary.txt`, `switch_candidates/`).

## Setup
- Detector: YOLOv8n (COCO, class 0 = person)
- Tracker: ByteTrack (Ultralytics built-in)
- Video: <describe your video: length, scene, how many people, camera moving or fixed>

## Results: unique people count

| Method | Count |
|---|---|
| Manual count (me watching the video) | **<N>** |
| Tracker unique IDs | **<N>** |
| Tracker IDs seen >= 15 frames | **<N>** |

### Why they differ
<Fill in using what you observed. Common reasons:>
- Someone was hidden behind another person/object, the track was lost, and they came back with a new ID (fragmentation), which makes the tracker count higher.
- Someone left the frame and re-entered, so they got a new ID.
- A brief false detection (poster, reflection, partial body) created a short-lived extra ID.
- (If tracker count is lower) two people overlapped and were merged, or the detector missed someone.

## ID switch

- **Frame:** <number>
- **What happened:** Person with ID <A> became ID <B>.
- **Why I think it happened:** <e.g. they were occluded by ___ for ~N frames, the track expired, and a new ID was created when they reappeared / two people crossed paths with similar clothing / detection confidence dropped when they turned away>

Screenshot (before | after): `output/switch_candidates/<file>.jpg`

![ID switch](output/switch_candidates/<file>.jpg)

## Possible improvements
- Try `--tracker botsort.yaml` (adds camera-motion compensation and optional ReID)
- Use a larger model (`yolov8s.pt` / `yolov8m.pt`) for fewer missed detections
- Increase the tracker's `track_buffer` so lost tracks survive longer occlusions
