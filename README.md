# Sports Video Tracking & Tactical Map Generation


## 1. Overview

This project implements a computer vision pipeline to detect, track, and visualize key entities in volleyball match videos and project them onto a 2D Hawk-Eye–style tactical map.

The system processes all provided input videos from the volleyball category and generates:

- An annotated video with bounding boxes and IDs
- A frame-synchronized 2D tactical map showing entity positions over time

Tracked entities include:
- Players
- Ball
- Referee / Umpire (heuristic-based)

## 2. Input Data

- Source: Videos provided by FutureSportler (Volleyball category)

- Number of videos: 5

- Video characteristics:
  - Mixed camera behavior (fixed and moving)
  - Different court orientations (North–South and East–West)
  - Presence of extra players and audience near court boundaries
  - Small, fast-moving ball with occasional occlusion

## 3. Approach
### 3.1 Detection

- Used YOLOv8 (Ultralytics) for object detection.
- The COCO person class is used as the base for player and referee detection.
- The sports ball class is used for ball detection when visible.
- Confidence and IoU thresholds are applied to reduce false positives.

### 3.2 Playable Region Filtering

To improve accuracy and reduce detection of non-relevant people:
- A dynamic playable-region filter is applied.
- Detections outside the court region (benches, audience, substitutes) are discarded.
- This significantly reduces false player detections.

### 3.3 Multi-Object Tracking

- YOLOv8’s built-in tracking (persist=True) is used to maintain stable IDs across frames.
- Each detected entity is assigned a consistent ID throughout its visible duration.

### 3.4 Team Classification

- Team assignment is based on spatial positioning:

Players on one side of the court are assigned Team A

Players on the opposite side are assigned Team B

Team labels are locked per track ID to prevent flickering.

Jersey color is intentionally not used, as some players wear different colors within the same team.

3.5 Tactical Map Generation

Player, ball, and referee positions are normalized relative to frame dimensions.

A 2D top-down court representation is rendered for each frame.

Entities are plotted with distinct visual encodings:

Red dots: Team A players

Blue dots: Team B players

Yellow dot: Ball (when detected)

Green dot: Referee / officials

The tactical map is frame-by-frame synchronized with the annotated video.

4. Output

For each input video, the system produces:

Annotated Video

Bounding boxes

Entity IDs

Team-based coloring

Tactical Map Video

2D top-down visualization

Synchronized entity motion

Court center line for orientation

All output videos are saved in the output/ directory and shared via Google Drive as required.

5. Assumptions

Only one volleyball court is present per video.

Players remain largely within the playable court region.

Team separation can be approximated using court-side positioning.

Referee/umpire appearance is visually similar to players and must be inferred heuristically.

Ball visibility depends on resolution and camera angle.

6. Limitations (Important)

This system is designed as a practical engineering solution, not a production-grade Hawk-Eye system.

Known limitations include:

Ball Detection

The volleyball is small and fast-moving.

Detection is intermittent when the ball is occluded or blurred.

Ball trajectory smoothing is limited without a specialized trained model.

Referee / Umpire Detection

No dedicated referee class exists in the base detection model.

Referee detection is heuristic-based and may occasionally misclassify nearby players.

Camera Motion

Videos with moving or zooming cameras affect spatial consistency.

Homography is approximated using normalized coordinates rather than full court keypoint calibration.

Tactical Map Accuracy

The map is an approximate projection, not an exact metric reconstruction.

Orientation differences across videos may affect perceived alignment.

7. Future Improvements

Given additional time and data, the following improvements would significantly increase accuracy:

Fine-tuning a ball-specific detection model

Dedicated court keypoint detection for accurate homography

Advanced tracking with Kalman filtering / ByteTrack

Referee classification using motion and positional priors

Event detection (serve, spike, rally transitions)

## 8. How to Run
 Activate virtual environment
venv\Scripts\activate

 Install dependencies
pip install -r requirements.txt

Run pipeline
python src/main.py

## 9. Conclusion

This project demonstrates an end-to-end computer vision pipeline for sports analytics, balancing accuracy, robustness, and practical constraints.
While not a replacement for professional Hawk-Eye systems, it provides a clear, extensible foundation aligned with real-world sports CV challenges.
