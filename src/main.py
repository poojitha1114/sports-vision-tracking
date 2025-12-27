import cv2
import os
from ultralytics import YOLO
from tactical_map import generate_tactical_map

DATA_DIR = "data"
OUTPUT_TRACKED_DIR = "output/tracked"
OUTPUT_TACTICAL_DIR = "output/tactical"

os.makedirs(OUTPUT_TRACKED_DIR, exist_ok=True)
os.makedirs(OUTPUT_TACTICAL_DIR, exist_ok=True)

# Load YOLOv8 model
model = YOLO("yolov8n.pt")

# Map COCO classes to our entities
CLASS_MAP = {
    0: "player",        # person
    32: "ball",         # sports ball
    1: "referee"        # assume some COCO class for referee/umpire (adjust if needed)
}

def process_video(video_path):
    video_name = os.path.basename(video_path)
    print(f"\nProcessing {video_name} ...")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"ERROR: Cannot open {video_path}")
        return

    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    tracked_output_path = os.path.join(
        OUTPUT_TRACKED_DIR,
        video_name.replace(".mp4", "_tracked.mp4")
    )

    tactical_output_path = os.path.join(
        OUTPUT_TACTICAL_DIR,
        video_name.replace(".mp4", "_tactical.mp4")
    )

    writer = cv2.VideoWriter(
        tracked_output_path,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height)
    )

    tracks_per_frame = {}
    frame_id = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # YOLO tracking
        results = model.track(
            frame,
            persist=True,
            conf=0.4,
            iou=0.5
        )

        frame_tracks = []

        if results and results[0].boxes is not None:
            boxes = results[0].boxes

            for box in boxes:
                cls_id = int(box.cls[0])
                entity_class = CLASS_MAP.get(cls_id, None)
                if entity_class is None:
                    continue  # skip unknown classes

                track_id = int(box.id[0]) if box.id is not None else None

                x1, y1, x2, y2 = box.xyxy[0].tolist()
                cx = (x1 + x2) / 2
                cy = (y1 + y2) / 2

                # Normalize coordinates for tactical map
                court_x = cx / width
                court_y = cy / height

                # Team heuristic (players only)
                team = None
                if entity_class == "player":
                    team = 0 if cx < width / 2 else 1

                frame_tracks.append({
                    "id": track_id,
                    "class": entity_class,
                    "team": team,
                    "x": court_x,
                    "y": court_y
                })

                # Draw bounding box on annotated video
                color = (0, 0, 255) if entity_class == "player" and team == 0 else \
                        (255, 0, 0) if entity_class == "player" else \
                        (0, 255, 255) if entity_class == "ball" else \
                        (0, 255, 0)
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)

        tracks_per_frame[frame_id] = frame_tracks
        writer.write(frame)

        frame_id += 1
        if frame_id % 50 == 0:
            print(f"{video_name}: processed {frame_id} frames")

    cap.release()
    writer.release()
    print(f"Finished {video_name} | Total frames: {frame_id}")

    # Generate tactical map video
    generate_tactical_map(
        tracks_per_frame=tracks_per_frame,
        output_path=tactical_output_path,
        fps=fps
    )


if __name__ == "__main__":
    print("Scanning data folder for videos...\n")

    videos = [
        os.path.join(DATA_DIR, f)
        for f in os.listdir(DATA_DIR)
        if f.lower().endswith(".mp4")
    ]

    for video in videos:
        process_video(video)
