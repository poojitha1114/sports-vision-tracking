import cv2
import numpy as np

def generate_tactical_map(tracks_per_frame, output_path, fps=30, map_width=640, map_height=360):
    """
    Generates a top-down 2D tactical map video with court lines, team dots, ball, and referee.

    Args:
        tracks_per_frame (dict): Dictionary with frame_id as key and list of tracked entities as value.
                                 Each entity is a dict with keys: id, class, team, x, y
        output_path (str): Path to save the tactical map video.
        fps (int): Frames per second for output video.
        map_width (int): Width of the tactical map video.
        map_height (int): Height of the tactical map video.
    """

    # Create VideoWriter
    writer = cv2.VideoWriter(
        output_path,
        cv2.VideoWriter_fourcc(*'mp4v'),
        fps,
        (map_width, map_height)
    )

    # Precompute center line
    center_line_x = map_width // 2

    for frame_id in sorted(tracks_per_frame.keys()):
        tactical_frame = np.zeros((map_height, map_width, 3), dtype=np.uint8)

        # Draw center line
        cv2.line(tactical_frame, (center_line_x, 0), (center_line_x, map_height), (255, 255, 255), 2)

        for entity in tracks_per_frame[frame_id]:
            # Map normalized coordinates to map size
            x = int(entity["x"] * map_width)
            y = int(entity["y"] * map_height)

            # Determine color and radius
            if entity["class"] == "player":
                color = (0, 0, 255) if entity["team"] == 0 else (255, 0, 0)
                radius = 6
            elif entity["class"] == "ball":
                color = (0, 255, 255)  # Yellow
                radius = 8
            elif entity["class"] == "referee":
                color = (0, 255, 0)  # Green
                radius = 7
            else:
                color = (128, 128, 128)
                radius = 5

            # Draw the entity
            cv2.circle(tactical_frame, (x, y), radius, color, -1)

        writer.write(tactical_frame)

    writer.release()
    print(f"Tactical map video saved to {output_path}")
