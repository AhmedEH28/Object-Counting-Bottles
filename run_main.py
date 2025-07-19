"""
Advanced Object Counting for Bottles with YOLOv8
Tracks and counts Juice, Water, CocaCola bottles with enhanced accuracy
Includes error handling, confidence thresholds, and multiple counting zones
"""

import cv2
import numpy as np
from collections import defaultdict, deque
from ultralytics import YOLO
import json
import time
from pathlib import Path

class AdvancedBottleCounter:
    def __init__(self, model_path, classes_to_count, class_names, confidence=0.5, iou=0.45):
        self.model = YOLO(model_path)
        self.classes_to_count = classes_to_count
        self.class_names = class_names
        self.confidence = confidence
        self.iou = iou

        self.track_history = defaultdict(lambda: deque(maxlen=10))
        self.counted_tracks = set()
        self.class_counts = {cls_id: {'IN': 0, 'OUT': 0} for cls_id in classes_to_count}
        self.counting_line = None

        self.fps_counter = deque(maxlen=30)
        self.frame_count = 0

    def set_counting_line(self, point1, point2):
        self.counting_line = (point1, point2)

    def is_crossing_line(self, track_id, current_center):
        if self.counting_line is None or track_id not in self.track_history:
            return None
        if len(self.track_history[track_id]) < 2:
            return None
        prev_center = self.track_history[track_id][-2]
        line_x = self.counting_line[0][0]
        prev_x, curr_x = prev_center[0], current_center[0]
        if prev_x < line_x <= curr_x:
            return 'IN'
        elif prev_x > line_x >= curr_x:
            return 'OUT'
        return None

    def draw_counting_line(self, frame):
        if self.counting_line:
            cv2.line(frame, self.counting_line[0], self.counting_line[1], (0, 255, 0), 3)

    def draw_info_panel(self, frame):
        # Info Panel
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (360, 180), (40, 40, 40), -1)
        frame = cv2.addWeighted(frame, 0.8, overlay, 0.2, 0)
        cv2.rectangle(frame, (10, 10), (360, 180), (0, 255, 0), 2)
        cv2.putText(frame, "BOTTLE COUNTER", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        y_offset = 60
        for cls_id, counts in self.class_counts.items():
            cls_name = self.class_names.get(cls_id, f"Class {cls_id}")
            text = f"{cls_name}: {counts['IN']} IN / {counts['OUT']} OUT"
            color = {0: (255, 100, 100), 1: (100, 255, 100), 2: (100, 100, 255)}.get(cls_id, (255, 255, 255))
            cv2.putText(frame, text, (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            y_offset += 25

    def draw_live_count(self, frame):
        # Live count at bottom left corner
        live_counts = []
        for cls_id in self.classes_to_count:
            live_counts.append(f"{self.class_names[cls_id]}: {self.class_counts[cls_id]['IN']}")
        
        # Position at bottom left
        y_start = frame.shape[0] - 100  # Start 100 pixels from bottom
        
        for i, count_text in enumerate(live_counts):
            y_pos = y_start + (i * 25)  # 25 pixels between each line
            
            # Get text size for background rectangle
            (text_w, text_h), _ = cv2.getTextSize(count_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            
            # Draw black background rectangle
            cv2.rectangle(frame, (10, y_pos - text_h - 5), (20 + text_w, y_pos + 5), (0, 0, 0), -1)
            
            # Draw text
            color = {0: (255, 100, 100), 1: (100, 255, 100), 2: (100, 100, 255)}.get(self.classes_to_count[i], (255, 255, 255))
            cv2.putText(frame, count_text, (15, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        if self.fps_counter:
            avg_fps = 1.0 / (sum(self.fps_counter) / len(self.fps_counter))
            cv2.putText(frame, f"FPS: {avg_fps:.1f}", (20, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)

    def process_frame(self, frame):
        start_time = time.time()
        results = self.model.track(
            frame,
            conf=self.confidence,
            iou=self.iou,
            classes=self.classes_to_count,
            persist=True,
            verbose=False
        )
        if results and results[0].boxes is not None and results[0].boxes.id is not None:
            boxes = results[0].boxes.xywh.cpu().numpy()
            track_ids = results[0].boxes.id.int().cpu().numpy()
            confidences = results[0].boxes.conf.cpu().numpy()
            class_ids = results[0].boxes.cls.int().cpu().numpy()

            for box, track_id, conf, cls_id in zip(boxes, track_ids, confidences, class_ids):
                if cls_id in self.classes_to_count:
                    center_x, center_y = int(box[0]), int(box[1])
                    center = (center_x, center_y)
                    self.track_history[track_id].append(center)
                    if track_id not in self.counted_tracks:
                        crossing = self.is_crossing_line(track_id, center)
                        if crossing == 'IN':
                            self.class_counts[cls_id][crossing] += 1
                            self.counted_tracks.add(track_id)

                    x1 = int(box[0] - box[2]/2)
                    y1 = int(box[1] - box[3]/2)
                    x2 = int(box[0] + box[2]/2)
                    y2 = int(box[1] + box[3]/2)
                    color = {0: (255, 100, 100), 1: (100, 255, 100), 2: (100, 100, 255)}.get(cls_id, (0, 255, 255))
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
                    label = f"{self.class_names[cls_id]} #{track_id}"
                    cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        self.draw_counting_line(frame)
        self.draw_info_panel(frame)
        self.draw_live_count(frame)
        self.fps_counter.append(time.time() - start_time)
        self.frame_count += 1
        return frame

    def save_results(self, output_path):
        results = {
            'frames': self.frame_count,
            'class_counts': self.class_counts
        }
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)


def main():
    MODEL_PATH = "best.pt"
    VIDEO_INPUT = "video3.mp4"
    VIDEO_OUTPUT = "video1R.mp4"
    RESULTS_JSON = "counting_results.json"

    REGION_POINTS = [(1100, 50), (1100, 700)]
    CLASSES_TO_COUNT = [0, 1, 2]
    CLASS_NAMES = {0: 'Juice', 1: 'Water', 2: 'CocaCola'}

    counter = AdvancedBottleCounter(MODEL_PATH, CLASSES_TO_COUNT, CLASS_NAMES)
    counter.set_counting_line(*REGION_POINTS)

    cap = cv2.VideoCapture(VIDEO_INPUT)
    if not cap.isOpened():
        print("❌ Error opening video")
        return

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    writer = cv2.VideoWriter(VIDEO_OUTPUT, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        processed = counter.process_frame(frame)
        writer.write(processed)

    cap.release()
    writer.release()
    counter.save_results(RESULTS_JSON)
    print("✅ Processing complete.")

if __name__ == "__main__":
    main()