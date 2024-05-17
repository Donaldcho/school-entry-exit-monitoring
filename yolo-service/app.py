import os
import cv2
import time
import logging
from flask import Flask, Response
from deep_sort_realtime.deepsort_tracker import DeepSort
from yolo_utils import load_yolo_model, handle_camera_disconnection, publish_message, generate_embeddings, create_rabbitmq_channel, initialize_resources, release_resources
from config import CONFIG
from feature_extractor import load_feature_extractor

app = Flask(__name__)

# Logging configuration
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# YOLOv5 model loading
model = load_yolo_model(CONFIG["confidence_threshold"])

# Feature extractor model loading (ResNet)
feature_extractor = load_feature_extractor()

# DeepSORT tracker initialization
deepsort_tracker = DeepSort(max_age=CONFIG["max_age"], n_init=CONFIG["n_init"], nms_max_overlap=1.0, max_cosine_distance=CONFIG["max_cosine_distance"])

def generate_frames():
    """
    Generates video frames with object detection and tracking.
    Captures video frames, runs YOLOv5 for object detection, uses DeepSORT for tracking,
    and publishes messages to RabbitMQ. Encodes and streams video frames with detected bounding boxes.
    """
    cap, connection, channel = initialize_resources()
    tracked_ids = set()

    while True:
        ret, frame = cap.read()
        if not ret:
            logger.warning("Camera disconnected. Reconnecting...")
            cap = handle_camera_disconnection(cap, CONFIG["camera_index"])
            continue

        # YOLOv5 inference
        results = model(frame)

        # Get bounding boxes and confidences
        detections = []
        crops = []
        for det in results.xyxy[0].cpu().numpy():
            x1, y1, x2, y2, conf, cls = det
            if int(cls) == 0 and conf > CONFIG["confidence_threshold"]:
                x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
                width, height = x2 - x1, y2 - y1
                detections.append(([x1, y1, width, height], conf, 'person'))
                crops.append(frame[y1:y2, x1:x2])

        # Generate embeddings for detections
        embeddings = generate_embeddings(crops, feature_extractor) if crops else []

        # DeepSORT tracking with embeddings
        tracks = deepsort_tracker.update_tracks(detections, embeds=embeddings, frame=frame)

        # Visualize and send messages
        for track in tracks:
            if not track.is_confirmed() or track.time_since_update > 1:
                continue
            bbox = track.to_tlbr()
            cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (0, 255, 0), 2)
            cv2.putText(frame, str(track.track_id), (int(bbox[0]), int(bbox[1] - 10)), 0, 0.75, (0, 255, 0), 2)

            if track.time_since_update == 0 and track.is_confirmed() and track.track_id not in tracked_ids:
                message = {
                    'track_id': track.track_id,
                    'bbox': [int(x) for x in bbox],
                    'timestamp': time.time(),
                }
                channel = publish_message(channel, message, CONFIG["rabbitmq_queue"], CONFIG["max_reconnect_attempts"])
                tracked_ids.add(track.track_id)

        # Encode and yield frame
        _, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    release_resources(cap, connection)

@app.route('/video_feed')
def video_feed():
    """
    Route to provide the video feed.
    """
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    try:
        app.run(host='0.0.0.0', port=5000, debug=True)
    except KeyboardInterrupt:
        logger.info("Server stopped by user.")
