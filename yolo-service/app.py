import os
import cv2
import time
import json
import torch
import pika
import logging
from flask import Flask, Response
from deep_sort_realtime.deepsort_tracker import DeepSort

app = Flask(__name__)

# Configuration constants
CONFIG = {
    "confidence_threshold": 0.5,
    "max_cosine_distance": 0.4,
    "max_reconnect_attempts": 5,
    "max_age": 30,
    "n_init": 3,
    "rabbitmq_host": os.getenv('RABBITMQ_HOST', 'localhost'),
    "rabbitmq_port": int(os.getenv('RABBITMQ_PORT', 5672)),
    "rabbitmq_queue": os.getenv('RABBITMQ_QUEUE', 'yolo_triggers'),
    "camera_index": int(os.getenv('CAMERA_INDEX', 1))
}

# Logging configuration
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# YOLOv5 model loading
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
model.conf = CONFIG["confidence_threshold"]

# Feature extractor model loading (ResNet)
feature_extractor = torch.hub.load('pytorch/vision:v0.10.0', 'resnet50', pretrained=True)
feature_extractor = torch.nn.Sequential(*(list(feature_extractor.children())[:-1]))
feature_extractor.eval()

# DeepSORT tracker initialization
deepsort_tracker = DeepSort(max_age=CONFIG["max_age"], n_init=CONFIG["n_init"], nms_max_overlap=1.0, max_cosine_distance=CONFIG["max_cosine_distance"])

def create_rabbitmq_channel():
    """Creates a RabbitMQ connection and channel."""
    connection = pika.BlockingConnection(pika.ConnectionParameters(
        host=CONFIG["rabbitmq_host"],
        port=CONFIG["rabbitmq_port"]
    ))
    channel = connection.channel()
    channel.queue_declare(queue=CONFIG["rabbitmq_queue"])
    return connection, channel

def initialize_resources():
    """Initializes camera and RabbitMQ resources."""
    cap = cv2.VideoCapture(CONFIG["camera_index"])
    connection, channel = create_rabbitmq_channel()
    return cap, connection, channel

def release_resources(cap, connection):
    """Releases camera and RabbitMQ resources."""
    if cap.isOpened():
        cap.release()
    connection.close()

def handle_camera_disconnection(cap):
    """Handles camera disconnection by attempting to reconnect."""
    cap.release()
    time.sleep(2)
    cap = cv2.VideoCapture(CONFIG["camera_index"])
    return cap

def publish_message(channel, message):
    """Publishes a message to the RabbitMQ queue with reconnection attempts."""
    reconnect_attempts = 0
    while True:
        try:
            channel.basic_publish(exchange='', routing_key=CONFIG["rabbitmq_queue"], body=json.dumps(message))
            break
        except pika.exceptions.StreamLostError:
            reconnect_attempts += 1
            if reconnect_attempts > CONFIG["max_reconnect_attempts"]:
                raise
            logger.warning("Connection lost, reconnecting...")
            time.sleep(2)
            connection, channel = create_rabbitmq_channel()
    return channel

def generate_embeddings(crops):
    """Generates embeddings for the cropped detections using the feature extractor."""
    embeddings = []
    for crop in crops:
        crop = cv2.resize(crop, (224, 224))
        crop = torch.tensor(crop).float().permute(2, 0, 1).unsqueeze(0)
        with torch.no_grad():
            embedding = feature_extractor(crop).squeeze().numpy()
        embeddings.append(embedding)
        logger.info(f"Generated embedding: {embedding}")
    return embeddings

def generate_frames():
    """Generates video frames with object detection and tracking."""
    cap, connection, channel = initialize_resources()
    tracked_ids = set()

    while True:
        ret, frame = cap.read()
        if not ret:
            logger.warning("Camera disconnected. Reconnecting...")
            cap = handle_camera_disconnection(cap)
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
        embeddings = generate_embeddings(crops) if crops else []

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
                channel = publish_message(channel, message)
                tracked_ids.add(track.track_id)

        # Encode and yield frame
        _, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    release_resources(cap, connection)

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    try:
        app.run(host='0.0.0.0', port=5000, debug=True)
    except KeyboardInterrupt:
        logger.info("Server stopped by user.")

