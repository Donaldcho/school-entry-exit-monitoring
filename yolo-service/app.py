import os
import cv2
import time
import json
import logging
import torch
import pika
import jwt
from flask import Flask, Response, request, jsonify
from flask_cors import CORS
from deep_sort_realtime.deepsort_tracker import DeepSort
from config import CONFIG
from feature_extractor import load_feature_extractor
from yolo_utils import load_yolo_model, handle_camera_disconnection, publish_message, generate_embeddings, create_rabbitmq_channel, initialize_resources, release_resources

from dotenv import load_dotenv
load_dotenv()

app = Flask(__name__)
CORS(app)

# Logging configuration
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# YOLOv5 model loading
model = load_yolo_model(CONFIG["confidence_threshold"])

# Feature extractor model loading (ResNet)
feature_extractor = load_feature_extractor()

# DeepSORT tracker initialization
deepsort_tracker = DeepSort(max_age=CONFIG["max_age"], n_init=CONFIG["n_init"], nms_max_overlap=1.0, max_cosine_distance=CONFIG["max_cosine_distance"])

def token_required(f):
    def decorator(*args, **kwargs):
        token = request.headers.get('x-access-tokens')
        if not token:
            return jsonify({'message': 'Token is missing!'}), 403
        try:
            data = jwt.decode(token, CONFIG["SECRET_KEY"], algorithms=["HS256"])
        except jwt.ExpiredSignatureError:
            return jsonify({'message': 'Token has expired!'}), 403
        except jwt.InvalidTokenError:
            return jsonify({'message': 'Token is invalid!'}), 403
        return f(*args, **kwargs)
    return decorator

def update_statistics(detections, start_time):
    try:
        detected_objects = len(detections)
        processing_time = int((time.time() - start_time) * 1000)
        stats_message = {
            'detectedObjects': detected_objects,
            'processingTime': processing_time
        }
        publish_message(stats_channel, stats_message, CONFIG["rabbitmq_queue"], CONFIG["max_reconnect_attempts"])
    except Exception as e:
        logger.error(f"Failed to update statistics: {e}")

def generate_frames():
    cap, connection, channel = initialize_resources()
    tracked_ids = set()

    while True:
        try:
            start_time = time.time()
            ret, frame = cap.read()
            if not ret:
                logger.warning("Camera disconnected. Reconnecting...")
                cap = handle_camera_disconnection(cap, CONFIG["camera_index"])
                continue

            results = model(frame)

            detections = []
            crops = []
            for det in results.xyxy[0].cpu().numpy():
                x1, y1, x2, y2, conf, cls = det
                if int(cls) == 0 and conf > CONFIG["confidence_threshold"]:
                    x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
                    width, height = x2 - x1, y2 - y1
                    detections.append(([x1, y1, width, height], conf, 'person'))
                    crops.append(frame[y1:y2, x1:x2])

            embeddings = generate_embeddings(crops, feature_extractor) if crops else []

            tracks = deepsort_tracker.update_tracks(detections, embeds=embeddings, frame=frame)

            for track in tracks:
                if not track.is_confirmed() or track.time_since_update > 1:
                    continue
                bbox = track.to_tlbr()
                cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2])), (int(bbox[3])), (0, 255, 0), 2)
                cv2.putText(frame, str(track.track_id), (int(bbox[0]), int(bbox[1] - 10)), 0, 0.75, (0, 255, 0), 2)

                if track.time_since_update == 0 and track.is_confirmed() and track.track_id not in tracked_ids:
                    message = {
                        'track_id': track.track_id,
                        'bbox': [int(x) for x in bbox],
                        'timestamp': time.time(),
                    }
                    channel = publish_message(channel, message, CONFIG["rabbitmq_queue"], CONFIG["max_reconnect_attempts"])
                    tracked_ids.add(track.track_id)

            update_statistics(detections, start_time)

            _, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

        except Exception as e:
            logger.error(f"Error in generating frames: {e}")
        finally:
            release_resources(cap, connection)

@app.route('/video_feed')
@token_required
def video_feed():
    try:
        return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')
    except Exception as e:
        logger.error(f"Error in /video_feed: {e}")
        return jsonify({'status': 'error', 'message': 'Internal server error'}), 500

@app.route('/login', methods=['POST'])
def login():
    auth = request.authorization
    if auth and auth.password == 'password':
        token = jwt.encode({'user': auth.username}, CONFIG["SECRET_KEY"], algorithm="HS256")
        return jsonify({'token': token})
    return jsonify({'message': 'Could not verify'}), 401

if __name__ == '__main__':
    try:
        app.run(host='0.0.0.0', port=5000, debug=True)
    except Exception as e:
        logger.error(f"Error starting Flask server: {e}")
    except KeyboardInterrupt:
        logger.info("Server stopped by user.")
