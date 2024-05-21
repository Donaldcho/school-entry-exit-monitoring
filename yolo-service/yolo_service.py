import os
import cv2
import time
import json
import logging
import pika
import asyncio
from aiohttp import web
from aiohttp import WSMsgType
from dotenv import load_dotenv
from prometheus_client import start_http_server, Counter, Summary
from config import CONFIG
from yolo_utils import load_yolo_model, generate_embeddings
from feature_extractor import load_feature_extractor
from deep_sort_realtime.deepsort_tracker import DeepSort

# Load environment variables from .env file
load_dotenv()

# Logging configuration
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Prometheus metrics
REQUEST_COUNT = Counter('request_count', 'Total web service requests')
REQUEST_LATENCY = Summary('request_latency_seconds', 'Latency of requests')

# Start Prometheus server
start_http_server(8000)

# Global variables for tracking status
start_time = time.time()
websockets = []

# YOLOv5 model loading
try:
    confidence_threshold = CONFIG.confidence_threshold
    model = load_yolo_model(confidence_threshold)
    logger.info(f"YOLOv5 model loaded with confidence threshold: {confidence_threshold}")
except AttributeError as e:
    logger.error(f"Error accessing configuration key: {e}")
    raise

# Feature extractor model loading (ResNet)
feature_extractor = load_feature_extractor()

# DeepSORT tracker initialization
try:
    max_age = CONFIG.max_age
    n_init = CONFIG.n_init
    max_cosine_distance = CONFIG.max_cosine_distance
    deepsort_tracker = DeepSort(
        max_age=max_age, 
        n_init=n_init, 
        nms_max_overlap=1.0, 
        max_cosine_distance=max_cosine_distance
    )
    logger.info("DeepSORT tracker initialized with max age: %s, n_init: %s, max cosine distance: %s", max_age, n_init, max_cosine_distance)
except AttributeError as e:
    logger.error(f"Error accessing configuration key: {e}")
    raise

# RabbitMQ connection setup with retry mechanism
def create_rabbitmq_channel():
    max_retries = 5
    retries = 0
    while retries < max_retries:
        try:
            credentials = pika.PlainCredentials(CONFIG.rabbitmq_username, CONFIG.rabbitmq_password)
            parameters = pika.ConnectionParameters(
                host=CONFIG.rabbitmq_host,
                port=CONFIG.rabbitmq_port,
                credentials=credentials,
                heartbeat=600,
                blocked_connection_timeout=300
            )
            connection = pika.BlockingConnection(parameters)
            channel = connection.channel()
            channel.queue_declare(queue=CONFIG.rabbitmq_queue, durable=True)
            logger.info("RabbitMQ connection established and queue declared")
            return connection, channel
        except pika.exceptions.ProbableAuthenticationError as e:
            logger.error(f"RabbitMQ authentication error: {e}")
            retries += 1
            time.sleep(5)
        except pika.exceptions.AMQPConnectionError as e:
            logger.error(f"RabbitMQ connection error: {e}")
            retries += 1
            time.sleep(5)
        except Exception as e:
            logger.error(f"Failed to establish RabbitMQ connection: {e}")
            retries += 1
            time.sleep(5)

    logger.error("Exceeded maximum retries for RabbitMQ connection")
    return None, None

# Reconnect and re-establish RabbitMQ channel if connection is lost
def reconnect_rabbitmq():
    global connection, channel
    connection, channel = create_rabbitmq_channel()
    if not connection or not channel:
        logger.error("Failed to re-establish RabbitMQ connection. Exiting...")
        exit(1)

# Initialize RabbitMQ resources
connection, channel = create_rabbitmq_channel()
if not connection or not channel:
    logger.error("Failed to establish RabbitMQ connection. Exiting...")
    exit(1)

def publish_message(channel, message):
    try:
        channel.basic_publish(
            exchange='',
            routing_key=CONFIG.rabbitmq_queue,
            body=json.dumps(message),
            properties=pika.BasicProperties(delivery_mode=2)  # make message persistent
        )
        logger.info(f"Message published to RabbitMQ: {message}")
    except pika.exceptions.ChannelClosedByBroker as e:
        logger.error(f"RabbitMQ channel closed by broker: {e}")
        reconnect_rabbitmq()
    except pika.exceptions.AMQPConnectionError as e:
        logger.error(f"RabbitMQ connection error: {e}")
        reconnect_rabbitmq()
    except Exception as e:
        logger.error(f"Failed to publish message: {e}")
        reconnect_rabbitmq()

async def generate_frames():
    """Generates video frames from the camera with YOLO processing and DeepSORT tracking."""
    cap = cv2.VideoCapture(CONFIG.camera_index)
    if not cap.isOpened():
        logger.error("Cannot open camera. Please check the camera connection and the camera index.")
        return

    tracked_ids = set()

    while True:
        try:
            ret, frame = cap.read()
            if not ret:
                logger.warning("Camera disconnected. Reconnecting...")
                cap = cv2.VideoCapture(CONFIG.camera_index)
                if not cap.isOpened():
                    logger.error("Failed to reconnect to the camera.")
                    break
                continue

            # YOLOv5 inference
            results = model(frame)

            # Get bounding boxes and confidences
            detections = []
            crops = []
            for det in results.xyxy[0].cpu().numpy():
                x1, y1, x2, y2, conf, cls = det
                if int(cls) == 0 and conf > confidence_threshold:
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
                    publish_message(channel, message)
                    for ws in websockets:
                        await ws.send_json(message)
                    tracked_ids.add(track.track_id)

            # Encode and yield frame
            _, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

        except Exception as e:
            logger.error(f"Error processing frame: {e}")
            break

    cap.release()

async def video_feed(request):
    REQUEST_COUNT.inc()
    with REQUEST_LATENCY.time():
        try:
            return web.Response(body=generate_frames(), content_type='multipart/x-mixed-replace; boundary=frame')
        except Exception as e:
            logger.error(f"Error in video_feed: {e}")
            return web.Response(status=500)

async def status(request):
    uptime = time.time() - start_time
    return web.json_response({
        "status": "running",
        "uptime": f"{uptime//3600:.0f}h {(uptime%3600)//60:.0f}m"
    })

async def websocket_handler(request):
    ws = web.WebSocketResponse()
    await ws.prepare(request)

    websockets.append(ws)
    try:
        async for msg in ws:
            if msg.type == WSMsgType.TEXT:
                await ws.send_str("Message received")
            elif msg.type == WSMsgType.ERROR:
                logger.error(f"WebSocket connection closed with exception: {ws.exception()}")
    finally:
        websockets.remove(ws)

    return ws

app = web.Application()
app.router.add_get('/video_feed', video_feed)
app.router.add_get('/status', status)
app.router.add_get('/ws', websocket_handler)

if __name__ == '__main__':
    try:
        web.run_app(app, host='0.0.0.0', port=5000)
    except KeyboardInterrupt:
        logger.info("Server stopped by user.")
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
