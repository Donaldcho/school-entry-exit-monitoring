import cv2
import time
import json
import torch
import pika
import logging
from config import CONFIG

logger = logging.getLogger(__name__)

def load_yolo_model(confidence_threshold):
    """Loads the YOLOv5 model with the specified confidence threshold."""
    model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
    model.conf = confidence_threshold
    return model

def handle_camera_disconnection(cap, camera_index):
    """Handles camera disconnection by attempting to reconnect."""
    cap.release()
    time.sleep(2)
    cap = cv2.VideoCapture(camera_index)
    return cap

def publish_message(channel, message, queue, max_reconnect_attempts):
    """Publishes a message to the RabbitMQ queue with reconnection attempts."""
    reconnect_attempts = 0
    while True:
        try:
            channel.basic_publish(exchange='', routing_key=queue, body=json.dumps(message))
            break
        except pika.exceptions.StreamLostError:
            reconnect_attempts += 1
            if reconnect_attempts > max_reconnect_attempts:
                raise
            logger.warning("Connection lost, reconnecting...")
            time.sleep(2)
            connection, channel = create_rabbitmq_channel()
    return channel

def generate_embeddings(crops, feature_extractor):
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
