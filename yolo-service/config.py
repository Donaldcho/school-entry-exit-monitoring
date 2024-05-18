import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

CONFIG = {
    "confidence_threshold": 0.5,
    "max_cosine_distance": 0.4,
    "max_reconnect_attempts": 5,
    "max_age": 30,
    "n_init": 3,
    "rabbitmq_host": os.getenv('RABBITMQ_HOST', 'localhost'),
    "rabbitmq_port": int(os.getenv('RABBITMQ_PORT', 5672)),
    "rabbitmq_queue": os.getenv('RABBITMQ_QUEUE', 'yolo_triggers'),
    "camera_index": int(os.getenv('CAMERA_INDEX', 0))
}
