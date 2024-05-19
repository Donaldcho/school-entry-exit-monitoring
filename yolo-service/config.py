import os
from dotenv import load_dotenv

load_dotenv()

CONFIG = {
    "confidence_threshold": float(os.getenv('CONFIDENCE_THRESHOLD', 0.5)),
    "max_cosine_distance": float(os.getenv('MAX_COSINE_DISTANCE', 0.4)),
    "max_reconnect_attempts": int(os.getenv('MAX_RECONNECT_ATTEMPTS', 5)),
    "max_age": int(os.getenv('MAX_AGE', 30)),
    "n_init": int(os.getenv('N_INIT', 3)),
    "rabbitmq_host": os.getenv('RABBITMQ_HOST', 'localhost'),
    "rabbitmq_port": int(os.getenv('RABBITMQ_PORT', 5672)),
    "rabbitmq_queue": os.getenv('RABBITMQ_QUEUE', 'yolo_triggers'),
    "camera_index": int(os.getenv('CAMERA_INDEX', 1))
}
