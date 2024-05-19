# YOLO Microservice

## Overview

The YOLO Microservice is a component of a school entry and exit monitoring system. It utilizes YOLOv5 for object detection, ResNet for feature extraction, and DeepSORT for object tracking. The service processes video feeds, detects and tracks individuals, generates embeddings for detected individuals, and communicates with RabbitMQ to publish detection results.

## Directory Structure

yolo-service/
│
├── app.py
├── Dockerfile
├── requirements.txt
├── .env
├── yolo_utils.py
├── config.py
└── feature_extractor.py


## Prerequisites

- Docker
- Python 3.9+
- RabbitMQ server
- Camera device (or video file)

## Installation

1. **Clone the repository:**

    ```bash
    git clone https://github.com/yourusername/yolo-service.git
    cd yolo-service
    ```

2. **Create and activate a virtual environment (optional but recommended):**

    ```bash
    python3 -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3. **Install the required dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

4. **Set up the environment variables:**

    Create a `.env` file in the root directory with the following content:

    ```env
    CONFIDENCE_THRESHOLD=0.5
    MAX_COSINE_DISTANCE=0.4
    MAX_RECONNECT_ATTEMPTS=5
    MAX_AGE=30
    N_INIT=3
    RABBITMQ_HOST=localhost
    RABBITMQ_PORT=5672
    RABBITMQ_QUEUE=yolo_triggers
    CAMERA_INDEX=0
    ```

## Running the Service

### Using Python

1. **Start the RabbitMQ server.**
   
2. **Run the YOLO Microservice:**

    ```bash
    python app.py
    ```

3. **Access the video feed:**

    Open your browser and navigate to `http://localhost:5000/video_feed`.

### Using Docker

1. **Build the Docker image:**

    ```bash
    docker build -t yolo-service .
    ```

2. **Run the Docker container:**

    ```bash
    docker run -d -p 5000:5000 --env-file .env yolo-service
    ```

3. **Access the video feed:**

    Open your browser and navigate to `http://localhost:5000/video_feed`.

## Project Structure

- **`app.py`**: The main application file. Defines the Flask application, initializes the YOLO model, feature extractor, and DeepSORT tracker, and sets up the video feed route.
- **`Dockerfile`**: Docker configuration for building and running the application.
- **`requirements.txt`**: Lists the Python dependencies required for the application.
- **`.env`**: Environment variables configuration file.
- **`config.py`**: Loads configuration settings from environment variables.
- **`feature_extractor.py`**: Contains the function to load the feature extractor model (ResNet).
- **`yolo_utils.py`**: Utility functions for YOLO model loading, camera handling, RabbitMQ operations, and generating embeddings.

## Configuration

The service can be configured using environment variables defined in the `.env` file. Key configuration options include:

- `CONFIDENCE_THRESHOLD`: The confidence threshold for YOLOv5 detections.
- `MAX_COSINE_DISTANCE`: The maximum cosine distance for DeepSORT.
- `MAX_RECONNECT_ATTEMPTS`: The maximum number of reconnection attempts for RabbitMQ.
- `MAX_AGE`: The maximum age for DeepSORT tracks.
- `N_INIT`: The number of frames before a track is confirmed in DeepSORT.
- `RABBITMQ_HOST`: The hostname of the RabbitMQ server.
- `RABBITMQ_PORT`: The port number of the RabbitMQ server.
- `RABBITMQ_QUEUE`: The RabbitMQ queue name for publishing detection results.
- `CAMERA_INDEX`: The camera index or video source for OpenCV.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request for any improvements or bug fixes.

## Contact

For any questions or inquiries, please contact deviceterra@gmail.com.

