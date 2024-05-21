Sure, here is the comprehensive `README.md` file in markdown format suitable for GitHub:

```markdown
# YOLO Microservice

## Overview

The YOLO Microservice is a component of a school entry and exit monitoring system. It utilizes YOLOv5 for object detection, ResNet for feature extraction, and DeepSORT for object tracking. The service processes video feeds, detects and tracks individuals, generates embeddings for detected individuals, and communicates with RabbitMQ to publish detection results. Additional features include real-time updates via WebSocket, authentication, logging, and monitoring.

## Directory Structure

```
yolo-service/
│
├── app.py
├── Dockerfile
├── requirements.txt
├── .env
├── yolo_utils.py
├── config.py
├── feature_extractor.py
├── README.md
```

## Prerequisites

- Docker
- Python 3.9+
- RabbitMQ server
- Camera device (or video file)
- Prometheus (for monitoring)
- Grafana (optional, for visualization)

## Installation

### 1. Clone the Repository

```sh
git clone https://github.com/yourusername/yolo-service.git
cd yolo-service
```

### 2. Create and Activate a Virtual Environment (Optional but Recommended)

```sh
python3 -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
```

### 3. Install the Required Dependencies

```sh
pip install -r requirements.txt
```

### 4. Set Up the Environment Variables

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
RABBITMQ_USERNAME=guest
RABBITMQ_PASSWORD=guest
CAMERA_INDEX=0  # Adjust based on available camera indices
API_KEY=your_api_key
```

## Running the Service

### Using Python

1. Start the RabbitMQ server.
2. Run the YOLO Microservice:

   ```sh
   python yolo_service.py
   ```

3. Access the video feed:

   Open your browser and navigate to `http://localhost:5000/video_feed`.

### Using Docker

1. Build the Docker image:

   ```sh
   docker build -t yolo-service .
   ```

2. Run the Docker container:

   ```sh
   docker run -d -p 5000:5000 --env-file .env yolo-service
   ```

3. Access the video feed:

   Open your browser and navigate to `http://localhost:5000/video_feed`.

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
- `API_KEY`: The API key for authentication.

## Security and Authentication

The service uses basic authentication to secure the endpoints. The API key is required to access the video feed and status endpoints. Set the `API_KEY` environment variable in the `.env` file.

## Real-Time Updates

The service provides real-time updates using WebSocket. You can connect to the WebSocket endpoint at `/ws` to receive real-time notifications.

### Example WebSocket Client

#### JavaScript

Create an `index.html` file with the following content and open it in a web browser:

```html
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>WebSocket Client</title>
    <style>
        #messages {
            border: 1px solid #ccc;
            padding: 10px;
            height: 300px;
            overflow-y: scroll;
        }
    </style>
</head>
<body>
    <h1>WebSocket Client</h1>
    <div id="messages"></div>

    <script>
        const ws = new WebSocket('ws://localhost:5000/ws');

        ws.onopen = () => {
            console.log('WebSocket connection established');
        };

        ws.onmessage = (event) => {
            const data = JSON.parse(event.data);
            console.log('Received data:', data);
            const messagesDiv = document.getElementById('messages');
            const message = document.createElement('div');
            message.textContent = `Track ID: ${data.track_id}, BBox: ${data.bbox}, Timestamp: ${new Date(data.timestamp * 1000).toLocaleString()}`;
            messagesDiv.appendChild(message);
            messagesDiv.scrollTop = messagesDiv.scrollHeight;
        };

        ws.onerror = (error) => {
            console.error('WebSocket error:', error);
        };

        ws.onclose = () => {
            console.log('WebSocket connection closed');
        };
    </script>
</body>
</html>
```

#### Python

Create a `websocket_client.py` file with the following content and run it using Python:

```python
import asyncio
import websockets
import json

async def listen():
    uri = "ws://localhost:5000/ws"
    async with websockets.connect(uri) as websocket:
        while True:
            try:
                message = await websocket.recv()
                data = json.loads(message)
                print(f"Received data: Track ID: {data['track_id']}, BBox: {data['bbox']}, Timestamp: {data['timestamp']}")
            except websockets.ConnectionClosed:
                print("WebSocket connection closed")
                break

if __name__ == "__main__":
    asyncio.get_event_loop().run_until_complete(listen())
```

## Monitoring and Logging

The service integrates with Prometheus for monitoring and logging. Metrics are exposed at port 8000.

### Prometheus Setup

1. Install Prometheus from the [official website](https://prometheus.io/download/).
2. Add the following job to your `prometheus.yml` configuration file:

```yaml
scrape_configs:
  - job_name: 'yolo_service'
    static_configs:
      - targets: ['localhost:8000']
```

3. Start Prometheus:

   ```sh
   prometheus --config.file=prometheus.yml
   ```

### Grafana Setup (Optional)

1. Install Grafana from the [official website](https://grafana.com/grafana/download).
2. Add Prometheus as a data source in Grafana.
3. Create dashboards to visualize the metrics collected by Prometheus.

### Example Metrics

The service exposes the following metrics:

- `request_count`: Total number of web service requests.
- `request_latency_seconds`: Latency of requests.

## Deployment

The service can be deployed using Docker and Kubernetes for scalability and high availability. An example `docker-compose.yml` and Kubernetes deployment file are provided.

### Docker Compose

```yaml
version: '3.8'

services:
  yolo_service:
    build: .
    ports:
      - "5000:5000"
    environment:
      - ENV_FOR_DYNACONF=default
    volumes:
      - .:/app
    deploy:
      replicas: 3
      resources:
        limits:
          cpus: '0.50'
          memory: 512M
      restart_policy:
        condition: on-failure
```

### Kubernetes Deployment

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: yolo-service
spec:
  replicas: 3
  selector:
    matchLabels:
      app: yolo-service
  template:
    metadata:
      labels:
        app: yolo-service
    spec:
      containers:
      - name: yolo-service
        image: your-docker-image
        ports:
        - containerPort: 5000
        env:
        - name: ENV_FOR_DYNACONF
          value: "default"
        resources:
          limits:
            memory: "512Mi"
            cpu: "0.5"
---
apiVersion: v1
kind: Service
metadata:
  name: yolo-service
spec:
  selector:
    app: yolo-service
  ports:
    - protocol: TCP
      port: 80
      targetPort: 5000
  type: LoadBalancer
```

## License

This project is licensed under the MIT License. See the LICENSE file for details.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request for any improvements or bug fixes.

## Acknowledgements

Special thanks to the contributors and the open-source community for their invaluable support and resources.
```

### Summary

This `README.md` file covers:

- Overview of the YOLO microservice.
- Installation instructions.
- Configuration settings.
- Instructions to run the service using Python or Docker.
- Setup for WebSocket clients (both JavaScript and Python).
- Monitoring and logging setup with Prometheus.
- Deployment instructions using Docker Compose

 and Kubernetes.
- Licensing and contribution guidelines.

This should provide a comprehensive guide for users and contributors to understand, set up, and use the YOLO microservice.