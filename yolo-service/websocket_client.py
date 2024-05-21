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
