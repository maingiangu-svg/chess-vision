"""Check the running LAN TLS/WebSocket server with verified CA trust."""
import argparse
import asyncio
import base64
import json
import ssl
from pathlib import Path
import cv2
import numpy as np
from websockets.asyncio.client import connect

async def check(url: str, ca: str, calibration_image: Path | None = None) -> None:
    context = ssl.create_default_context(cafile=ca)
    if calibration_image:
        encoded = base64.b64encode(calibration_image.read_bytes()).decode()
    else:
        ok, jpeg = cv2.imencode('.jpg', np.zeros((480, 640, 3), dtype=np.uint8))
        assert ok
        encoded = base64.b64encode(jpeg).decode()
    async with connect(url, ssl=context) as ws:
        if calibration_image:
            await ws.send(json.dumps({'type': 'frame', 'calibrate_templates': True, 'jpeg': encoded}))
            confirmation = json.loads(await ws.recv())
            assert confirmation['type'] == 'calibrated', confirmation
            for _ in range(3):
                await ws.send(json.dumps({'type': 'frame', 'jpeg': encoded}))
                result = json.loads(await ws.recv())
            assert result['move'] is not None, result
        else:
            await ws.send(json.dumps({'type': 'corners', 'points': [[50, 50], [590, 50], [590, 430], [50, 430]]}))
            await ws.send(json.dumps({'type': 'frame', 'jpeg': encoded}))
            result = json.loads(await ws.recv())
        assert result['type'] == 'result' and result['frame_size'][0] == 640
        print(json.dumps({'status': result['status'], 'corners': result['corners'], 'move': result['move'], 'latency_ms': result['latency_ms']}))

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--url', required=True)
    p.add_argument('--ca', required=True)
    p.add_argument('--calibration-image', type=Path)
    args = p.parse_args()
    asyncio.run(check(args.url, args.ca, args.calibration_image))
