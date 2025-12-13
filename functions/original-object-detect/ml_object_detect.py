# app.py
from flask import Flask, request, jsonify
import os
import tempfile
from datetime import datetime, timezone
import time
from multiprocessing import Pool
from pathlib import Path
from ultralytics import YOLO

# from config import ACCESS_KEY, BUCKET, ENDPOINT, SECRET_KEY
from minio import Minio
from zeus.monitor import ZeusMonitor
from zeus.device.cpu import get_current_cpu_index

minio_client = None
model = None

def is_box_valid(box: list) -> bool:
    for pos in box:
        if pos < 0:
            return False
    return True

app = Flask(__name__)

def get_timestamp_ms():
    # return int(round(datetime.now(timezone.utc).timestamp() * 1000))
    return time.time()

if os.path.exists("/.dockerenv"):
    endpoint = "minio:9000"      # inside container (use Docker network alias)
else:
    endpoint = "localhost:9000"

@app.post("/ping")
def ping(args=dict()):
    global minio_client
    global model

    req = request.get_json()
    result = dict()
    # -----------------------------------------------------------------------
    # Parse params
    # -----------------------------------------------------------------------
    timestamps = {
        "main_start_ms": 0.0,
        "main_end_ms": 0.0,
        "minio_get_ms": 0.0,
        "minio_put_ms": 0.0,
    }
    access_key = "ROOTNAME"
    secret_key = "CHANGEME123"
    bucket_name = "images"
    if minio_client is None:
        minio_client = Minio(
            endpoint=endpoint,
            access_key=access_key,
            secret_key=secret_key,
            secure=False,
        )
    image_name = req['image_name']

    if model is None:
        model = YOLO('yolov8s.pt')
    if minio_client is None:
        minio_client = Minio(
            endpoint=endpoint,
            access_key=access_key,
            secret_key=secret_key,
            secure=False,
        )

    object_classes = []
    object_boxes = []

    timestamps["main_start_ms"] = get_timestamp_ms()

    with tempfile.NamedTemporaryFile(suffix=".png") as fp:
        image_get_start_ms = get_timestamp_ms()
        minio_client.fget_object(bucket_name=bucket_name,
                                 object_name=image_name, file_path=fp.name)
        image_get_end_ms = get_timestamp_ms()
        timestamps['minio_get_ms'] += (image_get_end_ms - image_get_start_ms)

        results = model(
            source=fp.name,
            imgsz=640,
            device='cpu',
            conf=0.5,     # light prefilter; final check uses conf_thres
            verbose=False
        )

    # --------------------------------------------------------------------------
    # Post-process
        # --------------------------------------------------------------------------
        if results:
            r = results[0]  # batch size 1
            names = r.names if hasattr(r, "names") and r.names else getattr(model, "names", {})

            # r.boxes: per-detection tensors with xyxy, conf, cls
            if r.boxes is not None and len(r.boxes) > 0:
                for b in r.boxes:
                    # xyxy coords
                    xyxy = b.xyxy[0].tolist()  # [x1, y1, x2, y2]
                    cls_id = int(b.cls[0].item()) if hasattr(b.cls[0], "item") else int(b.cls[0])

                    if is_box_valid(xyxy):
                        cls_name = names.get(cls_id, str(cls_id))
                        object_classes.append(cls_name)
                        object_boxes.append([float(v) for v in xyxy])

        # --------------------------------------------------------------------------
        # Return result
        # --------------------------------------------------------------------------

    timestamps["main_end_ms"] = get_timestamp_ms()
    result['object_classes'] = object_classes
    result['object_boxes'] = object_boxes
    result["timestamps"] = timestamps
        
    return result


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8124)
