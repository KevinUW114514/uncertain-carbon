# app.py
from flask import request, jsonify
import io
import os
from datetime import datetime, timezone
from multiprocessing import Pool
from pathlib import Path
import time
import json

# from config import ACCESS_KEY, BUCKET, ENDPOINT, SECRET_KEY
from minio import Minio
from PIL import Image, ImageFilter

import redis

minio_client = None

r = redis.Redis(
    host="redis.ot-operators.svc.cluster.local",
    port=6379,
    decode_responses=True,
)

# app = Flask(__name__)

def get_timestamp_ms():
    # return int(round(datetime.now(timezone.utc).timestamp() * 1000))
    return time.time()

def minio_get_image(
    minio_client, 
    bucket_name, 
    image_name, 
    timestamps
):
    # minio_get_start_ms = get_timestamp_ms()
    recv = minio_client.get_object(
        bucket_name=bucket_name, object_name=image_name)
    bytes_data = recv.read()
    # minio_get_end_ms = get_timestamp_ms()
    # timestamps['minio_get_ms'] += (minio_get_end_ms - minio_get_start_ms)
    image = Image.open(io.BytesIO(bytes_data))
    return image

def minio_put_image(minio_client, bucket_name, image_name, image, timestamps):
    bytes_buffer = io.BytesIO()
    if Path(image_name).suffix == '.jpg' or Path(image_name).suffix == '.jpeg':
        fmt = 'JPEG'
    elif Path(image_name).suffix == '.png':
        fmt = 'PNG'
    else:
        raise Exception(
            'Unsupported image format: {}.'.format(Path(image_name).suffix))
    image.save(fp=bytes_buffer, format=fmt)
    bytes_buffer.seek(0)
    # minio_put_start_ms = get_timestamp_ms()
    minio_client.put_object(bucket_name=bucket_name,
                            object_name=image_name,
                            data=bytes_buffer,
                            length=bytes_buffer.getbuffer().nbytes)
    # minio_put_end_ms = get_timestamp_ms()
    # timestamps['minio_put_ms'] += (minio_put_end_ms - minio_put_start_ms)
    return


# if os.path.exists("/.dockerenv"):
#     endpoint = "10.52.2.162:9000"      # inside container (use Docker network alias)
# else:
#     endpoint = "localhost:9000"

endpoint = "minio.minio.svc.cluster.local:9000"

# @app.post("/ping")
def main():
    global minio_client
    req = request.get_json()
    result = dict()
    # -----------------------------------------------------------------------
    # Parse params
    # -----------------------------------------------------------------------
    # timestamps = {
    #     "main_start_ms": 0,
    #     "main_end_ms": 0,
    #     "minio_get_ms": 0,
    #     "minio_put_ms": 0,
    # }
    # timestamps["main_start_ms"] = get_timestamp_ms()
    # timestamps["main_start_ms"] = os.times()
    
    start_time = time.monotonic()
    access_key = "minioadmin"
    secret_key = "minioadmin123"
    src_bucket_name = "images"
    dst_bucket_name = "processed-images"
    if minio_client is None:
        minio_client = Minio(
            endpoint=endpoint,
            access_key=access_key,
            secret_key=secret_key,
            secure=False,
        )
    image_name = req['image_name']

    # -----------------------------------------------------------------------
    # Action execution
    # -----------------------------------------------------------------------
    timestamps = None
    image = minio_get_image(
        minio_client=minio_client,
        bucket_name=src_bucket_name,
        image_name=image_name,
        timestamps=timestamps,
    )

    image = image.transpose(Image.FLIP_LEFT_RIGHT)
    image = image.transpose(Image.FLIP_TOP_BOTTOM)
    image = image.transpose(Image.ROTATE_90)
    image = image.transpose(Image.ROTATE_180)
    image = image.transpose(Image.ROTATE_270)
    image = image.filter(ImageFilter.BLUR)
    image = image.filter(ImageFilter.CONTOUR)
    image = image.filter(ImageFilter.SHARPEN)
    image = image.convert("L")

    new_image_name = Path(image_name).stem + "_processed" + Path(image_name).suffix
    minio_put_image(
        minio_client=minio_client,
        bucket_name=dst_bucket_name,
        image_name=new_image_name,
        image=image,
        timestamps=timestamps,
    )

    # -----------------------------------------------------------------------
    # Return results
    # -----------------------------------------------------------------------
    # timestamps["main_end_ms"] = get_timestamp_ms()
    # timestamps["main_end_ms"] = os.times()
    result["duration"] = time.monotonic() - start_time

    
    r.lpush("ml-image-processing", json.dumps({
        "image_name": image_name
        })
    )

    return jsonify(result), 200


# if __name__ == "__main__":
#     app.run(host="0.0.0.0", port=8123)
