from ultralytics import YOLO
import urllib.request

def is_box_valid(box: list) -> bool:
    return all(pos >= 0 for pos in box)

# Download image (same one your demo used)
url = "https://raw.githubusercontent.com/zhreshold/mxnet-ssd/master/data/demo/dog.jpg"
urllib.request.urlretrieve(url, "dog.jpg")

# Load YOLO model (COCO classes)
model = YOLO("yolov8s.pt")   # small & CPU-friendly

# Inference (CPU)
results = model("dog.jpg")[0]

print("Detected objects:", len(results.boxes))
names = model.names  # dict: class_id -> class_name

for i, box in enumerate(results.boxes):
    cls_id = int(box.cls.item())
    score = float(box.conf.item())
    xyxy = box.xyxy.squeeze().tolist()  # [x1,y1,x2,y2]
    if is_box_valid(xyxy):
        print(f"{i}: id={cls_id}, name={names[cls_id]}, score={score:.3f}, bbox={xyxy}")

