from ultralytics import YOLO
import matplotlib.pyplot as plt
import skimage as ski

# Load a model
model = YOLO("yolo26n-seg.pt")  # load an official model

# Predict with the model
results = model("bus.jpg")  # predict on an image

# Access the results
for result in results:
    xy = result.masks.xy  # mask polygons in pixel coordinates
    xyn = result.masks.xyn  # normalized mask polygons
    masks = result.masks.data  # binary masks, shape (N,H,W), dtype torch.uint8
