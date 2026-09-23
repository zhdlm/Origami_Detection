from ultralytics import YOLO
import matplotlib.pyplot as plt
import skimage as ski

# Load a model
model = YOLO("yolo26n-seg.pt")  # load an official model
model.train(
    data="dataset/data.yaml",
    epochs=100,
    imgsz=512,
    batch=8,
    patience=20,
    device="cpu",       # change to 0 if using an NVIDIA GPU
    workers=2,
    plots=True,
)

metrics = model.val()
print(metrics)

model.predict(
    source="dataset/images/test",
    save=True,
    conf=0.25
)

model.save("yolo26n-seg-trained-synth.pt")
#Access the results
#for result in results:
#    xy = result.masks.xy  # mask polygons in pixel coordinates
#    xyn = result.masks.xyn  # normalized mask polygons
#    masks = result.masks.data  # binary masks, shape (N,H,W), dtype torch.uint8

