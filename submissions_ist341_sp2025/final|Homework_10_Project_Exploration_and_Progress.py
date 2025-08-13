### Data training ###
#
# All labeled photos have been uploaded to the link below on my Google Drive.
# After the data training finishes, the best weights model will be downloaded in order to use it
# with the local machine environment that allows for live cam capture.


from ultralytics import YOLO


DATA_YAML = '/content/drive/MyDrive/SmartCamAssistant/data.yaml'

model = YOLO('yolov5s.pt')  # or yolov5n.pt for a tiny model
model.train(data=DATA_YAML, epochs=30, imgsz=640)



# Download the best weights
from google.colab import files
files.download('runs/detect/train/weights/best.pt')



