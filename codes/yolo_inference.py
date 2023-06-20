from ultralytics import YOLO
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import os

image = 'data/bird.jpg'

# Load the model
model = YOLO('pretrained/yolov8n.pt')  # load a pretrained model

# Perform inference
results = model(source=image, conf=0.5)
names = model.names

def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)
    
def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)   
    
def show_box(box, ax, class_name=''):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2))
    ax.text(x0, y0-10, f"{class_name}") 

# results.show()
classess = []
for result in results:
    boxes = result.boxes
    for c in boxes.cls:
        classess.append(names[int(c)])
    
lst_boxes = boxes.xyxy.tolist()
print(len(lst_boxes))
print(classess)
bbox = lst_boxes[1]
print(bbox)