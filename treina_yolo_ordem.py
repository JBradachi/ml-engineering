# imports
from ultralytics import YOLO
import os
import shutil
import gc
import torch

# modelo de arquitetura normal para classificação de família
# basta treinar o yolo para identificar as famílias (264 classes)

# esse é o modelo com segmentação
model = YOLO('yolo26n-cls.pt')

# Executa o fine-tuning no dataset personalizado
results = model.train(
    data='/home/bradas/Downloads/insetos/ordem/ordem_yolo',
    epochs=100,
    imgsz=224,
    batch=64,
    workers=10,
    device=0,
    amp=True,
    cache=False,
    name='ordem_nano'
)
print(results) 
