import torch
from models import build_model
from hubconf import detr_resnet50

from torchsummary import summary

model = detr_resnet50()
summary(model)