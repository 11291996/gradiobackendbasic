#comet -> tracking tool
#install comet with pip install comet_ml
"""
export COMET_API_KEY="<Your API Key>"
export COMET_WORKSPACE="<Your Workspace Name>"
export COMET_PROJECT_NAME="<Your Project Name>"
"""
#.comet.config file can do this too
"""
[comet]
api_key=<Your API Key>
workspace=<Your Workspace Name>
project_name=<Your Project Name>
"""
#intialize comet
import comet_ml
comet_ml.init()

import requests
import torch
from PIL import Image
from torchvision import transforms
import gradio as gr

torch.hub.download_url_to_file("https://github.com/pytorch/hub/raw/master/images/dog.jpg", "dog.jpg")

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

model = torch.hub.load("pytorch/vision:v0.6.0", "resnet18", pretrained=True).eval()
model = model.to(device)

# Download human-readable labels for ImageNet.
response = requests.get("https://git.io/JJkYN")
labels = response.text.split("\n")


def predict(inp):
    inp = Image.fromarray(inp.astype("uint8"), "RGB")
    inp = transforms.ToTensor()(inp).unsqueeze(0)
    with torch.no_grad():
        prediction = torch.nn.functional.softmax(model(inp.to(device))[0], dim=0)
    return {labels[i]: float(prediction[i]) for i in range(1000)}


inputs = gr.Image()
outputs = gr.Label(num_top_classes=3)

io = gr.Interface(
    fn=predict, inputs=inputs, outputs=outputs, examples=["dog.jpg"]
)
io.launch(inline=False, share=True)

experiment = comet_ml.Experiment()
experiment.add_tag("image-classifier")

io.integrate(comet_ml=experiment)

#ONNX -> open neural network exchange, mentioned in pytorchbasic.py
import numpy as np
import math
import matplotlib.pyplot as plt
import cv2
import json
import gradio as gr
from huggingface_hub import hf_hub_download
from onnx import hub
import onnxruntime as ort

# loads ONNX model from ONNX Model Zoo
model = hub.load("efficientnet-lite4")
# loads the labels text file
labels = json.load(open("labels_map.txt", "r"))

# sets image file dimensions to 224x224 by resizing and cropping image from center
def pre_process_edgetpu(img, dims):
    output_height, output_width, _ = dims
    img = resize_with_aspectratio(img, output_height, output_width, inter_pol=cv2.INTER_LINEAR)
    img = center_crop(img, output_height, output_width)
    img = np.asarray(img, dtype='float32')
    # converts jpg pixel value from [0 - 255] to float array [-1.0 - 1.0]
    img -= [127.0, 127.0, 127.0]
    img /= [128.0, 128.0, 128.0]
    return img

# resizes the image with a proportional scale
def resize_with_aspectratio(img, out_height, out_width, scale=87.5, inter_pol=cv2.INTER_LINEAR):
    height, width, _ = img.shape
    new_height = int(100. * out_height / scale)
    new_width = int(100. * out_width / scale)
    if height > width:
        w = new_width
        h = int(new_height * height / width)
    else:
        h = new_height
        w = int(new_width * width / height)
    img = cv2.resize(img, (w, h), interpolation=inter_pol)
    return img

# crops the image around the center based on given height and width
def center_crop(img, out_height, out_width):
    height, width, _ = img.shape
    left = int((width - out_width) / 2)
    right = int((width + out_width) / 2)
    top = int((height - out_height) / 2)
    bottom = int((height + out_height) / 2)
    img = img[top:bottom, left:right]
    return img


sess = ort.InferenceSession(model)

def inference(img):
  img = cv2.imread(img)
  img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

  img = pre_process_edgetpu(img, (224, 224, 3))

  img_batch = np.expand_dims(img, axis=0)

  results = sess.run(["Softmax:0"], {"images:0": img_batch})[0]
  result = reversed(results[0].argsort()[-5:])
  resultdic = {}
  for r in result:
      resultdic[labels[str(r)]] = float(results[0][r])
  return resultdic

title = "EfficientNet-Lite4"
description = "EfficientNet-Lite 4 is the largest variant and most accurate of the set of EfficientNet-Lite model. It is an integer-only quantized model that produces the highest accuracy of all of the EfficientNet models. It achieves 80.4% ImageNet top-1 accuracy, while still running in real-time (e.g. 30ms/image) on a Pixel 4 CPU."
examples = [['catonnx.jpg']]
gr.Interface(inference, gr.Image(type="filepath"), "label", title=title, description=description, examples=examples).launch()

#wandb -> tracking tool

#wandb login in terminal

import torch, requests
from torchvision import transforms
from PIL import Image


model = torch.hub.load('pytorch/vision:v0.6.0', 'resnet18', pretrained=True).eval()


#download human-readable labels for ImageNet.
response = requests.get("https://git.io/JJkYN")
labels = response.text.split("\n")


def predict(inp):
  inp = Image.fromarray(inp.astype('uint8'), 'RGB')
  inp = transforms.ToTensor()(inp).unsqueeze(0)
  with torch.no_grad():
    prediction = torch.nn.functional.softmax(model(inp)[0], dim=0)
  return {labels[i]: float(prediction[i]) for i in range(1000)}

inputs = gr.inputs.Image()
outputs = gr.outputs.Label(num_top_classes=3)
io = gr.Interface(fn=predict, inputs=inputs, outputs=outputs)
io.launch(share=True)

import wandb
wandb.init(project="your-test-project")

io.integrate(wandb=wandb)

