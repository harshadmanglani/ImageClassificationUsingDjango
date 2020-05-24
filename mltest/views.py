from django.shortcuts import render
from django.http import JsonResponse
import base64
from django.core.files.base import ContentFile
from django.core.files.storage import default_storage
from django.conf import settings 
import datetime
import traceback
import cv2
from tensorflow import keras
import json
import numpy as np 
from tensorflow.keras.models import load_model
from deeplearningsettings import DLSettings

def predict(image_path):
    obj = DLSettings()
    model = load_model(obj.path)
    class_names = obj.output_classes
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    res = cv2.resize(img, dsize=(obj.pixels, obj.pixels))
    res = res/255.0
    res = np.asarray(res)
    res = res.reshape((-1, obj.pixels, obj.pixels, 3))
    y_prob = model.predict(res)
    probs = dict()
    for i in range(0, len(y_prob[0])):
        probs[i] = y_prob[0][i]
    probs = sorted(probs.items(), key=lambda x: x[1], reverse = True) 
    pred = "Most confident prediction: " + '\n'
    index = probs[0][0]
    labels = {value: key for key, value in class_names}
    name = labels[index].split('-')
    name = name[1:]
    name = '-'.join(name)
    name = name.capitalize()
    pred += str(name) + '\t' + str(int(probs[0][1]*100)) + '%' + "\n"
    return pred


def index(request):
    if  request.method == "POST":
        f = request.FILES['sentFile'] # here you get the files needed
        file_name = "pic.jpg"
        file_name_2 = default_storage.save(file_name, f)
        file_url = default_storage.url(file_name_2)

        response = predict(file_url)
        print(response)
        return render(request,'homepage.html',{"response": response})
    else:
        return render(request,'homepage.html')