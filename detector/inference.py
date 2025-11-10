import onnxruntime as ort
import numpy as np
import cv2
from django.conf import settings
import os

# تحميل النموذج مرة واحدة عند بدء السيرفر
model_path = os.path.join(settings.BASE_DIR, 'detector', 'resnet50.onnx')
session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])

def preprocess_image(image_path):
    """
    دالة لتحضير الصورة قبل تمريرها للنموذج
    """
    img = cv2.imread(image_path)
    img = cv2.resize(img, (224, 224))  # نفس حجم الإدخال الذي درّبت عليه النموذج
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))  # (C, H, W)
    img = np.expand_dims(img, axis=0)  # (1, 3, 224, 224)
    return img

def predict_liveness(image_path):
    """
    دالة تُرجع النتيجة: 'real' أو 'fake'
    """
    img = preprocess_image(image_path)
    inputs = {session.get_inputs()[0].name: img}
    outputs = session.run(None, inputs)
    pred = np.argmax(outputs[0])
    return 'real' if pred == 1 else 'fake'
