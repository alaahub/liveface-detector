from django.shortcuts import render, redirect
from django.conf import settings
from .models import AnalysisLog
from django.core.files.storage import default_storage

import onnxruntime as ort
import numpy as np
from PIL import Image
import cv2
import os
import tempfile
import io
from datetime import datetime
from shutil import copy2

from django.http import JsonResponse, HttpResponse
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as RLImage
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import inch

import cv2
import numpy as np
from django.http import JsonResponse
import tempfile, os
from PIL import Image


# ============================================================
# 🔹 Load ONNX model once when the server starts
# ============================================================

MODEL_PATH = os.path.join(os.path.dirname(__file__), 'resnet50.onnx')

try:
    ort_session = ort.InferenceSession(MODEL_PATH)
    print("✅ ONNX model loaded successfully.")
except Exception as e:
    print("❌ Error loading ONNX model:", e)
    ort_session = None


# ============================================================
# 🔹 Image preprocessing and prediction
# ============================================================

def preprocess_image(image_path, size=(224, 224)):
    """Read image and prepare for ONNX input"""
    img = Image.open(image_path).convert('RGB')
    img = img.resize(size)
    img = np.array(img).astype(np.float32) / 255.0  # Normalize [0,1]
    img = np.transpose(img, (2, 0, 1))  # HWC → CHW
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    return img


def predict_image(image_path):
    """Run ONNX model prediction and return label"""
    if ort_session is None:
        return "error"

    img_input = preprocess_image(image_path)
    input_name = ort_session.get_inputs()[0].name
    outputs = ort_session.run(None, {input_name: img_input})

    pred = np.argmax(outputs[0], axis=1)[0]

    # ⚠️ Adjust if your model has reversed output (0=fake, 1=real)
    return "real" if pred == 1 else "fake"

def predict_image2(image_path):
    """Run ONNX model prediction and return label and confidence score (0-100)"""
    if ort_session is None:
        return "error", 0.0

    img_input = preprocess_image(image_path)
    input_name = ort_session.get_inputs()[0].name
    outputs = ort_session.run(None, {input_name: img_input})

    # استخلاص الاحتمالات (Probabilities)
    probabilities = outputs[0][0]
    pred_index = np.argmax(probabilities)
    
    # استخلاص قيمة الثقة كنسبة مئوية (0-100)
    confidence_score = float(probabilities[pred_index]) * 100
    
    # تحديد النتيجة (Real: 1, Fake: 0)
    result_label = "real" if pred_index == 1 else "fake"
    
    return result_label, confidence_score # 🌟 العودة بالنتيجة والثقة معاً 🌟

# ============================================================
# 🔹 Django Views
# ============================================================

def upload_page(request):
    """Upload page for selecting multiple images"""
    return render(request, 'detector/upload.html')


def process_uploads(request):
    """Handle uploaded images, run model prediction, save results"""
    if request.method == 'POST':
        files = request.FILES.getlist('images')
        for f in files:
            # Save image record in database
            log = AnalysisLog.objects.create(image=f)

            # Full path to saved image
            image_path = log.image.path

            try:
                # Run prediction
                result = predict_image(image_path)
            except Exception as e:
                print("Prediction error:", e)
                result = "error"

            # Save result
            log.result = result
            log.save()

        # Redirect to results page
        return redirect('detector:results_page')

    return redirect('detector:upload_page')

def process_uploads2(request):
    """Handle uploaded images, run model prediction, save results"""
    if request.method == 'POST':
        files = request.FILES.getlist('images')
        for f in files:
            # Save image record in database
            log = AnalysisLog.objects.create(image=f)

            # Full path to saved image
            image_path = log.image.path

            try:
                # 🌟 تشغيل التنبؤ والحصول على النتيجة والثقة 🌟
                result, confidence = predict_image(image_path) 
            except Exception as e:
                print("Prediction error:", e)
                result = "error"
                confidence = 0.0 # قيمة افتراضية للثقة في حالة الخطأ

            # حفظ النتيجة والثقة
            log.result = result
            # 🌟 تخزين الثقة هنا 🌟
            log.confidence = round(confidence, 1) # تقريب الثقة وحفظها
            log.save()

        # Redirect to results page
        return redirect('detector:results_page')

    return redirect('detector:upload_page')

#def results_page(request):
 #   """Show analysis results"""
  #  logs = AnalysisLog.objects.order_by('-created_at')[:50]  # latest 50 entries
  #  return render(request, 'detector/results.html', {'logs': logs})

def results_page(request):
    """Show only unverified analysis results"""
    logs = AnalysisLog.objects.filter(is_verified=False).order_by('-created_at')[:50]
    return render(request, 'detector/results.html', {'logs': logs})


def clear_results(request):
    """Delete all results from the database"""
    AnalysisLog.objects.all().delete()
    return redirect('detector:results_page')


# ============================================================
# ✅ Updated Correct / Incorrect Handlers
# ============================================================

def mark_correct(request, log_id):
    """Mark a prediction as correct"""
    try:
        log = AnalysisLog.objects.get(id=log_id)
        log.is_verified = True
        log.is_correct = True
        log.save()
    except AnalysisLog.DoesNotExist:
        pass
    return redirect('detector:results_page')


def mark_incorrect(request, log_id):
    """Mark a prediction as incorrect and move it to training data"""
    try:
        log = AnalysisLog.objects.get(id=log_id)
        log.is_verified = True
        log.is_correct = False
        log.save()

        # نسخ الصورة لمجلد التدريب
        image_path = log.image.path
        training_dir = os.path.join(settings.MEDIA_ROOT, 'training_data')
        os.makedirs(training_dir, exist_ok=True)
        dest = os.path.join(training_dir, os.path.basename(image_path))
        copy2(image_path, dest)
    except AnalysisLog.DoesNotExist:
        pass
    return redirect('detector:results_page')


# ============================================================
# 🔹 Camera-related Views
# ============================================================

def camera_page(request):
    """Page to show live camera detection"""
    return render(request, 'detector/camera.html')

"""""
def process_frame(request):
    Receive a frame from browser and analyze it
    if request.method == 'POST' and 'frame' in request.FILES:
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.jpg')
        for chunk in request.FILES['frame'].chunks():
            temp_file.write(chunk)
        temp_file.close()

        try:
            result = predict_image(temp_file.name)
        except Exception as e:
            print("Frame prediction error:", e)
            result = "error"

        os.remove(temp_file.name)
        return JsonResponse({'result': result})

    #return JsonResponse({'result': 'no_frame'})
    return JsonResponse({
    'result': result,       # "real" or "fake"
    'confidence': 0.87,     # نسبة وهمية للتجربة
    'bbox': [100, 100, 200, 200]  # مثال لمستطيل
})
"""""


def process_frame(request):
    """Receive a frame from browser, analyze it, and return detection info."""
    if request.method == 'POST' and 'frame' in request.FILES:
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.jpg')
        for chunk in request.FILES['frame'].chunks():
            temp_file.write(chunk)
        temp_file.close()

        try:
            # اقرأ الصورة باستخدام OpenCV
            frame = cv2.imread(temp_file.name)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # ✅ استخدم كاشف الوجه المدمج في OpenCV
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(100, 100))

            # 🔹 تحليل النتيجة بالنموذج
            result = predict_image(temp_file.name)
            confidence = np.random.uniform(0.75, 0.98)  # نسبة وهمية مؤقتة للتجربة

            # 🔹 إعداد الإخراج
            response = {
                "result": result,
                "confidence": round(float(confidence), 2),
            }

            # أضف أول مستطيل إذا تم كشف وجه
            if len(faces) > 0:
                (x, y, w, h) = faces[0]
                response["bbox"] = [int(x), int(y), int(w), int(h)]
            else:
                response["bbox"] = None

        except Exception as e:
            print("Frame processing error:", e)
            response = {"result": "error", "confidence": 0, "bbox": None}

        finally:
            os.remove(temp_file.name)

        return JsonResponse(response)

    return JsonResponse({"result": "no_frame", "confidence": 0, "bbox": None})

def process_frame2(request):
    """Receive a frame from browser, analyze it, and return detection info."""
    if request.method == 'POST' and 'frame' in request.FILES:
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.jpg')
        for chunk in request.FILES['frame'].chunks():
            temp_file.write(chunk)
        temp_file.close()

        try:
            # ... (كشف الوجه باستخدام OpenCV) ...
            
            # 🔹 تحليل النتيجة بالنموذج
            # 🌟 الحصول على النتيجة والثقة الفعلية من predict_image 🌟
            result, confidence_score = predict_image(temp_file.name)
            
            # 🔹 إعداد الإخراج
            response = {
                "result": result,
                # 🌟 إرسال الثقة كنسبة بين 0-1 (كما تتوقعها واجهة JS) 🌟
                "confidence": round(float(confidence_score) / 100.0, 2), 
            }

            # ... (إضافة الـ bbox) ...

        except Exception as e:
            # ... (التعامل مع الأخطاء) ...
            response = {"result": "error", "confidence": 0, "bbox": None}

        finally:
            os.remove(temp_file.name)

        return JsonResponse(response)

    return JsonResponse({"result": "no_frame", "confidence": 0, "bbox": None})

# ============================================================
# 🔹 PDF Report Generation
# ============================================================

def download_report(request):
    """
    Generate a PDF report containing:
    - Project title and date
    - All analyzed images with results (real/fake)
    """
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4)
    styles = getSampleStyleSheet()
    story = []

    # 🏷️ Report title and date
    story.append(Paragraph("<b>Face Liveness Detection Report</b>", styles["Title"]))
    story.append(Spacer(1, 0.2 * inch))
    story.append(Paragraph(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", styles["Normal"]))
    story.append(Spacer(1, 0.3 * inch))

    # 🔹 Last 50 results
    logs = AnalysisLog.objects.order_by('-created_at')[:50]

    for log in logs:
        img_path = log.image.path
        result_text = f"<b>Result:</b> {'✅ Real' if log.result == 'real' else '❌ Fake'}"

        # Add image and result text
        try:
            story.append(RLImage(img_path, width=2 * inch, height=2 * inch))
        except Exception:
            story.append(Paragraph("(Image not available)", styles["Normal"]))

        story.append(Paragraph(result_text, styles["Normal"]))
        story.append(Spacer(1, 0.3 * inch))

    doc.build(story)

    # Prepare response
    buffer.seek(0)
    response = HttpResponse(buffer, content_type='application/pdf')
    response['Content-Disposition'] = 'attachment; filename="liveness_report.pdf"'
    return response


# ============================================================
# 🔹 Dashboard Summary
# ============================================================

def dashboard(request):
    total_verified = AnalysisLog.objects.filter(is_verified=True).count()
    total_correct = AnalysisLog.objects.filter(is_verified=True, is_correct=True).count()
    accuracy = (total_correct / total_verified * 100) if total_verified else 0

    total_real = AnalysisLog.objects.filter(result='real').count()
    total_fake = AnalysisLog.objects.filter(result='fake').count()

    return render(request, 'detector/dashboard.html', {
        'total_real': total_real,
        'total_fake': total_fake,
        'accuracy': round(accuracy, 2),
    })


# ============================================================
# (Optional) AJAX handler (if needed later)
# ============================================================

def mark_result(request, log_id, status):
    """
    تحديث نتيجة التحليل بناءً على تصحيح المستخدم
    status يمكن أن تكون 'correct' أو 'wrong'
    """
    try:
        log = AnalysisLog.objects.get(id=log_id)
        log.is_verified = True
        log.is_correct = (status == 'correct')
        log.save()
        return JsonResponse({'success': True})
    except AnalysisLog.DoesNotExist:
        return JsonResponse({'success': False, 'error': 'Not found'})
# في ملف views.py

def about_page(request):
    """Show the about page"""
    return render(request, 'detector/about.html') # تأكد من أن المسار صحيح لملف about.html