# 📁 detector/views.py

from django.shortcuts import render, redirect
from django.conf import settings
from .models import AnalysisLog
from django.core.files.storage import default_storage
from .inference import predict_liveness  # دالة التحليل باستخدام نموذج ONNX
import os

from django.http import HttpResponse
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import inch
from datetime import datetime
import io

# --------------------------------------------------------
# 🧱 عرض صفحة رفع الصور
# --------------------------------------------------------
def upload_page(request):
    """
    تعرض للمستخدم صفحة فيها زر اختيار الصور.
    """
    return render(request, 'detector/upload.html')


# --------------------------------------------------------
# 🧠 معالجة الصور المرفوعة وتحليلها باستخدام النموذج
# --------------------------------------------------------
def process_uploads(request):
    """
    عند إرسال المستخدم للصور:
    - نحفظ كل صورة مؤقتًا في مجلد uploads داخل MEDIA_ROOT
    - نمرر كل صورة إلى النموذج للتحليل
    - نحفظ النتيجة (real/fake) في قاعدة البيانات
    """
    if request.method == 'POST':
        files = request.FILES.getlist('images')

        for f in files:
            # 🖼️ حفظ الصورة مؤقتًا في مجلد media/uploads/
            path = default_storage.save(f'uploads/{f.name}', f)
            full_path = os.path.join(settings.MEDIA_ROOT, path)

            # 🔍 تحليل الصورة بالنموذج (resnet50.onnx)
            result = predict_liveness(full_path)

            # 🗃️ حفظ السجل في قاعدة البيانات
            AnalysisLog.objects.create(image=path, result=result)

        # ✅ بعد المعالجة ننتقل إلى صفحة النتائج
        return redirect('detector:results_page')

    # في حال الدخول إلى الصفحة مباشرة بدون POST
    return redirect('detector:upload_page')


# --------------------------------------------------------
# 📊 عرض صفحة النتائج
# --------------------------------------------------------
def results_page(request):
    """
    تعرض آخر 50 نتيجة تحليل (صورة + النتيجة)
    """
    logs = AnalysisLog.objects.order_by('-created_at')[:50]
    return render(request, 'detector/results.html', {'logs': logs})


# --------------------------------------------------------
# 🧹 مسح النتائج القديمة (Reset)
# --------------------------------------------------------
def clear_results(request):
    """
    عند الضغط على زر "Clear Results":
    - نحذف كل السجلات السابقة من قاعدة البيانات
    """
    AnalysisLog.objects.all().delete()
    return redirect('detector:results_page')




# --------------------------------------------------------
# 🧾 إنشاء تقرير PDF للنتائج الحالية
# --------------------------------------------------------
def download_report(request):
    """
    ينشئ تقرير PDF يحتوي على:
    - عنوان المشروع
    - تاريخ الإنشاء
    - كل الصور ونتائجها
    """

    # إنشاء استجابة فارغة لتخزين الملف الناتج
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4)

    # أنماط التنسيق
    styles = getSampleStyleSheet()
    story = []

    # 🏷️ العنوان الرئيسي
    story.append(Paragraph("<b>Face Liveness Detection Report</b>", styles["Title"]))
    story.append(Spacer(1, 0.2 * inch))
    story.append(Paragraph(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", styles["Normal"]))
    story.append(Spacer(1, 0.3 * inch))

    # جلب آخر 50 نتيجة من قاعدة البيانات
    logs = AnalysisLog.objects.order_by('-created_at')[:50]

    # ✅ إدراج كل صورة مع النتيجة
    for log in logs:
        img_path = os.path.join(settings.MEDIA_ROOT, str(log.image))
        result_text = f"<b>Result:</b> {'✅ Real' if log.result == 'real' else '❌ Fake'}"
        story.append(Image(img_path, width=2*inch, height=2*inch))
        story.append(Paragraph(result_text, styles["Normal"]))
        story.append(Spacer(1, 0.3 * inch))

    # توليد التقرير
    doc.build(story)

    # تجهيز الاستجابة النهائية
    buffer.seek(0)
    response = HttpResponse(buffer, content_type='application/pdf')
    response['Content-Disposition'] = 'attachment; filename="liveness_report.pdf"'
    return response