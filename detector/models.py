from django.db import models

class AnalysisLog(models.Model):
    image = models.ImageField(upload_to='uploads/')
    result = models.CharField(max_length=20, blank=True, null=True)  # 'real' or 'fake' or empty
    created_at = models.DateTimeField(auto_now_add=True)

    is_verified = models.BooleanField(default=False)
    is_correct = models.BooleanField(default=False)

    def __str__(self):
        return f"{self.result} - {self.created_at.strftime('%Y-%m-%d %H:%M:%S')}"
