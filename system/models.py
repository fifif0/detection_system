from django.db import models

class News(models.Model):
    content = models.TextField()
    result = models.TextField(max_length=100, default='Default Result')
    model_choice = models.CharField(max_length=100, default='Default Model')
    vector_choice = models.CharField(max_length=100, default='Default Vector')
    analyzed_date = models.DateTimeField(auto_now_add=True)
    entities = models.JSONField(default=dict)
    
    def __str__(self):
        return f"News analyzed on {self.analyzed_date}"


