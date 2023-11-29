from django.apps import AppConfig
from django import forms

class SystemConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'system'
    
class NewsForm(forms.Form):
    news_text = forms.CharField(widget=forms.Textarea(attrs={'placeholder': 'Wprowadź wiadomość do klasyfikacji'}), label='Wiadomość')
