from django.http import HttpResponseRedirect
from django.shortcuts import render
from django import forms
from django.conf import settings
import os

class NewsForm(forms.Form):

    MODEL_PATH = os.path.join(settings.BASE_DIR, 'models')
    VEC_PATH = os.path.join(settings.BASE_DIR, 'vector')

    MODEL_CHOICES = [
        (os.path.join(MODEL_PATH, 'LogisticRegression.pkl'), 'Logistic Regression'),
        (os.path.join(MODEL_PATH, 'RandomForest.pkl'),  'Random Forest'),
        (os.path.join(MODEL_PATH, 'DecisionTree.pkl'), 'Decision Tree'),
        (os.path.join(MODEL_PATH, 'GradientBoosting.pkl'), 'Gradient Boosting'),
        (os.path.join(MODEL_PATH, 'SupportVectorMachine.pkl'), 'Support Vector Machine')
    ]

    VECTOR_CHOICES = [
        (os.path.join(VEC_PATH, 'TfidfVectorizer.pkl'), 'TF-IDF'),
    ]   


    news_content = forms.CharField(
    widget=forms.Textarea(attrs={'class': 'form-class'}),
    label='Wprowadź wiadomość do analizy:'
    )
    model_choice = forms.ChoiceField(
    choices=MODEL_CHOICES,
    label='Wybierz model uczenia maszynowego:',
    widget=forms.Select(attrs={'class': 'form-class'})
    )
    vector_choice = forms.ChoiceField(
    choices=VECTOR_CHOICES,
    label='Wybierz sposób wektoryzacji:',
    widget=forms.Select(attrs={'class': 'form-class'})
    )