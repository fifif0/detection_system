from django.shortcuts import render
from .models import News
from .forms import NewsForm
from django.http import JsonResponse
from .news_classifier import manual_testing_from_pkl, output_lable, word_labelling, generate_search_queries, generate_combined_search_query, search_link
import json

def index(request):
    return render(request, 'index.html')

def about(request):
    return render(request, 'about.html')

def get_label_from_path(choices, path):
    return next((label for (p, label) in choices if p == path), None)

def analyze_news(request):
    if request.method == 'POST':
        form = NewsForm(request.POST)
        if form.is_valid():
            news = form.cleaned_data['news_content']
            model_choice = form.cleaned_data['model_choice']
            vector_choice = form.cleaned_data['vector_choice']

            #Uzyskaj etykiety
            model_label = get_label_from_path(NewsForm.MODEL_CHOICES, form.cleaned_data['model_choice'])
            vector_label = get_label_from_path(NewsForm.VECTOR_CHOICES, form.cleaned_data['vector_choice'])

            result = manual_testing_from_pkl(news, model_choice, vector_choice)
            resultToTEXT = output_lable(result)

            unique_entities = word_labelling(news)
            search_queries = generate_search_queries(unique_entities)
            news_entry = News(
                content=news,
                result=resultToTEXT,
                model_choice=model_label,
                vector_choice=vector_label,
                entities=unique_entities
            )
            news_entry.save()

            return render(request, 'result.html', {
                'news': news,
                'result': resultToTEXT,
                'model_choice': model_label,
                'vector_choice': vector_label,
                'search_queries': search_queries
            })
    else:
        form = NewsForm()
    return render(request, 'analyze_news.html', {'form': form})


def generate_query(request):
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            selected_entities = data.get('selectedEntities')
            query_url = generate_combined_search_query(selected_entities)
            return JsonResponse({'query_url': query_url})
        except Exception as e:
            return JsonResponse({'error': str(e)}, status=500)
    else:
        return JsonResponse({'error': 'Invalid request'}, status=400)

def search_web(request):
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            selected_entity = data.get('selectedSearchEntities')
            search_results = search_link(selected_entity)
            return JsonResponse({'search_results': search_results})
        except Exception as e:
            return JsonResponse({'error': str(e)}, status=500)
    else:
        return JsonResponse({'error': 'Invalid request'}, status=400)
