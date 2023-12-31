import pandas as pd
import re
import string
import pickle
import string
import spacy
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import  WordNetLemmatizer
from duckduckgo_search import DDGS
####
import requests
from bs4 import BeautifulSoup

def download_nltk_resources():
    resources = ['punkt', 'stopwords', 'averaged_perceptron_tagger', 'maxent_ne_chunker', 'words']
    for resource in resources:
        try:
            nltk.data.find(resource)
        except LookupError:
            nltk.download(resource)
 
#SYSTEM DETEKCJI DEZINFORMACJI
def load_model(filename):
    try:
        with open(filename, 'rb') as file:  
            return pickle.load(file)
    except Exception as e:
        print(f"Error loading model: {e}")

def clean_and_tokenize(text):
    # Zamiana na małe litery
    text = text.lower()
    # Usunięcie białych znaków na początku i końcu tekstu
    text = text.strip()
    # Usuwanie sekwencji w kwadratowych nawiasach
    text = re.sub('\[.*?\]', '', text)
    # Zamiana znaków specjalnych na spacje
    text = re.sub("\\W", " ", text)
    # Usuwanie URL
    text = re.sub('https?://\S+|www\.\S+', '', text)
    # Usuwanie tagów HTML
    text = re.sub('<.*?>+', '', text)
    # Usuwanie znaków interpunkcyjnych
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    # Usuwanie znaków nowej linii
    text = re.sub('\n', '', text)
    # Usuwanie słów zawierających cyfry
    text = re.sub('\w*\d\w*', '', text)
    # Usuwanie nadmiarowych białych znaków
    text = re.sub(r'\s+', ' ', text)
    # Tokenizacja
    tokens = word_tokenize(text)
    # Lematyzacja
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word.isalnum() and word not in stop_words]
    return ' '.join(tokens)

def output_lable(n):
    if n == 0:
        return "FAKE NEWS"
    elif n == 1:
        return "NOT A FAKE NEWS"

def manual_testing_from_pkl(news, file_model, file_vec):
    testing_news = {"text": [news]}
    new_def_test = pd.DataFrame (testing_news)
    new_def_test["text"] = new_def_test["text"].apply(clean_and_tokenize)
    vec = load_model(file_vec)
    new_xv_test = vec.transform(new_def_test["text"].astype('str'))
    model = load_model(file_model)
    pred = model.predict(new_xv_test)
    return pred
#KONIEC SYSTEMU DETEKCJI DEZINFORMACJI


#SYSTEM ZMIERANIA DODATKOWYCH DANYCH
        #MODUŁ TWORZENIA ZAPYTAŃ
def word_labelling(news):
    nlp = spacy.load("en_core_web_sm")
    doc = nlp(news)
    unique_entities = {}
    labels_to_skip = ["CARDINAL", "DATE"]
    for ent in doc.ents:
        if ent.label_ in labels_to_skip:
            continue

        if ent.label_ not in unique_entities:
            unique_entities[ent.label_] = []
        if ent.text not in unique_entities[ent.label_]:
            unique_entities[ent.label_].append(ent.text)
    return unique_entities

def generate_search_queries(unique_entities):
    search_queries = []

    for label, entities in unique_entities.items():
        for entity in entities:
            query_url = create_google_query(entity, 'all')
            search_queries.append((entity, query_url))

    return search_queries

def create_google_query(keywords, search_in='all'):
    base_url = "https://www.google.com/search?q="

    query_templates = {
        'intext': 'intext:"{}"',
    }

    if search_in in query_templates:
        query = query_templates[search_in].format(keywords)
    elif search_in == 'all':
        query = ' OR '.join(template.format(keywords) for template in query_templates.values())
    else:
        raise ValueError("Invalid search_in value")

    query_url = base_url + query.replace(' ', '+')
    return query_url


def generate_combined_search_query(selected_entities):
    base_url = "https://www.google.com/search?q="
    queries = {
        'intext': [],
        'intitle': [],
        'inurl': []
    }
    for entity in selected_entities:
        for query_type in queries.keys():
            query_fragment = create_google_query_fragment(entity, query_type)
            queries[query_type].append(query_fragment)

    full_query_urls = []
    for query_type, fragments in queries.items():
        combined_query = ' AND '.join(fragments)
        full_query_url = base_url + combined_query.replace(' ', '+')
        full_query_urls.append(full_query_url)

    return '\n'.join(full_query_urls)


def create_google_query_fragment(keywords, search_in):
    query_templates = {
        'intext': 'intext:"{}"',
        'intitle': 'intitle:"{}"',
        'inurl': 'inurl:"{}"'
    }
    return query_templates[search_in].format(keywords)

#KONIEC MODUŁU TWORZENIA ZAPYTAŃ
#MODUŁ WYSZUKIWANIA INFORMACJI
'''
def search_link(queries):
    query = '+'.join(queries)
    url = f"https://duckduckgo.com/html/?q={query}"

    headers = {'User-Agent': 'Mozilla/5.0'}
    response = requests.get(url, headers=headers)

    soup = BeautifulSoup(response.text, 'html.parser')
    results_list = []

    for result in soup.find_all('div', class_='result', limit=5):
        title = result.find('a', class_='result__a').text
        link = result.find('a', class_='result__a')['href']
        results_list.append({'Title': title, 'URL': link})

    return results_list
'''
def search_link(queries):
    print("test")
    with DDGS() as ddgs:
        results_list = []
        query = ' intext:"{}"'.format('" AND "'.join(queries))

        for result in ddgs.text(query):
            results_list.append({
                'Title': result['title'],
                'URL': result['href']
            })
            print(results_list)
        return results_list
#KONIEC MODUŁU WYSZUKIWANIA INFORMACJI
#KONIEC SYSTEMU ZBIERANIA DODATKOWYCH DANYCH