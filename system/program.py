import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import re
import string
import pickle
import string
import joblib
from tqdm import tqdm
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, precision_score, recall_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, VotingClassifier
from sklearn.svm import SVC
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import  WordNetLemmatizer
from nltk.chunk import ne_chunk
from imblearn.under_sampling import NearMiss, RandomUnderSampler


def download_nltk_resources():
    resources = ['punkt', 'stopwords', 'averaged_perceptron_tagger', 'maxent_ne_chunker', 'words']
    for resource in resources:
        try:
            nltk.data.find(resource)
        except LookupError:
            nltk.download(resource)

def save_model(model, filename):
    try:
        with open(filename, 'wb') as file:  
            pickle.dump(model, file)
    except Exception as e:
        print(f"Error saving model: {e}")

def load_model(filename):
    try:
        with open(filename, 'rb') as file:  
            return pickle.load(file)
    except Exception as e:
        print(f"Error loading model: {e}")

def load_data(fake_csv, true_csv):
    data_fake = pd.read_csv(fake_csv)
    data_true = pd.read_csv(true_csv)
    return data_fake, data_true

def preparing_data(data_fake, data_true):
    data_fake['class'] = 0
    data_true['class'] = 1
    data = pd.concat([data_fake, data_true], ignore_index=True)
    data = data.drop(['date'], axis = 1)
    data = shuffle(data, random_state=42)
    data = data.drop_duplicates(keep='first')
    data = data.dropna()
    data.reset_index(drop=True, inplace=True)
    '''
    rus = RandomUnderSampler(random_state=42)
    x_res, y_res = rus.fit_resample(data[['text']], data['class'])
    data_resampled = pd.DataFrame(x_res, columns=['text'])
    data_resampled['class'] = y_res
    
    nearmiss = NearMiss()
    x_res, y_res = nearmiss.fit_resample(data[['text']], data['class'])
    data_resampled = pd.DataFrame(x_res, columns=['text'])
    data_resampled['class'] = y_res'''
    return data

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

def split_data(x, y):
    x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.2, random_state=42)
    return x_train, x_test, y_train, y_test

#___________________________________________________

def train_models(x_train, y_train):
    models = {
        'RandomForestClassifier': RandomForestClassifier(),
        'LogisticRegression': LogisticRegression(),
        'DecisionTreeClassifier': DecisionTreeClassifier(),
        'GradientBoostingClassifier': GradientBoostingClassifier(random_state = 0),
        'SVC': SVC()
    }

    trained_models = {}
    for name, model in models.items():
        model.fit(x_train, y_train)
        trained_models[name] = model
        save_model(model, f'{name}_model.pkl')
    return trained_models

#___________________________________________________

def prepare_vectorizer(x_train, x_test):
    vectorization = TfidfVectorizer(ngram_range=(1,2), max_df=0.7, min_df=10)
    _train = vectorization.fit_transform(x_train)
    _test = vectorization.transform(x_test)
    save_model(vectorization, "TfidfVectorizer.pkl")
    return _train, _test, vectorization

def train_model_random_forest(x_train, y_train):
    model = RandomForestClassifier(random_state = 0)
    model.fit(x_train, y_train)
    save_model(model, "RandomForest.pkl")
    return model

def train_model_logistic_regression(x_train, y_train):
    model = LogisticRegression()
    model.fit(x_train, y_train)
    save_model(model, "LogisticRegression.pkl")
    return model

def train_model_decision_tree(x_train, y_train):
    model = DecisionTreeClassifier()
    model.fit(x_train, y_train)
    save_model(model, "DecisionTree.pkl")
    return model

def train_model_gradient_boosting(x_train, y_train):
    model = GradientBoostingClassifier()
    model.fit(x_train, y_train)
    save_model(model, "GradientBoosting.pkl")
    return model

def train_model_svc(x_train, y_train):
    model = SVC()
    model.fit(x_train, y_train)
    save_model(model, "SVC.pkl")
    return model

def to_manual_testing(data_fake, data_true):
    data_fake_manual_testing = data_fake.tail(10)
    for i in range(23480, 23470,-1):
        data_fake.drop([i], axis = 0, inplace = True)

    data_true_manual_testing = data_true.tail(10)
    for i in range(21416, 21406,-1):
        data_fake.drop([i], axis = 0, inplace = True)

    data_fake_manual_testing["class"] = 0
    data_true_manual_testing["class"] = 1
    return data_fake_manual_testing, data_true_manual_testing

def evaluate_model(model, x_test, y_test):
    y_pred = model.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    class_report = classification_report(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    print(f"Accuracy: {accuracy * 100:.2f}%")
    print(f"Precision: {precision * 100:.2f}%")
    print(f"Recall: {recall * 100:.2f}%")
    print("\nClassification Report:\n", class_report)
    '''
     sns.heatmap(conf_matrix, annot=True, fmt='d')
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.show()
    '''
   
    return y_pred

#___________________________________________________

class Evaluation:
    
    def __init__(self,model,x_train,x_test,y_train,y_test):
        self.model = model
        self.x_train = x_train
        self.x_test = x_test
        self.y_train = y_train
        self.y_test = y_test
        
    def train_evaluation(self):
        y_pred_train = self.model.predict(self.x_train)
        
        acc_scr_train = accuracy_score(self.y_train,y_pred_train)
        print("Accuracy Score On Training Data Set :",acc_scr_train)
        print()
        
        con_mat_train = confusion_matrix(self.y_train,y_pred_train)
        print("Confusion Matrix On Training Data Set :\n",con_mat_train)
        print()
        
        class_rep_train = classification_report(self.y_train,y_pred_train)
        print("Classification Report On Training Data Set :\n",class_rep_train)
        
        
    def test_evaluation(self):
        y_pred_test = self.model.predict(self.x_test)
        
        acc_scr_test = accuracy_score(self.y_test,y_pred_test)
        print("Accuracy Score On Testing Data Set :",acc_scr_test)
        print()
        
        con_mat_test = confusion_matrix(self.y_test,y_pred_test)
        print("Confusion Matrix On Testing Data Set :\n",con_mat_test)
        print()
        
        class_rep_test = classification_report(self.y_test,y_pred_test)
        print("Classification Report On Testing Data Set :\n",class_rep_test)

#___________________________________________________

def output_lable(n):
    if n == 0:
        return "Fake News"
    elif n == 1:
        return "Not A Fake News"

def manual_testing(news, vectorization, model):
    testing_news = {"text": [news]}
    new_def_test = pd.DataFrame (testing_news)
    new_def_test["text"] = new_def_test["text"].progress_apply(clean_and_tokenize)
    new_xv_test = vectorization.transform(new_def_test["text"].astype('str'))
    pred = model.predict(new_xv_test)
    return print("\nPrediction: {}".format(output_lable(pred)))

def manual_testing_from_pkl(news, file_model, file_vec):
    testing_news = {"text": [news]}
    new_def_test = pd.DataFrame (testing_news)
    new_def_test["text"] = new_def_test["text"].progress_apply(clean_and_tokenize)
    vec = load_model(file_vec)
    new_xv_test = vec.transform(new_def_test["text"].astype('str'))
    model = load_model(file_model)
    pred = model.predict(new_xv_test)
    return print("\nPrediction: {}".format(output_lable(pred)))


def main_menu():
    print("Wybierz algorytm do testowania:")
    print("1: Regresja Logistyczna")
    print("2: Las Losowy")
    print("3: Drzewo Decyzyjne")
    print("4: Gradient Boosting")
    print("5: Support Vector Classifier")
    print("0: Wyjście z programu")
    choice = input("Wprowadź numer opcji: ")
    return choice

def second_menu(model_name, model, vectorizer):
    while True:
        news_to_test = input(f"Wprowadź tekst do testowania dla {model_name} (lub wpisz 'back' aby wrócić lub 'exit' aby zakończyć): ")
        if news_to_test.lower() == 'back':
            return
        elif news_to_test.lower() == 'exit':
            print("Zakończono testowanie.")
            exit()
        else:
            manual_testing_from_pkl(news_to_test, model, vectorizer)

def evaluate_and_test_models():
    while True:
        choice = main_menu()
        if choice == "1":
            second_menu("Logistic Regression", "LogisticRegression.pkl", "TfidfVectorizer.pkl")
        elif choice == "2":
            second_menu("Random Forest", "RandomForest.pkl", "TfidfVectorizer.pkl")
        elif choice == "3":
            second_menu("Decision Tree", "DecisionTree.pkl", "TfidfVectorizer.pkl")
        elif choice == "4":
            second_menu("Gradient Boosting", "GradientBoosting.pkl", "TfidfVectorizer.pkl")
        elif choice == "5":
            second_menu("Support Vector Classifier", "SVC.pkl", "TfidfVectorizer.pkl")
        elif choice == "0":
            print("Zamykanie programu...")
            break
        else:
            print("Niepoprawna opcja, spróbuj ponownie.")



def main():
    #pobieranie potrzebnych pakietów
    #download_nltk_resources()
    tqdm.pandas()
    #Dane
    fake_csv_path = 'C:\\Users\\admin\\Desktop\\praca\\NOWYTEMAT\\Fake.csv'
    true_csv_path = 'C:\\Users\\admin\\Desktop\\praca\\NOWYTEMAT\\True.csv'

    print('Wczytywanie danych\n')
    data_fake, data_true = load_data(fake_csv_path, true_csv_path)
    print(data_fake, data_true)

    print('Przygotowanie danych\n')
    data = preparing_data(data_fake, data_true)
    print(data)
    
    print('Czyszczenie danych\n')
    data["text"] = data["text"].progress_apply(clean_and_tokenize)
    #data["title"] = data["title"].apply(clean_and_tokenize)
    #data["subject"] = data["subject"].apply(clean_and_tokenize)
    print(data)

    print('Podział danych\n')
    x = data['text']
    y = data['class']
    x_train, x_test, y_train, y_test = split_data(x, y)

    #print('Wektoryzacja danych\n')
    #train_vec, test_vec, vectorization = prepare_vectorizer(x_train, x_test)
 
    #print('Balansowanie danych treningowych\n')
    #nm = NearMiss()
    #x_train_res, y_train_res = nm.fit_resample(train_vec, y_train)

    '''
    print('Wektoryzacja modeli\n')
    logistic_regression = train_model_logistic_regression(x_train_res, y_train_res)
    print('Wektoryzacja modelu logistic_regression zakończona\n')
    decision_tree = train_model_decision_tree(x_train_res, y_train_res)
    print('Wektoryzacja modelu decision_tree zakończona\n')
    random_forest = train_model_random_forest(x_train_res, y_train_res)
    print('Wektoryzacja modelu random_forest zakończona\n')
    gradient_boosting = train_model_gradient_boosting(x_train_res, y_train_res)
    print('Wektoryzacja modelu gradient_boosting zakończona\n')
    svc = train_model_svc(x_train_res, y_train_res)
    print('Wektoryzacja modelu svc zakończona\n')
    ''' 
    '''
    print('Ewaluacja modeli\n')
    evaluate_model(logistic_regression, test_vec, y_test)
    print('Ewaluacja modelu logistic_regression zakończona\n')
    evaluate_model(random_forest, test_vec, y_test)
    print('Ewaluacja modelu random_forest zakończona\n')
    evaluate_model(decision_tree, test_vec, y_test)
    print('Ewaluacja modelu decision_tree zakończona\n')
    evaluate_model(gradient_boosting, test_vec, y_test)
    print('Ewaluacja modelu gradient_boosting zakończona\n')
    evaluate_model(svc, test_vec, y_test)
    print('Ewaluacja modelu svc zakończona\n')
    '''

    evaluate_and_test_models()

if __name__ == "__main__":
    main()