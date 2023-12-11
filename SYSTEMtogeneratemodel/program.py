import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import re
import string
import pickle
import string
from tqdm import tqdm
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, precision_score, recall_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.svm import SVC
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import  WordNetLemmatizer
from imblearn.under_sampling import NearMiss

#Instalacja zasobów potrzebnych do działania programu
def download_nltk_resources():
    resources = ['punkt', 'stopwords', 'averaged_perceptron_tagger', 'maxent_ne_chunker', 'words']
    for resource in resources:
        try:
            nltk.data.find(resource)
        except LookupError:
            nltk.download(resource)

#Zapisywanie modelu
def save_model(model, filename):
    try:
        with open(filename, 'wb') as file:  
            pickle.dump(model, file)
    except Exception as e:
        print(f"Error saving model: {e}")

#Wczytywanie modelu
def load_model(filename):
    try:
        with open(filename, 'rb') as file:  
            return pickle.load(file)
    except Exception as e:
        print(f"Error loading model: {e}")

#Wczytywanie danych
def load_data(fake_csv, true_csv):
    data_fake = pd.read_csv(fake_csv)
    data_true = pd.read_csv(true_csv)
    return data_fake, data_true

#Wstępne przygotowanie danych
def preparing_data(data_fake, data_true):
    data_fake['class'] = 0
    data_true['class'] = 1
    data = pd.concat([data_fake, data_true], ignore_index=True)
    data = data.drop(['date'], axis = 1)
    data = shuffle(data, random_state=42)
    data = data.drop_duplicates(keep='first')
    data = data.dropna()
    data.reset_index(drop=True, inplace=True)
    return data

#Czyszczenie danych
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

#Podział danych
def split_data(x, y):
    x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.2, random_state=42)
    return x_train, x_test, y_train, y_test

#Tworzenie wektoryzatora i przeprowadzenie wektoryzacji
def prepare_vectorizer(x_train, x_test):
    vectorization = TfidfVectorizer(ngram_range=(1,2), max_df=0.7, min_df=10)
    _train = vectorization.fit_transform(x_train)
    _test = vectorization.transform(x_test)
    save_model(vectorization, "//home//ubuntu//apraca-inz//system_detection//vector//TfidfVectorizer.pkl")
    return _train, _test, vectorization

#Trenowanie modelu klasyfikacyjnego - Random Forest
def train_model_random_forest(x_train, y_train):
    model = RandomForestClassifier(random_state = 0)
    model.fit(x_train, y_train)
    save_model(model, "//home//ubuntu//apraca-inz//system_detection//models//RandomForest.pkl")
    return model

#Trenowanie modelu klasyfikacyjnego - Logistic Regression
def train_model_logistic_regression(x_train, y_train):
    model = LogisticRegression()
    model.fit(x_train, y_train)
    save_model(model, "//home//ubuntu//apraca-inz//system_detection//models//LogisticRegression.pkl")
    return model

#Trenowanie modelu klasyfikacyjnego - Decision Tree
def train_model_decision_tree(x_train, y_train):
    model = DecisionTreeClassifier()
    model.fit(x_train, y_train)
    save_model(model, "//home//ubuntu//apraca-inz//system_detection//models//DecisionTree.pkl")
    return model

#Trenowanie modelu klasyfikacyjnego - Gradient Boosting
def train_model_gradient_boosting(x_train, y_train):
    model = GradientBoostingClassifier()
    model.fit(x_train, y_train)
    save_model(model, "//home//ubuntu//apraca-inz//system_detection//models//GradientBoosting.pkl")
    return model

#Trenowanie modelu klasyfikacyjnego - Support Vector Machine
def train_model_svc(x_train, y_train):
    model = SVC()
    model.fit(x_train, y_train)
    save_model(model, "//home//ubuntu//apraca-inz//system_detection//models//SupportVectorMachine.pkl")
    return model

#evaluacja modelu
def evaluate_model(model, x_test, y_test, file_txt, file_png):
    y_pred = model.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    class_report = classification_report(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)

    with open(file_txt, "w") as file:
        file.write(f"Accuracy: {accuracy * 100:.2f}%\n")
        file.write(f"Precision: {precision * 100:.2f}%\n")
        file.write(f"Recall: {recall * 100:.2f}%\n")
        file.write("\nClassification Report:\n")
        file.write(class_report)

    print(f"Accuracy: {accuracy * 100:.2f}%")
    print(f"Precision: {precision * 100:.2f}%")
    print(f"Recall: {recall * 100:.2f}%")
    print("\nClassification Report:\n", class_report)
    sns.heatmap(conf_matrix, annot=True, fmt='d')
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.savefig(file_png)
    return y_pred

#Nadawanie etykiety
def output_lable(n):
    if n == 0:
        return "Fake News"
    elif n == 1:
        return "Not A Fake News"

#Test manualny wytrenowanego modelu
def manual_testing(news, vectorization, model):
    testing_news = {"text": [news]}
    new_def_test = pd.DataFrame (testing_news)
    new_def_test["text"] = new_def_test["text"].progress_apply(clean_and_tokenize)
    new_xv_test = vectorization.transform(new_def_test["text"].astype('str'))
    pred = model.predict(new_xv_test)
    return print("\nPrediction: {}".format(output_lable(pred)))

#Test manualny zapisanego modelu
def manual_testing_from_pkl(news, file_model, file_vec):
    testing_news = {"text": [news]}
    new_def_test = pd.DataFrame (testing_news)
    new_def_test["text"] = new_def_test["text"].progress_apply(clean_and_tokenize)
    vec = load_model(file_vec)
    new_xv_test = vec.transform(new_def_test["text"].astype('str'))
    model = load_model(file_model)
    pred = model.predict(new_xv_test)
    return print("\nPrediction: {}".format(output_lable(pred)))

#Główne menu wyboru modelu
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

#Drugie menu 
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

#Przekierowanie do odpowiedniego modelu
def evaluate_and_test_models():
    while True:
        choice = main_menu()
        if choice == "1":
            second_menu("Logistic Regression", "//home//ubuntu//apraca-inz//system_detection//models//LogisticRegression.pkl", "//home//ubuntu//apraca-inz//system_detection//vector//TfidfVectorizer.pkl")
        elif choice == "2":
            second_menu("Random Forest", "//home//ubuntu//apraca-inz//system_detection//models//RandomForest.pkl", "//home//ubuntu//apraca-inz//system_detection//vector//TfidfVectorizer.pkl")
        elif choice == "3":
            second_menu("Decision Tree", "//home//ubuntu//apraca-inz//system_detection//models//DecisionTree.pkl", "//home//ubuntu//apraca-inz//system_detection//vector//TfidfVectorizer.pkl")
        elif choice == "4":
            second_menu("Gradient Boosting", "//home//ubuntu//apraca-inz//system_detection//models//GradientBoosting.pkl", "//home//ubuntu//apraca-inz//system_detection//vector//TfidfVectorizer.pkl")
        elif choice == "5":
            second_menu("Support Vector Machine", "//home//ubuntu//apraca-inz//system_detection//models//SupportVectorMachine.pkl", "//home//ubuntu//apraca-inz//system_detection//vector//TfidfVectorizer.pkl")
        elif choice == "0":
            print("Zamykanie programu...")
            break
        else:
            print("Niepoprawna opcja, spróbuj ponownie.")


#funkcja główna
def main():
    tqdm.pandas()
    

    print('Pobieranie potrzebnych pakietów\n')
    download_nltk_resources()

    #Dane
    fake_csv_path = '//home//ubuntu//apraca-inz//system_detection//SYSTEMtogeneratemodel//Datasets//Fake.csv'
    true_csv_path = '//home//ubuntu//apraca-inz//system_detection//SYSTEMtogeneratemodel//Datasets//True.csv'

    print('Wczytywanie danych\n')
    data_fake, data_true = load_data(fake_csv_path, true_csv_path)
    print(data_fake, data_true)

    print('Przygotowanie danych\n')
    data = preparing_data(data_fake, data_true)
    print(data)
    
    print('Czyszczenie danych\n')
    data["text"] = data["text"].progress_apply(clean_and_tokenize)
    print(data)

    print('Podział danych\n')
    x = data['text']
    y = data['class']
    x_train, x_test, y_train, y_test = split_data(x, y)

    print('Wektoryzacja danych\n')
    train_vec, test_vec, vectorization = prepare_vectorizer(x_train, x_test)

    print('Balansowanie danych treningowych\n')
    nm = NearMiss()
    x_train_res, y_train_res = nm.fit_resample(train_vec, y_train)

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
    print('Wektoryzacja modelu Support Vector Machine zakończona\n')
   
    print('Ewaluacja modeli\n')
    evaluate_model(logistic_regression, test_vec, y_test, "LogisticRegression.txt", "LogisticRegression.png")
    print('Ewaluacja modelu logistic_regression zakończona\n')

    evaluate_model(random_forest, test_vec, y_test, "RandomForest.txt", "RandomForest.png")
    print('Ewaluacja modelu random_forest zakończona\n')
    evaluate_model(decision_tree, test_vec, y_test, "DecisionTree.txt", "DecisionTree.png")
    print('Ewaluacja modelu decision_tree zakończona\n')
    evaluate_model(gradient_boosting, test_vec, y_test, "GradientBoosting.txt", "GradientBoosting.png")
    print('Ewaluacja modelu gradient_boosting zakończona\n')
    evaluate_model(svc, test_vec, y_test, "SupportVectorMachine.txt", "SupportVectorMachine.png")
    print('Ewaluacja modelu Support Vector Machine zakończona\n')
   
    evaluate_and_test_models()

if __name__ == "__main__":
    main()