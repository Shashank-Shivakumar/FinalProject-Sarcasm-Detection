import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, f1_score, classification_report
from lime.lime_text import LimeTextExplainer
from sklearn.pipeline import make_pipeline
import nltk
from nltk.corpus import stopwords
import pickle

#Analysis Text Classification,model training and evaluation, and model explanation using LIME

nltk.download('stopwords')
nltk.download('wordnet')

# Function to process text data
def process_text(data):
    lemma = nltk.WordNetLemmatizer()
    stopwords_set = set(stopwords.words('english'))
    processed_data = []
    for sent in data:
        words = [lemma.lemmatize(word) for word in sent.split() if word not in stopwords_set]
        processed_data.append(' '.join(words))
    return processed_data


# Function to evaluate the model
def evaluate_model(clf, X_train, y_train, X_test, y_test, name):
    y_train_pred = clf.predict(X_train)
    y_test_pred = clf.predict(X_test)
    print(f"{name} Train Accuracy: {accuracy_score(y_train, y_train_pred)}")
    print(f"{name} Train F1-score: {f1_score(y_train, y_train_pred)}")
    print(f"{name} Test Accuracy: {accuracy_score(y_test, y_test_pred)}")
    print(f"{name} Test F1-score: {f1_score(y_test, y_test_pred)}")
    print(f"Classification Report:\n{classification_report(y_test, y_test_pred)}")
    filename = f'finalized_{name}_model.sav'
    pickle.dump(clf, open(filename, 'wb'))

# Paths to the dataset files
train_file_path = 'Sarcasm_Headlines_Dataset.json'
test_file_path = 'Sarcasm_Headlines_Dataset_v2.json'
df_train = pd.read_json(train_file_path, lines=True)
df_test = pd.read_json(test_file_path, lines=True)

df = pd.concat([df_train, df_test], axis=0)

# Split data
df_train, df_test = train_test_split(df, test_size=0.2, stratify=df['is_sarcastic'], shuffle=True)
X_train, y_train = process_text(df_train['headline']), df_train['is_sarcastic']
X_test, y_test = process_text(df_test['headline']), df_test['is_sarcastic']

# Vectorization
tfidf = TfidfVectorizer()
X_train = tfidf.fit_transform(X_train)
X_test = tfidf.transform(X_test)

# Train Logistic Regression model
clf_lr = LogisticRegression(n_jobs=1, C=1e5)
clf_lr.fit(X_train, y_train)
evaluate_model(clf_lr, X_train, y_train, X_test, y_test, 'LR')

# Train Naive Bayes model
clf_nb = MultinomialNB()
clf_nb.fit(X_train, y_train)
evaluate_model(clf_nb, X_train, y_train, X_test, y_test, 'NB')

# LIME explanation
num_examples = 3

for _ in range(num_examples):
    idx = np.random.randint(0, len(df_test))
    headline = df_test.iloc[idx]["headline"]

    # Explain Logistic Regression
    c_lr = make_pipeline(tfidf, clf_lr)
    class_names = ["Non Sarcastic", "Sarcastic"]
    explainer_lr = LimeTextExplainer(class_names=class_names)
    exp_lr = explainer_lr.explain_instance(headline, c_lr.predict_proba, num_features=10)
    print(f"Logistic Regression - Example {idx}")
    print("Headline:", headline)
    print("Probability (Non sarcastic) =", c_lr.predict_proba([headline])[0, 1])
    print("Probability (sarcastic) =", c_lr.predict_proba([headline])[0, 0])
    print("True Class:", class_names[df_test.iloc[idx]["is_sarcastic"]])
    exp_lr.as_pyplot_figure()
    plt.title(f"LIME Explanation for Example {idx} - Logistic Regression")
    plt.savefig("LIME_explanation_logistic.png")
    print(f"Plot saved as LIME_explanation_logistic.png")
    plt.show()

    # Explain Naive Bayes
    c_nb = make_pipeline(tfidf, clf_nb)
    explainer_nb = LimeTextExplainer(class_names=class_names)
    exp_nb = explainer_nb.explain_instance(headline, c_nb.predict_proba, num_features=10)

    print(f"Naive Bayes - Example {idx}")
    print("Headline:", headline)
    print("Probability (Non sarcastic) =", c_nb.predict_proba([headline])[0, 1])
    print("Probability (sarcastic) =", c_nb.predict_proba([headline])[0, 0])
    print("True Class:", class_names[df_test.iloc[idx]["is_sarcastic"]])

    exp_nb.as_pyplot_figure()
    plt.title(f"LIME Explanation for Example {idx} - Naive Bayes")
    plt.savefig("LIME_explanation_naive.png")
    print(f"Plot saved as LIME_explanation_naive.png")
    plt.show()
