import numpy as np
import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, f1_score, classification_report
import lime
from lime.lime_text import LimeTextExplainer
import nltk
from sklearn.pipeline import make_pipeline
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import pickle
from transformers import pipeline
import os
# from text_summarizer import get_summary
# Ensure the necessary NLTK downloads
nltk.download('stopwords')
nltk.download('wordnet')


# Function to process text data
def process_text(data):
    lemma = nltk.WordNetLemmatizer()
    stopwords_set = set(stopwords.words('english'))
    processed_data = []
    for sent in data:
        words = [lemma.lemmatize(word) for word in sent.split() if word.lower() not in stopwords_set]
        processed_data.append(' '.join(words))
    return processed_data


# Function to evaluate the model and return metrics
def evaluate_model(clf, X_train, y_train, X_test, y_test):
    y_train_pred = clf.predict(X_train)
    y_test_pred = clf.predict(X_test)
    train_accuracy = accuracy_score(y_train, y_train_pred)
    test_accuracy = accuracy_score(y_test, y_test_pred)
    train_f1 = f1_score(y_train, y_train_pred)
    test_f1 = f1_score(y_test, y_test_pred)
    report = classification_report(y_test, y_test_pred)
    return train_accuracy, train_f1, test_accuracy, test_f1, report


# Function for text summarization
# def text_summarization(text):
#     try:
#         model_path = os.path.abspath("./Code/t5-sarcastic-headline-generator")
#         summarizer = pipeline("summarization", model=model_path)
#         summarized_text = summarizer(text, min_length=10, max_length=30)[0]['summary_text']
#         return summarized_text
#     except Exception as e:
#         return f"Error in summarization: {str(e)}"

def text_summarization(text):
    try:
        model_path = os.path.abspath("./t5-sarcastic-headline-generator")
        summarizer = pipeline("summarization", model=model_path)
        summarized_text = summarizer(text, min_length=10, max_length=30)[0]['summary_text']
        return summarized_text
    except Exception as e:
        return f"Error in summarization: {str(e)}"

# Function for sarcasm prediction
def predict_sarcasm(text, clf, tfidf):
    processed_text = process_text([text])
    tfidf_text = tfidf.transform(processed_text)
    prediction = clf.predict(tfidf_text)
    return "Sarcastic" if prediction[0] == 1 else "Non Sarcastic"


# Main function for the Streamlit app
def main():
    st.title("Sarcasm Detection in Headlines")

    # Project Overview
    st.header("Project Overview")
    st.write(
        "This project aims to detect sarcasm in news headlines using machine learning. It's an exploration into natural language processing and sentiment analysis.")

    # Sidebar with Project Info
    st.sidebar.header("Project Information")
    st.sidebar.markdown("""
        - **Project Title:** Sarcasm Detection Project
        - **Presented By:**  
            Namratha Prakash<br>
            Shashank Shivakumar
        - **Instructor:** Ning Rui
    """, unsafe_allow_html=True)

    # Dataset Display
    st.header("Dataset Exploration")
    uploaded_file = st.file_uploader("Upload your dataset (JSON format)", type=["json"])
    if uploaded_file is not None:
        df = pd.read_json(uploaded_file, lines=True)
        if st.checkbox("Show dataset head"):
            st.dataframe(df.head())

        # Data Analysis
        st.subheader("Data Analysis and Visualization")
        if st.checkbox("Show data analysis"):
            st.subheader("Sarcasm Count in Dataset")
            sarcasm_count = df['is_sarcastic'].value_counts()
            st.bar_chart(sarcasm_count)
            st.subheader("Top Words in Sarcastic vs Non-Sarcastic Headlines")
            st.image("Top Words in Sarcastic Headlines.png", caption="Top Words in Sarcastic Headlines")
            st.image("Top Words in Non-Sarcastic Headlines.png", caption="Top Words in Non-Sarcastic Headlines")

            # Average Word Length Visualization
            st.subheader("Average Word Length in Sarcastic vs Non-Sarcastic Text")
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
            word = df[df['is_sarcastic'] == 1]['headline'].str.split().apply(lambda x: [len(i) for i in x])
            sns.distplot(word.map(lambda x: np.mean(x)), ax=ax1, color='red')
            ax1.set_title('Sarcastic Text')
            word = df[df['is_sarcastic'] == 0]['headline'].str.split().apply(lambda x: [len(i) for i in x])
            sns.distplot(word.map(lambda x: np.mean(x)), ax=ax2, color='green')
            ax2.set_title('Not Sarcastic Text')
            fig.suptitle('Average Word Length in Each Text')
            st.pyplot(fig)

    # Model Training and Evaluation
    st.header("Model Training and Evaluation")
    if uploaded_file is not None:
        model_option = st.selectbox("Choose your model",
                                    ['Logistic Regression', 'Naive Bayes', 'LSTM', 'BERT', 'BertLSTM', 'BERTMLP',
                                     'RoBERTa'])

        if model_option in ['LSTM', 'BERT', 'BertLSTM', 'BERTMLP', 'RoBERTa', 'Text Summarization']:
            st.markdown("**Note:** Advanced models are demonstrated using pre-trained results.")

            # Replace 'path_to_your_screenshot.png' with the actual path or URL to your screenshot
            if model_option == 'BERT':
                st.image("BertRNN.png", caption="BERT Model Results")
            elif model_option == 'LSTM':
                st.image("lstm.png", caption="LSTM Model Results")
            elif model_option == 'BertLSTM':
                st.image("BertLSTM.png", caption="BERTLSTM Model Results")
            elif model_option == 'BERTMLP':
                st.image("BertMLP.png", caption="BERTMLP Model Results")
            elif model_option == 'RoBERTa':
                st.image("RoBerta.png", caption="RoBERTa Model Results")
            # elif model_option == 'Text Summarization':
            #     st.image("summ_rouge_vs_epoch.png", caption="Text Summarization Model Results")
        else:
            tfidf = TfidfVectorizer()
            if st.button("Train Model"):
                processed_data = process_text(df['headline'])
                labels = df['is_sarcastic']
                X_train, X_test, y_train, y_test = train_test_split(processed_data, labels, test_size=0.2,
                                                                    random_state=42)
                X_train = tfidf.fit_transform(X_train)
                X_test = tfidf.transform(X_test)

                if model_option == 'Logistic Regression':
                    clf = LogisticRegression(n_jobs=1, C=1e5)
                elif model_option == 'Naive Bayes':
                    clf = MultinomialNB()

                clf.fit(X_train, y_train)
                st.session_state['clf'] = clf  # Store the model in session state
                st.session_state['tfidf'] = tfidf  # Store the TFIDF vectorizer in session state

                train_accuracy, train_f1, test_accuracy, test_f1, report = evaluate_model(clf, X_train, y_train, X_test,
                                                                                          y_test)

                st.subheader("Model Performance Metrics:")
                st.write(f"Train Accuracy: {train_accuracy}")
                st.write(f"Test Accuracy: {test_accuracy}")
                st.write(f"Train F1 Score: {train_f1}")
                st.write(f"Test F1 Score: {test_f1}")
                st.text("Classification Report:")
                st.text(report)

                # Save the model
                filename = f'finalized_{model_option}_model.sav'
                pickle.dump(clf, open(filename, 'wb'))
                st.success(f"{model_option} model trained and saved successfully!")

    # LIME Explanations Section
    st.header("LIME Explanations for Model Predictions")
    if uploaded_file is not None and 'clf' in st.session_state and st.checkbox("Generate LIME Explanation"):
        clf = st.session_state['clf']  # Retrieve the model from session state
        tfidf = st.session_state['tfidf']  # Retrieve the TFIDF vectorizer from session state
        c = make_pipeline(tfidf, clf)  # Create the pipeline

        idx = np.random.randint(0, len(df))
        headline = df.iloc[idx]["headline"]
        explainer = LimeTextExplainer(class_names=["Non Sarcastic", "Sarcastic"])
        exp = explainer.explain_instance(headline, c.predict_proba, num_features=10)
        st.write(f"Example {idx}")
        st.write("Headline:", headline)
        st.write("True Class:", ["Non Sarcastic", "Sarcastic"][df.iloc[idx]["is_sarcastic"]])

        fig = exp.as_pyplot_figure()  # Use the as_pyplot_figure method directly
        st.pyplot(fig)

    # Real-time Text Summarization
    st.header("Real-time Text Summarization")
    user_input_summarization = st.text_area("Enter text to summarize")
    if user_input_summarization:
        summarized_text = text_summarization(user_input_summarization)
        st.subheader("Summarized Text:")
        st.write(summarized_text)

    # Real-time Sarcasm Prediction
    st.header("Real-time Sarcasm Prediction")
    user_input_sarcasm = st.text_input("Enter a headline to predict sarcasm")
    if user_input_sarcasm:
        if 'clf' in st.session_state:
            prediction_result = predict_sarcasm(user_input_sarcasm, st.session_state['clf'], st.session_state['tfidf'])
            st.write(f"Prediction: **{prediction_result}**")


if __name__ == "__main__":
    main()
