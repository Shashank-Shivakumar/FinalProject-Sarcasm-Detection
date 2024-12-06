import pandas as pd
import numpy as np
import string
import matplotlib.pyplot as plt
import seaborn as sns
from nltk import word_tokenize, pos_tag, ngrams
from nltk.corpus import stopwords
import re
from collections import Counter

from pandas.tests.tools.test_to_datetime import epochs
from textblob import TextBlob
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from gensim.models import Word2Vec
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
from wordcloud import WordCloud
import tensorflow as tf
import keras

from tensorflow.keras.preprocessing.text import Tokenizer
# from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Embedding, LSTM, Dense, SpatialDropout1D, Bidirectional, Dropout
from keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np

import torch
from transformers import BertTokenizer, BertForSequenceClassification, AdamW, get_linear_schedule_with_warmup
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import numpy as np
from sklearn.model_selection import train_test_split

from transformers import BertModel
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

from keras.layers import Conv1D, GlobalMaxPooling1D, Dropout
from torch.cuda.amp import GradScaler, autocast
from transformers import BertForSequenceClassification, AdamW
from transformers import get_linear_schedule_with_warmup

torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()
def clean_text(text):
    text = text.lower()  # Lowercasing
    text = re.sub(r'http\S+', '', text)  # Remove URLs
    text = re.sub(r'\d+', '', text)  # Remove digits
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    return text

def get_wordnet_pos(word):
    """Map POS tag to first character lemmatize() accepts"""
    tag = pos_tag([word])[0][1][0].upper()
    tag_dict = {"J": wordnet.ADJ, "N": wordnet.NOUN, "V": wordnet.VERB, "R": wordnet.ADV}
    return tag_dict.get(tag, wordnet.NOUN)

def lemmatize_text(text):
    return ' '.join([lemmatizer.lemmatize(w, get_wordnet_pos(w)) for w in nltk.word_tokenize(text)])
# Load the datasets
data_v1 = pd.read_json('Sarcasm_Headlines_Dataset.json', lines=True)
data_v2 = pd.read_json('Sarcasm_Headlines_Dataset_v2.json', lines=True)

# Display the head and description of both datasets
print('First Dataset Head:')
print(data_v1.head())
print('\nFirst Dataset Description:')
print(data_v1.describe(include='all'))

print('\nSecond Dataset Head:')
print(data_v2.head())
print('\nSecond Dataset Description:')
print(data_v2.describe(include='all'))

# Check the balance of sarcastic and non-sarcastic labels in each dataset
balance_v1 = data_v1['is_sarcastic'].value_counts(normalize=True)
balance_v2 = data_v2['is_sarcastic'].value_counts(normalize=True)
print('Balance of labels in first dataset (v1):')
print(balance_v1)
print('\nBalance of labels in second dataset (v2):')
print(balance_v2)

# Check for unique and common headlines
unique_headlines_v1 = data_v1['headline'].nunique()
unique_headlines_v2 = data_v2['headline'].nunique()
common_headlines = pd.Series(list(set(data_v1['headline']).intersection(set(data_v2['headline']))))
print('\nNumber of unique headlines in first dataset (v1):', unique_headlines_v1)
print('Number of unique headlines in second dataset (v2):', unique_headlines_v2)
print('\nNumber of common headlines between the datasets:', common_headlines.size)

# Data Cleaning and Preprocessing
data_v1.dropna(inplace=True)
data_v2.dropna(inplace=True)
common_headlines = set(data_v1['headline'])
data_v2_unique = data_v2[~data_v2['headline'].isin(common_headlines)]

# Merge the datasets
merged_data = pd.concat([data_v1, data_v2_unique]).reset_index(drop=True)

# Check the balance of the merged dataset
balance_merged = merged_data['is_sarcastic'].value_counts(normalize=True)
print('\nNumber of headlines after merging and removing duplicates:', merged_data.shape[0])
print('Balance of labels in the merged dataset:')
print(balance_merged)

# Save the merged dataset
merged_data.to_csv('cleaned_merged_sarcasm_dataset.csv', index=False)
print('\nMerged dataset saved as cleaned_merged_sarcasm_dataset.csv')
# Apply the cleaning and lemmatization functions
merged_data['cleaned_headline'] = merged_data['headline'].apply(clean_text)
merged_data['lemmatized_headline'] = merged_data['cleaned_headline'].apply(lemmatize_text)

# Feature Engineering
def word_count(text):
    return len(text.split())

def char_count(text):
    return len(text)

def punctuation_count(text):
    return len([char for char in text if char in string.punctuation])

merged_data['word_count'] = merged_data['headline'].apply(word_count)
merged_data['char_count'] = merged_data['headline'].apply(char_count)
merged_data['punctuation_count'] = merged_data['headline'].apply(punctuation_count)

# Text Length Analysis
merged_data['headline_length'] = merged_data['headline'].apply(len)
sns.boxplot(x='is_sarcastic', y='headline_length', data=merged_data)
plt.title('Distribution of Headline Lengths by Sarcasm Label')
plt.savefig("headline_lengths.png")
print("Plot saved as headline_lengths.png")
plt.show()

# Exclamation and Question Marks Analysis
merged_data['exclamation_mark'] = merged_data['headline'].apply(lambda x: x.count('!'))
merged_data['question_mark'] = merged_data['headline'].apply(lambda x: x.count('?'))

sns.countplot(x='exclamation_mark', hue='is_sarcastic', data=merged_data)
plt.title('Distribution of Exclamation Marks in Headlines')
plt.savefig("exclamation_marks.png")
print("Plot saved as exclamation_marks.png")
plt.show()

sns.countplot(x='question_mark', hue='is_sarcastic', data=merged_data)
plt.title('Distribution of Question Marks in Headlines')
plt.savefig("question_marks.png")
print("Plot saved as question_marks.png")
plt.show()

# Correcting Top Unigrams in Headlines
def plot_top_unigrams(text, title, n=20):
    # Tokenize the text to words
    tokens = [word for headline in text for word in word_tokenize(headline) if word.isalpha()]
    # Calculate frequency distribution
    fdist = nltk.FreqDist(tokens)
    # Prepare for plotting
    words_df = pd.DataFrame({'word': list(fdist.keys()), 'count': list(fdist.values())})
    d = words_df.nlargest(columns="count", n=n)
    # Plot
    plt.figure(figsize=(20,5))
    ax = sns.barplot(data=d, x="word", y="count")
    ax.set(ylabel = 'Count')
    plt.title(title)
    plt.savefig(f"{title}.png")
    print(f"Plot saved as {title}.png")
    plt.show()

# Now use the function to plot the top words for sarcastic and non-sarcastic headlines
plot_top_unigrams(merged_data[merged_data['is_sarcastic'] == 1]['headline'], 'Top Unigrams in Sarcastic Headlines')
plot_top_unigrams(merged_data[merged_data['is_sarcastic'] == 0]['headline'], 'Top Unigrams in Non-Sarcastic Headlines')

# Define function to calculate sentiment polarity
def calculate_sentiment(text):
    return TextBlob(text).sentiment.polarity

# Apply the sentiment analysis
merged_data['sentiment_polarity'] = merged_data['headline'].apply(calculate_sentiment)

# Visualize the sentiment polarity distribution
sns.boxplot(x='is_sarcastic', y='sentiment_polarity', data=merged_data)
plt.title('Sentiment Polarity Distribution by Sarcasm Label')
plt.savefig("sentiment_polarity.png")
print("Plot saved as sentiment_polarity.png")
plt.show()

# Define function to calculate multiple punctuation count
def count_multiple_punctuation(text):
    return len(re.findall(r'[\?!.]{2,}', text))

# Apply the function to count multiple punctuations
merged_data['multiple_punct_count'] = merged_data['headline'].apply(count_multiple_punctuation)

# Visualize the distribution of multiple consecutive punctuations
sns.countplot(x='multiple_punct_count', hue='is_sarcastic', data=merged_data)
plt.title('Distribution of Multiple Consecutive Punctuation Marks in Headlines')
plt.savefig("multiple_consecutive_punctuation_marks.png")
print("Plot saved as multiple_consecutive_punctuation_marks.png")
plt.show()

# Define function to flag large numbers
def contains_large_number(text):
    return int(bool(re.search(r'\b\d{5,}\b', text)))

# Apply the function to flag large numbers
merged_data['large_number_flag'] = merged_data['headline'].apply(contains_large_number)

# Visualize the presence of large numbers in headlines
sns.countplot(x='large_number_flag', hue='is_sarcastic', data=merged_data)
plt.title('Presence of Large Numbers in Headlines')
plt.savefig("large_numbers.png")
print("Plot saved as large_numbers.png")
plt.show()

# Define function to calculate all caps words count
def count_all_caps(text):
    return sum(word.isupper() for word in text.split() if len(word) > 1)

# Apply the function to count all caps words
merged_data['all_caps_count'] = merged_data['headline'].apply(count_all_caps)

# Visualize the use of all caps words
sns.boxplot(x='is_sarcastic', y='all_caps_count', data=merged_data)
plt.title('All Caps Word Count Distribution by Sarcasm Label')
plt.savefig("all_caps_words.png")
print("Plot saved as all_caps_words.png")
plt.show()

# Define function to calculate punctuation count
def calculate_punctuation_count(text):
    return len(re.findall(r'[^\w\s]', text))

# Apply the function to calculate punctuation count
merged_data['punctuation_count'] = merged_data['headline'].apply(calculate_punctuation_count)

# Visualize the punctuation count distribution
sns.boxplot(x='is_sarcastic', y='punctuation_count', data=merged_data)
plt.title('Punctuation Count Distribution by Sarcasm Label')
plt.savefig("punctuation_count.png")
print("Plot saved as punctuation_count.png")
plt.show()

# Tokenize and remove stop words from each headline
merged_data['filtered_tokens'] = merged_data['headline'].apply(
    lambda x: [word.lower() for word in word_tokenize(x) if word.isalpha() and word.lower() not in stop_words]
)

# Top Words Analysis
def plot_top_words(filtered_tokens_series, title):
    all_words = [word for tokens in filtered_tokens_series for word in tokens]
    word_freq = Counter(all_words)
    top_words_df = pd.DataFrame(word_freq.most_common(20), columns=['word', 'count'])
    plt.figure(figsize=(12, 8))
    sns.barplot(x='count', y='word', data=top_words_df)
    plt.title(title)
    plt.savefig(f"{title}.png")
    print(f"Plot saved as {title}.png")
    plt.show()

# Plot top words for sarcastic and non-sarcastic headlines
plot_top_words(merged_data[merged_data['is_sarcastic'] == 1]['filtered_tokens'], 'Top Words in Sarcastic Headlines')
plot_top_words(merged_data[merged_data['is_sarcastic'] == 0]['filtered_tokens'], 'Top Words in Non-Sarcastic Headlines')

# N-grams Analysis
def plot_top_ngrams(filtered_tokens_series, title, n=2):
    all_ngrams = ngrams([word for tokens in filtered_tokens_series for word in tokens], n)
    ngram_freq = Counter(all_ngrams)
    top_ngrams_df = pd.DataFrame(ngram_freq.most_common(20), columns=['n_gram', 'count'])
    top_ngrams_df['n_gram'] = top_ngrams_df['n_gram'].apply(lambda x: ' '.join(x))
    plt.figure(figsize=(12, 8))
    sns.barplot(x='count', y='n_gram', data=top_ngrams_df)
    plt.title(title)
    plt.savefig(f"{title}.png")
    print(f"Plot saved as {title}.png")
    plt.show()

# Plot top n-grams for sarcastic and non-sarcastic headlines
plot_top_ngrams(merged_data[merged_data['is_sarcastic'] == 1]['filtered_tokens'], 'Top Bigrams in Sarcastic Headlines', n=2)
plot_top_ngrams(merged_data[merged_data['is_sarcastic'] == 0]['filtered_tokens'], 'Top Bigrams in Non-Sarcastic Headlines', n=2)

# Combine all headlines for the word cloud
all_headlines = ' '.join(merged_data['lemmatized_headline'])

# Generate a word cloud
wordcloud = WordCloud(width=800, height=400, background_color='white').generate(all_headlines)

# Display the word cloud using matplotlib
plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title('Word Cloud of Headlines')
plt.savefig("word_cloud.png")
print("Plot saved as word_cloud.png")
plt.show()

# Save the enhanced dataset with new features
merged_data.to_csv('enhanced_sarcasm_dataset.csv', index=False)
print('\nEnhanced dataset saved as enhanced_sarcasm_dataset.csv')

# ##Before adding Preprocessing steps
# # TF-IDF Vectorization
# tfidf_vectorizer = TfidfVectorizer(max_features=5000)
# X_tfidf = tfidf_vectorizer.fit_transform(merged_data['headline']).toarray()
# # Word2Vec Model Training
# word2vec_model = Word2Vec(merged_data['filtered_tokens'], vector_size=100, window=5, min_count=2, workers=4)


##After adding preprocessing steps:
# TF-IDF Vectorization using lemmatized text
tfidf_vectorizer = TfidfVectorizer(max_features=5000)
X_tfidf = tfidf_vectorizer.fit_transform(merged_data['lemmatized_headline']).toarray()
# Word2Vec Model Training using tokenized lemmatized text
word2vec_model = Word2Vec(merged_data['lemmatized_headline'].apply(nltk.word_tokenize), vector_size=100, window=5, min_count=2, workers=4)

# Split the data for the machine learning model
X_train, X_test, y_train, y_test = train_test_split(
    X_tfidf,
    merged_data['is_sarcastic'],
    test_size=0.2,
    random_state=42
)

##Baseline Models
#1LogisticRegression
from sklearn.linear_model import LogisticRegression
# Initialize the model
lr_model = LogisticRegression(max_iter=1000)
# Train the model
lr_model.fit(X_train, y_train)
# Predict on the test set
y_pred_lr = lr_model.predict(X_test)
# Calculate accuracy
accuracy_lr = accuracy_score(y_test, y_pred_lr)
print(f"Logistic Regression Accuracy: {accuracy_lr}")

#2NaiveBayes
from sklearn.naive_bayes import MultinomialNB
# Initialize the model
nb_model = MultinomialNB()
# Train the model
nb_model.fit(X_train, y_train)
# Predict on the test set
y_pred_nb = nb_model.predict(X_test)
# Calculate accuracy
accuracy_nb = accuracy_score(y_test, y_pred_nb)
print(f"Naive Bayes Accuracy: {accuracy_nb}")

#SVM
# from sklearn.svm import SVC
# # Initialize the model
# svm_model = SVC()
# # Train the model
# svm_model.fit(X_train, y_train)
# # Predict on the test set
# y_pred_svm = svm_model.predict(X_test)
# # Calculate accuracy
# accuracy_svm = accuracy_score(y_test, y_pred_svm)
# print(f"SVM Accuracy: {accuracy_svm}")

# ##Ensemble Models
# #1RandomForest
# from sklearn.ensemble import RandomForestClassifier
# # Initialize the model
# rf_model = RandomForestClassifier()
# # Train the model
# rf_model.fit(X_train, y_train)
# # Predict on the test set
# y_pred_rf = rf_model.predict(X_test)
# # Calculate accuracy
# accuracy_rf = accuracy_score(y_test, y_pred_rf)
# print(f"Random Forest Accuracy: {accuracy_rf}")

# #2GradientBoosting
# from sklearn.ensemble import GradientBoostingClassifier
# # Initialize the model
# gb_model = GradientBoostingClassifier()
# # Train the model
# gb_model.fit(X_train, y_train)
# # Predict on the test set
# y_pred_gb = gb_model.predict(X_test)
# # Calculate accuracy
# accuracy_gb = accuracy_score(y_test, y_pred_gb)
# print(f"Gradient Boosting Accuracy: {accuracy_gb}")

##Deep Learning Approaches
# #1LSTM (Long Short-Term Memory networks):

# Hyperparameters
vocab_size = 10000
max_length = 120
embedding_dim = 64
oov_tok = '<OOV>'

# Tokenizing for Keras models
tokenizer_keras = Tokenizer(num_words=vocab_size, oov_token=oov_tok)
tokenizer_keras.fit_on_texts(merged_data['lemmatized_headline'])
sequences = tokenizer_keras.texts_to_sequences(merged_data['lemmatized_headline'])
padded = pad_sequences(sequences, maxlen=max_length, padding='post', truncating='post')

# Splitting the data
X_train, X_test, y_train, y_test = train_test_split(padded, merged_data['is_sarcastic'], test_size=0.2, random_state=42)

# LSTM Model with Bidirectional layer and Spatial Dropout
model_lstm = Sequential([
    Embedding(vocab_size, embedding_dim, input_length=max_length),
    SpatialDropout1D(0.3),
    Bidirectional(LSTM(64, dropout=0.2, recurrent_dropout=0.2)),
    Dense(64, activation='relu'),
    Dense(1, activation='sigmoid')
])

# Compile the model
model_lstm.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Model Summary
model_lstm.summary()

# Early Stopping to prevent overfitting
early_stopping = EarlyStopping(monitor='val_loss', patience=3, verbose=1)

# Train the model
history = model_lstm.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test), callbacks=[early_stopping])

# Predictions
y_pred = model_lstm.predict(X_test)
y_pred_classes = np.where(y_pred > 0.5, 1, 0)

# Evaluation Metrics
print("Classification Report:")
print(classification_report(y_test, y_pred_classes))

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_classes))

# Plotting accuracy and loss graphs

def plot_graphs(history, metric, filename):
    plt.plot(history.history[metric])
    plt.plot(history.history['val_' + metric])
    plt.xlabel("Epochs")
    plt.ylabel(metric)
    plt.legend([metric, 'val_' + metric])
    plt.title(f"{metric.capitalize()} Over Epochs")
    plt.savefig(f"{filename}.png")
    print(f"Plot saved as {filename}.png")
    plt.show()

plot_graphs(history, 'accuracy', 'accuracy_plot')
plot_graphs(history, 'loss', 'loss_plot')


# CNN Model
model_cnn = Sequential([
    Embedding(vocab_size, 32, input_length=max_length),
    Conv1D(64, 5, activation='relu'),
    GlobalMaxPooling1D(),
    Dense(64, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])

# Compile the model
model_cnn.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Model summary
model_cnn.summary()

# Callback for early stopping
early_stopping = EarlyStopping(monitor='val_loss', patience=2, restore_best_weights=True)

# Train the model
history_cnn = model_cnn.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test), callbacks=[early_stopping])

# Evaluate the model
eval_result = model_cnn.evaluate(X_test, y_test)
print(f'\nTest Loss: {eval_result[0]} / Test Accuracy: {eval_result[1]}')

# Generate classification report and confusion matrix
y_pred_cnn = model_cnn.predict(X_test).round()
print(classification_report(y_test, y_pred_cnn))
print(confusion_matrix(y_test, y_pred_cnn))



# Tranformer
# torch.cuda.empty_cache()


# Initialize BertTokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Split data into training and validation sets
train_texts, val_texts, train_labels, val_labels = train_test_split(
    merged_data['lemmatized_headline'].tolist(),
    merged_data['is_sarcastic'].tolist(),
    test_size=0.1,  # You can adjust the test size
    random_state=42
)



#Creating a Custom Dataset Class:
class SarcasmDataset(Dataset):
    def __init__(self, headlines, labels, tokenizer, max_token_len=16):
        self.headlines = headlines
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_token_len = max_token_len


    def __len__(self):
        return len(self.headlines)

    def __getitem__(self, index):
        headline = self.headlines[index]
        label = self.labels[index]

        encoding = self.tokenizer.encode_plus(
            headline,
            add_special_tokens=True,
            max_length=self.max_token_len,
            return_token_type_ids=False,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(self.labels[index], dtype=torch.long)  # Add this line
        }

# torch.cuda.empty_cache()

# Create instances of SarcasmDataset for training and validation
train_dataset = SarcasmDataset(train_texts, train_labels, tokenizer, max_token_len=16)
val_dataset = SarcasmDataset(val_texts, val_labels, tokenizer, max_token_len=16)

# Creating DataLoaders
train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=0, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=0, pin_memory=True)

#BERT model architecture for classification.
from transformers import BertForSequenceClassification, AdamW, BertConfig

# Load BertForSequenceClassification
model = BertForSequenceClassification.from_pretrained(
    "bert-base-uncased",
    num_labels=2,  # Binary classification
    output_attentions=False,
    output_hidden_states=False,
)

model.gradient_checkpointing_enable()

for param in model.bert.embeddings.parameters():
    param.requires_grad = False

# Freeze all but the last 4 layers of the encoder
for layer in model.bert.encoder.layer[:-4]:  # Change 4 to a smaller number if needed
    for param in layer.parameters():
        param.requires_grad = False

# torch.cuda.empty_cache()

scaler = GradScaler()

# Set the device and move the model to it
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# Optimizer and Scheduler
optimizer = AdamW(model.parameters(), lr=2e-5, eps=1e-8)

# Number of training epochs
epochs = 5  # Adjust the number of epochs based on your requirements

# Calculate the total number of training steps
total_steps = len(train_loader) * epochs

# Create the learning rate scheduler
scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=0,
    num_training_steps=total_steps
)

# Training loop
accumulation_steps = 4  # Adjust based on memory constraints and desired batch size
optimizer.zero_grad()

for epoch in range(epochs):  # Loop over epochs
    torch.cuda.empty_cache()  # Clear GPU cache
    total_train_loss = 0
    model.train()  # Set model to training mode

    # print(f"Before epoch {epoch + 1} starts:")
    # print(f"Memory allocated: {torch.cuda.memory_allocated() / 1024 ** 2:.2f} MB")
    # print(f"Max memory allocated: {torch.cuda.max_memory_allocated() / 1024 ** 2:.2f} MB")

    for step, batch in enumerate(train_loader):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        # Forward pass with mixed precision
        with autocast():
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss

        # print(f"After forward pass:")
        # print(f"Memory allocated: {torch.cuda.memory_allocated() / 1024 ** 2:.2f} MB")
        # print(f"Max memory allocated: {torch.cuda.max_memory_allocated() / 1024 ** 2:.2f} MB")

        # Backward pass
        loss = loss / accumulation_steps
        total_train_loss += loss.item()

        scaler.scale(loss).backward()

        # print(f"After backward pass of step {step + 1}:")
        # print(f"Memory allocated: {torch.cuda.memory_allocated() / 1024 ** 2:.2f} MB")
        # print(f"Max memory allocated: {torch.cuda.max_memory_allocated() / 1024 ** 2:.2f} MB")

        # Gradient accumulation
        if (step + 1) % accumulation_steps == 0:
            scaler.step(optimizer)  # Update model parameters
            scaler.update()  # Update the scale for next iteration
            optimizer.zero_grad()  # Reset gradients after updating
            torch.cuda.empty_cache()  # Clear cache after each gradient update

            # print(f"After optimizer step of step {step + 1}:")
            # print(f"Memory allocated: {torch.cuda.memory_allocated() / 1024 ** 2:.2f} MB")
            # print(f"Max memory allocated: {torch.cuda.max_memory_allocated() / 1024 ** 2:.2f} MB")

    avg_train_loss = total_train_loss / len(train_loader)
    print(f"Epoch {epoch + 1} - Average training loss: {avg_train_loss:.4f}")

# Assuming 'val_loader' is your DataLoader for the validation set
model.eval()  # Set the model to evaluation mode

# Initialize lists to store predictions and true labels
all_predictions = []
all_true_labels = []

# Evaluation loop
with torch.no_grad():
    for batch in val_loader:
        torch.cuda.empty_cache()
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        # print(f"During validation batch:")
        # print(f"Memory allocated: {torch.cuda.memory_allocated() / 1024 ** 2:.2f} MB")
        # print(f"Max memory allocated: {torch.cuda.max_memory_allocated() / 1024 ** 2:.2f} MB")

        outputs = model(input_ids, token_type_ids=None, attention_mask=attention_mask)
        logits = outputs.logits

        # Move logits and labels to CPU
        logits = logits.detach().cpu().numpy()
        label_ids = labels.to('cpu').numpy()

        # Store predictions and true labels
        all_predictions.extend(np.argmax(logits, axis=1).flatten())
        all_true_labels.extend(label_ids.flatten())

# Calculate evaluation metrics
accuracy = accuracy_score(all_true_labels, all_predictions)
precision, recall, f1, _ = precision_recall_fscore_support(all_true_labels, all_predictions, average='binary')

print(f"Accuracy: {accuracy}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1 Score: {f1}")

# torch.cuda.empty_cache()

# Assuming your dataloaders (train_loader and val_loader) are already defined and loaded with the dataset

# BERT + LSTM Model

from torch.cuda.amp import GradScaler, autocast

# Assuming BertLSTM and BertCNN are already defined
class BertLSTM(nn.Module):
    def __init__(self):
        super(BertLSTM, self).__init__()  # Ensure parent class is initialized first
        self.bert = BertModel.from_pretrained("bert-base-uncased")  # Initialize BERT

        # Freeze embeddings and earlier encoder layers
        for param in self.bert.embeddings.parameters():
            param.requires_grad = False
        for layer in self.bert.encoder.layer[:-2]:  # Freeze all but last 4 layers
            for param in layer.parameters():
                param.requires_grad = False

        # Add LSTM and classifier
        self.lstm = nn.LSTM(input_size=768, hidden_size=128, num_layers=1, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(128 * 2, 2)  # Bidirectional LSTM doubles the hidden size

    def forward(self, input_ids, attention_mask):
        with torch.no_grad():  # Disable gradient computation for BERT layers
            outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = outputs.last_hidden_state  # Extract token-level features

        lstm_out, _ = self.lstm(sequence_output)  # Pass through LSTM
        logits = self.fc(lstm_out[:, -1, :])  # Use the output of the last LSTM step
        return logits


class BertCNN(nn.Module):
    def __init__(self):
        super(BertCNN, self).__init__()  # Ensure parent class is initialized first
        self.bert = BertModel.from_pretrained("bert-base-uncased")  # Initialize BERT

        # Freeze embeddings and earlier encoder layers
        for param in self.bert.embeddings.parameters():
            param.requires_grad = False
        for layer in self.bert.encoder.layer[:-4]:  # Freeze all but last 4 layers
            for param in layer.parameters():
                param.requires_grad = False

        # Add convolutional layer and classifier
        self.conv = nn.Conv1d(in_channels=768, out_channels=128, kernel_size=2)
        self.relu = nn.ReLU()
        self.fc = nn.Linear(128, 2)  # Classifier layer

    def forward(self, input_ids, attention_mask):
        with torch.no_grad():  # Disable gradient computation for BERT layers
            outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = outputs.last_hidden_state  # Extract token-level features

        # Reshape for Conv1D: [batch_size, seq_len, hidden_size] -> [batch_size, hidden_size, seq_len]
        conv_input = sequence_output.permute(0, 2, 1)
        conv_out = self.conv(conv_input)  # Apply convolution
        conv_out = self.relu(conv_out)  # Apply ReLU activation

        # Max pooling
        pooled = torch.max(conv_out, 2).values  # Max pooling along sequence length
        logits = self.fc(pooled)  # Classifier layer
        return logits


# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Initialize models and move to device
bert_lstm_model = BertLSTM().to(device)
bert_cnn_model = BertCNN().to(device)

# Criterion and Optimizer
criterion = nn.CrossEntropyLoss()
optimizer_lstm = optim.Adam(bert_lstm_model.parameters(), lr=2e-5)
optimizer_cnn = optim.Adam(bert_cnn_model.parameters(), lr=2e-5)

# Mixed Precision Setup
scaler = GradScaler()

def train_model(model, train_loader, optimizer, criterion, epochs, device):
    for epoch in range(epochs):
        torch.cuda.empty_cache()

        model.train()
        # print(f"Before epoch {epoch + 1} starts TRAIN MODEL:")
        # print(f"Memory allocated: {torch.cuda.memory_allocated() / 1024 ** 2:.2f} MB")
        # print(f"Max memory allocated: {torch.cuda.max_memory_allocated() / 1024 ** 2:.2f} MB")

        total_loss = 0
        for batch in train_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            optimizer.zero_grad()
            torch.cuda.empty_cache()  # Clear cache after each gradient update

            # Forward pass with autocast
            with autocast():
                outputs = model(input_ids, attention_mask)
                loss = criterion(outputs, labels)

            # print(f"After forward pass of step {step + 1} Tranining Model:")
            # print(f"Memory allocated: {torch.cuda.memory_allocated() / 1024 ** 2:.2f} MB")
            # print(f"Max memory allocated: {torch.cuda.max_memory_allocated() / 1024 ** 2:.2f} MB")

            # Scales loss. Calls backward() on scaled loss to create scaled gradients.
            scaler.scale(loss).backward()

            # print(f"After backward pass of step {step + 1} Training Model:")
            # print(f"Memory allocated: {torch.cuda.memory_allocated() / 1024 ** 2:.2f} MB")
            # print(f"Max memory allocated: {torch.cuda.max_memory_allocated() / 1024 ** 2:.2f} MB")

            # scaler.step() first unscales the gradients of the optimizer's assigned params.
            # If these gradients do not contain infs or NaNs, optimizer.step() is then called,
            # otherwise, optimizer.step() is skipped.
            scaler.step(optimizer)

            # Updates the scale for next iteration.
            scaler.update()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        print(f'Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}')

# torch.cuda.empty_cache()
# Evaluation function
def evaluate_model(model, val_loader, device):
    model.eval()
    all_predictions = []
    all_true_labels = []

    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            outputs = model(input_ids, attention_mask)
            _, predicted = torch.max(outputs, 1)

            all_predictions.extend(predicted.tolist())
            all_true_labels.extend(labels.tolist())

    accuracy = accuracy_score(all_true_labels, all_predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(all_true_labels, all_predictions, average='binary')
    return accuracy, precision, recall, f1


# Training BERT + LSTM
train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=0, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=0, pin_memory=True)


# Training BERT + LSTM
print("Training BERT + LSTM...")
train_model(bert_lstm_model, train_loader, optimizer_lstm, criterion, epochs=5, device=device)

# Training BERT + CNN
print("Training BERT + CNN...")
train_model(bert_cnn_model, train_loader, optimizer_cnn, criterion, epochs=5, device=device)

# Evaluating BERT + LSTM
accuracy_lstm, precision_lstm, recall_lstm, f1_lstm = evaluate_model(bert_lstm_model, val_loader, device)
print(
    f"BERT + LSTM - Accuracy: {accuracy_lstm}, Precision: {precision_lstm}, Recall: {recall_lstm}, F1 Score: {f1_lstm}")

# Evaluating BERT + CNN
accuracy_cnn, precision_cnn, recall_cnn, f1_cnn = evaluate_model(bert_cnn_model, val_loader, device)
print(f"BERT + CNN - Accuracy: {accuracy_cnn}, Precision: {precision_cnn}, Recall: {recall_cnn}, F1 Score: {f1_cnn}")