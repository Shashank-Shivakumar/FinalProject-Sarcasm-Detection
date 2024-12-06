# This code implements a headline generation model using the T5 transformer architecture and
# evaluates its performance on a sarcastic news dataset. It includes data processing functions,
# tokenization, optional model training, and the generation of predictions for test data.
# The results, including ROUGE scores and random predictions, are visualized and saved to a CSV file.

import os
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from nltk import sent_tokenize
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from datasets import DatasetDict, Dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq, get_scheduler, pipeline
from accelerate import Accelerator
import evaluate
from tqdm import tqdm
import nltk
nltk.download('punkt_tab')

# Constants
data_splits = {'train': 0.8, 'val': 0.1, 'test': 0.1}
input_var = 'body'
target_var = 'headline'
word_limit = 30
model_checkpoint = "JulesBelveze/t5-small-headline-generator"
TRAIN_MODEL = True
random_seed = 42
batch_size = 16
num_train_epochs = 30

# Directories
code_dir = os.getcwd()
data_dir = code_dir#os.path.join(os.path.split(code_dir)[0], 'Code')
model_dir = os.path.join(os.path.split(code_dir)[0], 'Code')

# Function to filter training indices
def filter_indices(data):
    return np.array([len(x) > word_limit for x in data.body.str.split()])

# Function to display samples
def show_samples(data, num_samples=3, seed=random_seed):
    sample_data = data["train"].shuffle(seed=seed).select(range(num_samples))
    for example in sample_data:
        print(f"\n'>> Headline: {example[target_var]}'")
        print(f"'>> Body: {example[input_var]}'")

# Function for one-sentence summary
def generate_summary(text):
    return "\n".join(sent_tokenize(text)[:1])

# Function to evaluate baseline
def evaluate_baseline(data, metric):
    summaries = [generate_summary(text) for text in data[input_var]]
    return metric.compute(predictions=summaries, references=data[target_var])

# Function for post-processing text
def postprocess(predictions, labels):
    predictions = [pred.strip() for pred in predictions]
    labels = [label.strip() for label in labels]

    predictions = ["\n".join(sent_tokenize(pred)) for pred in predictions]
    labels = ["\n".join(sent_tokenize(label)) for label in labels]
    return predictions, labels

# Function to preprocess data
def preprocess(examples):
    model_inputs = tokenizer(
        examples[input_var],
        max_length=max_input_length,
        truncation=True
    )
    label = tokenizer(
        examples[target_var], max_length=max_target_length, truncation=True
    )
    model_inputs["labels"] = label["input_ids"]
    return model_inputs

# Function to get a summary
def get_summary(index):
    reviews = dataset['test'][index]["body"]
    titles = dataset["test"][index]["headline"]
    summaries = summarizer(dataset["test"][index]["body"], min_length=10, max_length=max_target_length)[0]["summary_text"]
    return reviews, titles, summaries

# Main code starts here

# Loading data
df = pd.read_json(os.path.join(data_dir, 'sarcastic_news_text.json'))
df = df[[input_var, target_var]]
train_val_df = df[filter_indices(df)]
test_df = df[~filter_indices(df)]
max_input_length = int(np.percentile([len(x) for x in train_val_df[input_var].str.split()], 99.5))
max_target_length = int(np.percentile([len(x) for x in train_val_df[target_var].str.split()], 99.5))

# Dataset creation
dataset = DatasetDict()
train_val_df = train_val_df.sample(frac=1, random_state=random_seed).reset_index()
total_percentage = 0
for split, percentage in data_splits.items():
    index_range = (int(total_percentage * len(train_val_df)), int((total_percentage + percentage) * len(train_val_df)))
    total_percentage = total_percentage + percentage
    dataset[split] = Dataset.from_pandas(train_val_df.iloc[index_range[0]:index_range[1]])

print("\nSample training data")
show_samples(dataset)

# Tokenizer initialization
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

print("\nTokenizing datasets")
tokenized_datasets = dataset.map(preprocess, batched=True)

# Lead-1 baseline
rouge_score = evaluate.load("rouge")
baseline_score = evaluate_baseline(dataset["val"], rouge_score)
print(f"\nBaseline score on validation dataset is: ")
for score, value in baseline_score.items():
    print(score, ": ", value)


# Lists to store Rouge scores for plotting
rouge_keys = ['rouge1', 'rouge2', 'rougeL', 'rougeLsum']
rouge_scores = {key: [] for key in rouge_keys}

# Training the summarizer
if TRAIN_MODEL:
    print('Training')
    lr = 2e-5

    # Fine-tuning
    tokenized_datasets.set_format("torch")
    model = AutoModelForSeq2SeqLM.from_pretrained(model_checkpoint)
    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)
    tokenized_datasets = tokenized_datasets.remove_columns(dataset["train"].column_names)
    features = [tokenized_datasets["train"][i] for i in range(2)]
    data_collator(features)

    # DataLoader definition
    train_dataloader = DataLoader(
        tokenized_datasets["train"],
        shuffle=True,
        collate_fn=data_collator,
        batch_size=batch_size,
    )
    eval_dataloader = DataLoader(
        tokenized_datasets["val"], collate_fn=data_collator, batch_size=batch_size
    )

    # Optimizer initialization
    optimizer = AdamW(model.parameters(), lr=lr)

    # Accelerator preparation
    accelerator = Accelerator()
    model, optimizer, train_dataloader, eval_dataloader = accelerator.prepare(
        model, optimizer, train_dataloader, eval_dataloader
    )

    num_update_steps_per_epoch = len(train_dataloader)
    num_training_steps = num_train_epochs * num_update_steps_per_epoch

    # Learning rate schedule definition
    lr_scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=num_training_steps,
    )

    # Model and optimizer saving directory
    output_directory = os.path.join(model_dir, "t5-sarcastic-headline-generator")

    progress_bar = tqdm(range(num_training_steps))

    print("\nFine-tuning pretrained model")
    for epoch in range(num_train_epochs):
        model.train()
        progress_bar.set_description(f"Epoch: [{epoch + 1}/{num_train_epochs}]")
        for step, batch in enumerate(train_dataloader):
            outputs = model(**batch)
            loss = outputs.loss
            accelerator.backward(loss)

            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            progress_bar.update(1)

        model.eval()
        for step, batch in enumerate(eval_dataloader):
            with torch.no_grad():
                generated_tokens = accelerator.unwrap_model(model).generate(
                    batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                )

                generated_tokens = accelerator.pad_across_processes(
                    generated_tokens, dim=1, pad_index=tokenizer.pad_token_id
                )

                labels = accelerator.pad_across_processes(
                    batch["labels"], dim=1, pad_index=tokenizer.pad_token_id
                )

                generated_tokens = accelerator.gather(generated_tokens).cpu().numpy()
                labels = accelerator.gather(labels).cpu().numpy()

                labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
                if isinstance(generated_tokens, tuple):
                    generated_tokens = generated_tokens[0]
                decoded_preds = tokenizer.batch_decode(
                    generated_tokens, skip_special_tokens=True
                )

                decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
                decoded_preds, decoded_labels = postprocess(
                    decoded_preds, decoded_labels
                )

                rouge_score.add_batch(predictions=decoded_preds, references=decoded_labels)

        result = rouge_score.compute()
        result = {k: round(v, 4) for k, v in result.items()}

        # Append Rouge scores to the lists
        for key in rouge_keys:
            rouge_scores[key].append(result[key])

        print(f"Epoch {epoch + 1}:", end=" ")
        for key, value in result.items():
            print(key, ' : ', value, end=" ")
        print(" ")

        accelerator.wait_for_everyone()
        unwrapped_model = accelerator.unwrap_model(model)
        unwrapped_model.save_pretrained(output_directory, save_function=accelerator.save)
        if accelerator.is_main_process:
            tokenizer.save_pretrained(output_directory)


# Predictions on test data after loading model
saved_model = os.path.join(model_dir, "t5-sarcastic-headline-generator")
summarizer = pipeline("summarization", model=saved_model)

# Plotting the Rouge scores
plt.figure(figsize=(10, 6))
for key in rouge_keys:
    plt.plot(range(1, num_train_epochs + 1), rouge_scores[key], label=key)

plt.xlabel('Epoch')
plt.ylabel('Rouge Score')
plt.title('Epoch vs Rouge Score')
plt.legend()
plt.savefig("epoch_rouge.png")
print(f"Plot saved as epoch_rouge.png")
plt.show()

# Print 3 random predictions
prediction_index = []
counter = 0
while counter < 3:
    flag = False
    index = random.randint(0, len(dataset['test']))
    if index not in prediction_index:
        flag = True
        counter = counter + 1
        prediction_index.append(index)
    if flag:
        review, title, summary = get_summary(index)
        print(f'\n-- Prediction {counter} --')
        print(f"'>>> Article: {review}'")
        print(f"\n'>>> Headline: {title}'")
        print(f"\n'>>> Summary: {summary}'")

print("\n Building Predictions.csv for all test data")
main_list = []
for i in tqdm(range(len(dataset['test']))):
    review, title, summary = get_summary(i)
    temp = [review, title, summary]
    main_list.append(temp)

# Save predictions into csv file
predictions_df = pd.DataFrame(main_list, columns=['Body', 'Headline', 'Summary'])
predictions_df.to_csv(os.path.join(data_dir, 'Summary_Predictions.csv'), index=False)
