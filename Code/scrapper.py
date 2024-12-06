import pandas as pd
import os
import bs4
from tqdm import tqdm
from urllib.request import urlopen
from urllib.error import URLError, HTTPError
from socket import gaierror
import gdown

# need to install lxml
def download_available_data(target_dir):
    current_directory = os.getcwd()
    if not os.path.exists(target_dir):
        os.mkdir(target_dir)
    os.chdir(target_dir)
    file_id_mapping = {
        "Sarcasm_Headlines_Dataset.json": "1NSpIABIfwZ3E1As6XVbNCX3do0D9MucI",
        "sarcastic_output.json": "19dS1iQ51oxRmiEkoArWUwW6BqXYDJGuo"
    }
    for key, value in file_id_mapping.items():
        if not os.path.exists(os.path.join(target_dir, key)):
            downloaded_file = gdown.download(id=value, quiet=False)
            if downloaded_file != key:
                print(f"{key} could not be downloaded")
    os.chdir(current_directory)


def extract_text_from_url(url):
    offset = 30
    try:
        source = urlopen(url).read()
        soup = bs4.BeautifulSoup(source, 'lxml')
    except (URLError, gaierror):
        try:
            new_url = url[offset:]
            source = urlopen(new_url).read()
            soup = bs4.BeautifulSoup(source, 'lxml')
        except (URLError, gaierror, ValueError, HTTPError):
            text = ""
            return text
    allowlist = ["p", "em", "i", "b"]
    blocklist = ["Sign Up", "HuffPost", "Huffington"]
    text_elements = list()
    for t in soup.find_all(text=True):
        if t.parent.name in allowlist:
            contains_blocked = False
            for block in blocklist:
                if t.find(block) != -1:
                    contains_blocked = True
            if not contains_blocked:
                text_elements.append(t)

    text = " ".join(text_elements)
    return text


code_directory = os.getcwd()
data_directory = os.path.join(os.path.split(code_directory)[0], 'Code')
download_available_data(data_directory)
df1 = pd.read_json(os.path.join(data_directory, r'Sarcasm_Headlines_Dataset.json'), lines=True)
df2 = pd.read_json(os.path.join(data_directory, r'Sarcasm_Headlines_Dataset_v2.json'), lines=True)
df2 = df2[['article_link', 'headline', 'is_sarcastic']]
final_data = pd.concat([df1, df2], ignore_index=True)
sarcastic_data = final_data.loc[final_data['is_sarcastic'] == 1]
sarcastic_data.reset_index(drop=True, inplace=True)
print(sarcastic_data.head())
print(sarcastic_data.columns)
print(sarcastic_data.shape)

sarcastic_data["body"] = [""] * len(sarcastic_data)
print("\nScraping TheOnion articles...")
with tqdm(total=len(sarcastic_data)) as progress_bar:
    for i, row in sarcastic_data.iterrows():
        body = extract_text_from_url(row[0])
        sarcastic_data.loc[i, "body"] = body
        progress_bar.update()

output_path = os.path.join(data_directory, "sarcastic_news_text.json")
sarcastic_data.to_json(output_path)