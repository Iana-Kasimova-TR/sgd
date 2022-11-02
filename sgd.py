from encodings import utf_8
import re
import spacy
import nltk
import sys
import string
from nltk import word_tokenize
import csv
import pandas as pd
import numpy as np
import random as rnd
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.stem.wordnet import WordNetLemmatizer
from collections import Counter, defaultdict
from scipy.sparse import csr_matrix
#stop_words = stopwords.words('english')
stop_words = stopwords.words('russian')
#nlp = spacy.load("en_core_web_sm")
nlp = spacy.load("ru_core_news_sm")
csv.field_size_limit(sys.maxsize)

def read_file(path_to_file):
    number_title = 3
    number_text = 4
    with open(path_to_file, newline="", encoding="utf_8") as csvfile:
        for record in csv.reader(csvfile, delimiter=","):
            yield record[number_title], record[number_text]

def clean_text_version_2(text):
    final_str = ""
    text = text.lower()
    text = re.sub(r'\n', '', text)

    #remove punctuation
    translator = str.maketrans('', '', string.punctuation)
    text = text.translate(translator)

    #Remove stop words
    text = text.split()
    useless_words = nltk.corpus.stopwords.words("english")
    useless_words += ['hi', 'im']

    text_filtered = [word for word in text if word not in useless_words]
    #Remove numbers
    text_filtered = [re.sub(r'\w*\d\w*', '', w) for w in text_filtered]

    text_filtered = nlp(" ".join(text_filtered))
    text_stemmed = [y.lemma_ for y in text_filtered]

    final_str = " ".join(text_stemmed)
    return final_str



def build_vocab(path_to_file, frequency=10):
    vocabulary = defaultdict() 
    vocabulary['UNKNOWN'] = 0
    feature_counter = defaultdict()
    feature_counter[0] = frequency + 1
    for title, text in read_file(path_to_file): 
        title = clean_text_version_2(title).split()
        text = clean_text_version_2(text).split()
        line = title + text 
        for word_item in line:
            if(len(word_item) < 3):
                continue
            if (word_item in vocabulary):
                idx = vocabulary.get(word_item)
                feature_counter[idx] += 1
            else:
                feature_idx = len(vocabulary)
                vocabulary[word_item] = feature_idx
                feature_counter[feature_idx] = 1 
    vocabulary = [token for token, idx in vocabulary.items() if feature_counter[idx] > frequency]
    vocabulary = {token: idx for idx, token in enumerate(vocabulary)}
    return vocabulary

def get_indexes(str_row, vocabulary):
    return [vocabulary.get(item, 0) for item in str_row]


def tokenize_text(path_to_file, vocabulary):
    titles = []
    texts = []
    for title, text in read_file(path_to_file): 
        title = clean_text_version_2(title).split()
        text = clean_text_version_2(text).split()
        title_idx = get_indexes(title, vocabulary)
        if (set(title_idx) == set([0])):
            continue
        text_idx = get_indexes(text, vocabulary)
        if (set(text_idx) == set([0])):
            continue
        titles.append(title_idx)
        if rnd.random():
            text_idx = list(set(text_idx) - set(title_idx))
        texts.append(text_idx)
    return titles, texts    

def scan_text(path_to_file, frequency):
    vocabulary = build_vocab(path_to_file, frequency)
    titles, texts = tokenize_text(path_to_file, vocabulary)
    return vocabulary, titles, texts 



    
#Removes Punctuations
def remove_punctuations(data):
    punct_tag=re.compile(r'[^\w\s]')
    data=punct_tag.sub(r'',data)
    return data

#Removes HTML syntaxes
def remove_html(data):
    html_tag=re.compile(r'<.*?>')
    data=html_tag.sub(r'',data)
    return data

#Removes URL data
def remove_url(data):
    url_clean= re.compile(r"https://\S+|www\.\S+")
    data=url_clean.sub(r'',data)
    return data

#Removes Emojis
def remove_emoji(data):
    emoji_clean= re.compile("["
                           u"\U0001F600-\U0001F64F"  # emoticons
                           u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                           u"\U0001F680-\U0001F6FF"  # transport & map symbols
                           u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                           u"\U00002702-\U000027B0"
                           u"\U000024C2-\U0001F251"
                           "]+", flags=re.UNICODE)
    data=emoji_clean.sub(r'',data)
    url_clean= re.compile(r"https://\S+|www\.\S+")
    data=url_clean.sub(r'',data)
    return data

    
def remove_abb(data):
    data = re.sub(r"&lt;strong&gt;", " ", data)
    data = re.sub(r"&lt;b&gt;", " ", data)
    data = re.sub(r"he's", "he is", data)
    data = re.sub(r"there's", "there is", data)
    data = re.sub(r"We're", "We are", data)
    data = re.sub(r"That's", "That is", data)
    data = re.sub(r"won't", "will not", data)
    data = re.sub(r"they're", "they are", data)
    data = re.sub(r"Can't", "Cannot", data)
    data = re.sub(r"wasn't", "was not", data)
    data = re.sub(r"don\x89Ûªt", "do not", data)
    data= re.sub(r"aren't", "are not", data)
    data = re.sub(r"isn't", "is not", data)
    data = re.sub(r"What's", "What is", data)
    data = re.sub(r"haven't", "have not", data)
    data = re.sub(r"hasn't", "has not", data)
    data = re.sub(r"There's", "There is", data)
    data = re.sub(r"He's", "He is", data)
    data = re.sub(r"It's", "It is", data)
    data = re.sub(r"You're", "You are", data)
    data = re.sub(r"I'M", "I am", data)
    data = re.sub(r"shouldn't", "should not", data)
    data = re.sub(r"wouldn't", "would not", data)
    data = re.sub(r"i'm", "I am", data)
    data = re.sub(r"I\x89Ûªm", "I am", data)
    data = re.sub(r"I'm", "I am", data)
    data = re.sub(r"Isn't", "is not", data)
    data = re.sub(r"Here's", "Here is", data)
    data = re.sub(r"you've", "you have", data)
    data = re.sub(r"you\x89Ûªve", "you have", data)
    data = re.sub(r"we're", "we are", data)
    data = re.sub(r"what's", "what is", data)
    data = re.sub(r"couldn't", "could not", data)
    data = re.sub(r"we've", "we have", data)
    data = re.sub(r"it\x89Ûªs", "it is", data)
    data = re.sub(r"doesn\x89Ûªt", "does not", data)
    data = re.sub(r"It\x89Ûªs", "It is", data)
    data = re.sub(r"Here\x89Ûªs", "Here is", data)
    data = re.sub(r"who's", "who is", data)
    data = re.sub(r"I\x89Ûªve", "I have", data)
    data = re.sub(r"y'all", "you all", data)
    data = re.sub(r"can\x89Ûªt", "cannot", data)
    data = re.sub(r"would've", "would have", data)
    data = re.sub(r"it'll", "it will", data)
    data = re.sub(r"we'll", "we will", data)
    data = re.sub(r"wouldn\x89Ûªt", "would not", data)
    data = re.sub(r"We've", "We have", data)
    data = re.sub(r"he'll", "he will", data)
    data = re.sub(r"Y'all", "You all", data)
    data = re.sub(r"Weren't", "Were not", data)
    data = re.sub(r"Didn't", "Did not", data)
    data = re.sub(r"they'll", "they will", data)
    data = re.sub(r"they'd", "they would", data)
    data = re.sub(r"DON'T", "DO NOT", data)
    data = re.sub(r"That\x89Ûªs", "That is", data)
    data = re.sub(r"they've", "they have", data)
    data = re.sub(r"i'd", "I would", data)
    data = re.sub(r"should've", "should have", data)
    data = re.sub(r"You\x89Ûªre", "You are", data)
    data = re.sub(r"where's", "where is", data)
    data = re.sub(r"Don\x89Ûªt", "Do not", data)
    data = re.sub(r"we'd", "we would", data)
    data = re.sub(r"i'll", "I will", data)
    data = re.sub(r"weren't", "were not", data)
    data = re.sub(r"They're", "They are", data)
    data = re.sub(r"Can\x89Ûªt", "Cannot", data)
    data = re.sub(r"you\x89Ûªll", "you will", data)
    data = re.sub(r"I\x89Ûªd", "I would", data)
    data = re.sub(r"let's", "let us", data)
    data = re.sub(r"it's", "it is", data)
    data = re.sub(r"can't", "cannot", data)
    data = re.sub(r"don't", "do not", data)
    data = re.sub(r"you're", "you are", data)
    data = re.sub(r"i've", "I have", data)
    data = re.sub(r"that's", "that is", data)
    data = re.sub(r"i'll", "I will", data)
    data = re.sub(r"doesn't", "does not",data)
    data = re.sub(r"i'd", "I would", data)
    data = re.sub(r"didn't", "did not", data)
    data = re.sub(r"ain't", "am not", data)
    data = re.sub(r"you'll", "you will", data)
    data = re.sub(r"I've", "I have", data)
    data = re.sub(r"Don't", "do not", data)
    data = re.sub(r"I'll", "I will", data)
    data = re.sub(r"I'd", "I would", data)
    data = re.sub(r"Let's", "Let us", data)
    data = re.sub(r"you'd", "You would", data)
    data = re.sub(r"It's", "It is", data)
    data = re.sub(r"Ain't", "am not", data)
    data = re.sub(r"Haven't", "Have not", data)
    data = re.sub(r"Could've", "Could have", data)
    data = re.sub(r"youve", "you have", data)  
    data = re.sub(r"donå«t", "do not", data)
    return data  

    
def clean_text(text):
    text = remove_abb(text)
    text = remove_punctuations(text)
    text = remove_emoji(text)
    text = remove_html(text)
    text = remove_url(text)
    text = text.lower()
    return text

def calculate_loss(anchor, truth, wrong):
    return max(0, 1 - np.dot(anchor, truth) + np.dot(anchor, wrong))  


def calculate_gradient(anchor, truth, wrong):
    return (-truth + wrong, -anchor, anchor)

#get mean vector
def doc_to_vec(doc_indexes, mtx_embed):
    return np.mean(mtx_embed[doc_indexes], axis = 0)

def get_document_term_sparse_mtx(doc_indices, words_dictionary):
    number_of_docs = len(doc_indices)
    row_indices = [np.full(len(item), idx) for idx, item in enumerate(doc_indices)]
    doc_indices = [item for sublist in doc_indices for item in sublist]
    row_indices = [item for sublist in row_indices for item in sublist]
    values = np.ones(len(doc_indices))
    return csr_matrix((values, (row_indices, doc_indices)), shape=(number_of_docs, len(words_dictionary)))

def shuffle_text(texts):
    shift_value = rnd.randrange(1, len(texts))  
    return texts[shift_value:] + texts[:shift_value]

