import numpy as np
import matplotlib.pylab as plt
from model import EmbeddingModel
import sgd

vocabulary, titles, texts = sgd.scan_text('train.csv', 10)

ratio = 0.8
middle = int(len(titles) * ratio)
train_titles, val_titles = titles[:middle], titles[middle:]
train_texts, val_texts = texts[:middle], texts[middle:]


model = EmbeddingModel(vocabulary)
model.train(train_titles, train_texts, val_titles, val_texts, number_of_epoch = 30)