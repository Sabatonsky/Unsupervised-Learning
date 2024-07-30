# -*- coding: utf-8 -*-
"""
Created on Wed Nov 22 15:48:59 2023

@author: Bannikov Maxim
"""

import numpy as np
import re
import pandas as pd
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from scipy.spatial.distance import pdist
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
nltk.download('stopwords')

def filter_tweet(s):
  url_finder = re.compile(r"(?:\@|https?\://)S+")
  s = s.lower()
  s = url_finder.sub("", s)
  return s

def purity(Y, C, categories):
    total = 0.0
    N = len(Y)
    for k in categories:
        max_intersection = 0
        for j in categories:
            intersection = ((C == k) & (Y == j)).sum()
            if intersection > max_intersection:
                max_intersection = intersection
        total += max_intersection
    return total / N

def main():
  #stops = list(stopwords.words('english'))
  stops = [
    'https', 
    'co', 
    'the',
    'about',
    'an',
    'and',
    'are',
    'at',
    'be',
    'can',
    'for',
    'from',
    'if',
    'in',
    'is',
    'it',
    'of',
    'on',
    'or',
    'that',
    'this',
    'to',
    'you',
    'your',
    'with',
  ]

  df = pd.read_csv(r"D:\Training_code\tweets.csv")
  text = df.text.tolist()
  text = [filter_tweet(s) for s in text]

  tfidf = TfidfVectorizer(max_features = 100, stop_words = stops)
  X = tfidf.fit_transform(text).toarray()
  N = X.shape[0]
  idx = np.random.choice(N, size = 2000, replace = False)
  x = X[idx]
  labels = df.handle[idx].tolist()
  pTrump = labels.count('realDonaldTrump') / len(labels)
  print('Hillary proportion:', 1 - pTrump)
  print('Trump proportion:', pTrump)

  dist_array = pdist(x)

  Z = linkage(dist_array, 'ward')
  print('Z.shape', Z.shape)
  plt.title('Ward')
  dendrogram(Z, color_threshold=8, labels=labels)
  plt.show()

  Y = np.array([1 if e=='realDonaldTrump' else 2 for e in labels])
  C = fcluster(Z, 8, criterion = 'distance')
  categories = set(C)
  print('values in C:', categories)
  
  p = purity(Y, C, categories)
  print("Total purity:", p)
  
  if (C == 1).sum() < (C == 2).sum():
    d = 1
    h = 2
  else:
    d = 2
    h = 1
    
  actually_donald = ((C == d) & (Y == 1)).sum()
  donald_cluster_size = (C == d).sum()
  print("purity of Trump cluster:", float(actually_donald)/donald_cluster_size)
  
  actually_hillary = ((C == h) & (Y == 2)).sum()
  hillary_cluster_size = (C == h).sum()
  print("purity of Hillary cluster:", float(actually_hillary)/hillary_cluster_size)
  
  from sklearn.ensemble import RandomForestClassifier
  
  rf = RandomForestClassifier()
  rf.fit(x, Y)
  print("classifier score:", rf.score(x, Y))
  
  w2i = tfidf.vocabulary_ #Запросили у tfidf словарь слов.
  
  d_avg = np.array(x[C==d].mean(axis=0)).flatten() #Вынимаем твиты, которые распознаны, как твиты трампа. Ищем среднее колво каждого слова по твитам.
  d_sorted = sorted(w2i.keys(), key=lambda w: -d_avg[w2i[w]]) #Сортируем ключи (сами слова) по убыванию среднего.
  
  print("Top 10 Trump cluster words:")
  print("\n".join(d_sorted[:10]))

  h_avg = np.array(x[C==h].mean(axis=0)).flatten() #Вынимаем твиты, которые распознаны, как твиты трампа. Ищем среднее колво каждого слова по твитам.
  h_sorted = sorted(w2i.keys(), key=lambda w: -h_avg[w2i[w]]) #Сортируем ключи (сами слова) по убыванию среднего.
  
  print("Top 10 Hillary cluster words:")
  print("\n".join(h_sorted[:10]))
  
if __name__ == '__main__':
  main()
