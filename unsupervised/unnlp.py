import spacy
import pandas as pd
import numpy as np
from sklearn import neighbors
from sklearn.metrics.pairwise import cosine_similarity

spacy_model = spacy.load("en_core_web_sm")

class NLP():
    
    def __init__(self, label_names, text, model):
        self.label_names = label_names
        self.text = text
        self.model = model
        
    def embed(self, tokens):
        self.tokens = tokens
        
        all_stopwords = self.model.Defaults.stop_words
        words = [x for x in self.tokens if x not in all_stopwords]
        docs = np.asarray([self.model(x).vector for x in words])

        if len(docs) > 0:
            centroid = docs.mean(axis=0)
        else:
            width = self.model.meta['vectors']['width']  # typically 300
            centroid = np.zeros(width)

        return centroid
  
    def text_classification(self, neighbors, number=None):
        
        self.neighbors = neighbors
        self.number = number
        
        if self.number == None:
            self.number = 1
        
        label_vector = np.array([self.embed(x) for x in self.label_names])
        neigh = self.neighbors.NearestNeighbors(n_neighbors=self.number)
        neigh.fit(label_vector)
        centroid = self.embed(self.text.split(' '))
        closest_label = neigh.kneighbors([centroid], return_distance=True)
        
        labels = []
        for i in range(0, self.number):
            labels.append([self.label_names[closest_label[1][0][i]], closest_label[0][0][i]])

        return labels
