# -*- coding: utf-8 -*-
"""
Adapted from code by Dr Sarah E Morgan (21/02/2021), based on Morgan et al. (2021):
https://doi.org/10.1101/2021.01.04.20248717 

Please cite Morgan et al. (2021) if you use this code or adapt it in your own work.
"""

import itertools
import numpy as np
from collections import Counter
from sklearn.decomposition import PCA
import nltk
import re
import string 
import sklearn
import gensim
import sys
from basic_meas import *

###### Use downloaded word2vec model: # https://github.com/eyaler/word2vec-slim
model = gensim.models.KeyedVectors.load_word2vec_format(r'c:\Users\guoji\Downloads\GoogleNews-vectors-negative300-SLIM.bin.gz', binary=True)
embedding_size = 300 


# function to take a sentence and pre-process, namely: expand contractions, remove stop words and remove punctuation
def process_sent(sent):
    stop_words = nltk.corpus.stopwords.words('english') # this is the list of stop words to exclude
    newStopWords = ['Um','um','Uh','uh','Eh','eh','Ehm','Em','em','Mmm','mmm','ah','Ah','Aah','aah','hmm','hmmm','Hmm','Hmmm','inaudible','Inaudible']
    #, 'umm','Umm','ummm','Ummm','ummmm','Ummmm','ummmmm','Ummmmm'
    stop_words.extend(newStopWords) # adds the new stop words above to the dictionary
     
    tokens = nltk.word_tokenize(sent) # tokenize sentence
    tokens = [t.lower() for t in tokens if t not in string.punctuation] # remove punctuation
    tokens = [w for w in tokens if not w in stop_words] # remove stopwords
    sent_proccess = ' '.join(tokens)
    return sent_proccess


# function to take a text and output pre-processed sentences:
def text2procsent(text):
     
    sentences=nltk.tokenize.sent_tokenize(text) # tokenize the text into sentences
    
    sentences_process=[] # pre-process the sentences to expand contractions, remove stop words and remove punctuation
    for sent in sentences:
        sent=process_sent(sent)
        sentences_process.append(sent)
        
    sentences_process = [x for x in sentences_process if x]
    
    sentences_split = [sentence.split() for sentence in sentences_process]
    
    return sentences_split


# function to count word frequency:
def map_word_frequency(document):
    return Counter(itertools.chain(*document))


# sentence2vec takes pre-processed, tokenised sentence list and outputs sentence embedding
# here we're using SIF sentence embedding (as proposed by: https://openreview.net/pdf?id=SyK00v5xx)
def sentence2vec(tokenised_sentence_list, embedding_size, word_emb_model, a = 1e-3):

    """
    Computing weighted average of the word vectors in the sentence;
    remove the projection of the average vectors on their first principal component.
    Taken from https://github.com/peter3125/sentence2vec and https://gist.github.com/bluemonk482/a4b2de9b5037d9ad69fa82da6ae67641
    """

    word_counts = map_word_frequency(tokenised_sentence_list)
    sentence_set=[]

    for sentence in tokenised_sentence_list:
        vs = np.zeros(embedding_size)
        sentence_length = len(sentence)
        for word in sentence:
            a_value = a / (a + word_counts[word]) # smooth inverse frequency, SIF
            try:
                vs = np.add(vs, np.multiply(a_value, word_emb_model[word])) # vs += sif * word_vector
            except:
                pass
        vs = np.divide(vs, sentence_length) # weighted average
        sentence_set.append(vs)

    # calculate PCA of this sentence set:
    pca = PCA(n_components=len(sentence_set))
    pca.fit(np.array(sentence_set))
    u = pca.explained_variance_ratio_  # the PCA vector
    u = np.multiply(u, np.transpose(u))  # u x uT

    if len(u) < embedding_size:
        for i in range(embedding_size - len(u)):
            u = np.append(u, 0)  # add needed extension for multiplication below

	# resulting sentence vectors, vs = vs - u x uT x vs
    sentence_vecs = []
    for vs in sentence_set:
        sub = np.multiply(u,vs)
        sentence_vecs.append(np.subtract(vs, sub))

    return sentence_vecs
    


# function to calculate features (coherence and max off diag) from the sentence embedding:
def coh_features(embedding):
    mymatrix=sklearn.metrics.pairwise.cosine_similarity(embedding) # calculate similarity matrix from sentence embedding
    nosent=len(mymatrix) # no. of sentences with relevant words included (note, sentences containing only stopwords aren't included here, so this may be different to the no. sentences calculated by basic_meas).
    
    # calculate coherence:
    coh=0
    for index in range(0,nosent-1):
        coh += (mymatrix[index,1+index])/(len(range(0,nosent-1)))
    
    # calculate max off-diagonal:
    mymatrix0=mymatrix
    np.fill_diagonal(mymatrix0, 0) # sets diagonal to zero in mymatrix0
    maxoffdiag=np.max(mymatrix0) # max off-diagonal (repetition measure)
    
    return coh, maxoffdiag




# meas_coh takes a text and outputs the metrics coherence and max off diagonal.
# It returns NaNs if there are <2 sentences containing non-stopwords, or all sentences contain exactly the same set of non-stopwords.
def meas_coh(text):
    
    nowords=get_no_words(text)
    
    if nowords==0:
        raise ValueError("no valid words after preprocessing")
       
    sentences_split = text2procsent(text)
    
    if len(sentences_split)<2:
        coh_result = float('NaN'), float('NaN') # if <2 sentences, return NaNs
        print("Error: fewer than two sentences provided.")
    elif all(x==sentences_split[0] for x in sentences_split): # if all sentences are the same, return NaNs
        coh_result = float('NaN'), float('NaN')
        print("Error: all sentences contain the same set of non-stop words.")
    else:
        embedding = sentence2vec(sentences_split, embedding_size, model) # calculate embedding
        if np.nansum(np.nansum(embedding))==0: # if embedding is all zeros/NaNs, return NaNs.
            coh_result = float('NaN'), float('NaN')
        else:
            coh_result = coh_features(embedding) # otherwise, extract the features from the embedding (coherence and max off diag)
        
    return coh_result

def extract_semantic_features(text: str):
    if not text or not text.strip():
        return {
            "semantic_coherence": None,
            "maximum_similarity": None
        }

    try:
        coherence, max_similarity = meas_coh(text)

        return {
            "semantic_coherence": coherence,
            "maximum_similarity": max_similarity
        }

    except Exception as e:
        print("[semantic feature error]", e)
        return {
            "semantic_coherence": None,
            "maximum_similarity": None
        }