#!/usr/bin/python
# -*- coding: utf-8 -*-
# code modified from https://github.com/guillermodoghel/speechgraph.git

from collections import Counter
import re
import numpy as np
import networkx as nx
import nltk.stem
import nltk


class _graphStatistics:
    def __init__(self, graph):
        self.graph = graph

    def statistics(self):
        res = {}
        graph = self.graph

        res["number_of_nodes"] = graph.number_of_nodes()
        res["number_of_edges"] = graph.number_of_edges()

        # Python 3: Counter(...).values()
        edge_counts = Counter(graph.edges())
        res["PE"] = int((np.array(list(edge_counts.values())) > 1).sum())

        weak_components = list(nx.weakly_connected_components(graph))
        strong_components = list(nx.strongly_connected_components(graph))
        res["LCC"] = max((len(c) for c in weak_components), default=0)
        res["LSC"] = max((len(c) for c in strong_components), default=0)

        # Python 3 / newer networkx compatibility
        degrees = [deg for _, deg in graph.degree()]
        if len(degrees) == 0:
            res["degree_average"] = 0.0
            res["degree_std"] = 0.0
        else:
            res["degree_average"] = float(np.mean(degrees))
            res["degree_std"] = float(np.std(degrees))

        # newer networkx: to_numpy_array
        adj_matrix = nx.to_numpy_array(graph, dtype=float)
        adj_matrix2 = np.dot(adj_matrix, adj_matrix)
        adj_matrix3 = np.dot(adj_matrix2, adj_matrix)

        res["L1"] = float(np.trace(adj_matrix))
        res["L2"] = float(np.trace(adj_matrix2))
        res["L3"] = float(np.trace(adj_matrix3))

        return res


class naiveGraph:
    def __init__(self):
        pass

    def _text2graph(self, text, word_tokenizer=lambda x: x.split(" ")):
        cleaned_text = re.sub(r"[^\w ]+", " ", text.lower().strip())
        words = [w for w in word_tokenizer(cleaned_text) if len(w) > 0]

        gr = nx.MultiDiGraph()
        gr.add_edges_from(zip(words[:-1], words[1:]))
        return gr

    def analyzeText(self, text):
        dgr = self._text2graph(text)
        return _graphStatistics(dgr).statistics()


class stemGraph:
    def __init__(self):
        pass

    def _text2graph(
        self,
        text,
        word_tokenizer=lambda x: x.split(" "),
        stemmer=lambda x: nltk.stem.snowball.EnglishStemmer().stem(x),
    ):
        cleaned_text = re.sub(r"[^\w ]+", " ", text.lower().strip())
        words = [w for w in word_tokenizer(cleaned_text) if len(w) > 0]
        stemmed_words = [stemmer(w) for w in words]

        gr = nx.MultiDiGraph()
        gr.add_edges_from(zip(stemmed_words[:-1], stemmed_words[1:]))
        return gr

    def analyzeText(self, text):
        dgr = self._text2graph(text)
        return _graphStatistics(dgr).statistics()


class posGraph:
    def __init__(self):
        pass

    def _text2graph(
        self,
        text,
        word_tokenizer=lambda x: x.split(" "),
        sentence_tokenizer=lambda x: x.split("."),
    ):
        sentences = sentence_tokenizer(text)
        tags = []

        for s in sentences:
            cleaned_text = re.sub(r"[^\w ]+", " ", s.lower().strip())
            words = [w for w in word_tokenizer(cleaned_text) if len(w) > 0]
            if not words:
                continue

            pos_tags = nltk.pos_tag(words)
            tags.extend(tag for _, tag in pos_tags)

        gr = nx.MultiDiGraph()
        gr.add_edges_from(zip(tags[:-1], tags[1:]))
        return gr

    def analyzeText(self, text):
        dgr = self._text2graph(text)
        return _graphStatistics(dgr).statistics()