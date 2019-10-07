# -*- coding:utf-8 -*-
import sys
sys.path.append('./src')
from common.utils import get_sentences,stopwords,punc_pattern
from typing import List,Set,Dict
import jieba
import math
import networkx as nx
import pandas as pd
import re


def tokenize(string):
    string = string.strip()
    return list(jieba.cut(string))

def remove_punc(sentence:List[str]):
    return [token for token in sentence if not re.search(punc_pattern,token)]

def remove_stopwords(sentence:List[str]):
    return [token for token in sentence if token not in stopwords]

def sentence_sim(sentence1:List[str],sentence2:List[str]):
    '''
    句子相似度计算公式参考pagerank原始论文实现
    '''
    if len(sentence1) == 0 or len(sentence2) == 0:
        return 0
    overlap = set(sentence1) & set(sentence2)
    if math.log(len(sentence1)) + math.log(len(sentence2)) == 0:
        return 0
    sim = len(overlap) / (math.log(len(sentence1)) + math.log(len(sentence2)))
    return sim

def generate_sentence_graph(sentences:List[str]):
    graph = nx.Graph()
    for i in range(len(sentences) - 1):
        sent_i = sentences[i]
        for j in range(i+1, len(sentences)):
            sent_j = sentences[j]
            sim = sentence_sim(remove_stopwords(remove_punc(tokenize(sent_i))),remove_stopwords(remove_punc(tokenize(sent_j))))
            if sim > 0:
                graph.add_edge(i,j)
                graph.edges[i,j]['weight'] = sim
    return graph

def auto_summary(content,sentence_nums=5):
    sentences = get_sentences(content)
    graph = generate_sentence_graph(sentences)
    pr = nx.pagerank(graph, weight='weight', alpha=0.9)
    selected_sent_nums = sorted(pr.items(),key=lambda x:x[1],reverse=True)[:sentence_nums]
    sorted_nums = sorted([num for num,pr_value in selected_sent_nums])
    return ''.join([sentences[num] for num in sorted_nums])

def load_data():
    data = pd.read_csv('./data/sqlResult_example.csv',encoding='gb18030')
    return data

def main():
    data = load_data()
    content = data.iloc[5]['content']
    summary = auto_summary(content,sentence_nums=3)
    print('news : {}'.format(content))
    print('summary : {}'.format(summary))

if __name__ == '__main__':
    main()