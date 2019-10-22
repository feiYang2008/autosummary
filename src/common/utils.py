from opencc import OpenCC
import codecs
import os
import re
import string
import json
from scipy import spatial
import jieba
from typing import List
import math
from gensim.models.word2vec import PathLineSentences
from gensim.models.word2vec import Word2Vec
from gensim.models.ldamodel import LdaModel
from gensim.corpora.dictionary import Dictionary


OP = OpenCC('t2s')
sent_cut_pattern = [
    re.compile(r'([。？！?])([^"\'”])'),
    re.compile(r'(\.{6})([^"\'”])'),
    re.compile(r'([。？！?]["\'”])([^\'"”])'),
]
zh_pattern = re.compile(r'^[\u4e00-\u9fa5]+$')
puncs = string.punctuation + '.,;《》？！“”‘’@#￥%…&×（）——+【】{};；●，。&～、|\s:：'
punc_pattern = re.compile(r'[{}]+'.format(puncs))

stopwords = []
with open('./data/chinese_stopwords.txt','r',encoding='utf8') as f:
    for line in f:
        line = line.strip()
        if len(line) > 0:
            stopwords.append(line.strip())
stopwords = set(stopwords)


def cut_sent(para):
    '''
    句子切分
    :param para:
    :return:
    '''
    for pat in sent_cut_pattern:
        para = pat.sub(r'\1\n\2',para)
    return para.split('\n')

def get_sentences(content):
    sents = []
    content = content.strip().replace('\r\n','\n')
    for para in content.split('\n'):
        sents.extend(cut_sent(para))
    return sents

def init_folder(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


def write_json(path,content):
    with codecs.open(path,'w',encoding='utf8') as f:
        json.dump(content,f)


def read_json(path):
    with codecs.open(path,'r',encoding='utf8') as f:
        data = json.load(f)
    return data

def cos_similarity(embd1,embd2):
    dis = spatial.distance.cosine(embd1,embd2)
    sim = 1. - dis
    return sim


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