# -*- coding:utf-8 -*-
from common.utils import get_sentences,cos_similarity
from common.SentSim import SentEmbedding
from typing import List,Set,Dict
import pandas as pd

def load_data():
    data = pd.read_csv('./data/sqlResult_example.csv',encoding='gb18030')
    return data

def get_all_sents(contents:List[str]):
    all_sents = []
    for news in contents:
        all_sents.append(get_sentences(news))
    return all_sents

def get_similarities(news_embd, sentences_embedding):
    similarities = []
    for i,embd in enumerate(sentences_embedding):
        if len(embd) == 0:
            continue
        similarities.append((i,cos_similarity(news_embd, embd)))
    return similarities

def sent_embedding(all_sents):
    sent_embd = SentEmbedding()
    sent_embd.prepare(all_sents)
    return sent_embd

def get_summary(sent_embd:SentEmbedding, content:str, sents_limit=5):
    news = content.strip().replace('\r\n','').replace('\n','')
    sentences = get_sentences(content)
    news_embedding, news_no_ind = sent_embd.sents_embedding([news])
    sentences_embedding, sents_no_ind = sent_embd.sents_embedding(sentences)
    sentences_embedding = sentences_embedding.tolist()
    for indx in sents_no_ind:
        sentences_embedding.insert(indx,[])
    similarities = get_similarities(news_embedding, sentences_embedding)
    similarities.sort(key=lambda x:x[1],reverse=True)
    sents_nums = [num for num,sim in similarities[:sents_limit]]
    return ''.join([sentences[i] for i in sorted(sents_nums)])


def main():
    data = load_data()
    all_sents = get_all_sents(data['content'].tolist())
    sent_embed = sent_embedding(all_sents)
    content = data.iloc[4]['content']
    summary = get_summary(sent_embed, content, sents_limit=5)
    print('news : {}'.format(content))
    print('summary : {}'.format(summary))
