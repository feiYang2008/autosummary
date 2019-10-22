# -*- coding:utf-8 -*-
import sys
sys.path.append('./src')
from common.utils import get_sentences,cos_similarity,cut_sent
from common.SentSim import SentEmbedding
from typing import List,Set,Dict
import pandas as pd
import numpy as np


class AutoSummary(object):
    def __init__(self):
        self.sent_embed = SentEmbedding()
        self.sim_threshold = 0.9

    def prepare_data(self):
        data = load_data()
        all_sents = get_all_sents(data['content'].dropna().tolist())
        self.sent_embed.prepare(all_sents)

    def summary(self, content:str, sent_limit=3):
        news = content.strip().replace('\r\n', '').replace('\n', '')
        sentences, weight_map = split_content(content)
        news_embedding, news_no_ind = self.sent_embed.sents_embedding([news])
        sentences_embedding, sents_no_ind = self.sent_embed.sents_embedding(sentences)
        sentences_embedding = sentences_embedding.tolist()
        for indx in sents_no_ind:
            sentences_embedding.insert(indx, [])
        similarities_content = get_similarities(news_embedding,
                                                sentences_embedding)
        similarities_combine_weight = combine_weight(similarities_content,
                                                     weight_map)
        similarities_combine_weight.sort(key=lambda x: x[1], reverse=True)
        cand_index = [index for index, sim in similarities_combine_weight][:5]
        cand_index = self.remove_repeate(cand_index, sentences_embedding,
                                         sim_threshold=self.sim_threshold)
        return ''.join([sentences[i] for i in sorted(cand_index[:sent_limit])])

    def summary_with_title(self, content:str, title:str, sent_limit=3):
        news = content.strip().replace('\r\n', '').replace('\n', '')
        title = title.replace('\r\n', '').replace('\n', '')
        sentences, weight_map = split_content(content)
        news_embedding, news_no_ind = self.sent_embed.sents_embedding([news])
        sentences_embedding, sents_no_ind = self.sent_embed.\
            sents_embedding(sentences)
        title_embedding, title_no_ind = self.sent_embed.sents_embedding([title])
        sentences_embedding = sentences_embedding.tolist()
        for indx in sents_no_ind:
            sentences_embedding.insert(indx, [])
        similarities_content = get_similarities(news_embedding,
                                                sentences_embedding)
        similarities_title = get_similarities(title_embedding,
                                              sentences_embedding)
        res_sim = get_combined_similarity(similarities_title,
                                          similarities_content, weight=0.3)
        similarities_combine_weight = combine_weight(res_sim, weight_map)
        similarities_combine_weight.sort(key=lambda x: x[1], reverse=True)
        cand_index = [index for index, sim in similarities_combine_weight][:5]
        cand_index = self.remove_repeate(cand_index, sentences_embedding,
                                         sim_threshold=self.sim_threshold)
        return ''.join([sentences[i] for i in sorted(cand_index[:sent_limit])])

    def remove_repeate(self, cand_index, sentences_embedding,
                       sim_threshold=0.6):
        accumulate_embed = [sentences_embedding[cand_index[0]]]
        res_cand = [cand_index[0]]
        for index in cand_index[1:]:
            embedding = sentences_embedding[index]
            embedding_acc = sum([np.array(emb) for emb in accumulate_embed]) \
                            / len(accumulate_embed)
            similarity = cos_similarity(embedding_acc, embedding)
            if similarity <= sim_threshold:
                res_cand.append(index)
                accumulate_embed.append(embedding)
        return res_cand


def get_combined_similarity(title_similarity, content_similarity, weight=0.5):
    title_sim = dict(title_similarity)
    res_sim = []
    for index,sim in content_similarity:
        if index in title_sim:
            res_sim.append((index,(1-weight)*sim + weight * title_sim[index]))
    return res_sim


def combine_weight(similarities, weights):
    res = []
    for index, sim in similarities:
        res.append((index, weights.get(index,1)*sim))
    return res

def split_content(content:str):
    '''
    首尾段和每段首尾句加权
    '''
    content = content.strip().replace('\r\n','\n')
    paras = content.split('\n')
    sentences = []
#     首尾段加权权重
    para_weight = 1.2
#     每段首尾句加权权重
    sent_weight = 1.2
    weight_map = {}
    for i,para in enumerate(paras):
        weight = 1
        if i == 0 or i == len(paras) -1:
            weight *= para_weight
        sent_split = cut_sent(para)
        for j,sent in enumerate(sent_split):
            weight_final = weight
            if j == 0 or j == len(sent_split) - 1:
                weight_final= weight * sent_weight
            weight_map[len(sentences)] = weight_final
            sentences.append(sent)
    return sentences,weight_map


def load_data():
    data = pd.read_csv('./data/sqlResult_1558435.csv',encoding='gb18030')
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

def main():
    auto_summary = AutoSummary()
    auto_summary.prepare_data()
    data = load_data()
    content = data.iloc[4]['content']
    summary = auto_summary.summary(content)
    print('news : {}'.format(content))
    print('summary : {}'.format(summary))


if __name__ == '__main__':
    main()