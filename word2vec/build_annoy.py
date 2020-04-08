# -*- encoding: utf-8 -*-
'''
@Author  :   wangjian
'''

# 使用腾讯词向量，建立annoy模型

from annoy import AnnoyIndex
import os
from tqdm import tqdm

# 下载腾讯词向量 https://ai.tencent.com/ailab/nlp/embedding.html
tencent_file = '/home/wangjian0110/Tencent_AILab_ChineseEmbedding/Tencent_AILab_ChineseEmbedding.txt'

def get_word_vec():
    with open(tencent_file,'r',encoding='utf8') as f:
        with tqdm(total=8824330) as pbar:
            for line in f:
                pbar.update(1)
                line=line.strip('\n')
                raw_line = line
                line=line.split(' ')
                if len(line)!=201:
                    print(raw_line)
                else:
                    word = line[0]
                    vec = [float(i) for i in line[1:]]
                    assert len(vec)==200
                    yield word,vec
def build_annoy_index(emb_size=200,metric='angular',num_trees=100):
    t=AnnoyIndex(emb_size,metric)
    num=-1
    for word,vec in get_word_vec():
        num+=1
        t.add_item(num,vec)
        # if num==100:
        #    break
        # print(word)
    t.build(num_trees)
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    save_file = os.path.join(cur_dir,'word2vec.ann')
    t.save(save_file)

if __name__=="__main__":
    build_annoy_index()  # 建立索引    i
