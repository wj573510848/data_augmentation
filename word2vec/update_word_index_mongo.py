# -*- encoding: utf-8 -*-
'''
@Author  :   wangjian
'''

import pymongo
from tqdm import tqdm


# 将annoy的index写入mongo
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
                    yield word

def update_index():
    client = pymongo.MongoClient()
    db = client.word2vec
    collection = db.tencent_weight
    num=-1
    for word in get_word_vec():
        num+=1
        collection.update_one({'word':word},{"$set":{"annoy_index":num}})
    collection.create_index('word')
    collection.create_index('annoy_index')
if __name__=="__main__":
    update_index()
