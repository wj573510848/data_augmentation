# -*- encoding: utf-8 -*-
'''
@Author  :   wangjian
'''

from annoy import AnnoyIndex
import os
import pymongo
import traceback

class annoy_id_model:
    def __init__(self):
        self._init_mongo()
    
    def _init_mongo(self):
        client = pymongo.MongoClient()
        db = client.word2vec
        self.collection = db.tencent_weight

    def word2id(self, word):
        res = self.collection.find_one({'word':word},{"_id":False,'word':True,'annoy_index':True})
        if res:
            id_ = int(res['annoy_index'])
            return id_
        return None
    
    def id2word(self, input_ids : list):
        res = self.collection.find({'annoy_index':{"$in":input_ids}},{'_id':False,'word':True,'annoy_index':True})
        id2word = {}
        for i in res:
            print(i)
            id2word[i['annoy_index']] = i['word']
        return id2word

class annoy_model:
    def __init__(self):
        self._load()
    
    def _load(self):
        cur_dir = os.path.dirname(os.path.abspath(__file__))
        model_file = os.path.join(cur_dir,'word2vec.ann')
        
        self.model = AnnoyIndex(200,'angular')
        self.model.load(model_file)
        self.id_model = annoy_id_model()

    def get_most_similar(self, word, topn=5):
        try:
            word_id = self.id_model.word2id(word)
            if word_id is None:
                return None
            else:
                raw_res = self.model.get_nns_by_item(i=word_id,n=topn,include_distances=True)
                res_ids, res_scores = raw_res
                id2word = self.id_model.id2word(res_ids)
                res_words = [id2word[i] for i in res_ids]
                # sqrt(2(1-cos(u,v)))
                res_scores = [self.to_cosin_distance(i) for i in res_scores]
                return (res_words,res_scores)
        except :
            print(traceback.format_exc())
            return None
    def to_cosin_distance(self,euclidean_distance):
        # sqrt(2(1-cos(u,v)))
        cosin_d = 1-euclidean_distance**2/2
        return cosin_d#0.5*(1+cosin_d)

if __name__=="__main__":
    model = annoy_model()
    res = model.get_most_similar('你好')
    print(res)
            
        


