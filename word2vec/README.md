腾讯词向量模型`https://ai.tencent.com/ailab/nlp/embedding.html`是腾讯AI Lab开源的大规模、高质量中文词向量。词向量的维度为200，词汇量为8824330。直接使用这个词向量来求相似度，可以使用gensim，`https://radimrehurek.com/gensim/models/word2vec.html`。

# gensim实现相似度模型

实现代码：

```python
from gensim.models import KeyedVectors

model = KeyedVectors.load_word2vec_format('Tencent_AILab_ChineseEmbedding.txt') # 耗时长
res = model.most_similar("测试")
print(res)
```

使用gensim有以下缺点：

* gensim加载模型耗时很长

* 占用内存很大，会将所有的词向量加载进入内存，占用内存很大（>10g）

* `most_similar`函数耗时较长。gensim使用的算法似乎是暴力求解（待验证），耗时较长，0.35秒



# 使用annoy加载腾讯词向量模型

annoy 是一种临近点搜索算法。github上的简介：Annoy (Approximate Nearest Neighbors Oh Yeah) is a C++ library with Python bindings to search for points in space that are close to a given query point. It also creates large read-only file-based data structures that are mmapped into memory so that many processes may share the same data.

* 底层算法使用c++实现，有python接口

* 将基于文件的数据映射到内存，因此只需要少量的内存

* 搜索结果与暴力搜素效果接近

## 建立Annoy模型

模型已经实现，主要有以下步骤，github地址：

1. 对腾讯词向量的词进行编码

    annoy的输入格式是(item_id,item_vector)，因此，需要对词向量的词进行数字编码。为了节省内存，我将腾讯词向量保存在mongodb，包括word与annoy_index两个字段。
    `update_word_index_mongo.py`

    需要搭建一个mongodb，参考`https://docs.mongodb.com/manual/administration/install-community/`

2. 建立Annoy模型

    使用腾讯词向量及Annoy的python接口，建立Annoy模型，这个过程需要占用很大的内存，花费比较长的时间。建立方法参考：`https://github.com/spotify/annoy`

    `build_annoy.py`

3. 使用

`annoy_similar.py`

```
from annoy_similar import annoy_model
model=annoy_model()
res=model.get_most_similar('你好')
print(res)
```

annoy模型使用的是Euclidean distance， 这里使用公式将其转化成了余弦距离，公式为 Euclidean_distance=sqrt(2-2*cos(u,v))


## 环境

Ubuntu 16.04
python 3.6

python库：
    tqdm
    gensim
    pymongo
    annoy
