
import json
import jieba
import re
import gensim


def read_proto():
    file = open('../../data/test.txt')

    lines = file.readlines()

    stirs = []

    for line in lines:
        j = json.loads(line)
        str =''
        for key in j.keys():
            if key != 'newsId':
                # 提取中文
                p = re.compile(r'[\u4e00-\u9fa5]')
                res = re.findall(p, j[key])
                result = ''.join(res)
                str+=result
        stirs.append(str)

    file.close()

    return stirs


if __name__ == '__main__':
    text = read_proto() # return [str]
    corpus = []
    for txt in text:
        seg_list = jieba.cut(txt) # 精确模式
        corpus.append(seg_list)
    print('corpus[0]: ', '-'.join(corpus[0]))

    # gensim
    sentences = [" ".join(seg_list).split() for seg_list in corpus]
    model = gensim.models.Word2Vec(sentences,
                                   sg=1, size=80, window=5, min_count=2,
                                   negative=3, sample=1e-3, hs=1, workers=2)
    model.wv.save_word2vec_format('../../static/word2vec_model.txt', binary=False)
