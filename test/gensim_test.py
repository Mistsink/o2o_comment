import gensim

sentences = ['凉皮 有 味道 了 吃完 一天 肚子 都 不舒服'.split(),
                 '帅哥 经理 又 帅 服务 又 好 凉皮 味道 又 不错 吃完 肚子 撑'.split()]

def word2vec():
    print('sentence: ', sentences)

    model = gensim.models.Word2Vec(sentences,
                                   sg=1, size=100, window=5, min_count=2,
                                   negative=3, sample=1e-3, hs=1, workers=2)
    model.wv.save_word2vec_format('../static/word2vec_model.txt', binary=False)
    model = gensim.models.KeyedVectors.load_word2vec_format('../static/word2vec_model.txt', binary=False)
    print('model: ', model)
    print('model["凉皮"]:', model['凉皮'])

def dic_corpus(texts):
    print('texts: ', texts)
    dic = gensim.corpora.Dictionary(texts)
    corpus = [dic.doc2bow(text) for text in texts]
    return dic, corpus

def text2tfidf():
    # model = gensim.models.Word2Vec(sentences, min_count=2, sg=1, negative=3)
    dic, corpus = dic_corpus(sentences)
    print('dic: ', dic)
    print('corpus: ', corpus)
    tf_idf = gensim.models.TfidfModel(corpus)
    corpus_idf = tf_idf[corpus]
    index = 1
    for doc in corpus_idf:
        print('the ', index, ' doc: ', doc)
        index+=1



if __name__ == '__main__':
    text2tfidf()