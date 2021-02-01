import jieba
from jieba import analyse

tf_idf = analyse.extract_tags
text_rank = analyse.textrank

def test_cut():
    sentence = '第二次来吃了，味道还可以，就在京港国际里面，逛累的可以来吃'

    seg_list = jieba.cut(sentence, cut_all=True)
    print('cut_all-True:（全模式）' + '--'.join(seg_list))

    seg_list = jieba.cut(sentence, cut_all=False)
    print('cut_all-False（精确模式）:' + '--'.join(seg_list))

    jieba.add_word('京港国际')
    jieba.add_word('还可以')

    seg_list = jieba.cut(sentence)
    print('jieba.cut（默认模式）' + '--'.join(seg_list))

    # 搜索引擎模式在精确模式的基础上对长词进行划分，提高分词的召回率
    seg_list = jieba.cut_for_search(sentence)
    print('jieba.cut_for_search(搜索引擎模式):' + '--'.join(seg_list))

text = '第一遍我不太建议大家陷入正则，而且这一节讲的正则，并不能帮你学会正则，建议阅读《js正则迷你书》，简单的理解，就是我们规定几个语法，能够方便的从一段文字里匹配到想要的内容'

def test_TF_IDF():
    keywords = tf_idf(text, withWeight=True)
    print('keywords:', keywords)
    for key, val in keywords:
        print(key + '---->' + str(val))
    return keywords

def test_TextRank():
    keywords = text_rank(text, withWeight=True)
    print('keywords:', keywords)
    for key, val in keywords:
        print(key + '---->' + str(val))
    return keywords

if __name__ == '__main__':
    test_TextRank()
