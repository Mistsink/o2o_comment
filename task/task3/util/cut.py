"""
jieba分词
"""
import pandas as pd
import jieba

stop_words = []


def __init__():
    stop_words = pd.read_table('../../../static/stop.txt', encoding='utf8', header=None)[0].to_list()


def cut_text(sentence):
    """
    去停用词、jieba分词
    :param sentence: string
    :return: string
    """
    tokens = jieba.lcut(sentence)
    # print('sentence:', sentence)
    tokens = [token for token in tokens if token not in stop_words]
    # print('tokens:', tokens)
    return ' '.join(tokens)
    return tokens


if __name__ == '__main__':
    pass
