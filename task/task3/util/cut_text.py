"""
jieba分词
"""
import pandas as pd
import jieba

stop_words = []


def __init__():
    stop_words = pd.read_table('../../../static/stop.txt', encoding='utf8', header=None)[0].to_list()


def cut_text(sentence, if_cut_stop_words=True, if_cut_all=False):
    """
    去停用词、jieba分词
    :param if_cut_all: boolean
    :param if_cut_stop_words: boolean
    :param sentence: string
    :return: string
    """
    tokens = jieba.lcut(sentence, cut_all=if_cut_all)
    # print('sentence:', sentence)
    if if_cut_stop_words:
        tokens = [token for token in tokens if token not in stop_words]
    # print('tokens:', tokens)
    return ' '.join(tokens)
    return tokens


if __name__ == '__main__':
    pass
