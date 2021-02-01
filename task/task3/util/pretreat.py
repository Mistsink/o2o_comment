"""
文本预处理方法
"""

import re


def Filter(text):
    """
    提取文字
    :param text: source string
    :return: string
    """
    text = re.sub(r'[A-Za-z0-9\!\=\?\%\[\]\,\ \(\)\（\）\>\<:&lt;\/#\. -----\_]', '', text)
    text = text.replace('图片', '')
    text = text.replace('\xa0', '') # 删除nbsp(即空格)

    cleaner = re.compile('<.*?>')
    text = re.sub(cleaner, ' ', text)
    text = re.sub(r'\\【.*?】+|\\《.*?》+|\\#.*?#+|[.!/_,$&%^*()～<>+?@|:❤☺~{}#]+|[——！\\\，。=？、：“”‘’￥……（）《》【】]', '', text)
    text = text.strip()
    return text


if __name__ == '__main__':
    print('\xa0')