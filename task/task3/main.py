import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer as TFIDF
from sklearn import svm

from task.task3.util.cut_text import cut_text

# key config
from task.task3.util.matrix_train import matrix_train
from task.task3.util.pretreat import Filter

KEY_COMMENT = 'comment'

train_text = pd.read_csv('../../data/train.csv', sep='\t')
test_text = pd.read_csv('../../data/test_new.csv')


train_text[KEY_COMMENT] = train_text[KEY_COMMENT].apply(lambda x: Filter(x))
test_text[KEY_COMMENT] = test_text[KEY_COMMENT].apply(lambda x: Filter(x))

train_comments_cut = [cut_text(sentence, True, True) for sentence in train_text[KEY_COMMENT].values]
test_comments_cut = [cut_text(sentence, True, True) for sentence in test_text[KEY_COMMENT].values]

tfidf = TFIDF(
    min_df=1,  # 最小支持长度
    max_features=150000,  # 取特征数量
    strip_accents='unicode',
    analyzer='word',
    token_pattern=r'\w{1,}',
    ngram_range=(1, 3),
    use_idf=1,
    smooth_idf=1,
    sublinear_tf=1,
)

X_train, _, X_test = matrix_train(train_text['label'], train_comments_cut, test_comments_cut, tfidf)

clf = svm.LinearSVC(loss='squared_hinge', dual=True, tol=0.0001,
                    C=1.0, multi_class='ovr', fit_intercept=True, intercept_scaling=1,
                    class_weight='balanced', verbose=0, random_state=None, max_iter=1000)
svm = clf.fit(X_train, train_text['label'])
svm_pre = svm.predict(X_test)
svm = pd.DataFrame(data=svm_pre, columns=['comment'])
svm['id'] = test_text['id']
svm = svm[['id', 'comment']]
svm.to_csv('./result/t_3_idf&svm.csv', index=False)

a = {17, 1753, 722, 1753, 531, 1296, 74, 229, 299, 529,
     642, 715, 761, 1335, 1411, 1660, 1675, 1682, 1744, 1767, 1780, 1907,
     238, 978, 1300, 1953, 229, 697, 328, 886, 1350, 1369, 1413, 1985, 1753,
     501, 1708, 1751, 1941, 1963, 531, 595, 1079, 1296, 1702, 613, 722,
     1111, 1556, 1556, 662, 1844
     }

"""
import re


def extractChinese(s):
    pattern = "[\u4e00-\u9fa5]+"  # 中文正则表达式
    regex = re.compile(pattern)  # 生成正则对象
    results = regex.findall(s)  # 匹配
    return "".join(results)


# 预处理数据
label = train_text['label']
train_data = []
for i in range(len(train_text['comment'])):
    train_data.append(' '.join(extractChinese(train_text['comment'][i])))
test_data = []
for i in range(len(test_text['comment'])):
    test_data.append(' '.join(extractChinese(test_text['comment'][i])))

tfidf = TFIDF(min_df=1,  # 最小支持长度
              max_features=150000,  # 取特征数量
              strip_accents='unicode',
              analyzer='word',
              token_pattern=r'\w{1,}',
              ngram_range=(1, 3),
              use_idf=1,
              smooth_idf=1,
              sublinear_tf=1,
              stop_words=None,

              )

# 合并训练和测试集以便进行TFIDF向量化操作
data_all = train_data + test_data
len_train = len(train_data)

tfidf.fit(data_all)

data_all = tfidf.transform(data_all)

# 恢复成训练集和测试集部分
train_x = data_all[:len_train]
test_x = data_all[len_train:]
print('TF-IDF处理结束.')

clf = svm.LinearSVC(loss='squared_hinge', dual=True, tol=0.0001,
                    C=1.0, multi_class='ovr', fit_intercept=True, intercept_scaling=1,
                    class_weight='balanced', verbose=0, random_state=None, max_iter=1000)
svm = clf.fit(train_x, label)
svm_pre = svm.predict(test_x)
svm = pd.DataFrame(data=svm_pre, columns=['comment'])
svm['id'] = test_text.id
svm = svm[['id', 'comment']]
svm.to_csv('svm.csv', index=False)
a = {17, 1753, 722, 1342, 1753, 531, 1296, 74, 229, 299, 529,
     642, 715, 761, 1335, 1411, 1660, 1675, 1682, 1744, 1767, 1780, 1907,
     238, 1953, 229, 581, 697, 328, 886, 1350, 1369, 1413, 1985, 1342, 1753,
     501, 1963, 531, 1296, 613, 722, 1111, 1556, 662, 1844
     }
"""
# 后处理

r = test_text
r['id'] = svm_pre
r0 = r[r.id == 0]
key_word = ['蚊子', '老鼠', '苍蝇', '酸臭']

# key_word2=['蚊子','剩','不新鲜','没熟','老鼠','烂','骚味','苍蝇','虫','臭','想吐','太硬']
for i in key_word:
    print(r0[r0['comment'].str.contains(i)])

key_word2 = ['剩', '不新鲜', '没熟', '烂']
for i in key_word2:
    print(r0[r0['comment'].str.contains(i)])

key_word3 = ['骚味', '苍蝇', '虫', '臭', '想吐', '太硬']
for i in key_word3:
    print(r0[r0['comment'].str.contains(i)])



# a = list(a)
# a.sort()

for i in a:
    svm.loc[i, 'comment'] = [1]
svm.to_csv('后处理svm.csv', index=False)
print('结束3.')
