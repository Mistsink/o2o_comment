import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold

from task.task3.util.matrix_train import matrix_train
from task.task3.util.cut import cut_text
from task.task3.util.pretreat import Filter

# const key
KEY_COMMENT = 'comment'

if __name__ == '__main__':
    train_text = pd.read_csv('../../data/train.csv', sep='\t')
    test_text = pd.read_csv('../../data/test_new.csv', sep=',')

    train_text[KEY_COMMENT] = train_text[KEY_COMMENT].apply(lambda text: Filter(text))
    test_text[KEY_COMMENT] = test_text[KEY_COMMENT].apply(lambda text: Filter(text))

    train_comments_cut = [cut_text(sentence) for sentence in train_text[KEY_COMMENT].values]
    test_comments_cut = [cut_text(sentence) for sentence in test_text[KEY_COMMENT].values]

    tfidf = TfidfVectorizer(min_df=20, ngram_range=[1, 1], token_pattern=r'\b\w+\b')

    X_train, Y_train, X_test = matrix_train(train_text['label'], train_comments_cut, test_comments_cut, tfidf)

    # 使用逻辑回归对文本进行分类，使用5折交叉验证
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=2019).split(X_train, Y_train)

    y_test_preds = np.zeros((len(X_test), 2))
    for i, (train_idx, valid_idx) in enumerate(skf):
        X_train_, y_train_ = X_train[train_idx], Y_train[train_idx]
        X_valid_, y_valid_ = X_train[valid_idx], Y_train[valid_idx]

        lr = LogisticRegression(C=1.2)
        lr.fit(X_train_, y_train_)

        y_valid_preds = lr.predict(X_valid_)
        y_test_preds += lr.predict_proba(X_test) / 5

        acc = accuracy_score(y_valid_, y_valid_preds)
        f1 = f1_score(y_valid_, y_valid_preds, average='macro')
        print("第{}折：accuracy->{:.5f}, f1_score->{:.5f}".format(i + 1, acc, f1))

    y_test = [np.argmax(r) for r in y_test_preds]
    sub = test_text.copy()
    sub['label'] = y_test
    sub[['id', 'label']].to_csv('./result/tfidf_vec.csv', index=None)

    # 获取所有单词的tf-idf值
    tfidf_matrix = tfidf.transform(train_comments_cut + test_comments_cut)
    words = tfidf.get_feature_names()
    tfidf_score_dict = {}
    for doc in range(len(train_comments_cut + test_comments_cut)):
        word_idx = tfidf_matrix[doc, :].nonzero()[1]
        for x in word_idx:
            if tfidf_score_dict.get(x, -1) == -1:
                tfidf_score_dict[x] = tfidf_matrix[doc, x]

    print("\n展示字典中前10个单词的tf-idf值：")
    for word, tfidf_score in [(words[idx], s) for (idx, s) in tfidf_score_dict.items()][:10]:
        print("{:10s}-->{:.10f}".format(word, tfidf_score))
