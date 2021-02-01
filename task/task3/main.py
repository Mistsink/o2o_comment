
import pandas as pd

from task.task3.util.pretreat import Filter

train_text = pd.read_csv('../../data/train.csv', sep='\t')
test_text = pd.read_csv('../../data/test_new.csv', sep=',')


train_text['comment'] = train_text['comment'].apply(lambda x: Filter(x))
test_text['comment'] = test_text['comment'].apply(lambda x: Filter(x))
print(train_text)