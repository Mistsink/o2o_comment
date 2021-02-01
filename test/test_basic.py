import pandas as pd
import re

def read_csv(file_path):
    df = pd.read_csv(file_path, sep='\n')
    print(type(df))
    print(df)
    return df

def write_csv(data_frame, file_path):
    df = pd.DataFrame(data_frame)
    print(df)
    df.to_csv(file_path, index=False)

def test_re():
    line = 'Cats are smarter than dogs'

    matchObj = re.match(r'(.*) are (.*)', line)

    print('line:', line)

    if matchObj:
        print('matchObj.group():', matchObj.group())
        print('matchObj.group(1):', matchObj.group(1))
        print('matchObj.group(2):', matchObj.group(2))
    else:
        print('No matched!')

    line = 'Cats are smarter than dogs'

    searchObj = re.search(r'are .*', line)
    print('line:', line)
    if matchObj:
        print('searchObj: ', searchObj.group())
        # print('searchObj.group():', searchObj.group())
        # print('searchObj.group(1):', searchObj.group(1))
    else:
        print('No searched!')

    phone = '2004-9595-559 # 这是一个国外的电话号码'
    num = re.sub(r'#(.*)', "", phone)
    print('#(.*)电话号码：', num)
    num = re.sub(r'[^0-9]', "", phone)
    print('[^0-9]电话号码：', num)

    pattern = re.compile(r'\d+')
    m = pattern.match('one12twothree34four')
    print(m)
    m = pattern.search('one12twothree34four', 2, 10)
    print(m)
    m = pattern.match('one12twothree34four', 3, 10)
    print(m)

    print(m.group(0))
    print(m.start(0))
    print(m.end(0))
    print(m.span(0))


if __name__ == '__main__':
    test_re()