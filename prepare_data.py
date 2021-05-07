import codecs
import glob
from tqdm import tqdm
import json
import re
import string
import unicodedata
from sklearn.model_selection import train_test_split


def iterate_files(folder_name):
    '''
    Read files in folder and save to list
    '''
    data = []
    files = glob.glob(folder_name + '/*.txt', recursive=True)
    for file_name_in in tqdm(files):
        # print('Read file %s' % file_name_in)
        with codecs.open(file_name_in, 'r', 'utf-8') as f:
            data.append(f.readlines())
    return data


def extract_info(data: list):
    '''
    Extract info in data and save in list of dict
    '''
    d = []
    for ele in tqdm(data):
        a = {}
        a['tag'] = ele[0]
        a['title'] = ele[1]
        a['snippet'] = ele[2]
        a['body'] = ' '.join(ele[3:])
        d.append(a)
    return d


def clean(text: str, punc=True):
    '''
    normalize
    remove '\n' in text and add '.' if text does not end with punctuation
    '''
    text = unicodedata.normalize('NFKC', text)
    clean_txt = re.sub(r'\n', '', text).strip()
    rm_space = re.sub(' +', ' ', clean_txt)
    if punc:
        if rm_space[-1] not in string.punctuation:
            rm_space += '.'
    return rm_space


def prepare_data(list_data: list):
    '''
    Extract, clean, concat title, snippet, body
    '''
    body, snippet, label, title = [], [], [], []
    for ele in list_data:
        body.append(ele.get('body'))
        snippet.append(ele.get('snippet'))
        label.append(ele.get('tag'))
        title.append(ele.get('title'))
    bodys = [clean(i) for i in body]
    snippets = [clean(i) for i in snippet]
    labels = [clean(i, punc=False) for i in label]
    titles = [clean(i) for i in title]
    inputs = [title+' '+snippet+' '+body for title,
              snippet, body in zip(titles, snippets, bodys)]
    return inputs, labels


def json_save(data, path):
    with codecs.open(path, 'w', 'utf=8') as fout:
        json.dump(data, fout, ensure_ascii=False)


def json_load(path):
    with codecs.open(path, "r", encoding="utf8") as read_file:
        data = json.load(read_file)
    return data


def split_ig_ratio(data: list, valid_ratio: float, test_ratio: float):
    '''
    Split data by index ignore label ratio
    '''
    test_idx = int(len(data)*test_ratio)
    valid_idx = test_idx + int(len(data)*valid_ratio)
    test = data[:test_idx]
    valid = data[test_idx:valid_idx]
    train = data[valid_idx:]
    return train, valid, test


def split_ratio(X, y, valid_ratio: float, test_ratio: float):
    X_train, X_rem, y_train, y_rem = train_test_split(
        X, y, test_size=valid_ratio+test_ratio, random_state=42, stratify=y)
    X_valid, X_test, y_valid, y_test = train_test_split(
        X_rem, y_rem, test_size=test_ratio/(valid_ratio+test_ratio), random_state=42, stratify=y_rem)
    return X_train, X_valid, X_test, y_train, y_valid, y_test

def convert_json(x, y):
    out = []
    for i, o in zip(x, y):
        a = {}
        a['content'] = i
        a['label'] = o
        out.append(a)
    return out
if __name__ == "__main__":
    # FOLDER_PATH = 'data/data'
    # data = iterate_files(FOLDER_PATH)
    # '''
    # Save file as json format and ignore the label ratio
    # '''
    # data = extract_info(data)
    # json_save(data, 'data/data.json')
    # train, valid, test = split_ig_ratio(data, 0.2, 0.1)

    '''
    Save file as json format and stratify the label ratio
    '''
    data = json_load(r'data\data.json')
    X, y = prepare_data(data)
    X_train, X_valid, X_test, y_train, y_valid, y_test = split_ratio(X, y, valid_ratio=0.2, test_ratio=0.1)
    train, valid, test = convert_json(X_train, y_train), convert_json(X_valid, y_valid), convert_json(X_test, y_test)
    json_save(train, 'data/train.json')
    json_save(valid, 'data/valid.json')
    json_save(test, 'data/test.json')

