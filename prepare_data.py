import codecs
import glob
from tqdm import tqdm
import json


def iterate_lines(folder_name):
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
    Extract info in data and save in dict
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


def json_save(data, path):
    with codecs.open(path, 'w', 'utf=8') as fout:
        json.dump(data, fout, ensure_ascii=False)


def json_load(path):
    with codecs.open(path, "r", encoding="utf8") as read_file:
        data = json.load(read_file)
    return data

def split_train(data:list, valid_ratio:float, test_ratio:float):
    test_idx = int(len(data)*test_ratio)
    valid_idx = test_idx + int(len(data)*valid_ratio)
    test = data[:test_idx]
    valid = data[test_idx:valid_idx]
    train = data[valid_idx:]
    return train, valid, test
if __name__ == "__main__":
    FOLDER_PATH = 'data/data'
    data = iterate_lines(FOLDER_PATH)
    data = extract_info(data)
    json_save(data, 'data/data.json')
    # data = json_load(r'data.json')
    train, valid, test = split_train(data, 0.2, 0.1)
    json_save(train, 'data/train.json')
    json_save(valid, 'data/valid.json')
    json_save(test, 'data/test.json')