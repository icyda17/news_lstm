import numpy as np
from Utils.prepare_data import load_data
import codecs
import json
import re
import string
import pickle
from tqdm import tqdm
from prepare_data import json_load, dup_dict, load_data, convert_json, json_save
from vncorenlp import VnCoreNLP

annotator = VnCoreNLP("/content/vncorenlp/VnCoreNLP-1.1.1.jar",
                      annotators="wseg,pos", max_heap_size='-Xmx2g')

EMAIL_PATTERN = re.compile(r'\w+@[^\.].*\.[a-z]{2,}\b')
LINK_PATTERN = re.compile(r'(http(\S+)?)|(www\.)\w+(\.\w+)|(\w+.\.(com|vn|net|org|edu|gov|mil|aero|asia|\
                            biz|cat|coop|info|int|jobs|mobi|museum|name|post|pro|tel|travel|xxx|ac|ad|ae|\
                            af|ag|ai|al|am|an|ao|aq|ar|as|at|au|aw|ax|az|ba|bb|bd|be|bf|bg|bh|bi|bj|bm|bn|\
                            bo|br|bs|bt|bv|bw|by|bz|ca|cc|cd|cf|cg|ch|ci|ck|cl|cm|cn|co|cr|cs|cu|cv|cx|cy|\
                            cz|dd|de|dj|dk|dm|do|dz|ec|ee|eg|eh|er|es|et|eu|fi|fj|fk|fm|fo|fr|ga|gb|gd|ge|\
                            gf|gg|gh|gi|gl|gm|gn|gp|gq|gr|gs|gt|gu|gw|gy|hk|hm|hn|hr|ht|hu|id|ie|il|im|in|\
                            io|iq|ir|is|it|je|jm|jo|jp|ke|kg|kh|ki|km|kn|kp|kr|kw|ky|kz|la|lb|lc|li|lk|lr|\
                            ls|lt|lu|lv|ly|ma|mc|md|me|mg|mh|mk|ml|mm|mn|mo|mp|mq|mr|ms|mt|mu|mv|mw|mx|my|\
                            mz|na|nc|ne|nf|ng|ni|nl|no|np|nr|nu|nz|om|pa|pe|pf|pg|ph|pk|pl|pm|pn|pr|ps|pt|\
                            pw|py|qa|re|ro|rs|ru|rw|sa|sb|sc|sd|se|sg|sh|si|sj|Ja|sk|sl|sm|sn|so|sr|ss|st|\
                            su|sv|sx|sy|sz|tc|td|tf|tg|th|tj|tk|tl|tm|tn|to|tp|tr|tt|tv|tw|tz|ua|ug|uk|us|\
                            uy|uz|va|vc|ve|vg|vi|vn|vu|wf|ws|ye|yt|yu|za|zm|zw))\b')  # https://gist.github.com/gruber/8891611
LINEBREAK_PATTERN = re.compile(r'\t')
SPACE_PATTERN = re.compile(r' +')
TAG_PATTERN = re.compile(r'<.*?>+')
MPERIOD_PATTERN = re.compile(r'\.+')

with open(r'../data/vietnamese-stopwords-dash.txt', encoding="utf8") as f:
    stopwords = f.readlines()
stopword = [re.sub(r'\n', '', item) for item in stopwords]


def remove_stopwords(text: str, stopword) -> str:
    temp_list = []
    for word in str(text).split():
        if word not in stopword:
            temp_list.append(word)
    return ' '.join(temp_list)


def tok_pos(text: str):
    '''
    segment and post tag. filter words that are N, V
    '''
    out = ''
    for i in annotator.annotate(text)['sentences']:
        for j in i:
            if j['posTag'] in ['N', 'V', 'Np', 'Nb', 'A', 'Nc', 'Np', 'Nu', 'Ny', 'M']:
                out += j['form'] + ' '
    return out[:-1]


def clean(text: str):
    text = EMAIL_PATTERN.sub(' ', text)
    text = LINK_PATTERN.sub(' ', text)
    text = LINEBREAK_PATTERN.sub(' ', text)
    text = TAG_PATTERN.sub(' ', text)
    text = MPERIOD_PATTERN.sub('.', text)
    text = SPACE_PATTERN.sub(' ', text)
    text = tok_pos(text)
    text = remove_stopwords(text, stopword)
    # text = text.lower().strip()
    return text

def clean_underscore(text:str):
  text = re.sub(r'_',' ',text).lower()
  text = re.sub(r' +',' ', text)
  return text

def read_json(path):
    data = json_load(path)
    # print("# item in : %s" %len(data))
    data = dup_dict(data)
    input, label = load_data(data)
    return input, label


def read_data(train_path, valid_path, test_path, fast_load=False):
    if fast_load:
        input_train, label_train = load_data(
            json_load('../data/train_cln.json'))
        input_valid, label_valid = load_data(
            json_load('../data/valid_cln.json'))
        input_test, label_test = load_data(json_load('../data/test_cln.json'))
    else:
        input_train, label_train = read_json(train_path)
        input_valid, label_valid = read_json(valid_path)
        input_test, label_test = read_json(test_path)
        input_train = [clean(i) for i in tqdm(input_train)]
        # train = convert_json(input_train, label_train) # save data and fast load
        # json_save(train, '../data/train_cln.json')
        input_valid = [clean(i) for i in tqdm(input_valid)]
        # valid = convert_json(input_valid, label_valid)
        # json_save(valid, '../data/valid_cln.json')
        input_test = [clean(i) for i in tqdm(input_test)]
        # test = convert_json(input_test, label_test)
        # json_save(test, '../data/test_cln.json')
    print("# item in train: %s" % len(input_train))
    print("# item in valid: %s" % len(input_valid))
    print("# item in test: %s" % len(input_test))
    return input_train, label_train, input_valid, label_valid, input_test, label_test


def construct_tensor_word(word_sentences, unknown_embedd, embedd_words, embedd_vectors,
                          embedd_dim, max_length):
    # shape: (#sentence, max_length of a sentence, dim)
    X = np.empty([len(word_sentences), max_length, embedd_dim])
    for i in range(len(word_sentences)):
        words = word_sentences[i].split()  # a sentence
        if len(words) > max_length:
            length = max_length
        else:
            length = len(words)
        for j in range(length):
            word = words[j].lower()
            try:
                embedd = embedd_vectors[embedd_words.index(word)]
            except:
                embedd = unknown_embedd
            X[i, j, :] = embedd
        # Zero out X after the end of the sequence
        X[i, length:] = np.zeros([1, embedd_dim])
    return X


def vector_tag(tag_list, label_dir, save=False):

    if save:
        assert label_dir, "Missing directory for label"
        tag_dict = {}
        val = 0
        for i in tag_list:
            if i not in tag_dict:
                tag_dict[i] = val
                val += 1
        json_save(tag_dict, label_dir)
    else:
        tag_dict = json_load(label_dir)

    out = [tag_dict[i] for i in tag_list]
    return np.array(out)


def create_vector_data(word_list, tag_list, unknown_embedd, embedd_words, embedd_vectors, embedd_dim,
                       max_length, save_train, label_dir=None):
    word_data = construct_tensor_word(word_list, unknown_embedd, embedd_words, embedd_vectors, embedd_dim,
                                      max_length)
    tag_data = vector_tag(tag_list, save=save_train, label_dir=label_dir)
    return word_data, tag_data


def create_data(word_dir, vector_dir, label_dir, word_list_train, word_list_dev, word_list_test,
                tag_list_train, tag_list_dev, tag_list_test, max_length):
    # load pre-trained vector. shape (#words, dim)
    embedd_vectors = np.load(vector_dir)
    with open(word_dir, 'rb') as handle:  # list words. len(#words)
        embedd_words = pickle.load(handle)
    embedd_dim = np.shape(embedd_vectors)[1]
    unknown_embedd = np.random.uniform(-0.01, 0.01, [1, embedd_dim])
    word_train, tag_train = create_vector_data(word_list_train, tag_list_train, unknown_embedd, embedd_words,
                                               embedd_vectors, embedd_dim, max_length, label_dir=label_dir, save_train=True)
    word_dev, tag_dev = create_vector_data(word_list_dev, tag_list_dev, unknown_embedd, embedd_words,
                                           embedd_vectors, embedd_dim, max_length, label_dir=label_dir, save_train=False)
    word_test, tag_test = create_vector_data(word_list_test, tag_list_test, unknown_embedd, embedd_words,
                                             embedd_vectors, embedd_dim, max_length, label_dir=label_dir, save_train=False)
    return np.array(word_train), np.array(tag_train), np.array(word_dev), np.array(tag_dev),\
         np.array(word_test), np.array(tag_test)
