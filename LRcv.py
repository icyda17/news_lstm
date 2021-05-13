from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegressionCV
from sklearn.pipeline import make_pipeline
from sklearn import metrics

import argparse
from Utils.prepare_data import json_load, load_data
from Utils.utils import clean_underscore

parser = argparse.ArgumentParser()
parser.add_argument("--train_dir", help="processed training directory")
parser.add_argument("--dev_dir", help="processed development directory")
parser.add_argument("--test_dir", help="processed testing directory")
args = parser.parse_args()


train_path = args.train_dir
valid_path = args.dev_dir
test_path = args.test_dir

print('Load data...')
input_train, label_train = load_data(json_load(train_path))
input_valid, label_valid = load_data(json_load(valid_path))
input_test, label_test = load_data(json_load(test_path))

input_train = [clean_underscore(i) for i in input_train]
input_valid = [clean_underscore(i) for i in input_valid]
input_test = [clean_underscore(i) for i in input_test]

print('Build model...')
vec = TfidfVectorizer(min_df = 3, ngram_range=(1,1), max_df=0.6)
clf = LogisticRegressionCV()
pipe = make_pipeline(vec, clf)
pipe.fit(input_train, label_train)

print('Evaluating model...')
def print_report(pipe, y, x):
  y_actuals = y
  y_preds = pipe.predict(x)
  report = metrics.classification_report(y_actuals, y_preds)
  print(report)
  print("accuracy: {:0.3f}".format(metrics.accuracy_score(y_actuals, y_preds)))

print('Train:')
print_report(pipe, label_train, input_train)
print('-'**50)
print('Valid:')
print_report(pipe, label_valid, input_valid)
print('-'**50)
print('Test:')
print_report(pipe, label_test, input_test)
