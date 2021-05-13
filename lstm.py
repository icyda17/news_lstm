import argparse
from Utils.utils import *
from Utils.prepare_data import _save_npy, _load_npy
from datetime import datetime
from keras.callbacks import EarlyStopping
from network import building_mdl


parser = argparse.ArgumentParser()
parser.add_argument("--word_dir", help="word surface dict directory")
parser.add_argument("--vector_dir", help="word vector dict directory")
parser.add_argument("--train_dir", help="training directory")
parser.add_argument("--dev_dir", help="development directory")
parser.add_argument("--test_dir", help="testing directory")
parser.add_argument("--label_dir", help="label dict directory")
parser.add_argument("--max_length", help="max length of sequence")
parser.add_argument("--output_length", help="number of output classes")
parser.add_argument("--num_lstm_layer", help="number of lstm layer")
parser.add_argument("--num_hidden_node", help="number of hidden node")
parser.add_argument("--dropout", help="dropout number: between 0 and 1")
parser.add_argument("--batch_size", help="batch size for training")
parser.add_argument("--patience", help="patience")
args = parser.parse_args()

word_dir = args.word_dir
vector_dir = args.vector_dir
train_path = args.train_dir
valid_path = args.dev_dir
test_path = args.test_dir
label_dir = args.label_dir
max_length = int(args.max_length)
num_lstm_layer = int(args.num_lstm_layer)
output_length = int(args.output_length)
num_hidden_node = int(args.num_hidden_node)
dropout = float(args.dropout)
batch_size = int(args.batch_size)
patience = int(args.patience)
# patience : number of epochs with no improvement after which training will be stopped
startTime = datetime.now()


print("Load data...")
input_train, label_train, input_valid, label_valid, input_test, label_test =\
    read_data(train_path, valid_path,
              test_path)  # *arg: fast_load = True if processed file existed

X_train, y_train, X_valid, y_valid, X_test, y_test = \
    create_data(word_dir, vector_dir, label_dir, input_train, input_valid, input_test,
                label_train, label_valid, label_test, max_length)

'''
# Save for fast load

_save_npy(X_train, 'data/data2id/X_train.npy')
_save_npy(X_dev, 'data/data2id/X_valid.npy')
_save_npy(X_test, 'data/data2id/X_test.npy')
_save_npy(y_train, 'data/data2id/y_train.npy')
_save_npy(y_dev, 'data/data2id/y_valid.npy')
_save_npy(y_test, 'data/data2id/y_test.npy')

'''
print('Building model...')
cls_model = building_mdl(
    num_lstm_layer, num_hidden_node, dropout, output_length)

print('Model summary...')
print(cls_model.summary())
early_stopping = EarlyStopping(patience=patience)
history = cls_model.fit(X_train, y_train, batch_size=batch_size, epochs=100,
                        validation_data=(X_valid, y_valid), callbacks=[early_stopping], shuffle=True)

print('Saving model...')
cls_model.save('model')

print('Evaluating model...')
cls_model.evaluate(X_test, y_test, batch_size=256)
