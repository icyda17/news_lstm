# split data to train, test, split
!python Utils/prepare_data.py
# Train LSTM
!python lstm.py --word_dir data/emb/words.pl --vector_dir data/emb/vectors.npy --train_dir data/train.json --dev_dir data/valid.json --test_dir data/test.json --label_dir data/emb/tags.pkl --max_length 200 --output_length 13 --num_lstm_layer 2 --num_hidden_node 128 --dropout 0.2 --batch_size 512 --patience 5
# Train LogisticRegressionCV
!python LRcv.py --train_dir data/train_cln.json --dev_dir data/valid_cln.json --test_dir data/test_cln.json