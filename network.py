from keras.layers import LSTM, Dense, Flatten
from keras.models import Sequential


def building_mdl(num_lstm_layer, num_hidden_node, dropout, output_length, max_length, dim):
    model = Sequential()
    model.add(LSTM(units=num_hidden_node, return_sequences=True, dropout=dropout,
                                 recurrent_dropout=dropout, input_shape = (max_length,dim)))
    for i in range(1, num_lstm_layer):
        model.add(LSTM(units=num_hidden_node, return_sequences=True, dropout=dropout,
                                 recurrent_dropout=dropout))
    model.add(Flatten())
    model.add(Dense(output_length, activation='softmax'))
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model