from tensorflow.keras import layers

def BiLSTM_Module(X_input):
    x = layers.Bidirectional(layers.LSTM(64))(X_input)
    return x