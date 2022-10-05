from util import *
from d2l import torch as d2l

batch_size, num_steps = 32, 35
train_iter, vocab = d2l.load_data_time_machine(batch_size, num_steps)

if __name__ == '__main__':
    device = d2l.try_gpu()
    num_hiddens = 256
    num_inputs = len(vocab)
    num_epochs, lr = 2000, 1
    lstm_layer = nn.LSTM(num_inputs, num_hiddens)
    net = RNNModel(lstm_layer, len(vocab))
    net = net.to(device)
    train(net, train_iter, vocab, lr, num_epochs, device)
