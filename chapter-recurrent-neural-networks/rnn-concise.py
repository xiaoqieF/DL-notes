from util import *
from d2l import torch as d2l

batch_size, num_steps = 32, 35
train_iter, vocab = d2l.load_data_time_machine(batch_size, num_steps)

if __name__ == '__main__':
    device = d2l.try_gpu()
    num_hiddens = 256
    num_epochs = 3000
    lr = 0.5
    rnn_layer = nn.RNN(len(vocab), num_hiddens)
    net = RNNModel(rnn_layer, len(vocab))
    net = net.to(device)
    train(net, train_iter, vocab, lr, num_epochs, device)