from d2l import torch as d2l
from util import *

batch_size, num_steps = 32, 35
train_iter, vocab = d2l.load_data_time_machine(batch_size, num_steps)

if __name__ == '__main__':
    vocab_size, num_hiddens, device = len(vocab), 256, d2l.try_gpu()
    num_inputs = vocab_size
    num_epochs, lr = 3000, 0.5
    gru_layer = nn.GRU(num_inputs, num_hiddens)
    net = RNNModel(gru_layer, vocab_size)
    net = net.to(device)
    train(net, train_iter, vocab, lr, num_epochs, device)
