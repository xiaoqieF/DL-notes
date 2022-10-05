import math
import torch
from torch import nn
from torch.nn import functional as F
from d2l import torch as d2l

batch_size, num_steps = 32, 35

train_iter, vocab = d2l.load_data_time_machine(batch_size, num_steps)


class RNNModel:
    def __init__(self, vocab_size, num_hiddens, device):
        self.vocab_size, self.num_hiddens = vocab_size, num_hiddens
        self.params = self.init_params(device)

    def __call__(self, inputs, state):
        # inputs.shape: [batch_size, num_steps]
        inputs = F.one_hot(inputs.T, self.vocab_size).type(torch.float32)
        # inputs.shape: [num_steps, batch_size, len(vocab)]
        W_xh, W_hh, b_h, W_hq, b_q = self.params
        H = state
        outputs = []
        # 对每个时间步进行遍历计算
        for X in inputs:
            H = torch.tanh(torch.mm(X, W_xh) + torch.mm(H, W_hh) + b_h)
            Y = torch.mm(H, W_hq) + b_q
            outputs.append(Y)
        # 输出 shape: [num_steps * batch_size, len(vocab)]
        return torch.cat(outputs, dim=0), H

    def begin_state(self, batch_size, device):
        return torch.zeros((batch_size, self.num_hiddens), device=device)

    def init_params(self, device):
        num_inputs = num_outputs = self.vocab_size

        def normal(shape):
            return torch.randn(size=shape, device=device) * 0.01

        W_xh = normal((num_inputs, self.num_hiddens))
        W_hh = normal((self.num_hiddens, self.num_hiddens))
        b_h = torch.zeros(self.num_hiddens, device=device)
        W_hq = normal((self.num_hiddens, num_outputs))
        b_q = torch.zeros(num_outputs, device=device)
        params = [W_xh, W_hh, b_h, W_hq, b_q]
        for param in params:
            param.requires_grad_(True)
        return params

    def grad_clipping(self, theta):
        norm = torch.sqrt(sum(torch.sum((p.grad ** 2)) for p in self.params))
        if norm > theta:
            for param in self.params:
                param.grad[:] *= theta / norm


def predict(prefix, num_preds, net, vocab, device):
    """在prefix后面生成新字符"""
    state = net.begin_state(batch_size=1, device=device)
    outputs = [vocab[prefix[0]]]
    get_input = lambda: torch.tensor([outputs[-1]], device=device).reshape((1, 1))
    for y in prefix[1:]:
        _, state = net(get_input(), state)
        outputs.append(vocab[y])
    for _ in range(num_preds):
        y, state = net(get_input(), state)
        outputs.append(int(y.argmax(dim=1).reshape(1)))
    return ''.join([vocab.idx_to_token[i] for i in outputs])


def train_epoch(net, train_iter, loss, updater, device, use_random_iter):
    state = None
    total_loss = 0.0
    for X, Y in train_iter:
        if state is None or use_random_iter:
            state = net.begin_state(X.shape[0], device)
        else:
            state.detach_()
        y = Y.T.reshape(-1)
        X, y = X.to(device), y.to(device)
        y_hat, state = net(X, state)
        # 其中 y_hat.shape:[num_steps * batch_size, len(vocab)], y.shape:[num_steps * batch_size]
        l = loss(y_hat, y.long()).mean()
        updater.zero_grad()
        l.backward()
        net.grad_clipping(1)
        updater.step()
        total_loss += l
    return math.exp(total_loss)


def train(net, train_iter, vocab, lr, num_epochs, device, use_random_iter=False):
    loss = nn.CrossEntropyLoss()
    updater = torch.optim.SGD(net.params, lr)
    p = lambda prefix: predict(prefix, 50, net, vocab, device)
    for epoch in range(num_epochs):
        ppl = train_epoch(net, train_iter, loss, updater, device, use_random_iter)
        if (epoch + 1) % 10 == 0:
            print(f'epoch:{epoch}, ppl:{ppl}, predict:{p("time traveller")}')


if __name__ == '__main__':
    num_epochs, lr = 2000, 0.5
    net = RNNModel(len(vocab), 512, d2l.try_gpu())
    train(net, train_iter, vocab, lr, num_epochs, d2l.try_gpu())
