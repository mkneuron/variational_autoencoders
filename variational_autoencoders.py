from chainer import functions as F
from chainer import links as L
from chainer import Chain, Variable
import numpy as np
from random import random


class RnnEncoder(Chain):
    def __init__(self, out, _in):
        super(RnnEncoder, self).__init__()

        # define layers
        with self.init_scope():
            self.l1 = L.LSTM(in_size=_in, out_size=out)
            self.l1.reset_state()

    def __call__(self, x):
        h = F.tanh(self.l1(x))

        return h


class LinearEncoder(Chain):
    def __init__(self, out, hidden, _in=None):
        super(LinearEncoder, self).__init__()

        # define layers
        with self.init_scope():
            self.l1 = L.Linear(in_size=_in, out_size=hidden)
            self.l2 = L.Linear(in_size=hidden, out_size=out)

    def __call__(self, x):
        h = F.tanh(self.l1(Variable(x)))
        h = F.tanh(self.l2(h))

        return h


class ConvolutionalEncoder(Chain):

    def __init__(self, out, hidden, _in=None, batch_size=1):
        super(ConvolutionalEncoder, self).__init__()

        # define layers
        with self.init_scope():
            self.l1 = L.Convolution2D(batch_size, out_channels=hidden, ksize=3,stride=1,pad=1)
            self.l2 = L.Linear(in_size=None, out_size=out)

    def __call__(self, x):
        x = Variable(np.expand_dims(x,axis=1))
        h = F.tanh(self.l1(x))
        h = self.l2(h)
        return h


class DeconvolutionalDecoder(Chain):

    def __init__(self, out, hidden, _in, batch_size=1):
        super(DeconvolutionalDecoder, self).__init__()

        assert (int(np.sqrt(hidden))**2) == hidden
        # define layers
        with self.init_scope():
            self.l1 = L.Linear(in_size=_in, out_size=hidden)
            self.l2 = L.Deconvolution2D(batch_size, out_channels=batch_size, ksize=3, stride=1, pad=1)

        self.batch_size = batch_size
        self.out = out
        self.hidden=hidden
        self._in = _in

    def __call__(self, x):

        h = F.softmax(self.l1(x))
        h = F.reshape(h, [self.batch_size, int(np.sqrt(self.hidden)),int(np.sqrt(self.hidden))])
        h = F.expand_dims(h, axis=0)
        h = F.tanh(self.l2(h))
        h = F.squeeze(h, axis=0)

        return h


class ConvDeconv(Chain):
    def __init__(self, out_e, out_d, hidden_e, hidden_d, _in_d=None, _in_e=None, batch_size=1):
        super(ConvDeconv, self).__init__()
        assert (int(np.sqrt(hidden_d)) ** 2) == hidden_d
        if not _in_d:
            _in_d=out_e

        # define layers
        with self.init_scope():
            self.l1 = L.Convolution2D(1, out_channels=batch_size, ksize=3,stride=1,pad=1)
            self.l2 = L.Linear(in_size=None, out_size=out_e)
            self.l3 = L.Linear(in_size=_in_d, out_size=hidden_d)
            self.l4 = L.Deconvolution2D(batch_size, out_channels=batch_size, ksize=3,stride=1,pad=1)

        self.batch_size = batch_size
        self.hidden = hidden_d
        self.hidden0 = hidden_e
        self._in_d = _in_d
        self._in_e = _in_e
        self.out_e = out_e
        self.out_d = out_d

    def __call__(self, x, return_latent=False, sampled_latent=None, ):
        x = Variable(np.expand_dims(x,axis=1))
        h = F.tanh(self.l1(x))
        h = F.softplus(self.l2(h))  # Softplus guarantees that log(output) is neither a NaN or infinity

        latent = h
        if sampled_latent:
            h = sampled_latent
        h = F.leaky_relu(self.l3(h))
        h = F.reshape(h, [self.batch_size, int(np.sqrt(self.hidden)),int(np.sqrt(self.hidden))])
        h = F.expand_dims(h, axis=0)
        h = F.tanh(self.l4(h))
        h = F.squeeze(h, axis=0)

        if return_latent:
            return h, latent
        else:
            return h, None

class AutoEncoder(Chain):
    """Variational autoencoder"""

    def __init__(self,latent_size, input_size, batch_size):
        super(AutoEncoder, self).__init__()
        with self.init_scope():
            "encoder layers"
            self.l1_e = L.Convolution2D(1, out_channels=batch_size, ksize=3,stride=1,pad=1)
            self.l2_means = L.Linear(in_size=None, out_size=latent_size)
            self.l3_latent = L.Linear(in_size=None, out_size=latent_size)
            self.l4_d = L.Linear(in_size=None, out_size=784)
            self.l5_d = L.Deconvolution2D(None, out_channels=1, ksize=3,stride=1,pad=1)

            self.latent_size = latent_size
            self.input_size = input_size
            self.batch_size = batch_size

    def __call__(self,x):
        "run as autoencoder"
        return self.decode(self.encode(x))

    def encode(self,x):
        x = F.expand_dims(x, axis=1)
        h = F.tanh(self.l1_e(x))
        means = F.softplus(self.l2_means(h))
        latent = F.softplus(self.l3_latent(h))  # Softplus guarantees that log(output) is neither a NaN or infinity
        return means, latent

    def decode(self,z):
        h = F.tanh(self.l4_d(z))
        h = F.reshape(h, [self.batch_size, 28,28])
        h = F.expand_dims(h, axis=1)
        h = F.tanh(self.l5_d(h))
        return h