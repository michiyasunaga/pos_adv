import numpy as np
import theano
from theano import gradient
from theano import tensor as T
from lasagne.layers import Layer
from lasagne import init

trng = theano.sandbox.rng_mrg.MRG_RandomStreams(123)

def generate_adv_example(embedded, loss, perturb_scale):
    # embedded: [n_examples, input_length, feature_dim]

    grad = gradient.grad(loss, embedded)
    grad = gradient.disconnected_grad(grad)

    shifted = embedded + T.max(T.abs_(embedded))+1.0
    grad_dim = (shifted/shifted).sum(axis=(1,2)).mean(axis=0) # grad dim for each example
    sqrt_grad_dim = T.sqrt(grad_dim) # sqrt(input_length * emb_dim)
    perturb = perturb_scale * sqrt_grad_dim * _scale_unit_l2(grad)

    return embedded + perturb


def adversarial_loss(ori_char_emb, ori_word_emb, loss_fn, loss=None, perturb_scale=0.02):
    print '** perturb_scale =', perturb_scale, '**'

    assert loss is not None
    char_emb_adv = generate_adv_example(ori_char_emb, loss, perturb_scale)
    word_emb_adv = generate_adv_example(ori_word_emb, loss, perturb_scale)

    return loss_fn(char_emb_adv, word_emb_adv, return_all=False)


def _scale_unit_l2(x):
    # shape(x) = (batch, num_timesteps, d)

    alpha = T.max(T.abs_(x), axis=(1,2), keepdims=True) + 1e-12
    l2_norm = alpha * T.sqrt(T.sum( (x/alpha)**2, axis=(1,2),
                                          keepdims=True) + 1e-6)
    x_unit = x / l2_norm
    return x_unit


# This requires adding dev & test vocab
class Normalized_EmbeddingLayer(Layer):
    def __init__(self, incoming, input_size, output_size, vocab_freqs,
                 W=init.Normal(), **kwargs):
        super(Normalized_EmbeddingLayer, self).__init__(incoming, **kwargs)

        self.input_size = input_size
        self.output_size = output_size

        self.vocab_freqs = T.as_tensor_variable(np.asarray(vocab_freqs, dtype=theano.config.floatX).reshape(-1, 1)) #shape: (vocab_size, 1)

        self.W = self.add_param(W, (input_size, output_size), name="W")
        self.W = self._normalize(self.W)

    def _normalize(self, emb):
        # emb: (vocab_size, emb_dim)

        weights = self.vocab_freqs / (self.vocab_freqs).sum()
        mean = (weights * emb).sum(axis=0, keepdims=True) # (1, emb_dim)
        var = (weights * (emb - mean)**2.).sum(axis=0, keepdims=True) # (1, emb_dim)
        stddev = T.sqrt(1e-6 + var)
        return (emb - mean) / stddev

    def get_output_shape_for(self, input_shape):
        return input_shape + (self.output_size, )

    def get_output_for(self, input, **kwargs):
        return self.W[input]
