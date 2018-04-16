# __author__ = 'max'
# modified by michihiro

import lasagne
import lasagne.nonlinearities as nonlinearities
from lasagne.layers import Gate
from lasagne_nlp.networks.crf import CRFLayer
from lasagne_nlp.networks.highway import HighwayDenseLayer

def build_BiLSTM(incoming, num_units, mask=None, grad_clipping=0, precompute_input=True, peepholes=False, dropout=True,
                 in_to_out=False):
    # construct the forward and backward rnns. Now, Ws are initialized by Glorot initializer with default arguments.
    # Need to try other initializers for specific tasks.

    # dropout for incoming
    if dropout:
        incoming = lasagne.layers.DropoutLayer(incoming, p=0.5)

    ingate_forward = Gate(W_in=lasagne.init.GlorotUniform(), W_hid=lasagne.init.GlorotUniform(),
                          W_cell=lasagne.init.Uniform(range=0.1))
    outgate_forward = Gate(W_in=lasagne.init.GlorotUniform(), W_hid=lasagne.init.GlorotUniform(),
                           W_cell=lasagne.init.Uniform(range=0.1))
    # according to Jozefowicz et al.(2015), init bias of forget gate to 1.
    forgetgate_forward = Gate(W_in=lasagne.init.GlorotUniform(), W_hid=lasagne.init.GlorotUniform(),
                              W_cell=lasagne.init.Uniform(range=0.1), b=lasagne.init.Constant(1.))
    # now use tanh for nonlinear function of cell, need to try pure linear cell
    cell_forward = Gate(W_in=lasagne.init.GlorotUniform(), W_hid=lasagne.init.GlorotUniform(), W_cell=None,
                        nonlinearity=nonlinearities.tanh)
    lstm_forward = lasagne.layers.LSTMLayer(incoming, num_units, mask_input=mask, grad_clipping=grad_clipping,
                                            nonlinearity=nonlinearities.tanh, peepholes=peepholes,
                                            precompute_input=precompute_input,
                                            ingate=ingate_forward, outgate=outgate_forward,
                                            forgetgate=forgetgate_forward, cell=cell_forward, name='forward')

    ingate_backward = Gate(W_in=lasagne.init.GlorotUniform(), W_hid=lasagne.init.GlorotUniform(),
                           W_cell=lasagne.init.Uniform(range=0.1))
    outgate_backward = Gate(W_in=lasagne.init.GlorotUniform(), W_hid=lasagne.init.GlorotUniform(),
                            W_cell=lasagne.init.Uniform(range=0.1))
    # according to Jozefowicz et al.(2015), init bias of forget gate to 1.
    forgetgate_backward = Gate(W_in=lasagne.init.GlorotUniform(), W_hid=lasagne.init.GlorotUniform(),
                               W_cell=lasagne.init.Uniform(range=0.1), b=lasagne.init.Constant(1.))
    # now use tanh for nonlinear function of cell, need to try pure linear cell
    cell_backward = Gate(W_in=lasagne.init.GlorotUniform(), W_hid=lasagne.init.GlorotUniform(), W_cell=None,
                         nonlinearity=nonlinearities.tanh)
    lstm_backward = lasagne.layers.LSTMLayer(incoming, num_units, mask_input=mask, grad_clipping=grad_clipping,
                                             nonlinearity=nonlinearities.tanh, peepholes=peepholes,
                                             precompute_input=precompute_input, backwards=True,
                                             ingate=ingate_backward, outgate=outgate_backward,
                                             forgetgate=forgetgate_backward, cell=cell_backward, name='backward')

    # concatenate the outputs of forward and backward RNNs to combine them.
    concat = lasagne.layers.concat([lstm_forward, lstm_backward], axis=2, name="bi-lstm")

    # dropout for output
    if dropout:
        concat = lasagne.layers.DropoutLayer(concat, p=0.5)

    if in_to_out:
        concat = lasagne.layers.concat([concat, incoming], axis=2)

    # the shape of BiRNN output (concat) is (batch_size, input_length, 2 * num_hidden_units)
    return concat

def build_BiLSTM_CNN(incoming1, incoming2, num_units, mask=None, grad_clipping=0, precompute_input=True,
                     peepholes=False, num_filters=20, dropout=True, in_to_out=False):
    # first get some necessary dimensions or parameters
    conv_window = 3
    _, sent_length, _ = incoming2.output_shape

    # dropout before cnn?
    if dropout:
        incoming1 = lasagne.layers.DropoutLayer(incoming1, p=0.5)

    # construct convolution layer
    cnn_layer = lasagne.layers.Conv1DLayer(incoming1, num_filters=num_filters, filter_size=conv_window, pad='full',
                                           nonlinearity=lasagne.nonlinearities.tanh, name='cnn')
    # infer the pool size for pooling (pool size should go through all time step of cnn)
    _, _, pool_size = cnn_layer.output_shape
    # construct max pool layer
    pool_layer = lasagne.layers.MaxPool1DLayer(cnn_layer, pool_size=pool_size)
    # reshape the layer to match lstm incoming layer [batch * sent_length, num_filters, 1] --> [batch, sent_length, num_filters]
    output_cnn_layer = lasagne.layers.reshape(pool_layer, (-1, sent_length, [1]))

    # finally, concatenate the two incoming layers together.
    incoming = lasagne.layers.concat([output_cnn_layer, incoming2], axis=2)

    return build_BiLSTM(incoming, num_units, mask=mask, grad_clipping=grad_clipping, peepholes=peepholes,
                        precompute_input=precompute_input, dropout=dropout, in_to_out=in_to_out)

def build_BiLSTM_CNN_CRF(incoming1, incoming2, num_units, num_labels, mask=None, grad_clipping=0, precompute_input=True,
                         peepholes=False, num_filters=20, dropout=True, in_to_out=False):
    bi_lstm_cnn = build_BiLSTM_CNN(incoming1, incoming2, num_units, mask=mask, grad_clipping=grad_clipping,
                                   precompute_input=precompute_input, peepholes=peepholes,
                                   num_filters=num_filters, dropout=dropout, in_to_out=in_to_out)

    return CRFLayer(bi_lstm_cnn, num_labels, mask_input=mask)

# char-bilstm and word-level bilstm
def build_BiLSTM_BiLSTM(incoming1, incoming2, num_units, mask=None, grad_clipping=0, precompute_input=True,
                     peepholes=False, num_filters=20, dropout=True, in_to_out=False):

    if dropout:
        incoming1 = lasagne.layers.DropoutLayer(incoming1, p=0.5)

    char_lstm_forward = lasagne.layers.LSTMLayer(incoming1, num_filters, grad_clipping=grad_clipping, nonlinearity=nonlinearities.tanh, peepholes=peepholes, precompute_input=precompute_input, backwards=False, only_return_final=True, name='char_forward')
    char_lstm_backward = lasagne.layers.LSTMLayer(incoming1, num_filters, grad_clipping=grad_clipping, nonlinearity=nonlinearities.tanh, peepholes=peepholes, precompute_input=precompute_input, backwards=True, only_return_final=True, name='char_backward') # (n_batch, num_filters)

    char_out = lasagne.layers.concat([char_lstm_forward, char_lstm_backward], axis=1, name="bi-lstm_char") # (n_batch, 2*num_filters)
    char_out = lasagne.layers.DenseLayer(char_out, num_units=num_filters, nonlinearity=nonlinearities.tanh) # (n_batch, num_filters)

    _, sent_length, _ = incoming2.output_shape
    # reshape the layer to match lstm incoming layer [batch * sent_length, 2*num_filters] --> [batch, sent_length, 2*num_filters]
    char_out = lasagne.layers.reshape(char_out, (-1, sent_length, [1]))

    # finally, concatenate the two incoming layers together.
    incoming = lasagne.layers.concat([char_out, incoming2], axis=2)

    return build_BiLSTM(incoming, num_units, mask=mask, grad_clipping=grad_clipping, peepholes=peepholes,
                        precompute_input=precompute_input, dropout=dropout, in_to_out=in_to_out)

def build_BiLSTM_BiLSTM_CRF(incoming1, incoming2, num_units, num_labels, mask=None, grad_clipping=0, precompute_input=True,
                         peepholes=False, num_filters=20, dropout=True, in_to_out=False):
    bilstm_bilstm = build_BiLSTM_BiLSTM(incoming1, incoming2, num_units, mask=mask, grad_clipping=grad_clipping,
                                   precompute_input=precompute_input, peepholes=peepholes,
                                   num_filters=num_filters, dropout=dropout, in_to_out=in_to_out)

    return CRFLayer(bilstm_bilstm, num_labels, mask_input=mask)
