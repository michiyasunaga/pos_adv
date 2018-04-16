# adapted from https://github.com/XuezheMax/LasagneNLP
# modified by michihiro

import time
import sys, os
import subprocess
import argparse
#from collections import OrderedDict
from lasagne_nlp.utils import utils
import lasagne_nlp.utils.data_processor as data_processor
from lasagne_nlp.utils.objectives import crf_loss, crf_accuracy
import lasagne
from lasagne import layers as Lyrs
import theano
import theano.tensor as T
from lasagne_nlp.networks.networks_bilstm_char_dense import build_BiLSTM_BiLSTM_CRF
from lasagne_nlp.utils.adversarial import adversarial_loss, Normalized_EmbeddingLayer
from lasagne.layers import Layer

import numpy as np

def main():
    parser = argparse.ArgumentParser(description='Tuning with bi-directional LSTM-CNN-CRF')
    parser.add_argument('--fine_tune', action='store_true', help='Fine tune the word embeddings')
    parser.add_argument('--embedding', choices=['word2vec', 'glove', 'senna', 'random', 'polyglot'], help='Embedding for words',
                        required=True)
    parser.add_argument('--embedding_dict', default=None, help='path for embedding dict')
    parser.add_argument('--batch_size', type=int, default=10, help='Number of sentences in each batch')
    parser.add_argument('--num_units', type=int, default=100, help='Number of hidden units in LSTM')
    parser.add_argument('--num_filters', type=int, default=20, help='Number of filters in CNN')
    parser.add_argument('--learning_rate', type=float, default=0.1, help='Learning rate')
    parser.add_argument('--decay_rate', type=float, default=0.1, help='Decay rate of learning rate')
    parser.add_argument('--grad_clipping', type=float, default=0, help='Gradient clipping')
    parser.add_argument('--gamma', type=float, default=1e-6, help='weight for regularization')
    parser.add_argument('--peepholes', action='store_true', help='Peepholes for LSTM')
    parser.add_argument('--oov', choices=['random', 'embedding'], help='Embedding for oov word', required=True)
    parser.add_argument('--update', choices=['sgd', 'momentum', 'nesterov', 'adadelta', 'adam'], help='update algorithm',
                        default='sgd')
    parser.add_argument('--regular', choices=['none', 'l2'], help='regularization for training', required=True)
    parser.add_argument('--dropout', action='store_true', help='Apply dropout layers')
    parser.add_argument('--patience', type=int, default=5, help='Patience for early stopping')
    parser.add_argument('--output_prediction', action='store_true', help='Output predictions to temp files')
    parser.add_argument('--train')
    parser.add_argument('--dev')
    parser.add_argument('--test')
    parser.add_argument('--exp_dir')
    parser.add_argument('--adv', type=float, default=0)
    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument('--reload', default=None, help='path for reloading')

    args = parser.parse_args()
    np.random.seed(args.seed)
    lasagne.random.set_rng(np.random)

    def construct_input_layer():
        if fine_tune:
            layer_input = lasagne.layers.InputLayer(shape=(None, max_length), input_var=input_var, name='input')
            layer_embedding = Normalized_EmbeddingLayer(layer_input, input_size=alphabet_size,
                    output_size=embedd_dim, vocab_freqs=word_freqs,
                    W=embedd_table, name='embedding')
            raw_layer = layer_embedding
        else:
            layer_input = lasagne.layers.InputLayer(shape=(None, max_length, embedd_dim), input_var=input_var, name='input')
            raw_layer = layer_input

        return raw_layer # [batch, max_sent_length, embedd_dim]

    def construct_char_input_layer():
        layer_char_input = lasagne.layers.InputLayer(shape=(None, max_sent_length, max_char_length),
                input_var=char_input_var, name='char-input')
        layer_char_input = lasagne.layers.reshape(layer_char_input, (-1, [2])) # [batch * max_sent_length, max_char_length]
        layer_char_embedding = Normalized_EmbeddingLayer(layer_char_input, input_size=char_alphabet_size,
                output_size=char_embedd_dim, vocab_freqs=char_freqs, W=char_embedd_table,
                name='char_embedding') # [n_examples, max_char_length, char_embedd_dim]

        #layer_char_input = lasagne.layers.DimshuffleLayer(layer_char_embedding, pattern=(0, 2, 1)) # [n_examples, char_embedd_dim, max_char_length]
        return layer_char_embedding

    logger = utils.get_logger("BiLSTM-BiLSTM-CRF")
    fine_tune = args.fine_tune
    oov = args.oov
    regular = args.regular
    embedding = args.embedding
    embedding_path = args.embedding_dict
    train_path = args.train
    dev_path = args.dev
    test_path = args.test
    update_algo = args.update
    grad_clipping = args.grad_clipping
    peepholes = args.peepholes
    gamma = args.gamma
    output_predict = args.output_prediction
    dropout = args.dropout

    exp_dir = args.exp_dir
    if not os.path.isdir(exp_dir): os.mkdir(exp_dir)
    exp_name = exp_dir.split('/')[-1]
    exp_mode = exp_name.split('_')[0] # 'pos' or 'ner', etc.

    save_dir = os.path.join(exp_dir, 'save')
    eval_dir = os.path.join(exp_dir, 'eval')
    if not os.path.isdir(save_dir): os.mkdir(save_dir)
    if not os.path.isdir(eval_dir): os.mkdir(eval_dir)
    eval_script = "./conlleval"

    if exp_mode=='pos':
        (word_col_in_data, label_col_in_data) = (0,1)
    elif exp_mode=='ner':
        (word_col_in_data, label_col_in_data) = (0,3)
    elif exp_mode=='chunk':
        (word_col_in_data, label_col_in_data) = (0,2)
    else:
        (word_col_in_data, label_col_in_data) = (1,3) # assume CoNLL-U style

    # load data
    X_train, Y_train, mask_train, X_dev, Y_dev, mask_dev, X_test, Y_test, mask_test, \
    (embedd_table, word_freqs), label_alphabet, \
    C_train, C_dev, C_test, (char_embedd_table, char_freqs) = data_processor.load_dataset_sequence_labeling(train_path, dev_path,
                test_path, word_col_in_data, label_col_in_data,
                label_name=exp_mode, oov=oov,
                fine_tune=True,
                embedding=embedding, embedding_path=embedding_path,
                use_character=True)
    num_labels = label_alphabet.size() - 1

    logger.info("constructing network...")
    # create variables
    target_var = T.imatrix(name='targets')
    mask_var = T.matrix(name='masks', dtype=theano.config.floatX)
    num_tokens = mask_var.sum(dtype=theano.config.floatX)
    if fine_tune:
        input_var = T.imatrix(name='inputs')
        num_data, max_length = X_train.shape
        alphabet_size, embedd_dim = embedd_table.shape
    else:
        input_var = T.tensor3(name='inputs', dtype=theano.config.floatX)
        num_data, max_length, embedd_dim = X_train.shape
    char_input_var = T.itensor3(name='char-inputs')
    num_data_char, max_sent_length, max_char_length = C_train.shape
    char_alphabet_size, char_embedd_dim = char_embedd_table.shape
    assert (max_length == max_sent_length)
    assert (num_data == num_data_char)

    # prepare initial input layer and embeddings
    char_layer = construct_char_input_layer()
    word_layer = construct_input_layer()
    char_emb = Lyrs.get_output(char_layer)
    word_emb = Lyrs.get_output(word_layer)

    # construct input and mask layers
    char_in_layer = Lyrs.InputLayer(shape=(None, max_char_length, char_embedd_dim))
    word_in_layer = Lyrs.InputLayer(shape=(None, max_length, embedd_dim))

    layer_mask = lasagne.layers.InputLayer(shape=(None, max_length), input_var=mask_var, name='mask')

    # construct bilstm_bilstm_crf
    num_units = args.num_units
    num_filters = args.num_filters
    logger.info("Network structure: hidden=%d, filter=%d" % (num_units, num_filters))

    bilstm_bilstm_crf = build_BiLSTM_BiLSTM_CRF(char_in_layer, word_in_layer, num_units, num_labels, mask=layer_mask,
                                           grad_clipping=grad_clipping, peepholes=peepholes, num_filters=num_filters,
                                           dropout=dropout)

    # compute loss
    def loss_from_embedding(char_emb, word_emb, deterministic=False, return_all=True):
        # get outpout of bi-lstm-cnn-crf shape [batch, length, num_labels, num_labels]
        energies = Lyrs.get_output(bilstm_bilstm_crf,
                    inputs={char_in_layer: char_emb, word_in_layer: word_emb}, deterministic=deterministic)
        loss = crf_loss(energies, target_var, mask_var).mean()
        if return_all:
            predict, corr = crf_accuracy(energies, target_var)
            corr = (corr * mask_var).sum(dtype=theano.config.floatX)
            return loss, predict, corr
        else:
            return loss

    loss_eval, prediction_eval, corr_eval = loss_from_embedding(char_emb, word_emb, deterministic=True)
    loss_train_ori, _, corr_train = loss_from_embedding(char_emb, word_emb)

    if args.adv:
        logger.info('Preparing adversarial training...')
        loss_train_adv = adversarial_loss(char_emb, word_emb, loss_from_embedding, loss_train_ori, perturb_scale=args.adv)
        loss_train = (loss_train_ori + loss_train_adv) / 2.0
    else:
        loss_train_adv = T.as_tensor_variable(np.asarray(0.0, dtype=theano.config.floatX))
        loss_train = loss_train_ori + loss_train_adv

    # l2 regularization?
    if regular == 'l2':
        l2_penalty = lasagne.regularization.regularize_network_params(bilstm_bilstm_crf, lasagne.regularization.l2)
        loss_train = loss_train + gamma * l2_penalty

    # Create update expressions for training.
    # hyper parameters to tune: learning rate, momentum, regularization.
    batch_size = args.batch_size
    learning_rate = 1.0 if update_algo == 'adadelta' else args.learning_rate
    decay_rate = args.decay_rate
    momentum = 0.9
    params = Lyrs.get_all_params(bilstm_bilstm_crf, trainable=True) + Lyrs.get_all_params(char_layer, trainable=True) + Lyrs.get_all_params(word_layer, trainable=True)
    updates = utils.create_updates(loss_train, params, update_algo, learning_rate, momentum=momentum)

    # Compile a function performing a training step on a mini-batch
    train_fn = theano.function([input_var, target_var, mask_var, char_input_var], [loss_train_ori, loss_train_adv, corr_train, num_tokens],
                               updates=updates)
    # Compile a second function evaluating the loss and accuracy of network
    eval_fn = theano.function([input_var, target_var, mask_var, char_input_var],
                              [loss_eval, corr_eval, num_tokens, prediction_eval])

    # reload saved model
    if args.reload is not None:
        logger.info('Reloading saved parameters from %s ...\n' % args.reload)
        with np.load(args.reload) as f:
            param_values = [f['arr_%d' % j] for j in range(len(f.files))]
        Lyrs.set_all_param_values(word_layer, param_values[0:1])
        Lyrs.set_all_param_values(char_layer, param_values[1:2])
        Lyrs.set_all_param_values(bilstm_bilstm_crf, param_values[2:])

    # Finally, launch the training loop.
    logger.info(
        "Start training: %s with regularization: %s(%f), dropout: %s, fine tune: %s (#training data: %d, batch size: %d, clip: %.1f, peepholes: %s) ..." \
        % (
            update_algo, regular, (0.0 if regular == 'none' else gamma), dropout, fine_tune, num_data, batch_size,
            grad_clipping,
            peepholes))
    num_batches = num_data / batch_size
    num_epochs = 1000
    best_acc = np.array([0.0, 0.0, 0.0])
    best_epoch_acc = np.array([0, 0, 0])
    best_acc_test_err = np.array([0.0, 0.0, 0.0])
    best_acc_test_corr = np.array([0.0, 0.0, 0.0])
    stop_count = 0
    lr = learning_rate
    patience = args.patience
    for epoch in range(1, num_epochs + 1):
        print
        print 'Epoch %d (learning rate=%.7f, decay rate=%.4f): ' % (epoch, lr, decay_rate)
        train_err_ori = 0.0
        train_err_adv = 0.0
        train_corr = 0.0
        train_total = 0
        train_inst = 0
        start_time = time.time()
        num_back = 0
        train_batches = 0

        epoch_save_dir = os.path.join(save_dir, 'epoch%d' % epoch)
        os.mkdir(epoch_save_dir)

        for batch in utils.iterate_minibatches(X_train, Y_train, masks=mask_train, char_inputs=C_train,
                                               batch_size=batch_size, shuffle=True):
            inputs, targets, masks, char_inputs = batch
            err_ori, err_adv, corr, num = train_fn(inputs, targets, masks, char_inputs)
            train_err_ori += err_ori * inputs.shape[0]
            train_err_adv += err_adv * inputs.shape[0]
            train_corr += corr
            train_total += num
            train_inst += inputs.shape[0]
            train_batches += 1
            time_ave = (time.time() - start_time) / train_batches
            time_left = (num_batches - train_batches) * time_ave

            # update log
            if train_batches % (num_batches//10) == 0:
                log_info = 'train: %d/%d L_ori: %.4f, L_adv: %.4f, acc: %.2f%%, time left: %.2fs\n' % (
                    min(train_batches * batch_size, num_data), num_data,
                    train_err_ori / train_inst, train_err_adv / train_inst, train_corr * 100 / train_total, time_left)
                sys.stdout.write(log_info)
                sys.stdout.flush()

                # save the parameter values
                #param_values = Lyrs.get_all_param_values(bilstm_bilstm_crf)
                #np.savez(epoch_save_dir + '/iter%d.npz' % train_batches, *param_values)

        # save the parameter values
        param_values = Lyrs.get_all_param_values(word_layer) + Lyrs.get_all_param_values(char_layer) + Lyrs.get_all_param_values(bilstm_bilstm_crf)
        np.savez(epoch_save_dir + '/final.npz', *param_values)

        # update training log after each epoch
        assert train_inst == num_data
        print 'train: %d/%d L_ori: %.4f, L_adv: %.4f, acc: %.2f%%, time: %.2fs' % (
            min(train_batches * batch_size, num_data), num_data,
            train_err_ori / train_inst, train_err_adv / train_inst, train_corr * 100 / train_total, time.time() - start_time)

        # evaluate performance on dev data
        dev_err = 0.0
        dev_corr = 0.0
        dev_total = 0
        dev_inst = 0
        for batch in utils.iterate_minibatches(X_dev, Y_dev, masks=mask_dev, char_inputs=C_dev, batch_size=batch_size):
            inputs, targets, masks, char_inputs = batch
            err, corr, num, predictions = eval_fn(inputs, targets, masks, char_inputs)
            dev_err += err * inputs.shape[0]
            dev_corr += corr
            dev_total += num
            dev_inst += inputs.shape[0]
            if output_predict:
                output_file = eval_dir+'/dev%d' % epoch
                utils.output_predictions(predictions, targets, masks, output_file, label_alphabet,
                                         is_flattened=False)

        print 'dev loss: %.4f, corr: %d, total: %d, acc: %.2f%%' % (
            dev_err / dev_inst, dev_corr, dev_total, dev_corr * 100 / dev_total)

        #update_loss = False
        update_acc = False
        if best_acc.min() > dev_corr / dev_total:
            stop_count += 1
        else:
            stop_count = 0
            if best_acc.min() < dev_corr / dev_total:
                update_acc = True
                idx_to_update = best_acc.argmin()
                best_acc[idx_to_update] = dev_corr / dev_total
                best_epoch_acc[idx_to_update] = epoch

        # evaluate on test data
        test_err = 0.0
        test_corr = 0.0
        test_total = 0
        test_inst = 0
        for batch in utils.iterate_minibatches(X_test, Y_test, masks=mask_test, char_inputs=C_test,
                                               batch_size=batch_size):
            inputs, targets, masks, char_inputs = batch
            err, corr, num, predictions = eval_fn(inputs, targets, masks, char_inputs)
            test_err += err * inputs.shape[0]
            test_corr += corr
            test_total += num
            test_inst += inputs.shape[0]
            if output_predict:
                output_file = eval_dir+'/test%d' % epoch
                utils.output_predictions(predictions, targets, masks, output_file, label_alphabet,
                                         is_flattened=False)

        # print out test result
        if stop_count > 0:
            print '(cf.',
        print 'test loss: %.4f, corr: %d, total: %d, acc: %.2f%%' % (
            test_err / test_inst, test_corr, test_total, test_corr * 100 / test_total),
        if output_predict and exp_mode in ['ner', 'chunk']:
            stdout = subprocess.check_output([eval_script], stdin=open(output_file))
            f1_score = stdout.split("\n")[1].split()[7] # this is string
            print ", f1:", f1_score
        else:
            print
        sys.stdout.flush()

        if update_acc:
            best_acc_test_err[idx_to_update] = test_err
            best_acc_test_corr[idx_to_update] = test_corr

        # stop if dev acc decrease 3 time straightly.
        if stop_count == patience:
            break

        # re-compile a function with new learning rate for training
        if update_algo not in ['adam','adadelta']:
            if decay_rate >= 0:
                lr = learning_rate / (1.0 + epoch * decay_rate)
            else:
                if stop_count > 0 and stop_count%3 == 0:
                    learning_rate /= 2.0
                    lr = learning_rate
            updates = utils.create_updates(loss_train, params, update_algo, lr, momentum=momentum)
            train_fn = theano.function([input_var, target_var, mask_var, char_input_var],
                                        [loss_train_ori, loss_train_adv , corr_train, num_tokens],
                                        updates=updates)



    # print best performance on test data.
    for i in range(len(best_epoch_acc)):
        logger.info("final best acc test performance (at epoch %d)" % best_epoch_acc[i])
        print 'test loss: %.4f, corr: %d, total: %d, acc: %.2f%%' % (
            best_acc_test_err[i] / test_inst, best_acc_test_corr[i], test_total, best_acc_test_corr[i] * 100 / test_total)


if __name__ == '__main__':
    main()
