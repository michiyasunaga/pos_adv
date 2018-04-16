# __author__ = 'max'
# modified by michihiro

"""
Alphabet maps objects to integer ids. It provides two way mapping from the index to the objects.
"""
import json
import os
import numpy as np
from lasagne_nlp.utils import utils as utils


class Alphabet:
    def __init__(self, name, keep_growing=True):
        self.__name = name

        self.instance2index = {}
        self.instances = []
        self.vocab_freqs = [0] # key: index, value: frequency
        # initial: <unknown> 0 freq

        self.keep_growing = keep_growing

        # Index 0 is occupied by default, all else following.
        self.default_index = 0 # this is for <unknown>
        self.next_index = 1

        self.logger = utils.get_logger('Alphabet')

    def add(self, instance):
        if instance not in self.instance2index:
            self.instances.append(instance)
            self.instance2index[instance] = self.next_index
            self.next_index += 1

    def get_index(self, instance): # mainly use this
        try:
            index = self.instance2index[instance]
            self.vocab_freqs[index] += 1
            return index
        except KeyError:
            if self.keep_growing:
                index = self.next_index
                self.add(instance)
                self.vocab_freqs.append(1)
                return index
            else:
                self.vocab_freqs[self.default_index] += 1
                return self.default_index

    def get_instance(self, index):
        if index == 0:
            # First index is occupied by the wildcard element.
            return None
        try:
            return self.instances[index - 1]
        except IndexError:
            self.logger.warn('unknown instance, return the first label.')
            return self.instances[0]

    def get_vocab_freqs(self):
        assert len(self.vocab_freqs) == len(self.instances) + 1
        return self.vocab_freqs
        #return np.asarray(self.vocab_freqs, dtype=theano.config.floatX) # shape: (vocab_size, )

    def size(self):
        return len(self.instances) + 1  #vocab_size

    def iteritems(self):
        return self.instance2index.iteritems()

    def enumerate_items(self, start=1):
        if start < 1 or start >= self.size():
            raise IndexError("Enumerate is allowed between [1 : size of the alphabet)")
        return zip(range(start, len(self.instances) + 1), self.instances[start - 1:])

    def close(self):
        self.keep_growing = False

    def open(self):
        self.keep_growing = True

    def get_content(self):
        return {'instance2index': self.instance2index, 'instances': self.instances}

    def from_json(self, data):
        self.instances = data["instances"]
        self.instance2index = data["instance2index"]

    def save(self, output_directory, name=None):
        """
        Save both alhpabet records to the given directory.
        :param output_directory: Directory to save model and weights.
        :param name: The alphabet saving name, optional.
        :return:
        """
        saving_name = name if name else self.__name
        try:
            json.dump(self.get_content(), open(os.path.join(output_directory, saving_name + ".json"), 'w'))
        except Exception as e:
            self.logger.warn("Alphabet is not saved: " % repr(e))

    def load(self, input_directory, name=None):
        """
        Load model architecture and weights from the give directory. This allow we use old models even the structure
        changes.
        :param input_directory: Directory to save model and weights
        :return:
        """
        loading_name = name if name else self.__name
        self.from_json(json.load(open(os.path.join(input_directory, loading_name + ".json"))))
