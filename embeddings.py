import numpy as np
import os

def _get_specs(file_path):
    '''
        Returns the vocabulary size and
        embedding dimension from Glove
        file name
    '''
    file_name = os.path.basename(file_path)
    dim = int(file_name.split('.')[-2][:-1])
    vocab_sizes = {42: 1917494, 840: 2196017,
                   27: 1193514, 6: 400000}
    toks = int(file_name.split('.')[-3][:-1])

    return vocab_sizes[toks], dim


class Glove(object):

    def __init__(self):
        self.embeddings = None
        self.word2idx = None

    def load_vectors(self, file_path, verbose=False):
        """Loads the word vectors from file.

        # Arguments
            file_path: Path to the glove vector
                text file. Note that the default
                file name should NOT be changed.
            verbose: Boolean. Set to `True` to
                display loading progress in percent
        """

        self.file_path = file_path
        self.vocab_size, self.dim = _get_specs(file_path)
        self.word2idx = {}
        self.embeddings = np.zeros([self.vocab_size, self.dim])

        with open(self.file_path, 'rb') as f:
            for i, line in enumerate(f):
                split_line = line.split()
                word = split_line[0].decode('UTF-8')

                self.embeddings[i] = np.asarray([float(val) for val in split_line[1:]])
                self.word2idx[word] = i

                if verbose:
                    percent_completed = round(i * 100 / self.vocab_size, 2)
                    print('\r' + str(percent_completed) + '%', end='')
            
            if verbose:
                print()
                print(str(len(self.word2idx)), "word vectors loaded. Embedding dimension :", self.dim)

    def get_embedding_matrix(self):
        """Returns the embedding matrix if the vectors
           have been loaded

            # Raises:
                AssertionError: If vectors have not been loaded
        """
        assert self.word2idx is not None, "Load the vectors first using `load_vectors()`"
        return self.embeddings

    def get_embedding_subset(self, word2idx):
        """ Create an embedding matrix for a
            subset of the original vocabulary

            # Arguments:
                word2idx: A dict mapping words in
                    the subset to it's numerical index

            # Returns:
                A numpy array of shape (len(word2idx), embedding_dim)
            
            # Example:
                ```python
                    glove = Glove()
                    glove.load_vectors(path, verbose=True)
                    groot_words = {'I': 0, 'am': 1, 'Groot':2}
                    glove.get_embedding_subset(groot_words)
                ```
        """
        assert self.word2idx is not None, "Load the vectors first using `load_vectors()`"
        subset = np.zeros([len(word2idx), self.embeddings.shape[1]])
        for word in word2idx.keys():
            if word in self.word2idx:
                subset[word2idx[word]] = self.embeddings[self.word2idx[word]]
            else:
                subset[word2idx[word]] = np.random.randn(self.embeddings.shape[1],)
        return subset


    def get_word_idxs(self):
        """Returns a dictionary mapping words to numerical index word 
           in the embedding matrix
        """
        assert self.word2idx is not None, "Load the vectors first using `load_vectors()`"
        return self.word2idx
