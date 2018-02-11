# glove-embeddings
Helper functions to load GloVe word vectors


## Usage
Import and initialize Glove

```python
from embeddings import Glove

glove = Glove()
```

### Loading the vectors
Load the vectors from the file using `load_vectors()`. Ensure that you do __NOT__ modify the default names of your files. You can set `verbose=True` to see the loading progress.
```python
glove_file_path = 'Downloads/glove.6B.50d.txt' # This is the path to your glove file
glove.load_vectors()
```

### Getting the embedding matrix
You can fetch the embedding matrix as shown below. The embedding matrix will be of shape `(vocab_size, embedding_dim)`.
```python
weights = glove.get_embedding_matrix()
print(weights.shape)

(400000, 50)
```

### Getting the word indices
Each row in the embedding matrix corresponds to the word vector of a word. The word indices denote the row number of a particular word. `get_word_idxs` returns a dictionary mapping words to corresponding word index.
```python
word_idxs = glove.get_word_idxs()
print(word_idxs['rocket'])

3034
```

### Creating a subset of vectors
`get_embedding_subset()` allows you to create a smaller embedding matrix from the original matrix with your own word indices. You must provide a word2idx dictionary mapping the words in your subset to their corresponing indices. The function selects the word vectors and orders them accordingly in a smaller matrix. If any word in the subset is not found, its vector is set randomly.

```python
groot_vocab = {'I':0, 'am':1, 'Groot':2}
groot_matrix = glove.get_embedding_subset(groot_vocab)
print(groot_matrix.shape)

(3, 50)
```
