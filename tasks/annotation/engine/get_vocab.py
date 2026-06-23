import json

class VocabPybind:
    """
    Lightweight vocabulary wrapper providing token-index bidirectional mapping.

    """
    def __init__(self, tokens, indices):
        """
        Initialize vocabulary mappings from tokens and indices.

        Parameters
        ----------
        tokens : iterable
            Sequence of token strings.
        indices : iterable
            Sequence of integer indices corresponding to tokens.
        """
        self.token_to_index = dict(zip(tokens, indices))
        self.index_to_token = {idx: tok for tok, idx in self.token_to_index.items()}
    
    def __getitem__(self, token):
        """
        Retrieve index for a given token.

        Parameters
        ----------
        token : str
            Token to look up.

        Returns
        -------
        int
            Index associated with the token.
        """
        return self.token_to_index[token]
    
    def get_token(self, index):
        """
        Retrieve token for a given index.

        Parameters
        ----------
        index : int
            Index to look up.

        Returns
        -------
        str or None
            Token associated with the index, or None if not found.
        """
        return self.index_to_token.get(index, None)
    
    def __len__(self):
        """
        Return the vocabulary size.

        Returns
        -------
        int
            Number of tokens in the vocabulary.
        """
        return len(self.token_to_index)


class Vocab:
    """
    High-level vocabulary interface with default index fallback.

    """
    def __init__(self, vocab_pybind):
        """
        Initialize the vocabulary wrapper.

        Parameters
        ----------
        vocab_pybind : VocabPybind
            Underlying vocabulary object providing token-index mappings.
        """
        self.vocab = vocab_pybind
        self.default_index = None
    
    def set_default_index(self, index: int):
        """
        Set the default index for unknown tokens.

        Parameters
        ----------
        index : int
            Index to return when a token is not found.
        """
        self.default_index = index

    def __getitem__(self, token: str) -> int:
        """
        Retrieve index for a token with fallback to default index.

        Parameters
        ----------
        token : str
            Token to look up.

        Returns
        -------
        int
            Index associated with the token, or the default index if missing.
        """
        return self.vocab.token_to_index.get(token, self.default_index)

    def to_token(self, index: int) -> str:
        """
        Retrieve token corresponding to an index.

        Parameters
        ----------
        index : int
            Index to look up.

        Returns
        -------
        str or None
            Token associated with the index, or None if not found.
        """
        return self.vocab.get_token(index)
    
    def __len__(self):
        """
        Return the vocabulary size.

        Returns
        -------
        int
            Number of tokens in the vocabulary.
        """
        return len(self.vocab)

    

def get_vocab(vocab_fpath):
    """
    Load vocabulary from a JSON file and construct a Vocab object.

    Parameters
    ----------
    vocab_fpath : str
        Path to the vocabulary JSON file containing token-index mappings.

    Returns
    -------
    Vocab
        Initialized vocabulary with default index set.
    """
    with open(vocab_fpath) as f:
        vocab_dict = json.load(f)
    tokens = list(vocab_dict.keys())
    indices = list(vocab_dict.values())
    vocab_pybind = VocabPybind(tokens, indices)
    vocab = Vocab(vocab_pybind)
    vocab.set_default_index(vocab["<pad>"])  
    return vocab
