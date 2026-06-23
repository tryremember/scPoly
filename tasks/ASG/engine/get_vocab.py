import json


class VocabPybind:
    def __init__(self, tokens, indices):
        self.token_to_index = dict(zip(tokens, indices))
        self.index_to_token = {idx: tok for tok, idx in self.token_to_index.items()}

    def __getitem__(self, token):
        return self.token_to_index[token]

    def get_token(self, index):
        return self.index_to_token.get(index, None)

    def __len__(self):
        return len(self.token_to_index)


class Vocab:
    def __init__(self, vocab_pybind):
        self.vocab = vocab_pybind
        self.default_index = None

    def set_default_index(self, index):
        self.default_index = index

    def __getitem__(self, token):
        return self.vocab.token_to_index.get(token, self.default_index)

    def to_token(self, index):
        return self.vocab.get_token(index)

    def __len__(self):
        return len(self.vocab)


def get_vocab(vocab_fpath):
    with open(vocab_fpath) as f:
        vocab_dict = json.load(f)

    tokens = list(vocab_dict.keys())
    indices = list(vocab_dict.values())
    vocab_pybind = VocabPybind(tokens, indices)
    vocab = Vocab(vocab_pybind)
    vocab.set_default_index(vocab["<pad>"])
    return vocab
