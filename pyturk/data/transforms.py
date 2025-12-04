
class TextToIds:
    def __init__(self, vocab):
        self.vocab = vocab # dict {token: id}

    def __call__(self, text):
        return [self.vocab.get(tok, 0) for tok in text.split()]
    
class PadSequence:
    def __init__(self, max_length, pad_id=0):
        self.max_length = max_length
        self.pad_id = pad_id

    def __call__(self, seq):
        if len(seq) >= self.max_length:
            return seq[:self.max_length]
        else:
            return seq + [self.pad_id] * (self.max_length - len(seq))