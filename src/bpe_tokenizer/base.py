class Tokenizer:
    def __init__(self):
        self.merges = {}   # (int,int) -> int
        self.vocab = {}    # int -> bytes

    def train(self, text, vocab_size, verbose=False):
        raise NotImplementedError   # tiap jenis tokenizer isi sendiri

    def encode(self, text):
        raise NotImplementedError

    def decode(self, ids):
        raise NotImplementedError

    def save(self, file_prefix):
        # nulis .model (buat di-load ulang) dan .vocab (buat dibaca manusia)
        ...

    def load(self, model_file):
        # baca file .model, rebuild merges + vocab
        ...