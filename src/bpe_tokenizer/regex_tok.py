from .base import Tokenizer, get_pair_stats, merge, replace_control_characters, render_token
import regex as re

GPT2_SPLIT_PATTERN = (
    r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
)

GPT4_SPLIT_PATTERN = (
    r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}++|\p{N}{1,3}+"""
    r"""| ?[^\s\p{L}\p{N}]++[\r\n]*+|\s++$|\s*[\r\n]|\s+(?!\S)|\s"""
)

class RegexTokenizer(Tokenizer):
 
    def __init__(self, pattern=None):
        super().__init__()     
        self.pattern = GPT4_SPLIT_PATTERN if pattern is None else pattern
        self.compiled_pattern = re.compile(self.pattern)
        self.special_tokens = {}           # str -> int
        self.inverse_special_tokens = {}   # int -> str
 
    def train(self, text, vocab_size, verbose=False):
        assert vocab_size >= 256
        num_merges = vocab_size - 256
 
        # "Hello world!!" -> ["Hello", " world", "!!"]
        text_chunks = re.findall(self.compiled_pattern, text)
 
        # Each chunk becomes its own list of byte IDs. So `ids` here is
        # a list of lists — this is how it differs from PART 3, which is just a single list.

        ids = [list(chunk.encode("utf-8")) for chunk in text_chunks]
 
        merges = {}
        vocab = {idx: bytes([idx]) for idx in range(256)}
 
        for i in range(num_merges):
            stats = {}
            for chunk_ids in ids:
                get_pair_stats(chunk_ids, stats)   
            
            if not stats:
                if verbose:
                    print(f"stop early at {i} merge: none is available to be merged")
                break
 
            pair = max(stats, key=stats.get)
            idx = 256 + i
            ids = [merge(chunk_ids, pair, idx) for chunk_ids in ids]
            merges[pair] = idx
            vocab[idx] = vocab[pair[0]] + vocab[pair[1]]
 
            if verbose:
                print(f"merge {i + 1}/{num_merges}: {pair} -> {idx} "
                    f"({vocab[idx]!r}) muncul {stats[pair]} kali")
 
        self.merges = merges
        self.vocab = vocab
 
    def register_special_tokens(self, special_tokens):
        self.special_tokens = dict(special_tokens)
        self.inverse_special_tokens = {v: k for k, v in special_tokens.items()}
 
    def decode(self, ids):
        part_bytes = []
        for idx in ids:
            if idx in self.vocab:
                part_bytes.append(self.vocab[idx])
            elif idx in self.inverse_special_tokens:
                part_bytes.append(self.inverse_special_tokens[idx].encode("utf-8"))
            else:
                raise ValueError(f"token id tidak valid: {idx}")
        text_bytes = b"".join(part_bytes)
        return text_bytes.decode("utf-8", errors="replace")
 
    def _encode_chunk(self, text_bytes):
        ids = list(text_bytes)
        while len(ids) >= 2:
            stats = get_pair_stats(ids)
            pair = min(stats, key=lambda p: self.merges.get(p, float("inf")))
            if pair not in self.merges:
                break
            ids = merge(ids, pair, self.merges[pair])
        return ids
 
    def encode_ordinary(self, text):
        text_chunks = re.findall(self.compiled_pattern, text)
        ids = []
        for chunk in text_chunks:
            ids.extend(self._encode_chunk(chunk.encode("utf-8")))
        return ids
 
    def encode(self, text, allowed_special="none_raise"):
        if allowed_special == "all":
            special = self.special_tokens
        elif allowed_special == "none":
            special = {}
        elif allowed_special == "none_raise":
            special = {}
            for token in self.special_tokens:
                if token in text:
                    raise ValueError(
                        f"teks berisi special token {token!r}; pakai "
                        f"allowed_special='all' atau 'none' untuk memilih secara eksplisit"
                    )
        elif isinstance(allowed_special, set):
            special = {k: v for k, v in self.special_tokens.items() if k in allowed_special}
        else:
            raise ValueError(f"allowed_special={allowed_special!r} tidak dikenali")
 
        if not special:
            return self.encode_ordinary(text)
        special_pattern = "(" + "|".join(re.escape(k) for k in special) + ")"
        special_chunks = re.split(special_pattern, text)
 
        ids = []
        for part in special_chunks:
            if part in special:
                ids.append(special[part])
            else:
                ids.extend(self.encode_ordinary(part))
        return ids
 
    def load(self, model_file):
        super().load(model_file)   # <-- jalankan versi induknya dulu
        self.compiled_pattern = re.compile(self.pattern)
        self.inverse_special_tokens = {v: k for k, v in self.special_tokens.items()}
