from .base import Tokenizer, get_pair_stats, merge, replace_control_characters, render_token


class BasicTokenizer(Tokenizer): 
    def train(self, text, vocab_size):  
        assert vocab_size >= 256 
        num_merges = vocab_size-256 
        ids = list(text.encode('utf-8')) 
        merges = {} 

        #assign default ascii to vocab
        vocab = {idx:bytes([idx]) for idx in range(256)} 


        for i in range(num_merges): 
            stats = get_pair_stats(ids) 
            if not stats: 
                break 

            pair = max(stats, key=stats.get) 
            idx = 256+i 

            ids = merge(ids, pair, idx) 
            merges[pair] = idx 
            vocab[idx] = vocab[pair[0]] + vocab[pair[1]]
            print(f"merge {i + 1}/{num_merges}: {pair} -> {idx} "
                f"({vocab[idx]!r}) muncul {stats[pair]} kali")
 
        
        self.merges = merges
        self.vocab = vocab 

    def decode(self, ids): 
        text_bytes = b"".join(self.vocab[idx] for idx in ids) 
        return text_bytes.decode("utf-8", errors='replace')

    def encode(self, text): 
        ids = list(text.encode('utf-8')) 
        ids = list(text.encode("utf-8"))
        while len(ids) >= 2:
            stats = get_pair_stats(ids) 
        
            if pair not in self.merges:
                break
        
        pair = min(stats, key=lambda p: self.merges.get(p, float("inf")))
        ids = merge(ids, pair, self.merges[pair])
        return ids