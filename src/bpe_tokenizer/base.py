import unicodedata 

import regex as re 
#concepts:
#tokenizer is needed to represent texts/another media into tensor by breakdown it and change it into numbers
#theres so much tokenizer methods, one of them is BPE (byte pair encoding), which is one of the most used tokenization method(and the one we try to built here) due to its flexibility and efficiency 

#examples of BPE tokenizer's advantages: if a certain word is not listed in our vocabulary, bpe can parse it into known words
#for example, if we have a words 'unbelievable', we can parse it into 'un', 'believe', 'able' which one of them is known to our vocabulary, and we can understand the meaning of the word by its parts. 

#second, the efficiency to compress several subwords into a shorter one. 
#at first, text is converted into index of size len(text), with bpe, for example: 'banana' ->(indexing) ->['b','a','n',a','n','a'] -> (bpe) -> ['b', 'an' , 'an', 'a'] -> ['b', 'anan','a'] -> ['banan', 'a'] ->['banana'], from 6 tokens itu 1

#additional concept: bytes is 
#CHAPTER 1: MERGE SUBWORDS 

#get_stats([1, 2, 1, 2, 3])  ->  {(1, 2): 2, (2, 1): 1, (2, 3): 1}
def get_pair_stats(ids, counts=None): 
    counts = {} if counts is None else counts
    for pair in zip(ids, ids[1:]): 
        counts[pair] = counts.get(pair, 0) +1 
    return counts 

#merge([1, 2, 3, 1, 2], (1, 2), 99)  ->  [99, 3, 99] 
def merge(ids, pair, idx): 
    new_ids = [] 
    i= 0
    while i < len(ids): #i = 0, len(ids)=1,
        if i< len(ids) - 1 and ids[i] == pair[0] and ids[i+1] == pair[1]: 
            new_ids.append(idx) 
            i += 2
        else: 
            new_ids.append(ids[i])
            i+=1 
    return new_ids 


#tokenize control characters (\n, \r,\t, . . .)
#C = control characters. control characters referring to a certain chars that have a certain role but not included in the context of the word/sentence itself
#for example '/n' to create a newline, '/t' to create a tab, etc. 
#def replace_control_characters('dog\n': str) -> str:  -> "dog\\u0009" 
def replace_control_characters(s: str) -> str:
    chars = [] 
    for ch in s: 
        if unicodedata.category(ch)[0] != "C": 
            chars.append(ch) 
        else: 
            chars.append(f"\\u{ord(ch):04x}")  #f"\\u{ord(ch):04x}" = "\\u0009"
    return "".join(chars) 


#decode token 
#render_token(b'dog\n': bytes) -> str:  -> "dog\\u0009"
def render_token(token: bytes) -> str: 
    token_str = token.decode("utf-8", errors="replace") #a.decode('UTF-8', errors='replace')
    return replace_control_characters(token_str) 

# CHAPTER 2 — BASELINE CLASS
#
# This class cannot be used directly. train/encode/decode is empty 
# the use is only for 2: (a) saving the state for all tokenizer, 
# (b) save()/load() so that no use of constant rewriting 

class Tokenizer: 
    """
    state in each tokenizer: 
    
    merges =  dict[(int, int) -> int] 
    vocab  = dict[int -> bytes] 
    pattern = str 
    special_tokens = dict[str -> int]

    """
    def __init__(self): 
        self.merges = {} #{(100, 111): 500} 
        self.pattern = ""  #str
        self.special_tokens = {} #{"<|endoftext|>": 100257} 
        self.vocab = self.build_vocab() #{100:b'dog, . . .}

    def train(self, text, vocab_size, verbose=False): 
        raise NotImplementedError() 

    def encode(self, text): 
        raise NotImplementedError() 

    def decode(self, ids): 
        raise NotImplementedError() 

    #core concept: 
    #at first, we need to assign our vocabulary (by characters), we use ASCII as our default vocabulary reference. we need to coupled the index and the ASCII characters equivalent of that index
    # {idx: bytes[idx]}, bytes = automatically returning the  ASCII value of that index, {1: b'a', 2: b'c'}, with a and b as the element in ASCII corresponding to the key index
    def build_vocab(self): #vocab -> dict[idx:bytes]
        vocab = {idx: bytes([idx]) for idx in range(256)} #{idx: bytes[idx]}
        for (p0, p1), idx in self.merges.items(): #merges.shape = {(a,b):idx}, merge will assign new idx above 256, e.g kntl, k=67, n=90, . . ., {(k,n):677}, {(t,l):888}, {(kn, tl): 9000}, that's one of the advantages of bpe, it can widen the understanding of words 
            vocab[idx] = vocab[p0] + vocab[p1] 
        for special, idx, in self.special_tokens.items(): 
            vocab[idx] = special.encode('utf-8') #value nya di encode ke utf-8 format

        return vocab 

    def save(self, file_prefix): 
        """
        format .model: 
        1. title
        2. pattern 
        3. length of special token 
        4. list 1-256 default ASCII
        5. list special token 
        6. list merged token
        """
        model_file = file_prefix + ".model"  
        with open(model_file, "w", encoding='utf-8') as f: 
            f.write("bpe-tokenizer v1\n") 
            f.write(f"{self.pattern}\n") 
            f.write(f"{len(self.special_tokens)}\n") 

            for idx in range(256): 
                f.write(f'{bytes([idx])}{idx}')

            for special, idx in self.special_tokens.items(): 
                f.write(f"{special} {idx}\n") 

            for idx1, idx2 in self.merges: 
                f.write(f"{idx1} {idx2}") 


    def load(self, model_file):
        
        assert model_file.endswith(".model") #assert is a guard clause, it is identical with if a then b, its just assert that certain event is fulfilled
        merges = {}
        special_tokens = {}
        ascii = {}
        idx = 256
        with open(model_file, "r", encoding="utf-8") as f:
            version = f.readline().strip() #.strip() = to strip a space in both ends, .rstrip() only stripping the right end of a sentence 
            assert version == "bpe-tokenizer v1", f"format tidak dikenal: {version}" #assert can be quite similiar with ternary operation, assert condition, return if true
            self.pattern = f.readline().rstrip("\n")
            num_special = int(f.readline().strip())
            
            for _ in range(num_special):
                special, special_idx = f.readline().strip().split(" ")
                special_tokens[special] = int(special_idx)

            for _ in range(idx): 
                byte, id = f.readline().strip().split(" ") 
                ascii[int(id)] = byte 

            for line in f:
                idx1, idx2 = map(int, line.split())
                merges[(idx1, idx2)] = idx
                idx += 1
        
        self.merges = merges
        self.special_tokens = special_tokens
        self.vocab = self._build_vocab()

            

"""
vocab terdiri dari 3 jenis dtype, (asciii, special tokens, merged tokens), maka build vocab, save and load, harus ada itu pokoknya, 
kecuali di save, tambahin self.pattern juga karena kita save pattern nya (regex) with .rstrip('\n') 
"""