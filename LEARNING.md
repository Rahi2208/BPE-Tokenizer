#setup environment 

curl -LsSf https://astral.sh/uv/install.sh | sh

mkdir bpe-tokenizer && cd bpe-tokenizer #create directory and change directory in cli 
uv init
uv add --dev pytest
uv add regex tiktoken 

#make a repo layout in vscode 

#set git 

//conect to github 
gh auth login 

git init 
git status 
git add -A 
git commit -m "init: project skeleton" 

gh repo create name --public --source=. --remote=origin 
git push -u origin master 

core concept: 
tokenization is one of the fundamental in NLP model training. 
it is essential to refer characters to bytes which the dtype that computer could understand. 

the overall rule is to build a vocabulary in traning phase and when we want to input new words, it will got encoded by the same pattern, then returned the equivalent value of that token 

there are different methods in tokenize text, but two that we'll focus on in this project is basic tokenizer and regex tokenizer (used in gpt-2 and gpt-4 model) 

Basic Tokenizer: 
1. parsed and encode it into utf-8 as the default vocab list(text.encode('utf-8')), 'banana' -> [23, 78, 88, 78, 88,78]

2. find the quantity of subwords in a word 
def get_pair_stats(ids), [23, 78, 88, 78,88, 78] -> [
    (23,78):1, (78, 88):2, (88, 78): 1] 

3. merge dominate subwords and create new vocab
def merge(ids, pair, idx), [23, 78, 88, 78, 88, 78], 
[(78,88): 257], [23, 257, 257, 78] -> [23, 258, 78] (process stop) 

4. replace control characters 
there are characters that doesnt have a meaning yet necessary in text, for example '\n', '\t', etc. it doesnt included in our vocab yet, so it doesnt have a meaning, we need to replace it with something. 
C = control character
def replace_control_characters(s:str)->str, C-> f"\\u{ord(ch):04x}"

5. decode token
from [67, 450, 152, . . ] - dog!

the problem with basic tokenization: because basic tokenization only accounts for the following token, words like 'dog!', 'dog.', 'dog@' will have different token (3 seperate tokens), it will cause unnecessary vocab like 
['!', '.', 'dog', 'dog!', 'dog.', . . ]

the solution: parse each token with regex pattern. 
The regex pattern splits text into chunks based on category (letters, numbers, punctuation, etc.) before BPE merging runs. Since merges can't cross chunk boundaries, a word like "dog" always stays separate from whatever punctuation follows it ("!", ".", "?"). As a result, during BPE training, "dog" will consistently appear as the same chunk in many places across the corpus, so BPE will naturally merge it into a single token — rather than ending up with separate tokens like "dog.", "dog!", "dog?". This saves vocab slots (which are limited in number) and makes the model generalize better, since one word is always represented by the same token across different contexts.

Regex Tokenizer: 
1. Split text using a regex pattern first (before encoding to bytes)

Before BPE runs, the text is split into rough chunks using a regex (the GPT-2/GPT-4 pattern typically separates words, numbers, punctuation, spaces, contractions). This is different from the basic tokenizer, which encodes everything straight to bytes with no splitting at all

import regex as re
pattern = r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
text = "banana split!"
chunks = re.findall(pattern, text)
# -> ['banana', ' split', '!']

2. Encode each chunk into utf-8 bytes separately
chunks_bytes = [list(ch.encode('utf-8')) for ch in chunks]
# -> [[98,97,110,97,110,97], [32,115,112,108,105,116], [33]]

3. Compute pair stats, but per chunk — pairs are NOT allowed to cross chunk boundaries

def get_pair_stats(chunks_bytes):
    stats = {}
    for ids in chunks_bytes:
        for pair in zip(ids, ids[1:]):
            stats[pair] = stats.get(pair, 0) + 1
    return stats

This is the key difference from the basic tokenizer: in the basic tokenizer, pairs are counted across one long continuous stream. In the regex tokenizer, pairs are only counted within a single chunk, so the last byte of chunk A can never be paired with the first byte of chunk B

4. Merge the most frequent pair — still per chunk, boundaries still enforced
def merge(chunks_bytes, pair, idx):
    new_chunks = []
    for ids in chunks_bytes:
        new_chunks.append(merge_single(ids, pair, idx))  # same merge function as basic
    return new_chunks

5. (utility, not a core step) Replace control characters

Same as the basic tokenizer — this function is only for printing/displaying tokens so characters like \n, \t don't mess up the output when visualized. It's not part of the actual training/encoding process.

6. Decode token

Exactly like the basic tokenizer: ID → look up vocab → get bytes → join all bytes together (chunks get merged back into one stream too during decoding) → utf-8 decode → original string.