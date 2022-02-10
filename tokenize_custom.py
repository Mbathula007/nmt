import spacy
import time
import random

# from torchtext.data import Field

import string


def remove_punc(text_list):
    table = str.maketrans('', '', string.punctuation)
    removed_punc_text = []
    for sent in text_list:
        sent = sent.lower()
        sentence = [w.translate(table) for w in sent.split(' ')]
        removed_punc_text.append(' '.join(sentence))
    return removed_punc_text


german_tok = spacy.load("de_core_news_sm")
english_tok = spacy.load("en_core_web_sm")


def tokenize_en(sentence):
    return [token.text for token in english_tok(sentence)]


def tokenize_ge(sentence):
    return [token.text for token in german_tok(sentence)]


class vocab(object):
    def __init__(self, loc, lang):
        self.file = open(loc)
        self.stoi = {"<pad>": 0, "<unk>": 1, "<sos>": 2, "<eos>": 3}
        self.itos = {0: "<pad>", 1: "<unk>", 2: "<sos>", 3: "<eos>"}
        self.lang = lang
        self.max_len = 0

    def build_vocab(self):
        curr_num = 4
        lines = self.file.readlines()
        lines = remove_punc(lines)
        random.shuffle(lines)
        for t in lines:
            token_sen = self.lang(t)
            self.max_len = max(self.max_len, len(token_sen))
            for tok in token_sen:
                if tok not in self.stoi:
                    self.stoi[str(tok)] = curr_num
                    self.itos[curr_num] = str(tok)
                    curr_num = curr_num + 1
        return self.stoi, self.itos


def load_data(file_name, vocab_obj):
    dataset = []
    file = open(file_name)
    lines = file.readlines()
    lines = remove_punc(lines)
    for t in lines:
        tokens = vocab_obj.lang(t)
        out_ = [vocab_obj.stoi["<sos>"]]
        for token in tokens:
            if str(token) not in vocab_obj.stoi:
                token = "<unk>"
            out_.append(vocab_obj.stoi[str(token)])
        out_.append(vocab_obj.stoi["<eos>"])
        dataset.append(out_)
    return dataset


"""
print("starting threads")
start_time = time.time()
# que = mp.Queue()
with concurrent.futures.ThreadPoolExecutor() as executor:
   f1 = executor.submit(load_data, "/home/sonu/Desktop/torch/data/nmt/train/train.en", eng_train)
   f2 = executor.submit(load_data, "/home/sonu/Desktop/torch/data/nmt/train/train.de", ger_train)
   dataset_eng = f1.result()
   dataset_ger = f2.result()
end_time = time.time()
print("took {} seconds for building dataset".format(end_time - start_time))

"""
