import pickle
import concurrent.futures
import time
import os

import torch.cuda
from torch.utils.data import DataLoader
from torch import optim

from model import translate, load_checkpoint, Transformer
from torchtext.data.metrics import bleu_score
from dataloader import map_dataset_test
from tokenize_custom import load_data, tokenize_ge, tokenize_en, vocab


def bleu(data, model, german, english, device):
    model.eval()
    targets = []
    outputs = []

    for example in data:
        src = example["inp"]
        trg = example["tar"]
        prediction = translate(src, model, german, english, device)  # remove <eos> token
        print(prediction, trg[:-1])
        targets.append([trg[:-1]])
        outputs.append(prediction.split())
    return bleu_score(outputs, targets, max_n=3, weights=[1 / 3, 1 / 3, 1 / 3])


eng_valid = vocab("/home/sonu/Desktop/torch/data/nmt/valid/val.en", tokenize_en)
ger_valid = vocab("/home/sonu/Desktop/torch/data/nmt/valid/val.de", tokenize_ge)
ger_stoi = open("/home/sonu/PycharmProjects/nmt/ger_stoi.pickle", "rb")
eng_stoi = open("/home/sonu/PycharmProjects/nmt/eng_stoi.pickle", "rb")
ger_itos = open("/home/sonu/PycharmProjects/nmt/ger_itos.pickle", "rb")
eng_itos = open("/home/sonu/PycharmProjects/nmt/eng_itos.pickle", "rb")
eng_valid.stoi, eng_valid.itos = pickle.load(eng_stoi), pickle.load(eng_itos)
ger_valid.stoi, ger_valid.itos = pickle.load(ger_stoi), pickle.load(ger_itos)
if not os.path.isfile("/home/sonu/PycharmProjects/nmt/inp_valid.pickle"):
    print("starting threads")
    start_time = time.time()
    # que = mp.Queue()
    with concurrent.futures.ThreadPoolExecutor() as executor:
        f1 = executor.submit(load_data, "/home/sonu/Desktop/torch/data/nmt/valid/val.en", eng_valid)
        f2 = executor.submit(load_data, "/home/sonu/Desktop/torch/data/nmt/valid/val.de", ger_valid)
        dataset_eng = f1.result()
        dataset_ger = f2.result()
    pickle.dump(dataset_ger, open("/home/sonu/PycharmProjects/nmt/inp_valid.pickle", "wb"))
    pickle.dump(dataset_eng, open("/home/sonu/PycharmProjects/nmt/out_valid.pickle", "wb"))
    end_time = time.time()
    print("took {} seconds for building dataset".format(end_time - start_time))
else:
    dataset_eng = pickle.load(open("/home/sonu/PycharmProjects/nmt/out_valid.pickle", "rb"))
    dataset_ger = pickle.load(open("/home/sonu/PycharmProjects/nmt/inp_valid.pickle", "rb"))

src_vocab_size = max(ger_valid.stoi.values()) + 1
trg_vocab_size = max(eng_valid.stoi.values()) + 1
src_pad_idx = eng_valid.stoi["<pad>"]
lr = 3e-4
# print(src_vocab_size, trg_vocab_size)
embedding_size = 1024
num_heads = 8
num_encoder_layers = 3
num_decoder_layers = 3
dropout = 0.10
max_len = 100
forward_expansion = 4
src_pad_idx = eng_valid.stoi["<pad>"]
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"using device {device}")
model = Transformer(
    embedding_size,
    src_vocab_size,
    trg_vocab_size,
    src_pad_idx,
    num_heads,
    num_encoder_layers,
    num_decoder_layers,
    forward_expansion,
    dropout,
    max_len,
    device
).to(device)

valid_dataset = map_dataset_test("/home/sonu/Desktop/torch/data/nmt/test/test.de",
                                 "/home/sonu/Desktop/torch/data/nmt/test/test.en")
optimizer = optim.Adam(model.parameters(), lr=lr)
load_checkpoint(torch.load("checkpoints/my_checkpoint5.pth.tar"), model, optimizer)
model.eval()
print(len(valid_dataset))
print(bleu(valid_dataset, model, ger_valid, eng_valid, device))
print(translate("ein pferd geht unter einer brücke neben einem boot", model, ger_valid, eng_valid, device))
