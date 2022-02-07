import concurrent.futures
import pickle
import time
import os

import numpy as np
import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from dataloader import iter_dataset, map_dataset
from tokenize_custom import tokenize_en, tokenize_ge, load_data
from tokenize_custom import vocab
from model import Transformer, translate,save_checkpoint

eng_train = vocab("/home/sonu/Desktop/torch/data/nmt/train/train.en", tokenize_en)
ger_train = vocab("/home/sonu/Desktop/torch/data/nmt/train/train.de", tokenize_ge)
if os.path.isfile("/home/sonu/PycharmProjects/nmt/ger_stoi.pickle"):
    ger_stoi = open("/home/sonu/PycharmProjects/nmt/ger_stoi.pickle", "rb")
    eng_stoi = open("/home/sonu/PycharmProjects/nmt/eng_stoi.pickle", "rb")
    ger_itos = open("/home/sonu/PycharmProjects/nmt/ger_itos.pickle", "rb")
    eng_itos = open("/home/sonu/PycharmProjects/nmt/eng_itos.pickle", "rb")
    eng_train.stoi, eng_train.itos = pickle.load(eng_stoi), pickle.load(eng_itos)
    ger_train.stoi, ger_train.itos = pickle.load(ger_stoi), pickle.load(ger_itos)
else:
    print("starting threads")
    start_time = time.time()
    with concurrent.futures.ThreadPoolExecutor() as executor:
        t1 = executor.submit(eng_train.build_vocab)
        t2 = executor.submit(ger_train.build_vocab)
        eng_train.stoi, eng_train.itos = t1.result()
        ger_train.stoi, ger_train.itos = t2.result()
    end_time = time.time()
    print("took {} seconds for building vocab".format(end_time - start_time))
    pickle.dump(eng_train.stoi, open("/home/sonu/PycharmProjects/nmt/eng_stoi.pickle", "wb"))
    pickle.dump(ger_train.stoi, open("/home/sonu/PycharmProjects/nmt/ger_stoi.pickle", "wb"))
    pickle.dump(eng_train.itos, open("/home/sonu/PycharmProjects/nmt/eng_itos.pickle", "wb"))
    pickle.dump(ger_train.itos, open("/home/sonu/PycharmProjects/nmt/ger_itos.pickle", "wb"))
map_ds = True
input_loc = "/home/sonu/Desktop/torch/data/nmt/train/train.en"
output_loc = "/home/sonu/Desktop/torch/data/nmt/train/train.de"
if map_ds:
    if not os.path.isfile("/home/sonu/PycharmProjects/nmt/inp.pickle"):
        print("starting threads")
        start_time = time.time()
        # que = mp.Queue()
        with concurrent.futures.ThreadPoolExecutor() as executor:
            f1 = executor.submit(load_data, "/home/sonu/Desktop/torch/data/nmt/train/train.en", eng_train)
            f2 = executor.submit(load_data, "/home/sonu/Desktop/torch/data/nmt/train/train.de", ger_train)
            dataset_eng = f1.result()
            dataset_ger = f2.result()
        pickle.dump(dataset_ger, open("/home/sonu/PycharmProjects/nmt/inp.pickle", "wb"))
        pickle.dump(dataset_eng, open("/home/sonu/PycharmProjects/nmt/out.pickle", "wb"))
        end_time = time.time()
        print("took {} seconds for building dataset".format(end_time - start_time))
    else:
        dataset_eng = pickle.load(open("/home/sonu/PycharmProjects/nmt/out.pickle", "rb"))
        dataset_ger = pickle.load(open("/home/sonu/PycharmProjects/nmt/inp.pickle", "rb"))
    dataset_ = map_dataset(dataset_eng, dataset_ger, max_len=100)
else:
    dataset_ = iter_dataset(input_loc, output_loc, eng_train, ger_train, max_len=100)
batch_size = 32
train_loader = DataLoader(dataset_, batch_size=batch_size, drop_last=True)

# print("for Ein the index is ",ger_train.stoi["Ein"])
ge = "Ein kleines Mädchen klettert in ein Spielhaus aus Holz."
en = "A little girl climbing into a wooden playhouse."

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("training on {} ".format(device))
load_model = False
save_model = True

num_epochs = 20
lr = 3e-4
src_vocab_size = max(ger_train.stoi.values()) + 1
trg_vocab_size = max(eng_train.stoi.values()) + 1
# print(src_vocab_size, trg_vocab_size)
embedding_size = 512
num_heads = 8
num_encoder_layers = 3
num_decoder_layers = 3
dropout = 0.10
max_len = 100
forward_expansion = 2048
src_pad_idx = eng_train.stoi["<pad>"]
writer = SummaryWriter("runs/loss_plot")
step = 0
step_ = 0

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
pad_idx = eng_train.stoi["<pad>"]
model.eval()
print(translate(ge, model, ger_train, eng_train, device))