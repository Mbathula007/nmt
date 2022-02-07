import concurrent.futures
import pickle
import time
import os

import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from dataloader import iter_dataset, map_dataset
from tokenize_custom import tokenize_en, tokenize_ge, load_data
from tokenize_custom import vocab
from model import Transformer, translate, save_checkpoint

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
ge = "Ein kleines Mädchen klettert in ein Spielhaus aus Holz"
en = "A little girl climbing into a wooden playhouse"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("training on {} ".format(device))
load_model = False
save_model = True

num_epochs = 20
lr = 3e-2
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

optimizer = optim.Adam(model.parameters(), lr=lr)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, factor=0.1, patience=10, verbose=True
)
pad_idx = eng_train.stoi["<pad>"]
criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)
for epoch in range(num_epochs):
    print(f"[Epoch {epoch} / {num_epochs}]")
    model.eval()
    with torch.no_grad():
        writer.add_text("translated eng", translate(ge, model, ger_train, eng_train, device),global_step=step_)
        writer.add_text("actual eng sentence is ", en, global_step=step_)
        writer.add_text("actual ger sentence is ", ge, global_step=step_)
        # print(translate(ge, model, ger_train, eng_train, device))
        # print(en)
    model.train()
    losses = []
    for batch_idx, batch in enumerate(train_loader):
        # Get input and targets and get to cuda
        inp_data = batch["inp"].to(device)
        target = batch["tar"].to(device)
        desired = batch["des"].to(device)
        # print(inp_data.shape,target.shape)
        # print(inp_data[0],target[0],desired[0])
        # Forward prop
        output = model(inp_data, target)
        # print(output.shape, target[:, 1:].shape)
        # Output is of shape (trg_len, batch_size, output_dim) but Cross Entropy Loss
        # doesn't take input in that form. For example if we have MNIST we want to have
        # output to be: (N, 10) and targets just (N). Here we can view it in a similar
        # way that we have output_words * batch_size that we want to send in into
        # our cost function, so we need to do some reshapin.
        # Let's also remove the start token while we're at it
        with torch.no_grad():
            output = output.reshape(-1, output.shape[2])
            desired = desired.reshape(-1)

        # print(target.shape, output.shape)
        optimizer.zero_grad()

        loss = criterion(output, desired)
        losses.append(loss.item())

        # Back prop
        loss.backward()
        # Clip to avoid exploding gradient issues, makes sure grads are
        # within a healthy range
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)

        # Gradient descent step
        optimizer.step()

        # plot to tensorboard
        writer.add_scalar("Training loss", loss, global_step=step)
        step += 1
        writer.add_scalar("num_elements", batch_size * batch_idx, global_step=step)
    with torch.no_grad():
        print(losses)
        mean_loss = sum(losses) / len(losses)
        writer.add_scalar("loss", mean_loss, global_step=step_)
        step_ = step_ + 1
    scheduler.step(mean_loss)
    if save_model:
        checkpoint = {
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
        }
        save_checkpoint(checkpoint, "checkpoints/my_checkpoint{}.pth.tar".format(epoch))
