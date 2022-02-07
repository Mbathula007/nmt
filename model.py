import torch
from torch import nn
from tokenize_custom import german_tok


class Transformer(nn.Module):
    def __init__(self, embedding_size,
                 src_vocab_size,
                 trg_vocab_size,
                 src_pad_idx,
                 num_heads,
                 num_encoder_layers,
                 num_decoder_layers,
                 forward_expansion,
                 dropout,
                 max_len,
                 device):
        super(Transformer, self).__init__()
        self.src_vocab_si = src_vocab_size
        self.tgt_vocab_si = trg_vocab_size
        self.src_word_embedding = nn.Embedding(src_vocab_size, embedding_size)
        self.src_pos_embedding = nn.Embedding(max_len, embedding_size)
        self.trg_word_embedding = nn.Embedding(trg_vocab_size, embedding_size)
        self.trg_pos_embedding = nn.Embedding(max_len, embedding_size)
        self.device = device
        self.transformer = nn.Transformer(
            embedding_size,
            num_heads,
            num_encoder_layers,
            num_decoder_layers,
            forward_expansion,
            dropout,
            batch_first=True
        )
        self.fc_out = nn.Linear(embedding_size, trg_vocab_size)
        self.dropout = nn.Dropout(dropout)
        self.pad_idx = src_pad_idx

    def make_src_mask(self, src):
        src_mask = src == self.pad_idx
        return src_mask

    def forward(self, src, trg):
        with torch.no_grad():
            # print("trg shape is ", trg.shape, src.shape)
            # print(torch.max(trg),torch.min(trg))
            N, src_seq_len, = src.shape
            N, trg_seq_len = trg.shape
            # print(N,)
            src_positions = (
                torch.arange(0, src_seq_len).unsqueeze(0).expand(N, src_seq_len).to(self.device)
            )
            # print("src_pos shape is ", src_positions.shape)
            trg_positions = (
                torch.arange(0, trg_seq_len).unsqueeze(0).expand(N, trg_seq_len).to(self.device)
            )
            # print("trg_pos shape is ", trg_positions.shape)
            src_padding_mask = self.make_src_mask(src).to(self.device)
            trg_mask = self.transformer.generate_square_subsequent_mask(trg_seq_len).to(self.device)
        embed_src = self.dropout(
            (self.src_word_embedding(src) + self.src_pos_embedding(src_positions))
        )
        embed_trg = self.dropout(
            (self.trg_word_embedding(trg) + self.trg_pos_embedding(trg_positions))
        )

        out = self.transformer(embed_src, embed_trg,
                               src_key_padding_mask=src_padding_mask,
                               tgt_mask=trg_mask)
        out = self.fc_out(out)
        return out


def translate(sentence, model, input_lang, output_lang, device, max_len=20):
    # print(list(input_lang.stoi.keys())[:5])
    tokens = [str(tok) for tok in german_tok(sentence.lower())]
    # print(tokens)
    # print(tokens)
    input_ = [input_lang.stoi["<sos>"]]
    for token in tokens:
        input_.append(input_lang.stoi[str(token)])
    input_.append(input_lang.stoi["<eos>"])
    input_tensor = torch.LongTensor(input_).unsqueeze(0).to(device)
    output_ = [output_lang.stoi["<sos>"]]
    output_tensor = torch.LongTensor(output_).unsqueeze(0).to(device)
    eos_output = output_lang.stoi["<eos>"]
    # print(input_tensor.shape)
    # print(output_tensor.shape)
    # print(output_tensor[:, -1].shape)
    while True:
        # print(output_tensor.shape)
        # print(output_tensor[:, -1].shape)
        if output_tensor.shape[1] > max_len:
            break
        if output_tensor[:, -1].item() == eos_output:
            break
        out_ = model(input_tensor, output_tensor)
        guess = out_.argmax(-1)[-1, :].unsqueeze(0)
        guess = guess[:, -1].unsqueeze(0)
        output_tensor = torch.cat([output_tensor, guess], dim=-1)
        # print("output shape is ",output_tensor.shape)
    translated_sentence = ""
    output_ = output_tensor.squeeze().tolist()
    for token in output_:
        translated_sentence = translated_sentence + str(output_lang.itos[token]) + str(" ")
    return translated_sentence


def save_checkpoint(state, filename="my_checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    torch.save(state, filename)
