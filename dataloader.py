from abc import ABC

import torch
from torch.utils.data import Dataset, IterableDataset
from tokenize_custom import tokenize_en, tokenize_ge
from itertools import cycle


class map_dataset(Dataset):
    def __init__(self, data_eng, data_ger, max_len=20):
        super().__init__()
        self.eng = data_eng
        self.ger = data_ger
        self.max_len = max_len

    def __len__(self):
        return len(self.eng)

    def __getitem__(self, idx):
        return {"inp": torch.LongTensor(self.ger[idx] + ([0] * (self.max_len - len(self.ger[idx])))),
                "tar": torch.LongTensor(self.eng[idx][:-1] + [0] * (self.max_len + 1 - len(self.eng[idx]))),
                "des": torch.LongTensor(self.eng[idx][1:] + [0] * (self.max_len + 1 - len(self.eng[idx])))}


class iter_dataset(IterableDataset, ABC):
    def __init__(self, input_file_name, output_file_name, eng, ger, max_len=20):
        super().__init__()
        self.input_ = input_file_name
        self.output_ = output_file_name
        self.eng = eng
        self.ger = ger
        self.max_len = max_len
        self.input_tok = tokenize_ge
        self.output_tok = tokenize_en

    def parse_line(self, line, lang, tokenizer):
        line_out = tokenizer(line)
        out_ = [lang.stoi["<sos>"]]
        for token in line_out:
            if token not in lang.stoi:
                token = "<unk>"
            out_.append(lang.stoi[token])
        # out_.append(lang.stoi["<eos>"])
        return out_ + ([0] * (self.max_len - len(out_)))

    def parse_file(self):
        with open(self.input_, 'r') as ob_in, open(self.output_, 'r') as ob_out:
            while True:
                line_in = ob_in.readline()
                line_out = ob_out.readline()
                if line_in is None or line_out is None:
                    break
                input_line = self.parse_line(line_in, self.ger, self.input_tok)
                output_line = self.parse_line(line_out, self.eng, self.output_tok)
                yield {"inp": torch.LongTensor(input_line), "tar": torch.LongTensor(output_line)}

    def get_stream(self):
        return self.parse_file()

    def __iter__(self):
        return self.get_stream()
