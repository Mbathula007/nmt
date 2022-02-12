from abc import ABC

import torch
from torch.utils.data import Dataset, IterableDataset
from tokenize_custom import tokenize_en, tokenize_ge, remove_punc
from itertools import cycle


def collate_fn(data):
    def pad_(sentence, max_len):
        return torch.cat((sentence, torch.zeros([max_len - sentence.shape[0]], dtype=torch.long)), dim=0).unsqueeze(0)

    out_data = {}
    max_len_inp = max(data, key=lambda x: x["inp"].shape[0])["inp"].shape[0]
    max_len_tar = max(data, key=lambda x: x["tar"].shape[0])["tar"].shape[0]
    data_ = sorted(data, key=lambda x: x["inp"].shape[0])
    for c in data_:
        inp_sen = c["inp"]
        tar_sen = c["tar"]
        des_sen = c["des"]
        if "inp" in out_data:
            out_data["inp"] = torch.cat((out_data["inp"], pad_(inp_sen, max_len_inp)), dim=0)
        else:
            out_data["inp"] = pad_(inp_sen, max_len_inp)
        if "tar" in out_data:
            out_data["tar"] = torch.cat((out_data["tar"], pad_(tar_sen, max_len_tar)), dim=0)
        else:
            out_data["tar"] = pad_(tar_sen, max_len_tar)
        if "des" in out_data:
            out_data["des"] = torch.cat((out_data["des"], pad_(des_sen, max_len_tar)), dim=0)
        else:
            out_data["des"] = pad_(des_sen, max_len_tar)
    return out_data


class map_dataset(Dataset):
    def __init__(self, data_eng, data_ger, max_len=20):
        super().__init__()
        self.eng = data_eng
        self.ger = data_ger
        self.max_len = max_len

    def __len__(self):
        return len(self.eng)

    def __getitem__(self, idx):
        return {"inp": torch.LongTensor(self.ger[idx]),
                "tar": torch.LongTensor(self.eng[idx]),
                "des": torch.LongTensor(self.eng[idx][1:])}


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


class map_dataset_test(Dataset):
    def __init__(self, inp_file, out_file):
        self.in_ = remove_punc(open(inp_file).readlines())
        self.ou_ = remove_punc(open(out_file).readlines())

    def __len__(self):
        return len(self.in_)

    def __getitem__(self, item):
        return {"inp": tokenize_ge(self.in_[item].lower()), "tar": tokenize_en(self.ou_[item].lower())}
