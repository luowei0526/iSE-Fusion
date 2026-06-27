import ast
import torch
import torch.utils.data as Data
from Biodata import Biodata
from transformers import AutoTokenizer
import sys

tokenizer = AutoTokenizer.from_pretrained("/mnt/data/lly/LW/multi_se/DNABERT2_Model", trust_remote_code=True, local_files_only=True)


class MyDataSet(Data.Dataset):
    def __init__(self, seq, MGW, EP, ProT, Roll, HelT, label):
        self.seq = seq  
        self.MGW = MGW
        self.EP = EP
        self.ProT = ProT
        self.Roll = Roll
        self.HelT = HelT        
        self.label = label  
        self.tokenizer = tokenizer
        self.max_len = 3000 

    def __getitem__(self, idx):
        seq = self.seq[idx]

        graph = Biodata(seq).encode()

        inputs = self.tokenizer(seq, return_tensors="pt", padding='max_length', max_length=750, truncation=True)
        input_ids = inputs.input_ids.squeeze(0)
        token_type_ids = inputs.token_type_ids.squeeze(0)
        attention_mask = inputs.attention_mask.squeeze(0)

        MGW = torch.tensor(ast.literal_eval(self.MGW[idx]), dtype=torch.float32).unsqueeze(1)
        EP = torch.tensor(ast.literal_eval(self.EP[idx]), dtype=torch.float32).unsqueeze(1)
        ProT = torch.tensor(ast.literal_eval(self.ProT[idx]), dtype=torch.float32).unsqueeze(1)
        Roll = torch.tensor(ast.literal_eval(self.Roll[idx]), dtype=torch.float32).unsqueeze(1)
        HelT = torch.tensor(ast.literal_eval(self.HelT[idx]), dtype=torch.float32).unsqueeze(1)

        shape_features = torch.cat([MGW, EP, ProT, Roll, HelT], dim=1)
        seq_len = shape_features.size(0)

        label = self.label[idx]

        if seq_len < self.max_len:
            pad_len = self.max_len - seq_len
            padding = torch.zeros(pad_len, 5)
            shape_features = torch.cat([shape_features, padding], dim=0)
        else:
            shape_features = shape_features[:self.max_len]

        return graph.x_src, graph.x_dst, graph.edge_index, input_ids, token_type_ids, attention_mask, shape_features, label

    def __len__(self):
        return len(self.seq)
