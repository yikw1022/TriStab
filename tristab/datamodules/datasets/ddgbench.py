
import torch
from torch.utils.data import ConcatDataset
import pandas as pd
import numpy as np
import pickle
import os
from Bio import pairwise2
from math import isnan
from tqdm import tqdm
from dataclasses import dataclass
from typing import Optional
from .utils import fermi_transform,tied_featurize,get_pdb,parse_pdb,alt_parse_PDB
import lmdb
import glob
from tristab import utils
log = utils.get_logger(__name__)
import json
from .batch import CoordBatchConverter
from .data_utils import Alphabet
from collections import defaultdict
ALPAHBET = 'ACDEFGHIKLMNPQRSTVWYX'

'''
Adapted from SPURS
https://github.com/luo-group/SPURS/blob/9cf686eb8304740775c4cfdd2437d36a96e97732/spurs/datamodules/datasets/ddgbench.py
'''


class ddgBenchDataset(torch.utils.data.Dataset):

    def __init__(self, pdb_dir, csv_fname, dataset_name):

        self.pdb_dir = pdb_dir
        df = pd.read_csv(csv_fname)
        self.df = df
        self.dataset_name = dataset_name

        self.wt_seqs = {}
        self.mut_rows = {}
        self.wt_names = df.PDB.unique()

        for wt_name in self.wt_names:
            wt_name_query = wt_name
            wt_name = wt_name[:-1]
            self.mut_rows[wt_name] = df.query('PDB == @wt_name_query').reset_index(drop=True)
            # if 'S669' not in self.pdb_dir:
            #     self.wt_seqs[wt_name] = self.mut_rows[wt_name].SEQ[0]
            

    
        self.structure_path = pdb_dir

        self.json_dataset = defaultdict(lambda: defaultdict(lambda: -1))
   
    def __len__(self):
        return len(self.wt_names)

    def __getitem__(self, index):
        """Batch retrieval fxn - each batch is a single protein"""

        wt_name = self.wt_names[index]
        chain = [wt_name[-1]]

        wt_name = wt_name.split(".pdb")[0][:-1]
        mut_data = self.mut_rows[wt_name]


        # modified PDB parser returns list of residue IDs so we can align them easier
  
        if isinstance(self.json_dataset[wt_name][chain[0]],int):
            pdb = alt_parse_PDB(os.path.join(self.pdb_dir,wt_name+".pdb"),chain)
            self.json_dataset[wt_name][chain[0]] = pdb
        pdb = self.json_dataset[wt_name][chain[0]]
        resn_list = pdb[0]["resn_list"]


        protein = get_pdb(pdb[0], wt_name, wt_name, check_assert=False)
        
        protein['mut_seq'] = []

        # --- landscape GT 构建 ---
        AA20 = 'ACDEFGHIKLMNPQRSTVWY'
        L = len(pdb[0]['seq'])
        landscape = torch.zeros(L, 20, dtype=torch.float32)
        landscape_mask = torch.zeros(L, 20, dtype=torch.float32)
        obs_mut_ids_list = []
        obs_mt_aa_ids_list = []

        wt_seq_aa_ids = []
        for aa in pdb[0]['seq']:
            if aa in AA20:
                wt_seq_aa_ids.append(AA20.index(aa))
            else:
                wt_seq_aa_ids.append(0)
        wt_seq_aa_ids = torch.tensor(wt_seq_aa_ids, dtype=torch.long)

        for i, row in mut_data.iterrows():
            mut_info = row.MUT
            wtAA, mutAA = mut_info[0], mut_info[-1]
            try:
                pos = mut_info[1:-1]
                pdb_idx = resn_list.index(pos)
            except ValueError:  # skip positions with insertion codes for now - hard to parse

                continue
            try:
                assert pdb[0]['seq'][pdb_idx] == wtAA
            except AssertionError:  # contingency for mis-alignments
                # if gaps are present, add these to idx (+10 to get any around the mutation site, kinda a hack)
                if 'S669' in self.pdb_dir:
                    gaps = [g for g in pdb[0]['seq'] if g == '-']
                else:
                    gaps = [g for g in pdb[0]['seq'][:pdb_idx + 10] if g == '-']                

                if len(gaps) > 0:
                    pdb_idx += len(gaps)
                else:
                    pdb_idx += 1

                if pdb_idx is None:
                    continue
                assert pdb[0]['seq'][pdb_idx] == wtAA
                

            
            wt = wtAA
            mut = mutAA
            
            ddG = None if row.DDG is None or isnan(row.DDG) else torch.tensor([row.DDG * -1.], dtype=torch.float32)
            wt_onehot = torch.zeros((21))
            wt_onehot[ALPAHBET.index(wt)] = 1
            mt_onehot = torch.zeros((21))
            mt_onehot[ALPAHBET.index(mut)] = 1
            append_tensor = torch.cat([wt_onehot,mt_onehot])
            append_tensor = append_tensor.float()

            protein['mut_ids'].append(pdb_idx)
            protein['ddG'].append(ddG)
            protein['append_tensors'].append(append_tensor)

            mut_sequence = list(pdb[0]['seq'])  # 复制原序列
            mut_sequence[pdb_idx] = mutAA  # 在突变位点替换氨基酸
            protein['mut_seq'].append(''.join(mut_sequence))  # 添加完整的突变序列

            # landscape GT 填充
            if ddG is not None and mut in AA20 and pdb_idx < L:
                aa_idx = AA20.index(mut)
                landscape[pdb_idx, aa_idx] = ddG.item()
                landscape_mask[pdb_idx, aa_idx] = 1.0
                obs_mut_ids_list.append(pdb_idx)
                obs_mt_aa_ids_list.append(aa_idx)

        protein['ddG'] = torch.stack(protein['ddG'])
        protein['append_tensors'] = torch.stack(protein['append_tensors'])
        protein['pdb_path'] = self.structure_path
        protein['dataset'] = self.dataset_name

        # landscape 新字段
        protein['landscape'] = landscape
        protein['landscape_mask'] = landscape_mask
        protein['wt_seq_aa_ids'] = wt_seq_aa_ids
        protein['obs_mut_ids'] = torch.tensor(obs_mut_ids_list, dtype=torch.long)
        protein['obs_mt_aa_ids'] = torch.tensor(obs_mt_aa_ids_list, dtype=torch.long)

        return protein
