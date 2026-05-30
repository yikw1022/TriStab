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
from .utils import get_pdb,parse_pdb
import lmdb
import glob
from tristab import utils
log = utils.get_logger(__name__)
import json
from .batch import CoordBatchConverter
from .data_utils import Alphabet
from .fireprot import FireProtDataset
from .ddgbench import ddgBenchDataset
from .ddggeo import ddgGeo
ALPAHBET = 'ACDEFGHIKLMNPQRSTVWYX'
from joblib import Parallel, delayed
from collections import defaultdict
import math
import random
            
'''
Adapted from SPURS
https://github.com/luo-group/SPURS/blob/9cf686eb8304740775c4cfdd2437d36a96e97732/spurs/datamodules/datasets/megascale.py

'''

class MegaScaleDataset(torch.utils.data.Dataset):

    def __init__(self, 
                 reduce: str = '',
                 split: str = 'train',
                 single_mut: bool = False,
                 mut_seq: bool=False,
                 std_ratio: float=0.75,
                 loss_ratio: float=1.,
                 train_ratio: float=0.05,
                 ):
        self.mut_seq = mut_seq
        self.split = split
        current_dir = os.path.dirname(os.path.abspath(__file__))
        root_path = os.path.join(current_dir,'../../../')
        fname = os.path.join(root_path,'data/dataset/megascale/Tsuboyama2023_Dataset2_Dataset3_20230416.csv')
        df = pd.read_csv(fname, usecols=["ddG_ML", "mut_type", "WT_name", "aa_seq", "dG_ML"])
        # remove unreliable data and more complicated mutations
        df = df.loc[df.ddG_ML != '-', :].reset_index(drop=True)
        df = df.loc[~df.mut_type.str.contains("ins") & ~df.mut_type.str.contains("del") & ~df.mut_type.str.contains(":"), :].reset_index(drop=True)

        self.df = df
        if  self.split!='test':
            mmseq_wt_search = os.path.join(root_path,'data/dataset/megascale/mmseq_mut_search_0.25.m8')
            ret = []
            with open(mmseq_wt_search, 'r') as f:
                for line in f.readlines():
                    second_column_value = int(line.split("\t")[1])
                    ret.append(second_column_value)
            # we dont want the rows in the ret
            previous_len = len(df)
            df = df.loc[~df.index.isin(ret), :].reset_index(drop=True)
            cur_len = len(df)
            log.info(f"removed {previous_len - cur_len} rows from the dataset")
            
        # load splits produced by mmseqs clustering
        with open( os.path.join(root_path,'data/dataset/megascale/mega_splits.pkl'), 'rb') as f:
            splits = pickle.load(f)  # this is a dict with keys train/val/test and items holding FULL PDB names for a given split
        
        self.split_wt_names = {
            "val": [],
            "test": [],
            "train": [],
            "train_s669": [],
            "all": [], 
            "cv_train_0": [],
            "cv_train_1": [],
            "cv_train_2": [],
            "cv_train_3": [],
            "cv_train_4": [],
            "cv_val_0": [],
            "cv_val_1": [],
            "cv_val_2": [],
            "cv_val_3": [],
            "cv_val_4": [],
            "cv_test_0": [],
            "cv_test_1": [],
            "cv_test_2": [],
            "cv_test_3": [],
            "cv_test_4": [],
        }

        self.wt_seqs = {}
        self.mut_rows = {}
        self.std_ratio = std_ratio
        self.loss_ratio = loss_ratio
        self.train_ratio = train_ratio
        if self.split == 'all':
            all_names = np.concatenate([splits['train'],splits['val'],splits['test']])
            self.split_wt_names[self.split] = all_names
        else:
            if reduce == 'prot' and split == 'train':
                n_prots_reduced = 58
                self.split_wt_names[self.split] = np.random.choice(splits["train"], n_prots_reduced)
            else:
                self.split_wt_names[self.split] = splits[self.split]
        self.wt_names = self.split_wt_names[self.split]
        
        removed_wt_names = []
        for wt_name in tqdm(self.wt_names):
            wt_rows = df.query('WT_name == @wt_name and mut_type == "wt"').reset_index(drop=True)
            self.mut_rows[wt_name] = df.query('WT_name == @wt_name and mut_type != "wt"').reset_index(drop=True)
            if type(reduce) is float and self.split == 'train':
                self.mut_rows[wt_name] = self.mut_rows[wt_name].sample(frac=float(reduce), replace=False)
            if len(wt_rows) == 0:
                # log.info(f'remove {wt_name}')
                removed_wt_names.append(wt_name)
            else:
                self.wt_seqs[wt_name] = wt_rows.aa_seq[0]
        previous_len = len(self.wt_names)
        self.wt_names = list(set(self.wt_names) - set(removed_wt_names))
        cur_len = len(self.wt_names)
        log.info(f"removed {previous_len - cur_len} wt names from the dataset")
            
            
        structure_path = os.path.join(root_path,'data/dataset/megascale/AlphaFold_model_PDBs/')
        structure_path_json = os.path.join(structure_path,"../parsed_structure.json")
        
        self.structure_path = structure_path
        
        parse_pdb(structure_path,structure_path_json)
                
        log.info("loading structure dataset")
        with open(structure_path_json, 'r') as file:
            self.json_dataset = json.load(file)
            
    def cal_index2mt(self, index):
        return self.index2mt[index]

    def __len__(self):

        return len(self.wt_names)   


    def _get_wt_item(self, index):

        # wt_name, mut_seq, wt_seq = self.cal_index2mt(index)
        
        wt_name = self.wt_names[index]
        wt_seq = self.wt_seqs[wt_name]
        mut_data = self.mut_rows[wt_name]

        wt_name = wt_name.split(".pdb")[0].replace("|",":")

        pdb = self.json_dataset[wt_name]
        protein = get_pdb(pdb,wt_seq,wt_name)
        
        if  self.mut_seq:
            protein['S'] = [protein['S']]
        
        # --- landscape GT 构建 ---
        AA20 = 'ACDEFGHIKLMNPQRSTVWY'
        L = len(wt_seq)
        landscape = torch.zeros(L, 20, dtype=torch.float32)
        landscape_mask = torch.zeros(L, 20, dtype=torch.float32)
        obs_mut_ids_list = []
        obs_mt_aa_ids_list = []

        # wt_seq_aa_ids: 整条 WT 序列每个位点的 AA20 index
        wt_seq_aa_ids = []
        for aa in wt_seq:
            if aa in AA20:
                wt_seq_aa_ids.append(AA20.index(aa))
            else:
                wt_seq_aa_ids.append(0)  # 非标准氨基酸映射到 0
        wt_seq_aa_ids = torch.tensor(wt_seq_aa_ids, dtype=torch.long)

        dataset_name = []
        for i in range(len(mut_data)):
            mut_seq = mut_data.iloc[i]

            if  self.mut_seq:
                pdb["seq"] = mut_seq.aa_seq
                mut_protein = get_pdb(pdb,mut_seq.aa_seq,wt_name)
                protein['S'].append(mut_protein['S'])
            
            if "ins" in mut_seq.mut_type or "del" in mut_seq.mut_type or ":" in mut_seq.mut_type:
                return None
            
            assert len(mut_seq.aa_seq) == len(wt_seq)
            wt = mut_seq.mut_type[0]
            mut = mut_seq.mut_type[-1]
            mut_id = int(mut_seq.mut_type[1:-1]) - 1
            assert wt_seq[mut_id] == wt
            assert mut_seq.aa_seq[mut_id] == mut
            
            if mut_seq.ddG_ML == '-':
                return None
            ddG = -torch.tensor([float(mut_seq.ddG_ML)], dtype=torch.float32)
            
            wt_onehot = torch.zeros((21))
            wt_onehot[ALPAHBET.index(wt)] = 1
            mt_onehot = torch.zeros((21))
            mt_onehot[ALPAHBET.index(mut)] = 1
            append_tensor = torch.cat([wt_onehot,mt_onehot])
            append_tensor = append_tensor.float()
            # split = fing_split_in_proteingym(mut_seq.aa_seq)
            
            
            protein['mut_ids'].append(mut_id)
            protein['ddG'].append(ddG)
            protein['append_tensors'].append(append_tensor)
            protein['mut_seq'].append(mut_seq.aa_seq)
            # dataset_name.append('megascale'+str(split))

            # landscape GT 填充
            if mut in AA20:
                aa_idx = AA20.index(mut)
                landscape[mut_id, aa_idx] = ddG.item()
                landscape_mask[mut_id, aa_idx] = 1.0
                obs_mut_ids_list.append(mut_id)
                obs_mt_aa_ids_list.append(aa_idx)

        protein['ddG'] = torch.stack(protein['ddG']).to(protein['X'].device,non_blocking=True)
        protein['append_tensors'] = torch.stack(protein['append_tensors'])
        # protein['dataset'] = dataset_name
        protein['dataset'] = 'megascale'
        protein['pdb_path'] = self.structure_path

        # landscape 新字段
        protein['landscape'] = landscape
        protein['landscape_mask'] = landscape_mask
        protein['wt_seq_aa_ids'] = wt_seq_aa_ids
        protein['obs_mut_ids'] = torch.tensor(obs_mut_ids_list, dtype=torch.long)
        protein['obs_mt_aa_ids'] = torch.tensor(obs_mt_aa_ids_list, dtype=torch.long)
        
        # print(protein['X'].shape,protein['mask'].shape,protein['chain_M'].shape,protein['chain_M_chain_M_pos'].shape,protein['residue_idx'].shape,protein['chain_encoding_all'].shape,protein['randn_1'].shape)
        if  self.mut_seq:
            protein['S'] = torch.cat(protein['S'],dim=0).clone()
            protein['X'] = protein['X'].expand(len(protein['S']),-1,-1,-1).clone()
            protein['mask'] = protein['mask'].expand(len(protein['S']),-1).clone()
            protein['chain_M'] = protein['chain_M'].expand(len(protein['S']),-1).clone()
            protein['chain_M_chain_M_pos'] = protein['chain_M_chain_M_pos'].expand(len(protein['S']),-1).clone()
            protein['residue_idx'] = protein['residue_idx'].expand(len(protein['S']),-1).clone()
            protein['chain_encoding_all'] = protein['chain_encoding_all'].expand(len(protein['S']),-1).clone()
            protein['randn_1'] = protein['randn_1'].expand(len(protein['S']),-1).clone()

        protein['std_ratio'] = self.std_ratio
        protein['loss_ratio'] = self.loss_ratio
        return protein

    def __getitem__(self, index):

        return self._get_wt_item(index)



    def collect_func(self, batch):
        return batch[0]
    
class Featurizer(object):
    def __init__(self, alphabet: Alphabet, 
                 to_pifold_format=False, 
                 coord_nan_to_zero=True,
                 atoms=('N', 'CA', 'C', 'O'),
                 single_mut = False,
                 mut_seq= False
                 ):
        self.alphabet = alphabet
        self.batcher = CoordBatchConverter(
            alphabet=alphabet,
            coord_pad_inf=alphabet.add_special_tokens,
            to_pifold_format=to_pifold_format, 
            coord_nan_to_zero=coord_nan_to_zero
        )
        self.single_mut = single_mut
        self.atoms = atoms
        self.cache = defaultdict(lambda: -1)
        self.mut_seq = mut_seq


    def __call__(self, raw_batch: dict):
        
        if not self.single_mut:
            
            # raw_batch = raw_batch[0]
            # if not self.mut_seq:
            #     seqs = [raw_batch['seq']]
            #     coords = [np.stack([raw_batch['coords'][atom] for atom in self.atoms], 1)]
            # else:
            #     seqs = [raw_batch['seq']]+raw_batch['mut_seq']
            #     coords = [np.stack([raw_batch['coords'][atom] for atom in self.atoms], 1)]*len(seqs)
            # coords, confidence, strs, tokens, lengths, coord_mask = self.batcher.from_lists(
            #     coords_list=coords, confidence_list=None, seq_list=seqs
            # ) 

            # if not self.mut_seq:
            #     raw_batch['tokens'] = tokens
            #     raw_batch['mut_tokens'] = None 
            # else:
            #     raw_batch['tokens'] = tokens
            #     raw_batch['mut_tokens'] = None
            
            # if True:
            #     ddg = raw_batch['ddG']
            #     raw_batch['ddG'] = ddg
            # raw_batch['ddG'] = raw_batch['ddG'].reshape(-1)
            
            raw_batch = raw_batch[0]
            wt_seq = [raw_batch['seq']]
            wt_coords = [np.stack([raw_batch['coords'][atom] for atom in self.atoms], 1)]
            
            coords, confidence, strs, tokens, lengths, coord_mask = self.batcher.from_lists(
                coords_list=wt_coords, confidence_list=None, seq_list=wt_seq
            )
            # 不再为每个 mut_seq 单独构造 mut_tokens（新模型不需要）
            # 旧模型兼容：mut_tokens 设为空列表
            raw_batch['tokens'] = tokens
            raw_batch['mut_tokens'] = []
            raw_batch['ddG'] = raw_batch['ddG'].reshape(-1)
            return raw_batch
            

        for protein in raw_batch:
            name = protein['name']
            if isinstance(self.cache[name], int):
                seqs = [protein['seq']]
                coords = [np.stack([protein['coords'][atom] for atom in self.atoms], 1)]
                
                coords, confidence, strs, tokens, lengths, coord_mask = self.batcher.from_lists(
                    coords_list=coords, confidence_list=None, seq_list=seqs
                )
                self.cache[name] = {
                    'tokens': tokens,
                    'mut_tokens': None
                }
            else:
                tokens = self.cache[name]['tokens']
                mut_tokens = self.cache[name]['mut_tokens']
            protein['tokens'] = tokens
            protein['mut_tokens'] = None  
        
        ddg = torch.stack([protein['ddG'] for protein in raw_batch])
        return {
            'raw_batch': raw_batch,
            'mut_ids': [protein['mut_ids'] for protein in raw_batch],
            'append_tensors' : torch.stack([protein['append_tensors'] for protein in raw_batch]),
            'ddG': ddg,
            'name': [protein['name']+protein['chain_ids'] for protein in raw_batch],
            'dataset': [protein['dataset'] for protein in raw_batch],
        }
    
class MegaScaleTestDatasets(torch.utils.data.Dataset):
    def __init__(self):
        current_dir = os.path.dirname(os.path.abspath(__file__))
        root_path = os.path.join(current_dir, '../../../')

        # ---- 10 个测试数据集 ----
        self.megascale = MegaScaleDataset(
            reduce='',
            split='test',
        )

        self.fireport = FireProtDataset(split='homologue-free')

        self.p53 = ddgBenchDataset(
            pdb_dir=os.path.join(root_path, 'data/dataset/p53/pdb'),
            csv_fname=os.path.join(root_path, 'data/dataset/p53/p53.csv'),
            dataset_name='p53',
        )

        self.s669 = ddgBenchDataset(
            pdb_dir=os.path.join(root_path, 'data/dataset/S669/pdb'),
            csv_fname=os.path.join(root_path, 'data/dataset/S669/s669_clean_dir.csv'),
            dataset_name='S669')

        self.S461 = ddgGeo(
            pdb_dir=os.path.join(root_path, 'data/dataset/geostab_data/ddG_cleaned/S461'),
            csv_fname=os.path.join(root_path, 'data/dataset/geostab_data/ddG_cleaned/S461.csv'),
            dataset_name='S461'
        )

        self.S783 = ddgGeo(
            pdb_dir=os.path.join(root_path, 'data/dataset/geostab_data/ddG_cleaned/S783'),
            csv_fname=os.path.join(root_path, 'data/dataset/geostab_data/ddG_cleaned/S783.csv'),
            dataset_name='S783'
        )

        self.S8754 = ddgGeo(
            pdb_dir=os.path.join(root_path, 'data/dataset/geostab_data/ddG_cleaned/S8754'),
            csv_fname=os.path.join(root_path, 'data/dataset/geostab_data/ddG_cleaned/S8754.csv'),
            dataset_name='S8754'
        )

        self.S2648 = ddgGeo(
            pdb_dir=os.path.join(root_path, 'data/dataset/geostab_data/ddG_cleaned/S2648'),
            csv_fname=os.path.join(root_path, 'data/dataset/geostab_data/ddG_cleaned/S2648.csv'),
            dataset_name='S2648',
            stage='test'
        )

        self.S571 = ddgGeo(
            pdb_dir=os.path.join(root_path, 'data/dataset/geostab_data/dTm_cleaned/S571'),
            csv_fname=os.path.join(root_path, 'data/dataset/geostab_data/dTm_cleaned/S571.csv'),
            dataset_name='S571'
        )

        self.S4346 = ddgGeo(
            pdb_dir=os.path.join(root_path, 'data/dataset/geostab_data/dTm_cleaned/S4346'),
            csv_fname=os.path.join(root_path, 'data/dataset/geostab_data/dTm_cleaned/S4346.csv'),
            dataset_name='S4346'
        )

        # 按顺序组成列表，方便 __len__ / __getitem__ 统一处理
        self._datasets = [
            self.megascale, self.p53, self.S461, self.S783,
            self.S2648, self.fireport, self.S8754, self.S4346,
            self.s669, self.S571,
        ]

    def __len__(self):
        return sum(len(d) for d in self._datasets)

    def __getitem__(self, index):
        for d in self._datasets:
            if index < len(d):
                return d[index]
            index -= len(d)
        raise IndexError(f"index {index} out of range for MegaScaleTestDatasets")
