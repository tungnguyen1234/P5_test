from torch.utils.data import DataLoader, Dataset, Sampler
from pathlib import Path
from collections import defaultdict
import json
import gzip
import random
from multiprocessing import Pool
import pickle
import math
from tqdm import tqdm
import torch
import numpy as np
import os
from torch.utils.data.distributed import DistributedSampler
from copy import deepcopy

from transformers import T5Tokenizer, T5TokenizerFast
from tokenization import P5Tokenizer, P5TokenizerFast

def load_json(file_path):
    with open(file_path, "r") as f:
        return json.load(f)

def load_pickle(filename):
    with open(filename, "rb") as f:
        return pickle.load(f)

def ReadLineFromFile(path):
    lines = []
    with open(path,'r') as fd:
        for line in fd:
            lines.append(line.rstrip('\n'))
    return lines

def parse(path):
    g = gzip.open(path, 'r')
    for l in g:
        yield eval(l)

    
class P5_Amazon_Dataset(Dataset):
    def __init__(self, all_tasks, task_list, tokenizer, args, sample_numbers, mode='train', split='toys', rating_augment=False, sample_type='random'): 
        self.all_tasks = all_tasks
        self.task_list = task_list
        self.tokenizer = tokenizer
        self.args = args
        self.sample_numbers = sample_numbers
        self.split = split
        self.rating_augment = rating_augment
        self.sample_type = sample_type
        
        print('Data sources: ', split.split(','))
        self.mode = mode
        if self.mode == 'train':
            self.review_data = load_pickle(os.path.join('data', split, 'review_splits.pkl'))['train']
            self.exp_data = load_pickle(os.path.join('data', split, 'exp_splits.pkl'))['train']
            if self.rating_augment:
                self.rating_data = load_pickle(os.path.join('data', split, 'rating_splits_augmented.pkl'))['train']
            else:
                self.rating_data = self.review_data
        elif self.mode == 'val':
            self.review_data = load_pickle(os.path.join('data', split, 'review_splits.pkl'))['val']
            self.exp_data = load_pickle(os.path.join('data', split, 'exp_splits.pkl'))['val']
            if self.rating_augment:
                self.rating_data = load_pickle(os.path.join('data', split, 'rating_splits_augmented.pkl'))['val']
            else:
                self.rating_data = self.review_data
        elif self.mode == 'test':
            self.review_data = load_pickle(os.path.join('data', split, 'review_splits.pkl'))['test']
            self.exp_data = load_pickle(os.path.join('data', split, 'exp_splits.pkl'))['test']
            if self.rating_augment:
                self.rating_data = load_pickle(os.path.join('data', split, 'rating_splits_augmented.pkl'))['test']
            else:
                self.rating_data = self.review_data
            self.zeroshot_exp_data = load_pickle(os.path.join('data', 'beauty', 'zeroshot_exp_splits.pkl')) # change to dataset to be transferred (e.g., 'beauty')
        else:
            raise NotImplementedError
            
        self.sequential_data = ReadLineFromFile(os.path.join('data', split, 'sequential_data.txt'))
        item_count = defaultdict(int)
        user_items = defaultdict()

        for line in self.sequential_data:
            user, items = line.strip().split(' ', 1)
            items = items.split(' ')
            items = [int(item) for item in items]
            user_items[user] = items
            for item in items:
                item_count[item] += 1
                
        self.all_item = list(item_count.keys())
        count = list(item_count.values())
        sum_value = np.sum([x for x in count])
        self.probability = [value / sum_value for value in count]
        self.user_items = user_items
        
        if self.mode == 'test':
            self.negative_samples = ReadLineFromFile(os.path.join('data', split, 'negative_samples.txt'))
            
        datamaps = load_json(os.path.join('data', split, 'datamaps.json'))
        self.user2id = datamaps['user2id']
        self.item2id = datamaps['item2id']
        self.user_list = list(datamaps['user2id'].keys())
        self.item_list = list(datamaps['item2id'].keys())
        self.id2item = datamaps['id2item']
        
        self.user_id2name = load_pickle(os.path.join('data', split, 'user_id2name.pkl'))
        
        self.meta_data = []
        for meta in parse(os.path.join('data', split, 'meta.json.gz')):
            self.meta_data.append(meta)
        self.meta_dict = {}
        for i, meta_item in enumerate(self.meta_data):
            self.meta_dict[meta_item['asin']] = i
            
        print('compute_datum_info')
        self.total_length = 0
        self.datum_info = []
        self.compute_datum_info()
        
    # compute_datum_info function intends to plan which data sample to be used for which task group according to the sample numbers in train_sample_numbers of pretrain.py
    def compute_datum_info(self):
        curr = 0
        for key in list(self.task_list.keys()):
            if key == 'rating':
                self.total_length += len(self.rating_data) * self.sample_numbers[key]
                for i in range(self.total_length - curr):
                    self.datum_info.append((i + curr, key, i // self.sample_numbers[key]))
                curr = self.total_length
            elif key == 'sequential':
                # The first group of sequential prompts (directly predict next item): 2-1 to 2-6 and 2-13
                if sum([0 < int(ind.split('-')[1]) <= 6 or int(ind.split('-')[1]) == 13 for ind in self.task_list[key]]):
                    self.total_length += len(self.sequential_data) * self.sample_numbers[key][0]
                    for i in range(self.total_length - curr):
                        self.datum_info.append((i + curr, key, i // self.sample_numbers[key][0]))
                    curr = self.total_length
                # The second group of sequential prompts (predict next item from a candidate list): 2-7 to 2-10
                if sum([6 < int(ind.split('-')[1]) <= 10 for ind in self.task_list[key]]):
                    self.total_length += len(self.sequential_data) * self.sample_numbers[key][1]
                    for i in range(self.total_length - curr):
                        self.datum_info.append((i + curr, key, i // self.sample_numbers[key][1]))
                    curr = self.total_length
                # The third group of sequential prompts (predict yes or no for each user-item pair): 2-11 to 2-12
                if sum([10 < int(ind.split('-')[1]) <= 12 for ind in self.task_list[key]]):
                    self.total_length += len(self.sequential_data) * self.sample_numbers[key][2]
                    for i in range(self.total_length - curr):
                        self.datum_info.append((i + curr, key, i // self.sample_numbers[key][2]))
                    curr = self.total_length
            elif key == 'explanation':
                self.total_length += len(self.exp_data) * self.sample_numbers[key]
                for i in range(self.total_length - curr):
                    self.datum_info.append((i + curr, key, i // self.sample_numbers[key]))
                curr = self.total_length
            elif key == 'review':
                self.total_length += len(self.review_data) * self.sample_numbers[key]
                for i in range(self.total_length - curr):
                    self.datum_info.append((i + curr, key, i // self.sample_numbers[key]))
                curr = self.total_length
            elif key == 'traditional':
                # The first group of direct recommendation prompts (predict yes or no for each user-item pair): 5-1 to 5-4
                if sum([0 < int(ind.split('-')[1]) <= 4 for ind in self.task_list[key]]):
                    self.total_length += len(self.user2id) * self.sample_numbers[key][0]
                    for i in range(self.total_length - curr):
                        self.datum_info.append((i + curr, key, i // self.sample_numbers[key][0]))
                    curr = self.total_length
                # The second group of direct recommendation prompts (choose one item from 100 candidates): 5-5 to 5-8
                if sum([4 < int(ind.split('-')[1]) <= 8 for ind in self.task_list[key]]):
                    self.total_length += len(self.user2id) * self.sample_numbers[key][1]
                    for i in range(self.total_length - curr):
                        self.datum_info.append((i + curr, key, i // self.sample_numbers[key][1]))
                    curr = self.total_length
            elif key == 'zeroshot':
                if sum([0 < int(ind.split('-')[1]) <= 7 for ind in self.task_list[key]]):
                    self.total_length += len(self.zeroshot_exp_data) * self.sample_numbers[key][0]
                    for i in range(self.total_length - curr):
                        self.datum_info.append((i + curr, key, i // self.sample_numbers[key][0]))
                    curr = self.total_length
            else:
                raise NotImplementedError
    
    # use Gaussian sampling to augment rating scores
    def gaussian_sampling(self, datum):
        if self.mode == 'train':
            if int(datum['overall']) == 1:
                sampled_rating = round(torch.normal(mean=torch.tensor((1.0+1.4)/2), std=torch.tensor((1.4-1.0)/4)).item(), 1)
            elif int(datum['overall']) == 2:
                sampled_rating = round(torch.normal(mean=torch.tensor((1.5+2.4)/2), std=torch.tensor((2.4-1.5)/4)).item(), 1)
            elif int(datum['overall']) == 3:
                sampled_rating = round(torch.normal(mean=torch.tensor((2.5+3.4)/2), std=torch.tensor((3.4-2.5)/4)).item(), 1)
            elif int(datum['overall']) == 4:
                sampled_rating = round(torch.normal(mean=torch.tensor((3.5+4.4)/2), std=torch.tensor((4.4-3.5)/4)).item(), 1)
            else:
                sampled_rating = round(torch.normal(mean=torch.tensor((4.5+5.0)/2), std=torch.tensor((5.0-4.5)/4)).item(), 1)
            if sampled_rating > 5.0:
                sampled_rating = 5.0
            if sampled_rating < 1.0:
                sampled_rating = 1.0
            return str(sampled_rating)
        else:
            return int(datum['overall'])
            
    def __len__(self):
        return self.total_length

    def __getitem__(self, idx):
        
        out_dict = {}
        out_dict['args'] = self.args
        
        loss_weight = 1.0
        
        datum_info_idx = self.datum_info[idx]
        assert datum_info_idx[0] == idx
        if len(datum_info_idx) == 3:
            task_name = datum_info_idx[1]
            datum_idx = datum_info_idx[2]
        elif len(datum_info_idx) == 4:
            task_name = datum_info_idx[1]
            datum_idx = datum_info_idx[2]
            task_idx = datum_info_idx[3]
        else:
            raise NotImplementedError
            
        if task_name == 'rating':            
            rating_datum = self.rating_data[datum_idx]
            task_candidates = self.task_list[task_name]
            task_idx = random.randint(0, len(task_candidates)-1) # random choose the task index for task_candidates
            task_template = self.all_tasks['rating'][task_candidates[task_idx]]
            assert task_template['task'] == 'rating'
            
            if task_template['id'] == '1-1':
                source_text = task_template['source'].format(self.user2id[rating_datum['reviewerID']], self.item2id[rating_datum['asin']])
                target_text = task_template['target'].format(self.gaussian_sampling(rating_datum))
            elif task_template['id'] == '1-2':
                if 'title' in self.meta_data[self.meta_dict[rating_datum['asin']]]:
                    title = self.meta_data[self.meta_dict[rating_datum['asin']]]['title']
                else:
                    title = 'unknown title'
                source_text = task_template['source'].format(self.user2id[rating_datum['reviewerID']], title) 
                target_text = task_template['target'].format(self.gaussian_sampling(rating_datum))
            elif task_template['id'] == '1-3':
                rand_prob = random.random()
                if rand_prob > 0.5:
                    source_text = task_template['source'].format(self.user2id[rating_datum['reviewerID']], self.item2id[rating_datum['asin']], int(rating_datum['overall']))
                    target_text = task_template['target'].format('yes')
                else:
                    overall_candidates = [_ for _ in range(0+1, 5+1) if _ != int(rating_datum['overall'])]
                    overall_idx = random.randint(0, len(overall_candidates)-1) # random choose the overall index for overall_candidates
                    source_text = task_template['source'].format(self.user2id[rating_datum['reviewerID']], self.item2id[rating_datum['asin']], overall_candidates[overall_idx])
                    target_text = task_template['target'].format('no')
            elif task_template['id'] == '1-4':
                source_text = task_template['source'].format(self.user2id[rating_datum['reviewerID']], self.item2id[rating_datum['asin']])
                if int(rating_datum['overall']) >= 4:
                    target_text = task_template['target'].format('like')
                else:
                    target_text = task_template['target'].format('dislike')
            elif task_template['id'] == '1-5':
                if 'title' in self.meta_data[self.meta_dict[rating_datum['asin']]]:
                    title = self.meta_data[self.meta_dict[rating_datum['asin']]]['title']
                else:
                    title = 'unknown title'
                source_text = task_template['source'].format(self.user2id[rating_datum['reviewerID']], self.item2id[rating_datum['asin']], title)
                target_text = task_template['target'].format(self.gaussian_sampling(rating_datum))
            elif task_template['id'] == '1-6':
                if 'reviewerName' in rating_datum:
                    user_desc = rating_datum['reviewerName']
                else:
                    user_desc = rating_datum['reviewerID']
                source_text = task_template['source'].format(user_desc, self.item2id[rating_datum['asin']])
                target_text = task_template['target'].format(self.gaussian_sampling(rating_datum))
            elif task_template['id'] == '1-7':
                if 'reviewerName' in rating_datum:
                    user_desc = rating_datum['reviewerName']
                else:
                    user_desc = rating_datum['reviewerID']
                if 'title' in self.meta_data[self.meta_dict[rating_datum['asin']]]:
                    title = self.meta_data[self.meta_dict[rating_datum['asin']]]['title']
                else:
                    title = 'unknown title'
                source_text = task_template['source'].format(user_desc, title)
                target_text = task_template['target'].format(self.gaussian_sampling(rating_datum))
            elif task_template['id'] == '1-8':
                rand_prob = random.random()
                if 'reviewerName' in rating_datum:
                    user_desc = rating_datum['reviewerName']
                else:
                    user_desc = rating_datum['reviewerID']
                if 'title' in self.meta_data[self.meta_dict[rating_datum['asin']]]:
                    title = self.meta_data[self.meta_dict[rating_datum['asin']]]['title']
                else:
                    title = 'unknown title'
                if rand_prob > 0.5:
                    source_text = task_template['source'].format(user_desc, int(rating_datum['overall']), title)
                    target_text = task_template['target'].format('yes')
                else:
                    overall_candidates = [_ for _ in range(0+1, 5+1) if _ != int(rating_datum['overall'])]
                    overall_idx = random.randint(0, len(overall_candidates)-1) # random choose the overall index for overall_candidates
                    source_text = task_template['source'].format(user_desc, overall_candidates[overall_idx], title)
                    target_text = task_template['target'].format('no')
            elif task_template['id'] == '1-9':
                if 'reviewerName' in rating_datum:
                    user_desc = rating_datum['reviewerName']
                else:
                    user_desc = rating_datum['reviewerID']
                if 'title' in self.meta_data[self.meta_dict[rating_datum['asin']]]:
                    title = self.meta_data[self.meta_dict[rating_datum['asin']]]['title']
                else:
                    title = 'unknown title'
                source_text = task_template['source'].format(user_desc, title)
                if int(rating_datum['overall']) >= 4:
                    target_text = task_template['target'].format('like')
                else:
                    target_text = task_template['target'].format('dislike')
            elif task_template['id'] == '1-10':
                if 'reviewerName' in rating_datum:
                    user_desc = rating_datum['reviewerName']
                else:
                    user_desc = rating_datum['reviewerID']
                if 'title' in self.meta_data[self.meta_dict[rating_datum['asin']]]:
                    title = self.meta_data[self.meta_dict[rating_datum['asin']]]['title']
                else:
                    title = 'unknown title'
                source_text = task_template['source'].format(user_desc, title)
                target_text = task_template['target'].format(self.gaussian_sampling(rating_datum))
            else:
                raise NotImplementedError
            
        elif task_name == 'sequential':
            sequential_datum = self.sequential_data[datum_idx]
            sequence = sequential_datum.split()
            user_id = sequence[0]
            user_desc = self.user_id2name[user_id]
            if self.mode == 'train':
                end_candidates = [_ for _ in range(max(2, len(sequence) - 6), len(sequence) - 3)]
                end_index = random.randint(0, len(end_candidates)-1)
                end_pos = end_candidates[end_index]
                start_candidates = [_ for _ in range(1, min(4, end_pos))]
                start_index = random.randint(0, len(start_candidates)-1)
                start_pos = start_candidates[start_index]
                purchase_history = sequence[start_pos:end_pos+1] # sample a history sequence from the full user purchase history
                target_item = sequence[end_pos+1]
            elif self.mode == 'val':
                purchase_history = sequence[1:-2]
                target_item = sequence[-2]
            elif self.mode == 'test':
                purchase_history = sequence[1:-1]
                target_item = sequence[-1]
            else:
                raise NotImplementedError
            
            task_candidates = self.task_list[task_name]
            task_idx = random.randint(0, len(task_candidates)-1) # random choose the task index for task_candidates
            task_template = self.all_tasks['sequential'][task_candidates[task_idx]]
            assert task_template['task'] == 'sequential'
            
            if task_template['id'] == '2-1':
                rand_prob = random.random()
                if rand_prob > 0.5:
                    source_text = task_template['source'].format(user_id, ' , '.join(purchase_history))
                else:
                    source_text = task_template['source'].format(user_id, ' -> '.join(purchase_history))
                target_text = task_template['target'].format(target_item)
            elif task_template['id'] == '2-2':
                rand_prob = random.random()
                if rand_prob > 0.5:
                    source_text = task_template['source'].format(user_id, ' , '.join(purchase_history))
                else:
                    source_text = task_template['source'].format(user_id, ' -> '.join(purchase_history))
                target_text = task_template['target'].format(target_item)
            elif task_template['id'] == '2-3':
                rand_prob = random.random()
                if rand_prob > 0.5:
                    source_text = task_template['source'].format(user_id, ' , '.join(purchase_history))
                else:
                    source_text = task_template['source'].format(user_id, ' -> '.join(purchase_history))
                target_text = task_template['target'].format(target_item)
            elif task_template['id'] == '2-4':
                rand_prob = random.random()
                if rand_prob > 0.5:
                    source_text = task_template['source'].format(user_desc, ' , '.join(purchase_history))
                else:
                    source_text = task_template['source'].format(user_desc, ' -> '.join(purchase_history))
                target_text = task_template['target'].format(target_item)
            elif task_template['id'] == '2-5':
                rand_prob = random.random()
                if rand_prob > 0.5:
                    source_text = task_template['source'].format(user_desc, ' , '.join(purchase_history))
                else:
                    source_text = task_template['source'].format(user_desc, ' -> '.join(purchase_history))
                target_text = task_template['target'].format(target_item)
            elif task_template['id'] == '2-6':
                rand_prob = random.random()
                if rand_prob > 0.5:
                    source_text = task_template['source'].format(user_desc, ' , '.join(purchase_history))
                else:
                    source_text = task_template['source'].format(user_desc, ' -> '.join(purchase_history))
                target_text = task_template['target'].format(target_item)
            elif task_template['id'] == '2-7' or task_template['id'] == '2-9':
                if self.mode in ['train', 'val']:
                    user_seq = self.user_items[user_id]
                    candidate_samples = []
                    candidate_num = 99
                    while len(candidate_samples) < candidate_num:
                        if self.sample_type == 'random':
                            sample_ids = np.random.choice(self.all_item, candidate_num, replace=False)
                        else:
                            sample_ids = np.random.choice(self.all_item, candidate_num, replace=False, p=self.probability)
                        sample_ids = [str(item) for item in sample_ids if item not in user_seq and item not in candidate_samples]
                        candidate_samples.extend(sample_ids)
                    candidate_samples = candidate_samples[:candidate_num]
                elif self.mode == 'test':
                    assert user_id == self.negative_samples[int(user_id)-1].split(' ', 1)[0]
                    candidate_samples = self.negative_samples[int(user_id)-1].split(' ', 1)[1].split(' ')
                else:
                    raise NotImplementedError
                candidate_samples.extend([target_item])
                random.shuffle(candidate_samples)
                rand_prob = random.random()
                if rand_prob > 0.5:
                    source_text = task_template['source'].format(user_id, ' , '.join(purchase_history), ' , '.join(candidate_samples))
                else:
                    source_text = task_template['source'].format(user_id, ' -> '.join(purchase_history), ' , '.join(candidate_samples))
                target_text = task_template['target'].format(target_item)
            elif task_template['id'] == '2-8' or task_template['id'] == '2-10':
                if self.mode in ['train', 'val']:
                    user_seq = self.user_items[user_id]
                    candidate_samples = []
                    candidate_num = 99
                    while len(candidate_samples) < candidate_num:
                        if self.sample_type == 'random':
                            sample_ids = np.random.choice(self.all_item, candidate_num, replace=False)
                        else:
                            sample_ids = np.random.choice(self.all_item, candidate_num, replace=False, p=self.probability)
                        sample_ids = [str(item) for item in sample_ids if item not in user_seq and item not in candidate_samples]
                        candidate_samples.extend(sample_ids)
                    candidate_samples = candidate_samples[:candidate_num]
                elif self.mode == 'test':
                    assert user_id == self.negative_samples[int(user_id)-1].split(' ', 1)[0]
                    candidate_samples = self.negative_samples[int(user_id)-1].split(' ', 1)[1].split(' ')
                else:
                    raise NotImplementedError
                candidate_samples.extend([target_item])
                random.shuffle(candidate_samples)
                rand_prob = random.random()
                if rand_prob > 0.5:
                    source_text = task_template['source'].format(user_desc, ' , '.join(purchase_history), ' , '.join(candidate_samples))
                else:
                    source_text = task_template['source'].format(user_desc, ' -> '.join(purchase_history), ' , '.join(candidate_samples))
                target_text = task_template['target'].format(target_item)
            elif task_template['id'] == '2-11':
                symbol_prob = random.random()
                if symbol_prob > 0.5:
                    symbol = ' , '
                else:
                    symbol = ' -> '
                rand_prob = random.random()
                if rand_prob > 0.5:
                    source_text = task_template['source'].format(user_id, symbol.join(purchase_history), target_item)
                    target_text = task_template['target'].format('yes')
                else:
                    user_seq = self.user_items[user_id]
                    candidate_samples = []
                    candidate_num = 1
                    while len(candidate_samples) < candidate_num:
                        if self.sample_type == 'random':
                            sample_ids = np.random.choice(self.all_item, candidate_num, replace=False)
                        else:
                            sample_ids = np.random.choice(self.all_item, candidate_num, replace=False, p=self.probability)
                        sample_ids = [str(item) for item in sample_ids if item not in user_seq and item not in candidate_samples]
                        candidate_samples.extend(sample_ids)
                    candidate_samples = candidate_samples[:candidate_num]
                    source_text = task_template['source'].format(user_id, symbol.join(purchase_history), candidate_samples[0])
                    target_text = task_template['target'].format('no')
            elif task_template['id'] == '2-12':
                symbol_prob = random.random()
                if symbol_prob > 0.5:
                    symbol = ' , '
                else:
                    symbol = ' -> '
                rand_prob = random.random()
                if rand_prob > 0.5:
                    source_text = task_template['source'].format(user_desc, symbol.join(purchase_history), target_item)
                    target_text = task_template['target'].format('yes')
                else:
                    user_seq = self.user_items[user_id]
                    candidate_samples = []
                    candidate_num = 1
                    while len(candidate_samples) < candidate_num:
                        if self.sample_type == 'random':
                            sample_ids = np.random.choice(self.all_item, candidate_num, replace=False)
                        else:
                            sample_ids = np.random.choice(self.all_item, candidate_num, replace=False, p=self.probability)
                        sample_ids = [str(item) for item in sample_ids if item not in user_seq and item not in candidate_samples]
                        candidate_samples.extend(sample_ids)
                    candidate_samples = candidate_samples[:candidate_num]
                    source_text = task_template['source'].format(user_desc, symbol.join(purchase_history), candidate_samples[0])
                    target_text = task_template['target'].format('no')
            elif task_template['id'] == '2-13':
                rand_prob = random.random()
                if rand_prob > 0.5:
                    source_text = task_template['source'].format(user_desc, ' , '.join(purchase_history))
                else:
                    source_text = task_template['source'].format(user_desc, ' -> '.join(purchase_history))
                target_text = task_template['target'].format(target_item)
            else:
                raise NotImplementedError
        
        elif task_name == 'explanation':
            exp_datum = self.exp_data[datum_idx]
            task_candidates = self.task_list[task_name]
            task_idx = random.randint(0, len(task_candidates)-1) # random choose the task index for task_candidates
            task_template = self.all_tasks['explanation'][task_candidates[task_idx]]
            assert task_template['task'] == 'explanation'
            
            if task_template['id'] == '3-1':
                if 'title' in self.meta_data[self.meta_dict[exp_datum['asin']]]:
                    title = self.meta_data[self.meta_dict[exp_datum['asin']]]['title']
                else:
                    title = 'unknown title'
                source_text = task_template['source'].format(self.user2id[exp_datum['reviewerID']], title)
                target_text = task_template['target'].format(exp_datum['explanation'])
            elif task_template['id'] == '3-2':
                source_text = task_template['source'].format(exp_datum['summary'], self.user2id[exp_datum['reviewerID']], self.item2id[exp_datum['asin']]) 
                target_text = task_template['target'].format(exp_datum['explanation'])
            elif task_template['id'] == '3-3':
                if 'title' in self.meta_data[self.meta_dict[exp_datum['asin']]]:
                    title = self.meta_data[self.meta_dict[exp_datum['asin']]]['title']
                else:
                    title = 'unknown title'
                source_text = task_template['source'].format(self.user2id[exp_datum['reviewerID']], int(exp_datum['overall']), title)
                target_text = task_template['target'].format(exp_datum['explanation'])
            elif task_template['id'] == '3-4':
                if 'reviewerName' in exp_datum:
                    user_desc = exp_datum['reviewerName']
                else:
                    user_desc = exp_datum['reviewerID']
                if 'title' in self.meta_data[self.meta_dict[exp_datum['asin']]]:
                    title = self.meta_data[self.meta_dict[exp_datum['asin']]]['title']
                else:
                    title = 'unknown title'
                source_text = task_template['source'].format(user_desc, title)
                target_text = task_template['target'].format(exp_datum['explanation'])
            elif task_template['id'] == '3-5':
                if 'reviewerName' in exp_datum:
                    user_desc = exp_datum['reviewerName']
                else:
                    user_desc = exp_datum['reviewerID']
                if 'title' in self.meta_data[self.meta_dict[exp_datum['asin']]]:
                    title = self.meta_data[self.meta_dict[exp_datum['asin']]]['title']
                else:
                    title = 'unknown title'
                source_text = task_template['source'].format(exp_datum['summary'], user_desc, title)
                target_text = task_template['target'].format(exp_datum['explanation'])
            elif task_template['id'] == '3-6':
                if 'reviewerName' in exp_datum:
                    user_desc = exp_datum['reviewerName']
                else:
                    user_desc = exp_datum['reviewerID']
                source_text = task_template['source'].format(user_desc, int(exp_datum['overall']), self.item2id[exp_datum['asin']])
                target_text = task_template['target'].format(exp_datum['explanation'])
            elif task_template['id'] == '3-7':
                source_text = task_template['source'].format(exp_datum['feature'], self.user2id[exp_datum['reviewerID']], self.item2id[exp_datum['asin']])
                target_text = task_template['target'].format(self.gaussian_sampling(exp_datum), exp_datum['explanation'])
            elif task_template['id'] == '3-8':
                if 'reviewerName' in exp_datum:
                    user_desc = exp_datum['reviewerName']
                else:
                    user_desc = exp_datum['reviewerID']
                source_text = task_template['source'].format(user_desc, self.item2id[exp_datum['asin']])
                target_text = task_template['target'].format(self.gaussian_sampling(exp_datum), exp_datum['explanation'])
            elif task_template['id'] == '3-9':
                if 'title' in self.meta_data[self.meta_dict[exp_datum['asin']]]:
                    title = self.meta_data[self.meta_dict[exp_datum['asin']]]['title']
                else:
                    title = 'unknown title'
                source_text = task_template['source'].format(exp_datum['feature'], self.user2id[exp_datum['reviewerID']], title)
                target_text = task_template['target'].format(exp_datum['explanation'])
            elif task_template['id'] == '3-10':
                if 'reviewerName' in exp_datum:
                    user_desc = exp_datum['reviewerName']
                else:
                    user_desc = exp_datum['reviewerID']
                if 'title' in self.meta_data[self.meta_dict[exp_datum['asin']]]:
                    title = self.meta_data[self.meta_dict[exp_datum['asin']]]['title']
                else:
                    title = 'unknown title'
                source_text = task_template['source'].format(exp_datum['feature'], user_desc, title)
                target_text = task_template['target'].format(exp_datum['explanation'])
            elif task_template['id'] == '3-11':
                source_text = task_template['source'].format(exp_datum['feature'], int(exp_datum['overall']), self.user2id[exp_datum['reviewerID']], self.item2id[exp_datum['asin']])
                target_text = task_template['target'].format(exp_datum['explanation'])
            elif task_template['id'] == '3-12':
                if 'reviewerName' in exp_datum:
                    user_desc = exp_datum['reviewerName']
                else:
                    user_desc = exp_datum['reviewerID']
                source_text = task_template['source'].format(exp_datum['feature'], int(exp_datum['overall']), user_desc, self.item2id[exp_datum['asin']])
                target_text = task_template['target'].format(exp_datum['explanation'])
            else:
                raise NotImplementedError
                
        elif task_name == 'review': 
            review_datum = self.review_data[datum_idx]
            task_candidates = self.task_list[task_name]
            task_idx = random.randint(0, len(task_candidates)-1) # random choose the task index for task_candidates
            task_template = self.all_tasks['review'][task_candidates[task_idx]]
            assert task_template['task'] == 'review'
            
            if task_template['id'] == '4-1':
                source_text = task_template['source'].format(self.user2id[review_datum['reviewerID']], review_datum['reviewText'])
                target_text = task_template['target'].format(review_datum['summary'])
            elif task_template['id'] == '4-2':
                source_text = task_template['source'].format(self.user2id[review_datum['reviewerID']], review_datum['reviewText'])
                target_text = task_template['target'].format(int(review_datum['overall']))
            elif task_template['id'] == '4-3':
                if 'reviewerName' in review_datum:
                    user_desc = review_datum['reviewerName']
                else:
                    user_desc = review_datum['reviewerID']
                source_text = task_template['source'].format(user_desc, review_datum['reviewText'])
                target_text = task_template['target'].format(review_datum['summary'])
            elif task_template['id'] == '4-4':
                if 'reviewerName' in review_datum:
                    user_desc = review_datum['reviewerName']
                else:
                    user_desc = review_datum['reviewerID']
                source_text = task_template['source'].format(user_desc, review_datum['reviewText'])
                target_text = task_template['target'].format(int(review_datum['overall']))
            else:
                raise NotImplementedError
            
        elif task_name == 'traditional':
            sequential_datum = self.sequential_data[datum_idx]
            sequence = sequential_datum.split()
            user_id = sequence[0]
            user_desc = self.user_id2name[user_id]
            if self.mode == 'train':
                target_candidates = sequence[1:-2]
                target_idx = random.randint(0, len(target_candidates)-1) # random choose the target index for target_candidates
                target_item = target_candidates[target_idx]
            elif self.mode == 'val':
                target_item = sequence[-2]
            elif self.mode == 'test':
                target_item = sequence[-1]
            else:
                raise NotImplementedError
            
            task_candidates = self.task_list[task_name]
            task_idx = random.randint(0, len(task_candidates)-1) # random choose the task index for task_candidates
            task_template = self.all_tasks['traditional'][task_candidates[task_idx]]
            assert task_template['task'] == 'traditional'
            
            if task_template['id'] == '5-1':
                rand_prob = random.random()
                if rand_prob > 0.5:
                    source_text = task_template['source'].format(user_id, target_item)
                    target_text = task_template['target'].format('yes')
                else:
                    user_seq = self.user_items[user_id]
                    candidate_samples = []
                    candidate_num = 1
                    while len(candidate_samples) < candidate_num:
                        if self.sample_type == 'random':
                            sample_ids = np.random.choice(self.all_item, candidate_num, replace=False)
                        else:
                            sample_ids = np.random.choice(self.all_item, candidate_num, replace=False, p=self.probability)
                        sample_ids = [str(item) for item in sample_ids if item not in user_seq and item not in candidate_samples]
                        candidate_samples.extend(sample_ids)
                    candidate_samples = candidate_samples[:candidate_num]
                    source_text = task_template['source'].format(user_id, candidate_samples[0])
                    target_text = task_template['target'].format('no')
            elif task_template['id'] == '5-2':
                rand_prob = random.random()
                if rand_prob > 0.5:
                    source_text = task_template['source'].format(target_item, user_desc)
                    target_text = task_template['target'].format('yes')
                else:
                    user_seq = self.user_items[user_id]
                    candidate_samples = []
                    candidate_num = 1
                    while len(candidate_samples) < candidate_num:
                        if self.sample_type == 'random':
                            sample_ids = np.random.choice(self.all_item, candidate_num, replace=False)
                        else:
                            sample_ids = np.random.choice(self.all_item, candidate_num, replace=False, p=self.probability)
                        sample_ids = [str(item) for item in sample_ids if item not in user_seq and item not in candidate_samples]
                        candidate_samples.extend(sample_ids)
                    candidate_samples = candidate_samples[:candidate_num]
                    source_text = task_template['source'].format(candidate_samples[0], user_desc)
                    target_text = task_template['target'].format('no')
            elif task_template['id'] == '5-3':
                rand_prob = random.random()
                if rand_prob > 0.5:
                    if 'title' in self.meta_data[self.meta_dict[self.id2item[target_item]]]:
                        title = self.meta_data[self.meta_dict[self.id2item[target_item]]]['title']
                    else:
                        title = 'unknown title'
                    source_text = task_template['source'].format(user_desc, title)
                    target_text = task_template['target'].format('yes')
                else:
                    user_seq = self.user_items[user_id]
                    candidate_samples = []
                    candidate_num = 1
                    while len(candidate_samples) < candidate_num:
                        if self.sample_type == 'random':
                            sample_ids = np.random.choice(self.all_item, candidate_num, replace=False)
                        else:
                            sample_ids = np.random.choice(self.all_item, candidate_num, replace=False, p=self.probability)
                        sample_ids = [str(item) for item in sample_ids if item not in user_seq and item not in candidate_samples]
                        candidate_samples.extend(sample_ids)
                    candidate_samples = candidate_samples[:candidate_num]
                    if 'title' in self.meta_data[self.meta_dict[self.id2item[candidate_samples[0]]]]:
                        title = self.meta_data[self.meta_dict[self.id2item[candidate_samples[0]]]]['title']
                    else:
                        title = 'unknown title'
                    source_text = task_template['source'].format(user_desc, title)
                    target_text = task_template['target'].format('no')
            elif task_template['id'] == '5-4':
                rand_prob = random.random()
                if rand_prob > 0.5:
                    if 'title' in self.meta_data[self.meta_dict[self.id2item[target_item]]]:
                        title = self.meta_data[self.meta_dict[self.id2item[target_item]]]['title']
                    else:
                        title = 'unknown title'
                    source_text = task_template['source'].format(user_id, title)
                    target_text = task_template['target'].format('yes')
                else:
                    user_seq = self.user_items[user_id]
                    candidate_samples = []
                    candidate_num = 1
                    while len(candidate_samples) < candidate_num:
                        if self.sample_type == 'random':
                            sample_ids = np.random.choice(self.all_item, candidate_num, replace=False)
                        else:
                            sample_ids = np.random.choice(self.all_item, candidate_num, replace=False, p=self.probability)
                        sample_ids = [str(item) for item in sample_ids if item not in user_seq and item not in candidate_samples]
                        candidate_samples.extend(sample_ids)
                    candidate_samples = candidate_samples[:candidate_num]
                    if 'title' in self.meta_data[self.meta_dict[self.id2item[candidate_samples[0]]]]:
                        title = self.meta_data[self.meta_dict[self.id2item[candidate_samples[0]]]]['title']
                    else:
                        title = 'unknown title'
                    source_text = task_template['source'].format(user_id, title)
                    target_text = task_template['target'].format('no')
            elif task_template['id'] == '5-5' or task_template['id'] == '5-6':
                user_seq = self.user_items[user_id]
                candidate_samples = []
                candidate_num = 99
                while len(candidate_samples) < candidate_num:
                    if self.sample_type == 'random':
                        sample_ids = np.random.choice(self.all_item, candidate_num, replace=False)
                    else:
                        sample_ids = np.random.choice(self.all_item, candidate_num, replace=False, p=self.probability)
                    sample_ids = [str(item) for item in sample_ids if item not in user_seq and item not in candidate_samples]
                    candidate_samples.extend(sample_ids)
                candidate_samples = candidate_samples[:candidate_num]
                candidate_samples.extend([target_item])
                random.shuffle(candidate_samples)
                source_text = task_template['source'].format(user_desc, ' , '.join(candidate_samples))
                target_text = task_template['target'].format(target_item)
            elif task_template['id'] == '5-7' or task_template['id'] == '5-8':
                user_seq = self.user_items[user_id]
                candidate_samples = []
                candidate_num = 99
                while len(candidate_samples) < candidate_num:
                    if self.sample_type == 'random':
                        sample_ids = np.random.choice(self.all_item, candidate_num, replace=False)
                    else:
                        sample_ids = np.random.choice(self.all_item, candidate_num, replace=False, p=self.probability)
                    sample_ids = [str(item) for item in sample_ids if item not in user_seq and item not in candidate_samples]
                    candidate_samples.extend(sample_ids)
                candidate_samples = candidate_samples[:candidate_num]
                candidate_samples.extend([target_item])
                random.shuffle(candidate_samples)
                source_text = task_template['source'].format(user_id, ' , '.join(candidate_samples))
                target_text = task_template['target'].format(target_item)
            else:
                raise NotImplementedError
                
        elif task_name == 'zeroshot' and self.mode == 'test':
            zeroshot_exp_datum = self.zeroshot_exp_data[datum_idx]
            task_candidates = self.task_list[task_name]
            task_idx = random.randint(0, len(task_candidates)-1) # random choose the task index for task_candidates
            task_template = self.all_tasks['zeroshot'][task_candidates[task_idx]]
            assert task_template['task'] == 'zeroshot'
            
            if task_template['id'] == 'Z-1':
                source_text = task_template['source'].format(self.user2id[zeroshot_exp_datum['reviewerID']], zeroshot_exp_datum['item_title'], \
                                                             zeroshot_exp_datum['brand'], zeroshot_exp_datum['price'])
                if int(zeroshot_exp_datum['overall']) >= 4:
                    target_text = task_template['target'].format('like')
                else:
                    target_text = task_template['target'].format('dislike')
            elif task_template['id'] == 'Z-2':
                if 'reviewerName' in zeroshot_exp_datum:
                    user_desc = zeroshot_exp_datum['reviewerName']
                else:
                    user_desc = zeroshot_exp_datum['reviewerID']
                source_text = task_template['source'].format(zeroshot_exp_datum['item_title'], zeroshot_exp_datum['brand'], \
                                                             zeroshot_exp_datum['price'], user_desc)
                target_text = task_template['target'].format(int(zeroshot_exp_datum['overall']))
            elif task_template['id'] == 'Z-3':
                source_text = task_template['source'].format(self.user2id[zeroshot_exp_datum['reviewerID']], zeroshot_exp_datum['item_title'], \
                                                             zeroshot_exp_datum['price'], zeroshot_exp_datum['brand'])
                target_text = task_template['target'].format(int(zeroshot_exp_datum['overall']))
            elif task_template['id'] == 'Z-4':
                if 'reviewerName' in zeroshot_exp_datum:
                    user_desc = zeroshot_exp_datum['reviewerName']
                else:
                    user_desc = zeroshot_exp_datum['reviewerID']
                source_text = task_template['source'].format(user_desc, zeroshot_exp_datum['item_title'], \
                                                             zeroshot_exp_datum['price'], zeroshot_exp_datum['brand'])
                if int(zeroshot_exp_datum['overall']) >= 4:
                    target_text = task_template['target'].format('like')
                else:
                    target_text = task_template['target'].format('dislike')
            elif task_template['id'] == 'Z-5':
                if 'reviewerName' in zeroshot_exp_datum:
                    user_desc = zeroshot_exp_datum['reviewerName']
                else:
                    user_desc = zeroshot_exp_datum['reviewerID']
                source_text = task_template['source'].format(user_desc, zeroshot_exp_datum['item_title'], \
                                                             zeroshot_exp_datum['brand'], zeroshot_exp_datum['price'])
                target_text = task_template['target'].format(zeroshot_exp_datum['explanation'])
            elif task_template['id'] == 'Z-6':
                source_text = task_template['source'].format(zeroshot_exp_datum['feature'], self.user2id[zeroshot_exp_datum['reviewerID']], \
                                                             int(zeroshot_exp_datum['overall']), zeroshot_exp_datum['item_title'], \
                                                             zeroshot_exp_datum['price'], zeroshot_exp_datum['brand'])
                target_text = task_template['target'].format(zeroshot_exp_datum['explanation'])
            elif task_template['id'] == 'Z-7':
                if 'reviewerName' in zeroshot_exp_datum:
                    user_desc = zeroshot_exp_datum['reviewerName']
                else:
                    user_desc = zeroshot_exp_datum['reviewerID']
                source_text = task_template['source'].format(zeroshot_exp_datum['item_title'], user_desc)
                target_text = task_template['target'].format(zeroshot_exp_datum['explanation'])
            else:
                raise NotImplementedError
            
        else:
            raise NotImplementedError
            
        input_ids = self.tokenizer.encode(
                source_text, padding=True, truncation=True, max_length=self.args.max_text_length)
        tokenized_text = self.tokenizer.tokenize(source_text)
        whole_word_ids = self.calculate_whole_word_ids(tokenized_text, input_ids)
        assert len(whole_word_ids) == len(input_ids)
        
        target_ids = self.tokenizer.encode(
                target_text, padding=True, truncation=True, max_length=self.args.gen_max_length)

        out_dict['input_ids'] = torch.LongTensor(input_ids)
        out_dict['input_length'] = len(input_ids)
        out_dict['whole_word_ids'] = torch.LongTensor(whole_word_ids)
        out_dict['target_ids'] = torch.LongTensor(target_ids)
        out_dict['target_length'] = len(target_ids)

        out_dict['source_text'] = source_text
        out_dict['tokenized_text'] = tokenized_text
        out_dict['target_text'] = target_text

        out_dict['task'] = task_template['task']

        out_dict['loss_weight'] = loss_weight

        return out_dict
    
    def calculate_whole_word_ids(self, tokenized_text, input_ids):
        whole_word_ids = []
        curr = 0
        for i in range(len(tokenized_text)):
            if tokenized_text[i].startswith('▁'):
                curr += 1
                whole_word_ids.append(curr)
            else:
                whole_word_ids.append(curr)
        last_item = whole_word_ids[len(input_ids) - 2]
        return whole_word_ids[:len(input_ids) - 1] + [0] ## the added [0] is for </s>
    
    def collate_fn(self, batch):
        batch_entry = {}

        B = len(batch)

        args = self.args

        S_W_L = max(entry['input_length'] for entry in batch)
        T_W_L = max(entry['target_length'] for entry in batch)

        input_ids = torch.ones(B, S_W_L, dtype=torch.long) * self.tokenizer.pad_token_id
        whole_word_ids = torch.ones(B, S_W_L, dtype=torch.long) * self.tokenizer.pad_token_id
        target_ids = torch.ones(B, T_W_L, dtype=torch.long) * self.tokenizer.pad_token_id

        loss_weights = torch.ones(B, dtype=torch.float)

        tasks = []
        source_text = []
        tokenized_text = []
        target_text = []

        for i, entry in enumerate(batch):
            input_ids[i, :entry['input_length']] = entry['input_ids']
            whole_word_ids[i, :entry['input_length']] = entry['whole_word_ids']
            target_ids[i, :entry['target_length']] = entry['target_ids']

            if 'task' in entry:
                tasks.append(entry['task'])

            if 'source_text' in entry:
                source_text.append(entry['source_text'])
                
            if 'tokenized_text' in entry:
                tokenized_text.append(entry['tokenized_text'])
                
            if 'target_text' in entry:
                target_text.append(entry['target_text'])

            if 'loss_weight' in entry:
                loss_weights[i] = entry['loss_weight']

        assert 't5' in args.backbone
        word_mask = target_ids != self.tokenizer.pad_token_id
        target_ids[~word_mask] = -100
        batch_entry['task'] = tasks

        batch_entry['source_text'] = source_text
        batch_entry['target_text'] = target_text

        batch_entry['input_ids'] = input_ids
        batch_entry['whole_word_ids'] = whole_word_ids
        batch_entry['target_ids'] = target_ids

        batch_entry['loss_weights'] = loss_weights

        return batch_entry

    
class P5_Yelp_Dataset(Dataset):
    def __init__(self, all_tasks, task_list, tokenizer, args, sample_numbers, mode='train', split='yelp', rating_augment=False, sample_type='random'): 
        self.all_tasks = all_tasks
        self.task_list = task_list
        self.tokenizer = tokenizer
        self.args = args
        self.sample_numbers = sample_numbers
        self.split = split
        self.rating_augment = rating_augment
        self.sample_type = sample_type
        
        print('Data sources: ', split.split(','))
        self.mode = mode
        if self.mode == 'train':
            self.review_data = load_pickle(os.path.join('data', split, 'review_splits.pkl'))['train']
            self.exp_data = load_pickle(os.path.join('data', split, 'exp_splits.pkl'))['train']
            if self.rating_augment:
                self.rating_data = load_pickle(os.path.join('data', split, 'rating_splits_augmented.pkl'))['train']
            else:
                self.rating_data = self.review_data
        elif self.mode == 'val':
            self.review_data = load_pickle(os.path.join('data', split, 'review_splits.pkl'))['val']
            self.exp_data = load_pickle(os.path.join('data', split, 'exp_splits.pkl'))['val']
            if self.rating_augment:
                self.rating_data = load_pickle(os.path.join('data', split, 'rating_splits_augmented.pkl'))['val']
            else:
                self.rating_data = self.review_data
        elif self.mode == 'test':
            self.review_data = load_pickle(os.path.join('data', split, 'review_splits.pkl'))['test']
            self.exp_data = load_pickle(os.path.join('data', split, 'exp_splits.pkl'))['test']
            if self.rating_augment:
                self.rating_data = load_pickle(os.path.join('data', split, 'rating_splits_augmented.pkl'))['test']
            else:
                self.rating_data = self.review_data
        else:
            raise NotImplementedError
            
        self.sequential_data = ReadLineFromFile(os.path.join('data', split, 'sequential_data.txt'))
        item_count = defaultdict(int)
        user_items = defaultdict()

        for line in self.sequential_data:
            user, items = line.strip().split(' ', 1)
            items = items.split(' ')
            items = [int(item) for item in items]
            user_items[user] = items
            for item in items:
                item_count[item] += 1
                
        self.all_item = list(item_count.keys())
        count = list(item_count.values())
        sum_value = np.sum([x for x in count])
        self.probability = [value / sum_value for value in count]
        self.user_items = user_items
        
        if self.mode == 'test':
            self.negative_samples = ReadLineFromFile(os.path.join('data', split, 'negative_samples.txt'))
            
        datamaps = load_json(os.path.join('data', split, 'datamaps.json'))
        self.user2id = datamaps['user2id']
        self.item2id = datamaps['item2id']
        self.user_list = list(datamaps['user2id'].keys())
        self.item_list = list(datamaps['item2id'].keys())
        self.id2item = datamaps['id2item']
        
        self.user_id2name = load_pickle(os.path.join('data', split, 'user_id2name.pkl'))
            
        self.meta_data = load_pickle(os.path.join('data', split, 'meta_data.pkl'))
        self.user_data = load_pickle(os.path.join('data', split, 'user_data.pkl'))
        self.meta_dict = {}
        for i, meta_item in enumerate(self.meta_data):
            self.meta_dict[meta_item['business_id']] = i
        self.user_meta_dict = {}
        for j, user_meta_item in enumerate(self.user_data):
            self.user_meta_dict[user_meta_item['user_id']] = j
            
        print('compute_datum_info')
        self.total_length = 0
        self.datum_info = []
        self.compute_datum_info()
        
    def compute_datum_info(self):
        curr = 0
        for key in list(self.task_list.keys()):
            if key == 'rating':
                self.total_length += len(self.rating_data) * self.sample_numbers[key]
                for i in range(self.total_length - curr):
                    self.datum_info.append((i + curr, key, i // self.sample_numbers[key]))
                curr = self.total_length
            elif key == 'sequential':
                if sum([0 < int(ind.split('-')[1]) <= 6 or int(ind.split('-')[1]) == 13 for ind in self.task_list[key]]):
                    self.total_length += len(self.sequential_data) * self.sample_numbers[key][0]
                    for i in range(self.total_length - curr):
                        self.datum_info.append((i + curr, key, i // self.sample_numbers[key][0]))
                    curr = self.total_length
                if sum([6 < int(ind.split('-')[1]) <= 10 for ind in self.task_list[key]]):
                    self.total_length += len(self.sequential_data) * self.sample_numbers[key][1]
                    for i in range(self.total_length - curr):
                        self.datum_info.append((i + curr, key, i // self.sample_numbers[key][1]))
                    curr = self.total_length
                if sum([10 < int(ind.split('-')[1]) <= 12 for ind in self.task_list[key]]):
                    self.total_length += len(self.sequential_data) * self.sample_numbers[key][2]
                    for i in range(self.total_length - curr):
                        self.datum_info.append((i + curr, key, i // self.sample_numbers[key][2]))
                    curr = self.total_length
            elif key == 'explanation':
                self.total_length += len(self.exp_data) * self.sample_numbers[key]
                for i in range(self.total_length - curr):
                    self.datum_info.append((i + curr, key, i // self.sample_numbers[key]))
                curr = self.total_length
            elif key == 'review':
                self.total_length += len(self.review_data) * self.sample_numbers[key]
                for i in range(self.total_length - curr):
                    self.datum_info.append((i + curr, key, i // self.sample_numbers[key]))
                curr = self.total_length
            elif key == 'traditional':
                if sum([0 < int(ind.split('-')[1]) <= 4 for ind in self.task_list[key]]):
                    self.total_length += len(self.user2id) * self.sample_numbers[key][0]
                    for i in range(self.total_length - curr):
                        self.datum_info.append((i + curr, key, i // self.sample_numbers[key][0]))
                    curr = self.total_length
                if sum([4 < int(ind.split('-')[1]) <= 8 for ind in self.task_list[key]]):
                    self.total_length += len(self.user2id) * self.sample_numbers[key][1]
                    for i in range(self.total_length - curr):
                        self.datum_info.append((i + curr, key, i // self.sample_numbers[key][1]))
                    curr = self.total_length
            else:
                raise NotImplementedError
    
    def gaussian_sampling(self, datum):
        if self.mode == 'train':
            if int(datum['overall']) == 1:
                sampled_rating = round(torch.normal(mean=torch.tensor((1.0+1.4)/2), std=torch.tensor((1.4-1.0)/4)).item(), 1)
            elif int(datum['overall']) == 2:
                sampled_rating = round(torch.normal(mean=torch.tensor((1.5+2.4)/2), std=torch.tensor((2.4-1.5)/4)).item(), 1)
            elif int(datum['overall']) == 3:
                sampled_rating = round(torch.normal(mean=torch.tensor((2.5+3.4)/2), std=torch.tensor((3.4-2.5)/4)).item(), 1)
            elif int(datum['overall']) == 4:
                sampled_rating = round(torch.normal(mean=torch.tensor((3.5+4.4)/2), std=torch.tensor((4.4-3.5)/4)).item(), 1)
            else:
                sampled_rating = round(torch.normal(mean=torch.tensor((4.5+5.0)/2), std=torch.tensor((5.0-4.5)/4)).item(), 1)
            if sampled_rating > 5.0:
                sampled_rating = 5.0
            if sampled_rating < 1.0:
                sampled_rating = 1.0
            return str(sampled_rating)
        else:
            return int(datum['overall'])
            
    def __len__(self):
        return self.total_length

    def __getitem__(self, idx):
        
        out_dict = {}
        out_dict['args'] = self.args
        
        loss_weight = 1.0
        
        datum_info_idx = self.datum_info[idx]
        assert datum_info_idx[0] == idx
        if len(datum_info_idx) == 3:
            task_name = datum_info_idx[1]
            datum_idx = datum_info_idx[2]
        elif len(datum_info_idx) == 4:
            task_name = datum_info_idx[1]
            datum_idx = datum_info_idx[2]
            task_idx = datum_info_idx[3]
        else:
            raise NotImplementedError
            
        if task_name == 'rating':            
            rating_datum = self.rating_data[datum_idx]
            task_candidates = self.task_list[task_name]
            task_idx = random.randint(0, len(task_candidates)-1) # random choose the task index for task_candidates
            task_template = self.all_tasks['rating'][task_candidates[task_idx]]
            assert task_template['task'] == 'rating'
            
            if task_template['id'] == '1-1':
                source_text = task_template['source'].format(self.user2id[rating_datum['reviewerID']], self.item2id[rating_datum['asin']])
                target_text = task_template['target'].format(self.gaussian_sampling(rating_datum))
            elif task_template['id'] == '1-2':
                if 'name' in self.meta_data[self.meta_dict[rating_datum['asin']]]:
                    title = self.meta_data[self.meta_dict[rating_datum['asin']]]['name']
                else:
                    title = 'unknown name'
                source_text = task_template['source'].format(self.user2id[rating_datum['reviewerID']], title) 
                target_text = task_template['target'].format(self.gaussian_sampling(rating_datum))
            elif task_template['id'] == '1-3':
                rand_prob = random.random()
                if rand_prob > 0.5:
                    source_text = task_template['source'].format(self.user2id[rating_datum['reviewerID']], self.item2id[rating_datum['asin']], int(rating_datum['overall']))
                    target_text = task_template['target'].format('yes')
                else:
                    overall_candidates = [_ for _ in range(0+1, 5+1) if _ != int(rating_datum['overall'])]
                    overall_idx = random.randint(0, len(overall_candidates)-1) # random choose the overall index for overall_candidates
                    source_text = task_template['source'].format(self.user2id[rating_datum['reviewerID']], self.item2id[rating_datum['asin']], overall_candidates[overall_idx])
                    target_text = task_template['target'].format('no')
            elif task_template['id'] == '1-4':
                source_text = task_template['source'].format(self.user2id[rating_datum['reviewerID']], self.item2id[rating_datum['asin']])
                if int(rating_datum['overall']) >= 4:
                    target_text = task_template['target'].format('like')
                else:
                    target_text = task_template['target'].format('dislike')
            elif task_template['id'] == '1-5':
                if 'name' in self.meta_data[self.meta_dict[rating_datum['asin']]]:
                    title = self.meta_data[self.meta_dict[rating_datum['asin']]]['name']
                else:
                    title = 'unknown name'
                source_text = task_template['source'].format(self.user2id[rating_datum['reviewerID']], self.item2id[rating_datum['asin']], title)
                target_text = task_template['target'].format(self.gaussian_sampling(rating_datum))
            elif task_template['id'] == '1-6':
                if 'name' in self.user_data[self.user_meta_dict[rating_datum['reviewerID']]]:
                    user_desc = self.user_data[self.user_meta_dict[rating_datum['reviewerID']]]['name']
                else:
                    user_desc = rating_datum['reviewerID']
                source_text = task_template['source'].format(user_desc, self.item2id[rating_datum['asin']])
                target_text = task_template['target'].format(self.gaussian_sampling(rating_datum))
            elif task_template['id'] == '1-7':
                if 'name' in self.user_data[self.user_meta_dict[rating_datum['reviewerID']]]:
                    user_desc = self.user_data[self.user_meta_dict[rating_datum['reviewerID']]]['name']
                else:
                    user_desc = rating_datum['reviewerID']
                if 'name' in self.meta_data[self.meta_dict[rating_datum['asin']]]:
                    title = self.meta_data[self.meta_dict[rating_datum['asin']]]['name']
                else:
                    title = 'unknown name'
                source_text = task_template['source'].format(user_desc, title)
                target_text = task_template['target'].format(self.gaussian_sampling(rating_datum))
            elif task_template['id'] == '1-8':
                rand_prob = random.random()
                if 'name' in self.user_data[self.user_meta_dict[rating_datum['reviewerID']]]:
                    user_desc = self.user_data[self.user_meta_dict[rating_datum['reviewerID']]]['name']
                else:
                    user_desc = rating_datum['reviewerID']
                if 'name' in self.meta_data[self.meta_dict[rating_datum['asin']]]:
                    title = self.meta_data[self.meta_dict[rating_datum['asin']]]['name']
                else:
                    title = 'unknown name'
                if rand_prob > 0.5:
                    source_text = task_template['source'].format(user_desc, int(rating_datum['overall']), title)
                    target_text = task_template['target'].format('yes')
                else:
                    overall_candidates = [_ for _ in range(0+1, 5+1) if _ != int(rating_datum['overall'])]
                    overall_idx = random.randint(0, len(overall_candidates)-1) # random choose the overall index for overall_candidates
                    source_text = task_template['source'].format(user_desc, overall_candidates[overall_idx], title)
                    target_text = task_template['target'].format('no')
            elif task_template['id'] == '1-9':
                if 'name' in self.user_data[self.user_meta_dict[rating_datum['reviewerID']]]:
                    user_desc = self.user_data[self.user_meta_dict[rating_datum['reviewerID']]]['name']
                else:
                    user_desc = rating_datum['reviewerID']
                if 'name' in self.meta_data[self.meta_dict[rating_datum['asin']]]:
                    title = self.meta_data[self.meta_dict[rating_datum['asin']]]['name']
                else:
                    title = 'unknown name'
                source_text = task_template['source'].format(user_desc, title)
                if int(rating_datum['overall']) >= 4:
                    target_text = task_template['target'].format('like')
                else:
                    target_text = task_template['target'].format('dislike')
            elif task_template['id'] == '1-10':
                if 'name' in self.user_data[self.user_meta_dict[rating_datum['reviewerID']]]:
                    user_desc = self.user_data[self.user_meta_dict[rating_datum['reviewerID']]]['name']
                else:
                    user_desc = rating_datum['reviewerID']
                if 'name' in self.meta_data[self.meta_dict[rating_datum['asin']]]:
                    title = self.meta_data[self.meta_dict[rating_datum['asin']]]['name']
                else:
                    title = 'unknown name'
                source_text = task_template['source'].format(user_desc, title)
                target_text = task_template['target'].format(self.gaussian_sampling(rating_datum))
            else:
                raise NotImplementedError
            
        elif task_name == 'sequential':
            sequential_datum = self.sequential_data[datum_idx]
            sequence = sequential_datum.split()
            user_id = sequence[0]
            user_desc = self.user_id2name[user_id]
            history_limit = 20
            if self.mode == 'train':
                end_candidates = [_ for _ in range(max(2, len(sequence) - 6), len(sequence) - 3)]
                end_index = random.randint(0, len(end_candidates)-1)
                end_pos = end_candidates[end_index]
                start_candidates = [_ for _ in range(1, min(4, end_pos))]
                start_index = random.randint(0, len(start_candidates)-1)
                start_pos = start_candidates[start_index]
                purchase_history = sequence[start_pos:end_pos+1]
                target_item = sequence[end_pos+1]
            elif self.mode == 'val':
                purchase_history = sequence[1:-2]
                target_item = sequence[-2]
            elif self.mode == 'test':
                purchase_history = sequence[1:-1]
                target_item = sequence[-1]
            else:
                raise NotImplementedError
            if len(purchase_history) > history_limit:
                purchase_history = purchase_history[-history_limit:]
            
            task_candidates = self.task_list[task_name]
            task_idx = random.randint(0, len(task_candidates)-1) # random choose the task index for task_candidates
            task_template = self.all_tasks['sequential'][task_candidates[task_idx]]
            assert task_template['task'] == 'sequential'
            
            if task_template['id'] == '2-1':
                rand_prob = random.random()
                if rand_prob > 0.5:
                    source_text = task_template['source'].format(user_id, ' , '.join(purchase_history))
                else:
                    source_text = task_template['source'].format(user_id, ' -> '.join(purchase_history))
                target_text = task_template['target'].format(target_item)
            elif task_template['id'] == '2-2':
                rand_prob = random.random()
                if rand_prob > 0.5:
                    source_text = task_template['source'].format(user_id, ' , '.join(purchase_history))
                else:
                    source_text = task_template['source'].format(user_id, ' -> '.join(purchase_history))
                target_text = task_template['target'].format(target_item)
            elif task_template['id'] == '2-3':
                rand_prob = random.random()
                if rand_prob > 0.5:
                    source_text = task_template['source'].format(user_id, ' , '.join(purchase_history))
                else:
                    source_text = task_template['source'].format(user_id, ' -> '.join(purchase_history))
                target_text = task_template['target'].format(target_item)
            elif task_template['id'] == '2-4':
                rand_prob = random.random()
                if rand_prob > 0.5:
                    source_text = task_template['source'].format(user_desc, ' , '.join(purchase_history))
                else:
                    source_text = task_template['source'].format(user_desc, ' -> '.join(purchase_history))
                target_text = task_template['target'].format(target_item)
            elif task_template['id'] == '2-5':
                rand_prob = random.random()
                if rand_prob > 0.5:
                    source_text = task_template['source'].format(user_desc, ' , '.join(purchase_history))
                else:
                    source_text = task_template['source'].format(user_desc, ' -> '.join(purchase_history))
                target_text = task_template['target'].format(target_item)
            elif task_template['id'] == '2-6':
                rand_prob = random.random()
                if rand_prob > 0.5:
                    source_text = task_template['source'].format(user_desc, ' , '.join(purchase_history))
                else:
                    source_text = task_template['source'].format(user_desc, ' -> '.join(purchase_history))
                target_text = task_template['target'].format(target_item)
            elif task_template['id'] == '2-7' or task_template['id'] == '2-9':
                if self.mode in ['train', 'val']:
                    user_seq = self.user_items[user_id]
                    candidate_samples = []
                    candidate_num = random.randint(79, 99)
                    while len(candidate_samples) < candidate_num:
                        if self.sample_type == 'random':
                            sample_ids = np.random.choice(self.all_item, candidate_num, replace=False)
                        else:
                            sample_ids = np.random.choice(self.all_item, candidate_num, replace=False, p=self.probability)
                        sample_ids = [str(item) for item in sample_ids if item not in user_seq and item not in candidate_samples]
                        candidate_samples.extend(sample_ids)
                    candidate_samples = candidate_samples[:candidate_num]
                elif self.mode == 'test':
                    assert user_id == self.negative_samples[int(user_id)-1].split(' ', 1)[0]
                    candidate_samples = self.negative_samples[int(user_id)-1].split(' ', 1)[1].split(' ')
                else:
                    raise NotImplementedError
                candidate_samples.extend([target_item])
                random.shuffle(candidate_samples)
                rand_prob = random.random()
                if rand_prob > 0.5:
                    source_text = task_template['source'].format(user_id, ' , '.join(purchase_history), ' , '.join(candidate_samples))
                else:
                    source_text = task_template['source'].format(user_id, ' -> '.join(purchase_history), ' , '.join(candidate_samples))
                target_text = task_template['target'].format(target_item)
            elif task_template['id'] == '2-8' or task_template['id'] == '2-10':
                if self.mode in ['train', 'val']:
                    user_seq = self.user_items[user_id]
                    candidate_samples = []
                    candidate_num = random.randint(79, 99)
                    while len(candidate_samples) < candidate_num:
                        if self.sample_type == 'random':
                            sample_ids = np.random.choice(self.all_item, candidate_num, replace=False)
                        else:
                            sample_ids = np.random.choice(self.all_item, candidate_num, replace=False, p=self.probability)
                        sample_ids = [str(item) for item in sample_ids if item not in user_seq and item not in candidate_samples]
                        candidate_samples.extend(sample_ids)
                    candidate_samples = candidate_samples[:candidate_num]
                elif self.mode == 'test':
                    assert user_id == self.negative_samples[int(user_id)-1].split(' ', 1)[0]
                    candidate_samples = self.negative_samples[int(user_id)-1].split(' ', 1)[1].split(' ')
                else:
                    raise NotImplementedError
                candidate_samples.extend([target_item])
                random.shuffle(candidate_samples)
                rand_prob = random.random()
                if rand_prob > 0.5:
                    source_text = task_template['source'].format(user_desc, ' , '.join(purchase_history), ' , '.join(candidate_samples))
                else:
                    source_text = task_template['source'].format(user_desc, ' -> '.join(purchase_history), ' , '.join(candidate_samples))
                target_text = task_template['target'].format(target_item)
            elif task_template['id'] == '2-11':
                symbol_prob = random.random()
                if symbol_prob > 0.5:
                    symbol = ' , '
                else:
                    symbol = ' -> '
                rand_prob = random.random()
                if rand_prob > 0.5:
                    source_text = task_template['source'].format(user_id, symbol.join(purchase_history), target_item)
                    target_text = task_template['target'].format('yes')
                else:
                    user_seq = self.user_items[user_id]
                    candidate_samples = []
                    candidate_num = 1
                    while len(candidate_samples) < candidate_num:
                        if self.sample_type == 'random':
                            sample_ids = np.random.choice(self.all_item, candidate_num, replace=False)
                        else:
                            sample_ids = np.random.choice(self.all_item, candidate_num, replace=False, p=self.probability)
                        sample_ids = [str(item) for item in sample_ids if item not in user_seq and item not in candidate_samples]
                        candidate_samples.extend(sample_ids)
                    candidate_samples = candidate_samples[:candidate_num]
                    source_text = task_template['source'].format(user_id, symbol.join(purchase_history), candidate_samples[0])
                    target_text = task_template['target'].format('no')
            elif task_template['id'] == '2-12':
                symbol_prob = random.random()
                if symbol_prob > 0.5:
                    symbol = ' , '
                else:
                    symbol = ' -> '
                rand_prob = random.random()
                if rand_prob > 0.5:
                    source_text = task_template['source'].format(user_desc, symbol.join(purchase_history), target_item)
                    target_text = task_template['target'].format('yes')
                else:
                    user_seq = self.user_items[user_id]
                    candidate_samples = []
                    candidate_num = 1
                    while len(candidate_samples) < candidate_num:
                        if self.sample_type == 'random':
                            sample_ids = np.random.choice(self.all_item, candidate_num, replace=False)
                        else:
                            sample_ids = np.random.choice(self.all_item, candidate_num, replace=False, p=self.probability)
                        sample_ids = [str(item) for item in sample_ids if item not in user_seq and item not in candidate_samples]
                        candidate_samples.extend(sample_ids)
                    candidate_samples = candidate_samples[:candidate_num]
                    source_text = task_template['source'].format(user_desc, symbol.join(purchase_history), candidate_samples[0])
                    target_text = task_template['target'].format('no')
            elif task_template['id'] == '2-13':
                rand_prob = random.random()
                if rand_prob > 0.5:
                    source_text = task_template['source'].format(user_desc, ' , '.join(purchase_history))
                else:
                    source_text = task_template['source'].format(user_desc, ' -> '.join(purchase_history))
                target_text = task_template['target'].format(target_item)
            else:
                raise NotImplementedError
        
        elif task_name == 'explanation':
            exp_datum = self.exp_data[datum_idx]
            task_candidates = self.task_list[task_name]
            task_idx = random.randint(0, len(task_candidates)-1) # random choose the task index for task_candidates
            task_template = self.all_tasks['explanation'][task_candidates[task_idx]]
            assert task_template['task'] == 'explanation'
            
            if task_template['id'] == '3-1':
                if 'name' in self.meta_data[self.meta_dict[exp_datum['asin']]]:
                    title = self.meta_data[self.meta_dict[exp_datum['asin']]]['name']
                else:
                    title = 'unknown name'
                source_text = task_template['source'].format(self.user2id[exp_datum['reviewerID']], title)
                target_text = task_template['target'].format(exp_datum['explanation'])
            elif task_template['id'] == '3-2':
                if 'name' in self.meta_data[self.meta_dict[exp_datum['asin']]]:
                    title = self.meta_data[self.meta_dict[exp_datum['asin']]]['name']
                else:
                    title = 'unknown name'
                source_text = task_template['source'].format(self.user2id[exp_datum['reviewerID']], int(exp_datum['overall']), title)
                target_text = task_template['target'].format(exp_datum['explanation'])
            elif task_template['id'] == '3-3':
                if 'name' in self.user_data[self.user_meta_dict[exp_datum['reviewerID']]]:
                    user_desc = self.user_data[self.user_meta_dict[exp_datum['reviewerID']]]['name']
                else:
                    user_desc = exp_datum['reviewerID']
                if 'name' in self.meta_data[self.meta_dict[exp_datum['asin']]]:
                    title = self.meta_data[self.meta_dict[exp_datum['asin']]]['name']
                else:
                    title = 'unknown name'
                source_text = task_template['source'].format(user_desc, title)
                target_text = task_template['target'].format(exp_datum['explanation'])
            elif task_template['id'] == '3-4':
                if 'name' in self.user_data[self.user_meta_dict[exp_datum['reviewerID']]]:
                    user_desc = self.user_data[self.user_meta_dict[exp_datum['reviewerID']]]['name']
                else:
                    user_desc = exp_datum['reviewerID']
                source_text = task_template['source'].format(user_desc, int(exp_datum['overall']), self.item2id[exp_datum['asin']])
                target_text = task_template['target'].format(exp_datum['explanation'])
            elif task_template['id'] == '3-5':
                source_text = task_template['source'].format(exp_datum['feature'], self.user2id[exp_datum['reviewerID']], self.item2id[exp_datum['asin']])
                target_text = task_template['target'].format(self.gaussian_sampling(exp_datum), exp_datum['explanation'])
            elif task_template['id'] == '3-6':
                if 'name' in self.user_data[self.user_meta_dict[exp_datum['reviewerID']]]:
                    user_desc = self.user_data[self.user_meta_dict[exp_datum['reviewerID']]]['name']
                else:
                    user_desc = exp_datum['reviewerID']
                source_text = task_template['source'].format(user_desc, self.item2id[exp_datum['asin']])
                target_text = task_template['target'].format(self.gaussian_sampling(exp_datum), exp_datum['explanation'])
            elif task_template['id'] == '3-7':
                if 'name' in self.meta_data[self.meta_dict[exp_datum['asin']]]:
                    title = self.meta_data[self.meta_dict[exp_datum['asin']]]['name']
                else:
                    title = 'unknown name'
                source_text = task_template['source'].format(exp_datum['feature'], self.user2id[exp_datum['reviewerID']], title)
                target_text = task_template['target'].format(exp_datum['explanation'])
            elif task_template['id'] == '3-8':
                if 'name' in self.user_data[self.user_meta_dict[exp_datum['reviewerID']]]:
                    user_desc = self.user_data[self.user_meta_dict[exp_datum['reviewerID']]]['name']
                else:
                    user_desc = exp_datum['reviewerID']
                if 'name' in self.meta_data[self.meta_dict[exp_datum['asin']]]:
                    title = self.meta_data[self.meta_dict[exp_datum['asin']]]['name']
                else:
                    title = 'unknown name'
                source_text = task_template['source'].format(exp_datum['feature'], user_desc, title)
                target_text = task_template['target'].format(exp_datum['explanation'])
            elif task_template['id'] == '3-9':
                source_text = task_template['source'].format(exp_datum['feature'], int(exp_datum['overall']), self.user2id[exp_datum['reviewerID']], self.item2id[exp_datum['asin']])
                target_text = task_template['target'].format(exp_datum['explanation'])
            elif task_template['id'] == '3-10':
                if 'name' in self.user_data[self.user_meta_dict[exp_datum['reviewerID']]]:
                    user_desc = self.user_data[self.user_meta_dict[exp_datum['reviewerID']]]['name']
                else:
                    user_desc = exp_datum['reviewerID']
                source_text = task_template['source'].format(exp_datum['feature'], int(exp_datum['overall']), user_desc, self.item2id[exp_datum['asin']])
                target_text = task_template['target'].format(exp_datum['explanation'])
            else:
                raise NotImplementedError
                
        elif task_name == 'review':
            review_datum = self.review_data[datum_idx]
            task_candidates = self.task_list[task_name]
            task_idx = random.randint(0, len(task_candidates)-1) # random choose the task index for task_candidates
            task_template = self.all_tasks['review'][task_candidates[task_idx]]
            assert task_template['task'] == 'review'
            
            if task_template['id'] == '4-1':
                source_text = task_template['source'].format(self.user2id[review_datum['reviewerID']], review_datum['reviewText'])
                target_text = task_template['target'].format(int(review_datum['overall']))
            elif task_template['id'] == '4-2':
                source_text = task_template['source'].format(self.user2id[review_datum['reviewerID']], review_datum['reviewText'])
                target_text = task_template['target'].format(int(review_datum['overall']))
            elif task_template['id'] == '4-3':
                if 'name' in self.user_data[self.user_meta_dict[review_datum['reviewerID']]]:
                    user_desc = self.user_data[self.user_meta_dict[review_datum['reviewerID']]]['name']
                else:
                    user_desc = review_datum['reviewerID']
                source_text = task_template['source'].format(user_desc, review_datum['reviewText'])
                target_text = task_template['target'].format(int(review_datum['overall']))
            else:
                raise NotImplementedError
            
        elif task_name == 'traditional':
            sequential_datum = self.sequential_data[datum_idx]
            sequence = sequential_datum.split()
            user_id = sequence[0]
            user_desc = self.user_id2name[user_id]
            if self.mode == 'train':
                target_candidates = sequence[1:-2]
                target_idx = random.randint(0, len(target_candidates)-1) # random choose the target index for target_candidates
                target_item = target_candidates[target_idx]
            elif self.mode == 'val':
                target_item = sequence[-2]
            elif self.mode == 'test':
                target_item = sequence[-1]
            else:
                raise NotImplementedError
            
            task_candidates = self.task_list[task_name]
            task_idx = random.randint(0, len(task_candidates)-1) # random choose the task index for task_candidates
            task_template = self.all_tasks['traditional'][task_candidates[task_idx]]
            assert task_template['task'] == 'traditional'
            
            if task_template['id'] == '5-1':
                rand_prob = random.random()
                if rand_prob > 0.5:
                    source_text = task_template['source'].format(user_id, target_item)
                    target_text = task_template['target'].format('yes')
                else:
                    user_seq = self.user_items[user_id]
                    candidate_samples = []
                    candidate_num = 1
                    while len(candidate_samples) < candidate_num:
                        if self.sample_type == 'random':
                            sample_ids = np.random.choice(self.all_item, candidate_num, replace=False)
                        else:
                            sample_ids = np.random.choice(self.all_item, candidate_num, replace=False, p=self.probability)
                        sample_ids = [str(item) for item in sample_ids if item not in user_seq and item not in candidate_samples]
                        candidate_samples.extend(sample_ids)
                    candidate_samples = candidate_samples[:candidate_num]
                    source_text = task_template['source'].format(user_id, candidate_samples[0])
                    target_text = task_template['target'].format('no')
            elif task_template['id'] == '5-2':
                rand_prob = random.random()
                if rand_prob > 0.5:
                    source_text = task_template['source'].format(target_item, user_desc)
                    target_text = task_template['target'].format('yes')
                else:
                    user_seq = self.user_items[user_id]
                    candidate_samples = []
                    candidate_num = 1
                    while len(candidate_samples) < candidate_num:
                        if self.sample_type == 'random':
                            sample_ids = np.random.choice(self.all_item, candidate_num, replace=False)
                        else:
                            sample_ids = np.random.choice(self.all_item, candidate_num, replace=False, p=self.probability)
                        sample_ids = [str(item) for item in sample_ids if item not in user_seq and item not in candidate_samples]
                        candidate_samples.extend(sample_ids)
                    candidate_samples = candidate_samples[:candidate_num]
                    source_text = task_template['source'].format(candidate_samples[0], user_desc)
                    target_text = task_template['target'].format('no')
            elif task_template['id'] == '5-3':
                rand_prob = random.random()
                if rand_prob > 0.5:
                    if 'name' in self.meta_data[self.meta_dict[self.id2item[target_item]]]:
                        title = self.meta_data[self.meta_dict[self.id2item[target_item]]]['name']
                    else:
                        title = 'unknown name'
                    source_text = task_template['source'].format(user_desc, title)
                    target_text = task_template['target'].format('yes')
                else:
                    user_seq = self.user_items[user_id]
                    candidate_samples = []
                    candidate_num = 1
                    while len(candidate_samples) < candidate_num:
                        if self.sample_type == 'random':
                            sample_ids = np.random.choice(self.all_item, candidate_num, replace=False)
                        else:
                            sample_ids = np.random.choice(self.all_item, candidate_num, replace=False, p=self.probability)
                        sample_ids = [str(item) for item in sample_ids if item not in user_seq and item not in candidate_samples]
                        candidate_samples.extend(sample_ids)
                    candidate_samples = candidate_samples[:candidate_num]
                    if 'name' in self.meta_data[self.meta_dict[self.id2item[candidate_samples[0]]]]:
                        title = self.meta_data[self.meta_dict[self.id2item[candidate_samples[0]]]]['name']
                    else:
                        title = 'unknown name'
                    source_text = task_template['source'].format(user_desc, title)
                    target_text = task_template['target'].format('no')
            elif task_template['id'] == '5-4':
                rand_prob = random.random()
                if rand_prob > 0.5:
                    if 'name' in self.meta_data[self.meta_dict[self.id2item[target_item]]]:
                        title = self.meta_data[self.meta_dict[self.id2item[target_item]]]['name']
                    else:
                        title = 'unknown name'
                    source_text = task_template['source'].format(user_id, title)
                    target_text = task_template['target'].format('yes')
                else:
                    user_seq = self.user_items[user_id]
                    candidate_samples = []
                    candidate_num = 1
                    while len(candidate_samples) < candidate_num:
                        if self.sample_type == 'random':
                            sample_ids = np.random.choice(self.all_item, candidate_num, replace=False)
                        else:
                            sample_ids = np.random.choice(self.all_item, candidate_num, replace=False, p=self.probability)
                        sample_ids = [str(item) for item in sample_ids if item not in user_seq and item not in candidate_samples]
                        candidate_samples.extend(sample_ids)
                    candidate_samples = candidate_samples[:candidate_num]
                    if 'name' in self.meta_data[self.meta_dict[self.id2item[candidate_samples[0]]]]:
                        title = self.meta_data[self.meta_dict[self.id2item[candidate_samples[0]]]]['name']
                    else:
                        title = 'unknown name'
                    source_text = task_template['source'].format(user_id, title)
                    target_text = task_template['target'].format('no')
            elif task_template['id'] == '5-5' or task_template['id'] == '5-6':
                user_seq = self.user_items[user_id]
                candidate_samples = []
                candidate_num = 99 # random.randint(19, 99)
                while len(candidate_samples) < candidate_num:
                    if self.sample_type == 'random':
                        sample_ids = np.random.choice(self.all_item, candidate_num, replace=False)
                    else:
                        sample_ids = np.random.choice(self.all_item, candidate_num, replace=False, p=self.probability)
                    sample_ids = [str(item) for item in sample_ids if item not in user_seq and item not in candidate_samples]
                    candidate_samples.extend(sample_ids)
                candidate_samples = candidate_samples[:candidate_num]
                candidate_samples.extend([target_item])
                random.shuffle(candidate_samples)
                source_text = task_template['source'].format(user_desc, ' , '.join(candidate_samples))
                target_text = task_template['target'].format(target_item)
            elif task_template['id'] == '5-7' or task_template['id'] == '5-8':
                user_seq = self.user_items[user_id]
                candidate_samples = []
                candidate_num = 99 # random.randint(19, 99)
                while len(candidate_samples) < candidate_num:
                    if self.sample_type == 'random':
                        sample_ids = np.random.choice(self.all_item, candidate_num, replace=False)
                    else:
                        sample_ids = np.random.choice(self.all_item, candidate_num, replace=False, p=self.probability)
                    sample_ids = [str(item) for item in sample_ids if item not in user_seq and item not in candidate_samples]
                    candidate_samples.extend(sample_ids)
                candidate_samples = candidate_samples[:candidate_num]
                candidate_samples.extend([target_item])
                random.shuffle(candidate_samples)
                source_text = task_template['source'].format(user_id, ' , '.join(candidate_samples))
                target_text = task_template['target'].format(target_item)
            else:
                raise NotImplementedError
            
        else:
            raise NotImplementedError
            
        input_ids = self.tokenizer.encode(
                source_text, padding=True, truncation=True, max_length=self.args.max_text_length)
        tokenized_text = self.tokenizer.tokenize(source_text)
        whole_word_ids = self.calculate_whole_word_ids(tokenized_text, input_ids)
        assert len(whole_word_ids) == len(input_ids)
        
        target_ids = self.tokenizer.encode(
                target_text, padding=True, truncation=True, max_length=self.args.gen_max_length)

        out_dict['input_ids'] = torch.LongTensor(input_ids)
        out_dict['input_length'] = len(input_ids)
        out_dict['whole_word_ids'] = torch.LongTensor(whole_word_ids)
        out_dict['target_ids'] = torch.LongTensor(target_ids)
        out_dict['target_length'] = len(target_ids)

        out_dict['source_text'] = source_text
        out_dict['tokenized_text'] = tokenized_text
        out_dict['target_text'] = target_text

        out_dict['task'] = task_template['task']

        out_dict['loss_weight'] = loss_weight

        return out_dict
    
    def calculate_whole_word_ids(self, tokenized_text, input_ids):
        whole_word_ids = []
        curr = 0
        for i in range(len(tokenized_text)):
            if tokenized_text[i].startswith('▁'):
                curr += 1
                whole_word_ids.append(curr)
            else:
                whole_word_ids.append(curr)
        last_item = whole_word_ids[len(input_ids) - 2]
        return whole_word_ids[:len(input_ids) - 1] + [0] # [0] for </s>
    
    def collate_fn(self, batch):
        batch_entry = {}

        B = len(batch)

        args = self.args

        S_W_L = max(entry['input_length'] for entry in batch)
        T_W_L = max(entry['target_length'] for entry in batch)

        input_ids = torch.ones(B, S_W_L, dtype=torch.long) * self.tokenizer.pad_token_id
        whole_word_ids = torch.ones(B, S_W_L, dtype=torch.long) * self.tokenizer.pad_token_id
        target_ids = torch.ones(B, T_W_L, dtype=torch.long) * self.tokenizer.pad_token_id

        loss_weights = torch.ones(B, dtype=torch.float)

        tasks = []
        source_text = []
        tokenized_text = []
        target_text = []

        for i, entry in enumerate(batch):
            input_ids[i, :entry['input_length']] = entry['input_ids']
            whole_word_ids[i, :entry['input_length']] = entry['whole_word_ids']
            target_ids[i, :entry['target_length']] = entry['target_ids']

            if 'task' in entry:
                tasks.append(entry['task'])

            if 'source_text' in entry:
                source_text.append(entry['source_text'])
                
            if 'tokenized_text' in entry:
                tokenized_text.append(entry['tokenized_text'])
                
            if 'target_text' in entry:
                target_text.append(entry['target_text'])

            if 'loss_weight' in entry:
                loss_weights[i] = entry['loss_weight']

        assert 't5' in args.backbone
        word_mask = target_ids != self.tokenizer.pad_token_id
        target_ids[~word_mask] = -100
        batch_entry['task'] = tasks

        batch_entry['source_text'] = source_text
        batch_entry['target_text'] = target_text

        batch_entry['input_ids'] = input_ids
        batch_entry['whole_word_ids'] = whole_word_ids
        batch_entry['target_ids'] = target_ids

        batch_entry['loss_weights'] = loss_weights

        return batch_entry
    

MOVIELENS_DATASETS = {'ml-1m', 'ml-20m', 'netflix', 'douban_monti'}


def load_raw_movielens_data(split, data_path='data', min_rating=4.0, train_ratio=0.8, val_ratio=0.1, seed=1234):
    """
    Load raw data from MovieLens, Netflix, or Douban_Monti files and convert to P5 format.

    Returns:
        rating_data: dict with 'train', 'val', 'test' splits, each a list of
                     {'reviewerID': str, 'asin': str, 'overall': float}
        user2id: dict mapping user string to user index
        item2id: dict mapping item string to item index
        id2item: dict mapping item index to item string
        user_items: dict mapping user to list of item indices (for sequential)
        sequential_data: list of "user_id item1 item2 ..." strings
        item_titles: dict mapping item_id to title (if available)
    """
    import pandas as pd
    from scipy import sparse
    import h5py
    from scipy.sparse import csc_matrix

    np.random.seed(seed)

    # Map split names to folder names (handle case differences)
    split_lower = split.lower()
    folder_map = {
        'ml-1m': 'ML-1M',
        'ml-20m': 'ML-20M',
        'netflix': 'Netflix',
        'douban_monti': 'Douban_monti'
    }
    folder_name = folder_map.get(split_lower, split)
    path = os.path.join(data_path, folder_name)

    print(f'Loading raw data from {path}...')

    # Load data based on dataset type
    if split_lower == 'ml-1m':
        # MovieLens 1M: movielens_1m_dataset.dat with :: delimiter
        data_file = os.path.join(path, 'movielens_1m_dataset.dat')
        if os.path.exists(data_file):
            data = np.genfromtxt(data_file, delimiter='::')
        else:
            # Try ratings.dat format
            data_file = os.path.join(path, 'ratings.dat')
            data = np.genfromtxt(data_file, delimiter='::')
        # Format: user_id, movie_id, rating, timestamp
        df = pd.DataFrame(data[:, :3], columns=['userId', 'movieId', 'rating'])

    elif split_lower == 'ml-20m':
        # MovieLens 20M: ratings.csv
        data_file = os.path.join(path, 'ratings.csv')
        df = pd.read_csv(data_file, usecols=['userId', 'movieId', 'rating'])

    elif split_lower == 'netflix':
        # Netflix: combined_data_*.txt files
        dfs = []
        for i in range(1, 5):
            file_path = os.path.join(path, f'combined_data_{i}.txt')
            if os.path.exists(file_path):
                temp_df = pd.read_csv(file_path, header=None, names=['userId', 'rating'], usecols=[0, 1])
                temp_df['rating'] = temp_df['rating'].astype(float)
                dfs.append(temp_df)

        data = pd.concat(dfs, ignore_index=True)
        data_nan = pd.DataFrame(pd.isnull(data.rating))
        data_nan = data_nan[data_nan['rating'] == True].reset_index()

        # Extract movie IDs
        diff = np.diff(data_nan['index'])
        movie_ids = np.arange(1, len(diff) + 1)
        movie_np = np.repeat(movie_ids, diff - 1)
        last_record = np.full(len(data) - data_nan.iloc[-1, 0] - 1, len(diff) + 1)
        movie_np = np.concatenate([movie_np, last_record])

        data = data[pd.notnull(data['rating'])]
        data['movieId'] = movie_np.astype(int)
        data['userId'] = data['userId'].astype(int)
        df = data[['userId', 'movieId', 'rating']].copy()

    elif split_lower == 'douban_monti':
        # Douban Monti: MATLAB file
        mat_file = os.path.join(path, 'douban_monti_dataset.mat')

        def load_matlab_field(path_file, name_field):
            db = h5py.File(path_file, 'r')
            ds = db[name_field]
            try:
                if 'ir' in ds.keys():
                    data = np.asarray(ds['data'])
                    ir = np.asarray(ds['ir'])
                    jc = np.asarray(ds['jc'])
                    out = csc_matrix((data, ir, jc)).astype(np.float32)
                else:
                    out = np.asarray(ds).astype(np.float32).T
            except AttributeError:
                out = np.asarray(ds).astype(np.float32).T
            db.close()
            return out

        M = load_matlab_field(mat_file, 'M')
        # M is users x movies matrix

        # Convert matrix to dataframe using sparse-friendly extraction
        if sparse.issparse(M):
            # Use sparse.find() to efficiently extract non-zero entries
            users, movies, ratings = sparse.find(M)
        else:
            users, movies = np.where(M > 0)
            ratings = M[users, movies]

        df = pd.DataFrame({
            'userId': users + 1,  # 1-indexed
            'movieId': movies + 1,  # 1-indexed
            'rating': ratings
        })
    else:
        raise ValueError(f'Unknown dataset: {split}')

    print(f'Loaded {len(df)} ratings')

    # Filter by minimum rating
    if min_rating > 0:
        original_len = len(df)
        df = df[df['rating'] >= min_rating].copy()
        print(f'Filtered ratings >= {min_rating}: {original_len} -> {len(df)}')

    # Create user and item mappings
    unique_users = sorted(df['userId'].unique())
    unique_items = sorted(df['movieId'].unique())

    user2id = {str(int(u)): str(i) for i, u in enumerate(unique_users)}
    item2id = {str(int(m)): str(i) for i, m in enumerate(unique_items)}
    id2item = {str(i): str(int(m)) for i, m in enumerate(unique_items)}

    print(f'Users: {len(user2id)}, Items: {len(item2id)}')

    # Shuffle and split data
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)
    n = len(df)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))

    train_df = df.iloc[:train_end]
    val_df = df.iloc[train_end:val_end]
    test_df = df.iloc[val_end:]

    print(f'Split: train={len(train_df)}, val={len(val_df)}, test={len(test_df)}')

    # Convert to P5 rating format
    def df_to_rating_data(sub_df):
        return [
            {
                'reviewerID': str(int(row['userId'])),
                'asin': str(int(row['movieId'])),
                'overall': float(row['rating'])
            }
            for _, row in sub_df.iterrows()
        ]

    rating_data = {
        'train': df_to_rating_data(train_df),
        'val': df_to_rating_data(val_df),
        'test': df_to_rating_data(test_df)
    }

    # Build user_items for sequential recommendations (using all data)
    user_items = defaultdict(list)
    for _, row in df.iterrows():
        user_str = str(int(row['userId']))
        item_mapped = item2id[str(int(row['movieId']))]
        user_items[user2id[user_str]].append(int(item_mapped))

    # Create sequential_data format: "user_id item1 item2 ..."
    sequential_data = []
    for user_id in sorted(user_items.keys(), key=int):
        items = user_items[user_id]
        if len(items) >= 3:  # Need at least 3 items for sequential tasks
            sequential_data.append(f"{user_id} " + " ".join(str(i) for i in items))

    # Load movie titles if available
    item_titles = {}
    if split_lower == 'ml-1m':
        movies_file = os.path.join(path, 'movies.dat')
        if os.path.exists(movies_file):
            with open(movies_file, 'r', encoding='latin-1') as f:
                for line in f:
                    parts = line.strip().split('::')
                    if len(parts) >= 2:
                        movie_id = str(int(parts[0]))
                        if movie_id in item2id:
                            item_titles[item2id[movie_id]] = parts[1]
    elif split_lower == 'ml-20m':
        movies_file = os.path.join(path, 'movies.csv')
        if os.path.exists(movies_file):
            movies_df = pd.read_csv(movies_file)
            for _, row in movies_df.iterrows():
                movie_id = str(int(row['movieId']))
                if movie_id in item2id:
                    item_titles[item2id[movie_id]] = row['title']
    elif split_lower == 'netflix':
        movies_file = os.path.join(path, 'movie_titles.csv')
        if os.path.exists(movies_file):
            try:
                movies_df = pd.read_csv(movies_file, encoding='latin-1', header=None,
                                       names=['movieId', 'year', 'title'])
                for _, row in movies_df.iterrows():
                    movie_id = str(int(row['movieId']))
                    if movie_id in item2id:
                        item_titles[item2id[movie_id]] = str(row['title'])
            except:
                pass

    return rating_data, user2id, item2id, id2item, dict(user_items), sequential_data, item_titles


class P5_MovieLens_Dataset(Dataset):
    """
    Dataset class for MovieLens, Netflix, and Douban_Monti datasets.
    These are interaction-heavy, text-light datasets that support:
    - Task Family 1: Rating prediction
    - Task Family 2: Sequential recommendation
    - Task Family 5: Direct (traditional) recommendation

    Unlike Amazon/Yelp datasets, these do NOT have review text or explanations.
    Movie titles are used as item descriptions where available.

    Loads data directly from raw files (DAT, CSV, MAT) instead of preprocessed pickles.
    """
    def __init__(self, all_tasks, task_list, tokenizer, args, sample_numbers,
                 mode='train', split='ml-1m', rating_augment=False, sample_type='random',
                 min_rating=4.0):
        self.all_tasks = all_tasks
        self.task_list = task_list
        self.tokenizer = tokenizer
        self.args = args
        self.sample_numbers = sample_numbers
        self.split = split
        self.rating_augment = rating_augment
        self.sample_type = sample_type
        self.min_rating = min_rating
        self.mode = mode

        print('Data sources: ', split.split(','))
        print(f'Filtering ratings >= {min_rating}')

        # Load data from raw files
        (all_rating_data, self.user2id, self.item2id, self.id2item,
         self.user_items, self.sequential_data, self.item_id2title) = load_raw_movielens_data(
            split, data_path='data', min_rating=min_rating
        )

        # Select the appropriate split
        if self.mode == 'train':
            self.rating_data = all_rating_data['train']
        elif self.mode == 'val':
            self.rating_data = all_rating_data['val']
        elif self.mode == 'test':
            self.rating_data = all_rating_data['test']
        else:
            raise NotImplementedError

        self.review_data = self.rating_data  # For compatibility

        print(f'Mode: {mode}, Rating data size: {len(self.rating_data)}')

        # Build item statistics from sequential data
        item_count = defaultdict(int)
        user_items_parsed = defaultdict()

        for line in self.sequential_data:
            parts = line.strip().split(' ', 1)
            if len(parts) < 2:
                continue
            user, items_str = parts
            items = items_str.split(' ')
            items = [int(item) for item in items]
            user_items_parsed[user] = items
            for item in items:
                item_count[item] += 1

        self.all_item = list(item_count.keys())
        count = list(item_count.values())
        sum_value = np.sum([x for x in count]) if count else 1
        self.probability = [value / sum_value for value in count] if count else []
        self.user_items = user_items_parsed if user_items_parsed else self.user_items

        # Generate negative samples for test mode
        if self.mode == 'test':
            self.negative_samples = self._generate_negative_samples()

        self.user_list = list(self.user2id.keys())
        self.item_list = list(self.item2id.keys())

        # User ID to name mapping (simple numeric for these datasets)
        self.user_id2name = {uid: f'user_{uid}' for uid in self.user2id.values()}

        print('compute_datum_info')
        self.total_length = 0
        self.datum_info = []
        self.compute_datum_info()

    def _generate_negative_samples(self, num_negatives=99):
        """Generate negative samples for test evaluation."""
        negative_samples = []
        all_items_set = set(self.all_item)

        for user_id in sorted(self.user_items.keys(), key=lambda x: int(x) if x.isdigit() else 0):
            user_seq = set(self.user_items.get(user_id, []))
            available = list(all_items_set - user_seq)
            if len(available) >= num_negatives:
                neg_items = np.random.choice(available, num_negatives, replace=False)
            else:
                neg_items = available
            negative_samples.append(f"{user_id} " + " ".join(str(i) for i in neg_items))

        return negative_samples

    def compute_datum_info(self):
        curr = 0
        for key in list(self.task_list.keys()):
            if key == 'rating':
                self.total_length += len(self.rating_data) * self.sample_numbers[key]
                for i in range(self.total_length - curr):
                    self.datum_info.append((i + curr, key, i // self.sample_numbers[key]))
                curr = self.total_length
            elif key == 'sequential':
                if sum([0 < int(ind.split('-')[1]) <= 6 or int(ind.split('-')[1]) == 13 for ind in self.task_list[key]]):
                    self.total_length += len(self.sequential_data) * self.sample_numbers[key][0]
                    for i in range(self.total_length - curr):
                        self.datum_info.append((i + curr, key, i // self.sample_numbers[key][0]))
                    curr = self.total_length
                if sum([6 < int(ind.split('-')[1]) <= 10 for ind in self.task_list[key]]):
                    self.total_length += len(self.sequential_data) * self.sample_numbers[key][1]
                    for i in range(self.total_length - curr):
                        self.datum_info.append((i + curr, key, i // self.sample_numbers[key][1]))
                    curr = self.total_length
                if sum([10 < int(ind.split('-')[1]) <= 12 for ind in self.task_list[key]]):
                    self.total_length += len(self.sequential_data) * self.sample_numbers[key][2]
                    for i in range(self.total_length - curr):
                        self.datum_info.append((i + curr, key, i // self.sample_numbers[key][2]))
                    curr = self.total_length
            elif key == 'traditional':
                if sum([0 < int(ind.split('-')[1]) <= 4 for ind in self.task_list[key]]):
                    self.total_length += len(self.user2id) * self.sample_numbers[key][0]
                    for i in range(self.total_length - curr):
                        self.datum_info.append((i + curr, key, i // self.sample_numbers[key][0]))
                    curr = self.total_length
                if sum([4 < int(ind.split('-')[1]) <= 8 for ind in self.task_list[key]]):
                    self.total_length += len(self.user2id) * self.sample_numbers[key][1]
                    for i in range(self.total_length - curr):
                        self.datum_info.append((i + curr, key, i // self.sample_numbers[key][1]))
                    curr = self.total_length
            else:
                raise NotImplementedError

    def gaussian_sampling(self, datum):
        if self.mode == 'train':
            if int(datum['overall']) == 1:
                sampled_rating = round(torch.normal(mean=torch.tensor((1.0+1.4)/2), std=torch.tensor((1.4-1.0)/4)).item(), 1)
            elif int(datum['overall']) == 2:
                sampled_rating = round(torch.normal(mean=torch.tensor((1.5+2.4)/2), std=torch.tensor((2.4-1.5)/4)).item(), 1)
            elif int(datum['overall']) == 3:
                sampled_rating = round(torch.normal(mean=torch.tensor((2.5+3.4)/2), std=torch.tensor((3.4-2.5)/4)).item(), 1)
            elif int(datum['overall']) == 4:
                sampled_rating = round(torch.normal(mean=torch.tensor((3.5+4.4)/2), std=torch.tensor((4.4-3.5)/4)).item(), 1)
            else:
                sampled_rating = round(torch.normal(mean=torch.tensor((4.5+5.0)/2), std=torch.tensor((5.0-4.5)/4)).item(), 1)
            if sampled_rating > 5.0:
                sampled_rating = 5.0
            if sampled_rating < 1.0:
                sampled_rating = 1.0
            return str(sampled_rating)
        else:
            return int(datum['overall'])

    def _get_title(self, item_id):
        """Get movie title for an item_id (mapped ID as string)."""
        if item_id in self.item_id2title:
            return self.item_id2title[item_id]
        return f'item_{item_id}'

    def __len__(self):
        return self.total_length

    def __getitem__(self, idx):

        out_dict = {}
        out_dict['args'] = self.args

        loss_weight = 1.0

        datum_info_idx = self.datum_info[idx]
        assert datum_info_idx[0] == idx
        if len(datum_info_idx) == 3:
            task_name = datum_info_idx[1]
            datum_idx = datum_info_idx[2]
        elif len(datum_info_idx) == 4:
            task_name = datum_info_idx[1]
            datum_idx = datum_info_idx[2]
            task_idx = datum_info_idx[3]
        else:
            raise NotImplementedError

        if task_name == 'rating':
            rating_datum = self.rating_data[datum_idx]
            task_candidates = self.task_list[task_name]
            task_idx = random.randint(0, len(task_candidates)-1)
            task_template = self.all_tasks['rating'][task_candidates[task_idx]]
            assert task_template['task'] == 'rating'

            user_id = self.user2id[rating_datum['reviewerID']]
            item_id = self.item2id[rating_datum['asin']]
            title = self._get_title(item_id)
            user_desc = self.user_id2name.get(user_id, f'user_{user_id}')

            if task_template['id'] == '1-1':
                source_text = task_template['source'].format(user_id, item_id)
                target_text = task_template['target'].format(self.gaussian_sampling(rating_datum))
            elif task_template['id'] == '1-2':
                source_text = task_template['source'].format(user_id, title)
                target_text = task_template['target'].format(self.gaussian_sampling(rating_datum))
            elif task_template['id'] == '1-3':
                rand_prob = random.random()
                if rand_prob > 0.5:
                    source_text = task_template['source'].format(user_id, item_id, int(rating_datum['overall']))
                    target_text = task_template['target'].format('yes')
                else:
                    overall_candidates = [_ for _ in range(1, 6) if _ != int(rating_datum['overall'])]
                    overall_idx = random.randint(0, len(overall_candidates)-1)
                    source_text = task_template['source'].format(user_id, item_id, overall_candidates[overall_idx])
                    target_text = task_template['target'].format('no')
            elif task_template['id'] == '1-4':
                source_text = task_template['source'].format(user_id, item_id)
                if int(rating_datum['overall']) >= 4:
                    target_text = task_template['target'].format('like')
                else:
                    target_text = task_template['target'].format('dislike')
            elif task_template['id'] == '1-5':
                source_text = task_template['source'].format(user_id, item_id, title)
                target_text = task_template['target'].format(self.gaussian_sampling(rating_datum))
            elif task_template['id'] == '1-6':
                source_text = task_template['source'].format(user_desc, item_id)
                target_text = task_template['target'].format(self.gaussian_sampling(rating_datum))
            elif task_template['id'] == '1-7':
                source_text = task_template['source'].format(user_desc, title)
                target_text = task_template['target'].format(self.gaussian_sampling(rating_datum))
            elif task_template['id'] == '1-8':
                rand_prob = random.random()
                if rand_prob > 0.5:
                    source_text = task_template['source'].format(user_desc, int(rating_datum['overall']), title)
                    target_text = task_template['target'].format('yes')
                else:
                    overall_candidates = [_ for _ in range(1, 6) if _ != int(rating_datum['overall'])]
                    overall_idx = random.randint(0, len(overall_candidates)-1)
                    source_text = task_template['source'].format(user_desc, overall_candidates[overall_idx], title)
                    target_text = task_template['target'].format('no')
            elif task_template['id'] == '1-9':
                source_text = task_template['source'].format(user_desc, title)
                if int(rating_datum['overall']) >= 4:
                    target_text = task_template['target'].format('like')
                else:
                    target_text = task_template['target'].format('dislike')
            elif task_template['id'] == '1-10':
                source_text = task_template['source'].format(user_desc, title)
                target_text = task_template['target'].format(self.gaussian_sampling(rating_datum))
            else:
                raise NotImplementedError

        elif task_name == 'sequential':
            sequential_datum = self.sequential_data[datum_idx]
            sequence = sequential_datum.split()
            user_id = sequence[0]
            user_desc = self.user_id2name.get(user_id, f'user_{user_id}')
            history_limit = 20  # Limit history length for large datasets
            if self.mode == 'train':
                end_candidates = [_ for _ in range(max(2, len(sequence) - 6), len(sequence) - 3)]
                end_index = random.randint(0, len(end_candidates)-1)
                end_pos = end_candidates[end_index]
                start_candidates = [_ for _ in range(1, min(4, end_pos))]
                start_index = random.randint(0, len(start_candidates)-1)
                start_pos = start_candidates[start_index]
                purchase_history = sequence[start_pos:end_pos+1]
                target_item = sequence[end_pos+1]
            elif self.mode == 'val':
                purchase_history = sequence[1:-2]
                target_item = sequence[-2]
            elif self.mode == 'test':
                purchase_history = sequence[1:-1]
                target_item = sequence[-1]
            else:
                raise NotImplementedError
            if len(purchase_history) > history_limit:
                purchase_history = purchase_history[-history_limit:]

            task_candidates = self.task_list[task_name]
            task_idx = random.randint(0, len(task_candidates)-1)
            task_template = self.all_tasks['sequential'][task_candidates[task_idx]]
            assert task_template['task'] == 'sequential'

            if task_template['id'] == '2-1':
                rand_prob = random.random()
                if rand_prob > 0.5:
                    source_text = task_template['source'].format(user_id, ' , '.join(purchase_history))
                else:
                    source_text = task_template['source'].format(user_id, ' -> '.join(purchase_history))
                target_text = task_template['target'].format(target_item)
            elif task_template['id'] == '2-2':
                rand_prob = random.random()
                if rand_prob > 0.5:
                    source_text = task_template['source'].format(user_id, ' , '.join(purchase_history))
                else:
                    source_text = task_template['source'].format(user_id, ' -> '.join(purchase_history))
                target_text = task_template['target'].format(target_item)
            elif task_template['id'] == '2-3':
                rand_prob = random.random()
                if rand_prob > 0.5:
                    source_text = task_template['source'].format(user_id, ' , '.join(purchase_history))
                else:
                    source_text = task_template['source'].format(user_id, ' -> '.join(purchase_history))
                target_text = task_template['target'].format(target_item)
            elif task_template['id'] == '2-4':
                rand_prob = random.random()
                if rand_prob > 0.5:
                    source_text = task_template['source'].format(user_desc, ' , '.join(purchase_history))
                else:
                    source_text = task_template['source'].format(user_desc, ' -> '.join(purchase_history))
                target_text = task_template['target'].format(target_item)
            elif task_template['id'] == '2-5':
                rand_prob = random.random()
                if rand_prob > 0.5:
                    source_text = task_template['source'].format(user_desc, ' , '.join(purchase_history))
                else:
                    source_text = task_template['source'].format(user_desc, ' -> '.join(purchase_history))
                target_text = task_template['target'].format(target_item)
            elif task_template['id'] == '2-6':
                rand_prob = random.random()
                if rand_prob > 0.5:
                    source_text = task_template['source'].format(user_desc, ' , '.join(purchase_history))
                else:
                    source_text = task_template['source'].format(user_desc, ' -> '.join(purchase_history))
                target_text = task_template['target'].format(target_item)
            elif task_template['id'] == '2-7' or task_template['id'] == '2-9':
                if self.mode in ['train', 'val']:
                    user_seq = self.user_items[user_id]
                    candidate_samples = []
                    candidate_num = random.randint(79, 99)
                    while len(candidate_samples) < candidate_num:
                        if self.sample_type == 'random':
                            sample_ids = np.random.choice(self.all_item, candidate_num, replace=False)
                        else:
                            sample_ids = np.random.choice(self.all_item, candidate_num, replace=False, p=self.probability)
                        sample_ids = [str(item) for item in sample_ids if item not in user_seq and item not in candidate_samples]
                        candidate_samples.extend(sample_ids)
                    candidate_samples = candidate_samples[:candidate_num]
                elif self.mode == 'test':
                    assert user_id == self.negative_samples[int(user_id)-1].split(' ', 1)[0]
                    candidate_samples = self.negative_samples[int(user_id)-1].split(' ', 1)[1].split(' ')
                else:
                    raise NotImplementedError
                candidate_samples.extend([target_item])
                random.shuffle(candidate_samples)
                rand_prob = random.random()
                if rand_prob > 0.5:
                    source_text = task_template['source'].format(user_id, ' , '.join(purchase_history), ' , '.join(candidate_samples))
                else:
                    source_text = task_template['source'].format(user_id, ' -> '.join(purchase_history), ' , '.join(candidate_samples))
                target_text = task_template['target'].format(target_item)
            elif task_template['id'] == '2-8' or task_template['id'] == '2-10':
                if self.mode in ['train', 'val']:
                    user_seq = self.user_items[user_id]
                    candidate_samples = []
                    candidate_num = random.randint(79, 99)
                    while len(candidate_samples) < candidate_num:
                        if self.sample_type == 'random':
                            sample_ids = np.random.choice(self.all_item, candidate_num, replace=False)
                        else:
                            sample_ids = np.random.choice(self.all_item, candidate_num, replace=False, p=self.probability)
                        sample_ids = [str(item) for item in sample_ids if item not in user_seq and item not in candidate_samples]
                        candidate_samples.extend(sample_ids)
                    candidate_samples = candidate_samples[:candidate_num]
                elif self.mode == 'test':
                    assert user_id == self.negative_samples[int(user_id)-1].split(' ', 1)[0]
                    candidate_samples = self.negative_samples[int(user_id)-1].split(' ', 1)[1].split(' ')
                else:
                    raise NotImplementedError
                candidate_samples.extend([target_item])
                random.shuffle(candidate_samples)
                rand_prob = random.random()
                if rand_prob > 0.5:
                    source_text = task_template['source'].format(user_desc, ' , '.join(purchase_history), ' , '.join(candidate_samples))
                else:
                    source_text = task_template['source'].format(user_desc, ' -> '.join(purchase_history), ' , '.join(candidate_samples))
                target_text = task_template['target'].format(target_item)
            elif task_template['id'] == '2-11':
                symbol_prob = random.random()
                if symbol_prob > 0.5:
                    symbol = ' , '
                else:
                    symbol = ' -> '
                rand_prob = random.random()
                if rand_prob > 0.5:
                    source_text = task_template['source'].format(user_id, symbol.join(purchase_history), target_item)
                    target_text = task_template['target'].format('yes')
                else:
                    user_seq = self.user_items[user_id]
                    candidate_samples = []
                    candidate_num = 1
                    while len(candidate_samples) < candidate_num:
                        if self.sample_type == 'random':
                            sample_ids = np.random.choice(self.all_item, candidate_num, replace=False)
                        else:
                            sample_ids = np.random.choice(self.all_item, candidate_num, replace=False, p=self.probability)
                        sample_ids = [str(item) for item in sample_ids if item not in user_seq and item not in candidate_samples]
                        candidate_samples.extend(sample_ids)
                    candidate_samples = candidate_samples[:candidate_num]
                    source_text = task_template['source'].format(user_id, symbol.join(purchase_history), candidate_samples[0])
                    target_text = task_template['target'].format('no')
            elif task_template['id'] == '2-12':
                symbol_prob = random.random()
                if symbol_prob > 0.5:
                    symbol = ' , '
                else:
                    symbol = ' -> '
                rand_prob = random.random()
                if rand_prob > 0.5:
                    source_text = task_template['source'].format(user_desc, symbol.join(purchase_history), target_item)
                    target_text = task_template['target'].format('yes')
                else:
                    user_seq = self.user_items[user_id]
                    candidate_samples = []
                    candidate_num = 1
                    while len(candidate_samples) < candidate_num:
                        if self.sample_type == 'random':
                            sample_ids = np.random.choice(self.all_item, candidate_num, replace=False)
                        else:
                            sample_ids = np.random.choice(self.all_item, candidate_num, replace=False, p=self.probability)
                        sample_ids = [str(item) for item in sample_ids if item not in user_seq and item not in candidate_samples]
                        candidate_samples.extend(sample_ids)
                    candidate_samples = candidate_samples[:candidate_num]
                    source_text = task_template['source'].format(user_desc, symbol.join(purchase_history), candidate_samples[0])
                    target_text = task_template['target'].format('no')
            elif task_template['id'] == '2-13':
                rand_prob = random.random()
                if rand_prob > 0.5:
                    source_text = task_template['source'].format(user_desc, ' , '.join(purchase_history))
                else:
                    source_text = task_template['source'].format(user_desc, ' -> '.join(purchase_history))
                target_text = task_template['target'].format(target_item)
            else:
                raise NotImplementedError

        elif task_name == 'traditional':
            sequential_datum = self.sequential_data[datum_idx]
            sequence = sequential_datum.split()
            user_id = sequence[0]
            user_desc = self.user_id2name.get(user_id, f'user_{user_id}')
            if self.mode == 'train':
                target_candidates = sequence[1:-2]
                target_idx = random.randint(0, len(target_candidates)-1)
                target_item = target_candidates[target_idx]
            elif self.mode == 'val':
                target_item = sequence[-2]
            elif self.mode == 'test':
                target_item = sequence[-1]
            else:
                raise NotImplementedError

            task_candidates = self.task_list[task_name]
            task_idx = random.randint(0, len(task_candidates)-1)
            task_template = self.all_tasks['traditional'][task_candidates[task_idx]]
            assert task_template['task'] == 'traditional'

            if task_template['id'] == '5-1':
                rand_prob = random.random()
                if rand_prob > 0.5:
                    source_text = task_template['source'].format(user_id, target_item)
                    target_text = task_template['target'].format('yes')
                else:
                    user_seq = self.user_items[user_id]
                    candidate_samples = []
                    candidate_num = 1
                    while len(candidate_samples) < candidate_num:
                        if self.sample_type == 'random':
                            sample_ids = np.random.choice(self.all_item, candidate_num, replace=False)
                        else:
                            sample_ids = np.random.choice(self.all_item, candidate_num, replace=False, p=self.probability)
                        sample_ids = [str(item) for item in sample_ids if item not in user_seq and item not in candidate_samples]
                        candidate_samples.extend(sample_ids)
                    candidate_samples = candidate_samples[:candidate_num]
                    source_text = task_template['source'].format(user_id, candidate_samples[0])
                    target_text = task_template['target'].format('no')
            elif task_template['id'] == '5-2':
                rand_prob = random.random()
                if rand_prob > 0.5:
                    source_text = task_template['source'].format(target_item, user_desc)
                    target_text = task_template['target'].format('yes')
                else:
                    user_seq = self.user_items[user_id]
                    candidate_samples = []
                    candidate_num = 1
                    while len(candidate_samples) < candidate_num:
                        if self.sample_type == 'random':
                            sample_ids = np.random.choice(self.all_item, candidate_num, replace=False)
                        else:
                            sample_ids = np.random.choice(self.all_item, candidate_num, replace=False, p=self.probability)
                        sample_ids = [str(item) for item in sample_ids if item not in user_seq and item not in candidate_samples]
                        candidate_samples.extend(sample_ids)
                    candidate_samples = candidate_samples[:candidate_num]
                    source_text = task_template['source'].format(candidate_samples[0], user_desc)
                    target_text = task_template['target'].format('no')
            elif task_template['id'] == '5-3':
                rand_prob = random.random()
                if rand_prob > 0.5:
                    title = self._get_title(target_item)
                    source_text = task_template['source'].format(user_desc, title)
                    target_text = task_template['target'].format('yes')
                else:
                    user_seq = self.user_items[user_id]
                    candidate_samples = []
                    candidate_num = 1
                    while len(candidate_samples) < candidate_num:
                        if self.sample_type == 'random':
                            sample_ids = np.random.choice(self.all_item, candidate_num, replace=False)
                        else:
                            sample_ids = np.random.choice(self.all_item, candidate_num, replace=False, p=self.probability)
                        sample_ids = [str(item) for item in sample_ids if item not in user_seq and item not in candidate_samples]
                        candidate_samples.extend(sample_ids)
                    candidate_samples = candidate_samples[:candidate_num]
                    title = self._get_title(candidate_samples[0])
                    source_text = task_template['source'].format(user_desc, title)
                    target_text = task_template['target'].format('no')
            elif task_template['id'] == '5-4':
                rand_prob = random.random()
                if rand_prob > 0.5:
                    title = self._get_title(target_item)
                    source_text = task_template['source'].format(user_id, title)
                    target_text = task_template['target'].format('yes')
                else:
                    user_seq = self.user_items[user_id]
                    candidate_samples = []
                    candidate_num = 1
                    while len(candidate_samples) < candidate_num:
                        if self.sample_type == 'random':
                            sample_ids = np.random.choice(self.all_item, candidate_num, replace=False)
                        else:
                            sample_ids = np.random.choice(self.all_item, candidate_num, replace=False, p=self.probability)
                        sample_ids = [str(item) for item in sample_ids if item not in user_seq and item not in candidate_samples]
                        candidate_samples.extend(sample_ids)
                    candidate_samples = candidate_samples[:candidate_num]
                    title = self._get_title(candidate_samples[0])
                    source_text = task_template['source'].format(user_id, title)
                    target_text = task_template['target'].format('no')
            elif task_template['id'] == '5-5' or task_template['id'] == '5-6':
                user_seq = self.user_items[user_id]
                candidate_samples = []
                candidate_num = 99
                while len(candidate_samples) < candidate_num:
                    if self.sample_type == 'random':
                        sample_ids = np.random.choice(self.all_item, candidate_num, replace=False)
                    else:
                        sample_ids = np.random.choice(self.all_item, candidate_num, replace=False, p=self.probability)
                    sample_ids = [str(item) for item in sample_ids if item not in user_seq and item not in candidate_samples]
                    candidate_samples.extend(sample_ids)
                candidate_samples = candidate_samples[:candidate_num]
                candidate_samples.extend([target_item])
                random.shuffle(candidate_samples)
                source_text = task_template['source'].format(user_desc, ' , '.join(candidate_samples))
                target_text = task_template['target'].format(target_item)
            elif task_template['id'] == '5-7' or task_template['id'] == '5-8':
                user_seq = self.user_items[user_id]
                candidate_samples = []
                candidate_num = 99
                while len(candidate_samples) < candidate_num:
                    if self.sample_type == 'random':
                        sample_ids = np.random.choice(self.all_item, candidate_num, replace=False)
                    else:
                        sample_ids = np.random.choice(self.all_item, candidate_num, replace=False, p=self.probability)
                    sample_ids = [str(item) for item in sample_ids if item not in user_seq and item not in candidate_samples]
                    candidate_samples.extend(sample_ids)
                candidate_samples = candidate_samples[:candidate_num]
                candidate_samples.extend([target_item])
                random.shuffle(candidate_samples)
                source_text = task_template['source'].format(user_id, ' , '.join(candidate_samples))
                target_text = task_template['target'].format(target_item)
            else:
                raise NotImplementedError

        else:
            raise NotImplementedError

        input_ids = self.tokenizer.encode(
                source_text, padding=True, truncation=True, max_length=self.args.max_text_length)
        tokenized_text = self.tokenizer.tokenize(source_text)
        whole_word_ids = self.calculate_whole_word_ids(tokenized_text, input_ids)
        assert len(whole_word_ids) == len(input_ids)

        target_ids = self.tokenizer.encode(
                target_text, padding=True, truncation=True, max_length=self.args.gen_max_length)

        out_dict['input_ids'] = torch.LongTensor(input_ids)
        out_dict['input_length'] = len(input_ids)
        out_dict['whole_word_ids'] = torch.LongTensor(whole_word_ids)
        out_dict['target_ids'] = torch.LongTensor(target_ids)
        out_dict['target_length'] = len(target_ids)

        out_dict['source_text'] = source_text
        out_dict['tokenized_text'] = tokenized_text
        out_dict['target_text'] = target_text

        out_dict['task'] = task_template['task']

        out_dict['loss_weight'] = loss_weight

        return out_dict

    def calculate_whole_word_ids(self, tokenized_text, input_ids):
        whole_word_ids = []
        curr = 0
        for i in range(len(tokenized_text)):
            if tokenized_text[i].startswith('▁'):
                curr += 1
                whole_word_ids.append(curr)
            else:
                whole_word_ids.append(curr)
        last_item = whole_word_ids[len(input_ids) - 2]
        return whole_word_ids[:len(input_ids) - 1] + [0]  # [0] for </s>

    def collate_fn(self, batch):
        batch_entry = {}

        B = len(batch)

        args = self.args

        S_W_L = max(entry['input_length'] for entry in batch)
        T_W_L = max(entry['target_length'] for entry in batch)

        input_ids = torch.ones(B, S_W_L, dtype=torch.long) * self.tokenizer.pad_token_id
        whole_word_ids = torch.ones(B, S_W_L, dtype=torch.long) * self.tokenizer.pad_token_id
        target_ids = torch.ones(B, T_W_L, dtype=torch.long) * self.tokenizer.pad_token_id

        loss_weights = torch.ones(B, dtype=torch.float)

        tasks = []
        source_text = []
        tokenized_text = []
        target_text = []

        for i, entry in enumerate(batch):
            input_ids[i, :entry['input_length']] = entry['input_ids']
            whole_word_ids[i, :entry['input_length']] = entry['whole_word_ids']
            target_ids[i, :entry['target_length']] = entry['target_ids']

            if 'task' in entry:
                tasks.append(entry['task'])

            if 'source_text' in entry:
                source_text.append(entry['source_text'])

            if 'tokenized_text' in entry:
                tokenized_text.append(entry['tokenized_text'])

            if 'target_text' in entry:
                target_text.append(entry['target_text'])

            if 'loss_weight' in entry:
                loss_weights[i] = entry['loss_weight']

        assert 't5' in args.backbone
        word_mask = target_ids != self.tokenizer.pad_token_id
        target_ids[~word_mask] = -100
        batch_entry['task'] = tasks

        batch_entry['source_text'] = source_text
        batch_entry['target_text'] = target_text

        batch_entry['input_ids'] = input_ids
        batch_entry['whole_word_ids'] = whole_word_ids
        batch_entry['target_ids'] = target_ids

        batch_entry['loss_weights'] = loss_weights

        return batch_entry


def get_loader(args, task_list, sample_numbers, split='toys', mode='train',
               batch_size=16, workers=4, distributed=False):

    if 't5' in args.backbone:
        tokenizer = P5Tokenizer.from_pretrained(
            args.backbone,
            max_length=args.max_text_length,
            do_lower_case=args.do_lower_case)

    if split == 'yelp':
        from all_yelp_templates import all_tasks as task_templates

        dataset = P5_Yelp_Dataset(
            task_templates,
            task_list,
            tokenizer,
            args,
            sample_numbers,
            mode=mode,
            split=split,
            rating_augment=False
        )
    elif split in MOVIELENS_DATASETS:
        from all_movielens_templates import all_tasks as task_templates

        dataset = P5_MovieLens_Dataset(
            task_templates,
            task_list,
            tokenizer,
            args,
            sample_numbers,
            mode=mode,
            split=split,
            rating_augment=False,
            min_rating=4.0  # Only use ratings >= 4 (positive interactions)
        )
    else:
        from all_amazon_templates import all_tasks as task_templates

        dataset = P5_Amazon_Dataset(
            task_templates,
            task_list,
            tokenizer,
            args,
            sample_numbers,
            mode=mode,
            split=split,
            rating_augment=False
        )

    if distributed:
        sampler = DistributedSampler(dataset)
    else:
        sampler = None

    if mode == 'train':
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=(sampler is None),
            num_workers=workers, pin_memory=True, sampler=sampler,
            collate_fn=dataset.collate_fn)
    else:
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=workers, pin_memory=True,
            sampler=sampler,
            shuffle=None if (sampler is not None) else False,
            collate_fn=dataset.collate_fn,
            drop_last=False)

    return loader
