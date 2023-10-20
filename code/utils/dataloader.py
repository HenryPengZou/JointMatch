import os
import sys
import torch
import pandas as pd
import numpy as np
import nlpaug.augmenter.word as naw
from transformers import BertTokenizer
from torch.utils.data import Dataset, DataLoader

class SEMIDataset(Dataset):
    def __init__(self, sents, sents_aug1, sents_aug2, labels=None):
        self.sents = sents
        self.sents_aug1 = sents_aug1
        self.sents_aug2 = sents_aug2
        self.labels = labels

    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return self.sents[idx], self.sents_aug1[idx], self.sents_aug2[idx], self.labels[idx]

class SEMINoAugDataset(Dataset):
    def __init__(self, sents, labels=None):
        self.sents = sents
        self.labels = labels

    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return self.sents[idx], self.labels[idx]

class MyCollator(object):
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, batch):
        sents, sents_aug1, sents_aug2 = [], [], []
        labels = []
        for sample in batch:
            if len(sample) == 2:
                sents.append(sample[0])
                labels.append(sample[1])
                sents_aug1 = None
                sents_aug2 = None
            elif len(sample) == 4:
                sents.append(sample[0])
                sents_aug1.append(sample[1])
                sents_aug2.append(sample[2])
                labels.append(sample[3])
    
        tokenized = self.tokenizer(sents, padding=True, truncation='longest_first', max_length=255, return_tensors='pt')
        labels = torch.LongTensor(labels) - 1
        if sents_aug1 is not None:
            # further add stochastic synoym replacement augmentation
            sents_aug1 = [naw.SynonymAug(aug_src='wordnet', aug_p=0.05).augment(sent)[0] for sent in sents_aug1]
            tokenized_aug1 = self.tokenizer(sents_aug1, padding=True, truncation='longest_first', max_length=255, return_tensors='pt')
        else:
            # add stochastic synoym replacement augmentation
            sents_aug1 = [naw.SynonymAug(aug_src='wordnet', aug_p=0.05).augment(sent)[0] for sent in sents]
            tokenized_aug1 = self.tokenizer(sents_aug1, padding=True, truncation='longest_first', max_length=255, return_tensors='pt')            
        if sents_aug2 is not None: 
            # further add stochastic synoym replacement augmentation
            sents_aug2 = [naw.SynonymAug(aug_src='wordnet', aug_p=0.05).augment(sent)[0] for sent in sents_aug2]
            tokenized_aug2 = self.tokenizer(sents_aug2, padding=True, truncation='longest_first', max_length=255, return_tensors='pt')
        else:
            tokenized_aug2 = None
        # return tokenized, tokenized_aug1, tokenized_aug2, labels
        return {'x': tokenized, 'x_w': tokenized_aug1,'x_s': tokenized_aug2, 'label': labels}

def train_split(labels, n_labeled_per_class, unlabeled_per_class=None):
    """Split the dataset into labeled and unlabeled subsets.
    Args:
        labels: labels of the training data
        n_labeled_per_class: number of labeled examples per class
        unlabeled_per_class: number of unlabeled examples per class
        Returns:
            train_labeled_idxs: list of labeled example indices
            train_unlabeled_idxs: list of unlabeled example indices
    """
    labels = np.array(labels)
    all_classes = set(labels)

    train_labeled_idxs = []
    train_unlabeled_idxs = []

    for i in all_classes:
        idxs = np.where(labels == i)[0]
        np.random.shuffle(idxs)
        train_labeled_idxs.extend(idxs[:n_labeled_per_class])
        if unlabeled_per_class:
            train_unlabeled_idxs.extend(idxs[n_labeled_per_class:n_labeled_per_class+unlabeled_per_class])
        else: 
            train_unlabeled_idxs.extend(idxs[n_labeled_per_class:])

    np.random.shuffle(train_labeled_idxs)
    np.random.shuffle(train_unlabeled_idxs)

    return train_labeled_idxs, train_unlabeled_idxs

def get_dataloader(data_path, n_labeled_per_class, bs, load_mode='semi'):
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    
    train_df = pd.read_csv(os.path.join(data_path,'train.csv'))
    dev_df = pd.read_csv(os.path.join(data_path,'dev.csv'))
    test_df = pd.read_csv(os.path.join(data_path,'test.csv'))
    
    labels = list(train_df["label"])
    num_class = len(set(labels))
    train_labeled_idxs, train_unlabeled_idxs = train_split(labels, n_labeled_per_class)
    train_l_df, train_u_df = train_df.iloc[train_labeled_idxs].reset_index(drop=True), train_df.iloc[train_unlabeled_idxs].reset_index(drop=True)
    if 'imdb' in data_path:
        train_l_df = pd.read_csv(os.path.join(data_path,'train_{}.csv'.format(n_labeled_per_class)))
        print('In this settting, we directly load the same labeled data split as in the SAT paper for fair comparison.')
        
    # check statistics info
    print('n_labeled_per_class: ', n_labeled_per_class)
    print('train_df samples: %d' % (train_df.shape[0]))
    print('train_labeled_df samples: %d' % (train_l_df.shape[0]))
    print('train_unlabeled_df samples: %d' % (train_u_df.shape[0]))

    print('Check n_smaples_per_class in the original training set: ', train_df['label'].value_counts().to_dict())
    print('Check n_smaples_per_class in the labeled training set: ', train_l_df['label'].value_counts().to_dict())
    print('Check n_smaples_per_class in the unlabeled training set: ', train_u_df['label'].value_counts().to_dict())

    if load_mode == 'semi':
        if 'yahoo' in data_path:
            bt_df = pd.read_csv(os.path.join(data_path, 'bt_train.csv'))
            bt_l_df, bt_u_df = bt_df.iloc[train_labeled_idxs].reset_index(drop=True), bt_df.iloc[train_unlabeled_idxs].reset_index(drop=True)
            train_dataset_l = SEMIDataset(train_l_df['content'].to_list(), train_l_df['synonym_aug'].to_list(), bt_l_df['back_translation'], labels=train_l_df['label'].to_list())
            train_dataset_u = SEMIDataset(train_u_df['content'].to_list(), train_u_df['synonym_aug'].to_list(), bt_u_df['back_translation'], labels=train_u_df['label'].to_list())
        else:
            train_dataset_l = SEMIDataset(train_l_df['content'].to_list(), train_l_df['synonym_aug'].to_list(), train_l_df['back_translation'], labels=train_l_df['label'].to_list())
            train_dataset_u = SEMIDataset(train_u_df['content'].to_list(), train_u_df['synonym_aug'].to_list(), train_u_df['back_translation'], labels=train_u_df['label'].to_list())
        train_loader_u = DataLoader(dataset=train_dataset_u, batch_size=bs, shuffle=True, collate_fn=MyCollator(tokenizer))
    
    elif load_mode == 'sup_baseline':
        train_dataset_l = SEMINoAugDataset(train_l_df['content'].to_list(), train_l_df['label'].to_list())
        train_loader_u = None
        
    train_loader_l = DataLoader(dataset=train_dataset_l, batch_size=bs, shuffle=True, collate_fn=MyCollator(tokenizer))

    dev_dataset = SEMINoAugDataset(dev_df['content'].to_list(), labels=dev_df['label'].to_list())
    test_dataset = SEMINoAugDataset(test_df['content'].to_list(), labels=test_df['label'].to_list())
    dev_loader = DataLoader(dataset=dev_dataset, batch_size=2*bs, shuffle=False, collate_fn=MyCollator(tokenizer))
    test_loader = DataLoader(dataset=test_dataset, batch_size=2*bs, shuffle=False, collate_fn=MyCollator(tokenizer))

    return train_loader_l, train_loader_u, dev_loader, test_loader, num_class



# Unit Test
if __name__ == '__main__':
    # go to the directory of data
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    os.chdir('../../data')
    print('current work directory: ', os.getcwd())

    n_labeled_per_class = 10
    bs = 32
    data_path_list = ['ag_news', 'yahoo', 'imdb']
    load_mode_list = ['semi'] # ['semi', 'baseline']

    for data_path in data_path_list:
        for load_mode in load_mode_list:
            print('\ndata_path: ', data_path)
            print('load_mode: ', load_mode)
            train_loader_l, train_loader_u, dev_loader, test_loader, num_class = get_dataloader(data_path, n_labeled_per_class, bs, load_mode)

            # check if the dataloader can work
            train_loader_l = iter(train_loader_l)
            batch = next(train_loader_l)
            print('batch: ', batch)
