import time
import utils

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import numpy as np
import ujson as json

class MySet(Dataset):
    def __init__(self, input_file):
        self.content = open('./data/' + input_file, 'r').readlines()
        self.content = list(map(lambda x: json.loads(x), self.content))  # Convert to a list of dictionaries
        self.lengths = list(map(lambda x: len(x['lngs']), self.content))  # Convert to a list of lengths

    def __getitem__(self, idx):
        return self.content[idx]

    def __len__(self):
        return len(self.content)

def collate_fn(data):
    stat_attrs = ['dist', 'time']
    info_attrs = ['driverID', 'dateID', 'weekID', 'timeID']
    traj_attrs = ['lngs', 'lats', 'states', 'time_gap', 'dist_gap']

    attr, traj = {}, {}

    # Calculate the length of the trajectory for each data point
    lens = np.asarray([len(item['lngs']) for item in data])  # Assuming lngs and other attributes have the same length

    # Process static attributes (e.g., dist, time)
    for key in stat_attrs:
        x = torch.FloatTensor([item[key] for item in data])
        attr[key] = utils.normalize(x, key)

    # Process informational attributes (e.g., driverID, dateID, weekID, timeID)
    for key in info_attrs:
        attr[key] = torch.LongTensor([item[key] for item in data])

    # Process trajectory attributes (lngs, lats, states, time_gap, dist_gap)
    for key in traj_attrs:
        # Extract sequences for each trajectory attribute
        seqs = [item[key] for item in data]
        
        # Get the maximum sequence length in this batch
        max_len = max(len(seq) for seq in seqs)
        
        # Create a padded array of zeros with shape (batch_size, max_len)
        padded = np.zeros((len(seqs), max_len), dtype=np.float32)
        
        # Fill the padded array with actual sequences
        for i, seq in enumerate(seqs):
            padded[i, :len(seq)] = seq

        # Normalize lngs, lats, time_gap, and dist_gap if necessary
        if key in ['lngs', 'lats', 'time_gap', 'dist_gap']:
            padded = utils.normalize(padded, key)

        # Convert the padded array into a PyTorch tensor
        traj[key] = torch.from_numpy(padded).float()

    # Store the sequence lengths to use for any RNNs that might process the trajectory data
    traj['lens'] = lens.tolist()

    return attr, traj


class BatchSampler:
    def __init__(self, dataset, batch_size):
        self.count = len(dataset)
        self.batch_size = batch_size
        self.lengths = dataset.lengths
        self.indices = list(range(self.count))  # Convert to a list for Python 3

    def __iter__(self):
        '''
        Divide the data into chunks with size = batch_size * 100
        Sort by the length in one chunk
        '''
        np.random.shuffle(self.indices)

        chunk_size = self.batch_size * 100
        chunks = (self.count + chunk_size - 1) // chunk_size

        # Re-arrange indices to minimize the padding
        for i in range(chunks):
            partial_indices = self.indices[i * chunk_size: (i + 1) * chunk_size]
            partial_indices.sort(key=lambda x: self.lengths[x], reverse=True)
            self.indices[i * chunk_size: (i + 1) * chunk_size] = partial_indices

        # Yield batch
        batches = (self.count - 1 + self.batch_size) // self.batch_size

        for i in range(batches):
            yield self.indices[i * self.batch_size: (i + 1) * self.batch_size]

    def __len__(self):
        return (self.count + self.batch_size - 1) // self.batch_size

def get_loader(input_file, batch_size):
    dataset = MySet(input_file=input_file)

    batch_sampler = BatchSampler(dataset, batch_size)

    data_loader = DataLoader(
        dataset=dataset,
        batch_size=1,
        collate_fn=lambda x: collate_fn(x),
        num_workers=4,
        batch_sampler=batch_sampler,
        pin_memory=True
    )

    return data_loader
