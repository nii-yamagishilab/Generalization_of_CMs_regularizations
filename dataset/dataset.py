'''
djlfsjlfkdj
'''
import os
import torch
import numpy as np
import random
import soundfile as sf
from torch.utils.data import Dataset, DataLoader

class CNSpoofDataset(Dataset):
    def __init__(self, task, data = 'train'):
        self.utt2wav = {}
        self.utt2label = {}
        self.utt2genre = {}
        self.utt2idx = {}
        self.idx2utt = {}
        self.dur = []
        self.utt2wav = self._read_kaldi_file(os.path.join('data', task, data, 'wav.scp'))
        self.utt2label = self._read_kaldi_file(os.path.join('data', task, data, 'utt2spk'))
        self.utt2genre = self._read_kaldi_file(os.path.join('data', task, data, 'utt2genre'))
        self.utt2idx = dict([(utt, idx) for idx, utt in enumerate(list(self.utt2label.keys()))])
        self.idx2utt = dict([(idx, utt) for idx, utt in enumerate(list(self.utt2label.keys()))])
        utt2dur = self._read_kaldi_file(os.path.join('data', task, data, 'utt2dur'))
        self.dur = list(utt2dur.values())
        self.label2idx = {'real': 1, 'spoof': 0}
        self.genre2idx = {
                'drama':0,
                'entertainment':1,
                'interview':2,
                'live_broadcast':3,
                'movie':4,
                'play':5,
                'recitation':6,
                'singing':7,
                'speech':8,
				'vlog':9
                }

    def _read_kaldi_file(self, path):
        utt2content = {}
        with open(path, 'r') as f:
            for line in f:
                line = line.rstrip()
                uttid, content = line.split()
                utt2content[uttid] = content
        return utt2content

    def __len__(self):
        return len(self.utt2label)

    def __getitem__(self, idx):
        uttid = self.idx2utt[idx]
        data, _ = sf.read(self.utt2wav[uttid])
        audio = torch.from_numpy(data.astype(np.float32)).unsqueeze(1)
        label = self.label2idx[self.utt2label[uttid]]
        genre = self.utt2genre[uttid]
        genreid = self.genre2idx[genre]
        return audio, label, genre, genreid, len(data)

class MetaDataset(Dataset):
    '''
    ssjdkfjlsf
    '''
    def __init__(self, task, data = 'train'):
        self.utt2wav = {}
        self.utt2label = {}
        self.utt2genre = {}
        self.genre2uttlist = {}
        self.utt2idx = {}
        self.idx2utt = {}
        self.dur = []
        self.utt2wav = self._read_kaldi_file(os.path.join('data', task, data, 'wav.scp'))
        self.utt2label = self._read_kaldi_file(os.path.join('data', task, data, 'utt2spk'))
        self.utt2genre = self._read_kaldi_file(os.path.join('data', task, data, 'utt2genre'))
        self.utt2idx = dict([(utt, idx) for idx, utt in enumerate(list(self.utt2label.keys()))])
        self.idx2utt = dict([(idx, utt) for idx, utt in enumerate(list(self.utt2label.keys()))])
        utt2dur = self._read_kaldi_file(os.path.join('data', task, data, 'utt2dur'))
        self.dur = list(utt2dur.values())
        self.label2idx = {'real': 1, 'spoof': 0}
        self.genres = list(set(self.utt2genre.values()))
        for uttid, genre in self.utt2genre.items():
            if not genre in self.genre2uttlist:
                self.genre2uttlist[genre] = []
            self.genre2uttlist[genre].append(uttid)
        self.collate = Collate()
        genres = list(set(self.utt2genre.values()))
        self.genre2idx = {}
        for idx, genre in enumerate(genres):
            self.genre2idx[genre] = idx

    def _read_kaldi_file(self, path):
        utt2content = {}
        with open(path, 'r') as f:
            for line in f:
                line = line.rstrip()
                uttid, content = line.split()
                utt2content[uttid] = content
        return utt2content

    def __len__(self):
        return len(self.utt2label)

    def __getitem__(self, idx):
        return idx

    def get_domain_num(self):
        return len(set(self.utt2genre.values()))

    def random_collate_fn(self, batch):
        batch = [self.idx2utt[idx] for idx in batch]
        num_meta_test_samples = 0
        while num_meta_test_samples < 2:
            selected_genres_list = [self.utt2genre[uttid] for uttid in batch]
            selected_genres = list(set(selected_genres_list))
            assert len(selected_genres) >= 2, "In this batch there is only one genre!"
            meta_train_genres = random.sample(selected_genres, k = len(selected_genres) - 1)
            meta_test_genres = []
            for genre in selected_genres:
                if not genre in meta_train_genres:
                    meta_test_genres.append(genre)
            meta_train_uttid = []
            meta_test_uttid = []
            for uttid in batch:
                if self.utt2genre[uttid] in meta_train_genres:
                    meta_train_uttid.append(uttid)
                else:
                    meta_test_uttid.append(uttid)
            num_meta_test_samples = len(meta_test_uttid)
        meta_train_audio = [torch.from_numpy(sf.read(self.utt2wav[uttid])[0].astype(np.float32)).unsqueeze(1) for uttid in meta_train_uttid]
        meta_test_audio = [torch.from_numpy(sf.read(self.utt2wav[uttid])[0].astype(np.float32)).unsqueeze(1) for uttid in meta_test_uttid]
        meta_train_label = torch.tensor([self.label2idx[self.utt2label[uttid]] for uttid in meta_train_uttid], dtype = torch.float32)
        meta_test_label = torch.tensor([self.label2idx[self.utt2label[uttid]] for uttid in meta_test_uttid], dtype = torch.float32)
        meta_train_length = [len(audio) for audio in meta_train_audio]
        meta_test_length = [len(audio) for audio in meta_test_audio]
        meta_train_datalengths = [min(train_length, self.collate.sampling_rate * self.collate.max_len) for train_length in meta_train_length]
        meta_test_datalengths = [min(test_length, self.collate.sampling_rate * self.collate.max_len) for test_length in meta_test_length]
        meta_train_genresid = torch.tensor([self.genre2idx[self.utt2genre[uttid]] for uttid in meta_train_uttid], dtype = torch.int64)
        meta_test_genresid = torch.tensor([self.genre2idx[self.utt2genre[uttid]] for uttid in meta_test_uttid], dtype = torch.int64)
        if self.collate.mode == 'trunc':
            meta_train_audio = self.collate._trunc(meta_train_audio)
            meta_test_audio = self.collate._trunc(meta_test_audio)
        else:
            meta_train_audio = self.collate._padding(meta_train_audio)
            meta_test_audio = self.collate._padding(meta_test_audio)
        meta_train_audio = torch.concat(meta_train_audio, dim = 1).permute(1, 0).view(len(meta_train_audio), -1, 1)
        meta_test_audio = torch.concat(meta_test_audio, dim = 1).permute(1, 0).view(len(meta_test_audio), -1, 1)
        return meta_train_audio, meta_train_label, meta_train_datalengths, meta_train_genresid, meta_test_audio, meta_test_label, meta_test_datalengths, meta_test_genresid #genres, genreids, datalengths   

    def balance_collate_fn(self, batch):
        batch_size = len(batch)
        num_per_genre = batch_size // len(self.genres)
        meta_train_genres = random.sample(self.genres, k = len(self.genres) - 1)
        meta_test_genres = []
        for genre in self.genres:
            if not genre in meta_train_genres:
                meta_test_genres.append(genre)
        meta_train_uttid_genre = [random.sample(self.genre2uttlist[genre], k = num_per_genre) for genre in meta_train_genres]
        meta_test_uttid_genre = [random.sample(self.genre2uttlist[genre], k = num_per_genre) for genre in meta_test_genres]
        meta_train_uttid, meta_test_uttid = [], []
        for uttidlist in meta_train_uttid_genre:
            meta_train_uttid += uttidlist
        for uttidlist in meta_test_uttid_genre:
            meta_test_uttid += uttidlist
        meta_train_audio = [torch.from_numpy(sf.read(self.utt2wav[uttid])[0].astype(np.float32)).unsqueeze(1) for uttid in meta_train_uttid]
        meta_test_audio = [torch.from_numpy(sf.read(self.utt2wav[uttid])[0].astype(np.float32)).unsqueeze(1) for uttid in meta_test_uttid]
        meta_train_label = torch.tensor([self.label2idx[self.utt2label[uttid]] for uttid in meta_train_uttid], dtype = torch.float32)
        meta_test_label = torch.tensor([self.label2idx[self.utt2label[uttid]] for uttid in meta_test_uttid], dtype = torch.float32)
        meta_train_length = [len(audio) for audio in meta_train_audio]
        meta_test_length = [len(audio) for audio in meta_test_audio]
        meta_train_datalengths = [min(train_length, self.collate.sampling_rate * self.collate.max_len) for train_length in meta_train_length]
        meta_test_datalengths = [min(test_length, self.collate.sampling_rate * self.collate.max_len) for test_length in meta_test_length]
        if self.collate.mode == 'trunc':
            meta_train_audio = self.collate._trunc(meta_train_audio)
            meta_test_audio = self.collate._trunc(meta_test_audio)
        else:
            meta_train_audio = self.collate._padding(meta_train_audio)
            meta_test_audio = self.collate._padding(meta_test_audio)
        meta_train_audio = torch.concat(meta_train_audio, dim = 1).permute(1, 0).view(len(meta_train_audio), -1, 1)
        meta_test_audio = torch.concat(meta_test_audio, dim = 1).permute(1, 0).view(len(meta_test_audio), -1, 1)
        return meta_train_audio, meta_train_label, meta_train_datalengths, meta_test_audio, meta_test_label, meta_test_datalengths #genres, genreids, datalengths

class Collate(object):
    def __init__(
            self,
            mode = 'padding',
            sampling_rate = 16000,
            max_len = 20,
            padding_value = 0.0
            ):
        self.mode = mode
        self.max_len = max_len
        self.padding_value = padding_value
        self.sampling_rate = sampling_rate

    def _trunc(self, batch):
        dim_size = batch[0].size()
        trailing_dims = dim_size[1:]

        # get the maximum length
        max_len = max([s.size(0) for s in batch])
        
        if all(x.shape[0] == max_len for x in batch):
            # if all data sequences in batch have the same length, no need to pad
            return batch
        else:
            # else, we need to pad 
            out_dims = (max_len, ) + trailing_dims
            
            output_batch = []
            for i, tensor in enumerate(batch):
                # check the rest of dimensions
                if tensor.size()[1:] != trailing_dims:
                    print("Data in batch has different dimensions:")
                    for data in batch:
                        print(str(data.size()))
                    raise RuntimeError('Fail to create batch data')
                # save padded results
                out_tensor = tensor.new_full(out_dims, self.padding_value)
                out_tensor[:tensor.size(0), ...] = tensor
                output_batch.append(out_tensor)
            return output_batch

    def _padding(self, batch):
        dim_size = batch[0].size()
        trailing_dims = dim_size[1:]

        # get the maximum length, if length > 20s, truncate the audio
        max_len = min(max([s.size(0) for s in batch]), self.max_len * self.sampling_rate)
        #  max_len = max([s.size(0) for s in batch])
        
        if all(x.shape[0] == max_len for x in batch):
            # if all data sequences in batch have the same length, no need to pad
            return batch
        else:
            # else, we need to pad 
            out_dims = (max_len, ) + trailing_dims
            
            output_batch = []
            for _, tensor in enumerate(batch):
                # check the rest of dimensions
                if tensor.size()[1:] != trailing_dims:
                    print("Data in batch has different dimensions:")
                    for data in batch:
                        print(str(data.size()))
                    raise RuntimeError('Fail to create batch data')
                # save padded results
                out_tensor = tensor.new_full(out_dims, self.padding_value)
                if tensor.size(0) <= max_len:
                    out_tensor[:tensor.size(0), ...] = tensor
                else:
                    start = torch.randint(0, tensor.size(0) - max_len, (1,))
                    out_tensor[:, ...] = tensor[start:start+max_len, ...]
                output_batch.append(out_tensor)
            return output_batch

if __name__ == "__main__":
    import sys
    from tqdm import tqdm
    meta_dataset = MetaDataset('ordinary', 'train')
    trainloader = DataLoader(meta_dataset,
                             batch_size = 300,
                             shuffle = True,
                             collate_fn = meta_dataset.random_collate_fn,
                             num_workers = 8,
                             pin_memory = True)
    for mtr_data, mtr_label, mtr_datalength, mtr_genre, mte_data, mte_label, mte_datalength, mte_genre in tqdm(trainloader):
        print(mtr_data.shape)
        print(mtr_label)
        print(mtr_datalength)
        print(type(mtr_genre))
        print(mte_data.shape)
        print(mte_label)
        print(mte_datalength)
        print(mte_genre)
