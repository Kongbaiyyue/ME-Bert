import torch
import numpy as np


class Graph_Bert_Dataset(object):
    def __init__(self, path, opt, smiles_field='Smiles', addH=True):
        if path.endswith('.txt') or path.endswith('.tsv'):
            self.df = pd.read_csv(path,sep='\t')
        else:
            self.df = pd.read_csv(path)
        self.smiles_field = smiles_field
        self.vocab = str2num
        self.devocab = num2str
        self.addH = addH
        self.opt = opt
        self.data_path = opt.data

    def get_data(self):

        data = self.df
        train_idx = []
        idx = data.sample(frac=0.9).index
        train_idx.extend(idx)

        data1 = data[data.index.isin(train_idx)]
        data2 = data[~data.index.isin(train_idx)]

        self.train_dataset = tf.data.Dataset.from_tensor_slices(data1[self.smiles_field].tolist())
        self.train_dataset = self.train_dataset.map(self.tf_numerical_smiles).padded_batch(256, padded_shapes=(
            tf.TensorShape([None]),tf.TensorShape([None,None]), tf.TensorShape([None]) ,tf.TensorShape([None]))).prefetch(50)

        self.test_dataset = tf.data.Dataset.from_tensor_slices(data2[self.smiles_field].tolist())
        self.test_dataset = self.test_dataset.map(self.tf_numerical_smiles).padded_batch(512, padded_shapes=(
            tf.TensorShape([None]), tf.TensorShape([None, None]), tf.TensorShape([None]),
            tf.TensorShape([None]))).prefetch(50)
        return self.train_dataset, self.test_dataset

    def load(self, path):
        vocab_dict = {}
        with open(path, 'r') as f:
            for line in f.readlines():
                vocab = line.strip().split('\t')
                vocab_dict[vocab[0]] = int(vocab[1])
            return vocab_dict

    def numerical_smiles(self, smiles):
        # smiles = smiles.numpy().decode()
        # atoms_list, adjoin_matrix = smiles2adjoin(smiles,explicit_hydrogens=self.addH)
        # atoms_list = ['<global>'] + atoms_list
        # vocab = self.load(opt.data + '.vocab.pt')
        tokens = smiles
        vocab = self.load('data/USPTO-50k_no_rxn/USPTO-50k_no_rxn.vocab.txt')
        vocab['<mask>'] = len(vocab)
        nums_list = [vocab.get(i, vocab['<unk>']) for i in tokens]

        # temp = torch.ones((len(nums_list), len(nums_list)))
        # temp[1:, 1:] = adjoin_matrix
        # adjoin_matrix = (1 - temp) * (-1e9)

        choices = np.random.permutation(len(nums_list)-1)[:max(int(len(nums_list)*0.15), 1)] + 1
        y = torch.tensor(nums_list, dtype=torch.int64)
        weight = torch.zeros(len(nums_list))
        for i in choices:
            rand = np.random.rand()
            weight[i] = 1
            if rand < 0.8:
                nums_list[i] = vocab['<mask>']
            elif rand < 0.9:
                nums_list[i] = int(np.random.rand() * 14 + 1)

        x = torch.tensor(nums_list, dtype=torch.int64)
        weight = weight.to(dtype=torch.float32)
        # return x, adjoin_matrix, y, weight
        return x, y, weight

    def tf_numerical_smiles(self, smiles):
        # x,adjoin_matrix,y,weight = tf.py_function(self.balanced_numerical_smiles,
        #                                           [data], [tf.int64, tf.float32 ,tf.int64,tf.float32])
        # x, adjoin_matrix, y, weight = self.numerical_smiles(opt, smiles)
        x, y, weight = self.numerical_smiles(smiles)

        # x.set_shape([None])
        # adjoin_matrix.set_shape([None,None])
        # y.set_shape([None])
        # weight.set_shape([None])
        # return x, adjoin_matrix, y, weight
        return x, y, weight