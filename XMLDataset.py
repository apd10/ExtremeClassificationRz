import torch
import numpy as np
import torch
from torch.utils import data
import pdb
import sklearn.datasets as ds

class GenSVMFormatParser(data.Dataset):
    def __init__(self, X_file, params, sparse=False):
        super(GenSVMFormatParser, self).__init__()
        #with open(X_file, 'r+') as xfile:
        #    self.X = xfile.readlines()
        self.data, self.labels = ds.load_svmlight_file(X_file, multilabel = True)

        self.length = self.data.shape[0]
        self.dim = params["dimension"]
        self.label_dim = params["label_dimension"]
        self.labels = [ np.array(i).astype(int) for i in self.labels]
        self.sparse = sparse

    def __len__(self):
        return self.length

    def __getitem__(self, index):

        label = np.zeros(self.label_dim)
        
        if (not self.sparse):
          data_point = np.zeros(self.dim)
          x = self.data[index].todense()
          data_point[:x.shape[1]] = np.array(x).reshape(-1)
        else:
          data_point = (self.data[index].indices, self.data[index].data)

        label[self.labels[index]] = 1

        return data_point, label

   
if __name__ == '__main__':     
    xfile = '/home/apd10/ExtremeClassificationRz/data/LF-AmazonTitles-131K/train.txt'
    params = {'dimension' : 40000, 'label_dimension' : 131073}
    parser = GenSVMFormatParser(xfile, params, sparse=True)
    for i in range(10):
        x = parser.__getitem__(i)
        pdb.set_trace()
