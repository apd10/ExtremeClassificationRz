import torch
import numpy as np
import torch
from torch.utils import data
import pdb
import sklearn.datasets as ds

class GenSVMFormatParser(data.Dataset):
    def __init__(self, X_file, params):
        super(GenSVMFormatParser, self).__init__()
        #with open(X_file, 'r+') as xfile:
        #    self.X = xfile.readlines()
        self.data, self.labels = ds.load_svmlight_file(X_file, multilabel = True)

        self.length = self.data.shape[0]
        self.dim = params["dimension"]
        self.label_dim = params["label_dimension"]
        self.labels = [ np.array(i).astype(int) for i in self.labels]

    def __len__(self):
        return self.length

    def __getitem__(self, index):

        data_point = np.zeros(self.dim)
        label = np.zeros(self.label_dim)
        
        x = self.data[index].todense()
        data_point[:x.shape[1]] = np.array(x).reshape(-1)

        label[self.labels[index]] = 1

#        tokens =  self.X[index].strip().split(" ")
#        for i in tokens[0].split(','):
#            label[int(i)] = 1
#
#        for t in tokens[1:]:
#            temp = t.split(':')
#            data_point[int(temp[0])] = float(temp[1])

        return data_point, label

   
if __name__ == '__main__':     
    xfile = '/home/apd10/XMLTrain/sample/x.txt'
    params = {'dimension' : 512, 'label_dimension' : 670091}
    parser = GenSVMFormatParser(xfile, params)
    for i in range(10):
        x = parser.__getitem__(i)
        pdb.set_trace()
