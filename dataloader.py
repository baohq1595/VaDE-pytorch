import torch
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader,TensorDataset

from metagenomics_dataset import GenomeDataset, GenomeDataset_v2

def get_mnist(data_dir='./data/mnist/',batch_size=128):
    train=MNIST(root=data_dir,train=True,download=True)
    test=MNIST(root=data_dir,train=False,download=True)

    X=torch.cat([train.data.float().view(-1,784)/255.,test.data.float().view(-1,784)/255.],0)
    Y=torch.cat([train.targets,test.targets],0)

    dataset=dict()
    dataset['X']=X
    dataset['Y']=Y

    dataloader=DataLoader(TensorDataset(X,Y),batch_size=batch_size,shuffle=True,num_workers=4)

    return dataloader,dataset

def get_genomics(data_dir='./data/gene/L1.fna', batch_size=128):
    genomics_dataset = GenomeDataset_v2(data_dir, return_raw=False)
    # X = genomics_dataset.data
    # Y = genomics_dataset.targets

    dataloader = DataLoader(genomics_dataset, batch_size=batch_size, shuffle=True)
    return dataloader, None
