import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from stl10_input import *
import os
import sys
if sys.version_info[0] == 2:
    import cPickle as pickle
else:
    import pickle


def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()



def load_pickle(f):
    if sys.version_info[0] == 2:
        return pickle.load(f)
    elif sys.version_info[0] == 3:
        return pickle.load(f, encoding='latin1')
    raise ValueError("invalid python version: {}".format(sys.version))


def load_CIFAR_batch(filename):
    """ load single batch of cifar """
    with open(filename, "rb") as f:
        datadict = load_pickle(f)
        X = datadict["data"]
        Y = datadict["labels"]
        X = X.reshape(10000, 3, 32, 32).transpose(0, 2, 3, 1)
        Y = np.array(Y)
        return X, Y



###  suggested reference: https://pytorch.org/tutorials/
# recipes/recipes/custom_dataset_transforms_loader.html?highlight=dataloader
# functions to show an image

class CIFAR10_loader(Dataset):
    def __init__(self,root,train=True,transform = None):
        pass
        self.transform = transform
        self.data = []
        self.label = []
        if train:
            for b in range(1, 6):
                f = os.path.join(root, "cifar-10-batches-py/data_batch_%d" % (b,))
                X, Y = load_CIFAR_batch(f)
                self.data.append(X)
                self.label.append(Y)
            self.data = np.vstack(self.data)
            self.label = np.hstack(self.label)
        else:
            f = os.path.join(root, "cifar-10-batches-py/test_batch")
            self.data, self.label = load_CIFAR_batch(f)


    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        img, target = self.data[item], self.label[item]

        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        return img, target



class STL10_Dataset(Dataset):
    def __init__(self,root,train=True,transform = None):
        pass
        self.transform = transform
        self.data = []
        self.label = []
        if train:
            datapth = './data/stl10_binary/train_X.bin'
            lbpth   = './data/stl10_binary/train_y.bin'
            self.data = read_all_images(datapth)
            self.label= read_labels(lbpth)
        else:
            datapth = './data/stl10_binary/test_X.bin'
            lbpth = './data/stl10_binary/test_y.bin'
            self.data = read_all_images(datapth)
            self.label = read_labels(lbpth)

        self.data,self.label = self.data_update(self.data,self.label)

    def data_update(self,data,label):
        lb_dict = {1:0,2:1,9:2,4:3,6:4}
        new_data = []
        new_label = []
        assert len(data)==len(label)
        for item in range(len(data)):
            if label[item] in lb_dict:
                new_data.append(data[item])
                new_label.append(lb_dict[label[item]])

        return new_data,new_label


    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        img, target = self.data[item], self.label[item]

        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        return img, target






if __name__ =='__main__':
    import torchvision.transforms as transforms

    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    trainset = CIFAR10_loader('./data',True,transform)
    trainloader = DataLoader(trainset, batch_size=4,
                                              shuffle=True, num_workers=2)
    dataiter = iter(trainloader)
    images, labels = dataiter.next()

