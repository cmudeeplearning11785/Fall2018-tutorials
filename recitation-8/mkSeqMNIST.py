import torch
from torchvision import datasets, transforms
import random
import numpy as np
from random import randint as rint
from scipy.misc import imsave
import os

def make():
    N = 10
    M = 20000
    space = 200
    overlap = 15

    random.seed(123456789)

    data = datasets.MNIST('/tmp/data', train=True, download=True,
            transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
                ]))

    ndata = torch.ByteTensor(np.lib.pad(data.train_data.numpy(), ((0,0),(4,4),(0,0)), 'constant'))

    dataset_data = np.zeros((M, 36, 0))
    dataset_labels = np.zeros((M, 36, 0))
    s = np.append(data.train_labels.view(-1,1,1).repeat(1,36,1).numpy()[:M], ndata.numpy()[:M], axis=2)
    for i in range(N):
        p = np.random.permutation(s)
        d = np.roll(p[:,:,1:], (0, rint(-4,4), 0), (0,1,2))
        if i == 0:
            dataset_data = d
        else:
            oq = rint(0, overlap-9) + 9
            dd = np.append(np.zeros((M, 36, dataset_data.shape[2]-oq)), d, axis=2)
            dataset_data =   np.append(dataset_data, np.zeros((M,36,28-oq)), axis=2)
            dataset_data += dd
        dataset_labels = np.append(dataset_labels, p[:,:,0:1], axis=2)

    dataset_labels = dataset_labels[:,0,:]
    # Creates a dataset of 60000 (28*N + (N-1)*overlap) * 36 images
    # containing N numbers in sequence and their labels
    images = []
    if not os.path.exists('./images'): os.makedirs('./images')
    for i in range(M):
        '''
        Randomly adding spacing bettween the numbers and then saving the images.
        '''
        img = np.zeros((36, 0))
        dist = torch.multinomial(torch.ones(N+1), space, replacement=True)
        for j in range(N+1):
            img = np.append(img, np.zeros((36, (dist==j).sum())), axis=1)
            img = np.append(img, dataset_data[i,:,28*j:28*(j+1)], axis=1)
        img = dataset_data[i,:,:]
        images.append(img)
        name = './images/img_' + ''.join(map(lambda x: str(int(x)), dataset_labels[i])) + '.png'
        imsave(name, img.clip(0, 255))
    dataset_data = np.array(images)

    if not os.path.exists('./dataset'): os.makedirs('./dataset')
    np.save("./dataset/data", dataset_data)
    np.save("./dataset/labels", dataset_labels)

if __name__ == "__main__":
    make()