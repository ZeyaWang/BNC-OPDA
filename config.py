import yaml
import easydict
from os.path import join

import data


class Dataset:
    def __init__(self, path, domains, files, prefix):
        self.path = path
        self.prefix = prefix
        self.domains = domains
        self.files = [(join(path, file)) for file in files]
        self.prefixes = [self.prefix] * len(self.domains)


import argparse
from argparse import Namespace
parser = argparse.ArgumentParser(description='Code for *Universal Domain Adaptation*',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)

## dataset and model
parser.add_argument('--dataset', type=str, default='office', help='office,officehome,visda,domainnet')
parser.add_argument('--source', type=int, default=0)
parser.add_argument('--target', type=int, default=1)
parser.add_argument('--batch_size', type=int, default=64, help='batch size')
parser.add_argument('--gpus', type=int, default=1, help='number of gpus')
parser.add_argument('--bottle_neck_dim', type=int, default=256, help='bottle_neck_dim')
parser.add_argument('--base_model', type=str, default='resnet50', help='resnet50, vgg16')

## training parameters
parser.add_argument('--lr', type=float, default=0.01) # previous training based on lr*lr_scale
parser.add_argument('--lr_scale', type=float, default=0.1) 
parser.add_argument('--weight_decay', type=float, default=0.001)
parser.add_argument('--momentum', type=float, default=0.9)
parser.add_argument('--total_epoch', type=int, default=30, help='total epochs')
parser.add_argument('--interval', type=int, default=1)

## BNC parameters
parser.add_argument('--max_k', type=int, default=100)
parser.add_argument('--covariance_prior', type=float, default=0.01)
parser.add_argument('--alpha', type=float, default=1)

## Loss parameters
parser.add_argument('--balance', type=float, default=0.01)
parser.add_argument('--lambdav', type=float, default=0.)

# Other
parser.add_argument('--KK', type=int, default=5)
parser.add_argument('--score', type=str, default='cos', help='cos or entropy')




domain_map = {'office': ['amazon', 'dslr', 'webcam'],
          'officehome': ['Art', 'Clipart', 'Product', 'Real_World'],
          'domainnet': ['Painting', 'Real', 'Sketch'],
          'visda': ['train']}


root_path = {'office': 'data/office',
            'officehome': 'data/OfficeHome',
            'visda': 'data/visda', 
            'domainnet': 'data/domainnet'}

if args.dataset == 'office':
    dataset = Dataset(
    path=root_path[args.dataset],
    domains=['amazon', 'dslr', 'webcam'],
    files=[
        'amazon_reorgnized.txt',
        'dslr_reorgnized.txt',
        'webcam_reorgnized.txt'
    ],
    prefix=root_path[args.dataset])
elif args.dataset == 'officehome':
    dataset = Dataset(
    path=root_path[args.dataset],
    domains=['Art', 'Clipart', 'Product', 'Real_World'],
    files=[
        'Art.txt',
        'Clipart.txt',
        'Product.txt',
        'Real_world.txt'
    ],
    prefix=root_path[args.dataset])
elif args.dataset == 'domainnet':
    dataset = Dataset(
    path=root_path[args.dataset],
    domains=['painting', 'real', 'sketch'],
    files=[
        'painting_updated.txt',
        'real_updated.txt',
        'sketch_updated.txt'
    ],
    prefix=root_path[args.dataset])
elif args.dataset == 'visda':
    dataset = Dataset(
    path=root_path[args.dataset],
    domains=['train', 'validation'],
    files=[
        'train/image_list.txt',
        'validation/image_list.txt',
    ],
    prefix=root_path[args.dataset])
    dataset.prefixes = [join(dataset.path, 'train'), join(dataset.path, 'validation')]
else:
    raise Exception(f'dataset {args.dataset} not supported!')


source_domain_name = dataset.domains[args.source]
target_domain_name = dataset.domains[args.target]
source_file = dataset.files[args.source]
target_file = dataset.files[args.target]