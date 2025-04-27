from config import *
from easydl import *
from collections import Counter
from torchvision.transforms.transforms import *
from torch.utils.data import DataLoader, WeightedRandomSampler, Subset
from torchvision import datasets, transforms
import torch


'''
assume classes across domains are the same.
[0 1 ..................................................................... N - 1]
|----common classes --||----source private classes --||----target private classes --|
'''

if args.target_type == 'OPDA':
    print('OPDA')
    n_share = {'office': 10,
                'officehome': 10,
                'visda': 6,
                'domainnet': 150}

    n_source_private = {'office': 10,
                'officehome': 5,
                'visda': 3,
                'domainnet': 50}

    n_total = {'office': 31,
                'officehome': 65,
                'visda': 12,
                'domainnet': 345}

elif args.target_type == 'OSDA':
    print('OSDA')
    n_share = {'office': 10,
                'officehome': 25,
                'visda': 6}

    n_source_private = {'office': 0,
                'officehome': 0,
                'visda': 0}

    n_total = {'office': 21,
                'officehome': 65,
                'visda': 12}

elif args.target_type == 'PDA':
    print('PDA')
    n_share = {'office': 10,
                'officehome': 25,
                'visda': 6}

    n_source_private = {'office': 21,
                'officehome': 40,
                'visda': 6}

    n_total = {'office': 31,
                'officehome': 65,
                'visda': 12}


data_workers = 3

a, b, c = n_share[args.dataset], n_source_private[args.dataset], n_total[args.dataset]
c = c - a - b # common, source-private, target-private
common_classes = [i for i in range(a)]
source_private_classes = [i + a for i in range(b)]

if args.dataset == "office" and args.target_type == "OSDA":
    target_private_classes = [i + a + b + 10 for i in range(c)]
else:
    target_private_classes = [i + a + b for i in range(c)]


source_classes = common_classes + source_private_classes
target_classes = common_classes + target_private_classes
num_src_cls = len(source_classes)

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

train_transform = Compose([
    Resize(256),
    RandomCrop(224),
    RandomHorizontalFlip(),
    ToTensor(),
    normalize # just added to accomodate glc
])

test_transform = Compose([
    Resize(256),
    CenterCrop(224),
    ToTensor(),
    normalize # just added to accomodate glc
])


target_test_ds = FileListDataset(list_path=target_file, path_prefix=dataset.prefixes[args.target],
                            transform=test_transform, filter=(lambda x: x in target_classes))

target_test_dl = DataLoader(dataset=target_test_ds, batch_size=args.batch_size, shuffle=False,
                             num_workers=1, drop_last=False,sampler=None)

# target_train_ds = FileListDataset(list_path=target_file, path_prefix=dataset.prefixes[args.target],
#                             transform=train_transform, filter=(lambda x: x in target_classes))
target_train_ds = FileListDataset(list_path=target_file, path_prefix=dataset.prefixes[args.target],
                            transform=test_transform, filter=(lambda x: x in target_classes))

target_train_ds.labels = [i for i in range(len(target_train_ds.datas))]
target_train_dl = DataLoader(dataset=target_train_ds, batch_size=args.batch_size,shuffle=True,
                             num_workers=data_workers, drop_last=True)

true_labels = np.array(target_test_ds.labels)[:, np.newaxis]