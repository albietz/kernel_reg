import numpy as np
import os
import sys
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils import data

import model

CIFAR_ROOT = 'data'


def set_arg(args, k, v):
    if 'k' not in args or getattr(args, k) is None:
        setattr(args, k, v)


def get_arg(args, k, default=None):
    val = None
    if k in args:
        val = getattr(args, k)
    return val if val is not None else default


def load_experiment(args):
    if args.experiment == 'cifar10_vgg':
        return load_cifar10_vgg(args, num_examples=50000)
    if args.experiment == 'cifar10_resnet':
        return load_cifar10_resnet(args, num_examples=50000)
    if args.experiment == 'cifar10small5000_vgg':
        return load_cifar10_vgg(args, num_examples=5000)
    if args.experiment == 'cifar10small5000_resnet':
        return load_cifar10_resnet(args, num_examples=5000)
    if args.experiment == 'cifar10small1000_vgg':
        return load_cifar10_vgg(args, num_examples=1000)
    if args.experiment == 'cifar10small1000_resnet':
        return load_cifar10_resnet(args, num_examples=1000)
    # w/o data augmentation
    if args.experiment == 'cifar10_vgg_no_aug':
        return load_cifar10_vgg(args, num_examples=50000, aug=False)
    if args.experiment == 'cifar10small_vgg_no_aug':
        return load_cifar10_vgg(args, num_examples=5000, aug=False)
    if args.experiment == 'cifar10_resnet_no_aug':
        return load_cifar10_resnet(args, num_examples=50000, aug=False)
    if args.experiment == 'cifar10small_resnet_no_aug':
        return load_cifar10_resnet(args, num_examples=5000, aug=False)
    if args.experiment == 'cifar10small1000_vgg_no_aug':
        return load_cifar10_vgg(args, num_examples=1000, aug=False)
    if args.experiment == 'cifar10small1000_resnet_no_aug':
        return load_cifar10_resnet(args, num_examples=1000, aug=False)

    if args.experiment == 'imnist_vggstable_no_aug':
        return load_imnist_vggstable(args, num_examples=60000, aug=False)
    if args.experiment == 'imnist_vggstable_aug':
        return load_imnist_vggstable(args, num_examples=60000, aug=True)
    if args.experiment == 'imnist1000_vggstable':
        return load_imnist_vggstable(args, num_examples=1000, aug=get_arg(args, 'aug'))
    if args.experiment == 'imnist300_vggstable':
        return load_imnist_vggstable(args, num_examples=300, aug=get_arg(args, 'aug'))


def cifar10_loaders(args, num_examples, aug=True):
    args.n_classes = 10

    if aug:
        transform_train = transforms.Compose([
                transforms.RandomCrop(32, 4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
            ])
    else:
        transform_train = transforms.Compose([
                transforms.ToTensor(),
            ])

    cifar_root = get_arg(args, 'data_dir', 'data')
    trainset = torchvision.datasets.CIFAR10(root=cifar_root, train=True,
            download=True, transform=transform_train)

    if num_examples!=50000:
        args.num_examples = num_examples 

        idxs = np.arange(50000) # shuffle examples first
        seed = get_arg(args, 'shuf_seed', 0)
        rnd = np.random.RandomState(42 + seed)
        rnd.shuffle(idxs)
        train_idxs = idxs[:args.num_examples]
        val_idxs = idxs[-10000:]

        train_sampler = torch.utils.data.sampler.SubsetRandomSampler(train_idxs)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size,
                num_workers=2, sampler=train_sampler)

        val_sampler = torch.utils.data.sampler.SubsetRandomSampler(val_idxs)
        valloader = torch.utils.data.DataLoader(trainset, batch_size=args.test_batch_size,
                num_workers=2, shuffle=False, sampler=val_sampler)

    else:
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size,
                shuffle=True, num_workers=2)
        valloader = None

    transform_test = transforms.Compose([
            transforms.ToTensor(),
        ])

    testset = torchvision.datasets.CIFAR10(root=cifar_root, train=False,
            download=True, transform=transform_test)

    testloader = torch.utils.data.DataLoader(testset, batch_size=args.test_batch_size,
            shuffle=False, num_workers=2)


    loaders = {'train': trainloader, 'val': valloader, 'test': testloader}

    return loaders


def imnist_loaders(args, num_examples, aug=True):
    import imnist
    args.n_classes = 10

    num_transformations = 100000

    transtensor = transforms.ToTensor()

    trainset = imnist.InfiMNIST(
            train=True, num_transformations=num_transformations, transform=transtensor)

    if num_examples != 60000:
        args.num_examples = num_examples

        # shuffle examples first
        idxs = np.arange(60000)
        seed = get_arg(args, 'shuf_seed', 0)
        rnd = np.random.RandomState(42 + seed)
        rnd.shuffle(idxs)
        train_idxs = idxs[:args.num_examples]
        print(train_idxs)
        val_idxs = idxs[-10000:]

        if aug:
            train_sampler = imnist.InfimnistSubsetSampler(
                    indices=train_idxs,
                    num_transformations=num_transformations)
        else:
            train_sampler = imnist.InfimnistSubsetSampler(indices=train_idxs)

        trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size,
                num_workers=2, sampler=train_sampler)

        val_sampler = imnist.InfimnistSubsetSampler(indices=val_idxs)
        valloader = torch.utils.data.DataLoader(trainset, batch_size=args.test_batch_size,
                num_workers=2, sampler=val_sampler)

    else:
        if aug:
            train_sampler = imnist.InfimnistSubsetSampler(
                    indices=np.arange(60000),
                    num_transformations=num_transformations)
        else:
            train_sampler = imnist.InfimnistSubsetSampler(indices=np.arange(60000))
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size,
                sampler=train_sampler, num_workers=2)
        valloader = None


    testset = imnist.InfiMNIST(train=False, transform=transtensor)
    testloader = torch.utils.data.DataLoader(testset, batch_size=args.test_batch_size,
            shuffle=False, num_workers=2)

    loaders = {'train': trainloader, 'val': valloader, 'test': testloader}

    if get_arg(args, 'defense_stability'):
        set_arg(args, 'stability_ex_per_batch', 32)
        set_arg(args, 'stability_tr_per_ex', 32)

        stability_sampler = imnist.InfimnistBatchedInfiniteSampler(
                train_idxs, num_transformations=num_transformations,
                tr_per_ex=args.stability_tr_per_ex)
        stabilityloader = torch.utils.data.DataLoader(trainset,
                batch_size=args.stability_tr_per_ex * args.stability_ex_per_batch,
                sampler=stability_sampler,
                num_workers=2)
        loaders['stability'] = stabilityloader

    if get_arg(args, 'defense_gradtangent'):
        set_arg(args, 'stability_ex_per_batch', 32)
        set_arg(args, 'stability_num_deformations', 30)

        tangentset = imnist.InfiMNISTRaw(
                train=True, num_transformations=num_transformations, tangent_only=True)
        tangent_sampler = imnist.InfimnistBatchedDeformInfiniteSampler(
                train_idxs, num_deformations=args.stability_num_deformations)
        tangentloader = torch.utils.data.DataLoader(tangentset,
                batch_size=args.stability_ex_per_batch * (args.stability_num_deformations + 1),
                sampler=tangent_sampler,
                num_workers=2)
        loaders['tangent'] = tangentloader

    return loaders


def load_cifar10_vgg(args, num_examples=50000, aug=True):
    args.batch_size = 128
    args.test_batch_size = 256
    set_arg(args, 'lr', 0.05)
    args.momentum = 0.9
    set_arg(args, 'sched_gamma', 0.5)
    set_arg(args, 'sched_step', 40 if num_examples!=50000 else 30)
    set_arg(args, 'wd', 0.)

    loaders = cifar10_loaders(args, num_examples=num_examples, aug=aug)

    net = model.cifar_vgg11()

    return net, loaders


def load_cifar10_resnet(args, num_examples=50000, aug=True):
    args.batch_size = 128
    args.test_batch_size = 256
    set_arg(args, 'lr', 0.1)
    args.momentum = 0.9
    set_arg(args, 'sched_gamma', 0.5)
    set_arg(args, 'sched_step', 40 if num_examples!=50000 else 30)
    set_arg(args, 'wd', 0.)

    loaders = cifar10_loaders(args, num_examples=num_examples, aug=aug)

    net = model.cifar_resnet18()

    return net, loaders


def load_imnist_vggstable(args, num_examples=60000, aug=True):
    args.batch_size = 128
    args.test_batch_size = 512
    set_arg(args, 'lr', 0.05)
    args.momentum = 0.9
    set_arg(args, 'sched_gamma', 0.5)
    set_arg(args, 'sched_step', 40 if num_examples!=60000 else 30)
    set_arg(args, 'wd', 0.)

    loaders = imnist_loaders(args, num_examples=num_examples, aug=aug)

    net = model.mnist_vgg5_stable()

    return net, loaders
