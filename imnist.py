import numpy as np
import sys
import torch
from torch.utils import data
from torch.autograd import grad

from infimnist_dataset import InfiMNIST, InfiMNISTRaw


class InfimnistSubsetSampler(data.sampler.Sampler):
    def __init__(self, indices, num_transformations=1):
        self.indices = indices
        self.num_transformations = num_transformations

    def __iter__(self):
        tr = np.random.randint(self.num_transformations, size=len(self.indices))
        return (self.indices[i] + 60000 * tr[i] for i in torch.randperm(len(self.indices)))

    def __len__(self):
        return len(self.indices)


class InfimnistBatchedInfiniteSampler(data.sampler.Sampler):
    '''iterates a subset as follows:
    (im1_orig, im1_tr1, im1_tr2, ..., im1_tr<tr_per_ex-1>, im2_orig, im2_tr1, ...)
    '''
    def __init__(self, indices, num_transformations=1, tr_per_ex=10):
        self.indices = indices
        self.num_transformations = num_transformations
        self.tr_per_ex = tr_per_ex

    def __iter__(self):
        return iter(self.loop())

    def __len__(self):
        return 2 ** 31

    def loop(self):
        while True:
            perm = np.random.permutation(len(self.indices))
            for idx in perm:
                yield self.indices[idx]
                for _ in range(self.tr_per_ex - 1):
                    yield self.indices[idx] + 60000 * np.random.randint(1, self.num_transformations)


class InfimnistBatchedDeformInfiniteSampler(data.sampler.Sampler):
    '''iterates a subset of images along with fixed deformation vectors as follows:
    (im1_orig, im1_delta1, im1_delta2, ..., im1_delta<num_deformations>, im2_orig, im2_delta1, ...)
    note: to be used with InfiMNISTRaw, with tangent_only=True
    '''
    def __init__(self, indices, num_deformations=15):
        self.indices = indices
        self.num_deformations = num_deformations

    def __iter__(self):
        return iter(self.loop())

    def __len__(self):
        return 2 ** 31

    def loop(self):
        while True:
            perm = np.random.permutation(len(self.indices))
            for idx in perm:
                for i in range(self.num_deformations + 1):
                    # InfiMNISTRaw returns tangent deformation vectors for i > 0
                    yield self.indices[idx] + 60000 * i  


class StabilityPenalty:
    def __init__(self, model, device, batched_loader, n_ex_per_batch, tr_per_ex):
        self.model = model
        self.device = device
        self.batched_loader = iter(batched_loader)
        self.n_ex_per_batch = n_ex_per_batch
        self.tr_per_ex = tr_per_ex

        self.ims_out = None

    def prepare(self):
        ims, _ = next(self.batched_loader)
        ims = ims.to(self.device)
        assert ims.shape[0] == self.n_ex_per_batch * self.tr_per_ex

        preds = self.model(ims)

        # preds for the reference images
        predsref = preds[np.arange(self.n_ex_per_batch).repeat(self.tr_per_ex), ...]
        diffs = (preds - predsref).pow(2)
        idxs = diffs.argmax(0)
        self.ims_out = ims[torch.cat((idxs, self.tr_per_ex * (idxs // self.tr_per_ex)))].detach()

    def loss(self):
        preds = self.model(self.ims_out)
        k = preds.shape[1]
        assert preds.shape[0] == 2 * k
        loss = (preds[:k] - preds[k:]).pow(2).trace()
        return loss


class TangentGradientPenalty:
    def __init__(self, model, device, batched_loader, n_ex_per_batch, num_deformations, n_classes=10):
        self.model = model
        self.device = device
        self.batched_loader = iter(batched_loader)
        self.n_ex_per_batch = n_ex_per_batch
        self.num_deformations = num_deformations
        self.n_classes = n_classes

        self.ims_out = None
        self.deform_out = None
        self.ref_idxs = np.arange(self.n_ex_per_batch) * (self.num_deformations + 1)
        self.deform_idxs = 1 + np.arange(self.n_ex_per_batch).repeat(self.num_deformations) \
                 + np.arange(self.n_ex_per_batch * self.num_deformations)

    def prepare(self):
        ims, _ = next(self.batched_loader)
        ims = ims.to(self.device)
        assert ims.shape[0] == self.n_ex_per_batch * (self.num_deformations + 1)

        im_ref = ims[self.ref_idxs, ...]
        deform = ims[self.deform_idxs, ...]

        alpha = torch.zeros(self.n_classes, self.n_ex_per_batch, self.num_deformations).to(self.device).requires_grad_()

        # im + sum_i alpha_i * tangent_vector_i
        ims_deform = im_ref.view(torch.Size([1, self.n_ex_per_batch]) + ims.shape[1:]) + \
            (alpha.view(self.n_classes, self.n_ex_per_batch, self.num_deformations, 1, 1, 1)
              * deform.view(torch.Size([1, self.n_ex_per_batch, self.num_deformations]) + ims.shape[1:])
            ).sum(dim=2)

        preds = self.model(ims_deform.view(torch.Size([-1]) + ims.shape[1:])).view(self.n_classes, self.n_ex_per_batch, self.n_classes).sum(dim=1).trace()
        g, = grad(preds, alpha)
        idxs = (g.view(self.n_classes, self.n_ex_per_batch, -1) ** 1).sum(dim=2).argmax(1)
        self.ims_out = im_ref[idxs]
        self.deform_out = deform.view(torch.Size([self.n_ex_per_batch, self.num_deformations]) + ims.shape[1:])[idxs]

    def loss(self):
        alpha = torch.zeros(self.n_classes, self.num_deformations, 1, 1, 1).to(self.device).requires_grad_()
        preds = self.model(self.ims_out + torch.sum(alpha * self.deform_out, dim=1)).trace()
        g, = grad(preds, alpha, create_graph=True)
        return torch.sum(g ** 2)
