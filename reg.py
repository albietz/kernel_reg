import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import grad


def l2_project(X, r):
    '''project data X onto l2 ball of radius r.'''
    n = X.shape[0]
    norms = X.data.view(n, -1).norm(dim=1).view(n, 1, 1, 1)
    X.data *= norms.clamp(0., r) / norms
    return X


class AdvPerturbationPenalty(object):
    '''||f||_delta^2 : Adversarial perturbation lower bound penalty (multiclass).

    regularize using sum_k (f_k(x_i + delta_ik) - f_k(x_i))^2
    '''
    def __init__(self, model, epsilon, device, n_classes=10, step_size=None, steps=5):
        self.model = model
        self.epsilon = epsilon
        self.device = device
        self.n_classes = n_classes
        self.steps = steps
        if step_size is not None:
            self.step_size = step_size
        else:
            self.step_size = 1.5 * epsilon / steps

        self.cum_range = \
                torch.arange(0, n_classes * n_classes, n_classes + 1, dtype=torch.int64).to(device)
        self.cls_range = \
                torch.arange(n_classes, dtype=torch.int64).to(device)
        self.ims_out = None
        self.deltas_out = None

    def prepare(self, ims):
        n = ims.shape[0]
        k = self.n_classes

        deltas = torch.zeros(torch.Size([k]) + ims.shape, requires_grad=True, device=self.device)
        for step in range(self.steps):
            if deltas.grad is not None:
                deltas.grad.zero_()

            # maximize perturbed predictions
            preds = self.model((ims.unsqueeze(0) + deltas).view(torch.Size([k * ims.shape[0]]) + ims.shape[1:]))
            # only keep perturbations on corresponding classes
            loss = - preds.view(k, ims.shape[0], k).sum(dim=1).trace()
            loss.backward()

            # normalized gradient update (constant step-length)
            deltas.data.sub_(self.step_size * deltas.grad / deltas.grad.view(k, n, -1).norm(dim=2).view(k, n, 1, 1, 1)).clamp(min=1e-5)

            # projection on L2 ball
            norms = deltas.data.view(k, n, -1).norm(dim=2).view(k, n, 1, 1, 1)
            deltas.data *= norms.clamp(0., self.epsilon) / norms

        # find and save maximizers
        preds = self.model((ims.unsqueeze(0) + deltas).view(torch.Size([k * ims.shape[0]]) + ims.shape[1:]))
        preds = preds.view(k, n, k) - self.model(ims).unsqueeze(0)
        preds = preds.transpose(0, 1).contiguous().view(n, -1)[:,self.cum_range]
        idxs = preds.argmax(0)
        self.ims_out = ims[idxs]
        self.deltas_out = deltas.view(torch.Size([k * n]) + ims.shape[1:])[idxs + n * self.cls_range].detach()

    def loss(self):
        # compute loss for backprop, using saved maximizers
        loss = (self.model(self.ims_out + self.deltas_out) - self.model(self.ims_out)).pow(2).trace()
        return loss


class GradientPenalty(object):
    '''||\nabla f||^2 : Supremum gradient penalty (multiclass).

    regularize using lmbda * sum_k sup_{i} ||\nabla f_k(x_i)||^2
    '''
    def __init__(self, model, lmbda, n_classes=10):
        self.model = model
        self.lmbda = lmbda
        self.n_classes = n_classes

        self.ims_out = None

    def prepare(self, ims):
        n = ims.shape[0]
        imsv = ims.repeat(self.n_classes, 1, 1, 1).clone().requires_grad_()
        preds = self.model(imsv).view(self.n_classes, n, self.n_classes).sum(dim=1).trace()
        g, = grad(preds, imsv)
        idxs = (g.view(self.n_classes, n, -1) ** 2).sum(dim=2).argmax(1)
        self.ims_out = ims[idxs]

    def loss(self):
        ims = self.ims_out.clone().requires_grad_()
        preds = self.model(ims).trace()
        g, = grad(preds, ims, create_graph=True)
        return self.lmbda * torch.sum(g ** 2)


class LossAvgGradL2(object):
    '''grad-l2 : Gradient l2 norm penalty on the loss.'''
    def __init__(self, model, loss_fn, lmbda):
        assert loss_fn.reduction == 'sum', 'need a sum reduction for the loss'
        self.model = model
        self.loss_fn = loss_fn
        self.lmbda = lmbda

    def loss(self, ims, labels):
        n = ims.shape[0]
        imsv = ims.clone().requires_grad_()
        preds = self.model(imsv)
        xeloss = self.loss_fn(preds, labels)
        g, = grad(xeloss, imsv, create_graph=True)
        return self.lmbda * torch.sum(g ** 2) / n


class LossAvgGradL1(object):
    '''grad-l1 : Gradient l1 norm penalty on the loss.'''
    def __init__(self, model, loss_fn, lmbda):
        assert loss_fn.reduction == 'sum', 'need a sum reduction for the loss'
        self.model = model
        self.loss_fn = loss_fn
        self.lmbda = lmbda

    def loss(self, ims, labels):
        n = ims.shape[0]
        imsv = ims.clone().requires_grad_()
        preds = self.model(imsv)
        xeloss = self.loss_fn(preds, labels)
        g, = grad(xeloss, imsv, create_graph=True)
        return self.lmbda * g.norm(1) / n


class PGDL2(object):
    '''PGD with l2 perturbations.'''
    def __init__(self, model, loss_fn, epsilon, step_size=None, steps=5, rand=False):
        self.model = model
        self.loss_fn = loss_fn
        self.epsilon = epsilon
        self.steps = steps
        self.rand = rand
        if step_size is not None:
            self.step_size = step_size
        else:
            self.step_size = 1.5 * epsilon / steps

    def __call__(self, ims, labels):
        n = ims.shape[0]
        if self.rand:
            deltas = torch.randn_like(ims, requires_grad=True)
            deltas.data *= self.epsilon
            l2_project(deltas, self.epsilon)
        else:
            deltas = torch.zeros_like(ims, requires_grad=True)
        for step in range(self.steps):
            if deltas.grad is not None:
                deltas.grad.zero_()
            preds = self.model(ims + deltas)
            loss = -self.loss_fn(preds, labels)
            loss.backward()

            # normalized (constant step-length) gradient step
            deltas.data.sub_(self.step_size * deltas.grad / deltas.grad.view(n, -1).norm(dim=1).view(n, 1, 1, 1).clamp(min=1e-7))

            # projection on L2 ball
            l2_project(deltas, self.epsilon)

        return (ims + deltas).detach()


class PGDLinf(object):
    '''PGD with l-infinity perturbations.'''
    def __init__(self, model, loss_fn, epsilon, step_size=None, steps=5, rand=True):
        self.model = model
        self.loss_fn = loss_fn
        self.epsilon = epsilon
        self.steps = steps
        self.rand = rand
        if step_size is not None:
            self.step_size = step_size
        else:
            self.step_size = 0.01

    def __call__(self, ims, labels):
        n = ims.shape[0]
        if self.rand:
            deltas = torch.rand_like(ims, requires_grad=True)
            deltas.data = self.epsilon * (2 * deltas.data - 1)
        else:
            deltas = torch.zeros_like(ims, requires_grad=True)
        for step in range(self.steps):
            if deltas.grad is not None:
                deltas.grad.zero_()
            preds = self.model(ims + deltas)
            loss = self.loss_fn(preds, labels)
            loss.backward()

            # maximize linearization (gradient sign update)
            deltas.data.add_(self.step_size * deltas.grad.sign())
            # projection on Linf ball
            deltas.data.clamp_(-self.epsilon, self.epsilon)
        return (ims + deltas).detach()
