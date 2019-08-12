'''
Code for the ICML 2019 paper "A Kernel Perspective for Regularizing Deep Neural Networks"
by A. Bietti, G. Mialon, D. Chen and J. Mairal.

This script runs training on (subsets of) cifar or mnist with our various regularization strategies.
The option `--experiment <exp>` is for choosing different architectures (1000 vs 5000 examples,
VGG or ResNet, with/without data augmentation, see experiment.py for a list). By default,
1000 examples with VGG and data augmentation are used.

Examples:
||f||_delta^2 (adversarial perturbation lower bound penalty) with epsilon = 1.0
> python main_smalldata.py --reg_adv_perturbation_penalty --epsilon 1.0

||\\nabla f||^2 (gradient lower bound penalty) with lambda = 0.1
> python main_smalldata.py --reg_gradient_penalty --lmbda 0.1

grad-l2 (gradient penalty on loss) with lambda = 0.1 + SN constraint with radius tau = 1.5
> python main_smalldata.py --reg_loss_gradl2 --lmbda 0.1 --reg_project_sn --tau 1.5

See below for other regularization strategies and parameters.
Some defaults are defined in experiment.py, but can also be given with options
(e.g. --lr <lr> for learning rate, or --wd <wd> for weight decay)
For example, for some weight decay on 5000 examples with ResNet-18:
> python main_smalldata.py --experiment cifar10small5000_resnet --wd 5e-4


The code for the various regularization strategies is in reg.py,
or in imnist.py for deformation-based penalties.

'''

import argparse
import logging
import math
import numpy as np
import os
import pickle
import time
import torch
import torch.nn as nn
import torch.optim as optim

import experiment
import reg
import spectral_norm

logging.basicConfig(level=logging.INFO)

MODEL_ROOT = 'models'
RESULT_ROOT = 'res'


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='small data training')
    parser.add_argument('--experiment', default='cifar10small1000_vgg', help='experiment')
    parser.add_argument('--interactive', action='store_true', help='do not save things models')
    parser.add_argument('--name', default='tmp', help='name of experiment variant')
    parser.add_argument('--data_dir', default='data', help='directory for (cifar) data')

    parser.add_argument('--num_epochs', default=501, type=int)
    parser.add_argument('--eval_delta', default=5, type=int,
                        help='evaluate every delta epochs')
    parser.add_argument('--shuf_seed', default=None, type=int,
                        help='seed for computing subsets of training set')
    parser.add_argument('--lr', default=None, type=float)
    parser.add_argument('--sched_step', default=None, type=int)
    parser.add_argument('--sched_gamma', default=None, type=float)
    parser.add_argument('--steps', default=5, type=int)
    parser.add_argument('--linf_step_size', default=None, type=float)
    parser.add_argument('--aug', action='store_true', help='imnist augmentation with deformations')

    parser.add_argument('--reg_adv_perturbation_penalty', action='store_true',
            help='regularize with adversarial perturbation penalty, ||f||_delta^2 (param: --epsilon)')
    parser.add_argument('--reg_gradient_penalty', action='store_true',
            help='regularize with max gradient penalty, ||\\nabla f||^2 (param: --lmbda)')
    parser.add_argument('--reg_pgdl2', action='store_true',
            help='regularize with pgd L2 (param: --epsilon)')
    parser.add_argument('--reg_pgdlinf', action='store_true',
            help='regularize with pgd Linf (param: --epsilon)')
    parser.add_argument('--reg_loss_gradl2', action='store_true',
            help='regularize with gradient L2 norm on the loss (param: --lmbda)')
    parser.add_argument('--reg_loss_gradl1', action='store_true',
            help='regularize with gradient L1 norm on the loss (param: --lmbda)')
    parser.add_argument('--reg_stability', action='store_true',
            help='imnist deformation stability penalty (param: --lmbdast)')
    parser.add_argument('--reg_gradtangent', action='store_true',
            help='imnist tangent deformation gradient penalty (param: --lmbdatan)')
    parser.add_argument('--reg_project_sn', action='store_true',
            help='regularize with spectral norm constraints (param: --tau)')
    parser.add_argument('--reg_penalize_sn_pi', action='store_true',
            help='spectral norm penalty using power iteration (param: --lmbdasn)')
    parser.add_argument('--reg_penalize_sn_svd', action='store_true',
            help='spectral norm penalty using SVD (param: --lmbdasn)')
    parser.add_argument('--reg_vat', action='store_true',
            help='regularize with VAT (param: --vat_alpha or --epsilon)')

    # options for regularization hyperparameters
    parser.add_argument('--epsilon', default=0.1, type=float,
                        help='perturbation epsilon (PGD L2/Linf, kernel multi')
    parser.add_argument('--wd', default=None, type=float,
                        help='weight decay parameter')
    parser.add_argument('--lmbda', default=0.1, type=float,
                        help='regularization parameter for gradient penalties')
    parser.add_argument('--lmbdast', default=0.1, type=float,
                        help='regularization parameter for stability penalty')
    parser.add_argument('--lmbdatan', default=0.1, type=float,
                        help='regularization parameter for tangent gradient penalty')
    parser.add_argument('--tau', default=0.5, type=float,
                        help='radius for spectral norm constraint')
    parser.add_argument('--kappa', default=2., type=float,
                        help='decay parameter for spectral norm constraints')
    parser.add_argument('--nb_it', default=2, type=int,
                        help='number of iteration for the power iteration method')
    parser.add_argument('--vat_xi', default=1e-6, type=float,
                        help='value of xi for VAT')
    parser.add_argument('--vat_alpha', default=1.0, type=float,
                        help='value of alpha for VAT')
    parser.add_argument('--lmbdasn', default=0.01, type=float,
                        help='regularization parameter for spectral norm penalties')
    parser.add_argument('--stability_ex_per_batch', default=None, type=int,
                        help='num ex per batch for stability penalty')
    parser.add_argument('--stability_tr_per_ex', default=None, type=int,
                        help='num tr per example for stability penalty')
    parser.add_argument('--stability_num_deformations', default=None, type=int,
                        help='num deformation vectors per example for tangent gradient penalty')
    args = parser.parse_args()

    torch.random.manual_seed(42)

    # load experiment
    logging.info('loading experiment {}'.format(args.experiment))
    net, loaders = experiment.load_experiment(args)
    trainloader = loaders['train']
    valloader = loaders['val']
    testloader = loaders['test']
    if not args.interactive:
        if not os.path.exists(os.path.join(MODEL_ROOT, args.experiment)):
            os.makedirs(os.path.join(MODEL_ROOT, args.experiment))
        if not os.path.exists(os.path.join(RESULT_ROOT, args.experiment)):
            os.makedirs(os.path.join(RESULT_ROOT, args.experiment))

    # device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    net.to(device)
    loss_fn = nn.CrossEntropyLoss()
    loss_fn_eval = nn.CrossEntropyLoss(reduction='sum')
    optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=args.momentum,
                          weight_decay=args.wd)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.sched_step, gamma=args.sched_gamma)

    ckp_fname = os.path.join(MODEL_ROOT, args.experiment, args.name + '.pth')
    if not args.interactive and os.path.exists(ckp_fname):
        state = torch.load(ckp_fname)
        net.load_state_dict(state['model'])
        optimizer.load_state_dict(state['optimizer'])
        scheduler.load_state_dict(state['scheduler'])

    res_fname = os.path.join(RESULT_ROOT, args.experiment, args.name + '.pkl')
    if not args.interactive and os.path.exists(res_fname):
        res = pickle.load(open(res_fname, 'rb'))
    else:
        res = [{} for epoch in range(args.num_epochs)]

    def eval(data='train'):
        if data == 'train':
            loader = trainloader
        elif data == 'val':
            loader = valloader
        else:
            loader = testloader
        count = 0
        lss = 0.
        acc = 0.
        for ims, labels in loader:
            count += labels.shape[0]
            ims, labels = ims.to(device), labels.to(device)
            preds = net(ims)
            if data == 'train':
                loss = loss_fn_eval(preds, labels)
                lss += loss.item()
            acc += (preds.argmax(1) == labels).sum().item()
        return lss / count, acc / count


    if args.reg_adv_perturbation_penalty:  # ||f||_delta^2
        perturb = reg.AdvPerturbationPenalty(net, epsilon=args.epsilon, n_classes=args.n_classes, 
            device=device, steps=args.steps)
    if args.reg_gradient_penalty:  # ||\nabla f||^2
        perturb = reg.GradientPenalty(net, lmbda=args.lmbda, n_classes=args.n_classes)
    if args.reg_pgdl2:
        perturb = reg.PGDL2(net, loss_fn, epsilon=args.epsilon, steps=args.steps)
    if args.reg_pgdlinf:
        perturb = reg.PGDLinf(net, loss_fn, epsilon=args.epsilon, steps=args.steps, step_size=args.linf_step_size)
    if args.reg_loss_gradl2:  # grad-l2
        perturb = reg.LossAvgGradL2(net, loss_fn_eval, lmbda=args.lmbda)
    if args.reg_loss_gradl1:  # grad-l1
        perturb = reg.LossAvgGradL1(net, loss_fn_eval, lmbda=args.lmbda)
    if args.reg_vat:
        import vat
        vat_penalty = vat.VAT(device, eps=args.epsilon, xi=args.vat_xi)
    if args.reg_stability:  # ||f||_\tau^2
        from imnist import StabilityPenalty
        stability_penalty = StabilityPenalty(
                net, device,
                batched_loader=loaders['stability'],
                n_ex_per_batch=args.stability_ex_per_batch,
                tr_per_ex=args.stability_tr_per_ex)
    if args.reg_gradtangent:  # ||D_\tau f||^2
        from imnist import TangentGradientPenalty
        gradtangent_penalty = TangentGradientPenalty(
                net, device,
                batched_loader=loaders['tangent'],
                n_ex_per_batch=args.stability_ex_per_batch,
                num_deformations=args.stability_num_deformations,
                n_classes=args.n_classes)

    if args.reg_penalize_sn_pi:
        # initialize singular vectors for power iteration
        v_net = {}
        for n, p in net.named_parameters():
            v_temp = torch.empty(p.data.view(p.size(0),-1).size(1), device='cuda')
            v_temp.normal_(0, 1)
            v_net[n] = v_temp

    # train
    for epoch in range(scheduler.last_epoch + 1, args.num_epochs):
        print('epoch', epoch, end=' ', flush=True)
        t = time.time()
        scheduler.step()
        for i, (ims, labels) in enumerate(trainloader):
            ims, labels = ims.to(device), labels.to(device)

            if args.reg_pgdl2 or args.reg_pgdlinf:
                ims = perturb(ims, labels)
            if args.reg_adv_perturbation_penalty or args.reg_gradient_penalty:
                perturb.prepare(ims)
            if args.reg_stability:
                stability_penalty.prepare()
            if args.reg_gradtangent:
                gradtangent_penalty.prepare()

            optimizer.zero_grad()
            preds = net(ims)

            loss = loss_fn(preds, labels)

            if args.reg_adv_perturbation_penalty or args.reg_gradient_penalty:
                penalty = perturb.loss()
                loss += penalty
            elif args.reg_loss_gradl2 or args.reg_loss_gradl1:
                penalty = perturb.loss(ims, labels)
                loss += penalty
            elif args.reg_vat:
                penalty = args.vat_alpha * vat_penalty(net, ims)
                loss += penalty

            if args.reg_stability:
                penalty = args.lmbdast * stability_penalty.loss()
                loss += penalty
            if args.reg_gradtangent:
                penalty = args.lmbdatan * gradtangent_penalty.loss()
                loss += penalty

            if args.reg_penalize_sn_pi:
                penalty = spectral_norm.sn_penalize_net(net, args.nb_it, v_net, svd=False)
                loss += args.lmbdasn * penalty
            elif args.reg_penalize_sn_svd:
                penalty = spectral_norm.sn_penalize_net(net, args.nb_it, v_net=None, svd=True)
                loss += args.lmbdasn * penalty

            loss.backward()

            for pi, p in enumerate(net.parameters()):
                assert torch.all(torch.isfinite(p.data))
                assert torch.all(torch.isfinite(p.grad))
            optimizer.step()

            if args.reg_project_sn:
                tau = args.tau * (1. + 4 * np.exp(-float(epoch) / args.kappa))
                spectral_norm.sn_project_net(net, tau)

            if i % 10 == 0:
                print('.', end='', flush=True)
        print()
        elapsed = time.time() - t
        t = time.time()

        # evaluate
        if epoch % args.eval_delta == 0:
            train_loss, train_acc = eval(data='train')
            if valloader is not None:
                _, val_acc = eval(data='val')
            else:
                val_acc = 0.
            _, test_acc = eval(data='test')
            logging.info('epoch: {} ({:.2f}+{:.2f}s), train loss: {:.4f}, train acc: {:.4f}, val acc: {:.4f}, test acc: {:.4f}'.format(
                epoch, elapsed, time.time() - t, train_loss, train_acc, val_acc, test_acc))
            res[epoch].update({'train_loss': train_loss, 'train_acc': train_acc,
                               'val_acc': val_acc, 'test_acc': test_acc})
            if args.reg_project_sn:
                res[epoch].update({'tau': tau})

            if not args.interactive:
                pickle.dump(res, open(res_fname, 'wb'))

                torch.save({'model': net.state_dict(),
                            'optimizer': optimizer.state_dict(),
                            'scheduler': scheduler.state_dict()}, ckp_fname)
