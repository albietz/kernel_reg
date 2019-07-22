import argparse
import numpy as np
import os
import pickle
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

import experiment
import reg

CIFAR_ROOT = 'data'
MODEL_ROOT = 'models'
RESULT_ROOT = 'resadv'

# values of epsilon for l2/linf to evaluate
epsilons = [0.1, 0.3, 1., 3.0]
epsilons_inf = [1, 2, 5, 8] # / 255 !


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='evaluate adversarial robustness')
    parser.add_argument('--experiment', default='cifar10_vgg')
    parser.add_argument('--name', default='tmp', help='name of experiment')
    parser.add_argument('--save', action='store_true', help='save norm file')
    parser.add_argument('--steps', default=40, type=int)
    parser.add_argument('--linf_step_size', default=0.01, type=float)
    parser.add_argument('--l2_step_size', default=None, type=float)
    args = parser.parse_args()

    net, loaders = experiment.load_experiment(args)
    testloader = loaders['test']

    # device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net.to(device)

    ckp_fname = os.path.join(MODEL_ROOT, args.experiment, args.name + '.pth')
    state = torch.load(ckp_fname)
    net.load_state_dict(state['model'])
    loss_fn = nn.CrossEntropyLoss()

    def eval(attack=None):
        loader = testloader
        count = 0
        acc = 0.
        for ims, labels in loader:
            count += labels.shape[0]
            ims, labels = ims.to(device), labels.to(device)
            if attack is not None:
                ims = attack(ims, labels)
            preds = net(ims)
            acc += (preds.argmax(1) == labels).sum().item()
        return acc / count

    res = {}
    acc_clean = eval()
    res['test_acc'] = acc_clean
    print('clean: {}'.format(acc_clean))

    accs_dict_l2 = {}
    accs_dict_linf = {}

    print('l2 - ', end='')
    for eps in epsilons:
        acc_adv = eval(attack=reg.PGDL2(net, loss_fn, epsilon=eps, steps=args.steps, step_size=args.l2_step_size))
        accs_dict_l2[eps] = acc_adv
        print('{}: {}'.format(eps, acc_adv), end=', ')
    print()
    print('linf - ', end='')
    for eps in epsilons_inf:
        acc_adv_inf = eval(attack=reg.PGDLinf(net, loss_fn, epsilon=eps / 255, steps=args.steps, step_size=args.linf_step_size))
        accs_dict_linf[eps] = acc_adv_inf
        print('{}: {}'.format(eps, acc_adv_inf), end=', ')
    print()

    res['adv_accs_l2'] = accs_dict_l2
    res['adv_accs_linf'] = accs_dict_linf

    print(res)

    if args.save:
        os.makedirs(os.path.join(RESULT_ROOT, args.experiment), exist_ok=True)
        fname = os.path.join(RESULT_ROOT, args.experiment, args.name + '.pkl')
        if os.path.exists(fname):
            res_final = pickle.load(open(fname, 'rb'))
            res_final.update(res)
        else:
            res_final = res
        pickle.dump(res_final, open(fname, 'wb'))
