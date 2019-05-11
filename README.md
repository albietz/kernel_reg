# Regularization of deep networks using the RKHS norm

This package provides a Pytorch implementation of various regularization methods for deep networks obtained via kernel methods,
by approximating the RKHS norm of the prediction function for a well-chosen kernel.
This is based on the following paper (see also [this paper](http://jmlr.org/papers/volume20/18-190/18-190.pdf) for theoretical background):

A. Bietti, G. Mialon, D. Chen, J. Mairal. [A Kernel Perspective for Regularizing Deep Neural Networks](https://arxiv.org/pdf/1810.00363.pdf). In *ICML*, 2019. 

The regularization penalties and constraints are implemented in `reg.py` and `spectral_norm.py`, and example usage is provided, e.g., in the script `main_smalldata.py`,
which was used to obtain the results on small datasets in the paper.

### Examples for regularization on small datasets

||f||_delta^2 (adversarial perturbation lower bound penalty) with epsilon = 1.0

```> python main_smalldata.py --reg_adv_perturbation_penalty --epsilon 1.0```

||\\nabla f||^2 (gradient lower bound penalty) with lambda = 0.1

```> python main_smalldata.py --reg_gradient_penalty --lmbda 0.1```

grad-l2 (gradient penalty on loss) with lambda = 0.1 + SN constraint with radius tau = 1.5

```> python main_smalldata.py --reg_loss_gradl2 --lmbda 0.1 --reg_project_sn --tau 1.5```

For other hyperparameters, some defaults are defined in experiment.py, but can also be given with options
(e.g. `--lr <lr>` for learning rate, or `--wd <wd>` for weight decay)
For example, for some weight decay on 5000 examples with ResNet-18:

```> python main_smalldata.py --experiment cifar10small5000_resnet --wd 5e-4```
