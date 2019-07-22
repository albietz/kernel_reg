# Regularization of deep networks using the RKHS norm

This package provides a Pytorch implementation of various regularization methods for deep networks obtained via kernel methods,
by approximating the RKHS norm of the prediction function for a well-chosen kernel.
This is based on the following paper (see also [this paper](http://jmlr.org/papers/volume20/18-190/18-190.pdf) for theoretical background):

A. Bietti, G. Mialon, D. Chen, J. Mairal. [A Kernel Perspective for Regularizing Deep Neural Networks](https://arxiv.org/pdf/1810.00363.pdf). In *ICML*, 2019. 

The regularization penalties and constraints are implemented in `reg.py` and `spectral_norm.py`, and example usage is provided, e.g., in the script `main.py`,
which was used to obtain the results in the paper.

### Examples for regularization on small datasets

||f||_delta^2 (adversarial perturbation lower bound penalty) with epsilon = 1.0, on cifar10 with 1000 examples, with data augmentation and a VGG-11 network

```> python main.py --experiment cifar10small1000_vgg --reg_adv_perturbation_penalty --epsilon 1.0```

||\\nabla f||^2 (gradient lower bound penalty) with lambda = 0.1

```> python main.py --experiment cifar10small1000_vgg --reg_gradient_penalty --lmbda 0.1```

grad-l2 (gradient penalty on loss) with lambda = 0.1 + SN constraint with radius tau = 1.5

```> python main.py --experiment cifar10small1000_vgg --reg_loss_gradl2 --lmbda 0.1 --reg_project_sn --tau 1.5```

For other hyperparameters, some defaults are defined in experiment.py, but can also be given with options
(e.g. `--lr <lr>` for learning rate, or `--wd <wd>` for weight decay)
For example, for some weight decay on 5000 examples with ResNet-18:

```> python main.py --experiment cifar10small5000_resnet --wd 5e-4```

### Example for robustness

PGD-l2 with epsilon = 1.0 + spectral norm constraint with radius tau = 0.8 on Cifar10 with data augmentation (this gives the best robust accuracy in Figure 1(right) of the paper). The name `robust_vgg` is used to save the model for evaluating robustness later

```> python main.py --experiment cifar10_vgg --name robust_vgg --reg_pgdl2 --epsilon 1.0 --reg_project_sn --tau 0.8 --kappa 50```

Now, evaluate the robustness of the model:

```> python compute_adv.py --experiment cifar10_vgg --name robust_vgg```

For an l2 adversary with epsilon_test = 1.0, this model should give about 50% robust test accuracy for the default PGD attack.
We later found that this default l2 attack, which is used in the paper, is a bit weak, and when using a stronger attack (adding `--l2_step_size 0.5 --steps 500` in this last command), the robust accuracy of this same model drops to **41.68%** (note that `--l2_step_size 0.5 --steps 40` gave about the same).
The best reported accuracy we found is 39.9% in [Rony et al. (2019)](http://openaccess.thecvf.com/content_CVPR_2019/papers/Rony_Decoupling_Direction_and_Norm_for_Efficient_Gradient-Based_L2_Adversarial_Attacks_CVPR_2019_paper.pdf), indicating that perhaps our model improves on the state-of-the art, though they are using a different attack, which may be stronger.
If you can further break our model, please drop me an email (here is [the .pth file](http://pascal.inrialpes.fr/data2/abietti/comb_pgdl2_project_sn50_1.0_0.8.pth) for the VGG model).
