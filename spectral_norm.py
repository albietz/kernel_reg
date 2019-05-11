import numpy as np
import torch
import copy


def sn_project_net(net, control):
    for n, p in net.named_parameters():
        if 'weight' in n:
            pp = torch.zeros_like(p.data).copy_(p.data)
            sn_project(p, control)


def sn_project(M, control):
    # computes the projection of a matrix M in the l_infinity ball
    # (decomposition + projection of the vector of singular values)
    M_hat = M.data.cpu().view(M.size(0), -1)
    M_size = M_hat.size()
    if M_size[0] < M_size[1]:
        M_hat = torch.t(M_hat)
    MtM = torch.mm(torch.t(M_hat), M_hat).cpu()
    MtM = MtM.numpy()
    sig, v = np.linalg.eigh(MtM + 0.000001 * np.identity(MtM.shape[0]))
    v = torch.from_numpy(v.astype('float32')).cuda()
    sig = sig.astype('float32')
    sig = np.sqrt(np.clip(sig, a_min=0.0, a_max=None))

    # projection on l_infinity ball
    s_proj = sig.clip(max=control)
    s_proj = torch.from_numpy(s_proj).type(torch.FloatTensor).cuda()
    sig = torch.from_numpy(sig).cuda()
    u = torch.mm(M_hat.cuda(), v)
    u /= sig

    M_hat = torch.mm(u, torch.mm(torch.diag(torch.squeeze(s_proj)), torch.t(v)))
    if M_size[0] < M_size[1]:
        M_hat = torch.t(M_hat)

    M.data = M_hat.view(M.size())


def sn_penalize_net(net, nb_it, v_net, svd):
    penalty = 0.
    for n, p in net.named_parameters():
        if 'weight' in n:
            if svd:
                u, v = compute_svd(p.data)
            else:
                u, v = power_iteration(nb_it, p.data, v_net[n])
                v_net[n] = v
            p_penalty = torch.matmul(u, torch.mv(p.view(p.size(0),-1),v)) ** 2
            penalty += p_penalty
    return penalty


def power_iteration(nb_it, M, v):
    with torch.no_grad():
        M_hat = M.view(M.size(0), -1)
        v_temp = copy.copy(v)
        for i in range(nb_it):
          v_temp = torch.mv(torch.t(M_hat), torch.mv(M_hat, v_temp))
        v_temp = v_temp / torch.norm(v_temp, p=2)
        u = torch.mv(M_hat, v_temp)
        sig = torch.norm(u, p=2)
        u = u / sig
    return u, v_temp


def compute_svd(M):
    with torch.no_grad():
        M_hat = M
        M_hat = M_hat.view(M.size(0), -1)
        M_size = M_hat.size()
        if M_size[0] < M_size[1]:
            M_hat = torch.t(M_hat)
        MtM = torch.mm(torch.t(M_hat), M_hat).cpu()
        MtM = MtM.numpy()
        sig, v = np.linalg.eigh(MtM + 0.000001 * np.identity(MtM.shape[0]))
        sig = sig.astype('float32')
        sig = np.sqrt(np.clip(sig, a_min=0.0, a_max=None))
        sig = torch.from_numpy(sig).cuda()
        v = torch.from_numpy(v.astype('float32')).cuda()
        u = torch.mm(M_hat, v)
        u /= sig
        # the eigenvalues are distributed in ascending order
        if M_size[0] < M_size[1]:
            u, v = v[:,-1], u[:,-1]
        else:
            v = v[:,-1]
            u = u[:,-1]
    return u, v

