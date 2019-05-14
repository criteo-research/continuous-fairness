import numpy as np
import torch


# Independence of 2 variables
def joint_2(X, Y, density, damping=1e-10):
    X = (X - X.mean()) / X.std()
    Y = (Y - Y.mean()) / Y.std()
    data = torch.cat([X.unsqueeze(-1), Y.unsqueeze(-1)], -1)
    joint_density = density(data)

    nbins = int(min(50, 5. / joint_density.std))
    #nbins = np.sqrt( Y.size/5 )
    x_centers = torch.linspace(-2.5, 2.5, nbins)
    y_centers = torch.linspace(-2.5, 2.5, nbins)

    xx, yy = torch.meshgrid([x_centers, y_centers])
    grid = torch.cat([xx.unsqueeze(-1), yy.unsqueeze(-1)], -1)
    h2d = joint_density.pdf(grid) + damping
    h2d /= h2d.sum()
    return h2d


def hgr(X, Y, density, damping = 1e-10):
    h2d = joint_2(X, Y, density, damping=damping)
    marginal_x = h2d.sum(dim=1).unsqueeze(1)
    marginal_y = h2d.sum(dim=0).unsqueeze(0)
    Q = h2d / (torch.sqrt(marginal_x) * torch.sqrt(marginal_y))
    return torch.svd(Q)[1][1]


def chi_2(X, Y, density, damping = 1e-10):
    h2d = joint_2(X, Y, density, damping=damping)
    marginal_x = h2d.sum(dim=1).unsqueeze(1)
    marginal_y = h2d.sum(dim=0).unsqueeze(0)
    Q = h2d / (torch.sqrt(marginal_x) * torch.sqrt(marginal_y))
    return ((Q ** 2).sum(dim=[0, 1]) - 1.)


# Independence of conditional variables

def joint_3(X, Y, Z, density, damping=1e-10):
    X = (X - X.mean()) / X.std()
    Y = (Y - Y.mean()) / Y.std()
    Z = (Z - Z.mean()) / Z.std()
    data = torch.cat([X.unsqueeze(-1), Y.unsqueeze(-1), Z.unsqueeze(-1)], -1)
    joint_density = density(data)  # + damping

    nbins = int(min(50, 5. / joint_density.std))
    x_centers = torch.linspace(-2.5, 2.5, nbins)
    y_centers = torch.linspace(-2.5, 2.5, nbins)
    z_centers = torch.linspace(-2.5, 2.5, nbins)
    xx, yy, zz = torch.meshgrid([x_centers, y_centers, z_centers])
    grid = torch.cat([xx.unsqueeze(-1), yy.unsqueeze(-1), zz.unsqueeze(-1)], -1)

    h3d = joint_density.pdf(grid) + damping
    h3d /= h3d.sum()
    return h3d


def hgr_cond(X, Y, Z, density):
    damping = 1e-10
    h3d = joint_3(X, Y, Z, density, damping=damping)
    marginal_xz = h3d.sum(dim=1).unsqueeze(1)
    marginal_yz = h3d.sum(dim=0).unsqueeze(0)
    Q = h3d / (torch.sqrt(marginal_xz) * torch.sqrt(marginal_yz))
    return np.array(([torch.svd(Q[:, :, i])[1][1] for i in range(Q.shape[2])]))


def chi_2_cond(X, Y, Z, density):
    damping = 0
    h3d = joint_3(X, Y, Z, density, damping=damping)
    marginal_xz = h3d.sum(dim=1).unsqueeze(1)
    marginal_yz = h3d.sum(dim=0).unsqueeze(0)
    Q = h3d / (torch.sqrt(marginal_xz) * torch.sqrt(marginal_yz))
    return ((Q ** 2).sum(dim=[0, 1]) - 1.)


def K(x):
    return torch.Tensor([1./np.sqrt(2*np.pi)]).float() * torch.exp(-x**2/2)

def kde1D(x, centers, bandwidth=None):
    if bandwidth is None:
        n = x.size(0)
        d = 1 #- 1
        bandwidth = (n * (d + 2) / 4.)**(-1. / (d + 4))
        bandwidth = bandwidth ** 2
    #TODO : better handling of borders using the missing part of the kernel
    return K((centers.reshape((-1,1)) - x)/bandwidth).mean(dim=1)/bandwidth


def binary_renyi2_differentiable(X,Y, reg=1e-4, nbins=0, standardize = True):
    #TODO: check that Y has exactly only two values and that X Y are of same length
    cy = Y.sum().item()
    if cy <= 1 or cy >= Y.size(0)-1:
        return torch.tensor(0.)
    if standardize:
        X = (X - X.mean())/X.std()
    if nbins== 0:
        nbins = int(np.sqrt(Y.size(0)))
    xmin = X.min().item()
    xmax = X.max().item()
    if xmin == xmax :
        return torch.Tensor(0.)
    centers = torch.linspace(xmin, xmax, nbins)
    delta = centers[1]-centers[0]
    bw = 2*(xmax-xmin)/nbins
    h0 = kde1D(X[Y==0], centers, bw) #* delta * Y.size(0)
    h1 = kde1D(X[Y==1], centers, bw) #* delta * Y.size(0)
    h2d = torch.stack((h0,h1),dim=1) + reg
    marginal_x=h2d.sum(dim=1)
    marginal_y=h2d.sum(dim=0)
    Q = torch.diag(1/marginal_x)@(h2d**2)@torch.diag(1/marginal_y)
    return (Q.sum()-1).clamp(reg,1)
    
