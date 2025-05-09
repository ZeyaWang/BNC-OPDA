import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import scipy.stats as stats
from scipy.special import beta as beta_f
from scipy import linalg
from torch.autograd import Function
import matplotlib.pyplot as plt


## Beta Mixture model from https://github.com/PaulAlbert31/LabelNoiseCorrection/blob/master/utils.py
## Some functions from https://github.com/thuml/Separate_to_Adapt/blob/master/utilities.py



def weighted_mean(x, w):
    return np.sum(w * x) / np.sum(w)

def fit_beta_weighted(x, w):
    x_bar = weighted_mean(x, w)
    s2 = weighted_mean((x - x_bar)**2, w)
    alpha = x_bar * ((x_bar * (1 - x_bar)) / s2 - 1)
    beta = alpha * (1 - x_bar) /x_bar
    return alpha, beta

class BetaMixture1D(object):
    def __init__(self, max_iters=10,
                 alphas_init=[1, 2],
                 betas_init=[2, 1],
                 weights_init=[0.5, 0.5]):
        self.alphas = np.array(alphas_init, dtype=np.float64)
        self.betas = np.array(betas_init, dtype=np.float64)
        self.weight = np.array(weights_init, dtype=np.float64)
        self.max_iters = max_iters
        self.lookup = np.zeros(100, dtype=np.float64)
        self.lookup_resolution = 100
        self.lookup_loss = np.zeros(100, dtype=np.float64)
        self.eps_nan = 1e-12

    def likelihood(self, x, y):
        return stats.beta.pdf(x, self.alphas[y], self.betas[y])

    def weighted_likelihood(self, x, y):  # p(k)*p(l|k) == p(y)*p(x|y)
        return self.weight[y] * self.likelihood(x, y)

    def probability(self, x):
        return sum(self.weighted_likelihood(x, y) for y in range(2))

    def posterior(self, x, y):
        return self.weighted_likelihood(x, y) / (self.probability(x) + self.eps_nan)

    def responsibilities(self, x):
        r =  np.array([self.weighted_likelihood(x, i) for i in range(2)])
        r[r <= self.eps_nan] = self.eps_nan
        r /= r.sum(axis=0)
        return r

    def score_samples(self, x):
        return -np.log(self.probability(x))

    def fit(self, x):
        x = np.copy(x)
        eps = 1e-4
        x[x >= 1 - eps] = 1 - eps
        x[x <= eps] = eps
        self.score_history = []
        self.weight_0 = []
        self.weight_1 = []
        for i in range(self.max_iters):
            # E-step
            r = self.responsibilities(x)
            # M-step
            self.alphas[0], self.betas[0] = fit_beta_weighted(x, r[0])
            self.alphas[1], self.betas[1] = fit_beta_weighted(x, r[1])
            self.weight = r.sum(axis=1)
            self.weight /= self.weight.sum()

            neg_log_likelihood = np.sum([self.score_samples(i) for i in x])
            self.score_history.append(neg_log_likelihood)
            self.weight_0.append(self.weight[0])
            self.weight_1.append(self.weight[1])
        return self

    def predict(self, x):
        return self.posterior(x, 1) > 0.5

    def create_lookup(self, y):
        x_l = np.linspace(0+self.eps_nan, 1-self.eps_nan, self.lookup_resolution)
        lookup_t = self.posterior(x_l, y)
        lookup_t[np.argmax(lookup_t):] = lookup_t.max()
        self.lookup = lookup_t
        self.lookup_loss = x_l

    def look_lookup(self, x, loss_max, loss_min, testing=False):
        if testing:
            x_i = x
        else:
            x_i = x.clone().cpu().numpy()
        x_i = np.array((self.lookup_resolution * x_i).astype(int))
        x_i[x_i < 0] = 0
        x_i[x_i == self.lookup_resolution] = self.lookup_resolution - 1
        return self.lookup[x_i]

    def plot(self, title, save_dir, save_signal=False):
        x = np.linspace(0, 1, 100)
        plt.plot(x, self.weighted_likelihood(x, 0), label='known')
        plt.plot(x, self.weighted_likelihood(x, 1), label='unknown')
        plt.plot(x, self.probability(x), lw=2, label='mixture')
        plt.legend()
        if save_signal:
            plt.title(title)
            plt.savefig(save_dir, dpi=300)
        plt.close()

    def __str__(self):
        return 'BetaMixture1D(w={}, a={}, b={})'.format(self.weight, self.alphas, self.betas)

    def calculate_criteria(self):
        self.K = ( self.weight[0] * beta_f(self.alphas[1], self.betas[1])) / ( self.weight[1] * beta_f(self.alphas[0], self.betas[0]))
        self.criteria = ((np.log(self.K)) - (self.betas[1] - self.betas[0])) / ( (self.alphas[1]-self.alphas[0]) - (self.betas[1]-self.betas[0]) )
        print(self.K, self.alphas[1]-self.alphas[0], beta_f(2,3))
        return self.criteria



if __name__ == '__main__':
    bmm_model = BetaMixture1D()
    ent = np.array([0.2561, 0.9088, 0.2953, 0.1632, 0.0078, 0.5816, 0.2132, 0.2497, 0.5979,
        0.8173, 0.0101, 0.0228, 0.7917, 0.4247, 0.4070, 0.4106, 0.8729, 0.6061,
        0.5244, 0.2345, 0.0075, 0.5542, 0.0186, 0.3946, 0.4838, 0.8111, 0.6810,
        0.7138, 0.5603, 0.8423, 0.7147, 0.5340, 0.3641, 0.5556, 0.4638, 0.932])
    #ent = np.array([0.2,0.3,0.2,0.9,0.99])
    #res = bmm_model.posterior(ent,1)
    bmm_model.fit(ent)
    res = bmm_model.posterior(ent,1)
    res