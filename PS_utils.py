"""
Author: Isaac Sheidlower, AABL Lab, Isaac.Sheidlower@tufts.edu
Github: https://github.com/IsaacSheidlower
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.distributions.normal import Normal

def sample_normal(agent, actor, observation, with_noise=False, max_action=2, env_only=False, kappa=.9):
    def get_dist(agent, actor, observation):
        observation = torch.Tensor(np.array(observation)).to('cpu')
        mu1, sigma1 = agent.actor.get_dist(observation)
        mu2, sigma2 = actor.actor.get_dist(observation)
        mu1 = mu1.detach().numpy()
        sigma1 = sigma1.detach().numpy()
        mu2 = mu2.detach().numpy()
        sigma2 = sigma2.detach().numpy()

        kl = np.maximum(np.tanh(np.log(np.sqrt(sigma2)/np.sqrt(sigma2)) + (sigma2+(mu1-mu2)**2)/(2*sigma2) - .5),0)
        for i in range(len(kl)):
            if kl[i] > kappa:
                kl[i] = kappa
        
        mu = mu1*(kl) + mu2*(1-(kl))
        sigma = np.zeros(2) 

        sigma = sigma2

        mu = torch.from_numpy(mu)
        sigma = torch.from_numpy(sigma)
        #print(mu, sigma)
        return Normal(mu, sigma), mu1, sigma1

    def get_dist_env(actor, observation):
        observation = torch.Tensor(np.array(observation)).to('cpu')
        mu2, sigma2 = actor.actor.get_dist(observation)
        mu2 = mu2.detach().numpy()
        sigma2 = sigma2.detach().numpy()

        mu = torch.from_numpy(mu2)
        sigma = torch.from_numpy(sigma2)
        #print(mu, sigma)
        return Normal(mu, sigma)

    if env_only is False:
        dist, mu, sigma = get_dist(agent, actor, observation)
        if with_noise:
            sample = dist.rsample().numpy()
        else:
            sample = dist.sample().numpy()
        #print(sample)
        sample = max_action * np.tanh(sample)
        return sample, dist, mu, sigma
    else:
        dist = get_dist_env(actor, observation)
        if with_noise:
            sample = dist.rsample().numpy()
        else:
            sample = dist.sample().numpy()
        #print(sample)
        sample = max_action * np.tanh(sample)
        return sample, dist

def sample_normal_multi(agent, actor, observation, with_noise=False, max_action=2):
    def get_dist(agent, actor, observation):
        observation = torch.Tensor(np.array(observation)).to('cpu')
        mu1, sigma1 = agent.actor.get_dist(observation)
        mu2, sigma2 = actor.actor.get_dist(observation)
        mu1 = mu1[0].detach().numpy()
        sigma1 = sigma1[0].detach().numpy()
        mu2 = mu2[0].detach().numpy()
        sigma2 = sigma2[0].detach().numpy()
        mu = (mu1 + mu2)/2
        sigma = np.zeros(2) 

        sigma = (sigma1+sigma2)/2 

        mu = torch.from_numpy(mu)
        sigma = torch.from_numpy(sigma)
        #print(mu, sigma)
        return Normal(mu, sigma), mu1, sigma1

    dist, mu, sigma = get_dist(agent, actor, observation)
    if with_noise:
        sample = dist.rsample().numpy()
    else:
        sample = dist.sample().numpy()
    #print(sample)
    sample = max_action * np.tanh(sample)
    return sample, dist, mu, sigma
