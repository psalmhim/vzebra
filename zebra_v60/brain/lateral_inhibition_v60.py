import torch

def lateral_inhibition_pair(left, right, lam=0.3):
    L = left - lam * right
    R = right - lam * left
    return L, R
