import numpy as np


def cosine_annealing(step, total_steps, lr_max, lr_min):
    return lr_min + (lr_max - lr_min) * 0.5 * \
                (1 + np.cos(step / total_steps * np.pi))

def constant_divide(step, div_steps, factors, warmup=None):
    dfactor = 1

    if warmup is None and step <= warmup:
        factors = np.logspace(1e4, 1, num=warmup)
        return factors[step]

    for dstep, f in zip(div_steps, factors):
        if step <= dstep:
            return dfactor
        dfactor /= f

    return dfactor
