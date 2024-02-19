from numpy import cos, pi


def cosine_scheduler(t, T, s):
    return cos((pi / 2) * ((t / T) + s)/(1 + s)) ** 2
