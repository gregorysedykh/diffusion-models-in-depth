# Denoising Diffusion Probabilistic Models

A **diffusion model** is a generative model that consists of two Markov
chains, one forward and one reverse.
Given an input (e.g. an image), the forward process will destroy the
information in the image by gradually adding Gaussian noise at each step
of the process {cite}`ho2020denoising`.
The reverse process' objective is to "invert" the forward process and
learn how to do so, starting from a noisy uninformative image and
step-by-step, estimating the structure of the information that was
destroyed in the image at each step and regenerating it until the first
step is reached, where we should obtain the original
image {cite}`ho2020denoising`.