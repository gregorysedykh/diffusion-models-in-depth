# Introduction

Diffusion models have gained widespread popularity since 2020, as models
such as *DALL-E*, *Stable Diffusion* {cite}`stablediffusion, rombach2022highresolution`
and *Midjourney*
have proven to be capable of generating high-quality images given a text
prompt. Furthermore, OpenAI's recent announcement of *Sora* has shown
that diffusion models have now also become highly capable of generating
minute long high-definition videos from a text
prompt {cite}`videoworldsimulators2024`.

These models date back to 2015, where the idea of a diffusion model
appeared, based on diffusion processes used in
thermodynamics {cite}`sohldickstein2015deep`.
Denoising Diffusion Probabilistic Models (DDPMs) were a development of
the original diffusion probabilistic model introduced in
2015 {cite}`ho2020denoising`.
Subsequently, OpenAI improved upon the original DDPMs which did not have
ideal log likelihoods {cite}`ho2020denoising` while also using less forward
passes and therefore speeding up the sampling
process {cite}`nichol2021improved`.
The most recent progress done by OpenAI has allowed their diffusion
models to obtain better metrics and better sample quality than
Generative Adversarial Networks (GANs) which were previously considered
the state-of-the-art in image generation {cite}`dhariwal2021diffusion`.

The fairly recent apparition of diffusion models means not only that
there is still a lot to be discovered about them, but also that progress
is being made rapidly.
The theory behind diffusion models was mainly founded when Ho et
al. {cite}`ho2020denoising` introduced their DDPMs in 2020, but many
improvements have been made upon their work since then.\
To understand what these models are and how they work, it is crucial to
understand how DDPMs were developed, what choices were made when
developing them and why these choices were made, as well as why changes
were made to the original model and how they were made.

This report aims to provide an overview of the theory behind diffusion
models, as well as a practical guide on how to implement a simple
diffusion model using PyTorch, in order to compare what the theory shows
us and what the practical implementation gives us.

```{bibliography}
```