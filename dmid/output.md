**Diffusion Models in Depth: From a Theoretical and a Practical
Perspective**

**Gregory Igor Sedykh**

June 28th 2024

![image](images/informatics_en.png){width="40%"}\
Bachelor thesis for the degree of Bachelor of Science in Computer Science
Supervised by Prof. Stéphane Marchand-Maillet










# Score-based formulation

The DDPM paper [@ho2020denoising] allows us to predict either the next
denoised image, the mean of the next denoised image or the noise
itself.\
However they mention another formulation of the DDPM model that is
equivalent: a score-based generative modeling version

## Score matching

This formulation uses score matching [@hyvarinen2005], either denoising
score matching [@vincent2010denoising] or sliced score
matching [@song2019sliced].\
Essentially, score matching avoids optimizing the likelihood which can
sometimes be difficult to find because of normalisation constants, and
instead optimises a score function which is the gradient of the
log-likelihood.\
Suppose we have a probability density function $p_\theta (x)$ that
models an i.i.d. dataset $\{x_i\}_{i=1}^N$ of the p.d.f. $p(x)$.

Let $f_\theta (x)$ be a real valued function which is an unnormalized
density function and $Z_\theta = \int e^{-f_\theta (x)} dx$ be the
normalisation constant such that $\int p_\theta (x) \; dx = 1$, where
$p_\theta(x)$ is defined as: $$\begin{aligned}
  p_\theta (x) &= \frac{e^{-f_\theta (x)}}{Z_\theta}
\end{aligned}$$ However for a lot of models, $Z_\theta$ cannot be
computed [@luo2022understanding].\
The score function[^1]
$s_\theta : \mathbb{R}^D \rightarrow \mathbb{R}^D$ is defined
as [@song2020generative]: $$\begin{aligned}
  s_\theta (x) &= \nabla_x \log p(x)
\end{aligned}$$ If we try to train a model to optimise this score
function, we get: $$\begin{aligned}
  s_\theta (x) &= \nabla_x \log p_\theta(x) \\
  &= \nabla_x \log \frac{e^{-f_\theta (x)}}{Z_\theta} \\
  &= \nabla_x \left( -f_\theta (x) - \log Z_\theta \right) \\
  &= - \nabla_x f_\theta (x) - \nabla_x \log Z_\theta \\
  &= - \nabla_x f_\theta (x)
\end{aligned}$$ Since $Z_\theta$ is a constant with respect to $x$,
$\nabla_x \log Z_\theta = 0$ which means we don't need to compute the
intractable normalisation constant $Z_\theta$.\
This means we could train the score network $s_\theta$ to match the true
score function $s(x) = \nabla_x \log p(x)$.\
The model should then optimise the following Fisher divergence
$J(\theta)$ [@hyvarinen2005; @luo2022understanding]: $$\begin{aligned}
  J(\theta) = \frac{1}{2} \mathbb{E}_{x \sim p(x)} \left[ \| s_\theta (x) - s(x) \|_2^2 \right]
\end{aligned}$$ However, since we don't know the true score function
$s(x)$ which is the score function of the true data distribution $p(x)$,
another formulation was shown by Hyvärinen [@hyvarinen2005] to be
equivalent: $$\begin{aligned}
  J(\theta) = \mathbb{E}_{x \sim p(x)} \left[ \text{ tr}(\nabla_x s_\theta (x)) + \frac{1}{2} \| s_\theta (x) \|_2^2 \right]
\end{aligned}$$ This is the score matching objective we want to
minimise [@hyvarinen2005].\
It can also be computed using the datapoints with the following
formula [@hyvarinen2005]: $$\begin{aligned}
  J(\theta) \approx \frac{1}{N} \sum_{i=1}^N \text{ tr}(\nabla_x s_\theta (x_i)) + \frac{1}{2} \| s_\theta (x_i) \|_2^2
\end{aligned}$$ A problem that arises with this formulation is that the
Jacobian matrix $\nabla_x s_\theta (x)$ is not suitable for deep neural
networks as it requires backpropagation to compute all its diagonal
elements to find the trace [@song2020generative; @song2019sliced]. To
solve this problem, Song et al. [@song2020generative] propose either
denoising score matching or sliced score matching:

**Denoising score matching** [@vincent2010denoising] is a method that
slightly noises a datapoint and then score matches the denoised
datapoint. This is done by convolving datapoints sampled from the true
data distribution $p$ with a noising kernel $q_\sigma$ such that
$p(x) \approx q_\sigma (x)$. Given a datapoint $x$, we get a noised
datapoint $\tilde{x}$ by sampling from $q_\sigma(\tilde{x} | x)$.\
The score matching now estimates the score of the noised data
distribution $q_\sigma(\tilde{x})$, which leads us to the following
objective [@song2020generative; @vincent2010denoising]:
$$\begin{aligned}
  J(\theta) = \frac{1}{2} \mathbb{E}_{q_\sigma(\tilde{x} | x) p(x)} \left[ \| s_\theta (\tilde{x}) - \nabla_{\tilde{x}} \log q_\sigma(\tilde{x} | x) \|_2^2 \right]
\end{aligned}$$ **Sliced score matching** [@song2019sliced] projects the
vectors of the vector field $s_\theta (x)$ onto random directions to
obtain a scalar field. This will approximate the trace of the Jacobian
matrix and it becomes much easier to compute [@song2020generative].\
We get the following objective [@song2019sliced]: $$\begin{aligned}
  J(\theta) = \mathbb{E}_{p_v} \mathbb{E}_{x \sim p(x)} \left[ v^T \nabla_x s_\theta (x) \, v + \frac{1}{2} \| s_\theta (x) \|_2^2 \right]
\end{aligned}$$ The approximation of the Jacobian matrix
$v^T \nabla_x s_\theta (x) \, v$ can be rewritten as [@song2019sliced]:
$$\begin{aligned}
  v^T \nabla_x s_\theta (x) \, v &= v^T \nabla_x (v^T s_\theta (x))
\end{aligned}$$ The gradient of the inner product
$\nabla_x (v^T s_\theta (x))$ requires only one backpropagation pass and
requires another inner product with $v^T$ to approximate the
trace [@song2019sliced].

## Sampling

Using the score function mentioned above, we can sample from the model
by using Langevin dynamics [@song2020generative; @WelTeh2011a], which
will give us the definition of score-based generative
modeling [@song2020generative].

Let $x_0 \sim \pi (x)$ where $\pi (x)$ is an arbitrary distribution
(such as a Gaussian distribution), $\epsilon > 0$ a fixed step size and
$z_t \sim \mathcal{N}_D (0, I)$.\
Using Stochastic Gradient Langevin Dynamics
(SGLD) [@song2020generative; @WelTeh2011a], we can recursively sample
using the following formula: $$\begin{aligned}
  x_t &= x_{t-1} + \frac{\epsilon}{2} \nabla_x \log p(x_{t-1}) + \sqrt{\epsilon} z_t
\end{aligned}$$ When $T \rightarrow \infty$ and
$\epsilon \rightarrow 0$, the distribution of $x_T$ will converge to the
true data distribution $p(x)$ therefore the sample will become one from
the true data distribution [@song2020generative; @WelTeh2011a].\
$z_t$ is used to add some Gaussian noise at each step because if there
was no stochasticity at each step, the sampling would become
deterministic and all the samples would essentially be the same.\
Since we don't know the true data distribution $p(x)$, we can use the
score function $s_\theta (x)$ trained to match the true score function
$s(x)$ to sample from the model.\
$$\begin{aligned}
  x_t &= x_{t-1} + \frac{\epsilon}{2} s_\theta (x_{t-1}) + \sqrt{\epsilon} z_t
\end{aligned}$$ There is however a problem with this method: the score
function is well approximated at high density regions where there are a
lot of data points but not at low density
regions [@song2020generative].\
Song and Ermon [@song2020generative] show this in
figure [1](#fig:scores){reference-type="ref" reference="fig:scores"} by
comparing the real score function $\nabla_x \log p(x)$ and the estimated
score function $s_\theta (x)$ for a mixture of two Gaussian
distributions.

<figure id="fig:scores">
<div class="center">
<img src="images/highlowdensity.png" style="width:75.0%" />
<p>.<span id="fig:scores" label="fig:scores"></span></p>
</div>
<figcaption>Data scores represent the real score function <span
class="math inline">\(\nabla_x \log p(x)\)</span> and estimated scores
reperesent the estimated score function <span
class="math inline">\(s_\theta (x)\)</span>. We can see that the regions
in the red rectangles are the darker high density regions where the
score function is well approximated, while the lighter lower density
regions outside the red rectangles were not well estimated <span
class="citation" data-cites="song2020generative">(Y. Song and Ermon
2020)</span></figcaption>
</figure>

This means that when sampling using Langevin dynamics, the samples will
ultimately be incorrect [@song2020generative]. To correct this, Song and
Ermon [@song2020generative] use annealed Langevin dynamics.\
The idea is to add noise to the data points in order to improve the
score function estimation in low density regions [@songblog]. However
adding too much noise will make the score function estimation worse in
high density regions [@songblog].\
Therefore, they control the amount of noise added using a scale of
differents standard deviations $\sigma_i$ with $i = 1, 2, \ldots, L$
such that
$\sigma_1 < \sigma_2 < \cdots < \sigma_L$ [@songblog; @song2020generative].\
To sample, the same Langevin dynamics method is used going from
$i = L, L-1, \ldots, 1$ with $\sigma_i$ decreasing at each step. This is
the annealed Langevin dynamics method they used, and the results can be
seen in figure [2](#fig:langevin_samples){reference-type="ref"
reference="fig:langevin_samples"}, where we clearly see that the
annealed Langevin dynamics samples are much more similar to the original
data samples than the simple Langevin dynamics
samples [@songblog; @song2020generative].\

<figure id="fig:langevin_samples">
<div class="center">
<img src="images/langevin_samples.png" />
</div>
<figcaption>Left: Samples from data distribution. Middle: Samples from
model using Langevin dynamics. Right: Samples from model using annealed
Langevin dynamics <span class="citation"
data-cites="song2020generative">(Y. Song and Ermon
2020)</span>.</figcaption>
</figure>

## Noise Conditional Score Networks

Noise Conditional Score Networks (NCSN) are a score-based generative
model that use a neural network to estimate the score function of the
previously mentioned noised distribution and then use annealed Langevin
dynamics to sample from the model [@song2020generative].

Given the increasing sequence of standard deviations $\sigma_i$ with
$i = 1, 2, \ldots, L$, the perturbed data distribution is defined
as [@song2020generative]: $$\begin{aligned}
  q_{\sigma_i}(x) &= \int p(t) \, \mathcal{N}(x; t, \sigma_i^2) \; dt
\end{aligned}$$ Song and Ermon [@song2020generative] then define the
Noise Conditional Score Network $s_\theta$ that estimates the scores of
all the perturbed data distributions $q_{\sigma_i}(x)$ given a data
point $x$: $$\begin{aligned}
  s_\theta : \mathbb{R}^D &\times \mathbb{R} \rightarrow \mathbb{R}^D \\
  s_\theta (x, \sigma) &\approx \nabla_x \log q_\sigma (x) \qquad \forall \sigma \in \{ \sigma_1, \sigma_2, \ldots, \sigma_L \}
\end{aligned}$$ To train the NCSN, the authors chose denoising score
matching [@vincent2010denoising] as its goal is to estimate the score
function of a noised data distribution while being quicker than sliced
score matching [@song2020generative; @song2019sliced].

Let $q_\sigma (\tilde{x} | x) = \mathcal{N}(\tilde{x} | x, \sigma^2, I)$
be the noise distribution that noised the data point $x$ to get the
noised data point $\tilde{x}$.\
Its score function is given by [@song2020generative]: $$\begin{aligned}
  \nabla_{\tilde{x}} \log q_\sigma (\tilde{x} | x) &= - \frac{\tilde{x} - x}{\sigma^2} 
\end{aligned}$$ Let $\lambda (\sigma_i)$ be a positive coefficient
function, often chosen to be
$\lambda (\sigma_i) = \sigma_i^2$ [@songblog; @song2020generative].\
The denoising score matching objective is given
by [@song2020generative]: $$\begin{aligned}
  \mathcal{L}(\theta, \{ \sigma_i \}_{i=1}^L) &= \frac{1}{2L} \sum^L_{i=1} \lambda(\sigma_i) \, \mathbb{E}_p(x) \mathbb{E}_{\tilde{x} \sim \mathcal{N}(x, \sigma^2 I)} \left[ \| s_\theta (\tilde{x}, \sigma_i) + \frac{\tilde{x} - x}{\sigma_i^2} \|_2^2 \right]
\end{aligned}$$

To sample from the model, Song and Ermon [@song2020generative] define an
algorithm that uses the previously mentioned annealed Langevin dynamics:

:::: algorithm
::: algorithmic
$\{\sigma_i\}_{i=1}^L$, $\epsilon$, $T$ $\tilde{x}_0 \sim \pi(x)$
$\alpha_i \leftarrow \epsilon \cdot \sigma_i^2 / \sigma_L^2$
$z_t \sim \mathcal{N}(0, I)$
$\tilde{x}_t \leftarrow \tilde{x}_{t-1} + \frac{\alpha_i}{2} s_\theta (\tilde{x}_{t-1}, \sigma_i) + \sqrt{\alpha_i} z_t$
$\tilde{x}_0 \leftarrow \tilde{x}_T$ **return** $\tilde{x}_T$
:::
::::

## Equivalence between DDPM and NCSN

Ho et al.'s DDPM paper [@ho2020denoising] showed that the DDPM model is
equivalent to the NCSN model with denoising score matching and annealed
Langevin dynamics.\
This can be seen by the fact that sampling algorithm of the DDPM paper
([\[alg:sampling\]](#alg:sampling){reference-type="ref"
reference="alg:sampling"}) is in fact very similar to the sampling
algorithm of the NCSN paper
([\[alg:ncsn\]](#alg:ncsn){reference-type="ref" reference="alg:ncsn"})
but rather than estimating a score, the DDPM model estimates the
noise [@ho2020denoising].\
Similarly, the score function $s_\theta (x, \sigma)$ can be rewritten in
terms of the DDPM's forward process as [@weng2021diffusion]:
$$\begin{aligned}
    s_\theta (x, \sigma) \approx \nabla_x \log q_\sigma (x) &\Leftrightarrow s_\theta (x_t, t) \approx \nabla_{x_t} \log q(x_t) \qquad \text{(see \ref{eq:forwardstep})} \label{eq:scoreestimation} \\
    &= \mathbb{E}_{q (x_0)} \left[ \nabla_{x_t} \log q(x_t \, | \, x_0) \right] \\
    &= \mathbb{E}_{q (x_0)} \left[ \nabla_{x_t} \log \mathcal{N} (\sqrt{\bar{\alpha}_t} x_0, (1 - \bar{\alpha}_t) I) \right] \\
    &= \mathbb{E}_{q (x_0)} \left[ \nabla_{x_t} \log e^{- \frac{1}{2} \left( \frac{x_t - \sqrt{\bar{\alpha}_t} x_0}{\sqrt{1 - \bar{\alpha}_t}} \right)^T \left( \frac{x_t - \sqrt{\bar{\alpha}_t} x_0}{\sqrt{1 - \bar{\alpha}_t}} \right)} \right] \\
    &= \mathbb{E}_{q (x_0)} \left[ \Sigma^{-1} (x_t - \sqrt{\bar{\alpha}_t} x_0) \right] \\
    &= \mathbb{E}_{q (x_0)} \left[ \Sigma^{-1} \left(x_t - \sqrt{\bar{\alpha}_t} \left( \frac{x_t - \sqrt{1 - \bar{\alpha}_t} \epsilon_\theta}{\sqrt{\bar{\alpha}_t}} \right) \right) \right] \\
    &= \mathbb{E}_{q (x_0)} \left[ \Sigma^{-1} \left( \sqrt{1 - \bar{\alpha}_t} \epsilon_\theta \right) \right] \\
    &= - \frac{1}{1 - \bar{\alpha}_t}  \sqrt{1 - \bar{\alpha}_t} \epsilon_\theta \\
    &= - \frac{\epsilon_\theta}{\sqrt{1 - \bar{\alpha}_t}}
  
\end{aligned}$$ We can see that this quantity is used in sampling
algorithm ([\[alg:sampling\]](#alg:sampling){reference-type="ref"
reference="alg:sampling"}) of the DDPM paper [@ho2020denoising].

# Improvements upon DDPMs {#improvements}

As mentioned previously, the DDPM paper [@ho2020denoising] had certain
choices that limited its performance.\
Several papers [@nichol2021improved; @song2022denoising] following it
have improved upon the original DDPM in various ways.

## Improved likelihood

Ho et al. [@ho2020denoising] admit that their model, despite its high
sample quality, did not have a competitive log-likelihood in comparison
to other likelihood-based models such as VAEs and autoregressive
models [@nichol2021improved].

The *Improved Denoising Diffusion Probabilistic Models* paper published
by Nichol et al. [@nichol2021improved] in 2021 implemented certain
changes in order to improve the log-likelihood of the model not only on
simpler datasets such as CIFAR-10 [@cifar10] that the original DDPM was
trained on, but also on more complex datasets such as
ImageNet [@kingma2022autoencoding; @nichol2021improved; @oord2016conditional].

In order to improve the log-likelihood, Nichol et
al. [@nichol2021improved] started by investigating the fixed variance
$\Sigma_\theta (x_t, t) = \sigma_t^2 I = \beta_t I$ of the original DDPM
and whether or not it could be worth learning it. \
Recall that Ho et al. [@ho2020denoising] saw no difference in sample
quality when using a learned variance $\tilde{\beta}_t$ and therefore
opted for using the fixed variance $\beta_t$.\
Nichol et al. [@nichol2021improved] came to the same conclusion, finding
that as the number of steps $T$ increased (the authors also mentioned
that they used $T = 4000$ for their models rather than $T = 1000$ from
the original paper), the difference between $\beta_t$ and
$\tilde{\beta}_t$ remained very small. However, they were still
interested in the possibility of learning the variance in order to
improve the log-likelihood [@nichol2021improved].\
For this, they used an interpolation approach, where they parametrised
the variance as an interpolation between the fixed variance $\beta_t$
and a learned variance $\tilde{\beta}_t$ in the $\log$ domain, with $v$
an output vector from the model containing one component per dimension:
$$\begin{aligned}
  \Sigma_\theta (x_t, t) &= \exp \left( v \log \beta_t + (1 - v) \log \tilde{\beta}_t \right)
\end{aligned}$$ As the simplified training objective $L_{\text{simple}}$
used in the original DDPM paper [@ho2020denoising] did not include the
variance, Nichol et al. [@nichol2021improved] introduced a new training
objective $L_{\text{hybrid}}$, defined as: $$\begin{aligned}
  L_{\text{hybrid}} &= L_{\text{simple}} + \lambda L_{\text{VLB}}
\end{aligned}$$ with $\lambda = 0.001$ in order to keep
$L_{\text{simple}}$ as the main training
objective [@nichol2021improved].

At first, they found that the hybrid objective $L_{\text{hybrid}}$
achieved a better log-likelihoods and was easier to optimise than just
$L_{\text{VLB}}$, contrary to what they had
expected [@nichol2021improved].\
However, they obtained the best log-likelihoods by optimising
$L_{\text{VLB}}$ directly but with importance samping, which is a
technique where samples of a difficult distribution are taken from a
distribution that is easier to compute and then the original
distribution is approximated by weighted samples [@nichol2021improved].

They defined it as: $$\begin{aligned}
  L_{\text{VLB}} &= \mathbb{E}_{t \sim p_t} \left[ \frac{L_t}{p_t} \right]
  &\text{where } p_t \propto \sqrt{\mathbb{E}\left[ L_t^2 \right]} \text{ and } \sum p_t = 1
\end{aligned}$$\
Another change that Nichol et al. [@nichol2021improved] made was to use
a cosine noising schedule rather than the linear noising schedule used
in the original DDPM paper [@ho2020denoising].\
They explain this change by the fact that towards the end of the forward
process, the image is already very noisy and yet more noise is added
while not really improving the model, particularly for $32 \times 32$
and $64 \times 64$ images [@nichol2021improved].\
In effect, the input images are destroyed by noise very quickly close to
the start of the noising process. Nichol and
Dhariwal [@nichol2021improved] found that up to 20% of the reverse
process could be skipped while losing very little image quality and that
therefore the last noising steps are redundant.

The cosine schedule they used adds noise at a slower rate throughout the
process, in particular at the beginning and the end.\
They define the cosine schedule as: $$\begin{aligned}
  f(t) &= \cos \left( \frac{\frac{t}{T} + s}{1 + s} \cdot \frac{\pi}{2} \right)^2 \\
  \bar{\alpha}_t &= \frac{f(t)}{f(0)} \\
  \beta_t &= 1 - \frac{\bar{\alpha}_t}{\bar{\alpha}_{t-1}}
\end{aligned}$$ $T$ is the total number of timesteps, $t$ is the
timestep and $s$ is a small offset used to prevent $\beta_t$ from being
too small when $t = 0$ in order to allow the noise $\epsilon$ to be
predicted more accurately at the beginning of the
process [@nichol2021improved].

With these changes, Nichol et al. [@nichol2021improved] were able to
achieve a negative log-likelihood (NLL) of $2.94$ bits/dim on
CIFAR-10 [@cifar10] and $3.53$ bits/dim on
ImageNet [@oord2016conditional], compared to the original DDPM
paper [@ho2020denoising] which had a NLL of $3.70$ on CIFAR-10 and
$3.77$ on ImageNet [@nichol2021improved].

<figure>
<div class="center">
<img src="images/comparison.png" style="width:60.0%" />
</div>
<figcaption>Comparison of NLLs of different models <span
class="citation" data-cites="nichol2021improved">(Nichol and Dhariwal
2021)</span></figcaption>
</figure>

Moreover, Nichol et al. [@nichol2021improved] were able to improve
sampling speed due to the aforementioned changes, even though they
trained their model with $T = 4000$ timesteps.\
They were able to train their model with $4000$ timesteps, but use only
$100$ timesteps for sampling which brought them near-optimal FID
(Fréchet Inception Distance) scores when using the $L_{\text{hybrid}}$
training objective [@nichol2021improved].\
They had tried to reduce the number of sampling steps with the original
fixed variance versions from the DDPM paper [@ho2020denoising] but found
that the sample quality degraded a lot more [@nichol2021improved].

## Denoising Diffusion Implicit Models (DDIMs)

As mentioned previously, the DDPM paper [@ho2020denoising] has a forward
process and a reverse process which are both Markov processes.\
While the forward process can be easily calculated for a specific
timestep $t$ with equation
([\[eq:quickforwardprocess\]](#eq:quickforwardprocess){reference-type="ref"
reference="eq:quickforwardprocess"}), the reverse process still requires
1000 timesteps in the case of Ho et al.'s [@ho2020denoising] model, and
Nichol et al.'s model requires 100 and 4000 timesteps for sampling and
training respectively [@nichol2021improved].\
This is time-consuming and computationally expensive compared to GANs
which only require one forward pass through the network rather than
thousands to produce a sample [@song2022denoising].

To address this issue, Song et al. [@song2022denoising] introduced the
Denoising Diffusion Implicit Models (DDIMs) in 2020.\
The main objective of DDIMs was to accelerate sampling by making the
forward process non-Markovian, which would allow the reverse process to
require less iterations [@song2022denoising].

Song et al. [@song2022denoising] define a family $\mathcal{Q}$ of
forward processes. Each forward process in $\mathcal{Q}$ is indexed by
$\sigma  = \{ \sigma_1, \sigma_2, \ldots, \sigma_T \} \in \mathbb{R}^T_{\geq 0}$,
where $\sigma_t$ is the standard deviation of the noise added at
timestep $t$: $$\begin{aligned}
  q_{\sigma}(x_{1:T}| \, x_0) &= q_{\sigma}(x_T | \, x_0) \prod_{t=2}^{T} q_{\sigma}(x_{t-1}| \, x_t, x_0)
\end{aligned}$$ where $$\begin{aligned}
  q_{\sigma}(x_T | \, x_0) &= \mathcal{N}(\sqrt{\alpha_T} x_0, (1 - \alpha_T) I)
\end{aligned}$$ and for all $t > 1$: $$\begin{aligned}
  q_{\sigma}(x_{t-1} | \, x_t, x_0) &= \mathcal{N}\left( \sqrt{\alpha_{t-1}} x_0 + \sqrt{1 - \alpha_{t-1} - \sigma_t^2} \cdot \frac{x_t - \sqrt{\alpha_t} x_0}{\sqrt{1 - \alpha_t}}, \: \sigma_t^2 I \right)
\end{aligned}$$ Song et al. [@song2022denoising] note that when
$\sigma_t^2 = 0$, then $x_{t-1}$ is fixed and known in advance.\
Therefore, we can see that as Song et al. [@song2022denoising] state,
the value of $\sigma$ controls the stochasticity (or more simply, the
randomness) of the forward process.

The authors use Bayes' rule similarly to the original DDPM
paper [@ho2020denoising] in equation
([\[eq:ddpmbayes\]](#eq:ddpmbayes){reference-type="ref"
reference="eq:ddpmbayes"}), but instead to find the forward process
$q_\sigma(x_t | \, x_{t-1}, x_0)$: $$\begin{aligned}
  q_{\sigma}(x_t | \, x_{t-1}, x_0) &= \frac{q_\sigma (x_{t-1} | \, x_t, x_0) \: q_\sigma (x_t | \, x_0)}{q_\sigma (x_{t-1} | \, x_0)}
\end{aligned}$$ Since $q_\sigma (x_t | \, x_{t-1}, x_0)$ means that
$x_t$ not only depends on $x_{t-1}$ but also on $x_0$, we see that the
forward process is no longer Markovian, which was what the authors were
aiming to achieve [@song2022denoising].

However, since the reverse process in the original DDPM
paper [@ho2020denoising] was an estimation of the forward process which
was Markovian, a new reverse process was defined.\
Their idea of the reverse process is to sample an image $x_T$ like in
the original DDPM paper [@ho2020denoising], predict $x_0$ from $x_T$ and
then use this prediction of $x_0$ to sample
$x_{T-1}$ [@song2022denoising].\
This is repeated until we reach $x_1$ and then $x_0$ is predicted from
$x_1$ [@song2022denoising].\
Similar to equation
([\[eq:x0fromxt\]](#eq:x0fromxt){reference-type="ref"
reference="eq:x0fromxt"}) from the DDPM paper, Song et
al. [@song2022denoising] define the prediction of $x_0$ from $x_t$ as:
$$\begin{aligned}
  f_\theta^{(t)} (x_t) &= \frac{x_t - \sqrt{1 - \bar{\alpha}_t} \: \epsilon(x_t, t)}{\sqrt{\bar{\alpha}_t}}
\end{aligned}$$ Given this, the reverse process is defined
as [@song2022denoising]: $$\begin{aligned}
  p_\theta (x_T) &= \mathcal{N}_D(0, I) \\
  p_\theta^{(t)} (x_{t-1} | \, x_t) &= \begin{cases}
    \mathcal{N}(f_\theta^{(1)}(x_{1}), \sigma_1^2 I) & \text{if } t = 1 \\
    q_\sigma(x_{t-1} | \, x_t, f_\theta^{(t)}(x_t)) & \text{if } t > 1
  \end{cases}
\end{aligned}$$ Therefore, $x_{t-1}$ is sampled from
$x_t$ [@song2022denoising]: $$\begin{aligned}
  x_{t-1} &= \sqrt{\alpha_{t-1}} \left( \frac{x_t - \sqrt{1 - \bar{\alpha}_t} \: \epsilon(x_t, t)}{\sqrt{\bar{\alpha}_t}} \right) + \sqrt{1 - \alpha_{t-1} - \sigma_t^2} \cdot \epsilon_\theta (x_t, t) + \sigma_t \epsilon_t
\end{aligned}$$ In order to train the model, Song et
al. [@song2022denoising] define the following *variational inference
objective*: $$\begin{aligned}
  J_\sigma & (\epsilon_\theta) = \mathbb{E}_{q_\sigma (x_{0:T})} \left[ \log q_\sigma (x_{1:T} | \, x_0) - \log p_\theta (x_{0:T}) \right] \\
  &= \mathbb{E}_{q_\sigma (x_{0:T})} \left[ \log q_\sigma (x_T | \, x_0) + \sum_{t=2}^{T} \log q_\sigma (x_{t-1} | \, x_t, x_0) - \sum_{t=1}^{T} \log p_\theta^{(t)} (x_{t-1} | \, x_t) - \log p_\theta (x_T) \right]
\end{aligned}$$ This would mean that for different choices of $\sigma$,
we would have a different objective to optimise [@song2022denoising].\
However, the authors [@song2022denoising] prove that $J_\sigma$ is in
fact equivalent to $L_\gamma$ for certain choices of $\gamma$, where
$L_\gamma$ is defined as: $$\begin{aligned}
  L_\gamma (\epsilon_\theta) &= \sum_{t=1}^{T} \gamma_t \mathbb{E}_{x_0 \sim q(x_0), \epsilon_t \sim \mathcal{N}(0, I)}  \left[ \| \epsilon_\theta^{(t)} (\sqrt{\alpha_t} x_0 + \sqrt{1 - \alpha_t} \epsilon_t) - \epsilon_t \|_2^2 \right]
\end{aligned}$$ where $\gamma$ is a vector of length $T$ with positive
values that depend on $\alpha$.\
We easily see that when $\gamma = 1$, we get $L_1$ which is the training
objective from the original DDPM
paper [@ho2020denoising; @song2022denoising].\
Song et al. prove that $$\begin{aligned}
  \forall \sigma > 0, \, \exists \, \gamma \in \mathbb{R}_{\geq 0}^T \text{ and } \exists \, C \in \mathbb{R} \text{ s.t. } J_\sigma = L_\gamma + C \notag
\end{aligned}$$ This means that the optimal solution for $J_\sigma$ is
the same as the optimal solution for $L_1$, which means that the
training objective for DDIMs can be kept the same as the one for DDPMs
if the parameters being trained $\theta$ are not the same for all
timesteps [@song2022denoising].

In order to speed up the sampling process, Song et
al. [@song2022denoising] skip some timesteps in the reverse process.\
To do this, the forward process is redefined on a subset of timesteps of
size $S$, $\left\{ x_{\tau_1}, \dots, x_{\tau_S} \right\}$ and such that
$q(x_{\tau_i} | \, x_0) = \mathcal{N}(\sqrt{\alpha_{\tau_i}} x_0, (1 - \alpha_{\tau_i})I)$\
Since this is a subset of the timesteps, the training can be done all
timesteps and the sampling can be done on the
subset [@song2022denoising].

With the same amount of training timesteps $T = 1000$ and the same
training objective as the DDPM paper [@ho2020denoising], Song et
al. [@song2022denoising] were able to achieve solid FID scores on
CIFAR-10 [@cifar10] and CelebA [@liu2015faceattributes] datasets, with
far fewer sampling timesteps, ranging from $10$ to $100$
timesteps [@song2022denoising].\
A new hyperparameter $\eta \in \mathbb{R}_+$ is introduced to control
the interpolation between the DDPM $\sigma$ and the deterministic
$\sigma$ of the DDIM [@song2022denoising]: $$\begin{aligned}
  \sigma_{\tau_i} (\eta) &= \eta \sqrt{\frac{1 - \alpha_{\tau_{i-1}}}{1 - \alpha_{\tau_i}}} \sqrt{1 - \frac{\alpha_{\tau_i}}{\alpha_{\tau_{i-1}}}}
\end{aligned}$$ The results obtained by Song et al. [@song2022denoising]
show that the DDIM outperformed the DDPM for 10, 20, 50, 100 and 1000
timesteps and was only slightly worse at $1000$ timesteps and
$\sigma_{\tau_i} = \hat{\sigma}_{\tau_i} = \sqrt{1 - \alpha_{\tau_i} / \alpha_{\tau_{i-1}}}$

<figure>
<div class="center">
<img src="images/ddim_table.png" />
</div>
<figcaption>Comparison of FID scores (lower is better) of DDPMs and
DDIMs trained on <span class="math inline">\(S\)</span> timesteps, <span
class="math inline">\(S &lt; T\)</span> and <span
class="math inline">\(\eta = 0\)</span> is DDIM, <span
class="math inline">\(\eta = 1\)</span> and <span
class="math inline">\(\hat{\sigma}\)</span> are DDPMs <span
class="citation" data-cites="song2022denoising">(J. Song, Meng, and
Ermon 2022)</span></figcaption>
</figure>

## Guidance

Recall that when sampling from a DDPM or a DDIM, we start with an image
of Gaussian noise $x_T$ and we denoise the image until we get some image
$x_0$ that resembles an image from the original dataset. We may however
be interested in sampling something specific; for instance, from some
text.\
In this case it is helpful to guide the denoising process towards
whatever we specified.

### Classifier guidance

Classifier guidance was introduced by Dhariwal and Nichol in their
*Diffusion Models Beat GANs on Image Synthesis*
paper [@dhariwal2021diffusion] in order to improve the samples
generated.\
To do this, they trained a classifier $p_\Phi (y \, | \, x_t, t)$ on
noisy images [@dhariwal2021diffusion] since pre-trained classifiers were
are generally trained on dataset images and not on noised ones.\
The reverse process is then guided towards a class $y$ using the
gradient
$\nabla_{x_t} \log p_\Phi (y \, | \, x_t, t)$ [@dhariwal2021diffusion].
The authors prove [@dhariwal2021diffusion] that if we have a reverse
process such as that from the DDPM $p_\theta (x_t, x_{t+1})$, to
condition it on class $y$, we can sample from the following process
instead: $$\begin{aligned}
  p_{\theta, \Phi} (x_t \, | \, x_{t+1}, y) &= Z p_\theta (x_t \, | \, x_{t+1}) \, p_\Phi(y \, | \, x_t)
\end{aligned}$$ where $Z$ is a normalization constant.

Using the score-based formulation, we can easily find the new objective
to train the model with classifier guidance [@luo2022understanding].\
Remember that the score-based generative model learned to estimate
$\nabla_{x_t} \log q(x_t)$
([\[eq:scoreestimation\]](#eq:scoreestimation){reference-type="ref"
reference="eq:scoreestimation"}).\
If we want to condition the model on class $y$, we are interested in
estimating
$\nabla_{x_t} \log q(x_t \, | \, y)$ [@luo2022understanding].\
With Bayes' rule, we can rewrite this as [@luo2022understanding]:
$$\begin{aligned}
  q(x_t \, | \, y) &= \frac{q(y \, | \, x_t) \, q(x_t)}{q(y)}\\
  \Leftrightarrow \log q(x_t \, | \, y) &= \log \left[ \frac{q(y \, | \, x_t) \, q(x_t)}{q(y)} \right] \\
  \Leftrightarrow \nabla_{x_t} \log q(x_t \, | \, y) &= \nabla_{x_t} \log q(x_t) + \nabla_{x_t} \log q(y \, | \, x_t) - \nabla_{x_t} \log q(y) \\
  &= \nabla_{x_t} \log q(x_t) + \nabla_{x_t} \log q(y \, | \, x_t)
\end{aligned}$$ We can see that $\nabla_{x_t} \log q(x_t)$ is the
original unconditional score from the score-based generative model and
$\nabla_{x_t} \log q(y \, | \, x_t)$ is the gradient of the
log-likelihood of the class $y$ given the image
$x_t$ [@dhariwal2021diffusion; @luo2022understanding].\
To control the strength of the guidance, the authors introduce a
hyperparameter $\gamma$ that weights the classifier
gradient [@dhariwal2021diffusion; @luo2022understanding]:
$$\begin{aligned}
  \nabla_{x_t} \log q(x_t \, | \, y) &= \nabla_{x_t} \log q(x_t) + \gamma \nabla_{x_t} \log q(y \, | \, x_t) \label{eq:classifier_guidance}
\end{aligned}$$ Trivially, when $\gamma = 0$, the model doesn't use the
classifier and we recover the original score-based generative model.

Dhariwal and Nichol [@dhariwal2021diffusion] gave two algorithms to
train the model with classifier guidance: one for DDPMs and one that
also works for DDIMs that uses the score-based generative model
formulation:

:::: algorithm
::: algorithmic
*Model* $\epsilon_\theta (x_t, t)$, *Classifier*
$p_\Phi (y \, | \, x_t)$ with $y$ the class label Sample
$x_T \sim \mathcal{N}(0, I)$
$\hat{\epsilon} \leftarrow \epsilon_\theta (x_t, t) - \sqrt{1 - \bar{\alpha}_t} \; \nabla_{x_t} \, p_\Phi (y \, | \, x_t)$
$x_{t-1} \leftarrow \sqrt{\bar{\alpha}_{t-1}} \left( \frac{x_t - \hat{\epsilon} \sqrt{1 - \bar{\alpha}_t}}{\sqrt{\bar{\alpha}_t}} \right) + \hat{\epsilon} \sqrt{1 - \bar{\alpha}_{t-1}}$
**return** $x_0$
:::
::::

### Classifier-free guidance

Despite the impressive performance of the *ablated diffusion model*
(ADM) that used classifier guidance, Ho and Salimans made an improvement
by using classifier-free guidance. [@ho2022classifierfree]\
Classifier guidance requires a classifier to be trained on noisy images.
This means that not only does it requires training a classifier in
addition to the generative model, but it also requires the classifier to
be trained on noisy images for which pre-trained classifiers do not
generally exist [@ho2022classifierfree].\
Therefore, Ho and Salimans [@ho2022classifierfree] introduced a
classifier-free guidance method which does not require a trained
classifier at all.

The authors use an unconditional diffusion model $q(x_t)$ and a
conditional diffusion model $q(x_t \, | \, y)$ where $y$ is the class
label [@ho2022classifierfree; @luo2022understanding].\
Starting from equation
([\[eq:classifier_guidance\]](#eq:classifier_guidance){reference-type="ref"
reference="eq:classifier_guidance"}): $$\begin{aligned}
  \nabla_{x_t} \log q(x_t \, | \, y) &= \nabla_{x_t} \log q(x_t) + \gamma \nabla_{x_t} \log q(y \, | \, x_t) \\
  &= \nabla_{x_t} \log q(x_t) + \gamma \left( \nabla_{x_t} \log q (x_t \, | \, y) - \nabla_{x_t} \log q(x_t) \right) \\
  &= \nabla_{x_t} \log q(x_t) + \gamma \nabla_{x_t} \log q (x_t \, | \, y) - \gamma \nabla_{x_t} \log q(x_t) \\
  &= \gamma \nabla_{x_t} \log q (x_t \, | \, y) + \nabla_{x_t} \log q(x_t) - \gamma \nabla_{x_t} \log q(x_t) \\
  &= \gamma \nabla_{x_t} \log q (x_t \, | \, y) + (1 - \gamma) \nabla_{x_t} \log q(x_t)
\end{aligned}$$ $\gamma \nabla_{x_t} \log q (x_t \, | \, y)$ is the
score of the conditional model and
$(1 - \gamma) \nabla_{x_t} \log q(x_t)$ is the score of the
unconditional model [@luo2022understanding].\
However it isn't necessary to train a conditional model and an
unconditional model as we can just train the conditional model
$q(x_t \, | \, y)$ and simply ignore the conditioning information $y$
(by setting it to null, for example) to get the unconditional model
$q(x_t)$ [@ho2022classifierfree; @luo2022understanding].\

# Further development

## Cascaded Diffusion Models

Training a diffusion model on high-resolution images is computationally
expensive and therefore time-consuming.\
To address this, Ho et al. [@ho2021cascaded] introduced the cascaded
diffusion model (CDM) which is essentially a pipeline of diffusion
models where the first one generates a low-resolution image and the
subsequent ones are trained to upsample the image to a higher
resolution.\
This method is much faster than training a single diffusion model on the
high-resolution image as most of the important features for sampling are
already present in the low-resolution image which are fast to train,
whereas training at bigger resolutions doesn't add much more detail
compared to the computation costs [@ho2021cascaded].

As mentioned, the first model is trained to generate an image as we have
seen before. The subsequent models are trained using what Ho et al. call
conditioning augmentation, a type of data augmentation, where the model
is given a lower-resolution image and is trained on modified versions of
the image to generate a higher-resolution and overall higher-quality
image [@ho2021cascaded].\
For low resolution images, the authors use added Gaussian noise and use
Gaussian blurring for higher resolution images [@ho2021cascaded].

<figure>
<div class="center">
<img src="images/cascadedUnet.png" style="width:80.0%" />
</div>
<figcaption>U-Net used for each of the model of a CDM pipeline: the
upsampling models are given an image to upsample <span
class="math inline">\(x_t\)</span> and an upsampled low-resolution image
<span class="math inline">\(z\)</span>. For each of the blocks <span
class="math inline">\(1, \dots, K\)</span>, the image is
upsampled/downsampled by 2, the channels are given by the channel
multipliers <span class="math inline">\(M_{1, \dots, K}\)</span>. <span
class="citation" data-cites="ho2021cascaded">(Ho et al.
2021)</span></figcaption>
</figure>

<figure>
<div class="center">
<img src="images/cascading.png" style="width:80.0%" />
</div>
<figcaption>Example of a cascading diffusion pipeline for generating a
256 <span class="math inline">\(\times\)</span> 256 resolution image
conditioned on a class <span class="citation"
data-cites="ho2021cascaded">(Ho et al. 2021)</span></figcaption>
</figure>

As a result, the authors achieve an FID of $4.88$ on ImageNet
$256 \times 256$ with a CDM, where the first model generates a
$32 \times 32$ image, the second model upsamples it to $64 \times 64$
and the third model upsamples it to $256 \times 256$ [@ho2021cascaded].\
As an example, *Imagen* [@saharia2022photorealistic] from Google is a
popular cascaded diffusion model that can efficiently generate
high-resolution images.

## Latent Diffusion Models

Latent Diffusion Models (LDMs) were introduced by Rombach et al. in
their paper *High-Resolution Image Synthesis with Latent Diffusion
Models* [@rombach2022highresolution].\
Rather than do the forward and reverse processes in the pixel space, the
authors propose to do them in the latent space in order to speed up the
computation by reducing the dimensionality of the
data [@rombach2022highresolution].\
In order to do this, a Variational Autoencoder (VAE) is used to first
encode the image $x$ into a latent representation $z$. The forward
process is then applied to the latent representation $z$ until we obtain
$z_T$ [@rombach2022highresolution]. As for the reverse process, the
model (a U-Net in their paper) will denoise the latent representation
$z_T$ until we obtain $\tilde{z}$ which is then passed through the
decoder of the VAE to obtain the denoised image
$\tilde{x}$ [@rombach2022highresolution]. This means that given an image
$x$, the encoder $\mathcal{E}$ will encode $x$ into a latent
representation $z$: $$\begin{aligned}
  z &= \mathcal{E}(x)
\end{aligned}$$ Once the forward process and reverse process are done in
the latent space, the decoder $\mathcal{D}$ will decode the denoised
latent representation $\tilde{z}$ into the denoised image $\tilde{x}$:
$$\begin{aligned}
  \tilde{x} &= \mathcal{D}(\tilde{z})
\end{aligned}$$ From the training objective
([\[eq:l_simple\]](#eq:l_simple){reference-type="ref"
reference="eq:l_simple"}), we get a new training objective for
LDMs [@rombach2022highresolution]: $$\begin{aligned}
  L_{\text{LDM}} &= \mathbb{E}_{\mathcal{E}(x), \epsilon \sim \mathcal{N}(0, I), t} \left[ \| \epsilon - \epsilon_\theta (z_t, t) \|_2^2 \right]
\end{aligned}$$

<figure>
<div class="center">
<img src="images/latentdiffusion.png" style="width:75.0%" />
</div>
<figcaption>Latent Diffusion processes <span class="citation"
data-cites="rombach2022highresolution">(Rombach et al.
2022)</span></figcaption>
</figure>

In terms of results, Rombach et al. [@rombach2022highresolution] managed
to achieve an FID of 3.60 on ImageNet $256 \times 256$ with 271 days of
training on a Nvidia V100 GPU, which is better than the 3.85 FID
achieved by Dhariwal and Nichol [@dhariwal2021diffusion] with their
ADM-G and ADM-U (guidance and upsampling) which required 349 days of
training on a V100 [@rombach2022highresolution].

An example of a current LDM is *Stable
Diffusion* [@stablediffusion; @rombach2022highresolution], which has
become very popular nowadays and is easily available for use to the
general public.

## Diffusion Transformer

Recently, a new alternative to the U-Net architecture has been proposed
by Peebles and Xie in their paper *Scalable Diffusion Models with
Transformers* [@peebles2023scalable].\
As the name suggests, the Diffusion Transformer (DiT) architecture is
based on the Transformer architecture [@vaswani2023attention] used for
Natural Language Processing (NLP).

The architecture builds upon Latent Diffusion Models
(LDMs) [@rombach2022highresolution], since it takes as input the latent
representation $z$.\
The first step is called *patchify*: given that we have a spatial input,
this input is to be converted into tokens. Positional encodings are
subsequently added to the tokens to give the model information about the
spatial structure of the image [@peebles2023scalable].\
This information is then passed through DiT blocks (see
figure [3](#fig:DiT){reference-type="ref" reference="fig:DiT"}) before
going through layer normalisation.\
Finally, the output is passed through a linear layer and then reshaped
back in order to return the noise and the covariance matrix
$\Sigma$ [@peebles2023scalable].

<figure id="fig:DiT">
<div class="center">
<img src="images/DiT.png" />
</div>
<figcaption>Diffusion Transformer architecture <span class="citation"
data-cites="peebles2023scalable">(Peebles and Xie
2023)</span></figcaption>
</figure>

Compared to the previous experiments on class-conditioned ImageNet
$256 \times 256$, the DiT model has the best FID of them all with
$2.27$ [@peebles2023scalable], showing that the DiT brings significant
improvement to sample quality.

Newer models such as *Sora* [@videoworldsimulators2024] and *Stable
Diffusion 3* use this new DiT architecture and have achieved impressive
visual results.

# Implementation

A simple implementation of a DDPM for the MNIST dataset in PyTorch is
given in the Jupyter notebook `DMID.ipynb`. The code aims to be as close
as possible to the original DDPM paper [@ho2020denoising]. It can be run
on a CPU or a CUDA-enabled GPU.

The code provides an explanation and implementation of a U-Net
architecture similar to the original one.\
Ho et al. used a self-attention block not only between the two
ResNet-like blocks in the bottleneck, but also at the encoder and
decoder stages at $16 \times 16$ resolution images. They trained the
model as such for CIFAR-10, CelebA-HQ and LSUN
dataset [@ho2020denoising].\
`DMID.ipynb` contains a model without self-attention blocks as this
visually gave worse results on the MNIST dataset, possibly because of
overfitting.

The forward process uses the linear noising schedule $\beta$ and the
reverse process uses the same $1000$ timesteps from the original DDPM
paper [@ho2020denoising], however the code is easily modifiable to use a
different noising schedule (e.g. cosine noising schedule) and a
different number of timesteps. Similarly, a learning rate of $10^{-4}$
was used as it was found to work well for the MNIST dataset but this can
also be changed to the original $2 \times 10^{-4}$ learning rate.

<figure>
<div class="center">
<img src="images/forward.png" />
</div>
<figcaption>Noised images at different timesteps of the forward
process</figcaption>
</figure>

\
The model contains 7'339'297 parameters, much less than Ho et al.'s
CIFAR-10 model which has 35.7 mln parameters and the CelebA-HQ model
with its 114 mln parameters [@ho2020denoising].\
On an Apple M2 Pro chip using MPS, training takes around 70s per epoch
for $1000$ timesteps and $32 \times 32$ MNIST images and sampling 1
image takes around 13.4s.\
This means that the model is not so complex but still takes a long time
to train and sample from without significant resources like a TPU v3-8
used by Ho et al. [@ho2020denoising].

<figure>
<div class="center">
<img src="images/sample.png" style="width:50.0%" />
</div>
<figcaption>Samples generated by the model</figcaption>
</figure>

# Conclusion

Throughout this report, we analysed Denoising Diffusion Probabilistic
Models (DDPMs) in detail by covering their mathematical foundation,
which includes the two Markovian processes and the simplified training
objective. Additionally, we explored the U-Net architecture employed in
training these models.\
The score-based generative model was also analysed, particularly its use
of score matching to estimate the score of the data distribution, and
demonstrated its equivalence to DDPMs.\
Several improvements to DDPMs were discussed, notably Denoising
Diffusion Implicit Models (DDIMs) and the guidance methods, both playing
a crucial role in simultaneously improving sample quality and speeding
up the sampling process.\
Subsequently, we explored the present and the future of diffusion
models, especially the Diffusion Transformer (DiT) architecture which is
now becoming more popular for models available to the public.\
Finally, a straightforward implementation of a DDPM for the MNIST
dataset implemented using Python and PyTorch was presented, together
with some samples generated by the model.



[^1]: In ML, the score function used is the Stein score, which is the
    data gradient of the log likelihood. However in statistics, the
    Fisher score is used and is the parameter gradient of the log
    likelihood $s_F (x) = \nabla_\theta \log p_\theta (x)$
