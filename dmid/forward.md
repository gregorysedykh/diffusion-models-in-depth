## Forward Process

The forward process is a Markov process that starts from the original
image $x_0$ and adds Gaussian noise during $T$ steps which results in a
more noisy image $x_t$ {cite}`@ho2020denoising`.
At each step $t \in \left[1, T\right]$, the noise added has variance
$\beta_t \in \{ \beta_1, \dots, \beta_T \}$.
This process is defined by $q$ which is a probability distribution that
takes as input an image $x_{t}$ and outputs its likelihood.
$$
\begin{aligned}
  q: \mathbb{R}^D \rightarrow [0, 1] 
\end{aligned}
$$ 
where $D$ is the data dimensionality (e.g. for a
$64 \times 64$ RGB image, $D = 64 \times 64 \times 3$).\
Pixels in the image $x_t$ are assumed to be mutually independent
conditionally to the previous timestep. This is reflected by the fact
that the covariance matrix ($\beta_t I$) is diagonal.

Formally, we obtain the following Markov process: 
$$
\begin{aligned}
  q\left(x_{1},\dots, x_{T} | x_0\right) = \prod_{t = 1}^T{q\left(x_t | x_{t - 1}\right)} \label{eq:forwardprocess}
\end{aligned}
$$ 
where: 
$$
\begin{aligned}
  q\left(x_t | x_{t-1}\right) = \mathcal{N}\left(x_t; \sqrt{1 - \beta_t}x_{t-1}, \beta_t I\right) \label{eq:forwardstep}
\end{aligned}
$$
Equation [\[eq:forwardstep\]](#eq:forwardstep){reference-type="ref"
reference="eq:forwardstep"} gives us a single step forward: given the
previous image $x_{t-1}$, $x_t$ is obtained by sampling a
$D$-dimensional Gaussian distribution with mean
$\sqrt{1 - \beta_t}x_{t-1}$ and variance $\beta_t I$.\
Equation [\[eq:forwardprocess\]](#eq:forwardprocess){reference-type="ref"
reference="eq:forwardprocess"} gives us the full forward process from
the original image $x_0$ to the final image $x_T$.

The reparametrisation trick says that for a univariate Gaussian
distribution where
$z \sim p\left(z | x, \mu \right) = \mathcal{N}\left(x, \sigma^2\right)$,
a valid reparametrisation would be $z = x + \sigma \epsilon$ where
$\epsilon \sim \mathcal{N}\left(0, 1\right)$ is just
noise {cite}`@kingma2022autoencoding`.\
In other words, sampling $z$ conditionally to $x$ from a Gaussian
distribution is equivalent to getting $z$ as $x$ with Gaussian noise of
variance $\sigma^2$ added to it. In this case, we can see that the image
$x_t$ becomes more random because of the Gaussian noise $\epsilon_t$
added to $x_{t-1}$ at each step, since we are sampling:
$$\begin{aligned}
  x_t &= q(x_t | x_{t-1}) \\
  &= \sqrt{1 - \beta_t} x_{t-1} + \sqrt{\beta_t}I \epsilon_t \qquad \epsilon_t \sim \mathcal{N}_D \left(0, I\right)
\end{aligned}$$ [It is important to note that as $T \rightarrow \infty$
and with a correct choice of $\beta_t$, $x_T$ will become a sample of an
isotropic Gaussian distribution
($\mathcal{N}\left(0, I\right)$)]{#isotropic} [@nichol2021improved; @sohldickstein2015deep].\
This is important for the reverse process, as it will allow us to take a
sample $x_T \sim \mathcal{N}\left(0, I\right)$ and reverse the forward
process to obtain the original image $x_0$ (however this cannot be done
so simply, as seen in section [4](#improvements){reference-type="ref"
reference="improvements"}) {cite}`@nichol2021improved`.

Ho et al. {cite}`@ho2020denoising` use the cascade of the reparametrisation
trick to be able to sample $x_t$ at any arbitrary step $t$ of the
forward process.\
This property is due to the fact that the space of Gaussian
distributions is closed under convolution. In other words, the addition
of two samples from Gaussian distributions is the same as obtaining a
sample from the convolution of the initial distributions (see
Appendix [8](#appendix:a){reference-type="ref" reference="appendix:a"}).

Let $\alpha_t = 1 - \beta_t$, then: $$\begin{aligned}
  x_t &\sim \mathcal{N}\left(\sqrt{1 - \beta_t} x_{t-1}, \beta_t I\right) \\
  x_t &= \sqrt{1 - \beta_t} x_{t-1} + \sqrt{\beta_t }I \epsilon_{t-1} \qquad \epsilon_{t - 1} \sim \mathcal{N}_D \left(0, I\right) \\
  &= \sqrt{\alpha_t} x_{t-1} + \sqrt{1 - \alpha_t} \epsilon_{t - 1}
\end{aligned}$$ From this, we can apply it again to $x_{t-1}$ and
obtain: $$\begin{aligned}
  x_{t-1} = \sqrt{\alpha_{t-1}} x_{t-2} + \sqrt{1 - \alpha_{t-1}} \epsilon_{t - 2}
\end{aligned}$$ Therefore, we get: $$\begin{aligned}
  x_t = \sqrt{\alpha_t \alpha_{t-1}} x_{t-2} + \sqrt{\alpha_t\left(1 - \alpha_{t-1}\right)} \epsilon_{t - 2} + \sqrt{1 - \alpha_t} \epsilon_{t - 1}
\end{aligned}$$ We can write: $$\begin{aligned}
  x_t &= \sqrt{\alpha_t \alpha_{t-1}} x_{t-2} + \mathcal{N}\left(0, \alpha_t\left(1 - \alpha_{t-1}\right)I \right) + \sqrt{1 - \alpha_t} \epsilon_{t - 1} \\
&= \sqrt{\alpha_t \alpha_{t-1}} x_{t-2} + \mathcal{N}\left(0, \alpha_t\left(1 - \alpha_{t-1}\right)I\right) + \mathcal{N}\left(0, (1 - \alpha_t) \right) \label{eq:convolution}
\end{aligned}$$\
The convolution of the two Gaussian distributions
$\mathcal{N}(\mu_1, \sigma_1^2)$ and $\mathcal{N}(\mu_2, \sigma_2^2)$
gives us a new Gaussian distribution with mean $\mu_1 + \mu_2$ and
variance $\sigma_1^2 + \sigma_2^2$ (see
Appendix [8](#appendix:a){reference-type="ref" reference="appendix:a"}):
$$\begin{aligned}
  x_t = \sqrt{\alpha_t \alpha_{t-1}} x_{t-2} + \mathcal{N}\left(0, \left(\alpha_t\left(1 - \alpha_{t-1}\right) + 1 - \alpha_t\right)I\right)
\end{aligned}$$\
Which finally gives us: $$\begin{aligned}
  x_t = \sqrt{\alpha_t \alpha_{t-1}} x_{t-2} + \sqrt{1 - \alpha_t \alpha_{t-1}} \bar{\epsilon}_{t - 2}
\end{aligned}$$ Where
$\epsilon_{t - 1}, \epsilon_{t - 2}, \ldots \sim \mathcal{N}\left(0, I\right)$
and $\bar{\epsilon}_{t - 2}$ is the convolution of the noise
distributions $\epsilon_{t - 1}$ and $\epsilon_{t - 2}$.

We can recursively apply this backwards until $x_0$.\
Let $\bar{\alpha}_t = \prod_{i=0}^{t}{\alpha_i}$: $$\begin{aligned}
  x_t = \sqrt{\bar{\alpha}_t} x_0 + \sqrt{1 - \bar{\alpha}_t} \epsilon
\end{aligned}$$ This will give us a way to directly compute a noisy
image $x_t$ at step $t$ of the forward process, given the original image
$x_0$ (we ignore the noise
$\epsilon \sim \mathcal{N}_D\left(0, I\right)$): $$\begin{aligned}
  q\left(x_t | x_0\right) = \mathcal{N}\left(\sqrt{\bar{\alpha}_t} x_0, \left(1 - \bar{\alpha}_t\right)I\right) \label{eq:quickforwardprocess}
\end{aligned}$$ We now also know that $1 - \bar{\alpha}_t$ is the
equivalent variance of the noise added to $x_0$ to obtain $x_t$ during
the forward process [@nichol2021improved].

Choosing the variances $\beta_t$ is an important step when developing
the forward process.\
Ho et al. [@ho2020denoising] chose to use a linear schedule starting
from $\beta_1 = 10^{-4}$ and increasing linearly until $\beta_T = 0.02$
in order for them to be small compared to the pixel values that were in
$\left[-1, 1\right]$.\
However we will see in section [4](#improvements){reference-type="ref"
reference="improvements"} that this is not the best choice, as this
destroys the images too quickly closer to the end of the process while
adding little to sample quality, which made the model be sub-optimal for
$64 \times 64$ and $32 \times 32$ images [@nichol2021improved].

Finally, Ho et al. [@ho2020denoising] chose $T = 1000$ in order to be
able to compare their model with Sohl et al.'s [@sohldickstein2015deep]
model, which also used $T = 1000$ for most image experiments.\
We will again see in section [4](#improvements){reference-type="ref"
reference="improvements"} that so many steps can make the sampling slow
and that there exist better choices [@nichol2021improved].