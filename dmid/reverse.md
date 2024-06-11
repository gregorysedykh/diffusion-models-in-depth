## Reverse Process

As mentioned previously, with a correct choice of $\beta_t$, $x_T$ will
become the sample of an isotropic Gaussian distribution over
$\mathbb{R}^D$ as
$T \rightarrow \infty$ [@nichol2021improved; @sohldickstein2015deep] .\
This should mean that we can take a sample
$x_T \sim \mathcal{N}_D \left(0, I\right)$ and reverse the forward
process to obtain the original image $x_0$. However, this means that we
need to use the entire dataset in order to figure out
$q\left(x_{t-1} | x_t\right)$, which in practice cannot be
done [@nichol2021improved].\
Therefore, an estimation $p_\theta (x_{t-1} | x_t)$ of $q(x_{t-1}|x_t)$
is found using a trained neural network (see
[2.2.3](#subsubsec:model_architecture){reference-type="ref"
reference="subsubsec:model_architecture"}), with $\theta$ the parameters
of the neural network [@nichol2021improved].

The reverse process is defined as follows [@ho2020denoising]:
$$\begin{aligned}
  p_{\theta}\left(x_0, \dots, x_T\right) = p\left(x_T\right) \: \prod_{t=1}^T p_{\theta}\left(x_{t-1} | x_t\right) \label{eq:reverseprocess}
\end{aligned}$$ where $$\begin{aligned}
  p_{\theta}\left(x_{t-1} | x_t\right) = \mathcal{N}\left(x_{t-1}; \mu_{\theta}\left(x_t, t\right), \Sigma_{\theta}\left(x_t, t\right)\right) \label{eq:reversestep}
\end{aligned}$$ and $$\begin{aligned}
  p(x_T) &= \mathcal{N}_D\left(0, I\right) \qquad \text{\hyperlink{isotropic}{(isotropic Gaussian distribution)}} \notag
\end{aligned}$$
Equation [\[eq:reverseprocess\]](#eq:reverseprocess){reference-type="ref"
reference="eq:reverseprocess"} gives us the full reverse process,
starting from the final image $x_T$ (sampled from noise) to the original
image $x_0$.\
Equation [\[eq:reversestep\]](#eq:reversestep){reference-type="ref"
reference="eq:reversestep"} gives us the reverse process at step $t$,
where we estimate the noise that was added to the image at step $t$ of
the forward process in order to find the image $x_{t-1}$.\
Two parameters must be estimated to find the reverse process at step $t$
(from $x_t$ to $x_{t-1}$): the mean $\mu_{\theta}\left(x_t, t\right)$
and the variance $\Sigma_{\theta}\left(x_t, t\right)$.\
At first, Ho et al. [@ho2020denoising] used neural networks to estimate
both the mean $\mu_\theta$ and the variance $\Sigma_\theta$, but explain
that estimating the variance was not necessary as there was little
difference between the estimated variance
$\sigma_t^2 = \tilde{\beta}_t = \frac{1 - \bar{\alpha}_{t-1}}{1 - \bar{\alpha}_t}$
and the actual values $\sigma_t^2 = \beta_t$. Thus they set
$\Sigma_{\theta}\left(x_t, t\right) = \sigma_t^2 I = \beta_t I$.\
As for the mean, it is estimated using a neural network trained to
optimise a variational lower bound ($VLB$) of the negative log
likelihood $\mathbb{E}\left[- \log p_{\theta} \left(x_0\right)\right]$.

### Variational Lower Bound

Computing the negative log likelihood
$\mathbb{E}\left[- \log p_{\theta} \left(x_0\right)\right]$ requires
knowing $p_{\theta} (x_0)$ which means going through $T$ steps, which is
computationally expensive.\
Thus, Ho et al. [@ho2020denoising] use a variational lower bound ($VLB$
or more commonly known as $ELBO$) to estimate the negative log
likelihood [@sohldickstein2015deep].

The variational lower bound given by Dederik et
al. [@kingma2022autoencoding] is defined as: $$\begin{aligned}
  VLB &= \mathbb{E}_{q_\phi (x | z)} \left[ - \log q_\phi (z | x) + \log p_\theta(x, z) \right] \\[10pt]
  &= - D_{KL} (q_\phi (z | x) \| \, p_\theta (z)) + \mathbb{E}_{q_\phi (z | x)} \left[ \log p_\theta (x | z) \right]
\end{aligned}$$ where $X, Z$ are jointly distributed random variables
with distribution $p_\theta (x, z)$, $x \sim p_\theta$ and $q_\phi$ is
any distribution.

This gives us: $$\begin{gathered}
  VLB = \log p_{\theta}\left(x_0\right) - D_{KL}\left(q\left(x_{1:T}|x_0\right) \| p_{\theta}\left(x_{1:T}|x_0\right)\right)
\end{gathered}$$ Since we need to minimise the loss with the negative
log likelihood, we write: $$\begin{gathered}
  - VLB = - \log p_{\theta}\left(x_0\right) + D_{KL}\left(q\left(x_{1:T}|x_0\right) \| p_{\theta}\left(x_{1:T}|x_0\right)\right)
\end{gathered}$$ We know that the Kullback-Leibler divergence $D_{KL}$
is non-negative, which means that we can write: $$\begin{aligned}
  - \log p_{\theta}\left(x_0\right) &\leq - \log p_{\theta}\left(x_0\right) + D_{KL}\left(q\left(x_{1:T}|x_0\right) \| p_{\theta}\left(x_{1:T}|x_0\right)\right) \\[10pt]
  &= - \log p_{\theta}\left(x_0\right) + \mathbb{E}_q \left[\log \frac{q\left(x_{1:T}|x_0\right)}{p_{\theta}\left(x_{1:T}|x_0\right)}\right] \\[10pt]
  &= - \log p_{\theta}\left(x_0\right) + \mathbb{E}_q \left[\log \frac{q\left(x_{1:T}|x_0\right)}{\frac{p_{\theta}\left(x_0, x_{1:T}\right)}{p_{\theta}\left(x_0\right)}}\right] \\[10pt]
  &= - \log p_{\theta}\left(x_0\right) + \mathbb{E}_q \left[\log \frac{q\left(x_{1:T}|x_0\right)}{\frac{p_{\theta}\left(x_{0:T}\right)}{p_{\theta}\left(x_0\right)}}\right] \\[10pt]
  &= - \log p_{\theta}\left(x_0\right) + \mathbb{E}_q \left[\log \frac{q\left(x_{1:T}|x_0\right)}{p_{\theta}\left(x_{0:T}\right)} + \log {p_{\theta}\left(x_0\right)}\right] \\[10pt]
  &= \mathbb{E}_q \left[\log \frac{q\left(x_{1:T}|x_0\right)}{p_{\theta}\left(x_{0:T}\right)}\right]
\end{aligned}$$ so that we obtain that: $$\begin{aligned}
  - \log p_{\theta}\left(x_0\right) \leq \mathbb{E}_q \left[\log \frac{q\left(x_{1:T}|x_0\right)}{p_{\theta}\left(x_{0:T}\right)}\right]
\end{aligned}$$

We can subsequently define $L_{VLB}$ as the loss to be minimised:
$$\begin{gathered}
  L_{VLB} = \mathbb{E}_q \left[\log \frac{q\left(x_{1:T}|x_0\right)}{p_{\theta}\left(x_{0:T}\right)}\right] 
\end{gathered}$$ Ho et al. [@ho2020denoising] reformulate the loss in
order to reduce variance: $$\begin{aligned}
  L_{VLB} &= \mathbb{E}_q \left[\log \frac{q\left(x_{1:T}|x_0\right)}{p_{\theta}\left(x_{0:T}\right)}\right] \\
  &= \mathbb{E}_q \left[\log \frac{\prod_{t \geq 1} q\left(x_t | x_{t-1}\right)}{p \left(x_T\right) \prod_{t \geq 1} p_{\theta}\left(x_{t-1} | x_t\right)}\right] \\
  &= \mathbb{E}_q \left[-\log\left(p \left(x_T\right)\right) + \log \frac{\prod_{t \geq 1} q\left(x_t | x_{t-1}\right)}{\prod_{t \geq 1} p_{\theta}\left(x_{t-1} | x_t\right)}\right] \\
  &= \mathbb{E}_q \left[-\log\left(p \left(x_T\right)\right) + \log \prod_{t \geq 1} q\left(x_t | x_{t-1}\right) - \log \prod_{t \geq 1} p_{\theta}\left(x_{t-1} | x_t\right)\right] \\
  &= \mathbb{E}_q \left[-\log\left(p \left(x_T\right)\right) + \sum_{t \geq 1} \log q\left(x_t | x_{t-1}\right) - \sum_{t \geq 1} \log p_{\theta}\left(x_{t-1} | x_t\right)\right] \\
  &= \mathbb{E}_q \left[-\log\left(p \left(x_T\right)\right) + \sum_{t \geq 1} \log q\left(x_t | x_{t-1}\right) - \log p_{\theta}\left(x_{t-1} | x_t\right)\right] \\
  &= \mathbb{E}_q \left[-\log\left(p \left(x_T\right)\right) + \sum_{t \geq 1} \log \frac{q\left(x_t | x_{t-1}\right)}{p_{\theta}\left(x_{t-1} | x_t\right)}\right] \\
  &= \mathbb{E}_q \left[-\log\left(p \left(x_T\right)\right) + \log \frac{q\left(x_1 | x_0\right)}{p_{\theta}\left(x_0 | x_1\right)} + \sum_{t \geq 2} \log \frac{q\left(x_t | x_{t-1}\right)}{p_{\theta}\left(x_{t-1} | x_t\right)}\right] \\
  &= \mathbb{E}_q \left[-\log\left(p \left(x_T\right)\right) + \log \frac{q\left(x_1 | x_0\right)}{p_{\theta}\left(x_0 | x_1\right)} + \sum_{t \geq 2} \log \frac{q\left(x_{t-1} | x_t, x_0\right) q\left(x_t | x_0\right)}{{q\left(x_{t-1} | x_0\right)} p_{\theta}\left(x_{t-1} | x_t\right)}\right] \label{eq:conditionedq}\\
  &= \mathbb{E}_q \left[-\log\left(p \left(x_T\right)\right) + \log \frac{q\left(x_1 | x_0\right)}{p_{\theta}\left(x_0 | x_1\right)} + \sum_{t \geq 2} \log \frac{q\left(x_{t-1} | x_t, x_0\right)}{p_{\theta}\left(x_{t-1} | x_t\right)} + \sum_{k \geq 2} \log \frac{q\left(x_t | x_0\right)}{q\left(x_{t-1} | x_0\right)}\right] \\
  &= \mathbb{E}_q \left[-\log\left(p \left(x_T\right)\right) + \log \frac{q\left(x_1 | x_0\right)}{p_{\theta}\left(x_0 | x_1\right)} + \log \frac{q\left(x_T | x_0\right)}{q\left(x_1 | x_0\right)} + \sum_{t \geq 2} \log \frac{q\left(x_{t-1} | x_t, x_0\right)}{p_{\theta}\left(x_{t-1} | x_t\right)}\right] \\
  &= \mathbb{E}_q \left[-\log\left(p \left(x_T\right)\right) + \log \frac{q\left(x_T | x_0\right)}{p_{\theta}\left(x_0 | x_1\right)} + \sum_{t \geq 2} \log \frac{q\left(x_{t-1} | x_t, x_0\right)}{p_{\theta}\left(x_{t-1} | x_t\right)}\right] \\
  &= \mathbb{E}_q \left[- \log\left(p_{\theta}\left(x_0 | x_1\right)\right) + \log \frac{q\left(x_T | x_0\right)}{p_{\theta}\left(x_T\right)} + \sum_{t \geq 2} \log \frac{q\left(x_{t-1} | x_t, x_0\right)}{p_{\theta}\left(x_{t-1} | x_t\right)}\right] \\
  &= \mathbb{E}_q \: \Bigg[ D_{KL}\left(q\left(x_T | x_0\right) \| \: p\left(x_T\right)\right) \Bigg.  \\
  &\qquad \qquad \Bigg. + \sum_{t \geq 2} D_{KL}\left(q\left(x_{t-1} | x_t, x_0\right) \| \: p_{\theta}\left(x_{t-1} | x_t\right)\right) - \log\left(p_{\theta}\left(x_0 | x_1\right)\right) \Bigg] \notag
\end{aligned}$$

This formulation of the loss can be split up into 3 parts:
$$\begin{aligned}
  L_T &= D_{KL}\left(q\left(x_T | x_0\right) \| \: p\left(x_T\right)\right) \label{eq:lT}\\
  L_{1:T-1} &= \sum_{t \geq 2} D_{KL}\left(q\left(x_{t-1} | x_t, x_0\right) \| \: p_{\theta}\left(x_{t-1} | x_t\right)\right) \\
  L_0 &=  - \log\left(p_{\theta}\left(x_0 | x_1\right)\right) \\
  \text{so that } L_{VLB} &= L_T + L_{1:T-1} + L_0
\end{aligned}$$\
It is worth noting that at equation
([\[eq:conditionedq\]](#eq:conditionedq){reference-type="ref"
reference="eq:conditionedq"}), the terms of $q$ are conditioned on
$x_0$; this is in order to be able to easily compute the forward process
posteriors, as they are much easier to find given the original image
$x_0$.\
Ho et al. [@ho2020denoising] define it as: $$\begin{gathered}
  q\left(x_{t-1} | x_t, x_0\right) = \mathcal{N}\left(x_{t-1}; \tilde{\mu}_t \left(x_t, x_0\right), \tilde{\beta_t} I\right)
\end{gathered}$$

Using Bayes' theorem, we find:

$$\begin{aligned}
  q\left(x_{t-1} | x_t, x_0\right) &= q\left(x_t | x_{t-1}, x_0\right) \frac{q\left(x_{t-1}| x_0\right)}{q\left(x_t | x_0\right)} \label{eq:ddpmbayes}
\end{aligned}$$ Assuming that
$q\left(x_t | x_{t-1}, x_0\right) = \mathcal{N}\left(x_{t-1}; \tilde{\mu}_t \left(x_t, x_0\right), \tilde{\beta_t} I\right)$,
then: $$\begin{aligned}
  q\left(x_{t-1} | x_t, x_0\right) = \mathcal{N} & \left(x_t; \sqrt{1 - \beta_t}x_{t-1}, \beta_t I\right) \frac{\mathcal{N}\left(\sqrt{\bar{\alpha}_{t-1}} x_0, \left(1 - \bar{\alpha}_{t-1}\right)I\right)}{\mathcal{N}\left(\sqrt{\bar{\alpha}_t} x_0, \left(1 - \bar{\alpha}_t\right)I\right)} \\[15pt]
  = \frac{1}{\sqrt{2 \pi \left(1 - \bar{\alpha}_t\right) \left(1 - \bar{\alpha}_{t-1}\right) \left(\sqrt{1 - \beta_t}\right)}} & \, e^{- \frac{1}{2} \left(\frac{\left(x_t - \sqrt{1 - \beta_t}x_{t-1}\right)^2}{\beta_t} + \frac{\left(x_{t-1} - \sqrt{\bar{\alpha}_{t-1}}x_0\right)^2}{1 - \bar{\alpha}_{t-1}} - \frac{\left(x_t - \sqrt{\alpha}_t x_0\right)^2}{1 - \bar{\alpha}_{t}}\right)} \label{eq:bayeseq}
\end{aligned}$$

*Note*: we rewrote $q\left(x_t | x_{t-1}, x_0\right)$ as
$q\left(x_t | x_{t-1}\right)$
([\[eq:forwardstep\]](#eq:forwardstep){reference-type="ref"
reference="eq:forwardstep"}) since $x_0$ adds no new information not
already present in $x_{t-1}$.

We can ignore the constant factor in front of
([\[eq:bayeseq\]](#eq:bayeseq){reference-type="ref"
reference="eq:bayeseq"}) and thus obtain: $$\begin{aligned}
  &\propto \exp\left(- \frac{1}{2} \left(\frac{x_t^2 - 2 \sqrt{\alpha_t} x_{t-1} x_t + \alpha_t x_{t-1}^2}{\beta_t} \right. \right. \\
  & \hspace{4cm} + \frac{x_{t-1}^2 - 2 \sqrt{\bar{\alpha}_{t-1}} x_0 x_{t-1} + \bar{\alpha}_{t-1} x_0^2}{ 1 - \bar{\alpha}_{t-1}}  \notag \\
  & \left. \left. \hspace{8cm} - \frac{\left(x_t - \sqrt{\alpha}_t x_0\right)^2}{1 - \bar{\alpha}_{t}}\right) \right) \notag
\end{aligned}$$ $$\begin{aligned}
  &= \exp\left(- \frac{1}{2} \left(\frac{x_t^2}{\beta_t} - \frac{2 \sqrt{\alpha_t} x_t x_{t-1}}{\beta_t} + \frac{\alpha_t x_{t-1}^2}{\beta_t} + \frac{x_{t-1}^2}{1 - \bar{\alpha}_{t-1}} \right. \right. \\
  & \left. \left. \hspace{4cm}  - \frac{2 \sqrt{\bar{\alpha}_{t-1}} x_0 x_{t-1}}{1 - \bar{\alpha}_{t-1}} + \frac{\bar{\alpha}_{t-1} x_0^2}{1 - \bar{\alpha}_{t-1}} - \frac{\left(x_t - \sqrt{\alpha}_t x_0\right)^2}{1 - \bar{\alpha}_{t}}\right) \right) \notag \\[10pt]
  &= \exp\left(- \frac{1}{2} \left(x_{t-1}^2 \left(\frac{\alpha_t}{\beta_t} + \frac{1}{1 - \bar{\alpha}_{t-1}}\right) + x_{t-1} \left(\frac{-2 \sqrt{\alpha_t} x_t}{\beta_t} - \frac{2 \sqrt{\bar{\alpha}_{t-1}} x_0 }{1 - \bar{\alpha}_{t-1}}\right) + C\left(x_t, x_0\right)\right) \right) 
\end{aligned}$$ where: $$\begin{gathered}
  C\left(x_t, x_0\right) = \frac{x_t^2}{\beta_t} + \frac{\bar{\alpha}_{t-1} x_0^2}{1 - \bar{\alpha}_{t-1}} - \frac{\left(x_t - \sqrt{\alpha}_t x_0\right)^2}{1 - \bar{\alpha}_{t}}
\end{gathered}$$ Considering $\alpha_t = 1 - \beta_t$ and
$\bar{\alpha}_t = \prod_{i=0}^{t}{\alpha_i}$, we can simplify the
expression to: $$\begin{aligned}
  &= \exp\left(- \frac{1}{2} \left(x_{t-1}^2 \left(\frac{\alpha_t}{1 - \alpha_t} + \frac{1}{1 - \bar{\alpha}_{t-1}}\right) \right. \right. \\
  &\hspace{4cm} \left. \left. - 2\left(\frac{\sqrt{\alpha_t} x_t}{1 - \alpha_t} + \frac{\sqrt{\bar{\alpha}_{t-1}} x_0}{1 - \bar{\alpha}_{t-1}}\right) x_{t-1} + C\left(x_t, x_0\right) \right) \right) \notag \\[10pt]
  &= \exp\left(- \frac{1}{2} \left(x_{t-1}^2 \left(\frac{\alpha_t \left(1 - \bar{\alpha}_{t-1}\right) + \left(1 - \alpha_t\right)}{\left(1 - \alpha_t\right)\left(1 - \bar{\alpha}_{t-1}\right)}\right) \right. \right. \\
  &\hspace{4cm} \left. \left. - 2\left(\frac{\sqrt{\alpha_t} x_t}{1 - \alpha_t} + \frac{\sqrt{\bar{\alpha}_{t-1}} x_0}{1 - \bar{\alpha}_{t-1}}\right) x_{t-1} + C\left(x_t, x_0\right) \right) \right) \notag \\[10pt]
  &= \exp\left(- \frac{1}{2} \left(x_{t-1}^2 \left(\frac{\alpha_t - \alpha_t \bar{\alpha}_{t-1} + 1 - \alpha_t}{\left(1 - \alpha_t\right)\left(1 - \bar{\alpha}_{t-1}\right)}\right) \right. \right. \\
  &\hspace{4cm} \left. \left. - 2\left(\frac{\sqrt{\alpha_t} x_t}{1 - \alpha_t} + \frac{\sqrt{\bar{\alpha}_{t-1}} x_0}{1 - \bar{\alpha}_{t-1}}\right) x_{t-1} + C\left(x_t, x_0\right) \right) \right) \notag \\[10pt]
  &= \exp\left(- \frac{1}{2} \left(x_{t-1}^2 \left(\frac{1 - \bar{\alpha}_{t-1}}{\left(1 - \alpha_t\right)\left(1 - \bar{\alpha}_{t-1}\right)}\right) \right. \right. \\
  &\hspace{4cm} \left. \left. - 2\left(\frac{\sqrt{\alpha_t} x_t}{1 - \alpha_t} + \frac{\sqrt{\bar{\alpha}_{t-1}} x_0}{1 - \bar{\alpha}_{t-1}}\right) x_{t-1} + C\left(x_t, x_0\right) \right) \right) \notag \\[10pt]
  &= \exp\left(- \frac{1}{2} \left( \left(\frac{1 - \bar{\alpha}_{t-1}}{\left(1 - \alpha_t\right)\left(1 - \bar{\alpha}_{t-1}\right)}\right)  \right. \right. \\
  &\hspace{4cm} \left. \left. \left(x_{t-1}^2 - 2 \frac{\frac{\sqrt{\alpha_t}x_t}{1 - \alpha_t} + \frac{\sqrt{\bar{\alpha}_{t-1}} x_0}{1 - \bar{\alpha}_{t-1}}}{\frac{1 - \bar{\alpha}_{t-1}}{\left(1 - \alpha_t\right)\left(1 - \bar{\alpha}_{t-1}\right)}} x_{t-1} + \frac{C\left(x_t, x_0\right)}{\frac{1 - \bar{\alpha}_{t-1}}{\left(1 - \alpha_t\right)\left(1 - \bar{\alpha}_{t-1}\right)}}\right) \right) \right) \notag \\[10pt]
  &= \exp \left(- \frac{1}{2} \left( \left(\frac{1 - \bar{\alpha}_{t-1}}{\left(1 - \alpha_t\right)\left(1 - \bar{\alpha}_{t-1}\right)}\right)  \right. \right. \\
  &\hspace{2cm} \left. \left. \left(x_{t-1}^2 - 2 \frac{\sqrt{\alpha_t} \left(1 - \bar{\alpha}_{t-1}\right) x_t + \sqrt{\bar{\alpha}_{t-1}}\left(1 - \alpha_t\right) x_0}{1 - \bar{\alpha}_t} + \frac{C\left(x_t, x_0\right)}{\frac{1 - \bar{\alpha}_{t-1}}{\left(1 - \alpha_t\right)\left(1 - \bar{\alpha}_{t-1}\right)}}\right) \right) \right) \notag 
\end{aligned}$$\
As for the final term, since: $$\begin{aligned}
  C \left(x_t, x_0\right) &= \frac{x_t^2}{\beta_t} + \frac{\bar{\alpha}_{t-1} x_0^2}{1 - \bar{\alpha}_{t-1}} - \frac{\left(x_t - \sqrt{\alpha}_t x_0\right)^2}{1 - \bar{\alpha}_{t}} \\
  &= \frac{x_t^2}{1 - \alpha_t} + \frac{\bar{\alpha}_{t-1} x_0^2}{1 - \bar{\alpha}_{t-1}} - \frac{x_t^2 - 2 \sqrt{\alpha}_t x_t x_0 + \alpha_t x_0^2}{1 - \bar{\alpha}_{t}}
\end{aligned}$$

We obtain:
$$C\left(x_t, x_0\right) / \frac{1 - \bar{\alpha}_{t-1}}{\left(1 - \alpha_t\right)\left(1 - \bar{\alpha}_{t-1}\right)}$$
$$\begin{aligned}
    &= \frac{\left(1 - \alpha_t\right)\left(1 - \bar{\alpha}_{t-1}\right)}{1 - \bar{\alpha}_{t-1}} \left( \frac{x_t^2}{1 - \alpha_t} + \frac{\bar{\alpha}_{t-1} x_0^2}{1 - \bar{\alpha}_{t-1}} - \frac{x_t^2 - 2 \sqrt{\alpha}_t x_t x_0 + \alpha_t x_0^2}{1 - \bar{\alpha}_{t}} \right) \\[15pt]
    &= x_t^2 \left(\frac{1 - \bar{\alpha}_{t-1}}{1 - \bar{\alpha}_t} - \frac{\left(1 - \alpha_t\right) \left(1 - \bar{\alpha}_{t-1}\right)}{\left(1 - \bar{\alpha}_t\right)^2}\right) \\[5pt]
    & \hspace{4cm}+ x_0^2 \left( \frac{\left(1 - \alpha_t\right)\left(\bar{\alpha}_{t-1}\right)}{1 - \bar{\alpha}_t} - \frac{\left(1 - \alpha_t\right)\left(1 - \bar{\alpha}_{t-1}\right)\bar{\alpha}_t}{\left(1 - \bar{\alpha}_t\right)^2} \right) \notag \\[5pt]
    & \hspace{7cm} + \frac{2 x_t x_0 \sqrt{\bar{\alpha}_t}\left(1 - \alpha_t\right)\left(1 - \bar{\alpha}_{t-1}\right)}{\left(1 - \bar{\alpha}_{t}\right)^2} \notag \\[15pt]
    &= x_t^2 \left(\frac{ \left( 1 - \bar{\alpha}_t \right) \left( 1 - \bar{\alpha}_{t-1} \right) - \left(1 - \alpha_t\right) \left(1 - \bar{\alpha}_{t-1}\right)}{\left(1 - \bar{\alpha}_t\right)^2}\right) \\[5pt]
    & \hspace{4cm} + x_0^2 \left( \frac{ \left( 1 - \bar{\alpha}_t \right) \left( 1 - \alpha_t \right) \bar{\alpha}_{t-1} - \left(1 - \alpha_t\right)\left(1 - \bar{\alpha}_{t-1}\right)\bar{\alpha}_t}{\left(1 - \bar{\alpha}_t\right)^2} \right) \notag \\[5pt]
    & \hspace{7cm} + \frac{2 x_t x_0 \sqrt{\bar{\alpha}_t}\left(1 - \alpha_t\right)\left(1 - \bar{\alpha}_{t-1}\right)}{\left(1 - \bar{\alpha}_{t}\right)^2} \notag \\[15pt]
    &= x_t^2 \frac{ \left( 1 - \bar{\alpha}_{t-1} \right)^2 \alpha_t }{\left(1 - \bar{\alpha}_t\right)^2} + x_0^2 \frac{\left( 1 - \bar{\alpha}_{t} \right)^2 \bar{\alpha}_{t-1}}{\left(1 - \bar{\alpha}_t\right)^2} + \frac{2 x_t x_0 \sqrt{\bar{\alpha}_t}\left(1 - \alpha_t\right)\left(1 - \bar{\alpha}_{t-1}\right)}{\left(1 - \bar{\alpha}_{t}\right)^2} \\[15pt]
    &= \frac{\left( x_t \sqrt{\alpha_t} \left( 1 - \bar{\alpha}_{t-1} \right) + x_0 \sqrt{\bar{\alpha}_{t-1}} \left( 1 - \bar{\alpha}_t \right) \right)^2}{\left(1 - \bar{\alpha}_{t}\right)^2} \\[15pt]
    &= \left(\frac{\left( x_t \sqrt{\alpha_t} \left( 1 - \bar{\alpha}_{t-1} \right) + x_0 \sqrt{\bar{\alpha}_{t-1}} \left( 1 - \bar{\alpha}_t \right) \right)}{\left(1 - \bar{\alpha}_{t}\right)} \right)^2
\end{aligned}$$ We can now replace this value into the original
expression: $$\begin{aligned}
  q(x_{t-1} | x_t, x_0) &\propto \exp \left(- \frac{1}{2} \left( \left(\frac{1}{\frac{\left(1 - \alpha_t\right)\left(1 - \bar{\alpha}_{t-1}\right)}{1 - \bar{\alpha}_{t-1}}}\right) \right. \right. \\[5pt]
  & \hspace{2cm} \left(x_{t-1}^2 -2 \frac{\sqrt{\alpha_t} \left(1 - \bar{\alpha}_{t-1}\right) x_t + \sqrt{\bar{\alpha}_{t-1}}\left(1 - \alpha_t\right) x_0}{1 - \bar{\alpha}_t} \right. \notag \\[5pt]
  & \hspace{3cm} \left. \left. \left. + \left(\frac{\left( x_t \sqrt{\alpha_t} \left( 1 - \bar{\alpha}_{t-1} \right) + x_0 \sqrt{\bar{\alpha}_{t-1}} \left( 1 - \bar{\alpha}_t \right) \right)}{\left(1 - \bar{\alpha}_{t}\right)} \right)^2 \right) \right) \right) \\[15pt]
  &= \exp \left(- \frac{1}{2} \left(\frac{1}{\frac{\left(1 - \alpha_t\right)\left(1 - \bar{\alpha}_{t-1}\right)}{1 - \bar{\alpha}_{t-1}}}\right) \right. \\
  & \hspace{3cm} \left. \left( x_{t-1} - \frac{\sqrt{\alpha_t}(1 - \bar{\alpha}_{t-1}) x_t + \sqrt{\bar{\alpha}_{t-1}}(1 - \alpha_t) x_0}{1 - \bar{\alpha}_t} \right)^2  \right) \notag
\end{aligned}$$ which through normalisation gives: $$\begin{aligned}
  q(x_{t-1} | x_t, x_0) &= \mathcal{N} \left(x_{t-1}; \frac{\sqrt{\alpha_t}(1 - \bar{\alpha}_{t-1}) x_t + \sqrt{\bar{\alpha}_{t-1}}(1 - \alpha_t) x_0}{1 - \bar{\alpha}_t} , \frac{\left(1 - \alpha_t\right)\left(1 - \bar{\alpha}_{t-1}\right)}{1 - \bar{\alpha}_{t}} I \right) \\[15pt]
  &= \mathcal{N} \left(x_{t-1}; \tilde{\mu}(x_t, x_0), \tilde{\beta}_t I \right)
\end{aligned}$$ \
We have therefore obtained $\tilde{\mu}(x_t, x_0)$ and $\tilde{\beta}_t$
mentioned in Ho et al.'s [@ho2020denoising] paper.\
The authors also rearrange $\tilde{\mu}(x_t, x_0)$ in order to remove
the dependency on $x_0$:\
$$\begin{aligned}
    x_t &= \sqrt{\bar{\alpha}_t} x_0 + \sqrt{1 - \bar{\alpha}_t} \epsilon \\
    \Leftrightarrow x_0 &= \frac{x_t - \sqrt{1 - \bar{\alpha}_t} \epsilon}{\sqrt{\bar{\alpha}_t}} \label{eq:x0fromxt} \\
    \Rightarrow \tilde{\mu}(x_t, t) &= \frac{\sqrt{\alpha_t}(1 - \bar{\alpha}_{t-1}) x_t + \sqrt{\bar{\alpha}_{t-1}}(1 - \alpha_t) \frac{x_t - \sqrt{1 - \bar{\alpha}_t} \epsilon}{\sqrt{\bar{\alpha}_t}}}{1 - \bar{\alpha}_t} \\
    &= \frac{\sqrt{\alpha_t}(1 - \bar{\alpha}_{t-1}) x_t + (1 - \alpha_t) \left(x_t - \sqrt{1 - \bar{\alpha}_t} \epsilon \right) \frac{\sqrt{\bar{\alpha}_{t-1}}}{\sqrt{\bar{\alpha}_t}}}{1 - \bar{\alpha}_t} \\
    &= \frac{\sqrt{\alpha_t}(1 - \bar{\alpha}_{t-1}) x_t}{{1 - \bar{\alpha}_t}} + \frac{(1 - \alpha_t) \left(x_t - \sqrt{1 - \bar{\alpha}_t} \epsilon \right) \frac{1}{\sqrt{\alpha_t}}}{1 - \bar{\alpha}_t} \\
    &= \frac{\sqrt{\alpha_t}(1 - \bar{\alpha}_{t-1})}{{1 - \bar{\alpha}_t}} x_t + \frac{(1 - \alpha_t)}{(1 - \bar{\alpha}_t) \sqrt{\alpha_t}} x_t - \frac{(1 - \alpha_t)(\sqrt{1 - \bar{\alpha}_t})}{(1 - \bar{\alpha}_t)\sqrt{\alpha_t}} \epsilon \\
    &= \left( \frac{\sqrt{\alpha_t} \sqrt{\alpha_t}(1 - \bar{\alpha}_{t-1})}{\sqrt{\alpha_t} ({1 - \bar{\alpha}_t})}+ \frac{(1 - \alpha_t)}{(1 - \bar{\alpha}_t) \sqrt{\alpha_t}} \right) x_t - \frac{(1 - \alpha_t)(\sqrt{1 - \bar{\alpha}_t})}{\sqrt{(1 - \bar{\alpha}_t)}\sqrt{(1 - \bar{\alpha}_t)}\sqrt{\alpha_t}} \epsilon \\
    &= \left( \frac{\alpha_t (1 - \bar{\alpha}_{t-1}) + (1 - \alpha_t)}{\sqrt{\alpha_t} ({1 - \bar{\alpha}_t})} \right) x_t - \frac{(1 - \alpha_t)}{\sqrt{(1 - \bar{\alpha}_t)}\sqrt{\alpha_t}} \epsilon \\
    &= \left( \frac{\alpha_t - \bar{\alpha}_{t} + 1 - \alpha_t}{\sqrt{\alpha_t} ({1 - \bar{\alpha}_t})} \right) x_t - \frac{1 - \alpha_t}{\sqrt{(1 - \bar{\alpha}_t)}\sqrt{\alpha_t}} \epsilon \\
    &= \frac{1}{\sqrt{\alpha_t}} x_t - \frac{1 - \alpha_t}{\sqrt{(1 - \bar{\alpha}_t)}\sqrt{\alpha_t}} \epsilon \\
    &= \frac{1}{\sqrt{\alpha_t}} \left( x_t - \frac{1 - \alpha_t}{\sqrt{(1 - \bar{\alpha}_t)}} \epsilon \right)
  
\end{aligned}$$

Another important point is that the term $L_T$
([\[eq:lT\]](#eq:lT){reference-type="ref" reference="eq:lT"}) can be
omitted since $q$ has no learnable parameters and with the fact that it
will almost become a Gaussian distribution, the Kullback-Leibler
Divergence\
$D_{KL}\left(q\left(x_T | x_0\right) \| \, p\left(x_T\right)\right)$
between a nearly Gaussian distribution and a Gaussian distribution
$p\left(x_T\right) = \mathcal{N}\left(x_t; 0, I\right)$ will be close to
0 [@ho2020denoising].\
We thus find ourselves with: $$\begin{gathered}
  L_{VLB} = \mathbb{E}_q \left[ \sum_{t \geq 2} D_{KL}\left(q\left(x_{t-1} | x_t, x_0\right) \| \, p_{\theta}\left(x_{t-1} | x_t\right)\right) - \log\left(p_{\theta}\left(x_0 | x_1\right)\right) \right]
\end{gathered}$$

### Simplified training objective

Ho et al. [@ho2020denoising] simplify the training objective by making
several changes in order to obtain an easy quantity for minimising the
loss.

First of all, $L_{1: T-1}$ can be written as such: $$\begin{aligned}
  L_{1:T-1} &= D_{KL}\left(q\left(x_{t-1} | x_t, x_0\right) \| \: p_{\theta}\left(x_{t-1} | x_t\right)\right) \\
  &= D_{KL}\left(\mathcal{N} (x_{t-1}; \tilde{\mu}_t \left(x_t, t \right), \sigma^2_t I ) \| \: \mathcal{N} \left( x_{t-1} ; \mu_{\theta}(x_t, t), \sigma^2_t I  \right) \right)
\end{aligned}$$ The Kullback-Leibler divergence between the two Gaussian
distributions of dimension $D$ is given by: $$\begin{aligned}
    & D_{KL}\left(\mathcal{N} (x_{t-1}; \tilde{\mu}_t \left(x_t, t \right), \sigma^2_t I ) \| \: \mathcal{N} \left( x_{t-1} ; \mu_{\theta}(x_t, t), \sigma^2_t I  \right) \right) \\
    &= \frac{1}{2} \left( tr\left( (\sigma^2_t I)^{-1} (\sigma^2_t I) \right) - D + \left( \mu_{\theta} - \tilde{\mu}_t \right)^T (\sigma^2_t I)^{-1} \left( \mu_{\theta} - \tilde{\mu}_t \right) + \log \frac{\det(\sigma^2_t I)}{\det(\sigma^2_t I)} \right) \\
    &= \frac{1}{2} \left( D - D + \left( \mu_{\theta} - \tilde{\mu}_t \right)^T (\sigma^2_t I)^{-1} \left( \mu_{\theta} - \tilde{\mu}_t \right) + \log 1 \right) \\
    &= \frac{1}{2} \left( \left( \mu_{\theta} - \tilde{\mu}_t \right)^T (\sigma^2_t I)^{-1} \left( \mu_{\theta} - \tilde{\mu}_t \right) \right) \\
    &= \frac{1}{2 \sigma^2_t} \left( \left( \mu_{\theta} - \tilde{\mu}_t \right)^T \left( \mu_{\theta} - \tilde{\mu}_t \right) \right) \\
    &= \frac{1}{2 \sigma^2_t} \| \mu_{\theta} - \tilde{\mu}_t \|_2^2
  
\end{aligned}$$\
However, Ho et al. [@ho2020denoising] rewrite this formula so that it
depends only on the noise $\epsilon$: $$\begin{aligned}
    \frac{1}{2 \sigma^2_t} \| \mu_{\theta} - \tilde{\mu}_t \|_2^2 &= \frac{1}{2 \sigma^2_t} \left\| \frac{1}{\sqrt{\alpha_t}} \left( x_t - \frac{1 - \alpha_t}{\sqrt{(1 - \bar{\alpha}_t)}} \epsilon \right) - \frac{1}{\sqrt{\alpha_t}} \left( x_t - \frac{1 - \alpha_t}{\sqrt{(1 - \bar{\alpha}_t)}} \epsilon_\theta \right)\right\|_2^2 \\
    &= \frac{1}{2 \sigma^2_t} \left\| \frac{1}{\sqrt{\alpha_t}} \frac{1 - \alpha_t}{\sqrt{(1 - \bar{\alpha}_t)}} (\epsilon - \epsilon_\theta) \right\|_2^2 \\
    &= \frac{1}{2 \sigma^2_t} \frac{1}{\alpha_t} \frac{(1 - \alpha_t)^2}{1 - \bar{\alpha}_t} \| \epsilon - \epsilon_\theta \|_2^2 \\
    &= \frac{\beta_t^2}{2 \sigma^2_t \alpha_t (1 - \bar{\alpha}_t)} \| \epsilon - \epsilon_\theta \|_2^2
  
\end{aligned}$$ Furthermore, they found that simply ignoring the factor
in front gave better results, giving us:
$$L_{1:T-1} = \| \epsilon - \epsilon_\theta \|_2^2$$\
As for the second term $L_0$, the authors define $p_\theta(x_0 | x_1)$
as: $$\begin{aligned}
  p_\theta(x_0 | x_1) &= \prod_{i=1}^{D} \int_{\delta_{-} (x_0^i)}^{\delta_{+} (x_0^i)} \mathcal{N}(x; \mu_\theta^i (x_1, 1), \sigma^2_1) \: dx
\end{aligned}$$ where $\delta_{+} (x)$ and $\delta_{-} (x)$ are defined
as: $$\begin{aligned}
  \delta_{+} (x) &= \begin{cases}
    \infty & \text{if } x = 1 \\
    x + \frac{1}{255} & \text{if } x < 1 
  \end{cases}\\
  \delta_{-} (x) &= \begin{cases}
    - \infty & \text{if } x = - 1 \\
    x - \frac{1}{255} & \text{if } x > -1 
  \end{cases}
\end{aligned}$$ with $D$ the data dimensionality and $i$ one coordinate
from the image.

This is a decoder for the last step in order to get pixel values between
$\{0, 1, \dots, 255 \}$ from the values in $[-1, 1]$ used for the neural
network [@ho2020denoising; @nichol2021improved]\
However the authors also ignore the entire term, leaving us with:
$$L_{\text{simple}}(\theta) = \mathbb{E}_q \left[ \| \epsilon - \epsilon_\theta \|_2^2 \right] \label{eq:l_simple}$$

### Model architecture {#subsubsec:model_architecture}

Ho et al. [@ho2020denoising] as well as the following
papers [@dhariwal2021diffusion; @nichol2021improved; @song2022denoising]
use a U-Net style architecture for the neural network to estimate the
noise.

The U-Net architecture is a convolutional neural network (CNN) that was
originally designed for image segmentation in biomedical
imaging [@ronneberger2015unet]. It consists of an encoder (or
contracting path) and a decoder (or expansive path) that are connected
by a bottleneck.\
The encoder takes in the noisy image as input and repeatedly applies
unpadded convolutions followed by max pooling layers in order to reduce
the spatial dimensions of the image [@ronneberger2015unet].\
The bottleneck contains 3 unpadded convolutions each followed by a
rectified linear unit (ReLU) activation
function [@ronneberger2015unet].\
The decoder then applies up-convolutions followed by unpadded
convolutions to increase the spatial dimensions of the image, with at
each step a concatenation of the feature maps from the
encoder [@ronneberger2015unet].

![Original U-Net
architecture [@ronneberger2015unet]](images/unet.png){width="\\textwidth"}

However, Ho et al. [@ho2020denoising] made some changes to the original
U-Net architecture.\
First, they state that they used 4 feature map resolutions for their
$32 \times 32$ resolution model, and 6 for the $256 \times 256$
resolution model. For each of these feature maps, they used two
convolutional residual blocks similar of those of a Wide
ResNet [@zagoruyko2017wide] consisting of a convolutional layer, a ReLU
activation function, a convolutional layer, a group normalisation layer,
and another ReLU activation function.\
The "residual" part of the block comes from the fact that the result at
the end of the block on the encoder side is concatenated with the result
at the end of the block on the decoder side in order to consider
features that could have been lost during the
downsampling [@lai2022rethinking].\
They used group normalisation [@wu2018group] rather than weight
normalisation [@salimans2016weight], which is a normalisation technique
that divides the channels into groups and computes the mean and standard
deviation for each group [@ho2020denoising].\
Finally, they state that they used self-attention layers from the
Transformer architecture [@vaswani2023attention] at the $16 \times 16$
resolution between the two convolutional residual
blocks [@ho2020denoising].

Since the model needs to be able to predict the noise at any timestep
$t$, the authors added a time embedding to the input of the network
using the sinusoidal positional embedding from the Transformer
architecture [@vaswani2023attention], which allows the parameters to be
shared across all timesteps.\
The sinusoidal positional embedding is given by [@vaswani2023attention]:
$$\begin{aligned}
  PE_{(pos, 2i)} &= \sin \left( \frac{pos}{10000^{2i/d_{model}}} \right) \\
  PE_{(pos, 2i+1)} &= \cos \left( \frac{pos}{10000^{2i/d_{model}}} \right) \\
  \text{where } \qquad & pos \text{ is the position and } i \text{ is the dimension} \notag
\end{aligned}$$ Ho et al. [@ho2020denoising] present two algorithms: one
for training the model and one for sampling from the model:\

:::: algorithm
::: algorithmic
$x_0 \sim q(x_0)$ $t \sim$ Uniform$(\{1,\ldots, T\})$
$\epsilon \sim \mathcal{N}(0, I)$ Take gradient descent step on:
$\nabla_\theta \| \epsilon - \epsilon_\theta \left( \sqrt{\bar{\alpha}_t} x_0 + \sqrt{1 - \bar{\alpha}_t} \epsilon, t \right) \|^2$
converged
:::
::::

:::: algorithm
::: algorithmic
$x_T \sim \mathcal{N}(0, I)$ $z \sim \mathcal{N}(0, I)$ if $t > 0$, else
$z = 0$
$x_{t-1} = \frac{1}{\sqrt{\alpha_t}} \left( x_t - \frac{1 - \alpha_t}{\sqrt{1 - \bar{\alpha_t}}} \epsilon_\theta (x_t, t) \right) + \sigma_t z$
**return** $x_0$
:::
::::