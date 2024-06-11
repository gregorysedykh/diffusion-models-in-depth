# Appendix A: Convolution of two Gaussian distributions

Equation () had given us: 
\begin{align}
  x_t &= \sqrt{\alpha_t \alpha_{t-1}} x_{t-2} + \mathcal{N}\left(0, \alpha_t\left(1 - \alpha_{t-1}\right)I \right) + \mathcal{N}\left(0, (1 - \alpha_t) I \right)
\end{align}
We can find the distribution of the sum of two
independant 1D random variables
$X \sim \mathcal{N}\left(\mu_X, \sigma_X^2\right)$ and
$Y \sim \mathcal{N}\left(\mu_Y, \sigma_Y^2\right)$ as the convolution of
two Gaussian distributions.

Let $Z = X + Y$ with $X$ and $Y$ independent. 
\begin{align}
  f_Z(z) &= \int_{-\infty}^{\infty} f_X(x) f_Y(z - x) \: dx \\
  &= \int_{-\infty}^{\infty} \frac{1}{\sqrt{2 \pi \sigma_X^2}} \exp \left[{-\frac{(x - \mu_X)^2}{2 \sigma_X^2}} \right] \frac{1}{\sqrt{2 \pi \sigma_Y^2}} \exp \left[{-\frac{(z - x - \mu_Y)^2}{2 \sigma_Y^2}} \right] \: dx \\[10pt]
  &= \frac{1}{2 \pi \sigma_X \sigma_Y} \int_{-\infty}^{\infty} \exp \left[{-\frac{(x - \mu_X)^2}{2 \sigma_X^2}} \right] \exp \left[{-\frac{(z - x - \mu_Y)^2}{2 \sigma_Y^2}} \right] \: dx \\[10pt]
  &= \frac{1}{2 \pi \sigma_X \sigma_Y} \int_{-\infty}^{\infty} \exp\left[{-\frac{(x - \mu_X)^2}{2 \sigma_X^2} - \frac{(z - x - \mu_Y)^2}{2 \sigma_Y^2}} \right] \: dx \\[10pt]
  &= \frac{1}{2 \pi \sigma_X \sigma_Y} \int_{-\infty}^{\infty} \exp\left[{-\frac{x^2 - 2 \mu_X x + \mu_X^2}{2 \sigma_X^2} - \frac{z^2 + x^2 - 2 x z - 2 z \mu_Y + 2 \mu_Y x + \mu_Y^2}{2 \sigma_Y^2}} \right] \: dx \\[10pt]
  &= \frac{1}{2 \pi \sigma_X \sigma_Y} \int_{-\infty}^{\infty} \exp\left[{-\frac{\sigma_Y^2(x^2 - 2 \mu_X x + \mu_X^2)}{2 \sigma_X^2 \sigma_Y^2} - \frac{\sigma_X^2(z^2 + x^2 - 2 x z - 2 z \mu_Y + 2 \mu_Y x + \mu_Y^2)}{2 \sigma_X^2 \sigma_Y^2 }} \right] \: dx \\[10pt]
  &= \frac{1}{2 \pi \sigma_X \sigma_Y} \int_{-\infty}^{\infty} \exp\left[{-\frac{\sigma_Y^2 x^2 - 2 \mu_X \sigma_Y^2 x + \mu_X^2 \sigma_Y^2 + \sigma_X^2 z^2 + \sigma_X^2 x^2 - 2xz\sigma_X^2 + 2 z \mu_Y \sigma_X^2 - 2\mu_Y x \sigma_X^2 + \sigma_X^2 \mu_Y^2}{2 \sigma_X^2 \sigma_Y^2}} \right] \: dx \\[10pt]
  &= \frac{1}{2 \pi \sigma_X \sigma_Y} \int_{-\infty}^{\infty} \exp\left[{-\frac{\left(\sigma_X^2 + \sigma_Y^2\right) x^2 - 2 x \left( \mu_X \sigma_Y^2 + z \sigma_X^2 - \mu_Y \sigma_X^2 \right) + \sigma_X^2 \left( z - \mu_Y \right)^2 + \mu_X^2 \sigma_Y^2}{2 \sigma_X^2 \sigma_Y^2}} \right] \: dx \\[10pt]
  &\text{Let $\sigma_Z = \sqrt{\sigma_X^2 + \sigma_Y^2}$}. \\
  &= \frac{1}{\sqrt{2 \pi} \frac{\sigma_X \sigma_Y}{\sigma_Z}} \frac{1}{\sqrt{2 \pi} \sigma_Z} \int_{-\infty}^{\infty} \exp\left[{-\frac{x^2 - 2x \frac{\sigma_X^2 (z - \mu_Y) + \sigma_Y^2 \mu_X}{\sigma_Z^2} + \frac{\sigma_X^2 \left( z - \mu_Y \right)^2 + \mu_X^2 \sigma_Y^2}{\sigma_Z^2}}{2 \left(\frac{\sigma_X \sigma_Y}{\sigma_Z}\right)^2}} \right] \: dx \\[10pt]
  &= \frac{1}{\sqrt{2 \pi} \frac{\sigma_X \sigma_Y}{\sigma_Z}} \frac{1}{\sqrt{2 \pi} \sigma_Z} \int_{-\infty}^{\infty} \exp\left[{-\frac{\left( x - \frac{\sigma_X^2 (z - \mu_Y) + \sigma_Y^2 \mu_X}{\sigma_Z^2}\right)^2 - \left( \frac{\sigma_X^2 (z - \mu_Y) + \sigma_Y^2 \mu_X}{\sigma_Z^2} \right)^2 + \frac{\sigma_X^2 (z - \mu_Y)^2 + \sigma_Y^2 \mu_X^2}{\sigma_Z^2}}{2 \left(\frac{\sigma_X \sigma_Y}{\sigma_Z}\right)^2}} \right] \: dx  \\[10pt]
  &= \int_{-\infty}^{\infty} \frac{1}{\sqrt{2 \pi} \frac{\sigma_X \sigma_Y}{\sigma_Z}} \exp\left[{-\frac{\left( x - \frac{\sigma_X^2 (z - \mu_Y) + \sigma_Y^2 \mu_X}{\sigma_Z^2}\right)^2}{2 \left(\frac{\sigma_X \sigma_Y}{\sigma_Z}\right)^2}} \right] \\[10pt]
  & \hspace{3cm} \frac{1}{\sqrt{2 \pi} \sigma_Z} \exp \left[ - \frac{\sigma_Z^2 \left( \sigma_X^2 (z - \mu_Y)^2 + \sigma_Y^2 \mu_X^2 \right) - \left( \sigma_X^2 (z - \mu_Y) + \sigma_Y^2 \mu_X \right)^2}{2 \sigma_Z^2 (\sigma_X \sigma_Y)^2} \right] \: dx  \\[10pt]
  &= \int_{-\infty}^{\infty} \frac{1}{\sqrt{2 \pi} \frac{\sigma_X \sigma_Y}{\sigma_Z}} \exp\left[{-\frac{\left( x - \frac{\sigma_X^2 (z - \mu_Y) + \sigma_Y^2 \mu_X}{\sigma_Z^2}\right)^2}{2 \left(\frac{\sigma_X \sigma_Y}{\sigma_Z}\right)^2}} \right] \frac{1}{\sqrt{2 \pi} \sigma_Z} \exp \left[ - \frac{(z - (\mu_X + \mu_Y))^2}{2 \sigma_Z^2} \right] \: dx  \\[10pt]
  &= \frac{1}{\sqrt{2 \pi} \sigma_Z} \exp \left[ - \frac{(z - (\mu_X + \mu_Y))^2}{2 \sigma_Z^2} \right] \int_{-\infty}^{\infty} \frac{1}{\sqrt{2 \pi} \frac{\sigma_X \sigma_Y}{\sigma_Z}} \exp\left[{-\frac{\left( x - \frac{\sigma_X^2 (z - \mu_Y) + \sigma_Y^2 \mu_X}{\sigma_Z^2}\right)^2}{2 \left(\frac{\sigma_X \sigma_Y}{\sigma_Z}\right)^2}} \right] \: dx  \\[10pt]
  &= \frac{1}{\sqrt{2 \pi} \sigma_Z} \exp \left[ - \frac{(z - (\mu_X + \mu_Y))^2}{2 \sigma_Z^2} \right] \\[10pt]
  &= \mathcal{N}\left(z; \mu_X + \mu_Y, \sigma_X^2 + \sigma_Y^2 \right)
\end{align}

Given this, we can refer back to our case and therefore we find:
\begin{align}
  x_t &= \sqrt{\alpha_t \alpha_{t-1}} x_{t-2} + \mathcal{N}\left(0, \alpha_t\left(1 - \alpha_{t-1}\right)I \right) + \mathcal{N}\left(0, (1 - \alpha_t) I \right) \\
  &= \sqrt{\alpha_t \alpha_{t-1}} x_{t-2} + \mathcal{N}\left(0, \alpha_t \left((1 - \alpha_{t-1}) + 1 - \alpha_t \right)I \right) 
\end{align}
