# Tutorial on Variational Autoencoders (VAEs)

## Abstract
- Unsupervised learning for complicated distributions
- Trained through stochastic gradient descent
- Can generate images
- Possibly predict the future of the image[link](https://arxiv.org/abs/1606.07873)

## Introduction

- Generative models attempt to "capture the dependencies between pixels" in images
- Goal: Synthesize images that look "real" with high probability
- Nice example: Generate more handwritten text from an example

Issues:
  1. Strong assumptions about structure of the data
  2. Might need harsh approximations
  3. Computationally expensive

- VAE assumptions are weak {why?} and train fast through backprop
- VAEs make approximations {paper says it isn't a big deal. Is that true?}

- Paper came from a presentation (not linked) [Related Lecture](https://www.youtube.com/watch?v=P78QYjWh5sM)

### Latent Variable Models

- The more complicated the relationship between dependencies, the more difficult to train {seems obvious}
- Model should first decide what type of image to construct before it starts generating pixels
- Call this the **latent variable**
- We don't necessarily know this variable
- For every pixel $x$ in the space $X$ there should be a setting of the latent variable $z$ (from the space $Z$) generate it
- Let's say there is some deterministic function with parameters $\theta$:

$$f : Z \times \Theta \rightarrow X (1)$$

because $z$ is random, $f(z; \theta)$ is also random.

Goal: optimize the fixed $\theta$s so that the function generates $X$s like the ones seen in the dataset.

$$P(X) = \int P(X|z; \theta) P(z) dz (2)$$
- $P(X|z;\theta)$ is Gaussian
- The variance term $\sigma^2$ is a hyperparameter {He just uses the identity matrix later}
- Use gradient descent to optimize P(X) ($P(X|z)$ must be continuous)

{Side note: Says that we can't use the Dirac delta function. Why would we want that?}

## Variational Autoencoders

-VAEs approximately maximize equation 2
- Autoencoders encode and decode
- No tuning parameters {what about $\sigma$ is that learned?}

Two problems with equation 2:
1. How are the latent variables defined
2. How do we solve the integral?

#### Defining Latent Variables
- Let the encoder decide
- *VAEs just assume that the samples can be drawn from a simple distribution (Gaussian) and mapped through a function*
- This works because we have many variables to choose from and Neural networks are universal approximators
- {Follow up in Devroye et al. http://www.eirene.de/Devroye.pdf}

#### Maximizing the Equation
- {This is the same as solving the integral, yes? I think this is the variational method}
- Take the gradient of Equation 2 (Stochastic Gradient Descent time)

Compute:

$$P(X) \approx \frac{1}{n} \sum_i P(X|Z_i) (3)$$
- Lots of pixels are a problem
- Euclidean distance on a 2d image means the log prob of P(X) is proportional to square of the distance between f(z) and X
- Practical example: the exact same image as the image, shift down one pixel, may have a higher distance than one missing a key feature (a 2 without a loop)
- we need a better objective function!

### Setting Up the Objective

- for most z, P(X|z) will be nearly 0
- we need to sample zs that are likely to produce Xs
New function:

$$Q(z|X)$$

If the space of z values is small it'll be easier to compute the expected value of $E[X|z]$ {because of sparsity?}
Q(z) is not necessarily Gaussian.
E[X|z] need to be related first P(X)

Definition of Kullback-Leiber Divergence (The "distance" between two distributions)

$$D[Q[z] || P(z|X)] = E_{z~Q}[\log Q(z) - \log P(z|X)] (4)$$

After applying Bayes and rearranging and making Q(z) dependent on X:

$$\log P(X) - D[Q(z|X)||P(z|X)] = E_{z~Q}[\log P(X|z)] - D[Q(z|X)||P(z)] (5)$$

- We want to maximize $P(X)$, and minimize the DL term.. 
- The right side can be optimized through SGD
In the autoencoder framework: Q encodes X into z. P decodes to reconstruct X
- A bit of math magic if happening here. With an arbitrarily high-capacity model (neural net), Q(z|X) approaches P(z|X)
- We can calculate Q(z|X), not P(z|X)
- {Try solving this by hand}

### Optimizing the Objective

Usual choice for Q(z|X) {You get to choose?}

$$Q(z|X) = N(z|\mu(X;\Theta), \Sigma(X;\Theta))$$

{That doesn't look like a theta!}
- Where $\mu$ and $\Sigma$ and deterministic functions learned from the data
- This is what the neural network approximates
- $\Sigma$ must be diagonal {Probably cleans up the math}

Recall that P(z) is N(0, I) (An assumption):

The papers tells us the that the KL-divergence between the two multivariate Gaussians is:

$$D[N(\mu(X), \Sigma(X))||N(0,I] = \frac{1}{2}(tr(\Sigma(X)) + (\mu(X))^T(\mu(X)) - k - \logdet(\Sigma(X)))$$

Note: $\tr$ gives the sum of all diagonals in a matrix 


