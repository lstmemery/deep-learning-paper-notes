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

$$f : Z \times \Theta \rightarrow X$$

because $z$ is random, $f(z; \theta)$ is also random.

Goal: optimize the fixed $\theta$s so that the function generates $X$s like the ones seen in the dataset.

$$P(X) = \int P(X|z; \theta) P(z) dz$$
