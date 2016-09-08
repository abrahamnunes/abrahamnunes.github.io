---
layout: post
title: Computing gradients of reinforcement learning models for optimization
author: Abraham Nunes
published: true
status: publish
draft: false
tags: computational_psychiatry reinforcement_learning
---


<script type="text/x-mathjax-config">
MathJax.Hub.Config({
  tex2jax: {inlineMath: [['$','$'], ['\\(','\\)']]}
});
</script>
<script type="text/javascript"
  src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML">
</script>

## Introduction

When fitting reinforcement learning models to behavioural data the efficiency of optimization algorithms can be dramatically improved by providing the analytical gradients to the optimization function (e.g. `fminunc` if using Matlab, or `scipy.optimize.minimize` if using python, etc.).

## Procedures

Consider the case when one seeks to find the maximum a posteriori estimate (MAP) for a vector of parameters $\boldsymbol\theta$

$$
\hat{\boldsymbol\theta} = \underset{\boldsymbol\theta}{\text{argmax}}\;\;P(\mathcal{A}|\boldsymbol\theta)P(\boldsymbol\theta|\boldsymbol\phi),
$$

where $\mathcal{A}$ is a set of actions taken by a group of subjects, $\boldsymbol\theta$ are the model parameters, and $\boldsymbol\phi$ are the parameters of the prior distribution of $\boldsymbol\theta$. Since the logarithm is a monotonically increasing function, the above equation can be written as

$$
\hat{\boldsymbol\theta} = \underset{\boldsymbol\theta}{\text{argmax}}\;\; \log P(\mathcal{A}|\boldsymbol\theta) + \log P(\boldsymbol\theta|\boldsymbol\phi).
$$

The right hand side of the above equation must be differentiated with respect to the parameters $\boldsymbol\theta$:

$$
\begin{aligned}
\frac{\partial}{\partial \boldsymbol\theta} \log P(\mathcal{A}, \boldsymbol\theta | \boldsymbol\phi)  & = \frac{\partial}{\partial \boldsymbol\theta} \Big[ \log P(\mathcal{A}|\boldsymbol\theta) + \log P(\boldsymbol\theta|\boldsymbol\phi) \Big] \\
& = \frac{\partial}{\partial \boldsymbol\theta}  \log P(\mathcal{A}|\boldsymbol\theta) + \frac{\partial}{\partial \boldsymbol\theta}  \log P(\boldsymbol\theta|\boldsymbol\phi),
\end{aligned}
$$

where the elements of the right hand side can be considered individually.

## The Prior Distribution

As per Huys et al. (2011), we assume that the parameters $\boldsymbol\theta$ are distributed according to a multivariate Gaussian with hyperparameters $\boldsymbol\phi = \lbrace \phi_\mu \phi_\Sigma \rbrace$, where $\phi_\mu$ is the prior mean, and $\phi_\Sigma$ is the prior covariance matrix:

$$
P(\boldsymbol\theta|\boldsymbol\phi) = \frac{1}{(2\pi)^{\frac{K}{2}}|\phi_\Sigma|^{\frac{1}{2}}} \exp \Big\lbrace - \frac{1}{2}(\boldsymbol\theta - \phi_\mu)^\top \phi_\Sigma ^{-1} (\boldsymbol\theta - \phi_\mu) \Big\rbrace,
$$

where $K$ is the dimension of $\boldsymbol\theta$. For example, if the model you are fitting has parameters $\boldsymbol\theta = \lbrace \alpha, \beta, \omega \rbrace$, then $K=3$. Taking the log of the multivariate Gaussian probability density function (pdf) yields

$$
\log P(\boldsymbol\theta|\boldsymbol\phi) = -\frac{K}{2}\log 2\pi - \frac{1}{2} \log |\phi_\Sigma| - \frac{1}{2}(\boldsymbol\theta - \phi_\mu)^\top \phi_\Sigma ^{-1} (\boldsymbol\theta - \phi_\mu).
$$

Taking the derivative, with respect to $\boldsymbol\theta$ proceeds as follows

$$
\begin{aligned}
\frac{\partial}{\partial \boldsymbol\theta} \log P(\boldsymbol\theta|\boldsymbol\phi)  & = \frac{\partial}{\partial \boldsymbol\theta} \Bigg[ -\frac{K}{2}\log 2\pi - \frac{1}{2} \log |\phi_\Sigma| - \frac{1}{2}(\boldsymbol\theta - \phi_\mu)^\top \phi_\Sigma ^{-1} (\boldsymbol\theta - \phi_\mu) \Bigg] \\
& = - \frac{\partial}{\partial \boldsymbol\theta} \frac{1}{2}(\boldsymbol\theta - \phi_\mu)^\top \phi_\Sigma ^{-1} (\boldsymbol\theta - \phi_\mu).
\end{aligned}
$$

I like to break things down to facilitate simpler notation and use a very simple looking chain rule. Let

$$
\begin{aligned}
f & = - \frac{1}{2}(\boldsymbol\theta - \phi_\mu)^\top \phi_\Sigma ^{-1} (\boldsymbol\theta - \phi_\mu), \\
g & = (\boldsymbol\theta - \phi_\mu)^\top \phi_\Sigma ^{-1} (\boldsymbol\theta - \phi_\mu), \text{  and} \\
X & = (\boldsymbol\theta - \phi_\mu),
\end{aligned}
$$

and thus,

$$
\begin{aligned}
f & = - \frac{1}{2}g, \\
g & = X^\top \phi_\Sigma ^{-1} X, \text{  and} \\
X & = (\boldsymbol\theta - \phi_\mu),
\end{aligned}
$$

The chain rule is then easily represented as

$$
\frac{\partial f}{\partial \boldsymbol\theta} = \frac{\partial f}{\partial g} \frac{\partial g}{\partial X} \frac{\partial X}{\partial \boldsymbol\theta},
$$

and we can compute each derivative individually as follows

$$
\begin{aligned}
\frac{\partial f}{\partial g} = - \frac{1}{2}, \\
\frac{\partial g}{\partial X} = 2 \phi_\Sigma^{-1}X \\
\frac{\partial X}{\partial \boldsymbol\theta} = 1.
\end{aligned}
$$

Combining these, we find that the derivative of the prior distribution with respect to the parameters $\boldsymbol\theta$, when distributed according to a multivariate Gaussian, is

$$
\frac{\partial}{\partial \boldsymbol\theta} \log P(\boldsymbol\theta|\boldsymbol\phi) = - \phi_\Sigma^{-1} (\boldsymbol\theta - \phi_\mu).
$$

### The Log-Likelihood Function

The log-likelihood function is perhaps the more interesting derivative to compute because it will vary based on the model and the number of parameters in each model. A RL model consists of two components: an _observation model_ and a _learning model_. The observation model


### Observation Model

The observation model consists of the process by which the agent selects actions. Classically---and herein---we implement the observation model as a softmax function characterizing the probability of action $a_t$ given the current state of the learning model, $\mathcal{Q}_t(s_t, a_t)$ and the model parameters $\boldsymbol\theta$: $P(a_t | \mathcal{Q} _t(s_t, a_t); \boldsymbol\theta)$.

$$
P(a _t | \mathcal{Q} _t(s _t, a_t); \boldsymbol\theta) = \frac{e^{\beta \mathcal{Q} _t(s _t, a _t)}}{\sum _{a'} e^{\beta \mathcal{Q} _t(s _t, a')}}
$$

The probability of the entire series of actions selected by subject $i$ is

$$
P(\lbrace a _t \rbrace _{t = 1}^T | \lbrace \mathcal{Q} _t(s _t, a _t) \rbrace _{t = 1}^T; \boldsymbol\theta).
$$

Assuming that if conditioned by $\mathcal{Q}$, the actions are independent across trials, one can express this as the product of individual observation probabilities at each time $t$

$$
P(\lbrace a _t \rbrace _{t = 1}^T | \lbrace \mathcal{Q} _t(s _t, a _t) \rbrace _{t = 1}^T; \boldsymbol\theta) = \prod _{t = 1}^T P( a _t | \mathcal{Q} _t(s _t, a _t) ; \boldsymbol\theta).
$$

Taking the log of both sides,

$$
\begin{aligned}
\log P(\lbrace a _t \rbrace _{t = 1}^T | \lbrace \mathcal{Q} _t(s _t, a _t) \rbrace _{t = 1}^T; \boldsymbol\theta) & = \sum _{t = 1}^T \log P( a _t | \mathcal{Q} _t(s _t, a _t) ; \boldsymbol\theta) \\\\
& = \sum _{t = 1}^T \Big[ \beta \mathcal{Q} _t(s _t, a _t; \boldsymbol\theta _{\mathcal{Q}}) - \log \sum _{a'} e^{\beta \mathcal{Q} _t(s _t, a'; \boldsymbol\theta _{\mathcal{Q}})} \Big]
\end{aligned}
$$

we have the log-likelihood of the participants' actions, which we would like to maximize with respect to the parameters $\boldsymbol\theta$:

$$
\begin{aligned}
\frac{\partial}{\partial \boldsymbol\theta} \log P(\lbrace a _t \rbrace _{t = 1}^T | \lbrace \mathcal{Q} _t(s _t, a _t) \rbrace _{t = 1}^T; \boldsymbol\theta) & = \frac{\partial}{\partial \boldsymbol\theta} \sum _{t = 1}^T \log P( a _t | \mathcal{Q} _t(s _t, a _t) ; \boldsymbol\theta) \\\\
& = \sum _{t = 1}^T \frac{\partial}{\partial \boldsymbol\theta} \log P( a _t | \mathcal{Q} _t(s _t, a _t) ; \boldsymbol\theta) \\\\
& = \sum _{t = 1}^T \frac{\partial}{\partial  \boldsymbol\theta} \Big[ \beta \mathcal{Q} _t(s _t, a _t;\boldsymbol\theta _{\mathcal{Q}}) - \log \sum _{a'} e^{\beta \mathcal{Q} _t(s _t, a'; \boldsymbol\theta _{\mathcal{Q}})} \Big] \\\\
& = \sum _{t = 1}^T \Big[ \frac{\partial}{\partial \boldsymbol\theta} \beta \mathcal{Q} _t(s _t, a _t; \boldsymbol\theta _{\mathcal{Q}}) - \frac{\partial}{\partial \boldsymbol\theta} \log \sum _{a'} e^{\beta \mathcal{Q} _t(s _t, a'; \boldsymbol\theta _{\mathcal{Q}})} \Big] \\\\
& = \sum _{t = 1}^T \Big[ \frac{\partial}{\partial \boldsymbol\theta} \beta \mathcal{Q} _t(s _t, a _t; \boldsymbol\theta _{\mathcal{Q}}) - \frac{e^{\beta \mathcal{Q} _t(s _t, \lbrace a' \rbrace _{k = 1}^K; \boldsymbol\theta _{\mathcal{Q}})}}{\sum _{a'} e^{\beta \mathcal{Q} _t(s _t, a'; \boldsymbol\theta _{\mathcal{Q}})}} \frac{\partial}{\partial \boldsymbol\theta} \beta \mathcal{Q} _t(s _t, \lbrace a' \rbrace _{k = 1}^K; \boldsymbol\theta _{\mathcal{Q}}) \Big] \\\\
& = \sum _{t = 1}^T \Big[ \frac{\partial}{\partial \boldsymbol\theta} \beta \mathcal{Q} _t(s _t, a _t; \boldsymbol\theta _{\mathcal{Q}}) - \sum _{a'} P(a' | \mathcal{Q} _t(s _t, a'), \boldsymbol\theta) \frac{\partial}{\partial \boldsymbol\theta} \beta \mathcal{Q} _t(s _t, a'; \boldsymbol\theta _{\mathcal{Q}}) \Big] \\\\
\end{aligned}
$$

### A Rescorla-Wagner Learning Rule

Consider the following learning rule

$$
\mathcal{Q}_{t}(s_t, a_t; \alpha) = (1-\alpha) \mathcal{Q}_{t-1}(s_t, a_t; \alpha) + \alpha \Big(r_t - \mathcal{Q}_{t-1}(s_t, a_t; \alpha)\Big).
$$

We will be computing the derivatives of our RL model using this Rescorla-Wagner learning rule.

### Specific Derivatives

Now that we have specified the base form of the derivative of the observation model and a basic learning model with respect to the parameter vector $\boldsymbol\theta$, we can compute the derivatives with respect to the individual parameters.

For notational simplicity, let

$$
\log P(\lbrace a _t \rbrace _{t = 1}^T | \lbrace \mathcal{Q} _t(s _t, a _t) \rbrace _{t = 1}^T; \boldsymbol\theta) = \log \mathcal{L}(\boldsymbol\theta)
$$

Then the derivative with respect to the learning rate $\alpha$ is

$$
\begin{aligned}
\frac{\partial \log \mathcal{L}(\boldsymbol\theta)}{\partial \alpha} & = \sum _{t = 1}^T \Bigg[ \beta \frac{\partial}{\partial \alpha} \mathcal{Q} _t(s _t, a _t; \boldsymbol\theta _{\mathcal{Q}}) - \sum _{a'} P(a' | \mathcal{Q} _t(s _t, a'), \boldsymbol\theta) \beta \frac{\partial}{\partial \alpha} \mathcal{Q} _t(s _t, a'; \boldsymbol\theta _{\mathcal{Q}}) \Bigg], \\
\end{aligned}
$$

where

$$
\begin{aligned}
\frac{\partial}{\partial \alpha} \mathcal{Q} _t(s _t, a _t; \boldsymbol\theta _{\mathcal{Q}}) & = \frac{\partial}{\partial \alpha} \Bigg[ (1-\alpha)\mathcal{Q}_{t-1}(s_t, a_t; \alpha) + r_t \Bigg] \\
& = \frac{\partial}{\partial \alpha} (1-\alpha)\mathcal{Q}_{t-1}(s_t, a_t; \alpha) + \frac{\partial}{\partial \alpha} r_t, \\
\end{aligned}
$$

and using the product rule,

$$
\begin{aligned}
\frac{\partial}{\partial \alpha} \mathcal{Q} _t(s _t, a _t; \boldsymbol\theta _{\mathcal{Q}}) & = \Bigg[ \Bigg(\frac{\partial}{\partial \alpha} (1-\alpha) \Bigg) + (1-\alpha) \Bigg( \frac{\partial}{\partial \alpha} \mathcal{Q}_{t-1}(s_t, a_t; \alpha) \Bigg) \Bigg] + r_t \\
& = (1-\alpha)\mathcal{Q}_{t-1}(s_t, a_t; \alpha) + \Big(r_t - \mathcal{Q}_{t-1}(s_t, a_t; \alpha) \Big)
\end{aligned}
$$

The derivative with respect to the inverse temperature $\beta$ is

$$
\begin{aligned}
\frac{\partial \log \mathcal{L}(\boldsymbol\theta)}{\partial \beta} & = \sum _{t = 1}^T \Big[ \frac{\partial}{\partial \beta} \beta \mathcal{Q}_t(s _t, a _t; \boldsymbol\theta _{\mathcal{Q}}) - \sum _{a'} P(a' | \mathcal{Q} _t(s _t, a'), \boldsymbol\theta) \frac{\partial}{\partial \beta} \beta \mathcal{Q} _t(s _t, a'; \boldsymbol\theta _{\mathcal{Q}}) \Big] \\
& = \sum _{t = 1}^T \Big[ \mathcal{Q} _t(s _t, a _t; \boldsymbol\theta _{\mathcal{Q}}) - \sum _{a'} P(a' | \mathcal{Q}_t(s _t, a'), \boldsymbol\theta) \mathcal{Q} _t(s _t, a'; \boldsymbol\theta _{\mathcal{Q}}) \Big]. \\
\end{aligned}
$$

### Combining the Derivatives of the Likelihood and Priors

$$
\begin{aligned}
\frac{\partial}{\partial \boldsymbol\theta} \log P(\mathcal{A}, \boldsymbol\theta | \boldsymbol\phi) & = \frac{\partial}{\partial \boldsymbol\theta}  \log P(\mathcal{A}|\boldsymbol\theta) + \frac{\partial}{\partial \boldsymbol\theta}  \log P(\boldsymbol\theta|\boldsymbol\phi) \\
& = \left[ {\begin{array}{c} \frac{\partial }{\partial \alpha} \log \mathcal{L}(\boldsymbol\theta) \\ \frac{\partial }{\partial \beta} \log \mathcal{L}(\boldsymbol\theta) \end{array}} \right] - \phi_\Sigma^{-1} (\boldsymbol\theta - \phi_\mu)
\end{aligned}
$$

If using a function such as `fminunc`, the objective function would return both the log posterior, as well as its gradient. For instance, as follows:

$$
\begin{aligned}
& \text{function} \;\; \Bigg[ \log P(\mathcal{A}, \boldsymbol\theta | \boldsymbol\phi), \;\; \frac{\partial}{\partial \boldsymbol\theta} \log P(\mathcal{A}, \boldsymbol\theta | \boldsymbol\phi) \Bigg] = \text{logposterior}(\boldsymbol\theta) \\
& ... \\
& \text{end}
\end{aligned}
$$

Then one could apply `fminunc` as usual

```matlab
[theta, varargout] = fminunc(logposterior, theta_initial, options)
```
