---
    layout: post
    title: A simple example of model fitting with `fitr`
    author: Abraham Nunes
    date: September 10, 2016
    published: true
    status: publish
    keywords: fitr reinforcement_learning computational_psychiatry
---

## Introduction

Here I present a simple example of how to fit reinforcement learning models to behavioural data using our [`fitr` package](https://abrahamnunes.github.io/fitr) which is currently still under extensive development. The plots generated herein use code I have compiled into a toolbox called [`pqplot`](https://abrahamnunes.github.io/pqplot).

This post won't review how tasks are structured. Rather, it will show the simplest example of model fitting, assuming that the tasks and likelihood functions are built in to the `fitr` toolbox.

## Generating Synthetic Data from a Go-Nogo Task

Unless you have already collected behavioural data from real subjects, we can simulate some subjects on a task in order to generate synthetic data. We will use a simple two state, two action task with the following reward structure:

|           | Action 1                  | Action 2                  |
| -------   | --------                  | --------                  |
| State 1   | `1*Binomial(n=1, p=0.7)`  | `-1*Binomial(n=1, p=0.7)` |
| State 2   | `-1*Binomial(n=1, p=0.7)` | `1*Binomial(n=1, p=0.7)`  |

This can be thought of as a go-nogo task with two states. There will be 200 trials in the task, and the transition probability across trials (i.e. $P(s_1 \to s_2)$ or $P(s_2 \to s_1)$) is 0.5. These task parameters are specified as follows:

``` matlab

% Task parameters
taskparams.ntrials  = 200;
taskparams.nstates  = 2;
taskparams.nactions = 2;
taskparams.preward  = [0.7, 0.7; 0.7, 0.7];
taskparams.rewards  = [1, -1; -1, 1];
taskparams.ptrans   = [0.5; 0.5];

```

Now we will generate a group of subjects. The subjects' learning rule will be a simple Rescorla-Wagner rule with learning rate $\alpha$ and softmax inverse temperature $\beta$:

$$
Q_t(s_t, a_t; \alpha) = (1-\alpha)Q_{t-1}(s_t, a_t; \alpha) + \alpha \Big(r_t - Q_{t-1}(s_t, a_t; \alpha)\Big)
$$

$$
P(a_t|Q_t(s_t, a_t); \beta) = \frac{e^{\beta Q_t(s_t, a_t; \alpha)}}{\sum_{a'} e^{\beta Q_t(s_t, a'; \alpha)}}
$$

For the present example, the learning rule is encoded in the task code, which we will present below. Setting up a group of subjects is straightforward: just specify the number of subjects, initialize an $N_{subject} \times K_{parameter}$ array within the subject structure, and populate each column with parameters from the distributions of your choice. Here, we use a beta distribution for the learning rate and a gamma distribution for the inverse temperature.

``` matlab
subjects.N           = 20;
subjects.params      = zeros(subjects.N, 2);
subjects.params(:,1) = betarnd(1.1, 1.1, [subjects.N, 1]); %learning rate
subjects.params(:,2) = gamrnd(5., 1., [subjects.N, 1]); %inverse temperature
```

The current task we are implementing can be fouind in the `gonogobandit.m` class within the `fitr-matlab/tasks` folder. To generate data, simply run

``` matlab
results = gonogobandit.vanilla(subjects, taskparams);
```

The `results` structure is as follows

```
results =

1x50 struct array with fields:

    S
    rpe
    A
    R
```

where `results(i).S` is a vector of states for subject $i$, `results(i).A` is a vector of actions, `results(i).R` is a vector of rewards received, and `results(i).rpe` is a vector of reward prediction errors. Only `S`, `A`, and `R` are required for model fitting, though.

## Composing models

Models are simple structures with likelihood functions and the respective parameters required for those likelihood functions. Parameters in this example are specified from built-in functions that return a structure with fields `name` and `rng`. The `rng` field can take values of "unit" (meaning interval from [0, 1]), "pos" (meaning interval from [0, +infinity)), or otherwise if specified by the user. These parameter generating functions are found in `fitr-matlab/utils/rlparam.m` class.

Let's generate 2 models:

``` matlab
model1.lik      = @gnbanditll.lrbeta;
model1.param    = rlparam.learningrate();
model1.param(2) = rlparam.inversetemp();

model2.lik      = @gnbanditll.lrbetarho;
model2.param    = rlparam.learningrate();
model2.param(2) = rlparam.inversetemp();
model2.param(3) = rlparam.rewardsensitivity();

model3.lik      = @gnbanditll.lr2beta;
model3.param    = rlparam.learningrate();
model3.param(2) = rlparam.learningrate();
model3.param(3) = rlparam.inversetemp();

model4.lik      = @gnbanditll.lr2betarho;
model4.param    = rlparam.learningrate();
model4.param(2) = rlparam.learningrate();
model4.param(3) = rlparam.inversetemp();
model4.param(4) = rlparam.rewardsensitivity();

model5.lik      = @gnbanditll.randmodel;
model5.param    = rlparam.inversetemp();
```

Note that each model's likelihood function is specified in the `gnbanditll.m` class, located in   `fitr-matlab/likelihood_functions`.

## Fitting Models

We can now run the model fitting procedures, which are drawn from Huys et al. (2011).

First, we specify the model fitting options, including `maxiters`, which is self-explanatory, `nstarts`, which specifies the number of random parameter initializations for optimization, and `climit`, which specifies the stopping criterion. Model fitting stops when the current fit's log-posterior probability is within `climit` absolute difference from the last fit iteration.

Model fitting is done with the `fitmodel()` function, which accepts the experimental results the model specification, and options as arguments.

``` matlab
fitoptions.maxiters   = 1000;
fitoptions.nstarts    = 2;
fitoptions.climit     = 10;

fit1 = fitmodel(results, model1, fitoptions);
fit2 = fitmodel(results, model2, fitoptions);
fit3 = fitmodel(results, model3, fitoptions);
fit4 = fitmodel(results, model4, fitoptions);
fit5 = fitmodel(results, model5, fitoptions);
```

## Model Selection

We implement the Bayesian Model Selection of Rigoux et al. (2014), the code of which was drawn from Samuel Gershman's []`mfit` package](https://github.com/sjgershm/mfit). It can be implemented as follows:

``` matlab

models(1).name = 'Model 1';
models(1).fit  = fit1;

models(2).name = 'Model 2';
models(2).fit  = fit2;

models(3).name = 'Model 3';
models(3).fit  = fit3;

models(4).name = 'Model 4';
models(4).fit  = fit4;

models(5).name = 'Model 5';
models(5).fit  = fit5;

bms = BMS(models);

```

## Plotting results

We can now look at how well the model selection and parameter estimation procedures worked in this simple case:

Model selection results are as follows:

<img src="/figures/rwmodelselres.svg" width="100%">

The model selection metric we use from the above plot is the "Protected Exceedance Probability," which was reviewed by Rigoux et al. (2014).

The model parameter fits are presented in the plot below. Here, each column is a model parameter, $\alpha$ being the learning rate and $\beta$ being the softmax parameter. Dark lines represent the actual parameter values (in unconstrained space), and light lines represent the parameter estimates (in unconstrained space). Note that Model 1, which was specified by the Bayesian Model Selection procedures as having the highest protected exceedance probability, has the best parameter fit overall across each parameter in the model.

<img src="/figures/rwdemoaeplots.svg" width="100%">

## Next Steps

I hope to build some documentation soon describing the overall structure of the `fitr` package, and I hope to expand the number of built in models. First, I am working on speeding up convergence and improving parameter estimates by allowing the model fitting procedure to make use of analytical gradients.

## References and Further Reading

1. Huys, Q. J. M., Cools, R., Gölzer, M., Friedel, E., Heinz, A., Dolan, R. J., & Dayan, P. (2011). Disentangling the roles of approach, activation and valence in instrumental and pavlovian responding. PLoS Computational Biology, 7(4). http://doi.org/10.1371/journal.pcbi.1002028
2. Gershman, S. J. (2016). Empirical priors for reinforcement learning models. Journal of Mathematical Psychology, 71, 1–6. http://doi.org/10.1016/j.jmp.2016.01.006
3. Rigoux, L., Stephan, K. E., Friston, K. J., & Daunizeau, J. (2014). Bayesian model selection for group studies - Revisited. NeuroImage, 84, 971–985. http://doi.org/10.1016/j.neuroimage.2013.08.065
4. Daw, N. D. (2011). Trial-by-trial data analysis using computational models. Decision Making, Affect, and Learning: Attention and Performance XXIII, 1–26. http://doi.org/10.1093/acprof:oso/9780199600434.003.0001
