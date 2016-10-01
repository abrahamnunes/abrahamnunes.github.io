---
    layout: post
    title: Quick and Dirty Overview of the Bellman Equation
    author: Abraham Nunes
    date: September 30, 2016
    published: true
    status: publish
    tags: reinforcement_learning computational_psychiatry computational_neuroscience
---

<script type="text/x-mathjax-config">
MathJax.Hub.Config({
  tex2jax: {inlineMath: [['$','$'], ['\\(','\\)']]}
});
</script>
<script type="text/javascript"
  src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML">
</script>

# Markov Decision Process

A Markov decision process (MDP) has the following elements:

- A set of states, $s_t \in \mathcal{S}$
- A set of actions which depend on the current state, $a_t \in \mathcal{A}(s)$
- A policy which maps from state to action, $\pi(s) \in \mathcal{A}(s)$
- State transition probabilities, $ \mathcal{T} (s_{t+1} | s_t, a_t) $
- A reward function $ \mathcal{R} (s_{t+1} | s_t, a_t)$. We can also denote the reward received at the current time step as $r_t$

The agent operating within such a process will generally accumulate value at each state according to behaviour under some policy. This value function is typically denoted as $V^{\pi}(s)$, but for state-action pairs is typically denoted as $\mathcal{Q}^{\pi}(s, a)$.

The goal of the agent solving the MDP is to find the optimal policy such that total future value is maximized. The value functions under the optimal policy are typically denoted as either $V^{*} (s)$ or $\mathcal{Q}^{*} (s, a)$.

Over time, the reward obtained will accumulate. At the $k^{th}$ time step, the reward is $\gamma^{k} V$, where $0 < \gamma < 1$ represents a discount factor. One can liken this to a measure of impulsivity or impatience, in behavioural terms.

# Understanding the Bellman Equations

If you were to measure the value of the current state you are in, how would you do this? The intuitive way would simply be to tally the value of any rewarding properties obtained at the present instant. This, however, misses an important fact: that the current state partially determines which states one can end up in later. As such, we can expand on the previous point by stating that the value of a current state would be the sum of the reward received in the moment, plus the total value of all future rewards expected as a result. This is closer to the true answer, but we must add one final touch: the future is worth less than the present (on account of the uncertainty of the future), so we must \textit{discount} future rewards when adding them to the current reward. Adding current reward to a discounted total future reward results in what business-folk call the \textit{net present value}.

You might have noticed a problem: we can't know the future, especially _far_ into the future. This is where the Bellman equations become interesting due to the property of _recursion_. Recall that the value of the present state represents the net present value of all states in the future. This necessarily means that the value of the next state, $s'$, represents the net present value of all future rewards thereafter. If we replace the prime (') notation with subscripts denoting the time step (i.e. $s_{0}$ is the initial state, $s_{2}$ is the state at time step 2, etc.), we can see this recursion in action:

$$
\begin{equation}
V(s_{0}) = r_{0} +
\gamma \sum_{s_{1} \in \mathcal{S}} \mathcal{T}(s_{1}|s_{0}, a_{0})\mathcal{R}(s_{1}|s_{0}, a_{0}) +
\gamma^{2} \sum_{s_{2} \in \mathcal{S}} \mathcal{T}(s_{2}|s_{1}, a_{1})\mathcal{R}(s_{2}|s_{1}, a_{1}) + \cdots
\end{equation}
$$

which can be summarized as follows:

$$
\begin{aligned}
V(s) & = \sum_{t = 0}^{T} \gamma^{t} \sum_{s_{t+1} \in \mathcal{S}} \mathcal{T}(s_{t+1}|s_{t}, a_{t})\mathcal{R}(s_{t+1}|s_{t}, a_{t}) \\
	 & = \sum_{t = 0}^{T} \gamma^{t} \Bigg\langle \mathcal{R}(s_{t+1}|s_{t}, a_{t}) \Bigg\rangle_{\mathcal{T}},
\end{aligned}
$$

and where the angled brackets $\langle \cdot \rangle_{\mathcal{T}}$ denote an expectation under probability measure $\mathcal{T}$. As such,

$$
\begin{equation}
V(s_{t}) = r_{t} +
\gamma \sum_{s_{t+1} \in \mathcal{S}} \mathcal{T}(s_{t+1}|s_{t}, a_{t})V(s_{t+1}).
\end{equation}
$$

Bellman was concerned with finding the _optimal policy_, which in plain language means choosing the best possible action at each given state, where "best possible action" means the action that maximizes total future reward. The optimal control policy is thus

$$
\begin{equation}
\pi^{*}(s_{t}) = {\mathrm{arg} \max}_{a_{t}} \Bigg[\sum_{s_{t+1} \in \mathcal{S}} \mathcal{T}(s_{t+1}|s_{t}, a_{t})V^{*}(s_{t+1}) \Bigg] ,
\end{equation}
$$

and it serves to maximize the value $V^{*}(s_{t})$ at the present state:

$$
V^{*}(s_{t}) = r_{t} + \max_{a_{t}} \Bigg[\gamma \sum_{s_{t+1} \in \mathcal{S}} \mathcal{T}(s_{t+1}|s_{t}, a_{t})V^{*}(s_{t+1}) \Bigg]
$$

Typically---except for a few exceptions---this must be solved numerically.
