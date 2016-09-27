---
    layout: post
    title: Notes on Noradrenergic Modulation of Reinforcement Learning
    author: Abraham Nunes
    date: September 25, 2016
    published: false
    status: draft
    tags: computational_psychiatry computational_neuroscience computational_neuromodulation
---

<script type="text/x-mathjax-config">
MathJax.Hub.Config({
  tex2jax: {inlineMath: [['$','$'], ['\\(','\\)']]}
});
</script>
<script type="text/javascript"
  src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML">
</script>

The role of dopamine (DA) in reinforcement learning (RL) is extensively studied, but relatively less is understood about the effects of norepinephrine (NE). This post includes some notes (mainly point form) on the potential effects of NE signaling on computational models of RL as they pertain to rodent or primate behavioural data.

# A Simple Model of Decision-Making

In the RL framework, decision-making can be understood as a combination of three elements:

- Valuation
- Action-selection
- Learning

## Valuation

Doya (2008) presented an elegant description of valuation, highlighting that decisions are difficult because "decisions can result in rewards or punishments of different amounts at different timings with different probabilities." He outlined this mathematically as follows:

$$
V = f(\text{amount}) \times g(\text{delay}) \times h(\text{probability}),
$$

where $f$ is the _utility function__, $g$ is the _temporal discounting function_, and $h$ is the _probabilistic outcome valuation function_. The function $h$ allows us to account for the fact that probabilistic outcomes may be over- or under-valued.

## Action-Selection

In addition to valuation of various states or actions, the decision-maker must select the actions he or she will impart on the world. In RL, this is often done through a softmax probability function:

$$
P(a_t) = \frac{e^{\beta V(a_t)}}{\sum_{a' \in \mathcal{A}} e^{\beta V(a')}},
$$

where $a_t$ is the action taken at time $t$ and $\mathcal{A}$ denotes the space of all available actions. The parameter $\beta$ is the inverse softmax temperature which represents stochasticity in action-selection.

## Learning

When learning the behavioural policy that maximizes total future reward, one must correctly associate reward values with the appropriate stimuli and actions. Doya (2008) proposed three methods by which these associations may be learned in a dynamic environment:

- Eligibility traces
- Temporal difference learning
- Model-based learning

# Factors Affecting Decisions and Learning

Doya (2008) described three factors affecting decisions and learning:

- __Needs & Desires__
    - Should be reflected in the decision maker's utility curve
    - Utility functions typically have sigmoid shape, reflecting threshold and saturation points for consumption
- __Risk & Uncertainty__
    - Can be modulated by adding nonlinearity to the probabilistic outcome valuation function $h$, such that one accounts for behaviours that are
        - Risk-averse
        - Risk-seeking
    - Uncertainties in decision making stem from
        - Randomness in environmental dynamics
        - "Unexpected variation of the environment"
            - Unexpected changes in the environment should bias a decision-maker toward exploration and a higher learning rate
                - Exploration may be induced by reducing the inverse temperature parameter in the softmax action selection formula
        - The decision-maker's limited knowledge
            - Limited knowledge about the environment should bias a decision-maker toward exploration and a higher learning rate
- __Time Spent & Time Remaining__
    - Effects on the learning rate
        - When the environment is constant, the optimal approach is to
            - Start with high learning rate
            - Decay learning rate inversely proportional to the number of trials
        - When environment is dynamic, optimal approach is to
            - Set the learning rate based on an estimate of the remaining time (don't overwrite knowledge while it is still valid!)
    - Effects on temporal discounting
        - Discounting rates should depend on the estimated amount of time remaining. If there is limited time remaining for decision-making, the time horizon should be set short, and vice versa.

# Norepinephrine and Neural Gain



# References

- Doya, K. (2008). Modulators of decision making. Nature Neuroscience, 11(4), 410–416. http://doi.org/10.1038/nn2077
- Hauser, T. U., Fiore, V. G., Moutoussis, M., & Dolan, R. J. (2016). Computational Psychiatry of ADHD: Neural Gain Impairments across Marrian Levels of Analysis. Trends in Neurosciences, xx, 1–11. http://doi.org/10.1016/j.tins.2015.12.009
-
