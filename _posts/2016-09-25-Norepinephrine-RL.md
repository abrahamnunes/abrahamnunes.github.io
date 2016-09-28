---
    layout: post
    title: Rough Notes on Noradrenergic Modulation of Reinforcement Learning
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

The role of dopamine (DA) in reinforcement learning (RL) is extensively studied, but relatively less is understood about the effects of norepinephrine (NE). This post includes some notes (very roughly done, mainly in point form) on the potential effects of NE signaling on computational models of RL as they pertain to rodent or primate behavioural data.

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

- Aston-Jones & Cohen's (2005) _Adaptive Gain Theory_ of locus ceruleus noradrenergic function associates noradrenergic signaling with the balance of exploration and exploitation during intertemporal choice:
    - There are two modes of locus ceruleus neuronal operation (Aston-Jones & Cohen, 2005):
        - Tonic activation
            - Moderate tonic LC activity are associated with task engagement and accurate task performance (Jepma et al. 2010)
            - Elevated tonic LC activity are associated with distractible behaviour and poor task performance (Jepma et al. 2010)
            - Low to absent tonic LC activity are associated with drowsiness and inattention (Jepma et al. 2010)
            - Jepma et al. (2010) tested the hypothesis that tonic LC activity promotes exploratory control state by administering reboxetine (selective norepinephrine reuptake inhibitor), citalopram, or placebo to subjects performing a 4-armed bandit task:
                - Reboxetine was used to selectively increase norepinephrine levels
                - Citalopram is known to increase serotonin levels without changing norepinephrine levels in the prefrontal cortex
                - Subjects included 52 healthy university students without past history of psychiatric or relevant somatic illnesses.
                - Jepma et al. (2010) fit three reinforcement learning models to behavioural data on a gambling task. The resulting parameter estimates are presented in the following table:

![Table of parameter fits](http://www.abrahamnunes.com/images/plots/jepmaparams.png)

                - In all models fit by the authors, the inverse temperature parameter governing the gain on softmax action selection.
        - Phasic activation
            - Salient and arousing stimuli elicit phasic activation of locus ceruleus neurons (Aston-Jones & Cohen, 2005)
            - Phasic LC activity is not present during very low or elevated tonic levels of LC activity. Rather, phasic LC activity is typically only observed during moderate tonic levels of LC activity (Jepma et al. 2010)
    - Administration of propranolol to human subjects during performance of an intertemporal risky choice task has demonstrated mixed results, with one study demonstrating no effect on the overall proportion of risky choices (Campbell-Meiklejohn et al. 2010), and another demonstraing increased choice stochasticity at higher loss probabilities (Rogers et al. 2004)


![Softmax Surface](http://www.abrahamnunes.com/images/plots/softmaxsurface.png)

# References

- Aston-Jones, G. and Cohen, J.D. (2005) AN INTEGRATIVE THEORY OF LOCUS COERULEUS-NOREPINEPHRINE FUNCTION: Adaptive Gain and Optimal Performance. Annu. Rev. Neurosci. 28, 403–450
- Campbell-Meiklejohn, D., Wakeley, J., Herbert, V., et al., (2010). Serotonin and dopamine play complementary roles in gambling to recover losses. Neuropsychopharmacology. 36(2): 402-410.
- Doya, K. (2008). Modulators of decision making. Nature Neuroscience, 11(4), 410–416. http://doi.org/10.1038/nn2077
- Hauser, T. U., Fiore, V. G., Moutoussis, M., & Dolan, R. J. (2016). Computational Psychiatry of ADHD: Neural Gain Impairments across Marrian Levels of Analysis. Trends in Neurosciences, xx, 1–11. http://doi.org/10.1016/j.tins.2015.12.009
- Jepma, M. et al. (2010) The role of the noradrenergic system in the exploration-exploitation trade-off: a psychopharmacological study. Front. Hum. Neurosci. 4, 170
- Rogers, R.D., Lancaster, M., Wakeley, J., Bhagwagar, Z., (2004). Effects of beta-adrenoceptor blockade on components of human decision making. Psychopharmacology. 172(2):157-164.
