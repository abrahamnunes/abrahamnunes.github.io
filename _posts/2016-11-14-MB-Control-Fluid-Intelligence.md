---
    layout: post
    title: The development of model-based control as it relates to fluid intelligence
    author: Abraham Nunes
    date: November 14, 2016
    published: true
    status: publish
    draft: false
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

I am quite interested in the natural history of behavioural control over the lifespan, and have been excited to see a couple of studies about this lately. The most recent paper I read on the topic is by Potter et al. (1) which just came up in my newsfeed. I've made some notes about it here.

Potter et al. (1) sought to determine whether age-related changes in statistical learning, working memory, and fluid reasoning contributed to the positive association between age and model-based (MB) control, which had been shown by Decker et al. (2).

## Methods

### Subjects

|  Group        |   Definition      |    N      |   N Post-Exclusion    |
| -------       |  ------------     |   -----   |  ------------------   |
| Children      |   Age 9-12 years  |   22      |   19                  |
| Adolescents   |   Age 13-17 years |   23      |   22                  |
| Adults        |   Age 18-25 years |   24      |   23                  |

There were some additional missing data for the remaining subjects.

- One child did not have statistical learning data acquired
- WASI matrix-reasoning not completed for 1 adolescent and 2 adults
- 14 children, 17 adolescents, and 18 adults also completed listening recall subtest of Automated Working Memory Assessment

### Tasks

|   Domain Tested           |   Task                                |
|  ---------------          |  ------                               |
| Reinforcement learning    | Two-step task                         |
| Statistical learning      | _See Below_                           |
| Fluid reasoning           | WASI matrix reasoning and vocabulary  |

#### Two-step task

- The authors used an interesting modification of the two-step task which has a story-line that facilitates completion by children. If I am not mistaken, this may be the same one available through Wouter Kool's [`tradeoffs`](https://github.com/wkool/tradeoffs) repo.
- The task structure is the usual one for the two-step task (3), with a couple of modifications:
    - Only 150 trials were done, rather than the usual 201
        - __This may have important implications for model-fitting__
- Although neuroimaging data were not reported, the task was completed in an fMRI scanner
    - __I am unclear about why the neuroimaging data were not reported__

##### Models

The model fit to the behavioural data by Potter et al. (1) consists of an observation model

$$
P(a_t | \mathcal{Q}_t (s_t, a_t; w, \alpha, \lambda), \beta) = \frac{e^{\beta \mathcal{Q}_t (s_t, a_t; w, \alpha, \lambda) + p\cdot\mathrm{rep}(a_t)}}{\sum_{a' \in \mathcal{A}} e^{\mathcal{Q}_t (s_t, a'; w, \alpha, \lambda)+ p\cdot\mathrm{rep}(a')}},
$$

where $\beta$ is the inverse softmax temperature (decision randomness) and $\mathrm{rep}(a_t)$ is an indicator function taking value 1 if $a_t$ is the same action taken at the same step of the last trial (i.e. $I[a_t = a_{t-1}]$) weighted by a 'perseveration' parameter $p$. The function $\mathcal{Q}_t(s_t, a_t; w, \alpha, \lambda)$ is the hybrid learming model with MB and model-free (MF) components weighted by a parameter $w$

$$
\mathcal{Q}_t(s_t, a_t; w, \alpha, \lambda) = w \mathcal{Q}_t^{MB}(s_t, a_t) + (1-w)  \mathcal{Q}_t^{MF}(s_t, a_t; \alpha, \lambda).
$$

The MF component $\mathcal{Q}_t^{MF}$ is the SARSA($\lambda$) rule

$$
\begin{aligned}
\mathcal{Q}_t^{MF}(s_t^{(Step\,2)}, a_t^{(Step\,2)}; \alpha, \lambda) & = \mathcal{Q}_{t-1}^{MF}(s_t^{(Step\,2)}, a_t^{(Step\,2)}; \alpha, \lambda) + \alpha \Big(r_t - \mathcal{Q}_{t-1}^{MF}(s_t^{(Step\,2)}, a_t^{(Step\,2)}; \alpha, \lambda) \Big) \\
\mathcal{Q}_t^{MF}(s_t^{(Step\,1)}, a_t^{(Step\,1)}; \alpha, \lambda) & = \mathcal{Q}_{t-1}^{MF}(s_t^{(Step\,1)}, a_t^{(Step\,1)}; \alpha, \lambda) + \alpha \Big(\lambda \mathcal{Q}_{t-1}^{MF}(s_t^{(Step\,2)}, a_t^{(Step\,2)}; \alpha, \lambda) -  \mathcal{Q}_{t-1}^{MF}(s_t^{(Step\,1)}, a_t^{(Step\,1)}; \alpha, \lambda) \Big),
\end{aligned}
$$

where $\alpha$ is the learning rate and $\lambda$ is an eligibility trace parameter that governs the amount by which value at the second step of trial $t$ backs up to the first step of trial $t$. __I am unclear whether the authors decayed the previous MF values by $1-\alpha$, i.e. $(1-\alpha)\mathcal{Q}_{t-1}^{MF} + \cdots$.

The MB learning rule was not explicitly reported by Potter et al. (1), but is generally take the form of Bellman's equation:

$$
\forall a_t \in \mathcal{A}, \;\; \mathcal{Q}_t^{MB}(s_t^{(Step\,1)}, a_t) = \sum_{s_t^{(Step\,2)} \in \mathcal{S}^{(Step\,2)}} \mathcal{T}(s_{t}^{(Step\,2)}|s_t^{(Step\,1)}, a_t) \, \max_{a' \in \mathcal{A}} \mathcal{Q}_{t-1}^{MF}(s_t^{(Step\,2)}, a'; \alpha, \lambda),
$$

where $\mathcal{T}$ is a transition matrix. __I am unclear whether the transition matrix was specified or learned by the model__. In the case of 150 trials, this may not be a trivial matter. However, I suspect that since the authors had subjects perform 50 trials before the main task, that the MB learning rule did not include a module for learning the transition probabilities.

##### Model-Fitting

This was done in the same hierarchical Bayesian fashion as Daw et al. (3), which used empirical priors for the parameters, as follows:

|   Parameters          |   Empirical Prior Distribution    |
|  ------------         |  ------------------------------   |
|   $\alpha, \gamma, w$ |   $Beta(a=1.1, b=1.1)$            |
|   p                   |   $\mathcal{N}(\mu=0, \sigma=1)$  |
|   $\beta$             |   $Gamma(a=1.2, b=5)$             |

The use of empirical priors differs from approaches such as that by Huys et al. (4), which computes the MAP estimates of parameters and MLE of hyperparameters jointly using expectation maximization. This is the approach employed in our group's [`fitr`](https://abrahamnunes.github.io/fitr) toolbox.

##### Model-Comparison

The authors did not fit multiple models to their data, and so model-comparison was not done. Since the models fit to the two-step task are similar in form, it is unclear whether this is of great consequence. However, with appropriate model-selection procedures that guard against overfitting, they may have been able to improve the fit of their models by including multiple variations on the form described above.

##### Theory-Free Analysis

The authors also conducted the usual linear mixed-effects regression using the first-stage "stay vs. switch" as the response variable. In this case, one looks for an interaction between reward at the last trial (1 or 0), and whether the transition at the last trial was common or rare.


#### Statistical Learning Task

This task involves twelve stimuli which are grouped into four "triplets" (i.e. each triplet has three stimuli which belong to it, but not to any of the other triplets). For example triplets may include (A,B,C), (D,E,F), (G,H,I), (J,K,L). The subjects initially have a familiarization phase where the stimuli are presented in sequence, one by one. However, the subjects are not aware of the triplet structure, and stimuli are presented in a fixed inter-triplet order, with triplets being interleaved. For example, a sequence may have included _A$\to$B$\to$C$\to$G$\to$H$\to$I$\to$D$\to$E$\to$F..._ which one can see preserves the order of symbols within their assigned triplet, but interleaves the triplets themselves. Again, the subjects are unaware of this structure.

In the second paprt of this task, subjects are presented with 32 trials of two triplets (with all symbols displayed at once). However, at this point, one of the triplets being presented were never observed before. For example (D, E, F) and (K, H, A). It may be the case that the never-before seen triplet included totally new symbols, but this was not specified in the paper. Subjects at this stage were required to identify which triplet was more familiar to them.

#### Fluid Reasoning Task

Thse included matrix reasoning (fluid reasoning) and vocabulary sections (crystallized intelligence) of the Weschler Abbreviated Scale of Intelligence. The latter was included to determine whether any observed effects of fluid reasoning were due to a more broadly constructed concept of intelligence.

#### Working Memory Task

This task was the listening-recall subtest of the Automated Working Memory Assessment. In this task, subjects are read 8 single sentences and 7 pairs of sentences. At the end of each sentence, the subject states whether the sentence was true or false, then repeats the last word of the sentence. For the section with sentence pairs, the subject reports whether each sentence is true or false immediately after the respective sentence is read, but recalls the last word of each sentence once both sentences have been read. The score is computed by pooling both processing (i.e. correctly identifying true vs. false) and recall portions. Overall, this task is meant to assess the ability to hold information in memory (for recall) despite interference (processing).

#### Mediation Analysis

I am acutually unfamiliar with this type of analysis, and will edit this post once I learn a bit more about it.

## Results

### Two-step task models

The distribution of parameter estimates for all subjects were as follows:

|   Parameter   |   Median  |   IQR         |
|  -----------  |  -------- |  ------       |
|   w           |   0.52    |   0.16-0.71   |
|   $\alpha$    |   0.42    |   0.05-0.76   |
|   $\lambda$   |   0.62    |   0.29-0.90   |
|   $\beta$     |   2.91    |   2.40-4.02   |
|   $p$         |   0.14    |   -0.04-0.36  |

The authors found that MB control increased with age (correlation of age group with MB weight parameter $w$ was 0.30, p=0.01). This was also shown through the linear mixed-effects regression, where all age groups showed a significant main effect of prior-trial reward (which indicates MF control), but only adolescents and adults showed significant main effects of reward-transition interaction (which indicates use of MB control). That adults and adolescents used MB control was also reflected in analysis of reaction times following rare transitions. The idea here is that if one is 'surprised' by a rare transition, his or her reaction time at the second stage will slow. Thus, slower reaction times at the second stage choice may reflect increased use of MB control.

Importantly, all groups showed equal explicit understanding of the transition structure when asked specific questions about the transition probabilities.

### Statistical learning

All groups demonstrated performance on this task at above-chance levels, with accuracy improving with age. However, statistical learning did not mediate the relationship between age and MB control. The authors indicated that this may have been related to small sample size.

The authors commented---quite reasonably---that statistical learning may be involved in the process of learning a cognitive model of the task. To determine whether this was the case, they compared the second-stage reaction times following rare transitions against the statistical learning measures, finding a positive relationship (r=0.40, p=0.0008). However, it stands that in the study by Pearson et al. (1), statistical learning had no mediating influence on the increased use of MB control with age.


### Fluid Reasoning

Fluid reasoning
- Increased with age (r=0.53, p$<0.0001$)
- Correlated with MB control parameter $w$ (r=0.41, p=0.0001)
- Fully mediated the relationship between age and MB control
    - __I need to look at this type of analysis more closely to learn exactly what this means__
    - The mediating role of fluid reasoning was robust to testing against crystallized intelligence, which despite showing a significant relationships with both age and MB control, did not mediate the relationship between age and MB control.
- Fully mediated the relationship between (A) statistical learning and MB control, and (B) age and statistical learning in a directionally specific fashion (i.e. statistical reasoning did not mediate the relationship between fluid reasoning and MB control)

The authors concluded that (A) the age related effect of fluid reasoning was a specific mediator (independent of general intelligence) of the relationship between age and MB control, (B) that fluid reasoning may account for the relationship between statistical learning ability and MB control, and (C) that fluid reasoning is an important factor in the development of statistical learning with age.

### Working Memory

- Unfortunately, 45\% of the subjects reached ceiling level performance on the working memory task, limiting the authors' analysis of the effects of working memory on MB control
    - Notwithstanding, they still found a positive correlation of working memory task performance and the MB control parameter $w$ (r=0.31, p=0.03)
    - Working memory performance did not significantly correlate with age


## My Remarks and Implications

Fluid reasoning, and not crystallized intelligence, is likely an important component of---or contributor to---MB control. To this end, we must consider the utility of using certain common IQ tests, such as the North American Adult Reading Test (7) as covariates in analyses of MB and MF control modeling

Working memory is likely important for MB control (8), but we must be careful to use probes that can account for ceiling effects. To this end, using a task such as Operation Span (OPSPAN; Refs. 9, 10), which can be extended in terms of recall and processing demands (i.e. longer sequences) may be beneficial to avoid ceiling effects.

The finding that children were similarly able to appreciate the transition dynamics in the task, as reflected by both their answers to questions about transition structure as well as their reaction time data, compared to adults is interesting in light of their lesser use of MB control. Potter et al. (1) suggested that this was related to a reduced ability to use that knowledge in decision making. I wonder whether this may be at all associated with the pattern of developmental myelination of corticostriatal projections. Studying a similar task after diffusion tensor imaging may be of interest in the future.

MB control involves building an internal representation of environmental state dynamics and using that model to traverse potential sequences of states and actions during decision-making. Control of the balance between MB and MF learning, represented by the parameter $w$ has been called "arbitration," (5) although the mechanisms of this arbitration are unclear. It is possible that the balance of MB and MF control is determined by the precision of estimates from the MB and MF systems (6) in which case an individual's ability to (A) maintain an accurate representation of the environmental transition model and (B) implement that model efficiently "online" during decision-making will significantly influence the expression of MB control. It may be the case that fluid reasoning influences the ability of "model traversal" during decision-making. The relationship with working memory, however, would be of greater interpretability, since traversing a decision tree would require accurate maintenance of a representation specifying the current path taken down that tree. This would be necessary to back-up value accurately.

## References

1. Potter, T. C., Bryce, N. V, & Hartley, C. A. (2016). Cognitive components underpinning the development of model-based learning. Developmental Cognitive Neuroscience, http://doi.org/10.1016/j.dcn.2016.10.005
2. Decker, J. H., Otto, A. R., Daw, N. D., & Hartley, C. A. (2016). From Creatures of Habit to Goal-Directed Learners: Tracking the Developmental Emergence of Model-Based Reinforcement Learning. Psychological Science, 27(6), 848–858. http://doi.org/10.1177/0956797616639301
3. Daw, N. D., Gershman, S. J., Seymour, B., Dayan, P., & Dolan, R. J. (2011). Model-based influences on humans’ choices and striatal prediction errors. Neuron, 69(6), 1204–1215. http://doi.org/10.1016/j.neuron.2011.02.027
4. Huys, Q. J. M., Cools, R., Gölzer, M., Friedel, E., Heinz, A., Dolan, R. J., & Dayan, P. (2011). Disentangling the roles of approach, activation and valence in instrumental and pavlovian responding. PLoS Computational Biology, 7(4). http://doi.org/10.1371/journal.pcbi.1002028
5. O’Doherty, J. P., Lee, S. W., & McNamee, D. (2015). The structure of reinforcement-learning mechanisms in the human brain. Current Opinion in Behavioral Sciences, 1(April 2016), 94–100. http://doi.org/10.1016/j.cobeha.2014.10.004
6. Wan Lee, S., Shimojo, S., & O’Doherty, J. P. (2014). Neural Computations Underlying Arbitration between Model-Based and Model-free Learning. Neuron, 81(3), 687–699. http://doi.org/10.1016/j.neuron.2013.11.028
7. Uttl, B. (2002). North American Adult Reading Test: age norms, reliability, and validity. Journal of Clinical and Experimental Neuropsychology, 24(8), 1123–37. http://doi.org/10.1076/jcen.24.8.1123.8375
8. Otto, A. R., Raio, C. M., Chiang, A., Phelps, E. A., & Daw, N. D. (2013). Working-memory capacity protects model-based learning from stress. Proceedings of the National Academy of Sciences of the United States of America, 110(52), 20941–20946. http://doi.org/10.1073/pnas.1312011110
9. Conway, A. R. a, Kane, M. J., & Al, C. E. T. (2005). Working memory span tasks : A methodological review and user ’ s guide. Psychonomic Bulletin & Review, 12(5), 769–786. http://doi.org/10.3758/BF03196772
10. Unsworth, N., Heitz, R. P., Schrock, J. C., & Engle, R. W. (2005). An automated version of the operation span task. Behavior Research Methods, 37(3), 498–505. http://doi.org/10.3758/BF03192720
