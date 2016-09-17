---
	layout: post
	title: Inflammation and Reward Sensitivity
	author: Abraham Nunes
	date: September 16, 2016
	published: true
	status: publish
	keywords: computational_psychiatry paper_review reinforcement_learning neurobiology
---

<script type="text/x-mathjax-config">
MathJax.Hub.Config({
  tex2jax: {inlineMath: [['$','$'], ['\\(','\\)']]}
});
</script>
<script type="text/javascript"
  src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML">
</script>

This post includes some brief notes on the recent paper by Harrison, Voon, et al. [1] in _Biological Psychiatry_, which studied the effects of systemic inflammation on reinforcement learning in humans.

Harrison et al [1] used a task previously implemented by Pessiglione et al. [2] whose reward structure is represented by the following table:

|	State Name 	|	Action 1 Value		|	Action 2 Value			|
|	----------	|	--------------		|	--------------			|
|	"Gain"		|	`+1*binornd(1, 0.8)`|	`+1*binornd(1, 0.2)`	|
|	"Neutral"	|	0					|	0						|
|	"Loss"		|	`-1*binornd(1, 0.8)`|	`-1*binornd(1, 0.2)`	|

Each state consisted of two unique visual stimuli, from which the subject was required to select one of the two. As such, in the above table, we refer to the stimuli as "Action 1" and "Action 2", since they can be considered representative of the actions available to the subject in each given state.

The authors modeled subjects' behavioural data using a two-parameter Rescorla-Wagner rule as the learning model

$$
Q_{t+1}(s_t, a_t) = Q_t(s_t, a_t) + \alpha \delta_t, 
$$

where $\delta_t = R_t - Q_t(s_t, a_t)$. The free parameters $\alpha$ and $R_t$ represent learning rate and subjective reward, respectively. The observation model consisted of a standard softmax with inverse temperature parameter $\beta$. Inverse temperature is quite well named by Harrison et al. as "choice randomness" [1].

The authors found no association between inflammation and the learning rate or inverse temperature parameters, but did observe a statistically significant association between subjective reward $R_t$ and inflammation. Specifically, the authors observed an increase in the magnitude of subjective value of the punishment stimuli during the inflammation condition. These results were consistent with those of Huys et al. [12], who found that anhedonia was related almost exclusively to reward sensitivity, rather than learning rate or otherwise.

Using model-based fMRI, Harrison et al. [1] replicated findings from [2] demonstrating correlations with reward prediction error in the ventral striatum, as well as punishment prediction error correlation (a negative correlation with reward prediction error) in the left insula. The authors then went on to demonstrate that the inflammation condition was associated with the following statistically significant changes on fMRI:

- Reduced encoding of reward prediction error in the ventral striatum
- Increased right insula encoding of punishment prediction error

> NB: I do not quite understand how to interpret the change involving the right insula, given that the initial correlation of negative reward prediction errors only in the left insula. To this end, I need to further develop an understanding of model-based fMRI, which I am currently working on for the `fitr` package. 

The study by Harrison et al. [1] suggests that mild systemic inflammation may be associated with heightened punishment sensitivity, and that these neurocomputational processes may be the result of inflammation-related changes at the ventral striatum and insulae. They review the relevance of their findings by noting that impaired reward appraisal is observed in so-called "sickness behaviour," which is characteristic of several conditions, including depression. As such, the results of Harrison et al. [1] may be a starting point for bridging neuroimmunological theories of depression with observable phenotypes [9, 10]. This study also highlights that ventral striatal encoding of reward prediction error may be sensitive to systemic inflammation; the authors suggest that this may "afford one element of an efficient mechanism for the rapid reorientation of behaviour in the face of acute infection." 

This shift in sensitivity from reward to punishment may be relevant to the learned helplessness model of depression, in which abnormal reward vs. punishment sensitivity is implicated [11]

Given the known association between mesolimbic dopaminergic signalling and reward prediction error, one may assume that the changes observed by Harrison et al. [1] have dopaminergic underpinnings. However, the authors astutely note that their study could not address this question directly. Notwithstanding, Haloperidol administration during the same task by Pessiglione et al. [2] showed similar results to the present study with respect to differences in ventral striatal reward prediction error signalling. Harrison et al. hypothesize, then, that inflammation may have altered dopamine release at the ventral striatum. This hypothesis has been tested in rodents, wherein systemic inflammation resulted in (A) abnormal dopaminergic tone at the ventral striatum [3], and in humans, whose presynaptic dopaminergic sythesis and release is reduced after systemic inflammation [4]. 

## Some interesting references made by Harrison et al. [1]

- Some cytokines, such as IFN-$\alpha$ inhibit dopamine synthesis by limiting the amount of tetrahydrobiopterin in the CNS [5]. 
	- Tetrahydrobiopterin is an essential cofactor for the rate limiting enzyme of dopamine synthesis: tyrosine hydroxylase
- Inflammation can increase the expression of monoamine transporters [6-8] and indoleamine 2,3-dioxygenase (a tryptophan-degrading enzyme) [6]

# References 

1. Harrison et al. (2016) A Neurocomputational Account of How Inflammation Enhances Sensitivity to Punishments Versus Rewards. _Biol Psychiatry_. 80:73-81
2. Pessiglione et al. (2006) Dopamine-dependent prediction errors underpin reward-seeking behaviour in humans. _Nature_. 442:1042-45
3. Borowski et al. (1998) Lipopolysaccharide, central in vivo biogenic amine variations, and anhedonia. _Neuroreport_. 9:3797-3802
4. Capuron et al. (2012) Dopaminergic mechanisms of reduced basal ganglia responses to hedonic reward during interferon alfa administration. _Arch Gen Psychiatry_. 69:1044-1053
5. Kitagami et al. (2003) Mechanism of systemically injected interferon-alpha impeding monoamine biosynthesis in rats: Role of nitric oxide as a signal crossing the blood-brain barrier. _Brain Res_. 978: 104-114
6. Felger et al. (2012) Cytokine effects on the basal ganglia and dopamine function: The subcortical source of inflammatory malaise. _Front Neuroendocrinol_. 33:315-327=
7. Kamata et al. (2000) Effect of single intracerebrovascular injection of alpha-interferon on monoamine concentrations in the rat brain. _Eur Neuropsychopharmacol_ 10:129-132
8. Shuto et al. (1997) Repeated interferon-alpha administration inhibits dopaminergic neural activity in the mouse brain. _Brain Res_ 747:348-351
9. Dantzer et al. (2008) From inflammation to sickness and depression: When the immune system subjugates the brain. _Nat Rev Neurosci_ 9:46-56
10. Dowlati et al. (2010) A meta-analysis of cytokines in major depression. _Biol Psychiatry_. 67:446-457
11. Seligman, ME (1972) Learned helplessness. _Annu Rev Med_. 23:407-412
12. Huys et al. (2013) Mapping anhedonia onto reinforcement learning: A behavioural meta-analysis. _Biol Mood Anxiety Disord_ 3:1-16