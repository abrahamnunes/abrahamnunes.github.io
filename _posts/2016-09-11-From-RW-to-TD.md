---
    layout: post
    title: From Classical to Instrumental Conditioning
    author: Abraham Nunes
    date: September 11, 2016
    published: false
    status: draft
    keywords: reinforcement_learning computational_psychiatry machine_learning
---

<script type="text/x-mathjax-config">
MathJax.Hub.Config({
  tex2jax: {inlineMath: [['$','$'], ['\\(','\\)']]}
});
</script>
<script type="text/javascript"
  src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML">
</script>

# The Rescorla-Wagner rule

$$
\mathcal{Q}_t(s_t, a_t) = \mathcal{Q}_{t-1}(s_t, a_t) + \alpha \Big(r_t - \mathcal{Q}_{t-1}(s_t, a_t) \Big)
$$

# Temporal difference rules

## SARSA

$$
\mathcal{Q}_t(s_t, a_t) = \mathcal{Q}_{t-1}(s_t, a_t) + \alpha \Big(r_t + \gamma \mathcal{Q}_{t-1}(s_{t+1}, a_{t+1}) - \mathcal{Q}_{t-1}(s_t, a_t) \Big)
$$

## Q-Learning

$$
\mathcal{Q}_t(s_t, a_t) = \mathcal{Q}_{t-1}(s_t, a_t) + \alpha \Big(r_t +  \gamma \underset{a \in \mathcal{A}}{\max} \mathcal{Q}_{t-1}(s_{t+1}, a) - \mathcal{Q}_{t-1}(s_t, a_t) \Big)
$$
