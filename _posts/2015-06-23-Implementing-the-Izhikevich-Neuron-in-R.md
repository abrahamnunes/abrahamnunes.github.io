---
layout: post
title: Implementing the Izhikevich Neuron in R
author: Abraham Nunes
published: true
status: publish
draft: false
tags: Computational_Neuroscience
---
 
<script type="text/x-mathjax-config">
MathJax.Hub.Config({
  tex2jax: {inlineMath: [['$','$'], ['\\(','\\)']]}
});
</script>
<script type="text/javascript"
  src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML">
</script>
 
The following is an implementation of the Izhikevich Neuron in R. More details can be found at [Eugene Izhikevich's webpage](http://www.izhikevich.org/publications/spikes.htm).
 
Essentially, the Izhikevich model seeks to balance realistic modeling of neuronal membrane dynamics with computational efficiency. The model is a set of coupled differential equations:
 
$$
\frac{dv}{dt} = 0.04v^{2} + 5v + 140 - u + I
$$
 
and 
 
$$
\frac{du}{dt} = a(bv - u).
$$
 
Where 
 
+ $v$ is a dimensionless variable representing membrane potential
+ $u$ is a dimensionless variable representing membrane potential recovery
+ $a$ is a dimensionless parameter representing the time scale of $u$
+ $b$ is a dimensionless parameter representing the sensitivity of $u$ to subthreshold fluctuations in $v$
+ $c$ is a dimensionless parameter representing the after-spike reset value of $v$ (caused by fast high-threshold $K^{+}$ conductance)
+ $d$ is a dimensionless parameter representing the after-spike reset value of $u$ (caused by slow high-threshold $Na^{+}$ and $K^{+}$ conductances)
+ $I$ is an externally applied current
 
Setting the parameters $a = 0.02$, $b = 0.2$, $c = -65$, and $d = 2$, we will generate a fast-spiking neuron simulation.
 

{% highlight r %}
#SPECIFY PARAMETERS
nsteps     = 500 #number of time steps over which to integrate
a          = 0.02
b          = 0.2
c          = -65
d          = 2
I          = matrix(0, nrow = nsteps, ncol = 1)
I[100:400] = 6
v_res      = 30
v          = matrix(NA, nrow = nsteps, ncol = 1)
u          = matrix(NA, nrow = nsteps, ncol = 1)
v[1]       = -70
u[1]       = b * v[1]
  
 
#IZHIKEVICH NEURON FUNCTION
Izhikevich = function(v, u, a, b, c, d, I, v_res) {
  
  if (v >= v_res) {
    v = c
    u = u + d
  } else {
    dv = (0.04 * (v^2)) + (5*v) + 140 - u + I
    v = v + dv
    
    du = a * ((b*v)-u)
    u = u + du
  }
  
  
  
  return(c(v, u))
}
 
#INTEGRATION VIA THE EULER METHOD
for (i in 1:(nsteps-1)) {
  res = Izhikevich(v[i], u[i], a, b, c, d, I[i], v_res)
  
  v[i+1] = min(res[1], v_res)
  u[i+1] = res[2]
    
}
 
#PLOT
plot(seq(1, nsteps, 1), v, 
     type = 'l',
     main = "Izhikevich Neuron",
     xlab = "Time (ms)",
     ylab = "Membrane Potential (mV)")
{% endhighlight %}

![plot of chunk unnamed-chunk-1](/figures/unnamed-chunk-1-1.png) 
 
Of particular interest is that other neuron types can be simulated by adjusting the model parameters. For instance, setting the parameters $a = 0.02$, $b = 0.2$, $c = -50$, and $d = 2$ will generate a chattering neuron.
 
![plot of chunk unnamed-chunk-2](/figures/unnamed-chunk-2-1.png) 
 
Parameters for further neuron types may be found at Eugene Izhikevich's page (link above).
