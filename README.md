# Meta-Learning Map-Elites Mutations

Quality-Diversity optimization is a global optimization method that aims to find a   
diverse set of high-performing solutions to a given problem, rather than just one.

In this work, a meta-learning method for quality-diversity optimization is proposed  
by learning the mutation operator of map-elites via a convolutional neural network.  

Experimental results shown its effectiveness in first-order non-convex optimization.  
The convergence speed is remarkable versus SotA black-box optimization methods.

# Comparison with Nevergrad NGOpt-4

Running Meta Map Elites 11 times on Schwefel-30 ...  
The median score of Meta Map Elites: **1.4462890625**

Running Nevergrad NGOpt4 11 times on Schwefel-30 ...  
The median score of Nevergrad NGOpt4: 889.81640625

Run **compare_opts.py** to reproduce the above results.  
P.S. The maximum amount of function evaluations is 10K.

# Large-Scale Optimization Experiments
Check **meta_map_elites_10k.py** to try with 1000-dims.  
If you want to optimize 100K dims, check my other work:  
https://github.com/kayuksel/genmeta-vs-nevergrad
