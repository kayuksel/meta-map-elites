# Meta-Learning Map-Elites Mutations

In this work, a meta-learning method for quality-diversity optimization is proposed  
by learning the mutation operator of map-elites via a convolutional neural network.  

Experimental results shown its effectiveness in first-order non-convex optimization.  
The convergence speed is remarkable versus SotA black-box optimization methods.

# Comparison with Nevergrad NGOpt

Running Meta Map Elites 11 times on Schwefel-30 ...  
The median score of Meta Map Elites: **1.4462890625**

Running Nevergrad NGOpt4 11 times on Schwefel-30 ...  
The median score of Nevergrad NGOp4: 889.81640625

Run **compare_opts.py** to reproduce the above results.  
P.S. the maximum function evaluation amount is 10K.
