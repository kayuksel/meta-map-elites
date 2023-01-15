import subprocess

print('Running Meta Map Elites 11 times on Schwefel-30 ...')
results = []
for i in range(11):
    result = subprocess.run(["python", "meta_map_elites.py"], capture_output=True)
    result = result.stdout.decode()
    print(result)
    results.append(float(result))

results.sort()
median = results[len(results)//2]

print("The median loss of Meta Map Elites:", median)

import numpy, torch
import nevergrad as ng
        
def schwefel(x):
    x = x.tanh() * 500
    return 418.9829 * x.shape[1] - (x * x.abs().sqrt().sin()).sum(dim=1)

def schwefel_f(x):
    x = torch.from_numpy(x).cuda().float().unsqueeze(0)
    return schwefel(x).item()

print('Running Nevergrad NGOpt4 11 times on Schwefel-30')
results = []
for i in range(11):
    optimizer = ng.optimizers.NGOpt4(parametrization=30, budget=10000)
    recommendation = optimizer.minimize(schwefel_f)
    result = schwefel_f(recommendation.value)
    print(result)
    results.append(result)

median = numpy.median(numpy.array(results))

print("The median loss of Nevergrad NGOp4:", median)
