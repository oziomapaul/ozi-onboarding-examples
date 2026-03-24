from dimod.generators import random_knapsack
from dwave.system import LeapHybridCQMSampler
cqm = random_knapsack(10)
sampler = LeapHybridCQMSampler()
sampleset = sampler.sample_cqm(cqm,
                               time_limit=180,
                               label="SDK Examples - Bin Packing")
print(sampleset.first)
print("I got here on time")