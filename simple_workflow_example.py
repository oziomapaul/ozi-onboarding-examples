from dimod import ConstrainedQuadraticModel, Integer
from dwave.system import LeapHybridCQMSampler
from dwave.optimization import Model

#define two integers and a cqm model
i = Integer('i', upper_bound=4)
j = Integer('j', upper_bound=4)
cqm = ConstrainedQuadraticModel()

#set objective: 
cqm.set_objective(-i*j)

#set constraints:
cqm.add_constraint(2*i+2*j <= 8, "Max perimeter")

#solve or sample?
with LeapHybridCQMSampler() as sampler:
    sampleset = sampler.sample_cqm(cqm)  
    feasible_results = sampleset.filter(lambda d: d.is_feasible)
    print(feasible_results.first)

print("I got here safely!")

print("_____________________________NL Model______________________________")
#simple nl model
model = Model() #create an object of Model class
i = model.integer(lower_bound=-5, upper_bound=5) #create an integer
c = model.constant(4) #create a constant
y = i**2 - c*i #define your objective function, here as y = i squared - c times i
model.minimize(y) #solve your model
