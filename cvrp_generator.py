#input
demand = [0, 34, 12, 65, 10, 43, 27, 55, 61, 22]
sites = [(15, 38), (23, -19), (44, 62), (3, 12), (-56, -21), (-53, 2),
         (33, 63), (14, -33), (42, 41), (13, -62)]

#import generator to create CVRP model (object of generator)
from dwave.optimization.generators import capacitated_vehicle_routing
model = capacitated_vehicle_routing(
    demand=demand,
    number_of_vehicles=2,
    vehicle_capacity=200,
    locations_x=[x for x,y in sites],
    locations_y=[y for x,y in sites])

#import sampler to solve problem
from dwave.system import LeapHybridNLSampler
sampler = LeapHybridNLSampler()

#Submit the model to the selected solver.
results = sampler.sample(
    model,
    time_limit=10)

print(results.result().info['timing']['charge_time'])

#print results
num_samples = model.states.size()
route, = model.iter_decisions()
route1, route2 = route.iter_successors()

for i in range(min(3, num_samples)):
    print(f"Objective value {int(model.objective.state(i))} for \n" \
    f"\t Route 1: {route1.state(i)} \t Route 2: {route2.state(i)} \n" \
    f"\t Feasible: {all(sym.state(i) for sym in model.iter_constraints())}")

model.states.resize(1)
route.set_state(0, [route1.state(0), route2.state(0)])
results = sampler.sample(model)

print("_______Results after sampling________")
for i in range(min(3, num_samples)):
    print(f"Objective value {int(model.objective.state(i))} for \n" \
    f"\t Route 1: {route1.state(i)} \t Route 2: {route2.state(i)} \n" \
    f"\t Feasible: {all(sym.state(i) for sym in model.iter_constraints())}")
    
print("Hello!")