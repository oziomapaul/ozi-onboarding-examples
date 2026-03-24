print("Starting!")

#step 1: import generator
from dwave.optimization.generators import flow_shop_scheduling

#step 2: define parameter (processing times of 2 jobs on 3 machines)
processing_times = [[10, 5, 7], [20, 10, 15]]

#step 3: create an object of the model using the fss generator and the defined parameter
model = flow_shop_scheduling(processing_times=processing_times)

#step 4: import and define the stride solver
from dwave.system import LeapHybridNLSampler
sampler = LeapHybridNLSampler()

#step 5: submit the model to the selected solver.
results = sampler.sample(
    model,
    time_limit=1,
    label="Examples - Flow Shop Scheduling")

#step 6: Read results from the model
#Q: states is the assignment of a value to a symbol (so like initial solutions?) And not input parameters?
#Q: In flow_shop_scheduler.py, line 304 already does call_stride_solver, so do we need step 4?

decisions = next(model.iter_decisions())
print("Solution (decisions):", decisions.state(0))

order = decisions.state(0).astype(int)   # convert floats to ints
order_list = order.tolist()
order_str = " → ".join([f"Job {j}" for j in order_list])
print(f"The solution (order of jobs on the machines) should be: {order_str}")

#print(results.result())
print("End!")