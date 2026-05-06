# import statements

import dimod
from dwave.system import LeapHybridCQMSampler, LeapHybridNLSampler
from dwave.optimization import Model
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

print("Hello, starting now!")

TIME_LIMIT = 30
PARALLEL = True

def solve_binary_CQM(G_d, v_start, v_end, time_limit=5):
    D = nx.incidence_matrix(G_d, oriented=True).toarray()

    cqm = dimod.CQM()
    X = dimod.BinaryArray(G_d.edges())

    # minimise total flow
    cqm.set_objective(sum(X))

    # constraint for each vertex: flow in = flow out, except at start and finish nodes
    net_flow = D @ X
    target = np.zeros(G_d.number_of_nodes())
    target[v_start] = +1
    target[v_end] = -1
    for i in range(len(target)):
        cqm.add_constraint(net_flow[i] == target[i])

    sampler = LeapHybridCQMSampler()
    sampleset = sampler.sample_cqm(
        cqm=cqm,
        time_limit=time_limit,
        label="binary-CQM"
    )

    # get (one of the) best feasible solution found
    solution = sampleset.filter(lambda d: d.is_feasible).first.sample

    return [e for e in solution if solution[e]]

def solve_binary_NL(G_d, v_start, v_end, time_limit=5):
    D = nx.incidence_matrix(G_d, oriented=True).toarray()
    
    model = Model()
    X = model.binary(G_d.number_of_edges())

    # minimise total flow
    model.minimize(sum(X))

    # constraint for each vertex: flow in = flow out, except at start and finish nodes
    net_flow = D @ X
    target = np.zeros(G_d.number_of_nodes())
    target[v_start] = +1
    target[v_end] = -1
    model.add_constraint((net_flow == model.constant(target)).all())
    
    sampler = LeapHybridNLSampler()
    sampler.sample(
        model=model,
        time_limit=time_limit,
        label="binary-NL"
    )

    # get (one of the) best feasible solution found
    model.lock()
    try:
        solution = next(X.state(i) for i in range(model.states.size()) if model.feasible(i))
    except StopIteration:
        return None
    
    return [e for i, e in enumerate(G_d.edges()) if solution[i]]

if __name__ == "__main__":
    G0 = nx.hexagonal_lattice_graph(5, 5, with_positions=True)
    G = G0.to_directed()
    pos = nx.get_node_attributes(G, 'pos')  # can use nx.spring_layout(G) if no positions
    v_start = 0
    v_end = G.number_of_nodes()-1

    NUM_RUNS = 5

    def timed_call(func, *args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        return result, end - start

    runners = (solve_binary_CQM, solve_binary_NL)
    runner_names = ["binary_CQM", "binary_NL"]
    params = (G, v_start, v_end)

    # Build job list: (model_name, solver_function)
    jobs = []
    for name, solver in zip(runner_names, runners):
        for run_id in range(NUM_RUNS):
            jobs.append((name, run_id + 1, solver))

    records = []

    # Run all jobs in parallel
    with ThreadPoolExecutor() as exec:
        futures = {
            exec.submit(timed_call, solver, *params, time_limit=TIME_LIMIT): (name, run_id)
            for (name, run_id, solver) in jobs
        }

        for fut in as_completed(futures):
            name, run_id = futures[fut]
            sol, t = fut.result()
            edge_count = len(sol) if sol is not None else None

            records.append({
                "Model": name,
                "Run": run_id,
                "Solution": sol,      # <-- added
                "Edges": edge_count,
                "Time (s)": t
            })

    # Convert to DataFrame
    df = pd.DataFrame(records)
    print("\n=== All Runs ===")
    print(df)
    print()

    # Optional: averages
    df_avg = df.groupby("Model").agg({
        "Edges": "mean",
        "Time (s)": "mean"
    }).reset_index()

    print("=== Averages Over 5 Runs ===")
    print(df_avg)
    print()

    # Rebuild solutions list: one solution per solver (first run)
    solutions = []
    for name in runner_names:
        first = next(rec for rec in records if rec["Model"] == name)
        solutions.append(first["Solution"])

    fig, axes = plt.subplots(1, len(solutions))

    for solution, ax in zip(solutions, axes):
        nx.draw(
            G, pos, ax=ax,
            node_color=['r' if i in (v_start, v_end) else 'y' for i in range(len(G))],
            node_size=[200 if i in (v_start, v_end) else 0 for i in range(len(G))],
            connectionstyle="arc3,rad=0.1"  
        )
        if solution is not None:
            nx.draw_networkx_edges(
                G, pos, ax=ax,
                edgelist=solution,
                edge_color='r',
                node_size=0,
                width=4,
                connectionstyle="arc3,rad=0.1"  
            )
        ax.axis('equal')

    plt.show()

plt.figure(figsize=(8, 5))
df.boxplot(column="Time (s)", by="Model")
plt.title("Solver Runtime Distribution")
plt.suptitle("")  # removes the automatic pandas title
plt.ylabel("Time (s)")
plt.show()

plt.figure(figsize=(8, 5))
df.boxplot(column="Edges", by="Model")
plt.title("Path Length Distribution")
plt.suptitle("")
plt.ylabel("Number of Edges")
plt.show()

print("Hello, ending now!")
