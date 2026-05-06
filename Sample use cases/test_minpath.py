# import statements

import dimod
from dwave.system import LeapHybridCQMSampler, LeapHybridNLSampler
from dwave.optimization import Model
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
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

def solve_list_NL(G_d, v_start, v_end, time_limit=5):
    model = Model()

    A = model.constant(nx.to_numpy_array(G_d))
    path = model.list(len(G_d), min_size=2)
    model.add_constraint(A[path[:-1], path[1:]].all())
    model.add_constraint((path[:] == v_start).any())
    model.add_constraint((path[:] == v_end).any())
    model.minimize(path.size())
    
    sampler = LeapHybridNLSampler()
    sampler.sample(
        model=model,
        time_limit=time_limit,
        label="list-NL"
    )

    # get (one of the) best feasible solution found
    model.lock()  
    try:  
        solution = next(path.state(i) for i in range(model.states.size()) if model.feasible(i))
    except StopIteration:
        return None
    
    # convert list of vertex indices into a list of edge tuples (v1, v2) using node labels in the graph
    nodes = list(G_d.nodes)
    return [(nodes[int(i)],nodes[int(j)]) for i,j in zip(solution, solution[1:])]


if __name__ == "__main__":
    G0 = nx.hexagonal_lattice_graph(5, 5, with_positions=True)
    G = G0.to_directed()
    pos = nx.get_node_attributes(G, 'pos')  # can use nx.spring_layout(G) if no positions
    v_start = 0
    v_end = G.number_of_nodes()-1

    runners = (solve_binary_CQM, solve_binary_NL, solve_list_NL)
    params = (G, v_start, v_end)

    if PARALLEL:
        with ThreadPoolExecutor() as exec:
            futures = [
                exec.submit(r, *params, time_limit=TIME_LIMIT) for r in runners
            ]
            solutions = [
                f.result() for f in as_completed(futures)
            ]
    else:
        solutions = [r(*params, time_limit=TIME_LIMIT) for r in runners]
    
    # Build a comparison table to show and compare results
    runner_names = ["binary_CQM", "binary_NL", "binary_NL_List"]
    edge_counts = [len(sol) if sol is not None else None for sol in solutions]

    print("\n=== Solver Comparison ===")
    print(f"{'Model':<15} | {'#Edges'}")
    print("-" * 26)
    for name, count in zip(runner_names, edge_counts):
        print(f"{name:<15} | {count}")
    print()

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

print("Hello, ending now!")
