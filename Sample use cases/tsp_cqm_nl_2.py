# import statements
print("Starting imports")
import dimod
from dwave.system import LeapHybridCQMSampler, LeapHybridNLSampler
from concurrent.futures import ThreadPoolExecutor, as_completed
from dwave.optimization import Model
import numpy as np
import time as time

print("Starting Solvers")
# CQM TSP -----------------------------
print("Starting CQM Program")

def tsp_cqm(dist):
    n = len(dist)
    cqm = dimod.CQM()

    # y[i][j] = 1 if tour goes from i to j
    y = [[dimod.Binary(f"y_{i}_{j}") for j in range(n)] for i in range(n)]

    # MTZ order variables
    u = [dimod.Integer(f"u_{i}", lower_bound=1, upper_bound=n) for i in range(n)]

    # No self-loops
    for i in range(n):
        cqm.add_constraint(y[i][i] == 0)

    # Outgoing edges
    for i in range(n):
        cqm.add_constraint(sum(y[i][j] for j in range(n)) == 1)

    # Incoming edges
    for j in range(n):
        cqm.add_constraint(sum(y[i][j] for i in range(n)) == 1)

    # Fix root
    cqm.add_constraint(u[0] == 1)

    # MTZ subtour elimination
    for i in range(1, n):
        for j in range(1, n):
            if i != j:
                cqm.add_constraint(u[i] - u[j] + n * y[i][j] <= n - 1)

    # Objective
    obj = 0
    for i in range(n):
        for j in range(n):
            obj += dist[i][j] * y[i][j]

    # Strong integrality penalties
    for i in range(n):
        obj += 1000 * (1 - sum(y[i][j] for j in range(n)))**2
    for j in range(n):
        obj += 1000 * (1 - sum(y[i][j] for i in range(n)))**2

    cqm.set_objective(obj)

    sampler = LeapHybridCQMSampler()
    sampleset = sampler.sample_cqm(cqm, time_limit=5)
    sol = sampleset.first.sample

    # Reconstruct tour
    successor = {}
    for i in range(n):
        for j in range(n):
            if sol[f"y_{i}_{j}"] > 0.5:
                successor[i] = j

    # If any city has no successor → invalid solution
    if len(successor) != n:
        print("CQM returned invalid successor map:", successor)
        return None

    # Build tour
    tour = [0]
    current = 0
    for _ in range(n - 1):
        current = successor[current]
        tour.append(current)

    print("CQM tour:", tour)
    print("CQM objective (distance):", tour_length(tour, dist))
    return tour

# NL TSP -----------------------------
print("Starting NL Program")
def tsp_nl(dist):
    n = len(dist)
    model = Model()

    D = model.constant(np.array(dist))

    # Path is a list of length n containing city indices
    path = model.list(n, min_size=n, max_size=n)

    # All cities must appear exactly once
    for city in range(n):
        model.add_constraint((path[:] == city).sum() == 1)

    # Objective: sum of distances along the path
    model.minimize((D[path[:-1], path[1:]]).sum() + D[path[-1], path[0]])

    sampler = LeapHybridNLSampler()
    sampler.sample(model, time_limit=5)

    model.lock()
    for i in range(model.states.size()):
        if model.feasible(i):
            tour = list(map(int, path.state(i)))
            print("NL tour:", tour)
            print("NL objective (distance):", tour_length(tour, dist))
            return tour

    print("No feasible NL solution")
    return None

# Helper: compute tour length -----------------------------
def tour_length(tour, dist):
    total = 0
    n = len(tour)
    for i in range(n):
        j = (i + 1) % n
        total += dist[tour[i]][tour[j]]
    return total

# Helper: run solver with timer tracker + print tour -----------------------------
def run_solver_with_timing(name, solver_fn, dist):
    start = time.time()
    tour = solver_fn(dist)
    runtime = time.time() - start

    if tour is None:
        return {
            "name": name,
            "tour": None,
            "closed": None,
            "runtime": runtime
        }

    # Rotate so tour starts at 0 (optional)
    if 0 in tour:
        idx = tour.index(0)
        tour = tour[idx:] + tour[:idx]

    closed = tour + [tour[0]]

    return {
        "name": name,
        "tour": tour,
        "closed": closed,
        "runtime": runtime
    }


# Helper: run all formulations -----------------------------
def run_all(dist):
    runners = [
        ("CQM", tsp_cqm),
        ("NL", tsp_nl)
    ]

    results = []
    with ThreadPoolExecutor() as exec:
        futures = [
            exec.submit(run_solver_with_timing, name, fn, dist)
            for name, fn in runners
        ]
        for f in as_completed(futures):
            results.append(f.result())

    return results

# Helper: print all results -----------------------------
def print_results(results):
    for r in results:
        print(f"{r['name']} closed tour: {r['closed']}, ran in {r['runtime']:.4f} seconds")

# Main -----------------------------
if __name__ == "__main__":
    dist = [
        [0, 4, 7, 9, 3, 8, 6, 5, 9, 7, 14, 6],
        [4, 0, 5, 6, 2, 7, 18, 6, 5, 9, 3, 7],
        [7, 5, 0, 14, 6, 3, 5, 8, 7, 6, 5, 4],
        [9, 6, 4, 0, 5, 2, 4, 7, 8, 5, 16, 3],
        [3, 2, 6, 5, 0, 4, 17, 6, 3, 8, 2, 5],
        [8, 7, 3, 2, 4, 0, 3, 5, 7, 4, 6, 12],
        [6, 8, 5, 4, 7, 3, 0, 12, 6, 3, 5, 4],
        [5, 6, 8, 7, 6, 15, 2, 0, 4, 2, 6, 3],
        [9, 15, 7, 8, 3, 7, 6, 4, 0, 5, 3, 6],
        [7, 19, 6, 5, 8, 4, 3, 2, 5, 0, 4, 3],
        [4, 3, 5, 6, 2, 6, 5, 6, 13, 4, 0, 5],
        [6, 17, 4, 3, 5, 2, 4, 3, 6, 3, 5, 0]
    ]

    results = run_all(dist)
    print_results(results)