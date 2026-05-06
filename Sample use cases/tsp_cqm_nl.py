# import statements
print("Starting imports")
import dimod
from dwave.system import LeapHybridCQMSampler, LeapHybridNLSampler
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
    print_closed_tour(tour)
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
            print_closed_tour(tour)
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

# Helper: print tour -----------------------------
def print_closed_tour(tour):
    print("Closed tour:", tour + [tour[0]])

# Main -----------------------------
if __name__ == "__main__":
    dist = [
        [0, 2, 9, 10],
        [1, 0, 6, 4],
        [15, 7, 0, 8],
        [6, 3, 12, 0]
    ]

    tsp_cqm(dist)
    tsp_nl(dist)
