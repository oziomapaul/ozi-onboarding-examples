print("Starting imports")
import dimod
from dwave.system import LeapHybridNLSampler
from dwave.optimization import Model
import numpy as np

print("Defining NL model")

def sbrp_binary():

    # Problem sizes
    n_students = 12
    n_schools = 3

    # Heterogeneous cab capacities
    capacities = np.array([2,3,1,2,5,4,2,3], dtype=int)
    n_cabs = len(capacities)

    # School for each student
    school_of_student = np.array([
        0, 0, 0, 0,
        1, 1, 1, 1,
        2, 2, 2, 2
    ], dtype=int)

    # Pickup times of students
    pickup_time = np.array([
        0, 1, 2, 0,
        1, 2, 0, 1,
        2, 0, 1, 2
    ], dtype=int)

    # Build model
    model = Model()

    # Decision variables
    x = model.binary((n_students, n_cabs))      # student → cab (which students are in what cab?)
    y = model.binary((n_cabs, n_schools))       # cab → school (what cab goes to what school?)

    # Constraints -------------------------
    # 1. Each student assigned to exactly one cab
    for i in range(n_students):
        model.add_constraint(x[i, :].sum() == 1)

    # 2. Cab capacity
    for k in range(n_cabs):
        model.add_constraint(x[:, k].sum() <= capacities[k])

    # 3. Each cab serves at most one school
    for k in range(n_cabs):
        model.add_constraint(y[k, :].sum() <= 1)

    # 4. Student’s school must match cab’s school
    for i in range(n_students):
        s = school_of_student[i]
        for k in range(n_cabs):
            model.add_constraint(x[i, k] <= y[k, s])

    # Objective: Minimise number of cabs used
    print("Building objective function")
    leftover_penalty = 0
    for k in range(n_cabs):
        leftover_penalty += capacities[k] - x[:, k].sum()

    use_penalty = y.sum()

    alpha = 0.1     # leftover seats (secondary)
    beta  = 10.0    # number of cabs (primary)

    model.minimize(beta * use_penalty + alpha * leftover_penalty)

    #model.minimize(y.sum())

    # Solve
    print("Minimizing model")
    sampler = LeapHybridNLSampler()
    sampler.sample(model, time_limit=5)
    model.lock()

    # Extract best feasible state
    print("Printing results: ")
    best_state = None
    best_obj = float("inf")

    for s in range(model.states.size()):
        if model.feasible(s):
            obj = model.objective.state(s)
            if obj < best_obj:
                best_obj = obj
                best_state = s

    if best_state is None:
        print("No feasible assignment found")
        return None, None, pickup_time, capacities

    x_state = x.state(best_state)
    y_state = y.state(best_state)

    print(f"Optimal feasible assignment found (state {best_state}) with objective {best_obj}")
    print(f"Total number of nodes {model.num_nodes():,}, "
          + f"total state size {round(model.state_size() / 10**9, 3)}GB")
    
    # Print only used cabs
    print("\n=== USED CABS ===")
    for k in range(n_cabs):
        assigned = [i for i in range(n_students) if x_state[i, k] > 0.5]
        if len(assigned) == 0:
            continue

        assigned.sort(key=lambda i: pickup_time[i])

        school = None
        for s in range(n_schools):
            if y_state[k, s] > 0.5:
                school = s
                break

        print(f"Cab {k} ({capacities[k]} seats): school {school}, students {assigned}")

    return x_state, y_state, pickup_time, capacities

def print_assignments(x_state, y_state, pickup_time, capacities,
                      n_students=12, n_cabs=16, n_schools=3):
    print("\n=== STUDENT → CAB → SCHOOL ASSIGNMENTS (ordered by pickup time) ===")
    for i in range(n_students):
        cab = None
        for k in range(n_cabs):
            if x_state[i, k] > 0.5:
                cab = k
                break

        school = None
        if cab is not None:
            for s in range(n_schools):
                if y_state[cab, s] > 0.5:
                    school = s
                    break

        print(
            f"student {i}, cab {cab} ({capacities[cab] if cab is not None else '-'} seats), "
            f"school {school}, pickup_time {pickup_time[i]}"
        )


# MAIN BLOCK -------------------------
if __name__ == "__main__":
    x_state, y_state, pickup_time, capacities = sbrp_binary()

    if x_state is not None:
        print_assignments(x_state, y_state, pickup_time, capacities)

    