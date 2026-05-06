# import statements

print("Starting imports")
import dimod
from dwave.system import LeapHybridCQMSampler, LeapHybridNLSampler
from dwave.optimization import Model
import numpy as np
import time as time

print("Defining NL model")
def sbrp_assignment_nl():

    # Problem sizes
    n_students = 12
    n_schools = 3

    # Heterogeneous cab capacities
    capacities = np.array([5,3,5,3, 2,4,2,2, 4,3,2,2, 2,5,5,3], dtype=int)
    n_cabs = len(capacities)

    # Data -------------------
    # School for each student (values in {0,1,2})
    school_of_student = np.array([
        0, 0, 0, 0,   # students 0-3 -> school 0
        1, 1, 1, 1,   # students 4-7 -> school 1
        2, 2, 2, 2    # students 8-11 -> school 2
    ], dtype=int)

    # Pickup time slots for each student (0=8:00, 1=8:15, 2=8:30)
    pickup_time = np.array([
        0, 1, 2, 0,
        1, 2, 0, 1,
        2, 0, 1, 2
    ], dtype=int)

    # Pairwise time difference |t_i - t_j|
    time_diff = np.abs(pickup_time[:, None] - pickup_time[None, :])

    # Build model
    model = Model()
    time_diff_const = model.constant(time_diff)

    # Decision variables -------------------
    # x[i, k] = 1 if student i is assigned to cab k
    x = model.binary((n_students, n_cabs))

    # y[k, s] = 1 if cab k serves school s
    y = model.binary((n_cabs, n_schools))

    # Constraints -------------------------
    # 1. Each student assigned to exactly one cab
    for i in range(n_students):
        model.add_constraint(x[i, :].sum() == 1)

    # 2. Cab capacity (heterogeneous)
    for k in range(n_cabs):
        model.add_constraint(x[:, k].sum() <= capacities[k])

    # 3. Each cab serves at most one school
    for k in range(n_cabs):
        model.add_constraint(y[k, :].sum() <= 1)

    # 4. Student’s school must match cab’s school:
    #    x[i,k] <= y[k, school_of_student[i]]
    for i in range(n_students):
        s = school_of_student[i]
        for k in range(n_cabs):
            model.add_constraint(x[i, k] <= y[k, s])

    # Objective: time mismatch + capacity wastage -------------------------
    print("Now calculating objectives")

    # Time penalty: penalize mixing very different pickup times in same cab
    time_penalty = 0
    for k in range(n_cabs):
        for i in range(n_students):
            for j in range(n_students):
                time_penalty += 0.1 * time_diff_const[i, j] * x[i, k] * x[j, k]

    # Wastage penalty: unused seats in cabs
    waste_penalty = 0
    for k in range(n_cabs):
        waste_penalty += (capacities[k] - x[:, k].sum())**2

    use_penalty = 0
    for k in range(n_cabs):
        use_penalty += y[k, :].sum()

    alpha = 0.5
    beta = 0.12

    model.minimize(time_penalty + alpha * waste_penalty + beta * use_penalty)

    # Solve with NL hybrid solver -------------------------
    sampler = LeapHybridNLSampler()
    sampler.sample(model, time_limit=5)

    model.lock()

    # Extract best feasible state
    best_state = None
    best_obj = float("inf")

    for state_index in range(model.states.size()):
        if model.feasible(state_index):
            obj_val = model.objective.state(state_index)
            if obj_val < best_obj:
                best_obj = obj_val
                best_state = state_index

    if best_state is None:
        print("No feasible assignment found")
        return None, None

    x_state = x.state(best_state)
    y_state = y.state(best_state)

    print(f"Optimal feasible assignment found (state {best_state}) with objective {best_obj}")

    # Print only used cabs
    print("\n=== USED CABS ===")
    for k in range(n_cabs):
        assigned_students = [i for i in range(n_students) if x_state[i, k] > 0.5]
        if len(assigned_students) == 0:
            continue

        school = None
        for s in range(n_schools):
            if y_state[k, s] > 0.5:
                school = s
                break

        print(f"Cab {k}: school {school}, students {assigned_students}")

    return x_state, y_state


def print_assignments(x_state, y_state, n_students=12, n_cabs=16, n_schools=3):
    print("\n=== STUDENT → CAB → SCHOOL ASSIGNMENTS ===")
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

        print(f"student {i}, cab {cab}, school {school}")


# MAIN BLOCK -------------------------
if __name__ == "__main__":
    x_state, y_state = sbrp_assignment_nl()

    if x_state is not None:
        print_assignments(x_state, y_state)
