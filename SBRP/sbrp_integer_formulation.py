print("Starting imports")
import numpy as np
from dwave.system import LeapHybridNLSampler
from dwave.optimization import Model

print("Defining optimized NL model")

def sbrp_integer():

    # Data
    # Problem sizes
    n_students = 12
    n_schools = 3

    capacities = np.array([2, 3, 1, 2, 5, 4, 2, 3], dtype=int)
    n_cabs = len(capacities)

    school_of_student = np.array([
        0, 0, 0, 0,
        1, 1, 1, 1,
        2, 2, 2, 2
    ], dtype=int)

    pickup_time = np.array([
        0, 1, 2, 0,
        1, 2, 0, 1,
        2, 0, 1, 2
    ], dtype=int)

    # Build model
    model = Model()

    # Decision variables
    # 1. What student(s) go into what cab?; student_cab[i] ∈ {0, ..., n_cabs-1}
    student_cab = model.integer(n_students, 0, n_cabs - 1)

    # 2. What cab goes to what school?; cab_school[k] ∈ {0, ..., n_schools-1}
    cab_school = model.integer(n_cabs, 0, n_schools - 1)
    
    # 3. Is a cab used or not
    used = model.binary(n_cabs)
    
    # Constraints: Adding constraints to model
    print("Adding constraints...")

    assigned_count = [
        (student_cab == k).sum()
        for k in range(n_cabs)
    ]

    # Constraint 1: Capacity
    for k in range(n_cabs):
        model.add_constraint(assigned_count[k] <= capacities[k])

    # Constraint 2: Link used[k] to assignments
    for k in range(n_cabs):
        model.add_constraint(assigned_count[k] <= capacities[k] * used[k])

    # Constraint 3: School matching
    # If student i is in cab k, cab_school[k] == school_of_student[i]
    # (student_cab[i] == k) * (cab_school[k] - s) == 0
    school_const = model.constant(school_of_student)

    school_match = (cab_school[student_cab] == school_const)

    model.add_constraint(school_match.sum() == n_students)

    # Objective:
    #   β * (# cabs used) + α * (total leftover seats)
    alpha = 0.1
    beta = 10.0

    use_penalty = used.sum()

    leftover_penalty = sum(
        (capacities[k] - assigned_count[k]) * used[k]
        for k in range(n_cabs)
    )

    model.minimize(beta * use_penalty + alpha * leftover_penalty)

    # Solve
    print("Minimizing model")
    sampler = LeapHybridNLSampler()
    sampler.sample(model, time_limit=5)

    model.lock()

    # Extract best feasible
    print("Printing results:")

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

    student_cab_state = student_cab.state(best_state)
    cab_school_state = cab_school.state(best_state)
    used_state = used.state(best_state)

    print(f"\nBest state {best_state}, objective {best_obj}")
    print(f"Nodes: {model.num_nodes():,}")
    print(f"State size: {round(model.state_size()/1e9, 3)} GB")

    # Print results: used cabs
    print("\n=== USED CABS ===")
    for k in range(n_cabs):
        if used_state[k] < 0.5:
            continue

        assigned = [
            i for i in range(n_students)
            if int(round(student_cab_state[i])) == k
        ]

        assigned.sort(key=lambda i: pickup_time[i])
        school = int(round(cab_school_state[k]))

        print(f"Cab {k} ({capacities[k]} seats): school {school}, students {assigned}")

    return student_cab_state, cab_school_state, pickup_time, capacities


def print_assignments(student_cab_state, cab_school_state, pickup_time, capacities):

    print("\n=== STUDENT → CAB → SCHOOL ===")

    for i in range(len(student_cab_state)):
        cab = int(round(student_cab_state[i]))
        school = int(round(cab_school_state[cab]))

        print(
            f"student {i} → cab {cab} ({capacities[cab]} seats) → "
            f"school {school}, pickup_time {pickup_time[i]}"
        )


# MAIN function
if __name__ == "__main__":

    student_cab_state, cab_school_state, pickup_time, capacities = sbrp_integer()

    if student_cab_state is not None:
        print_assignments(student_cab_state, cab_school_state, pickup_time, capacities)
