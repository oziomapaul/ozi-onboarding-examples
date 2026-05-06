print("Starting imports")
import numpy as np
from dwave.system import LeapHybridNLSampler
from dwave.optimization import Model

print("Defining NL model (disjoint set formulation)")

def sbrp_disjoint_sets():

    # Data
    # Problem sizes
    n_students = 12
    n_schools = 3

    # Cab capacities
    capacities = np.array([2, 3, 1, 2, 5, 4, 2, 3], dtype=int)
    n_cabs = len(capacities)

    # School for each student
    school_of_student = np.array([
        0, 0, 0, 0,
        1, 1, 1, 1,
        2, 2, 2, 2
    ], dtype=int)

    # Pickup times (for printing only)
    pickup_time = np.array([
        0, 1, 2, 0,
        1, 2, 0, 1,
        2, 0, 1, 2
    ], dtype=int)

    # Build model
    model = Model()

    # Constants
    school_const = model.constant(school_of_student)

    # Decision variables
    # 1. What student(s) go into what cab?
    # cab_sets[k][i] = 1 if student i is assigned to cab k, else 0
    #
    # The disjoint-set variable guarantees that a student cannot appear
    # in more than one cab.
    cab_partition, cab_sets = model.disjoint_bit_sets(
        primary_set_size=n_students,
        num_disjoint_sets=n_cabs
    )

    # 2. What cab goes to what school?; cab_school[k] ∈ {0, ..., n_schools-1}
    cab_school = model.integer(n_cabs, 0, n_schools - 1)

    # 3. Is a cab used or not
    used = model.binary(n_cabs)

    # Helper: Assigned count per cab
    # Since cab_sets[k] is a bit-set, summing it gives the number of students in cab k
    assigned_count = [
        cab_sets[k].sum()
        for k in range(n_cabs)
    ]

    print("Adding constraints to model...")
    # Constraint 1: Capacity
    for k in range(n_cabs):
        model.add_constraint(assigned_count[k] <= capacities[k])

    # Constraint 2: Every student must be assigned to exactly one cab
    # Since disjoint_bit_sets guarantees "at most one cab", this constraint forces "all students assigned".
    total_assigned = sum(assigned_count[k] for k in range(n_cabs))
    model.add_constraint(total_assigned == n_students)

    # Constraint 3: Link used[k] to assignments; to enforce capacities
    for k in range(n_cabs):
        model.add_constraint(assigned_count[k] <= capacities[k] * used[k])
        model.add_constraint(used[k] <= assigned_count[k])

    # Constraint 4: School matching
    # cab_sets[k][i] is 1 only when student i is assigned to cab k.
    # So this expression penalizes only assigned students whose school does not match.
    for k in range(n_cabs):
        school_mismatch = cab_sets[k] * ((school_const - cab_school[k]) ** 2)
        model.add_constraint(school_mismatch.sum() == 0)

    print("Calculating objective function")
    # Objective:
    use_penalty = used.sum()

    leftover_penalty = sum(
        (capacities[k] - assigned_count[k]) * used[k]
        for k in range(n_cabs)
    )

    alpha = 0.1   # leftover seats (secondary)
    beta = 10.0   # number of cabs (primary)

    model.minimize(beta * use_penalty + alpha * leftover_penalty)

    # Solve
    print("Minimizing model")
    sampler = LeapHybridNLSampler()
    sampler.sample(model, time_limit=5)
    model.lock()

    # Extract best feasible state
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

    cab_school_state = cab_school.state(best_state)
    used_state = used.state(best_state)

    print(f"Optimal feasible assignment found (state {best_state}) with objective {best_obj}")
    print(f"Total nodes: {model.num_nodes():,}, "
          f"State size: {round(model.state_size() / 1e9, 3)} GB")

    # Convert cab bit-sets back into student → cab assignment for printing
    student_cab_state = np.full(n_students, -1, dtype=int)

    print("\n=== USED CABS ===")
    for k in range(n_cabs):
        cab_bits = cab_sets[k].state(best_state)

        assigned = [
            i for i in range(n_students)
            if round(cab_bits[i]) == 1
        ]

        for i in assigned:
            student_cab_state[i] = k

        if used_state[k] < 0.5:
            continue

        assigned.sort(key=lambda i: pickup_time[i])
        school = int(round(cab_school_state[k]))

        print(f"Cab {k} ({capacities[k]} seats): school {school}, students {assigned}")

    return student_cab_state, cab_school_state, pickup_time, capacities


def print_assignments(student_cab_state, cab_school_state, pickup_time, capacities,
                      n_students=12, n_cabs=8, n_schools=3):
    print("\n=== STUDENT → CAB → SCHOOL ASSIGNMENTS ===")
    for i in range(n_students):
        cab = int(round(student_cab_state[i]))
        school = int(round(cab_school_state[cab]))
        print(
            f"student {i}, cab {cab} ({capacities[cab]} seats), "
            f"school {school}, pickup_time {pickup_time[i]}"
        )


# MAIN
if __name__ == "__main__":
    student_cab_state, cab_school_state, pickup_time, capacities = sbrp_disjoint_sets()

    if student_cab_state is not None:
        print_assignments(student_cab_state, cab_school_state, pickup_time, capacities)