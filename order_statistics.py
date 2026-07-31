"""
Order statistics of Uniform(0,1) draws: equivalences, closed forms,
sort-free generation via spacings, and extreme value asymptotics.
"""
import random
import statistics

N = 500_000


# ============================================================
# PART 1: n=2 special case -- max/min have simple closed forms
# ============================================================
def part1_pair_case():
    u = [random.random() for _ in range(N)]
    max2 = [max(random.random(), random.random()) for _ in range(N)]
    min2 = [min(random.random(), random.random()) for _ in range(N)]
    sqrt_u = [x ** 0.5 for x in u]
    one_minus_sqrt = [1 - (1 - x) ** 0.5 for x in u]

    def stats(name, data):
        print(f"{name:18} mean={statistics.fmean(data):.6f}")

    print("=== Part 1: n=2 draws (max/min have closed forms) ===")
    stats("max(U1,U2)", max2)
    stats("sqrt(U)", sqrt_u)
    stats("min(U1,U2)", min2)
    stats("1-sqrt(1-U)", one_minus_sqrt)
    print("Theory: E[max]=2/3, E[min]=1/3\n")


# ============================================================
# PART 2: general n draws, k-th order statistic ~ Beta(k, n-k+1)
# ============================================================
def kth_order_stat(n, k):
    draws = [random.random() for _ in range(n)]
    draws.sort()
    return draws[k - 1]


def part2_general_order_stat(n=7, k=3):
    direct = [kth_order_stat(n, k) for _ in range(N)]
    beta_transform = [random.betavariate(k, n - k + 1) for _ in range(N)]
    max_transform = [random.random() ** (1 / n) for _ in range(N)]      # k=n closed form
    min_transform = [1 - (1 - random.random()) ** (1 / n) for _ in range(N)]  # k=1 closed form

    print(f"=== Part 2: n={n} draws, k={k}-th order statistic ~ Beta(k, n-k+1) ===")
    print(f"Theory: E[X_(k)] = k/(n+1) = {k/(n+1):.6f}")
    print(f"  direct (sort n draws): {statistics.fmean(direct):.6f}")
    print(f"  Beta(k,n-k+1) sampler: {statistics.fmean(beta_transform):.6f}")
    print(f"Max only (k=n): theory {n/(n+1):.6f}, "
          f"U^(1/n) transform: {statistics.fmean(max_transform):.6f}")
    print(f"Min only (k=1): theory {1/(n+1):.6f}, "
          f"1-(1-U)^(1/n) transform: {statistics.fmean(min_transform):.6f}\n")


# ============================================================
# PART 3: spacings between order statistics (sort-free generation)
# ============================================================
def part3_spacings(n=6, batches=20000):
    draws_batches = [sorted(random.random() for _ in range(n)) for _ in range(batches)]
    spacings = [[0.0] * (n + 1) for _ in range(batches)]
    for b, batch in enumerate(draws_batches):
        prev = 0.0
        for i, x in enumerate(batch):
            spacings[b][i] = x - prev
            prev = x
        spacings[b][n] = 1.0 - prev

    means = [statistics.fmean(spacings[b][i] for b in range(batches)) for i in range(n + 1)]
    print(f"=== Part 3: spacings between order statistics (n={n} draws -> {n+1} gaps) ===")
    print(f"Theory: E[each spacing] = 1/(n+1) = {1/(n+1):.5f}")
    print("Empirical means:", [f"{m:.5f}" for m in means])

    # Sort-free trick: normalized Exponential(1) draws have the same joint
    # law as the spacings -- cumulative sums give order statistics directly.
    exp_spacings = []
    for _ in range(batches):
        e = [random.expovariate(1) for _ in range(n + 1)]
        s = sum(e)
        exp_spacings.append([x / s for x in e])
    exp_means = [statistics.fmean(row[i] for row in exp_spacings) for i in range(n + 1)]
    print("Normalized-Exponential trick means:", [f"{m:.5f}" for m in exp_means])
    print("(cumsum of these gives order statistics without sorting)\n")


# ============================================================
# PART 4: extreme value asymptotics as n -> infinity (fixed k)
# ============================================================
def part4_extreme_value_asymptotics(trials=20000):
    print("=== Part 4: extreme value asymptotics ===")
    print("n * min(U_1..U_n)  ->  Exponential(1)")
    for n_big in [10, 100, 1000, 10000, 100000]:
        mins = [min(random.random() for _ in range(n_big)) for _ in range(trials)]
        scaled = [n_big * m for m in mins]
        print(f"  n={n_big:6d}  E[n*min] = {statistics.fmean(scaled):.4f}  (theory -> 1.0000)")

    print("\nn * (1 - max(U_1..U_n))  ->  Exponential(1)  [by U <-> 1-U symmetry]")
    for n_big in [10, 100, 1000, 10000, 100000]:
        maxs = [max(random.random() for _ in range(n_big)) for _ in range(trials)]
        scaled = [n_big * (1 - m) for m in maxs]
        print(f"  n={n_big:6d}  E[n*(1-max)] = {statistics.fmean(scaled):.4f}  (theory -> 1.0000)")


if __name__ == "__main__":
    part1_pair_case()
    part2_general_order_stat(n=7, k=3)
    part3_spacings(n=6)
    part4_extreme_value_asymptotics()
