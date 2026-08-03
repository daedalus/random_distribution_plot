import hashlib
import random
import struct
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

N1 = 10_000
N2 = 100_000
N3 = 50_000

# ========== generate all data ==========

# 12 simple distributions
simple = {
    "random()": [random.random() for _ in range(N1)],
    "uniform(0, 10)": [random.uniform(0, 10) for _ in range(N1)],
    "gauss(0, 1)": [random.gauss(0, 1) for _ in range(N1)],
    "expovariate(1)": [random.expovariate(1) for _ in range(N1)],
    "betavariate(2, 5)": [random.betavariate(2, 5) for _ in range(N1)],
    "paretovariate(3)": [random.paretovariate(3) for _ in range(N1)],
    "triangular(0, 10, 5)": [random.triangular(0, 10, 5) for _ in range(N1)],
    "randint(1, 6)": [random.randint(1, 6) for _ in range(N1)],
    "binomialvariate(10, 0.5)": [random.binomialvariate(10, 0.5) for _ in range(N1)],
    "gammavariate(2, 1)": [random.gammavariate(2, 1) for _ in range(N1)],
    "lognormvariate(0, 0.5)": [random.lognormvariate(0, 0.5) for _ in range(N1)],
    "vonmisesvariate(0, 3)": [random.vonmisesvariate(0, 3) for _ in range(N1)],
}

# -- existing identities --
max_of_two   = [max(random.random(), random.random()) for _ in range(N2)]
sqrt_rand    = [random.random() ** 0.5 for _ in range(N2)]
min_of_two   = [min(random.random(), random.random()) for _ in range(N2)]
one_minus_sqrt = [1 - random.random() ** 0.5 for _ in range(N2)]

inner_point = []
for _ in range(N2):
    a, b = random.random(), random.random()
    lo, hi = min(a, b), max(a, b)
    inner_point.append(random.random() * (hi - lo) + lo)
beta22 = [random.betavariate(2, 2) for _ in range(N2)]

minus_log_u = [-np.log(random.random()) for _ in range(N2)]
expo        = [random.expovariate(1) for _ in range(N2)]

max_of_three   = [max(random.random(), random.random(), random.random()) for _ in range(N2)]
cube_root      = [random.random() ** (1/3) for _ in range(N2)]
min_of_three   = [min(random.random(), random.random(), random.random()) for _ in range(N2)]
one_minus_cube = [1 - random.random() ** (1/3) for _ in range(N2)]

sum_of_two = [random.random() + random.random() for _ in range(N2)]

# -- from order_statistics.py --
order53_direct = []
order53_beta   = []
for _ in range(N3):
    draws = sorted(random.random() for _ in range(5))
    order53_direct.append(draws[2])
    order53_beta.append(random.betavariate(3, 3))

n_spacings = 5
spacings = [[] for _ in range(n_spacings + 1)]
for _ in range(N3):
    draws = sorted(random.random() for _ in range(n_spacings))
    prev = 0.0
    for i in range(n_spacings):
        spacings[i].append(draws[i] - prev)
        prev = draws[i]
    spacings[n_spacings].append(1.0 - prev)

extreme_panels = {}
for n in [10, 100]:
    extreme_panels[n] = [n * min(random.random() for _ in range(n)) for _ in range(N3)]

# -- new: gamma from summed exponentials --
# sum_{i=1..k} -ln(U_i)  =  Gamma(k, 1)
gamma_sum = {}
for k in [2, 3]:
    ones = []
    for _ in range(N3):
        s = sum(-np.log(random.random()) for _ in range(k))
        ones.append(s)
    gamma_sum[k] = ones
gamma_direct = {k: [random.gammavariate(k, 1) for _ in range(N3)] for k in [2, 3]}

# -- new: CLT convergence — Irwin-Hall(12) vs Normal(6, 1) --
irwin_hall_12 = [sum(random.random() for _ in range(12)) for _ in range(N3)]

# ========== build figure (7 rows x 4 cols = 28 panels) ==========

fig, axes = plt.subplots(7, 4, figsize=(16, 22))
axs = axes.ravel()

# rows 0-2: 12 simple distributions (indices 0-11)
for i, (label, data) in enumerate(simple.items()):
    if isinstance(data[0], int):
        bins = np.arange(min(data) - 0.5, max(data) + 1.5, 1)
    else:
        bins = 50
    axs[i].hist(data, bins=bins, density=True, alpha=0.7, edgecolor="white", linewidth=0.3)
    axs[i].set_title(label, fontsize=10)
    axs[i].tick_params(labelsize=8)

# theoretical PDF lines for simple distributions
# panel 0: Uniform(0,1)
x = np.linspace(0, 1, 200)
axs[0].plot(x, np.ones_like(x), "k--", lw=1.5, label="Uniform(0,1)")
axs[0].legend(fontsize=8)
# panel 1: Uniform(0,10)
x = np.linspace(0, 10, 200)
axs[1].plot(x, np.full_like(x, 0.1), "k--", lw=1.5, label="Uniform(0,10)")
axs[1].legend(fontsize=8)
# panel 2: Normal(0,1)
x = np.linspace(-4, 4, 200)
axs[2].plot(x, stats.norm.pdf(x), "k--", lw=1.5, label="Normal(0,1)")
axs[2].legend(fontsize=8)
# panel 3: Exp(1)
x = np.linspace(0, 5, 200)
axs[3].plot(x, stats.expon.pdf(x, scale=1), "k--", lw=1.5, label="Exp(1)")
axs[3].legend(fontsize=8)
# panel 4: Beta(2,5)
x = np.linspace(0, 1, 200)
axs[4].plot(x, stats.beta.pdf(x, 2, 5), "k--", lw=1.5, label="Beta(2,5)")
axs[4].legend(fontsize=8)
# panel 5: Pareto(3)
x = np.linspace(1, 5, 200)
axs[5].plot(x, stats.pareto.pdf(x, 3, scale=1), "k--", lw=1.5, label="Pareto(3)")
axs[5].legend(fontsize=8)
# panel 6: Triangular(0,10,5)
x = np.linspace(0, 10, 200)
axs[6].plot(x, stats.triang.pdf(x, 0.5, loc=0, scale=10), "k--", lw=1.5, label="Triangular(0,10,5)")
axs[6].legend(fontsize=8)
# panel 7: Discrete Uniform(1,6)
x = np.arange(1, 7)
axs[7].plot(x, np.full(6, 1/6), "k--", lw=1.5, marker="o", markersize=3, label="Discrete Uniform(1,6)")
axs[7].legend(fontsize=8)
# panel 8: Binomial(10,0.5)
x = np.arange(0, 11)
axs[8].plot(x, stats.binom.pmf(x, 10, 0.5), "k--", lw=1.5, marker="o", markersize=3, label="Binomial(10,0.5)")
axs[8].legend(fontsize=8)
# panel 9: Gamma(2,1)
x = np.linspace(0, 8, 200)
axs[9].plot(x, stats.gamma.pdf(x, 2, scale=1), "k--", lw=1.5, label="Gamma(2,1)")
axs[9].legend(fontsize=8)
# panel 10: LogNormal(0,0.5)
x = np.linspace(0.01, 3, 200)
axs[10].plot(x, stats.lognorm.pdf(x, 0.5, scale=1), "k--", lw=1.5, label="LogNormal(0,0.5)")
axs[10].legend(fontsize=8)
# panel 11: VonMises(0,3)
x = np.linspace(-np.pi, np.pi, 200)
axs[11].plot(x, stats.vonmises.pdf(x, 3, loc=0), "k--", lw=1.5, label="VonMises(0,3)")
axs[11].legend(fontsize=8)

# row 3 col 0: max of 2 vs sqrt — both ~ Beta(2,1)
axs[12].hist(max_of_two, bins=80, density=True, alpha=0.5, label="max(U₁,U₂)", color="steelblue")
axs[12].hist(sqrt_rand,  bins=80, density=True, alpha=0.5, label="√U",         color="coral")
x = np.linspace(0, 1, 200)
axs[12].plot(x, stats.beta.pdf(x, 2, 1), "k--", lw=1.5, label="Beta(2,1)")
axs[12].set_title("max(U₁,U₂) = √U", fontsize=10)
axs[12].legend(fontsize=8)

# row 3 col 1: min of 2 vs 1 - sqrt — both ~ Beta(1,2)
axs[13].hist(min_of_two, bins=80, density=True, alpha=0.5, label="min(U₁,U₂)", color="seagreen")
axs[13].hist(one_minus_sqrt, bins=80, density=True, alpha=0.5, label="1−√U",   color="darkorange")
x = np.linspace(0, 1, 200)
axs[13].plot(x, stats.beta.pdf(x, 1, 2), "k--", lw=1.5, label="Beta(1,2)")
axs[13].set_title("min(U₁,U₂) = 1 − √U", fontsize=10)
axs[13].legend(fontsize=8)

# row 3 col 2: midpoint vs Beta(2,2) — both ~ Beta(2,2)
axs[14].hist(inner_point, bins=80, density=True, alpha=0.5, label="point in [min,max]", color="purple")
axs[14].hist(beta22,      bins=80, density=True, alpha=0.5, label="Beta(2,2)",           color="gold")
x = np.linspace(0, 1, 200)
axs[14].plot(x, stats.beta.pdf(x, 2, 2), "k--", lw=1.5, label="Beta(2,2)")
axs[14].set_title("uniform in [min,max] = Beta(2,2)", fontsize=10)
axs[14].legend(fontsize=8)

# row 3 col 3: -ln(U) vs expovariate(1) — both ~ Exp(1)
axs[15].hist(minus_log_u, bins=80, density=True, alpha=0.5, label="−ln(U)",          color="teal")
axs[15].hist(expo,        bins=80, density=True, alpha=0.5, label="expovariate(1)",   color="salmon")
x = np.linspace(0, 5, 200)
axs[15].plot(x, stats.expon.pdf(x, scale=1), "k--", lw=1.5, label="Exp(1)")
axs[15].set_title("−ln(U) = Exp(1)", fontsize=10)
axs[15].legend(fontsize=8)

# row 4 col 0: max of 3 vs cube root — both ~ Beta(3,1)
axs[16].hist(max_of_three, bins=80, density=True, alpha=0.5, label="max(U₁,U₂,U₃)", color="steelblue")
axs[16].hist(cube_root,    bins=80, density=True, alpha=0.5, label="U^(1/3)",        color="coral")
x = np.linspace(0, 1, 200)
axs[16].plot(x, stats.beta.pdf(x, 3, 1), "k--", lw=1.5, label="Beta(3,1)")
axs[16].set_title("max(U₁,U₂,U₃) = U^(1/3)", fontsize=10)
axs[16].legend(fontsize=8)

# row 4 col 1: min of 3 vs 1 - cube root — both ~ Beta(1,3)
axs[17].hist(min_of_three,   bins=80, density=True, alpha=0.5, label="min(U₁,U₂,U₃)", color="seagreen")
axs[17].hist(one_minus_cube, bins=80, density=True, alpha=0.5, label="1−U^(1/3)",     color="darkorange")
x = np.linspace(0, 1, 200)
axs[17].plot(x, stats.beta.pdf(x, 1, 3), "k--", lw=1.5, label="Beta(1,3)")
axs[17].set_title("min(U₁,U₂,U₃) = 1 − U^(1/3)", fontsize=10)
axs[17].legend(fontsize=8)

# row 4 col 2: U1 + U2
axs[18].hist(sum_of_two, bins=80, density=True, alpha=0.7, color="orchid", edgecolor="white", linewidth=0.3)
axs[18].set_title("U₁ + U₂  (Irwin−Hall)", fontsize=10)
x = np.linspace(0, 2, 200)
axs[18].plot(x, np.where(x <= 1, x, 2 - x), "k--", lw=1.5, label="triangular(0,2,1)")
axs[18].legend(fontsize=8)

# row 4 col 3: k-th order stat — median of 5 = Beta(3,3)
axs[19].hist(order53_direct, bins=60, density=True, alpha=0.6, label="sorted(U₁..U₅)[3]", color="steelblue")
axs[19].hist(order53_beta,   bins=60, density=True, alpha=0.6, label="Beta(3,3)",          color="coral")
axs[19].set_title("k-th order stat = Beta(k, n−k+1)", fontsize=10)
axs[19].legend(fontsize=8)
xs = np.linspace(0, 1, 200)
axs[19].plot(xs, 30 * xs**2 * (1 - xs)**2, "k--", lw=1.5)
axs[19].text(0.5, 0.95, "n=5, k=3 (median)", transform=axs[19].transAxes,
             ha="center", va="top", fontsize=9)

# row 5 col 0: spacings
space_colors = plt.cm.viridis(np.linspace(0, 1, n_spacings + 1))
for i in range(n_spacings + 1):
    axs[20].hist(spacings[i], bins=60, density=True, alpha=0.5,
                 color=space_colors[i], label=f"S{i+1}")
axs[20].set_title("Spacings between sorted Uniforms", fontsize=10)
axs[20].set_xlabel("n=5 draws → 6 gaps")
axs[20].legend(fontsize=6, ncol=2)
xs = np.linspace(0, 1, 200)
axs[20].plot(xs, 5 * (1 - xs)**4, "k--", lw=1.5, label="Beta(1,5) PDF")
axs[20].legend(fontsize=7)

# row 5 col 1: extreme value
for n_val, data in extreme_panels.items():
    axs[21].hist(data, bins=80, density=True, alpha=0.5,
                 color={10: "coral", 100: "steelblue"}[n_val], label=f"n={n_val} × min")
xs = np.linspace(0, 6, 200)
axs[21].plot(xs, np.exp(-xs), "k--", lw=1.5, label="Exp(1) PDF")
axs[21].set_title("n × min(U₁..Uₙ)  →  Exp(1)", fontsize=10)
axs[21].set_xlim(0, 6)
axs[21].legend(fontsize=8)

# row 5 col 2: Gamma from summed exponentials
axs[22].hist(gamma_sum[2],    bins=80, density=True, alpha=0.4, label="∑−ln(U) k=2", color="steelblue")
axs[22].hist(gamma_direct[2], bins=80, density=True, alpha=0.4, label="Gamma(2,1)",   color="coral")
axs[22].hist(gamma_sum[3],    bins=80, density=True, alpha=0.4, label="∑−ln(U) k=3", color="seagreen")
axs[22].hist(gamma_direct[3], bins=80, density=True, alpha=0.4, label="Gamma(3,1)",   color="gold")
x = np.linspace(0, 8, 200)
axs[22].plot(x, stats.gamma.pdf(x, 2, scale=1), "k--", lw=1.5, label="Gamma(2,1) theory")
axs[22].plot(x, stats.gamma.pdf(x, 3, scale=1), "k--", lw=1.5, label="Gamma(3,1) theory")
axs[22].set_title("∑−ln(U) = Gamma(k, 1)", fontsize=10)
axs[22].legend(fontsize=7)

# row 5 col 3: CLT — Irwin-Hall(12) vs Normal(6,1)
axs[23].hist(irwin_hall_12, bins=80, density=True, alpha=0.6, color="orchid", label="U₁+...+U₁₂")
xx = np.linspace(2, 10, 200)
axs[23].plot(xx, 1/np.sqrt(2*np.pi) * np.exp(-(xx-6)**2/2), "k--", lw=1.5, label="Normal(6,1)")
axs[23].set_title("Irwin−Hall(12)  ≈  Normal(6,1)  (CLT)", fontsize=10)
axs[23].legend(fontsize=8)

# row 6 col 0: identity summary
axs[24].axis("off")
text = (
    "Hierarchy of Uniform identities\n\n"
    "−ln(U)           = Exp(1)\n"
    "∑−ln(U) [k]      = Gamma(k,1)\n"
    "Gamma / sum       = Dirichlet\n"
    "cumsum(Dirichlet) = Order stats\n\n"
    "k-th order stat (n draws)\n"
    "  = Beta(k, n−k+1)\n\n"
    "Spacings marginals\n"
    "  = Beta(1, n)\n\n"
    "Extreme value:\n"
    "  n·min → Exp(1),  n·(1−max) → Exp(1)\n\n"
    "Irwin−Hall(m) ≈ Normal(m/2, m/12)\n"
    "  (CLT, already close at m=12)"
)
axs[24].text(0.5, 0.5, text, transform=axs[24].transAxes,
    ha="center", va="center", fontsize=9.5, family="monospace",
    bbox=dict(boxstyle="round,pad=0.6", facecolor="lightgray", alpha=0.3))

for idx in range(25, 28):
    axs[idx].axis("off")

plt.suptitle("Distributions from Python's random module", fontsize=14, y=1.01)
plt.tight_layout()
plt.savefig("random_all.png", dpi=150, bbox_inches="tight")
print("Saved random_all.png")

# ========== SHA256 limb distributions ==========
N_sha = 50_000

rand32 = [random.getrandbits(32) for _ in range(N_sha)]

limbs_8 = []
limbs_16 = []
limbs_32 = []
for val in rand32:
    h = hashlib.sha256(struct.pack(">I", val)).digest()
    for i in range(32):
        limbs_8.append(h[i])
    for i in range(0, 32, 2):
        limbs_16.append(struct.unpack(">H", h[i:i+2])[0])
    for i in range(0, 32, 4):
        limbs_32.append(struct.unpack(">I", h[i:i+4])[0])

fig2, axs2 = plt.subplots(1, 3, figsize=(16, 5))

axs2[0].hist(limbs_8, bins=256, density=True, alpha=0.7, edgecolor="white", linewidth=0.3, color="steelblue")
axs2[0].set_title("SHA256 8-bit limbs (32 per hash)", fontsize=10)
axs2[0].set_xlabel("value (0–255)")
axs2[0].set_ylabel("density")
axs2[0].tick_params(labelsize=8)

axs2[1].hist(limbs_16, bins=256, density=True, alpha=0.7, edgecolor="white", linewidth=0.3, color="coral")
axs2[1].set_title("SHA256 16-bit limbs (16 per hash)", fontsize=10)
axs2[1].set_xlabel("value (0–65535)")
axs2[1].set_ylabel("density")
axs2[1].tick_params(labelsize=8)

axs2[2].hist(limbs_32, bins=256, density=True, alpha=0.7, edgecolor="white", linewidth=0.3, color="seagreen")
axs2[2].set_title("SHA256 32-bit limbs (8 per hash)", fontsize=10)
axs2[2].set_xlabel("value (0–4294967295)")
axs2[2].set_ylabel("density")
axs2[2].tick_params(labelsize=8)

plt.suptitle("SHA256(random(32-bit)) limb distributions", fontsize=14)
plt.tight_layout()
plt.savefig("sha256_limbs.png", dpi=150, bbox_inches="tight")
print("Saved sha256_limbs.png")
