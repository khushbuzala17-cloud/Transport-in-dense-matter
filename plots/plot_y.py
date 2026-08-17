import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter

# ==========================================
# Read the solved EOS data
# ==========================================
file_name = r"/home/khushbu/Desktop/khushbu/phD/EOS codes/final_eos_solved_output.xlsx"   # Change if needed

sheet_name = "eta_0.0"      # <-- Change to the sheet you want

df = pd.read_excel(file_name, sheet_name=sheet_name)

# ==========================================
# Plot yn, yp, ys and ye vs nB
# ==========================================
plt.figure(figsize=(8,6))

plt.plot(df["nB"], df["yn"], label=r"$y_n$", linewidth=1)
plt.plot(df["nB"], df["yp"], label=r"$y_p$", linewidth=1)
plt.plot(df["nB"], df["ys"], label=r"$y_s$", linewidth=1)
plt.plot(df["nB"], df["ye"], label=r"$y_e$", linewidth=1)
#plt.plot(df["nB"], df["f"], label=r"$f$", linewidth=1)
plt.plot(df["nB"], df["yu"], label=r"$yu$", linewidth=1)
plt.plot(df["nB"], df["yd"], label=r"$yd$", linewidth=1)

# ---------- Logarithmic y-axis ----------
plt.yscale('log')

# Limits
plt.ylim(0.005, 2)

# Tick locations like the paper
ticks = [0.005, 0.01, 0.05, 0.1, 0.5, 1.0,1.22,1.5]
plt.yticks(ticks)

# Show ticks as decimals instead of 10^-2
plt.gca().yaxis.set_major_formatter(ScalarFormatter())
plt.gca().set_yticklabels(
    ['0.005', '0.010', '0.050', '0.100', '0.500', '1','1.22','1.5']
)

# ----------------------------------------

plt.xlabel(r"$n_B$ (fm$^{-3}$)", fontsize=14)
plt.ylabel(r"$Y_i$", fontsize=14)
plt.title(r"$\eta=0.0$")
plt.grid(True, which='both', linestyle='--', alpha=0.5)
plt.legend(fontsize=12)

plt.tight_layout()
plt.savefig("particle_fractions_vs_nb(eta=0.0).png", dpi=300)
plt.show()