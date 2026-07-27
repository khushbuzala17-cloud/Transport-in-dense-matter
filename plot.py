import pandas as pd
import matplotlib.pyplot as plt

# ==========================================
# Read the solved EOS data
# ==========================================
file_name = r"/home/khushbu/Desktop/khushbu/phD/EOS codes/final_eos_solved_output1.xlsx"   # Change if needed

df = pd.read_excel(file_name)

# ==========================================
# Plot yn, yp, ys and ye vs nB
# ==========================================
plt.figure(figsize=(8,6))

plt.plot(df["nB"], df["yn"], label=r"$y_n$", linewidth=1)
plt.plot(df["nB"], df["yp"], label=r"$y_p$", linewidth=1)
plt.plot(df["nB"], df["ys"], label=r"$y_s$", linewidth=1)
plt.plot(df["nB"], df["ye"], label=r"$y_e$", linewidth=1)
plt.plot(df["nB"], df["f"], label=r"$f$", linewidth=1)
#plt.plot(df["nB"], df["yu"], label=r"$yu$", linewidth=1)
#plt.plot(df["nB"], df["yd"], label=r"$yd$", linewidth=1)

plt.xlabel(r"$n_B$ (fm$^{-3}$)", fontsize=14)
plt.ylabel("Particle Fraction", fontsize=14)
plt.title("Particle Fractions vs Baryon Density", fontsize=15)

plt.grid(True, linestyle="--", alpha=0.5)
plt.legend(fontsize=12)

plt.tight_layout()

# Save figure
plt.savefig("particle_fractions_vs_nb1.png", dpi=300)

# Show figure
plt.show()