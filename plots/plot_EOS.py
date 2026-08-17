import pandas as pd
import matplotlib.pyplot as plt

# ==========================================
# Excel file path
# ==========================================
file_name = r"/home/khushbu/Desktop/khushbu/phD/EOS codes/max+gibs(paper)/final_eos_solved_output_NJL.xlsx"   # Change to your file

# ==========================================
# Define the sheet names manually
# ==========================================
sheet_names = [
    "eta_0.0_NJL",
    "eta_0.3_NJL",
    "eta_0.6_NJL",
    "eta_0.9_NJL"

]

# Optional: Labels you want in the legend
labels = [
    "$\eta = 0.0$",
    "$\eta = 0.3$",
    "$\eta = 0.6$",
    "$\eta = 0.9$"

]

# Different line styles (optional)
styles = ['-', '-', '-', '-','-']

# ==========================================
# Plot f vs nb from all sheets
# ==========================================
plt.figure(figsize=(8,6))

for i, sheet in enumerate(sheet_names):

    # Read the current sheet
    df = pd.read_excel(file_name, sheet_name=sheet)

    # Plot
    plt.plot(df["eps_total"],
             df["P_total"],
             linewidth=2,
             label=labels[i])

# ==========================================
# Formatting
# ==========================================
plt.xlabel(r"$\epsilon$ (MeV fm$^{-3}$)", fontsize=14)
plt.ylabel(r"$P$ (MeV fm$^{-3}$)", fontsize=14)

plt.title(r"EOS", fontsize=15)

plt.xlim(left=0)
#plt.ylim(0, 1)

plt.grid(True, linestyle="--", alpha=0.5)
plt.legend(fontsize=11)

plt.tight_layout()
plt.savefig("p vs e_NJL.png", dpi=300)
plt.show()