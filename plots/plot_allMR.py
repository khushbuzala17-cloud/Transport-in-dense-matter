import pandas as pd
import matplotlib.pyplot as plt

# ==========================================
# Excel file path
# ==========================================
file_name = r"/home/khushbu/Desktop/khushbu/phD/EOS codes/max+gibs(paper)/Final MR curve_NJL.xlsx"   # Change to your file

# ==========================================
# Define the sheet names manually
# ==========================================
sheet_names = [
    "eta_0.0",
    "eta_0.3",
    "eta_0.6",
    "eta_1.0"
]

# Optional: Labels you want in the legend
labels = [
    "$\eta = 0.0$",
    "$\eta = 0.3$",
    "$\eta = 0.6$",
    "$\eta = 1.0$"
]

# Different line styles (optional)
styles = ['-', '-', '-', '-']

# ==========================================
# Plot f vs nb from all sheets
# ==========================================
plt.figure(figsize=(8,6))

for i, sheet in enumerate(sheet_names):

    # Read the current sheet
    df = pd.read_excel(file_name, sheet_name=sheet)

    # Plot
    plt.plot(df["Radius (km)"],
             df["Mass (M_sun)"],
             linestyle=styles[i],
             linewidth=2,
             label=labels[i])

# ==========================================
# Formatting
# ==========================================
plt.xlabel(r"R (km)", fontsize=14)
plt.ylabel(r"M ($M_\odot$))", fontsize=14)

plt.title(r"MR", fontsize=15)

plt.xlim(10,12)
#plt.ylim(0, 1)

plt.grid(True, linestyle="--", alpha=0.5)
plt.legend(fontsize=11)

plt.tight_layout()
plt.savefig("MR.png", dpi=300)
plt.show()