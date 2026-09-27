import matplotlib.pyplot as plt
import pypsa

file_path = "results/test-elec/networks/solved_2050.nc"
n = pypsa.Network(file_path)

print("--- Network Overview ---")
print(n)

print("\n--- Optimal Capacities (MW) ---")
capacities = n.generators.groupby("carrier").p_nom_opt.sum()
print(capacities)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Plot A: Optimal capacity
capacities.plot(kind="bar", ax=axes[0], color="skyblue", edgecolor="black")
axes[0].set_title("Optimal Installed Capacity by Carrier (MW)")
axes[0].set_ylabel("Capacity [MW]")
axes[0].set_xlabel("Carrier")
axes[0].grid(axis="y", linestyle="--", alpha=0.7)

# Plot B: Hourly dispatch profile (pandas-compatible)
dispatch_by_carrier = n.generators_t.p.T.groupby(n.generators.carrier).sum().T
dispatch_by_carrier.plot(ax=axes[1], lw=1.5)
axes[1].set_title("Hourly Electricity Generation Profile")
axes[1].set_ylabel("Production [MW]")
axes[1].set_xlabel("Snapshot")
axes[1].grid(True, linestyle="--", alpha=0.7)
axes[1].legend(loc="upper right")

plt.tight_layout()
plt.show()