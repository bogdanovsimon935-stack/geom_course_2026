import os
import matplotlib.pyplot as plt

def plot_pressure_over_time(table, output_dir):
    
    os.makedirs(output_dir, exist_ok=True)
    
    
    fig, ax = plt.subplots(figsize=(8, 5))
    
    time = table["time_h"]
    p_boundary = table["boundary_pressure_mpa"]
    p_well = table["well_pressure_mpa"]
    
    
    ax.plot(time, p_boundary, label="Давление на границе", color="tab:blue", marker="o")
    ax.plot(time, p_well, label="Давление в скважине", color="tab:orange", marker="s")
    
   
    min_index = table["well_pressure_mpa"].idxmin()
    min_time = table.loc[min_index, "time_h"]
    min_pressure = table.loc[min_index, "well_pressure_mpa"]
    
    ax.annotate(
        f"Min P = {min_pressure:.2f} МПа",
        xy=(min_time, min_pressure),
        xytext=(min_time + (time.max() - time.min()) * 0.05, min_pressure + (p_boundary.max() - p_well.min()) * 0.1),
        arrowprops=dict(facecolor="black", shrink=0.05, width=1, headwidth=6),
        fontsize=10
    )
    
    
    ax.set_title("Изменение давления во времени")
    ax.set_xlabel("Время, ч")
    ax.set_ylabel("Давление, МПа")
    ax.grid(True, which="both", linestyle=":", alpha=0.3)
    ax.legend()
    
    fig.tight_layout()
    
    
    fig.savefig(os.path.join(output_dir, "pressure_over_time.png"), bbox_inches="tight", dpi=200)
    fig.savefig(os.path.join(output_dir, "pressure_over_time.svg"), bbox_inches="tight")
    
    
    plt.close(fig)


def plot_pressure_by_radius(table, output_dir):
    
    os.makedirs(output_dir, exist_ok=True)
    
    
    fig, ax = plt.subplots(figsize=(8, 5))
    
    radius = table["radius_m"]
    pressure = table["pressure_mpa"]
    
    
    ax.plot(radius, pressure, marker="o", markersize=6, linestyle="none", color="tab:green")
    
    
    ax.set_xscale("log")
    
    
    ax.set_title("Давление в наблюдательных скважинах")
    ax.set_xlabel("Расстояние, м")
    ax.set_ylabel("Давление, МПа")
    ax.grid(True, which="both", linestyle=":", alpha=0.3)
    
    fig.tight_layout()
    
   
    fig.savefig(os.path.join(output_dir, "pressure_by_radius.png"), bbox_inches="tight", dpi=200)
    fig.savefig(os.path.join(output_dir, "pressure_by_radius.svg"), bbox_inches="tight")
    
    
    plt.close(fig)
 