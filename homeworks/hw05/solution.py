import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

# Автоматически находим корень репозитория geom_course_2026
current_file = Path(__file__).resolve()
repo_root = current_file.parents[2] 

if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from homeworks.hw01.data_utils import ROOT

# Пути к файлам
INPUT_NPZ_PATH = ROOT / "homeworks" / "hw04" / "field_data.npz"
REQUIRED_SVG_PATH = ROOT / "figures" / "hw04" / "pressure_maps.svg"

OUTPUT_CSV_DIR = ROOT / "homeworks" / "hw05"
OUTPUT_CSV_PATH = OUTPUT_CSV_DIR / "pressure_long.csv"

OUTPUT_GIF_DIR = ROOT / "figures" / "hw05"
OUTPUT_GIF_PATH = OUTPUT_GIF_DIR / "pressure_animation.gif"


def main():
    # Создаем директории для результатов, если их нет
    OUTPUT_CSV_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_GIF_DIR.mkdir(parents=True, exist_ok=True)

    # =========================================================================
    # 1. Проверьте результат ДЗ №4
    # =========================================================================
    print("Шаг 1: Проверка результатов ДЗ №4...")
    assert INPUT_NPZ_PATH.exists(), f"Файл не найден: {INPUT_NPZ_PATH}"
    assert REQUIRED_SVG_PATH.exists(), f"Файл не найден: {REQUIRED_SVG_PATH}"
    assert REQUIRED_SVG_PATH.stat().st_size > 0, "Файл pressure_maps.svg пустой!"

    data = np.load(INPUT_NPZ_PATH)
    
    time_h = data["time_h"]
    radius_m = data["radius_m"]
    well_id = data["well_id"]
    pressure_field = data["pressure_field"]
    contour_levels = data["contour_levels"]

    assert time_h.shape == (13,)
    assert radius_m.shape == (7,)
    assert well_id.shape == (7,)
    assert pressure_field.shape == (13, 7)
    assert contour_levels.shape == (9,)
    assert np.all(np.diff(time_h) > 0), "Время должно строго возрастать"
    assert np.all(np.diff(radius_m) > 0), "Радиус должен строго возрастать"

    # =========================================================================
    # 2. Создайте длинную таблицу
    # =========================================================================
    print("Шаг 2: Создание длинной таблицы...")
    time_grid, radius_grid = np.meshgrid(time_h, radius_m, indexing="ij")
    time_index, radius_index = np.indices(pressure_field.shape)
    
    well_names_long = well_id[radius_index.reshape(-1)]

    long_table = pd.DataFrame({
        "time_h": time_grid.reshape(-1),
        "radius_m": radius_grid.reshape(-1),
        "well_id": well_names_long,
        "pressure_mpa": pressure_field.reshape(-1)
    })

    assert long_table.shape == (91, 4)
    assert long_table["time_h"].nunique() == 13
    assert long_table["radius_m"].nunique() == 7
    assert not long_table.isna().any().any()

    long_table.to_csv(OUTPUT_CSV_PATH, index=False)
    
    # Повторная проверка загрузки CSV
    loaded_table = pd.read_csv(OUTPUT_CSV_PATH)
    assert loaded_table.shape == (91, 4)

    # =========================================================================
    # 3. Создайте анимацию
    # =========================================================================
    print("Шаг 3: Построение GIF-анимации...")
    fig_anim, ax = plt.subplots(figsize=(8, 5))
    ax.set_xscale("log")
    ax.set_xlabel("Расстояние, м")
    ax.set_ylabel("Давление, МПа")
    ax.grid(True, which="both", linestyle=":", alpha=0.5)

    margin = 0.1
    ax.set_ylim(pressure_field.min() - margin, pressure_field.max() + margin)
    ax.set_xlim(radius_m.min() * 0.8, radius_m.max() * 1.2)

    line, = ax.plot([], [], color="tab:blue", linestyle="-", linewidth=2)
    points, = ax.plot([], [], color="tab:red", marker="o", linestyle="None", markersize=8)

    def update(frame_index):
        current_pressure = pressure_field[frame_index]
        line.set_data(radius_m, current_pressure)
        points.set_data(radius_m, current_pressure)
        ax.set_title(f"Профиль давления во времени: t = {time_h[frame_index]:.1f} ч")
        return line, points

    ani = FuncAnimation(fig_anim, update, frames=len(time_h), blit=True)
    writer = PillowWriter(fps=2)
    ani.save(OUTPUT_GIF_PATH, writer=writer)
    plt.close(fig_anim)
    
    print("[Успех] Расчетный скрипт выполнен. Файлы созданы!")


if __name__ == "__main__":
    main()
