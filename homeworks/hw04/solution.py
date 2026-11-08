import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

current_file = Path(__file__).resolve()
repo_root = current_file.parents[2]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from homeworks.hw01.data_utils import ROOT


INPUT_NPZ_PATH = ROOT / "homeworks" / "hw03" / "viz_data.npz"

OUTPUT_DIR_FIG = ROOT / "figures" / "hw04"
OUTPUT_DIR_DATA = ROOT / "homeworks" / "hw04"
OUTPUT_NPZ_PATH = OUTPUT_DIR_DATA / "field_data.npz"


def main():
    
    OUTPUT_DIR_FIG.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR_DATA.mkdir(parents=True, exist_ok=True)

    
    print("Шаг 1: Загрузка данных...")
    assert INPUT_NPZ_PATH.exists(), f"Файл не найден: {INPUT_NPZ_PATH}. Запустите сначала скрипт hw03!"
    
    data = np.load(INPUT_NPZ_PATH)

    required_keys = {
        "time_h", "boundary_pressure_mpa", "well_pressure_mpa", "pressure_drop_mpa",
        "well_id", "radius_m", "observation_pressure_mpa", "observation_drop_mpa"
    }
    assert required_keys.issubset(data.files), "Ошибка: в файле не хватает нужных ключей!"

    
    time_h = data["time_h"]
    boundary_pressure_mpa = data["boundary_pressure_mpa"]
    well_pressure_mpa = data["well_pressure_mpa"]
    pressure_drop_mpa = data["pressure_drop_mpa"]
    well_id = data["well_id"]
    radius_m = data["radius_m"]
    observation_pressure_mpa = data["observation_pressure_mpa"]
    observation_drop_mpa = data["observation_drop_mpa"]

   
    assert time_h.shape == (13,)
    assert well_pressure_mpa.shape == (13,)
    assert well_id.shape == (7,)
    assert radius_m.shape == (7,)
    assert observation_pressure_mpa.shape == (7,)
    assert np.all(np.diff(time_h) > 0), "Ошибка: Время должно строго возрастать!"
    assert np.all(np.diff(radius_m) > 0), "Ошибка: Радиус должен строго возрастать!"


    
    print("Шаг 2: Создание учебного поля через broadcasting...")
    
    
    time_component = well_pressure_mpa - well_pressure_mpa[0]
    
    
    radius_component = observation_pressure_mpa - observation_pressure_mpa[0]

    
    pressure_field = (
        well_pressure_mpa[0]
        + time_component[:, np.newaxis]
        + radius_component[np.newaxis, :]
    )

    assert pressure_field.shape == (13, 7), f"Неверная форма поля: {pressure_field.shape}"


    
    print("Шаг 3-4: Согласование осей и подготовка уровней...")
    

    field_for_plot = pressure_field.T
    assert field_for_plot.shape == (7, 13)


    vmin = pressure_field.min()
    vmax = pressure_field.max()
    levels = np.linspace(vmin, vmax, 9)

    assert (radius_m > 0).all(), "Ошибка: Радиусы должны быть строго больше нуля для log-шкалы!"


    
    print("Шаг 5-8: Построение сдвоенного графика сравнения...")
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

 
    mesh = ax1.pcolormesh(
        time_h,
        radius_m,
        field_for_plot,
        shading="auto",
        cmap="viridis",
        vmin=vmin,
        vmax=vmax,
        edgecolors="white",  
        linewidths=0.5
    )
    ax1.set_yscale("log")
    ax1.set_title("Панель А. Тепловая карта по ячейкам (pcolormesh)")
    ax1.set_xlabel("Время, ч")
    ax1.set_ylabel("Расстояние, м")
    cbar1 = fig.colorbar(mesh, ax=ax1)
    cbar1.set_label("Давление, МПа")

  
    filled = ax2.contourf(
        time_h,
        radius_m,
        field_for_plot,
        levels=levels,
        cmap="viridis",
        vmin=vmin,
        vmax=vmax
    )
    lines = ax2.contour(
        time_h,
        radius_m,
        field_for_plot,
        levels=levels,
        colors="black",
        linewidths=0.6
    )
    ax2.clabel(lines, fmt="%.2f", fontsize=9)
    ax2.set_yscale("log")
    ax2.set_title("Панель B. Заливка по уровням и изолинии (contourf)")
    ax2.set_xlabel("Время, ч")
    ax2.set_ylabel("Расстояние, м")
    cbar2 = fig.colorbar(filled, ax=ax2)
    cbar2.set_label("Давление, МПа")

    plt.tight_layout()
    
  
    fig.savefig(OUTPUT_DIR_FIG / "pressure_maps.png", dpi=200)
    fig.savefig(OUTPUT_DIR_FIG / "pressure_maps.svg")
    plt.close(fig)


   
    print("Шаг 9: Построение графиков только изолиний...")
    
    fig_lines, ax_only_lines = plt.subplots(figsize=(8, 6))
    
    lines_only = ax_only_lines.contour(
        time_h,
        radius_m,
        field_for_plot,
        levels=levels,
        colors="black",
        linewidths=0.8
    )
    ax_only_lines.clabel(lines_only, fmt="%.2f", fontsize=9)
    
    ax_only_lines.set_yscale("log")
    ax_only_lines.set_title("Изолинии поля давления (без цветной заливки)")
    ax_only_lines.set_xlabel("Время, ч")
    ax_only_lines.set_ylabel("Расстояние, м")
    
    plt.tight_layout()
    
    # Сохраняем
    fig_lines.savefig(OUTPUT_DIR_FIG / "pressure_isolines.png", dpi=200)
    fig_lines.savefig(OUTPUT_DIR_FIG / "pressure_isolines.svg")
    plt.close(fig_lines)


  
    print("Шаг 10: Экспорт результатов в field_data.npz...")
    
    np.savez(
        OUTPUT_NPZ_PATH,
        time_h=time_h,
        radius_m=radius_m,
        well_id=well_id,
        pressure_field=pressure_field,
        contour_levels=levels
    )

    
    check_data = np.load(OUTPUT_NPZ_PATH)
    assert "pressure_field" in check_data
    assert check_data["pressure_field"].shape == (13, 7)
    
    print("\n[Успех] Домашнее задание №4 полностью выполнено и прошло все проверки!")


if __name__ == "__main__":
    main()
