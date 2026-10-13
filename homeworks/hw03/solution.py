import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

current_file = Path(__file__).resolve()
repo_root = current_file.parents[2]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from homeworks.hw01.data_utils import ROOT


PUMPING_PATH = ROOT / "data" / "raw" / "hw01" / "pumping_test.txt"
WELLS_PATH = ROOT / "data" / "processed" / "hw01" / "wells_clean.csv"

REQUIRED_FIGURES = [
    ROOT / "figures" / "hw02" / "pressure_over_time.svg",
    ROOT / "figures" / "hw02" / "pressure_by_radius.svg",
]


OUTPUT_DIR_FIG = ROOT / "figures" / "hw03"
OUTPUT_DIR_DATA = ROOT / "homeworks" / "hw03"
OUTPUT_NPZ_PATH = OUTPUT_DIR_DATA / "viz_data.npz"


def main():
    
    OUTPUT_DIR_FIG.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR_DATA.mkdir(parents=True, exist_ok=True)

    
    print("Шаг 1: Проверка существования SVG из ДЗ №2...")
    for path in REQUIRED_FIGURES:
        assert path.exists(), f"Файл не найден: {path}"
        assert path.stat().st_size > 0, f"Файл пустой: {path}"

   
    print("Загрузка таблиц...")
    pumping_test = pd.read_csv(PUMPING_PATH, sep="\t")
    wells = pd.read_csv(WELLS_PATH)

    
    assert pumping_test.shape == (13, 4), f"Неверная форма pumping_test: {pumping_test.shape}"
    assert wells.shape[0] == 7, f"Неверное количество строк в wells: {wells.shape[0]}"
    assert not pumping_test.isna().any().any(), "В pumping_test есть пропуски (NaN)!"
    assert not wells.isna().any().any(), "В wells есть пропуски (NaN)!"

    
    print("Шаг 2: Подготовка и сортировка массивов...")
    # Из pumping_test получите:
    time_h = pumping_test["time_h"].to_numpy()
    boundary_pressure_mpa = pumping_test["boundary_pressure_mpa"].to_numpy()
    well_pressure_mpa = pumping_test["well_pressure_mpa"].to_numpy()

    
    well_id = wells["well_id"].to_numpy()
    radius_m = wells["radius_m"].to_numpy()
    observation_pressure_mpa = wells["pressure_mpa"].to_numpy()

    
    order = np.argsort(radius_m)
    well_id = well_id[order]
    radius_m = radius_m[order]
    observation_pressure_mpa = observation_pressure_mpa[order]

    
    pressure_drop_mpa = boundary_pressure_mpa - well_pressure_mpa
    observation_drop_mpa = 12.0 - observation_pressure_mpa

    
    print("Шаг 3: Создание композиции 2х2 ...")
    fig, axes = plt.subplots(
        2,
        2,
        figsize=(12, 8),
        constrained_layout=True,
    )

    
    ax_a = axes[0, 0]
    ax_a.plot(time_h, boundary_pressure_mpa, marker='o', label='Давление на границе')
    ax_a.plot(time_h, well_pressure_mpa, marker='s', label='Давление в скважине')
    ax_a.set_title("Панель A. Изменение давления во времени")
    ax_a.set_xlabel("Время, ч")
    ax_a.set_ylabel("Давление, МПа")
    ax_a.legend()
    ax_a.grid(True)

    
    ax_b = axes[0, 1]
    sc = ax_b.scatter(radius_m, observation_pressure_mpa, c=observation_pressure_mpa, cmap='viridis', s=100, zorder=3)
    cbar = fig.colorbar(sc, ax=ax_b)
    cbar.set_label("Давление, МПа")
    ax_b.set_title("Панель B. Давление в наблюдательных скважинах")
    ax_b.set_xlabel("Расстояние (радиус), м")
    ax_b.set_ylabel("Давление, МПа")
    ax_b.grid(True)

    
    ax_c = axes[1, 0]
    
    x_labels = [str(wid) for wid in well_id]
    ax_c.bar(x_labels, observation_drop_mpa, color='skyblue', edgecolor='black', zorder=3)
    ax_c.set_title("Панель C. Падение давления для семи скважин")
    ax_c.set_xlabel("Идентификатор скважины ")
    ax_c.set_ylabel("Падение давления, МПа")
    ax_c.grid(True, axis='y')

    
    ax_d = axes[1, 1]
    ax_d.boxplot([boundary_pressure_mpa, well_pressure_mpa], tick_labels=["На границе", "В скважине"])
    ax_d.set_title("Панель D. Диаграмма размаха диапазонов давления")
    ax_d.set_ylabel("Давление, МПа")
    ax_d.grid(True, axis="y", linestyle=":", alpha=0.3)

    
    fig.savefig(OUTPUT_DIR_FIG / "pressure_overview.png", dpi=200)
    fig.savefig(OUTPUT_DIR_FIG / "pressure_overview.svg")
    plt.close(fig)


    
    print("Шаг 4: Построение сравнения boxplot и violin plot ...")
    fig2, axes2 = plt.subplots(
        1,
        2,
        figsize=(12, 6),
        constrained_layout=True,
    )
    
    data_to_plot = [boundary_pressure_mpa, well_pressure_mpa]
    labels_to_plot = ['На границе', 'В скважине']

    
    
    axes2[0].boxplot(data_to_plot, tick_labels=labels_to_plot)
    axes2[0].set_title("Панель A. Диаграмма размаха (Boxplot)")
    axes2[0].set_ylabel("Давление, МПа")
    axes2[0].grid(True, axis="y", linestyle=":", alpha=0.3)

    
    axes2[1].violinplot(data_to_plot, showmeans=True, showmedians=True)
    axes2[1].set_xticks([1, 2])
    axes2[1].set_xticklabels(labels_to_plot)
    axes2[1].set_title("B. Скрипичный график (Violin plot)")
    axes2[1].set_ylabel("Давление, МПа")
    axes2[1].grid(True, axis='y')

    
    fig2.savefig(OUTPUT_DIR_FIG / "distributions.png", dpi=200)
    fig2.savefig(OUTPUT_DIR_FIG / "distributions.svg")
    plt.close(fig2)


   
    print("Шаг 5: Рисунки успешно сохранены в figures/hw03.")

    
    
    
    
       
    print("Шаг 6: Сохранение и проверка viz_data.npz...")
    np.savez(
        OUTPUT_NPZ_PATH,
        time_h=time_h,
        boundary_pressure_mpa=boundary_pressure_mpa,
        well_pressure_mpa=well_pressure_mpa,
        pressure_drop_mpa=pressure_drop_mpa,
        well_id=well_id.astype(str), 
        radius_m=radius_m,
        observation_pressure_mpa=observation_pressure_mpa,
        observation_drop_mpa=observation_drop_mpa,
    )


    loaded = np.load(OUTPUT_NPZ_PATH, allow_pickle=True)
    assert list(loaded.keys()) == [
        'time_h', 'boundary_pressure_mpa', 'well_pressure_mpa', 'pressure_drop_mpa',
        'well_id', 'radius_m', 'observation_pressure_mpa', 'observation_drop_mpa'
    ], "Ошибка: Ключи в npz файле не совпадают!"
    assert loaded['time_h'].shape == (13,), "Ошибка: Изменилась форма массива time_h!"
    assert loaded['well_id'].shape == (7,), "Ошибка: Изменилась форма массива well_id!"
    
    print("\nУспех")

    
    
    
    
    
    
    
    
    
    loaded = np.load(OUTPUT_NPZ_PATH)
    assert list(loaded.keys()) == [
        'time_h', 'boundary_pressure_mpa', 'well_pressure_mpa', 'pressure_drop_mpa',
        'well_id', 'radius_m', 'observation_pressure_mpa', 'observation_drop_mpa'
    ], "Ошибка: Ключи в npz файле не совпадают!"
    assert loaded['time_h'].shape == (13,), "Ошибка: Форма массива time_h изменилась!"
    assert loaded['well_id'].shape == (7,), "Ошибка: Форма массива well_id изменилась!"
    
    print("\nПроверка NPZ успешно пройдена! Все пункты чек-листа выполнены.")


if __name__ == "__main__":
    main()

