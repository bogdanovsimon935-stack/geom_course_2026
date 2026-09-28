import sys
from pathlib import Path
import pandas as pd

current_file = Path(__file__).resolve()
repo_root = current_file.parents[2]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))


from homeworks.hw01.data_utils import ROOT
from plot_utils import plot_pressure_over_time, plot_pressure_by_radius


PUMPING_TEST_PATH = ROOT / "data" / "raw" / "hw01" / "pumping_test.txt"
WELLS_CLEAN_PATH = ROOT / "data" / "processed" / "hw01" / "wells_clean.csv"
figures_dir = ROOT / "figures" / "hw02"

def main():
   
    print("Чтение файла pumping_test.txt...")
    pumping_test = pd.read_csv(PUMPING_TEST_PATH, sep="\t")
    
    print("Построение графика изменения давления во времени...")
    plot_pressure_over_time(pumping_test, figures_dir)

    print("Чтение файла wells_clean.csv...")
    wells_clean = pd.read_csv(WELLS_CLEAN_PATH)
    
   
    wells_sorted = wells_clean.sort_values("radius_m")
    
    
    assert (wells_sorted["radius_m"] > 0).all(), "Ошибка: Радиус должен быть больше 0 для логарифмической шкалы!"
    
    print("Построение графика зависимости давления от расстояния...")
    plot_pressure_by_radius(wells_sorted, figures_dir)

    print("Запуск проверок assert для созданных файлов...")
    assert (figures_dir / "pressure_over_time.png").exists()
    assert (figures_dir / "pressure_over_time.svg").exists()
    assert (figures_dir / "pressure_by_radius.png").exists()
    assert (figures_dir / "pressure_by_radius.svg").exists()
    
    print("\n Все 4 файла графиков успешно сгенерированы в figures/hw02.")

if __name__ == "__main__":
    main()
