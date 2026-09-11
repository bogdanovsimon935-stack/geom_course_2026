from pathlib import Path
import numpy as np
import pandas as pd
from data_utils import (
    read_wells, 
    read_layers, 
    read_pumping_test, 
    validate_and_clean_data,
    calculate_columns,
    calculate_numpy_math,
    calculate_statistics_and_slices,
    create_numpy_arrays,
    DatasetInfo
)

ROOT = Path(__file__).resolve().parent.parent.parent


def main():
    processed_dir = ROOT / "data" / "processed" / "hw01"
    exports_dir = ROOT / "exports" / "hw01"
    raw_dir = ROOT / "data" / "raw" / "hw01"
    
    
    processed_dir.mkdir(parents=True, exist_ok=True)
    exports_dir.mkdir(parents=True, exist_ok=True)
    
   
    wells_raw = read_wells(raw_dir / "wells.csv")
    layers_raw = read_layers(raw_dir / "layers.xlsx")
    pumping_raw = read_pumping_test(raw_dir / "pumping_test.txt")
    
    info_wells = DatasetInfo("Таблица Wells", wells_raw)
    info_layers = DatasetInfo("Таблица Layers", layers_raw)
    info_pumping = DatasetInfo("Таблица Pumping Test", pumping_raw)
    
    print("\n Раздел 12: Вывод метода describe() ")
    print(info_wells.describe())
    print(info_layers.describe())
    print(info_pumping.describe())
    
    
    wells_clean = validate_and_clean_data(wells_raw, layers_raw)
    wells_clean = calculate_columns(wells_clean)
    wells_clean, pumping_processed = calculate_numpy_math(wells_clean, pumping_raw)
    calculate_statistics_and_slices(wells_clean, layers_raw)
    
    
    pressure_matrix, pressure_cube = create_numpy_arrays(pumping_processed)
    

    pressure_transposed = pressure_matrix.T
    flat = pressure_cube.reshape(-1)
    restored = flat.reshape(pressure_cube.shape)
    assert np.allclose(restored, pressure_cube)
    
    print("\n--- Раздел 13: Экспорт результатов и верификация ---")
    
    wells_clean.to_csv(processed_dir / "wells_clean.csv", index=False)
    
    
    stats_summary = {
        "Столбец": ["pressure_mpa", "radius_m", "thickness_m", "porosity_fraction"],
        "Минимум": [wells_clean["pressure_mpa"].min(), wells_clean["radius_m"].min(), layers_raw["thickness_m"].min(), layers_raw["porosity_fraction"].min()],
        "Максимум": [wells_clean["pressure_mpa"].max(), wells_clean["radius_m"].max(), layers_raw["thickness_m"].max(), layers_raw["porosity_fraction"].max()],
        "Среднее": [wells_clean["pressure_mpa"].mean(), wells_clean["radius_m"].mean(), layers_raw["thickness_m"].mean(), layers_raw["porosity_fraction"].mean()]
    }
    df_summary = pd.DataFrame(stats_summary)
    df_summary.to_excel(processed_dir / "table_summary.xlsx", index=False, engine="openpyxl")
    
   
    np.save(processed_dir / "pressure_matrix.npy", pressure_matrix)
    np.savez(processed_dir / "pressure_cube.npz", pressure_cube=pressure_cube) 
    
   
    txt_path = exports_dir / "results.txt"
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("=== ОТЧЕТ ПО ОБРАБОТКЕ ДАННЫХ ===\n\n")
        f.write(f"Форма двумерного массива (pressure_matrix): {pressure_matrix.shape}\n")
        f.write(f"Форма трёхмерного массива (pressure_cube): {pressure_cube.shape}\n\n")
        f.write("Сводная статистика по ключевым параметрам:\n")
        f.write(df_summary.to_string(index=False))
        

    matrix_loaded = np.load(processed_dir / "pressure_matrix.npy")
    
    cube_npz_loaded = np.load(processed_dir / "pressure_cube.npz")
    cube_loaded = cube_npz_loaded["pressure_cube"]
    

    assert np.allclose(matrix_loaded, pressure_matrix), "Ошибка: Восстановленная матрица 2D повреждена!"
    assert np.allclose(cube_loaded, pressure_cube), "Ошибка: Восстановленный куб 3D поврежден!"
    
    print(f"[УСПЕХ] Все 5 выходных файлов успешно сгенерированы:")
    print(f"  1. {processed_dir / 'wells_clean.csv'}")
    print(f"  2. {processed_dir / 'table_summary.xlsx'}")
    print(f"  3. {processed_dir / 'pressure_matrix.npy'}")
    print(f"  4. {processed_dir / 'pressure_cube.npz'}")
    print(f"  5. {txt_path}")
    print("[УСПЕХ] Обратная проверка через np.allclose подтвердила полную идентичность данных!")

if __name__ == "__main__":
    main()
