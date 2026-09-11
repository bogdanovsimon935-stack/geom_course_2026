from pathlib import Path
import pandas as pd
import numpy as np


ROOT = Path(__file__).resolve().parents[2]


class DatasetInfo:
    def __init__(self, name, table):
        self.name = name
        self.rows, self.columns = table.shape
        
    def describe(self):
        return f"{self.name}: {self.rows} строк, {self.columns} столбцов"


def print_table_info(name, table):
    print(f"\n Проверка таблицы: {name}")
    print(table.shape)       
    print(table.columns)     
    print(table.dtypes)      
    print(table.head())      


def read_wells(path):
    table = pd.read_csv(path)
    print_table_info("wells.csv", table)
    return table


def read_layers(path):
    table = pd.read_excel(path, engine="openpyxl")
    print_table_info("layers.xlsx", table)
    return table


def read_pumping_test(path):
    table = pd.read_csv(path, sep="\t")
    print_table_info("pumping_test.txt", table)
    return table


def validate_and_clean_data(wells, layers):
    print("\nПроверка пропусков в wells ")
    print(wells.isna().sum())
    
    wells_clean = wells.dropna(subset=["pressure_mpa"]).copy()
    
    print("\nПроверка трех условий через assert ")
    assert (wells_clean["radius_m"] > 0).all(), "Ошибка: Радиус скважины должен быть больше 0!"
    assert (layers["thickness_m"] > 0).all(), "Ошибка: Толщина слоя должна быть больше 0!"
    assert layers["porosity_fraction"].between(0, 1).all(), "Ошибка: Пористость должна быть в диапазоне от 0 до 1!"
    
    print("Все три проверки assert успешно пройдены!")
    return wells_clean


def calculate_columns(wells_clean):
    print("\nВыполнение расчетов для столбцов")
    wells_clean["pressure_pa"] = wells_clean["pressure_mpa"] * 1_000_000
    wells_clean["pressure_difference_mpa"] = 12 - wells_clean["pressure_mpa"]
    wells_clean["relative_change_percent"] = (wells_clean["pressure_difference_mpa"] / 12) * 100
    return wells_clean


def calculate_numpy_math(wells_clean, pumping_test):
    print("\nИспользование математических функций NumPy")
    theta_rad = wells_clean["azimuth_deg"] * np.pi / 180
    x_check = wells_clean["radius_m"] * np.cos(theta_rad)
    y_check = wells_clean["radius_m"] * np.sin(theta_rad)
    
    assert np.allclose(x_check, wells_clean["x_m"], atol=0.02), "Ошибка: Проверка x_check не пройдена!"
    assert np.allclose(y_check, wells_clean["y_m"], atol=0.02), "Ошибка: Проверка y_check не пройдена!"
    print("Тригонометрическая проверка координат по азимуту выполнена (atol=0.02).")
    
    wells_clean["x_check"] = x_check
    wells_clean["y_check"] = y_check
    wells_clean["log_radius"] = np.log(wells_clean["radius_m"])
    
    pumping_test["decay"] = np.exp(-pumping_test["time_h"] / 36)
    
    first_decay_value = pumping_test["decay"].iloc[0]
    assert np.isclose(first_decay_value, 1.0, atol=0.05), f"Ошибка: Первое значение распада {first_decay_value} не равно ~1!"
    assert (pumping_test["decay"].diff().dropna() <= 0).all(), "Ошибка: Значения последовательности decay не уменьшаются!"
    
    return wells_clean, pumping_test


def calculate_statistics_and_slices(wells_clean, layers):
    print("\nСтатистики и Срезы")
    pressure = wells_clean["pressure_mpa"].to_numpy()
    print(f"1. Первое: {pressure[0]}, Последнее: {pressure[-1]}")
    print(f"2. Первые три: {pressure[:3]}")
    print(f"3. Каждое второе: {pressure[::2]}")
    
    mask_pressure = pressure < pressure.mean()
    print(f"4. Давление ниже среднего: {len(pressure[mask_pressure])}")
    
    mask_distance = wells_clean["radius_m"] > 100
    print(f"5. Скважины дальше 100 м: {len(wells_clean[mask_distance])}")
    return wells_clean


def create_numpy_arrays(pumping_test):
    print("\n Создаём двумерный массив")
    pressure_matrix = pumping_test[["boundary_pressure_mpa", "well_pressure_mpa"]].iloc[:13].to_numpy()
    
    print(pressure_matrix.ndim)   
    print(pressure_matrix.shape)  
    print(pressure_matrix.size)   
    print(pressure_matrix.dtype)  
    
    print("\nРаздел 10: Создаем трехмерный массив")
    experiment_1 = pressure_matrix
    experiment_2 = pressure_matrix + 0.05
    
    pressure_cube = np.stack([experiment_1, experiment_2], axis=0)
    
    print("первая таблица:\n", pressure_cube[0])
    print("вторая таблица:\n", pressure_cube[1])
    print("первая книга первой таблицы:\n", pressure_cube[0, 0])
    print("столбец давления в скважине для приведенных таблиц:\n", pressure_cube[:, :, 1])
    
    return pressure_matrix, pressure_cube
