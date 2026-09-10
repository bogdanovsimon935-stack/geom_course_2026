"""Создаёт учебные исходные данные для ДЗ №1.

Скрипт имитирует результаты опытно-фильтрационных работ.
"""

from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "data" / "raw" / "hw01"


def main() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    radius_m = np.array([5.0, 10.0, 20.0, 40.0, 80.0, 160.0, 320.0, 500.0])
    azimuth_deg = np.array([15.0, 55.0, 95.0, 135.0, 210.0, 250.0, 305.0, 345.0])
    azimuth_rad = np.deg2rad(azimuth_deg)

    wells = pd.DataFrame(
        {
            "well_id": [f"OBS-{number:02d}" for number in range(1, 9)],
            "radius_m": radius_m,
            "azimuth_deg": azimuth_deg,
            "x_m": np.round(radius_m * np.cos(azimuth_rad), 2),
            "y_m": np.round(radius_m * np.sin(azimuth_rad), 2),
            "pressure_mpa": [9.42, 9.83, 10.24, 10.67, np.nan, 11.36, 11.73, 12.00],
            "temperature_c": [13.8, 13.9, 14.1, 14.0, 14.2, 14.3, 14.1, 14.0],
            "quality_flag": ["ok", "ok", "ok", "ok", "missing", "ok", "ok", "boundary"],
        }
    )

    layers = pd.DataFrame(
        {
            "layer": ["A", "B", "C", "D"],
            "top_depth_m": [42.0, 46.0, 53.0, 58.0],
            "bottom_depth_m": [46.0, 53.0, 58.0, 60.0],
            "thickness_m": [4.0, 7.0, 5.0, 2.0],
            "kx_m2": [0.8e-13, 2.1e-13, 1.4e-13, 0.5e-13],
            "ky_m2": [0.5e-13, 1.4e-13, 0.9e-13, 0.3e-13],
            "porosity_fraction": [0.18, 0.24, 0.21, 0.15],
            "compressibility_1_pa": [4.0e-10, 5.2e-10, 4.6e-10, 3.5e-10],
        }
    )

    time_h = np.array([0.0, 1.0, 2.0, 4.0, 8.0, 12.0, 24.0, 36.0, 48.0, 72.0, 96.0, 120.0, 168.0])
    well_pressure_mpa = 12.0 - 4.0 * (1.0 - np.exp(-time_h / 36.0))
    pumping_test = pd.DataFrame(
        {
            "time_h": time_h,
            "boundary_pressure_mpa": np.full(time_h.shape, 12.0),
            "well_pressure_mpa": np.round(well_pressure_mpa, 4),
            "stage": np.where(time_h == 0, "background", "pumping"),
        }
    )

    wells.to_csv(RAW_DIR / "wells.csv", index=False, encoding="utf-8")
    layers.to_excel(RAW_DIR / "layers.xlsx", index=False)
    pumping_test.to_csv(RAW_DIR / "pumping_test.txt", index=False, sep="\t", encoding="utf-8")

    print("Созданы учебные данные:")
    for path in sorted(RAW_DIR.iterdir()):
        print(f"- {path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
