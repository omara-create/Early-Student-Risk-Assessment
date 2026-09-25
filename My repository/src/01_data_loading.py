"""01. Validate and inventory the public OULAD input tables."""
from common import load_oulad, save_csv, set_seed, load_config

def main():
    cfg = load_config()
    set_seed(cfg["random_seed"])
    data = load_oulad()

    rows = []
    for name, df in data.items():
        rows.append({
            "table": name,
            "rows": len(df),
            "columns": len(df.columns),
            "missing_cells": int(df.isna().sum().sum()),
            "duplicate_rows": int(df.duplicated().sum())
        })
    inventory = __import__("pandas").DataFrame(rows)
    save_csv(inventory, "01_data_inventory.csv")
    for name, df in data.items():
        print(f"{name:24s} {df.shape}")

if __name__ == "__main__":
    main()
