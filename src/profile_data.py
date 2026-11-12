import pandas as pd
from src.data_loader import load_all

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 50)

tables = load_all()

for name, df in tables.items():
    print(f"\n===== {name} {df.shape} =====")
    print(df.dtypes.to_string())
    nulls = df.isna().sum()
    print("nulls:\n", nulls[nulls > 0].to_string() or "none")
    print("dup rows:", df.duplicated().sum())
    print(df.head(3).to_string())


def check(left, lcol, right, rcol, ln, rn):
    miss = ~left[lcol].isin(right[rcol])
    print(f"{ln}.{lcol} -> {rn}.{rcol}: {miss.sum()} orphan ({miss.mean():.2%})")


print("\n===== JOIN CHECKS =====")
o = tables["orders"]
check(o, "order_id", tables["picking"], "order_id", "orders", "picking")
check(o, "order_id", tables["delivery"], "order_id", "orders", "delivery")
check(o, "store_id", tables["inventory"], "store_id", "orders", "inventory")
check(o, "store_id", tables["workforce"], "store_id", "orders", "workforce")