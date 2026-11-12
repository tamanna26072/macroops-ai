import pandas as pd
from src.config import RAW

FILES = ["orders", "inventory", "picking", "delivery", "workforce"]


def load_all() -> dict[str, pd.DataFrame]:
    return {name: pd.read_csv(RAW / f"{name}.csv") for name in FILES}


if __name__ == "__main__":
    for name, df in load_all().items():
        print(name, df.shape)