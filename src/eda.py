# %%
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from src.config import PROCESSED, OUTPUTS

FIG = OUTPUTS / "eda"
FIG.mkdir(parents=True, exist_ok=True)
sns.set_theme(style="whitegrid")


def save(name):
    plt.tight_layout()
    plt.savefig(FIG / f"{name}.png", dpi=120)
    plt.close()


df = pd.read_parquet(PROCESSED / "master.parquet")
ops = df[df.has_ops_detail].copy()

# %% headline numbers
print("orders:", len(df))
print("breach rate:", round(df.sla_breached.mean(), 4))
print(df.status.value_counts())
print("avg delay (breached only):", round(df.loc[df.sla_breached == 1, "delivery_delay_minutes"].mean(), 2))

# %% status mix
df.status.value_counts().plot(kind="bar")
plt.title("Order status")
save("01_status")

# %% breach by hour
df.groupby("hour").sla_breached.mean().plot(marker="o")
plt.title("SLA breach rate by hour")
plt.ylabel("breach rate")
save("02_breach_by_hour")

# %% breach by weekday
df.groupby("dow").sla_breached.mean().plot(kind="bar")
plt.title("SLA breach rate by weekday (0=Mon)")
save("03_breach_by_dow")

# %% worst stores
(df.groupby("store_id").sla_breached.mean()
   .sort_values(ascending=False).head(15).plot(kind="bar"))
plt.title("Top 15 stores by breach rate")
save("04_worst_stores")

# %% delay distribution
sns.histplot(df.loc[df.sla_breached == 1, "delivery_delay_minutes"], bins=50)
plt.title("Delay minutes (breached orders)")
save("05_delay_dist")

# %% daily trend
df.groupby("order_date").sla_breached.mean().plot()
plt.title("Daily breach rate")
save("06_daily_trend")

# %% ops drivers vs breach
cols = ["pick_duration_minutes", "items_missing", "assignment_delay_minutes", "distance_km"]
fig, axes = plt.subplots(2, 2, figsize=(10, 8))
for ax, c in zip(axes.ravel(), cols):
    sns.boxplot(data=ops, x="sla_breached", y=c, ax=ax)
    ax.set_title(c)
save("07_ops_drivers")

# %% correlation
num = ops[["sla_breached", "item_count", "order_value", "promised_minutes", "hour",
           "pick_duration_minutes", "items_missing", "assignment_delay_minutes",
           "distance_km", "stockout_rate", "pending_ratio"]]
sns.heatmap(num.corr(), annot=True, fmt=".2f", cmap="coolwarm")
plt.title("Correlation (orders with ops detail)")
save("08_corr")

print("charts saved in", FIG)