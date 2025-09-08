import pandas as pd
import plotly.graph_objects as go
from pathlib import Path

DATA_PATH = Path(__file__).with_name("agriculture_crop_yield.csv")

_df = pd.read_csv(DATA_PATH)
agg = (
    _df.groupby(["State", "Crop_Type"], as_index=False)["Total_Revenue"]
    .sum()
    .rename(columns={"Total_Revenue": "Revenue"})
)

state_totals = agg.groupby("State")["Revenue"].sum().reset_index()

labels, parents, values = [], [], []

for _, r in state_totals.iterrows():
    labels.append(r["State"])
    parents.append("")
    values.append(float(r["Revenue"]))

for _, r in agg.iterrows():
    labels.append(r["Crop_Type"])
    parents.append(r["State"])
    values.append(float(r["Revenue"]))

fig = go.Figure(go.Sunburst(labels=labels, parents=parents, values=values, branchvalues="total"))
fig.write_image(Path(__file__).with_name("sunburst.png"))
print("Saved sunburst.png")
