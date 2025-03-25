import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import geopandas as gpd
from shapely.geometry import Point
import geodatasets
import folium
from folium.plugins import HeatMap


df = pd.read_csv("Electric_Vehicle_Population_Data.csv")
world = gpd.read_file(geodatasets.get_path("naturalearth.land"))

print(df.head())
print(df.info())
print(df.isnull().sum())

df.drop(columns=["VIN (1-10)", "DOL Vehicle ID"], inplace=True)

df["Electric Vehicle Type"] = df["Electric Vehicle Type"].replace({
    "Battery Electric Vehicle (BEV)": "BEV",
    "Plug-in Hybrid Electric Vehicle (PHEV)": "PHEV"
})

df["Model Year"] = pd.to_numeric(df["Model Year"], errors="coerce")

df["Electric Range"] = df["Electric Range"].fillna(df["Electric Range"].median())
df["Base MSRP"] = df["Base MSRP"].fillna(df["Base MSRP"].median())

sns.set_style("whitegrid")
plt.figure(figsize=(12, 6))
sns.histplot(df["Model Year"], bins=20, kde=True, color="blue")
plt.xlabel("Model Year")
plt.ylabel("Count")
plt.title("Distribution of Electric Vehicles by Model Year")
plt.savefig("ev_distribution_by_year.png")
plt.show()
print(df["Model Year"].describe())

plt.figure(figsize=(8, 6))
sns.countplot(data=df, x="Electric Vehicle Type", hue="Electric Vehicle Type", palette="viridis", legend=False)
plt.xlabel("Electric Vehicle Type")
plt.ylabel("Count")
plt.title("Comparison of BEV vs PHEV")
plt.savefig("bev_vs_phev_comparison.png")
plt.show()
print(df["Electric Vehicle Type"].value_counts())

plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x="Base MSRP", y="Electric Range", hue="Electric Vehicle Type", alpha=0.7)
plt.xscale("log")
plt.xlabel("Base MSRP (log scale)")
plt.ylabel("Electric Range")
plt.title("Relationship Between Base MSRP and Electric Range")
plt.legend(title="Vehicle Type")
plt.savefig("msrp_vs_range.png")
plt.show()
print(df[["Base MSRP", "Electric Range"]].describe())

m = folium.Map(location=[47.7511, -120.7401], zoom_start=7)

if "Latitude" in df.columns and "Longitude" in df.columns:
    heat_data = df[["Latitude", "Longitude"]].dropna().values.tolist()
    HeatMap(heat_data, radius=10, blur=15).add_to(m)

m.save("ev_distribution_map.html")

plt.figure(figsize=(12, 6))
top_makes = df["Make"].value_counts().head(10)
print(top_makes)

sns.barplot(x=top_makes.index, y=top_makes.values, hue=top_makes.index, dodge=False, palette="Blues_r")
plt.xlabel("Manufacturer")
plt.ylabel("Vehicle Count")
plt.title("Top 10 Electric Vehicle Manufacturers")
plt.xticks(rotation=45)
plt.legend([],[], frameon=False)
plt.savefig("top_makes_distribution.png")
plt.show()

plt.figure(figsize=(12, 6))
sns.boxplot(data=df, x="Model Year", y="Electric Range", hue="Model Year", dodge=False, palette="coolwarm")
plt.xlabel("Model Year")
plt.ylabel("Electric Range (miles)")
plt.title("Electric Range Variation Over the Years")
plt.xticks(rotation=45)
plt.legend([],[], frameon=False)
plt.savefig("electric_range_over_years.png")
plt.show()
print(df.groupby("Model Year")["Electric Range"].describe())