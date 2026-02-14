"""
Descriptive statistics and crisis/failure summary for Final data.
- Descriptive stats and histograms for GFDD.SI.04, GFDD.SI.02, GFDD.SI.01
- Evolution graphs: mean by geographical area + 3 outliers per variable
- % of failures (crisis) overall and top 5 countries by crisis count
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns
from pathlib import Path
from scipy.stats import pearsonr
from matplotlib.lines import Line2D

# Path to Final data (in 1. Clean data folder)
SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR.parent / "1. Clean data"
FINAL_DATA_PATH = DATA_DIR / "Final data.xlsx"

# Main variables and column names
MAIN_VARS = ["GFDD.SI.04", "GFDD.SI.02", "GFDD.SI.01"]  # Credit to Deposit, NPL, Z-Score
MAIN_LABELS = {"GFDD.SI.04": "Credit to Deposit (%)", "GFDD.SI.02": "NPL (%)", "GFDD.SI.01": "Z-Score"}
CRISIS_COL = "GFDD.OI.19"
TIME_COL = "Time"
COUNTRY_COL = "Country Name"
NUM_OUTLIERS = 3


def get_country_region(country_name):
    """Map country names to geographic regions."""
    region_mapping = {
        'Albania': 'Europe', 'Andorra': 'Europe', 'Armenia': 'Europe', 'Austria': 'Europe',
        'Azerbaijan': 'Europe', 'Belarus': 'Europe', 'Belgium': 'Europe', 'Bosnia and Herzegovina': 'Europe',
        'Bulgaria': 'Europe', 'Channel Islands': 'Europe', 'Croatia': 'Europe', 'Cyprus': 'Europe',
        'Czech Republic': 'Europe', 'Denmark': 'Europe', 'Estonia': 'Europe', 'Faroe Islands': 'Europe',
        'Finland': 'Europe', 'France': 'Europe', 'Georgia': 'Europe', 'Germany': 'Europe',
        'Gibraltar': 'Europe', 'Greece': 'Europe', 'Greenland': 'Europe', 'Hungary': 'Europe',
        'Iceland': 'Europe', 'Ireland': 'Europe', 'Isle of Man': 'Europe', 'Italy': 'Europe',
        'Kosovo': 'Europe', 'Latvia': 'Europe', 'Liechtenstein': 'Europe', 'Lithuania': 'Europe',
        'Luxembourg': 'Europe', 'Malta': 'Europe', 'Moldova': 'Europe', 'Monaco': 'Europe',
        'Montenegro': 'Europe', 'Netherlands': 'Europe', 'North Macedonia': 'Europe', 'Norway': 'Europe',
        'Poland': 'Europe', 'Portugal': 'Europe', 'Romania': 'Europe', 'Russian Federation': 'Europe',
        'San Marino': 'Europe', 'Serbia': 'Europe', 'Slovak Republic': 'Europe', 'Slovenia': 'Europe',
        'Spain': 'Europe', 'Sweden': 'Europe', 'Switzerland': 'Europe', 'Turkey': 'Europe',
        'Ukraine': 'Europe', 'United Kingdom': 'Europe',
        'Afghanistan': 'Asia', 'Bangladesh': 'Asia', 'Bhutan': 'Asia', 'Brunei Darussalam': 'Asia',
        'Cambodia': 'Asia', 'China': 'Asia', 'Hong Kong SAR, China': 'Asia', 'India': 'Asia',
        'Indonesia': 'Asia', 'Japan': 'Asia', 'Kazakhstan': 'Asia', 'Korea, Dem. Rep.': 'Asia',
        'Korea, Rep.': 'Asia', 'Kyrgyz Republic': 'Asia', 'Lao PDR': 'Asia', 'Macao SAR, China': 'Asia',
        'Malaysia': 'Asia', 'Maldives': 'Asia', 'Mongolia': 'Asia', 'Myanmar': 'Asia',
        'Nepal': 'Asia', 'Pakistan': 'Asia', 'Philippines': 'Asia', 'Singapore': 'Asia',
        'Sri Lanka': 'Asia', 'Taiwan, China': 'Asia', 'Tajikistan': 'Asia', 'Thailand': 'Asia',
        'Timor-Leste': 'Asia', 'Turkmenistan': 'Asia', 'Uzbekistan': 'Asia', 'Vietnam': 'Asia',
        'Bahrain': 'Middle East', 'Iran, Islamic Rep.': 'Middle East', 'Iraq': 'Middle East',
        'Israel': 'Middle East', 'Jordan': 'Middle East', 'Kuwait': 'Middle East', 'Lebanon': 'Middle East',
        'Oman': 'Middle East', 'Qatar': 'Middle East', 'Saudi Arabia': 'Middle East',
        'Syrian Arab Republic': 'Middle East', 'United Arab Emirates': 'Middle East',
        'West Bank and Gaza': 'Middle East', 'Yemen, Rep.': 'Middle East',
        'Antigua and Barbuda': 'Americas', 'Argentina': 'Americas', 'Aruba': 'Americas',
        'Bahamas, The': 'Americas', 'Barbados': 'Americas', 'Belize': 'Americas', 'Bolivia': 'Americas',
        'Brazil': 'Americas', 'British Virgin Islands': 'Americas', 'Canada': 'Americas',
        'Cayman Islands': 'Americas', 'Chile': 'Americas', 'Colombia': 'Americas', 'Costa Rica': 'Americas',
        'Cuba': 'Americas', 'Curaçao': 'Americas', 'Dominica': 'Americas', 'Dominican Republic': 'Americas',
        'Ecuador': 'Americas', 'El Salvador': 'Americas', 'Grenada': 'Americas', 'Guatemala': 'Americas',
        'Guyana': 'Americas', 'Haiti': 'Americas', 'Honduras': 'Americas', 'Jamaica': 'Americas',
        'Mexico': 'Americas', 'Nicaragua': 'Americas', 'Panama': 'Americas', 'Paraguay': 'Americas',
        'Peru': 'Americas', 'Puerto Rico': 'Americas', 'St. Kitts and Nevis': 'Americas',
        'St. Lucia': 'Americas', 'St. Vincent and the Grenadines': 'Americas', 'Suriname': 'Americas',
        'Trinidad and Tobago': 'Americas', 'United States': 'Americas', 'Uruguay': 'Americas',
        'Venezuela, RB': 'Americas', 'Virgin Islands (U.S.)': 'Americas',
        'Algeria': 'Africa', 'Angola': 'Africa', 'Benin': 'Africa', 'Botswana': 'Africa',
        'Burkina Faso': 'Africa', 'Burundi': 'Africa', 'Cabo Verde': 'Africa', 'Cameroon': 'Africa',
        'Central African Republic': 'Africa', 'Chad': 'Africa', 'Comoros': 'Africa',
        'Congo, Dem. Rep.': 'Africa', 'Congo, Rep.': 'Africa', "Cote d'Ivoire": 'Africa',
        'Djibouti': 'Africa', 'Egypt, Arab Rep.': 'Africa', 'Equatorial Guinea': 'Africa',
        'Eritrea': 'Africa', 'Eswatini': 'Africa', 'Ethiopia': 'Africa', 'Gabon': 'Africa',
        'Gambia, The': 'Africa', 'Ghana': 'Africa', 'Guinea': 'Africa', 'Guinea-Bissau': 'Africa',
        'Kenya': 'Africa', 'Lesotho': 'Africa', 'Liberia': 'Africa', 'Libya': 'Africa',
        'Madagascar': 'Africa', 'Malawi': 'Africa', 'Mali': 'Africa', 'Mauritania': 'Africa',
        'Mauritius': 'Africa', 'Morocco': 'Africa', 'Mozambique': 'Africa', 'Namibia': 'Africa',
        'Niger': 'Africa', 'Nigeria': 'Africa', 'Rwanda': 'Africa', 'Senegal': 'Africa',
        'Seychelles': 'Africa', 'Sierra Leone': 'Africa', 'Somalia': 'Africa', 'South Africa': 'Africa',
        'South Sudan': 'Africa', 'Sudan': 'Africa', 'São Tomé and Principe': 'Africa', 'Tanzania': 'Africa',
        'Togo': 'Africa', 'Tunisia': 'Africa', 'Uganda': 'Africa', 'Zambia': 'Africa', 'Zimbabwe': 'Africa',
        'Australia': 'Oceania', 'Fiji': 'Oceania', 'French Polynesia': 'Oceania', 'Guam': 'Oceania',
        'Kiribati': 'Oceania', 'Marshall Islands': 'Oceania', 'Micronesia, Fed. Sts.': 'Oceania',
        'Nauru': 'Oceania', 'New Caledonia': 'Oceania', 'New Zealand': 'Oceania', 'Palau': 'Oceania',
        'Papua New Guinea': 'Oceania', 'Samoa': 'Oceania', 'Solomon Islands': 'Oceania',
        'Tonga': 'Oceania', 'Tuvalu': 'Oceania', 'Vanuatu': 'Oceania',
        'East Asia & Pacific (IDA total)': 'Asia', 'Europe & Central Asia (IDA total)': 'Europe',
        'Latin America & Caribbean (IDA total)': 'Americas', 'Middle East & North Africa (IDA total)': 'Middle East',
        'South Asia (IDA total)': 'Asia', 'Sub-Saharan Africa (IDA total)': 'Africa',
        'IBRD only': 'Other', 'IDA total': 'Other',
        'Least developed countries: UN classification': 'Other',
        'Small island developing states: UN classification': 'Other',
        'Bermuda': 'Americas', 'Sint Maarten (Dutch part)': 'Americas', 'Turks and Caicos Islands': 'Americas',
    }
    return region_mapping.get(country_name, "Other")


def calculate_mean_time_series(df, country_group, col):
    """Mean time series for a group of countries. Returns (times, means)."""
    all_times = set()
    country_data_dict = {}
    for country in country_group:
        country_data = df[df[COUNTRY_COL] == country].sort_values(TIME_COL)
        if len(country_data) > 0:
            times = country_data[TIME_COL].values
            values = country_data[col].values
            all_times.update(times)
            country_data_dict[country] = dict(zip(times, values))
    sorted_times = sorted(all_times)
    mean_values = []
    valid_times = []
    for time_point in sorted_times:
        values_at_time = []
        for country in country_group:
            if country in country_data_dict and time_point in country_data_dict[country]:
                val = country_data_dict[country][time_point]
                if pd.notna(val):
                    values_at_time.append(float(val))
        if len(values_at_time) > 0:
            mean_values.append(np.mean(values_at_time))
            valid_times.append(time_point)
    return np.array(valid_times), np.array(mean_values)


def group_countries_by_region(df, countries, col, max_outliers=3):
    """Group countries by region and return top max_outliers by deviation from regional mean."""
    region_groups = {}
    for country in countries:
        region = get_country_region(country)
        region_groups.setdefault(region, []).append(country)
    country_deviations = []
    for region, region_countries in region_groups.items():
        if len(region_countries) < 2:
            continue
        times, region_mean = calculate_mean_time_series(df, region_countries, col)
        if len(times) == 0:
            continue
        for country in region_countries:
            country_data = df[df[COUNTRY_COL] == country].sort_values(TIME_COL)
            if len(country_data) == 0:
                continue
            country_values, region_values = [], []
            for time_point in times:
                cv = country_data[country_data[TIME_COL] == time_point][col].values
                if len(cv) > 0 and pd.notna(cv[0]):
                    country_values.append(float(cv[0]))
                    idx = np.where(times == time_point)[0]
                    if len(idx) > 0:
                        region_values.append(float(region_mean[idx[0]]))
            if len(country_values) < 3:
                continue
            try:
                ca, ra = np.array(country_values), np.array(region_values)
                if np.std(ca) > 0 and np.std(ra) > 0:
                    corr, _ = pearsonr(ca, ra)
                    mad = np.mean(np.abs(ca - ra))
                    rs = np.std(ra)
                    norm_dev = mad / rs if rs > 0 else mad
                    score = (1 - corr) + norm_dev
                    if not np.isnan(corr):
                        country_deviations.append((country, corr, mad, score))
            except Exception:
                pass
    country_deviations.sort(key=lambda x: x[3], reverse=True)
    outlier_countries = [c for c, *_ in country_deviations[:max_outliers]]
    return region_groups, outlier_countries


def load_final_data():
    return pd.read_excel(FINAL_DATA_PATH)


def failures_overall(df):
    """% of failures (crisis) overall."""
    if CRISIS_COL not in df.columns:
        return pd.DataFrame()
    valid = df[CRISIS_COL].dropna()
    total = len(valid)
    crisis_count = (valid == 1).sum()
    pct = (crisis_count / total * 100) if total > 0 else 0.0
    return pd.DataFrame({
        "Metric": ["Crisis (failures)", "No crisis", "Total (valid)", "% failures"],
        "Value": [int(crisis_count), int((valid == 0).sum()), total, round(pct, 2)],
    })


def top5_countries_crisis(df):
    """Top 5 countries by number of crises."""
    if CRISIS_COL not in df.columns or COUNTRY_COL not in df.columns:
        return pd.DataFrame()
    g = df.groupby(COUNTRY_COL)[CRISIS_COL].sum().sort_values(ascending=False).head(5)
    return g.reset_index().rename(columns={CRISIS_COL: "Crisis_count"})


def plot_histograms(df):
    """Histograms for the three main variables."""
    vars_present = [v for v in MAIN_VARS if v in df.columns]
    if not vars_present:
        return
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for ax, col in zip(axes, vars_present):
        data = df[col].dropna()
        ax.hist(data, bins=30, edgecolor="black", alpha=0.7)
        ax.set_xlabel(MAIN_LABELS.get(col, col))
        ax.set_ylabel("Frequency")
        ax.set_title(MAIN_LABELS.get(col, col))
        ax.grid(True, alpha=0.3)
    plt.tight_layout()
    out_path = SCRIPT_DIR / "descriptive_histograms.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Histograms saved: {out_path}")


def plot_evolution_by_variable(df, col):
    """One evolution graph: mean by geographical area + 3 outliers."""
    if col not in df.columns or TIME_COL not in df.columns or COUNTRY_COL not in df.columns:
        return
    countries = sorted(df[COUNTRY_COL].unique())
    region_groups, outlier_countries = group_countries_by_region(df, countries, col, max_outliers=NUM_OUTLIERS)
    region_order = ["Europe", "Asia", "Americas", "Africa", "Middle East", "Oceania", "Other"]
    region_colors = plt.cm.tab10(np.linspace(0, 1, 10))
    region_color_map = {r: region_colors[i % len(region_colors)] for i, r in enumerate(region_order)}
    outlier_colors_list = ["#FF0000", "#0000FF", "#00FF00", "#FF00FF", "#FFA500"]
    outlier_colors = [mcolors.to_rgba(c) for c in outlier_colors_list]

    fig, ax = plt.subplots(figsize=(12, 6))
    for region in region_order:
        if region not in region_groups or not region_groups[region]:
            continue
        times, means = calculate_mean_time_series(df, region_groups[region], col)
        if len(times) > 0:
            ax.plot(times, means, marker="o", markersize=4, linewidth=2, alpha=0.8,
                    color=region_color_map.get(region, "gray"), label=f"{region} (mean)")
    for i, country in enumerate(outlier_countries):
        sub = df[df[COUNTRY_COL] == country].sort_values(TIME_COL)
        if len(sub) > 0:
            ax.plot(sub[TIME_COL], sub[col], marker="s", markersize=4, linewidth=1.5, linestyle="--",
                    color=outlier_colors[i % len(outlier_colors)], label=country, alpha=0.9)
    ax.set_xlabel("Time (Year)", fontsize=11)
    ax.set_ylabel(MAIN_LABELS.get(col, col), fontsize=11)
    ax.set_title(f"Evolution of {MAIN_LABELS.get(col, col)}\nby Geographic Area (mean) and {NUM_OUTLIERS} Outliers", fontsize=12)
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    suffix = {"GFDD.SI.04": "SI04", "GFDD.SI.02": "SI02", "GFDD.SI.01": "SI01"}.get(col, col)
    out_path = SCRIPT_DIR / f"evolution_{suffix}.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Evolution graph saved: {out_path}")


def main():
    df = load_final_data()
    print("Loaded:", FINAL_DATA_PATH)
    print(f"Shape: {df.shape[0]} rows, {df.shape[1]} columns\n")

    # Descriptive statistics for main variables
    vars_present = [v for v in MAIN_VARS if v in df.columns]
    if vars_present:
        desc = df[vars_present].describe().round(4)
        print("=" * 60)
        print("DESCRIPTIVE STATISTICS (main variables)")
        print("=" * 60)
        print(desc)
        # Save descriptive stats table
        desc_path = SCRIPT_DIR / "descriptive_stats_main_variables.xlsx"
        desc.to_excel(desc_path)
        print(f"\nSaved: {desc_path}")

    # Histograms
    plot_histograms(df)

    # % failures overall
    print("\n" + "=" * 60)
    print("% FAILURES (CRISIS) OVERALL")
    print("=" * 60)
    print(failures_overall(df).to_string(index=False))

    # Top 5 countries by crisis count
    print("\n" + "=" * 60)
    print("TOP 5 COUNTRIES WITH THE MOST CRISES")
    print("=" * 60)
    print(top5_countries_crisis(df).to_string(index=False))

    # Evolution graphs: mean by region + 3 outliers per variable
    print("\n" + "=" * 60)
    print("EVOLUTION GRAPHS (mean by geographical area + 3 outliers)")
    print("=" * 60)
    for col in MAIN_VARS:
        if col in df.columns:
            plot_evolution_by_variable(df, col)

    print("\nDone.")


if __name__ == "__main__":
    main()
