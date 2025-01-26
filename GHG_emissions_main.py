import os
import logging
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

# ------------------------------------------------------------------------------
# DIRECTORY SETUP
# ------------------------------------------------------------------------------
BASE_DIR = Path("D:/GHG")  # Adjust if needed
DATA_DIR = BASE_DIR / "data"
CHARTS_DIR = BASE_DIR / "charts"
LOGS_DIR = BASE_DIR / "logs"

# Create all necessary folders if they do not exist
for dir_path in [DATA_DIR, DATA_DIR, CHARTS_DIR, LOGS_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# ------------------------------------------------------------------------------
# LOGGING CONFIGURATION
# ------------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler(LOGS_DIR / "ghg_report.log"),  # Log file in logs/ folder
        logging.StreamHandler()                            # Also print logs to console
    ]
)

# ------------------------------------------------------------------------------
# FUNCTION DEFINITIONS
# ------------------------------------------------------------------------------
def read_ghg_data(file_path: Path) -> pd.DataFrame:
    """
    Reads GHG data from the provided Excel file, specifically
    from the 'GHG_totals_by_country' sheet.
    """
    try:
        logging.info("Reading GHG emissions data from Excel...")
        ghg_data = pd.read_excel(file_path, sheet_name="GHG_totals_by_country")

        # Clean column names (strip any surrounding spaces)
        ghg_data.columns = [col.strip() for col in ghg_data.columns]

        # Ensure 'Country' column has valid entries
        ghg_data = ghg_data.dropna(subset=["Country"])

        logging.info("Successfully read GHG data.")
        return ghg_data
    except Exception as e:
        logging.error(f"Failed to read GHG data: {e}")
        return None


def read_countries_by_continent(file_path: Path) -> dict:
    """
    Reads a CSV file that maps countries to continents and
    organizes them into a dictionary: {Continent: [list of countries], ...}.
    """
    try:
        logging.info("Reading country-continent mapping from CSV...")
        countries_df = pd.read_csv(file_path)

        # Group by Continent and create a list of countries for each
        mapping = countries_df.groupby('Continent')['Country'].apply(list).to_dict()

        logging.info("Successfully read country-continent mapping.")
        return mapping
    except Exception as e:
        logging.error(f"Failed to read continent-country mapping: {e}")
        return None


def plot_stacked_bar_chart(ghg_data: pd.DataFrame, continent_mapping: dict, output_path: Path):
    """
    Creates a stacked bar chart of global GHG emissions by country and continent.
    Only includes countries that contribute >= 0.5% to the GLOBAL TOTAL in 2023.
    """
    try:
        logging.info("Creating stacked bar chart of GHG emissions...")

        # Locate the row for "GLOBAL TOTAL" to get total 2023 GHG
        global_total_row = ghg_data[ghg_data['Country'] == 'GLOBAL TOTAL']
        if global_total_row.empty:
            logging.warning("No row found for 'GLOBAL TOTAL'. Chart may be inaccurate.")
            return
        global_total_2023 = global_total_row[2023].values[0]

        # Remove the GLOBAL TOTAL row from main data
        ghg_data = ghg_data[ghg_data['Country'] != 'GLOBAL TOTAL']

        # Calculate each country's % share in 2023 and filter
        ghg_data['Percentage'] = (ghg_data[2023] / global_total_2023) * 100
        ghg_data = ghg_data[ghg_data['Percentage'] >= 0.5]

        # Build a dict: {Continent: Series of country 2023 emissions}
        continent_data = {}
        for continent, countries in continent_mapping.items():
            subset = ghg_data[ghg_data['Country'].isin(countries)]
            continent_data[continent] = subset.set_index('Country')[2023]

        # Combine into one DataFrame with continents as rows, countries as columns
        stacked_data = pd.DataFrame(continent_data).fillna(0).T

        # Plot
        plt.figure(figsize=(14, 8))
        ax = stacked_data.plot(kind='bar', stacked=True, figsize=(14, 8), width=0.8)
        ax.set_title("Global GHG Emissions by Country and Continent (2023)", fontsize=16)
        ax.set_ylabel("GHG Emissions (MtCO₂)", fontsize=12)
        ax.set_xlabel("")

        # Legend settings
        plt.legend(loc="upper left", bbox_to_anchor=(1, 1))
        plt.tight_layout()

        # Save the plot
        plt.savefig(output_path, dpi=300)
        plt.close()

        logging.info(f"Stacked bar chart created and saved to: {output_path}")

    except Exception as e:
        logging.error(f"Failed to create stacked bar chart: {e}")


# ------------------------------------------------------------------------------
# MAIN
# ------------------------------------------------------------------------------
def main():
    """
    Main entry point:
      1. Read GHG data from archive/EDGAR_2024_GHG_booklet_2024.xlsx
      2. Read country-continent mapping from data/Countries_by_continents.csv
      3. Create a stacked bar chart for 2023 GHG emissions by continent
    """
    ghg_file_path = DATA_DIR / "EDGAR_2024_GHG_booklet_2024.xlsx"
    csv_file_path = DATA_DIR / "Countries_by_continents.csv"
    chart_output_path = CHARTS_DIR / "stacked_bar_chart.png"

    # 1. Read GHG data
    ghg_data = read_ghg_data(ghg_file_path)
    if ghg_data is None:
        logging.error("No GHG data. Exiting.")
        return

    # 2. Read country-continent CSV
    continent_mapping = read_countries_by_continent(csv_file_path)
    if continent_mapping is None:
        logging.error("No continent mapping. Exiting.")
        return

    # 3. Plot & save the stacked bar chart
    plot_stacked_bar_chart(ghg_data, continent_mapping, chart_output_path)


if __name__ == "__main__":
    main()
