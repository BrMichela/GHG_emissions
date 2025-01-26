"""
Created on Wed Jun 12 11:29:49 2024
@author: brondim
Updated for enhanced charting and reporting functionality.
"""
import os
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
import datetime
import numpy as np
import logging
from matplotlib.patches import Patch
from matplotlib.cm import get_cmap
from matplotlib.colors import to_hex
import requests
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=pd.errors.SettingWithCopyWarning)


# =============================================================================
# 1. DEFINE FOLDER PATHS AND BASIC CONFIG
# =============================================================================

# Set base_path dynamically to the current script's location or allow an override
base_path = os.getenv("GHG_BASE_PATH", Path(__file__).resolve().parent)

# Define paths for data and charts
archive_path = os.path.join(base_path, "data")
charts_path = os.path.join(base_path, "charts")

# Ensure these directories exist
os.makedirs(archive_path, exist_ok=True)
os.makedirs(charts_path, exist_ok=True)

print(f"Base path: {base_path}")
print(f"Data directory: {archive_path}")
print(f"Charts directory: {charts_path}")


# =============================================================================
# 2. UTILITY FUNCTIONS
# =============================================================================

# Set up the Chrome driver
chrome_driver_path = "C:\\Mambaforge\\Scripts\\chromedriver.exe"
service = Service(executable_path=chrome_driver_path)

# Configure Chrome options for Selenium
options = Options()
options.add_argument("--window-position=-10000,0")
options.add_experimental_option(
    "prefs", {
        "download.prompt_for_download": False,
        "download.directory_upgrade": True,
        "safebrowsing_for_trusted_sources_enabled": False,
        "safebrowsing.enabled": False,
        "download.default_directory": archive_path,  # Set the default download directory
    }
)
options.add_argument("--disable-gpu")
options.add_argument("--disable-software-rasterizer")


def download_ghg_file():
    """
    Downloads the GHG data file for the current year if it does not already exist.
    """
    browser = None  # Initialize browser to None
    try:
        # Get the current year
        current_year = datetime.datetime.now().year

        # Adjust the year for the download URL (default to current year, or use the previous year if current year is not available)
        year_in_url = current_year if current_year <= 2024 else current_year - 1

        # Construct the download URL and file path
        file_url = f"https://edgar.jrc.ec.europa.eu/booklet/EDGAR_{year_in_url}_GHG_booklet_{year_in_url}.xlsx"
        file_path = os.path.join(archive_path, f"EDGAR_{year_in_url}_GHG_booklet_{year_in_url}.xlsx")

        # Check if the file already exists
        if os.path.exists(file_path):
            print(f"The GHG file for {year_in_url} already exists at {file_path}. Skipping download.")
            return

        # Set up the browser to download
        browser = webdriver.Chrome(service=service, options=options)
        browser.set_window_size(1024, 600)
        browser.maximize_window()
        browser.get(file_url)

        # Wait for the download to complete
        import time
        time.sleep(10)  # Adjust delay as needed
        print(f"GHG file downloaded successfully and saved to {file_path}.")
    except Exception as e:
        print(f"An error occurred while downloading the GHG file: {e}")
    finally:
        if browser is not None:
            browser.quit()

def download_class_file():
    """
    Automates the download of the World Bank CLASS.xlsx file if it does not already exist.
    """
    browser = None  # Initialize browser to None
    try:
        # Define file URL and target file path
        page_url = "https://datahelpdesk.worldbank.org/knowledgebase/articles/906519"
        download_link_xpath = "//a[contains(@href, 'ResourceDownload?resource_unique_id=DR0090755')]"
        file_path = os.path.join(archive_path, "CLASS.xlsx")

        # Check if the file already exists
        if os.path.exists(file_path):
            print(f"The World Bank CLASS.xlsx file already exists at {file_path}. Skipping download.")
            return

        # Set up the browser
        browser = webdriver.Chrome(service=service, options=options)
        browser.get(page_url)

        # Wait for the page to load and locate the download link
        browser.implicitly_wait(10)  # Adjust wait time as needed
        download_link = browser.find_element("xpath", download_link_xpath)

        # Click the download link
        print("Clicking download link...")
        download_link.click()

        # Wait for the file to appear in the directory
        def wait_for_file(file_path, timeout=30):
            import time
            start_time = time.time()
            while not os.path.exists(file_path):
                if time.time() - start_time > timeout:
                    raise FileNotFoundError(f"Timeout: {file_path} was not created.")
                time.sleep(1)
            print(f"File found: {file_path}")

        wait_for_file(file_path)

        print(f"CLASS.xlsx file downloaded successfully and saved to {file_path}.")
    except Exception as e:
        print(f"An error occurred while downloading the CLASS.xlsx file: {e}")
    finally:
        if browser is not None:
            browser.quit()

def read_world_bank_data():
    """
    Reads the manually downloaded World Bank 'CLASS.xlsx' from `archive_path`.
    Returns a DataFrame if successful, otherwise None.
    """
    try:
        wb_file = os.path.join(archive_path, "CLASS.xlsx")
        wb_data = pd.read_excel(wb_file)
        print("World Bank data loaded.")
        return wb_data
    except Exception as e:
        print(f"Error reading CLASS.xlsx: {e}")
        return None

def read_ghg_data():
    """
    Reads the 'GHG_totals_by_country' sheet from the EDGAR GHG Excel file.
    Returns a DataFrame if successful, otherwise None.
    """
    try:
        ghg_file_path = os.path.join(archive_path, "EDGAR_2024_GHG_booklet_2024.xlsx")
        df_ghg = pd.read_excel(ghg_file_path, sheet_name="GHG_totals_by_country")
        print("GHG totals data loaded.")
        return df_ghg
    except Exception as e:
        print(f"Error reading GHG data: {e}")
        return None

def read_ghg_per_capita_data():
    """
    Reads the 'GHG_per_capita_by_country' sheet from the EDGAR GHG Excel file.
    Returns a DataFrame if successful, otherwise None.
    """
    try:
        ghg_file_path = os.path.join(archive_path, "EDGAR_2024_GHG_booklet_2024.xlsx")
        df_ghg_capita = pd.read_excel(ghg_file_path, sheet_name="GHG_per_capita_by_country")
        print("GHG per capita data loaded.")
        return df_ghg_capita
    except Exception as e:
        print(f"Error reading GHG per capita data: {e}")
        return None

# Function to read the GHG emissions data
def read_ghg_continents():
    """
    Reads the 'GHG_totals_by_country' sheet from the EDGAR GHG Excel file,
    processes the data, and returns a sorted DataFrame.
    """
    try:
        # Path to the GHG data Excel file
        ghg_file_path = os.path.join(archive_path, "EDGAR_2024_GHG_booklet_2024.xlsx")

        # Ensure the file exists
        if not os.path.exists(ghg_file_path):
            print(f"File not found: {ghg_file_path}")
            return None

        # Read the specific sheet
        ghg_data = pd.read_excel(ghg_file_path, sheet_name="GHG_totals_by_country")

        # Clean column names by stripping spaces
        ghg_data.columns = [col.strip() if isinstance(col, str) else col for col in ghg_data.columns]

        # Drop rows with NaN values
        ghg_data = ghg_data.dropna()

        # Drop specific rows
        ghg_data = ghg_data[
            (ghg_data['Country'] != 'EU27') & (ghg_data['Country'] != 'International Shipping')
        ]

        # Ensure the column for sorting (e.g., 2023) exists
        if 2023 not in ghg_data.columns:
            print("Year 2023 column not found in data.")
            return None

        # Sort by 2023 emissions
        ghg_data_sorted = ghg_data.sort_values(by=2023, ascending=False)

        print("GHG data read and processed successfully.")
        return ghg_data_sorted

    except Exception as e:
        print(f"An error occurred while reading GHG data: {e}")
        return None


def read_countries_by_continent(csv_path):
    """
    Reads a CSV file named 'Countries by continents.csv' from `archive_path`.
    The CSV file must contain columns 'Continent' and 'Country'.
    Returns a dictionary grouped by continent or None if an error occurs.
    """
    csv_filename = "Countries by continents.csv"
    csv_path = os.path.join(archive_path, csv_filename)

    # Check if the file exists
    if not os.path.exists(csv_path):
        print(f"File '{csv_filename}' does not exist in {archive_path}.")
        return None

    try:
        # Read the CSV file
        countries_df = pd.read_csv(csv_path)

        # Ensure required columns exist
        if 'Continent' not in countries_df.columns or 'Country' not in countries_df.columns:
            print(f"CSV file must contain 'Continent' and 'Country' columns.")
            return None

        # Group by continent and return as dictionary
        countries_by_continent = (
            countries_df.groupby('Continent')['Country']
            .apply(list)
            .to_dict()
        )
        print(f"Countries grouped by continent successfully.")
        return countries_by_continent

    except Exception as e:
        print(f"Error reading 'Countries by continents.csv': {e}")
        return None


# =============================================================================
# 3. CHART FUNCTIONS
# =============================================================================

def create_chart_1(df_ghg):
    try:
        # Define country groups
        euro_area = [
            "Austria", "Belgium", "Cyprus", "Estonia", "Finland", "France",
            "Germany", "Greece", "Ireland", "Italy", "Latvia", "Lithuania",
            "Luxembourg", "Malta", "Netherlands", "Portugal", "Slovakia",
            "Slovenia", "Spain", "Croatia"
        ]
        
        # Assign region labels
        df_ghg["Region"] = df_ghg["Country"].apply(
            lambda x: "Euro area" if x in euro_area else (
                "European Union" if x == "EU27" else (
                    "World" if x == "GLOBAL TOTAL" else None
                )
            )
        )
        
        # Melt data
        year_columns = df_ghg.columns[2:]
        data_melted = df_ghg.melt(
            id_vars=["Country", "Region"],
            var_name="Year",
            value_name="GHG Emissions"
        )
        data_melted["Year"] = pd.to_numeric(data_melted["Year"], errors="coerce")
        
        # Group and pivot
        grouped_data = (
            data_melted
            .groupby(["Year", "Region"])["GHG Emissions"]
            .sum()
            .reset_index()
        )
        pivot_data = grouped_data.pivot(index="Year", columns="Region", values="GHG Emissions")
        
        # Plot
        fig, ax = plt.subplots(figsize=(14, 8))
        fig.patch.set_facecolor('#E8E8E8')  # Set figure background color
        ax.set_facecolor('#E8E8E8')         # Set plot background color
        
        pivot_data.plot(
            ax=ax,
            marker='o',
            linewidth=2,
            cmap="tab10",
            markersize=6
        )
        ax.set_yscale('log')
        plt.title("Chart 1: Evolution of GHG Emissions (Euro Area, EU27, and World)", fontsize=20, fontweight='bold', fontname='Arial')
        plt.xlabel("Year", fontsize=14, fontname='Arial')
        plt.ylabel("GHG Emissions (MtCO₂e)", fontsize=14, fontname='Arial')
        plt.xticks(fontsize=12, fontname='Arial')
        plt.yticks(fontsize=12, fontname='Arial')
        plt.grid(True, linestyle='--', alpha=0.6)
        legend = plt.legend(title="Region", fontsize=12, title_fontsize=13,
                            loc="upper left", bbox_to_anchor=(1, 1))
        legend.get_frame().set_facecolor('#E8E8E8')  # Match the grey background
        legend.get_frame().set_edgecolor('black')    # Add a black border

        plt.tight_layout()
        
        chart_path = os.path.join(charts_path, "chart_1.png")
        plt.savefig(chart_path, dpi=300, bbox_inches='tight', facecolor=fig.get_facecolor())
        plt.close()
        
        print(f"Chart 1 saved to: {chart_path}")
        return chart_path
    except Exception as e:
        print(f"Error creating Chart 1: {e}")
        return None


def process_and_reshape_ghg_data(ghg_data, world_bank_data):
    """
    Merges GHG per-capita data with World Bank income group data.
    Returns a pivoted (year vs. income group) DataFrame of average GHG emissions per capita.
    """
    try:
        # Reshape GHG data from wide to long
        melted = ghg_data.melt(
            id_vars=['Country'],
            var_name='Year',
            value_name='GHG emissions per capita'
        )

        # Filter out invalid year rows
        melted = melted[pd.to_numeric(melted['Year'], errors='coerce').notnull()]
        melted['Year'] = melted['Year'].astype(int)
        
        # Merge with WB data
        merged_data = pd.merge(
            melted,
            world_bank_data[['Economy', 'Income group']],
            left_on='Country',
            right_on='Economy',
            how='inner'
        )
        
        # Group and pivot
        grouped = (
            merged_data
            .groupby(['Income group', 'Year'])['GHG emissions per capita']
            .mean()
            .unstack('Income group')
        )
        return grouped
    except Exception as e:
        print(f"Error processing GHG data: {e}")
        return None

def create_chart_2(aggregated_data):
    try:
        # Create figure and axes objects explicitly
        fig, ax = plt.subplots(figsize=(14, 8))
        
        # Set grey background
        fig.patch.set_facecolor('#E8E8E8')  # Figure background
        ax.set_facecolor('#E8E8E8')         # Plot background
        
        # Plot your data
        aggregated_data.plot(
            kind='line',
            marker='o',
            cmap="tab20",
            markersize=5,
            ax=ax
        )

        # Title and axis labels
        plt.title('Chart 2: GHG Emissions per Capita Over Time by Income Group',
                  fontsize=20, fontweight='bold', fontname='Arial')
        plt.xlabel('Year', fontsize=14, fontname='Arial')
        plt.ylabel('Average GHG Emissions per Capita (CO₂e)', fontsize=14, fontname='Arial')

        # Use a log scale if desired
        plt.yscale('log')

        # Make grid lines white for contrast
        plt.grid(color='white', linestyle='-', linewidth=1, alpha=0.7)

        # Rotate x labels and keep them readable
        plt.xticks(fontsize=12, rotation=45, fontname='Arial')
        legend = plt.legend(title="Income Groups", fontsize=12, title_fontsize=13,
                            loc="upper left", bbox_to_anchor=(1, 1))
        legend.get_frame().set_facecolor('#E8E8E8')  # Match the grey background
        legend.get_frame().set_edgecolor('black')    # Add a black border

        plt.tight_layout()


        # Save the figure
        chart_path = os.path.join(charts_path, "chart_2.png")
        plt.savefig(chart_path, dpi=300, bbox_inches='tight', facecolor=fig.get_facecolor())
        plt.close(fig)

        print(f"Chart 2 saved to: {chart_path}")
        return chart_path
    except Exception as e:
        print(f"Error creating Chart 2: {e}")
        return None

import matplotlib.cm as cm
import matplotlib.colors as mcolors
from matplotlib.colors import to_hex
import seaborn as sns

from matplotlib.colors import to_hex
import matplotlib.cm as cm

def plot_stacked_bar_chart():
    # Read the GHG data
    ghg_data = read_ghg_continents()
    if ghg_data is None:
        print("GHG data could not be loaded.")
        return

    # Read the CSV for country-continent mapping
    csv_filename = "Countries by continents.csv"
    csv_path = os.path.join(archive_path, csv_filename)
    countries_by_continent = read_countries_by_continent(csv_path)
    if countries_by_continent is None:
        print("Country-continent mapping could not be loaded.")
        return

    # Remove 'GLOBAL TOTAL' from the country-level data
    ghg_data_filtered = ghg_data[ghg_data['Country'] != 'GLOBAL TOTAL']

    # Access 'GLOBAL TOTAL' value as the benchmark for calculations
    if 'GLOBAL TOTAL' not in ghg_data['Country'].values:
        print("'GLOBAL TOTAL' not found in the data.")
        return
    global_total_ghg = ghg_data.loc[ghg_data['Country'] == 'GLOBAL TOTAL', 2023].values[0]

    # Calculate percentage contributions and filter by threshold
    ghg_data_filtered['Percentage'] = (ghg_data_filtered[2023] / global_total_ghg) * 100
    ghg_data_filtered = ghg_data_filtered[ghg_data_filtered['Percentage'] >= 0.5]

    # Prepare data for the stacked bar chart
    continent_data = {}
    for continent, countries in countries_by_continent.items():
        continent_countries = ghg_data_filtered[ghg_data_filtered['Country'].isin(countries)]
        continent_data[continent] = continent_countries.set_index('Country')[2023]

    # Create a DataFrame for the stacked bar chart
    stacked_data = pd.DataFrame(continent_data).fillna(0).T

    # Calculate total emissions per continent and order them
    continent_totals = stacked_data.sum(axis=1)
    continent_percentages = (continent_totals / continent_totals.sum() * 100).sort_values(ascending=False)
    stacked_data = stacked_data.loc[continent_percentages.index]  # Reorder continents

    # Sort countries within each continent by their emissions
    for continent in stacked_data.index:
        stacked_data.loc[continent] = stacked_data.loc[continent].sort_values(ascending=False)

    # Define the number of countries
    num_countries = len(stacked_data.columns)
    
    # Combine multiple professional palettes dynamically
    # Start with tab10, and add colors from other palettes if needed
    base_palette = [plt.cm.tab20b(i / 20) for i in range(20)]  # Tab20 has 20 distinct colors
    if num_countries > len(base_palette):  # Extend the palette if more countries are needed
        base_palette.extend([plt.cm.tab20(i / 20) for i in range(20)])  # Add Tab20 colors
    
    # Limit the palette to the number of countries needed
    palette_for_countries = base_palette[:num_countries]
    
    # Convert colors to HEX for uniformity
    country_colors = [to_hex(color) for color in palette_for_countries]
    
    # Map each country to a unique color
    color_map = {country: color for country, color in zip(stacked_data.columns, country_colors)}
    
    # Adjust the figure size and layout
    fig, ax = plt.subplots(figsize=(16, 14))
    
    # Set grey background
    fig.patch.set_facecolor('#E8E8E8')  # Figure background
    ax.set_facecolor('#E8E8E8')         # Plot background

    # Plot the data with unique colors for each country
    for country in stacked_data.columns:
        ax.bar(
            stacked_data.index, 
            stacked_data[country], 
            label=country, 
            color=color_map[country], 
            width=0.8, 
            bottom=stacked_data.loc[:, :country].sum(axis=1) - stacked_data[country]
        )
    # Add space above the bars
    max_emission = stacked_data.sum(axis=1).max()
    ax.set_ylim(0, max_emission * 1.1)
    # Title and labels
    ax.set_title('Chart 3: Global GHG Emissions by Country and Continent (2023)', fontsize=18, fontweight='bold')
    ax.set_ylabel('GHG Emissions (Million Tonnes of CO2 Equivalent)', fontsize=14)
    ax.tick_params(axis='x', labelrotation=45, labelsize=14)
    ax.tick_params(axis='y', labelsize=14)
    ax.grid(axis='y', linestyle='--', linewidth=0.7)

    # Create a hierarchical legend
    legend_elements = []
    for continent in stacked_data.index:
        percentage = continent_percentages[continent]
        # Add a "continent header" with percentage to the legend
        legend_elements.append(Patch(facecolor='none', edgecolor='none', label=f"{continent} ({percentage:.1f}%)"))
        for country in stacked_data.loc[continent].sort_values(ascending=False).index:
            if stacked_data.loc[continent, country] > 0:  # Include only countries with emissions
                legend_elements.append(
                    Patch(facecolor=color_map[country], edgecolor='black', label=f"  - {country}")
                )
    
    # Add the custom hierarchical legend
    legend = ax.legend(
        handles=legend_elements,
        title='Countries by Continent',
        bbox_to_anchor=(1.05, 1),  # Position legend outside the plot
        loc='upper left',
        fontsize=12,  # Increased font size for better readability
        frameon=True
    )
    
    # Set legend background color to match the chart
    legend.get_frame().set_facecolor('#E8E8E8')  # Match the grey background
    legend.get_frame().set_edgecolor('black')    # Optional: Add a border for clarity
    
    # Optionally increase the title font size
    legend.set_title('Countries by Continent', prop={'size': 14})

    plt.subplots_adjust(bottom=0.3)  # Increased bottom margin for space after the note
    
    # Footnote
    fig.text(0.5, 0.02, 
             "Note: Only countries contributing ≥0.5% to global emissions are displayed.",
             ha='center', fontsize=12, style='italic')

    # Make sure layout fits all elements (note, legend, etc.)
    plt.tight_layout(rect=[0, 0.05, 1, 1])

    # === SAVE THE CHART IN D:\GHG\charts ===
    charts_path = os.path.join(base_path, "charts")
    os.makedirs(charts_path, exist_ok=True)  # Ensure the directory exists

    output_file = os.path.join(charts_path, "Chart_3.png")
    plt.savefig(output_file, dpi=300, facecolor=fig.get_facecolor())
    plt.close()
    
    print(f"Chart 3 saved to '{output_file}'.")



# =============================================================================
# 4. MAIN EXECUTION
# =============================================================================

def main():
    # 1. Download GHG data (automated)
    download_ghg_file()
    # 2. Prompt user for World Bank CLASS.xlsx data
    download_class_file()
    # 3. Read the data
    world_bank_data = read_world_bank_data()
    ghg_data = read_ghg_data()
    ghg_per_capita_data = read_ghg_per_capita_data()

    if world_bank_data is not None and ghg_data is not None and ghg_per_capita_data is not None:
        # 4. Create Chart 1 (total GHG)
        chart1_path = create_chart_1(ghg_data)

        # 5. Process data for Chart 2 (per-capita)
        aggregated_data = process_and_reshape_ghg_data(ghg_per_capita_data, world_bank_data)
        if aggregated_data is not None:
            # 6. Create Chart 2 (per-capita)
            chart2_path = create_chart_2(aggregated_data)
        else:
            print("Failed to process and reshape GHG per-capita data.")
    else:
        print("Failed to read necessary data for chart generation.")

if __name__ == "__main__":
    main()
    plot_stacked_bar_chart()
