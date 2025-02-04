# Inferno Insights: A Real-Time Comparative Analysis of Fire Weather Indices and Active Wildfire Data

## Overview

This project provides a real-time comparative analysis of wildfire risk in Los Angeles County, California, by visualizing three distinct fire weather indices:

* **Canadian Forest Fire Weather Index (FWI):** A numerically-based index used to estimate the fire danger based on noon weather readings.
* **Fosberg Fire Weather Index (FFWI):** A fire danger index sensitive to weather and fuel moisture variations.
* **Modified Fosberg Fire Weather Index (mFFWI):** An enhanced version of FFWI that incorporates drought conditions via the Keetch-Byram Drought Index (KBDI).

The project overlays these indices on an interactive map displaying current fire locations, enabling users to assess the relative severity of fire danger predicted by each method and correlate this with real-time fire events.


## Data Sources

* **Weather Data:** Real-time and historical weather data (temperature, relative humidity, wind speed, precipitation) is acquired from the Open-Meteo API ([https://open-meteo.com/](https://open-meteo.com/)).  See the [Open-Meteo Documentation](https://open-meteo.com/en/docs) for details on the API. The specific API call used in this project can be found in the `open-meteo` file in the References folder.
* **Fire Data (Perimeters/Boundaries):** Active fire perimeter data is fetched dynamically from the Cal Fire website ([https://www.fire.ca.gov/incidents/](https://www.fire.ca.gov/incidents/)).  
* **Fuel Moisture Data:** Default initial values for Duff Moisture Code (DMC) and Drought Code (DC), components of the FWI, are used. Adjustment provisions based on historical data or expert knowledge can be implemented in future versions.

## Repository Structure

```
Inferno_Insights/
├── Project 3 Proposal - Inferno Insights.pdf  # Project documentation
├── References/  # Contains background references
├── Canadian Forest Fire Weather Index (FWI) System/  # FWI dataset
├── mFFWI/  # Modified Fosberg Fire Weather Index data
├── NRDS4_FDI/  # Additional fire danger index data
├── open-meteo/  # Weather data source
├── LA_Combined_Fire_Indices_Including_Fire_Boundaries_Combination_Markers_Final/
│   ├── fire_index_mapping_with_fire_boundaries_refined.ipynb  # Analysis notebook
│   ├── fire_index_mapping_with_fire_boundaries_refined_Explanaion.pdf  # Explanation of methods
│   ├── final_map_with_colormaps.png  # Visual representation of fire risks
│   ├── fire_index_mapping_with_fire_boundaries_refined_Code.py  # Python code
│   ├── fire_index_mapping_with_fire_boundaries_refined_Results.csv  # Results of analysis
│   ├── .cache.sqlite  # Cache for faster access
├── LA_Combined_Fire_Indices_Including_Fire_Boundaries_Combination_Markers_Final_Sqlite/
│   ├── fire_index_mapping_with_fire_boundaries_refined.ipynb  # Analysis notebook
│   ├── fire_index_mapping_with_fire_boundaries_refined_Explanaion.pdf  # Explanation of methods
│   ├── final_map_with_colormaps.png  # Visual representation of fire risks
│   ├── fire_index_mapping_with_fire_boundaries_refined_Code.py  # Python code
│   ├── fire_index_mapping_with_fire_boundaries_refined_Results.csv  # Results of analysis
│   ├── .cache.sqlite  # Cache for faster access
│   ├── fire_weather.db #SQLite database used to store results
```

## Methodology

1. **Data Acquisition:** Weather and fire data is retrieved via APIs.
2. **Index Calculation:** Python scripts, using libraries like `pyfwi`, `pandas` and `numpy`, calculate FWI, FFWI, and mFFWI based on weather data. KBDI is also calculated for mFFWI, and equilibrium moisture content (m) is computed as a supplementary indicator. The calculations for these indices are detailed in the references provided (see `mFFWI.md`, `Canadian Forest Fire Weather Index (FWI) System.pdf`, `NRDS4_FDI.pdf`).
3. **Data Storage & Retrieval (with SQLAlchemy & SQLite):**  A local SQLite database (`fire_weather.db`/`.cache.sqlite`) is used for efficient storage and retrieval of fire weather indices.  SQLAlchemy is employed for database interactions.
4. **GeoJSON Processing & Filtering (with Shapely):** Shapely is used to process and filter the GeoJSON data for California, extracting specifically Los Angeles County data for optimized performance and more focused visualization.  
5. **Visualization (with Folium & Leaflet):** Folium, a Python library built on Leaflet.js, generates interactive maps. Color-coded overlays visualize spatial distribution of index values, and markers show active fires.  Layer control is implemented for dynamic overlay management. Interactive elements allow date/time and geographic filtering.


## File Structure and Description

The project consists of five main versions represented by different code iterations. See the "Summary Explanation of the Progress Made with the First Five Code Iterations" file for an overview of the changes made in each version.


* **`fire_index_mapping.py` (Initial Version):** This initial script fetches weather data, calculates fire weather indices (FWI, FFDI, mFFWI), and creates basic interactive maps using Folium.  It has some performance issues with GeoJSON processing.


* **`la_county_map_from_CNRA_GeoJson.py` (Extracting LA County):** This script focuses on efficiently extracting the Los Angeles County data from the larger California GeoJSON to improve performance of the mapping.


* **`fire_index_mapping_with_fire_boundaries.py` (First & Second Versions - Combining Indices & Boundaries, Adding Colormaps & Legends):**
    * First version: Integrates LA County boundary extraction into the main script and creates separate base maps for each index to avoid marker overlap.
    * Second version: Adds colormaps and legends for each index.


* **`fire_index_mapping_with_fire_boundaries.py` (Final Version - Template & Serialization Improvements):**
    * This version improves performance by using Jinja2 templates and custom JSON encoding for simplified JSON handling. Layer control is also added for interactive toggling of layers, as well as offsetting markers to enhance visibility.


Each of the final two versions has corresponding subfolders with the actual code, explanation files, result images (if any), and other relevant files like the `.cache.sqlite` database generated for storing and retrieving the calculated data.

## Ethical Considerations

* **Data Accuracy & Limitations:** The visualizations clearly show the limitations of the models used, acknowledging potential uncertainties in weather predictions and fire behavior.
* **Data Attribution:** The `References` folder documents the sources of data and code, attributing credit where due.
* **Bias:**  Potential biases, such as those introduced by using default values for fuel moisture, are noted in the explanation files.

The project is intended for informational use and should not replace expert consultation for critical decision-making.

## Installation and Usage

### Requirements

Ensure you have Python installed along with the following dependencies:

```sh
pip install -r requirements.txt
```

### Running the Analysis

Execute the main analysis script:

```sh
python fire_index_mapping_with_fire_boundaries_refined_Code.py
```

Or run the Jupyter Notebook for a step-by-step analysis:

```sh
jupyter notebook fire_index_mapping_with_fire_boundaries_refined.ipynb
```
You can also see the webpage output of running the repository at https://AlexGerwer.github.io/Inferno_Insights/
The map should load, displaying the fire index markers and the color scale for each fire index box. You can interact with the map by zooming and panning, and you can toggle the visibility of different fire indices, various fire index combinations, and the fire boundary map using the layer control.

## Results

The project outputs an interactive map displaying fire danger indices, with color-coded risk levels for better visualization. This allows researchers and decision-makers to compare various fire weather indices in real time.

## License
The project is released under the MIT License, which is a permissive open-source license that allows users to freely use, modify, distribute, and sublicense the software with minimal restrictions. This means that anyone can incorporate this project into their own work, whether for personal, academic, or commercial purposes, as long as the original copyright notice and license terms are included. For full details, refer to the LICENSE file in the repository.
