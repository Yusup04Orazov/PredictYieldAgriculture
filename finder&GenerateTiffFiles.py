import os
import numpy as np
import h5py
import matplotlib.pyplot as plt

# Define the function to generate and save .tiff images
def generate_tiff_image(year, day_of_year):
    assets_folder = 'assets'
    year_folder = os.path.join(assets_folder, f'year_{year}')
    day_folder = os.path.join(year_folder, f'day_{day_of_year}')
    hdf5_filename = os.path.join(day_folder, f'day_{day_of_year}.hdf')

    if not os.path.exists(hdf5_filename):
        print(f"Data for year {year}, day {day_of_year} does not exist.")
        return

    with h5py.File(hdf5_filename, 'r') as hdf:
        temperature = hdf['temperature'][:]
        precipitation = hdf['precipitation'][:]
        solar_radiation = hdf['solar_radiation'][:]
        humidity = hdf['humidity'][:]

        # Define the longitudes and latitudes (assuming they are the same as in the original data generation script)
        latitudes = np.linspace(40.0, 42.5, temperature.shape[0])
        longitudes = np.linspace(-100.0, -96.0, temperature.shape[1])

        # Function to create and save tiff images
        def save_tiff(data, title, colormap, filename):
            plt.figure(figsize=(10, 6))
            plt.contourf(longitudes, latitudes, data, cmap=colormap)
            plt.colorbar(label=title)
            plt.title(f'{title} Distribution on Year {year} Day {day_of_year}')
            plt.xlabel('Longitude')
            plt.ylabel('Latitude')
            plt.savefig(filename, format='tiff')
            plt.close()

        # Create and save tiff images for each variable
        save_tiff(temperature, 'Temperature (Celsius)', 'coolwarm', os.path.join(day_folder, f'temperature_day_{day_of_year}.tiff'))
        save_tiff(precipitation, 'Precipitation (mm)', 'Blues', os.path.join(day_folder, f'precipitation_day_{day_of_year}.tiff'))
        save_tiff(solar_radiation, 'Solar Radiation (W/m²)', 'YlOrRd', os.path.join(day_folder, f'solar_radiation_day_{day_of_year}.tiff'))
        save_tiff(humidity, 'Humidity (%)', 'BrBG', os.path.join(day_folder, f'humidity_day_{day_of_year}.tiff'))

# Main script
def main():
    year = int(input("Enter the year: "))
    day_of_year = int(input("Enter the day of the year: "))
    generate_tiff_image(year, day_of_year)
    print(f"Images for year {year}, day {day_of_year} have been generated and saved.")

if __name__ == "__main__":
    main()
