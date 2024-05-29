import os
import numpy as np
import h5py
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

# Function to load data for a given year
def load_yearly_data(year, create_if_missing=False):
    """
    Loads daily climate data for a specified year.
    
    Parameters:
    year (int): The year for which to load the data.
    create_if_missing (bool): If True, creates an empty folder for the year if it does not exist.
    
    Returns:
    np.ndarray or None: A numpy array containing the average daily values of temperature, precipitation, 
                        solar radiation, and humidity for the specified year, or None if no data is found.
    """
    assets_folder = 'assets'
    year_folder = os.path.join(assets_folder, f'year_{year}')
    
    if not os.path.exists(year_folder):
        if create_if_missing:
            os.makedirs(year_folder)
            print(f"No data found for year {year}. Created empty folder for the year.")
        else:
            print(f"No data found for year {year}.")
        return None

    daily_data = []
    for day_of_year in range(1, 366):
        day_folder = os.path.join(year_folder, f'day_{day_of_year}')
        hdf5_filename = os.path.join(day_folder, f'day_{day_of_year}.hdf')
        if not os.path.exists(hdf5_filename):
            continue

        with h5py.File(hdf5_filename, 'r') as hdf:
            temperature = hdf['temperature'][:]
            precipitation = hdf['precipitation'][:]
            solar_radiation = hdf['solar_radiation'][:]
            humidity = hdf['humidity'][:]

            # Calculate average values for the day
            avg_temperature = np.mean(temperature)
            avg_precipitation = np.mean(precipitation)
            avg_solar_radiation = np.mean(solar_radiation)
            avg_humidity = np.mean(humidity)

            daily_data.append([avg_temperature, avg_precipitation, avg_solar_radiation, avg_humidity])

    if not daily_data:
        print(f"No valid data found for year {year}.")
        return None

    return np.array(daily_data)

# Function to load data for all years except the target year and future years
def load_training_data(start_year, end_year, exclude_year):
    """
    Loads and prepares training data by excluding the target year and future years.
    
    Parameters:
    start_year (int): The starting year of the data range.
    end_year (int): The ending year of the data range.
    exclude_year (int): The year to exclude from the training data.
    
    Returns:
    tuple: A tuple containing two numpy arrays:
           - X: The training data (climate data).
           - y: The synthetic crop yield data corresponding to the climate data.
    """
    X = []
    y = []
    for year in range(start_year, exclude_year):
        yearly_data = load_yearly_data(year)
        if yearly_data is not None:
            X.extend(yearly_data)
            # Generate synthetic crop yield as a function of the climate data
            synthetic_yield = np.mean(yearly_data[:, 0]) * 0.5 + np.mean(yearly_data[:, 1]) * 0.3 + np.mean(yearly_data[:, 2]) * 0.1 + np.mean(yearly_data[:, 3]) * 0.1
            y.extend([synthetic_yield] * len(yearly_data))
    return np.array(X), np.array(y)

# Function to calculate actual crop yield based on climate data
def calculate_actual_yield(yearly_data):
    """
    Calculates the actual crop yield based on the provided climate data.
    
    Parameters:
    yearly_data (np.ndarray): The climate data for a specific year.
    
    Returns:
    float: The calculated crop yield based on the climate data.
    """
    return np.mean(yearly_data[:, 0]) * 0.5 + np.mean(yearly_data[:, 1]) * 0.3 + np.mean(yearly_data[:, 2]) * 0.1 + np.mean(yearly_data[:, 3]) * 0.1

# Function to predict crop yield for a given year
def predict_crop_yield(year, model, scaler):
    """
    Predicts the crop yield for a specified year using a trained model and scaler.
    
    Parameters:
    year (int): The year for which to predict the crop yield.
    model (LinearRegression): The trained linear regression model.
    scaler (StandardScaler): The scaler used to normalize the training data.
    
    Prints:
    - Actual crop yield for the specified year.
    - Predicted crop yield for the specified year.
    - Details of the prediction including model coefficients, intercept, and feature means.
    """
    yearly_data = load_yearly_data(year, create_if_missing=True)
    if yearly_data is None:
        return

    yearly_data_scaled = scaler.transform(yearly_data)
    predicted_yield = model.predict(yearly_data_scaled)
    predicted_yield_mean = np.mean(predicted_yield)

    if np.any(yearly_data):
        actual_yield = calculate_actual_yield(yearly_data)
        print(f"Actual crop yield for year {year}: {actual_yield:.2f}")
    else:
        print(f"No actual crop yield data available for year {year}.")
        
    print(f"Predicted crop yield for year {year}: {predicted_yield_mean:.2f}")
    print("\nDetails of Prediction:")
    print(f"Model Coefficients: {model.coef_}")
    print(f"Intercept: {model.intercept_}")
    print(f"Scaled Feature Means: {np.mean(yearly_data_scaled, axis=0)}")
    print(f"Unscaled Feature Means: {np.mean(yearly_data, axis=0)}")

def main():
    """
    Main function to predict the crop yield for a specified year.
    
    Prompts the user to enter the year for which to predict the crop yield.
    Loads the training data excluding the specified year and future years.
    Trains a linear regression model using the training data.
    Predicts the crop yield for the specified year and prints the result.
    """
    # Input the year to predict crop yield
    year_to_predict = int(input("Enter the year to predict crop yield: "))

    # Load training data excluding the year to predict and future years
    X, y = load_training_data(2000, year_to_predict, year_to_predict)

    # Scale the data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Train the model
    model = LinearRegression()
    model.fit(X_scaled, y)

    # Predict the crop yield for the specified year
    predict_crop_yield(year_to_predict, model, scaler)

if __name__ == "__main__":
    main()


# Example output from test running
# Enter the year to predict crop yield: 2022
# Actual crop yield for year 2022: 34.13
# Predicted crop yield for year 2022: 34.12

# Details of Prediction:
# Model Coefficients: [ 7.26153157e+11  3.66210938e-04 -7.26153157e+11  1.91599268e-04]
# Intercept: 31.53794622265683
# Scaled Feature Means: [ 1.49440555  0.02026574  1.49440555 -0.08470927]
# Unscaled Feature Means: [ 12.38030246   2.50168391 211.90151232  59.94463214]


