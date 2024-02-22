import os
import json
import rasterio
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import geopandas as gpd
from statsmodels.tsa.arima.model import ARIMA


def stitch_rasters(raster_paths: list[str]) -> np.ndarray:
    """Stitch rasters together into a 3D array

    Args:
        raster_paths (list[str]): ordered list of raster paths to stitch

    Returns:
        np.ndarray: array of shape (n_rasters, height, width)
    """

    arrays = []
    for path in raster_paths:
        with rasterio.open(path, "r") as src:
            arrays.append(src.read(1))
    return np.stack(arrays)


def get_raster_path_by_time(
    year: int, month: int, day: int, hour: int, type="count"
) -> str:
    """Gets the path to the raster corresponding to a given hour on a given day

    Args:
        year (int): 2023 or 2024
        month (int): month number (1-12)
        day (int): day number (1-31)
        hour (int): hour number (0-23)
        type (str, optional): count data of speed data. Defaults to "count".

    Returns:
        str: inferred path to the raster
    """
    base_dir: str = None
    if type == "count":
        if year == 2023:
            base_dir = "data/london_bus_data/distinctJourneyCounts_London_271023to080124/distinctJourneyCounts/2023"
        else:
            base_dir = "data/london_bus_data/distinctJourneyCounts_London_271023to080124/distinctJourneyCounts/2024"
    else:
        if year == 2023 and month < 12:
            base_dir = "data/london_bus_data/averageSpeeds_London_271023to301123"
        else:
            base_dir = "data/london_bus_data/averageSpeeds_London_011223to080124"

    day_dir = os.path.join(base_dir, str(month), str(day))
    # add leading zeros to hour
    hour_str = str(hour).zfill(2)
    day_str = str(day).zfill(2)
    month_str = str(month).zfill(2)
    return os.path.join(day_dir, f"{year}{month_str}{day_str}{hour_str}_3600_50.gtiff")


def calculate_pct_difference(
    change_arr: np.ndarray, source_arr: np.ndarray
) -> np.ndarray:
    """Calculates the percentage difference between the change array and the
    source array. If the source array is 0, the percentage difference is set to -1.

    Args:
        change_arr (np.ndarray): array to calculate the percentage difference for
        source_arr (np.ndarray): source array

    Returns:
        np.ndarray: array of elementwise percentage difference.
    """
    result_arr = np.zeros(change_arr.shape)
    for row in range(change_arr.shape[0]):
        for col in range(change_arr.shape[1]):
            if source_arr[row, col] == 0:
                result_arr[row, col] = -1
            else:
                result_arr[row, col] = (
                    (change_arr[row, col] - source_arr[row, col]) / source_arr[row, col]
                ) * 100
            result_arr[row, col] = round(result_arr[row, col], 0)
    return result_arr


def save_raster(
    data: np.ndarray, meta: dict, path: str, dtype: str = "float32"
) -> None:
    """Saves a raster to disk

    Args:
        data (np.ndarray): raster data
        meta (dict): raster metadata
        path (str): path to save the raster
        dtype (str, optional): data type of the raster. Defaults to "float32".
    """
    meta.update({"dtype": dtype})
    with rasterio.open(path, "w", **meta) as dst:
        dst.write(data)


def aggergate_by_day(dates: dict, aggregation_method="sum") -> np.ndarray:
    """Aggregates a set of dates into a single array averaged daily.

    Args:
        dates (dict): dictionary containing the dates to aggregate

    Returns:
        np.ndarray: aggregated array
    """
    result_arr = None
    for year in dates:
        for month in dates[year]:
            for day in dates[year][month]:
                day_paths = [
                    get_raster_path_by_time(year, month, day, hour, type="count")
                    for hour in range(24)
                ]
                if aggregation_method == "sum":
                    day_aggregate = stitch_rasters(day_paths).sum(axis=0)
                else:
                    day_aggregate = stitch_rasters(day_paths).mean(axis=0)
                day_aggregate = np.expand_dims(day_aggregate, axis=0)
                if result_arr is None:
                    result_arr = day_aggregate
                else:
                    result_arr = np.concatenate([result_arr, day_aggregate])
    return result_arr.mean(axis=0)


def aggergate_by_hour(dates: dict) -> np.ndarray:
    """Aggregates a set of dates into a single array averaged daily.

    Args:
        dates (dict): dictionary containing the dates to aggregate

    Returns:
        np.ndarray: aggregated array of shape (n_hours, height, width)
    """
    result_arr = None
    for year in dates:
        for month in dates[year]:
            for day in dates[year][month]:
                day_paths = [
                    get_raster_path_by_time(year, month, day, hour, type="count")
                    for hour in range(24)
                ]
                # Shape (1, 24, height, width)
                day_arr = stitch_rasters(day_paths).expand_dims(axis=0)
                if result_arr is None:
                    result_arr = day_arr
                else:
                    result_arr = np.concatenate([result_arr, day_arr])
    return result_arr.mean(axis=0)


def get_sensor_cell_locations(
    lat: int,
    long: int,
    raster_crs: rasterio.crs.CRS,
    transform: rasterio.transform.Affine,
) -> tuple[int, int]:
    """_summary_

    Args:
        lat (int): _description_
        long (int): _description_
        raster_crs (rasterio.crs.CRS): _description_
        transform (rasterio.transform.Affine): _description_

    Returns:
        tuple[int, int]: _description_
    """
    # Make geodf from lat long
    gdf = gpd.GeoDataFrame(geometry=gpd.points_from_xy([long], [lat]), crs="EPSG:4326")
    # Convert to raster crs
    gdf = gdf.to_crs(raster_crs)
    # Get the values of the raster at the points
    row, col = rasterio.transform.rowcol(transform, gdf.geometry.x, gdf.geometry.y)
    return row[0], col[0]


def get_all_speed_data() -> np.ndarray:
    """Reads all of the hourly speed data into a numpy array.

    Returns:
        np.ndarray: array of shape (n_hours, height, width)
    """
    full_arr = []
    # 2023 speed data
    for month in [10, 11, 12]:
        for day in range(1, 32):
            for hour in range(24):
                raster_path = get_raster_path_by_time(
                    2023, month, day, hour, type="speed"
                )
                try:
                    with rasterio.open(raster_path, "r") as src:
                        raster_arr = src.read(1)
                except:
                    print(f"Skipping: {month}/{day}/{hour}")
                    if month == 12:
                        # make array of -2
                        raster_arr = np.full((src.height, src.width), -2)
                        full_arr.append(raster_arr)
                    continue
                full_arr.append(raster_arr)

    # 2024 data
    for month in range(1, 2):
        for day in range(1, 32):
            for hour in range(24):
                raster_path = get_raster_path_by_time(
                    2024, month, day, hour, type="speed"
                )
                try:
                    with rasterio.open(raster_path, "r") as src:
                        raster_arr = src.read(1)
                except:
                    print(f"Skipping: {month}/{day}/{hour}")
                    continue
                full_arr.append(raster_arr)
    full_arr = np.array(full_arr)
    return full_arr


def get_sensor_locations(sensor_json_path: str) -> dict:
    """Reads in the json containing sensor information and returns a
    dictionary of name: (lat, long) pairs.

    Args:
        sensor_json_path (str): path to sensor json.

    Returns:
        dict: dictionary of sensor name: (lat, long) pairs.
    """

    with open(sensor_json_path, "r") as f:
        sensor_data = json.load(f)
    sensor_locations = {}
    for sensor in sensor_data["Sites"]["Site"]:
        site_code = sensor["@SiteCode"]
        lat = float(sensor["@Latitude"])
        long = float(sensor["@Longitude"])
        sensor_locations[site_code] = (lat, long)
    return sensor_locations


def create_sensor_geojson(output_path: str) -> None:
    sensor_location_dict = get_sensor_locations("sensors.json")

    # Save as gdf
    sensor_gdf = gpd.GeoDataFrame(
        geometry=gpd.points_from_xy(
            [x[1] for x in sensor_location_dict.values()],
            [x[0] for x in sensor_location_dict.values()],
        )
    )
    sensor_gdf["site_code"] = sensor_location_dict.keys()
    sensor_gdf.to_file(output_path, driver="GeoJSON")


def merge_air_quality_csvs(path_to_folder: str) -> pd.DataFrame:
    """Merges all of the air quality csvs in a folder into a single dataframe.

    Args:
        path_to_folder (str): _description_

    Returns:
        pd.DataFrame: _description_
    """
    merged_data = {}
    for index, csv_file in enumerate(os.listdir(path_to_folder)):
        current_df = pd.read_csv(os.path.join(path_to_folder, csv_file))
        if index == 0:
            merged_data["date_time"] = list(current_df["MeasurementDateGMT"])
        # Get the column name
        column_name = current_df.columns[-1]
        if len(current_df[column_name]) < 1753:
            continue
        site_code = csv_file.split("_")[0]
        pollutant = csv_file.split("_")[1]
        merged_data[f"{site_code}_{pollutant}"] = list(current_df[column_name])
    return pd.DataFrame(merged_data)


def get_site_data_frame(
    site_code: str, pollutant: str, air_pollution_df: pd.DataFrame
) -> pd.DataFrame:
    """Gets the data for a specific site from the air pollution dataframe.

    Args:
        site_code (str): _description_
        air_pollution_df (pd.DataFrame): _description_

    Returns:
        pd.DataFrame: _description_
    """
    site_columns = [col for col in air_pollution_df.columns if site_code in col]
    site_columns = [col for col in site_columns if pollutant in col]
    return air_pollution_df[site_columns]


if __name__ == "__main__":
    sensor_dict = get_sensor_locations("sensors.json")
    # merged_df = merge_air_quality_csvs("data/air_quality_csvs")
    # merged_df.to_csv("data/merged_air_quality.csv", index=False)
    air_pollution_df = pd.read_csv("data/merged_air_quality.csv")
    air_pollution_df = air_pollution_df.set_index("date_time")

    ce3_no2 = get_site_data_frame("CE3", "NO2", air_pollution_df)
    print(ce3_no2)
    ce3_no2.plot()
    plt.show()
