import os
import json
import rasterio
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import geopandas as gpd
from statsmodels.tsa.arima.model import ARIMA
from tqdm import tqdm
from scipy import stats


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
        dst.write(data, 1)


def aggergate_by_day(
    dates: dict, aggregation_method="sum", data_type="speed"
) -> np.ndarray:
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
                    get_raster_path_by_time(year, month, day, hour, type=data_type)
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


def aggergate_by_hour(dates: dict, data_type="speed") -> np.ndarray:
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
                    get_raster_path_by_time(year, month, day, hour, type=data_type)
                    for hour in range(24)
                ]
                # Shape (1, 24, height, width)
                day_arr = np.expand_dims(stitch_rasters(day_paths), axis=0)
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


def get_all_data(data_type="speed") -> np.ndarray:
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
                    2023, month, day, hour, type=data_type
                )
                try:
                    with rasterio.open(raster_path, "r") as src:
                        raster_arr = src.read(1)
                except:
                    # print(f"Skipping: {month}/{day}/{hour}")
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
                    2024, month, day, hour, type=data_type
                )
                try:
                    with rasterio.open(raster_path, "r") as src:
                        raster_arr = src.read(1)
                except:
                    # print(f"Skipping: {month}/{day}/{hour}")
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


def get_tile_correlation(
    air_pollution_df: pd.DataFrame,
    site_code: str,
    pollutant: str,
    tile_row: int,
    tile_col: int,
    bus_data: np.ndarray,
    bus_data_type: str = "speed",
) -> tuple[float, float, float]:
    """Gets the correlation between a given tile and a given pollutant.

    Args:
        site_code (str): _description_
        pollutant (str): _description_
        raster_crs (rasterio.crs.CRS): _description_
        transform (rasterio.transform.Affine): _description_
        tile_row (int): _description_
        tile_col (int): _description_
        bus_data_tye (str, optional): _description_. Defaults to "speed".

    Returns:
        float: _description_
    """
    # adjust for length difference
    air_pollution_length = len(air_pollution_df)
    bus_data_length = len(bus_data)
    if air_pollution_length > bus_data_length:
        air_pollution_df = air_pollution_df.iloc[:bus_data_length]
    else:
        bus_data = bus_data[:air_pollution_length]
    pollutant_df = get_site_data_frame(site_code, pollutant, air_pollution_df)
    # Supress SettingWithCopyWarning
    pd.options.mode.chained_assignment = None
    pollutant_df[f"{bus_data_type}_bus_data"] = bus_data[:, tile_row, tile_col]
    pollutant_df = pollutant_df.replace(-1, np.nan)
    pollutant_df = pollutant_df.replace(-2, np.nan)
    pollutant_df = pollutant_df.dropna()
    if len(pollutant_df) < 100:
        return -2
    y = pollutant_df[f"{site_code}_{pollutant}"]
    x = pollutant_df[f"{bus_data_type}_bus_data"]
    correlation = stats.pearsonr(x, y)
    confidence_interval = correlation.confidence_interval()
    return confidence_interval[0], correlation[0], confidence_interval[1]


def create_correlation_raster(
    raster_meta: dict,
    correlation_data: dict,
    output_path: str,
) -> None:
    """Creates a raster of the correlation between a given site and a given pollutant.

    Args:
        site_code (str): _description_
        raster_meta (dict): _description_
        correlation_data (dict): _description_
        output_path (str): _description_
    """
    correlation_arr = np.zeros((raster_meta["height"], raster_meta["width"]))
    # fill with 2
    correlation_arr.fill(2)
    # print(correlation_data)
    for cell in correlation_data:
        row, col = cell
        correlation_arr[row, col] = correlation_data[cell]
        # print(correlation_data[cell])
    save_raster(correlation_arr, raster_meta, output_path)


if __name__ == "__main__":
    with rasterio.open(
        "/home/mattbarker/projects/cusp_data_drive/CUSP2024/data/london_bus_data/averageSpeeds_London_011223to080124/1/2/2024010200_3600_50.gtiff",
        "r",
    ) as f:
        raster_crs = f.meta["crs"]
        transform = f.meta["transform"]

    air_pollution_df = pd.read_csv("data/merged_air_quality.csv")
    air_pollution_df = air_pollution_df.set_index("date_time")
    sensor_dict = get_sensor_locations("sensors.json")
    bus_speed_data = get_all_data(data_type="speed")
    bus_count_data = get_all_data(data_type="count")
    max_row = bus_speed_data.shape[1]
    max_col = bus_speed_data.shape[2]
    distance_correlations = {}
    for site in tqdm(list(sensor_dict.keys())):
        lat, long = sensor_dict[site]
        row, col = get_sensor_cell_locations(lat, long, raster_crs, transform)
        if row < 0 or row >= max_row or col < 0 or col >= max_col:
            continue
        speed_correlation_data = {}
        count_correlation_data = {}
        for row_offset in range(-100, 101):
            for col_offset in range(-100, 101):
                current_row = row + row_offset
                current_col = col + col_offset
                if (
                    current_row < 0
                    or current_row >= max_row
                    or current_col < 0
                    or current_col >= max_col
                ):
                    continue
                try:
                    speed_corr = get_tile_correlation(
                        air_pollution_df,
                        site,
                        "NO2",
                        current_row,
                        current_col,
                        bus_speed_data,
                        bus_data_type="speed",
                    )
                except KeyError:
                    continue
                if speed_corr == -2:
                    continue
                count_corr = get_tile_correlation(
                    air_pollution_df,
                    site,
                    "NO2",
                    current_row,
                    current_col,
                    bus_count_data,
                    bus_data_type="count",
                )
                dist = abs(row_offset) + abs(col_offset)
                if site not in distance_correlations:
                    distance_correlations[site] = {
                        dist: {"speed": [speed_corr], "count": [count_corr]}
                    }
                else:
                    if dist not in distance_correlations[site]:
                        distance_correlations[site][dist] = {
                            "speed": [speed_corr],
                            "count": [count_corr],
                        }
                    else:
                        distance_correlations[site][dist]["speed"].append(speed_corr)
                        distance_correlations[site][dist]["count"].append(count_corr)
                # print(count_corr)
                # print(distance_correlations)
                speed_correlation_data[(current_row, current_col)] = speed_corr
                count_correlation_data[(current_row, current_col)] = count_corr
                # print(speed_corr, count_corr)

        with open("distance_correlations_with_confidence.json", "w") as f:
            json.dump(distance_correlations, f)
        if speed_correlation_data == {}:
            continue
        speed_raster_path = f"correlation_rasters/{site}_speed_correlations.tif"
        count_raster_path = f"correlation_rasters/{site}_count_correlations.tif"
        # create_correlation_raster(f.meta, speed_correlation_data, speed_raster_path)
        # create_correlation_raster(f.meta, count_correlation_data, count_raster_path)
    with open("distance_correlations.json", "w") as f:
        json.dump(distance_correlations, f)
    # speed_data = get_all_data(data_type="speed")[:-23, :, :]
    # count_data = get_all_data(data_type="count")[:-23, :, :]
    # # merged_df = merge_air_quality_csvs("data/air_quality_csvs")
    # # merged_df.to_csv("data/merged_air_quality.csv", index=False)
    # air_pollution_df = pd.read_csv("data/merged_air_quality.csv")
    # air_pollution_df = air_pollution_df.set_index("date_time")
    # correlation_dict = {}
    # all_data_dict = {}
    # for index, site_code in enumerate(sensor_dict):
    #     try:
    #         no2_df = get_site_data_frame(site_code, "NO2", air_pollution_df)
    #         site_lat_long = sensor_dict[site_code]
    #         site_row, site_col = get_sensor_cell_locations(
    #             site_lat_long[0], site_lat_long[1], raster_crs, transform
    #         )
    #         site_speed_data = speed_data[:, site_row, site_col]
    #         site_count_data = count_data[:, site_row, site_col]
    #         no2_df["bus_speed_data"] = site_speed_data
    #         no2_df["bus_count_data"] = site_count_data
    #         # Set -1 to nan
    #         no2_df = no2_df.replace(-1, np.nan)
    #         no2_df = no2_df.replace(-2, np.nan)
    #         all_data_dict[f"{site_code}_NO2"] = list(no2_df[f"{site_code}_NO2"])
    #         all_data_dict[f"{site_code}_bus_speed"] = list(no2_df["bus_speed_data"])
    #         all_data_dict[f"{site_code}_bus_count"] = list(no2_df["bus_count_data"])

    #         # drop_nans = no2_df.dropna()
    #         # if len(drop_nans) < 100:
    #         #     continue
    #         # # get lagged pollution
    #         # no2_df["NO2_lag1"] = no2_df[f"{site_code}_NO2"].shift(1)

    #         # correlation = no2_df.corr()
    #         # # Get correlation between speed and pollution
    #         # speed_no_lag_corr = correlation["speed_data"][f"{site_code}_NO2"]
    #         # speed_lag_corr = correlation["speed_data"]["NO2_lag1"]
    #         # correlation_dict[site_code] = {
    #         #     "speed_no_lag_corr": speed_no_lag_corr,
    #         #     "speed_lag_corr": speed_lag_corr,
    #         # }
    #     except:
    #         print(f"Failed for site: {site_code}")
    #         continue
    # all_data_df = pd.DataFrame(all_data_dict, index=air_pollution_df.index)
    # all_data_df.to_csv("data/all_data.csv")
    # correlation_df = pd.DataFrame(correlation_dict).T
    # print(correlation_df)
    # # save to csv
    # correlation_df.to_csv("data/correlation.csv")

    # # # get wm6 no2 data
    # # wm6_no2 = get_site_data_frame("WAA", "NO2", air_pollution_df)
    # # wm6_lat_long = sensor_dict["WAA"]
    # # wm6_row, wm6_col = get_sensor_cell_locations(
    # #     wm6_lat_long[0], wm6_lat_long[1], raster_crs, transform
    # # )
    # # wm6_speed_data = speed_data[:, wm6_row, wm6_col]
    # # wm6_no2["speed_data"] = wm6_speed_data
    # # wm6_no2 = wm6_no2.replace(-1, np.nan)
    # # wm6_no2 = wm6_no2.replace(-2, np.nan)
    # # wm6_no2 = wm6_no2.dropna()
    # # wm6_no2.to_csv("data/waa_no2.csv")
    # # wm6_no2.plot()
    # # plt.show()

    # # protest_data = [
    # #     get_raster_path_by_time(2023, 10, 28, hour, type="count") for hour in range(24)
    # # ]
    # # protest_arr = stitch_rasters(protest_data)
    # # protest_sensor = sensor_dict["WAA"]
    # # protest_row, protest_col = get_sensor_cell_locations(
    # #     protest_sensor[0], protest_sensor[1], raster_crs, transform
    # # )
    # # comparison_arr = aggergate_by_hour({2023: {11: [4, 11, 18, 25]}})
    # # print(protest_arr.shape)
    # # print(comparison_arr.shape)
    # # for hour in range(24):
    # #     result = calculate_pct_difference(
    # #         protest_arr[hour, :, :], comparison_arr[hour, :, :]
    # #     )
    # #     print(f"Time: {hour}", result[protest_row, protest_col])
