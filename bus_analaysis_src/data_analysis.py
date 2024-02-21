import os
import rasterio
import numpy as np


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
                    change_arr[row, col] - source_arr[row, col] / source_arr[row, col]
                ) * 100


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
    meta.update({"count": 1, "dtype": dtype})
    with rasterio.open(path, "w", **meta) as dst:
        dst.write(data, 1)


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
    return result_arr


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


if __name__ == "__main__":
    raster_meta = rasterio.open(
        get_raster_path_by_time(2023, 12, 7, 0, type="count")
    ).meta
