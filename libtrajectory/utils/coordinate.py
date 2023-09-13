import pandas as pd
from geopy.distance import geodesic
from pandarallel import pandarallel
pandarallel.initialize()


def device_distance(device1: pd.DataFrame,
                    col1: list,
                    device2: pd.DataFrame,
                    col2: list,
                    distance=None,
                    distance_name="distance"):
    """
    计算device1与device2之间的距离
    :param device1: pd.DataFrame
    :param col1: list
        The columns names corresponding to device1, in order [device, longitude, latitude], Cannot include distance
    :param device2: pd.DataFrame
    :param col2: list
        The columns names corresponding to device2, in order [device, longitude, latitude], Cannot include distance
    :param distance: int
        None: 保留device1与device2两两之间距离
        int: 保留device1与device2之间距离小于space的数据
    :param distance_name: str
    :return: pd.DataFrame, columns in order [device1, device2, distance]
    """
    device1 = device1.drop_duplicates(subset=col1)
    device2 = device2.drop_duplicates(subset=col2)

    device1.index = [0] * len(device1)
    device2.index = [0] * len(device2)
    device = device1.merge(device2, how="outer", left_index=True, right_index=True)
    device[distance_name] = device.parallel_apply(
        lambda row: geodesic((row[col1[2]], row[col1[1]]), (row[col2[2]], row[col2[1]])).m
        , axis=1)  # (latitude, longitude)
    # face_latitude
    """
    self.space
    
    """
    if distance is not None:
        device = device[device[distance_name] <= distance]
    col = [col1[0], col2[0], distance_name]
    device = device[col]
    return device
