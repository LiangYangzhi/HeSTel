import pandas as pd
from geopy.distance import geodesic


def calculate_device_distance(device1: pd.DataFrame, col1: dict,
                              device2: pd.DataFrame, col2: dict,
                              distance_name="distance"):
    """
    计算device1与device2之间的距离
    :param device1: pd.DataFrame
    :param col1: dict
        The columns names corresponding to device1,
        in order {device: device, longitude: longitude, latitude: latitude, coverage/Non: coverage},
        Cannot include distance
    :param device2: pd.DataFrame
    :param col2: dict
        The columns names corresponding to device2,
        in order {device: device, longitude: longitude, latitude: latitude, coverage/Non: coverage},
        Cannot include distance
    :param distance_name: str
    :return: pd.DataFrame, columns in order [device1, device2, distance, coverage/None]
    """
    col = [col1['device'], col1['longitude'], col1['latitude']]
    if col1.get("coverage", None):
        col.append(col1['coverage'])
    device1 = device1[col]
    device1 = device1.drop_duplicates(subset=col)
    device1 = device1.dropna(subset=col)

    col = [col2['device'], col2['longitude'], col2['latitude']]
    if col2.get("coverage", None):
        col.append(col2['coverage'])
    device2 = device2[col]
    device2 = device2.drop_duplicates(subset=col)
    device2 = device2.dropna(subset=col)

    device1.index = [0] * len(device1)
    device2.index = [0] * len(device2)
    # [device, longitude, latitude, coverage / None]
    device = device1.merge(device2, how="outer", left_index=True, right_index=True)
    device[distance_name] = device.apply(
        lambda row: geodesic((row[col1['latitude']], row[col1['longitude']]),
                             (row[col2['latitude']], row[col2['longitude']])).m, axis=1)

    if col1.get("coverage", None):
        device['distance<=coverage'] = device.apply(
            lambda row: 1 if row[distance_name] <= row[col1['coverage']] else 0, axis=1)
        device = device[device['distance<=coverage'] == 1]
        device.drop(columns=['distance<=coverage'], inplace=True)
        col = [col1['device'], col2['device'], distance_name, col1["coverage"]]

    elif col2.get("coverage", None):
        device['distance<=coverage'] = device.apply(
            lambda row: 1 if row[distance_name] <= row[col2['coverage']] else 0, axis=1)
        device = device[device['distance<=coverage'] == 1]
        device.drop(columns=['distance<=coverage'], inplace=True)
        col = [col1['device'], col2['device'], distance_name, col2["coverage"]]

    else:
        col = [col1['device'], col2['device'], distance_name]
    device = device[col]
    return device
