import datetime
import os
import pandas as pd

from pymongo import MongoClient
from libtrajectory.utils.mongo import mongo_to_df
from libtrajectory.utils.time_utils import parse_time


class ImsiDataset(object):
    def __init__(self, config):
        self.config = config
        self.client = self.config.get("client")
        self.db = self.config.get("db")
        self.collection = self.config.get("collection")
        self.condition = self.config.get("condition", {})
        self.rename = self.config.get("rename", None)
        self.columns = self.config.get("columns", None)
        self.city = self.config.get("city", "zigong")
        self.time_column = self.config.get("time_column")
        self.start_time = self.config.get("start_time")
        self.end_time = self.config.get("end_time")
        self.inplace = self.config.get("inplace", False)
        self.device = self.config.get("device")
        self.place = self.config.get("place")

    def _map(self):
        # map table
        client = MongoClient(self.place.get("client"))
        db = client[self.place.get("db")]
        collection = db[self.place.get("collection")]
        cursor = collection.find()
        place_lon = {}
        place_lat = {}
        for doc in cursor:
            if "longitude" in doc:
                place_lon[str(doc["_id"])] = doc["longitude"]
                place_lat[str(doc["_id"])] = doc["latitude"]
        cursor.close()

        client = MongoClient(self.device.get("client"))
        db = client[self.device.get("db")]
        collection = db[self.device.get("collection")]
        cursor = collection.find(self.device.get("condition", {}))
        device_place = {}
        for doc in cursor:
            device_place[doc["deviceId"]] = doc["placeId"]
        cursor.close()

        device_lon = {device: place_lon[placeId] for device, placeId in device_place.items()}
        device_lat = {device: place_lat[placeId] for device, placeId in device_place.items()}
        return device_place, device_lon, device_lat

    def _time_read(self, start: datetime.datetime, end: datetime.datetime) -> pd.DataFrame or None:
        condition = self.condition
        condition[self.time_column] = {"$gte": int(start.timestamp()), "$lt": int(end.timestamp())}
        print(f"condition: {condition}")
        df = mongo_to_df(self.client, self.db, self.collection, condition)
        if df.shape[0] == 0:
            loop_start_str = start.strftime("%Y_%m_%d")
            print(f"{loop_start_str}, no data")
            return None
        return df

    def _imsi(self):
        start_time = parse_time(self.start_time)
        end_time = parse_time(self.end_time)
        days = (end_time - start_time).days + 1
        device_place, device_lon, device_lat = self._map()
        for day in range(days):
            loop_start_time = start_time + datetime.timedelta(days=day)
            loop_end_time = start_time + datetime.timedelta(days=day + 1)
            loop_start_str = loop_start_time.strftime("%Y_%m_%d")
            print(f"loop: {day}/{days}, {loop_start_time}----{loop_end_time}, running")
            imsi = self._time_read(loop_start_time, loop_end_time)

            if not isinstance(imsi, pd.DataFrame):
                continue
            if self.rename:
                imsi.rename(columns=self.rename, inplace=True)
            imsi.sort_values(by='time', ascending=True, inplace=True)
            imsi["longitude"] = imsi["deviceId"].map(device_lon)
            imsi["latitude"] = imsi["deviceId"].map(device_lat)
            imsi["placeId"] = imsi["deviceId"].map(device_place)
            if self.columns:
                imsi = imsi[self.columns]

            path = f"./libtrajectory/dataset/imsi/{self.city}/{loop_start_str}.csv"
            if os.path.exists(path):
                print(f"path: {path} is existed")
                if self.inplace:
                    imsi.to_csv(path, index=False)
                    print("The file has been overwritten")
                else:
                    print("The file not overwritten")
            else:
                imsi.to_csv(path, index=False)
                print(f"The file save in: {path}")

    def _device(self):
        """
        get device table and three map table
        map table1: deviceId: placeId
        map table2: deviceId: placeId
        map table3: deviceId: placeId
        :return: device, map table
        """
        client = MongoClient(self.device.get("client"))
        db = client[self.device.get("db")]
        collection = db[self.device.get("collection")]
        cursor = collection.find(self.device.get("condition", {}))
        device = []
        for doc in cursor:
            doc['longitude'] = doc['devicePos']['longitude']
            doc['latitude'] = doc['devicePos']['latitude']
            device.append(doc)
        device = pd.DataFrame(device)

        if self.device.get("rename", None):
            device.rename(columns=self.device.get("rename"), inplace=True)
        if self.device.get("columns", None):
            device = device[self.device.get("columns")]

        path = f"./libtrajectory/dataset/imsi/{self.city}/{self.device.get('name', 'device')}.csv"
        if os.path.exists(path):
            print(f"path: {path} is existed")
            if self.device.get("inplace", False):
                device.to_csv(path, index=False)
                print("The file has been overwritten")
            else:
                print("The file not overwritten")
        else:
            device.to_csv(path, index=False)
            print(f"The file save in: {path}")

    def _place(self):
        client = self.place.get("client")
        db = self.place.get("db")
        collection = self.place.get("collection")
        place = mongo_to_df(client, db, collection)
        if self.place.get("rename", None):
            place.rename(columns=self.place.get("rename"), inplace=True)
        if self.place.get("columns", None):
            place = place[self.place.get("columns")]

        path = f"./libtrajectory/dataset/imsi/{self.city}/{self.place.get('name', place)}.csv"
        if os.path.exists(path):
            print(f"path: {path} is existed")
            if self.place.get("inplace", False):
                place.to_csv(path, index=False)
                print("The file has been overwritten")
            else:
                print("The file not overwritten")
        else:
            place.to_csv(path, index=False)
            print(f"The file save in: {path}")

    def get_data(self):
        print("-" * 10, "imsi")
        self._imsi()
        print("-" * 10, "device")
        self._device()
        print("-" * 10, "place")
        self._place()
