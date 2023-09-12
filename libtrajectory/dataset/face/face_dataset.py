import datetime
import os

import pandas as pd
from pymongo import MongoClient
from libtrajectory.utils.time_utils import parse_time
from libtrajectory.utils.mongo import mongo_to_df


class FaceDataset(object):
    def __init__(self, config):
        self.config = config
        self.client = self.config.get("client")
        self.db = self.config.get("db")
        self.collection = self.config.get("collection")
        self.condition = self.config.get("condition", {})
        self.rename = self.config.get("rename", None)
        self.col = self.config.get("columns", None)
        self.city = self.config.get("city", "zigong")
        self.time_column = self.config.get("time_column")
        self.start_time = self.config.get("start_time")
        self.end_time = self.config.get("end_time")
        self.inplace = self.config.get("inplace", False)
        self.device = self.config.get("device")
        self.place = self.config.get("place")

    def _map(self):
        # map table
        client = MongoClient(self.device.get("client"))
        db = client[self.device.get("db")]
        collection = db[self.device.get("collection")]
        cursor = collection.find(self.device.get("condition", {}))

        device_place = {}
        device_lon = {}
        device_lat = {}
        for doc in cursor:
            if "placeId" in doc:
                device_place[doc["_id"]] = doc["placeId"]
            device_lon[doc["_id"]] = doc["bd09ll_lon"]
            device_lat[doc["_id"]] = doc["bd09ll_lat"]
        cursor.close()
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

    def _face(self):
        start_time = parse_time(self.start_time)
        end_time = parse_time(self.end_time)
        days = (end_time - start_time).days + 1
        device_place, device_lon, device_lat = self._map()
        for day in range(days):
            loop_start_time = start_time + datetime.timedelta(days=day)
            loop_end_time = start_time + datetime.timedelta(days=day + 1)
            loop_start_str = loop_start_time.strftime("%Y_%m_%d")
            print(f"loop: {day}/{days}, {loop_start_time}----{loop_end_time}, running")
            face = self._time_read(loop_start_time, loop_end_time)

            if not isinstance(face, pd.DataFrame):
                continue
            if self.rename:
                face.rename(columns=self.rename, inplace=True)
            face.sort_values(by='time', ascending=True, inplace=True)
            face["longitude"] = face["deviceId"].map(device_lon)
            face["latitude"] = face["deviceId"].map(device_lat)
            face["placeId"] = face["deviceId"].map(device_place)
            if self.col:
                face = face[self.col]

            path = f"./libtrajectory/dataset/face/{self.city}/{loop_start_str}.csv"
            if os.path.exists(path):
                print(f"path: {path} is existed")
                if self.inplace:
                    face.to_csv(path, index=False)
                    print("The file has been overwritten")
                else:
                    print("The file not overwritten")
            else:
                face.to_csv(path, index=False)
                print(f"The file save in: {path}")

    def _device(self):
        """
        get device table and three map table
        map table1: deviceId: placeId
        map table2: deviceId: placeId
        map table3: deviceId: placeId
        :return: device, map table
        """
        client = self.device.get("client")
        db = self.device.get("db")
        collection = self.device.get("collection")
        condition = self.device.get("condition", {})
        device = mongo_to_df(client, db, collection, condition)
        if self.device.get("rename", None):
            device.rename(columns=self.device.get("rename"), inplace=True)
        if self.device.get("columns", None):
            device = device[self.device.get("columns")]

        path = f"./libtrajectory/dataset/face/{self.city}/{self.device.get('name', 'device')}.csv"
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

        path = f"./libtrajectory/dataset/face/{self.city}/{self.place.get('name', place)}.csv"
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
        print("-" * 10, "face")
        self._face()
        print("-" * 10, "device")
        self._device()
        print("-" * 10, "place")
        self._place()
