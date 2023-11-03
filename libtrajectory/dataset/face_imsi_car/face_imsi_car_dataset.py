import os


from libtrajectory.utils.mongo import mongo_to_df


class FaceImsiData(object):
    def __init__(self, config):
        self.config = config
        self.client = self.config.get("client")
        self.db = self.config.get("db")
        self.collection = self.config.get("collection")
        self.condition = self.config.get("condition", {})
        self.rename = self.config.get("rename", None)
        self.columns = self.config.get("columns", None)
        self.city = self.config.get("city", "zigong")
        self.name = self.config.get("name")
        self.inplace = self.config.get("inplace", False)

    def get_data(self):
        df = mongo_to_df(self.client, self.db, self.collection, self.condition)
        if self.rename:
            df.rename(columns=self.rename, inplace=True)
        if self.columns:
            df = df[self.columns]

        path = f"./libtrajectory/dataset/face_imsi_car/{self.name}.csv"
        if os.path.exists(path):
            print(f"path: {path} is existed")
            if self.inplace:
                df.to_csv(path, index=False)
                print("The file has been overwritten")
            else:
                print("The file not overwritten")
        else:
            df.to_csv(path, index=False)
            print(f"The file save in: {path}")


class CarImsiData(object):
    def __init__(self, config):
        self.config = config
        self.client = self.config.get("client")
        self.db = self.config.get("db")
        self.collection = self.config.get("collection")
        self.condition = self.config.get("condition", {})
        self.rename = self.config.get("rename", None)
        self.columns = self.config.get("columns", None)
        self.city = self.config.get("city", "zigong")
        self.name = self.config.get("name")
        self.inplace = self.config.get("inplace", False)

    def get_data(self):
        df = mongo_to_df(self.client, self.db, self.collection, self.condition)
        if self.rename:
            df.rename(columns=self.rename, inplace=True)
        if self.columns:
            df = df[self.columns]

        path = f"./libtrajectory/dataset/face_imsi_car/{self.name}.csv"
        if os.path.exists(path):
            print(f"path: {path} is existed")
            if self.inplace:
                df.to_csv(path, index=False)
                print("The file has been overwritten")
            else:
                print("The file not overwritten")
        else:
            df.to_csv(path, index=False)
            print(f"The file save in: {path}")
