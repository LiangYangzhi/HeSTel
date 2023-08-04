from libtrajectory.dataset.abstract_dataset import AbstractDataset


class FaceDataset(AbstractDataset):
    def __init__(self, config):
        self.config = config
        self.face_config = self.config.get("face")
        self.face_mongo = self.face_config.get("mongo")

