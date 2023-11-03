from libtrajectory.dataset.face.face_dataset import FaceDataset
from libtrajectory.dataset.face_imsi_car.face_imsi_car_dataset import FaceImsiData, CarImsiData
from libtrajectory.config.config_parser import parse_config
from libtrajectory.dataset.imsi.imsi_dataset import ImsiDataset

if __name__ == "__main__":
    # face
    # config = parse_config("dataset/FaceZiGong20220119Dataset")
    # face = FaceDataset(config)
    # face.get_data()

    # imsi
    config = parse_config("dataset/ImsiZiGong20220119Dataset")
    imsi = ImsiDataset(config)
    imsi.get_data()

    # # car
    # config = parse_config("dataset/CarZiGong20220119Dataset")
    # car = CarDataset(config)
    # car.get_data()

    # face-imsi
    # config = parse_config("dataset/faceImsiZiGong20220119Dataset")
    # face_imsi = FaceImsiData(config)
    # face_imsi.get_data()

    # car-imsi
    # config = parse_config("dataset/carImsiZiGong20220119Dataset")
    # car_imsi = CarImsiData(config)
    # car_imsi.get_data()
