import pandas as pd


def create_face_imsi_car():
    face_imsi = pd.read_csv("./faceImsiZigong20220119.csv")
    face_imsi = face_imsi[["fid", 'imsi']]
    print(f"face_imsi: {face_imsi.shape}")
    print(face_imsi.info())

    imsi_car = pd.read_csv("./carImsiZigong20220119.csv")
    imsi_car = imsi_car[["license", 'imsi']]
    print(f"imsi_car: {imsi_car.shape}")
    print(imsi_car.info())

    face_imsi_car = face_imsi.merge(imsi_car, how="inner")
    print(f"imsi_car: {face_imsi_car.shape}")
    print(face_imsi_car.info())
    face_imsi_car.to_csv("./faceImsiCarZigong20220119.csv", index=False)


if __name__ == "__main__":
    create_face_imsi_car()