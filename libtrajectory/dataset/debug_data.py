"""
generate small batch dataset for debug
"""
import pandas as pd


def face_imsi():
    face = pd.read_csv("./libtrajectory/dataset/face/zigong/2022_02_21.csv")
    print(face.info())
    imsi = pd.read_csv("./libtrajectory/dataset/imsi/zigong/2022_02_21.csv")
    print(imsi.info())
    label = pd.read_csv("./libtrajectory/dataset/face_imsi_car/faceImsiZigong20220119.csv")
    print(label.info())

    face = face[face.time.isin(range(1645423200, 1645426800))]
    face = face[face.fid.isin(label.fid.unique())]
    fid_num = face.fid.value_counts().to_dict()
    need1_fid = []
    for k, v in fid_num.items():
        if v > 5:
            need1_fid.append(k)

    imsi = imsi[imsi.time.isin(range(1645423200, 1645426800))]
    print(imsi.info())
    need2_fid = []
    for f in need1_fid:
        i = label[label.fid.isin([f])].imsi.values[0]
        if imsi[imsi.imsi.isin([i])].shape[0] > 5:
            need2_fid.append(f)

    face = face[face.fid.isin(need2_fid)]
    print(len(face.fid.unique()))
    print(face.info())

    face.to_csv("./libtrajectory/dataset/face/zigong/1949_10_01.csv")
    imsi.to_csv("./libtrajectory/dataset/imsi/zigong/1949_10_01.csv")


def imsi_face():
    face = pd.read_csv("./libtrajectory/dataset/face/zigong/2022_02_21.csv")
    print(face.info())
    imsi = pd.read_csv("./libtrajectory/dataset/imsi/zigong/2022_02_21.csv")
    print(imsi.info())
    label = pd.read_csv("./libtrajectory/dataset/face_imsi_car/faceImsiZigong20220119.csv")
    print(label.info())

    imsi = imsi[imsi.time.isin(range(1645423200, 1645426800))]
    imsi = imsi[imsi.imsi.isin(label.imsi.unique())]
    imsi_num = imsi.imsi.value_counts().to_dict()
    need1_imsi = []
    for k, v in imsi_num.items():
        if v > 10:
            need1_imsi.append(k)
    print(len(need1_imsi))
    face = face[face.time.isin(range(1645423200, 1645426800))]

    need2_imsi = []
    for i in need1_imsi:
        f = label[label.imsi.isin([i])].fid.values[0]
        if face[face.fid.isin([f])].shape[0] > 5:
            need2_imsi.append(i)

    imsi = imsi[imsi.imsi.isin(need2_imsi)]
    print(len(imsi.imsi.unique()))
    print(imsi.info())

    face.to_csv("./libtrajectory/dataset/face/zigong/1949_11_01.csv")
    imsi.to_csv("./libtrajectory/dataset/imsi/zigong/1949_11_01.csv")


