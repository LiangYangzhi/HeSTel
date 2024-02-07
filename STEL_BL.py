import logging
from libTrajectory.preprocessing.STEL.signature_BL import Preprocessor
from libTrajectory.model.knn_query import knn_query


log_path = "./libTrajectory/logs/STEL_BL/"
data_path = "./libTrajectory/dataset/AIS/"


def pipeline():
    logging.basicConfig(filename=f'{log_path}STEL_BL.log', format='%(asctime)s - %(levelname)s - %(message)s',
                        level=logging.INFO)
    preprocessor = Preprocessor(f"{data_path}sample100.csv",
                                {"test1": f"{data_path}sample10.csv", "test2": f"{data_path}sample30.csv"})
    test_data = preprocessor.sequential()
    for k, v in test_data.items():
        print(k)
        v1, v2 = v
        distances, indices = knn_query(v1, v2)
        print(distances, indices)


if __name__ == "__main__":
    pipeline()
