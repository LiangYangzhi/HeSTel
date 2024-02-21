import logging
from libTrajectory.preprocessing.STEL.signature_BL import Preprocessor
from libTrajectory.evaluator.knn_query import knn_query


log_path = "./libTrajectory/logs/STEL_BL/"
data_path = "./libTrajectory/dataset/AIS/"


def pipeline():
    logging.basicConfig(filename=f'{log_path}STEL_BL.log', format='%(asctime)s - %(levelname)s - %(message)s',
                        level=logging.INFO)
    preprocessor = Preprocessor(f"{data_path}multiA.csv",
                                {"test1": f"{data_path}test1K.csv", "test2": f"{data_path}test3K.csv"})

    # 序列signature
    test_data = preprocessor.sequential()
    for k, v in test_data.items():
        logging.info(f"{k}")
        v1, v2 = v
        distances, indices = knn_query(v1, v2)

    # 时间signature
    test_data = preprocessor.temporal()
    for k, v in test_data.items():
        logging.info(f"{k}")
        v1, v2 = v
        distances, indices = knn_query(v1, v2, distance='emd')

    # 空间signature
    test_data = preprocessor.spatial()
    for k, v in test_data.items():
        logging.info(f"{k}")
        v1, v2 = v
        distances, indices = knn_query(v1, v2)

    # 时空signature
    test_data = preprocessor.spatiotemporal()
    for k, v in test_data.items():
        logging.info(f"{k}")
        v1, v2 = v
        distances, indices = knn_query(v1, v2)


if __name__ == "__main__":
    pipeline()
