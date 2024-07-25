import logging
from libTrajectory.preprocessing.STEL.signature_BL import Preprocessor
from libTrajectory.evaluator.knn_query import evaluator


log_path = "./libTrajectory/logs/STEL/baseline/"
# path = "./libTrajectory/dataset/ais/"
path = "./libTrajectory/dataset/taxi/"
# lab 环境
# scikit-learn             1.3.0
# scipy                    1.11.2
# numpy                    1.25.2


def pipeline():
    logging.basicConfig(filename=f'{log_path}taxi_1.log', format='%(asctime)s - %(message)s',
                        level=logging.INFO)
    preprocessor = Preprocessor(f"{path}", {"test1": "test1K.csv", "test2": "test3K.csv"})

    methods = ['year_month', 'month_day', 'week_day', 'day_hour']

    # 序列signature
    test_data = preprocessor.sequential()
    for k, v in test_data.items():
        logging.info(f"{k}")
        v1, v2 = v
        distances, indices = evaluator(v1, v2)

    # 时间signature
    for method in methods:
        test_data = preprocessor.temporal(method=method)
        for k, v in test_data.items():
            logging.info(f"{k}")
            v1, v2 = v
            distances, indices = evaluator(v1, v2, method=method)

    # 空间signature
    test_data = preprocessor.spatial()
    for k, v in test_data.items():
        logging.info(f"{k}")
        v1, v2 = v
        distances, indices = evaluator(v1, v2)

    # 时空signature
    for method in methods:
        test_data = preprocessor.spatiotemporal(method=method)
        for k, v in test_data.items():
            logging.info(f"{k}")
            v1, v2 = v
            distances, indices = evaluator(v1, v2)


if __name__ == "__main__":
    pipeline()
