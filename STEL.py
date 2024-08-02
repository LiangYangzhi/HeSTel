import logging
from libTrajectory.config.config_parser import parse_config
from libTrajectory.preprocessing.STEL.preprocessor import Preprocessor
from libTrajectory.executor.STEL import Executor


def pipeline():
    path = config['path']
    test_file = {"test1": "test1K.csv", "test2": "test3K.csv"}
    log_path = f"./libTrajectory/logs/STEL/{name.split('_')[-1]}/"
    logging.basicConfig(filename=f'{log_path}{config["executor"]["net_name"]}.log',
                        format='%(asctime)s - %(message)s', level=logging.INFO)

    train_tid, test_tid, enhance_tid = Preprocessor(path, test_file, config['preprocessing']).get(method='load')
    executor = Executor(path, log_path, config['executor'])
    executor.train(train_tid, enhance_tid, test_tid)


if __name__ == "__main__":
    name = "STEL_taxi"  # "STEL_ais", "STEL_taxi"
    config = parse_config(name)
    pipeline()
