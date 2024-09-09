import logging
from libTrajectory.config.config_parser import parse_config
from libTrajectory.preprocessing.STEL.preprocessor import Preprocessor
from libTrajectory.executor.STEL import Executor


def pipeline():
    log_path = f"./libTrajectory/logs/STEL/{name.split('_')[-1]}/"
    logging.basicConfig(filename=f'{log_path}{config["executor"]["net_name"]}.log',
                        format='%(asctime)s - %(message)s', level=logging.INFO)
    logging.info(f"config: {config}")

    train_tid, test_tid, enhance_tid = Preprocessor(config).get(method='load')
    executor = Executor(log_path, config)
    executor.train(train_tid, enhance_tid, test_tid)


if __name__ == "__main__":
    name = "STEL_taxi"  # "STEL_ais", "STEL_taxi"
    config = parse_config(name)
    pipeline()
