from libtrajectory.config.config_parser import parse_config
from libtrajectory.pipeline.traj_correlation_pipeline import pipeline

# import argparse  Todo

if __name__ == "__main__":
    config = {
        "preprocessing": parse_config("/preprocessing/traj_correlation_preprocessing"),
        "model": parse_config("/model/lightgbm_model")
    }
    pipeline(config)
