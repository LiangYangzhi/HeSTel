import json

from libtrajectory.config.config_parser import parse_config
from libtrajectory.pipeline.traj_correlation_pipeline import pipeline
import argparse

# import argparse  Todo
from libtrajectory.utils.judge_type import str2bool

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # 命令行传参
    parser.add_argument('--preprocessing_file', type=str, default="traj_correlation_preprocessing")
    parser.add_argument('--test', type=str2bool, default=False, help='testing with small datasets')
    parser.add_argument('--save', type=str2bool, default=False, help='the name of task')
    args = parser.parse_args()

    config = {
        "preprocessing": parse_config(f"/preprocessing/{args.preprocessing_file}"),
        "test": args.test,  # 是否使用小批量的标注进行
        "model": parse_config("/model/lightgbm_model"),  # model params
        "saved_model": args.save,
        "save_path": f"/trained_model/ST_trajectory_correlation"
    }
    print(json.dumps(config, indent=4, ensure_ascii=False))
    pipeline(config)
