import json
import argparse

from libtrajectory.config.config_parser import parse_config
from libtrajectory.pipeline.STEL_binary_classification import pipeline
from libtrajectory.utils.judge_type import str2bool, str2int

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # 命令行传参
    parser.add_argument('--preprocessing_file', type=str, default="STEL_binary_classification_car_imsi")
    parser.add_argument('--sample', type=str2int, default=0, help='testing with small datasets')
    args = parser.parse_args()

    config = {
        "preprocessing": parse_config(f"/preprocessing/{args.preprocessing_file}"),
        "sample": args.sample,  # 是否使用小批量的标注进行
        "model": parse_config("/model/lightgbm_model"),  # model params
    }
    print(json.dumps(config, indent=4, ensure_ascii=False))
    pipeline(config)
