import json
import argparse

from libtrajectory.config.config_parser import parse_config
from libtrajectory.pipeline.STEL_label_generation import pipeline
from libtrajectory.utils.judge_type import str2int


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # 命令行传参
    parser.add_argument('--preprocessing_file', type=str,
                        default="STEL_label_generation_imsi_car")
    parser.add_argument('--sample', type=str2int, default=0,
                        help='data1 sample, if sample == 0: data no sample')
    args = parser.parse_args()

    config = {
        "preprocessing": parse_config(f"/preprocessing/{args.preprocessing_file}"),
        "sample": args.sample
    }
    print(json.dumps(config, indent=4, ensure_ascii=False))
    pipeline(config)
