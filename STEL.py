import logging
from libTrajectory.config.config_parser import parse_config
from libTrajectory.preprocessing.STEL.preprocessor import Preprocessor
from libTrajectory.executor.STEL import Executor, BLExecutor


def bl_pipeline():
    logging.basicConfig(filename=f"{log_path}{name.split('_', 2)[-1]}.log",
                        format='%(asctime)s - %(message)s', level=logging.INFO)
    logging.info(f"name: {name}")
    print(f"name: {name}")
    logging.info(f"config: {config}")
    train_tid, test_tid, _ = Preprocessor(config).get(method='load')
    executor = BLExecutor(log_path, config)

    # signature
    if "signature" in name:
        logging.info("signature")
        print("signature")
        executor.signature(config)

    # 本方法提供graph，作为以下baseline的输入
    # gcn
    if "gcn_cross" in name:
        logging.info("gcn_cross")
        print("gcn_cross")
        executor.gcn_cross(train_tid, test_tid)
    if "gcn_cosine" in name:
        logging.info("gcn_cosine")
        print("gcn_cosine")
        executor.gcn_cosine(train_tid, test_tid)
    # graph
    if "graph_cross" in name:
        logging.info("graph_cross")
        print("graph_cross")
        executor.graph_cross(train_tid, test_tid)  # batch_size不宜过大
    if "graph_cosine" in name:
        logging.info("graph_cosine")
        print("graph_cosine")
        executor.graph_cosine(train_tid, test_tid)
    # transformer
    if "transformer_cross" in name:
        logging.info("transformer_cross")
        print("transformer_cross")
        executor.transformer_cross(train_tid, test_tid)
    if "transformer_cosine" in name:
        logging.info("transformer_cosine")
        print("transformer_cosine")
        executor.transformer_cosine(train_tid, test_tid)
    # lstm
    if "lstm_cross" in name:
        logging.info("lstm_cross")
        print("lstm_cross")
        executor.lstm_cross(train_tid, test_tid)
    if "lstm_cosine" in name:
        logging.info("lstm_cosine")
        print("lstm_cosine")
        executor.lstm_cosine(train_tid, test_tid)


def pipeline():
    if "bl" in name:
        bl_pipeline()
        return None
    logging.basicConfig(filename=f"{log_path}{name.split('_')[0]}.log",
                        format='%(asctime)s - %(message)s', level=logging.INFO)
    logging.info(f"name: {name}")
    print(f"name: {name}")
    logging.info(f"config: {config}")
    print("STEL")

    train_tid, test_tid, enhance_tid = Preprocessor(config).get(method='load')
    executor = Executor(log_path, config)
    executor.train(train_tid, enhance_tid, test_tid)
    # executor.infer(test_tid, "model")


if __name__ == "__main__":
    """
    "STEL_ais", "STEL_taxi",
    
    "STEL_ais_bl_signature", "STEL_taxi_bl_signature", 
    "STEL_ais_bl_graph_cross", "STEL_taxi_bl_graph_cross",     
    "STEL_ais_bl_graph_cosine", "STEL_taxi_bl_graph_cosine", 
    "STEL_ais_bl_gcn_cross", "STEL_taxi_bl_gcn_cross",     
    "STEL_ais_bl_gcn_cosine", "STEL_taxi_bl_gcn_cosine",
    "STEL_ais_bl_transformer_cross", "STEL_taxi_bl_transformer_cross", 
    "STEL_ais_bl_transformer_cosine", "STEL_taxi_bl_transformer_cosine", 
    "STEL_ais_bl_lstm_cross", "STEL_taxi_bl_lstm_cross",
    "STEL_ais_bl_lstm_cosine", "STEL_taxi_bl_lstm_cosine"
    """
    name = "STEL_taxi_bl_gcn_cross"
    config = parse_config(name)
    log_path = config['path'].replace("dataset", "logs/STEL")
    pipeline()
