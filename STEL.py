import logging
from libTrajectory.config.config_parser import parse_config
from libTrajectory.preprocessing.STEL.preprocessor import Preprocessor
from libTrajectory.executor.STEL import Executor, BLExecutor, AbExecutor


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
        print(f"net: {config['executor']['net_name']}")
        executor.gcn_cross(train_tid, test_tid)
    if "gcn_cosine" in name:
        logging.info("gcn_cosine")
        print(f"net: {config['executor']['net_name']}")
        executor.gcn_cosine(train_tid, test_tid)
    # graph
    if "graph_cross" in name:
        logging.info("graph_cross")
        print(f"net: {config['executor']['net_name']}")
        executor.graph_cross(train_tid, test_tid)  # batch_size不宜过大
    if "graph_cosine" in name:
        logging.info("graph_cosine")
        print(f"net: {config['executor']['net_name']}")
        executor.graph_cosine(train_tid, test_tid)
    # transformer
    if "transformer_cross" in name:
        logging.info("transformer_cross")
        print(f"net: {config['executor']['net_name']}")
        executor.transformer_cross(train_tid, test_tid)
    if "transformer_cosine" in name:
        logging.info("transformer_cosine")
        print(f"net: {config['executor']['net_name']}")
        executor.transformer_cosine(train_tid, test_tid)
    # lstm
    if "lstm_cross" in name:
        logging.info("lstm_cross")
        print(f"net: {config['executor']['net_name']}")
        executor.lstm_cross(train_tid, test_tid)
    if "lstm_cosine" in name:
        logging.info("lstm_cosine")
        print(f"net: {config['executor']['net_name']}")
        executor.lstm_cosine(train_tid, test_tid)


def ab_pipeline():
    logging.basicConfig(filename=f"{log_path}{name.split('_', 2)[-1]}.log",
                        format='%(asctime)s - %(message)s', level=logging.INFO)
    logging.info(f"name: {name}")
    print(f"name: {name}")
    logging.info(f"config: {config}")
    print(f"net: {config['executor']['net_name']}")

    train_tid, test_tid, enhance_tid = Preprocessor(config).get(method='load')

    # augment
    if "augment" in name:
        logging.info(f"augment")
        print(f"augment")
        executor = Executor(log_path, config)
        executor.train(train_tid, enhance_tid, test_tid)

    # graph
    if "trajectory_graph" in name:
        logging.info(f"trajectory_graph")
        print(f"trajectory_graph")
        executor = Executor(log_path, config)
        executor.train(train_tid, enhance_tid, test_tid, graph_method='trajectory_graph')
    if "visit_graph" in name:
        logging.info(f"visit_graph")
        print(f"visit_graph")
        executor = Executor(log_path, config)
        executor.train(train_tid, enhance_tid, test_tid, graph_method='visit_graph')

    # net
    if "gcn_net" in name:
        logging.info(f"gcn_net")
        print(f"gcn_net")
        executor = Executor(log_path, config)
        executor.train(train_tid, enhance_tid, test_tid, net_method='gcn_net')
    if "graph_net" in name:
        logging.info(f"graph_net")
        print(f"graph_net")
        executor = Executor(log_path, config)
        executor.train(train_tid, enhance_tid, test_tid, net_method='graph_net')
    if "gat_net" in name:
        logging.info(f"gat_net")
        print(f"gat_net")
        executor = Executor(log_path, config)
        executor.train(train_tid, enhance_tid, test_tid, net_method="gat_net")

    # loss
    if "cross_loss" in name:
        logging.info(f"cross_loss")
        print(f"cross_loss")
        executor = AbExecutor(log_path, config)
        executor.cross_loss(train_tid, enhance_tid, test_tid)
    if "cosine_loss" in name:
        logging.info(f"cosine_loss")
        print(f"cosine_loss")
        executor = AbExecutor(log_path, config)
        executor.cosine_loss(train_tid, enhance_tid, test_tid)

    # strategy
    if "retention_strategy" in name:
        logging.info(f"retention_strategy")
        print(f"retention_strategy")
        executor = Executor(log_path, config)
        executor.train(train_tid, enhance_tid, test_tid)
    if "deletion_strategy" in name:
        logging.info(f"deletion_strategy")
        print(f"deletion_strategy")
        executor = Executor(log_path, config)
        executor.train(train_tid, enhance_tid, test_tid)
    if "no_strategy" in name:
        logging.info(f"no_strategy")
        print(f"no_strategy")
        executor = Executor(log_path, config)
        executor.train(train_tid, enhance_tid, test_tid)


def pipeline():
    if "bl" in name:
        bl_pipeline()
        return None
    elif "ab" in name:
        ab_pipeline()
        return None

    logging.basicConfig(filename=f"{log_path}{name.split('_')[0]}.log",
                        format='%(asctime)s - %(message)s', level=logging.INFO)
    logging.info(f"name: {name}")
    print(f"name: {name}")
    logging.info(f"config: {config}")
    print(f"net: {config['executor']['net_name']}")

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
    
    "STEL_ais_ab_augment_parameter_1_1_1", "STEL_taxi_ab_augment_parameter_1_1_1", 
    "STEL_ais_ab_augment_parameter_1_1_10", "STEL_taxi_ab_augment_parameter_1_1_10", 
    "STEL_ais_ab_augment_parameter_1_10_1", "STEL_taxi_ab_augment_parameter_1_10_1", 
    
    "STEL_ais_ab_trajectory_graph", "STEL_taxi_ab_trajectory_graph", 
    "STEL_ais_ab_visit_graph", "STEL_taxi_ab_visit_graph", 
    "STEL_ais_ab_gcn_net", "STEL_taxi_ab_gcn_net", 
    "STEL_ais_ab_graph_net", "STEL_taxi_ab_graph_net", 
    "STEL_ais_ab_gat_net", "STEL_taxi_ab_gat_net", 
    "STEL_ais_ab_cross_loss", "STEL_taxi_ab_cross_loss", 
    "STEL_ais_ab_cosine_loss", "STEL_taxi_ab_cosine_loss", 
    "STEL_ais_ab_retention_strategy", "STEL_taxi_ab_retention_strategy", 
    "STEL_ais_ab_deletion_strategy", "STEL_taxi_ab_deletion_strategy", 
    "STEL_ais_ab_no_strategy", "STEL_taxi_ab_no_strategy", 
    
    """
    name = "STEL_ais"
    config = parse_config(name)
    log_path = config['path'].replace("dataset", "logs/STEL")
    pipeline()
