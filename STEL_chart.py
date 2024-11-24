from libTrajectory.config.config_parser import parse_config
from libTrajectory.evaluator.chart import line, graph_bar, loss_bar, embedding_scatter, graph_bar2, merge_bar

# from libTrajectory.evaluator.embedding_compare import EmbeddingCompare

if __name__ == "__main__":
    # line(["./libTrajectory/logs/STEL/ais/model.json",
    #       "./libTrajectory/logs/STEL/taxi/model.json",
    #       "./libTrajectory/logs/STEL/ais/bl_signature_spatiotemporal.json",
    #       "./libTrajectory/logs/STEL/taxi/bl_signature_spatiotemporal.json",
    #       "./libTrajectory/logs/STEL/ais/bl_transformer_cross.json",
    #       "./libTrajectory/logs/STEL/taxi/bl_transformer_cross.json"],
    #      "./libTrajectory/logs/STEL/bl_complete_dataset.pdf")

    # graph_bar(["./libTrajectory/logs/STEL/small_ais/model.json",
    #            "./libTrajectory/logs/STEL/small_taxi/model.json",
    #            "./libTrajectory/logs/STEL/small_ais/ab_visit_graph.json",
    #            "./libTrajectory/logs/STEL/small_taxi/ab_visit_graph.json",
    #            "./libTrajectory/logs/STEL/small_ais/ab_trajectory_graph.json",
    #            "./libTrajectory/logs/STEL/small_taxi/ab_trajectory_graph.json"],
    #           "./libTrajectory/logs/STEL/ab_graph.pdf")

    # loss_bar(["./libTrajectory/logs/STEL/small_ais/model.json",
    #           "./libTrajectory/logs/STEL/small_taxi/model.json",
    #           "./libTrajectory/logs/STEL/small_ais/ab_cross_loss.json",
    #           "./libTrajectory/logs/STEL/small_taxi/ab_cross_loss.json",
    #           "./libTrajectory/logs/STEL/small_ais/ab_cosine_loss.json",
    #           "./libTrajectory/logs/STEL/small_taxi/ab_cosine_loss.json"],
    #          "./libTrajectory/logs/STEL/ab_loss.pdf")

    merge_bar([["./libTrajectory/logs/STEL/small_ais/model.json",
               "./libTrajectory/logs/STEL/small_taxi/model.json",
               "./libTrajectory/logs/STEL/small_ais/ab_visit_graph.json",
               "./libTrajectory/logs/STEL/small_taxi/ab_visit_graph.json",
               "./libTrajectory/logs/STEL/small_ais/ab_trajectory_graph.json",
               "./libTrajectory/logs/STEL/small_taxi/ab_trajectory_graph.json"],
                ["./libTrajectory/logs/STEL/small_ais/model.json",
                 "./libTrajectory/logs/STEL/small_taxi/model.json",
                 "./libTrajectory/logs/STEL/small_ais/ab_cross_loss.json",
                 "./libTrajectory/logs/STEL/small_taxi/ab_cross_loss.json",
                 "./libTrajectory/logs/STEL/small_ais/ab_cosine_loss.json",
                 "./libTrajectory/logs/STEL/small_taxi/ab_cosine_loss.json"]
                ],
               "./libTrajectory/logs/STEL/ab_graph&loss.pdf")

    # init_embedding1, init_embedding2, train_embedding1, train_embedding2 = [], [], [], []
    # for name in ["STEL_ais_embedding_compare"]:
    #     config = parse_config(name)
    #     print(config['test_file'])
    #     ie1, ie2, te1, te2 = EmbeddingCompare(config).run()
    #     init_embedding1.append(ie1)
    #     init_embedding2.append(ie2)
    #     train_embedding1.append(te1)
    #     train_embedding2.append(te2)
    #
    # embedding_scatter(init_embedding1, init_embedding2, train_embedding1, train_embedding2,
    #                   title=['Distribution of the embedding before training',
    #                          'Distribution of the embedding after training'],
    #                   save_path="./libTrajectory/logs/STEL/embedding.pdf")
