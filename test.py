from libTrajectory.evaluator.chart import line

if __name__ == "__main__":
    line(["./libTrajectory/logs/STEL/ais/model.json",
          "./libTrajectory/logs/STEL/taxi/model.json",
          "./libTrajectory/logs/STEL/ais/bl_signature_spatiotemporal.json",
          "./libTrajectory/logs/STEL/taxi/bl_signature_spatiotemporal.json",
          "./libTrajectory/logs/STEL/ais/bl_transformer_cross.json",
          "./libTrajectory/logs/STEL/taxi/bl_transformer_cross.json"],
         "./libTrajectory/logs/STEL/baseline_compare.pdf")
