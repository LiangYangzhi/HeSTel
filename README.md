# LibTrajectory

LibTrajectory是一个专门研究时空轨迹数据的实验室，可为研究人员提供了可靠的实验工具和便捷的开发框架。

LibTrajectory currently supports the following tasks:

* Trajectory Similar
* Community Detection

## Overall Framework

LibTrajectory的整体框架如下所示，包含了以下七个主要模块：

- **Cofiguration Module**：负责管理框架中涉及的所有参数。对应包的名称：**config**
- **Data Module**：负责加载数据集。对应包的名称：**dataset**
- **Preprocessing Module**：负责数据预处理操作。对应包的名称：**preprocessing**
- **Model Module**：负责初始化基线模型和自定义模型。对应包的名称：**model**
- **Evaluation Module**：负责通过多个指标评估模型预测结果和可视化操作。对应包的名称：**evaluator**
- **Execution Module**：负责模型训练与预测。对应包的名称：**executor**
- pipeline：负责编排整体流程。对应包的名称：**pipeline**

## Dataset





