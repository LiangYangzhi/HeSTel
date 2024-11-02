import matplotlib.pyplot as plt
import numpy as np
import matplotlib.path as mpath


def line_parse_json(file_name: list):
    dataset = {"ais_test1": [], "ais_test2": [], "taxi_test1": [], "taxi_test2": []}
    test = ['test1', 'test2']
    acc = ['Acc@1', 'Acc@2', 'Acc@3', 'Acc@4', 'Acc@5']
    colors = ['r', 'r', 'r', 'r',
              'b', 'b', 'b', 'b',
              'g', 'g', 'g', 'g']
    star = mpath.Path.unit_regular_star(6)
    circle = mpath.Path.unit_circle()
    cut_star = mpath.Path(
        vertices=np.concatenate([circle.vertices, star.vertices[::-1, ...]]),
        codes=np.concatenate([circle.codes, star.codes]))
    markers = [star, star, star, star,
               circle, circle, circle, circle,
               cut_star, cut_star, cut_star, cut_star]
    line_styles = ["solid", "solid", "solid", "solid",
                   "dashed", "dashed", "dashed", "dashed",
                   "dashdot", "dashdot", "dashdot", "dashdot"]

    for path in file_name:
        with open(path) as user_file:
            data = user_file.read()
        data = eval(data)
        for key in test:
            sub_data = data[key]
            dic = {}
            dic['x'] = acc
            dic['y'] = [sub_data[i] for i in acc]
            dic['color'] = colors.pop(0)  # return pop element
            dic['marker'] = markers.pop(0)
            dic['linestyle'] = line_styles.pop(0)
            # STEL 都在path内，所以要挡在第一位
            for i in ['STEL', 'transformer', 'signature', 'lstm', 'graph', 'gcn']:
                if i in path:
                    dic['label'] = i

            if "ais" in path:
                dataset[f'ais_{key}'].append(dic)
            elif "taxi" in path:
                dataset[f'taxi_{key}'].append(dic)

    return dataset


def line(file_name, save_path):
    dataset = line_parse_json(file_name)
    # 创建示例数据

    # 创建折线图
    fig = plt.figure(figsize=(12, 10))
    plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1, wspace=0.2, hspace=0.3)
    plt.subplot(221)
    data = dataset['ais_test1']
    for dic in data:
        plt.plot(dic['x'], dic['y'], label=dic['label'], marker=dic['marker'],
                 color=dic['color'], linestyle=dic['linestyle'], linewidth=2, markersize=10)
    plt.title('AIS test dataset of 1000')
    plt.grid(True)
    plt.ylim(0.25, 0.99)

    plt.subplot(222)
    data = dataset['ais_test2']
    for dic in data:
        plt.plot(dic['x'], dic['y'], label=dic['label'], marker=dic['marker'],
                 color=dic['color'], linestyle=dic['linestyle'], linewidth=2, markersize=10)
    plt.title('AIS test dataset of 3000')
    plt.grid(True)
    plt.ylim(0.25, 0.99)

    plt.subplot(223)
    data = dataset['taxi_test1']
    for dic in data:
        plt.plot(dic['x'], dic['y'], label=dic['label'], marker=dic['marker'],
                 color=dic['color'], linestyle=dic['linestyle'], linewidth=2, markersize=10)
    plt.title('T-Drive test dataset of 1000')
    plt.grid(True)
    plt.ylim(0.25, 0.99)

    plt.subplot(224)
    data = dataset['taxi_test2']
    labels = []
    for dic in data:
        label, = plt.plot(dic['x'], dic['y'], label=dic['label'], marker=dic['marker'],
                          color=dic['color'], linestyle=dic['linestyle'], linewidth=2, markersize=10)
        labels.append(label)
    plt.title('T-Drive test dataset of 3000')
    plt.grid(True)
    plt.ylim(0.25, 0.99)

    # 添加图例
    fig.legend(labels, [label.get_label() for label in labels], loc='lower center', ncol=3)
    # 显示图表
    plt.savefig(save_path, format='pdf')
    plt.show()


def graph_bar_parse_json(file_name: list):
    test = ['test1', 'test2']
    acc = 'Acc@1'  # 'Acc@1', 'Acc@2', 'Acc@3', 'Acc@4', 'Acc@5'
    dic = {}
    file_method = ['STEL', 'visit', 'trajectory']  # STEL 都在path内，所以要挡在第一位
    for path in file_name:
        with open(path) as user_file:
            data = user_file.read()
        data = eval(data)
        for key in test:
            sub_data = data[key]
            acc_data = round(sub_data[acc], 4)
            for i in file_method:
                if i in path:
                    name = i
                    continue
            if "ais" in path:
                dic[f'{name}_ais_{key}'] = acc_data
            elif "taxi" in path:
                dic[f'{name}_taxi_{key}'] = acc_data
    dataset = {
        "colores": ['#0000FF', '#33FF57', '#FF8C00'],
        "labels": ['Number 1000 of AIS', 'Number 3000 of AIS', 'Number 1000 of T-Drive', 'Number 3000 of T-Drive'],
        "hatches": ['//', '\\\\', '--'],
        "data": [[dic[f"{i}_{j}"] for j in ['ais_test1', 'ais_test2', 'taxi_test1', 'taxi_test2']]
                 for i in file_method]
    }
    return dataset


def graph_bar(file_name, save_path):
    dataset = graph_bar_parse_json(file_name)
    print(dataset)
    colores = dataset['colores']
    labels = dataset['labels']
    hatches = dataset['hatches']
    data = dataset['data']

    # 设置条形图宽度
    bar_width = 0.15
    # 计算每组数据的位置
    x = np.arange(len(labels))
    # 创建一个Figure和一个Axes对象
    fig, ax = plt.subplots(figsize=(12, 8))

    # 绘制多条形图，并自定义颜色和标签
    ax.bar(x - 2 * bar_width, data[0], width=bar_width, label="STEL", color=colores[0],
           alpha=0.7, edgecolor='black', linewidth=1.2, hatch=hatches[0])
    ax.bar(x - 0.5 * bar_width, data[1], width=bar_width, label="Visit Graph", color=colores[1],
           alpha=0.7, edgecolor='black', linewidth=1.2, hatch=hatches[1])
    ax.bar(x + 1 * bar_width, data[2], width=bar_width, label="Trajectory Graph", color=colores[2],
           alpha=0.7, edgecolor='black', linewidth=1.2, hatch=hatches[2])

    # 添加标签和标题
    ax.set_xlabel('Dataset')
    ax.set_ylabel('ACC')
    ax.set_title('Ablation experiment of graph structure')
    # 设置x轴刻度和标签
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    # 添加图例
    ax.legend()

    # 添加数据标签
    for i, v in enumerate(data[0]):
        ax.text(i - 2 * bar_width, v + 0.01, str(v), color='black', fontsize=7, ha='center')
    for i, v in enumerate(data[1]):
        ax.text(i - 0.5 * bar_width, v + 0.01, str(v), color='black', fontsize=7, ha='center')
    for i, v in enumerate(data[2]):
        ax.text(i + 1 * bar_width, v + 0.01, str(v), color='black', fontsize=7, ha='center')
    plt.savefig(save_path, format='pdf')
    plt.show()


def loss_bar_parse_json(file_name: list):
    test = ['test1', 'test2']
    acc = 'Acc@1'  # 'Acc@1', 'Acc@2', 'Acc@3', 'Acc@4', 'Acc@5'
    dic = {}
    file_method = ['STEL', 'cross', 'cosine']  # STEL 都在path内，所以要挡在第一位
    for path in file_name:
        with open(path) as user_file:
            data = user_file.read()
        data = eval(data)
        for key in test:
            sub_data = data[key]
            acc_data = round(sub_data[acc], 4)
            for i in file_method:
                if i in path:
                    name = i
                    continue
            if "ais" in path:
                dic[f'{name}_ais_{key}'] = acc_data
            elif "taxi" in path:
                dic[f'{name}_taxi_{key}'] = acc_data
    dataset = {
        "colores": ['#00FF00', '#00FF7F', '#EEEE00'],
        "labels": ['Number 1000 of AIS', 'Number 3000 of AIS', 'Number 1000 of T-Drive', 'Number 3000 of T-Drive'],
        "hatches": ['*', '//', 'xx'],
        "data": [[dic[f"{i}_{j}"] for j in ['ais_test1', 'ais_test2', 'taxi_test1', 'taxi_test2']]
                 for i in file_method]
    }
    return dataset


def loss_bar(file_name, save_path):
    dataset = loss_bar_parse_json(file_name)
    print(dataset)
    colores = dataset['colores']
    labels = dataset['labels']
    hatches = dataset['hatches']
    data = dataset['data']

    # 设置条形图宽度
    bar_width = 0.15
    # 计算每组数据的位置
    x = np.arange(len(labels))
    # 创建一个Figure和一个Axes对象
    fig, ax = plt.subplots(figsize=(12, 8))

    # 绘制多条形图，并自定义颜色和标签
    ax.bar(x - 2 * bar_width, data[0], width=bar_width, label="STEL", color=colores[0],
           alpha=0.7, edgecolor='black', linewidth=1.2, hatch=hatches[0])
    ax.bar(x - 0.5 * bar_width, data[1], width=bar_width, label="Cross Entropy", color=colores[1],
           alpha=0.7, edgecolor='black', linewidth=1.2, hatch=hatches[1])
    ax.bar(x + 1 * bar_width, data[2], width=bar_width, label="Cosine Embedding", color=colores[2],
           alpha=0.7, edgecolor='black', linewidth=1.2, hatch=hatches[2])

    # 添加标签和标题
    ax.set_xlabel('Dataset')
    ax.set_ylabel('ACC')
    ax.set_title('Ablation experiment of loss function')
    # 设置x轴刻度和标签
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    # 添加图例
    ax.legend()

    # 添加数据标签
    for i, v in enumerate(data[0]):
        ax.text(i - 2 * bar_width, -v + 0.01, str(v), color='black', fontsize=7, ha='center')
    for i, v in enumerate(data[1]):
        ax.text(i - 0.5 * bar_width, -v + 0.01, str(v), color='black', fontsize=7, ha='center')
    for i, v in enumerate(data[2]):
        ax.text(i + 1 * bar_width, -v + 0.01, str(v), color='black', fontsize=7, ha='center')
    plt.savefig(save_path, format='pdf')
    plt.show()


# def loss_barh(file_name, save_path):
#     dataset = loss_bar_parse_json(file_name)
#     print(dataset)
#     colores = dataset['colores']
#     labels = dataset['labels']
#     hatches = dataset['hatches']
#     data = dataset['data']
#
#     # 设置条形图宽度
#     bar_height = 0.15
#     # 计算每组数据的位置
#     x = np.arange(len(labels))
#     # 创建一个Figure和一个Axes对象
#     fig, ax = plt.subplots(figsize=(12, 8))
#
#     # 绘制多条形图，并自定义颜色和标签
#     ax.barh(data[0], x * bar_height, height=bar_height, label="STEL", color=colores[0],
#            alpha=0.7, edgecolor='black', linewidth=1.2, hatch=hatches[0])
#     ax.barh(data[1], x + 6 * bar_height, height=bar_height, label="Cross Entropy", color=colores[1],
#            alpha=0.7, edgecolor='black', linewidth=1.2, hatch=hatches[1])
#     ax.barh(data[2], x + 10 * bar_height, height=bar_height, label="Cosine Embedding", color=colores[2],
#            alpha=0.7, edgecolor='black', linewidth=1.2, hatch=hatches[2])
#
#     # 添加标签和标题
#     ax.set_ylabel('Dataset')
#     ax.set_xlabel('ACC')
#     ax.set_title('Ablation experiment of loss function')
#     # 设置x轴刻度和标签
#     ax.set_yticks(x)
#     ax.set_yticklabels(labels)
#     # 添加图例
#     ax.legend()
#
#     # 添加数据标签
#     for i, v in enumerate(data[0]):
#         ax.text(i - 2 * bar_height, -v + 0.01, str(v), color='black', fontsize=7, ha='center')
#     for i, v in enumerate(data[1]):
#         ax.text(i - 0.5 * bar_height, -v + 0.01, str(v), color='black', fontsize=7, ha='center')
#     for i, v in enumerate(data[2]):
#         ax.text(i + 1 * bar_height, -v + 0.01, str(v), color='black', fontsize=7, ha='center')
#     plt.savefig(save_path, format='pdf')
#     plt.show()


def embedding_scatter(init_embedding1: list, init_embedding2: list, train_embedding1: list, train_embedding2: list,
                      title: list, save_path: str):
    plt.figure(figsize=(10, 5))
    plt.subplots_adjust(left=0.08, right=0.99, top=0.94, bottom=0.1, wspace=0.2, hspace=0.3)
    plt.subplot(121)
    plt.scatter(init_embedding1[0][:, 0], init_embedding1[0][:, 1], color='blue', label='Active Trajectory')
    plt.scatter(init_embedding2[0][:, 0], init_embedding2[0][:, 1], color='red', label='Passive Trajectory')
    plt.xlabel('Dimension 1')
    plt.ylabel('Dimension 2')
    plt.title(title[0])
    plt.legend()

    plt.subplot(122)
    plt.scatter(train_embedding1[0][:, 0], train_embedding1[0][:, 1], color='blue', label='Active Trajectory')
    plt.scatter(train_embedding2[0][:, 0], train_embedding2[0][:, 1], color='red', label='Passive Trajectory')
    plt.xlabel('Dimension 1')
    plt.ylabel('Dimension 2')
    plt.title(title[1])
    plt.legend()

    plt.savefig(save_path, format='pdf')
    plt.show()

