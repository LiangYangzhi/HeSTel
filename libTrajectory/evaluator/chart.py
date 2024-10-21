import matplotlib.pyplot as plt


def parse_json(file_name: list):
    dataset = {"ais_test1": [], "ais_test2": [], "taxi_test1": [], "taxi_test2": []}
    test = ['test1', 'test2']
    acc = ['Acc@1', 'Acc@2', 'Acc@3', 'Acc@4', 'Acc@5']
    colors = ['#FF00FF', '#FF00FF', '#FF00FF', '#FF00FF',
              '#FFFF00', '#FFFF00', '#FFFF00', '#FFFF00',
              '#00BFFF', '#00BFFF', '#00BFFF', '#00BFFF',
              '#00C5CD', '#00C5CD', '#00C5CD', '#00C5CD']
    # [cm.Greys, cm.Purples, cm.Blues, cm.Greens, cm.Oranges, cm.Reds, cm.YlOrBr, cm.YlOrRd, cm.OrRd, cm.PuRd, cm.RdPu,
    #       cm.BuPu, cm.GnBu, cm.PuBu, cm.YlGnBu, cm.PuBuGn, cm.BuGn, cm.YlGn]
    line_styles = ['-',  '-', '-', '-',
                   ':', ':', ':', ':',
                   '--', '--', '--', '--',
                   '-.', '-.', '-.', '-.'
                   ]

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
            dic['linestyle'] = line_styles.pop(0)
            for i in ['STEL', 'transformer', 'signature', 'lstm', 'graph', 'gcn']:
                if i in path:
                    dic['label'] = i

            if "ais" in path:
                dataset[f'ais_{key}'].append(dic)
            elif "taxi" in path:
                dataset[f'taxi_{key}'].append(dic)

    return dataset


def line(file_name, save_path):
    dataset = parse_json(file_name)
    # 创建示例数据

    # 创建折线图
    fig = plt.figure(figsize=(12, 10))
    plt.subplots_adjust(left=0.05, right=0.99, top=0.96, bottom=0.08, wspace=0.2, hspace=0.3)
    plt.subplot(221)
    data = dataset['ais_test1']
    for dic in data:
        plt.plot(dic['x'], dic['y'], label=dic['label'], marker='o',
                 color=dic['color'], linestyle=dic['linestyle'], linewidth=2)
    plt.title('AIS test dataset of 1000')
    plt.ylim(0.25, 0.99)

    plt.subplot(222)
    data = dataset['ais_test2']
    for dic in data:
        plt.plot(dic['x'], dic['y'], label=dic['label'], marker='o',
                 color=dic['color'], linestyle=dic['linestyle'], linewidth=2)
    plt.title('AIS test dataset of 3000')
    plt.ylim(0.25, 0.99)

    plt.subplot(223)
    data = dataset['taxi_test1']
    for dic in data:
        plt.plot(dic['x'], dic['y'], label=dic['label'], marker='o',
                 color=dic['color'], linestyle=dic['linestyle'], linewidth=2)
    plt.title('T-Drive test dataset of 1000')
    plt.ylim(0.25, 0.99)

    plt.subplot(224)
    data = dataset['taxi_test2']
    labels = []
    for dic in data:
        label, = plt.plot(dic['x'], dic['y'], label=dic['label'], marker='o',
                          color=dic['color'], linestyle=dic['linestyle'], linewidth=2)
        labels.append(label)
    plt.title('T-Drive test dataset of 3000')
    plt.ylim(0.25, 0.99)

    # 添加图例
    fig.legend(labels, [label.get_label() for label in labels], loc='lower center', ncol=3)
    # 显示图表
    plt.savefig(save_path, format='pdf')
    plt.show()
