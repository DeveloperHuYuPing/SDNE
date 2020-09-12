import warnings
import networkx as nx


class DefaultConfig(object):
    # 数据集参数
    lables_data_root = 'data/wiki_labels.txt'  # 标签存放路径
    origin_graph_root = 'data/Wiki_edgelist.txt'
    train_graph_root = 'data/Wiki_train_edgelist.txt'
    load_sdne_path = 'checkpoints/SDNE.pth'
    load_trainGraph_path = 'checkpoints/TrainGraph.pth'
    deepwalk_train_embedding = 'data/DeepWalk_train.txt'
    classification_result_path = 'result/multi_label_classification_result.txt'
    link_prediction_result_path = 'result/link_prediction_result.txt'
    visualization_result_path = 'result/visualization_result.png'
    deepwalk_link_prediction_result_path = 'result/deepwalk_link_prediction_result.txt'
    test_sizes = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]  # classification
    train_size = 0.9  # link prediction
    precisionK_list = [2, 10, 100, 500, 1000]  # link prediction

    # 训练参数
    verbose = 1
    epochs = 100

    # 模型参数 sdne
    graph = nx.read_edgelist(origin_graph_root, create_using=nx.DiGraph(), data=[('weight', int)])
    train_graph = nx.read_edgelist(train_graph_root, create_using=nx.DiGraph(), data=[('weight', int)])
    hidden_layers = [256, 128]
    alpha = 1e-5
    beta = 5
    v = 1e-5

    def parse(self, kwargs):  # 根据字典kwargs更新参数
        for k, v in kwargs.items():
            if not hasattr(self, k):
                warnings.warn('Warning:opt has not attribute %s' % k)
            setattr(self, k, v)
