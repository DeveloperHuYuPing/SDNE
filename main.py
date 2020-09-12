from inspect import getsource

import fire

from config import DefaultConfig
from data import Data
from models import SDNE
from utils.utils import *
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

opt = DefaultConfig()


def train(**kwargs):
    opt.parse(kwargs)
    sdne = SDNE(opt.graph, opt.hidden_layers, opt.alpha, opt.beta, opt.v)
    sdne.train(opt.epochs, opt.verbose)
    sdne.save(opt.load_sdne_path)
    trainGraph = SDNE(opt.train_graph, opt.hidden_layers, opt.alpha, opt.beta, opt.v)
    trainGraph.train(opt.epochs, opt.verbose)
    trainGraph.save(opt.load_trainGraph_path)


def multi_label_classification(**kwargs):
    opt.parse(kwargs)
    sdne = SDNE(opt.graph, opt.hidden_layers, opt.alpha, opt.beta, opt.v)
    sdne.load(opt.load_sdne_path)
    tmp = np.array(list(sdne.getEmbeddings().values()))
    embeddings, result = [], []
    data = Data(opt.lables_data_root)
    for node in range(sdne.nodeSize):
        embeddings.append(tmp[sdne.nodeToIdx[str(node)]])
    for test_size in opt.test_sizes:
        result.append('test_size:' + str(test_size))
        result.append(str(classification(embeddings, data.y, testSize=test_size)))
    for i in range(len(result)):
        print(result[i])
    with open(opt.classification_result_path, 'w') as f:
        for line in result:
            line = line + '\n'
            f.write(line)


def link_prediction(**kwargs):
    opt.parse(kwargs)
    sdne = SDNE(opt.graph, opt.hidden_layers, opt.alpha, opt.beta, opt.v)
    sdne.load(opt.load_sdne_path)
    trainGraph = SDNE(opt.train_graph, opt.hidden_layers, opt.alpha, opt.beta, opt.v)
    trainGraph.load(opt.load_trainGraph_path)
    embedding = np.array(list(trainGraph.getEmbeddings().values()))
    precisionK = linkPrediction(embedding, trainGraph, sdne, opt.precisionK_list)
    result = []
    for x in range(len(precisionK)):
        result.append('precision@%d : %f' % (opt.precisionK_list[x], precisionK[x]))
    for i in range(len(result)):
        print(result[i])
    with open(opt.link_prediction_result_path, 'w') as f:
        for line in result:
            line = line + '\n'
            f.write(line)


def visualization(**kwargs):
    opt.parse(kwargs)
    sdne = SDNE(opt.graph, opt.hidden_layers, opt.alpha, opt.beta, opt.v)
    sdne.load(opt.load_sdne_path)
    embeddings = sdne.getEmbeddings()
    data = Data(opt.lables_data_root)
    emb = []
    for node in data.X:
        emb.append(embeddings[node])
    tsne = TSNE()
    node_tsned = tsne.fit_transform(np.asarray(emb), data.y)
    color_idx = {}
    for i in range(len(data.X)):
        color_idx.setdefault(data.y[i][0], [])
        color_idx[data.y[i][0]].append(i)
    for c, idx in color_idx.items():
        plt.scatter(node_tsned[idx, 0], node_tsned[idx, 1], label=c)
    plt.legend()
    plt.savefig(opt.visualization_result_path)
    plt.show()


def deepWalk_link_prediction():
    sdne = SDNE(opt.graph, opt.hidden_layers, opt.alpha, opt.beta, opt.v)
    sdne.load(opt.load_sdne_path)
    trainGraph = SDNE(opt.train_graph, opt.hidden_layers, opt.alpha, opt.beta, opt.v)
    trainGraph.load(opt.load_trainGraph_path)
    embedding, node, = [], []
    with open(opt.deepwalk_train_embedding, 'r') as f:
        f.readline()
        for line in f.readlines():
            a = line.strip().split(' ')
            node.append(a[0])
            for i in range(len(a)):
                a[i] = float(a[i])
            embedding.append(a[1:])
    embeddings = []
    for i in range(sdne.nodeSize):
        embeddings.append(0)
    for i in range(len(node)):
        embeddings[sdne.nodeToIdx[node[i]]] = embedding[i]
    embeddings = np.array(embeddings)
    precisionK = linkPrediction(embeddings, trainGraph, sdne, opt.precisionK_list)
    result = []
    print('DeepWalk link prediction')
    for x in range(len(precisionK)):
        result.append('precision@%d : %f' % (opt.precisionK_list[x], precisionK[x]))
    for i in range(len(result)):
        print(result[i])
    with open(opt.deepwalk_link_prediction_result_path, 'w') as f:
        for line in result:
            line = line + '\n'
            f.write(line)


def help():  # 打印帮助的信息
    print('''
    usage : python {0} <function> [--args=value,]
    <function> := train | multi_label_classification | link_prediction | visualization | help
    example:
            python {0} train --verbose=1
            python {0} multi_label_classification 
            python {0} link_prediction
            python {0} visualization
            python {0} help
    available args:'''.format(__file__))
    source = (getsource(opt.__class__))
    print(source)


if __name__ == '__main__':
    fire.Fire()
