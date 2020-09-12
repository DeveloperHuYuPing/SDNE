from torch.utils import data
import random
from config import DefaultConfig
from models import SDNE

opt = DefaultConfig()


def getTrainData():
    with open(opt.origin_graph_root, 'r') as forigin:
        list = forigin.readlines()
        k = int(len(list) * opt.train_size)
        random.shuffle(list)
        sdne = SDNE(opt.graph, opt.hidden_layers, opt.alpha, opt.beta, opt.v)
        _, _1, degreeMatrix = sdne.getALMatrix()  # (2405, 2405)除了对角线，其余元素的值都为0
        degreeMatrix = degreeMatrix.tocsr()
        print(degreeMatrix)
        while len(list) > k:
            a = random.randint(0, len(list) - 1)
            b = list[a].strip().split(' ')
            if degreeMatrix[sdne.nodeToIdx[b[0]], sdne.nodeToIdx[b[0]]] > 2 and \
                    degreeMatrix[sdne.nodeToIdx[b[1]], sdne.nodeToIdx[b[1]]] > 2:
                del list[a]
                print(degreeMatrix[sdne.nodeToIdx[b[0]], sdne.nodeToIdx[b[0]]],
                      degreeMatrix[sdne.nodeToIdx[b[1]], sdne.nodeToIdx[b[1]]])
                degreeMatrix[sdne.nodeToIdx[b[0]], sdne.nodeToIdx[b[0]]] -= 1
                degreeMatrix[sdne.nodeToIdx[b[1]], sdne.nodeToIdx[b[1]]] -= 1
    with open(opt.train_graph_root, 'w') as f:
        for line in list:
            line = line + '\n'
        f.writelines(list)


class Data(data.Dataset):
    def __init__(self, root):
        self.X, self.y,  = [], []
        with open(root, 'r') as f:
            for line in f.readlines():
                a = line.strip().split(' ')
                self.X.append(a[0])
                self.y.append(a[1:])

    def __getitem__(self, index):
        node = self.X[index]
        lable = self.y[index]
        return node, lable

    def __len__(self):  # 返回节点个数
        return len(self.X)
