import scipy.sparse as sparse
import torch as t
import torch.nn as nn


class BasicModule(nn.Module):
    def __init__(self):
        super(BasicModule, self).__init__()

    def load(self, path):  # 加载指定路径的模型
        self.load_state_dict(t.load(path))

    def save(self, path):  # 保存模型
        t.save(self.state_dict(), path)


class Model(nn.Module):
    def __init__(self, inputDim, hidLayers, alpha, beta, v):
        super(Model, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.v = v
        inputDim1 = inputDim
        encoderlayers = []
        i = 1
        for layerDim in hidLayers:
            encoderlayers.append(nn.Linear(inputDim, layerDim))
            encoderlayers.append(nn.ReLU())
            inputDim = layerDim
            i += 1
        self.encoder = nn.Sequential(*encoderlayers)
        decoderlayers = []
        i = 1
        for layerDim in reversed(hidLayers[:-1]):
            decoderlayers.append(nn.Linear(inputDim, layerDim))
            decoderlayers.append(nn.ReLU())
            inputDim = layerDim
            i += 1
        decoderlayers.append(nn.Linear(inputDim, inputDim1))
        decoderlayers.append(nn.ReLU())
        self.decoder = nn.Sequential(*decoderlayers)
        self.weightList = self.getWeightList()

    def forward(self, X, L):
        Y = self.encoder(X)
        X1 = self.decoder(Y)
        self.weightList = self.getWeightList()
        loss1 = self.alpha * 2 * t.trace(t.matmul(t.matmul(Y.transpose(0, 1), L), Y))
        B = t.ones_like(X)
        B[X != 0] = self.beta
        loss2 = t.norm((X1 - X) * B, 2)
        regLoss = 0
        for name, w in self.weightList:
            regLoss += t.norm(w, 2)
        regLoss *= self.v * 0.5
        return loss2 + loss1 + regLoss

    def getWeightList(self):
        weightList = []
        for name, param in self.named_parameters():
            if 'weight' in name:
                weight = (name, param)
                weightList.append(weight)
        return weightList


class SDNE(BasicModule):
    def __init__(self, graph, hidden_layers, alpha, beta, v):
        super().__init__()
        self.graph = graph
        self.hidden_layers = hidden_layers
        self.nodeSize = self.graph.number_of_nodes()
        self.edgeSize = self.graph.number_of_edges()
        self.idxToNode, self.nodeToIdx = [], {}
        i = 0
        for node in self.graph.nodes():
            self.nodeToIdx[node] = i
            self.idxToNode.append(node)
            i += 1
        self.sdne = Model(self.nodeSize, hidden_layers, alpha, beta, v)
        self.embeddings = {}
        aMatrix, lMatrix, _ = self.getALMatrix()
        self.aMatrix = t.from_numpy(aMatrix.toarray()).float().to('cpu')
        self.lMatrix = t.from_numpy(lMatrix.toarray()).float().to('cpu')

    def getEmbeddings(self):
        if not self.embeddings:
            with t.no_grad():
                self.sdne.eval()
                for i, embedding in enumerate(self.sdne.encoder(self.aMatrix).numpy()):
                    self.embeddings[self.idxToNode[i]] = embedding
        return self.embeddings

    def train(self, epochs, verbose=1):
        for i in range(int(len(self.hidden_layers) / 2)):
            nn.init.xavier_uniform_(self.sdne.encoder[2 * i].weight)
            nn.init.zeros_(self.sdne.encoder[2 * i].bias)
            nn.init.xavier_uniform_(self.sdne.decoder[-2 * (i + 1)].weight)
            nn.init.zeros_(self.sdne.decoder[-2 * (i + 1)].bias)
        optimizer = t.optim.Adam(self.sdne.parameters())
        self.sdne.to('cpu')
        for epoch in range(epochs):
            optimizer.zero_grad()
            loss = self.sdne(self.aMatrix, self.lMatrix)
            loss_epoch = loss.item()
            loss.backward()
            optimizer.step()
            if verbose > 0:
                print('Epoch {0},loss {1}'.format(epoch + 1, round(loss_epoch / self.nodeSize, 4)))

    def getALMatrix(self):
        nodeSize = self.nodeSize
        aMatrixData, aMatrixRow, aMatrixCol = [], [], []
        for v1, v2 in self.graph.edges():
            edgeWeight = self.graph[v1][v2].get('weight', 1.0)
            aMatrixData.append(edgeWeight)
            aMatrixRow.append(self.nodeToIdx[v1])
            aMatrixCol.append(self.nodeToIdx[v2])
        aMatrix = sparse.csr_matrix((aMatrixData, (aMatrixRow, aMatrixCol)), (nodeSize, nodeSize))
        aMatrix1 = sparse.csr_matrix((aMatrixData + aMatrixData,
                                      (aMatrixRow + aMatrixCol, aMatrixCol + aMatrixRow)), (nodeSize, nodeSize))
        degreeMatrix = sparse.diags(aMatrix1.sum(1).flatten().tolist()[0])
        lMatrix = degreeMatrix - aMatrix1
        return aMatrix, lMatrix, degreeMatrix
