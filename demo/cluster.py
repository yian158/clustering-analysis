import scipy.spatial as spt
from scipy.spatial import distance
import networkx as nx
import numpy as np
import pymetis


def build_KNN_graph(data, N, k):
    points = [i for i in range(N)]
    edges = []

    tree = spt.cKDTree(data)
    kneighbor_indexs = []
    neighbor_indexs = []
    distance_withneighbor = []
    for i in range(N):
        point = data[i]
        distances, indexs = tree.query(point, k*3 if k*3 < N else N)
        distance_withneighbor.append([d for d in distances[1:] if d <= distances[k]])
        kneighbor_indexs.append(indexs[1:len(distance_withneighbor[i])+1])
        neighbor_indexs.append([indexs[i] for i in range(len(indexs)) if distances[i] <= distances[1]])
        edges.extend([(str(i), str(kneighbor_indexs[i][j]), distance_withneighbor[i][j]) for j in range(len(kneighbor_indexs[i]))])

    n_rknn = np.zeros(N)
    for item in kneighbor_indexs:
        n_rknn[item] = n_rknn[item] + 1

    KNN_graph = nx.Graph()
    for node in points:
        KNN_graph.add_node(str(node))
    for edge in edges:
        KNN_graph.add_edge(edge[0], edge[1], weight=edge[2])

    return KNN_graph, kneighbor_indexs, neighbor_indexs, n_rknn


def split_KNN_graph(KNN_graph, outliers, kneighbor_indexs, neighbor_indexs, thsi):
    KNN_graph_splitted = nx.Graph()
    for node in KNN_graph.nodes:
        if int(node) not in outliers:
            KNN_graph_splitted.add_node(node)
    for edge in KNN_graph.edges:
        p1, p2 = int(edge[0]), int(edge[1])
        if p1 not in outliers and p2 not in outliers:
            if (p1 in neighbor_indexs[p2] and p2 in neighbor_indexs[p1]) or (len(set(kneighbor_indexs[p1]).intersection(kneighbor_indexs[p2])) >= thsi):
                KNN_graph_splitted.add_edge(edge[0], edge[1])

    return KNN_graph_splitted


def fun_dis(c1, c2, kneighbor_indexs, data):
    res = 100000000
    ep_c1 = []
    ep_c2 = []
    for p1 in c1:
        for p2 in c2:
            if len(set(kneighbor_indexs[p1]).intersection(kneighbor_indexs[p2])) > 0 or (p2 in kneighbor_indexs[p1] and p1 in kneighbor_indexs[p2]):
                ep_c1.append(p1)
                ep_c2.append(p2)

    if len(ep_c1):
        p_c1_set = list(set(ep_c1))
        p_c2_set = list(set(ep_c2))

        rp_c1 = set(c1).difference(p_c1_set)
        ip_c1 = [p for item in [list(rp_c1.intersection(kneighbor_indexs[p])) for p in p_c1_set] for p in item]
        rp_c2 = set(c2).difference(p_c2_set)
        ip_c2 = [p for item in [list(rp_c2.intersection(kneighbor_indexs[p])) for p in p_c2_set] for p in item]

        p_c1_set.extend(list(set(ip_c1)))
        p_c2_set.extend(list(set(ip_c2)))

        res = np.mean(distance.cdist(np.array(data)[p_c1_set], np.array(data)[p_c2_set], metric='euclidean'))

    return res


def sec_screen(maxsi_index, res_cluster, data):
    n = len(maxsi_index[0])
    dm = np.full((n, 3), np.inf)
    for i in range(n):
        c1, c2 = maxsi_index[0][i], maxsi_index[1][i]
        diss = distance.cdist(np.array(data)[res_cluster[c1]], np.array(data)[res_cluster[c2]], metric='euclidean')
        dm[i, 0] = np.mean(diss)
        dm[i, 1] = np.amin(diss)
        dm[i, 2] = np.amax(diss)

    mindis = np.amin(dm[:, 1])
    mindis_index = np.where(dm[:, 1] <= mindis)

    if len(mindis_index[0]) <= 1:
        return maxsi_index[0][mindis_index[0][0]], maxsi_index[1][mindis_index[0][0]]
    else:
        mindis = np.amin(dm[:, 0])
        mindis_index = np.where(dm[:, 0] <= mindis)

        if len(mindis_index[0]) <= 1:
            return maxsi_index[0][mindis_index[0][0]], maxsi_index[1][mindis_index[0][0]]
        else:
            mindis = np.amin(dm[:, 2])
            mindis_index = np.where(dm[:, 2] <= mindis)
            return maxsi_index[0][mindis_index[0][0]], maxsi_index[1][mindis_index[0][0]]


def merge(subgraph, K, outliers, kneighbor_indexs, data):
    res_cluster = []
    for item in subgraph:
        if len(item) > 1:
            res_cluster.append([int(p) for p in item])
        else:
            outliers.append(int(list(item)[0]))

    while len(res_cluster) > K:
        n = len(res_cluster)
        dm = np.full((n, n), np.inf)

        for i in range(n):
            for j in range(i+1, n):
                dm[i][j] = fun_dis(res_cluster[i], res_cluster[j], kneighbor_indexs, data)

        mindi = np.amin(dm)
        mindi_index = np.where(dm <= mindi)
        rc_i, rc_j = (mindi_index[0][0], mindi_index[1][0]) if len(mindi_index[0]) <= 1 else sec_screen(mindi_index, res_cluster, data)

        res_cluster[rc_i].extend(res_cluster.pop(rc_j))
        print('.', end='')

    return res_cluster


def fun_cutstd(subcluster, neighbor_indexs):
    if len(subcluster) <= 2:
        return np.inf, []

    adjacency_list = []
    for p in subcluster:
        adjacency_list.append([subcluster.index(str(nei)) for nei in neighbor_indexs[int(p)] if str(nei) in subcluster])
    edgecuts, parts = pymetis.part_graph(2, adjacency_list)

    return edgecuts, parts


def cut(graph, K, neighbor_indexs):
    res_cluster = [list(item) for item in graph]
    edgecuts = []
    partcuts = []
    for subcluster in res_cluster:
        edgecut, partcut = fun_cutstd(subcluster, neighbor_indexs)
        edgecuts.append(edgecut)
        partcuts.append(partcut)

    while len(res_cluster) < K:
        minec = np.amin(edgecuts)
        minec_index = np.where(edgecuts <= minec)

        if len(minec_index[0]) == 1:
            c_index = minec_index[0][0]
        else:
            ens = [len(subc) for index, subc in enumerate(res_cluster) if index in minec_index[0]]
            minen = np.amin(ens)
            minen_index = np.where(ens <= minen)
            c_index = minec_index[0][minen_index[0][0]]

        parts = partcuts[c_index]
        c = res_cluster[c_index]

        c1_index = np.where(np.array(parts) <= 0)[0]
        c1 = list(np.array(c)[c1_index])
        c2_index = np.where(np.array(parts) >= 1)[0]
        c2 = list(np.array(c)[c2_index])
        res_cluster[c_index] = c1
        res_cluster.append(c2)

        if len(res_cluster) >= K:
            break

        edgecut1, partcut1 = fun_cutstd(c1, neighbor_indexs)
        edgecut2, partcut2 = fun_cutstd(c2, neighbor_indexs)
        edgecuts[c_index] = edgecut1
        edgecuts.append(edgecut2)
        partcuts[c_index] = partcut1
        partcuts.append(partcut2)

    return [list(map(int, item)) for item in res_cluster]


def assign_outliers(res_subgraph, outliers, data):
    K = len(res_subgraph)
    while len(outliers) > 0:
        dm = np.full((len(outliers), K), np.inf)
        for i, op in enumerate(outliers):
            dm[i, :] = [np.amin(distance.cdist([np.array(data)[op]], np.array(data)[gra], metric='euclidean')) for gra in res_subgraph]

        mindis = np.amin(dm)
        mindis_index = np.where(dm <= mindis)

        ops = []
        for i in range(len(mindis_index[0])):
            opi = mindis_index[0][i]
            l = mindis_index[1][i]
            if outliers[opi] not in ops:
                res_subgraph[l].append(outliers[opi])
                ops.append(outliers[opi])

        for op in ops:
            outliers.remove(op)
        if len(outliers)%50 == 0:
            print('.', end='')

    return res_subgraph


def cluster(data, K, ka=10, la=0.2, si=0.65):
    n_rows, n_cols = data.shape
    if K > n_rows:
        raise TypeError('Warning: K should not be larger than N.')

    k = ka
    if k > n_rows:
        print('Warning: N>k: k will be reset as N.')
        k = n_rows

    print('Clustering.', end='')
    points = [i for i in range(n_rows)]
    KNN_graph, kneighbor_indexs, neighbor_indexs, n_rknn = build_KNN_graph(data, n_rows, k)

    outliers = [i for i in points if n_rknn[i] <= sorted(n_rknn)[round(n_rows*la)]]
    KNN_graph_splitted = split_KNN_graph(KNN_graph, outliers, kneighbor_indexs, neighbor_indexs, round(k*si))
    subgraph = list(nx.connected_components(KNN_graph_splitted))

    if len(subgraph) == K:
        res_subgraph = [list(map(int, item)) for item in subgraph]
    else:
        res_subgraph = merge(subgraph, K, outliers, kneighbor_indexs, data) if len(subgraph) > K else cut(subgraph, K, kneighbor_indexs)
    res_cluster = assign_outliers(res_subgraph, outliers, data)

    cluster_label = np.zeros(n_rows)
    for index, clu in enumerate(res_cluster):
        cluster_label[clu] = (index+1)
    print('\nDone')

    return res_cluster, cluster_label
