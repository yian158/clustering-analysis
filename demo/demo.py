from prepare_input import read_file
from cluster import cluster
import numpy as np

if __name__ == "__main__":
    # Data sets to be clustered (Rows of X correspond to points, columns correspond to variables)
    input_file_name = './datasets/Glass.mat'
    # K : The number of clusters
    K = 6

    dataset = read_file(input_file_name)
    # cluster
    clusters, label = cluster(dataset, K, ka=10, la=1, si=(1/np.power(2, 0.5)))

