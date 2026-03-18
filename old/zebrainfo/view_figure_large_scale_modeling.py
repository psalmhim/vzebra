import networkx as nx
import networkx.algorithms as na
import matplotlib.pyplot as plt
import numpy as np
import random
import copy
import scale_free_degree_distribution as scc


def fun_draw_sc_network(pos, sc_mat, save_file_name):
    G = nx.Graph()
    edge_list = []
    ppos = []
    for i in range(72):

        ppos.append([pos[i, 0], pos[i, 1]])

        for j in range(i + 1, 72):
            if sc_mat[i][j] > 5:

                G.add_edge(i, j)
                edge_list.append([i, j])

    ig = plt.figure(figsize=(2.75 * 2, 2.11 * 2))
    plt.axes([0.07, 0.07, 0.90, 0.90])
    nx.draw_networkx_edges(
        G, ppos, edgelist=edge_list, edge_color=[0.0, 0.0, 0.0, 0.8], width=0.5
    )
    nx.draw_networkx_nodes(G, ppos, node_size=55.1, node_color=[0.7, 0.3, 0, 0.5])
    ax = plt.gca()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["left"].set_visible(False)
    plt.savefig(save_file_name, format="svg", transparent=True, dpi=600)
    plt.show()


def fun_draw_cell_pos_network(cell_pos, save_file_name):
    ig = plt.figure(figsize=(2.75 * 2, 2.11 * 2))
    plt.axes([0.07, 0.07, 0.90, 0.90])

    cc = np.random.rand(72, 3)

    for i in range(72):

        qq = np.where(cell_pos[:, 3] == i + 1)
        if len(qq[0]) > 1:

            plt.scatter(
                cell_pos[qq, 0], cell_pos[qq, 1], cell_pos[qq, 2], c=cc[i], s=0.5
            )

    ax = plt.gca()

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.set_xticks([])
    ax.set_yticks([])
    plt.savefig(save_file_name, format="svg", transparent=True, dpi=600)
    plt.show()


def fun_figure_one_scale_model(save_file_name):

    n1 = 300

    scale_edges = scc.fun_make_scale_edges(n1)

    degree_num = np.zeros([300, 1])

    for ee in scale_edges:
        degree_num[ee[0]] += 1
        degree_num[ee[1]] += 1

    core_peri = []
    for ee in degree_num:
        core_peri.append(1 / (ee))

    high_degree = []
    middle_degree = []
    low_degree = []

    for i in range(300):

        if core_peri[i] <= 0.2:
            high_degree.append(i)
        if core_peri[i] > 0.2 and core_peri[i] <= 0.8:
            middle_degree.append(i)
        if core_peri[i] > 0.8:
            low_degree.append(i)

    pos = []
    for i in range(300):

        theta = np.random.random() * 2 * 3.141
        r = 1 / (float(degree_num[i])) * 5
        pos.append([r * np.cos(theta), r * np.sin(theta) + 10])

    G = nx.Graph()
    for i in range(len(scale_edges)):
        G.add_edge(scale_edges[i][0], scale_edges[i][1])

    ig = plt.figure(figsize=(2.75, 2.75))
    plt.axes([0.07, 0.07, 0.90, 0.90])
    nx.draw_networkx_edges(
        G, pos, edgelist=scale_edges, edge_color=[0.7, 0.3, 0, 0.5], width=2
    )
    nx.draw_networkx_nodes(G, pos, node_size=0.6, node_color=[0, 0, 0, 0.5])
    ax = plt.gca()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["left"].set_visible(False)
    plt.savefig(save_file_name, format="svg", transparent=True, dpi=600)
    plt.show()

    return


pos = np.loadtxt("zebrainfo/test_pos.txt", delimiter=",")
sc_mat = np.loadtxt("zebrainfo/zebra_sc.txt")

# fun_draw_sc_network(pos,sc_mat,'./fig9.svg');
cell_pos = np.load("zebrainfo/zebrafish_cell_xyz.npy")
cell_pos = np.array(cell_pos)
qq = np.where(cell_pos[:, 3] == 1)
fun_draw_cell_pos_network(cell_pos, "./fig10.svg")

# fun_figure_one_scale_model('./fig11.svg');
