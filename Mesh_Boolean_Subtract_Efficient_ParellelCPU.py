import numpy as np
import pandas as pd
import scipy.io
from scipy.spatial import ConvexHull, cKDTree
import matplotlib.pyplot as plt
from alphashape import alphashape
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon
import os
from concurrent.futures import ProcessPoolExecutor
import time

# Define function for distance calculation at the top level
def calculate_distances(start, end, center_laminap, center_micro, threshold):
    tree = cKDTree(center_micro)
    min_dis, min_idx = tree.query(center_laminap[start:end])
    enf1 = np.where(min_dis < threshold)[0] + start
    enn = np.where(min_dis >= threshold)[0] + start
    return enf1, enn

def main():
    start_time = time.time()

    # Load data using pandas
    NP = pd.read_csv('Node_LaminaParent.txt', header=None).values
    EP = pd.read_csv('Element_LaminaParent.txt', header=None).values.astype(int)
    NM = pd.read_csv('Node_UnCutMicro.txt', header=None).values
    EM = pd.read_csv('Element_UnCutMicro.txt', header=None).values.astype(int)

    # Parameters
    Threshold = 0.01
    SF1 = 0.9
    SF2 = 0.7

    # Estimating Center of Elements for Full Lamina & Micro
    Center_LaminaP = np.mean(NP[EP[:, 1:] - 1, 1:], axis=1)
    Center_Micro = np.mean(NM[EM[:, 1:] - 1, 1:], axis=1)

    # Boundary of Micro
    E = np.unique(EM[:, 1:])
    Nodes_Surf = NM[E - 1, 1:]

    # Handle duplicate nodes and boundaries
    Nodes_Surf, idx = np.unique(Nodes_Surf, axis=0, return_index=True)
    hull = ConvexHull(Nodes_Surf)
    bound_nodes = Nodes_Surf[hull.vertices]

    shp = alphashape(bound_nodes, SF1)
    shp = alphashape(bound_nodes, SF2)

    # Reindex hull simplices to match bound_nodes
    hull_simplices = hull.simplices
    unique_indices = {original_idx: new_idx for new_idx, original_idx in enumerate(hull.vertices)}
    reindexed_simplices = np.vectorize(unique_indices.get)(hull_simplices)

    # Plot alphashape
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_trisurf(bound_nodes[:, 0], bound_nodes[:, 1], bound_nodes[:, 2], triangles=reindexed_simplices, cmap=plt.cm.Spectral)
    plt.show()

    # Node Centers
    Node_Parent_e2 = NP[EP[:, 1] - 1, 1:]
    Node_Parent_e3 = NP[EP[:, 2] - 1, 1:]
    Node_Parent_e4 = NP[EP[:, 3] - 1, 1:]
    Node_Parent_e5 = NP[EP[:, 4] - 1, 1:]

    Node_Parent_emean = (Node_Parent_e2 + Node_Parent_e3 + Node_Parent_e4 + Node_Parent_e5) / 4

    # Check if centers are inside the shape
    polygon = Polygon(bound_nodes)
    in_shape = np.array([polygon.contains(Point(*node)) for node in Node_Parent_emean])
    out_shape = ~in_shape

    Selected_Element_Parent_In = EP[in_shape]
    Selected_Element_Parent_Out = EP[out_shape]

    # Corresponding Nodes
    Corresponding_NodesV_In = np.unique(Selected_Element_Parent_In[:, 1:])
    Selected_Node_Parent_In = NP[Corresponding_NodesV_In - 1]

    Corresponding_NodesV_Out = np.unique(Selected_Element_Parent_Out[:, 1:])
    Selected_Node_Parent_Out = NP[Corresponding_NodesV_Out - 1]

    # Plot elements
    fig = plt.figure()
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.scatter(Node_Parent_emean[in_shape, 0], Node_Parent_emean[in_shape, 1], Node_Parent_emean[in_shape, 2], c='r')
    ax1.set_title('Elements inside Micro')
    ax1.set_xlabel('x(MicroM)')
    ax1.set_ylabel('y(MicroM)')
    ax1.set_zlabel('z(MicroM)')
    ax1.axis('equal')
    ax1.grid(True)

    ax2 = fig.add_subplot(122, projection='3d')
    ax2.scatter(Node_Parent_emean[out_shape, 0], Node_Parent_emean[out_shape, 1], Node_Parent_emean[out_shape, 2], c='b')
    ax2.set_title('Elements outside Micro')
    ax2.set_xlabel('x(MicroM)')
    ax2.set_ylabel('y(MicroM)')
    ax2.set_zlabel('z(MicroM)')
    ax2.axis('equal')
    ax2.grid(True)

    plt.show()

    # Split the work across multiple processes
    num_cores = os.cpu_count()
    chunk_size = len(Center_LaminaP) // num_cores

    futures = []
    with ProcessPoolExecutor(max_workers=num_cores) as executor:
        for i in range(num_cores):
            start = i * chunk_size
            end = (i + 1) * chunk_size if i != num_cores - 1 else len(Center_LaminaP)
            futures.append(executor.submit(calculate_distances, start, end, Center_LaminaP, Center_Micro, Threshold))

    # Collect results from all processes
    ENF1 = []
    ENN = []
    for future in futures:
        enf1_chunk, enn_chunk = future.result()
        ENF1.extend(enf1_chunk)
        ENN.extend(enn_chunk)

    ENF1 = np.array(ENF1)
    ENN = np.array(ENN)

    ENF = ENF1

    # Nodes & Elements of Lamina Fibers
    ELV_F = np.unique(EP[ENF, 1:])
    Nodes_Lamina_Fibers = NP[ELV_F - 1]
    Elements_Lamina_Fibers = EP[ENF]

    # Nodes & Elements of Neural Tissue
    ELV_N = np.unique(EP[ENN, 1:])
    Nodes_Neural_Tissue = NP[ELV_N - 1]
    Elements_Neural_Tissue = EP[ENN]

    scipy.io.savemat('Nodes_Lamina_Fibers.mat', {'Nodes_Lamina_Fibers': Nodes_Lamina_Fibers})
    scipy.io.savemat('Elements_Lamina_Fibers.mat', {'Elements_Lamina_Fibers': Elements_Lamina_Fibers})
    scipy.io.savemat('Nodes_Neural_Tissue.mat', {'Nodes_Neural_Tissue': Nodes_Neural_Tissue})
    scipy.io.savemat('Elements_Neural_Tissue.mat', {'Elements_Neural_Tissue': Elements_Neural_Tissue})

    # Generate inp for smooth (C3D10 mesh) for Lamina Fibers
    def Matlab2Abaqus(Nodes, Elements, Elements_Sets, Filename):
        with open(Filename, 'w') as file:
            file.write('*NODE, NSET=NODE\n')
            if Nodes.shape[1] == 3:
                for node in Nodes:
                    file.write(f'{int(node[0])}, {node[1]:.8f}, {node[2]:.8f}\n')
            elif Nodes.shape[1] == 4:
                for node in Nodes:
                    file.write(f'{int(node[0])}, {node[1]:.8f}, {node[2]:.8f}, {node[3]:.8f}\n')
            file.write('\n')

            for element_set in Elements_Sets:
                file.write(f'*ELEMENT, ELSET={element_set["Name"]}, TYPE={element_set["Elements_Type"]}\n')
                for elem_idx in element_set["Elements"]:
                    elem = Elements[elem_idx]
                    elem_str = ', '.join(map(str, elem))
                    file.write(f'{elem_idx + 1}, {elem_str}\n')
                file.write('\n')

    Nodes = Nodes_Lamina_Fibers
    Elements_D = Elements_Lamina_Fibers[:, 1:]

    Elements = [Elements_D[i, :] for i in range(len(Elements_Lamina_Fibers))]

    Elements_Sets = [{'Name': 'Set1', 'Elements_Type': 'C3D8', 'Elements': list(range(len(Elements_Lamina_Fibers)))}]

    currentFolder = os.getcwd()
    Filename = os.path.join(currentFolder, 'Solid.inp')
    Matlab2Abaqus(Nodes, Elements, Elements_Sets, Filename)

    # Generate inp for smooth (C3D10 mesh) for Lamina Neural Tissue
    Nodes = Nodes_Neural_Tissue
    Elements_D = Elements_Neural_Tissue[:, 1:]

    Elements = [Elements_D[i, :] for i in range(len(Elements_Neural_Tissue))]

    Elements_Sets = [{'Name': 'Set1', 'Elements_Type': 'C3D8', 'Elements': list(range(len(Elements_Neural_Tissue)))}]

    Filename = os.path.join(currentFolder, 'Fluid.inp')
    Matlab2Abaqus(Nodes, Elements, Elements_Sets, Filename)

    end_time = time.time()
    print(f"Computation Time: {end_time - start_time} seconds")

if __name__ == '__main__':
    main()
