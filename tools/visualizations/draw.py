import numpy as np
from enum import Enum
from numpy import linalg as LA
import matplotlib.pyplot as plt
import trimesh
import pymeshlab
import os

from tools.visualizations import Renderer
from tools.utils import io

# TODO move to one location
class JointType(Enum):
    ROT = 1
    TRANS = 2
    BOTH = 3

class Visualizer(Renderer):
    def __init__(self, vertices=None, faces=None, colors=None, normals=None, mask=None):
        super().__init__(vertices, faces, colors, normals, mask)
        pass

    def get_curling_arrow(color):
        arrow_ply = '/localhome/yma50/Documents/blender/arrow.ply'

        ms = pymeshlab.MeshSet()
        ms.load_new_mesh(arrow_ply)
        m = ms.current_mesh()
        vertices = m.vertex_matrix()
        normals = m.vertex_normal_matrix()
        faces = m.face_matrix()

        mesh = trimesh.Trimesh(vertices, faces=faces, vertex_colors=color, vertex_normals=normals)
        return mesh

if __name__ == '__main__':
    origin = np.array([0, 1, 0])
    axis = np.array([1, 0, 0])
    color = [0, 255, 0]
    v = Visualizer()
    v.add_trimesh_arrows([origin], [axis], colors=[color])
    v.show()
    # v.export('./test_arrow.ply')