import os
import logging
import trimesh
import pyrender
import imageio
import time
import numpy as np
from PIL import Image

from tools.utils import io
from matplotlib import cm

log = logging.getLogger('renderer')

from pyrender.constants import (DEFAULT_SCENE_SCALE,
                                DEFAULT_Z_FAR, DEFAULT_Z_NEAR)

os.environ['PYOPENGL_PLATFORM'] = 'egl'


class Renderer:
    def __init__(self, vertices=None, faces=None, colors=None, normals=None, mask=None):
        self.vertices = vertices
        self.faces = faces
        self.colors = colors
        self.mask = mask
        self.trimesh = None
        self.point_cloud = None
        self.trimesh_list = []
        self.point_cloud_list = []
        self.scene = pyrender.Scene()
        self.scene.ambient_light = [1.0, 1.0, 1.0]
        self.vertex_normals = None
        self.caption = None
        self.point_size = 2

        if vertices is not None:
            self.add_geometry(vertices, faces, colors, normals, mask)

    head_body_ratio = 1.0 / 4

    def reset(self):
        self.vertices = None
        self.faces = None
        self.colors = None
        self.mask = None
        self.trimesh = None
        self.point_cloud = None
        self.trimesh_list = []
        self.point_cloud_list = []
        self.scene = pyrender.Scene()
        self.scene.ambient_light = [1.0, 1.0, 1.0]
        self.vertex_normals = None

    def add_caption(self, caption):
        self.caption = caption

    @staticmethod
    def rgba_by_index(index, cmap_name='Set1', alpha=1.0):
        if index > 8:
            index = index % 9
        rgba = np.asarray(list(cm.get_cmap(cmap_name)(index)))
        rgba[-1] = alpha
        return rgba

    @staticmethod
    def colors_from_mask(mask, empty_first=True):
        unique_val = np.sort(np.unique(mask))
        colors = np.empty([mask.shape[0], 4])
        for i, val in enumerate(unique_val):
            if empty_first and i == 0:
                rgba = [0.5, 0.5, 0.5, 0.5]
            else:
                rgba = Renderer.rgba_by_index(val)
            colors[mask == val] = rgba
        return colors

    def load(self, mesh_path):
        self.trimesh = trimesh.load(mesh_path, force='mesh')
        assert isinstance(self.trimesh, trimesh.base.Trimesh)

    def add_geometry(self, vertices, faces=None, colors=None, normals=None, mask=None):
        if colors is None and mask is not None:
            colors = Renderer.colors_from_mask(mask, empty_first=True)
        if faces is not None:
            geo = trimesh.base.Trimesh(vertices, faces=faces, vertex_colors=colors, vertex_normals=normals)
            self.add_trimesh(geo)
        else:
            geo = trimesh.points.PointCloud(vertices, vertex_colors=colors)
            self.vertex_normals = normals if self.vertex_normals is None else np.concatenate(
                (self.vertex_normals, normals), axis=0)
            self.add_point_cloud(geo)

    def add_trimesh(self, mesh):
        self.trimesh_list.append(mesh)

    def add_point_cloud(self, point_cloud):
        self.point_cloud_list.append(point_cloud)

    def merge_point_clouds(self):
        all_vertices = None
        all_colors = None
        for point_cloud in self.point_cloud_list:
            vertices = point_cloud.vertices
            colors = point_cloud.colors
            if colors.shape[0] == 1:
                colors = np.tile(colors, (vertices.shape[0], 1))
            all_vertices = vertices if all_vertices is None else np.vstack((all_vertices, vertices))
            all_colors = colors if all_colors is None else np.vstack((all_colors, colors))
        self.point_cloud = trimesh.points.PointCloud(all_vertices, colors=all_colors)

    @staticmethod
    def draw_arrow(color=None, radius=0.01, length=0.5):
        if color is None:
            color = [1.0, 0.0, 0.0, 1.0]
        head_length = length * Renderer.head_body_ratio
        body_length = length - head_length
        head_transformation = np.eye(4)
        head_transformation[:3, 3] += [0, 0, body_length / 2.0]
        head = trimesh.creation.cone(3 * radius, head_length, sections=10, transform=head_transformation)
        body = trimesh.creation.cylinder(radius, body_length, sections=10)
        arrow = head + body
        arrow.visual.vertex_colors = color
        return arrow

    def add_arrows(self, positions, axes, color=None, radius=0.01, length=0.5):
        log.debug('add arrow')
        transformations = []
        z_axis = [0, 0, 1]
        for i, pos in enumerate(positions):
            transformation = trimesh.geometry.align_vectors(z_axis, axes[i])
            transformation[:3, 3] += pos + axes[i] * (1 - Renderer.head_body_ratio) / 2 * length
            transformations.append(transformation)
        arrow = Renderer.draw_arrow(color, radius, length)
        arrows = pyrender.Mesh.from_trimesh(arrow, poses=transformations)
        self.scene.add(arrows)

    def add_trimesh_arrows(self, positions, axes, colors=[None], radius=0.01, length=0.5):
        log.debug('add trimesh arrow')
        arrows = []
        z_axis = [0, 0, 1]
        for i, pos in enumerate(positions):
            arrow_length = length if isinstance(length, float) else length[i]
            if arrow_length < 10e-6:
                continue
            transformation = trimesh.geometry.align_vectors(z_axis, axes[i])
            transformation[:3, 3] += pos + axes[i] * (1 - Renderer.head_body_ratio) / 2 * arrow_length
            if len(colors) != len(positions):
                color = None
            else:
                color = colors[i]
            arrow = Renderer.draw_arrow(color, radius, arrow_length)
            arrow.apply_transform(transformation)
            arrows.append(arrow)
        self.trimesh_list += arrows

    def _merge_geometries(self):
        if len(self.trimesh_list) > 0:
            log.debug('concatenate triangle meshes')
            self.trimesh = trimesh.util.concatenate(self.trimesh_list)
        if len(self.point_cloud_list) > 0:
            log.debug('concatenate point clouds')
            self.merge_point_clouds()

    def _add_geometries_to_scenen(self):
        if self.trimesh is None or self.point_cloud is None:
            self._merge_geometries()
        if self.trimesh is not None:
            log.debug('add trimesh to scene')
            mesh = pyrender.Mesh.from_trimesh(self.trimesh, smooth=False)
            self.scene.add(mesh)
        if self.point_cloud is not None:
            log.debug('add point cloud to scene')
            rgb_color = self.point_cloud.colors.astype(float) / 255.0
            rgb_color[:, :3] = rgb_color[:, :3] * rgb_color[:, 3].reshape(-1, 1)
            if self.vertex_normals is None:
                point_cloud = pyrender.Mesh.from_points(self.point_cloud.vertices, colors=rgb_color)
            else:
                point_cloud = pyrender.Mesh.from_points(self.point_cloud.vertices, colors=rgb_color,
                                                        normals=self.vertex_normals)
            self.scene.add(point_cloud)

    def show(self, window_size=None, window_name='Default Renderer', non_block=False):
        self._add_geometries_to_scenen()
        if window_size is None:
            window_size = [800, 600]
        if non_block:
            v = pyrender.Viewer(self.scene, viewport_size=window_size, window_title=window_name,
                                point_size=self.point_size,
                                caption=self.caption, run_in_thread=True)

            time.sleep(1.0)
            v.close_external()
            while v.is_active:
                pass
        else:
            v = pyrender.Viewer(self.scene, viewport_size=window_size, window_title=window_name,
                                point_size=self.point_size,
                                caption=self.caption)

    def _compute_initial_camera_pose(self, angle=0):
        centroid = self.scene.centroid
        scale = self.scene.scale
        if scale == 0.0:
            scale = DEFAULT_SCENE_SCALE

        look_at_pos = centroid
        h_fov = np.pi / 6.0
        dist = scale / (1 * np.tan(h_fov))
        camera_pos = dist * np.array([np.cos(angle), np.sin(angle), 1.0]) + centroid

        forward = camera_pos - look_at_pos
        forward /= np.linalg.norm(forward)
        world_up = np.array([0, 1, 0])
        right = np.cross(world_up, forward)
        up = np.cross(forward, right)

        look_at = np.vstack((right, up, forward, camera_pos))
        cp = np.eye(4)
        cp[:3, :4] = look_at.T

        return cp

    def render(self, fig_path, as_gif=False, fig_size=None):
        self._add_geometries_to_scenen()

        if fig_size is None:
            fig_size = [1024, 768]
        renderer = pyrender.OffscreenRenderer(viewport_width=fig_size[0], viewport_height=fig_size[1],
                                              point_size=self.point_size)
        z_far = max(self.scene.scale * 10.0, DEFAULT_Z_FAR)
        if self.scene.scale == 0:
            z_near = DEFAULT_Z_NEAR
        else:
            z_near = min(self.scene.scale / 10.0, DEFAULT_Z_NEAR)
        cam = pyrender.PerspectiveCamera(yfov=np.pi / 6.0, znear=z_near, zfar=z_far)
        cam_pose = self._compute_initial_camera_pose()
        cam_node = pyrender.Node(camera=cam, matrix=cam_pose)
        self.scene.add_node(cam_node)

        if as_gif:
            io.ensure_dir_exists(os.path.dirname(fig_path))
            with imageio.get_writer(fig_path, mode='I', fps=10) as writer:
                for i in range(0, 360, 10):
                    transform = self._compute_initial_camera_pose(np.pi / 180 * i)
                    self.scene.set_pose(cam_node, pose=transform)
                    color, depth = renderer.render(self.scene, pyrender.constants.RenderFlags.SKIP_CULL_FACES)
                    writer.append_data(color)
        else:
            color, depth = renderer.render(self.scene)
            image = Image.fromarray(color.astype('uint8'), 'RGB')
            io.ensure_dir_exists(os.path.dirname(fig_path))
            image.save(fig_path)

        renderer.delete()
        del renderer
        return color, depth

    def export(self, mesh_path):
        if self.trimesh is None or self.point_cloud is None:
            self._merge_geometries()
        if self.point_cloud is not None:
            mesh = trimesh.base.Trimesh(self.point_cloud.vertices, vertex_colors=self.point_cloud.colors)
        if self.trimesh is not None:
            if self.point_cloud is not None:
                mesh = trimesh.util.concatenate(self.trimesh, mesh)
            else:
                mesh = self.trimesh
        io.ensure_dir_exists(os.path.dirname(mesh_path))
        mesh.export(mesh_path)
