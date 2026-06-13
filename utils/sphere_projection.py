import os.path

from models.siren import Siren
from utils.mesh import Mesh
from utils.camera import PerspectiveCamera
from embreex import rtcore_scene
from embreex.mesh_construction import TriangleMesh
import time
from PIL import Image
from pyproj import Proj

from utils.video import VideoWriter
from tqdm import tqdm
import numpy as np
import torch
import hashlib


def hash_tuple(tup):
    """
    Hash a tuple of values
    :param tup: Tuple of values
    :return: Hash value
    """
    tuple_str = ""
    for item in tup:
        if isinstance(item, float):
            item = str(f"{item:.4f}")

        tuple_str += str(item) + "_"

    return hashlib.sha256(tuple_str.encode()).hexdigest()[:8]


class SphereProjection:
    def __init__(self, mesh: Mesh, height, width, max_elevation=60.0, max_azimuth=120.0, max_range=0.5,
                 num_views=1, max_views=16, save_path="data/projections",
                 device='cuda:0'):
        """
        :param mesh: Mesh object. The mesh should be an icosphere.
        :param height: The patch on the sphere will be divided to a grid of height x width
        :param width: The patch on the sphere will be divided to a grid of height x width
        :param max_elevation: Maximum value for the elevation of the center of the patch
        :param max_azimuth: Maximum value for the azimuth of the center of the patch
        :param max_range: This determines the relative size of the patch to the size of the sphere
                          1.0 will cover the whole sphere. To avoid distortions use values < 0.6
        :param num_views: Number of views to sample from the cached projections
        :param max_views: Number of random views to generate
        :param save_path: Path to save the cached projections
        :param device: PyTorch device to store the data
        """
        assert mesh.is_icosphere
        with torch.no_grad():
            self.device = device
            self.V = mesh.vertices.cpu().numpy()
            self.F = mesh.faces.cpu().numpy()
            self.scene = rtcore_scene.EmbreeScene()
            self.mesh_name = mesh.mesh_name
            self.mesh = TriangleMesh(self.scene, vertices=self.V, indices=self.F)

            self.max_elevation = max_elevation
            self.max_azimuth = max_azimuth
            self.max_range = max_range * (6371 * 1000 * np.sqrt(2))  # To convert to earth size

            self.height = height
            self.width = width
            self.max_views = max_views
            self.num_views = num_views

            self.save_path = save_path

            self.config_hash = hash_tuple((self.mesh_name, self.max_elevation, self.max_azimuth,
                                           self.max_range, self.height, self.width, self.max_views))

            self._cache_views()

    @torch.no_grad()
    def _cache_views(self):
        config_hash = self.config_hash
        if os.path.exists(os.path.join(self.save_path, f"{config_hash}.pt")):
            print(f"Loading cached projections from {self.save_path}/{config_hash}.pt")
            data = torch.load(os.path.join(self.save_path, f"{config_hash}.pt"), map_location=self.device)
            self.barycentric_coords = data['barycentric_coords']
            self.faces = data['faces']
            return

        barycentric_coords_list = []
        face_list = []

        for _ in tqdm(range(self.max_views), desc="Caching random views "):
            center_phi = np.random.uniform(-self.max_azimuth // 2, self.max_azimuth // 2)
            center_theta = np.random.uniform(-self.max_elevation // 2, self.max_elevation // 2)
            self.proj_lambert = Proj(proj="laea", lat_0=center_theta, lon_0=center_phi)
            x = np.linspace(-self.max_range, self.max_range, self.width)
            y = np.linspace(self.max_range, -self.max_range, self.height)
            x, y = np.meshgrid(x, y)
            x, y = x.flatten(), y.flatten()
            phi, theta = self.proj_lambert(x, y, inverse=True)
            theta = np.radians(theta)
            phi = np.radians(phi)
            ray_origins = np.zeros((self.height * self.width, 3), dtype=np.float32)
            ray_directions = np.stack([
                np.cos(theta) * np.cos(phi),
                np.sin(theta),
                np.cos(theta) * np.sin(phi),
            ], axis=-1).astype(np.float32)  # Assuming that the y-axis is the up vector

            ray_intersections = self.scene.run(ray_origins, ray_directions, output=1)

            u, v = ray_intersections['u'], ray_intersections['v']
            w = 1.0 - u - v
            face_ids = ray_intersections['primID']

            barycentric_coords = np.stack([w, u, v], axis=-1).reshape(self.height, self.width, 3)
            faces = self.F[face_ids].reshape(self.height, self.width, 3)

            barycentric_coords_list.append(barycentric_coords)
            face_list.append(faces)

        barycentric_coords = np.stack(barycentric_coords_list, axis=0)
        faces = np.stack(face_list, axis=0)
        self.barycentric_coords = torch.tensor(barycentric_coords, dtype=torch.float32, device=self.device)
        self.faces = torch.tensor(faces, dtype=torch.int64, device=self.device)

        torch.save({
            'barycentric_coords': self.barycentric_coords,
            'faces': self.faces
        }, os.path.join(self.save_path, f"{config_hash}.pt"))

    @torch.no_grad()
    def generate_random_view_projections(self):
        """
        :return:
            - barycentric_coords: Barycentric coordinates of the vertices in the patch
            - faces: Faces of the patch
        """
        config_hash = hash_tuple((self.mesh_name, self.max_elevation, self.max_azimuth,
                                  self.max_range, self.height, self.width, self.max_views))
        if config_hash != self.config_hash:
            self._cache_views()

        view_indices = np.random.choice(self.max_views, self.num_views, replace=self.num_views > self.max_views)
        barycentric_coords = self.barycentric_coords[view_indices]
        faces = self.faces[view_indices]

        return barycentric_coords, faces


if __name__ == "__main__":
    from utils.misc import auto_device
    from utils.render import Renderer3D

    device = auto_device()
    with torch.no_grad():
        icosphere = Mesh.load_icosphere(2 ** 4, device=device)
        projection = SphereProjection(icosphere, height=256, width=256, max_elevation=90.0, max_azimuth=180.0,
                                      max_range=0.3, num_views=5, max_views=16, save_path="data/projections/",
                                      device=device)

        barycentric_coords, faces = projection.generate_random_view_projections()

        feature_dim = 3
        vertex_features = torch.rand((2, icosphere.vertices.shape[0], feature_dim), device=device)

        vertex_features = vertex_features * 0.1 + icosphere.vertices.unsqueeze(0) * 0.9

        interpolated_features = icosphere.interpolate(vertex_features, barycentric_coords, faces)
        print(interpolated_features.shape)

        from utils.camera import PerspectiveCamera

        # Render the mesh from 6 random viewpoints
        camera = PerspectiveCamera.generate_random_view_cameras(1, distance=2.1, k=1, max_elevation=0.0, height=512,
                                                                width=512,
                                                                max_azimuth=0.0, device=device)
        # camera = None
        # Use vs_shader = "ray" for sphere projection rendering
        # Use vs_shader = "raster" for standard rasterization rendering
        renderer = Renderer3D(background_color=1.0, vs_shader="raster")
        mesh = icosphere

        siren = Siren(feature_dim, 3, 32, 2, 3, outermost_linear=False).to(device)

        # siren = None
        rendered_image = renderer.render(mesh, vertex_features, camera, None, siren)
        # rendered_image: [batch_size, num_views, height, width, num_features]

        image = Renderer3D.to_pil(torch.tensor(rendered_image)).show()

        renderer = Renderer3D(background_color=1.0, vs_shader="ray")


        rendered_image = renderer.render(mesh, vertex_features, None, projection, siren)
        # rendered_image: [batch_size, num_views, height, width, num_features]

        image = Renderer3D.to_pil(torch.tensor(rendered_image)).show()