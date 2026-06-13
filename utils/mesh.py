import numpy as np
import torch
from pathlib import Path

from utils.icosphere import icosphere


def _parse_obj_vertex_index(face_token, num_vertices, line_number):
    """Parse the vertex index from an OBJ face token."""
    vertex_index = face_token.split("/")[0]
    if vertex_index == "":
        raise ValueError(f"Missing vertex index in OBJ face on line {line_number}.")

    vertex_index = int(vertex_index)
    if vertex_index > 0:
        return vertex_index - 1
    if vertex_index < 0:
        return num_vertices + vertex_index
    raise ValueError(f"OBJ vertex indices are 1-based; got 0 on line {line_number}.")


def import_mesh(obj_path):
    """Import vertices and triangular faces from a Wavefront OBJ file."""
    vertices = []
    faces = []

    with open(obj_path, "r", encoding="utf-8") as obj_file:
        for line_number, line in enumerate(obj_file, start=1):
            line = line.split("#", 1)[0].strip()
            if not line:
                continue

            parts = line.split()
            if parts[0] == "v":
                if len(parts) < 4:
                    raise ValueError(
                        f"OBJ vertex on line {line_number} has fewer than 3 coordinates."
                    )
                vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])
            elif parts[0] == "f":
                face = [
                    _parse_obj_vertex_index(token, len(vertices), line_number)
                    for token in parts[1:]
                ]
                if len(face) != 3:
                    raise ValueError(
                        f"Only triangular OBJ faces are supported; line {line_number} has {len(face)} vertices."
                    )
                faces.append(face)

    if len(vertices) == 0:
        raise ValueError(f"No vertices found in OBJ file: {obj_path}")
    if len(faces) == 0:
        raise ValueError(f"No faces found in OBJ file: {obj_path}")

    vertices = torch.tensor(vertices, dtype=torch.float32)
    faces = torch.tensor(faces, dtype=torch.int64)

    if torch.any(faces < 0) or torch.any(faces >= vertices.shape[0]):
        raise ValueError(
            f"OBJ face references a vertex outside the available range: {obj_path}"
        )

    return vertices, faces


def subdivide_trianglemesh(vertices, faces, subdivision_iter):
    """Subdivide triangular faces by adding edge midpoints."""
    if subdivision_iter <= 0:
        return vertices, faces

    device = vertices.device
    vertices_dtype = vertices.dtype
    faces_dtype = faces.dtype
    vertices_list = vertices.detach().cpu().tolist()
    faces_list = faces.detach().cpu().tolist()

    for _ in range(subdivision_iter):
        edge_midpoints = {}
        new_faces = []

        def midpoint_index(v0, v1):
            edge = tuple(sorted((v0, v1)))
            if edge not in edge_midpoints:
                midpoint = [
                    (vertices_list[v0][axis] + vertices_list[v1][axis]) / 2.0
                    for axis in range(3)
                ]
                edge_midpoints[edge] = len(vertices_list)
                vertices_list.append(midpoint)
            return edge_midpoints[edge]

        for v0, v1, v2 in faces_list:
            m01 = midpoint_index(v0, v1)
            m12 = midpoint_index(v1, v2)
            m20 = midpoint_index(v2, v0)
            new_faces.extend(
                [
                    [v0, m01, m20],
                    [v1, m12, m01],
                    [v2, m20, m12],
                    [m01, m12, m20],
                ]
            )

        faces_list = new_faces

    vertices = torch.tensor(vertices_list, dtype=vertices_dtype, device=device)
    faces = torch.tensor(faces_list, dtype=faces_dtype, device=device)
    return vertices, faces


class Mesh:
    def __init__(self, vertices, faces, mesh_name="Mesh", device="cuda:0"):
        """
        :param vertices: numpy array or torch tensor with shape [num_vertices, 3]
        :param faces: numpy array or torch tensor with shape [num_faces, 3]
        :param mesh_name: name of the mesh
        :param device: pytorch device to store the mesh data

        The mesh class has the following attributes:
        - vertices: torch tensor with shape [num_vertices, 3]
        - faces: torch tensor with shape [num_faces, 3]
        - edges: torch tensor with shape [num_edges, 2]
        - edge_index: torch tensor with shape [2, 2 * num_edges] used for message passing
        - face_normals: torch tensor with shape [num_faces, 3]
        - vertex_normals: torch tensor with shape [num_vertices, 3]
        - laplacian_matrix: torch sparse tensor with shape [num_vertices, num_vertices]
        - Nv, Nf, Ne: number of vertices, faces, and edges in the mesh
        """
        self.mesh_name = mesh_name
        self.is_icosphere = "icosphere" in mesh_name.lower()

        if isinstance(vertices, np.ndarray):
            self.vertices = torch.tensor(vertices, dtype=torch.float32, device=device)
        else:
            self.vertices = vertices.to(device)

        if isinstance(faces, np.ndarray):
            self.faces = torch.tensor(faces, dtype=torch.int64, device=device)
        else:
            self.faces = faces.to(device)

        with torch.no_grad():
            # Extract edges from faces of the mesh
            edges = torch.column_stack(
                [self.faces, torch.roll(self.faces, shifts=-1, dims=1)]
            )
            edges = edges.reshape(-1, 2)
            edges, _ = torch.sort(edges, dim=1)
            edges = torch.unique(edges, dim=0)
            edges_idx = torch.argsort(edges, dim=0)[:, 0]
            self.edges = edges[edges_idx]

            bi_edges = torch.cat([edges, edges[:, [1, 0]]], dim=0)
            # bi_edges = np.sort(bi_edges, axis=1)
            sorted_ids = torch.argsort(bi_edges, dim=0, descending=False)[:, 0]
            self.edge_index = bi_edges[sorted_ids].T

        # Normalize the mesh so that it fits into a unit sphere
        self.normalize_into_unit_sphere()

        # Compute the normals for faces and vertices of the mesh
        self._compute_normals()

        # Compute the Laplacian matrix of the mesh
        self._compute_laplacian()

        self.Nv, self.Nf, self.Ne = (
            self.vertices.shape[0],
            self.faces.shape[0],
            self.edges.shape[0],
        )

    @staticmethod
    def load_from_obj(obj_path, subdivision_iter=0, **kwargs):
        """Load a mesh from an obj file."""
        vertices, faces = import_mesh(obj_path)
        mesh_name = Path(obj_path).stem

        if subdivision_iter > 0:
            vertices, faces = subdivide_trianglemesh(vertices, faces, subdivision_iter)

        return Mesh(vertices, faces, mesh_name=mesh_name, **kwargs)

    @staticmethod
    def load_icosphere(subdivision_freq=2**6, **kwargs):
        """
        Load an icosphere mesh with a given number of subdivisions.

        :param subdivision_freq: Subdivision frequency

        :return: Mesh object of the icosphere
        the mesh will have 12 + 10 * (subdivision_freq**2 -1) vertices and 20 * (subdivision_freq**2) faces
        Setting subdivision_freq=2**n will create an icosphere sphere with n recursive subdivisions.
        """
        vertices, faces = icosphere(nu=subdivision_freq)
        return Mesh(vertices, faces, mesh_name=f"Icosphere{subdivision_freq}", **kwargs)

    @torch.no_grad()
    def normalize_into_unit_sphere(self):
        """Normalize the vertices of the mesh into a unit sphere."""
        center = torch.mean(self.vertices, dim=0)
        scale = torch.max(torch.norm(self.vertices - center, p=2, dim=1))
        self.vertices = (self.vertices - center) / scale

    @torch.no_grad()
    def _compute_normals(self):
        """Compute the normals for faces and vertices of the mesh."""
        vertex_normals = torch.zeros_like(self.vertices)
        vertices_faces = self.vertices[self.faces]

        faces_normals = torch.cross(
            vertices_faces[:, 2] - vertices_faces[:, 1],
            vertices_faces[:, 0] - vertices_faces[:, 1],
            dim=1,
        )
        self.face_normals = torch.nn.functional.normalize(
            faces_normals, eps=1e-6, dim=1
        )

        # NOTE: this is already applying the area weighting as the magnitude
        # of the cross product is 2 x area of the triangle.
        vertex_normals = vertex_normals.index_add(0, self.faces[:, 0], faces_normals)
        vertex_normals = vertex_normals.index_add(0, self.faces[:, 1], faces_normals)
        vertex_normals = vertex_normals.index_add(0, self.faces[:, 2], faces_normals)

        self.vertex_normals = torch.nn.functional.normalize(
            vertex_normals, eps=1e-6, dim=1
        )

    @torch.no_grad()
    def _compute_laplacian(self):
        """Compute the Laplacian matrix (D - A) of the mesh using the adjacency matrix in a sparse form."""
        device = self.vertices.device
        num_vertices = self.vertices.shape[0]
        edge_list = self.edges

        # Create a tensor with ones to fill the adjacency matrix at the edge positions
        ones = torch.ones(edge_list.shape[0], dtype=torch.float32).to(device)

        # Create a sparse adjacency matrix using the edge list and ones
        adj_matrix_sparse = torch.sparse_coo_tensor(
            edge_list.t(),
            ones,
            size=(num_vertices, num_vertices),
            dtype=torch.float32,
            device=device,
        )

        # Make the adjacency matrix symmetric for undirected graphs
        adj_matrix_sparse = adj_matrix_sparse + adj_matrix_sparse.t()

        # Compute the degree (valence) for each vertex
        degree_vals = torch.sparse.sum(adj_matrix_sparse, dim=1).values()
        self.min_valence = torch.min(degree_vals).item()
        self.max_valence = torch.max(degree_vals).item()
        self.average_valence = torch.mean(degree_vals).item()

        degree_matrix_sparse = torch.sparse_coo_tensor(
            torch.stack([torch.arange(num_vertices), torch.arange(num_vertices)]).to(
                device
            ),
            degree_vals,
            size=(num_vertices, num_vertices),
            dtype=torch.float32,
            device=device,
        )
        self.laplacian_matrix = degree_matrix_sparse - adj_matrix_sparse

    def vertex2face_features(self, vertex_features: torch.Tensor) -> torch.Tensor:
        """
        :param vertex_features: A torch tensor with shape [batch_size, num_vertices, num_features]
        :return: A torch tensor with shape [batch_size, num_faces, 3, num_features]
        """
        return vertex_features[:, self.faces]

    def interpolate(
        self,
        vertex_features: torch.Tensor,
        barycentric_coords: torch.Tensor,
        faces: torch.Tensor,
    ) -> torch.Tensor:
        """
        :param vertex_features: A torch tensor with shape [batch_size, num_vertices, num_features]
        :param barycentric_coords: A torch tensor with shape [..., 3] representing the barycentric coordinates for each triangle
        :param faces: A torch tensor with shape [..., 3] representing the vertex indices for each triangle


        :return: A torch tensor with shape [..., num_features]
        Interpolates the vertex features using the barycentric coordinates. This process is similar to rasterization.
        """
        V1 = faces[..., 0]  # first vertex of the face [...]
        V2 = faces[..., 1]  # second vertex of the face [...]
        V3 = faces[..., 2]  # third vertex of the face [...]
        w = barycentric_coords[..., 0]  # weight for the first vertex [...]
        u = barycentric_coords[..., 1]  # weight for the second vertex [...]
        v = barycentric_coords[..., 2]  # weight for the third vertex [...]

        feature1 = vertex_features[
            :, V1
        ]  # features of the first vertex [batch_size, ..., num_features]
        feature2 = vertex_features[
            :, V2
        ]  # features of the second vertex [batch_size, ..., num_features]
        feature3 = vertex_features[
            :, V3
        ]  # features of the third vertex [batch_size, ..., num_features]
        interpolated_feature = (
            feature1 * w[None, ..., None]
            + feature2 * u[None, ..., None]
            + feature3 * v[None, ..., None]
        )

        return interpolated_feature

    def __repr__(self):
        return (
            f"Mesh("
            f"\n\tName = {self.mesh_name}"
            f"\n\tVertices = {self.vertices.shape[0]},"
            f"\n\tFaces = {self.faces.shape[0]},"
            f"\n\tEdges = {self.edges.shape[0]},"
            f"\n\tValence (min, max, average) = {(self.min_valence, self.max_valence, self.average_valence)},"
            f"\n\tDevice = {self.vertices.device},"
            f"\n)"
        )


if __name__ == "__main__":
    from utils.misc import auto_device

    device = auto_device()
    with torch.no_grad():
        mesh1 = Mesh.load_from_obj(
            "data/meshes/cat/cat.obj", subdivision_iter=1, device=device
        )
        mesh2 = Mesh.load_icosphere(2**6, device=device)
        mesh3 = Mesh.load_from_obj(
            "data/meshes/cat/cat.obj", subdivision_iter=0, device=device
        )

        print(mesh1)

        print(mesh2)

        print(mesh3)

        V = mesh2.vertices
        F = mesh2.faces
        T = V[F]

        from utils.camera import PerspectiveCamera
        from utils.render import Renderer3D
        from models.siren import Siren

        # Render the mesh from 6 random viewpoints
        camera = PerspectiveCamera.generate_random_view_cameras(
            4,
            distance=2.1,
            k=1,
            max_azimuth=180.0,
            max_elevation=90.0,
            height=512,
            width=512,
            device=device,
        )

        # Use vs_shader = "ray" for sphere projection rendering
        # Use vs_shader = "raster" for standard rasterization rendering
        renderer = Renderer3D(background_color=1.0, vs_shader="raster")
        mesh = mesh1
        vertex_features = torch.rand((1, mesh.vertices.shape[0], 3), device=device)
        vertex_features[:, :, :3] = mesh.vertex_normals  # use vertex normals as the first 3 features
        feature_dim = vertex_features.shape[-1]

        siren = Siren(feature_dim, 3, 32, 2, 3, outermost_linear=False).to(device)

        siren = None
        rendered_image = renderer.render(mesh, vertex_features, camera, None, siren)
        # rendered_image: [batch_size, num_views, height, width, num_features]

        image = Renderer3D.to_pil(torch.tensor(rendered_image)).show()
