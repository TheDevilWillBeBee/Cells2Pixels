import numpy as np
from PIL import Image
import torch

from models.siren import Siren
from utils.mesh import Mesh
from utils.camera import PerspectiveCamera
from utils.sphere_projection import SphereProjection

def transform_barycoords(barycoords):
    def normalize_triangle_distribution(x, start, end, mode):
        x = (x - start) / (end - start)
        mode = (mode - start) / (end - start)
        y = torch.zeros_like(x)
        if mode > 0.0:
            y[x <= mode] = x[x <= mode] ** 2.0 / mode
        y[x > mode] = 1.0 - (1.0 - x[x > mode])**2.0 / (1.0 - mode)

        return y
    a = torch.max(barycoords, dim=-1, keepdim=True)[0]
    c = torch.min(barycoords, dim=-1, keepdim=True)[0]
    b = 1.0 - a - c

    # Testing this part
    a = normalize_triangle_distribution(a, 1/3, 1.0, 0.5)
    b = normalize_triangle_distribution(b, 0.0, 0.5, 1/3)
    c = normalize_triangle_distribution(c, 0.0, 1/3, 0.0)

    a = (a - 0.5) * 2.0
    b = (b - 0.5) * 2.0
    c = (c - 0.5) * 2.0

    return torch.cat([a, b, c], dim=-1)

# From Kaolin
def spherical_harmonic_lighting(imnormal, lights):
    r"""Creates lighting effects.

    Follows convention set by *Wojciech Jarosz* in 
    `Efficient Monte Carlo Methods for Light Transport in Scattering Media`_.

    .. deprecated:: 0.13.0
       This function is deprecated. Use :func:`kaolin.render.lighting.sh9_irradiance`.

    Args:
        imnormal (torch.FloatTensor):
            per pixel normal, of shape :math:`(\text{batch_size}, \text{height}, \text{width}, 3)`
        lights (torch.FloatTensor):
            spherical harmonic lighting parameters, of shape :math:`(\text{batch_size}, 9)`

    Returns:
        (torch.FloatTensor):
            lighting effect, shape of :math:`(\text{batch_size}, \text{height}, \text{width})`

    .. _Efficient Monte Carlo Methods for Light Transport in Scattering Media:

    https://cs.dartmouth.edu/~wjarosz/publications/dissertation/appendixB.pdf

    """
    # SH lighting
    # light effect
    x = imnormal[:, :, :, 0]
    y = imnormal[:, :, :, 1]
    z = imnormal[:, :, :, 2]

    # spherical harmonic parameters
    band0 = 0.28209479177 * torch.ones_like(x)
    band1_m1 = 0.4886025119 * x
    band1_0 = 0.4886025119 * z
    band1_p1 = 0.4886025119 * y
    band2_m2 = 1.09254843059 * (x * y)
    band2_m1 = 1.09254843059 * (y * z)
    band2_0 = 0.94617469575 * (z * z) - 0.31539156525
    band2_p1 = 0.77254840404 * (x * z)
    band2_p2 = 0.38627420202 * (x * x - y * y)

    bands = torch.stack([band0,
                         band1_m1, band1_0, band1_p1,
                         band2_m2, band2_m1, band2_0, band2_p1, band2_p2],
                        dim=3)
    lighting_effect = torch.sum(bands * lights.view(-1, 1, 1, 9),
                                dim=3)

    return lighting_effect

class VertexShader:
    """
    :return:
        image_features: [batch_size*num_views, height, width, num_features]
        barycentric_coords: [batch_size*num_views, height, width, 3]
        background: [batch_size*num_views, height, width, 1]
        image_vertices: [batch_size*num_views, height, width, 3] # the 3D positions of the pixels in the world coordinate
        image_normals: [batch_size*num_views, height, width, 3] # the 3D normal vectors of the pixels in the world coordinate
    """

    @staticmethod
    def raster_vs_shader(mesh: Mesh, camera: PerspectiveCamera, vertex_features: torch.Tensor,
                         vertex_displacement: torch.Tensor = None,
                         coordinate_system='world', knum=30, sigmainv=7000.0):
        from kaolin.render.mesh import prepare_vertices, dibr_rasterization

        """
        Render the mesh using the specified camera.
        :param mesh: Mesh object
        :param camera: PerspectiveCamera object
            should have the following attributes: camera.projection_matrix, camera.transform_matrix
            camera.proj_matrix: torch tensor with shape [3, 1]
            camera.transform_matrix: torch tensor with shape [num_views, 4, 3]
        :param vertex_features: torch tensor with shape [batch_size, num_vertices, num_features]
        :param vertex_displacement: torch tensor with shape [batch_size, num_vertices, 3]
        :param coordinate_system: 'world' or 'camera'. The

        :param knum: Number of nearest faces to consider for each pixel
        :param sigmainv: Smoothing factor for the blending faces to make the rendering differentiable

        :return:
            image_features: [batch_size*num_views, height, width, num_features] # the rendered features for each pixel
            barycentric_coords: [batch_size*num_views, height, width, 3] # the barycentric coordinates for each pixel
            background: [batch_size*num_views, height, width, 1] # the mask showing if a pixel is on the mesh or not
            image_vertices: [batch_size*num_views, height, width, 3] # the 3D positions of the pixels in the specified coordinate system
            image_normals: [batch_size*num_views, height, width, 3] # the 3D normal vectors of the pixels in the specified coordinate system


        """
        device = vertex_features.device
        batch_size = vertex_features.shape[0]
        num_views = camera.transform_matrix.shape[0]
        num_faces = mesh.faces.shape[0]

        camera_proj = camera.projection_matrix
        camera_transform = camera.transform_matrix.repeat(batch_size, 1, 1)  # [batch_size, num_views, 4, 3]

        vertices = mesh.vertices[None, ...].repeat(batch_size * num_views, 1, 1)  # [batch_size, num_vertices, 3]
        if vertex_displacement is not None:
            # Displace the vertices using the vertex displacement
            vertices = vertices + vertex_displacement.repeat_interleave(repeats=num_views, dim=0)

        # face_vertices_camera [batch_size*num_views, num_faces, 3, 3] the world coordinates of the vertices for each face
        # face_vertices_image_camera [batch_size*num_views, num_faces, 3, 2] the projected coordinates of the vertices for each face
        # face_normals_camera [batch_size*num_views, num_faces, 3] the normals of the faces
        face_vertices_camera, face_vertices_image_camera, face_normals_camera = prepare_vertices(vertices,
                                                                                                 mesh.faces,
                                                                                                 camera_proj=camera_proj,
                                                                                                 camera_transform=camera_transform)

        face_normals_world = mesh.face_normals[None, ...].repeat(batch_size * num_views, 1,
                                                                 1)  # [num_views, num_faces, 3]
        face_vertices_world = mesh.vertex2face_features(mesh.vertices[None, ...]).repeat(batch_size * num_views, 1, 1,
                                                                                         1)  # [batch_size*num_views, num_faces, 3, 3]
        
        face_normals_world = mesh.vertex2face_features(mesh.vertex_normals[None, ...]).repeat(batch_size * num_views, 1, 1,
                                                                                 1)  # [batch_size*num_views, num_faces, 3, 3]
        
        barycoords = torch.zeros(batch_size * num_views, num_faces, 3, 3, device=device)
        barycoords[:, :, 0, 0] = 1.0
        barycoords[:, :, 1, 1] = 1.0
        barycoords[:, :, 2, 2] = 1.0
        if coordinate_system == 'world':
            # extra_features: [batch_size*num_views, num_faces, 3, 9]
            extra_features = torch.cat([
                barycoords,
                face_vertices_world,
                face_normals_world,
#                 face_normals_world[:, :, None, :].repeat(1, 1, 3, 1),  # [batch_size*num_views, num_faces, 3, 3]
            ], dim=-1)
        elif coordinate_system == 'camera':
            # extra_features: [batch_size*num_views, num_faces, 3, 9]
            extra_features = torch.cat([
                barycoords,
                face_vertices_camera,
                face_normals_world,
#                 face_normals_camera[:, :, None, :].repeat(1, 1, 3, 1),  # [batch_size*num_views, num_faces, 3, 3]
            ], dim=-1)
        else:
            raise ValueError("coordinate_system should be either 'world' or 'camera'")

        # extra_features = extra_features.repeat(batch_size, 1, 1, 1)  # [batch_size*num_views, num_faces, 3, 9]

        face_features = mesh.vertex2face_features(vertex_features)  # [batch_size, num_faces, 3, num_features]
        background = torch.ones(face_features.shape[0], face_features.shape[1], 3, 1, device=device)
        face_features = torch.cat([face_features, background], dim=-1)  # [batch_size, num_faces, 3, num_features+1]
        face_features = face_features.repeat_interleave(repeats=num_views, dim=0)  # [batch_size*num_views, ...]

        # [batch_size*num_views, num_faces, 3, num_features + 9 + 1]
        face_features = torch.cat([extra_features, face_features], dim=-1)

        # image_features [batch_size*num_views, height, width, num_features + 1 + 3 + 3] the rendered features for each pixel
        # soft_mask [batch_size*num_views, height, width] the mask showing if a pixel is on the mesh or not
        # face_ids [batch_size*num_views, height, width] the face id for each pixel
        # Using the actual face normals face_normals_z=face_normals_camera[..., -1] will cause artifacts in the rendered image.
        # Instead, we use a constant value of 1.0 for the face normals.
        (image_features,
         soft_mask,
         face_ids) = dibr_rasterization(height=camera.height, width=camera.width,
                                        face_vertices_z=face_vertices_camera[..., -1],
                                        face_vertices_image=face_vertices_image_camera,
                                        face_features=face_features,
                                        # face_normals_z=face_normals_camera[..., -1],
                                        face_normals_z=torch.ones(batch_size * num_views, num_faces, device=device),
                                        knum=knum, sigmainv=sigmainv,
                                        rast_backend='cuda')

        barycentric_coords = image_features[..., :3]  # [batch_size*num_views, height, width, 3]
        image_vertices = image_features[..., 3:6]  # [batch_size*num_views, height, width, 3]
        image_normals = image_features[..., 6:9]  # [batch_size*num_views, height, width, 3]

        background = (image_features[..., -1:] < 0.95).float()  # [batch_size*num_views, height, width, 1]
        image_features = image_features[..., 9:-1]  # [batch_size*num_views, height, width, num_features]

        return image_features, barycentric_coords, background, image_vertices, image_normals

    @staticmethod
    def ray_vs_shader(mesh: Mesh, projection: SphereProjection, vertex_features: torch.Tensor):
        """
        Render the mesh using the specified sphere projection.
        This method only works when the mesh geometry is fixed.
        :param mesh: Mesh object
        :param projection: SphereProjection object
            should have the following attributes: projection.barycentric_coords, projection.faces
            projection.barycentric_coords: torch tensor with shape [num_views, num_faces, 3]
            projection.faces: torch tensor with shape [num_views, num_faces, 3]
        :param vertex_features: torch tensor with shape [batch_size, num_vertices, num_features]

        :return:
            image_features: [batch_size*num_views, height, width, num_features]
            barycentric_coords: [batch_size*num_views, height, width, 3]
            background: [batch_size*num_views, height, width, 1]
            image_vertices: [batch_size*num_views, height, width, 3] # the 3D positions of the pixels in the world coordinate
            image_normals: [batch_size*num_views, height, width, 3] # the 3D normal vectors of the pixels in the world coordinate
        """

        def merge_view_batch(x):
            b, v, h, w, c = x.shape
            return x.view(b * v, h, w, c)

        assert mesh.mesh_name == projection.mesh_name, "The mesh and projection should be the same"
        batch_size = vertex_features.shape[0]

        barycentric_coords, faces = projection.generate_random_view_projections()
        # barycentric_coords: [num_views, height, width, 3]
        # faces: [num_views, height, width, 3]

        image_features = mesh.interpolate(vertex_features, barycentric_coords,
                                          faces)  # [batch_size, num_views, height, width, num_features]
        image_features = merge_view_batch(image_features)  # [batch_size*num_views, height, width, num_features]

        vertex_normals = mesh.vertex_normals[None, ...]  # [1, num_vertices, 3]
        image_normals = mesh.interpolate(vertex_normals, barycentric_coords,
                                         faces)  # [1, num_views, height, width, 3]
        image_normals = merge_view_batch(image_normals)  # [1*num_views, height, width, 3]
        image_normals = image_normals.repeat(batch_size, 1, 1, 1)  # [batch_size*num_views, height, width, 3]

        vertex_positions = mesh.vertices[None, ...]  # [1, num_vertices, 3]
        image_vertices = mesh.interpolate(vertex_positions, barycentric_coords,
                                          faces)  # [1, num_views, height, width, 3]
        image_vertices = merge_view_batch(image_vertices)  # [1*num_views, height, width, 3]
        image_vertices = image_vertices.repeat(batch_size, 1, 1, 1)  # [batch_size*num_views, height, width, 3]

        barycentric_coords = barycentric_coords.repeat(batch_size, 1, 1, 1)  # [batch_size*num_views, height, width, 3]
        background = torch.zeros_like(image_features[..., -1:])  # [batch_size*num_views, height, width, 1]

        return image_features, barycentric_coords, background, image_vertices, image_normals


class FragmentShader:
    """
    :param shader_config: dict A dictionary representing the configuration of the fragment shader.
    :param texture_maps: torch tensor with shape [N, height, width, num_features]
                         This tensor represents the interpolated input to the fragment shader such as albedo,
                         roughness, and other texture maps.
    :param background: torch tensor with shape [N, height, width, 1]
    :param normals: torch tensor with shape [N, height, width, 3]
                    This tensor contains the normal vector in the world coordinate system for each pixel.
                    It is interpolated from the vertex normals of the mesh.
    :param positions: torch tensor with shape [N, height, width, 3]
                      This tensor contains the position of the pixel in the world coordinate system.
                      It is interpolated from the vertex positions of the mesh.
    :param target_channels: dict A dictionary representing the correspondence between channel indices and textures.
    :param camera_positions: torch tensor with shape [N, 3]
                    This is not needed in the simple_fs_shader. But for PBR we need the location of the camera
    :param point_light_positions: torch tensor with shape [N, 3] showing the position of the point light

    :param parallax_mapping: bool, whether to use parallax mapping or not. Only used in the PBR shader. Only relevant for 2D rendering.

    :return: rendered images in RGB [N, height, width, num_features]
    """

    @staticmethod
    def vanilla_fs_shader(shader_config: dict, texture_maps: torch.Tensor, background: torch.Tensor,
                          normals: torch.Tensor, positions: torch.Tensor, camera_positions: torch.Tensor,
                          point_light_positions: torch.Tensor, target_channels: dict):
        """
        This method doesn't combine different texture maps into an RGB image. It simply renders the texture maps independently.
        """
        background_color = shader_config['background_color'] if 'background_color' in shader_config else 0.0
        lighting = 1.0
        texture_maps = texture_maps * lighting

        texture_maps = texture_maps * (1.0 - background) + background * background_color

        return texture_maps  # [batch_size*num_views, height, width, num_features]

    @staticmethod
    def simple_fs_shader(shader_config: dict, texture_maps: torch.Tensor, background: torch.Tensor,
                         normals: torch.Tensor, positions: torch.Tensor, camera_positions: torch.Tensor,
                         point_light_positions: torch.Tensor, target_channels: dict):
        ambient_light_strength = shader_config['ambient_light'] if 'ambient_light' in shader_config else 0.5
        directional_light_strength = shader_config['directional_light'] if 'directional_light' in shader_config else 0.5
        background_color = shader_config['background_color'] if 'background_color' in shader_config else 0.0
        lighting = ambient_light_strength
        if directional_light_strength > 0.0:
            with torch.no_grad():
                sh_lights = torch.tensor([0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0], device=texture_maps.device).unsqueeze(0)
                directional_light = spherical_harmonic_lighting(normals, sh_lights)  # [batch_size, height, width]
                directional_light = torch.clamp(directional_light, 0.0, 1.0)  # [batch_size, height, width, 1]
                lighting += directional_light * directional_light_strength
                lighting = lighting.unsqueeze(-1)

        texture_maps = texture_maps * lighting

        texture_maps = texture_maps * (1.0 - background) + background * background_color

        return texture_maps  # [batch_size*num_views, height, width, num_features]

    @staticmethod
    def pbr_fs_shader(shader_config: dict, texture_maps: torch.Tensor, background: torch.Tensor,
                      normals: torch.Tensor, positions: torch.Tensor, camera_positions: torch.Tensor,
                      point_light_positions: torch.Tensor, target_channels: dict, parallax_mapping=False):
        # We assume a single point light that is located at the same direction as the camera at a distance of 2.0
        # N, H, W, C = texture_maps.shape

        def dot_product(x, y, zero_clamp=True):
            """
            :param x: [N, H, W, C]
            :param y: [N, H, W, C]
            :param zero_clamp: bool, whether to clamp the result to 0.0

            :return: [N, H, W, 1]
            """
            res = torch.sum(x * y, dim=-1, keepdim=True)
            if zero_clamp:
                res = torch.clamp(res, 0.0)

            return res

        def fresnel_schlick(cos_theta, F0):
            """
            :param cos_theta: Cosine of the angle between the view direction and the half vector [N, H, W, 1]
            :param F0: Fresnel reflectance at normal incidence [N, H, W, 3]
            """
            cos_theta = torch.clip(cos_theta, 0.0)
            return F0 + (1.0 - F0) * torch.clip(1.0 - cos_theta, 0.0, 1.0) ** 5

        def distribution_ggx(N, H, roughness):
            """
            :param N: Surface Normal vector [N, H, W, 3]
            :param H: Half vector [N, H, W, 3]
            :param roughness: Roughness value [N, H, W, 1]
            """
            a = roughness ** 2
            a2 = a * a
            NdotH = dot_product(N, H, zero_clamp=True)
            denom = (NdotH ** 2 * (a2 - 1.0) + 1.0)
            denom = denom * denom * torch.pi
            return a2 / (denom + 1e-8)

        def geometry_schlick_ggx(cosine, roughness):
            """
            :param cosine: Cosine of the angle between two directions
            :param roughness: Roughness value [N, H, W, 1]
            """
            r = roughness + 1.0
            k = (r * r) / 8.0
            return cosine / (cosine * (1.0 - k) + k)

        def geometry_smith(N, V, L, roughness):
            """
            :param N: Surface Normal vector [N, H, W, 3]
            :param V: View direction [N, H, W, 3]
            :param L: Light direction [N, H, W, 3]
            :param roughness: Roughness value [N, H, W, 1]
            """
            NdotV = dot_product(N, V, zero_clamp=True)
            NdotL = dot_product(N, L, zero_clamp=True)
            ggx2 = geometry_schlick_ggx(NdotV, roughness)
            ggx1 = geometry_schlick_ggx(NdotL, roughness)
            return ggx1 * ggx2

        # print(positions.shape, camera_positions[:, None, None, :].shape)
        V = camera_positions[:, None, None, :] - positions  # Vectors from the surface to the camera
        V = V / (1e-8 + torch.norm(V, dim=-1, keepdim=True))  # Normalize the vectors

        L = point_light_positions[:, None, None, :] - positions  # Vectors from the surface to the light
        L = L / (1e-8 + torch.norm(L, dim=-1, keepdim=True))  # Normalize the vectors

        H = (L + V)  # Half vector between the light and the view direction
        H = H / (1e-8 + torch.norm(H, dim=-1, keepdim=True))  # Normalize the vectors

        target_channels = target_channels.copy()  # Make a copy to avoid modifying the original dictionary
        if "hra" in target_channels:
            target_channels["height"] = [target_channels["hra"][0]]
            target_channels["roughness"] = [target_channels["hra"][1]]
            target_channels["ambient_occlusion"] = [target_channels["hra"][2]]

        if "height" in target_channels:
            height_channels = target_channels["height"]
            height_map = texture_maps[..., height_channels]
        else:
            height_map = torch.ones_like(texture_maps[..., :1])

        if parallax_mapping:
            # Only use in the 2D case where positions are in range [-1, 1]
            depth_scale = 0.15
            displacement_map = (1.0 - height_map) * depth_scale  # [N, H, W, 1]
            found_mask = torch.zeros_like(displacement_map, dtype=torch.bool)
            best_texture_coords = positions[..., [1, 0]].clone()  # [N, H, W, 2]
            # best_texture_coords = torch.zeros_like(positions[..., :2])  # [N, H, W, 2]
            P = V[..., :-1] / (1e-8 + V[..., -1:])
            num_steps = 50
            total_sum = 0
            for depth_index in range(0, num_steps + 2):
                depth = depth_scale * depth_index / num_steps
                offset = P * depth  # [N, H, W, 2]
                texture_coords = positions[..., :2] - offset  # [N, H, W, 2]
                texture_coords = texture_coords[..., [1, 0]]

                new_displacement_map = torch.nn.functional.grid_sample(displacement_map.permute(0, 3, 1, 2),
                                                                       texture_coords,
                                                                       padding_mode="reflection", mode="bicubic",
                                                                       align_corners=True).permute(0, 2, 3, 1)

                condition = new_displacement_map <= depth
                update_mask = torch.logical_and(condition, ~found_mask)
                found_mask = torch.logical_or(found_mask, update_mask)
                total_sum += update_mask.float().sum()
                # print(depth_index, total_sum.item() / found_mask.numel(), depth)
                update_mask = update_mask.repeat(1, 1, 1, 2)  # [N, H, W, 2]

                best_texture_coords = torch.where(update_mask, texture_coords, best_texture_coords)

            background = torch.logical_or(best_texture_coords > 1.0, best_texture_coords < -1.0).any(dim=-1,
                                                                                                     keepdim=True).float()

            texture_maps = torch.nn.functional.grid_sample(texture_maps.permute(0, 3, 1, 2), best_texture_coords,
                                                           padding_mode="reflection", mode="bicubic",
                                                           align_corners=True).permute(0, 2, 3, 1)

        if "albedo" in target_channels:
            albedo_channels = target_channels["albedo"]
            albedo_map = texture_maps[..., albedo_channels]
        else:
            albedo_map = torch.ones_like(texture_maps[..., :3]) * 0.5

        if "normal" in target_channels:
            normal_channels = target_channels["normal"]
            normal_map = texture_maps[..., normal_channels]
            normal_map = normal_map - 0.5
            normal_map = normal_map / (1e-8 + torch.norm(normal_map, dim=-1, keepdim=True))
        else:
            normal_map = torch.zeros_like(texture_maps[..., :3])
            normal_map[..., 2] = 1.0

        if "roughness" in target_channels:
            roughness_channels = target_channels["roughness"]
            roughness_map = texture_maps[..., roughness_channels]
            roughness_map = torch.clip(roughness_map, 0.1, 1.0)
        else:
            roughness_map = torch.ones_like(texture_maps[..., :1]) * 0.5

        if "ambient_occlusion" in target_channels:
            AO_channels = target_channels["ambient_occlusion"]
            AO_map = texture_maps[..., AO_channels]
            AO_map = torch.clip(AO_map, 0.05, 1.0)
        else:
            AO_map = torch.ones_like(texture_maps[..., :1]) * 1.0

        t_hat = torch.zeros_like(normals)
        t_hat[..., 0] = 1e-8  # To avoid division by zero when the surface normal is in the x direction
        t_hat[..., 1] = normals[..., 2]
        t_hat[..., 2] = -normals[..., 1]
        t_hat = t_hat / (1e-8 + torch.norm(t_hat, dim=-1, keepdim=True))

        b_hat = torch.cross(normals, t_hat, dim=-1)
        b_hat = b_hat / (1e-8 + torch.norm(b_hat, dim=-1, keepdim=True))

        N = normal_map[..., 0:1] * t_hat + normal_map[..., 1:2] * b_hat + normal_map[...,
                                                                          2:3] * normals  # Surface normal after applying the normal map
        N = N / (1e-8 + torch.norm(N, dim=-1, keepdim=True))  # Normalize the surface normal

        distance = torch.norm(positions - point_light_positions[:, None, None, :], dim=-1, keepdim=True)
        attenuation = 1.0 / (distance * distance)
        point_light_strength = shader_config['point_light'] if 'point_light' in shader_config else 1.0
        radiance = point_light_strength * attenuation * 15.0

        metalic = 0.0
        F0 = torch.ones_like(albedo_map) * 0.04
        F0 = F0 * (1.0 - metalic) + albedo_map * metalic  # Fresnel-Schlick approximation

        NDF = distribution_ggx(N, H, roughness_map)  # Normal Distribution Function
        G = geometry_smith(N, V, L, roughness_map)  # Geometry Function
        F = fresnel_schlick(torch.sum(V * H, dim=-1, keepdim=True), F0)  # Fresnel Function

        kS = F
        kD = 1.0 - kS
        kD = kD * (1.0 - metalic)  # Diffuse reflection

        numerator = NDF * G * F
        NdotV = dot_product(N, V, zero_clamp=True)
        NdotL = dot_product(N, L, zero_clamp=True)
        denominator = 4.0 * NdotV * NdotL + 1e-8
        specular = numerator / denominator  # Specular reflection

        # Add outgoing radiance Lo
        Lo = (kD * albedo_map / torch.pi + specular) * radiance * NdotL

        ambient_light_strength = shader_config['ambient_light'] if 'ambient_light' in shader_config else 0.5
        ambient = ambient_light_strength * albedo_map * AO_map

        color = ambient + Lo

        background_color = shader_config['background_color'] if 'background_color' in shader_config else 1.0
        color = color * (1.0 - background) + background * background_color

        return color

    @staticmethod
    def pbr_2d_fs_shader(shader_config: dict, texture_maps: torch.Tensor, target_channels: dict):
        device = texture_maps.device
        b, H, W, c = texture_maps.shape
        background = 1.0
        surface_normals = torch.zeros(1, H, W, 3, device=device)
        surface_normals[..., 2] = 1.0

        x, y = torch.meshgrid(torch.linspace(-1.0, 1.0, H, device=device),
                              torch.linspace(-1.0, 1.0, W, device=device))
        z = torch.zeros_like(x)
        positions = torch.stack([x, y, z], dim=-1).unsqueeze(0).to(device)
        positions = positions.repeat(b, 1, 1, 1)

        camera_positions = torch.tensor([[0.0, 0.0, 2.0]], device=device)

        point_light_positions = (camera_positions / (
                1e-8 + torch.norm(camera_positions, dim=-1, keepdim=True))) * 2.0

        return FragmentShader.pbr_fs_shader(shader_config, texture_maps, background, surface_normals, positions,
                                            camera_positions, point_light_positions, target_channels,
                                            parallax_mapping=True)


class Renderer:
    def __init__(self, ambient_light=1.0, directional_light=0.0, point_light=0.0, fs_shader="vanilla"):
        """
        :param ambient_light: Intensity of the ambient light
        :param directional_light: Intensity of the directional light
        :param point_light: Intensity of the point light
        :param fs_shader: Fragment Shader mode, either "vanilla" or "simple" or "pbr".
        """
        assert fs_shader in ["vanilla", "simple", "pbr"], "fs_shader should be either 'vanilla' or 'simple' or 'pbr'"
        self.fs_shader = fs_shader

        self.ambient_light = ambient_light
        self.directional_light = directional_light
        self.point_light = point_light
        self.fs_shader_config = {
            "ambient_light": ambient_light,
            "directional_light": directional_light,
            "point_light": point_light,
        }

    @staticmethod
    @torch.no_grad()
    def to_pil(rendered_features, target_channels=None,  # Default to RGB channels
               batch_stack='vertical', view_stack='horizontal', target_stack='vertical'):
        """
        :param rendered_features: A tensor of shape [batch_size, num_views, height, width, num_features]
        :param target_channels: The channels to be rendered. tuple or dictionary of tuples.
                                Example: (0, 3) or {"rgb": (0, 3)}. Default renders the first 3 channels.
        :param batch_stack: Whether to stack the batch elements vertically or horizontally.
        :param view_stack: Whether to stack the views vertically or horizontally.
        :param target_stack: Whether to stack the target channels vertically or horizontally.

        :return: A PIL Image showing the rendered images.
        """
        # @ TODO This function is not consistent with the meshnca.render_channels
        assert rendered_features.dim() == 5, "The input tensor should have 5 dimensions"

        if not isinstance(target_channels, dict):
            target_channels = {"rgb": [0, 1, 2]}
        else:
            target_channels = target_channels

        batch_size, num_views, height, width, num_features = rendered_features.shape
        rendered_features = rendered_features.cpu().numpy()
        rendered_features = np.clip(rendered_features * 255.0, 0.0, 255.0).astype(np.uint8)

        stack_batch = np.vstack if batch_stack == 'vertical' else np.hstack
        stack_view = np.vstack if view_stack == 'vertical' else np.hstack
        stack_target = np.vstack if target_stack == 'vertical' else np.hstack

        features = stack_batch(
            [stack_view(rendered_features[i]) for i in range(batch_size)]
        )  # [batch_size*height, num_views*width, num_features]

        image_list = []
        for key, channels in sorted(target_channels.items()):
            image = features[..., channels]
            if image.shape[-1] == 1:
                image = np.repeat(image, 3, axis=-1)

            image_list.append(image)

        image = Image.fromarray(stack_target(image_list))
        return image


class Renderer2D(Renderer):
    def __init__(self, scale_factor: int = 8, padding="circular", mode="bilinear", background_color=1.0, **kwargs):
        """
        :param scale_factor: int, size of the patch that will be passed into the siren decoder
        :param padding: str, the padding mode for upsampling. Default is "circular".
        :param mode: str, the interpolation mode to smooth out the cell features. Default is "bilinear".
        :param kwargs: Other arguments to be passed to the Renderer constructor.
        """
        super().__init__(**kwargs)
        self.scale_factor = scale_factor
        self.padding = padding
        self.mode = mode
        self.fs_shader_config['background_color'] = background_color

        assert self.fs_shader in ["vanilla", "pbr"], "fs_shader should be either 'vanilla' or 'pbr' for Renderer2D"

    def render(self, cell_features: torch.Tensor, siren: Siren, target_channels: dict = None, fs_shader=None,
               hard_clamp=False):
        """
        :param cell_features: A tensor of shape [batch_size, height, width, channels]
                              This represents the state of the cells in the NCA.
        :param siren: Siren MLP decoder to convert [cell_features, sub-cell coords] to output features.
        :param target_channels: dict A dictionary representing the correspondence between channel indices and textures.
        :param fs_shader: Fragment Shader mode, either "vanilla" or "simple" or "pbr". If None, use the class attribute.
        :param hard_clamp: bool, whether to clamp the rendered image to [0, 1]

        :return: A tensor of shape [batch_size, height * scale_factor, width * scale_factor, num_features]
                 num_features = siren.output_dim if fs_shader = "vanilla" else 3
        """

        def get_mgrid(L, dim=2, flatten=True, theta=0.0, device="cuda:0"):
            """
            Generate a mesh grid of coordinates in the range [-1, 1] for each dimension.
            """
            # tensors = tuple(dim * [torch.linspace(-1, 1, steps=L, device=device)])
            tensors = tuple(dim * [(torch.arange(L, device=device) / L - 0.5 + 0.5 / L) * 2.0])
            mgrid = torch.stack(torch.meshgrid(*tensors), dim=-1)
            if flatten:
                mgrid = mgrid.reshape(-1, dim)

            if theta != 0.0 and dim == 2:
                x = mgrid[..., 0]
                y = mgrid[..., 1]
                u = np.cos(theta) * x - np.sin(theta) * y
                v = np.sin(theta) * x + np.cos(theta) * y
                mgrid = torch.stack([x, y], dim=-1)

            return mgrid  # [sidelen, sidelen, dim] if flatten=False else [sidelen*sidelen, dim]

        b, h, w, c = cell_features.shape
        scale_factor = self.scale_factor
        coords = get_mgrid(scale_factor, 2,
                           flatten=False, device=cell_features.device)[None, ...].expand(b, -1, -1, -1)

        coords = coords.repeat(1, h, w, 1)  # [b, h, w, 2]
        x = torch.nn.functional.pad(cell_features.permute(0, 3, 1, 2), [1, 1, 1, 1],
                                    self.padding)  # [b, c, h + 2, w + 2]
        x_upscale = torch.nn.functional.interpolate(x, scale_factor=scale_factor, mode="bilinear")
        x_upscale = x_upscale[:, :, scale_factor:-scale_factor, scale_factor:-scale_factor].permute(0, 2, 3, 1)
        output = siren(x_upscale, coords)  # b, h, w, 3

        if fs_shader is None:
            fs_shader = self.fs_shader

        if fs_shader == "pbr":
            output = FragmentShader.pbr_2d_fs_shader(self.fs_shader_config, output, target_channels)

        if hard_clamp:
            output = torch.clamp(output, 0.0, 1.0)

        return output

    @staticmethod
    @torch.no_grad()
    def to_pil(rendered_features, target_channels=(0, 3),
               batch_stack='horizontal', target_stack='vertical'):
        """
        :param rendered_features: A tensor of shape [batch_size, height, width, num_features]
        :param target_channels: The channels to be rendered. tuple or dictionary of tuples.
                                Example: (0, 3) or {"rgb": (0, 3)}. Default renders the first 3 channels.
        :param batch_stack: Whether to stack the batch elements vertically or horizontally.
        :param target_stack: Whether to stack the target channels vertically or horizontally.

        :return: A PIL Image showing the rendered images.
        """
        x = rendered_features[:, None]  # [batch_size, 1, height, width, num_features] Add a fake view
        return Renderer.to_pil(x, target_channels, batch_stack=batch_stack, target_stack=target_stack,
                               view_stack='horizontal')


class Renderer3D(Renderer):
    def __init__(self, ambient_light=0.28, directional_light=1.0, point_light=0.0,
                 background_color=0.0, vs_shader="raster", fs_shader="vanilla", height_scale=0.03, transform_bary=True):
        """
        :param ambient_light: Intensity of the ambient light
        :param directional_light: Intensity of the directional light
        :param point_light: Intensity of the point light
        :param background_color: float, 1.0 is white and 0.0 is black
        :param vs_shader: Vertex Shader mode, either "raster" or "ray"
        :param fs_shader: Fragment Shader mode, either "vanilla" or "simple" or "pbr".
        """
        super().__init__(ambient_light, directional_light, point_light, fs_shader)

        assert vs_shader in ["raster", "ray"], "vs_shader should be either 'raster' or 'ray'"
        self.vs_shader = vs_shader
        self.height_scale = height_scale

        self.background_color = background_color
        self.fs_shader_config['background_color'] = background_color
        self.transform_bary = transform_bary

    def render(self, mesh: Mesh, vertex_features: torch.Tensor,
               camera: PerspectiveCamera, projection: SphereProjection,
               siren: Siren, target_channels: dict = None, fs_shader=None, hard_clamp=False) -> torch.Tensor:
        """
        Render the mesh using the specified camera.
        :param mesh: Mesh object
        :param camera: PerspectiveCamera object. Will be used if render_mode is "raster"
            should have the following attributes: camera.projection_matrix, camera.transform_matrix
            camera.proj_matrix: torch tensor with shape [3, 1]
            camera.transform_matrix: torch tensor with shape [num_views, 4, 3]
        :param projection: SphereProjection object. Will be used if render_mode is "ray"
        :param vertex_features: torch tensor with shape [batch_size, num_vertices, num_features]

        :param siren: Siren MLP decoder to convert [vertex_features, barycoords] to texture maps
        :param fs_shader: Fragment Shader mode, either "vanilla" or "simple" or "pbr". If None, use the class attribute.
                        "vanilla": Render the features without any 3D effect. Output dim: num_features
                        "simple": Use simple directional lighting to render the features. Output dim: num_features
                        "pbr": Use PBR lighting to render the features into an RGB image. Output dim: 3
        :param target_channels: dict A dictionary representing the correspondence between channel indices and textures.
        :param hard_clamp: bool, whether to clamp the rendered image to [0, 1]

        :return: the rendered images from all views [batch_size, num_views, height, width, num_features]
        """
        device = vertex_features.device
        batch_size = vertex_features.shape[0]
        fs_shader = fs_shader if fs_shader is not None else self.fs_shader
        if self.vs_shader == "ray":
            num_views = projection.num_views
            image_features, barycentric_coords, background, image_vertices, image_normals = VertexShader.ray_vs_shader(
                mesh,
                projection,
                vertex_features)
        else:
            num_views = camera.transform_matrix.shape[0]
            vertex_displacement = None
            if target_channels is not None and 'hra' in target_channels:  # Displace the vertices before rasterization
                height_channels = [target_channels['hra'][0]]
                b, n, _ = vertex_features.shape
                barycoords = torch.zeros(b, n, 3, device=device)
                barycoords[:, :, 0] = 1.0
                barycoords[:, :, 1] = -1.0
                barycoords[:, :, 2] = -1.0
                vertex_height = siren(vertex_features, barycoords)[..., height_channels]  # [batch_size, num_vertices, 1]
                # print(vertex_height.mean(), vertex_height.std())
#                 vertex_height = (vertex_height - vertex_height.mean()) * self.height_scale
                vertex_height = vertex_height * self.height_scale           
                # vertex_height = torch.clip(vertex_height ** 2, 0.0, 1.0) * 0.1
                vertex_displacement = vertex_height * mesh.vertex_normals[None, ...]  # [batch_size, num_vertices, 3]

            image_features, barycentric_coords, background, image_vertices, image_normals = VertexShader.raster_vs_shader(
                mesh,
                camera,
                vertex_features,
                vertex_displacement=vertex_displacement,
                coordinate_system='camera' if fs_shader == "simple" else 'world')

        if siren is None:
            texture_maps = image_features
        else:
            if self.transform_bary:
                barycoords = transform_barycoords(barycentric_coords)  # [batch_size*num_views, num_vertices, 3]
            else:
                barycoords = barycentric_coords
#             barycoords = (barycentric_coords - 0.5) * 2.0
            texture_maps = siren(image_features, barycoords)  # [batch_size*num_views, height, width, render_channels]

        if fs_shader in ['vanilla', 'simple']:
            _, h, w, c = texture_maps.shape
            if fs_shader == "simple":
                rendered_images = FragmentShader.simple_fs_shader(self.fs_shader_config, texture_maps, background,
                                                                  image_normals, None, None,
                                                                  None, None)
            else:
                rendered_images = FragmentShader.vanilla_fs_shader(self.fs_shader_config, texture_maps, background,
                                                                   image_normals, None, None,
                                                                   None, target_channels)
            rendered_images = rendered_images.view(batch_size, num_views, h, w, c)
        else:
            _, h, w, c = texture_maps.shape
            assert self.vs_shader == "raster", "PBR shader only works with rasterization for now"
            camera_positions = camera.position.repeat(batch_size, 1)  # [batch_size*num_views, 3]
            point_light_positions = (camera_positions / (
                    1e-8 + torch.norm(camera_positions, dim=-1, keepdim=True))) * 2.0
            rendered_images = FragmentShader.pbr_fs_shader(self.fs_shader_config, texture_maps, background,
                                                           image_normals, image_vertices, camera_positions,
                                                           point_light_positions, target_channels)
            rendered_images = rendered_images.view(batch_size, num_views, h, w, 3)

        if hard_clamp:
            rendered_images = torch.clamp(rendered_images, 0.0, 1.0)

        return rendered_images

    def __repr__(self):
        return f"Renderer(ambient_light={self.ambient_light}, directional_light={self.directional_light}, " \
               f"\n\tpoint_light={self.point_light}, background_color={self.background_color})"



if __name__ == '__main__':
    from utils.mesh import Mesh
    from utils.camera import PerspectiveCamera
    from utils.misc import auto_device

    device = auto_device()
    

    # mesh = Mesh.load_from_obj('data/meshes/mug/mug.obj', device=device)
    # mesh = Mesh.load_from_obj('data/meshes/airplane/airplane.obj', device=device)
    mesh = Mesh.load_icosphere(2 ** 5, device=device)
    print(mesh)

    with torch.no_grad():
        from utils.video import VideoWriter
        from tqdm import tqdm
        from PIL import Image

        # torch.manual_seed(42)
        # np.random.seed(42)
        # Render the mesh from 6 random viewpoints
        camera = PerspectiveCamera.generate_random_view_cameras(2, distance=2.0, k=1, device=device,
                                                                max_azimuth=360.0, max_elevation=180.0,
                                                                height=256, width=256)
        # projection = SphereProjection(mesh, height=1024, width=1024, max_elevation=180.0, max_azimuth=360.0,
        #                               max_range=0.55, num_views=2, max_views=3,
        #                               save_path="data/projections/",
        #                               device=device)
        projection = None
        vs_shader = "raster" # "raster", "ray"
        renderer = Renderer3D(background_color=1.0, ambient_light=0.5, directional_light=0.5, point_light=0.5,
                              vs_shader=vs_shader)

        print(camera.distance)
        print(camera.look_at, camera.fov)

        feature_dim = 16
        vertex_features = torch.zeros((2, mesh.vertices.shape[0], feature_dim), device=device)
        vertex_features[0] = torch.rand_like(vertex_features[0]) * 0.5

        siren = Siren(feature_dim, 3, 32,
                      2, 3, outermost_linear=False, num_frequencies=2, activation="relu").to(device)

        # siren = None
        target_channels = {
            "albedo": [0, 1, 2],  # Channels 0, 1, 2
        }
        rendered_image = renderer.render(mesh, vertex_features, camera, projection, siren,
                                         target_channels, fs_shader="simple", hard_clamp=False)
        # rendered_image: [batch_size, num_views, height, width, num_features]

        rendered_image = rendered_image.cpu().numpy()
        image = Renderer3D.to_pil(torch.tensor(rendered_image)).show()
        # image = Renderer.to_pil(torch.tensor(rendered_image), batch_stack='horizontal', view_stack='vertical').show()

        with VideoWriter('tmp.mp4', fps=30.0) as video:
            for i in tqdm(range(16)):
                rendered_image = renderer.render(mesh, vertex_features, camera, projection, siren, target_channels, fs_shader="simple").cpu().numpy()
                image = Renderer3D.to_pil(torch.tensor(rendered_image))
                camera.rotateY(1.0)
                video.add(image)
        
