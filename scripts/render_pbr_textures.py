import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
import torch
from utils.render import Renderer3D, FragmentShader



if __name__ == '__main__':
    renderer = Renderer3D(background_color=1.0, ambient_light=0.5, directional_light=0.5, point_light=0.5,
                          vs_shader="raster")

    import os
    from utils.misc import load_texture_image, process_output_channels, auto_device
    from tqdm import tqdm


    device = auto_device()

    with torch.no_grad():
        num_channels = {  # Number of channels in the model to assign to each target image
            "albedo": 3,
            "normal": 3,
            "hra": 3,  # Height, Roughness, Ambient Occlusion
            # "height": 1,
            # "roughness": 1,
            # "ambient_occlusion": 1,
            
        }
        total_channels, target_channels = process_output_channels(num_channels)
        print(target_channels)

        H, W = 1024, 1024
        shader_config = {
            "background_color": 1.0,
            "ambient_light": 0.5,
            "directional_light": 0.0,
            "point_light": 0.7,
        }

        for target in tqdm(os.listdir("data/pbr_textures/")):
            # if target != "Sci-fi_Wall_010":
            # if target != "Stylized_blocks_001":
            #
            #     continue
            target_images_path = {
                "albedo": f"data/pbr_textures/{target}/albedo.jpg",
                "normal": f"data/pbr_textures/{target}/normal.jpg",
                "hra": f"data/pbr_textures/{target}/hra.jpg",
                # "height": f"data/pbr_textures/{target}/height.jpg",
                # "roughness": f"data/pbr_textures/{target}/roughness.jpg",
                # "ambient_occlusion": f"data/pbr_textures/{target}/ambient_occlusion.jpg",
            }

            # Load target images
            albedo = load_texture_image(target_images_path["albedo"], (H, W))[0]
            normal = load_texture_image(target_images_path["normal"], (H, W))[0]
            hra = load_texture_image(target_images_path["hra"], (H, W))[0]
            # height = load_texture_image(target_images_path["height"], (H, W))[0].mean(dim=1, keepdim=True)
            # roughness = load_texture_image(target_images_path["roughness"], (H, W))[0].mean(dim=1, keepdim=True)
            # ambient_occlusion = load_texture_image(target_images_path["ambient_occlusion"], (H, W))[0].mean(dim=1,
            #                                                                                                 keepdim=True)
            
            texture_maps = torch.zeros((1, 9, H, W), device="cpu")
            texture_maps[:, target_channels["albedo"]] = albedo
            texture_maps[:, target_channels["hra"]] = hra
            texture_maps[:, target_channels["normal"]] = normal
            texture_maps = texture_maps.to(device)

            # texture_maps = torch.cat([albedo, height, normal, roughness, ambient_occlusion], dim=1).to(device)
            texture_maps = texture_maps.permute(0, 2, 3, 1)  # [N, H, W, C]

            color = FragmentShader.pbr_2d_fs_shader(shader_config, texture_maps, target_channels)
            image = Renderer3D.to_pil(color[:, None])

            # image.show()
            # exit()
            image.save(f"data/pbr_textures/{target}/rendered.jpg")
#