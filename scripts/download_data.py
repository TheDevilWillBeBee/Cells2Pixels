import os
import zipfile

import gdown


def download_zip_and_extract(url, zip_path, output_dir, dataset_name):
    if os.path.exists(output_dir):
        print(f"{dataset_name} is already downloaded.")
        print(f"Remove the existing folder at {output_dir} to download again.\n")
        return

    gdown.download(url, zip_path, quiet=False)
    print(f"Unzipping the {dataset_name}")
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(output_dir)

    os.remove(zip_path)


def process_pbr_textures(pbr_texture_dir):
    from PIL import Image

    if not os.path.exists(pbr_texture_dir):
        return

    print("Combining height, roughness, and ambient occlusion maps into hra.jpg")
    for texture_name in os.listdir(pbr_texture_dir):
        if texture_name == ".DS_Store":
            continue

        texture_dir = os.path.join(pbr_texture_dir, texture_name)
        if not os.path.isdir(texture_dir):
            continue

        hra_path = os.path.join(texture_dir, "hra.jpg")
        if os.path.exists(hra_path):
            continue

        height_map = Image.open(os.path.join(texture_dir, "height.jpg"))
        roughness_map = Image.open(os.path.join(texture_dir, "roughness.jpg"))
        ambient_occlusion_map = Image.open(
            os.path.join(texture_dir, "ambient_occlusion.jpg")
        )

        height_map = height_map.convert("L")
        roughness_map = roughness_map.convert("L")
        ambient_occlusion_map = ambient_occlusion_map.convert("L")
        hra_image = Image.merge(
            "RGB", (height_map, roughness_map, ambient_occlusion_map)
        )
        hra_image.save(hra_path)


def main():
    # Download the datasets
    if not os.path.exists("data"):
        os.makedirs("data")

    if not os.path.exists("data/projections"):
        os.makedirs("data/projections")

    downloads = [
        {
            "dataset_name": "Radiance Field dataset",
            "url": "https://drive.google.com/uc?id=16FJErc6aLdI5qeQAgNunAQDWywQ9Kjrs",
            "zip_path": "data/radiance_fields.zip",
            "output_dir": "data/radiance_fields",
        },
        {
            "dataset_name": "PBR Texture dataset",
            "url": "https://drive.google.com/uc?id=1TsA3fyr1OU_1C4Rk5rtBp-Zzr_54Bofn",
            "zip_path": "data/pbr_textures.zip",
            "output_dir": "data/pbr_textures",
        },
        {
            "dataset_name": "Mesh dataset",
            "url": "https://drive.google.com/uc?id=136uliL3tcQinNXg3LXcJB4_Qf-HXg2A3",
            "zip_path": "data/meshes.zip",
            "output_dir": "data/meshes",
        },
        {
            "dataset_name": "High Resolution Texture dataset",
            "url": "https://drive.google.com/uc?id=11BdUtM4V2JhN6u-JjB2ux8Z4kRgWUe3c",
            "zip_path": "data/textures_hr.zip",
            "output_dir": "data/textures_hr",
        },
        {
            "dataset_name": "Morphology Images dataset (Transparent PNG Images)",
            "url": "https://drive.google.com/uc?id=1WdrfZTLTJ5S0G2LzasSVyLiLLIXymE9V",
            "zip_path": "data/morphology_png.zip",
            "output_dir": "data/morphology_png",
        },
        {
            "dataset_name": "3D Texture dataset (Textures with a black or white background)",
            "url": "https://drive.google.com/uc?id=1rqf1f6vWRFFCthTSr-VdYj-p5pzu6gi8",
            "zip_path": "data/textures_3d.zip",
            "output_dir": "data/textures_3d",
        },
        {
            "dataset_name": "Solid Texture dataset (3D Volume Textures)",
            "url": "https://drive.google.com/uc?id=1Xa5RXDs_sEW7HihdMwG3duG-vViraY0f",
            "zip_path": "data/solid_textures.zip",
            "output_dir": "data/solid_textures",
        }

    ]

    for download in downloads:
        download_zip_and_extract(**download)

    process_pbr_textures("data/pbr_textures")


if __name__ == "__main__":
    main()
