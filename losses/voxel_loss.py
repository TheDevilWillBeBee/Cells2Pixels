import torch
import numpy as np
import trimesh
import struct
from PIL import Image


class VoxelLoss(torch.nn.Module):
    def __init__(
        self,
        mesh_path,
        solid_texture_vol_path,
        res=128,
        padding=32,
        l2_weight=1.0,
        l1_weight=1.0,
        device="cuda:0",
    ):
        super(VoxelLoss, self).__init__()
        self.mesh_path = mesh_path
        self.solid_texture_vol_path = solid_texture_vol_path
        self.res = res
        self.padding = padding
        self.l2_weight = l2_weight
        self.l1_weight = l1_weight
        self.device = device
        vol = self.create_colored_occupancy(
            mesh_path, solid_texture_vol_path, res=res
        )  # (res, res, res, 4)
        self.grid_size = (res + 2 * padding, res + 2 * padding, res + 2 * padding)

        vol = np.pad(
            vol,
            ((padding, padding), (padding, padding), (padding, padding), (0, 0)),
            mode="constant",
            constant_values=0,
        )
        with torch.no_grad():
            self.vol = torch.tensor(
                vol, device=device, dtype=torch.float32
            )  # (res + 2 * padding, res + 2 * padding, res + 2 * padding, 4)

    @staticmethod
    def read_vol(filepath):
        with open(filepath, "rb") as f:
            header_bytes = f.read(4096)
            header_struct = struct.Struct("4si256s?iii")
            magic, version, texName, wrap, volSize, numChannels, bytesPerChannel = (
                header_struct.unpack_from(header_bytes)
            )

            if magic != b"VOLU":
                raise ValueError(f"Invalid magic header: {magic}")
            if version != 4:
                raise ValueError(f"Unsupported version: {version}")
            if bytesPerChannel != 1:
                raise ValueError(
                    f"Unsupported bytesPerChannel: {bytesPerChannel} (only 1 supported)"
                )

            vol_bytes = volSize * volSize * volSize * numChannels
            data = np.frombuffer(f.read(vol_bytes), dtype=np.uint8)
            if data.size != vol_bytes:
                raise ValueError("Unexpected end of file while reading volume data")

            volume = data.reshape((volSize, volSize, volSize, numChannels))
            print("Successfully loaded volume from ", filepath, "Shape:", volume.shape)
        return volume

    @staticmethod
    def create_colored_occupancy(mesh_path, vol_path, res=256):
        # Load and normalize mesh to fit within unit cube
        mesh = trimesh.load(mesh_path, force="mesh")
        mesh.apply_translation(-mesh.centroid)
        scale = 1.0 / max(mesh.extents)
        mesh.apply_scale(scale)

        # Voxelize mesh at target resolution
        voxel_grid = mesh.voxelized(pitch=1.0 / (res - 1)).fill()
        occupancy = voxel_grid.matrix.astype(np.float32)
        h, w, d = occupancy.shape

        # Pad symmetrically so occupancy is centered inside res³ grid
        pad_h = (res - h) // 2
        pad_w = (res - w) // 2
        pad_d = (res - d) // 2
        pad_h_extra = res - (h + pad_h)
        pad_w_extra = res - (w + pad_w)
        pad_d_extra = res - (d + pad_d)

        occupancy = np.pad(
            occupancy,
            ((pad_h, pad_h_extra), (pad_w, pad_w_extra), (pad_d, pad_d_extra)),
            mode="constant",
            constant_values=0,
        )

        # Read solid texture volume
        volume = VoxelLoss.read_vol(vol_path)
        vol_size = volume.shape[0]

        # Repeat the volume along axes to reach the target res³
        repeats = [res // vol_size + (1 if res % vol_size != 0 else 0)] * 3
        tiled_volume = np.tile(volume, (*repeats, 1))[
            :res, :res, :res, : volume.shape[3]
        ]

        # Normalize color values to [0,1]
        color = tiled_volume.astype(np.float32) / 255.0

        # Combine RGB with occupancy as alpha
        colored_occupancy = np.concatenate([color, occupancy[..., None]], axis=-1)
        colored_occupancy[..., :3] *= colored_occupancy[..., -1:]

        return colored_occupancy

    @staticmethod
    @torch.no_grad()
    def calculate_psnr(vol1, vol2, max_val=1.0):
        """
        Computes the PSNR (Peak Signal-to-Noise Ratio) between two volumes.
        """
        vol1 = torch.clip(vol1, 0.0, 1.0)
        vol2 = torch.clip(vol2, 0.0, 1.0)
        mse = torch.mean((vol1 - vol2) ** 2)
        psnr = 20 * np.log10(max_val) - 10 * torch.log10(mse + 1e-8)
        return psnr

    def forward(self, input_dict, return_summary=True):
        """
        Calculate voxel difference loss based on the input dictionary.
        :param input_dict: A dictionary containing 'generated_voxel' key and 'alpha' key
                           The generated_voxel should be in the shape of [batch_size, height, width, depth, 4].
                           The generated voxel should normally be in range [0, 1].
                           alpha is the upsampled alpha channel from the NCA state, shape [batch_size, height, width, depth, 1].

        :param return_summary: If True, returns a summary of the loss.
        :return: Loss value and optionally a summary.
        """
        gen_voxel = input_dict["generated_voxel"]  # (B, H, W, D, 4)
        alpha = input_dict["alpha"] * 2.0  # (B, H, W, D, 1)
        B, H, W, D, C = gen_voxel.shape
        assert C == 4, "Generated voxel must have 4 channels (RGBA)."
        assert (
            H == self.res + 2 * self.padding
            and W == self.res + 2 * self.padding
            and D == self.res + 2 * self.padding
        ), f"Generated voxel must have resolution {self.res}³."

        loss_log = {}
        summary = None

        vol = self.vol.unsqueeze(0)  # (1, res, res, res, 4)
        l2_loss = ((gen_voxel - vol) ** 2).mean()
        l1_loss = (torch.abs(gen_voxel - vol)).mean()

        occupancy_l2_loss = ((alpha - vol[..., -1:]) ** 2).mean()

        occupancy_l1_loss = (torch.abs(alpha - vol[..., -1:])).mean()

        psnr = self.calculate_psnr(gen_voxel, vol)

        loss_log = {
            "L2": l2_loss,
            "L1": l1_loss,
            "Occupancy L2": occupancy_l2_loss,
            "Occupancy L1": occupancy_l1_loss,
            "PSNR": psnr,
        }

        if return_summary:
            summary = {
                "total_loss": l2_loss + l1_loss + occupancy_l2_loss + occupancy_l1_loss,
                **{k: v.item() for k, v in loss_log.items()},
            }

        loss = 0.0
        loss += self.l2_weight * l2_loss + self.l1_weight * l1_loss
        loss += occupancy_l2_loss + occupancy_l1_loss  # Always include occupancy loss

        if return_summary:
            # Create an image showing 3 axis aligned slices of the generated voxel and target voxel.
            # Slices are taken from the middle of the voxel grid.
            # Output a PIL image with maximum 4 rows considering the first 4 elements in the batch.
            # Visualize as PNG image with RGBA channels.
            with torch.no_grad():
                slice_indices = [H // 2, W // 2, D // 2]
                slice_titles = ["XY Slice", "XZ Slice", "YZ Slice"]
                num_images = min(B, 4)
                slice_images = []
                for i in range(num_images):
                    slices = []
                    for axis, idx in enumerate(slice_indices):
                        if axis == 0:  # XY slice
                            gen_slice = gen_voxel[i, idx, :, :, :].cpu().numpy()
                            target_slice = vol[0, idx, :, :, :].cpu().numpy()
                        elif axis == 1:  # XZ slice
                            gen_slice = gen_voxel[i, :, idx, :, :].cpu().numpy()
                            target_slice = vol[0, :, idx, :, :].cpu().numpy()
                        else:  # YZ slice
                            gen_slice = gen_voxel[i, :, :, idx, :].cpu().numpy()
                            target_slice = vol[0, :, :, idx, :].cpu().numpy()

                        # Convert to uint8
                        gen_slice = np.clip(gen_slice, 0.0, 1.0)
                        target_slice = np.clip(target_slice, 0.0, 1.0)
                        gen_slice_img = (gen_slice * 255).astype(np.uint8)
                        target_slice_img = (target_slice * 255).astype(np.uint8)

                        # Concatenate generated and target slices side by side
                        combined = np.concatenate(
                            [gen_slice_img, target_slice_img], axis=1
                        )
                        slices.append(combined)

                    # Concatenate all slices horizontally
                    full_image = np.concatenate(slices, axis=0)
                    slice_images.append(full_image)
                final_image = np.concatenate(slice_images, axis=1)
                summary["slice_images"] = Image.fromarray(final_image)

        return loss, loss_log, summary
