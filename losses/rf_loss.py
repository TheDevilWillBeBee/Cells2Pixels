import os

from PIL import Image, ImageOps
import numpy as np
import torch
from lpips import LPIPS

from losses.image_loss import imread
from utils.camera import PerspectiveCamera
from utils.volumetric_render import RaySampler


class RadianceFieldLoss(torch.nn.Module):
    """
    For now we only support square images.

    :param scene_path: Path to the scene directory containing camera parameters and images.
    :param image_size: The input image will be resized to this size.
    :param l2_weight: Weight for the L2 loss component.
    :param l1_weight: Weight for the L1 loss component.
    :param device: Torch device to use for the loss computation (e.g., 'cuda:0' or 'cpu').
    """

    def __init__(self, scene_path, image_size=(256, 256), l2_weight=1.0, l1_weight=1.0,
                 lpips_weight=0.0,
                 device='cuda:0'):
        # Consider adding the functionality to sample only parts of the rays for each image
        # (if there are speed and performacne issues and we can't increase the number of samples per ray).
        super(RadianceFieldLoss, self).__init__()
        self.scene_path = scene_path
        self.image_size = image_size
        self.l2_weight = l2_weight
        self.l1_weight = l1_weight
        self.lpips_weight = lpips_weight
        self.device = device

        self.train_cameras, self.train_images_np, self.train_images, self.train_image_names = self.load_targets(
            tag="train")
        self.test_cameras, self.test_images_np, self.test_images, self.test_image_names = self.load_targets(tag="test")

        print("Successfully loaded target images.")

        # self.target_image_pil = Image.fromarray(np.uint8(self.target_image_np * 255))

        if self.lpips_weight > 0:
            self.lpips_loss_fn = LPIPS(net='vgg').to(self.device)

    def load_targets(self, tag="train"):
        cameras, image_names = PerspectiveCamera.from_json(
            os.path.join(self.scene_path, f'{tag}_camera_params.json'),
            self.device, return_keys=True)

        cameras.height = self.image_size[0]
        cameras.width = self.image_size[1]

        valid_idx = []
        image_list = []
        for i, image_name in enumerate(image_names):
            image_path = os.path.join(self.scene_path, f'{tag}', f'{image_name}')
            if os.path.exists(image_path):
                valid_idx.append(i)
                image_list.append(imread(image_path, max_size=self.image_size, mode='RGBA', center_crop=False))

        image_names = list(np.array(image_names)[valid_idx])
        cameras = cameras.sample_batch(len(valid_idx), valid_idx)

        images_np = np.array(image_list)
        images_np[..., 3:4] = (images_np[..., 3:4] > 0.5) * 1.0
        images_np[..., :3] *= images_np[..., 3:4]
        images_torch = torch.tensor(images_np, dtype=torch.float32,
                                    device=self.device).permute(0, 3, 1, 2)

        return cameras, images_np, images_torch, image_names

    def get_random_views(self, batch_size=1, mode='train'):
        assert mode in ['train', 'test'], "Mode must be 'train' or 'test'."
        cameras = self.train_cameras if mode == 'train' else self.test_cameras
        images = self.train_images if mode == 'train' else self.test_images

        view_idx = np.random.choice(len(images), batch_size, replace=True)
        cameras = cameras.sample_batch(batch_size, view_idx)

        return cameras, view_idx

    @staticmethod
    @torch.no_grad()
    def calculate_psnr(img1, img2, max_val=1.0):
        """
        Computes the PSNR (Peak Signal-to-Noise Ratio) between two images.

        Args:
            img1 (Tensor): First image tensor of shape [B, C, H, W].
            img2 (Tensor): Second image tensor of shape [B, C, H, W].
            max_val (float): Maximum possible pixel value of the images (1.0 for normalized, 255 for raw).

        Returns:
            Tensor: PSNR value (scalar tensor if batch size is 1, else shape [B]).
        """
        img1 = torch.clip(img1, 0.0, 1.0)
        img2 = torch.clip(img2, 0.0, 1.0)
        mse = torch.mean((img1 - img2) ** 2)
        psnr = 20 * np.log10(max_val) - 10 * torch.log10(mse + 1e-8)
        return psnr

    def forward(self, input_dict, return_summary=True):
        """
        Calculate the image loss based on the input dictionary.
        :param input_dict: A dictionary containing 'generated_images' key with a batch of images.
                           The generated images should be in the shape of [batch_size, 4, height, width].
                           The generated images should normally be in range [0, 1].
        :param return_summary: If True, returns a summary of the loss.
        :return: Loss value and optionally a summary.
        """
        B, num_views, _, h, w = input_dict['generated_images_l1l2'].shape
        view_idx = input_dict['view_idx']
        mode = input_dict['mode']

        assert mode in ["train", "test"]
        if mode == 'train':
            target_images = self.train_images[view_idx]  # [num_views, 4, H, W]
        else:
            target_images = self.test_images[view_idx]  # [num_views, 4, H, W]

        l1l2_sampler = input_dict.get('sampler_l1l2', RaySampler(target_images.shape[2:]))
        lpips_sampler = input_dict.get('sampler_lpips', l1l2_sampler)

        target_images_l1l2 = l1l2_sampler.sample_pixels(target_images)  # [num_views, 4, h, w]
        target_images_l1l2 = target_images_l1l2.unsqueeze(0).expand(B, -1, -1, -1, -1)  # [B, num_views, 4, h, w]
        target_images_l1l2 = target_images_l1l2.reshape(B * num_views, 4, h, w)  # Flatten batch and views

        generated_images_l1l2 = input_dict['generated_images_l1l2']  # [B, num_views 4, h, w]
        generated_images_l1l2 = generated_images_l1l2.reshape(B * num_views, 4, h, w)  # Flatten batch and views

        alpha_l1l2 = input_dict['alpha_l1l2'] * 2.0  # [B, num_views, 1, h, w]

        alpha_l1l2 = alpha_l1l2.reshape(B * num_views, 1, h, w)  # Flatten batch and views

        loss = 0.0
        loss_log = {}
        if self.lpips_weight > 0:
            target_images_lpips = lpips_sampler.sample_pixels(target_images)  # [num_views, 4, h, w]
            target_images_lpips = target_images_lpips.unsqueeze(0).expand(B, -1, -1, -1, -1)  # [B, num_views, 4, h, w]
            target_images_lpips = target_images_lpips.reshape(B * num_views, 4, h, w)  # Flatten batch and views

            generated_images_lpips = input_dict.get('generated_images_lpips',
                                                    generated_images_l1l2)  # [B, num_views, 4, h, w]
            generated_images_lpips = generated_images_lpips.reshape(B * num_views, 4, h, w)

            alpha_lpips = input_dict.get('alpha_lpips', alpha_l1l2)  # [B, num_views, 1, h, w]
            alpha_lpips = alpha_lpips.reshape(B * num_views, 1, h, w)  # Flatten batch and views

            # chn = np.random.choice([0, 1, 2, 3], size=3, replace=False)
            # Sample random channels for each element in the batch
            # chn = torch.stack([torch.randperm(4, device=self.device)[:3] for _ in range(B * num_views)], dim=0)
            # x = generated_images_lpips[torch.arange(B * num_views)[:, None], chn]
            # y = target_images_lpips[torch.arange(B * num_views)[:, None], chn]

            x = generated_images_lpips[:, :3, :, :]  # Use only RGB channels for LPIPS
            y = target_images_lpips[:, :3, :, :]  # Use only RGB channels for LPIPS

            lpips_loss = self.lpips_loss_fn(x, y, normalize=True).mean()
            loss += lpips_loss * self.lpips_weight
            loss_log['LPIPS'] = lpips_loss

            generated_images_l1l2 = torch.cat([generated_images_l1l2, generated_images_lpips], dim=0)
            target_images_l1l2 = torch.cat([target_images_l1l2, target_images_lpips], dim=0)
            alpha_l1l2 = torch.cat([alpha_l1l2, alpha_lpips], dim=0)

        l2_loss = ((generated_images_l1l2 - target_images_l1l2) ** 2).mean()
        l1_loss = torch.abs(generated_images_l1l2 - target_images_l1l2).mean()
        psnr = RadianceFieldLoss.calculate_psnr(generated_images_l1l2, target_images_l1l2)

        shape_l2_loss = ((target_images_l1l2[:, 3:4] - alpha_l1l2) ** 2).mean()
        shape_l1_loss = torch.abs(target_images_l1l2[:, 3:4] - alpha_l1l2).mean()

        loss_log['L2'] = l2_loss
        loss_log['L1'] = l1_loss
        loss_log['PSNR'] = psnr
        loss_log['Shape L2'] = shape_l2_loss
        loss_log['Shape L1'] = shape_l1_loss

        loss += l2_loss * self.l2_weight + l1_loss * self.l1_weight
        loss += shape_l2_loss + shape_l1_loss  # Shape loss is not weighted

        summary = None
        if return_summary:
            summary = {
                "images_l1l2": visualize_differences(target_images_l1l2, generated_images_l1l2),
            }
            if self.lpips_weight > 0:
                summary["images_lpips"] = visualize_differences(target_images_lpips, generated_images_lpips)

        return loss, loss_log, summary


@torch.no_grad()
def visualize_differences(target: torch.Tensor, generated: torch.Tensor):
    """
    Visualizes the differences between a target image and a batch of generated images.

    Args:
        target: Tensor of shape [1, 4, H, W] (RGBA image)
        generated: Tensor of shape [B, 4, H, W] (batch of RGBA images)
    """
    assert target.shape[1] == 4 and generated.shape[1] == 4, "Images must be RGBA"
    assert target.shape[2:] == generated.shape[2:], "Target and generated images must be same size"

    generated = generated[:4]  # Limit to first 4 images for visualization
    target = target[:4]

    b, _, h, w = generated.shape
    target = torch.clip(target, 0.0, 1.0)
    generated = torch.clip(generated, 0.0, 1.0)

    # Split into RGB and alpha
    target_rgb, target_alpha = target[:, :3], target[:, 3:]
    generated_rgb, generated_alpha = generated[:, :3], generated[:, 3:]
    union_alpha = torch.clip(target_alpha + generated_alpha, 0, 1.0)

    # Compute absolute differences
    diff_rgb = torch.abs(target_rgb - generated_rgb)
    diff_alpha = torch.abs(target_alpha - generated_alpha)

    diff_rgb = torch.clip(diff_rgb, 0.0, 1.0)
    diff_alpha = torch.clip(diff_alpha, 0.0, 1.0)

    # Concatenate for visualization: [B, 4, H, W] each, so final shape will be [B, 4*W, H]

    # Convert diff_alpha to RGBA for visualization (gray to RGBA)
    diff_alpha_rgb = diff_alpha.expand(-1, 3, -1, -1)
    diff_alpha_rgba = torch.cat([diff_alpha_rgb, union_alpha], dim=1)

    diff_rgb = torch.cat([diff_rgb, union_alpha], dim=1)  # Add alpha channel as 1s

    def to_image_grid(images):
        x = images.permute(0, 2, 3, 1)  # [B, H, W, C]
        return np.vstack(x.cpu().numpy())

    col1 = to_image_grid(target)
    col2 = to_image_grid(generated)
    col3 = to_image_grid(diff_rgb)
    col4 = to_image_grid(torch.cat([1.0 - target_alpha.expand(-1, 3, -1, -1), torch.ones_like(target_alpha)],
                                   dim=1))  # Add alpha channel as 1s
    col5 = to_image_grid(torch.cat([1.0 - generated_alpha.expand(-1, 3, -1, -1), torch.ones_like(generated_alpha)],
                                   dim=1))
    col6 = to_image_grid(diff_alpha_rgba)
    final_image = np.hstack((col1, col2, col3, col4, col5, col6))

    final_image = (final_image * 255.0).astype(np.uint8)  # Convert to uint8 for display
    return Image.fromarray(final_image).convert("RGBA")


if __name__ == '__main__':
    rf_loss_fn = RadianceFieldLoss(scene_path='../data/radiance_fields/lego', image_size=(512, 512),
                                   lpips_weight=0.0,
                                   device="cpu")

    camera, view_idx = rf_loss_fn.get_random_views(batch_size=3, mode='train')
    sampler = RaySampler((512, 512), mode='stride', stride=2)

    # print(rf_loss_fn.train_images_np.shape)
    # x = rf_loss_fn.train_images_np[0][..., :3]
    # x = Image.fromarray((x * 255).astype(np.uint8))
    # x.show()
    # print(rf_loss_fn.train_images[0])
    # exit()

    batch_size = 2
    x = torch.rand(batch_size, len(view_idx), 4, 256, 256)  # Simulated generated images [B, num_views, 4, H, W]
    alpha = torch.ones_like(x[:, :, 3:4])  # Assuming alpha channel is the 4th channel
    input_dict = {
        'generated_images_l1l2': x,
        'alpha_l1l2': alpha,
        'view_idx': view_idx,
        'mode': 'train',
        'sampler_l1l2': sampler
    }

    loss, loss_log, summary = rf_loss_fn(input_dict, return_summary=True)
    summary['images_l1l2'].show()
    # summary['images_lpips'].show()
