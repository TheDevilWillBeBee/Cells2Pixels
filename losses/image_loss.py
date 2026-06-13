from PIL import Image, ImageOps
import numpy as np
import torch
import requests
import io
from lpips import LPIPS
import torchvision.transforms.functional as TF


def imread(url, max_size=None, mode=None, center_crop=False):
    if isinstance(max_size, int):
        max_size = (max_size, max_size)
    if url.startswith(('http:', 'https:')):
        # wikimedia requires a user agent
        headers = {
            "User-Agent": "Requests in Colab/0.0 (https://colab.research.google.com/; no-reply@google.com) requests/0.0"
        }
        r = requests.get(url, headers=headers)
        f = io.BytesIO(r.content)
    else:
        f = url
    img = Image.open(f)

    if max_size is not None:
        img.thumbnail(max_size, Image.LANCZOS)

    if center_crop:
        width, height = img.size  # Get dimensions
        new_width = min(width, height)
        new_height = min(width, height)
        left = (width - new_width) / 2
        top = (height - new_height) / 2
        right = (width + new_width) / 2
        bottom = (height + new_height) / 2
        # Crop the center of the image
        img = img.crop((left, top, right, bottom))
    else:
        # pad the image to be a square
        width, height = img.size
        if width > height:
            padding = (0, (width - height) // 2, 0, (width - height + 1) // 2)
            img = ImageOps.expand(img, padding, fill=(255, 255, 255, 0))
        elif height > width:
            padding = ((height - width) // 2, 0, (height - width + 1) // 2, 0)
            img = ImageOps.expand(img, padding, fill=(255, 255, 255, 0))

    if mode is not None:
        img = img.convert(mode)
    img = np.float32(img) / 255.0

    return img


class ImageLoss(torch.nn.Module):
    """
    :param target_path: Path to the target image. It should be a png image with 4 channels (RGBA). The alpha channel determines the mask.
    :param image_size: The input image will be resized to this size.
    :param padding: The size of the padding to be applied to the borders of the image. This helps with training the GrowingNCA model.
    :param l2_weight: Weight for the L2 loss component.
    :param l1_weight: Weight for the L1 loss component.
    :param premultiply_alpha: If True, the RGB channels will be premultiplied by the alpha channel.
    :param device: Torch device to use for the loss computation (e.g., 'cuda:0' or 'cpu').
    """

    def __init__(self, target_path, image_size=(256, 256), padding=(64, 64), l2_weight=1.0, l1_weight=1.0,
                 lpips_weight=0.0,
                 premultiply_alpha=True,
                 device='cuda:0'):
        super(ImageLoss, self).__init__()
        self.target_path = target_path
        self.image_size = image_size
        self.padding = padding
        self.l2_weight = l2_weight
        self.l1_weight = l1_weight
        self.lpips_weight = lpips_weight
        self.device = device
        self.grid_size = (image_size[0] + 2 * padding[0], image_size[1] + 2 * padding[1])

        target_image_np = imread(target_path, max_size=self.image_size, mode='RGBA', center_crop=False)
        self.target_image_np = np.pad(target_image_np, ((padding[0], padding[0]), (padding[1], padding[1]), (0, 0)),
                                      "constant")
        self.target_image_pil = Image.fromarray(np.uint8(self.target_image_np * 255))

        self.target_image = torch.tensor(self.target_image_np).permute(2, 0, 1).unsqueeze(0).to(self.device)
        if premultiply_alpha:
            self.target_image[:, :3] *= self.target_image[:, 3:4]

        if self.lpips_weight > 0:
            self.lpips_loss_fn = LPIPS(net='vgg', verbose=False).to(self.device)

    @staticmethod
    def from_emoji(target_emoji='🦎', image_size=512, device='cuda:0'):
        """
        Create an ImageLoss instance from an emoji.
        :param target_emoji: The emoji to be used as the target image.
        :param image_size: The size of the image.
        """
        assert image_size in [512, 128]
        if image_size == 512:
            padding = (128, 128)
            image_size = (512, 512)
        else:
            padding = (32, 32)
            image_size = (128, 128)

        emoji_code = hex(ord(target_emoji))[2:].lower()
        url = f"https://github.com/googlefonts/noto-emoji/blob/main/png/512/emoji_u{emoji_code}.png?raw=true"
        return ImageLoss(target_path=url, image_size=image_size, padding=padding, l1_weight=1.0,
                         l2_weight=1.0, premultiply_alpha=True, device=device)

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
        b, _, h, w = input_dict['generated_images'].shape

        generated_images = input_dict['generated_images'].to(self.device)  # [B, 4, H, W]
        alpha = input_dict['alpha'].to(self.device) * 2.0  # [B, 1, H, W]

        l2_loss = ((generated_images - self.target_image) ** 2).mean()
        l1_loss = torch.abs(generated_images - self.target_image).mean()
        psnr = ImageLoss.calculate_psnr(generated_images, self.target_image)

        shape_l2_loss = ((self.target_image[:, 3:4] - alpha) ** 2).mean()
        shape_l1_loss = torch.abs(self.target_image[:, 3:4] - alpha).mean()

        loss_log = {
            'L2': l2_loss,
            'L1': l1_loss,
            "PSNR": psnr,
            "Shape L2": shape_l2_loss,
            "Shape L1": shape_l1_loss,
        }

        loss = l2_loss * self.l2_weight + l1_loss * self.l1_weight
        loss += shape_l2_loss + shape_l1_loss  # Shape loss is not weighted

        if self.lpips_weight > 0:
            # chn = np.random.choice([0, 1, 2, 3], size=3, replace=False)
            # Sample random channels for each element in the batch
            x = generated_images[..., self.padding[0]:-self.padding[0], self.padding[1]:-self.padding[1]]
            y = self.target_image[..., self.padding[0]:-self.padding[0], self.padding[1]:-self.padding[1]]
            y = y.expand(b, -1, -1, -1)  # Repeat target image for batch size

            chn = torch.stack([torch.randperm(4, device=self.device)[:3] for _ in range(b)], dim=0)
            x = x[torch.arange(b)[:, None], chn]
            y = y[torch.arange(b)[:, None], chn]

            lpips_loss = self.lpips_loss_fn(x, y, normalize=True).mean()
            loss += lpips_loss * self.lpips_weight
            loss_log['LPIPS'] = lpips_loss

        summary = None
        if return_summary:
            summary = {
                "images": visualize_differences(self.target_image, generated_images)
            }

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

    b, _, h, w = generated.shape
    target = torch.clip(target, 0.0, 1.0)
    generated = torch.clip(generated, 0.0, 1.0)
    target_repeated = target.expand(b, -1, -1, -1)

    # Split into RGB and alpha
    target_rgb, target_alpha = target_repeated[:, :3], target_repeated[:, 3:]
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

    col1 = to_image_grid(target_repeated)
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
    target_emoji = '🦎😀💥👁🐠🦋🐞🕸🥨🎄'[1]
    image_loss_fn2 = ImageLoss.from_emoji(target_emoji, image_size=512, device="cpu")
    image_loss_fn1 = ImageLoss(target_path='../data/png_images/chameleon.png', image_size=(512, 512),
                               padding=(128, 128), lpips_weight=1.0,
                               device="cpu")
    # image_loss_fn2 = ImageLoss(target_path='../data/png_images/portrait.png', image_size=(512, 512), padding=(0, 0),
    #                            device="mps")

    # visualize_differences(image_loss_fn1.target_image, image_loss_fn2.target_image.repeat(2, 1, 1, 1)).show()

    # print(image_loss_fn1.target_image.max(), image_loss_fn1.target_image.mean(), image_loss_fn1.target_image.min())

    input_dict = {
        'generated_images': image_loss_fn1.target_image.repeat(2, 1, 1, 1),
        'alpha': torch.ones((2, 1, 768, 768), device="cpu") * 0.5
    }

    loss, loss_log, summary = image_loss_fn1(input_dict)
