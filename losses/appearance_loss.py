import torch
import torchvision.models as torch_models
import torchvision.transforms.functional as TF
import torch.nn.functional as F
import torchvision
import numpy as np

from PIL import Image

from utils.misc import load_texture_image


class AppearanceLoss(torch.nn.Module):
    def __init__(self,
                 target_image_size=(256, 256),
                 patch_size=(256, 256),
                 num_scales=1,
                 relative_scale=1,
                 target_images_path="data/textures/bubbly_0101.jpg",
                 total_channels=3,
                 output_channels={"RGB": [0, 1, 2]},
                 vgg_layers=(1, 6, 11, 18, 25),
                 max_image_size=(2048, 2048),
                 include_image_as_feature=False,
                 num_pseudo_targets=0,
                 ac_loss_weight=0.0,
                 ac_loss_kwargs={},
                 device='cuda:0'):
        """
        :param target_image_size: Resolution for loading the target images
        :param patch_size: Maximum Resolution of the target image
        :param num_scales: Number of scales to be used for the style loss
        :param relative_scale: Relative scale of the target image to the generated image. 1.0 = same size, 2.0 = twice the size
        :param target_images_path: str or a dictionary of strings, path to the target images
        :param total_channels: Total number of channels in the target images
        :param output_channels: A dictionary showing the output channels for each target image
        :param vgg_layers: VGG layers used for calculating the style loss
        :param max_image_size: Images larger than this size will be center cropped
        :param include_image_as_feature: Whether to inclued the input image in the returned features
        :param num_pseudo_targets: Number of pseudo targets to be created by mixing channels from the target images.
        :param ac_loss_weight: Weight for the auto-correlation loss
        :param ac_kwargs: Dictionary of arguments for the AutoCorrelationLoss
        :param device: PyTorch device
        """
        super(AppearanceLoss, self).__init__()

        self.target_image_size = target_image_size
        self.patch_size = patch_size
        self.num_scales = num_scales
        self.relative_scale = relative_scale
        self.max_image_size = max_image_size
        self.include_image_as_feature = include_image_as_feature
        self.num_pseudo_targets = num_pseudo_targets

        if not isinstance(target_images_path, dict):
            self.target_images_path = {"RGB": target_images_path}
        else:
            self.target_images_path = target_images_path

        self.total_channels = total_channels
        self.output_channels = output_channels

        self.vgg_layers = vgg_layers

        self.device = device

        vgg = torch_models.vgg16(weights=torch_models.VGG16_Weights.IMAGENET1K_V1).features.to(device)
        vgg_layers = []
        for l in vgg:
            # if isinstance(l, torch.nn.MaxPool2d):
            #     l = torch.nn.AvgPool2d(l.kernel_size, l.stride, l.padding, l.ceil_mode)
            vgg_layers.append(l)
        self.vgg = torch.nn.Sequential(*vgg_layers)

        self._load_target_images()
        if self.num_pseudo_targets > 0:
            self._create_pseudo_targets()

        self.style_loss_fn = OptimalTransportLoss(n_samples=1024, device=self.device)

        self.ac_loss_weight = ac_loss_weight
        if self.ac_loss_weight > 0.0:
            self.ac_loss_fn = AutoCorrelationLoss(**ac_loss_kwargs)
        

    def get_vgg_features(self, x, flatten=False):
        """

        :param x: input images [b, c, h, w]
        :param flatten: Whether to flatten the returned features to remove the spatial dimensions
        

        :return: A list of pytorch tensors containing the features extracted from different layers of the VGG network.
        """
        vgg_layers = self.vgg_layers
        mean = torch.tensor([0.485, 0.456, 0.406], device=x.device)[:, None, None]
        std = torch.tensor([0.229, 0.224, 0.225], device=x.device)[:, None, None]
        x = (x - mean) / std
        b, c, h, w = x.shape

        features = []
        if self.include_image_as_feature:
            features = [x.reshape(b, c, h * w)] if flatten else [x]

        for i, layer in enumerate(self.vgg[:max(vgg_layers) + 1]):
            x = layer(x)
            if i in vgg_layers:
                b, c, h, w = x.shape
                if flatten:
                    features.append(x.reshape(b, c, h * w))
                else:
                    features.append(x)
        return features

    @torch.no_grad()
    def _load_target_images(self):
        target_images = []
        for i, key in enumerate(sorted(self.target_images_path)):
            target_image_path = self.target_images_path[key]

            target_image, _ = load_texture_image(target_image_path, self.target_image_size)

            target_image = target_image.to(self.device)
            target_images.append(target_image)

        self.target_images = torch.cat(target_images, dim=0)  # [n_targets, 3, h, w]
        self.target_features = self.extract_features(self.target_images)

    def _create_pseudo_targets(self):
        np.random.seed(42)  # For reproducibility
        n_targets, _, h, w = self.target_images.shape

        self.pseudo_target_images = []
        self.pseudo_output_channels = {}
        for j in range(self.num_pseudo_targets):
            pseudo_target = []
            channels = []
            for _ in range(3):
                chn = None
                target_index = 1000
                while chn is None or chn in channels:
                    target_index = np.random.randint(n_targets)
                    target_key = sorted(self.target_images_path)[target_index]
                    oc = self.output_channels[target_key]
                    random_idx = np.random.randint(len(oc))
                    chn = oc[random_idx]
                pseudo_target.append(self.target_images[target_index, random_idx])
                channels.append(chn)
            self.pseudo_output_channels[f"pseudo_target_{j}"] = channels

            pseudo_target = torch.stack(pseudo_target, dim=0)  # [3, h, w]
            self.pseudo_target_images.append(pseudo_target)

        self.pseudo_target_images = torch.stack(self.pseudo_target_images, dim=0)  # [num_pseudo_targets, 3, h, w]
        self.pseudo_target_features = self.extract_features(self.pseudo_target_images)
        print("Pseudo Target Channels: ", self.pseudo_output_channels)

    def extract_features(self, target_images):
        target_features = []
        sx, sy = self.patch_size
        sx = int(sx * 2 ** (self.relative_scale - 1))
        sy = int(sy * 2 ** (self.relative_scale - 1))
        for i in range(self.num_scales):
            x = F.interpolate(target_images, size=(sx, sy), mode='bilinear', align_corners=False)
            if sx > self.max_image_size[0] or sy > self.max_image_size[1]:
                x = TF.center_crop(x, self.max_image_size)
            target_features.append(self.get_vgg_features(x))
            sx, sy = sx * 2, sy * 2

        return target_features

    def forward(self, input_dict, return_summary=True):
        """

        :param input_dict:  A dictionary containing the necessary tensors for calculating the appearance loss.
                            required keys: ['rendered_images']: a tensor of shape [b, c, h, w]
                            The rendered images should be in range [0, 1].
                            The number of channels c should match the total number of target channels.
        :param return_summary: Whether to return a summary dictionary
        :return: A tuple (loss, loss_log dictionary, summary dictionary)
        """

        channels = input_dict['rendered_images'].shape[1]
        assert channels == self.total_channels, \
            f"Target images have {self.total_channels} channels in total," \
            f"but the rendered images have {channels} channels."

        loss = 0.0
        style_loss = 0.0
        ac_loss = 0.0

        summary = None
        for scale in range(self.num_scales):
            rendered_images = input_dict['rendered_images']
            if scale != 0:
                h, w = rendered_images.shape[2:4]
                h, w = h // (2 ** scale), w // (2 ** scale)
                random_crop = torchvision.transforms.RandomCrop((h, w))
                num_random_crops = max(1, int(2.0 * scale))
                rendered_images = [random_crop(rendered_images) for _ in range(num_random_crops)]
                rendered_images = torch.cat(rendered_images, dim=0)  # [b * num_random_crops, c, h, w]
            xs = []
            for key in sorted(self.output_channels):
                oc = self.output_channels[key]
                x = rendered_images[:, oc]

                # # Resize if the image size is different from the target image size
                # if x.shape[2] != self.target_image_size[0] or x.shape[3] != target_image_size.image_size[1]:
                #     x = TF.resize(x, self.target_image_size, antialias=True)

                # Repeat the mono-color images to 3 channels
                if len(oc) == 1:
                    x = x.repeat(1, 3, 1, 1)

                xs.append(x)

            target_images = self.target_images
            target_features = self.target_features[scale]
            if self.num_pseudo_targets > 0: # Include a single random pseudo target
                idx = np.random.randint(self.num_pseudo_targets)
                key = f"pseudo_target_{idx}"
                oc = self.pseudo_output_channels[key]
                x = rendered_images[:, oc]
                xs.append(x)
                target_images = torch.cat(
                    [target_images, self.pseudo_target_images[idx: idx + 1]], dim=0)

                res = []
                for ptf, tf in zip(self.pseudo_target_features[scale], self.target_features[scale]):
                    res.append(torch.cat([tf, ptf[idx:idx + 1]], dim=0))

                target_features = res

            generated_images = torch.stack(xs, dim=1)  # [b, n_targets + 1, 3, h, w]

            b, n_targets, _, h, w = generated_images.shape

            if return_summary and scale == 0:
                with torch.no_grad():
                    images = generated_images.permute(0, 1, 3, 4, 2)  # [b, n_targets, h, w, 3]
                    target_images = TF.resize(target_images, [h, w], antialias=True)
                    images = torch.cat([target_images.permute(0, 2, 3, 1).unsqueeze(0), images], dim=0)
                    images = torch.hstack([
                        torch.vstack([images[i, j] for j in range(images.shape[1])])
                        for i in range(images.shape[0])
                    ])
                    images = torch.clamp(images, 0.0, 1.0).cpu().numpy()
                    images = (images * 255).astype('uint8')
                    summary = {
                        "images": Image.fromarray(images)
                    }
            generated_images = generated_images.view(-1, 3, h, w) # [b * n_targets, 3, h, w]
            # Resize the generated images to the patch size
            generated_images = F.interpolate(generated_images, size=self.patch_size, mode='bilinear',
                                             align_corners=False)

            generated_features = self.get_vgg_features(generated_images)

            style_loss += self.style_loss_fn(target_features, generated_features)  # [n_targets]

            if scale == 0 and self.ac_loss_weight > 0.0:
                ac_loss += self.ac_loss_fn(target_features, generated_features)  # [n_targets]

        style_loss = style_loss / self.num_scales  # [n_targets]
        loss += style_loss + self.ac_loss_weight * ac_loss
        loss_log = {
            f"{k}": style_loss[i] for i, k in enumerate(sorted(self.output_channels))
        }
        if self.ac_loss_weight > 0.0:
            for i, k in enumerate(sorted(self.output_channels)):
                loss_log[f"AC-{k}"] = ac_loss[i]
        
        if self.num_pseudo_targets > 0:
            loss_log['pseudo_target'] = loss[-1]  # Last element is the pseudo target loss

        return loss.mean(), loss_log, summary


class OptimalTransportLoss(torch.nn.Module):
    def __init__(self, n_samples=1024, device='cuda:0'):
        super().__init__()

        self.n_samples = n_samples

        self.device = device
        self.color_transform = torch.tensor(
            [[0.577350, 0.577350, 0.577350],
             [-0.577350, 0.788675, -0.211325],
             [-0.577350, -0.211325, 0.788675]], device=device, requires_grad=False)  # [3, 3] matrix

    def rgb_to_yuv(self, rgb):
        """
        :param rgb: Torch tensor of shape [b, n, 3]
        :return: YUV colors of shape [b, n, 3]
        """
        return torch.einsum('bnc,ck->bnk', rgb, self.color_transform)

    def color_matching_loss(self, x, y):
        """
        Color matching distance between two RGB images.
        :param x: (b, n, 3)
        :param y: (b, m, 3)

        :return: (b, n, m)
        """
        x_yuv = self.rgb_to_yuv(x)
        y_yuv = self.rgb_to_yuv(y)

        pairwise_distance = self.pairwise_distances_l2(x_yuv, y_yuv) + self.pairwise_distances_cos(x_yuv, y_yuv)

        m1, m1_inds = pairwise_distance.min(1)
        m2, m2_inds = pairwise_distance.min(2)

        remd = torch.max(m1.mean(dim=1), m2.mean(dim=1))

        return remd

    @staticmethod
    def pairwise_distances_l2(x, y):
        """
        Pairwise L2 distance between two flattened feature sets.
        :param x: (b, n, c)
        :param y: (b, m, c)

        :return: (b, n, m)
        """
        # x, y: (b, n or m, c)
        x_norm = torch.norm(x, dim=2, keepdim=True) ** 2  # (b, n, 1)
        y_t = y.transpose(1, 2)  # (b, c, m) (m may be different from n)
        y_norm = torch.norm(y_t, dim=1, keepdim=True) ** 2  # (b, 1, m)
        cross = torch.matmul(x, y_t)
        dist = x_norm + y_norm - 2.0 * cross  # x + y is of shape b, n, m because of point-wise adding (broadcasting)
        return torch.clamp(dist, 1e-5, 1e5) / x.size(2)

    @staticmethod
    def pairwise_distances_cos(x, y):
        """
        Pairwise Cosine distance between two flattened feature sets.
        :param x: (b, n, c)
        :param y: (b, m, c)

        :return: (b, n, m)
        """
        x_norm = torch.norm(x, dim=2, keepdim=True)  # (b, n, 1)
        y_t = y.transpose(1, 2)  # (b, c, m) (m may be different from n)
        y_norm = torch.norm(y_t, dim=1, keepdim=True)  # (b, 1, m)
        dist = 1. - torch.matmul(x, y_t) / (x_norm * y_norm + 1e-10)  # (b, n, m)
        return dist

    @staticmethod
    def style_loss(x, y, metric="cos"):
        """
        Relaxed Earth Mover's Distance (EMD) between two sets of features.
        :param x: (b, n, c)
        :param y: (b, m, c)
        :param metric: Either 'cos' or 'L2'

        :return: (b, n, m)
        """
        if metric == "cos":
            pairwise_distance = OptimalTransportLoss.pairwise_distances_cos(x, y)
        else:
            pairwise_distance = OptimalTransportLoss.pairwise_distances_l2(x, y)

        m1, m1_inds = pairwise_distance.min(1)
        m2, m2_inds = pairwise_distance.min(2)

        remd = torch.max(m1.mean(dim=1), m2.mean(dim=1))

        return remd

    @staticmethod
    def moment_loss(x, y):
        """
        Calculates the distance between the first and second moments of two sets of features.
        :param x: (b, n, c)
        :param y: (b, m, c)

        :return: (b, n, m)
        """
        mu_x = torch.mean(x, 1, keepdim=True)
        mu_y = torch.mean(y, 1, keepdim=True)
        mu_diff = torch.abs(mu_x - mu_y).mean(dim=(1, 2))

        x_c = x - mu_x
        y_c = y - mu_y
        x_cov = torch.matmul(x_c.transpose(1, 2), x_c) / (x.shape[1] - 1)
        y_cov = torch.matmul(y_c.transpose(1, 2), y_c) / (y.shape[1] - 1)

        cov_diff = torch.abs(x_cov - y_cov).mean(dim=(1, 2))
        return mu_diff + cov_diff

    def forward(self, target_features, generated_features):
        """
        Calculate the optimal transport style loss between two sets of features.

        :param target_features:     List of features for the target images.
                                    Each feature is of shape (n_targets, c, h, w) with varying c, h, w
        :param generated_features:  List of features for the generated images.
                                    Each feature is of shape (b * n_targets, c, h, w) with varying c, h, w

        :return: The OT style loss between the target and generated features [n_targets]
        """
        loss = 0.0

        # Iterate over the VGG layers
        for i, (y, x) in enumerate(zip(target_features, generated_features)):
            layer_weight = 1.0

            n_targets, c_y, h_y, w_y = y.shape

            b, c_x, h_x, w_x = x.shape
            batch_size = b // n_targets
            assert batch_size * n_targets == b, "Batch size must be a multiple of the number of target images"

            # We repeat the target features to match the batch size of the generated features
            # y = y.repeat_interleave(repeats=batch_size, dim=0)
            y = y.repeat(batch_size, 1, 1, 1)

            n_x, n_y = h_x * w_x, h_y * w_y
            x = x.view(b, c_x, n_x)
            y = y.view(b, c_y, n_y)

            # We randomly select n_samples point from the features to calculate the OT loss
            n_samples = min(n_x, n_y, self.n_samples)

            indices_x = torch.argsort(torch.rand(b, 1, n_x, device=x.device), dim=-1)[..., :n_samples]
            x = x.gather(-1, indices_x.expand(b, c_x, n_samples))

            indices_y = torch.argsort(torch.rand(b, 1, n_y, device=y.device), dim=-1)[..., :n_samples]
            y = y.gather(-1, indices_y.expand(b, c_y, n_samples))

            x = x.transpose(1, 2)  # (b, n_samples, c)
            y = y.transpose(1, 2)  # (b, n_samples, c)

            if i == 0 and c_x == c_y == 3:
                layer_loss = self.color_matching_loss(x, y)
            else:
                layer_loss = OptimalTransportLoss.style_loss(x, y) + OptimalTransportLoss.moment_loss(x, y)
            loss += layer_loss * layer_weight

        loss = loss.view(batch_size, n_targets).mean(dim=0)
        return loss  # [n_targets]


class AutoCorrelationLoss(torch.nn.Module):
    """
    FFT-based auto-correlation loss on selected VGG feature layers.

    Inputs to forward():
        target_features:  List[T_l], where each T_l has shape (n_targets, C_l, H_l, W_l)
        generated_features: List[G_l], where each G_l has shape (B, C_l, H_l, W_l)
            with B = batch_size * n_targets

    Returns:
        loss_vec: Tensor of shape (n_targets,), averaged across the batch repeats,
                  and summed across the selected layers.

    Notes:
        - 'layers' are indices relative to the list returned from your get_vgg_features()
          (i.e., 0 is the first item in that list).
        - We aggregate channels BEFORE inverse FFT (sum |FFT_c|^2 over channels)
          to reduce memory.
    """
    def __init__(self,
                 layers=(0, 1),               # use first two VGG layers by default (relu1_1, relu2_1 in your setup)
                 reduction="l1",              # "l1" or "mse"
                 normalize=True,              # divide by zero-lag so maps are scale-invariant
                 exclude_zero_lag=True,       # set zero-lag to 0 to avoid trivial agreement
                 center_crop=None,            # None or int or (half_h, half_w); crops around (H//2, W//2) after fftshift
                 fftshift=False,              # if True, center zero-lag in the middle before optional cropping
                 layer_weights=None,          # None or list of weights (same length as 'layers')
                 eps=1e-6,
                 fft_norm="backward"):        # FFT norm; "backward" is fine since we normalize
        super().__init__()
        assert reduction in ("l1", "mse")
        self.layers = tuple(layers)
        self.reduction = reduction
        self.normalize = normalize
        self.exclude_zero_lag = exclude_zero_lag
        self.center_crop = center_crop
        self.fftshift = fftshift
        self.layer_weights = layer_weights
        self.eps = eps
        self.fft_norm = fft_norm

    @staticmethod
    def _roll_center(x):
        """Roll so that zero-lag (0,0) moves to the spatial center."""
        H, W = x.shape[-2], x.shape[-1]
        return torch.roll(x, shifts=(H // 2, W // 2), dims=(-2, -1))

    @staticmethod
    def _center_crop(x, half_hw):
        """Center crop with half sizes (half_h, half_w). Keeps (2*half_h+1, 2*half_w+1)."""
        H, W = x.shape[-2], x.shape[-1]
        if isinstance(half_hw, int):
            half_h = half_w = half_hw
        else:
            half_h, half_w = half_hw
        top = max(H // 2 - half_h, 0)
        left = max(W // 2 - half_w, 0)
        h = min(2 * half_h + 1, H)
        w = min(2 * half_w + 1, W)
        return x[..., top:top + h, left:left + w]

    def _autocorr_map(self, feat):
        """
        Compute channel-aggregated autocorrelation map for a feature tensor.
        feat: (B, C, H, W) -> returns (B, H, W)
        """
        B, C, H, W = feat.shape

        # zero-mean per channel to prevent DC dominance
        f = feat - feat.mean(dim=(-2, -1), keepdim=True)

        # real FFT over spatial dims
        Fz = torch.fft.rfft2(f, dim=(-2, -1), norm=self.fft_norm)         # (B, C, H, W//2+1)
        power = (Fz.real ** 2 + Fz.imag ** 2).sum(dim=1)                   # (B, H, W//2+1)

        # inverse FFT to get circular autocorrelation
        ac = torch.fft.irfft2(power, s=(H, W), dim=(-2, -1), norm=self.fft_norm)  # (B, H, W)

        if self.normalize:
            # normalize so ac[..., 0, 0] == 1
            z = ac[..., 0:1, 0:1].clamp_min(self.eps)
            ac = ac / z

        if self.fftshift:
            ac = self._roll_center(ac)

        if self.exclude_zero_lag:
            if self.fftshift:
                ac[..., H // 2, W // 2] = 0
            else:
                ac[..., 0, 0] = 0

        if self.center_crop is not None:
            # If fftshift=False, we still crop around the spatial center—this keeps a symmetric window of lags
            # (for circular AC, it is equivalent to cropping around zero-lag after a conceptual shift).
            if not self.fftshift:
                ac = self._roll_center(ac)
                ac = self._center_crop(ac, self.center_crop)
                ac = self._roll_center(ac)  # roll back so gradients still correspond to original indexing
            else:
                ac = self._center_crop(ac, self.center_crop)
        return ac

    def _layer_loss(self, gen_ac, tgt_ac, reduction):
        """
        gen_ac: (B, H, W)
        tgt_ac: (n_targets, H, W) -> will be repeated over batch repeats
        returns: (n_targets,)
        """
        n_targets = tgt_ac.shape[0]
        B = gen_ac.shape[0]
        assert B % n_targets == 0, "Generated feature batch must be a multiple of n_targets."
        batch_size = B // n_targets

        tgt_ac_rep = tgt_ac.repeat(batch_size, 1, 1)  # (B, H, W)

        if reduction == "l1":
            d = torch.abs(gen_ac - tgt_ac_rep).mean(dim=(-2, -1))  # (B,)
        else:  # mse
            d = ((gen_ac - tgt_ac_rep) ** 2).mean(dim=(-2, -1))    # (B,)

        return d.view(batch_size, n_targets).mean(dim=0)           # (n_targets,)

    def forward(self, target_features, generated_features):
        """
        Compute per-target auto-correlation loss (summed over selected layers).

        :param target_features:    list of tensors per VGG layer, each (n_targets, C, H, W)
        :param generated_features: list of tensors per VGG layer, each (B, C, H, W) with B = batch_size * n_targets
        :return: Tensor (n_targets,)
        """
        if self.layers is not None and len(self.layers) == 0:
            raise ValueError("AutoCorrelationLoss: 'layers' must contain at least one layer index.")

        loss_vec = None
        for i, (t, g) in enumerate(zip(target_features, generated_features)):
            if self.layers is not None and i not in self.layers:
                continue

            # Do not backprop through target statistics
            with torch.no_grad():
                t_ac = self._autocorr_map(t)  # (n_targets, H, W)

            g_ac = self._autocorr_map(g)      # (B, H, W)

            li = self._layer_loss(g_ac, t_ac, self.reduction)  # (n_targets,)

            if self.layer_weights is not None:
                # weight for this logical layer index within 'layers'
                try:
                    w = float(self.layer_weights[self.layers.index(i)])
                except Exception:
                    w = 1.0
            else:
                w = 1.0

            loss_vec = li * w if loss_vec is None else (loss_vec + li * w)

        if loss_vec is None:
            raise ValueError("AutoCorrelationLoss: none of the specified 'layers' matched the provided feature lists.")
        return loss_vec


if __name__ == '__main__':
    # loss_fn = AppearanceLoss(target_images_path="../data/textures/bubbly_0101.jpg")
    from utils.misc import process_output_channels

    channels_dict = {  # Number of channels in the model to assign to each target image
        "albedo": 3,
        "height": 1,
        "normal": 3,
        # "roughness": 1,
        # "ambient_occlusion": 1,
    }
    total_channels, output_channels = process_output_channels(channels_dict)
    print(output_channels)
    loss_fn = AppearanceLoss(
        target_image_size=(1024, 1024),
        patch_size=(256, 256),
        num_scales=1,
        relative_scale=1,
        total_channels=total_channels,
        output_channels=output_channels,
        target_images_path={
            "albedo": "../data/pbr_textures/Abstract_008/albedo.jpg",
            "height": "../data/pbr_textures/Abstract_008/height.jpg",
            "normal": "../data/pbr_textures/Abstract_008/normal.jpg",
        },
        device="mps",
        num_pseudo_targets=5,
        ac_loss_weight=1.0,
        ac_kwargs={},
    )

    with torch.no_grad():
        print(loss_fn.target_images.shape)
        x = loss_fn.target_images
        # x = F.interpolate(x, size=(512, 512), mode='bilinear', align_corners=False)
        # x = torchvision.transforms.RandomCrop((512, 512))(x)
        # x = x.reshape(-1, 512, 512).unsqueeze(0).repeat(4, 1, 1, 1)

        x = x.reshape(-1, 1024, 1024).unsqueeze(0).repeat(4, 1, 1, 1)
        print(x.shape)
        x = x[:, [0, 1, 2, 3, 6, 7, 8]]
        # x[0, :3] = torch.rand_like(x[0, :3])
        x[:, :3] = torch.roll(x[:, :3], 50, dims=2)
        input_dict = {
            'rendered_images': x
        }

        loss, loss_log, summary = loss_fn(input_dict, return_summary=False)
        loss, loss_log, summary = loss_fn(input_dict, return_summary=True)
    print(loss_log)
    print(summary)
    summary['images'].show()

    print(loss_fn.target_images.shape)
