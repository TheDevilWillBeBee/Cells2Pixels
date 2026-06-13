import torch
import torchvision.transforms.functional as TF
import numpy as np
import os
import sys
from models.optic_flow import MSOEmultiscale
from utils.misc import flow_to_image, plot_vec_field


class MotionLoss(torch.nn.Module):
    def __init__(self, loss_type="vector_field", target_name="circular", image_size=(128, 128),
                 strength_loss_weight=1.0, direction_loss_weight=1.0, base_num_steps=1, device='cuda:0'):
        super(MotionLoss, self).__init__()

        assert loss_type == "vector_field"
        self.image_size = image_size
        if isinstance(image_size, int):
            self.image_size = [image_size, image_size]

        self.strength_loss_weight = strength_loss_weight
        self.direction_loss_weight = direction_loss_weight
        self.base_num_steps = base_num_steps
        self.device = device

        print('Target Vector Field: ', target_name)
        self.target_vector_field = get_motion_vector_field_by_name(target_name,
                                                                   img_size=self.image_size).to(device)

        self.cos_sim = torch.nn.CosineSimilarity(dim=1)
        self.motion_model = _load_MSOEmultiscale_model().to(device).eval()
        print(f"Successfully Loaded the pretrained Optic Flow model.")

        self._create_losses()

    def _create_losses(self):
        self.loss_mapper = {}
        self.loss_weights = {}

        if self.strength_loss_weight > 0:
            self.loss_mapper['strength'] = self.get_motion_strength_loss
            self.loss_weights['strength'] = self.strength_loss_weight

        if self.direction_loss_weight > 0:
            self.loss_mapper['direction'] = self.get_cosine_dist
            self.loss_weights['direction'] = self.direction_loss_weight

    def get_motion_strength_loss(self, optic_flow, num_steps=1):
        motion_strength = torch.norm(optic_flow, dim=1) * self.base_num_steps / num_steps
        target_strength = torch.norm(self.target_vector_field, dim=1)
        motion_strength_loss = torch.abs(motion_strength - target_strength)

        direction_cos_sim = self.cos_sim(optic_flow, self.target_vector_field)
        cos_loss = 1.0 - torch.mean(direction_cos_sim, dim=[1, 2], keepdim=True)
        # Keep the batch dimension

        alpha = (1.0 - torch.clip(cos_loss, 0.0, 1.0)).detach()
        motion_strength_loss = motion_strength_loss * alpha
        motion_strength_loss = torch.mean(motion_strength_loss)

        return motion_strength_loss

    def get_cosine_dist(self, optic_flow, num_steps=1):
        direction_cos_sim = self.cos_sim(optic_flow, self.target_vector_field)
        direction_loss = 1.0 - torch.mean(direction_cos_sim)
        return direction_loss

    def get_opticflow(self, image1, image2, size=(128, 128), return_summary=False, num_steps=1):
        image1_size = image1.shape[2]
        image2_size = image2.shape[2]
        if image1_size != size[0]:
            image1 = TF.resize(image1, size)
        if image2_size != size[0]:
            image2 = TF.resize(image2, size)

        # MSOEnet accepts grayscale [0,1]
        #         x1 = (image1 + 1.0) / 2.0
        #         x2 = (image2 + 1.0) / 2.0
        x1 = image1
        x2 = image2
        x1 = TF.rgb_to_grayscale(x1)
        x2 = TF.rgb_to_grayscale(x2)
        image_cat = torch.stack([x1, x2], dim=-1)
        flow, _ = self.motion_model(image_cat, return_features=True)

        if return_summary:
            flow_img = flow_to_image(flow[0].permute(1, 2, 0).detach().cpu().numpy()).transpose(2, 0, 1)
            rescaled_flow = flow * self.base_num_steps / num_steps
            mean_rescaled_flow = torch.mean(rescaled_flow, dim=0)
            flow_vector_field = plot_vec_field(mean_rescaled_flow.detach().cpu().numpy(), name='Generated')
        else:
            flow_img = None
            flow_vector_field = None

        return flow, flow_img, flow_vector_field

    def forward(self, input_dict, return_summary=True):
        """
        Images are assumed to be in the range [0, 1]
        :param input_dict: A dictionary containing the following keys:
            - 'image_before': The image before the NCA step, shape [B, 3, H, W].
            - 'image_after': The image after the NCA step, shape [B, 3, H, W].
        """
        generated_image_before_nca = input_dict['image_before']
        generated_image_after_nca = input_dict['image_after']
        num_steps = input_dict['step_n']

        optic_flow, flow_img, flow_vector_field = self.get_opticflow(generated_image_before_nca,
                                                                     generated_image_after_nca,
                                                                     size=self.image_size,
                                                                     return_summary=return_summary,
                                                                     num_steps=num_steps)

        loss = 0
        loss_log_dict = {}

        for loss_name in self.loss_mapper:
            loss_weight = self.loss_weights[loss_name]
            loss_func = self.loss_mapper[loss_name]
            cur_loss = loss_func(optic_flow, num_steps)
            loss_log_dict[loss_name] = cur_loss
            loss += loss_weight * cur_loss

        summary = None
        if return_summary:
            summary = {}
            summary['generated_OF'] = flow_img[None, ...]
            summary['generated_VF'] = flow_vector_field

            target_flow_vector_field = plot_vec_field(self.target_vector_field[0].detach().cpu().numpy(),
                                                      name='Target')
            summary['target_VF'] = target_flow_vector_field

        return loss, loss_log_dict, summary


def show_progress(block_num, block_size, total_size):
    downloaded = block_num * block_size
    if total_size <= 0:
        sys.stderr.write(f"\rDownloaded {downloaded / (1024 * 1024):.1f} MB")
        sys.stderr.flush()
        return

    downloaded = min(downloaded, total_size)
    percent = 100.0 * downloaded / total_size
    sys.stderr.write(
        f"\rDownloading: {percent:6.2f}% "
        f"({downloaded / (1024 * 1024):.1f}/{total_size / (1024 * 1024):.1f} MB)"
    )
    if downloaded >= total_size:
        sys.stderr.write("\n")
    sys.stderr.flush()


def _load_MSOEmultiscale_model(model_name="two_stream_dynamic_model", download=False):
    models_path = "pretrained"
    assert model_name == 'two_stream_dynamic_model'
    if not os.path.exists(f'{models_path}/two_stream/{model_name}.pth'):
        download = True

    if download:

        if os.path.exists(f'{models_path}/two_stream/{model_name}.pth'):
            os.system(f"rm -rf {models_path}/two_stream")
        import gdown
        url = 'https://drive.google.com/uc?id=10qoSx0P3TJzf17bUN42x1ZAFNjr-J69f'
        output = f'{models_path}/two_stream/{model_name}.pth'
        os.system(f"mkdir -p {models_path}/two_stream/")
        gdown.download(url, output, quiet=False)

    model = MSOEmultiscale()
    states_dict = torch.load(f'{models_path}/two_stream/{model_name}.pth')
    model.load_state_dict(states_dict)
    model = model.eval()

    return model


# TODO: Rewrite this function using tensor operations instead of for loops
def get_motion_vector_field_by_name(motion_vector_field_name, img_size=[128, 128]):
    try:
        motion_direction = int(motion_vector_field_name)
        simple_direction = True
    except:
        simple_direction = False
    if simple_direction:
        motion_direction = int(motion_vector_field_name)
        torch_pi = torch.FloatTensor([3.1416])
        motion_rad = motion_direction / 180.0 * torch_pi
        target_motion_vec = torch.zeros((1, 2, img_size[0], img_size[1]))
        target_motion_vec[:, 0, ...] = torch.cos(motion_rad)
        target_motion_vec[:, 1, ...] = torch.sin(motion_rad)

        return target_motion_vec

    target_motion_vec = torch.zeros((1, 2, img_size[0], img_size[1]))

    if 'grad' in motion_vector_field_name:
        # For example grad_0_180
        # The first degree determines the direction of the motion
        # The second one determines the direction of motion magnitude gradient
        theta = int(motion_vector_field_name.split("_")[1])
        phi = int(motion_vector_field_name.split("_")[2])
        torch_pi = torch.FloatTensor([3.1416])

        theta = theta / 180.0 * torch_pi
        phi = phi / 180.0 * torch_pi

        center_x = img_size[0] // 2
        center_y = img_size[1] // 2
        for i in range(-center_x, center_x):
            for j in range(-center_y, center_y):
                alpha = (j * torch.cos(phi) + i * torch.sin(phi))

                target_motion_vec[0, 0, center_x + i, center_y + j] = alpha
                target_motion_vec[0, 1, center_x + i, center_y + j] = alpha

        # Adjust the minimum motion strength to 0.2
        target_motion_vec = target_motion_vec - target_motion_vec.min() + 0.2
        target_motion_vec[:, 0, ...] *= torch.cos(theta)
        target_motion_vec[:, 1, ...] *= torch.sin(theta)

        avg_motion_strength = torch.norm(target_motion_vec, dim=1).mean()
        target_motion_vec = target_motion_vec / avg_motion_strength


    elif motion_vector_field_name == 'hyperbolic':
        center_x = img_size[0] // 2
        center_y = img_size[1] // 2
        max_radius = (center_x ** 2 + center_y ** 2) ** 0.5
        for i in range(-center_x, center_x):
            for j in range(-center_y, center_y):
                radius = (i ** 2 + j ** 2) ** 0.5
                if radius == 0:
                    continue
                cosine = 4.0 * i / max_radius
                sine = 4.0 * j / max_radius
                target_motion_vec[0, 0, center_x + i, center_y + j] = cosine  # sine + pi/2
                target_motion_vec[0, 1, center_x + i, center_y + j] = sine  # cosine + pi/2

        avg_motion_strength = torch.norm(target_motion_vec, dim=1).mean()
        target_motion_vec = target_motion_vec / avg_motion_strength

    elif motion_vector_field_name == 'circular':
        center_x = img_size[0] // 2
        center_y = img_size[1] // 2
        max_radius = (center_x ** 2 + center_y ** 2) ** 0.5
        for i in range(-center_x, center_x):
            for j in range(-center_y, center_y):
                radius = (i ** 2 + j ** 2) ** 0.5
                if radius == 0:
                    continue
                cosine = 4.0 * i / max_radius
                sine = 4.0 * j / max_radius
                target_motion_vec[0, 0, center_x + i, center_y + j] = cosine  # sine + pi/2
                target_motion_vec[0, 1, center_x + i, center_y + j] = -sine  # cosine + pi/2

        avg_motion_strength = torch.norm(target_motion_vec, dim=1).mean()
        target_motion_vec = target_motion_vec / avg_motion_strength
    elif motion_vector_field_name == 'circle':
        center_x = img_size[0] // 2
        center_y = img_size[1] // 2
        for i in range(-center_x, center_x):
            for j in range(-center_y, center_y):
                radius = (i ** 2 + j ** 2) ** 0.5
                if radius == 0:
                    continue
                cosine = i / radius
                sine = j / radius
                target_motion_vec[0, 0, center_x + i, center_y + j] = cosine  # sine + pi/2
                target_motion_vec[0, 1, center_x + i, center_y + j] = -sine  # cosine + pi/2
    elif motion_vector_field_name == 'converge':
        center_x = img_size[0] // 2
        center_y = img_size[1] // 2
        for i in range(-center_x, center_x):
            for j in range(-center_y, center_y):
                radius = (i ** 2 + j ** 2) ** 0.5
                if radius == 0:
                    continue
                cosine = i / radius
                sine = j / radius
                target_motion_vec[0, 0, center_x + i, center_y + j] = -sine  #
                target_motion_vec[0, 1, center_x + i, center_y + j] = -cosine  #
    elif motion_vector_field_name == 'diverge':
        center_x = img_size[0] // 2
        center_y = img_size[1] // 2
        for i in range(-center_x, center_x):
            for j in range(-center_y, center_y):
                radius = (i ** 2 + j ** 2) ** 0.5
                if radius == 0:
                    continue
                cosine = i / radius
                sine = j / radius
                target_motion_vec[0, 0, center_x + i, center_y + j] = sine  #
                target_motion_vec[0, 1, center_x + i, center_y + j] = cosine  #
    elif motion_vector_field_name == '2block_x':
        center_x = img_size[0] // 2
        center_y = img_size[1] // 2
        torch_pi = torch.FloatTensor([3.1416])

        for i in range(-center_x, center_x):
            for j in range(-center_y, center_y):
                if i >= 0 and j >= 0:
                    rad = 0
                elif i < 0 and j < 0:
                    rad = 180
                elif i >= 0 and j < 0:
                    rad = 0
                elif i < 0 and j >= 0:
                    rad = 180
                motion_rad = rad / 180.0 * torch_pi
                target_motion_vec[0, 0, center_x + i, center_y + j] = torch.cos(motion_rad)
                target_motion_vec[0, 1, center_x + i, center_y + j] = torch.sin(motion_rad)

    elif motion_vector_field_name == '2block_y':
        center_x = img_size[0] // 2
        center_y = img_size[1] // 2
        torch_pi = torch.FloatTensor([3.1416])

        for i in range(-center_x, center_x):
            for j in range(-center_y, center_y):
                if i >= 0 and j >= 0:
                    rad = 90
                elif i < 0 and j < 0:
                    rad = -90
                elif i >= 0 and j < 0:
                    rad = 90
                elif i < 0 and j >= 0:
                    rad = -90
                motion_rad = rad / 180.0 * torch_pi
                target_motion_vec[0, 0, center_x + i, center_y + j] = torch.cos(motion_rad)
                target_motion_vec[0, 1, center_x + i, center_y + j] = torch.sin(motion_rad)
    elif motion_vector_field_name == '3block':
        center_x = img_size[0] // 2
        center_y = img_size[1] // 2
        torch_pi = torch.FloatTensor([3.1416])

        for i in range(-center_x, center_x):
            for j in range(-center_y, center_y):
                if i >= 0 and j >= 0:
                    rad = 0
                elif i < 0 and j < 0:
                    rad = 90
                elif i >= 0 and j < 0:
                    rad = 0
                elif i < 0 and j >= 0:
                    rad = 180
                motion_rad = rad / 180.0 * torch_pi
                target_motion_vec[0, 0, center_x + i, center_y + j] = torch.cos(motion_rad)
                target_motion_vec[0, 1, center_x + i, center_y + j] = torch.sin(motion_rad)
    elif motion_vector_field_name == '4block':
        center_x = img_size[0] // 2
        center_y = img_size[1] // 2
        torch_pi = torch.FloatTensor([3.1416])

        for i in range(-center_x, center_x):
            for j in range(-center_y, center_y):
                if i >= 0 and j >= 0:
                    rad = 0
                elif i < 0 and j < 0:
                    rad = 180
                elif i >= 0 and j < 0:
                    rad = 90
                elif i < 0 and j >= 0:
                    rad = 270
                motion_rad = rad / 180.0 * torch_pi
                target_motion_vec[0, 0, center_x + i, center_y + j] = torch.cos(motion_rad)
                target_motion_vec[0, 1, center_x + i, center_y + j] = torch.sin(motion_rad)
    else:
        print('Not Implemented Motion Field')
        exit()

    return target_motion_vec


if __name__ == "__main__":
    with torch.no_grad():
        device = torch.device("mps")
        loss_fn = MotionLoss("vector_field", target_name="circular", image_size=(128, 128), base_num_steps=4,
                             device=device)

        x_before = torch.rand((1, 3, 128, 128), device=device)
        x_after = torch.roll(x_before, shifts=1, dims=2)

        input_dict = {
            'image_before': x_before,
            'image_after': x_after,
            'step_n': 4
        }

        loss, loss_log, summary = loss_fn(input_dict, return_summary=True)

        print(loss)
        print(summary)
        summary["generated_flow_vector_field"].show()
