import numpy as np
import torch
from tqdm import tqdm

from losses.loss import Loss
from models.nca2d import GrowingNCA
from models.siren import Siren
from training.common import (
    TestOptions,
    device_config,
    load_checkpoint_pair,
    load_graft_if_configured,
    normalize_model_grads,
    save_checkpoint,
    set_seed,
)
from training.tasks.base import BaseTask
from utils.render import Renderer2D
from utils.video import VideoWriter


class Growing2DTask(BaseTask):
    def _build(self, load: bool = False):
        nca_kwargs = device_config(self.config["nca"]["nca_kwargs"], self.device)
        model = GrowingNCA(**nca_kwargs).to(self.device)
        nca_output_dim = model.channels
        if self.config["nca"]["output_type"] == "z":
            nca_output_dim *= model.perception_kernels
        siren = Siren(in_features=nca_output_dim, coord_dim=2, out_features=4, **self.config["siren"]).to(self.device)
        if load:
            load_checkpoint_pair(self.config, model, siren, device=self.device)
        return model, siren

    def _loss_renderer_grid(self):
        loss_fn = Loss(**self.config["loss"])
        renderer = Renderer2D(**self.config["renderer"])
        grid_size = loss_fn.loss_mapper["image"].grid_size
        grid_size = (grid_size[0] // renderer.scale_factor, grid_size[1] // renderer.scale_factor)
        return loss_fn, renderer, grid_size

    def train(self) -> None:
        set_seed(self.config.get("seed", 43))
        model, siren = self._build()
        self._log_counts(model, siren, "GrowingNCA")
        rep_start = load_graft_if_configured(self.config, "nca", model, siren, self.device)
        with torch.no_grad():
            loss_fn, renderer, grid_size = self._loss_renderer_grid()

        train_cfg = self.config["train"]
        for repetition in range(rep_start, train_cfg["num_repetitions"]):
            with torch.no_grad():
                pool = model.seed(train_cfg["pool_size"], grid_size[0], grid_size[1])

            optimizer, scheduler = self._optimizer(list(model.parameters()) + list(siren.parameters()))
            for epoch in tqdm(range(train_cfg["epochs"]), desc=f"Repetition {repetition + 1}/{train_cfg['num_repetitions']}"):
                log_step = epoch + repetition * train_cfg["epochs"]
                with torch.no_grad():
                    batch_idx = np.random.choice(len(pool), train_cfg["batch_size"], replace=False)
                    x = pool[batch_idx]
                    if log_step % train_cfg["inject_seed_interval"] == 0:
                        x[:1] = model.seed(1, grid_size[0], grid_size[1])

                step_n = np.random.randint(*train_cfg["step_range"])
                for _ in range(step_n):
                    x, z = model(x)

                x_render = x if self.config["nca"]["output_type"] == "s" else z
                rendered = renderer.render(x_render.permute(0, 2, 3, 1), siren, None, fs_shader="vanilla")
                rendered = rendered.permute(0, 3, 1, 2)
                x_up = torch.nn.functional.interpolate(x, scale_factor=renderer.scale_factor, mode="bilinear")
                living_mask = model.get_living_mask(x_up).float()
                rendered = rendered * living_mask
                input_dict = {
                    "generated_images": rendered,
                    "alpha": x_up[:, 3:4],
                    "nca_state": x,
                }

                return_summary = log_step % train_cfg["summary_interval"] == 0
                loss, loss_log, summary = loss_fn(input_dict, return_summary=return_summary)
                if return_summary and "image-images" in summary:
                    self._save_logged_image("nca_output", summary["image-images"], log_step)

                loss.backward()
                with torch.no_grad():
                    normalize_model_grads(model)
                    optimizer.step()
                    optimizer.zero_grad()
                    scheduler.step()
                    pool[batch_idx] = x
                    loss_fn.update_loss_weights(loss_log, log_step)

                self.logger.log_metrics(loss_log, step=log_step)

            save_checkpoint(self.config, model, siren, suffix=f"_{repetition + 1}")

        save_checkpoint(self.config, model, siren)

    @torch.no_grad()
    def test(self, options: TestOptions) -> None:
        model, siren = self._build(load=True)
        loss_fn, renderer, grid_size = self._loss_renderer_grid()
        output_dir = self._output_dir(options)
        x = model.seed(1, grid_size[0], grid_size[1])
        z = None
        for _ in tqdm(range(options.steps), desc="Test rollout"):
            x, z = model(x)

        def render_frame(state, perception=None):
            x_render = state if self.config["nca"]["output_type"] == "s" or perception is None else perception
            image = renderer.render(x_render.permute(0, 2, 3, 1), siren, None, fs_shader="vanilla")
            x_up = torch.nn.functional.interpolate(state, scale_factor=renderer.scale_factor, mode="bilinear")
            mask = model.get_living_mask(x_up).float().permute(0, 2, 3, 1)
            alpha = image[..., 3:4] * mask
            rgb = image[..., :3] * mask + (1.0 - alpha) * self.config["renderer"].get("background_color", 1.0)
            return Renderer2D.to_pil(rgb)

        if options.save_image:
            image = render_frame(x, z)
            image.save(output_dir / f"{self.config['experiment_name']}_test.png")

        if options.save_video:
            x = model.seed(1, grid_size[0], grid_size[1])
            with VideoWriter(str(output_dir / f"{self.config['experiment_name']}_test.mp4"), fps=options.fps) as video:
                for _ in tqdm(range(options.video_frames), desc="Test video"):
                    image = render_frame(x)
                    video.add(image)
                    x, _ = model(x)

