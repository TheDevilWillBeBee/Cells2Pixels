import numpy as np
import torch
from tqdm import tqdm

from losses.loss import Loss
from models.nca2d import GrowingNCA, NCA, NoiseNCA, PENCA
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
from utils.misc import process_output_channels
from utils.render import Renderer2D
from utils.video import VideoWriter


NCA_2D_TYPES = {
    "NCA": NCA,
    "NoiseNCA": NoiseNCA,
    "PENCA": PENCA,
    "GrowingNCA": GrowingNCA,
}


class Texture2DTask(BaseTask):
    def _build(self, load: bool = False):
        nca_kwargs = device_config(self.config["nca"]["nca_kwargs"], self.device)
        nca_type = self.config["nca"]["type"]
        model = NCA_2D_TYPES[nca_type](**nca_kwargs).to(self.device)
        total_channels, output_channels = process_output_channels(self.config["num_channels"])
        nca_output_type = self.config["nca"].get("output_type", "s")
        nca_output_dim = model.channels
        if nca_output_type == "z":
            nca_output_dim *= model.perception_kernels
        siren = Siren(in_features=nca_output_dim, coord_dim=2, out_features=total_channels, **self.config["siren"]).to(self.device)
        if load:
            load_checkpoint_pair(self.config, model, siren, device=self.device)
        return model, siren, output_channels

    def _loss_renderer_grid(self, output_channels):
        total_channels = sum(len(v) for v in output_channels.values())
        self.config["loss"]["appearance_loss_kwargs"]["total_channels"] = total_channels
        self.config["loss"]["appearance_loss_kwargs"]["output_channels"] = output_channels
        loss_fn = Loss(**self.config["loss"])
        renderer = Renderer2D(**self.config["renderer"])
        return loss_fn, renderer, tuple(self.config["train"]["nca_grid_size"])

    def train(self) -> None:
        set_seed(self.config.get("seed", 42))
        model, siren, output_channels = self._build()
        self._log_counts(model, siren, self.config["nca"]["type"])
        rep_start = load_graft_if_configured(self.config, "nca", model, siren, self.device)
        with torch.no_grad():
            loss_fn, renderer, grid_size = self._loss_renderer_grid(output_channels)

        train_cfg = self.config["train"]
        num_reps = train_cfg.get("num_repetitions", 1)
        for repetition in range(rep_start, num_reps):
            with torch.no_grad():
                pool = model.seed(train_cfg["pool_size"], grid_size[0], grid_size[1])
            optimizer, scheduler = self._optimizer(list(model.parameters()) + list(siren.parameters()))

            for epoch in tqdm(range(train_cfg["epochs"]), desc=f"Repetition {repetition + 1}/{num_reps}"):
                log_step = epoch + repetition * train_cfg["epochs"]
                with torch.no_grad():
                    batch_idx = np.random.choice(len(pool), train_cfg["batch_size"], replace=False)
                    x = pool[batch_idx]
                    if log_step % train_cfg["inject_seed_interval"] == 0:
                        x[:1] = model.seed(1, grid_size[0], grid_size[1])

                step_n = np.random.randint(*train_cfg["step_range"])
                x0 = x
                for _ in range(step_n):
                    x, z = model(x)
                x_render = x if self.config["nca"].get("output_type", "s") == "s" else z
                rendered = renderer.render(x_render.permute(0, 2, 3, 1), siren, None, fs_shader="vanilla")
                rendered = rendered.permute(0, 3, 1, 2)

                of_channels = np.random.permutation(model.channels)[:3]
                input_dict = {
                    "rendered_images": rendered,
                    "nca_state": x,
                    "image_before": (x0[:, of_channels] + 1.0) / 2.0,
                    "image_after": (x[:, of_channels] + 1.0) / 2.0,
                    "step_n": step_n,
                }
                return_summary = log_step % train_cfg["summary_interval"] == 0
                loss, loss_log, summary = loss_fn(input_dict, return_summary=return_summary)

                if return_summary:
                    if train_cfg.get("pbr_training", False):
                        with torch.no_grad():
                            image = renderer.render(
                                x_render.permute(0, 2, 3, 1),
                                siren,
                                target_channels=output_channels,
                                fs_shader="pbr",
                            )
                            self._save_logged_image("pbr_output", Renderer2D.to_pil(image), log_step)
                    if "appearance-images" in summary:
                        self._save_logged_image("rendered_images", summary["appearance-images"], log_step)
                    if "motion-generated_VF" in summary:
                        self._save_logged_image("generated_vector_field", summary["motion-generated_VF"], log_step)
                    if log_step == 0 and "motion-target_VF" in summary:
                        self._save_logged_image("target_vector_field", summary["motion-target_VF"], log_step)

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
        model, siren, output_channels = self._build(load=True)
        _, renderer, grid_size = self._loss_renderer_grid(output_channels)
        output_dir = self._output_dir(options)
        x = model.seed(1, grid_size[0], grid_size[1])
        z = None
        for _ in tqdm(range(options.steps), desc="Test rollout"):
            x, z = model(x)

        def render_frame(state, perception=None):
            x_render = state if self.config["nca"].get("output_type", "s") == "s" or perception is None else perception
            fs_shader = "pbr" if self.config["train"].get("pbr_training", False) else self.config["renderer"].get("fs_shader", "vanilla")
            image = renderer.render(x_render.permute(0, 2, 3, 1), siren, target_channels=output_channels, fs_shader=fs_shader)
            return Renderer2D.to_pil(image, target_channels=output_channels if fs_shader == "vanilla" else None)

        if options.save_image:
            render_frame(x, z).save(output_dir / f"{self.config['experiment_name']}_test.png")

        if options.save_video:
            x = model.seed(1, grid_size[0], grid_size[1])
            with VideoWriter(str(output_dir / f"{self.config['experiment_name']}_test.mp4"), fps=options.fps) as video:
                for i in tqdm(range(options.video_frames), desc="Test video"):
                    step_n = 1 if i < options.video_frames // 3 else 4
                    image = render_frame(x)
                    video.add(image)
                    for _ in range(step_n):
                        x, _ = model(x)

