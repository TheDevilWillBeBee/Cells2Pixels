import copy

import numpy as np
import torch
from tqdm import tqdm

from losses.loss import Loss
from models.nca3d import GrowingVNCA
from models.siren import Siren
from training.common import (
    TestOptions,
    autocast_context,
    device_config,
    load_checkpoint_pair,
    load_graft_if_configured,
    make_grad_scaler,
    normalize_model_grads,
    precision_from_config,
    save_checkpoint,
    set_seed,
)
from training.tasks.base import BaseTask
from utils.camera import PerspectiveCamera
from utils.video import VideoWriter
from utils.volumetric_render import RendererRF


class GrowingRFTask(BaseTask):
    model_class = GrowingVNCA

    def _build(self, load: bool = False):
        precision = precision_from_config(self.config)
        nca_kwargs = device_config(self.config["nca"]["nca_kwargs"], self.device)
        nca_kwargs["precision"] = precision
        model = self.model_class(**nca_kwargs).to(self.device)
        nca_output_dim = model.channels
        if self.config["nca"]["output_type"] == "z":
            nca_output_dim *= model.perception_kernels
        sh_degree = self.config["renderer"].get("sh_degree", 0)
        output_dim = 1 + 3 * ((sh_degree + 1) ** 2)
        siren = Siren(in_features=nca_output_dim, coord_dim=3, out_features=output_dim, **self.config["siren"]).to(self.device)
        if load:
            load_checkpoint_pair(self.config, model, siren, device=self.device)
        return model, siren, precision

    def train(self) -> None:
        set_seed(self.config.get("seed", 42))
        model, siren, precision = self._build()
        self._log_counts(model, siren, "GrowingVNCA")
        load_graft_if_configured(self.config, "nca", model, siren, self.device)
        with torch.no_grad():
            loss_fn = Loss(**self.config["loss"])
            renderer = RendererRF(**self.config["renderer"])
            grid_size = self.config["nca"]["grid_size"]

        train_cfg = self.config["train"]
        only_siren = self.config.get("only_siren", False)
        pool_size = 1 if only_siren else train_cfg["pool_size"]
        batch_size = 1 if only_siren else train_cfg["batch_size"]
        accumulation_steps = (train_cfg["virtual_batch_size"] + batch_size - 1) // batch_size

        for repetition in range(train_cfg["num_repetitions"]):
            with torch.no_grad():
                pool = model.seed(pool_size, *grid_size)
                if only_siren:
                    x = pool[0:1]
                    z = None
                    with autocast_context(self.device, precision):
                        for _ in range(512):
                            x, z = model(x)

            parameters = list(siren.parameters()) if only_siren else list(model.parameters()) + list(siren.parameters())
            optimizer, scheduler = self._optimizer(parameters)
            scaler = make_grad_scaler(self.device, precision)
            accumulation_counter = 0
            for epoch in tqdm(range(train_cfg["epochs"] * accumulation_steps), desc=f"Repetition {repetition + 1}/{train_cfg['num_repetitions']}"):
                log_step = (epoch + repetition * train_cfg["epochs"] * accumulation_steps) // accumulation_steps
                if not only_siren:
                    with torch.no_grad():
                        batch_idx = np.random.choice(len(pool), batch_size, replace=False)
                        x = pool[batch_idx]
                        if log_step % train_cfg["inject_seed_interval"] == 0 and accumulation_counter == 0:
                            x[:1] = model.seed(1, *grid_size)
                    step_n = np.random.randint(*train_cfg["step_range"])
                    with autocast_context(self.device, precision):
                        for _ in range(step_n):
                            x, z = model(x)
                    pool[batch_idx] = x.detach()

                x_render = (x if self.config["nca"]["output_type"] == "s" else z).to(torch.float32)
                camera, view_idx = loss_fn["rf"].get_random_views(train_cfg["num_views"], mode="train")
                l1l2_sampler_kwargs = copy.copy(self.config["renderer"]["sampler_kwargs"])
                l1l2_sampler_kwargs["mode"] = "stride"
                rgb, _, opacity, alpha, sampler = renderer.render(x_render, camera, siren, sampler_kwargs=l1l2_sampler_kwargs)
                generated = torch.cat([rgb, opacity], dim=2)
                input_dict = {
                    "generated_images_l1l2": generated,
                    "alpha_l1l2": alpha,
                    "view_idx": view_idx,
                    "nca_state": x,
                    "sampler_l1l2": sampler,
                    "mode": "train",
                }
                if self.config["loss"]["rf_loss_kwargs"].get("lpips_weight", 0.0) > 0.0:
                    input_dict["generated_images_lpips"] = generated
                    input_dict["alpha_lpips"] = alpha
                    input_dict["sampler_lpips"] = sampler

                return_summary = log_step % train_cfg["summary_interval"] == 0 and accumulation_counter == 0
                loss, loss_log, summary = loss_fn(input_dict, return_summary=return_summary)
                accumulation_counter += 1

                if return_summary:
                    with torch.no_grad():
                        x_test = x_render[-1:]
                        camera_test = camera.sample_batch(1, [0])
                        rgb_test, depth_test, opacity_test, living_test, _ = renderer.render(
                            x_test,
                            camera_test,
                            siren,
                            num_samples=512,
                            num_fine_samples=512,
                            background_color=1.0,
                            perturb=False,
                            batchify_rays=True,
                            sampler_kwargs={},
                        )
                        self._save_logged_image(
                            "high_resolution_output",
                            renderer.to_pil((rgb_test, depth_test / 8.0, opacity_test, living_test)),
                            log_step,
                        )
                        for key in ("l1l2", "lpips"):
                            if f"rf-images_{key}" in summary:
                                self._save_logged_image(f"nca_output_{key}", summary[f"rf-images_{key}"], log_step)

                if precision == torch.float16:
                    scaler.scale(loss).backward()
                else:
                    loss.backward()

                if accumulation_counter == accumulation_steps:
                    with torch.no_grad():
                        if not only_siren:
                            normalize_model_grads(model)
                        if precision == torch.float16:
                            scaler.step(optimizer)
                            scaler.update()
                        else:
                            optimizer.step()
                        optimizer.zero_grad()
                        scheduler.step()
                    accumulation_counter = 0
                    self.logger.log_metrics(loss_log, step=log_step)

            save_checkpoint(self.config, model, siren, suffix=f"_{repetition + 1}")

        save_checkpoint(self.config, model, siren)

    @torch.no_grad()
    def test(self, options: TestOptions) -> None:
        test_config = copy.deepcopy(self.config)
        test_config["precision"] = "float32"
        self.config = test_config
        model, siren, _ = self._build(load=True)
        output_dir = self._output_dir(options)
        grid_size = self.config["nca"]["grid_size"]
        x = model.seed(1, *grid_size)
        camera = PerspectiveCamera(
            fov=60.0,
            elevation=[15.0],
            azimuth=[0.0],
            distance=[3.0],
            bounds=[1.0, 8.0],
            height=256,
            width=256,
            device=self.device,
        )
        renderer_cfg = copy.deepcopy(self.config["renderer"])
        renderer_cfg.update(
            {
                "num_samples": 256,
                "num_fine_samples": 256,
                "perturb": False,
                "background_color": 1.0,
                "sampler_kwargs": {"mode": "stride", "stride": 1},
            }
        )
        renderer = RendererRF(**renderer_cfg)

        if options.save_image:
            for _ in tqdm(range(options.steps), desc="Test rollout"):
                x, _ = model(x)
            rgb, depth, opacity, living, _ = renderer.render(x, camera, siren, batchify_rays=True)
            renderer.to_pil((rgb, depth / 8.0, opacity, living)).save(output_dir / f"{self.config['experiment_name']}_test.png")

        if options.save_video:
            x = model.seed(1, *grid_size)
            with VideoWriter(str(output_dir / f"{self.config['experiment_name']}_test.mp4"), fps=options.fps) as video:
                for _ in tqdm(range(options.video_frames), desc="Test video"):
                    rgb, depth, opacity, living, _ = renderer.render(x, camera, siren, batchify_rays=True)
                    video.add(renderer.to_pil((rgb, depth / 8.0, opacity, living)))
                    for _ in range(2):
                        x, _ = model(x)
                    camera.rotateY(1.0)

