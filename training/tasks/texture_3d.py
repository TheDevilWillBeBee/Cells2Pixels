import copy

import numpy as np
import torch
from tqdm import tqdm

from losses.loss import Loss
from models.nca3d import VNCA
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
from utils.misc import process_output_channels
from utils.video import VideoWriter
from utils.volumetric_render import RendererRF


class Texture3DTask(BaseTask):
    def _build(self, load: bool = False):
        precision = precision_from_config(self.config)
        nca_kwargs = device_config(self.config["nca"]["nca_kwargs"], self.device)
        nca_kwargs["precision"] = precision
        model = VNCA(**nca_kwargs).to(self.device)
        total_channels, output_channels = process_output_channels(self.config["num_channels"])
        nca_output_dim = model.channels
        if self.config["nca"]["output_type"] == "z":
            nca_output_dim *= model.perception_kernels
        sh_degree = self.config["renderer"].get("sh_degree", 0)
        output_dim = 1 + total_channels * ((sh_degree + 1) ** 2)
        siren = Siren(in_features=nca_output_dim, coord_dim=3, out_features=output_dim, **self.config["siren"]).to(self.device)
        if load:
            load_checkpoint_pair(self.config, model, siren, device=self.device)
        return model, siren, precision, output_channels

    def train(self) -> None:
        set_seed(self.config.get("seed", 42))
        model, siren, precision, output_channels = self._build()
        self._log_counts(model, siren, "VNCA")
        load_graft_if_configured(self.config, "nca", model, siren, self.device)
        total_channels = sum(len(v) for v in output_channels.values())
        self.config["loss"]["appearance_loss_kwargs"]["total_channels"] = total_channels
        self.config["loss"]["appearance_loss_kwargs"]["output_channels"] = output_channels
        with torch.no_grad():
            loss_fn = Loss(**self.config["loss"])
            renderer = RendererRF(**self.config["renderer"])
            grid_size = self.config["nca"]["grid_size"]

        train_cfg = self.config["train"]
        batch_size = train_cfg["batch_size"]
        accumulation_steps = (train_cfg["virtual_batch_size"] + batch_size - 1) // batch_size

        for repetition in range(train_cfg["num_repetitions"]):
            with torch.no_grad():
                pool = model.seed(train_cfg["pool_size"], *grid_size)

            parameters = list(model.parameters()) + list(siren.parameters())
            optimizer, scheduler = self._optimizer(parameters)
            scaler = make_grad_scaler(self.device, precision)
            accumulation_counter = 0
            for epoch in tqdm(range(train_cfg["epochs"] * accumulation_steps), desc=f"Repetition {repetition + 1}/{train_cfg['num_repetitions']}"):
                log_step = (epoch + repetition * train_cfg["epochs"] * accumulation_steps) // accumulation_steps
                virtual_epoch = epoch // accumulation_steps
                with torch.no_grad():
                    batch_idx = np.random.choice(len(pool), batch_size, replace=False)
                    x = pool[batch_idx]
                    if virtual_epoch % train_cfg["inject_seed_interval"] == 0 and accumulation_counter == 0:
                        x[:1] = model.seed(1, *grid_size)

                step_n = np.random.randint(*train_cfg["step_range"])
                x0 = None
                z0 = None
                with autocast_context(self.device, precision):
                    for st in range(step_n):
                        x, z = model(x)
                        if st == 0:
                            x0 = x.clone().detach()
                            z0 = z.clone().detach() if z is not None else None
                pool[batch_idx] = x.detach()

                x_render = (x if self.config["nca"]["output_type"] == "s" else z).to(torch.float32)
                x0_render = (x0 if self.config["nca"]["output_type"] == "s" else z0).to(torch.float32)
                camera = PerspectiveCamera.generate_random_view_cameras(**device_config(train_cfg["camera"], self.device))
                rgb_sampler_kwargs = copy.copy(self.config["renderer"]["sampler_kwargs"])
                of_sampler_kwargs = copy.copy(self.config["renderer"]["sampler_kwargs"])
                rgb_sampler_kwargs.update({"mode": "stride", "stride": 1})
                of_sampler_kwargs.update({"mode": "stride", "stride": 8})

                rgb, depth, opacity, alpha, sampler = renderer.render(x_render, camera, siren, sampler_kwargs=rgb_sampler_kwargs)
                rgb = rgb.view(-1, rgb.shape[2], rgb.shape[3], rgb.shape[4])
                with torch.no_grad():
                    of_before, _, _, _, _ = renderer.render(x0_render, camera, siren, sampler_kwargs=of_sampler_kwargs, num_samples=64)
                    of_before = of_before.view(-1, of_before.shape[2], of_before.shape[3], of_before.shape[4])
                of_after, _, _, _, _ = renderer.render(x_render, camera, siren, sampler_kwargs=of_sampler_kwargs, num_samples=64)
                of_after = of_after.view(-1, of_after.shape[2], of_after.shape[3], of_after.shape[4])

                input_dict = {
                    "rendered_images": rgb,
                    "nca_state": x,
                    "image_before": of_before,
                    "image_after": of_after,
                    "step_n": step_n,
                }
                return_summary = log_step % train_cfg["summary_interval"] == 0 and accumulation_counter == 0
                loss, loss_log, summary = loss_fn(input_dict, return_summary=return_summary)
                accumulation_counter += 1

                if return_summary:
                    with torch.no_grad():
                        x_test = x_render[-1:]
                        camera_test = camera.sample_batch(1, [0])
                        rgb_test, depth_test, opacity_test, _, _ = renderer.render(
                            x_test,
                            camera_test,
                            siren,
                            num_samples=512,
                            num_fine_samples=0,
                            perturb=False,
                            batchify_rays=True,
                            sampler_kwargs={},
                        )
                        self._save_logged_image("high_resolution_output", renderer.to_pil((rgb_test, depth_test / 8.0, opacity_test, None)), log_step)

                if precision == torch.float16:
                    scaler.scale(loss).backward()
                else:
                    loss.backward()
                if accumulation_counter == accumulation_steps:
                    with torch.no_grad():
                        normalize_model_grads(model)
                        if precision == torch.float16:
                            scaler.step(optimizer)
                            scaler.update()
                        else:
                            optimizer.step()
                        optimizer.zero_grad()
                        scheduler.step()
                        loss_fn.update_loss_weights(loss_log, log_step)
                    accumulation_counter = 0
                    self.logger.log_metrics(loss_log, step=log_step)

            save_checkpoint(self.config, model, siren, suffix=f"_{repetition + 1}")
        save_checkpoint(self.config, model, siren)

    @torch.no_grad()
    def test(self, options: TestOptions) -> None:
        test_config = copy.deepcopy(self.config)
        test_config["precision"] = "float32"
        self.config = test_config
        model, siren, _, _ = self._build(load=True)
        output_dir = self._output_dir(options)
        grid_size = self.config["nca"]["grid_size"]
        x = model.seed(1, *grid_size)
        camera = PerspectiveCamera(
            fov=60.0,
            elevation=[0.0],
            azimuth=[0.0],
            distance=[2.5],
            bounds=[0.1, 100.0],
            height=256,
            width=256,
            k=1.0,
            device=self.device,
        )
        renderer_cfg = copy.deepcopy(self.config["renderer"])
        renderer_cfg.update(
            {
                "num_samples": 256,
                "num_fine_samples": 0,
                "perturb": False,
                "background_color": 1.0,
                "sampler_kwargs": {"mode": "stride", "stride": 1},
            }
        )
        renderer = RendererRF(**renderer_cfg)
        if options.save_image:
            for _ in tqdm(range(options.steps), desc="Test rollout"):
                x, _ = model(x)
            rgb, depth, opacity, _, _ = renderer.render(x, camera, siren, batchify_rays=True)
            renderer.to_pil((rgb, depth / 8.0, opacity, None)).save(output_dir / f"{self.config['experiment_name']}_test.png")
        if options.save_video:
            x = model.seed(1, *grid_size)
            with VideoWriter(str(output_dir / f"{self.config['experiment_name']}_test.mp4"), fps=options.fps) as video:
                for i in tqdm(range(options.video_frames), desc="Test video"):
                    rgb, depth, opacity, _, _ = renderer.render(x, camera, siren, batchify_rays=True)
                    video.add(renderer.to_pil((rgb, depth / 8.0, opacity, None)))
                    for _ in range(min(4, 2 ** (i // 15))):
                        x, _ = model(x)
                    camera.rotateY(1.0)

