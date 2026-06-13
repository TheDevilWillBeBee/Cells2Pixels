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


class GrowingVoxelTask(BaseTask):
    def _build(self, load: bool = False):
        precision = precision_from_config(self.config)
        nca_kwargs = device_config(self.config["nca"]["nca_kwargs"], self.device)
        nca_kwargs["precision"] = precision
        model = GrowingVNCA(**nca_kwargs).to(self.device)
        nca_output_dim = model.channels
        if self.config["nca"]["output_type"] == "z":
            nca_output_dim *= model.perception_kernels
        siren = Siren(in_features=nca_output_dim, coord_dim=3, out_features=4, **self.config["siren"]).to(self.device)
        if load:
            load_checkpoint_pair(self.config, model, siren, device=self.device)
        return model, siren, precision

    @staticmethod
    def _mgrid(length: int, dim: int, device):
        tensors = tuple(dim * [(torch.arange(length, device=device) / length - 0.5 + 0.5 / length) * 2.0])
        return torch.stack(torch.meshgrid(*tensors), dim=-1)

    def _generated_voxel(self, model, siren, x, z, nca_grid_size, scale_factor, precision):
        x_render = x if self.config["nca"]["output_type"] == "s" else z
        x_render = x_render.to(torch.float32)
        batch_size = x_render.shape[0]
        coords = self._mgrid(scale_factor, 3, x.device)
        coords = coords[None, ...].expand(batch_size, -1, -1, -1, -1)
        coords = coords.repeat(1, *nca_grid_size, 1)
        x_pad = torch.nn.functional.pad(x_render, [1, 1, 1, 1, 1, 1], "circular")
        x_up = torch.nn.functional.interpolate(x_pad, scale_factor=scale_factor, mode="trilinear")
        x_up = x_up[:, :, scale_factor:-scale_factor, scale_factor:-scale_factor, scale_factor:-scale_factor]
        living_mask = model.get_living_mask(x_up).float().permute(0, 2, 3, 4, 1)
        x_up = x_up.permute(0, 2, 3, 4, 1)
        with autocast_context(x_up.device, precision):
            generated = siren(x_up, coords)
        return generated * living_mask, x_up

    def train(self) -> None:
        set_seed(self.config.get("seed", 42))
        model, siren, precision = self._build()
        self._log_counts(model, siren, "GrowingVNCA")
        load_graft_if_configured(self.config, "nca", model, siren, self.device)
        with torch.no_grad():
            loss_fn = Loss(**self.config["loss"])
            grid_size = loss_fn.loss_mapper["voxel"].grid_size
            scale_factor = self.config["scale_factor"]
            nca_grid_size = [s // scale_factor for s in grid_size]

        train_cfg = self.config["train"]
        accumulation_steps = (train_cfg["virtual_batch_size"] + train_cfg["batch_size"] - 1) // train_cfg["batch_size"]
        for repetition in range(train_cfg["num_repetitions"]):
            with torch.no_grad():
                pool = model.seed(train_cfg["pool_size"], *nca_grid_size)
            inject_interval = train_cfg["inject_seed_interval"] * (2 ** repetition)
            optimizer, scheduler = self._optimizer(list(model.parameters()) + list(siren.parameters()))
            scaler = make_grad_scaler(self.device, precision)
            accumulation_counter = 0
            for epoch in tqdm(range(train_cfg["epochs"] * accumulation_steps), desc=f"Repetition {repetition + 1}/{train_cfg['num_repetitions']}"):
                log_step = (epoch + repetition * train_cfg["epochs"] * accumulation_steps) // accumulation_steps
                with torch.no_grad():
                    batch_idx = np.random.choice(len(pool), train_cfg["batch_size"], replace=False)
                    x = pool[batch_idx]
                    if log_step % inject_interval == 0 and accumulation_counter == 0:
                        x[:1] = model.seed(1, *nca_grid_size)
                step_n = np.random.randint(*train_cfg["step_range"])
                with autocast_context(self.device, precision):
                    for _ in range(step_n):
                        x, z = model(x)
                pool[batch_idx] = x.detach()
                generated, x_up = self._generated_voxel(model, siren, x, z, nca_grid_size, scale_factor, precision)
                input_dict = {"generated_voxel": generated, "nca_state": x, "alpha": x_up[..., 3:4]}
                return_summary = log_step % train_cfg["summary_interval"] == 0 and accumulation_counter == 0
                loss, loss_log, summary = loss_fn(input_dict, return_summary=return_summary)
                accumulation_counter += 1
                if return_summary and "voxel-slice_images" in summary:
                    self._save_logged_image("voxel_slices", summary["voxel-slice_images"], log_step)
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
                    accumulation_counter = 0
                    self.logger.log_metrics(loss_log, step=log_step)
            save_checkpoint(self.config, model, siren, suffix=f"_{repetition + 1}")
        save_checkpoint(self.config, model, siren)

    @torch.no_grad()
    def test(self, options: TestOptions) -> None:
        model, siren, precision = self._build(load=True)
        loss_fn = Loss(**self.config["loss"])
        grid_size = loss_fn.loss_mapper["voxel"].grid_size
        scale_factor = self.config["scale_factor"]
        nca_grid_size = [s // scale_factor for s in grid_size]
        output_dir = self._output_dir(options)
        x = model.seed(1, *nca_grid_size)
        z = None
        for _ in tqdm(range(options.steps), desc="Test rollout"):
            x, z = model(x)
        generated, x_up = self._generated_voxel(model, siren, x, z, nca_grid_size, scale_factor, precision)
        _, _, summary = loss_fn({"generated_voxel": generated, "nca_state": x, "alpha": x_up[..., 3:4]}, return_summary=True)
        if "voxel-slice_images" in summary:
            summary["voxel-slice_images"].save(output_dir / f"{self.config['experiment_name']}_test.png")

