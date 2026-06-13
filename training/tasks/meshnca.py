import numpy as np
import torch
from tqdm import tqdm

from losses.loss import Loss
from models.meshnca import MeshNCA
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
from utils.camera import PerspectiveCamera
from utils.mesh import Mesh
from utils.misc import process_output_channels
from utils.render import Renderer3D
from utils.sphere_projection import SphereProjection
from utils.video import VideoWriter


class MeshNCATask(BaseTask):
    def _build(self, load: bool = False):
        meshnca_kwargs = {
            key: value
            for key, value in self.config["meshnca"].items()
            if key != "graft_initialization"
        }
        model = MeshNCA(**device_config(meshnca_kwargs, self.device)).to(self.device)
        total_channels, output_channels = process_output_channels(self.config["num_channels"])
        siren = Siren(in_features=model.channels, coord_dim=3, out_features=total_channels, **self.config["siren"]).to(self.device)
        if load:
            load_checkpoint_pair(self.config, model, siren, device=self.device)
        return model, siren, output_channels

    def train(self) -> None:
        set_seed(self.config.get("seed", 42))
        model, siren, output_channels = self._build()
        self._log_counts(model, siren, "MeshNCA")
        load_graft_if_configured(self.config, "meshnca", model, siren, self.device)
        with torch.no_grad():
            train_cfg = self.config["train"]
            icosphere = Mesh.load_icosphere(**device_config(train_cfg["icosphere"], self.device))
            pool = model.seed(train_cfg["pool_size"], icosphere.Nv)
            total_channels = sum(len(v) for v in output_channels.values())
            self.config["loss"]["appearance_loss_kwargs"]["total_channels"] = total_channels
            self.config["loss"]["appearance_loss_kwargs"]["output_channels"] = output_channels
            loss_fn = Loss(**self.config["loss"])
            renderer = Renderer3D(**self.config["renderer"])
            test_renderer = Renderer3D(**self.config["test_renderer"])
            camera_config = device_config(train_cfg["camera"], self.device)
            projection = SphereProjection(mesh=icosphere, **device_config(train_cfg["projection"], self.device))

        optimizer, scheduler = self._optimizer(list(model.parameters()) + list(siren.parameters()))
        for epoch in tqdm(range(train_cfg["epochs"]), desc="MeshNCA"):
            with torch.no_grad():
                batch_idx = np.random.choice(len(pool), train_cfg["batch_size"], replace=False)
                x = pool[batch_idx]
                if epoch % train_cfg["inject_seed_interval"] == 0:
                    x[:1] = model.seed(1, icosphere.Nv)
            step_n = np.random.randint(*train_cfg["step_range"])
            for _ in range(step_n):
                x = model(x, icosphere, None)
            camera = PerspectiveCamera.generate_random_view_cameras(**camera_config)
            rendered = renderer.render(icosphere, x, camera, projection, siren)
            rendered = torch.flatten(rendered, start_dim=0, end_dim=1).permute(0, 3, 1, 2)
            input_dict = {"rendered_images": rendered, "nca_state": x}
            return_summary = epoch % train_cfg["summary_interval"] == 0
            loss, loss_log, summary = loss_fn(input_dict, return_summary=return_summary)
            if return_summary:
                with torch.no_grad():
                    test_camera = PerspectiveCamera.generate_random_view_cameras(1, distance=2.5, device=self.device, k=1.0)
                    test_image = test_renderer.render(icosphere, x[:1], test_camera, None, siren, target_channels=output_channels)
                    self._save_logged_image("pbr_output", Renderer3D.to_pil(test_image), epoch)
                    if "appearance-images" in summary:
                        self._save_logged_image("rendered_images", summary["appearance-images"], epoch)
            loss.backward()
            with torch.no_grad():
                normalize_model_grads(model)
                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()
                pool[batch_idx] = x
            self.logger.log_metrics(loss_log, step=epoch)
        save_checkpoint(self.config, model, siren)

    @torch.no_grad()
    def test(self, options: TestOptions) -> None:
        model, siren, output_channels = self._build(load=True)
        output_dir = self._output_dir(options)
        mesh_config = device_config(self.config["train"].get("test_mesh", {}), self.device)
        if mesh_config.get("obj_path"):
            mesh = Mesh.load_from_obj(**mesh_config)
        else:
            mesh = Mesh.load_icosphere(subdivision_freq=2 ** 6, device=self.device)
        renderer = Renderer3D(**self.config["test_renderer"])
        x = model.seed(1, mesh.Nv)
        for _ in tqdm(range(options.steps), desc="Test rollout"):
            x = torch.clip(model(x, mesh, None), -1.0, 1.0)
        camera = PerspectiveCamera(elevation=[35.0], azimuth=[45.0], distance=[2.0], k=1.0, height=1024, width=1024, device=self.device)
        if options.save_image:
            image = renderer.render(mesh, x, camera, None, siren, target_channels=output_channels)
            Renderer3D.to_pil(image).save(output_dir / f"{self.config['experiment_name']}_test.png")
        if options.save_video:
            x = model.seed(1, mesh.Nv)
            with VideoWriter(str(output_dir / f"{self.config['experiment_name']}_test.mp4"), fps=options.fps) as video:
                for i in tqdm(range(options.video_frames), desc="Test video"):
                    image = renderer.render(mesh, x, camera, None, siren, target_channels=output_channels)
                    video.add(Renderer3D.to_pil(image))
                    step_n = 1 if i < options.video_frames // 4 else 4
                    for _ in range(step_n):
                        x = torch.clip(model(x, mesh, None), -1.0, 1.0)
                    camera.rotateY(1.0)

