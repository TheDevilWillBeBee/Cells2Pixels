import argparse
import copy
from pathlib import Path

from logger import get_logger_class
from training.common import TestOptions, ensure_experiment, load_yaml
from training.tasks import get_task
from training.tasks.targets import append_target_name_to_experiment


def parse_args():
    parser = argparse.ArgumentParser(description="Unified single-target NCA training script.")
    parser.add_argument("--config", type=str, required=True, help="Path to a YAML config.")
    parser.add_argument("--overwrite", action="store_true", help="Allow training into an existing completed experiment.")
    parser.add_argument("--skip-train", action="store_true", help="Only run --test using an existing checkpoint.")

    parser.add_argument("--test", action="store_true", help="Run the task's common test pass after training.")
    parser.add_argument("--test-output-dir", type=str, default="outputs", help="Directory for test images/videos.")
    parser.add_argument("--test-steps", type=int, default=512, help="NCA rollout steps before the test image.")
    parser.add_argument("--test-video-frames", type=int, default=240, help="Number of test video frames.")
    parser.add_argument("--test-fps", type=float, default=30.0, help="Test video frames per second.")
    parser.add_argument("--no-test-image", action="store_true", help="Do not save the test image.")
    parser.add_argument("--no-test-video", action="store_true", help="Do not save the test video.")
    return parser.parse_args()


def create_logger(config: dict):
    logger_cfg = dict(config.get("logger") or {})
    backend = logger_cfg.pop("backend", None) or "disabled"
    if backend == "local" and "run_dir" not in logger_cfg and "experiment_path" in config:
        logger_cfg["run_dir"] = config["experiment_path"]
    return get_logger_class(backend)(**logger_cfg)


def main():
    args = parse_args()
    config = copy.deepcopy(load_yaml(args.config))

    if "task" not in config:
        raise ValueError("Config must define a top-level 'task' field.")
    task_name = config["task"]
    print(f"Task: {task_name}")
    target_name = append_target_name_to_experiment(config)
    if target_name is not None:
        print(f"Target: {target_name}")

    if not args.skip_train:
        exp_path = ensure_experiment(config, args.config, overwrite=args.overwrite)
        print(f"Experiment path: {exp_path}")
    elif "experiment_path" not in config:
        config["experiment_path"] = str(Path("experiments") / config["experiment_name"].replace(" ", "_"))

    logger = create_logger(config)
    logger_started = False
    try:
        logger.start_run(config.get("experiment_name"))
        logger_started = True
        task = get_task(task_name, config, logger)

        if not args.skip_train:
            task.train()
            logger.log_artifact(str(Path(config["experiment_path"]) / "config.yaml"), "config.yaml")

        if args.test:
            task.test(
                TestOptions(
                    enabled=True,
                    output_dir=args.test_output_dir,
                    steps=args.test_steps,
                    video_frames=args.test_video_frames,
                    fps=args.test_fps,
                    save_image=not args.no_test_image,
                    save_video=not args.no_test_video,
                )
            )
    except Exception:
        if logger_started:
            logger.end_run("FAILED")
        raise
    else:
        if logger_started:
            logger.end_run("FINISHED")


if __name__ == "__main__":
    main()
