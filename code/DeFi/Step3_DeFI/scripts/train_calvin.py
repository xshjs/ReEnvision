import argparse
import logging
import math
from pathlib import Path
import sys
import os
from hydra import compose, initialize
from datetime import datetime
from time import time
# This is for using the locally installed repo clone when using slurm
sys.path.insert(0, Path(__file__).absolute().parents[1].as_posix())
import hydra
from omegaconf import DictConfig, ListConfig, OmegaConf, open_dict
from accelerate import Accelerator
import torch
from glob import glob
from copy import deepcopy
from collections import OrderedDict
from policy_models.utils.utils import (
    get_git_commit_hash,
    get_last_checkpoint,
    initialize_pretrained_weights,
    print_system_env_info,
)


#################################################################################
#                             Training Helper Functions                         #
#################################################################################


@torch.no_grad()
def update_ema(ema_model, model, decay=0.9999):
    """
    Step the EMA model towards the current model.
    """
    ema_params = OrderedDict(ema_model.named_parameters())
    model_params = OrderedDict(model.named_parameters())

    for name, param in model_params.items():
        name = name.replace("module.", "")
        # Consider applying only to params that require_grad to avoid small numerical changes of pos_embed
        ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)


def requires_grad(model, flag=True):
    """
    Set requires_grad flag for all parameters in a model.
    """
    for p in model.parameters():
        p.requires_grad = flag


def create_logger(logging_dir):
    """
    Create a logger that writes to a log file and stdout.
    """
    logging.basicConfig(
        level=logging.INFO,
        format='[\033[34m%(asctime)s\033[0m] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[logging.StreamHandler(), logging.FileHandler(f"{logging_dir}/log.txt")]
    )
    logger = logging.getLogger(__name__)
    return logger


@torch.no_grad()
def run_validation(accelerator, model, val_loader, device, max_batches=None):
    model.eval()

    totals = {
        "loss_sum": 0.0,
        "first_mae_sum": 0.0,
        "chunk_mae_sum": 0.0,
        "xyz_mae_sum": 0.0,
        "rot_mae_sum": 0.0,
        "gripper_mae_sum": 0.0,
        "num_batches": 0,
        "num_samples": 0,
    }

    for batch_idx, data_batch in enumerate(val_loader):
        if max_batches is not None and batch_idx >= max_batches:
            break

        with accelerator.autocast():
            loss = model(data_batch)

        pred_actions = model.eval_forward(
            {"rgb_obs": data_batch["rgb_obs"]},
            {"lang_text": data_batch["lang_text"]},
        )
        gt_actions = data_batch["actions"]

        if gt_actions.dim() == 2:
            gt_actions = gt_actions.unsqueeze(1)
        if pred_actions.dim() == 2:
            pred_actions = pred_actions.unsqueeze(1)

        seq_len = min(pred_actions.shape[1], gt_actions.shape[1])
        pred_actions = pred_actions[:, :seq_len]
        gt_actions = gt_actions[:, :seq_len]

        abs_err = (pred_actions - gt_actions).abs()
        first_abs_err = abs_err[:, 0]
        batch_size = gt_actions.shape[0]

        totals["loss_sum"] += float(loss.detach().item()) * batch_size
        totals["first_mae_sum"] += float(first_abs_err.mean().item()) * batch_size
        totals["chunk_mae_sum"] += float(abs_err.mean().item()) * batch_size
        totals["xyz_mae_sum"] += float(first_abs_err[:, :3].mean().item()) * batch_size
        totals["rot_mae_sum"] += float(first_abs_err[:, 3:6].mean().item()) * batch_size
        totals["gripper_mae_sum"] += float(first_abs_err[:, 6].mean().item()) * batch_size
        totals["num_batches"] += 1
        totals["num_samples"] += batch_size

    model.train()

    if totals["num_samples"] == 0:
        return None

    reduced = {}
    for key, value in totals.items():
        tensor = torch.tensor(value, device=device, dtype=torch.float64)
        reduced[key] = accelerator.gather_for_metrics(tensor.unsqueeze(0)).sum().item()

    denom = max(int(reduced["num_samples"]), 1)
    return {
        "val_loss": reduced["loss_sum"] / denom,
        "val_first_mae": reduced["first_mae_sum"] / denom,
        "val_chunk_mae": reduced["chunk_mae_sum"] / denom,
        "val_xyz_mae": reduced["xyz_mae_sum"] / denom,
        "val_rot_mae": reduced["rot_mae_sum"] / denom,
        "val_gripper_mae": reduced["gripper_mae_sum"] / denom,
        "val_num_batches": int(reduced["num_batches"]),
        "val_num_samples": int(reduced["num_samples"]),
    }


#################################################################################
#                                  Training Loop                                #
#################################################################################


def train(cfg: DictConfig) -> None:
    accelerator = Accelerator()
    assert torch.cuda.is_available(), "Training currently requires at least one GPU."
    device = accelerator.device
    # new added
    torch.set_float32_matmul_precision('medium')

    if accelerator.is_main_process:
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        experiment_dir = f"../outputs/calvin_train/{timestamp}"
        checkpoint_dir = f"{experiment_dir}/saved_models"
        os.makedirs(checkpoint_dir, exist_ok=True)
        logger = create_logger(experiment_dir)
        logger.info(f"Experiment directory created at {experiment_dir}")
        logger.info(f"Training with the following config:\n{OmegaConf.to_yaml(cfg)}")

    datamodule = hydra.utils.instantiate(cfg.datamodule)
    datamodule.setup()
    if accelerator.is_main_process:
        logger.info(f"Global batch size {cfg.batch_size:,} num_processes ({accelerator.num_processes})")
    train_loader = datamodule.train_dataloader()["lang"]
    val_loader = datamodule.val_dataloader()["lang"]
    
    # Load Model
    model = hydra.utils.instantiate(cfg.model)
    
    if cfg.ckpt_path:
        state_dict = torch.load(cfg.ckpt_path, map_location='cpu')
        if isinstance(state_dict, dict):
            if "state_dict" in state_dict:
                ckpt_weights = state_dict["state_dict"]
            elif "model" in state_dict:
                ckpt_weights = state_dict["model"]
            else:
                ckpt_weights = state_dict
        else:
            ckpt_weights = state_dict
        load_result = model.load_state_dict(ckpt_weights, strict=False)
        model_keys = set(model.state_dict().keys())
        ckpt_keys = set(ckpt_weights.keys())
        matched_keys = (model_keys & ckpt_keys) - set(load_result.missing_keys)

        if accelerator.is_main_process:
            print("✅ Successfully loaded parameters.")
            if matched_keys:
                print("✅ Matched keys (these parameters were loaded successfully):")
                for k in sorted(matched_keys):
                    print(f"   - {k}")
            if load_result.missing_keys:
                print("\n⚠️ Missing keys (in model but not in checkpoint):")
                for k in load_result.missing_keys:
                    print(f"   - {k}")
            if load_result.unexpected_keys:
                print("\n⚠️ Unexpected keys (in checkpoint but not in model):")
                for k in load_result.unexpected_keys:
                    print(f"   - {k}")
            if not load_result.missing_keys and not load_result.unexpected_keys:
                print("\n🎉 All keys matched perfectly.")

    model = model.to(device)
    model.process_device()

    if accelerator.is_main_process:
        logger.info(f"DiT Parameters: {sum(p.numel() for p in model.parameters()):,}")

    opt = model.configure_optimizers()["optimizer"]
    Ir_scheduler = model.configure_optimizers()["lr_scheduler"]["scheduler"]

    model.on_train_start()
    if accelerator.is_main_process:
        logger.info(f"model parameter init")
    ema = deepcopy(model).to(device)  # Create an EMA of the model for use after training
    requires_grad(ema, False)
    update_ema(ema, model, decay=0)  # Ensure EMA is initialized with synced weights
    ema.eval()
    model.train()
    model, opt, loader, val_loader = accelerator.prepare(model, opt, train_loader, val_loader)

    train_steps = 0
    log_steps = 0
    running_loss = 0
    start_time = time()
    best_val_first_mae = math.inf
    max_steps = int(cfg.get("max_steps", 0) or 0)

    if accelerator.is_main_process:
        logger.info(f"Training for {cfg.max_epochs} epochs...")
        if max_steps > 0:
            logger.info(f"Stopping after {max_steps} optimizer steps.")

    for epoch in range(cfg.max_epochs):
        if accelerator.is_main_process:
            logger.info(f"Beginning epoch {epoch}...")
        running_loss = 0

        for idx,data_batch in enumerate(loader):
            with accelerator.autocast():
                loss = model(data_batch)
            opt.zero_grad()
            accelerator.backward(loss)
            opt.step()
            Ir_scheduler.step()
            update_ema(ema, model)
            running_loss += loss
            log_steps += 1
            train_steps += 1
            if train_steps % cfg.log_every == 0:
                torch.cuda.synchronize()
                end_time = time()
                steps_per_sec = log_steps / (end_time - start_time)
                # Reduce loss history over all processes:
                avg_loss = torch.tensor(running_loss / log_steps, device=device)
                avg_loss = avg_loss.item()

                if accelerator.is_main_process:
                    logger.info(
                        f"(step={train_steps:07d}) Train total Loss : {avg_loss:.6f}, Train Steps/Sec: {steps_per_sec:.2f}")
                
                # Reset monitoring variables:
                running_loss = 0
                log_steps = 0
                start_time = time()

            if max_steps > 0 and train_steps >= max_steps:
                if accelerator.is_main_process:
                    logger.info(f"Reached max_steps={max_steps}; saving checkpoint.")
                break

        val_metrics = run_validation(
            accelerator=accelerator,
            model=model,
            val_loader=val_loader,
            device=device,
            max_batches=cfg.get("val_num_batches"),
        )
        accelerator.wait_for_everyone()

        if accelerator.is_main_process:
            log_steps = 0
            checkpoint = {
                "model": model.module.state_dict() if accelerator.num_processes > 1 else model.state_dict(),
                "args": cfg,
                "epoch": epoch,
                "train_steps": train_steps,
            }

            if val_metrics is not None:
                checkpoint["val_metrics"] = val_metrics
                logger.info(
                    "Validation: "
                    f"loss={val_metrics['val_loss']:.6f}, "
                    f"first_mae={val_metrics['val_first_mae']:.6f}, "
                    f"chunk_mae={val_metrics['val_chunk_mae']:.6f}, "
                    f"xyz_mae={val_metrics['val_xyz_mae']:.6f}, "
                    f"rot_mae={val_metrics['val_rot_mae']:.6f}, "
                    f"gripper_mae={val_metrics['val_gripper_mae']:.6f}, "
                    f"num_batches={val_metrics['val_num_batches']}"
                )

            last_path = f"{checkpoint_dir}/last.pt"
            torch.save(checkpoint, last_path)

            epoch_path = f"{checkpoint_dir}/epoch_{epoch:03d}.pt"
            torch.save(checkpoint, epoch_path)

            checkpoint_path = f"{checkpoint_dir}/{train_steps:07d}.pt"
            torch.save(checkpoint, checkpoint_path)

            if val_metrics is not None and val_metrics["val_first_mae"] < best_val_first_mae:
                best_val_first_mae = val_metrics["val_first_mae"]
                best_path = f"{checkpoint_dir}/best_val.pt"
                torch.save(checkpoint, best_path)
                logger.info(
                    f"Saved new best validation checkpoint: {best_path} "
                    f"(val_first_mae={best_val_first_mae:.6f})"
                )

        if max_steps > 0 and train_steps >= max_steps:
            break


if __name__ == "__main__":
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    os.environ['TORCH_USE_CUDA_DSA'] = "1"
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["TOKENIZERS_PARALLELISM"] = 'True'
    os.environ['HYDRA_FULL_ERROR'] = '1'

    print(torch.cuda.is_available())
    print(torch.cuda.device_count())
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_model_path", type=str, default="")
    parser.add_argument("--text_encoder_path", type=str, default="")
    parser.add_argument("--root_data_dir", type=str, default="")
    parser.add_argument("--token_ckpt_path", type=str, default="")
    parser.add_argument("--t5_model_path", type=str, default="")
    parser.add_argument("--language_goal_path", type=str, default="")
    parser.add_argument("--batch_size", type=int, default=28)
    parser.add_argument("--max_epochs", type=int, default=None)
    parser.add_argument("--max_steps", type=int, default=None)
    parser.add_argument("--num_workers", type=int, default=None)
    parser.add_argument("--save_every", type=int, default=2000)
    parser.add_argument("--val_num_batches", type=int, default=20)
    parser.add_argument("--config_name", type=str, default="VPP_Calvinabc_train")
    
    args = parser.parse_args()
    
    with initialize(config_path="../policy_conf", job_name=args.config_name):
        cfg = compose(config_name=args.config_name)
    cfg.model.pretrained_model_path = args.video_model_path
    cfg.model.text_encoder_path = args.text_encoder_path
    cfg.root_data_dir = args.root_data_dir
    cfg.datamodule.root_data_dir = args.root_data_dir
    cfg.batch_size = args.batch_size
    if args.num_workers is not None:
        cfg.num_workers = args.num_workers
        cfg.datamodule.datasets.lang_dataset.num_workers = args.num_workers
    if args.max_epochs is not None:
        cfg.max_epochs = args.max_epochs
    with open_dict(cfg):
        cfg.val_num_batches = args.val_num_batches
        cfg.save_every = args.save_every
        if args.max_steps is not None:
            cfg.max_steps = args.max_steps
    if args.token_ckpt_path:
        cfg.ckpt_path = args.token_ckpt_path
        print(f"[INFO] Overriding cfg.ckpt_path with args.token_ckpt_path: {cfg.ckpt_path}")
    if args.t5_model_path:
        cfg.model.t5_model_path = args.t5_model_path
        print(f"[INFO] Overriding cfg.model.t5_model_path with args.t5_model_path: {cfg.model.t5_model_path}")
    if args.language_goal_path:
        cfg.model.language_goal_path = args.language_goal_path
        print(
            f"[INFO] Overriding cfg.model.language_goal_path with args.language_goal_path: {cfg.model.language_goal_path}"
        )
    train(cfg)
