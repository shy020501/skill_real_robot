import os
import random
import time
import hydra
import wandb
from hydra.utils import instantiate
from omegaconf import OmegaConf
from tqdm import tqdm
from pathlib import Path
import warnings

import numpy as np
import torch
import torch.nn as nn
import quest.utils.utils as utils
from pyinstrument import Profiler
from quest.utils.logger import Logger
import gc

OmegaConf.register_new_resolver("eval", eval, replace=True)


def seed_everything(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    random.seed(worker_seed)
    np.random.seed(worker_seed)


@hydra.main(config_path="config", version_base=None)
def main(cfg):
    device = cfg.device
    seed = cfg.seed
    seed_everything(seed)
    train_cfg = cfg.training

    # create model
    model = instantiate(cfg.algo.policy,
                        shape_meta=cfg.task.shape_meta)
    model.to(device)
    model.train()

    # start training
    optimizers = model.get_optimizers()
    schedulers = model.get_schedulers(optimizers)

    scaler = torch.cuda.amp.GradScaler(enabled=train_cfg.use_amp)

    experiment_dir, experiment_name = utils.get_experiment_dir(cfg)
    os.makedirs(experiment_dir, exist_ok=True)

    start_epoch, steps, wandb_id = 0, 0, None
    if train_cfg.auto_continue:
        checkpoint_path = experiment_dir.rsplit('/', 1)[0] + f'/stage_{cfg.stage - 1}'
        if 'libero' in checkpoint_path and cfg.stage == 2:
            checkpoint_path = checkpoint_path.replace('10', '90') # since we want to initialize the model from the libero_90 benchmark
    elif train_cfg.resume and len(os.listdir(experiment_dir)) > 0: 
        checkpoint_path = experiment_dir
    else: 
        checkpoint_path = cfg.checkpoint_path
    
    if checkpoint_path is not None:
        checkpoint_path = utils.get_latest_checkpoint(checkpoint_path)
        print(f'loading from checkpoint {checkpoint_path}')
        state_dict = utils.load_state(checkpoint_path, map_location="cpu")
        loaded_state_dict = state_dict['model']
        
        # Below line allows loading state dicts with some mismatched parameters
        utils.soft_load_state_dict(model, loaded_state_dict)

        # resuming training since we are loading a checkpoint training the same stage
        if cfg.stage == state_dict['stage']:
            print('loading from checkpoint')
            for optimizer, opt_state_dict in zip(optimizers, state_dict['optimizers']):
                optimizer.load_state_dict(opt_state_dict)
                for state in optimizer.state.values():
                    for k, v in state.items():
                        if torch.is_tensor(v):
                            state[k] = v.to(device)
            for scheduler, sch_state_dict in zip(schedulers, state_dict['schedulers']):
                scheduler.load_state_dict(sch_state_dict)
            scaler.load_state_dict(state_dict['scaler'])
            start_epoch = state_dict['epoch']
            steps = state_dict['steps']
            wandb_id = state_dict['wandb_id']
    else:
        print('starting from scratch')

    dataset = instantiate(cfg.task.dataset)
    model.preprocess_dataset(dataset, use_tqdm=train_cfg.use_tqdm)
    dataloader_generator = torch.Generator()
    dataloader_generator.manual_seed(seed)
    train_dataloader = instantiate(
        cfg.train_dataloader, 
        dataset=dataset,
        worker_init_fn=seed_worker,
        generator=dataloader_generator)

    print('Saving to:', experiment_dir)
    print('Experiment name:', experiment_name)

    wandb.init(
        dir=experiment_dir,
        name=experiment_name,
        config=OmegaConf.to_container(cfg, resolve=True),
        id=wandb_id,
        **cfg.logging
    )

    logger = Logger(train_cfg.log_interval)

    print('Training...')

    for epoch in range(start_epoch, train_cfg.n_epochs + 1):
        t0 = time.time()
        model.train()
        training_loss = 0.0
        if train_cfg.do_profile:
            profiler = Profiler()
            profiler.start()
        for idx, data in enumerate(tqdm(train_dataloader, disable=not train_cfg.use_tqdm)):
            data = utils.map_tensor_to_device(data, device)
            
            for optimizer in optimizers:
                optimizer.zero_grad()

            with torch.autograd.set_detect_anomaly(False):
                with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=train_cfg.use_amp):
                    loss, info = model.compute_loss(data)
            
                scaler.scale(loss).backward()
            
            for optimizer in optimizers:
                scaler.unscale_(optimizer)
            if train_cfg.grad_clip is not None:
                grad_norm = nn.utils.clip_grad_norm_(
                    model.parameters(), train_cfg.grad_clip
                )

            for optimizer in optimizers:
                scaler.step(optimizer)
            
            scaler.update()

            info.update({
                'epoch': epoch
            })
            if train_cfg.grad_clip is not None:
                info.update({
                    "grad_norm": grad_norm.item(),
                })  
            info = {cfg.logging_folder: info}
            training_loss += loss.item()
            steps += 1
            logger.update(info, steps)

            if train_cfg.cut and idx > train_cfg.cut:
                break

        if train_cfg.do_profile:
            profiler.stop()
            profiler.print()

        training_loss /= len(train_dataloader)
        t1 = time.time()
        print(
            f"[info] Epoch: {epoch:3d} | train loss: {training_loss:5.5f} | time: {(t1-t0)/60:4.2f}"
        )

        if epoch % train_cfg.save_interval == 0 and epoch > 0:
            if cfg.training.save_all_checkpoints:
                model_checkpoint_name_ep = os.path.join(
                        experiment_dir, f"multitask_model_epoch_{epoch:04d}.pth"
                    )
            else:
                model_checkpoint_name_ep = os.path.join(
                        experiment_dir, f"multitask_model.pth"
                    )
            utils.save_state({
                'model': model,
                'optimizers': optimizers,
                'schedulers': schedulers,
                'scaler': scaler,
                'epoch': epoch,
                'stage': cfg.stage,
                'steps': steps,
                'wandb_id': wandb.run.id,
                'experiment_dir': experiment_dir,
                'experiment_name': experiment_name,
                'config': OmegaConf.to_container(cfg, resolve=True)
            }, model_checkpoint_name_ep)

        [scheduler.step() for scheduler in schedulers]
    print("[info] finished learning\n")
    wandb.finish()

if __name__ == "__main__":
    main()
