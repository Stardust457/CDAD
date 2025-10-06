import os
import torch
import sys

from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger

from .load_data import prepare_loader
from .CDAD import CDAD

import wandb

def run(args):
    # wandb
    name = f'{args.wandb_name}_{args.normal_class}_{args.hf_path}'
    project = f'{args.run_type}_{args.dataset_name}'
    os.environ["WANDB_API_KEY"] = args.wandb_api_key
    wandb.login()
    wandb.init(project=project, entity=args.wandb_entity, name=name)
    wandb_logger = WandbLogger()

    # device
    if not torch.cuda.is_available():
        print("CUDA is not available. Exiting.")
        sys.exit()

    device = torch.device("cuda:0")
    print("Device:", device)

    # lightning set-up
    trainer = Trainer(
        log_every_n_steps=args.log_every_n_steps,
        logger=wandb_logger,
        accelerator="gpu",
        devices=1,
        max_epochs=args.epochs,
        callbacks=[
            ModelCheckpoint(save_weights_only=True, mode="max", monitor=f"val_{args.val_monitor}"),
            LearningRateMonitor("epoch")
        ],
        enable_progress_bar=False
    )

    # data loaders
    train_loader, test_loader = prepare_loader(image_size=args.image_size,
                                                        path=args.data_dir,
                                                        dataset_name=args.dataset_name,
                                                        class_name=args.normal_class,
                                                        batch_size=args.batch_size,
                                                        test_batch_size=args.test_batch_size,
                                                        num_workers=args.num_workers,
                                                        seed=args.seed,
                                                        shots=args.shots)

    # seeding
    seed_everything(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # train / load model
    if args.run_type == 'cdad':
        if args.load_checkpoint:
            model = CDAD.load_from_checkpoint(args.checkpoint_dir)
            checkpoint_dir = args.checkpoint_dir
        else:
            model = CDAD(lr=args.lr,
                        lr_decay_factor=args.lr_decay_factor,
                        lr_adaptor=args.lr_adaptor,#改动
                        hf_path=args.hf_path,
                        layers_to_extract_from=args.layers_to_extract_from,
                        hidden_dim=args.hidden_dim,
                        wd=args.wd,
                        epochs=args.epochs,
                        noise_std=args.noise_std,
                        dsc_layers=args.dsc_layers,
                        dsc_heads=args.dsc_heads,
                        dsc_dropout=args.dsc_dropout,
                        pool_size=args.pool_size,
                        image_size=args.image_size,
                        num_fake_patches=args.num_fake_patches,
                        fake_feature_type=args.fake_feature_type,
                        top_k=args.top_k,
                        log_pixel_metrics=args.log_pixel_metrics,
                        smoothing_sigma=args.smoothing_sigma,
                        smoothing_radius=args.smoothing_radius)
            trainer.fit(model, train_loader, test_loader)
            checkpoint_dir = trainer.checkpoint_callback.best_model_path
            model = CDAD.load_from_checkpoint(checkpoint_dir)
    else:
        print("This is not a valid method name.")
        sys.exit()

    # test
    test_result = trainer.test(model, test_loader, verbose=True)

    print("Checkpoint directory:", checkpoint_dir)

    # wandb
    wandb.finish()

    return