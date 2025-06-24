# train_depth.py
import argparse, os, yaml, torch
from pathlib import Path

# —— NOVOS helpers ——————————————————————————————————————
from panorai.depth.training.train_utils import (
    build_dataloaders,           # ➊
    build_model_optim_sched,     # ➋
    build_loss_fn,
    search_best_scale,
    TransformerUnfreezeScheduler, # ➌
    build_teacher_model 
)
from panorai.depth.trainers.depth_trainer import DepthTrainer
# ————————————————————————————————————————————————

# ---------- argparse & overrides ----------
def _parse():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--override", nargs="*", default=[])
    return p.parse_args()


def _apply_overrides(cfg, ovs):
    for ov in ovs:
        if "=" not in ov:
            continue
        k, v = ov.split("=", 1)
        tgt = cfg
        for kk in k.split(".")[:-1]:
            tgt = tgt.setdefault(kk, {})
        tgt[k.split(".")[-1]] = yaml.safe_load(v)
    return cfg

# ─────────────── Prewarm function ────────────────
import time
def prewarm_cache(trainloader, max_batches=None):
    """
    Iterate through trainloader to populate LMDB cache.
    If max_batches is None, runs through all batches.
    """
    print(f"🚀 Pre-warming LMDB cache for {max_batches or 'all'} batches …")
    for i, _ in enumerate(trainloader):
        if max_batches is not None and i + 1 >= max_batches:
            break
    print(f"✅ Pre-warmed {i+1} batches into LMDB.")

# ---------- main ----------
def main():
    args = _parse()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    cfg = _apply_overrides(cfg, args.override)

    device = cfg.get("device", "mps")
    teacher_model = build_teacher_model(cfg)

    # ➊ dataloaders
    if cfg.get("refine", False):
        trainloader, valloader = build_dataloaders(cfg, teacher_model)
    else:
        trainloader, valloader = build_dataloaders(cfg, None
        )

    # ─── Pre-warm LMDB cache ─────────────────────────
    # optionally limit via cfg["prewarm_batches"]
    # prewarm_batches = cfg.get("prewarm_batches", None)
    # prewarm_cache(trainloader, max_batches=prewarm_batches)

    # ➋ model / optim / sched
    depth_model, optimizer, scheduler, base_optim = build_model_optim_sched(cfg, trainloader)
    loss_fn = build_loss_fn(cfg)

    save_path = Path(cfg.get("checkpoint_dir", ".checkpoints")) / cfg["model_name"]
    save_path.mkdir(parents=True, exist_ok=True)

    # checkpoint (antes da compilação!)
    if cfg.get("load_from"):
        ckpt = torch.load(cfg["load_from"], map_location=device, weights_only=False)
        target = depth_model._orig_mod if hasattr(depth_model, "_orig_mod") else depth_model
        target.load_state_dict(ckpt["model"], strict=False)  # strict=False ignora chaves extras
        print(f"✅ checkpoint {cfg['load_from']} carregado")

    # -------- best-scale --------
    best_scale = cfg.get("best_scale")
    if best_scale is None and cfg.get("estimate_best_scale"):
        best_scale = search_best_scale(depth_model, trainloader, cfg)
        cfg["best_scale"] = best_scale
    best_scale = best_scale or 1.0
    print(f"ℹ️ best_scale = {best_scale}")

    trainer = DepthTrainer(
        model=depth_model,
        trainloader=trainloader,
        valloader=valloader,
        loss_fn=loss_fn,
        max_depth=cfg["max_depth"],
        best_scale=best_scale,
        num_warmup_epochs=cfg.get("warmup_epochs", 0),
        adaptive_scaling=cfg.get("adaptive_scaling", False),
        grad_clip=cfg.get("grad_clip", 50.0),
        grad_accum=cfg.get("grad_accum", 2),
        debug=cfg.get("debug", True),
        device=device,
        compile_model=cfg.get("compile", False),
        verbose=cfg.get("verbose", True),
        noise_warmup_epochs=cfg.get("noise_warmup_epochs", 0),
        compute_metrics=False
    )

    # ➌ progressive unfreeze
    if cfg.get("freeze", False):
        unfreeze_sched = TransformerUnfreezeScheduler(
            model=depth_model,
            start_block=cfg.get("unfreeze_start_block", 23),
            min_block=cfg.get("unfreeze_min_block", 17),
            grad_threshold=cfg.get("grad_threshold", 2e-3),
            patience=cfg.get("unfreeze_patience", 10),
            warmup_epochs=cfg.get("warmup_epochs", 5),
            min_epochs_per_unfreeze=cfg.get("min_epochs_per_unfreeze", 5),
            logger=trainer.print,
            log_path=save_path / "unfreeze_log.jsonl",
        )
        trainer.set_unfreeze_scheduler(unfreeze_sched)

    trainer.set_accelerator(
            optimizer,
            scheduler,
            lr_getter=lambda m: m.parameters(),
            base_optim_class=base_optim        # ← pass AdamW (or your base_optim) directly
        )
    

    # salvar cfg final
    with open(save_path / "final_config.yaml", "w") as f:
        yaml.dump(cfg, f)

    trainer.train(
        epochs=cfg["epochs"],
        val_interval=cfg.get("val_interval", 1),
        save_dir=str(save_path),
        start_epoch=0,
    )


if __name__ == "__main__":
    main()