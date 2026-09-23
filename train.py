import TransformerLM as tlm
import nn_utils as nu
import torch
import argparse
import numpy as np
import logging
import signal
import time
from pathlib import Path

SEED = 12345

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


class _DisabledRun:
    """Minimal stand-in for a wandb run, so training works without wandb."""

    def __init__(self):
        self.summary = {}
        self.url = None

    def log(self, *args, **kwargs):
        pass

    def finish(self, *args, **kwargs):
        pass


def init_run(project, mode=None, name=None, config=None):
    """Init wandb if it is usable, else return a no-op run object.
    The entity is left to wandb's own resolution.
    """
    try:
        import wandb
    except ImportError:
        logger.warning("wandb not installed — running without experiment tracking")
        return _DisabledRun()

    try:
        run = wandb.init(project=project, name=name, mode=mode, config=config)
    except Exception as e:
        logger.warning("wandb.init failed (%s) — running without experiment tracking", e)
        return _DisabledRun()

    if run is None:  # wandb returns None when disabled via WANDB_DISABLED
        return _DisabledRun()
    return run


def sync_device(device):
    """Block the host until the device queue is drained (no-op on CPU)."""
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


DTYPES = {
    "float32": torch.float32,
    # will introduce more after updating AdamW, attention, and gradient scaling
    # "float16": torch.float16,
    # "bfloat16": torch.bfloat16,
}
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    # --- data
    ap.add_argument("--data", type=str, default="dataset.npy",
                    help="Path to memmap-able .npy token array (train split)")
    ap.add_argument("--val_data", type=str, default=None,
                    help="Optional .npy token array for validation loss")
    # --- training
    ap.add_argument("--training_steps", type=int, default=20000)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--log_every", type=int, default=10, help="Console + wandb scalar interval (steps)")
    ap.add_argument("--eval_every", type=int, default=500, help="Validation interval (steps); 0 disables")
    ap.add_argument("--eval_batches", type=int, default=20, help="Batches averaged per validation eval")
    # --- model
    ap.add_argument("--vocab_size", type=int, default=10000)
    ap.add_argument("--context_len", type=int, default=256)
    ap.add_argument("--d_model", type=int, default=512) # usually 768, but faster this way
    ap.add_argument("--d_ff", type=int, default=1344)
    ap.add_argument("--rope_theta", type=float, default=10000.0)
    ap.add_argument("--num_layers", type=int, default=4)
    ap.add_argument("--n_heads", type=int, default=16)
    ap.add_argument("--eps", type=float, default=1e-6)
    ap.add_argument("--dtype", type=str, default="float32", choices=list(DTYPES))
    # --- optimizer
    ap.add_argument("--beta1", type=float, default=0.9)
    ap.add_argument("--beta2", type=float, default=0.95)
    ap.add_argument("--weight_decay", type=float, default=0.1)
    # --- lr scheduler
    # the scheduler sets the lr every step, so it is the single source of truth for lr;
    # AdamW's initial lr is irrelevant (overwritten before the first step)
    ap.add_argument("--max_lr", type=float, default=1e-3)
    ap.add_argument("--min_lr", type=float, default=1e-4, help="Final lr; ~0.1x max_lr is typical")
    ap.add_argument("--warmup_it", type=int, default=500)
    ap.add_argument("--cos_cycle_it", type=int, default=None,
                    help="Cosine annealing length; defaults to training_steps")
    # --- gradient clipping
    ap.add_argument("--max_grad_norm", type=float, default=1.0)
    # --- experiment tracking (wandb)
    # No entity is passed: wandb uses WANDB_ENTITY.
    ap.add_argument("--wandb_project", type=str, default="transformerLM")
    ap.add_argument("--wandb_run_name", type=str, default=None)
    ap.add_argument("--wandb_mode", type=str, default=None,
                    choices=["online", "offline", "disabled"],
                    help="Defaults to WANDB_MODE env var, else wandb's own default (online)")
    # --- checkpointing
    ap.add_argument("--checkpoint_dir", type=str, default="checkpoints")
    ap.add_argument("--checkpoint_every", type=int, default=1000, help="0 disables periodic saves")
    ap.add_argument("--resume_from", type=str, default=None, help="Checkpoint path to resume from")
    args = ap.parse_args()

    if args.cos_cycle_it is None:
        args.cos_cycle_it = args.training_steps
    dtype = DTYPES[args.dtype]

    ## Seeding the run
    train_ss, val_ss = np.random.SeedSequence(SEED).spawn(2)
    train_gen = np.random.default_rng(train_ss)
    val_gen   = np.random.default_rng(val_ss)
    torch.manual_seed(SEED)

    # device
    if torch.cuda.is_available():
        device = torch.device('cuda')
        torch.cuda.manual_seed_all(SEED)
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
        torch.mps.manual_seed(SEED)
    else:
        device = torch.device('cpu')
    logger.info(f'torch.device is {device}, dtype is {args.dtype}')

    # wandb.init (when tracking is enabled) installs its own SIGINT handler
    # that swallows Ctrl+C, so
    # KeyboardInterrupt would never reach the training loop — take it back.
    # First Ctrl+C: finish the current step, checkpoint, exit cleanly.
    # Second Ctrl+C: raise KeyboardInterrupt and bail out immediately.
    stop_requested = False
    def _handle_sigint(signum, frame):
        global stop_requested
        if stop_requested:
            raise KeyboardInterrupt
        stop_requested = True
        logger.warning("Ctrl+C received — will checkpoint and exit after this step (press again to force quit)")
    signal.signal(signal.SIGINT, _handle_sigint)

    dataset = np.load(args.data, mmap_mode='r')
    val_dataset = np.load(args.val_data, mmap_mode='r') if args.val_data else None

    # Run sanity checks
    #  batch size × total step count × context length
    if not torch.cuda.is_available():
        assert args.batch_size * args.training_steps * args.context_len < 4.1e7,f"""
    Let's keep it humble. Fixing the batch size and the context length 
    -> training_steps: {4.1e7//(args.batch_size * args.context_len)}"""
    assert args.warmup_it < args.cos_cycle_it, \
        "Scheduler: warmup_it must be smaller than cos_cycle_it (which defaults to training_steps)"
    assert args.log_every > 0, "log_every must be strictly greater than 0"
    assert len(dataset) > args.context_len, f"""
    A dataset (len: {len(dataset)}) shorter than context_len + 1 ({args.context_len}) will produce invalid sampling bounds.
    """
    if val_dataset is not None:
        assert len(val_dataset) > args.context_len, f"""
        A validation dataset (len: {len(val_dataset)}) shorter than context_len + 1 ({args.context_len}) will produce 
        invalid sampling bounds."""

    model = tlm.TransformerLM(
        args.vocab_size, args.context_len, args.num_layers, args.rope_theta,
        args.d_model, args.n_heads, args.d_ff, args.eps, device, dtype,
    )
    model = torch.compile(model)

    optimizer = nu.AdamW(
        model.parameters(),
        lr=args.max_lr,
        betas=(args.beta1, args.beta2),
        weight_decay=args.weight_decay,
    )

    ckpt_dir = Path(args.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    start_step = 0
    if args.resume_from:
        start_step = tlm.load_checkpoint(args.resume_from, model, optimizer, train_gen, device)
        logger.info(f"resumed from {args.resume_from} at step {start_step}")


    @torch.no_grad()
    def eval_loss() -> float:
        model.eval()
        losses = []
        for _ in range(args.eval_batches):
            inputs, targets = tlm.data_loading(val_dataset, args.batch_size, args.context_len, val_gen, device)
            logits = model(inputs)
            losses.append(nu.cross_entropy_loss(logits, targets).item())

        return float(np.mean(losses))


    tokens_per_step = args.batch_size * args.context_len
    eval_tokens = args.eval_batches * args.batch_size * args.context_len
    skipped_steps = 0
    interrupted = False
    step = start_step - 1
    t_train_start = time.perf_counter()

    # track this run
    run = init_run(args.wandb_project, args.wandb_mode, args.wandb_run_name, vars(args))
    n_params = sum(p.numel() for p in model.parameters())
    logger.info("model has %.2fM parameters", n_params / 1e6)
    run.summary["n_params"] = n_params

    best_score = float("inf")

    try:
        for step in range(start_step, args.training_steps):
            if stop_requested:
                interrupted = True
                break
            step_t0 = time.perf_counter()

            inputs, targets = tlm.data_loading(dataset, args.batch_size, args.context_len, train_gen, device)

            lr = nu.lr_cosine_scheduler(step, args.max_lr, args.min_lr, args.warmup_it, args.cos_cycle_it)
            optimizer.update_lr(lr)

            model.train()
            logits = model(inputs)
            loss = nu.cross_entropy_loss(logits, targets)

            if not torch.isfinite(loss):
                logger.warning("non-finite loss at step %d, skipping step", step)
                skipped_steps += 1
                optimizer.zero_grad(set_to_none=True)
                continue

            loss.backward()
            total_norm = nu.gradient_clipping(model.parameters(), args.max_grad_norm)

            if torch.isfinite(total_norm):
                optimizer.step()
            else:
                logger.warning("non-finite grad norm at step %d, skipping update", step)
                skipped_steps += 1

            optimizer.zero_grad(set_to_none=True)

            # train + perf scalars every log_every steps, independent of validation
            if step % args.log_every == 0:
                sync_device(device)
                step_s = time.perf_counter() - step_t0
                tok_per_s = tokens_per_step / step_s if step_s > 0 else float("nan")
                loss_v, grad_norm_v = loss.item(), total_norm.item()
                logger.info("step %d/%d | loss %.4f | lr %.2e | grad_norm %.2f | %.1f ms | %.0f tok/s",
                            step, args.training_steps, loss_v, lr, grad_norm_v, step_s * 1e3, tok_per_s)
                run.log({
                    "train/loss": loss_v,
                    "train/perplexity": float(np.exp(min(loss_v, 20.0))),
                    "train/lr": lr,
                    "train/grad_norm": grad_norm_v,
                    "train/skipped_steps": skipped_steps,
                    "perf/step_time_s": step_s,
                    "perf/tokens_per_s": tok_per_s,
                }, step=step)

            if val_dataset is not None and args.eval_every > 0 and step % args.eval_every == 0:
                sync_device(device)  # don't bill pending step work to the eval timer
                eval_t0 = time.perf_counter()
                vl = eval_loss()
                sync_device(device)
                eval_s = time.perf_counter() - eval_t0
                eval_tok_per_s = eval_tokens / eval_s if eval_s > 0 else float("nan")
                run.log({
                    "val/loss": vl,
                    "val/perplexity": float(np.exp(min(vl, 20.0))),
                    "perf/eval_time_s": eval_s,
                    "perf/eval_tokens_per_s": eval_tok_per_s,
                }, step=step)
                logger.info("step %d | val loss %.4f | eval %.1f ms | %.0f tok/s",
                            step, vl, eval_s * 1e3, eval_tok_per_s)

                if vl < best_score:
                    best_score = vl
                    run.summary["best_val_loss"] = vl

            if args.checkpoint_every > 0 and (step + 1) % args.checkpoint_every == 0:
                tlm.save_checkpoint(model, optimizer, step, ckpt_dir / "latest.pt", train_gen.bit_generator.state)
                logger.info(f"Step {step}: Checkpoint saved to {ckpt_dir / "latest.pt"}")

    except KeyboardInterrupt:
        logger.warning(f"force-interrupted at step {step}")
        interrupted = True
    finally:
        if interrupted:
            logger.warning("interrupted at step %d — saving checkpoint before exit", step)
            tlm.save_checkpoint(model, optimizer, step, ckpt_dir / "interrupted.pt", train_gen.bit_generator.state)
            logger.info(f"checkpoint saved to {ckpt_dir / 'interrupted.pt'}")
        else:
            tlm.save_checkpoint(model, optimizer, step, ckpt_dir / "final.pt", train_gen.bit_generator.state)
            logger.info(f"checkpoint saved to {ckpt_dir / 'final.pt'}")
        run.summary["interrupted"] = interrupted
        run.summary["wall_time_s"] = time.perf_counter() - t_train_start
        run.finish()
