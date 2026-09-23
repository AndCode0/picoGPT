picoGPT
=======

A personal project: I'm following Stanford's CS336 (Language Modeling from
Scratch) and implementing everything myself -- BPE tokenizer, transformer LM,
attention with RoPE, RMSNorm, AdamW, LR schedule, gradient clipping, the
training loop. No training framework, no pretrained weights.

I tried to keep it as general as I could on the way (device auto-detect,
configurable model/optimizer/schedule, checkpoint + resume, optional wandb), but
the scope is limited: it's a learning exercise, not a library. It is only tested
on small models, on a single Apple Silicon (MPS) machine, with the CS336 data
format. Expect rough edges anywhere else.

Files
-----
bpe.py            byte-level BPE training (multiprocess, mmap'd corpus)
tokenizer.py      the tokenizer itself + a (hardcoded) benchmark scaffold
TransformerLM.py  model, batch sampling, checkpoint load/save
nn_utils.py       linear, attention, RoPE, RMSNorm, cross-entropy, AdamW, scheduler
helpers.py        vocab / merges (de)serialization
train.py          training loop

Setup
-----
uv is required. Python 3.12 is pinned

    uv sync

On Linux/CUDA the PyPI torch wheel is CPU-only; install torch from the CUDA
index instead -- see the note at the bottom of pyproject.toml.


1. Train a tokenizer
--------------------
    uv run bpe.py --corpus data/owt_train.txt --vocab_size 32000 --num_processes 8

Writes trained_bpe/owt_train/vocab_32k.json and trained_bpe/owt_train/merges_32k.txt.
The subdirectory is the corpus filename without its extension (--name overrides
it). The size label is the vocabulary actually produced: a corpus that runs out
of merge candidates stops short of --vocab_size and gets named for where it
stopped (327 tokens -> vocab_327.json). Other flags: --special_tokens,
--output_dir, --num_processes (defaults to all cores).

    uv run bpe.py --help


2. Turn text into token ids
---------------------------
train.py consumes a memmap-able .npy array of token ids. To produces
one do something like:

    uv run python -c "
    import numpy as np, pathlib
    from tokenizer import Tokenizer
    tok = Tokenizer.from_files(
        vocab_filepath=pathlib.Path('``path/to/your_vocab.json``'),
        merges_filepath=pathlib.Path('``path/to/your_merges.txt``'),
        special_tokens=['<|endoftext|>'],
    )
    ids = tok.encode(pathlib.Path('data/data_to_train.txt').read_text())
    np.save('dataset.npy', np.array(ids, dtype=np.uint16))  # uint32 if vocab > 65535
    "


3. Train
--------
    uv run train.py --data dataset.npy --val_data val.npy \
        --vocab_size 32000 --context_len 256 \
        --d_model 512 --d_ff 1344 --num_layers 4 --n_heads 16 \
        --batch_size 128 --training_steps 1000 \
        --max_lr 1e-3 --min_lr 1e-4 --warmup_it 500 --weight_decay 0.1 \
        --log_every 10 --eval_every 500 --eval_batches 20 \
        --checkpoint_every 1000 --checkpoint_dir checkpoints

    --vocab_size   must match the tokenizer you trained
    --val_data     optional; without it there is no val/loss and no eval timing
    --log_every    console + wandb scalar interval
    --checkpoint_every 0 disables periodic saves (final.pt is always written)
    --eval_batches how many batches the validation loss averages over

1000 steps is not a typo: away from CUDA there is an assert that
batch_size x training_steps x context_len < 4.1e7 tokens (128 x 1000 x 256),
so a laptop run has to stay small. The check is skipped on CUDA, where
--training_steps 20000 is the default.

Resume from a checkpoint:

    uv run train.py --data dataset.npy --val_data val.npy ... \
        --resume_from checkpoints/latest.pt

Experiment tracking is optional and degrades to a no-op if wandb is missing or
you are not logged in. No entity is hardcoded: wandb resolves WANDB_ENTITY, or
the default entity of whoever is logged in.

    uv run train.py ... --wandb_project transformerLM --wandb_run_name baseline
    uv run train.py ... --wandb_mode offline     # log locally, sync later
    uv run train.py ... --wandb_mode disabled    # no wandb at all

All flags and their defaults:

    uv run train.py --help
