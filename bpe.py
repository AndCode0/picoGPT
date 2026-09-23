import argparse
from typing import BinaryIO
import regex as re
import heapq
import os
import mmap
import multiprocessing as mp
from collections import Counter
from pathlib import Path
from helpers import save_vocab, save_merges


_PAT = re.compile(
        r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    )

def find_chunk_boundaries(
    file: BinaryIO,
    effective_processes: int,
    tokens: list[bytes],
    max_scan_bytes: int | None = None,
) -> list[tuple[int, int]]:
    """Chunk boundaries snapped to the nearest following special token.
    """
    max_token_len = len(tokens[0])

    original_pos = file.tell()
    try:
        file.seek(0, os.SEEK_END)
        file_size = file.tell()

        if file_size == 0:
            return []
        if effective_processes != 1:
            desired_num_chunks = 4 * effective_processes
            desired_num_chunks = min(desired_num_chunks, file_size)
            chunk_size = file_size // desired_num_chunks
        else:
            chunk_size = 256 * 1024 * 1024
            desired_num_chunks = file_size // chunk_size

        if max_scan_bytes is None:
            max_scan_bytes = 2 * chunk_size + 65_536

        boundaries: set[int] = {0, file_size}

        with mmap.mmap(file.fileno(), 0, access=mmap.ACCESS_READ) as mm:
            for i in range(1, desired_num_chunks):
                guess = i * chunk_size
                # +token_len so a token straddling the budget edge is caught
                scan_end = min(guess + max_scan_bytes + max_token_len, file_size)

                best_pos, best_len = -1, 0
                for tok in tokens:
                    pos = mm.find(tok, guess, scan_end)
                    if pos != -1 and (best_pos == -1 or pos < best_pos):
                        best_pos, best_len = pos, len(tok)
                        if pos == guess:
                            break   # can't do better than the guess itself

                if best_pos != -1:
                    boundaries.add(best_pos)

        b = sorted(boundaries)
        return [(s, e) for s, e in zip(b, b[1:]) if s < e]
    finally:
        file.seek(original_pos)

def _count_pretokens(args: tuple[str, int, int, re.Pattern]) -> dict[str, int]:
    path, start, end, tokens_split = args
    with open(path, "rb") as f:
        f.seek(start)
        text = f.read(end - start).decode("utf-8")

    counts: dict[str, int] = {}

    def _gaps():
        cur = 0
        for m in tokens_split.finditer(text):
            yield cur, m.start()
            cur = m.end()
        yield cur, len(text)

    for s, e in _gaps():
        for m in _PAT.finditer(text, s, e):
            tok = m.group()
            # Equivalent to counts[tok] = counts.get(tok, 0) + 1
            try:
                counts[tok] += 1
            except KeyError:
                counts[tok] = 1

    return counts

def _pretokenize(text_path: str, special_tokens: list[str], num_processes: int) -> dict[str, int]:

    effective_processes = min(max(1, num_processes), os.cpu_count() or 1)

    sorted_tokens = sorted(special_tokens, key=len, reverse=True)
    encoded = [t.encode("utf-8") for t in sorted_tokens]
    split_pat = "|".join(re.escape(t) for t in sorted_tokens)
    tokens_split = re.compile(split_pat)

    with open(text_path, "rb") as f:
        chunks = find_chunk_boundaries(f, effective_processes, encoded)

    if not chunks:
        return {}

    jobs = [(text_path, s, e, tokens_split) for s, e in chunks]

    total: dict[str, int] = Counter()

    if effective_processes <= 1 or len(jobs) <= 1:
        for job in jobs:
            total.update(_count_pretokens(job))
        return dict(total)

    with mp.Pool(effective_processes) as pool:
        for partial in pool.imap_unordered(_count_pretokens, jobs):
            total.update(partial)

    return dict(total)

class _HeapItem:
    __slots__ = ("count", "key", "pair")
    def __init__(self, count: int, key: tuple[bytes, bytes], pair: tuple[int, int]):
        self.count = count
        self.key = key
        self.pair = pair

    def __lt__(self, other:"_HeapItem") -> bool:
        if self.count != other.count:
            return self.count > other.count # we want a min-heap to pop the HIGHEST count
        return self.key > other.key

def _size_label(n: int) -> str:
    """32000 -> "32k"; a count that isn't a whole thousand stays exact (327 -> "327")."""
    return f"{n // 1000}k" if n and n % 1000 == 0 else str(n)


def train_bpe(
        input_path: str,
        vocab_size: int,
        special_tokens: list[str],
        num_processes: int,
) ->  tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """
    Returns:
        vocab:  token id -> token bytes
        merges: list of (bytes, bytes) in creation order
    """
    assert special_tokens, "Special_tokens must be non-empty strings/bytes"
    special_tokens = list(special_tokens)
    n_special_tokens = len(special_tokens)
    if vocab_size < 256 + n_special_tokens:
        raise ValueError(
            f"vocab_size={vocab_size} cannot fit 256 byte tokens "
            f"+ {n_special_tokens} special tokens"
        )
    num_merges = vocab_size - 256 - n_special_tokens

    pretoken_counts = _pretokenize(
                            input_path,
                            special_tokens,
                            num_processes
                        )

    # List of tokens created from the coarse-grained tokenization via regex
    words: list[list[int]] = []
    # List s.t. freqs[i] = number of times tokens[i] appears in the corpus
    freqs: list[int] = []
    for tok, c in pretoken_counts.items():
        words.append(list(tok.encode("utf-8")))
        freqs.append(c)
    del pretoken_counts

    pair_counts: dict[tuple[int, int], int] = {}
    pair_words: dict[tuple[int, int], set[int]] = {}
    for wi, (w, c) in enumerate(zip(words, freqs)):
        prev = -1
        for x in w:
            if prev >= 0:
                p = (prev, x)
                pair_counts[p] = pair_counts.get(p, 0) + c
                s = pair_words.get(p)
                if s is None:
                    pair_words[p] = {wi}
                else:
                    s.add(wi)
            prev = x

    vocab_list: list[bytes] = [bytes([i]) for i in range(256)]
    vocab_list.extend(t.encode("utf-8") for t in special_tokens)

    heap = [
        _HeapItem(c, (vocab_list[p[0]], vocab_list[p[1]]), p)
        for p, c in pair_counts.items()
    ]
    heapq.heapify(heap)

    merges: list[tuple[bytes, bytes]] = []

    for _ in range(num_merges):
        best = None
        while heap:
            item = heapq.heappop(heap)
            if pair_counts.get(item.pair) == item.count:
                best = item
                break
        if best is None:
            break

        A, B = best.pair
        a_bytes, b_bytes = best.key
        Z = len(vocab_list)
        vocab_list.append(a_bytes + b_bytes)
        merges.append(best.key)

        # The merged pair is fully consumed across all words that contain it.
        affected = pair_words.pop(best.pair, ())
        pair_counts.pop(best.pair, None)
        changed: set[tuple[int, int]] = set()

        for wi in affected:
            w = words[wi]
            n = len(w)
            if n < 2:
                continue
            c = freqs[wi]

            i = 0
            new_w: list[int] = []
            append = new_w.append
            fresh_prev = False      # was the last appended token a fresh Z?
            while i < n:
                if i + 1 < n and w[i] == A and w[i + 1] == B:
                    # -- merge site at old position (i, i+1)
                    # always remove LEFT pair (w[i-1], A)
                    if i > 0:
                        lp = (w[i - 1], A)
                        if lp != best.pair:
                            v = pair_counts.get(lp, 0) - c
                            if v > 0:
                                pair_counts[lp] = v
                            else:
                                pair_counts.pop(lp, None)
                            changed.add(lp)
                    # remove RIGHT pair (B, w[i+2]) IF right neighbor
                    # w[i+2] is not the start of new merge site
                    if i + 2 < n:
                        right = w[i + 2]
                        next_is_site = (
                            right == A and i + 3 < n and w[i + 3] == B
                        )
                        if not next_is_site:
                            rp = (B, right)
                            if rp != best.pair:
                                v = pair_counts.get(rp, 0) - c
                                if v > 0:
                                    pair_counts[rp] = v
                                else:
                                    pair_counts.pop(rp, None)
                                changed.add(rp)
                    # always add LEFT pair (prev_in_new, Z) if new_w isn't empty
                    if new_w:
                        pa = (new_w[-1], Z)
                        pair_counts[pa] = pair_counts.get(pa, 0) + c
                        changed.add(pa)
                        s = pair_words.get(pa)
                        if s is None:
                            pair_words[pa] = {wi}
                        else:
                            s.add(wi)
                    append(Z)
                    fresh_prev = True
                    i += 2
                else:
                    # if the previous emitted token was a fresh Z, add (Z, this)
                    if fresh_prev:
                        pa = (new_w[-1], w[i])
                        pair_counts[pa] = pair_counts.get(pa, 0) + c
                        changed.add(pa)
                        s = pair_words.get(pa)
                        if s is None:
                            pair_words[pa] = {wi}
                        else:
                            s.add(wi)
                    append(w[i])
                    fresh_prev = False
                    i += 1

            words[wi] = new_w

        for p in changed:
            cnt = pair_counts.get(p)
            if cnt:
                heapq.heappush(
                    heap, _HeapItem(cnt, (vocab_list[p[0]], vocab_list[p[1]]), p)
                )
    return {i: tok for i, tok in enumerate(vocab_list)}, merges


if __name__ == "__main__":
    import time

    ap = argparse.ArgumentParser(description="Train a byte-level BPE tokenizer.")
    ap.add_argument("--corpus", type=Path, required=True, help="Input text file")
    ap.add_argument("--vocab_size", type=int, default=32000,
                    help="Total vocabulary size, byte tokens and specials included")
    ap.add_argument("--num_processes", type=int, default=os.cpu_count(),
                    help="Pre-tokenization workers (default: all cores)")
    ap.add_argument("--special_tokens", type=str, nargs="*", default=["<|endoftext|>"])
    ap.add_argument("--output_dir", type=Path, default=Path("trained_bpe/"))
    ap.add_argument("--name", type=str, default=None,
                    help="Subdirectory of --output_dir; defaults to the corpus filename without extension")
    args = ap.parse_args()

    # trained_bpe/<corpus name>/vocab_<size>.json and merges_<size>.txt,
    # e.g. trained_bpe/owt_train/vocab_32k.json
    out_dir = args.output_dir / (args.name or args.corpus.stem)

    t0 = time.perf_counter()
    vocab, merges = train_bpe(
        args.corpus,
        vocab_size=args.vocab_size,
        special_tokens=args.special_tokens,
        num_processes=args.num_processes
    )
    print(f"{len(vocab)} tokens, {len(merges)} merges "
          f"in {time.perf_counter() - t0:.2f}s")
    print("first merges:", merges[:10])
    print("longest token:", max(vocab.values(), key=len))

    # label the files with what came out, not with what was asked for: a corpus
    # that runs out of merge candidates stops short of --vocab_size
    size_label = _size_label(len(vocab))
    vocab_path = out_dir / f"vocab_{size_label}.json"
    merges_path = out_dir / f"merges_{size_label}.txt"

    out_dir.mkdir(parents=True, exist_ok=True)
    save_vocab(vocab, vocab_path)
    save_merges(merges, merges_path)
    print(f"wrote {vocab_path}, {merges_path}")