from pathlib import Path
from collections.abc import Iterable, Iterator
from helpers import load_vocab, load_merges
import regex as re

_PAT = re.compile(
        r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    )

class Tokenizer:
    def __init__(
            self,
            vocab: dict[int, bytes],
            merges: list[tuple[bytes, bytes]],
            special_tokens: list[str] | None = None
    ):
        self.vocab = vocab
        self.byte_decoder: dict[bytes, int] = {v:k for k, v in vocab.items()}
        self.merge_ranks: dict[tuple[int, int], int] = {}
        self.merge_results: dict[tuple[int, int], int] = {}
        for i, (a, b) in enumerate(merges):
            pair = (self.byte_decoder[a], self.byte_decoder[b])
            self.merge_ranks[pair] = i
            self.merge_results[pair] = self.byte_decoder[a + b]
        self.byte_to_id: list[int] = [self.byte_decoder[bytes([i])] for i in range(256)]
        self._word_cache: dict[bytes, list[int]] = {}

        self.tokens_split = None
        if special_tokens:
            sorted_tokens = sorted(special_tokens,key=len, reverse=True)
            split_pat = "|".join(re.escape(t) for t in sorted_tokens)
            self.tokens_split = re.compile(split_pat)

    @classmethod
    def from_files(
            cls,
            vocab_filepath: Path | str,
            merges_filepath: Path | str,
            special_tokens: list[str] | None = None
    ):
        return cls(
            vocab=load_vocab(vocab_filepath),
            merges=load_merges(merges_filepath),
            special_tokens=special_tokens)

    def _bpe_word(self, world: bytes) -> list[int]:
        tokens = [self.byte_to_id[b] for b in world]
        rank = self.merge_ranks.get
        inf = float("inf")
        while len(tokens) > 1:
            pair = min(zip(tokens, tokens[1:]), key=lambda p: rank(p, inf))
            if pair not in self.merge_ranks:
                break
            idx = self.merge_results[pair]
            temp = []
            i = 0
            while i < len(tokens):
                if tokens[i] == pair[0] and i + 1 < len(tokens) and tokens[i + 1] == pair[1]:
                    temp.append(idx)
                    i += 2
                else:
                    temp.append(tokens[i])
                    i += 1
            tokens = temp
        return tokens

    def encode(self, text: str) -> list[int]:
        ids = []

        def _iter_text():
            if self.tokens_split is None:
                yield 0, len(text), False
                return
            cur = 0
            for m in self.tokens_split.finditer(text):
                yield cur, m.start(), False
                yield m.start(), m.end(), True
                cur = m.end()
            yield cur, len(text), False

        for s, e, is_special in _iter_text():
            if is_special:
                chunk = text[s:e].encode("utf-8")
                ids.append(self.byte_decoder[chunk])
            else:
                for m in _PAT.finditer(text, s, e):
                    word = m.group().encode("utf-8")
                    token_ids = self._word_cache.get(word)
                    if token_ids is None:
                        token_ids = self._bpe_word(word)
                        self._word_cache[word] = token_ids

                    ids.extend(token_ids)

        return ids

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        for chunk in iterable:
            yield from self.encode(chunk)

    def decode(self, ids: list[int]) -> str:
        tokens = b"".join(map(self.vocab.__getitem__, ids))
        return tokens.decode("utf-8", "replace")

if __name__ == "__main__":
    # import sys
    # vocab_filepath, merges_filepath = sys.argv[1:]
    import time
    vocab_filepath = Path("./trained_bpe/owt_vocab_32k.json")
    merges_filepath = Path("./trained_bpe/owt_merges_32k.txt")
    text = Path("../data/owt_train.txt")
    original_text = text.read_text()
    # raw_bytes_count = len(list(tinystory_sample.read_bytes()))

    ## ------- Benchmark ``compression rate``
    #
    # eot = "<|endoftext|>"
    # special_tokens = [eot]
    # tok_in_special = len(list(eot.encode()))
    # st = re.compile(re.escape(eot))
    # count = sum(1 for _ in st.finditer(original_text))
    #
    # t0 = time.perf_counter()
    # tokenizer = Tokenizer.from_files(
    #     vocab_filepath=vocab_filepath,
    #     merges_filepath=merges_filepath,
    #     special_tokens=special_tokens
    # )
    # ids = tokenizer.encode(original_text)
    # print(f"compression ratio {(len(uncompressed_text)-(tok_in_special*count))/(len(ids)-count)} tokens"
    #       f"in {time.perf_counter() - t0:.2f}s")
    # string_decoded = tokenizer.decode(ids)
    # print("Yay" if original_text == string_decoded else "Nope")

    # ------- Benchmark ``throughput``
    special_tokens = ["<|endoftext|>"]
    tokenizer = Tokenizer.from_files(
        vocab_filepath=vocab_filepath,
        merges_filepath=merges_filepath,
        special_tokens=special_tokens
    )

    t0 = time.perf_counter()
    ids = tokenizer.encode(original_text)
    end_time = time.perf_counter()

    elapsed_seconds = end_time - t0
    # bytes_per_second = raw_bytes_count / elapsed_seconds

    # mb_per_second = bytes_per_second / (1024 * 1024)

    # print(f"Processed {raw_bytes_count} bytes in {elapsed_seconds:.4f} seconds.")
    # print(f"Throughput: {mb_per_second:.2f} MB/s")