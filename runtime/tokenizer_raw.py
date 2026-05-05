"""tokenizer_raw.py — Standalone tokenizer bypass per DSv4-Flash.

Usa tokenizers.Tokenizer.from_file(...) invece di transformers.AutoTokenizer
per evitare il drift HF: 'PreTrainedConfig has no attribute max_position_embeddings'
causato da rope_scaling fields del DSv4 config non supportati nel transformers
attuale del container.

Vocab: 129280 (DeepSeek V4 BPE).

Test:
  >>> from tokenizer_raw import RawTokenizer
  >>> t = RawTokenizer()
  >>> t.encode("Hello", add_bos=False)
  [19923]
  >>> t.decode([19923])
  'Hello'
"""
from __future__ import annotations
import os
from pathlib import Path
from typing import List

DEFAULT_PATH = os.path.join(os.environ.get("DSV4_WEIGHTS", ""), "tokenizer.json")


class RawTokenizer:
    def __init__(self, path: str = DEFAULT_PATH):
        from tokenizers import Tokenizer
        self._tok = Tokenizer.from_file(str(Path(path)))
        self.vocab_size = self._tok.get_vocab_size()
        # DSv4 special tokens (BOS=0, EOS=1)
        self.bos_token_id = 0
        self.eos_token_id = 1

    def encode(self, text: str, add_bos: bool = True) -> List[int]:
        ids = self._tok.encode(text).ids
        if add_bos and (not ids or ids[0] != self.bos_token_id):
            ids = [self.bos_token_id] + ids
        return ids

    def decode(self, ids: List[int]) -> str:
        ids_clean = [int(i) for i in ids if 0 <= int(i) < self.vocab_size]
        try:
            return self._tok.decode(ids_clean, skip_special_tokens=False)
        except Exception as e:
            return f"<decode err: {e}>"

    def __repr__(self) -> str:
        return f"<RawTokenizer vocab={self.vocab_size}>"


def _self_test():
    t = RawTokenizer()
    print(f"[tokenizer_raw] {t}")
    cases = [("Hello", "Hello"),
             ("The capital of France is", "The capital of France is"),
             ("2+2=", "2+2="),
             ("Ciao, come stai?", "Ciao, come stai?")]
    all_ok = True
    for txt, expect in cases:
        ids = t.encode(txt, add_bos=False)
        dec = t.decode(ids)
        ok = dec == expect
        all_ok = all_ok and ok
        print(f"  encode({txt!r}) ids={ids} decode={dec!r} ok={ok}")
    return all_ok


if __name__ == "__main__":
    ok = _self_test()
    print(f"[tokenizer_raw] all_ok={ok}")
