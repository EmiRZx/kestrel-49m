# data/

Everything here is generated, and nothing here is committed — the corpus is ~12 GB and the shards are
reproducible from the scripts.

```
data/
  tokenizer/vocab16384/     tokenizer.json + meta.json     <- scripts/train_tokenizer.py
  tokens_v16384/            train_*.bin, val_*.bin, wt103_*.bin, manifests,
                            decontam.npz, prepare_report.json   <- scripts/prepare_data.py
  tokens_pilot/             a 300 M-token slice of the same mix, for ablations and smoke tests
  hf/                       HuggingFace datasets cache (set HF_HOME to point here)
```

Rebuild from scratch:

```bash
python scripts/train_tokenizer.py --vocab-size 16384
python scripts/prepare_data.py --tokens 6e9 --vocab-size 16384
```

`decontam.npz` is the window-hash index of the scored evaluation splits. It is kept next to the
shards on purpose: it makes the decontamination claim auditable rather than merely asserted, and
`prepare_data.py` reuses it instead of rebuilding when re-running.
