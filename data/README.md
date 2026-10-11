# data/

Nothing in here is committed except this file, `LICENSES.md` and `splits/`.

```
data/
├── raw/        # output of scripts/download_public.sh and gated downloads
├── processed/  # crops, cached embeddings
└── splits/     # fixed train/val/test ID lists (committed)
```

Get the public sets with `bash scripts/download_public.sh`. Gated sets
(First Impressions V2, MIT Interview, RAF-DB, AffectNet, BIWI, Gaze360) need
access requests first; see `LICENSES.md`.
