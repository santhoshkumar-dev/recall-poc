# Visual search recovery audit

## Outcome

Recall's MobileCLIP image vectors were valid, but text inputs were padded with the EOT token
(`49407`) instead of zero. On the installed ONNX model this collapsed distinct prompts:
animal/building cosine was `0.992`; the corrected tokenizer produces `0.748`. The runtime now
enforces zero padding, exact 77-token inputs, exact projected output names, 512 dimensions,
finite non-zero vectors, and post-projection L2 normalization.

Visual work is now a durable stage independent of OCR. Missing models use `waiting_model`, model
installation activates runtimes and wakes work without restart, and prompt categories or optional
WD tags cannot block or dominate primary visual retrieval.

## Schema v7 and migration

- `embedding_profiles` records role, immutable checkpoint revision, artifact hashes, dimensions,
  preprocessing/tokenizer/schema versions, and shared vector-space identity.
- Image and E5 rows carry profile provenance and dimensions. Search reads only the active profile
  and rejects malformed, trailing-byte, non-finite, zero-norm, or wrong-size blobs.
- Legacy MobileCLIP image rows are reused only when their known model version, dimension, byte
  length, finite values, and norm validate. Legacy E5 rows lacked provenance and are rebuilt.
- Broken prompt classifications and old searchable WD-tag chunks are removed. Reindex jobs are
  idempotent, resumable, generation-counted, and keep compatible data until replacement.

## Retrieval and UI

- Visual-intent queries use visual RRF weight `0.75`; optional tags are capped at `0.05` and never
  qualify a result alone. Prompt-bank categories are diagnostic only.
- Absolute cosine/z-score inclusion gates were removed. Raw score, median, MAD, and z remain
  development diagnostics.
- Simple object plurals use the same MobileCLIP prompts as their singular form. Region hits collapse
  to the parent and result cards deduplicate by content hash.
- Results use a responsive, lazy-loaded thumbnail grid. Thumbnail access accepts only an asset ID
  and resolves a trusted app-data file in Rust.

## Measured validation

Reference environment: the local Windows development machine, MobileCLIP2-S0 CPU execution, and a
61-asset labeled evaluation library.

| Check | Result |
| --- | --- |
| Six dog/cat/building singular/plural queries, top-1 | Relevant for all six |
| Precision@5 / recall@10 | `1.00` / `1.00` for all six |
| Singular/plural top-10 Jaccard | dog `1.00`, cat `0.82`, building `1.00` |
| Warm end-to-end search | p50 `187 ms`, p95 `191 ms` at 61 assets |
| MobileCLIP runtime load | `863 ms` |
| First / warm image embedding | `92 ms` / p50 `87 ms` |
| Paired artifact size | `301,833,494` bytes |
| Exact 512-d scan kernel | p95 `3 ms` at 10k; `38 ms` at 100k |

Reproduce with `RECALL_MODEL_DIR` set to the installed models directory and `RECALL_APP_DATA` set
to a disposable copy of an evaluation database:

```powershell
cargo test --manifest-path src-tauri/Cargo.toml installed_text_encoder_uses_zero_padding_and_separates_concepts --lib -- --ignored --nocapture
cargo test --manifest-path src-tauri/Cargo.toml benchmark_installed_open_visual_queries --lib -- --ignored --nocapture
cargo test --manifest-path src-tauri/Cargo.toml benchmark_installed_visual_model --lib -- --ignored --nocapture
cargo test --manifest-path src-tauri/Cargo.toml benchmark_exact_visual_scan_scaling --lib -- --ignored --nocapture
```

## Remaining limitations

- The installed-artifact regression verifies token IDs, output contract, normalization, and concept
  separation. A checked-in official PyTorch reference-vector fixture with cosine `>=0.999` is still
  pending because the repository does not contain the licensed reference checkpoint/fixtures.
- Production streams validated visual blobs instead of loading all vectors at once, but SQLite still
  performs an exact scan. ANN remains deferred until a representative 100k end-to-end database
  benchmark—not only the scan kernel—misses the latency budget.
- Peak process working set and full OCR-plus-visual indexing throughput were not captured by the
  in-process benchmark and must be measured on release hardware.
- The available local labeled corpus contains five relevant files per evaluated class. It passes the
  retrieval thresholds, but an eight-per-class public-domain manifest still needs license-vetted
  assets before it can be committed.
- GIF/TIFF use the first frame/page. AVIF, HEIC, scanned-PDF page rendering, mid-ONNX cancellation,
  captioning, and VLM retrieval remain out of scope.
