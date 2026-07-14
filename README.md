# Recall

Recall is a Windows desktop proof of concept for private local file search. Users approve one or more folders, then Recall extracts and indexes supported files in place. OCR, embeddings, keyword retrieval, and semantic ranking run locally.

There is no account, hosted backend, telemetry service, remote inference endpoint, or production Node.js server.

## License and hackathon source

Recall is released as **AGPL-3.0-or-later** for this hackathon submission. See
[`LICENSE`](LICENSE), [`NOTICE`](NOTICE), and
[`THIRD_PARTY_NOTICES`](THIRD_PARTY_NOTICES). The complete source required to
build the demo is this repository; the Windows development and installer steps
below are the corresponding build instructions.

## POC capabilities

- Tauri 2 native Windows shell with a Next.js static-export frontend.
- Native recursive folder selection and scanning.
- SQLite persistence with FTS5 and a restart-safe indexing queue.
- TXT, Markdown, page-preserving text PDF, PNG, JPEG, and WebP extraction.
- PP-OCRv6 Tiny at a 1280px cap by default, with local multilingual-e5-small INT8 embeddings.
- Hybrid ranking: 75% cosine similarity and 25% normalized FTS5/BM25.
- Exact snippets, PDF page citations, file-type/folder filters, open, reveal, and copy-path actions.
- Explicit model setup followed by offline operation.
- Pause/resume, rescan, missing-file reconciliation, isolated failures, and persisted recovery.

Live folder watching, scanned-PDF OCR (PDF text layers are supported), generative answers,
authentication, cloud sync, and Office formats are intentionally out of scope for this milestone.

## Multimodal retrieval (optional visual search)

Recall can additionally index images with **MobileCLIP2-S0** for visual and cross-modal
search, kept in a vector space completely separate from the E5 text space.

- **Optional model.** Enable *MobileCLIP2-S0* in Privacy → Developer model lab → *Visual
  image search model*. It downloads a community ONNX export (paired vision + text encoders,
  CLIP BPE tokenizer, 512-d, ~290 MB) with pinned SHA256, and runs on the local `ort`
  runtime. Disabled by default — the app boots and searches text without it.
- **Four retrieval channels, fused by rank.** Exact text (FTS5), semantic text (E5),
  visual (MobileCLIP text→image), visual categories (zero-shot prompt bank), plus metadata
  and filename/folder signals. Channels are normalized independently and combined with
  intent-aware Reciprocal-Rank Fusion (RRF, k=60); weights vary by detected query intent.
- **Zero-shot categories.** An editable prompt bank (`src-tauri/src/visual/category_prompts.rs`)
  scores each image into its top visual categories; used as ranking boosts, never as filters.
- **Generic metadata + summaries.** Deterministic (no-LLM) extraction of dates, amounts,
  URLs, emails, phones and identifiers, plus a structured searchable summary embedded with E5.
- **Ticket-aware document evidence.** OCR/PDF text and filenames are classified locally as
  train/flight tickets, hotel bookings, invoices, or receipts; PNRs, train/flight numbers,
  booking/invoice IDs, dates, routes, and amounts are stored as typed entities. Document and
  entity matches are visible in the retrieval inspector and prevent a visual-only ticket label
  from ranking as strong evidence.
- **Tall screenshot coverage.** Whole-image vectors are retained for photo search; tall images
  also receive adaptive vertical regions, with all region hits deduplicated to the parent
  asset in results.
- **Adaptive image quality pipeline.** Images are EXIF-oriented and alpha-composited before OCR
  and visual encoding. Large, panoramic, and tall images receive deterministic aspect or pixel-grid
  regions; every region is embedded, classified, stored with source geometry, and aggregated back
  to one parent result. A visual pipeline update automatically refreshes only stale image vectors.
- **Stable identity + provenance.** SHA-256 content identities survive renames/moves within a
  watched folder. Reduced per-stage provenance records the extractor/version, source region,
  confidence, and queue job for deterministic document analysis.
- **Explainable results.** Each result shows *why* it matched; a per-query retrieval inspector
  (toggle in Search) shows per-channel ranks, scores, intents and latency.
- **Scoped reindexing.** Enabling/switching the visual model regenerates image embeddings and
  categories only — OCR and document-text embeddings are untouched.

## Windows prerequisites

Install:

1. Node.js 20 or newer.
2. Rust using `rustup` with the stable MSVC toolchain.
3. Microsoft Visual Studio 2022 Build Tools with **Desktop development with C++**.
4. Microsoft Edge WebView2 Runtime (included with current Windows versions).

Keep at least 10 GB free on the Windows system drive for the C++ workload, Windows SDK, Cargo registry, and installer tooling.

## Development

```powershell
npm install
npm test
npm run build
npm run tauri:dev
```

`npm run build` produces the static frontend in `out/`. Production does not use `next start`, API routes, Server Actions, middleware, or a Node server.

## Windows installer

```powershell
npm run tauri:build
```

Tauri is configured to build both NSIS `.exe` and WiX `.msi` packages. The installed application launches without Node.js.

## First run

1. Select **Download models**. Recall downloads the PP-OCRv6 Tiny and multilingual-e5-small INT8 packs into its application-data directory. Interrupted downloads resume automatically.
2. Choose one or more folders. Only `.txt`, `.md`, `.pdf`, `.png`, `.jpg`, `.jpeg`, and `.webp` files are considered.
3. Wait for the durable queue to finish, then search in natural language.
4. Disconnect networking and repeat searches to validate cached offline inference.

If models are missing, text documents can still be indexed for keyword search. Image jobs remain pending until OCR is installed.

## Sample corpus

Choose the [`sample-data`](sample-data) directory during onboarding. It contains safe synthetic fixtures for every supported format plus [`expected-queries.json`](sample-data/expected-queries.json) with repeatable queries and expected sources/pages.

[`hackathon-evaluation.json`](sample-data/hackathon-evaluation.json) defines the required local
regression queries, evidence expectations, and measurements to report for the hackathon demo.

Regenerate binary fixtures with:

```powershell
python scripts/generate_fixtures.py
```

## Model benchmarks

The default profile is **PP-OCRv6 Tiny**, a 1280px OCR cap, and **multilingual-e5-small INT8**. It prioritizes bulk indexing speed while the benchmark quality checks ensure that the expected text and retrieval result are still produced.

After installing the model packs, run the local-only benchmarks from `src-tauri`:

```powershell
$env:RECALL_MODEL_DIR = "$env:APPDATA\com.recall.desktop\models"
cargo test benchmark_installed_embedding_models --lib -- --ignored --nocapture
cargo test benchmark_installed_ocr_models --lib -- --ignored --nocapture
```

The OCR benchmark measures PP-OCRv6 Tiny and Small on `sample-data/restaurant-card.jpg`, checks for “Green Pepper Kitchen”, and prints median latency. The embedding benchmark measures throughput and checks that the restaurant document ranks first for its query. Install Tiny and Small through the Developer model lab before running the OCR comparison.

## Local data and security

Recall stores `recall.db`, `models/`, and `thumbnails/` beneath the Tauri application-data directory. Originals stay where they are and are never modified.

The frontend receives no generic filesystem command. Open, reveal, and copy actions accept an asset ID; Rust resolves it through SQLite, canonicalizes both paths, and verifies that the source remains under its approved folder root before performing the action.

## Architecture

```text
Next.js static UI
        |
        | typed Tauri commands and events
        v
Rust native core
  |-- approved folder scanner + SHA-256 reconciliation
  |-- persistent SQLite job queue
  |-- TXT/Markdown/PDF/image extraction
  |-- PP-OCRv6 Tiny OCR + multilingual-e5-small INT8 inference
  |-- FTS5/BM25 + in-process cosine ranking
  `-- validated Windows open/reveal/clipboard actions
```

The queue resets interrupted `processing` jobs to `pending` at startup. A failed or corrupt file is marked independently and does not terminate later jobs.
