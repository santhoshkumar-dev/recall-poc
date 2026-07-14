# Recall

Recall is a Windows desktop proof of concept for private local file search. Users approve one or more folders, then Recall extracts and indexes supported files in place. OCR, embeddings, keyword retrieval, and semantic ranking run locally.

There is no account, hosted backend, telemetry service, remote inference endpoint, or production Node.js server.

## POC capabilities

- Tauri 2 native Windows shell with a Next.js static-export frontend.
- Native recursive folder selection and scanning.
- SQLite persistence with FTS5 and a restart-safe indexing queue.
- TXT, Markdown, page-preserving text PDF, PNG, JPEG, and WebP extraction.
- English OCR with `ocrs` and local embeddings with FastEmbed `all-MiniLM-L6-v2`.
- Hybrid ranking: 75% cosine similarity and 25% normalized FTS5/BM25.
- Exact snippets, PDF page citations, file-type/folder filters, open, reveal, and copy-path actions.
- Explicit model setup followed by offline operation.
- Pause/resume, rescan, missing-file reconciliation, isolated failures, and persisted recovery.

Live folder watching, scanned-PDF OCR, generative answers, authentication, cloud sync, and Office formats are intentionally out of scope for this milestone.

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

1. Select **Download models**. Recall downloads two compact OCR model files and the FastEmbed model into its application-data directory.
2. Choose one or more folders. Only `.txt`, `.md`, `.pdf`, `.png`, `.jpg`, `.jpeg`, and `.webp` files are considered.
3. Wait for the durable queue to finish, then search in natural language.
4. Disconnect networking and repeat searches to validate cached offline inference.

If models are missing, text documents can still be indexed for keyword search. Image jobs remain pending until OCR is installed.

## Sample corpus

Choose the [`sample-data`](sample-data) directory during onboarding. It contains safe synthetic fixtures for every supported format plus [`expected-queries.json`](sample-data/expected-queries.json) with repeatable queries and expected sources/pages.

Regenerate binary fixtures with:

```powershell
python scripts/generate_fixtures.py
```

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
  |-- ocrs OCR + FastEmbed ONNX inference
  |-- FTS5/BM25 + in-process cosine ranking
  `-- validated Windows open/reveal/clipboard actions
```

The queue resets interrupted `processing` jobs to `pending` at startup. A failed or corrupt file is marked independently and does not terminate later jobs.
