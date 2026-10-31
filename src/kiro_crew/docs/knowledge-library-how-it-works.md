# Knowledge Library

Semantic search over your own documents, folders, and generated artifacts. Add a
file and its text is chunked, indexed, and searchable by meaning rather than by
exact wording, with every result citing the source it came from.

## What gets indexed

Uploaded documents and synced folders are indexed when you add them. Two more
sources are opt-in, both off by default: saved artifacts, with
`knowledge.auto_ingest_artifacts` (kinds from `knowledge.auto_ingest_artifact_kinds`,
default markdown, text, html and json), and documents the agent reads during its
work, with `knowledge.auto_add_documents`. Text is split
into overlapping chunks so a match can be quoted in context, and entities and
relations found in each chunk are linked into a graph that connects passages
across different files.

A single web page can be added too: Add Source, then Web Page, then paste an
`https://` link. Kiro Crew fetches it once itself, with no agent involved, and stores
the text, with the link and the fetch time on the source row; Sync fetches it
again. Only hosts listed in `knowledge.doc_ingest_hosts` are fetched, and that list
is empty by default, so nothing is fetched until you add a host, for example
`kirocrew config set knowledge.doc_ingest_hosts '["example.com"]'`. A host matches
exactly: `example.com` does not cover `docs.example.com`.

## What search returns

A ranked list of passages, each with a citation naming where it came from:
`**Source:**` for a document, `**File:**` for a file inside a synced folder,
`**Artifact:**` for something you saved, and `**Link:**` for a URL. Results
combine keyword and meaning-based matching, so an exact phrase and a paraphrase
both find the passage.

## Duplicates

Near-identical passages from different sources are collapsed to one result, so
uploading the same document twice — directly and through a folder sync — does not
double every hit.

## Removing documents

A file panel offers a Remove toggle (green book icon, and a "Remove from
Knowledge Library" row in the ⋯ menu) only for a document the panel itself
added, as its own per-file source. A file that is in the library through a
synced folder shows the green badge without a Remove action; remove it from the
Knowledge page instead. Removing deletes that file's source and all of the
chunks and graph entities indexed from it; the file itself stays on disk. The
panel then offers Add again, unless another source still covers the file.

The Knowledge page removes sources at larger granularity: the Knowledge
Detail view deletes a single item, and the Knowledge page supports bulk
deletion of a selection.

Deletion is permanent. There is no undo, and it cascades to everything
derived from the removed sources — indexed chunks, extracted graph entities
and their search contributions are all dropped with the source.

## Three levels of quality

Search runs three legs and fuses whatever they return, so it degrades in a defined
order rather than failing:

1. **Full** — the embedding model is loaded, so meaning-based, keyword and graph
   matching all contribute to the ranking.
2. **Keyword and graph** — no embedding model, so results come from full-text and
   graph matching. Exact wording still works; paraphrases are weaker.
3. **Keyword only** — the graph leg finds no entity matching your terms, so
   full-text matching answers alone. Keyword search is a local index with no model
   behind it, so it is always available.

When the embedding model is unavailable, the agent's search result says so: its
header, or its "No relevant knowledge found." answer, notes that only keyword
results were used.

## Settings

`knowledge.auto_ingest_artifacts` (default `false`) indexes saved artifacts and
keeps them in sync; `knowledge.auto_add_documents` (default `false`) lets the
agent add documents it reads. `knowledge.extraction_pool_size` bounds how many chunks are processed at once
(1–10; idle workers are released after five minutes). `knowledge.extraction_provider`
moves extraction to a local model (below). The embedding model can be
pointed elsewhere with `KIROCREW_EMBED_MODEL_URL` or `KIROCREW_EMBED_MODEL_PATH`.

## Running extraction on a local model

Each chunk costs one extraction call. By default that call runs on the agent
harness, which means cloud tokens. Extraction needs no tools (a chunk goes in,
a small JSON object comes out), so it can run on a model served on your own
machine instead:

```json
"knowledge": {
  "extraction_provider": "openai_compat",
  "extraction_base_url": "http://localhost:13305/api/v1",
  "extraction_endpoint_model": "<model name as your server lists it>"
}
```

Any server with an OpenAI-compatible `/chat/completions` route works, for
example Lemonade or llama.cpp's `llama-server`. Each chunk is one independent
request with thinking turned off and the reply held to a capped JSON schema, so
no earlier chunk is re-sent. To compare two models, change
`extraction_endpoint_model` and re-ingest. The new model is used from the next
ingest once the pool is idle.

A few things to know:

- **Only loopback by default.** A URL whose host is not `localhost`,
  `127.0.0.0/8` or `::1` is refused unless `knowledge.extraction_allow_remote`
  is `true`, because the chunks are your documents. A host name that happens to
  resolve to your own machine is refused too. Every allow and every refusal is
  written to the security event log.
- **No fallback to the cloud.** If the server is down, slow or returns an
  error, the request fails within `knowledge.extraction_timeout_secs` (default
  180), the rest of that document's chunks are not sent in the same pass, and
  the log names the URL. Those chunks are still indexed and searchable, just
  without entities, and nothing is sent to the harness instead. In a folder
  source, only the failed chunks are extracted again on later scans: the first
  retry waits an hour and the second two hours; after that they wait until the
  file changes.
- **Server settings matter.** Run as many parallel slots as
  `extraction_pool_size`, and give each slot enough context for a chunk and its
  reply: a request over the slot's context fails. With llama.cpp on AMD
  integrated GPUs, prefer the Vulkan build: the ROCm build has been seen
  returning text from the previous request inside the next one, which would put
  one document's content into another's entities.
- The source's topic-and-themes summary, refreshed after each ingest, runs on
  the same model without the schema. It is not refreshed when every chunk of
  the document failed, since nothing new reached it.

For how the graph itself is built and stored, see the contributor spec
`docs/system-specs/modules/knowledge.md`.
