# The per-file analysis cache: shutter-select's consumer contract

One JSON per source file lands under `<out>/cache/` (default out dir:
`<root>/_selects/`), named `<key>.json` where `key` is the first 16 hex
chars of the sha1 of the source's resolved absolute path. This file is
the interface other tools build on. The designed consumer is
[shutter-clip](https://github.com/keivanmalhani/shutter-clip)'s `rank`
subcommand, which never imports this package: it reads these files, and
runs the `shutter-select` CLI as a subprocess when they are missing or
stale.

## Evolution rules

- `schema_version` is an integer and gates every read. It is 1.
- Evolution is additive only: new fields may appear at any time, and
  consumers must ignore fields they do not know.
- Removing a field or changing a field's meaning bumps `schema_version`
  and gets a ledger entry in the project's mistake log.
- Decisions are deliberately NOT stored here. Composite percentiles are
  run-relative, so a cached decision would go stale the moment more
  footage joins the shoot. The cache holds facts about one file;
  opinions about the run are computed at emit time, every time.

## Validity: how a consumer must gate a cache file

Match shutter-select's own `load_cache` order, so a file reads as stale
here exactly when it would over there:

1. `schema_version == 1`, else reject.
2. `source.mtime` equals the file's current mtime (within 1e-6 s), else
   stale.
3. `source.size` equals the file's current byte size, else stale.

## Top level

| field | type | meaning |
| --- | --- | --- |
| `schema_version` | int | contract version, currently 1 |
| `engine` | object | `{name: "shutter-select", version}` that wrote the file |
| `generated_at` | str | UTC ISO timestamp, seconds precision |
| `source` | object | identity and probe facts, below |
| `audio` | object | file-level audio measures, below |
| `transcript` | object | whisper output, below |
| `segments` | array | one row per scored segment, below |

## `source`

| field | type | meaning |
| --- | --- | --- |
| `path` | str | resolved absolute path at analysis time |
| `name` | str | file name |
| `mtime` | float | source mtime, half of the staleness gate |
| `size` | int | source bytes, the other half |
| `duration` | float | seconds |
| `fps` | [int, int] | exact rational frame rate as [numerator, denominator]; 29.97 is [30000, 1001], never a rounded float |
| `width`, `height` | int | pixels |
| `vcodec` | str | video codec name |
| `acodec` | str or null | audio codec name |
| `has_audio` | bool | whether an audio stream exists |
| `rotation` | int | display rotation in degrees from the container |

## `audio` (file level)

Present with at least `has_audio`. When the file has decodable audio:

| field | type | meaning |
| --- | --- | --- |
| `has_audio` | bool | false when extraction failed or no stream |
| `peak_db` | float | file peak in dBFS |
| `clipped_ratio` | float | fraction of samples at or above full scale |
| `noise_floor_db` | float | 10th percentile of frame RMS, the quiet floor |
| `integrated_lufs` | float or null | EBU R128 integrated loudness via ffmpeg, null on failure |
| `silences` | array | `[start, end]` second pairs below the silence threshold |

## `transcript`

| field | type | meaning |
| --- | --- | --- |
| `language` | str | whisper auto-detect unless forced |
| `language_probability` | float | detector confidence |
| `spans` | array | `{start, end, text}` speech spans in seconds, word-tightened boundaries |
| `words` | array or null | `{start, end, word}` entries; populated only when analysis ran with `--words`, the idea-5 captions contract |

## `segments` - the scored units

Each row carries timing and class, then raw features. Raw features are
stored precisely so a consumer can re-weight without re-running
analysis; nothing here is normalized.

| field | type | meaning |
| --- | --- | --- |
| `index` | int | position within this file |
| `t_in`, `t_out`, `duration` | float | segment bounds in seconds |
| `klass` | str | `speech` or `broll` |
| `speech_ratio` | float | fraction of the segment covered by raw speech spans |
| `transcript` | str | joined take text, empty for b-roll |
| `words_per_second` | float | transcript word count over duration |
| `rms_db` | float | segment RMS level in dBFS (deliberately named rms_db, not LUFS) |
| `peak_db` | float | segment peak in dBFS |
| `clipped` | bool | clipped-sample ratio above threshold inside the segment |
| `silence_ratio` | float | overlap with file silence spans over duration |
| `noise_margin_db` | float | P90 frame RMS minus the file noise floor, a heuristic speech-over-noise margin, floored at 0 |
| `sharpness` | float | mean Laplacian variance over sampled frames |
| `mean_luma` | float | mean luma of sampled frames, 0-255 |
| `crushed_frac` | float | fraction of pixels at the black end |
| `blown_frac` | float | fraction of pixels at the white end |
| `motion` | float | mean absolute frame difference across samples |
| `frames_sampled` | int | frames the visual pass decoded; 0 means visual features are meaningless for this row |
| `face_ratio` | float or null | fraction of sampled frames with a YuNet detection; null when the faces pass did not run. Null means unmeasured, which consumers must treat as neutral, never as zero |

## Re-ranking guidance for consumers

Build your own composite from the raws: percentile-rank features within
your own pool and class, weight them your way, and treat ties as ties
(identical rows must never diverge on input order - averaged tie ranks
are the family convention). shutter-clip's `rank` does exactly this
with social weights; its source is the reference consumer
implementation.
