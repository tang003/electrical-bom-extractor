---
name: electrical-bom-extractor
description: >-
  Extract structured BOM (Excel + JSON), bounding-box coordinate evidence, and anomaly review reports
  from industrial electrical control cabinet schematic PDFs (both EPLAN drawings with embedded Part Lists
  and pure CAD schematics). Use this skill whenever the user wants to parse electrical schematic PDFs into
  a BOM, verify schematic-vs-BOM consistency, or evaluate BOM extraction accuracy.
---

# Industrial Electrical Schematic BOM Extractor (`electrical-bom-extractor`)

This skill orchestrates the decoupled `bom_extractor` Python core engine to extract high-accuracy Bills of Materials (BOM) from industrial electrical schematic PDFs without relying on a monolithic single-prompt LLM approach.

## Core Architecture & Dual-Mode Pipeline

1. **Stage 1: Page Structure & Mode Classification (`classifier.py`)**
   - Normalizes PDF page rotation (`0°/90°/180°/270°` via `page.rotation_matrix`) so coordinates are always in visual top-left `(0, 0)` to bottom-right `(width, height)` space.
   - Classifies every page (`COVER`, `CABINET_LAYOUT`, `INSTALL_LAYOUT`, `MAIN_CIRCUIT`, `CONTROL_CIRCUIT`, `PLC_WIRING`, `INSTRUMENT_WIRING`, `TERMINAL_DIAGRAM`, `PART_LIST`).
   - Routes the PDF to either:
     - **`PART_LIST_MODE`** (when EPLAN `部件列表` pages exist, e.g. `双钱L9_20261020.pdf` Pages 21–28): Extracts tabular rows by coordinate grid + cross-verifies against schematic & layout pages (Pages 2–20).
     - **`SCHEMATIC_MODE`** (pure CAD schematics without embedded part lists, e.g. `C26368.pdf`): Extracts native text/tags/models spatially, crops vector-outline regions (`crops/*.png`) for VLM recognition, deduplicates coils/contacts across pages, and infers symmetric circuit topology (e.g., 45KW Star-Delta motor branches).

2. **Stage 2: Deterministic Engineering Rules & Anomaly Detection (`rules.py`)**
   - Filters non-procurement field placeholders (`DT1~17`, `SP1~8`, quantity `0` items like `-SPD`) into a recoverable `filtered_items` list.
   - Appends mandatory PLC/breaker accessories (`+6ES7592-1BM00-0XB0`, `+6ES7590-1AB60-0AA0+6ES7954-8LL04-0AA0`, `+EZDFASB2`, `+GVAE11`) and standard panel series mappings (`XB2` indicator lights/buttons, `RXM2` relays).
   - Flags conflicts & omissions (`MISSING_IN_PART_LIST`, `PARAM_MISMATCH`, `BRAND_TYPO`, `LAYOUT_MISMATCH`) with exact page and normalized `bbox` coordinates.

## Standard Execution Steps

### 1. Run Extraction on a Single PDF
Run the CLI helper script [extract_bom.py](./scripts/extract_bom.py):
```powershell
python .agents/skills/electrical-bom-extractor/scripts/extract_bom.py --pdf "<path/to/drawing.pdf>" --output-dir "output/<project_name>"
```

### 2. Inspect Intermediate Artifacts
Before finalizing or when debugging a new customer drawing, inspect the generated intermediate files in `output/<project_name>/`:
- `stage1_page_structure.json`: Page types, text block counts, and vector-text warnings.
- `stage2_raw_extracted.json`: Raw extracted components prior to rule filtering.
- `crops/`: High-DPI PNG crops of non-text / CAD vector-outline regions.
- `<stem>_提取结果.json`: Complete JSON with `itemized_bom`, `aggregated_bom`, `filtered_items`, and `review_items`.
- `<stem>_标准BOM清单.xlsx`: Multi-sheet styled Excel BOM ready for procurement.

### 3. Run Accuracy Evaluation (When Ground-Truth BOM is Provided)
Compare the extraction output against a human standard Excel BOM:
```powershell
python .agents/skills/electrical-bom-extractor/scripts/extract_bom.py --pdf "<path/to/drawing.pdf>" --ground-truth "<path/to/standard_bom.xlsx>"
```

## Domain Rules Reference
For detailed electrical design rules (handling `KM1` vs `KM1-1`/`KM1-2`, coil/contact deduplication, terminal strips, and explicit vs inferred items), read [electrical_rules.md](./references/electrical_rules.md).
