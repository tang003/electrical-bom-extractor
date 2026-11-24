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
   - By default, NEW projects use **`SCHEMATIC_MODE` with `independent-schematic-v1`**. Detect embedded parts-table pages/regions and mask BOTH pixels and native/OCR evidence into `schematic_only.pdf`. Keep page indexes and exclusion masks for review. Never send the original table to schematic AI.
   - Extract the schematic BOM first and persist `independent_snapshot.json` BEFORE reading the original parts list. Store the table separately as `declared_parts`; compare full scoped tags, component roles, model/parameters, quantities and units in `reconciliation`. Unknown lengths/quantities stay null.
   - Historical `PART_LIST_MODE` projects are read-only with respect to machine reanalysis; clone through the independent-analysis endpoint. Preserve existing engineer edits. Human adoption of a table model only changes the working BOM and explicit provenance, never the independent snapshot. AI caches use masked-view hash plus agent version, and missing masks fail closed.
   - Verify table isolation visually for each unfamiliar layout. Automated detection is not a guarantee of arbitrary table formats. Ground-truth engineer BOMs remain evaluation data, never extraction inputs or catalogue rules.

   - Explicit **`TABLE_EXTRACT`** uses `PART_LIST_MODE` with `parts-table-only-v1` and `summary.table_extraction`. Read literal table rows without schematic analysis, engineering additions, automatic AI or quantity filtering. Preserve blanks, zeroes, grouped tags and raw fields. Different table pages remain separate. Missing tables do not trigger a schematic fallback. Export clearly labels “未经原理图核对”. This explicit mode is distinct from historical table-led projects.
   - Mode changes create new projects. Preserve all original engineer work. `POST /api/projects/{id}/table-extraction` accepts optional page selection and reuses the original view. Only physically coincident CAD borders with matching native entities may be deduplicated; do not deduplicate separate pages or versions merely because their text matches.

2. **Stage 2: Deterministic Engineering Rules & Anomaly Detection (`rules.py`)**
   - Filters non-procurement field placeholders (`DT1~17`, `SP1~8`, quantity `0` items like `-SPD`) into a recoverable `filtered_items` list.
   - Stores enterprise accessory and panel-series choices in pending `selection_suggestions`, without rewriting drawing models. Never derive extraction rules from the engineer answer sheet. Preserve raw evidence when humans accept a selection.
   - Retains terminal/cable candidates and unresolved component relationships. Unknown quantities, repeated-template totals and cable lengths remain null, never implicit zero or one.
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
- `schematic_only.pdf`, `isolation_plan.json`: Masked AI view and excluded/mixed-page masks.
- `independent_snapshot.json`, `declared_parts.json`: Independently saved facts and literal table reference.
- `crops/`: High-DPI PNG crops of non-text / CAD vector-outline regions.
- `<stem>_提取结果.json`: Complete JSON with `itemized_bom`, `aggregated_bom`, `filtered_items`, and `review_items`.
- `<stem>_标准BOM清单.xlsx`: Compact employee Excel: one materials sheet, plus unresolved issues only when applicable. Full provenance remains in the separate diagnostics export, never claim automatic extraction is approved procurement.

### 3. Run Accuracy Evaluation (When Ground-Truth BOM is Provided)
Compare the extraction output against a human standard Excel BOM:
```powershell
python .agents/skills/electrical-bom-extractor/scripts/extract_bom.py --pdf "<path/to/drawing.pdf>" --ground-truth "<path/to/standard_bom.xlsx>"
```


- Default Excel exports are compact: `物料清单` only in `TABLE_EXTRACT`, and optionally `待确认事项` in schematic-check mode. Use `export_to_excel(..., diagnostic=True)` or `/export/excel/diagnostics` for raw fields, full snapshots, evidence, page diagnostics and task history. Neither export may mutate saved facts or engineer decisions.

## Domain Rules Reference
For detailed electrical design rules (handling `KM1` vs `KM1-1`/`KM1-2`, coil/contact deduplication, terminal strips, and explicit vs inferred items), read [electrical_rules.md](./references/electrical_rules.md).
