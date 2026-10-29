# STaR-Body manuscript

The revised CVPR-format manuscript and supplementary material share editable Word masters with their directly exported PDFs.

- [Main paper PDF](STaR-Body_CVPR_2027.pdf) · [editable Word](STaR-Body_CVPR_2027.docx)
- [Supplement PDF](STaR-Body_Supplement.pdf) · [editable Word](STaR-Body_Supplement.docx)
- [Authoring sources and figures](source/)

The paper includes 20 source-verified published baseline settings from 14 methods, the archived full-model result, matched feedback controls, per-sequence analysis, and EMA diagnostics. The benchmark table records each row's split; the full-model numbers remain the archived 3DPW validation measurements. Source values and protocol variants are preserved in `source/published_benchmark_rows.json`. This directory is a manuscript revision, not a conference submission.

The overview preserves the user's dense Gemini reference, revised with OpenAI image generation and precise connectivity corrections. Feedback and frame-ownership figures also use edited OpenAI image-generation drafts; their anatomy, mesh and attention illustrations are schematic. Numerical charts use archived measurements. The main manuscript includes an LLM Usage Statement.

Files named `overview_reference_user.png`, `overview_gemini_original.jpg`, `overview_dense_generated.png` and `identity_gemini_style_v2.png` are reference or pre-edit raster artwork; use the `*_final` assets for the corrected publication figures. The overview generation prompts and its provenance are recorded in `source/figures/overview_generation_record.json`.

## Rebuild

Install `source/requirements-writing.txt` in a Python environment. On Windows with Microsoft Word installed, run `python source/build_documents.py` and then `python source/export_with_word.py` sequentially. The builder uses Office's MathML-to-OMML stylesheet to retain native editable equations. Final typography can depend on installed fonts and Word version; inspect every rendered page after rebuilding. Figure SVG sources retain editable overlays where provided; generated raster bases are not claimed to be fully vector-editable.
