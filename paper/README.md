# STaR-Body manuscript

The revised CVPR-format manuscript and supplementary material share editable Word masters with their directly exported PDFs.

- [Main paper PDF](STaR-Body_CVPR_2027.pdf) · [editable Word](STaR-Body_CVPR_2027.docx)
- [Supplement PDF](STaR-Body_Supplement.pdf) · [editable Word](STaR-Body_Supplement.docx)
- [Authoring sources and figures](source/)

The results are the recorded shared-protocol 3DPW validation study. The public evidence distinguishes raw C10/parent evaluations from transcribed control records. This directory is a manuscript release, not a conference submission.

The overview uses a corrected Gemini draft. Feedback and frame-ownership figures use edited OpenAI image-generation drafts; their anatomy, mesh and attention illustrations are schematic. Numerical charts use archived measurements. The main manuscript includes an LLM Usage Statement.

Files named `overview_gemini_original.jpg` and `identity_gemini_style_v2.png` are pre-edit raster bases for the SVG overlays; use the `*_final` assets for the corrected publication figures.

## Rebuild

Install `source/requirements-writing.txt` in a Python environment. On Windows with Microsoft Word installed, run `python source/build_documents.py` and then `python source/export_with_word.py` sequentially. The builder uses Office's MathML-to-OMML stylesheet to retain native editable equations. Final typography can depend on installed fonts and Word version; inspect every rendered page after rebuilding. Figure SVG sources retain editable overlays where provided; generated raster bases are not claimed to be fully vector-editable.
