# Fig. 5: robustness comparison under ten attack types

Run `python generate_fig5.py` with matplotlib, numpy, and pandas installed.
The script reads the bundled CSVs and writes PDF, SVG, 600 dpi PNG/TIFF,
a preview, a long-form source-data table, and an audit JSON under `output/`.

This is a restyling of the archived experiment results, not a new model evaluation.
It preserves the original panel order and displayed conditions: vertex addition
and noise show 10%, 50%, and 90% at each of their three strengths. No numerical
values are changed; scaling percentages are displayed as equivalent scale factors.
There are 385 finite values and 17 unavailable values. Missing Lin18 values in
four attack panels remain absent, without interpolation or substituted results.

Style: 5-by-2 grid, DejaVu Sans, distinct method colors and markers, red VGCN-B
curve, left-aligned bold panel titles, left/bottom spines, light horizontal grid,
and one shared legend. SVG text remains editable and PDF fonts are embedded.

The dashed line is the original nominal NC level of 0.90. It is not a claim of
validated ownership-verification accuracy. Only method-level average values are
available in the source tables, so no uncertainty bands or invented replicates
are shown. The manuscript caption supplies the interpretation of the figure.
