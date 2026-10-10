# Spherical two-view R,t evidence paper

This directory contains the editable manuscript, anonymized aggregate evidence,
reproducible scientific figures, and PDF renderer for the PanorAi spherical
two-view R,t study. It contains no dataset media, registered clouds, or private
dataset identifiers.

## Build

The figure generator requires Matplotlib and NumPy. The PDF renderer requires
ReportLab and Pillow.

```bash
python benchmarks/two_view_overlap_proxy/paper/generate_figures.py
python benchmarks/two_view_overlap_proxy/paper/build_pdf.py
```

The final PDF is written to:

```text
output/pdf/panorai_spherical_two_view_rt_evidence.pdf
```

For visual verification, render every page with Poppler and inspect the PNGs:

```bash
pdftoppm -png -r 120 \
  output/pdf/panorai_spherical_two_view_rt_evidence.pdf \
  /tmp/panorai-rt-paper-page
```

## Evidence boundary

The manuscript reports retrospective results and controlled timing. It does
not claim prospective release reliability, metric translation observability,
dense-stereo performance, bundle adjustment, or multiview estimation.
