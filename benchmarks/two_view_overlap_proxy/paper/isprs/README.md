# ISPRS two-column edition

This directory renders the spherical two-view evidence manuscript according to
the January 2024 ISPRS full-paper geometry:

- A4 paper;
- 25 mm top and bottom margins;
- 20 mm left and right margins;
- one-column title, author, keywords, and abstract block;
- two 82 mm body columns separated by 6 mm;
- Times New Roman, 9 pt body text;
- 100-250 word abstract and no more than six keywords;
- approximately 6-8 pages.

The current PDF is a camera-ready-style draft with the author and a neutral
affiliation line. For double-blind review, replace author and affiliation with
the event's required anonymized fields.

## Build

```bash
python benchmarks/two_view_overlap_proxy/paper/isprs/generate_isprs_figures.py
python benchmarks/two_view_overlap_proxy/paper/isprs/build_isprs_pdf.py
```

The renderer writes:

```text
output/pdf/panorai_spherical_two_view_rt_isprs.pdf
```

The numerical evidence is read from the parent paper directory and is not
modified by this layout edition.
