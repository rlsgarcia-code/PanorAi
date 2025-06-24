~~~md
<!-- ─────────────────────────────────────────────────────────────
     🗂  docs/tutorials/00_quick_start.md  ── runnable notebook
     convert to .md with Jupytext or paste as is
───────────────────────────────────────────────────────────── -->

# Quick Start 🌍 → 📦 → 🪄 → 🌍

```python
# ╔═ Install once
# !pip install panorai
```

## 1. Load a panorama

```python
from panorai.data import DataFactory
eq = DataFactory.from_file("my_pano.jpg", data_type="equirectangular")
eq.show()
```

## 2. Sample one rectilinear view

```python
face = eq.to_gnomonic(lat=30, lon=60, fov=90)
face.show()
```

## 3. Sample many views (cube)

```python
faces = eq.to_gnomonic_face_set(fov=60, sampling_method="cube")
faces[0].show()
```

## 4. Pretend we ran a neural net…

```python
# for demo just invert colours
for f in faces:
    f.data = 255 - f.data
```

## 5. Blend back

```python
pano_out = faces.to_equirectangular(eq_shape=(512,1024), blend_method="gaussian")
pano_out.show()
```

---

Run this notebook locally or online (Binder badge coming soon).  
You just completed the full projection-process-blend loop! 🎉