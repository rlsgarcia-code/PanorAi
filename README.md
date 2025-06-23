# **PanorAi: Spherical Image Processing & Projection**

**PanorAi** is a framework for working with **spherical (equirectangular) images**, enabling efficient transformation into **Gnomonic projections** and back to equirectangular format. It provides flexible **samplers** and **blenders** to optimize projection and reconstruction processes.

---

## **🚀 Quick Start**

### **Installation**
```bash
pip install panorai
```

### **1️⃣ Load an Equirectangular Image**
Convert an image to an **EquirectangularImage** object.
```python
from panorai.data import DataFactory

eq_image = DataFactory.from_file("path/to/image.png", data_type="equirectangular")
```

---

## **📌 Core Functions**

### **2️⃣ Convert to Gnomonic Projection**
Extract a **rectilinear (Gnomonic) face** from the equirectangular image.
```python
face = eq_image.to_gnomonic(lat=45, lon=90, fov=60)
face.show()
```

### **3️⃣ Convert Back to Equirectangular**
Reproject a gnomonic face back to equirectangular.
```python
eq_reprojected = face.to_equirectangular(eq_shape=(512, 1024))
eq_reprojected.show()
```

---

## **🛠️ Advanced Usage**

### **4️⃣ Convert to Multiple Gnomonic Faces**
Use **sampling strategies** (e.g., `"cube"`, `"fibonacci"`) to extract multiple faces.
```python
face_set = eq_image.to_gnomonic_face_set(fov=60, sampling_method="cube")
face_set[0].show()  # View first face
```

### **5️⃣ Reconstruct Using a Blender**
Back-project multiple faces using different blending methods (`"closest"`, `"average"`).
```python
eq_reconstructed = face_set.to_equirectangular(eq_shape=(512, 1024), blender_name="closest")
eq_reconstructed.show()
```

---

## **🔧 Configuring Samplers & Blenders**
You can **fine-tune sampling & blending strategies** using `ConfigManager`.

### **Set Custom Sampler**
```python
from panorai.samplers.config import SamplerConfig

sampler_config = SamplerConfig(n_points=5)
```

### **Select Blender**
```python
from panorai.blenders.registry import BlenderRegistry

blender = BlenderRegistry.get("average")  # Options: "closest", "average", etc.
```

---


## **📌 Summary**
| Feature                 | Function |
|-------------------------|----------|
| Load Image              | `DataFactory.from_file()` |
| Convert to Gnomonic     | `to_gnomonic(lat, lon, fov)` |
| Convert to Face Set     | `to_gnomonic_face_set(fov, sampling_method)` |
| Convert Back to EQ      | `to_equirectangular(eq_shape, blender_name)` |
| Use Samplers & Blenders | `ConfigManager`, `BlenderRegistry` |
---

## **📚 Next Steps**
- Experiment with **different samplers (`"cube"`, `"fibonacci"`)**.
- Try **blenders (`"closest"`, `"average"`)** for optimal reconstructions.
- Use **Torch tensors** for deep learning integration.

🔗 **[PanorAi Documentation](#)** (Link to full API reference)

---