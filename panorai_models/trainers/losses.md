## 📌 Region-Aware Attention Map (Training Loss Modifier)

This attention mechanism modulates per-pixel loss values based on the **local structure and magnitude of prediction error**.

### 💡 Key Concepts:
- Computes an attention map from the **per-pixel error** (e.g. L1, log error).
- Uses **clamping** to avoid focusing on noisy outliers.
- Applies **spatial smoothing** (`avg_pool2d`) to spread attention across error clusters.
- Final attention weights are normalized to `[0, 1]` and used to scale loss contributions.

### 🎓 Inspired By:

- **Hard Example Mining / Error Weighting**  
  *Shrivastava et al., Online Hard Example Mining (CVPR 2016)*  
  [arXiv:1604.03540](https://arxiv.org/abs/1604.03540)

- **Structured Attention over Spatial Context**  
  *Kim et al., Structured Attention Networks (ICLR 2017)*  
  [arXiv:1702.00887](https://arxiv.org/abs/1702.00887)

- **Attention for Dense Prediction**  
  *Oktay et al., Attention U-Net (MICCAI 2018)*  
  [arXiv:1804.03999](https://arxiv.org/abs/1804.03999)

- **Loss Max-Pooling (Selective Supervision)**  
  *Bulo et al., Loss Max-Pooling (CVPR 2017)*  
  [arXiv:1612.02424](https://arxiv.org/abs/1612.02424)

- **Adaptive Weighting in Depth Estimation**  
  *Bhat et al., AdaBins (CVPR 2021)*  
  [arXiv:2011.14141](https://arxiv.org/abs/2011.14141)

This design provides a flexible, interpretable, and gradient-safe way to **emphasize structured error regions** during training.