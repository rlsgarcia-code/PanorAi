import { encodeMask } from "./planar-model.ts";
import type { PlaneRegion, RleMask } from "./planar-model.ts";

/** The local classifier must not silently replace a user contour by a box. */
export function validateClassifierSeeds(
  region: PlaneRegion,
  automatic: boolean,
) {
  const lastRoiEdit = [...region.history]
    .reverse()
    .find(
      (item) =>
        (item.tool === "polygon" && item.target === "roi") ||
        item.action === "prepare-segmentation",
    );
  if (
    automatic ||
    !region.roi?.runs.length ||
    (lastRoiEdit?.action === "prepare-segmentation" &&
      lastRoiEdit.roi !== "manual")
  )
    throw new Error(
      "Classificador: desenhe o limite da região antes de segmentar.",
    );
  if (
    !(region.positive ?? region.mask).runs.length ||
    !region.negative?.runs.length
  )
    throw new Error(
      "Classificador: pinte exemplos com Brush + e Brush − dentro do limite.",
    );
}

/** Explicit native ROI around the rough mask, padded by eight angular degrees.
 * Longitude is periodic; no image-color or depth-derived support is used.
 * The result is persisted in the input revision, not a hidden backend default.
 */
export function suggestSegmentationRoi(
  mask: Uint8Array,
  shape: [number, number],
  support: Uint8Array | null = null,
): RleMask {
  const [h, w] = shape;
  if (mask.length !== h * w || (support && support.length !== mask.length))
    throw new Error("Grade da ROI incompatível.");
  let low = h,
    high = -1;
  const cols = new Uint8Array(w);
  for (let y = 0; y < h; y++)
    for (let x = 0; x < w; x++)
      if (mask[y * w + x] && (!support || support[y * w + x])) {
        low = Math.min(low, y);
        high = Math.max(high, y);
        cols[x] = 1;
      }
  if (high < 0)
    throw new Error("Pinte uma estimativa da região antes de segmentar.");
  let gap = 0,
    end = 0,
    run = 0;
  for (let i = 0; i < 2 * w; i++) {
    run = cols[i % w] ? 0 : run + 1;
    if (run > gap) {
      gap = run;
      end = i;
    }
  }
  const padY = Math.max(
    Math.ceil((h * 8) / 180),
    Math.ceil((high - low + 1) * 0.15),
  );
  const span = w - gap,
    padX = Math.max(Math.ceil((w * 8) / 360), Math.ceil(span * 0.15));
  const start = (end + 1 - padX + 2 * w) % w,
    length = Math.min(w, span + 2 * padX);
  const values = new Uint8Array(mask.length);
  for (
    let y = Math.max(0, low - padY);
    y <= Math.min(h - 1, high + padY);
    y++
  ) {
    const first = Math.min(length, w - start);
    values.fill(1, y * w + start, y * w + start + first);
    if (first < length) values.fill(1, y * w, y * w + length - first);
  }
  if (support) for (let i = 0; i < values.length; i++) values[i] &= support[i];
  return encodeMask(values, shape);
}
