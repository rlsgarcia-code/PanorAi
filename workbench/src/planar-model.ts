import {
  cross,
  dot,
  normalize,
  rayToViewPixel,
  vectorToBearing,
  viewPixelToRay,
} from "./geometry.ts";
import type { Point2, Vec3, ViewState } from "./model.ts";
export type PlanarTool =
  "navigate" | "zoom" | "positive" | "negative" | "polygon" | "tie";

export function planarToolCursor(tool: PlanarTool): string {
  if (tool === "navigate") return "grab";
  if (tool === "zoom") return "zoom-in";
  if (tool !== "positive" && tool !== "negative") return "crosshair";
  const color = tool === "positive" ? "#ffc857" : "#ff6e79";
  const vertical = tool === "positive" ? "M16 11v10" : "";
  const svg = `<svg xmlns="http://www.w3.org/2000/svg" width="32" height="32"><circle cx="16" cy="16" r="13" fill="#121b1290" stroke="${color}" stroke-width="2"/><path d="M11 16h10${vertical}" fill="none" stroke="${color}" stroke-width="2"/></svg>`;
  return `url("data:image/svg+xml,${encodeURIComponent(svg)}") 16 16, crosshair`;
}

/** Project the same spherical-cap boundary used by paintDisc into the view. */
export function brushOutline(
  point: Point2,
  size: { width: number; height: number },
  view: ViewState,
  radiusDeg: number,
): Point2[] {
  const ray = viewPixelToRay(point, size, view);
  const tangent = normalize(
    cross(
      ray,
      Math.abs(ray.y) < 0.9 ? { x: 0, y: 1, z: 0 } : { x: 1, y: 0, z: 0 },
    ),
  );
  const bitangent = cross(ray, tangent);
  const radius = (radiusDeg * Math.PI) / 180;
  return Array.from({ length: 64 }, (_, i) => {
    const angle = (2 * Math.PI * i) / 64;
    const a = Math.sin(radius) * Math.cos(angle),
      b = Math.sin(radius) * Math.sin(angle);
    return rayToViewPixel(
      {
        x: ray.x * Math.cos(radius) + tangent.x * a + bitangent.x * b,
        y: ray.y * Math.cos(radius) + tangent.y * a + bitangent.y * b,
        z: ray.z * Math.cos(radius) + tangent.z * a + bitangent.z * b,
      },
      size,
      view,
    );
  }).filter((p): p is Point2 => p !== null);
}
export interface RleMask {
  shape_hw: [number, number];
  runs: number[];
}
export interface PlanarView {
  view_id: string;
  image_key: string;
  rotation: number[][];
  pose_convention: "world_to_camera" | "camera_to_world";
  center_world?: number[];
  translation?: number[];
  support: "full" | RleMask;
  baseline_range?: number[][];
  baseline_validity?: RleMask;
}
export interface PlaneRegion {
  plane_id: string;
  view_id: string;
  label: string;
  state: "draft" | "accepted" | "rejected" | "needs-review";
  mask: RleMask;
  positive?: RleMask;
  negative?: RleMask;
  roi?: RleMask;
  interpretation: "seed" | "final";
  history: Record<string, unknown>[];
  source: string;
}
export interface Correspondence {
  plane_id: string;
  track_id: string;
  source: string;
  observations: {
    view_id: string;
    pixel_xy: [number, number];
  }[];
}
export interface PlanarDocument {
  schema: "panorai-planar-scene/v1";
  geometry_convention: "panorai-geometry-v1";
  units: "m";
  scene_id?: string;
  revision?: string;
  views: PlanarView[];
  regions: PlaneRegion[];
  correspondences: Correspondence[];
}
export function encodeMask(
  values: Uint8Array,
  shape: [number, number],
): RleMask {
  if (values.length !== shape[0] * shape[1])
    throw new Error("Grade da máscara incompatível.");
  const runs: number[] = [];
  for (let i = 0; i < values.length;) {
    if (!values[i]) {
      i++;
      continue;
    }
    const start = i;
    while (i < values.length && values[i]) i++;
    runs.push(start, i - start);
  }
  return { shape_hw: shape, runs };
}
export function decodeMask(mask: RleMask): Uint8Array {
  const size = mask.shape_hw[0] * mask.shape_hw[1];
  if (
    !Number.isSafeInteger(size) ||
    size < 1 ||
    size > 160000000 ||
    mask.runs.length % 2
  )
    throw new Error("Máscara inválida.");
  const result = new Uint8Array(size);
  let previous = 0;
  for (let i = 0; i < mask.runs.length; i += 2) {
    const start = mask.runs[i],
      length = mask.runs[i + 1];
    if (
      !Number.isInteger(start) ||
      !Number.isInteger(length) ||
      start < previous ||
      length < 1 ||
      start + length > size
    )
      throw new Error("RLE inválido.");
    result.fill(1, start, start + length);
    previous = start + length;
  }
  return result;
}
/** A circular brush is a spherical cap, including wrap and pole coverage. */
export function paintDisc(
  values: Uint8Array,
  shape: [number, number],
  direction: Vec3,
  radiusDeg: number,
  value = 1,
  support: Uint8Array | null = null,
): void {
  const ray = normalize(direction),
    { yawDeg, pitchDeg } = vectorToBearing(ray);
  const [height, width] = shape,
    radius = (radiusDeg * Math.PI) / 180,
    cosine = Math.cos(radius);
  const minY = Math.max(
    0,
    Math.floor(((90 - pitchDeg - radiusDeg) / 180) * height - 0.5),
  );
  const maxY = Math.min(
    height - 1,
    Math.ceil(((90 - pitchDeg + radiusDeg) / 180) * height - 0.5),
  );
  const longitudeRadius =
    Math.abs(pitchDeg) + radiusDeg >= 89.99
      ? 180
      : (Math.asin(
          Math.min(1, Math.sin(radius) / Math.cos((pitchDeg * Math.PI) / 180)),
        ) *
          180) /
        Math.PI;
  const centerX = ((yawDeg + 180) / 360) * width - 0.5;
  const span = Math.min(
    width,
    Math.ceil((longitudeRadius / 360) * width) * 2 + 4,
  );
  const start = Math.floor(centerX - span / 2);
  for (let y = minY; y <= maxY; y++) {
    const lat = Math.PI / 2 - ((y + 0.5) / height) * Math.PI;
    for (let i = 0; i < span; i++) {
      const x = (((start + i) % width) + width) % width;
      const lon = ((x + 0.5) / width) * 2 * Math.PI - Math.PI;
      const candidate = {
        x: Math.cos(lat) * Math.sin(lon),
        y: Math.sin(lat),
        z: Math.cos(lat) * Math.cos(lon),
      };
      if (
        (!support || support[y * width + x]) &&
        dot(ray, candidate) >= cosine - 1e-12
      )
        values[y * width + x] = value;
    }
  }
}
function inside(point: Point2, polygon: Point2[]): boolean {
  let result = false;
  for (let i = 0, j = polygon.length - 1; i < polygon.length; j = i++) {
    const a = polygon[i],
      b = polygon[j];
    if (
      a.y > point.y !== b.y > point.y &&
      point.x < ((b.x - a.x) * (point.y - a.y)) / (b.y - a.y) + a.x
    )
      result = !result;
  }
  return result;
}
/** Rasterize only pixels visible in the frozen editing view, at native centres. */
export function polygonMask(
  shape: [number, number],
  polygon: Point2[],
  view: ViewState,
  size: {
    width: number;
    height: number;
  },
): Uint8Array {
  if (polygon.length < 3)
    throw new Error("O polígono precisa de três vértices.");
  const [height, width] = shape,
    result = new Uint8Array(height * width);
  for (let y = 0; y < height; y++) {
    const lat = Math.PI / 2 - ((y + 0.5) * Math.PI) / height;
    for (let x = 0; x < width; x++) {
      const lon = ((x + 0.5) * 2 * Math.PI) / width - Math.PI;
      const pixel = rayToViewPixel(
        {
          x: Math.cos(lat) * Math.sin(lon),
          y: Math.sin(lat),
          z: Math.cos(lat) * Math.cos(lon),
        },
        size,
        view,
      );
      if (pixel && inside(pixel, polygon)) result[y * width + x] = 1;
    }
  }
  return result;
}
