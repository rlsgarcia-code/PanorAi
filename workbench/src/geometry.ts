import type { Point2, SphericalFovRegion, Vec3, ViewState } from "./model.ts";

const DEG = Math.PI / 180;
const RAD = 180 / Math.PI;

export function clamp(value: number, low: number, high: number): number {
  return Math.min(high, Math.max(low, value));
}

export function wrapDegrees(value: number): number {
  return ((((value + 180) % 360) + 360) % 360) - 180;
}

export function normalize(vector: Vec3): Vec3 {
  const norm = Math.hypot(vector.x, vector.y, vector.z);
  if (norm === 0) throw new Error("Cannot normalize a zero vector.");
  return { x: vector.x / norm, y: vector.y / norm, z: vector.z / norm };
}

export function dot(a: Vec3, b: Vec3): number {
  return a.x * b.x + a.y * b.y + a.z * b.z;
}

export function cross(a: Vec3, b: Vec3): Vec3 {
  return {
    x: a.y * b.z - a.z * b.y,
    y: a.z * b.x - a.x * b.z,
    z: a.x * b.y - a.y * b.x,
  };
}

export function bearingToVector(yawDeg: number, pitchDeg: number): Vec3 {
  const yaw = yawDeg * DEG;
  const pitch = pitchDeg * DEG;
  const cosPitch = Math.cos(pitch);
  return {
    x: Math.sin(yaw) * cosPitch,
    y: Math.sin(pitch),
    z: Math.cos(yaw) * cosPitch,
  };
}

export function vectorToBearing(vector: Vec3): { yawDeg: number; pitchDeg: number } {
  const unit = normalize(vector);
  return {
    yawDeg: wrapDegrees(Math.atan2(unit.x, unit.z) * RAD),
    pitchDeg: Math.asin(clamp(unit.y, -1, 1)) * RAD,
  };
}

export function viewBasis(view: Pick<ViewState, "yawDeg" | "pitchDeg">): {
  right: Vec3;
  up: Vec3;
  forward: Vec3;
} {
  const forward = bearingToVector(view.yawDeg, view.pitchDeg);
  const yaw = view.yawDeg * DEG;
  const right = normalize({ x: Math.cos(yaw), y: 0, z: -Math.sin(yaw) });
  const up = normalize(cross(forward, right));
  return { right, up, forward };
}

export function verticalFovDeg(horizontalFovDeg: number, aspect: number): number {
  if (!(aspect > 0)) throw new Error("Aspect ratio must be positive.");
  return 2 * Math.atan(Math.tan((horizontalFovDeg * DEG) / 2) / aspect) * RAD;
}

export function viewPixelToRay(
  point: Point2,
  size: { width: number; height: number },
  view: ViewState,
): Vec3 {
  const { right, up, forward } = viewBasis(view);
  const tanHalfX = Math.tan((view.horizontalFovDeg * DEG) / 2);
  const tanHalfY = tanHalfX / (size.width / size.height);
  const planeX = ((point.x / size.width) * 2 - 1) * tanHalfX;
  const planeY = (1 - (point.y / size.height) * 2) * tanHalfY;
  return normalize({
    x: forward.x + right.x * planeX + up.x * planeY,
    y: forward.y + right.y * planeX + up.y * planeY,
    z: forward.z + right.z * planeX + up.z * planeY,
  });
}

export function rayToViewPixel(
  ray: Vec3,
  size: { width: number; height: number },
  view: ViewState,
): Point2 | null {
  const { right, up, forward } = viewBasis(view);
  const depth = dot(ray, forward);
  if (depth <= 1e-6) return null;
  const tanHalfX = Math.tan((view.horizontalFovDeg * DEG) / 2);
  const tanHalfY = tanHalfX / (size.width / size.height);
  const planeX = dot(ray, right) / depth;
  const planeY = dot(ray, up) / depth;
  return {
    x: ((planeX / tanHalfX + 1) / 2) * size.width,
    y: ((1 - planeY / tanHalfY) / 2) * size.height,
  };
}

export function rayToErpPixel(
  ray: Vec3,
  size: { width: number; height: number },
): Point2 {
  const { yawDeg, pitchDeg } = vectorToBearing(ray);
  return {
    x: ((yawDeg + 180) / 360) * size.width,
    y: ((90 - pitchDeg) / 180) * size.height,
  };
}

export function erpPixelToRay(
  point: Point2,
  size: { width: number; height: number },
): Vec3 {
  return bearingToVector(
    (point.x / size.width) * 360 - 180,
    90 - (point.y / size.height) * 180,
  );
}

export function angularDistanceDeg(a: Vec3, b: Vec3): number {
  return Math.acos(clamp(dot(normalize(a), normalize(b)), -1, 1)) * RAD;
}

export function regionFromViewRectangle(
  start: Point2,
  end: Point2,
  size: { width: number; height: number },
  view: ViewState,
): SphericalFovRegion {
  const left = Math.min(start.x, end.x);
  const right = Math.max(start.x, end.x);
  const top = Math.min(start.y, end.y);
  const bottom = Math.max(start.y, end.y);
  const center = { x: (left + right) / 2, y: (top + bottom) / 2 };
  const centerRay = viewPixelToRay(center, size, view);
  const horizontalFovDeg = angularDistanceDeg(
    viewPixelToRay({ x: left, y: center.y }, size, view),
    viewPixelToRay({ x: right, y: center.y }, size, view),
  );
  const verticalFov = angularDistanceDeg(
    viewPixelToRay({ x: center.x, y: top }, size, view),
    viewPixelToRay({ x: center.x, y: bottom }, size, view),
  );
  const bearing = vectorToBearing(centerRay);
  return {
    centerYawDeg: bearing.yawDeg,
    centerPitchDeg: bearing.pitchDeg,
    horizontalFovDeg: clamp(horizontalFovDeg, 0.25, 179),
    verticalFovDeg: clamp(verticalFov, 0.25, 179),
    rollDeg: 0,
  };
}

export function sampleRegionBoundary(region: SphericalFovRegion, samplesPerEdge = 20): Vec3[] {
  const basis = viewBasis({ yawDeg: region.centerYawDeg, pitchDeg: region.centerPitchDeg });
  const tanX = Math.tan((region.horizontalFovDeg * DEG) / 2);
  const tanY = Math.tan((region.verticalFovDeg * DEG) / 2);
  const points: Vec3[] = [];
  const add = (x: number, y: number) => {
    points.push(normalize({
      x: basis.forward.x + basis.right.x * x + basis.up.x * y,
      y: basis.forward.y + basis.right.y * x + basis.up.y * y,
      z: basis.forward.z + basis.right.z * x + basis.up.z * y,
    }));
  };
  for (let i = 0; i < samplesPerEdge; i += 1) add(-tanX + (2 * tanX * i) / samplesPerEdge, tanY);
  for (let i = 0; i < samplesPerEdge; i += 1) add(tanX, tanY - (2 * tanY * i) / samplesPerEdge);
  for (let i = 0; i < samplesPerEdge; i += 1) add(tanX - (2 * tanX * i) / samplesPerEdge, -tanY);
  for (let i = 0; i <= samplesPerEdge; i += 1) add(-tanX, -tanY + (2 * tanY * i) / samplesPerEdge);
  return points;
}
