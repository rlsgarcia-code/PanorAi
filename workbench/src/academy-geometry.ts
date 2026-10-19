import type { Point2, Vec3 } from "./model.ts";

const TAU = Math.PI * 2;

export function academyNormalize(value: Vec3): Vec3 {
  const length = Math.hypot(value.x, value.y, value.z);
  if (!(length > 1e-12)) throw new Error("A construção é degenerada.");
  return { x: value.x / length, y: value.y / length, z: value.z / length };
}

export function academyCross(a: Vec3, b: Vec3): Vec3 {
  return {
    x: a.y * b.z - a.z * b.y,
    y: a.z * b.x - a.x * b.z,
    z: a.x * b.y - a.y * b.x,
  };
}

export function academyDot(a: Vec3, b: Vec3): number {
  return a.x * b.x + a.y * b.y + a.z * b.z;
}

export function canonicalErpPixelToRay(point: Point2, shape: { width: number; height: number }): Vec3 {
  const { longitudeRad: longitude, latitudeRad: latitude } = canonicalErpPixelToLonLat(point, shape);
  const cosLatitude = Math.cos(latitude);
  return {
    x: Math.sin(longitude) * cosLatitude,
    y: Math.sin(latitude),
    z: Math.cos(longitude) * cosLatitude,
  };
}

export function canonicalErpPixelToLonLat(
  point: Point2,
  shape: { width: number; height: number },
): { longitudeRad: number; latitudeRad: number; longitudeDeg: number; latitudeDeg: number } {
  const longitudeRad = TAU * (point.x + 0.5) / shape.width - Math.PI;
  const latitudeRad = Math.PI / 2 - Math.PI * (point.y + 0.5) / shape.height;
  return {
    longitudeRad,
    latitudeRad,
    longitudeDeg: longitudeRad * 180 / Math.PI,
    latitudeDeg: latitudeRad * 180 / Math.PI,
  };
}

export function canonicalRayToErpPixel(ray: Vec3, shape: { width: number; height: number }): Point2 {
  const unit = academyNormalize(ray);
  const longitude = Math.atan2(unit.x, unit.z);
  const latitude = Math.asin(Math.max(-1, Math.min(1, unit.y)));
  return {
    x: ((longitude + Math.PI) / TAU * shape.width - 0.5 + shape.width) % shape.width,
    y: (Math.PI / 2 - latitude) / Math.PI * shape.height - 0.5,
  };
}

export function greatCircleNormal(start: Vec3, end: Vec3): Vec3 {
  return academyNormalize(academyCross(start, end));
}

export function vanishingDirection(firstNormal: Vec3, secondNormal: Vec3): Vec3 {
  return academyNormalize(academyCross(firstNormal, secondNormal));
}

export function angleDeg(a: Vec3, b: Vec3): number {
  const cosine = Math.max(-1, Math.min(1, academyDot(academyNormalize(a), academyNormalize(b))));
  return Math.acos(cosine) * 180 / Math.PI;
}

export function sampleGreatCircle(normalValue: Vec3, sampleCount = 240): Vec3[] {
  const normal = academyNormalize(normalValue);
  const helper = Math.abs(normal.y) < 0.9 ? { x: 0, y: 1, z: 0 } : { x: 1, y: 0, z: 0 };
  const first = academyNormalize(academyCross(normal, helper));
  const second = academyNormalize(academyCross(normal, first));
  return Array.from({ length: sampleCount + 1 }, (_, index) => {
    const theta = TAU * index / sampleCount;
    return {
      x: first.x * Math.cos(theta) + second.x * Math.sin(theta),
      y: first.y * Math.cos(theta) + second.y * Math.sin(theta),
      z: first.z * Math.cos(theta) + second.z * Math.sin(theta),
    };
  });
}
