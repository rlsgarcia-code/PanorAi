import assert from "node:assert/strict";
import { describe, it } from "node:test";

import {
  angularDistanceDeg,
  bearingToVector,
  erpPixelToRay,
  rayToErpPixel,
  rayToViewPixel,
  regionFromViewRectangle,
  vectorToBearing,
  viewPixelToRay,
  wrapDegrees,
} from "../src/geometry.ts";
import {
  angleDeg,
  canonicalErpPixelToLonLat,
  canonicalErpPixelToRay,
  canonicalRayToErpPixel,
  greatCircleNormal,
  vanishingDirection,
} from "../src/academy-geometry.ts";

describe("spherical coordinate conversions", () => {
  it("round-trips bearings across the longitude seam", () => {
    for (const yawDeg of [-180, -179.9, -90, 0, 90, 179.9]) {
      for (const pitchDeg of [-89, -45, 0, 45, 89]) {
        const bearing = vectorToBearing(bearingToVector(yawDeg, pitchDeg));
        assert.ok(Math.abs(wrapDegrees(bearing.yawDeg - yawDeg)) < 1e-10);
        assert.ok(Math.abs(bearing.pitchDeg - pitchDeg) < 1e-10);
      }
    }
  });

  it("round-trips ERP pixel centres", () => {
    const size = { width: 2048, height: 1024 };
    const samples = [
      { x: 0.5, y: 0.5 },
      { x: 1024, y: 512 },
      { x: 2047.5, y: 1023.5 },
    ];
    for (const sample of samples) {
      const restored = rayToErpPixel(erpPixelToRay(sample, size), size);
      assert.ok(Math.abs(restored.x - sample.x) < 1e-9);
      assert.ok(Math.abs(restored.y - sample.y) < 1e-9);
    }
  });
});

describe("gnomonic view geometry", () => {
  const size = { width: 800, height: 500 };
  const view = { yawDeg: 42, pitchDeg: -17, horizontalFovDeg: 70 };

  it("maps the view centre to its bearing", () => {
    const ray = viewPixelToRay({ x: 400, y: 250 }, size, view);
    const bearing = vectorToBearing(ray);
    assert.ok(Math.abs(bearing.yawDeg - 42) < 1e-10);
    assert.ok(Math.abs(bearing.pitchDeg + 17) < 1e-10);
  });

  it("round-trips visible pixels", () => {
    for (const point of [{ x: 100, y: 90 }, { x: 400, y: 250 }, { x: 700, y: 410 }]) {
      const restored = rayToViewPixel(viewPixelToRay(point, size, view), size, view);
      assert.notEqual(restored, null);
      assert.ok(Math.abs((restored?.x ?? 0) - point.x) < 1e-9);
      assert.ok(Math.abs((restored?.y ?? 0) - point.y) < 1e-9);
    }
  });

  it("creates a finite spherical FOV from a drawn rectangle", () => {
    const region = regionFromViewRectangle({ x: 250, y: 150 }, { x: 550, y: 350 }, size, view);
    assert.ok(Math.abs(region.centerYawDeg - view.yawDeg) < 1e-10);
    assert.ok(Math.abs(region.centerPitchDeg - view.pitchDeg) < 1e-10);
    assert.ok(region.horizontalFovDeg > 20);
    assert.ok(region.horizontalFovDeg < view.horizontalFovDeg);
    assert.ok(region.verticalFovDeg > 10);
    assert.equal(region.rollDeg, 0);
  });

  it("measures antipodal rays as 180 degrees", () => {
    assert.ok(Math.abs(angularDistanceDeg({ x: 0, y: 0, z: 1 }, { x: 0, y: 0, z: -1 }) - 180) < 1e-10);
  });
});

describe("academy single-view constructions", () => {
  it("matches the PanorAi pixel-centre convention at the ERP front ray", () => {
    const ray = canonicalErpPixelToRay({ x: 3.5, y: 1.5 }, { width: 8, height: 4 });
    assert.ok(Math.abs(ray.x) < 1e-12);
    assert.ok(Math.abs(ray.y) < 1e-12);
    assert.ok(Math.abs(ray.z - 1) < 1e-12);
    const pixel = canonicalRayToErpPixel(ray, { width: 8, height: 4 });
    assert.ok(Math.abs(pixel.x - 3.5) < 1e-12);
    assert.ok(Math.abs(pixel.y - 1.5) < 1e-12);
  });

  it("maps the ERP centre to zero longitude and latitude", () => {
    const angles = canonicalErpPixelToLonLat({ x: 3.5, y: 1.5 }, { width: 8, height: 4 });
    assert.ok(Math.abs(angles.longitudeDeg) < 1e-12);
    assert.ok(Math.abs(angles.latitudeDeg) < 1e-12);
  });

  it("intersects two interpretation planes at a vanishing direction", () => {
    const direction = { x: 1, y: 0, z: 0 };
    const first = greatCircleNormal(direction, { x: 0, y: 0, z: 1 });
    const second = greatCircleNormal(direction, { x: 0, y: 1, z: 0 });
    const vanishing = vanishingDirection(first, second);
    assert.ok(Math.abs(Math.abs(vanishing.x) - 1) < 1e-12);
    assert.ok(Math.abs(vanishing.y) < 1e-12);
    assert.ok(Math.abs(vanishing.z) < 1e-12);
  });

  it("reports orthogonal directions as ninety degrees", () => {
    assert.ok(Math.abs(angleDeg({ x: 1, y: 0, z: 0 }, { x: 0, y: 0, z: 1 }) - 90) < 1e-12);
  });
});
