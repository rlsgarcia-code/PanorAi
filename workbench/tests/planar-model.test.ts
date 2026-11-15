import assert from "node:assert/strict";
import test from "node:test";
import {
  angularDistanceDeg,
  bearingToVector,
  viewPixelToRay,
} from "../src/geometry.ts";
import {
  decodeMask,
  encodeMask,
  paintDisc,
  polygonMask,
  brushOutline,
  planarToolCursor,
} from "../src/planar-model.ts";
test("tool cursors distinguish navigation, painting and erasing", () => {
  assert.equal(planarToolCursor("navigate"), "grab");
  assert.match(planarToolCursor("positive"), /data:image\/svg\+xml/);
  assert.notEqual(planarToolCursor("positive"), planarToolCursor("negative"));
  assert.equal(planarToolCursor("polygon"), "crosshair");
});
test("brush preview uses the actual spherical radius, including off-axis and polar views", () => {
  const size = { width: 680, height: 430 };
  for (const view of [
    { yawDeg: 0, pitchDeg: 0, horizontalFovDeg: 72 },
    { yawDeg: 179, pitchDeg: 85, horizontalFovDeg: 110 },
  ])
    for (const point of [
      { x: 340, y: 215 },
      { x: 80, y: 40 },
    ]) {
      const center = viewPixelToRay(point, size, view);
      for (const radius of [0.1, 1, 8]) {
        const outline = brushOutline(point, size, view, radius);
        assert.equal(outline.length, 64);
        for (const pixel of outline)
          assert.ok(
            Math.abs(
              angularDistanceDeg(center, viewPixelToRay(pixel, size, view)) -
                radius,
            ) < 1e-8,
          );
      }
    }
});
test("RLE roundtrip preserves exact native masks and rejects overlapping runs", () => {
  const source = Uint8Array.from([1, 1, 0, 0, 1, 0, 1, 1]);
  assert.deepEqual(decodeMask(encodeMask(source, [2, 4])), source);
  assert.throws(() => decodeMask({ shape_hw: [2, 4], runs: [0, 3, 2, 2] }));
});
test("spherical brush crosses the seam without painting the opposite meridian", () => {
  const mask = new Uint8Array(72 * 144);
  paintDisc(mask, [72, 144], bearingToVector(179, 0), 8);
  assert.ok(mask[36 * 144]);
  assert.ok(mask[36 * 144 + 143]);
  assert.equal(mask[36 * 144 + 72], 0);
});
test("spherical brush covers the north polar ring", () => {
  const mask = new Uint8Array(180 * 360);
  paintDisc(mask, [180, 360], bearingToVector(0, 90), 3);
  assert.equal(
    mask.slice(0, 360).reduce((a, b) => a + b, 0),
    360,
  );
  assert.equal(mask[90 * 360 + 180], 0);
});
test("polygon stays seam safe on the canonical native grid", () => {
  const mask = polygonMask(
    [72, 144],
    [
      { x: 20, y: 20 },
      { x: 80, y: 20 },
      { x: 80, y: 80 },
      { x: 20, y: 80 },
    ],
    { yawDeg: 180, pitchDeg: 0, horizontalFovDeg: 70 },
    { width: 100, height: 100 },
  );
  assert.equal(mask[36 * 144], 1);
  assert.equal(mask[36 * 144 + 143], 1);
  assert.equal(mask[36 * 144 + 72], 0);
});
