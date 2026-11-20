import assert from "node:assert/strict";
import test from "node:test";
import {
  suggestSegmentationRoi,
  validateClassifierSeeds,
} from "../src/planar-workflow.ts";
import type { PlaneRegion } from "../src/planar-model.ts";
import { decodeMask } from "../src/planar-model.ts";

test("quick segmentation ROI contains the rough seed with surrounding background", () => {
  const shape: [number, number] = [180, 360],
    seed = new Uint8Array(180 * 360);
  seed[90 * 360 + 180] = 1;
  const roi = decodeMask(suggestSegmentationRoi(seed, shape));
  assert.equal(roi[90 * 360 + 180], 1);
  assert.equal(roi[85 * 360 + 175], 1);
  assert.equal(roi[90 * 360 + 100], 0);
  assert.ok(roi.reduce((a, b) => a + b, 0) > 1);
});

test("classifier requires both classes and never silently reuses an automatic ROI", () => {
  const mask = { shape_hw: [10, 20] as [number, number], runs: [30, 3] };
  const region: PlaneRegion = {
    plane_id: "floor",
    view_id: "a",
    mask,
    positive: mask,
    negative: { ...mask, runs: [50, 3] },
    roi: mask,
    state: "draft",
    label: "Floor",
    interpretation: "seed",
    source: "human",
    history: [],
  };
  assert.doesNotThrow(() => validateClassifierSeeds(region, false));
  assert.throws(() => validateClassifierSeeds(region, true), /limite/);
  assert.throws(
    () => validateClassifierSeeds({ ...region, roi: undefined }, false),
    /limite/,
  );
  assert.throws(
    () => validateClassifierSeeds({ ...region, negative: undefined }, false),
    /Brush/,
  );
  region.history.push({
    action: "prepare-segmentation",
    roi: "automatic-8deg-or-15pct",
  });
  assert.throws(() => validateClassifierSeeds(region, false), /limite/);
  region.history.push({ tool: "polygon", target: "roi" });
  assert.doesNotThrow(() => validateClassifierSeeds(region, false));
});
test("automatic ROI wraps the seam and never expands through unsupported pixels", () => {
  const shape: [number, number] = [90, 180],
    seed = new Uint8Array(90 * 180),
    support = new Uint8Array(seed.length).fill(1);
  seed[45 * 180] = 1;
  seed[45 * 180 + 179] = 1;
  support[44 * 180] = 0;
  const roi = decodeMask(suggestSegmentationRoi(seed, shape, support));
  assert.equal(roi[45 * 180], 1);
  assert.equal(roi[45 * 180 + 179], 1);
  assert.equal(roi[45 * 180 + 90], 0);
  assert.equal(roi[44 * 180], 0);
  assert.throws(() =>
    suggestSegmentationRoi(new Uint8Array(seed.length), shape),
  );
});
