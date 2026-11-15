import assert from "node:assert/strict";
import test from "node:test";
import {
  dragPlanarView,
  zoomPlanarView,
  planarHistoryShortcut,
} from "../src/planar-navigation.ts";
import { planarToolCursor } from "../src/planar-model.ts";

const initial = { yawDeg: 170, pitchDeg: 0, horizontalFovDeg: 72 };
test("hand rotation wraps longitude, clamps latitude and leaves original view unchanged", () => {
  const moved = dragPlanarView(
    initial,
    { x: 100, y: 100 },
    { x: -100, y: 2000 },
    400,
    "navigate",
  );
  assert.equal(moved.yawDeg, -154); //170 - (-200)*72/400 =206 -> -154
  assert.equal(moved.pitchDeg, 89.9);
  assert.equal(moved.horizontalFovDeg, 72);
  assert.deepEqual(initial, { yawDeg: 170, pitchDeg: 0, horizontalFovDeg: 72 });
});
test("zoom drag and buttons change FOV only, bounded to 15..120 degrees", () => {
  const zoomed = dragPlanarView(
    initial,
    { x: 0, y: 100 },
    { x: 200, y: 0 },
    400,
    "zoom",
  );
  assert.ok(zoomed.horizontalFovDeg < 72);
  assert.equal(zoomed.yawDeg, 170);
  assert.equal(zoomed.pitchDeg, 0);
  assert.equal(zoomPlanarView(initial, 0.001).horizontalFovDeg, 15);
  assert.equal(zoomPlanarView(initial, 1000).horizontalFovDeg, 120);
  assert.throws(() => zoomPlanarView(initial, Number.NaN));
  assert.throws(() => zoomPlanarView(initial, 0));
  assert.equal(planarToolCursor("zoom"), "zoom-in");
});
test("history shortcuts support macOS and Windows without stealing ordinary Z/Alt keys", () => {
  const event = {
    key: "z",
    ctrlKey: false,
    metaKey: false,
    shiftKey: false,
    altKey: false,
  };
  assert.equal(planarHistoryShortcut(event), null);
  assert.equal(planarHistoryShortcut({ ...event, metaKey: true }), "undo");
  assert.equal(
    planarHistoryShortcut({
      ...event,
      ctrlKey: true,
      shiftKey: true,
      key: "Z",
    }),
    "redo",
  );
  assert.equal(
    planarHistoryShortcut({ ...event, ctrlKey: true, key: "y" }),
    "redo",
  );
  assert.equal(
    planarHistoryShortcut({ ...event, ctrlKey: true, altKey: true }),
    null,
  );
});
