import assert from "node:assert/strict";
import test from "node:test";
import { fillClosedContours } from "../src/planar-fill.ts";

// Independent discrete pixel fixtures: foreground border, native ERP rows,
// periodic longitude, no color/depth-derived support. Exact integer oracle.
function rectangle(shape: [number, number], x0 = 3, x1 = 9, y0 = 2, y1 = 7) {
  const mask = new Uint8Array(shape[0] * shape[1]);
  for (let y = y0; y <= y1; y++)
    for (let x = x0; x <= x1; x++)
      if (y === y0 || y === y1 || x === x0 || x === x1)
        mask[y * shape[1] + x] = 1;
  return mask;
}
test("closed contour fills only its enclosed interior and is idempotent", () => {
  const shape: [number, number] = [10, 14],
    mask = rectangle(shape);
  assert.equal(fillClosedContours(mask, shape), 20);
  for (let y = 0; y < 10; y++)
    for (let x = 0; x < 14; x++)
      assert.equal(
        mask[y * 14 + x],
        Number(y >= 2 && y <= 7 && x >= 3 && x <= 9),
      );
  assert.equal(fillClosedContours(mask, shape), 0);
});
test("an open contour does not fill or bridge its gap", () => {
  const shape: [number, number] = [10, 14],
    mask = rectangle(shape);
  mask[4 * 14 + 9] = 0;
  const original = mask.slice();
  assert.equal(fillClosedContours(mask, shape), 0);
  assert.deepEqual(mask, original);
});
test("fill respects support and persistent negative exclusions, updating positive seeds", () => {
  const shape: [number, number] = [10, 14],
    mask = rectangle(shape);
  const support = new Uint8Array(mask.length).fill(1),
    excluded = new Uint8Array(mask.length),
    positive = mask.slice();
  support[4 * 14 + 5] = 0;
  excluded[5 * 14 + 6] = 1;
  assert.equal(
    fillClosedContours(mask, shape, { support, excluded, positive }),
    18,
  );
  assert.equal(mask[4 * 14 + 5], 0);
  assert.equal(mask[5 * 14 + 6], 0);
  assert.deepEqual(mask, positive);
  assert.equal(mask[4 * 14 + 6], 1);
});
test("closed contour across longitude seam fills the wrapped interior only", () => {
  const shape: [number, number] = [10, 12],
    mask = new Uint8Array(120);
  for (let y = 3; y <= 7; y++)
    for (const x of [10, 11, 0, 1, 2])
      if (y === 3 || y === 7 || x === 10 || x === 2) mask[y * 12 + x] = 1;
  assert.equal(fillClosedContours(mask, shape), 9);
  for (let y = 0; y < 10; y++)
    for (let x = 0; x < 12; x++)
      assert.equal(
        mask[y * 12 + x],
        Number(y >= 3 && y <= 7 && [10, 11, 0, 1, 2].includes(x)),
      );
});
test("polar contour fills only the explicitly selected cap, not the whole sphere", () => {
  const shape: [number, number] = [10, 12];
  for (const [row, pole] of [
    [6, "north"],
    [3, "south"],
  ] as const) {
    const mask = new Uint8Array(120);
    mask.fill(1, row * 12, (row + 1) * 12);
    assert.equal(fillClosedContours(mask, shape), 0); // no guessed spherical outside
    assert.equal(fillClosedContours(mask, shape, { pole }), 72);
    for (let y = 0; y < 10; y++)
      for (let x = 0; x < 12; x++)
        assert.equal(
          mask[y * 12 + x],
          Number(pole === "north" ? y <= row : y >= row),
        );
  }
});
test("empty masks and invalid buffer dimensions fail safely", () => {
  assert.equal(fillClosedContours(new Uint8Array(120), [10, 12]), 0);
  assert.throws(() => fillClosedContours(new Uint8Array(12), [10, 12]));
  assert.throws(() =>
    fillClosedContours(new Uint8Array(120), [10, 12], {
      support: new Uint8Array(2),
    }),
  );
});
