import assert from "node:assert/strict";
import { describe, it } from "node:test";
import { experimentFamilies, visibleCatalogFamilies } from "../src/catalog-model.ts";

describe("catalog discovery model", () => {
  it("renders a future family without a frontend allow-list change", () => {
    const families = [
      { id: "foundations", title: "Foundations", summary: "Known" },
      { id: "future-family", title: "Future", summary: "Unknown at build time" },
    ];
    const visible = visibleCatalogFamilies(families, [
      { family: "foundations" }, { family: "future-family" },
    ]);
    assert.deepEqual(visible.map((item) => item.id), ["foundations", "future-family"]);
  });

  it("groups workspace experiments from response data", () => {
    assert.deepEqual(
      experimentFamilies([{ family: "future-family" }, { family: "foundations" }, { family: "future-family" }]),
      ["future-family", "foundations"],
    );
  });
});
