import assert from "node:assert/strict";
import { describe, it } from "node:test";

import {
  createRecipeDraft,
  goalMatchesQuery,
  recipeHandoffUrl,
  recipeUsesSpecializedWorkspace,
  requirementLabel,
  type DiscoveryContext,
  type DiscoveryRecommendation,
  type DiscoveryRecipe,
} from "../src/discovery-model.ts";

const recipe: DiscoveryRecipe = {
  id: "spherical-dog",
  goal_id: "features",
  title: "Detectar features DoG esféricas",
  summary: "Mostre keypoints sobre a ERP.",
  kind: "operation",
  required: ["erp-a"],
  outputs: ["keypoints", "descriptors"],
  route: "/runner.html?component=dog",
  steps: [
    {
      id: "step-1",
      component_id: "auto--dog",
      title: "SphericalDoGFeatures",
      public_api: "panorai.applications.SphericalDoGFeatures",
    },
  ],
  stability: "experimental",
  runtime_ready: true,
  diagnostic: null,
  source: "curated",
  state: "ready",
  missing: [],
  score: 2100,
};

const context: DiscoveryContext = {
  inputs: [
    {
      id: "erp-a",
      kind: "erp",
      role: "a",
      source: "sample",
      name: "Studio",
    },
  ],
  facts: {},
};

const goal: DiscoveryRecommendation = {
  id: "features",
  title: "Detectar e descrever features",
  question: "Quero encontrar pontos distintivos",
  summary: "Extraia keypoints, bearings e descritores.",
  keywords: ["feature", "keypoint", "dog", "descritor", "ponto"],
  state: "ready",
  recommended: recipe,
  alternatives: [],
  total_candidates: 1,
};

describe("goal discovery model", () => {
  it("matches problem language without requiring API names", () => {
    assert.equal(goalMatchesQuery(goal, "pontos distintivos"), true);
    assert.equal(goalMatchesQuery(goal, "descritor"), true);
    assert.equal(goalMatchesQuery(goal, "profundidade"), false);
  });

  it("creates an editable draft without executing anything", () => {
    const draft = createRecipeDraft(
      recipe,
      context,
      new Date("2026-11-15T12:00:00.000Z"),
    );
    assert.equal(draft.schema, "panorai-workbench-recipe-draft/v1");
    assert.deepEqual(draft.input_ids, ["erp-a"]);
    assert.deepEqual(draft.steps.map((step) => step.component_id), ["auto--dog"]);
    assert.equal(
      recipeHandoffUrl(recipe, draft),
      "/runner.html?component=auto--dog&draft=recipe-1794744000000-spherical-dog",
    );
  });

  it("keeps specialized routes and names missing requirements for people", () => {
    const specialized = { ...recipe, kind: "specialized" as const, route: "/known-height-planar.html" };
    const draft = createRecipeDraft(specialized, context, new Date(0));
    assert.equal(
      recipeHandoffUrl(specialized, draft),
      "/known-height-planar.html?draft=recipe-0-spherical-dog",
    );
    assert.equal(requirementLabel("erp-b"), "segunda ERP");
    assert.equal(requirementLabel("camera-height"), "altura da câmera");
    assert.equal(requirementLabel("scalar-raster"), "raster escalar de um canal");
  });

  it("routes an existing visual bench without pretending it is the generic Runner", () => {
    const visualRecipe = { ...recipe, route: "/single-view.html?experiment=dog-op" };
    assert.equal(recipeUsesSpecializedWorkspace(visualRecipe), true);
    assert.match(
      recipeHandoffUrl(visualRecipe, createRecipeDraft(visualRecipe, context, new Date(0))),
      /^\/single-view\.html/,
    );
  });
});
