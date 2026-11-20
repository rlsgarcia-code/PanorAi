import assert from "node:assert/strict";
import test from "node:test";
import {
  acceptDirectionCandidate,
  canAcceptConstraint,
  canDrawPlanarLine,
  directionLineVisualState,
  derivePlanarWorkspaceFlow,
  dependentPlaneIds,
  findMatchingOrientationProposal,
  floorFamiliesFromDiscovery,
  guidedLineEvidenceActions,
  hasHorizontalAlignmentEvidence,
  hasMeasuredHeight,
  lineRefinementRequiresDecision,
  migrateKnownHeightScene,
  normalizePlanarDiscoveryTuning,
  numericSigmaParameter,
  orientationEvidenceId,
  planarContactPreviewMessage,
  planarDiscoveryPreset,
  planarRecoveryStep,
  relevantOrientationProposals,
  revise,
  suggestedConstraint,
  type KnownHeightScene,
} from "../src/known-height-model.ts";

function scene(): KnownHeightScene {
  return {
    schema: "panorai-known-height-planar-scene/v1",
    scene_id: "test",
    revision: "workbench-1",
    camera_height_m: 1.6,
    camera_height_sigma_m: null,
    down_erp_xy: [4, 3],
    horizon_erp_xy: [],
    horizontal_vanishing_erp_xy: [],
    floor_line_families_erp_xy: [],
    regions: [{
      plane_id: "floor", view_id: "single-view", label: "Piso",
      state: "accepted", mask: { shape_hw: [2, 4], runs: [4, 4] },
      interpretation: "final", history: [], source: "test",
    }],
    constraints: [{
      constraint_id: "floor", plane_id: "floor", kind: "floor-anchor",
      reference_plane_id: null, contact_erp_xy: [], state: "accepted", source: "test",
    }],
    provenance: {},
  };
}

test("human actions create a new immutable revision", () => {
  const original = scene();
  const updated = revise(original, "change-height", value => { value.camera_height_m = 1.8; });
  assert.equal(original.camera_height_m, 1.6);
  assert.equal(updated.camera_height_m, 1.8);
  assert.equal(updated.revision, "workbench-2");
  assert.equal(updated.provenance.parent_revision, "workbench-1");
});

test("accepted floor suggests a generic reviewed orthogonal relation, never coplanarity", () => {
  const suggestion = suggestedConstraint(scene(), "wall");
  assert.equal(suggestion.kind, "orthogonal-to");
  assert.equal(suggestion.reference_plane_id, "floor");
  assert.equal(suggestion.state, "proposed");
});

test("connected constraints require two explicit contact observations", () => {
  const suggestion = suggestedConstraint(scene(), "wall");
  assert.equal(canAcceptConstraint(suggestion), false);
  suggestion.contact_erp_xy.push([1, 1], [2, 1]);
  assert.equal(canAcceptConstraint(suggestion), true);
});

test("editing one plane invalidates only that plane and its descendants", () => {
  const constraints = [
    { ...scene().constraints[0] },
    { ...suggestedConstraint(scene(), "wall"), reference_plane_id: "floor" },
    { ...suggestedConstraint(scene(), "poster"), plane_id: "poster", reference_plane_id: "wall" },
    { ...suggestedConstraint(scene(), "table"), plane_id: "table", reference_plane_id: "floor" },
  ];
  assert.deepEqual(new Set(dependentPlaneIds(constraints, "wall")), new Set(["wall", "poster"]));
});

test("missing scientific uncertainty uses only a numeric form fallback", () => {
  const original = scene();
  assert.equal(original.camera_height_sigma_m, null);
  assert.equal(numericSigmaParameter(original.camera_height_sigma_m), 0.02);
  assert.equal(original.camera_height_sigma_m, null);
  assert.equal(numericSigmaParameter(0.01), 0.01);
});

test("real-image preparation requires a measured height and two horizontal cues", () => {
  const original = scene();
  original.camera_height_m = null;
  original.horizon_erp_xy = [];
  assert.equal(hasMeasuredHeight(original), false);
  assert.equal(hasHorizontalAlignmentEvidence(original), false);
  original.camera_height_m = 1.72;
  original.floor_line_families_erp_xy = [
    [[[1, 1], [2, 1]], [[1, 2], [2, 2]]],
    [[[3, 1], [3, 2]], [[4, 1], [4, 2]]],
  ];
  assert.equal(hasMeasuredHeight(original), true);
  assert.equal(hasHorizontalAlignmentEvidence(original), true);
});

test("only an actionable line recommendation blocks further observations", () => {
  assert.equal(lineRefinementRequiresDecision("suggested", true), true);
  assert.equal(lineRefinementRequiresDecision("suggested", false), false);
  assert.equal(lineRefinementRequiresDecision("no-supported-edge", false), false);
  assert.equal(lineRefinementRequiresDecision("unavailable", false), false);
});

test("a missing planar line can be drawn in assisted and manual discovery", () => {
  assert.equal(canDrawPlanarLine("assisted", false), true);
  assert.equal(canDrawPlanarLine("manual", false), true);
  assert.equal(canDrawPlanarLine("assisted", true), false);
  assert.equal(canDrawPlanarLine("assisted", false, true), false);
});

test("plane membership is distinct from directional grouping", () => {
  const unassigned=guidedLineEvidenceActions(["a"],[]);
  assert.equal(unassigned.canAssociate,true);
  assert.equal(unassigned.canEstimateDirection,false);
  const oneAssigned=guidedLineEvidenceActions(["a"],["a"]);
  assert.equal(oneAssigned.canAssociate,false);
  assert.equal(oneAssigned.canEstimateDirection,false);
  const direction=guidedLineEvidenceActions(["a","b"],["a","b","c"]);
  assert.equal(direction.associatedCount,3);
  assert.equal(direction.canEstimateDirection,true);
  const mixed=guidedLineEvidenceActions(["a","outside"],["a","b"]);
  assert.equal(mixed.canAssociate,true);
  assert.equal(mixed.canEstimateDirection,false);
});

test("direction expansion gives accepted members and unreviewed candidates distinct visual states", () => {
  const selected=["selected"];
  const members=["member","selected"];
  const candidates=["candidate","member"];
  const plane=["plane","candidate"];
  assert.equal(directionLineVisualState("selected",selected,members,candidates,plane),"selected");
  assert.equal(directionLineVisualState("member",selected,members,candidates,plane),"direction-member");
  assert.equal(directionLineVisualState("candidate",selected,members,candidates,plane),"expansion-candidate");
  assert.equal(directionLineVisualState("plane",selected,members,candidates,plane),"plane-member");
  assert.equal(directionLineVisualState("other",selected,members,candidates,plane),"other");
});

test("accepting an expansion candidate adds it to both the direction and the active plane", () => {
  assert.deepEqual(
    acceptDirectionCandidate(["line-a"],["line-a","line-b"],"line-c"),
    {directionLineIds:["line-a","line-c"],planeLineIds:["line-a","line-b","line-c"]},
  );
});

test("an existing proposal is reused instead of creating a duplicate proposal id", () => {
  const proposals = [
    { proposal_id: "unrelated", bundle_ids: ["d1", "d3"] },
    { proposal_id: "ready", bundle_ids: ["d2", "d1"] },
  ];
  assert.equal(findMatchingOrientationProposal(proposals, ["d1", "d2"])?.proposal_id, "ready");
  assert.equal(findMatchingOrientationProposal(proposals, ["d1"]), undefined);
  assert.equal(findMatchingOrientationProposal(proposals, ["d1", "d2", "d3"]), undefined);
});

test("orientation evidence IDs are deterministic and scoped to the plane", () => {
  assert.equal(orientationEvidenceId("floor", "proposal-a"), "orientation:floor:proposal-a");
  assert.notEqual(
    orientationEvidenceId("floor", "proposal-a"),
    orientationEvidenceId("wall", "proposal-a"),
  );
});

test("scene migration repairs duplicate orientation evidence IDs and their constraints", () => {
  const value=scene();
  value.orientations=[
    {plane_id:"floor",proposal_id:"shared",normal_camera:[0,1,0],line_ids:[],bundle_ids:[],confidence:1,residual_deg:0,state:"accepted",source:"human"},
    {plane_id:"wall",proposal_id:"shared",normal_camera:[1,0,0],line_ids:[],bundle_ids:[],confidence:1,residual_deg:0,state:"accepted",source:"human"},
  ];
  value.constraints.push({constraint_id:"wall",plane_id:"wall",kind:"orthogonal-to",reference_plane_id:"floor",contact_erp_xy:[[1,1]],state:"accepted",source:"test",orientation_id:"shared"});
  const migrated=migrateKnownHeightScene(value);
  assert.deepEqual(migrated.orientations?.map(item=>item.proposal_id),[
    orientationEvidenceId("floor","shared"),orientationEvidenceId("wall","shared"),
  ]);
  assert.equal(migrated.constraints.find(item=>item.plane_id==="wall")?.orientation_id,orientationEvidenceId("wall","shared"));
});

test("orientation review shows only the proposal for the selected directions", () => {
  const proposals = [
    { proposal_id: "broad", plane_id: "wall", bundle_ids: ["d1", "d2", "d3", "d4"], confidence: 0 },
    { proposal_id: "selected", plane_id: null, bundle_ids: ["d2", "d1"], confidence: .98 },
  ];
  assert.deepEqual(
    relevantOrientationProposals(proposals, "wall", ["d1", "d2"]).map(item => item.proposal_id),
    ["selected"],
  );
});

test("orientation review resolves a plane-scoped accepted evidence ID through its selected directions", () => {
  const proposals = [
    { proposal_id: "selected", plane_id: null, bundle_ids: ["d1", "d2"], confidence: .98 },
    { proposal_id: "other", plane_id: "wall", bundle_ids: ["d3", "d4"], confidence: .50 },
  ];
  assert.deepEqual(
    relevantOrientationProposals(
      proposals,
      "wall",
      ["d2", "d1"],
      orientationEvidenceId("wall", "selected"),
    ).map(item => item.proposal_id),
    ["selected"],
  );
});

test("planar discovery presets expose their actual gnomonic working resolution", () => {
  assert.equal(planarDiscoveryPreset("sparse").atlasFaceSize, 256);
  assert.equal(planarDiscoveryPreset("balanced").atlasFaceSize, 384);
  assert.equal(planarDiscoveryPreset("dense").atlasFaceSize, 512);
  assert.deepEqual(
    [planarDiscoveryPreset("balanced").cannyLow, planarDiscoveryPreset("balanced").cannyHigh],
    [25, 105],
  );
  assert.equal(planarDiscoveryPreset("balanced").localContrastClipLimit, 2);
});

test("custom discovery tuning preserves high resolution and bounds invalid detector values", () => {
  const tuning = normalizePlanarDiscoveryTuning({
    ...planarDiscoveryPreset("balanced"),
    atlasFaceSize: 768,
    cannyLow: 180,
    cannyHigh: 100,
    minimumSegmentPx: 2,
    minimumEdgeSupport: 1.4,
    localContrastClipLimit: 12,
  });
  assert.equal(tuning.atlasFaceSize, 768);
  assert.equal(tuning.cannyLow, 180);
  assert.equal(tuning.cannyHigh, 181);
  assert.equal(tuning.minimumSegmentPx, 4);
  assert.equal(tuning.minimumEdgeSupport, 1);
  assert.equal(tuning.localContrastClipLimit, 8);
});

test("a reviewed orientation proposal becomes exactly two cloned line families", () => {
  const first: [[number, number], [number, number]] = [[1, 2], [3, 4]];
  const second: [[number, number], [number, number]] = [[5, 6], [7, 8]];
  const third: [[number, number], [number, number]] = [[9, 10], [11, 12]];
  const fourth: [[number, number], [number, number]] = [[13, 14], [15, 16]];
  const families = floorFamiliesFromDiscovery(
    { bundle_ids: ["a", "b"] },
    [{ bundle_id: "a", line_ids: ["l1", "l2"] }, { bundle_id: "b", line_ids: ["l3", "l4"] }],
    [
      { line_id: "l1", erp_endpoints_xy: first, state: "accepted" },
      { line_id: "l2", erp_endpoints_xy: second, state: "detected" },
      { line_id: "l3", erp_endpoints_xy: third, state: "detected" },
      { line_id: "l4", erp_endpoints_xy: fourth, state: "accepted" },
    ],
  );
  assert.deepEqual(families, [[first, second], [third, fourth]]);
  families[0][0][0][0] = 999;
  assert.equal(first[0][0], 1);
});

test("a discovery proposal cannot use missing or rejected evidence", () => {
  assert.throws(
    () => floorFamiliesFromDiscovery(
      { bundle_ids: ["a", "missing"] },
      [{ bundle_id: "a", line_ids: ["l1", "l2"] }],
      [{ line_id: "l1", erp_endpoints_xy: [[1, 1], [2, 2]], state: "accepted" }, { line_id: "l2", erp_endpoints_xy: [[2, 1], [3, 2]], state: "accepted" }],
    ),
    /Bundle missing/,
  );
  assert.throws(
    () => floorFamiliesFromDiscovery(
      { bundle_ids: ["a", "b"] },
      [{ bundle_id: "a", line_ids: ["l1", "l2"] }, { bundle_id: "b", line_ids: ["l3", "l4"] }],
      [
        { line_id: "l1", erp_endpoints_xy: [[1, 1], [2, 2]], state: "rejected" },
        { line_id: "l2", erp_endpoints_xy: [[2, 1], [3, 2]], state: "accepted" },
        { line_id: "l3", erp_endpoints_xy: [[3, 1], [4, 2]], state: "accepted" },
        { line_id: "l4", erp_endpoints_xy: [[4, 1], [5, 2]], state: "accepted" },
      ],
    ),
    /precisa de duas retas/,
  );
});

test("the floor workflow exposes one current decision in metric order", () => {
  const base = {
    activePlaneId: "floor", heightReady: true, floorOrientationReady: true,
    planeOrientationReady: false, relationReady: false, maskReady: true,
    maskAccepted: false, metricReady: false, floorMetricReady: false,
  };
  const review = derivePlanarWorkspaceFlow(base);
  assert.equal(review.scope, "floor");
  assert.equal(review.current, "review");
  assert.deepEqual(review.phases.filter(item => item.state === "current").map(item => item.step), ["review"]);
  assert.equal(review.canAddPlane, false);
  const ready = derivePlanarWorkspaceFlow({ ...base, relationReady: true, maskAccepted: true, metricReady: true, floorMetricReady: true });
  assert.equal(ready.current, "ready");
  assert.equal(ready.canAddPlane, true);
});

test("a new plane is oriented and positioned before segmentation", () => {
  const base = {
    activePlaneId: "wall", heightReady: true, floorOrientationReady: true,
    planeOrientationReady: false, relationReady: false, maskReady: false,
    maskAccepted: false, metricReady: false, floorMetricReady: true,
  };
  assert.equal(derivePlanarWorkspaceFlow(base).current, "orientation");
  assert.equal(derivePlanarWorkspaceFlow({ ...base, planeOrientationReady: true }).current, "relation");
  assert.equal(derivePlanarWorkspaceFlow({ ...base, planeOrientationReady: true, relationReady: true }).current, "segmentation");
  assert.equal(derivePlanarWorkspaceFlow({ ...base, planeOrientationReady: true, relationReady: true, maskReady: true }).current, "review");
  assert.equal(derivePlanarWorkspaceFlow({ ...base, planeOrientationReady: true, relationReady: true, maskReady: true, maskAccepted: true }).current, "metric");
});

test("scientific unresolved reasons route the user back to the decision that needs revision", () => {
  assert.equal(planarRecoveryStep("orthogonal-orientation-needs-review"),"orientation");
  assert.equal(planarRecoveryStep("insufficient-valid-contact-points"),"relation");
  assert.equal(planarRecoveryStep("accepted-mask-required"),"review");
  assert.equal(planarRecoveryStep("unsupported-planar-relation"),"metric");
  assert.equal(planarRecoveryStep(undefined),null);
});

test("contact preview failures give actionable guidance instead of failing silently", () => {
  assert.match(planarContactPreviewMessage("cursor ray is tangent to the reference plane"), /mais abaixo no piso/);
  assert.match(planarContactPreviewMessage("cursor does not intersect the reference plane in front of the camera"), /região visível do piso/);
  assert.match(planarContactPreviewMessage("unexpected failure"), /Tente outro ponto/);
});
