import assert from "node:assert/strict";
import test from "node:test";

import { createMutationQueue, denseAnchorStatus, denseSceneGuidance, erpAngularMeasurement, essentialControlState, imageRequestIsCurrent, invalidatePendingAction, isMetricMapScale, mergeCaptureGroupMetadata, metrologyCalibrationIsUsable, pairAssessment, pairRequestIsCurrent, projectRequestIsCurrent, replaceActiveRecord, replaceCachedRecord, sourcePixelPoints, sphericalPairCandidates, stitchingControlState, streetViewFlow, workflowCopy } from "../src/model.ts";

test("workflow copy keeps all four intake routes explicit", () => {
  assert.equal(workflowCopy("spherical_registered").title, "ERPs registradas");
  assert.equal(workflowCopy("spherical_unregistered").title, "Registro a estimar");
  assert.match(workflowCopy("perspective_uncalibrated").detail, /calibração/i);
});

test("guided workflow keeps all nine operator stages explicit and evidence-driven", () => {
  const initial = streetViewFlow({
    hasProject: false,
    imageCount: 0,
    calibratedImageCount: 0,
    usesProvidedPoses: false,
    registeredPairCount: 0,
    acceptedPairCount: 0,
    mapCurrent: false,
    denseAccepted: false,
    navigationReady: false,
    canPublish: false,
    published: false,
    publishedCount: 0,
  });
  assert.deepEqual(initial.map((stage) => stage.id), [
    "upload", "single-view", "multi-view", "adjust", "dense", "navigate", "review", "publish", "library",
  ]);
  assert.equal(initial[0].state, "active");
  assert.equal(initial[1].state, "blocked");

  const mapped = streetViewFlow({
    hasProject: true,
    imageCount: 4,
    calibratedImageCount: 2,
    usesProvidedPoses: false,
    registeredPairCount: 3,
    acceptedPairCount: 2,
    mapCurrent: true,
    denseAccepted: false,
    navigationReady: true,
    canPublish: true,
    published: false,
    publishedCount: 1,
  });
  assert.equal(mapped[2].state, "done");
  assert.equal(mapped[4].state, "optional");
  assert.equal(mapped[5].state, "ready");
  assert.equal(mapped[7].state, "ready");
  assert.match(mapped[8].reason, /1 ambiente/);

  const direct = streetViewFlow({
    hasProject: true,
    imageCount: 4,
    calibratedImageCount: 0,
    usesProvidedPoses: true,
    registeredPairCount: 0,
    acceptedPairCount: 0,
    mapCurrent: false,
    denseAccepted: false,
    navigationReady: true,
    canPublish: true,
    published: false,
    publishedCount: 0,
  });
  assert.equal(direct[2].state, "done");
  assert.equal(direct[3].state, "done");
  assert.match(direct[2].reason, /poses fornecidas/i);
});

test("spherical intake exposes every unordered pair for explicit selection", () => {
  assert.deepEqual(sphericalPairCandidates(["a.png", "b.png", "c.png"]), [
    { id: "pair-001", reference_filename: "a.png", source_filename: "b.png", provenance: { selection: "user-intake" } },
    { id: "pair-002", reference_filename: "a.png", source_filename: "c.png", provenance: { selection: "user-intake" } },
    { id: "pair-003", reference_filename: "b.png", source_filename: "c.png", provenance: { selection: "user-intake" } },
  ]);
});

test("pair loads are committed only for the current project, pair, and sequence", () => {
  const request = { projectId: "project-a", pairId: "pair-1", sequence: 4 };
  assert.equal(pairRequestIsCurrent(request, { projectId: "project-a", pairId: "pair-1", sequence: 4 }), true);
  assert.equal(pairRequestIsCurrent(request, { projectId: "project-a", pairId: "pair-2", sequence: 4 }), false);
  assert.equal(pairRequestIsCurrent(request, { projectId: "project-b", pairId: "pair-1", sequence: 4 }), false);
  assert.equal(pairRequestIsCurrent(request, { projectId: "project-a", pairId: "pair-1", sequence: 5 }), false);
});

test("project loads are committed only for the current project and sequence", () => {
  const request = { projectId: "project-a", sequence: 4 };
  assert.equal(projectRequestIsCurrent(request, { projectId: "project-a", sequence: 4 }), true);
  assert.equal(projectRequestIsCurrent(request, { projectId: "project-b", sequence: 4 }), false);
  assert.equal(projectRequestIsCurrent(request, { projectId: "project-a", sequence: 5 }), false);
});

test("metrology loads are committed only for the current project, image, and sequence", () => {
  const request = { projectId: "project-a", imageId: "image-1", sequence: 4 };
  assert.equal(imageRequestIsCurrent(request, { projectId: "project-a", imageId: "image-1", sequence: 4 }), true);
  assert.equal(imageRequestIsCurrent(request, { projectId: "project-a", imageId: "image-2", sequence: 4 }), false);
  assert.equal(imageRequestIsCurrent(request, { projectId: "project-b", imageId: "image-1", sequence: 4 }), false);
  assert.equal(imageRequestIsCurrent(request, { projectId: "project-a", imageId: "image-1", sequence: 5 }), false);
});

test("evidence mutations reach the backend in submission order", async () => {
  const enqueue = createMutationQueue();
  const order: string[] = [];
  let releaseFirst!: () => void;
  const firstGate = new Promise<void>((resolve) => {
    releaseFirst = resolve;
  });
  const first = enqueue(async () => {
    order.push("first-start");
    await firstGate;
    order.push("first-finish");
    return 1;
  });
  const second = enqueue(async () => {
    order.push("second-start");
    return 2;
  });
  await Promise.resolve();
  assert.deepEqual(order, ["first-start"]);
  releaseFirst();
  assert.deepEqual(await Promise.all([first, second]), [1, 2]);
  assert.deepEqual(order, ["first-start", "first-finish", "second-start"]);
  await assert.rejects(enqueue(async () => Promise.reject(new Error("save failed"))));
  assert.equal(await enqueue(async () => 3), 3);
});

test("Essential run and reestimate controls are mutually exclusive", () => {
  assert.deepEqual(essentialControlState(true, true), {
    runDisabled: true,
    reestimateDisabled: true,
  });
  assert.deepEqual(essentialControlState(false, true), {
    runDisabled: false,
    reestimateDisabled: false,
  });
  assert.deepEqual(essentialControlState(false, false), {
    runDisabled: false,
    reestimateDisabled: true,
  });
});

test("stitching run, promotion, and build share one in-flight guard", () => {
  assert.deepEqual(stitchingControlState({
    operationRunning: true,
    perspectiveProject: true,
    canBuild: true,
    promotionReady: true,
    stitchingStale: false,
  }), {
    runDisabled: true,
    promoteDisabled: true,
    buildDisabled: true,
  });
  assert.deepEqual(stitchingControlState({
    operationRunning: false,
    perspectiveProject: true,
    canBuild: true,
    promotionReady: true,
    stitchingStale: false,
  }), {
    runDisabled: false,
    promoteDisabled: false,
    buildDisabled: false,
  });
});

test("metric map scale recognizes every public mapper variant", () => {
  assert.equal(isMetricMapScale("metric-baseline-prior"), true);
  assert.equal(isMetricMapScale("metric-range-prior"), true);
  assert.equal(isMetricMapScale("metric-baseline-and-range-priors"), true);
  assert.equal(isMetricMapScale("arbitrary"), false);
  assert.equal(isMetricMapScale(undefined), false);
});

test("dense status uses mapper-eligible anchors", () => {
  assert.equal(denseAnchorStatus({
    boundCount: 0,
    requiredCount: 3,
    mapScale: "arbitrary",
    appliedCount: 0,
  }), "0/3 âncoras ligadas a tracks · adicione medidas em inliers distintos.");
  assert.equal(denseAnchorStatus({
    boundCount: 3,
    requiredCount: 3,
    mapScale: "arbitrary",
    appliedCount: 0,
  }), "3/3 âncoras ligadas a tracks · execute novamente o mapa.");
});

test("editing metrology inputs invalidates the persisted calibration", () => {
  assert.equal(metrologyCalibrationIsUsable("image-a", "image-a", false), true);
  assert.equal(metrologyCalibrationIsUsable("image-a", "image-a", true), false);
  assert.equal(metrologyCalibrationIsUsable("image-a", "image-b", false), false);
  const request = { projectId: "project-a", imageId: "image-a", sequence: 7 };
  assert.equal(imageRequestIsCurrent(request, {
    projectId: "project-a",
    imageId: "image-a",
    sequence: invalidatePendingAction(request.sequence),
  }), false);
});

test("editing evidence invalidates an in-flight global map response", () => {
  const request = { projectId: "project-a", sequence: 11 };
  assert.equal(projectRequestIsCurrent(request, {
    projectId: "project-a",
    sequence: invalidatePendingAction(request.sequence),
  }), false);
});

test("completed saves replace cached records even after workspace navigation", () => {
  const updated = { id: "project-a", revision: 3, updated_at: "2026-10-20T12:00:03+00:00" };
  const older = { id: "project-a", revision: 2, updated_at: "2026-10-20T12:00:02+00:00" };
  const cached = [
    { id: "project-a", revision: 1, updated_at: "2026-10-20T12:00:01+00:00" },
    { id: "project-b", revision: 2, updated_at: "2026-10-20T12:00:02+00:00" },
  ];
  assert.deepEqual(replaceCachedRecord(cached, updated), [
    updated,
    cached[1],
  ]);
  assert.equal(replaceActiveRecord(cached[0], updated), updated);
  assert.equal(replaceActiveRecord(cached[1], updated), cached[1]);
  assert.equal(replaceCachedRecord([updated], older)[0], updated);
  assert.equal(replaceActiveRecord(updated, older), updated);
});

test("capture group editor merges assignments without discarding camera metadata", () => {
  const merged = JSON.parse(mergeCaptureGroupMetadata(
    JSON.stringify({ "left.png": { horizontal_fov_deg: 92 }, global_note: "kept" }),
    [
      { filename: "left.png", captureGroup: "hall-a" },
      { filename: "right.png", captureGroup: "hall-a" },
      { filename: "next.png", captureGroup: "hall-b" },
    ],
  ));

  assert.deepEqual(merged, {
    "left.png": { horizontal_fov_deg: 92, capture_group: "hall-a" },
    global_note: "kept",
    "right.png": { capture_group: "hall-a" },
    "next.png": { capture_group: "hall-b" },
  });
  assert.throws(
    () => mergeCaptureGroupMetadata("{}", [{ filename: "bad.png", captureGroup: "hall a" }]),
    /Grupo inválido/,
  );
});

test("sourcePixelPoints converts display coordinates to source pixel centres", () => {
  assert.deepEqual(
    sourcePixelPoints([{ x: 50, y: 25 }, { x: 100, y: 50 }], { width: 200, height: 100 }, { width: 1000, height: 500 }),
    [[250, 125], [500, 250]],
  );
});

test("ERP measurement follows great-circle geometry and wraps the seam", () => {
  const quarterTurn = erpAngularMeasurement([179.5, 89.5], [269.5, 89.5], { width: 360, height: 180 });
  assert.ok(Math.abs(quarterTurn.angleDeg - 90) < 1e-10);
  assert.equal(quarterTurn.horizontalWrappedPixels, 90);
  const seam = erpAngularMeasurement([359, 89.5], [0, 89.5], { width: 360, height: 180 });
  assert.ok(Math.abs(seam.angleDeg - 1) < 1e-10);
  assert.equal(seam.horizontalWrappedPixels, 1);
  assert.equal(seam.pixelDistance, 1);
});

test("pairAssessment turns frozen overlap evidence into registration guidance", () => {
  assert.deepEqual(
    pairAssessment({ relation: "registered_overlap", minimum_directional_overlap: 0.9419 }),
    {
      label: "Registro com alta sobreposição",
      shortLabel: "overlap alto",
      tone: "strong",
      guidance: "Comece pela sobreposição; use correção manual somente se estruturas estáveis divergirem.",
    },
  );
  assert.equal(
    pairAssessment({ relation: "registered_overlap", minimum_directional_overlap: 0.6934 }).tone,
    "useful",
  );
  assert.deepEqual(
    pairAssessment({ relation: "overlap_candidate", minimum_directional_overlap: null }),
    {
      label: "Pose ainda não estimada",
      shortLabel: "aguardando pose",
      tone: "unknown",
      guidance: "Compare o par candidato; a estimativa deve vir das APIs públicas PanorAi.",
    },
  );
});

test("denseSceneGuidance chooses the missing geometric prerequisite", () => {
  const summary = {
    image_count: 4,
    pair_count: 6,
    registered_pair_count: 2,
    accepted_pair_count: 2,
    accepted_component_size: 3,
    metric_scale_available: false,
    metric_scale_sources: [],
    bound_range_prior_count: 1,
    range_prior_min_count: 3,
    global_map_eligible: true,
    global_map_status: "current" as const,
    dense_scene_eligible: false,
    dense_scene_blockers: ["metric-scale-required" as const],
  };
  assert.deepEqual(denseSceneGuidance(summary), {
    state: "scale",
    title: "Faltam 2 medidas radiais",
    detail: "1/3 âncoras estão ligadas a inliers distintos. Sem escala, o TSDF não pode ser criado.",
    action: "Adicionar âncora",
  });
  assert.equal(denseSceneGuidance({
    ...summary,
    metric_scale_available: true,
    metric_scale_sources: ["range-prior"],
    bound_range_prior_count: 3,
    dense_scene_eligible: true,
    dense_scene_blockers: [],
  }).state, "ready");
});
