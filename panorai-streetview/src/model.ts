export type Workflow =
  | "spherical_registered"
  | "spherical_unregistered"
  | "perspective_calibrated"
  | "perspective_uncalibrated";

export type ReviewState = "unreviewed" | "accepted" | "rejected";
export type RegionMode = "all" | "prioritize" | "restrict";

export interface ImageRecord {
  id: string;
  filename: string;
  sha256: string;
  media_type: string;
  width: number;
  height: number;
  size_bytes: number;
  metadata: Record<string, unknown>;
}

export interface ObjectRegion {
  id: string | null;
  object_id: string;
  label: string;
  image_id: string;
  bbox_xyxy: [number, number, number, number];
  state: "accepted" | "rejected";
  created_at: string | null;
}

export interface ManualCorrespondence {
  id: string | null;
  pair_id: string;
  reference_xy: [number, number];
  source_xy: [number, number];
  state: "accepted" | "rejected";
  created_at: string | null;
}

export interface MetrologyLine {
  family: "floor_a" | "floor_b";
  start_xy: [number, number];
  end_xy: [number, number];
}

export interface MetrologyFamilyRecord {
  family: "floor_a" | "floor_b";
  direction_panorama: [number, number, number];
  inlier_indices: number[];
  outlier_indices: number[];
  residuals_deg: number[];
  median_residual_deg: number;
  p90_residual_deg: number;
}

export interface FloorCalibrationRecord {
  schema: "panorai-streetview-floor-calibration/v1";
  project_id: string;
  image_id: string;
  revision: number;
  created_at: string;
  engine: Record<string, unknown>;
  shape_hw: [number, number];
  camera_height_m: number;
  floor_hint_xy: [number, number];
  lines: MetrologyLine[];
  rotation_panorama_to_world: number[][];
  floor_normal_panorama: [number, number, number];
  family_separation_deg: number;
  floor_normal_uncertainty_deg: number;
  families: MetrologyFamilyRecord[];
}

export interface FloorMeasurement {
  calibration_revision: number;
  point_world_m: [number, number, number];
  horizontal_distance_m: number;
  radial_distance_m: number;
}

export interface VerticalMeasurement {
  calibration_revision: number;
  base_world_m: [number, number, number];
  top_world_m: [number, number, number];
  height_m: number;
  horizontal_residual_m: number;
}

export interface DenseAnnotation {
  id: string | null;
  kind: "include_region" | "exclude_region" | "range_anchor" | "surface";
  image_id: string;
  label: string;
  points_xy: Array<[number, number]>;
  radial_range: number | null;
  feature_index: number | null;
  binding_pair_id: string | null;
  binding_distance_px: number | null;
  engine_effect: "staged" | "pair_validity" | "range_prior";
  state: "accepted" | "rejected";
  created_at: string | null;
}

export interface ImagePairRecord {
  id: string;
  reference_image_id: string;
  source_image_id: string;
  relation: "registered_overlap" | "overlap_candidate";
  baseline_m: number | null;
  minimum_directional_overlap: number | null;
  provenance: Record<string, unknown>;
}

export interface IntakeImagePair {
  id: string;
  reference_filename: string;
  source_filename: string;
  provenance: { selection: "user-intake" };
}

export interface ProjectRecord {
  schema: "panorai-streetview-project/v1";
  id: string;
  name: string;
  workflow: Workflow;
  created_at: string;
  updated_at: string;
  panorai_runtime: { version?: string; module_file?: string };
  images: ImageRecord[];
  pairs: ImagePairRecord[];
  object_regions: ObjectRegion[];
  manual_correspondences: ManualCorrespondence[];
  dense_annotations: DenseAnnotation[];
  status: "draft" | "ready" | "blocked" | "built";
}

export interface DetectionRecord {
  index: number;
  xy: [number, number];
  response: number;
  scale: number;
  face_id: string;
}

export interface MatchRecord {
  index: number;
  feature_index_reference: number;
  feature_index_source: number;
  reference_xy: [number, number];
  source_xy: [number, number];
  descriptor_distance: number;
  ratio_score: number | null;
  auto_valid: boolean;
  region_match: boolean;
  review_state: ReviewState;
  inlier: boolean;
  residual_deg: number | null;
}

export interface PoseEvidence {
  rotation: number[][];
  translation_direction: number[];
  essential_matrix: number[][];
  num_inliers: number;
  auto_inlier_count: number;
  manual_inlier_count: number;
  median_parallax_deg: number;
  cheirality_ratio: number;
  degenerate: boolean;
  degeneracy_reasons: string[];
  accepted: boolean;
  quality: {
    inlier_ratio: number;
    median_residual_deg: number;
    raw_quality_score: number;
    rejection_reasons: string[];
    [key: string]: unknown;
  };
  compute_backend: string;
  translation_scale: "unobservable";
  pose_convention: "source-from-reference";
}

export interface ObjectSummary {
  object_id: string;
  label: string;
  reference_region_id: string;
  source_region_id: string;
  reference_bbox_xyxy: [number, number, number, number];
  source_bbox_xyxy: [number, number, number, number];
  reference_detection_count: number;
  source_detection_count: number;
  match_count: number;
}

export interface PairRegistration {
  schema: "panorai-streetview-pair-registration/v1";
  id: string;
  project_id: string;
  pair_id: string;
  revision: number;
  created_at: string;
  updated_at: string;
  region_mode: RegionMode;
  engine: Record<string, unknown>;
  feature_counts: { reference: number; source: number };
  detection_payload: { bounded: boolean; limit_per_image: number };
  detections: { reference: DetectionRecord[]; source: DetectionRecord[] };
  matches: MatchRecord[];
  object_summaries: ObjectSummary[];
  manual_correspondence_count: number;
  active_correspondence_count: number;
  pending_match_reviews: number[];
  quality_gates: QualityGate[];
  recommended_edit: { tool: string; title: string; detail: string } | null;
  revision_comparison: RevisionComparison | null;
  pose: PoseEvidence | null;
}

export interface QualityGate {
  id: string;
  label: string;
  value: number | string;
  threshold: number | string;
  threshold_kind: "minimum" | "maximum" | "equals";
  passed: boolean;
  action: string;
}

export interface RevisionComparison {
  from_revision: number;
  to_revision: number;
  solution_changed: boolean;
  inlier_delta: number;
  quality_delta: number | null;
  parallax_delta_deg: number | null;
  accepted_before: boolean;
  accepted_after: boolean;
}

export interface RegistrationRevision {
  revision: number;
  updated_at: string;
  region_mode: RegionMode;
  accepted: boolean;
  num_inliers: number;
  quality_score: number | null;
  median_parallax_deg: number | null;
  accepted_reviews: number;
  rejected_reviews: number;
  manual_correspondence_count: number;
}

export interface RegistrationHistory {
  schema: "panorai-streetview-registration-history/v1";
  project_id: string;
  pair_id: string;
  revisions: RegistrationRevision[];
}

export interface SetupPairState {
  pair_id: string;
  reference_image_id: string;
  source_image_id: string;
  status: "not-run" | "review" | "stale" | "accepted";
  revision: number | null;
  accepted: boolean;
  num_inliers: number;
  quality_score: number | null;
  pending_review_count: number;
  evidence_newer_than_pose: boolean;
  baseline_m: number | null;
}

export interface SceneSetup {
  schema: "panorai-streetview-scene-setup/v1";
  project_id: string;
  workflow: Workflow;
  summary: {
    image_count: number;
    pair_count: number;
    registered_pair_count: number;
    accepted_pair_count: number;
    accepted_component_size: number;
    metric_scale_available: boolean;
    metric_scale_sources: Array<"baseline-prior" | "range-prior">;
    bound_range_prior_count: number;
    range_prior_min_count: number;
    global_map_eligible: boolean;
    global_map_status: "not-run" | "current" | "stale" | "failed";
    dense_scene_eligible: boolean;
    dense_scene_blockers: Array<
      | "current-global-map-required"
      | "metric-scale-required"
      | "three-registered-views-required"
    >;
  };
  pairs: SetupPairState[];
  steps: Array<{
    id: string;
    title: string;
    status: "ready" | "action" | "planned" | "blocked" | "optional";
    detail: string;
    engine: string;
  }>;
  recommended_action: {
    kind: string;
    pair_id: string | null;
    label: string;
    detail: string;
  };
  profiles: Record<string, Record<string, unknown>>;
  capabilities: Capabilities["engines"];
}

export interface GlobalMapPose {
  image_id: string;
  rotation_world_to_panorama: number[][];
  center_world: [number, number, number];
  display_xy: [number, number];
  active_track_count: number;
  max_reprojection_deg: number | null;
}

export interface GlobalMapRecord {
  schema: "panorai-streetview-global-map/v1";
  project_id: string;
  revision: number;
  created_at: string;
  stale: boolean;
  engine: { application: string; interface: string; options: Record<string, unknown> };
  success: boolean;
  failure_reasons: string[];
  scale: string;
  reference_image_id: string | null;
  input_edge_revisions: Record<string, number>;
  range_prior_candidates: Array<{
    annotation_id: string;
    panorama_id: string;
    feature_index: number;
    range_m: number;
    binding_pair_id: string;
  }>;
  range_priors_applied: number;
  range_prior_min_count: number;
  edges: Array<{
    pair_id: string;
    revision: number;
    reference_image_id: string;
    source_image_id: string;
    num_inliers: number;
    quality_score: number;
  }>;
  skipped_pairs: Array<{ pair_id: string; reason: string }>;
  poses: GlobalMapPose[];
  points: { total_count: number; displayed_count: number; display_xy: Array<[number, number]> };
  track_count: number;
  diagnostics: Record<string, unknown>;
  recommended_edit: { kind: string; pair_id: string | null; title: string; detail: string };
}

export interface DensePreviewRecord {
  schema: "panorai-streetview-dense-preview/v1";
  project_id: string;
  pair_id: string;
  revision: number;
  created_at: string;
  stale: boolean;
  engine: Record<string, unknown>;
  registration_revision: number;
  reference_image_id: string;
  source_image_id: string;
  preview_shape_hw: [number, number];
  scale_status: "metric" | "scale-free";
  units: string;
  input_mask: {
    eligible_pixel_count: number;
    eligible_fraction: number;
    include_annotation_ids: string[];
    exclude_annotation_ids: string[];
    semantics: string;
  };
  source_input_mask: {
    eligible_pixel_count: number;
    eligible_fraction: number;
    include_annotation_ids: string[];
    exclude_annotation_ids: string[];
    semantics: string;
  };
  valid_pixel_count: number;
  valid_fraction: number;
  valid_outside_input_count: number;
  median_range: number | null;
  median_confidence: number | null;
  range_min: number | null;
  range_max: number | null;
  reverse: {
    scale_status: "metric" | "scale-free";
    units: string;
    valid_pixel_count: number;
    valid_fraction: number;
    valid_outside_input_count: number;
    median_range: number | null;
    median_confidence: number | null;
    range_min: number | null;
    range_max: number | null;
    pose_provenance: string;
  };
  diagnostics: Record<string, unknown>;
}

export interface DenseSceneRecord {
  schema: "panorai-streetview-dense-scene/v1";
  project_id: string;
  revision: number;
  created_at: string;
  stale: boolean;
  profile: "interactive" | "reviewed";
  engine: Record<string, unknown>;
  global_map_revision: number;
  global_map_created_at: string | null;
  global_map_scale: string;
  scene: {
    units: string;
    frame: string;
    view_count: number;
    view_ids: string[];
    shape_hw: [number, number];
  };
  view_masks: Array<{
    image_id: string;
    eligible_pixel_count: number;
    eligible_fraction: number;
    include_annotation_ids: string[];
    exclude_annotation_ids: string[];
    semantics: string;
  }>;
  status: "accepted" | "rejected";
  accepted: boolean;
  reasons: string[];
  range_maps: Array<{
    image_id: string;
    valid_pixel_count: number;
    valid_fraction: number;
    median_range_m: number | null;
    median_confidence: number | null;
    source_ids: string[];
  }>;
  surface: {
    file_available: boolean;
    vertex_count: number;
    face_count: number;
    bounds_xyz: [number[], number[]] | null;
    colored_vertex_fraction: number;
    display_xy: Array<[number, number]>;
    display_rgb: Array<[number, number, number]>;
  };
  diagnostics: Record<string, unknown>;
}

export interface StitchingGroup {
  group_id: string;
  engine: Record<string, unknown>;
  status: "accepted" | "rejected";
  accepted: boolean;
  reasons: string[];
  sources: Array<{
    image_id: string;
    filename: string;
    sha256: string;
    shape_hw: [number, number];
    working_shape_hw: [number, number];
    horizontal_fov_deg: number | null;
    fov_source: "supplied" | "estimated";
  }>;
  output: {
    shape_hw: [number, number];
    support_fraction: number;
    valid_fraction: number;
    overlap_pixel_fraction: number;
    file_available: boolean;
  } | null;
  diagnostics: {
    input_image_count: number;
    feature_counts: number[];
    candidate_pair_count: number;
    accepted_pair_count: number;
    rejected_pairs: string[];
    reference_image_id: string;
    estimated_fov_count: number;
    rotation_initial_cost: number;
    rotation_final_cost: number;
    coverage: number;
    overlap_fraction: number;
    gains: number[];
    model: string;
  } | null;
  intrinsics: Array<{
    horizontal_fov_deg: number;
    vertical_fov_deg: number;
    fx: number;
    fy: number;
    cx: number;
    cy: number;
  }>;
  orientations: Array<{ image_id: string; camera_to_world: number[][] }>;
  pairwise: Array<{
    image_a_id: string;
    image_b_id: string;
    num_inliers: number;
    median_angular_error_deg: number;
    estimated_fov: boolean;
    observable: boolean;
    spatial_coverage: number;
  }>;
}

export interface StitchingRecord {
  schema: "panorai-streetview-stitching/v2";
  project_id: string;
  revision: number;
  created_at: string;
  stale: boolean;
  workflow: "perspective_calibrated" | "perspective_uncalibrated";
  model: "central-rotation-only";
  engine: Record<string, unknown>;
  status: "accepted" | "partial" | "rejected";
  accepted_group_count: number;
  rejected_group_count: number;
  ready_for_promotion: boolean;
  promoted_project_id: string | null;
  groups: StitchingGroup[];
}

export interface BuildStage {
  id: string;
  title: string;
  status: "ready" | "requires_input" | "blocked" | "planned" | "optional";
  engine: string;
  stability: "stable" | "experimental" | "application";
  detail: string;
}

export interface BuildPlan {
  project_id: string;
  workflow: Workflow;
  can_build_now: boolean;
  next_action: string;
  stages: BuildStage[];
  warnings: string[];
}

export interface Capabilities {
  runtime: { version: string; module_file: string };
  policy: string;
  engines: Record<string, { available: boolean; public_api: string; stability: string; diagnostic: string | null }>;
}

const WORKFLOW_COPY: Record<Workflow, { title: string; detail: string; badge: string }> = {
  spherical_registered: {
    title: "ERPs registradas",
    detail: "Panoramas esféricos com poses e pares de navegação conhecidos.",
    badge: "rota direta",
  },
  spherical_unregistered: {
    title: "Registro a estimar",
    detail: "Panoramas sem poses: pares candidatos, pose relativa e grafo global.",
    badge: "estimar poses",
  },
  perspective_calibrated: {
    title: "Fotos calibradas",
    detail: "Grupo pinhole com FOV ou fx conhecido; overlap e orientações SO(3) são estimados.",
    badge: "converter em ERP",
  },
  perspective_uncalibrated: {
    title: "Fotos sem calibração",
    detail: "Calibração pinhole central: o FOV compartilhado é estimado quando o overlap é observável.",
    badge: "calibrar + converter",
  },
};

export function workflowCopy(workflow: Workflow) {
  return WORKFLOW_COPY[workflow];
}

export function sphericalPairCandidates(filenames: string[]): IntakeImagePair[] {
  const candidates: IntakeImagePair[] = [];
  for (let reference = 0; reference < filenames.length; reference += 1) {
    for (let source = reference + 1; source < filenames.length; source += 1) {
      candidates.push({
        id: `pair-${String(candidates.length + 1).padStart(3, "0")}`,
        reference_filename: filenames[reference],
        source_filename: filenames[source],
        provenance: { selection: "user-intake" },
      });
    }
  }
  return candidates;
}

export function createMutationQueue() {
  let tail: Promise<void> = Promise.resolve();
  return function enqueue<T>(mutation: () => Promise<T>): Promise<T> {
    const result = tail.then(mutation, mutation);
    tail = result.then(
      () => undefined,
      () => undefined,
    );
    return result;
  };
}

export function essentialControlState(
  operationRunning: boolean,
  hasRegistration: boolean,
): { runDisabled: boolean; reestimateDisabled: boolean } {
  return {
    runDisabled: operationRunning,
    reestimateDisabled: operationRunning || !hasRegistration,
  };
}

export function stitchingControlState(input: {
  operationRunning: boolean;
  perspectiveProject: boolean;
  canBuild: boolean;
  promotionReady: boolean;
  stitchingStale: boolean;
}): { runDisabled: boolean; promoteDisabled: boolean; buildDisabled: boolean } {
  return {
    runDisabled: input.operationRunning || !input.perspectiveProject || !input.canBuild,
    promoteDisabled: input.operationRunning || input.stitchingStale || !input.promotionReady,
    buildDisabled: input.operationRunning || !input.canBuild,
  };
}

export function isMetricMapScale(scale: string | null | undefined): boolean {
  return typeof scale === "string" && scale.startsWith("metric");
}

export function denseAnchorStatus(input: {
  boundCount: number;
  requiredCount: number;
  mapScale: string | null | undefined;
  appliedCount: number;
}): string {
  if (isMetricMapScale(input.mapScale)) {
    return `${input.appliedCount} priors aplicados · mapa em escala métrica.`;
  }
  const action = input.boundCount >= input.requiredCount
    ? "execute novamente o mapa"
    : "adicione medidas em inliers distintos";
  return `${input.boundCount}/${input.requiredCount} âncoras ligadas a tracks · ${action}.`;
}

export function metrologyCalibrationIsUsable(
  calibrationImageId: string | null,
  editImageId: string | null,
  draftDirty: boolean,
): boolean {
  return !draftDirty
    && calibrationImageId !== null
    && calibrationImageId === editImageId;
}

export function invalidatePendingAction(sequence: number): number {
  return sequence + 1;
}

export function replaceCachedRecord<T extends { id: string; updated_at: string }>(
  records: T[],
  updated: T,
): T[] {
  return records.map((record) => (
    record.id === updated.id && updated.updated_at > record.updated_at
      ? updated
      : record
  ));
}

export function replaceActiveRecord<T extends { id: string; updated_at: string }>(
  active: T | null,
  updated: T,
): T | null {
  return active?.id === updated.id && updated.updated_at > active.updated_at
    ? updated
    : active;
}

export function pairRequestIsCurrent(
  request: { projectId: string; pairId: string; sequence: number },
  current: { projectId: string | null; pairId: string | null; sequence: number },
): boolean {
  return request.projectId === current.projectId
    && request.pairId === current.pairId
    && request.sequence === current.sequence;
}

export function projectRequestIsCurrent(
  request: { projectId: string; sequence: number },
  current: { projectId: string | null; sequence: number },
): boolean {
  return request.projectId === current.projectId
    && request.sequence === current.sequence;
}

export function imageRequestIsCurrent(
  request: { projectId: string; imageId: string; sequence: number },
  current: { projectId: string | null; imageId: string | null; sequence: number },
): boolean {
  return request.projectId === current.projectId
    && request.imageId === current.imageId
    && request.sequence === current.sequence;
}

export function mergeCaptureGroupMetadata(
  rawMetadata: string,
  assignments: Array<{ filename: string; captureGroup: string }>,
): string {
  const parsed: unknown = JSON.parse(rawMetadata.trim() || "{}");
  if (!parsed || typeof parsed !== "object" || Array.isArray(parsed)) {
    throw new Error("Metadados devem ser um objeto JSON por nome de arquivo.");
  }
  const metadata = parsed as Record<string, unknown>;
  for (const assignment of assignments) {
    const group = assignment.captureGroup.trim();
    if (!/^[A-Za-z0-9_-]+$/.test(group)) {
      throw new Error(`Grupo inválido para ${assignment.filename}. Use letras, números, hífen ou underscore.`);
    }
    const existing = metadata[assignment.filename] ?? {};
    if (!existing || typeof existing !== "object" || Array.isArray(existing)) {
      throw new Error(`Metadados de ${assignment.filename} devem ser um objeto JSON.`);
    }
    metadata[assignment.filename] = {
      ...(existing as Record<string, unknown>),
      capture_group: group,
    };
  }
  return JSON.stringify(metadata);
}

export function denseSceneGuidance(summary: SceneSetup["summary"]): {
  state: "ready" | "map" | "scale" | "views";
  title: string;
  detail: string;
  action: string;
} {
  if (summary.dense_scene_eligible) {
    return {
      state: "ready",
      title: "Cena pronta para MVS + TSDF",
      detail: "O mapa é atual, métrico e contém ao menos três câmeras. Revise as máscaras antes da reconstrução.",
      action: "Revisar máscaras",
    };
  }
  if (summary.dense_scene_blockers.includes("current-global-map-required")) {
    return {
      state: "map",
      title: "Primeiro feche o mapa global",
      detail: `O componente aceito contém ${summary.accepted_component_size}/${summary.image_count} vistas; a reconstrução não usa poses parciais.`,
      action: "Ir ao mapa",
    };
  }
  if (summary.dense_scene_blockers.includes("metric-scale-required")) {
    const missing = Math.max(0, summary.range_prior_min_count - summary.bound_range_prior_count);
    return {
      state: "scale",
      title: missing ? `Faltam ${missing} medidas radiais` : "Âncoras prontas para o mapa",
      detail: missing
        ? `${summary.bound_range_prior_count}/${summary.range_prior_min_count} âncoras estão ligadas a inliers distintos. Sem escala, o TSDF não pode ser criado.`
        : "Reexecute o mapa global para aplicar os SphericalRangePrior e obter coordenadas métricas.",
      action: missing ? "Adicionar âncora" : "Aplicar no mapa",
    };
  }
  return {
    state: "views",
    title: "Conecte ao menos três panoramas",
    detail: "O perfil público de reconstrução densa exige três vistas registradas no mesmo frame.",
    action: "Abrir fila de pares",
  };
}

export function pairAssessment(pair: Pick<ImagePairRecord, "relation" | "minimum_directional_overlap">): {
  label: string;
  shortLabel: string;
  tone: "strong" | "useful" | "limited" | "unknown";
  guidance: string;
} {
  const overlap = pair.minimum_directional_overlap;
  if (overlap == null) {
    return {
      label: pair.relation === "registered_overlap" ? "Pose fornecida; sobreposição não medida" : "Pose ainda não estimada",
      shortLabel: pair.relation === "registered_overlap" ? "não medido" : "aguardando pose",
      tone: "unknown",
      guidance: pair.relation === "registered_overlap"
        ? "Inspecione lado a lado e registre evidência antes de aceitar este vizinho."
        : "Compare o par candidato; a estimativa deve vir das APIs públicas PanorAi.",
    };
  }
  if (overlap >= 0.85) {
    return {
      label: "Registro com alta sobreposição",
      shortLabel: "overlap alto",
      tone: "strong",
      guidance: "Comece pela sobreposição; use correção manual somente se estruturas estáveis divergirem.",
    };
  }
  if (overlap >= 0.65) {
    return {
      label: "Sobreposição útil para inspeção",
      shortLabel: "overlap útil",
      tone: "useful",
      guidance: "Compare regiões compartilhadas e revise matches ou bounding boxes se houver deriva no registro.",
    };
  }
  return {
    label: "Sobreposição limitada",
    shortLabel: "revisar par",
    tone: "limited",
    guidance: "Confirme se este vizinho deve permanecer no grafo antes de aplicar qualquer ajuste.",
  };
}

export function sourcePixelPoints(
  points: Array<{ x: number; y: number }>,
  display: { width: number; height: number },
  source: { width: number; height: number },
): Array<[number, number]> {
  return points.map((point) => [
    (point.x / display.width) * source.width,
    (point.y / display.height) * source.height,
  ]);
}

export interface SphericalMeasurement {
  angleDeg: number;
  pixelDistance: number;
  horizontalWrappedPixels: number;
  deltaYPixels: number;
}

export function erpAngularMeasurement(
  start: [number, number],
  end: [number, number],
  image: { width: number; height: number },
): SphericalMeasurement {
  if (!(image.width > 0) || !(image.height > 0)) throw new Error("ERP dimensions must be positive");
  const clamp = (value: number, size: number) => Math.max(0, Math.min(size - 1, value));
  const a: [number, number] = [clamp(start[0], image.width), clamp(start[1], image.height)];
  const b: [number, number] = [clamp(end[0], image.width), clamp(end[1], image.height)];
  const ray = ([x, y]: [number, number]): [number, number, number] => {
    const longitude = ((x + 0.5) / image.width) * 2 * Math.PI - Math.PI;
    const latitude = Math.PI / 2 - ((y + 0.5) / image.height) * Math.PI;
    const cosLatitude = Math.cos(latitude);
    return [cosLatitude * Math.sin(longitude), Math.sin(latitude), cosLatitude * Math.cos(longitude)];
  };
  const ra = ray(a);
  const rb = ray(b);
  const dot = Math.max(-1, Math.min(1, ra[0] * rb[0] + ra[1] * rb[1] + ra[2] * rb[2]));
  const rawDx = Math.abs(b[0] - a[0]);
  const wrappedDx = Math.min(rawDx, image.width - rawDx);
  const dy = b[1] - a[1];
  return {
    angleDeg: Math.acos(dot) * 180 / Math.PI,
    pixelDistance: Math.hypot(wrappedDx, dy),
    horizontalWrappedPixels: wrappedDx,
    deltaYPixels: dy,
  };
}
