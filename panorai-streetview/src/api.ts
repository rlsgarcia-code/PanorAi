import type {
  BuildPlan,
  Capabilities,
  DenseAnnotation,
  DensePreviewRecord,
  DenseSceneRecord,
  FloorCalibrationRecord,
  FloorMeasurement,
  GlobalMapRecord,
  PairRegistration,
  ProjectRecord,
  RegistrationHistory,
  RegionMode,
  ReviewState,
  SceneSetup,
  StitchingRecord,
  MetrologyLine,
  IntakeImagePair,
  VerticalMeasurement,
  Workflow,
} from "./model.ts";

async function decode<T>(response: Response): Promise<T> {
  const payload = await response.json() as T & { detail?: string };
  if (!response.ok) throw new Error(payload.detail ?? `HTTP ${response.status}`);
  return payload;
}

export async function listProjects(): Promise<ProjectRecord[]> {
  const payload = await decode<{ projects: ProjectRecord[] }>(await fetch("/api/v1/projects"));
  return payload.projects;
}

export async function loadCapabilities(): Promise<Capabilities> {
  return decode<Capabilities>(await fetch("/api/v1/capabilities"));
}

export async function loadPlan(projectId: string): Promise<BuildPlan> {
  return decode<BuildPlan>(await fetch(`/api/v1/projects/${projectId}/build-plan`));
}

export async function loadSetup(projectId: string): Promise<SceneSetup> {
  return decode<SceneSetup>(await fetch(`/api/v1/projects/${projectId}/setup`));
}

export async function loadGlobalMap(projectId: string): Promise<GlobalMapRecord | null> {
  const response = await fetch(`/api/v1/projects/${projectId}/global-map`);
  if (response.status === 404) return null;
  return decode<GlobalMapRecord | null>(response);
}

export async function runGlobalMap(projectId: string): Promise<GlobalMapRecord> {
  return decode<GlobalMapRecord>(await fetch(`/api/v1/projects/${projectId}/global-map`, { method: "POST" }));
}

export async function createProject(input: {
  name: string;
  workflow: Workflow;
  files: File[];
  metadata: string;
  pairs: IntakeImagePair[];
}): Promise<ProjectRecord> {
  const form = new FormData();
  form.append("name", input.name);
  form.append("workflow", input.workflow);
  form.append("metadata", input.metadata || "{}");
  form.append("pairs", JSON.stringify(input.pairs));
  input.files.forEach((file) => form.append("images", file));
  return decode<ProjectRecord>(await fetch("/api/v1/projects", { method: "POST", body: form }));
}

export async function addObjectRegion(input: {
  projectId: string;
  imageId: string;
  objectId: string;
  label: string;
  bbox: [number, number, number, number];
}): Promise<ProjectRecord> {
  return decode<ProjectRecord>(await fetch(`/api/v1/projects/${input.projectId}/object-regions`, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({
      object_id: input.objectId,
      label: input.label,
      image_id: input.imageId,
      bbox_xyxy: input.bbox,
      state: "accepted",
    }),
  }));
}

export async function addManualCorrespondence(input: {
  projectId: string;
  pairId: string;
  reference: [number, number];
  source: [number, number];
}): Promise<ProjectRecord> {
  return decode<ProjectRecord>(await fetch(`/api/v1/projects/${input.projectId}/manual-correspondences`, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({
      pair_id: input.pairId,
      reference_xy: input.reference,
      source_xy: input.source,
      state: "accepted",
    }),
  }));
}

export async function addDenseAnnotation(input: {
  projectId: string;
  imageId: string;
  kind: DenseAnnotation["kind"];
  points: Array<[number, number]>;
  label: string;
  radialRange?: number;
}): Promise<ProjectRecord> {
  return decode<ProjectRecord>(await fetch(`/api/v1/projects/${input.projectId}/dense-annotations`, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({
      kind: input.kind,
      image_id: input.imageId,
      label: input.label,
      points_xy: input.points,
      radial_range: input.radialRange ?? null,
      state: "accepted",
    }),
  }));
}

export async function loadDensePreview(projectId: string, pairId: string): Promise<DensePreviewRecord | null> {
  const response = await fetch(`/api/v1/projects/${projectId}/pairs/${pairId}/dense-preview`);
  if (response.status === 404) return null;
  return decode<DensePreviewRecord | null>(response);
}

export async function runDensePreview(projectId: string, pairId: string): Promise<DensePreviewRecord> {
  return decode<DensePreviewRecord>(await fetch(`/api/v1/projects/${projectId}/pairs/${pairId}/dense-preview`, { method: "POST" }));
}

export function densePreviewImageUrl(projectId: string, pairId: string, revision: number): string {
  return `/api/v1/projects/${projectId}/pairs/${pairId}/dense-preview/image?revision=${revision}`;
}

export async function loadDenseScene(projectId: string): Promise<DenseSceneRecord | null> {
  const response = await fetch(`/api/v1/projects/${projectId}/dense-scene`);
  if (response.status === 404) return null;
  return decode<DenseSceneRecord | null>(response);
}

export async function runDenseScene(projectId: string, profile: "interactive" | "reviewed" = "interactive"): Promise<DenseSceneRecord> {
  return decode<DenseSceneRecord>(await fetch(`/api/v1/projects/${projectId}/dense-scene`, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({ profile }),
  }));
}

export function denseSceneMeshUrl(projectId: string, revision: number): string {
  return `/api/v1/projects/${projectId}/dense-scene/mesh?revision=${revision}`;
}

export async function loadStitching(projectId: string): Promise<StitchingRecord | null> {
  const response = await fetch(`/api/v1/projects/${projectId}/stitching`);
  if (response.status === 404) return null;
  return decode<StitchingRecord | null>(response);
}

export async function runStitching(projectId: string): Promise<StitchingRecord> {
  return decode<StitchingRecord>(await fetch(`/api/v1/projects/${projectId}/stitching`, { method: "POST" }));
}

export function stitchingImageUrl(projectId: string, groupId: string, revision: number): string {
  return `/api/v1/projects/${projectId}/stitching/groups/${encodeURIComponent(groupId)}/image?revision=${revision}`;
}

export async function promoteStitching(projectId: string): Promise<ProjectRecord> {
  return decode<ProjectRecord>(await fetch(`/api/v1/projects/${projectId}/stitching/promote`, { method: "POST" }));
}

export async function loadPairRegistration(projectId: string, pairId: string): Promise<PairRegistration | null> {
  const response = await fetch(`/api/v1/projects/${projectId}/pairs/${pairId}/registration`);
  if (response.status === 404) return null;
  return decode<PairRegistration | null>(response);
}

export async function loadRegistrationHistory(projectId: string, pairId: string): Promise<RegistrationHistory> {
  return decode<RegistrationHistory>(await fetch(`/api/v1/projects/${projectId}/pairs/${pairId}/registration/history`));
}

export async function runPairRegistration(projectId: string, pairId: string, regionMode: RegionMode): Promise<PairRegistration> {
  return decode<PairRegistration>(await fetch(`/api/v1/projects/${projectId}/pairs/${pairId}/registration`, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({ region_mode: regionMode }),
  }));
}

export async function reestimatePairRegistration(projectId: string, pairId: string, regionMode: RegionMode): Promise<PairRegistration> {
  return decode<PairRegistration>(await fetch(`/api/v1/projects/${projectId}/pairs/${pairId}/registration/reestimate`, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({ region_mode: regionMode }),
  }));
}

export async function reviewMatch(input: {
  projectId: string;
  pairId: string;
  matchIndex: number;
  state: ReviewState;
  revision: number;
}): Promise<PairRegistration> {
  return decode<PairRegistration>(await fetch(`/api/v1/projects/${input.projectId}/pairs/${input.pairId}/registration/matches/${input.matchIndex}`, {
    method: "PATCH",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({ state: input.state, revision: input.revision }),
  }));
}

export async function createSceneManifest(projectId: string): Promise<Record<string, unknown>> {
  return decode<Record<string, unknown>>(await fetch(`/api/v1/projects/${projectId}/scene-manifest`, { method: "POST" }));
}

export async function loadFloorCalibration(projectId: string, imageId: string): Promise<FloorCalibrationRecord | null> {
  const response = await fetch(`/api/v1/projects/${projectId}/metrology/${imageId}`);
  if (response.status === 404) return null;
  return decode<FloorCalibrationRecord | null>(response);
}

export async function calibrateFloor(input: {
  projectId: string;
  imageId: string;
  cameraHeightM: number;
  floorHint: [number, number];
  lines: MetrologyLine[];
}): Promise<FloorCalibrationRecord> {
  return decode<FloorCalibrationRecord>(await fetch(`/api/v1/projects/${input.projectId}/metrology/calibrate`, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({
      image_id: input.imageId,
      camera_height_m: input.cameraHeightM,
      floor_hint_xy: input.floorHint,
      lines: input.lines,
    }),
  }));
}

export async function measureFloorPoint(input: {
  projectId: string;
  imageId: string;
  calibrationRevision: number;
  point: [number, number];
}): Promise<FloorMeasurement> {
  return decode<FloorMeasurement>(await fetch(`/api/v1/projects/${input.projectId}/metrology/floor-point`, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({
      image_id: input.imageId,
      calibration_revision: input.calibrationRevision,
      point_xy: input.point,
    }),
  }));
}

export async function measureVertical(input: {
  projectId: string;
  imageId: string;
  calibrationRevision: number;
  base: [number, number];
  top: [number, number];
}): Promise<VerticalMeasurement> {
  return decode<VerticalMeasurement>(await fetch(`/api/v1/projects/${input.projectId}/metrology/vertical`, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({
      image_id: input.imageId,
      calibration_revision: input.calibrationRevision,
      base_xy: input.base,
      top_xy: input.top,
    }),
  }));
}
