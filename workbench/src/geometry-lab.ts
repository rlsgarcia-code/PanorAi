import "./geometry.css";

import { clamp, cross, normalize, rayToViewPixel, wrapDegrees } from "./geometry.ts";
import type { Point2, Vec3, ViewState } from "./model.ts";
import { renderGnomonic } from "./renderer.ts";
import type { PanoramaPixels } from "./renderer.ts";

interface FeaturePayload {
  id: string;
  erp_xy: [number, number] | null;
  bearing_xyz: [number, number, number];
  response: number;
  scale: number;
  angle_deg: number;
  face_id: string;
  valid: boolean;
}

interface MatchPayload {
  index: number;
  feature_index_a: number;
  feature_index_b: number;
  bearing_a: [number, number, number];
  bearing_b: [number, number, number];
  descriptor_distance: number;
  ratio_score: number | null;
  valid: boolean;
  inlier: boolean;
  residual_deg: number | null;
  epipolar_plane_normal_b: [number, number, number] | null;
}

interface PosePayload {
  rotation_b_from_a: number[][];
  translation_direction_b: number[];
  essential_matrix: number[][];
  accepted: boolean;
  rejection_reasons: string[];
  num_inliers: number;
  num_trials: number;
  median_parallax_deg: number;
  cheirality_ratio: number;
  degenerate: boolean;
  degeneracy_reasons: string[];
}

interface PairAnalysis {
  schema: string;
  profile: string;
  features: { a: FeaturePayload[]; b: FeaturePayload[] };
  matches: MatchPayload[];
  pose: PosePayload | null;
  interpretation: Record<string, string>;
}

interface ParameterSpec { name: string; label: string; kind: string; default: unknown; choices?: string[] }
interface ExperimentComponent { id:string; title:string; surface_id:string; family:string; public_api:string; summary:string; availability:string; adapter_status:string; parameter_schema:ParameterSpec[]; input_modes:string[] }
interface RunArtifact { id:string; type:string; label:string; mime_type:string }
interface PersistedRun { id:string; experiment_id:string; workspace:string; status:string; created_at:string; parameters:Record<string,unknown>; artifacts:RunArtifact[]; metrics:Record<string,unknown>; events:Array<{state:string;message:string}>; error?:string }

interface PairImage {
  file: File;
  name: string;
  canvas: HTMLCanvasElement;
  pixels: PanoramaPixels;
  nativeWidth: number;
  nativeHeight: number;
}

type Stage = "features" | "matches" | "pose" | "semantics" | "hypotheses";
type Side = "a" | "b";

const app = document.querySelector<HTMLDivElement>("#app");
if (!app) throw new Error("Missing application root.");

app.innerHTML = `
  <div class="geometry-shell">
    <header class="topbar">
      <div class="brand-block">
        <div class="mark" aria-hidden="true"><span></span><span></span></div>
        <div><strong>PanorAi Workbench</strong><small>Two View · par registrado</small></div>
      </div>
      <nav class="workspace-nav" aria-label="Ambiente">
        <a href="/">Catálogo</a>
        <a href="/single-view.html">Single View</a>
        <a class="active" href="/two-view.html">Two View</a>
        <span>Multi View · em breve</span>
      </nav>
      <div class="top-actions">
        <span class="backend-state" id="backendState"><i></i><span>Verificando backend</span></span>
      </div>
    </header>

    <main class="geometry-workspace">
      <aside class="stage-rail" aria-label="Etapas do pipeline">
        <small>Bancada Two View</small>
        <div class="two-parameter-panel" id="twoParameters"></div>
        <div class="stage-list">
          <button class="stage-button active" data-stage="features"><span class="stage-number">01</span><span class="stage-copy"><strong>Keypoints</strong><small>resposta, escala, orientação e face</small></span></button>
          <button class="stage-button" data-stage="matches"><span class="stage-number">02</span><span class="stage-copy"><strong>Matches</strong><small>distância, razão e validade</small></span></button>
          <button class="stage-button" data-stage="pose"><span class="stage-number">03</span><span class="stage-copy"><strong>R, t e erro</strong><small>inliers, epipolar e reprojeção</small></span></button>
          <button class="stage-button" data-stage="semantics" disabled><span class="stage-number">04</span><span class="stage-copy"><strong>CAM e regiões</strong><small>runner semântico ainda não ligado</small></span></button>
          <button class="stage-button" data-stage="hypotheses" disabled><span class="stage-number">05</span><span class="stage-copy"><strong>Hipóteses</strong><small>transferência A↔B e fusão</small></span></button>
        </div>
        <div class="two-history"><span>RUNS PERSISTIDOS</span><div id="twoRuns"></div></div>
      </aside>

      <section class="pair-area" aria-label="Comparação do par">
        <div class="pair-toolbar">
          <input id="fileA" type="file" accept="image/png,image/jpeg,image/webp" hidden />
          <input id="fileB" type="file" accept="image/png,image/jpeg,image/webp" hidden />
          <button class="file-slot" id="chooseA"><b>A</b><span id="nameA">Escolher ERP</span></button>
          <button class="file-slot" id="chooseB"><b>B</b><span id="nameB">Escolher ERP</span></button>
          <span class="spacer"></span>
          <select id="experiment" class="profile-select" aria-label="Experimento"></select>
          <select id="target" class="profile-select" aria-label="Aplicar em" hidden><option value="a">somente A</option><option value="b">somente B</option><option value="both">A e B</option></select>
          <select id="profile" class="profile-select" aria-label="Perfil de features">
            <option value="relative-pose-reference">relative-pose-reference</option>
            <option value="sift-flann">sift-flann</option>
            <option value="orb-hamming">orb-hamming</option>
          </select>
          <button class="primary run-button" id="runPair" disabled>Executar par</button>
        </div>

        <div class="pair-views">
          <div class="pair-pane" id="paneA">
            <canvas id="canvasA" aria-label="Vista gnomônica A"></canvas>
            <div class="pane-label"><b>A</b><span id="paneNameA">sem imagem</span></div>
            <div class="pane-empty"><b>Imagem A</b><span>Carregue um panorama ERP</span></div>
          </div>
          <div class="pair-pane" id="paneB">
            <canvas id="canvasB" aria-label="Vista gnomônica B"></canvas>
            <div class="pane-label"><b>B</b><span id="paneNameB">sem imagem</span></div>
            <div class="pane-empty"><b>Imagem B</b><span>Carregue um panorama ERP</span></div>
          </div>
        </div>

        <div class="pair-footer">
          <label><input type="checkbox" id="lockViews" checked /> sincronizar orientação</label>
          <label>exibir
            <select id="pointLimit" class="profile-select">
              <option value="250">top 250</option>
              <option value="500" selected>top 500</option>
              <option value="1000">top 1000</option>
              <option value="0">todos</option>
            </select>
          </label>
          <span>Arraste para orientar · roda para FOV · clique num KP/match para inspecionar</span>
          <span class="coordinates" id="coordinates">yaw 0.0° · pitch 0.0° · FOV 72°</span>
        </div>
      </section>

      <aside class="evidence-inspector" aria-label="Evidência da etapa">
        <div class="evidence-head"><small id="stageEyebrow">Etapa 01</small><h1 id="stageTitle">Keypoints extraídos</h1></div>
        <div class="metric-strip">
          <div class="metric"><small>A</small><b id="metricA">—</b></div>
          <div class="metric"><small>B</small><b id="metricB">—</b></div>
          <div class="metric"><small id="metricCLabel">Matches</small><b id="metricC">—</b></div>
        </div>
        <div class="evidence-body" id="evidenceBody"><div class="evidence-note">Carregue duas ERPs e execute o perfil. O backend retornará cada estágio separadamente.</div></div>
        <div class="evidence-foot" id="evidenceFoot">Nenhuma estimativa foi executada.</div>
      </aside>
    </main>
  </div>
`;

function required<T extends Element>(selector: string): T {
  const value = document.querySelector<T>(selector);
  if (!value) throw new Error(`Missing ${selector}.`);
  return value;
}

const canvasA = required<HTMLCanvasElement>("#canvasA");
const canvasB = required<HTMLCanvasElement>("#canvasB");
const fileInputA = required<HTMLInputElement>("#fileA");
const fileInputB = required<HTMLInputElement>("#fileB");
const runButton = required<HTMLButtonElement>("#runPair");
const profileSelect = required<HTMLSelectElement>("#profile");
const experimentSelect = required<HTMLSelectElement>("#experiment");
const targetSelect = required<HTMLSelectElement>("#target");
const evidenceBody = required<HTMLDivElement>("#evidenceBody");
const evidenceFoot = required<HTMLDivElement>("#evidenceFoot");
const backendState = required<HTMLSpanElement>("#backendState");
const lockViews = required<HTMLInputElement>("#lockViews");

let imageA: PairImage | null = null;
let imageB: PairImage | null = null;
let analysis: PairAnalysis | null = null;
let genericRun: PersistedRun | null = null;
let stage: Stage = "features";
let selectedMatch = 0;
let selectedFeature: { side: Side; index: number } | null = null;
let rendering = false;
let components: ExperimentComponent[] = [];
let persistedRuns: PersistedRun[] = [];
const views: Record<Side, ViewState> = {
  a: { yawDeg: 0, pitchDeg: 0, horizontalFovDeg: 72 },
  b: { yawDeg: 0, pitchDeg: 0, horizontalFovDeg: 72 },
};
let drag: { side: Side; start: Point2; startView: ViewState } | null = null;
const adapterLabels: Record<string, string> = { ready: "pronto", "requires-input": "requer entrada", inspect: "inspecionar", blocked: "bloqueado" };

function selectedComponent(): ExperimentComponent | undefined {
  return components.find((item) => item.id === experimentSelect.value);
}

function parameterValue(spec: ParameterSpec): unknown {
  if (spec.name === "profile") return profileSelect.value;
  const control = document.querySelector<HTMLInputElement | HTMLSelectElement>(`[data-two-param="${spec.name}"]`);
  if (!control) return spec.default;
  if (spec.kind === "integer") return Number.parseInt(control.value, 10);
  if (spec.kind === "number") return Number.parseFloat(control.value);
  return control.value;
}

function resolvedTwoParameters(): Record<string, unknown> {
  const component = selectedComponent();
  const values: Record<string, unknown> = {};
  component?.parameter_schema.forEach((spec) => { values[spec.name] = parameterValue(spec); });
  if (component?.input_modes.includes("single")) values._target = targetSelect.value;
  return values;
}

function renderTwoParameters(): void {
  const component = selectedComponent();
  const host = required<HTMLDivElement>("#twoParameters");
  if (!component) { host.replaceChildren(); return; }
  host.innerHTML = `<strong>${component.title}</strong><small>${component.family} · ${adapterLabels[component.adapter_status] ?? component.adapter_status}</small>` + component.parameter_schema.filter((spec) => spec.name !== "profile").map((spec) => `
    <label>${spec.label}${spec.kind === "select"
      ? `<select data-two-param="${spec.name}">${(spec.choices ?? []).map((choice) => `<option value="${choice}" ${choice === spec.default ? "selected" : ""}>${choice}</option>`).join("")}</select>`
      : `<input data-two-param="${spec.name}" type="number" value="${String(spec.default ?? "")}" step="${spec.kind === "integer" ? "1" : "any"}"/>`}</label>
  `).join("");
  profileSelect.hidden = component.id !== "relative-pose";
  targetSelect.hidden = !component.input_modes.includes("single");
  runButton.disabled = !(imageA && imageB) || component.availability !== "available" || component.adapter_status !== "ready";
  history.replaceState(null, "", `/two-view.html?experiment=${encodeURIComponent(component.id)}`);
}

async function loadTwoRuns(): Promise<void> {
  const response = await fetch("/api/v1/runs?limit=30");
  if (!response.ok) return;
  const payload = await response.json() as { runs: PersistedRun[] };
  persistedRuns = payload.runs.filter((run) => run.workspace === "two" && run.experiment_id && components.some((item) => item.id === run.experiment_id));
  const host = required<HTMLDivElement>("#twoRuns"); host.replaceChildren();
  persistedRuns.slice(0, 8).forEach((run) => {
    const button = document.createElement("button"); button.className = "two-run";
    button.innerHTML = `<strong>${run.experiment_id}</strong><small>${run.status} · ${run.id.slice(-8)}</small>`;
    button.addEventListener("click", () => void openPersistedRun(run)); host.append(button);
  });
}

async function openPersistedRun(run: PersistedRun): Promise<void> {
  genericRun = run;
  const poseArtifact = run.artifacts.find((item) => item.type === "pose");
  if (poseArtifact) {
    const response = await fetch(`/api/v1/runs/${run.id}/artifacts/${poseArtifact.id}`);
    if (response.ok) {
      analysis = await response.json() as PairAnalysis;
      genericRun = null;
      selectedMatch = analysis.matches.find((match) => match.inlier)?.index ?? 0;
      stage = "features"; scheduleRender(); return;
    }
  }
  scheduleRender();
}

async function waitForPersistedRun(run: PersistedRun): Promise<PersistedRun> {
  let current = run;
  while (current.status === "created" || current.status === "running") {
    await new Promise((resolve) => window.setTimeout(resolve, 250));
    const response = await fetch(`/api/v1/runs/${current.id}`);
    if (!response.ok) throw new Error(`Não foi possível acompanhar ${current.id}`);
    current = await response.json() as PersistedRun;
    genericRun = current;
    scheduleRender();
  }
  return current;
}

async function loadWorkspace(): Promise<void> {
  const response = await fetch("/api/v1/workspaces/two");
  if (!response.ok) throw new Error(`HTTP ${response.status}`);
  const payload = await response.json() as { experiments: ExperimentComponent[] };
  components = payload.experiments;
  const families = [...new Set(components.map((item) => item.family))];
  experimentSelect.replaceChildren();
  families.forEach((family) => {
    const items = components.filter((item) => item.family === family);
    if (!items.length) return;
    const group = document.createElement("optgroup"); group.label = family;
    items.forEach((item) => { const option = document.createElement("option"); option.value = item.id; option.textContent = `${item.title}${item.availability === "available" && item.adapter_status !== "blocked" ? "" : " · indisponível"}`; group.append(option); });
    experimentSelect.append(group);
  });
  const wanted = new URLSearchParams(location.search).get("experiment");
  experimentSelect.value = components.some((item) => item.id === wanted) ? wanted! : "relative-pose";
  renderTwoParameters();
  await loadTwoRuns();
}

function canvasPoint(event: PointerEvent, canvas: HTMLCanvasElement): Point2 {
  const bounds = canvas.getBoundingClientRect();
  return { x: (event.clientX - bounds.left) / bounds.width * canvas.width, y: (event.clientY - bounds.top) / bounds.height * canvas.height };
}

function resize(canvas: HTMLCanvasElement): void {
  const bounds = canvas.getBoundingClientRect();
  const scale = Math.min(1, 680 / Math.max(bounds.width, 1));
  const width = Math.max(1, Math.round(bounds.width * scale));
  const height = Math.max(1, Math.round(bounds.height * scale));
  if (canvas.width !== width || canvas.height !== height) { canvas.width = width; canvas.height = height; }
}

async function loadImage(file: File): Promise<PairImage> {
  const url = URL.createObjectURL(file);
  try {
    const image = new Image();
    image.src = url;
    await image.decode();
    const width = Math.min(4096, image.naturalWidth);
    const height = Math.max(1, Math.round(width * image.naturalHeight / image.naturalWidth));
    const canvas = document.createElement("canvas");
    canvas.width = width; canvas.height = height;
    const context = canvas.getContext("2d", { willReadFrequently: true });
    if (!context) throw new Error("Canvas 2D unavailable.");
    context.drawImage(image, 0, 0, width, height);
    return {
      file,
      name: file.name,
      canvas,
      pixels: { width, height, data: context.getImageData(0, 0, width, height).data },
      nativeWidth: image.naturalWidth,
      nativeHeight: image.naturalHeight,
    };
  } finally {
    URL.revokeObjectURL(url);
  }
}

function featureList(side: Side): FeaturePayload[] {
  return analysis?.features[side] ?? [];
}

function pointLimit(): number {
  return Number(required<HTMLSelectElement>("#pointLimit").value);
}

function visibleFeatures(side: Side): Array<{ feature: FeaturePayload; index: number }> {
  const ordered = featureList(side)
    .map((feature, index) => ({ feature, index }))
    .sort((left, right) => right.feature.response - left.feature.response);
  const limit = pointLimit();
  return limit === 0 ? ordered : ordered.slice(0, limit);
}

function visibleMatches(): MatchPayload[] {
  if (!analysis) return [];
  const ordered = [...analysis.matches].sort((left, right) => {
    if (left.inlier !== right.inlier) return left.inlier ? -1 : 1;
    return (left.residual_deg ?? Number.POSITIVE_INFINITY) - (right.residual_deg ?? Number.POSITIVE_INFINITY);
  });
  const limit = pointLimit();
  const visible = limit === 0 ? ordered : ordered.slice(0, limit);
  const selected = analysis.matches[selectedMatch];
  return selected && !visible.includes(selected) ? [selected, ...visible] : visible;
}

function drawPoint(context: CanvasRenderingContext2D, point: Point2, color: string, radius: number, selected = false): void {
  context.beginPath(); context.arc(point.x, point.y, radius, 0, Math.PI * 2);
  context.fillStyle = color; context.fill();
  if (selected) { context.strokeStyle = "#ffffff"; context.lineWidth = 2; context.stroke(); }
}

function drawGreatCircle(context: CanvasRenderingContext2D, normalValue: Vec3, view: ViewState, size: { width: number; height: number }): void {
  const normal = normalize(normalValue);
  const reference = Math.abs(normal.y) < 0.9 ? { x: 0, y: 1, z: 0 } : { x: 1, y: 0, z: 0 };
  const u = normalize(cross(normal, reference));
  const v = normalize(cross(normal, u));
  context.save(); context.strokeStyle = "#ffc857"; context.lineWidth = 2; context.setLineDash([7, 5]);
  context.beginPath();
  let previous: Point2 | null = null;
  for (let index = 0; index <= 180; index += 1) {
    const angle = index / 180 * Math.PI * 2;
    const ray = { x: u.x * Math.cos(angle) + v.x * Math.sin(angle), y: u.y * Math.cos(angle) + v.y * Math.sin(angle), z: u.z * Math.cos(angle) + v.z * Math.sin(angle) };
    const point = rayToViewPixel(ray, size, view);
    if (!point || (previous && Math.hypot(point.x - previous.x, point.y - previous.y) > size.width * 0.35)) {
      previous = null;
      continue;
    }
    if (previous) context.lineTo(point.x, point.y); else context.moveTo(point.x, point.y);
    previous = point;
  }
  context.stroke(); context.restore();
}

function drawEvidence(context: CanvasRenderingContext2D, side: Side, canvas: HTMLCanvasElement): void {
  if (!analysis) return;
  if (stage === "features") {
    for (const { feature, index } of visibleFeatures(side)) {
      const point = rayToViewPixel({ x: feature.bearing_xyz[0], y: feature.bearing_xyz[1], z: feature.bearing_xyz[2] }, canvas, views[side]);
      if (point && point.x >= 0 && point.x <= canvas.width && point.y >= 0 && point.y <= canvas.height) {
        const selected = selectedFeature?.side === side && selectedFeature.index === index;
        drawPoint(context, point, selected ? "#ffc857" : "rgba(255,255,255,0.72)", selected ? 6 : clamp(1.5 + Math.log2(Math.max(feature.scale, 1)) * 0.35, 1.5, 4), selected);
      }
    }
    return;
  }
  for (const match of visibleMatches()) {
    const bearing = side === "a" ? match.bearing_a : match.bearing_b;
    const point = rayToViewPixel({ x: bearing[0], y: bearing[1], z: bearing[2] }, canvas, views[side]);
    if (!point || point.x < 0 || point.x > canvas.width || point.y < 0 || point.y > canvas.height) continue;
    drawPoint(context, point, match.inlier ? "#79ae70" : "#c76f60", match.index === selectedMatch ? 6 : 3, match.index === selectedMatch);
  }
  const selected = analysis.matches[selectedMatch];
  if (stage === "pose" && side === "b" && selected?.epipolar_plane_normal_b) {
    const [x, y, z] = selected.epipolar_plane_normal_b;
    drawGreatCircle(context, { x, y, z }, views.b, canvas);
  }
}

function renderPane(side: Side): void {
  const canvas = side === "a" ? canvasA : canvasB;
  const image = side === "a" ? imageA : imageB;
  resize(canvas);
  const context = canvas.getContext("2d");
  if (!context) return;
  context.fillStyle = "#060706"; context.fillRect(0, 0, canvas.width, canvas.height);
  if (!image) return;
  renderGnomonic(context, image.pixels, views[side], canvas.width, canvas.height);
  drawEvidence(context, side, canvas);
}

function matrixRows(matrix: number[][]): string {
  return matrix.map((row) => row.map((value) => value.toFixed(4).padStart(8, " ")).join(" ")).join("<br>");
}

function renderInspector(): void {
  const title = required<HTMLHeadingElement>("#stageTitle");
  const eyebrow = required<HTMLSpanElement>("#stageEyebrow");
  const metricA = required<HTMLElement>("#metricA");
  const metricB = required<HTMLElement>("#metricB");
  const metricC = required<HTMLElement>("#metricC");
  const metricCLabel = required<HTMLElement>("#metricCLabel");
  const titles: Record<Stage, [string, string]> = {
    features: ["Etapa 01", "Keypoints extraídos"],
    matches: ["Etapa 02", "Correspondências"],
    pose: ["Etapa 03", "Pose relativa e resíduos"],
    semantics: ["Etapa 04", "CAM e regiões"],
    hypotheses: ["Etapa 05", "Hipóteses multivista"],
  };
  [eyebrow.textContent, title.textContent] = titles[stage];
  metricA.textContent = analysis ? String(analysis.features.a.length) : "—";
  metricB.textContent = analysis ? String(analysis.features.b.length) : "—";
  metricC.textContent = analysis ? String(stage === "pose" ? analysis.pose?.num_inliers ?? 0 : analysis.matches.length) : "—";
  metricCLabel.textContent = stage === "pose" ? "Inliers" : "Matches";
  evidenceBody.replaceChildren();
  if (!analysis) {
    if (genericRun) {
      evidenceBody.innerHTML = `<div class="evidence-note">Run ${genericRun.status}. ${genericRun.error ?? `Métricas: ${JSON.stringify(genericRun.metrics)}`}</div><div class="artifact-links">${genericRun.artifacts.map((item) => `<a href="/api/v1/runs/${genericRun!.id}/artifacts/${item.id}" target="_blank">${item.label} · ${item.type}</a>`).join("")}</div>`;
      evidenceFoot.textContent = `${genericRun.experiment_id} · ${genericRun.id}`;
      return;
    }
    evidenceBody.innerHTML = '<div class="evidence-note">Carregue duas ERPs e execute o perfil. Nenhuma evidência é inferida no frontend.</div>';
    return;
  }
  if (stage === "features") {
    const chosen = selectedFeature ? featureList(selectedFeature.side)[selectedFeature.index] : null;
    evidenceBody.innerHTML = `<div class="evidence-note">O viewport mostra ${pointLimit() === 0 ? "todos os" : `os ${pointLimit()} maiores`} pontos por resposta. O total permanece nos indicadores; clique num ponto para inspecioná-lo.</div>`;
    if (chosen) {
      const bearing = chosen.bearing_xyz.map((value) => value.toFixed(5)).join("  ");
      evidenceBody.innerHTML += `<div class="matrix"><b>KP ${selectedFeature?.side.toUpperCase()} · ${chosen.id}</b>face ${chosen.face_id}<br>response ${chosen.response.toFixed(5)} · scale ${chosen.scale.toFixed(3)} · angle ${chosen.angle_deg.toFixed(2)}°<br>bearing ${bearing}</div>`;
    }
  } else if (stage === "pose") {
    if (!analysis.pose) {
      evidenceBody.innerHTML = '<div class="evidence-note">Nenhuma pose atingiu o mínimo do estimador. Matches continuam disponíveis para auditoria.</div>';
    } else {
      const pose = analysis.pose;
      evidenceBody.innerHTML = `
        <div class="evidence-note">${pose.accepted ? "Pose aceita pela política explícita do PanorAi." : `Pose retornada, mas rejeitada: ${pose.rejection_reasons.join(", ") || "sem motivo serializado"}.`}</div>
        <div class="matrix"><b>R · panorama B a partir de A</b>${matrixRows(pose.rotation_b_from_a)}</div>
        <div class="matrix"><b>t · direção, sem escala métrica</b>${pose.translation_direction_b.map((value) => value.toFixed(5)).join("  ")}</div>
      `;
    }
  } else {
    evidenceBody.innerHTML = '<div class="evidence-note">Verde: inlier geométrico. Vermelho: match fora do consenso. Selecione uma linha para localizar o par nas duas vistas.</div>';
  }
  if (stage === "matches" || stage === "pose") {
    const list = document.createElement("div"); list.className = "match-list";
    visibleMatches().forEach((match) => {
      const button = document.createElement("button"); button.className = "match-row"; button.classList.toggle("selected", match.index === selectedMatch);
      button.innerHTML = `<span class="match-id">#${String(match.index).padStart(4, "0")}</span><span><strong>${match.residual_deg === null ? "sem pose" : `${match.residual_deg.toFixed(3)}°`}</strong><br><small>d=${match.descriptor_distance.toFixed(2)}${match.ratio_score === null ? "" : ` · r=${match.ratio_score.toFixed(3)}`}</small></span><i class="match-dot ${match.inlier ? "inlier" : ""}"></i>`;
      button.addEventListener("click", () => { selectedMatch = match.index; focusSelected(); scheduleRender(); });
      list.append(button);
    });
    evidenceBody.append(list);
  }
  const poseState = analysis.pose ? (analysis.pose.accepted ? "aceita" : "rejeitada") : "ausente";
  evidenceFoot.textContent = `Perfil ${analysis.profile}. Pose ${poseState}. Zero residual é evidência válida; suporte e validade são campos separados.`;
}

function focusSelected(): void {
  const match = analysis?.matches[selectedMatch];
  if (!match) return;
  const bearingToAngles = (bearing: [number, number, number]) => ({
    yawDeg: Math.atan2(bearing[0], bearing[2]) * 180 / Math.PI,
    pitchDeg: Math.asin(clamp(bearing[1], -1, 1)) * 180 / Math.PI,
  });
  Object.assign(views.a, bearingToAngles(match.bearing_a));
  Object.assign(views.b, bearingToAngles(match.bearing_b));
}

function render(): void {
  rendering = false;
  renderPane("a"); renderPane("b"); renderInspector();
  required<HTMLSpanElement>("#coordinates").textContent = `yaw ${views.a.yawDeg.toFixed(1)}° · pitch ${views.a.pitchDeg.toFixed(1)}° · FOV ${views.a.horizontalFovDeg.toFixed(0)}°`;
}

function scheduleRender(): void {
  if (rendering) return;
  rendering = true;
  requestAnimationFrame(render);
}

function bindCanvas(canvas: HTMLCanvasElement, side: Side): void {
  canvas.addEventListener("pointerdown", (event) => {
    const point = canvasPoint(event, canvas);
    if (analysis && stage === "features") {
      let nearest: { index: number; distance: number } | null = null;
      for (const { feature, index } of visibleFeatures(side)) {
        const [x, y, z] = feature.bearing_xyz;
        const projected = rayToViewPixel({ x, y, z }, canvas, views[side]);
        if (!projected) continue;
        const distance = Math.hypot(projected.x - point.x, projected.y - point.y);
        if (distance < 12 && (!nearest || distance < nearest.distance)) nearest = { index, distance };
      }
      if (nearest) { selectedFeature = { side, index: nearest.index }; scheduleRender(); return; }
    }
    if (analysis && (stage === "matches" || stage === "pose")) {
      let best: { index: number; distance: number } | null = null;
      for (const match of analysis.matches) {
        const bearing = side === "a" ? match.bearing_a : match.bearing_b;
        const projected = rayToViewPixel({ x: bearing[0], y: bearing[1], z: bearing[2] }, canvas, views[side]);
        if (!projected) continue;
        const distance = Math.hypot(projected.x - point.x, projected.y - point.y);
        if (distance < 12 && (!best || distance < best.distance)) best = { index: match.index, distance };
      }
      if (best) { selectedMatch = best.index; focusSelected(); scheduleRender(); return; }
    }
    canvas.setPointerCapture(event.pointerId);
    drag = { side, start: point, startView: { ...views[side] } };
  });
  canvas.addEventListener("pointermove", (event) => {
    if (!drag || drag.side !== side) return;
    const point = canvasPoint(event, canvas);
    const sensitivity = drag.startView.horizontalFovDeg / canvas.width;
    const next = {
      yawDeg: wrapDegrees(drag.startView.yawDeg - (point.x - drag.start.x) * sensitivity),
      pitchDeg: clamp(drag.startView.pitchDeg + (point.y - drag.start.y) * sensitivity, -89, 89),
    };
    views[side].yawDeg = next.yawDeg; views[side].pitchDeg = next.pitchDeg;
    if (lockViews.checked) {
      const other: Side = side === "a" ? "b" : "a";
      views[other].yawDeg = next.yawDeg; views[other].pitchDeg = next.pitchDeg;
    }
    scheduleRender();
  });
  canvas.addEventListener("pointerup", () => { drag = null; });
  canvas.addEventListener("wheel", (event) => {
    event.preventDefault();
    const fov = clamp(views[side].horizontalFovDeg * Math.exp(event.deltaY * 0.001), 20, 120);
    views[side].horizontalFovDeg = fov;
    if (lockViews.checked) views[side === "a" ? "b" : "a"].horizontalFovDeg = fov;
    scheduleRender();
  }, { passive: false });
}

async function assignFile(side: Side, file: File): Promise<void> {
  const loaded = await loadImage(file);
  if (side === "a") imageA = loaded; else imageB = loaded;
  required<HTMLSpanElement>(`#name${side.toUpperCase()}`).textContent = loaded.name;
  required<HTMLSpanElement>(`#paneName${side.toUpperCase()}`).textContent = `${loaded.name} · ${loaded.nativeWidth}×${loaded.nativeHeight}`;
  required<HTMLDivElement>(`#pane${side.toUpperCase()}`).classList.add("loaded");
  analysis = null;
  genericRun = null;
  renderTwoParameters();
  scheduleRender();
}

required<HTMLButtonElement>("#chooseA").addEventListener("click", () => fileInputA.click());
required<HTMLButtonElement>("#chooseB").addEventListener("click", () => fileInputB.click());
fileInputA.addEventListener("change", () => { const file = fileInputA.files?.[0]; if (file) void assignFile("a", file); });
fileInputB.addEventListener("change", () => { const file = fileInputB.files?.[0]; if (file) void assignFile("b", file); });
required<HTMLSelectElement>("#pointLimit").addEventListener("change", scheduleRender);
experimentSelect.addEventListener("change", renderTwoParameters);

document.querySelectorAll<HTMLButtonElement>(".stage-button").forEach((button) => {
  button.addEventListener("click", () => {
    stage = button.dataset.stage as Stage;
    document.querySelectorAll(".stage-button").forEach((item) => item.classList.remove("active"));
    button.classList.add("active");
    scheduleRender();
  });
});

runButton.addEventListener("click", async () => {
  if (!imageA || !imageB) return;
  const component = selectedComponent();
  if (!component) return;
  runButton.disabled = true; runButton.textContent = "Processando…";
  evidenceFoot.textContent = `Executando ${component.public_api}…`;
  const form = new FormData();
  form.append("experiment_id", component.id); form.append("workspace", "two");
  form.append("parameters", JSON.stringify(resolvedTwoParameters()));
  form.append("image_a", imageA.file); form.append("image_b", imageB.file);
  try {
    const response = await fetch("/api/v1/runs", { method: "POST", body: form });
    if (!response.ok) { const error = await response.json(); throw new Error(error.detail ?? `HTTP ${response.status}`); }
    let run = await response.json() as PersistedRun;
    genericRun = run; scheduleRender();
    run = await waitForPersistedRun(run);
    if (run.status === "failed") throw new Error(run.error ?? "Falha ao executar o experimento");
    await loadTwoRuns();
    await openPersistedRun(run);
    if (component.id === "relative-pose" && analysis) {
      stage = "features";
      document.querySelectorAll(".stage-button").forEach((item) => item.classList.toggle("active", (item as HTMLElement).dataset.stage === stage));
    }
  } catch (error) {
    evidenceBody.innerHTML = `<div class="evidence-note">${error instanceof Error ? error.message : "Falha ao executar o par."}</div>`;
    evidenceFoot.textContent = "Execução falhou sem produzir estimativa.";
  } finally {
    runButton.disabled = false; runButton.textContent = "Executar par"; scheduleRender();
  }
});

async function checkBackend(): Promise<void> {
  try {
    const response = await fetch("/api/v1/capabilities");
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    const payload = await response.json() as { panorai_version: string };
    backendState.classList.add("ready");
    backendState.querySelector("span")!.textContent = `PanorAi ${payload.panorai_version}`;
  } catch {
    backendState.classList.add("error");
    backendState.querySelector("span")!.textContent = "Backend indisponível";
  }
}

bindCanvas(canvasA, "a"); bindCanvas(canvasB, "b");
new ResizeObserver(scheduleRender).observe(required<HTMLElement>(".pair-views"));
void Promise.all([checkBackend(), loadWorkspace()]);
scheduleRender();
