import "./academy.css";

import {
  academyCross,
  academyDot,
  academyNormalize,
  angleDeg,
  canonicalErpPixelToLonLat,
  canonicalErpPixelToRay,
  canonicalRayToErpPixel,
  greatCircleNormal,
  sampleGreatCircle,
  vanishingDirection,
} from "./academy-geometry.ts";
import type { Point2, Vec3 } from "./model.ts";

if (new URLSearchParams(location.search).get("embedded") === "1") document.body.classList.add("embedded");

type Lesson = "projective" | "rays" | "lines" | "vanishing" | "calibration";
type Group = "x" | "z";

interface Segment {
  id: string;
  group: Group;
  start: Point2;
  end: Point2;
}

interface AcademyStatus {
  panorai_version: string;
  available: Array<{ id: string; public_api: string; stability: string }>;
  requires_panorai_update: Array<{ id: string; reason: string; owner: string }>;
}

interface LessonGeometry {
  normals: Vec3[];
  vpX: Vec3 | null;
  vpZ: Vec3 | null;
  horizon: Vec3 | null;
}

const app = document.querySelector<HTMLDivElement>("#app");
if (!app) throw new Error("Missing application root.");

function get2DContext(target: HTMLCanvasElement): CanvasRenderingContext2D {
  const value = target.getContext("2d");
  if (!value) throw new Error("Canvas 2D unavailable.");
  return value;
}

app.innerHTML = `
  <div class="academy-shell">
    <header class="academy-topbar">
      <a class="academy-brand" href="/" aria-label="PanorAi Workbench">
        <span class="academy-mark" aria-hidden="true"><i></i><i></i></span>
        <span><strong>PanorAi Workbench</strong><small>Academia experimental</small></span>
      </a>
      <nav class="academy-nav" aria-label="Áreas do Workbench">
        <a href="/">Laboratório</a>
        <a class="active" href="/academy.html">Academia</a>
        <a href="/geometry.html">Detecção</a>
        <a href="/segmentation.html">Segmentação</a>
      </nav>
      <div class="academy-runtime" id="runtimeState"><i></i><span>PanorAI Main · verificando</span></div>
    </header>

    <main class="academy-workspace">
      <aside class="course-rail">
        <div class="course-heading">
          <span>TRILHA 01 · VISÃO ÚNICA</span>
          <h1>Single view<br/>geometry</h1>
          <p>Da coordenada ERP à direção de fuga, sem esconder a geometria esférica.</p>
        </div>
        <div class="course-progress"><span><i></i></span><small>fundamentos + 4 conceitos</small></div>
        <nav class="lesson-list" aria-label="Conceitos">
          <button class="lesson active" data-lesson="projective"><b>00</b><span><strong>Geometria projetiva</strong><small>homogêneas e infinito</small></span><em>base</em></button>
          <button class="lesson" data-lesson="rays"><b>01</b><span><strong>Pixels → raios</strong><small>lat/lon e bearing XYZ</small></span><em>pronto</em></button>
          <button class="lesson" data-lesson="lines"><b>02</b><span><strong>Linhas → planos</strong><small>great circles na esfera</small></span><em>pronto</em></button>
          <button class="lesson" data-lesson="vanishing"><b>03</b><span><strong>Vanishing directions</strong><small>interseção antipodal</small></span><em>pronto</em></button>
          <button class="lesson" data-lesson="calibration"><b>04</b><span><strong>Calibração</strong><small>direções ortogonais e K</small></span><em class="agent">agent</em></button>
        </nav>
        <div class="source-note">
          <span>REFERÊNCIA</span>
          <strong>Hartley & Zisserman</strong>
          <p>Cap. 2: plano projetivo. Cap. 8: pontos de fuga, linhas de fuga e calibração por uma vista.</p>
        </div>
      </aside>

      <section class="learning-stage" aria-label="Experimento interativo">
        <header class="experiment-toolbar">
          <div>
            <span class="experiment-kicker">CASO 01 / CONSTRUÇÃO MANUAL</span>
            <strong id="experimentTitle">Posicione um bearing na esfera</strong>
          </div>
          <div class="toolbar-actions">
            <input id="imageInput" type="file" accept="image/png,image/jpeg,image/webp" hidden />
            <button class="grid-toggle active" id="toggleGrid" aria-pressed="true"><i></i> Grade lat/lon</button>
            <button class="quiet-action" id="loadDemo">Restaurar demo</button>
            <button class="upload-action" id="openImage">Abrir panorama <span>↗</span></button>
          </div>
        </header>

        <div class="panorama-stage" id="panoramaStage">
          <canvas id="panoramaCanvas" aria-label="Panorama ERP para seleção de linhas"></canvas>
          <div class="stage-badge"><span>ERP</span><b id="imageName">atrium-study.synthetic</b></div>
          <div class="coordinate-hover" id="coordinateHover">lon — · lat —</div>
          <div class="bearing-sphere" aria-label="Visualização 3D do bearing selecionado">
            <div><span id="sphereTitle">BEARING 3D</span><b id="sphereReadout">[0, 0, 1]</b></div>
            <canvas id="bearingCanvas"></canvas>
          </div>
          <div class="stage-instruction"><i></i><span id="instructionText"><b>MODO PONTO</b> · clique em qualquer posição da ERP.</span></div>
          <div class="vp-legend"><span><i class="legend-x"></i> direção X</span><span><i class="legend-z"></i> direção Z</span><span><i class="legend-h"></i> linha de fuga</span></div>
        </div>

        <div class="selection-strip" id="selectionStrip">
          <div class="selection-copy"><span id="selectionTitle">ENTRADA · UM PONTO</span><p id="selectionHelp">Clique para converter um centro de pixel em direção 3D.</p></div>
          <div class="selection-content" id="selectionContent"></div>
          <button id="clearSelection" class="clear-action">Centralizar</button>
        </div>

        <div class="evidence-bar">
          <div><span id="metric1Label">LONGITUDE λ</span><strong id="metric1Value">—</strong></div>
          <div><span id="metric2Label">LATITUDE φ</span><strong id="metric2Value">—</strong></div>
          <div><span id="metric3Label">BEARING XYZ</span><strong id="metric3Value">—</strong></div>
          <div class="evidence-source"><span id="sourceLabel">FONTE DO RAIO</span><strong id="raySource">prévia local</strong></div>
        </div>
      </section>

      <aside class="concept-panel" aria-label="Caderno matemático">
        <div class="concept-head"><span id="conceptStep">CONCEITO 01</span><small>notação adaptada à ERP</small></div>
        <div class="concept-scroll">
          <section class="concept-intro" id="conceptIntro"></section>
          <section class="formula-block" id="formulaBlock"></section>
          <section class="live-values" id="liveValues"></section>
          <section class="interpretation" id="interpretation"></section>
          <section class="library-boundary" id="libraryBoundary"></section>
        </div>
        <div class="bearing-geometry-popover" id="bearingGeometryTooltip" role="tooltip">
          <header><span>DECOMPOSIÇÃO DO BEARING</span><b>S² → altura + plano XZ</b></header>
          <svg viewBox="0 0 520 255" role="img" aria-labelledby="bearingDiagramTitle bearingDiagramDescription">
            <title id="bearingDiagramTitle">Projeções triangulares de um bearing sobre a esfera unitária</title>
            <desc id="bearingDiagramDescription">O primeiro triângulo decompõe o raio unitário em altura seno de phi e projeção horizontal cosseno de phi. O segundo decompõe essa projeção horizontal nos eixos X e Z usando lambda.</desc>
            <defs>
              <radialGradient id="sphereWash" cx="36%" cy="30%" r="70%"><stop offset="0" stop-color="#fff" stop-opacity=".12"/><stop offset="1" stop-color="#fff" stop-opacity=".015"/></radialGradient>
              <marker id="diagramArrow" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="5" markerHeight="5" orient="auto-start-reverse"><path d="M0 0L10 5L0 10Z" fill="currentColor"/></marker>
            </defs>
            <g class="diagram-half">
              <text class="diagram-step" x="20" y="22">1 · ELEVAÇÃO φ</text>
              <circle class="sphere-fill" cx="125" cy="133" r="88"/>
              <ellipse class="sphere-grid" cx="125" cy="133" rx="88" ry="30"/>
              <ellipse class="sphere-grid sphere-meridian" cx="125" cy="133" rx="31" ry="88"/>
              <path class="axis-line" d="M28 133H225M125 229V35"/>
              <path class="projection-line" d="M125 133H178V61"/>
              <path class="bearing-line" d="M125 133L178 61"/>
              <path class="angle-arc" d="M153 133A28 28 0 0 0 142 111"/>
              <path class="right-angle" d="M168 133V123H178"/>
              <circle class="bearing-point" cx="178" cy="61" r="4"/>
              <text class="diagram-label" x="183" y="59">b</text>
              <text class="diagram-angle" x="151" y="116">φ</text>
              <text class="diagram-component" x="144" y="148">ρ = cos φ</text>
              <text class="diagram-component" x="184" y="100">Y = sin φ</text>
              <text class="diagram-muted" x="30" y="148">equador</text>
              <text class="diagram-unit" x="139" y="91">‖b‖ = 1</text>
            </g>
            <path class="diagram-transfer" d="M236 133H274" marker-end="url(#diagramArrow)"/>
            <g class="diagram-half">
              <text class="diagram-step" x="286" y="22">2 · AZIMUTE λ NO PLANO XZ</text>
              <ellipse class="plane-disc" cx="390" cy="133" rx="99" ry="82"/>
              <path class="axis-line" d="M390 226V43M286 160H486"/>
              <path class="projection-line" d="M390 154V87H447"/>
              <path class="bearing-line" d="M390 154L447 87"/>
              <path class="angle-arc" d="M390 126A28 28 0 0 1 408 133"/>
              <path class="right-angle" d="M390 97H400V87"/>
              <circle class="bearing-point" cx="447" cy="87" r="4"/>
              <text class="diagram-axis x" x="474" y="154">+X</text>
              <text class="diagram-axis z" x="397" y="51">+Z</text>
              <text class="diagram-angle" x="401" y="122">λ</text>
              <text class="diagram-component" x="404" y="169">X = ρ sin λ</text>
              <text class="diagram-component" x="311" y="117">Z = ρ cos λ</text>
              <text class="diagram-unit" x="424" y="119">ρ</text>
            </g>
          </svg>
          <div class="diagram-result">
            <span><i>1</i><b>Incline por φ</b><small>Y = sin φ · ρ = cos φ</small></span>
            <span><i>2</i><b>Gire ρ por λ</b><small>X = ρ sin λ · Z = ρ cos λ</small></span>
          </div>
          <footer><small>SELEÇÃO ATUAL</small><span id="diagramAngles">λ — · φ —</span><b id="diagramVector">b = [—, —, —]</b></footer>
        </div>
        <footer class="concept-foot"><span>Leitura + construção + evidência</span><button id="nextConcept">Próximo conceito <b>→</b></button></footer>
      </aside>
    </main>
  </div>
`;

function required<T extends Element>(selector: string): T {
  const element = document.querySelector<T>(selector);
  if (!element) throw new Error(`Missing element ${selector}.`);
  return element;
}

const canvas = required<HTMLCanvasElement>("#panoramaCanvas");
const context = get2DContext(canvas);
const bearingCanvas = required<HTMLCanvasElement>("#bearingCanvas");
const bearingContext = get2DContext(bearingCanvas);
const imageInput = required<HTMLInputElement>("#imageInput");
const sourceCanvas = document.createElement("canvas");
const sourceContext = get2DContext(sourceCanvas);

const lessonOrder: Lesson[] = ["projective", "rays", "lines", "vanishing", "calibration"];
const groupColors: Record<Group, string> = { x: "#ffca5c", z: "#67d6c5" };
let lesson: Lesson = "projective";
let selectedSegment = 0;
let backendReady = false;
let status: AcademyStatus | null = null;
let dragStart: Point2 | null = null;
let segments: Segment[] = [];
let rays: Vec3[] = [];
let selectedPoint: Point2 = { x: 699.5, y: 349.5 };
let pointRay: Vec3 = { x: 0, y: 0, z: 1 };
let gridEnabled = true;

function demoSegments(): Segment[] {
  return [
    { id: "X₁", group: "x", start: { x: 154, y: 218 }, end: { x: 555, y: 310 } },
    { id: "X₂", group: "x", start: { x: 171, y: 510 }, end: { x: 555, y: 387 } },
    { id: "Z₁", group: "z", start: { x: 845, y: 309 }, end: { x: 1258, y: 205 } },
    { id: "Z₂", group: "z", start: { x: 845, y: 391 }, end: { x: 1240, y: 526 } },
  ];
}

function makeDemoPanorama(): void {
  sourceCanvas.width = 1400;
  sourceCanvas.height = 700;
  const sky = sourceContext.createLinearGradient(0, 0, 0, 700);
  sky.addColorStop(0, "#26302e");
  sky.addColorStop(0.46, "#535447");
  sky.addColorStop(1, "#171b19");
  sourceContext.fillStyle = sky;
  sourceContext.fillRect(0, 0, 1400, 700);

  const glow = sourceContext.createRadialGradient(700, 330, 30, 700, 330, 590);
  glow.addColorStop(0, "rgba(244,224,176,.62)");
  glow.addColorStop(0.35, "rgba(180,169,132,.19)");
  glow.addColorStop(1, "rgba(5,8,8,.6)");
  sourceContext.fillStyle = glow;
  sourceContext.fillRect(0, 0, 1400, 700);

  sourceContext.fillStyle = "rgba(17,21,20,.72)";
  sourceContext.beginPath(); sourceContext.moveTo(0, 0); sourceContext.lineTo(570, 292); sourceContext.lineTo(570, 410); sourceContext.lineTo(0, 700); sourceContext.closePath(); sourceContext.fill();
  sourceContext.beginPath(); sourceContext.moveTo(1400, 0); sourceContext.lineTo(830, 292); sourceContext.lineTo(830, 410); sourceContext.lineTo(1400, 700); sourceContext.closePath(); sourceContext.fill();
  sourceContext.fillStyle = "rgba(225,211,174,.14)";
  sourceContext.fillRect(570, 292, 260, 118);
  sourceContext.fillStyle = "rgba(8,11,11,.64)";
  sourceContext.fillRect(640, 294, 120, 116);

  sourceContext.lineWidth = 3;
  for (let index = 0; index < 5; index += 1) {
    const offset = index * 55;
    sourceContext.strokeStyle = `rgba(228,218,187,${0.15 + index * 0.035})`;
    sourceContext.beginPath(); sourceContext.moveTo(0, 120 + offset); sourceContext.lineTo(570, 315 + offset * 0.12); sourceContext.stroke();
    sourceContext.beginPath(); sourceContext.moveTo(1400, 120 + offset); sourceContext.lineTo(830, 315 + offset * 0.12); sourceContext.stroke();
  }
  sourceContext.strokeStyle = "rgba(248,231,180,.28)";
  sourceContext.lineWidth = 2;
  for (let x = 80; x < 1400; x += 132) {
    sourceContext.beginPath(); sourceContext.moveTo(x, 90); sourceContext.lineTo(700 + (x - 700) * 0.18, 350); sourceContext.stroke();
    sourceContext.beginPath(); sourceContext.moveTo(x, 650); sourceContext.lineTo(700 + (x - 700) * 0.16, 360); sourceContext.stroke();
  }
  sourceContext.fillStyle = "rgba(255,255,255,.035)";
  for (let y = 80; y < 680; y += 22) sourceContext.fillRect(0, y, 1400, 1);

  segments = demoSegments();
  selectedSegment = 0;
  selectedPoint = { x: 699.5, y: 349.5 };
  required<HTMLElement>("#imageName").textContent = "atrium-study.synthetic";
  syncCanvasSize();
  syncRays();
}

function syncCanvasSize(): void {
  const bounds = canvas.getBoundingClientRect();
  const targetWidth = Math.max(720, Math.round(bounds.width));
  const targetHeight = Math.max(360, Math.round(bounds.height));
  if (canvas.width !== targetWidth || canvas.height !== targetHeight) {
    canvas.width = targetWidth;
    canvas.height = targetHeight;
  }
  render();
}

function sourceToCanvas(point: Point2): Point2 {
  return {
    x: (point.x + 0.5) / sourceCanvas.width * canvas.width,
    y: (point.y + 0.5) / sourceCanvas.height * canvas.height,
  };
}

function eventToSource(event: PointerEvent): Point2 {
  const bounds = canvas.getBoundingClientRect();
  return {
    x: Math.max(-0.5, Math.min(sourceCanvas.width - 0.5, (event.clientX - bounds.left) / bounds.width * sourceCanvas.width - 0.5)),
    y: Math.max(-0.5, Math.min(sourceCanvas.height - 0.5, (event.clientY - bounds.top) / bounds.height * sourceCanvas.height - 0.5)),
  };
}

function drawPolyline(points: Point2[], color: string, width: number, dashed = false): void {
  context.save();
  context.strokeStyle = color;
  context.lineWidth = width;
  context.setLineDash(dashed ? [7, 6] : []);
  context.beginPath();
  let previous: Point2 | null = null;
  for (const sourcePoint of points) {
    const point = sourceToCanvas(sourcePoint);
    if (!previous || Math.abs(sourcePoint.x - previous.x) > sourceCanvas.width * 0.45) context.moveTo(point.x, point.y);
    else context.lineTo(point.x, point.y);
    previous = sourcePoint;
  }
  context.stroke();
  context.restore();
}

function formatDegrees(value: number, positive: string, negative: string): string {
  const suffix = value > 0.0001 ? positive : value < -0.0001 ? negative : "";
  return `${Math.abs(value).toFixed(1)}°${suffix}`;
}

function drawLatLonGrid(): void {
  if (!gridEnabled) return;
  context.save();
  context.font = "600 8px ui-monospace, monospace";
  context.textBaseline = "top";
  for (let longitude = -150; longitude <= 180; longitude += 30) {
    const sourceX = (longitude + 180) / 360 * sourceCanvas.width - 0.5;
    const x = sourceToCanvas({ x: sourceX, y: 0 }).x;
    context.strokeStyle = longitude === 0 ? "rgba(255,202,92,.55)" : "rgba(236,241,231,.20)";
    context.lineWidth = longitude === 0 ? 1.3 : 0.75;
    context.beginPath(); context.moveTo(x, 0); context.lineTo(x, canvas.height); context.stroke();
    if (longitude < 180) {
      const label = longitude === 0 ? "0°" : `${Math.abs(longitude)}°${longitude < 0 ? "W" : "E"}`;
      const width = context.measureText(label).width;
      context.fillStyle = "rgba(5,8,7,.68)"; context.fillRect(x - width / 2 - 4, 31, width + 8, 14);
      context.fillStyle = longitude === 0 ? "#ffca5c" : "rgba(238,242,235,.72)";
      context.fillText(label, x - width / 2, 34);
    }
  }
  for (let latitude = -75; latitude <= 75; latitude += 15) {
    const sourceY = (90 - latitude) / 180 * sourceCanvas.height - 0.5;
    const y = sourceToCanvas({ x: 0, y: sourceY }).y;
    context.strokeStyle = latitude === 0 ? "rgba(255,202,92,.55)" : "rgba(236,241,231,.20)";
    context.lineWidth = latitude === 0 ? 1.3 : 0.75;
    context.beginPath(); context.moveTo(0, y); context.lineTo(canvas.width, y); context.stroke();
    const label = latitude === 0 ? "0°" : `${Math.abs(latitude)}°${latitude < 0 ? "S" : "N"}`;
    const width = context.measureText(label).width;
    context.fillStyle = "rgba(5,8,7,.68)"; context.fillRect(7, y - 7, width + 8, 14);
    context.fillStyle = latitude === 0 ? "#ffca5c" : "rgba(238,242,235,.72)";
    context.fillText(label, 11, y - 5);
  }
  context.restore();
}

function drawSelectedPoint(): void {
  const point = sourceToCanvas(selectedPoint);
  context.save();
  context.strokeStyle = "rgba(255,255,255,.95)";
  context.fillStyle = "#ffca5c";
  context.lineWidth = 1.5;
  context.shadowColor = "rgba(255,202,92,.7)";
  context.shadowBlur = 14;
  context.beginPath(); context.arc(point.x, point.y, 7, 0, Math.PI * 2); context.fill(); context.stroke();
  context.shadowBlur = 0;
  context.beginPath(); context.moveTo(point.x - 17, point.y); context.lineTo(point.x - 9, point.y); context.moveTo(point.x + 9, point.y); context.lineTo(point.x + 17, point.y); context.moveTo(point.x, point.y - 17); context.lineTo(point.x, point.y - 9); context.moveTo(point.x, point.y + 9); context.lineTo(point.x, point.y + 17); context.stroke();
  context.fillStyle = "#ffca5c"; context.font = "650 9px ui-monospace, monospace";
  context.fillText("p → d", point.x + 13, point.y - 13);
  context.restore();
}

function sphereProject(ray: Vec3, radius: number, center: Point2): { x: number; y: number; depth: number } {
  const yaw = -35 * Math.PI / 180;
  const pitch = 18 * Math.PI / 180;
  const x1 = Math.cos(yaw) * ray.x + Math.sin(yaw) * ray.z;
  const z1 = -Math.sin(yaw) * ray.x + Math.cos(yaw) * ray.z;
  const y1 = Math.cos(pitch) * ray.y - Math.sin(pitch) * z1;
  const z2 = Math.sin(pitch) * ray.y + Math.cos(pitch) * z1;
  return { x: center.x + x1 * radius, y: center.y - y1 * radius, depth: z2 };
}

function spherePath(points: Vec3[], color: string, width: number): void {
  const center = { x: bearingCanvas.width / 2, y: bearingCanvas.height / 2 + 4 };
  const radius = Math.min(bearingCanvas.width, bearingCanvas.height) * 0.36;
  bearingContext.beginPath();
  points.forEach((ray, index) => {
    const point = sphereProject(ray, radius, center);
    if (index === 0) bearingContext.moveTo(point.x, point.y); else bearingContext.lineTo(point.x, point.y);
  });
  bearingContext.strokeStyle = color; bearingContext.lineWidth = width; bearingContext.stroke();
}

function sphereDepthPath(points: Vec3[], color: string, width: number, frontOpacity = 1, backOpacity = .25): void {
  const center = { x: bearingCanvas.width / 2, y: bearingCanvas.height / 2 + 4 };
  const radius = Math.min(bearingCanvas.width, bearingCanvas.height) * 0.36;
  const projected = points.map((ray) => sphereProject(ray, radius, center));
  for (const front of [false, true]) {
    bearingContext.save();
    bearingContext.strokeStyle = color;
    bearingContext.globalAlpha = front ? frontOpacity : backOpacity;
    bearingContext.lineWidth = width * devicePixelRatio;
    bearingContext.lineCap = "round";
    bearingContext.setLineDash(front ? [] : [3 * devicePixelRatio, 4 * devicePixelRatio]);
    for (let index = 1; index < projected.length; index += 1) {
      const first = projected[index - 1];
      const second = projected[index];
      if (((first.depth + second.depth) / 2 >= 0) !== front) continue;
      bearingContext.beginPath(); bearingContext.moveTo(first.x, first.y); bearingContext.lineTo(second.x, second.y); bearingContext.stroke();
    }
    bearingContext.restore();
  }
}

function fillSpherePlane(points: Vec3[], color: string): void {
  const center = { x: bearingCanvas.width / 2, y: bearingCanvas.height / 2 + 4 };
  const radius = Math.min(bearingCanvas.width, bearingCanvas.height) * 0.36;
  bearingContext.save();
  bearingContext.beginPath();
  points.forEach((ray, index) => {
    const point = sphereProject(ray, radius, center);
    if (index === 0) bearingContext.moveTo(point.x, point.y); else bearingContext.lineTo(point.x, point.y);
  });
  bearingContext.closePath(); bearingContext.fillStyle = color; bearingContext.globalAlpha = .075; bearingContext.fill();
  bearingContext.restore();
}

function drawSphereRay(ray: Vec3, color: string, label: string, radius: number, center: Point2): void {
  const end = sphereProject(ray, radius, center);
  bearingContext.beginPath(); bearingContext.moveTo(center.x, center.y); bearingContext.lineTo(end.x, end.y);
  bearingContext.strokeStyle = color; bearingContext.lineWidth = 1.5 * devicePixelRatio; bearingContext.stroke();
  bearingContext.beginPath(); bearingContext.arc(end.x, end.y, 3.2 * devicePixelRatio, 0, Math.PI * 2);
  bearingContext.fillStyle = color; bearingContext.fill();
  bearingContext.font = `${7 * devicePixelRatio}px ui-monospace, monospace`;
  bearingContext.fillText(label, end.x + 4 * devicePixelRatio, end.y - 3 * devicePixelRatio);
}

function renderBearingSphere(geometry: LessonGeometry): void {
  const bounds = bearingCanvas.getBoundingClientRect();
  bearingCanvas.width = Math.max(160, Math.round(bounds.width * devicePixelRatio));
  bearingCanvas.height = Math.max(160, Math.round(bounds.height * devicePixelRatio));
  bearingContext.clearRect(0, 0, bearingCanvas.width, bearingCanvas.height);
  const center = { x: bearingCanvas.width / 2, y: bearingCanvas.height / 2 + 4 };
  const radius = Math.min(bearingCanvas.width, bearingCanvas.height) * 0.36;
  bearingContext.save();
  bearingContext.setLineDash([4 * devicePixelRatio, 4 * devicePixelRatio]);
  for (const latitude of [-60, -30, 0, 30, 60]) {
    const phi = latitude * Math.PI / 180;
    spherePath(Array.from({ length: 73 }, (_, index) => {
      const lambda = index / 72 * Math.PI * 2 - Math.PI;
      return { x: Math.sin(lambda) * Math.cos(phi), y: Math.sin(phi), z: Math.cos(lambda) * Math.cos(phi) };
    }), latitude === 0 ? "rgba(255,202,92,.42)" : "rgba(235,241,231,.18)", latitude === 0 ? 1.3 : 1);
  }
  for (const longitude of [-120, -60, 0, 60, 120]) {
    const lambda = longitude * Math.PI / 180;
    spherePath(Array.from({ length: 49 }, (_, index) => {
      const phi = -Math.PI / 2 + index / 48 * Math.PI;
      return { x: Math.sin(lambda) * Math.cos(phi), y: Math.sin(phi), z: Math.cos(lambda) * Math.cos(phi) };
    }), longitude === 0 ? "rgba(255,202,92,.42)" : "rgba(235,241,231,.18)", longitude === 0 ? 1.3 : 1);
  }
  bearingContext.setLineDash([]);
  const axes: Array<{ ray: Vec3; label: string; color: string }> = [
    { ray: { x: 1, y: 0, z: 0 }, label: "+X", color: "#ee8874" },
    { ray: { x: 0, y: 1, z: 0 }, label: "+Y", color: "#9bd887" },
    { ray: { x: 0, y: 0, z: 1 }, label: "+Z", color: "#84b8ec" },
  ];
  bearingContext.font = `${8 * devicePixelRatio}px ui-monospace, monospace`;
  axes.forEach(({ ray, label, color }) => {
    const end = sphereProject(ray, radius * 1.12, center);
    bearingContext.beginPath(); bearingContext.moveTo(center.x, center.y); bearingContext.lineTo(end.x, end.y);
    bearingContext.strokeStyle = color; bearingContext.lineWidth = 1.2 * devicePixelRatio; bearingContext.stroke();
    bearingContext.fillStyle = color; bearingContext.fillText(label, end.x + 3 * devicePixelRatio, end.y);
  });
  if (lesson === "lines") {
    const normal = geometry.normals[selectedSegment];
    const magnitude = normal ? Math.hypot(normal.x, normal.y, normal.z) : 0;
    const color = groupColors[segments[selectedSegment].group];
    if (normal && magnitude > .5) {
      const circle = sampleGreatCircle(normal, 180);
      fillSpherePlane(circle, color);
      geometry.normals.forEach((other, index) => {
        if (index === selectedSegment || Math.hypot(other.x, other.y, other.z) < .5) return;
        sphereDepthPath(sampleGreatCircle(other, 120), groupColors[segments[index].group], .75, .24, .09);
      });
      sphereDepthPath(circle, color, 2.8, 1, .34);
      const firstRay = rays[selectedSegment * 2];
      const secondRay = rays[selectedSegment * 2 + 1];
      if (firstRay) drawSphereRay(firstRay, color, "d₁", radius, center);
      if (secondRay) drawSphereRay(secondRay, color, "d₂", radius, center);
    }
    required<HTMLElement>("#sphereTitle").textContent = "GREAT CIRCLE 3D";
    required<HTMLElement>("#sphereReadout").textContent = `${segments[selectedSegment].id} · nᵀd=0`;
  } else if (lesson === "vanishing" || lesson === "calibration") {
    geometry.normals.forEach((normal, index) => {
      if (Math.hypot(normal.x, normal.y, normal.z) < .5) return;
      sphereDepthPath(sampleGreatCircle(normal, 120), groupColors[segments[index].group], .9, .38, .12);
    });
    if (geometry.vpX) drawSphereRay(geometry.vpX, groupColors.x, "vₓ", radius, center);
    if (geometry.vpZ) drawSphereRay(geometry.vpZ, groupColors.z, "vᵤ", radius, center);
    required<HTMLElement>("#sphereTitle").textContent = "DIREÇÕES DE FUGA";
    required<HTMLElement>("#sphereReadout").textContent = "±v";
  } else {
    const bearingEnd = sphereProject(pointRay, radius, center);
    if (lesson === "projective") {
    const opposite = sphereProject({ x: -pointRay.x, y: -pointRay.y, z: -pointRay.z }, radius, center);
    bearingContext.save();
    bearingContext.setLineDash([4 * devicePixelRatio, 4 * devicePixelRatio]);
    bearingContext.beginPath(); bearingContext.moveTo(center.x, center.y); bearingContext.lineTo(opposite.x, opposite.y);
    bearingContext.strokeStyle = "rgba(255,202,92,.58)"; bearingContext.lineWidth = 1.6 * devicePixelRatio; bearingContext.stroke();
    bearingContext.beginPath(); bearingContext.arc(opposite.x, opposite.y, 3 * devicePixelRatio, 0, Math.PI * 2);
    bearingContext.fillStyle = "rgba(255,202,92,.72)"; bearingContext.fill();
    bearingContext.restore();
    }
    bearingContext.beginPath(); bearingContext.moveTo(center.x, center.y); bearingContext.lineTo(bearingEnd.x, bearingEnd.y);
    bearingContext.strokeStyle = "#ffca5c"; bearingContext.lineWidth = 2.5 * devicePixelRatio; bearingContext.stroke();
    bearingContext.beginPath(); bearingContext.arc(bearingEnd.x, bearingEnd.y, 4 * devicePixelRatio, 0, Math.PI * 2);
    bearingContext.fillStyle = "#ffca5c"; bearingContext.fill();
    required<HTMLElement>("#sphereTitle").textContent = lesson === "projective" ? "DIREÇÃO PROJETIVA" : "BEARING 3D";
    required<HTMLElement>("#sphereReadout").textContent = lesson === "projective" ? "d ~ −d" : formatVector(pointRay, 2);
  }
  bearingContext.restore();
}

function currentGeometry(): LessonGeometry {
  const normals: Vec3[] = [];
  for (let index = 0; index < 4; index += 1) {
    try { normals.push(greatCircleNormal(rays[index * 2], rays[index * 2 + 1])); }
    catch { normals.push({ x: 0, y: 0, z: 0 }); }
  }
  let vpX: Vec3 | null = null;
  let vpZ: Vec3 | null = null;
  let horizon: Vec3 | null = null;
  try { vpX = vanishingDirection(normals[0], normals[1]); } catch { /* incomplete */ }
  try { vpZ = vanishingDirection(normals[2], normals[3]); } catch { /* incomplete */ }
  try { if (vpX && vpZ) horizon = academyNormalize(academyCross(vpX, vpZ)); } catch { /* degenerate */ }
  return { normals, vpX, vpZ, horizon };
}

function renderMarker(ray: Vec3, color: string, label: string): void {
  for (const direction of [ray, { x: -ray.x, y: -ray.y, z: -ray.z }]) {
    const point = sourceToCanvas(canonicalRayToErpPixel(direction, { width: sourceCanvas.width, height: sourceCanvas.height }));
    context.beginPath(); context.arc(point.x, point.y, 8, 0, Math.PI * 2);
    context.fillStyle = "rgba(8,11,10,.9)"; context.fill();
    context.strokeStyle = color; context.lineWidth = 2.5; context.stroke();
    context.fillStyle = color; context.font = "600 10px ui-monospace, monospace";
    context.fillText(label, point.x + 13, point.y + 4);
  }
}

function render(): void {
  if (!sourceCanvas.width) return;
  context.clearRect(0, 0, canvas.width, canvas.height);
  context.drawImage(sourceCanvas, 0, 0, canvas.width, canvas.height);
  context.fillStyle = "rgba(3,6,5,.12)";
  context.fillRect(0, 0, canvas.width, canvas.height);
  drawLatLonGrid();
  const geometry = currentGeometry();
  if (lesson === "lines" || lesson === "vanishing" || lesson === "calibration") {
    geometry.normals.forEach((normal, index) => {
      if (Math.hypot(normal.x, normal.y, normal.z) < 0.5) return;
      const samples = sampleGreatCircle(normal).map((ray) => canonicalRayToErpPixel(ray, { width: sourceCanvas.width, height: sourceCanvas.height }));
      drawPolyline(samples, index < 2 ? "rgba(255,202,92,.45)" : "rgba(103,214,197,.45)", 1.2, true);
    });
  }
  if ((lesson === "vanishing" || lesson === "calibration") && geometry.horizon) {
    const samples = sampleGreatCircle(geometry.horizon).map((ray) => canonicalRayToErpPixel(ray, { width: sourceCanvas.width, height: sourceCanvas.height }));
    drawPolyline(samples, "rgba(240,242,235,.7)", 1.25, true);
  }
  if (lesson === "rays") drawSelectedPoint();
  else if (lesson !== "projective") {
    segments.forEach((segment, index) => {
      const start = sourceToCanvas(segment.start);
      const end = sourceToCanvas(segment.end);
      const selected = index === selectedSegment;
      context.save();
      context.strokeStyle = groupColors[segment.group]; context.lineWidth = selected ? 4 : 2.5;
      context.shadowColor = selected ? groupColors[segment.group] : "transparent"; context.shadowBlur = selected ? 12 : 0;
      context.beginPath(); context.moveTo(start.x, start.y); context.lineTo(end.x, end.y); context.stroke();
      for (const point of [start, end]) {
        context.beginPath(); context.arc(point.x, point.y, selected ? 5 : 3.5, 0, Math.PI * 2); context.fillStyle = groupColors[segment.group]; context.fill();
      }
      context.fillStyle = groupColors[segment.group]; context.font = "650 10px ui-monospace, monospace";
      context.fillText(segment.id, start.x + 8, start.y - 9);
      context.restore();
    });
  }
  if (geometry.vpX && (lesson === "vanishing" || lesson === "calibration")) renderMarker(geometry.vpX, groupColors.x, "vₓ");
  if (geometry.vpZ && (lesson === "vanishing" || lesson === "calibration")) renderMarker(geometry.vpZ, groupColors.z, "vᵤ");
  updateEvidence(geometry);
  renderBearingSphere(geometry);
}

function formatVector(vector: Vec3 | null, digits = 3): string {
  if (!vector) return "—";
  return `[${vector.x.toFixed(digits)}, ${vector.y.toFixed(digits)}, ${vector.z.toFixed(digits)}]`;
}

function updateEvidence(geometry = currentGeometry()): void {
  const label1 = required<HTMLElement>("#metric1Label");
  const label2 = required<HTMLElement>("#metric2Label");
  const label3 = required<HTMLElement>("#metric3Label");
  const value1 = required<HTMLElement>("#metric1Value");
  const value2 = required<HTMLElement>("#metric2Value");
  const value3 = required<HTMLElement>("#metric3Value");
  const sourceLabel = required<HTMLElement>("#sourceLabel");
  const sourceValue = required<HTMLElement>("#raySource");
  if (lesson === "projective") {
    label1.textContent = "PONTO FINITO"; value1.textContent = "w ≠ 0";
    label2.textContent = "PONTO IDEAL"; value2.textContent = "w = 0";
    label3.textContent = "EQUIVALÊNCIA"; value3.textContent = "x̃ ~ αx̃";
    sourceLabel.textContent = "PONTE PARA STITCHING"; sourceValue.textContent = "x′ ~ Hx";
  } else if (lesson === "rays") {
    const angles = canonicalErpPixelToLonLat(selectedPoint, { width: sourceCanvas.width, height: sourceCanvas.height });
    label1.textContent = "LONGITUDE λ"; value1.textContent = formatDegrees(angles.longitudeDeg, "E", "W");
    label2.textContent = "LATITUDE φ"; value2.textContent = formatDegrees(angles.latitudeDeg, "N", "S");
    label3.textContent = "BEARING XYZ"; value3.textContent = formatVector(pointRay, 2);
    sourceLabel.textContent = "FONTE DO RAIO";
    sourceValue.textContent = backendReady ? "PanorAI geometry-v1" : "prévia local";
  } else {
    const angle = geometry.vpX && geometry.vpZ ? Math.min(angleDeg(geometry.vpX, geometry.vpZ), 180 - angleDeg(geometry.vpX, geometry.vpZ)) : null;
    label1.textContent = "DIREÇÃO X"; value1.textContent = formatVector(geometry.vpX, 2);
    label2.textContent = "DIREÇÃO Z"; value2.textContent = formatVector(geometry.vpZ, 2);
    label3.textContent = "ÂNGULO ENTRE ELAS"; value3.textContent = angle === null ? "—" : `${angle.toFixed(1)}°`;
    sourceLabel.textContent = "FONTE DOS RAIOS";
    sourceValue.textContent = backendReady ? "PanorAI geometry-v1" : "prévia local";
  }
  renderConcept(geometry);
}

function renderConcept(geometry = currentGeometry()): void {
  const index = lessonOrder.indexOf(lesson);
  required<HTMLElement>("#conceptStep").textContent = `CONCEITO ${String(index).padStart(2, "0")}`;
  const intro = required<HTMLElement>("#conceptIntro");
  const formula = required<HTMLElement>("#formulaBlock");
  const values = required<HTMLElement>("#liveValues");
  const interpretation = required<HTMLElement>("#interpretation");
  const boundary = required<HTMLElement>("#libraryBoundary");

  if (lesson === "projective") {
    intro.innerHTML = `<span>ANTES DA CÂMERA</span><h2>Pontos, retas<br/>e o infinito.</h2><p>Começamos pela linguagem do plano projetivo. Ela descreve incidência e direções sem escolher ainda um modelo de câmera — e será reutilizada em pontos de fuga e stitching.</p>`;
    formula.innerHTML = `<small>COORDENADAS HOMOGÊNEAS · P²</small><div class="equation vector"><b>x̃</b> = (x, y, w)<sup>T</sup></div><div class="equation vector"><b>x̃</b> ~ α<b>x̃</b>, &nbsp;α ≠ 0</div><div class="equation projective-pair"><span><b>w ≠ 0</b><small>ponto finito</small></span><span><b>w = 0</b><small>ponto ideal</small></span></div><div class="equation vector"><b>l</b><sup>T</sup><b>x̃</b> = 0</div>`;
    values.innerHTML = `<span>OPERAÇÕES FUNDAMENTAIS</span><dl><dt>ponto finito</dt><dd>[x, y, 1]</dd><dt>direção ideal</dt><dd>[dₓ, dᵧ, 0]</dd><dt>reta por 2 pontos</dt><dd>l = x̃₁ × x̃₂</dd><dt>interseção</dt><dd>x̃ = l₁ × l₂</dd></dl>`;
    interpretation.innerHTML = `<b>Leia geometricamente</b><p>Escalar não muda o objeto: <b>x̃</b> e α<b>x̃</b> são o mesmo ponto. Por isso uma direção esférica e sua antípoda, <b>d</b> ~ −<b>d</b>, representam o mesmo eixo projetivo. Retas paralelas encontram-se em um ponto ideal.</p>`;
  } else if (lesson === "rays") {
    const angles = canonicalErpPixelToLonLat(selectedPoint, { width: sourceCanvas.width, height: sourceCanvas.height });
    intro.innerHTML = `<span>DO RASTER À ESFERA</span><h2>Clique em um ponto.<br/>Observe o bearing.</h2><p>Neste primeiro conceito a entrada é somente um centro de pixel. A grade localiza longitude e latitude; o globo mostra a mesma seleção como direção 3D.</p>`;
    formula.innerHTML = `<small>FUNÇÃO ERP → S² · GEOMETRY-V1</small><div class="equation"><i>λ(x)</i> = 2π <em>(x + 0.5) / W</em> − π</div><div class="equation"><i>φ(y)</i> = π/2 − π <em>(y + 0.5) / H</em></div><div class="equation vector bearing-function" id="bearingFunctionTrigger" tabindex="0" aria-describedby="bearingGeometryTooltip"><b>b(λ, φ)</b> = <span>[sin λ cos φ,<br/>sin φ, cos λ cos φ]</span><em class="geometry-cue">ESFERA + TRIÂNGULOS ↗</em></div>`;
    required<HTMLElement>("#diagramAngles").textContent = `λ ${angles.longitudeDeg.toFixed(1)}° · φ ${angles.latitudeDeg.toFixed(1)}°`;
    required<HTMLElement>("#diagramVector").textContent = `b = ${formatVector(pointRay, 2)}`;
    values.innerHTML = `<span>SUBSTITUIÇÃO DA SELEÇÃO</span><dl><dt>pixel (x, y)</dt><dd>(${selectedPoint.x.toFixed(1)}, ${selectedPoint.y.toFixed(1)})</dd><dt>(λ, φ)</dt><dd>(${angles.longitudeDeg.toFixed(2)}°, ${angles.latitudeDeg.toFixed(2)}°)</dd><dt>bearing [X,Y,Z]</dt><dd>${formatVector(pointRay)}</dd><dt>shape (H,W)</dt><dd>(${sourceCanvas.height}, ${sourceCanvas.width})</dd></dl>`;
    interpretation.innerHTML = `<b>Leia a função</b><p><b>X</b> cresce à direita, <b>Y</b> aponta para cima e <b>Z</b> é a frente. A longitude percorre a imagem horizontalmente; a latitude vai de +90° no norte a −90° no sul.</p>`;
  } else if (lesson === "lines") {
    intro.innerHTML = `<span>LINHA VISÍVEL</span><h2>Na esfera, uma reta da cena vive em um grande círculo.</h2><p>Dois raios sobre a mesma linha definem um plano que passa pelo centro da câmera. A interseção desse plano com a esfera é o seu great circle.</p>`;
    formula.innerHTML = `<small>PLANO DE INTERPRETAÇÃO</small><div class="equation vector"><b>nᵢ</b> = normalize(<b>dᵢ₁</b> × <b>dᵢ₂</b>)</div><div class="equation"><b>nᵢ</b><sup>T</sup><b>d</b> = 0</div><p>O globo destaca ${segments[selectedSegment].id}: arco contínuo na frente, tracejado atrás. A mesma circunferência aparece como curva tracejada quando aberta na ERP.</p>`;
    values.innerHTML = `<span>NORMAIS DOS PLANOS</span><dl>${geometry.normals.map((normal, idx) => `<dt>${segments[idx]?.id ?? "—"}</dt><dd>${formatVector(normal)}</dd>`).join("")}</dl>`;
    interpretation.innerHTML = `<b>Por que a linha curva?</b><p>A ERP abre a esfera usando longitude no eixo horizontal e latitude no vertical. Fora do equador e dos meridianos, a latitude de um great circle varia de forma não linear com a longitude — por isso o círculo vira uma curva na imagem.</p>`;
  } else if (lesson === "vanishing") {
    intro.innerHTML = `<span>DIREÇÃO DE FUGA</span><h2>Paralelas compartilham uma direção, não uma posição.</h2><p>Cada par de planos da mesma família se intersecta numa direção 3D. Na esfera o resultado é antipodal: <i>v</i> e −<i>v</i> representam o mesmo eixo.</p>`;
    formula.innerHTML = `<small>INTERSEÇÃO DE GREAT CIRCLES</small><div class="equation vector"><b>vₓ</b> ∝ <b>n₁</b> × <b>n₂</b></div><div class="equation vector"><b>vᵤ</b> ∝ <b>n₃</b> × <b>n₄</b></div><div class="equation vector"><b>ℓ</b><sub>∞</sub> ∝ <b>vₓ</b> × <b>vᵤ</b></div>`;
    values.innerHTML = `<span>RESULTADO VIVO</span><dl><dt>vₓ</dt><dd>${formatVector(geometry.vpX)}</dd><dt>vᵤ</dt><dd>${formatVector(geometry.vpZ)}</dd><dt>n horizonte</dt><dd>${formatVector(geometry.horizon)}</dd></dl>`;
    interpretation.innerHTML = `<b>O que observar</b><p>Mova qualquer segmento. A direção muda porque a evidência angular mudou; a linha branca reúne as duas direções no plano dominante.</p>`;
  } else {
    const orthogonality = geometry.vpX && geometry.vpZ ? Math.abs(academyDot(geometry.vpX, geometry.vpZ)) : null;
    intro.innerHTML = `<span>CALIBRAÇÃO DE UMA VISTA</span><h2>Ortogonalidade impõe restrições sobre a câmera.</h2><p>No modelo pinhole, direções ortogonais satisfazem uma relação com a imagem da cônica absoluta. Na ERP, os raios já estão calibrados pelo modelo esférico; estimar uma câmera desconhecida exige um contrato próprio.</p>`;
    formula.innerHTML = `<small>FORMULAÇÃO CLÁSSICA · PINHOLE</small><div class="equation"><b>v₁</b><sup>T</sup> ω <b>v₂</b> = 0</div><div class="equation">ω = K<sup>−T</sup>K<sup>−1</sup></div><div class="equation"><i>θ</i> = acos(<b>d₁</b><sup>T</sup><b>d₂</b>)</div>`;
    values.innerHTML = `<span>DIAGNÓSTICO DIDÁTICO</span><dl><dt>|vₓᵀvᵤ|</dt><dd>${orthogonality === null ? "—" : orthogonality.toFixed(4)}</dd><dt>ângulo agudo</dt><dd>${geometry.vpX && geometry.vpZ ? `${Math.min(angleDeg(geometry.vpX, geometry.vpZ), 180 - angleDeg(geometry.vpX, geometry.vpZ)).toFixed(2)}°` : "—"}</dd><dt>estimativa K</dt><dd>indisponível</dd></dl>`;
    interpretation.innerHTML = `<b>Limite honesto</b><p>Este valor ajuda a testar a hipótese de ortogonalidade, mas não é uma calibração robusta nem uma estimativa de K.</p>`;
  }

  if (lesson === "projective") {
    boundary.innerHTML = `<span class="boundary-state ready">BASE PARA STITCHING</span><strong>Homografia: x′ ~ Hx</strong><p>Uma homografia preserva incidências no plano projetivo. Mais adiante ela alinhará vistas em casos como cena plana ou rotação pura, sem exigir que este passo introduza a matriz de câmera.</p>`;
  } else if (lesson === "calibration") {
    boundary.innerHTML = `<span class="boundary-state">REQUER PANORAI MAIN</span><strong>Próxima entrega da biblioteca</strong><p>Um agent atualizará o PanorAI com ajuste robusto de great circles, estimador de vanishing directions e calibração de uma vista. O Workbench não cria um backend paralelo.</p>`;
  } else if (!backendReady) {
    boundary.innerHTML = `<span class="boundary-state">PRÉVIA DIDÁTICA LOCAL</span><strong>Backend PanorAI indisponível</strong><p>As fórmulas continuam exploráveis no navegador, mas a evidência só recebe o selo PanorAI Main quando o endpoint de geometria confirma os raios.</p>`;
  } else {
    boundary.innerHTML = `<span class="boundary-state ready">PANORAI MAIN · STABLE</span><strong>Raio calculado pela API pública</strong><p><code>panorai.geometry.erp_pixels_to_rays</code></p>`;
  }
}

function renderSelectionControls(): void {
  const content = required<HTMLElement>("#selectionContent");
  const title = required<HTMLElement>("#selectionTitle");
  const help = required<HTMLElement>("#selectionHelp");
  const clear = required<HTMLButtonElement>("#clearSelection");
  if (lesson === "projective") {
    title.textContent = "PASSO 00 · LINGUAGEM";
    help.textContent = "Nenhuma marcação: primeiro reconheça equivalência, incidência e infinito.";
    clear.textContent = "Modo leitura";
    clear.disabled = true;
    content.className = "selection-content projective-summary";
    content.innerHTML = `<div><span>EQUIVALÊNCIA</span><b>[x,y,w] ~ α[x,y,w]</b></div><div><span>FINITO</span><b>w ≠ 0</b></div><div><span>IDEAL</span><b>w = 0</b></div><div><span>STITCHING</span><b>x′ ~ Hx</b></div>`;
    return;
  }
  clear.disabled = false;
  if (lesson === "rays") {
    const angles = canonicalErpPixelToLonLat(selectedPoint, { width: sourceCanvas.width, height: sourceCanvas.height });
    title.textContent = "ENTRADA · UM PONTO";
    help.textContent = "Clique para converter um centro de pixel em direção 3D.";
    clear.textContent = "Centralizar";
    content.className = "selection-content point-selection";
    content.innerHTML = `<div class="point-swatch"><i></i><span><b>Ponto p</b><small>clique simples</small></span></div><div><span>PIXEL</span><b>(${selectedPoint.x.toFixed(1)}, ${selectedPoint.y.toFixed(1)})</b></div><div><span>LON / LAT</span><b>${formatDegrees(angles.longitudeDeg, "E", "W")} · ${formatDegrees(angles.latitudeDeg, "N", "S")}</b></div><div><span>XYZ</span><b>${formatVector(pointRay, 2)}</b></div>`;
    return;
  }
  title.textContent = "ENTRADAS · QUATRO LINHAS";
  help.textContent = "Escolha um segmento e arraste sobre a linha observada.";
  clear.textContent = "Limpar atual";
  content.className = "selection-content segment-slots";
  content.innerHTML = segments.map((segment, index) => `
      <button class="segment-slot ${index === selectedSegment ? "active" : ""}" data-segment="${index}" style="--slot-color:${groupColors[segment.group]}">
        <i></i><span><b>${segment.id}</b><small>${segment.group === "x" ? "família X" : "família Z"}</small></span>
      </button>
    `).join("");
  document.querySelectorAll<HTMLButtonElement>("[data-segment]").forEach((button) => button.addEventListener("click", () => {
    selectedSegment = Number(button.dataset.segment);
    renderSelectionControls(); updateInstruction(); render();
  }));
}

function updateInstruction(): void {
  const stage = required<HTMLElement>("#panoramaStage");
  const title = required<HTMLElement>("#experimentTitle");
  const legend = required<HTMLElement>(".vp-legend");
  if (lesson === "projective") {
    stage.dataset.mode = "reading";
    title.textContent = "A linguagem antes da projeção";
    legend.hidden = true;
    required<HTMLElement>("#instructionText").innerHTML = `<b>MODO LEITURA</b> · nenhuma seleção ainda; observe a grade e o par antipodal d ~ −d.`;
    return;
  }
  if (lesson === "rays") {
    stage.dataset.mode = "point";
    title.textContent = "Posicione um bearing na esfera";
    legend.hidden = true;
    required<HTMLElement>("#instructionText").innerHTML = `<b>MODO PONTO</b> · clique uma vez na ERP para mover o bearing.`;
    return;
  }
  stage.dataset.mode = "line";
  title.textContent = "Duas famílias de linhas paralelas";
  legend.hidden = false;
  const segment = segments[selectedSegment];
  required<HTMLElement>("#instructionText").innerHTML = `<b>MODO LINHA</b> · arraste <b>${segment.id}</b> sobre uma linha da família ${segment.group.toUpperCase()}.`;
}

function fallbackRays(): Vec3[] {
  return segments.flatMap((segment) => [segment.start, segment.end]).map((point) => canonicalErpPixelToRay(point, { width: sourceCanvas.width, height: sourceCanvas.height }));
}

async function syncRays(): Promise<void> {
  pointRay = canonicalErpPixelToRay(selectedPoint, { width: sourceCanvas.width, height: sourceCanvas.height });
  rays = fallbackRays();
  required<HTMLElement>("#raySource").textContent = "prévia local";
  renderSelectionControls(); updateInstruction(); render();
  try {
    const selectedPoints = [selectedPoint, ...segments.flatMap((segment) => [segment.start, segment.end])];
    const response = await fetch("/api/v1/academy/single-view/rays", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        image_shape_hw: [sourceCanvas.height, sourceCanvas.width],
        points_xy: selectedPoints.map((point) => [point.x, point.y]),
      }),
    });
    if (!response.ok) throw new Error(`backend ${response.status}`);
    const payload = await response.json() as { rays_xyz: number[][] };
    const lifted = payload.rays_xyz.map(([x, y, z]) => ({ x, y, z }));
    pointRay = lifted[0];
    rays = lifted.slice(1);
    backendReady = true;
    required<HTMLElement>("#raySource").textContent = "PanorAI geometry-v1";
    renderSelectionControls();
    render();
  } catch {
    backendReady = false;
    required<HTMLElement>("#raySource").textContent = "prévia local · backend offline";
  }
}

async function loadStatus(): Promise<void> {
  const runtime = required<HTMLElement>("#runtimeState");
  try {
    const response = await fetch("/api/v1/academy/single-view/status");
    if (!response.ok) throw new Error(`backend ${response.status}`);
    status = await response.json() as AcademyStatus;
    runtime.classList.add("ready");
    runtime.querySelector("span")!.textContent = `PanorAI Main · ${status.panorai_version}`;
  } catch {
    runtime.classList.add("offline");
    runtime.querySelector("span")!.textContent = "PanorAI Main · backend offline";
  }
}

function selectLesson(next: Lesson): void {
  lesson = next;
  document.querySelectorAll<HTMLButtonElement>(".lesson").forEach((button) => button.classList.toggle("active", button.dataset.lesson === lesson));
  renderSelectionControls();
  updateInstruction();
  render();
}

canvas.addEventListener("pointerdown", (event) => {
  if (lesson === "projective") return;
  if (lesson === "rays") {
    selectedPoint = eventToSource(event);
    syncRays();
    return;
  }
  dragStart = eventToSource(event);
  canvas.setPointerCapture(event.pointerId);
  segments[selectedSegment].start = dragStart;
  segments[selectedSegment].end = dragStart;
  rays = fallbackRays(); render();
});
canvas.addEventListener("pointermove", (event) => {
  const hover = eventToSource(event);
  const angles = canonicalErpPixelToLonLat(hover, { width: sourceCanvas.width, height: sourceCanvas.height });
  required<HTMLElement>("#coordinateHover").textContent = `lon ${formatDegrees(angles.longitudeDeg, "E", "W")} · lat ${formatDegrees(angles.latitudeDeg, "N", "S")}`;
  if (!dragStart) return;
  segments[selectedSegment].end = eventToSource(event);
  rays = fallbackRays(); render();
});
canvas.addEventListener("pointerup", (event) => {
  if (lesson === "projective" || lesson === "rays") return;
  if (!dragStart) return;
  segments[selectedSegment].end = eventToSource(event);
  dragStart = null;
  selectedSegment = (selectedSegment + 1) % segments.length;
  syncRays();
});
canvas.addEventListener("pointerleave", () => {
  required<HTMLElement>("#coordinateHover").textContent = "lon — · lat —";
});

document.querySelectorAll<HTMLButtonElement>(".lesson").forEach((button) => button.addEventListener("click", () => selectLesson(button.dataset.lesson as Lesson)));
required<HTMLButtonElement>("#nextConcept").addEventListener("click", () => selectLesson(lessonOrder[(lessonOrder.indexOf(lesson) + 1) % lessonOrder.length]));
required<HTMLButtonElement>("#openImage").addEventListener("click", () => imageInput.click());
required<HTMLButtonElement>("#loadDemo").addEventListener("click", makeDemoPanorama);
required<HTMLButtonElement>("#toggleGrid").addEventListener("click", () => {
  gridEnabled = !gridEnabled;
  const button = required<HTMLButtonElement>("#toggleGrid");
  button.classList.toggle("active", gridEnabled);
  button.setAttribute("aria-pressed", String(gridEnabled));
  render();
});
required<HTMLButtonElement>("#clearSelection").addEventListener("click", () => {
  if (lesson === "projective") return;
  if (lesson === "rays") {
    selectedPoint = { x: sourceCanvas.width / 2 - 0.5, y: sourceCanvas.height / 2 - 0.5 };
    syncRays();
    return;
  }
  const point = segments[selectedSegment].start;
  segments[selectedSegment].end = { ...point };
  syncRays();
});
imageInput.addEventListener("change", async () => {
  const file = imageInput.files?.[0];
  if (!file) return;
  const url = URL.createObjectURL(file);
  try {
    const image = new Image(); image.src = url; await image.decode();
    const width = Math.min(4096, image.naturalWidth);
    const height = Math.max(1, Math.round(width * image.naturalHeight / image.naturalWidth));
    sourceCanvas.width = width; sourceCanvas.height = height;
    sourceContext.drawImage(image, 0, 0, width, height);
    const sx = width / 1400; const sy = height / 700;
    segments = demoSegments().map((segment) => ({ ...segment, start: { x: segment.start.x * sx, y: segment.start.y * sy }, end: { x: segment.end.x * sx, y: segment.end.y * sy } }));
    selectedPoint = { x: width / 2 - 0.5, y: height / 2 - 0.5 };
    required<HTMLElement>("#imageName").textContent = file.name;
    syncRays();
  } finally { URL.revokeObjectURL(url); }
});

new ResizeObserver(syncCanvasSize).observe(required<HTMLElement>("#panoramaStage"));
makeDemoPanorama();
loadStatus();

void backendReady;
void status;
