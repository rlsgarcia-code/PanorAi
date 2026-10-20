import "./styles.css";

import { clamp, regionFromViewRectangle, wrapDegrees } from "./geometry.ts";
import type {
  AnnotationDocument,
  PanoramaRecord,
  Point2,
  ReviewState,
  SphericalAnnotation,
  ViewState,
} from "./model.ts";
import { drawErpAnnotations, drawViewAnnotations, renderGnomonic } from "./renderer.ts";
import type { PanoramaPixels } from "./renderer.ts";
import { createDocument, downloadDocument, parseDocument } from "./storage.ts";

if (new URLSearchParams(location.search).get("embedded") === "1") document.body.classList.add("embedded");

const app = document.querySelector<HTMLDivElement>("#app");
if (!app) throw new Error("Missing application root.");

app.innerHTML = `
  <div class="app-shell">
    <header class="topbar">
      <div class="brand-block">
        <div class="mark" aria-hidden="true"><span></span><span></span></div>
        <div><strong>PanorAi Workbench</strong><small>Segmentação e anotação · protótipo local</small></div>
      </div>
      <nav class="workspace-nav" aria-label="Ambiente">
        <a href="/">Laboratório</a>
        <a href="/geometry.html">Detecção</a>
        <a class="active" href="/segmentation.html">Segmentação</a>
      </nav>
      <div class="file-meta">
        <span id="fileName">procedural-grid.erp</span>
        <span id="fileSize">2048 × 1024</span>
      </div>
      <div class="top-actions">
        <input id="imageInput" type="file" accept="image/png,image/jpeg,image/webp" hidden />
        <input id="jsonInput" type="file" accept="application/json,.json" hidden />
        <button id="openImage" class="quiet">Abrir ERP</button>
        <button id="importJson" class="quiet">Importar JSON</button>
        <button id="exportJson" class="primary">Exportar</button>
      </div>
    </header>

    <main class="workspace">
      <section class="view-column" aria-label="Área de anotação">
        <div class="view-toolbar">
          <div class="mode-switch" role="group" aria-label="Modo de interação">
            <button id="navigateMode" class="active" aria-pressed="true"><kbd>V</kbd> Navegar</button>
            <button id="annotateMode" aria-pressed="false"><kbd>A</kbd> Anotar região</button>
          </div>
          <div class="view-readout" id="viewReadout">0.0° / 0.0° · FOV 72°</div>
        </div>

        <div class="perspective-stage" id="perspectiveStage">
          <canvas id="viewCanvas" aria-label="Projeção gnomônica interativa"></canvas>
          <div class="reticle" aria-hidden="true"><i></i><i></i></div>
          <div class="stage-help" id="stageHelp">Arraste para orientar · roda para alterar o FOV</div>
          <div class="selection-label" id="selectionLabel" hidden></div>
        </div>

        <div class="erp-section">
          <div class="erp-heading">
            <span>Contexto equiretangular</span>
            <small>Clique para reposicionar a vista</small>
          </div>
          <canvas id="erpCanvas" aria-label="Panorama equiretangular de contexto"></canvas>
        </div>
      </section>

      <aside class="inspector" aria-label="Inspetor de anotações">
        <div class="inspector-head">
          <div><small>Documento local</small><h1>Regiões esféricas</h1></div>
          <span class="count" id="annotationCount">0</span>
        </div>

        <section class="editor" id="editor">
          <label>Rótulo
            <input id="labelInput" type="text" value="unlabeled" autocomplete="off" />
          </label>
          <label>Estado
            <select id="stateInput">
              <option value="draft">Rascunho</option>
              <option value="needs-review">Revisar</option>
              <option value="accepted">Aceita</option>
              <option value="rejected">Rejeitada</option>
            </select>
          </label>
          <div class="region-readout" id="regionReadout">
            Selecione <b>Anotar região</b> e arraste sobre o objeto.
          </div>
          <div class="editor-actions">
            <button id="focusSelected" class="quiet" disabled>Centralizar</button>
            <button id="deleteSelected" class="danger" disabled>Excluir</button>
          </div>
        </section>

        <div class="list-heading"><span>Anotações</span><small id="filterSummary">todas</small></div>
        <div class="annotation-list" id="annotationList"></div>

        <footer class="inspector-foot">
          <span class="status-dot"></span>
          <span id="statusText">Dados permanecem neste navegador até exportar.</span>
        </footer>
      </aside>
    </main>
  </div>
`;

function required<T extends Element>(selector: string): T {
  const element = document.querySelector<T>(selector);
  if (!element) throw new Error(`Missing element ${selector}.`);
  return element;
}

const viewCanvas = required<HTMLCanvasElement>("#viewCanvas");
const erpCanvas = required<HTMLCanvasElement>("#erpCanvas");
const perspectiveStage = required<HTMLDivElement>("#perspectiveStage");
const imageInput = required<HTMLInputElement>("#imageInput");
const jsonInput = required<HTMLInputElement>("#jsonInput");
const labelInput = required<HTMLInputElement>("#labelInput");
const stateInput = required<HTMLSelectElement>("#stateInput");
const annotationList = required<HTMLDivElement>("#annotationList");
const annotationCount = required<HTMLSpanElement>("#annotationCount");
const regionReadout = required<HTMLDivElement>("#regionReadout");
const viewReadout = required<HTMLDivElement>("#viewReadout");
const statusText = required<HTMLSpanElement>("#statusText");
const fileName = required<HTMLSpanElement>("#fileName");
const fileSize = required<HTMLSpanElement>("#fileSize");
const selectionLabel = required<HTMLDivElement>("#selectionLabel");
const navigateModeButton = required<HTMLButtonElement>("#navigateMode");
const annotateModeButton = required<HTMLButtonElement>("#annotateMode");
const focusSelectedButton = required<HTMLButtonElement>("#focusSelected");
const deleteSelectedButton = required<HTMLButtonElement>("#deleteSelected");

let panoramaCanvas = document.createElement("canvas");
let panoramaPixels: PanoramaPixels;
let panorama: PanoramaRecord = {
  id: "procedural-grid",
  fileName: "procedural-grid.erp",
  width: 2048,
  height: 1024,
};
let annotations: SphericalAnnotation[] = [];
let selectedId: string | null = null;
let mode: "navigate" | "annotate" = "navigate";
let view: ViewState = { yawDeg: 0, pitchDeg: 0, horizontalFovDeg: 72 };
let interaction:
  | { kind: "pan"; start: Point2; startView: ViewState }
  | { kind: "box"; start: Point2; end: Point2 }
  | null = null;
let renderRequested = false;

function createProceduralPanorama(width = 2048, height = 1024): PanoramaPixels {
  panoramaCanvas.width = width;
  panoramaCanvas.height = height;
  const context = panoramaCanvas.getContext("2d", { willReadFrequently: true });
  if (!context) throw new Error("Canvas 2D is unavailable.");
  const gradient = context.createLinearGradient(0, 0, 0, height);
  gradient.addColorStop(0, "#20261f");
  gradient.addColorStop(0.52, "#34382e");
  gradient.addColorStop(1, "#171916");
  context.fillStyle = gradient;
  context.fillRect(0, 0, width, height);
  context.strokeStyle = "rgba(222, 230, 210, 0.13)";
  context.lineWidth = 2;
  for (let lon = -180; lon <= 180; lon += 15) {
    const x = ((lon + 180) / 360) * width;
    context.beginPath(); context.moveTo(x, 0); context.lineTo(x, height); context.stroke();
  }
  for (let lat = -75; lat <= 75; lat += 15) {
    const y = ((90 - lat) / 180) * height;
    context.beginPath(); context.moveTo(0, y); context.lineTo(width, y); context.stroke();
  }
  context.fillStyle = "rgba(255, 200, 87, 0.16)";
  context.fillRect(width * 0.44, height * 0.32, width * 0.12, height * 0.36);
  context.strokeStyle = "rgba(255, 200, 87, 0.8)";
  context.lineWidth = 5;
  context.strokeRect(width * 0.44, height * 0.32, width * 0.12, height * 0.36);
  context.fillStyle = "rgba(255,255,255,0.68)";
  context.font = "500 28px system-ui";
  context.fillText("Abra um panorama ERP para começar", 48, height - 54);
  const image = context.getImageData(0, 0, width, height);
  return { width, height, data: image.data };
}

panoramaPixels = createProceduralPanorama();

function canvasPoint(event: PointerEvent, canvas: HTMLCanvasElement): Point2 {
  const rect = canvas.getBoundingClientRect();
  return {
    x: ((event.clientX - rect.left) / rect.width) * canvas.width,
    y: ((event.clientY - rect.top) / rect.height) * canvas.height,
  };
}

function selectedAnnotation(): SphericalAnnotation | undefined {
  return annotations.find((annotation) => annotation.id === selectedId);
}

function setStatus(message: string): void {
  statusText.textContent = message;
}

function setMode(nextMode: "navigate" | "annotate"): void {
  mode = nextMode;
  interaction = null;
  navigateModeButton.classList.toggle("active", mode === "navigate");
  annotateModeButton.classList.toggle("active", mode === "annotate");
  navigateModeButton.setAttribute("aria-pressed", String(mode === "navigate"));
  annotateModeButton.setAttribute("aria-pressed", String(mode === "annotate"));
  perspectiveStage.dataset.mode = mode;
  required<HTMLDivElement>("#stageHelp").textContent = mode === "navigate"
    ? "Arraste para orientar · roda para alterar o FOV"
    : "Arraste um retângulo em torno do objeto · Esc cancela";
  scheduleRender();
}

function resizeCanvas(canvas: HTMLCanvasElement, maxWidth: number): void {
  const rect = canvas.getBoundingClientRect();
  const scale = Math.min(1, maxWidth / Math.max(rect.width, 1));
  const width = Math.max(1, Math.round(rect.width * scale));
  const height = Math.max(1, Math.round(rect.height * scale));
  if (canvas.width !== width || canvas.height !== height) {
    canvas.width = width;
    canvas.height = height;
  }
}

function renderView(): void {
  resizeCanvas(viewCanvas, 760);
  const context = viewCanvas.getContext("2d");
  if (!context) return;
  renderGnomonic(context, panoramaPixels, view, viewCanvas.width, viewCanvas.height);
  drawViewAnnotations(context, annotations, selectedId, view, viewCanvas);
  if (interaction?.kind === "box") {
    const { start, end } = interaction;
    context.save();
    context.fillStyle = "rgba(255, 200, 87, 0.12)";
    context.strokeStyle = "#ffc857";
    context.lineWidth = 2;
    context.setLineDash([8, 5]);
    context.fillRect(start.x, start.y, end.x - start.x, end.y - start.y);
    context.strokeRect(start.x, start.y, end.x - start.x, end.y - start.y);
    context.restore();
    const region = regionFromViewRectangle(start, end, viewCanvas, view);
    selectionLabel.hidden = false;
    selectionLabel.textContent = `${region.horizontalFovDeg.toFixed(1)}° × ${region.verticalFovDeg.toFixed(1)}°`;
  } else {
    selectionLabel.hidden = true;
  }
  viewReadout.textContent = `${view.yawDeg.toFixed(1)}° / ${view.pitchDeg.toFixed(1)}° · FOV ${view.horizontalFovDeg.toFixed(0)}°`;
}

function renderErp(): void {
  resizeCanvas(erpCanvas, 1100);
  const context = erpCanvas.getContext("2d");
  if (!context) return;
  context.drawImage(panoramaCanvas, 0, 0, erpCanvas.width, erpCanvas.height);
  context.fillStyle = "rgba(9, 10, 8, 0.12)";
  context.fillRect(0, 0, erpCanvas.width, erpCanvas.height);
  drawErpAnnotations(context, annotations, selectedId, erpCanvas);
  const x = ((wrapDegrees(view.yawDeg) + 180) / 360) * erpCanvas.width;
  const y = ((90 - view.pitchDeg) / 180) * erpCanvas.height;
  context.save();
  context.strokeStyle = "#ffc857";
  context.lineWidth = 1.5;
  context.beginPath(); context.moveTo(x, 0); context.lineTo(x, erpCanvas.height); context.stroke();
  context.beginPath(); context.arc(x, y, 5, 0, Math.PI * 2); context.fillStyle = "#ffc857"; context.fill();
  context.restore();
}

function renderInspector(): void {
  annotationCount.textContent = String(annotations.length);
  annotationList.replaceChildren();
  if (annotations.length === 0) {
    const empty = document.createElement("div");
    empty.className = "empty-list";
    empty.innerHTML = "<strong>Nenhuma região</strong><span>Crie a primeira região no viewport.</span>";
    annotationList.append(empty);
  }
  annotations.forEach((annotation, index) => {
    const button = document.createElement("button");
    button.className = "annotation-row";
    button.classList.toggle("selected", annotation.id === selectedId);
    button.type = "button";
    const number = document.createElement("span");
    number.className = "annotation-index";
    number.textContent = String(index + 1).padStart(2, "0");
    const copy = document.createElement("span");
    copy.className = "annotation-copy";
    const title = document.createElement("strong");
    title.textContent = annotation.label || "unlabeled";
    const detail = document.createElement("small");
    detail.textContent = `${annotation.region.horizontalFovDeg.toFixed(1)}° × ${annotation.region.verticalFovDeg.toFixed(1)}°`;
    const state = document.createElement("span");
    state.className = `state state-${annotation.state}`;
    state.textContent = annotation.state;
    copy.append(title, detail);
    button.append(number, copy, state);
    button.addEventListener("click", () => {
      selectedId = annotation.id;
      syncEditor();
      scheduleRender();
    });
    annotationList.append(button);
  });
  syncEditor();
}

function syncEditor(): void {
  const selected = selectedAnnotation();
  focusSelectedButton.disabled = !selected;
  deleteSelectedButton.disabled = !selected;
  if (!selected) {
    regionReadout.innerHTML = "Selecione <b>Anotar região</b> e arraste sobre o objeto.";
    return;
  }
  labelInput.value = selected.label;
  stateInput.value = selected.state;
  regionReadout.innerHTML = `
    <span><small>Centro</small><b>${selected.region.centerYawDeg.toFixed(2)}° / ${selected.region.centerPitchDeg.toFixed(2)}°</b></span>
    <span><small>Campo angular</small><b>${selected.region.horizontalFovDeg.toFixed(2)}° × ${selected.region.verticalFovDeg.toFixed(2)}°</b></span>
  `;
}

function render(): void {
  renderRequested = false;
  renderView();
  renderErp();
  renderInspector();
}

function scheduleRender(): void {
  if (renderRequested) return;
  renderRequested = true;
  requestAnimationFrame(render);
}

function addAnnotation(region: SphericalAnnotation["region"]): void {
  const timestamp = new Date().toISOString();
  const annotation: SphericalAnnotation = {
    id: crypto.randomUUID(),
    label: labelInput.value.trim() || "unlabeled",
    state: stateInput.value as ReviewState,
    region,
    createdAt: timestamp,
    updatedAt: timestamp,
    source: "human",
  };
  annotations.push(annotation);
  selectedId = annotation.id;
  setStatus("Região criada. Exporte o JSON para persistir fora do navegador.");
  scheduleRender();
}

viewCanvas.addEventListener("pointerdown", (event) => {
  const point = canvasPoint(event, viewCanvas);
  viewCanvas.setPointerCapture(event.pointerId);
  interaction = mode === "navigate"
    ? { kind: "pan", start: point, startView: { ...view } }
    : { kind: "box", start: point, end: point };
  scheduleRender();
});

viewCanvas.addEventListener("pointermove", (event) => {
  if (!interaction) return;
  const point = canvasPoint(event, viewCanvas);
  if (interaction.kind === "pan") {
    const sensitivity = interaction.startView.horizontalFovDeg / viewCanvas.width;
    view.yawDeg = wrapDegrees(interaction.startView.yawDeg - (point.x - interaction.start.x) * sensitivity);
    view.pitchDeg = clamp(interaction.startView.pitchDeg + (point.y - interaction.start.y) * sensitivity, -89, 89);
  } else {
    interaction.end = point;
  }
  scheduleRender();
});

viewCanvas.addEventListener("pointerup", (event) => {
  if (interaction?.kind === "box") {
    const end = canvasPoint(event, viewCanvas);
    const extent = Math.hypot(end.x - interaction.start.x, end.y - interaction.start.y);
    if (extent >= 10) addAnnotation(regionFromViewRectangle(interaction.start, end, viewCanvas, view));
  }
  interaction = null;
  scheduleRender();
});

viewCanvas.addEventListener("wheel", (event) => {
  event.preventDefault();
  view.horizontalFovDeg = clamp(view.horizontalFovDeg * Math.exp(event.deltaY * 0.001), 20, 120);
  scheduleRender();
}, { passive: false });

erpCanvas.addEventListener("pointerdown", (event) => {
  const point = canvasPoint(event, erpCanvas);
  view.yawDeg = (point.x / erpCanvas.width) * 360 - 180;
  view.pitchDeg = clamp(90 - (point.y / erpCanvas.height) * 180, -89, 89);
  scheduleRender();
});

navigateModeButton.addEventListener("click", () => setMode("navigate"));
annotateModeButton.addEventListener("click", () => setMode("annotate"));
required<HTMLButtonElement>("#openImage").addEventListener("click", () => imageInput.click());
required<HTMLButtonElement>("#importJson").addEventListener("click", () => jsonInput.click());
required<HTMLButtonElement>("#exportJson").addEventListener("click", () => {
  downloadDocument(createDocument(panorama, annotations));
  setStatus(`${annotations.length} anotação(ões) exportada(s).`);
});

labelInput.addEventListener("input", () => {
  const selected = selectedAnnotation();
  if (!selected) return;
  selected.label = labelInput.value;
  selected.updatedAt = new Date().toISOString();
  scheduleRender();
});

stateInput.addEventListener("change", () => {
  const selected = selectedAnnotation();
  if (!selected) return;
  selected.state = stateInput.value as ReviewState;
  selected.updatedAt = new Date().toISOString();
  scheduleRender();
});

focusSelectedButton.addEventListener("click", () => {
  const selected = selectedAnnotation();
  if (!selected) return;
  view.yawDeg = selected.region.centerYawDeg;
  view.pitchDeg = selected.region.centerPitchDeg;
  view.horizontalFovDeg = clamp(selected.region.horizontalFovDeg * 1.8, 20, 110);
  scheduleRender();
});

deleteSelectedButton.addEventListener("click", () => {
  if (!selectedId) return;
  annotations = annotations.filter((annotation) => annotation.id !== selectedId);
  selectedId = null;
  setStatus("Região removida.");
  scheduleRender();
});

imageInput.addEventListener("change", async () => {
  const file = imageInput.files?.[0];
  if (!file) return;
  const url = URL.createObjectURL(file);
  try {
    const image = new Image();
    image.src = url;
    await image.decode();
    const proxyWidth = Math.min(image.naturalWidth, 4096);
    const proxyHeight = Math.max(1, Math.round(proxyWidth * image.naturalHeight / image.naturalWidth));
    panoramaCanvas = document.createElement("canvas");
    panoramaCanvas.width = proxyWidth;
    panoramaCanvas.height = proxyHeight;
    const context = panoramaCanvas.getContext("2d", { willReadFrequently: true });
    if (!context) throw new Error("Canvas 2D is unavailable.");
    context.drawImage(image, 0, 0, proxyWidth, proxyHeight);
    panoramaPixels = { width: proxyWidth, height: proxyHeight, data: context.getImageData(0, 0, proxyWidth, proxyHeight).data };
    panorama = {
      id: file.name.replace(/\.[^.]+$/, "") || "panorama",
      fileName: file.name,
      width: image.naturalWidth,
      height: image.naturalHeight,
    };
    annotations = [];
    selectedId = null;
    view = { yawDeg: 0, pitchDeg: 0, horizontalFovDeg: 72 };
    fileName.textContent = file.name;
    fileSize.textContent = `${image.naturalWidth} × ${image.naturalHeight}`;
    setStatus(proxyWidth < image.naturalWidth
      ? `ERP aberta; proxy de visualização ${proxyWidth} × ${proxyHeight}.`
      : "ERP aberta localmente.");
    scheduleRender();
  } catch (error) {
    setStatus(error instanceof Error ? error.message : "Não foi possível abrir a imagem.");
  } finally {
    URL.revokeObjectURL(url);
    imageInput.value = "";
  }
});

jsonInput.addEventListener("change", async () => {
  const file = jsonInput.files?.[0];
  if (!file) return;
  try {
    const document = parseDocument(JSON.parse(await file.text())) as AnnotationDocument;
    annotations = document.annotations;
    selectedId = annotations[0]?.id ?? null;
    if (document.panorama.fileName !== panorama.fileName) {
      setStatus(`JSON importado para ${document.panorama.fileName}; abra a ERP correspondente para revisar.`);
    } else {
      setStatus(`${annotations.length} anotação(ões) importada(s).`);
    }
    scheduleRender();
  } catch (error) {
    setStatus(error instanceof Error ? error.message : "JSON inválido.");
  } finally {
    jsonInput.value = "";
  }
});

window.addEventListener("keydown", (event) => {
  if (event.target instanceof HTMLInputElement || event.target instanceof HTMLSelectElement) return;
  if (event.key.toLowerCase() === "a") setMode("annotate");
  if (event.key.toLowerCase() === "v") setMode("navigate");
  if (event.key === "Escape") {
    interaction = null;
    setMode("navigate");
  }
  if ((event.key === "Delete" || event.key === "Backspace") && selectedId) deleteSelectedButton.click();
});

new ResizeObserver(scheduleRender).observe(perspectiveStage);
setMode("navigate");
scheduleRender();
