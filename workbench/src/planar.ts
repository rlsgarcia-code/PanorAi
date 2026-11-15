import "./planar.css";
import "./image-tool-rail.css";
import {
  clamp,
  erpPixelToRay,
  rayToErpPixel,
  rayToViewPixel,
  viewPixelToRay,
} from "./geometry.ts";
import type { Point2, ViewState } from "./model.ts";
import { fillClosedContours } from "./planar-fill.ts";
import { planarHistoryShortcut } from "./planar-navigation.ts";
import { moveSphericalViewport, zoomSphericalViewport } from "./spherical-interaction.ts";
import { mountImageToolRail } from "./image-tool-rail.ts";
import {
  suggestSegmentationRoi,
  validateClassifierSeeds,
} from "./planar-workflow.ts";
import { renderGnomonic } from "./renderer.ts";
import type { PanoramaPixels } from "./renderer.ts";
import {
  decodeMask,
  encodeMask,
  paintDisc,
  polygonMask,
  brushOutline,
  planarToolCursor,
} from "./planar-model.ts";
import type {
  PlaneRegion,
  PlanarDocument,
  RleMask,
  PlanarTool,
} from "./planar-model.ts";
type Side = "a" | "b";
type Tool = PlanarTool;
interface Picture {
  file?: File;
  canvas: HTMLCanvasElement;
  pixels: PanoramaPixels;
}
interface Artifact {
  id: string;
  type: string;
  label: string;
  mime_type: string;
}
interface Run {
  id: string;
  status: string;
  artifacts: Artifact[];
  parameters: Record<string, unknown>;
  metrics: Record<string, unknown>;
  error?: string;
  events: {
    state: string;
    message: string;
  }[];
  planar_scene?: PlanarDocument;
}
interface Track {
  plane_id: string;
  track_id: string;
  point_world: number[] | null;
  reason: string | null;
  residual_deg: number;
  observations: {
    view_id: string;
    pixel_xy: [number, number];
  }[];
}
interface Plane {
  plane_id: string;
  state: string;
  normal_world: number[] | null;
  offset_m: number | null;
  diagnostics: Record<string, unknown>;
}
interface Result {
  planes: Plane[];
  tracks: Track[];
  proposals: PlaneRegion[];
  regions: PlaneRegion[];
  segmentation_alternatives: {
    plane_id: string;
    view_id: string;
    masks: RleMask[];
  }[];
  ranges: {
    view_id: string;
    valid_pixels: number;
    conflict_pixels: number;
  }[];
}
const app = document.querySelector<HTMLDivElement>("#app")!;
app.innerHTML = `
<header class="planar-top"><a class="planar-brand" href="/"><strong>PanorAi Workbench</strong><small>Multi View · reconstrução de planos</small></a><nav><a href="/">Catálogo</a><a href="/single-view.html">Single View</a><a href="/two-view.html">Two View</a><a class="active" href="/multi-view.html">Multi View</a></nav><small id="runtime">Verificando runtime…</small></header>
<main class="planar-layout">
 <aside class="planar-sidebar">
  <section><h1>Cena registrada</h1><small>Panoramas + poses métricas</small><input id="scene-file" type="file" accept=".json" hidden/><input id="image-files" type="file" accept="image/*" multiple hidden/><button class="file-control" id="open-scene">1 · Abrir cena JSON</button><button class="file-control" id="open-images">2 · Carregar panoramas</button><button class="file-control" id="save">Salvar revisão</button><button class="file-control" id="download-scene">Exportar anotações</button><small id="revision">Nenhuma cena carregada</small></section>
  <section><h2>Vistas</h2><div class="view-list" id="views"></div></section>
  <section><h2>Planos</h2><button id="add-plane">+ Novo plano</button><div class="plane-list" id="planes"></div></section>
  <section><h2>Histórico</h2><div id="runs"></div></section>
 </aside>
 <section class="planar-main">
  <div class="planar-toolbar"><label>A <select id="view-a" aria-label="Vista A"></select></label><label>B <select id="view-b" aria-label="Vista B"></select></label><button id="edit-tab" class="active">Editar</button><button id="result-tab">Resultados</button><small id="dimensions">Resolução nativa</small></div>
  <div class="canvas-controls" role="toolbar" aria-label="Histórico e navegação">
   <div class="control-group"><button id="undo" title="Desfazer · Ctrl/⌘ Z" disabled><span aria-hidden="true">↶</span> Desfazer</button><button id="redo" title="Refazer · Ctrl/⌘ Shift Z ou Ctrl Y" disabled><span aria-hidden="true">↷</span> Refazer</button></div>
   <div class="control-group fov-controls"><label>Vista ativa<select id="navigation-side" aria-label="Vista ativa para navegação"><option value="a">A</option><option value="b">B</option></select></label><button id="zoom-out" aria-label="Afastar" title="Afastar · aumentar FOV">−</button><label class="fov-control">FOV<input id="fov" aria-label="Campo de visão horizontal" type="range" min="15" max="120" step="1" value="72"/></label><output id="fov-value" for="fov">72°</output><button id="zoom-in" aria-label="Aproximar" title="Aproximar · reduzir FOV">+</button></div>
   <small class="navigation-help" id="navigation-help">Arraste com a mão · Espaço temporário · roda para zoom</small>
  </div>
  <div class="pair-workspace" id="edit-workspace"><div class="planar-pane"><canvas id="canvas-a" aria-label="Vista gnomônica A"></canvas><div class="pane-tag" id="tag-a">A · referência</div><div class="empty-planar" id="empty-a"><strong>Marque uma superfície</strong><p>Abra uma cena com poses e seus panoramas. Pinte um seed ou desenhe a seleção de um plano.</p></div></div><div class="planar-pane"><canvas id="canvas-b" aria-label="Vista gnomônica B"></canvas><div class="pane-tag" id="tag-b">B · apoio</div><div class="empty-planar" id="empty-b"><strong>Confira em outra vista</strong><p>Revise a máscara projetada, inspecione as correspondências e aceite a região antes de gerar o range.</p></div></div></div>
  <div class="planar-result" id="results"><div class="result-gallery" id="gallery"></div></div>
  <div class="erp-strip" id="erp-strip"><canvas id="erp-a" aria-label="Contexto ERP A"></canvas><canvas id="erp-b" aria-label="Contexto ERP B"></canvas></div>
  <div class="pipeline-status" id="pipeline"><span>Marcar</span><span>Segmentar</span><span>Confirmar</span><span>Reconstruir</span></div>
  <div class="planar-feedback" id="feedback" role="status">Selecione uma cena registrada. As máscaras são mantidas na grade ERP nativa.</div>
 </section>
 <aside class="planar-inspector">
  <section class="workflow-step" id="step-draw"><h2><span class="step-number">1</span> Marcar plano</h2>
   <p class="step-help">Pinte uma estimativa da superfície. Use a borracha para excluir objetos à frente.</p>
   <label>Nome do plano<input id="plane-label" value="Plano"/></label>
   <p class="tool-rail-note">Use a régua junto às imagens para mão, zoom, pontos, brush e polígonos.</p>
   <button id="close-polygon" class="full-control" hidden>Fechar polígono</button>
   <label>Raio do brush (graus)<input id="radius" type="range" min="0.1" max="8" step="0.1" value="1"/></label>
   <label class="check"><input id="brush-fill-closed" type="checkbox"/><span>Preencher interior ao fechar (Brush +)</span></label>
   <button id="fill-contours" class="full-control">Preencher contornos existentes</button>
  </section>
  <section class="workflow-step" id="step-segment"><h2><span class="step-number">2</span> Segmentar</h2>
   <p class="step-help" id="segmentation-hint">Ajuste a estimativa aos limites da superfície. A área de busca é automática.</p>
   <button id="draw-roi" class="full-control" hidden>Desenhar limite da região</button>
   <button id="segment" class="primary-planar">Segmentar região</button><button id="cancel" class="full-control" hidden>Cancelar execução</button>
  </section>
  <section class="workflow-step" id="step-review"><h2><span class="step-number">3</span> Revisar e confirmar</h2>
   <p class="step-help" id="review-hint">Confira a máscara sobre a imagem. Corrija com o brush, se necessário.</p>
   <small id="region-state">Escolha ou crie um plano</small>
   <div class="tool-grid"><button id="accept">Aceitar máscara</button><button id="reject">Rejeitar máscara</button></div>
   <div id="proposal-actions"></div>
  </section>
  <section class="workflow-step" id="step-reconstruct"><h2><span class="step-number">4</span> Reconstruir</h2>
   <p class="step-help" id="reconstruct-hint">Confirme ao menos uma máscara para estimar o plano. Depois, revise as propostas nas outras vistas.</p>
   <button id="reconstruct" class="primary-planar">Estimar planos e ranges</button>
  </section>
  <details class="advanced-options" id="advanced-options"><summary>Avançado <span>Métodos e parâmetros</span></summary><div class="advanced-body">
   <label>Segmentação<select id="segmentation" aria-label="Segmentação"><option value="simple">Simples · GrabCut</option><option value="watershed">Watershed · sementes e bordas</option><option value="classifier">Classificador local · positivos / negativos</option><option value="sam">SAM 2.1</option></select></label>
   <p class="step-help" id="method-hint">GrabCut usa aparência e suavidade local.</p>
   <label class="check" id="sam-terms-label" hidden><input id="sam-terms" type="checkbox"/><span>Revisei e aceito os termos upstream do SAM 2.1 para carregar seus pesos.</span></label>
   <label>Área de busca<select id="roi-mode"><option value="auto">Automática</option><option value="manual">ROI desenhada manualmente</option></select></label>
   <label>Desenho como<select id="interpretation"><option value="seed">Seed para segmentar</option><option value="final">Seleção final</option></select></label>
   <label>Polígono define<select id="polygon-target"><option value="mask">Região / seed</option><option value="roi">ROI de expansão</option></select></label>
   <label>Correspondências<select id="matching"><option value="sift">SIFT + pontos manuais</option><option value="manual">Somente pontos manuais</option></select></label>
   <p class="tool-rail-note">Pares de pontos usam a ferramenta Ponto da mesma régua.</p>
   <label>Resolução da vista de trabalho<select id="face-size"><option value="512">512</option><option value="1024" selected>1024</option><option value="2048">2048</option></select></label>
   <label>Tolerância do plano (m)<input id="threshold" type="number" min="0.0001" step="0.005" value="0.03"/></label>
   <label>Sigma máximo do range (m)<input id="sigma" type="number" min="0.001" step="0.05" value="0.25"/></label>
   <h3>Diagnósticos e alternativas</h3><div class="planar-evidence" id="evidence">Os diagnósticos aparecerão após uma execução.</div>
  </div></details>
 </aside>
</main>`;
function el<T extends HTMLElement>(id: string): T {
  return document.getElementById(id) as T;
}
function text(value: unknown): string {
  return String(value ?? "—").replace(
    /[&<>"']/g,
    (c) =>
      ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[
        c
      ]!,
  );
}
function message(value: string, error = false) {
  el("feedback").textContent = value;
  el("feedback").classList.toggle("error", error);
}
const pictures = new Map<string, Picture>();
let doc: PlanarDocument | null = null,
  planeId = "",
  busy = false,
  dirty = false,
  result: Result | null = null,
  activeRun: Run | null = null;
let tool: Tool = "positive",
  activeSide: Side = "a";
let drawingLimit = false;
let spaceHeld = false;
const polygonRedo: Point2[] = [];
let hover: { side: Side; point: Point2 } | null = null;
const states: Record<Side, ViewState> = {
  a: { yawDeg: 0, pitchDeg: 0, horizontalFovDeg: 72 },
  b: { yawDeg: 0, pitchDeg: 0, horizontalFovDeg: 72 },
};
let polygon: {
  side: Side;
  points: Point2[];
  view: ViewState;
} | null = null;
let drag: {
  side: Side;
  mode: Tool;
  point: Point2;
  view: ViewState;
  previous: Point2;
  stroke: Point2[];
} | null = null;
let tie: {
  view_id: string;
  pixel_xy: [number, number];
} | null = null;
let pointDrag:null|{side:Side;trackIndex:number;observationIndex:number}=null;
const undo: string[] = [],
  redo: string[] = [];
const planarImageTools=mountImageToolRail(el("edit-workspace"),{label:"Navegação e seleção multivista",active:"brush-positive",tools:[
  {id:"navigate",kind:"navigate",label:"Mão",title:"Mover a vista ativa"},
  {id:"zoom",kind:"zoom",label:"Zoom",title:"Aproximar ou afastar a vista ativa"},
  {id:"point",kind:"point",label:"Ponto",title:"Criar ou selecionar uma correspondência",separatorBefore:true},
  {id:"line",kind:"line",label:"Reta",title:"O estágio atual não solicita retas",enabled:false},
  {id:"brush-positive",kind:"brush",label:"Brush positivo",title:"Marcar pixels do plano",separatorBefore:true},
  {id:"brush-negative",kind:"brush",label:"Brush negativo",title:"Excluir pixels do plano"},
  {id:"polygon",kind:"polygon",label:"Polígono",title:"Delimitar uma região"},
],onSelect:id=>setTool(id==="point"?"tie":id==="brush-positive"?"positive":id==="brush-negative"?"negative":id as Tool)});
let frame = 0;
const maskCache = new WeakMap<RleMask, Uint8Array>();
function denseMask(mask: RleMask): Uint8Array {
  let values = maskCache.get(mask);
  if (!values) {
    values = decodeMask(mask);
    maskCache.set(mask, values);
  }
  return values;
}
let livePaint: {
  region: PlaneRegion;
  mask: Uint8Array;
  fg: Uint8Array;
  bg: Uint8Array;
  support: Uint8Array | null;
  size: [number, number];
} | null = null;
function viewId(side: Side): string {
  return el<HTMLSelectElement>(`view-${side}`).value;
}
function picture(side: Side): Picture | undefined {
  return pictures.get(viewId(side));
}
function shape(side: Side): [number, number] {
  const p = picture(side);
  if (!p) throw new Error("Carregue os panoramas.");
  return [p.pixels.height, p.pixels.width];
}
function region(side: Side, create = false): PlaneRegion | undefined {
  if (!doc || !planeId || !picture(side)) return;
  let r = doc.regions.find(
    (r) => r.plane_id === planeId && r.view_id === viewId(side),
  );
  if (!r && create) {
    const size = shape(side);
    r = {
      plane_id: planeId,
      view_id: viewId(side),
      label: el<HTMLInputElement>("plane-label").value,
      state: "draft",
      mask: encodeMask(new Uint8Array(size[0] * size[1]), size),
      interpretation: el<HTMLSelectElement>("interpretation").value as
        "seed" | "final",
      history: [],
      source: "human",
    };
    doc.regions.push(r);
  }
  return r;
}
function checkpoint() {
  if (!doc) return;
  undo.push(JSON.stringify(doc));
  if (undo.length > 40) undo.shift();
  redo.length = 0;
}
function changed() {
  dirty = true;
  el("revision").textContent =
    "Alterações locais · resultados anteriores precisam ser recalculados";
  refreshLists();
  schedule();
}
function navigationMode(): Tool {
  return drag?.mode ?? (spaceHeld ? "navigate" : tool);
}
function syncTools() {
  const mode = navigationMode();
  planarImageTools.setActive(
    mode === "tie"
      ? "point"
      : mode === "positive"
        ? "brush-positive"
        : mode === "negative"
          ? "brush-negative"
          : mode,
  );
  document.querySelectorAll<HTMLButtonElement>("[data-tool]").forEach((b) => {
    b.classList.toggle("active", b.dataset.tool === mode);
    b.setAttribute("aria-pressed", String(b.dataset.tool === mode));
  });
  for (const side of ["a", "b"] as Side[]) {
    el<HTMLCanvasElement>(`canvas-${side}`).style.cursor =
      mode === "navigate" && drag?.side === side
        ? "grabbing"
        : planarToolCursor(mode);
    updateBrushPreview(side);
  }
}
function updateViewControls() {
  const locked = busy || !!drag || !!polygon;
  el<HTMLSelectElement>("navigation-side").value = activeSide;
  el<HTMLInputElement>("fov").value = String(
    states[activeSide].horizontalFovDeg,
  );
  el("fov-value").textContent =
    `${Math.round(states[activeSide].horizontalFovDeg)}°`;
  el<HTMLButtonElement>("undo").disabled =
    busy || !!drag || !(undo.length || polygon?.points.length);
  el<HTMLButtonElement>("redo").disabled =
    busy || !!drag || !(redo.length || polygonRedo.length);
  for (const id of ["fov", "zoom-in", "zoom-out", "navigation-side"])
    el<HTMLButtonElement | HTMLInputElement | HTMLSelectElement>(id).disabled =
      locked;
  for (const mode of ["navigate", "zoom"])
    planarImageTools.setEnabled(mode, !locked);
  for (const side of ["a", "b"] as Side[])
    el(`canvas-${side}`).parentElement!.classList.toggle(
      "pane-active",
      side === activeSide,
    );
}
function zoomActive(factor: number) {
  if (busy || drag || polygon) return;
  states[activeSide] = zoomSphericalViewport(states[activeSide], factor);
  schedule();
}
function setTool(value: Tool) {
  if (busy || drag) return;
  if (polygon && (value === "navigate" || value === "zoom")) {
    message(
      "Feche com Enter ou cancele com Esc o polígono antes de mover a vista.",
    );
    return;
  }
  tool = value;
  planarImageTools.setActive(value==="tie"?"point":value==="positive"?"brush-positive":value==="negative"?"brush-negative":value);
  drawingLimit = false;
  polygon = null;
  polygonRedo.length = 0;
  tie = null;
  el("close-polygon").hidden = value !== "polygon";
  syncTools();
  message(
    value === "tie"
      ? "Clique no mesmo ponto físico em A e B. Repita para distribuir apoios sobre o plano."
      : value === "polygon"
        ? "Clique para adicionar vértices. Enter fecha; Esc cancela."
        : value === "positive"
          ? "Brush + ativo: arraste na imagem para pintar. Sem plano selecionado, a primeira pincelada cria um plano."
          : value === "negative"
            ? "Brush − ativo: arraste para apagar e marcar exclusões da superfície."
            : value === "zoom"
              ? "Zoom ativo: arraste para cima para aproximar, para baixo para afastar. O FOV também pode ser ajustado na barra."
              : "Mão ativa: arraste para girar a esfera. Segure Espaço para usar a mão temporariamente; roda para zoom.",
  );
  schedule();
}
function updateBrushPreview(side: Side) {
  const outline = document.querySelector<SVGSVGElement>(
    `#brush-preview-${side}`,
  );
  if (!outline) return;
  const visible =
    hover?.side === side &&
    picture(side) &&
    !busy &&
    (navigationMode() === "positive" || navigationMode() === "negative");
  outline.style.display = visible ? "block" : "none";
  if (!visible || !hover) return;
  const canvas = el<HTMLCanvasElement>(`canvas-${side}`);
  outline.setAttribute("viewBox", `0 0 ${canvas.width} ${canvas.height}`);
  outline.dataset.mode = tool;
  const points = brushOutline(
    hover.point,
    canvas,
    states[side],
    Number(el<HTMLInputElement>("radius").value),
  );
  outline
    .querySelector("path")!
    .setAttribute(
      "d",
      points.map((p, i) => `${i ? "L" : "M"}${p.x},${p.y}`).join(" ") + " Z",
    );
}
function startPlane(side: Side) {
  if (!doc) return;
  planeId = `plane-${crypto.randomUUID().slice(0, 8)}`;
  el<HTMLInputElement>("plane-label").value =
    `Plano ${new Set(doc.regions.map((r) => r.plane_id)).size + 1}`;
  region(side, true);
}
function schedule() {
  if (!frame)
    frame = requestAnimationFrame(() => {
      frame = 0;
      render();
    });
}
function maskAt(side: Side): {
  mask: Uint8Array;
  support: Uint8Array | null;
  roi: Uint8Array | null;
  negative: Uint8Array | null;
} | null {
  const r = region(side),
    v = doc?.views.find((v) => v.view_id === viewId(side));
  if (!v || !r) return null;
  return {
    mask: livePaint?.region === r ? livePaint.mask : denseMask(r.mask),
    support: v.support === "full" ? null : denseMask(v.support),
    roi: r.roi ? denseMask(r.roi) : null,
    negative:
      livePaint?.region === r
        ? livePaint.bg
        : r.negative
          ? denseMask(r.negative)
          : null,
  };
}
function render() {
  updateViewControls();
  const nativeSizes = (["a", "b"] as Side[]).map((side) => {
    const p = picture(side);
    return p ? `${p.pixels.width}×${p.pixels.height}` : "—";
  });
  el("dimensions").textContent = `${nativeSizes.join(" | ")} · range nativo`;
  for (const side of ["a", "b"] as Side[]) {
    const canvas = el<HTMLCanvasElement>(`canvas-${side}`),
      erp = el<HTMLCanvasElement>(`erp-${side}`),
      p = picture(side);
    const bounds = canvas.getBoundingClientRect();
    canvas.width = Math.max(1, Math.min(680, Math.round(bounds.width)));
    canvas.height = Math.max(
      1,
      Math.round((bounds.height / Math.max(bounds.width, 1)) * canvas.width),
    );
    el(`empty-${side}`).hidden = !!p;
    const ctx = canvas.getContext("2d", { willReadFrequently: true })!,
      context = erp.getContext("2d", { willReadFrequently: true })!;
    if (!p) {
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      continue;
    }
    renderGnomonic(ctx, p.pixels, states[side], canvas.width, canvas.height);
    const selected = maskAt(side),
      r = region(side);
    if (selected) {
      const output = ctx.getImageData(0, 0, canvas.width, canvas.height);
      const roiPreview = selected.roi
        ? new Uint8Array(canvas.width * canvas.height)
        : null;
      for (let y = 0; y < canvas.height; y++)
        for (let x = 0; x < canvas.width; x++) {
          const ray = viewPixelToRay(
              { x: x + 0.5, y: y + 0.5 },
              canvas,
              states[side],
            ),
            pixel = rayToErpPixel(ray, p.pixels);
          const px =
              ((Math.floor(pixel.x) % p.pixels.width) + p.pixels.width) %
              p.pixels.width,
            py = Math.min(
              p.pixels.height - 1,
              Math.max(0, Math.floor(pixel.y)),
            );
          const i = py * p.pixels.width + px,
            j = (y * canvas.width + x) * 4;
          if (roiPreview) roiPreview[y * canvas.width + x] = selected.roi![i];
          if (selected.support && !selected.support[i]) {
            output.data[j] *= 0.2;
            output.data[j + 1] *= 0.2;
            output.data[j + 2] *= 0.2;
          }
          if (selected.mask[i]) {
            output.data[j] = output.data[j] * 0.6 + 255 * 0.4;
            output.data[j + 1] = output.data[j + 1] * 0.6 + 200 * 0.4;
            output.data[j + 2] = output.data[j + 2] * 0.6 + 87 * 0.4;
          }
          if (selected.negative?.[i]) {
            output.data[j] = 245;
            output.data[j + 1] *= 0.4;
            output.data[j + 2] *= 0.4;
          }
          if (selected.roi && !selected.roi[i]) {
            output.data[j] *= 0.55;
            output.data[j + 1] *= 0.55;
            output.data[j + 2] *= 0.55;
          }
        }
      if (roiPreview)
        for (let y = 1; y < canvas.height - 1; y++)
          for (let x = 1; x < canvas.width - 1; x++) {
            const i = y * canvas.width + x;
            if (
              roiPreview[i] &&
              (!roiPreview[i - 1] ||
                !roiPreview[i + 1] ||
                !roiPreview[i - canvas.width] ||
                !roiPreview[i + canvas.width])
            ) {
              output.data[i * 4] = 90;
              output.data[i * 4 + 1] = 220;
              output.data[i * 4 + 2] = 255;
            }
          }
      ctx.putImageData(output, 0, 0);
    }
    // Matches remain backend decisions; this only projects returned observations.
    const tracks =
      result?.tracks.filter((t) => t.plane_id === planeId) ??
      doc?.correspondences
        .filter((t) => t.plane_id === planeId)
        .map((t) => ({ ...t, reason: null })) ??
      [];
    for (const track of tracks.slice(0, 350)) {
      const observation = track.observations.find(
        (o) => o.view_id === viewId(side),
      );
      if (!observation) continue;
      const ray = erpPixelToRay(
          {
            x: observation.pixel_xy[0] + 0.5,
            y: observation.pixel_xy[1] + 0.5,
          },
          p.pixels,
        ),
        xy = rayToViewPixel(ray, canvas, states[side]);
      if (!xy) continue;
      ctx.beginPath();
      ctx.arc(xy.x, xy.y, 3, 0, Math.PI * 2);
      ctx.strokeStyle = track.reason ? "#ff6e79" : "#c3ee99";
      ctx.lineWidth = 1.5;
      ctx.stroke();
    }
    if (polygon?.side === side) {
      ctx.beginPath();
      polygon.points.forEach((point, i) =>
        i ? ctx.lineTo(point.x, point.y) : ctx.moveTo(point.x, point.y),
      );
      ctx.strokeStyle = "#ffc857";
      ctx.lineWidth = 2;
      ctx.stroke();
      polygon.points.forEach((point) => {
        ctx.beginPath();
        ctx.arc(point.x, point.y, 3, 0, Math.PI * 2);
        ctx.fillStyle = "#ffc857";
        ctx.fill();
      });
    }
    erp.width = 600;
    erp.height = Math.max(
      1,
      Math.round((600 * p.pixels.height) / p.pixels.width),
    );
    context.drawImage(p.canvas, 0, 0, erp.width, erp.height);
    if (selected) {
      const overlay = context.getImageData(0, 0, erp.width, erp.height);
      for (let y = 0; y < erp.height; y++)
        for (let x = 0; x < erp.width; x++) {
          const i =
            Math.min(
              p.pixels.height - 1,
              Math.floor(((y + 0.5) / erp.height) * p.pixels.height),
            ) *
              p.pixels.width +
            Math.min(
              p.pixels.width - 1,
              Math.floor(((x + 0.5) / erp.width) * p.pixels.width),
            );
          if (selected.mask[i]) {
            const j = (y * erp.width + x) * 4;
            overlay.data[j] = overlay.data[j] * 0.5 + 127;
            overlay.data[j + 1] = overlay.data[j + 1] * 0.5 + 100;
            overlay.data[j + 2] = overlay.data[j + 2] * 0.5 + 44;
          }
        }
      context.putImageData(overlay, 0, 0);
    }
    el(`tag-${side}`).textContent =
      `${side.toUpperCase()} · ${viewId(side)} · ${r?.state ?? "sem máscara"}`;
    updateBrushPreview(side);
  }
  const r = region(activeSide);
  el("region-state").textContent = r
    ? `${r.view_id} · ${r.state} · ${r.mask.runs.filter((_, i) => i % 2).reduce((a, b) => a + b, 0)} pixels`
    : `Vista ${activeSide.toUpperCase()} · sem região`;
  const marked = !!r?.mask.runs.length;
  el<HTMLButtonElement>("segment").disabled = busy || !marked;
  el<HTMLButtonElement>("accept").disabled =
    busy || !marked || r?.state === "accepted";
  el<HTMLButtonElement>("reject").disabled = busy || !marked;
  const accepted =
    doc?.regions.filter((r) => r.state === "accepted").length ?? 0;
  el<HTMLButtonElement>("reconstruct").disabled = busy || !accepted;
  el("step-draw").classList.toggle("step-done", marked);
  el("step-review").classList.toggle("step-done", r?.state === "accepted");
  el("review-hint").textContent =
    r?.state === "accepted"
      ? "Máscara confirmada. Marque outro plano ou avance para reconstruir."
      : ["simple", "sam", "watershed", "classifier"].includes(r?.source ?? "")
        ? "Segmentação aplicada à imagem. Corrija se necessário e clique em Aceitar máscara."
        : "Confira a máscara sobre a imagem. Corrija com o brush, se necessário.";
  el("segment").textContent =
    r?.state === "accepted" ? "Segmentar novamente" : "Segmentar região";
  el("reconstruct-hint").textContent = accepted
    ? `${accepted} máscara(s) confirmada(s). Estime o plano e revise as propostas nas outras vistas.`
    : "Confirme ao menos uma máscara antes de estimar o plano.";
}
function refreshLists() {
  const ids = [...new Set(doc?.regions.map((r) => r.plane_id) ?? [])];
  if (planeId && !ids.includes(planeId)) ids.push(planeId);
  el("planes").innerHTML = ids
    .map(
      (id) =>
        `<button data-plane="${text(id)}" class="${id === planeId ? "active" : ""}"><strong>${text(doc?.regions.find((r) => r.plane_id === id)?.label ?? id)}</strong><small>${doc?.regions.filter((r) => r.plane_id === id && r.state === "accepted").length ?? 0} vistas aceitas</small></button>`,
    )
    .join("");
  el("planes")
    .querySelectorAll<HTMLButtonElement>("[data-plane]")
    .forEach(
      (button) =>
        (button.onclick = () => {
          planeId = button.dataset.plane!;
          const r = doc?.regions.find((r) => r.plane_id === planeId);
          el<HTMLInputElement>("plane-label").value = r?.label ?? planeId;
          renderEvidence();
          schedule();
          refreshLists();
        }),
    );
}
function installDocument(next: PlanarDocument) {
  if (
    next.schema !== "panorai-planar-scene/v1" ||
    next.geometry_convention !== "panorai-geometry-v1" ||
    next.units !== "m" ||
    !Array.isArray(next.views) ||
    next.views.length < 2
  )
    throw new Error(
      "Use uma cena panorai-planar-scene/v1, geometry-v1, em metros, com 2+ vistas.",
    );
  next.regions ??= [];
  next.correspondences ??= [];
  doc = next;
  planeId = next.regions[0]?.plane_id ?? "";
  el<HTMLInputElement>("plane-label").value = next.regions[0]?.label ?? "Plano";
  undo.length = 0;
  redo.length = 0;
  polygonRedo.length = 0;
  spaceHeld = false;
  result = null;
  for (const side of ["a", "b"] as Side[]) {
    const select = el<HTMLSelectElement>(`view-${side}`);
    select.innerHTML = next.views
      .map(
        (v) => `<option value="${text(v.view_id)}">${text(v.view_id)}</option>`,
      )
      .join("");
    select.selectedIndex = side === "a" ? 0 : 1;
  }
  el("views").innerHTML = next.views
    .map(
      (v) =>
        `<button data-view="${text(v.view_id)}"><strong>${text(v.view_id)}</strong><small>${text(v.image_key)}</small></button>`,
    )
    .join("");
  el("views")
    .querySelectorAll<HTMLButtonElement>("[data-view]")
    .forEach(
      (button) =>
        (button.onclick = () => {
          el<HTMLSelectElement>("view-b").value = button.dataset.view!;
          schedule();
        }),
    );
  refreshLists();
  setTool(tool);
  schedule();
}
async function loadPicture(blob: Blob, file?: File): Promise<Picture> {
  const url = URL.createObjectURL(blob);
  try {
    const image = new Image();
    image.src = url;
    await image.decode();
    const canvas = document.createElement("canvas");
    canvas.width = image.naturalWidth;
    canvas.height = image.naturalHeight;
    const context = canvas.getContext("2d", { willReadFrequently: true })!;
    context.drawImage(image, 0, 0);
    return {
      file,
      canvas,
      pixels: {
        width: canvas.width,
        height: canvas.height,
        data: context.getImageData(0, 0, canvas.width, canvas.height).data,
      },
    };
  } finally {
    URL.revokeObjectURL(url);
  }
}
function point(event: PointerEvent, canvas: HTMLCanvasElement): Point2 {
  const r = canvas.getBoundingClientRect();
  return {
    x: ((event.clientX - r.left) / r.width) * canvas.width,
    y: ((event.clientY - r.top) / r.height) * canvas.height,
  };
}
function supportMask(side: Side): Uint8Array | null {
  const v = doc!.views.find((v) => v.view_id === viewId(side))!;
  return v.support === "full" ? null : denseMask(v.support);
}
function paint(side: Side, points: Point2[]) {
  const r = region(side, true);
  if (!r) return;
  const canvas = el<HTMLCanvasElement>(`canvas-${side}`),
    radius = Number(el<HTMLInputElement>("radius").value);
  if (livePaint?.region !== r) {
    const mask = denseMask(r.mask).slice();
    livePaint = {
      region: r,
      mask,
      fg: r.positive
        ? denseMask(r.positive).slice()
        : new Uint8Array(mask.length),
      bg: r.negative
        ? denseMask(r.negative).slice()
        : new Uint8Array(mask.length),
      support: supportMask(side),
      size: shape(side),
    };
  }
  const { mask, fg, bg, support, size } = livePaint;
  for (const p of points) {
    const ray = viewPixelToRay(p, canvas, states[side]);
    paintDisc(mask, size, ray, radius, tool === "negative" ? 0 : 1, support);
    paintDisc(fg, size, ray, radius, tool === "negative" ? 0 : 1, support);
    paintDisc(bg, size, ray, radius, tool === "negative" ? 1 : 0, support);
  }
  r.state = "draft";
  r.interpretation = el<HTMLSelectElement>("interpretation").value as
    "seed" | "final";
}
for (const side of ["a", "b"] as Side[]) {
  const canvas = el<HTMLCanvasElement>(`canvas-${side}`);
  canvas.tabIndex = 0;
  canvas.style.cursor = planarToolCursor(tool);
  const preview = document.createElementNS("http://www.w3.org/2000/svg", "svg");
  preview.id = `brush-preview-${side}`;
  preview.classList.add("brush-preview");
  preview.setAttribute("aria-hidden", "true");
  preview.append(
    document.createElementNS("http://www.w3.org/2000/svg", "path"),
  );
  canvas.parentElement!.append(preview);
  canvas.onpointerdown = (event) => {
    if (busy || !picture(side) || event.button !== 0) return;
    event.preventDefault();
    canvas.focus({ preventScroll: true });
    activeSide = side;
    const mode = navigationMode();
    const navigating = mode === "navigate" || mode === "zoom";
    if (navigating && polygon) {
      message("Feche ou cancele o polígono antes de mover a vista.");
      return;
    }
    const p = point(event, canvas);
    canvas.setPointerCapture(event.pointerId);
    if (mode === "polygon") {
      if (!planeId) {
        checkpoint();
        startPlane(side);
      }
      if (!polygon || polygon.side !== side)
        polygon = { side, points: [], view: { ...states[side] } };
      polygon.points.push(p);
      polygonRedo.length = 0;
      schedule();
      return;
    }
    if (mode === "tie") {
      if (!doc || !planeId) return;
      let nearest:null|{trackIndex:number;observationIndex:number;distance:number}=null;
      doc.correspondences.forEach((track,trackIndex)=>{if(track.plane_id!==planeId)return;track.observations.forEach((item,observationIndex)=>{if(item.view_id!==viewId(side))return;const ray=erpPixelToRay({x:item.pixel_xy[0]+.5,y:item.pixel_xy[1]+.5},picture(side)!.pixels),projected=rayToViewPixel(ray,canvas,states[side]);if(!projected)return;const distance=Math.hypot(projected.x-p.x,projected.y-p.y);if(distance<=14&&(!nearest||distance<nearest.distance))nearest={trackIndex,observationIndex,distance}})});
      const selectedObservation=nearest as null|{trackIndex:number;observationIndex:number;distance:number};
      if(selectedObservation){checkpoint();pointDrag={side,trackIndex:selectedObservation.trackIndex,observationIndex:selectedObservation.observationIndex};planarImageTools.setReadout("PONTO SELECIONADO","arraste para ajuste fino");return}
      const pixel = rayToErpPixel(
          viewPixelToRay(p, canvas, states[side]),
          picture(side)!.pixels,
        ),
        observation = {
          view_id: viewId(side),
          pixel_xy: [pixel.x - 0.5, pixel.y - 0.5] as [number, number],
        };
      if (!tie) {
        tie = observation;
        message("Agora clique no mesmo ponto na outra vista.");
      } else if (tie.view_id !== observation.view_id) {
        checkpoint();
        doc.correspondences.push({
          plane_id: planeId,
          track_id: crypto.randomUUID(),
          source: "human",
          observations: [tie, observation],
        });
        tie = null;
        changed();
        message(
          `${doc.correspondences.length} tracks manuais salvos na revisão local.`,
        );
      }
      return;
    }
    let started = false;
    if (!planeId && (mode === "positive" || mode === "negative")) {
      checkpoint();
      startPlane(side);
      started = true;
    }
    if (!navigating && !planeId) {
      message("Crie um plano primeiro.", true);
      return;
    }
    if (!navigating && !started) checkpoint();
    drag = {
      side,
      mode,
      point: p,
      previous: p,
      view: { ...states[side] },
      stroke: [p],
    };
    if (!navigating) paint(side, [p]);
    syncTools();
    schedule();
  };
  canvas.onpointermove = (event) => {
    const p = point(event, canvas);
    hover = { side, point: p };
    updateBrushPreview(side);
    if(pointDrag?.side===side&&doc){const pixel=rayToErpPixel(viewPixelToRay(p,canvas,states[side]),picture(side)!.pixels),observation=doc.correspondences[pointDrag.trackIndex]?.observations[pointDrag.observationIndex];if(observation){observation.pixel_xy=[pixel.x-.5,pixel.y-.5];planarImageTools.setReadout("AJUSTE FINO",`ERP x ${observation.pixel_xy[0].toFixed(2)} · y ${observation.pixel_xy[1].toFixed(2)}`);schedule()}return}
    if (!drag || drag.side !== side) return;
    if (drag.mode === "navigate" || drag.mode === "zoom") {
      states[side] = moveSphericalViewport(
        drag.view,
        drag.point,
        p,
        canvas.width,
        drag.mode,
      );
    } else {
      const steps = Math.max(
        1,
        Math.ceil(Math.hypot(p.x - drag.previous.x, p.y - drag.previous.y) / 3),
      );
      paint(
        side,
        Array.from({ length: steps }, (_, i) => ({
          x: drag!.previous.x + ((p.x - drag!.previous.x) * (i + 1)) / steps,
          y: drag!.previous.y + ((p.y - drag!.previous.y) * (i + 1)) / steps,
        })),
      );
    }
    drag.previous = p;
    schedule();
  };
  canvas.onpointerup = () => {
    if(pointDrag?.side===side){pointDrag=null;planarImageTools.setReadout(null);changed();return}
    if (
      drag?.side === side &&
      (drag.mode === "positive" || drag.mode === "negative")
    ) {
      let filled = 0;
      if (livePaint) {
        const { region: r, mask, fg, bg, size, support } = livePaint;
        if (
          drag.mode === "positive" &&
          el<HTMLInputElement>("brush-fill-closed").checked
        )
          filled = fillClosedContours(mask, size, {
            support,
            excluded: bg,
            positive: fg,
            pole: states[side].pitchDeg >= 0 ? "north" : "south",
          });
        r.mask = encodeMask(mask, size);
        r.positive = encodeMask(fg, size);
        r.negative = encodeMask(bg, size);
        maskCache.set(r.mask, mask);
        maskCache.set(r.positive, fg);
        maskCache.set(r.negative, bg);
        livePaint = null;
      }
      const r = region(side);
      const ending = drag.previous;
      drag.stroke.push(ending);
      r?.history.push({
        tool: drag.mode,
        fill_closed:
          drag.mode === "positive" &&
          el<HTMLInputElement>("brush-fill-closed").checked,
        filled_pixels: filled,
        at: new Date().toISOString(),
        radius_deg: Number(el<HTMLInputElement>("radius").value),
        view: { ...states[side] },
        rays_xyz: drag.stroke.map((p) => {
          const ray = viewPixelToRay(p, canvas, states[side]);
          return [ray.x, ray.y, ray.z];
        }),
      });
      changed();
      if (filled)
        message(
          `Contorno preenchido: ${filled.toLocaleString("pt-BR")} pixels adicionados. Desfazer reverte o traço e o preenchimento juntos.`,
        );
    }
    drag = null;
    syncTools();
    schedule();
  };
  canvas.onpointercancel = () => {
    pointDrag=null;planarImageTools.setReadout(null);
    drag = null;
    livePaint = null;
    hover = null;
    syncTools();
    schedule();
    updateBrushPreview(side);
  };
  canvas.onpointerleave = () => {
    if (hover?.side === side) hover = null;
    updateBrushPreview(side);
  };
  canvas.addEventListener("pointermove", (event) => {
    if (
      drag?.side === side &&
      (drag.mode === "positive" || drag.mode === "negative")
    )
      drag.stroke.push(point(event, canvas));
  });
  canvas.addEventListener(
    "wheel",
    (event) => {
      event.preventDefault();
      if (busy || polygon || drag) return;
      activeSide = side;
      states[side] = zoomSphericalViewport(
        states[side],
        Math.exp(clamp(event.deltaY * 0.001, -5, 5)),
      );
      schedule();
    },
    { passive: false },
  );
  el<HTMLCanvasElement>(`erp-${side}`).onpointerdown = (event) => {
    if (busy || drag) return;
    activeSide = side;
    const c = el<HTMLCanvasElement>(`erp-${side}`),
      p = point(event, c);
    states[side].yawDeg = (p.x / c.width) * 360 - 180;
    states[side].pitchDeg = clamp(90 - (p.y / c.height) * 180, -89.9, 89.9);
    polygon = null;
    schedule();
  };
  el<HTMLSelectElement>(`view-${side}`).onchange = () => {
    activeSide = side;
    polygon = null;
    tie = null;
    schedule();
  };
}
function closePolygon() {
  if (busy) return;
  if (!polygon || polygon.points.length < 3) return;
  checkpoint();
  const { side, points, view } = polygon,
    r = region(side, true)!;
  const size = shape(side),
    mask = polygonMask(
      size,
      points,
      view,
      el<HTMLCanvasElement>(`canvas-${side}`),
    ),
    supported = supportMask(side);
  if (supported) for (let i = 0; i < mask.length; i++) mask[i] &= supported[i];
  if (el<HTMLSelectElement>("polygon-target").value === "roi")
    r.roi = encodeMask(mask, size);
  else {
    r.mask = encodeMask(mask, size);
    r.positive = undefined;
    r.negative = undefined;
  }
  r.state = "draft";
  r.interpretation = el<HTMLSelectElement>("interpretation").value as
    "seed" | "final";
  r.history.push({
    tool: "polygon",
    target: el<HTMLSelectElement>("polygon-target").value,
    view,
    points,
    rays_xyz: points.map((p) => {
      const ray = viewPixelToRay(
        p,
        el<HTMLCanvasElement>(`canvas-${side}`),
        view,
      );
      return [ray.x, ray.y, ray.z];
    }),
    at: new Date().toISOString(),
  });
  polygon = null;
  polygonRedo.length = 0;
  if (drawingLimit) {
    drawingLimit = false;
    el<HTMLSelectElement>("polygon-target").value = "mask";
    setTool("positive");
    message(
      "Limite definido (azul). Pinte exemplos + na superfície e − nos objetos a excluir; depois clique Segmentar.",
    );
  }
  changed();
}
el("close-polygon").onclick = closePolygon;
el("fill-contours").onclick = () => {
  if (busy || drag) return;
  const r = region(activeSide);
  if (!r) {
    message(
      "Selecione um plano e desenhe um contorno com o Brush + primeiro.",
      true,
    );
    return;
  }
  const mask = denseMask(r.mask).slice(),
    size = shape(activeSide);
  const positive = r.positive
    ? denseMask(r.positive).slice()
    : new Uint8Array(mask.length);
  const filled = fillClosedContours(mask, size, {
    support: supportMask(activeSide),
    excluded: r.negative ? denseMask(r.negative) : null,
    positive,
    pole: states[activeSide].pitchDeg >= 0 ? "north" : "south",
  });
  if (!filled) {
    message(
      "Nenhum interior novo para preencher. Feche o contorno sem aberturas; exclusões são preservadas.",
    );
    return;
  }
  checkpoint();
  r.mask = encodeMask(mask, size);
  r.positive = encodeMask(positive, size);
  maskCache.set(r.mask, mask);
  maskCache.set(r.positive, positive);
  r.state = "draft";
  r.history.push({
    tool: "fill-closed",
    filled_pixels: filled,
    view: { ...states[activeSide] },
    at: new Date().toISOString(),
  });
  changed();
  message(
    `Contorno preenchido: ${filled.toLocaleString("pt-BR")} pixels adicionados. Revise a máscara antes de aceitar.`,
  );
};
document
  .querySelectorAll<HTMLButtonElement>("[data-tool]")
  .forEach((b) => (b.onclick = () => setTool(b.dataset.tool as Tool)));
el("add-plane").onclick = () => {
  if (!doc) {
    message("Abra uma cena primeiro.", true);
    return;
  }
  checkpoint();
  startPlane(activeSide);
  setTool("positive");
  changed();
};
el("accept").onclick = () => {
  const r = region(activeSide);
  if (!r || !r.mask.runs.length) return;
  checkpoint();
  r.state = "accepted";
  r.interpretation = "final";
  r.history.push({ action: "accept", at: new Date().toISOString() });
  changed();
};
el("reject").onclick = () => {
  const r = region(activeSide);
  if (!r) return;
  checkpoint();
  r.state = "rejected";
  changed();
};
el<HTMLInputElement>("plane-label").onchange = () => {
  if (!doc) return;
  checkpoint();
  for (const r of doc.regions)
    if (r.plane_id === planeId)
      r.label = el<HTMLInputElement>("plane-label").value;
  changed();
};
function undoAnnotation() {
  if (busy || drag) return;
  if (polygon?.points.length) {
    polygonRedo.push(polygon.points.pop()!);
    schedule();
    return;
  }
  if (!doc || !undo.length) return;
  redo.push(JSON.stringify(doc));
  doc = JSON.parse(undo.pop()!) as PlanarDocument;
  if (!doc.regions.some((r) => r.plane_id === planeId))
    planeId = doc.regions[0]?.plane_id ?? "";
  changed();
}
function redoAnnotation() {
  if (busy || drag) return;
  if (polygon && polygonRedo.length) {
    polygon.points.push(polygonRedo.pop()!);
    schedule();
    return;
  }
  if (!doc || !redo.length) return;
  undo.push(JSON.stringify(doc));
  doc = JSON.parse(redo.pop()!) as PlanarDocument;
  if (!doc.regions.some((r) => r.plane_id === planeId))
    planeId = doc.regions[0]?.plane_id ?? "";
  changed();
}
el("undo").onclick = undoAnnotation;
el("redo").onclick = redoAnnotation;
el("zoom-in").onclick = () => zoomActive(1 / 1.2);
el("zoom-out").onclick = () => zoomActive(1.2);
el<HTMLSelectElement>("navigation-side").onchange = () => {
  if (busy || drag || polygon) return;
  activeSide = el<HTMLSelectElement>("navigation-side").value as Side;
  schedule();
};
el<HTMLInputElement>("fov").oninput = () => {
  if (busy || drag || polygon) return;
  states[activeSide] = {
    ...states[activeSide],
    horizontalFovDeg: Number(el<HTMLInputElement>("fov").value),
  };
  schedule();
};
el<HTMLInputElement>("radius").oninput = () => {
  for (const side of ["a", "b"] as Side[]) updateBrushPreview(side);
};
el<HTMLSelectElement>("segmentation").onchange = () => {
  const method = el<HTMLSelectElement>("segmentation").value;
  el("sam-terms-label").hidden = method !== "sam";
  el("draw-roi").hidden = !["watershed", "classifier"].includes(method);
  const hints: Record<string, string> = {
    simple: "GrabCut usa aparência e suavidade local.",
    watershed:
      "As sementes + iniciam os vales; bordas da imagem formam barreiras. Use − para impedir vazamentos. O limite azul restringe a expansão.",
    classifier:
      "Aprende cores com seus exemplos + e −, somente nesta região. Não reconhece planos nem usa modelo externo. Desenhe o limite e pinte as duas classes dentro dele.",
    sam: "SAM usa os prompts marcados e requer os termos upstream para carregar os pesos.",
  };
  el("method-hint").textContent = hints[method];
  el("segmentation-hint").textContent =
    method === "classifier"
      ? "Defina um limite → pinte exemplos + e − → Segmentar. Azul: limite; dourado: região; vermelho: negativos."
      : method === "watershed"
        ? "Pinte sementes + e exclusões −. A expansão segue bordas; desenhe um limite se precisar restringi-la."
        : "Ajuste a estimativa aos limites da superfície. A área de busca é automática por padrão.";
  if (method === "classifier")
    el<HTMLSelectElement>("roi-mode").value = "manual";
};
el("draw-roi").onclick = () => {
  if (busy || !doc) return;
  el<HTMLSelectElement>("roi-mode").value = "manual";
  el<HTMLSelectElement>("polygon-target").value = "roi";
  setTool("polygon");
  drawingLimit = true;
  message(
    "Clique nos vértices do limite e pressione Enter para fechar. Isso não marca toda a área como positiva.",
  );
};
window.onkeydown = (event) => {
  if (
    (event.target as HTMLElement).closest(
      "input,select,textarea,[contenteditable=true]",
    )
  )
    return;
  const history = planarHistoryShortcut(event);
  if (history) {
    event.preventDefault();
    if (history === "undo") undoAnnotation();
    else redoAnnotation();
    return;
  }
  if (busy || event.metaKey || event.ctrlKey || event.altKey) return;
  if (event.code === "Space") {
    event.preventDefault();
    if (!drag && !spaceHeld) {
      spaceHeld = true;
      syncTools();
    }
    return;
  }
  if (event.key === "Enter" && polygon) {
    event.preventDefault();
    closePolygon();
  }
  if (event.key === "Escape") {
    polygon = null;
    polygonRedo.length = 0;
    drawingLimit = false;
    tie = null;
    schedule();
  }
  const tools: Record<string, Tool> = {
    v: "navigate",
    h: "navigate",
    z: "zoom",
    b: "positive",
    e: "negative",
    p: "polygon",
    t: "tie",
  };
  if (tools[event.key.toLowerCase()]) setTool(tools[event.key.toLowerCase()]);
};
window.onkeyup = (event) => {
  if (event.code === "Space" && spaceHeld) {
    event.preventDefault();
    spaceHeld = false;
    syncTools();
  }
};
window.addEventListener("blur", () => {
  spaceHeld = false;
  syncTools();
});
async function request(url: string, options?: RequestInit) {
  const response = await fetch(url, options);
  if (!response.ok) {
    const value = (await response.json()) as {
      detail?: string;
    };
    throw new Error(value.detail ?? `HTTP ${response.status}`);
  }
  return response;
}
function jsonOptions(value: unknown): RequestInit {
  return {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(value),
  };
}
async function saveRevision() {
  if (!doc) throw new Error("Abra uma cena primeiro.");
  if (!doc.views.every((v) => pictures.has(v.view_id)))
    throw new Error("Carregue todas as imagens da cena.");
  let response: Response;
  if (doc.scene_id && doc.revision) {
    response = await request(
      `/api/v1/planar/scenes/${doc.scene_id}/${doc.revision}/revisions`,
      jsonOptions(doc),
    );
  } else {
    const form = new FormData();
    form.set("document", JSON.stringify(doc));
    for (const v of doc.views) {
      const file = pictures.get(v.view_id)!.file;
      if (!file) throw new Error("Imagem original indisponível.");
      form.append("images", file, v.image_key);
    }
    response = await request("/api/v1/planar/scenes", {
      method: "POST",
      body: form,
    });
  }
  const saved = (await response.json()) as {
    scene_id: string;
    revision: string;
    document: PlanarDocument;
  };
  doc = saved.document;
  doc.scene_id = saved.scene_id;
  doc.revision = saved.revision;
  dirty = false;
  history.replaceState(
    null,
    "",
    `/multi-view.html?scene=${doc.scene_id}&revision=${doc.revision}`,
  );
  el("revision").textContent = `Revisão ${doc.revision.slice(0, 12)}`;
  message("Revisão imutável salva.");
  return saved;
}
function guarded(action: () => Promise<unknown> | unknown) {
  return async () => {
    try {
      await action();
    } catch (error) {
      message(error instanceof Error ? error.message : String(error), true);
    }
  };
}
el("save").onclick = guarded(saveRevision);
el("open-scene").onclick = () => el<HTMLInputElement>("scene-file").click();
el<HTMLInputElement>("scene-file").onchange = guarded(async () => {
  const file = el<HTMLInputElement>("scene-file").files?.[0];
  if (!file) return;
  const parsed = JSON.parse(await file.text()) as PlanarDocument;
  delete parsed.scene_id;
  delete parsed.revision;
  pictures.clear();
  installDocument(parsed);
  dirty = true;
  message("Cena lida. Carregue as imagens com os nomes image_key do JSON.");
});
el("open-images").onclick = () => el<HTMLInputElement>("image-files").click();
el<HTMLInputElement>("image-files").onchange = guarded(async () => {
  if (!doc) throw new Error("Abra o JSON da cena primeiro.");
  const files = [...(el<HTMLInputElement>("image-files").files ?? [])];
  for (const v of doc.views) {
    const file = files.find((f) => f.name === v.image_key);
    if (file) pictures.set(v.view_id, await loadPicture(file, file));
  }
  const first = pictures.get(doc.views[0].view_id);
  if (first)
    el("dimensions").textContent =
      `ERP ${first.pixels.width} × ${first.pixels.height} · range nativo`;
  schedule();
  message(`${pictures.size}/${doc.views.length} panoramas carregados.`);
});
el("download-scene").onclick = () => {
  if (!doc) return;
  const url = URL.createObjectURL(
    new Blob([JSON.stringify(doc, null, 2)], { type: "application/json" }),
  );
  const a = document.createElement("a");
  a.href = url;
  a.download = "planar-scene.json";
  a.click();
  URL.revokeObjectURL(url);
};
function parameters(operation: string) {
  return {
    operation,
    segmentation: el<HTMLSelectElement>("segmentation").value,
    matching: el<HTMLSelectElement>("matching").value,
    face_size: Number(el<HTMLSelectElement>("face-size").value),
    plane_threshold_m: Number(el<HTMLInputElement>("threshold").value),
    maximum_range_sigma_m: Number(el<HTMLInputElement>("sigma").value),
    accept_sam_terms: el<HTMLInputElement>("sam-terms").checked ? "yes" : "no",
    region_index:
      operation === "segment" && doc
        ? doc.regions.indexOf(region(activeSide)!)
        : -1,
  };
}
function prepareSegmentation() {
  const r = region(activeSide);
  if (!r?.mask.runs.length)
    throw new Error("Primeiro pinte uma estimativa da região na etapa 1.");
  const automatic = el<HTMLSelectElement>("roi-mode").value === "auto";
  if (el<HTMLSelectElement>("segmentation").value === "classifier")
    validateClassifierSeeds(r, automatic);
  if (!automatic && !r.roi)
    throw new Error(
      "Desenhe a ROI em Avançado ou escolha a área de busca automática.",
    );
  const roi = automatic
    ? suggestSegmentationRoi(
        denseMask(r.mask),
        shape(activeSide),
        supportMask(activeSide),
      )
    : r.roi!;
  checkpoint();
  r.roi = roi;
  r.state = "draft";
  r.interpretation = "seed";
  r.history.push({
    action: "prepare-segmentation",
    roi: automatic ? "automatic-8deg-or-15pct" : "manual",
    at: new Date().toISOString(),
  });
  changed();
}
async function execute(operation: string) {
  if (busy) return;
  if (!doc) throw new Error("Abra uma cena.");
  if (operation === "segment") prepareSegmentation();
  busy = true;
  document
    .querySelectorAll<HTMLButtonElement | HTMLInputElement | HTMLSelectElement>(
      ".planar-sidebar button,.planar-inspector button:not(#cancel),.planar-inspector input,.planar-inspector select,.planar-toolbar select,.canvas-controls button,.canvas-controls input,.canvas-controls select",
    )
    .forEach((control) => (control.disabled = true));
  el("cancel").hidden = false;
  el("pipeline").classList.add("run-busy");
  try {
    if (dirty || !doc.revision) await saveRevision();
    let run = (await (
      await request(
        `/api/v1/planar/scenes/${doc.scene_id}/${doc.revision}/runs`,
        jsonOptions(parameters(operation)),
      )
    ).json()) as Run;
    activeRun = run;
    while (["created", "running"].includes(run.status)) {
      message(
        `${run.status} · ${run.id} · processando a revisão ${doc.revision!.slice(0, 8)}`,
      );
      await new Promise((resolve) => setTimeout(resolve, 500));
      run = (await (await request(`/api/v1/runs/${run.id}`)).json()) as Run;
      activeRun = run;
    }
    if (run.status !== "completed") throw new Error(run.error ?? run.status);
    await openRun(run);
    if (operation === "segment" && result && doc) {
      const segmented = (result as Result).regions.find(
        (r) => r.plane_id === planeId && r.view_id === viewId(activeSide),
      );
      if (segmented) {
        checkpoint();
        const index = doc.regions.findIndex(
          (r) =>
            r.plane_id === segmented.plane_id &&
            r.view_id === segmented.view_id,
        );
        doc.regions[index] = structuredClone(segmented);
        changed();
        showResults(false);
        el("review-hint").textContent =
          "Segmentação aplicada à imagem. Corrija se necessário e clique em Aceitar máscara.";
        el("step-segment").classList.add("step-done");
        el("step-review").scrollIntoView({
          block: "nearest",
          behavior: "smooth",
        });
      }
    }
    await loadHistory();
    message(
      operation === "segment"
        ? "Segmentação pronta na imagem. Revise e confirme na etapa 3."
        : "Plano estimado. Revise as propostas nas outras vistas antes do range final.",
    );
  } finally {
    busy = false;
    el("pipeline").classList.remove("run-busy");
    el("cancel").hidden = true;
    document
      .querySelectorAll<
        HTMLButtonElement | HTMLInputElement | HTMLSelectElement
      >(
        ".planar-sidebar button,.planar-inspector button:not(#cancel),.planar-inspector input,.planar-inspector select,.planar-toolbar select,.canvas-controls button,.canvas-controls input,.canvas-controls select",
      )
      .forEach((control) => (control.disabled = false));
    schedule();
  }
}
el("segment").onclick = guarded(() => execute("segment"));
el("reconstruct").onclick = guarded(() => execute("reconstruct"));
el("cancel").onclick = guarded(async () => {
  if (activeRun)
    await request(`/api/v1/runs/${activeRun.id}/cancel`, { method: "POST" });
});
function artifactUrl(run: Run, artifact: Artifact) {
  return `/api/v1/runs/${run.id}/artifacts/${artifact.id}`;
}
async function openRun(run: Run) {
  activeRun = run;
  const artifact = run.artifacts.find((a) => a.id === "planar-result");
  if (!artifact) return;
  result = (await (await request(artifactUrl(run, artifact))).json()) as Result;
  el("gallery").innerHTML =
    `<figure><canvas id="support-cloud" width="700" height="380" style="width:100%;background:#121b12;touch-action:none" aria-label="Apoios triangulados em world; arraste para girar"></canvas><figcaption>Apoios triangulados em world · arraste para girar · revisão ${text(run.planar_scene?.revision?.slice(0, 12))}</figcaption></figure>` +
    run.artifacts
      .filter((a) => a.mime_type.startsWith("image/"))
      .map(
        (a) =>
          `<figure><img src="${artifactUrl(run, a)}" alt="${text(a.label)}"/><figcaption>${text(a.label)}</figcaption></figure>`,
      )
      .join("") +
    run.artifacts
      .filter((a) => a.type === "archive")
      .map(
        (a) => `<a href="${artifactUrl(run, a)}" download>${text(a.label)}</a>`,
      )
      .join("");
  drawSupportCloud(result);
  el("pipeline").innerHTML = run.events
    .map(
      (e) =>
        `<span class="${e.state === "completed" ? "done" : ""}">${text(e.message)}</span>`,
    )
    .join("");
  renderEvidence();
  schedule();
}
function renderEvidence() {
  const panel = el("evidence");
  const proposals = el("proposal-actions");
  proposals.replaceChildren();
  if (!result) {
    panel.textContent =
      "Execute o pipeline para inspecionar plano, apoios e propostas.";
    return;
  }
  const p = result.planes.find((p) => p.plane_id === planeId),
    tracks = result.tracks.filter((t) => t.plane_id === planeId),
    d = p?.diagnostics ?? {};
  panel.innerHTML = `<strong>${text(p?.state ?? "segmentação pronta para revisão")}</strong><dl><dt>Tracks aceitos</dt><dd>${tracks.filter((t) => !t.reason).length}/${tracks.length}</dd><dt>Normal world</dt><dd>${p?.normal_world?.map((x) => x.toFixed(3)).join(", ") ?? "—"}</dd><dt>Offset (m)</dt><dd>${p?.offset_m?.toFixed(4) ?? "—"}</dd><dt>RMSE fitting (m)</dt><dd>${typeof d.fit_rmse_m === "number" ? d.fit_rmse_m.toPrecision(3) : "—"}</dd><dt>RMSE reservado (m)</dt><dd>${typeof d.heldout_rmse_m === "number" ? d.heldout_rmse_m.toPrecision(3) : "—"}</dd></dl>${d.reason ? `<p>${text(d.reason)}</p>` : ""}`;
  result.proposals
    .filter((r) => r.plane_id === planeId)
    .forEach((r, i) => {
      const b = document.createElement("button");
      b.textContent = `Revisar proposta em ${r.view_id}`;
      b.onclick = () => {
        if (!doc) return;
        checkpoint();
        const existing = doc.regions.findIndex(
          (x) => x.plane_id === r.plane_id && x.view_id === r.view_id,
        );
        if (existing >= 0) doc.regions[existing] = structuredClone(r);
        else doc.regions.push(structuredClone(r));
        el<HTMLSelectElement>("view-b").value = r.view_id;
        activeSide = "b";
        message(
          `Proposta ${i + 1} carregada. Corrija e aceite a máscara em B.`,
        );
        changed();
      };
      proposals.append(b);
    });
  result.segmentation_alternatives
    .filter((a) => a.plane_id === planeId)
    .forEach((alternative) => {
      alternative.masks.forEach((mask, i) => {
        const b = document.createElement("button");
        b.textContent = `${alternative.view_id} · alternativa ${i + 1}`;
        b.onclick = () => {
          if (!doc) return;
          checkpoint();
          const proposed = result!.regions.find(
            (r) => r.plane_id === planeId && r.view_id === alternative.view_id,
          )!;
          const r = {
            ...structuredClone(proposed),
            mask,
            state: "draft" as const,
            interpretation: "final" as const,
          };
          const index = doc.regions.findIndex(
            (x) => x.plane_id === planeId && x.view_id === r.view_id,
          );
          if (index < 0) doc.regions.push(r);
          else doc.regions[index] = r;
          activeSide = "b";
          el<HTMLSelectElement>("view-b").value = r.view_id;
          changed();
        };
        panel.append(b);
      });
    });
  const metrics = document.createElement("p");
  metrics.textContent = result.ranges
    .map(
      (r) =>
        `${r.view_id}: ${r.valid_pixels} pixels de plano, ${r.conflict_pixels} conflitos`,
    )
    .join(" · ");
  panel.append(metrics);
}
function drawSupportCloud(value: Result) {
  const canvas = el<HTMLCanvasElement>("support-cloud"),
    ctx = canvas.getContext("2d")!,
    points = value.tracks
      .filter((t) => t.point_world)
      .map((t) => ({ p: t.point_world!, id: t.plane_id }));
  if (!points.length) {
    ctx.fillStyle = "#c4d0b7";
    ctx.fillText("Nenhum apoio 3D aceito nesta execução", 24, 40);
    return;
  }
  const center = [0, 1, 2].map(
      (i) => points.reduce((sum, p) => sum + p.p[i], 0) / points.length,
    ),
    radius = Math.max(
      0.01,
      ...points.map((p) => Math.hypot(...p.p.map((v, i) => v - center[i]))),
    );
  let angle = 0.5,
    last: number | null = null;
  const draw = () => {
    ctx.fillStyle = "#121b12";
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    ctx.strokeStyle = "#39472e";
    ctx.strokeRect(30, 30, 640, 320);
    for (const p of points) {
      const x = p.p[0] - center[0],
        y = p.p[1] - center[1],
        z = p.p[2] - center[2],
        px = 350 + ((x * Math.cos(angle) + z * Math.sin(angle)) / radius) * 250,
        py =
          190 -
          ((y * 0.8 + (-x * Math.sin(angle) + z * Math.cos(angle)) * 0.3) /
            radius) *
            140;
      ctx.beginPath();
      ctx.arc(px, py, 3, 0, Math.PI * 2);
      ctx.fillStyle = p.id === planeId ? "#ffc857" : "#b7cba0";
      ctx.fill();
    }
    ctx.fillStyle = "#b8c9a8";
    ctx.font = "12px system-ui";
    ctx.fillText(
      `${points.length} pontos de apoio · extensão ${radius.toFixed(2)} m · geometria world`,
      42,
      canvas.height - 15,
    );
  };
  canvas.onpointerdown = (e) => {
    last = e.clientX;
    canvas.setPointerCapture(e.pointerId);
  };
  canvas.onpointermove = (e) => {
    if (last === null) return;
    angle += (e.clientX - last) * 0.01;
    last = e.clientX;
    draw();
  };
  canvas.onpointerup = () => {
    last = null;
  };
  canvas.onpointercancel = () => {
    last = null;
  };
  draw();
}
async function loadHistory() {
  const value = (await (await request("/api/v1/runs?limit=50")).json()) as {
    runs: Run[];
  };
  const host = el("runs");
  host.replaceChildren();
  for (const run of value.runs.filter(
    (r) => r.planar_scene?.scene_id === doc?.scene_id,
  )) {
    const b = document.createElement("button");
    b.className = "history-row";
    b.textContent = `${run.status} · ${run.id.slice(-8)}`;
    b.onclick = guarded(() => openRun(run));
    host.append(b);
  }
}
function showResults(visible: boolean) {
  el("results").classList.toggle("visible", visible);
  el("edit-workspace").style.display = visible ? "none" : "grid";
  el("erp-strip").style.display = visible ? "none" : "grid";
  el("result-tab").classList.toggle("active", visible);
  el("edit-tab").classList.toggle("active", !visible);
  schedule();
}
el("result-tab").onclick = () => showResults(true);
el("edit-tab").onclick = () => showResults(false);
window.addEventListener("resize", schedule);
schedule();
async function loadDefaultRegisteredScene() {
  const sampleId = "poly-haven-studio-known-pose-v1";
  const sceneResponse = await request(
    `/api/v1/samples/${sampleId}/planar-scene`,
  );
  const document = (await sceneResponse.json()) as PlanarDocument;
  installDocument(document);
  for (const [index, view] of doc!.views.entries()) {
    const side = index === 0 ? "a" : "b";
    const response = await request(`/api/v1/samples/${sampleId}/${side}.png`);
    const blob = await response.blob();
    const file = new File([blob], view.image_key, {
      type: blob.type || "image/png",
    });
    pictures.set(view.view_id, await loadPicture(file, file));
  }
  const first = pictures.get(doc!.views[0].view_id)!;
  el("dimensions").textContent =
    `ERP ${first.pixels.width} × ${first.pixels.height} · pose conhecida`;
  el("revision").textContent = "Exemplo CC0 · ainda não salvo";
  dirty = true;
  message(
    "Par CC0 com pose métrica carregado. Marque um plano ou troque a cena.",
  );
  schedule();
}
void guarded(async () => {
  const runtime = (await (await request("/api/v1/runtime")).json()) as {
    panorai: {
      version: string;
    };
  };
  el("runtime").textContent =
    `PanorAi ${runtime.panorai.version} · Experimental`;
  const query = new URLSearchParams(location.search),
    id = query.get("scene"),
    revision = query.get("revision");
  if (id && revision) {
    const payload = (await (
      await request(`/api/v1/planar/scenes/${id}/${revision}`)
    ).json()) as {
      document: PlanarDocument;
    };
    installDocument(payload.document);
    for (let i = 0; i < doc!.views.length; i++) {
      const response = await request(
        `/api/v1/planar/scenes/${id}/${revision}/images/${i}`,
      );
      pictures.set(
        doc!.views[i].view_id,
        await loadPicture(await response.blob()),
      );
    }
    dirty = false;
    el("revision").textContent = `Revisão ${revision.slice(0, 12)}`;
    await loadHistory();
    message(
      "Cena pronta. Pinte uma estimativa na etapa 1 e clique em Segmentar região.",
    );
    schedule();
  } else {
    await loadDefaultRegisteredScene();
  }
})();
