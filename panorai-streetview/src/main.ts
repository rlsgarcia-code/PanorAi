import "./styles.css";

import {
  addDenseAnnotation,
  addManualCorrespondence,
  addObjectRegion,
  calibrateFloor,
  createProject,
  createSceneManifest,
  densePreviewImageUrl,
  denseSceneMeshUrl,
  listProjects,
  loadFloorCalibration,
  loadCapabilities,
  loadDensePreview,
  loadDenseScene,
  loadGlobalMap,
  loadPairRegistration,
  loadPlan,
  loadRegistrationHistory,
  loadSetup,
  loadStitching,
  promoteStitching,
  measureFloorPoint,
  measureVertical,
  reestimatePairRegistration,
  reviewMatch,
  runGlobalMap,
  runDensePreview,
  runDenseScene,
  runPairRegistration,
  runStitching,
  stitchingImageUrl,
} from "./api.ts";
import { createMutationQueue, denseAnchorStatus, denseSceneGuidance, erpAngularMeasurement, essentialControlState, imageRequestIsCurrent, invalidatePendingAction, isMetricMapScale, mergeCaptureGroupMetadata, metrologyCalibrationIsUsable, pairAssessment, pairRequestIsCurrent, projectRequestIsCurrent, replaceActiveRecord, replaceCachedRecord, sourcePixelPoints, sphericalPairCandidates, stitchingControlState, streetViewFlow, workflowCopy } from "./model.ts";
import type {
  BuildPlan,
  Capabilities,
  DenseAnnotation,
  DensePreviewRecord,
  DenseSceneRecord,
  FloorCalibrationRecord,
  FloorMeasurement,
  GlobalMapRecord,
  ImagePairRecord,
  ImageRecord,
  MatchRecord,
  MetrologyLine,
  PairRegistration,
  ProjectRecord,
  QualityGate,
  RegistrationHistory,
  RegionMode,
  SceneSetup,
  StitchingRecord,
  StreetViewFlowStageId,
  VerticalMeasurement,
  Workflow,
} from "./model.ts";

const root = document.querySelector<HTMLDivElement>("#app");
if (!root) throw new Error("Missing application root");

root.innerHTML = `
  <div class="app-shell">
    <header class="topbar">
      <div class="brand"><span class="orbit" aria-hidden="true"><i></i></span><div><strong>PanorAi</strong><b>Street View</b></div></div>
      <div class="project-crumb"><span>Ambiente local</span><i>/</i><strong id="crumb">Novo projeto</strong></div>
      <div class="runtime" id="runtime"><i></i><span>conectando ao engine</span></div>
      <button class="primary compact" id="newProject">Novo projeto</button>
    </header>
    <main class="workspace">
      <aside class="project-rail flow-rail">
        <div class="flow-rail-head"><small>Fluxo do ambiente</small><strong>Construir Street View</strong><span id="flowSummary">Comece pelo upload</span></div>
        <nav id="flowSteps" class="flow-steps" aria-label="Etapas do fluxo"></nav>
        <details class="environment-switcher">
          <summary><span>Trocar ambiente</span><small id="projectCount">0</small></summary>
          <div id="projectList" class="project-list"></div>
        </details>
        <div class="rail-foot"><span>Dados locais</span><small>imagens e evidências permanecem fora do pacote PanorAi</small></div>
      </aside>
      <section class="stage-column">
        <div class="stage-toolbar">
          <div class="pair-picker">
            <span>Par ativo</span>
            <select id="pairSelect" aria-label="Par de panoramas ativo"></select>
            <b id="pairQuality">—</b>
          </div>
          <div class="view-modes" aria-label="Modo de comparação">
            <button data-view="compare" class="active">Lado a lado</button>
            <button data-view="overlay">Sobrepor</button>
            <button data-view="blink">Piscar</button>
          </div>
          <div class="evidence-modes" aria-label="Evidência geométrica">
            <button data-evidence="detections">Detecções</button>
            <button data-evidence="matches">Matches</button>
            <button data-evidence="inliers">Inliers</button>
          </div>
          <div class="stage-meta"><span id="imageName">Nenhum par</span><small id="imageMeta">—</small></div>
          <button class="quiet" id="previousImage" aria-label="Par anterior">←</button>
          <button class="quiet" id="nextImage" aria-label="Próximo par">→</button>
        </div>
        <div class="visual-stage" id="visualStage">
          <div class="measurement-readout" id="measurementReadout" hidden><span><small>Medição esférica</small><strong id="measurementValue">—</strong><b id="measurementPixels">—</b></span><button id="clearMeasurement" aria-label="Limpar medição">×</button></div>
          <div class="empty-state" id="emptyState"><span class="empty-orbit"></span><strong>Escolha um projeto</strong><p>Suba panoramas ou grupos de fotos para definir uma rota de build.</p><button class="primary" id="emptyCreate">Criar projeto</button></div>
          <div class="pair-canvas" id="pairCanvas" data-view="compare" hidden>
            <div class="pair-pane reference" id="referencePane">
              <div class="pane-label"><b>Referência</b><span id="referenceName">—</span></div>
              <div class="pane-tools" data-pane-tools="reference" aria-label="Ferramentas do panorama de referência">
                <button data-pane-action="pan" title="Pan e tilt deste panorama">Pan/tilt</button>
                <button data-pane-action="measure" title="Medir neste panorama">Medir</button>
                <i></i>
                <button data-pane-action="zoom-out" aria-label="Reduzir zoom da referência">−</button>
                <output data-pane-zoom="reference">100%</output>
                <button data-pane-action="zoom-in" aria-label="Aumentar zoom da referência">+</button>
                <button data-pane-action="fit" title="Ajustar este panorama ao quadro">Ajustar</button>
              </div>
              <img id="sourceImage" alt="Panorama de referência" />
              <svg id="constraintLayer" aria-label="Camada de correções manuais" hidden></svg>
            </div>
            <div class="pair-pane source" id="targetPane">
              <div class="pane-label"><b>Vizinha</b><span id="targetName">—</span></div>
              <div class="pane-tools" data-pane-tools="source" aria-label="Ferramentas do panorama vizinho">
                <button data-pane-action="pan" title="Pan e tilt deste panorama">Pan/tilt</button>
                <button data-pane-action="measure" title="Medir neste panorama">Medir</button>
                <i></i>
                <button data-pane-action="zoom-out" aria-label="Reduzir zoom do vizinho">−</button>
                <output data-pane-zoom="source">100%</output>
                <button data-pane-action="zoom-in" aria-label="Aumentar zoom do vizinho">+</button>
                <button data-pane-action="fit" title="Ajustar este panorama ao quadro">Ajustar</button>
              </div>
              <img id="targetImage" alt="Panorama vizinho" />
            </div>
            <svg id="evidenceLayer" class="evidence-layer" aria-label="Detecções e correspondências" hidden></svg>
            <label class="overlay-control" id="overlayControl" hidden>Referência <input id="overlayOpacity" type="range" min="10" max="90" value="50" /> Vizinha</label>
          </div>
          <div class="stage-label" id="stageLabel" hidden>Diagnóstico visual do registro</div>
        </div>
        <div class="pair-footer">
          <div class="pair-strip" id="pairStrip"></div>
          <div class="correction-tools" id="toolGroup">
            <div><small>Evidência humana</small><span id="toolHint">Revise matches ou adicione uma observação.</span></div>
            <button data-tool="manual_match">Novo match</button>
            <button data-tool="object_box">Objeto / BB</button>
            <button data-tool="dense_include">Região densa</button>
          </div>
        </div>
      </section>
      <aside class="inspector">
        <div class="inspector-head"><div><small id="flowStageEyebrow">Etapa ativa</small><h1 id="flowStageTitle">Escolha um ambiente</h1></div><span class="route-badge" id="workflowBadge">—</span></div>
        <p class="workflow-detail" id="flowStageDetail">O fluxo orienta cada decisão sem esconder as ferramentas especializadas.</p>
        <div class="tabs work-tabs" aria-label="Ferramentas da etapa"><button class="active" data-tab="setup">Visão geral</button><button data-tab="stitching">Costura ERP</button><button data-tab="map">Mapa 2D</button><button data-tab="registration">Pares & Essential</button><button data-tab="metrology">Piso & métrica</button><button data-tab="objects">Objetos & BB</button><button data-tab="dense">Estrutura 3D</button><button data-tab="navigation">Teste</button><button data-tab="build">Publicação</button><button data-tab="library">Biblioteca</button></div>
        <div class="tab-panel active" id="setupPanel">
          <div class="setup-command"><div><small>Preparação da cena</small><strong id="setupHeadline">Lendo o projeto</strong><p id="setupGuidance">O setup combina estado da aplicação e pré-requisitos públicos PanorAi.</p></div><span id="setupProgress">0%</span></div>
          <div class="setup-summary"><div><small>Pares executados</small><strong id="setupRegistered">—</strong></div><div><small>Pares aceitos</small><strong id="setupAccepted">—</strong></div><div><small>Componente</small><strong id="setupComponent">—</strong></div><div><small>Escala</small><strong id="setupScale">—</strong></div></div>
          <div class="setup-next"><div><small>Próxima decisão</small><strong id="setupActionTitle">—</strong><p id="setupActionDetail">—</p></div><button class="primary" id="setupAction">Abrir</button></div>
          <div class="setup-route" id="setupRoute"></div>
          <div class="setup-pairs-head"><small>Fila de pares</small><span>status por evidência</span></div>
          <div class="setup-pairs" id="setupPairs"></div>
        </div>
        <div class="tab-panel" id="stitchingPanel">
          <div class="stitch-command"><div><small>Fotos convencionais → ERP</small><strong id="stitchHealth">Costura ainda não executada</strong><p id="stitchGuidance">Captura pinhole central rotacional; parallax forte e overlap desconectado são recusados.</p></div><button class="primary" id="runStitching">Executar costura</button></div>
          <div class="stitch-group-list" id="stitchGroups"></div>
          <div class="stitch-source-list" id="stitchSources"></div>
          <div class="stitch-output"><img id="stitchImage" alt="ERP parcial produzido pelo PanoramaStitcher" hidden/><div id="stitchEmpty">O ERP parcial e sua validade aparecerão aqui.</div></div>
          <dl class="stitch-metrics"><div><dt>Cobertura válida</dt><dd id="stitchCoverage">—</dd></div><div><dt>Overlap ERP</dt><dd id="stitchOverlap">—</dd></div><div><dt>Pares aceitos</dt><dd id="stitchPairs">—</dd></div><div><dt>FOV estimados</dt><dd id="stitchEstimatedFov">—</dd></div></dl>
          <div class="stitch-result-foot"><span id="stitchResultDetail">Aguardando execução.</span><a id="downloadStitching" download>Baixar ERP PNG</a></div>
          <div class="stitch-pair-list" id="stitchPairList"></div>
          <div class="stitch-promote"><div><small>Próxima etapa</small><strong id="stitchPromoteHealth">São necessários dois ERPs aceitos</strong><p id="stitchPromoteDetail">A promoção cria pares candidatos sem inventar poses.</p></div><button class="primary" id="promoteStitching" disabled>Promover para registro</button></div>
        </div>
        <div class="tab-panel" id="mapPanel">
          <div class="map-command"><div><small>Registro multivista</small><strong id="mapHealth">Mapa ainda não executado</strong><p id="mapGuidance">Aceite arestas que conectem ao menos três panoramas.</p></div><button class="primary" id="runGlobalMap" disabled>Executar mapa</button></div>
          <div class="map-viewport"><svg id="globalMapCanvas" viewBox="0 0 400 250" role="img" aria-label="Mapa global de câmeras e tracks"></svg><span id="mapScaleBadge">escala —</span></div>
          <dl class="map-metrics"><div><dt>Câmeras</dt><dd id="mapCameraCount">—</dd></div><div><dt>Arestas</dt><dd id="mapEdgeCount">—</dd></div><div><dt>Tracks</dt><dd id="mapTrackCount">—</dd></div><div><dt>Reproj. P90</dt><dd id="mapReprojection">—</dd></div></dl>
          <div class="map-recommendation"><div><small>Próxima correção</small><strong id="mapActionTitle">—</strong><p id="mapActionDetail">—</p></div><button id="mapAction">Abrir par</button></div>
          <div class="map-diagnostics" id="mapDiagnostics"></div>
          <div class="map-edge-list" id="mapEdgeList"></div>
        </div>
        <div class="tab-panel" id="registrationPanel">
          <div class="essential-command">
            <div><small>Geometria do par</small><strong id="registrationHealth">Pose não estimada</strong><p id="registrationGuidance">Execute features e Essential para produzir evidência revisável.</p></div>
            <button class="primary" id="runEssential">Rodar Essential</button>
          </div>
          <div class="registration-control-row"><label>Regiões de objeto<select id="regionMode"><option value="all">Não usar</option><option value="prioritize">Priorizar</option><option value="restrict">Restringir</option></select></label><button class="quiet" id="reestimateEssential" disabled>Reestimar com revisão</button></div>
          <div class="revision-delta" id="revisionDelta" hidden></div>
          <dl class="pair-metrics evidence-metrics"><div><dt>Features</dt><dd id="featureValue">—</dd></div><div><dt>Matches</dt><dd id="matchValue">—</dd></div><div><dt>Inliers</dt><dd id="inlierValue">—</dd></div><div><dt>Qualidade</dt><dd id="qualityValue">—</dd></div><div><dt>Paralaxe</dt><dd id="parallaxValue">—</dd></div><div><dt>Escala t</dt><dd>livre</dd></div></dl>
          <div class="edit-guidance" id="editGuidance"><div><small>Leitura do engine</small><strong>Rode o Essential para receber uma ação sugerida.</strong><p>Os gates abaixo usam o perfil público PanorAi.</p></div><button id="editGuidanceAction" hidden>Aplicar</button></div>
          <div class="quality-gates" id="qualityGates"></div>
          <div class="revision-history"><div class="revision-history-head"><small>Revisões</small><span id="historyCount">0 snapshots</span></div><div id="historyList"></div></div>
          <div class="match-toolbar"><button data-match-filter="all" class="active">Todos</button><button data-match-filter="inlier">Inliers</button><button data-match-filter="review">Revisar</button><span id="matchSelection">Selecione uma linha no canvas</span></div>
          <div class="match-list" id="matchList"><div class="constraints-empty"><strong>Sem execução</strong><span>Rode o Essential para visualizar detecções, matches e resíduos.</span></div></div>
        </div>
        <div class="tab-panel" id="metrologyPanel">
          <div class="metrology-intro"><small>Escala em uma vista</small><strong>Alinhar piso e medir em metros</strong><p>Marque duas famílias de direções paralelas sobre o piso e indique um ponto abaixo do horizonte. O PanorAi estima a rotação de nível; a altura conhecida da câmera fixa a escala. Isto não registra panoramas nem estima translação.</p></div>
          <div class="manual-target"><span>Panorama ativo</span><div><button type="button" class="active" data-manual-side="reference">Referência</button><button type="button" data-manual-side="source">Vizinha</button></div></div>
          <label class="metrology-height">Altura do centro da câmera <span><input id="cameraHeight" type="number" min="0.1" step="0.01" value="1.60"/><b>m</b></span></label>
          <div class="metrology-steps">
            <section><span>1</span><div><strong>Duas direções do piso</strong><p>Trace pelo menos duas observações por família.</p><div class="metrology-tools"><button data-tool="floor_a">Família A</button><button data-tool="floor_b">Família B</button></div><small id="floorLineCount">A 0/2 · B 0/2</small></div></section>
            <section><span>2</span><div><strong>Qual lado é o piso?</strong><p>Clique uma região visível do piso para resolver o sinal do plano.</p><button data-tool="floor_hint">Indicar piso</button><small id="floorHintState">Ainda não indicado</small></div></section>
            <section><span>3</span><div><strong>Calibrar e revisar</strong><p>Uma nova execução cria uma revisão auditável.</p><button class="primary" id="runFloorCalibration" disabled>Calibrar piso</button><button class="quiet" id="clearFloorDraft">Limpar marcações</button></div></section>
          </div>
          <div class="metrology-diagnostics" id="metrologyDiagnostics"><strong>Sem calibração</strong><p>Conclua as marcações para habilitar medições métricas.</p></div>
          <div class="metric-tools"><button data-tool="floor_measure" disabled>Medir ponto no piso</button><button data-tool="vertical_measure" disabled>Medir altura vertical</button></div>
          <div class="metric-result" id="metricResult"><small>Resultado</small><strong>—</strong><p>Selecione uma ferramenta métrica após calibrar.</p></div>
        </div>
        <div class="tab-panel" id="objectsPanel">
          <div class="object-intro"><small>Objeto compartilhado</small><strong>Replicar uma região entre vistas</strong><p>Desenhe o mesmo objeto nas duas imagens. Os features internos podem ser priorizados ou usados como restrição.</p></div>
          <div class="object-form"><label>ID do objeto<input id="objectId" value="objeto-a" /></label><label>Nome<input id="objectLabel" value="Objeto A" /></label><div class="manual-target"><span>Desenhar em</span><div><button type="button" class="active" data-manual-side="reference">Referência</button><button type="button" data-manual-side="source">Vizinha</button></div></div><button class="primary wide" type="button" data-tool="object_box">Desenhar bounding box</button></div>
          <div class="object-list" id="objectList"></div>
          <div class="object-match-list" id="objectMatchList"></div>
        </div>
        <div class="tab-panel" id="densePanel">
          <div class="dense-intro"><small>Cognição para o denso</small><strong>Validade, escala e superfície</strong><p>Incluir/Excluir altera o estéreo nas duas direções. Âncoras próximas a inliers viram SphericalRangePrior; o traço Superfície continua anotação visual até existir um prior público compatível.</p></div>
          <div class="dense-tools"><button data-tool="dense_include">Incluir região</button><button data-tool="dense_exclude">Excluir região</button><button data-tool="dense_surface">Superfície</button><button data-tool="dense_anchor">Âncora radial</button></div>
          <label>Rótulo<input id="denseLabel" value="estrutura" /></label><label>Range radial para âncora<input id="denseRange" type="number" min="0.01" step="0.1" value="2.0" /></label>
          <div class="manual-target"><span>Aplicar em</span><div><button type="button" class="active" data-manual-side="reference">Referência</button><button type="button" data-manual-side="source">Vizinha</button></div></div>
          <div class="dense-status"><i></i><span id="denseStatus">Nenhuma âncora ligada a um track.</span></div>
          <div class="dense-preview-command"><div><small>Preview do par</small><strong id="densePreviewHealth">Ainda não executado</strong><p id="densePreviewDetail">Usa a pose aceita e aplica Incluir/Excluir como validade explícita na referência.</p></div><button class="primary" id="runDensePreview" disabled>Rodar preview</button></div>
          <div class="dense-preview-view"><img id="densePreviewImage" alt="Ranges radiais referência e vizinha estimados pelo PanorAi" hidden/><div id="densePreviewEmpty">Os dois mapas direcionais de range aparecerão aqui.</div></div>
          <div class="dense-direction-labels"><span>Referência → vizinha</span><span>Vizinha → referência</span></div>
          <dl class="dense-preview-metrics"><div><dt>Válido A / B</dt><dd id="denseValidFraction">—</dd></div><div><dt>Máscara A / B</dt><dd id="denseMaskFraction">—</dd></div><div><dt>Range mediano A / B</dt><dd id="denseMedianRange">—</dd></div><div><dt>Confiança A / B</dt><dd id="denseMedianConfidence">—</dd></div></dl>
          <div class="dense-scene-command" data-state="map"><div><small>Cena multivista</small><strong id="denseSceneHealth">Lendo pré-requisitos</strong><p id="denseSceneDetail">Mapa global, escala métrica e três vistas são verificados separadamente.</p></div><button id="denseSceneAction">Abrir</button></div>
          <div class="dense-scene-gates"><span id="denseMapGate">Mapa —</span><span id="denseScaleGate">Escala —</span><span id="denseViewsGate">Vistas —</span></div>
          <div class="dense-scene-result" id="denseSceneResult" hidden>
            <svg id="denseSceneCanvas" viewBox="0 0 360 190" role="img" aria-label="Projeção da superfície densa reconstruída"></svg>
            <dl><div><dt>Ranges</dt><dd id="denseSceneRangeCount">—</dd></div><div><dt>Vértices</dt><dd id="denseSceneVertexCount">—</dd></div><div><dt>Faces</dt><dd id="denseSceneFaceCount">—</dd></div><div><dt>Cor</dt><dd id="denseSceneColorFraction">—</dd></div></dl>
            <div class="dense-scene-result-foot"><span id="denseSceneResultDetail">—</span><a id="downloadDenseScene" download>Baixar PLY</a></div>
          </div>
          <div class="constraint-list" id="denseList"></div>
        </div>
        <div class="tab-panel" id="navigationPanel">
          <div class="navigation-intro"><small>Teste antes de publicar</small><strong>Percorra o grafo como um visitante</strong><p>Use os vizinhos explícitos do panorama ativo. Medidas em metros ficam disponíveis somente nas imagens com piso calibrado e altura de câmera registrada.</p></div>
          <div class="navigation-current"><div><small>Panorama ativo</small><strong id="navigationCurrent">—</strong><span id="navigationMetricState">Métrica ainda não verificada</span></div><b id="navigationNodeCount">0 nós</b></div>
          <div class="navigation-neighbors-head"><small>Destinos conectados</small><span>clique para abrir o par</span></div>
          <div class="navigation-neighbors" id="navigationNeighbors"></div>
          <div class="navigation-actions"><button data-tool="pan">Pan / tilt</button><button data-tool="floor_measure" id="navigationMeasureFloor" disabled>Medir no piso</button><button data-tool="vertical_measure" id="navigationMeasureVertical" disabled>Medir altura</button></div>
          <div class="navigation-checklist"><span><i></i> Transição usa aresta explícita</span><span><i></i> Panorama preserva pixel-center</span><span id="navigationMetricCheck"><i></i> Escala métrica pendente</span></div>
        </div>
        <div class="tab-panel" id="buildPanel">
          <div class="publish-intro"><small>Publicação local</small><strong>Fechar uma versão navegável</strong><p>O manifesto só é criado quando o grafo e as poses atuais satisfazem os bloqueios da rota. A publicação preserva o projeto para reabertura.</p></div>
          <div class="next-action"><small>Verificação</small><p id="nextAction">Crie ou abra um projeto.</p></div>
          <div class="stage-list" id="stageList"></div>
          <button class="primary wide" id="buildButton" disabled>Build indisponível</button>
          <p class="action-feedback" id="actionFeedback"></p>
        </div>
        <div class="tab-panel" id="libraryPanel">
          <div class="library-intro"><small>Ambientes navegáveis</small><strong>Street Views publicados</strong><p>Abra uma versão publicada para testar a navegação ou retomar o projeto de origem.</p></div>
          <div class="published-list" id="publishedList"></div>
        </div>
      </aside>
    </main>
  </div>
  <dialog id="projectDialog">
    <form method="dialog" id="projectForm">
      <div class="dialog-head"><div><small>Novo ambiente</small><h2>Como chegam as imagens?</h2></div><button value="cancel" class="icon-button" aria-label="Fechar">×</button></div>
      <label class="field">Nome do projeto<input id="projectName" required placeholder="Corredor norte" /></label>
      <fieldset class="workflow-grid" id="workflowGrid"></fieldset>
      <label class="dropzone" id="dropzone"><input id="imageFiles" type="file" accept="image/png,image/jpeg,image/webp" multiple required /><strong>Solte imagens aqui</strong><span>PNG, JPEG ou WebP · até 160 MiB por arquivo</span><small id="fileSummary">Nenhum arquivo selecionado</small></label>
      <section class="pair-editor" id="pairEditor" hidden><div><small>Definir conexões</small><strong>Quais panoramas formam pares?</strong><p>Confirme somente vistas com conteúdo em comum. Nenhuma pose, vizinhança ou overlap é inferido pela ordem do upload.</p></div><div id="pairRows"></div></section>
      <section class="capture-group-editor" id="captureGroupEditor" hidden><div><small>Organizar tomadas</small><strong>Quais fotos pertencem à mesma rotação?</strong><p>Use o mesmo grupo nas fotos que formam um ERP. Cada grupo precisa de pelo menos duas imagens.</p></div><div id="captureGroupRows"></div></section>
      <details><summary>Metadados de câmera</summary><p>JSON por nome de arquivo. Agrupe fotos da mesma rotação com <code>capture_group</code>. Foto calibrada: <code>horizontal_fov_deg</code> ou <code>intrinsics.fx</code>. ERP registrada: <code>pose.rotation_world_to_panorama</code> + <code>pose.center_world</code>.</p><textarea id="metadataInput" spellcheck="false">{}</textarea></details>
      <div class="dialog-actions"><button value="cancel" class="quiet">Cancelar</button><button value="default" class="primary" id="createButton">Criar projeto</button></div>
      <p class="form-error" id="formError"></p>
    </form>
  </dialog>
`;

function element<T extends Element>(selector: string): T {
  const value = document.querySelector<T>(selector);
  if (!value) throw new Error(`Missing ${selector}`);
  return value;
}

function escapeHtml(value: string): string {
  return value.replace(/[&<>'"]/g, (character) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", "'": "&#39;", '"': "&quot;" })[character] ?? character);
}

const projectDialog = element<HTMLDialogElement>("#projectDialog");
const projectForm = element<HTMLFormElement>("#projectForm");
const captureGroupEditor = element<HTMLElement>("#captureGroupEditor");
const captureGroupRows = element<HTMLDivElement>("#captureGroupRows");
const pairEditor = element<HTMLElement>("#pairEditor");
const pairRows = element<HTMLDivElement>("#pairRows");
const projectList = element<HTMLDivElement>("#projectList");
const visualStage = element<HTMLDivElement>("#visualStage");
const pairCanvas = element<HTMLDivElement>("#pairCanvas");
const referencePane = element<HTMLDivElement>("#referencePane");
const targetPane = element<HTMLDivElement>("#targetPane");
const sourceImage = element<HTMLImageElement>("#sourceImage");
const targetImage = element<HTMLImageElement>("#targetImage");
const constraintLayer = element<SVGSVGElement>("#constraintLayer");
const evidenceLayer = element<SVGSVGElement>("#evidenceLayer");
const emptyState = element<HTMLDivElement>("#emptyState");
const stageList = element<HTMLDivElement>("#stageList");
const matchList = element<HTMLDivElement>("#matchList");
const objectList = element<HTMLDivElement>("#objectList");
const objectMatchList = element<HTMLDivElement>("#objectMatchList");
const denseList = element<HTMLDivElement>("#denseList");
const densePreviewImage = element<HTMLImageElement>("#densePreviewImage");
const buildButton = element<HTMLButtonElement>("#buildButton");
const fileInput = element<HTMLInputElement>("#imageFiles");
const workflowGrid = element<HTMLFieldSetElement>("#workflowGrid");
const pairSelect = element<HTMLSelectElement>("#pairSelect");
const pairStrip = element<HTMLDivElement>("#pairStrip");
const setupRoute = element<HTMLDivElement>("#setupRoute");
const setupPairs = element<HTMLDivElement>("#setupPairs");
const qualityGates = element<HTMLDivElement>("#qualityGates");
const historyList = element<HTMLDivElement>("#historyList");
const globalMapCanvas = element<SVGSVGElement>("#globalMapCanvas");
const mapDiagnostics = element<HTMLDivElement>("#mapDiagnostics");
const mapEdgeList = element<HTMLDivElement>("#mapEdgeList");
const measurementReadout = element<HTMLDivElement>("#measurementReadout");
const flowSteps = element<HTMLElement>("#flowSteps");
const publishedList = element<HTMLDivElement>("#publishedList");

let projects: ProjectRecord[] = [];
let capabilities: Capabilities | null = null;
let activeProject: ProjectRecord | null = null;
let activePlan: BuildPlan | null = null;
let activeRegistration: PairRegistration | null = null;
let activeSetup: SceneSetup | null = null;
let activeHistory: RegistrationHistory | null = null;
let activeGlobalMap: GlobalMapRecord | null = null;
let activeDensePreview: DensePreviewRecord | null = null;
let activeDenseScene: DenseSceneRecord | null = null;
let activeStitching: StitchingRecord | null = null;
let activeStitchGroupId: string | null = null;
let activePairIndex = 0;
let tool: "inspect" | "pan" | "measure" | "object_box" | "manual_match" | "dense_include" | "dense_exclude" | "dense_surface" | "dense_anchor" | "floor_a" | "floor_b" | "floor_hint" | "floor_measure" | "vertical_measure" = "inspect";
let viewMode: "compare" | "overlay" | "blink" = "compare";
let evidenceMode: "none" | "detections" | "matches" | "inliers" = "none";
let matchFilter: "all" | "inlier" | "review" = "all";
let selectedMatchIndex: number | null = null;
let activeObjectId: string | null = null;
let activeTab: "setup" | "stitching" | "map" | "registration" | "metrology" | "objects" | "dense" | "navigation" | "build" | "library" = "setup";
let activeFlowStage: StreetViewFlowStageId = "upload";
let manualSide: "reference" | "source" = "reference";
let drawingStart: { x: number; y: number } | null = null;
let pendingManualReference: [number, number] | null = null;
type PanoramaSide = "reference" | "source";
type PanoramaViewport = { zoom: number; panX: number; panY: number };
const freshViewport = (): PanoramaViewport => ({ zoom: 1, panX: 0, panY: 0 });
let viewports: Record<PanoramaSide, PanoramaViewport> = { reference: freshViewport(), source: freshViewport() };
let panGesture: { pointerId: number; side: PanoramaSide; clientX: number; clientY: number; panX: number; panY: number } | null = null;
let measurement: { imageId: string; side: "reference" | "source"; start: [number, number]; end: [number, number] } | null = null;
let measurementGesture: { pointerId: number; pane: HTMLDivElement } | null = null;
let activeMetrology: FloorCalibrationRecord | null = null;
let calibratedImageIds = new Set<string>();
let metrologyLines: MetrologyLine[] = [];
let floorHint: [number, number] | null = null;
let floorMeasurement: { xy: [number, number]; value: FloorMeasurement } | null = null;
let verticalMeasurement: { base: [number, number]; top: [number, number]; value: VerticalMeasurement } | null = null;
let pendingVerticalBase: [number, number] | null = null;
let metricError: string | null = null;
let metrologyError: string | null = null;
let metrologyDraftDirty = false;
let pairLoadSequence = 0;
let projectLoadSequence = 0;
let metrologyLoadSequence = 0;
let densePreviewSequence = 0;
let essentialSequence = 0;
let essentialOperationRunning = false;
let metrologyActionSequence = 0;
let stitchingSequence = 0;
let stitchingPromotionSequence = 0;
let stitchingOperationRunning = false;
let denseSceneSequence = 0;
let globalMapSequence = 0;
let manifestSequence = 0;
const enqueueEvidenceMutation = createMutationQueue();
const enqueueMatchReviewMutation = createMutationQueue();
const captureGroupAssignments = new Map<string, string>();
const selectedIntakePairKeys = new Set<string>();
let knownIntakePairKeys = new Set<string>();

const workflows: Workflow[] = ["spherical_registered", "spherical_unregistered", "perspective_calibrated", "perspective_uncalibrated"];
workflowGrid.innerHTML = workflows.map((workflow, index) => {
  const copy = workflowCopy(workflow);
  return `<label class="workflow-option"><input type="radio" name="workflow" value="${workflow}" ${index === 1 ? "checked" : ""}/><span><i>${String(index + 1).padStart(2, "0")}</i><strong>${copy.title}</strong><small>${copy.detail}</small><b>${copy.badge}</b></span></label>`;
}).join("");

function selectedWorkflow(): Workflow {
  return element<HTMLInputElement>('input[name="workflow"]:checked').value as Workflow;
}

function renderCaptureGroupEditor(): void {
  const files = Array.from(fileInput.files ?? []);
  const perspective = selectedWorkflow().startsWith("perspective_");
  captureGroupEditor.hidden = !perspective || !files.length;
  captureGroupRows.replaceChildren();
  if (captureGroupEditor.hidden) return;
  const liveNames = new Set(files.map((file) => file.name));
  Array.from(captureGroupAssignments.keys()).forEach((name) => {
    if (!liveNames.has(name)) captureGroupAssignments.delete(name);
  });
  files.forEach((file) => {
    const group = captureGroupAssignments.get(file.name) ?? "capture-1";
    captureGroupAssignments.set(file.name, group);
    const row = document.createElement("label");
    row.className = "capture-group-row";
    row.innerHTML = `<span><strong>${escapeHtml(file.name)}</strong><small>${(file.size / 1024 / 1024).toFixed(2)} MiB</small></span><input aria-label="Grupo de ${escapeHtml(file.name)}" required pattern="[A-Za-z0-9_-]+" value="${escapeHtml(group)}"/><b>ERP</b>`;
    const input = row.querySelector<HTMLInputElement>("input")!;
    input.oninput = () => captureGroupAssignments.set(file.name, input.value);
    captureGroupRows.append(row);
  });
}

function intakePairKey(reference: string, source: string): string {
  return `${reference}\u0000${source}`;
}

function renderPairEditor(): void {
  const filenames = Array.from(fileInput.files ?? []).map((file) => file.name);
  const spherical = selectedWorkflow().startsWith("spherical_");
  const candidates = sphericalPairCandidates(filenames);
  const liveKeys = new Set(candidates.map((pair) => intakePairKey(pair.reference_filename, pair.source_filename)));
  for (const key of Array.from(selectedIntakePairKeys)) {
    if (!liveKeys.has(key)) selectedIntakePairKeys.delete(key);
  }
  for (const key of liveKeys) {
    if (!knownIntakePairKeys.has(key)) selectedIntakePairKeys.add(key);
  }
  knownIntakePairKeys = liveKeys;
  pairEditor.hidden = !spherical || filenames.length < 2;
  pairRows.replaceChildren();
  if (pairEditor.hidden) return;
  candidates.forEach((pair) => {
    const key = intakePairKey(pair.reference_filename, pair.source_filename);
    const row = document.createElement("label");
    row.className = "pair-editor-row";
    row.innerHTML = `<input type="checkbox" ${selectedIntakePairKeys.has(key) ? "checked" : ""}/><span><strong>${escapeHtml(pair.reference_filename)}</strong><i>↔</i><strong>${escapeHtml(pair.source_filename)}</strong></span><b>candidato</b>`;
    const input = row.querySelector<HTMLInputElement>("input")!;
    input.onchange = () => input.checked ? selectedIntakePairKeys.add(key) : selectedIntakePairKeys.delete(key);
    pairRows.append(row);
  });
}

workflowGrid.onchange = () => {
  renderCaptureGroupEditor();
  renderPairEditor();
};

function imageUrl(project: ProjectRecord, image: ImageRecord): string {
  return `/api/v1/projects/${project.id}/images/${image.id}`;
}

function currentPair(): ImagePairRecord | null {
  return activeProject?.pairs[activePairIndex] ?? null;
}

function imageForPair(pair: ImagePairRecord, side: "reference" | "source"): ImageRecord | null {
  const imageId = side === "reference" ? pair.reference_image_id : pair.source_image_id;
  return activeProject?.images.find((image) => image.id === imageId) ?? null;
}

function currentEditImage(): { record: ImageRecord; element: HTMLImageElement; pane: HTMLDivElement } | null {
  const pair = currentPair();
  if (!pair) return null;
  const record = imageForPair(pair, manualSide);
  if (!record) return null;
  return manualSide === "reference"
    ? { record, element: sourceImage, pane: referencePane }
    : { record, element: targetImage, pane: targetPane };
}

function selectTab(target: typeof activeTab): void {
  activeTab = target;
  document.querySelectorAll<HTMLButtonElement>("[data-tab]").forEach((item) => item.classList.toggle("active", item.dataset.tab === target));
  element("#setupPanel").classList.toggle("active", target === "setup");
  element("#stitchingPanel").classList.toggle("active", target === "stitching");
  element("#mapPanel").classList.toggle("active", target === "map");
  element("#registrationPanel").classList.toggle("active", target === "registration");
  element("#metrologyPanel").classList.toggle("active", target === "metrology");
  element("#objectsPanel").classList.toggle("active", target === "objects");
  element("#densePanel").classList.toggle("active", target === "dense");
  element("#navigationPanel").classList.toggle("active", target === "navigation");
  element("#buildPanel").classList.toggle("active", target === "build");
  element("#libraryPanel").classList.toggle("active", target === "library");
  if (target === "map") renderGlobalMap();
  if (target === "stitching") renderStitching();
  if (target === "metrology") renderMetrology();
  if (target === "navigation") renderNavigationTest();
  if (target === "library") renderPublishedEnvironments();
  window.requestAnimationFrame(() => renderEvidence());
}

function flowState() {
  const summary = activeSetup?.summary;
  const mapCurrent = Boolean(activeGlobalMap?.success && !activeGlobalMap.stale);
  return streetViewFlow({
    hasProject: Boolean(activeProject),
    imageCount: activeProject?.images.length ?? 0,
    calibratedImageCount: calibratedImageIds.size,
    usesProvidedPoses: activeProject?.workflow === "spherical_registered",
    registeredPairCount: summary?.registered_pair_count ?? 0,
    acceptedPairCount: summary?.accepted_pair_count ?? 0,
    mapCurrent,
    denseAccepted: Boolean(activeDenseScene?.accepted && !activeDenseScene.stale),
    navigationReady: Boolean(activeProject && activeProject.pairs.length && (activeProject.workflow === "spherical_registered" || mapCurrent)),
    canPublish: Boolean(activePlan?.can_build_now || (mapCurrent && activeProject?.workflow === "spherical_unregistered")),
    published: activeProject?.status === "built",
    publishedCount: projects.filter((project) => project.status === "built").length,
  });
}

function renderContextTabs(): void {
  const tabsByStage: Record<StreetViewFlowStageId, typeof activeTab[]> = {
    upload: [],
    "single-view": isPerspectiveProject() ? ["stitching"] : ["metrology"],
    "multi-view": isPerspectiveProject() ? ["stitching"] : ["registration", "map"],
    adjust: ["registration", "objects", "map"],
    dense: ["dense"],
    navigate: ["navigation", "metrology"],
    review: ["setup", "metrology", "registration", "objects", "map", "dense"],
    publish: ["build"],
    library: ["library"],
  };
  const visible = new Set(tabsByStage[activeFlowStage]);
  document.querySelectorAll<HTMLButtonElement>("[data-tab]").forEach((button) => {
    button.hidden = !visible.has(button.dataset.tab as typeof activeTab);
  });
}

function renderFlow(): void {
  const stages = flowState();
  const current = stages.find((stage) => stage.id === activeFlowStage) ?? stages[0];
  flowSteps.replaceChildren();
  stages.forEach((stage) => {
    const button = document.createElement("button");
    button.className = `flow-step state-${stage.state} ${stage.id === activeFlowStage ? "active" : ""}`;
    button.dataset.flowStage = stage.id;
    button.title = `${stage.detail} ${stage.reason}`;
    button.innerHTML = `<span>${String(stage.index).padStart(2, "0")}</span><i></i><div><strong>${escapeHtml(stage.label)}</strong><small>${escapeHtml(stage.reason)}</small></div><b>${stage.state === "done" ? "feito" : stage.state === "ready" ? "pronto" : stage.state === "optional" ? "opcional" : stage.state === "active" ? "agora" : "bloqueado"}</b>`;
    button.onclick = () => selectFlowStage(stage.id);
    flowSteps.append(button);
  });
  element<HTMLElement>("#flowStageEyebrow").textContent = `Etapa ${String(current.index).padStart(2, "0")} de 09`;
  element<HTMLElement>("#flowStageTitle").textContent = current.label;
  element<HTMLElement>("#flowStageDetail").textContent = current.detail;
  element<HTMLElement>("#flowSummary").textContent = current.reason;
  renderContextTabs();
}

function selectFlowStage(stage: StreetViewFlowStageId): void {
  activeFlowStage = stage;
  renderFlow();
  if (stage === "upload") {
    openDialog();
    return;
  }
  const target: Record<Exclude<StreetViewFlowStageId, "upload">, typeof activeTab> = {
    "single-view": isPerspectiveProject() ? "stitching" : "metrology",
    "multi-view": isPerspectiveProject() ? "stitching" : (activeSetup?.summary.registered_pair_count ? "map" : "registration"),
    adjust: "registration",
    dense: "dense",
    navigate: "navigation",
    review: "setup",
    publish: "build",
    library: "library",
  };
  selectTab(target[stage]);
}

function setViewMode(mode: typeof viewMode): void {
  viewMode = mode;
  pairCanvas.dataset.view = mode;
  document.querySelectorAll<HTMLButtonElement>("[data-view]").forEach((button) => button.classList.toggle("active", button.dataset.view === mode));
  element<HTMLElement>("#overlayControl").hidden = mode === "compare";
  window.requestAnimationFrame(() => renderOverlay());
}

function setEvidenceMode(mode: Exclude<typeof evidenceMode, "none">): void {
  evidenceMode = evidenceMode === mode ? "none" : mode;
  if (evidenceMode !== "none" && viewMode !== "compare") setViewMode("compare");
  document.querySelectorAll<HTMLButtonElement>("[data-evidence]").forEach((button) => button.classList.toggle("active", button.dataset.evidence === evidenceMode));
  window.requestAnimationFrame(() => renderEvidence());
}

function paneForSide(side: PanoramaSide): HTMLDivElement {
  return side === "reference" ? referencePane : targetPane;
}

function sideForPane(pane: HTMLDivElement): PanoramaSide {
  return pane === referencePane ? "reference" : "source";
}

function clampViewportPan(side: PanoramaSide): void {
  const pane = paneForSide(side);
  const viewport = viewports[side];
  const maxX = Math.max(0, (viewport.zoom - 1) * pane.clientWidth / 2);
  const maxY = Math.max(0, (viewport.zoom - 1) * pane.clientHeight / 2);
  viewport.panX = Math.max(-maxX, Math.min(maxX, viewport.panX));
  viewport.panY = Math.max(-maxY, Math.min(maxY, viewport.panY));
}

function applyViewportTransform(side: PanoramaSide): void {
  clampViewportPan(side);
  const pane = paneForSide(side);
  const viewport = viewports[side];
  pane.style.setProperty("--viewport-zoom", String(viewport.zoom));
  pane.style.setProperty("--viewport-pan-x", `${viewport.panX}px`);
  pane.style.setProperty("--viewport-pan-y", `${viewport.panY}px`);
  const output = document.querySelector<HTMLOutputElement>(`[data-pane-zoom="${side}"]`);
  if (output) output.value = `${Math.round(viewport.zoom * 100)}%`;
  window.requestAnimationFrame(() => { renderOverlay(); renderEvidence(); });
}

function setViewportZoom(side: PanoramaSide, next: number, anchor?: { x: number; y: number; pane: HTMLDivElement }): void {
  const viewport = viewports[side];
  const previous = viewport.zoom;
  const zoom = Math.max(1, Math.min(8, next));
  if (anchor && previous !== zoom) {
    const bounds = anchor.pane.getBoundingClientRect();
    const dx = anchor.x - (bounds.left + bounds.width / 2);
    const dy = anchor.y - (bounds.top + bounds.height / 2);
    viewport.panX = dx - (zoom / previous) * (dx - viewport.panX);
    viewport.panY = dy - (zoom / previous) * (dy - viewport.panY);
  }
  viewport.zoom = zoom;
  applyViewportTransform(side);
}

function resetViewport(side?: PanoramaSide): void {
  const sides: PanoramaSide[] = side ? [side] : ["reference", "source"];
  sides.forEach((target) => {
    viewports[target] = freshViewport();
    applyViewportTransform(target);
  });
}

function sourcePointForEvent(event: PointerEvent, side: "reference" | "source"): [number, number] | null {
  const pair = currentPair();
  if (!pair) return null;
  const image = imageForPair(pair, side);
  const imageElement = side === "reference" ? sourceImage : targetImage;
  if (!image) return null;
  const bounds = imageElement.getBoundingClientRect();
  if (!(bounds.width > 0) || !(bounds.height > 0)) return null;
  if (event.clientX < bounds.left || event.clientX > bounds.right || event.clientY < bounds.top || event.clientY > bounds.bottom) return null;
  return [
    Math.max(0, Math.min(image.width - 1, ((event.clientX - bounds.left) / bounds.width) * image.width)),
    Math.max(0, Math.min(image.height - 1, ((event.clientY - bounds.top) / bounds.height) * image.height)),
  ];
}

function renderMeasurementReadout(): void {
  const image = activeProject?.images.find((item) => item.id === measurement?.imageId);
  if (!measurement || !image) {
    measurementReadout.hidden = true;
    return;
  }
  const result = erpAngularMeasurement(measurement.start, measurement.end, image);
  measurementReadout.hidden = false;
  element<HTMLElement>("#measurementValue").textContent = `${result.angleDeg.toFixed(2)}°`;
  element<HTMLElement>("#measurementPixels").textContent = `${result.pixelDistance.toFixed(1)} px ERP · distância angular`;
}

function markMetrologyDraftDirty(): void {
  metrologyActionSequence = invalidatePendingAction(metrologyActionSequence);
  metrologyDraftDirty = true;
  floorMeasurement = null;
  verticalMeasurement = null;
  pendingVerticalBase = null;
  metricError = null;
}

function renderMetrology(): void {
  const edit = currentEditImage();
  const countA = metrologyLines.filter((line) => line.family === "floor_a").length;
  const countB = metrologyLines.filter((line) => line.family === "floor_b").length;
  element<HTMLElement>("#floorLineCount").textContent = `A ${countA}/2 · B ${countB}/2`;
  element<HTMLElement>("#floorHintState").textContent = floorHint ? "Piso indicado" : "Ainda não indicado";
  const ready = Boolean(edit && floorHint && countA >= 2 && countB >= 2);
  element<HTMLButtonElement>("#runFloorCalibration").disabled = !ready;
  const calibrationUsable = metrologyCalibrationIsUsable(
    activeMetrology?.image_id ?? null,
    edit?.record.id ?? null,
    metrologyDraftDirty,
  );
  document.querySelectorAll<HTMLButtonElement>('[data-tool="floor_measure"], [data-tool="vertical_measure"]').forEach((button) => {
    button.disabled = !calibrationUsable;
  });
  const diagnostics = element<HTMLElement>("#metrologyDiagnostics");
  if (metrologyError) {
    diagnostics.innerHTML = `<strong>Calibração recusada</strong><p>${escapeHtml(metrologyError)}</p>`;
  } else if (metrologyDraftDirty && activeMetrology?.image_id === edit?.record.id) {
    diagnostics.innerHTML = "<strong>Calibração desatualizada</strong><p>As entradas mudaram; recalibre o piso antes de medir.</p>";
  } else if (!activeMetrology || activeMetrology.image_id !== edit?.record.id) {
    diagnostics.innerHTML = "<strong>Sem calibração nesta imagem</strong><p>Conclua as marcações para habilitar medições métricas.</p>";
  } else {
    const families = activeMetrology.families.map((family) => `${family.family === "floor_a" ? "A" : "B"}: ${family.inlier_indices.length} inliers · P90 ${family.p90_residual_deg.toFixed(2)}°`).join("<br>");
    diagnostics.innerHTML = `<small>Revisão ${activeMetrology.revision}</small><strong>Piso alinhado · incerteza ${activeMetrology.floor_normal_uncertainty_deg.toFixed(2)}°</strong><p>Separação ${activeMetrology.family_separation_deg.toFixed(1)}°<br>${families}</p><code>R panorama → mundo<br>${activeMetrology.rotation_panorama_to_world.map((row) => row.map((value) => value.toFixed(4)).join("  ")).join("<br>")}</code>`;
  }
  const result = element<HTMLElement>("#metricResult");
  if (metricError) {
    result.innerHTML = `<small>Medição recusada</small><strong>Geometria não observável</strong><p>${escapeHtml(metricError)}</p>`;
  } else if (verticalMeasurement) {
    result.innerHTML = `<small>Altura vertical</small><strong>${verticalMeasurement.value.height_m.toFixed(3)} m</strong><p>Resíduo horizontal ${verticalMeasurement.value.horizontal_residual_m.toFixed(3)} m · revisão ${activeMetrology?.revision ?? "—"}</p>`;
  } else if (floorMeasurement) {
    const [x, y] = floorMeasurement.value.point_world_m;
    result.innerHTML = `<small>Ponto no piso</small><strong>${floorMeasurement.value.horizontal_distance_m.toFixed(3)} m</strong><p>Raio ${floorMeasurement.value.radial_distance_m.toFixed(3)} m · mundo (${x.toFixed(2)}, ${y.toFixed(2)}, 0.00)</p>`;
  } else if (pendingVerticalBase) {
    result.innerHTML = "<small>Altura vertical</small><strong>Base definida</strong><p>Agora clique o topo da mesma vertical.</p>";
  } else {
    result.innerHTML = "<small>Resultado</small><strong>—</strong><p>Selecione uma ferramenta métrica após calibrar.</p>";
  }
}

async function loadMetrologyForEdit(): Promise<void> {
  const edit = currentEditImage();
  metrologyLoadSequence += 1;
  metrologyActionSequence += 1;
  element<HTMLButtonElement>("#runFloorCalibration").textContent = "Calibrar piso";
  floorMeasurement = null;
  verticalMeasurement = null;
  pendingVerticalBase = null;
  metricError = null;
  metrologyError = null;
  if (!activeProject || !edit) {
    activeMetrology = null;
    metrologyLines = [];
    floorHint = null;
    renderMetrology();
    return;
  }
  const request = {
    projectId: activeProject.id,
    imageId: edit.record.id,
    sequence: metrologyLoadSequence,
  };
  const result = await loadFloorCalibration(request.projectId, request.imageId);
  if (!imageRequestIsCurrent(request, {
    projectId: activeProject?.id ?? null,
    imageId: currentEditImage()?.record.id ?? null,
    sequence: metrologyLoadSequence,
  })) return;
  activeMetrology = result;
  if (activeMetrology) calibratedImageIds.add(activeMetrology.image_id);
  metrologyDraftDirty = false;
  metrologyLines = activeMetrology ? [...activeMetrology.lines] : [];
  floorHint = activeMetrology?.floor_hint_xy ?? null;
  if (activeMetrology) element<HTMLInputElement>("#cameraHeight").value = activeMetrology.camera_height_m.toFixed(2);
  renderMetrology();
  renderFlow();
  renderOverlay();
}

async function calibrateActiveFloor(): Promise<void> {
  const edit = currentEditImage();
  if (!activeProject || !edit || !floorHint) return;
  const request = {
    projectId: activeProject.id,
    imageId: edit.record.id,
    sequence: metrologyActionSequence + 1,
  };
  metrologyActionSequence = request.sequence;
  const button = element<HTMLButtonElement>("#runFloorCalibration");
  button.disabled = true;
  button.textContent = "Calibrando…";
  metrologyError = null;
  try {
    const result = await calibrateFloor({
      projectId: request.projectId,
      imageId: request.imageId,
      cameraHeightM: Number(element<HTMLInputElement>("#cameraHeight").value),
      floorHint,
      lines: metrologyLines,
    });
    if (!imageRequestIsCurrent(request, {
      projectId: activeProject?.id ?? null,
      imageId: currentEditImage()?.record.id ?? null,
      sequence: metrologyActionSequence,
    })) return;
    activeMetrology = result;
    calibratedImageIds.add(result.image_id);
    metrologyDraftDirty = false;
    floorMeasurement = null;
    verticalMeasurement = null;
    pendingVerticalBase = null;
    metricError = null;
  } catch (error) {
    if (imageRequestIsCurrent(request, {
      projectId: activeProject?.id ?? null,
      imageId: currentEditImage()?.record.id ?? null,
      sequence: metrologyActionSequence,
    })) metrologyError = error instanceof Error ? error.message : "Evidência insuficiente";
  } finally {
    if (imageRequestIsCurrent(request, {
      projectId: activeProject?.id ?? null,
      imageId: currentEditImage()?.record.id ?? null,
      sequence: metrologyActionSequence,
    })) {
      button.textContent = "Calibrar piso";
      renderMetrology();
      renderFlow();
      if (activeTab === "navigation") renderNavigationTest();
      renderOverlay();
    }
  }
}

async function applyMetrologyPoint(point: { x: number; y: number }): Promise<void> {
  const edit = currentEditImage();
  const [xy] = sourcePoints([point]);
  if (!activeProject || !edit || !xy) return;
  metricError = null;
  metrologyError = null;
  if (tool === "floor_hint") {
    floorHint = xy;
    markMetrologyDraftDirty();
  } else if (tool === "floor_measure") {
    if (!metrologyCalibrationIsUsable(activeMetrology?.image_id ?? null, edit.record.id, metrologyDraftDirty) || !activeMetrology) return;
    const request = {
      projectId: activeProject.id,
      imageId: edit.record.id,
      calibrationRevision: activeMetrology.revision,
      sequence: metrologyActionSequence + 1,
    };
    metrologyActionSequence = request.sequence;
    try {
      const value = await measureFloorPoint({ projectId: request.projectId, imageId: request.imageId, calibrationRevision: request.calibrationRevision, point: xy });
      if (!imageRequestIsCurrent(request, {
        projectId: activeProject?.id ?? null,
        imageId: currentEditImage()?.record.id ?? null,
        sequence: metrologyActionSequence,
      })) return;
      floorMeasurement = { xy, value };
      verticalMeasurement = null;
    } catch (error) {
      if (imageRequestIsCurrent(request, {
        projectId: activeProject?.id ?? null,
        imageId: currentEditImage()?.record.id ?? null,
        sequence: metrologyActionSequence,
      })) metricError = error instanceof Error ? error.message : "Raio inválido";
    }
  } else if (tool === "vertical_measure") {
    if (!metrologyCalibrationIsUsable(activeMetrology?.image_id ?? null, edit.record.id, metrologyDraftDirty) || !activeMetrology) return;
    if (!pendingVerticalBase) {
      pendingVerticalBase = xy;
      floorMeasurement = null;
      verticalMeasurement = null;
    } else {
      const base = pendingVerticalBase;
      pendingVerticalBase = null;
      const request = {
        projectId: activeProject.id,
        imageId: edit.record.id,
        calibrationRevision: activeMetrology.revision,
        sequence: metrologyActionSequence + 1,
      };
      metrologyActionSequence = request.sequence;
      try {
        const value = await measureVertical({ projectId: request.projectId, imageId: request.imageId, calibrationRevision: request.calibrationRevision, base, top: xy });
        if (!imageRequestIsCurrent(request, {
          projectId: activeProject?.id ?? null,
          imageId: currentEditImage()?.record.id ?? null,
          sequence: metrologyActionSequence,
        })) return;
        verticalMeasurement = { base, top: xy, value };
      } catch (error) {
        if (imageRequestIsCurrent(request, {
          projectId: activeProject?.id ?? null,
          imageId: currentEditImage()?.record.id ?? null,
          sequence: metrologyActionSequence,
        })) metricError = error instanceof Error ? error.message : "Geometria inválida";
      }
    }
  }
  renderMetrology();
  renderOverlay();
}

function setTool(nextTool: typeof tool): void {
  tool = tool === nextTool ? "inspect" : nextTool;
  document.querySelectorAll<HTMLButtonElement>("[data-tool]").forEach((button) => button.classList.toggle("active", button.dataset.tool === tool));
  document.querySelectorAll<HTMLButtonElement>("[data-pane-action]").forEach((button) => button.classList.toggle("active", button.dataset.paneAction === tool));
  visualStage.dataset.tool = tool;
  element<HTMLElement>("#toolHint").textContent = tool === "manual_match"
    ? pendingManualReference ? "Clique o ponto correspondente na vizinha." : "Clique primeiro na referência e depois na vizinha."
    : tool === "pan" ? "Arraste a vista selecionada; cada panorama mantém seu próprio enquadramento."
      : tool === "measure" ? "Arraste entre dois pontos para medir o ângulo na esfera."
    : tool === "object_box" ? "Arraste o bounding box do objeto no lado selecionado."
      : tool === "floor_a" ? "Trace uma direção paralela do piso para a família A."
        : tool === "floor_b" ? "Trace outra direção paralela do piso para a família B."
          : tool === "floor_hint" ? "Clique uma região visível do piso."
            : tool === "floor_measure" ? "Clique no piso para intersectar o raio com o plano métrico."
              : tool === "vertical_measure" ? pendingVerticalBase ? "Clique o topo da vertical." : "Clique primeiro a base no piso."
      : tool.startsWith("dense_") ? "Marque a evidência densa no lado selecionado."
        : "Revise matches ou adicione uma observação.";
  if (tool === "object_box") selectTab("objects");
  else if (tool.startsWith("dense_")) selectTab("dense");
  else if (tool.startsWith("floor_") || tool === "vertical_measure") selectTab("metrology");
  else if (tool === "manual_match") selectTab("registration");
}

function renderProjects(): void {
  element<HTMLElement>("#projectCount").textContent = String(projects.length);
  projectList.replaceChildren();
  if (!projects.length) {
    projectList.innerHTML = '<div class="rail-empty">Nenhum projeto local.</div>';
    return;
  }
  projects.forEach((project) => {
    const button = document.createElement("button");
    button.className = `project-row ${activeProject?.id === project.id ? "active" : ""}`;
    const copy = workflowCopy(project.workflow);
    button.innerHTML = `<i></i><span><strong>${escapeHtml(project.name)}</strong><small>${project.status === "built" ? "Publicado" : escapeHtml(copy.title)} · ${project.images.length} imagens</small></span>`;
    button.onclick = () => void openProject(project);
    projectList.append(button);
  });
  renderPublishedEnvironments();
}

function renderPlan(): void {
  stageList.replaceChildren();
  if (!activePlan) {
    buildButton.disabled = true;
    buildButton.textContent = "Build indisponível";
    return;
  }
  activePlan.stages.forEach((stage, index) => {
    const row = document.createElement("div");
    row.className = `build-stage status-${stage.status}`;
    row.innerHTML = `<span class="stage-index">${String(index + 1).padStart(2, "0")}</span><div><strong>${escapeHtml(stage.title)}</strong><small>${escapeHtml(stage.detail)}</small><code>${escapeHtml(stage.engine)}</code></div><b>${escapeHtml(stage.status.replace("_", " "))}</b>`;
    stageList.append(row);
  });
  element<HTMLElement>("#nextAction").textContent = activePlan.next_action;
  const estimatedMapCanPublish = Boolean(activeProject?.workflow === "spherical_unregistered" && activeGlobalMap?.success && !activeGlobalMap.stale);
  const canPublish = activePlan.can_build_now || estimatedMapCanPublish;
  buildButton.disabled = stitchingOperationRunning || !canPublish;
  buildButton.textContent = canPublish
    ? activePlan.workflow.startsWith("perspective_")
      ? "Executar costura ERP"
      : activeProject?.status === "built" ? "Publicar nova revisão" : "Publicar ambiente"
    : activePlan.workflow === "spherical_unregistered"
      ? "Estimativa ainda não executada"
      : "Build bloqueado nesta etapa";
  renderFlow();
}

function isPerspectiveProject(project: ProjectRecord | null = activeProject): boolean {
  return project?.workflow === "perspective_calibrated" || project?.workflow === "perspective_uncalibrated";
}

function captureGroupForImage(image: ImageRecord): string {
  const value = image.metadata.capture_group ?? image.metadata.group_id ?? "capture-1";
  return String(value);
}

function renderStitching(): void {
  const button = element<HTMLButtonElement>("#runStitching");
  const promote = element<HTMLButtonElement>("#promoteStitching");
  const image = element<HTMLImageElement>("#stitchImage");
  const empty = element<HTMLElement>("#stitchEmpty");
  const download = element<HTMLAnchorElement>("#downloadStitching");
  const groupList = element<HTMLElement>("#stitchGroups");
  const sources = element<HTMLElement>("#stitchSources");
  const pairs = element<HTMLElement>("#stitchPairList");
  const perspective = isPerspectiveProject();
  const controls = stitchingControlState({
    operationRunning: stitchingOperationRunning,
    perspectiveProject: perspective,
    canBuild: Boolean(activePlan?.can_build_now),
    promotionReady: Boolean(activeStitching?.ready_for_promotion),
    stitchingStale: Boolean(activeStitching?.stale),
  });
  button.disabled = controls.runDisabled;
  buildButton.disabled = controls.buildDisabled;
  button.textContent = activeStitching ? "Reexecutar PanoramaStitcher" : "Executar PanoramaStitcher";
  promote.disabled = controls.promoteDisabled;
  groupList.replaceChildren();
  sources.replaceChildren();
  pairs.replaceChildren();

  if (!activeProject || !perspective) {
    element<HTMLElement>("#stitchHealth").textContent = "Disponível para grupos de fotos";
    element<HTMLElement>("#stitchGuidance").textContent = "Crie uma rota Fotos calibradas ou Fotos sem calibração.";
    image.hidden = true;
    empty.hidden = false;
    download.hidden = true;
    element<HTMLElement>("#stitchPromoteHealth").textContent = "São necessários dois ERPs aceitos";
    return;
  }

  const pendingGroups = Array.from(new Set(activeProject.images.map(captureGroupForImage)));
  const resultGroups = activeStitching?.groups ?? [];
  const groupIds = resultGroups.length ? resultGroups.map((group) => group.group_id) : pendingGroups;
  if (!activeStitchGroupId || !groupIds.includes(activeStitchGroupId)) {
    activeStitchGroupId = resultGroups.find((group) => group.accepted)?.group_id ?? groupIds[0] ?? null;
  }
  groupIds.forEach((groupId) => {
    const result = resultGroups.find((item) => item.group_id === groupId);
    const count = result?.sources.length ?? activeProject?.images.filter((item) => captureGroupForImage(item) === groupId).length ?? 0;
    const groupButton = document.createElement("button");
    groupButton.className = `stitch-group ${groupId === activeStitchGroupId ? "active" : ""} ${result ? result.status : "pending"}`;
    groupButton.innerHTML = `<span>${escapeHtml(groupId)}</span><small>${count} fotos</small><b>${result ? result.accepted ? "ERP aceito" : "recusado" : "aguardando"}</b>`;
    groupButton.onclick = () => {
      activeStitchGroupId = groupId;
      renderStitching();
    };
    groupList.append(groupButton);
  });

  const selectedGroup = resultGroups.find((group) => group.group_id === activeStitchGroupId) ?? null;
  const selectedImages = selectedGroup
    ? selectedGroup.sources.map((source) => activeProject?.images.find((item) => item.id === source.image_id)).filter((item): item is ImageRecord => Boolean(item))
    : activeProject.images.filter((item) => captureGroupForImage(item) === activeStitchGroupId);
  selectedImages.forEach((sourceImage, index) => {
    const recorded = selectedGroup?.sources[index];
    const calibrated = selectedGroup?.intrinsics[index];
    const row = document.createElement("div");
    row.className = "stitch-source-row";
    const fov = calibrated?.horizontal_fov_deg ?? recorded?.horizontal_fov_deg ?? null;
    const provenance = recorded?.fov_source === "supplied" ? "fornecido" : "estimado pelo engine";
    row.innerHTML = `<span>${String(index + 1).padStart(2, "0")}</span><div><strong>${escapeHtml(sourceImage.filename)}</strong><small>${sourceImage.width}×${sourceImage.height} · ${fov == null ? "FOV a estimar" : `${fov.toFixed(2)}° · ${provenance}`}</small></div><b>${selectedGroup?.accepted ? "costurada" : "entrada"}</b>`;
    sources.append(row);
  });

  if (!activeStitching) {
    element<HTMLElement>("#stitchHealth").textContent = "Costura ainda não executada";
    element<HTMLElement>("#stitchGuidance").textContent = activeProject.workflow === "perspective_calibrated"
      ? "Usará o FOV/fx fornecido e estimará as orientações SO(3)."
      : "Estimará FOV compartilhado e orientações somente quando o overlap for observável.";
    image.hidden = true;
    empty.hidden = false;
    empty.textContent = "O ERP parcial e sua validade aparecerão aqui.";
    download.hidden = true;
    element<HTMLElement>("#stitchCoverage").textContent = "—";
    element<HTMLElement>("#stitchOverlap").textContent = "—";
    element<HTMLElement>("#stitchPairs").textContent = "—";
    element<HTMLElement>("#stitchEstimatedFov").textContent = "—";
    element<HTMLElement>("#stitchResultDetail").textContent = "Aguardando execução pública.";
    element<HTMLElement>("#stitchPromoteHealth").textContent = "São necessários dois ERPs aceitos";
    element<HTMLElement>("#stitchPromoteDetail").textContent = "Cada capture_group é validado isoladamente antes de entrar no registro.";
    return;
  }

  element<HTMLElement>("#stitchHealth").textContent = activeStitching.stale
    ? `Costura rev ${activeStitching.revision} desatualizada`
    : `${activeStitching.accepted_group_count}/${activeStitching.groups.length} grupos produziram ERP · rev ${activeStitching.revision}`;
  element<HTMLElement>("#stitchGuidance").textContent = selectedGroup?.accepted
    ? `O grupo ${selectedGroup.group_id} passou; revise cobertura, overlap, FOV e pares.`
    : selectedGroup
      ? `O grupo ${selectedGroup.group_id} foi isolado: ${selectedGroup.reasons.join("; ")}`
      : "Selecione um grupo para inspecionar o resultado.";
  if (selectedGroup?.accepted && selectedGroup.output) {
    image.src = stitchingImageUrl(activeProject.id, selectedGroup.group_id, activeStitching.revision);
    image.hidden = false;
    empty.hidden = true;
    download.href = image.src;
    download.download = `${activeProject.name.toLowerCase().replace(/[^a-z0-9]+/g, "-")}-${selectedGroup.group_id}-erp.png`;
    download.hidden = false;
    element<HTMLElement>("#stitchCoverage").textContent = `${(selectedGroup.output.valid_fraction * 100).toFixed(1)}%`;
    element<HTMLElement>("#stitchOverlap").textContent = `${(selectedGroup.output.overlap_pixel_fraction * 100).toFixed(1)}%`;
    element<HTMLElement>("#stitchPairs").textContent = `${selectedGroup.diagnostics?.accepted_pair_count ?? 0}/${selectedGroup.diagnostics?.candidate_pair_count ?? 0}`;
    element<HTMLElement>("#stitchEstimatedFov").textContent = String(selectedGroup.diagnostics?.estimated_fov_count ?? 0);
    element<HTMLElement>("#stitchResultDetail").textContent = `${selectedGroup.output.shape_hw[1]}×${selectedGroup.output.shape_hw[0]} · suporte e validade preservados separadamente`;
  } else {
    image.hidden = true;
    empty.hidden = false;
    empty.textContent = "Nenhum ERP foi promovido a resultado válido.";
    download.hidden = true;
    element<HTMLElement>("#stitchCoverage").textContent = "recusado";
    element<HTMLElement>("#stitchOverlap").textContent = "—";
    element<HTMLElement>("#stitchPairs").textContent = "0";
    element<HTMLElement>("#stitchEstimatedFov").textContent = "—";
    element<HTMLElement>("#stitchResultDetail").textContent = selectedGroup?.reasons.join("; ") ?? "Nenhum grupo selecionado.";
  }

  selectedGroup?.pairwise.forEach((pair) => {
    const row = document.createElement("div");
    row.className = "stitch-pair-row";
    const a = activeProject?.images.find((item) => item.id === pair.image_a_id)?.filename ?? pair.image_a_id;
    const b = activeProject?.images.find((item) => item.id === pair.image_b_id)?.filename ?? pair.image_b_id;
    row.innerHTML = `<div><strong>${escapeHtml(a)} ↔ ${escapeHtml(b)}</strong><small>${pair.num_inliers} inliers · mediana ${pair.median_angular_error_deg.toFixed(3)}° · cobertura ${(pair.spatial_coverage * 100).toFixed(1)}%</small></div><b>${pair.observable ? "observável" : "fraco"}</b>`;
    pairs.append(row);
  });

  const accepted = activeStitching.accepted_group_count;
  element<HTMLElement>("#stitchPromoteHealth").textContent = activeStitching.promoted_project_id
    ? `${accepted} ERPs já promovidos para registro`
    : activeStitching.ready_for_promotion
      ? `${accepted} ERPs prontos para registro`
      : `${accepted}/2 ERPs aceitos`;
  element<HTMLElement>("#stitchPromoteDetail").textContent = activeStitching.ready_for_promotion
    ? "Será criado um projeto sem poses; todas as relações começam como candidatos a Essential."
    : "Execute ao menos dois grupos observáveis. Grupos recusados não contaminam o projeto seguinte.";
  promote.disabled = controls.promoteDisabled;
  promote.textContent = activeStitching.promoted_project_id ? "Abrir projeto ERP" : `Promover ${accepted} ERPs`;
}

async function executeStitching(): Promise<void> {
  if (stitchingOperationRunning || !activeProject || !isPerspectiveProject()) return;
  const request = {
    projectId: activeProject.id,
    sequence: stitchingSequence + 1,
  };
  stitchingSequence = request.sequence;
  stitchingOperationRunning = true;
  renderPlan();
  renderStitching();
  const button = element<HTMLButtonElement>("#runStitching");
  button.textContent = "Detectando + calibrando + fundindo…";
  element<HTMLElement>("#stitchGuidance").textContent = "Executando os seis estágios públicos do PanoramaStitcher.";
  try {
    const result = await runStitching(request.projectId);
    if (!projectRequestIsCurrent(request, {
      projectId: activeProject?.id ?? null,
      sequence: stitchingSequence,
    })) return;
    activeStitching = result;
    activeStitchGroupId = activeStitching.groups.find((group) => group.accepted)?.group_id ?? activeStitching.groups[0]?.group_id ?? null;
  } catch (error) {
    if (projectRequestIsCurrent(request, {
      projectId: activeProject?.id ?? null,
      sequence: stitchingSequence,
    })) element<HTMLElement>("#stitchGuidance").textContent = error instanceof Error ? error.message : "Falha na costura";
  } finally {
    stitchingOperationRunning = false;
    renderPlan();
    renderStitching();
  }
}

async function executeStitchingPromotion(): Promise<void> {
  if (stitchingOperationRunning || !activeProject || !activeStitching?.ready_for_promotion) return;
  const sourceProject = activeProject;
  const request = {
    projectId: sourceProject.id,
    sequence: stitchingPromotionSequence + 1,
  };
  stitchingPromotionSequence = request.sequence;
  stitchingOperationRunning = true;
  renderPlan();
  renderStitching();
  const button = element<HTMLButtonElement>("#promoteStitching");
  button.textContent = "Criando pares candidatos…";
  try {
    const promoted = await promoteStitching(request.projectId);
    if (!projectRequestIsCurrent(request, {
      projectId: activeProject?.id ?? null,
      sequence: stitchingPromotionSequence,
    })) return;
    const index = projects.findIndex((project) => project.id === promoted.id);
    if (index >= 0) projects[index] = promoted;
    else projects.unshift(promoted);
    await openProject(promoted);
  } catch (error) {
    if (projectRequestIsCurrent(request, {
      projectId: activeProject?.id ?? null,
      sequence: stitchingPromotionSequence,
    })) {
      element<HTMLElement>("#stitchPromoteDetail").textContent = error instanceof Error ? error.message : "Falha ao promover ERPs";
      renderStitching();
    }
  } finally {
    stitchingOperationRunning = false;
    renderPlan();
    renderStitching();
  }
}

function renderDenseAnnotations(): void {
  const dense = activeProject?.dense_annotations ?? [];
  const boundAnchorCount = activeSetup?.summary.bound_range_prior_count
    ?? dense.filter((item) => item.engine_effect === "range_prior").length;
  const requiredAnchors = activeGlobalMap?.range_prior_min_count ?? 3;
  element<HTMLElement>("#denseStatus").textContent = denseAnchorStatus({
    boundCount: boundAnchorCount,
    requiredCount: requiredAnchors,
    mapScale: activeGlobalMap?.scale,
    appliedCount: activeGlobalMap?.range_priors_applied ?? 0,
  });
  denseList.replaceChildren();
  if (!dense.length) {
    denseList.innerHTML = '<div class="constraints-empty"><strong>Nenhuma evidência densa</strong><span>Regiões e âncoras serão preservadas sem virar profundidade válida por inferência.</span></div>';
  } else {
    dense.forEach((annotation, index) => {
      const image = activeProject?.images.find((item) => item.id === annotation.image_id);
      const row = document.createElement("div");
      row.className = "constraint-row";
      const binding = annotation.engine_effect === "range_prior"
        ? ` · feature ${annotation.feature_index} · Δ${annotation.binding_distance_px?.toFixed(1)} px`
        : annotation.engine_effect === "pair_validity"
          ? " · máscara do estéreo A↔B"
          : annotation.kind === "range_anchor" ? " · sem inlier próximo" : " · anotação visual";
      const effect = annotation.engine_effect === "range_prior"
        ? "prior"
        : annotation.engine_effect === "pair_validity" ? "stereo mask" : "visual";
      row.innerHTML = `<span>${String(index + 1).padStart(2, "0")}</span><div><strong>${escapeHtml(annotation.kind.replace("_", " "))} · ${escapeHtml(annotation.label)}</strong><small>${escapeHtml(image?.filename ?? annotation.image_id)}${annotation.radial_range == null ? "" : ` · ${annotation.radial_range.toFixed(2)} m`}${binding}</small></div><b>${effect}</b>`;
      denseList.append(row);
    });
  }
  renderDensePreview();
  renderDenseSceneReadiness();
}

function renderDenseSceneReadiness(): void {
  const command = document.querySelector<HTMLElement>(".dense-scene-command")!;
  const action = element<HTMLButtonElement>("#denseSceneAction");
  if (!activeSetup) {
    command.dataset.state = "map";
    element<HTMLElement>("#denseSceneHealth").textContent = "Lendo pré-requisitos";
    element<HTMLElement>("#denseSceneDetail").textContent = "Mapa global, escala métrica e três vistas são verificados separadamente.";
    action.disabled = true;
    renderDenseSceneResult();
    return;
  }
  const summary = activeSetup.summary;
  const guidance = denseSceneGuidance(summary);
  command.dataset.state = guidance.state;
  element<HTMLElement>("#denseSceneHealth").textContent = activeDenseScene
    ? activeDenseScene.stale
      ? `Cena rev ${activeDenseScene.revision} desatualizada`
      : activeDenseScene.accepted
        ? `Superfície aceita · rev ${activeDenseScene.revision}`
        : `Reconstrução recusada · rev ${activeDenseScene.revision}`
    : guidance.title;
  element<HTMLElement>("#denseSceneDetail").textContent = activeDenseScene
    ? activeDenseScene.accepted
      ? `${activeDenseScene.surface.vertex_count} vértices e ${activeDenseScene.surface.face_count} faces produzidos pelo pipeline público.`
      : `O motor não promoveu superfície: ${activeDenseScene.reasons.join("; ")}.`
    : guidance.detail;
  element<HTMLElement>("#denseMapGate").textContent = `Mapa ${summary.global_map_status === "current" ? "✓ atual" : `· ${summary.global_map_status}`}`;
  element<HTMLElement>("#denseScaleGate").textContent = `Escala ${summary.metric_scale_available ? `✓ ${summary.metric_scale_sources.join(" + ")}` : "· ausente"}`;
  element<HTMLElement>("#denseViewsGate").textContent = `Vistas ${summary.accepted_component_size >= 3 ? "✓ " : "· "}${summary.accepted_component_size} (mín. 3)`;
  action.disabled = false;
  action.textContent = guidance.state === "ready"
    ? activeDenseScene ? "Reexecutar MVS + TSDF" : "Executar MVS + TSDF"
    : guidance.action;
  action.onclick = () => {
    if (guidance.state === "map" || (guidance.state === "scale" && summary.bound_range_prior_count >= summary.range_prior_min_count)) {
      selectTab("map");
    } else if (guidance.state === "scale") {
      setTool("dense_anchor");
    } else if (guidance.state === "views") {
      selectTab("setup");
    } else {
      void executeDenseScene();
    }
  };
  renderDenseSceneResult();
}

function renderDenseSceneResult(): void {
  const panel = element<HTMLElement>("#denseSceneResult");
  const canvas = element<SVGSVGElement>("#denseSceneCanvas");
  canvas.replaceChildren();
  panel.hidden = !activeDenseScene;
  if (!activeDenseScene) return;
  const record = activeDenseScene;
  const points = record.surface.display_xy;
  const colors = record.surface.display_rgb;
  if (points.length) {
    points.forEach(([x, y], index) => {
      const circle = document.createElementNS("http://www.w3.org/2000/svg", "circle");
      circle.setAttribute("cx", String(180 + x * 155));
      circle.setAttribute("cy", String(95 + y * 78));
      circle.setAttribute("r", "1.5");
      const color = colors[index] ?? [180, 180, 180];
      circle.setAttribute("fill", `rgb(${color[0]},${color[1]},${color[2]})`);
      circle.setAttribute("opacity", ".8");
      canvas.append(circle);
    });
  } else {
    const textNode = document.createElementNS("http://www.w3.org/2000/svg", "text");
    textNode.setAttribute("x", "180");
    textNode.setAttribute("y", "95");
    textNode.setAttribute("text-anchor", "middle");
    textNode.textContent = record.accepted ? "Superfície sem amostra visual" : "Nenhuma superfície promovida";
    canvas.append(textNode);
  }
  element<HTMLElement>("#denseSceneRangeCount").textContent = `${record.range_maps.length}/${record.scene.view_count}`;
  element<HTMLElement>("#denseSceneVertexCount").textContent = String(record.surface.vertex_count);
  element<HTMLElement>("#denseSceneFaceCount").textContent = String(record.surface.face_count);
  element<HTMLElement>("#denseSceneColorFraction").textContent = `${(record.surface.colored_vertex_fraction * 100).toFixed(0)}%`;
  element<HTMLElement>("#denseSceneResultDetail").textContent = record.stale
    ? "Resultado anterior às últimas poses ou máscaras."
    : record.accepted
      ? `${record.scene.view_count} vistas · ${record.profile} · ${record.global_map_scale}`
      : record.reasons.join("; ");
  const download = element<HTMLAnchorElement>("#downloadDenseScene");
  download.hidden = !record.surface.file_available || !activeProject;
  if (!download.hidden && activeProject) {
    download.href = denseSceneMeshUrl(activeProject.id, record.revision);
    download.download = `${activeProject.name.toLowerCase().replace(/[^a-z0-9]+/g, "-")}-surface.ply`;
  } else {
    download.removeAttribute("href");
  }
}

async function executeDenseScene(): Promise<void> {
  if (!activeProject || !activeSetup?.summary.dense_scene_eligible) return;
  const request = {
    projectId: activeProject.id,
    sequence: denseSceneSequence + 1,
  };
  denseSceneSequence = request.sequence;
  const button = element<HTMLButtonElement>("#denseSceneAction");
  button.disabled = true;
  button.textContent = "MVS + TSDF…";
  element<HTMLElement>("#denseSceneDetail").textContent = "Estimando ranges em cada referência, fundindo TSDF e validando a superfície.";
  try {
    const result = await runDenseScene(request.projectId, "interactive");
    if (!projectRequestIsCurrent(request, {
      projectId: activeProject?.id ?? null,
      sequence: denseSceneSequence,
    })) return;
    activeDenseScene = result;
  } catch (error) {
    if (projectRequestIsCurrent(request, {
      projectId: activeProject?.id ?? null,
      sequence: denseSceneSequence,
    })) element<HTMLElement>("#denseSceneDetail").textContent = error instanceof Error ? error.message : "Falha na reconstrução densa";
  } finally {
    if (projectRequestIsCurrent(request, {
      projectId: activeProject?.id ?? null,
      sequence: denseSceneSequence,
    })) renderDenseSceneReadiness();
  }
}

function renderDensePreview(): void {
  const pair = currentPair();
  const button = element<HTMLButtonElement>("#runDensePreview");
  const pairState = activeSetup?.pairs.find((item) => item.pair_id === pair?.id);
  button.disabled = !activeProject || !pair || pairState?.status !== "accepted";
  element<HTMLElement>("#densePreviewHealth").textContent = !activeDensePreview
    ? "Ainda não executado"
    : activeDensePreview.stale
      ? `Preview rev ${activeDensePreview.revision} desatualizado`
      : `Preview rev ${activeDensePreview.revision} · ${activeDensePreview.scale_status === "metric" ? "métrico" : "escala do baseline"}`;
  element<HTMLElement>("#densePreviewDetail").textContent = !activeDensePreview
    ? "Usa a pose aceita; cada panorama vira referência uma vez e recebe sua própria máscara Incluir/Excluir."
    : `${activeDensePreview.input_mask.include_annotation_ids.length + activeDensePreview.source_input_mask.include_annotation_ids.length} inclusão(ões), ${activeDensePreview.input_mask.exclude_annotation_ids.length + activeDensePreview.source_input_mask.exclude_annotation_ids.length} exclusão(ões) consumidas nos dois sentidos.`;
  element<HTMLElement>("#denseValidFraction").textContent = activeDensePreview ? `${(activeDensePreview.valid_fraction * 100).toFixed(1)}% / ${(activeDensePreview.reverse.valid_fraction * 100).toFixed(1)}%` : "—";
  element<HTMLElement>("#denseMaskFraction").textContent = activeDensePreview ? `${(activeDensePreview.input_mask.eligible_fraction * 100).toFixed(1)}% / ${(activeDensePreview.source_input_mask.eligible_fraction * 100).toFixed(1)}%` : "—";
  element<HTMLElement>("#denseMedianRange").textContent = activeDensePreview?.median_range == null || activeDensePreview.reverse.median_range == null ? "—" : `${activeDensePreview.median_range.toFixed(2)} / ${activeDensePreview.reverse.median_range.toFixed(2)} ${activeDensePreview.units}`;
  element<HTMLElement>("#denseMedianConfidence").textContent = activeDensePreview?.median_confidence == null || activeDensePreview.reverse.median_confidence == null ? "—" : `${activeDensePreview.median_confidence.toFixed(2)} / ${activeDensePreview.reverse.median_confidence.toFixed(2)}`;
  const empty = element<HTMLElement>("#densePreviewEmpty");
  if (activeDensePreview && activeProject && pair) {
    densePreviewImage.src = densePreviewImageUrl(activeProject.id, pair.id, activeDensePreview.revision);
    densePreviewImage.hidden = false;
    empty.hidden = true;
  } else {
    densePreviewImage.removeAttribute("src");
    densePreviewImage.hidden = true;
    empty.hidden = false;
  }
}

function renderPairStrip(): void {
  pairStrip.replaceChildren();
  pairSelect.replaceChildren();
  if (!activeProject?.pairs.length) {
    pairSelect.disabled = true;
    pairSelect.append(new Option("Nenhum par definido", ""));
    pairStrip.innerHTML = '<div class="pair-strip-empty"><strong>Sem pares para registrar</strong><span>Adicione ou estime vizinhanças para iniciar a inspeção.</span></div>';
    return;
  }
  pairSelect.disabled = false;
  activeProject.pairs.forEach((pair, index) => {
    const reference = imageForPair(pair, "reference");
    const source = imageForPair(pair, "source");
    if (!reference || !source) return;
    const overlap = pair.minimum_directional_overlap == null ? "a estimar" : `${Math.round(pair.minimum_directional_overlap * 100)}%`;
    const baseline = pair.baseline_m == null ? "escala livre" : `${pair.baseline_m.toFixed(2)} m`;
    pairSelect.append(new Option(`Par ${String(index + 1).padStart(2, "0")} · ${baseline} · ${overlap}`, String(index)));
    const button = document.createElement("button");
    button.className = index === activePairIndex ? "active" : "";
    button.innerHTML = `<span class="pair-index">${String(index + 1).padStart(2, "0")}</span><span class="pair-thumbs"><img src="${imageUrl(activeProject!, reference)}" alt=""/><img src="${imageUrl(activeProject!, source)}" alt=""/></span><span class="pair-card-copy"><strong>${baseline}</strong><small>${overlap} de sobreposição</small></span>`;
    button.onclick = () => void selectPair(index);
    pairStrip.append(button);
  });
  pairSelect.value = String(activePairIndex);
}

function renderNavigationTest(): void {
  const pair = currentPair();
  const currentImage = pair ? imageForPair(pair, "reference") : activeProject?.images[0] ?? null;
  const metricReady = Boolean(currentImage && calibratedImageIds.has(currentImage.id));
  element<HTMLElement>("#navigationCurrent").textContent = currentImage?.filename ?? "Nenhum panorama";
  element<HTMLElement>("#navigationNodeCount").textContent = `${activeProject?.images.length ?? 0} nó(s)`;
  element<HTMLElement>("#navigationMetricState").textContent = metricReady
    ? "Piso e altura de câmera calibrados"
    : "Calibre o piso desta imagem para medir em metros";
  element<HTMLElement>("#navigationMetricCheck").innerHTML = `<i></i> ${metricReady ? "Métrica disponível nesta vista" : "Escala métrica pendente"}`;
  element<HTMLButtonElement>("#navigationMeasureFloor").disabled = !metricReady;
  element<HTMLButtonElement>("#navigationMeasureVertical").disabled = !metricReady;
  const neighbors = element<HTMLDivElement>("#navigationNeighbors");
  neighbors.replaceChildren();
  if (!activeProject?.pairs.length) {
    neighbors.innerHTML = '<div class="constraints-empty"><strong>Sem destinos conectados</strong><span>Volte ao registro multivista e aceite ao menos um par.</span></div>';
    return;
  }
  activeProject.pairs.forEach((candidate, index) => {
    const reference = imageForPair(candidate, "reference");
    const source = imageForPair(candidate, "source");
    if (!reference || !source) return;
    const button = document.createElement("button");
    button.className = index === activePairIndex ? "active" : "";
    button.innerHTML = `<span>${String(index + 1).padStart(2, "0")}</span><img src="${imageUrl(activeProject!, source)}" alt=""/><div><strong>${escapeHtml(source.filename)}</strong><small>via ${escapeHtml(reference.filename)} · ${candidate.baseline_m == null ? "escala livre" : `${candidate.baseline_m.toFixed(2)} m`}</small></div><b>abrir →</b>`;
    button.onclick = async () => {
      await selectPair(index);
      activeFlowStage = "navigate";
      renderFlow();
      selectTab("navigation");
    };
    neighbors.append(button);
  });
}

function renderPublishedEnvironments(): void {
  const published = projects.filter((project) => project.status === "built");
  publishedList.replaceChildren();
  if (!published.length) {
    publishedList.innerHTML = '<div class="library-empty"><span>09</span><strong>Nenhum ambiente publicado</strong><p>Finalize os bloqueios da etapa 8 para que o Street View apareça aqui.</p><button class="primary" id="libraryGoPublish">Ir para publicação</button></div>';
    const action = document.querySelector<HTMLButtonElement>("#libraryGoPublish");
    if (action) action.onclick = () => selectFlowStage("publish");
    return;
  }
  published.forEach((project) => {
    const button = document.createElement("button");
    const preview = project.images[0];
    button.className = "published-environment";
    button.innerHTML = `${preview ? `<img src="${imageUrl(project, preview)}" alt=""/>` : ""}<div><small>Street View publicado</small><strong>${escapeHtml(project.name)}</strong><span>${project.images.length} panorama(s) · ${project.pairs.length} conexão(ões)</span></div><b>Navegar →</b>`;
    button.onclick = async () => {
      await openProject(project);
      selectFlowStage("navigate");
    };
    publishedList.append(button);
  });
}

function renderSetup(): void {
  setupRoute.replaceChildren();
  setupPairs.replaceChildren();
  if (!activeSetup || !activeProject) return;
  const project = activeProject;
  const summary = activeSetup.summary;
  const complete = activeSetup.steps.filter((step) => step.status === "ready").length;
  const progress = Math.round((complete / Math.max(1, activeSetup.steps.length)) * 100);
  element<HTMLElement>("#setupProgress").textContent = `${progress}%`;
  element<HTMLElement>("#setupHeadline").textContent = summary.global_map_status === "current"
    ? "Mapa global aceito"
    : summary.global_map_eligible
      ? "Pares prontos para o mapa global"
      : `${summary.accepted_pair_count}/${summary.pair_count} pares aceitos`;
  element<HTMLElement>("#setupGuidance").textContent = summary.metric_scale_available
    ? "A cena tem evidência métrica; avance somente com pares aceitos e conectados."
    : "O registro pode avançar em escala arbitrária; o denso métrico continua bloqueado.";
  element<HTMLElement>("#setupRegistered").textContent = `${summary.registered_pair_count}/${summary.pair_count}`;
  element<HTMLElement>("#setupAccepted").textContent = `${summary.accepted_pair_count}/${summary.pair_count}`;
  element<HTMLElement>("#setupComponent").textContent = `${summary.accepted_component_size}/${summary.image_count}`;
  element<HTMLElement>("#setupScale").textContent = summary.metric_scale_available ? "métrica" : "livre";
  element<HTMLElement>("#setupActionTitle").textContent = activeSetup.recommended_action.label;
  element<HTMLElement>("#setupActionDetail").textContent = activeSetup.recommended_action.detail;
  const action = element<HTMLButtonElement>("#setupAction");
  action.textContent = ["global-map", "inspect-map", "rerun-map-for-scale"].includes(activeSetup.recommended_action.kind)
    ? "Ver mapa"
    : ["scale", "dense-scene"].includes(activeSetup.recommended_action.kind)
      ? "Abrir denso"
      : "Abrir par";
  action.onclick = () => {
    const pairId = activeSetup?.recommended_action.pair_id;
    if (pairId && activeProject) {
      const index = activeProject.pairs.findIndex((pair) => pair.id === pairId);
      if (index >= 0) void selectPairAndOpenRegistration(index);
    } else if (["global-map", "inspect-map", "rerun-map-for-scale"].includes(activeSetup?.recommended_action.kind ?? "")) {
      selectTab("map");
    } else if (["scale", "dense-scene"].includes(activeSetup?.recommended_action.kind ?? "")) {
      selectTab("dense");
    } else {
      selectTab("build");
    }
  };
  activeSetup.steps.forEach((step, index) => {
    const row = document.createElement("div");
    row.className = `setup-step status-${step.status}`;
    row.innerHTML = `<span>${String(index + 1).padStart(2, "0")}</span><i></i><div><strong>${escapeHtml(step.title)}</strong><small>${escapeHtml(step.detail)}</small><code>${escapeHtml(step.engine)}</code></div><b>${escapeHtml(step.status)}</b>`;
    setupRoute.append(row);
  });
  activeSetup.pairs.forEach((state, index) => {
    const pair = project.pairs.find((item) => item.id === state.pair_id);
    if (!pair) return;
    const reference = project.images.find((item) => item.id === pair.reference_image_id);
    const source = project.images.find((item) => item.id === pair.source_image_id);
    const button = document.createElement("button");
    button.className = `setup-pair status-${state.status} ${pair.id === currentPair()?.id ? "active" : ""}`;
    const score = state.quality_score == null ? "—" : `${Math.round(state.quality_score * 100)}`;
    const pending = state.pending_review_count || state.evidence_newer_than_pose
      ? `<em>${state.pending_review_count || 1} edição pendente</em>`
      : "";
    button.innerHTML = `<span>${String(index + 1).padStart(2, "0")}</span><div><strong>${escapeHtml(reference?.filename ?? pair.reference_image_id)} ↔ ${escapeHtml(source?.filename ?? pair.source_image_id)}</strong><small>${state.revision == null ? "não executado" : `rev ${state.revision} · ${state.num_inliers} inliers · Q${score}`}</small></div>${pending}<b>${escapeHtml(state.status)}</b>`;
    button.onclick = () => void selectPairAndOpenRegistration(index);
    setupPairs.append(button);
  });
}

function mapCoordinate(value: number, span: number, margin: number): number {
  return margin + ((value + 1) / 2) * (span - margin * 2);
}

function cameraMapTransform(poses: GlobalMapRecord["poses"]): (point: [number, number]) => [number, number] {
  if (!poses.length) return (point) => point;
  const xs = poses.map((pose) => pose.display_xy[0]);
  const ys = poses.map((pose) => pose.display_xy[1]);
  const centerX = (Math.min(...xs) + Math.max(...xs)) / 2;
  const centerY = (Math.min(...ys) + Math.max(...ys)) / 2;
  const radius = Math.max(
    ...poses.map((pose) => Math.max(Math.abs(pose.display_xy[0] - centerX), Math.abs(pose.display_xy[1] - centerY))),
    1e-6,
  );
  return ([x, y]) => [(x - centerX) / radius * 0.72, (y - centerY) / radius * 0.72];
}

function renderGlobalMap(): void {
  globalMapCanvas.innerHTML = "";
  mapDiagnostics.replaceChildren();
  mapEdgeList.replaceChildren();
  const runButton = element<HTMLButtonElement>("#runGlobalMap");
  const accepted = activeSetup?.summary.accepted_pair_count ?? 0;
  runButton.disabled = !activeProject || accepted === 0;
  element<HTMLElement>("#mapCameraCount").textContent = activeGlobalMap ? String(activeGlobalMap.poses.length) : "—";
  element<HTMLElement>("#mapEdgeCount").textContent = activeGlobalMap ? String(activeGlobalMap.edges.length) : "—";
  element<HTMLElement>("#mapTrackCount").textContent = activeGlobalMap ? String(activeGlobalMap.track_count) : "—";
  element<HTMLElement>("#mapScaleBadge").textContent = isMetricMapScale(activeGlobalMap?.scale) ? "escala métrica" : activeGlobalMap ? "escala relativa" : "escala —";
  const reprojection = activeGlobalMap?.diagnostics.reprojection_p90_deg;
  element<HTMLElement>("#mapReprojection").textContent = typeof reprojection === "number" ? `${reprojection.toFixed(3)}°` : "—";

  if (!activeProject || !activeSetup) return;
  if (!activeGlobalMap) {
    element<HTMLElement>("#mapHealth").textContent = activeSetup.summary.global_map_eligible ? "Componente pronto para mapear" : "Mapa ainda não elegível";
    element<HTMLElement>("#mapGuidance").textContent = activeSetup.summary.global_map_eligible
      ? "Execute o mapper público e inspecione consistência, tracks e resíduos."
      : `O componente aceito contém ${activeSetup.summary.accepted_component_size}/${activeSetup.summary.image_count} vistas.`;
  } else {
    element<HTMLElement>("#mapHealth").textContent = activeGlobalMap.success
      ? activeGlobalMap.stale ? "Mapa desatualizado por novas evidências" : "Mapa global aceito"
      : `Mapa recusado · ${activeGlobalMap.failure_reasons.join(", ")}`;
    element<HTMLElement>("#mapGuidance").textContent = activeGlobalMap.success
      ? `Revisão ${activeGlobalMap.revision} · coordenadas ${isMetricMapScale(activeGlobalMap.scale) ? "métricas" : "em escala arbitrária"}.`
      : "Nenhuma geometria parcial foi promovida; use o diagnóstico para corrigir o grafo.";
  }

  const positions = new Map<string, [number, number]>();
  if (activeGlobalMap?.success) {
    const transform = cameraMapTransform(activeGlobalMap.poses);
    activeGlobalMap.poses.forEach((pose) => positions.set(pose.image_id, transform(pose.display_xy)));
    activeGlobalMap.points.display_xy.forEach(([x, y]) => {
      const [px, py] = transform([x, y]);
      if (Math.abs(px) <= 1 && Math.abs(py) <= 1) {
        globalMapCanvas.insertAdjacentHTML("beforeend", `<circle class="map-point" cx="${mapCoordinate(px, 400, 28)}" cy="${mapCoordinate(py, 250, 24)}" r="1"/>`);
      }
    });
  } else {
    activeProject.images.forEach((image, index) => {
      const angle = (index / Math.max(1, activeProject!.images.length)) * Math.PI * 2 - Math.PI / 2;
      positions.set(image.id, [Math.cos(angle) * 0.78, Math.sin(angle) * 0.78]);
    });
  }
  const mapEdges = activeGlobalMap?.edges ?? [];
  mapEdges.forEach((edge) => {
    const a = positions.get(edge.reference_image_id);
    const b = positions.get(edge.source_image_id);
    if (!a || !b) return;
    globalMapCanvas.insertAdjacentHTML("beforeend", `<line class="map-edge" data-map-pair="${escapeHtml(edge.pair_id)}" x1="${mapCoordinate(a[0], 400, 28)}" y1="${mapCoordinate(a[1], 250, 24)}" x2="${mapCoordinate(b[0], 400, 28)}" y2="${mapCoordinate(b[1], 250, 24)}"/>`);
  });
  positions.forEach(([x, y], imageId) => {
    const image = activeProject!.images.find((item) => item.id === imageId);
    const index = activeProject!.images.findIndex((item) => item.id === imageId);
    const cx = mapCoordinate(x, 400, 28); const cy = mapCoordinate(y, 250, 24);
    const pose = activeGlobalMap?.poses.find((item) => item.image_id === imageId);
    const reference = activeGlobalMap?.reference_image_id === imageId ? " reference" : "";
    globalMapCanvas.insertAdjacentHTML("beforeend", `<g class="map-camera${reference}" transform="translate(${cx} ${cy})"><circle r="10"/><path d="M 0 -15 L -6 -6 L 6 -6 Z"/><text x="14" y="4">${String(index + 1).padStart(2, "0")}</text><title>${escapeHtml(image?.filename ?? imageId)} · ${pose?.active_track_count ?? 0} tracks</title></g>`);
  });
  globalMapCanvas.querySelectorAll<SVGLineElement>("[data-map-pair]").forEach((line) => {
    line.onclick = () => {
      const index = activeProject?.pairs.findIndex((pair) => pair.id === line.dataset.mapPair) ?? -1;
      if (index >= 0) void selectPairAndOpenRegistration(index);
    };
  });

  const recommended = activeGlobalMap?.recommended_edit;
  element<HTMLElement>("#mapActionTitle").textContent = recommended?.title ?? "Conecte três vistas";
  element<HTMLElement>("#mapActionDetail").textContent = recommended?.detail ?? "Estime pares até formar um componente multivista.";
  const action = element<HTMLButtonElement>("#mapAction");
  action.hidden = !recommended?.pair_id;
  action.onclick = () => {
    const index = activeProject?.pairs.findIndex((pair) => pair.id === recommended?.pair_id) ?? -1;
    if (index >= 0) void selectPairAndOpenRegistration(index);
  };

  if (!activeGlobalMap) {
    mapDiagnostics.innerHTML = '<div class="diagnostic-empty">O mapper produzirá diagnóstico sem expor geometria parcial em caso de falha.</div>';
    return;
  }
  const diagnostics = activeGlobalMap.diagnostics;
  const diagnosticRows = [
    ["Arestas admitidas", diagnostics.admitted_edge_count],
    ["Tracks candidatos", diagnostics.track_count],
    ["Conflitos de track", diagnostics.track_conflict_count],
    ["Suporte de profundidade", typeof diagnostics.translation_positive_depth_ratio === "number" ? `${Math.round(diagnostics.translation_positive_depth_ratio * 100)}%` : "—"],
    ["Corroboração multivista", diagnostics.multiview_corroboration_passed === true ? "passou" : diagnostics.multiview_corroboration_passed === false ? "falhou" : "—"],
    ["Backend BA", diagnostics.bundle_compute_backend],
  ];
  diagnosticRows.forEach(([label, value]) => {
    mapDiagnostics.insertAdjacentHTML("beforeend", `<div><span>${escapeHtml(String(label))}</span><b>${escapeHtml(String(value ?? "—"))}</b></div>`);
  });
  activeGlobalMap.edges.forEach((edge) => {
    const pair = activeProject?.pairs.find((item) => item.id === edge.pair_id);
    const reference = activeProject?.images.find((item) => item.id === pair?.reference_image_id);
    const source = activeProject?.images.find((item) => item.id === pair?.source_image_id);
    const button = document.createElement("button");
    button.className = "map-edge-row";
    button.innerHTML = `<span>${escapeHtml(edge.pair_id)}</span><div><strong>${escapeHtml(reference?.filename ?? edge.reference_image_id)} ↔ ${escapeHtml(source?.filename ?? edge.source_image_id)}</strong><small>rev ${edge.revision} · ${edge.num_inliers} inliers · Q${Math.round(edge.quality_score * 100)}</small></div><b>inspecionar</b>`;
    button.onclick = () => {
      const index = activeProject?.pairs.findIndex((item) => item.id === edge.pair_id) ?? -1;
      if (index >= 0) void selectPairAndOpenRegistration(index);
    };
    mapEdgeList.append(button);
  });
}

function gateValue(gate: QualityGate): string {
  if (typeof gate.value === "string") return gate.value;
  if (["inlier-ratio", "coverage", "cheirality"].includes(gate.id)) return `${Math.round(gate.value * 100)}%`;
  if (["residual", "parallax", "rotation-stability", "translation-stability"].includes(gate.id)) return `${gate.value.toFixed(2)}°`;
  return String(gate.value);
}

function renderQualityDiagnostics(): void {
  qualityGates.replaceChildren();
  const guidance = element<HTMLDivElement>("#editGuidance");
  const guidanceLabel = guidance.querySelector<HTMLElement>("small")!;
  const guidanceTitle = guidance.querySelector<HTMLElement>("strong")!;
  const guidanceDetail = guidance.querySelector<HTMLElement>("p")!;
  const guidanceButton = guidance.querySelector<HTMLButtonElement>("#editGuidanceAction")!;
  const gates = activeRegistration?.quality_gates ?? [];
  const recommended = activeRegistration?.recommended_edit;
  if (!activeRegistration || !recommended) {
    guidanceLabel.textContent = "Leitura do engine";
    guidanceTitle.textContent = "Rode o Essential para receber uma ação sugerida.";
    guidanceDetail.textContent = "Os gates usam o perfil público PanorAi.";
    guidanceButton.hidden = true;
    qualityGates.innerHTML = '<div class="diagnostic-empty">Sem diagnóstico geométrico.</div>';
    return;
  }
  guidanceLabel.textContent = activeRegistration.pending_match_reviews?.length ? "Edição ainda não estimada" : "Leitura do engine";
  guidanceTitle.textContent = recommended.title;
  guidanceDetail.textContent = recommended.detail;
  guidanceButton.hidden = false;
  guidanceButton.textContent = recommended.tool === "next_pair" ? "Próximo par" : recommended.tool === "inspect" ? "Mostrar inliers" : "Abrir ferramenta";
  guidanceButton.onclick = () => applyRecommendedEdit(recommended.tool);
  const ordered = [...gates].sort((left, right) => Number(left.passed) - Number(right.passed));
  ordered.forEach((gate) => {
    const row = document.createElement("div");
    row.className = `quality-gate ${gate.passed ? "passed" : "failed"}`;
    const relation = gate.threshold_kind === "minimum" ? "mín" : gate.threshold_kind === "maximum" ? "máx" : "alvo";
    row.innerHTML = `<i></i><div><strong>${escapeHtml(gate.label)}</strong><small>${escapeHtml(gate.action)}</small></div><span>${escapeHtml(gateValue(gate))}<small>${relation} ${escapeHtml(String(gate.threshold))}</small></span>`;
    qualityGates.append(row);
  });
}

function applyRecommendedEdit(recommendedTool: string): void {
  if (recommendedTool === "next_pair") {
    void selectPairAndOpenRegistration(activePairIndex + 1);
  } else if (recommendedTool === "matches") {
    evidenceMode = "matches";
    matchFilter = "review";
    document.querySelectorAll<HTMLButtonElement>("[data-evidence]").forEach((item) => item.classList.toggle("active", item.dataset.evidence === "matches"));
    document.querySelectorAll<HTMLButtonElement>("[data-match-filter]").forEach((item) => item.classList.toggle("active", item.dataset.matchFilter === "review"));
    renderMatchList();
    renderEvidence();
  } else if (recommendedTool === "manual_match" || recommendedTool === "object_box") {
    setTool(recommendedTool as "manual_match" | "object_box");
    renderPairWorkspace();
  } else {
    evidenceMode = "inliers";
    document.querySelectorAll<HTMLButtonElement>("[data-evidence]").forEach((item) => item.classList.toggle("active", item.dataset.evidence === "inliers"));
    renderEvidence();
  }
}

function renderRevisionEvidence(): void {
  const delta = element<HTMLDivElement>("#revisionDelta");
  const comparison = activeRegistration?.revision_comparison;
  if (!comparison) {
    delta.hidden = true;
  } else {
    delta.hidden = false;
    const quality = comparison.quality_delta == null ? "sem comparação" : `${comparison.quality_delta >= 0 ? "+" : ""}${(comparison.quality_delta * 100).toFixed(1)} qualidade`;
    const inliers = `${comparison.inlier_delta >= 0 ? "+" : ""}${comparison.inlier_delta} inliers`;
    delta.innerHTML = `<span>REV ${comparison.from_revision} → ${comparison.to_revision}</span><strong>${inliers}</strong><i>·</i><strong>${quality}</strong><b>${comparison.accepted_after ? "aceita" : "revisar"}</b>`;
  }
  historyList.replaceChildren();
  const revisions = activeHistory?.revisions ?? [];
  element<HTMLElement>("#historyCount").textContent = `${revisions.length} snapshot${revisions.length === 1 ? "" : "s"}`;
  if (!revisions.length) {
    historyList.innerHTML = '<div class="diagnostic-empty">O próximo cálculo cria o primeiro snapshot imutável.</div>';
    return;
  }
  revisions.slice(0, 5).forEach((revision) => {
    const row = document.createElement("div");
    row.className = `history-row ${revision.accepted ? "accepted" : "review"}`;
    row.innerHTML = `<span>R${revision.revision}</span><div><strong>${revision.num_inliers} inliers · Q${revision.quality_score == null ? "—" : Math.round(revision.quality_score * 100)}</strong><small>${revision.region_mode} · ${revision.accepted_reviews} aprovados · ${revision.rejected_reviews} rejeitados</small></div><b>${revision.accepted ? "aceita" : "revisar"}</b>`;
    historyList.append(row);
  });
}

function renderRegistrationSummary(pair: ImagePairRecord): void {
  if (!activeProject) return;
  const pairQuality = element<HTMLElement>("#pairQuality");
  const pose = activeRegistration?.pose;
  if (!activeRegistration) {
    const assessment = pairAssessment(pair);
    pairQuality.textContent = "não executado";
    pairQuality.dataset.tone = "unknown";
    element<HTMLElement>("#registrationHealth").textContent = "Pose não estimada";
    element<HTMLElement>("#registrationGuidance").textContent = assessment.guidance;
    element<HTMLElement>("#featureValue").textContent = "—";
    element<HTMLElement>("#matchValue").textContent = "—";
    element<HTMLElement>("#inlierValue").textContent = "—";
    element<HTMLElement>("#qualityValue").textContent = "—";
    element<HTMLElement>("#parallaxValue").textContent = "—";
    element<HTMLButtonElement>("#reestimateEssential").disabled = true;
    renderQualityDiagnostics();
    renderRevisionEvidence();
    renderMatchList();
    renderObjectList();
    return;
  }
  const accepted = Boolean(pose?.accepted);
  pairQuality.textContent = pose ? `${pose.num_inliers} inliers` : "sem solução";
  pairQuality.dataset.tone = accepted ? "strong" : pose ? "limited" : "unknown";
  element<HTMLElement>("#registrationHealth").textContent = pose
    ? accepted ? "Essential aceita pelo gate" : "Essential requer revisão"
    : "Essential sem solução";
  element<HTMLElement>("#registrationGuidance").textContent = pose
    ? `R e direção de t em escala livre · revisão ${activeRegistration.revision}`
    : "A correspondência ativa não sustentou uma hipótese com o mínimo de inliers.";
  element<HTMLElement>("#featureValue").textContent = `${activeRegistration.feature_counts.reference} / ${activeRegistration.feature_counts.source}`;
  element<HTMLElement>("#matchValue").textContent = String(activeRegistration.matches.length);
  element<HTMLElement>("#inlierValue").textContent = pose ? String(pose.num_inliers) : "0";
  element<HTMLElement>("#qualityValue").textContent = pose ? `${Math.round(pose.quality.raw_quality_score * 100)} / 100` : "—";
  element<HTMLElement>("#parallaxValue").textContent = pose ? `${pose.median_parallax_deg.toFixed(2)}°` : "—";
  element<HTMLSelectElement>("#regionMode").value = activeRegistration.region_mode;
  const pendingReviewCount = activeRegistration.pending_match_reviews?.length ?? 0;
  const reestimateButton = element<HTMLButtonElement>("#reestimateEssential");
  reestimateButton.disabled = false;
  reestimateButton.textContent = pendingReviewCount
    ? `Aplicar ${pendingReviewCount} revisão${pendingReviewCount === 1 ? "" : "ões"}`
    : "Reestimar com revisão";
  renderQualityDiagnostics();
  renderRevisionEvidence();
  renderMatchList();
  renderObjectList();
}

function renderMatchList(): void {
  matchList.replaceChildren();
  if (!activeRegistration) {
    matchList.innerHTML = '<div class="constraints-empty"><strong>Sem execução</strong><span>Rode o Essential para visualizar detecções, matches e resíduos.</span></div>';
    return;
  }
  const matches = activeRegistration.matches.filter((match) => matchFilter === "all"
    || (matchFilter === "inlier" && match.inlier)
    || (matchFilter === "review" && match.review_state !== "accepted"));
  appendMatchRows(matchList, matches);
}

function appendMatchRows(container: HTMLElement, matches: MatchRecord[]): void {
  matches.forEach((match) => {
    const row = document.createElement("div");
    row.className = `match-row ${match.inlier ? "inlier" : ""} ${match.review_state} ${selectedMatchIndex === match.index ? "selected" : ""}`;
    row.dataset.matchIndex = String(match.index);
    row.innerHTML = `<button class="match-main"><span>#${String(match.index + 1).padStart(3, "0")}</span><i></i><div><strong>${match.inlier ? "inlier" : "outlier"}${match.region_match ? " · objeto" : ""}</strong><small>dist ${match.descriptor_distance.toFixed(2)}${match.residual_deg == null ? "" : ` · res ${match.residual_deg.toFixed(3)}°`}</small></div></button><div class="match-decisions"><button data-review="accepted" aria-label="Aprovar match">✓</button><button data-review="rejected" aria-label="Rejeitar match">×</button></div>`;
    row.querySelector<HTMLButtonElement>(".match-main")!.onclick = () => {
      selectedMatchIndex = match.index;
      element<HTMLElement>("#matchSelection").textContent = `Match #${match.index + 1} · ${match.review_state}`;
      renderMatchList();
      renderEvidence();
    };
    row.querySelectorAll<HTMLButtonElement>("[data-review]").forEach((button) => {
      button.classList.toggle("active", button.dataset.review === match.review_state);
      button.onclick = () => void updateMatchReview(match.index, button.dataset.review as "accepted" | "rejected");
    });
    container.append(row);
  });
}

function objectEvidence(objectId: string): {
  referenceBbox: [number, number, number, number];
  sourceBbox: [number, number, number, number];
} | null {
  const pair = currentPair();
  if (!activeProject || !pair) return null;
  const reference = activeProject.object_regions.find((region) => region.object_id === objectId && region.image_id === pair.reference_image_id && region.state === "accepted");
  const source = activeProject.object_regions.find((region) => region.object_id === objectId && region.image_id === pair.source_image_id && region.state === "accepted");
  return reference && source ? { referenceBbox: reference.bbox_xyxy, sourceBbox: source.bbox_xyxy } : null;
}

function pointInside(xy: [number, number], bbox: [number, number, number, number]): boolean {
  return xy[0] >= bbox[0] && xy[0] <= bbox[2] && xy[1] >= bbox[1] && xy[1] <= bbox[3];
}

function matchesForObject(objectId: string): MatchRecord[] {
  const regions = objectEvidence(objectId);
  if (!activeRegistration || !regions) return [];
  return activeRegistration.matches.filter((match) => pointInside(match.reference_xy, regions.referenceBbox) && pointInside(match.source_xy, regions.sourceBbox));
}

function renderObjectList(): void {
  objectList.replaceChildren();
  objectMatchList.replaceChildren();
  const pair = currentPair();
  if (!activeProject || !pair) return;
  const relevant = activeProject.object_regions.filter((region) => region.image_id === pair.reference_image_id || region.image_id === pair.source_image_id);
  if (!relevant.length) {
    objectList.innerHTML = '<div class="constraints-empty"><strong>Nenhum objeto compartilhado</strong><span>Desenhe o mesmo ID nas duas vistas para ativar a restrição.</span></div>';
    return;
  }
  const ids = [...new Set(relevant.map((region) => region.object_id))];
  if (activeObjectId && !ids.includes(activeObjectId)) activeObjectId = null;
  ids.forEach((objectId) => {
    const observations = relevant.filter((region) => region.object_id === objectId);
    const summary = activeRegistration?.object_summaries.find((item) => item.object_id === objectId);
    const row = document.createElement("button");
    row.className = `object-row ${observations.length === 2 ? "complete" : "pending"} ${activeObjectId === objectId ? "active" : ""}`;
    row.innerHTML = `<div><small>${observations.length}/2 vistas</small><strong>${escapeHtml(observations[0]?.label ?? objectId)}</strong><span>${summary ? `${summary.reference_detection_count} + ${summary.source_detection_count} detecções visíveis · ${summary.match_count} matches` : "rode ou reestime para contar features"}</span></div><b>${observations.length === 2 ? "pareado" : "replicar"}</b>`;
    row.onclick = () => {
      activeObjectId = activeObjectId === objectId ? null : objectId;
      evidenceMode = activeObjectId ? "matches" : "none";
      document.querySelectorAll<HTMLButtonElement>("[data-evidence]").forEach((item) => item.classList.toggle("active", item.dataset.evidence === evidenceMode));
      renderObjectList();
      renderEvidence();
    };
    objectList.append(row);
  });
  if (!activeObjectId) {
    objectMatchList.innerHTML = '<div class="object-evidence-empty">Selecione um objeto pareado para isolar suas detecções e revisar os matches.</div>';
    return;
  }
  const objectMatches = matchesForObject(activeObjectId);
  const summary = activeRegistration?.object_summaries.find((item) => item.object_id === activeObjectId);
  const heading = document.createElement("div");
  heading.className = "object-evidence-heading";
  heading.innerHTML = `<div><small>Evidência em ${escapeHtml(activeObjectId)}</small><strong>${summary ? `${summary.reference_detection_count + summary.source_detection_count} detecções exibidas · ${objectMatches.length} matches` : "Reestime para indexar a região"}</strong></div><button type="button" data-object-detections>Ver detecções</button>`;
  heading.querySelector<HTMLButtonElement>("[data-object-detections]")!.onclick = () => {
    evidenceMode = evidenceMode === "detections" ? "matches" : "detections";
    document.querySelectorAll<HTMLButtonElement>("[data-evidence]").forEach((item) => item.classList.toggle("active", item.dataset.evidence === evidenceMode));
    renderEvidence();
  };
  objectMatchList.append(heading);
  if (!objectMatches.length) {
    objectMatchList.insertAdjacentHTML("beforeend", '<div class="object-evidence-empty">Nenhum match atravessa os dois bounding boxes. Ajuste as regiões ou adicione uma correspondência manual.</div>');
    return;
  }
  appendMatchRows(objectMatchList, objectMatches);
}

function renderPairWorkspace(): void {
  const pair = currentPair();
  const perspective = isPerspectiveProject();
  document.querySelector<HTMLElement>(".pair-picker")!.hidden = perspective;
  document.querySelector<HTMLElement>(".view-modes")!.hidden = perspective;
  document.querySelector<HTMLElement>(".evidence-modes")!.hidden = perspective;
  element<HTMLButtonElement>("#previousImage").hidden = perspective;
  element<HTMLButtonElement>("#nextImage").hidden = perspective;
  document.querySelector<HTMLElement>(".pair-footer")!.hidden = perspective;
  if (!activeProject || !pair) {
    measurementReadout.hidden = true;
    pairCanvas.hidden = true;
    evidenceLayer.setAttribute("hidden", "");
    constraintLayer.setAttribute("hidden", "");
    emptyState.hidden = false;
    if (activeProject) {
      if (perspective) {
        const groupCount = new Set(activeProject.images.map(captureGroupForImage)).size;
        const thumbnails = activeProject.images.map((image) => `<figure><img src="${imageUrl(activeProject!, image)}" alt="${escapeHtml(image.filename)}"/><figcaption>${escapeHtml(captureGroupForImage(image))} · ${escapeHtml(image.filename)}</figcaption></figure>`).join("");
        emptyState.innerHTML = `<div class="perspective-stage"><small>${groupCount} tomada(s) · ${activeProject.images.length} fotos</small><div>${thumbnails}</div><strong>Construir um ERP por tomada</strong><p>PanoramaStitcher estima overlap, FOV ausente e orientações SO(3) em cada grupo isoladamente. Captura com translação/parallax é recusada.</p><button class="primary" id="openStitching">Abrir Costura ERP</button></div>`;
        element<HTMLButtonElement>("#openStitching").onclick = () => selectTab("stitching");
      } else {
        emptyState.innerHTML = '<span class="empty-orbit"></span><strong>O registro ainda não tem pares</strong><p>Defina vizinhanças ou execute a estimativa de pose para comparar panoramas.</p><button class="primary" id="emptyCreate">Adicionar imagens</button>';
        element<HTMLButtonElement>("#emptyCreate").onclick = openDialog;
      }
    }
    const groupCount = perspective && activeProject ? new Set(activeProject.images.map(captureGroupForImage)).size : 0;
    element<HTMLElement>("#imageName").textContent = perspective ? `${groupCount} tomada(s)` : "Nenhum par";
    element<HTMLElement>("#imageMeta").textContent = perspective ? `${activeProject?.images.length ?? 0} fotos pinhole` : "—";
    element<HTMLElement>("#stageLabel").hidden = true;
    element<HTMLElement>("#toolGroup").hidden = perspective;
    return;
  }
  const reference = imageForPair(pair, "reference");
  const source = imageForPair(pair, "source");
  if (!reference || !source) return;
  document.querySelectorAll<HTMLButtonElement>("[data-manual-side]").forEach((button) => button.classList.toggle("active", button.dataset.manualSide === manualSide));
  emptyState.hidden = true;
  element<HTMLElement>("#toolGroup").hidden = false;
  document.querySelectorAll<HTMLElement>(".pane-tools").forEach((tools) => { tools.hidden = false; });
  pairCanvas.hidden = false;
  constraintLayer.removeAttribute("hidden");
  sourceImage.src = imageUrl(activeProject, reference);
  sourceImage.alt = `Referência · ${reference.filename}`;
  targetImage.src = imageUrl(activeProject, source);
  targetImage.alt = `Vizinha · ${source.filename}`;
  element<HTMLElement>("#referenceName").textContent = reference.filename;
  element<HTMLElement>("#targetName").textContent = source.filename;
  element<HTMLElement>("#imageName").textContent = pair.id;
  const overlap = pair.minimum_directional_overlap == null ? "overlap a estimar" : `${(pair.minimum_directional_overlap * 100).toFixed(1)}% overlap`;
  const baseline = pair.baseline_m == null ? "escala livre" : `${pair.baseline_m.toFixed(2)} m baseline`;
  element<HTMLElement>("#imageMeta").textContent = `${baseline} · ${overlap}`;
  element<HTMLElement>("#stageLabel").textContent = activeRegistration?.pose
    ? `Essential · ${activeRegistration.pose.num_inliers} inliers · revisão ${activeRegistration.revision}`
    : `Diagnóstico visual · ${pair.relation === "registered_overlap" ? "poses fornecidas" : "pose desconhecida"} · ${pair.id}`;
  element<HTMLElement>("#stageLabel").hidden = false;
  renderRegistrationSummary(pair);
  setViewMode(viewMode);
  window.requestAnimationFrame(() => {
    applyViewportTransform("reference");
    applyViewportTransform("source");
  });
}

function renderOverlay(preview?: { kind: string; points: Array<{ x: number; y: number }> }): void {
  const edit = currentEditImage();
  if (!activeProject || !edit) return;
  const { record: image, element: editImage, pane } = edit;
  if (constraintLayer.parentElement !== pane) pane.append(constraintLayer);
  referencePane.classList.toggle("manual-active", manualSide === "reference" && tool !== "inspect");
  targetPane.classList.toggle("manual-active", manualSide === "source" && tool !== "inspect");
  const width = Math.max(1, editImage.offsetWidth);
  const height = Math.max(1, editImage.offsetHeight);
  constraintLayer.setAttribute("viewBox", `0 0 ${width} ${height}`);
  constraintLayer.setAttribute("width", String(width));
  constraintLayer.setAttribute("height", String(height));
  constraintLayer.style.width = `${width}px`;
  constraintLayer.style.height = `${height}px`;
  constraintLayer.innerHTML = `<rect class="interaction-surface" x="0" y="0" width="${width}" height="${height}"/>`;
  const sx = width / image.width;
  const sy = height / image.height;
  activeProject.object_regions.filter((item) => item.image_id === image.id && item.state === "accepted").forEach((region) => {
    const [x0, y0, x1, y1] = region.bbox_xyxy;
    constraintLayer.insertAdjacentHTML("beforeend", `<rect class="object-box" x="${x0 * sx}" y="${y0 * sy}" width="${(x1 - x0) * sx}" height="${(y1 - y0) * sy}"/><text class="object-label" x="${x0 * sx + 6}" y="${y0 * sy + 17}">${escapeHtml(region.label)}</text>`);
  });
  activeProject.dense_annotations.filter((item) => item.image_id === image.id && item.state === "accepted").forEach((annotation) => {
    if (annotation.kind === "range_anchor") {
      const [point] = annotation.points_xy;
      constraintLayer.insertAdjacentHTML("beforeend", `<circle class="dense-anchor" cx="${point[0] * sx}" cy="${point[1] * sy}" r="8"/><text class="dense-label" x="${point[0] * sx + 11}" y="${point[1] * sy - 8}">${annotation.radial_range?.toFixed(2)} r</text>`);
    } else {
      const [a, b] = annotation.points_xy;
      const x = Math.min(a[0], b[0]) * sx;
      const y = Math.min(a[1], b[1]) * sy;
      constraintLayer.insertAdjacentHTML("beforeend", `<rect class="dense-region ${annotation.kind}" x="${x}" y="${y}" width="${Math.abs(b[0] - a[0]) * sx}" height="${Math.abs(b[1] - a[1]) * sy}"/><text class="dense-label" x="${x + 6}" y="${y + 17}">${escapeHtml(annotation.label)}</text>`);
    }
  });
  const pair = currentPair();
  if (pair) {
    activeProject.manual_correspondences.filter((item) => item.pair_id === pair.id && item.state === "accepted").forEach((match) => {
      const point = manualSide === "reference" ? match.reference_xy : match.source_xy;
      constraintLayer.insertAdjacentHTML("beforeend", `<circle class="manual-match-point" cx="${point[0] * sx}" cy="${point[1] * sy}" r="6"/>`);
    });
  }
  if (measurement?.imageId === image.id) {
    const result = erpAngularMeasurement(measurement.start, measurement.end, image);
    const [x1, y1] = [measurement.start[0] * sx, measurement.start[1] * sy];
    const [x2, y2] = [measurement.end[0] * sx, measurement.end[1] * sy];
    const labelX = Math.min(width - 92, Math.max(8, (x1 + x2) / 2 + 8));
    const labelY = Math.min(height - 10, Math.max(20, (y1 + y2) / 2 - 10));
    constraintLayer.insertAdjacentHTML("beforeend", `<line class="measurement-line" x1="${x1}" y1="${y1}" x2="${x2}" y2="${y2}"/><circle class="measurement-point" cx="${x1}" cy="${y1}" r="5"/><circle class="measurement-point" cx="${x2}" cy="${y2}" r="5"/><text class="measurement-label" x="${labelX}" y="${labelY}">${result.angleDeg.toFixed(2)}°</text>`);
  }
  metrologyLines.forEach((line, index) => {
    const className = line.family === "floor_a" ? "floor-family-a" : "floor-family-b";
    constraintLayer.insertAdjacentHTML("beforeend", `<line class="floor-direction ${className}" x1="${line.start_xy[0] * sx}" y1="${line.start_xy[1] * sy}" x2="${line.end_xy[0] * sx}" y2="${line.end_xy[1] * sy}"/><text class="floor-direction-label ${className}" x="${line.start_xy[0] * sx + 7}" y="${line.start_xy[1] * sy - 7}">${line.family === "floor_a" ? "A" : "B"}${index + 1}</text>`);
  });
  if (floorHint) constraintLayer.insertAdjacentHTML("beforeend", `<circle class="floor-hint" cx="${floorHint[0] * sx}" cy="${floorHint[1] * sy}" r="8"/><text class="floor-hint-label" x="${floorHint[0] * sx + 11}" y="${floorHint[1] * sy - 9}">piso</text>`);
  if (floorMeasurement) constraintLayer.insertAdjacentHTML("beforeend", `<circle class="metric-point" cx="${floorMeasurement.xy[0] * sx}" cy="${floorMeasurement.xy[1] * sy}" r="7"/><text class="metric-label" x="${floorMeasurement.xy[0] * sx + 11}" y="${floorMeasurement.xy[1] * sy - 9}">${floorMeasurement.value.horizontal_distance_m.toFixed(2)} m</text>`);
  if (pendingVerticalBase) constraintLayer.insertAdjacentHTML("beforeend", `<circle class="metric-base" cx="${pendingVerticalBase[0] * sx}" cy="${pendingVerticalBase[1] * sy}" r="7"/>`);
  if (verticalMeasurement) constraintLayer.insertAdjacentHTML("beforeend", `<line class="metric-vertical" x1="${verticalMeasurement.base[0] * sx}" y1="${verticalMeasurement.base[1] * sy}" x2="${verticalMeasurement.top[0] * sx}" y2="${verticalMeasurement.top[1] * sy}"/><circle class="metric-base" cx="${verticalMeasurement.base[0] * sx}" cy="${verticalMeasurement.base[1] * sy}" r="6"/><circle class="metric-top" cx="${verticalMeasurement.top[0] * sx}" cy="${verticalMeasurement.top[1] * sy}" r="6"/><text class="metric-label" x="${verticalMeasurement.top[0] * sx + 11}" y="${verticalMeasurement.top[1] * sy - 9}">${verticalMeasurement.value.height_m.toFixed(2)} m</text>`);
  if (preview && preview.points.length === 2) {
    const a = preview.points[0]; const b = preview.points[1];
    if (preview.kind === "floor_a" || preview.kind === "floor_b") {
      constraintLayer.insertAdjacentHTML("beforeend", `<line class="floor-direction ${preview.kind === "floor_a" ? "floor-family-a" : "floor-family-b"}" x1="${a.x}" y1="${a.y}" x2="${b.x}" y2="${b.y}"/>`);
    } else {
      constraintLayer.insertAdjacentHTML("beforeend", `<rect class="preview-box" x="${Math.min(a.x, b.x)}" y="${Math.min(a.y, b.y)}" width="${Math.abs(b.x - a.x)}" height="${Math.abs(b.y - a.y)}"/>`);
    }
  }
}

function renderEvidence(): void {
  const pair = currentPair();
  if (!activeProject || !pair || !activeRegistration || evidenceMode === "none" || pairCanvas.hidden) {
    evidenceLayer.setAttribute("hidden", "");
    evidenceLayer.innerHTML = "";
    return;
  }
  const canvasBounds = pairCanvas.getBoundingClientRect();
  const referenceBounds = sourceImage.getBoundingClientRect();
  const sourceBounds = targetImage.getBoundingClientRect();
  evidenceLayer.removeAttribute("hidden");
  evidenceLayer.setAttribute("viewBox", `0 0 ${Math.max(1, canvasBounds.width)} ${Math.max(1, canvasBounds.height)}`);
  evidenceLayer.innerHTML = "";
  const map = (xy: [number, number], image: ImageRecord, bounds: DOMRect): [number, number] => [
    bounds.left - canvasBounds.left + (xy[0] / image.width) * bounds.width,
    bounds.top - canvasBounds.top + (xy[1] / image.height) * bounds.height,
  ];
  const reference = imageForPair(pair, "reference");
  const source = imageForPair(pair, "source");
  if (!reference || !source) return;
  if (evidenceMode === "detections") {
    const selectedRegions = activeTab === "objects" && activeObjectId ? objectEvidence(activeObjectId) : null;
    for (const [side, detections, image, bounds] of [
      ["reference", activeRegistration.detections.reference, reference, referenceBounds],
      ["source", activeRegistration.detections.source, source, sourceBounds],
    ] as const) {
      const bbox = side === "reference" ? selectedRegions?.referenceBbox : selectedRegions?.sourceBbox;
      detections.filter((detection) => !bbox || pointInside(detection.xy, bbox)).forEach((detection) => {
        const [x, y] = map(detection.xy, image, bounds);
        evidenceLayer.insertAdjacentHTML("beforeend", `<circle class="feature-dot ${side}" cx="${x}" cy="${y}" r="1.7"/>`);
      });
    }
  } else {
    const objectMatches = activeTab === "objects" && activeObjectId ? new Set(matchesForObject(activeObjectId).map((match) => match.index)) : null;
    const visible = activeRegistration.matches
      .filter((match) => !objectMatches || objectMatches.has(match.index))
      .filter((match) => evidenceMode === "matches" || match.inlier)
      .sort((a, b) => Number(b.index === selectedMatchIndex) - Number(a.index === selectedMatchIndex) || Number(b.inlier) - Number(a.inlier))
      .slice(0, 220);
    visible.forEach((match) => {
      const [x1, y1] = map(match.reference_xy, reference, referenceBounds);
      const [x2, y2] = map(match.source_xy, source, sourceBounds);
      const state = match.review_state === "rejected" ? "rejected" : match.inlier ? "inlier" : "outlier";
      const selected = match.index === selectedMatchIndex ? "selected" : "";
      evidenceLayer.insertAdjacentHTML("beforeend", `<line class="match-line ${state} ${selected}" data-match-index="${match.index}" x1="${x1}" y1="${y1}" x2="${x2}" y2="${y2}"/><circle class="match-end ${state}" cx="${x1}" cy="${y1}" r="2.5"/><circle class="match-end ${state}" cx="${x2}" cy="${y2}" r="2.5"/>`);
    });
    evidenceLayer.querySelectorAll<SVGLineElement>("[data-match-index]").forEach((line) => {
      line.onclick = () => {
        selectedMatchIndex = Number(line.dataset.matchIndex);
        selectTab("registration");
        renderMatchList();
        renderEvidence();
      };
    });
  }
}

async function openProject(project: ProjectRecord): Promise<void> {
  const request = {
    projectId: project.id,
    sequence: projectLoadSequence + 1,
  };
  projectLoadSequence = request.sequence;
  activeProject = project;
  activePlan = null;
  activeSetup = null;
  activeGlobalMap = null;
  activePairIndex = 0;
  manualSide = "reference";
  viewMode = "compare";
  tool = "inspect";
  viewports = { reference: freshViewport(), source: freshViewport() };
  measurement = null;
  activeObjectId = null;
  activeRegistration = null;
  activeHistory = null;
  activeDensePreview = null;
  activeDenseScene = null;
  activeStitching = null;
  activeStitchGroupId = null;
  activeMetrology = null;
  calibratedImageIds = new Set();
  const runEssentialButton = element<HTMLButtonElement>("#runEssential");
  runEssentialButton.disabled = false;
  runEssentialButton.textContent = "Rodar Essential";
  metrologyLines = [];
  floorHint = null;
  floorMeasurement = null;
  verticalMeasurement = null;
  pendingVerticalBase = null;
  metricError = null;
  metrologyError = null;
  renderPlan();
  renderSetup();
  renderGlobalMap();
  renderDenseSceneReadiness();
  renderStitching();
  renderPairWorkspace();
  const result = await Promise.all([
    loadPlan(request.projectId),
    loadSetup(request.projectId),
    loadGlobalMap(request.projectId),
    loadDenseScene(request.projectId),
    loadStitching(request.projectId),
  ]);
  if (!projectRequestIsCurrent(request, {
    projectId: activeProject?.id ?? null,
    sequence: projectLoadSequence,
  })) return;
  [activePlan, activeSetup, activeGlobalMap, activeDenseScene, activeStitching] = result;
  const calibrations = await Promise.all(project.images.map((image) => loadFloorCalibration(project.id, image.id)));
  if (!projectRequestIsCurrent(request, {
    projectId: activeProject?.id ?? null,
    sequence: projectLoadSequence,
  })) return;
  calibratedImageIds = new Set(calibrations.filter((record): record is FloorCalibrationRecord => Boolean(record)).map((record) => record.image_id));
  const copy = workflowCopy(project.workflow);
  element<HTMLElement>("#crumb").textContent = project.name;
  element<HTMLElement>("#workflowBadge").textContent = copy.badge;
  renderProjects();
  renderPlan();
  renderSetup();
  renderGlobalMap();
  renderDenseAnnotations();
  renderStitching();
  renderPairWorkspace();
  renderPairStrip();
  activeFlowStage = isPerspectiveProject(project)
    ? "single-view"
    : project.status === "built"
      ? "navigate"
      : calibratedImageIds.size < project.images.length
        ? "single-view"
        : activeGlobalMap?.success && !activeGlobalMap.stale
          ? "adjust"
          : "multi-view";
  renderFlow();
  selectFlowStage(activeFlowStage);
  await loadRegistrationForPair();
  if (!projectRequestIsCurrent(request, {
    projectId: activeProject?.id ?? null,
    sequence: projectLoadSequence,
  })) return;
  await loadMetrologyForEdit();
  renderFlow();
}

async function loadRegistrationForPair(): Promise<void> {
  const pair = currentPair();
  if (!activeProject || !pair) {
    pairLoadSequence += 1;
    activeRegistration = null;
    activeHistory = null;
    activeDensePreview = null;
    return;
  }
  const request = {
    projectId: activeProject.id,
    pairId: pair.id,
    sequence: pairLoadSequence + 1,
  };
  pairLoadSequence = request.sequence;
  const result = await Promise.all([
    loadPairRegistration(request.projectId, request.pairId),
    loadRegistrationHistory(request.projectId, request.pairId),
    loadSetup(request.projectId),
    loadGlobalMap(request.projectId),
    loadDensePreview(request.projectId, request.pairId),
    loadDenseScene(request.projectId),
  ]);
  if (!pairRequestIsCurrent(request, {
    projectId: activeProject?.id ?? null,
    pairId: currentPair()?.id ?? null,
    sequence: pairLoadSequence,
  })) return;
  [activeRegistration, activeHistory, activeSetup, activeGlobalMap, activeDensePreview, activeDenseScene] = result;
  selectedMatchIndex = null;
  renderSetup();
  renderGlobalMap();
  renderPairWorkspace();
  renderObjectList();
  renderDenseAnnotations();
  renderFlow();
}

async function refreshSetupAndHistory(): Promise<void> {
  const pair = currentPair();
  if (!activeProject) return;
  const request = {
    projectId: activeProject.id,
    pairId: pair?.id ?? null,
    sequence: pairLoadSequence + 1,
  };
  pairLoadSequence = request.sequence;
  const requests: [Promise<SceneSetup>, Promise<RegistrationHistory | null>, Promise<GlobalMapRecord | null>, Promise<DensePreviewRecord | null>, Promise<DenseSceneRecord | null>] = [
    loadSetup(request.projectId),
    pair ? loadRegistrationHistory(request.projectId, pair.id) : Promise.resolve(null),
    loadGlobalMap(request.projectId),
    pair ? loadDensePreview(request.projectId, pair.id) : Promise.resolve(null),
    loadDenseScene(request.projectId),
  ];
  const result = await Promise.all(requests);
  if (
    request.projectId !== activeProject?.id
    || request.pairId !== (currentPair()?.id ?? null)
    || request.sequence !== pairLoadSequence
  ) return;
  [activeSetup, activeHistory, activeGlobalMap, activeDensePreview, activeDenseScene] = result;
  renderSetup();
  renderGlobalMap();
  renderRevisionEvidence();
  renderDenseAnnotations();
  renderFlow();
}

function openDialog(): void {
  element<HTMLElement>("#formError").textContent = "";
  projectDialog.showModal();
}

function stagePoint(event: PointerEvent): { x: number; y: number } {
  const edit = currentEditImage();
  const imageBounds = edit?.element.getBoundingClientRect();
  if (!edit || !imageBounds) return { x: 0, y: 0 };
  return {
    x: Math.max(0, Math.min(edit.element.offsetWidth, ((event.clientX - imageBounds.left) / imageBounds.width) * edit.element.offsetWidth)),
    y: Math.max(0, Math.min(edit.element.offsetHeight, ((event.clientY - imageBounds.top) / imageBounds.height) * edit.element.offsetHeight)),
  };
}

function sourcePoints(points: Array<{ x: number; y: number }>): Array<[number, number]> {
  const edit = currentEditImage();
  if (!edit) return [];
  return sourcePixelPoints(points, { width: edit.element.offsetWidth, height: edit.element.offsetHeight }, edit.record);
}

function cacheProject(project: ProjectRecord): void {
  projects = replaceCachedRecord(projects, project);
  activeProject = replaceActiveRecord(activeProject, project);
  renderProjects();
}

function adoptProject(project: ProjectRecord): void {
  cacheProject(project);
  renderDenseAnnotations();
  renderObjectList();
  renderPairWorkspace();
}

function invalidateGlobalMapForEvidence(): number {
  globalMapSequence = invalidatePendingAction(globalMapSequence);
  if (activeGlobalMap) activeGlobalMap = { ...activeGlobalMap, stale: true };
  const button = element<HTMLButtonElement>("#runGlobalMap");
  button.textContent = "Executar mapa";
  renderGlobalMap();
  renderFlow();
  return globalMapSequence;
}

async function restoreGlobalMapAfterFailedEvidence(
  projectId: string,
  sequence: number,
  error: unknown | null,
): Promise<void> {
  const request = { projectId, sequence };
  if (!projectRequestIsCurrent(request, {
    projectId: activeProject?.id ?? null,
    sequence: globalMapSequence,
  })) return;
  try {
    const related = await Promise.all([
      loadSetup(projectId),
      loadGlobalMap(projectId),
      loadDenseScene(projectId),
    ]);
    if (!projectRequestIsCurrent(request, {
      projectId: activeProject?.id ?? null,
      sequence: globalMapSequence,
    })) return;
    [activeSetup, activeGlobalMap, activeDenseScene] = related;
    renderSetup();
    renderGlobalMap();
    renderDenseSceneReadiness();
    renderPlan();
    renderFlow();
  } catch {
    // Preserve the original save failure; a later workspace reload retries state.
  }
  if (error !== null && projectRequestIsCurrent(request, {
    projectId: activeProject?.id ?? null,
    sequence: globalMapSequence,
  })) {
    element<HTMLElement>("#toolHint").textContent = error instanceof Error ? error.message : "Falha ao salvar evidência";
  }
}

async function saveObjectBox(points: Array<{ x: number; y: number }>): Promise<void> {
  const edit = currentEditImage();
  if (!activeProject || !edit) return;
  const request = {
    projectId: activeProject.id,
    imageId: edit.record.id,
    projectSequence: projectLoadSequence,
  };
  const stored = sourcePoints(points);
  if (stored.length !== 2) return;
  const [a, b] = stored;
  const bbox: [number, number, number, number] = [Math.min(a[0], b[0]), Math.min(a[1], b[1]), Math.max(a[0], b[0]), Math.max(a[1], b[1])];
  const objectId = element<HTMLInputElement>("#objectId").value.trim() || "objeto-a";
  const label = element<HTMLInputElement>("#objectLabel").value.trim() || "Objeto A";
  const mapSequence = invalidateGlobalMapForEvidence();
  let project: ProjectRecord;
  try {
    project = await enqueueEvidenceMutation(() => addObjectRegion({
      projectId: request.projectId,
      imageId: request.imageId,
      objectId,
      label,
      bbox,
    }));
  } catch (error) {
    await restoreGlobalMapAfterFailedEvidence(request.projectId, mapSequence, error);
    return;
  }
  cacheProject(project);
  if (
    activeProject?.id !== request.projectId
    || currentEditImage()?.record.id !== request.imageId
    || request.projectSequence !== projectLoadSequence
  ) return;
  adoptProject(project);
  manualSide = manualSide === "reference" ? "source" : "reference";
  renderPairWorkspace();
  await refreshSetupAndHistory();
}

async function saveManualMatchPoint(point: { x: number; y: number }): Promise<void> {
  const pair = currentPair();
  if (!activeProject || !pair) return;
  const [stored] = sourcePoints([point]);
  if (!stored) return;
  if (manualSide === "reference") {
    pendingManualReference = stored;
    manualSide = "source";
    element<HTMLElement>("#toolHint").textContent = "Referência marcada. Clique o mesmo ponto na vizinha.";
    renderPairWorkspace();
    return;
  }
  if (!pendingManualReference) {
    manualSide = "reference";
    element<HTMLElement>("#toolHint").textContent = "Comece pela referência.";
    renderPairWorkspace();
    return;
  }
  const reference = pendingManualReference;
  const request = {
    projectId: activeProject.id,
    pairId: pair.id,
    projectSequence: projectLoadSequence,
  };
  const mapSequence = invalidateGlobalMapForEvidence();
  let project: ProjectRecord;
  try {
    project = await enqueueEvidenceMutation(() => addManualCorrespondence({
      projectId: request.projectId,
      pairId: request.pairId,
      reference,
      source: stored,
    }));
  } catch (error) {
    await restoreGlobalMapAfterFailedEvidence(request.projectId, mapSequence, error);
    return;
  }
  cacheProject(project);
  if (
    activeProject?.id !== request.projectId
    || currentPair()?.id !== request.pairId
    || request.projectSequence !== projectLoadSequence
  ) return;
  pendingManualReference = null;
  manualSide = "reference";
  adoptProject(project);
  element<HTMLElement>("#toolHint").textContent = "Match manual salvo. Reestime para aplicá-lo.";
  await refreshSetupAndHistory();
}

async function saveDenseEvidence(points: Array<{ x: number; y: number }>): Promise<void> {
  const edit = currentEditImage();
  if (!activeProject || !edit || !tool.startsWith("dense_")) return;
  const request = {
    projectId: activeProject.id,
    imageId: edit.record.id,
    projectSequence: projectLoadSequence,
  };
  const kindMap: Record<string, DenseAnnotation["kind"]> = {
    dense_include: "include_region",
    dense_exclude: "exclude_region",
    dense_surface: "surface",
    dense_anchor: "range_anchor",
  };
  const kind = kindMap[tool];
  const source = sourcePoints(points);
  const label = element<HTMLInputElement>("#denseLabel").value.trim() || "dense-evidence";
  const radialRange = kind === "range_anchor" ? Number(element<HTMLInputElement>("#denseRange").value) : undefined;
  const mapSequence = invalidateGlobalMapForEvidence();
  let project: ProjectRecord;
  try {
    project = await enqueueEvidenceMutation(() => addDenseAnnotation({
      projectId: request.projectId,
      imageId: request.imageId,
      kind,
      points: source,
      label,
      radialRange,
    }));
  } catch (error) {
    await restoreGlobalMapAfterFailedEvidence(request.projectId, mapSequence, error);
    return;
  }
  cacheProject(project);
  if (
    activeProject?.id !== request.projectId
    || currentEditImage()?.record.id !== request.imageId
    || request.projectSequence !== projectLoadSequence
  ) return;
  adoptProject(project);
  await refreshSetupAndHistory();
  renderDenseAnnotations();
}

async function updateMatchReview(matchIndex: number, state: "accepted" | "rejected"): Promise<void> {
  const pair = currentPair();
  if (!activeProject || !pair || !activeRegistration) return;
  const request = {
    projectId: activeProject.id,
    pairId: pair.id,
    projectSequence: projectLoadSequence,
  };
  const mapSequence = invalidateGlobalMapForEvidence();
  try {
    await enqueueMatchReviewMutation(async () => {
      if (
        activeProject?.id !== request.projectId
        || currentPair()?.id !== request.pairId
        || projectLoadSequence !== request.projectSequence
        || !activeRegistration
      ) {
        await restoreGlobalMapAfterFailedEvidence(request.projectId, mapSequence, null);
        return;
      }
      const result = await reviewMatch({
        projectId: request.projectId,
        pairId: request.pairId,
        matchIndex,
        state,
        revision: activeRegistration.revision,
      });
      if (
        activeProject?.id !== request.projectId
        || currentPair()?.id !== request.pairId
        || projectLoadSequence !== request.projectSequence
      ) return;
      activeRegistration = result;
      await loadRegistrationForPair();
      if (
        activeProject?.id !== request.projectId
        || currentPair()?.id !== request.pairId
        || projectLoadSequence !== request.projectSequence
      ) return;
      renderRegistrationSummary(pair);
      renderEvidence();
    });
  } catch (error) {
    await restoreGlobalMapAfterFailedEvidence(request.projectId, mapSequence, error);
  }
}

document.querySelectorAll<HTMLButtonElement>("[data-tool]").forEach((button) => {
  button.onclick = () => {
    if (button.dataset.tool === "manual_match") {
      pendingManualReference = null;
      manualSide = "reference";
    }
    setTool(button.dataset.tool as typeof tool);
    renderPairWorkspace();
  };
});

document.querySelectorAll<HTMLButtonElement>("[data-view]").forEach((button) => {
  button.onclick = () => setViewMode(button.dataset.view as typeof viewMode);
});

document.querySelectorAll<HTMLButtonElement>("[data-evidence]").forEach((button) => {
  button.onclick = () => setEvidenceMode(button.dataset.evidence as "detections" | "matches" | "inliers");
});

document.querySelectorAll<HTMLButtonElement>("[data-match-filter]").forEach((button) => {
  button.onclick = () => {
    matchFilter = button.dataset.matchFilter as typeof matchFilter;
    document.querySelectorAll<HTMLButtonElement>("[data-match-filter]").forEach((item) => item.classList.toggle("active", item === button));
    renderMatchList();
  };
});

document.querySelectorAll<HTMLButtonElement>("[data-manual-side]").forEach((button) => {
  button.onclick = () => {
    manualSide = button.dataset.manualSide as typeof manualSide;
    document.querySelectorAll<HTMLButtonElement>("[data-manual-side]").forEach((item) => item.classList.toggle("active", item.dataset.manualSide === manualSide));
    void loadMetrologyForEdit();
  };
});

element<HTMLButtonElement>("#runFloorCalibration").onclick = () => void calibrateActiveFloor();
element<HTMLButtonElement>("#clearFloorDraft").onclick = () => {
  metrologyLines = [];
  floorHint = null;
  markMetrologyDraftDirty();
  metrologyError = null;
  renderMetrology();
  renderOverlay();
};
element<HTMLInputElement>("#cameraHeight").oninput = () => {
  markMetrologyDraftDirty();
  metrologyError = null;
  renderMetrology();
};

element<HTMLButtonElement>("#clearMeasurement").onclick = () => {
  measurement = null;
  renderMeasurementReadout();
  renderOverlay();
};

document.querySelectorAll<HTMLElement>("[data-pane-tools]").forEach((toolbar) => {
  const side = toolbar.dataset.paneTools as PanoramaSide;
  toolbar.querySelectorAll<HTMLButtonElement>("[data-pane-action]").forEach((button) => {
    button.onclick = (event) => {
      event.stopPropagation();
      manualSide = side;
      const action = button.dataset.paneAction;
      if (action === "pan" || action === "measure") {
        setTool(action);
      } else if (action === "zoom-out") {
        setViewportZoom(side, viewports[side].zoom - 0.25);
      } else if (action === "zoom-in") {
        setViewportZoom(side, viewports[side].zoom + 0.25);
      } else if (action === "fit") {
        resetViewport(side);
      }
      renderPairWorkspace();
    };
  });
});

pairCanvas.addEventListener("wheel", (event) => {
  if (pairCanvas.hidden) return;
  event.preventDefault();
  const pane = (event.target as Element).closest<HTMLDivElement>(".pair-pane") ?? referencePane;
  const side = sideForPane(pane);
  manualSide = side;
  const direction = event.deltaY < 0 ? 1 : -1;
  setViewportZoom(side, viewports[side].zoom + direction * 0.25, { x: event.clientX, y: event.clientY, pane });
}, { passive: false });

pairCanvas.addEventListener("pointerdown", (event) => {
  if (tool !== "pan" && tool !== "measure") return;
  const pane = (event.target as Element).closest<HTMLDivElement>(".pair-pane");
  if (!pane) return;
  event.preventDefault();
  event.stopPropagation();
  manualSide = sideForPane(pane);
  pairCanvas.setPointerCapture(event.pointerId);
  if (tool === "pan") {
    const viewport = viewports[manualSide];
    panGesture = { pointerId: event.pointerId, side: manualSide, clientX: event.clientX, clientY: event.clientY, panX: viewport.panX, panY: viewport.panY };
    pairCanvas.classList.add("is-panning");
    return;
  }
  const point = sourcePointForEvent(event, manualSide);
  const image = currentEditImage()?.record;
  if (!point || !image) return;
  measurement = { imageId: image.id, side: manualSide, start: point, end: point };
  measurementGesture = { pointerId: event.pointerId, pane };
  renderMeasurementReadout();
  renderOverlay();
}, true);

pairCanvas.addEventListener("pointermove", (event) => {
  if (panGesture?.pointerId === event.pointerId) {
    const viewport = viewports[panGesture.side];
    viewport.panX = panGesture.panX + event.clientX - panGesture.clientX;
    viewport.panY = panGesture.panY + event.clientY - panGesture.clientY;
    applyViewportTransform(panGesture.side);
    return;
  }
  if (measurementGesture?.pointerId === event.pointerId && measurement) {
    const point = sourcePointForEvent(event, measurement.side);
    if (!point) return;
    measurement.end = point;
    renderMeasurementReadout();
    renderOverlay();
  }
});

const finishViewportGesture = (event: PointerEvent): void => {
  if (panGesture?.pointerId === event.pointerId) {
    panGesture = null;
    pairCanvas.classList.remove("is-panning");
  }
  if (measurementGesture?.pointerId === event.pointerId) measurementGesture = null;
  if (pairCanvas.hasPointerCapture(event.pointerId)) pairCanvas.releasePointerCapture(event.pointerId);
};
pairCanvas.addEventListener("pointerup", finishViewportGesture);
pairCanvas.addEventListener("pointercancel", finishViewportGesture);

window.addEventListener("keydown", (event) => {
  const target = event.target as HTMLElement;
  if (target.matches("input, textarea, select")) return;
  if (event.key === "+" || event.key === "=") setViewportZoom(manualSide, viewports[manualSide].zoom + 0.25);
  else if (event.key === "-") setViewportZoom(manualSide, viewports[manualSide].zoom - 0.25);
  else if (event.key === "0") resetViewport(manualSide);
  else if (event.key.toLowerCase() === "p") setTool("pan");
  else if (event.key.toLowerCase() === "m") setTool("measure");
  else return;
  event.preventDefault();
});

constraintLayer.addEventListener("pointerdown", (event) => {
  if (tool === "inspect" || tool === "pan" || tool === "measure") return;
  const point = stagePoint(event);
  if (tool === "manual_match") {
    void saveManualMatchPoint(point);
    return;
  }
  if (tool === "dense_anchor") {
    void saveDenseEvidence([point]);
    return;
  }
  if (tool === "floor_hint" || tool === "floor_measure" || tool === "vertical_measure") {
    void applyMetrologyPoint(point);
    return;
  }
  drawingStart = point;
  constraintLayer.setPointerCapture(event.pointerId);
});

constraintLayer.addEventListener("pointermove", (event) => {
  if (!drawingStart) return;
  renderOverlay({ kind: tool, points: [drawingStart, stagePoint(event)] });
});

constraintLayer.addEventListener("pointerup", (event) => {
  if (!drawingStart) return;
  const end = stagePoint(event);
  const start = drawingStart;
  drawingStart = null;
  if (Math.hypot(end.x - start.x, end.y - start.y) < 5) { renderOverlay(); return; }
  if (tool === "object_box") void saveObjectBox([start, end]);
  else if (tool.startsWith("dense_")) void saveDenseEvidence([start, end]);
  else if (tool === "floor_a" || tool === "floor_b") {
    const [startXY, endXY] = sourcePoints([start, end]);
    if (startXY && endXY) {
      metrologyLines.push({ family: tool, start_xy: startXY, end_xy: endXY });
      metrologyError = null;
      markMetrologyDraftDirty();
      renderMetrology();
      renderOverlay();
    }
  }
});

projectForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  const submitter = (event as SubmitEvent).submitter as HTMLButtonElement | null;
  if (submitter?.value === "cancel") { projectDialog.close(); return; }
  const files = Array.from(fileInput.files ?? []);
  const workflow = selectedWorkflow();
  try {
    element<HTMLButtonElement>("#createButton").disabled = true;
    const rawMetadata = element<HTMLTextAreaElement>("#metadataInput").value;
    const metadata = workflow.startsWith("perspective_")
      ? mergeCaptureGroupMetadata(rawMetadata, files.map((file) => ({
        filename: file.name,
        captureGroup: captureGroupAssignments.get(file.name) ?? "capture-1",
      })))
      : rawMetadata;
    const candidates = sphericalPairCandidates(files.map((file) => file.name));
    const pairs = workflow.startsWith("spherical_")
      ? candidates.filter((pair) => selectedIntakePairKeys.has(intakePairKey(pair.reference_filename, pair.source_filename)))
      : [];
    if (workflow.startsWith("spherical_") && files.length >= 2 && !pairs.length) {
      throw new Error("Selecione ao menos um par de panoramas com conteúdo em comum.");
    }
    const project = await createProject({
      name: element<HTMLInputElement>("#projectName").value,
      workflow,
      files,
      metadata,
      pairs,
    });
    projects.unshift(project);
    projectDialog.close();
    projectForm.reset();
    captureGroupAssignments.clear();
    selectedIntakePairKeys.clear();
    knownIntakePairKeys.clear();
    renderCaptureGroupEditor();
    renderPairEditor();
    element<HTMLTextAreaElement>("#metadataInput").value = "{}";
    await openProject(project);
  } catch (error) {
    element<HTMLElement>("#formError").textContent = error instanceof Error ? error.message : "Falha ao criar o projeto";
  } finally {
    element<HTMLButtonElement>("#createButton").disabled = false;
  }
});

fileInput.onchange = () => {
  const files = Array.from(fileInput.files ?? []);
  element<HTMLElement>("#fileSummary").textContent = files.length ? `${files.length} arquivo(s) · ${(files.reduce((sum, file) => sum + file.size, 0) / 1024 / 1024).toFixed(1)} MiB` : "Nenhum arquivo selecionado";
  renderCaptureGroupEditor();
  renderPairEditor();
};

document.querySelectorAll<HTMLButtonElement>("[data-tab]").forEach((button) => {
  button.onclick = () => selectTab(button.dataset.tab as typeof activeTab);
});

async function selectPair(index: number): Promise<boolean> {
  if (!activeProject?.pairs.length) return false;
  const projectId = activeProject.id;
  activePairIndex = (index + activeProject.pairs.length) % activeProject.pairs.length;
  const pairId = currentPair()?.id;
  if (!pairId) return false;
  manualSide = "reference";
  pendingManualReference = null;
  measurement = null;
  viewports = { reference: freshViewport(), source: freshViewport() };
  activeObjectId = null;
  activeRegistration = null;
  activeHistory = null;
  activeDensePreview = null;
  const runEssentialButton = element<HTMLButtonElement>("#runEssential");
  runEssentialButton.disabled = false;
  runEssentialButton.textContent = "Rodar Essential";
  renderPairWorkspace();
  renderPairStrip();
  renderDenseAnnotations();
  await loadRegistrationForPair();
  if (activeProject?.id !== projectId || currentPair()?.id !== pairId) return false;
  await loadMetrologyForEdit();
  return activeProject?.id === projectId && currentPair()?.id === pairId;
}

async function selectPairAndOpenRegistration(index: number): Promise<void> {
  if (await selectPair(index)) selectTab("registration");
}

element<HTMLButtonElement>("#newProject").onclick = openDialog;
element<HTMLButtonElement>("#emptyCreate").onclick = openDialog;
element<HTMLButtonElement>("#previousImage").onclick = () => {
  void selectPair(activePairIndex - 1);
};
element<HTMLButtonElement>("#nextImage").onclick = () => {
  void selectPair(activePairIndex + 1);
};
element<HTMLButtonElement>("#runDensePreview").onclick = async () => {
  const pair = currentPair();
  if (!activeProject || !pair) return;
  const request = {
    projectId: activeProject.id,
    pairId: pair.id,
    sequence: densePreviewSequence + 1,
  };
  densePreviewSequence = request.sequence;
  const button = element<HTMLButtonElement>("#runDensePreview");
  button.disabled = true;
  button.textContent = "Processando…";
  try {
    const result = await runDensePreview(request.projectId, request.pairId);
    if (!pairRequestIsCurrent(request, {
      projectId: activeProject?.id ?? null,
      pairId: currentPair()?.id ?? null,
      sequence: densePreviewSequence,
    })) return;
    activeDensePreview = result;
    renderDensePreview();
  } catch (error) {
    if (pairRequestIsCurrent(request, {
      projectId: activeProject?.id ?? null,
      pairId: currentPair()?.id ?? null,
      sequence: densePreviewSequence,
    })) element<HTMLElement>("#densePreviewHealth").textContent = error instanceof Error ? error.message : "Falha no preview";
  } finally {
    if (pairRequestIsCurrent(request, {
      projectId: activeProject?.id ?? null,
      pairId: currentPair()?.id ?? null,
      sequence: densePreviewSequence,
    })) {
      button.textContent = "Rodar preview";
      renderDensePreview();
    }
  }
};
element<HTMLButtonElement>("#runStitching").onclick = () => {
  void executeStitching();
};
element<HTMLButtonElement>("#promoteStitching").onclick = () => {
  void executeStitchingPromotion();
};
pairSelect.onchange = () => {
  void selectPair(Number(pairSelect.value));
};
element<HTMLInputElement>("#overlayOpacity").oninput = (event) => {
  const opacity = Number((event.currentTarget as HTMLInputElement).value) / 100;
  pairCanvas.style.setProperty("--overlay-opacity", String(opacity));
};
async function executeEssential(reestimate: boolean): Promise<void> {
  if (essentialOperationRunning) return;
  const pair = currentPair();
  if (!activeProject || !pair) return;
  const request = {
    projectId: activeProject.id,
    pairId: pair.id,
    sequence: essentialSequence + 1,
  };
  essentialSequence = request.sequence;
  essentialOperationRunning = true;
  const mapSequence = invalidateGlobalMapForEvidence();
  const runButton = element<HTMLButtonElement>("#runEssential");
  const reestimateButton = element<HTMLButtonElement>("#reestimateEssential");
  const button = reestimate ? reestimateButton : runButton;
  const original = button.textContent;
  const busyControls = essentialControlState(true, Boolean(activeRegistration));
  runButton.disabled = busyControls.runDisabled;
  reestimateButton.disabled = busyControls.reestimateDisabled;
  button.textContent = reestimate ? "Reestimando…" : "Extraindo + estimando…";
  element<HTMLElement>("#registrationGuidance").textContent = "Executando o perfil público PanorAi; esta operação pode levar alguns segundos.";
  try {
    const mode = element<HTMLSelectElement>("#regionMode").value as RegionMode;
    if (reestimate) await reestimatePairRegistration(request.projectId, request.pairId, mode);
    else await runPairRegistration(request.projectId, request.pairId, mode);
    if (!pairRequestIsCurrent(request, {
      projectId: activeProject?.id ?? null,
      pairId: currentPair()?.id ?? null,
      sequence: essentialSequence,
    })) return;
    await loadRegistrationForPair();
    if (!pairRequestIsCurrent(request, {
      projectId: activeProject?.id ?? null,
      pairId: currentPair()?.id ?? null,
      sequence: essentialSequence,
    })) return;
    evidenceMode = "inliers";
    document.querySelectorAll<HTMLButtonElement>("[data-evidence]").forEach((item) => item.classList.toggle("active", item.dataset.evidence === "inliers"));
    renderRegistrationSummary(pair);
    renderPairWorkspace();
  } catch (error) {
    await restoreGlobalMapAfterFailedEvidence(request.projectId, mapSequence, error);
    if (pairRequestIsCurrent(request, {
      projectId: activeProject?.id ?? null,
      pairId: currentPair()?.id ?? null,
      sequence: essentialSequence,
    })) element<HTMLElement>("#registrationGuidance").textContent = error instanceof Error ? error.message : "Falha ao estimar o Essential";
  } finally {
    essentialOperationRunning = false;
    if (pairRequestIsCurrent(request, {
      projectId: activeProject?.id ?? null,
      pairId: currentPair()?.id ?? null,
      sequence: essentialSequence,
    })) {
      button.textContent = original;
      const controls = essentialControlState(false, Boolean(activeRegistration));
      runButton.disabled = controls.runDisabled;
      reestimateButton.disabled = controls.reestimateDisabled;
    }
  }
}

element<HTMLButtonElement>("#runEssential").onclick = () => void executeEssential(false);
element<HTMLButtonElement>("#reestimateEssential").onclick = () => void executeEssential(true);
element<HTMLButtonElement>("#runGlobalMap").onclick = async () => {
  if (!activeProject) return;
  const request = {
    projectId: activeProject.id,
    sequence: globalMapSequence + 1,
  };
  globalMapSequence = request.sequence;
  const button = element<HTMLButtonElement>("#runGlobalMap");
  const original = button.textContent;
  button.disabled = true;
  button.textContent = "Mapeando…";
  element<HTMLElement>("#mapGuidance").textContent = "Compondo rotações, tracks, centros e ajuste global com PanorAi.";
  try {
    const map = await runGlobalMap(request.projectId);
    if (!projectRequestIsCurrent(request, {
      projectId: activeProject?.id ?? null,
      sequence: globalMapSequence,
    })) return;
    const related = await Promise.all([
      loadSetup(request.projectId),
      loadDenseScene(request.projectId),
    ]);
    if (!projectRequestIsCurrent(request, {
      projectId: activeProject?.id ?? null,
      sequence: globalMapSequence,
    })) return;
    activeGlobalMap = map;
    [activeSetup, activeDenseScene] = related;
    renderSetup();
    renderGlobalMap();
    renderDenseSceneReadiness();
  } catch (error) {
    if (projectRequestIsCurrent(request, {
      projectId: activeProject?.id ?? null,
      sequence: globalMapSequence,
    })) element<HTMLElement>("#mapGuidance").textContent = error instanceof Error ? error.message : "Falha ao executar o mapa global";
  } finally {
    if (projectRequestIsCurrent(request, {
      projectId: activeProject?.id ?? null,
      sequence: globalMapSequence,
    })) {
      button.disabled = false;
      button.textContent = original;
    }
  }
};
buildButton.onclick = async () => {
  const estimatedMapCanPublish = Boolean(activeProject?.workflow === "spherical_unregistered" && activeGlobalMap?.success && !activeGlobalMap.stale);
  if (!activeProject || (!activePlan?.can_build_now && !estimatedMapCanPublish)) return;
  if (isPerspectiveProject()) {
    const projectId = activeProject.id;
    selectTab("stitching");
    await executeStitching();
    if (activeProject?.id !== projectId) return;
    element<HTMLElement>("#actionFeedback").textContent = activeStitching?.accepted_group_count
      ? `${activeStitching.accepted_group_count} grupo(s) produziram ERPs aceitos.`
      : "Os grupos foram recusados; revise o diagnóstico da costura.";
    return;
  }
  const sourceProject = activeProject;
  const request = {
    projectId: sourceProject.id,
    sequence: manifestSequence + 1,
  };
  manifestSequence = request.sequence;
  try {
    const manifest = await createSceneManifest(request.projectId);
    if (!projectRequestIsCurrent(request, {
      projectId: activeProject?.id ?? null,
      sequence: manifestSequence,
    })) return;
    const blob = new Blob([JSON.stringify(manifest, null, 2)], { type: "application/json" });
    const anchor = document.createElement("a");
    anchor.href = URL.createObjectURL(blob);
    anchor.download = `${sourceProject.name.toLowerCase().replace(/[^a-z0-9]+/g, "-")}-scene.json`;
    anchor.click();
    URL.revokeObjectURL(anchor.href);
    const publishedAt = typeof manifest.published_at === "string" ? manifest.published_at : new Date().toISOString();
    const publishedProject: ProjectRecord = { ...sourceProject, status: "built", updated_at: publishedAt };
    projects = projects.map((project) => project.id === publishedProject.id ? publishedProject : project);
    activeProject = publishedProject;
    element<HTMLElement>("#actionFeedback").textContent = "Ambiente publicado. Ele já está disponível na biblioteca navegável.";
    renderProjects();
    renderPlan();
    renderFlow();
  } catch (error) {
    if (projectRequestIsCurrent(request, {
      projectId: activeProject?.id ?? null,
      sequence: manifestSequence,
    })) element<HTMLElement>("#actionFeedback").textContent = error instanceof Error ? error.message : "Falha no manifesto";
  }
};

sourceImage.onload = () => { renderOverlay(); renderEvidence(); };
targetImage.onload = () => { renderOverlay(); renderEvidence(); };
window.addEventListener("resize", () => { renderOverlay(); renderEvidence(); });

async function initialize(): Promise<void> {
  try {
    [projects, capabilities] = await Promise.all([listProjects(), loadCapabilities()]);
    const runtime = element<HTMLElement>("#runtime");
    runtime.classList.add("ready");
    runtime.innerHTML = `<i></i><span>PanorAi ${escapeHtml(capabilities.runtime.version)}</span>`;
    renderProjects();
    renderFlow();
    if (projects[0]) await openProject(projects[0]);
  } catch (error) {
    const runtime = element<HTMLElement>("#runtime");
    runtime.classList.add("failed");
    runtime.innerHTML = `<i></i><span>${escapeHtml(error instanceof Error ? error.message : "backend indisponível")}</span>`;
  }
}

void initialize();
