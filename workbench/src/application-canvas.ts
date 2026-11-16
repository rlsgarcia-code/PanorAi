import "./application-canvas.css";
import { storeDiscoveryAsset, storeRecipeDraft } from "./discovery-input-store.ts";
import type { DiscoveryInput, RecipeDraft } from "./discovery-model.ts";
import { getApplicationRecipe, type ApplicationRecipe } from "./application-recipes.ts";

interface FieldSchema { name:string; default:unknown; kind:string; upstream?:boolean }
interface Component { id:string; symbol:string; title:string; summary?:string; public_api:string; execution_kind:string; status:string; configuration_schema:FieldSchema[]; input_schema:FieldSchema[] }
interface CatalogPayload { components:Component[]; runtime:{panorai:{version:string}} }
interface Sample { id:string;name:string;shape_hw:[number,number];url:string;filename:string;attribution:string }
interface Pair { id:string;name:string;shape_hw:[number,number];image_a_url:string;image_b_url:string;filename_a:string;filename_b:string }
interface SamplesPayload { samples:Sample[];pairs:Pair[] }
type NodeKind="input"|"operation"|"output";
interface FlowNode { id:string;kind:NodeKind;title:string;x:number;y:number;parents:string[];component?:Component;role?:"a"|"b";publicApi?:string;note?:string;recipeStepId?:string }

const app=document.querySelector<HTMLDivElement>("#app");if(!app)throw new Error("Missing application root");
app.innerHTML=`<div class="flow-shell"><header class="flow-topbar"><a class="flow-brand" href="/"><span class="flow-mark"><i></i><i></i></span><span><strong>PanorAi Workbench</strong><small>Canvas</small></span></a><nav><a href="/">Visão geral</a><a class="active" href="/applications.html">Aplicações</a><a href="/catalog.html">Catálogo técnico</a><a href="/known-height-planar.html">Planos 3D</a></nav><span id="runtime">Conectando…</span></header>
<main class="flow-workspace"><aside class="flow-library"><a class="back-applications" href="/applications.html">← Todas as aplicações</a><header><span>PIPELINE PRONTO</span><h1 id="recipeTitle">Carregando fluxo…</h1><p id="recipeSummary"></p></header><div class="input-mode"><button id="oneInput" class="active">Uma imagem</button><button id="twoInputs">Duas imagens</button></div><div class="preset-actions"><button data-preset="project">Projeção</button><button data-preset="features">Features</button><button data-preset="blank">Fluxo vazio</button></div><label class="operation-search"><span>Adicionar operação</span><input id="operationSearch" type="search" placeholder="projeção, SLIC, pose…"/></label><div class="operation-list" id="operationList"></div></aside>
<section class="flow-main"><header class="flow-toolbar"><div><button id="canvasMode" class="active">Canvas</button><button id="previewMode">Visualização</button></div><span id="flowState">Preset editável</span><div><button id="codeButton">Código Python</button><button id="executeButton" class="primary">Revisar e executar</button></div></header><div class="flow-canvas" id="flowCanvas"><svg id="edgeLayer" aria-hidden="true"></svg><div id="nodeLayer"></div><div class="canvas-help">Arraste os nós · selecione para editar conexões</div></div><div class="flow-preview" id="flowPreview" hidden><div class="preview-images" id="previewImages"></div><div><span>VISUALIZAÇÃO</span><strong id="previewTitle">Entrada</strong><small>Expanda o resultado de qualquer etapa sem perder o fluxo.</small></div></div></section>
<aside class="flow-inspector"><div id="inspectorEmpty"><span>INSPECTOR</span><strong>Selecione um nó</strong></div><div id="inspector" hidden></div></aside></main>
<aside class="code-drawer" id="codeDrawer" hidden><header><span>CÓDIGO PYTHON EQUIVALENTE</span><button id="closeCode">×</button></header><pre id="pythonCode"></pre><footer><span>Imports públicos · revise parâmetros antes de usar</span><button id="copyCode">Copiar código</button></footer></aside></div>`;

const get=<T extends Element>(selector:string):T=>{const value=document.querySelector<T>(selector);if(!value)throw new Error(`Missing ${selector}`);return value};
const escapeHtml=(value:string):string=>value.replace(/[&<>'"]/g,char=>({"&":"&amp;","<":"&lt;",">":"&gt;","'":"&#39;",'"':"&quot;"})[char]??char);
interface DragState { id:string;pointerId:number;grabX:number;grabY:number }
const CANVAS_PADDING_X=12,CANVAS_PADDING_Y=18,FALLBACK_NODE_WIDTH=170,FALLBACK_NODE_HEIGHT=88;
let catalog:CatalogPayload|null=null,samples:SamplesPayload|null=null,nodes:FlowNode[]=[],selectedId:string|null=null,inputCount:1|2=1,activePreset="project",activeRecipe:ApplicationRecipe|null=null,assets:DiscoveryInput[]=[],assetBlobs=new Map<string,Blob>(),dragState:DragState|null=null;
const canvas=get<HTMLElement>("#flowCanvas"),nodeLayer=get<HTMLElement>("#nodeLayer"),edgeLayer=get<SVGElement>("#edgeLayer"),operationList=get<HTMLElement>("#operationList"),inspector=get<HTMLElement>("#inspector"),inspectorEmpty=get<HTMLElement>("#inspectorEmpty"),previewImages=get<HTMLElement>("#previewImages");

function uid(prefix:string):string{return`${prefix}-${Math.random().toString(36).slice(2,8)}`}
function readyComponents():Component[]{return(catalog?.components??[]).filter(component=>component.status==="ready")}
function findComponent(pattern:RegExp):Component|undefined{return readyComponents().find(component=>pattern.test(component.symbol)||pattern.test(component.title))}
function findRecipeComponent(patterns:string[]):Component|undefined{const normalized=patterns.map(value=>value.toLocaleLowerCase("en"));return readyComponents().find(component=>{const haystack=`${component.symbol} ${component.title} ${component.public_api}`.toLocaleLowerCase("en");return normalized.some(pattern=>haystack.includes(pattern.toLocaleLowerCase("en")))})}
function inputNodes():FlowNode[]{return nodes.filter(node=>node.kind==="input")}
function operationNodes():FlowNode[]{return nodes.filter(node=>node.kind==="operation")}
function nodeById(id:string):FlowNode|undefined{return nodes.find(node=>node.id===id)}
function nodeElement(id:string):HTMLElement|null{return nodeLayer.querySelector<HTMLElement>(`[data-node="${CSS.escape(id)}"]`)}
function updateSelectedNode():void{nodeLayer.querySelectorAll<HTMLElement>("[data-node]").forEach(element=>element.classList.toggle("selected",element.dataset.node===selectedId))}
function clampNode(node:FlowNode,element:HTMLElement,clientX:number,clientY:number,grabX:number,grabY:number):void{
  const rect=canvas.getBoundingClientRect(),width=element.offsetWidth||FALLBACK_NODE_WIDTH,height=element.offsetHeight||FALLBACK_NODE_HEIGHT;
  const maxX=Math.max(CANVAS_PADDING_X,rect.width-width-CANVAS_PADDING_X),maxY=Math.max(CANVAS_PADDING_Y,rect.height-height-CANVAS_PADDING_Y);
  node.x=Math.max(CANVAS_PADDING_X,Math.min(maxX,clientX-rect.left-grabX));
  node.y=Math.max(CANVAS_PADDING_Y,Math.min(maxY,clientY-rect.top-grabY));
}
function clampNodesToCanvas():void{
  if(canvas.hidden||canvas.clientWidth===0||canvas.clientHeight===0)return;
  const rect=canvas.getBoundingClientRect();
  nodes.forEach(node=>{const element=nodeElement(node.id);if(!element)return;clampNode(node,element,rect.left+node.x,rect.top+node.y,0,0);element.style.transform=`translate(${node.x}px,${node.y}px)`});
  renderEdges()
}
function defaultGraph(preset:string):void{
  activeRecipe=null;
  activePreset=preset;selectedId=null;const canvasWidth=Math.max(canvas.clientWidth,520),inputX=40,outputX=Math.max(inputX,canvasWidth-FALLBACK_NODE_WIDTH-CANVAS_PADDING_X),operationX=Math.round((inputX+outputX)/2),inputs:FlowNode[]=[{id:"input-a",kind:"input",title:"Imagem A",x:inputX,y:150,parents:[],role:"a"}];if(inputCount===2)inputs.push({id:"input-b",kind:"input",title:"Imagem B",x:inputX,y:330,parents:[],role:"b"});
  const operations:FlowNode[]=[];
  if(preset!=="blank"){
    const patterns=inputCount===2?[/(ExtractSphericalFeaturePair|RelativePose|Match)/i]:preset==="features"?[/SphericalDogFeatures|DoG/i]:[/equirectangular_to_gnomonic/i];
    patterns.forEach((pattern,index)=>{const component=findComponent(pattern);if(component)operations.push({id:uid("step"),kind:"operation",title:component.title||component.symbol,x:operationX+index*220,y:inputCount===2?230:180,parents:inputs.map(node=>node.id),component})});
  }
  const previous=operations.at(-1)?.id??inputs[0].id;nodes=[...inputs,...operations,{id:"output",kind:"output",title:"Resultado",x:outputX,y:inputCount===2?230:180,parents:[previous]}];renderAll();clampNodesToCanvas()
}
function recipeGraph(recipe:ApplicationRecipe):void{
  activeRecipe=recipe;activePreset=recipe.id;inputCount=recipe.inputCount;selectedId=null;
  get<HTMLButtonElement>("#oneInput").classList.toggle("active",inputCount===1);get<HTMLButtonElement>("#twoInputs").classList.toggle("active",inputCount===2);
  get<HTMLElement>("#recipeTitle").textContent=recipe.title;get<HTMLElement>("#recipeSummary").textContent=recipe.summary;
  const canvasWidth=Math.max(canvas.clientWidth,720),inputX=32,outputX=Math.max(520,canvasWidth-FALLBACK_NODE_WIDTH-CANVAS_PADDING_X),inputs:FlowNode[]=[{id:"input-a",kind:"input",title:"Imagem A",x:inputX,y:inputCount===2?115:190,parents:[],role:"a"}];
  if(inputCount===2)inputs.push({id:"input-b",kind:"input",title:"Imagem B",x:inputX,y:315,parents:[],role:"b"});
  const operationPositions=recipe.steps.length===1?[Math.round((inputX+outputX)/2)]:recipe.steps.map((_,index)=>index===0?inputX+195:outputX-190),operations=recipe.steps.map((step,index):FlowNode=>({id:step.id,kind:"operation",title:step.title,x:operationPositions[index],y:inputCount===2?215:190,parents:[...step.parents],component:findRecipeComponent(step.componentPatterns),publicApi:step.publicApi,note:step.note,recipeStepId:step.id}));
  nodes=[...inputs,...operations,{id:"output",kind:"output",title:recipe.outputs.join(" · "),x:outputX,y:inputCount===2?215:190,parents:[operations.at(-1)?.id??inputs[0].id]}];
  updatePrimaryAction();renderAll();clampNodesToCanvas()
}
function renderNodes():void{
  nodeLayer.innerHTML=nodes.map(node=>`<button class="flow-node ${node.kind} ${node.id===selectedId?"selected":""} ${node.kind==="operation"&&!node.component?"inspect-only":""}" data-node="${escapeHtml(node.id)}" style="transform:translate(${node.x}px,${node.y}px)"><i></i><span>${node.kind==="input"?`INPUT ${node.role?.toUpperCase()}`:node.kind==="output"?"OUTPUT":escapeHtml(node.component?.execution_kind.toUpperCase()??"API PÚBLICA")}</span><strong>${escapeHtml(node.title)}</strong>${node.component||node.publicApi?`<small>${escapeHtml(node.component?.public_api??node.publicApi??"")}</small>`:""}<b></b></button>`).join("");
  nodeLayer.querySelectorAll<HTMLElement>("[data-node]").forEach(element=>{
    element.onpointerdown=event=>{
      if(event.button!==0)return;
      const node=nodeById(element.dataset.node!);if(!node)return;
      const rect=canvas.getBoundingClientRect();selectedId=node.id;
      dragState={id:node.id,pointerId:event.pointerId,grabX:event.clientX-rect.left-node.x,grabY:event.clientY-rect.top-node.y};
      element.setPointerCapture(event.pointerId);element.classList.add("dragging");updateSelectedNode();renderInspector();renderEdges();event.preventDefault()
    };
    element.onpointermove=event=>{
      if(!dragState||dragState.id!==element.dataset.node||dragState.pointerId!==event.pointerId)return;
      const node=nodeById(dragState.id);if(!node)return;
      clampNode(node,element,event.clientX,event.clientY,dragState.grabX,dragState.grabY);
      element.style.transform=`translate(${node.x}px,${node.y}px)`;renderEdges()
    };
    const finishDrag=(event:PointerEvent):void=>{
      if(!dragState||dragState.id!==element.dataset.node||dragState.pointerId!==event.pointerId)return;
      element.classList.remove("dragging");dragState=null
    };
    element.onpointerup=event=>{finishDrag(event);if(element.hasPointerCapture(event.pointerId))element.releasePointerCapture(event.pointerId)};
    element.onpointercancel=finishDrag;element.onlostpointercapture=finishDrag
  })
}
function renderEdges():void{
  const width=canvas.clientWidth,height=canvas.clientHeight;edgeLayer.setAttribute("viewBox",`0 0 ${width} ${height}`);edgeLayer.innerHTML=nodes.flatMap(target=>target.parents.map(parentId=>{const source=nodeById(parentId);if(!source)return"";const sourceElement=nodeElement(source.id),targetElement=nodeElement(target.id),sourceWidth=sourceElement?.offsetWidth||FALLBACK_NODE_WIDTH,sourceHeight=sourceElement?.offsetHeight||FALLBACK_NODE_HEIGHT,targetHeight=targetElement?.offsetHeight||FALLBACK_NODE_HEIGHT,x1=source.x+sourceWidth,y1=source.y+sourceHeight/2,x2=target.x,y2=target.y+targetHeight/2,curve=Math.max(45,(x2-x1)*.45);return`<path d="M ${x1} ${y1} C ${x1+curve} ${y1}, ${x2-curve} ${y2}, ${x2} ${y2}"/><circle cx="${x1}" cy="${y1}" r="3"/><circle cx="${x2}" cy="${y2}" r="3"/>`})).join("")
}
function renderOperations():void{
  const query=get<HTMLInputElement>("#operationSearch").value.toLocaleLowerCase("pt-BR");
  const values=readyComponents().filter(item=>`${item.title} ${item.symbol} ${item.public_api}`.toLocaleLowerCase("pt-BR").includes(query)).slice(0,query?30:8);
  operationList.innerHTML=values.map(item=>`<button data-component="${escapeHtml(item.id)}"><span>${escapeHtml(item.execution_kind)}</span><strong>${escapeHtml(item.title||item.symbol)}</strong><small>${escapeHtml(item.public_api)}</small><b>+</b></button>`).join("");
  operationList.querySelectorAll<HTMLButtonElement>("[data-component]").forEach(button=>{button.onclick=()=>{
    const component=readyComponents().find(item=>item.id===button.dataset.component);if(!component)return;
    const output=nodeById("output")!,previous=output.parents[0]??inputNodes()[0].id;
    const node:FlowNode={id:uid("step"),kind:"operation",title:component.title||component.symbol,x:Math.max(270,output.x-230),y:output.y,parents:[previous],component};
    output.parents=[node.id];output.x=Math.max(output.x,node.x+250);nodes.splice(nodes.indexOf(output),0,node);selectedId=node.id;renderAll()
  }});
}
function renderInspector():void{
  const node=selectedId?nodeById(selectedId):undefined;inspector.hidden=!node;inspectorEmpty.hidden=Boolean(node);if(!node)return;
  const candidates=nodes.filter(candidate=>candidate.id!==node.id&&candidate.kind!=="output"),asset=node.kind==="input"?assets.find(item=>item.role===node.role):undefined;
  const publicApi=node.component?.public_api??node.publicApi;
  inspector.innerHTML=`<header><span>${node.kind.toUpperCase()}</span><h2>${escapeHtml(node.title)}</h2>${publicApi?`<code>${escapeHtml(publicApi)}</code>`:""}</header>${asset?`<section><span>IMAGEM</span><p>${escapeHtml(asset.name??asset.id)}</p><button class="replace-input" id="replaceInput">Trocar imagem</button><input id="replaceInputFile" type="file" accept="image/png,image/jpeg,image/webp" hidden/></section>`:""}<section><span>RECEBE DE</span><div class="parent-list">${node.kind==="input"?"<small>Entrada raiz do fluxo</small>":candidates.map(candidate=>`<label><input type="checkbox" data-parent="${escapeHtml(candidate.id)}" ${node.parents.includes(candidate.id)?"checked":""}/><i></i>${escapeHtml(candidate.title)}</label>`).join("")}</div></section>${publicApi?`<section><span>CONTRATO</span><p>${escapeHtml(node.note??node.component?.summary??"Operação derivada da API pública instalada.")}</p>${node.component?`<dl><div><dt>Tipo</dt><dd>${escapeHtml(node.component.execution_kind)}</dd></div><div><dt>Configuração</dt><dd>${node.component.configuration_schema.length} campo(s)</dd></div><div><dt>Inputs</dt><dd>${node.component.input_schema.length} campo(s)</dd></div></dl>`:'<div class="runtime-warning">Visível no pipeline, mas sem adapter automático nesta instalação.</div>'}</section>`:""}${node.kind==="operation"?'<button class="remove-node" id="removeNode">Remover do fluxo</button>':""}`;
  inspector.querySelectorAll<HTMLInputElement>("[data-parent]").forEach(input=>input.onchange=()=>{if(input.checked){if(!node.parents.includes(input.dataset.parent!))node.parents.push(input.dataset.parent!)}else node.parents=node.parents.filter(id=>id!==input.dataset.parent);renderEdges();renderCode()});inspector.querySelector<HTMLButtonElement>("#removeNode")?.addEventListener("click",()=>{const replacement=node.parents[0]??inputNodes()[0].id;nodes.forEach(candidate=>candidate.parents=candidate.parents.map(id=>id===node.id?replacement:id));nodes=nodes.filter(candidate=>candidate.id!==node.id);selectedId=null;renderAll()})
  const replaceButton=inspector.querySelector<HTMLButtonElement>("#replaceInput"),replaceFile=inspector.querySelector<HTMLInputElement>("#replaceInputFile");
  if(replaceButton&&replaceFile&&asset){replaceButton.onclick=()=>replaceFile.click();replaceFile.onchange=()=>{const file=replaceFile.files?.[0];if(!file)return;const descriptor:DiscoveryInput={...asset,id:node.role==="b"?"erp-b":"erp-a",source:"upload",name:file.name,mime_type:file.type||"application/octet-stream"};assets=assets.map(item=>item.role===node.role?descriptor:item);assetBlobs.set(descriptor.id,file);void storeDiscoveryAsset(descriptor,file);renderInspector();renderPreview()}}
}
function pythonIdentifier(value:string):string{return value.replace(/[^a-zA-Z0-9_]/g,"_").replace(/^\d/,"_$&").toLowerCase()}
function renderCode():void{
  if(activeRecipe){get<HTMLElement>("#pythonCode").textContent=activeRecipe.python;return}
  const operations=operationNodes(),imports=new Map<string,string[]>();
  operations.forEach(node=>{if(!node.component)return;const parts=node.component.public_api.split("."),symbol=parts.pop()!,module=parts.join(".");imports.set(module,[...new Set([...(imports.get(module)??[]),symbol])])});
  const hasGnomonic=operations.some(node=>node.component?.symbol==="equirectangular_to_gnomonic");
  if(hasGnomonic)imports.set("panorai.geometry",[...new Set([...(imports.get("panorai.geometry")??[]),"GnomonicSpec"])]);
  const lines=["import numpy as np","from PIL import Image",...Array.from(imports,([module,symbols])=>`from ${module} import ${symbols.join(", ")}`),"",'erp_a = np.asarray(Image.open("image_a.jpg").convert("RGB"), dtype=np.float32) / 255.0'];
  if(inputCount===2)lines.push('erp_b = np.asarray(Image.open("image_b.jpg").convert("RGB"), dtype=np.float32) / 255.0');
  lines.push("");
  operations.forEach((node,index)=>{
    const component=node.component!,symbol=component.public_api.split(".").at(-1)!,variable=pythonIdentifier(`result_${index+1}`),parents=node.parents.map(id=>{const parent=nodeById(id);if(parent?.kind==="input")return parent.role==="b"?"erp_b":"erp_a";const parentIndex=operations.findIndex(item=>item.id===id);return parentIndex>=0?pythonIdentifier(`result_${parentIndex+1}`):"erp_a"});
    if(symbol==="equirectangular_to_gnomonic")lines.push(`${variable} = ${symbol}(`,`    ${parents[0]??"erp_a"},`,`    GnomonicSpec(output_shape_hw=(1024, 1024)),`,`    interpolation="bilinear",`,`)`);
    else if(component.execution_kind==="stage"||component.execution_kind==="application")lines.push(`${variable} = ${symbol}()(${parents.join(", ")})`);
    else lines.push(`# Revise os parâmetros públicos de ${symbol}`,`${variable} = ${symbol}(${parents.join(", ")})`)
  });
  if(operations.length)lines.push("",`result = ${pythonIdentifier(`result_${operations.length}`)}`);else lines.push("result = erp_a");
  get<HTMLElement>("#pythonCode").textContent=lines.join("\n")
}
function renderAll():void{renderNodes();renderEdges();renderInspector();renderOperations();renderCode();get<HTMLElement>("#flowState").textContent=activeRecipe?`${activeRecipe.title} · ${operationNodes().length} etapas`:`${inputCount} input${inputCount===1?"":"s"} · ${operationNodes().length} operação${operationNodes().length===1?"":"ões"}`}
async function loadInputs():Promise<void>{if(!samples)return;if(inputCount===1){const sample=samples.samples.find(item=>item.id==="poly-haven-studio-cc0")??samples.samples[0],blob=await fetch(sample.url).then(response=>response.blob()),descriptor:DiscoveryInput={id:"erp-a",kind:"erp",role:"a",source:"sample",name:sample.name,shape_hw:sample.shape_hw,mime_type:blob.type,semantic:"panorai:rgb-erp"};assets=[descriptor];assetBlobs=new Map([[descriptor.id,blob]]);await storeDiscoveryAsset(descriptor,blob)}else{const pair=samples.pairs[0],aBlob=await fetch(pair.image_a_url).then(response=>response.blob()),bBlob=await fetch(pair.image_b_url).then(response=>response.blob()),a:DiscoveryInput={id:"erp-a",kind:"erp",role:"a",source:"sample",name:pair.filename_a,shape_hw:pair.shape_hw,mime_type:aBlob.type,semantic:"panorai:rgb-erp"},b:DiscoveryInput={id:"erp-b",kind:"erp",role:"b",source:"sample",name:pair.filename_b,shape_hw:pair.shape_hw,mime_type:bBlob.type,semantic:"panorai:rgb-erp"};assets=[a,b];assetBlobs=new Map([[a.id,aBlob],[b.id,bBlob]]);await Promise.all([storeDiscoveryAsset(a,aBlob),storeDiscoveryAsset(b,bBlob)])}renderPreview()}
function renderPreview():void{previewImages.innerHTML=assets.map(asset=>`<figure><img src="${URL.createObjectURL(assetBlobs.get(asset.id)!)}" alt="${escapeHtml(asset.name??asset.id)}"/><figcaption>${escapeHtml(asset.role.toUpperCase())} · ${escapeHtml(asset.name??asset.id)}</figcaption></figure>`).join("");get<HTMLElement>("#previewTitle").textContent=selectedId?nodeById(selectedId)?.title??"Resultado":"Entradas do fluxo"}
async function setInputCount(value:1|2):Promise<void>{inputCount=value;activeRecipe=null;get<HTMLButtonElement>("#oneInput").classList.toggle("active",value===1);get<HTMLButtonElement>("#twoInputs").classList.toggle("active",value===2);get<HTMLElement>("#recipeTitle").textContent="Fluxo livre";get<HTMLElement>("#recipeSummary").textContent="Combine operações públicas com a entrada escolhida.";updatePrimaryAction();await loadInputs();defaultGraph(value===2?"pose":"project")}
function updatePrimaryAction():void{const button=get<HTMLButtonElement>("#executeButton");button.disabled=false;if(activeRecipe?.availability==="setup"){button.textContent="Ver configuração"}else if(activeRecipe?.route){button.textContent="Abrir bancada"}else{button.textContent="Revisar e executar"}}
function handoff():void{const operations=operationNodes();if(activeRecipe?.availability==="setup"){renderCode();get<HTMLElement>("#codeDrawer").hidden=false;return}if(activeRecipe?.route){location.href=activeRecipe.route;return}if(!operations.length||operations.some(node=>!node.component)){get<HTMLElement>("#flowState").textContent="Este fluxo ainda precisa de um adapter executável";return}const operationIds=new Set(operations.map(node=>node.id)),connections=Object.fromEntries(operations.map(node=>[node.id,node.parents.filter(id=>operationIds.has(id))]));const draft:RecipeDraft={schema:"panorai-workbench-recipe-draft/v1",id:`canvas-${Date.now()}`,recipe_id:`canvas-${activePreset}`,title:activeRecipe?.title??`Fluxo visual · ${operations.length} operação${operations.length===1?"":"ões"}`,created_at:new Date().toISOString(),steps:operations.map(node=>({id:node.id,component_id:node.component!.id,title:node.title,public_api:node.component!.public_api})),connections,input_ids:assets.map(asset=>asset.id),source_route:null};storeRecipeDraft(draft);location.href=`/runner.html?component=${encodeURIComponent(operations[0].component!.id)}&draft=${encodeURIComponent(draft.id)}`}
async function initialize():Promise<void>{const [catalogResponse,samplesResponse]=await Promise.all([fetch("/api/v1/catalog"),fetch("/api/v1/samples")]);if(!catalogResponse.ok||!samplesResponse.ok)throw new Error("Backend indisponível");catalog=await catalogResponse.json() as CatalogPayload;samples=await samplesResponse.json() as SamplesPayload;get<HTMLElement>("#runtime").textContent=`PanorAi ${catalog.runtime.panorai.version}`;const params=new URLSearchParams(location.search),applicationId=params.get("application");if(applicationId){const recipe=getApplicationRecipe(applicationId);inputCount=recipe.inputCount;await loadInputs();recipeGraph(recipe);return}inputCount=params.get("inputs")==="2"?2:1;activePreset=params.get("preset")??(inputCount===2?"pose":"project");get<HTMLElement>("#recipeTitle").textContent="Fluxo livre";get<HTMLElement>("#recipeSummary").textContent="Combine operações públicas com a entrada escolhida.";await loadInputs();defaultGraph(activePreset);updatePrimaryAction()}
get<HTMLInputElement>("#operationSearch").oninput=renderOperations;get<HTMLButtonElement>("#oneInput").onclick=()=>void setInputCount(1);get<HTMLButtonElement>("#twoInputs").onclick=()=>void setInputCount(2);document.querySelectorAll<HTMLButtonElement>("[data-preset]").forEach(button=>button.onclick=()=>{get<HTMLElement>("#recipeTitle").textContent=button.dataset.preset==="blank"?"Fluxo vazio":button.textContent??"Fluxo livre";get<HTMLElement>("#recipeSummary").textContent="Combine operações públicas com a entrada escolhida.";defaultGraph(button.dataset.preset!);updatePrimaryAction()});get<HTMLButtonElement>("#canvasMode").onclick=()=>{canvas.hidden=false;get<HTMLElement>("#flowPreview").hidden=true;get<HTMLButtonElement>("#canvasMode").classList.add("active");get<HTMLButtonElement>("#previewMode").classList.remove("active");clampNodesToCanvas()};get<HTMLButtonElement>("#previewMode").onclick=()=>{renderPreview();canvas.hidden=true;get<HTMLElement>("#flowPreview").hidden=false;get<HTMLButtonElement>("#previewMode").classList.add("active");get<HTMLButtonElement>("#canvasMode").classList.remove("active")};get<HTMLButtonElement>("#codeButton").onclick=()=>{renderCode();get<HTMLElement>("#codeDrawer").hidden=false};get<HTMLButtonElement>("#closeCode").onclick=()=>get<HTMLElement>("#codeDrawer").hidden=true;get<HTMLButtonElement>("#copyCode").onclick=async()=>navigator.clipboard.writeText(get<HTMLElement>("#pythonCode").textContent??"");get<HTMLButtonElement>("#executeButton").onclick=handoff;window.addEventListener("resize",clampNodesToCanvas);void initialize().catch(error=>{get<HTMLElement>("#flowState").textContent=error instanceof Error?error.message:"Falha ao carregar"});
