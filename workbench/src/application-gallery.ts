import "./application-gallery.css";
import { APPLICATION_RECIPES, type ApplicationCategory, type ApplicationRecipe } from "./application-recipes.ts";

interface CatalogPayload { runtime:{panorai:{version:string}} }
interface SampleDescriptor { id:string;name:string;url:string;attribution:string }
interface SamplesPayload { samples:SampleDescriptor[] }

const manualImages: Partial<Record<ApplicationRecipe["visual"], string>> = {
  edges: new URL("../docs/assets/user-manual/03-single-view-resultado.png", import.meta.url).href,
  regions: new URL("../docs/assets/user-manual/03-single-view-resultado.png", import.meta.url).href,
  features: new URL("../docs/assets/user-manual/discovery-recipe.png", import.meta.url).href,
  planes: new URL("../docs/assets/user-manual/09-known-height-descoberta.png", import.meta.url).href,
  pose: new URL("../docs/assets/user-manual/05-two-view-matches.png", import.meta.url).href,
  stereo: new URL("../docs/assets/user-manual/05-two-view-matches.png", import.meta.url).href,
  depth: new URL("../docs/assets/user-manual/03-single-view-resultado.png", import.meta.url).href,
  segmentation: new URL("../docs/assets/user-manual/03-single-view-resultado.png", import.meta.url).href,
  reconstruction: new URL("../docs/assets/user-manual/06-multi-view.png", import.meta.url).href,
};

const app=document.querySelector<HTMLDivElement>("#app");if(!app)throw new Error("Missing application root");
const escapeHtml=(value:string):string=>value.replace(/[&<>'"]/g,char=>({"&":"&amp;","<":"&lt;",">":"&gt;","'":"&#39;",'"':"&quot;"})[char]??char);

app.innerHTML=`<div class="gallery-shell">
  <header class="gallery-nav"><a class="gallery-brand" href="/"><span class="gallery-mark"><i></i><i></i></span><span><strong>PanorAi</strong><small>Workbench</small></span></a><nav><a href="/">Visão geral</a><a class="active" href="/applications.html">Aplicações</a><a href="/catalog.html">Catálogo técnico</a><a href="/known-height-planar.html">Planos 3D</a></nav><span id="runtime">Conectando…</span></header>
  <main>
    <section class="gallery-intro"><div><span>APLICAÇÕES</span><h1>Escolha o problema.<br/>O fluxo já vem montado.</h1></div><p>Exemplos para pesquisa e desenvolvimento em visão computacional esférica. Abra um fluxo, inspecione as operações e adapte o pipeline.</p></section>
    <section class="gallery-controls"><div class="input-tabs" role="group" aria-label="Filtrar por entrada"><button data-input="all" class="active">Todas</button><button data-input="single">Uma imagem</button><button data-input="pair">Duas imagens</button></div><div class="category-tabs" role="group" aria-label="Filtrar por tema"><button data-category="all" class="active">Todos</button><button data-category="geometry">Geometria</button><button data-category="analysis">Análise</button><button data-category="features">Features</button><button data-category="depth">Depth</button><button data-category="3d">3D</button><button data-category="models">Modelos</button></div></section>
    <section class="application-list" id="applicationList" aria-live="polite"></section>
    <section class="gallery-catalog"><span>PRECISA DE OUTRA OPERAÇÃO?</span><div><h2>Explore a API por objeto, tipo e estabilidade.</h2><a href="/catalog.html">Abrir catálogo técnico <b>→</b></a></div></section>
  </main>
  <footer><span id="applicationCount">— aplicações</span><span>Nenhum fluxo é executado ao abrir</span></footer>
</div>`;

let inputFilter:"all"|"single"|"pair"="all",categoryFilter:"all"|ApplicationCategory="all",sampleUrl="";
const list=document.querySelector<HTMLElement>("#applicationList")!;

function imageFor(recipe:ApplicationRecipe):string{return recipe.visual==="projection"?sampleUrl:(manualImages[recipe.visual]??sampleUrl)}
function filteredRecipes():ApplicationRecipe[]{return APPLICATION_RECIPES.filter(recipe=>(inputFilter==="all"||(inputFilter==="single"?recipe.inputCount===1:recipe.inputCount===2))&&(categoryFilter==="all"||recipe.categories.includes(categoryFilter)))}
function statusLabel(recipe:ApplicationRecipe):string{return recipe.availability==="ready"?"PRONTO":recipe.availability==="specialized"?"BANCADA PRÓPRIA":"CONFIGURAÇÃO NECESSÁRIA"}
function render():void{
  const recipes=filteredRecipes();
  list.innerHTML=recipes.map((recipe)=>`<article class="application-row ${recipe.featured?"featured":""}" data-application="${escapeHtml(recipe.id)}">
    <a class="application-visual visual-${recipe.visual}" href="/canvas.html?application=${encodeURIComponent(recipe.id)}" aria-label="Abrir ${escapeHtml(recipe.title)} no canvas">
      <img src="${escapeHtml(imageFor(recipe))}" alt="Visual de ${escapeHtml(recipe.title)}"/>
      <span class="visual-index">${escapeHtml(recipe.index)}</span><span class="visual-action">Abrir fluxo <b>↗</b></span>
    </a>
    <div class="application-copy"><div class="application-meta"><span>${escapeHtml(recipe.audience)}</span><i class="availability ${recipe.availability}">${statusLabel(recipe)}</i></div><h2><a href="/canvas.html?application=${encodeURIComponent(recipe.id)}">${escapeHtml(recipe.title)}</a></h2><p>${escapeHtml(recipe.summary)}</p><small>${escapeHtml(recipe.detail)}</small><dl><div><dt>ENTRADA</dt><dd>${escapeHtml(recipe.inputLabel)}</dd></div><div><dt>RESULTADO</dt><dd>${recipe.outputs.map(escapeHtml).join(" · ")}</dd></div><div><dt>RUNTIME</dt><dd>${escapeHtml(recipe.runtimeNote)}</dd></div></dl><a class="open-flow" href="/canvas.html?application=${encodeURIComponent(recipe.id)}">Ver pipeline montado <b>→</b></a></div>
  </article>`).join("");
  document.querySelector<HTMLElement>("#applicationCount")!.textContent=`${recipes.length} aplicaç${recipes.length===1?"ão":"ões"}`;
  if(!recipes.length)list.innerHTML='<div class="gallery-empty"><strong>Nenhuma aplicação neste recorte.</strong><button id="clearFilters">Limpar filtros</button></div>';
  document.querySelector<HTMLButtonElement>("#clearFilters")?.addEventListener("click",()=>{inputFilter="all";categoryFilter="all";syncFilters();render()});
}
function syncFilters():void{
  document.querySelectorAll<HTMLButtonElement>("[data-input]").forEach(button=>button.classList.toggle("active",button.dataset.input===inputFilter));
  document.querySelectorAll<HTMLButtonElement>("[data-category]").forEach(button=>button.classList.toggle("active",button.dataset.category===categoryFilter));
}

document.querySelectorAll<HTMLButtonElement>("[data-input]").forEach(button=>button.onclick=()=>{inputFilter=button.dataset.input as typeof inputFilter;syncFilters();render()});
document.querySelectorAll<HTMLButtonElement>("[data-category]").forEach(button=>button.onclick=()=>{categoryFilter=button.dataset.category as typeof categoryFilter;syncFilters();render()});

async function initialize():Promise<void>{
  const params=new URLSearchParams(location.search),requested=params.get("input");if(requested==="single"||requested==="pair")inputFilter=requested;
  const [catalogResponse,samplesResponse]=await Promise.all([fetch("/api/v1/catalog"),fetch("/api/v1/samples")]);
  if(!catalogResponse.ok||!samplesResponse.ok)throw new Error("Backend indisponível");
  const catalog=await catalogResponse.json() as CatalogPayload,samples=await samplesResponse.json() as SamplesPayload,sample=samples.samples.find(item=>item.id==="poly-haven-studio-cc0")??samples.samples[0];
  sampleUrl=sample?.url??"";document.querySelector<HTMLElement>("#runtime")!.textContent=`PanorAi ${catalog.runtime.panorai.version}`;syncFilters();render();
}
void initialize().catch(()=>{document.querySelector<HTMLElement>("#runtime")!.textContent="Backend indisponível";syncFilters();render()});
