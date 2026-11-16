import "./home.css";

interface SampleDescriptor { name:string; url:string; attribution:string }
interface SamplesPayload { samples:SampleDescriptor[] }
interface CatalogPayload { runtime:{panorai:{version:string}}; counts:{surfaces:number;automatic_ready:number} }

const app=document.querySelector<HTMLDivElement>("#app");if(!app)throw new Error("Missing application root");
app.innerHTML=`<div class="home-shell">
  <header class="home-nav"><a class="home-brand" href="/"><span class="home-mark"><i></i><i></i></span><span><strong>PanorAi</strong><small>Workbench</small></span></a><nav><a class="active" href="/">Visão geral</a><a href="/applications.html">Aplicações</a><a href="/catalog.html">Catálogo técnico</a><a href="/known-height-planar.html">Planos 3D</a></nav><span id="runtime">Conectando…</span></header>
  <main>
    <section class="home-hero"><div class="hero-copy"><span>SPHERICAL IMAGE PROCESSING & PROJECTION</span><h1>Visão computacional<br/>na esfera.</h1><p>Explore aplicações prontas ou componha operações públicas do PanorAi em um fluxo visual.</p><div><a class="primary" href="/applications.html">Explorar aplicações <b>→</b></a><a href="/catalog.html">Abrir catálogo técnico</a></div></div><figure><img id="heroImage" alt="Panorama interno usado como exemplo público"/><figcaption><b>ERP · 360°</b><span id="heroCaption">Amostra pública</span></figcaption></figure></section>
    <section class="home-paths" id="applications"><header><span>APLICAÇÕES</span><h2>Comece pelo que quer obter.</h2></header><div class="path-list">
      <a href="/applications.html?input=single"><span>01</span><div><strong>Uma imagem</strong><p>Projeções, filtros, regiões, features e planos.</p></div><b>→</b></a>
      <a href="/applications.html?input=pair"><span>02</span><div><strong>Duas imagens</strong><p>Matches, pose relativa, stereo e range.</p></div><b>→</b></a>
      <a href="/known-height-planar.html"><span>03</span><div><strong>Mapeamento planar</strong><p>Descoberta assistida, relações métricas e preview 3D.</p></div><b>→</b></a>
      <a href="/segmentation.html"><span>04</span><div><strong>Segmentação</strong><p>Máscaras revisáveis com CPU ou modelos opcionais.</p></div><b>→</b></a>
    </div></section>
    <section class="home-library"><div><span>API PÚBLICA</span><h2>Da exploração ao código.</h2><p>O canvas gera uma composição Python inspecionável. O catálogo expõe contratos, estabilidade, imports e disponibilidade da instalação ativa.</p></div><dl><div><dt id="surfaceCount">—</dt><dd>superfícies públicas</dd></div><div><dt id="readyCount">—</dt><dd>operações prontas</dd></div><div><dt>1 / 2</dt><dd>imagens por fluxo</dd></div></dl><a href="/catalog.html">Explorar a biblioteca <b>→</b></a></section>
  </main>
  <footer><span>PanorAi Workbench</span><span>API pública instalada · sem execução automática</span></footer>
</div>`;

async function initialize():Promise<void>{
  const [catalogResponse,samplesResponse]=await Promise.all([fetch("/api/v1/catalog"),fetch("/api/v1/samples")]);
  if(!catalogResponse.ok||!samplesResponse.ok)throw new Error("Backend indisponível");
  const catalog=await catalogResponse.json() as CatalogPayload,samples=await samplesResponse.json() as SamplesPayload,sample=samples.samples.find(item=>item.name==="Poly Haven Studio")??samples.samples[0];
  document.querySelector<HTMLElement>("#runtime")!.textContent=`PanorAi ${catalog.runtime.panorai.version}`;
  document.querySelector<HTMLElement>("#surfaceCount")!.textContent=String(catalog.counts.surfaces);
  document.querySelector<HTMLElement>("#readyCount")!.textContent=String(catalog.counts.automatic_ready);
  if(sample){document.querySelector<HTMLImageElement>("#heroImage")!.src=sample.url;document.querySelector<HTMLElement>("#heroCaption")!.textContent=`${sample.name} · ${sample.attribution}`}
}
void initialize().catch(()=>{document.querySelector<HTMLElement>("#runtime")!.textContent="Backend indisponível"});
