export type ImageToolKind="navigate"|"zoom"|"point"|"line"|"adjust"|"undo"|"brush"|"polygon"|"projection";
export interface ImageToolDefinition{id:string;kind:ImageToolKind;label:string;title?:string;enabled?:boolean;separatorBefore?:boolean;behavior?:"select"|"action"}
export interface ImageToolRailOptions{tools:ImageToolDefinition[];active?:string|null;label?:string;onSelect?:(id:string)=>void}
export interface ImageToolRailController{
  element:HTMLElement;
  setActive(id:string|null):void;
  setEnabled(id:string,enabled:boolean):void;
  setReadout(title:string|null,detail?:string):void;
  setToolLabel(id:string,label:string,title?:string):void;
  showExtension(content:HTMLElement):void;
  hideExtension():void;
  replaceTools(tools:ImageToolDefinition[]):void;
  destroy():void;
}

const icons:Record<ImageToolKind,string>={
  navigate:'<path d="M7 12V6a1.5 1.5 0 0 1 3 0v5-7a1.5 1.5 0 0 1 3 0v7-6a1.5 1.5 0 0 1 3 0v6-4a1.5 1.5 0 0 1 3 0v8c0 4-2 6-6 6h-1c-2 0-3-1-4-3l-4-5c-1-2 1-3 2-2l2 2Z"/>',
  zoom:'<circle cx="10" cy="10" r="6"/><path d="m15 15 6 6M7 10h6m-3-3v6"/>',
  point:'<path d="M12 3v4m0 10v4M3 12h4m10 0h4"/><circle cx="12" cy="12" r="2.4"/>',
  line:'<path d="M5 18 19 6"/><circle cx="5" cy="18" r="2"/><circle cx="19" cy="6" r="2"/>',
  adjust:'<path d="M4 7h8m4 0h4M4 17h4m4 0h8"/><circle cx="14" cy="7" r="2"/><circle cx="10" cy="17" r="2"/>',
  undo:'<path d="M9 7 4 12l5 5"/><path d="M5 12h8c4 0 7 2 7 6"/>',
  brush:'<path d="m14 4 6 6-9 9H5v-6Z"/><path d="m12 6 6 6"/>',
  polygon:'<path d="m5 6 13-2 2 13-10 3-6-7Z"/><circle cx="5" cy="6" r="1.4"/><circle cx="18" cy="4" r="1.4"/><circle cx="20" cy="17" r="1.4"/><circle cx="10" cy="20" r="1.4"/><circle cx="4" cy="13" r="1.4"/>',
  projection:'<path d="M3 5h18v14H3Z"/><path d="M8 5c-2 4-2 10 0 14m8-14c2 4 2 10 0 14M3 12h18"/>',
};

export function normalizedToolDefinitions(tools:ImageToolDefinition[]):ImageToolDefinition[]{
  const ids=new Set<string>();
  return tools.map(tool=>{if(!tool.id.trim())throw new Error("Image tool id must not be empty.");if(ids.has(tool.id))throw new Error(`Duplicate image tool id: ${tool.id}`);ids.add(tool.id);return{...tool,enabled:tool.enabled!==false}});
}

export function mountImageToolRail(host:HTMLElement,options:ImageToolRailOptions):ImageToolRailController{
  host.classList.add("image-tool-host");
  const element=document.createElement("div");element.className="image-tool-rail";element.setAttribute("role","toolbar");element.setAttribute("aria-label",options.label??"Ferramentas da imagem");
  const readout=document.createElement("div");readout.className="image-tool-readout";readout.hidden=true;readout.setAttribute("aria-live","polite");
  const extension=document.createElement("div");extension.className="image-tool-extension";extension.hidden=true;
  let active=options.active??null,buttons=new Map<string,HTMLButtonElement>();
  const replaceTools=(raw:ImageToolDefinition[])=>{const tools=normalizedToolDefinitions(raw);element.replaceChildren();buttons=new Map();for(const tool of tools){if(tool.separatorBefore){const separator=document.createElement("span");separator.className="image-tool-separator";separator.setAttribute("aria-hidden","true");element.append(separator)}const button=document.createElement("button");button.type="button";button.className="image-tool-button";button.dataset.imageTool=tool.id;button.dataset.kind=tool.kind;button.dataset.behavior=tool.behavior??"select";button.title=tool.title??tool.label;button.setAttribute("aria-label",tool.label);button.setAttribute("aria-pressed",String(tool.behavior!=="action"&&tool.id===active));button.disabled=tool.enabled===false;button.classList.toggle("active",tool.behavior!=="action"&&tool.id===active);button.innerHTML=`<svg viewBox="0 0 24 24" aria-hidden="true">${icons[tool.kind]}</svg>`;button.onpointerdown=event=>event.stopPropagation();button.onclick=()=>{if(button.disabled)return;if(button.dataset.behavior!=="action"){active=tool.id;for(const [id,item] of buttons){const selected=id===active&&item.dataset.behavior!=="action";item.classList.toggle("active",selected);item.setAttribute("aria-pressed",String(selected))}}options.onSelect?.(tool.id)};buttons.set(tool.id,button);element.append(button)}element.append(readout,extension)};
  replaceTools(options.tools);host.append(element);
  return{element,setActive(id){active=id;for(const [key,button] of buttons){const selected=key===id&&button.dataset.behavior!=="action";button.classList.toggle("active",selected);button.setAttribute("aria-pressed",String(selected))}},setEnabled(id,enabled){const button=buttons.get(id);if(button)button.disabled=!enabled},setReadout(title,detail=""){readout.hidden=!title;readout.innerHTML=title?`<strong>${title}</strong><span>${detail}</span>`:""},setToolLabel(id,label,title){const button=buttons.get(id);if(!button)return;button.setAttribute("aria-label",label);button.title=title??label},showExtension(content){extension.replaceChildren(content);extension.hidden=false},hideExtension(){extension.hidden=true;extension.replaceChildren()},replaceTools,destroy(){element.remove();host.classList.remove("image-tool-host")}};
}
