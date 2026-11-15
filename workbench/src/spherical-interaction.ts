import {
  canonicalErpPixelToRay,
  canonicalRayToErpPixel,
  academyDot,
  academyNormalize,
} from "./academy-geometry.ts";
import { clamp, rayToViewPixel, viewPixelToRay, wrapDegrees } from "./geometry.ts";
import type { Point2, ViewState } from "./model.ts";

export type SphericalPrimitive="point"|"line"|"box"|"polygon"|"brush"|"correspondence";
export type ProjectionMode="gnomonic"|"erp-context";
export interface RasterSize{width:number;height:number}
export type ErpPoint=[number,number];
export type ErpLine=[ErpPoint,ErpPoint];
export interface SphericalInteractionContract{
  semantic:string;
  primitive:SphericalPrimitive;
  projection:ProjectionMode;
  coordinateSpace:"erp-pixel-centre";
  interpolation:"none"|"nearest";
  requiredViews:1|2;
}
export interface SemanticInputMetadata{semantic?:string;kind?:string;grid?:string;units?:string}
export interface EndpointHit{familyIndex:number;lineIndex:number;endpointIndex:0|1;distancePx:number}
export interface PointHit{pointIndex:number;distancePx:number}

/** Resolve interaction only from explicit semantic metadata, never array shape or values. */
export function interactionContractFor(metadata:SemanticInputMetadata):SphericalInteractionContract|null{
  const token=[metadata.semantic,metadata.kind,metadata.grid].filter(Boolean).join(":").toLowerCase();
  const base={semantic:metadata.semantic??metadata.kind??"",projection:"gnomonic" as const,coordinateSpace:"erp-pixel-centre" as const};
  if(/correspondence|match/.test(token))return{...base,primitive:"correspondence",interpolation:"none",requiredViews:2};
  if(/erp[-_: ]?line|great[-_: ]?circle|line[-_: ]?family/.test(token))return{...base,primitive:"line",interpolation:"none",requiredViews:1};
  if(/erp[-_: ]?point|bearing|keypoint|seed/.test(token))return{...base,primitive:"point",interpolation:"none",requiredViews:1};
  if(/mask|labels|support|validity/.test(token))return{...base,primitive:"brush",interpolation:"nearest",requiredViews:1};
  if(/region|polygon|roi/.test(token))return{...base,primitive:"polygon",interpolation:"nearest",requiredViews:1};
  return null;
}

export function zoomSphericalViewport(view:ViewState,factor:number):ViewState{if(!Number.isFinite(factor)||factor<=0)throw new Error("Fator de zoom inválido.");return{...view,horizontalFovDeg:clamp(view.horizontalFovDeg*factor,15,120)}}
export function moveSphericalViewport(view:ViewState,start:Point2,end:Point2,width:number,mode:"navigate"|"zoom"):ViewState{
  if(mode==="zoom")return zoomSphericalViewport(view,Math.exp((end.y-start.y)*.005));const scale=view.horizontalFovDeg/Math.max(1,width);return{...view,yawDeg:wrapDegrees(view.yawDeg-(end.x-start.x)*scale),pitchDeg:clamp(view.pitchDeg+(end.y-start.y)*scale,-89.9,89.9)};
}

export function chartPointToErp(point:Point2,chart:RasterSize,view:ViewState,erp:RasterSize):ErpPoint{
  const value=canonicalRayToErpPixel(viewPixelToRay(point,chart,view),erp);return[value.x,value.y];
}
export function erpPointToChart(point:ErpPoint,erp:RasterSize,chart:RasterSize,view:ViewState):Point2|null{return rayToViewPixel(canonicalErpPixelToRay({x:point[0],y:point[1]},erp),chart,view)}
export function chartLineToErp(start:Point2,end:Point2,chart:RasterSize,view:ViewState,erp:RasterSize):ErpLine{return[chartPointToErp(start,chart,view,erp),chartPointToErp(end,chart,view,erp)]}

/** The observed finite image line is a minor great-circle arc in ERP context. */
export function sampleErpLineArc(line:ErpLine,erp:RasterSize,samples=64):ErpPoint[]{
  const first=canonicalErpPixelToRay({x:line[0][0],y:line[0][1]},erp),second=canonicalErpPixelToRay({x:line[1][0],y:line[1][1]},erp),cosine=Math.max(-1,Math.min(1,academyDot(first,second))),angle=Math.acos(cosine),sine=Math.sin(angle);
  if(angle<1e-10)return[line[0],line[1]];if(Math.abs(sine)<1e-8)throw new Error("Extremos antipodais não definem um arco observável único.");
  return Array.from({length:samples+1},(_,index)=>{const t=index/samples,a=Math.sin((1-t)*angle)/sine,b=Math.sin(t*angle)/sine,ray=academyNormalize({x:first.x*a+second.x*b,y:first.y*a+second.y*b,z:first.z*a+second.z*b}),pixel=canonicalRayToErpPixel(ray,erp);return[pixel.x,pixel.y] as ErpPoint});
}
export function nearestObservationEndpoint(families:ErpLine[][],point:Point2,erp:RasterSize,chart:RasterSize,view:ViewState,maximumDistancePx=18):EndpointHit|null{
  let best:EndpointHit|null=null;families.forEach((family,familyIndex)=>family.forEach((line,lineIndex)=>line.forEach((endpoint,rawEndpointIndex)=>{const projected=erpPointToChart(endpoint,erp,chart,view);if(!projected)return;const distancePx=Math.hypot(projected.x-point.x,projected.y-point.y);if(distancePx<=maximumDistancePx&&(!best||distancePx<best.distancePx))best={familyIndex,lineIndex,endpointIndex:rawEndpointIndex as 0|1,distancePx}})));return best;
}

/** Select a visible typed ERP point by its gnomonic handle, without inferring semantics from values. */
export function nearestObservationPoint(points:readonly ErpPoint[],point:Point2,erp:RasterSize,chart:RasterSize,view:ViewState,maximumDistancePx=18):PointHit|null{
  let best:PointHit|null=null;
  points.forEach((value,pointIndex)=>{const projected=erpPointToChart(value,erp,chart,view);if(!projected)return;const distancePx=Math.hypot(projected.x-point.x,projected.y-point.y);if(distancePx<=maximumDistancePx&&(!best||distancePx<best.distancePx))best={pointIndex,distancePx}});
  return best;
}
