import type { RleMask } from "./planar-model.ts";
export type ReviewState="draft"|"accepted"|"rejected"|"needs-review";
export type ConstraintKind="floor-anchor"|"vertical-to-floor"|"horizontal-to-plane"|"orthogonal-to"|"parallel-to"|"coplanar-with";
export interface KnownHeightRegion{plane_id:string;view_id:"single-view";label:string;state:ReviewState;mask:RleMask;positive?:RleMask;negative?:RleMask;roi?:RleMask;interpretation:"seed"|"final";history:Record<string,unknown>[];source:string}
export interface KnownHeightConstraint{constraint_id:string;plane_id:string;kind:ConstraintKind;reference_plane_id:string|null;contact_erp_xy:[number,number][];state:"proposed"|"accepted"|"rejected";source:string;orientation_id?:string|null}
export interface PlanarOrientationEvidence{plane_id:string;proposal_id:string;normal_camera:[number,number,number];line_ids:string[];bundle_ids:string[];confidence:number;residual_deg:number;state:ReviewState;source:"automatic"|"human"}
export interface PlanarPreview{plane_id:string;state:"orientation-only"|"positioned-draft"|"resolved";normal_world:[number,number,number]|null;offset_m:number|null;contact_world_xyz:number[][];footprint_world_xyz:number[][];source_line_ids:string[];source_region_id:string|null;publishable:boolean;diagnostics:Record<string,unknown>}
export type ErpPoint=[number,number];
export type ErpLine=[ErpPoint,ErpPoint];
export interface DiscoveryLineLike{line_id:string;erp_endpoints_xy:ErpLine;state:string}
export interface DiscoveryBundleLike{bundle_id:string;line_ids:string[]}
export interface DiscoveryProposalLike{proposal_id?:string;normal_camera?:[number,number,number];bundle_ids:string[];line_ids?:string[];confidence?:number;source?:string;plane_id?:string|null}
export type DiscoveryPresetName="sparse"|"balanced"|"dense";
export interface DiscoveryTuning{
  atlasFaceSize:number;
  cannyLow:number;
  cannyHigh:number;
  localContrastClipLimit:number;
  minimumSegmentPx:number;
  minimumEdgeSupport:number;
  minimumBundleLines:number;
  minimumBundleSupportDeg:number;
  maximumLines:number;
  maximumProposals:number;
}
export interface KnownHeightScene{schema:string;scene_id:string;revision:string;camera_height_m:number|null;camera_height_sigma_m:number|null;down_erp_xy:ErpPoint;horizon_erp_xy:ErpPoint[];horizontal_vanishing_erp_xy:ErpPoint[];floor_line_families_erp_xy:ErpLine[][];regions:KnownHeightRegion[];constraints:KnownHeightConstraint[];orientations?:PlanarOrientationEvidence[];provenance:Record<string,unknown>}
export type PlanarWorkspaceStep="height"|"orientation"|"relation"|"segmentation"|"review"|"metric"|"ready";
export interface PlanarWorkspaceEvidence{
  activePlaneId:string;
  heightReady:boolean;
  floorOrientationReady:boolean;
  planeOrientationReady:boolean;
  relationReady:boolean;
  maskReady:boolean;
  maskAccepted:boolean;
  metricReady:boolean;
  floorMetricReady:boolean;
}
export interface PlanarWorkspacePhase{step:PlanarWorkspaceStep;label:string;state:"done"|"current"|"locked"}
export interface PlanarWorkspaceFlow{scope:"floor"|"plane";current:PlanarWorkspaceStep;phases:PlanarWorkspacePhase[];canAddPlane:boolean}
export function planarRecoveryStep(reason:string|undefined):PlanarWorkspaceStep|null{
  if(!reason)return null;
  if(reason.includes("orientation"))return"orientation";
  if(reason.includes("contact")||reason.includes("reference-plane")||reason.includes("cycle"))return"relation";
  if(reason.includes("mask"))return"review";
  return"metric";
}
export function planarContactPreviewMessage(detail:unknown):string{
  const value=String(detail??"").toLowerCase();
  if(value.includes("tangent"))return"O ponto está perto demais do horizonte. Aponte mais abaixo no piso para obter uma interseção estável.";
  if(value.includes("in front of the camera"))return"Este ponto não cruza o piso à frente da câmera. Escolha uma região visível do piso.";
  return"Não foi possível posicionar o plano neste ponto. Tente outro ponto visível no plano de referência.";
}
export function derivePlanarWorkspaceFlow(evidence:PlanarWorkspaceEvidence):PlanarWorkspaceFlow{
  const floor=evidence.activePlaneId==="floor";
  const definitions:Array<[PlanarWorkspaceStep,string,boolean]>=floor?[
    ["height","Altura",evidence.heightReady],
    ["orientation","Orientar piso",evidence.floorOrientationReady],
    ["segmentation","Segmentar piso",evidence.maskReady],
    ["review","Revisar máscara",evidence.maskAccepted],
    ["relation","Confirmar piso",evidence.relationReady],
    ["metric","Gerar mapa",evidence.metricReady],
  ]:[
    ["orientation","Orientar plano",evidence.planeOrientationReady],
    ["relation","Posicionar no piso",evidence.relationReady],
    ["segmentation","Segmentar superfície",evidence.maskReady],
    ["review","Revisar máscara",evidence.maskAccepted],
    ["metric","Publicar plano",evidence.metricReady],
  ];
  const pending=definitions.findIndex(([, ,done])=>!done),current=pending<0?"ready":definitions[pending][0];
  return{
    scope:floor?"floor":"plane",
    current,
    phases:[...definitions.map(([step,label,done],index)=>({step,label,state:done?"done":index===pending?"current":"locked"} as PlanarWorkspacePhase)),...(pending<0?[{step:"ready" as const,label:floor?"Piso pronto":"Plano pronto",state:"current" as const}]:[])],
    canAddPlane:evidence.floorMetricReady,
  };
}
export function orientationEvidenceId(planeId:string,proposalId:string):string{return `orientation:${encodeURIComponent(planeId)}:${encodeURIComponent(proposalId)}`}
export function migrateKnownHeightScene(value:KnownHeightScene):KnownHeightScene{
  const orientations=Array.isArray(value.orientations)?value.orientations:[],counts=new Map<string,number>();
  for(const item of orientations)counts.set(item.proposal_id,(counts.get(item.proposal_id)??0)+1);
  const replacements=new Map<string,string>();
  const migratedOrientations=orientations.map(item=>{
    if((counts.get(item.proposal_id)??0)<2)return item;
    const replacement=orientationEvidenceId(item.plane_id,item.proposal_id);
    replacements.set(`${item.plane_id}\u0000${item.proposal_id}`,replacement);
    return{...item,proposal_id:replacement};
  });
  const constraints=value.constraints.map(item=>{
    const orientationId=item.orientation_id??null;
    const replacement=orientationId===null?undefined:replacements.get(`${item.plane_id}\u0000${orientationId}`);
    return{...item,orientation_id:replacement??orientationId};
  });
  return{...value,schema:"panorai-known-height-planar-scene/v2",orientations:migratedOrientations,constraints};
}
export function revise(scene:KnownHeightScene,action:string,mutate:(copy:KnownHeightScene)=>void):KnownHeightScene{const copy=structuredClone(scene);mutate(copy);const version=Number(scene.revision.match(/(\d+)$/)?.[1]??0)+1;copy.revision=`workbench-${version}`;copy.provenance={...copy.provenance,last_human_action:action,parent_revision:scene.revision};return copy}
export function acceptedPlaneIds(scene:KnownHeightScene):string[]{return scene.regions.filter(r=>r.state==="accepted"&&r.interpretation==="final").map(r=>r.plane_id)}
export function suggestedConstraint(scene:KnownHeightScene,planeId:string):KnownHeightConstraint{const accepted=acceptedPlaneIds(scene),floor=accepted.find(id=>scene.constraints.some(c=>c.plane_id===id&&c.kind==="floor-anchor"&&c.state==="accepted")),reference=floor??accepted[0]??null;const kind:ConstraintKind=reference?"orthogonal-to":"floor-anchor";return{constraint_id:`${planeId}-${kind}`,plane_id:planeId,kind,reference_plane_id:reference,contact_erp_xy:[],state:"proposed",source:"workbench-suggestion",orientation_id:null}}
export function canAcceptConstraint(item:KnownHeightConstraint):boolean{if(item.kind==="floor-anchor"||item.kind==="coplanar-with")return true;if(item.kind==="parallel-to")return item.contact_erp_xy.length>=1;if(item.kind==="orthogonal-to"&&item.orientation_id)return item.contact_erp_xy.length>=1;return item.contact_erp_xy.length>=2}
export function dependentPlaneIds(constraints:Iterable<KnownHeightConstraint>,rootPlaneId:string):string[]{
  const values=[...constraints],affected=new Set([rootPlaneId]);
  let changed=true;
  while(changed){changed=false;for(const item of values)if(item.reference_plane_id&&affected.has(item.reference_plane_id)&&!affected.has(item.plane_id)){affected.add(item.plane_id);changed=true}}
  return [...affected];
}
export function numericSigmaParameter(value:number|null,fallback=0.02):number{return value===null?fallback:value}
export function hasMeasuredHeight(scene:KnownHeightScene):boolean{return scene.camera_height_m!==null&&Number.isFinite(scene.camera_height_m)&&scene.camera_height_m>0}
export function hasHorizontalAlignmentEvidence(scene:KnownHeightScene):boolean{return scene.horizontal_vanishing_erp_xy.length>=2||scene.horizon_erp_xy.length>=2||scene.floor_line_families_erp_xy.filter(family=>family.length>=2).length>=2}
export function lineRefinementRequiresDecision(status:string,hasRecommendation:boolean):boolean{return status==="suggested"&&hasRecommendation}
export function canDrawPlanarLine(mode:"assisted"|"manual",reviewBlocks:boolean,discoveryBusy=false):boolean{
  return(mode==="assisted"||mode==="manual")&&!reviewBlocks&&!discoveryBusy;
}
export function guidedLineEvidenceActions(selectedLineIds:Iterable<string>,planeLineIds:Iterable<string>){
  const selected=[...new Set(selectedLineIds)],associated=new Set(planeLineIds);
  const newlyAssociated=selected.filter(id=>!associated.has(id));
  return{
    selectedCount:selected.length,
    associatedCount:associated.size,
    newlyAssociatedCount:newlyAssociated.length,
    canAssociate:newlyAssociated.length>0,
    canEstimateDirection:selected.length>=2&&selected.every(id=>associated.has(id)),
  };
}
export type DirectionLineVisualState="selected"|"direction-member"|"expansion-candidate"|"plane-member"|"other";
export function directionLineVisualState(
  lineId:string,
  selectedLineIds:Iterable<string>,
  directionLineIds:Iterable<string>,
  candidateLineIds:Iterable<string>,
  planeLineIds:Iterable<string>,
):DirectionLineVisualState{
  const has=(values:Iterable<string>)=>values instanceof Set?values.has(lineId):new Set(values).has(lineId);
  if(has(selectedLineIds))return"selected";
  if(has(directionLineIds))return"direction-member";
  if(has(candidateLineIds))return"expansion-candidate";
  if(has(planeLineIds))return"plane-member";
  return"other";
}
export function acceptDirectionCandidate(directionLineIds:Iterable<string>,planeLineIds:Iterable<string>,lineId:string){
  return{
    directionLineIds:[...new Set([...directionLineIds,lineId])].sort(),
    planeLineIds:[...new Set([...planeLineIds,lineId])].sort(),
  };
}
export function findMatchingOrientationProposal<T extends DiscoveryProposalLike>(proposals:Iterable<T>,bundleIds:Iterable<string>):T|undefined{
  const expected=[...new Set(bundleIds)].sort();
  if(expected.length<2)return undefined;
  return [...proposals].find(proposal=>{
    const actual=[...new Set(proposal.bundle_ids)].sort();
    return actual.length===expected.length&&actual.every((id,index)=>id===expected[index]);
  });
}
export function relevantOrientationProposals<T extends DiscoveryProposalLike>(
  proposals:Iterable<T>,
  planeId:string,
  selectedBundleIds:Iterable<string>,
  acceptedProposalId?:string,
):T[]{
  const values=[...proposals],matching=findMatchingOrientationProposal(values,selectedBundleIds),acceptedExists=acceptedProposalId!==undefined&&values.some(proposal=>proposal.proposal_id===acceptedProposalId),chosenId=acceptedExists?acceptedProposalId:matching?.proposal_id;
  if(chosenId)return values.filter(proposal=>proposal.proposal_id===chosenId);
  return values.filter(proposal=>proposal.plane_id===planeId);
}
export function planarDiscoveryPreset(name:DiscoveryPresetName):DiscoveryTuning{
  if(name==="sparse")return{atlasFaceSize:256,cannyLow:60,cannyHigh:160,localContrastClipLimit:0,minimumSegmentPx:32,minimumEdgeSupport:.55,minimumBundleLines:3,minimumBundleSupportDeg:10,maximumLines:240,maximumProposals:12};
  if(name==="dense")return{atlasFaceSize:512,cannyLow:18,cannyHigh:70,localContrastClipLimit:3,minimumSegmentPx:10,minimumEdgeSupport:.2,minimumBundleLines:2,minimumBundleSupportDeg:5,maximumLines:600,maximumProposals:36};
  return{atlasFaceSize:384,cannyLow:25,cannyHigh:105,localContrastClipLimit:2,minimumSegmentPx:14,minimumEdgeSupport:.3,minimumBundleLines:3,minimumBundleSupportDeg:8,maximumLines:480,maximumProposals:24};
}
export function normalizePlanarDiscoveryTuning(value:DiscoveryTuning):DiscoveryTuning{
  const finite=(candidate:number,fallback:number)=>Number.isFinite(candidate)?candidate:fallback;
  const clampValue=(candidate:number,minimum:number,maximum:number)=>Math.min(maximum,Math.max(minimum,candidate));
  const atlasFaceSize=Math.round(clampValue(finite(value.atlasFaceSize,384),128,1024));
  const cannyLow=clampValue(finite(value.cannyLow,50),1,254);
  const cannyHigh=clampValue(finite(value.cannyHigh,140),cannyLow+1,255);
  return{
    ...value,
    atlasFaceSize,
    cannyLow,
    cannyHigh,
    localContrastClipLimit:clampValue(finite(value.localContrastClipLimit,2),0,8),
    minimumSegmentPx:clampValue(finite(value.minimumSegmentPx,24),4,256),
    minimumEdgeSupport:clampValue(finite(value.minimumEdgeSupport,.45),.05,1),
    minimumBundleLines:Math.round(clampValue(finite(value.minimumBundleLines,3),2,12)),
    minimumBundleSupportDeg:clampValue(finite(value.minimumBundleSupportDeg,8),.1,180),
    maximumLines:Math.round(clampValue(finite(value.maximumLines,480),1,5000)),
    maximumProposals:Math.round(clampValue(finite(value.maximumProposals,24),1,500)),
  };
}
export function floorFamiliesFromDiscovery(
  proposal:DiscoveryProposalLike,
  bundles:DiscoveryBundleLike[],
  lines:DiscoveryLineLike[],
):ErpLine[][]{
  if(proposal.bundle_ids.length<2)throw new Error("A proposta precisa de pelo menos dois bundles");
  const lineById=new Map(lines.map(line=>[line.line_id,line]));
  const bundleById=new Map(bundles.map(bundle=>[bundle.bundle_id,bundle]));
  return proposal.bundle_ids.slice(0,2).map(bundleId=>{
    const bundle=bundleById.get(bundleId);
    if(!bundle)throw new Error(`Bundle ${bundleId} não está disponível nesta revisão`);
    const family=bundle.line_ids.map(lineId=>lineById.get(lineId)).filter((line):line is DiscoveryLineLike=>line!==undefined&&line.state!=="rejected").slice(0,2);
    if(family.length<2)throw new Error(`Bundle ${bundleId} precisa de duas retas revisadas`);
    return family.map(line=>structuredClone(line.erp_endpoints_xy));
  });
}
