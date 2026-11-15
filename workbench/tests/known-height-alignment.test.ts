import assert from "node:assert/strict";
import test from "node:test";
import { chartLineToErp, chartPointToErp, erpPointToChart, interactionContractFor, nearestObservationEndpoint, nearestObservationPoint, sampleErpLineArc } from "../src/spherical-interaction.ts";
import { canonicalErpPixelToRay, greatCircleNormal, academyDot } from "../src/academy-geometry.ts";

const erp={width:2048,height:1024},chart={width:900,height:600},view={yawDeg:38,pitchDeg:-24,horizontalFovDeg:64};

test("gnomonic line endpoints round-trip through canonical ERP pixel centres",()=>{
  const source:[{x:number;y:number},{x:number;y:number}]=[{x:180.25,y:210.75},{x:720.5,y:390.125}],line=chartLineToErp(source[0],source[1],chart,view,erp);
  for(let index=0;index<2;index++){
    const restored=erpPointToChart(line[index],erp,chart,view);
    assert.ok(restored);
    assert.ok(Math.abs(restored!.x-source[index].x)<1e-9);
    assert.ok(Math.abs(restored!.y-source[index].y)<1e-9);
  }
  assert.ok(line.every(([x,y])=>x>=-.5&&x<erp.width-.5&&y>=-.5&&y<erp.height-.5));
});

test("sampled ERP evidence remains on the endpoint great circle",()=>{
  const line=chartLineToErp({x:160,y:250},{x:730,y:330},chart,view,erp),normal=greatCircleNormal(canonicalErpPixelToRay({x:line[0][0],y:line[0][1]},erp),canonicalErpPixelToRay({x:line[1][0],y:line[1][1]},erp));
  for(const [x,y] of sampleErpLineArc(line,erp,80)){
    const ray=canonicalErpPixelToRay({x,y},erp);
    assert.ok(Math.abs(academyDot(normal,ray))<1e-10);
  }
});

test("endpoint adjustment selects only a nearby visible handle",()=>{
  const line=chartLineToErp({x:140,y:170},{x:700,y:410},chart,view,erp),families=[[line],[]];
  const hit=nearestObservationEndpoint(families,{x:145,y:174},erp,chart,view,12);
  assert.deepEqual(hit&&{familyIndex:hit.familyIndex,lineIndex:hit.lineIndex,endpointIndex:hit.endpointIndex},{familyIndex:0,lineIndex:0,endpointIndex:0});
  assert.equal(nearestObservationEndpoint(families,{x:450,y:20},erp,chart,view,8),null);
});

test("point adjustment selects a projected ERP observation and preserves pixel-centre round-trip",()=>{
  const source={x:328.5,y:271.25},observation=chartPointToErp(source,chart,view,erp),other=chartPointToErp({x:760,y:480},chart,view,erp);
  const hit=nearestObservationPoint([other,observation],{x:331,y:269},erp,chart,view,9);
  assert.deepEqual(hit&&{pointIndex:hit.pointIndex},{pointIndex:1});
  const restored=erpPointToChart(observation,erp,chart,view);
  assert.ok(restored);
  assert.ok(Math.abs(restored!.x-source.x)<1e-9);
  assert.ok(Math.abs(restored!.y-source.y)<1e-9);
  assert.equal(nearestObservationPoint([observation],{x:20,y:20},erp,chart,view,6),null);
});

test("catalog semantics choose reusable interaction contracts without component names",()=>{
  assert.deepEqual(interactionContractFor({semantic:"panorai:erp-line"}),{semantic:"panorai:erp-line",primitive:"line",projection:"gnomonic",coordinateSpace:"erp-pixel-centre",interpolation:"none",requiredViews:1});
  assert.equal(interactionContractFor({semantic:"panorai:mask"})?.interpolation,"nearest");
  assert.equal(interactionContractFor({kind:"spherical-correspondence"})?.requiredViews,2);
  assert.equal(interactionContractFor({kind:"ambiguous-ndarray"}),null);
});
