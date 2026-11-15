import assert from "node:assert/strict";
import test from "node:test";
import { normalizedToolDefinitions } from "../src/image-tool-rail.ts";

test("shared image rail normalizes contextual availability without hiding tools",()=>{
  const tools=normalizedToolDefinitions([
    {id:"navigate",kind:"navigate",label:"Mão"},
    {id:"point",kind:"point",label:"Ponto",enabled:false,separatorBefore:true},
  ]);
  assert.equal(tools[0].enabled,true);
  assert.equal(tools[1].enabled,false);
  assert.equal(tools[1].separatorBefore,true);
});

test("shared image rail rejects ambiguous duplicate tool ids",()=>{
  assert.throws(()=>normalizedToolDefinitions([
    {id:"point",kind:"point",label:"Ponto A"},
    {id:"point",kind:"point",label:"Ponto B"},
  ]),/Duplicate image tool id/);
});

test("shared image rail preserves projection actions outside selection state",()=>{
  const [projection]=normalizedToolDefinitions([
    {id:"projection",kind:"projection",label:"Mostrar ERP",behavior:"action"},
  ]);
  assert.equal(projection.behavior,"action");
  assert.equal(projection.enabled,true);
});
