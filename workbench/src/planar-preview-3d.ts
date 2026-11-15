import * as THREE from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";
import type { PlanarPreview } from "./known-height-model.ts";

const COLORS={"orientation-only":0x8a9187,"positioned-draft":0xffc857,resolved:0x5fd6d0};

export class PlanarPreview3D{
  private renderer:THREE.WebGLRenderer;
  private scene=new THREE.Scene();
  private camera=new THREE.PerspectiveCamera(48,1,.01,1000);
  private controls:OrbitControls;
  private content=new THREE.Group();
  private observer:ResizeObserver;
  private pointerStart:THREE.Vector2|null=null;
  constructor(private host:HTMLElement,private onSelect?:(planeId:string)=>void){
    this.renderer=new THREE.WebGLRenderer({antialias:true,alpha:false});
    this.renderer.setPixelRatio(Math.min(devicePixelRatio,2));this.renderer.setClearColor(0x050706,1);host.append(this.renderer.domElement);
    this.camera.position.set(4,3.2,5);this.controls=new OrbitControls(this.camera,this.renderer.domElement);this.controls.enableDamping=true;
    this.scene.add(new THREE.AxesHelper(1.2),this.content);const light=new THREE.HemisphereLight(0xffffff,0x20241f,2);this.scene.add(light);
    const cameraBody=new THREE.Mesh(new THREE.BoxGeometry(.18,.12,.28),new THREE.MeshBasicMaterial({color:0xf5f1e7,wireframe:true}));cameraBody.name="camera";this.scene.add(cameraBody);
    const loop=()=>{this.controls.update();this.renderer.render(this.scene,this.camera);requestAnimationFrame(loop)};loop();
    this.renderer.domElement.addEventListener("pointerdown",event=>{this.pointerStart=new THREE.Vector2(event.clientX,event.clientY)});
    this.renderer.domElement.addEventListener("pointerup",event=>{
      if(!this.pointerStart||Math.hypot(event.clientX-this.pointerStart.x,event.clientY-this.pointerStart.y)>5)return;
      const bounds=this.renderer.domElement.getBoundingClientRect(),pointer=new THREE.Vector2((event.clientX-bounds.left)/bounds.width*2-1,-((event.clientY-bounds.top)/bounds.height)*2+1),raycaster=new THREE.Raycaster();
      raycaster.setFromCamera(pointer,this.camera);const hit=raycaster.intersectObjects(this.content.children,true).find(item=>typeof item.object.userData.planeId==="string");
      if(hit)this.onSelect?.(hit.object.userData.planeId as string);
    });
    this.observer=new ResizeObserver(()=>this.resize());this.observer.observe(host);this.resize();
  }
  private resize(){const width=Math.max(1,this.host.clientWidth),height=Math.max(1,this.host.clientHeight);this.renderer.setSize(width,height,false);this.camera.aspect=width/height;this.camera.updateProjectionMatrix()}
  update(previews:PlanarPreview[],activePlane:string){
    this.content.clear();
    for(const preview of previews){
      const color=COLORS[preview.state],normal=preview.normal_world?new THREE.Vector3(...preview.normal_world).normalize():null,active=preview.plane_id===activePlane;
      if(preview.state==="orientation-only"&&normal){
        const arrow=new THREE.ArrowHelper(normal,new THREE.Vector3(),active?1.5:1,color,0.18,.1);arrow.userData.planeId=preview.plane_id;this.content.add(arrow);
        const ghost=new THREE.Mesh(new THREE.PlaneGeometry(1.4,1.4,7,7),new THREE.MeshBasicMaterial({color,wireframe:true,transparent:true,opacity:.22,side:THREE.DoubleSide}));
        ghost.position.copy(normal.clone().multiplyScalar(.6));ghost.quaternion.setFromUnitVectors(new THREE.Vector3(0,0,1),normal);ghost.userData.planeId=preview.plane_id;this.content.add(ghost);continue;
      }
      if(!normal||preview.offset_m===null)continue;
      const origin=normal.clone().multiplyScalar(-preview.offset_m),grid=new THREE.GridHelper(active?5:3,12,color,color);
      grid.material.transparent=true;grid.material.opacity=preview.state==="resolved"?.62:.28;grid.quaternion.setFromUnitVectors(new THREE.Vector3(0,1,0),normal);grid.position.copy(origin);grid.userData.planeId=preview.plane_id;this.content.add(grid);
      for(const point of preview.contact_world_xyz){const marker=new THREE.Mesh(new THREE.SphereGeometry(.06,12,8),new THREE.MeshBasicMaterial({color:0xffc857}));marker.position.set(point[0],point[1],point[2]);marker.userData.planeId=preview.plane_id;this.content.add(marker)}
      if(preview.footprint_world_xyz.length){const points=preview.footprint_world_xyz.filter((_,index)=>index%Math.max(1,Math.ceil(preview.footprint_world_xyz.length/2500))===0).flat(),geometry=new THREE.BufferGeometry();geometry.setAttribute("position",new THREE.Float32BufferAttribute(points,3));const cloud=new THREE.Points(geometry,new THREE.PointsMaterial({color,size:.025,transparent:true,opacity:.8}));cloud.userData.planeId=preview.plane_id;this.content.add(cloud)}
    }
  }
  frame(){const box=new THREE.Box3().setFromObject(this.content);if(box.isEmpty())return;const sphere=box.getBoundingSphere(new THREE.Sphere());this.controls.target.copy(sphere.center);this.camera.position.copy(sphere.center.clone().add(new THREE.Vector3(1, .8,1).normalize().multiplyScalar(Math.max(2,sphere.radius*2.7))));this.controls.update()}
  dispose(){this.observer.disconnect();this.controls.dispose();this.renderer.dispose()}
}
