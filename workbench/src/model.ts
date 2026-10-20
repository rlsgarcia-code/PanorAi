export const ANNOTATION_SCHEMA = "panorai-spherical-annotations/v0" as const;

export type ReviewState = "draft" | "accepted" | "rejected" | "needs-review";

export interface SphericalFovRegion {
  centerYawDeg: number;
  centerPitchDeg: number;
  horizontalFovDeg: number;
  verticalFovDeg: number;
  rollDeg: 0;
}

export interface SphericalAnnotation {
  id: string;
  label: string;
  state: ReviewState;
  region: SphericalFovRegion;
  createdAt: string;
  updatedAt: string;
  source: "human" | "model-suggestion";
}

export interface PanoramaRecord {
  id: string;
  fileName: string;
  width: number;
  height: number;
  sha256?: string;
}

export interface AnnotationDocument {
  schema: typeof ANNOTATION_SCHEMA;
  geometryConvention: "panorai-geometry-v1";
  panorama: PanoramaRecord;
  annotations: SphericalAnnotation[];
  exportedAt: string;
}

export interface ViewState {
  yawDeg: number;
  pitchDeg: number;
  horizontalFovDeg: number;
}

export interface Point2 {
  x: number;
  y: number;
}

export interface Vec3 {
  x: number;
  y: number;
  z: number;
}
