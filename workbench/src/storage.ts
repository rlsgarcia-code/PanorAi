import { ANNOTATION_SCHEMA } from "./model.ts";
import type { AnnotationDocument, PanoramaRecord, SphericalAnnotation } from "./model.ts";

export function createDocument(
  panorama: PanoramaRecord,
  annotations: SphericalAnnotation[],
): AnnotationDocument {
  return {
    schema: ANNOTATION_SCHEMA,
    geometryConvention: "panorai-geometry-v1",
    panorama,
    annotations,
    exportedAt: new Date().toISOString(),
  };
}

export function parseDocument(value: unknown): AnnotationDocument {
  if (typeof value !== "object" || value === null) throw new Error("Annotation document must be an object.");
  const candidate = value as Partial<AnnotationDocument>;
  if (candidate.schema !== ANNOTATION_SCHEMA) throw new Error(`Unsupported schema: ${String(candidate.schema)}`);
  if (candidate.geometryConvention !== "panorai-geometry-v1") throw new Error("Unsupported geometry convention.");
  if (!candidate.panorama || typeof candidate.panorama.fileName !== "string") throw new Error("Missing panorama metadata.");
  if (!Array.isArray(candidate.annotations)) throw new Error("Annotations must be an array.");
  for (const annotation of candidate.annotations) {
    if (!annotation || typeof annotation.id !== "string" || typeof annotation.label !== "string") {
      throw new Error("Invalid annotation record.");
    }
    if (!annotation.region || !Number.isFinite(annotation.region.centerYawDeg)) {
      throw new Error(`Invalid spherical region for annotation ${annotation.id}.`);
    }
  }
  return candidate as AnnotationDocument;
}

export function downloadDocument(document: AnnotationDocument): void {
  const blob = new Blob([`${JSON.stringify(document, null, 2)}\n`], { type: "application/json" });
  const url = URL.createObjectURL(blob);
  const anchor = window.document.createElement("a");
  anchor.href = url;
  anchor.download = `${document.panorama.id || "panorama"}.annotations.json`;
  anchor.click();
  URL.revokeObjectURL(url);
}
