import {
  rayToErpPixel,
  rayToViewPixel,
  sampleRegionBoundary,
  viewPixelToRay,
} from "./geometry.ts";
import type { Point2, SphericalAnnotation, ViewState } from "./model.ts";

export interface PanoramaPixels {
  width: number;
  height: number;
  data: Uint8ClampedArray;
}

function sampleBilinear(source: PanoramaPixels, x: number, y: number): [number, number, number, number] {
  const wrappedX = ((x % source.width) + source.width) % source.width;
  const clampedY = Math.max(0, Math.min(source.height - 1, y));
  const x0 = Math.floor(wrappedX);
  const y0 = Math.floor(clampedY);
  const x1 = (x0 + 1) % source.width;
  const y1 = Math.min(y0 + 1, source.height - 1);
  const fx = wrappedX - x0;
  const fy = clampedY - y0;
  const pixel = (px: number, py: number, channel: number) => source.data[(py * source.width + px) * 4 + channel];
  const out: number[] = [];
  for (let channel = 0; channel < 4; channel += 1) {
    const top = pixel(x0, y0, channel) * (1 - fx) + pixel(x1, y0, channel) * fx;
    const bottom = pixel(x0, y1, channel) * (1 - fx) + pixel(x1, y1, channel) * fx;
    out.push(Math.round(top * (1 - fy) + bottom * fy));
  }
  return out as [number, number, number, number];
}

export function renderGnomonic(
  context: CanvasRenderingContext2D,
  source: PanoramaPixels,
  view: ViewState,
  width: number,
  height: number,
): void {
  const output = context.createImageData(width, height);
  for (let y = 0; y < height; y += 1) {
    for (let x = 0; x < width; x += 1) {
      const ray = viewPixelToRay({ x: x + 0.5, y: y + 0.5 }, { width, height }, view);
      const sourcePoint = rayToErpPixel(ray, source);
      const rgba = sampleBilinear(source, sourcePoint.x - 0.5, sourcePoint.y - 0.5);
      const offset = (y * width + x) * 4;
      output.data[offset] = rgba[0];
      output.data[offset + 1] = rgba[1];
      output.data[offset + 2] = rgba[2];
      output.data[offset + 3] = 255;
    }
  }
  context.putImageData(output, 0, 0);
}

function drawPolyline(
  context: CanvasRenderingContext2D,
  points: Array<Point2 | null>,
  close: boolean,
  discontinuityPx: number,
): void {
  context.beginPath();
  let previous: Point2 | null = null;
  let first: Point2 | null = null;
  for (const point of points) {
    if (point === null) {
      previous = null;
      continue;
    }
    const discontinuous = previous !== null && Math.hypot(point.x - previous.x, point.y - previous.y) > discontinuityPx;
    if (previous === null || discontinuous) {
      context.moveTo(point.x, point.y);
      if (first === null) first = point;
    } else {
      context.lineTo(point.x, point.y);
    }
    previous = point;
  }
  if (close && previous !== null && first !== null && Math.hypot(first.x - previous.x, first.y - previous.y) <= discontinuityPx) {
    context.lineTo(first.x, first.y);
  }
  context.stroke();
}

export function drawViewAnnotations(
  context: CanvasRenderingContext2D,
  annotations: SphericalAnnotation[],
  selectedId: string | null,
  view: ViewState,
  size: { width: number; height: number },
): void {
  context.save();
  context.lineJoin = "round";
  for (const annotation of annotations) {
    const selected = annotation.id === selectedId;
    context.strokeStyle = selected ? "#ffc857" : "rgba(255, 255, 255, 0.82)";
    context.lineWidth = selected ? 3 : 1.5;
    context.setLineDash(selected ? [] : [7, 5]);
    const points = sampleRegionBoundary(annotation.region).map((ray) => rayToViewPixel(ray, size, view));
    drawPolyline(context, points, true, Math.max(size.width, size.height) * 0.45);
  }
  context.restore();
}

export function drawErpAnnotations(
  context: CanvasRenderingContext2D,
  annotations: SphericalAnnotation[],
  selectedId: string | null,
  displaySize: { width: number; height: number },
): void {
  context.save();
  for (const annotation of annotations) {
    const selected = annotation.id === selectedId;
    context.strokeStyle = selected ? "#ffc857" : "rgba(255, 255, 255, 0.78)";
    context.lineWidth = selected ? 2.5 : 1.25;
    context.setLineDash(selected ? [] : [5, 4]);
    const points = sampleRegionBoundary(annotation.region).map((ray) => rayToErpPixel(ray, displaySize));
    drawPolyline(context, points, false, displaySize.width * 0.25);
  }
  context.restore();
}
