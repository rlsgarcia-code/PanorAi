import { moveSphericalViewport, zoomSphericalViewport } from "./spherical-interaction.ts";
import type { Point2, ViewState } from "./model.ts";

export function zoomPlanarView(view: ViewState, factor: number): ViewState {
  return zoomSphericalViewport(view, factor);
}

/** UI orbit, in angular degrees; changes the editing view, never the camera pose. */
export function dragPlanarView(
  view: ViewState,
  start: Point2,
  end: Point2,
  width: number,
  mode: "navigate" | "zoom",
): ViewState {
  return moveSphericalViewport(view, start, end, width, mode);
}

export function planarHistoryShortcut(event: {
  key: string;
  metaKey: boolean;
  ctrlKey: boolean;
  shiftKey: boolean;
  altKey: boolean;
}): "undo" | "redo" | null {
  if (!(event.metaKey || event.ctrlKey) || event.altKey) return null;
  if (event.key.toLowerCase() === "z") return event.shiftKey ? "redo" : "undo";
  if (event.key.toLowerCase() === "y" && !event.shiftKey) return "redo";
  return null;
}
