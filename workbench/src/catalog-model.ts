export interface CatalogFamily { id: string; title: string; summary: string }
export interface CatalogSurfaceFamily { family: string }

/** Preserve server order while retaining families unknown to this frontend. */
export function visibleCatalogFamilies<T extends CatalogFamily>(
  families: readonly T[],
  surfaces: readonly CatalogSurfaceFamily[],
): T[] {
  const present = new Set(surfaces.map((surface) => surface.family));
  return families.filter((family) => present.has(family.id));
}

/** Group adapters from data instead of a compiled list of API levels. */
export function experimentFamilies(
  experiments: readonly { family: string | null }[],
): string[] {
  return [...new Set(experiments.map((item) => item.family ?? "other"))];
}
