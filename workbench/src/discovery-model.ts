export type DiscoveryState = "ready" | "needs-input" | "unavailable";
export type DiscoveryInputKind =
  | "erp"
  | "mask"
  | "labels"
  | "support"
  | "pose"
  | "registered-scene"
  | "run-artifact";

export interface DiscoveryInput {
  id: string;
  kind: DiscoveryInputKind;
  role: "a" | "b" | "auxiliary";
  source: string;
  name?: string;
  shape_hw?: [number, number];
  mime_type?: string;
  semantic?: string;
}

export interface DiscoveryContext {
  inputs: DiscoveryInput[];
  facts: {
    camera_height_m?: number;
    registered?: boolean;
    has_pose?: boolean;
  };
}

export interface DiscoveryStep {
  id: string;
  component_id: string;
  title: string;
  public_api: string;
}

export interface DiscoveryRecipe {
  id: string;
  goal_id: string;
  title: string;
  summary: string;
  kind: "operation" | "pipeline" | "specialized";
  required: string[];
  outputs: string[];
  route: string | null;
  steps: DiscoveryStep[];
  stability: string | null;
  runtime_ready: boolean;
  diagnostic: string | null;
  source: "curated" | "automatic";
  state: DiscoveryState;
  missing: string[];
  score: number;
}

export interface DiscoveryRecommendation {
  id: string;
  title: string;
  question: string;
  summary: string;
  keywords: string[];
  state: DiscoveryState;
  recommended: DiscoveryRecipe | null;
  alternatives: DiscoveryRecipe[];
  total_candidates: number;
}

export interface DiscoveryResponse {
  schema: "panorai-workbench-discovery/v1";
  context: DiscoveryContext;
  capabilities: string[];
  recommendations: DiscoveryRecommendation[];
}

export interface DiscoveryDefinition {
  schema: "panorai-workbench-discovery/v1";
  default_sample_id: string;
  default_context: DiscoveryContext;
  runtime: unknown;
  counts: {
    goals: number;
    recipes: number;
    discoverable_components: number;
    technical_only_components: number;
  };
}

export interface RecipeDraft {
  schema: "panorai-workbench-recipe-draft/v1";
  id: string;
  recipe_id: string;
  title: string;
  created_at: string;
  steps: DiscoveryStep[];
  connections?: Record<string, string[]>;
  input_ids: string[];
  source_route: string | null;
}

export function foldDiscoveryText(value: string): string {
  return value
    .normalize("NFKD")
    .replace(/[\u0300-\u036f]/g, "")
    .toLocaleLowerCase("pt-BR");
}

export function goalMatchesQuery(
  goal: DiscoveryRecommendation,
  query: string,
): boolean {
  const folded = foldDiscoveryText(query.trim());
  if (!folded) return true;
  return foldDiscoveryText(
    [goal.title, goal.question, goal.summary, ...goal.keywords].join(" "),
  ).includes(folded);
}

export const requirementLabels: Record<string, string> = {
  "erp-a": "ERP principal",
  "erp-b": "segunda ERP",
  mask: "máscara",
  labels: "labels",
  support: "suporte explícito",
  pose: "pose conhecida",
  "camera-height": "altura da câmera",
  "registered-scene": "cena registrada",
  "run-artifact": "resultado tipado anterior",
  "scalar-raster": "raster escalar de um canal",
  raster: "raster tipado",
};

export function requirementLabel(requirement: string): string {
  return requirementLabels[requirement] ?? requirement;
}

export function createRecipeDraft(
  recipe: DiscoveryRecipe,
  context: DiscoveryContext,
  now = new Date(),
): RecipeDraft {
  return {
    schema: "panorai-workbench-recipe-draft/v1",
    id: `recipe-${now.getTime()}-${recipe.id}`,
    recipe_id: recipe.id,
    title: recipe.title,
    created_at: now.toISOString(),
    steps: recipe.steps.map((step) => ({ ...step })),
    input_ids: context.inputs.map((input) => input.id),
    source_route: recipe.route,
  };
}

export function recipeHandoffUrl(recipe: DiscoveryRecipe, draft: RecipeDraft): string {
  const route = recipeUsesSpecializedWorkspace(recipe)
    ? recipe.route
    : `/runner.html?component=${encodeURIComponent(recipe.steps[0].component_id)}`;
  if (!route) return "/?view=technical";
  const url = new URL(route, "http://workbench.local");
  url.searchParams.set("draft", draft.id);
  return `${url.pathname}${url.search}`;
}

export function recipeUsesSpecializedWorkspace(recipe: DiscoveryRecipe): boolean {
  return recipe.kind === "specialized"
    || recipe.steps.length === 0
    || Boolean(recipe.route && !recipe.route.startsWith("/runner.html"));
}
