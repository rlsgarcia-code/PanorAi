"""Goal-oriented projection of the installed Workbench catalog.

This module never resolves browser-provided imports.  Curated recipes name
catalog surfaces/symbols or existing Workbench experiment IDs; responses emit
only component IDs and routes already issued by :mod:`panorai_workbench.catalog`.
"""

from __future__ import annotations

from dataclasses import dataclass
import re
import unicodedata
from typing import Any, Iterable, Mapping

from .catalog import experiment_catalog


SCHEMA = "panorai-workbench-discovery/v1"
DEFAULT_SAMPLE_ID = "poly-haven-studio-cc0"

_INPUT_KINDS = frozenset(
    {
        "erp",
        "mask",
        "labels",
        "support",
        "pose",
        "registered-scene",
        "run-artifact",
    }
)
_INPUT_ROLES = frozenset({"a", "b", "auxiliary"})
_FACT_KEYS = frozenset({"camera_height_m", "registered", "has_pose"})
_SEMANTIC_CAPABILITIES = {
    "panorai:raster": "raster",
    "panorai:scalar-raster": "scalar-raster",
    "panorai:mask": "mask",
    "panorai:labels": "labels",
    "panorai:categorical-raster": "labels",
    "panorai:support": "support",
}


@dataclass(frozen=True)
class Goal:
    id: str
    title: str
    question: str
    summary: str
    keywords: tuple[str, ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "title": self.title,
            "question": self.question,
            "summary": self.summary,
            "keywords": list(self.keywords),
        }


GOALS = (
    Goal(
        "project-navigate",
        "Projetar e navegar",
        "Quero olhar a esfera por outra projeção",
        "Abra vistas gnomônicas ou um cubemap sem alterar a ERP original.",
        ("projetar", "navegar", "gnomonic", "perspective", "cubemap", "fov"),
    ),
    Goal(
        "filter-inspect",
        "Filtrar e inspecionar bordas",
        "Quero realçar estrutura visual",
        "Aplique filtros e detectores esféricos preservando suporte e validade.",
        ("filtro", "blur", "canny", "edge", "borda", "gradiente", "contraste"),
    ),
    Goal(
        "regions",
        "Dividir e analisar regiões",
        "Quero separar a imagem em regiões coerentes",
        "Crie superpixels, máscaras ou medidas de regiões; SLIC não é segmentação semântica.",
        (
            "segmentar",
            "regiao",
            "region",
            "slic",
            "superpixel",
            "mask",
            "label",
            "watershed",
        ),
    ),
    Goal(
        "features",
        "Detectar e descrever features",
        "Quero encontrar pontos distintivos",
        "Extraia keypoints, bearings e descritores para inspeção ou etapas posteriores.",
        ("feature", "keypoint", "dog", "descritor", "descriptor", "ponto", "rootsift"),
    ),
    Goal(
        "planes",
        "Descobrir e medir planos",
        "Quero entender piso, paredes e outros planos",
        "Descubra orientações por retas ou produza geometria métrica com uma âncora válida.",
        ("plano", "piso", "parede", "planar", "vanishing", "fuga", "medir", "altura"),
    ),
    Goal(
        "align-pair",
        "Relacionar duas vistas",
        "Quero alinhar ou estimar o movimento entre panoramas",
        "Use features e correspondências para estimar pose relativa esférica.",
        (
            "alinhar",
            "pose",
            "match",
            "correspondencia",
            "essential",
            "epipolar",
            "duas imagens",
        ),
    ),
    Goal(
        "depth-range",
        "Estimar range ou profundidade",
        "Quero recuperar distância a partir de duas vistas",
        "Execute stereo somente quando o par e sua geometria necessária estiverem disponíveis.",
        ("depth", "range", "profundidade", "distancia", "stereo", "disparity"),
    ),
    Goal(
        "reconstruct",
        "Reconstruir uma cena",
        "Quero combinar evidências em geometria 3D",
        "Construa uma cena registrada, nuvem de pontos ou reconstrução planar revisável.",
        (
            "reconstruir",
            "reconstruction",
            "3d",
            "point cloud",
            "mesh",
            "multiview",
            "tsdf",
        ),
    ),
    Goal(
        "analyze-results",
        "Analisar máscaras e resultados",
        "Quero medir ou continuar um resultado existente",
        "Use labels, máscaras ou artefatos tipados como entrada de operações posteriores.",
        ("analisar", "metrica", "measure", "histogram", "resultado", "mask", "label"),
    ),
)
_GOAL_BY_ID = {goal.id: goal for goal in GOALS}


@dataclass(frozen=True)
class RecipeTemplate:
    id: str
    goal_id: str
    title: str
    summary: str
    required: tuple[str, ...]
    outputs: tuple[str, ...]
    priority: int
    surface_id: str | None = None
    symbol: str | None = None
    experiment_id: str | None = None
    kind: str = "operation"
    unavailable_diagnostic: str | None = None


CURATED_RECIPES = (
    RecipeTemplate(
        "gnomonic-view",
        "project-navigate",
        "Abrir uma vista gnomônica",
        "Escolha direção, FOV e resolução; a ERP permanece como fonte exata.",
        ("erp-a",),
        ("projected-view", "support"),
        100,
        "canonical-geometry",
        "equirectangular_to_gnomonic",
    ),
    RecipeTemplate(
        "cubemap-view",
        "project-navigate",
        "Projetar em cubemap",
        "Produza as faces canônicas com suporte explícito.",
        ("erp-a",),
        ("views", "support"),
        90,
        experiment_id="cubemap-projection",
    ),
    RecipeTemplate(
        "filters-edges",
        "filter-inspect",
        "Inspecionar filtros e bordas",
        "Compare blur, gradientes e Canny na esfera.",
        ("erp-a",),
        ("raster", "metrics"),
        100,
        experiment_id="filters-edges",
    ),
    RecipeTemplate(
        "spherical-slic",
        "regions",
        "Dividir em superpixels esféricos",
        "SLIC cria labels regionais; não afirma classes semânticas.",
        ("erp-a",),
        ("labels", "preview"),
        100,
        experiment_id="spherical-slic",
    ),
    RecipeTemplate(
        "region-analysis",
        "regions",
        "Medir regiões e fronteiras",
        "Continue a partir de labels para obter limites e tabelas.",
        ("labels",),
        ("boundaries", "table"),
        80,
        experiment_id="region-ops",
    ),
    RecipeTemplate(
        "spherical-dog",
        "features",
        "Detectar features DoG esféricas",
        "Mostre keypoints sobre a ERP e mantenha os descritores exatos.",
        ("erp-a",),
        ("keypoints", "descriptors"),
        100,
        # The installed catalog exposes the runnable detector through the
        # current `dog-op` bench. The older application adapter remains in the
        # technical catalog with its own diagnostic.
        experiment_id="dog-op",
    ),
    RecipeTemplate(
        "planar-discovery",
        "planes",
        "Descobrir orientações planares",
        "Detecte retas em cartas gnomônicas e revise direções e propostas.",
        ("erp-a",),
        ("lines", "vanishing-bundles", "plane-orientations"),
        100,
        experiment_id="spherical-planar-discovery",
        kind="specialized",
    ),
    RecipeTemplate(
        "known-height-planar",
        "planes",
        "Mapear planos por altura conhecida",
        "Posicione piso e planos relacionados somente após revisão humana.",
        ("erp-a", "camera-height"),
        ("planes", "range", "point-cloud"),
        95,
        experiment_id="known-height-planar-mapping",
        kind="specialized",
    ),
    RecipeTemplate(
        "relative-pose",
        "align-pair",
        "Estimar pose relativa esférica",
        "Extraia features, revise matches e estime R e a direção de t.",
        ("erp-a", "erp-b"),
        ("features", "matches", "pose"),
        100,
        experiment_id="relative-pose",
        kind="pipeline",
    ),
    RecipeTemplate(
        "spherical-stereo",
        "depth-range",
        "Estimar range por stereo esférico",
        "Use um par e pose conhecida; validade e confiança permanecem separadas.",
        ("erp-a", "erp-b", "pose"),
        ("range", "confidence", "validity"),
        100,
        experiment_id="spherical-stereo",
        kind="pipeline",
    ),
    RecipeTemplate(
        "classical-stereo",
        "depth-range",
        "Comparar stereo clássico",
        "Inspecione disparity e validade em métodos clássicos disponíveis.",
        ("erp-a", "erp-b", "pose"),
        ("disparity", "validity"),
        80,
        experiment_id="classical-stereo",
        kind="pipeline",
    ),
    RecipeTemplate(
        "human-guided-planar",
        "reconstruct",
        "Reconstrução planar multivista guiada",
        "Revise regiões, contatos e planos de uma cena registrada.",
        ("registered-scene",),
        ("planes", "range", "archive"),
        100,
        experiment_id="human-guided-planar",
        kind="specialized",
    ),
    RecipeTemplate(
        "dense-reconstruction",
        "reconstruct",
        "Reconstrução esférica densa",
        "Converta uma cena registrada em evidência densa e geometria 3D.",
        ("registered-scene",),
        ("range", "point-cloud"),
        90,
        "spherical-applications-dense-reconstruction",
        "SphericalDenseReconstruction",
        kind="pipeline",
    ),
    RecipeTemplate(
        "morphology",
        "analyze-results",
        "Refinar uma máscara",
        "Aplique morfologia esférica com nearest-neighbour.",
        ("mask",),
        ("mask", "metrics"),
        100,
        experiment_id="morphology",
    ),
    RecipeTemplate(
        "cam-sam",
        "regions",
        "Propor instâncias com CAM e SAM",
        "Fluxo opcional que exige evidência semântica e checkpoint aceito explicitamente.",
        ("erp-a",),
        ("proposals", "masks", "instances"),
        45,
        experiment_id="cam-sam-pipeline",
        kind="pipeline",
        unavailable_diagnostic=(
            "Requer evidência semântica, checkpoint SAM 2.1 e o extra "
            "panorai[deep-learning-sam]; nada é baixado automaticamente."
        ),
    ),
)


_CLASSIFIERS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("planes", ("planar", "plane", "vanishing", "greatcircleline", "linefit")),
    ("depth-range", ("stereo", "depth", "range", "disparity", "costvolume")),
    ("reconstruct", ("reconstruct", "pointcloud", "mesh", "tsdf", "mapper")),
    ("align-pair", ("relativepose", "essential", "epipolar", "triangulat", "match")),
    ("features", ("feature", "keypoint", "descriptor", "dog", "rootsift")),
    (
        "regions",
        (
            "slic",
            "segment",
            "watershed",
            "region",
            "morpholog",
            "connected",
            "label",
            "mask",
        ),
    ),
    (
        "filter-inspect",
        ("gaussian", "blur", "canny", "edge", "gradient", "filter", "sobel", "laplac"),
    ),
    (
        "project-navigate",
        ("gnomonic", "cubemap", "projection", "projector", "reproject"),
    ),
    (
        "analyze-results",
        ("metric", "measure", "histogram", "evaluate", "report", "score"),
    ),
)


def _fold(value: str) -> str:
    value = unicodedata.normalize("NFKD", value)
    return "".join(char for char in value if not unicodedata.combining(char)).lower()


def _compact(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", _fold(value))


def classify_component(component: Mapping[str, Any]) -> tuple[str, ...]:
    """Classify a ready component conservatively from public metadata."""

    text = _compact(
        " ".join(
            str(component.get(key, "")) for key in ("symbol", "title", "public_api")
        )
    )
    goals = [
        goal for goal, needles in _CLASSIFIERS if any(item in text for item in needles)
    ]
    return tuple(dict.fromkeys(goals))


def _component_requirements(component: Mapping[str, Any]) -> tuple[str, ...] | None:
    required = [
        item for item in component.get("input_schema", ()) if item.get("required")
    ]
    result: list[str] = []
    erp_count = 0
    for field in required:
        kind = field.get("kind")
        name = _fold(str(field.get("name", "")))
        semantic = _fold(str(field.get("semantic", "")))
        semantic_requirement = _SEMANTIC_CAPABILITIES.get(semantic)
        if semantic_requirement is not None:
            result.append(semantic_requirement)
        elif kind == "erp-image":
            result.append("erp-a" if erp_count == 0 else "erp-b")
            erp_count += 1
        elif kind == "mask":
            result.append("mask")
        elif "label" in name or "label" in semantic:
            result.append("labels")
        elif any(token in name for token in ("pose", "rotation", "translation")):
            result.append("pose")
        elif "scene" in name or "registered" in name:
            result.append("registered-scene")
        elif kind in {"boolean", "integer", "number", "select", "text"}:
            # Scalar public inputs remain editable in the Runner.
            continue
        elif field.get("upstream") is True:
            result.append("run-artifact")
        else:
            return None
    return tuple(dict.fromkeys(result))


def _capabilities(context: Mapping[str, Any]) -> frozenset[str]:
    capabilities: set[str] = set()
    for item in context.get("inputs", ()):
        kind = item["kind"]
        role = item.get("role", "auxiliary")
        if kind == "erp":
            capabilities.add("erp-b" if role == "b" else "erp-a")
            capabilities.add("raster")
        elif kind == "pose":
            capabilities.add("pose")
        elif kind == "registered-scene":
            capabilities.add("registered-scene")
        elif kind == "run-artifact":
            capabilities.add("run-artifact")
        else:
            capabilities.add(kind)
        semantic_capability = _SEMANTIC_CAPABILITIES.get(item.get("semantic"))
        if semantic_capability is not None:
            capabilities.add(semantic_capability)
    facts = context.get("facts", {})
    if facts.get("has_pose"):
        capabilities.add("pose")
    if facts.get("registered"):
        capabilities.add("registered-scene")
    if (
        isinstance(facts.get("camera_height_m"), (int, float))
        and facts["camera_height_m"] > 0
    ):
        capabilities.add("camera-height")
    return frozenset(capabilities)


def validate_context(raw: Mapping[str, Any] | None) -> dict[str, Any]:
    """Validate the closed metadata-only recommendation context."""

    raw = raw or {}
    unknown = set(raw) - {"inputs", "facts"}
    if unknown:
        raise ValueError(
            f"unknown discovery context field(s): {', '.join(sorted(unknown))}"
        )
    inputs = raw.get("inputs", [])
    facts = raw.get("facts", {})
    if not isinstance(inputs, list) or not isinstance(facts, dict):
        raise ValueError("inputs must be an array and facts must be an object")
    normalized_inputs: list[dict[str, Any]] = []
    for index, item in enumerate(inputs):
        if not isinstance(item, dict):
            raise ValueError(f"inputs[{index}] must be an object")
        unknown_item = set(item) - {
            "id",
            "kind",
            "role",
            "source",
            "name",
            "shape_hw",
            "mime_type",
            "semantic",
        }
        if unknown_item:
            raise ValueError(f"inputs[{index}] contains unsupported metadata")
        kind = item.get("kind")
        role = item.get("role", "auxiliary")
        if kind not in _INPUT_KINDS or role not in _INPUT_ROLES:
            raise ValueError(f"inputs[{index}] has an unsupported kind or role")
        semantic = item.get("semantic")
        if semantic is not None and (
            not isinstance(semantic, str)
            or not semantic.startswith("panorai:")
            or len(semantic) > 80
        ):
            raise ValueError(f"inputs[{index}].semantic must be a PanorAi semantic ID")
        shape = item.get("shape_hw")
        if shape is not None and (
            not isinstance(shape, list)
            or len(shape) != 2
            or not all(isinstance(value, int) and value > 0 for value in shape)
        ):
            raise ValueError(
                f"inputs[{index}].shape_hw must contain two positive integers"
            )
        normalized_inputs.append(
            {
                key: item[key]
                for key in (
                    "id",
                    "kind",
                    "role",
                    "source",
                    "name",
                    "shape_hw",
                    "mime_type",
                    "semantic",
                )
                if key in item
            }
        )
    unknown_facts = set(facts) - _FACT_KEYS
    if unknown_facts:
        raise ValueError(
            f"unsupported discovery fact(s): {', '.join(sorted(unknown_facts))}"
        )
    normalized_facts: dict[str, Any] = {}
    for key, value in facts.items():
        if key == "camera_height_m":
            if (
                not isinstance(value, (int, float))
                or isinstance(value, bool)
                or value <= 0
            ):
                raise ValueError("camera_height_m must be positive")
            normalized_facts[key] = float(value)
        elif not isinstance(value, bool):
            raise ValueError(f"{key} must be boolean")
        else:
            normalized_facts[key] = value
    return {"inputs": normalized_inputs, "facts": normalized_facts}


def _indices(
    catalog: Mapping[str, Any],
) -> tuple[dict[tuple[str, str], dict[str, Any]], dict[str, dict[str, Any]]]:
    components = {
        (item["surface_id"], item["symbol"]): item for item in catalog["components"]
    }
    experiments = {item["id"]: item for item in catalog["experiments"]}
    return components, experiments


def _recipe_from_template(
    template: RecipeTemplate,
    components: Mapping[tuple[str, str], dict[str, Any]],
    experiments: Mapping[str, dict[str, Any]],
) -> dict[str, Any]:
    target: dict[str, Any] | None
    component: dict[str, Any] | None = None
    if template.experiment_id:
        target = experiments.get(template.experiment_id)
        if target and target.get("component_id"):
            component = next(
                (
                    item
                    for item in components.values()
                    if item["id"] == target["component_id"]
                ),
                None,
            )
        route = None if target is None else target.get("route")
        runtime_ready = bool(target and target.get("adapter_status") == "ready")
        diagnostic = None if target is None else target.get("diagnostic")
        stability = None if target is None else target.get("stability")
    else:
        component = components.get((template.surface_id or "", template.symbol or ""))
        target = component
        route = None if component is None else component.get("route")
        runtime_ready = bool(component and component.get("status") == "ready")
        diagnostic = None if component is None else component.get("diagnostic")
        stability = None if component is None else component.get("tier")
    if target is None:
        runtime_ready = False
        diagnostic = "O componente não existe no catálogo instalado."
    if not runtime_ready and template.unavailable_diagnostic:
        diagnostic = template.unavailable_diagnostic
    steps = []
    if component is not None:
        steps.append(
            {
                "id": "step-1",
                "component_id": component["id"],
                "title": component["title"],
                "public_api": component["public_api"],
            }
        )
    return {
        "id": template.id,
        "goal_id": template.goal_id,
        "title": template.title,
        "summary": template.summary,
        "kind": template.kind,
        "required": list(template.required),
        "outputs": list(template.outputs),
        "priority": template.priority,
        "route": route,
        "steps": steps,
        "stability": stability,
        "runtime_ready": runtime_ready,
        "diagnostic": diagnostic,
        "source": "curated",
    }


def _automatic_recipes(
    catalog: Mapping[str, Any], curated_component_ids: set[str]
) -> Iterable[dict[str, Any]]:
    for component in catalog["components"]:
        if component["status"] != "ready" or component["id"] in curated_component_ids:
            continue
        goal_ids = classify_component(component)
        requirements = _component_requirements(component)
        # Zero-input helpers, backend probes and configuration factories belong
        # in the technical catalog even when their surface name matches a goal.
        if not goal_ids or not requirements:
            continue
        for goal_id in goal_ids:
            yield {
                "id": f"auto--{goal_id}--{component['id']}",
                "goal_id": goal_id,
                "title": component["title"],
                "summary": component.get("summary")
                or "Componente público tipado do catálogo instalado.",
                "kind": "operation",
                "required": list(requirements),
                "outputs": [
                    str(
                        (component.get("output_schema") or {}).get("semantic")
                        or (component.get("output_schema") or {}).get("kind")
                        or "structured-result"
                    )
                ],
                "priority": 20,
                "route": component["route"],
                "steps": [
                    {
                        "id": "step-1",
                        "component_id": component["id"],
                        "title": component["title"],
                        "public_api": component["public_api"],
                    }
                ],
                "stability": component["tier"],
                "runtime_ready": True,
                "diagnostic": None,
                "source": "automatic",
            }


def _catalog_projection(
    catalog: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    components, experiments = _indices(catalog)
    curated = [
        _recipe_from_template(template, components, experiments)
        for template in CURATED_RECIPES
    ]
    curated_component_ids = {
        step["component_id"] for recipe in curated for step in recipe["steps"]
    }
    recipes = curated + list(_automatic_recipes(catalog, curated_component_ids))
    classification = []
    for component in catalog["components"]:
        if component["status"] != "ready":
            continue
        goals = classify_component(component)
        requirements = _component_requirements(component)
        classification.append(
            {
                "component_id": component["id"],
                "goal_ids": list(goals),
                "classification": (
                    "discoverable" if goals and requirements else "technical-only"
                ),
                "reason": (
                    None
                    if goals and requirements
                    else "Sem resultado ou input semanticamente inequívoco para recomendar."
                ),
            }
        )
    return recipes, classification


def discovery_catalog(catalog: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Return the stable goal/recipe projection for the installed catalog."""

    installed = dict(catalog or experiment_catalog())
    recipes, classification = _catalog_projection(installed)
    return {
        "schema": SCHEMA,
        "catalog": installed["catalog"],
        "runtime": installed["runtime"],
        "default_sample_id": DEFAULT_SAMPLE_ID,
        "default_context": {
            "inputs": [
                {
                    "id": "erp-a",
                    "kind": "erp",
                    "role": "a",
                    "source": "sample",
                    "name": "Poly Haven Studio",
                    "shape_hw": [512, 1024],
                    "mime_type": "image/jpeg",
                    "semantic": "panorai:rgb-erp",
                }
            ],
            "facts": {},
        },
        "goals": [goal.as_dict() for goal in GOALS],
        "recipes": recipes,
        "component_discovery": classification,
        "counts": {
            "goals": len(GOALS),
            "recipes": len(recipes),
            "discoverable_components": sum(
                item["classification"] == "discoverable" for item in classification
            ),
            "technical_only_components": sum(
                item["classification"] == "technical-only" for item in classification
            ),
        },
    }


def recommendations(
    raw_context: Mapping[str, Any] | None,
    catalog: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Rank recipes for a closed, metadata-only typed input context."""

    context = validate_context(raw_context)
    capabilities = _capabilities(context)
    projection = discovery_catalog(catalog)
    by_goal: dict[str, list[dict[str, Any]]] = {goal.id: [] for goal in GOALS}
    for raw_recipe in projection["recipes"]:
        recipe = dict(raw_recipe)
        missing = [item for item in recipe["required"] if item not in capabilities]
        if not recipe["runtime_ready"]:
            state = "unavailable"
            state_rank = 0
        elif missing:
            state = "needs-input"
            state_rank = 1
        else:
            state = "ready"
            state_rank = 2
        recipe["state"] = state
        recipe["missing"] = missing
        recipe["score"] = state_rank * 1000 + int(recipe["priority"])
        by_goal[recipe["goal_id"]].append(recipe)

    results = []
    for goal in GOALS:
        ranked = sorted(
            by_goal[goal.id],
            key=lambda item: (-item["score"], item["title"].casefold(), item["id"]),
        )
        recommended = ranked[0] if ranked else None
        alternatives = ranked[1:4]
        # Keep one curated blocked method visible. It communicates a product
        # boundary (for example, an optional SAM runtime) that a long list of
        # automatic ready operations must not hide.
        curated_blocked = next(
            (
                item
                for item in ranked[1:]
                if item["source"] == "curated" and item["state"] == "unavailable"
            ),
            None,
        )
        if curated_blocked is not None and curated_blocked not in alternatives:
            alternatives = [*alternatives[:2], curated_blocked]
        goal_state = "unavailable" if recommended is None else recommended["state"]
        results.append(
            {
                **goal.as_dict(),
                "state": goal_state,
                "recommended": recommended,
                "alternatives": alternatives,
                "total_candidates": len(ranked),
            }
        )
    return {
        "schema": SCHEMA,
        "context": context,
        "capabilities": sorted(capabilities),
        "recommendations": results,
    }


__all__ = [
    "DEFAULT_SAMPLE_ID",
    "GOALS",
    "SCHEMA",
    "classify_component",
    "discovery_catalog",
    "recommendations",
    "validate_context",
]
