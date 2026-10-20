"""Goal-oriented scene setup derived from public PanorAi contracts."""

from __future__ import annotations

from typing import Any

from panorai.applications import (
    PanoramicDenseStereo,
    SphericalDenseReconstruction,
    SphericalRelativePose,
)
from panorai.reconstruction import SphericalGlobalMapperOptions

from .engines import engine_capabilities
from .models import ProjectRecord
from .registration import PairRegistrationService


def scene_setup(
    project: ProjectRecord,
    registration: PairRegistrationService,
    global_map: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Return actionable setup state without inventing missing geometry."""

    pair_states = [_pair_state(project, pair, registration) for pair in project.pairs]
    accepted = [item for item in pair_states if item["status"] == "accepted"]
    baseline_pairs = [pair for pair in project.pairs if pair.baseline_m is not None]
    mapper_options = SphericalGlobalMapperOptions()
    bound_range_priors = registration.validated_range_prior_records(project)
    map_current = bool(
        global_map and global_map.get("success") and not global_map.get("stale")
    )
    if map_current and global_map:
        admitted_pair_ids = set(global_map.get("input_edge_revisions", {}))
        mapped_image_ids = set(global_map.get("component_image_ids", ()))
        baseline_pairs = [
            pair
            for pair in baseline_pairs
            if {pair.reference_image_id, pair.source_image_id} <= mapped_image_ids
        ]
        bound_range_priors = [
            item
            for item in bound_range_priors
            if item["binding_pair_id"] in admitted_pair_ids
            and item["panorama_id"] in mapped_image_ids
        ]
    measured_baselines = [pair.baseline_m for pair in baseline_pairs]
    connected_images = _largest_component_size(project)
    accepted_component = _largest_accepted_component_size(project, pair_states)
    graph_eligible = (
        accepted_component >= mapper_options.min_panoramas
        and len(accepted) >= mapper_options.min_panoramas - 1
    )
    map_metric = bool(
        map_current and str(global_map.get("scale", "")).startswith("metric")
    )
    metric_scale_sources = []
    if measured_baselines:
        metric_scale_sources.append("baseline-prior")
    if map_metric and global_map and global_map.get("range_priors_applied", 0):
        metric_scale_sources.append("range-prior")
    metric_scale_available = bool(metric_scale_sources)
    dense_scene_blockers = []
    if not map_current:
        dense_scene_blockers.append("current-global-map-required")
    if not map_metric:
        dense_scene_blockers.append("metric-scale-required")
    if not global_map or len(global_map.get("poses", ())) < 3:
        dense_scene_blockers.append("three-registered-views-required")
    dense_scene_eligible = not dense_scene_blockers
    steps = [
        {
            "id": "images",
            "title": "Imagens esféricas",
            "status": "ready" if project.images else "action",
            "detail": f"{len(project.images)} ERP(s) preservadas com dimensões e SHA-256.",
            "engine": "panorai-streetview.application",
        },
        {
            "id": "pairs",
            "title": "Grafo de pares",
            "status": "ready" if project.pairs else "action",
            "detail": (
                f"{len(project.pairs)} par(es); maior componente contém "
                f"{connected_images}/{len(project.images)} imagens."
            ),
            "engine": "panorai-streetview.application",
        },
        {
            "id": "relative-pose",
            "title": "Componente de poses relativas",
            "status": "ready" if graph_eligible else "action",
            "detail": (
                f"{len(accepted)}/{len(pair_states)} pares aceitos; componente "
                f"revisado contém {accepted_component}/{len(project.images)} vistas."
            ),
            "engine": "panorai.applications.SphericalRelativePose",
        },
        {
            "id": "global-map",
            "title": "Mapa global",
            "status": (
                "ready" if map_current else "action" if graph_eligible else "blocked"
            ),
            "detail": (
                (
                    f"Revisão {global_map['revision']} aceita com "
                    f"{len(global_map['poses'])} câmeras e "
                    f"{global_map['track_count']} tracks."
                    if map_current
                    else "A evidência atende ao mínimo; execute SphericalGlobalMapper."
                )
                if graph_eligible
                else (
                    f"Requer ao menos {mapper_options.min_panoramas} panoramas em "
                    "um componente de arestas aceitas."
                )
            ),
            "engine": "panorai.reconstruction.SphericalGlobalMapper",
        },
        {
            "id": "scale",
            "title": "Escala métrica",
            "status": "ready" if metric_scale_available else "action",
            "detail": (
                (
                    f"{len(measured_baselines)} baseline(s) métrica(s) disponível(is)."
                    if measured_baselines
                    else (
                        f"{global_map.get('range_priors_applied', 0)} "
                        "SphericalRangePrior aplicado(s) ao mapa."
                    )
                )
                if metric_scale_available
                else (
                    f"{len(bound_range_priors)}/{mapper_options.range_prior_min_count} "
                    "âncoras radiais ligadas a features; uma baseline medida ou o "
                    "mínimo de SphericalRangePrior deve ser aplicado pelo mapper."
                )
            ),
            "engine": (
                "panorai.reconstruction.SphericalBaselinePrior | "
                "panorai.reconstruction.SphericalRangePrior"
            ),
        },
        {
            "id": "dense-pair",
            "title": "Preview denso do par",
            "status": "optional" if accepted else "blocked",
            "detail": (
                "Elegível em unidades de baseline; continua não métrico sem baseline medida."
                if accepted and not metric_scale_available
                else (
                    "Elegível para range métrico no par."
                    if accepted
                    else "Requer ao menos uma pose relativa aceita."
                )
            ),
            "engine": "panorai.applications.PanoramicDenseStereo",
        },
        {
            "id": "dense-scene",
            "title": "Reconstrução densa da cena",
            "status": "ready" if dense_scene_eligible else "blocked",
            "detail": (
                "Cena métrica atual com três ou mais vistas; o perfil público pode "
                "executar MVS, TSDF e extração de superfície."
                if dense_scene_eligible
                else _dense_scene_blocker_detail(
                    dense_scene_blockers,
                    len(bound_range_priors),
                    mapper_options.range_prior_min_count,
                )
            ),
            "engine": "panorai.applications.SphericalDenseReconstruction",
        },
    ]
    return {
        "schema": "panorai-streetview-scene-setup/v1",
        "project_id": project.id,
        "workflow": project.workflow,
        "summary": {
            "image_count": len(project.images),
            "pair_count": len(project.pairs),
            "registered_pair_count": sum(
                item["status"] != "not-run" for item in pair_states
            ),
            "accepted_pair_count": len(accepted),
            "accepted_component_size": accepted_component,
            "metric_scale_available": metric_scale_available,
            "metric_scale_sources": metric_scale_sources,
            "bound_range_prior_count": len(bound_range_priors),
            "range_prior_min_count": mapper_options.range_prior_min_count,
            "global_map_eligible": graph_eligible,
            "global_map_status": (
                "current"
                if map_current
                else "stale"
                if global_map and global_map.get("stale")
                else "failed"
                if global_map and not global_map.get("success")
                else "not-run"
            ),
            "dense_scene_eligible": dense_scene_eligible,
            "dense_scene_blockers": dense_scene_blockers,
        },
        "pairs": pair_states,
        "steps": steps,
        "recommended_action": _recommended_action(
            pair_states,
            graph_eligible,
            map_current,
            map_metric,
            len(bound_range_priors),
            mapper_options.range_prior_min_count,
            dense_scene_eligible,
            global_map,
        ),
        "profiles": _profiles(),
        "capabilities": engine_capabilities()["engines"],
    }


def _pair_state(project, pair, registration) -> dict[str, Any]:
    try:
        record = registration.load(project.id, pair.id)
    except KeyError:
        record = None
    pose = None if record is None else record.get("pose")
    pending_review_count = (
        0 if record is None else len(record.get("pending_match_reviews", []))
    )
    evidence_newer = bool(
        record and registration.evidence_is_stale(project, pair, record)
    )
    if record is None:
        status = "not-run"
    elif pose is None or not pose.get("accepted"):
        status = "review"
    elif pending_review_count or evidence_newer:
        status = "stale"
    else:
        status = "accepted"
    return {
        "pair_id": pair.id,
        "reference_image_id": pair.reference_image_id,
        "source_image_id": pair.source_image_id,
        "status": status,
        "revision": None if record is None else record["revision"],
        "accepted": bool(pose and pose.get("accepted")),
        "num_inliers": 0 if pose is None else pose["num_inliers"],
        "quality_score": (
            None if pose is None else pose["quality"]["raw_quality_score"]
        ),
        "pending_review_count": pending_review_count,
        "evidence_newer_than_pose": evidence_newer,
        "baseline_m": pair.baseline_m,
    }


def _largest_component_size(project: ProjectRecord) -> int:
    edges = [(pair.reference_image_id, pair.source_image_id) for pair in project.pairs]
    return _component_size([item.id for item in project.images], edges)


def _largest_accepted_component_size(project, pair_states) -> int:
    accepted_ids = {
        item["pair_id"] for item in pair_states if item["status"] == "accepted"
    }
    edges = [
        (pair.reference_image_id, pair.source_image_id)
        for pair in project.pairs
        if pair.id in accepted_ids
    ]
    return _component_size([item.id for item in project.images], edges)


def _component_size(nodes: list[str], edges: list[tuple[str, str]]) -> int:
    if not nodes:
        return 0
    adjacency = {node: set() for node in nodes}
    for left, right in edges:
        adjacency[left].add(right)
        adjacency[right].add(left)
    seen: set[str] = set()
    largest = 0
    for start in nodes:
        if start in seen:
            continue
        stack = [start]
        size = 0
        while stack:
            node = stack.pop()
            if node in seen:
                continue
            seen.add(node)
            size += 1
            stack.extend(adjacency[node] - seen)
        largest = max(largest, size)
    return largest


def _recommended_action(
    pair_states,
    graph_eligible: bool,
    map_current: bool,
    map_metric: bool,
    bound_range_prior_count: int,
    range_prior_min_count: int,
    dense_scene_eligible: bool,
    global_map: dict[str, Any] | None,
) -> dict[str, Any]:
    priorities = (
        ("stale", "Reestimar par editado"),
        ("review", "Corrigir par fraco"),
    )
    for status, label in priorities:
        pair = next((item for item in pair_states if item["status"] == status), None)
        if pair is not None:
            return {
                "kind": status,
                "pair_id": pair["pair_id"],
                "label": label,
                "detail": {
                    "stale": "Há decisões humanas ainda não incorporadas à pose.",
                    "review": "A hipótese não passou por todos os gates geométricos.",
                    "not-run": "Este par ainda não produziu evidência de pose.",
                }[status],
            }
    unrun = next((item for item in pair_states if item["status"] == "not-run"), None)
    if (
        graph_eligible
        and global_map
        and not global_map.get("success")
        and global_map["recommended_edit"]["kind"] in {"add-edge", "close-loop"}
        and unrun is not None
    ):
        return {
            "kind": "not-run",
            "pair_id": unrun["pair_id"],
            "label": global_map["recommended_edit"]["title"],
            "detail": global_map["recommended_edit"]["detail"],
        }
    if graph_eligible and not map_current:
        return {
            "kind": "global-map",
            "pair_id": None,
            "label": "Executar mapa global",
            "detail": (
                global_map["recommended_edit"]["detail"]
                if global_map and not global_map.get("success")
                else "As arestas aceitas já formam um componente elegível."
            ),
        }
    if map_current and not map_metric:
        ready = bound_range_prior_count >= range_prior_min_count
        return {
            "kind": "rerun-map-for-scale" if ready else "scale",
            "pair_id": None,
            "label": ("Aplicar âncoras ao mapa" if ready else "Definir escala métrica"),
            "detail": (
                f"{bound_range_prior_count} âncoras estão ligadas a features; "
                "execute novamente o mapa para aplicar SphericalRangePrior."
                if ready
                else (
                    f"Adicione {range_prior_min_count - bound_range_prior_count} "
                    "medida(s) radial(is) em inliers distintos ou informe uma baseline."
                )
            ),
        }
    if dense_scene_eligible:
        return {
            "kind": "dense-scene",
            "pair_id": None,
            "label": "Cena pronta para reconstrução densa",
            "detail": ("Revise as máscaras por panorama antes de executar MVS e TSDF."),
        }
    if map_current:
        return {
            "kind": "inspect-map",
            "pair_id": None,
            "label": "Inspecionar mapa global",
            "detail": global_map["recommended_edit"]["detail"],
        }
    if unrun is not None:
        return {
            "kind": "not-run",
            "pair_id": unrun["pair_id"],
            "label": "Executar próximo par",
            "detail": "Este par pode ampliar ou reforçar o componente global.",
        }
    return {
        "kind": "pairs",
        "pair_id": None,
        "label": "Ampliar componente aceito",
        "detail": "Aceite pares que conectem ao menos três panoramas.",
    }


def _dense_scene_blocker_detail(
    blockers: list[str], bound_count: int, required_count: int
) -> str:
    details = {
        "current-global-map-required": "execute um mapa global atual",
        "metric-scale-required": (
            f"estabeleça escala métrica ({bound_count}/{required_count} "
            "âncoras radiais ligadas)"
        ),
        "three-registered-views-required": "conecte ao menos três vistas",
    }
    return "Bloqueada: " + "; ".join(details[item] for item in blockers) + "."


def _profiles() -> dict[str, dict[str, Any]]:
    relative = SphericalRelativePose.reference().describe()
    pair_dense = PanoramicDenseStereo().describe()
    scene_dense = SphericalDenseReconstruction.reviewed_profile(
        voxel_size=0.05
    ).describe()
    return {
        "relative_pose": _profile(relative),
        "pair_dense": _profile(pair_dense),
        "scene_dense": _profile(scene_dense),
    }


def _profile(value: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value.get(key)
        for key in (
            "application",
            "interface",
            "stability",
            "profile_id",
            "profile_status",
            "input",
            "output",
            "route",
            "scale",
            "translation",
        )
        if key in value
    }
