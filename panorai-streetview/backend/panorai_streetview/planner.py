"""Deterministic build routing for the four supported intake situations."""

from __future__ import annotations

from .engines import engine_capabilities
from .models import BuildPlan, BuildStage, ProjectRecord, is_valid_registered_pose
from .store import SAFE_ID
from .stitching import _horizontal_fov


def _stage(
    stage_id: str,
    title: str,
    status: str,
    engine: str,
    stability: str,
    detail: str,
) -> BuildStage:
    return BuildStage(
        id=stage_id,
        title=title,
        status=status,
        engine=engine,
        stability=stability,
        detail=detail,
    )


def _has_registered_poses(project: ProjectRecord) -> bool:
    if not project.images:
        return False
    return all(
        is_valid_registered_pose(image.metadata.get("pose")) for image in project.images
    )


def _has_pinhole_fov(project: ProjectRecord) -> bool:
    if not project.images:
        return False
    try:
        return all(_horizontal_fov(image) is not None for image in project.images)
    except (OverflowError, TypeError, ValueError):
        return False


def _has_valid_pinhole_metadata(project: ProjectRecord) -> bool:
    """Accept absent FOVs while rejecting camera metadata the engine cannot use."""

    if not project.images:
        return False
    try:
        for image in project.images:
            _horizontal_fov(image)
    except (OverflowError, TypeError, ValueError):
        return False
    return True


def _perspective_groups_ready(project: ProjectRecord) -> bool:
    explicit = any(
        "capture_group" in image.metadata or "group_id" in image.metadata
        for image in project.images
    )
    counts: dict[str, int] = {}
    for image in project.images:
        raw = image.metadata.get("capture_group", image.metadata.get("group_id"))
        if explicit and raw is None:
            return False
        group_id = "capture-1" if raw is None else str(raw).strip()
        if not SAFE_ID.fullmatch(group_id):
            return False
        counts[group_id] = counts.get(group_id, 0) + 1
    return bool(counts) and all(count >= 2 for count in counts.values())


def build_plan(project: ProjectRecord) -> BuildPlan:
    capabilities = engine_capabilities()["engines"]
    stages = [
        _stage(
            "validate-inputs",
            "Validar imagens e proveniência",
            "ready",
            "panorai-streetview.application",
            "application",
            "Confere leitura, dimensões, SHA-256 e metadados sem inferir validade pelo valor dos pixels.",
        )
    ]
    warnings: list[str] = []

    if project.workflow == "spherical_registered":
        poses_ready = _has_registered_poses(project)
        pairs_ready = bool(project.pairs)
        stages.extend(
            [
                _stage(
                    "registered-poses",
                    "Importar poses registradas",
                    "ready" if poses_ready else "requires_input",
                    "panorai-streetview.application",
                    "application",
                    "Usa R_world_to_panorama e center_world fornecidos; não reestima nem altera a referência.",
                ),
                _stage(
                    "registered-pairs",
                    "Validar grafo de pares",
                    "ready" if pairs_ready else "requires_input",
                    "panorai-streetview.application",
                    "application",
                    f"{len(project.pairs)} par(es) explícito(s) conectam as vistas sem inferir vizinhança pela ordem de upload.",
                ),
                _stage(
                    "scene-manifest",
                    "Montar grafo navegável",
                    "ready" if poses_ready and pairs_ready else "planned",
                    "panorai-streetview.application",
                    "application",
                    "Gera nós e vizinhanças auditáveis preservando as poses de entrada.",
                ),
            ]
        )
        can_build = poses_ready and pairs_ready
        next_action = (
            "Gerar o manifesto navegável."
            if can_build
            else (
                "Forneça uma pose completa para cada panorama."
                if not poses_ready
                else "Declare ao menos um par de navegação entre panoramas."
            )
        )
    elif project.workflow == "spherical_unregistered":
        feature_ready = capabilities["features"]["available"]
        pose_ready = capabilities["relative_pose"]["available"]
        mapper_ready = capabilities["global_mapper"]["available"]
        pairs_ready = bool(project.pairs)
        stages.extend(
            [
                _stage(
                    "candidate-pairs",
                    "Preparar pares candidatos",
                    "ready" if pairs_ready else "requires_input",
                    "panorai-streetview.application",
                    "application",
                    f"{len(project.pairs)} par(es) candidato(s), sem consumir pose, baseline ou overlap de ground truth.",
                ),
                _stage(
                    "spherical-features",
                    "Extrair e casar features esféricas",
                    "planned" if feature_ready else "blocked",
                    capabilities["features"]["public_api"],
                    "stable",
                    "Perfil de pose relativa com bearings no quadro do panorama.",
                ),
                _stage(
                    "relative-poses",
                    "Estimar poses relativas",
                    "planned" if pose_ready else "blocked",
                    capabilities["relative_pose"]["public_api"],
                    "experimental",
                    "R e direção de t; escala métrica não é observável por um par.",
                ),
                _stage(
                    "global-map",
                    "Resolver reconstrução global",
                    "planned" if mapper_ready else "blocked",
                    capabilities["global_mapper"]["public_api"],
                    "experimental",
                    "Rotação global, centros em escala arbitrária, tracks, pontos e bundle adjustment esférico.",
                ),
            ]
        )
        can_build = False
        next_action = (
            "Execute features, pose relativa e mapa global pelas APIs públicas PanorAi; "
            "as poses de ground truth não fazem parte deste projeto."
            if pairs_ready
            else "Defina pares candidatos sem fornecer poses."
        )
        warnings.append(
            "A reconstrução automática permanece Experimental e pode abstir por geometria insuficiente."
        )
    elif project.workflow == "perspective_calibrated":
        stitch_ready = capabilities["perspective_stitching"]["available"]
        camera_ready = _has_pinhole_fov(project)
        groups_ready = _perspective_groups_ready(project)
        stages.extend(
            [
                _stage(
                    "camera-metadata",
                    "Validar câmera pinhole",
                    "ready" if camera_ready else "requires_input",
                    "panorai.calibration.PinholeIntrinsics",
                    "experimental",
                    "Cada imagem precisa de horizontal_fov_deg ou intrinsics.fx; distorção deve ser none.",
                ),
                _stage(
                    "stitch-to-erp",
                    "Converter grupos para ERP",
                    "planned" if stitch_ready and camera_ready else "blocked",
                    capabilities["perspective_stitching"]["public_api"],
                    "experimental",
                    "Detecta overlap, estima orientações SO(3), transforma suporte/validade e gera ERP parcial.",
                ),
            ]
        )
        can_build = stitch_ready and camera_ready and groups_ready
        next_action = (
            "Execute PanoramaStitcher e revise overlap, calibração e suporte do ERP."
            if can_build
            else (
                "Informe horizontal_fov_deg ou intrinsics.fx para cada imagem pinhole."
                if not camera_ready
                else "Cada capture_group precisa de pelo menos duas fotos."
            )
        )
    else:
        calibration_ready = capabilities["camera_calibration"]["available"]
        stitch_ready = capabilities["perspective_stitching"]["available"]
        groups_ready = _perspective_groups_ready(project)
        camera_metadata_ready = _has_valid_pinhole_metadata(project)
        stages.extend(
            [
                _stage(
                    "robust-calibration",
                    "Estimar FOV e rotações",
                    (
                        "planned"
                        if calibration_ready and camera_metadata_ready
                        else "requires_input"
                        if not camera_metadata_ready
                        else "blocked"
                    ),
                    capabilities["camera_calibration"]["public_api"],
                    "experimental",
                    "Estima FOV pinhole compartilhado e rotação a partir de correspondências robustas; pode se abster.",
                ),
                _stage(
                    "stitch-to-erp",
                    "Converter grupos para ERP",
                    "planned" if stitch_ready and camera_metadata_ready else "blocked",
                    capabilities["perspective_stitching"]["public_api"],
                    "experimental",
                    "PanoramaStitcher executa calibração observável, registro SO(3), warp e blend em uma rota pública.",
                ),
            ]
        )
        can_build = (
            calibration_ready
            and stitch_ready
            and groups_ready
            and camera_metadata_ready
        )
        next_action = (
            "Execute PanoramaStitcher; ele estimará o FOV quando o overlap tornar a calibração observável."
            if can_build
            else (
                "Corrija os metadados da câmera: use pinhole central sem distorção; o FOV pode ficar ausente."
                if not camera_metadata_ready
                else "Forneça pelo menos duas fotos com overlap em cada capture_group central rotacional."
            )
        )

    stages.append(
        _stage(
            "dense-scene",
            "Reconstrução densa opcional",
            "optional" if capabilities["dense_stereo"]["available"] else "blocked",
            capabilities["dense_stereo"]["public_api"],
            "experimental",
            (
                f"{len(project.dense_annotations)} evidência(s) humana(s) preparada(s), "
                "mas a API densa ainda não as consome; requer pose e escala compatíveis."
                if project.dense_annotations
                else "Só é elegível com poses e escala compatíveis; não define a navegação básica."
            ),
        )
    )
    return BuildPlan(
        project_id=project.id,
        workflow=project.workflow,
        can_build_now=can_build,
        next_action=next_action,
        stages=stages,
        warnings=warnings,
    )
