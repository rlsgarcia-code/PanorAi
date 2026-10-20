"""Validated persistence models for Street View projects."""

from __future__ import annotations

from datetime import UTC, datetime
import math
from typing import Any, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    FiniteFloat,
    field_validator,
    model_validator,
)


Workflow = Literal[
    "spherical_registered",
    "spherical_unregistered",
    "perspective_calibrated",
    "perspective_uncalibrated",
]


def _require_finite_json(value: Any, *, path: str) -> Any:
    if isinstance(value, dict):
        for key, item in value.items():
            _require_finite_json(item, path=f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _require_finite_json(item, path=f"{path}[{index}]")
    elif isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"{path} must contain only finite numbers")
    return value


def is_valid_registered_pose(value: Any, *, tolerance: float = 1e-6) -> bool:
    """Return whether ``value`` is a finite rigid world-to-panorama pose.

    The MVP accepts rotation matrices only.  Rejecting scaled, reflected, or
    non-finite matrices here prevents an invalid registration from becoming a
    buildable scene merely because its JSON has the expected shape.
    """

    if not isinstance(value, dict):
        return False
    rotation = value.get("rotation_world_to_panorama")
    center = value.get("center_world")
    if not (
        isinstance(rotation, list)
        and len(rotation) == 3
        and all(isinstance(row, list) and len(row) == 3 for row in rotation)
        and isinstance(center, list)
        and len(center) == 3
    ):
        return False
    try:
        rows = [[float(component) for component in row] for row in rotation]
        centre = [float(component) for component in center]
    except (OverflowError, TypeError, ValueError):
        return False
    if not all(math.isfinite(component) for row in rows for component in row):
        return False
    if not all(math.isfinite(component) for component in centre):
        return False
    for row_index in range(3):
        for other_index in range(3):
            dot = sum(
                rows[row_index][component] * rows[other_index][component]
                for component in range(3)
            )
            expected = 1.0 if row_index == other_index else 0.0
            if abs(dot - expected) > tolerance:
                return False
    determinant = (
        rows[0][0] * (rows[1][1] * rows[2][2] - rows[1][2] * rows[2][1])
        - rows[0][1] * (rows[1][0] * rows[2][2] - rows[1][2] * rows[2][0])
        + rows[0][2] * (rows[1][0] * rows[2][1] - rows[1][1] * rows[2][0])
    )
    return abs(determinant - 1.0) <= tolerance


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


class ImageRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    filename: str
    object_name: str
    sha256: str
    media_type: str
    width: int = Field(gt=0)
    height: int = Field(gt=0)
    size_bytes: int = Field(gt=0)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("metadata")
    @classmethod
    def validate_metadata_numbers(cls, value: dict[str, Any]):
        return _require_finite_json(value, path="metadata")


class ImagePairInput(BaseModel):
    """Filename-addressed pair supplied while a project is ingested."""

    model_config = ConfigDict(extra="forbid")

    id: str = Field(min_length=1, max_length=120, pattern=r"^[a-zA-Z0-9_-]+$")
    reference_filename: str = Field(min_length=1)
    source_filename: str = Field(min_length=1)
    baseline_m: FiniteFloat | None = Field(default=None, gt=0)
    minimum_directional_overlap: float | None = Field(default=None, ge=0, le=1)
    provenance: dict[str, Any] = Field(default_factory=dict)

    @field_validator("provenance")
    @classmethod
    def validate_provenance_numbers(cls, value: dict[str, Any]):
        return _require_finite_json(value, path="provenance")

    @model_validator(mode="after")
    def validate_distinct_filenames(self):
        if self.reference_filename == self.source_filename:
            raise ValueError("a pair must reference two distinct images")
        return self


class ImagePairRecord(BaseModel):
    """Auditable relation between two ingested images."""

    model_config = ConfigDict(extra="forbid")

    id: str = Field(min_length=1, max_length=120, pattern=r"^[a-zA-Z0-9_-]+$")
    reference_image_id: str
    source_image_id: str
    relation: Literal["registered_overlap", "overlap_candidate"]
    baseline_m: FiniteFloat | None = Field(default=None, gt=0)
    minimum_directional_overlap: float | None = Field(default=None, ge=0, le=1)
    provenance: dict[str, Any] = Field(default_factory=dict)

    @field_validator("provenance")
    @classmethod
    def validate_provenance_numbers(cls, value: dict[str, Any]):
        return _require_finite_json(value, path="provenance")

    @model_validator(mode="after")
    def validate_distinct_images(self):
        if self.reference_image_id == self.source_image_id:
            raise ValueError("a pair must reference two distinct images")
        return self


class ObjectRegion(BaseModel):
    """One human-localized object instance in one source panorama."""

    model_config = ConfigDict(extra="forbid")

    id: str | None = None
    object_id: str = Field(min_length=1, max_length=80)
    label: str = Field(min_length=1, max_length=120)
    image_id: str
    bbox_xyxy: tuple[float, float, float, float]
    state: Literal["accepted", "rejected"] = "accepted"
    created_at: str | None = None

    @field_validator("bbox_xyxy")
    @classmethod
    def validate_bbox(cls, value: tuple[float, float, float, float]):
        if not all(math.isfinite(coordinate) for coordinate in value):
            raise ValueError("bounding-box coordinates must be finite")
        x0, y0, x1, y1 = value
        if x1 <= x0 or y1 <= y0:
            raise ValueError("bounding box must have positive width and height")
        return value


class ManualCorrespondence(BaseModel):
    """One operator-provided ERP pixel-center correspondence for a pair."""

    model_config = ConfigDict(extra="forbid")

    id: str | None = None
    pair_id: str = Field(min_length=1, max_length=120)
    reference_xy: tuple[float, float]
    source_xy: tuple[float, float]
    state: Literal["accepted", "rejected"] = "accepted"
    created_at: str | None = None

    @field_validator("reference_xy", "source_xy")
    @classmethod
    def validate_pixel(cls, value: tuple[float, float]):
        if not all(math.isfinite(coordinate) for coordinate in value):
            raise ValueError("manual correspondence coordinates must be finite")
        return value


class MetrologyLine(BaseModel):
    model_config = ConfigDict(extra="forbid")
    family: Literal["floor_a", "floor_b"]
    start_xy: tuple[float, float]
    end_xy: tuple[float, float]

    @field_validator("start_xy", "end_xy")
    @classmethod
    def validate_pixel(cls, value: tuple[float, float]):
        if not all(math.isfinite(coordinate) for coordinate in value):
            raise ValueError("metrology coordinates must be finite")
        return value

    @model_validator(mode="after")
    def validate_extent(self):
        if self.start_xy == self.end_xy:
            raise ValueError(
                "a floor direction observation requires two distinct points"
            )
        return self


class FloorCalibrationRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    image_id: str
    camera_height_m: FiniteFloat = Field(gt=0)
    floor_hint_xy: tuple[float, float]
    lines: list[MetrologyLine] = Field(min_length=4)
    max_residual_deg: FiniteFloat = Field(default=3.0, gt=0, lt=90)

    @field_validator("floor_hint_xy")
    @classmethod
    def validate_floor_hint(cls, value: tuple[float, float]):
        if not all(math.isfinite(coordinate) for coordinate in value):
            raise ValueError("floor hint coordinates must be finite")
        return value


class MetrologyFamilyRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    family: Literal["floor_a", "floor_b"]
    direction_panorama: tuple[float, float, float]
    inlier_indices: list[int]
    outlier_indices: list[int]
    residuals_deg: list[float]
    median_residual_deg: float
    p90_residual_deg: float


class FloorCalibrationRecord(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    schema_name: Literal["panorai-streetview-floor-calibration/v1"] = Field(
        default="panorai-streetview-floor-calibration/v1", alias="schema"
    )
    project_id: str
    image_id: str
    revision: int = Field(ge=1)
    created_at: str
    engine: dict[str, Any]
    shape_hw: tuple[int, int]
    camera_height_m: FiniteFloat = Field(gt=0)
    floor_hint_xy: tuple[float, float]
    lines: list[MetrologyLine]
    rotation_panorama_to_world: list[list[float]]
    floor_normal_panorama: tuple[float, float, float]
    family_separation_deg: float
    floor_normal_uncertainty_deg: float
    families: list[MetrologyFamilyRecord]


class FloorMeasurementRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    image_id: str
    calibration_revision: int = Field(ge=1)
    point_xy: tuple[float, float]


class VerticalMeasurementRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    image_id: str
    calibration_revision: int = Field(ge=1)
    base_xy: tuple[float, float]
    top_xy: tuple[float, float]
    max_horizontal_residual_m: FiniteFloat | None = Field(default=0.15, gt=0)


class DenseAnnotation(BaseModel):
    """Human dense evidence with an explicit public-engine binding state."""

    model_config = ConfigDict(extra="forbid")

    id: str | None = None
    kind: Literal["include_region", "exclude_region", "range_anchor", "surface"]
    image_id: str
    label: str = Field(default="dense-evidence", min_length=1, max_length=120)
    points_xy: list[tuple[float, float]]
    radial_range: FiniteFloat | None = Field(default=None, gt=0)
    feature_index: int | None = Field(default=None, ge=0)
    binding_pair_id: str | None = None
    binding_distance_px: FiniteFloat | None = Field(default=None, ge=0)
    engine_effect: Literal["staged", "pair_validity", "range_prior"] = "staged"
    state: Literal["accepted", "rejected"] = "accepted"
    created_at: str | None = None

    @model_validator(mode="before")
    @classmethod
    def normalize_pair_validity_effect(cls, value):
        if (
            isinstance(value, dict)
            and value.get("kind")
            in {
                "include_region",
                "exclude_region",
            }
            and value.get("engine_effect", "staged") == "staged"
        ):
            return {**value, "engine_effect": "pair_validity"}
        return value

    @model_validator(mode="after")
    def validate_geometry(self):
        expected = 1 if self.kind == "range_anchor" else 2
        if len(self.points_xy) != expected:
            raise ValueError(f"{self.kind} requires {expected} point(s)")
        if any(
            not math.isfinite(coordinate)
            for point in self.points_xy
            for coordinate in point
        ):
            raise ValueError("dense annotation coordinates must be finite")
        if expected == 2 and self.points_xy[0] == self.points_xy[1]:
            raise ValueError("dense region corners must be distinct")
        if self.kind == "range_anchor" and self.radial_range is None:
            raise ValueError("range_anchor requires radial_range")
        if self.kind != "range_anchor" and self.radial_range is not None:
            raise ValueError("only range_anchor accepts radial_range")
        if self.kind != "range_anchor" and any(
            value is not None
            for value in (
                self.feature_index,
                self.binding_pair_id,
                self.binding_distance_px,
            )
        ):
            raise ValueError("only range_anchor accepts a feature binding")
        if self.engine_effect == "range_prior" and (
            self.kind != "range_anchor" or self.feature_index is None
        ):
            raise ValueError("range_prior effect requires a bound range anchor")
        if self.engine_effect == "pair_validity" and self.kind not in {
            "include_region",
            "exclude_region",
        }:
            raise ValueError(
                "pair_validity effect requires an include or exclude region"
            )
        return self


class ProjectRecord(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    schema_name: Literal["panorai-streetview-project/v1"] = Field(
        default="panorai-streetview-project/v1", alias="schema"
    )
    id: str
    name: str = Field(min_length=1, max_length=120)
    workflow: Workflow
    created_at: str
    updated_at: str
    panorai_runtime: dict[str, Any]
    images: list[ImageRecord]
    pairs: list[ImagePairRecord] = Field(default_factory=list)
    object_regions: list[ObjectRegion] = Field(default_factory=list)
    manual_correspondences: list[ManualCorrespondence] = Field(default_factory=list)
    dense_annotations: list[DenseAnnotation] = Field(default_factory=list)
    status: Literal["draft", "ready", "blocked", "built"] = "draft"

    @model_validator(mode="after")
    def validate_references(self):
        image_ids = {image.id for image in self.images}
        if len(image_ids) != len(self.images):
            raise ValueError("image IDs must be unique")
        pair_ids = {pair.id for pair in self.pairs}
        if len(pair_ids) != len(self.pairs):
            raise ValueError("pair IDs must be unique")
        if any(
            pair.reference_image_id not in image_ids
            or pair.source_image_id not in image_ids
            for pair in self.pairs
        ):
            raise ValueError("pair references an unknown image")
        if any(item.image_id not in image_ids for item in self.object_regions):
            raise ValueError("object region references an unknown image")
        if any(item.image_id not in image_ids for item in self.dense_annotations):
            raise ValueError("dense annotation references an unknown image")
        if any(item.pair_id not in pair_ids for item in self.manual_correspondences):
            raise ValueError("manual correspondence references an unknown pair")
        return self


class RegistrationRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    region_mode: Literal["all", "prioritize", "restrict"] = "all"


class DenseSceneRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    profile: Literal["interactive", "reviewed"] = "interactive"


class MatchReview(BaseModel):
    model_config = ConfigDict(extra="forbid")

    state: Literal["unreviewed", "accepted", "rejected"]
    revision: int = Field(ge=1)


class BuildStage(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    title: str
    status: Literal["ready", "requires_input", "blocked", "planned", "optional"]
    engine: str
    stability: Literal["stable", "experimental", "application"]
    detail: str


class BuildPlan(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    schema_name: Literal["panorai-streetview-build-plan/v1"] = Field(
        default="panorai-streetview-build-plan/v1", alias="schema"
    )
    project_id: str
    workflow: Workflow
    can_build_now: bool
    next_action: str
    stages: list[BuildStage]
    warnings: list[str] = Field(default_factory=list)
