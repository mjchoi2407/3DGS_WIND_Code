"""Teacher trajectory conversion, fixtures, and label generation."""

from .cloth_metrics import (
    DEFAULT_REFERENCE_SURFACE_DENSITY_KG_M2,
    ClothMetricError,
    ClothMetricSpec,
    make_cloth_metric_spec,
    mesh_triangle_areas_m2,
)
from .physics_registry import (
    ArtifactReference,
    TeacherPhysicsError,
    TeacherPhysicsRegistry,
    TractionIdentity,
    validate_source_membership,
    validate_traction_identity,
)
from .initial_state import TeacherInitialDisplacement, make_cantilever_initial_displacement
from .sample_meshes import (
    DEFAULT_DIMENSIONS_M,
    MeshValidationReport,
    SampleClothMesh,
    SampleMeshError,
    SampleMeshKind,
    export_sample_mesh,
    make_handkerchief,
    make_rectangular_flag,
    make_sample_mesh,
    make_triangular_flag,
    mesh_edges,
    validate_sample_mesh,
    write_sample_npz,
    write_sample_obj,
)
from .trajectory import TeacherTrajectoryError, WindSample, held_force_work
from .trajectory_io import TeacherTrajectoryArtifact, inspect_teacher_run
from .wind_programs import CompiledWindProgram, WindProgram, WindProgramError, WindSegment
from .newton_wind_runner import TeacherWindSuiteArtifact, inspect_teacher_wind_suite, run_teacher_wind_suite
from .common_probes import ProbeMappingPolicy, TeacherProbeSet
from .teacher_probe_map import TeacherProbeMap, TeacherProbeMappingError, build_teacher_probe_map, inspect_teacher_probe_artifact
from .probe_trajectory import TeacherProbeTrajectoryArtifact, extract_teacher_probe_trajectory, iter_teacher_probe_chunks

__all__ = [
    "ProbeMappingPolicy",
    "TeacherProbeSet",
    "TeacherProbeMap",
    "TeacherProbeMappingError",
    "build_teacher_probe_map",
    "inspect_teacher_probe_artifact",
    "TeacherProbeTrajectoryArtifact",
    "extract_teacher_probe_trajectory",
    "iter_teacher_probe_chunks",
    "TeacherInitialDisplacement",
    "make_cantilever_initial_displacement",
    "CompiledWindProgram",
    "WindProgram",
    "WindProgramError",
    "WindSegment",
    "TeacherWindSuiteArtifact",
    "inspect_teacher_wind_suite",
    "run_teacher_wind_suite",
    "TeacherTrajectoryArtifact",
    "TeacherTrajectoryError",
    "WindSample",
    "held_force_work",
    "inspect_teacher_run",
    "ArtifactReference",
    "TeacherPhysicsError",
    "TeacherPhysicsRegistry",
    "TractionIdentity",
    "validate_source_membership",
    "validate_traction_identity",
    "DEFAULT_REFERENCE_SURFACE_DENSITY_KG_M2",
    "ClothMetricError",
    "ClothMetricSpec",
    "DEFAULT_DIMENSIONS_M",
    "MeshValidationReport",
    "SampleClothMesh",
    "SampleMeshError",
    "SampleMeshKind",
    "export_sample_mesh",
    "make_cloth_metric_spec",
    "make_handkerchief",
    "make_rectangular_flag",
    "make_sample_mesh",
    "make_triangular_flag",
    "mesh_edges",
    "mesh_triangle_areas_m2",
    "validate_sample_mesh",
    "write_sample_npz",
    "write_sample_obj",
]
