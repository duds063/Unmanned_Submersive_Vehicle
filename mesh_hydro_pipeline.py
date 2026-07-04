"""Mesh-to-hydrodynamics pipeline for fast design-to-simulation iteration.

This module ingests STL/OBJ geometry, optionally decimates faces, estimates
projected areas and displaced volume, and emits Fossen-friendly hydro matrices.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

try:
    import trimesh
except Exception as exc:  # pragma: no cover
    trimesh = None
    _TRIMESH_IMPORT_ERROR = exc
else:
    _TRIMESH_IMPORT_ERROR = None


@dataclass
class MeshHydroResult:
    name: str
    source_mesh_path: str
    mesh_hash_sha256: str
    units_scale_to_m: float
    original_faces: int
    decimated_faces: int
    length_m: float
    beam_m: float
    beam_total_m: float
    draft_m: float
    volume_m3: float
    area_projected_xy_m2: float
    area_projected_xz_m2: float
    area_projected_yz_m2: float
    linear_damping_matrix: np.ndarray
    quadratic_damping_matrix: np.ndarray
    added_mass_6x6: np.ndarray

    def to_vehicle_profile_payload(self) -> dict[str, Any]:
        mass_estimate = float(1025.0 * max(self.volume_m3, 1e-6))
        ixx = 0.0833 * mass_estimate * (self.beam_total_m ** 2 + self.draft_m ** 2)
        iyy = 0.0833 * mass_estimate * (self.length_m ** 2 + self.draft_m ** 2)
        izz = 0.0833 * mass_estimate * (self.length_m ** 2 + self.beam_total_m ** 2)

        return {
            "name": self.name,
            "mass_kg": mass_estimate,
            "inertia_kgm2": [
                [ixx, 0.0, 0.0],
                [0.0, iyy, 0.0],
                [0.0, 0.0, izz],
            ],
            "length_m": self.length_m,
            "beam_m": self.beam_m,
            "beam_total_m": self.beam_total_m,
            "draft_m": self.draft_m,
            "thruster_port_position_m": [-0.45 * self.length_m, 0.5 * self.beam_total_m, 0.0],
            "thruster_starboard_position_m": [-0.45 * self.length_m, -0.5 * self.beam_total_m, 0.0],
            "cog_position_m": [0.0, 0.0, -0.25 * self.draft_m],
            "cob_position_m": [0.0, 0.0, -0.5 * self.draft_m],
            "metadata": {
                "source": "mesh_hydro_pipeline",
                "mesh": {
                    "source_mesh_path": self.source_mesh_path,
                    "mesh_hash_sha256": self.mesh_hash_sha256,
                    "units_scale_to_m": self.units_scale_to_m,
                    "original_faces": self.original_faces,
                    "decimated_faces": self.decimated_faces,
                },
                "hydro": {
                    "linear_damping_matrix": self.linear_damping_matrix.tolist(),
                    "quadratic_damping_matrix": self.quadratic_damping_matrix.tolist(),
                    "added_mass_6x6": self.added_mass_6x6.tolist(),
                    "fluid_density_kg_m3": 1025.0,
                    "gravity_m_s2": 9.81,
                    "displaced_volume_m3": self.volume_m3,
                    "include_restoring": True,
                    "include_damping": True,
                    "include_added_mass": True,
                    "include_added_coriolis": True,
                },
            },
            "X_uu": float(self.quadratic_damping_matrix[0, 0]),
            "Y_vv": float(self.quadratic_damping_matrix[1, 1]),
            "Z_ww": float(self.quadratic_damping_matrix[2, 2]),
            "K_pp": float(self.quadratic_damping_matrix[3, 3]),
            "M_qq": float(self.quadratic_damping_matrix[4, 4]),
            "N_rr": float(self.quadratic_damping_matrix[5, 5]),
            "X_u": float(self.linear_damping_matrix[0, 0]),
            "Y_v": float(self.linear_damping_matrix[1, 1]),
            "Z_w": float(self.linear_damping_matrix[2, 2]),
            "K_p": float(self.linear_damping_matrix[3, 3]),
            "M_q": float(self.linear_damping_matrix[4, 4]),
            "N_r": float(self.linear_damping_matrix[5, 5]),
            "X_udot": float(self.added_mass_6x6[0, 0]),
            "Y_vdot": float(self.added_mass_6x6[1, 1]),
            "Z_wdot": float(self.added_mass_6x6[2, 2]),
            "K_pdot": float(self.added_mass_6x6[3, 3]),
            "M_qdot": float(self.added_mass_6x6[4, 4]),
            "N_rdot": float(self.added_mass_6x6[5, 5]),
        }


class MeshHydroPipeline:
    def __init__(self, fluid_density_kg_m3: float = 1025.0):
        self.fluid_density_kg_m3 = float(fluid_density_kg_m3)

    @staticmethod
    def _ensure_trimesh() -> None:
        if trimesh is None:
            raise RuntimeError(
                "trimesh is required for mesh processing. Install dependencies and retry."
            ) from _TRIMESH_IMPORT_ERROR

    @staticmethod
    def _sha256(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    @staticmethod
    def _decimate(mesh, target_faces: int):
        if target_faces <= 0:
            return mesh
        if len(mesh.faces) <= target_faces:
            return mesh
        try:
            decimated = mesh.simplify_quadratic_decimation(target_faces)
            if decimated is not None and len(decimated.faces) > 0:
                return decimated
        except Exception:
            pass
        return mesh

    @staticmethod
    def _projected_area(points_2d: np.ndarray) -> float:
        if points_2d.shape[0] < 3:
            return 0.0
        mins = np.min(points_2d, axis=0)
        maxs = np.max(points_2d, axis=0)
        spans = np.maximum(maxs - mins, 0.0)
        return float(spans[0] * spans[1])

    def process(
        self,
        mesh_path: str | Path,
        target_faces: int = 800,
        units_scale_to_m: float = 1.0,
        name: str | None = None,
    ) -> MeshHydroResult:
        self._ensure_trimesh()
        mesh_path = Path(mesh_path)
        if not mesh_path.exists():
            raise FileNotFoundError(f"Mesh file not found: {mesh_path}")

        loaded = trimesh.load_mesh(str(mesh_path), force="mesh")
        if loaded is None:
            raise ValueError(f"Could not load mesh: {mesh_path}")

        mesh = loaded.copy()
        mesh.apply_scale(float(units_scale_to_m))
        mesh.remove_unreferenced_vertices()

        if not mesh.is_watertight:
            mesh = mesh.convex_hull

        original_faces = int(len(mesh.faces))
        mesh = self._decimate(mesh, int(target_faces))
        decimated_faces = int(len(mesh.faces))

        bounds = mesh.bounds
        dims = bounds[1] - bounds[0]
        length_m = float(max(dims[0], 1e-3))
        beam_total_m = float(max(dims[1], 1e-3))
        beam_m = 0.5 * beam_total_m
        draft_m = float(max(dims[2], 1e-3))

        volume_m3 = float(abs(mesh.volume))
        if volume_m3 <= 1e-8:
            volume_m3 = float(length_m * beam_total_m * draft_m * 0.6)

        vertices = np.asarray(mesh.vertices, dtype=float)
        area_xy = self._projected_area(vertices[:, [0, 1]])
        area_xz = self._projected_area(vertices[:, [0, 2]])
        area_yz = self._projected_area(vertices[:, [1, 2]])

        q_diag = np.array([
            0.5 * self.fluid_density_kg_m3 * max(area_yz, 1e-4) * 0.12,
            0.5 * self.fluid_density_kg_m3 * max(area_xz, 1e-4) * 0.9,
            0.5 * self.fluid_density_kg_m3 * max(area_xy, 1e-4) * 0.9,
            0.5 * self.fluid_density_kg_m3 * max(area_xy, 1e-4) * (beam_total_m ** 2) * 0.08,
            0.5 * self.fluid_density_kg_m3 * max(area_yz, 1e-4) * (length_m ** 2) * 0.05,
            0.5 * self.fluid_density_kg_m3 * max(area_xz, 1e-4) * (length_m ** 2) * 0.05,
        ], dtype=float)

        l_diag = 0.12 * q_diag
        displaced_mass = self.fluid_density_kg_m3 * volume_m3
        a_diag = np.array([
            0.12 * displaced_mass,
            0.95 * displaced_mass,
            0.95 * displaced_mass,
            0.08 * displaced_mass * (beam_total_m ** 2),
            0.18 * displaced_mass * (length_m ** 2),
            0.18 * displaced_mass * (length_m ** 2),
        ], dtype=float)

        linear_damping = np.diag(l_diag)
        quadratic_damping = np.diag(q_diag)
        added_mass = np.diag(a_diag)

        coupling_strength = 0.02
        quadratic_damping[0, 5] = quadratic_damping[5, 0] = coupling_strength * q_diag[0]
        quadratic_damping[1, 5] = quadratic_damping[5, 1] = coupling_strength * q_diag[1]
        linear_damping[0, 5] = linear_damping[5, 0] = coupling_strength * l_diag[0]
        added_mass[0, 5] = added_mass[5, 0] = coupling_strength * a_diag[0]

        return MeshHydroResult(
            name=name or mesh_path.stem,
            source_mesh_path=str(mesh_path),
            mesh_hash_sha256=self._sha256(mesh_path),
            units_scale_to_m=float(units_scale_to_m),
            original_faces=original_faces,
            decimated_faces=decimated_faces,
            length_m=length_m,
            beam_m=beam_m,
            beam_total_m=beam_total_m,
            draft_m=draft_m,
            volume_m3=volume_m3,
            area_projected_xy_m2=float(area_xy),
            area_projected_xz_m2=float(area_xz),
            area_projected_yz_m2=float(area_yz),
            linear_damping_matrix=linear_damping,
            quadratic_damping_matrix=quadratic_damping,
            added_mass_6x6=added_mass,
        )
