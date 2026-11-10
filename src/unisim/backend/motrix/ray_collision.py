from __future__ import annotations

import re
from typing import Any, Sequence

import numpy as np

from unisim.backend.ray_scene import ray_collision_from_mujoco_model
from unisim.ray_query import RayCollisionDescription, RayMeshDescription


def collision_from_motrix_model(
    motrix_model: Any,
    source_model: Any,
    body_names: Sequence[str] | None = None,
) -> RayCollisionDescription:
    """Project source geoms and Motrix's compiled convex-hull colliders."""
    source = ray_collision_from_mujoco_model(source_model, body_names=body_names)
    if source.geom_mesh_ids is None:
        raise RuntimeError("Motrix ray collision projection lost the source mesh axis")
    mesh_ids = np.array(source.geom_mesh_ids, dtype=np.intp, copy=True)
    meshes = list(source.meshes)
    from motrixsim import low as motrix_low

    low_model = motrix_model.low
    shape_names = tuple(name.lower() for name in motrix_low.ColliderShape.type_names())
    convex_hull_shape = shape_names.index("convexhull")

    for geom_index, geom_name in enumerate(source.geom_names):
        if mesh_ids[geom_index] < 0:
            continue

        source_geom_id = _source_geom_id(source_model, geom_name)
        if not (
            int(source_model.geom_contype[source_geom_id])
            or int(source_model.geom_conaffinity[source_geom_id])
        ):
            continue

        native_geom = motrix_model.get_geom(geom_name)
        if native_geom is None:
            native_geom = next(
                (
                    geom
                    for geom in motrix_model.geoms
                    if geom.name is None and int(geom.index) == source_geom_id
                ),
                None,
            )
        if native_geom is None:
            raise NotImplementedError(
                f"Motrix ray mesh {geom_name!r} is absent from the compiled model"
            )
        native_pose = np.asarray(native_geom.local_pose, dtype=np.float64)
        same_pose_meshes = []
        for other_index, other_name in enumerate(source.geom_names):
            if other_index == geom_index or mesh_ids[other_index] < 0:
                continue
            other_source_id = _source_geom_id(source_model, other_name)
            if not (
                int(source_model.geom_contype[other_source_id])
                or int(source_model.geom_conaffinity[other_source_id])
            ):
                continue
            other_geom = motrix_model.get_geom(other_name)
            if other_geom is None:
                continue
            same_link = (
                (native_geom.link is None and other_geom.link is None)
                or (
                    native_geom.link is not None
                    and other_geom.link is not None
                    and int(native_geom.link.index) == int(other_geom.link.index)
                )
            )
            if same_link and np.allclose(
                np.asarray(other_geom.local_pose), native_pose, rtol=0.0, atol=1e-6
            ):
                same_pose_meshes.append(other_name)
        if same_pose_meshes:
            raise NotImplementedError(
                f"Motrix ray mesh {geom_name!r} shares its collider pose with "
                f"another active mesh {same_pose_meshes[0]!r}"
            )
        if native_geom.link is None:
            parent_ids = {0}
        else:
            link_index = int(native_geom.link.index)
            if link_index < 0 or link_index >= len(low_model.link_low_maps):
                raise NotImplementedError(
                    f"Motrix ray mesh {geom_name!r} has no low-level link mapping"
                )
            link_map = low_model.link_low_maps[link_index]
            if link_map.ty != "link":
                raise NotImplementedError(
                    f"Motrix ray mesh {geom_name!r} maps to unsupported {link_map.ty!r} state"
                )
            parent_ids = set(range(int(link_map.index), int(link_map.index + link_map.num)))

        source_pos = np.asarray(source_model.geom_pos[source_geom_id], dtype=np.float64)
        source_quat = np.asarray(source_model.geom_quat[source_geom_id], dtype=np.float64)
        source_rotation = _rotation_matrix(source_quat)
        native_pos = native_pose[:3]
        native_quat_xyzw = native_pose[3:]
        candidates = []
        for collider in low_model.colliders:
            if (
                collider.parent.ty != "link"
                or int(collider.parent.index) not in parent_ids
                or int(collider.shape) != convex_hull_shape
            ):
                continue
            if not np.allclose(
                np.asarray(collider.translation, dtype=np.float64),
                native_pos,
                rtol=0.0,
                atol=1e-6,
            ):
                continue
            collider_quat_xyzw = np.asarray(collider.rotation, dtype=np.float64)
            if min(
                np.linalg.norm(collider_quat_xyzw - native_quat_xyzw),
                np.linalg.norm(collider_quat_xyzw + native_quat_xyzw),
            ) > 1e-5:
                continue
            candidates.append(collider)

        if not candidates:
            raise NotImplementedError(
                f"Motrix ray mesh {geom_name!r} has no matching compiled convex-hull collider"
            )

        source_mesh = meshes[int(mesh_ids[geom_index])]
        replacement_points: list[np.ndarray] = []
        replacement_indices: list[np.ndarray] = []
        vertex_offset = 0
        for collider in candidates:
            hull_index = int(collider.asset_index)
            if hull_index < 0 or hull_index >= len(low_model.convex_hulls):
                raise NotImplementedError(
                    f"Motrix ray mesh {geom_name!r} has an invalid convex-hull asset"
                )
            hull = low_model.convex_hulls[hull_index]
            points = np.asarray(hull.points, dtype=np.float64)
            scale = np.asarray(collider.mesh_scale, dtype=np.float64)
            translation = np.asarray(collider.translation, dtype=np.float64)
            collider_rotation = _rotation_matrix(
                np.asarray(collider.rotation, dtype=np.float64)[[3, 0, 1, 2]]
            )
            if (
                points.ndim != 2
                or points.shape[1] != 3
                or scale.shape != (3,)
                or translation.shape != (3,)
                or np.any(scale == 0.0)
                or not np.isfinite(points).all()
                or not np.isfinite(scale).all()
                or not np.isfinite(translation).all()
            ):
                raise NotImplementedError(f"Motrix ray mesh {geom_name!r} has invalid hull data")

            # Motrix stores hull points in collider-local coordinates; ray meshes
            # use the source geom frame, so transform through the shared link frame.
            points_link = (points * scale) @ collider_rotation.T + translation
            points_geom = (points_link - source_pos) @ source_rotation
            indices = _triangulate_hull(hull.face_vertex_indices, hull.face_indices_start)
            replacement_points.append(points_geom.astype(np.float32))
            replacement_indices.append(indices + vertex_offset)
            vertex_offset += points.shape[0]

        native_mesh = RayMeshDescription(
            asset_id=f"motrix:{geom_name}:{source_mesh.asset_id}",
            points=np.concatenate(replacement_points),
            indices=np.concatenate(replacement_indices),
        )
        mesh_ids[geom_index] = len(meshes)
        meshes.append(native_mesh)

    return RayCollisionDescription(
        scene=source.scene,
        geom_names=source.geom_names,
        geom_tags=source.geom_tags,
        meshes=tuple(meshes),
        geom_mesh_ids=mesh_ids,
        body_parent_ids=source.body_parent_ids,
    )


def _rotation_matrix(quat_wxyz: np.ndarray) -> np.ndarray:
    quat = np.asarray(quat_wxyz, dtype=np.float64)
    if quat.shape != (4,) or not np.isfinite(quat).all():
        raise NotImplementedError("Motrix ray geometry has an invalid quaternion")
    norm = float(np.linalg.norm(quat))
    if norm <= 0.0:
        raise NotImplementedError("Motrix ray geometry has a zero quaternion")
    w, x, y, z = quat / norm
    return np.asarray(
        (
            (1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)),
            (2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)),
            (2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)),
        ),
        dtype=np.float64,
    )


def _source_geom_id(source_model: Any, geom_name: str) -> int:
    try:
        return int(source_model.geom(geom_name).id)
    except KeyError:
        match = re.fullmatch(r"geom_(\d+)", geom_name)
        if match is None:
            raise NotImplementedError(
                f"Motrix ray mesh {geom_name!r} has no source geom identity"
            ) from None
        geom_id = int(match.group(1))
        if geom_id < 0 or geom_id >= int(source_model.ngeom):
            raise NotImplementedError(
                f"Motrix ray mesh {geom_name!r} has an invalid source geom identity"
            ) from None
        return geom_id


def _triangulate_hull(
    face_vertex_indices: Sequence[int], face_indices_start: Sequence[int]
) -> np.ndarray:
    flat = np.asarray(face_vertex_indices, dtype=np.int32)
    starts = np.asarray(face_indices_start, dtype=np.intp)
    if flat.ndim != 1 or starts.ndim != 1 or starts.size == 0:
        raise NotImplementedError("Motrix convex hull has invalid face topology")
    ends = np.concatenate((starts[1:], np.asarray((flat.size,), dtype=np.intp)))
    triangles: list[tuple[int, int, int]] = []
    for start, end in zip(starts, ends, strict=True):
        face = flat[int(start) : int(end)]
        if face.size < 3:
            raise NotImplementedError(
                "Motrix convex hull has a face with fewer than three vertices"
            )
        triangles.extend(
            (int(face[0]), int(face[index]), int(face[index + 1]))
            for index in range(1, face.size - 1)
        )
    return np.asarray(triangles, dtype=np.int32).reshape(-1)
