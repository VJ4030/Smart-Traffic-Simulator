"""Procedural mesh construction and GPU buffers.

Everything you see in Smart Traffic India is generated from code - there are no binary
model files in the repository.  :class:`MeshBuilder` assembles primitives (boxes,
tapered boxes, cylinders, spheres and extruded profiles) into *material groups*; each
group becomes one draw call, which keeps the frame budget low even on integrated GPUs.

Concepts: geometric primitives, custom meshes, hierarchical part assembly, surface
normals, UV mapping, vertex buffers / index buffers, efficient geometry reuse.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from OpenGL import GL

Vec3 = Tuple[float, float, float]
Vec2 = Tuple[float, float]


# --------------------------------------------------------------------------------------
# materials
# --------------------------------------------------------------------------------------
@dataclass(frozen=True)
class Material:
    """Surface description used for the per-draw lighting uniforms."""

    name: str
    color: Vec3 = (0.8, 0.8, 0.8)
    specular: float = 0.15
    emissive: float = 0.0
    emissive_color: Vec3 = (1.0, 1.0, 1.0)
    texture: str = ""          # name of a GL texture bound for this surface ("" = none)

    def tinted(self, color: Vec3, name: Optional[str] = None) -> "Material":
        return Material(name or self.name, color, self.specular, self.emissive,
                        self.emissive_color, self.texture)

    def with_emissive(self, strength: float, color: Optional[Vec3] = None) -> "Material":
        return Material(self.name, self.color, self.specular, strength,
                        color or self.emissive_color, self.texture)


# --------------------------------------------------------------------------------------
# mesh data
# --------------------------------------------------------------------------------------
@dataclass
class Surface:
    """One material group: interleaved vertices + triangle indices."""

    material: Material
    positions: np.ndarray           # (N, 3) float32
    normals: np.ndarray             # (N, 3) float32
    uvs: np.ndarray                 # (N, 2) float32
    indices: np.ndarray             # (M,)  uint32

    @property
    def vertex_count(self) -> int:
        return int(self.positions.shape[0])

    @property
    def triangle_count(self) -> int:
        return int(self.indices.shape[0] // 3)

    def interleaved(self) -> np.ndarray:
        return np.hstack([self.positions, self.normals, self.uvs]).astype(np.float32)

    def transformed(self, matrix: np.ndarray) -> "Surface":
        """Bake a model transform into the vertices (used for hierarchical parts)."""
        from .transform import normal_matrix

        pos = np.hstack([self.positions, np.ones((self.vertex_count, 1), dtype=np.float32)])
        new_pos = (matrix @ pos.T).T[:, :3].astype(np.float32)
        nmat = normal_matrix(matrix)
        new_norm = (nmat @ self.normals.T).T.astype(np.float32)
        lengths = np.linalg.norm(new_norm, axis=1, keepdims=True)
        lengths[lengths < 1e-8] = 1.0
        new_norm = new_norm / lengths
        return Surface(self.material, new_pos, new_norm, self.uvs.copy(), self.indices.copy())


@dataclass
class MeshData:
    """A complete model: one or more material surfaces plus cached bounds."""

    surfaces: List[Surface] = field(default_factory=list)
    name: str = "mesh"

    @property
    def triangle_count(self) -> int:
        return sum(s.triangle_count for s in self.surfaces)

    def bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        if not self.surfaces:
            z = np.zeros(3, dtype=np.float32)
            return z, z.copy()
        mins = np.array([s.positions.min(axis=0) for s in self.surfaces])
        maxs = np.array([s.positions.max(axis=0) for s in self.surfaces])
        return mins.min(axis=0), maxs.max(axis=0)

    def radius(self) -> float:
        lo, hi = self.bounds()
        return float(np.linalg.norm(hi - lo) * 0.5)

    def transformed(self, matrix: np.ndarray) -> "MeshData":
        return MeshData([s.transformed(matrix) for s in self.surfaces], self.name)

    def merge(self, other: "MeshData") -> "MeshData":
        """Merge two models, combining surfaces that share a material."""
        table: Dict[str, Surface] = {}
        for surf in list(self.surfaces) + list(other.surfaces):
            key = surf.material.name
            if key in table:
                prev = table[key]
                offset = prev.vertex_count
                table[key] = Surface(
                    prev.material,
                    np.vstack([prev.positions, surf.positions]),
                    np.vstack([prev.normals, surf.normals]),
                    np.vstack([prev.uvs, surf.uvs]),
                    np.concatenate([prev.indices, surf.indices + offset]).astype(np.uint32),
                )
            else:
                table[key] = surf
        return MeshData(list(table.values()), f"{self.name}+{other.name}")


# --------------------------------------------------------------------------------------
# builder
# --------------------------------------------------------------------------------------
class _Group:
    __slots__ = ("material", "positions", "normals", "uvs", "indices")

    def __init__(self, material: Material) -> None:
        self.material = material
        self.positions: List[float] = []
        self.normals: List[float] = []
        self.uvs: List[float] = []
        self.indices: List[int] = []

    def vertex(self, pos: Vec3, normal: Vec3, uv: Vec2) -> int:
        self.positions.extend(pos)
        self.normals.extend(normal)
        self.uvs.extend(uv)
        return len(self.positions) // 3 - 1

    def to_surface(self) -> Surface:
        return Surface(
            self.material,
            np.array(self.positions, dtype=np.float32).reshape(-1, 3),
            np.array(self.normals, dtype=np.float32).reshape(-1, 3),
            np.array(self.uvs, dtype=np.float32).reshape(-1, 2),
            np.array(self.indices, dtype=np.uint32),
        )


def _cross(a: Sequence[float], b: Sequence[float]) -> Vec3:
    return (a[1] * b[2] - a[2] * b[1],
            a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0])


def _sub(a: Sequence[float], b: Sequence[float]) -> Vec3:
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def _dot(a: Sequence[float], b: Sequence[float]) -> float:
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def _norm(v: Sequence[float]) -> Vec3:
    n = math.sqrt(_dot(v, v))
    if n < 1e-9:
        return (0.0, 1.0, 0.0)
    return (v[0] / n, v[1] / n, v[2] / n)


class MeshBuilder:
    """Accumulates primitives into named material groups."""

    def __init__(self, name: str = "builder") -> None:
        self.name = name
        self.groups: Dict[str, _Group] = {}

    # ------------------------------------------------------------------ internals
    def group(self, material: Material) -> _Group:
        grp = self.groups.get(material.name)
        if grp is None:
            grp = _Group(material)
            self.groups[material.name] = grp
        return grp

    def _quad(self, material: Material, verts: Sequence[Vec3], uv: Sequence[Vec2],
              ref: Optional[Sequence[float]] = None,
              normal: Optional[Vec3] = None) -> None:
        """Add a quad.  With ``ref`` the winding is auto-corrected so that the normal
        points away from the shape centre (guarantees correct back-face culling)."""
        a, b, c = verts[0], verts[1], verts[2]
        n = _cross(_sub(b, a), _sub(c, a))
        if normal is not None:
            n = normal
        else:
            n = _norm(n)
            if ref is not None:
                centre = ((a[0] + b[0] + c[0] + verts[3][0]) * 0.25,
                          (a[1] + b[1] + c[1] + verts[3][1]) * 0.25,
                          (a[2] + b[2] + c[2] + verts[3][2]) * 0.25)
                if _dot(n, _sub(centre, ref)) < 0.0:
                    verts = list(reversed(verts))
                    uv = list(reversed(uv))
                    n = (-n[0], -n[1], -n[2])
        grp = self.group(material)
        i0 = grp.vertex(verts[0], n, uv[0])
        i1 = grp.vertex(verts[1], n, uv[1])
        i2 = grp.vertex(verts[2], n, uv[2])
        i3 = grp.vertex(verts[3], n, uv[3])
        grp.indices.extend([i0, i1, i2, i0, i2, i3])

    def add_quad_auto(self, material: Material, verts: Sequence[Vec3],
                      uv: Optional[Sequence[Vec2]] = None,
                      ref: Optional[Sequence[float]] = None,
                      normal: Optional[Vec3] = None) -> None:
        """Public quad builder with automatic outward winding (see :meth:`_quad`)."""
        use_uv = uv if uv is not None else [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]
        self._quad(material, verts, use_uv, ref, normal)

    def add_tri_auto(self, material: Material, verts: Sequence[Vec3],
                     uv: Optional[Sequence[Vec2]] = None,
                     ref: Optional[Sequence[float]] = None,
                     normal: Optional[Vec3] = None) -> None:
        use_uv = uv if uv is not None else [(0.0, 0.0), (1.0, 0.0), (0.5, 1.0)]
        self._tri(material, verts, use_uv, normal, ref)

    def _tri(self, material: Material, verts: Sequence[Vec3], uv: Sequence[Vec2],
             normal: Optional[Vec3] = None, ref: Optional[Sequence[float]] = None) -> None:
        a, b, c = verts
        n = normal if normal is not None else _norm(_cross(_sub(b, a), _sub(c, a)))
        if normal is None and ref is not None:
            centre = ((a[0] + b[0] + c[0]) / 3.0, (a[1] + b[1] + c[1]) / 3.0,
                      (a[2] + b[2] + c[2]) / 3.0)
            if _dot(n, _sub(centre, ref)) < 0.0:
                verts = (a, c, b)
                uv = (uv[0], uv[2], uv[1])
                n = (-n[0], -n[1], -n[2])
        grp = self.group(material)
        i0 = grp.vertex(verts[0], n, uv[0])
        i1 = grp.vertex(verts[1], n, uv[1])
        i2 = grp.vertex(verts[2], n, uv[2])
        grp.indices.extend([i0, i1, i2])

    # ------------------------------------------------------------------ primitives
    def add_box(self, material: Material, size: Sequence[float], center: Sequence[float] = (0, 0, 0),
                top_scale: float = 1.0, uv_scale: float = 1.0, mirror_z: bool = False) -> None:
        """Axis-aligned (optionally tapered on top) box."""
        sx, sy, sz = size[0], size[1], size[2]
        cx, cy, cz = center
        x0, x1 = cx - sx / 2, cx + sx / 2
        y0, y1 = cy - sy / 2, cy + sy / 2
        z0, z1 = cz - sz / 2, cz + sz / 2
        tx = sx * 0.5 * top_scale
        tz = sz * 0.5 * top_scale
        ref = (cx, cy, cz)
        u = uv_scale
        # bottom / top
        self._quad(material, [(x0, y0, z0), (x1, y0, z0), (x1, y0, z1), (x0, y0, z1)],
                   [(0, 0), (u, 0), (u, u), (0, u)], ref)
        if abs(top_scale - 1.0) < 1e-6:
            self._quad(material, [(x0, y1, z0), (x1, y1, z0), (x1, y1, z1), (x0, y1, z1)],
                       [(0, 0), (u, 0), (u, u), (0, u)], ref)
        else:
            self._quad(material,
                       [(cx - tx, y1, cz - tz), (cx + tx, y1, cz - tz),
                        (cx + tx, y1, cz + tz), (cx - tx, y1, cz + tz)],
                       [(0, 0), (u, 0), (u, u), (0, u)], ref)
        # sides
        self._quad(material, [(x1, y0, z0), (x1, y0, z1), (x1, y1, z1), (x1, y1, z0)],
                   [(0, 0), (u, 0), (u, u), (0, u)], ref)
        self._quad(material, [(x0, y0, z1), (x0, y0, z0), (x0, y1, z0), (x0, y1, z1)],
                   [(0, 0), (u, 0), (u, u), (0, u)], ref)
        self._quad(material, [(x0, y0, z1), (x1, y0, z1), (x1, y1, z1), (x0, y1, z1)],
                   [(0, 0), (u, 0), (u, u), (0, u)], ref)
        self._quad(material, [(x1, y0, z0), (x0, y0, z0), (x0, y1, z0), (x1, y1, z0)],
                   [(0, 0), (u, 0), (u, u), (0, u)], ref)
        if mirror_z:
            self.add_box(material, (sx, sy, sz), (cx, cy, -cz), top_scale, uv_scale, False)

    def add_taper_box(self, material: Material, bottom: Sequence[float], top: Sequence[float],
                      height: float, base_center: Sequence[float] = (0, 0, 0)) -> None:
        """Box whose top face is a different rectangle than its bottom face.
        ``bottom``/``top`` are ``(size_x, size_z)``.  Used for car cabins and roofs."""
        bx, bz = bottom
        tx, tz = top
        cx, y0, cz = base_center
        y1 = y0 + height
        b = [(cx - bx / 2, y0, cz - bz / 2), (cx + bx / 2, y0, cz - bz / 2),
             (cx + bx / 2, y0, cz + bz / 2), (cx - bx / 2, y0, cz + bz / 2)]
        t = [(cx - tx / 2, y1, cz - tz / 2), (cx + tx / 2, y1, cz - tz / 2),
             (cx + tx / 2, y1, cz + tz / 2), (cx - tx / 2, y1, cz + tz / 2)]
        ref = (cx, y0 + height * 0.5, cz)
        for i in range(4):
            j = (i + 1) % 4
            self._quad(material, [b[i], b[j], t[j], t[i]], [(0, 0), (1, 0), (1, 1), (0, 1)], ref)
        self._quad(material, t, [(0, 0), (1, 0), (1, 1), (0, 1)], ref)
        self._quad(material, b, [(0, 0), (1, 0), (1, 1), (0, 1)], ref)

    def add_slab(self, material: Material, size: Sequence[float], center: Sequence[float],
                 thickness: float = 0.06, uv_scale: float = 1.0) -> None:
        self.add_box(material, (size[0], thickness, size[1]), center, 1.0, uv_scale)

    def add_cylinder(self, material: Material, radius: float, half_length: float,
                     center: Sequence[float] = (0, 0, 0), segments: int = 16,
                     axis: str = "z", caps: bool = True, cap_material: Optional[Material] = None,
                     radius_bottom: Optional[float] = None) -> None:
        """Cylinder / cone aligned to a world axis.  Smooth-shaded around the rim."""
        segments = max(3, int(segments))
        cx, cy, cz = center
        r0 = radius if radius_bottom is None else radius_bottom
        ring_low, ring_high = [], []
        uvs = []
        for i in range(segments + 1):
            a = 2.0 * math.pi * i / segments
            ca, sa = math.cos(a), math.sin(a)
            if axis == "z":
                ring_low.append((cx + r0 * ca, cy + r0 * sa, cz - half_length))
                ring_high.append((cx + radius * ca, cy + radius * sa, cz + half_length))
            elif axis == "y":
                ring_low.append((cx + r0 * ca, cy - half_length, cz + r0 * sa))
                ring_high.append((cx + radius * ca, cy + half_length, cz + radius * sa))
            else:  # x
                ring_low.append((cx - half_length, cy + r0 * ca, cz + r0 * sa))
                ring_high.append((cx + half_length, cy + radius * ca, cz + radius * sa))
            uvs.append((i / segments, 0.0))
        grp = self.group(material)
        base = []
        for i in range(segments + 1):
            nx, ny, nz = self._axis_normal(ring_low[i], center, axis)
            base.append(grp.vertex(ring_low[i], (nx, ny, nz), (uvs[i][0], 0.0)))
            nx, ny, nz = self._axis_normal(ring_high[i], center, axis)
            base.append(grp.vertex(ring_high[i], (nx, ny, nz), (uvs[i][0], 1.0)))
        for i in range(segments):
            i0 = base[i * 2]
            i1 = base[i * 2 + 2]
            i2 = base[i * 2 + 3]
            i3 = base[i * 2 + 1]
            grp.indices.extend([i0, i1, i2, i0, i2, i3])
        if caps:
            cap_mat = cap_material or material
            self._cap(cap_mat, ring_low, center, axis, half_length, r0, False)
            if radius <= 1e-6:
                return
            self._cap(cap_mat, ring_high, center, axis, half_length, radius, True)

    def _axis_normal(self, point: Sequence[float], center: Sequence[float], axis: str) -> Vec3:
        dx = point[0] - center[0]
        dy = point[1] - center[1]
        dz = point[2] - center[2]
        if axis == "z":
            return _norm((dx, dy, 0.0))
        if axis == "y":
            return _norm((dx, 0.0, dz))
        return _norm((0.0, dy, dz))

    def _cap(self, material: Material, ring: Sequence[Vec3], center: Sequence[float],
             axis: str, half_length: float, radius: float, positive: bool) -> None:
        grp = self.group(material)
        n = {"z": (0.0, 0.0, 1.0), "y": (0.0, 1.0, 0.0), "x": (1.0, 0.0, 0.0)}[axis]
        if not positive:
            n = (-n[0], -n[1], -n[2])
        local_center = [sum(p[i] for p in ring) / max(len(ring), 1) for i in range(3)]
        idx_center = grp.vertex(tuple(local_center), n, (0.5, 0.5))
        ring_idx = [grp.vertex(p, n, (0.5 + (p[0] - center[0]) / (2 * radius + 1e-6),
                                      0.5 + (p[1] - center[1]) / (2 * radius + 1e-6)))
                    for p in ring]
        count = len(ring) - 1
        for i in range(count):
            a, b = ring_idx[i], ring_idx[i + 1]
            if positive:
                grp.indices.extend([idx_center, a, b])
            else:
                grp.indices.extend([idx_center, b, a])

    def add_sphere(self, material: Material, radius: float, center: Sequence[float] = (0, 0, 0),
                   segments: int = 10, rings: int = 6, squash: float = 1.0) -> None:
        grp = self.group(material)
        cx, cy, cz = center
        grid: List[List[int]] = []
        for r in range(rings + 1):
            phi = math.pi * r / rings
            y = math.cos(phi)
            rad = math.sin(phi)
            row = []
            for s in range(segments + 1):
                theta = 2.0 * math.pi * s / segments
                nx, ny, nz = rad * math.cos(theta), y, rad * math.sin(theta)
                pos = (cx + radius * nx, cy + radius * ny * squash, cz + radius * nz)
                row.append(grp.vertex(pos, (nx, ny, nz), (s / segments, r / rings)))
            grid.append(row)
        for r in range(rings):
            for s in range(segments):
                a, b = grid[r][s], grid[r][s + 1]
                c, d = grid[r + 1][s + 1], grid[r + 1][s]
                grp.indices.extend([a, b, c, a, c, d])

    def add_profile(self, material: Material, profile: Sequence[Vec2],
                    z0: float, z1: float,
                    normal_override: Optional[Vec3] = None) -> None:
        """Extrude a closed 2D polygon (listed counter-clockwise in XY) along Z."""
        pts = list(profile)
        ref = (sum(p[0] for p in pts) / len(pts), sum(p[1] for p in pts) / len(pts), (z0 + z1) * 0.5)
        n = len(pts)
        for i in range(n):
            p0, p1 = pts[i], pts[(i + 1) % n]
            self._quad(material,
                       [(p0[0], p0[1], z0), (p1[0], p1[1], z0), (p1[0], p1[1], z1), (p0[0], p0[1], z1)],
                       [(0, 0), (1, 0), (1, 1), (0, 1)],
                       ref if normal_override is None else None,
                       normal_override)
        # caps (fan from the centroid - valid for convex / star shaped profiles)
        for z, flip in ((z1, False), (z0, True)):
            grp = self.group(material)
            nv = (0.0, 0.0, 1.0 if not flip else -1.0)
            cidx = grp.vertex((ref[0], ref[1], z), nv, (0.5, 0.5))
            ring = [grp.vertex((p[0], p[1], z), nv,
                               (0.5 + (p[0] - ref[0]) * 0.1, 0.5 + (p[1] - ref[1]) * 0.1)) for p in pts]
            for i in range(len(pts)):
                a, b = ring[i], ring[(i + 1) % len(pts)]
                grp.indices.extend([cidx, a, b] if not flip else [cidx, b, a])

    def add_flat(self, material: Material, size: Sequence[float], center: Sequence[float],
                 color_uv_scale: float = 1.0) -> None:
        """Horizontal quad facing +Y, centred on ``center`` at its own y coordinate."""
        sx, sz = size
        cx, cy, cz = center
        hx, hz = sx / 2, sz / 2
        self._quad(material,
                   [(cx - hx, cy, cz - hz), (cx - hx, cy, cz + hz),
                    (cx + hx, cy, cz + hz), (cx + hx, cy, cz - hz)],
                   [(0, 0), (0, color_uv_scale), (color_uv_scale, color_uv_scale), (color_uv_scale, 0)],
                   None, (0.0, 1.0, 0.0))

    def add_wheel(self, tyre: Material, hub: Material, radius: float, width: float,
                  center: Sequence[float], segments: int = 14, spin_uv: float = 1.0,
                  mirror: bool = False) -> None:
        """Wheel aligned to the local Z axis (car width axis) with a visible hub cap."""
        self.add_cylinder(tyre, radius, width * 0.5, center, segments, axis="z",
                          caps=True, cap_material=hub)
        cx, cy, cz = center
        for side in (-1, 1):
            self.add_cylinder(hub, radius * 0.55, 0.012, (cx, cy, cz + side * (width * 0.5 + 0.005)),
                              max(6, segments // 2), axis="z", caps=True)
        if mirror:
            self.add_wheel(tyre, hub, radius, width, (cx, cy, -cz), segments, spin_uv, False)

    # ------------------------------------------------------------------ output
    def build(self, name: Optional[str] = None) -> MeshData:
        surfaces = [g.to_surface() for g in self.groups.values() if g.indices]
        return MeshData(surfaces, name or self.name)


# --------------------------------------------------------------------------------------
# GPU side
# --------------------------------------------------------------------------------------
class VaoMesh:
    """Uploads a :class:`MeshData` into OpenGL vertex/index buffers."""

    ATTRIB_POS = 0
    ATTRIB_NORMAL = 1
    ATTRIB_UV = 2

    def __init__(self, mesh: MeshData) -> None:
        self.mesh = mesh
        self.name = mesh.name
        self._vaos: List[int] = []
        self._vbos: List[int] = []
        self._ebos: List[int] = []
        self._counts: List[int] = []
        self._materials: List[Material] = []
        self._upload()

    def _upload(self) -> None:
        for surface in self.mesh.surfaces:
            vao = GL.glGenVertexArrays(1)
            vbo = GL.glGenBuffers(1)
            ebo = GL.glGenBuffers(1)
            GL.glBindVertexArray(vao)
            data = surface.interleaved()
            GL.glBindBuffer(GL.GL_ARRAY_BUFFER, vbo)
            GL.glBufferData(GL.GL_ARRAY_BUFFER, data.nbytes, data, GL.GL_STATIC_DRAW)
            stride = 8 * 4
            GL.glEnableVertexAttribArray(self.ATTRIB_POS)
            GL.glVertexAttribPointer(self.ATTRIB_POS, 3, GL.GL_FLOAT, GL.GL_FALSE, stride, ctypes_void(0))
            GL.glEnableVertexAttribArray(self.ATTRIB_NORMAL)
            GL.glVertexAttribPointer(self.ATTRIB_NORMAL, 3, GL.GL_FLOAT, GL.GL_FALSE, stride, ctypes_void(12))
            GL.glEnableVertexAttribArray(self.ATTRIB_UV)
            GL.glVertexAttribPointer(self.ATTRIB_UV, 2, GL.GL_FLOAT, GL.GL_FALSE, stride, ctypes_void(24))
            GL.glBindBuffer(GL.GL_ELEMENT_ARRAY_BUFFER, ebo)
            indices = surface.indices.astype(np.uint32)
            GL.glBufferData(GL.GL_ELEMENT_ARRAY_BUFFER, indices.nbytes, indices, GL.GL_STATIC_DRAW)
            GL.glBindVertexArray(0)
            self._vaos.append(vao)
            self._vbos.append(vbo)
            self._ebos.append(ebo)
            self._counts.append(len(indices))
            self._materials.append(surface.material)

    @property
    def triangle_count(self) -> int:
        return sum(c // 3 for c in self._counts)

    def parts(self) -> Iterable[Tuple[int, int, Material]]:
        """Yield ``(vao, index_count, material)`` for each material group."""
        return zip(self._vaos, self._counts, self._materials)

    def delete(self) -> None:
        for vao in self._vaos:
            GL.glDeleteVertexArrays(1, [vao])
        for buf in self._vbos + self._ebos:
            GL.glDeleteBuffers(1, [buf])
        self._vaos.clear()
        self._vbos.clear()
        self._ebos.clear()
        self._counts.clear()
        self._materials.clear()


def ctypes_void(offset: int):
    """Small shim so we do not need to import ctypes at module scope."""
    import ctypes
    return ctypes.c_void_p(offset)
