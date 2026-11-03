"""OpenGL renderer.

Pipeline in one line: **PyOpenGL 3.3 core** with hand written GLSL, CPU frustum culling,
one shared dynamic vertex format, a single instanced draw call for every glowing sprite
(signal lenses, street lamps, headlights, rain streaks) and the pygame-drawn HUD uploaded
as a full screen textured quad.

Design notes
------------
* Geometry lives in :class:`~.mesh.VaoMesh` (one VAO/VBO per mesh, uploaded on first use)
  and is never rebuilt per frame.
* :class:`Material` objects carry the look of a surface (diffuse, specular, emissive,
  optional texture); the renderer only sets uniforms, it never touches geometry.
* Vehicles are drawn hierarchically: body first, then each wheel with its own
  ``vehicle * translate * rotate`` matrix, which is what makes wheel spin visible.
* Everything that can be batched is batched: all glow/rain billboards go out in a single
  instanced call and every mesh keeps one VAO per material group.  A typical driving view
  costs roughly 300-900 draw calls (37 chunks x their material groups plus vehicles) and
  about 120k triangles, which an integrated GPU handles comfortably; chunk distance and
  frustum culling roughly halve both numbers.  Measured numbers from the sandbox (software
  rasteriser) are printed by ``tools/benchmark.py`` - they are not laptop numbers.
"""

from __future__ import annotations

import math
import time
from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import pygame
from OpenGL import GL

from . import settings as S
from .mesh import Material, MeshData, VaoMesh
from .shaders import build_programs
from .textures import build_textures
from .transform import (Frustum, matmul, model_matrix, normal_matrix, rotation_y,
                        rotation_z, translation)
from .vehicle_models import VehicleModel, resolve_material

Vec3 = Tuple[float, float, float]

MAX_SPRITES = 4096
MAX_RAIN = 2000


def _as_float32(matrix) -> np.ndarray:
    return np.asarray(matrix, dtype=np.float32).reshape(4, 4)


class Renderer:
    """Owns the window, the GL programs, the caches and the per-frame state."""

    def __init__(self, settings: S.Settings, quality: Dict[str, object],
                 splash: bool = False) -> None:
        self.settings = settings
        self.quality = dict(quality)
        self.width = int(settings.width)
        self.height = int(settings.height)
        self.window = pygame.display.get_surface()
        if self.window is None:
            self._create_window(splash)
        pygame.display.set_caption(S.APP_TITLE)
        self._mesh_cache: Dict[int, VaoMesh] = {}
        self.textures: Dict[str, int] = {}
        self.stats: Dict[str, float] = {
            "draw_calls": 0, "triangles": 0, "meshes": 0, "sprites": 0,
            "chunks_drawn": 0, "chunks_culled": 0, "frame": 0, "cpu_ms": 0.0,
        }
        self.programs = build_programs(core_profile=True)
        self.uses_core = True
        self._wireframe = False
        self._setup_state()
        self._create_sky()
        self._create_sprite_batch()
        self._create_shadow_meshes()
        self._create_ui()
        self._create_ui_quad()
        self.load_textures()
        self.gl_info = self._query_gl_info()
        self._shadow_map_size = 1024 if self.quality.get("shadows") else 0
        self._shadow_texture = None
        self._shadow_fbo = None
        self._clear_color = (0.62, 0.72, 0.85)
        self._detail_range = float(self.quality.get("far_plane", 420.0)) * 0.92
        self.frame_time = 0.0

    # ------------------------------------------------------------------ window/GL
    def _create_window(self, splash: bool) -> None:
        pygame.display.gl_set_attribute(pygame.GL_CONTEXT_MAJOR_VERSION, 3)
        pygame.display.gl_set_attribute(pygame.GL_CONTEXT_MINOR_VERSION, 3)
        pygame.display.gl_set_attribute(pygame.GL_CONTEXT_PROFILE_MASK,
                                        pygame.GL_CONTEXT_PROFILE_CORE)
        pygame.display.gl_set_attribute(pygame.GL_DEPTH_SIZE, 24)
        if self.quality.get("multisample"):
            pygame.display.gl_set_attribute(pygame.GL_MULTISAMPLEBUFFERS, 1)
            pygame.display.gl_set_attribute(pygame.GL_MULTISAMPLESAMPLES,
                                            int(self.quality.get("multisample", 2)))
        flags = pygame.OPENGL | pygame.DOUBLEBUF
        if self.settings.fullscreen:
            flags |= pygame.FULLSCREEN
        try:
            self.window = pygame.display.set_mode((self.width, self.height), flags)
        except pygame.error:
            pygame.display.gl_set_attribute(pygame.GL_CONTEXT_PROFILE_MASK, 0)
            self.window = pygame.display.set_mode((self.width, self.height), flags)
        pygame.event.pump()
        if splash:
            self.clear_only()

    def _query_gl_info(self) -> Dict[str, object]:
        try:
            version = GL.glGetString(GL.GL_VERSION)
            renderer = GL.glGetString(GL.GL_RENDERER)
            vendor = GL.glGetString(GL.GL_VENDOR)
            glsl = GL.glGetString(GL.GL_SHADING_LANGUAGE_VERSION)
        except Exception:                               # pragma: no cover
            return {"version": "unknown", "renderer": "unknown", "vendor": "unknown",
                    "glsl": "unknown"}
        decode = lambda text: text.decode("utf-8", "replace") if isinstance(text, bytes) else str(text)
        return {"version": decode(version), "renderer": decode(renderer),
                "vendor": decode(vendor), "glsl": decode(glsl)}

    def _setup_state(self) -> None:
        GL.glEnable(GL.GL_DEPTH_TEST)
        GL.glDepthFunc(GL.GL_LEQUAL)
        GL.glEnable(GL.GL_CULL_FACE)
        GL.glCullFace(GL.GL_BACK)
        GL.glFrontFace(GL.GL_CCW)
        GL.glCullFace(GL.GL_BACK)
        GL.glEnable(GL.GL_BLEND)
        GL.glBlendFunc(GL.GL_SRC_ALPHA, GL.GL_ONE_MINUS_SRC_ALPHA)
        GL.glDisable(GL.GL_LINE_SMOOTH)

    # ------------------------------------------------------------------ resources
    def _create_sky(self) -> None:
        """Full screen quad for the procedural sky shader (see ``shaders.SKY_*``)."""
        quad = np.array([-1.0, -1.0,
                         -1.0, 1.0,
                         1.0, -1.0,
                         1.0, 1.0], dtype=np.float32)
        vao = GL.glGenVertexArrays(1)
        GL.glBindVertexArray(vao)
        vbo = GL.glGenBuffers(1)
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, vbo)
        GL.glBufferData(GL.GL_ARRAY_BUFFER, quad.nbytes, quad, GL.GL_STATIC_DRAW)
        GL.glEnableVertexAttribArray(0)
        GL.glVertexAttribPointer(0, 2, GL.GL_FLOAT, False, 8, None)
        GL.glBindVertexArray(0)
        self._sky_vao = vao
        self._sky_count = 4
        self._sky_buffers = (vbo,)

    def _create_sprite_batch(self) -> None:
        """Instanced billboards.

        Vertex layout (matches ``shaders.SPRITE_VERT``)::

            location 0  vec2  a_quad    -0.5..0.5 quad corner
            location 3  vec3  i_center  world position      (per instance)
            location 4  vec2  i_size    world size          (per instance)
            location 5  vec4  i_color   rgb + alpha         (per instance)
        """
        self._sprite_data = np.zeros((MAX_SPRITES, 9), dtype=np.float32)
        self._sprite_count = 0
        vao = GL.glGenVertexArrays(1)
        GL.glBindVertexArray(vao)
        quad = np.array([[-0.5, -0.5], [0.5, -0.5], [0.5, 0.5], [-0.5, 0.5]],
                        dtype=np.float32)
        vbo = GL.glGenBuffers(1)
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, vbo)
        GL.glBufferData(GL.GL_ARRAY_BUFFER, quad.nbytes, quad, GL.GL_STATIC_DRAW)
        GL.glEnableVertexAttribArray(0)
        GL.glVertexAttribPointer(0, 2, GL.GL_FLOAT, False, 8, None)
        instance_vbo = GL.glGenBuffers(1)
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, instance_vbo)
        GL.glBufferData(GL.GL_ARRAY_BUFFER, self._sprite_data.nbytes, None, GL.GL_STREAM_DRAW)
        stride = 9 * 4
        for location, components, offset in ((3, 3, 0), (4, 2, 12), (5, 4, 20)):
            GL.glEnableVertexAttribArray(location)
            GL.glVertexAttribPointer(location, components, GL.GL_FLOAT, False, stride,
                                     _offset(offset))
            GL.glVertexAttribDivisor(location, 1)
        GL.glBindVertexArray(0)
        self._sprite_vao = vao
        self._sprite_instance_vbo = instance_vbo

    def _create_shadow_meshes(self) -> None:
        self._blob_dark = MeshData([], "blob_dark")
        self._blob_soft = MeshData([], "blob_soft")

    def _create_ui(self) -> None:
        self._ui_texture = GL.glGenTextures(1)
        GL.glBindTexture(GL.GL_TEXTURE_2D, self._ui_texture)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MIN_FILTER, GL.GL_LINEAR)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MAG_FILTER, GL.GL_LINEAR)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_S, GL.GL_CLAMP_TO_EDGE)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_T, GL.GL_CLAMP_TO_EDGE)
        GL.glTexImage2D(GL.GL_TEXTURE_2D, 0, GL.GL_RGBA8, self.width, self.height, 0,
                        GL.GL_RGBA, GL.GL_UNSIGNED_BYTE, None)
        self._ui_size = (self.width, self.height)

    def load_textures(self) -> None:
        """Upload every procedural texture (asphalt, kerb, zebra, sign atlas)."""
        for name, surface in build_textures(self.quality).items():
            self.upload_texture(name, surface)

    def upload_texture(self, name: str, surface: pygame.Surface) -> int:
        if surface is None:
            return 0
        texture = self.textures.get(name) or GL.glGenTextures(1)
        self.textures[name] = texture
        GL.glBindTexture(GL.GL_TEXTURE_2D, texture)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MIN_FILTER, GL.GL_LINEAR_MIPMAP_LINEAR)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MAG_FILTER, GL.GL_LINEAR)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_S, GL.GL_REPEAT)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_T, GL.GL_REPEAT)
        GL.glPixelStorei(GL.GL_UNPACK_ALIGNMENT, 1)
        # flipped=False keeps the pygame row order, so UV v = 0 is the *top* row - the same
        # convention the sign atlas packs with (see signage.SignAtlas.add).
        data = pygame.image.tostring(surface, "RGBA", False)
        GL.glTexImage2D(GL.GL_TEXTURE_2D, 0, GL.GL_RGBA8, surface.get_width(),
                        surface.get_height(), 0, GL.GL_RGBA, GL.GL_UNSIGNED_BYTE, data)
        GL.glGenerateMipmap(GL.GL_TEXTURE_2D)
        return texture

    def mesh_vao(self, mesh: MeshData) -> VaoMesh:
        key = id(mesh)
        vao = self._mesh_cache.get(key)
        if vao is None:
            vao = VaoMesh(mesh)
            self._mesh_cache[key] = vao
        return vao

    def release_mesh(self, mesh: MeshData) -> None:
        vao = self._mesh_cache.pop(id(mesh), None)
        if vao is not None:
            vao.delete()

    # ------------------------------------------------------------------ frame
    def begin_frame(self, camera, lighting: Dict[str, object],
                    ground_visible: bool = True) -> None:
        """Clear the frame and upload everything that is constant for the whole frame."""
        fog = lighting.get("fog_color", (0.6, 0.7, 0.85))
        GL.glClearColor(float(fog[0]), float(fog[1]), float(fog[2]), 1.0)
        GL.glClear(GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT)
        self.stats["draw_calls"] = 0
        self.stats["triangles"] = 0
        self.stats["meshes"] = 0
        self.stats["chunks_drawn"] = 0
        self.stats["chunks_culled"] = 0
        self.stats["sprites"] = 0
        self._sprite_count = 0
        self._view = np.asarray(camera.view, dtype=np.float32)
        self._view_projection = np.asarray(camera.view_projection, dtype=np.float32)
        self._camera_position = tuple(float(v) for v in np.asarray(camera.position))
        self._lighting = lighting
        self._ground_visible = ground_visible
        sun = lighting.get("sun_dir", (0.4, 0.8, 0.3))
        length = max(float(np.linalg.norm(np.asarray(sun, dtype=np.float32))), 1e-5)
        self._sun = tuple(float(v) / length for v in sun)
        # ---- per-frame world lighting uniforms --------------------------------------
        program = self.programs["world"]
        program.use()
        program.set_mat4("u_view_projection", self._view_projection)
        program.set_vec3("u_camera_pos", *self._camera_position)
        program.set_vec3("u_light_dir", *self._sun)
        program.set_color("u_light_color", lighting.get("light", (1.0, 0.97, 0.9)))
        program.set_color("u_ambient_sky", lighting.get("ambient_sky", (0.4, 0.45, 0.55)))
        program.set_color("u_ambient_ground",
                          lighting.get("ambient_ground", (0.2, 0.2, 0.18)))
        program.set_color("u_fog_color", fog)
        program.set_float("u_fog_density", float(lighting.get("fog_density", 0.006)))
        program.set_float("u_shadow_strength", float(lighting.get("shadow", 0.45)))
        program.set_float("u_flash", float(getattr(camera, "flash", 0.0)))
        program.set_int("u_sign_texture", 0)
        self.draw_sky(camera, lighting)

    @property
    def view_projection(self) -> np.ndarray:
        return self._view_projection

    def draw_sky(self, camera, lighting: Dict[str, object]) -> None:
        """Procedural sky + sun + stars (a single full screen triangle strip)."""
        program = self.programs["sky"]
        program.use()
        try:
            inverse = np.asarray(camera.inverse_view_projection(), dtype=np.float32)
        except Exception:                                    # pragma: no cover - defensive
            inverse = np.identity(4, dtype=np.float32)
        program.set_mat4("u_inv_view_proj", inverse)
        program.set_vec3("u_camera_pos", *self._camera_position)
        program.set_color("u_zenith", lighting.get("zenith", (0.2, 0.36, 0.72)))
        program.set_color("u_horizon", lighting.get("horizon", (0.72, 0.82, 0.94)))
        program.set_color("u_ground", lighting.get("ground", (0.28, 0.26, 0.22)))
        program.set_vec3("u_sun_dir", *self._sun)
        program.set_color("u_sun_color", lighting.get("sun_color", (1.0, 0.96, 0.86)))
        program.set_float("u_star_intensity", float(lighting.get("star_intensity", 0.0)))
        program.set_float("u_sun_size", float(lighting.get("sun_size", 0.9965)))
        GL.glDepthMask(GL.GL_FALSE)
        GL.glDisable(GL.GL_DEPTH_TEST)
        GL.glBindVertexArray(self._sky_vao)
        GL.glDrawArrays(GL.GL_TRIANGLE_STRIP, 0, 4)
        GL.glBindVertexArray(0)
        GL.glEnable(GL.GL_DEPTH_TEST)
        GL.glDepthMask(GL.GL_TRUE)
        self.stats["draw_calls"] += 1

    # ------------------------------------------------------------------ meshes
    def _material_uniforms(self, program, material: Optional[Material],
                           alpha: float, force_no_texture: bool) -> None:
        if material is None:
            program.set_color("u_color", (0.8, 0.8, 0.8))
            program.set_color("u_emissive", (0.0, 0.0, 0.0))
            program.set_float("u_specular", 0.2)
            program.set_int("u_use_texture", 0)
        else:
            program.set_color("u_color", material.color)
            program.set_float("u_specular", material.specular)
            strength = float(material.emissive)
            ec = material.emissive_color
            program.set_color("u_emissive", (ec[0] * strength, ec[1] * strength,
                                             ec[2] * strength))
            texture = self.textures.get(material.texture) if material.texture else None
            if texture and not force_no_texture:
                GL.glActiveTexture(GL.GL_TEXTURE0)
                GL.glBindTexture(GL.GL_TEXTURE_2D, texture)
                program.set_int("u_sign_texture", 0)
                program.set_int("u_use_texture", 1)
            else:
                program.set_int("u_use_texture", 0)
        program.set_float("u_alpha", alpha)

    def draw_mesh(self, mesh: MeshData, model=None, material_resolver=None,
                  alpha: float = 1.0, force_no_texture: bool = False,
                  cull: bool = True) -> None:
        """Draw a :class:`MeshData`; ``material_resolver(name)`` may override materials."""
        if mesh is None:
            return
        vao = self.mesh_vao(mesh)
        program = self.programs["world"]
        program.use()
        model_gl = _as_float32(model if model is not None else np.identity(4))
        program.set_mat4("u_model", model_gl)
        program.set_mat3("u_normal_mat", normal_matrix(model_gl))
        program.set_mat4("u_mvp", matmul(self._view_projection, model_gl))
        if not cull:
            GL.glDisable(GL.GL_CULL_FACE)
        for part_vao, count, material in vao.parts():
            resolved = material
            if material_resolver is not None:
                resolved = material_resolver(material)
            self._material_uniforms(program, resolved, alpha, force_no_texture)
            GL.glBindVertexArray(part_vao)
            GL.glDrawElements(GL.GL_TRIANGLES, count, GL.GL_UNSIGNED_INT, None)
            self.stats["draw_calls"] += 1
            self.stats["triangles"] += count // 3
        GL.glBindVertexArray(0)
        if not cull:
            GL.glEnable(GL.GL_CULL_FACE)
        self.stats["meshes"] += 1

    def draw_chunks(self, chunks: Sequence[object], frustum: Frustum,
                    material_resolver=None) -> None:
        """Draw visible city chunks (frustum + distance culled)."""
        for chunk in chunks:
            mesh = getattr(chunk, "mesh", None)
            if mesh is None:
                continue
            centre = getattr(chunk, "center", (0.0, 0.0, 0.0))
            half = getattr(chunk, "half_extents", (22.0, 22.0, 22.0))
            radius = max(float(half[0]), float(half[2])) * 1.45 + 6.0
            if not frustum.sphere_visible((float(centre[0]), float(centre[1]),
                                           float(centre[2])), radius):
                self.stats["chunks_culled"] += 1
                continue
            # distance cull: fog hides anything beyond the far plane, so drawing it would
            # only cost fill rate and draw calls (this is the cheap half of LOD)
            distance = math.hypot(float(centre[0]) - self._camera_position[0],
                                  float(centre[2]) - self._camera_position[2])
            if distance - radius > self._detail_range:
                self.stats["chunks_culled"] += 1
                continue
            self.stats["chunks_drawn"] += 1
            self.draw_mesh(mesh, None, material_resolver)

    # ------------------------------------------------------------------ vehicles
    def draw_vehicle(self, model: VehicleModel, vehicle, paint: Vec3 = (0.8, 0.3, 0.2),
                     albedo: Optional[Vec3] = None,
                     resolver_kwargs: Optional[Dict[str, object]] = None) -> None:
        """Draw a hierarchical vehicle: body, then each wheel with its own transform."""
        kwargs = dict(resolver_kwargs or {})
        if albedo is not None:
            kwargs["albedo"] = albedo
        body_y = (float(getattr(vehicle, "y", 0.0))
                  + float(getattr(vehicle, "bump_offset", 0.0)))
        body_matrix = model_matrix((vehicle.x, body_y, vehicle.z),
                                   vehicle.yaw, 1.0,
                                   pitch=float(getattr(vehicle, "body_pitch", 0.0)),
                                   roll=float(getattr(vehicle, "body_roll", 0.0)))
        kwargs.setdefault("paint", paint)

        def resolve(material: Material) -> Material:
            return resolve_material(material.name, **kwargs)

        self.draw_mesh(model.body, body_matrix, resolve)
        wheel_mesh = model.wheel
        for index, (ox, oy, oz) in enumerate(model.wheels):
            wheel = vehicle.wheels[index] if index < len(vehicle.wheels) else None
            spin = float(getattr(wheel, "spin", 0.0))
            steer = float(getattr(wheel, "steer", 0.0))
            local = translation(ox, oy, oz)
            if steer:
                local = local @ _as_float32(rotation_y(steer))
            if spin:
                local = local @ _as_float32(rotation_z(spin))
            matrix = body_matrix @ _as_float32(local)
            self.draw_mesh(wheel_mesh, matrix, resolve)

    def draw_blob_shadow(self, x: float, z: float, half_length: float, half_width: float,
                         yaw: float, strength: float = 1.0) -> None:
        """A cheap soft shadow: two translucent discs under the car."""
        alpha = max(0.0, min(0.55, 0.42 * strength))
        size = max(half_length, half_width) * 1.15
        model = model_matrix((x, 0.035, z), yaw, 1.0)
        quad = self._soft_shadow_mesh()
        scale = np.diag([size, 1.0, size, 1.0]).astype(np.float32)
        self.draw_mesh(quad, model @ scale, None, alpha=alpha, force_no_texture=True,
                       cull=False)

    def _soft_shadow_mesh(self) -> MeshData:
        """A soft radial disc built from two fading rings (cached)."""
        if not getattr(self, "_shadow_built", False):
            from .mesh import MeshBuilder
            builder = MeshBuilder("shadow")
            dark = Material("shadow_core", (0.02, 0.02, 0.03), specular=0.0)
            fade = Material("shadow_fade", (0.05, 0.05, 0.07), specular=0.0)
            centre = (0.0, 0.0, 0.0)
            radius = 0.5
            segments = 18
            builder.add_flat(dark, (1.0, 1.0), centre, 1.0)
            ring_inner = [(radius * 0.5 * math.cos(2 * math.pi * i / segments), 0.0,
                           radius * 0.5 * math.sin(2 * math.pi * i / segments))
                          for i in range(segments)]
            ring_outer = [(radius * math.cos(2 * math.pi * i / segments), 0.0,
                           radius * math.sin(2 * math.pi * i / segments))
                          for i in range(segments)]
            for i in range(segments):
                a = ring_inner[i]
                b = ring_inner[(i + 1) % segments]
                c = ring_outer[(i + 1) % segments]
                d = ring_outer[i]
                builder.add_quad_auto(fade, [a, b, c, d], ref=(0.0, 0.0, 0.0))
            self._blob_dark = builder.build("shadow")
            self._shadow_built = True
        return self._blob_dark

    # ------------------------------------------------------------------ sprites
    def push_sprite(self, position: Vec3, size: Tuple[float, float],
                    colour: Tuple[float, float, float, float]) -> None:
        """Queue one camera facing billboard (glow, lamp, rain streak, particle)."""
        if self._sprite_count >= MAX_SPRITES:
            return
        row = self._sprite_data[self._sprite_count]
        row[0] = position[0]
        row[1] = position[1]
        row[2] = position[2]
        row[3] = max(0.02, float(size[0]))
        row[4] = max(0.02, float(size[1]))
        row[5] = float(colour[0])
        row[6] = float(colour[1])
        row[7] = float(colour[2])
        row[8] = float(colour[3])
        self._sprite_count += 1

    def push_sprite_px(self, position: Vec3, width: float, height: float,
                       colour: Tuple[float, float, float, float]) -> None:
        self.push_sprite(position, (width, height), colour)

    def push_rain(self, camera, intensity: float, dt: float = 0.0) -> None:
        """Rain streaks as instanced billboards that follow the camera."""
        if intensity <= 0.02:
            return
        budget = int(self.quality.get("rain_particles", 500) or 0)
        count = int(min(MAX_RAIN, budget * max(0.0, min(1.5, intensity))))
        if count <= 0:
            return
        if not hasattr(self, "_rain_state"):
            rng = np.random.default_rng(1234)
            self._rain_state = rng.uniform(-1.0, 1.0, size=(MAX_RAIN, 3)).astype(np.float32)
        state = self._rain_state
        del dt
        pos = self._camera_position
        now = time.time()
        for i in range(count):
            sx, sy, sz = float(state[i, 0]), float(state[i, 1]), float(state[i, 2])
            fall = (sy * 7.0 + now * 26.0) % 14.0
            x = pos[0] + sx * 40.0
            z = pos[2] + sz * 40.0
            y = 12.5 - fall
            self.push_sprite((x, y, z), (0.11, 1.4), (0.62, 0.70, 0.82, 0.30))

    def flush_sprites(self) -> None:
        """Draw every queued billboard in one instanced call."""
        if self._sprite_count <= 0:
            return
        program = self.programs["sprite"]
        program.use()
        program.set_mat4("u_view_proj", self._view_projection)
        program.set_vec3("u_camera_right", *self._camera_right())
        program.set_vec3("u_camera_up", *self._camera_up())
        packed = np.ascontiguousarray(self._sprite_data[:self._sprite_count])
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self._sprite_instance_vbo)
        GL.glBufferSubData(GL.GL_ARRAY_BUFFER, 0, packed.nbytes, packed)
        GL.glDisable(GL.GL_CULL_FACE)
        GL.glDepthMask(GL.GL_FALSE)
        GL.glBindVertexArray(self._sprite_vao)
        GL.glDrawArraysInstanced(GL.GL_TRIANGLE_FAN, 0, 4, self._sprite_count)
        GL.glBindVertexArray(0)
        GL.glDepthMask(GL.GL_TRUE)
        GL.glEnable(GL.GL_CULL_FACE)
        self.stats["sprites"] += int(self._sprite_count)
        self.stats["draw_calls"] += 1
        self._sprite_count = 0

    def _camera_right(self) -> Vec3:
        matrix = self._view
        return (float(matrix[0][0]), float(matrix[1][0]), float(matrix[2][0]))

    def _camera_up(self) -> Vec3:
        matrix = self._view
        return (float(matrix[0][1]), float(matrix[1][1]), float(matrix[2][1]))

    # ------------------------------------------------------------------ UI
    def draw_ui(self, surface: pygame.Surface) -> None:
        """Blit the pygame-drawn HUD/menus over the 3D frame as one textured quad."""
        width, height = surface.get_size()
        if (width, height) != self._ui_size:
            self._ui_size = (width, height)
            GL.glBindTexture(GL.GL_TEXTURE_2D, self._ui_texture)
            GL.glTexImage2D(GL.GL_TEXTURE_2D, 0, GL.GL_RGBA8, width, height, 0,
                            GL.GL_RGBA, GL.GL_UNSIGNED_BYTE, None)
        GL.glActiveTexture(GL.GL_TEXTURE0)
        GL.glBindTexture(GL.GL_TEXTURE_2D, self._ui_texture)
        GL.glPixelStorei(GL.GL_UNPACK_ALIGNMENT, 1)
        data = pygame.image.tostring(surface, "RGBA", False)
        GL.glTexSubImage2D(GL.GL_TEXTURE_2D, 0, 0, 0, width, height, GL.GL_RGBA,
                           GL.GL_UNSIGNED_BYTE, data)
        program = self.programs["ui"]
        program.use()
        program.set_int("u_texture", 0)
        program.set_float("u_alpha", 1.0)
        GL.glDisable(GL.GL_DEPTH_TEST)
        GL.glDisable(GL.GL_CULL_FACE)
        GL.glBindVertexArray(self._ui_vao)
        GL.glDrawArrays(GL.GL_TRIANGLE_STRIP, 0, 4)
        GL.glBindVertexArray(0)
        GL.glEnable(GL.GL_CULL_FACE)
        GL.glEnable(GL.GL_DEPTH_TEST)
        self.stats["draw_calls"] += 1

    def _create_ui_quad(self) -> None:
        """A clip-space quad with UVs flipped for a top-left origin surface.

        ``shaders.UI_VERT`` expects positions already in clip space, so the quad below is
        literally the whole screen; the UVs are ordered so that texture row 0 (the top of
        the pygame surface) lands at the top of the screen.
        """
        quad = np.array([-1.0, -1.0, 0.0, 1.0,      # bottom-left  -> uv (0,1)
                         -1.0, 1.0, 0.0, 0.0,       # top-left     -> uv (0,0)
                         1.0, -1.0, 1.0, 1.0,       # bottom-right -> uv (1,1)
                         1.0, 1.0, 1.0, 0.0],       # top-right    -> uv (1,0)
                        dtype=np.float32)  # v grows downwards, matching pygame rows
        vao = GL.glGenVertexArrays(1)
        GL.glBindVertexArray(vao)
        vbo = GL.glGenBuffers(1)
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, vbo)
        GL.glBufferData(GL.GL_ARRAY_BUFFER, quad.nbytes, quad, GL.GL_STATIC_DRAW)
        GL.glEnableVertexAttribArray(0)
        GL.glVertexAttribPointer(0, 2, GL.GL_FLOAT, False, 16, None)
        GL.glEnableVertexAttribArray(2)
        GL.glVertexAttribPointer(2, 2, GL.GL_FLOAT, False, 16, _offset(8))
        GL.glBindVertexArray(0)
        self._ui_vao = vao
        self._ui_vbo = vbo

    # ------------------------------------------------------------------ misc
    def clear_only(self, color=(0.05, 0.06, 0.08)) -> None:
        GL.glClearColor(color[0], color[1], color[2], 1.0)
        GL.glClear(GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT)
        pygame.display.flip()

    def present(self, dt: float = 0.0) -> None:
        self.frame_time = dt
        self.stats["frame"] += 1
        pygame.display.flip()

    def resize(self, width: int, height: int) -> None:
        self.width, self.height = int(width), int(height)
        GL.glViewport(0, 0, self.width, self.height)
        self._ui_size = (self.width, self.height)
        GL.glBindTexture(GL.GL_TEXTURE_2D, self._ui_texture)
        GL.glTexImage2D(GL.GL_TEXTURE_2D, 0, GL.GL_RGBA8, self.width, self.height, 0,
                        GL.GL_RGBA, GL.GL_UNSIGNED_BYTE, None)

    def toggle_wireframe(self) -> bool:
        self._wireframe = not self._wireframe
        GL.glPolygonMode(GL.GL_FRONT_AND_BACK,
                         GL.GL_LINE if self._wireframe else GL.GL_FILL)
        return self._wireframe

    def gl_error(self) -> int:
        return int(GL.glGetError())

    def read_pixels(self) -> np.ndarray:
        """Read the front buffer (used by the screenshot tool)."""
        GL.glPixelStorei(GL.GL_PACK_ALIGNMENT, 1)
        data = GL.glReadPixels(0, 0, self.width, self.height, GL.GL_RGB,
                               GL.GL_UNSIGNED_BYTE)
        array = np.frombuffer(data, dtype=np.uint8).reshape(self.height, self.width, 3)
        return np.flipud(array)

    def shutdown(self) -> None:
        for vao in self._mesh_cache.values():
            vao.delete()
        self._mesh_cache.clear()
        for program in self.programs.values():
            program.delete()
        for texture in self.textures.values():
            GL.glDeleteTextures([texture])
        self.textures.clear()
        if hasattr(self, "_sky_buffers"):
            GL.glDeleteBuffers(2, list(self._sky_buffers))
        if hasattr(self, "_sky_vao"):
            GL.glDeleteVertexArrays(1, [self._sky_vao])


def _offset(byte_offset: int):
    """ctypes void pointer for glVertexAttribPointer offsets."""
    import ctypes
    return ctypes.c_void_p(byte_offset)
