"""GLSL shader sources and a thin compile/link helper.

The project targets the OpenGL 3.3 core profile, which is available on virtually every
Windows laptop GPU (including Intel HD/UHD integrated graphics) and on Mesa/llvmpipe
under Linux.  A Pygame OpenGL 2.1 fallback path also exists; in that case the renderer
automatically re-targets the shaders with a ``#version 120`` compatibility header so the
game still runs on very old drivers.

Shaders implemented here:
  * ``WORLD_*``   - lit, fogged geometry with per-draw material uniforms
  * ``DEPTH_*``   - shadow-map depth-only pass
  * ``SKY_*``     - procedural sky dome (gradient + sun disc + stars)
  * ``SPRITE_*``  - instanced camera-facing billboards (light glows, rain streaks)
"""

from __future__ import annotations

from typing import Dict

from OpenGL import GL

GLSL_VERSION_330 = "#version 330 core\n"
GLSL_VERSION_120 = "#version 120\n"


def _extract_version(source: str) -> str:
    first = source.lstrip().splitlines()[0]
    return first if first.startswith("#version") else ""


def _downgrade(source: str) -> str:
    """Best-effort conversion of a 330-core shader to GLSL 1.20.

    Only the syntax features actually used in this project are translated:
    texture lookups, ``in``/``out`` qualifiers, ``layout(location)`` and instanced
    attributes.  Attribute indices are bound explicitly by the renderer in that case.
    """
    if _extract_version(source) == "#version 330 core":
        source = source.replace("#version 330 core\n", "#version 120\n", 1)
    source = source.replace("in vec", "attribute vec").replace("in float", "attribute float")
    source = source.replace("in int", "attribute int")
    source = source.replace("uniform mat4", "uniform mat4")
    # vertex stage outputs
    source = source.replace("\nout vec", "\nvarying vec")
    source = source.replace("\nout float", "\nvarying float")
    source = source.replace("\nout mat", "\nvarying mat")
    # fragment stage inputs
    source = source.replace("\nin vec", "\nvarying vec")
    source = source.replace("\nin float", "\nvarying float")
    source = source.replace("out vec4 FragColor;", "")
    source = source.replace("texture(", "texture2D(")
    source = source.replace("layout(location = 0) ", "")
    source = source.replace("layout(location = 1) ", "")
    source = source.replace("layout(location = 2) ", "")
    source = source.replace("layout(location = 3) ", "")
    source = source.replace("layout(location = 4) ", "")
    source = source.replace("layout(location = 5) ", "")
    source = source.replace("gl_PointSize", "gl_PointSize")
    source = source.replace("gl_FragColor", "gl_FragColor")
    if "gl_FragColor" not in source and "FragColor" in source:
        source = source.replace("FragColor", "gl_FragColor")
    return source


# --------------------------------------------------------------------------------------
# world (lit geometry)
# --------------------------------------------------------------------------------------
WORLD_VERT = """
#version 330 core
layout(location = 0) in vec3 a_pos;
layout(location = 1) in vec3 a_normal;
layout(location = 2) in vec2 a_uv;

uniform mat4 u_mvp;
uniform mat4 u_model;
uniform mat3 u_normal_mat;

out vec3 v_world_pos;
out vec3 v_normal;
out vec2 v_uv;

void main() {
    vec4 world = u_model * vec4(a_pos, 1.0);
    v_world_pos = world.xyz;
    v_normal = normalize(u_normal_mat * a_normal);
    v_uv = a_uv;
    gl_Position = u_mvp * vec4(a_pos, 1.0);
}
"""

WORLD_FRAG = """
#version 330 core
in vec3 v_world_pos;
in vec3 v_normal;
in vec2 v_uv;

uniform vec3 u_color;
uniform vec3 u_emissive;
uniform vec3 u_light_dir;          // normalised, pointing *from* the surface *to* the sun
uniform vec3 u_light_color;
uniform vec3 u_ambient_sky;
uniform vec3 u_ambient_ground;
uniform vec3 u_fog_color;
uniform vec3 u_camera_pos;
uniform sampler2D u_sign_texture;
uniform float u_fog_density;
uniform float u_specular;
uniform float u_shadow_strength;
uniform float u_flash;             // 0..1 red flash used when the player takes damage
uniform float u_alpha;             // <1 for blended surfaces such as blob shadows
uniform int u_use_texture;         // 1: multiply by the bound texture (signage atlas)

out vec4 FragColor;

void main() {
    vec3 N = normalize(v_normal);
    vec3 L = normalize(u_light_dir);
    vec3 V = normalize(u_camera_pos - v_world_pos);

    // hemisphere ambient: sky colour from above, bounce colour from below
    float hemi = 0.5 + 0.5 * N.y;
    vec3 ambient = mix(u_ambient_ground, u_ambient_sky, hemi);

    float ndl = max(dot(N, L), 0.0);
    vec3 diffuse = u_light_color * ndl;

    // Blinn-Phong specular gives metals/glass/paint their highlight
    vec3 H = normalize(L + V);
    float spec = pow(max(dot(N, H), 0.0), 48.0) * u_specular * ndl;

    vec3 lit = u_color * (ambient + diffuse * u_shadow_strength);
    vec3 color = lit + vec3(spec) + u_emissive;

    // exponential-squared distance fog, colour matched to the current sky
    float dist = length(u_camera_pos - v_world_pos);
    float fog = 1.0 - exp(-pow(dist * u_fog_density, 2.0));
    color = mix(color, u_fog_color, clamp(fog, 0.0, 1.0));

    if (u_use_texture == 1) {
        vec3 tex = texture(u_sign_texture, v_uv).rgb;
        color *= tex * 1.15 + 0.10;
    }
    color = mix(color, vec3(1.0, 0.15, 0.12), u_flash * 0.45);
    FragColor = vec4(color, u_alpha);
}
"""

# --------------------------------------------------------------------------------------
# depth-only pass (shadow map)
# --------------------------------------------------------------------------------------
DEPTH_VERT = """
#version 330 core
layout(location = 0) in vec3 a_pos;
uniform mat4 u_light_mvp;
void main() {
    gl_Position = u_light_mvp * vec4(a_pos, 1.0);
}
"""

DEPTH_FRAG = """
#version 330 core
void main() { }
"""

# --------------------------------------------------------------------------------------
# procedural sky
# --------------------------------------------------------------------------------------
SKY_VERT = """
#version 330 core
layout(location = 0) in vec2 a_pos;
uniform mat4 u_inv_view_proj;
uniform vec3 u_camera_pos;
out vec3 v_dir;
void main() {
    vec4 near = u_inv_view_proj * vec4(a_pos, -1.0, 1.0);
    vec4 far  = u_inv_view_proj * vec4(a_pos,  1.0, 1.0);
    v_dir = normalize(far.xyz / far.w - near.xyz / near.w);
    gl_Position = vec4(a_pos, 0.999999, 1.0);
}
"""

SKY_FRAG = """
#version 330 core
in vec3 v_dir;
uniform vec3 u_zenith;
uniform vec3 u_horizon;
uniform vec3 u_ground;
uniform vec3 u_sun_dir;
uniform vec3 u_sun_color;
uniform float u_star_intensity;
uniform float u_sun_size;
out vec4 FragColor;

float hash(vec2 p) {
    return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453123);
}

void main() {
    vec3 d = normalize(v_dir);
    float t = clamp(d.y, -1.0, 1.0);
    vec3 sky = mix(u_horizon, u_zenith, pow(clamp(t, 0.0, 1.0), 0.55));
    if (t < 0.0) {
        sky = mix(u_horizon, u_ground, clamp(-t * 3.0, 0.0, 1.0));
    }
    // sun / moon disc with a soft glow
    float cosang = dot(d, normalize(u_sun_dir));
    float disc = smoothstep(1.0 - u_sun_size, 1.0 - u_sun_size * 0.35, cosang);
    float glow = pow(max(cosang, 0.0), 220.0) * 0.8 + pow(max(cosang, 0.0), 12.0) * 0.18;
    sky += u_sun_color * (disc * 4.0 + glow);

    // stars fade in at night and above the horizon
    if (u_star_intensity > 0.01) {
        vec2 grid = floor(d.xz * 220.0 / max(abs(d.y) + 0.05, 0.05));
        float s = hash(grid);
        float star = smoothstep(0.995, 1.0, s) * u_star_intensity * smoothstep(0.0, 0.35, d.y);
        sky += vec3(star) * 3.0;
    }
    FragColor = vec4(sky, 1.0);
}
"""

# --------------------------------------------------------------------------------------
# instanced billboards: light glows, rain streaks, particle sparks
# --------------------------------------------------------------------------------------
SPRITE_VERT = """
#version 330 core
layout(location = 0) in vec2 a_quad;              // -1..1 quad corner
layout(location = 3) in vec3 i_center;
layout(location = 4) in vec2 i_size;              // width, height (world units)
layout(location = 5) in vec4 i_color;

uniform mat4 u_view_proj;
uniform vec3 u_camera_right;
uniform vec3 u_camera_up;

out vec2 v_quad;
out vec4 v_color;

void main() {
    vec3 world = i_center + u_camera_right * (a_quad.x * i_size.x * 0.5)
                          + u_camera_up * (a_quad.y * i_size.y * 0.5);
    v_quad = a_quad;
    v_color = i_color;
    gl_Position = u_view_proj * vec4(world, 1.0);
}
"""

SPRITE_FRAG = """
#version 330 core
in vec2 v_quad;
in vec4 v_color;
out vec4 FragColor;
void main() {
    float r = length(v_quad);
    float alpha = smoothstep(1.0, 0.15, r);
    if (alpha <= 0.001) discard;
    FragColor = vec4(v_color.rgb, v_color.a * alpha);
}
"""

UI_VERT = """
#version 330 core
layout(location = 0) in vec2 a_pos;
layout(location = 2) in vec2 a_uv;
out vec2 v_uv;
void main() {
    v_uv = a_uv;
    gl_Position = vec4(a_pos, 0.0, 1.0);
}
"""

UI_FRAG = """
#version 330 core
in vec2 v_uv;
uniform sampler2D u_texture;
uniform float u_alpha;
out vec4 FragColor;
void main() {
    vec4 c = texture(u_texture, v_uv);
    FragColor = vec4(c.rgb, c.a * u_alpha);
}
"""

SHADERS: Dict[str, Dict[str, str]] = {
    "ui": {"vertex": UI_VERT, "fragment": UI_FRAG},
    "world": {"vertex": WORLD_VERT, "fragment": WORLD_FRAG},
    "depth": {"vertex": DEPTH_VERT, "fragment": DEPTH_FRAG},
    "sky": {"vertex": SKY_VERT, "fragment": SKY_FRAG},
    "sprite": {"vertex": SPRITE_VERT, "fragment": SPRITE_FRAG},
}


class ShaderError(RuntimeError):
    pass


class ShaderProgram:
    """Compiled + linked GLSL program with a uniform-location cache."""

    def __init__(self, vertex_source: str, fragment_source: str, name: str = "program",
                 core_profile: bool = True) -> None:
        self.name = name
        self.core_profile = core_profile
        if not core_profile:
            vertex_source = _downgrade(vertex_source)
            fragment_source = _downgrade(fragment_source)
        self.program = GL.glCreateProgram()
        vs = self._compile(vertex_source, GL.GL_VERTEX_SHADER)
        fs = self._compile(fragment_source, GL.GL_FRAGMENT_SHADER)
        GL.glAttachShader(self.program, vs)
        GL.glAttachShader(self.program, fs)
        GL.glLinkProgram(self.program)
        if not GL.glGetProgramiv(self.program, GL.GL_LINK_STATUS):
            log = GL.glGetProgramInfoLog(self.program)
            raise ShaderError(f"{name}: link failed: {log!r}")
        GL.glDeleteShader(vs)
        GL.glDeleteShader(fs)
        self._uniforms: Dict[str, int] = {}

    def _compile(self, source: str, kind: int) -> int:
        shader = GL.glCreateShader(kind)
        GL.glShaderSource(shader, source)
        GL.glCompileShader(shader)
        if not GL.glGetShaderiv(shader, GL.GL_COMPILE_STATUS):
            log = GL.glGetShaderInfoLog(shader)
            listing = "\n".join(f"{i+1:>4}: {line}" for i, line in enumerate(source.splitlines()))
            raise ShaderError(f"{self.name}: compile failed: {log!r}\n{listing}")
        return shader

    def use(self) -> None:
        GL.glUseProgram(self.program)

    def location(self, uniform: str) -> int:
        loc = self._uniforms.get(uniform)
        if loc is None:
            loc = GL.glGetUniformLocation(self.program, uniform)
            self._uniforms[uniform] = loc
        return loc

    def set_int(self, uniform: str, value: int) -> None:
        loc = self.location(uniform)
        if loc >= 0:
            GL.glUniform1i(loc, int(value))

    def set_float(self, uniform: str, value: float) -> None:
        loc = self.location(uniform)
        if loc >= 0:
            GL.glUniform1f(loc, float(value))

    def set_vec2(self, uniform: str, x: float, y: float) -> None:
        loc = self.location(uniform)
        if loc >= 0:
            GL.glUniform2f(loc, float(x), float(y))

    def set_vec3(self, uniform: str, x: float, y: float, z: float) -> None:
        loc = self.location(uniform)
        if loc >= 0:
            GL.glUniform3f(loc, float(x), float(y), float(z))

    def set_color(self, uniform: str, color) -> None:
        self.set_vec3(uniform, color[0], color[1], color[2])

    def set_vec4(self, uniform: str, color) -> None:
        loc = self.location(uniform)
        if loc >= 0:
            GL.glUniform4f(loc, float(color[0]), float(color[1]), float(color[2]), float(color[3]))

    def set_mat4(self, uniform: str, matrix) -> None:
        loc = self.location(uniform)
        if loc >= 0:
            GL.glUniformMatrix4fv(loc, 1, GL.GL_TRUE, matrix.astype("float32"))
        return None

    def set_mat3(self, uniform: str, matrix) -> None:
        loc = self.location(uniform)
        if loc >= 0:
            GL.glUniformMatrix3fv(loc, 1, GL.GL_TRUE, matrix.astype("float32"))

    def delete(self) -> None:
        if self.program:
            GL.glDeleteProgram(self.program)
            self.program = 0


def build_programs(core_profile: bool = True) -> Dict[str, ShaderProgram]:
    """Compile every program used by the renderer."""
    return {name: ShaderProgram(src["vertex"], src["fragment"], name=name,
                                core_profile=core_profile)
            for name, src in SHADERS.items()}
