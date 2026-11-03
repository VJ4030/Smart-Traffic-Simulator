"""Traffic signals: physical signal heads, adaptive control and glow rendering.

Two things live here:

1. :class:`SignalController` - the behaviour.  Every signalised junction runs a four stage
   cycle (green -> amber -> all-red clearance -> green for the crossing road) with
   **adaptive green durations** computed from the measured queue length of both approaches,
   clamped between a minimum and a maximum, plus a starvation guard so a busy direction
   can never be locked out.
2. :class:`SignalGeometry` (built by :meth:`SignalController.build_mesh`) - the physical
   signals.  Poles, mast arms, backing boards and lamp housings are baked into the static
   world mesh, while the *lenses* are drawn as instanced camera-facing glow sprites.  That
   is what makes the active lamp readable from 200 m away.

The module is pygame/OpenGL free so the state machine is unit-testable.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

from .mesh import Material, MeshBuilder, MeshData
from .road import Junction, Road, RoadNetwork
from .transform import clamp

RED = "red"
AMBER = "amber"
GREEN = "green"

STATE_ORDER = [RED, AMBER, GREEN]

#: control parameters (seconds)
MIN_GREEN = 9.0
MAX_GREEN = 34.0
AMBER_TIME = 3.0
ALL_RED_TIME = 1.6
STARVATION_TIME = 70.0
DEMAND_PER_VEHICLE = 1.35      # extra green seconds granted per queued vehicle
MAX_DEMAND_VEHICLES = 14

MAT_SIGNAL_POLE = Material("signal_pole", (0.30, 0.31, 0.33), specular=0.5)
MAT_SIGNAL_BOX = Material("signal_box", (0.11, 0.115, 0.12), specular=0.35)
MAT_SIGNAL_BACK = Material("signal_backing", (0.09, 0.10, 0.11), specular=0.20)
MAT_SIGNAL_LENS_OFF = Material("signal_lens", (0.10, 0.10, 0.11), specular=0.7)
MAT_SIGNAL_ARM = Material("signal_arm", (0.34, 0.35, 0.37), specular=0.5)

LAMP_COLORS = {
    RED: (1.0, 0.12, 0.09),
    AMBER: (1.0, 0.62, 0.08),
    GREEN: (0.16, 0.95, 0.36),
}
LAMP_DIM = 0.22      # brightness of an inactive lens


def _vec_from_yaw(yaw: float) -> Tuple[float, float]:
    return (math.cos(yaw), -math.sin(yaw))


def _sizes(facing: Tuple[float, float], along: float, height: float,
           width: float) -> Tuple[float, float, float]:
    """Box size whose *along* extent follows the facing direction."""
    if abs(facing[0]) >= abs(facing[1]):
        return (along, height, width)
    return (width, height, along)


@dataclass
class SignalHead:
    """One physical signal head facing one approach of a junction."""

    junction_id: int
    axis: str                      # "z" or "x" - the road the approach belongs to
    direction: int                 # +1 / -1 along that axis
    x: float                       # housing centre (above the carriageway)
    z: float
    yaw: float                     # direction the lenses face (towards oncoming traffic)
    pole_x: float
    pole_z: float
    arm_length: float
    height: float = 6.2
    lamp_spacing: float = 0.62
    overhead: bool = True

    @property
    def facing(self) -> Tuple[float, float]:
        return _vec_from_yaw(self.yaw)

    @property
    def lateral(self) -> Tuple[float, float]:
        fx, fz = self.facing
        return (-fz, fx)

    def lamp_positions(self) -> Dict[str, Tuple[float, float, float]]:
        """World positions of the red / amber / green lenses (red on top)."""
        fx, fz = self.facing
        cx, cz = self.x + fx * 0.44, self.z + fz * 0.44
        return {
            RED: (cx, self.height + self.lamp_spacing, cz),
            AMBER: (cx, self.height, cz),
            GREEN: (cx, self.height - self.lamp_spacing, cz),
        }


@dataclass
class JunctionPhase:
    """Adaptive controller state of a single junction."""

    junction: Junction
    active_axis: str = "z"
    state: str = GREEN
    timer: float = 0.0
    green_duration: float = MIN_GREEN
    served_since: Dict[str, float] = field(
        default_factory=lambda: {"z": 0.0, "x": STARVATION_TIME * 0.5})
    demand: Dict[str, int] = field(default_factory=lambda: {"z": 0, "x": 0})

    @property
    def other_axis(self) -> str:
        return "x" if self.active_axis == "z" else "z"


class SignalController:
    """Adaptive, deterministic traffic-signal controller for every signalised junction."""

    def __init__(self, network: RoadNetwork,
                 queue_provider: Optional[Callable[[Junction, str, int], int]] = None) -> None:
        self.network = network
        self.queue_provider = queue_provider
        self.phases: Dict[int, JunctionPhase] = {}
        self.heads: List[SignalHead] = []
        self.transitions = 0
        for junc in network.junctions:
            if not junc.signalised:
                continue
            self.phases[junc.id] = JunctionPhase(junc)
            self.heads.extend(self._build_heads(junc))
        # stagger the initial phases so the whole city is not in sync
        for i, phase in enumerate(self.phases.values()):
            phase.active_axis = "z" if i % 2 == 0 else "x"
            phase.state = GREEN
            phase.timer = float(i % 3) * 2.0

    # ------------------------------------------------------------------ geometry
    def _build_heads(self, junc: Junction) -> List[SignalHead]:
        heads: List[SignalHead] = []
        for road in (junc.road_a, junc.road_b):
            for direction in (1, -1):
                heads.append(self._make_head(junc, road, direction))
        return heads

    def _make_head(self, junc: Junction, road: Road, direction: int,
                   overhead: bool = True) -> SignalHead:
        """Signal head for traffic approaching this junction travelling ``direction``.

        The pole stands on the left kerb of the approach (left-hand traffic), the mast arm
        reaches over the carriageway and the lenses face the oncoming driver.
        """
        hw = road.half_width
        travel_yaw = road.direction_yaw(direction)
        facing_yaw = travel_yaw + math.pi
        if road.axis == "z":
            # travelling +Z: stop line is at smaller z, the left kerb is +X
            stop = junc.z - direction * (hw + 5.4)
            pole = (junc.x + direction * (hw + 1.3), stop)
            arm_length = hw * 0.55 + 1.3
            housing = (junc.x + direction * (hw * 0.45 + 1.3), stop)
        else:
            # travelling +X: stop line is at smaller x, the left kerb is -Z
            stop = junc.x - direction * (hw + 5.4)
            pole = (stop, junc.z - direction * (hw + 1.3))
            arm_length = hw * 0.55 + 1.3
            housing = (stop, junc.z - direction * (hw * 0.45 + 1.3))
        return SignalHead(junc.id, road.axis, direction, housing[0], housing[1],
                          facing_yaw, pole[0], pole[1], arm_length,
                          height=6.2 if overhead else 3.5, overhead=overhead)

    def build_mesh(self) -> MeshData:
        """Bake every signal pole, arm, board and housing into one static mesh."""
        b = MeshBuilder("traffic_signals")
        for junc in self.network.junctions:
            if not junc.signalised:
                continue
            for road in (junc.road_a, junc.road_b):
                for direction in (1, -1):
                    self._add_signal_structure(b, junc, road, direction)
        return b.build("traffic_signals")

    def _add_signal_structure(self, b: MeshBuilder, junc: Junction, road: Road,
                              direction: int) -> None:
        head = self._make_head(junc, road, direction)
        post = self._make_head(junc, road, direction, overhead=False)
        fx, fz = head.facing
        lx, lz = head.lateral

        # ---- vertical pole on the kerb + horizontal mast arm -----------------------
        b.add_cylinder(MAT_SIGNAL_POLE, 0.15, 3.6, (head.pole_x, 3.6, head.pole_z), 8, axis="y")
        arm_mid_x = (head.pole_x + head.x) * 0.5
        arm_mid_z = (head.pole_z + head.z) * 0.5
        arm_size = ((head.arm_length, 0.17, 0.17) if road.axis == "z"
                    else (0.17, 0.17, head.arm_length))
        b.add_box(MAT_SIGNAL_ARM, arm_size, (arm_mid_x, 7.15, arm_mid_z))
        b.add_box(MAT_SIGNAL_BACK, (0.20, 0.50, 0.20), (head.x, 6.95, head.z))
        # ---- overhead head: backing board + 3 housings + lenses --------------------
        b.add_box(MAT_SIGNAL_BACK, _sizes((fx, fz), 0.10, 2.05, 0.86), (head.x, head.height, head.z))
        for lamp_y in (head.height + head.lamp_spacing, head.height,
                       head.height - head.lamp_spacing):
            bx = head.x + fx * 0.20
            bz = head.z + fz * 0.20
            b.add_box(MAT_SIGNAL_BOX, _sizes((fx, fz), 0.42, 0.62, 0.62), (bx, lamp_y, bz),
                      top_scale=0.85)
            b.add_box(MAT_SIGNAL_LENS_OFF, _sizes((fx, fz), 0.12, 0.36, 0.36),
                      (head.x + fx * 0.34, lamp_y, head.z + fz * 0.34))
        # ---- post mounted repeating head (very common at Indian junctions) ---------
        b.add_box(MAT_SIGNAL_BACK, _sizes((fx, fz), 0.10, 1.80, 0.70),
                  (head.pole_x, post.height, head.pole_z))
        for lamp_y in (post.height + 0.55, post.height, post.height - 0.55):
            b.add_box(MAT_SIGNAL_BOX, _sizes((fx, fz), 0.40, 0.56, 0.50),
                      (head.pole_x + fx * 0.18, lamp_y, head.pole_z + fz * 0.18),
                      top_scale=0.85)
            b.add_box(MAT_SIGNAL_LENS_OFF, _sizes((fx, fz), 0.12, 0.32, 0.32),
                      (head.pole_x + fx * 0.34, lamp_y, head.pole_z + fz * 0.34))
        del lx, lz

    # ------------------------------------------------------------------ behaviour
    def update(self, dt: float) -> None:
        """Advance every junction's adaptive cycle by ``dt`` seconds."""
        for phase in self.phases.values():
            phase.timer += dt
            for axis in ("z", "x"):
                phase.served_since[axis] += dt
            phase.demand = self._measure_demand(phase)
            if phase.state == GREEN:
                if phase.timer >= phase.green_duration:
                    phase.state = AMBER
                    phase.timer = 0.0
                    self.transitions += 1
            elif phase.state == AMBER:
                if phase.timer >= AMBER_TIME:
                    phase.state = RED            # all-red clearance stage
                    phase.timer = 0.0
            else:                                # clearance finished -> hand over
                if phase.timer >= ALL_RED_TIME:
                    phase.active_axis = phase.other_axis
                    phase.state = GREEN
                    phase.timer = 0.0
                    phase.served_since[phase.active_axis] = 0.0
                    phase.green_duration = self._green_duration(phase)
                    self.transitions += 1

    def _measure_demand(self, phase: JunctionPhase) -> Dict[str, int]:
        if self.queue_provider is None:
            return {"z": 0, "x": 0}
        demand: Dict[str, int] = {}
        for axis in ("z", "x"):
            total = 0
            for direction in (1, -1):
                try:
                    total += max(0, int(self.queue_provider(phase.junction, axis, direction)))
                except Exception:                       # pragma: no cover - defensive
                    total += 0
            demand[axis] = min(total, MAX_DEMAND_VEHICLES)
        return demand

    def _green_duration(self, phase: JunctionPhase) -> float:
        """Adaptive green: proportional to the queue, clamped, with a starvation guard."""
        own = phase.demand.get(phase.active_axis, 0)
        cross = phase.demand.get(phase.other_axis, 0)
        duration = MIN_GREEN + DEMAND_PER_VEHICLE * own
        waited = phase.served_since.get(phase.other_axis, 0.0)
        if cross > 0 and waited > STARVATION_TIME:
            duration = MIN_GREEN          # serve the starved direction quickly
        return clamp(duration, MIN_GREEN, MAX_GREEN)

    def state_for(self, junction: Junction, axis: str, direction: int) -> str:
        """Current signal state seen by a vehicle on ``axis`` heading ``direction``."""
        phase = self.phases.get(junction.id)
        if phase is None:
            return GREEN
        if phase.active_axis == axis:
            if phase.state == GREEN:
                return GREEN
            if phase.state == AMBER:
                return AMBER
            return RED
        return RED

    def state_at(self, x: float, z: float, yaw: float,
                 search_radius: float = 48.0) -> Tuple[str, Optional[Junction], float]:
        """Signal state the player should obey, plus the junction and the distance to it."""
        fx, fz = math.cos(yaw), -math.sin(yaw)
        best: Optional[Tuple[float, Junction]] = None
        for junc in self.network.junctions:
            if not junc.signalised:
                continue
            dx, dz = junc.x - x, junc.z - z
            distance = math.hypot(dx, dz)
            if distance > search_radius or distance < 1e-3:
                continue
            ahead = (dx * fx + dz * fz) / distance
            if ahead < 0.35:
                continue
            if best is None or distance < best[0]:
                best = (distance, junc)
        if best is None:
            return GREEN, None, 0.0
        distance, junc = best
        road = self.network.road_at(x, z, 2.0)
        if road is not None:
            axis = road.axis
            direction = self.network.direction_of_travel(road, x, z)
        else:
            axis = "z" if abs(fz) >= abs(fx) else "x"
            direction = 1 if ((fz > 0) if axis == "z" else (fx > 0)) else -1
        return self.state_for(junc, axis, direction), junc, distance

    def stop_line_distance(self, x: float, z: float, yaw: float) -> float:
        """Distance from the vehicle to the stop line of the next signal ahead."""
        _, junc, distance = self.state_at(x, z, yaw)
        if junc is None:
            return -1.0
        road = self.network.road_at(x, z, 2.0)
        offset = (road.half_width + 5.4) if road is not None else junc.half_size
        return max(0.0, distance - offset)

    # ------------------------------------------------------------------ rendering
    def sprite_instances(self) -> List[Tuple[Tuple[float, float, float],
                                             Tuple[float, float],
                                             Tuple[float, float, float, float]]]:
        """Instanced glow sprites for every lamp lens (one draw call for all signals)."""
        out: List[Tuple[Tuple[float, float, float], Tuple[float, float],
                         Tuple[float, float, float, float]]] = []
        for head in self.heads:
            state = self.state_for_next(head)
            for lamp_state, position in head.lamp_positions().items():
                color = LAMP_COLORS[lamp_state]
                if lamp_state == state:
                    out.append((position, (0.44, 0.44), (color[0], color[1], color[2], 1.0)))
                    out.append((position, (1.60, 1.60), (color[0], color[1], color[2], 0.28)))
                else:
                    out.append((position, (0.30, 0.30),
                                (color[0] * LAMP_DIM, color[1] * LAMP_DIM,
                                 color[2] * LAMP_DIM, 0.85)))
        return out

    def state_for_next(self, head: SignalHead) -> str:
        """State of one specific head (junction phase, not the viewer's position)."""
        phase = self.phases.get(head.junction_id)
        if phase is None:
            return GREEN
        if phase.active_axis != head.axis:
            return RED
        return phase.state

    def queue_snapshot(self) -> Dict[str, Dict[str, int]]:
        """Human readable demand table (used by tests and the debug overlay)."""
        return {phase.junction.name: dict(phase.demand) for phase in self.phases.values()}

    def statistics(self) -> Dict[str, int]:
        return {"junctions": len(self.phases), "heads": len(self.heads),
                "transitions": self.transitions}
