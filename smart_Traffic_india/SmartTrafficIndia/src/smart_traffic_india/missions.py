"""Missions: career objectives, parking challenges and driving-school lessons.

A mission is a list of **phases**.  Every phase has a kind, a target and parameters, and
the runtime evaluates the live world every frame:

====================  ==========================================================
phase kind            completion condition
====================  ==========================================================
``goto``              reach a point within a radius
``stop``              reach a point, stop (speed < 1.5 m/s) and hold for N seconds
``park``              park inside a marked bay: position, angle, collisions, time
``signals``           pass N signalised junctions while the light is green
``speed_limit``       drive a distance while never exceeding a speed
``breakers``          cross N speed breakers below a speed
``give_way``          an ambulance approaches and the player yields
====================  ==========================================================

That covers every objective in the brief, and adding a mission is pure data.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from .road import LANDMARKS, RoadNetwork
from .vehicle import Vehicle


# --------------------------------------------------------------------------------------
# phase + mission definitions
# --------------------------------------------------------------------------------------
@dataclass
class Phase:
    """One objective inside a mission."""

    kind: str
    x: float = 0.0
    z: float = 0.0
    radius: float = 9.0
    hold: float = 1.0
    count: int = 2
    speed: float = 22.0
    distance: float = 200.0
    label: str = ""
    lot: str = ""
    bay: int = 0
    angle_tolerance: float = 42.0
    bay_tolerance: float = 1.35

    def position(self) -> Tuple[float, float]:
        return (self.x, self.z)


@dataclass
class Mission:
    """A complete mission definition."""

    id: str
    title: str
    objective: str
    detail: str
    phases: List[Phase]
    reward: int
    time_limit: float = -1.0
    start: Tuple[float, float, float] = (4.0, -40.0, -math.pi / 2)
    traffic_level: Optional[str] = None
    time_of_day: Optional[str] = None
    weather: Optional[str] = None
    category: str = "career"
    tip: str = ""

    @property
    def destination(self) -> Tuple[float, float]:
        last = self.phases[-1]
        return (last.x, last.z)


def _landmark(name: str) -> Tuple[float, float]:
    return LANDMARKS.get(name, (0.0, 0.0))


#: ---------------------------------------------------------------- career missions
def build_career_missions() -> List[Mission]:
    hospital = _landmark("Civil Hospital")
    station = _landmark("Central Railway Station")
    bazaar = _landmark("Main Bazaar")
    industrial = _landmark("Industrial Estate")
    mall = _landmark("Metro Mall Downtown")
    airport = _landmark("Airport Road Junction")
    terminal = _landmark("City Bus Terminal")
    toll = _landmark("Bypass Toll Plaza")

    return [
        Mission(
            "m01_first_day", "First Day On The Road",
            "Drive to the Civil Hospital",
            "Learn the controls: follow MG Road, keep left and reach the hospital gate.",
            [Phase("goto", hospital[0], hospital[1], 15.0, label="Civil Hospital gate")],
            reward=1400, start=(6.0, -60.0, -math.pi / 2), time_of_day="morning",
            tip="Press C to change camera, L for headlights and H for the horn."),
        Mission(
            "m02_passenger", "Passenger Pickup",
            "Pick up a passenger at the bazaar, drop them at the station",
            "Stop completely at the pickup point, then head to the railway station.",
            [Phase("stop", bazaar[0] + 4.0, bazaar[1] + 8.0, 13.0, hold=1.5,
                   label="Bazaar pickup point"),
             Phase("goto", station[0] + 6.0, station[1] + 10.0, 16.0,
                   label="Central Railway Station")],
            reward=2400, start=(2.0, 40.0, -math.pi / 2), traffic_level="heavy",
            tip="Buses and autos stop without warning in the bazaar - keep your distance."),
        Mission(
            "m03_hospital_delivery", "Hospital Delivery Run",
            "Deliver medical supplies to the Industrial Estate",
            "The clinic is waiting: reach the estate within the time limit.",
            [Phase("stop", hospital[0] - 6.0, hospital[1] + 6.0, 13.0, hold=1.0,
                   label="Hospital supply bay"),
             Phase("goto", industrial[0] - 8.0, industrial[1] + 8.0, 17.0,
                   label="Industrial Estate")],
            reward=2800, time_limit=150.0, start=(-88.0, -70.0, 0.0),
            traffic_level="medium",
            tip="Speed breakers near the hospital - ease over them."),
        Mission(
            "m04_junction", "Busy Junction Challenge",
            "Cross the city centre junction and two more signals on green",
            "Read the signal, stop at the line and cross only when it is green.",
            [Phase("signals", count=3, label="Signals crossed on green"),
             Phase("goto", mall[0] - 10.0, mall[1] - 10.0, 16.0, label="Metro Mall")],
            reward=1900, start=(-4.0, -30.0, -math.pi / 2), traffic_level="heavy",
            tip="Amber means prepare to stop - do not accelerate through it."),
        Mission(
            "m05_signal_compliance", "Signal Compliance Test",
            "Pass five signals without a single violation",
            "A traffic constable is watching. Cross five green lights cleanly.",
            [Phase("signals", count=5, label="Clean signal crossings")],
            reward=2600, time_limit=210.0, start=(0.0, -120.0, -math.pi / 2),
            traffic_level="busy",
            tip="Jumping a red light costs 35 points and breaks your perfect run."),
        Mission(
            "m06_parallel_parking", "Parallel Parking Test",
            "Park neatly inside the marked bay in the bazaar",
            "Align with the bay, reverse in gently and stop inside the lines.",
            [Phase("park", lot="market", bay=2, radius=8.0,
                   angle_tolerance=38.0, bay_tolerance=1.25, label="Bazaar parking bay")],
            reward=2200, time_limit=110.0, start=(-20.0, 70.0, math.pi),
            tip="Use the close camera (C) for parking - you can see both corners."),
        Mission(
            "m07_construction", "Road Works Ahead",
            "Navigate the road works and deliver cement to the estate",
            "Barricades and a closed lane - slow down and squeeze past.",
            [Phase("goto", -70.0, -62.0, 15.0, label="Road works zone"),
             Phase("goto", industrial[0] + 6.0, industrial[1] - 10.0, 17.0,
                   label="Industrial Estate yard")],
            reward=2400, start=(-4.0, -110.0, -math.pi / 2), weather="cloudy",
            tip="Barricades are solid: hitting them damages the bodywork."),
        Mission(
            "m08_rush_hour", "Rush Hour Sprint",
            "Beat the evening rush to the bypass toll plaza",
            "Traffic is at its worst: smooth driving pays more than raw speed.",
            [Phase("goto", toll[0] - 4.0, toll[1] - 20.0, 22.0, label="Bypass Toll Plaza")],
            reward=3200, time_limit=230.0, start=(2.0, -150.0, -math.pi / 2),
            traffic_level="busy", time_of_day="evening",
            tip="Overtake only when the neighbouring lane is genuinely clear."),
        Mission(
            "m09_night_shift", "Night Shift",
            "Deliver a late consignment to the airport road",
            "Headlights on, fewer cars but faster ones, harder to read the signals.",
            [Phase("stop", terminal[0] + 10.0, terminal[1] + 16.0, 15.0, hold=1.0,
                   label="Bus terminal cargo shed"),
             Phase("goto", airport[0] + 4.0, airport[1] - 16.0, 19.0,
                   label="Airport Road Junction")],
            reward=3400, time_limit=210.0, start=(-60.0, -140.0, 0.0),
            time_of_day="night", traffic_level="light",
            tip="Headlights are automatic at night; use the horn at blind corners."),
        Mission(
            "m10_code_blue", "Code Blue",
            "Clear the way for an ambulance, then report to the hospital",
            "An ambulance is approaching from behind - slow down and let it pass.",
            [Phase("give_way", label="Give way to the ambulance"),
             Phase("goto", hospital[0] + 8.0, hospital[1] - 8.0, 16.0,
                   label="Civil Hospital emergency bay")],
            reward=4200, time_limit=190.0, start=(4.0, -20.0, -math.pi / 2),
            traffic_level="heavy",
            tip="Pull to the left and slow down: giving way takes two seconds."),
    ]


#: ---------------------------------------------------------------- parking challenges
def build_parking_challenges() -> List[Mission]:
    return [
        Mission(
            "p01_reverse", "Reverse Bay Parking",
            "Reverse into the hospital bay",
            "Drive past the bay, then reverse in with the car straight inside the lines.",
            [Phase("park", lot="hospital", bay=0, radius=8.0, angle_tolerance=34.0,
                   bay_tolerance=1.25, label="Hospital bay 1")],
            reward=1500, time_limit=110.0, start=(38.0, -78.0, 0.0), category="parking",
            tip="Straighten up before you enter the bay for a higher score."),
        Mission(
            "p02_parallel", "Parallel Parking",
            "Slot into the narrow bazaar bay",
            "The classic test: pull alongside, reverse in, straighten and stop.",
            [Phase("park", lot="market", bay=1, radius=7.0, angle_tolerance=30.0,
                   bay_tolerance=1.1, label="Bazaar kerb bay")],
            reward=1800, time_limit=100.0, start=(-24.0, 92.0, math.pi), category="parking",
            tip="Small steering corrections beat one big turn."),
        Mission(
            "p03_oldcity", "Old City Street Parking",
            "Park on the crowded Purani Basti street",
            "Narrow lane, parked autos everywhere - take your time.",
            [Phase("park", lot="oldcity", bay=3, radius=7.5, angle_tolerance=36.0,
                   bay_tolerance=1.3, label="Old City bay 4")],
            reward=2000, time_limit=115.0, start=(-92.0, 14.0, math.pi / 2), category="parking",
            tip="Watch the parked auto-rickshaws on your right."),
        Mission(
            "p04_mall", "Mall Basement Parking",
            "Park in the mall row with the clock running",
            "Two tight rows, one marked bay - a timed precision test.",
            [Phase("park", lot="mall", bay=6, radius=7.0, angle_tolerance=28.0,
                   bay_tolerance=1.05, label="Mall bay 7")],
            reward=2400, time_limit=90.0, start=(60.0, 14.0, -math.pi / 2), category="parking",
            tip="Use the handbrake (Space) once you are inside the bay."),
    ]


#: ---------------------------------------------------------------- driving school
def build_school_lessons() -> List[Mission]:
    school = _landmark("Sunrise Public School")
    return [
        Mission(
            "s01_start_stop", "Lesson 1 - Start and Stop",
            "Reach the school zone, then stop inside the marker",
            "Throttle is W or the up arrow, brake is S or the down arrow.",
            [Phase("goto", school[0], school[1] + 30.0, 15.0, label="School zone marker"),
             Phase("stop", school[0], school[1] + 12.0, 12.0, hold=1.5,
                   label="Stop marker")],
            reward=600, start=(-40.0, 76.0, -math.pi / 2), category="school",
            tip="Smooth braking: come off the throttle early, then brake gently."),
        Mission(
            "s02_signal", "Lesson 2 - Signals and Stop Lines",
            "Stop at the red light, then cross on green",
            "The stop line is painted before every signalised junction.",
            [Phase("signals", count=1, label="One signal crossed on green"),
             Phase("goto", 0.0, 52.0, 16.0, label="Downtown boulevard")],
            reward=700, start=(-4.0, -30.0, -math.pi / 2), category="school",
            tip="The HUD signal box mirrors the light you are facing."),
        Mission(
            "s03_keep_left", "Lesson 3 - Keep Left",
            "Drive 320 m on the correct side of the road",
            "India drives on the left: staying in your lane avoids head-on traffic.",
            [Phase("speed_limit", distance=320.0, speed=45.0, label="Keep left for 320 m")],
            reward=800, start=(4.0, -120.0, -math.pi / 2), category="school",
            tip="Wrong-side driving is penalised after a fair warning."),
        Mission(
            "s04_breakers", "Lesson 4 - Speed Breakers",
            "Cross two speed breakers below 18 km/h",
            "The humps are real geometry: hit them fast and your suspension pays.",
            [Phase("breakers", count=2, speed=18.0, label="Breakers crossed slowly")],
            reward=800, start=(-36.0, 30.0, -math.pi / 2), category="school",
            tip="Brake before the hump, not on it."),
        Mission(
            "s05_parking", "Lesson 5 - Parking Basics",
            "Park inside the hospital bay at your own pace",
            "No time pressure - get the car straight and inside the lines.",
            [Phase("park", lot="hospital", bay=2, radius=8.0, angle_tolerance=45.0,
                   bay_tolerance=1.6, label="Hospital practice bay")],
            reward=900, start=(34.0, -52.0, 0.0), category="school",
            tip="The parking score rewards a centred car and a straight angle."),
    ]


MISSION_CATEGORIES = ("career", "parking", "school")


# --------------------------------------------------------------------------------------
# runtime
# --------------------------------------------------------------------------------------
@dataclass
class MissionResult:
    """Outcome of a finished mission."""

    success: bool
    mission_id: str
    title: str
    reward: int = 0
    score: int = 0
    perfect: bool = False
    message: str = ""
    stats: Dict[str, object] = field(default_factory=dict)
    reason: str = ""


class MissionRuntime:
    """Runs one mission against the live world."""

    def __init__(self, mission: Mission, network: RoadNetwork,
                 parking_lots: Optional[Dict[str, object]] = None,
                 player_car: str = "hatchback") -> None:
        self.mission = mission
        self.network = network
        self.lots = parking_lots or {}
        self.player_car = player_car
        self.phase_index = 0
        self.elapsed = 0.0
        self.hold_timer = 0.0
        self.phase_progress = 0.0
        self.signals_passed = 0
        self.breakers_done = 0
        self.distance_driven = 0.0
        self.park_best_score = 0
        self.parked = False
        self.park_message = ""
        self.give_way_done = False
        self.give_way_timer = 0.0
        self.finished = False
        self.success = False
        self.fail_reason = ""
        self.events: List[Tuple[str, str]] = []
        self._prev_junction: Optional[int] = None
        self._last_breaker: Optional[int] = None

    # ------------------------------------------------------------------ info
    @property
    def phase(self) -> Optional[Phase]:
        if 0 <= self.phase_index < len(self.mission.phases):
            return self.mission.phases[self.phase_index]
        return None

    @property
    def objective_text(self) -> str:
        phase = self.phase
        if phase is None:
            return self.mission.title
        return phase.label or self.mission.objective

    @property
    def time_left(self) -> float:
        if self.mission.time_limit <= 0:
            return -1.0
        return max(0.0, self.mission.time_limit - self.elapsed)

    def progress(self) -> float:
        total = max(1, len(self.mission.phases))
        done = self.phase_index / total
        return min(1.0, done + self.phase_progress / total)

    def destination(self) -> Optional[Tuple[float, float]]:
        phase = self.phase
        if phase is None:
            return None
        if phase.kind == "park":
            lot = self.lots.get(phase.lot)
            if lot is not None:
                bays = lot.bays()                      # type: ignore[attr-defined]
                if bays:
                    bay = bays[min(phase.bay, len(bays) - 1)]
                    return (bay[0], bay[1])
        return phase.position()

    def distance_to_destination(self, player: Vehicle) -> float:
        target = self.destination()
        if target is None:
            return -1.0
        return math.hypot(target[0] - player.x, target[1] - player.z)

    def direction_hint(self, player: Vehicle) -> str:
        target = self.destination()
        if target is None:
            return "ahead"
        fx, fz = player.forward()
        dx, dz = target[0] - player.x, target[1] - player.z
        distance = math.hypot(dx, dz) or 1.0
        ahead = (dx * fx + dz * fz) / distance
        if ahead > 0.6:
            return "ahead"
        if ahead < -0.6:
            return "behind"
        side = dx * (-fz) + dz * fx
        return "right" if side > 0 else "left"

    # ------------------------------------------------------------------ update
    def update(self, dt: float, player: Vehicle, signals, traffic=None,
               events: Optional[List[Tuple[str, str]]] = None) -> Optional[MissionResult]:
        """Advance the mission.  Returns a result exactly once, when it ends."""
        if self.finished:
            return None
        self.elapsed += dt
        phase = self.phase
        if phase is None:
            result = self._finish(True, "All objectives complete")
            if events is not None:
                events.extend(self.events)
                self.events.clear()
            return result
        self.distance_driven += abs(player.speed) * dt

        if self.mission.time_limit > 0 and self.elapsed > self.mission.time_limit:
            result = self._finish(False, "Out of time")
            if events is not None:
                events.extend(self.events)
                self.events.clear()
            return result

        kind = phase.kind
        if kind == "goto":
            self._update_goto(phase, player)
        elif kind == "stop":
            self._update_stop(phase, player, dt)
        elif kind == "park":
            self._update_park(phase, player, dt)
        elif kind == "signals":
            self._update_signals(phase, player, signals)
        elif kind == "speed_limit":
            self._update_speed_limit(phase, player, dt)
        elif kind == "breakers":
            self._update_breakers(phase, player)
        elif kind == "give_way":
            self._update_give_way(phase, player, traffic, dt)
        if events is not None:
            events.extend(self.events)
            self.events.clear()
        return None

    # -- phase implementations -------------------------------------------------------
    def _advance(self) -> None:
        self.phase_index += 1
        self.phase_progress = 0.0
        self.hold_timer = 0.0

    def _update_goto(self, phase: Phase, player: Vehicle) -> None:
        distance = math.hypot(phase.x - player.x, phase.z - player.z)
        self.phase_progress = max(0.0, 1.0 - distance / max(phase.radius * 6.0, 1.0))
        if distance <= phase.radius:
            self.events.append(("checkpoint",
                                f"Checkpoint reached: {phase.label or 'waypoint'}"))
            self._advance()

    def _update_stop(self, phase: Phase, player: Vehicle, dt: float) -> None:
        distance = math.hypot(phase.x - player.x, phase.z - player.z)
        if distance <= phase.radius and abs(player.speed) < 1.5:
            self.hold_timer += dt
            self.phase_progress = min(1.0, self.hold_timer / max(phase.hold, 0.1))
            if self.hold_timer >= phase.hold:
                self.events.append(("checkpoint",
                                    f"Stopped at {phase.label or 'the marker'}"))
                self._advance()
        else:
            self.hold_timer = max(0.0, self.hold_timer - dt)
            self.phase_progress = max(0.0, 1.0 - distance / 45.0)

    def _update_park(self, phase: Phase, player: Vehicle, dt: float) -> None:
        lot = self.lots.get(phase.lot)
        if lot is None:
            self._advance()
            return
        bays = lot.bays()                                   # type: ignore[attr-defined]
        if not bays:
            self._advance()
            return
        bx, bz, bay_yaw = bays[min(phase.bay, len(bays) - 1)]
        dx, dz = player.x - bx, player.z - bz
        along = dx * math.cos(bay_yaw) - dz * math.sin(bay_yaw)
        across = dx * math.sin(bay_yaw) + dz * math.cos(bay_yaw)
        yaw_error = abs(math.atan2(math.sin(player.yaw - bay_yaw),
                                   math.cos(player.yaw - bay_yaw)))
        angle_deg = math.degrees(yaw_error)
        in_bay = (abs(across) <= phase.bay_tolerance and abs(along) <= 3.2
                  and angle_deg <= phase.angle_tolerance)
        stopped = abs(player.speed) < 1.2
        position_quality = max(0.0, 1.0 - abs(across) / max(phase.bay_tolerance, 0.1))
        angle_quality = max(0.0, 1.0 - angle_deg / max(phase.angle_tolerance, 0.1))
        quality = position_quality * 0.6 + angle_quality * 0.4
        self.phase_progress = quality * (1.0 if stopped else 0.6)
        self.park_best_score = max(self.park_best_score, int(quality * 100))
        if in_bay and stopped:
            self.hold_timer += dt
            self.park_message = f"Parking quality {int(quality * 100)}% - hold position"
            if self.hold_timer >= 1.4:
                self.parked = True
                self.events.append(("checkpoint",
                                    f"Parked cleanly ({int(quality * 100)}% quality)"))
                self._advance()
        else:
            self.hold_timer = 0.0
            if not stopped:
                self.park_message = "Stop the car inside the bay"
            elif angle_deg > phase.angle_tolerance:
                self.park_message = "Straighten up"
            else:
                self.park_message = "Move into the bay"

    def _update_signals(self, phase: Phase, player: Vehicle, signals) -> None:
        if signals is None:
            return
        state, junction, _distance = signals.state_at(player.x, player.z, player.yaw)
        if junction is None:
            self._prev_junction = None
            return
        if junction.id != self._prev_junction:
            self._prev_junction = junction.id
            if state in ("green", "amber"):
                self.signals_passed += 1
                self.events.append(("checkpoint",
                                    f"Signal crossed on {state} ({self.signals_passed}/"
                                    f"{phase.count})"))
            else:
                self.events.append(("warning", "Signal run on red - it does not count"))
        self.phase_progress = min(1.0, self.signals_passed / max(phase.count, 1))
        if self.signals_passed >= phase.count:
            self._advance()

    def _update_speed_limit(self, phase: Phase, player: Vehicle, dt: float) -> None:
        if player.speed_kmh <= phase.speed:
            self.phase_progress = min(
                1.0, self.phase_progress + dt * abs(player.speed) / max(phase.distance, 1.0))
        else:
            self.events.append(("warning", f"Keep under {phase.speed:.0f} km/h"))
            self.phase_progress = max(0.0, self.phase_progress - dt * 0.05)
        if self.phase_progress >= 1.0:
            self.events.append(("checkpoint", "Lane discipline maintained"))
            self._advance()

    def _update_breakers(self, phase: Phase, player: Vehicle) -> None:
        for hazard in self.network.hazards:
            if hazard.kind != "speed_breaker" or id(hazard) == self._last_breaker:
                continue
            if hazard.contains(player.x, player.z):
                self._last_breaker = id(hazard)
                if player.speed_kmh <= phase.speed:
                    self.breakers_done += 1
                    self.events.append(("checkpoint",
                                        f"Breaker crossed slowly ({self.breakers_done}/"
                                        f"{phase.count})"))
                else:
                    self.events.append(("warning", "Too fast over the speed breaker"))
        if self.breakers_done >= phase.count:
            self._advance()
        self.phase_progress = min(1.0, self.breakers_done / max(phase.count, 1))

    def _update_give_way(self, phase: Phase, player: Vehicle, traffic, dt: float) -> None:
        if traffic is None:
            self.give_way_done = True
            self._advance()
            return
        emergency_exists = getattr(traffic, "emergency", None) is not None
        near = (traffic.emergency_near(player.x, player.z, 60.0)
                if hasattr(traffic, "emergency_near") else False)
        if near and not self.give_way_done:
            if player.speed_kmh < 25.0:
                self.give_way_timer += dt
            else:
                self.give_way_timer = max(0.0, self.give_way_timer - dt * 0.5)
            self.phase_progress = min(1.0, self.give_way_timer / 2.5)
            self.park_message = "Ambulance behind you - slow down and keep left"
            if self.give_way_timer >= 2.5:
                self.give_way_done = True
                self.events.append(("checkpoint", "Ambulance given way - well done"))
                self._advance()
        elif not near and not emergency_exists:
            self.give_way_done = True
            self._advance()

    # ------------------------------------------------------------------ finishing
    def _finish(self, success: bool, reason: str) -> MissionResult:
        self.finished = True
        self.success = success
        score = int(self.phase_index * 100 + self.park_best_score * 2)
        return MissionResult(success, self.mission.id, self.mission.title,
                             reward=self.mission.reward if success else 0,
                             score=score, message=reason, reason=reason,
                             stats={"elapsed": round(self.elapsed, 1),
                                    "parking_quality": self.park_best_score,
                                    "signals": self.signals_passed})

    def abort(self, reason: str = "Mission aborted") -> MissionResult:
        return self._finish(False, reason)
