"""The game: state machine, driving loop, menu wiring and frame composition.

Running order of a frame
------------------------
1. drain pygame events (input + menus),
2. advance the simulation with a **fixed time step** (deterministic physics) while the
   renderer runs at whatever rate the display allows,
3. compose the scene: sky, static city chunks (frustum culled), traffic signals, AI
   vehicles, the player car, glow/rain billboards, blob shadows,
4. draw the pygame HUD / menu on top and present.

Every state (``main_menu``, ``driving``, ``paused``, ``garage``, ``results`` ...) has one
entry point for events and one for rendering, so adding a screen never touches the loop.
"""

from __future__ import annotations

import math
import time
from typing import Dict, List, Optional, Sequence, Tuple

import pygame

from . import settings as S
from .audio import AudioEngine
from .camera import Camera
from .garage import Garage
from .hud import HUD, HUDSnapshot
from .input_manager import InputManager
from .menus import (ACTION, GarageScreen, MenuScreen, build_help_menu, build_main_menu,
                    build_pause_menu, build_results_menu, build_settings_menu)
from .missions import (Mission, MissionResult, MissionRuntime, build_career_missions,
                       build_parking_challenges, build_school_lessons)
from .particles import ParticleSystem
from .player import Player
from .renderer import Renderer
from .save_manager import SaveManager
from .signage import build_atlas
from .vehicle_models import ModelLibrary
from .world import World

# --------------------------------------------------------------------------------------
# states
# --------------------------------------------------------------------------------------
MAIN_MENU = "main_menu"
CAREER = "career"
PARKING = "parking"
SCHOOL = "school"
BRIEFING = "briefing"
DRIVING = "driving"
PAUSED = "paused"
RESULTS = "results"
GARAGE = "garage"
SETTINGS = "settings"
HELP = "help"

PETROL_STATION = (-62.0, -26.0)


class Game:
    """Top level application object."""

    def __init__(self, settings: Optional[S.Settings] = None, want_hud: bool = True) -> None:
        pygame.init()
        # NOTE: the mixer is initialised by AudioEngine, which synthesises its samples at
        # a fixed rate - initialising it twice would change the pitch of every sound.
        self.save = SaveManager()
        self.settings = settings or self.save.load_settings()
        self.settings = self.settings.sanitise()
        if not S.python_version_ok():                       # pragma: no cover
            raise RuntimeError("Python 3.9 or newer is required")
        self.garage = Garage(self.save)
        quality = self.settings.quality_flags
        self.atlas = build_atlas(int(quality.get("max_sign_textures", 8)))
        self.models = ModelLibrary()
        self.models.preload()
        self.renderer = Renderer(self.settings, quality)
        if self.atlas.available and self.atlas.surface is not None:
            self.renderer.upload_texture("signs", self.atlas.surface)
        self.renderer.clear_only()
        self.flash("Generating the city ...")
        self.world = World(self.settings, self.atlas, self.models)
        self.world.seeds_pedestrians((4.0, -40.0), 12)
        self.camera = Camera(self._aspect(), mode=self.settings.camera_mode)
        self.camera.set_far(float(quality.get("far_plane", 420.0)))
        self.camera.collision_hint = self.world.environment.colliders()
        self.hud = HUD(self.settings.width, self.settings.height)
        self.input = InputManager(invert_steering=self.settings.invert_steering)
        self.player: Optional[Player] = None
        self.mission: Optional[Mission] = None
        self.runtime: Optional[MissionRuntime] = None
        self.pending_mission: Optional[Mission] = None
        self.state = MAIN_MENU
        self.previous_state = MAIN_MENU
        self.running = True
        self.toasts: List[Tuple[str, str, float]] = []
        self.session_reward = 0
        self.last_result: Optional[MissionResult] = None
        self.free_drive = True
        self.show_fps = self.settings.show_fps
        self.look_back = False
        self.autosave_timer = 0.0
        self.fps = 0.0
        self.frame_times: List[float] = []
        self.last_frame_time = 0.0
        self.wireframe = False
        self.paused_from = DRIVING
        self.backdrop_timer = 0.0
        self.minimap_roads = self._build_minimap_roads()
        self.help_menu = build_help_menu(self.input.control_help())
        self.settings_menu = None
        self.mission_menu: Optional[MenuScreen] = None
        self.main_menu = build_main_menu()
        self.pause_menu = build_pause_menu()
        self.results_menu: Optional[MenuScreen] = None
        self.garage_screen = GarageScreen(self.garage)
        self.particles = ParticleSystem(capacity=300,
                                        quality=float(quality.get("traffic_scale", 1.0)))
        self.audio = AudioEngine(enabled=self.settings.master_volume > 0.0,
                                 master=self.settings.master_volume)
        self.world.traffic.event_sink = self._world_event
        self.gl_info = self.renderer.gl_info
        self._spawn_showcase_vehicle()

    # ------------------------------------------------------------------ helpers
    def _aspect(self) -> float:
        video = pygame.display.Info()
        width = self.settings.width or video.current_w or 1280
        height = self.settings.height or video.current_h or 720
        return max(0.5, width / max(height, 1))

    def flash(self, message: str) -> None:
        """Draw a one-off loading message (used while the city is generated)."""
        surface = pygame.Surface((self.settings.width, self.settings.height))
        surface.fill((10, 12, 18))
        font = pygame.font.Font(None, 44)
        small = pygame.font.Font(None, 26)
        title = font.render(S.APP_TITLE, True, (255, 168, 40))
        text = small.render(message, True, (210, 214, 222))
        surface.blit(title, ((self.settings.width - title.get_width()) // 2,
                            self.settings.height // 2 - 60))
        surface.blit(text, ((self.settings.width - text.get_width()) // 2,
                            self.settings.height // 2))
        pygame.display.flip()

    def _build_minimap_roads(self) -> List[Tuple[float, float, float, float, float]]:
        out = []
        for road in self.world.network.roads:
            lo, hi = min(road.start, road.end), max(road.start, road.end)
            if road.axis == "z":
                out.append((road.center, lo, road.center, hi, road.half_width * 2))
            else:
                out.append((lo, road.center, hi, road.center, road.half_width * 2))
        return out

    def _spawn_showcase_vehicle(self) -> None:
        """A car on the road in front of the menu camera so the title screen has life."""
        key = self.garage.selected_key()
        player = Player(key, self.garage.selected_upgrades(),
                        start=(1.8, -46.0, math.pi / 2), network=self.world.network)
        player.vehicle.fuel = player.vehicle.spec.tank
        player.vehicle.damage = self.garage.damage_of(key)
        self.player = player
        self.camera.mode = self.settings.camera_mode
        self.camera.cinematic = True
        self.world.prime_traffic(player, 0.5)
        self.world.seeds_pedestrians((player.x, player.z), 8)

    def _world_event(self, kind: str, message: str) -> None:
        self.add_toast(kind, message)

    def add_toast(self, kind: str, message: str, seconds: float = 4.0) -> None:
        self.toasts.append((kind, message, seconds))
        if len(self.toasts) > 8:
            self.toasts.pop(0)

    # ------------------------------------------------------------------ main loop
    def run(self) -> None:
        clock = pygame.time.Clock()
        accumulator = 0.0
        while self.running:
            frame_start = time.perf_counter()
            raw_dt = clock.tick(240) / 1000.0
            dt = min(raw_dt, S.MAX_FRAME_TIME)
            self._process_events()

            if self.state == DRIVING:
                accumulator += dt
                steps = 0
                while accumulator >= S.FIXED_DT and steps < S.MAX_STEPS_PER_FRAME:
                    self._step(S.FIXED_DT)
                    accumulator -= S.FIXED_DT
                    steps += 1
                if steps >= S.MAX_STEPS_PER_FRAME:
                    accumulator = 0.0
            else:
                self._idle_step(dt)
                accumulator = 0.0

            self._render(dt)
            self.input.end_frame()

            frame_time = time.perf_counter() - frame_start
            self.frame_times.append(frame_time)
            if len(self.frame_times) > 60:
                self.frame_times.pop(0)
            average = sum(self.frame_times) / len(self.frame_times)
            self.fps = 1.0 / max(average, 1e-5)
            self.last_frame_time = frame_time
            self.save.add_playtime(dt)
            self.autosave_timer += dt
            if self.autosave_timer > 30.0:
                self.autosave_timer = 0.0
                self.persist()
        self.shutdown()

    def _process_events(self) -> None:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
                continue
            self.input.process_event(event)
            if self.state in (MAIN_MENU, CAREER, PARKING, SCHOOL, SETTINGS, HELP, RESULTS):
                self._menu_event(event)
            elif self.state == BRIEFING:
                self._briefing_event(event)
            elif self.state == DRIVING:
                self._driving_event(event)
            elif self.state == PAUSED:
                self._pause_event(event)
            elif self.state == GARAGE:
                self._garage_event(event)

    # ------------------------------------------------------------------ simulation
    def _step(self, dt: float) -> None:
        player = self.player
        if player is None:
            return
        inputs = self.input.driving_snapshot()
        player.weather_rain = self.world.environment.rain_intensity > 0.35
        feedback = player.update(dt, inputs, self.world.signals, self.world.hazards)
        for warning, seconds in feedback.warnings:
            self.add_toast("warning", warning, seconds)
        for kind, message in feedback.events:
            self.add_toast(kind, message, 4.0)
        if feedback.horn:
            self.player.vehicle.horn_timer = 0.4
        frame = self.world.update(dt, player, time_cycle=self.free_drive)
        # ---- feedback: particles + audio (never affects physics) ----------------------
        wet = self.world.environment.rain_intensity > 0.3
        offroad = getattr(player.vehicle, "surface", "road") == "offroad"
        locked = bool(inputs.get("handbrake")) or player.vehicle.brake_light > 0.85
        self.particles.vehicle_feedback(player.vehicle, dt, wet, offroad, locked)
        self.particles.update(dt)
        self.audio.update(dt)
        self.audio.update_from_vehicle(player.vehicle, float(inputs.get("throttle", False)),
                                       wet)
        for event in frame.collisions:
            if not event.is_player:
                continue
            self.audio.crash(event.severity)
            fx, fz = player.vehicle.forward()
            self.particles.debris((player.vehicle.x + fx * player.vehicle.half_length, 0.6,
                                   player.vehicle.z + fz * player.vehicle.half_length),
                                  event.severity)
        # speed limit compliance is handled inside the player controller; damage is charged
        # to the save file so the garage can repair it later
        self.garage.set_damage(self.garage.selected_key(), player.vehicle.damage)
        if player.vehicle.fuel <= 0.01 and not getattr(self, "_fuel_toast", False):
            self._fuel_toast = True
            self.add_toast("fuel", "Out of fuel - find the petrol pump (F)")
        if player.vehicle.fuel > 4.0:
            self._fuel_toast = False
        if frame.emergency_near and self.state == DRIVING:
            self.world.traffic.emergency_announced = True
        # mission progress
        if self.runtime is not None and not self.free_drive:
            events: List[Tuple[str, str]] = []
            result = self.runtime.update(dt, player.vehicle, self.world.signals,
                                         self.world.traffic, events)
            for kind, message in events:
                self.add_toast(kind, message, 3.5)
            if result is not None:
                self._finish_mission(result)
        # toasts decay
        self.toasts = [(k, m, t - dt) for k, m, t in self.toasts if t - dt > 0.0]

    def _idle_step(self, dt: float) -> None:
        """World keeps living behind the menus (cinematic camera)."""
        self.backdrop_timer += dt
        self.audio.stop_engine()
        self.audio.update(dt)
        self.particles.update(dt)
        # the world keeps breathing behind the menus (the camera is updated in _render)
        if self.state == MAIN_MENU:
            self.world.signals.update(dt)
            self.world.environment.update(dt, time_cycle=False)
        else:
            self.world.signals.update(dt * 0.35)
            self.world.environment.update(dt, time_cycle=False)

    # ------------------------------------------------------------------ state changes
    def _start_free_drive(self) -> None:
        player = self.player
        assert player is not None
        start = (6.0, -60.0, -math.pi / 2)
        player.teleport(*start)
        player.vehicle.fuel = max(player.vehicle.fuel, 18.0)
        self.mission = None
        self.runtime = None
        self.free_drive = True
        self.camera.cinematic = False
        self.world.traffic.set_level(self.settings.traffic_level)
        self.world.environment.set_time(self.settings.time_of_day)
        self.world.environment.set_weather(self.settings.weather)
        self.world.traffic.despawn_emergency()
        self.world.prime_traffic(player, 0.75)
        self.world.seeds_pedestrians((player.x, player.z), 10)
        self.state = DRIVING
        self.add_toast("explore", "Free Drive - the whole city is yours", 5.0)

    def _start_mission(self, mission: Mission) -> None:
        key = self.garage.selected_key()
        self.player = Player(key, self.garage.selected_upgrades(),
                             start=mission.start, network=self.world.network)
        self.player.vehicle.damage = self.garage.damage_of(key)
        self.player.vehicle.fuel = max(20.0, self.player.vehicle.spec.tank * 0.85)
        self.mission = mission
        self.runtime = MissionRuntime(mission, self.world.network, self.world.parking_lots, key)
        self.free_drive = False
        self.camera.cinematic = False
        self.world.apply_mission(mission, self.player)
        self.world.prime_traffic(self.player, 0.65)
        self.world.seeds_pedestrians((self.player.x, self.player.z), 8)
        self.state = DRIVING
        self.add_toast("reward", f"{mission.title}", 4.0)
        if mission.tip:
            self.add_toast("explore", mission.tip, 6.0)
        self.session_reward = 0

    def _finish_mission(self, result: MissionResult) -> None:
        self.last_result = result
        player = self.player
        assert player is not None
        summary = player.summary()
        perfect = bool(result.success and summary["violations"] == 0 and summary["collisions"] == 0)
        if result.success:
            bonus = int(result.reward * S.MISSION_REWARD_PERFECT_BONUS) if perfect else 0
            total = result.reward + bonus + int(result.score * 0.5)
            self.save.record_mission(result.mission_id, total, perfect,
                                     int(summary["collisions"]), int(summary["violations"]))
            self.session_reward = total
            self.add_toast("reward", f"Mission complete! +Rs {total:,}", 6.0)
        else:
            self.save.record_mission("", 0, False, int(summary["collisions"]),
                                     int(summary["violations"]))
        best = 0
        if self.mission is not None:
            best = int(getattr(self.runtime, "park_best_score", 0))
            if best:
                self.save.record_parking(self.mission.id, best)
        distance_km = float(summary["distance_km"])
        self.save.add_distance(distance_km)
        self.save.save()
        title = "Mission Complete" if result.success else "Mission Failed"
        subtitle = (f"{result.title}   |   {result.message}"
                    + (f"   |   Parking {best}%" if best else "")
                    + (f"   |   Rs {self.session_reward:,} earned" if self.session_reward else ""))
        self.results_menu = build_results_menu(title, subtitle)
        self.state = RESULTS

    def _exit_to_menu(self) -> None:
        self.camera.cinematic = True
        self.mission = None
        self.runtime = None
        self.free_drive = True
        self.world.traffic.set_level(self.settings.traffic_level)
        self.world.traffic.despawn_emergency()
        self.state = MAIN_MENU
        self.persist()

    def shutdown(self) -> None:
        """Save progress and release GL / audio resources (safe to call twice)."""
        if getattr(self, "_shut_down", False):
            return
        self._shut_down = True
        try:
            self.persist()
        except Exception:                                    # pragma: no cover
            pass
        self.audio.shutdown()
        self.renderer.shutdown()
        pygame.quit()

    def persist(self) -> None:
        self.save.store_settings(self.settings)
        if self.player is not None:
            self.save.set_damage(self.garage.selected_key(), self.player.vehicle.damage)
        self.save.save()

    # ------------------------------------------------------------------ events: menus
    def _menu_event(self, event: pygame.event.Event) -> None:
        if self.state == MAIN_MENU:
            action = self.main_menu.handle_event(event)
            if action:
                self._dispatch_main(action)
        elif self.state in (CAREER, PARKING, SCHOOL):
            if self.mission_menu is not None:
                action = self.mission_menu.handle_event(event)
                if action is None and event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                    self.state = MAIN_MENU
                elif action:
                    self._dispatch_mission_list(action)
        elif self.state == SETTINGS:
            self._ensure_settings_menu()
            action = self.settings_menu.handle_event(event) if self.settings_menu else None
            if action:
                self._dispatch_settings(action)
            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                self._apply_settings()
                self.state = MAIN_MENU
        elif self.state == HELP:
            action = self.help_menu.handle_event(event)
            if action or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                self.state = MAIN_MENU
        elif self.state == RESULTS:
            if self.results_menu is not None:
                action = self.results_menu.handle_event(event)
                if action:
                    self._dispatch_results(action)

    def _dispatch_main(self, action: str) -> None:
        tag = action.split(":", 1)[1] if ":" in action else action
        if tag == "career":
            self.mission_menu = self._build_mission_menu(
                "Career Mode", "Choose a mission (Esc to go back)", build_career_missions(),
                "mission")
            self.state = CAREER
        elif tag == "free":
            self._start_free_drive()
        elif tag == "parking":
            self.mission_menu = self._build_mission_menu(
                "Parking Challenges", "Precision parking with a timer", build_parking_challenges(),
                "mission")
            self.state = PARKING
        elif tag == "school":
            self.mission_menu = self._build_mission_menu(
                "Driving School", "Five lessons, no pressure", build_school_lessons(), "mission")
            self.state = SCHOOL
        elif tag == "garage":
            self.garage_screen.view = "cars"
            self.state = GARAGE
        elif tag == "settings":
            self._ensure_settings_menu()
            self.state = SETTINGS
        elif tag == "help":
            self.help_menu = build_help_menu(self.input.control_help())
            self.state = HELP
        elif tag == "exit":
            self.running = False

    def _build_mission_menu(self, title: str, subtitle: str, missions: Sequence[Mission],
                            tag_prefix: str) -> MenuScreen:
        menu = MenuScreen(title, subtitle, "Enter: brief and start   |   Esc: back")
        completed = set(self.save.data.get("completed_missions", []))
        for mission in missions:
            state = "done" if mission.id in completed else f"Rs {mission.reward:,}"
            hint = mission.objective if len(mission.objective) < 62 else mission.objective[:59] + "..."
            menu.add(f"{mission.title}   [{state}]", ACTION, hint=hint,
                     tag=f"{tag_prefix}:{mission.id}")
        return menu

    def _dispatch_mission_list(self, action: str) -> None:
        tag = action.split(":", 1)[1] if action.count(":") else action
        mission_id = action.split(":", 2)[2] if action.count(":") >= 2 else ""
        pool: List[Mission] = []
        if self.state == CAREER:
            pool = build_career_missions()
        elif self.state == PARKING:
            pool = build_parking_challenges()
        elif self.state == SCHOOL:
            pool = build_school_lessons()
        for mission in pool:
            if mission.id == mission_id:
                self.pending_mission = mission
                self.state = BRIEFING
                self.camera.cinematic = False
                return
        del tag

    def _dispatch_results(self, action: str) -> None:
        tag = action.split(":", 1)[1] if ":" in action else action
        if tag == "continue":
            self._start_free_drive()
        elif tag == "next":
            self._start_next_mission()
        elif tag == "retry":
            if self.mission is not None:
                self._start_mission(self.mission)
            else:
                self._start_free_drive()
        elif tag == "garage":
            self.garage_screen.view = "cars"
            self.state = GARAGE
        elif tag == "mainmenu":
            self._exit_to_menu()

    def _start_next_mission(self) -> None:
        missions = build_career_missions()
        current_id = self.mission.id if self.mission else None
        index = 0
        for i, mission in enumerate(missions):
            if mission.id == current_id:
                index = i + 1
                break
        if index < len(missions):
            self._start_mission(missions[index])
        else:
            self.add_toast("reward", "Career complete - all missions cleared!", 6.0)
            self._start_free_drive()

    def _ensure_settings_menu(self) -> None:
        self.settings_menu = build_settings_menu(self.settings)
        self.settings_menu.footer = "Left/Right change   |   Esc saves and returns"

    def _dispatch_settings(self, action: str) -> None:
        tag = action.split(":", 1)[1] if ":" in action else action
        if tag == "back":
            self._apply_settings()
            self.state = MAIN_MENU
        else:
            self._apply_settings(save=False)

    def _apply_settings(self, save: bool = True) -> None:
        self.settings.sanitise()
        self.input.invert_steering = self.settings.invert_steering
        self.world.traffic.set_level(self.settings.traffic_level)
        self.world.quality = self.settings.quality_flags
        self.world.environment.quality = self.settings.quality_flags
        self.renderer.quality = self.settings.quality_flags
        self.audio.set_master(self.settings.master_volume)
        self.world.environment.set_weather(self.settings.weather)
        self.world.environment.set_time(self.settings.time_of_day)
        self.camera.set_far(float(self.settings.quality_flags.get("far_plane", 420.0)))
        if self.state != DRIVING:
            self.camera.mode = self.settings.camera_mode
        if save:
            self.save.store_settings(self.settings)
            self.save.save()

    def _briefing_event(self, event: pygame.event.Event) -> None:
        if event.type != pygame.KEYDOWN:
            return
        if event.key in (pygame.K_RETURN, pygame.K_KP_ENTER, pygame.K_SPACE):
            if self.pending_mission is not None:
                self._start_mission(self.pending_mission)
                self.pending_mission = None
        elif event.key == pygame.K_ESCAPE:
            self.pending_mission = None
            self.state = MAIN_MENU

    def _pause_event(self, event: pygame.event.Event) -> None:
        if event.type != pygame.KEYDOWN:
            return
        if event.key == pygame.K_ESCAPE:
            self.state = self.paused_from
            return
        action = self.pause_menu.handle_event(event)
        if not action:
            return
        tag = action.split(":", 1)[1] if ":" in action else action
        if tag == "resume":
            self.state = self.paused_from
        elif tag == "restart":
            if self.mission is not None:
                self._start_mission(self.mission)
            else:
                self._start_free_drive()
        elif tag == "settings":
            self.paused_from = PAUSED
            self._ensure_settings_menu()
            self.state = SETTINGS
        elif tag == "help":
            self.paused_from = PAUSED
            self.help_menu = build_help_menu(self.input.control_help())
            self.state = HELP
        elif tag == "mainmenu":
            self._exit_to_menu()

    def _driving_event(self, event: pygame.event.Event) -> None:
        if event.type == pygame.KEYDOWN:
            if event.key in (pygame.K_ESCAPE, pygame.K_p):
                self.paused_from = DRIVING
                self.state = PAUSED
            elif event.key == pygame.K_c:
                mode = self.camera.cycle_mode()
                self.settings.camera_mode = mode
                self.add_toast("explore", f"Camera: {mode}", 2.0)
            elif event.key == pygame.K_m:
                self.settings.show_minimap = not self.settings.show_minimap
            elif event.key == pygame.K_f:
                self._try_refuel()
            elif event.key == pygame.K_r:
                self._unstick()
            elif event.key in (pygame.K_F3, pygame.K_g):
                self.wireframe = self.renderer.toggle_wireframe()
                self.add_toast("explore", f"Geometry view {'ON' if self.wireframe else 'OFF'}",
                               2.0)
            elif event.key in (pygame.K_F1, pygame.K_h):
                if event.key == pygame.K_F1:
                    self.add_toast("explore", "Hold H for the horn - press F1 for help", 2.5)
                    self.help_menu = build_help_menu(self.input.control_help())
                    self.paused_from = DRIVING
                    self.state = HELP
        elif event.type == pygame.KEYUP and event.key == pygame.K_b:
            self.look_back = False

    def _garage_event(self, event: pygame.event.Event) -> None:
        action = self.garage_screen.handle_event(event)
        if not action:
            return
        parts = action.split(":")
        verb = parts[1] if len(parts) > 1 else ""
        if verb == "exit":
            self.state = MAIN_MENU
            return
        if verb == "view":
            return
        if verb == "select" and len(parts) > 2:
            car = parts[2]
            if self.garage.select(car):
                self.add_toast("reward", f"{self.garage.display_name(car)} selected", 3.0)
                self._refresh_player_car(car)
        elif verb == "buy" and len(parts) > 2:
            car = parts[2]
            if self.garage.buy(car):
                self.garage.select(car)
                self.add_toast("reward",
                               f"Purchased {self.garage.display_name(car)} for "
                               f"Rs {self.garage.price_of(car):,}", 5.0)
                self._refresh_player_car(car)
            else:
                self.add_toast("violation", "Not enough money for that car", 3.0)
        elif verb == "upgrade" and len(parts) > 3:
            car, system = parts[2], parts[3]
            cost = self.garage.upgrade_cost(car, system)
            if self.garage.buy_upgrade(car, system):
                self.add_toast("reward", f"{system.title()} upgraded (-Rs {cost:,})", 4.0)
                self._refresh_player_car(car)
            else:
                self.add_toast("violation", "Upgrade not possible (money or max level)", 3.0)
        elif verb == "paint" and len(parts) > 3:
            car, paint = parts[2], parts[3]
            if self.garage.set_paint(car, paint):
                self.add_toast("reward", f"Resprayed in {paint}", 4.0)
        elif verb == "repair" and len(parts) > 2:
            car = parts[2]
            cost = self.garage.repair_cost(car)
            if self.garage.repair(car):
                self.add_toast("reward", f"Bodywork repaired (-Rs {cost:,})", 4.0)
            else:
                self.add_toast("violation", "Not enough money to repair", 3.0)

    def _refresh_player_car(self, car: str) -> None:
        if self.player is None:
            return
        from .vehicle import get_spec
        from .vehicle_models import ModelLibrary
        self.player.car_key = car
        self.player.upgrades = self.garage.selected_upgrades()
        damage = self.player.vehicle.damage
        fuel = self.player.vehicle.fuel
        self.player.vehicle.spec = self.garage.spec_of(car)
        self.player.vehicle.wheel_radius = self.player.vehicle.spec.wheel_radius
        self.player.vehicle.damage = damage
        self.player.vehicle.fuel = fuel
        del get_spec, ModelLibrary

    # ------------------------------------------------------------------ driving actions
    def _try_refuel(self) -> None:
        player = self.player
        if player is None:
            return
        distance = math.hypot(player.x - PETROL_STATION[0], player.z - PETROL_STATION[1])
        if distance > 16.0:
            self.add_toast("warning", "No petrol pump here - look for INDANE petrol pump", 3.0)
            return
        missing = player.vehicle.spec.tank - player.vehicle.fuel
        cost = int(missing * 105)
        if missing < 0.5:
            self.add_toast("explore", "Tank is already full", 2.5)
            return
        if self.save.money < cost:
            self.add_toast("violation", f"Refuelling needs Rs {cost:,}", 3.0)
            return
        self.save.add_money(-cost)
        player.vehicle.refuel()
        self.add_toast("reward", f"Refuelled for Rs {cost:,}", 4.0)

    def _unstick(self) -> None:
        player = self.player
        if player is None:
            return
        player.vehicle.unstick(self.world.network)
        player.vehicle.damage = min(100.0, player.vehicle.damage + 1.0)
        self.add_toast("warning", "Car recovered to the nearest road (+1 damage)", 3.0)

    # ------------------------------------------------------------------ rendering
    def _render(self, dt: float) -> None:
        renderer = self.renderer
        # ---- camera ------------------------------------------------------------------
        # The camera is updated here (once per *rendered* frame) rather than in the fixed
        # physics step: it is a view concern, it needs the real frame time for its damping
        # and it must be valid in every state - including the very first frame, before any
        # simulation step has run (the projection would otherwise still be identity).
        player = self.player
        if player is not None:
            self.camera.cinematic = self.state == MAIN_MENU
            self.camera.mode = (self.settings.camera_mode if self.state != DRIVING
                                else self.camera.mode)
            # full frame time while driving or watching the title orbit; slowed down
            # inside menus so the background stays calm
            if self.state in (DRIVING, MAIN_MENU):
                camera_dt = dt
            else:
                camera_dt = dt * 0.35
            terrain = float(getattr(player.vehicle, "bump_offset", 0.0)) * 0.6
            self.camera.update(camera_dt, player.vehicle, terrain_height=terrain,
                               reverse_view=self.look_back and self.state == DRIVING)
        lighting = self.world.environment.lighting()
        renderer.begin_frame(self.camera, lighting)
        # ---- static city ------------------------------------------------------------
        renderer.draw_mesh(self.world.static_meshes["ground"])
        frustum = self.camera.frustum()
        renderer.draw_chunks(self.world.environment.chunk_list(), frustum)
        renderer.draw_mesh(self.world.static_meshes["roads"])
        renderer.draw_mesh(self.world.static_meshes["signals"])
        # ---- vehicles ---------------------------------------------------------------
        self._render_vehicles(lighting)
        # ---- glow sprites (signals, street lights, headlights, rain) ----------------
        self._collect_sprites(lighting, dt)
        renderer.flush_sprites()
        # ---- HUD / menus -------------------------------------------------------------
        surface = pygame.Surface((self.settings.width, self.settings.height), pygame.SRCALPHA)
        if self.state == DRIVING and self.settings.show_hud:
            surface.blit(self.hud.render(self._hud_snapshot()), (0, 0))
        if self.state in (MAIN_MENU,):
            self._draw_title(surface)
            self.main_menu.render(surface)
        elif self.state in (CAREER, PARKING, SCHOOL):
            self._draw_menu_backdrop(surface)
            if self.mission_menu:
                self.mission_menu.render(surface)
        elif self.state == SETTINGS:
            self._draw_menu_backdrop(surface)
            if self.settings_menu:
                self.settings_menu.render(surface)
        elif self.state == HELP:
            self._draw_menu_backdrop(surface)
            self.help_menu.render(surface)
        elif self.state == GARAGE:
            self.garage_screen.render(surface)
        elif self.state == PAUSED:
            self._draw_menu_backdrop(surface)
            self.pause_menu.render(surface)
        elif self.state == BRIEFING:
            self._draw_briefing(surface)
        elif self.state == RESULTS:
            self._draw_results(surface)
        renderer.draw_ui(surface)
        renderer.present(dt)

    def _render_vehicles(self, lighting: Dict[str, object]) -> None:
        renderer = self.renderer
        night = float(lighting.get("night_factor", 0.0))
        shadow_strength = max(0.25, 1.0 - 0.45 * night)
        camera_position = self.camera.position
        frustum = self.camera.frustum()
        # ---- AI traffic --------------------------------------------------------------
        for vehicle in list(self.world.traffic.vehicles) + list(self.world.traffic.parked):
            distance = math.hypot(vehicle.x - camera_position[0], vehicle.z - camera_position[2])
            if distance > 190.0:
                continue
            if not frustum.sphere_visible((vehicle.x, 1.0, vehicle.z),
                                          vehicle.half_length + 1.0):
                continue
            model = self.models.traffic(vehicle.kind)
            paint = self._traffic_paint(vehicle)
            renderer.draw_blob_shadow(vehicle.x, vehicle.z, vehicle.half_length,
                                      vehicle.half_width, vehicle.yaw, shadow_strength)
            kwargs = {
                "lights_on": vehicle.vehicle.headlights or night > 0.35,
                "brake": vehicle.vehicle.brake_light > 0.3,
                "indicator": vehicle.vehicle.indicator,
                "hazards": vehicle.vehicle.hazard_lights,
                "beacons": bool(vehicle.emergency and self.world.environment.streetlights_on)
                           or vehicle.emergency,
            }
            renderer.draw_vehicle(model, vehicle.vehicle, paint, resolver_kwargs=kwargs)
        # ---- player -------------------------------------------------------------------
        player = self.player
        if player is None:
            return
        vehicle = player.vehicle
        distance = math.hypot(vehicle.x - camera_position[0], vehicle.z - camera_position[2])
        if distance < 240.0:
            model = self.models.player(player.car_key)
            renderer.draw_blob_shadow(vehicle.x, vehicle.z, vehicle.half_length,
                                      vehicle.half_width, vehicle.yaw, shadow_strength)
            kwargs = {
                "lights_on": vehicle.headlights or night > 0.3,
                "brake": vehicle.brake_light > 0.3 or vehicle.reverse_light,
                "indicator": 0 if vehicle.hazard_lights else vehicle.indicator,
                "hazards": vehicle.hazard_lights,
                "beacons": False,
                "albedo": None,
            }
            renderer.draw_vehicle(model, vehicle, self.garage.selected_paint(),
                                  resolver_kwargs=kwargs)

    def _traffic_paint(self, vehicle) -> Tuple[float, float, float]:
        """Stable pseudo-random body colour per AI vehicle (no per-frame RNG needed)."""
        index = abs(hash(vehicle.name)) % len(S.P.TRAFFIC_PAINT)
        if vehicle.kind == "ambulance":
            return (0.92, 0.93, 0.95)
        if vehicle.kind == "police":
            return (0.90, 0.91, 0.93)
        if vehicle.kind == "bus":
            return (0.30, 0.35, 0.62)
        if vehicle.kind == "truck":
            return (0.55, 0.45, 0.25)
        if vehicle.kind == "auto":
            return (0.88, 0.72, 0.12)
        return S.P.TRAFFIC_PAINT[index]

    def _collect_sprites(self, lighting: Dict[str, object], dt: float) -> None:
        renderer = self.renderer
        camera = self.camera
        night = float(lighting.get("night_factor", 0.0))
        # ---- traffic signal lenses (one instanced batch for the whole city) ----------
        for instance in self.world.signals.sprite_instances():
            position, size, color = instance
            distance = math.hypot(position[0] - camera.position[0],
                                  position[2] - camera.position[2])
            if distance < 170.0:
                renderer.push_sprite(position, size, color)
        # ---- street lights -----------------------------------------------------------
        if self.world.environment.streetlights_on:
            for position in self.world.environment.street_lights:
                distance = math.hypot(position[0] - camera.position[0],
                                      position[2] - camera.position[2])
                if distance < 110.0:
                    renderer.push_sprite((position[0], position[1] - 0.15, position[2]),
                                         (2.4, 2.4), (1.0, 0.86, 0.58, 0.30))
                    renderer.push_sprite((position[0], position[1] - 0.15, position[2]),
                                         (0.8, 0.8), (1.0, 0.94, 0.76, 0.85))
        # ---- vehicle lamps -----------------------------------------------------------
        blink = int(time.time() * 2.4) % 2 == 0
        ambient = max(night, 0.25 * float(getattr(self.world.environment, "rain_intensity", 0.0)))
        for vehicle in self.world.traffic.vehicles:
            distance = math.hypot(vehicle.x - camera.position[0], vehicle.z - camera.position[2])
            if distance > 110.0:
                continue
            model = self.models.traffic(vehicle.kind)
            self._push_vehicle_lamps(model, vehicle.vehicle, distance, night, blink,
                                     emergency=vehicle.emergency)
        player = self.player
        if player is not None:
            model = self.models.player(player.car_key)
            distance = math.hypot(player.x - camera.position[0], player.z - camera.position[2])
            self._push_vehicle_lamps(model, player.vehicle, distance, night, blink,
                                     emergency=False)
            # headlight cones (a soft pool of light in front of the car at night)
            if player.vehicle.headlights and night > 0.25:
                fx, fz = player.vehicle.forward()
                for side in (-1.0, 1.0):
                    rx, rz = player.vehicle.right()
                    px = player.x + fx * (player.vehicle.half_length + 4.5) + rx * side * 0.7
                    pz = player.z + fz * (player.vehicle.half_length + 4.5) + rz * side * 0.7
                    renderer.push_sprite((px, 0.15, pz), (5.5, 1.2),
                                         (1.0, 0.96, 0.82, 0.16 * night))
        # ---- mission destination beacon ----------------------------------------------
        if self.runtime is not None:
            target = self.runtime.destination()
            if target is not None:
                pulse = 0.5 + 0.5 * math.sin(time.time() * 3.0)
                renderer.push_sprite((target[0], 1.4, target[1]), (4.0, 4.0),
                                     (1.0, 0.68, 0.15, 0.20 + 0.12 * pulse))
                renderer.push_sprite((target[0], 3.6, target[1]), (1.2, 1.2),
                                     (1.0, 0.72, 0.20, 0.85))
                renderer.push_sprite((target[0], 6.0 + pulse * 0.6, target[1]), (0.7, 0.7),
                                     (1.0, 0.85, 0.45, 0.9))
        # ---- particles (tyre smoke, spray, dust, debris) ------------------------------
        self.particles.push_to(renderer, self.camera.position)
        # ---- rain ---------------------------------------------------------------------
        rain = float(lighting.get("rain", 0.0))
        if rain > 0.02:
            renderer.push_rain(camera, rain, dt)
        del ambient

    def _push_vehicle_lamps(self, model, vehicle, distance: float, night: float,
                            blink: bool, emergency: bool = False) -> None:
        renderer = self.renderer
        anchors = model.light_anchors
        fx, fz = vehicle.forward()
        rx, rz = vehicle.right()
        fog = max(0.15, 1.0 - distance / 130.0)

        def world(anchor, dy: float = 0.0):
            return (vehicle.x + fx * anchor[0] + rx * anchor[2],
                    anchor[1] + dy,
                    vehicle.z + fz * anchor[0] + rz * anchor[2])

        head_on = vehicle.headlights or night > 0.45
        if head_on:
            for anchor in anchors.get("head", []):
                renderer.push_sprite(world(anchor), (0.75, 0.75),
                                     (1.0, 0.96, 0.84, 0.95 * fog))
                renderer.push_sprite(world(anchor), (2.6, 2.4),
                                     (1.0, 0.94, 0.80, 0.16 * fog * max(night, 0.4)))
        if vehicle.brake_light > 0.25:
            for anchor in anchors.get("tail", []):
                renderer.push_sprite(world(anchor), (0.55, 0.55),
                                     (1.0, 0.16, 0.10, 0.9 * fog * min(1.0, vehicle.brake_light)))
        elif head_on:
            for anchor in anchors.get("tail", []):
                renderer.push_sprite(world(anchor), (0.30, 0.30), (0.9, 0.12, 0.08, 0.35 * fog))
        if vehicle.hazard_lights or (blink and vehicle.indicator):
            anchors_ind = (anchors.get("indicator_front", []) + anchors.get("indicator_rear", []))
            for anchor in anchors_ind:
                renderer.push_sprite(world(anchor), (0.42, 0.42), (1.0, 0.62, 0.10, 0.95 * fog))
        if emergency:
            flash = int(time.time() * 6.0) % 2 == 0
            for index, anchor in enumerate(anchors.get("beacon", [])):
                colour = (1.0, 0.12, 0.10, 0.9) if (flash == (index % 2 == 0)) else (0.15, 0.35, 1.0, 0.9)
                renderer.push_sprite(world(anchor, 0.05), (0.9, 0.9),
                                     tuple(c * fog for c in colour[:3]) + (colour[3] * fog,))
                renderer.push_sprite(world(anchor, 0.05), (2.6, 2.6),
                                     (colour[0] * 0.5, colour[1] * 0.5, colour[2] * 0.5,
                                      0.22 * fog))
        elif vehicle.hazards if hasattr(vehicle, "hazards") else False:
            pass

    # ------------------------------------------------------------------ UI overlays
    def _hud_snapshot(self) -> HUDSnapshot:
        player = self.player
        assert player is not None
        runtime = self.runtime
        signal_state, junction, distance = self.world.signals.state_at(player.x, player.z,
                                                                      player.yaw)
        minimap = {
            "x": player.x, "z": player.z, "yaw": player.yaw,
            "range": 280.0,
            "roads": self.minimap_roads,
            "parking": [(lot.x, lot.z) for lot in self.world.parking_lots.values()],
            "traffic": [(v.x, v.z) for v in self.world.traffic.nearby(player.x, player.z, 200.0)][:70],
            "destination": runtime.destination() if runtime else None,
        }
        objective = "Free Drive"
        detail = ""
        if runtime is not None:
            objective = runtime.objective_text
            detail = runtime.mission.detail
        return HUDSnapshot(
            speed_kmh=player.speed_kmh,
            gear="R" if player.vehicle.speed < -0.2 else "D",
            fuel_fraction=player.vehicle.fuel_fraction(),
            damage=player.vehicle.damage,
            max_speed=player.vehicle.spec.max_speed,
            car_name=player.vehicle.spec.name,
            money=self.save.money,
            score=int(player.score),
            objective=objective,
            objective_detail=detail,
            destination_distance=runtime.distance_to_destination(player.vehicle) if runtime else -1.0,
            time_left=runtime.time_left if runtime else -1.0,
            signal_state=signal_state,
            signal_distance=distance,
            rule=player.active_rule,
            warnings=[(text, seconds) for text, seconds in
                      self._active_warnings(player)],
            toasts=list(self.toasts),
            collisions=player.collisions,
            violations=player.violations,
            combo=player.combo,
            zone=self.world.zone,
            time_of_day=self.world.environment.time_preset.upper(),
            weather=self.world.environment.weather.upper(),
            traffic_count=len(self.world.traffic.vehicles),
            traffic_level=self.world.traffic.level,
            fps=self.fps,
            show_fps=self.settings.show_fps,
            show_minimap=self.settings.show_minimap,
            minimap=minimap,
            emergency=bool(self.world.traffic.emergency_near(player.x, player.z, 60.0)),
            direction_hint=runtime.direction_hint(player.vehicle) if runtime else "",
            parking_hint=runtime.park_message if runtime else "",
            mission_progress=runtime.progress() if runtime else -1.0,
            indicator=0 if player.vehicle.hazard_lights else player.vehicle.indicator,
            headlights=player.vehicle.headlights,
            wrong_way=player.active_rule == "KEEP LEFT" and player.wrong_side_cooldown > 0.0,
        )

    def _active_warnings(self, player) -> List[Tuple[str, float]]:
        out = []
        for hazard in self.world.network.hazards_near(player.x, player.z, 26.0):
            pass
        return out

    def _draw_title(self, surface: pygame.Surface) -> None:
        """Bottom branding strip for the main menu (the panel owns the top of the screen)."""
        width, height = surface.get_size()
        strip = pygame.Surface((width, 56), pygame.SRCALPHA)
        strip.fill((6, 8, 12, 170))
        surface.blit(strip, (0, height - 56))
        font = pygame.font.Font(None, 24)
        small = pygame.font.Font(None, 20)
        title = font.render(S.APP_TITLE, True, (255, 176, 48))
        surface.blit(title, (20, height - 46))
        info = ("OpenGL " + str(self.gl_info.get("version", ""))[:26]
                + "   |   " + str(self.gl_info.get("renderer", ""))[:34]
                + "   |   WASD or arrows to drive, Esc for the menu")
        surface.blit(small.render(info, True, (150, 158, 170)), (20, height - 24))
        del font

    def _draw_menu_backdrop(self, surface: pygame.Surface) -> None:
        shade = pygame.Surface(surface.get_size(), pygame.SRCALPHA)
        shade.fill((5, 7, 11, 130))
        surface.blit(shade, (0, 0))

    def _draw_briefing(self, surface: pygame.Surface) -> None:
        mission = self.pending_mission
        if mission is None:
            return
        width, height = surface.get_size()
        panel = pygame.Rect(int(width * 0.16), int(height * 0.18),
                            int(width * 0.68), int(height * 0.62))
        from .hud import _panel
        _panel(surface, panel, radius=18, color=(10, 13, 19, 236), border=(70, 78, 96, 240))
        font_title = pygame.font.Font(None, 46)
        font = pygame.font.Font(None, 28)
        small = pygame.font.Font(None, 22)
        surface.blit(font_title.render(mission.title, True, (255, 176, 48)),
                     (panel.x + 28, panel.y + 24))
        surface.blit(font.render(mission.objective, True, (232, 236, 244)),
                     (panel.x + 30, panel.y + 86))
        y = panel.y + 128
        for line in _wrap_text(mission.detail, small, panel.width - 60):
            surface.blit(small.render(line, True, (176, 184, 196)), (panel.x + 30, y))
            y += 24
        y += 10
        info = [f"Reward: Rs {mission.reward:,}"]
        if mission.time_limit > 0:
            info.append(f"Time limit: {mission.time_limit:.0f} s")
        if mission.traffic_level:
            info.append(f"Traffic: {mission.traffic_level}")
        if mission.time_of_day:
            info.append(f"Time: {mission.time_of_day}")
        if mission.weather:
            info.append(f"Weather: {mission.weather}")
        surface.blit(font.render("   |   ".join(info), True, (110, 210, 255)),
                     (panel.x + 30, y))
        y += 40
        if mission.tip:
            for line in _wrap_text("Tip: " + mission.tip, small, panel.width - 60):
                surface.blit(small.render(line, True, (255, 200, 96)), (panel.x + 30, y))
                y += 22
        footer = small.render("Enter: start driving      Esc: back to the menu",
                              True, (150, 158, 170))
        surface.blit(footer, (panel.x + 30, panel.bottom - 40))

    def _draw_results(self, surface: pygame.Surface) -> None:
        self._draw_menu_backdrop(surface)
        result = self.last_result
        if result is None:
            return
        if self.results_menu is not None:
            self.results_menu.render(surface)
        player = self.player
        if player is None:
            return
        summary = player.summary()
        width, height = surface.get_size()
        font = pygame.font.Font(None, 26)
        panel = pygame.Rect(int(width * 0.12), int(height * 0.60), int(width * 0.76), 150)
        from .hud import _panel
        _panel(surface, panel, radius=14, color=(10, 13, 19, 224), border=(70, 78, 96, 240))
        lines = [
            f"Distance driven: {summary['distance_km']:.2f} km",
            f"Time: {summary['time_s']:.0f} s     Top speed: {summary['top_speed']:.0f} km/h",
            f"Collisions: {summary['collisions']}     Violations: {summary['violations']}"
            f"     Potholes hit: {summary['potholes']}",
            f"Mission score: {result.score}     Reward: Rs {result.reward:,}",
        ]
        y = panel.y + 18
        for line in lines:
            surface.blit(font.render(line, True, (208, 214, 224)), (panel.x + 24, y))
            y += 30


def _wrap_text(text: str, font: pygame.font.Font, max_width: int) -> List[str]:
    words = text.split()
    lines: List[str] = []
    current = ""
    for word in words:
        candidate = f"{current} {word}".strip()
        if font.size(candidate)[0] <= max_width:
            current = candidate
        else:
            if current:
                lines.append(current)
            current = word
    if current:
        lines.append(current)
    return lines or [""]
