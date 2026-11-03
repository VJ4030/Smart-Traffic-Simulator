"""Entry point / launcher.

``python main.py`` (or ``python -m smart_traffic_india``) ends up here.  The launcher is
deliberately defensive: it checks the Python and dependency versions, prints a readable
diagnostic when the OpenGL context cannot be created, and falls back from a GL 3.3 core
context to a compatibility context with GLSL 1.20 shaders on older drivers.

Command line
------------
``--quality low|medium|high``   override the quality preset
``--width N --height N``        window size
``--windowed``                  force a window (default is windowed anyway, ``--fullscreen`` inverts)
``--free``                      jump straight into Free Drive
``--mission <id>``              jump straight into a career/parking/school mission
``--traffic light|medium|heavy``  override traffic density
``--time morning|day|evening|night``  override the time of day
``--weather clear|cloudy|rain``  override the weather
``--frames N``                  render N frames and quit (smoke test / screenshot tool)
``--screenshot PATH``           save a PNG after the smoke run
``--list-missions``             print every mission id and exit
"""

from __future__ import annotations

import argparse
import sys
import time
import traceback

from . import settings as S


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="SmartTrafficIndia",
                                     description=f"{S.APP_TITLE} - {S.APP_TAGLINE}")
    parser.add_argument("--quality", choices=sorted(S.QUALITY_PRESETS), default=None)
    parser.add_argument("--width", type=int, default=None)
    parser.add_argument("--height", type=int, default=None)
    parser.add_argument("--windowed", action="store_true")
    parser.add_argument("--fullscreen", action="store_true")
    parser.add_argument("--free", action="store_true", help="start in Free Drive")
    parser.add_argument("--mission", default=None, help="start a specific mission id")
    parser.add_argument("--traffic", choices=S.TRAFFIC_LEVELS, default=None)
    parser.add_argument("--time", dest="time_of_day", choices=S.TIME_PRESETS, default=None)
    parser.add_argument("--weather", choices=S.WEATHER_PRESETS, default=None)
    parser.add_argument("--frames", type=int, default=0, help="render N frames then exit")
    parser.add_argument("--screenshot", default=None, help="PNG path for a smoke screenshot")
    parser.add_argument("--list-missions", action="store_true")
    parser.add_argument("--version", action="store_true")
    return parser


def _print_environment() -> None:
    print(f"{S.APP_TITLE} {S.APP_VERSION}")
    print(f"Python {sys.version.split()[0]} on {sys.platform}")
    try:
        import pygame
        print(f"pygame {pygame.version.ver}")
    except Exception as exc:                              # pragma: no cover
        print(f"pygame missing: {exc}")
    try:
        import OpenGL
        print(f"PyOpenGL {OpenGL.__version__}")
    except Exception as exc:                              # pragma: no cover
        print(f"PyOpenGL missing: {exc}")
    try:
        import numpy
        print(f"numpy {numpy.__version__}")
    except Exception as exc:                              # pragma: no cover
        print(f"numpy missing: {exc}")
    print(f"Save file: {S.save_file_path()}")


def _check_dependencies() -> None:
    missing = []
    for module, package in (("pygame", "pygame"), ("OpenGL", "PyOpenGL"), ("numpy", "numpy")):
        try:
            __import__(module)
        except ImportError:
            missing.append(package)
    if missing:
        print("Missing dependencies: " + ", ".join(missing))
        print("Install them with:  python -m pip install -r requirements.txt")
        raise SystemExit(2)


def _list_missions() -> None:
    from .missions import build_career_missions, build_parking_challenges, build_school_lessons
    for label, builder in (("career", build_career_missions),
                           ("parking", build_parking_challenges),
                           ("school", build_school_lessons)):
        print(f"\n[{label}]")
        for mission in builder():
            timer = f"  ({mission.time_limit:.0f}s)" if mission.time_limit else ""
            print(f"  {mission.id:<18} {mission.title:<34} Rs {mission.reward:,}{timer}")
            print(f"      {mission.objective}")
    print("\nStart one with:  python main.py --mission m03_school_run")


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    if args.version:
        _print_environment()
        return 0
    if args.list_missions:
        _list_missions()
        return 0

    _check_dependencies()
    from .save_manager import SaveManager
    from .game import Game

    save = SaveManager()
    settings = save.load_settings()
    if args.quality:
        settings.quality = args.quality
    if args.width:
        settings.width = max(640, args.width)
    if args.height:
        settings.height = max(360, args.height)
    if args.fullscreen:
        settings.fullscreen = True
    if args.windowed:
        settings.fullscreen = False
    if args.traffic:
        settings.traffic_level = args.traffic
    if args.time_of_day:
        settings.time_of_day = args.time_of_day
    if args.weather:
        settings.weather = args.weather

    try:
        game = Game(settings)
    except Exception:                                      # pragma: no cover - driver issues
        print("\nCould not start the game.\n")
        traceback.print_exc()
        print(
            "\nCommon causes:\n"
            "  * No OpenGL driver / software renderer installed. Update your GPU driver\n"
            "    (Intel, NVIDIA, AMD) and confirm 'OpenGL 3.3' support with a tool such as\n"
            "    GPU Caps Viewer.\n"
            "  * Running inside a remote/virtual machine without 3D acceleration: install\n"
            "    Mesa ('apt install libgl1-mesa-dri') or run with a physical display.\n"
            "  * Missing Python packages: python -m pip install -r requirements.txt\n"
        )
        return 1

    if args.free:
        game._start_free_drive()
    elif args.mission:
        from .missions import (build_career_missions, build_parking_challenges,
                               build_school_lessons)
        wanted = args.mission
        for builder in (build_career_missions, build_parking_challenges, build_school_lessons):
            for mission in builder():
                if mission.id == wanted or mission.id.startswith(wanted):
                    game._start_mission(mission)
                    break
            else:
                continue
            break
        else:
            print(f"Unknown mission id '{wanted}'. Use --list-missions to see them all.")
            game.shutdown()
            return 2

    if args.frames:
        return _smoke_run(game, args)
    try:
        game.run()
    except KeyboardInterrupt:                              # pragma: no cover
        print("\nInterrupted - saving and quitting.")
        game.persist()
        game.shutdown()
    return 0


def _smoke_run(game, args) -> int:
    """Render a limited number of frames - used by tools/render_smoke.py and CI."""
    import numpy as np
    import pygame

    from . import settings as S
    dt = S.FIXED_DT
    frames = max(1, args.frames)
    rendered = 0
    start = time.perf_counter()
    for index in range(frames):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                frames = index
                break
        if game.state == "driving":
            game._step(dt)
        else:
            game._idle_step(dt)
        game._render(dt)
        game.input.end_frame()
        rendered += 1
    elapsed = time.perf_counter() - start
    stats = game.renderer.stats
    print(f"{S.APP_TITLE} {S.APP_VERSION} smoke run")
    print(f"  frames      : {rendered}  ({elapsed / max(rendered, 1) * 1000:.1f} ms/frame)"
          " - software renderer in this sandbox, not your laptop")
    print(f"  draw calls  : {stats.get('draw_calls')} (last frame)")
    print(f"  triangles   : {int(stats.get('triangles', 0))} (last frame)")
    print(f"  chunks      : {int(stats.get('chunks_drawn', 0))} drawn, "
          f"{int(stats.get('chunks_culled', 0))} culled")
    print(f"  sprites     : {int(stats.get('sprites', 0))} instanced billboards")
    print(f"  GL          : {game.gl_info.get('version')} | {game.gl_info.get('renderer')}")
    print(f"  world       : {game.world.statistics()['traffic']}")
    if args.screenshot:
        pixels = game.renderer.read_pixels()
        surface = pygame.image.frombuffer(np.ascontiguousarray(pixels).tobytes(),
                                          (pixels.shape[1], pixels.shape[0]), "RGB")
        pygame.image.save(surface, args.screenshot)
        print(f"  screenshot  : {args.screenshot}")
    game.persist()
    game.shutdown()
    return 0


if __name__ == "__main__":                                 # pragma: no cover
    raise SystemExit(main())
