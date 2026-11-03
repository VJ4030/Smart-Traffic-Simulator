"""Settings: defaults, validation, cycling, save paths."""

from __future__ import annotations

import json
import os

import pytest

from smart_traffic_india import settings as S


def test_app_metadata():
    assert S.APP_TITLE.startswith("Smart Traffic India")
    assert S.VERSION == S.APP_VERSION
    assert S.python_version_ok() is True


def test_fixed_timestep_is_sane():
    assert abs(S.FIXED_DT - 1.0 / 60.0) < 1e-9
    assert S.MAX_STEPS_PER_FRAME >= 2
    assert S.MAX_FRAME_TIME >= 4 * S.FIXED_DT


def test_quality_presets_are_complete():
    for name in S.QUALITY_ORDER:
        preset = S.QUALITY_PRESETS[name]
        for key in ("shadows", "rain_particles", "far_plane", "traffic_scale",
                    "pedestrian_scale", "fog_scale", "glow_scale", "max_sign_textures",
                    "multisample"):
            assert key in preset, f"{name} is missing {key}"
        assert preset["far_plane"] > 100.0
    # a stronger preset must never be cheaper than a weaker one
    assert (S.QUALITY_PRESETS["high"]["far_plane"]
            >= S.QUALITY_PRESETS["low"]["far_plane"])
    assert (S.QUALITY_PRESETS["ultra"]["traffic_scale"]
            >= S.QUALITY_PRESETS["low"]["traffic_scale"])


def test_traffic_levels_are_ordered():
    maxt = [S.TRAFFIC_LEVELS[name]["max_vehicles"] for name in S.TRAFFIC_LEVEL_ORDER]
    assert maxt == sorted(maxt)
    assert S.TRAFFIC_LEVELS["busy"]["max_vehicles"] > 40


def test_default_profile_shape():
    profile = S.default_profile()
    for key in ("money", "unlocked", "owned", "upgrades", "paints", "damage", "last_car",
                "completed_missions", "best_parking", "school_lessons", "settings"):
        assert key in profile
    assert profile["money"] == S.START_MONEY
    assert "hatchback" in profile["owned"]
    assert profile["stats"]["distance_km"] == 0.0


def test_settings_sanitise_clamps_bad_values():
    value = S.Settings(width=-50, height=99999, quality="ultra-mega", traffic_level="insane",
                       time_of_day="dawn", weather="hail", camera_mode="zeppelin",
                       master_volume=4.0)
    value.sanitise()
    assert value.width >= 800
    assert value.height >= 480
    assert value.quality in S.QUALITY_PRESETS
    assert value.traffic_level in S.TRAFFIC_LEVELS
    assert value.time_of_day in value.option_list("time_of_day")
    assert value.weather in value.option_list("weather")
    assert value.camera_mode in value.option_list("camera_mode")
    assert value.master_volume <= 1.0


def test_settings_cycle_wraps_around():
    value = S.Settings()
    seen = set()
    for _ in range(len(value.option_list("camera_mode")) + 1):
        value.cycle_option("camera_mode")
        seen.add(value.camera_mode)
    assert len(seen) == len(value.option_list("camera_mode"))
    value.cycle_option("camera_mode", -1)
    assert value.camera_mode in value.option_list("camera_mode")


def test_settings_round_trip():
    value = S.Settings(quality="medium", traffic_level="heavy", show_minimap=False)
    restored = S.Settings.from_dict(value.to_dict())
    assert restored.quality == "medium"
    assert restored.traffic_level == "heavy"
    assert restored.show_minimap is False
    # unknown keys must be ignored instead of exploding
    assert S.Settings.from_dict({"nonsense": 1, "quality": "low"}).quality == "low"
    assert S.Settings.from_dict(None).quality in S.QUALITY_PRESETS


def test_quality_flags_returns_a_copy():
    value = S.Settings()
    flags = value.quality_flags
    flags["far_plane"] = -1
    assert value.quality_flags["far_plane"] > 0


def test_save_paths_honour_data_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("STI_DATA_DIR", str(tmp_path / "sti"))
    directory = S.user_data_dir()
    assert os.path.isdir(directory)
    assert S.save_file_path().startswith(str(tmp_path))
    assert S.config_file_path().endswith("config.json")


def test_palette_and_paints():
    assert len(S.P.PAINTS) >= 8
    assert len(S.P.TRAFFIC_PAINT) >= 6
    for colour in list(S.P.PAINTS.values()) + S.P.TRAFFIC_PAINT:
        assert len(colour) == 3
        assert all(0.0 <= channel <= 1.0 for channel in colour)


def test_default_bindings_cover_the_controls():
    for action in ("throttle", "brake", "steer_left", "steer_right", "handbrake", "horn",
                   "lights", "camera", "indicator_left", "indicator_right", "reset",
                   "pause", "refuel", "map"):
        assert action in S.DEFAULT_BINDINGS
        assert S.DEFAULT_BINDINGS[action]
