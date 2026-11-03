"""Audio: engine, horn, siren, impact and indicator sounds generated from numpy waves.

There are **no sound files** in the repository.  Every sample is synthesised at start-up
into a :class:`pygame.mixer.Sound`:

* ``engine``  - 10 short loops recorded at different engine speeds; the game picks the
  loop closest to the current rpm and rides its volume, which gives a convincing
  "gear shift" without a real audio engine
* ``horn``    - two-tone Indian horn
* ``crash``   - filtered noise burst
* ``tick``    - indicator relay click
* ``siren``   - looping two-tone ambulance siren
* ``skid``    - noise sweep while the tyres slide

The module degrades gracefully: if the mixer is unavailable (head-less CI, broken audio
driver) every method becomes a no-op and :attr:`AudioEngine.available` reports ``False``.
"""

from __future__ import annotations

import math
from typing import Dict, List

import numpy as np

try:                                                    # pragma: no cover - platform
    import pygame
    from pygame import mixer
except Exception:                                       # pragma: no cover
    pygame = None                                       # type: ignore[assignment]
    mixer = None                                        # type: ignore[assignment]

SAMPLE_RATE = 22050
ENGINE_STEPS = 10


def _to_sound(samples: np.ndarray, volume: float = 1.0) -> "mixer.Sound":
    """Convert a float array in -1..1 into a stereo int16 pygame Sound."""
    data = np.clip(samples * volume, -1.0, 1.0)
    stereo = np.repeat((data * 32767.0).astype(np.int16)[:, None], 2, axis=1)
    return mixer.Sound(buffer=np.ascontiguousarray(stereo))


def _engine_loop(frequency: float, duration: float = 0.35) -> np.ndarray:
    """A small-band engine note: pulse train + two harmonics + noise, loop friendly."""
    t = np.linspace(0.0, duration, int(SAMPLE_RATE * duration), endpoint=False)
    # wrap the waveform so the loop does not click
    phase = 2.0 * math.pi * frequency * t
    wave = (np.sin(phase) * 0.55
            + np.sin(2.0 * phase) * 0.25
            + np.sin(3.0 * phase) * 0.12
            + np.sin(0.5 * phase) * 0.14)
    # exhaust "putter": amplitude modulation at half the firing frequency
    wave *= 0.75 + 0.25 * np.sin(2.0 * math.pi * frequency * 0.5 * t)
    rng = np.random.default_rng(7)
    wave += rng.normal(0.0, 0.05, size=t.shape)
    fade = int(SAMPLE_RATE * 0.01)
    wave[:fade] *= np.linspace(0.0, 1.0, fade)
    wave[-fade:] *= np.linspace(1.0, 0.0, fade)
    return np.tanh(wave * 1.4) * 0.5


def _horn_wave(duration: float = 0.6) -> np.ndarray:
    t = np.linspace(0.0, duration, int(SAMPLE_RATE * duration), endpoint=False)
    wave = (np.sign(np.sin(2.0 * math.pi * 420.0 * t)) * 0.35
            + np.sign(np.sin(2.0 * math.pi * 330.0 * t)) * 0.35
            + np.sin(2.0 * math.pi * 840.0 * t) * 0.12)
    envelope = np.minimum(1.0, t * 40.0) * np.clip((duration - t) * 8.0, 0.0, 1.0)
    return wave * envelope * 0.5


def _crash_wave(duration: float = 0.45, seed: int = 3) -> np.ndarray:
    rng = np.random.default_rng(seed)
    t = np.linspace(0.0, duration, int(SAMPLE_RATE * duration), endpoint=False)
    noise = rng.normal(0.0, 1.0, size=t.shape)
    # cheap one-pole low-pass for a "metal on metal" thud
    filtered = np.zeros_like(noise)
    alpha = 0.35
    previous = 0.0
    for i in range(noise.size):
        previous = previous + alpha * (noise[i] - previous)
        filtered[i] = previous
    envelope = np.exp(-t * 11.0)
    return np.tanh(filtered * 3.2) * envelope * 0.8


def _tick_wave(duration: float = 0.05) -> np.ndarray:
    t = np.linspace(0.0, duration, int(SAMPLE_RATE * duration), endpoint=False)
    return np.sin(2.0 * math.pi * 1500.0 * t) * np.exp(-t * 90.0) * 0.5


def _siren_wave(duration: float = 1.6) -> np.ndarray:
    t = np.linspace(0.0, duration, int(SAMPLE_RATE * duration), endpoint=False)
    sweep = 660.0 + 220.0 * np.sin(2.0 * math.pi * 1.1 * t)
    phase = np.cumsum(2.0 * math.pi * sweep / SAMPLE_RATE)
    return (np.sin(phase) * 0.5 + np.sin(phase * 1.5) * 0.16) * 0.55


def _skid_wave(duration: float = 0.5, seed: int = 5) -> np.ndarray:
    rng = np.random.default_rng(seed)
    t = np.linspace(0.0, duration, int(SAMPLE_RATE * duration), endpoint=False)
    noise = rng.normal(0.0, 1.0, size=t.shape)
    modulated = noise * (0.4 + 0.6 * np.abs(np.sin(2.0 * math.pi * 22.0 * t)))
    return modulated * 0.35


class AudioEngine:
    """Small wrapper that owns the mixer channels and the synthesised samples."""

    def __init__(self, enabled: bool = True, master: float = 0.6) -> None:
        self.master = max(0.0, min(1.0, master))
        self.available = False
        self.enabled = enabled
        self.sounds: Dict[str, object] = {}
        self.engine_frequencies: List[float] = []
        self._engine_channel = None
        self._siren_channel = None
        self._skid_channel = None
        self._engine_index = -1
        self._horn_timer = 0.0
        self.failures = 0
        if not enabled or mixer is None:
            return
        try:
            current = mixer.get_init()
            if current and int(current[0]) != SAMPLE_RATE:
                # the samples are synthesised at SAMPLE_RATE; playing them on a 44.1 kHz
                # mixer would shift the pitch, so make the mixer match the samples.
                mixer.quit()
                current = None
            if not current:
                mixer.pre_init(SAMPLE_RATE, -16, 2, 512)
                mixer.init(SAMPLE_RATE, -16, 2, 512)
            self._build_samples()
            self.available = True
        except Exception:                                # pragma: no cover - audio driver
            self.available = False
            self.failures += 1

    def _build_samples(self) -> None:
        self.sounds["horn"] = _to_sound(_horn_wave(), 0.9)
        self.sounds["crash"] = _to_sound(_crash_wave(), 0.9)
        self.sounds["crash_soft"] = _to_sound(_crash_wave(0.25, 4), 0.7)
        self.sounds["tick"] = _to_sound(_tick_wave(), 0.8)
        self.sounds["siren"] = _to_sound(_siren_wave(), 0.45)
        self.sounds["skid"] = _to_sound(_skid_wave(), 0.4)
        engine: List[object] = []
        for index in range(ENGINE_STEPS):
            frequency = 26.0 + index * 7.5
            engine.append(_to_sound(_engine_loop(frequency), 0.5))
            self.engine_frequencies.append(frequency)
        self.sounds["engine"] = engine

    # ------------------------------------------------------------------ playback
    def set_master(self, volume: float) -> None:
        self.master = max(0.0, min(1.0, volume))
        if self.available:
            for sound in self.sounds.values():
                if isinstance(sound, list):
                    for item in sound:
                        item.set_volume(self.master)
                else:
                    sound.set_volume(self.master)
            if self._engine_channel is not None:
                self._engine_channel.set_volume(self.master * 0.5)

    def set_engine(self, rpm_fraction: float, load: float) -> None:
        """rpm_fraction 0..1, load 0..1 - used to pick the loop and set its volume."""
        if not self.available:
            return
        fraction = max(0.0, min(1.0, rpm_fraction))
        index = int(round(fraction * (ENGINE_STEPS - 1)))
        if index != self._engine_index:
            engine = self.sounds.get("engine")
            if isinstance(engine, list) and engine:
                if self._engine_channel is not None:
                    self._engine_channel.stop()
                self._engine_channel = engine[index].play(loops=-1)
                self._engine_index = index
        if self._engine_channel is not None:
            volume = (0.22 + 0.5 * load) * (0.5 + 0.5 * fraction)
            self._engine_channel.set_volume(self.master * min(0.9, volume))

    def stop_engine(self) -> None:
        if self._engine_channel is not None:
            self._engine_channel.stop()
            self._engine_channel = None
            self._engine_index = -1

    def horn(self) -> None:
        if self.available and self.master > 0.0:
            self.sounds["horn"].play()

    def crash(self, severity: float = 1.0) -> None:
        if not self.available:
            return
        key = "crash" if severity > 0.6 else "crash_soft"
        self.sounds[key].play()

    def tick(self) -> None:
        if self.available:
            self.sounds["tick"].play()

    def set_siren(self, active: bool, proximity: float = 1.0) -> None:
        if not self.available:
            return
        if active and self._siren_channel is None:
            self._siren_channel = self.sounds["siren"].play(loops=-1)
        if self._siren_channel is not None:
            if not active:
                self._siren_channel.stop()
                self._siren_channel = None
            else:
                self._siren_channel.set_volume(self.master * max(0.0, min(1.0, proximity)) * 0.7)

    def set_skid(self, active: bool) -> None:
        if not self.available:
            return
        if active and self._skid_channel is None:
            self._skid_channel = self.sounds["skid"].play(loops=-1)
        elif not active and self._skid_channel is not None:
            self._skid_channel.stop()
            self._skid_channel = None

    def update(self, dt: float) -> None:
        """Call once per frame; keeps the looping channels tidy."""
        self._horn_timer = max(0.0, self._horn_timer - dt)

    def update_from_vehicle(self, vehicle, throttle: float, wet: bool) -> None:
        spec = vehicle.spec
        top = max(1.0, spec.max_speed / 3.6)
        fraction = min(1.0, abs(vehicle.speed) / top)
        self.set_engine(fraction, throttle)
        slipping = abs(vehicle.lateral_accel) > 2.4 and abs(vehicle.speed) > 4.0
        self.set_skid(slipping or (wet and abs(vehicle.speed) > 14.0))

    def shutdown(self) -> None:
        self.stop_engine()
        self.set_siren(False)
        self.set_skid(False)

    def statistics(self) -> dict:
        return {"available": self.available, "enabled": self.enabled,
                "master": self.master, "sounds": len(self.sounds)}
