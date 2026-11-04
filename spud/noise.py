"""Sensor noise models, so policies trained in SPUD don't meet a perfect world only in simulation.

Each model is a small dataclass with defaults in the range of common hardware. Turn noise on per
sensor in YAML with `noise: true` (defaults), `noise: false`, or a dict overriding any field:

    - {type: lidar, robot: zoomie0, noise: {range_std: 0.02, dropout: 0.05}}

Numbers are "typical", not a calibration of any specific product: measure yours and override.
"""

from __future__ import annotations

from dataclasses import dataclass, fields

import numpy as np


@dataclass
class LidarNoise:
    range_std: float = 0.01          # m, at zero range
    range_std_per_m: float = 0.005   # extra std per meter of range (0.5%)
    quantization: float = 0.005      # m
    dropout: float = 0.01            # random missing returns
    grazing_dropout: float = 0.6     # extra dropout probability as incidence approaches 90 deg
    min_range: float = 0.05          # returns closer than this are invalid (inside the housing)

    def apply(self, rng, dist, normals, dirs):
        """dist (N,) with inf for no hit; normals (N,3); dirs (N,3) unit ray directions."""
        d = dist.copy()
        hit = np.isfinite(d)
        cos = np.abs(np.einsum("ij,ij->i", normals, dirs))  # 1 = head-on, 0 = grazing
        std = self.range_std + self.range_std_per_m * np.where(hit, d, 0)
        d[hit] += rng.normal(0, std[hit])
        if self.quantization > 0:
            d[hit] = np.round(d[hit] / self.quantization) * self.quantization
        p_drop = self.dropout + self.grazing_dropout * (1 - cos) ** 4
        lost = hit & (rng.random(len(d)) < p_drop)
        d[lost | (d < self.min_range)] = np.inf
        # Intensity: Lambertian return falling with range squared, normalized to [0, 1] at 1 m head-on.
        intensity = np.where(np.isfinite(d), np.clip(cos / np.maximum(d, 0.1) ** 2, 0, 1), 0.0)
        return d, intensity


@dataclass
class DepthNoise:
    # Kinect-like axial noise: sigma(z) = a + b * (z - z0)^2, after Nguyen, Izadi & Lovell (3DIMPVT 2012).
    a: float = 0.0012
    b: float = 0.0019
    z0: float = 0.4
    min_depth: float = 0.2
    max_depth: float = 8.0
    edge_holes: float = 0.5          # probability of a hole at a depth discontinuity
    quantization: float = 0.001

    def apply(self, rng, depth):
        z = depth.astype(np.float64)
        z = z + rng.normal(0, 1, z.shape) * (self.a + self.b * (z - self.z0) ** 2)
        if self.quantization > 0:
            z = np.round(z / self.quantization) * self.quantization
        gy, gx = np.gradient(depth)
        edges = np.hypot(gx, gy) > 0.05
        holes = edges & (rng.random(z.shape) < self.edge_holes)
        z[holes | (depth < self.min_depth) | (depth > self.max_depth)] = 0.0  # 0 = invalid, like real sensors
        return z.astype(np.float32)


@dataclass
class CameraNoise:
    shot: float = 0.02      # photon noise scale: std = shot * sqrt(signal), signal in [0, 1]
    read: float = 0.01      # sensor read noise std
    gain: float = 1.0       # exposure multiplier

    def apply(self, rng, rgb):
        x = rgb.astype(np.float32) / 255.0 * self.gain
        x = x + rng.normal(0, 1, x.shape) * np.sqrt(self.shot**2 * np.clip(x, 0, None) + self.read**2)
        return (np.clip(x, 0, 1) * 255 + 0.5).astype(np.uint8)


@dataclass
class ImuNoise:
    # Continuous-time densities as on MEMS IMU datasheets; dt is the read period.
    gyro_noise: float = 0.004        # rad/s/sqrt(Hz)
    gyro_bias_walk: float = 0.0002   # rad/s^2/sqrt(Hz)
    accel_noise: float = 0.03        # m/s^2/sqrt(Hz)
    accel_bias_walk: float = 0.002   # m/s^3/sqrt(Hz)
    dt: float = 0.02

    def __post_init__(self):
        self.gyro_bias = np.zeros(3)
        self.accel_bias = np.zeros(3)

    def apply(self, rng, gyro, accel):
        s = np.sqrt(self.dt)
        self.gyro_bias += rng.normal(0, self.gyro_bias_walk * s, 3)
        self.accel_bias += rng.normal(0, self.accel_bias_walk * s, 3)
        g = gyro + self.gyro_bias + rng.normal(0, self.gyro_noise / s, 3)
        a = accel + self.accel_bias + rng.normal(0, self.accel_noise / s, 3)
        return g, a


def make(model_cls, cfg):
    """`noise:` YAML value -> a model instance or None."""
    if cfg in (None, False):
        return None
    if cfg is True:
        return model_cls()
    names = {f.name for f in fields(model_cls)}
    unknown = set(cfg) - names
    if unknown:
        raise ValueError(f"unknown {model_cls.__name__} fields {sorted(unknown)}; known: {sorted(names)}")
    return model_cls(**cfg)
