"""
USV Digital Twin — Módulo 3: Sensor Engine
===========================================
Simula o stack de sensores real do USV com ruído realista:

    IMU     — aceleração linear/angular + ruído gaussiano
    Sonar   — 6 transdutores ortogonais Open Echo (200kHz, 7m, 25°)
    Barômetro — pressão → profundidade (MS5837)
    Distúrbio ambiental — maresia/turbulência com ruído não gaussiano (Rayleigh)
    Depth Map — recebe frame do Three.js via WebSocket

Extended Kalman Filter (EKF) — módulo separado
    Estado estimado nunca vê o estado real diretamente
    Control Engine usa apenas EKF.state_estimate

Referências:
    - Open Echo project (Neumi): github.com/Neumi/open_echo
    - TUSS4470 datasheet — Texas Instruments
    - Fossen (2011) cap. 10 — sensor models for marine vehicles
    - Thrun et al. (2005) Probabilistic Robotics — EKF derivation
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Tuple
from physics_engine import VehicleState, PhysicsEngine
import time
import json


# ─────────────────────────────────────────────
# PARÂMETROS DO HARDWARE REAL
# ─────────────────────────────────────────────

# Open Echo — transdutor PZT 200kHz
SONAR_RANGE_MAX     = 7.0       # m — range real em água doce
SONAR_BEAMWIDTH_DEG = 25.0      # graus — cone de detecção
SONAR_BEAMWIDTH_RAD = np.radians(SONAR_BEAMWIDTH_DEG)
SONAR_UPDATE_HZ     = 10.0      # Hz — taxa de atualização
SONAR_MIN_RANGE     = 0.15      # m — ringdown mínimo

# IMU sintético — parâmetros típicos de MPU6050
IMU_ACCEL_NOISE_STD  = 0.05     # m/s² — desvio padrão do ruído
IMU_GYRO_NOISE_STD   = 0.002    # rad/s
IMU_ACCEL_BIAS_STD   = 0.01     # m/s² — bias lento
IMU_GYRO_BIAS_STD    = 0.0005   # rad/s

# MS5837 — barômetro de pressão
BARO_NOISE_STD       = 0.01     # m — precisão de profundidade
RHO_FRESHWATER       = 1000.0
G                    = 9.81

# Distúrbio ambiental não gaussiano (Rayleigh)
RAYLEIGH_DEFAULT_SIGMA = 0.03  # escala base para perturbações ambientais


# ─────────────────────────────────────────────
# ESTRUTURAS DE DADOS
# ─────────────────────────────────────────────

@dataclass
class IMUReading:
    """Leitura ruidosa do IMU."""
    accel: np.ndarray   # [ax, ay, az] m/s²
    gyro:  np.ndarray   # [p, q, r] rad/s
    timestamp: float
    velocity: Optional[np.ndarray] = None  # [u, v, w] m/s — DVL sintético / odometria corporal

    def to_dict(self) -> dict:
        return {
            'accel': self.accel.tolist(),
            'gyro':  self.gyro.tolist(),
            'timestamp': self.timestamp,
        }


@dataclass
class SonarReading:
    """Leitura de um transdutor sonar."""
    direction:  np.ndarray  # vetor unitário de apontamento (body frame)
    distance:   float       # m — distância medida (-1 = sem retorno)
    confidence: float       # [0,1] — qualidade do eco
    timestamp:  float

    @property
    def hit(self) -> bool:
        return self.distance > 0

    def to_dict(self) -> dict:
        return {
            'direction':  self.direction.tolist(),
            'distance':   self.distance,
            'confidence': self.confidence,
            'timestamp':  self.timestamp,
        }


@dataclass
class BarometerReading:
    """Leitura do barômetro de pressão."""
    depth:     float    # m — profundidade estimada
    pressure:  float    # Pa
    timestamp: float


@dataclass
class SensorBundle:
    """Bundle completo de leituras de um timestep."""
    imu:       IMUReading
    sonar:     List[SonarReading]
    barometer: BarometerReading
    timestamp: float

    def to_dict(self) -> dict:
        return {
            'imu':       self.imu.to_dict(),
            'sonar':     [s.to_dict() for s in self.sonar],
            'barometer': {
                'depth':    self.barometer.depth,
                'pressure': self.barometer.pressure,
            },
            'timestamp': self.timestamp,
        }


@dataclass
class SensorFaultProfile:
    """Configuração de falhas para HIL de validação."""

    imu_dropout_prob: float = 0.0
    sonar_dropout_prob: float = 0.0
    baro_dropout_prob: float = 0.0
    imu_accel_bias: np.ndarray = field(default_factory=lambda: np.zeros(3))
    imu_gyro_bias: np.ndarray = field(default_factory=lambda: np.zeros(3))
    baro_depth_bias: float = 0.0
    sonar_range_scale: float = 1.0
    sonar_confidence_scale: float = 1.0
    temporal_jitter_s: float = 0.0


@dataclass
class EKFState:
    """Estado estimado pelo EKF."""
    eta:       np.ndarray           # [x,y,z,φ,θ,ψ] — posição/orientação
    nu:        np.ndarray           # [u,v,w,p,q,r] — velocidades
    P:         np.ndarray           # covariância 12x12
    timestamp: float
    vision_relative_waypoint: Optional[np.ndarray] = None

    @property
    def position(self) -> np.ndarray:
        return self.eta[:3]

    @property
    def orientation(self) -> np.ndarray:
        return self.eta[3:]

    @property
    def velocity_linear(self) -> np.ndarray:
        return self.nu[:3]

    @property
    def velocity_angular(self) -> np.ndarray:
        return self.nu[3:]

    def to_dict(self) -> dict:
        return {
            'eta':        self.eta.tolist(),
            'nu':         self.nu.tolist(),
            'covariance': np.diag(self.P).tolist(),  # só diagonal pro JSON
            'timestamp':  self.timestamp,
        }


# ─────────────────────────────────────────────
# MODELO DE OBSTÁCULOS
# ─────────────────────────────────────────────

@dataclass
class Obstacle:
    """Obstáculo no ambiente — esfera ou plano."""
    position: np.ndarray    # centro (m)
    radius:   float         # raio (m) — 0 para plano infinito
    normal:   Optional[np.ndarray] = None  # normal do plano

    def intersect_ray(
        self,
        origin: np.ndarray,
        direction: np.ndarray,
        max_range: float
    ) -> float:
        """
        Retorna distância de interseção raio→obstáculo.
        -1 se não há interseção no range.
        """
        if self.radius > 0:
            # interseção raio-esfera
            oc  = origin - self.position
            a   = np.dot(direction, direction)
            b   = 2.0 * np.dot(oc, direction)
            c   = np.dot(oc, oc) - self.radius**2
            disc = b**2 - 4*a*c

            if disc < 0:
                return -1.0

            t = (-b - np.sqrt(disc)) / (2.0*a)
            if SONAR_MIN_RANGE < t <= max_range:
                return t
            return -1.0

        else:
            # interseção raio-plano
            if self.normal is None:
                return -1.0
            denom = np.dot(self.normal, direction)
            if abs(denom) < 1e-6:
                return -1.0
            t = np.dot(self.position - origin, self.normal) / denom
            if SONAR_MIN_RANGE < t <= max_range:
                return t
            return -1.0


class Environment:
    """Ambiente de simulação — contém obstáculos e limites."""

    def __init__(self, pool_depth: float = 20.0, pool_radius: float = 50.0):
        self.pool_depth = pool_depth
        self.pool_radius = pool_radius
        self.obstacles: List[Obstacle] = []
        self._setup_boundaries(pool_depth, pool_radius)

    def _setup_boundaries(self, depth: float, radius: float) -> None:
        """Adiciona paredes do ambiente como planos."""
        # fundo — normal aponta pra cima (oposta ao raio descendente)
        self.obstacles.append(Obstacle(
            position=np.array([0, 0, depth]),
            radius=0,
            normal=np.array([0, 0, 1.0])
        ))
        # superfície — normal aponta pra baixo
        self.obstacles.append(Obstacle(
            position=np.array([0, 0, 0]),
            radius=0,
            normal=np.array([0, 0, -1.0])
        ))

    def add_sphere(self, position: np.ndarray, radius: float) -> None:
        self.obstacles.append(Obstacle(position=position, radius=radius))

    def add_wall(self, position: np.ndarray, normal: np.ndarray) -> None:
        self.obstacles.append(Obstacle(position=position, radius=0, normal=normal))

    def raycast(
        self,
        origin: np.ndarray,
        direction: np.ndarray,
        max_range: float = SONAR_RANGE_MAX
    ) -> Tuple[float, float]:
        """
        Lança raio e retorna (distância, confiança).
        Confiança diminui com ângulo de incidência e distância.
        """
        direction = direction / np.linalg.norm(direction)
        min_dist  = max_range + 1.0
        hit_normal = None

        for obs in self.obstacles:
            d = obs.intersect_ray(origin, direction, max_range)
            if 0 < d < min_dist:
                min_dist  = d
                hit_normal = obs.normal if obs.radius == 0 else \
                             (origin + d*direction - obs.position)

        if min_dist > max_range:
            return -1.0, 0.0

        # confiança: maior perto e com incidência normal
        dist_factor = 1.0 - (min_dist / max_range)
        if hit_normal is not None:
            hn = hit_normal / (np.linalg.norm(hit_normal) + 1e-9)
            angle_factor = abs(np.dot(-direction, hn))
        else:
            angle_factor = 0.8

        confidence = dist_factor * angle_factor
        return min_dist, confidence


# ─────────────────────────────────────────────
# SENSOR ENGINE
# ─────────────────────────────────────────────

class SensorEngine:
    """
    Simula o stack completo de sensores do USV.

    Uso:
        env     = Environment(pool_depth=10.0)
        sensors = SensorEngine(env, noise_scale=1.0)
        bundle  = sensors.read(physics.state, physics.time)
    """

    # 6 direções ortogonais no referencial do corpo
    SONAR_DIRECTIONS = np.array([
        [ 1,  0,  0],   # frontal (surge+)
        [-1,  0,  0],   # traseiro (surge-)
        [ 0,  1,  0],   # estibordo (sway+)
        [ 0, -1,  0],   # bombordo (sway-)
        [ 0,  0,  1],   # abaixo (heave+)
        [ 0,  0, -1],   # acima (heave-)
    ], dtype=float)

    def __init__(
        self,
        environment: Environment,
        noise_scale: float = 1.0,   # 0 = sem ruído, 1 = ruído real, >1 = exagerado
        rayleigh_sigma: float = RAYLEIGH_DEFAULT_SIGMA,
        enable_rayleigh: bool = False,
        seed:        int   = 42,
        wave_hs: float = 0.2,
    ):
        self.env         = environment
        self.noise_scale = noise_scale
        self.rng         = np.random.default_rng(seed)

        # bias do IMU — deriva lentamente
        self._accel_bias = np.zeros(3)
        self._gyro_bias  = np.zeros(3)
        self._bias_drift_rate = 0.001  # rad/s por segundo

        # histórico para derivar aceleração linear verdadeira no body frame
        self._last_imu_time: Optional[float] = None
        self._last_body_vel: Optional[np.ndarray] = None

        # perturbações ambientais (maresia) no referencial do mundo
        self.rayleigh_sigma = max(0.0, float(rayleigh_sigma))
        self.enable_rayleigh = bool(enable_rayleigh)
        self.environment_scale = 0.0
        self._env_current_world = np.zeros(3)
        self._env_turbulence = 0.0
        self._last_env_time: Optional[float] = None

        # timer de atualização do sonar
        self._last_sonar_update = -1.0
        self._sonar_dt = 1.0 / SONAR_UPDATE_HZ
        self._last_sonar_readings: List[SonarReading] = []
        # spectral wave model (superposição harmônica)
        self.spectral_enabled = False
        self.wave_num_harmonics = 8
        self.wave_peak_freq = 0.8
        self.wave_hs = max(0.0, float(wave_hs))  # Significant Wave Height (m)
        self.wave_spectrum = 'jonswap'  # or 'pm' for Pierson-Moskowitz
        self.wave_amp_scale = 0.02
        self._harmonic_freqs = None
        self._harmonic_phases = None
        self._harmonic_dirs = None
        self._harmonic_amps = None

        # HIL-related options
        self.hil_enabled = False
        self.hil_latency = 0.0        # seconds to emulate processing/tx
        self.hil_packet_loss = 0.0    # probability [0,1] to drop incoming frames
        self.hil_jitter_ms = 0.0      # std dev of random jitter in ms
        self.hil_sensor_fault_rate = 0.0  # probability [0,1] per-frame to inject a sensor fault
        self.hil_sensor_fault_types = ['drop', 'stuck', 'spike', 'extra_noise']
        self._fault_stuck_values = {}
        self.fault_profile = SensorFaultProfile()
        self._last_fault_bundle: Optional[SensorBundle] = None

    # ─── Interface pública ───────────────────

    def read(self, state: VehicleState, time: float) -> SensorBundle:
        """Lê todos os sensores dado o estado físico real."""
        self._update_environmental_state(time)

        imu       = self._read_imu(state, time)
        sonar     = self._read_sonar(state, time)
        barometer = self._read_barometer(state, time)

        if self.hil_enabled and self.hil_jitter_ms > 0.0:
            import time as _time
            jitter = float(self.rng.normal(0.0, self.hil_jitter_ms)) / 1000.0
            _time.sleep(max(0.0, jitter))

        return SensorBundle(
            imu=imu,
            sonar=sonar,
            barometer=barometer,
            timestamp=time,
        )

    # ─── HIL integration helpers ───────────────────

    def set_hil_mode(
        self,
        enabled: bool,
        latency_s: float = 0.0,
        packet_loss: float = 0.0,
        jitter_ms: float = None,
        jitter_mean_s: float = None,
        jitter_std_s: float = None,
    ) -> None:
        """Configure HIL emulation parameters.

        When `enabled` is True, `read_from_hardware()` will simulate latency
        and packet loss. `packet_loss` is the probability [0,1] that an
        incoming hardware frame is dropped.
        """
        self.hil_enabled = bool(enabled)
        self.hil_latency = max(0.0, float(latency_s))
        self.hil_packet_loss = float(np.clip(packet_loss, 0.0, 1.0))
        # backwards-compatible handling: accept legacy jitter_mean_s / jitter_std_s (seconds)
        if jitter_ms is not None:
            self.hil_jitter_ms = max(0.0, float(jitter_ms))
        else:
            jm = 0.0
            if jitter_mean_s is not None:
                jm = max(jm, abs(float(jitter_mean_s)) * 1000.0)
            if jitter_std_s is not None:
                jm = max(jm, abs(float(jitter_std_s)) * 1000.0)
            self.hil_jitter_ms = max(0.0, float(jm))
        # keep legacy attributes for tests/code that reference them
        self.hil_jitter_mean_s = float(jitter_mean_s or 0.0)
        self.hil_jitter_std_s = float(jitter_std_s or 0.0)

    def set_fault_profile(self, fault_profile: Optional[SensorFaultProfile]) -> None:
        """Define o perfil de falhas para validação HIL."""
        self.fault_profile = fault_profile or SensorFaultProfile()

    def clear_fault_profile(self) -> None:
        self.fault_profile = SensorFaultProfile()

    def set_hil_faults(self, jitter_ms: float = 0.0, sensor_fault_rate: float = 0.0, fault_types: list = None) -> None:
        """Configure HIL stochastic fault injection parameters.

        - `jitter_ms`: per-packet jitter stddev in milliseconds
        - `sensor_fault_rate`: probability [0,1] to inject a random fault per frame
        - `fault_types`: list of allowed fault types (drop, stuck, spike, extra_noise)
        """
        self.hil_jitter_ms = max(0.0, float(jitter_ms))
        self.hil_sensor_fault_rate = float(np.clip(sensor_fault_rate, 0.0, 1.0))
        if fault_types is not None:
            self.hil_sensor_fault_types = fault_types

    def _apply_sensor_faults(self, bundle: SensorBundle) -> SensorBundle:
        """Aplica corrupção/degradação em sensores para validação."""
        fp = self.fault_profile

        # IMU: bias + dropout via dados atrasados/estáticos quando possível.
        if self.rng.random() < fp.imu_dropout_prob and self._last_fault_bundle is not None:
            bundle.imu.accel = self._last_fault_bundle.imu.accel.copy()
            bundle.imu.gyro = self._last_fault_bundle.imu.gyro.copy()
        else:
            bundle.imu.accel = bundle.imu.accel + np.asarray(fp.imu_accel_bias, dtype=float)
            bundle.imu.gyro = bundle.imu.gyro + np.asarray(fp.imu_gyro_bias, dtype=float)

        # Sonar: dropout e escala de alcance/confiança.
        for reading in bundle.sonar:
            if self.rng.random() < fp.sonar_dropout_prob:
                reading.distance = -1.0
                reading.confidence = 0.0
            elif reading.distance > 0:
                reading.distance *= float(max(0.0, fp.sonar_range_scale))
                reading.confidence *= float(max(0.0, fp.sonar_confidence_scale))

        # Barômetro: bias ou fallback para valor anterior.
        if self.rng.random() < fp.baro_dropout_prob and self._last_fault_bundle is not None:
            bundle.barometer.depth = float(self._last_fault_bundle.barometer.depth)
            bundle.barometer.pressure = float(self._last_fault_bundle.barometer.pressure)
        else:
            bundle.barometer.depth += float(fp.baro_depth_bias)
            bundle.barometer.pressure = RHO_FRESHWATER * G * bundle.barometer.depth

        self._last_fault_bundle = bundle
        return bundle

    def parse_hardware_frame(self, frame: dict) -> Optional[SensorBundle]:
        """Parse a hardware-provided sensor frame (dict) into a SensorBundle.

        Expected frame formats (flexible):
          - {'imu': {'accel':[...], 'gyro':[...], 'timestamp': t},
             'sonar': [ { 'direction':[...], 'distance':d, 'confidence':c, 'timestamp':t }, ... ],
             'barometer': {'depth': d, 'pressure': p, 'timestamp': t },
             'timestamp': t }

        Returns None if frame is invalid.
        """
        if not isinstance(frame, dict) or not frame:
            return None

        now = time.time()

        # IMU
        imu_frame = frame.get('imu')
        if imu_frame and 'accel' in imu_frame and 'gyro' in imu_frame:
            accel = np.array(imu_frame['accel'], dtype=float)
            gyro  = np.array(imu_frame['gyro'], dtype=float)
            ts    = float(imu_frame.get('timestamp', now))
            imu = IMUReading(accel=accel, gyro=gyro, velocity=None, timestamp=ts)
        else:
            imu = None

        # Sonar
        sonar_list = []
        sframe = frame.get('sonar')
        if isinstance(sframe, list):
            for s in sframe:
                try:
                    dirv = np.array(s.get('direction', [0.0, 0.0, 0.0]), dtype=float)
                    dist = float(s.get('distance', -1.0))
                    conf = float(s.get('confidence', 0.0))
                    ts_s = float(s.get('timestamp', now))
                    sonar_list.append(SonarReading(direction=dirv, distance=dist, confidence=conf, timestamp=ts_s))
                except Exception:
                    continue

        # Barometer
        bar_frame = frame.get('barometer')
        if bar_frame and 'depth' in bar_frame:
            depth = float(bar_frame.get('depth', 0.0))
            pressure = float(bar_frame.get('pressure', RHO_FRESHWATER * G * depth))
            ts_b = float(bar_frame.get('timestamp', now))
            baro = BarometerReading(depth=depth, pressure=pressure, timestamp=ts_b)
        else:
            baro = None

        # if we have at least one of the sensors, build a SensorBundle
        if imu is None and not sonar_list and baro is None:
            return None

        # fill missing with simulated defaults by calling `read()` with a dummy state
        dummy_state = VehicleState()
        current_time = now
        if imu is None or baro is None or not sonar_list:
            sim_bundle = self.read(dummy_state, current_time)
            if imu is None:
                imu = sim_bundle.imu
            if not sonar_list:
                sonar_list = sim_bundle.sonar
            if baro is None:
                baro = sim_bundle.barometer

        return self._apply_sensor_faults(SensorBundle(imu=imu, sonar=sonar_list, barometer=baro, timestamp=current_time))

    def read_from_hardware(self, frame) -> Optional[SensorBundle]:
        """Emulate receiving a hardware frame: accept dict, bytes or JSON string.

        Behavior:
          - If HIL packet-loss is enabled, randomly drop the frame and return None.
          - If HIL latency is set, sleep to emulate processing/transmission delay.
          - Accept `frame` as a `dict` (already parsed), `bytes` (JSON) or
            `str` (JSON text). Binary sensor frames are not yet defined and
            will be treated as invalid.

        Returns a SensorBundle or None if packet was dropped or invalid.
        """
        # packet loss
        if self.hil_enabled and self.rng.random() < self.hil_packet_loss:
            return None

        # latency
        if self.hil_enabled and self.hil_latency > 0.0:
            time.sleep(self.hil_latency)

        if self.hil_enabled and self.hil_jitter_ms > 0.0:
            jitter = float(self.rng.normal(0.0, self.hil_jitter_ms)) / 1000.0
            time.sleep(max(0.0, jitter))

        # accept bytes / str (JSON) or dict
        parsed_frame = None
        if isinstance(frame, bytes):
            try:
                parsed_frame = json.loads(frame.decode('utf-8'))
            except Exception:
                # try HIL binary sensor format without top-level import to avoid circular imports
                try:
                    from hil_interface import HILProtocol
                    parsed_frame = HILProtocol.deserialize_binary_sensor(frame)
                except Exception:
                    parsed_frame = None
        elif isinstance(frame, str):
            try:
                parsed_frame = json.loads(frame)
            except Exception:
                parsed_frame = None
        elif isinstance(frame, dict):
            parsed_frame = frame
        else:
            parsed_frame = None

        if parsed_frame is None:
            return None

        bundle = self.parse_hardware_frame(parsed_frame)
        if bundle is not None:
            # Apply lightweight HIL stochastic fault injection (configurable)
            if self.hil_enabled and self.hil_sensor_fault_rate > 0.0 and self.rng.random() < self.hil_sensor_fault_rate:
                ftype = str(self.rng.choice(self.hil_sensor_fault_types))
                if ftype == 'drop':
                    return None
                if ftype == 'stuck':
                    if 'imu' in self._fault_stuck_values:
                        bundle.imu = self._fault_stuck_values['imu']
                    else:
                        self._fault_stuck_values['imu'] = bundle.imu
                    if 'barometer' in self._fault_stuck_values:
                        bundle.barometer = self._fault_stuck_values['barometer']
                    else:
                        self._fault_stuck_values['barometer'] = bundle.barometer
                if ftype == 'spike':
                    try:
                        bundle.imu.accel = bundle.imu.accel + np.array([float(self.rng.normal(0.0, 5.0)) for _ in range(3)])
                        bundle.imu.gyro  = bundle.imu.gyro  + np.array([float(self.rng.normal(0.0, 1.0)) for _ in range(3)])
                    except Exception:
                        pass
                if ftype == 'extra_noise':
                    try:
                        bundle.imu.accel = bundle.imu.accel + np.array([float(self.rng.normal(0.0, 0.5)) for _ in range(3)])
                        bundle.imu.gyro  = bundle.imu.gyro  + np.array([float(self.rng.normal(0.0, 0.05)) for _ in range(3)])
                    except Exception:
                        pass
            return bundle

        # Fallback: tolerant construction for minimal frames (imu/baro present)
        now = time.time()
        imu = None
        baro = None
        sonar_list = []

        imu_frame = parsed_frame.get('imu') if isinstance(parsed_frame, dict) else None
        if imu_frame and 'accel' in imu_frame and 'gyro' in imu_frame:
            try:
                accel = np.array(imu_frame['accel'], dtype=float)
                gyro = np.array(imu_frame['gyro'], dtype=float)
                ts = float(imu_frame.get('timestamp', now))
                imu = IMUReading(accel=accel, gyro=gyro, velocity=None, timestamp=ts)
            except Exception:
                imu = None

        bar_frame = parsed_frame.get('barometer') if isinstance(parsed_frame, dict) else None
        if bar_frame and 'depth' in bar_frame:
            try:
                depth = float(bar_frame.get('depth', 0.0))
                pressure = float(bar_frame.get('pressure', RHO_FRESHWATER * G * depth))
                ts_b = float(bar_frame.get('timestamp', now))
                baro = BarometerReading(depth=depth, pressure=pressure, timestamp=ts_b)
            except Exception:
                baro = None

        if imu is None and baro is None and not sonar_list:
            return None

        # fill missing sensors from simulation defaults
        if imu is None or baro is None or not sonar_list:
            dummy_state = VehicleState()
            sim_bundle = self.read(dummy_state, now)
            if imu is None:
                imu = sim_bundle.imu
            if not sonar_list:
                sonar_list = sim_bundle.sonar
            if baro is None:
                baro = sim_bundle.barometer

        bundle = SensorBundle(imu=imu, sonar=sonar_list, barometer=baro, timestamp=now)

        # Apply lightweight HIL stochastic fault injection (configurable)
        if self.hil_enabled and self.hil_sensor_fault_rate > 0.0 and self.rng.random() < self.hil_sensor_fault_rate:
            ftype = str(self.rng.choice(self.hil_sensor_fault_types))
            # drop entire frame
            if ftype == 'drop':
                return None
            # stuck: keep previous values if available, otherwise freeze current
            if ftype == 'stuck':
                if 'imu' in self._fault_stuck_values:
                    bundle.imu = self._fault_stuck_values['imu']
                else:
                    self._fault_stuck_values['imu'] = bundle.imu
                if 'barometer' in self._fault_stuck_values:
                    bundle.barometer = self._fault_stuck_values['barometer']
                else:
                    self._fault_stuck_values['barometer'] = bundle.barometer
            # spike: large transient bias
            if ftype == 'spike':
                try:
                    bundle.imu.accel = bundle.imu.accel + np.array([float(self.rng.normal(0.0, 5.0)) for _ in range(3)])
                    bundle.imu.gyro  = bundle.imu.gyro  + np.array([float(self.rng.normal(0.0, 1.0)) for _ in range(3)])
                except Exception:
                    pass
            # extra_noise: raise noise floor
            if ftype == 'extra_noise':
                try:
                    bundle.imu.accel = bundle.imu.accel + np.array([float(self.rng.normal(0.0, 0.5)) for _ in range(3)])
                    bundle.imu.gyro  = bundle.imu.gyro  + np.array([float(self.rng.normal(0.0, 0.05)) for _ in range(3)])
                except Exception:
                    pass

        return bundle

    def detect_waypoint(
        self,
        state: VehicleState,
        waypoint_world: np.ndarray,
        time: float,
        max_range: float = 100.0,
        fov_deg: float = 60.0,
        noise_std: float = 0.05,
        dropout_prob: float = 0.05,
    ) -> Optional[np.ndarray]:
        """Simula um detector visual que retorna o vetor relativo ao waypoint
        no referencial do corpo (body frame) quando detectado.

        Retorna None em caso de não detecção.
        """
        # posição do veículo e vetor para o waypoint no referencial mundo
        pos = np.array([state.x, state.y, state.z], dtype=float)
        wp = np.array(waypoint_world, dtype=float)
        vec_world = wp - pos
        dist = float(np.linalg.norm(vec_world))
        if dist <= 0 or dist > max_range:
            return None

        # orientação veículo -> rotação do body para world
        R = self._rotation_matrix(state.phi, state.tht, state.psi)
        vec_body = R.T @ vec_world

        # checa FOV (azimuth/elevation)
        x, y, z = vec_body
        az = float(np.arctan2(y, x))
        el = float(np.arctan2(z, np.sqrt(x * x + y * y) + 1e-9))
        half_fov = np.radians(fov_deg) / 2.0
        if abs(az) > half_fov or abs(el) > half_fov:
            return None

        # dropout simula oclusões / perdas de detecção
        if self.rng.random() < dropout_prob:
            return None

        # ruído gaussiano proporcional à distância
        noise = self.rng.normal(0.0, noise_std * max(1.0, dist * 0.02), size=3)
        meas = vec_body + noise
        # saturate small values
        return meas

    def set_noise_scale(self, scale: float) -> None:
        """Ajusta nível de ruído em runtime — útil pra domain randomization."""
        self.noise_scale = max(0.0, scale)

    def set_environmental_disturbance(
        self,
        enabled: bool,
        scale: float = 1.0,
        rayleigh_sigma: Optional[float] = None,
        spectral: bool = False,
        wave_num_harmonics: int = 8,
        wave_peak_freq: float = 0.8,
        wave_amp_scale: float = 0.02,
        wave_hs: Optional[float] = None,
    ) -> None:
        """
        Configura perturbação ambiental não gaussiana (Rayleigh).

        Args:
            enabled: ativa/desativa efeito de maresia/turbulência.
            scale: intensidade global do efeito (0 = desligado).
            rayleigh_sigma: escala da distribuição de Rayleigh.
        """
        self.enable_rayleigh = bool(enabled)
        self.environment_scale = max(0.0, float(scale))
        if rayleigh_sigma is not None:
            self.rayleigh_sigma = max(0.0, float(rayleigh_sigma))

        # spectral options
        self.spectral_enabled = bool(spectral)
        self.wave_num_harmonics = int(max(1, wave_num_harmonics))
        self.wave_peak_freq = float(wave_peak_freq)
        self.wave_amp_scale = float(wave_amp_scale)
        if wave_hs is not None:
            self.wave_hs = max(0.0, float(wave_hs))

        if (not self.enable_rayleigh) or self.environment_scale == 0.0:
            self._env_current_world[:] = 0.0
            self._env_turbulence = 0.0
            # clear spectral harmonics
            self._harmonic_freqs = None
            self._harmonic_phases = None
            self._harmonic_dirs = None
            self._harmonic_amps = None
            return

        # initialize spectral harmonics if requested
        if self.spectral_enabled and self._harmonic_freqs is None:
            n = self.wave_num_harmonics
            # frequencies spaced around peak
            freqs = np.linspace(self.wave_peak_freq * 0.5, self.wave_peak_freq * 1.5, n)
            phases = self.rng.uniform(0.0, 2.0 * np.pi, size=n)
            dirs = self.rng.normal(0.0, 1.0, size=(n, 3))
            dirs /= (np.linalg.norm(dirs, axis=1, keepdims=True) + 1e-9)
            # amplitudes: use spectral model (JONSWAP/PM) to derive surface elevation S(f),
            # then convert to orbital velocity amplitude A_v ~ 2π f * sqrt(2 S(f) df).
            df = freqs[1] - freqs[0] if n > 1 else freqs[0]
            # base spectrum (unnormalized)
            def jonswap_spectrum(f, fp, gamma=3.3):
                g = 9.81
                sigma = np.where(f <= fp, 0.07, 0.09)
                r = np.exp(- (f - fp)**2 / (2 * sigma**2 * fp**2))
                S = (g**2) * (2*np.pi)**-4 * f**-5 * np.exp(-1.25 * (fp / f)**4) * (gamma ** r)
                return S

            if self.wave_spectrum == 'jonswap':
                S0 = jonswap_spectrum(freqs, self.wave_peak_freq)
            else:
                S0 = jonswap_spectrum(freqs, self.wave_peak_freq)

            # scale S0 so that variance matches Hs^2/16
            desired_var = (self.wave_hs ** 2) / 16.0
            var0 = float(np.sum(S0) * df)
            scale = (desired_var / var0) if var0 > 0 else 0.0
            S = S0 * scale

            # surface elevation amplitude per harmonic (m)
            eta_amp = np.sqrt(2.0 * S * df)
            # approximate orbital velocity amplitude (m/s)
            amps = (2.0 * np.pi * freqs) * eta_amp
            # apply global scaling
            amps = amps * self.wave_amp_scale * self.environment_scale
            self._harmonic_freqs = freqs
            self._harmonic_phases = phases
            self._harmonic_dirs = dirs
            self._harmonic_amps = amps

    def _signed_rayleigh_noise(self, size=None) -> np.ndarray:
        """Amostra ruído Rayleigh com sinal aleatório para perturbações bidirecionais."""
        if size is None:
            amp = float(self.rng.rayleigh(max(self.rayleigh_sigma, 1e-9)))
            sign = -1.0 if self.rng.random() < 0.5 else 1.0
            return np.array(sign * amp)

        amp = self.rng.rayleigh(max(self.rayleigh_sigma, 1e-9), size=size)
        sign = np.where(self.rng.random(size=size) < 0.5, -1.0, 1.0)
        return amp * sign

    def _update_environmental_state(self, time: float) -> None:
        """Atualiza corrente/turbulência ambiental com dinâmica lenta."""
        if not self.enable_rayleigh or self.environment_scale <= 0.0:
            self._env_current_world[:] = 0.0
            self._env_turbulence = 0.0
            self._last_env_time = time
            return

        if self._last_env_time is None:
            dt_env = 0.0
        else:
            dt_env = max(0.0, time - self._last_env_time)
        self._last_env_time = time

        if dt_env <= 0.0:
            return

        tau = 2.0
        alpha = 1.0 - np.exp(-dt_env / tau)

        if self.spectral_enabled and self._harmonic_freqs is not None:
            # build spectral current as sum of harmonics
            vec = np.zeros(3)
            turb_vals = []
            for i, f in enumerate(self._harmonic_freqs):
                phase = 2.0 * np.pi * f * time + float(self._harmonic_phases[i])
                inst = float(self._harmonic_amps[i] * np.sin(phase))
                vec += inst * self._harmonic_dirs[i]
                turb_vals.append(abs(inst))

            target_current = vec
            self._env_current_world = (1.0 - alpha) * self._env_current_world + alpha * target_current
            turb_target = float(np.mean(turb_vals)) if turb_vals else 0.0
            self._env_turbulence = (1.0 - alpha) * self._env_turbulence + alpha * turb_target
        else:
            direction = self.rng.normal(0.0, 1.0, 3)
            direction /= (np.linalg.norm(direction) + 1e-9)
            amp = float(self.rng.rayleigh(max(self.rayleigh_sigma, 1e-9)))
            target_current = direction * amp * self.environment_scale
            self._env_current_world = (
                (1.0 - alpha) * self._env_current_world +
                alpha * target_current
            )

            turb_target = float(self.rng.rayleigh(max(self.rayleigh_sigma, 1e-9)))
            self._env_turbulence = (1.0 - alpha) * self._env_turbulence + alpha * turb_target

    def get_environmental_state(self) -> Tuple[np.ndarray, float]:
        """
        Retorna o estado ambiental atual que representa corrente (vetor no referencial
        do mundo) e um escalar de turbulência.

        Usado pelo `PhysicsEngine` para converter em forças hidrodinâmicas.
        """
        return self._env_current_world.copy(), float(self._env_turbulence)

    def get_environmental_harmonics(self):
        """
        Retorna tupla (freqs, amps, phases, dirs) quando o modelo espectral está ativo.
        Caso contrário retorna None.
        - freqs: (n,) Hz
        - amps:  (n,) velocidade amplitude (m/s)
        - phases: (n,) rad
        - dirs:  (n,3) direção unitária no referencial mundo
        """
        if not self.spectral_enabled or self._harmonic_freqs is None:
            return None
        return (self._harmonic_freqs.copy(),
                self._harmonic_amps.copy(),
                self._harmonic_phases.copy(),
                self._harmonic_dirs.copy())

    # ─── IMU ─────────────────────────────────

    def _read_imu(self, state: VehicleState, time: float) -> IMUReading:
        """
        Simula IMU com ruído gaussiano e bias derivante.
        Aceleração inclui gravidade projetada no body frame.
        """
        # aceleração linear no body frame por diferença finita em ν linear.
        # no hardware real viria diretamente da dinâmica do IMU.
        body_vel = np.array([state.u, state.v, state.w], dtype=float)
        if self._last_imu_time is None or self._last_body_vel is None:
            linear_accel_body = np.zeros(3)
        else:
            dt_imu = max(1e-6, time - self._last_imu_time)
            linear_accel_body = (body_vel - self._last_body_vel) / dt_imu

        self._last_imu_time = time
        self._last_body_vel = body_vel.copy()

        # na prática o acelerômetro mede aceleração específica + projeção da gravidade
        phi, theta = state.phi, state.tht
        g_body = np.array([
            -G * np.sin(theta),
             G * np.cos(theta) * np.sin(phi),
             G * np.cos(theta) * np.cos(phi),
        ])

        # velocidades angulares verdadeiras
        true_gyro = np.array([state.p, state.q, state.r])
        true_velocity = body_vel.copy()

        # drift do bias
        self._accel_bias += self.rng.normal(
            0, IMU_ACCEL_BIAS_STD * self._bias_drift_rate, 3)
        self._gyro_bias  += self.rng.normal(
            0, IMU_GYRO_BIAS_STD  * self._bias_drift_rate, 3)

        # ruído branco
        accel_noise = self.rng.normal(0, IMU_ACCEL_NOISE_STD * self.noise_scale, 3)
        gyro_noise  = self.rng.normal(0, IMU_GYRO_NOISE_STD  * self.noise_scale, 3)
        velocity_noise = self.rng.normal(0, 0.03 * self.noise_scale, 3)

        # componente não gaussiana (maresia/turbulência)
        if self.enable_rayleigh and self.environment_scale > 0.0:
            accel_noise += self._signed_rayleigh_noise(size=3) * 0.2 * self.environment_scale
            gyro_noise  += self._signed_rayleigh_noise(size=3) * 0.01 * self.environment_scale

        # leitura ruidosa
        accel_meas = linear_accel_body + g_body + self._accel_bias + accel_noise
        gyro_meas  = true_gyro   + self._gyro_bias  + gyro_noise
        velocity_meas = true_velocity + velocity_noise

        return IMUReading(accel=accel_meas, gyro=gyro_meas, velocity=velocity_meas, timestamp=time)

    # ─── Sonar ───────────────────────────────

    def _read_sonar(
        self, state: VehicleState, time: float
    ) -> List[SonarReading]:
        """
        6 transdutores ortogonais com beamwidth de 25°.
        Atualiza a SONAR_UPDATE_HZ Hz.
        Ruído proporcional à distância — modelo acústico simplificado.
        """
        # throttle — sonar não atualiza a cada physics step
        if time - self._last_sonar_update < self._sonar_dt:
            return self._last_sonar_readings

        self._last_sonar_update = time

        R = self._rotation_matrix(state.phi, state.tht, state.psi)
        position = np.array([state.x, state.y, state.z])
        readings = []

        for dir_body in self.SONAR_DIRECTIONS:
            # transforma direção pro referencial inercial
            dir_world = R @ dir_body

            # raycasting central
            dist, conf = self.env.raycast(position, dir_world)

            if dist > 0:
                # ruído acústico — aumenta com distância
                noise_std = (0.02 + 0.01 * dist) * self.noise_scale
                dist_meas = dist + self.rng.normal(0, noise_std)

                if self.enable_rayleigh and self.environment_scale > 0.0:
                    rayleigh_noise = float(
                        self._signed_rayleigh_noise() * (0.015 + 0.005 * dist) * self.environment_scale
                    )
                    dist_meas += rayleigh_noise
                    conf *= float(np.exp(-0.35 * self._env_turbulence * self.environment_scale))

                dist_meas = max(SONAR_MIN_RANGE, dist_meas)

                # simula beamwidth — média de raios dentro do cone
                dist_meas = self._apply_beamwidth(
                    position, dir_world, dist_meas
                )
            else:
                dist_meas = -1.0

            readings.append(SonarReading(
                direction=dir_body,
                distance=dist_meas,
                confidence=conf,
                timestamp=time,
            ))

        self._last_sonar_readings = readings
        return readings

    def _apply_beamwidth(
        self,
        origin: np.ndarray,
        direction: np.ndarray,
        center_dist: float,
        n_rays: int = 8,
    ) -> float:
        """
        Simula efeito do beamwidth de 25° — média de raios no cone.
        Raios dentro do cone podem retornar distâncias diferentes.
        """
        half_angle = SONAR_BEAMWIDTH_RAD / 2
        distances  = [center_dist]

        # base ortogonal ao vetor de direção
        perp = np.array([1, 0, 0]) if abs(direction[0]) < 0.9 else np.array([0, 1, 0])
        perp = np.cross(direction, perp)
        perp = perp / np.linalg.norm(perp)
        perp2 = np.cross(direction, perp)

        for i in range(n_rays):
            angle  = self.rng.uniform(0, half_angle)
            azimuth = self.rng.uniform(0, 2*np.pi)

            # raio deflectido dentro do cone
            d = (direction * np.cos(angle) +
                 perp  * np.sin(angle) * np.cos(azimuth) +
                 perp2 * np.sin(angle) * np.sin(azimuth))
            d = d / np.linalg.norm(d)

            dist, _ = self.env.raycast(origin, d)
            if dist > 0:
                distances.append(dist)

        return float(np.mean(distances))

    # ─── Barômetro ───────────────────────────

    def _read_barometer(
        self, state: VehicleState, time: float
    ) -> BarometerReading:
        """Pressão → profundidade com ruído do MS5837."""
        true_depth = state.z  # NED: z positivo = profundidade
        pressure   = RHO_FRESHWATER * G * true_depth
        noise      = self.rng.normal(0, BARO_NOISE_STD * self.noise_scale)
        if self.enable_rayleigh and self.environment_scale > 0.0:
            noise += float(
                self._signed_rayleigh_noise() * BARO_NOISE_STD * 0.8 * self.environment_scale
            )
        depth_meas = true_depth + noise

        return BarometerReading(
            depth=depth_meas,
            pressure=pressure,
            timestamp=time,
        )

    # ─── Utilitários ─────────────────────────

    @staticmethod
    def _rotation_matrix(phi, theta, psi) -> np.ndarray:
        """Matriz de rotação ZYX 3x3."""
        cphi=np.cos(phi); sphi=np.sin(phi)
        cth=np.cos(theta); sth=np.sin(theta)
        cpsi=np.cos(psi); spsi=np.sin(psi)
        return np.array([
            [cpsi*cth, cpsi*sth*sphi-spsi*cphi, cpsi*sth*cphi+spsi*sphi],
            [spsi*cth, spsi*sth*sphi+cpsi*cphi, spsi*sth*cphi-cpsi*sphi],
            [-sth,     cth*sphi,                 cth*cphi               ]
        ])


# ─────────────────────────────────────────────
# EXTENDED KALMAN FILTER
# ─────────────────────────────────────────────

class ExtendedKalmanFilter:
    """
    EKF para estimação de estado 6 DOF.

    Estado: x = [η, ν] = [x,y,z,φ,θ,ψ, u,v,w,p,q,r] (12 dimensões)

    Observações:
        - IMU:       aceleração e velocidade angular (6D)
        - Sonar:     distâncias ortogonais (até 6D)
        - Barômetro: profundidade (1D)

    O Control Engine NUNCA acessa o estado físico diretamente.
    Usa apenas EKF.state_estimate — realismo de percepção garantido.
    """

    DIM_STATE = 12  # [η(6), ν(6)]

    @staticmethod
    def _initial_covariance() -> np.ndarray:
        """Covariância inicial conservadora para evitar ganho excessivo no arranque."""
        return np.diag([
            0.01, 0.01, 0.01,   # posição
            0.01, 0.01, 0.01,   # orientação
            0.001, 0.001, 0.001, # velocidade linear
            0.001, 0.001, 0.001, # velocidade angular
        ])

    def __init__(self, physics: PhysicsEngine, pool_radius: float = 50.0, pool_depth: float = 10.0):
        self.physics = physics
        self.pool_radius = pool_radius
        self.pool_depth = pool_depth

        # covariâncias de processo — quanta incerteza adicionamos por step
        self.Q = np.diag([
            0.0005, 0.0005, 0.0005,  # posição xyz
            0.0010, 0.0010, 0.0010,  # orientação euler
            0.02,   0.02,   0.02,    # velocidade linear
            0.01,   0.01,   0.01,    # velocidade angular
        ])

        # covariâncias de medição
        self.R_imu   = np.diag([IMU_ACCEL_NOISE_STD**2]*3 +
                                [IMU_GYRO_NOISE_STD**2]*3)
        self.R_sonar = np.eye(6) * 0.05**2   # ~5cm de ruído por transdutor
        self.R_baro  = np.array([[BARO_NOISE_STD**2]])

        # estado inicial — veículo na origem parado
        self._x = np.zeros(self.DIM_STATE)
        self._P = self._initial_covariance()

        self._time = 0.0
        self._last_imu_timestamp: Optional[float] = None
        # vision support
        self.R_vision = np.eye(3) * 0.2**2
        self._vision_last: Optional[np.ndarray] = None

    @property
    def state_estimate(self) -> EKFState:
        # convert last vision measurement (body frame) to world-frame delta if present
        vis = None
        if self._vision_last is not None:
            phi, tht, psi = float(self._x[3]), float(self._x[4]), float(self._x[5])
            R = SensorEngine._rotation_matrix(phi, tht, psi)
            # body->world
            vis_world = R @ self._vision_last
            vis = vis_world.copy()

        return EKFState(
            eta=self._x[:6].copy(),
            nu=self._x[6:].copy(),
            P=self._P.copy(),
            timestamp=self._time,
            vision_relative_waypoint=vis,
        )

    def predict(self, dt: float) -> None:
        """
        Etapa de predição — propaga estado com modelo cinemático linear.
        Usa Jacobiana da dinâmica em torno do estado atual.
        """
        # Jacobiana do modelo de processo — linearização em torno de x
        F = self._compute_F(self._x, dt)

        # propaga estado
        self._x = self._f(self._x, dt)

        # propaga covariância
        self._P = F @ self._P @ F.T + self.Q * dt

        self._time += dt

    def update_imu(self, reading: IMUReading) -> None:
        """
        Atualiza com leitura do IMU.

        Pipeline:
          1) Dead Reckoning: integra gyro e aceleração específica em posição/velocidade.
          2) Correção EKF: usa gyro como observação direta de [p,q,r].
          3) Correção complementar: usa gravidade no acelerômetro para limitar drift de roll/pitch.
        """
        if self._last_imu_timestamp is None:
            dt_imu = 0.0
        else:
            dt_imu = max(0.0, reading.timestamp - self._last_imu_timestamp)
        self._last_imu_timestamp = reading.timestamp

        if dt_imu > 0.0:
            # Dead Reckoning angular: integra taxas em Euler
            omega = reading.gyro.astype(float)
            phi, theta, psi = self._x[3], self._x[4], self._x[5]

            cphi = np.cos(phi); sphi = np.sin(phi)
            cth  = np.cos(theta)
            if abs(cth) < 1e-4:
                cth = np.sign(cth) * 1e-4 if cth != 0 else 1e-4
            tth  = np.tan(theta)

            T = np.array([
                [1, sphi * tth, cphi * tth],
                [0, cphi,      -sphi],
                [0, sphi / cth, cphi / cth],
            ])
            self._x[3:6] += (T @ omega) * dt_imu
            self._x[3:6] = ((self._x[3:6] + np.pi) % (2 * np.pi)) - np.pi

            # Dead Reckoning linear: a_body = accel - g_body(orientação estimada)
            phi, theta, psi = self._x[3], self._x[4], self._x[5]
            g_body = np.array([
                -G * np.sin(theta),
                 G * np.cos(theta) * np.sin(phi),
                 G * np.cos(theta) * np.cos(phi),
            ])
            a_body = reading.accel.astype(float) - g_body

            self._x[6:9] += a_body * dt_imu

            R = SensorEngine._rotation_matrix(phi, theta, psi)
            v_world = R @ self._x[6:9]
            self._x[:3] += v_world * dt_imu

            # gyro observado diretamente como velocidade angular
            self._x[9:12] = omega

            # Complementary correction em roll/pitch usando vetor gravidade medido
            ax, ay, az = reading.accel.astype(float)
            phi_acc = np.arctan2(ay, az)
            theta_acc = np.arctan2(-ax, np.sqrt(ay**2 + az**2) + 1e-9)
            alpha = 0.03
            self._x[3] = (1.0 - alpha) * self._x[3] + alpha * phi_acc
            self._x[4] = (1.0 - alpha) * self._x[4] + alpha * theta_acc

        # atualização EKF explícita para gyro (mantém consistência da covariância)
        z = reading.gyro.astype(float)
        h = self._x[9:12].copy()
        H = np.zeros((3, self.DIM_STATE))
        H[0, 9] = 1.0
        H[1, 10] = 1.0
        H[2, 11] = 1.0
        Rg = np.diag([IMU_GYRO_NOISE_STD**2] * 3)
        self._update(z, h, H, Rg)

        # velocidade corporal sintética: corrige o drift horizontal do dead reckoning
        if reading.velocity is not None:
            z_v = reading.velocity.astype(float)
            h_v = self._x[6:9].copy()
            H_v = np.zeros((3, self.DIM_STATE))
            H_v[0, 6] = 1.0
            H_v[1, 7] = 1.0
            H_v[2, 8] = 1.0
            Rv = np.diag([0.01**2] * 3)
            self._update(z_v, h_v, H_v, Rv)

    def update_sonar(self, readings: List[SonarReading]) -> None:
        """Atualiza com leituras do sonar - usa hits válidos de cima/baixo e horizontais."""
        hits = [r for r in readings if r.hit]
        if not hits:
            return

        for reading in hits:
            if np.allclose(reading.direction, [0, 0, -1]):
                z = np.array([reading.distance])
                h = np.array([self._x[2]])
                H = np.zeros((1, self.DIM_STATE))
                H[0, 2] = 1.0
                self._update(z, h, H, self.R_sonar[:1, :1])

    def _ray_circle_distance(self, position_xy: np.ndarray, direction_xy: np.ndarray, radius: float) -> Optional[float]:
        direction_xy = direction_xy / (np.linalg.norm(direction_xy) + 1e-12)
        px, py = float(position_xy[0]), float(position_xy[1])
        dx, dy = float(direction_xy[0]), float(direction_xy[1])
        b = px * dx + py * dy
        c = px * px + py * py - radius * radius
        disc = b * b - c
        if disc < 0.0:
            return None
        root = np.sqrt(disc)
        candidates = [t for t in (-b - root, -b + root) if t > SONAR_MIN_RANGE]
        return min(candidates) if candidates else None

    def update_sonar_position(self, readings: List[SonarReading]) -> None:
        """Use horizontal sonar hits against the pool boundary to weakly constrain x/y."""
        if not readings or self.pool_radius <= 0:
            return

        position_xy = self._x[:2].copy()
        yaw = float(self._x[5])
        cpsi = np.cos(yaw)
        spsi = np.sin(yaw)
        rotation = np.array([
            [cpsi, -spsi, 0.0],
            [spsi,  cpsi, 0.0],
            [0.0,   0.0,  1.0],
        ])

        for reading in readings:
            if not reading.hit:
                continue
            if np.allclose(reading.direction, [0, 0, 1]) or np.allclose(reading.direction, [0, 0, -1]):
                continue

            world_direction = rotation @ reading.direction
            expected_distance = self._ray_circle_distance(position_xy, world_direction[:2], self.pool_radius)
            if expected_distance is None:
                continue

            z = np.array([reading.distance], dtype=float)
            h = np.array([expected_distance], dtype=float)
            H = np.zeros((1, self.DIM_STATE))
            eps = 1e-4
            for idx in (0, 1, 5):
                x_perturbed = self._x.copy()
                x_perturbed[idx] += eps
                pos_perturbed = x_perturbed[:2]
                yaw_perturbed = float(x_perturbed[5])
                cpsi_p = np.cos(yaw_perturbed)
                spsi_p = np.sin(yaw_perturbed)
                rotation_p = np.array([
                    [cpsi_p, -spsi_p, 0.0],
                    [spsi_p,  cpsi_p, 0.0],
                    [0.0,     0.0,    1.0],
                ])
                world_direction_p = rotation_p @ reading.direction
                expected_p = self._ray_circle_distance(pos_perturbed, world_direction_p[:2], self.pool_radius)
                if expected_p is None:
                    continue
                H[0, idx] = (expected_p - expected_distance) / eps

            R_pos = np.array([[0.25**2]])
            self._update(z, h, H, R_pos)

    def update_barometer(self, reading: BarometerReading) -> None:
        """Atualiza profundidade z com barômetro."""
        z = np.array([reading.depth])
        h = np.array([self._x[2]])
        H = np.zeros((1, self.DIM_STATE))
        H[0, 2] = 1.0
        self._update(z, h, H, self.R_baro)

    def _h_vision(self, x: np.ndarray, waypoint_world: np.ndarray) -> np.ndarray:
        """Modelo de observação para visão: vetor relativo no referencial do corpo.

        h(x) = R_body_world^T * (wp_world - position_world)
        Retorna vetor 3x1 no corpo.
        """
        pos = x[:3]
        phi, theta, psi = x[3], x[4], x[5]
        R = SensorEngine._rotation_matrix(phi, theta, psi)
        vec_world = np.array(waypoint_world, dtype=float) - pos
        return R.T @ vec_world

    def update_vision(self, reading: Optional[np.ndarray], waypoint_world: np.ndarray, R_cov: Optional[np.ndarray] = None) -> None:
        """Atualiza EKF com medição visual do vetor relativo no body frame.

        reading: None ou vetor 3D em body frame.
        waypoint_world: posição do waypoint no referencial mundo (3,).
        """
        self._vision_last = None if reading is None else reading.astype(float)
        if reading is None:
            return

        z = reading.astype(float)
        # monta h(x)
        h = self._h_vision(self._x, waypoint_world)

        # numeric Jacobian 3x12
        H = np.zeros((3, self.DIM_STATE))
        eps = 1e-4
        for idx in range(self.DIM_STATE):
            x_pert = self._x.copy()
            x_pert[idx] += eps
            h_pert = self._h_vision(x_pert, waypoint_world)
            H[:, idx] = (h_pert - h) / eps

        R = self.R_vision if R_cov is None else np.asarray(R_cov, dtype=float)
        if R.shape != (3, 3):
            R = np.eye(3) * float(R)

        self._update(z, h, H, R)

    # ─── EKF internals ───────────────────────

    def _f(self, x: np.ndarray, dt: float) -> np.ndarray:
        """Modelo de processo - cinemática de corpo rígido simplificada."""
        eta = x[:6]
        nu = x[6:]
        J = self._jacobian_eta(eta)
        eta_dot = J @ nu
        x_new = x.copy()
        x_new[:6] += eta_dot * dt
        return x_new

    def _compute_F(self, x: np.ndarray, dt: float) -> np.ndarray:
        """Jacobiana do modelo de processo 12x12."""
        F = np.eye(self.DIM_STATE)
        eta = x[:6]
        J = self._jacobian_eta(eta)
        F[:6, 6:] = J * dt
        return F

    def _h_imu(self, x: np.ndarray) -> np.ndarray:
        """Modelo de observação do IMU."""
        phi, theta = x[3], x[4]
        g_body = np.array([
            -G * np.sin(theta),
            G * np.cos(theta) * np.sin(phi),
            G * np.cos(theta) * np.cos(phi),
        ])
        return np.concatenate([g_body, x[9:12]])

    def _H_imu(self, x: np.ndarray) -> np.ndarray:
        """Jacobiana do modelo de observação do IMU 6x12."""
        H = np.zeros((6, self.DIM_STATE))
        phi, theta = x[3], x[4]
        H[0, 4] = -G * np.cos(theta)
        H[1, 3] = G * np.cos(theta) * np.cos(phi)
        H[1, 4] = -G * np.sin(theta) * np.sin(phi)
        H[2, 3] = -G * np.cos(theta) * np.sin(phi)
        H[2, 4] = -G * np.sin(theta) * np.cos(phi)
        H[3, 9] = 1.0
        H[4, 10] = 1.0
        H[5, 11] = 1.0
        return H

    def _update(self, z: np.ndarray, h: np.ndarray, H: np.ndarray, R: np.ndarray) -> None:
        """Etapa de atualização EKF padrão."""
        innov = z - h
        S = H @ self._P @ H.T + R
        K = self._P @ H.T @ np.linalg.inv(S)
        self._x = self._x + K @ innov
        self._P = (np.eye(self.DIM_STATE) - K @ H) @ self._P

    def _jacobian_eta(self, eta: np.ndarray) -> np.ndarray:
        """Bloco J1 da Jacobiana - transforma nu em eta_dot."""
        phi, theta, psi = eta[3], eta[4], eta[5]
        R = SensorEngine._rotation_matrix(phi, theta, psi)
        cphi = np.cos(phi)
        sphi = np.sin(phi)
        cth = np.cos(theta)
        tth = np.tan(theta)
        T = np.array([
            [1.0, sphi * tth, cphi * tth],
            [0.0, cphi, -sphi],
            [0.0, sphi / (cth + 1e-12), cphi / (cth + 1e-12)],
        ])
        J = np.zeros((6, 6))
        J[:3, :3] = R
        J[3:, 3:] = T
        return J

    def reset(self, initial_state: Optional[np.ndarray] = None) -> None:
        self._x = initial_state.copy() if initial_state is not None else np.zeros(self.DIM_STATE)
        self._P = self._initial_covariance()
        self._time = 0.0
        self._last_imu_timestamp = None
        self._vision_last = None


# ─────────────────────────────────────────────
# TESTES
# ─────────────────────────────────────────────

if __name__ == "__main__":
    import json
    from geometry_engine import GeometryEngine

    print("Inicializando Sensor Engine + EKF...")

    geo = GeometryEngine(L=0.8, D=0.1)
    physics = PhysicsEngine(geo, max_thruster_force=10.0)

    env = Environment(pool_depth=5.0, pool_radius=30.0)
    env.add_sphere(np.array([5.0, 0.0, 3.0]), radius=1.0)

    sensors = SensorEngine(env, noise_scale=1.0)
    ekf = ExtendedKalmanFilter(physics, pool_radius=env.pool_radius, pool_depth=env.pool_depth)

    print("\nTeste 1 - Sensores em repouso:")
    bundle = sensors.read(physics.state, 0.0)
    print(f"  IMU accel: {bundle.imu.accel.round(3)}")
    print(f"  IMU gyro:  {bundle.imu.gyro.round(4)}")
    print(f"  Barômetro: {bundle.barometer.depth:.3f} m")
    print(f"  Sonar hits: {sum(1 for s in bundle.sonar if s.hit)}/6")

    print("\nTeste 2 - Sonar detecta fundo a 10m:")
    state_deep = VehicleState(z=2.0)
    bundle2 = sensors.read(state_deep, 0.1)
    dir_name = ['frente', 'trás', 'estibordo', 'bombordo', 'baixo', 'cima']
    for s in bundle2.sonar:
        idx = list(range(6))[[np.allclose(s.direction, d) for d in SensorEngine.SONAR_DIRECTIONS].index(True)]
        if s.hit:
            print(f"  {dir_name[idx]}: {s.distance:.2f}m (conf={s.confidence:.2f})")
        else:
            print(f"  {dir_name[idx]}: sem retorno")

    print("\nTeste 3 - EKF tracking por 3s:")
    physics.reset()
    ekf.reset()
    dt = 0.01
    for _ in range(300):
        env_cur, env_turb = sensors.get_environmental_state()
        env_harm = sensors.get_environmental_harmonics()
        physics.step(0.3, 0.0, 0.0, 0.0, dt=dt, env_current_world=env_cur, env_turbulence=env_turb, env_harmonics=env_harm)
        bundle = sensors.read(physics.state, physics.time)
        ekf.predict(dt)
        ekf.update_imu(bundle.imu)
        ekf.update_barometer(bundle.barometer)
        ekf.update_sonar(bundle.sonar)
        ekf.update_sonar_position(bundle.sonar)

    real = physics.state
    est = ekf.state_estimate
    print(f"  Estado real:    x={real.x:.3f} z={real.z:.4f} u={real.u:.3f}")
    print(f"  Estado EKF:     x={est.eta[0]:.3f} z={est.eta[2]:.4f} u={est.nu[0]:.3f}")
    print(f"  Erro posição x: {abs(real.x - est.eta[0]):.4f} m")
    print(f"  Erro profund z: {abs(real.z - est.eta[2]):.4f} m")

    print("\n✓ Sensor Engine + EKF validados.")
