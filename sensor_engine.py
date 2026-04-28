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
class EKFState:
    """Estado estimado pelo EKF."""
    eta:       np.ndarray           # [x,y,z,φ,θ,ψ] — posição/orientação
    nu:        np.ndarray           # [u,v,w,p,q,r] — velocidades
    P:         np.ndarray           # covariância 12x12
    timestamp: float

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
        self.wave_hs = max(0.01, float(wave_hs))  # Significant Wave Height (m)
        self.wave_spectrum = 'jonswap'  # or 'pm' for Pierson-Moskowitz
        self.wave_amp_scale = 0.02
        self.wave_hs = 0.2
        self.wave_spectrum = 'jonswap'
        self._harmonic_freqs = None
        self._harmonic_phases = None
        self._harmonic_dirs = None
        self._harmonic_amps = None

    # ─── Interface pública ───────────────────

    def read(self, state: VehicleState, time: float) -> SensorBundle:
        """Lê todos os sensores dado o estado físico real."""
        self._update_environmental_state(time)

        imu       = self._read_imu(state, time)
        sonar     = self._read_sonar(state, time)
        barometer = self._read_barometer(state, time)

        return SensorBundle(
            imu=imu,
            sonar=sonar,
            barometer=barometer,
            timestamp=time,
        )

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
            self.wave_hs = max(0.01, float(wave_hs))

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

        # drift do bias
        self._accel_bias += self.rng.normal(
            0, IMU_ACCEL_BIAS_STD * self._bias_drift_rate, 3)
        self._gyro_bias  += self.rng.normal(
            0, IMU_GYRO_BIAS_STD  * self._bias_drift_rate, 3)

        # ruído branco
        accel_noise = self.rng.normal(0, IMU_ACCEL_NOISE_STD * self.noise_scale, 3)
        gyro_noise  = self.rng.normal(0, IMU_GYRO_NOISE_STD  * self.noise_scale, 3)

        # componente não gaussiana (maresia/turbulência)
        if self.enable_rayleigh and self.environment_scale > 0.0:
            accel_noise += self._signed_rayleigh_noise(size=3) * 0.2 * self.environment_scale
            gyro_noise  += self._signed_rayleigh_noise(size=3) * 0.01 * self.environment_scale

        # leitura ruidosa
        accel_meas = linear_accel_body + g_body + self._accel_bias + accel_noise
        gyro_meas  = true_gyro   + self._gyro_bias  + gyro_noise

        return IMUReading(accel=accel_meas, gyro=gyro_meas, timestamp=time)

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

    def __init__(self, physics: PhysicsEngine):
        self.physics = physics

        # covariâncias de processo — quanta incerteza adicionamos por step
        self.Q = np.diag([
            0.01, 0.01, 0.01,   # posição xyz
            0.005, 0.005, 0.005, # orientação euler
            0.1,  0.1,  0.1,    # velocidade linear
            0.05, 0.05, 0.05,   # velocidade angular
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

    @property
    def state_estimate(self) -> EKFState:
        return EKFState(
            eta=self._x[:6].copy(),
            nu=self._x[6:].copy(),
            P=self._P.copy(),
            timestamp=self._time,
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

    def update_sonar(self, readings: List[SonarReading]) -> None:
        """Atualiza com leituras do sonar — só usa hits válidos."""
        hits = [r for r in readings if r.hit]
        if not hits:
            return

        # Usa apenas o sonar apontado para cima como medida de profundidade.
        # O sonar para baixo enxerga o fundo e não deve ser confundido com z.
        for reading in hits:
            if np.allclose(reading.direction, [0, 0, -1]):
                z = np.array([reading.distance])
                h = np.array([self._x[2]])  # z estimado
                H = np.zeros((1, self.DIM_STATE))
                H[0, 2] = 1.0
                self._update(z, h, H, self.R_sonar[:1, :1])

    def update_barometer(self, reading: BarometerReading) -> None:
        """Atualiza profundidade z com barômetro."""
        z = np.array([reading.depth])
        h = np.array([self._x[2]])
        H = np.zeros((1, self.DIM_STATE))
        H[0, 2] = 1.0
        self._update(z, h, H, self.R_baro)

    # ─── EKF internals ───────────────────────

    def _f(self, x: np.ndarray, dt: float) -> np.ndarray:
        """Modelo de processo — cinemática de corpo rígido simplificada."""
        eta = x[:6]
        nu  = x[6:]

        J = self._jacobian_eta(eta)
        eta_dot = J @ nu

        x_new = x.copy()
        x_new[:6] += eta_dot * dt
        # velocidades mantidas constantes na predição (sem modelo de força)
        # o Physics Engine é a fonte de verdade — EKF só filtra

        return x_new

    def _compute_F(self, x: np.ndarray, dt: float) -> np.ndarray:
        """Jacobiana do modelo de processo 12x12."""
        F = np.eye(self.DIM_STATE)
        eta = x[:6]
        J   = self._jacobian_eta(eta)
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
        # aceleração depende de phi (idx 3) e theta (idx 4)
        phi, theta = x[3], x[4]
        H[0, 4] = -G * np.cos(theta)
        H[1, 3] =  G * np.cos(theta) * np.cos(phi)
        H[1, 4] = -G * np.sin(theta) * np.sin(phi)
        H[2, 3] = -G * np.cos(theta) * np.sin(phi)
        H[2, 4] = -G * np.sin(theta) * np.cos(phi)
        # giroscópio observa diretamente p,q,r (idx 9,10,11)
        H[3, 9]  = 1.0
        H[4, 10] = 1.0
        H[5, 11] = 1.0
        return H

    def _update(
        self,
        z: np.ndarray,
        h: np.ndarray,
        H: np.ndarray,
        R: np.ndarray
    ) -> None:
        """Etapa de atualização EKF padrão."""
        innov = z - h                           # inovação
        S     = H @ self._P @ H.T + R          # covariância da inovação
        K     = self._P @ H.T @ np.linalg.inv(S)  # ganho de Kalman
        self._x = self._x + K @ innov
        self._P = (np.eye(self.DIM_STATE) - K @ H) @ self._P

    def _jacobian_eta(self, eta: np.ndarray) -> np.ndarray:
        """Bloco J1 da Jacobiana — transforma nu em eta_dot."""
        phi, theta, psi = eta[3], eta[4], eta[5]
        R = SensorEngine._rotation_matrix(phi, theta, psi)
        cphi=np.cos(phi); sphi=np.sin(phi)
        cth=np.cos(theta)
        tth=np.tan(theta)
        T = np.array([
            [1, sphi*tth,  cphi*tth],
            [0, cphi,     -sphi    ],
            [0, sphi/cth,  cphi/cth]
        ])
        J = np.zeros((6, 6))
        J[:3, :3] = R
        J[3:, 3:] = T
        return J

    def reset(self, initial_state: Optional[np.ndarray] = None) -> None:
        self._x = initial_state if initial_state is not None else np.zeros(self.DIM_STATE)
        self._P = self._initial_covariance()
        self._time = 0.0
        self._last_imu_timestamp = None


# ─────────────────────────────────────────────
# TESTES
# ─────────────────────────────────────────────

if __name__ == "__main__":
    import json
    from geometry_engine import GeometryEngine

    print("Inicializando Sensor Engine + EKF...")

    # setup
    geo     = GeometryEngine(L=0.8, D=0.1)
    physics = PhysicsEngine(geo, max_thruster_force=10.0)

    env     = Environment(pool_depth=5.0, pool_radius=30.0)   # pool de 5m — fundo atingível
    env.add_sphere(np.array([5.0, 0.0, 3.0]), radius=1.0)   # obstáculo esférico

    sensors = SensorEngine(env, noise_scale=1.0)
    ekf     = ExtendedKalmanFilter(physics)

    # Teste 1 — leitura de sensores em repouso
    print("\nTeste 1 — Sensores em repouso:")
    bundle = sensors.read(physics.state, 0.0)
    print(f"  IMU accel: {bundle.imu.accel.round(3)}")
    print(f"  IMU gyro:  {bundle.imu.gyro.round(4)}")
    print(f"  Barômetro: {bundle.barometer.depth:.3f} m")
    print(f"  Sonar hits: {sum(1 for s in bundle.sonar if s.hit)}/6")

    # Teste 2 — sonar detecta fundo
    print("\nTeste 2 — Sonar detecta fundo a 10m:")
    state_deep = VehicleState(z=2.0)  # 2m de profundidade
    bundle2 = sensors.read(state_deep, 0.1)
    for s in bundle2.sonar:
        dir_name = ['frente','trás','estibordo','bombordo','baixo','cima']
        idx = list(range(6))[
            [np.allclose(s.direction, d) for d in SensorEngine.SONAR_DIRECTIONS].index(True)
        ]
        if s.hit:
            print(f"  {dir_name[idx]}: {s.distance:.2f}m (conf={s.confidence:.2f})")
        else:
            print(f"  {dir_name[idx]}: sem retorno")

    # Teste 3 — EKF converge pro estado real
    print("\nTeste 3 — EKF tracking por 3s:")
    physics.reset()
    ekf.reset()
    dt = 0.01

    for i in range(300):
        env_cur, env_turb = sensors.get_environmental_state()
        env_harm = sensors.get_environmental_harmonics()
        physics.step(0.3, 0.0, 0.0, 0.0, dt=dt, env_current_world=env_cur, env_turbulence=env_turb, env_harmonics=env_harm)
        bundle = sensors.read(physics.state, physics.time)

        ekf.predict(dt)
        ekf.update_imu(bundle.imu)
        ekf.update_barometer(bundle.barometer)
        ekf.update_sonar(bundle.sonar)

    real  = physics.state
    est   = ekf.state_estimate

    print(f"  Estado real:    x={real.x:.3f} z={real.z:.4f} u={real.u:.3f}")
    print(f"  Estado EKF:     x={est.eta[0]:.3f} z={est.eta[2]:.4f} u={est.nu[0]:.3f}")
    print(f"  Erro posição x: {abs(real.x - est.eta[0]):.4f} m")
    print(f"  Erro profund z: {abs(real.z - est.eta[2]):.4f} m")

    print("\n✓ Sensor Engine + EKF validados.")
