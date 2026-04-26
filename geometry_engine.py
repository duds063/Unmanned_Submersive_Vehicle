"""
USV Digital Twin — Módulo 1: Geometry Engine
=============================================
Calcula coeficientes hidrodinâmicos e parâmetros de mesh 3D
a partir das dimensões do casco cilíndrico com pontas de Von Kármán.

Referências:
    - Fossen, T.I. (2011). Handbook of Marine Craft Hydrodynamics and Motion Control
    - Hoerner, S.F. (1965). Fluid-Dynamic Drag
    - Von Kármán ogive: razão ponta/diâmetro fixada em 2.5

Convenção de eixos (NED — North-East-Down):
    x: surge  (frontal)
    y: sway   (lateral)
    z: heave  (vertical)
    p: roll   (rotação em x)
    q: pitch  (rotação em y)
    r: yaw    (rotação em z)
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict


# ─────────────────────────────────────────────
# CONSTANTES FÍSICAS
# ─────────────────────────────────────────────
RHO_FRESHWATER = 1000.0   # kg/m³ — densidade água doce a 4°C
VON_KARMAN_RATIO = 2.5    # razão comprimento_ponta / diâmetro — fixada


# ─────────────────────────────────────────────
# ESTRUTURAS DE DADOS
# ─────────────────────────────────────────────

@dataclass
class HullGeometry:
    """Parâmetros geométricos do casco."""
    L: float                    # comprimento total (m)
    D: float                    # diâmetro (m)
    R: float = field(init=False)           # raio (m)
    L_vk: float = field(init=False)        # comprimento de cada ponta Von Kármán (m)
    L_cyl: float = field(init=False)       # comprimento do corpo cilíndrico central (m)
    L_D_ratio: float = field(init=False)   # razão de aspecto
    A_frontal: float = field(init=False)   # área frontal projetada (m²)
    A_lateral: float = field(init=False)   # área lateral projetada (m²)
    volume: float = field(init=False)      # volume deslocado (m³)
    mass_hull: float = field(init=False)   # massa estimada do casco (kg) — aço inox

    def __post_init__(self):
        self.R       = self.D / 2
        self.L_vk    = VON_KARMAN_RATIO * self.D
        self.L_cyl   = self.L - 2 * self.L_vk
        self.L_D_ratio = self.L / self.D

        # áreas projetadas
        self.A_frontal = np.pi * self.R**2
        self.A_lateral = self.L * self.D

        # volume: dois cones Von Kármán + cilindro central
        # Von Kármán approximado como cone para volume
        vol_vk   = (2/3) * np.pi * self.R**2 * self.L_vk  # dois cones
        vol_cyl  = np.pi * self.R**2 * self.L_cyl
        self.volume = vol_vk + vol_cyl

        # massa estimada — casco de aço inox 1mm de espessura
        rho_steel   = 7900.0  # kg/m³
        thickness   = 0.001   # m
        area_total  = np.pi * self.D * self.L  # superfície lateral aproximada
        self.mass_hull = rho_steel * area_total * thickness

    def validate(self) -> None:
        """Valida que a geometria é fisicamente consistente."""
        if self.L_cyl <= 0:
            raise ValueError(
                f"Corpo cilíndrico tem comprimento negativo ({self.L_cyl:.3f}m). "
                f"Aumente L ou diminua D. "
                f"Mínimo L = {2 * self.L_vk:.3f}m para D={self.D}m."
            )
        if self.L_D_ratio < 3:
            print(f"⚠️  Razão L/D = {self.L_D_ratio:.1f} é baixa. "
                  f"Veículos subaquáticos eficientes tipicamente usam L/D > 5.")
        if self.L_D_ratio > 15:
            print(f"⚠️  Razão L/D = {self.L_D_ratio:.1f} é muito alta. "
                  f"Rigidez estrutural pode ser problema.")


@dataclass
class HydrodynamicCoefficients:
    """Coeficientes hidrodinâmicos no referencial do corpo."""

    # Coeficientes de arrasto quadrático (drag) — N/(m/s)²
    X_uu: float   # surge
    Y_vv: float   # sway
    Z_ww: float   # heave
    K_pp: float   # roll
    M_qq: float   # pitch
    N_rr: float   # yaw

    # Coeficientes de arrasto linear (skin friction) — N/(m/s)
    X_u: float
    Y_v: float
    Z_w: float
    K_p: float
    M_q: float
    N_r: float

    # Massa adicionada — kg e kg·m²
    X_udot: float   # surge
    Y_vdot: float   # sway
    Z_wdot: float   # heave
    K_pdot: float   # roll
    M_qdot: float   # pitch
    N_rdot: float   # yaw

    def to_drag_matrix_quadratic(self) -> np.ndarray:
        """Retorna matriz diagonal de arrasto quadrático 6x6."""
        return np.diag([
            self.X_uu, self.Y_vv, self.Z_ww,
            self.K_pp, self.M_qq, self.N_rr
        ])

    def to_drag_matrix_linear(self) -> np.ndarray:
        """Retorna matriz diagonal de arrasto linear 6x6."""
        return np.diag([
            self.X_u, self.Y_v, self.Z_w,
            self.K_p, self.M_q, self.N_r
        ])

    def to_added_mass_matrix(self) -> np.ndarray:
        """Retorna matriz de massa adicionada 6x6."""
        return np.diag([
            self.X_udot, self.Y_vdot, self.Z_wdot,
            self.K_pdot, self.M_qdot, self.N_rdot
        ])


@dataclass
class MeshParameters:
    """Parâmetros pro Three.js renderizar o casco."""
    L: float
    D: float
    R: float
    L_vk: float
    L_cyl: float
    von_karman_ratio: float
    segments_radial: int = 32    # resolução circular do mesh
    segments_axial: int  = 64    # resolução axial do mesh


# ─────────────────────────────────────────────
# GEOMETRY ENGINE
# ─────────────────────────────────────────────

class GeometryEngine:
    """
    Calcula coeficientes hidrodinâmicos e parâmetros de mesh
    a partir das dimensões do casco.

    Uso:
        engine = GeometryEngine(L=0.8, D=0.1)
        coeffs = engine.coefficients
        mesh   = engine.mesh_params
    """

    def __init__(
        self,
        L: float,
        D: float,
        rho: float = RHO_FRESHWATER,
        roll_damping_factor: float = 5.0,
    ):
        self.rho  = rho
        self.hull = HullGeometry(L=L, D=D)
        self.hull.validate()
        self.roll_damping_factor = max(1.0, float(roll_damping_factor))

        self._coefficients: HydrodynamicCoefficients = None
        self._mesh_params:  MeshParameters           = None

        self._compute()

    # ─── Propriedades públicas ───────────────

    @property
    def coefficients(self) -> HydrodynamicCoefficients:
        return self._coefficients

    @property
    def mesh_params(self) -> MeshParameters:
        return self._mesh_params

    @property
    def hull_geometry(self) -> HullGeometry:
        return self.hull

    # ─── Computação interna ──────────────────

    def _compute(self) -> None:
        self._coefficients = self._compute_hydrodynamics()
        self._mesh_params  = self._compute_mesh()

    def _cd_frontal(self) -> float:
        """
        Coeficiente de arrasto frontal — Von Kármán ogive.
        Baseado em teoria de corpos esbeltos + correção empírica.
        Fossen (2011) eq. 7.45
        """
        LD = self.hull.L_D_ratio
        return 0.04 + 0.04 / (LD ** 2)

    def _cd_lateral(self) -> float:
        """
        Coeficiente de arrasto lateral — cilindro infinito corrigido
        por efeitos de extremidade.
        Hoerner (1965) — cilindro finito com razão L/D variável.
        """
        LD = self.hull.L_D_ratio
        return 1.1 - 0.3 / LD

    def _compute_hydrodynamics(self) -> HydrodynamicCoefficients:
        h   = self.hull
        rho = self.rho

        Cd_f = self._cd_frontal()
        Cd_l = self._cd_lateral()

        # ── Arrasto quadrático ────────────────────────────────────────
        # F = 0.5 * rho * Cd * A * v²
        X_uu = 0.5 * rho * Cd_f * h.A_frontal
        Y_vv = 0.5 * rho * Cd_l * h.A_lateral
        Z_ww = Y_vv  # simetria radial do cilindro

        # Arrasto rotacional — momento de arrasto por rotação
        # M = 0.5 * rho * Cd * A * arm² * omega²
        # Roll usa área lateral e braço radial do casco; Cd empírico mais alto.
        K_pp = 0.5 * rho * 2.0 * h.A_lateral * h.R ** 2
        M_qq = 0.5 * rho * Cd_l * h.A_lateral * (h.L / 6) ** 2
        N_rr = M_qq  # simetria pitch/yaw

        # Modelo simplificado de aletas estabilizadoras: aumenta damping em roll.
        K_pp *= self.roll_damping_factor

        # ── Arrasto linear (skin friction) ────────────────────────────
        # Estimativa baseada em número de Reynolds típico Re~10^5
        # Cf ~ 0.074 / Re^0.2 — Prandtl turbulento
        # Simplificado: 10% do arrasto quadrático como referência
        X_u = 0.1 * X_uu
        Y_v = 0.1 * Y_vv
        Z_w = 0.1 * Z_ww
        K_p = 0.1 * K_pp
        M_q = 0.1 * M_qq
        N_r = 0.1 * N_rr

        # ── Massa adicionada ──────────────────────────────────────────
        # Teoria de corpos esbeltos — Lamb (1932)
        # Surge: massa adicionada pequena pra corpo esbelto
        # Sway/heave: massa adicionada ≈ massa de fluido deslocado
        m_fluid = rho * h.volume

        X_udot = 0.1  * m_fluid   # esbelto — massa adicionada frontal baixa
        Y_vdot = 1.0  * m_fluid   # lateral — massa adicionada alta
        Z_wdot = Y_vdot

        # Momentos de inércia adicionados
        I_ref  = m_fluid * h.R ** 2
        K_pdot = 0.05 * I_ref    # roll — cilindro simétrico, baixo
        M_qdot = 0.20 * I_ref * (h.L / h.D) ** 2   # pitch — comprimento aumenta
        N_rdot = M_qdot

        return HydrodynamicCoefficients(
            X_uu=X_uu, Y_vv=Y_vv, Z_ww=Z_ww,
            K_pp=K_pp, M_qq=M_qq, N_rr=N_rr,
            X_u=X_u,   Y_v=Y_v,   Z_w=Z_w,
            K_p=K_p,   M_q=M_q,   N_r=N_r,
            X_udot=X_udot, Y_vdot=Y_vdot, Z_wdot=Z_wdot,
            K_pdot=K_pdot, M_qdot=M_qdot, N_rdot=N_rdot,
        )

    def _compute_mesh(self) -> MeshParameters:
        h = self.hull
        return MeshParameters(
            L=h.L,
            D=h.D,
            R=h.R,
            L_vk=h.L_vk,
            L_cyl=h.L_cyl,
            von_karman_ratio=VON_KARMAN_RATIO,
        )

    def to_dict(self) -> Dict:
        """Serializa tudo pra JSON — usado pelo Flask/WebSocket."""
        c = self._coefficients
        m = self._mesh_params
        h = self.hull

        return {
            "geometry": {
                "L": h.L,
                "D": h.D,
                "R": h.R,
                "L_vk": h.L_vk,
                "L_cyl": h.L_cyl,
                "L_D_ratio": h.L_D_ratio,
                "A_frontal": h.A_frontal,
                "A_lateral": h.A_lateral,
                "volume": h.volume,
                "mass_hull": h.mass_hull,
            },
            "drag_quadratic": {
                "X_uu": c.X_uu, "Y_vv": c.Y_vv, "Z_ww": c.Z_ww,
                "K_pp": c.K_pp, "M_qq": c.M_qq, "N_rr": c.N_rr,
            },
            "drag_linear": {
                "X_u": c.X_u, "Y_v": c.Y_v, "Z_w": c.Z_w,
                "K_p": c.K_p, "M_q": c.M_q, "N_r": c.N_r,
            },
            "added_mass": {
                "X_udot": c.X_udot, "Y_vdot": c.Y_vdot, "Z_wdot": c.Z_wdot,
                "K_pdot": c.K_pdot, "M_qdot": c.M_qdot, "N_rdot": c.N_rdot,
            },
            "mesh": {
                "L": m.L, "D": m.D, "R": m.R,
                "L_vk": m.L_vk,
                "L_cyl": m.L_cyl,
                "von_karman_ratio": m.von_karman_ratio,
                "segments_radial": m.segments_radial,
                "segments_axial": m.segments_axial,
            },
            "fluid": {
                "rho": self.rho,
                "name": "freshwater",
            }
        }

    def summary(self) -> str:
        """Print legível dos coeficientes calculados."""
        c = self._coefficients
        h = self.hull
        return f"""
╔══════════════════════════════════════════════════════╗
║           USV GEOMETRY ENGINE — SUMMARY              ║
╠══════════════════════════════════════════════════════╣
║  Casco                                               ║
║    L = {h.L:.3f} m    D = {h.D:.3f} m    L/D = {h.L_D_ratio:.1f}        ║
║    Ponta Von Kármán: {h.L_vk:.3f} m (ratio={VON_KARMAN_RATIO})         ║
║    Corpo cilíndrico: {h.L_cyl:.3f} m                    ║
║    Volume: {h.volume*1e3:.2f} L    Massa casco: {h.mass_hull:.2f} kg    ║
╠══════════════════════════════════════════════════════╣
║  Arrasto Quadrático                                  ║
║    Surge X_uu = {c.X_uu:.4f} N/(m/s)²                 ║
║    Sway  Y_vv = {c.Y_vv:.4f} N/(m/s)²                 ║
║    Heave Z_ww = {c.Z_ww:.4f} N/(m/s)²                 ║
║    Roll  K_pp = {c.K_pp:.6f} N/(m/s)²               ║
║    Pitch M_qq = {c.M_qq:.4f} N/(m/s)²                 ║
║    Yaw   N_rr = {c.N_rr:.4f} N/(m/s)²                 ║
╠══════════════════════════════════════════════════════╣
║  Massa Adicionada                                    ║
║    Surge X_udot = {c.X_udot:.4f} kg                    ║
║    Sway  Y_vdot = {c.Y_vdot:.4f} kg                    ║
║    Pitch M_qdot = {c.M_qdot:.4f} kg·m²                 ║
╚══════════════════════════════════════════════════════╝"""


# ─────────────────────────────────────────────
# TESTES RÁPIDOS
# ─────────────────────────────────────────────

if __name__ == "__main__":
    import json

    # Dimensões de referência — torpedo pequeno
    engine = GeometryEngine(L=0.8, D=0.1)
    print(engine.summary())

    # Serialização pra JSON
    data = engine.to_dict()
    print("\nJSON output (primeiros campos):")
    print(json.dumps({k: data[k] for k in ['geometry', 'drag_quadratic']}, indent=2))

    # Teste de validação — geometria inválida
    print("\nTestando geometria inválida (L/D muito baixo):")
    try:
        engine_bad = GeometryEngine(L=0.3, D=0.1)
    except ValueError as e:
        print(f"✓ Erro capturado: {e}")

    # Comparação de arrasto frontal vs lateral
    ratio = engine.coefficients.Y_vv / engine.coefficients.X_uu
    print(f"\nRazão arrasto lateral/frontal: {ratio:.1f}x")
    print(f"(Esperado: ~10-15x para Von Kármán com L/D={engine.hull.L_D_ratio:.1f})")
