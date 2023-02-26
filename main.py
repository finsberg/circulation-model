"""Attempt to implement model in Circulation model
in https://doi.org/10.1016/j.jcp.2022.111083
"""
from dataclasses import dataclass, field
from typing import Sequence
import numpy as np

from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import matplotlib.animation as animation

import regazzoni2022


def phi(t, t_C, t_R, T_C, T_R, T_HB):

    if 0 <= (x := (t - t_C) % T_HB) < T_C:
        return 0.5 * (1 - np.cos((np.pi / T_C) * x))

    if 0 <= (x := (t - t_R) % T_HB) < T_R:
        return 0.5 * (1 + np.cos((np.pi / T_R) * x))

    return 0


@dataclass
class Chamber:
    E_pass: float
    E_act_max: float
    t_C: float
    T_C: float
    T_R: float
    V_0: float
    T_HB: float = 1.0
    name: str = "Chamber"
    _V: float = field(default=0.0, repr=False)

    @property
    def t_R(self) -> float:
        return self.t_C + self.T_C

    def E(self, t: float) -> float:
        return self.E_pass + self.E_act_max * phi(
            t=t, t_C=self.t_C, t_R=self.t_R, T_C=self.T_C, T_R=self.T_R, T_HB=self.T_HB
        )

    def V(self, t) -> float:
        return self._V

    @property
    def p_ext(self) -> float:
        return 0

    def p(self, t) -> float:
        return self.p_ext + self.E(t) * (self.V(t) - self.V_0)


@dataclass
class Circulation:
    R: float
    C: float
    L: float
    name: str
    _Q: float = field(default=0.0, repr=False)
    _p: float = field(default=0.0, repr=False)

    def Q(self, t) -> float:
        return self._Q

    def p(self, t) -> float:
        return self._p


@dataclass
class Valve:
    fst: Chamber | Circulation
    snd: Chamber | Circulation
    R_min: float
    R_max: float
    name: str = "Valve"

    def p1(self, t: float) -> float:
        return self.fst.p(t)

    def p2(self, t: float) -> float:
        return self.snd.p(t)

    def R(self, t: float) -> float:
        return self.R_min if self.p1(t) < self.p2(t) else self.R_max

    def Q(self, t: float) -> float:
        return (self.p1(t) - self.p2(t)) / self.R(t)


@dataclass
class System:
    LV: Chamber
    RV: Chamber
    LA: Chamber
    RA: Chamber
    AR_SYS: Circulation
    AR_PUL: Circulation
    VEN_SYS: Circulation
    VEN_PUL: Circulation
    MV: Valve
    AV: Valve
    TV: Valve
    PV: Valve

    @property
    def num_c1(self) -> int:
        return 12

    @property
    def num_c2(self) -> int:
        return 8

    @property
    def c1(self) -> np.ndarray:
        return np.array(
            [
                self.LA._V,
                self.LV._V,
                self.RA._V,
                self.RV._V,
                self.AR_SYS._p,
                self.VEN_SYS._p,
                self.AR_PUL._p,
                self.VEN_PUL._p,
                self.AR_SYS._Q,
                self.VEN_SYS._Q,
                self.AR_PUL._Q,
                self.VEN_PUL._Q,
            ]
        )

    @c1.setter
    def c1(self, y: Sequence) -> None:
        if not len(y) == 12:
            raise ValueError("State must have length 12")
        self.LA._V = y[0]
        self.LV._V = y[1]
        self.RA._V = y[2]
        self.RV._V = y[3]
        self.AR_SYS._p = y[4]
        self.VEN_SYS._p = y[5]
        self.AR_PUL._p = y[6]
        self.VEN_PUL._p = y[7]
        self.AR_SYS._Q = y[8]
        self.VEN_SYS._Q = y[9]
        self.AR_PUL._Q = y[10]
        self.VEN_PUL._Q = y[11]

    def c2(self, t, c1) -> np.ndarray:
        self.c1 = c1
        return np.array(
            [
                self.LV.p(t),
                self.LA.p(t),
                self.RV.p(t),
                self.RA.p(t),
                self.MV.Q(t),
                self.AV.Q(t),
                self.TV.Q(t),
                self.PV.Q(t),
            ]
        )

    def __call__(self, t, y):

        self.c1 = y
        values = np.zeros(12)
        # V_LA
        values[0] = self.VEN_PUL.Q(t) - self.MV.Q(t)
        # V_LV
        values[1] = self.MV.Q(t) - self.AV.Q(t)
        # V_RA
        values[2] = self.VEN_SYS.Q(t) - self.TV.Q(t)
        # V_RV
        values[3] = self.TV.Q(t) - self.PV.Q(t)
        # p_AR_SYS
        values[4] = (1 / self.AR_SYS.C) * (self.AV.Q(t) - self.AR_SYS.Q(t))
        # p_VEN_SYS
        values[5] = (1 / self.VEN_SYS.C) * (self.AR_SYS.Q(t) - self.VEN_SYS.Q(t))
        # p_AR_PUL
        values[6] = (1 / self.AR_PUL.C) * (self.PV.Q(t) - self.AR_PUL.Q(t))
        # p_VEN_PUL
        values[7] = (1 / self.VEN_PUL.C) * (self.AR_PUL.Q(t) - self.VEN_PUL.Q(t))
        # Q_SYS_AR
        values[8] = (self.AR_SYS.R / self.AR_SYS.L) * (
            -self.AR_SYS.Q(t) - ((self.VEN_SYS.p(t) - self.AR_SYS.p(t)) / self.AR_SYS.R)
        )
        # Q_SYS_VEN
        values[9] = (self.VEN_SYS.R / self.VEN_SYS.L) * (
            -self.VEN_SYS.Q(t) - ((self.RA.p(t) - self.VEN_SYS.p(t)) / self.VEN_SYS.R)
        )
        # Q_PUL_AR
        values[10] = (self.AR_PUL.R / self.AR_PUL.L) * (
            -self.AR_PUL.Q(t) - ((self.VEN_PUL.p(t) - self.AR_PUL.p(t)) / self.AR_PUL.R)
        )
        # Q_PUL_VEN
        values[11] = (self.VEN_PUL.R / self.VEN_PUL.L) * (
            -self.VEN_PUL.Q(t) - ((self.LA.p(t) - self.VEN_PUL.p(t)) / self.AR_PUL.R)
        )
        return values


def main():

    R_min = 0.0075
    R_max = 75_000
    T_HB = 0.8

    params = regazzoni2022.parameters

    chambers = {}
    for name in ["LV", "RV", "RA", "LA"]:
        values: regazzoni2022.Chamber = getattr(params, name)
        chambers[name] = Chamber(
            E_pass=values.E_pass.m,
            E_act_max=values.E_act_max.m,
            V_0=values.V_0.m,
            t_C=values.t_C.m,
            T_C=values.T_C.m,
            T_R=values.T_R.m,
            T_HB=T_HB,
            name=name,
        )
        print(chambers[name])

    circs = {}
    for name in ["AR_SYS", "AR_PUL", "VEN_SYS", "VEN_PUL"]:
        values: regazzoni2022.Circulation = getattr(params, name)
        circs[name] = Circulation(R=values.R.m, C=values.C.m, L=values.L.m, name=name)
        print(circs[name])

    MV = Valve(
        fst=chambers["LA"],
        snd=chambers["LV"],
        R_min=R_min,
        R_max=R_max,
        name="Mitral",
    )
    AV = Valve(
        fst=chambers["LV"],
        snd=circs["AR_SYS"],
        R_min=R_min,
        R_max=R_max,
        name="Aortic",
    )
    TV = Valve(
        fst=chambers["RA"],
        snd=chambers["RV"],
        R_min=R_min,
        R_max=R_max,
        name="Tricuspid",
    )
    PV = Valve(
        fst=chambers["RV"],
        snd=circs["VEN_PUL"],
        R_min=R_min,
        R_max=R_max,
        name="Pulmonary",
    )

    system = System(
        **chambers,
        **circs,
        MV=MV,
        AV=AV,
        TV=TV,
        PV=PV,
    )

    y0 = np.array(
        [
            65.0,  # V_LA
            120.0,  # V_LV
            65.0,  # V_RA
            145.0,  # V_RV
            80.0,  # p_AR_SYS
            30.0,  # p_VEN_SYS
            35.0,  # p_AR_PUL
            24.0,  # p_VEN_PUL
            0.0,  # Q_AR_SYS
            0.0,  # Q_VEN_SYS
            0.0,  # Q_AR_PUL
            0.0,  # Q_VEN_PUL
        ]
    )
    res = solve_ivp(system, t_span=(0.0, 100.0), y0=y0, method="Radau")

    t = res.t
    c1 = res.y
    c2 = np.zeros((system.num_c2, t.size))
    for i, ti in enumerate(t):
        c2[:, i] = system.c2(ti, c1[:, i])

    V_LV = c1[1, :]
    V_RV = c1[3, :]

    P_LV = c2[0, :]
    P_RV = c2[2, :]

    animate = True

    if animate:
        fig, ax = plt.subplots(1, 2)

        def animate(i):
            start = max(i - 100, 0)
            ax[0].clear()
            ax[0].set_xlim(0, 70)
            ax[0].set_ylim(0, 40)
            ax[0].text(0.02, 0.90, f"time = {t[i]:.1f}", transform=ax[0].transAxes)
            ax[0].plot(V_LV[start:i], P_LV[start:i])
            ax[1].clear()
            ax[1].set_xlim(70, 150)
            ax[1].set_ylim(0, 90)
            ax[1].plot(V_RV[start:i], P_RV[start:i])

        ani = animation.FuncAnimation(fig, animate, interval=1, frames=len(V_LV))
        plt.show()

    fig, ax = plt.subplots(1, 2)

    ax[0].plot(V_LV, P_LV)
    ax[0].set_title("LV")
    ax[0].set_xlabel("Volume [ml]")
    ax[0].set_ylabel("Pressure [mmHg]")

    ax[1].plot(V_RV, P_RV)
    ax[1].set_title("RV")
    ax[1].set_xlabel("Volume [ml]")
    ax[1].set_ylabel("Pressure [mmHg]")
    plt.show()


if __name__ == "__main__":
    main()
