"""Attempt to implement model in Circulation model 
in https://doi.org/10.1016/j.jcp.2022.111083
"""
from dataclasses import dataclass
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import matplotlib.animation as animation


def phi(t, t_C, t_R, T_C, T_R, T_HB):

    if 0 <= (x := (t - t_C) % T_HB) < T_C:
        return 0.5 * (1 - np.cos((np.pi / T_C) * x))

    if 0 <= (x := (t - t_R) % T_HB) < T_R:
        return 0.5 * (1 - np.cos((np.pi / T_R) * x))

    return 0


@dataclass
class Chamber:
    E_pass: float
    E_act_max: float
    t_C: float
    T_C: float
    T_R: float
    V_0: float
    T_HB: float = 0.8
    _V: float = 0.0

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
    _Q: float = 0.0
    _p: float = 0.0

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

    def _update_states(self, y):
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

    def __call__(self, t, y):

        self._update_states(y)
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

    LV = Chamber(E_pass=0.05, E_act_max=0.55, V_0=10.0, t_C=0.0, T_C=0.272, T_R=0.12)
    RV = Chamber(E_pass=0.05, E_act_max=0.55, V_0=10.0, t_C=0.0, T_C=0.272, T_R=0.12)
    LA = Chamber(E_pass=0.09, E_act_max=0.07, V_0=4.0, t_C=0.6, T_C=0.104, T_R=0.68)
    RA = Chamber(E_pass=0.07, E_act_max=0.06, V_0=4.0, t_C=0.56, T_C=0.064, T_R=0.64)

    AR_SYS = Circulation(R=0.8, C=1.2, L=5e-3, name="Arterial (systemic)")
    AR_PUL = Circulation(R=0.1625, C=10.0, L=5e-4, name="Arterial (pulmonary)")
    VEN_SYS = Circulation(R=0.26, C=60.0, L=5e-4, name="Venous (systemic)")
    VEN_PUL = Circulation(R=0.1625, C=16.0, L=5e-4, name="Venous (pulmonary)")

    MV = Valve(fst=LA, snd=LV, R_min=R_min, R_max=R_max, name="Mitral")
    AV = Valve(fst=LV, snd=AR_SYS, R_min=R_min, R_max=R_max, name="Aortic")
    TV = Valve(fst=RA, snd=RV, R_min=R_min, R_max=R_max, name="Tricuspid")
    PV = Valve(fst=RV, snd=VEN_PUL, R_min=R_min, R_max=R_max, name="Pulmonary")

    system = System(
        LV=LV,
        RV=RV,
        LA=LA,
        RA=RA,
        AR_SYS=AR_SYS,
        AR_PUL=AR_PUL,
        VEN_SYS=VEN_SYS,
        VEN_PUL=VEN_PUL,
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
    res = solve_ivp(system, t_span=(0.0, 10.0), y0=y0)
    V_LV = res.y[1, :]
    t = res.t
    P_LV = [LV.p(ti) for ti in t]
    fig, ax = plt.subplots()

    def animate(i):
        ax.clear()
        ax.set_xlim(30, 190)
        ax.set_ylim(0, 20)
        ax.plot(V_LV[:i], P_LV[:i])

    ani = animation.FuncAnimation(fig, animate, interval=20, frames=len(V_LV))
    # ani.save("movie.mp4")

    plt.show()


if __name__ == "__main__":
    main()
