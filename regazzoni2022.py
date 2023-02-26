from typing import Literal
from dataclasses import dataclass
from pint import UnitRegistry


ureg = UnitRegistry()


@dataclass
class Chamber:
    E_act_max: float
    E_pass: float
    T_C: float
    T_R: float
    V_0: float
    t_C: float = 0.0
    alpha: float = 0.0005
    name: str = "Chamber"

    @property
    def t_R(self) -> float:
        return self.t_C + self.T_C


@dataclass
class Valve:
    ...


@dataclass
class Circulation:
    R: float
    C: float
    L: float
    name: str = "Circulation"


@dataclass
class Parameters:
    LV: Chamber
    RV: Chamber
    RA: Chamber
    LA: Chamber
    AR_SYS: Circulation
    AR_PUL: Circulation
    VEN_SYS: Circulation
    VEN_PUL: Circulation


def _circulation_values(name: Literal["AR_SYS", "AR_PUL", "VEN_SYS", "VEN_PUL"]):
    if name == "AR_SYS":
        R = 0.8 * ureg.mmHg * ureg.s / ureg.mL
        C = 1.2 * ureg.mL / ureg.mmHg
        L = 5e-3 * ureg.mmHg * (ureg.s**2) / ureg.mL

    elif name == "AR_PUL":
        R = 0.1625 * ureg.mmHg * ureg.s / ureg.mL
        C = 10.0 * ureg.mL / ureg.mmHg
        L = 5e-4 * ureg.mmHg * (ureg.s**2) / ureg.mL

    elif name == "VEN_SYS":
        R = 0.26 * ureg.mmHg * ureg.s / ureg.mL
        C = 60.0 * ureg.mL / ureg.mmHg
        L = 5e-4 * ureg.mmHg * (ureg.s**2) / ureg.mL

    elif name == "VEN_PUL":
        R = 0.1625 * ureg.mmHg * ureg.s / ureg.mL
        C = 16.0 * ureg.mL / ureg.mmHg
        L = 5e-4 * ureg.mmHg * (ureg.s**2) / ureg.mL

    else:
        msg = (
            f"Unsupported valve {name!r}. "
            "Expected 'AR_SYS', 'AR_PUL', 'VEN_SYS' or 'VEN_PUL'"
        )
        raise ValueError(msg)

    return Circulation(R=R, C=C, L=L, name=name)


def _valve_values(name: Literal["TV", "PV", "MV", "AV"]):
    if name == "TV":
        ...
    elif name == "PV":
        ...
    elif name == "MV":
        ...
    elif name == "AV":
        ...
    else:
        msg = f"Unsupported valve {name!r}. Expected 'TV', 'PV', 'MV' or 'AV'"
        raise ValueError(msg)


def _chamber_values(name: Literal["LV", "RV", "LA", "RA"]):
    if name == "RA":
        E_act_max = 0.06 * ureg.mmHg / ureg.mL
        E_pass = 0.07 * ureg.mmHg / ureg.mL

        T_C = 0.064 * ureg.s
        T_R = 0.64 * ureg.s
        t_C = 0.56 * ureg.s
        V_0 = 4.0 * ureg.mL

    elif name == "RV":

        E_act_max = 0.55 * ureg.mmHg / ureg.mL
        E_pass = 0.05 * ureg.mmHg / ureg.mL

        T_C = 0.272 * ureg.s
        T_R = 0.12 * ureg.s
        t_C = 0.0 * ureg.s
        V_0 = 10.0 * ureg.mL

    elif name == "LA":

        E_act_max = 0.07 * ureg.mmHg / ureg.mL
        E_pass = 0.09 * ureg.mmHg / ureg.mL

        T_C = 0.104 * ureg.s
        T_R = 0.68 * ureg.s
        t_C = 0.60 * ureg.s
        V_0 = 4.0 * ureg.mL

    elif name == "LV":

        E_act_max = 2.75 * ureg.mmHg / ureg.mL
        E_pass = 0.08 * ureg.mmHg / ureg.mL

        T_C = 0.272 * ureg.s
        T_R = 0.12 * ureg.s
        t_C = 0.0 * ureg.s
        V_0 = 5.0 * ureg.mL

    else:
        msg = f"Unsupported chamber {name!r}. Expected 'LA', 'LV', 'RV' or 'RV'"
        raise ValueError(msg)

    return Chamber(
        E_act_max=E_act_max,
        E_pass=E_pass,
        T_C=T_C,
        T_R=T_R,
        t_C=t_C,
        V_0=V_0,
        name=name,
    )


parameters = Parameters(
    LV=_chamber_values("LV"),
    RV=_chamber_values("RV"),
    LA=_chamber_values("LA"),
    RA=_chamber_values("RA"),
    AR_SYS=_circulation_values("AR_SYS"),
    AR_PUL=_circulation_values("AR_PUL"),
    VEN_SYS=_circulation_values("VEN_SYS"),
    VEN_PUL=_circulation_values("VEN_PUL"),
)
