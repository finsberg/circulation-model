"""Parameters from http://venus.ceride.gov.ar/ojs/index.php/mc/article/view/3419
"""
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
    t_R: float
    t_C: float
    alpha: float
    name: str = "Chamber"


def _valve_values(valve: Literal["TV", "PV", "MV", "AV"]):
    if valve == "TV":
        ...
    elif valve == "PV":
        ...
    elif valve == "MV":
        ...
    elif valve == "AV":
        ...
    else:
        msg = f"Unsupported valve {valve!r}. Expected 'TV', 'PV', 'MV' or 'AV'"
        raise ValueError(msg)


def _chamber_values(name: Literal["LV", "RV", "LA", "RA"]):
    if name == "RA":
        E_act_max = 79.98 * ureg.dyn / (ureg.cm**2 * ureg.mL)
        E_act_max = E_act_max.to(ureg.mmHg / ureg.mL)

        E_pass = 93.31 * ureg.dyn / (ureg.cm**2 * ureg.mL)
        E_pass = E_pass.to(ureg.mmHg / ureg.mL)

        T_C = 0.17 * ureg.s
        T_R = 0.17 * ureg.s
        t_C = 0.80 * ureg.s
        t_R = 0.97 * ureg.s
        V_0 = 4.0 * ureg.mL
        alpha = 0.0005 * ureg.dimensionless

    elif name == "RV":

        E_act_max = 733.15 * ureg.dyn / (ureg.cm**2 * ureg.mL)
        E_act_max = E_act_max.to(ureg.mmHg / ureg.mL)

        E_pass = 66.65 * ureg.dyn / (ureg.cm**2 * ureg.mL)
        E_pass = E_pass.to(ureg.mmHg / ureg.mL)

        T_C = 0.34 * ureg.s
        T_R = 0.15 * ureg.s
        t_C = 0.0 * ureg.s
        t_R = 0.0 * ureg.s
        V_0 = 10.0 * ureg.mL
        alpha = 0.0005 * ureg.dimensionless

    elif name == "LA":

        E_act_max = 93.31 * ureg.dyn / (ureg.cm**2 * ureg.mL)
        E_act_max = E_act_max.to(ureg.mmHg / ureg.mL)

        E_pass = 119.97 * ureg.dyn / (ureg.cm**2 * ureg.mL)
        E_pass = E_pass.to(ureg.mmHg / ureg.mL)

        T_C = 0.17 * ureg.s
        T_R = 0.17 * ureg.s
        t_C = 0.80 * ureg.s
        t_R = 0.97 * ureg.s
        V_0 = 4.0 * ureg.mL
        alpha = 0.0005 * ureg.dimensionless

    elif name == "LV":

        E_act_max = 3665.75 * ureg.dyn / (ureg.cm**2 * ureg.mL)
        E_act_max = E_act_max.to(ureg.mmHg / ureg.mL)

        E_pass = 106.64 * ureg.dyn / (ureg.cm**2 * ureg.mL)
        E_pass = E_pass.to(ureg.mmHg / ureg.mL)

        T_C = 0.34 * ureg.s
        T_R = 0.15 * ureg.s
        t_C = 0.0 * ureg.s
        t_R = 0.0 * ureg.s
        V_0 = 5.0 * ureg.mL
        alpha = 0.0005 * ureg.dimensionless

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
        t_R=t_R,
        alpha=alpha,
        name=name,
    )


@dataclass
class Parameters:
    LV: Chamber
    RV: Chamber
    RA: Chamber
    LA: Chamber


parameters = Parameters(
    LV=_chamber_values("LV"),
    RV=_chamber_values("RV"),
    LA=_chamber_values("LA"),
    RA=_chamber_values("RA"),
)

print(parameters.LV)
print(parameters.RV)
print(parameters.RA)
print(parameters.LA)
