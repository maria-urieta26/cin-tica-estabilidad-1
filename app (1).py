import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from scipy import stats

R = 8.314462
st.set_page_config(page_title="Cinética y estabilidad", layout="wide")
st.title("Cinética y estabilidad de principios activos")
st.caption("Orden de reacción (método integral), k, t50/t98/t95/t90, Arrhenius y predicción de estabilidad.")

MANUAL = """
**¿Para qué sirve?** Analiza datos de tiempo y concentración de un principio activo: evalúa los órdenes cero, uno y dos
(método integral), calcula k, t50, t98, t95 y t90, ajusta Arrhenius y predice la estabilidad a otras temperaturas.
No requiere instalar nada: se usa directamente en el navegador.

**Paso 1. Datos del ensayo.** Escribe la temperatura del ensayo y elige la unidad de tiempo y de concentración de tus datos
y la unidad en que quieres los resultados (la app convierte antes de calcular). Si conviertes entre mol y masa, ingresa la masa
molar (g/mol). Si usas otra unidad (por ejemplo, % p/v), marca «No convertir». En la tabla escribe o pega las parejas
tiempo–concentración (mínimo 3, una por fila, con decimales). Los datos de ejemplo (piroxicam) deben borrarse antes de ingresar los tuyos;
verifica que no quede ninguna fila sobrante.

**Paso 2. Orden de reacción.** La app ajusta tres rectas y elige el modelo con mayor R²:

| Orden | Gráfico | Ecuación | k | Tiempo para que quede la fracción q |
|---|---|---|---|---|
| 0 | [A] vs t | [A] = [A]0 − k·t | −pendiente | t = (1 − q)·[A]0 / k |
| 1 | ln[A] vs t | ln[A] = ln[A]0 − k·t | −pendiente | t = −ln q / k |
| 2 | 1/[A] vs t | 1/[A] = 1/[A]0 + k·t | pendiente | t = (1/q − 1) / (k·[A]0) |

q = 0,50 da t50; q = 0,98 da t98; q = 0,95 da t95; q = 0,90 da t90 (vida útil). Si la diferencia de R² entre los dos mejores
modelos es menor a 0,005, la app avisa que el orden es ambiguo (suele ocurrir con poca degradación).

**Paso 3. Arrhenius.** Ingresa la temperatura (°C) y la k a esa temperatura (mínimo 2 temperaturas; con 3 o más hay intervalos
de confianza). Elige la unidad de tiempo en que están tus k (la app las convierte a la unidad de los resultados). El botón «Agregar k del análisis actual» pasa la k del paso 2 a la tabla.
Se calculan 1/T, ln k, Ea, A y R².

**Paso 4. Predicción.** Elige una temperatura con el deslizador. La tabla muestra k, t50, t98, t95 y t90
(con intervalo de confianza del 95 %) para las condiciones de estabilidad: prolongada (25 °C / 60 % HR, 12 meses), alternativa
(30 °C / 60 % HR), acelerada (40 °C / 75 % HR, 6 meses) y refrigerador. Indica si hay extrapolación y si el t90 cubre la duración del estudio.
Al final aparece una conclusión automática.

**Problemas frecuentes.** R² muy bajo en todos los modelos: revisa filas sobrantes o errores de digitación. «Revisa las unidades»:
las k de Arrhenius no están en la misma unidad de tiempo que el paso 1. No aparece la predicción: faltan al menos 2 temperaturas en el paso 3.

**Limitaciones.** Se supone el mismo mecanismo de degradación en todo el intervalo de temperatura y las mismas condiciones de pH, luz,
humedad y oxígeno del ensayo. La conclusión automática se genera con reglas y no sustituye el juicio del analista.
"""
with st.expander("📖 Manual de usuario (clic para abrir)"):
    st.markdown(MANUAL)


# ---------- funciones de cálculo ----------
def fit(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    n = len(x)
    r = stats.linregress(x, y)
    resid = y - (r.slope * x + r.intercept)
    s = np.sqrt((resid ** 2).sum() / (n - 2)) if n > 2 else np.nan
    tc = stats.t.ppf(0.975, n - 2) if n > 2 else np.nan
    return dict(m=r.slope, b=r.intercept, r2=r.rvalue ** 2, n=n, mx=x.mean(),
                sxx=((x - x.mean()) ** 2).sum(), s=s, tc=tc)


def t_for(order, k, A0, q):
    """Tiempo para que quede la fracción q de la concentración inicial."""
    if not k > 0:
        return np.nan
    if order == 0:
        return (1 - q) * A0 / k
    if order == 1:
        return -np.log(q) / k
    return (1 / q - 1) / (A0 * k)


def scatter_fit(x, y, m, b, xl, yl, title):
    xs = np.array([min(x), max(x)])
    fig = go.Figure()
    fig.add_scatter(x=x, y=y, mode="markers", name="Datos")
    fig.add_scatter(x=xs, y=m * xs + b, mode="lines", name="Ajuste")
    fig.update_layout(title=title, xaxis_title=xl, yaxis_title=yl, height=320,
                      margin=dict(l=10, r=10, t=40, b=10), showlegend=False)
    return fig


# ---------- 1. datos ----------
st.header("1. Datos del ensayo")
TIME = {"segundos": 1, "minutos": 60, "horas": 3600, "días": 86400, "semanas": 604800,
        "meses (30 d)": 2592000, "años (365 d)": 31536000}
CONC = {"mol/L = mol/dm³": ("mol", 1), "mmol/L = mmol/dm³": ("mol", 1e-3), "µmol/L": ("mol", 1e-6),
        "g/L = mg/mL": ("g", 1), "mg/L = µg/mL": ("g", 1e-3), "µg/L": ("g", 1e-6)}
Tt = st.number_input("Temperatura del ensayo (°C)", value=40.0)
sin = st.checkbox("No convertir: uso mis propias unidades (por ejemplo, % p/v)")
if sin:
    x1, x2 = st.columns(2)
    ut_in = ut = x1.text_input("Unidad de tiempo", "semanas")
    uc_in = uc = x2.text_input("Unidad de concentración", "mmol/dm³")
    ft = fc = 1.0
else:
    x1, x2, x3, x4 = st.columns(4)
    ut_in = x1.selectbox("Tiempo: unidad de tus datos", list(TIME), index=4)
    ut = x2.selectbox("Tiempo: unidad de los resultados", list(TIME), index=4)
    uc_in = x3.selectbox("Concentración: unidad de tus datos", list(CONC), index=1)
    uc = x4.selectbox("Concentración: unidad de los resultados", list(CONC), index=1)
    ft = TIME[ut_in] / TIME[ut]
    (ki, fi), (ko, fo) = CONC[uc_in], CONC[uc]
    if ki == ko:
        fc = fi / fo
    else:
        M = st.number_input("Masa molar del fármaco (g/mol), necesaria para pasar entre mol y masa", min_value=0.0, value=0.0)
        if M <= 0:
            st.warning("Ingresa la masa molar para convertir entre unidades molares y de masa.")
            st.stop()
        fc = fi * M / fo if ki == "mol" else fi / (M * fo)
ejemplo = pd.DataFrame({"t": [0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
                        "C": [100.05, 82.0, 60.5, 40.06, 22.0, 11.5]})
df = st.data_editor(ejemplo, num_rows="dynamic", width="stretch", key="datos",
                    column_config={"t": st.column_config.NumberColumn(f"Tiempo ({ut_in})", format="%.4g", step=0.01),
                                   "C": st.column_config.NumberColumn(f"Concentración ({uc_in})", format="%.6g", step=0.0001)})
df = df.astype(float)
st.caption("Los datos de ejemplo son el ejercicio del piroxicam (semanas, mmol/dm³). Bórralos y escribe los tuyos. "
           f"Los cálculos se hacen en {ut} y {uc}.")
df = df.dropna()

if len(df) < 3:
    st.warning("Ingresa al menos 3 parejas tiempo–concentración.")
    st.stop()

t = df["t"].to_numpy(float) * ft
c = df["C"].to_numpy(float) * fc
positivo = (c > 0).all()

# ---------- 2. orden de reacción ----------
st.header("2. Orden de reacción y parámetros cinéticos")
ucs = uc.split(" = ")[0]
modelos = [dict(o=0, n="Orden cero", yl="[A]", y=c, ku=f"{ucs}/{ut}", ok=True),
           dict(o=1, n="Orden uno", yl="ln[A]", y=np.log(c) if positivo else None, ku=f"1/{ut}", ok=positivo),
           dict(o=2, n="Orden dos", yl="1/[A]", y=1 / c if positivo else None, ku=f"1/({ucs}·{ut})", ok=positivo)]
for m in modelos:
    if m["ok"]:
        m["f"] = fit(t, m["y"])
        m["k"] = m["f"]["m"] if m["o"] == 2 else -m["f"]["m"]
        b = m["f"]["b"]
        m["A0"] = b if m["o"] == 0 else np.exp(b) if m["o"] == 1 else 1 / b
validos = sorted([m for m in modelos if m["ok"]], key=lambda m: -m["f"]["r2"])
mejor = validos[0]
ambiguo = len(validos) > 1 and mejor["f"]["r2"] - validos[1]["f"]["r2"] < 0.005

msg = f"**Modelo más probable: {mejor['n']}** (R² = {mejor['f']['r2']:.4f})."
if ambiguo:
    deg = (1 - c[np.argmax(t)] / c[np.argmin(t)]) * 100
    st.warning(msg + f" La diferencia de R² con el segundo modelo es menor a 0,005: los datos no distinguen bien el orden. "
               f"En este ensayo se degradó solo {deg:.1f} % del principio activo; con poca degradación las tres "
               "funciones se parecen a una recta (diap. 89-90). Extiende el ensayo hasta mayor conversión, o "
               "confirma el orden con el método de la semivida (diap. 91) usando varias concentraciones iniciales.")
else:
    st.success(msg)

cols = st.columns(3)
for col, m in zip(cols, modelos):
    if m["ok"]:
        col.plotly_chart(scatter_fit(t, m["y"], m["f"]["m"], m["f"]["b"], f"t ({ut})", m["yl"],
                                     f"{m['yl']} vs t"), width="stretch")
    else:
        col.info("No se puede calcular con concentraciones ≤ 0.")

tabla = pd.DataFrame([{"Modelo": m["n"], "Pendiente": m["f"]["m"], "Intercepto": m["f"]["b"],
                       "R²": m["f"]["r2"], "k": m["k"], "Unidades de k": m["ku"]}
                      for m in validos])
st.dataframe(tabla, width="stretch", hide_index=True)

o, k, A0 = mejor["o"], mejor["k"], mejor["A0"]
params = pd.DataFrame({"Parámetro": ["k", "[A]0 ajustada", "t50", "t98", "t95", "t90"],
                       "Valor": [f"{k:.4g} {mejor['ku']}", f"{A0:.4g} {uc}"] +
                                [f"{t_for(o, k, A0, q):.4g} {ut}" for q in (.5, .98, .95, .9)]})
st.table(params)

# ---------- 3. Arrhenius ----------
st.header("3. Módulo Arrhenius")
if "arr" not in st.session_state:
    st.session_state.arr = pd.DataFrame({"T_C": pd.Series(dtype=float), "k": pd.Series(dtype=float)})


def cargar_ejemplo():
    st.session_state.arr = pd.DataFrame({"T_C": [30.0, 40.0, 50.0, 60.0], "k": [0.0183, 0.0498, 0.131, 0.309]})
    st.session_state["ut_k"] = "horas"


if sin or ut not in TIME:
    ut_k, fk = ut, 1.0
    st.caption(f"Las k de esta tabla deben estar en 1/{ut}.")
else:
    if st.session_state.get("ut_k") not in TIME:
        st.session_state["ut_k"] = ut
    ut_k = st.selectbox("Unidad de tiempo de las k de esta tabla", list(TIME), key="ut_k")
    fk = TIME[ut] / TIME[ut_k]
    st.caption(f"Escribe cada k en 1/{ut_k}: la app la convierte sola a 1/{ut}, la unidad de los resultados. "
               "Para orden 0 y 2, la unidad de concentración de k debe coincidir con la de los resultados.")
b1, b2 = st.columns(2)
if b1.button("Agregar k del análisis actual"):
    a = st.session_state.arr
    a = a[a["T_C"] != Tt]
    st.session_state.arr = pd.concat([a, pd.DataFrame({"T_C": [Tt], "k": [k / fk]})]).sort_values("T_C").reset_index(drop=True)
b2.button("Cargar ejemplo de Arrhenius (k en 1/h)", on_click=cargar_ejemplo)
arr = st.data_editor(st.session_state.arr, num_rows="dynamic", width="stretch", key="arr_ed").dropna()
arr = arr[arr["k"] > 0].copy()
arr["k"] = arr["k"] * fk

F = FE = None
if len(arr) >= 2:
    invT = 1 / (arr["T_C"].to_numpy(float) + 273.15)
    lnk = np.log(arr["k"].to_numpy(float))
    F = fit(invT, lnk)
    Ea = -F["m"] * R / 1000
    ic = F["tc"] * F["s"] / np.sqrt(F["sxx"]) * R / 1000 if F["n"] > 2 else np.nan
    st.dataframe(pd.DataFrame({"T (°C)": arr["T_C"], "1/T (K⁻¹)": invT, f"k ({mejor['ku']})": arr["k"], "ln k": lnk}),
                 width="stretch", hide_index=True)
    a1, a2, a3 = st.columns(3)
    a1.metric("Ea (kJ/mol)", f"{Ea:.4g}", None if np.isnan(ic) else f"± {ic:.3g} (IC 95 %)", delta_color="off")
    a2.metric("A (factor preexponencial)", f"{np.exp(F['b']):.4g}")
    a3.metric("R²", f"{F['r2']:.4f}")
    st.plotly_chart(scatter_fit(1000 * invT, lnk, F["m"] / 1000, F["b"], "1000/T (K⁻¹)", "ln k",
                                "Gráfico de Arrhenius"), width="stretch")
else:
    st.info("Ingresa al menos 2 temperaturas con su k.")

# ---------- 4. predicción ----------
st.header("4. Predicción de estabilidad")
if F is None:
    st.stop()
FX = F


def k_at(T):
    x = 1 / (T + 273.15)
    l = FX["b"] + FX["m"] * x
    w = FX["tc"] * FX["s"] * np.sqrt(1 / FX["n"] + (x - FX["mx"]) ** 2 / FX["sxx"]) if FX["n"] > 2 else 0
    return np.exp(l), np.exp(l - w), np.exp(l + w)


Tmin, Tmax = arr["T_C"].min(), arr["T_C"].max()
if Tmax - Tmin > 100:
    st.warning("El intervalo de temperaturas ensayado supera 100 K: A solo puede considerarse constante en ≤ 50–100 K (diap. 74).")
T0 = st.slider("Temperatura de almacenamiento (°C)", 0, 60, 25)
if not Tmin <= T0 <= Tmax:
    st.warning(f"{T0} °C está fuera del intervalo ensayado ({Tmin:g}–{Tmax:g} °C): es una extrapolación y asume el mismo mecanismo.")
K0 = k_at(T0)
razon = k / k_at(Tt)[0]
if not 0.67 < razon < 1.5:
    st.warning(f"Revisa las unidades: según el modelo, k a {Tt:g} °C sería {k_at(Tt)[0]:.4g}, pero el análisis del paso 2 dio "
               f"{k:.4g} {mejor['ku']}. Las k de la tabla de Arrhenius deben estar en la misma unidad de tiempo que el paso 1.")
st.metric(f"t90 a {T0} °C ({ut})", f"{t_for(o, K0[0], A0, .9):.4g}")

a_meses = TIME[ut] / TIME["meses (30 d)"] if ut in TIME else None
# condiciones de los estudios de estabilidad (diap. 60): (nombre, T, duración del estudio en meses)
esc = [("Prolongada 25 ± 2 °C / 60 ± 5 % HR (12 meses)", 25, 12), ("Alternativa 30 ± 2 °C / 60 ± 5 % HR", 30, None),
       ("Acelerada 40 ± 2 °C / 75 ± 5 % HR (6 meses)", 40, 6), ("Refrigerador", 5, None), ("Temperatura elegida", T0, None)]
filas = []
for nombre, T, dur in esc:
    kk, lo, hi = k_at(T)
    t90c, t90_cons = t_for(o, kk, A0, .9), t_for(o, hi, A0, .9)
    r90 = f"{t90c:.4g}" + (f" ({t_for(o, hi, A0, .9):.4g}–{t_for(o, lo, A0, .9):.4g})" if FX["n"] > 2 else "")
    fila = {"Escenario": nombre, "T (°C)": T, f"k ({mejor['ku']})": f"{kk:.4g}",
            f"t50 ({ut})": f"{t_for(o, kk, A0, .5):.4g}", "t98": f"{t_for(o, kk, A0, .98):.4g}",
            "t95": f"{t_for(o, kk, A0, .95):.4g}", "t90 (IC 95 %)": r90,
            "¿Extrapolado?": "Sí" if not Tmin <= T <= Tmax else "No"}
    if a_meses:
        fila["t90 (meses)"] = f"{t90c * a_meses:.3g}"
        fila["¿Cubre el estudio?"] = ("Sí" if t90_cons * a_meses >= dur else "No") if dur else "—"
    filas.append(fila)
st.dataframe(pd.DataFrame(filas), width="stretch", hide_index=True)
st.caption("«Cubre el estudio» compara el límite conservador del t90 (IC 95 %) con la duración del ensayo de estabilidad.")

# ---------- conclusión automática (punto 6 del enunciado) ----------
st.subheader("Conclusión automática")
st.caption("Se genera con reglas a partir de la predicción de arriba y cambia con la temperatura elegida en el deslizador.")


def tt(q, kk):
    return t_for(o, kk, A0, q)


Kt = k_at(Tt)
fk_ = K0[0] / Kt[0]
st.dataframe(pd.DataFrame([{"Parámetro": n, f"A {Tt:g} °C ({ut})": f"{tt(q, Kt[0]):.4g}",
                            f"A {T0} °C ({ut})": f"{tt(q, K0[0]):.4g}", "Cambio": f"×{1 / fk_:.3g}"}
                           for n, q in (("t50", .5), ("t98", .98), ("t95", .95), ("t90", .9))]),
             width="stretch", hide_index=True)
ic_txt = "" if np.isnan(ic) else f" ± {ic:.3g}"
mes = (lambda x: f" (≈ {x * a_meses:.3g} meses)" if a_meses else "")
rango = f", con intervalo de confianza del 95 % de {tt(.9, K0[2]):.4g} a {tt(.9, K0[1]):.4g} {ut}" if F["n"] > 2 else ""
fuera = "" if Tmin <= T0 <= Tmax else f" {T0} °C está fuera del intervalo ensayado ({Tmin:g}–{Tmax:g} °C): es una extrapolación."
sentido = "se acortan" if fk_ > 1 else "se alargan" if fk_ < 1 else "no cambian"
lineas = [
    f"**1. Constante cinética.** Con la ecuación de Arrhenius (Ea = {Ea:.4g}{ic_txt} kJ/mol, R² = {F['r2']:.4f}), k pasa de "
    f"{Kt[0]:.4g} a {Tt:g} °C a {K0[0]:.4g} {mejor['ku']} a {T0} °C (×{fk_:.3g}).",
    f"**2. Cambio en t50, t98, t95 y t90.** Al pasar de {Tt:g} °C a {T0} °C, los cuatro tiempos {sentido} en el mismo factor "
    f"(×{1 / fk_:.3g}), porque con la concentración inicial fija cada tiempo es inversamente proporcional a k. A {T0} °C: "
    f"t50 = {tt(.5, K0[0]):.4g} {ut}, t98 = {tt(.98, K0[0]):.4g} {ut}, t95 = {tt(.95, K0[0]):.4g} {ut} y "
    f"t90 = {tt(.9, K0[0]):.4g} {ut}{mes(tt(.9, K0[0]))}{rango}.{fuera}",
]
caut = ["se supone el mismo mecanismo de degradación en todo el intervalo de temperatura y las mismas condiciones de pH, luz, humedad y oxígeno del ensayo"]
if ambiguo:
    caut.insert(0, "el orden de reacción es ambiguo (R² casi iguales), lo que afecta todos los tiempos")
if F["n"] < 3:
    caut.insert(0, "con solo 2 temperaturas no hay intervalo de confianza")
lineas.append("**Cautelas.** " + (lambda x: x[0].upper() + x[1:])("; ".join(caut)) + ".")
st.info("\n\n".join(lineas))
