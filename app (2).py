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
**¿Para qué sirve?** Analiza datos de tiempo y concentración de un principio activo: evalúa los órdenes cero, uno y dos (método integral),
calcula k, t50, t98, t95 y t90, ajusta Arrhenius y predice la estabilidad a otras temperaturas. No requiere instalar nada.

**Paso 1. Datos del ensayo.** Las tablas empiezan vacías: escribe tus datos o pulsa «Cargar ejemplo». Se aceptan decimales con punto o con coma
(100.0 o 100,0). Elige las unidades de tus datos y las de los resultados (la app convierte antes de calcular). Si cambias una unidad
o una celda, la tabla no se borra; solo se vacía con el botón «Borrar tabla».

**Paso 2. Orden de reacción.** Se ajustan tres rectas y se elige el modelo con mayor R²:

| Orden | Gráfico | Ecuación | k | Tiempo para que quede la fracción q |
|---|---|---|---|---|
| 0 | [A] vs t | [A] = [A]0 − k·t | −pendiente | t = (1 − q)·[A]0 / k |
| 1 | ln[A] vs t | ln[A] = ln[A]0 − k·t | −pendiente | t = −ln q / k |
| 2 | 1/[A] vs t | 1/[A] = 1/[A]0 + k·t | pendiente | t = (1/q − 1) / (k·[A]0) |

q = 0,50 da t50; 0,98 da t98; 0,95 da t95; 0,90 da t90 (vida útil). Si los dos mejores R² difieren menos de 0,005, la app avisa que el orden es ambiguo.

**Paso 3. Arrhenius.** Ingresa la temperatura en °C (la columna en K se calcula sola) y la k a esa temperatura; elige la unidad de tiempo de tus k.
Mínimo 2 temperaturas (con 3 o más hay intervalos de confianza). La app muestra Ea, A (con las unidades de k), R², la ecuación completa de la recta,
y el gráfico de la influencia de la temperatura sobre k con su interpretación.

**Paso 4. Predicción.** Elige una temperatura con el deslizador. La tabla da k, t50, t98, t95 y t90 para varias condiciones de almacenamiento, y la
conclusión automática resume lo que ocurre con la estabilidad a la temperatura elegida frente a otras temperaturas.

**Limitaciones.** Se supone el mismo mecanismo de degradación en todo el intervalo de temperatura y las mismas condiciones de pH, luz, humedad y oxígeno.
La conclusión se genera con reglas y no sustituye el juicio del analista.
"""
with st.expander("📖 Manual de usuario (clic para abrir)"):
    st.markdown(MANUAL)


# ---------- utilidades ----------
def txt(x):
    return "" if x is None or (not isinstance(x, str) and pd.isna(x)) else str(x).strip()


def num(x):
    """Convierte texto a número aceptando punto o coma decimal; devuelve nan si no es un número."""
    s = txt(x).replace(" ", "").replace(",", ".")
    try:
        return float(s)
    except ValueError:
        return np.nan


def fmt(x, d=5):
    """Mínimo d decimales y al menos 5 cifras significativas; notación científica para magnitudes extremas."""
    try:
        x = float(x)
    except (TypeError, ValueError):
        return "—"
    if not np.isfinite(x):
        return "—"
    if x == 0:
        return f"{0:.{d}f}"
    ax = abs(x)
    if ax >= 1e7 or ax < 1e-9:
        return f"{x:.{d}e}"
    return f"{x:.{max(d, 4 - int(np.floor(np.log10(ax))))}f}"


def vacia(cols):
    return pd.DataFrame({c: pd.Series(dtype="object") for c in cols})


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
for kk_, v_ in (("ut_in", "semanas"), ("ut", "semanas"), ("uc_in", "mmol/L = mmol/dm³"), ("uc", "mmol/L = mmol/dm³")):
    st.session_state.setdefault(kk_, v_)
st.session_state.setdefault("datos_base", vacia(["Tiempo", "Concentración"]))
st.session_state.setdefault("datos_ver", 0)

nombre = st.text_input("Nombre del principio activo (opcional; aparece en la conclusión)", key="nombre")
Tt = num(st.text_input("Temperatura del ensayo (°C)", value="40", key="Tt_txt"))
if np.isnan(Tt):
    st.error("La temperatura del ensayo debe ser un número (se acepta punto o coma).")
    st.stop()
sin = st.checkbox("No convertir: uso mis propias unidades (por ejemplo, % p/v)")
if sin:
    x1, x2 = st.columns(2)
    ut_in = ut = x1.text_input("Unidad de tiempo", "semanas")
    uc_in = uc = x2.text_input("Unidad de concentración", "mmol/dm³")
    ft = fc = 1.0
else:
    x1, x2, x3, x4 = st.columns(4)
    ut_in = x1.selectbox("Tiempo: unidad de tus datos", list(TIME), key="ut_in")
    ut = x2.selectbox("Tiempo: unidad de los resultados", list(TIME), key="ut")
    uc_in = x3.selectbox("Concentración: unidad de tus datos", list(CONC), key="uc_in")
    uc = x4.selectbox("Concentración: unidad de los resultados", list(CONC), key="uc")
    ft = TIME[ut_in] / TIME[ut]
    (ki, fi), (ko, fo) = CONC[uc_in], CONC[uc]
    if ki == ko:
        fc = fi / fo
    else:
        M = num(st.text_input("Masa molar del fármaco (g/mol), necesaria para pasar entre mol y masa", key="M_txt"))
        if np.isnan(M) or M <= 0:
            st.warning("Ingresa la masa molar para convertir entre unidades molares y de masa.")
            st.stop()
        fc = fi * M / fo if ki == "mol" else fi / (M * fo)


def cargar_datos_ejemplo():
    st.session_state.datos_base = pd.DataFrame({"Tiempo": ["0", "1", "2", "3", "4", "5"],
                                                "Concentración": ["100.05", "82", "60.5", "40.06", "22", "11.5"]})
    st.session_state.datos_ver += 1
    st.session_state.update(ut_in="semanas", ut="semanas", uc_in="mmol/L = mmol/dm³", uc="mmol/L = mmol/dm³")


def borrar_datos():
    st.session_state.datos_base = vacia(["Tiempo", "Concentración"])
    st.session_state.datos_ver += 1


st.caption(f"Escribe el tiempo (en {ut_in}) y la concentración (en {uc_in}). Se acepta punto o coma decimal (100.0 o 100,0). "
           "Las filas se pueden agregar al final de la tabla.")
bt1, bt2, _ = st.columns([1, 1, 3])
bt1.button("Cargar ejemplo (piroxicam)", on_click=cargar_datos_ejemplo)
bt2.button("Borrar tabla", on_click=borrar_datos)
df_raw = st.data_editor(st.session_state.datos_base, num_rows="dynamic", width="stretch",
                        key=f"datos_ed_{st.session_state.datos_ver}",
                        column_config={"Tiempo": st.column_config.TextColumn("Tiempo"),
                                       "Concentración": st.column_config.TextColumn("Concentración")})
T_ = df_raw["Tiempo"].map(num)
C_ = df_raw["Concentración"].map(num)
fila_vacia = (df_raw["Tiempo"].map(txt) == "") & (df_raw["Concentración"].map(txt) == "")
mal = ~fila_vacia & (T_.isna() | C_.isna())
if mal.any():
    st.warning("Se ignoraron las filas " + ", ".join(str(i + 1) for i in np.where(mal.to_numpy())[0]) +
               ": están incompletas o tienen texto que no es un número.")
df = pd.DataFrame({"t": T_, "C": C_})[~fila_vacia & ~mal]
if len(df) < 3:
    st.info("Ingresa al menos 3 parejas tiempo–concentración, o pulsa «Cargar ejemplo».")
    st.stop()
st.caption(f"Se leyeron {len(df)} puntos. Los cálculos se hacen en {ut} y {uc}.")
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

msg = f"**Modelo más probable: {mejor['n']}** (R² = {fmt(mejor['f']['r2'])})."
if ambiguo:
    deg = (1 - c[np.argmax(t)] / c[np.argmin(t)]) * 100
    st.warning(msg + f" La diferencia de R² con el segundo modelo es menor a 0,005: los datos no distinguen bien el orden. "
               f"En este ensayo se degradó solo {deg:.2f} % del principio activo; con poca degradación las tres "
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

st.dataframe(pd.DataFrame([{"Modelo": m["n"], "Pendiente": fmt(m["f"]["m"]), "Intercepto": fmt(m["f"]["b"]),
                            "R²": fmt(m["f"]["r2"]), "k": fmt(m["k"]), "Unidades de k": m["ku"]} for m in validos]),
             width="stretch", hide_index=True)

o, k, A0 = mejor["o"], mejor["k"], mejor["A0"]
st.table(pd.DataFrame({"Parámetro": ["k", "[A]0 ajustada", "t50", "t98", "t95", "t90"],
                       "Valor": [f"{fmt(k)} {mejor['ku']}", f"{fmt(A0)} {ucs}"] +
                                [f"{fmt(t_for(o, k, A0, q))} {ut}" for q in (.5, .98, .95, .9)]}))

# ---------- 3. Arrhenius ----------
st.header("3. Módulo Arrhenius")
st.session_state.setdefault("arr_base", vacia(["T (°C)", "T (K)", "k"]))
st.session_state.setdefault("arr_ver", 0)


def cargar_arr_ejemplo():
    Ts = [30, 40, 50, 60]
    st.session_state.arr_base = pd.DataFrame({"T (°C)": [str(x) for x in Ts], "T (K)": [f"{x + 273.15:.2f}" for x in Ts],
                                              "k": ["0.0183", "0.0498", "0.131", "0.309"]})
    st.session_state.arr_ver += 1
    st.session_state["ut_k"] = "horas"


def borrar_arr():
    st.session_state.arr_base = vacia(["T (°C)", "T (K)", "k"])
    st.session_state.arr_ver += 1


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
st.caption("Escribe la temperatura en °C; la columna en K se calcula sola. Se acepta punto o coma decimal.")
arr_raw = st.data_editor(st.session_state.arr_base, num_rows="dynamic", width="stretch", disabled=["T (K)"],
                         key=f"arr_ed_{st.session_state.arr_ver}",
                         column_config={"T (°C)": st.column_config.TextColumn("T (°C)"),
                                        "T (K)": st.column_config.TextColumn("T (K)", help="°C + 273,15 (automático)"),
                                        "k": st.column_config.TextColumn("k")})
Tn_raw = arr_raw["T (°C)"].map(num)
K_calc = ["" if np.isnan(v) else f"{v + 273.15:.2f}" for v in Tn_raw]
if [txt(v) for v in arr_raw["T (K)"]] != K_calc:      # recalcula K sin perder lo escrito
    nuevo = arr_raw.copy()
    nuevo["T (K)"] = K_calc
    st.session_state.arr_base = nuevo
    st.session_state.arr_ver += 1
    st.rerun()
b1, b2, b3 = st.columns(3)
if b1.button("Agregar k del análisis actual"):
    nuevo = arr_raw[~(Tn_raw.round(6) == round(Tt, 6))]
    fila = pd.DataFrame({"T (°C)": [f"{Tt:g}"], "T (K)": [f"{Tt + 273.15:.2f}"], "k": [f"{k / fk:.8g}"]})
    st.session_state.arr_base = pd.concat([nuevo, fila], ignore_index=True)
    st.session_state.arr_ver += 1
    st.rerun()
b2.button("Cargar ejemplo de Arrhenius (k en 1/h)", on_click=cargar_arr_ejemplo)
b3.button("Borrar tabla de Arrhenius", on_click=borrar_arr)

arr = pd.DataFrame({"T_C": Tn_raw, "k": arr_raw["k"].map(num)})
arr_vacia = (arr_raw["T (°C)"].map(txt) == "") & (arr_raw["k"].map(txt) == "")
arr_mal = ~arr_vacia & (arr["T_C"].isna() | arr["k"].isna() | (arr["k"] <= 0))
if arr_mal.any():
    st.warning("Se ignoraron las filas " + ", ".join(str(i + 1) for i in np.where(arr_mal.to_numpy())[0]) +
               " de la tabla de Arrhenius: están incompletas, no son números o k no es positiva.")
arr = arr[~arr_vacia & ~arr_mal].copy()
arr["k"] = arr["k"] * fk

F = None
if len(arr) >= 2:
    Tk = arr["T_C"].to_numpy(float) + 273.15
    invT = 1 / Tk
    lnk = np.log(arr["k"].to_numpy(float))
    F = fit(invT, lnk)
    Ea = -F["m"] * R / 1000
    ic = F["tc"] * F["s"] / np.sqrt(F["sxx"]) * R / 1000 if F["n"] > 2 else np.nan
    Tmin, Tmax = arr["T_C"].min(), arr["T_C"].max()
    st.dataframe(pd.DataFrame({"T (°C)": [f"{v:g}" for v in arr["T_C"]], "T (K)": [f"{v:.2f}" for v in Tk],
                               "1/T (K⁻¹)": [fmt(v) for v in invT], f"k ({mejor['ku']})": [fmt(v) for v in arr["k"]],
                               "ln k": [fmt(v) for v in lnk]}), width="stretch", hide_index=True)
    a1, a2, a3 = st.columns(3)
    a1.metric("Ea (kJ/mol)", fmt(Ea), None if np.isnan(ic) else f"± {fmt(ic)} (IC 95 %)", delta_color="off")
    a2.metric(f"A, factor preexponencial ({mejor['ku']})", fmt(np.exp(F["b"])))
    a3.metric("R²", fmt(F["r2"]))
    st.markdown("**Ecuación de la recta de Arrhenius**")
    st.latex(r"\ln k = \ln A - \frac{E_a}{R}\left(\frac{1}{T}\right)")
    st.latex(r"\ln k = " + f"{F['b']:.5f}" + (" - " if F["m"] < 0 else " + ") + f"{abs(F['m']):.5f}" + r"\left(\frac{1}{T}\right)")
    st.caption(f"T en kelvin; R² = {F['r2']:.5f}. ln A = {F['b']:.5f}, de modo que A = {fmt(np.exp(F['b']))} {mejor['ku']}; "
               f"Ea/R = {abs(F['m']):.5f} K, de modo que Ea = {fmt(Ea)} kJ/mol (R = 8,314462 J/mol·K).")
    st.plotly_chart(scatter_fit(1000 * invT, lnk, F["m"] / 1000, F["b"], "1000/T (K⁻¹)", "ln k",
                                "Gráfico de Arrhenius: ln k vs 1/T"), width="stretch")

    # influencia de la temperatura sobre k
    st.subheader("Influencia de la temperatura sobre k")
    Tg = np.linspace(Tmin - 10, Tmax + 10, 150)
    fig = go.Figure()
    fig.add_scatter(x=Tg, y=np.exp(F["b"] + F["m"] / (Tg + 273.15)), mode="lines", name="k según Arrhenius")
    fig.add_scatter(x=arr["T_C"], y=arr["k"], mode="markers", name="k experimental")
    fig.update_layout(xaxis_title="Temperatura (°C)", yaxis_title=f"k ({mejor['ku']})", height=360,
                      margin=dict(l=10, r=10, t=20, b=10), legend=dict(orientation="h", y=-0.2))
    st.plotly_chart(fig, width="stretch")
    k_lo, k_hi = np.exp(F["b"] + F["m"] / (Tmin + 273.15)), np.exp(F["b"] + F["m"] / (Tmax + 273.15))
    Tm = arr["T_C"].mean()
    q10 = np.exp(F["m"] / (Tm + 10 + 273.15) - F["m"] / (Tm + 273.15))
    if Ea > 0 and k_hi > k_lo:
        st.info(f"**Interpretación.** La constante cinética aumenta con la temperatura: pasa de {fmt(k_lo)} a {Tmin:g} °C a "
                f"{fmt(k_hi)} {mejor['ku']} a {Tmax:g} °C (×{k_hi / k_lo:.2f}). A mayor temperatura, más moléculas alcanzan la energía de "
                f"activación (Ea = {fmt(Ea)} kJ/mol) y la degradación es más rápida. Cada 10 °C de aumento, k se multiplica por ≈ {q10:.2f}.")
    else:
        st.warning("Los datos no muestran que k aumente con la temperatura (Ea ≤ 0). Revisa las k y sus unidades.")
else:
    st.info("Ingresa al menos 2 temperaturas con su k.")

# ---------- 4. predicción ----------
st.header("4. Predicción de estabilidad")
if F is None:
    st.stop()


def k_at(T):
    x = 1 / (T + 273.15)
    l = F["b"] + F["m"] * x
    w = F["tc"] * F["s"] * np.sqrt(1 / F["n"] + (x - F["mx"]) ** 2 / F["sxx"]) if F["n"] > 2 else 0
    return np.exp(l), np.exp(l - w), np.exp(l + w)


if Tmax - Tmin > 100:
    st.warning("El intervalo de temperaturas ensayado supera 100 K: A solo puede considerarse constante en ≤ 50–100 K (diap. 74).")
T0 = st.slider("Temperatura de almacenamiento (°C)", 0, int(max(60, np.ceil(Tmax) + 20)), 25)
if not Tmin <= T0 <= Tmax:
    st.warning(f"{T0} °C está fuera del intervalo ensayado ({Tmin:g}–{Tmax:g} °C): es una extrapolación y asume el mismo mecanismo.")
K0 = k_at(T0)
Kt = k_at(Tt)
if not 0.67 < k / Kt[0] < 1.5:
    st.warning(f"Revisa las unidades: según Arrhenius, k a {Tt:g} °C sería {fmt(Kt[0])}, pero el análisis del paso 2 dio "
               f"{fmt(k)} {mejor['ku']}. Las k de la tabla de Arrhenius deben corresponder al mismo compuesto, condiciones y unidades.")
st.metric(f"t90 a {T0} °C ({ut})", fmt(t_for(o, K0[0], A0, .9)))


def tt(q, kk):
    return t_for(o, kk, A0, q)


a_meses = TIME[ut] / TIME["meses (30 d)"] if ut in TIME else None
esc = [("Prolongada 25 ± 2 °C / 60 ± 5 % HR (12 meses)", 25, 12), ("Alternativa 30 ± 2 °C / 60 ± 5 % HR", 30, None),
       ("Acelerada 40 ± 2 °C / 75 ± 5 % HR (6 meses)", 40, 6), ("Refrigerador", 5, None), ("Temperatura elegida", T0, None)]
filas = []
for nom, T, dur in esc:
    kk, lo, hi = k_at(T)
    r90 = fmt(tt(.9, kk)) + (f" ({fmt(tt(.9, hi))}–{fmt(tt(.9, lo))})" if F["n"] > 2 else "")
    fila = {"Escenario": nom, "T (°C)": f"{T:g}", f"k ({mejor['ku']})": fmt(kk), f"t50 ({ut})": fmt(tt(.5, kk)),
            "t98": fmt(tt(.98, kk)), "t95": fmt(tt(.95, kk)), "t90 (IC 95 %)": r90,
            "¿Extrapolado?": "Sí" if not Tmin <= T <= Tmax else "No"}
    if a_meses:
        fila["t90 (meses)"] = fmt(tt(.9, kk) * a_meses)
        fila["¿Cubre el estudio?"] = ("Sí" if tt(.9, hi) * a_meses >= dur else "No") if dur else "—"
    filas.append(fila)
st.dataframe(pd.DataFrame(filas), width="stretch", hide_index=True)
st.caption("«Cubre el estudio» compara el límite conservador del t90 (IC 95 %) con la duración del ensayo de estabilidad.")

# ---------- conclusión automática ----------
st.subheader("Conclusión automática")
st.caption("Se genera con reglas a partir de la predicción de arriba y cambia con la temperatura elegida en el deslizador.")
fk_ = K0[0] / Kt[0]
st.dataframe(pd.DataFrame([{"Parámetro": n, f"A {Tt:g} °C ({ut})": fmt(tt(q, Kt[0])),
                            f"A {T0:g} °C ({ut})": fmt(tt(q, K0[0])), "Cambio": f"×{1 / fk_:.2f}"}
                           for n, q in (("t50", .5), ("t98", .98), ("t95", .95), ("t90", .9))]),
             width="stretch", hide_index=True)

farm = f"el fármaco {nombre.strip()}" if nombre.strip() else "el fármaco"
ku = mejor["ku"]
mes = (lambda x: f" (≈ {fmt(x * a_meses, 2)} meses)" if a_meses else "")


def comparar(T):
    """Compara la degradación a T0 con la de otra temperatura T, a partir del t90."""
    r = tt(.9, K0[0]) / tt(.9, k_at(T)[0])
    if not np.isfinite(r) or r <= 0:
        return None
    f_ = max(r, 1 / r)
    if f_ < 1.2:
        return f"prácticamente igual a la de {T:g} °C"
    return f"{f_:.2f} veces más {'lenta' if r > 1 else 'rápida'} que a {T:g} °C"


r_t = tt(.9, K0[0]) / tt(.9, Kt[0])
if abs(T0 - Tt) < 1e-9:
    p1 = (f"Las predicciones se hicieron a la misma temperatura del ensayo ({Tt:g} °C), por lo que no hay cambio respecto a ella. "
          f"Para ver el efecto de la temperatura, mueva el deslizador.")
else:
    inten = max(r_t, 1 / r_t)
    if inten < 1.2:
        p1 = f"Las predicciones realizadas demostraron que {farm} presenta una estabilidad similar a {T0:g} °C y a {Tt:g} °C (temperatura del ensayo)."
    else:
        p1 = (f"Las predicciones realizadas demostraron que {farm} presenta {'mayor' if r_t > 1 else 'menor'} estabilidad a {T0:g} °C "
              f"que a {Tt:g} °C (temperatura del ensayo), con una degradación {'considerablemente ' if inten >= 2 else ''}"
              f"{'más lenta' if r_t > 1 else 'más rápida'} ({comparar(Tt)}).")
rango = f", con un intervalo de confianza del 95 % de {fmt(tt(.9, K0[2]))} a {fmt(tt(.9, K0[1]))} {ut}" if F["n"] > 2 else ""
p2 = (f"A la temperatura elegida, {T0:g} °C, se obtuvo k = {fmt(K0[0])} {ku}, t50 = {fmt(tt(.5, K0[0]))} {ut}, "
      f"t98 = {fmt(tt(.98, K0[0]))} {ut}, t95 = {fmt(tt(.95, K0[0]))} {ut} y t90 = {fmt(tt(.9, K0[0]))} {ut}{mes(tt(.9, K0[0]))}{rango}. "
      f"Es decir, la mitad del fármaco se degrada en {fmt(tt(.5, K0[0]))} {ut} y el 10 % (vida útil, t90) en {fmt(tt(.9, K0[0]))} {ut}.")
otras = [T for T in sorted({5, 25, 30, 40, round(Tt, 6)}) if abs(T - T0) > 1e-9]
cmp_ = [x for x in (comparar(T) for T in otras) if x]
p3 = ("Frente a otras temperaturas, la degradación a " + f"{T0:g} °C es " + "; ".join(cmp_) + " (según t90).") if cmp_ else ""
q10 = np.exp(F["m"] / (T0 + 10 + 273.15) - F["m"] / (T0 + 273.15))
p4 = (f"Cada 10 °C de aumento a partir de {T0:g} °C multiplica k por ≈ {q10:.2f} (Ea = {fmt(Ea)} kJ/mol), de modo que "
      f"{'almacenar a menor temperatura prolonga' if Ea > 0 else 'la temperatura no mejora'} la estabilidad.")
if a_meses:
    est = []
    for T, dur, nom in ((25, 12, "prolongada"), (40, 6, "acelerada")):
        cons = tt(.9, k_at(T)[2]) * a_meses
        est.append(f"a {T} °C ({nom}, {dur} meses) el t90 conservador es {fmt(cons, 2)} meses y "
                   + ("cubre el estudio" if cons >= dur else "no alcanza a cubrirlo"))
    p4 += " En los estudios de estabilidad, " + "; ".join(est) + "."
caut = ["se supone el mismo mecanismo de degradación en todo el intervalo de temperatura y las mismas condiciones de pH, luz, humedad y oxígeno del ensayo"]
if Tmin > T0 or T0 > Tmax:
    caut.insert(0, f"{T0:g} °C es una extrapolación fuera de {Tmin:g}–{Tmax:g} °C")
if ambiguo:
    caut.insert(0, "el orden de reacción es ambiguo (R² casi iguales), lo que afecta todos los tiempos")
if F["n"] < 3:
    caut.insert(0, "con solo 2 temperaturas no hay intervalo de confianza")
p5 = "Cautelas: " + "; ".join(caut) + "."
st.info("\n\n".join(p for p in (p1, p2, p3, p4, p5) if p))
