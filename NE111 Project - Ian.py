from typing import Tuple, Dict, Any
import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
import streamlit as st

st.set_page_config(page_title="Histogram Fitter", layout="wide")

def parse_text_data(text: str) -> np.ndarray:
    """Convert pasted text containing numbers into a NumPy array."""
    if not text:
        return np.array([])

    cleaned = text.replace(",", " ")
    tokens = cleaned.split()

    values = []
    for t in tokens:
        try:
            values.append(float(t))
        except ValueError:
            pass

    return np.array(values)

def histogram_density(data: np.ndarray, bins: int = 30) -> Tuple[np.ndarray, np.ndarray]:
    counts, edges = np.histogram(data, bins=bins, density=True)
    centers = (edges[:-1] + edges[1:]) / 2
    return centers, counts


def compute_sse(x: np.ndarray, y: np.ndarray, pdf) -> float:
    return float(np.sum((y - pdf(x)) ** 2))


def compute_max_abs_error(x: np.ndarray, y: np.ndarray, pdf) -> float:
    return float(np.max(np.abs(y - pdf(x))))


def fit_distribution(dist: stats.rv_continuous, data: np.ndarray):
    params = dist.fit(data)
    shape_names = []

    if dist.shapes:
        shape_names = [s.strip() for s in dist.shapes.split(",")]

    names = shape_names + ["loc", "scale"]
    param_dict = {name: float(val) for name, val in zip(names, params)}

    return params, param_dict


def make_pdf_function(dist: stats.rv_continuous, params: Tuple) -> Any:
    def pdf(x):
        return dist.pdf(x, *params[:-2], loc=params[-2], scale=params[-1])
    return pdf

DISTRIBUTIONS = {
    "Normal": stats.norm,
    "Lognormal": stats.lognorm,
    "Gamma": stats.gamma,
    "Weibull": stats.weibull_min,
    "Exponential": stats.expon,
    "Beta": stats.beta,
    "Chi-square": stats.chi2,
    "Student-t": stats.t,
    "Pareto": stats.pareto,
    "Uniform": stats.uniform,
    "Laplace": stats.laplace,
    "Logistic": stats.logistic,
}

st.title("Histogram Fitter")
st.write(
    "Upload or paste numerical data, choose a probability distribution, and "
    "automatically fit a curve to your histogram. You can also adjust parameters "
    "manually to explore different fits."
)

# Sidebar: data input
with st.sidebar.expander("Data Input", expanded=True):
    mode = st.radio("Input method", ["Paste numbers", "Upload CSV"])
    pasted_text = st.text_area("Paste data (comma/space/newline separated)")
    uploaded = st.file_uploader("CSV or TXT", type=["csv", "txt"])
    csv_has_header = st.checkbox("CSV has header", True)
    csv_col_name = st.text_input("Column name (optional)", value="")

# Sidebar: fitting options
with st.sidebar.expander("Fitting Options"):
    dist_name = st.selectbox("Distribution", list(DISTRIBUTIONS.keys()))
    bins = st.slider("Histogram bins", 5, 80, 30)
    auto_fit = st.checkbox("Automatic fit", True)
    manual_mode = st.checkbox("Manual fitting", False)

data = np.array([])

if mode == "Paste numbers":
    data = parse_text_data(pasted_text)

elif mode == "Upload CSV" and uploaded is not None:
    try:
        df = pd.read_csv(uploaded) if uploaded.name.endswith(".csv") else pd.read_table(uploaded)

        if csv_col_name and csv_col_name in df.columns:
            col = df[csv_col_name].dropna()
        else:
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) == 0:
                st.error("No numeric columns found in file.")
                st.stop()
            col = df[numeric_cols[0]].dropna()

        data = col.to_numpy()
    except Exception as e:
        st.error(f"Failed to read file: {e}")
        st.stop()

if data.size == 0:
    st.info("No data loaded yet.")
    st.stop()

data = data[np.isfinite(data)]
if data.size == 0:
    st.error("Data contains no valid numerical values.")
    st.stop()

dist = DISTRIBUTIONS[dist_name]
fitted_params = None
param_dict = {}
metrics = {}

if auto_fit and not manual_mode:
    try:
        fitted_params, param_dict = fit_distribution(dist, data)
    except Exception as e:
        st.warning(f"Fit failed: {e}")

if manual_mode and fitted_params is None:
    try:
        fitted_params, param_dict = fit_distribution(dist, data)
    except Exception:
        num_shapes = len(dist.shapes.split(",")) if dist.shapes else 0
        fallback = [1.0] * num_shapes + [np.min(data), np.std(data)]
        fitted_params = tuple(fallback)

        names = []
        if dist.shapes:
            names = [s.strip() for s in dist.shapes.split(",")]
        names += ["loc", "scale"]
        param_dict = {n: float(v) for n, v in zip(names, fitted_params)}

centers, density = histogram_density(data, bins=bins)
pdf_func = make_pdf_function(dist, fitted_params) if fitted_params else (lambda x: np.zeros_like(x))

# Evaluate fit quality
try:
    sse = compute_sse(centers, density, pdf_func)
    max_err = compute_max_abs_error(centers, density, pdf_func)

    if fitted_params:
        cdf = lambda x: dist.cdf(x, *fitted_params[:-2], loc=fitted_params[-2], scale=fitted_params[-1])
        ks_stat, ks_p = stats.kstest(data, cdf)
    else:
        ks_stat, ks_p = np.nan, np.nan

    metrics = {
        "SSE": sse,
        "MaxAbsError": max_err,
        "KS_stat": float(ks_stat),
        "KS_pvalue": float(ks_p),
    }
except Exception as e:
    metrics = {"error": str(e)}

tab1, tab2, tab3 = st.tabs(["Visualization", "Fit Parameters", "Manual Fit & Export"])

with tab1:
    st.header("Histogram and Fitted Distribution")

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(data, bins=bins, density=True, alpha=0.6, edgecolor="black")

    try:
        x = np.linspace(np.min(data), np.max(data), 400)
        ax.plot(x, pdf_func(x), linewidth=2, label=f"{dist_name} fit")
    except Exception:
        pass

    ax.set_xlabel("x")
    ax.set_ylabel("Density")
    ax.legend()
    st.pyplot(fig)

    st.subheader("Fit Summary")
    st.write(f"Data points: {data.size}")
    st.write(f"Distribution: {dist_name}")
    st.write(metrics)

with tab2:
    st.header("Fitted Parameters")

    if param_dict:
        st.table(pd.DataFrame.from_dict(param_dict, orient="index", columns=["value"]))
    else:
        st.info("No parameters available.")

    st.subheader("Goodness of Fit")
    st.json(metrics)

with tab3:
    st.header("Manual Parameter Adjustment")

    if manual_mode:
        shapes = []
        if dist.shapes:
            shapes = [s.strip() for s in dist.shapes.split(",")]

        manual_params = []

        for s in shapes:
            manual_params.append(
                st.slider(s, -10.0, 10.0, float(param_dict.get(s, 1.0)))
            )

        loc_default = param_dict.get("loc", np.min(data))
        scale_default = param_dict.get("scale", np.std(data))

        loc = st.slider("loc", float(np.min(data)), float(np.max(data)), float(loc_default))
        scale = st.slider("scale", 1e-6, float(np.ptp(data) + 1e-6), float(abs(scale_default)))

        manual_params.extend([loc, scale])
        manual_tuple = tuple(manual_params)

        manual_pdf = make_pdf_function(dist, manual_tuple)

        fig2, ax2 = plt.subplots(figsize=(8, 4))
        ax2.hist(data, bins=bins, density=True, alpha=0.6, edgecolor="black")
        ax2.plot(xs := np.linspace(np.min(data), np.max(data), 400),
                 manual_pdf(xs), linewidth=2, label="Manual fit")
        ax2.legend()
        st.pyplot(fig2)

        st.write({
            "SSE_manual": compute_sse(centers, density, manual_pdf),
            "MaxAbsError_manual": compute_max_abs_error(centers, density, manual_pdf)
        })

        out_df = pd.DataFrame({
            "param": shapes + ["loc", "scale"],
            "value": manual_params
        })
        st.download_button(
            "Download manual parameters (CSV)",
            out_df.to_csv(index=False).encode("utf-8"),
            "manual_fit_params.csv"
        )
    else:
        st.info("Enable manual mode to adjust parameters.")

    st.markdown("---")

    if param_dict and not manual_mode:
        export_df = pd.DataFrame({
            "param": list(param_dict.keys()),
            "value": list(param_dict.values())
        })
        st.download_button(
            "Download automatic fit parameters (CSV)",
            export_df.to_csv(index=False).encode("utf-8"),
            "auto_fit_params.csv"
        )