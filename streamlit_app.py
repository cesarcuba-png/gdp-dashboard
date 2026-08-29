import sys
import warnings
from typing import List, Dict

import numpy as np
import pandas as pd
import scipy.stats as stats
from scipy.optimize import minimize
import yfinance as yf

import plotly.graph_objects as go
import plotly.express as px
import streamlit as st

warnings.filterwarnings('ignore')

# ==============================================================================
# STREAMLIT CONFIG
# ==============================================================================
st.set_page_config(
    page_title="Quant Portfolio Engine — GS / JPM Institutional Analytics",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem; font-weight: 800;
        background: linear-gradient(90deg, #c9a84c, #d4af37, #f0d060);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        margin-bottom: 0px;
    }
    .sub-header { font-size: 1.0rem; color: #94a3b8; margin-bottom: 20px; }
    .kpi-card {
        background-color: #0f172a; padding: 14px 18px; border-radius: 12px;
        border: 1px solid #1e293b; text-align: center;
        box-shadow: 0 4px 6px -1px rgba(0,0,0,0.1);
    }
    .kpi-title { font-size: 0.78rem; font-weight: 600; color: #94a3b8; text-transform: uppercase; letter-spacing: 0.05em; }
    .kpi-value { font-size: 1.45rem; font-weight: 800; margin-top: 4px; font-family: 'Courier New', monospace; }
    .kpi-sub { font-size: 0.7rem; color: #64748b; margin-top: 2px; }
    .profile-badge-con { background:#1a5276; color:#fff; padding:4px 12px; border-radius:20px; font-size:0.75rem; font-weight:700; }
    .profile-badge-mod { background:#f39c12; color:#0f172a; padding:4px 12px; border-radius:20px; font-size:0.75rem; font-weight:700; }
    .profile-badge-agr { background:#c0392b; color:#fff; padding:4px 12px; border-radius:20px; font-size:0.75rem; font-weight:700; }
    .screening-pass { color: #10b981; font-weight: 700; }
    .screening-fail { color: #ef4444; font-weight: 700; }
</style>
""", unsafe_allow_html=True)

# ==============================================================================
# 1. GLOBAL PARAMETERS ()
# ==============================================================================
PERIODO = '5y'
RF_RATE_DEFAULT = 0.045
REBALANCEO_DIAS = 63
COSTO_REBALANCEO_BPS = 15
N_SIMS_MC = 5000
MC_HORIZON_YEARS = 5

PERFILES = {
    'Conservador': {
        'descripcion': 'Preservación de capital con ingresos estables. Max Drawdown tolerable: -10%',
        'vol_max': 0.12, 'bounds': (0.02, 0.20), 'objetivo': 'min_volatility', 'color': '#1a5276',
    },
    'Moderado': {
        'descripcion': 'Crecimiento balanceado con control de riesgo. Max Drawdown tolerable: -20%',
        'vol_max': 0.20, 'bounds': (0.02, 0.18), 'objetivo': 'max_sharpe', 'color': '#f39c12',
    },
    'Agresivo': {
        'descripcion': 'Máximo retorno asumiendo alta volatilidad. Max Drawdown tolerable: -40%',
        'vol_max': 0.35, 'bounds': (0.01, 0.25), 'objetivo': 'max_return', 'color': '#c0392b',
    }
}

SCREENING_PARAMS = {
    'Conservador': {'sharpe_min': 0.20, 'vol_max': 0.30, 'mdd_max': -0.30, 'alpha_min': -0.05, 'corr_max': 0.75, 'max_activos': 10},
    'Moderado':    {'sharpe_min': 0.30, 'vol_max': 0.45, 'mdd_max': -0.40, 'alpha_min': -0.02, 'corr_max': 0.70, 'max_activos': 12},
    'Agresivo':    {'sharpe_min': 0.05, 'vol_max': 0.75, 'mdd_max': -0.65, 'alpha_min': -0.15, 'corr_max': 0.80, 'max_activos': 10},
}

# ==============================================================================
# 2. UNIVERSO DE ACTIVOS (41 candidatos + 2 benchmarks)
# ==============================================================================
UNIVERSO_CANDIDATOS = [
    'NVDA', 'AMD', 'INTC', 'AVGO', 'QCOM',
    'MSFT', 'GOOGL', 'AMZN', 'META', 'CRM', 'ORCL', 'SAP', 'IBM',
    'DELL', 'AAPL', 'HPQ', 'SONY',
    'TSLA', 'TM', 'F', 'GM', 'RIVN',
    'LLY', 'PFE', 'MRK', 'JNJ', 'AZN', 'UNH',
    'JPM', 'GS', 'BAC', 'C', 'WFC', 'BAP',
    'KO', 'PEP', 'PG', 'KMB', 'WMT',
    'GLD', 'XLE',
]
BENCHMARKS = ['SPY', 'QQQ']
TODOS_TICKERS = list(set(UNIVERSO_CANDIDATOS + BENCHMARKS))
TICKERS_SIN_EBITDA = ['GLD', 'XLE', 'SPY', 'QQQ', 'JPM', 'GS', 'BAC', 'C', 'WFC', 'BAP']

UNIVERSO_INFO = [
    {"ticker": "NVDA", "name": "Nvidia Corp", "sector": "Semiconductores / IA"},
    {"ticker": "AMD", "name": "Advanced Micro Devices", "sector": "Semiconductores"},
    {"ticker": "INTC", "name": "Intel Corp", "sector": "Semiconductores"},
    {"ticker": "AVGO", "name": "Broadcom Inc", "sector": "Semiconductores"},
    {"ticker": "QCOM", "name": "Qualcomm Inc", "sector": "Semiconductores"},
    {"ticker": "MSFT", "name": "Microsoft Corp", "sector": "Software / Cloud"},
    {"ticker": "GOOGL", "name": "Alphabet Inc", "sector": "Servicios Digitales"},
    {"ticker": "AMZN", "name": "Amazon.com Inc", "sector": "Consumo Cíclico / AWS"},
    {"ticker": "META", "name": "Meta Platforms", "sector": "Redes Sociales / IA"},
    {"ticker": "CRM", "name": "Salesforce Inc", "sector": "SaaS / CRM"},
    {"ticker": "ORCL", "name": "Oracle Corp", "sector": "Cloud / DB"},
    {"ticker": "SAP", "name": "SAP SE", "sector": "ERP / Enterprise"},
    {"ticker": "IBM", "name": "IBM Corp", "sector": "Hybrid Cloud / IA"},
    {"ticker": "DELL", "name": "Dell Technologies", "sector": "Hardware / AI Servers"},
    {"ticker": "AAPL", "name": "Apple Inc", "sector": "Hardware / Consumo"},
    {"ticker": "HPQ", "name": "HP Inc", "sector": "Hardware"},
    {"ticker": "SONY", "name": "Sony Group", "sector": "Electrónica / Gaming"},
    {"ticker": "TSLA", "name": "Tesla Inc", "sector": "Automotriz / EV"},
    {"ticker": "TM", "name": "Toyota Motor", "sector": "Automotriz"},
    {"ticker": "F", "name": "Ford Motor", "sector": "Automotriz"},
    {"ticker": "GM", "name": "General Motors", "sector": "Automotriz"},
    {"ticker": "RIVN", "name": "Rivian Automotive", "sector": "EV Startup"},
    {"ticker": "LLY", "name": "Eli Lilly", "sector": "Salud / GLP-1"},
    {"ticker": "PFE", "name": "Pfizer Inc", "sector": "Pharma"},
    {"ticker": "MRK", "name": "Merck & Co", "sector": "Pharma"},
    {"ticker": "JNJ", "name": "Johnson & Johnson", "sector": "Salud Diversificado"},
    {"ticker": "AZN", "name": "AstraZeneca", "sector": "Pharma / Oncología"},
    {"ticker": "UNH", "name": "UnitedHealth Group", "sector": "Seguros Salud"},
    {"ticker": "JPM", "name": "JPMorgan Chase", "sector": "Banca Global"},
    {"ticker": "GS", "name": "Goldman Sachs", "sector": "Banca de Inversión"},
    {"ticker": "BAC", "name": "Bank of America", "sector": "Banca Comercial"},
    {"ticker": "C", "name": "Citigroup", "sector": "Banca Global"},
    {"ticker": "WFC", "name": "Wells Fargo", "sector": "Banca Comercial"},
    {"ticker": "BAP", "name": "Credicorp Ltd", "sector": "Financiero LATAM / BVL"},
    {"ticker": "KO", "name": "Coca-Cola Co", "sector": "Consumo Defensivo"},
    {"ticker": "PEP", "name": "PepsiCo Inc", "sector": "Consumo Defensivo"},
    {"ticker": "PG", "name": "Procter & Gamble", "sector": "Consumo Defensivo"},
    {"ticker": "KMB", "name": "Kimberly-Clark", "sector": "Consumo Defensivo"},
    {"ticker": "WMT", "name": "Walmart Inc", "sector": "Retail / Consumo"},
    {"ticker": "GLD", "name": "SPDR Gold Shares", "sector": "ETF Cobertura Oro"},
    {"ticker": "XLE", "name": "Energy Select SPDR", "sector": "ETF Energía"},
    {"ticker": "SPY", "name": "S&P 500 ETF", "sector": "Benchmark"},
    {"ticker": "QQQ", "name": "Invesco QQQ", "sector": "Benchmark Tech"},
]
DF_UNIVERSO = pd.DataFrame(UNIVERSO_INFO)

# ==============================================================================
# 3. DATA DOWNLOAD
# ==============================================================================
@st.cache_data(ttl=3600, show_spinner="Descargando datos de Yahoo Finance...")
def descargar_datos(tickers, period='5y'):
    try:
        data = yf.download(tickers, period=period, progress=False, auto_adjust=True)
        if isinstance(data, pd.DataFrame) and 'Close' in data:
            prices = data['Close']
        elif isinstance(data, pd.DataFrame) and 'Adj Close' in data:
            prices = data['Adj Close']
        else:
            prices = data
        prices = prices.dropna(how='all', axis=1)
        min_obs = int(len(prices) * 0.80)
        prices = prices.loc[:, prices.count() >= min_obs].ffill()
        if len(prices) > 100:
            return prices, False
    except Exception:
        pass
    np.random.seed(42)
    dates = pd.date_range(end=pd.Timestamp.today(), periods=1260, freq='B')
    mock = {}
    for t in tickers:
        drift = 0.22 if t in UNIVERSO_CANDIDATOS[:15] else 0.10
        vol = 0.28 if t in ['NVDA', 'AMD', 'TSLA'] else 0.16
        ret = np.random.normal(drift/252, vol/np.sqrt(252), len(dates))
        mock[t] = 100.0 * np.exp(np.cumsum(ret))
    return pd.DataFrame(mock, index=dates), True

@st.cache_data(ttl=3600, show_spinner="Descargando datos fundamentales (EBITDA)...")
def descargar_ebitda(tickers):
    margins = {}
    for ticker in tickers:
        try:
            info = yf.Ticker(ticker).info
            ebitda = info.get('ebitda', None)
            revenue = info.get('totalRevenue', None)
            if ebitda and revenue and revenue > 0:
                margins[ticker] = ebitda / revenue
            else:
                margins[ticker] = 0.0
        except Exception:
            margins[ticker] = 0.0
    df = pd.DataFrame([{'Ticker': k, 'EBITDA_Margin': v} for k, v in margins.items()])
    df.loc[df['Ticker'].isin(TICKERS_SIN_EBITDA), 'EBITDA_Margin'] = np.nan
    return df

# ==============================================================================
# 4. QUANTITATIVE METRICS ENGINE
# ==============================================================================
def calcular_metricas(prices, rf=RF_RATE_DEFAULT):
    returns = prices.pct_change().dropna()
    years = max(len(prices) / 252.0, 0.5)
    bench_col = 'SPY' if 'SPY' in prices.columns else prices.columns[0]
    bench_returns = returns[bench_col]
    rf_daily = (1 + rf) ** (1/252) - 1
    metricas = []
    for ticker in prices.columns:
        p = prices[ticker]
        ret = returns[ticker]
        cagr = (p.iloc[-1] / p.iloc[0]) ** (1/years) - 1.0
        vol = ret.std() * np.sqrt(252)
        sharpe = (cagr - rf) / vol if vol > 0 else 0.0
        excess_below = np.minimum(ret - rf_daily, 0)
        downside_dev = np.sqrt((excess_below**2).mean()) * np.sqrt(252)
        sortino = (cagr - rf) / downside_dev if downside_dev > 0 else 0.0
        cum = (1 + ret).cumprod()
        peak = cum.cummax()
        mdd = ((cum - peak) / peak).min()
        excess_ret = ret - rf_daily
        excess_bench = bench_returns - rf_daily
        if excess_bench.var() > 0:
            beta = excess_ret.cov(excess_bench) / excess_bench.var()
            alpha_daily = excess_ret.mean() - beta * excess_bench.mean()
            alpha = alpha_daily * 252
        else:
            beta, alpha = 1.0, 0.0
        calmar = cagr / abs(mdd) if mdd != 0 else 0.0
        metricas.append({
            'Ticker': ticker, 'CAGR (%)': cagr*100, 'Volatilidad (%)': vol*100,
            'Sharpe': sharpe, 'Sortino': sortino, 'Max Drawdown (%)': mdd*100,
            'Beta': beta, 'Alpha (%)': alpha*100, 'Calmar': calmar
        })
    return pd.DataFrame(metricas).sort_values('Sharpe', ascending=False)

# ==============================================================================
# 5. ALGORITHMIC SCREENING: MULTIFACTOR + CORRELATION FILTER
# ==============================================================================
def criba_algoritmica(df_metricas, returns_df, params, df_ebitda):
    candidatos = df_metricas[~df_metricas['Ticker'].isin(BENCHMARKS)].copy()
    mask = (
        (candidatos['Sharpe'] >= params['sharpe_min']) &
        (candidatos['Volatilidad (%)'] <= params['vol_max'] * 100) &
        (candidatos['Max Drawdown (%)'] >= params['mdd_max'] * 100) &
        (candidatos['Alpha (%)'] >= params['alpha_min'] * 100)
    )
    supervivientes = candidatos[mask].copy()
    eliminados_df = candidatos[~mask][['Ticker', 'Sharpe', 'Volatilidad (%)', 'Max Drawdown (%)']].copy()
    if len(supervivientes) < 3:
        supervivientes = candidatos.nlargest(params['max_activos'], 'Sharpe')

    supervivientes = supervivientes.merge(df_ebitda, on='Ticker', how='left')
    mediana_ebitda = supervivientes['EBITDA_Margin'].median()
    supervivientes['EBITDA_Margin'] = supervivientes['EBITDA_Margin'].fillna(mediana_ebitda if pd.notna(mediana_ebitda) else 0.0)
    for col in ['Sharpe', 'Alpha (%)', 'EBITDA_Margin']:
        col_min, col_max = supervivientes[col].min(), supervivientes[col].max()
        rng = col_max - col_min if col_max != col_min else 1.0
        supervivientes[f'{col}_norm'] = (supervivientes[col] - col_min) / rng
    supervivientes['Score_Compuesto'] = (
        0.50 * supervivientes['Sharpe_norm'] +
        0.30 * supervivientes['Alpha (%)_norm'] +
        0.20 * supervivientes['EBITDA_Margin_norm']
    )
    supervivientes = supervivientes.sort_values('Score_Compuesto', ascending=False)

    tickers_ranked = supervivientes['Ticker'].tolist()
    tickers_disponibles = [t for t in tickers_ranked if t in returns_df.columns]
    seleccionados = []
    rechazados_corr = []
    log_entries = []

    for ticker in tickers_disponibles:
        if len(seleccionados) >= params['max_activos']:
            break
        if not seleccionados:
            seleccionados.append(ticker)
            log_entries.append((ticker, 'ACEPTADO', 0.0, '-', 'Primero del ranking'))
            continue
        corrs = returns_df[[ticker] + seleccionados].corr()[ticker]
        max_corr = corrs.drop(ticker).abs().max()
        par_max = corrs.drop(ticker).abs().idxmax()
        if max_corr <= params['corr_max']:
            seleccionados.append(ticker)
            log_entries.append((ticker, 'ACEPTADO', max_corr, par_max, f'corr {max_corr:.2f} < {params["corr_max"]}'))
        else:
            rechazados_corr.append((ticker, par_max, max_corr))
            log_entries.append((ticker, 'RECHAZADO', max_corr, par_max, f'corr {max_corr:.2f} > {params["corr_max"]}'))

    return seleccionados, supervivientes, eliminados_df, rechazados_corr, log_entries

# ==============================================================================
# 6. MARKOWITZ OPTIMIZATION
# ==============================================================================
def optimizar_portafolio(tickers, returns, objetivo='max_sharpe', bounds_range=(0.02, 0.20), vol_max=0.25, rf=RF_RATE_DEFAULT):
    tickers_valid = [t for t in tickers if t in returns.columns]
    if len(tickers_valid) < 3:
        return None, None, None
    mean_ret = returns[tickers_valid].mean() * 252
    cov_mat = returns[tickers_valid].cov() * 252
    n = len(tickers_valid)

    def neg_sharpe(w):
        p_ret = np.sum(mean_ret.values * w)
        p_vol = np.sqrt(np.dot(w.T, np.dot(cov_mat.values, w)))
        return -(p_ret - rf) / p_vol if p_vol > 1e-10 else 0

    def portfolio_vol(w):
        return np.sqrt(np.dot(w.T, np.dot(cov_mat.values, w)))

    def neg_return(w):
        return -np.sum(mean_ret.values * w)

    bounds = tuple((bounds_range[0], bounds_range[1]) for _ in range(n))
    cons = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0}]

    if objetivo == 'min_volatility':
        res = minimize(portfolio_vol, np.ones(n)/n, method='SLSQP', bounds=bounds, constraints=cons)
    elif objetivo == 'max_return':
        cons.append({'type': 'ineq', 'fun': lambda w: vol_max - portfolio_vol(w)})
        res = minimize(neg_return, np.ones(n)/n, method='SLSQP', bounds=bounds, constraints=cons)
    else:
        cons.append({'type': 'ineq', 'fun': lambda w: vol_max - portfolio_vol(w)})
        res = minimize(neg_sharpe, np.ones(n)/n, method='SLSQP', bounds=bounds, constraints=cons)

    w_opt = res.x if res.success else np.ones(n) / n
    port_ret = np.sum(mean_ret.values * w_opt)
    port_vol = np.sqrt(np.dot(w_opt.T, np.dot(cov_mat.values, w_opt)))
    port_sharpe = (port_ret - rf) / port_vol if port_vol > 0 else 0
    return tickers_valid, w_opt, {'CAGR': port_ret, 'Vol': port_vol, 'Sharpe': port_sharpe}

def simular_rebalanceo(returns_sub, tickers, pesos_target, freq=REBALANCEO_DIAS, costo_bps=COSTO_REBALANCEO_BPS):
    valid_t = [t for t in tickers if t in returns_sub.columns]
    if not valid_t:
        return pd.Series(dtype=float)
    ret = returns_sub[valid_t]
    n_days = len(ret)
    weights = np.array(pesos_target[:len(valid_t)], dtype=float)
    daily_returns = np.zeros(n_days)
    for t in range(n_days):
        day_ret = ret.iloc[t].values
        daily_returns[t] = np.dot(weights, day_ret)
        weights = weights * (1 + day_ret)
        s = weights.sum()
        if s > 0:
            weights = weights / s
        if (t + 1) % freq == 0 and t < n_days - 1:
            turnover = np.sum(np.abs(weights - pesos_target[:len(valid_t)])) / 2
            cost = turnover * (costo_bps / 10000)
            daily_returns[t] -= cost
            weights = np.array(pesos_target[:len(valid_t)], dtype=float)
    return pd.Series(daily_returns, index=ret.index)

# ==============================================================================
# 7. SIDEBAR CONTROLS
# ==============================================================================
st.markdown('<p class="main-header">📈 Quant Portfolio Engine — Análisis Institucional GS / JPM</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Criba Algorítmica Multifactorial + Optimización Markowitz + Monte Carlo + Walk-Forward | Framework Senior Analyst</p>', unsafe_allow_html=True)

st.sidebar.header("⚙️ Configuración del Sistema")

view_mode = st.sidebar.radio(
    "👁️ Vista del Dashboard:",
    ["Vista Operativa (Mesa de Trading)", "Vista Gerencial (Comité de Inversiones)"]
)
badge_class = "profile-badge-mod" if "Operativa" in view_mode else "profile-badge-con"
st.sidebar.markdown(f'<span class="{badge_class}">{"MODO OPERATIVO" if "Operativa" in view_mode else "MODO GERENCIAL"}</span>', unsafe_allow_html=True)

st.sidebar.markdown("---")
st.sidebar.subheader("💰 Capital & Macro")
capital_pen = st.sidebar.number_input("Capital Inicial (S/ PEN):", min_value=10000.0, value=1_000_000.0, step=50000.0)
fx_rate = st.sidebar.number_input("Tipo de Cambio PEN/USD:", min_value=1.0, value=3.35, step=0.05)
capital_usd = capital_pen / fx_rate
rf_rate_input = st.sidebar.number_input("Tasa Libre Riesgo (%):", min_value=0.0, max_value=15.0, value=4.5, step=0.25) / 100.0

st.sidebar.markdown("---")
st.sidebar.subheader("🎯 Perfil de Inversor Activo")
perfil_activo = st.sidebar.selectbox(
    "Selecciona el perfil para la vista operativa:",
    list(PERFILES.keys()),
    index=1,
    format_func=lambda x: {"Conservador": "🛡️ Conservador", "Moderado": "🎯 Moderado", "Agresivo": "🚀 Agresivo"}[x]
)

st.sidebar.markdown("---")
st.sidebar.subheader("📊 Ventana de Datos")
periodo_sel = st.sidebar.selectbox("Periodo Histórico:", ["3y", "5y", "10y"], index=1)

st.sidebar.markdown("---")
st.sidebar.info(f"**Capital:** S/ {capital_pen:,.0f} PEN → ${capital_usd:,.0f} USD\n\n"
                f"**Rf:** {rf_rate_input*100:.1f}% | **Universo:** {len(TODOS_TICKERS)} activos\n\n"
                f"**Rebalanceo:** ~trimestral ({REBALANCEO_DIAS}d) | Costo: {COSTO_REBALANCEO_BPS} bps")

# ==============================================================================
# 8. LOAD DATA & RUN FULL PIPELINE
# ==============================================================================
precios_df, datos_sinteticos = descargar_datos(TODOS_TICKERS, periodo_sel)
returns_df = precios_df.pct_change().dropna()
df_ebitda = descargar_ebitda(UNIVERSO_CANDIDATOS)
df_metricas = calcular_metricas(precios_df, rf=rf_rate_input)

if datos_sinteticos:
    st.warning("⚠️ Yahoo Finance no disponible. Usando datos SINTÉTICOS (los resultados son ilustrativos, NO reales).")

# Run algorithmic screening for all profiles
seleccion_por_perfil = {}
for pnombre, sparams in SCREENING_PARAMS.items():
    sel, scores, elim, rech, log = criba_algoritmica(df_metricas, returns_df, sparams, df_ebitda)
    seleccion_por_perfil[pnombre] = {'tickers': sel, 'scores': scores, 'eliminados': elim, 'rechazados_corr': rech, 'log': log}

# Optimize each profile
resultados_perfiles = {}
for perfil, config in PERFILES.items():
    tickers_p = seleccion_por_perfil[perfil]['tickers']
    tickers_v, pesos, metricas_port = optimizar_portafolio(
        tickers_p, returns_df, objetivo=config['objetivo'],
        bounds_range=config['bounds'], vol_max=config['vol_max'], rf=rf_rate_input
    )
    if tickers_v is None:
        continue
    ret_port = simular_rebalanceo(returns_df, tickers_v, pesos)
    cum_port = (1 + ret_port).cumprod()
    mdd_port = ((cum_port - cum_port.cummax()) / cum_port.cummax()).min()
    resultados_perfiles[perfil] = {
        'tickers': tickers_v, 'pesos': pesos, 'metricas': metricas_port,
        'config': config, 'ret_port': ret_port, 'mdd_real': mdd_port
    }

# Active profile data
act_data = resultados_perfiles.get(perfil_activo)

# ==============================================================================
# 9. KPI CARDS (for active profile)
# ==============================================================================
if act_data:
    ret_p = act_data['ret_port']
    var_95 = np.percentile(ret_p, 5)
    cvar_95 = ret_p[ret_p <= var_95].mean() if (ret_p <= var_95).any() else var_95
    hhi = np.sum(act_data['pesos']**2)
    n_eff = 1.0 / hhi if hhi > 0 else len(act_data['tickers'])

    spy_ret = returns_df['SPY'] if 'SPY' in returns_df.columns else returns_df.iloc[:, 0]
    excess_r = ret_p - spy_ret.reindex(ret_p.index, fill_value=0)
    info_ratio = (excess_r.mean() * 252) / (excess_r.std() * np.sqrt(252)) if excess_r.std() > 0 else 0
    tracking_error = excess_r.std() * np.sqrt(252)

    rf_daily = (1 + rf_rate_input) ** (1/252) - 1
    excess_below = np.minimum(ret_p - rf_daily, 0)
    dd_comp = np.sqrt((excess_below**2).mean()) * np.sqrt(252)
    sortino_port = (act_data['metricas']['CAGR'] - rf_rate_input) / dd_comp if dd_comp > 0 else 0

    st.markdown(f"### 📊 KPIs Institucionales — Perfil {perfil_activo}")
    kc1, kc2, kc3, kc4 = st.columns(4)
    kc5, kc6, kc7, kc8 = st.columns(4)

    def kpi_card(col, title, value, color, sub=""):
        with col:
            st.markdown(f'<div class="kpi-card"><div class="kpi-title">{title}</div>'
                        f'<div class="kpi-value" style="color:{color};">{value}</div>'
                        f'<div class="kpi-sub">{sub}</div></div>', unsafe_allow_html=True)

    kpi_card(kc1, "Sharpe Ratio", f"{act_data['metricas']['Sharpe']:.2f}", "#06b6d4", "Retorno / Riesgo total")
    kpi_card(kc2, "Sortino Ratio", f"{sortino_port:.2f}", "#10b981", "Penaliza solo downside")
    kpi_card(kc3, "CAGR (Anual)", f"{act_data['metricas']['CAGR']*100:.2f}%", "#34d399", "Crecimiento compuesto")
    kpi_card(kc4, "Max Drawdown", f"{act_data['mdd_real']*100:.2f}%", "#f87171", "Caída máxima histórica")
    kpi_card(kc5, "VaR 95% (Diario)", f"{var_95*100:.2f}%", "#fbbf24", f"Pérdida máx: S/ {capital_pen*abs(var_95):,.0f}")
    kpi_card(kc6, "CVaR 95% (ES)", f"{cvar_95*100:.2f}%", "#f59e0b", f"Cola esperada: S/ {capital_pen*abs(cvar_95):,.0f}")
    kpi_card(kc7, "Tracking Error", f"{tracking_error*100:.2f}%", "#a855f7", "Desviación vs SPY")
    kpi_card(kc8, "Info Ratio vs SPY", f"{info_ratio:.2f}", "#64748b", f"HHI: {hhi:.3f} | N_eff: {n_eff:.1f}")

st.markdown("---")

# ==============================================================================
# 10. VIEW-DEPENDENT CONTENT
# ==============================================================================
IS_OPERATIVA = "Operativa" in view_mode

if IS_OPERATIVA:
    # ══════════════════════════════════════════════════════════════════════════
    # VISTA OPERATIVA — Mesa de Trading: granularidad, ejecución, datos crudos
    # ══════════════════════════════════════════════════════════════════════════
    tab_o1, tab_o2, tab_o3, tab_o4, tab_o5 = st.tabs([
        "🔬 Criba Algorítmica",
        "🎯 Órdenes de Compra",
        "🔗 Correlaciones",
        "📊 Frontera Eficiente & Backtest",
        "🔍 Universo & Métricas",
    ])

    # --- OPERATIVA TAB 1: Screening detallado ---
    with tab_o1:
        st.subheader("🔬 Criba Algorítmica Multifactorial — Detalle de Selección")
        st.caption("Proceso: Filtro de Umbrales → Score (0.50×Sharpe + 0.30×Alpha + 0.20×EBITDA) → Eliminación por Correlación")
        st.info("**Nota:** El universo de 43 activos contiene survivorship bias. Los resultados representan el 'mejor portafolio dentro de este universo'.")

        for pnombre in PERFILES:
            sel_data = seleccion_por_perfil[pnombre]
            sparams = SCREENING_PARAMS[pnombre]
            with st.expander(f"📋 {pnombre.upper()} — {len(sel_data['tickers'])} activos seleccionados", expanded=(pnombre == perfil_activo)):
                col_params, col_scores = st.columns([4, 8])
                with col_params:
                    st.markdown("**Umbrales del Filtro:**")
                    st.markdown(f"- Sharpe ≥ {sparams['sharpe_min']}")
                    st.markdown(f"- Volatilidad ≤ {sparams['vol_max']*100:.0f}%")
                    st.markdown(f"- Max Drawdown ≥ {sparams['mdd_max']*100:.0f}%")
                    st.markdown(f"- Alpha ≥ {sparams['alpha_min']*100:.0f}%")
                    st.markdown(f"- Correlación máxima: {sparams['corr_max']}")
                    st.markdown(f"- Máx activos: {sparams['max_activos']}")
                    st.markdown(f"\n**Eliminados por umbrales:** {len(sel_data['eliminados'])}")
                    st.markdown(f"**Eliminados por correlación:** {len(sel_data['rechazados_corr'])}")
                with col_scores:
                    if not sel_data['scores'].empty:
                        display_cols = ['Ticker', 'Score_Compuesto', 'Sharpe', 'Alpha (%)', 'EBITDA_Margin', 'Volatilidad (%)']
                        avail_cols = [c for c in display_cols if c in sel_data['scores'].columns]
                        df_show = sel_data['scores'][avail_cols].head(15).copy()
                        df_show = df_show.merge(
                            DF_UNIVERSO[['ticker', 'name']].rename(columns={'ticker': 'Ticker', 'name': 'Descripción'}),
                            on='Ticker',
                            how='left'
                        )
                        df_show['Seleccionado'] = df_show['Ticker'].apply(lambda x: '✅' if x in sel_data['tickers'] else '❌')
                        st.dataframe(df_show, use_container_width=True, hide_index=True)
                st.markdown("**Log de Filtro Correlacional:**")
                log_df = pd.DataFrame(sel_data['log'], columns=['Ticker', 'Estado', 'Max Corr', 'Par', 'Razón'])
                log_df = log_df.merge(
                    DF_UNIVERSO[['ticker', 'name']].rename(columns={'ticker': 'Ticker', 'name': 'Descripción'}),
                    on='Ticker',
                    how='left'
                )
                st.dataframe(log_df, use_container_width=True, hide_index=True)

    # --- OPERATIVA TAB 2: Órdenes de compra ---
    with tab_o2:
        if act_data:
            st.subheader(f"🎯 Orden de Ejecución — Perfil {perfil_activo} ({PERFILES[perfil_activo]['objetivo'].replace('_',' ').title()})")
            st.caption(f"{PERFILES[perfil_activo]['descripcion']}")
            col_chart, col_table = st.columns([4, 8])
            pesos_dict = dict(zip(act_data['tickers'], act_data['pesos']))
            active_weights = {k: v for k, v in pesos_dict.items() if v > 0.005}

            with col_chart:
                fig_donut = px.pie(names=list(active_weights.keys()), values=list(active_weights.values()),
                    hole=0.45, color_discrete_sequence=px.colors.sequential.Tealgrn)
                fig_donut.update_layout(margin=dict(t=20, b=20, l=20, r=20), height=380)
                st.plotly_chart(fig_donut, use_container_width=True)

            with col_table:
                st.markdown("#### 📝 Orden Discreta de Compra — Lote por Acción")
                latest_prices = precios_df.iloc[-1]
                alloc_data = []
                cash_left = capital_usd
                for ticker, w in active_weights.items():
                    target_val = capital_usd * w
                    price = latest_prices.get(ticker, 100.0) if ticker in latest_prices.index else 100.0
                    shares = int(target_val // price) if price > 0 else 0
                    val_usd = shares * price
                    cash_left -= val_usd
                    alloc_data.append({
                        "Ticker": ticker, "Peso Óptimo": f"{w*100:.2f}%",
                        "# Acciones": shares, "Precio USD": f"${price:,.2f}",
                        "Monto USD": f"${val_usd:,.2f}", "Monto PEN": f"S/ {val_usd*fx_rate:,.2f}"
                    })
                st.dataframe(pd.DataFrame(alloc_data), use_container_width=True, hide_index=True)
                st.info(f"**Efectivo remanente:** ${cash_left:,.2f} USD (S/ {cash_left*fx_rate:,.2f} PEN) para comisiones y slippage")

            st.markdown("---")
            st.markdown("#### Métricas de Riesgo Avanzadas del Portafolio Activo")
            col_r1, col_r2, col_r3 = st.columns(3)
            ret_p = act_data['ret_port']
            var99 = np.percentile(ret_p, 1)
            cvar99 = ret_p[ret_p <= var99].mean() if (ret_p <= var99).any() else var99
            hhi_val = np.sum(act_data['pesos']**2)
            pesos_sorted = sorted(act_data['pesos'], reverse=True)
            top2_conc = sum(pesos_sorted[:2]) * 100

            with col_r1:
                st.metric("VaR 99% (diario)", f"{var99*100:.2f}%", delta=f"${capital_usd*abs(var99):,.0f} USD en riesgo")
                st.metric("CVaR 99%", f"{cvar99*100:.2f}%")
            with col_r2:
                st.metric("HHI (concentración)", f"{hhi_val:.4f}", delta=f"N efectivo: {1/hhi_val:.1f} activos")
                st.metric("Top-2 concentración", f"{top2_conc:.1f}%", delta="⚠️ Alta" if top2_conc > 50 else "OK")
            with col_r3:
                spy_ret_op = returns_df['SPY'] if 'SPY' in returns_df.columns else returns_df.iloc[:, 0]
                excess_op = ret_p - spy_ret_op.reindex(ret_p.index, fill_value=0)
                te_op = excess_op.std() * np.sqrt(252)
                ir_op = (excess_op.mean() * 252) / te_op if te_op > 0 else 0
                st.metric("Information Ratio vs SPY", f"{ir_op:.2f}")
                st.metric("Tracking Error vs SPY", f"{te_op*100:.2f}%")

    # --- OPERATIVA TAB 3: Correlaciones ---
    with tab_o3:
        st.subheader("🔗 Matrices de Correlación — Detalle por Perfil")
        for pnombre in PERFILES:
            if pnombre not in resultados_perfiles:
                continue
            tickers_corr = [t for t in resultados_perfiles[pnombre]['tickers'] if t in returns_df.columns]
            if len(tickers_corr) < 2:
                continue
            with st.expander(f"Perfil {pnombre.upper()} — {len(tickers_corr)} activos", expanded=(pnombre == perfil_activo)):
                corr_matrix = returns_df[tickers_corr].corr()
                col_map, col_diag = st.columns([7, 5])
                with col_map:
                    fig_corr = px.imshow(corr_matrix, text_auto=".2f", aspect="auto",
                        color_continuous_scale="RdBu_r", zmin=-1.0, zmax=1.0, title=f"Correlación — {pnombre}")
                    fig_corr.update_layout(height=420, margin=dict(t=30, b=20))
                    st.plotly_chart(fig_corr, use_container_width=True)
                with col_diag:
                    upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
                    corr_vals = upper_tri.stack().values
                    st.markdown(f"**Correlación media:** {corr_vals.mean():.3f}")
                    st.markdown(f"**Correlación máxima:** {corr_vals.max():.3f}")
                    st.markdown(f"**Correlación mínima:** {corr_vals.min():.3f}")
                    st.markdown(f"**Pares con corr > 0.7:** {(corr_vals > 0.7).sum()}")
                    st.markdown(f"**Pares con corr < 0.3:** {(corr_vals < 0.3).sum()} (buena diversificación)")
                    high_pairs = [(corr_matrix.index[i], corr_matrix.columns[j], corr_matrix.iloc[i,j])
                                  for i in range(len(corr_matrix)) for j in range(i+1, len(corr_matrix))
                                  if abs(corr_matrix.iloc[i,j]) > 0.60]
                    if high_pairs:
                        st.warning("**Pares con correlación > 0.60:**")
                        for t1, t2, c in sorted(high_pairs, key=lambda x: -abs(x[2]))[:5]:
                            st.write(f"- {t1} ↔ {t2}: **{c:.3f}**")
                    else:
                        st.success("✅ Ningún par excede 0.60 — diversificación óptima")

    # --- OPERATIVA TAB 4: Frontera Eficiente & Backtest ---
    with tab_o4:
        st.subheader("📊 Frontera Eficiente & Backtest Histórico")
        col_front, col_back = st.columns(2)

        with col_front:
            st.markdown("#### Nube de Portafolios Simulados + Óptimos por Perfil")
            all_opt_tickers = list(set(t for d in resultados_perfiles.values() for t in d['tickers']))
            valid_opt = [t for t in all_opt_tickers if t in returns_df.columns]
            if len(valid_opt) >= 3:
                mean_ret_all = returns_df[valid_opt].mean() * 252
                cov_mat_all = returns_df[valid_opt].cov() * 252
                np.random.seed(123)
                sim_v, sim_r, sim_s = [], [], []
                for _ in range(3000):
                    w = np.random.dirichlet(np.ones(len(valid_opt)))
                    pr = np.sum(mean_ret_all.values * w)
                    pv = np.sqrt(np.dot(w.T, np.dot(cov_mat_all.values, w)))
                    sim_r.append(pr * 100); sim_v.append(pv * 100)
                    sim_s.append((pr - rf_rate_input) / pv if pv > 0 else 0)

                fig_ef = go.Figure()
                fig_ef.add_trace(go.Scatter(x=sim_v, y=sim_r, mode='markers',
                    marker=dict(size=3, color=sim_s, colorscale='Viridis', showscale=True, colorbar=dict(title="Sharpe")),
                    name="Simulados", opacity=0.4))
                markers = {'Conservador': 'square', 'Moderado': 'diamond', 'Agresivo': 'triangle-up'}
                for pn, pd_data in resultados_perfiles.items():
                    fig_ef.add_trace(go.Scatter(
                        x=[pd_data['metricas']['Vol']*100], y=[pd_data['metricas']['CAGR']*100],
                        mode='markers+text', text=[f" {pn}"], textposition='top right',
                        marker=dict(size=16, color=PERFILES[pn]['color'], symbol=markers[pn], line=dict(width=2, color='white')),
                        name=f"{pn} (S={pd_data['metricas']['Sharpe']:.2f})"))
                for bench in ['SPY', 'QQQ']:
                    brow = df_metricas[df_metricas['Ticker'] == bench]
                    if not brow.empty:
                        br = brow.iloc[0]
                        fig_ef.add_trace(go.Scatter(x=[br['Volatilidad (%)']], y=[br['CAGR (%)']],
                            mode='markers+text', text=[f" {bench}"], textposition='bottom right',
                            marker=dict(size=12, color='red', symbol='x', line=dict(width=2)),
                            name=f"Benchmark {bench}"))
                fig_ef.update_layout(xaxis_title="Volatilidad (%)", yaxis_title="CAGR (%)", height=480, margin=dict(t=20, b=20))
                st.plotly_chart(fig_ef, use_container_width=True)

        with col_back:
            st.markdown("#### Evolución del Capital (USD)")
            fig_bt = go.Figure()
            spy_ret_s = returns_df['SPY'] if 'SPY' in returns_df.columns else returns_df.iloc[:, 0]
            cum_spy = (1 + spy_ret_s).cumprod() * capital_usd
            fig_bt.add_trace(go.Scatter(x=cum_spy.index, y=cum_spy.values, mode='lines',
                name=f'SPY → ${cum_spy.iloc[-1]:,.0f}', line=dict(color='gray', dash='dash', width=2)))
            for pn, pd_data in resultados_perfiles.items():
                cum_p = (1 + pd_data['ret_port']).cumprod() * capital_usd
                fig_bt.add_trace(go.Scatter(x=cum_p.index, y=cum_p.values, mode='lines',
                    name=f'{pn} → ${cum_p.iloc[-1]:,.0f}', line=dict(color=PERFILES[pn]['color'], width=2.5)))
            fig_bt.add_hline(y=capital_usd, line_dash="dot", line_color="gray", annotation_text=f"Capital: ${capital_usd:,.0f}")
            fig_bt.update_layout(yaxis_title="Valor USD", height=480, margin=dict(t=20, b=20))
            st.plotly_chart(fig_bt, use_container_width=True)

        st.markdown("---")
        st.markdown("#### 📊 Proyección de Capital — 1, 3 y 5 Años")
        perfil_proy_op = st.selectbox(
            "Selecciona un perfil para ver la proyección:",
            list(resultados_perfiles.keys()),
            index=list(resultados_perfiles.keys()).index(perfil_activo) if perfil_activo in resultados_perfiles else 0,
            key="perfil_proy_operativa"
        )
        if perfil_proy_op in resultados_perfiles:
            col_proy_op1, col_proy_op2 = st.columns(2)
            pd_op = resultados_perfiles[perfil_proy_op]
            cagr_op = pd_op['metricas']['CAGR']
            vol_op = pd_op['metricas']['Vol']

            with col_proy_op1:
                bar_op_data = []
                for pn, pd_d in resultados_perfiles.items():
                    cg = pd_d['metricas']['CAGR']
                    for yr in [1, 3, 5]:
                        bar_op_data.append({"Perfil": pn, "Horizonte": f"{yr}Y", "Capital_USD": capital_usd*(1+cg)**yr})
                fig_bop = px.bar(pd.DataFrame(bar_op_data), x="Horizonte", y="Capital_USD", color="Perfil",
                    barmode="group", text_auto="$,.0f",
                    color_discrete_map={pn: PERFILES[pn]['color'] for pn in PERFILES})
                fig_bop.add_hline(y=capital_usd, line_dash="dot", line_color="gray")
                fig_bop.update_layout(yaxis_title="Capital USD", height=400, margin=dict(t=20, b=20), yaxis_tickformat="$,.0f")
                fig_bop.update_traces(textposition='outside', textfont_size=9)
                st.plotly_chart(fig_bop, use_container_width=True)

            with col_proy_op2:
                meses_op = np.arange(0, 61)
                cap_b = capital_usd * (1 + cagr_op) ** (meses_op / 12)
                cap_u = capital_usd * (1 + cagr_op + vol_op) ** (meses_op / 12)
                cap_l = capital_usd * (1 + max(cagr_op - vol_op, -0.99)) ** (meses_op / 12)
                fig_top = go.Figure()
                fig_top.add_trace(go.Scatter(x=meses_op/12, y=cap_u, mode='lines', line=dict(width=0), showlegend=False))
                fig_top.add_trace(go.Scatter(x=meses_op/12, y=cap_l, mode='lines', line=dict(width=0),
                    fill='tonexty', fillcolor=f'rgba({int(PERFILES[perfil_proy_op]["color"][1:3],16)},{int(PERFILES[perfil_proy_op]["color"][3:5],16)},{int(PERFILES[perfil_proy_op]["color"][5:7],16)},0.15)',
                    showlegend=False))
                fig_top.add_trace(go.Scatter(x=meses_op/12, y=cap_b, mode='lines',
                    name=f'{perfil_proy_op} (CAGR {cagr_op*100:.1f}%)',
                    line=dict(color=PERFILES[perfil_proy_op]['color'], width=3)))
                for yr in [1, 3, 5]:
                    cy = capital_usd * (1 + cagr_op)**yr
                    fig_top.add_trace(go.Scatter(x=[yr], y=[cy], mode='markers+text',
                        text=[f"Año {yr}: ${cy:,.0f}"], textposition='top center', textfont=dict(size=9),
                        marker=dict(size=12, color=PERFILES[perfil_proy_op]['color'], symbol='diamond',
                            line=dict(width=2, color='white')), showlegend=False))
                fig_top.add_hline(y=capital_usd, line_dash="dot", line_color="gray")
                fig_top.update_layout(xaxis_title="Años", yaxis_title="Capital USD", height=400,
                    margin=dict(t=20, b=20), yaxis_tickformat="$,.0f")
                st.plotly_chart(fig_top, use_container_width=True)

        st.markdown("---")
        st.markdown("#### Drawdown Histórico por Perfil")
        fig_dd = go.Figure()
        for pn, pd_data in resultados_perfiles.items():
            cum_p = (1 + pd_data['ret_port']).cumprod()
            dd = (cum_p - cum_p.cummax()) / cum_p.cummax() * 100
            fig_dd.add_trace(go.Scatter(x=dd.index, y=dd.values, mode='lines', fill='tozeroy',
                name=f'{pn} (Max: {dd.min():.1f}%)', line=dict(color=PERFILES[pn]['color'], width=1)))
        fig_dd.update_layout(yaxis_title="Drawdown (%)", height=300, margin=dict(t=20, b=20))
        st.plotly_chart(fig_dd, use_container_width=True)

    # --- OPERATIVA TAB 5: Universo & Métricas ---
    with tab_o5:
        st.subheader("🔍 Universo Completo & Métricas Cuantitativas")
        col_univ, col_met = st.columns([5, 7])
        with col_univ:
            st.markdown("#### Catálogo del Universo (43 activos)")
            st.dataframe(DF_UNIVERSO, use_container_width=True, hide_index=True)
        with col_met:
            st.markdown("#### Métricas Calculadas")
            display_met = df_metricas[['Ticker', 'CAGR (%)', 'Volatilidad (%)', 'Sharpe', 'Sortino',
                                        'Max Drawdown (%)', 'Beta', 'Alpha (%)', 'Calmar']].copy()
            st.dataframe(display_met.style.format({
                'CAGR (%)': '{:.2f}%', 'Volatilidad (%)': '{:.2f}%', 'Sharpe': '{:.2f}',
                'Sortino': '{:.2f}', 'Max Drawdown (%)': '{:.2f}%', 'Beta': '{:.2f}',
                'Alpha (%)': '{:.2f}%', 'Calmar': '{:.2f}'
            }), use_container_width=True, hide_index=True)

        st.markdown("---")
        st.markdown("#### Risk-Return Scatter (Activos Individuales)")
        all_selected = list(set(t for d in seleccion_por_perfil.values() for t in d['tickers']))
        fig_rr = go.Figure()
        df_sel_op = df_metricas[df_metricas['Ticker'].isin(all_selected)]
        df_nosel_op = df_metricas[~df_metricas['Ticker'].isin(all_selected)]
        if not df_nosel_op.empty:
            fig_rr.add_trace(go.Scatter(x=df_nosel_op['Volatilidad (%)'], y=df_nosel_op['CAGR (%)'],
                mode='markers+text', text=df_nosel_op['Ticker'], textposition='top center', textfont=dict(size=7),
                marker=dict(size=8, color='#bdc3c7', opacity=0.5), name='No seleccionado'))
        if not df_sel_op.empty:
            fig_rr.add_trace(go.Scatter(x=df_sel_op['Volatilidad (%)'], y=df_sel_op['CAGR (%)'],
                mode='markers+text', text=df_sel_op['Ticker'], textposition='top center', textfont=dict(size=9, color='white'),
                marker=dict(size=12, color='#27ae60', line=dict(width=1, color='white')), name='Seleccionado'))
        fig_rr.add_hline(y=rf_rate_input*100, line_dash="dot", line_color="gray", annotation_text="Rf")
        fig_rr.update_layout(xaxis_title="Volatilidad (%)", yaxis_title="CAGR (%)", height=500, margin=dict(t=20, b=20))
        st.plotly_chart(fig_rr, use_container_width=True)

else:
    # ══════════════════════════════════════════════════════════════════════════
    # VISTA GERENCIAL — Comité de Inversiones: KPIs consolidados, proyecciones,
    # dictamen ejecutivo, Monte Carlo, Walk-Forward, comparativa de perfiles
    # ══════════════════════════════════════════════════════════════════════════
    tab_g1, tab_g2, tab_g3, tab_g4 = st.tabs([
        "🏛️ Dictamen & Recomendación",
        "📈 Comparativa de Perfiles",
        "🎲 Proyección Monte Carlo",
        "✅ Validación Walk-Forward",
    ])

    # --- GERENCIAL TAB 1: Dictamen ejecutivo ---
    with tab_g1:
        st.subheader("🏛️ Dictamen para el Comité de Inversiones")
        st.markdown("""
> **Resumen Ejecutivo del Analista Senior:**
> Se evaluaron **43 activos** del mercado estadounidense mediante un framework de **selección algorítmica
> multifactorial** (50% Sharpe + 30% Alpha + 20% EBITDA Margin) con **filtro correlacional greedy**
> para construir **3 portafolios diversificados** por perfil de riesgo.
> La optimización emplea **Markowitz Mean-Variance (SLSQP)** con restricciones de volatilidad máxima
> y simulación de **rebalanceo trimestral con costos de transacción** (15 bps por turno).
        """)

        st.markdown("---")
        st.markdown("#### 📋 Resumen de Capital Proyectado a 1, 3 y 5 Años")
        horizon_data = []
        for pn, pd_data in resultados_perfiles.items():
            cagr_g = pd_data['metricas']['CAGR']
            for anos in [1, 3, 5]:
                cap_usd_g = capital_usd * (1 + cagr_g)**anos
                horizon_data.append({
                    "Perfil": pn, "Horizonte": f"{anos} año{'s' if anos > 1 else ''}",
                    "Capital USD": f"${cap_usd_g:,.0f}",
                    "Capital PEN": f"S/ {cap_usd_g*fx_rate:,.0f}",
                    "Ganancia PEN": f"S/ {(cap_usd_g*fx_rate - capital_pen):,.0f}",
                    "Retorno Total": f"{(cap_usd_g/capital_usd - 1)*100:+.1f}%"
                })
        st.dataframe(pd.DataFrame(horizon_data), use_container_width=True, hide_index=True)

        st.markdown("---")
        st.markdown("#### 📊 Proyección Visual de Capital por Perfil")
        perfil_proy_g = st.selectbox(
            "Selecciona un perfil para ver la proyección detallada:",
            list(resultados_perfiles.keys()),
            index=list(resultados_perfiles.keys()).index(perfil_activo) if perfil_activo in resultados_perfiles else 0,
            key="perfil_proy_gerencial"
        )

        col_proy_bar, col_proy_line = st.columns(2)
        with col_proy_bar:
            st.markdown("#### Comparativa de Perfiles — Capital a 1, 3 y 5 Años")
            bar_proy_data = []
            for pn, pd_data in resultados_perfiles.items():
                cagr_bp = pd_data['metricas']['CAGR']
                for anos in [1, 3, 5]:
                    bar_proy_data.append({
                        "Perfil": pn, "Horizonte": f"{anos}Y",
                        "Capital_USD": capital_usd * (1 + cagr_bp)**anos,
                        "Años": anos
                    })
            df_bar_proy = pd.DataFrame(bar_proy_data)
            fig_bar_proy = px.bar(
                df_bar_proy, x="Horizonte", y="Capital_USD", color="Perfil",
                barmode="group", text_auto="$,.0f",
                color_discrete_map={pn: PERFILES[pn]['color'] for pn in PERFILES}
            )
            fig_bar_proy.add_hline(y=capital_usd, line_dash="dot", line_color="gray",
                annotation_text=f"Capital Inicial: ${capital_usd:,.0f}")
            fig_bar_proy.update_layout(
                yaxis_title="Capital USD", height=450, margin=dict(t=20, b=20),
                yaxis_tickformat="$,.0f"
            )
            fig_bar_proy.update_traces(textposition='outside', textfont_size=10)
            st.plotly_chart(fig_bar_proy, use_container_width=True)

        with col_proy_line:
            st.markdown(f"#### Trayectoria de Crecimiento — {perfil_proy_g}")
            if perfil_proy_g in resultados_perfiles:
                pd_sel = resultados_perfiles[perfil_proy_g]
                cagr_sel = pd_sel['metricas']['CAGR']
                vol_sel = pd_sel['metricas']['Vol']
                meses = np.arange(0, 61)
                cap_base = capital_usd * (1 + cagr_sel) ** (meses / 12)
                cap_upper = capital_usd * (1 + cagr_sel + vol_sel) ** (meses / 12)
                cap_lower = capital_usd * (1 + max(cagr_sel - vol_sel, -0.99)) ** (meses / 12)
                fig_traj = go.Figure()
                fig_traj.add_trace(go.Scatter(
                    x=meses/12, y=cap_upper, mode='lines', name='Optimista (+1σ)',
                    line=dict(width=0), showlegend=False))
                fig_traj.add_trace(go.Scatter(
                    x=meses/12, y=cap_lower, mode='lines', name='Pesimista (-1σ)',
                    line=dict(width=0), fill='tonexty',
                    fillcolor=f'rgba({int(PERFILES[perfil_proy_g]["color"][1:3],16)},{int(PERFILES[perfil_proy_g]["color"][3:5],16)},{int(PERFILES[perfil_proy_g]["color"][5:7],16)},0.15)',
                    showlegend=False))
                fig_traj.add_trace(go.Scatter(
                    x=meses/12, y=cap_base, mode='lines', name=f'{perfil_proy_g} (CAGR {cagr_sel*100:.1f}%)',
                    line=dict(color=PERFILES[perfil_proy_g]['color'], width=3)))
                for yr in [1, 3, 5]:
                    cap_yr = capital_usd * (1 + cagr_sel)**yr
                    ganancia_yr = cap_yr - capital_usd
                    fig_traj.add_trace(go.Scatter(
                        x=[yr], y=[cap_yr], mode='markers+text',
                        text=[f"Año {yr}\n${cap_yr:,.0f}\n(+${ganancia_yr:,.0f})"],
                        textposition='top center', textfont=dict(size=10),
                        marker=dict(size=14, color=PERFILES[perfil_proy_g]['color'],
                            line=dict(width=2, color='white'), symbol='diamond'),
                        showlegend=False))
                fig_traj.add_hline(y=capital_usd, line_dash="dot", line_color="gray")
                fig_traj.update_layout(
                    xaxis_title="Años", yaxis_title="Capital USD",
                    height=450, margin=dict(t=20, b=20),
                    yaxis_tickformat="$,.0f"
                )
                st.plotly_chart(fig_traj, use_container_width=True)

                col_p1, col_p3, col_p5 = st.columns(3)
                for col_px, yr in zip([col_p1, col_p3, col_p5], [1, 3, 5]):
                    cap_yr = capital_usd * (1 + cagr_sel)**yr
                    gan_usd = cap_yr - capital_usd
                    gan_pen = gan_usd * fx_rate
                    ret_total = (cap_yr / capital_usd - 1) * 100
                    with col_px:
                        st.markdown(f'<div class="kpi-card">'
                            f'<div class="kpi-title">Año {yr}</div>'
                            f'<div class="kpi-value" style="color:{PERFILES[perfil_proy_g]["color"]};">'
                            f'${cap_yr:,.0f}</div>'
                            f'<div class="kpi-sub">+${gan_usd:,.0f} USD | S/ {gan_pen:,.0f} PEN | {ret_total:+.1f}%</div>'
                            f'</div>', unsafe_allow_html=True)

        st.markdown("---")
        col_g1, col_g2 = st.columns(2)
        with col_g1:
            st.markdown("#### 🛡️ Evaluación de Riesgos Clave")
            st.markdown("""
- **Riesgo Cambiario (PEN/USD):** Invertir en activos USD genera cobertura natural ante depreciación del Sol.
- **Riesgo de Cola (CVaR):** El Expected Shortfall controla pérdidas en escenarios extremos (peor 2.5% de días).
- **Concentración (HHI):** El filtro correlacional + bounds de Markowitz limitan la concentración excesiva.
- **Survivorship Bias:** El universo manual no incluye empresas deslistadas; resultados son optimistas.
- **Lookback Bias:** Se mitiga con bounds estrictos y validación walk-forward 60/40.
            """)
        with col_g2:
            st.markdown("#### ⚙️ Reglas Tácticas de Gobernanza")
            st.markdown(f"""
- **Rebalanceo:** Trimestral (~{REBALANCEO_DIAS} días hábiles) con costo de {COSTO_REBALANCEO_BPS} bps incluido.
- **Banda de Tolerancia:** Ejecutar ajuste cuando un peso se desvíe ±5% del objetivo.
- **Score de Selección:** 50% Sharpe + 30% Alpha + 20% EBITDA Margin.
- **Mejoras Recomendadas:** Black-Litterman, shrinkage de Ledoit-Wolf, o HRP (Lopez de Prado).
            """)

        st.markdown("---")
        st.markdown("""
#### ⚠️ Disclaimer
- Los resultados se basan en datos históricos y **NO garantizan rendimientos futuros**.
- La optimización de Markowitz es sensible a errores de estimación en medias y covarianzas.
- Este análisis es informativo y no constituye asesoría de inversión personalizada.
        """)

    # --- GERENCIAL TAB 2: Comparativa consolidada de perfiles ---
    with tab_g2:
        st.subheader("📈 Comparativa Consolidada — Los 3 Perfiles vs Benchmarks")

        comp_data = []
        for pn, pd_data in resultados_perfiles.items():
            rp_g = pd_data['ret_port']
            var95_g = np.percentile(rp_g, 5)
            cvar95_g = rp_g[rp_g <= var95_g].mean() if (rp_g <= var95_g).any() else var95_g
            hhi_g = np.sum(pd_data['pesos']**2)
            comp_data.append({
                "Perfil": pn,
                "CAGR": f"{pd_data['metricas']['CAGR']*100:.2f}%",
                "Volatilidad": f"{pd_data['metricas']['Vol']*100:.2f}%",
                "Sharpe": f"{pd_data['metricas']['Sharpe']:.2f}",
                "Max DD": f"{pd_data['mdd_real']*100:.1f}%",
                "VaR 95%": f"{var95_g*100:.2f}%",
                "CVaR 95%": f"{cvar95_g*100:.2f}%",
                "HHI": f"{hhi_g:.3f}",
                "# Activos": len(pd_data['tickers']),
                "Capital 5Y": f"${capital_usd*(1+pd_data['metricas']['CAGR'])**5:,.0f}",
            })
        spy_m = df_metricas[df_metricas['Ticker'] == 'SPY']
        if not spy_m.empty:
            sm = spy_m.iloc[0]
            comp_data.append({
                "Perfil": "🔴 SPY (Bench)", "CAGR": f"{sm['CAGR (%)']:.2f}%",
                "Volatilidad": f"{sm['Volatilidad (%)']:.2f}%", "Sharpe": f"{sm['Sharpe']:.2f}",
                "Max DD": f"{sm['Max Drawdown (%)']:.1f}%", "VaR 95%": "—", "CVaR 95%": "—",
                "HHI": "1.000", "# Activos": 1,
                "Capital 5Y": f"${capital_usd*(1+sm['CAGR (%)']/100)**5:,.0f}",
            })
        st.dataframe(pd.DataFrame(comp_data), use_container_width=True, hide_index=True)

        st.markdown("---")
        col_evol, col_bars = st.columns(2)

        with col_evol:
            st.markdown("#### Evolución del Capital — Todos los Perfiles vs SPY")
            fig_bt_g = go.Figure()
            spy_ret_g = returns_df['SPY'] if 'SPY' in returns_df.columns else returns_df.iloc[:, 0]
            cum_spy_g = (1 + spy_ret_g).cumprod() * capital_usd
            fig_bt_g.add_trace(go.Scatter(x=cum_spy_g.index, y=cum_spy_g.values, mode='lines',
                name=f'SPY → ${cum_spy_g.iloc[-1]:,.0f}', line=dict(color='gray', dash='dash', width=2)))
            for pn, pd_data in resultados_perfiles.items():
                cum_p_g = (1 + pd_data['ret_port']).cumprod() * capital_usd
                fig_bt_g.add_trace(go.Scatter(x=cum_p_g.index, y=cum_p_g.values, mode='lines',
                    name=f'{pn} → ${cum_p_g.iloc[-1]:,.0f}', line=dict(color=PERFILES[pn]['color'], width=2.5)))
            fig_bt_g.add_hline(y=capital_usd, line_dash="dot", line_color="gray")
            fig_bt_g.update_layout(yaxis_title="Valor USD", height=420, margin=dict(t=20, b=20))
            st.plotly_chart(fig_bt_g, use_container_width=True)

        with col_bars:
            st.markdown("#### Métricas Clave por Perfil")
            perfiles_names = list(resultados_perfiles.keys())
            fig_bars = go.Figure()
            fig_bars.add_trace(go.Bar(name='CAGR (%)',
                x=perfiles_names, y=[resultados_perfiles[p]['metricas']['CAGR']*100 for p in perfiles_names],
                marker_color='#34d399'))
            fig_bars.add_trace(go.Bar(name='Vol (%)',
                x=perfiles_names, y=[resultados_perfiles[p]['metricas']['Vol']*100 for p in perfiles_names],
                marker_color='#f87171'))
            fig_bars.add_trace(go.Bar(name='Sharpe',
                x=perfiles_names, y=[resultados_perfiles[p]['metricas']['Sharpe'] for p in perfiles_names],
                marker_color='#06b6d4'))
            fig_bars.update_layout(barmode='group', height=420, margin=dict(t=20, b=20))
            st.plotly_chart(fig_bars, use_container_width=True)

        st.markdown("---")
        st.markdown("#### Composición de Portafolios — Vista Comparativa")
        pie_cols = st.columns(len(resultados_perfiles))
        for idx, (pn, pd_data) in enumerate(resultados_perfiles.items()):
            with pie_cols[idx]:
                st.markdown(f"**{pn}** — Sharpe: {pd_data['metricas']['Sharpe']:.2f}")
                pw = dict(zip(pd_data['tickers'], pd_data['pesos']))
                aw = {k: v for k, v in pw.items() if v > 0.005}
                fig_pie_g = px.pie(names=list(aw.keys()), values=list(aw.values()), hole=0.4,
                    color_discrete_sequence=px.colors.sequential.Tealgrn)
                fig_pie_g.update_layout(margin=dict(t=10, b=10, l=10, r=10), height=280, showlegend=True,
                    legend=dict(font=dict(size=9)))
                st.plotly_chart(fig_pie_g, use_container_width=True)

    # --- GERENCIAL TAB 3: Monte Carlo ---
    with tab_g3:
        st.subheader(f"🎲 Proyección Monte Carlo — {N_SIMS_MC:,} trayectorias, {MC_HORIZON_YEARS} años")
        st.caption("Cono de incertidumbre estocástico (GBM) basado en el drift y volatilidad histórica de cada perfil.")
        np.random.seed(777)
        horizon_days = 252 * MC_HORIZON_YEARS

        mc_summary = []
        for pn, pd_data in resultados_perfiles.items():
            rp_mc = pd_data['ret_port']
            mu_mc, sigma_mc = rp_mc.mean(), rp_mc.std()
            sims_mc = np.zeros((N_SIMS_MC, horizon_days))
            for i in range(N_SIMS_MC):
                dr = np.random.normal(mu_mc, sigma_mc, horizon_days)
                sims_mc[i] = capital_usd * np.cumprod(1 + dr)
            final_mc = sims_mc[:, -1]
            p5, p25, p50, p75, p95 = [np.percentile(final_mc, p) for p in [5, 25, 50, 75, 95]]
            prob_loss = (final_mc < capital_usd).mean() * 100
            mc_summary.append({"Perfil": pn, "P5": p5, "P25": p25, "P50": p50, "P75": p75, "P95": p95, "prob_loss": prob_loss, "sims": sims_mc})

        mc_summary_df = pd.DataFrame([{
            "Perfil": m["Perfil"],
            "P5 (peor)": f"${m['P5']:,.0f}", "P25": f"${m['P25']:,.0f}",
            "P50 (mediana)": f"${m['P50']:,.0f}", "P75": f"${m['P75']:,.0f}",
            "P95 (mejor)": f"${m['P95']:,.0f}", "Prob. Pérdida": f"{m['prob_loss']:.1f}%",
            "Retorno Mediano": f"{(m['P50']/capital_usd-1)*100:+.1f}%",
        } for m in mc_summary])
        st.dataframe(mc_summary_df, use_container_width=True, hide_index=True)

        st.markdown("---")
        st.markdown("#### Cono de Incertidumbre Forward — Bandas P10 / P50 / P90")
        fig_cone = go.Figure()
        time_idx = np.linspace(0, MC_HORIZON_YEARS, horizon_days)
        for m in mc_summary:
            p10_path = np.percentile(m['sims'], 10, axis=0)
            p50_path = np.median(m['sims'], axis=0)
            p90_path = np.percentile(m['sims'], 90, axis=0)
            color = PERFILES[m['Perfil']]['color']
            fig_cone.add_trace(go.Scatter(x=time_idx, y=p90_path, mode='lines', name=f"{m['Perfil']} P90",
                line=dict(color=color, width=0.5, dash='dot'), showlegend=False))
            fig_cone.add_trace(go.Scatter(x=time_idx, y=p10_path, mode='lines', name=f"{m['Perfil']} P10",
                line=dict(color=color, width=0.5, dash='dot'), fill='tonexty', fillcolor=f'rgba({int(color[1:3],16)},{int(color[3:5],16)},{int(color[5:7],16)},0.10)', showlegend=False))
            fig_cone.add_trace(go.Scatter(x=time_idx, y=p50_path, mode='lines',
                name=f"{m['Perfil']} Mediana: ${p50_path[-1]:,.0f}", line=dict(color=color, width=3)))
        fig_cone.add_hline(y=capital_usd, line_dash="dot", line_color="gray", annotation_text=f"Capital: ${capital_usd:,.0f}")
        fig_cone.add_hline(y=capital_usd*2, line_dash="dash", line_color="gray", annotation_text="2x Capital", opacity=0.3)
        fig_cone.update_layout(xaxis_title="Años", yaxis_title="Capital USD", height=500, margin=dict(t=20, b=20))
        st.plotly_chart(fig_cone, use_container_width=True)

    # --- GERENCIAL TAB 4: Walk-Forward ---
    with tab_g4:
        st.subheader("✅ Validación Walk-Forward — Estabilidad Out-of-Sample")
        st.caption("Split 60/40: los primeros 60% de datos como 'pasado', los últimos 40% como 'presente'. "
                   "Se evalúa si la estrategia mantiene su Sharpe fuera de la muestra de calibración.")

        split_idx = int(len(returns_df) * 0.60)
        returns_train = returns_df.iloc[:split_idx]
        returns_test = returns_df.iloc[split_idx:]

        col_wf_info, col_wf_table = st.columns([4, 8])
        with col_wf_info:
            st.info(f"**In-Sample (60%):**\n{returns_train.index[0].strftime('%Y-%m-%d')} → {returns_train.index[-1].strftime('%Y-%m-%d')}\n({len(returns_train)} días)")
            st.info(f"**Out-of-Sample (40%):**\n{returns_test.index[0].strftime('%Y-%m-%d')} → {returns_test.index[-1].strftime('%Y-%m-%d')}\n({len(returns_test)} días)")
            st.markdown("""
**Interpretación:**
- **Cambio Sharpe ≥ 0%:** La estrategia es estable o mejora.
- **Cambio Sharpe > -30%:** Degradación leve, aceptable.
- **Cambio Sharpe < -30%:** Posible sobreajuste (overfitting).
            """)

        with col_wf_table:
            wf_data = []
            for pn, pd_data in resultados_perfiles.items():
                ret_is = simular_rebalanceo(returns_train, pd_data['tickers'], pd_data['pesos'])
                sharpe_is = (ret_is.mean()*252 - rf_rate_input) / (ret_is.std()*np.sqrt(252)) if len(ret_is)>0 and ret_is.std()>0 else 0
                cagr_is = ((1+ret_is).cumprod().iloc[-1])**(252/max(len(ret_is),1)) - 1 if len(ret_is)>0 else 0
                ret_oos = simular_rebalanceo(returns_test, pd_data['tickers'], pd_data['pesos'])
                sharpe_oos = (ret_oos.mean()*252 - rf_rate_input) / (ret_oos.std()*np.sqrt(252)) if len(ret_oos)>0 and ret_oos.std()>0 else 0
                cagr_oos = ((1+ret_oos).cumprod().iloc[-1])**(252/max(len(ret_oos),1)) - 1 if len(ret_oos)>0 else 0
                spy_oos = returns_test['SPY'] if 'SPY' in returns_test.columns else returns_test.iloc[:, 0]
                sharpe_spy_oos = (spy_oos.mean()*252 - rf_rate_input) / (spy_oos.std()*np.sqrt(252)) if spy_oos.std()>0 else 0
                cambio = ((sharpe_oos - sharpe_is) / abs(sharpe_is)) * 100 if sharpe_is != 0 else 0
                wf_data.append({
                    "Perfil": pn,
                    "Sharpe In-Sample": f"{sharpe_is:.2f}", "CAGR IS": f"{cagr_is*100:.2f}%",
                    "Sharpe Out-Sample": f"{sharpe_oos:.2f}", "CAGR OOS": f"{cagr_oos*100:.2f}%",
                    "SPY (OOS)": f"{sharpe_spy_oos:.2f}",
                    "Δ Sharpe": f"{cambio:+.1f}%",
                    "Supera SPY": "✅" if sharpe_oos > sharpe_spy_oos else "❌"
                })
            st.dataframe(pd.DataFrame(wf_data), use_container_width=True, hide_index=True)

        st.markdown("---")
        st.markdown("#### Comparativa Visual: In-Sample vs Out-of-Sample")
        fig_wf_g = go.Figure()
        for pn, pd_data in resultados_perfiles.items():
            ret_is = simular_rebalanceo(returns_train, pd_data['tickers'], pd_data['pesos'])
            ret_oos = simular_rebalanceo(returns_test, pd_data['tickers'], pd_data['pesos'])
            s_is = (ret_is.mean()*252 - rf_rate_input) / (ret_is.std()*np.sqrt(252)) if len(ret_is)>0 and ret_is.std()>0 else 0
            s_oos = (ret_oos.mean()*252 - rf_rate_input) / (ret_oos.std()*np.sqrt(252)) if len(ret_oos)>0 and ret_oos.std()>0 else 0
            fig_wf_g.add_trace(go.Bar(name=f'{pn} In-Sample', x=[pn], y=[s_is],
                marker_color=PERFILES[pn]['color'], opacity=0.5,
                text=[f"{s_is:.2f}"], textposition='outside'))
            fig_wf_g.add_trace(go.Bar(name=f'{pn} Out-Sample', x=[pn], y=[s_oos],
                marker_color=PERFILES[pn]['color'],
                text=[f"{s_oos:.2f}"], textposition='outside'))
        spy_oos_g = returns_test['SPY'] if 'SPY' in returns_test.columns else returns_test.iloc[:, 0]
        s_spy_g = (spy_oos_g.mean()*252 - rf_rate_input) / (spy_oos_g.std()*np.sqrt(252)) if spy_oos_g.std()>0 else 0
        fig_wf_g.add_hline(y=s_spy_g, line_dash="dash", line_color="red",
            annotation_text=f"SPY OOS: {s_spy_g:.2f}", annotation_position="top right")
        fig_wf_g.update_layout(barmode='group', yaxis_title="Sharpe Ratio", height=400, margin=dict(t=20, b=20))
        st.plotly_chart(fig_wf_g, use_container_width=True)
