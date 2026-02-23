from flask import Flask, render_template, request, jsonify
import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch
import seaborn as sns
from datetime import datetime, timedelta
import io
import base64
import warnings
import os
import json
from scipy import stats as scipy_stats

warnings.filterwarnings('ignore')

app = Flask(__name__)
app.config['SECRET_KEY'] = 'monte-carlo-saham-2024'
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0

TRADING_DAYS_PER_YEAR = 252

# ============================================
# TICKER NORMALIZATION
# ============================================

IDX_POPULAR = {
    'BBCA','BBRI','BMRI','TLKM','ASII','UNVR','GOTO','BYAN','BUKA','INDF',
    'GGRM','HMSP','ANTM','KLBF','CPIN','MDKA','ICBP','PGAS','JSMR','ADRO',
    'PTBA','INCO','TPIA','SMGR','TOWR','SIDO','MYOR','HEAL','EXCL','ISAT',
    'BBNI','BRIS','ARTO','EMTK','ESSA','DOID','MBMA','AMMN','NICE','HRUM'
}
US_POPULAR = {
    'AAPL','MSFT','GOOGL','AMZN','META','TSLA','NVDA','NFLX','AMD','INTC',
    'JPM','BAC','V','MA','DIS','UBER','ABNB','PLTR','COIN','RBLX','SHOP'
}

def normalize_ticker(raw):
    t = raw.strip().upper()
    if '.' in t:
        return t, 'explicit'
    if t in US_POPULAR:
        return t, 'US'
    if t in IDX_POPULAR:
        return t + '.JK', 'IDX'
    if len(t) <= 4 and t.isalpha():
        return t + '.JK', 'IDX_guess'
    return t, 'US_guess'

def is_indonesian(ticker):
    return ticker.endswith('.JK')

def price_fmt(price, ticker):
    if is_indonesian(ticker):
        return f"Rp{price:,.0f}"
    return f"${price:,.2f}"

# ============================================
# DATA FETCHING
# ============================================

def get_stock_data(ticker, years_history):
    try:
        stock = yf.Ticker(ticker)
        end_date = datetime.now()
        start_date = end_date - timedelta(days=years_history * 365 + 60)
        hist = stock.history(start=start_date, end=end_date)
        if len(hist) < 30:
            return None, "Data saham tidak ditemukan atau terlalu sedikit."
        prices = hist['Close']
        info = {}
        try:
            info = stock.info
        except Exception:
            pass
        return {
            'prices': prices,
            'company_name': info.get('longName', info.get('shortName', ticker)),
            'sector': info.get('sector', '-'),
            'current_price': float(prices.iloc[-1]),
            'start_date': prices.index[0].strftime('%d %b %Y'),
            'end_date': prices.index[-1].strftime('%d %b %Y'),
            'trading_days': len(prices),
        }, None
    except Exception as e:
        return None, str(e)

# ============================================
# MONTE CARLO (Standard GBM)
# ============================================

def calc_log_returns(prices):
    return np.log(prices / prices.shift(1)).dropna()

def run_monte_carlo(prices, days, simulations):
    returns = calc_log_returns(prices)
    mu_daily = float(returns.mean())
    sigma_daily = float(returns.std())
    mu_annual = mu_daily * TRADING_DAYS_PER_YEAR
    sigma_annual = sigma_daily * np.sqrt(TRADING_DAYS_PER_YEAR)
    mu_adj_annual = mu_annual - 0.5 * sigma_annual ** 2
    mu_adj_daily = mu_adj_annual / TRADING_DAYS_PER_YEAR

    last_price = float(prices.iloc[-1])
    np.random.seed(None)
    epsilon = np.random.standard_normal((simulations, days))
    daily_log_ret = mu_adj_daily + sigma_daily * epsilon
    log_paths = np.cumsum(daily_log_ret, axis=1)
    price_paths = last_price * np.exp(log_paths)
    final_prices = price_paths[:, -1]

    # Skewness & Kurtosis for context
    skew = float(scipy_stats.skew(returns))
    kurt = float(scipy_stats.kurtosis(returns))

    stats = {
        'last_price': last_price,
        'mean': float(np.mean(final_prices)),
        'median': float(np.median(final_prices)),
        'std': float(np.std(final_prices)),
        'min': float(np.min(final_prices)),
        'max': float(np.max(final_prices)),
        'q5': float(np.percentile(final_prices, 5)),
        'q25': float(np.percentile(final_prices, 25)),
        'q75': float(np.percentile(final_prices, 75)),
        'q95': float(np.percentile(final_prices, 95)),
        'prob_up': float(np.mean(final_prices > last_price) * 100),
        'prob_down': float(np.mean(final_prices <= last_price) * 100),
        'var_95_abs': float(last_price - np.percentile(final_prices, 5)),
        'var_95_pct': float((last_price - np.percentile(final_prices, 5)) / last_price * 100),
        'var_99_abs': float(last_price - np.percentile(final_prices, 1)),
        'var_99_pct': float((last_price - np.percentile(final_prices, 1)) / last_price * 100),
        'mu_daily': mu_daily,
        'sigma_daily': sigma_daily,
        'mu_annual': mu_annual,
        'sigma_annual': sigma_annual,
        'mu_adj_annual': mu_adj_annual,
        'n_returns': len(returns),
        'sample_returns': returns.values[:5].tolist(),
        'skewness': skew,
        'kurtosis': kurt,
        'returns_array': returns.values.tolist(),
    }

    sample_idx = np.random.choice(simulations, min(60, simulations), replace=False)
    paths_display = {
        'p5': np.percentile(price_paths, 5, axis=0).tolist(),
        'p25': np.percentile(price_paths, 25, axis=0).tolist(),
        'p50': np.percentile(price_paths, 50, axis=0).tolist(),
        'p75': np.percentile(price_paths, 75, axis=0).tolist(),
        'p95': np.percentile(price_paths, 95, axis=0).tolist(),
        'mean': np.mean(price_paths, axis=0).tolist(),
        'sample_paths': price_paths[sample_idx].tolist(),
        'final_prices': final_prices.tolist(),
    }

    return price_paths, stats, paths_display

# ============================================
# BLACK SWAN SIMULATION (Jump Diffusion)
# Merton (1976) Jump Diffusion Model
# ============================================

def run_black_swan(prices, days, simulations, jump_intensity=2.0, jump_mean=-0.15, jump_std=0.20):
    """
    Merton Jump Diffusion Model:
    dS = S*(mu*dt + sigma*dW + J*dN)
    where dN is Poisson process, J is jump size (log-normal)
    
    Parameters:
    - jump_intensity: lambda = avg jumps per year (default 2 = twice/year)
    - jump_mean: average log jump size (negative = mostly crashes)
    - jump_std: std of jump size distribution
    """
    returns = calc_log_returns(prices)
    mu_daily = float(returns.mean())
    sigma_daily = float(returns.std())

    # Annualize
    mu_annual = mu_daily * TRADING_DAYS_PER_YEAR
    sigma_annual = sigma_daily * np.sqrt(TRADING_DAYS_PER_YEAR)

    # Jump parameters (daily)
    lam_daily = jump_intensity / TRADING_DAYS_PER_YEAR  # jumps per day
    k = np.exp(jump_mean + 0.5 * jump_std**2) - 1  # expected jump size

    # Compensated drift (Itô + jump compensation)
    mu_adj_daily = (mu_annual - 0.5 * sigma_annual**2 - jump_intensity * k) / TRADING_DAYS_PER_YEAR

    last_price = float(prices.iloc[-1])
    np.random.seed(None)

    # Standard GBM component
    epsilon = np.random.standard_normal((simulations, days))

    # Jump component - Poisson arrivals
    jump_arrivals = np.random.poisson(lam_daily, (simulations, days))

    # Jump sizes (log-normal)
    jump_sizes = np.zeros((simulations, days))
    for i in range(simulations):
        for j in range(days):
            n_jumps = jump_arrivals[i, j]
            if n_jumps > 0:
                # Sum of n_jumps log-normal jumps
                jump_sizes[i, j] = np.sum(np.random.normal(jump_mean, jump_std, n_jumps))

    # Daily log returns with jumps
    daily_log_ret = mu_adj_daily + sigma_daily * epsilon + jump_sizes
    log_paths = np.cumsum(daily_log_ret, axis=1)
    price_paths = last_price * np.exp(log_paths)
    final_prices = price_paths[:, -1]

    stats = {
        'last_price': last_price,
        'mean': float(np.mean(final_prices)),
        'median': float(np.median(final_prices)),
        'std': float(np.std(final_prices)),
        'min': float(np.min(final_prices)),
        'max': float(np.max(final_prices)),
        'q5': float(np.percentile(final_prices, 5)),
        'q25': float(np.percentile(final_prices, 25)),
        'q75': float(np.percentile(final_prices, 75)),
        'q95': float(np.percentile(final_prices, 95)),
        'prob_up': float(np.mean(final_prices > last_price) * 100),
        'prob_down': float(np.mean(final_prices <= last_price) * 100),
        'var_95_abs': float(last_price - np.percentile(final_prices, 5)),
        'var_95_pct': float((last_price - np.percentile(final_prices, 5)) / last_price * 100),
        'var_99_abs': float(last_price - np.percentile(final_prices, 1)),
        'var_99_pct': float((last_price - np.percentile(final_prices, 1)) / last_price * 100),
        'mu_daily': mu_daily,
        'sigma_daily': sigma_daily,
        'mu_annual': mu_annual,
        'sigma_annual': sigma_annual,
        'mu_adj_annual': mu_adj_daily * TRADING_DAYS_PER_YEAR,
        'n_returns': len(returns),
        'jump_intensity': jump_intensity,
        'jump_mean': jump_mean,
        'jump_std': jump_std,
        'k': k,
        # Count how many simulations experienced at least one major jump
        'pct_with_jumps': float(np.mean(np.sum(jump_arrivals, axis=1) > 0) * 100),
        'avg_jumps_per_sim': float(np.mean(np.sum(jump_arrivals, axis=1))),
    }

    sample_idx = np.random.choice(simulations, min(60, simulations), replace=False)
    paths_display = {
        'p5': np.percentile(price_paths, 5, axis=0).tolist(),
        'p25': np.percentile(price_paths, 25, axis=0).tolist(),
        'p50': np.percentile(price_paths, 50, axis=0).tolist(),
        'p75': np.percentile(price_paths, 75, axis=0).tolist(),
        'p95': np.percentile(price_paths, 95, axis=0).tolist(),
        'mean': np.mean(price_paths, axis=0).tolist(),
        'sample_paths': price_paths[sample_idx].tolist(),
        'final_prices': final_prices.tolist(),
    }

    return price_paths, stats, paths_display

# ============================================
# VISUALIZATION — Clean, Light, Professional
# ============================================

PALETTE = {
    'bg': '#F8F9FC',
    'surface': '#FFFFFF',
    'surface2': '#F1F4F9',
    'border': '#E2E8F0',
    'blue': '#2563EB',
    'blue_light': '#DBEAFE',
    'green': '#16A34A',
    'green_light': '#DCFCE7',
    'red': '#DC2626',
    'red_light': '#FEE2E2',
    'amber': '#D97706',
    'amber_light': '#FEF3C7',
    'slate': '#64748B',
    'dark': '#0F172A',
    'mid': '#334155',
}

def make_chart(price_paths, stats, paths_display, ticker, days, simulations, mode='standard'):
    P = PALETTE
    is_bs = (mode == 'blackswan')

    accent = P['red'] if is_bs else P['blue']
    accent_light = P['red_light'] if is_bs else P['blue_light']

    last_price = stats['last_price']
    final_prices = np.array(paths_display['final_prices'])
    dr = np.arange(1, days + 1)
    p5 = np.array(paths_display['p5'])
    p25 = np.array(paths_display['p25'])
    p75 = np.array(paths_display['p75'])
    p95 = np.array(paths_display['p95'])
    mean_p = np.array(paths_display['mean'])
    pfmt = 'Rp{:,.0f}' if is_indonesian(ticker) else '${:,.2f}'

    fig = plt.figure(figsize=(22, 14), facecolor=P['bg'])
    fig.patch.set_linewidth(0)

    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.42, wspace=0.30,
                           left=0.05, right=0.97, top=0.90, bottom=0.07)

    def style_ax(ax, grid_axis='both'):
        ax.set_facecolor(P['surface'])
        for spine in ax.spines.values():
            spine.set_color(P['border'])
            spine.set_linewidth(0.8)
        ax.tick_params(colors=P['slate'], labelsize=8.5, length=3)
        ax.grid(True, color=P['border'], lw=0.6, alpha=0.8, axis=grid_axis)
        ax.set_axisbelow(True)

    # ── 1. Price Paths ──
    ax1 = fig.add_subplot(gs[0, :2])
    style_ax(ax1)

    sample = np.array(paths_display['sample_paths'])
    for path in sample[:50]:
        ax1.plot(dr, path, color=accent, alpha=0.06, linewidth=0.6)

    ax1.fill_between(dr, p5, p95, color=accent, alpha=0.07, label='_nolegend_')
    ax1.fill_between(dr, p25, p75, color=accent, alpha=0.14, label='_nolegend_')

    ax1.plot(dr, p95, color=P['green'], lw=1.8, ls='--',
             label=f'P95 (Skenario Terbaik): {pfmt.format(p95[-1])}')
    ax1.plot(dr, mean_p, color=accent, lw=2.4,
             label=f'Rata-rata: {pfmt.format(mean_p[-1])}')
    ax1.plot(dr, p5, color=P['red'], lw=1.8, ls='--',
             label=f'P5 (Skenario Terburuk): {pfmt.format(p5[-1])}')
    ax1.axhline(last_price, color=P['dark'], lw=1.4, ls=':', alpha=0.5,
                label=f'Harga Saat Ini: {pfmt.format(last_price)}')

    title_prefix = 'Black Swan (Jump Diffusion)' if is_bs else 'Geometric Brownian Motion'
    ax1.set_title(f'Jalur Simulasi {title_prefix} — {ticker}',
                  color=P['dark'], fontsize=11.5, fontweight='700', pad=12, loc='left')
    ax1.set_xlabel('Hari Trading ke Depan', color=P['slate'], fontsize=9)
    ax1.set_ylabel('Harga', color=P['slate'], fontsize=9)

    legend = ax1.legend(fontsize=8, loc='upper left', framealpha=1,
                        facecolor=P['surface'], edgecolor=P['border'],
                        labelcolor=P['mid'])
    legend.get_frame().set_linewidth(0.8)

    # Annotation band labels
    ax1.text(dr[-1]*0.97, p95[-1], '90%', color=P['slate'], fontsize=7,
             va='center', ha='right', alpha=0.7)

    # ── 2. Probability Donut ──
    ax2 = fig.add_subplot(gs[0, 2])
    ax2.set_facecolor(P['surface'])
    ax2.set_aspect('equal')
    for s in ax2.spines.values(): s.set_visible(False)

    colors_pie = [P['green'] if stats['prob_up'] > 50 else P['red'],
                  P['red'] if stats['prob_up'] > 50 else P['green']]
    # Swap so that "up" is always green
    colors_pie = [P['green'], P['red']]

    wedges, _ = ax2.pie(
        [stats['prob_up'], stats['prob_down']],
        colors=colors_pie, startangle=90, counterclock=False,
        wedgeprops=dict(width=0.55, edgecolor=P['bg'], linewidth=3)
    )

    # Center text
    ax2.text(0, 0.18, f"{stats['prob_up']:.1f}%", ha='center', va='center',
             color=P['green'] if stats['prob_up'] > 50 else P['red'],
             fontsize=22, fontweight='800')
    ax2.text(0, -0.10, 'Probabilitas Naik', ha='center', va='center',
             color=P['mid'], fontsize=9, fontweight='600')
    ax2.text(0, -0.38, f"{stats['prob_down']:.1f}% Probabilitas Turun",
             ha='center', color=P['red'], fontsize=8.5)

    ax2.set_title('Distribusi Probabilitas', color=P['dark'], fontsize=11,
                  fontweight='700', pad=12)
    ax2.text(0, -0.65,
             f"Berdasarkan {simulations:,} simulasi independen\n"
             f"Persentase simulasi yang berakhir di atas\nharga saat ini ({pfmt.format(last_price)})",
             ha='center', color=P['slate'], fontsize=7.5, linespacing=1.5)

    # ── 3. Final Price Distribution ──
    ax3 = fig.add_subplot(gs[1, :2])
    style_ax(ax3, 'y')

    q5v, q95v = stats['q5'], stats['q95']
    n_bins = min(70, max(25, simulations // 12))
    counts, bins = np.histogram(final_prices, bins=n_bins)
    bin_width = (bins[1] - bins[0]) * 0.88

    for i in range(len(bins)-1):
        mid = (bins[i] + bins[i+1]) / 2
        if mid < q5v:
            c = P['red']
            alpha = 0.75
        elif mid > q95v:
            c = P['green']
            alpha = 0.75
        else:
            c = accent
            alpha = 0.55
        ax3.bar(mid, counts[i], width=bin_width, color=c, alpha=alpha, edgecolor='none')

    ax3.axvline(last_price, color=P['dark'], lw=2, ls=':', alpha=0.6,
                label='Harga Saat Ini')
    ax3.axvline(stats['mean'], color=accent, lw=2, ls='--',
                label=f"Rata-rata: {pfmt.format(stats['mean'])}")
    ax3.axvline(q5v, color=P['red'], lw=1.8, ls='-.',
                label=f"P5 / VaR 95%: {pfmt.format(q5v)}")
    ax3.axvline(q95v, color=P['green'], lw=1.8, ls='-.',
                label=f"P95 / Upside: {pfmt.format(q95v)}")

    ax3.axvspan(final_prices.min(), q5v, color=P['red'], alpha=0.05)
    ax3.axvspan(q95v, final_prices.max(), color=P['green'], alpha=0.05)

    ax3.set_title(f'Distribusi Harga Akhir setelah {days} Hari Trading',
                  color=P['dark'], fontsize=11.5, fontweight='700', pad=12, loc='left')
    ax3.set_xlabel('Harga Prediksi', color=P['slate'], fontsize=9)
    ax3.set_ylabel('Frekuensi Simulasi', color=P['slate'], fontsize=9)
    legend3 = ax3.legend(fontsize=8, loc='upper right', framealpha=1,
                         facecolor=P['surface'], edgecolor=P['border'], labelcolor=P['mid'])
    legend3.get_frame().set_linewidth(0.8)

    # Zone labels
    y_top = ax3.get_ylim()[1]
    ax3.text((final_prices.min() + q5v)/2, y_top*0.88, 'Zona\nRisiko',
             ha='center', color=P['red'], fontsize=7.5, fontweight='600', alpha=0.8)
    ax3.text((q95v + final_prices.max())/2, y_top*0.88, 'Zona\nUpside',
             ha='center', color=P['green'], fontsize=7.5, fontweight='600', alpha=0.8)

    # ── 4. Risk Summary Panel ──
    ax4 = fig.add_subplot(gs[1, 2])
    ax4.set_facecolor(P['surface'])
    ax4.axis('off')
    for s in ax4.spines.values(): s.set_visible(False)

    ax4.set_xlim(0, 1)
    ax4.set_ylim(0, 1)

    rows = [
        ('HARGA & PROYEKSI', None, P['slate'], 'header'),
        ('Harga Saat Ini', pfmt.format(last_price), P['dark'], 'row'),
        ('Proyeksi Rata-rata', pfmt.format(stats['mean']), accent, 'row'),
        ('Median Proyeksi', pfmt.format(stats['median']), P['mid'], 'row'),
        ('Expected Return', f"{(stats['mean']/last_price-1)*100:+.1f}%", P['green'] if stats['mean']>last_price else P['red'], 'row'),
        (None, None, None, 'spacer'),
        ('ANALISIS RISIKO (VaR)', None, P['slate'], 'header'),
        ('P5 — Terburuk 5%', pfmt.format(q5v), P['red'], 'row'),
        ('VaR 95% (Nominal)', pfmt.format(stats['var_95_abs']), P['red'], 'row'),
        ('VaR 95% (%)', f"-{stats['var_95_pct']:.1f}%", P['red'], 'row'),
        ('VaR 99% (%)', f"-{stats['var_99_pct']:.1f}%", P['red'], 'row'),
        (None, None, None, 'spacer'),
        ('SKENARIO UPSIDE', None, P['slate'], 'header'),
        ('P95 — Terbaik 5%', pfmt.format(q95v), P['green'], 'row'),
        ('Potensi Upside', f"+{(q95v/last_price-1)*100:.1f}%", P['green'], 'row'),
        (None, None, None, 'spacer'),
        ('PARAMETER MODEL', None, P['slate'], 'header'),
        ('μ Tahunan', f"{stats['mu_annual']*100:.2f}%", accent, 'row'),
        ('σ Tahunan (Volatilitas)', f"{stats['sigma_annual']*100:.2f}%", accent, 'row'),
        ('Data Historis', f"{stats['n_returns']} hari rtn", P['mid'], 'row'),
    ]

    if is_bs:
        rows.extend([
            (None, None, None, 'spacer'),
            ('PARAMETER BLACK SWAN', None, P['red'], 'header'),
            ('Intensitas Lompatan (λ)', f"{stats.get('jump_intensity', 2):.1f}×/tahun", P['red'], 'row'),
            ('Sim. dgn Jump', f"{stats.get('pct_with_jumps', 0):.1f}%", P['red'], 'row'),
        ])

    y = 0.98
    for label, value, color, kind in rows:
        if kind == 'spacer':
            y -= 0.02
            continue
        if kind == 'header':
            ax4.text(0.02, y, label, color=color, fontsize=7.5, fontweight='700',
                     transform=ax4.transAxes, va='top', alpha=0.7)
            ax4.plot([0.02, 0.98], [y - 0.018, y - 0.018], color=P['border'],
                     lw=0.8, transform=ax4.transAxes)
            y -= 0.048
        else:
            ax4.text(0.02, y, label, color=P['slate'], fontsize=7.8,
                     transform=ax4.transAxes, va='top')
            ax4.text(0.98, y, value, color=color, fontsize=7.8, fontweight='600',
                     transform=ax4.transAxes, va='top', ha='right')
            y -= 0.043

    ax4.set_title('Ringkasan Statistik', color=P['dark'], fontsize=11,
                  fontweight='700', pad=12)

    # ── Super title ──
    mode_label = '+ Black Swan (Jump Diffusion)' if is_bs else '— Geometric Brownian Motion'
    fig.text(0.5, 0.955, f'Simulasi Monte Carlo {mode_label}',
             ha='center', color=P['dark'], fontsize=14, fontweight='800')
    fig.text(0.5, 0.940,
             f'{ticker}  ·  {stats["n_returns"]} hari data historis  ·  {simulations:,} simulasi  ·  horizon {days} hari trading',
             ha='center', color=P['slate'], fontsize=9)

    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=140, bbox_inches='tight',
                facecolor=P['bg'], edgecolor='none')
    buf.seek(0)
    url = base64.b64encode(buf.getvalue()).decode()
    plt.close()
    return url

# ============================================
# ROUTES
# ============================================

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/simulate', methods=['POST'])
def simulate():
    try:
        raw_ticker = request.form.get('ticker', '').strip()
        days = int(request.form.get('days', 30))
        simulations = int(request.form.get('simulations', 1000))
        years = int(request.form.get('years', 3))
        use_blackswan = request.form.get('blackswan', 'false') == 'true'

        # Black Swan params (with defaults)
        jump_intensity = float(request.form.get('jump_intensity', 2.0))
        jump_mean = float(request.form.get('jump_mean', -0.15))
        jump_std = float(request.form.get('jump_std', 0.20))

        if not raw_ticker:
            return render_template('error.html', msg="Kode saham harus diisi.")

        ticker, exchange_hint = normalize_ticker(raw_ticker)

        errors = []
        if not (1 <= days <= 504):
            errors.append(f"Hari prediksi harus 1–504.")
        if not (100 <= simulations <= 10000):
            errors.append("Jumlah simulasi harus 100–10.000.")
        if not (1 <= years <= 10):
            errors.append("Data historis harus 1–10 tahun.")
        if errors:
            return render_template('error.html', msg=" | ".join(errors))

        stock_data, error = get_stock_data(ticker, years)
        if error and exchange_hint == 'IDX_guess':
            stock_data2, err2 = get_stock_data(raw_ticker.upper(), years)
            if stock_data2:
                ticker = raw_ticker.upper()
                stock_data, error = stock_data2, None
        if error or stock_data is None:
            return render_template('error.html',
                msg=f"Saham '{raw_ticker}' tidak ditemukan. Pastikan kode benar. ({error})")

        warnings_list = []
        avail = stock_data['trading_days']
        if days > TRADING_DAYS_PER_YEAR:
            warnings_list.append(
                f"Horizon prediksi {days} hari ({days/TRADING_DAYS_PER_YEAR:.1f} tahun) — akurasi berkurang untuk horizon panjang.")
        if days > avail // 2:
            warnings_list.append(
                f"Periode prediksi ({days} hari) melebihi 50% data historis yang tersedia ({avail} hari).")

        # Run simulation
        if use_blackswan:
            price_paths, stats, paths_display = run_black_swan(
                stock_data['prices'], days, simulations,
                jump_intensity=jump_intensity,
                jump_mean=jump_mean,
                jump_std=jump_std
            )
            mode = 'blackswan'
            warnings_list.append(
                "Mode Black Swan aktif: model jump diffusion Merton diterapkan. "
                f"Intensitas lompatan {jump_intensity:.1f}×/tahun dengan magnitude rata-rata {jump_mean*100:.0f}%."
            )
        else:
            price_paths, stats, paths_display = run_monte_carlo(
                stock_data['prices'], days, simulations
            )
            mode = 'standard'

        plot_url = make_chart(price_paths, stats, paths_display, ticker, days, simulations, mode=mode)

        pf = lambda v: price_fmt(v, ticker)

        calc_details = {
            'ticker': ticker,
            'last_price': stats['last_price'],
            'last_price_fmt': pf(stats['last_price']),
            'mu_daily': stats['mu_daily'],
            'sigma_daily': stats['sigma_daily'],
            'mu_annual': stats['mu_annual'],
            'sigma_annual': stats['sigma_annual'],
            'mu_adj_annual': stats['mu_adj_annual'],
            'n_returns': stats['n_returns'],
            'days': days,
            'simulations': simulations,
            'is_idr': is_indonesian(ticker),
            'currency': 'Rp' if is_indonesian(ticker) else '$',
            'mode': mode,
            'skewness': stats.get('skewness', 0),
            'kurtosis': stats.get('kurtosis', 0),
        }
        if use_blackswan:
            calc_details.update({
                'jump_intensity': stats.get('jump_intensity', jump_intensity),
                'jump_mean': stats.get('jump_mean', jump_mean),
                'jump_std': stats.get('jump_std', jump_std),
                'pct_with_jumps': stats.get('pct_with_jumps', 0),
                'avg_jumps_per_sim': stats.get('avg_jumps_per_sim', 0),
            })

        result = {
            'ticker': ticker,
            'ticker_short': ticker.replace('.JK', ''),
            'company_name': stock_data['company_name'],
            'sector': stock_data['sector'],
            'exchange_hint': exchange_hint,
            'days': days,
            'simulations': simulations,
            'years': years,
            'mode': mode,
            'warnings': warnings_list,
            'last_price': pf(stats['last_price']),
            'mean_price': pf(stats['mean']),
            'median_price': pf(stats['median']),
            'std_price': pf(stats['std']),
            'min_price': pf(stats['min']),
            'max_price': pf(stats['max']),
            'q5_price': pf(stats['q5']),
            'q25_price': pf(stats['q25']),
            'q75_price': pf(stats['q75']),
            'q95_price': pf(stats['q95']),
            'prob_up': f"{stats['prob_up']:.1f}%",
            'prob_down': f"{stats['prob_down']:.1f}%",
            'var_95_abs': pf(stats['var_95_abs']),
            'var_95_pct': f"{stats['var_95_pct']:.1f}%",
            'var_99_abs': pf(stats['var_99_abs']),
            'var_99_pct': f"{stats['var_99_pct']:.1f}%",
            'mu_annual': f"{stats['mu_annual']*100:.2f}%",
            'sigma_annual': f"{stats['sigma_annual']*100:.2f}%",
            'mu_daily': f"{stats['mu_daily']*100:.4f}%",
            'sigma_daily': f"{stats['sigma_daily']*100:.4f}%",
            'skewness': f"{stats.get('skewness', 0):.3f}",
            'kurtosis': f"{stats.get('kurtosis', 0):.3f}",
            'plot_url': plot_url,
            'trading_days': stock_data['trading_days'],
            'start_date': stock_data['start_date'],
            'end_date': stock_data['end_date'],
            'upside_pct': f"+{(stats['q95']/stats['last_price']-1)*100:.1f}%",
            'downside_pct': f"-{stats['var_95_pct']:.1f}%",
            'expected_return': f"{(stats['mean']/stats['last_price']-1)*100:+.1f}%",
            'calc_details': json.dumps(calc_details),
            # Black Swan specific
            'jump_intensity': stats.get('jump_intensity', jump_intensity) if use_blackswan else None,
            'jump_mean_pct': f"{jump_mean*100:.0f}%" if use_blackswan else None,
            'pct_with_jumps': f"{stats.get('pct_with_jumps', 0):.1f}%" if use_blackswan else None,
        }

        return render_template('result.html', r=result)

    except Exception as e:
        import traceback
        traceback.print_exc()
        return render_template('error.html', msg=f"Error: {str(e)}")


@app.route('/calculation')
def calculation():
    data = request.args.get('data', '{}')
    try:
        calc = json.loads(data)
    except Exception:
        calc = {}
    return render_template('calculation.html', c=calc)


@app.route('/health')
def health():
    return {'status': 'ok', 'ts': datetime.now().isoformat()}


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)