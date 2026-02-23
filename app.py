from flask import Flask, render_template, request, jsonify
import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from datetime import datetime, timedelta
import io
import base64
import warnings
import os
import json

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
    # heuristic: short alpha = IDX
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
# MONTE CARLO
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
        'sample_epsilon': [float(np.random.standard_normal()) for _ in range(3)],
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
# VISUALIZATION — dark theme
# ============================================

def make_chart(price_paths, stats, paths_display, ticker, days, simulations, dark=True):
    BG = '#12131a' if dark else 'white'
    CARD = '#1c1f2e' if dark else '#f8fafc'
    BLUE = '#4f9cf9'
    GREEN = '#22c55e'
    RED = '#ef4444'
    YELLOW = '#f59e0b'
    TEXT = '#e2e8f0' if dark else '#1e293b'
    GRID = '#2a2d3e' if dark else '#e2e8f0'

    last_price = stats['last_price']
    final_prices = np.array(paths_display['final_prices'])
    dr = np.arange(1, days + 1)
    p5 = np.array(paths_display['p5'])
    p25 = np.array(paths_display['p25'])
    p75 = np.array(paths_display['p75'])
    p95 = np.array(paths_display['p95'])
    mean_p = np.array(paths_display['mean'])
    pfmt = 'Rp{:,.0f}' if is_indonesian(ticker) else '${:,.2f}'

    fig = plt.figure(figsize=(20, 13), facecolor=BG)
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.32,
                           left=0.06, right=0.97, top=0.91, bottom=0.08)

    # ── 1. Paths ──
    ax1 = fig.add_subplot(gs[0, :2])
    ax1.set_facecolor(CARD)
    for s in ax1.spines.values(): s.set_color(GRID)
    ax1.tick_params(colors=TEXT, labelsize=8)

    sample = np.array(paths_display['sample_paths'])
    for path in sample[:40]:
        ax1.plot(dr, path, color=BLUE, alpha=0.07, linewidth=0.5)
    ax1.fill_between(dr, p5, p95, color=BLUE, alpha=0.10)
    ax1.fill_between(dr, p25, p75, color=BLUE, alpha=0.18)
    ax1.plot(dr, p95, color=GREEN, lw=1.6, ls='--', label=f'P95 Terbaik: {pfmt.format(p95[-1])}')
    ax1.plot(dr, mean_p, color=YELLOW, lw=2.2, label=f'Rata-rata: {pfmt.format(mean_p[-1])}')
    ax1.plot(dr, p5, color=RED, lw=1.6, ls='--', label=f'P5 Terburuk: {pfmt.format(p5[-1])}')
    ax1.axhline(last_price, color='white' if dark else 'black', lw=1.2, ls=':', alpha=0.6,
                label=f'Harga Sekarang: {pfmt.format(last_price)}')
    ax1.set_title(f'Jalur Simulasi Monte Carlo — {ticker} ({simulations:,} Simulasi)', color=TEXT, fontsize=11, fontweight='bold')
    ax1.set_xlabel('Hari Trading ke Depan', color=TEXT, fontsize=9)
    ax1.set_ylabel('Harga', color=TEXT, fontsize=9)
    ax1.legend(fontsize=7.5, facecolor=CARD, edgecolor=GRID, labelcolor=TEXT, loc='upper left', framealpha=0.9)
    ax1.grid(True, color=GRID, lw=0.5, alpha=0.7)

    # ── Anotasi Grafik 1 ──
    note1 = ("Setiap garis tipis = 1 simulasi perjalanan harga.\n"
             "Area biru = rentang kemungkinan harga (50% & 90%).\n"
             "P95 = 5% simulasi terbaik | P5 = 5% terburuk.")
    ax1.text(0.01, 0.02, note1, transform=ax1.transAxes, color='#94a3b8',
             fontsize=7, va='bottom', style='italic')

    # ── 2. Donut Probability ──
    ax2 = fig.add_subplot(gs[0, 2])
    ax2.set_facecolor(CARD)
    ax2.set_aspect('equal')
    ax2.pie([stats['prob_up'], stats['prob_down']],
            colors=[GREEN, RED], startangle=90, counterclock=False,
            wedgeprops=dict(width=0.52, edgecolor=CARD, linewidth=3))
    ax2.text(0, 0.15, f"{stats['prob_up']:.1f}%", ha='center', va='center',
             color=GREEN, fontsize=20, fontweight='bold')
    ax2.text(0, -0.18, 'Prob. Naik', ha='center', va='center', color=TEXT, fontsize=9)
    ax2.text(0, -0.45, f"Prob. Turun: {stats['prob_down']:.1f}%", ha='center',
             color=RED, fontsize=8.5)
    ax2.set_title('Probabilitas Arah Harga', color=TEXT, fontsize=10, fontweight='bold')
    for s in ax2.spines.values(): s.set_color(GRID)
    ax2.text(0, -0.75, "Dihitung dari % simulasi yang berakhir\ndi atas / bawah harga sekarang.",
             ha='center', color='#94a3b8', fontsize=7, style='italic')

    # ── 3. Distribusi Harga Akhir ──
    ax3 = fig.add_subplot(gs[1, :2])
    ax3.set_facecolor(CARD)
    for s in ax3.spines.values(): s.set_color(GRID)
    ax3.tick_params(colors=TEXT, labelsize=8)

    n_bins = min(60, max(20, simulations // 15))
    counts, bins = np.histogram(final_prices, bins=n_bins)
    q5v, q95v = stats['q5'], stats['q95']
    for i in range(len(bins)-1):
        mid = (bins[i] + bins[i+1]) / 2
        color = RED if mid < q5v else GREEN if mid > q95v else BLUE
        ax3.bar(mid, counts[i], width=(bins[1]-bins[0])*0.92, color=color, alpha=0.75, edgecolor='none')

    ax3.axvline(last_price, color='white' if dark else 'black', lw=2, ls=':', label='Harga Sekarang')
    ax3.axvline(stats['mean'], color=YELLOW, lw=2, ls='--', label=f"Mean: {pfmt.format(stats['mean'])}")
    ax3.axvline(q5v, color=RED, lw=1.8, ls='-.', label=f"P5 / VaR 95%: {pfmt.format(q5v)}")
    ax3.axvline(q95v, color=GREEN, lw=1.8, ls='-.', label=f"P95 / Upside: {pfmt.format(q95v)}")
    ax3.axvspan(final_prices.min(), q5v, color=RED, alpha=0.07)
    ax3.axvspan(q95v, final_prices.max(), color=GREEN, alpha=0.07)

    ax3.set_title(f'Distribusi Harga Akhir setelah {days} Hari Trading', color=TEXT, fontsize=11, fontweight='bold')
    ax3.set_xlabel('Harga Prediksi', color=TEXT, fontsize=9)
    ax3.set_ylabel('Jumlah Simulasi', color=TEXT, fontsize=9)
    ax3.legend(fontsize=7.5, facecolor=CARD, edgecolor=GRID, labelcolor=TEXT, loc='upper right', framealpha=0.9)
    ax3.grid(True, color=GRID, lw=0.5, alpha=0.5, axis='y')

    note3 = ("Merah = Zona Risiko (5% terburuk) | Biru = Zona Normal | Hijau = Zona Upside (5% terbaik).\n"
             "Grafik ini menunjukkan seberapa sering setiap harga muncul di akhir simulasi.")
    ax3.text(0.01, 0.02, note3, transform=ax3.transAxes, color='#94a3b8',
             fontsize=7, va='bottom', style='italic')

    # ── 4. Risk Summary ──
    ax4 = fig.add_subplot(gs[1, 2])
    ax4.set_facecolor(CARD)
    ax4.axis('off')
    for s in ax4.spines.values(): s.set_color(GRID)

    rows = [
        ('HARGA REFERENSI', None, '#64748b'),
        ('Harga Sekarang', pfmt.format(last_price), TEXT),
        ('Mean Prediksi', pfmt.format(stats['mean']), YELLOW),
        ('Median Prediksi', pfmt.format(stats['median']), TEXT),
        ('Expected Return', f"+{(stats['mean']/last_price-1)*100:.1f}%", YELLOW),
        (None, None, None),
        ('SKENARIO RISIKO', None, RED),
        ('P5 — Terburuk 5%', pfmt.format(q5v), RED),
        ('VaR 95% (Rp)', pfmt.format(stats['var_95_abs']), RED),
        ('VaR 95% (%)', f"-{stats['var_95_pct']:.1f}%", RED),
        ('VaR 99% (%)', f"-{stats['var_99_pct']:.1f}%", RED),
        (None, None, None),
        ('SKENARIO NAIK', None, GREEN),
        ('P95 — Terbaik 5%', pfmt.format(q95v), GREEN),
        ('Potensi Upside', f"+{(q95v/last_price-1)*100:.1f}%", GREEN),
        (None, None, None),
        ('PARAMETER GBM', None, BLUE),
        ('μ Tahunan', f"{stats['mu_annual']*100:.2f}%", BLUE),
        ('σ Tahunan (Vol)', f"{stats['sigma_annual']*100:.2f}%", BLUE),
        ('Data Historis', f"{stats['n_returns']} hari", BLUE),
    ]

    ax4.set_xlim(0, 1); ax4.set_ylim(0, 1)
    y = 0.97
    for label, value, color in rows:
        if label is None:
            y -= 0.025
            continue
        if value is None:
            ax4.text(0.03, y, label, color=color, fontsize=7.5, fontweight='bold',
                     transform=ax4.transAxes, va='top')
            ax4.plot([0.03, 0.97], [y - 0.015, y - 0.015], color=GRID, lw=0.8, transform=ax4.transAxes)
            y -= 0.045
        else:
            ax4.text(0.03, y, label, color='#94a3b8', fontsize=7.5, transform=ax4.transAxes, va='top')
            ax4.text(0.97, y, value, color=color, fontsize=7.5, fontweight='bold',
                     transform=ax4.transAxes, va='top', ha='right')
            y -= 0.045

    ax4.set_title('Ringkasan Risiko & Statistik', color=TEXT, fontsize=10, fontweight='bold')

    fig.text(0.5, 0.965, f'Analisis Monte Carlo — {ticker}',
             ha='center', color=TEXT, fontsize=14, fontweight='bold')
    fig.text(0.5, 0.952,
             f'{stats["n_returns"]} hari data historis | {simulations:,} simulasi | Horizon {days} hari trading',
             ha='center', color='#64748b', fontsize=9)

    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=130, bbox_inches='tight', facecolor=BG)
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

        if not raw_ticker:
            return render_template('error.html', msg="Kode saham harus diisi!")

        ticker, exchange_hint = normalize_ticker(raw_ticker)

        errors = []
        if not (1 <= days <= 504):
            errors.append(f"Hari prediksi harus 1–504 (maks 2 tahun = {2*TRADING_DAYS_PER_YEAR} hari trading).")
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
                f"Prediksi {days} hari ({days/TRADING_DAYS_PER_YEAR:.1f}× setahun) — akurasi berkurang untuk horizon panjang.")
        if days > avail // 2:
            warnings_list.append(
                f"Periode prediksi ({days} hari) melebihi 50% data historis ({avail} hari trading tersedia).")

        price_paths, stats, paths_display = run_monte_carlo(stock_data['prices'], days, simulations)
        plot_url = make_chart(price_paths, stats, paths_display, ticker, days, simulations, dark=True)
        print_url = make_chart(price_paths, stats, paths_display, ticker, days, simulations, dark=False)

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
            'sample_returns': [f"{r*100:.4f}%" for r in stats['sample_returns']],
            'sample_epsilon': [round(e, 4) for e in stats['sample_epsilon']],
            'is_idr': is_indonesian(ticker),
            'currency': 'Rp' if is_indonesian(ticker) else '$',
        }

        result = {
            'ticker': ticker,
            'ticker_short': ticker.replace('.JK', ''),
            'company_name': stock_data['company_name'],
            'sector': stock_data['sector'],
            'exchange_hint': exchange_hint,
            'days': days,
            'simulations': simulations,
            'years': years,
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
            'plot_url': plot_url,
            'print_url': print_url,
            'trading_days': stock_data['trading_days'],
            'start_date': stock_data['start_date'],
            'end_date': stock_data['end_date'],
            'upside_pct': f"+{(stats['q95']/stats['last_price']-1)*100:.1f}%",
            'downside_pct': f"-{stats['var_95_pct']:.1f}%",
            'expected_return': f"{(stats['mean']/stats['last_price']-1)*100:+.1f}%",
            'calc_details': json.dumps(calc_details),
        }

        return render_template('result.html', r=result)

    except Exception as e:
        return render_template('error.html', msg=f"Error: {str(e)}")


@app.route('/calculation')
def calculation():
    return render_template('calculation.html')


@app.route('/health')
def health():
    return {'status': 'ok', 'ts': datetime.now().isoformat()}


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)