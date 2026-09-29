import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, peak_prominences
from scipy.optimize import brentq
from pathlib import Path
import warnings

# compatibilidad numpy: trapezoid (>=2.0) o trapz (<2.0)
_trapz_compat = getattr(np, 'trapezoid', getattr(np, 'trapz', None))

# --- casos de prueba: se leen de casos.py (7 valores arbitrarios de Q_alpha) ---
from casos import Q_CASOS
Q_alpha_values = list(Q_CASOS)
L_values = [2.5]   # a = 2.5 (radio del pozo)

x_min = 1e-6
max_peaks_per_function = 10
x_max = float(np.pi * (max(Q_alpha_values) - 0.5 + max_peaks_per_function + 8.0))
nx = 800000

prominence_relative = 0.015
min_segment_points = 150
quadratic_half_window = 8
bound_scan_points = 300000
peak_match_tolerance_x = 0.75 * np.pi

output_dir = Path('salida_Qalpha')
output_dir.mkdir(exist_ok=True)
plots_dir = output_dir / 'plots'
plots_dir.mkdir(exist_ok=True)


def alpha_from_Qalpha(Qalpha):
    # Relacion del PDF (Sec. 1):  Q_alpha = alpha/pi + 1/2  =>  alpha = pi (Q_alpha - 1/2)
    # Umbral (nuevo estado acotado) en Q_alpha entero;  alpha_N = (2N-1) pi/2.
    return np.pi * (Qalpha - 0.5)


def V2_from_alpha_L(alpha, L):
    return alpha**2 / (2.0 * L**2)


def process_dispersion_functions(alpha, L, x):
    k = x / L
    V2 = V2_from_alpha_L(alpha, L)
    with np.errstate(divide='ignore', invalid='ignore', over='ignore'):
        q = np.sqrt(2.0 * V2 + k**2)
        gamma = q * L
        tan_gamma = np.tan(gamma)
        cos_gamma = np.cos(gamma)
        phi = np.arctan((k / q) * tan_gamma)
        cos_phi = np.cos(phi)
        lk = ((2.0 * q) / (q**2 + k**2 * tan_gamma**2)) * (
            (1.0 - (k**2 / q**2)) * tan_gamma + (k**2 / q) * L * (1.0 / cos_gamma)**2
        )
        tau = (1.0 / k) * (lk - 2.0 * L)
        Tdt = (1.0 / k) * lk
        TdP = (1.0 / k) * (lk - (np.sin(2.0 * phi) / k))
        Pk = (k**2 / (L * q**3)) * (2.0 * q * L - np.sin(2.0 * gamma)) * (cos_phi**2 / cos_gamma**2)
        sigmaphi = ((4.0 * np.pi) / k**2) * np.sin(phi)**2   # ec.(3) PDF (quitado el factor 2 del original)
    functions = {
        'lk': lk,
        'tau': tau,
        'Tdt': Tdt,
        'TdP': TdP,
        'Pk': Pk,
        'sigmaphi': sigmaphi,
    }
    cleaned = {}
    for name, y in functions.items():
        y = np.asarray(y, dtype=float)
        y[~np.isfinite(y)] = np.nan
        cleaned[name] = y
    return k, cleaned


def contiguous_true_segments(mask):
    idx = np.flatnonzero(mask)
    if idx.size == 0:
        return []
    splits = np.where(np.diff(idx) > 1)[0] + 1
    blocks = np.split(idx, splits)
    return [(b[0], b[-1] + 1) for b in blocks if b.size > 0]


def linear_baseline_from_points(x1, y1, x2, y2, x):
    if x2 == x1:
        return np.full_like(x, y1, dtype=float)
    m = (y2 - y1) / (x2 - x1)
    return y1 + m * (x - x1)


def refine_peak_position(x_seg, y_seg, idx):
    if idx <= 0 or idx >= len(x_seg) - 1:
        return float(x_seg[idx]), float(y_seg[idx])
    x0, x1, x2 = x_seg[idx - 1], x_seg[idx], x_seg[idx + 1]
    y0, y1, y2 = y_seg[idx - 1], y_seg[idx], y_seg[idx + 1]
    try:
        a, b, c = np.polyfit([x0, x1, x2], [y0, y1, y2], 2)
    except Exception:
        return float(x1), float(y1)
    if a >= 0:
        return float(x1), float(y1)
    xv = -b / (2.0 * a)
    if xv < x0 or xv > x2:
        return float(x1), float(y1)
    yv = a * xv**2 + b * xv + c
    return float(xv), float(yv)


def refine_minimum_position(x_seg, y_seg, idx):
    x_ref, y_ref = refine_peak_position(x_seg, -y_seg, idx)
    return x_ref, -y_ref


def choose_local_boundaries(peaks, minima, left_bases, right_bases):
    boundary_info = []
    minima = np.sort(np.unique(minima))
    for i, peak_idx in enumerate(peaks):
        left_candidates = minima[minima < peak_idx]
        right_candidates = minima[minima > peak_idx]
        if left_candidates.size > 0 and right_candidates.size > 0:
            left_idx = int(left_candidates[-1])
            right_idx = int(right_candidates[0])
            method = 'between_minima'
        else:
            left_idx = int(left_bases[i])
            right_idx = int(right_bases[i])
            method = 'prominence_fallback'
        if left_idx >= peak_idx or right_idx <= peak_idx:
            left_idx = int(left_bases[i])
            right_idx = int(right_bases[i])
            method = 'prominence_fallback'
        boundary_info.append((left_idx, right_idx, method))
    return boundary_info


def halfprom_width_refined(x_seg, y_seg, peak_idx, left_base_idx, right_base_idx):
    x_peak, y_peak = refine_peak_position(x_seg, y_seg, peak_idx)
    x_left_base = x_seg[left_base_idx]
    y_left_base = y_seg[left_base_idx]
    x_right_base = x_seg[right_base_idx]
    y_right_base = y_seg[right_base_idx]
    y_base_peak = np.interp(
        x_peak,
        [x_left_base, x_right_base],
        [y_left_base, y_right_base],
    )
    prominence = y_peak - y_base_peak
    if not np.isfinite(prominence) or prominence <= 0:
        return np.nan, np.nan, np.nan, x_peak, y_peak, prominence, y_base_peak
    level = y_base_peak + 0.5 * prominence
    left_cross = np.nan
    for j in range(peak_idx, left_base_idx, -1):
        yj = y_seg[j] - level
        yj1 = y_seg[j - 1] - level
        if yj == 0:
            left_cross = x_seg[j]
            break
        if yj * yj1 <= 0:
            left_cross = np.interp(0.0, [yj1, yj], [x_seg[j - 1], x_seg[j]])
            break
    right_cross = np.nan
    for j in range(peak_idx, right_base_idx):
        yj = y_seg[j] - level
        yj1 = y_seg[j + 1] - level
        if yj == 0:
            right_cross = x_seg[j]
            break
        if yj * yj1 <= 0:
            right_cross = np.interp(0.0, [yj, yj1], [x_seg[j], x_seg[j + 1]])
            break
    width = right_cross - left_cross if np.isfinite(left_cross) and np.isfinite(right_cross) else np.nan
    return width, left_cross, right_cross, x_peak, y_peak, prominence, y_base_peak


def curvature_width(x_seg, y_seg, peak_idx, left_base, right_base, prominence):
    left = max(left_base, peak_idx - quadratic_half_window)
    right = min(right_base, peak_idx + quadratic_half_window)
    if right - left + 1 < 5:
        return np.nan
    xx = x_seg[left:right + 1] - x_seg[peak_idx]
    yy = y_seg[left:right + 1]
    try:
        a, b, c = np.polyfit(xx, yy, 2)
    except Exception:
        return np.nan
    if a >= 0 or prominence <= 0:
        return np.nan
    return float(np.sqrt(-prominence / a))


def area_width_refined(x_seg, y_seg, peak_idx, left_base, right_base, x_peak):
    if right_base <= left_base:
        return np.nan, np.nan
    x_local = x_seg[left_base:right_base + 1]
    y_local = y_seg[left_base:right_base + 1]
    baseline = linear_baseline_from_points(x_local[0], y_local[0], x_local[-1], y_local[-1], x_local)
    corrected = y_local - baseline
    corrected[corrected < 0] = 0
    peak_height = np.interp(x_peak, x_local, y_local) - np.interp(x_peak, x_local, baseline)
    if peak_height <= 0:
        return np.nan, np.nan
    area = _trapz_compat(corrected, x_local)
    return float(area / peak_height), float(peak_height)


def detect_peaks_and_widths(x, y, L, alpha, function_name):
    records = []
    finite_mask = np.isfinite(y)
    segments = contiguous_true_segments(finite_mask)
    V2 = V2_from_alpha_L(alpha, L)
    for start, end in segments:
        if end - start < min_segment_points:
            continue
        x_seg = x[start:end]
        y_seg = y[start:end]
        finite_dynamic_range = np.nanmax(y_seg) - np.nanmin(y_seg)
        if not np.isfinite(finite_dynamic_range) or finite_dynamic_range <= 0:
            continue
        local_distance = max(4, int(0.0015 * len(y_seg)))
        peaks, _ = find_peaks(y_seg, distance=local_distance)
        if peaks.size == 0:
            continue
        minima, _ = find_peaks(-y_seg, distance=local_distance)
        prominences, left_bases, right_bases = peak_prominences(y_seg, peaks)
        if prominences.size == 0:
            continue
        prom_cut = prominence_relative * np.nanmax(prominences)
        keep = prominences >= prom_cut
        peaks = peaks[keep]
        prominences = prominences[keep]
        left_bases = left_bases[keep]
        right_bases = right_bases[keep]
        if peaks.size == 0:
            continue
        boundary_info = choose_local_boundaries(peaks, minima, left_bases, right_bases)
        for i, peak_idx_seg in enumerate(peaks):
            left_idx_seg, right_idx_seg, width_method = boundary_info[i]
            width_x_halfprom, left_x_half, right_x_half, x_peak, y_peak, prominence, y_base_peak = halfprom_width_refined(
                x_seg, y_seg, peak_idx_seg, left_idx_seg, right_idx_seg
            )
            width_k_halfprom = width_x_halfprom / L if np.isfinite(width_x_halfprom) else np.nan
            width_x_curv = curvature_width(x_seg, y_seg, peak_idx_seg, left_idx_seg, right_idx_seg, prominence)
            width_k_curv = width_x_curv / L if np.isfinite(width_x_curv) else np.nan
            width_x_area, corrected_height = area_width_refined(x_seg, y_seg, peak_idx_seg, left_idx_seg, right_idx_seg, x_peak)
            width_k_area = width_x_area / L if np.isfinite(width_x_area) else np.nan
            k_peak = x_peak / L
            q_peak = np.sqrt(k_peak**2 + 2.0 * V2)
            xq_peak = q_peak * L
            global_peak_idx = start + peak_idx_seg
            left_boundary_g = start + left_idx_seg
            right_boundary_g = start + right_idx_seg
            x_left_boundary = x[left_boundary_g]
            x_right_boundary = x[right_boundary_g]
            y_left_boundary = y[left_boundary_g]
            y_right_boundary = y[right_boundary_g]
            x_left_min_ref, y_left_min_ref = refine_minimum_position(x_seg, y_seg, left_idx_seg)
            x_right_min_ref, y_right_min_ref = refine_minimum_position(x_seg, y_seg, right_idx_seg)
            records.append({
                'funcion': function_name,
                'peak_index_temp': global_peak_idx,
                'k_peak': k_peak,
                'x_peak': x_peak,
                'q_peak': q_peak,
                'xq_peak': xq_peak,
                'y_peak': y_peak,
                'prominence': prominence,
                'baseline_en_pico': y_base_peak,
                'width_method': width_method,
                'width_k_halfprom': width_k_halfprom,
                'width_x_halfprom': width_x_halfprom,
                'width_k_curv': width_k_curv,
                'width_x_curv': width_x_curv,
                'width_k_area': width_k_area,
                'width_x_area': width_x_area,
                'left_half_x': left_x_half,
                'right_half_x': right_x_half,
                'left_half_k': left_x_half / L if np.isfinite(left_x_half) else np.nan,
                'right_half_k': right_x_half / L if np.isfinite(right_x_half) else np.nan,
                'left_boundary_k': x_left_boundary / L,
                'right_boundary_k': x_right_boundary / L,
                'left_boundary_x': x_left_boundary,
                'right_boundary_x': x_right_boundary,
                'left_boundary_y': y_left_boundary,
                'right_boundary_y': y_right_boundary,
                'left_min_k_refined': x_left_min_ref / L if np.isfinite(x_left_min_ref) else np.nan,
                'right_min_k_refined': x_right_min_ref / L if np.isfinite(x_right_min_ref) else np.nan,
                'left_min_x_refined': x_left_min_ref,
                'right_min_x_refined': x_right_min_ref,
                'left_min_y_refined': y_left_min_ref,
                'right_min_y_refined': y_right_min_ref,
            })
    if not records:
        return []
    records = sorted(records, key=lambda r: r['k_peak'])[:max_peaks_per_function]
    filtered = []
    for rec in records:
        if not filtered:
            filtered.append(rec)
            continue
        if abs(rec['x_peak'] - filtered[-1]['x_peak']) > 5e-4:
            filtered.append(rec)
    for i, rec in enumerate(filtered, start=1):
        rec['peak_order'] = i
        rec.pop('peak_index_temp', None)
    return filtered


def bound_function_xq(xq, alpha):
    inside = alpha**2 - xq**2
    if inside < 0:
        return np.nan
    chiL = np.sqrt(inside)
    return xq * np.cos(xq) + chiL * np.sin(xq)


def compute_bound_states(alpha, L):
    xq_grid = np.linspace(1e-10, alpha - 1e-10, bound_scan_points)
    inside = alpha**2 - xq_grid**2
    chiL = np.sqrt(np.clip(inside, 0.0, None))
    vals = xq_grid * np.cos(xq_grid) + chiL * np.sin(xq_grid)
    vals[inside < 0] = np.nan
    roots_xq = []
    for i in range(len(xq_grid) - 1):
        f1 = vals[i]
        f2 = vals[i + 1]
        if not np.isfinite(f1) or not np.isfinite(f2):
            continue
        x1 = xq_grid[i]
        x2 = xq_grid[i + 1]
        if f1 == 0.0:
            root = x1
        elif f1 * f2 > 0:
            continue
        else:
            try:
                root = brentq(lambda z: bound_function_xq(z, alpha), x1, x2, xtol=1e-15, rtol=1e-13, maxiter=500)
            except ValueError:
                continue
        if root <= 0 or root >= alpha:
            continue
        if any(abs(root - r) < 1e-10 for r in roots_xq):
            continue
        roots_xq.append(root)
    roots_xq = sorted(roots_xq)
    records = []
    for i, xq in enumerate(roots_xq, start=1):
        q = xq / L
        chiL = np.sqrt(max(alpha**2 - xq**2, 0.0))
        chi = chiL / L
        records.append({
            'bound_order': i,
            'xq_bound': xq,
            'q_bound': q,
            'chiL_bound': chiL,
            'chi_bound': chi,
            'k_bound_imag': chi,
            'E_bound': -0.5 * chi**2,
        })
    return records


def compare_with_infinite_well(Qalpha, alpha, L, bound_records, peak_records):
    nb = len(bound_records)
    comparisons = []
    for rec in bound_records:
        n = rec['bound_order']
        xq_inf = n * np.pi
        q_inf = xq_inf / L
        diff_pct = 100.0 * (rec['xq_bound'] - xq_inf) / xq_inf if xq_inf != 0 else np.nan
        comparisons.append({
            'Qalpha': Qalpha,
            'alpha': alpha,
            'L': L,
            'tipo': 'bound_state',
            'funcion': 'bound',
            'order_local': n,
            'order_infinite': n,
            'valor_finito': rec['q_bound'],
            'valor_finito_xL': rec['xq_bound'],
            'valor_infinito': q_inf,
            'valor_infinito_xL': xq_inf,
            'diferencia_absoluta': rec['q_bound'] - q_inf,
            'diferencia_absoluta_xL': rec['xq_bound'] - xq_inf,
            'diferencia_porcentual': diff_pct,
        })
    grouped = {}
    for rec in peak_records:
        grouped.setdefault(rec['funcion'], []).append(rec)
    for func, items in grouped.items():
        items = sorted(items, key=lambda r: r['peak_order'])
        for rec in items:
            n_inf = nb + rec['peak_order']
            xq_inf = n_inf * np.pi
            q_inf = xq_inf / L
            diff_pct = 100.0 * (rec['xq_peak'] - xq_inf) / xq_inf if xq_inf != 0 else np.nan
            comparisons.append({
                'Qalpha': Qalpha,
                'alpha': alpha,
                'L': L,
                'tipo': 'peak_qstar',
                'funcion': func,
                'order_local': rec['peak_order'],
                'order_infinite': n_inf,
                'valor_finito': rec['q_peak'],
                'valor_finito_xL': rec['xq_peak'],
                'valor_infinito': q_inf,
                'valor_infinito_xL': xq_inf,
                'diferencia_absoluta': rec['q_peak'] - q_inf,
                'diferencia_absoluta_xL': rec['xq_peak'] - xq_inf,
                'diferencia_porcentual': diff_pct,
            })
    return comparisons


def consecutive_peak_comparisons(Qalpha, alpha, L, peak_records):
    rows = []
    grouped = {}
    for rec in peak_records:
        grouped.setdefault(rec['funcion'], []).append(rec)
    for func, items in grouped.items():
        items = sorted(items, key=lambda r: r['peak_order'])
        for i in range(len(items) - 1):
            a = items[i]
            b = items[i + 1]
            rows.append({
                'Qalpha': Qalpha,
                'alpha': alpha,
                'L': L,
                'funcion': func,
                'peak_n': a['peak_order'],
                'peak_n1': b['peak_order'],
                'k_n': a['k_peak'],
                'k_n1': b['k_peak'],
                'delta_k': b['k_peak'] - a['k_peak'],
                'x_n': a['x_peak'],
                'x_n1': b['x_peak'],
                'delta_x': b['x_peak'] - a['x_peak'],
                'q_n': a['q_peak'],
                'q_n1': b['q_peak'],
                'delta_q': b['q_peak'] - a['q_peak'],
                'xq_n': a['xq_peak'],
                'xq_n1': b['xq_peak'],
                'delta_xq': b['xq_peak'] - a['xq_peak'],
                'width_k_n': a['width_k_halfprom'],
                'width_k_n1': b['width_k_halfprom'],
                'delta_width_k': b['width_k_halfprom'] - a['width_k_halfprom'],
                'width_x_n': a['width_x_halfprom'],
                'width_x_n1': b['width_x_halfprom'],
                'delta_width_x': b['width_x_halfprom'] - a['width_x_halfprom'],
                'spacing_vs_pi_over_L': (b['q_peak'] - a['q_peak']) / (np.pi / L),
                'spacing_xq_vs_pi': (b['xq_peak'] - a['xq_peak']) / np.pi,
            })
    return rows


def summarize_trends(df_peaks, df_pairs, df_infinite):
    rows = []
    if df_peaks.empty:
        return pd.DataFrame()
    for (Qalpha, alpha, L, func), sub in df_peaks.groupby(['Qalpha', 'alpha', 'L', 'funcion']):
        sub = sub.sort_values('peak_order')
        width_slope_k = np.nan
        width_slope_x = np.nan
        mwk = sub['width_k_halfprom'].notna()
        if mwk.sum() >= 2:
            width_slope_k = np.polyfit(sub['peak_order'][mwk], sub['width_k_halfprom'][mwk], 1)[0]
        mwx = sub['width_x_halfprom'].notna()
        if mwx.sum() >= 2:
            width_slope_x = np.polyfit(sub['peak_order'][mwx], sub['width_x_halfprom'][mwx], 1)[0]
        pair_sub = df_pairs[(df_pairs['Qalpha'] == Qalpha) & (df_pairs['alpha'] == alpha) & (df_pairs['L'] == L) & (df_pairs['funcion'] == func)].sort_values('peak_n')
        spacing_slope_k = np.nan
        spacing_slope_q = np.nan
        msk = pair_sub['delta_k'].notna()
        if msk.sum() >= 2:
            spacing_slope_k = np.polyfit(pair_sub['peak_n'][msk], pair_sub['delta_k'][msk], 1)[0]
        msq = pair_sub['delta_q'].notna()
        if msq.sum() >= 2:
            spacing_slope_q = np.polyfit(pair_sub['peak_n'][msq], pair_sub['delta_q'][msq], 1)[0]
        if np.isfinite(spacing_slope_q):
            if spacing_slope_q > 1e-4:
                spacing_trend = 'se alejan'
            elif spacing_slope_q < -1e-4:
                spacing_trend = 'se acercan'
            else:
                spacing_trend = 'espaciado casi constante'
        else:
            spacing_trend = 'sin datos suficientes'
        if np.isfinite(width_slope_k):
            if width_slope_k > 1e-4:
                width_trend_k = 'se ensanchan'
            elif width_slope_k < -1e-4:
                width_trend_k = 'se angostan'
            else:
                width_trend_k = 'ancho casi constante'
        else:
            width_trend_k = 'sin datos suficientes'
        if np.isfinite(width_slope_x):
            if width_slope_x > 1e-4:
                width_trend_x = 'se ensanchan'
            elif width_slope_x < -1e-4:
                width_trend_x = 'se angostan'
            else:
                width_trend_x = 'ancho casi constante'
        else:
            width_trend_x = 'sin datos suficientes'
        inf_sub = df_infinite[(df_infinite['Qalpha'] == Qalpha) & (df_infinite['L'] == L) & (df_infinite['funcion'] == func) & (df_infinite['tipo'] == 'peak_qstar')].sort_values('order_local')
        rows.append({
            'Qalpha': Qalpha,
            'alpha': alpha,
            'L': L,
            'funcion': func,
            'num_peaks': len(sub),
            'num_pairs': len(pair_sub),
            'spacing_slope_k': spacing_slope_k,
            'spacing_slope_q': spacing_slope_q,
            'spacing_trend': spacing_trend,
            'mean_delta_k': pair_sub['delta_k'].mean() if len(pair_sub) else np.nan,
            'mean_delta_q': pair_sub['delta_q'].mean() if len(pair_sub) else np.nan,
            'mean_delta_xq': pair_sub['delta_xq'].mean() if len(pair_sub) else np.nan,
            'mean_spacing_xq_vs_pi': pair_sub['spacing_xq_vs_pi'].mean() if len(pair_sub) else np.nan,
            'width_slope_k': width_slope_k,
            'width_slope_x': width_slope_x,
            'width_trend_k': width_trend_k,
            'width_trend_x': width_trend_x,
            'mean_width_k': sub['width_k_halfprom'].mean(),
            'mean_width_x': sub['width_x_halfprom'].mean(),
            'fraction_between_minima': (sub['width_method'] == 'between_minima').mean(),
            'mean_diff_pct_vs_infinite_qstar': inf_sub['diferencia_porcentual'].mean() if len(inf_sub) else np.nan,
            'max_abs_diff_pct_vs_infinite_qstar': inf_sub['diferencia_porcentual'].abs().max() if len(inf_sub) else np.nan,
        })
    return pd.DataFrame(rows)


def plot_case_functions(Qalpha, alpha, L, k, functions, peaks_df, bound_df):
    for func, y in functions.items():
        plt.figure(figsize=(8, 4.8))
        plt.plot(k, y, lw=1.0)
        subset = peaks_df[(peaks_df['Qalpha'] == Qalpha) & (peaks_df['L'] == L) & (peaks_df['funcion'] == func)]
        for _, row in subset.iterrows():
            plt.plot(row['k_peak'], row['y_peak'], 'o')
            if np.isfinite(row.get('left_min_k_refined', np.nan)) and np.isfinite(row.get('right_min_k_refined', np.nan)):
                plt.plot([row['left_min_k_refined'], row['right_min_k_refined']], [row['left_min_y_refined'], row['right_min_y_refined']], 's', ms=3)
            if np.isfinite(row.get('left_half_k', np.nan)) and np.isfinite(row.get('right_half_k', np.nan)):
                y_half = row['baseline_en_pico'] + 0.5 * row['prominence']
                plt.hlines(y_half, row['left_half_k'], row['right_half_k'], lw=0.9)
        for _, row in bound_df.iterrows():
            plt.axvline(row['q_bound'], ls='--', lw=0.8, alpha=0.35)
        plt.xlabel('k')
        plt.ylabel(func)
        plt.title(fr'{func}, $Q_\alpha={Qalpha:g}$, $L={L:g}$')
        if func == 'tau':
            finite_vals = y[np.isfinite(y)]
            if finite_vals.size:
                plt.ylim(bottom=-L, top=np.nanmax(finite_vals) * 1.05)
            else:
                plt.ylim(bottom=-L)
        plt.grid(True, alpha=0.25)
        plt.tight_layout()
        plt.savefig(plots_dir / f'{func}_Qa{Qalpha:g}_L{L:g}.png', dpi=200)
        plt.close()


def plot_spacing_and_width(df_pairs):
    if df_pairs.empty:
        return
    for (Qalpha, L, func), sub in df_pairs.groupby(['Qalpha', 'L', 'funcion']):
        sub = sub.sort_values('peak_n')
        plt.figure(figsize=(7, 4.4))
        plt.plot(sub['peak_n'], sub['delta_q'], marker='o')
        plt.xlabel('n')
        plt.ylabel(r'$q_{n+1}^* - q_n^*$')
        plt.title(fr'{func}, $Q_\alpha={Qalpha:g}$, $L={L:g}$')
        plt.grid(True, alpha=0.25)
        plt.tight_layout()
        plt.savefig(plots_dir / f'spacing_q_{func}_Qa{Qalpha:g}_L{L:g}.png', dpi=200)
        plt.close()

        plt.figure(figsize=(7, 4.4))
        plt.plot(sub['peak_n'], sub['width_k_n'], marker='o', label=r'$w_n$')
        plt.plot(sub['peak_n1'], sub['width_k_n1'], marker='s', label=r'$w_{n+1}$')
        plt.xlabel('n')
        plt.ylabel('ancho en k (half-prominence)')
        plt.title(fr'{func}, $Q_\alpha={Qalpha:g}$, $L={L:g}$')
        plt.grid(True, alpha=0.25)
        plt.legend()
        plt.tight_layout()
        plt.savefig(plots_dir / f'width_{func}_Qa{Qalpha:g}_L{L:g}.png', dpi=200)
        plt.close()


def main():
    x = np.linspace(x_min, x_max, nx)
    peak_rows = []
    pair_rows = []
    bound_rows = []
    infinite_rows = []
    for Qalpha in Q_alpha_values:
        alpha = alpha_from_Qalpha(Qalpha)
        for L in L_values:
            print(f'Procesando Q_alpha={Qalpha}, alpha={alpha:.5f}, L={L}')
            k, functions = process_dispersion_functions(alpha, L, x)
            case_peaks = []
            for func_name, y in functions.items():
                peaks = detect_peaks_and_widths(x, y, L, alpha, func_name)
                for rec in peaks:
                    rec['Qalpha'] = Qalpha
                    rec['alpha'] = alpha
                    rec['L'] = L
                    rec['V0_abs'] = alpha**2 / (2.0 * L**2)
                case_peaks.extend(peaks)
            case_bounds = compute_bound_states(alpha, L)
            for rec in case_bounds:
                rec['Qalpha'] = Qalpha
                rec['alpha'] = alpha
                rec['L'] = L
                rec['V0_abs'] = alpha**2 / (2.0 * L**2)
            peak_rows.extend(case_peaks)
            bound_rows.extend(case_bounds)
            pair_rows.extend(consecutive_peak_comparisons(Qalpha, alpha, L, case_peaks))
            infinite_rows.extend(compare_with_infinite_well(Qalpha, alpha, L, case_bounds, case_peaks))
            peaks_df_case = pd.DataFrame(case_peaks)
            bound_df_case = pd.DataFrame(case_bounds)
            plot_case_functions(Qalpha, alpha, L, k, functions, peaks_df_case, bound_df_case)
    df_peaks = pd.DataFrame(peak_rows)
    df_pairs = pd.DataFrame(pair_rows)
    df_bounds = pd.DataFrame(bound_rows)
    df_infinite = pd.DataFrame(infinite_rows)
    df_summary = summarize_trends(df_peaks, df_pairs, df_infinite)
    if not df_peaks.empty:
        peak_cols = [
            'Qalpha', 'alpha', 'L', 'V0_abs', 'funcion', 'peak_order', 'k_peak', 'x_peak',
            'q_peak', 'xq_peak', 'y_peak', 'prominence', 'baseline_en_pico', 'width_method',
            'width_k_halfprom', 'width_x_halfprom', 'width_k_curv', 'width_x_curv',
            'width_k_area', 'width_x_area', 'left_half_k', 'right_half_k', 'left_half_x', 'right_half_x',
            'left_boundary_k', 'right_boundary_k', 'left_boundary_x', 'right_boundary_x',
            'left_boundary_y', 'right_boundary_y',
            'left_min_k_refined', 'right_min_k_refined', 'left_min_x_refined', 'right_min_x_refined',
            'left_min_y_refined', 'right_min_y_refined'
        ]
        df_peaks = df_peaks[peak_cols].sort_values(['Qalpha', 'L', 'funcion', 'peak_order'])
        df_peaks.to_csv(output_dir / 'picos_por_funcion.csv', index=False)
    if not df_pairs.empty:
        df_pairs = df_pairs.sort_values(['Qalpha', 'L', 'funcion', 'peak_n'])
        df_pairs.to_csv(output_dir / 'comparacion_consecutiva_picos.csv', index=False)
        plot_spacing_and_width(df_pairs)
    if not df_bounds.empty:
        bound_cols = ['Qalpha', 'alpha', 'L', 'V0_abs', 'bound_order', 'xq_bound', 'q_bound', 'chiL_bound', 'chi_bound', 'k_bound_imag', 'E_bound']
        df_bounds = df_bounds[bound_cols].sort_values(['Qalpha', 'L', 'bound_order'])
        df_bounds.to_csv(output_dir / 'estados_acotados.csv', index=False)
    if not df_infinite.empty:
        df_infinite = df_infinite.sort_values(['Qalpha', 'L', 'tipo', 'funcion', 'order_local'])
        df_infinite.to_csv(output_dir / 'comparacion_pozo_infinito.csv', index=False)
    if not df_summary.empty:
        df_summary = df_summary.sort_values(['Qalpha', 'L', 'funcion'])
        df_summary.to_csv(output_dir / 'resumen_tendencias.csv', index=False)
    print('\nArchivos guardados en:')
    print(output_dir.resolve())


if __name__ == '__main__':
    warnings.filterwarnings('ignore', category=RuntimeWarning)
    main()