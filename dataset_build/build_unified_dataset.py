"""
Build unified training dataset combining table exact values + calibrated digitized data.
"""
import csv, os, numpy as np
from collections import OrderedDict

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(HERE, 'build')
os.makedirs(OUT_DIR, exist_ok=True)

# ============================================================
# 1. TABLE VALUES (EXACT)
# ============================================================
tables = OrderedDict()

# Table 3-2: NiCrC flow series @ -200V bias
tables['NiCrC_flow'] = [
    (0,   1.38, 1.38, 0.79, 120.60,  0.02, 110.60, 3.58),
    (20,  0.09, 0.09, 0.31,  35.21, -0.17,  20.41, 2.60),
    (40,  0.02, 0.02, 0.03,   5.01, -0.61,  14.08, 2.06),
    (60,  0.12, 0.12, 0.35,  34.12, -0.22,  26.30, 2.70),
    (80,  0.25, 0.25, 0.48,  40.21, -0.16,  40.21, 2.96),
    (100, 0.34, 0.34, 0.51,  87.50,  0.14,  60.00, 3.02),
]

# Table 3-3: NiCrC bias series @ 40 sccm
tables['NiCrC_bias'] = [
    (50,   0.32, 0.32, 0.65,  40.21, -0.10, 40.11, 2.86),
    (100,  0.19, 0.19, 0.55,  31.23, -0.18, 32.12, 2.70),
    (200,  0.02, 0.02, 0.03,   5.01, -0.61, 10.53, 2.04),
    (400,  0.43, 0.43, 0.70,  30.12, -0.11, 47.62, 3.12),
    (600,  0.87, 0.87, 0.77, 110.22,  0.01, 76.92, 3.30),
    (800,  1.35, 1.35, 0.80, 112.54,  0.02, 66.67, 3.46),
]

# Table 3-4: NiCrC time series @ 40 sccm, -200V
tables['NiCrC_time'] = [
    (10,    1.11, 1.11, 0.90,  97.21, 62.50, 2.96),
    (30,    0.83, 0.83, 0.85,  61.22, 58.82, 2.86),
    (60,    0.64, 0.64, 0.78,  57.50, 43.48, 2.78),
    (300,   0.08, 0.08, 0.06,  12.50, 18.87, 2.10),
    (600,   0.05, 0.05, 0.05,  10.00, 15.38, 2.08),
    (1800,  0.04, 0.04, 0.04,   7.51, 17.54, 2.06),
    (3600,  0.02, 0.02, 0.02,   5.02, 11.49, 2.01),
]

# Table 4-3: HEA time series
tables['HEA_time'] = [
    (30,    0.68, 0.68, 73.00, 0.58, 80.12, 2.96),
    (60,    0.45, 0.45, 49.00, 0.53, 55.12, 2.83),
    (300,   0.17, 0.17, 23.00, 0.38, 30.12, 2.45),
    (600,   0.09, 0.09, 15.00, 0.08, 14.12, 2.21),
    (1800,  0.06, 0.06, 14.12, 0.05, 15.12, 2.01),
    (3600,  0.05, 0.05, 12.50, 0.03, 13.12, 1.97),
]

# Mechanical properties from text (sparse but exact)
tables['NiCrC_flow_mech'] = [
    (0,   12.41, 190.3, None),
    (20,  12.45, None,  None),
    (40,  None,  None,  2.69),
    (60,  18.7,  None,  None),
    (80,  18.7,  None,  None),
    (100, 20.31, 205.5, 7.95),
]

tables['NiCrC_bias_mech'] = [
    (50,   9.23, 183.2,  None),
    (200,  15.22, 192.2, 4.77),
    (800,  10.23, 179.56, 3.43),
]

tables['HEA_flow_mech'] = [
    (0,   12.02, 186.6, None),
    (60,  None,  None,  1.23),
    (100, 20.31, 224.9, None),
]

tables['HEA_bias_mech'] = [
    (50,   None,  191.1, None),
    (400,  18.21, 198.5, None),
    (800,  15.05, 192.1, None),
]

# Write table-only datasets
for name, data in tables.items():
    out = os.path.join(OUT_DIR, f'{name}_table.csv')
    with open(out, 'w', encoding='utf-8-sig', newline='') as f:
        w = csv.writer(f)
        if name == 'NiCrC_flow':
            w.writerow(['material','series','param','Sa_nm','W_nm','alpha','xi_nm','beta','kcutoff_inv_nm','gamma'])
            for r in data:
                w.writerow(['NiCrC', 'flow', r[0], r[1], r[2], r[3], r[4], r[5], r[6], r[7]])
        elif name == 'NiCrC_bias':
            w.writerow(['material','series','param','Sa_nm','W_nm','alpha','xi_nm','beta','kcutoff_inv_nm','gamma'])
            for r in data:
                w.writerow(['NiCrC', 'bias', r[0], r[1], r[2], r[3], r[4], r[5], r[6], r[7]])
        elif name == 'NiCrC_time':
            w.writerow(['material','series','param','Sa_nm','W_nm','alpha','xi_nm','kcutoff_inv_nm','gamma'])
            for r in data:
                w.writerow(['NiCrC', 'time', r[0], r[1], r[2], r[3], r[4], r[5], r[6]])
        elif name == 'HEA_time':
            w.writerow(['material','series','param','Sa_nm','W_nm','xi_nm','alpha','kcutoff_inv_nm','gamma'])
            for r in data:
                w.writerow(['HEA', 'time', r[0], r[1], r[2], r[3], r[4], r[5], r[6]])
        elif name in ('NiCrC_flow_mech',):
            w.writerow(['material','series','param','H_GPa','E_GPa','stress_GPa'])
            for r in data:
                w.writerow(['NiCrC', 'flow', r[0], r[1], r[2], r[3]])
        elif name in ('NiCrC_bias_mech',):
            w.writerow(['material','series','param','H_GPa','E_GPa','stress_GPa'])
            for r in data:
                w.writerow(['NiCrC', 'bias', r[0], r[1], r[2], r[3]])
        elif name in ('HEA_flow_mech',):
            w.writerow(['material','series','param','H_GPa','E_GPa','stress_GPa'])
            for r in data:
                w.writerow(['HEA', 'flow', r[0], r[1], r[2], r[3]])
        elif name in ('HEA_bias_mech',):
            w.writerow(['material','series','param','H_GPa','E_GPa','stress_GPa'])
            for r in data:
                w.writerow(['HEA', 'bias', r[0], r[1], r[2], r[3]])
    print(f'  {name}_table.csv')

# ============================================================
# 2. DIGITIZED DATA - calibration check
# ============================================================
# Compare digitized values against known table/text references

digi_dir = os.path.join(HERE, 'thesis_extracted_data')

print('\nComparing digitized vs reference for available points:')

# NiCrC flow hardness: table says F100(100sccm) = 20.31 GPa
digi_file = os.path.join(digi_dir, 'NiCrC_mech_vs_flow.csv')
if os.path.exists(digi_file):
    with open(digi_file, 'r') as f:
        rows = list(csv.reader(f))
    xs = np.array([float(r[0]) for r in rows[1:]])
    ys = np.array([float(r[1]) for r in rows[1:]])
    # Find points near x=100
    near = ys[np.abs(xs - 100) < 5]
    if len(near) > 0:
        print(f'  NiCrC mech_vs_flow @100sccm: digitized H={np.mean(near):.2f} GPa (table: 20.31)')
    # Also near x=0 (F0 should be 12.41)
    near0 = ys[np.abs(xs - 0) < 5]
    if len(near0) > 0:
        print(f'  NiCrC mech_vs_flow @0sccm: digitized H={np.mean(near0):.2f} GPa (table: 12.41)')

# NiCrC bias hardness: V200(200V) = 15.22 GPa
digi_file = os.path.join(digi_dir, 'NiCrC_Hardness_vs_bias.csv')
if os.path.exists(digi_file):
    with open(digi_file, 'r') as f:
        rows = list(csv.reader(f))
    xs = np.array([float(r[0]) for r in rows[1:]])
    ys = np.array([float(r[1]) for r in rows[1:]])
    near = ys[np.abs(xs - 200) < 10]
    if len(near) > 0:
        print(f'  NiCrC Hardness vs bias @200V: digitized H={np.mean(near):.2f} GPa (table: 15.22)')
    near = ys[np.abs(xs - 50) < 10]
    if len(near) > 0:
        print(f'  NiCrC Hardness vs bias @50V: digitized H={np.mean(near):.2f} GPa (table: 9.23)')
    near = ys[np.abs(xs - 800) < 20]
    if len(near) > 0:
        print(f'  NiCrC Hardness vs bias @800V: digitized H={np.mean(near):.2f} GPa (table: 10.23)')

# HEA bias - Sa values
digi_file = os.path.join(digi_dir, 'HEA_Sa_vs_bias.csv')
if os.path.exists(digi_file):
    with open(digi_file, 'r') as f:
        rows = list(csv.reader(f))
    xs = np.array([float(r[0]) for r in rows[1:]])
    ys = np.array([float(r[1]) for r in rows[1:]])

# HEA mech vs bias
digi_file = os.path.join(digi_dir, 'HEA_mech_vs_bias.csv')
if os.path.exists(digi_file):
    with open(digi_file, 'r') as f:
        rows = list(csv.reader(f))
    xs = np.array([float(r[0]) for r in rows[1:]])
    ys = np.array([float(r[1]) for r in rows[1:]])
    near = ys[np.abs(xs - 400) < 15]
    if len(near) > 0:
        print(f'  HEA mech vs bias @400V: digitized H={np.mean(near):.2f} GPa (table: 18.21)')

# Sa values from table for calibration check
# NiCrC flow: table says F40(40sccm)=0.02
# But we don't have good digitized Sa for flow... 

# Let me check the direct-rendered/canny approach output
digi_file = os.path.join(HERE, 'thesis_extracted_data', 'NiCrC_Sa_vs_flow_calib2.csv')
if os.path.exists(digi_file):
    with open(digi_file, 'r') as f:
        rows = list(csv.reader(f))
    if len(rows) > 1:
        xs = np.array([float(r[0]) for r in rows[1:]])
        ys = np.array([float(r[1]) for r in rows[1:]])
        # Check near x=40
        near = ys[np.abs(xs - 40) < 3]
        if len(near) > 0:
            print(f'  Sa_vs_flow @40sccm (direct+canny): Sa={np.mean(near):.4f} nm (table: 0.02)')
        near0 = ys[np.abs(xs - 0) < 3]
        if len(near0) > 0:
            print(f'  Sa_vs_flow @0sccm (direct+canny): Sa={np.mean(near0):.4f} nm (table: 1.38)')
        near100 = ys[np.abs(xs - 100) < 3]
        if len(near100) > 0:
            print(f'  Sa_vs_flow @100sccm (direct+canny): Sa={np.mean(near100):.4f} nm (table: 0.34)')

# ============================================================
# 3. BUILD UNIFIED DATASET
# ============================================================
print('\n--- Building unified dataset ---')

# For ML training, we want: [material_type, series, param_value, property_name, property_value]
# where param_value = C2H2 flow (sccm) OR bias (V) OR time (s)

# Start with exact table data
all_rows = []
# Column headers: material, series, param, C2H2_sccm, bias_V, time_s, Sa_nm, H_GPa, E_GPa, stress_GPa, ...

# Write merged CSV with all available data
with open(os.path.join(OUT_DIR, 'NiCrC_master.csv'), 'w', newline='') as f:
    w = csv.writer(f)
    w.writerow(['material','series','C2H2_sccm','bias_V','time_s','Sa_nm','H_GPa','E_GPa','stress_GPa','H_Estar'])
    # Flow series: entries from table 3-2 + mechanical
    for (flow, sa, wv, alpha, xi, beta, kcut, gamma), (_, h, e, stress) in zip(
        tables['NiCrC_flow'], tables['NiCrC_flow_mech']):
        w.writerow(['NiCrC','flow',flow,200,None,sa,h,e,stress,None])
    # Bias series
    for (bias, sa, wv, alpha, xi, beta, kcut, gamma), (_, h, e, stress) in zip(
        tables['NiCrC_bias'], tables['NiCrC_bias_mech']):
        w.writerow(['NiCrC','bias',40,bias,None,sa,h,e,stress,None])
    # Time series
    for (t, sa, wv, alpha, xi, kcut, gamma) in tables['NiCrC_time']:
        w.writerow(['NiCrC','time',40,200,t,sa,None,None,None,None])

with open(os.path.join(OUT_DIR, 'HEA_master.csv'), 'w', newline='') as f:
    w = csv.writer(f)
    w.writerow(['material','series','C2H2_sccm','bias_V','time_s','Sa_nm','H_GPa','E_GPa','stress_GPa'])
    # Flow series from text
    for (flow, h, e, stress) in tables['HEA_flow_mech']:
        w.writerow(['HEA','flow',flow,None,None,None,h,e,stress])
    # Bias series from text
    for (bias, h, e, stress) in tables['HEA_bias_mech']:
        w.writerow(['HEA','bias',None,bias,None,None,h,e,stress])
    # Time series from table 4-3
    for (t, sa, wv, xi, alpha, kcut, gamma) in tables['HEA_time']:
        w.writerow(['HEA','time',None,None,t,sa,None,None,None])

print('  NiCrC_master.csv - exact table data')
print('  HEA_master.csv - exact table data')

# ============================================================
# 4. COMBINE WITH DIGITIZED DATA
# ============================================================
print('\n--- Merging digitized mechanical data ---')

# Load digitized CSVs and merge with master
# For each digitized series, take the dense x,y points and add them to the master

digitized_sources = {
    # (filename, material, series, input_param, output_property, x_units)
    'NiCrC_Hardness_vs_bias': ('NiCrC', 'bias', 'bias_V', 'H_GPa', 'V'),
    'NiCrC_HEstar_vs_bias': ('NiCrC', 'bias', 'bias_V', 'H_Estar', 'V'),
    'NiCrC_Stress_vs_bias': ('NiCrC', 'bias', 'bias_V', 'stress_GPa', 'V'),
    'NiCrC_mech_vs_flow': ('NiCrC', 'flow', 'C2H2_sccm', 'H_GPa', 'sccm'),
    'HEA_mech_vs_bias': ('HEA', 'bias', 'bias_V', 'H_GPa', 'V'),
    'HEA_Sa_vs_bias': ('HEA', 'bias', 'bias_V', 'Sa_nm', 'V'),
    'HEA_Sa_vs_flow': ('HEA', 'flow', 'C2H2_sccm', 'Sa_nm', 'sccm'),
    'HEA_Sa_vs_time': ('HEA', 'time', 'time_s', 'Sa_nm', 's'),
}

# Also check for _calib versions
for prefix, info in digitized_sources.items():
    for suffix in ['', '_calib', '_calib2']:
        path = os.path.join(digi_dir, f'{prefix}{suffix}.csv')
        if os.path.exists(path):
            with open(path, 'r') as f:
                rows = list(csv.reader(f))
            if len(rows) <= 1:
                continue
            xs = np.array([float(r[0]) for r in rows[1:]])
            ys = np.array([float(r[1]) for r in rows[1:]])
            if len(xs) < 10:
                continue
            mat, series, iparam, oprop, xunit = info
            
            # Save to individual calibrated CSV
            cal_path = os.path.join(OUT_DIR, f'{prefix}_merged.csv')
            with open(cal_path, 'w', encoding='utf-8-sig', newline='') as f:
                w = csv.writer(f)
                w.writerow(['x', 'y', 'x_unit', 'y_property', 'material', 'series'])
                for x, y in zip(xs, ys):
                    w.writerow([f'{x:.4f}', f'{y:.4f}', xunit, oprop, mat, series])
            
            print(f'  {prefix}: {len(xs)} pts ({oprop} vs {iparam})')
            break

# ============================================================
# 5. COMBINED MASTER DATASET
# ============================================================
print('\n--- Building combined master dataset ---')

# Create one unified CSV with ALL data points
all_rows = []
all_rows.append(['material','C2H2_sccm','bias_V','time_s','Sa_nm','H_GPa','E_GPa','stress_GPa','H_Estar','source'])

# Add table data
for (flow, sa, wv, alpha, xi, beta, kcut, gamma), (_, h, e, stress) in zip(
    tables['NiCrC_flow'], tables['NiCrC_flow_mech']):
    all_rows.append(['NiCrC', flow, 200, None, sa, h, e, stress, None, 'table'])

for (bias, sa, wv, alpha, xi, beta, kcut, gamma), (_, h, e, stress) in zip(
    tables['NiCrC_bias'], tables['NiCrC_bias_mech']):
    all_rows.append(['NiCrC', 40, bias, None, sa, h, e, stress, None, 'table'])

for (t, sa, wv, alpha, xi, kcut, gamma) in tables['NiCrC_time']:
    all_rows.append(['NiCrC', 40, 200, t, sa, None, None, None, None, 'table'])

for (flow, h, e, stress) in tables['HEA_flow_mech']:
    all_rows.append(['HEA', flow, None, None, None, h, e, stress, None, 'table'])

for (bias, h, e, stress) in tables['HEA_bias_mech']:
    all_rows.append(['HEA', None, bias, None, None, h, e, stress, None, 'table'])

for (t, sa, wv, xi, alpha, kcut, gamma) in tables['HEA_time']:
    all_rows.append(['HEA', None, None, t, sa, None, None, None, None, 'table'])

# Add digitized data (dense points from figures)
for prefix, info in digitized_sources.items():
    for suffix in ['', '_calib', '_calib2']:
        path = os.path.join(digi_dir, f'{prefix}{suffix}.csv')
        if os.path.exists(path):
            with open(path, 'r') as f:
                rows = list(csv.reader(f))
            if len(rows) <= 1:
                continue
            xs = np.array([float(r[0]) for r in rows[1:]])
            ys = np.array([float(r[1]) for r in rows[1:]])
            if len(xs) < 10:
                continue
            mat, series, iparam, oprop, xunit = info
            
            # Subsample to avoid excessive density (keep ~50 pts per curve)
            n_keep = min(len(xs), 100)
            idx = np.linspace(0, len(xs)-1, n_keep, dtype=int)
            
            for i in idx:
                row = [mat, None, None, None, None, None, None, None, None, f'digi_{prefix}']
                if series == 'flow':
                    row[1] = f'{xs[i]:.1f}'
                    row[2] = 200  # fixed bias
                elif series == 'bias':
                    row[1] = 40  # fixed flow
                    row[2] = f'{xs[i]:.1f}'
                elif series == 'time':
                    row[1] = 40
                    row[2] = 200
                    row[3] = f'{xs[i]:.1f}'
                
                # Map property
                if oprop == 'Sa_nm':
                    row[4] = f'{ys[i]:.4f}'
                elif oprop == 'H_GPa':
                    row[5] = f'{ys[i]:.4f}'
                elif oprop == 'E_GPa':
                    row[6] = f'{ys[i]:.4f}'
                elif oprop == 'stress_GPa':
                    row[7] = f'{ys[i]:.4f}'
                elif oprop == 'H_Estar':
                    row[8] = f'{ys[i]:.6f}'
                
                all_rows.append(row)
            break

# Write combined
with open(os.path.join(OUT_DIR, 'combined_master.csv'), 'w', encoding='utf-8-sig', newline='') as f:
    w = csv.writer(f)
    for row in all_rows:
        w.writerow(row)

print(f'\nCombined dataset: {len(all_rows)-1} data points')
print('  -> dataset_build/build/combined_master.csv')
print(f'\nFiles created in {OUT_DIR}/:')
for f in sorted(os.listdir(OUT_DIR)):
    print(f'  {f}')
