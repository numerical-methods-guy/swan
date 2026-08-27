#!/usr/bin/env python3
import argparse
import csv
import math
from pathlib import Path
import sys

try:
    import yaml
except ImportError:
    print('This script requires PyYAML: pip install pyyaml', file=sys.stderr)
    raise


def load_cfg(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def get_in(d, *path, default=None):
    cur = d
    for key in path:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def first_present(cfg, candidates, default=None):
    for cand in candidates:
        if isinstance(cand, tuple):
            val = get_in(cfg, *cand, default=None)
        else:
            val = cfg.get(cand, None) if isinstance(cfg, dict) else None
        if val is not None:
            return val
    return default


def load_geometry(cfg):
    lrnlat = int(first_present(cfg, [('data', 'nlat'), 'nlat']))
    lrnlon = int(first_present(cfg, [('data', 'nlon'), 'nlon']))
    lam = first_present(cfg, [('lam',), 'lam'], default={})
    if isinstance(lam, tuple):
        lam = lam[0]
    if not isinstance(lam, dict):
        lam = {}

    pL = int(first_present(lam, ['patch_nlat_lr', 'patchnlatlr'], default=first_present(cfg, ['patch_nlat_lr', 'patchnlatlr'])))
    pN = int(first_present(lam, ['patch_nlon_lr', 'patchnlonlr'], default=first_present(cfg, ['patch_nlon_lr', 'patchnlonlr'])))
    R = int(first_present(lam, ['halo_radius', 'haloradius'], default=first_present(cfg, ['halo_radius', 'haloradius'])))
    ex = int(first_present(lam, ['exclude_pole_rows', 'excludepolerows'], default=first_present(cfg, ['exclude_pole_rows', 'excludepolerows'], default=4)))
    s = int(first_present(lam, ['refinement_factor_lat', 'refinementfactorlat'], default=first_present(cfg, ['refinement_factor_lat', 'refinementfactorlat'])))
    return dict(lrnlat=lrnlat, lrnlon=lrnlon, pL=pL, pN=pN, R=R, ex=ex, s=s)


def load_rotation(cfg):
    enabled = first_present(
        cfg,
        [
            ('rotation', 'enabled'),
            'rotation_enabled',
            'rotationenabled',
        ],
        default=False,
    )
    pole_lat = first_present(
        cfg,
        [
            ('rotation', 'pole_target_lat_deg'),
            ('rotation', 'poletargetlatdeg'),
            'rotation_pole_target_lat_deg',
            'rotationpoletargetlatdeg',
        ],
        default=0.0,
    )
    pole_lon = first_present(
        cfg,
        [
            ('rotation', 'pole_target_lon_deg'),
            ('rotation', 'poletargetlondeg'),
            'rotation_pole_target_lon_deg',
            'rotationpoletargetlondeg',
        ],
        default=0.0,
    )
    return dict(enabled=bool(enabled), pole_lat=float(pole_lat), pole_lon=float(pole_lon))


def deg2rad(x):
    return x * math.pi / 180.0


def rad2deg(x):
    return x * 180.0 / math.pi


def latlon_to_vec(lat_deg, lon_deg):
    lat = deg2rad(lat_deg)
    lon = deg2rad(lon_deg)
    clat = math.cos(lat)
    return [clat * math.cos(lon), clat * math.sin(lon), math.sin(lat)]


def vec_to_latlon(v):
    x, y, z = v
    lon = math.atan2(y, x)
    hyp = math.hypot(x, y)
    lat = math.atan2(z, hyp)
    lon_deg = rad2deg(lon) % 360.0
    lat_deg = rad2deg(lat)
    return lat_deg, lon_deg


def dot(a, b):
    return a[0]*b[0] + a[1]*b[1] + a[2]*b[2]


def cross(a, b):
    return [
        a[1]*b[2] - a[2]*b[1],
        a[2]*b[0] - a[0]*b[2],
        a[0]*b[1] - a[1]*b[0],
    ]


def norm(v):
    return math.sqrt(dot(v, v))


def normalize(v):
    n = norm(v)
    if n == 0:
        return [0.0, 0.0, 0.0]
    return [v[0]/n, v[1]/n, v[2]/n]


def matmul_vec(M, v):
    return [
        M[0][0]*v[0] + M[0][1]*v[1] + M[0][2]*v[2],
        M[1][0]*v[0] + M[1][1]*v[1] + M[1][2]*v[2],
        M[2][0]*v[0] + M[2][1]*v[1] + M[2][2]*v[2],
    ]


def rotation_matrix_from_north_pole_target(target_lat_deg, target_lon_deg):
    k = [0.0, 0.0, 1.0]
    t = normalize(latlon_to_vec(target_lat_deg, target_lon_deg))
    c = max(-1.0, min(1.0, dot(k, t)))

    if abs(c - 1.0) < 1e-12:
        return [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]

    if abs(c + 1.0) < 1e-12:
        return [[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]]

    v = cross(k, t)
    s = norm(v)
    vx, vy, vz = v
    K = [[0.0, -vz, vy], [vz, 0.0, -vx], [-vy, vx, 0.0]]
    K2 = [
        [K[0][0]*K[0][0] + K[0][1]*K[1][0] + K[0][2]*K[2][0], K[0][0]*K[0][1] + K[0][1]*K[1][1] + K[0][2]*K[2][1], K[0][0]*K[0][2] + K[0][1]*K[1][2] + K[0][2]*K[2][2]],
        [K[1][0]*K[0][0] + K[1][1]*K[1][0] + K[1][2]*K[2][0], K[1][0]*K[0][1] + K[1][1]*K[1][1] + K[1][2]*K[2][1], K[1][0]*K[0][2] + K[1][1]*K[1][2] + K[1][2]*K[2][2]],
        [K[2][0]*K[0][0] + K[2][1]*K[1][0] + K[2][2]*K[2][0], K[2][0]*K[0][1] + K[2][1]*K[1][1] + K[2][2]*K[2][1], K[2][0]*K[0][2] + K[2][1]*K[1][2] + K[2][2]*K[2][2]],
    ]
    factor = (1.0 - c) / (s * s)
    I = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    R = [[I[i][j] + K[i][j] + factor * K2[i][j] for j in range(3)] for i in range(3)]
    return R


def rotate_original_to_rotated_frame(lat_deg, lon_deg, pole_target_lat_deg, pole_target_lon_deg):
    R = rotation_matrix_from_north_pole_target(pole_target_lat_deg, pole_target_lon_deg)
    v = latlon_to_vec(lat_deg, lon_deg)
    v_rot = matmul_vec(R, v)
    return vec_to_latlon(v_rot)


def nearest_lat_row(lat_deg, nlat):
    dlat = 180.0 / nlat
    centers = [90.0 - (i + 0.5) * dlat for i in range(nlat)]
    return min(range(nlat), key=lambda i: abs(centers[i] - lat_deg))


def lon_ang_dist(a, b):
    return abs((a - b + 180.0) % 360.0 - 180.0)


def nearest_lon_col(lon_deg, nlon):
    dlon = 360.0 / nlon
    lon_deg = lon_deg % 360.0
    centers = [((j + 0.5) * dlon) % 360.0 for j in range(nlon)]
    return min(range(nlon), key=lambda j: lon_ang_dist(centers[j], lon_deg))


def lat_lon_bounds(lat0, lon0, pL, pN, lrnlat, lrnlon):
    dlat = 180.0 / lrnlat
    dlon = 360.0 / lrnlon
    lat_top = 90.0 - lat0 * dlat
    lat_bot = 90.0 - (lat0 + pL) * dlat
    lon_left = (lon0 * dlon) % 360.0
    lon_right = ((lon0 + pN) * dlon) % 360.0
    return lat_top, lat_bot, lon_left, lon_right


def build_table(geom):
    lrnlat, lrnlon, pL, pN, R, ex, s = geom['lrnlat'], geom['lrnlon'], geom['pL'], geom['pN'], geom['R'], geom['ex'], geom['s']
    latmin = ex + R
    latmax = lrnlat - ex - R - pL
    if latmin > latmax:
        raise ValueError(f'No valid patches: latmin={latmin} > latmax={latmax}')
    lat_starts = list(range(latmin, latmax + 1, pL))
    lon_starts = list(range(0, lrnlon, pN))
    rows = []
    for lat0 in lat_starts:
        for lon0 in lon_starts:
            lat_band = (lat0 - latmin) // pL
            lon_band = lon0 // pN
            patch_idx = lat_band * len(lon_starts) + lon_band
            lat_top, lat_bot, lon_left, lon_right = lat_lon_bounds(lat0, lon0, pL, pN, lrnlat, lrnlon)
            rows.append({
                'patch_idx': patch_idx,
                'lat_band': lat_band,
                'lon_band': lon_band,
                'lat0_lr': lat0,
                'lon0_lr': lon0,
                'lat_top_deg': round(lat_top, 6),
                'lat_bot_deg': round(lat_bot, 6),
                'lon_left_deg': round(lon_left, 6),
                'lon_right_deg': round(lon_right, 6),
                'hr_lat0': lat0 * s,
                'hr_lon0': lon0 * s,
                'lr_patch_shape': f'{pL}x{pN}',
                'lr_window_shape': f'{pL + 2*R}x{pN + 2*R}',
                'hr_patch_shape': f'{pL * s}x{pN * s}',
            })
    meta = {
        'latmin': latmin,
        'latmax': latmax,
        'lat_starts': lat_starts,
        'lon_starts': lon_starts,
        'patches_per_row': len(lon_starts),
        'num_patch_rows': len(lat_starts),
        'total_patches_per_ic': len(rows),
    }
    return rows, meta


def containing_manifest_patch(row, col, geom, meta):
    pL, pN, latmin, latmax = geom['pL'], geom['pN'], meta['latmin'], meta['latmax']
    covered = (row >= latmin) and (row <= latmax + pL - 1)
    lat0 = latmin + ((max(row, latmin) - latmin) // pL) * pL
    lat0 = max(latmin, min(lat0, latmax))
    lon0 = (col // pN) * pN
    lat_band = (lat0 - latmin) // pL
    lon_band = lon0 // pN
    patch_idx = lat_band * meta['patches_per_row'] + lon_band
    return lat0, lon0, patch_idx, covered


def write_csv(rows, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main():
    ap = argparse.ArgumentParser(description='Generate LAM patch lookup table, with optional rotation-aware point lookup.')
    ap.add_argument('--config', required=True, help='Path to config YAML')
    ap.add_argument('--output_csv', default='lam_patch_lookup.csv', help='Output CSV path')
    ap.add_argument('--print_rows', type=int, default=12)
    ap.add_argument('--query_lat', type=float, default=None, help='Latitude of point of interest')
    ap.add_argument('--query_lon', type=float, default=None, help='Longitude of point of interest')
    ap.add_argument('--query_frame', choices=['original', 'rotated'], default='original', help='Frame for query coordinates')
    ap.add_argument('--rotation', choices=['auto', 'on', 'off'], default='auto', help='Use config rotation, force on, or force off')
    ap.add_argument('--pole_target_lat_deg', type=float, default=None, help='Override rotation pole target latitude')
    ap.add_argument('--pole_target_lon_deg', type=float, default=None, help='Override rotation pole target longitude')
    args = ap.parse_args()

    if (args.query_lat is None) ^ (args.query_lon is None):
        raise ValueError('Provide both --query_lat and --query_lon, or omit both.')

    cfg = load_cfg(args.config)
    geom = load_geometry(cfg)
    rot_cfg = load_rotation(cfg)

    pole_lat = args.pole_target_lat_deg if args.pole_target_lat_deg is not None else rot_cfg['pole_lat']
    pole_lon = args.pole_target_lon_deg if args.pole_target_lon_deg is not None else rot_cfg['pole_lon']
    rotation_enabled = rot_cfg['enabled'] if args.rotation == 'auto' else (args.rotation == 'on')

    rows, meta = build_table(geom)
    write_csv(rows, args.output_csv)

    print('LAM patch lookup table')
    print(f"LR grid: {geom['lrnlat']} x {geom['lrnlon']}")
    print(f"Patch LR: {geom['pL']} x {geom['pN']}")
    print(f"Halo radius: {geom['R']}")
    print(f"Exclude pole rows: {geom['ex']}")
    print(f"Refinement: {geom['s']}x")
    print(f"Rotation enabled for query: {rotation_enabled}")
    print(f"Rotation pole target (deg): ({pole_lat}, {pole_lon})")
    print(f"Patches per IC: {meta['total_patches_per_ic']}")
    print(f"Saved CSV to {args.output_csv}")

    if rows:
        print('')
        preview = rows[:args.print_rows]
        headers = list(preview[0].keys())
        widths = {h: max(len(h), max(len(str(r[h])) for r in preview)) for h in headers}
        print(' | '.join(h.ljust(widths[h]) for h in headers))
        print('-+-'.join('-' * widths[h] for h in headers))
        for r in preview:
            print(' | '.join(str(r[h]).ljust(widths[h]) for h in headers))

    if args.query_lat is not None:
        qlat, qlon = args.query_lat, args.query_lon % 360.0
        if rotation_enabled and args.query_frame == 'original':
            rot_lat, rot_lon = rotate_original_to_rotated_frame(qlat, qlon, pole_lat, pole_lon)
        else:
            rot_lat, rot_lon = qlat, qlon

        row = nearest_lat_row(rot_lat, geom['lrnlat'])
        col = nearest_lon_col(rot_lon, geom['lrnlon'])
        lat0, lon0, patch_idx, covered = containing_manifest_patch(row, col, geom, meta)
        lat_top, lat_bot, lon_left, lon_right = lat_lon_bounds(lat0, lon0, geom['pL'], geom['pN'], geom['lrnlat'], geom['lrnlon'])

        print('\nQuery result')
        print(f'Input point ({args.query_frame} frame): lat={qlat}, lon={qlon}')
        print(f'Rotated-frame point used for lookup: lat={rot_lat:.6f}, lon={rot_lon:.6f}')
        print(f'Nearest LR gridpoint row,col: {row},{col}')
        print(f'Recommended plot_lat0_lr, plot_lon0_lr: {lat0},{lon0}')
        print(f'Patch index: {patch_idx}')
        print(f'Patch lat range: {lat_bot:.6f} to {lat_top:.6f} deg')
        print(f'Patch lon range: {lon_left:.6f} to {lon_right:.6f} deg')
        print(f'Point inside manifest-covered latitude band: {covered}')
        if not covered:
            print('Warning: the query point lies outside the latitude band covered by valid manifest patches.')


if __name__ == '__main__':
    main()
