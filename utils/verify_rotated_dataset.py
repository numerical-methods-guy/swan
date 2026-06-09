import argparse
import json
import h5py
import numpy as np
import yaml


def _check(cond, msg, errors, ok_msgs=None):
    if cond:
        if ok_msgs is not None:
            ok_msgs.append(msg)
    else:
        errors.append(msg)


def _finite(arr):
    return np.isfinite(arr).all()


def _shape_tuple(x):
    return tuple(int(v) for v in x.shape)


def compare_to_reference(h5_path, reference_h5_path, sample_ics=2, expect_different=False, diff_atol=1e-6):
    errors = []
    warnings = []
    ok = []
    metrics = {}

    compare_keys = ['lr/fields', 'lr/winds', 'hr/fields', 'hr/winds']

    with h5py.File(h5_path, 'r') as hf, h5py.File(reference_h5_path, 'r') as ref:
        for a in ['lr_nlat', 'lr_nlon', 'hr_nlat', 'hr_nlon', 'rollout_steps', 'upscale_factor_lat', 'upscale_factor_lon']:
            _check(a in hf.attrs and a in ref.attrs, f"reference attr present in both files: {a}", errors, ok)
        for key in compare_keys:
            _check(key in hf and key in ref, f"reference dataset present in both files: {key}", errors, ok)
        if errors:
            return {'ok': False, 'errors': errors, 'warnings': warnings, 'checks_passed': ok, 'metrics': metrics}

        for a in ['lr_nlat', 'lr_nlon', 'hr_nlat', 'hr_nlon', 'rollout_steps', 'upscale_factor_lat', 'upscale_factor_lon']:
            _check(int(hf.attrs[a]) == int(ref.attrs[a]), f"reference match: {a}", errors, ok)

        for key in compare_keys:
            _check(_shape_tuple(hf[key]) == _shape_tuple(ref[key]), f"reference shape match: {key}", errors, ok)

        if errors:
            return {'ok': False, 'errors': errors, 'warnings': warnings, 'checks_passed': ok, 'metrics': metrics}

        n = min(sample_ics, int(hf.attrs['num_ics']), int(ref.attrs['num_ics']))
        if n < sample_ics:
            warnings.append(f"reference comparison checked only {n} ICs because one dataset is smaller than requested sample_ics={sample_ics}")
        n = max(1, n)

        mean_abs_diffs = {}
        max_abs_diffs = {}
        exact_equal_flags = {}

        for key in compare_keys:
            mad_vals = []
            max_vals = []
            exact_flags = []
            for i in range(n):
                a = np.asarray(hf[key][i], dtype=np.float64)
                b = np.asarray(ref[key][i], dtype=np.float64)
                diff = np.abs(a - b)
                mad_vals.append(float(diff.mean()))
                max_vals.append(float(diff.max()))
                exact_flags.append(bool(np.array_equal(a, b)))
            mean_abs_diffs[key] = float(np.mean(mad_vals))
            max_abs_diffs[key] = float(np.max(max_vals))
            exact_equal_flags[key] = bool(all(exact_flags))
            ok.append(f"reference diff computed: {key}")

        metrics['mean_abs_diff'] = mean_abs_diffs
        metrics['max_abs_diff'] = max_abs_diffs
        metrics['exact_equal'] = exact_equal_flags

        if expect_different:
            changed = any(max_abs_diffs[k] > diff_atol for k in compare_keys)
            _check(changed, f"dataset differs from reference on sampled arrays (atol>{diff_atol})", errors, ok)
            if not changed:
                warnings.append('Compared datasets appear numerically identical on sampled arrays; if you expected a rotated non-identity dataset, check whether rotation was actually enabled/applied.')
        else:
            if all(exact_equal_flags.values()):
                warnings.append('Compared datasets are exactly equal on sampled arrays. This is expected for an identity-rotation test, but unexpected for a non-identity rotation.')

    return {'ok': len(errors) == 0, 'errors': errors, 'warnings': warnings, 'checks_passed': ok, 'metrics': metrics}


def verify_dataset(h5_path, config_path=None, sample_ics=2, check_rotation=False):
    errors = []
    warnings = []
    ok = []

    cfg = None
    if config_path is not None:
        with open(config_path) as f:
            cfg = yaml.safe_load(f)

    required_attrs = [
        'lr_nlat', 'lr_nlon', 'hr_nlat', 'hr_nlon', 'dt', 'dt_solver', 'nsteps',
        'num_ics', 'rollout_steps', 'upscale_factor_lat', 'upscale_factor_lon',
        'ic_type', 'seed_base', 'lr_inp_mean', 'lr_inp_var', 'lr_wind_mean',
        'lr_wind_var', 'hr_inp_mean', 'hr_inp_var', 'hr_wind_mean', 'hr_wind_var'
    ]

    required_datasets = {
        'lr/t0': ('num_ics', 3, 'lr_nlat', 'lr_nlon'),
        'lr/t1': ('num_ics', 3, 'lr_nlat', 'lr_nlon'),
        'lr/w0': ('num_ics', 2, 'lr_nlat', 'lr_nlon'),
        'lr/w1': ('num_ics', 2, 'lr_nlat', 'lr_nlon'),
        'lr/fields': ('num_ics', 'rollout_steps+1', 3, 'lr_nlat', 'lr_nlon'),
        'lr/winds': ('num_ics', 'rollout_steps+1', 2, 'lr_nlat', 'lr_nlon'),
        'hr/t0': ('num_ics', 3, 'hr_nlat', 'hr_nlon'),
        'hr/t1': ('num_ics', 3, 'hr_nlat', 'hr_nlon'),
        'hr/w0': ('num_ics', 2, 'hr_nlat', 'hr_nlon'),
        'hr/w1': ('num_ics', 2, 'hr_nlat', 'hr_nlon'),
        'hr/fields': ('num_ics', 'rollout_steps+1', 3, 'hr_nlat', 'hr_nlon'),
        'hr/winds': ('num_ics', 'rollout_steps+1', 2, 'hr_nlat', 'hr_nlon'),
    }

    with h5py.File(h5_path, 'r') as hf:
        for a in required_attrs:
            _check(a in hf.attrs, f"attr present: {a}", errors, ok)

        if errors:
            return {'ok': False, 'errors': errors, 'warnings': warnings, 'checks_passed': ok}

        attrs = {k: hf.attrs[k] for k in hf.attrs.keys()}
        lr_nlat = int(attrs['lr_nlat'])
        lr_nlon = int(attrs['lr_nlon'])
        hr_nlat = int(attrs['hr_nlat'])
        hr_nlon = int(attrs['hr_nlon'])
        num_ics = int(attrs['num_ics'])
        rollout_steps = int(attrs['rollout_steps'])
        s_lat = int(attrs['upscale_factor_lat'])
        s_lon = int(attrs['upscale_factor_lon'])
        dt = float(attrs['dt'])
        dt_solver = float(attrs['dt_solver'])
        nsteps = int(attrs['nsteps'])

        _check(s_lat == s_lon, f"isotropic upscale factors: {s_lat}", errors, ok)
        _check(hr_nlat == lr_nlat * s_lat, f"hr_nlat == lr_nlat * scale ({hr_nlat})", errors, ok)
        _check(hr_nlon == lr_nlon * s_lon, f"hr_nlon == lr_nlon * scale ({hr_nlon})", errors, ok)
        _check(nsteps >= 1, f"nsteps >= 1 ({nsteps})", errors, ok)
        _check(abs(round(dt / dt_solver) - nsteps) < 1e-9, f"nsteps matches round(dt/dt_solver) ({nsteps})", errors, ok)
        _check(num_ics > 0, f"num_ics > 0 ({num_ics})", errors, ok)
        _check(rollout_steps >= 1, f"rollout_steps >= 1 ({rollout_steps})", errors, ok)

        for key, expected in required_datasets.items():
            _check(key in hf, f"dataset present: {key}", errors, ok)
        if errors:
            return {'ok': False, 'errors': errors, 'warnings': warnings, 'checks_passed': ok}

        def resolve_dim(x):
            if x == 'num_ics':
                return num_ics
            if x == 'rollout_steps+1':
                return rollout_steps + 1
            if x == 'lr_nlat':
                return lr_nlat
            if x == 'lr_nlon':
                return lr_nlon
            if x == 'hr_nlat':
                return hr_nlat
            if x == 'hr_nlon':
                return hr_nlon
            return x

        for key, expected in required_datasets.items():
            actual = _shape_tuple(hf[key])
            exp = tuple(resolve_dim(x) for x in expected)
            _check(actual == exp, f"shape ok: {key} = {actual}", errors, ok)
            _check(str(hf[key].dtype) == 'float32', f"dtype ok: {key} = float32", errors, ok)

        stat_specs = {
            'lr_inp_mean': 3, 'lr_inp_var': 3, 'lr_wind_mean': 2, 'lr_wind_var': 2,
            'hr_inp_mean': 3, 'hr_inp_var': 3, 'hr_wind_mean': 2, 'hr_wind_var': 2,
        }
        for k, n in stat_specs.items():
            arr = np.asarray(attrs[k])
            _check(arr.shape == (n,), f"stat shape ok: {k} = {(n,)}", errors, ok)
            _check(_finite(arr), f"stat finite: {k}", errors, ok)
            if 'var' in k:
                _check((arr > 0).all(), f"stat positive: {k}", errors, ok)

        sample_ics = max(1, min(sample_ics, num_ics))
        for i in range(sample_ics):
            for key in ['lr/t0', 'lr/t1', 'lr/w0', 'lr/w1', 'lr/fields', 'lr/winds', 'hr/t0', 'hr/t1', 'hr/w0', 'hr/w1', 'hr/fields', 'hr/winds']:
                arr = hf[key][i]
                _check(_finite(arr), f"finite values: {key}[{i}]", errors, ok)

            _check(np.allclose(hf['lr/t0'][i], hf['lr/fields'][i, 0], atol=0, rtol=0), f"lr/t0 matches lr/fields[:,0] for ic {i}", errors, ok)
            _check(np.allclose(hf['lr/t1'][i], hf['lr/fields'][i, 1], atol=0, rtol=0), f"lr/t1 matches lr/fields[:,1] for ic {i}", errors, ok)
            _check(np.allclose(hf['lr/w0'][i], hf['lr/winds'][i, 0], atol=0, rtol=0), f"lr/w0 matches lr/winds[:,0] for ic {i}", errors, ok)
            _check(np.allclose(hf['lr/w1'][i], hf['lr/winds'][i, 1], atol=0, rtol=0), f"lr/w1 matches lr/winds[:,1] for ic {i}", errors, ok)
            _check(np.allclose(hf['hr/t0'][i], hf['hr/fields'][i, 0], atol=0, rtol=0), f"hr/t0 matches hr/fields[:,0] for ic {i}", errors, ok)
            _check(np.allclose(hf['hr/t1'][i], hf['hr/fields'][i, 1], atol=0, rtol=0), f"hr/t1 matches hr/fields[:,1] for ic {i}", errors, ok)
            _check(np.allclose(hf['hr/w0'][i], hf['hr/winds'][i, 0], atol=0, rtol=0), f"hr/w0 matches hr/winds[:,0] for ic {i}", errors, ok)
            _check(np.allclose(hf['hr/w1'][i], hf['hr/winds'][i, 1], atol=0, rtol=0), f"hr/w1 matches hr/winds[:,1] for ic {i}", errors, ok)

        if cfg is not None:
            dc = cfg['data']
            lamc = cfg['lam']
            _check(lr_nlat == int(dc['nlat']), f"config match: lr_nlat == data.nlat ({lr_nlat})", errors, ok)
            _check(lr_nlon == int(dc['nlon']), f"config match: lr_nlon == data.nlon ({lr_nlon})", errors, ok)
            _check(abs(dt - float(dc['dt'])) < 1e-12, f"config match: dt == data.dt ({dt})", errors, ok)
            _check(abs(dt_solver - float(dc['dt_solver'])) < 1e-12, f"config match: dt_solver == data.dt_solver ({dt_solver})", errors, ok)
            _check(s_lat == int(lamc['refinement_factor_lat']), f"config match: upscale_factor_lat ({s_lat})", errors, ok)
            _check(s_lon == int(lamc['refinement_factor_lon']), f"config match: upscale_factor_lon ({s_lon})", errors, ok)

            try:
                from lam_patch_dataset import LAMPatchDataset
                ds = LAMPatchDataset(
                    h5_path=h5_path,
                    patch_nlat_lr=int(lamc['patch_nlat_lr']),
                    patch_nlon_lr=int(lamc['patch_nlon_lr']),
                    halo_radius=int(lamc['halo_radius']),
                    exclude_pole_rows=int(lamc.get('exclude_pole_rows', 4)),
                    split='train',
                    normalize=True,
                    preload=False,
                )
                _check(len(ds) > 0, f"LAMPatchDataset builds successfully with {len(ds)} patches", errors, ok)
                sample = ds[0]
                expected_lr = (5, int(lamc['patch_nlat_lr']) + 2 * int(lamc['halo_radius']), int(lamc['patch_nlon_lr']) + 2 * int(lamc['halo_radius']))
                expected_hr0 = (5, int(lamc['patch_nlat_lr']) * s_lat, int(lamc['patch_nlon_lr']) * s_lon)
                expected_hr1 = (3, int(lamc['patch_nlat_lr']) * s_lat, int(lamc['patch_nlon_lr']) * s_lon)
                _check(tuple(sample['lr_halo'].shape) == expected_lr, f"patch sample shape ok: lr_halo = {expected_lr}", errors, ok)
                _check(tuple(sample['hr_patch_t0'].shape) == expected_hr0, f"patch sample shape ok: hr_patch_t0 = {expected_hr0}", errors, ok)
                _check(tuple(sample['hr_patch_t1'].shape) == expected_hr1, f"patch sample shape ok: hr_patch_t1 = {expected_hr1}", errors, ok)
                _check(np.isfinite(sample['lr_halo'].numpy()).all(), "patch sample finite: lr_halo", errors, ok)
                _check(np.isfinite(sample['hr_patch_t0'].numpy()).all(), "patch sample finite: hr_patch_t0", errors, ok)
                _check(np.isfinite(sample['hr_patch_t1'].numpy()).all(), "patch sample finite: hr_patch_t1", errors, ok)
            except Exception as e:
                errors.append(f"LAMPatchDataset smoke test failed: {type(e).__name__}: {e}")

        if check_rotation:
            expected_rotation_attrs = [
                'rotation_enabled',
                'rotation_pole_target_lat_deg',
                'rotation_pole_target_lon_deg',
                'rotation_method',
                'rotation_interpolation',
            ]
            for a in expected_rotation_attrs:
                if a not in hf.attrs:
                    warnings.append(f"rotation metadata missing: {a}")
                else:
                    ok.append(f"rotation metadata present: {a}")

    return {
        'ok': len(errors) == 0,
        'errors': errors,
        'warnings': warnings,
        'checks_passed': ok,
    }


def main():
    p = argparse.ArgumentParser(description='Verify SWE paired HDF5 dataset integrity.')
    p.add_argument('--h5_path', required=True)
    p.add_argument('--config', default=None)
    p.add_argument('--sample_ics', type=int, default=2)
    p.add_argument('--check_rotation', action='store_true')
    p.add_argument('--reference_h5_path', default=None)
    p.add_argument('--expect_different', action='store_true')
    p.add_argument('--diff_atol', type=float, default=1e-6)
    p.add_argument('--json', action='store_true')
    args = p.parse_args()

    result = verify_dataset(
        h5_path=args.h5_path,
        config_path=args.config,
        sample_ics=args.sample_ics,
        check_rotation=args.check_rotation,
    )

    if args.reference_h5_path is not None:
        ref_result = compare_to_reference(
            h5_path=args.h5_path,
            reference_h5_path=args.reference_h5_path,
            sample_ics=args.sample_ics,
            expect_different=args.expect_different,
            diff_atol=args.diff_atol,
        )
        result['reference_comparison'] = ref_result
        result['ok'] = result['ok'] and ref_result['ok']

    if args.json:
        print(json.dumps(result, indent=2))
    else:
        print(f"Dataset: {args.h5_path}")
        print(f"Status : {'PASS' if result['ok'] else 'FAIL'}")
        print(f"Checks : {len(result['checks_passed'])} passed")
        if result['warnings']:
            print(f"Warnings: {len(result['warnings'])}")
            for w in result['warnings']:
                print(f"  [warn] {w}")
        if result['errors']:
            print(f"Errors: {len(result['errors'])}")
            for e in result['errors']:
                print(f"  [fail] {e}")
        else:
            print("No primary-dataset errors found.")

        if 'reference_comparison' in result:
            ref = result['reference_comparison']
            print(f"Reference file: {args.reference_h5_path}")
            print(f"Reference status: {'PASS' if ref['ok'] else 'FAIL'}")
            if ref['warnings']:
                print(f"Reference warnings: {len(ref['warnings'])}")
                for w in ref['warnings']:
                    print(f"  [warn] {w}")
            if ref['errors']:
                print(f"Reference errors: {len(ref['errors'])}")
                for e in ref['errors']:
                    print(f"  [fail] {e}")
            metrics = ref.get('metrics', {})
            if metrics:
                print('Reference diff metrics:')
                for key, val in metrics.get('mean_abs_diff', {}).items():
                    print(f"  mean_abs_diff[{key}] = {val:.6e}")
                for key, val in metrics.get('max_abs_diff', {}).items():
                    print(f"  max_abs_diff[{key}] = {val:.6e}")
                for key, val in metrics.get('exact_equal', {}).items():
                    print(f"  exact_equal[{key}] = {val}")

    raise SystemExit(0 if result['ok'] else 1)


if __name__ == '__main__':
    main()
