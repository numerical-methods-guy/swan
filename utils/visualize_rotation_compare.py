#!/usr/bin/env python3
import argparse
import os

import h5py
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

CHANNEL_NAMES = ["Geopotential h", "Vorticity", "Divergence"]
CMAPS = ["viridis", "RdBu_r", "RdBu_r"]


def _latlon_axes(nlat, nlon):
    lats = np.linspace(90.0, -90.0, nlat)
    lons = np.linspace(0.0, 360.0, nlon, endpoint=False)
    return lats, lons


def _lat_to_row(lat_deg, nlat):
    return int(np.argmin(np.abs(_latlon_axes(nlat, 1)[0] - lat_deg)))


def _lon_to_col(lon_deg, nlon):
    lons = _latlon_axes(1, nlon)[1]
    lon_deg = lon_deg % 360.0
    return int(np.argmin(np.abs(((lons - lon_deg + 180.0) % 360.0) - 180.0)))


def _crop_lon_cyclic(arr, lon0, width):
    nlon = arr.shape[-1]
    lon0 = lon0 % nlon
    if lon0 + width <= nlon:
        return arr[..., lon0:lon0 + width]
    return np.concatenate([arr[..., lon0:], arr[..., :(lon0 + width) % nlon]], axis=-1)


def _box_from_indices(lat0, lon0, patch_h, patch_w, nlat, nlon):
    dlat = 180.0 / nlat
    dlon = 360.0 / nlon
    lat_top = 90.0 - lat0 * dlat
    lat_bot = lat_top - patch_h * dlat
    lon_left = (lon0 * dlon) % 360.0
    lon_width = patch_w * dlon
    return {
        "lon_left": lon_left,
        "lon_width": lon_width,
        "lat_bot": lat_bot,
        "lat_height": patch_h * dlat,
    }


def _style_geo_axes(ax):
    ax.set_xlabel("Longitude (°E)")
    ax.set_ylabel("Latitude (°N)")
    ax.set_xticks(np.arange(0, 361, 60))
    ax.set_yticks(np.arange(-90, 91, 30))


def _draw_target(ax, lat_deg, lon_deg, color="k"):
    ax.scatter([lon_deg % 360.0], [lat_deg], s=110, marker="x", c=color, linewidths=2.0, zorder=5)


def _draw_box(ax, box, color="cyan", lw=2.0):
    if box is None:
        return
    lon_left = box["lon_left"]
    lon_width = box["lon_width"]
    lat_bot = box["lat_bot"]
    lat_height = box["lat_height"]
    if lon_left + lon_width <= 360.0:
        ax.add_patch(Rectangle((lon_left, lat_bot), lon_width, lat_height,
                               fill=False, edgecolor=color, linewidth=lw, zorder=4))
    else:
        first_width = 360.0 - lon_left
        second_width = lon_width - first_width
        ax.add_patch(Rectangle((lon_left, lat_bot), first_width, lat_height,
                               fill=False, edgecolor=color, linewidth=lw, zorder=4))
        ax.add_patch(Rectangle((0.0, lat_bot), second_width, lat_height,
                               fill=False, edgecolor=color, linewidth=lw, zorder=4))


def plot_global_comparison(native_arr, rotated_arr, title, out_path, cmap="viridis",
                           target_lat_deg=None, target_lon_deg=None,
                           rotated_box=None):
    diff = rotated_arr - native_arr
    vmax_diff = np.nanmax(np.abs(diff)) + 1e-12
    extent = [0.0, 360.0, -90.0, 90.0]
    fig, axes = plt.subplots(1, 3, figsize=(19, 5.5))
    ims = [native_arr, rotated_arr, diff]
    titles = ["Native", "Rotated", "Rotated - Native"]
    cmaps = [cmap, cmap, "RdBu_r"]
    vmins = [np.nanmin(native_arr), np.nanmin(rotated_arr), -vmax_diff]
    vmaxs = [np.nanmax(native_arr), np.nanmax(rotated_arr), vmax_diff]

    for i, (ax, img, ttl, cm, vmin, vmax_) in enumerate(zip(axes, ims, titles, cmaps, vmins, vmaxs)):
        im = ax.imshow(img, origin="upper", cmap=cm, vmin=vmin, vmax=vmax_,
                       aspect="auto", extent=extent)
        ax.set_title(ttl)
        _style_geo_axes(ax)
        if i == 1 and target_lat_deg is not None and target_lon_deg is not None:
            _draw_target(ax, target_lat_deg, target_lon_deg, color="black")
            _draw_box(ax, rotated_box, color="cyan")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_rotated_overview(rotated_arr, title, out_path, cmap="viridis",
                          target_lat_deg=None, target_lon_deg=None,
                          rotated_box=None):
    fig, ax = plt.subplots(figsize=(9, 4.8))
    im = ax.imshow(rotated_arr, origin="upper", cmap=cmap, aspect="auto",
                   extent=[0.0, 360.0, -90.0, 90.0])
    ax.set_title(title)
    _style_geo_axes(ax)
    if target_lat_deg is not None and target_lon_deg is not None:
        _draw_target(ax, target_lat_deg, target_lon_deg, color="black")
    _draw_box(ax, rotated_box, color="cyan")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_patch_comparison(rotated_arr, lat0, lon0, patch_h, patch_w, title, out_path, cmap="viridis"):
    assert patch.shape[0] > 0, f"Empty latitude slice: lat0={lat0}, patch_h={patch_h}, arr_nlat={rotated_arr.shape[0]}"
    assert patch.shape[1] > 0, f"Empty longitude slice: lon0={lon0}, patch_w={patch_w}, arr_nlon={rotated_arr.shape[1]}"

    fig, ax = plt.subplots(figsize=(6, 4))
    im = ax.imshow(patch, origin="upper", cmap=cmap, aspect="auto")
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description="Compare native and rotated SWE HDF5 datasets visually.")
    p.add_argument("--native_h5", required=True)
    p.add_argument("--rotated_h5", required=True)
    p.add_argument("--ic_idx", type=int, default=0)
    p.add_argument("--step", type=int, default=0)
    p.add_argument("--output_dir", default="rotation_compare")
    p.add_argument("--pole_target_lat_deg", type=float, default=0.0)
    p.add_argument("--pole_target_lon_deg", type=float, default=0.0)
    p.add_argument("--patch_nlat_lr", type=int, default=None,
                   help="Optional LR patch height. If omitted with patch_nlon_lr, only global views are generated.")
    p.add_argument("--patch_nlon_lr", type=int, default=None,
                   help="Optional LR patch width. If omitted with patch_nlat_lr, only global views are generated.")
    args = p.parse_args()

    if (args.patch_nlat_lr is None) ^ (args.patch_nlon_lr is None):
        raise ValueError("Provide both --patch_nlat_lr and --patch_nlon_lr, or omit both for global-only plots.")

    os.makedirs(args.output_dir, exist_ok=True)

    with h5py.File(args.native_h5, "r") as nf, h5py.File(args.rotated_h5, "r") as rf:
        lr_nlat = int(rf.attrs["lr_nlat"])
        lr_nlon = int(rf.attrs["lr_nlon"])
        hr_nlat = int(rf.attrs["hr_nlat"])
        hr_nlon = int(rf.attrs["hr_nlon"])
        s_lat = int(rf.attrs["upscale_factor_lat"])
        s_lon = int(rf.attrs["upscale_factor_lon"])
        assert s_lat == s_lon
        s = s_lat

        for grp in ["lr", "hr"]:
            for name in ["fields", "winds"]:
                assert nf[f"{grp}/{name}"].shape == rf[f"{grp}/{name}"].shape

        native_lr = np.asarray(nf["lr/fields"][args.ic_idx, args.step])
        rotated_lr = np.asarray(rf["lr/fields"][args.ic_idx, args.step])
        native_hr = np.asarray(nf["hr/fields"][args.ic_idx, args.step])
        rotated_hr = np.asarray(rf["hr/fields"][args.ic_idx, args.step])

        target_lr_row = _lat_to_row(args.pole_target_lat_deg, lr_nlat)
        target_lr_col = _lon_to_col(args.pole_target_lon_deg, lr_nlon)
        target_hr_row = _lat_to_row(args.pole_target_lat_deg, hr_nlat)
        target_hr_col = _lon_to_col(args.pole_target_lon_deg, hr_nlon)

        have_patch = args.patch_nlat_lr is not None and args.patch_nlon_lr is not None
        lat0_lr = lon0_lr = lat0_hr = lon0_hr = None
        lr_box = hr_box = None

        if have_patch:
            lat0_lr = max(0, min(target_lr_row - args.patch_nlat_lr // 2, lr_nlat - args.patch_nlat_lr))
            lon0_lr = (target_lr_col - args.patch_nlon_lr // 2) % lr_nlon
            patch_nlat_hr = args.patch_nlat_lr * s
            patch_nlon_hr = args.patch_nlon_lr * s
            lat0_hr = max(0, min(target_hr_row - patch_nlat_hr // 2, hr_nlat - patch_nlat_hr))
            lon0_hr = (target_hr_col - patch_nlon_hr // 2) % hr_nlon
            lr_box = _box_from_indices(lat0_lr, lon0_lr, args.patch_nlat_lr, args.patch_nlon_lr, lr_nlat, lr_nlon)
            hr_box = _box_from_indices(lat0_hr, lon0_hr, patch_nlat_hr, patch_nlon_hr, hr_nlat, hr_nlon)

        for ch, (name, cmap) in enumerate(zip(CHANNEL_NAMES, CMAPS)):
            plot_global_comparison(
                native_lr[ch],
                rotated_lr[ch],
                f"LR step={args.step} ic={args.ic_idx} :: {name}",
                os.path.join(args.output_dir, f"lr_{ch}_global.png"),
                cmap=cmap,
                target_lat_deg=args.pole_target_lat_deg,
                target_lon_deg=args.pole_target_lon_deg,
                rotated_box=lr_box,
            )
            plot_global_comparison(
                native_hr[ch],
                rotated_hr[ch],
                f"HR step={args.step} ic={args.ic_idx} :: {name}",
                os.path.join(args.output_dir, f"hr_{ch}_global.png"),
                cmap=cmap,
                target_lat_deg=args.pole_target_lat_deg,
                target_lon_deg=args.pole_target_lon_deg,
                rotated_box=hr_box,
            )
            plot_rotated_overview(
                rotated_lr[ch],
                f"Rotated LR target view :: {name}",
                os.path.join(args.output_dir, f"lr_{ch}_rotated_target.png"),
                cmap=cmap,
                target_lat_deg=args.pole_target_lat_deg,
                target_lon_deg=args.pole_target_lon_deg,
                rotated_box=lr_box,
            )
            plot_rotated_overview(
                rotated_hr[ch],
                f"Rotated HR target view :: {name}",
                os.path.join(args.output_dir, f"hr_{ch}_rotated_target.png"),
                cmap=cmap,
                target_lat_deg=args.pole_target_lat_deg,
                target_lon_deg=args.pole_target_lon_deg,
                rotated_box=hr_box,
            )

        if have_patch:
            for ch, (name, cmap) in enumerate(zip(CHANNEL_NAMES, CMAPS)):
                plot_patch_comparison(
                    rotated_lr[ch], lat0_lr, lon0_lr, args.patch_nlat_lr, args.patch_nlon_lr,
                    f"Rotated LR patch near mapped original North Pole :: {name}",
                    os.path.join(args.output_dir, f"lr_{ch}_pole_patch.png"),
                    cmap=cmap,
                )
                plot_patch_comparison(
                    rotated_hr[ch], lat0_hr, lon0_hr, patch_nlat_hr, patch_nlon_hr,
                    f"Rotated HR patch near mapped original North Pole :: {name}",
                    os.path.join(args.output_dir, f"hr_{ch}_pole_patch.png"),
                    cmap=cmap,
                )

        with open(os.path.join(args.output_dir, "patch_location.txt"), "w") as f:
            f.write(f"LR target row,col: {target_lr_row},{target_lr_col}\n")
            f.write(f"HR target row,col: {target_hr_row},{target_hr_col}\n")
            f.write(f"pole_target_lat_deg={args.pole_target_lat_deg}\n")
            f.write(f"pole_target_lon_deg={args.pole_target_lon_deg}\n")
            if have_patch:
                f.write(f"LR top-left row,col: {lat0_lr},{lon0_lr}\n")
                f.write(f"HR top-left row,col: {lat0_hr},{lon0_hr}\n")
                f.write(f"patch_nlat_lr={args.patch_nlat_lr}\n")
                f.write(f"patch_nlon_lr={args.patch_nlon_lr}\n")
            else:
                f.write("No patch requested; global-only visualizations were generated.\n")


if __name__ == "__main__":
    main()