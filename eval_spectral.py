#!/usr/bin/env python3
"""PSD + coherence diagnostics for PARADIS forecast outputs."""

import argparse
import os
import glob
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch_harmonics as th


def compute_psd_from_realsht(coeffs):
    """PSD from RealSHT coeffs [..., L, M] -> [..., L]."""
    power_lm = torch.view_as_real(coeffs).pow(2).sum(dim=-1)
    psd_m0 = power_lm[..., 0]
    psd_m_pos = 2.0 * torch.sum(power_lm[..., 1:], dim=-1)
    return psd_m0 + psd_m_pos


def compute_coherence_from_realsht(pred_coeffs, true_coeffs):
    """Per-degree coherence [..., L, M] -> [..., L]."""
    leading_dims = pred_coeffs.shape[:-2]
    L = pred_coeffs.shape[-2]
    coh = torch.zeros((*leading_dims, L), device=pred_coeffs.device)
    pred_psd = compute_psd_from_realsht(pred_coeffs)
    true_psd = compute_psd_from_realsht(true_coeffs)
    tiny = torch.finfo(pred_psd.dtype).tiny
    for l in range(L):
        cross = torch.conj(pred_coeffs[..., l, 0]) * true_coeffs[..., l, 0]
        if l >= 1:
            cross_m = torch.sum(
                torch.conj(pred_coeffs[..., l, 1:l + 1]) * true_coeffs[..., l, 1:l + 1],
                dim=-1,
            )
            cross = cross + 2.0 * cross_m
        num = torch.abs(cross)
        den = torch.sqrt(torch.clamp(pred_psd[..., l] * true_psd[..., l], min=tiny))
        coh[..., l] = num / den
    return torch.clamp(coh, 0.0, 1.0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--forecast_dir', type=str, required=True,
                        help='Folder with ic*_prediction_*.pt and ic*_truth_*.pt')
    parser.add_argument('--stats_path', type=str, required=True,
                        help='Path to stats.pt from the training run')
    parser.add_argument('--output_dir', type=str, default=None)
    parser.add_argument('--steps', type=int, nargs='+', default=[1, 5, 10, 20, 30, 40])
    parser.add_argument('--tag', type=str, default='')
    parser.add_argument('--per_step', action='store_true',
                        help='Save one figure per step instead of overlaying all steps')
    args = parser.parse_args()

    out_dir = args.output_dir or args.forecast_dir
    os.makedirs(out_dir, exist_ok=True)

    stats = torch.load(args.stats_path, map_location='cpu')
    inp_mean = stats['inp_mean'].float()
    inp_var  = stats['inp_var'].float()

    pred_files = sorted(glob.glob(os.path.join(args.forecast_dir, 'ic*_prediction_*.pt')))
    if not pred_files:
        raise FileNotFoundError(f'No prediction files in {args.forecast_dir}')

    sample = torch.load(pred_files[0], map_location='cpu')
    _, nlat, nlon = sample['fields'].shape
    print(f'Grid: nlat={nlat}, nlon={nlon}')

    sht = th.RealSHT(nlat, nlon, grid='equiangular')
    channel_names = ['geopotential', 'vorticity', 'divergence']
    pad_width = len(str(max(args.steps)))

    prefix = os.path.basename(pred_files[0]).split('prediction')[0]

    results = {}
    for step in args.steps:
        pred_path = os.path.join(args.forecast_dir, f'{prefix}prediction_{step:0{pad_width}d}.pt')
        true_path = os.path.join(args.forecast_dir, f'{prefix}truth_{step:0{pad_width}d}.pt')
        if not os.path.exists(pred_path) or not os.path.exists(true_path):
            print(f'Step {step}: files missing, skipping')
            continue

        pred_f = torch.load(pred_path, map_location='cpu')['fields'].float()
        true_f = torch.load(true_path, map_location='cpu')['fields'].float()

        pred_unnorm = pred_f * torch.sqrt(inp_var) + inp_mean
        true_unnorm = true_f * torch.sqrt(inp_var) + inp_mean

        # SHT over all channels at once: [3, nlat, nlon] -> [3, L, M]
        c_pred = sht(pred_unnorm)
        c_true = sht(true_unnorm)

        psd_pred  = compute_psd_from_realsht(c_pred).numpy()       # [3, L]
        psd_true  = compute_psd_from_realsht(c_true).numpy()
        coherence = compute_coherence_from_realsht(c_pred, c_true).numpy()
        amp_ratio = psd_pred / (psd_true + 1e-30)

        results[step] = dict(psd_pred=psd_pred, psd_true=psd_true,
                             coherence=coherence, amp_ratio=amp_ratio)
        print(f'Step {step}: coherence[0,5]={coherence[0,5]:.3f}')

    if not results:
        print('No results.')
        return

    L = list(results.values())[0]['psd_pred'].shape[-1]
    degree = np.arange(L)
    steps_done = sorted(results.keys())
    colors = plt.cm.plasma(np.linspace(0.1, 0.9, len(steps_done)))
    tag = f'_{args.tag}' if args.tag else ''

    if args.per_step:
        per_step_dir = os.path.join(out_dir, 'per_step')
        os.makedirs(per_step_dir, exist_ok=True)
        for step in steps_done:
            r = results[step]
            fig, axes = plt.subplots(3, 3, figsize=(15, 12))
            for ci, ch in enumerate(channel_names):
                axes[ci, 0].plot(degree[1:], r['coherence'][ci, 1:], linewidth=1.5)
                axes[ci, 0].axhline(0.5, color='k', linestyle='--', linewidth=0.8, alpha=0.5)
                axes[ci, 0].set_ylim(0, 1)
                axes[ci, 0].set_title(f'{ch} coherence')
                axes[ci, 0].set_xlabel('Degree $l$')
                axes[ci, 0].grid(True, alpha=0.3)

                axes[ci, 1].semilogy(degree[1:], r['psd_pred'][ci, 1:], label='pred', linewidth=1.5)
                axes[ci, 1].semilogy(degree[1:], r['psd_true'][ci, 1:], label='truth', linestyle='--', linewidth=1.5)
                axes[ci, 1].set_title(f'{ch} PSD')
                axes[ci, 1].set_xlabel('Degree $l$')
                axes[ci, 1].legend(fontsize=8)
                axes[ci, 1].grid(True, alpha=0.3, which='both')

                axes[ci, 2].plot(degree[1:], r['amp_ratio'][ci, 1:], linewidth=1.5)
                axes[ci, 2].axhline(1.0, color='k', linestyle='--', linewidth=0.8, alpha=0.5)
                axes[ci, 2].set_title(f'{ch} amp ratio')
                axes[ci, 2].set_xlabel('Degree $l$')
                axes[ci, 2].grid(True, alpha=0.3)

            fig.suptitle(f'Step {step} (t={step}h)', fontsize=14)
            plt.tight_layout()
            plt.savefig(os.path.join(per_step_dir, f'step_{step:02d}{tag}.png'), dpi=120, bbox_inches='tight')
            plt.close()
        print(f'Per-step plots saved to {per_step_dir}')
        return

    # Coherence plot
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ci, ch in enumerate(channel_names):
        ax = axes[ci]
        for i, step in enumerate(steps_done):
            ax.plot(degree[1:], results[step]['coherence'][ci, 1:],
                    color=colors[i], label=f't={step}', linewidth=1.5)
        ax.axhline(0.5, color='k', linestyle='--', linewidth=0.8, alpha=0.5)
        ax.set_xlabel('Degree $l$')
        ax.set_ylabel('Coherence')
        ax.set_title(ch)
        ax.set_ylim(0, 1)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    fig.suptitle('Coherence (prediction vs truth)')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f'coherence{tag}.png'), dpi=150, bbox_inches='tight')
    plt.close()

    # PSD plot
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ci, ch in enumerate(channel_names):
        ax = axes[ci]
        for i, step in enumerate(steps_done):
            ax.semilogy(degree[1:], results[step]['psd_pred'][ci, 1:],
                        color=colors[i], label=f'pred t={step}', linewidth=1.5)
            ax.semilogy(degree[1:], results[step]['psd_true'][ci, 1:],
                        color=colors[i], linestyle='--', linewidth=1.0, alpha=0.6)
        ax.set_xlabel('Degree $l$')
        ax.set_ylabel('Power')
        ax.set_title(ch)
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3, which='both')
    fig.suptitle('PSD: solid=pred, dashed=truth')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f'psd{tag}.png'), dpi=150, bbox_inches='tight')
    plt.close()

    # Amp ratio plot
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ci, ch in enumerate(channel_names):
        ax = axes[ci]
        for i, step in enumerate(steps_done):
            ax.plot(degree[1:], results[step]['amp_ratio'][ci, 1:],
                    color=colors[i], label=f't={step}', linewidth=1.5)
        ax.axhline(1.0, color='k', linestyle='--', linewidth=0.8, alpha=0.5)
        ax.set_xlabel('Degree $l$')
        ax.set_ylabel('PSD ratio (pred/truth)')
        ax.set_title(ch)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    fig.suptitle('Amplitude ratio (pred/truth)')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f'amp_ratio{tag}.png'), dpi=150, bbox_inches='tight')
    plt.close()

    print(f'Plots saved to {out_dir}')


if __name__ == '__main__':
    main()
