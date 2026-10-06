from scan import plot_scan

import numpy as np

import argparse, os


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='re-zoom plots from *_scan.npz written by scan.py, without reloading the ensemble')
    parser.add_argument('npz', nargs='+', help='path(s) to *_scan.npz produced by scan.py')
    parser.add_argument('--xlim', type=float, nargs=2, default=None, help='default: full scanned range')
    parser.add_argument('--ylim', type=float, nargs=2, default=None, help='default: curves max inside xlim')
    parser.add_argument('--suffix', default='_zoom', help='appended to the npz name for the output png')
    args = parser.parse_args()

    for path in args.npz:
        d = np.load(path)
        wc = os.path.basename(path).replace('_scan.npz', '')
        out = path.replace('.npz', f'{args.suffix}.png')
        plot_scan(d['grid'], d['nll'], d['nll_true'] if 'nll_true' in d else None, wc, out, args.xlim, args.ylim)
        print(f'Saved: {out}')
