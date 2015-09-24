import sys
import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec

dr = 'PyCORN'
if dr not in sys.path: sys.path.append(dr)
from pycorn import pc_res3

blue = '#377EB8'
green = '#4DAF4A'
parser = argparse.ArgumentParser()

parser.add_argument("inp_res",
                    help = "Input .res file(s)",
                    nargs='+',
                    metavar = "<file>.res")
parser.add_argument("-e", "--ext", default='pdf',
                    help = "Image type to use, e.g. 'jpg', 'png', 'eps', or 'pdf' (default: pdf)")
parser.add_argument("--xmin", default=None, type=float,
                    help = "Lower bound on the x-axis")
parser.add_argument("--xmax", default=None, type=float,
                    help = "Upper bound on the x-axis")
parser.add_argument("--ymin", default=0, type=float,
                    help = "Lower bound on the y-axis")
parser.add_argument("--ymax", default=None, type=float,
                    help = "Upper bound on the y-axis")
parser.add_argument("--ymin2", default=None, type=float,
                    help = "Lower bound on the right y-axis")
parser.add_argument("--ymax2", default=None, type=float,
                    help = "Upper bound on the right y-axis")
parser.add_argument("--dpi", default=None, type=int,
                    help = "DPI (dots per inch) for raster images (png, jpg, etc.)")
parser.add_argument("--no_fractions",
                    dest="fractions",
                    help="Disable plotting of fractions",
                    action = "store_false")
parser.add_argument("--frac_cutoff",
                    dest="frac_cutoff",
                    help="The ratio at which fractions are considered too"
                    "small to plot the number for. Should be between 0 and 1.",
                    type=float,
                    default=0.6)

args = parser.parse_args()

def mapper(min_val, max_val, perc):
    '''
    calculate relative position in delta min/max
    '''
    x = abs(max_val - min_val) * perc
    if min_val < 0:
        return (x - abs(min_val))
    else:
        return (x + min_val)


def expander(min_val, max_val, perc):
    '''
    expand -/+ direction of two values by a percentage of their delta
    '''
    delta = abs(max_val - min_val)
    x = delta * perc
    return (min_val - x, max_val + x)

for fname in args.inp_res:
    path = Path(fname)
    pc = pc_res3(fname)
    pc.load()
    UVx, UV = np.array(pc['UV']['data']).T
    ix = np.isfinite(UVx)
    if args.xmin is not None:
        ix = ix & (UVx >= args.xmin)
    if args.xmax is not None:
        ix = ix & (UVx <= args.xmax)
    UV -= min(UV[ix])
    xmin = min(UVx) if args.xmin is None else args.xmin
    xmax = max(UVx) if args.xmax is None else args.xmax
    condx, cond = np.array(pc['Cond']['data']).T
    
    if args.fractions:
        frac_data = pc['Fractions']['data']
        frac_x, frac_y = list(zip(*frac_data))
        frac_delta = [abs(a - b) for a, b in zip(frac_x, frac_x[1:])]
        frac_delta.append(frac_delta[-1])
        
        if args.xmin is None: xmin = min(frac_x)
        if args.xmax is None: xmax = frac_x[-1] + frac_delta[-1]
    
    for name in 'general', 'sizeex':
        if args.fractions:
            fig = plt.figure(figsize=(8, 6)) 
            gs1, gs2 = gridspec.GridSpec(2, 1, height_ratios=[19, 1], hspace=0) 
            ax = plt.subplot(gs1)
            ax_fracs = plt.subplot(gs2, sharex=ax)
        else:
            fig, ax = plt.subplots(figsize=(4,3))
        ax.plot(UVx, UV, color=blue, lw=2)

        ax2 = ax.twinx()
        ax2.plot(condx, cond, color=green, lw=2)
        
        ax_to_label = ax_fracs if args.fractions else ax
        ax_to_label.set_xlabel('Elution Volume (ml)')
        ax.set_ylabel('Absorbance (mAu)', color=blue)
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(args.ymin, args.ymax)
        ax2.set_ylabel(
            r'Conductivity $\left(\frac{\mathrm{mS}}{\mathrm{cm}}\right)$',
            color=green)

        for tl in ax.get_yticklabels():
            tl.set_color(blue)

        for tl in ax2.get_yticklabels():
            tl.set_color(green)
        
        ymax2 = args.ymax2
        if name == 'sizeex':
            mx = max(cond)
            exp = int(np.log10(mx/2.))
            ymax2 = round(mx*2, -exp)*2
        ax2.set_ylim(args.ymin2, ymax2)
        
        if args.fractions:
            frac_font_size = 8
            if len(frac_data) < 20:
                frac_font_size = 12
            
            frac_delta_median = np.median(frac_delta)
            for (x, fracname), delta in zip(frac_data, frac_delta):
                try:
                    _ = int(fracname)
                except ValueError:
                    fracname = ''
                    
                ax_fracs.axvline(x=x, color='r', linewidth=0.85)
                
                if delta < args.frac_cutoff * frac_delta_median:
                    continue
                ax_fracs.annotate(str(fracname), xy=(x + delta * 0.6, 0.1),
                         horizontalalignment='center', verticalalignment='bottom', size=frac_font_size, rotation=90)
            ax_fracs.set_ylim(0, 1)
            ax_fracs.set_yticks([])
            plt.setp(ax.get_xticklabels(), visible=False)
            plt.setp(ax.get_xticklines(), visible=False)
            ax_fracs.get_xaxis().set_tick_params(
                which='both', direction='out', top='off')
        
        outputpath = path.with_name('{}-{}.{}'.format(
                    path.stem, name, args.ext))
        plt.savefig(str(outputpath), bbox_inches='tight', dpi=args.dpi)
        plt.close()
        print(outputpath)
