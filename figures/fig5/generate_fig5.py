"""Replot the ten robustness panels from archived numerical results.

No model evaluation, interpolation, or substitution of missing results is performed.
"""
from pathlib import Path
import hashlib
import json
import re

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
DATA = HERE / 'data'
OUTPUT = HERE / 'output'
METHODS = ['Tan24','Wu25','Xi24','Xi25','Lin18','Proposed']
LABELS = ['Tan et al. (2024)','Wu et al. (2025)','Xi et al. (2024)',
          'Xi et al. (2025)','Lin et al. (2018)','VGCN-B']
COLORS = dict(zip(METHODS,['#0072B2','#E69F00','#009E73','#CC79A7','#D55E00','#D62728']))
MARKERS = dict(zip(METHODS,['v','s','D','^','o','P']))
PANELS = [
    (1,'Vertex deletion','Deleted vertices (%)'),
    (2,'Vertex addition','Added vertices (%)'),
    (3,'Object deletion','Deleted objects (%)'),
    (10,'Sequence shuffling','Reordering mode'),
    (6,'Translation',r'Translation ($\Delta x$, $\Delta y$)'),
    (7,'Scaling','Scale factor'),
    (8,'Rotation','Rotation (degree)'),
    (9,'Flipping','Flipping mode'),
    (5,'Cropping','Cropping mode'),
    (4,'Noise','Perturbed vertices (%)'),
]


def configure_matplotlib():
    plt.rcParams.update({
        'font.family':'DejaVu Sans','font.size':7.0,'axes.labelsize':7.0,
        'axes.titlesize':7.8,'legend.fontsize':7.3,'xtick.labelsize':6.4,
        'ytick.labelsize':6.4,'axes.linewidth':.65,'xtick.major.width':.55,
        'ytick.major.width':.55,'xtick.major.size':2.6,'ytick.major.size':2.6,
        'svg.fonttype':'none','pdf.fonttype':42,'ps.fonttype':42,
        'savefig.bbox':'tight','savefig.pad_inches':.04,
    })


def prepare_panel(number):
    frame = pd.read_csv(DATA / f'Fig{number}_avg_nc_comparison.csv')
    if list(frame.columns) != ['Label','Proposed','Tan24','Wu25','Xi24','Xi25','Lin18']:
        raise ValueError(f'Unexpected data schema for panel source {number}')
    frame['source_row'] = np.arange(len(frame))
    groups = None
    if number in (2,4):
        pattern = r'^(?:强度:\s*)?([\d.]+)-(\d+)%$'
        parsed = frame.Label.str.extract(pattern)
        if parsed.isna().any().any():
            raise ValueError('Unrecognized attack intensity label')
        frame['strength'] = parsed[0].astype(float)
        frame['percent'] = parsed[1].astype(int)
        frame = frame[frame.percent.isin([10,50,90])].copy()
        expected = [0.,1.,2.] if number == 2 else [.4,.6,.8]
        if frame.strength.drop_duplicates().tolist() != expected or len(frame) != 9:
            raise ValueError('Attack-grid selection differs from original figure')
        ticks = frame.percent.astype(str).tolist()
        groups = [(f'Strength {s:g}', i*3+1) for i,s in enumerate(expected)]
    elif number in (1,3):
        ticks = frame.Label.str.rstrip('%').tolist()
    elif number == 7:
        ticks = [f'{float(x.rstrip("%"))/100:g}' for x in frame.Label]
    elif number == 8:
        ticks = frame.Label.str.rstrip('°').tolist()
    elif number == 10:
        ticks = ['Reverse\nvertices','Shuffle\nvertices','Reverse\nobjects','Shuffle\nobjects']
    elif number == 6:
        ticks = ['(20, 0)','(0, 20)','(20, 20)','(20, 40)','(30, 10)']
    elif number == 9:
        ticks = ['Both axes','X-axis','Y-axis']
    elif number == 5:
        ticks = ['Center X\n50%','Center Y\n50%','Upper-left','Lower-right','Random\n40%']
    else:
        raise ValueError(number)
    values = frame[METHODS].to_numpy(dtype=float)
    if np.any((values[np.isfinite(values)] < 0) | (values[np.isfinite(values)] > 1.000001)):
        raise ValueError('NC outside its numerical range')
    assert len(ticks) == len(frame)
    return frame, ticks, groups


def main():
    configure_matplotlib()
    OUTPUT.mkdir(exist_ok=True)
    fig, axes = plt.subplots(5,2,figsize=(7.35,8.6),constrained_layout=True)
    fig.get_layout_engine().set(rect=(0,.005,1,.933),h_pad=.048,w_pad=.025,hspace=.07,wspace=.04)
    long_rows = []
    for i,(number,title,xlabel) in enumerate(PANELS):
        ax = axes.flat[i]
        frame,ticks,groups = prepare_panel(number)
        x = np.arange(len(frame))
        for method in METHODS:
            values = frame[method].to_numpy(dtype=float)
            for position,(_,row) in enumerate(frame.iterrows()):
                long_rows.append({'panel':chr(97+i),'attack':title,'source_figure':number,
                    'source_row':int(row.source_row),'source_label':row.Label,
                    'display_position':position,'display_tick':ticks[position].replace('\n',' '),
                    'method':method,'nc':row[method]})
            if not np.isfinite(values).any():
                continue
            proposed = method == 'Proposed'
            ax.plot(x,values,color=COLORS[method],marker=MARKERS[method],
                markersize=3.5 if proposed else 3.0,markeredgewidth=.4,
                linewidth=1.2 if proposed else .95,alpha=1 if proposed else .88,
                zorder=5 if proposed else 3,label=LABELS[METHODS.index(method)])
        ax.set_ylim(.35,1.04)
        ax.set_yticks([.4,.6,.8,1.0])
        ax.set_xlim(-.35,len(frame)-.65)
        ax.grid(axis='y',color='#D8D8D8',linewidth=.45,alpha=.9,zorder=0)
        ax.axhline(.90,color='#777777',linewidth=.6,linestyle=(0,(3,2)),zorder=1)
        ax.spines[['top','right']].set_visible(False)
        ax.set_title(f'({chr(97+i)}) {title}',loc='left',pad=3,fontweight='bold')
        ax.set_xticks(x,ticks)
        ax.set_xlabel(xlabel,labelpad=2.2)
        if i % 2 == 0:
            ax.set_ylabel('NC',labelpad=2)
        ax.tick_params(axis='both',pad=1.4)
        if groups:
            for boundary in (2.5,5.5):
                ax.axvline(boundary,color='#CBCBCB',linewidth=.5,linestyle=(0,(3,2)),zorder=0)
            for label,center in groups:
                ax.text(center,-.24,label,transform=ax.get_xaxis_transform(),
                    ha='center',va='top',fontsize=6.2)
            ax.xaxis.labelpad=18
    handles = [Line2D([],[],color=COLORS[m],marker=MARKERS[m],markersize=4,
                      linewidth=1.2 if m=='Proposed' else .95,markeredgewidth=.4) for m in METHODS]
    fig.legend(handles,LABELS,loc='upper center',bbox_to_anchor=(.5,.997),ncol=3,
               frameon=False,handlelength=1.65,handletextpad=.4,columnspacing=1.9,
               labelspacing=.6,borderaxespad=0,fontsize=7.4)
    fig.canvas.draw()
    for suffix in ['pdf','svg','png','tiff']:
        kwargs = {'dpi':600} if suffix in ('png','tiff') else {}
        if suffix == 'tiff':
            kwargs['pil_kwargs'] = {'compression':'tiff_lzw'}
        fig.savefig(OUTPUT/f'Fig5_robustness.{suffix}',**kwargs)
    fig.savefig(OUTPUT/'Fig5_preview.png',dpi=160)
    pd.DataFrame(long_rows).to_csv(OUTPUT/'Fig5_source_data.csv',index=False,float_format='%.6f')
    report = {
        'purpose':'Restyle the ten original robustness panels without changing numerical results.',
        'archetype':'quantitative grid, 5 rows by 2 columns',
        'canvas_inches':[7.35,8.6],'raster_dpi':600,'methods':METHODS,
        'axis_range':[.35,1.04],'reference_line':.90,
        'reference_line_meaning':'Original nominal NC level, not evidence of calibrated verification accuracy.',
        'plotted_values':sum(pd.notna(r['nc']) for r in long_rows),
        'missing_values':sum(pd.isna(r['nc']) for r in long_rows),
        'missing_data_policy':'NaN values remain absent; no interpolation or replacement.',
        'display_selection':'Addition and noise retain the original displayed 10/50/90-percent subset at each strength.',
        'scale_display':'Percentages expressed as equivalent scale factors, e.g. 90% = 0.9.',
        'statistics':'Archived method-level mean NC; no uncertainty or replicate information supplied in these tables.',
        'input_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(DATA.glob('*.csv'))},
        'matplotlib_version':matplotlib.__version__,
    }
    (OUTPUT/'figure_audit.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    plt.close(fig)
    print(json.dumps({'output':str(OUTPUT),'plotted_values':report['plotted_values'],'missing_values':report['missing_values']},indent=2))


if __name__ == '__main__':
    main()
