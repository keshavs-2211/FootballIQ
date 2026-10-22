"""Player and team occupancy maps; measured footprints, not ball possession."""
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from pathlib import Path
import analytics_settings as CFG
from .common import setup_pitch, savefig, PALETTE


def grid_from(g):
    nx=int(np.ceil(CFG.PITCH_LENGTH_M/CFG.HEATMAP_BIN_SIZE_M))
    ny=int(np.ceil(CFG.PITCH_WIDTH_M/CFG.HEATMAP_BIN_SIZE_M))
    counts,_,_=np.histogram2d(g.pitch_y_m,g.pitch_x_m,bins=[ny,nx],range=[[0,CFG.PITCH_WIDTH_M],[0,CFG.PITCH_LENGTH_M]])
    return gaussian_filter(counts,sigma=CFG.HEATMAP_SIGMA_BINS)


def draw(g, title, output, colorbar_label='Relative observed occupancy'):
    h=grid_from(g)
    fig,ax=plt.subplots(figsize=(13,8),facecolor='#0c1725')
    setup_pitch(ax,title)
    if h.max()>0:
        # Local normalization; cannot compare absolute intensity between images.
        heat=np.ma.masked_where(h/h.max()<.035,h/h.max())
        im=ax.imshow(heat,extent=[0,CFG.PITCH_LENGTH_M,CFG.PITCH_WIDTH_M,0],
                     cmap='turbo',interpolation='bilinear',alpha=.77,zorder=2,vmin=0,vmax=1)
        cb=fig.colorbar(im,ax=ax,orientation='horizontal',pad=.028,shrink=.52,aspect=36)
        cb.ax.tick_params(colors='#e2ecec',labelsize=8)
        cb.set_label(f'{colorbar_label} (each chart scaled separately)',color='#e2ecec',fontsize=9)
    ax.text(.01,-.05,f'Observed player positions: {len(g):,}  |  normalized separately',
            transform=ax.transAxes,color='#c9d8d9',fontsize=9)
    savefig(fig,output)


def generate(data,root):
    base=data[data.is_observed] if CFG.HEATMAP_OBSERVED_ONLY else data
    root=Path(root); out=[]
    for team in CFG.TEAMS:
        g=base[base.team==team]
        if not g.empty:
            p=root/'heatmaps'/f'{team.lower().replace(" ", "_")}_heatmap.png'
            draw(g,f'{team.upper()}  /  POSITION HEATMAP',p)
            out.append(p)
    for tid,g in base.groupby('track_id'):
        if len(g)<CFG.MIN_PLAYER_HEATMAP_FRAMES or g.team.iloc[0]=='Unknown': continue
        p=root/'heatmaps'/'players'/f'player_{int(tid):03d}_{g.team.iloc[0].replace(" ", "_").lower()}.png'
        draw(g,f'PLAYER #{int(tid)}  /  {g.team.iloc[0].upper()}',p)
        out.append(p)
    return out
