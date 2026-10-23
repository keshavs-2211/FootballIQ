"""Data validation, helpers, visual language shared across FootballIQ analytics."""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle, Arc

import analytics_settings as CFG

PALETTE = {'Team A': '#4bb9ef', 'Team B': '#fa6d6d', 'Unknown': '#aab5bd'}
DARK = '#0c1725'

def read_players(path):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(
            f'Professional player coordinates missing: {path}\n'
            'First run: python refine_coordinates.py (with CLIP set in offline_settings.py).'
        )
    data = pd.read_csv(path, low_memory=False)
    required = {'frame', 'track_id', 'pitch_x_m', 'pitch_y_m', 'team', 'class_name'}
    missing = required - set(data.columns)
    if missing:
        raise ValueError(f'{path}: missing columns {sorted(missing)}. Expected professional_coordinates.csv format.')
    for col in ('frame','track_id','pitch_x_m','pitch_y_m'):
        data[col] = pd.to_numeric(data[col], errors='coerce')
    data = data.dropna(subset=['frame','track_id','pitch_x_m','pitch_y_m']).copy()
    data['frame'] = data.frame.astype(int)
    data['track_id'] = data.track_id.astype(int)
    data['team'] = data.team.fillna('Unknown').astype(str).str.strip()
    data.loc[~data.team.isin((*CFG.TEAMS,)), 'team'] = 'Unknown'
    data['class_name'] = data.class_name.fillna('player').astype(str).str.lower().str.strip()
    data = data[data.class_name.isin(('player','goalkeeper'))]
    if 'is_observed' in data:
        data['is_observed'] = pd.to_numeric(data.is_observed, errors='coerce').fillna(0).astype(int).eq(1)
    elif 'position_source' in data:
        data['is_observed'] = data.position_source.eq('observed_smoothed')
    else:
        raise ValueError('Missing is_observed and position_source: cannot safely distinguish predicted from measured player locations.')
    # Refiner admits small off-field margins; metrics use pitch-only positions.
    data = data[data.pitch_x_m.between(0,CFG.PITCH_LENGTH_M) & data.pitch_y_m.between(0,CFG.PITCH_WIDTH_M)]
    data = data.sort_values(['track_id','frame','is_observed']).drop_duplicates(['track_id','frame'], keep='last')
    if data.empty:
        raise ValueError('No valid player pitch coordinates in the input CSV.')
    # Stabilize minor occasional label noise without merging track IDs.
    stable_team = {}
    for tid, group in data.groupby('track_id'):
        known = group[group.team != 'Unknown'].team.value_counts()
        stable_team[tid] = known.index[0] if len(known) and known.iloc[0]/known.sum() >= 0.8 else 'Unknown'
    data['team'] = data.track_id.map(stable_team)
    return data.sort_values(['frame','track_id']).reset_index(drop=True)


def guess_fps(data, video=None):
    if video is not None and Path(video).is_file():
        try:
            import cv2
            cap = cv2.VideoCapture(str(video))
            fps = float(cap.get(cv2.CAP_PROP_FPS)) if cap.isOpened() else 0
            cap.release()
            if 1 < fps <= 240:
                return fps, 'video'
        except ImportError:
            pass
    if 'timestamp_seconds' in data and data.timestamp_seconds.notna().sum() > 3:
        times = pd.to_numeric(data.timestamp_seconds,errors='coerce')
        valid = pd.DataFrame({'frame':data.frame, 'time':times}).dropna().drop_duplicates('frame').sort_values('frame')
        if len(valid) >= 3:
            df, dt = np.diff(valid.frame.to_numpy()), np.diff(valid.time.to_numpy())
            ok = (df>0) & (dt>0)
            if ok.any():
                est = float(np.median(df[ok] / dt[ok]))
                if 1 < est <= 240:
                    return est, 'CSV timestamp_seconds'
    raise RuntimeError('FPS unavailable. Check VIDEO in offline_settings.py or provide timestamp_seconds in player CSV.')


def setup_pitch(ax, title='', dark=True):
    ax.set_xlim(-3,108)
    ax.set_ylim(71,-3)  # pitch coordinate y=0 at top
    ax.set_aspect('equal', adjustable='box')
    ax.set_facecolor('#135b3e' if dark else '#e1f2e4')
    for x in range(0,105,15):
        if (x//15)%2:
            ax.add_patch(Rectangle((x,0),15,68,facecolor='#ffffff', alpha=.035,zorder=0))
    line='#e8f5ee' if dark else '#27533a'
    kw=dict(fill=False, edgecolor=line,linewidth=1.1,zorder=4)
    ax.add_patch(Rectangle((0,0),105,68,**kw))
    ax.plot([52.5,52.5],[0,68],color=line,lw=1.0,zorder=4)
    ax.add_patch(Circle((52.5,34),9.15,**kw))
    ax.add_patch(Circle((52.5,34),.35,color=line,zorder=4))
    for x in (0,105):
        if x==0:
            ax.add_patch(Rectangle((0,13.84),16.5,40.32,**kw))
            ax.add_patch(Rectangle((0,24.84),5.5,18.32,**kw))
            ax.add_patch(Circle((11,34),.32,color=line,zorder=4))
            ax.add_patch(Arc((11,34),18.3,18.3,theta1=-53,theta2=53,color=line,lw=1,zorder=4))
        else:
            ax.add_patch(Rectangle((88.5,13.84),16.5,40.32,**kw))
            ax.add_patch(Rectangle((99.5,24.84),5.5,18.32,**kw))
            ax.add_patch(Circle((94,34),.32,color=line,zorder=4))
            ax.add_patch(Arc((94,34),18.3,18.3,theta1=127,theta2=233,color=line,lw=1,zorder=4))
    ax.set_xticks([]); ax.set_yticks([])
    for spine in ax.spines.values(): spine.set_visible(False)
    if title: ax.set_title(title,color='#e8f5ee' if dark else DARK,loc='left',weight='bold',fontsize=12,pad=12)
    return ax


def savefig(fig, path):
    path=Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=145, facecolor=fig.get_facecolor(), bbox_inches='tight')
    plt.close(fig)


def short_time(seconds):
    if pd.isna(seconds): return 'N/A'
    m,s=divmod(int(seconds),60)
    return f'{m}m {s:02d}s'
