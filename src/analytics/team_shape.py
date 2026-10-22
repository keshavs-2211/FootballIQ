"""Simultaneous team spread and average pitch locations (not a fixed tactical formation)."""
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull, QhullError
import analytics_settings as CFG
from .common import setup_pitch, savefig, PALETTE


def _observed_outfield(data):
    g=data[data.is_observed] if CFG.SHAPE_OBSERVED_ONLY else data
    if not CFG.SHAPE_INCLUDE_GOALKEEPERS: g=g[g.class_name!='goalkeeper']
    return g[g.team.isin(CFG.TEAMS)]


def analyze(data,fps,root):
    base=_observed_outfield(data)
    samples=[]
    for (frame,team),g in base.groupby(['frame','team']):
        # Unique IDs, measured simultaneously; 4+ outfield players needed.
        if len(g)<CFG.SHAPE_MIN_PLAYERS: continue
        xy=g[['pitch_x_m','pitch_y_m']].to_numpy()
        width=float(np.percentile(xy[:,1],90)-np.percentile(xy[:,1],10))
        depth=float(np.percentile(xy[:,0],90)-np.percentile(xy[:,0],10))
        area=float('nan')
        if len(xy)>=3:
            try: area=float(ConvexHull(xy).volume)
            except QhullError: pass
        samples.append(dict(frame=int(frame),team=team,players_visible=len(g),
            center_x_m=round(float(np.mean(xy[:,0])),2),center_y_m=round(float(np.mean(xy[:,1])),2),
            width_m_p10_p90=round(width,2),depth_m_p10_p90=round(depth,2),
            convex_hull_area_m2=round(area,2)))
    columns=['frame','team','players_visible','center_x_m','center_y_m','width_m_p10_p90','depth_m_p10_p90','convex_hull_area_m2']
    per_frame=pd.DataFrame(samples,columns=columns)
    team_summary=[]
    for team in CFG.TEAMS:
        g=per_frame[per_frame.team==team]
        team_summary.append(dict(team=team,eligible_frames=len(g),
            eligible_time_s=round(len(g)/fps,2),
            mean_observed_outfield_players=round(float(g.players_visible.mean()),2) if len(g) else None,
            average_width_m=round(float(g.width_m_p10_p90.mean()),2) if len(g) else None,
            average_depth_m=round(float(g.depth_m_p10_p90.mean()),2) if len(g) else None,
            average_convex_hull_area_m2=round(float(g.convex_hull_area_m2.mean()),2) if g.convex_hull_area_m2.notna().any() else None,
            mean_team_center_x_m=round(float(g.center_x_m.mean()),2) if len(g) else None,
            mean_team_center_y_m=round(float(g.center_y_m.mean()),2) if len(g) else None))
    summary=pd.DataFrame(team_summary)
    root=Path(root)
    for team in CFG.TEAMS:
        g=base[base.team==team]
        fig,ax=plt.subplots(figsize=(13,8),facecolor='#0c1725')
        setup_pitch(ax,f'{team.upper()}  /  MEAN TRACKLET POSITIONS')
        for tid,group in g.groupby('track_id'):
            if len(group)<CFG.MIN_POSITION_SAMPLES_FOR_LABEL: continue
            x,y=group[['pitch_x_m','pitch_y_m']].median().to_numpy()
            # Median over observed positions; not a simultaneous lineup.
            ax.scatter(x,y,s=225,color=PALETTE[team],edgecolors='white',linewidths=1.2,zorder=6,alpha=.93)
            ax.text(x,y,str(tid),color='#0c1725',fontsize=9,ha='center',va='center',weight='bold',zorder=7)
        ax.text(.01,-.05,'Each marker is the median observed position of a TRACK ID (not a verified shirt number).',
                transform=ax.transAxes,color='#c9d8d9',fontsize=9)
        savefig(fig,root/'formations'/f'{team.lower().replace(" ","_")}_average_positions.png')
    # Tactical spread trend plotted for frames with enough simultaneous positions.
    fig,axes=plt.subplots(2,1,figsize=(12,7),sharex=True,facecolor='#0c1725')
    for ax,metric,title in zip(axes,('width_m_p10_p90','depth_m_p10_p90'),('Side-to-side spread / width','Long-axis spread / depth')):
        ax.set_facecolor('#122536')
        for team in CFG.TEAMS:
            g=per_frame[per_frame.team==team].sort_values('frame')
            if not g.empty: ax.plot(g.frame/fps,g[metric],color=PALETTE[team],label=team,lw=1.65,alpha=.92)
        ax.set_title(title,color='#eef5f7',loc='left',fontsize=11)
        ax.tick_params(colors='#d0dce0'); ax.grid(alpha=.15)
        ax.set_ylabel('metres',color='#c4d6db')
    if len(per_frame):
        axes[0].legend(facecolor='#122536',labelcolor='white')
    axes[-1].set_xlabel('Video time (seconds)',color='#d0dce0')
    fig.suptitle('FOOTBALLIQ  /  TEAM SPATIAL SHAPE',fontsize=15,color='white',weight='bold')
    fig.tight_layout()
    savefig(fig,root/'formations'/'team_spread_over_time.png')
    return per_frame,summary
