"""Noise-robust *tracklet* movement statistics from observed calibrated feet."""
import numpy as np
import pandas as pd
import analytics_settings as CFG


def _track_stats(group, fps):
    obs=group[group.is_observed].sort_values('frame') if CFG.MOVEMENT_OBSERVED_ONLY else group.sort_values('frame')
    obs=obs.drop_duplicates('frame')
    required=max(2,int(round(CFG.MOVEMENT_INTERVAL_S * fps)))
    max_frames=max(required+1,int(round(CFG.MOVEMENT_MAX_GAP_S * fps)))
    points=obs[['frame','pitch_x_m','pitch_y_m']].to_numpy(dtype=float)
    # Fixed-duration displacement estimates mitigate noise in pitch homography.
    # Nonoverlapping movement intervals prevent double counting.
    distances=[]; speeds=[]; valid_duration=0.; rejected=0
    idx=0
    while idx<len(points)-1:
        start=int(points[idx,0])
        # closest endpoint >= baseline; reject when occluded for too long
        j=int(np.searchsorted(points[:,0],start+required,side='left'))
        if j>=len(points): break
        delta_frames=points[j,0]-points[idx,0]
        if delta_frames>max_frames:
            idx+=1; continue
        secs=float(delta_frames/fps)
        d=float(np.hypot(*(points[j,1:3]-points[idx,1:3])))
        speed=d/secs
        if speed>CFG.MOVEMENT_MAX_SPEED_MPS:
            rejected+=1; idx=j; continue
        d=max(0.,d-CFG.MOVEMENT_JITTER_DEADZONE_M)
        speeds.append(d/secs); distances.append(d)
        valid_duration+=secs
        idx=j
    samples=np.array(speeds)
    return {
        'observed_frames':int(len(obs)),
        'estimated_or_missing_rows':int(len(group)-len(obs)),
        'track_span_s':round((points[-1,0]-points[0,0])/fps,2) if len(points)>1 else 0.,
        'measured_movement_time_s':round(valid_duration,2),
        'distance_m_est':round(float(sum(distances)),2),
        'avg_moving_speed_kmh_est':round(float(np.mean(samples[samples>=0.6])*3.6),2) if (samples>=0.6).any() else 0.,
        'peak_sustained_speed_kmh_est':round(float(np.percentile(samples,95)*3.6),2) if len(samples) else 0.,
        'walking_time_s_est':round(float((samples[(samples>=0.1)&(samples<2.0)].size)*CFG.MOVEMENT_INTERVAL_S),2),
        'jogging_time_s_est':round(float((samples[(samples>=2.0)&(samples<5.5)].size)*CFG.MOVEMENT_INTERVAL_S),2),
        'running_time_s_est':round(float((samples[samples>=5.5].size)*CFG.MOVEMENT_INTERVAL_S),2),
        'valid_movement_segments':len(distances),
        'rejected_jump_segments':rejected,
    }


def analyze(data, fps):
    out=[]
    for tid,g in data.groupby('track_id',sort=True):
        team=g.team.iloc[0]
        label='goalkeeper' if (g.class_name=='goalkeeper').sum()>len(g)/2 else 'player'
        r={'track_id':int(tid),'team':team,'role':label,'first_frame':int(g.frame.min()), 'last_frame':int(g.frame.max())}
        r.update(_track_stats(g,fps))
        out.append(r)
    tracks=pd.DataFrame(out).sort_values(['team','distance_m_est'],ascending=[True,False])
    rows=[]
    for team in CFG.TEAMS:
        g=tracks[tracks.team==team]
        rows.append({
            'team':team,'tracklets':len(g),'distance_m_est_sum_tracklets':round(float(g.distance_m_est.sum()),2),
            'mean_peak_sustained_kmh_est':round(float(g.loc[g.valid_movement_segments>=2,'peak_sustained_speed_kmh_est'].mean()),2) if (g.valid_movement_segments>=2).any() else None,
            'movement_time_s_sum_tracklets':round(float(g.measured_movement_time_s.sum()),2),
            'observed_frames_sum_tracklets':int(g.observed_frames.sum()),
            'jump_segments_rejected':int(g.rejected_jump_segments.sum()),
        })
    return tracks,pd.DataFrame(rows)
