
"""FootballIQ hybrid 22-player selection and conservative short-gap ID recovery.

Raw ByteTrack identities remain available in the per-frame diagnostics.
Stable IDs in the player CSV are estimates, NOT confirmed shirt numbers.
This module deliberately avoids merging uncertain identities.
"""
from collections import Counter, defaultdict
import math
import cv2
import numpy as np

MAX_ACTIVE = 22
MAX_PER_TEAM = 11
MAX_REID_GAP = 24
DUPLICATE_IOU = 0.76
MIN_REID_APPEARANCE = 0.89


def _foot(obj):
    x1, y1, x2, y2 = [float(v) for v in obj['bbox']]
    return np.array([(x1 + x2) * .5, y2], dtype=float)


def _iou(a, b):
    ax1, ay1, ax2, ay2 = map(float, a['bbox'])
    bx1, by1, bx2, by2 = map(float, b['bbox'])
    w = max(0, min(ax2, bx2) - max(ax1, bx1))
    h = max(0, min(ay2, by2) - max(ay1, by1))
    union = max(1., (ax2-ax1)*(ay2-ay1)+(bx2-bx1)*(by2-by1)-w*h)
    return w*h/union


def _appearance(frame, obj):
    x1, y1, x2, y2 = map(float, obj['bbox'])
    h, w = frame.shape[:2]
    bw, bh = x2-x1, y2-y1
    if bw < 8 or bh < 16:
        return None
    a = max(0, int(x1+.25*bw)); b = min(w, int(x1+.75*bw))
    c = max(0, int(y1+.23*bh)); d = min(h, int(y1+.55*bh))
    if b-a < 4 or d-c < 4:
        return None
    image = cv2.resize(frame[c:d,a:b], (24,24), interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0,1], None, [18,8], [0,180,0,256])
    cv2.normalize(hist, hist, alpha=1.0, norm_type=cv2.NORM_L1)
    return hist


def _similarity(a, b):
    if a is None or b is None:
        return 0.0
    return max(0.0, min(1.0, 1.0 - cv2.compareHist(a,b,cv2.HISTCMP_BHATTACHARYYA)))


def _team_compatible(old, new):
    return old == 'Unknown' or new == 'Unknown' or old == new


class ActivePlayerManager:
    """Select visible players; re-ID only with short gap + unambiguous evidence."""
    def __init__(self, fps=25.):
        self.fps = max(1., float(fps))
        self.raw_to_stable = {}
        self.last = {}
        self.raw_age = Counter()
        self.next_id = 1
        self.seen_raw = set()
        self.diagnostic_totals = Counter()

    def _candidate(self, obj, frame, assigned, occupied):
        raw = int(obj['track_id'])
        stable = self.raw_to_stable.get(raw)
        # A stable ID cannot represent two simultaneously visible players.
        if stable is not None and stable not in occupied:
            return stable, False
        if stable is not None:
            self.raw_to_stable.pop(raw, None)
        position = _foot(obj)
        desc = obj['_appearance']
        choices = []
        for sid, info in self.last.items():
            if sid in occupied or sid in assigned:
                continue
            gap = frame-info['frame']
            if not (1 <= gap <= min(MAX_REID_GAP, max(3, int(self.fps*.8)))):
                continue
            if info['raw'] in self.seen_raw:
                continue
            if not _team_compatible(info['team'], obj['team']):
                continue
            if (info['role'] == 'goalkeeper') != (obj['class_name'] == 'goalkeeper'):
                continue
            h = max(15, float(obj['bbox'][3]-obj['bbox'][1]))
            dist = float(np.linalg.norm(position - info['foot']))
            maximum = min(85., max(20., .40*h) + gap*.09*h)
            if dist > maximum:
                continue
            sim = _similarity(desc, info['appearance'])
            if sim < MIN_REID_APPEARANCE:
                continue
            # Require geometry and appearance together; two kit-matched teammates
            # in different locations must not share an identity.
            score = .55*sim + .45*(1 - dist/maximum)
            choices.append((score, sid))
        choices.sort(reverse=True)
        if choices and (len(choices)==1 or choices[0][0]-choices[1][0]>=.09):
            return choices[0][1], True
        stable = self.next_id
        self.next_id += 1
        return stable, False

    def select(self, frame_index, candidates, frame):
        # Duplicate same-ID observations can occur if upstream changes.
        candidates = [dict(obj) for obj in candidates if obj.get('class_name') in ('player','goalkeeper')]
        self.seen_raw = {int(obj['track_id']) for obj in candidates}
        for obj in candidates:
            raw = int(obj['track_id'])
            self.raw_age[raw] += 1
            obj['_appearance'] = _appearance(frame, obj)
        # Prefer established tracks, then confident detections. Still allow
        # new tracks if fewer than 22 reliable tracks have been observed.
        candidates.sort(key=lambda o: (
            int(o['track_id']) in self.raw_to_stable,
            min(12, self.raw_age[int(o['track_id'])]),
            float(o.get('confidence', 0)),
        ), reverse=True)
        unique = []
        duplicates = 0
        for obj in candidates:
            if any(_iou(obj, prior)>=DUPLICATE_IOU for prior in unique):
                duplicates += 1
                continue
            unique.append(obj)
        selected = []
        teams = Counter()
        assigned = set()
        occupied = {self.raw_to_stable[int(o['track_id'])] for o in unique
                    if int(o['track_id']) in self.raw_to_stable}
        # Process known established tracks first; reserve their IDs while matching.
        reids = 0
        for obj in unique:
            team = str(obj.get('team','Unknown'))
            if len(selected) >= MAX_ACTIVE or (team != 'Unknown' and teams[team] >= MAX_PER_TEAM):
                continue
            raw = int(obj['track_id'])
            known_id = self.raw_to_stable.get(raw)
            if known_id is not None and known_id not in assigned:
                sid = known_id
                recovered = False
            else:
                sid, recovered = self._candidate(obj, frame_index, assigned, occupied)
            if sid in assigned:
                continue
            if recovered:
                reids += 1
            self.raw_to_stable[raw] = sid
            assigned.add(sid)
            original = raw
            obj['track_id'] = sid
            obj['raw_track_id'] = original
            self.last[sid] = dict(frame=frame_index, foot=_foot(obj), team=team,
                                  appearance=obj['_appearance'],raw=original,
                                  role=obj['class_name'])
            obj.pop('_appearance', None)
            selected.append(obj)
            if team != 'Unknown':
                teams[team] += 1
        # Don't let an old dormant raw ID be permanently mapped to an identity
        # that a new raw ID has inherited.
        active_raw = {item['raw_track_id'] for item in selected}
        for raw, sid in list(self.raw_to_stable.items()):
            if raw not in active_raw and self.last.get(sid, {}).get('raw') != raw:
                self.raw_to_stable.pop(raw, None)
        if frame_index % 200 == 0:
            for sid, info in list(self.last.items()):
                if frame_index-info['frame'] > max(150,int(self.fps*6)):
                    self.last.pop(sid, None)
            for raw in list(self.raw_age):
                if raw not in self.seen_raw and raw not in self.raw_to_stable:
                    self.raw_age.pop(raw, None)
        diagnostic = dict(raw_tracks=len(candidates), distinct_candidates=len(unique),
                          duplicate_tracks_removed=duplicates, active_players=len(selected),
                          cap_or_team_filtered=len(unique)-len(selected), reid_matches=reids,
                          active_team_a=teams.get('Team A', 0),active_team_b=teams.get('Team B', 0),
                          active_unknown=sum(o['team']=='Unknown' for o in selected))
        self.diagnostic_totals.update({k:v for k,v in diagnostic.items()
                                       if k in ('duplicate_tracks_removed','cap_or_team_filtered','reid_matches')})
        return selected, diagnostic


def finalize_refined_tracks():
    """Clean the second-pass CSV *before* heatmaps, statistics and rendering."""
    from pathlib import Path
    import pandas as pd
    import offline_settings as S
    path=Path(S.OUT_CSV)
    if not path.is_file():
        raise FileNotFoundError(f'Cannot finalize missing coordinates: {path}')
    data=pd.read_csv(path)
    if data.empty:
        raise RuntimeError('No refined player positions to finalize')
    required={'frame','track_id','team','confidence','is_observed','pitch_x_m','pitch_y_m'}
    if not required.issubset(data.columns):
        raise RuntimeError('Unexpected refined coordinates schema: '+str(required-set(data.columns)))
    # First remove implausibly brief tracks. A real cameo is not permanently
    # erased: the complete source is kept in the .pre_fi_final.csv backup.
    observed=data.loc[data['is_observed'].astype(str).isin(['1','1.0','True'])]
    evidence=observed.groupby('track_id').size()
    minimum=6  # six measured positions is short enough for brief match footage
    strong_ids=set(evidence[evidence>=minimum].index)
    # If this would remove everything (e.g. 4-frame test footage), preserve it.
    if strong_ids:
        data=data[data.track_id.isin(strong_ids)].copy()
    # For duplicate same-frame IDs keep observed, high confidence rows.
    data['_observed_rank']=pd.to_numeric(data['is_observed'],errors='coerce').fillna(0)
    data['_confidence_rank']=pd.to_numeric(data['confidence'],errors='coerce').fillna(0)
    data=data.sort_values(['frame','_observed_rank','_confidence_rank'],ascending=[True,False,False])
    data=data.drop_duplicates(['frame','track_id'],keep='first')
    kept=[]
    summary=Counter()
    for frame, rows in data.groupby('frame',sort=True):
        accepted=[]
        teams=Counter()
        for row in rows.to_dict('records'):
            team=str(row.get('team','Unknown'))
            if len(accepted)>=MAX_ACTIVE or (team in ('Team A','Team B') and teams[team]>=MAX_PER_TEAM):
                summary['capacity_filtered']+=1
                continue
            # Conservative overlap removal: same pitch spot, one is an estimate.
            duplicate=False
            for previous in accepted:
                dist=math.hypot(row['pitch_x_m']-previous['pitch_x_m'],row['pitch_y_m']-previous['pitch_y_m'])
                if (dist < .35 and row['track_id']!=previous['track_id']
                    and (row['_observed_rank']==0 or previous['_observed_rank']==0)):
                    duplicate=True;break
            if duplicate:
                summary['overlap_filtered']+=1
                continue
            accepted.append(row)
            if team in ('Team A','Team B'):
                teams[team]+=1
        kept.extend(accepted)
        summary['max_simultaneous']=max(summary['max_simultaneous'],len(accepted))
    result=pd.DataFrame(kept).drop(columns=['_observed_rank','_confidence_rank'])
    backup=path.with_name(path.stem+'.pre_fi_final.csv')
    # If rerun on an existing output, don't overwrite its original backup.
    if not backup.exists():
        data_source=pd.read_csv(path)
        data_source.to_csv(backup,index=False)
    temp=path.with_suffix('.fi_tmp.csv')
    result.to_csv(temp,index=False,float_format='%.4f')
    temp.replace(path)
    print(f'FootballIQ final tracks: {len(result)} rows, {result.track_id.nunique()} identities, '
          f'max {summary["max_simultaneous"]} visible per frame')
    print('Removed estimated duplicates:',summary['overlap_filtered'],
          '| excess player rows:',summary['capacity_filtered'])
    print('Previous refined coordinates saved:', backup)
    return summary


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser(description='Finalize FootballIQ refined tracks')
    parser.add_argument('--clip', default=None)
    args=parser.parse_args()
    if args.clip:
        from web_pipeline_entry import configure
        configure(args.clip)
    finalize_refined_tracks()
