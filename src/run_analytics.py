"""Generate offline FootballIQ player + team analytics without rerunning YOLO.

Usage:
    # offline_settings.py: CLIP = 'bluevsred'
    python run_analytics.py
"""
import json
from pathlib import Path
import analytics_settings as C
from analytics.common import read_players, guess_fps
from analytics.movement_metrics import analyze as movement_analyze
from analytics.heatmaps import generate as heatmap_generate
from analytics.team_shape import analyze as shape_analyze
from analytics.dashboard import generate as dashboard_generate


def run():
    print(f'FootballIQ analytics / clip: {C.CLIP}')
    data=read_players(C.PLAYER_CSV)
    fps,fps_source=guess_fps(data,C.VIDEO)
    print(f'Loaded {len(data):,} pitch coordinates | FPS={fps:.3f} ({fps_source})')
    tracks,teams=movement_analyze(data,fps)
    root=Path(C.OUTPUT_ROOT); root.mkdir(parents=True,exist_ok=True)
    Path(C.CSV_ROOT).mkdir(parents=True,exist_ok=True)
    movement_csv=Path(C.CSV_ROOT)/f'{C.CLIP}_player_statistics.csv'
    team_csv=Path(C.CSV_ROOT)/f'{C.CLIP}_team_statistics.csv'
    shape_csv=Path(C.CSV_ROOT)/f'{C.CLIP}_team_shape_by_frame.csv'
    shape_summary_csv=Path(C.CSV_ROOT)/f'{C.CLIP}_team_shape_summary.csv'
    tracks.to_csv(movement_csv,index=False)
    teams.to_csv(team_csv,index=False)
    shape_frames,shape_teams=shape_analyze(data,fps,root)
    shape_frames.to_csv(shape_csv,index=False)
    shape_teams.to_csv(shape_summary_csv,index=False)
    print('Generating heatmaps...')
    heatmaps=heatmap_generate(data,root)
    print('Rendering dashboard...')
    dashboard_generate(tracks,teams,shape_teams,C.CLIP,fps,data,root)
    report={
        'clip':C.CLIP,'fps':round(fps,3),'fps_source':fps_source,
        'coordinate_rows':len(data),'tracklets':len(tracks),
        'observed_rows':int(data.is_observed.sum()),'estimated_rows':int((~data.is_observed).sum()),
        'team_stats':teams.to_dict(orient='records'),
        'shape_stats':json.loads(shape_teams.to_json(orient='records')),
        'player_heatmaps_created':len([p for p in heatmaps if 'players' in p.parts]),
        'warnings':[
            'Distances and speeds estimate visible tracklet movement, NOT full-match physical totals.',
            'Track IDs can split or switch between players; no automatic identity stitching is done.',
            'Only observed/smoothed positions contribute to movement; large gaps and unrealistic jumps are rejected.',
            'Heatmap intensities normalize independently; team colours do not encode numerical comparisons.',
            'Shape uses only frames with sufficient simultaneous observed outfield players; direction of attack is not assumed.',
            'Camera homography jitter, pitch calibration and short video duration may bias results.',
        ]}
    out_json=Path(C.CSV_ROOT)/f'{C.CLIP}_analytics_summary.json'
    out_json.write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf-8')
    print('\n=== Analytics finished ===')
    print(f'  Tracklets: {len(tracks)} | measured positions: {int(data.is_observed.sum()):,}')
    print(f'  Player heatmaps: {report["player_heatmaps_created"]}')
    print(f'  Statistics: {movement_csv}, {team_csv}')
    print(f'  Team shape: {shape_csv}, {shape_summary_csv}')
    print(f'  Charts: {root}')
    print(f'  Summary: {out_json}')
    print('NOTE: numbers cover observed portions of the clip, not verified player totals.')
    return report


if __name__=='__main__':
    run()
