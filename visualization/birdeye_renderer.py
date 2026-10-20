import cv2
import numpy as np


class BirdEyeRenderer:
    def __init__(self, width=960, height=650, margin=45):
        self.width, self.height, self.margin = width, height, margin
        self.length, self.pitch_width = 105.0, 68.0
        self.sx = (width - 2 * margin) / self.length
        self.sy = (height - 2 * margin) / self.pitch_width
        self.base = self._pitch()

    def pixel(self, x, y):
        return (round(self.margin + x*self.sx),
                round(self.margin + y*self.sy))

    def _pitch(self):
        im = np.full((self.height, self.width, 3), (35, 150, 55), np.uint8)
        line = (225, 240, 225)
        P = self.pixel
        cv2.rectangle(im, P(0, 0), P(105, 68), line, 2)
        cv2.line(im, P(52.5, 0), P(52.5, 68), line, 2)
        cv2.circle(im, P(52.5,34), round(9.15 * self.sx), line, 2)
        cv2.circle(im, P(52.5,34), 3, line, -1)
        pa_top, pa_bottom = (68-40.32)/2, (68+40.32)/2
        ga_top, ga_bottom = (68-18.32)/2, (68+18.32)/2
        cv2.rectangle(im, P(0, pa_top), P(16.5,pa_bottom), line, 2)
        cv2.rectangle(im, P(88.5,pa_top), P(105,pa_bottom), line, 2)
        cv2.rectangle(im, P(0, ga_top), P(5.5,ga_bottom), line, 2)
        cv2.rectangle(im, P(99.5,ga_top), P(105,ga_bottom), line, 2)
        for x in (11,94):
            cv2.circle(im, P(x,34), 3, line, -1)
        # Four short corner arcs in pitch-metre aspect coordinates.
        for cx, cy, a1, a2 in [(0,0,0,90), (105,0,90,180),
                                (105,68,180,270), (0,68,270,360)]:
            cv2.ellipse(im, P(cx,cy), (round(self.sx),round(self.sy)),
                        0, a1, a2, line, 2)
        return im

    def render(self, players, ball=None, frame_index=0, fps=25):
        im = self.base.copy()
        for p in players:
            x, y = p['pitch_x'], p['pitch_y']
            if not 0 <= x <= self.length or not 0 <= y <= self.pitch_width:
                continue
            team = p.get('team', 'Unknown')
            color = ((45, 220, 255) if team == 'Team A'
                     else (45, 60, 235) if team == 'Team B'
                     else (170, 170, 170))
            center = self.pixel(x, y)
            cv2.circle(im, (center[0]+2, center[1]+3), 13, (45,75,45), -1)
            cv2.circle(im, center, 12, color, -1, cv2.LINE_AA)
            cv2.circle(im, center, 12, (245,245,245), 1, cv2.LINE_AA)
            label = 'GK' if p.get('class_name') == 'goalkeeper' else str(p.get('track_id', '?'))
            font_scale = .37 if len(label) <= 3 else .29
            size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)
            cv2.putText(im, label, (center[0]-size[0]//2,center[1]+size[1]//2),
                        cv2.FONT_HERSHEY_SIMPLEX,font_scale,(10,10,10),1,cv2.LINE_AA)
        if ball is not None:
            x, y = ball
            if 0 <= x <= self.length and 0 <= y <= self.pitch_width:
                pt = self.pixel(x,y)
                cv2.circle(im,pt,6,(245,245,245),-1,cv2.LINE_AA)
                cv2.circle(im,pt,6,(5,5,5),2,cv2.LINE_AA)
                cv2.circle(im,pt,2,(5,5,5),-1,cv2.LINE_AA)
        cv2.rectangle(im, (0,0), (self.width,32), (20,80,35), -1)
        cv2.putText(im,f'FootballIQ TACTICAL VIEW | {frame_index/max(fps,.01):.1f}s',
                    (14,23),cv2.FONT_HERSHEY_SIMPLEX,.55,(245,245,245),1,cv2.LINE_AA)
        return im
