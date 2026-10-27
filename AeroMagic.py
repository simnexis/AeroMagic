"""AeroMagic — hand-tracked mystic spheres.
Install: python -m pip install opencv-python mediapipe numpy
Controls: open palm activates a sphere; Q/Esc exits.
"""
import math, time, random
from functools import lru_cache
import cv2
import numpy as np
import mediapipe as mp

CAMERA_INDEX = 0
CAMERA_W, CAMERA_H = 720, 420
CANVAS = 720
GOLD = (8, 112, 255)         # BGR: rich amber-gold
PALE_GOLD = (35, 185, 255)
WHITE_GOLD = (190, 235, 255)
CRIMSON = (105, 55, 255)     # warm rose-magenta accent


def clamp(x, lo, hi): return max(lo, min(hi, x))


def hand_anchor(landmarks, width, height):
    """Return the palm center in image pixels, not the fingertips."""
    # Wrist plus the four finger MCP joints gives a stable palm anchor.
    ids = (0, 5, 9, 13, 17)
    x = sum(landmarks[i].x for i in ids) / len(ids)
    y = sum(landmarks[i].y for i in ids) / len(ids)
    return (int(clamp(x * width, 0, width - 1)),
            int(clamp(y * height, 0, height - 1)))



def is_open_palm(landmarks):
    """Detect an open palm using fingertip-to-wrist distances and finger extension."""
    wrist = landmarks[0]
    # Fingertip should be farther from the wrist than its PIP joint.
    finger_pairs = ((8, 6), (12, 10), (16, 14), (20, 18))
    extended = 0
    for tip_id, pip_id in finger_pairs:
        tip = landmarks[tip_id]
        pip = landmarks[pip_id]
        tip_dist = math.hypot(tip.x - wrist.x, tip.y - wrist.y)
        pip_dist = math.hypot(pip.x - wrist.x, pip.y - wrist.y)
        if tip_dist > pip_dist * 1.12:
            extended += 1
    # Thumb is checked separately because its direction differs from the fingers.
    thumb_tip, thumb_ip = landmarks[4], landmarks[3]
    thumb_extended = math.hypot(thumb_tip.x-wrist.x, thumb_tip.y-wrist.y) > math.hypot(thumb_ip.x-wrist.x, thumb_ip.y-wrist.y) * 1.05
    return extended >= 3 and thumb_extended



def is_palm_facing_camera(landmarks, handedness_label):
    """Estimate whether the palm side faces the camera from 3D landmarks."""
    wrist, index_mcp, pinky_mcp = landmarks[0], landmarks[5], landmarks[17]
    ax, ay, az = index_mcp.x-wrist.x, index_mcp.y-wrist.y, index_mcp.z-wrist.z
    bx, by, bz = pinky_mcp.x-wrist.x, pinky_mcp.y-wrist.y, pinky_mcp.z-wrist.z
    nx = ay*bz - az*by
    ny = az*bx - ax*bz
    nz = ax*by - ay*bx
    norm = math.sqrt(nx*nx + ny*ny + nz*nz)
    if norm < 1e-6:
        return False
    # Handedness normalization for the mirrored selfie-camera view.
    expected_sign = 1.0 if handedness_label == 'Right' else -1.0
    return (nz / norm) * expected_sign > 0.22


def make_sigil(size=CANVAS):
    """Build the expensive, detailed spell geometry once, then reuse it."""
    img = np.zeros((size,size,3), np.uint8)
    c = size//2
    def ring(r, color=GOLD, thick=1): cv2.circle(img,(c,c),int(r),color,thick,cv2.LINE_AA)
    # concentric seals and fine engraved tracks
    for r,th in [(0.475,2),(0.458,1),(0.425,1),(0.405,2),(0.365,1),(0.345,2),(0.285,2),(0.255,1),(0.205,2),(0.145,2),(0.112,1),(0.072,2)]:
        ring(size*r, GOLD if th==2 else (0,105,175), th)
    # 16 radial dividers and small node marks
    for i in range(32):
        a=math.tau*i/32
        r0=size*(0.375 if i%2==0 else 0.397); r1=size*0.418
        p0=(int(c+math.cos(a)*r0),int(c+math.sin(a)*r0)); p1=(int(c+math.cos(a)*r1),int(c+math.sin(a)*r1))
        cv2.line(img,p0,p1,GOLD if i%2==0 else (0,125,190),1,cv2.LINE_AA)
        if i%2==0:
            x=int(c+math.cos(a)*size*.445); y=int(c+math.sin(a)*size*.445)
            cv2.circle(img,(x,y),2,PALE_GOLD,-1,cv2.LINE_AA)
    # nested polygonal geometry
    for sides, radius, phase, thick in [(8,.32,math.pi/8,2),(8,.32,0,1),(4,.19,math.pi/4,2),(4,.19,0,1),(8,.125,math.pi/8,1)]:
        pts=[]
        for i in range(sides):
            a=math.tau*i/sides+phase
            pts.append((int(c+math.cos(a)*size*radius),int(c+math.sin(a)*size*radius)))
        cv2.polylines(img,[np.array(pts,np.int32)],True,GOLD,thick,cv2.LINE_AA)
    # pointed sigils around inner octagon
    for i in range(8):
        a=math.tau*i/8
        r=size*.30
        tip=(int(c+math.cos(a)*r),int(c+math.sin(a)*r))
        side1=(int(c+math.cos(a+.19)*size*.265),int(c+math.sin(a+.19)*size*.265))
        side2=(int(c+math.cos(a-.19)*size*.265),int(c+math.sin(a-.19)*size*.265))
        cv2.polylines(img,[np.array([side1,tip,side2],np.int32)],False,PALE_GOLD,2,cv2.LINE_AA)
    # outer glyph ring: procedural angular rune marks, varied per segment
    rng=random.Random(42)
    for i in range(112):
        a=math.tau*i/112
        r=size*.438
        x=int(c+math.cos(a)*r); y=int(c+math.sin(a)*r)
        tang=np.array([-math.sin(a),math.cos(a)])
        radial=np.array([math.cos(a),math.sin(a)])
        length=rng.randint(5,13); width=rng.randint(3,8)
        p1=(int(x+tang[0]*length),int(y+tang[1]*length))
        p2=(int(x-radial[0]*width),int(y-radial[1]*width))
        p3=(int(x+radial[0]*width),int(y+radial[1]*width))
        if i%4==0:
            cv2.rectangle(img,(x-3,y-3),(x+3,y+3),GOLD,1,cv2.LINE_AA)
        elif i%3==0:
            cv2.polylines(img,[np.array([p2,p1,p3],np.int32)],False,PALE_GOLD,1,cv2.LINE_AA)
        else:
            cv2.line(img,(x,y),p1,GOLD,1,cv2.LINE_AA)
            cv2.circle(img,(x,y),1,WHITE_GOLD,-1,cv2.LINE_AA)
    # central focal geometry
    cv2.circle(img,(c,c),int(size*.045),PALE_GOLD,1,cv2.LINE_AA)
    cv2.circle(img,(c,c),int(size*.018),WHITE_GOLD,2,cv2.LINE_AA)
    return img

SIGIL = make_sigil()
SIGIL_GLOW = cv2.GaussianBlur(SIGIL, (0,0), 3.0)

@lru_cache(maxsize=160)
def scaled_assets(diameter, angle_bucket=0):
    """Cache resized art, glow, and radial reveal map for smooth animated rotation."""
    diameter=int(diameter)
    angle_bucket=int(angle_bucket) % 72
    angle=angle_bucket*5.0
    matrix=cv2.getRotationMatrix2D((CANVAS/2,CANVAS/2), angle, 1.0)
    rotated=cv2.warpAffine(SIGIL,matrix,(CANVAS,CANVAS),flags=cv2.INTER_LINEAR,
                           borderMode=cv2.BORDER_CONSTANT,borderValue=(0,0,0))
    rotated_glow=cv2.warpAffine(SIGIL_GLOW,matrix,(CANVAS,CANVAS),flags=cv2.INTER_LINEAR,
                                borderMode=cv2.BORDER_CONSTANT,borderValue=(0,0,0))
    interp=cv2.INTER_AREA if diameter<CANVAS else cv2.INTER_LINEAR
    art=cv2.resize(rotated,(diameter,diameter),interpolation=interp)
    glow=cv2.resize(rotated_glow,(diameter,diameter),interpolation=interp)
    glow=cv2.GaussianBlur(glow,(0,0),max(1.2,diameter*.012))
    yy,xx=np.ogrid[:diameter,:diameter]
    dist=np.sqrt((xx-diameter/2)**2+(yy-diameter/2)**2).astype(np.float32)
    return art,glow,dist

def overlay_sigil(frame, center, radius, progress, t, hand_seed=0):
    """Render the sigil inside a brighter, layered arcane energy sphere."""
    h, w = frame.shape[:2]
    cx, cy = map(int, center)
    r = int(radius)
    if r < 30:
        return
    diameter = int(clamp(2 * r, 80, 440))
    # Keep the complete outer rim inside the camera frame. Previously the
    # circle was centered directly on the palm, so it was clipped at the
    # top/left/right edges and looked like a broken outline.
    edge_margin = diameter // 2 + 5
    if w > 2 * edge_margin:
        cx = int(clamp(cx, edge_margin, w - edge_margin))
    else:
        cx = w // 2
    if h > 2 * edge_margin:
        cy = int(clamp(cy, edge_margin, h - edge_margin))
    else:
        cy = h // 2
    angle_bucket = int((t * (22 if hand_seed == 0 else -29)) + hand_seed * 13) % 72
    art, glow, dist = scaled_assets(diameter, angle_bucket)

    x0, y0 = cx - diameter // 2, cy - diameter // 2
    x1, y1 = x0 + diameter, y0 + diameter
    ax0, ay0, ax1, ay1 = max(0, x0), max(0, y0), min(w, x1), min(h, y1)
    if ax0 >= ax1 or ay0 >= ay1:
        return
    sx0, sy0 = ax0 - x0, ay0 - y0
    sx1, sy1 = sx0 + (ax1 - ax0), sy0 + (ay1 - ay0)
    roi = frame[ay0:ay1, ax0:ax1]
    art_roi = art[sy0:sy1, sx0:sx1]
    glow_roi = glow[sy0:sy1, sx0:sx1]
    dist_roi = dist[sy0:sy1, sx0:sx1]

    # Soft, translucent spherical volume: a radial amber core with a warm rim.
    yy, xx = np.ogrid[sy0:sy1, sx0:sx1]
    rr = np.sqrt((xx - diameter * 0.5) ** 2 + (yy - diameter * 0.5) ** 2)
    norm = np.clip(rr / max(1.0, diameter * 0.5), 0.0, 1.0)
    core = np.clip(1.0 - norm, 0.0, 1.0) ** 2
    rim = np.exp(-((norm - 0.91) / 0.045) ** 2)
    shimmer = 0.88 + 0.12 * math.sin(t * 2.6 + hand_seed * 1.7)
    sphere = np.zeros_like(roi)
    sphere[..., 0] = np.clip(core * 24 + rim * 58, 0, 255).astype(np.uint8)
    sphere[..., 1] = np.clip(core * 42 + rim * 72, 0, 255).astype(np.uint8)
    sphere[..., 2] = np.clip(core * 76 + rim * 132, 0, 255).astype(np.uint8)
    cv2.addWeighted(roi, 1.0, sphere, 0.42 * shimmer, 0, dst=roi)

    # Three-dimensional orbital bands, drawn as luminous elliptical rings.
    orbit_layer = np.zeros_like(frame)
    orbit_color = (18, 128, 255)
    orbit_color_hot = (150, 220, 255)
    orbit_r = max(30, int(diameter * 0.43))
    for i, (scale, tilt, speed, phase) in enumerate([
        (1.00, 0.28, 34, 0.0),
        (0.88, 0.56, -25, 1.8),
        (0.73, 0.82, 19, 3.2),
    ]):
        axes = (max(8, int(orbit_r * scale)), max(4, int(orbit_r * scale * tilt)))
        angle = (t * speed + phase * 57 + hand_seed * 23) % 360
        color = orbit_color_hot if i == 1 else orbit_color
        cv2.ellipse(orbit_layer, (cx, cy), axes, angle, 0, 360, color, 2 if i == 1 else 1, cv2.LINE_AA)
        # Fine rose-gold counter-orbit adds color separation without overpowering the sigil.
        if i == 0:
            cv2.ellipse(orbit_layer, (cx, cy), axes, (angle + 11) % 360, 22, 156, CRIMSON, 1, cv2.LINE_AA)
    # Keep orbital artwork inside the visible frame and add a soft halo.
    orbit_small = cv2.resize(orbit_layer, (max(1, w // 4), max(1, h // 4)), interpolation=cv2.INTER_AREA)
    orbit_blur = cv2.GaussianBlur(orbit_small, (0, 0), 2.2)
    orbit_blur = cv2.resize(orbit_blur, (w, h), interpolation=cv2.INTER_LINEAR)
    cv2.addWeighted(frame, 1.0, orbit_blur, 0.58, 0, dst=frame)
    cv2.add(frame, orbit_layer, dst=frame)

    # The original detailed mandala remains the visual centerpiece.
    reveal = clamp(progress, 0, 1) * diameter * 0.5
    mask = dist_roi <= reveal
    aura = cv2.convertScaleAbs(glow_roi, alpha=1.65)
    cv2.addWeighted(roi, 1.0, aura, 0.46, 0, dst=roi)
    if np.any(mask):
        visible = np.zeros_like(art_roi)
        visible[mask] = art_roi[mask]
        cv2.add(roi, visible, dst=roi)
    edge = np.abs(dist_roi - reveal) < max(2, diameter * 0.009)
    roi[edge] = np.maximum(roi[edge], np.array([35, 155, 255], dtype=np.uint8))

    # Small moving sparks and brighter nodes create a sense of energy circulating in 3D.
    for i in range(18):
        phase = (t * (0.22 + (i % 4) * 0.035) + i / 18.0 + hand_seed * 0.137) % 1.0
        ang = math.tau * phase
        orbit = r * (0.20 + 0.72 * ((phase + progress * 0.13) % 1.0))
        px = int(cx + math.cos(ang) * orbit)
        py = int(cy + math.sin(ang) * orbit * (0.62 + 0.12 * math.sin(t + i)))
        if 3 <= px < w - 3 and 3 <= py < h - 3:
            cv2.circle(frame, (px, py), 3 if i % 6 == 0 else 1, (20, 150, 255), -1, cv2.LINE_AA)
            if i % 6 == 0:
                cv2.circle(frame, (px, py), 5, (105, 55, 255), 1, cv2.LINE_AA)

def main():
    cap=cv2.VideoCapture(CAMERA_INDEX,cv2.CAP_DSHOW if hasattr(cv2,'CAP_DSHOW') else 0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,CAMERA_W); cap.set(cv2.CAP_PROP_FRAME_HEIGHT,CAMERA_H)
    cap.set(cv2.CAP_PROP_BUFFERSIZE,1)
    if not cap.isOpened():
        print('ERROR: Cannot open webcam. Close other camera apps or change CAMERA_INDEX.'); return

    hands_mod=mp.solutions.hands
    tracker=hands_mod.Hands(static_image_mode=False,max_num_hands=2,model_complexity=0,
                            min_detection_confidence=.48,min_tracking_confidence=.48)

    states={0:{'center':None,'radius':110.0,'charge':0.0,'last_seen':0.0,
               'open_frames':0,'closed_frames':0,'active':False},
            1:{'center':None,'radius':110.0,'charge':0.0,'last_seen':0.0,
               'open_frames':0,'closed_frames':0,'active':False}}

    last_t=time.perf_counter()
    rng=random.Random()
    try:
        while True:
            ok,frame=cap.read()
            if not ok: break
            frame=cv2.flip(frame,1)
            h,w=frame.shape[:2]
            now=time.perf_counter()
            dt=min(.05,now-last_t); last_t=now

            rgb=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB); rgb.flags.writeable=False
            result=tracker.process(rgb)
            detected=[]

            if result.multi_hand_landmarks:
                handedness=result.multi_handedness or []
                for idx,hand in enumerate(result.multi_hand_landmarks):
                    lm=hand.landmark
                    label='Right' if idx==0 else 'Left'
                    if idx<len(handedness) and handedness[idx].classification:
                        label=handedness[idx].classification[0].label
                    sid=0 if label=='Left' else 1
                    anchor=hand_anchor(lm,w,h)

                    palm_length=math.hypot(lm[9].x-lm[0].x,lm[9].y-lm[0].y)
                    palm_width=math.hypot(lm[5].x-lm[17].x,lm[5].y-lm[17].y)
                    palm_scale=math.sqrt(max(1e-6,palm_length*palm_width))*min(w,h)
                    target_radius=clamp(palm_scale*2.65,58,232)
                    detected.append((sid,lm,anchor,target_radius,label))

            seen=set()
            for sid,lm,anchor,target_radius,label in detected:
                st=states[sid]; seen.add(sid)
                st['last_seen']=now

                palm_ok=is_palm_facing_camera(lm,label)
                raw_open=is_open_palm(lm) and palm_ok
                if raw_open:
                    st['open_frames']=min(4,st['open_frames']+1)
                    st['closed_frames']=0
                    if st['open_frames']>=2:
                        st['active']=True
                else:
                    st['closed_frames']=min(8,st['closed_frames']+1)
                    st['open_frames']=0
                    if st['closed_frames']>=5:
                        st['active']=False

                if st['center'] is None:
                    st['center']=(float(anchor[0]),float(anchor[1]))
                pos_alpha=1.0-math.exp(-dt*16.0)
                size_delta=target_radius-st['radius']
                size_speed=15.0 if abs(size_delta)>18.0 else 6.5
                size_alpha=1.0-math.exp(-dt*size_speed)
                st['center']=(st['center'][0]+(anchor[0]-st['center'][0])*pos_alpha,
                              st['center'][1]+(anchor[1]-st['center'][1])*pos_alpha)
                st['radius']+=size_delta*size_alpha
                if st['active']:
                    st['charge']=min(1.0,st['charge']+dt/0.58)
                else:
                    st['charge']=max(0.0,st['charge']-dt*2.5)

            for sid,st in states.items():
                missing_for=now-st['last_seen'] if st['last_seen'] else 999.0
                if sid not in seen and missing_for>.20:
                    st['active']=False
                    st['open_frames']=0
                    st['closed_frames']=5
                    st['charge']=max(0.0,st['charge']-dt*2.2)
                if st['center'] is not None and st['charge']>.01:
                    overlay_sigil(frame,st['center'],st['radius'],st['charge'],now,sid)

            cv2.imshow('AETHERION | Mystic Interface',frame)
            key=cv2.waitKey(1)&0xFF
            if key in (ord('q'),27): break
    finally:
        tracker.close(); cap.release(); cv2.destroyAllWindows()

if __name__=='__main__': main()
