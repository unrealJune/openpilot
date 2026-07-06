#!/usr/bin/env python3
"""Honda/Acura Bosch-A radar azimuth calibration helper. Runs ON the comma3x (SSH), car ignition ON.

The dealer procedure aims the radar against a target board; we have a better reference -- the vision
model. Two modes:

FOLLOW mode (default; passenger reads the screen while someone drives):
    Accumulates (radar track <-> vision lead) pairs exactly like the offline roadtrip regression
    (unique range match, vRel agreement, confident vision) and continuously fits azimuth scale and
    boresight on the laterally-excited subset. Drive behind a car and make gentle lane changes /
    sweeping curves until the excited-pair count and the fit converge. Ctrl+C saves pairs to
    /data/radar_cal_pairs.npz and prints the final numbers to feed back into
    opendbc BOSCH_RADAR_LAT_SCALE_DEG_PER_LSB (and boresight, if ever non-negligible).

TARGET mode (--target; parked, "big sheet" style):
    Park on level ground facing a strong reflector (corner reflector, metal pole, or a parked car's
    bumper) 10-30 m ahead with nothing else nearby. The app averages the nearest track's azimuth at
    each position you confirm (center, then measured lateral offsets) and solves boresight + a
    near-range scale check. A tape measure and a helper moving the target are all you need.

Usage on device:
    cd /data/openpilot && PYTHONPATH=. /usr/local/venv/bin/python selfdrive/debug/honda_radar_cal.py
    ... --target   for the parked procedure
"""
import argparse
import math
import os
import sys
import time

import numpy as np

from cereal import messaging

# must match the shipped parser so raw azimuth counts can be recovered from published yRel
LAT_SCALE_SHIPPED = 0.001186   # deg/LSB (opendbc BOSCH_RADAR_LAT_SCALE_DEG_PER_LSB)
CAM_OFF = 2.2                  # m, radar<->camera longitudinal offset (RADAR_TO_CAMERA_HONDA_BOSCH_A)
PAIRS_PATH = "/data/radar_cal_pairs.npz"

EXCITED_Y = 1.0                # m, |vision y| for a pair to constrain the fit
MATCH_RANGE = 2.5              # m, unique range match gate
MATCH_VREL = 3.0               # m/s, vRel agreement gate


def recover_raw(y_rel: float, d_rel: float) -> float:
  """Invert the shipped projection yRel = +dRel*sin(deg2rad(k*raw)) back to raw LSB counts."""
  s = max(min(y_rel / max(d_rel, 1e-3), 1.0), -1.0)
  return math.degrees(math.asin(s)) / LAT_SCALE_SHIPPED


def bracket_fit(x: np.ndarray, y: np.ndarray):
  """OLS both directions brackets the slope under noise-in-x; returns (slope_gm, c0, lo, hi)."""
  xm, ym = x.mean(), y.mean()
  xx = x - xm; yy = y - ym
  sxx = float(xx @ xx); syy = float(yy @ yy); sxy = float(xx @ yy)
  if sxx <= 0 or syy <= 0 or sxy == 0:
    return float('nan'), float('nan'), float('nan'), float('nan')
  b_yx = sxy / sxx
  b_xy = sxy / syy
  gm = math.copysign(math.sqrt(abs(b_yx * (1.0 / b_xy))), b_yx)
  c0 = xm - ym / gm
  return gm, c0, b_yx, 1.0 / b_xy


def fit_report(raws, drels, yvs):
  raws = np.asarray(raws); drels = np.asarray(drels); yvs = np.asarray(yvs)
  exc = np.abs(yvs) >= EXCITED_Y
  n_exc = int(exc.sum())
  out = dict(n=len(yvs), n_exc=n_exc, k=float('nan'), c0=float('nan'),
             k_lo=float('nan'), k_hi=float('nan'), rms=float('nan'))
  if n_exc < 40:
    return out
  u = raws[exc]; ang = yvs[exc] / drels[exc]
  gm, c0, lo, hi = bracket_fit(u, ang)
  # vision angle = gm*(raw - c0); parser needs yRel = -vision_y => k_parser = -gm (deg after conversion)
  out["k"] = -gm * 180 / math.pi
  out["c0"] = c0
  out["k_lo"] = -hi * 180 / math.pi
  out["k_hi"] = -lo * 180 / math.pi
  pred = gm * (u - c0) * drels[exc]
  out["rms"] = float(np.sqrt(np.mean((yvs[exc] - pred) ** 2)))
  return out


def follow_mode():
  sm = messaging.SubMaster(['liveTracks', 'modelV2', 'carState'])
  raws, drels, yvs, ts = [], [], [], []
  if os.path.exists(PAIRS_PATH):
    old = np.load(PAIRS_PATH)
    raws, drels, yvs, ts = [list(old[k]) for k in ("raws", "drels", "yvs", "ts")]
    print(f"resuming with {len(raws)} saved pairs from {PAIRS_PATH}")
  last_draw = 0.0
  try:
    while True:
      sm.update(100)
      if not (sm.updated['modelV2'] and sm.valid['liveTracks']):
        continue
      leads = sm['modelV2'].leadsV3
      if len(leads) == 0 or len(leads[0].x) == 0:
        continue
      L = leads[0]
      if L.prob < 0.8 or L.xStd[0] > 4.0:
        continue
      v_ego = sm['carState'].vEgo
      want = L.x[0] - CAM_OFF
      close = [p for p in sm['liveTracks'].points if abs(p.dRel - want) < MATCH_RANGE]
      if len(close) != 1 or close[0].dRel < 5.0:
        continue
      p = close[0]
      if abs((p.vRel + v_ego) - L.v[0]) > MATCH_VREL:
        continue
      raws.append(recover_raw(p.yRel, p.dRel))
      drels.append(p.dRel)
      yvs.append(L.y[0])
      ts.append(time.monotonic())
      now = time.monotonic()
      if now - last_draw > 1.0:
        last_draw = now
        r = fit_report(raws, drels, yvs)
        bar_y = yvs[-1]
        bar = " " * 20
        pos = int(max(min(bar_y, 5.0), -5.0) / 5.0 * 10) + 10
        bar = bar[:pos] + "#" + bar[pos + 1:]
        sys.stdout.write("\x1b[2J\x1b[H")
        print("=== Bosch-A radar azimuth calibration -- FOLLOW mode ===")
        print("Drive behind a car; make gentle lane changes and sweeping curves.\n")
        print(f"pairs: {r['n']}   excited (|y|>={EXCITED_Y} m): {r['n_exc']}   (want 300+ excited)")
        print(f"lead now: d={drels[-1]:5.1f} m  y_vis={bar_y:+5.2f} m  [-5m {bar} +5m]")
        if not math.isnan(r["k"]):
          print(f"\nfit:  k = {r['k']:.6f} deg/LSB   bracket [{r['k_lo']:.6f}, {r['k_hi']:.6f}]")
          print(f"      boresight = {r['c0']:+5.0f} LSB ({r['c0'] * abs(r['k']):+.3f} deg)")
          print(f"      excited residual RMS = {r['rms']:.2f} m")
          print(f"\nshipped k = {LAT_SCALE_SHIPPED}; update opendbc BOSCH_RADAR_LAT_SCALE_DEG_PER_LSB")
          print("if the converged fit disagrees beyond its bracket.")
        else:
          print("\nfit: need more lateral excitation (lane changes) ...")
        print("\nCtrl+C to stop and save.")
  except KeyboardInterrupt:
    pass
  np.savez(PAIRS_PATH, raws=np.array(raws), drels=np.array(drels), yvs=np.array(yvs), ts=np.array(ts))
  r = fit_report(raws, drels, yvs)
  print(f"\nsaved {len(raws)} pairs -> {PAIRS_PATH}")
  print(f"FINAL: k = {r['k']:.6f} deg/LSB  bracket [{r['k_lo']:.6f},{r['k_hi']:.6f}]  "
        f"boresight {r['c0']:+.0f} LSB  excited n={r['n_exc']}  RMS {r['rms']:.2f} m")


def target_mode():
  sm = messaging.SubMaster(['liveTracks'])
  print("=== Bosch-A radar azimuth calibration -- TARGET mode ===")
  print("Park facing a strong reflector 10-30 m ahead, nothing else nearby.")
  print("At each prompt: place the target at the stated LATERAL offset from the car's centerline")
  print("(tape-measure; + = LEFT as seen from the driver seat), keep the same distance, press Enter.\n")
  positions = []
  for label, y_true in (("centered (0.0 m)", 0.0), ("1.0 m LEFT", 1.0), ("1.0 m RIGHT", -1.0)):
    input(f"--> target {label}; press Enter to sample...")
    samples = []
    t_end = time.monotonic() + 5.0
    while time.monotonic() < t_end:
      sm.update(100)
      pts = sm['liveTracks'].points
      if not pts:
        continue
      p = min(pts, key=lambda q: q.dRel)   # nearest track = the target
      if p.dRel < 5.0 or p.dRel > 40.0:
        continue
      samples.append((recover_raw(p.yRel, p.dRel), p.dRel))
    if len(samples) < 10:
      print(f"    !! only {len(samples)} samples -- is the target visible / ignition on? skipping")
      continue
    arr = np.array(samples)
    raw_med = float(np.median(arr[:, 0])); d_med = float(np.median(arr[:, 1]))
    print(f"    raw azimuth median {raw_med:+.0f} LSB at {d_med:.1f} m ({len(samples)} samples)")
    positions.append((y_true, raw_med, d_med))
  if len(positions) >= 2:
    # vision-frame angle target: radard yRel = -lead.y => target angle for raw is -(y_true)/d
    ys = np.array([-(y / d) for y, _, d in positions])       # target sin(angle) ~ angle
    us = np.array([r for _, r, _ in positions])
    A = np.vstack([us, np.ones(len(us))]).T
    (a, b), *_ = np.linalg.lstsq(A, ys, rcond=None)
    k = a * 180 / math.pi
    c0 = -b / a if a else float('nan')
    print(f"\nRESULT: scale k = {k:.6f} deg/LSB (shipped {LAT_SCALE_SHIPPED})")
    print(f"        boresight = {c0:+.0f} LSB = {c0 * k:+.3f} deg")
    print("Boresight is the number the dealer sheet procedure pins; apply it in opendbc if |deg| > 0.15.")
  elif len(positions) == 1:
    y, r, d = positions[0]
    if y == 0.0:
      c0 = r
      print(f"\nRESULT (center-only): boresight = {c0:+.0f} LSB = {c0 * LAT_SCALE_SHIPPED:+.3f} deg")
  else:
    print("no usable positions; nothing to report")


if __name__ == "__main__":
  ap = argparse.ArgumentParser()
  ap.add_argument("--target", action="store_true", help="parked target-board procedure (boresight)")
  args = ap.parse_args()
  target_mode() if args.target else follow_mode()
