#!/usr/bin/env python3
import math
import numpy as np
from collections import deque
from typing import Any

import capnp
from cereal import messaging, log, car, custom
from openpilot.common.filter_simple import FirstOrderFilter
from openpilot.common.params import Params
from openpilot.common.realtime import DT_MDL, Priority, config_realtime_process
from openpilot.common.swaglog import cloudlog
from openpilot.common.simple_kalman import KF1D

from opendbc.car import structs
from opendbc.car.honda.values import HONDA_BOSCH_A
from opendbc.car.hyundai.values import HyundaiFlags
from opendbc.sunnypilot.car.hyundai.values import HyundaiFlagsSP


# Default lead acceleration decay set to 50% at 1s
_LEAD_ACCEL_TAU = 1.5

# radar tracks
SPEED, ACCEL = 0, 1     # Kalman filter states enum

# stationary qualification parameters
V_EGO_STATIONARY = 4.   # no stationary object flag below this speed

RADAR_TO_CENTER = 2.7   # (deprecated) RADAR is ~ 2.7m ahead from center of car
RADAR_TO_CAMERA = 1.52  # RADAR is ~ 1.5m ahead from center of mesh frame
# Honda/Acura Bosch-A fine-track radar: the track range consistently reads ~1-3 m SHORTER than the
# vision lead distance minus the generic 1.52 m offset (measured on drives 00000005/00000007: median
# +1.1 to +3.3 m depending on range/scene). With the generic offset the true lead sits at the EDGE of
# the association distance gate at short range (offset eats 3+ m of the 5 m floor), costing fusion and
# adding switch flips. 2.2 m centers the gate for this platform (replay-validated; the residual
# scene-dependent bias is absorbed by the bumpless transfer below, not by chasing a per-drive constant).
RADAR_TO_CAMERA_HONDA_BOSCH_A = 2.2

# --- radar<->vision lead association (fusion) ---
# The vision lead's distance both selects and gates the radar track it fuses with. At long range the model's
# distance is noisy (swings >10m frame to frame) and the old gate was purely proportional (0.25 * dist, i.e.
# ~25m at 100m) with no memory, so a *different*, closer radar object could drift in and out of the gate and
# make the fused leadOne flip-flop between it and the vision-only lead. That injected phantom closing velocity
# (spurious braking / refusing to close a gap). Three guards fix this while leaving the near-range braking
# zone (where the 5m floor dominates) unchanged:
MATCH_VISION_DIST_GATE_MAX = 12.0   # m, cap on the association distance gate (reject far-off / different objects)
MATCH_VISION_DIST_RC = 0.3          # s, low-pass on the vision distance used for association (kills noise-driven toggling)
MATCH_SWITCH_FRAMES = 3             # a newly-matched radar track must persist this many frames before it becomes the lead

# Lateral sanity gate on the radar<->vision association. is_sane had NO lateral test at all: a radar
# return dead ahead could fuse to a vision lead 5 m off-path (wrong object -> phantom vLead collapse;
# observed drive 00000005 t=16.9 s), and an adjacent-lane track could fuse to the in-path lead. Track
# yRel convention is -lead.y. Range-scaled because this radar's azimuth is only trustworthy near-field
# (scale is calibrated at MEDIUM confidence; far tracks can sit meters off laterally), so far leads get a
# wider gate rather than losing fusion entirely. Replay-validated on drives 00000003/00000005/00000007:
# true-phantom closings 87 -> 24 with ~4 pts fused-coverage cost (radar defers to vision when unsure).
MATCH_VISION_LAT_GATE_BASE = 2.5    # m, lateral gate at/below LAT_GATE_START
MATCH_VISION_LAT_GATE_SLOPE = 0.05  # m of extra gate per meter of range beyond LAT_GATE_START
MATCH_VISION_LAT_GATE_START = 20.0  # m

# Bumpless source transfer for the published lead. The fused (radar) and vision-only estimates of the
# SAME physical lead disagree by 2-6 m in range (night vision range bias, radar reference point), so
# every radar<->vision source switch used to STEP dRel/vLead by that disagreement -- the planner sees a
# phantom gap change / velocity kick about once a second in dense traffic (slot churn), driving surging,
# phantom braking and follow-too-far. On a switch the discontinuity is captured as an offset and decayed
# to zero over ~0.4 s: the planner gets the new source's DYNAMICS immediately but never the DC step.
# A jump too large to be the same object is a REAL lead change and must step through unmodified.
# Replay-validated: source-switch dRel steps >2 m: 152 -> 18 across 7 segments.
BUMPLESS_DECAY = 0.85          # per 20 Hz frame; offset halves in ~4 frames, gone in ~0.4 s
BUMPLESS_MAX_STEP = 8.0        # m; a bigger discontinuity is a genuine lead change -> no smoothing

# --- native-Doppler false-closing guard ---
# Uses opendbc RadarPoint.vRelNative: the radar's OWN published relative velocity (Doppler) for the
# primary track, on cars that emit it (Honda/Acura Bosch-A attaches the 0x2C8 selected-lead Doppler to
# fine slot 0; NaN elsewhere). The range-DERIVED vRel occasionally reads a spurious hard closing (LSB
# chatter, or a closer object's range motion) -- the phantom-brake / follow-too-far driver. When the
# INDEPENDENT native Doppler AND the vision lead BOTH disagree with that closing, it is an artifact, so
# the fused lead's velocity is damped to the (still-conservative) corroborated value. Vision-gated: it
# can never mask a real closing that vision sees, and it no-ops when the native Doppler is unavailable.
NATIVE_DOPPLER_FALSE_CLOSE_MARGIN = 1.5  # m/s: track must close this much harder than BOTH corroborators

# --- vision closing floor (202605 vision-only parity on approach detection) ---
# The mirror image of the Doppler guard: the fused radar vRel can UNDER-report a real closing (KF spin-up
# on a freshly acquired track, or residual identity mixing in dense same-range scenes), which made the
# fused stack notice an approach 1-2 s LATER than vision-only did (replay, drives 00000003/00000007).
# When the confident vision lead claims meaningfully MORE closing than the fused track, adopt vision's
# estimate -- exactly the signal the vision-only stack would have published, so approach-detection can
# never be later than vision-only by construction. When radar sees MORE closing than vision (e.g. vision
# night-blind at range), radar still wins: this floor only ever makes the published lead MORE cautious.
VISION_CLOSING_FLOOR_MARGIN = 1.0  # m/s: vision must claim this much more closing before it overrides


class KalmanParams:
  def __init__(self, dt: float):
    # Lead Kalman Filter params, calculating K from A, C, Q, R requires the control library.
    # hardcoding a lookup table to compute K for values of radar_ts between 0.01s and 0.2s
    assert dt > .01 and dt < .2, "Radar time step must be between .01s and 0.2s"
    self.A = [[1.0, dt], [0.0, 1.0]]
    self.C = [1.0, 0.0]
    #Q = np.matrix([[10., 0.0], [0.0, 100.]])
    #R = 1e3
    #K = np.matrix([[ 0.05705578], [ 0.03073241]])
    dts = [i * 0.01 for i in range(1, 21)]
    K0 = [0.12287673, 0.14556536, 0.16522756, 0.18281627, 0.1988689,  0.21372394,
          0.22761098, 0.24069424, 0.253096,   0.26491023, 0.27621103, 0.28705801,
          0.29750003, 0.30757767, 0.31732515, 0.32677158, 0.33594201, 0.34485814,
          0.35353899, 0.36200124]
    K1 = [0.29666309, 0.29330885, 0.29042818, 0.28787125, 0.28555364, 0.28342219,
          0.28144091, 0.27958406, 0.27783249, 0.27617149, 0.27458948, 0.27307714,
          0.27162685, 0.27023228, 0.26888809, 0.26758976, 0.26633338, 0.26511557,
          0.26393339, 0.26278425]
    self.K = [[np.interp(dt, dts, K0)], [np.interp(dt, dts, K1)]]


class Track:
  def __init__(self, identifier: int, v_lead: float, kalman_params: KalmanParams):
    self.identifier = identifier
    self.cnt = 0
    self.aLeadTau = FirstOrderFilter(_LEAD_ACCEL_TAU, 0.45, DT_MDL)
    self.K_A = kalman_params.A
    self.K_C = kalman_params.C
    self.K_K = kalman_params.K
    self.kf = KF1D([[v_lead], [0.0]], self.K_A, self.K_C, self.K_K)
    self.vRelNative = float('nan')  # radar's native Doppler for this track (NaN if the radar doesn't publish one)

  def update(self, d_rel: float, y_rel: float, v_rel: float, v_lead: float, measured: float,
             v_rel_native: float = float('nan')):
    # relative values, copy
    self.dRel = d_rel   # LONG_DIST
    self.yRel = y_rel   # -LAT_DIST
    self.vRel = v_rel   # REL_SPEED
    self.vLead = v_lead
    self.measured = measured   # measured or estimate
    self.vRelNative = v_rel_native  # radar-published Doppler (RX-only cross-check; NaN when unavailable)

    # computed velocity and accelerations
    if self.cnt > 0:
      self.kf.update(self.vLead)

    self.vLeadK = float(self.kf.x[SPEED][0])
    self.aLeadK = float(self.kf.x[ACCEL][0])

    # Learn if constant acceleration
    if abs(self.aLeadK) < 0.5:
      self.aLeadTau.x = _LEAD_ACCEL_TAU
    else:
      self.aLeadTau.update(0.0)

    self.cnt += 1

  def get_RadarState(self, model_prob: float = 0.0):
    return {
      "dRel": float(self.dRel),
      "yRel": float(self.yRel),
      "vRel": float(self.vRel),
      "vLead": float(self.vLead),
      "vLeadK": float(self.vLeadK),
      "aLeadK": float(self.aLeadK),
      "aLeadTau": float(self.aLeadTau.x),
      "status": True,
      "fcw": self.is_potential_fcw(model_prob),
      "modelProb": model_prob,
      "radar": True,
      "radarTrackId": self.identifier,
    }

  def potential_low_speed_lead(self, v_ego: float):
    # stop for stuff in front of you and low speed, even without model confirmation
    # Radar points closer than 0.75, are almost always glitches on toyota radars
    return abs(self.yRel) < 1.0 and (v_ego < V_EGO_STATIONARY) and (0.75 < self.dRel < 25)

  def is_potential_fcw(self, model_prob: float):
    return model_prob > .9

  def __str__(self):
    ret = f"x: {self.dRel:4.1f}  y: {self.yRel:4.1f}  v: {self.vRel:4.1f}  a: {self.aLeadK:4.1f}"
    return ret


def laplacian_pdf(x: float, mu: float, b: float):
  b = max(b, 1e-4)
  return math.exp(-abs(x-mu)/b)


def match_vision_to_track(v_ego: float, lead: capnp._DynamicStructReader, tracks: dict[int, Track], state: dict,
                          radar_to_camera: float = RADAR_TO_CAMERA):
  # Low-pass the vision distance used to pick and gate the radar track. The raw per-frame model distance is
  # very noisy at range; gating on it is what let a closer object drift in/out of the gate. The reported
  # leadOne distance downstream is still the raw radar/vision value, only the association uses the smoothed one.
  raw_offset = lead.x[0] - radar_to_camera
  offset_vision_dist = state.get('offset', raw_offset)
  offset_vision_dist += (DT_MDL / (MATCH_VISION_DIST_RC + DT_MDL)) * (raw_offset - offset_vision_dist)
  state['offset'] = offset_vision_dist

  def prob(c):
    prob_d = laplacian_pdf(c.dRel, offset_vision_dist, lead.xStd[0])
    prob_y = laplacian_pdf(c.yRel, -lead.y[0], lead.yStd[0])
    prob_v = laplacian_pdf(c.vRel + v_ego, lead.v[0], lead.vStd[0])

    # This isn't exactly right, but it's a good heuristic
    return prob_d * prob_y * prob_v

  def is_sane(c):
    # Cap the gate so a radar return that is much closer/farther than the vision lead (i.e. a different
    # object) is not fused to it. Stationary radar points can be false positives.
    dist_gate = min(max(offset_vision_dist * .25, 5.0), MATCH_VISION_DIST_GATE_MAX)
    dist_sane = abs(c.dRel - offset_vision_dist) < dist_gate
    vel_sane = (abs(c.vRel + v_ego - lead.v[0]) < 10) or (v_ego + c.vRel > 3)
    # A track laterally far from the vision lead is a DIFFERENT object no matter how well the range
    # matches (track yRel convention is -lead.y). Range-scaled: see MATCH_VISION_LAT_GATE_*.
    lat_gate = MATCH_VISION_LAT_GATE_BASE + MATCH_VISION_LAT_GATE_SLOPE * max(c.dRel - MATCH_VISION_LAT_GATE_START, 0.0)
    lat_sane = abs(c.yRel + lead.y[0]) < lat_gate
    return dist_sane and vel_sane and lat_sane

  track = max(tracks.values(), key=prob)
  if not is_sane(track):
    track = None

  # Association hysteresis: require a *new* radar track to be the best sane match for MATCH_SWITCH_FRAMES
  # consecutive frames before it becomes the fused lead, holding the current one meanwhile. This stops
  # single-frame associations from flip-flopping leadOne between two objects. Losing the current track
  # (it disappears or goes insane) still drops immediately.
  prev_id = state.get('track_id', -1)
  cand_id = track.identifier if track is not None else -1
  if cand_id != prev_id and cand_id != -1:
    state['cand_cnt'] = state.get('cand_cnt', 0) + 1 if state.get('cand_id') == cand_id else 1
    state['cand_id'] = cand_id
    if state['cand_cnt'] < MATCH_SWITCH_FRAMES:
      prev_track = tracks.get(prev_id)
      track = prev_track if (prev_track is not None and is_sane(prev_track)) else None
  else:
    state['cand_id'] = cand_id
    state['cand_cnt'] = 0

  state['track_id'] = track.identifier if track is not None else -1
  return track


def get_RadarState_from_vision(lead_msg: capnp._DynamicStructReader, v_ego: float, model_v_ego: float, lead_prob: float,
                               radar_to_camera: float = RADAR_TO_CAMERA):
  lead_v_rel_pred = lead_msg.v[0] - model_v_ego
  return {
    "dRel": float(lead_msg.x[0] - radar_to_camera),
    "yRel": float(-lead_msg.y[0]),
    "vRel": float(lead_v_rel_pred),
    "vLead": float(v_ego + lead_v_rel_pred),
    "vLeadK": float(v_ego + lead_v_rel_pred),
    "aLeadK": float(lead_msg.a[0]),
    "aLeadTau": 0.3,
    "fcw": False,
    "modelProb": float(lead_prob),
    "status": True,
    "radar": False,
    "radarTrackId": -1,
  }


def get_lead(v_ego: float, ready: bool, tracks: dict[int, Track], lead_msg: capnp._DynamicStructReader,
             model_v_ego: float, lead_prob: float, CP: structs.CarParams, CP_SP: structs.CarParamsSP,
             low_speed_override: bool = True, state: dict | None = None,
             radar_to_camera: float = RADAR_TO_CAMERA) -> dict[str, Any]:
  # Determine leads, this is where the essential logic happens
  if state is None:
    state = {}
  if len(tracks) > 0 and ready and lead_prob > .5:
    track = match_vision_to_track(v_ego, lead_msg, tracks, state, radar_to_camera)
  else:
    track = None
    state.clear()  # reset association hysteresis when there is no vision lead to match against

  lead_dict = {'status': False}
  if track is not None:
    lead_dict = track.get_RadarState(lead_prob)
    lead_dict = get_custom_yrel(CP, CP_SP, lead_dict, lead_msg)
    lead_dict = apply_native_doppler_guard(lead_dict, track, lead_msg, v_ego, model_v_ego)
    lead_dict = apply_vision_closing_floor(lead_dict, lead_msg, v_ego, model_v_ego)
  elif (track is None) and ready and (lead_prob > .5):
    lead_dict = get_RadarState_from_vision(lead_msg, v_ego, model_v_ego, lead_prob, radar_to_camera)

  if low_speed_override:
    low_speed_tracks = [c for c in tracks.values() if c.potential_low_speed_lead(v_ego)]
    if len(low_speed_tracks) > 0:
      closest_track = min(low_speed_tracks, key=lambda c: c.dRel)

      # Only choose new track if it is actually closer than the previous one
      if (not lead_dict['status']) or (closest_track.dRel < lead_dict['dRel']):
        lead_dict = closest_track.get_RadarState()

  return lead_dict


def get_custom_yrel(CP: structs.CarParams, CP_SP: structs.CarParamsSP, lead_dict: dict[str, Any],
                    lead_msg: capnp._DynamicStructReader) -> dict[str, Any]:
  if CP.brand == "hyundai" and (CP_SP.flags & HyundaiFlagsSP.ENHANCED_SCC or
                                CP.flags & (HyundaiFlags.CANFD_CAMERA_SCC | HyundaiFlags.CAMERA_SCC)):
    lead_dict['yRel'] = float(-lead_msg.y[0])

  return lead_dict


def apply_native_doppler_guard(lead_dict: dict[str, Any], track: 'Track',
                               lead_msg: capnp._DynamicStructReader, v_ego: float, model_v_ego: float) -> dict[str, Any]:
  # False-closing guard for a radar-fused lead. The radar track's vRel is DERIVED from range and can read
  # a spurious hard closing; the radar's OWN native Doppler (track.vRelNative) and the vision lead are two
  # INDEPENDENT closing-speed estimates. When BOTH say the lead is closing meaningfully less than the
  # track's vRel claims, the closing is an artifact -> damp the fused lead to the corroborated velocity.
  # No-op when the native Doppler is unavailable (NaN), so it only affects cars/tracks that publish one.
  vrn = getattr(track, 'vRelNative', float('nan'))
  if not math.isfinite(vrn):
    return lead_dict
  vrel = lead_dict['vRel']
  vision_vrel = float(lead_msg.v[0] - model_v_ego)
  m = NATIVE_DOPPLER_FALSE_CLOSE_MARGIN
  if vrel < vrn - m and vrel < vision_vrel - m:
    # Both independent sources agree the lead is closing less. Use the MORE-closing (safer) of the two
    # corroborators so the correction never erases a closing that either one still sees.
    corrected = min(vrn, vision_vrel)
    lead_dict['vRel'] = float(corrected)
    lead_dict['vLead'] = float(v_ego + corrected)
    lead_dict['vLeadK'] = float(v_ego + corrected)
    # The spurious closing also biased the tracked accel negative; a corroborated-absent closing is not a
    # real deceleration, so neutralize it to keep the correction internally consistent (no phantom brake).
    lead_dict['aLeadK'] = float(max(lead_dict['aLeadK'], 0.0))
  return lead_dict


def apply_vision_closing_floor(lead_dict: dict[str, Any], lead_msg: capnp._DynamicStructReader,
                               v_ego: float, model_v_ego: float) -> dict[str, Any]:
  # See VISION_CLOSING_FLOOR_MARGIN: a radar-fused lead may never report meaningfully LESS closing than
  # the confident vision estimate it is fused to. Adopting vision's own vRel/vLead/aLead here is exactly
  # what the vision-only (202605) stack would have published, so this can only move behavior TOWARD the
  # known-good baseline -- and only in the cautious direction.
  vis_vrel = float(lead_msg.v[0] - model_v_ego)
  if vis_vrel < lead_dict['vRel'] - VISION_CLOSING_FLOOR_MARGIN:
    lead_dict['vRel'] = vis_vrel
    lead_dict['vLead'] = float(v_ego + vis_vrel)
    lead_dict['vLeadK'] = float(v_ego + vis_vrel)
    lead_dict['aLeadK'] = float(min(lead_dict['aLeadK'], lead_msg.a[0]))
  return lead_dict


def apply_bumpless_transfer(lead_dict: dict[str, Any], bstate: dict) -> dict[str, Any]:
  # Bumpless source transfer (see BUMPLESS_* above): when the published lead switches source (radar-fused
  # <-> vision-only, or a different radar track) the raw dRel/vLead step by the inter-source disagreement
  # even though it is the SAME physical car. Capture the discontinuity as an offset and decay it to zero,
  # so downstream consumers see the new source's dynamics immediately but never the DC step. The offset
  # state lives OUTSIDE the association state (it must survive hysteresis resets).
  if not lead_dict.get('status'):
    bstate.clear()
    return lead_dict
  src = (lead_dict.get('radar', False), lead_dict.get('radarTrackId', -1))
  prev = bstate.get('bp')
  if prev is not None:
    off_d, off_v, prev_src, prev_raw_d, prev_raw_v = prev
    if src != prev_src:
      jump_d = lead_dict['dRel'] - prev_raw_d
      jump_v = lead_dict['vLeadK'] - prev_raw_v
      if abs(jump_d) < BUMPLESS_MAX_STEP:
        off_d -= jump_d
        off_v -= jump_v
      else:
        off_d = off_v = 0.0  # too large to be the same object: a real lead change must step through
    off_d *= BUMPLESS_DECAY
    off_v *= BUMPLESS_DECAY
    if abs(off_d) < 0.05:
      off_d = 0.0
    if abs(off_v) < 0.05:
      off_v = 0.0
  else:
    off_d = off_v = 0.0
  raw_d, raw_v = lead_dict['dRel'], lead_dict['vLeadK']
  lead_dict = dict(lead_dict)
  lead_dict['dRel'] = float(raw_d + off_d)
  lead_dict['vLeadK'] = float(raw_v + off_v)
  lead_dict['vLead'] = float(lead_dict['vLead'] + off_v)
  bstate['bp'] = (off_d, off_v, src, raw_d, raw_v)
  return lead_dict


class RadarD:
  def __init__(self, CP: structs.CarParams, CP_SP: structs.CarParams, delay: float = 0.0):
    self.CP = CP
    self.CP_SP = CP_SP

    self.current_time = 0.0

    self.tracks: dict[int, Track] = {}
    self.kalman_params = KalmanParams(DT_MDL)
    self.lead_prob_filters = [FirstOrderFilter(0.0, 0.2, DT_MDL) for _ in range(2)]
    self.lead_states: list[dict] = [{}, {}]  # per-lead radar<->vision association hysteresis state
    self.lead_bumpless: list[dict] = [{}, {}]  # per-lead bumpless source-transfer offsets (survives resets)
    # per-platform radar<->camera longitudinal reference (see RADAR_TO_CAMERA_HONDA_BOSCH_A)
    self.radar_to_camera = RADAR_TO_CAMERA_HONDA_BOSCH_A if CP.carFingerprint in HONDA_BOSCH_A else RADAR_TO_CAMERA

    self.v_ego = 0.0
    self.v_ego_hist = deque([0.0], maxlen=int(round(delay / DT_MDL))+1)
    self.last_v_ego_frame = -1

    self.radar_state: capnp._DynamicStructBuilder | None = None
    self.radar_state_valid = False

    self.ready = False

  def update(self, sm: messaging.SubMaster, rr: car.RadarData):
    self.ready = sm.seen['modelV2']
    self.current_time = 1e-9*max(sm.logMonoTime.values())

    if sm.recv_frame['carState'] != self.last_v_ego_frame:
      self.v_ego = sm['carState'].vEgo
      self.v_ego_hist.append(self.v_ego)
      self.last_v_ego_frame = sm.recv_frame['carState']

    ar_pts = {pt.trackId: [pt.dRel, pt.yRel, pt.vRel, pt.measured, getattr(pt, 'vRelNative', float('nan'))] for pt in rr.points}

    # *** remove missing points from meta data ***
    for ids in list(self.tracks.keys()):
      if ids not in ar_pts:
        self.tracks.pop(ids, None)

    # *** compute the tracks ***
    for ids in ar_pts:
      rpt = ar_pts[ids]

      # align v_ego by a fixed time to align it with the radar measurement
      v_lead = rpt[2] + self.v_ego_hist[0]

      # create the track if it doesn't exist or it's a new track
      if ids not in self.tracks:
        self.tracks[ids] = Track(ids, v_lead, self.kalman_params)
      self.tracks[ids].update(rpt[0], rpt[1], rpt[2], v_lead, rpt[3], rpt[4])

    # *** publish radarState ***
    self.radar_state_valid = sm.all_checks()
    self.radar_state = log.RadarState.new_message()
    self.radar_state.mdMonoTime = sm.logMonoTime['modelV2']
    self.radar_state.radarErrors = rr.errors
    self.radar_state.carStateMonoTime = sm.logMonoTime['carState']

    if len(sm['modelV2'].velocity.x):
      model_v_ego = sm['modelV2'].velocity.x[0]
    else:
      model_v_ego = self.v_ego
    leads_v3 = sm['modelV2'].leadsV3
    if len(leads_v3) > 1:
      for i in range(2):
        # Asymmetric filter on lead prob to keep lead when uncertain
        lead_prob = leads_v3[i].prob
        if lead_prob > self.lead_prob_filters[i].x:
          self.lead_prob_filters[i].x = lead_prob
        else:
          self.lead_prob_filters[i].update(lead_prob)

      lead_one = get_lead(self.v_ego, self.ready, self.tracks, leads_v3[0], model_v_ego, self.lead_prob_filters[0].x,
                          self.CP, self.CP_SP, low_speed_override=True, state=self.lead_states[0],
                          radar_to_camera=self.radar_to_camera)
      lead_two = get_lead(self.v_ego, self.ready, self.tracks, leads_v3[1], model_v_ego, self.lead_prob_filters[1].x,
                          self.CP, self.CP_SP, low_speed_override=False, state=self.lead_states[1],
                          radar_to_camera=self.radar_to_camera)
      self.radar_state.leadOne = apply_bumpless_transfer(lead_one, self.lead_bumpless[0])
      self.radar_state.leadTwo = apply_bumpless_transfer(lead_two, self.lead_bumpless[1])

  def publish(self, pm: messaging.PubMaster):
    assert self.radar_state is not None

    radar_msg = messaging.new_message("radarState")
    radar_msg.valid = self.radar_state_valid
    radar_msg.radarState = self.radar_state
    pm.send("radarState", radar_msg)


# fuses camera and radar data for best lead detection
def main() -> None:
  config_realtime_process(5, Priority.CTRL_LOW)

  # wait for stats about the car to come in from controls
  cloudlog.info("radard is waiting for CarParams")
  CP = messaging.log_from_bytes(Params().get("CarParams", block=True), car.CarParams)
  cloudlog.info("radard got CarParams")

  cloudlog.info("radard is waiting for CarParamsSP")
  CP_SP = messaging.log_from_bytes(Params().get("CarParamsSP", block=True), custom.CarParamsSP)
  cloudlog.info("radard got CarParamsSP")

  # *** setup messaging
  sm = messaging.SubMaster(['modelV2', 'carState', 'liveTracks'], poll='modelV2')
  pm = messaging.PubMaster(['radarState'])

  RD = RadarD(CP, CP_SP, CP.radarDelay)

  while 1:
    sm.update()

    RD.update(sm, sm['liveTracks'])
    RD.publish(pm)


if __name__ == "__main__":
  main()
