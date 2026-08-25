#pragma once

/*
 * People counting - multi object tracker (IoU / greedy association).
 *
 * Constraints (see project spec):
 *   - C++17, -fno-rtti -fno-exceptions -fno-threadsafe-statics
 *   - no throw / try / catch / dynamic_cast / typeid
 *   - no runtime heap allocation: every buffer below is a fixed size static array
 *   - fixed point math only on the per frame hot path
 *
 * Coordinate convention: el_box_t from the YOLO family algorithms stores the box
 * *center* in (x, y) and the extent in (w, h) (see core/algorithm/el_algorithm_yolo.cpp:193).
 * All pixel values here live in the data path resolution reported by el_img_t
 * (240x240 on the current Grove Vision AI V2 build) and are never hard coded:
 * the caller passes the live frame resolution every update.
 */

#include <cstdint>

namespace sscma::extension::counter {

#define PC_MAX_TRACKS   32
#define PC_MAX_LINES    4
#define PC_MAX_ROIS     4
#define PC_ROI_VERTICES 4
#define PC_MAX_DETS     32

/* the detection class we track; the deployed model is Swift-YOLO Nano single
 * class "person", so this is always 0. */
#define PC_TARGET_CLASS 0

/* the firmware ROM budget is tight (see the report): keep the counting code
 * size optimised and out of line even though the project builds at -O2 */
#if defined(__GNUC__) && !defined(__clang__)
    #define PC_SMALL __attribute__((noinline, optimize("Os")))
#else
    #define PC_SMALL __attribute__((noinline))
#endif

enum : uint8_t { PC_NEW = 0, PC_TENTATIVE = 1, PC_CONFIRMED = 2, PC_COASTING = 3 };

enum : uint8_t { PC_ANCHOR_CENTER = 0, PC_ANCHOR_BOTTOM = 1 };

struct pc_track_t {
    int16_t  x, y, w, h;        /* pixels, box center + extent */
    int16_t  ax, ay;            /* anchor point */
    int16_t  prev_ax, prev_ay;  /* anchor of the previous frame (velocity source) */
    int16_t  vx, vy;            /* pixels / frame */
    int32_t  id;                /* monotonic, never reused */
    uint16_t hits;
    uint16_t misses;
    uint8_t  state;
    uint8_t  side_mask;   /* bit i: sign of track vs line i (1 = positive) */
    uint8_t  side_valid;  /* bit i: side of line i initialised */
    uint8_t  in_roi_mask; /* bit i: anchor currently inside roi i */
    uint8_t  active;
};

struct pc_det_t {
    int16_t x, y, w, h; /* center + extent, pixels */
    int16_t ax, ay;     /* anchor */
    int16_t box_index;  /* index of the source box in the result list */
};

struct pc_tracker_cfg_t {
    uint16_t iou_q10;     /* IoU threshold, Q10 (0..1024) */
    uint16_t max_miss;    /* recycle after this many consecutive misses */
    uint16_t min_hits;    /* TENTATIVE -> CONFIRMED */
    uint16_t gate_dist;   /* anchor distance gate, pixels (derived from width) */
    uint8_t  anchor_mode; /* PC_ANCHOR_CENTER / PC_ANCHOR_BOTTOM */
};

/* ---------------------------------------------------------------- helpers */

static inline int32_t pc_abs32(int32_t v) { return v < 0 ? -v : v; }

/* IoU in Q10 between two center-extent boxes. */
static inline int32_t pc_iou_q10(
  int16_t ax, int16_t ay, int16_t aw, int16_t ah, int16_t bx, int16_t by, int16_t bw, int16_t bh) {
    const int32_t al = (int32_t)ax - (aw >> 1), ar = al + aw;
    const int32_t at = (int32_t)ay - (ah >> 1), ab = at + ah;
    const int32_t bl = (int32_t)bx - (bw >> 1), br = bl + bw;
    const int32_t bt = (int32_t)by - (bh >> 1), bb = bt + bh;

    const int32_t il = al > bl ? al : bl;
    const int32_t ir = ar < br ? ar : br;
    const int32_t it = at > bt ? at : bt;
    const int32_t ib = ab < bb ? ab : bb;

    const int32_t iw = ir - il;
    const int32_t ih = ib - it;
    if (iw <= 0 || ih <= 0) return 0;

    const int32_t inter = iw * ih;
    const int32_t uni   = (int32_t)aw * ah + (int32_t)bw * bh - inter;
    if (uni <= 0) return 0;

    return (inter << 10) / uni;
}

/* ---------------------------------------------------------------- tracker */

class PcTracker {
   public:
    PcTracker() { reset(); }

    PC_SMALL void reset() {
        for (int i = 0; i < PC_MAX_TRACKS; ++i) _tracks[i] = pc_track_t{};
        _next_id  = 1;
        _n_active = 0;
    }

    pc_track_t*       tracks() { return _tracks; }
    const pc_track_t* tracks() const { return _tracks; }
    static int        capacity() { return PC_MAX_TRACKS; }

    /*
     * One tracking step.
     *  dets     : detections of this frame (already filtered by class)
     *  n_dets   : number of detections
     *  cfg      : live tracker configuration
     *  out_tid  : out_tid[i] receives the track id assigned to dets[i]
     *             (-1 when the track is not yet CONFIRMED, per the output spec)
     */
    PC_SMALL void step(const pc_det_t* dets, int n_dets, const pc_tracker_cfg_t& cfg, int32_t* out_tid) {
        if (n_dets > PC_MAX_DETS) n_dets = PC_MAX_DETS;

        /* 1. predict: constant velocity extrapolation of the anchor */
        for (int t = 0; t < PC_MAX_TRACKS; ++t) {
            pc_track_t& tr = _tracks[t];
            if (!tr.active) continue;
            tr.prev_ax = tr.ax;
            tr.prev_ay = tr.ay;
            tr.ax      = (int16_t)(tr.ax + tr.vx);
            tr.ay      = (int16_t)(tr.ay + tr.vy);
            tr.x       = (int16_t)(tr.x + tr.vx);
            tr.y       = (int16_t)(tr.y + tr.vy);
        }

        /* 2. cost matrix (IoU Q10) with an anchor distance gate */
        const int32_t gate2 = (int32_t)cfg.gate_dist * (int32_t)cfg.gate_dist;
        for (int d = 0; d < n_dets; ++d) {
            _det_taken[d] = 0;
            out_tid[d]    = -1;
            for (int t = 0; t < PC_MAX_TRACKS; ++t) {
                const pc_track_t& tr = _tracks[t];
                if (!tr.active) {
                    _cost[d][t] = 0;
                    continue;
                }
                const int32_t dx = (int32_t)dets[d].ax - tr.ax;
                const int32_t dy = (int32_t)dets[d].ay - tr.ay;
                if (dx * dx + dy * dy > gate2) {
                    _cost[d][t] = 0;
                    continue;
                }
                _cost[d][t] = (int16_t)pc_iou_q10(
                  dets[d].x, dets[d].y, dets[d].w, dets[d].h, tr.x, tr.y, tr.w, tr.h);
            }
        }
        for (int t = 0; t < PC_MAX_TRACKS; ++t) _trk_taken[t] = 0;

        /* 3. greedy association: repeatedly take the globally best pair */
        for (;;) {
            int32_t best  = (int32_t)cfg.iou_q10;
            int     bestd = -1, bestt = -1;
            for (int d = 0; d < n_dets; ++d) {
                if (_det_taken[d]) continue;
                for (int t = 0; t < PC_MAX_TRACKS; ++t) {
                    if (_trk_taken[t] || !_tracks[t].active) continue;
                    if ((int32_t)_cost[d][t] >= best) {
                        best  = _cost[d][t];
                        bestd = d;
                        bestt = t;
                    }
                }
            }
            if (bestd < 0) break;
            _det_taken[bestd] = 1;
            _trk_taken[bestt] = 1;
            m_update_track(_tracks[bestt], dets[bestd], cfg);
            out_tid[bestd] = m_reported_id(_tracks[bestt]);
        }

        /* 4a. unmatched tracks -> coasting */
        for (int t = 0; t < PC_MAX_TRACKS; ++t) {
            pc_track_t& tr = _tracks[t];
            if (!tr.active || _trk_taken[t]) continue;
            tr.misses++;
            if (tr.state == PC_CONFIRMED) tr.state = PC_COASTING;
        }

        /* 4b. unmatched detections -> new tracks */
        for (int d = 0; d < n_dets; ++d) {
            if (_det_taken[d]) continue;
            int slot = m_alloc_slot();
            if (slot < 0) continue;
            pc_track_t& tr = _tracks[slot];
            tr             = pc_track_t{};
            tr.active      = 1;
            tr.id          = _next_id++;
            tr.x           = dets[d].x;
            tr.y           = dets[d].y;
            tr.w           = dets[d].w;
            tr.h           = dets[d].h;
            tr.ax          = dets[d].ax;
            tr.ay          = dets[d].ay;
            tr.prev_ax     = dets[d].ax;
            tr.prev_ay     = dets[d].ay;
            tr.hits        = 1;
            tr.misses      = 0;
            tr.state       = PC_TENTATIVE; /* NEW -> TENTATIVE on the first frame */
            out_tid[d]     = m_reported_id(tr);
        }

        /* 5. state machine / recycling */
        _n_active = 0;
        for (int t = 0; t < PC_MAX_TRACKS; ++t) {
            pc_track_t& tr = _tracks[t];
            if (!tr.active) continue;
            if (tr.misses > cfg.max_miss) {
                tr.active = 0;
                continue;
            }
            /* suppress short lived tracks born from false positives */
            if (tr.state == PC_TENTATIVE && tr.misses > 1) {
                tr.active = 0;
                continue;
            }
            ++_n_active;
        }
    }

    int active_count() const { return _n_active; }

   private:
    static int32_t m_reported_id(const pc_track_t& tr) {
        return (tr.state == PC_CONFIRMED || tr.state == PC_COASTING) ? tr.id : -1;
    }

    static void m_update_track(pc_track_t& tr, const pc_det_t& det, const pc_tracker_cfg_t& cfg) {
        (void)cfg;
        tr.x  = det.x;
        tr.y  = det.y;
        tr.w  = det.w;
        tr.h  = det.h;
        tr.vx = (int16_t)(det.ax - tr.prev_ax);
        tr.vy = (int16_t)(det.ay - tr.prev_ay);
        tr.ax = det.ax;
        tr.ay = det.ay;
        tr.hits++;
        tr.misses = 0;
        if (tr.state == PC_NEW) tr.state = PC_TENTATIVE;
        if (tr.hits >= cfg.min_hits) tr.state = PC_CONFIRMED;
        else if (tr.state == PC_COASTING) tr.state = PC_CONFIRMED;
    }

    int m_alloc_slot() {
        for (int t = 0; t < PC_MAX_TRACKS; ++t)
            if (!_tracks[t].active) return t;
        return -1;
    }

    pc_track_t _tracks[PC_MAX_TRACKS];
    int16_t    _cost[PC_MAX_DETS][PC_MAX_TRACKS];
    uint8_t    _det_taken[PC_MAX_DETS];
    uint8_t    _trk_taken[PC_MAX_TRACKS];
    int32_t    _next_id;
    int        _n_active;
};

}  // namespace sscma::extension::counter
