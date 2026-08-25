#pragma once

/*
 * People counting - line crossing (in/out) + region occupancy.
 *
 * All configuration is stored in normalised 0..1000 coordinates and converted
 * to pixels whenever the live frame resolution changes, so a resolution switch
 * does not invalidate the configuration. The resolution is read from the frame
 * (el_img_t) every update - it is never hard coded.
 *
 * No heap, no exceptions, no RTTI. Every buffer is a fixed size static array.
 */

#include <cstdint>
#include <forward_list>

#include "core/el_types.h"
#include "sscma/extension/counter/pc_tracker.hpp"

namespace sscma::extension::counter {

#define PC_NORM_MAX 1000

#define PC_CFG_MAGIC 0x50433031u /* 'PC01' */

#define PC_DEF_IOU_Q10     307 /* 0.30 */
#define PC_DEF_MAX_MISS    8   /* ~0.44 s @ 18 fps */
#define PC_DEF_MIN_HITS    3
#define PC_DEF_ANCHOR_MODE PC_ANCHOR_CENTER

/* persisted blob: normalised geometry + tracker knobs, never pixels */
struct pc_persist_t {
    uint32_t magic;
    int16_t  line_n[PC_MAX_LINES][4];                 /* x1,y1,x2,y2 */
    int16_t  roi_n[PC_MAX_ROIS][PC_ROI_VERTICES * 2]; /* x0,y0,x1,y1,... */
    uint8_t  line_en[PC_MAX_LINES];
    uint8_t  roi_en[PC_MAX_ROIS];
    uint16_t iou_q10;
    uint16_t max_miss;
    uint16_t min_hits;
    uint8_t  anchor_mode;
    uint8_t  reserved;
};

struct pc_line_t {
    int16_t x1, y1, x2, y2; /* pixels */
    int32_t count_ab, count_ba;
    uint8_t enabled;
};

struct pc_roi_t {
    int16_t px[PC_ROI_VERTICES], py[PC_ROI_VERTICES]; /* pixels */
    int32_t enter_count;
    int32_t current;
    uint8_t enabled;
};

/* -------------------------------------------------------------- geometry */

/*
 * Signed side of a point against the directed line (x1,y1) -> (x2,y2);
 * 0 means exactly on it.
 *
 * Positive is the left hand side of the direction vector. count_ab ("in") is
 * incremented on a negative -> positive transition, count_ba ("out") on the
 * reverse, so the installer picks the in/out sense by choosing the order of the
 * two endpoints in AT+CNTLINE.
 */
static inline int pc_side(const pc_line_t& l, int16_t x, int16_t y) {
    const int32_t v = (int32_t)(x - l.x1) * (int32_t)(l.y2 - l.y1) -
                      (int32_t)(y - l.y1) * (int32_t)(l.x2 - l.x1);
    return v > 0 ? 1 : (v < 0 ? -1 : 0);
}

/*
 * Crossing number / ray casting point in polygon, integer only.
 *
 * The float form of the test is
 *     x < px[i] + (px[j] - px[i]) * (y - py[i]) / (py[j] - py[i])
 * Multiplying it out by dy = py[j] - py[i] reverses the comparison whenever
 * dy < 0, so the sign of dy has to be folded back in - without the xor below
 * every downward edge votes the wrong way and the test fails for points inside
 * the polygon (caught by the host self test).
 */
static inline bool pc_point_in_poly(const int16_t* px, const int16_t* py, int n, int16_t x, int16_t y) {
    bool inside = false;
    for (int i = 0, j = n - 1; i < n; j = i++) {
        if ((py[i] > y) == (py[j] > y)) continue;
        const int32_t dy  = (int32_t)py[j] - py[i];
        const int32_t lhs = (int32_t)(x - px[i]) * dy;
        const int32_t rhs = (int32_t)(y - py[i]) * ((int32_t)px[j] - px[i]);
        /* strict on both branches: folding dy < 0 into a negated "<" would let
         * the equality case (the point sitting exactly on the edge) through */
        if (dy > 0 ? (lhs < rhs) : (lhs > rhs)) inside = !inside;
    }
    return inside;
}

/* ---------------------------------------------- tiny non allocating writer */

/*
 * A single shared output buffer serves both the per frame "counts" field and
 * the AT command replies. Both run on the same executor task queue, so they are
 * never in flight at the same time. Using a plain char writer instead of
 * std::string / concat_strings<> keeps the ROM cost of this feature down.
 */
#define PC_TEXT_CAP 768

struct PcWriter {
    char* buf;
    int   cap;
    int   n;

    void init(char* b, int c) {
        buf = b;
        cap = c - 1;
        n   = 0;
        buf[0] = '\0';
    }

    PC_SMALL void str(const char* s) {
        while (*s && n < cap) buf[n++] = *s++;
        buf[n] = '\0';
    }

    PC_SMALL void num(int32_t v) {
        char    tmp[12];
        int     k   = 0;
        uint8_t neg = 0;
        if (v < 0) {
            neg = 1;
            v   = -v;
        }
        do {
            tmp[k++] = (char)('0' + (v % 10));
            v /= 10;
        } while (v);
        if (neg) tmp[k++] = '-';
        while (k > 0 && n < cap) buf[n++] = tmp[--k];
        buf[n] = '\0';
    }

    /* "key": value, */
    PC_SMALL void kv(const char* key, int32_t v, const char* tail) {
        str(key);
        num(v);
        str(tail);
    }
};

inline char*     pc_text_buf();
inline PcWriter& pc_writer();

/* ------------------------------------------------------------- counter */

class PcCounter {
   public:
    PcCounter() { defaults(); }

    PC_SMALL void defaults() {
        _p        = pc_persist_t{};
        _p.magic  = PC_CFG_MAGIC;
        _p.iou_q10     = PC_DEF_IOU_Q10;
        _p.max_miss    = PC_DEF_MAX_MISS;
        _p.min_hits    = PC_DEF_MIN_HITS;
        _p.anchor_mode = PC_DEF_ANCHOR_MODE;
        for (int i = 0; i < PC_MAX_LINES; ++i) _lines[i] = pc_line_t{};
        for (int i = 0; i < PC_MAX_ROIS; ++i) _rois[i] = pc_roi_t{};
        _w = _h = 0;
        _n_boxes = 0;
        _tracker.reset();
    }

    pc_persist_t&       persist() { return _p; }
    const pc_persist_t& persist() const { return _p; }

    void sanitize() {
        if (_p.magic != PC_CFG_MAGIC) {
            defaults();
            return;
        }
        if (_p.iou_q10 == 0 || _p.iou_q10 > 1024) _p.iou_q10 = PC_DEF_IOU_Q10;
        if (_p.max_miss == 0 || _p.max_miss > 240) _p.max_miss = PC_DEF_MAX_MISS;
        if (_p.min_hits == 0 || _p.min_hits > 240) _p.min_hits = PC_DEF_MIN_HITS;
        if (_p.anchor_mode > PC_ANCHOR_BOTTOM) _p.anchor_mode = PC_DEF_ANCHOR_MODE;
        _w = _h = 0; /* force a pixel cache rebuild */
    }

    /* --------------------------------------------------------- config API */

    PC_SMALL bool set_line(int idx, int16_t x1, int16_t y1, int16_t x2, int16_t y2) {
        if (idx < 0 || idx >= PC_MAX_LINES) return false;
        if (!m_norm_ok(x1) || !m_norm_ok(y1) || !m_norm_ok(x2) || !m_norm_ok(y2)) return false;
        _p.line_n[idx][0] = x1;
        _p.line_n[idx][1] = y1;
        _p.line_n[idx][2] = x2;
        _p.line_n[idx][3] = y2;
        /* all zero disables, a degenerate line is rejected as well */
        _p.line_en[idx]   = (x1 == x2 && y1 == y2) ? 0 : 1;
        _lines[idx].count_ab = 0;
        _lines[idx].count_ba = 0;
        m_rebuild_pixels();
        m_clear_sides();
        return true;
    }

    PC_SMALL bool set_roi(int idx, const int16_t* xy /* 8 values */) {
        if (idx < 0 || idx >= PC_MAX_ROIS) return false;
        uint8_t nonzero = 0;
        for (int i = 0; i < PC_ROI_VERTICES * 2; ++i) {
            if (!m_norm_ok(xy[i])) return false;
            if (xy[i] != 0) nonzero = 1;
        }
        for (int i = 0; i < PC_ROI_VERTICES * 2; ++i) _p.roi_n[idx][i] = xy[i];
        _p.roi_en[idx]        = nonzero;
        _rois[idx].enter_count = 0;
        _rois[idx].current     = 0;
        m_rebuild_pixels();
        m_clear_sides();
        return true;
    }

    PC_SMALL bool set_cfg(int32_t iou_q10, int32_t max_miss, int32_t min_hits, int32_t anchor_mode) {
        if (iou_q10 <= 0 || iou_q10 > 1024) return false;
        if (max_miss <= 0 || max_miss > 240) return false;
        if (min_hits <= 0 || min_hits > 240) return false;
        if (anchor_mode < 0 || anchor_mode > PC_ANCHOR_BOTTOM) return false;
        _p.iou_q10     = (uint16_t)iou_q10;
        _p.max_miss    = (uint16_t)max_miss;
        _p.min_hits    = (uint16_t)min_hits;
        _p.anchor_mode = (uint8_t)anchor_mode;
        return true;
    }

    PC_SMALL void reset_counts() {
        for (int i = 0; i < PC_MAX_LINES; ++i) {
            _lines[i].count_ab = 0;
            _lines[i].count_ba = 0;
        }
        for (int i = 0; i < PC_MAX_ROIS; ++i) {
            _rois[i].enter_count = 0;
            _rois[i].current     = 0;
        }
        _tracker.reset();
        _n_boxes = 0;
    }

    const pc_line_t& line(int i) const { return _lines[i]; }
    const pc_roi_t&  roi(int i) const { return _rois[i]; }

    /* ---------------------------------------------------------- per frame */

    /* update from the detection list of the current frame */
    PC_SMALL void update(const std::forward_list<el_box_t>& boxes, uint16_t width, uint16_t height) {
        if (width == 0 || height == 0) return;
        if (width != _w || height != _h) {
            _w = width;
            _h = height;
            m_rebuild_pixels();
            m_clear_sides();
        }

        /* collect detections; keep the list order so the JSON writer can map
         * box index -> track id without a second pass */
        int n_det = 0;
        _n_boxes  = 0;
        for (const auto& b : boxes) {
            if (_n_boxes >= PC_MAX_DETS) break;
            const int bi   = _n_boxes++;
            _box_tid[bi]   = -1;
            if (b.target != PC_TARGET_CLASS) continue;
            pc_det_t& d = _dets[n_det];
            d.x         = (int16_t)b.x;
            d.y         = (int16_t)b.y;
            d.w         = (int16_t)b.w;
            d.h         = (int16_t)b.h;
            d.ax        = d.x;
            d.ay        = (_p.anchor_mode == PC_ANCHOR_BOTTOM) ? (int16_t)(d.y + (d.h >> 1)) : d.y;
            d.box_index = (int16_t)bi;
            ++n_det;
        }

        pc_tracker_cfg_t cfg;
        cfg.iou_q10     = _p.iou_q10;
        cfg.max_miss    = _p.max_miss;
        cfg.min_hits    = _p.min_hits;
        cfg.anchor_mode = _p.anchor_mode;
        /* the spec's 80 px gate was given for a 320 px wide data path; keep it
         * proportional so it survives a resolution change */
        cfg.gate_dist = (uint16_t)(width / 4u);
        if (cfg.gate_dist < 16) cfg.gate_dist = 16;

        _tracker.step(_dets, n_det, cfg, _det_tid);

        for (int i = 0; i < n_det; ++i) _box_tid[_dets[i].box_index] = _det_tid[i];

        m_count();
    }

    int32_t box_track_id(int index) const {
        return (index >= 0 && index < _n_boxes) ? _box_tid[index] : -1;
    }

    /* "counts": {...} into the shared text buffer, no allocation */
    PC_SMALL const char* counts_json() {
        PcWriter& w = pc_writer();
        w.init(pc_text_buf(), PC_TEXT_CAP);
        w.str("\"counts\": {\"lines\": [");
        const char* delim = "";
        for (int i = 0; i < PC_MAX_LINES; ++i) {
            if (!_p.line_en[i]) continue;
            w.str(delim);
            w.kv("{\"id\": ", i, ", \"in\": ");
            w.kv("", _lines[i].count_ab, ", \"out\": ");
            w.kv("", _lines[i].count_ba, "}");
            delim = ", ";
        }
        w.str("], \"rois\": [");
        delim = "";
        for (int i = 0; i < PC_MAX_ROIS; ++i) {
            if (!_p.roi_en[i]) continue;
            w.str(delim);
            w.kv("{\"id\": ", i, ", \"cur\": ");
            w.kv("", _rois[i].current, ", \"entered\": ");
            w.kv("", _rois[i].enter_count, "}");
            delim = ", ";
        }
        w.str("]}");
        return w.buf;
    }

   private:
    static bool m_norm_ok(int16_t v) { return v >= 0 && v <= PC_NORM_MAX; }

    int16_t m_nx(int16_t n) const { return (int16_t)(((int32_t)n * _w) / PC_NORM_MAX); }
    int16_t m_ny(int16_t n) const { return (int16_t)(((int32_t)n * _h) / PC_NORM_MAX); }

    PC_SMALL void m_rebuild_pixels() {
        if (_w == 0 || _h == 0) return;
        for (int i = 0; i < PC_MAX_LINES; ++i) {
            _lines[i].enabled = _p.line_en[i];
            _lines[i].x1      = m_nx(_p.line_n[i][0]);
            _lines[i].y1      = m_ny(_p.line_n[i][1]);
            _lines[i].x2      = m_nx(_p.line_n[i][2]);
            _lines[i].y2      = m_ny(_p.line_n[i][3]);
        }
        for (int i = 0; i < PC_MAX_ROIS; ++i) {
            _rois[i].enabled = _p.roi_en[i];
            for (int v = 0; v < PC_ROI_VERTICES; ++v) {
                _rois[i].px[v] = m_nx(_p.roi_n[i][v * 2]);
                _rois[i].py[v] = m_ny(_p.roi_n[i][v * 2 + 1]);
            }
        }
    }

    /* geometry changed: drop the cached sides so the next frame re-seeds them
     * instead of emitting a spurious crossing */
    void m_clear_sides() {
        pc_track_t* tr = _tracker.tracks();
        for (int t = 0; t < PcTracker::capacity(); ++t) {
            tr[t].side_mask   = 0;
            tr[t].side_valid  = 0;
            tr[t].in_roi_mask = 0;
        }
    }

    PC_SMALL void m_count() {
        for (int i = 0; i < PC_MAX_ROIS; ++i) _rois[i].current = 0;

        pc_track_t* tr = _tracker.tracks();
        for (int t = 0; t < PcTracker::capacity(); ++t) {
            pc_track_t& k = tr[t];
            if (!k.active) continue;
            const bool countable = (k.state == PC_CONFIRMED || k.state == PC_COASTING);
            if (!countable) continue;

            /* line crossing */
            for (int i = 0; i < PC_MAX_LINES; ++i) {
                if (!_lines[i].enabled) continue;
                const int s = pc_side(_lines[i], k.ax, k.ay);
                if (s == 0) continue; /* exactly on the line: keep the last side */
                const uint8_t bit = (uint8_t)(1u << i);
                const uint8_t cur = (uint8_t)(s > 0 ? bit : 0);
                if (!(k.side_valid & bit)) { /* first observation never counts */
                    k.side_valid = (uint8_t)(k.side_valid | bit);
                    k.side_mask  = (uint8_t)((k.side_mask & ~bit) | cur);
                    continue;
                }
                const uint8_t prev = (uint8_t)(k.side_mask & bit);
                if (prev == cur) continue;
                if (cur) ++_lines[i].count_ab; /* negative -> positive */
                else ++_lines[i].count_ba;     /* positive -> negative */
                k.side_mask = (uint8_t)((k.side_mask & ~bit) | cur);
            }

            /* region occupancy */
            for (int i = 0; i < PC_MAX_ROIS; ++i) {
                if (!_rois[i].enabled) continue;
                const uint8_t bit = (uint8_t)(1u << i);
                const bool inside = pc_point_in_poly(_rois[i].px, _rois[i].py, PC_ROI_VERTICES, k.ax, k.ay);
                if (inside) {
                    ++_rois[i].current;
                    if (!(k.in_roi_mask & bit)) {
                        k.in_roi_mask = (uint8_t)(k.in_roi_mask | bit);
                        ++_rois[i].enter_count;
                    }
                } else {
                    k.in_roi_mask = (uint8_t)(k.in_roi_mask & ~bit);
                }
            }
        }
    }

    pc_persist_t _p;
    pc_line_t    _lines[PC_MAX_LINES];
    pc_roi_t     _rois[PC_MAX_ROIS];
    PcTracker    _tracker;
    pc_det_t     _dets[PC_MAX_DETS];
    int32_t      _det_tid[PC_MAX_DETS];
    int32_t      _box_tid[PC_MAX_DETS];
    int          _n_boxes;
    uint16_t     _w, _h;
};

/* shared text buffer + writer (BSS, no dynamic allocation) */
inline char* pc_text_buf() {
    static char buf[PC_TEXT_CAP];
    return buf;
}

inline PcWriter& pc_writer() {
    static PcWriter w{};
    return w;
}

/* single global instance (BSS, no dynamic allocation) */
inline PcCounter& pc_counter() {
    static PcCounter instance;
    return instance;
}

/* the invoke loop calls this for every algorithm; only box results are tracked */
inline void pc_on_results(const std::forward_list<el_box_t>& boxes, uint16_t width, uint16_t height) {
    pc_counter().update(boxes, width, height);
}

template <typename T> inline void pc_on_results(const T&, uint16_t, uint16_t) {}

}  // namespace sscma::extension::counter
