#pragma once

/*
 * AT command surface for the people counting extension.
 *
 * Parser constraints (sscma/repl/server.hpp):
 *   - arguments must be decimal, a "0x" prefix is silently truncated to 0
 *   - _argc = count(',') + 1, so the argument count is fixed per command
 *     (hence the fixed quadrilateral ROI)
 *
 * All geometry is exchanged in normalised 0..1000 coordinates, so the
 * configuration survives a resolution change.
 *
 *   AT+CNTLINE=<idx>,<x1>,<y1>,<x2>,<y2>                       (5 args)
 *   AT+CNTLINE?
 *   AT+CNTROI=<idx>,<x1>,<y1>,<x2>,<y2>,<x3>,<y3>,<x4>,<y4>    (9 args)
 *   AT+CNTROI?
 *   AT+CNTCFG=<iou_q10>,<max_miss>,<min_hits>,<anchor_mode>    (4 args)
 *   AT+CNTCFG?
 *   AT+CNTRST
 *
 * Disable a line or a region by sending all coordinates as 0.
 *
 * Everything below writes into the shared PcWriter buffer instead of building
 * std::string values: the sscma app sits at ~99.5% of CM55M_S_APP_ROM, so the
 * concat_strings<> template instantiations this would otherwise pull in are not
 * affordable.
 */

#include <cstdint>
#include <cstdlib>
#include <string>
#include <vector>

#include "sscma/definations.hpp"
#include "sscma/extension/counter/pc_counter.hpp"
#include "sscma/static_resource.hpp"

#define SSCMA_STORAGE_KEY_CONF_COUNTER "sscma#conf#counter"

namespace sscma::callback {

using namespace sscma::extension::counter;

enum : uint8_t {
    PC_CMD_SET_LINE = 0,
    PC_CMD_GET_LINE,
    PC_CMD_SET_ROI,
    PC_CMD_GET_ROI,
    PC_CMD_SET_CFG,
    PC_CMD_GET_CFG,
    PC_CMD_RESET,
};

inline void counter_load_config() {
    auto& cnt = pc_counter();
    auto& cfg = cnt.persist();
    if (static_resource->storage->contains(SSCMA_STORAGE_KEY_CONF_COUNTER)) {
        auto kv = el_make_storage_kv(SSCMA_STORAGE_KEY_CONF_COUNTER, cfg);
        *static_resource->storage >> kv;
        cnt.sanitize();
    } else {
        *static_resource->storage << el_make_storage_kv(SSCMA_STORAGE_KEY_CONF_COUNTER, cfg);
    }
}

PC_SMALL inline bool counter_save_config() {
    return static_resource->storage->emplace(
      el_make_storage_kv(SSCMA_STORAGE_KEY_CONF_COUNTER, pc_counter().persist()));
}

PC_SMALL inline void counter_write_lines(PcWriter& w) {
    const auto& cfg = pc_counter().persist();
    w.str("[");
    const char* delim = "";
    for (int i = 0; i < PC_MAX_LINES; ++i) {
        w.str(delim);
        w.kv("{\"id\": ", i, ", \"enabled\": ");
        w.kv("", cfg.line_en[i], ", \"line\": [");
        for (int v = 0; v < 4; ++v) w.kv("", cfg.line_n[i][v], v == 3 ? "]" : ", ");
        w.kv(", \"in\": ", pc_counter().line(i).count_ab, ", \"out\": ");
        w.kv("", pc_counter().line(i).count_ba, "}");
        delim = ", ";
    }
    w.str("]");
}

PC_SMALL inline void counter_write_rois(PcWriter& w) {
    const auto& cfg = pc_counter().persist();
    w.str("[");
    const char* delim = "";
    for (int i = 0; i < PC_MAX_ROIS; ++i) {
        w.str(delim);
        w.kv("{\"id\": ", i, ", \"enabled\": ");
        w.kv("", cfg.roi_en[i], ", \"roi\": [");
        for (int v = 0; v < PC_ROI_VERTICES * 2; ++v)
            w.kv("", cfg.roi_n[i][v], v == PC_ROI_VERTICES * 2 - 1 ? "]" : ", ");
        w.kv(", \"cur\": ", pc_counter().roi(i).current, ", \"entered\": ");
        w.kv("", pc_counter().roi(i).enter_count, "}");
        delim = ", ";
    }
    w.str("]");
}

PC_SMALL inline void counter_write_cfg(PcWriter& w) {
    const auto& cfg = pc_counter().persist();
    w.kv("{\"iou_q10\": ", cfg.iou_q10, ", \"max_miss\": ");
    w.kv("", cfg.max_miss, ", \"min_hits\": ");
    w.kv("", cfg.min_hits, ", \"anchor_mode\": ");
    w.kv("", cfg.anchor_mode, "}");
}

/*
 * One entry point for every AT+CNT* command. Registered through a single lambda
 * expression (see main_task.hpp) so all seven commands share one closure type
 * and one executor task instantiation.
 */
PC_SMALL inline void counter_cmd(const std::vector<std::string>& argv, void* caller, uint8_t kind) {
    bool      ok = true;
    PcWriter& w  = pc_writer();

    switch (kind) {
    case PC_CMD_SET_LINE:
        ok = pc_counter().set_line((int)std::atoi(argv[1].c_str()),
                                   (int16_t)std::atoi(argv[2].c_str()),
                                   (int16_t)std::atoi(argv[3].c_str()),
                                   (int16_t)std::atoi(argv[4].c_str()),
                                   (int16_t)std::atoi(argv[5].c_str())) &&
             counter_save_config();
        break;
    case PC_CMD_SET_ROI: {
        int16_t xy[PC_ROI_VERTICES * 2];
        for (int i = 0; i < PC_ROI_VERTICES * 2; ++i) xy[i] = (int16_t)std::atoi(argv[i + 2].c_str());
        ok = pc_counter().set_roi((int)std::atoi(argv[1].c_str()), xy) && counter_save_config();
    } break;
    case PC_CMD_SET_CFG:
        ok = pc_counter().set_cfg(std::atoi(argv[1].c_str()),
                                  std::atoi(argv[2].c_str()),
                                  std::atoi(argv[3].c_str()),
                                  std::atoi(argv[4].c_str())) &&
             counter_save_config();
        break;
    case PC_CMD_RESET:
        pc_counter().reset_counts();
        break;
    default:
        break;
    }

    w.init(pc_text_buf(), PC_TEXT_CAP);
    w.str("\r{\"type\": 0, \"name\": \"");
    w.str(argv[0].c_str());
    w.kv("\", \"code\": ", ok ? EL_OK : EL_EINVAL, ", \"data\": ");
    switch (kind) {
    case PC_CMD_SET_LINE:
    case PC_CMD_GET_LINE:
        counter_write_lines(w);
        break;
    case PC_CMD_SET_ROI:
    case PC_CMD_GET_ROI:
        counter_write_rois(w);
        break;
    default:
        counter_write_cfg(w);
        break;
    }
    w.str("}\n");

    static_cast<Transport*>(caller)->send_bytes(w.buf, (size_t)w.n);
}

}  // namespace sscma::callback
