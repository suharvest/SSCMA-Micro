#include "ma_model_bisenetv2.h"
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <forward_list>
#include <utility>
#include <vector>
#include <cstring>
#include "../math/ma_math.h"

constexpr char TAG[] = "ma::model::bisenetv2";

namespace ma::model {

BiSeNetV2::BiSeNetV2(Engine* p_engine_) : Segmentor(p_engine_, "bisenetv2", MA_MODEL_TYPE_BISENETV2) {
    MA_ASSERT(p_engine_ != nullptr);

    output_ = p_engine_->getOutput(0);
    is_int8_output_ = (output_.type == MA_TENSOR_TYPE_S8);

    // Output shape: [1, num_classes, H, W] (NCHW)
    if (output_.shape.size == 4) {
        num_classes_ = output_.shape.dims[1];
        mask_h_      = output_.shape.dims[2];
        mask_w_      = output_.shape.dims[3];
    } else if (output_.shape.size == 3) {
        num_classes_ = output_.shape.dims[0];
        mask_h_      = output_.shape.dims[1];
        mask_w_      = output_.shape.dims[2];
    } else {
        num_classes_ = 0;
        mask_h_      = 0;
        mask_w_      = 0;
    }

    labels_.resize(mask_h_ * mask_w_, 0);
    max_vals_s8_.resize(mask_h_ * mask_w_);
    max_vals_f32_.resize(mask_h_ * mask_w_);
}

BiSeNetV2::~BiSeNetV2() {}

bool BiSeNetV2::isValid(Engine* engine) {
    const auto inputs_count  = engine->getInputSize();
    const auto outputs_count = engine->getOutputSize();

    if (inputs_count != 1 || outputs_count < 1) {
        return false;
    }

    const auto& output_shape = engine->getOutputShape(0);

    if (output_shape.size != 3 && output_shape.size != 4) {
        return false;
    }

    int n, c, h, w;
    if (output_shape.size == 4) {
        n = output_shape.dims[0];
        c = output_shape.dims[1];
        h = output_shape.dims[2];
        w = output_shape.dims[3];
    } else {
        n = 1;
        c = output_shape.dims[0];
        h = output_shape.dims[1];
        w = output_shape.dims[2];
    }

    if (n != 1 || c < 2 || h < 32 || w < 32) {
        return false;
    }

    // BiSeNetV2: 1 output, [1, C, H, W] spatial dims >> C
    // YOLO11 seg: 2 outputs, [1, 4+1+32+C, anchors]
    // Classifier: 1 output, [1, C] (no spatial dims)
    if (outputs_count != 1) {
        return false;
    }

    return true;
}

ma_err_t BiSeNetV2::postprocess() {
    results_.clear();

    int pixel_count = mask_h_ * mask_w_;

    // Per-class bbox accumulators
    struct ClsInfo {
        int x1, y1, x2, y2, count;
    };
    ClsInfo* cls_info = new ClsInfo[num_classes_];
    for (int c = 0; c < num_classes_; ++c) {
        cls_info[c] = {mask_w_, mask_h_, 0, 0, 0};
    }

    // Pass 1: channel-major argmax (cache-friendly sequential access)
    if (is_int8_output_) {
        auto* data = output_.data.s8;
        // Separate buffer for running max (labels_ stores class indices)
        memcpy(max_vals_s8_.data(), data, pixel_count);
        for (int i = 0; i < pixel_count; ++i) labels_[i] = 0;
        for (int c = 1; c < num_classes_; ++c) {
            auto* ch = data + (size_t)c * pixel_count;
            for (int i = 0; i < pixel_count; ++i) {
                if (ch[i] > max_vals_s8_[i]) {
                    labels_[i] = static_cast<uint8_t>(c);
                    max_vals_s8_[i] = ch[i];
                }
            }
        }
    } else if (output_.type == MA_TENSOR_TYPE_F32) {
        auto* data = output_.data.f32;
        memcpy(max_vals_f32_.data(), data, pixel_count * sizeof(float));
        for (int i = 0; i < pixel_count; ++i) labels_[i] = 0;
        for (int c = 1; c < num_classes_; ++c) {
            auto* ch = data + (size_t)c * pixel_count;
            for (int i = 0; i < pixel_count; ++i) {
                if (ch[i] > max_vals_f32_[i]) {
                    labels_[i] = static_cast<uint8_t>(c);
                    max_vals_f32_[i] = ch[i];
                }
            }
        }
    } else {
        delete[] cls_info;
        return MA_ENOTSUP;
    }

    // Pass 2: single scan labels_ → bbox for all classes
    for (int i = 0; i < pixel_count; ++i) {
        int c = labels_[i];
        int x = i % mask_w_;
        int y = i / mask_w_;
        if (x < cls_info[c].x1) cls_info[c].x1 = x;
        if (x > cls_info[c].x2) cls_info[c].x2 = x;
        if (y < cls_info[c].y1) cls_info[c].y1 = y;
        if (y > cls_info[c].y2) cls_info[c].y2 = y;
        cls_info[c].count++;
    }

    // Pass 3: build masks only for non-empty classes
    for (int c = 0; c < num_classes_; ++c) {
        if (cls_info[c].count == 0) continue;

        ma_segm2f_t seg;
        seg.mask.width  = mask_w_;
        seg.mask.height = mask_h_;
        seg.mask.data.assign(pixel_count / 8 + 1, 0);

        for (int j = 0; j < mask_h_; ++j) {
            int row_off = j * mask_w_;
            for (int i = 0; i < mask_w_; ++i) {
                if (labels_[row_off + i] == c) {
                    seg.mask.data[row_off / 8 + i / 8] |= (1 << (i % 8));
                }
            }
        }

        float cx = (cls_info[c].x1 + cls_info[c].x2 + 1) / 2.0f / mask_w_;
        float cy = (cls_info[c].y1 + cls_info[c].y2 + 1) / 2.0f / mask_h_;
        float bw = (cls_info[c].x2 - cls_info[c].x1 + 1) / (float)mask_w_;
        float bh = (cls_info[c].y2 - cls_info[c].y1 + 1) / (float)mask_h_;
        float conf = (float)cls_info[c].count / pixel_count;

        seg.box = {.x = cx, .y = cy, .w = bw, .h = bh, .score = conf, .target = c};

        results_.emplace_front(std::move(seg));
    }

    delete[] cls_info;
    return MA_OK;
}

const std::vector<uint8_t>& BiSeNetV2::getLabels() const {
    return labels_;
}

int BiSeNetV2::getNumClasses() const {
    return num_classes_;
}

}  // namespace ma::model
