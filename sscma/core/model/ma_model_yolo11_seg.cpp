#include "ma_model_yolo11_seg.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <forward_list>
#include <math.h>
#include <numeric>
#include <utility>
#include <vector>

#include "../math/ma_math.h"
#include "../utils/ma_nms.h"

constexpr char TAG[] = "ma::model::yolo11_seg";

namespace ma::model {

Yolo11Seg::Yolo11Seg(Engine* p_engine_) : Segmentor(p_engine_, "yolo11_seg", MA_MODEL_TYPE_YOLO11_SEG) {
    MA_ASSERT(p_engine_ != nullptr);

    bboxes_ = p_engine_->getOutput(0);
    protos_ = p_engine_->getOutput(1);

    num_class_  = bboxes_.shape.dims[1] - 36;  // 4 + 1 + 32
    num_record_ = bboxes_.shape.dims[2];
}

Yolo11Seg::~Yolo11Seg() {}

bool Yolo11Seg::isValid(Engine* engine) {

    const auto inputs_count  = engine->getInputSize();
    const auto outputs_count = engine->getOutputSize();

    if (inputs_count != 1 || outputs_count != 2) {
        return false;
    }
    const auto& input_shape  = engine->getInputShape(0);
    const auto& output_shape = engine->getOutputShape(0);
    const auto& mask_shape   = engine->getOutputShape(1);

    // Validate input shape
    if (input_shape.size != 4) {
        return false;
    }

    int n = input_shape.dims[0], h = input_shape.dims[1], w = input_shape.dims[2], c = input_shape.dims[3];
    bool is_nhwc = c == 3 || c == 1;

    if (!is_nhwc)
        std::swap(h, c);


    if (n != 1 || h < 32 || h % 32 != 0 || (c != 3 && c != 1)) {
        return false;
    }

    // Calculate expected output size based on input
    int s = w >> 5, m = w >> 4, l = w >> 3;
    int ibox_len = (s * s + m * m + l * l);

    // Validate output shape
    if ((output_shape.size != 3 && output_shape.size != 4) || mask_shape.size != 4) {
        return false;
    }

    if (output_shape.dims[0] != 1 || output_shape.dims[2] != ibox_len || output_shape.dims[1] < 37) {
        return false;
    }

    if (mask_shape.dims[0] != 1 || mask_shape.dims[1] != 32 || mask_shape.dims[2] != w >> 2 || mask_shape.dims[3] != w >> 2) {
        return false;
    }

    return true;
}

ma_err_t Yolo11Seg::postprocess() {
    results_.clear();
    if (bboxes_.type == MA_TENSOR_TYPE_F32) {
        return postProcessF32();
    }
    return MA_ENOTSUP;
}

ma_err_t Yolo11Seg::postProcessF32() {

    std::forward_list<ma_bbox_ext_t> multi_level_bboxes;
    auto* data = bboxes_.data.f32;

    // Per-record max over classes, one class plane at a time: the planes are
    // contiguous, so this reads memory sequentially instead of striding
    // num_record_ floats per class for every record.
    const float* scores = data + num_record_ * 4;
    cell_max_f32_.assign(num_record_, -INFINITY);
    float* cell_max = cell_max_f32_.data();
    for (int c = 0; c < num_class_; c++) {
        const float* plane = scores + num_record_ * c;
        for (decltype(num_record_) n = 0; n < num_record_; n++) {
            cell_max[n] = plane[n] > cell_max[n] ? plane[n] : cell_max[n];
        }
    }

    for (decltype(num_record_) i = 0; i < num_record_; ++i) {

        // Kept when some class score is >= threshold_score_, as before.
        if (cell_max[i] < threshold_score_)
            continue;

        // The last class holding the maximum, as the per-record scan picked.
        int target = num_class_ - 1;
        while (scores[num_record_ * target + i] != cell_max[i]) {
            target--;
        }
        // The score at target, as the scan kept it (cell_max[i] can differ in the sign of a zero).
        float max = scores[num_record_ * target + i];

        float x = data[i];
        float y = data[i + num_record_];
        float w = data[i + num_record_ * 2];
        float h = data[i + num_record_ * 3];


        ma_bbox_ext_t bbox;
        bbox.level  = 0;
        bbox.index  = i;
        bbox.x      = x / img_.width;
        bbox.y      = y / img_.height;
        bbox.w      = w / img_.width;
        bbox.h      = h / img_.height;
        bbox.score  = max;
        bbox.target = target;

        multi_level_bboxes.emplace_front(std::move(bbox));
    }

    ma::utils::nms(multi_level_bboxes, threshold_nms_, threshold_score_, false, true);

    if (multi_level_bboxes.empty())
        return MA_OK;

    // fetch mask
    for (auto& bbox : multi_level_bboxes) {
        ma_segm2f_t seg;
        seg.box         = {.x = bbox.x, .y = bbox.y, .w = bbox.w, .h = bbox.h, .score = bbox.score, .target = bbox.target};
        seg.mask.width  = protos_.shape.dims[2];
        seg.mask.height = protos_.shape.dims[3];
        seg.mask.data.resize(protos_.shape.dims[2] * protos_.shape.dims[3] / 8, 0);  // bitwise

        const int mask_size = protos_.shape.dims[2] * protos_.shape.dims[3];

        std::vector<float> masks(mask_size, 0.0f);

        // TODO: parallel for
        for (int j = 0; j < protos_.shape.dims[1]; ++j) {
            float mask_in = bboxes_.data.f32[bbox.index + num_record_ * (4 + num_class_ + j)];
            for (int i = 0; i < mask_size; ++i) {
                masks[i] += mask_in * protos_.data.f32[j * mask_size + i];
            }
        }

        int x1 = (bbox.x - bbox.w / 2) * protos_.shape.dims[2];
        int y1 = (bbox.y - bbox.h / 2) * protos_.shape.dims[3];
        int x2 = (bbox.x + bbox.w / 2) * protos_.shape.dims[2];
        int y2 = (bbox.y + bbox.h / 2) * protos_.shape.dims[3];

        for (int i = 0; i < protos_.shape.dims[2]; i++) {
            for (int j = 0; j < protos_.shape.dims[3]; j++) {
                if (i < y1 || i >= y2 || j < x1 || j >= x2) [[likely]] {
                    continue;
                }
                if (masks[i * protos_.shape.dims[3] + j] > 0.5) {
                    seg.mask.data[i * protos_.shape.dims[3] / 8 + j / 8] |= (1 << (j % 8));
                }
            }
        }

        results_.emplace_front(std::move(seg));
    }


    return MA_OK;
}

}  // namespace ma::model
