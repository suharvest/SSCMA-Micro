#ifndef _MA_MODEL_BISENETV2_H_
#define _MA_MODEL_BISENETV2_H_

#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector>

#include "ma_model_segmentor.h"

namespace ma::model {

class BiSeNetV2 : public Segmentor {
private:
    ma_tensor_t output_;
    int32_t num_classes_;
    int32_t mask_h_;
    int32_t mask_w_;

    // Semantic segmentation label map (H x W, per-pixel class ID)
    std::vector<uint8_t> labels_;
    std::vector<int8_t> max_vals_s8_;
    std::vector<float> max_vals_f32_;
    bool is_int8_output_;

protected:
    ma_err_t postprocess() override;

public:
    BiSeNetV2(Engine* engine);
    ~BiSeNetV2();

    static bool isValid(Engine* engine);

    const std::vector<uint8_t>& getLabels() const;
    int getNumClasses() const;
};

}  // namespace ma::model

#endif  // _MA_MODEL_BISENETV2_H_
