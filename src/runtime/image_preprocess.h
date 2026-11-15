#ifndef __SD_RUNTIME_IMAGE_PREPROCESS_H__
#define __SD_RUNTIME_IMAGE_PREPROCESS_H__

#include <map>
#include <string>
#include <utility>
#include <vector>

#include "core/tensor.hpp"
#include "stable-diffusion.h"

namespace sd {

    std::pair<int, int> select_image_size_bucket(int source_width, int source_height, int area, int factor, int fallback_width, int fallback_height);
    Tensor<float> resize_and_center_crop(const Tensor<float>& image, int width, int height, ops::InterpolateMode filter = ops::InterpolateMode::Lanczos);
    struct CropWindow {
        int x      = 0;
        int y      = 0;
        int width  = 0;  // <= 0: the whole picture
        int height = 0;
    };
    // Pillow's LANCZOS resize of an 8-bit picture (or of a crop window of it): values are scaled to 0..255 and
    // truncated, resized in Pillow's fixed point (horizontal pass, then vertical, each rounded to 8 bits) and read
    // back as value / 255.
    Tensor<float> resize_lanczos_8bit(const Tensor<float>& image, int width, int height, CropWindow crop = {});
    // Aspect-preserving size that puts the short edge inside [min_edge, max_edge], each side rounded (halves to even)
    // to a multiple of `multiple`, at least one multiple.
    std::pair<int, int> fit_short_edge_size(int width, int height, int min_edge, int max_edge, int multiple);
    // torch's "area" interpolation (adaptive average pooling): each output value is the float sum of its input box,
    // row by row, divided by the box height and then by its width.
    Tensor<float> resize_area(const Tensor<float>& image, int width, int height);
    // Scale to `megapixels` (units of 2^20 pixels) with the area filter, then cover the nearest smaller multiple of
    // `multiple` on each side and crop the centre. The size alone, then the picture.
    std::pair<int, int> scale_total_pixels_to_multiple_size(int width, int height, double megapixels, int multiple);
    Tensor<float> scale_total_pixels_to_multiple(const Tensor<float>& image, double megapixels, int multiple);
    // The centred window with the aspect ratio of width x height, as the reference's centre crop picks it.
    CropWindow center_crop_window(int image_width, int image_height, int width, int height);

    enum class ImageTarget {
        Init,
        End,
        Mask,
        Control,
        Ref,
        IPAdapter,
        ID,
        ControlFrame,
    };

    enum class ImageResizeMode {
        Auto,
        None,
        Stretch,
        Crop,
        CropResize,
        FitPad,
    };

    struct ImageTransform {
        int source_width  = 0;
        int source_height = 0;
        int x             = 0;
        int y             = 0;
        int crop_width    = 0;
        int crop_height   = 0;
        int resize_width  = 0;
        int resize_height = 0;
        int width         = 0;
        int height        = 0;
        int pad_x         = 0;
        int pad_y         = 0;
    };

    struct ImagePreprocessRule {
        ImageTarget target = ImageTarget::Init;
        int index          = -1;
        std::map<std::string, std::string> options;
    };

    class ImagePreprocessor {
        std::vector<ImagePreprocessRule> rules_;
        mutable bool valid_ = true;
        ImageTransform init_transform_;
        bool has_init_transform_ = false;
        bool prepared_           = false;
        std::vector<sd_image_t> owned_images_;
        std::vector<sd_image_t> ref_images_;
        std::vector<sd_image_t> id_images_;
        std::vector<sd_image_t> control_frames_;

        bool fail(const std::string& message) const;
        std::map<std::string, std::string> resolve_options(ImageTarget target, int index) const;
        Tensor<float> apply_transform(const Tensor<float>& image, const std::map<std::string, std::string>& options, ImageTransform plan, const std::string& label, ops::InterpolateMode default_filter) const;

        bool prepare_image(sd_image_t& image, ImageTarget target, int index, int width, int height);
        bool prepare_array(sd_image_t*& images, int count, ImageTarget target, std::vector<sd_image_t>& storage, int width = -1, int height = -1);

    public:
        explicit ImagePreprocessor(const char* rules = nullptr);
        ~ImagePreprocessor();
        ImagePreprocessor(const ImagePreprocessor&)            = delete;
        ImagePreprocessor& operator=(const ImagePreprocessor&) = delete;
        bool prepare_inputs(sd_img_gen_params_t& params, int width, int height);
        bool prepare_inputs(sd_vid_gen_params_t& params, int width, int height);
        bool is_valid() const { return valid_; }
        bool validate_inputs(const sd_img_gen_params_t& params) const;
        bool validate_inputs(const sd_vid_gen_params_t& params) const;
        Tensor<float> apply_geometry(const Tensor<float>& image, ImageTarget target, int index, int width, int height, ImageResizeMode default_mode = ImageResizeMode::Stretch, ops::InterpolateMode default_filter = ops::InterpolateMode::Nearest, ImageTransform* plan_out = nullptr) const;
        Tensor<float> preprocess_input(sd_image_t image, ImageTarget target, int index = 0, int width = -1, int height = -1);
    };

}  // namespace sd

#endif  // __SD_RUNTIME_IMAGE_PREPROCESS_H__
