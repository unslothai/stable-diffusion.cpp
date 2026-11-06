#ifndef __SD_PIPELINE_COSMOS3_TRANSFER_H__
#define __SD_PIPELINE_COSMOS3_TRANSFER_H__

#include "request.h"
#include "stable-diffusion.h"

class StableDiffusionGGML;

namespace sd::pipeline {

    bool generate_cosmos3_transfer_video(StableDiffusionGGML* sd,
                                         const sd_vid_gen_params_t* params,
                                         const GenerationRequest& request,
                                         sd_image_t** frames_out,
                                         int* num_frames_out);

}  // namespace sd::pipeline

#endif  // __SD_PIPELINE_COSMOS3_TRANSFER_H__
