#include "model/diffusion/cosmos3.hpp"
#include "conditioning/conditioner.hpp"
#include "model_definition.h"
#include "pipeline/model_builders.h"

static bool build_cosmos3(const sd::model_builders::Context& ctx, const TokenizerConfig& tokenizers, sd::model_builders::CoreRunners& result) {
    SD_UNUSED(tokenizers);
    result.conditioner = std::make_shared<Cosmos3Conditioner>();
    result.diffusion   = std::make_shared<Cosmos3::Cosmos3Runner>(ctx.backends.runtime_backend(SDBackendModule::DIFFUSION),
                                                                  ctx.tensor_storage_map,
                                                                  "model.diffusion_model",
                                                                  ctx.weight_manager,
                                                                  ctx.params.model_args);
    return true;
}

const std::vector<ModelDefinition>& cosmos3_definitions() {
    static const std::vector<ModelDefinition> definitions = [] {
        std::vector<ModelDefinition> result;
        {
            // Sampling is fixed by the transfer pipeline (4 distilled steps, no CFG), so the policy is unused.
            ModelDefinition definition;
            definition.version                          = VERSION_COSMOS3;
            definition.geometry                         = {2, 48, 3};
            definition.name                             = "Cosmos3";
            definition.traits.is_dit                    = true;
            definition.traits.supports_video_generation = true;
            definition.traits.vae_kind                  = ModelVAEKind::WAN;
            definition.traits.video_tae                 = true;
            definition.sampling                         = {FLOW_PRED};
            definition.sampling.default_sample_method   = EULER_SAMPLE_METHOD;
            definition.sampling.default_scheduler       = DISCRETE_SCHEDULER;
            definition.build                            = build_cosmos3;
            result.push_back(std::move(definition));
        }
        return result;
    }();
    return definitions;
}
