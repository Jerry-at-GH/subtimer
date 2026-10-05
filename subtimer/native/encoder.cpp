#include "parakeet.cpp"

extern "C" void subtimer_parakeet_release_workspace(parakeet_context* ctx) {
    if (ctx->sched) {
        ggml_backend_sched_free(ctx->sched);
        ctx->sched = nullptr;
    }
}
static ggml_cgraph* subtimer_encoder_graph(parakeet_context* ctx, int T_mel) {
    const auto& m = ctx->model;
    const auto& hp = m.hparams;
    const int n_mels = (int)hp.n_mels;

    ggml_init_params ip = {
         ctx->compute_meta.size(),
         ctx->compute_meta.data(),
         true,
    };
    ggml_context* ctx0 = ggml_init(ip);
    ggml_cgraph* gf = ggml_new_graph_custom(ctx0, 8192, false);

    const int T = (T_mel + 7) / 8;
    ggml_tensor* cur = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, hp.d_model, T);
    ggml_set_name(cur, "preencoded");
    ggml_set_input(cur);


    if (hp.xscaling) {
        const float xscale = sqrtf((float)hp.d_model);
        cur = ggml_scale(ctx0, cur, xscale);
    }


    ggml_tensor* pos_enc = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, (int)hp.d_model, 2 * T - 1);
    ggml_set_name(pos_enc, "pos_enc");
    ggml_set_input(pos_enc);


    ggml_tensor* local_mask = nullptr;
    ggml_tensor* window_band_mask = nullptr;
    const bool use_local_attn =
        hp.att_context_left >= 0 && hp.att_context_right >= 0 && T > hp.att_context_left + hp.att_context_right + 1;


    const bool use_windowed_attn =
        core_conformer::fc_windowed_attn() &&
        core_conformer::fc_window_attn_applicable(T, hp.att_context_left, hp.att_context_right);
    if (getenv("CRISPASR_FC_MEM_DEBUG"))
        fprintf(stderr, "[fc-encode] T=%d windowed=%d local=%d\n", T, (int)use_windowed_attn, (int)use_local_attn);
    if (use_windowed_attn) {
        const int BS = core_conformer::fc_window_block_size(hp.att_context_left, hp.att_context_right);
        const int NB = (T + BS - 1) / BS;
        window_band_mask = ggml_new_tensor_3d(ctx0, GGML_TYPE_F32, 3 * BS, BS, NB);
        ggml_set_name(window_band_mask, "window_band_mask");
        ggml_set_input(window_band_mask);
    } else if (use_local_attn) {
        local_mask = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, T, T);
        ggml_set_name(local_mask, "local_attn_mask");
        ggml_set_input(local_mask);
    }


    core_conformer::BlockParams bp = {
        (int)hp.d_model, (int)hp.n_heads,     (int)hp.head_dim,     (int)hp.conv_kernel,
        kLayerNormEps,   hp.att_context_left, hp.att_context_right, (int)hp.global_tokens,
    };
    bp.manual_attn = core_conformer::fc_gpu_manual_attn(ctx->backend);
    for (uint32_t il = 0; il < hp.n_layers; il++) {
        cur = core_conformer::build_block(ctx0, cur, pos_enc, T, m.enc[il], bp, local_mask, nullptr, window_band_mask);
    }

    ggml_set_name(cur, "enc_out");
    ggml_build_forward_expand(gf, cur);
    ggml_free(ctx0);
    return gf;
}

static std::vector<float> subtimer_encode_features(parakeet_context* ctx, const float* mel, int n_mels, int T_mel,
                                              int* out_T_enc) {
    if (n_mels != (int)ctx->model.hparams.n_mels) {
        fprintf(stderr, "parakeet: mel feature mismatch (%d vs %d)\n", n_mels, (int)ctx->model.hparams.n_mels);
        return {};
    }

    if (!ctx->sched) {
        ggml_backend_t backends[2] = {ctx->backend, ctx->backend_cpu};
        int n_be = (ctx->backend != ctx->backend_cpu) ? 2 : 1;
        ctx->sched = ggml_backend_sched_new(backends, nullptr, n_be, 8192, false, false);
        crispasr_imatrix_install(ctx->sched);
    }
    if (ctx->compute_meta.empty()) {
        ctx->compute_meta.resize(ggml_tensor_overhead() * 8192 + ggml_graph_overhead_custom(8192, false));
    }

    ggml_cgraph* gf = subtimer_encoder_graph(ctx, T_mel);
    ggml_backend_sched_reset(ctx->sched);
    if (!ggml_backend_sched_alloc_graph(ctx->sched, gf)) {
        fprintf(stderr, "parakeet: failed to alloc encoder graph\n");
        return {};
    }


    ggml_tensor* mel_in = ggml_graph_get_tensor(gf, "preencoded");
    ggml_backend_tensor_set(mel_in, mel, 0, (size_t)ctx->model.hparams.d_model * ((T_mel + 7) / 8) * sizeof(float));

    ggml_tensor* pos_in = ggml_graph_get_tensor(gf, "pos_enc");
    int T_enc = (int)pos_in->ne[1];
    T_enc = (T_enc + 1) / 2;
    auto pe = core_conformer::make_pos_enc((int)ctx->model.hparams.d_model, T_enc);
    ggml_backend_tensor_set(pos_in, pe.data(), 0, pe.size() * sizeof(float));


    ggml_tensor* local_mask_in = ggml_graph_get_tensor(gf, "local_attn_mask");
    if (local_mask_in) {
        const auto& hp = ctx->model.hparams;
        auto lm = core_conformer::make_local_attn_mask(T_enc, hp.att_context_left, hp.att_context_right,
                                                       (int)hp.global_tokens);
        ggml_backend_tensor_set(local_mask_in, lm.data(), 0, lm.size() * sizeof(float));
    }


    ggml_tensor* band_mask_in = ggml_graph_get_tensor(gf, "window_band_mask");
    if (band_mask_in) {
        const auto& hp = ctx->model.hparams;
        const int BS = core_conformer::fc_window_block_size(hp.att_context_left, hp.att_context_right);
        auto bm = core_conformer::make_window_band_mask(T_enc, hp.att_context_left, hp.att_context_right,
                                                        (int)hp.global_tokens, BS);
        ggml_backend_tensor_set(band_mask_in, bm.data(), 0, bm.size() * sizeof(float));
    }


    if (ggml_backend_sched_graph_compute(ctx->sched, gf) != GGML_STATUS_SUCCESS) {
        fprintf(stderr, "parakeet: encoder graph compute failed\n");
        return {};
    }

    ggml_tensor* out = ggml_graph_get_tensor(gf, "enc_out");
    if (!out) {
        fprintf(stderr, "parakeet: missing enc_out tensor\n");
        return {};
    }
    const int d = (int)out->ne[0];
    const int Te = (int)out->ne[1];
    if (out_T_enc)
        *out_T_enc = Te;

    std::vector<float> result((size_t)d * Te);
    ggml_backend_tensor_get(out, result.data(), 0, result.size() * sizeof(float));
    return result;
}


static std::vector<float> subtimer_preencode(parakeet_context* ctx, const std::vector<float>& mel, int T_mel, int tile) {
    const auto& hp = ctx->model.hparams;
    const int frames = (T_mel + 7) / 8;
    const int d = hp.d_model;
    std::vector<float> output((size_t)frames*d);
    ggml_backend_t backends[2] = {ctx->backend, ctx->backend_cpu};
    auto sched = ggml_backend_sched_new(backends, nullptr, ctx->backend != ctx->backend_cpu ? 2 : 1, 8192, false, false);
    std::vector<uint8_t> meta(ggml_tensor_overhead()*8192+ggml_graph_overhead_custom(8192,false));
    for (int first=0; first<frames; first+=tile/8) {
        const int last=std::min(frames, first+tile/8);
        const int start=std::max(0, first*8-8);
        const int stop=std::min(T_mel, last*8+8);
        ggml_backend_sched_reset(sched);
        ggml_init_params params={meta.size(),meta.data(),true};
        auto gctx=ggml_init(params);
        auto gf=ggml_new_graph_custom(gctx,8192,false);
        auto input=ggml_new_tensor_2d(gctx,GGML_TYPE_F32,hp.n_mels,stop-start);
        ggml_set_input(input);
        int count=0;
        auto out=core_conformer::build_pre_encode(gctx,input,ctx->model.pre_encode,hp.subsampling_channels,&count);
        ggml_build_forward_expand(gf,out);
        if (!ggml_backend_sched_alloc_graph(sched,gf)) {ggml_free(gctx);ggml_backend_sched_free(sched);return {};}
        ggml_backend_tensor_set(input,mel.data()+(size_t)start*hp.n_mels,0,(size_t)(stop-start)*hp.n_mels*sizeof(float));
        if (ggml_backend_sched_graph_compute(sched,gf)!=GGML_STATUS_SUCCESS) {ggml_free(gctx);ggml_backend_sched_free(sched);return {};}
        ggml_backend_tensor_get(out,output.data()+(size_t)first*d,(size_t)(first-start/8)*d*sizeof(float),(size_t)(last-first)*d*sizeof(float));
        ggml_free(gctx);
    }
    ggml_backend_sched_free(sched);
    return output;
}
extern "C" float* subtimer_parakeet_encode(parakeet_context* ctx,const float* samples,int n_samples,int tile,int* out_T,int* out_d) {
    if (!ctx || !samples || n_samples<=0 || tile<8 || tile%8) return nullptr;
    int T_mel=0;
    auto mel=parakeet_compute_mel_impl(ctx,samples,n_samples,T_mel);
    auto features=subtimer_preencode(ctx,mel,T_mel,tile);
    if (features.empty()) return nullptr;
    auto encoded=subtimer_encode_features(ctx,features.data(),ctx->model.hparams.n_mels,T_mel,out_T);
    if (encoded.empty()) return nullptr;
    *out_d=ctx->model.hparams.d_model;
    float* result=(float*)malloc(encoded.size()*sizeof(float));
    memcpy(result,encoded.data(),encoded.size()*sizeof(float));
    return result;
}
