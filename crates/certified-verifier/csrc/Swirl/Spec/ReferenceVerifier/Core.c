// Lean compiler output
// Module: Swirl.Spec.ReferenceVerifier.Core
// Imports: public import Init public meta import Init public import Fundamentals.Spec.Runtime.Config public import Fundamentals.Spec.Runtime.Core
#include <lean/lean.h>
#if defined(__clang__)
#pragma clang diagnostic ignored "-Wunused-parameter"
#pragma clang diagnostic ignored "-Wunused-label"
#elif defined(__GNUC__) && !defined(__CLANG__)
#pragma GCC diagnostic ignored "-Wunused-parameter"
#pragma GCC diagnostic ignored "-Wunused-label"
#pragma GCC diagnostic ignored "-Wunused-but-set-variable"
#endif
#ifdef __cplusplus
extern "C" {
#endif
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dfv_Swirl_Protocol_Noninteractive_TranscriptEvent_observe___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dfv_Swirl_Protocol_Noninteractive_TranscriptEvent_observe(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dfv_Swirl_Protocol_Noninteractive_TranscriptEvent_observeCommit___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dfv_Swirl_Protocol_Noninteractive_TranscriptEvent_observeCommit(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dfv_Swirl_Protocol_Noninteractive_TranscriptEvent_observe___redArg(lean_object* v_value_1_){
_start:
{
lean_object* v___x_2_; 
v___x_2_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_2_, 0, v_value_1_);
return v___x_2_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dfv_Swirl_Protocol_Noninteractive_TranscriptEvent_observe(lean_object* v_F_3_, lean_object* v_Digest_4_, lean_object* v_value_5_){
_start:
{
lean_object* v___x_6_; 
v___x_6_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_6_, 0, v_value_5_);
return v___x_6_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dfv_Swirl_Protocol_Noninteractive_TranscriptEvent_observeCommit___redArg(lean_object* v_digest_7_){
_start:
{
lean_object* v___x_8_; 
v___x_8_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_8_, 0, v_digest_7_);
return v___x_8_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dfv_Swirl_Protocol_Noninteractive_TranscriptEvent_observeCommit(lean_object* v_F_9_, lean_object* v_Digest_10_, lean_object* v_digest_11_){
_start:
{
lean_object* v___x_12_; 
v___x_12_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_12_, 0, v_digest_11_);
return v___x_12_;
}
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_swirl_x2drbr_x2dfv_Fundamentals_Spec_Runtime_Config(uint8_t builtin);
lean_object* initialize_swirl_x2drbr_x2dfv_Fundamentals_Spec_Runtime_Core(uint8_t builtin);
void lean_initialize_runtime_module();
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_swirl_x2drbr_x2dfv_Swirl_Spec_ReferenceVerifier_Core(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
lean_initialize_runtime_module();
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_swirl_x2drbr_x2dfv_Fundamentals_Spec_Runtime_Config(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_swirl_x2drbr_x2dfv_Fundamentals_Spec_Runtime_Core(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
#ifdef __cplusplus
}
#endif
