// Lean compiler output
// Module: Batteries.Data.List.Count
// Imports: public import Init public meta import Init public import Batteries.Data.List.Lemmas
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
lean_object* l_List_get___redArg(lean_object*, lean_object*);
lean_object* lp_batteries_List_countBefore___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_batteries_List_idxOfNth___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_List_idxToSigmaCount___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_List_idxToSigmaCount(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_List_sigmaCountToIdx___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_List_sigmaCountToIdx(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_List_idxToSigmaCount___redArg(lean_object* v_inst_1_, lean_object* v_xs_2_, lean_object* v_i_3_){
_start:
{
lean_object* v___x_4_; lean_object* v___x_5_; lean_object* v___x_6_; 
lean_inc(v_i_3_);
v___x_4_ = l_List_get___redArg(v_xs_2_, v_i_3_);
lean_inc(v___x_4_);
v___x_5_ = lp_batteries_List_countBefore___redArg(v_inst_1_, v___x_4_, v_xs_2_, v_i_3_);
v___x_6_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_6_, 0, v___x_4_);
lean_ctor_set(v___x_6_, 1, v___x_5_);
return v___x_6_;
}
}
LEAN_EXPORT lean_object* lp_batteries_List_idxToSigmaCount(lean_object* v_00_u03b1_7_, lean_object* v_inst_8_, lean_object* v_inst_9_, lean_object* v_xs_10_, lean_object* v_i_11_){
_start:
{
lean_object* v___x_12_; 
v___x_12_ = lp_batteries_List_idxToSigmaCount___redArg(v_inst_8_, v_xs_10_, v_i_11_);
return v___x_12_;
}
}
LEAN_EXPORT lean_object* lp_batteries_List_sigmaCountToIdx___redArg(lean_object* v_inst_13_, lean_object* v_xs_14_, lean_object* v_xc_15_){
_start:
{
lean_object* v_fst_16_; lean_object* v_snd_17_; lean_object* v___x_18_; 
v_fst_16_ = lean_ctor_get(v_xc_15_, 0);
lean_inc(v_fst_16_);
v_snd_17_ = lean_ctor_get(v_xc_15_, 1);
lean_inc(v_snd_17_);
lean_dec_ref(v_xc_15_);
v___x_18_ = lp_batteries_List_idxOfNth___redArg(v_inst_13_, v_fst_16_, v_xs_14_, v_snd_17_);
return v___x_18_;
}
}
LEAN_EXPORT lean_object* lp_batteries_List_sigmaCountToIdx(lean_object* v_00_u03b1_19_, lean_object* v_inst_20_, lean_object* v_xs_21_, lean_object* v_xc_22_){
_start:
{
lean_object* v___x_23_; 
v___x_23_ = lp_batteries_List_sigmaCountToIdx___redArg(v_inst_20_, v_xs_21_, v_xc_22_);
return v___x_23_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_batteries_Batteries_Data_List_Lemmas(uint8_t builtin);
void lean_initialize_runtime_module();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_batteries_Batteries_Data_List_Count(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize_runtime_module();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_batteries_Batteries_Data_List_Lemmas(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_batteries_Batteries_Data_List_Count(uint8_t builtin) {
lean_object * res;
if (_G_meta_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_meta_initialized = true;
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_batteries_Batteries_Data_List_Lemmas(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_batteries_Batteries_Data_List_Count(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_batteries_Batteries_Data_List_Lemmas(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_batteries_Batteries_Data_List_Count(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_batteries_Batteries_Data_List_Count(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_batteries_Batteries_Data_List_Count(builtin);
}
#ifdef __cplusplus
}
#endif
