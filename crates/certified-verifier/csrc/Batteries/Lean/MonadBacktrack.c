// Lean compiler output
// Module: Batteries.Lean.MonadBacktrack
// Imports: public import Init public meta import Init public import Lean.Util.MonadBacktrack
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
lean_object* l_Lean_withoutModifyingState___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_withoutModifyingState_x27___redArg___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_withoutModifyingState_x27___redArg___lam__1(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_withoutModifyingState_x27___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_withoutModifyingState_x27(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_withoutModifyingState_x27___redArg___lam__0(lean_object* v_result_1_, lean_object* v_toPure_2_, lean_object* v_finalState_3_){
_start:
{
lean_object* v___x_4_; lean_object* v___x_5_; 
v___x_4_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_4_, 0, v_result_1_);
lean_ctor_set(v___x_4_, 1, v_finalState_3_);
v___x_5_ = lean_apply_2(v_toPure_2_, lean_box(0), v___x_4_);
return v___x_5_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_withoutModifyingState_x27___redArg___lam__1(lean_object* v_inst_6_, lean_object* v_toPure_7_, lean_object* v_toBind_8_, lean_object* v_result_9_){
_start:
{
lean_object* v_saveState_10_; lean_object* v___f_11_; lean_object* v___x_12_; 
v_saveState_10_ = lean_ctor_get(v_inst_6_, 0);
lean_inc(v_saveState_10_);
lean_dec_ref(v_inst_6_);
v___f_11_ = lean_alloc_closure((void*)(lp_batteries_Lean_withoutModifyingState_x27___redArg___lam__0), 3, 2);
lean_closure_set(v___f_11_, 0, v_result_9_);
lean_closure_set(v___f_11_, 1, v_toPure_7_);
v___x_12_ = lean_apply_4(v_toBind_8_, lean_box(0), lean_box(0), v_saveState_10_, v___f_11_);
return v___x_12_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_withoutModifyingState_x27___redArg(lean_object* v_inst_13_, lean_object* v_inst_14_, lean_object* v_inst_15_, lean_object* v_x_16_){
_start:
{
lean_object* v_toApplicative_17_; lean_object* v_toBind_18_; lean_object* v_toPure_19_; lean_object* v___f_20_; lean_object* v___x_21_; lean_object* v___x_22_; 
v_toApplicative_17_ = lean_ctor_get(v_inst_13_, 0);
v_toBind_18_ = lean_ctor_get(v_inst_13_, 1);
v_toPure_19_ = lean_ctor_get(v_toApplicative_17_, 1);
lean_inc_n(v_toBind_18_, 2);
lean_inc(v_toPure_19_);
lean_inc_ref(v_inst_14_);
v___f_20_ = lean_alloc_closure((void*)(lp_batteries_Lean_withoutModifyingState_x27___redArg___lam__1), 4, 3);
lean_closure_set(v___f_20_, 0, v_inst_14_);
lean_closure_set(v___f_20_, 1, v_toPure_19_);
lean_closure_set(v___f_20_, 2, v_toBind_18_);
v___x_21_ = lean_apply_4(v_toBind_18_, lean_box(0), lean_box(0), v_x_16_, v___f_20_);
v___x_22_ = l_Lean_withoutModifyingState___redArg(v_inst_13_, v_inst_15_, v_inst_14_, v___x_21_);
return v___x_22_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_withoutModifyingState_x27(lean_object* v_m_23_, lean_object* v_s_24_, lean_object* v_00_u03b1_25_, lean_object* v_inst_26_, lean_object* v_inst_27_, lean_object* v_inst_28_, lean_object* v_x_29_){
_start:
{
lean_object* v___x_30_; 
v___x_30_ = lp_batteries_Lean_withoutModifyingState_x27___redArg(v_inst_26_, v_inst_27_, v_inst_28_, v_x_29_);
return v___x_30_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_Lean_Util_MonadBacktrack(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_batteries_Batteries_Lean_MonadBacktrack(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_Lean_Util_MonadBacktrack(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_batteries_Batteries_Lean_MonadBacktrack(uint8_t builtin) {
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
lean_object* initialize_Lean_Util_MonadBacktrack(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_batteries_Batteries_Lean_MonadBacktrack(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Lean_Util_MonadBacktrack(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_batteries_Batteries_Lean_MonadBacktrack(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_batteries_Batteries_Lean_MonadBacktrack(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_batteries_Batteries_Lean_MonadBacktrack(builtin);
}
#ifdef __cplusplus
}
#endif
