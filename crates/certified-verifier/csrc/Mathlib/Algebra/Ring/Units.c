// Lean compiler output
// Module: Mathlib.Algebra.Ring.Units
// Imports: public import Init public meta import Init public import Mathlib.Algebra.Ring.InjSurj public import Mathlib.Algebra.Group.Units.Hom public import Mathlib.Algebra.Ring.Hom.Defs
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
LEAN_EXPORT lean_object* lp_mathlib_Units_instNeg___redArg___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Units_instNeg___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Units_instNeg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Units_instNeg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Units_instHasDistribNeg___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Units_instHasDistribNeg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Units_instHasDistribNeg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Units_instNeg___redArg___lam__0(lean_object* v_inst_1_, lean_object* v_u_2_){
_start:
{
lean_object* v_val_3_; lean_object* v_inv_4_; lean_object* v___x_6_; uint8_t v_isShared_7_; uint8_t v_isSharedCheck_13_; 
v_val_3_ = lean_ctor_get(v_u_2_, 0);
v_inv_4_ = lean_ctor_get(v_u_2_, 1);
v_isSharedCheck_13_ = !lean_is_exclusive(v_u_2_);
if (v_isSharedCheck_13_ == 0)
{
v___x_6_ = v_u_2_;
v_isShared_7_ = v_isSharedCheck_13_;
goto v_resetjp_5_;
}
else
{
lean_inc(v_inv_4_);
lean_inc(v_val_3_);
lean_dec(v_u_2_);
v___x_6_ = lean_box(0);
v_isShared_7_ = v_isSharedCheck_13_;
goto v_resetjp_5_;
}
v_resetjp_5_:
{
lean_object* v___x_8_; lean_object* v___x_9_; lean_object* v___x_11_; 
lean_inc(v_inst_1_);
v___x_8_ = lean_apply_1(v_inst_1_, v_val_3_);
v___x_9_ = lean_apply_1(v_inst_1_, v_inv_4_);
if (v_isShared_7_ == 0)
{
lean_ctor_set(v___x_6_, 1, v___x_9_);
lean_ctor_set(v___x_6_, 0, v___x_8_);
v___x_11_ = v___x_6_;
goto v_reusejp_10_;
}
else
{
lean_object* v_reuseFailAlloc_12_; 
v_reuseFailAlloc_12_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_12_, 0, v___x_8_);
lean_ctor_set(v_reuseFailAlloc_12_, 1, v___x_9_);
v___x_11_ = v_reuseFailAlloc_12_;
goto v_reusejp_10_;
}
v_reusejp_10_:
{
return v___x_11_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Units_instNeg___redArg(lean_object* v_inst_14_){
_start:
{
lean_object* v___f_15_; 
v___f_15_ = lean_alloc_closure((void*)(lp_mathlib_Units_instNeg___redArg___lam__0), 2, 1);
lean_closure_set(v___f_15_, 0, v_inst_14_);
return v___f_15_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Units_instNeg(lean_object* v_00_u03b1_16_, lean_object* v_inst_17_, lean_object* v_inst_18_){
_start:
{
lean_object* v___f_19_; 
v___f_19_ = lean_alloc_closure((void*)(lp_mathlib_Units_instNeg___redArg___lam__0), 2, 1);
lean_closure_set(v___f_19_, 0, v_inst_18_);
return v___f_19_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Units_instNeg___boxed(lean_object* v_00_u03b1_20_, lean_object* v_inst_21_, lean_object* v_inst_22_){
_start:
{
lean_object* v_res_23_; 
v_res_23_ = lp_mathlib_Units_instNeg(v_00_u03b1_20_, v_inst_21_, v_inst_22_);
lean_dec_ref(v_inst_21_);
return v_res_23_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Units_instHasDistribNeg___redArg(lean_object* v_inst_24_){
_start:
{
lean_object* v___f_25_; 
v___f_25_ = lean_alloc_closure((void*)(lp_mathlib_Units_instNeg___redArg___lam__0), 2, 1);
lean_closure_set(v___f_25_, 0, v_inst_24_);
return v___f_25_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Units_instHasDistribNeg(lean_object* v_00_u03b1_26_, lean_object* v_inst_27_, lean_object* v_inst_28_){
_start:
{
lean_object* v___f_29_; 
v___f_29_ = lean_alloc_closure((void*)(lp_mathlib_Units_instNeg___redArg___lam__0), 2, 1);
lean_closure_set(v___f_29_, 0, v_inst_28_);
return v___f_29_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Units_instHasDistribNeg___boxed(lean_object* v_00_u03b1_30_, lean_object* v_inst_31_, lean_object* v_inst_32_){
_start:
{
lean_object* v_res_33_; 
v_res_33_ = lp_mathlib_Units_instHasDistribNeg(v_00_u03b1_30_, v_inst_31_, v_inst_32_);
lean_dec_ref(v_inst_31_);
return v_res_33_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Ring_InjSurj(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Group_Units_Hom(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Ring_Hom_Defs(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Ring_Units(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Ring_InjSurj(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Group_Units_Hom(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Ring_Hom_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_Algebra_Ring_Units(uint8_t builtin) {
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
lean_object* initialize_mathlib_Mathlib_Algebra_Ring_InjSurj(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Group_Units_Hom(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Ring_Hom_Defs(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Algebra_Ring_Units(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Ring_InjSurj(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Group_Units_Hom(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Ring_Hom_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Ring_Units(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_Algebra_Ring_Units(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_Algebra_Ring_Units(builtin);
}
#ifdef __cplusplus
}
#endif
