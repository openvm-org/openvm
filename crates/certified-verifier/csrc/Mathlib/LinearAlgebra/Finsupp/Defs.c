// Lean compiler output
// Module: Mathlib.LinearAlgebra.Finsupp.Defs
// Imports: public import Init public meta import Init public import Mathlib.Algebra.Module.Equiv.Defs public import Mathlib.Algebra.Module.LinearMap.End public import Mathlib.Algebra.Module.Pi public import Mathlib.Data.Finsupp.SMul
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
lean_object* lp_mathlib_instMulZeroClassOfSemiring___redArg(lean_object*);
lean_object* lp_mathlib_AddMonoid_toAddZeroClass___redArg(lean_object*);
lean_object* lp_mathlib_AddZeroClass_toAddZero___redArg(lean_object*);
lean_object* lp_mathlib_Finsupp_applyAddHom___redArg___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Finsupp_lapply___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Finsupp_lapply(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Finsupp_lapply___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Module_subsingletonEquiv___redArg___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Module_subsingletonEquiv___redArg___lam__0___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Module_subsingletonEquiv___redArg___lam__1(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Module_subsingletonEquiv___redArg___lam__1___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Module_subsingletonEquiv___redArg___lam__2(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Module_subsingletonEquiv___redArg___lam__2___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Module_subsingletonEquiv___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Module_subsingletonEquiv___redArg___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Module_subsingletonEquiv(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Module_subsingletonEquiv___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Finsupp_lapply___redArg(lean_object* v_a_1_){
_start:
{
lean_object* v___f_2_; 
v___f_2_ = lean_alloc_closure((void*)(lp_mathlib_Finsupp_applyAddHom___redArg___lam__0), 2, 1);
lean_closure_set(v___f_2_, 0, v_a_1_);
return v___f_2_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Finsupp_lapply(lean_object* v_00_u03b1_3_, lean_object* v_M_4_, lean_object* v_R_5_, lean_object* v_inst_6_, lean_object* v_inst_7_, lean_object* v_inst_8_, lean_object* v_a_9_){
_start:
{
lean_object* v___f_10_; 
v___f_10_ = lean_alloc_closure((void*)(lp_mathlib_Finsupp_applyAddHom___redArg___lam__0), 2, 1);
lean_closure_set(v___f_10_, 0, v_a_9_);
return v___f_10_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Finsupp_lapply___boxed(lean_object* v_00_u03b1_11_, lean_object* v_M_12_, lean_object* v_R_13_, lean_object* v_inst_14_, lean_object* v_inst_15_, lean_object* v_inst_16_, lean_object* v_a_17_){
_start:
{
lean_object* v_res_18_; 
v_res_18_ = lp_mathlib_Finsupp_lapply(v_00_u03b1_11_, v_M_12_, v_R_13_, v_inst_14_, v_inst_15_, v_inst_16_, v_a_17_);
lean_dec(v_inst_16_);
lean_dec_ref(v_inst_15_);
lean_dec_ref(v_inst_14_);
return v_res_18_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Module_subsingletonEquiv___redArg___lam__0(lean_object* v_toZero_19_, lean_object* v_x_20_){
_start:
{
lean_inc(v_toZero_19_);
return v_toZero_19_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Module_subsingletonEquiv___redArg___lam__0___boxed(lean_object* v_toZero_21_, lean_object* v_x_22_){
_start:
{
lean_object* v_res_23_; 
v_res_23_ = lp_mathlib_Module_subsingletonEquiv___redArg___lam__0(v_toZero_21_, v_x_22_);
lean_dec(v_x_22_);
lean_dec(v_toZero_21_);
return v_res_23_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Module_subsingletonEquiv___redArg___lam__1(lean_object* v___f_24_, lean_object* v_x_25_){
_start:
{
lean_object* v___x_26_; lean_object* v___x_27_; 
v___x_26_ = lean_box(0);
v___x_27_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_27_, 0, v___x_26_);
lean_ctor_set(v___x_27_, 1, v___f_24_);
return v___x_27_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Module_subsingletonEquiv___redArg___lam__1___boxed(lean_object* v___f_28_, lean_object* v_x_29_){
_start:
{
lean_object* v_res_30_; 
v_res_30_ = lp_mathlib_Module_subsingletonEquiv___redArg___lam__1(v___f_28_, v_x_29_);
lean_dec(v_x_29_);
return v_res_30_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Module_subsingletonEquiv___redArg___lam__2(lean_object* v_toZero_31_, lean_object* v_x_32_){
_start:
{
lean_inc(v_toZero_31_);
return v_toZero_31_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Module_subsingletonEquiv___redArg___lam__2___boxed(lean_object* v_toZero_33_, lean_object* v_x_34_){
_start:
{
lean_object* v_res_35_; 
v_res_35_ = lp_mathlib_Module_subsingletonEquiv___redArg___lam__2(v_toZero_33_, v_x_34_);
lean_dec_ref(v_x_34_);
lean_dec(v_toZero_33_);
return v_res_35_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Module_subsingletonEquiv___redArg(lean_object* v_inst_36_, lean_object* v_inst_37_){
_start:
{
lean_object* v___x_38_; lean_object* v_toZero_39_; lean_object* v___x_40_; lean_object* v___x_41_; lean_object* v_toZero_42_; lean_object* v___x_44_; uint8_t v_isShared_45_; uint8_t v_isSharedCheck_52_; 
v___x_38_ = lp_mathlib_instMulZeroClassOfSemiring___redArg(v_inst_36_);
v_toZero_39_ = lean_ctor_get(v___x_38_, 1);
lean_inc(v_toZero_39_);
lean_dec_ref(v___x_38_);
v___x_40_ = lp_mathlib_AddMonoid_toAddZeroClass___redArg(v_inst_37_);
v___x_41_ = lp_mathlib_AddZeroClass_toAddZero___redArg(v___x_40_);
v_toZero_42_ = lean_ctor_get(v___x_41_, 0);
v_isSharedCheck_52_ = !lean_is_exclusive(v___x_41_);
if (v_isSharedCheck_52_ == 0)
{
lean_object* v_unused_53_; 
v_unused_53_ = lean_ctor_get(v___x_41_, 1);
lean_dec(v_unused_53_);
v___x_44_ = v___x_41_;
v_isShared_45_ = v_isSharedCheck_52_;
goto v_resetjp_43_;
}
else
{
lean_inc(v_toZero_42_);
lean_dec(v___x_41_);
v___x_44_ = lean_box(0);
v_isShared_45_ = v_isSharedCheck_52_;
goto v_resetjp_43_;
}
v_resetjp_43_:
{
lean_object* v___f_46_; lean_object* v___f_47_; lean_object* v___f_48_; lean_object* v___x_50_; 
v___f_46_ = lean_alloc_closure((void*)(lp_mathlib_Module_subsingletonEquiv___redArg___lam__0___boxed), 2, 1);
lean_closure_set(v___f_46_, 0, v_toZero_39_);
v___f_47_ = lean_alloc_closure((void*)(lp_mathlib_Module_subsingletonEquiv___redArg___lam__1___boxed), 2, 1);
lean_closure_set(v___f_47_, 0, v___f_46_);
v___f_48_ = lean_alloc_closure((void*)(lp_mathlib_Module_subsingletonEquiv___redArg___lam__2___boxed), 2, 1);
lean_closure_set(v___f_48_, 0, v_toZero_42_);
if (v_isShared_45_ == 0)
{
lean_ctor_set(v___x_44_, 1, v___f_48_);
lean_ctor_set(v___x_44_, 0, v___f_47_);
v___x_50_ = v___x_44_;
goto v_reusejp_49_;
}
else
{
lean_object* v_reuseFailAlloc_51_; 
v_reuseFailAlloc_51_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_51_, 0, v___f_47_);
lean_ctor_set(v_reuseFailAlloc_51_, 1, v___f_48_);
v___x_50_ = v_reuseFailAlloc_51_;
goto v_reusejp_49_;
}
v_reusejp_49_:
{
return v___x_50_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Module_subsingletonEquiv___redArg___boxed(lean_object* v_inst_54_, lean_object* v_inst_55_){
_start:
{
lean_object* v_res_56_; 
v_res_56_ = lp_mathlib_Module_subsingletonEquiv___redArg(v_inst_54_, v_inst_55_);
lean_dec_ref(v_inst_55_);
return v_res_56_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Module_subsingletonEquiv(lean_object* v_R_57_, lean_object* v_M_58_, lean_object* v_00_u03b9_59_, lean_object* v_inst_60_, lean_object* v_inst_61_, lean_object* v_inst_62_, lean_object* v_inst_63_){
_start:
{
lean_object* v___x_64_; 
v___x_64_ = lp_mathlib_Module_subsingletonEquiv___redArg(v_inst_60_, v_inst_62_);
return v___x_64_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Module_subsingletonEquiv___boxed(lean_object* v_R_65_, lean_object* v_M_66_, lean_object* v_00_u03b9_67_, lean_object* v_inst_68_, lean_object* v_inst_69_, lean_object* v_inst_70_, lean_object* v_inst_71_){
_start:
{
lean_object* v_res_72_; 
v_res_72_ = lp_mathlib_Module_subsingletonEquiv(v_R_65_, v_M_66_, v_00_u03b9_67_, v_inst_68_, v_inst_69_, v_inst_70_, v_inst_71_);
lean_dec(v_inst_71_);
lean_dec_ref(v_inst_70_);
return v_res_72_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Module_Equiv_Defs(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Module_LinearMap_End(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Module_Pi(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Data_Finsupp_SMul(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_LinearAlgebra_Finsupp_Defs(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Module_Equiv_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Module_LinearMap_End(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Module_Pi(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Data_Finsupp_SMul(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_LinearAlgebra_Finsupp_Defs(uint8_t builtin) {
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
lean_object* initialize_mathlib_Mathlib_Algebra_Module_Equiv_Defs(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Module_LinearMap_End(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Module_Pi(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Data_Finsupp_SMul(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_LinearAlgebra_Finsupp_Defs(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Module_Equiv_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Module_LinearMap_End(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Module_Pi(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_Finsupp_SMul(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_LinearAlgebra_Finsupp_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_LinearAlgebra_Finsupp_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_LinearAlgebra_Finsupp_Defs(builtin);
}
#ifdef __cplusplus
}
#endif
