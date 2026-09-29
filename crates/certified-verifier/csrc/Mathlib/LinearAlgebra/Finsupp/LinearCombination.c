// Lean compiler output
// Module: Mathlib.LinearAlgebra.Finsupp.LinearCombination
// Imports: public import Init public meta import Init public import Mathlib.Algebra.Module.Submodule.Equiv public import Mathlib.Data.Finsupp.Option public import Mathlib.LinearAlgebra.Finsupp.Supported
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
lean_object* lp_mathlib_Finset_sum___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Fintype_linearCombination___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Fintype_linearCombination___redArg___lam__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Fintype_linearCombination___redArg___lam__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Fintype_linearCombination___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Fintype_linearCombination(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Fintype_linearCombination___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Fintype_bilinearCombination___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Fintype_bilinearCombination___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Fintype_bilinearCombination___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Fintype_bilinearCombination(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Fintype_bilinearCombination___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Fintype_linearCombination___redArg___lam__0(lean_object* v_f_1_, lean_object* v_v_2_, lean_object* v_inst_3_, lean_object* v_i_4_){
_start:
{
lean_object* v___x_5_; lean_object* v___x_6_; lean_object* v___x_7_; 
lean_inc(v_i_4_);
v___x_5_ = lean_apply_1(v_f_1_, v_i_4_);
v___x_6_ = lean_apply_1(v_v_2_, v_i_4_);
v___x_7_ = lean_apply_2(v_inst_3_, v___x_5_, v___x_6_);
return v___x_7_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Fintype_linearCombination___redArg___lam__1(lean_object* v_v_8_, lean_object* v_inst_9_, lean_object* v_inst_10_, lean_object* v_inst_11_, lean_object* v_f_12_){
_start:
{
lean_object* v___f_13_; lean_object* v___x_14_; 
v___f_13_ = lean_alloc_closure((void*)(lp_mathlib_Fintype_linearCombination___redArg___lam__0), 4, 3);
lean_closure_set(v___f_13_, 0, v_f_12_);
lean_closure_set(v___f_13_, 1, v_v_8_);
lean_closure_set(v___f_13_, 2, v_inst_9_);
v___x_14_ = lp_mathlib_Finset_sum___redArg(v_inst_10_, v_inst_11_, v___f_13_);
return v___x_14_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Fintype_linearCombination___redArg___lam__1___boxed(lean_object* v_v_15_, lean_object* v_inst_16_, lean_object* v_inst_17_, lean_object* v_inst_18_, lean_object* v_f_19_){
_start:
{
lean_object* v_res_20_; 
v_res_20_ = lp_mathlib_Fintype_linearCombination___redArg___lam__1(v_v_15_, v_inst_16_, v_inst_17_, v_inst_18_, v_f_19_);
lean_dec_ref(v_inst_17_);
return v_res_20_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Fintype_linearCombination___redArg(lean_object* v_inst_21_, lean_object* v_inst_22_, lean_object* v_inst_23_, lean_object* v_v_24_){
_start:
{
lean_object* v___f_25_; 
v___f_25_ = lean_alloc_closure((void*)(lp_mathlib_Fintype_linearCombination___redArg___lam__1___boxed), 5, 4);
lean_closure_set(v___f_25_, 0, v_v_24_);
lean_closure_set(v___f_25_, 1, v_inst_23_);
lean_closure_set(v___f_25_, 2, v_inst_22_);
lean_closure_set(v___f_25_, 3, v_inst_21_);
return v___f_25_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Fintype_linearCombination(lean_object* v_00_u03b1_26_, lean_object* v_M_27_, lean_object* v_R_28_, lean_object* v_inst_29_, lean_object* v_inst_30_, lean_object* v_inst_31_, lean_object* v_inst_32_, lean_object* v_v_33_){
_start:
{
lean_object* v___f_34_; 
v___f_34_ = lean_alloc_closure((void*)(lp_mathlib_Fintype_linearCombination___redArg___lam__1___boxed), 5, 4);
lean_closure_set(v___f_34_, 0, v_v_33_);
lean_closure_set(v___f_34_, 1, v_inst_32_);
lean_closure_set(v___f_34_, 2, v_inst_31_);
lean_closure_set(v___f_34_, 3, v_inst_29_);
return v___f_34_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Fintype_linearCombination___boxed(lean_object* v_00_u03b1_35_, lean_object* v_M_36_, lean_object* v_R_37_, lean_object* v_inst_38_, lean_object* v_inst_39_, lean_object* v_inst_40_, lean_object* v_inst_41_, lean_object* v_v_42_){
_start:
{
lean_object* v_res_43_; 
v_res_43_ = lp_mathlib_Fintype_linearCombination(v_00_u03b1_35_, v_M_36_, v_R_37_, v_inst_38_, v_inst_39_, v_inst_40_, v_inst_41_, v_v_42_);
lean_dec_ref(v_inst_39_);
return v_res_43_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Fintype_bilinearCombination___redArg___lam__0(lean_object* v_inst_44_, lean_object* v_inst_45_, lean_object* v_inst_46_, lean_object* v_v_47_, lean_object* v___y_48_){
_start:
{
lean_object* v___x_49_; 
v___x_49_ = lp_mathlib_Fintype_linearCombination___redArg___lam__1(v_v_47_, v_inst_44_, v_inst_45_, v_inst_46_, v___y_48_);
return v___x_49_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Fintype_bilinearCombination___redArg___lam__0___boxed(lean_object* v_inst_50_, lean_object* v_inst_51_, lean_object* v_inst_52_, lean_object* v_v_53_, lean_object* v___y_54_){
_start:
{
lean_object* v_res_55_; 
v_res_55_ = lp_mathlib_Fintype_bilinearCombination___redArg___lam__0(v_inst_50_, v_inst_51_, v_inst_52_, v_v_53_, v___y_54_);
lean_dec_ref(v_inst_51_);
return v_res_55_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Fintype_bilinearCombination___redArg(lean_object* v_inst_56_, lean_object* v_inst_57_, lean_object* v_inst_58_){
_start:
{
lean_object* v___f_59_; 
v___f_59_ = lean_alloc_closure((void*)(lp_mathlib_Fintype_bilinearCombination___redArg___lam__0___boxed), 5, 3);
lean_closure_set(v___f_59_, 0, v_inst_58_);
lean_closure_set(v___f_59_, 1, v_inst_57_);
lean_closure_set(v___f_59_, 2, v_inst_56_);
return v___f_59_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Fintype_bilinearCombination(lean_object* v_00_u03b1_60_, lean_object* v_M_61_, lean_object* v_R_62_, lean_object* v_inst_63_, lean_object* v_inst_64_, lean_object* v_inst_65_, lean_object* v_inst_66_, lean_object* v_S_67_, lean_object* v_inst_68_, lean_object* v_inst_69_, lean_object* v_inst_70_){
_start:
{
lean_object* v___f_71_; 
v___f_71_ = lean_alloc_closure((void*)(lp_mathlib_Fintype_bilinearCombination___redArg___lam__0___boxed), 5, 3);
lean_closure_set(v___f_71_, 0, v_inst_66_);
lean_closure_set(v___f_71_, 1, v_inst_65_);
lean_closure_set(v___f_71_, 2, v_inst_63_);
return v___f_71_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Fintype_bilinearCombination___boxed(lean_object* v_00_u03b1_72_, lean_object* v_M_73_, lean_object* v_R_74_, lean_object* v_inst_75_, lean_object* v_inst_76_, lean_object* v_inst_77_, lean_object* v_inst_78_, lean_object* v_S_79_, lean_object* v_inst_80_, lean_object* v_inst_81_, lean_object* v_inst_82_){
_start:
{
lean_object* v_res_83_; 
v_res_83_ = lp_mathlib_Fintype_bilinearCombination(v_00_u03b1_72_, v_M_73_, v_R_74_, v_inst_75_, v_inst_76_, v_inst_77_, v_inst_78_, v_S_79_, v_inst_80_, v_inst_81_, v_inst_82_);
lean_dec(v_inst_81_);
lean_dec_ref(v_inst_80_);
lean_dec_ref(v_inst_76_);
return v_res_83_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Module_Submodule_Equiv(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Data_Finsupp_Option(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_LinearAlgebra_Finsupp_Supported(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_LinearAlgebra_Finsupp_LinearCombination(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Module_Submodule_Equiv(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Data_Finsupp_Option(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_LinearAlgebra_Finsupp_Supported(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_LinearAlgebra_Finsupp_LinearCombination(uint8_t builtin) {
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
lean_object* initialize_mathlib_Mathlib_Algebra_Module_Submodule_Equiv(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Data_Finsupp_Option(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_LinearAlgebra_Finsupp_Supported(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_LinearAlgebra_Finsupp_LinearCombination(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Module_Submodule_Equiv(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_Finsupp_Option(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_LinearAlgebra_Finsupp_Supported(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_LinearAlgebra_Finsupp_LinearCombination(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_LinearAlgebra_Finsupp_LinearCombination(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_LinearAlgebra_Finsupp_LinearCombination(builtin);
}
#ifdef __cplusplus
}
#endif
