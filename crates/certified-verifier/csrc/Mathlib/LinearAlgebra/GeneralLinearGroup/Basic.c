// Lean compiler output
// Module: Mathlib.LinearAlgebra.GeneralLinearGroup.Basic
// Imports: public import Init public meta import Init public import Mathlib.Algebra.Module.Equiv.Basic
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
lean_object* lp_mathlib_LinearEquiv_arrowCongrAddEquiv___redArg(lean_object*, lean_object*);
lean_object* lp_mathlib_Units_mapEquiv___redArg(lean_object*);
lean_object* lp_mathlib_LinearEquiv_symm___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LinearMap_GeneralLinearGroup_toLinearEquiv___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LinearMap_GeneralLinearGroup_toLinearEquiv(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LinearMap_GeneralLinearGroup_toLinearEquiv___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LinearMap_GeneralLinearGroup_ofLinearEquiv___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LinearMap_GeneralLinearGroup_ofLinearEquiv(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LinearMap_GeneralLinearGroup_ofLinearEquiv___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LinearMap_GeneralLinearGroup_generalLinearEquiv___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LinearMap_GeneralLinearGroup_generalLinearEquiv(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LinearMap_GeneralLinearGroup_congrLinearEquiv___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LinearMap_GeneralLinearGroup_congrLinearEquiv(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LinearMap_GeneralLinearGroup_congrLinearEquiv___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LinearMap_GeneralLinearGroup_toLinearEquiv___redArg(lean_object* v_f_1_){
_start:
{
lean_object* v_val_2_; lean_object* v_inv_3_; lean_object* v___x_5_; uint8_t v_isShared_6_; uint8_t v_isSharedCheck_10_; 
v_val_2_ = lean_ctor_get(v_f_1_, 0);
v_inv_3_ = lean_ctor_get(v_f_1_, 1);
v_isSharedCheck_10_ = !lean_is_exclusive(v_f_1_);
if (v_isSharedCheck_10_ == 0)
{
v___x_5_ = v_f_1_;
v_isShared_6_ = v_isSharedCheck_10_;
goto v_resetjp_4_;
}
else
{
lean_inc(v_inv_3_);
lean_inc(v_val_2_);
lean_dec(v_f_1_);
v___x_5_ = lean_box(0);
v_isShared_6_ = v_isSharedCheck_10_;
goto v_resetjp_4_;
}
v_resetjp_4_:
{
lean_object* v___x_8_; 
if (v_isShared_6_ == 0)
{
v___x_8_ = v___x_5_;
goto v_reusejp_7_;
}
else
{
lean_object* v_reuseFailAlloc_9_; 
v_reuseFailAlloc_9_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_9_, 0, v_val_2_);
lean_ctor_set(v_reuseFailAlloc_9_, 1, v_inv_3_);
v___x_8_ = v_reuseFailAlloc_9_;
goto v_reusejp_7_;
}
v_reusejp_7_:
{
return v___x_8_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_LinearMap_GeneralLinearGroup_toLinearEquiv(lean_object* v_R_11_, lean_object* v_M_12_, lean_object* v_inst_13_, lean_object* v_inst_14_, lean_object* v_inst_15_, lean_object* v_f_16_){
_start:
{
lean_object* v___x_17_; 
v___x_17_ = lp_mathlib_LinearMap_GeneralLinearGroup_toLinearEquiv___redArg(v_f_16_);
return v___x_17_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LinearMap_GeneralLinearGroup_toLinearEquiv___boxed(lean_object* v_R_18_, lean_object* v_M_19_, lean_object* v_inst_20_, lean_object* v_inst_21_, lean_object* v_inst_22_, lean_object* v_f_23_){
_start:
{
lean_object* v_res_24_; 
v_res_24_ = lp_mathlib_LinearMap_GeneralLinearGroup_toLinearEquiv(v_R_18_, v_M_19_, v_inst_20_, v_inst_21_, v_inst_22_, v_f_23_);
lean_dec(v_inst_22_);
lean_dec_ref(v_inst_21_);
lean_dec_ref(v_inst_20_);
return v_res_24_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LinearMap_GeneralLinearGroup_ofLinearEquiv___redArg(lean_object* v_f_25_){
_start:
{
lean_object* v_toLinearMap_26_; lean_object* v___x_27_; lean_object* v_toLinearMap_28_; lean_object* v___x_30_; uint8_t v_isShared_31_; uint8_t v_isSharedCheck_35_; 
v_toLinearMap_26_ = lean_ctor_get(v_f_25_, 0);
lean_inc(v_toLinearMap_26_);
v___x_27_ = lp_mathlib_LinearEquiv_symm___redArg(v_f_25_);
v_toLinearMap_28_ = lean_ctor_get(v___x_27_, 0);
v_isSharedCheck_35_ = !lean_is_exclusive(v___x_27_);
if (v_isSharedCheck_35_ == 0)
{
lean_object* v_unused_36_; 
v_unused_36_ = lean_ctor_get(v___x_27_, 1);
lean_dec(v_unused_36_);
v___x_30_ = v___x_27_;
v_isShared_31_ = v_isSharedCheck_35_;
goto v_resetjp_29_;
}
else
{
lean_inc(v_toLinearMap_28_);
lean_dec(v___x_27_);
v___x_30_ = lean_box(0);
v_isShared_31_ = v_isSharedCheck_35_;
goto v_resetjp_29_;
}
v_resetjp_29_:
{
lean_object* v___x_33_; 
if (v_isShared_31_ == 0)
{
lean_ctor_set(v___x_30_, 1, v_toLinearMap_28_);
lean_ctor_set(v___x_30_, 0, v_toLinearMap_26_);
v___x_33_ = v___x_30_;
goto v_reusejp_32_;
}
else
{
lean_object* v_reuseFailAlloc_34_; 
v_reuseFailAlloc_34_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_34_, 0, v_toLinearMap_26_);
lean_ctor_set(v_reuseFailAlloc_34_, 1, v_toLinearMap_28_);
v___x_33_ = v_reuseFailAlloc_34_;
goto v_reusejp_32_;
}
v_reusejp_32_:
{
return v___x_33_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_LinearMap_GeneralLinearGroup_ofLinearEquiv(lean_object* v_R_37_, lean_object* v_M_38_, lean_object* v_inst_39_, lean_object* v_inst_40_, lean_object* v_inst_41_, lean_object* v_f_42_){
_start:
{
lean_object* v___x_43_; 
v___x_43_ = lp_mathlib_LinearMap_GeneralLinearGroup_ofLinearEquiv___redArg(v_f_42_);
return v___x_43_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LinearMap_GeneralLinearGroup_ofLinearEquiv___boxed(lean_object* v_R_44_, lean_object* v_M_45_, lean_object* v_inst_46_, lean_object* v_inst_47_, lean_object* v_inst_48_, lean_object* v_f_49_){
_start:
{
lean_object* v_res_50_; 
v_res_50_ = lp_mathlib_LinearMap_GeneralLinearGroup_ofLinearEquiv(v_R_44_, v_M_45_, v_inst_46_, v_inst_47_, v_inst_48_, v_f_49_);
lean_dec(v_inst_48_);
lean_dec_ref(v_inst_47_);
lean_dec_ref(v_inst_46_);
return v_res_50_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LinearMap_GeneralLinearGroup_generalLinearEquiv___redArg(lean_object* v_inst_51_, lean_object* v_inst_52_, lean_object* v_inst_53_){
_start:
{
lean_object* v___x_54_; lean_object* v___x_55_; lean_object* v___x_56_; 
lean_inc(v_inst_53_);
lean_inc_ref(v_inst_52_);
lean_inc_ref(v_inst_51_);
v___x_54_ = lean_alloc_closure((void*)(lp_mathlib_LinearMap_GeneralLinearGroup_toLinearEquiv___boxed), 6, 5);
lean_closure_set(v___x_54_, 0, lean_box(0));
lean_closure_set(v___x_54_, 1, lean_box(0));
lean_closure_set(v___x_54_, 2, v_inst_51_);
lean_closure_set(v___x_54_, 3, v_inst_52_);
lean_closure_set(v___x_54_, 4, v_inst_53_);
v___x_55_ = lean_alloc_closure((void*)(lp_mathlib_LinearMap_GeneralLinearGroup_ofLinearEquiv___boxed), 6, 5);
lean_closure_set(v___x_55_, 0, lean_box(0));
lean_closure_set(v___x_55_, 1, lean_box(0));
lean_closure_set(v___x_55_, 2, v_inst_51_);
lean_closure_set(v___x_55_, 3, v_inst_52_);
lean_closure_set(v___x_55_, 4, v_inst_53_);
v___x_56_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_56_, 0, v___x_54_);
lean_ctor_set(v___x_56_, 1, v___x_55_);
return v___x_56_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LinearMap_GeneralLinearGroup_generalLinearEquiv(lean_object* v_R_57_, lean_object* v_M_58_, lean_object* v_inst_59_, lean_object* v_inst_60_, lean_object* v_inst_61_){
_start:
{
lean_object* v___x_62_; 
v___x_62_ = lp_mathlib_LinearMap_GeneralLinearGroup_generalLinearEquiv___redArg(v_inst_59_, v_inst_60_, v_inst_61_);
return v___x_62_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LinearMap_GeneralLinearGroup_congrLinearEquiv___redArg(lean_object* v_e_u2081_u2082_63_){
_start:
{
lean_object* v___x_64_; lean_object* v___x_65_; 
lean_inc_ref(v_e_u2081_u2082_63_);
v___x_64_ = lp_mathlib_LinearEquiv_arrowCongrAddEquiv___redArg(v_e_u2081_u2082_63_, v_e_u2081_u2082_63_);
v___x_65_ = lp_mathlib_Units_mapEquiv___redArg(v___x_64_);
return v___x_65_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LinearMap_GeneralLinearGroup_congrLinearEquiv(lean_object* v_R_u2081_66_, lean_object* v_R_u2082_67_, lean_object* v_M_u2081_68_, lean_object* v_M_u2082_69_, lean_object* v_inst_70_, lean_object* v_inst_71_, lean_object* v_inst_72_, lean_object* v_inst_73_, lean_object* v_inst_74_, lean_object* v_inst_75_, lean_object* v_00_u03c3_u2081_u2082_76_, lean_object* v_00_u03c3_u2082_u2081_77_, lean_object* v_inst_78_, lean_object* v_inst_79_, lean_object* v_e_u2081_u2082_80_){
_start:
{
lean_object* v___x_81_; 
v___x_81_ = lp_mathlib_LinearMap_GeneralLinearGroup_congrLinearEquiv___redArg(v_e_u2081_u2082_80_);
return v___x_81_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LinearMap_GeneralLinearGroup_congrLinearEquiv___boxed(lean_object* v_R_u2081_82_, lean_object* v_R_u2082_83_, lean_object* v_M_u2081_84_, lean_object* v_M_u2082_85_, lean_object* v_inst_86_, lean_object* v_inst_87_, lean_object* v_inst_88_, lean_object* v_inst_89_, lean_object* v_inst_90_, lean_object* v_inst_91_, lean_object* v_00_u03c3_u2081_u2082_92_, lean_object* v_00_u03c3_u2082_u2081_93_, lean_object* v_inst_94_, lean_object* v_inst_95_, lean_object* v_e_u2081_u2082_96_){
_start:
{
lean_object* v_res_97_; 
v_res_97_ = lp_mathlib_LinearMap_GeneralLinearGroup_congrLinearEquiv(v_R_u2081_82_, v_R_u2082_83_, v_M_u2081_84_, v_M_u2082_85_, v_inst_86_, v_inst_87_, v_inst_88_, v_inst_89_, v_inst_90_, v_inst_91_, v_00_u03c3_u2081_u2082_92_, v_00_u03c3_u2082_u2081_93_, v_inst_94_, v_inst_95_, v_e_u2081_u2082_96_);
lean_dec(v_00_u03c3_u2082_u2081_93_);
lean_dec(v_00_u03c3_u2081_u2082_92_);
lean_dec(v_inst_91_);
lean_dec(v_inst_90_);
lean_dec_ref(v_inst_89_);
lean_dec_ref(v_inst_88_);
lean_dec_ref(v_inst_87_);
lean_dec_ref(v_inst_86_);
return v_res_97_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Module_Equiv_Basic(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_LinearAlgebra_GeneralLinearGroup_Basic(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Module_Equiv_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_LinearAlgebra_GeneralLinearGroup_Basic(uint8_t builtin) {
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
lean_object* initialize_mathlib_Mathlib_Algebra_Module_Equiv_Basic(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_LinearAlgebra_GeneralLinearGroup_Basic(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Module_Equiv_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_LinearAlgebra_GeneralLinearGroup_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_LinearAlgebra_GeneralLinearGroup_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_LinearAlgebra_GeneralLinearGroup_Basic(builtin);
}
#ifdef __cplusplus
}
#endif
