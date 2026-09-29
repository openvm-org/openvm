// Lean compiler output
// Module: Mathlib.Algebra.GradedMulAction
// Imports: public import Init public meta import Init public import Mathlib.Algebra.GradedMonoid
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
LEAN_EXPORT lean_object* lp_mathlib_GradedMonoid_GMul_toGSMul___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_GradedMonoid_GMul_toGSMul___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_GradedMonoid_GMul_toGSMul(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_GradedMonoid_GMul_toGSMul___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_GradedMonoid_GSMul_toSMul___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_GradedMonoid_GSMul_toSMul___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_GradedMonoid_GSMul_toSMul(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_GradedMonoid_GMonoid_toGMulAction___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_GradedMonoid_GMonoid_toGMulAction(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_GradedMonoid_GMonoid_toGMulAction___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_GradedMonoid_GMulAction_toMulAction___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_GradedMonoid_GMulAction_toMulAction(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_GradedMonoid_GMulAction_toMulAction___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_SetLike_toGSMul___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_SetLike_toGSMul___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_SetLike_toGSMul___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_SetLike_toGSMul(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_SetLike_toGSMul___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_GradedMonoid_GMul_toGSMul___redArg___lam__0(lean_object* v_inst_1_, lean_object* v_i_2_, lean_object* v_j_3_, lean_object* v___y_4_, lean_object* v___y_5_){
_start:
{
lean_object* v___x_6_; 
v___x_6_ = lean_apply_4(v_inst_1_, v_i_2_, v_j_3_, v___y_4_, v___y_5_);
return v___x_6_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_GradedMonoid_GMul_toGSMul___redArg(lean_object* v_inst_7_){
_start:
{
lean_object* v___f_8_; 
v___f_8_ = lean_alloc_closure((void*)(lp_mathlib_GradedMonoid_GMul_toGSMul___redArg___lam__0), 5, 1);
lean_closure_set(v___f_8_, 0, v_inst_7_);
return v___f_8_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_GradedMonoid_GMul_toGSMul(lean_object* v_00_u03b9A_9_, lean_object* v_A_10_, lean_object* v_inst_11_, lean_object* v_inst_12_){
_start:
{
lean_object* v___f_13_; 
v___f_13_ = lean_alloc_closure((void*)(lp_mathlib_GradedMonoid_GMul_toGSMul___redArg___lam__0), 5, 1);
lean_closure_set(v___f_13_, 0, v_inst_12_);
return v___f_13_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_GradedMonoid_GMul_toGSMul___boxed(lean_object* v_00_u03b9A_14_, lean_object* v_A_15_, lean_object* v_inst_16_, lean_object* v_inst_17_){
_start:
{
lean_object* v_res_18_; 
v_res_18_ = lp_mathlib_GradedMonoid_GMul_toGSMul(v_00_u03b9A_14_, v_A_15_, v_inst_16_, v_inst_17_);
lean_dec(v_inst_16_);
return v_res_18_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_GradedMonoid_GSMul_toSMul___redArg___lam__0(lean_object* v_inst_19_, lean_object* v_inst_20_, lean_object* v_x_21_, lean_object* v_y_22_){
_start:
{
lean_object* v_fst_23_; lean_object* v_snd_24_; lean_object* v_fst_25_; lean_object* v_snd_26_; lean_object* v___x_28_; uint8_t v_isShared_29_; uint8_t v_isSharedCheck_35_; 
v_fst_23_ = lean_ctor_get(v_x_21_, 0);
lean_inc(v_fst_23_);
v_snd_24_ = lean_ctor_get(v_x_21_, 1);
lean_inc(v_snd_24_);
lean_dec_ref(v_x_21_);
v_fst_25_ = lean_ctor_get(v_y_22_, 0);
v_snd_26_ = lean_ctor_get(v_y_22_, 1);
v_isSharedCheck_35_ = !lean_is_exclusive(v_y_22_);
if (v_isSharedCheck_35_ == 0)
{
v___x_28_ = v_y_22_;
v_isShared_29_ = v_isSharedCheck_35_;
goto v_resetjp_27_;
}
else
{
lean_inc(v_snd_26_);
lean_inc(v_fst_25_);
lean_dec(v_y_22_);
v___x_28_ = lean_box(0);
v_isShared_29_ = v_isSharedCheck_35_;
goto v_resetjp_27_;
}
v_resetjp_27_:
{
lean_object* v___x_30_; lean_object* v___x_31_; lean_object* v___x_33_; 
lean_inc(v_fst_25_);
lean_inc(v_fst_23_);
v___x_30_ = lean_apply_2(v_inst_19_, v_fst_23_, v_fst_25_);
v___x_31_ = lean_apply_4(v_inst_20_, v_fst_23_, v_fst_25_, v_snd_24_, v_snd_26_);
if (v_isShared_29_ == 0)
{
lean_ctor_set(v___x_28_, 1, v___x_31_);
lean_ctor_set(v___x_28_, 0, v___x_30_);
v___x_33_ = v___x_28_;
goto v_reusejp_32_;
}
else
{
lean_object* v_reuseFailAlloc_34_; 
v_reuseFailAlloc_34_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_34_, 0, v___x_30_);
lean_ctor_set(v_reuseFailAlloc_34_, 1, v___x_31_);
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
LEAN_EXPORT lean_object* lp_mathlib_GradedMonoid_GSMul_toSMul___redArg(lean_object* v_inst_36_, lean_object* v_inst_37_){
_start:
{
lean_object* v___f_38_; 
v___f_38_ = lean_alloc_closure((void*)(lp_mathlib_GradedMonoid_GSMul_toSMul___redArg___lam__0), 4, 2);
lean_closure_set(v___f_38_, 0, v_inst_36_);
lean_closure_set(v___f_38_, 1, v_inst_37_);
return v___f_38_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_GradedMonoid_GSMul_toSMul(lean_object* v_00_u03b9A_39_, lean_object* v_00_u03b9M_40_, lean_object* v_A_41_, lean_object* v_M_42_, lean_object* v_inst_43_, lean_object* v_inst_44_){
_start:
{
lean_object* v___f_45_; 
v___f_45_ = lean_alloc_closure((void*)(lp_mathlib_GradedMonoid_GSMul_toSMul___redArg___lam__0), 4, 2);
lean_closure_set(v___f_45_, 0, v_inst_43_);
lean_closure_set(v___f_45_, 1, v_inst_44_);
return v___f_45_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_GradedMonoid_GMonoid_toGMulAction___redArg(lean_object* v_inst_46_){
_start:
{
lean_object* v_toGMul_47_; lean_object* v___f_48_; 
v_toGMul_47_ = lean_ctor_get(v_inst_46_, 0);
lean_inc(v_toGMul_47_);
lean_dec_ref(v_inst_46_);
v___f_48_ = lean_alloc_closure((void*)(lp_mathlib_GradedMonoid_GMul_toGSMul___redArg___lam__0), 5, 1);
lean_closure_set(v___f_48_, 0, v_toGMul_47_);
return v___f_48_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_GradedMonoid_GMonoid_toGMulAction(lean_object* v_00_u03b9A_49_, lean_object* v_A_50_, lean_object* v_inst_51_, lean_object* v_inst_52_){
_start:
{
lean_object* v___x_53_; 
v___x_53_ = lp_mathlib_GradedMonoid_GMonoid_toGMulAction___redArg(v_inst_52_);
return v___x_53_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_GradedMonoid_GMonoid_toGMulAction___boxed(lean_object* v_00_u03b9A_54_, lean_object* v_A_55_, lean_object* v_inst_56_, lean_object* v_inst_57_){
_start:
{
lean_object* v_res_58_; 
v_res_58_ = lp_mathlib_GradedMonoid_GMonoid_toGMulAction(v_00_u03b9A_54_, v_A_55_, v_inst_56_, v_inst_57_);
lean_dec_ref(v_inst_56_);
return v_res_58_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_GradedMonoid_GMulAction_toMulAction___redArg(lean_object* v_inst_59_, lean_object* v_inst_60_){
_start:
{
lean_object* v___f_61_; 
v___f_61_ = lean_alloc_closure((void*)(lp_mathlib_GradedMonoid_GSMul_toSMul___redArg___lam__0), 4, 2);
lean_closure_set(v___f_61_, 0, v_inst_59_);
lean_closure_set(v___f_61_, 1, v_inst_60_);
return v___f_61_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_GradedMonoid_GMulAction_toMulAction(lean_object* v_00_u03b9A_62_, lean_object* v_00_u03b9M_63_, lean_object* v_A_64_, lean_object* v_M_65_, lean_object* v_inst_66_, lean_object* v_inst_67_, lean_object* v_inst_68_, lean_object* v_inst_69_){
_start:
{
lean_object* v___f_70_; 
v___f_70_ = lean_alloc_closure((void*)(lp_mathlib_GradedMonoid_GSMul_toSMul___redArg___lam__0), 4, 2);
lean_closure_set(v___f_70_, 0, v_inst_68_);
lean_closure_set(v___f_70_, 1, v_inst_69_);
return v___f_70_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_GradedMonoid_GMulAction_toMulAction___boxed(lean_object* v_00_u03b9A_71_, lean_object* v_00_u03b9M_72_, lean_object* v_A_73_, lean_object* v_M_74_, lean_object* v_inst_75_, lean_object* v_inst_76_, lean_object* v_inst_77_, lean_object* v_inst_78_){
_start:
{
lean_object* v_res_79_; 
v_res_79_ = lp_mathlib_GradedMonoid_GMulAction_toMulAction(v_00_u03b9A_71_, v_00_u03b9M_72_, v_A_73_, v_M_74_, v_inst_75_, v_inst_76_, v_inst_77_, v_inst_78_);
lean_dec_ref(v_inst_76_);
lean_dec_ref(v_inst_75_);
return v_res_79_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_SetLike_toGSMul___redArg___lam__0(lean_object* v_inst_80_, lean_object* v_i_81_, lean_object* v_j_82_, lean_object* v_a_83_, lean_object* v_b_84_){
_start:
{
lean_object* v___x_85_; 
v___x_85_ = lean_apply_2(v_inst_80_, v_a_83_, v_b_84_);
return v___x_85_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_SetLike_toGSMul___redArg___lam__0___boxed(lean_object* v_inst_86_, lean_object* v_i_87_, lean_object* v_j_88_, lean_object* v_a_89_, lean_object* v_b_90_){
_start:
{
lean_object* v_res_91_; 
v_res_91_ = lp_mathlib_SetLike_toGSMul___redArg___lam__0(v_inst_86_, v_i_87_, v_j_88_, v_a_89_, v_b_90_);
lean_dec(v_j_88_);
lean_dec(v_i_87_);
return v_res_91_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_SetLike_toGSMul___redArg(lean_object* v_inst_92_){
_start:
{
lean_object* v___f_93_; 
v___f_93_ = lean_alloc_closure((void*)(lp_mathlib_SetLike_toGSMul___redArg___lam__0___boxed), 5, 1);
lean_closure_set(v___f_93_, 0, v_inst_92_);
return v___f_93_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_SetLike_toGSMul(lean_object* v_00_u03b9A_94_, lean_object* v_00_u03b9B_95_, lean_object* v_S_96_, lean_object* v_R_97_, lean_object* v_N_98_, lean_object* v_M_99_, lean_object* v_inst_100_, lean_object* v_inst_101_, lean_object* v_inst_102_, lean_object* v_inst_103_, lean_object* v_A_104_, lean_object* v_B_105_, lean_object* v_inst_106_){
_start:
{
lean_object* v___f_107_; 
v___f_107_ = lean_alloc_closure((void*)(lp_mathlib_SetLike_toGSMul___redArg___lam__0___boxed), 5, 1);
lean_closure_set(v___f_107_, 0, v_inst_102_);
return v___f_107_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_SetLike_toGSMul___boxed(lean_object* v_00_u03b9A_108_, lean_object* v_00_u03b9B_109_, lean_object* v_S_110_, lean_object* v_R_111_, lean_object* v_N_112_, lean_object* v_M_113_, lean_object* v_inst_114_, lean_object* v_inst_115_, lean_object* v_inst_116_, lean_object* v_inst_117_, lean_object* v_A_118_, lean_object* v_B_119_, lean_object* v_inst_120_){
_start:
{
lean_object* v_res_121_; 
v_res_121_ = lp_mathlib_SetLike_toGSMul(v_00_u03b9A_108_, v_00_u03b9B_109_, v_S_110_, v_R_111_, v_N_112_, v_M_113_, v_inst_114_, v_inst_115_, v_inst_116_, v_inst_117_, v_A_118_, v_B_119_, v_inst_120_);
lean_dec(v_B_119_);
lean_dec(v_A_118_);
lean_dec(v_inst_117_);
return v_res_121_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_GradedMonoid(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_Algebra_GradedMulAction(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_GradedMonoid(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_Algebra_GradedMulAction(uint8_t builtin) {
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
lean_object* initialize_mathlib_Mathlib_Algebra_GradedMonoid(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Algebra_GradedMulAction(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_GradedMonoid(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_GradedMulAction(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_Algebra_GradedMulAction(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_Algebra_GradedMulAction(builtin);
}
#ifdef __cplusplus
}
#endif
