// Lean compiler output
// Module: Mathlib.Data.Finset.NoncommProd
// Imports: public import Init public meta import Init public import Mathlib.Algebra.BigOperators.Group.Finset.Basic public import Mathlib.Algebra.Group.Commute.Hom public import Mathlib.Algebra.Group.Pi.Lemmas public import Mathlib.Data.Fintype.Basic
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
lean_object* lp_mathlib_Multiset_attach___redArg(lean_object*);
lean_object* l_List_foldrTR___redArg(lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_AddMonoid_toAddZeroClass___redArg(lean_object*);
lean_object* lp_mathlib_AddZeroClass_toAddZero___redArg(lean_object*);
lean_object* lp_mathlib_Multiset_map___redArg(lean_object*, lean_object*);
lean_object* lp_mathlib_Monoid_toMulOneClass___redArg(lean_object*);
lean_object* lp_mathlib_MulOneClass_toMulOne___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Multiset_noncommFoldr___redArg___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Multiset_noncommFoldr___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Multiset_noncommFoldr(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Multiset_noncommFold___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Multiset_noncommFold(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Multiset_noncommProd___redArg___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Multiset_noncommProd___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Multiset_noncommProd___redArg___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Multiset_noncommProd(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Multiset_noncommProd___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Multiset_noncommSum___redArg___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Multiset_noncommSum___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Multiset_noncommSum___redArg___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Multiset_noncommSum(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Multiset_noncommSum___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Finset_noncommProd___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Finset_noncommProd___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Finset_noncommProd(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Finset_noncommProd___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Finset_noncommSum___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Finset_noncommSum___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Finset_noncommSum(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Finset_noncommSum___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Multiset_noncommFoldr___redArg___lam__0(lean_object* v_f_1_, lean_object* v___y_2_, lean_object* v___y_3_){
_start:
{
lean_object* v___x_4_; 
v___x_4_ = lean_apply_2(v_f_1_, v___y_2_, v___y_3_);
return v___x_4_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Multiset_noncommFoldr___redArg(lean_object* v_f_5_, lean_object* v_s_6_, lean_object* v_b_7_){
_start:
{
lean_object* v___f_8_; lean_object* v___x_9_; lean_object* v___x_10_; 
v___f_8_ = lean_alloc_closure((void*)(lp_mathlib_Multiset_noncommFoldr___redArg___lam__0), 3, 1);
lean_closure_set(v___f_8_, 0, v_f_5_);
v___x_9_ = lp_mathlib_Multiset_attach___redArg(v_s_6_);
v___x_10_ = l_List_foldrTR___redArg(v___f_8_, v_b_7_, v___x_9_);
return v___x_10_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Multiset_noncommFoldr(lean_object* v_00_u03b1_11_, lean_object* v_00_u03b2_12_, lean_object* v_f_13_, lean_object* v_s_14_, lean_object* v_comm_15_, lean_object* v_b_16_){
_start:
{
lean_object* v___x_17_; 
v___x_17_ = lp_mathlib_Multiset_noncommFoldr___redArg(v_f_13_, v_s_14_, v_b_16_);
return v___x_17_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Multiset_noncommFold___redArg(lean_object* v_op_18_, lean_object* v_s_19_, lean_object* v_b_20_){
_start:
{
lean_object* v___x_21_; 
v___x_21_ = lp_mathlib_Multiset_noncommFoldr___redArg(v_op_18_, v_s_19_, v_b_20_);
return v___x_21_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Multiset_noncommFold(lean_object* v_00_u03b1_22_, lean_object* v_op_23_, lean_object* v_assoc_24_, lean_object* v_s_25_, lean_object* v_comm_26_, lean_object* v_b_27_){
_start:
{
lean_object* v___x_28_; 
v___x_28_ = lp_mathlib_Multiset_noncommFoldr___redArg(v_op_23_, v_s_25_, v_b_27_);
return v___x_28_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Multiset_noncommProd___redArg___lam__0(lean_object* v_toMul_29_, lean_object* v_x1_30_, lean_object* v_x2_31_){
_start:
{
lean_object* v___x_32_; 
v___x_32_ = lean_apply_2(v_toMul_29_, v_x1_30_, v_x2_31_);
return v___x_32_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Multiset_noncommProd___redArg(lean_object* v_inst_33_, lean_object* v_s_34_){
_start:
{
lean_object* v___x_35_; lean_object* v___x_36_; lean_object* v_toOne_37_; lean_object* v_toMul_38_; lean_object* v___f_39_; lean_object* v___x_40_; 
v___x_35_ = lp_mathlib_Monoid_toMulOneClass___redArg(v_inst_33_);
v___x_36_ = lp_mathlib_MulOneClass_toMulOne___redArg(v___x_35_);
v_toOne_37_ = lean_ctor_get(v___x_36_, 0);
lean_inc(v_toOne_37_);
v_toMul_38_ = lean_ctor_get(v___x_36_, 1);
lean_inc(v_toMul_38_);
lean_dec_ref(v___x_36_);
v___f_39_ = lean_alloc_closure((void*)(lp_mathlib_Multiset_noncommProd___redArg___lam__0), 3, 1);
lean_closure_set(v___f_39_, 0, v_toMul_38_);
v___x_40_ = lp_mathlib_Multiset_noncommFoldr___redArg(v___f_39_, v_s_34_, v_toOne_37_);
return v___x_40_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Multiset_noncommProd___redArg___boxed(lean_object* v_inst_41_, lean_object* v_s_42_){
_start:
{
lean_object* v_res_43_; 
v_res_43_ = lp_mathlib_Multiset_noncommProd___redArg(v_inst_41_, v_s_42_);
lean_dec_ref(v_inst_41_);
return v_res_43_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Multiset_noncommProd(lean_object* v_00_u03b1_44_, lean_object* v_inst_45_, lean_object* v_s_46_, lean_object* v_comm_47_){
_start:
{
lean_object* v___x_48_; 
v___x_48_ = lp_mathlib_Multiset_noncommProd___redArg(v_inst_45_, v_s_46_);
return v___x_48_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Multiset_noncommProd___boxed(lean_object* v_00_u03b1_49_, lean_object* v_inst_50_, lean_object* v_s_51_, lean_object* v_comm_52_){
_start:
{
lean_object* v_res_53_; 
v_res_53_ = lp_mathlib_Multiset_noncommProd(v_00_u03b1_49_, v_inst_50_, v_s_51_, v_comm_52_);
lean_dec_ref(v_inst_50_);
return v_res_53_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Multiset_noncommSum___redArg___lam__0(lean_object* v_toAdd_54_, lean_object* v_x1_55_, lean_object* v_x2_56_){
_start:
{
lean_object* v___x_57_; 
v___x_57_ = lean_apply_2(v_toAdd_54_, v_x1_55_, v_x2_56_);
return v___x_57_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Multiset_noncommSum___redArg(lean_object* v_inst_58_, lean_object* v_s_59_){
_start:
{
lean_object* v___x_60_; lean_object* v___x_61_; lean_object* v_toZero_62_; lean_object* v_toAdd_63_; lean_object* v___f_64_; lean_object* v___x_65_; 
v___x_60_ = lp_mathlib_AddMonoid_toAddZeroClass___redArg(v_inst_58_);
v___x_61_ = lp_mathlib_AddZeroClass_toAddZero___redArg(v___x_60_);
v_toZero_62_ = lean_ctor_get(v___x_61_, 0);
lean_inc(v_toZero_62_);
v_toAdd_63_ = lean_ctor_get(v___x_61_, 1);
lean_inc(v_toAdd_63_);
lean_dec_ref(v___x_61_);
v___f_64_ = lean_alloc_closure((void*)(lp_mathlib_Multiset_noncommSum___redArg___lam__0), 3, 1);
lean_closure_set(v___f_64_, 0, v_toAdd_63_);
v___x_65_ = lp_mathlib_Multiset_noncommFoldr___redArg(v___f_64_, v_s_59_, v_toZero_62_);
return v___x_65_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Multiset_noncommSum___redArg___boxed(lean_object* v_inst_66_, lean_object* v_s_67_){
_start:
{
lean_object* v_res_68_; 
v_res_68_ = lp_mathlib_Multiset_noncommSum___redArg(v_inst_66_, v_s_67_);
lean_dec_ref(v_inst_66_);
return v_res_68_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Multiset_noncommSum(lean_object* v_00_u03b1_69_, lean_object* v_inst_70_, lean_object* v_s_71_, lean_object* v_comm_72_){
_start:
{
lean_object* v___x_73_; 
v___x_73_ = lp_mathlib_Multiset_noncommSum___redArg(v_inst_70_, v_s_71_);
return v___x_73_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Multiset_noncommSum___boxed(lean_object* v_00_u03b1_74_, lean_object* v_inst_75_, lean_object* v_s_76_, lean_object* v_comm_77_){
_start:
{
lean_object* v_res_78_; 
v_res_78_ = lp_mathlib_Multiset_noncommSum(v_00_u03b1_74_, v_inst_75_, v_s_76_, v_comm_77_);
lean_dec_ref(v_inst_75_);
return v_res_78_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Finset_noncommProd___redArg(lean_object* v_inst_79_, lean_object* v_s_80_, lean_object* v_f_81_){
_start:
{
lean_object* v___x_82_; lean_object* v___x_83_; 
v___x_82_ = lp_mathlib_Multiset_map___redArg(v_f_81_, v_s_80_);
v___x_83_ = lp_mathlib_Multiset_noncommProd___redArg(v_inst_79_, v___x_82_);
return v___x_83_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Finset_noncommProd___redArg___boxed(lean_object* v_inst_84_, lean_object* v_s_85_, lean_object* v_f_86_){
_start:
{
lean_object* v_res_87_; 
v_res_87_ = lp_mathlib_Finset_noncommProd___redArg(v_inst_84_, v_s_85_, v_f_86_);
lean_dec_ref(v_inst_84_);
return v_res_87_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Finset_noncommProd(lean_object* v_00_u03b1_88_, lean_object* v_00_u03b2_89_, lean_object* v_inst_90_, lean_object* v_s_91_, lean_object* v_f_92_, lean_object* v_comm_93_){
_start:
{
lean_object* v___x_94_; 
v___x_94_ = lp_mathlib_Finset_noncommProd___redArg(v_inst_90_, v_s_91_, v_f_92_);
return v___x_94_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Finset_noncommProd___boxed(lean_object* v_00_u03b1_95_, lean_object* v_00_u03b2_96_, lean_object* v_inst_97_, lean_object* v_s_98_, lean_object* v_f_99_, lean_object* v_comm_100_){
_start:
{
lean_object* v_res_101_; 
v_res_101_ = lp_mathlib_Finset_noncommProd(v_00_u03b1_95_, v_00_u03b2_96_, v_inst_97_, v_s_98_, v_f_99_, v_comm_100_);
lean_dec_ref(v_inst_97_);
return v_res_101_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Finset_noncommSum___redArg(lean_object* v_inst_102_, lean_object* v_s_103_, lean_object* v_f_104_){
_start:
{
lean_object* v___x_105_; lean_object* v___x_106_; 
v___x_105_ = lp_mathlib_Multiset_map___redArg(v_f_104_, v_s_103_);
v___x_106_ = lp_mathlib_Multiset_noncommSum___redArg(v_inst_102_, v___x_105_);
return v___x_106_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Finset_noncommSum___redArg___boxed(lean_object* v_inst_107_, lean_object* v_s_108_, lean_object* v_f_109_){
_start:
{
lean_object* v_res_110_; 
v_res_110_ = lp_mathlib_Finset_noncommSum___redArg(v_inst_107_, v_s_108_, v_f_109_);
lean_dec_ref(v_inst_107_);
return v_res_110_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Finset_noncommSum(lean_object* v_00_u03b1_111_, lean_object* v_00_u03b2_112_, lean_object* v_inst_113_, lean_object* v_s_114_, lean_object* v_f_115_, lean_object* v_comm_116_){
_start:
{
lean_object* v___x_117_; 
v___x_117_ = lp_mathlib_Finset_noncommSum___redArg(v_inst_113_, v_s_114_, v_f_115_);
return v___x_117_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Finset_noncommSum___boxed(lean_object* v_00_u03b1_118_, lean_object* v_00_u03b2_119_, lean_object* v_inst_120_, lean_object* v_s_121_, lean_object* v_f_122_, lean_object* v_comm_123_){
_start:
{
lean_object* v_res_124_; 
v_res_124_ = lp_mathlib_Finset_noncommSum(v_00_u03b1_118_, v_00_u03b2_119_, v_inst_120_, v_s_121_, v_f_122_, v_comm_123_);
lean_dec_ref(v_inst_120_);
return v_res_124_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_BigOperators_Group_Finset_Basic(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Group_Commute_Hom(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Group_Pi_Lemmas(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Data_Fintype_Basic(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_Data_Finset_NoncommProd(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_BigOperators_Group_Finset_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Group_Commute_Hom(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Group_Pi_Lemmas(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Data_Fintype_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_Data_Finset_NoncommProd(uint8_t builtin) {
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
lean_object* initialize_mathlib_Mathlib_Algebra_BigOperators_Group_Finset_Basic(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Group_Commute_Hom(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Group_Pi_Lemmas(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Data_Fintype_Basic(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Data_Finset_NoncommProd(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_BigOperators_Group_Finset_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Group_Commute_Hom(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Group_Pi_Lemmas(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_Fintype_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Data_Finset_NoncommProd(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_Data_Finset_NoncommProd(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_Data_Finset_NoncommProd(builtin);
}
#ifdef __cplusplus
}
#endif
