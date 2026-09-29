// Lean compiler output
// Module: Mathlib.Algebra.Group.Subgroup.MulOppositeLemmas
// Imports: public import Init public meta import Init public import Mathlib.Algebra.Group.Subgroup.Basic public import Mathlib.Algebra.Group.Subgroup.MulOpposite public import Mathlib.Algebra.Group.Submonoid.MulOpposite public import Mathlib.Logic.Encodable.Basic
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
lean_object* lp_mathlib_Monoid_toMulOneClass___redArg(lean_object*);
lean_object* lp_mathlib_MulOneClass_toMulOne___redArg(lean_object*);
lean_object* lp_mathlib_Mul_toSMulMulOpposite___redArg___lam__0(lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Submonoid_instSMulSubtypeMem___redArg___lam__0(lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_AddMonoid_toAddZeroClass___redArg(lean_object*);
lean_object* lp_mathlib_AddZeroClass_toAddZero___redArg(lean_object*);
lean_object* lp_mathlib_Subgroup_equivOp(lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Equiv_symm___redArg(lean_object*);
lean_object* lp_mathlib_Encodable_ofEquiv___redArg(lean_object*, lean_object*);
lean_object* lp_mathlib_AddSubgroup_equivOp(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subgroup_instSMul___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subgroup_instSMul___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subgroup_instSMul(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subgroup_instSMul___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddSubgroup_instVAdd___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddSubgroup_instVAdd___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddSubgroup_instVAdd(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddSubgroup_instVAdd___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subgroup_instEncodableSubtypeMulOppositeMemOp___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subgroup_instEncodableSubtypeMulOppositeMemOp___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subgroup_instEncodableSubtypeMulOppositeMemOp(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subgroup_instEncodableSubtypeMulOppositeMemOp___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddSubgroup_instEncodableSubtypeAddOppositeMemOp___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddSubgroup_instEncodableSubtypeAddOppositeMemOp___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddSubgroup_instEncodableSubtypeAddOppositeMemOp(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddSubgroup_instEncodableSubtypeAddOppositeMemOp___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subgroup_instSMul___redArg(lean_object* v_inst_1_){
_start:
{
lean_object* v_toMonoid_2_; lean_object* v___x_3_; lean_object* v___x_4_; lean_object* v_toMul_5_; lean_object* v___f_6_; lean_object* v___f_7_; 
v_toMonoid_2_ = lean_ctor_get(v_inst_1_, 0);
v___x_3_ = lp_mathlib_Monoid_toMulOneClass___redArg(v_toMonoid_2_);
v___x_4_ = lp_mathlib_MulOneClass_toMulOne___redArg(v___x_3_);
v_toMul_5_ = lean_ctor_get(v___x_4_, 1);
lean_inc(v_toMul_5_);
lean_dec_ref(v___x_4_);
v___f_6_ = lean_alloc_closure((void*)(lp_mathlib_Mul_toSMulMulOpposite___redArg___lam__0), 3, 1);
lean_closure_set(v___f_6_, 0, v_toMul_5_);
v___f_7_ = lean_alloc_closure((void*)(lp_mathlib_Submonoid_instSMulSubtypeMem___redArg___lam__0), 3, 1);
lean_closure_set(v___f_7_, 0, v___f_6_);
return v___f_7_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subgroup_instSMul___redArg___boxed(lean_object* v_inst_8_){
_start:
{
lean_object* v_res_9_; 
v_res_9_ = lp_mathlib_Subgroup_instSMul___redArg(v_inst_8_);
lean_dec_ref(v_inst_8_);
return v_res_9_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subgroup_instSMul(lean_object* v_G_10_, lean_object* v_inst_11_, lean_object* v_H_12_){
_start:
{
lean_object* v___x_13_; 
v___x_13_ = lp_mathlib_Subgroup_instSMul___redArg(v_inst_11_);
return v___x_13_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subgroup_instSMul___boxed(lean_object* v_G_14_, lean_object* v_inst_15_, lean_object* v_H_16_){
_start:
{
lean_object* v_res_17_; 
v_res_17_ = lp_mathlib_Subgroup_instSMul(v_G_14_, v_inst_15_, v_H_16_);
lean_dec_ref(v_inst_15_);
return v_res_17_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddSubgroup_instVAdd___redArg(lean_object* v_inst_18_){
_start:
{
lean_object* v_toAddMonoid_19_; lean_object* v___x_20_; lean_object* v___x_21_; lean_object* v_toAdd_22_; lean_object* v___f_23_; lean_object* v___f_24_; 
v_toAddMonoid_19_ = lean_ctor_get(v_inst_18_, 0);
v___x_20_ = lp_mathlib_AddMonoid_toAddZeroClass___redArg(v_toAddMonoid_19_);
v___x_21_ = lp_mathlib_AddZeroClass_toAddZero___redArg(v___x_20_);
v_toAdd_22_ = lean_ctor_get(v___x_21_, 1);
lean_inc(v_toAdd_22_);
lean_dec_ref(v___x_21_);
v___f_23_ = lean_alloc_closure((void*)(lp_mathlib_Mul_toSMulMulOpposite___redArg___lam__0), 3, 1);
lean_closure_set(v___f_23_, 0, v_toAdd_22_);
v___f_24_ = lean_alloc_closure((void*)(lp_mathlib_Submonoid_instSMulSubtypeMem___redArg___lam__0), 3, 1);
lean_closure_set(v___f_24_, 0, v___f_23_);
return v___f_24_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddSubgroup_instVAdd___redArg___boxed(lean_object* v_inst_25_){
_start:
{
lean_object* v_res_26_; 
v_res_26_ = lp_mathlib_AddSubgroup_instVAdd___redArg(v_inst_25_);
lean_dec_ref(v_inst_25_);
return v_res_26_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddSubgroup_instVAdd(lean_object* v_G_27_, lean_object* v_inst_28_, lean_object* v_H_29_){
_start:
{
lean_object* v___x_30_; 
v___x_30_ = lp_mathlib_AddSubgroup_instVAdd___redArg(v_inst_28_);
return v___x_30_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddSubgroup_instVAdd___boxed(lean_object* v_G_31_, lean_object* v_inst_32_, lean_object* v_H_33_){
_start:
{
lean_object* v_res_34_; 
v_res_34_ = lp_mathlib_AddSubgroup_instVAdd(v_G_31_, v_inst_32_, v_H_33_);
lean_dec_ref(v_inst_32_);
return v_res_34_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subgroup_instEncodableSubtypeMulOppositeMemOp___redArg(lean_object* v_inst_35_, lean_object* v_H_36_, lean_object* v_inst_37_){
_start:
{
lean_object* v___x_38_; lean_object* v___x_39_; lean_object* v___x_40_; 
v___x_38_ = lp_mathlib_Subgroup_equivOp(lean_box(0), v_inst_35_, v_H_36_);
v___x_39_ = lp_mathlib_Equiv_symm___redArg(v___x_38_);
v___x_40_ = lp_mathlib_Encodable_ofEquiv___redArg(v_inst_37_, v___x_39_);
return v___x_40_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subgroup_instEncodableSubtypeMulOppositeMemOp___redArg___boxed(lean_object* v_inst_41_, lean_object* v_H_42_, lean_object* v_inst_43_){
_start:
{
lean_object* v_res_44_; 
v_res_44_ = lp_mathlib_Subgroup_instEncodableSubtypeMulOppositeMemOp___redArg(v_inst_41_, v_H_42_, v_inst_43_);
lean_dec_ref(v_inst_41_);
return v_res_44_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subgroup_instEncodableSubtypeMulOppositeMemOp(lean_object* v_G_45_, lean_object* v_inst_46_, lean_object* v_H_47_, lean_object* v_inst_48_){
_start:
{
lean_object* v___x_49_; 
v___x_49_ = lp_mathlib_Subgroup_instEncodableSubtypeMulOppositeMemOp___redArg(v_inst_46_, v_H_47_, v_inst_48_);
return v___x_49_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subgroup_instEncodableSubtypeMulOppositeMemOp___boxed(lean_object* v_G_50_, lean_object* v_inst_51_, lean_object* v_H_52_, lean_object* v_inst_53_){
_start:
{
lean_object* v_res_54_; 
v_res_54_ = lp_mathlib_Subgroup_instEncodableSubtypeMulOppositeMemOp(v_G_50_, v_inst_51_, v_H_52_, v_inst_53_);
lean_dec_ref(v_inst_51_);
return v_res_54_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddSubgroup_instEncodableSubtypeAddOppositeMemOp___redArg(lean_object* v_inst_55_, lean_object* v_H_56_, lean_object* v_inst_57_){
_start:
{
lean_object* v___x_58_; lean_object* v___x_59_; lean_object* v___x_60_; 
v___x_58_ = lp_mathlib_AddSubgroup_equivOp(lean_box(0), v_inst_55_, v_H_56_);
v___x_59_ = lp_mathlib_Equiv_symm___redArg(v___x_58_);
v___x_60_ = lp_mathlib_Encodable_ofEquiv___redArg(v_inst_57_, v___x_59_);
return v___x_60_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddSubgroup_instEncodableSubtypeAddOppositeMemOp___redArg___boxed(lean_object* v_inst_61_, lean_object* v_H_62_, lean_object* v_inst_63_){
_start:
{
lean_object* v_res_64_; 
v_res_64_ = lp_mathlib_AddSubgroup_instEncodableSubtypeAddOppositeMemOp___redArg(v_inst_61_, v_H_62_, v_inst_63_);
lean_dec_ref(v_inst_61_);
return v_res_64_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddSubgroup_instEncodableSubtypeAddOppositeMemOp(lean_object* v_G_65_, lean_object* v_inst_66_, lean_object* v_H_67_, lean_object* v_inst_68_){
_start:
{
lean_object* v___x_69_; 
v___x_69_ = lp_mathlib_AddSubgroup_instEncodableSubtypeAddOppositeMemOp___redArg(v_inst_66_, v_H_67_, v_inst_68_);
return v___x_69_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddSubgroup_instEncodableSubtypeAddOppositeMemOp___boxed(lean_object* v_G_70_, lean_object* v_inst_71_, lean_object* v_H_72_, lean_object* v_inst_73_){
_start:
{
lean_object* v_res_74_; 
v_res_74_ = lp_mathlib_AddSubgroup_instEncodableSubtypeAddOppositeMemOp(v_G_70_, v_inst_71_, v_H_72_, v_inst_73_);
lean_dec_ref(v_inst_71_);
return v_res_74_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Group_Subgroup_Basic(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Group_Subgroup_MulOpposite(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Group_Submonoid_MulOpposite(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Logic_Encodable_Basic(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Group_Subgroup_MulOppositeLemmas(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Group_Subgroup_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Group_Subgroup_MulOpposite(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Group_Submonoid_MulOpposite(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Logic_Encodable_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_Algebra_Group_Subgroup_MulOppositeLemmas(uint8_t builtin) {
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
lean_object* initialize_mathlib_Mathlib_Algebra_Group_Subgroup_Basic(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Group_Subgroup_MulOpposite(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Group_Submonoid_MulOpposite(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Logic_Encodable_Basic(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Algebra_Group_Subgroup_MulOppositeLemmas(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Group_Subgroup_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Group_Subgroup_MulOpposite(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Group_Submonoid_MulOpposite(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Logic_Encodable_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Group_Subgroup_MulOppositeLemmas(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_Algebra_Group_Subgroup_MulOppositeLemmas(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_Algebra_Group_Subgroup_MulOppositeLemmas(builtin);
}
#ifdef __cplusplus
}
#endif
