// Lean compiler output
// Module: Mathlib.Algebra.GroupWithZero.Action.Units
// Imports: public import Init public meta import Init public import Mathlib.Algebra.Group.Action.Units public import Mathlib.Algebra.GroupWithZero.Action.Defs public import Mathlib.Algebra.GroupWithZero.Units.Basic
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
lean_object* lp_mathlib_Units_instSMul___redArg___lam__0(lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_GroupWithZero_toDivInvMonoid___redArg(lean_object*);
lean_object* lp_mathlib_DivInvOneMonoid_toInvOneClass___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_smulRight___redArg___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_smulRight___redArg___lam__1(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_smulRight___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_smulRight(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Units_instSMulZeroClass___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Units_instSMulZeroClass(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Units_instSMulZeroClass___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Units_instDistribSMulUnits___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Units_instDistribSMulUnits(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Units_instDistribSMulUnits___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Units_instDistribMulAction___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Units_instDistribMulAction(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Units_instDistribMulAction___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Units_instMulDistribMulAction___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Units_instMulDistribMulAction(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Units_instMulDistribMulAction___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_smulRight___redArg___lam__0(lean_object* v_inst_1_, lean_object* v_a_2_, lean_object* v_b_3_){
_start:
{
lean_object* v___x_4_; 
v___x_4_ = lean_apply_2(v_inst_1_, v_a_2_, v_b_3_);
return v___x_4_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_smulRight___redArg___lam__1(lean_object* v_toInv_5_, lean_object* v_a_6_, lean_object* v_inst_7_, lean_object* v_b_8_){
_start:
{
lean_object* v___x_9_; lean_object* v___x_10_; 
v___x_9_ = lean_apply_1(v_toInv_5_, v_a_6_);
v___x_10_ = lean_apply_2(v_inst_7_, v___x_9_, v_b_8_);
return v___x_10_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_smulRight___redArg(lean_object* v_inst_11_, lean_object* v_inst_12_, lean_object* v_a_13_){
_start:
{
lean_object* v___x_14_; lean_object* v___x_15_; lean_object* v_toInv_16_; lean_object* v___x_18_; uint8_t v_isShared_19_; uint8_t v_isSharedCheck_25_; 
v___x_14_ = lp_mathlib_GroupWithZero_toDivInvMonoid___redArg(v_inst_11_);
v___x_15_ = lp_mathlib_DivInvOneMonoid_toInvOneClass___redArg(v___x_14_);
lean_dec_ref(v___x_14_);
v_toInv_16_ = lean_ctor_get(v___x_15_, 1);
v_isSharedCheck_25_ = !lean_is_exclusive(v___x_15_);
if (v_isSharedCheck_25_ == 0)
{
lean_object* v_unused_26_; 
v_unused_26_ = lean_ctor_get(v___x_15_, 0);
lean_dec(v_unused_26_);
v___x_18_ = v___x_15_;
v_isShared_19_ = v_isSharedCheck_25_;
goto v_resetjp_17_;
}
else
{
lean_inc(v_toInv_16_);
lean_dec(v___x_15_);
v___x_18_ = lean_box(0);
v_isShared_19_ = v_isSharedCheck_25_;
goto v_resetjp_17_;
}
v_resetjp_17_:
{
lean_object* v___f_20_; lean_object* v___f_21_; lean_object* v___x_23_; 
lean_inc(v_a_13_);
lean_inc(v_inst_12_);
v___f_20_ = lean_alloc_closure((void*)(lp_mathlib_Equiv_smulRight___redArg___lam__0), 3, 2);
lean_closure_set(v___f_20_, 0, v_inst_12_);
lean_closure_set(v___f_20_, 1, v_a_13_);
v___f_21_ = lean_alloc_closure((void*)(lp_mathlib_Equiv_smulRight___redArg___lam__1), 4, 3);
lean_closure_set(v___f_21_, 0, v_toInv_16_);
lean_closure_set(v___f_21_, 1, v_a_13_);
lean_closure_set(v___f_21_, 2, v_inst_12_);
if (v_isShared_19_ == 0)
{
lean_ctor_set(v___x_18_, 1, v___f_21_);
lean_ctor_set(v___x_18_, 0, v___f_20_);
v___x_23_ = v___x_18_;
goto v_reusejp_22_;
}
else
{
lean_object* v_reuseFailAlloc_24_; 
v_reuseFailAlloc_24_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_24_, 0, v___f_20_);
lean_ctor_set(v_reuseFailAlloc_24_, 1, v___f_21_);
v___x_23_ = v_reuseFailAlloc_24_;
goto v_reusejp_22_;
}
v_reusejp_22_:
{
return v___x_23_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_smulRight(lean_object* v_00_u03b1_27_, lean_object* v_00_u03b2_28_, lean_object* v_inst_29_, lean_object* v_inst_30_, lean_object* v_a_31_, lean_object* v_ha_32_){
_start:
{
lean_object* v___x_33_; 
v___x_33_ = lp_mathlib_Equiv_smulRight___redArg(v_inst_29_, v_inst_30_, v_a_31_);
return v___x_33_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Units_instSMulZeroClass___redArg(lean_object* v_inst_34_){
_start:
{
lean_object* v___f_35_; 
v___f_35_ = lean_alloc_closure((void*)(lp_mathlib_Units_instSMul___redArg___lam__0), 3, 1);
lean_closure_set(v___f_35_, 0, v_inst_34_);
return v___f_35_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Units_instSMulZeroClass(lean_object* v_M_36_, lean_object* v_00_u03b1_37_, lean_object* v_inst_38_, lean_object* v_inst_39_, lean_object* v_inst_40_){
_start:
{
lean_object* v___f_41_; 
v___f_41_ = lean_alloc_closure((void*)(lp_mathlib_Units_instSMul___redArg___lam__0), 3, 1);
lean_closure_set(v___f_41_, 0, v_inst_40_);
return v___f_41_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Units_instSMulZeroClass___boxed(lean_object* v_M_42_, lean_object* v_00_u03b1_43_, lean_object* v_inst_44_, lean_object* v_inst_45_, lean_object* v_inst_46_){
_start:
{
lean_object* v_res_47_; 
v_res_47_ = lp_mathlib_Units_instSMulZeroClass(v_M_42_, v_00_u03b1_43_, v_inst_44_, v_inst_45_, v_inst_46_);
lean_dec(v_inst_45_);
lean_dec_ref(v_inst_44_);
return v_res_47_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Units_instDistribSMulUnits___redArg(lean_object* v_inst_48_){
_start:
{
lean_object* v___f_49_; 
v___f_49_ = lean_alloc_closure((void*)(lp_mathlib_Units_instSMul___redArg___lam__0), 3, 1);
lean_closure_set(v___f_49_, 0, v_inst_48_);
return v___f_49_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Units_instDistribSMulUnits(lean_object* v_M_50_, lean_object* v_00_u03b1_51_, lean_object* v_inst_52_, lean_object* v_inst_53_, lean_object* v_inst_54_){
_start:
{
lean_object* v___f_55_; 
v___f_55_ = lean_alloc_closure((void*)(lp_mathlib_Units_instSMul___redArg___lam__0), 3, 1);
lean_closure_set(v___f_55_, 0, v_inst_54_);
return v___f_55_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Units_instDistribSMulUnits___boxed(lean_object* v_M_56_, lean_object* v_00_u03b1_57_, lean_object* v_inst_58_, lean_object* v_inst_59_, lean_object* v_inst_60_){
_start:
{
lean_object* v_res_61_; 
v_res_61_ = lp_mathlib_Units_instDistribSMulUnits(v_M_56_, v_00_u03b1_57_, v_inst_58_, v_inst_59_, v_inst_60_);
lean_dec_ref(v_inst_59_);
lean_dec_ref(v_inst_58_);
return v_res_61_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Units_instDistribMulAction___redArg(lean_object* v_inst_62_){
_start:
{
lean_object* v___f_63_; 
v___f_63_ = lean_alloc_closure((void*)(lp_mathlib_Units_instSMul___redArg___lam__0), 3, 1);
lean_closure_set(v___f_63_, 0, v_inst_62_);
return v___f_63_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Units_instDistribMulAction(lean_object* v_M_64_, lean_object* v_00_u03b1_65_, lean_object* v_inst_66_, lean_object* v_inst_67_, lean_object* v_inst_68_){
_start:
{
lean_object* v___f_69_; 
v___f_69_ = lean_alloc_closure((void*)(lp_mathlib_Units_instSMul___redArg___lam__0), 3, 1);
lean_closure_set(v___f_69_, 0, v_inst_68_);
return v___f_69_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Units_instDistribMulAction___boxed(lean_object* v_M_70_, lean_object* v_00_u03b1_71_, lean_object* v_inst_72_, lean_object* v_inst_73_, lean_object* v_inst_74_){
_start:
{
lean_object* v_res_75_; 
v_res_75_ = lp_mathlib_Units_instDistribMulAction(v_M_70_, v_00_u03b1_71_, v_inst_72_, v_inst_73_, v_inst_74_);
lean_dec_ref(v_inst_73_);
lean_dec_ref(v_inst_72_);
return v_res_75_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Units_instMulDistribMulAction___redArg(lean_object* v_inst_76_){
_start:
{
lean_object* v___f_77_; 
v___f_77_ = lean_alloc_closure((void*)(lp_mathlib_Units_instSMul___redArg___lam__0), 3, 1);
lean_closure_set(v___f_77_, 0, v_inst_76_);
return v___f_77_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Units_instMulDistribMulAction(lean_object* v_M_78_, lean_object* v_00_u03b1_79_, lean_object* v_inst_80_, lean_object* v_inst_81_, lean_object* v_inst_82_){
_start:
{
lean_object* v___f_83_; 
v___f_83_ = lean_alloc_closure((void*)(lp_mathlib_Units_instSMul___redArg___lam__0), 3, 1);
lean_closure_set(v___f_83_, 0, v_inst_82_);
return v___f_83_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Units_instMulDistribMulAction___boxed(lean_object* v_M_84_, lean_object* v_00_u03b1_85_, lean_object* v_inst_86_, lean_object* v_inst_87_, lean_object* v_inst_88_){
_start:
{
lean_object* v_res_89_; 
v_res_89_ = lp_mathlib_Units_instMulDistribMulAction(v_M_84_, v_00_u03b1_85_, v_inst_86_, v_inst_87_, v_inst_88_);
lean_dec_ref(v_inst_87_);
lean_dec_ref(v_inst_86_);
return v_res_89_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Group_Action_Units(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_GroupWithZero_Action_Defs(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_GroupWithZero_Units_Basic(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_Algebra_GroupWithZero_Action_Units(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Group_Action_Units(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_GroupWithZero_Action_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_GroupWithZero_Units_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_Algebra_GroupWithZero_Action_Units(uint8_t builtin) {
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
lean_object* initialize_mathlib_Mathlib_Algebra_Group_Action_Units(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_GroupWithZero_Action_Defs(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_GroupWithZero_Units_Basic(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Algebra_GroupWithZero_Action_Units(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Group_Action_Units(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_GroupWithZero_Action_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_GroupWithZero_Units_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_GroupWithZero_Action_Units(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_Algebra_GroupWithZero_Action_Units(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_Algebra_GroupWithZero_Action_Units(builtin);
}
#ifdef __cplusplus
}
#endif
