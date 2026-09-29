// Lean compiler output
// Module: Mathlib.Algebra.GroupWithZero.Units.Equiv
// Imports: public import Init public meta import Init public import Mathlib.Algebra.Group.Units.Equiv public import Mathlib.Algebra.GroupWithZero.Units.Basic
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
lean_object* lp_mathlib_Units_mk0___redArg(lean_object*, lean_object*);
lean_object* lp_mathlib_Units_mulLeft___redArg(lean_object*, lean_object*);
lean_object* lp_mathlib_Units_mulRight___redArg(lean_object*, lean_object*);
lean_object* lp_mathlib_CommGroupWithZero_toGroupWithZero___redArg(lean_object*);
lean_object* lp_mathlib_GroupWithZero_toDivInvMonoid___redArg(lean_object*);
lean_object* lp_mathlib_MonoidWithZero_toMulZeroOneClass___redArg(lean_object*);
lean_object* lp_mathlib_MulZeroOneClass_toMulZeroClass___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_unitsEquivNeZero___redArg___lam__0(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_unitsEquivNeZero___redArg___lam__0___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_unitsEquivNeZero___redArg___lam__1(lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_unitsEquivNeZero___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_unitsEquivNeZero___redArg___lam__0___boxed, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_unitsEquivNeZero___redArg___closed__0 = (const lean_object*)&lp_mathlib_unitsEquivNeZero___redArg___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_unitsEquivNeZero___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_unitsEquivNeZero(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_mulLeft_u2080___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_mulLeft_u2080(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_mulRight_u2080___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_mulRight_u2080(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_divRight_u2080___redArg___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_divRight_u2080___redArg___lam__1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_divRight_u2080___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_divRight_u2080(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_divLeft_u2080___redArg___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_divLeft_u2080___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_divLeft_u2080(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_unitsEquivNeZero___redArg___lam__0(lean_object* v_a_1_){
_start:
{
lean_object* v_val_2_; 
v_val_2_ = lean_ctor_get(v_a_1_, 0);
lean_inc(v_val_2_);
return v_val_2_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_unitsEquivNeZero___redArg___lam__0___boxed(lean_object* v_a_3_){
_start:
{
lean_object* v_res_4_; 
v_res_4_ = lp_mathlib_unitsEquivNeZero___redArg___lam__0(v_a_3_);
lean_dec_ref(v_a_3_);
return v_res_4_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_unitsEquivNeZero___redArg___lam__1(lean_object* v_inst_5_, lean_object* v_a_6_){
_start:
{
lean_object* v___x_7_; 
v___x_7_ = lp_mathlib_Units_mk0___redArg(v_inst_5_, v_a_6_);
return v___x_7_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_unitsEquivNeZero___redArg(lean_object* v_inst_9_){
_start:
{
lean_object* v___f_10_; lean_object* v___f_11_; lean_object* v___x_12_; 
v___f_10_ = ((lean_object*)(lp_mathlib_unitsEquivNeZero___redArg___closed__0));
v___f_11_ = lean_alloc_closure((void*)(lp_mathlib_unitsEquivNeZero___redArg___lam__1), 2, 1);
lean_closure_set(v___f_11_, 0, v_inst_9_);
v___x_12_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_12_, 0, v___f_10_);
lean_ctor_set(v___x_12_, 1, v___f_11_);
return v___x_12_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_unitsEquivNeZero(lean_object* v_G_u2080_13_, lean_object* v_inst_14_){
_start:
{
lean_object* v___x_15_; 
v___x_15_ = lp_mathlib_unitsEquivNeZero___redArg(v_inst_14_);
return v___x_15_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_mulLeft_u2080___redArg(lean_object* v_inst_16_, lean_object* v_a_17_){
_start:
{
lean_object* v_toMonoidWithZero_18_; lean_object* v_toMonoid_19_; lean_object* v___x_20_; lean_object* v___x_21_; 
v_toMonoidWithZero_18_ = lean_ctor_get(v_inst_16_, 0);
v_toMonoid_19_ = lean_ctor_get(v_toMonoidWithZero_18_, 0);
lean_inc_ref(v_toMonoid_19_);
v___x_20_ = lp_mathlib_Units_mk0___redArg(v_inst_16_, v_a_17_);
v___x_21_ = lp_mathlib_Units_mulLeft___redArg(v_toMonoid_19_, v___x_20_);
lean_dec_ref(v_toMonoid_19_);
return v___x_21_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_mulLeft_u2080(lean_object* v_G_u2080_22_, lean_object* v_inst_23_, lean_object* v_a_24_, lean_object* v_ha_25_){
_start:
{
lean_object* v___x_26_; 
v___x_26_ = lp_mathlib_Equiv_mulLeft_u2080___redArg(v_inst_23_, v_a_24_);
return v___x_26_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_mulRight_u2080___redArg(lean_object* v_inst_27_, lean_object* v_a_28_){
_start:
{
lean_object* v_toMonoidWithZero_29_; lean_object* v_toMonoid_30_; lean_object* v___x_31_; lean_object* v___x_32_; 
v_toMonoidWithZero_29_ = lean_ctor_get(v_inst_27_, 0);
v_toMonoid_30_ = lean_ctor_get(v_toMonoidWithZero_29_, 0);
lean_inc_ref(v_toMonoid_30_);
v___x_31_ = lp_mathlib_Units_mk0___redArg(v_inst_27_, v_a_28_);
v___x_32_ = lp_mathlib_Units_mulRight___redArg(v_toMonoid_30_, v___x_31_);
lean_dec_ref(v_toMonoid_30_);
return v___x_32_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_mulRight_u2080(lean_object* v_G_u2080_33_, lean_object* v_inst_34_, lean_object* v_a_35_, lean_object* v_ha_36_){
_start:
{
lean_object* v___x_37_; 
v___x_37_ = lp_mathlib_Equiv_mulRight_u2080___redArg(v_inst_34_, v_a_35_);
return v___x_37_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_divRight_u2080___redArg___lam__0(lean_object* v_toDiv_38_, lean_object* v_a_39_, lean_object* v_x_40_){
_start:
{
lean_object* v___x_41_; 
v___x_41_ = lean_apply_2(v_toDiv_38_, v_x_40_, v_a_39_);
return v___x_41_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_divRight_u2080___redArg___lam__1(lean_object* v_toMul_42_, lean_object* v_a_43_, lean_object* v_x_44_){
_start:
{
lean_object* v___x_45_; 
v___x_45_ = lean_apply_2(v_toMul_42_, v_x_44_, v_a_43_);
return v___x_45_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_divRight_u2080___redArg(lean_object* v_inst_46_, lean_object* v_a_47_){
_start:
{
lean_object* v___x_48_; lean_object* v_toDiv_49_; lean_object* v_toMonoidWithZero_50_; lean_object* v___x_51_; lean_object* v___x_52_; lean_object* v_toMul_53_; lean_object* v___x_55_; uint8_t v_isShared_56_; uint8_t v_isSharedCheck_62_; 
lean_inc_ref(v_inst_46_);
v___x_48_ = lp_mathlib_GroupWithZero_toDivInvMonoid___redArg(v_inst_46_);
v_toDiv_49_ = lean_ctor_get(v___x_48_, 2);
lean_inc(v_toDiv_49_);
lean_dec_ref(v___x_48_);
v_toMonoidWithZero_50_ = lean_ctor_get(v_inst_46_, 0);
lean_inc_ref(v_toMonoidWithZero_50_);
lean_dec_ref(v_inst_46_);
v___x_51_ = lp_mathlib_MonoidWithZero_toMulZeroOneClass___redArg(v_toMonoidWithZero_50_);
v___x_52_ = lp_mathlib_MulZeroOneClass_toMulZeroClass___redArg(v___x_51_);
v_toMul_53_ = lean_ctor_get(v___x_52_, 0);
v_isSharedCheck_62_ = !lean_is_exclusive(v___x_52_);
if (v_isSharedCheck_62_ == 0)
{
lean_object* v_unused_63_; 
v_unused_63_ = lean_ctor_get(v___x_52_, 1);
lean_dec(v_unused_63_);
v___x_55_ = v___x_52_;
v_isShared_56_ = v_isSharedCheck_62_;
goto v_resetjp_54_;
}
else
{
lean_inc(v_toMul_53_);
lean_dec(v___x_52_);
v___x_55_ = lean_box(0);
v_isShared_56_ = v_isSharedCheck_62_;
goto v_resetjp_54_;
}
v_resetjp_54_:
{
lean_object* v___f_57_; lean_object* v___f_58_; lean_object* v___x_60_; 
lean_inc(v_a_47_);
v___f_57_ = lean_alloc_closure((void*)(lp_mathlib_Equiv_divRight_u2080___redArg___lam__0), 3, 2);
lean_closure_set(v___f_57_, 0, v_toDiv_49_);
lean_closure_set(v___f_57_, 1, v_a_47_);
v___f_58_ = lean_alloc_closure((void*)(lp_mathlib_Equiv_divRight_u2080___redArg___lam__1), 3, 2);
lean_closure_set(v___f_58_, 0, v_toMul_53_);
lean_closure_set(v___f_58_, 1, v_a_47_);
if (v_isShared_56_ == 0)
{
lean_ctor_set(v___x_55_, 1, v___f_58_);
lean_ctor_set(v___x_55_, 0, v___f_57_);
v___x_60_ = v___x_55_;
goto v_reusejp_59_;
}
else
{
lean_object* v_reuseFailAlloc_61_; 
v_reuseFailAlloc_61_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_61_, 0, v___f_57_);
lean_ctor_set(v_reuseFailAlloc_61_, 1, v___f_58_);
v___x_60_ = v_reuseFailAlloc_61_;
goto v_reusejp_59_;
}
v_reusejp_59_:
{
return v___x_60_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_divRight_u2080(lean_object* v_G_u2080_64_, lean_object* v_inst_65_, lean_object* v_a_66_, lean_object* v_ha_67_){
_start:
{
lean_object* v___x_68_; 
v___x_68_ = lp_mathlib_Equiv_divRight_u2080___redArg(v_inst_65_, v_a_66_);
return v___x_68_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_divLeft_u2080___redArg___lam__0(lean_object* v_toDiv_69_, lean_object* v_a_70_, lean_object* v_x_71_){
_start:
{
lean_object* v___x_72_; 
v___x_72_ = lean_apply_2(v_toDiv_69_, v_a_70_, v_x_71_);
return v___x_72_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_divLeft_u2080___redArg(lean_object* v_inst_73_, lean_object* v_a_74_){
_start:
{
lean_object* v___x_75_; lean_object* v___x_76_; lean_object* v_toDiv_77_; lean_object* v___f_78_; lean_object* v___x_79_; 
v___x_75_ = lp_mathlib_CommGroupWithZero_toGroupWithZero___redArg(v_inst_73_);
v___x_76_ = lp_mathlib_GroupWithZero_toDivInvMonoid___redArg(v___x_75_);
v_toDiv_77_ = lean_ctor_get(v___x_76_, 2);
lean_inc(v_toDiv_77_);
lean_dec_ref(v___x_76_);
v___f_78_ = lean_alloc_closure((void*)(lp_mathlib_Equiv_divLeft_u2080___redArg___lam__0), 3, 2);
lean_closure_set(v___f_78_, 0, v_toDiv_77_);
lean_closure_set(v___f_78_, 1, v_a_74_);
lean_inc_ref(v___f_78_);
v___x_79_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_79_, 0, v___f_78_);
lean_ctor_set(v___x_79_, 1, v___f_78_);
return v___x_79_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_divLeft_u2080(lean_object* v_G_u2080_80_, lean_object* v_inst_81_, lean_object* v_a_82_, lean_object* v_ha_83_){
_start:
{
lean_object* v___x_84_; 
v___x_84_ = lp_mathlib_Equiv_divLeft_u2080___redArg(v_inst_81_, v_a_82_);
return v___x_84_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Group_Units_Equiv(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_GroupWithZero_Units_Basic(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_Algebra_GroupWithZero_Units_Equiv(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Group_Units_Equiv(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_GroupWithZero_Units_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_Algebra_GroupWithZero_Units_Equiv(uint8_t builtin) {
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
lean_object* initialize_mathlib_Mathlib_Algebra_Group_Units_Equiv(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_GroupWithZero_Units_Basic(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Algebra_GroupWithZero_Units_Equiv(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Group_Units_Equiv(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_GroupWithZero_Units_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_GroupWithZero_Units_Equiv(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_Algebra_GroupWithZero_Units_Equiv(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_Algebra_GroupWithZero_Units_Equiv(builtin);
}
#ifdef __cplusplus
}
#endif
