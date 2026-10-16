// Lean compiler output
// Module: Mathlib.Algebra.Ring.Opposite
// Imports: public import Init public meta import Init public import Mathlib.Algebra.Group.Equiv.Opposite public import Mathlib.Algebra.GroupWithZero.Opposite public import Mathlib.Algebra.Ring.Hom.Defs public import Mathlib.Data.Int.Cast.Basic
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
lean_object* lp_mathlib_MulOpposite_instMul___redArg___lam__0(lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_MulOpposite_instAdd___redArg___lam__0(lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_AddOpposite_instAddMonoid___redArg(lean_object*);
lean_object* lp_mathlib_NonUnitalNonAssocSemiring_toDistrib___redArg(lean_object*);
lean_object* lp_mathlib_AddOpposite_instMul___redArg___lam__0(lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_MulOpposite_instAddCommMonoid___redArg(lean_object*);
lean_object* lp_mathlib_NonAssocSemiring_toMulZeroOneClass___redArg(lean_object*);
lean_object* lp_mathlib_MulOpposite_instMulZeroOneClass___redArg(lean_object*);
lean_object* lp_mathlib_NonAssocSemiring_toAddCommMonoidWithOne___redArg(lean_object*);
lean_object* lp_mathlib_MulOpposite_instAddMonoid___redArg(lean_object*);
lean_object* lp_mathlib_Semiring_toNonUnitalSemiring___redArg(lean_object*);
lean_object* lp_mathlib_Semiring_toNonAssocSemiring___redArg(lean_object*);
lean_object* lp_mathlib_Semiring_toMonoidWithZero___redArg(lean_object*);
lean_object* lp_mathlib_MulOpposite_instMonoidWithZero___redArg(lean_object*);
lean_object* lp_mathlib_Ring_toNonAssocRing___redArg(lean_object*);
lean_object* lp_mathlib_NonAssocRing_toAddCommGroupWithOne___redArg(lean_object*);
lean_object* lp_mathlib_AddCommGroupWithOne_toAddGroupWithOne___redArg(lean_object*);
lean_object* lp_mathlib_AddGroupWithOne_toAddGroup___redArg(lean_object*);
lean_object* lp_mathlib_MulOpposite_instAddGroup___redArg(lean_object*);
lean_object* lp_mathlib_MulOpposite_instAddCommGroup___redArg(lean_object*);
lean_object* lp_mathlib_AddMonoid_toAddZeroClass___redArg(lean_object*);
lean_object* lp_mathlib_AddMonoidHom_mulOp(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_AddMonoidHom_mulUnop___redArg(lean_object*, lean_object*);
lean_object* lp_mathlib_NonUnitalNonAssocRing_toNonUnitalNonAssocSemiring___redArg(lean_object*);
lean_object* lp_mathlib_NonAssocRing_toNonAssocSemiring___redArg(lean_object*);
lean_object* lp_mathlib_AddOpposite_instMulZeroOneClass___redArg(lean_object*);
lean_object* lp_mathlib_AddOpposite_instMonoidWithZero___redArg(lean_object*);
lean_object* lp_mathlib_MulOpposite_unop___boxed(lean_object*, lean_object*);
lean_object* l_Function_comp(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_AddCommGroupWithOne_toAddCommMonoidWithOne___redArg(lean_object*);
lean_object* lp_mathlib_AddOpposite_instSubNegMonoid___redArg(lean_object*);
lean_object* lp_mathlib_MulOpposite_op___boxed(lean_object*, lean_object*);
lean_object* lp_mathlib_Equiv_symm___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulOpposite_instDistrib___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulOpposite_instDistrib(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulOpposite_instNatCast___redArg___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulOpposite_instNatCast___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulOpposite_instNatCast(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddOpposite_instNatCast___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddOpposite_instNatCast(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulOpposite_instIntCast___redArg___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulOpposite_instIntCast___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulOpposite_instIntCast(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddOpposite_instIntCast___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddOpposite_instIntCast(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulOpposite_instAddMonoidWithOne___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulOpposite_instAddMonoidWithOne(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulOpposite_instAddCommMonoidWithOne___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulOpposite_instAddCommMonoidWithOne(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulOpposite_instAddGroupWithOne___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulOpposite_instAddGroupWithOne(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulOpposite_instAddCommGroupWithOne___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulOpposite_instAddCommGroupWithOne(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulOpposite_instNonUnitalNonAssocSemiring___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulOpposite_instNonUnitalNonAssocSemiring(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulOpposite_instNonUnitalSemiring___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulOpposite_instNonUnitalSemiring(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulOpposite_instNonAssocSemiring___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulOpposite_instNonAssocSemiring(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulOpposite_instSemiring___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulOpposite_instSemiring(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulOpposite_instNonUnitalCommSemiring___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulOpposite_instNonUnitalCommSemiring(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulOpposite_instCommSemiring___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulOpposite_instCommSemiring(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulOpposite_instNonUnitalNonAssocRing___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulOpposite_instNonUnitalNonAssocRing(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulOpposite_instNonUnitalRing___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulOpposite_instNonUnitalRing(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulOpposite_instNonAssocRing___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulOpposite_instNonAssocRing(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulOpposite_instRing___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulOpposite_instRing(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulOpposite_instNonUnitalCommRing___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulOpposite_instNonUnitalCommRing(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulOpposite_instCommRing___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulOpposite_instCommRing(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddOpposite_instDistrib___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddOpposite_instDistrib(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddOpposite_instAddCommMonoidWithOne___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddOpposite_instAddCommMonoidWithOne(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddOpposite_instAddCommGroupWithOne___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddOpposite_instAddCommGroupWithOne(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddOpposite_instNonUnitalNonAssocSemiring___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddOpposite_instNonUnitalNonAssocSemiring(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddOpposite_instNonUnitalSemiring___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddOpposite_instNonUnitalSemiring(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddOpposite_instNonAssocSemiring___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddOpposite_instNonAssocSemiring(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddOpposite_instSemiring___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddOpposite_instSemiring(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddOpposite_instNonUnitalCommSemiring___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddOpposite_instNonUnitalCommSemiring(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddOpposite_instCommSemiring___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddOpposite_instCommSemiring(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddOpposite_instNonUnitalNonAssocRing___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddOpposite_instNonUnitalNonAssocRing(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddOpposite_instNonUnitalRing___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddOpposite_instNonUnitalRing(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddOpposite_instNonAssocRing___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddOpposite_instNonAssocRing(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddOpposite_instRing___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddOpposite_instRing(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddOpposite_instNonUnitalCommRing___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddOpposite_instNonUnitalCommRing(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddOpposite_instCommRing___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddOpposite_instCommRing(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalRingHom_toOpposite___redArg___lam__0(lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_NonUnitalRingHom_toOpposite___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*1, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_MulOpposite_op___boxed, .m_arity = 2, .m_num_fixed = 1, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1))} };
static const lean_object* lp_mathlib_NonUnitalRingHom_toOpposite___redArg___closed__0 = (const lean_object*)&lp_mathlib_NonUnitalRingHom_toOpposite___redArg___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalRingHom_toOpposite___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalRingHom_toOpposite(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalRingHom_toOpposite___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_NonUnitalRingHom_fromOpposite___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*1, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_MulOpposite_unop___boxed, .m_arity = 2, .m_num_fixed = 1, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1))} };
static const lean_object* lp_mathlib_NonUnitalRingHom_fromOpposite___redArg___closed__0 = (const lean_object*)&lp_mathlib_NonUnitalRingHom_fromOpposite___redArg___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalRingHom_fromOpposite___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalRingHom_fromOpposite(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalRingHom_fromOpposite___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalRingHom_op___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalRingHom_op___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalRingHom_op___redArg___lam__1(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalRingHom_op___redArg___lam__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalRingHom_op___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalRingHom_op___redArg___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalRingHom_op(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalRingHom_op___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalRingHom_unop___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalRingHom_unop___redArg___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalRingHom_unop(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalRingHom_unop___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RingHom_toOpposite___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RingHom_toOpposite(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RingHom_toOpposite___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RingHom_fromOpposite___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RingHom_fromOpposite(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RingHom_fromOpposite___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RingHom_op___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RingHom_op(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RingHom_unop___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RingHom_unop(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulOpposite_instDistrib___redArg(lean_object* v_inst_1_){
_start:
{
lean_object* v_toMul_2_; lean_object* v_toAdd_3_; lean_object* v___x_5_; uint8_t v_isShared_6_; uint8_t v_isSharedCheck_12_; 
v_toMul_2_ = lean_ctor_get(v_inst_1_, 0);
v_toAdd_3_ = lean_ctor_get(v_inst_1_, 1);
v_isSharedCheck_12_ = !lean_is_exclusive(v_inst_1_);
if (v_isSharedCheck_12_ == 0)
{
v___x_5_ = v_inst_1_;
v_isShared_6_ = v_isSharedCheck_12_;
goto v_resetjp_4_;
}
else
{
lean_inc(v_toAdd_3_);
lean_inc(v_toMul_2_);
lean_dec(v_inst_1_);
v___x_5_ = lean_box(0);
v_isShared_6_ = v_isSharedCheck_12_;
goto v_resetjp_4_;
}
v_resetjp_4_:
{
lean_object* v___f_7_; lean_object* v___f_8_; lean_object* v___x_10_; 
v___f_7_ = lean_alloc_closure((void*)(lp_mathlib_MulOpposite_instMul___redArg___lam__0), 3, 1);
lean_closure_set(v___f_7_, 0, v_toMul_2_);
v___f_8_ = lean_alloc_closure((void*)(lp_mathlib_MulOpposite_instAdd___redArg___lam__0), 3, 1);
lean_closure_set(v___f_8_, 0, v_toAdd_3_);
if (v_isShared_6_ == 0)
{
lean_ctor_set(v___x_5_, 1, v___f_8_);
lean_ctor_set(v___x_5_, 0, v___f_7_);
v___x_10_ = v___x_5_;
goto v_reusejp_9_;
}
else
{
lean_object* v_reuseFailAlloc_11_; 
v_reuseFailAlloc_11_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_11_, 0, v___f_7_);
lean_ctor_set(v_reuseFailAlloc_11_, 1, v___f_8_);
v___x_10_ = v_reuseFailAlloc_11_;
goto v_reusejp_9_;
}
v_reusejp_9_:
{
return v___x_10_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulOpposite_instDistrib(lean_object* v_R_13_, lean_object* v_inst_14_){
_start:
{
lean_object* v___x_15_; 
v___x_15_ = lp_mathlib_MulOpposite_instDistrib___redArg(v_inst_14_);
return v___x_15_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulOpposite_instNatCast___redArg___lam__0(lean_object* v_inst_16_, lean_object* v_n_17_){
_start:
{
lean_object* v___x_18_; 
v___x_18_ = lean_apply_1(v_inst_16_, v_n_17_);
return v___x_18_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulOpposite_instNatCast___redArg(lean_object* v_inst_19_){
_start:
{
lean_object* v___f_20_; 
v___f_20_ = lean_alloc_closure((void*)(lp_mathlib_MulOpposite_instNatCast___redArg___lam__0), 2, 1);
lean_closure_set(v___f_20_, 0, v_inst_19_);
return v___f_20_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulOpposite_instNatCast(lean_object* v_R_21_, lean_object* v_inst_22_){
_start:
{
lean_object* v___f_23_; 
v___f_23_ = lean_alloc_closure((void*)(lp_mathlib_MulOpposite_instNatCast___redArg___lam__0), 2, 1);
lean_closure_set(v___f_23_, 0, v_inst_22_);
return v___f_23_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddOpposite_instNatCast___redArg(lean_object* v_inst_24_){
_start:
{
lean_object* v___f_25_; 
v___f_25_ = lean_alloc_closure((void*)(lp_mathlib_MulOpposite_instNatCast___redArg___lam__0), 2, 1);
lean_closure_set(v___f_25_, 0, v_inst_24_);
return v___f_25_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddOpposite_instNatCast(lean_object* v_R_26_, lean_object* v_inst_27_){
_start:
{
lean_object* v___f_28_; 
v___f_28_ = lean_alloc_closure((void*)(lp_mathlib_MulOpposite_instNatCast___redArg___lam__0), 2, 1);
lean_closure_set(v___f_28_, 0, v_inst_27_);
return v___f_28_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulOpposite_instIntCast___redArg___lam__0(lean_object* v_inst_29_, lean_object* v_n_30_){
_start:
{
lean_object* v___x_31_; 
v___x_31_ = lean_apply_1(v_inst_29_, v_n_30_);
return v___x_31_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulOpposite_instIntCast___redArg(lean_object* v_inst_32_){
_start:
{
lean_object* v___f_33_; 
v___f_33_ = lean_alloc_closure((void*)(lp_mathlib_MulOpposite_instIntCast___redArg___lam__0), 2, 1);
lean_closure_set(v___f_33_, 0, v_inst_32_);
return v___f_33_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulOpposite_instIntCast(lean_object* v_R_34_, lean_object* v_inst_35_){
_start:
{
lean_object* v___f_36_; 
v___f_36_ = lean_alloc_closure((void*)(lp_mathlib_MulOpposite_instIntCast___redArg___lam__0), 2, 1);
lean_closure_set(v___f_36_, 0, v_inst_35_);
return v___f_36_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddOpposite_instIntCast___redArg(lean_object* v_inst_37_){
_start:
{
lean_object* v___f_38_; 
v___f_38_ = lean_alloc_closure((void*)(lp_mathlib_MulOpposite_instIntCast___redArg___lam__0), 2, 1);
lean_closure_set(v___f_38_, 0, v_inst_37_);
return v___f_38_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddOpposite_instIntCast(lean_object* v_R_39_, lean_object* v_inst_40_){
_start:
{
lean_object* v___f_41_; 
v___f_41_ = lean_alloc_closure((void*)(lp_mathlib_MulOpposite_instIntCast___redArg___lam__0), 2, 1);
lean_closure_set(v___f_41_, 0, v_inst_40_);
return v___f_41_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulOpposite_instAddMonoidWithOne___redArg(lean_object* v_inst_42_){
_start:
{
lean_object* v_toNatCast_43_; lean_object* v_toAddMonoid_44_; lean_object* v_toOne_45_; lean_object* v___x_47_; uint8_t v_isShared_48_; uint8_t v_isSharedCheck_54_; 
v_toNatCast_43_ = lean_ctor_get(v_inst_42_, 0);
v_toAddMonoid_44_ = lean_ctor_get(v_inst_42_, 1);
v_toOne_45_ = lean_ctor_get(v_inst_42_, 2);
v_isSharedCheck_54_ = !lean_is_exclusive(v_inst_42_);
if (v_isSharedCheck_54_ == 0)
{
v___x_47_ = v_inst_42_;
v_isShared_48_ = v_isSharedCheck_54_;
goto v_resetjp_46_;
}
else
{
lean_inc(v_toOne_45_);
lean_inc(v_toAddMonoid_44_);
lean_inc(v_toNatCast_43_);
lean_dec(v_inst_42_);
v___x_47_ = lean_box(0);
v_isShared_48_ = v_isSharedCheck_54_;
goto v_resetjp_46_;
}
v_resetjp_46_:
{
lean_object* v___f_49_; lean_object* v___x_50_; lean_object* v___x_52_; 
v___f_49_ = lean_alloc_closure((void*)(lp_mathlib_MulOpposite_instNatCast___redArg___lam__0), 2, 1);
lean_closure_set(v___f_49_, 0, v_toNatCast_43_);
v___x_50_ = lp_mathlib_MulOpposite_instAddMonoid___redArg(v_toAddMonoid_44_);
if (v_isShared_48_ == 0)
{
lean_ctor_set(v___x_47_, 1, v___x_50_);
lean_ctor_set(v___x_47_, 0, v___f_49_);
v___x_52_ = v___x_47_;
goto v_reusejp_51_;
}
else
{
lean_object* v_reuseFailAlloc_53_; 
v_reuseFailAlloc_53_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v_reuseFailAlloc_53_, 0, v___f_49_);
lean_ctor_set(v_reuseFailAlloc_53_, 1, v___x_50_);
lean_ctor_set(v_reuseFailAlloc_53_, 2, v_toOne_45_);
v___x_52_ = v_reuseFailAlloc_53_;
goto v_reusejp_51_;
}
v_reusejp_51_:
{
return v___x_52_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulOpposite_instAddMonoidWithOne(lean_object* v_R_55_, lean_object* v_inst_56_){
_start:
{
lean_object* v___x_57_; 
v___x_57_ = lp_mathlib_MulOpposite_instAddMonoidWithOne___redArg(v_inst_56_);
return v___x_57_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulOpposite_instAddCommMonoidWithOne___redArg(lean_object* v_inst_58_){
_start:
{
lean_object* v___x_59_; 
v___x_59_ = lp_mathlib_MulOpposite_instAddMonoidWithOne___redArg(v_inst_58_);
return v___x_59_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulOpposite_instAddCommMonoidWithOne(lean_object* v_R_60_, lean_object* v_inst_61_){
_start:
{
lean_object* v___x_62_; 
v___x_62_ = lp_mathlib_MulOpposite_instAddMonoidWithOne___redArg(v_inst_61_);
return v___x_62_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulOpposite_instAddGroupWithOne___redArg(lean_object* v_inst_63_){
_start:
{
lean_object* v___x_64_; lean_object* v___x_65_; lean_object* v_toIntCast_66_; lean_object* v_toAddMonoidWithOne_67_; lean_object* v___x_69_; uint8_t v_isShared_70_; uint8_t v_isSharedCheck_79_; 
v___x_64_ = lp_mathlib_AddGroupWithOne_toAddGroup___redArg(v_inst_63_);
v___x_65_ = lp_mathlib_MulOpposite_instAddGroup___redArg(v___x_64_);
v_toIntCast_66_ = lean_ctor_get(v_inst_63_, 0);
v_toAddMonoidWithOne_67_ = lean_ctor_get(v_inst_63_, 1);
v_isSharedCheck_79_ = !lean_is_exclusive(v_inst_63_);
if (v_isSharedCheck_79_ == 0)
{
lean_object* v_unused_80_; lean_object* v_unused_81_; lean_object* v_unused_82_; 
v_unused_80_ = lean_ctor_get(v_inst_63_, 4);
lean_dec(v_unused_80_);
v_unused_81_ = lean_ctor_get(v_inst_63_, 3);
lean_dec(v_unused_81_);
v_unused_82_ = lean_ctor_get(v_inst_63_, 2);
lean_dec(v_unused_82_);
v___x_69_ = v_inst_63_;
v_isShared_70_ = v_isSharedCheck_79_;
goto v_resetjp_68_;
}
else
{
lean_inc(v_toAddMonoidWithOne_67_);
lean_inc(v_toIntCast_66_);
lean_dec(v_inst_63_);
v___x_69_ = lean_box(0);
v_isShared_70_ = v_isSharedCheck_79_;
goto v_resetjp_68_;
}
v_resetjp_68_:
{
lean_object* v___x_71_; lean_object* v_toNeg_72_; lean_object* v_toSub_73_; lean_object* v_toZSMul_74_; lean_object* v___f_75_; lean_object* v___x_77_; 
v___x_71_ = lp_mathlib_MulOpposite_instAddMonoidWithOne___redArg(v_toAddMonoidWithOne_67_);
v_toNeg_72_ = lean_ctor_get(v___x_65_, 1);
lean_inc(v_toNeg_72_);
v_toSub_73_ = lean_ctor_get(v___x_65_, 2);
lean_inc(v_toSub_73_);
v_toZSMul_74_ = lean_ctor_get(v___x_65_, 3);
lean_inc(v_toZSMul_74_);
lean_dec_ref(v___x_65_);
v___f_75_ = lean_alloc_closure((void*)(lp_mathlib_MulOpposite_instIntCast___redArg___lam__0), 2, 1);
lean_closure_set(v___f_75_, 0, v_toIntCast_66_);
if (v_isShared_70_ == 0)
{
lean_ctor_set(v___x_69_, 4, v_toZSMul_74_);
lean_ctor_set(v___x_69_, 3, v_toSub_73_);
lean_ctor_set(v___x_69_, 2, v_toNeg_72_);
lean_ctor_set(v___x_69_, 1, v___x_71_);
lean_ctor_set(v___x_69_, 0, v___f_75_);
v___x_77_ = v___x_69_;
goto v_reusejp_76_;
}
else
{
lean_object* v_reuseFailAlloc_78_; 
v_reuseFailAlloc_78_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_78_, 0, v___f_75_);
lean_ctor_set(v_reuseFailAlloc_78_, 1, v___x_71_);
lean_ctor_set(v_reuseFailAlloc_78_, 2, v_toNeg_72_);
lean_ctor_set(v_reuseFailAlloc_78_, 3, v_toSub_73_);
lean_ctor_set(v_reuseFailAlloc_78_, 4, v_toZSMul_74_);
v___x_77_ = v_reuseFailAlloc_78_;
goto v_reusejp_76_;
}
v_reusejp_76_:
{
return v___x_77_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulOpposite_instAddGroupWithOne(lean_object* v_R_83_, lean_object* v_inst_84_){
_start:
{
lean_object* v___x_85_; 
v___x_85_ = lp_mathlib_MulOpposite_instAddGroupWithOne___redArg(v_inst_84_);
return v___x_85_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulOpposite_instAddCommGroupWithOne___redArg(lean_object* v_inst_86_){
_start:
{
lean_object* v___x_87_; lean_object* v___x_88_; lean_object* v_toAddCommGroup_89_; lean_object* v___x_91_; uint8_t v_isShared_92_; uint8_t v_isSharedCheck_101_; 
v___x_87_ = lp_mathlib_AddCommGroupWithOne_toAddGroupWithOne___redArg(v_inst_86_);
v___x_88_ = lp_mathlib_MulOpposite_instAddGroupWithOne___redArg(v___x_87_);
v_toAddCommGroup_89_ = lean_ctor_get(v_inst_86_, 0);
v_isSharedCheck_101_ = !lean_is_exclusive(v_inst_86_);
if (v_isSharedCheck_101_ == 0)
{
lean_object* v_unused_102_; lean_object* v_unused_103_; lean_object* v_unused_104_; 
v_unused_102_ = lean_ctor_get(v_inst_86_, 3);
lean_dec(v_unused_102_);
v_unused_103_ = lean_ctor_get(v_inst_86_, 2);
lean_dec(v_unused_103_);
v_unused_104_ = lean_ctor_get(v_inst_86_, 1);
lean_dec(v_unused_104_);
v___x_91_ = v_inst_86_;
v_isShared_92_ = v_isSharedCheck_101_;
goto v_resetjp_90_;
}
else
{
lean_inc(v_toAddCommGroup_89_);
lean_dec(v_inst_86_);
v___x_91_ = lean_box(0);
v_isShared_92_ = v_isSharedCheck_101_;
goto v_resetjp_90_;
}
v_resetjp_90_:
{
lean_object* v___x_93_; lean_object* v_toAddMonoidWithOne_94_; lean_object* v_toIntCast_95_; lean_object* v_toNatCast_96_; lean_object* v_toOne_97_; lean_object* v___x_99_; 
v___x_93_ = lp_mathlib_MulOpposite_instAddCommGroup___redArg(v_toAddCommGroup_89_);
v_toAddMonoidWithOne_94_ = lean_ctor_get(v___x_88_, 1);
lean_inc_ref(v_toAddMonoidWithOne_94_);
v_toIntCast_95_ = lean_ctor_get(v___x_88_, 0);
lean_inc(v_toIntCast_95_);
lean_dec_ref(v___x_88_);
v_toNatCast_96_ = lean_ctor_get(v_toAddMonoidWithOne_94_, 0);
lean_inc(v_toNatCast_96_);
v_toOne_97_ = lean_ctor_get(v_toAddMonoidWithOne_94_, 2);
lean_inc(v_toOne_97_);
lean_dec_ref(v_toAddMonoidWithOne_94_);
if (v_isShared_92_ == 0)
{
lean_ctor_set(v___x_91_, 3, v_toOne_97_);
lean_ctor_set(v___x_91_, 2, v_toNatCast_96_);
lean_ctor_set(v___x_91_, 1, v_toIntCast_95_);
lean_ctor_set(v___x_91_, 0, v___x_93_);
v___x_99_ = v___x_91_;
goto v_reusejp_98_;
}
else
{
lean_object* v_reuseFailAlloc_100_; 
v_reuseFailAlloc_100_ = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(v_reuseFailAlloc_100_, 0, v___x_93_);
lean_ctor_set(v_reuseFailAlloc_100_, 1, v_toIntCast_95_);
lean_ctor_set(v_reuseFailAlloc_100_, 2, v_toNatCast_96_);
lean_ctor_set(v_reuseFailAlloc_100_, 3, v_toOne_97_);
v___x_99_ = v_reuseFailAlloc_100_;
goto v_reusejp_98_;
}
v_reusejp_98_:
{
return v___x_99_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulOpposite_instAddCommGroupWithOne(lean_object* v_R_105_, lean_object* v_inst_106_){
_start:
{
lean_object* v___x_107_; 
v___x_107_ = lp_mathlib_MulOpposite_instAddCommGroupWithOne___redArg(v_inst_106_);
return v___x_107_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulOpposite_instNonUnitalNonAssocSemiring___redArg(lean_object* v_inst_108_){
_start:
{
lean_object* v_toAddCommMonoid_109_; lean_object* v___x_110_; lean_object* v___x_111_; lean_object* v___x_112_; lean_object* v_toMul_113_; lean_object* v___x_115_; uint8_t v_isShared_116_; uint8_t v_isSharedCheck_120_; 
v_toAddCommMonoid_109_ = lean_ctor_get(v_inst_108_, 0);
lean_inc_ref(v_toAddCommMonoid_109_);
v___x_110_ = lp_mathlib_MulOpposite_instAddCommMonoid___redArg(v_toAddCommMonoid_109_);
v___x_111_ = lp_mathlib_NonUnitalNonAssocSemiring_toDistrib___redArg(v_inst_108_);
v___x_112_ = lp_mathlib_MulOpposite_instDistrib___redArg(v___x_111_);
v_toMul_113_ = lean_ctor_get(v___x_112_, 0);
v_isSharedCheck_120_ = !lean_is_exclusive(v___x_112_);
if (v_isSharedCheck_120_ == 0)
{
lean_object* v_unused_121_; 
v_unused_121_ = lean_ctor_get(v___x_112_, 1);
lean_dec(v_unused_121_);
v___x_115_ = v___x_112_;
v_isShared_116_ = v_isSharedCheck_120_;
goto v_resetjp_114_;
}
else
{
lean_inc(v_toMul_113_);
lean_dec(v___x_112_);
v___x_115_ = lean_box(0);
v_isShared_116_ = v_isSharedCheck_120_;
goto v_resetjp_114_;
}
v_resetjp_114_:
{
lean_object* v___x_118_; 
if (v_isShared_116_ == 0)
{
lean_ctor_set(v___x_115_, 1, v_toMul_113_);
lean_ctor_set(v___x_115_, 0, v___x_110_);
v___x_118_ = v___x_115_;
goto v_reusejp_117_;
}
else
{
lean_object* v_reuseFailAlloc_119_; 
v_reuseFailAlloc_119_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_119_, 0, v___x_110_);
lean_ctor_set(v_reuseFailAlloc_119_, 1, v_toMul_113_);
v___x_118_ = v_reuseFailAlloc_119_;
goto v_reusejp_117_;
}
v_reusejp_117_:
{
return v___x_118_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulOpposite_instNonUnitalNonAssocSemiring(lean_object* v_R_122_, lean_object* v_inst_123_){
_start:
{
lean_object* v___x_124_; 
v___x_124_ = lp_mathlib_MulOpposite_instNonUnitalNonAssocSemiring___redArg(v_inst_123_);
return v___x_124_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulOpposite_instNonUnitalSemiring___redArg(lean_object* v_inst_125_){
_start:
{
lean_object* v___x_126_; 
v___x_126_ = lp_mathlib_MulOpposite_instNonUnitalNonAssocSemiring___redArg(v_inst_125_);
return v___x_126_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulOpposite_instNonUnitalSemiring(lean_object* v_R_127_, lean_object* v_inst_128_){
_start:
{
lean_object* v___x_129_; 
v___x_129_ = lp_mathlib_MulOpposite_instNonUnitalNonAssocSemiring___redArg(v_inst_128_);
return v___x_129_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulOpposite_instNonAssocSemiring___redArg(lean_object* v_inst_130_){
_start:
{
lean_object* v_toNonUnitalNonAssocSemiring_131_; lean_object* v___x_132_; lean_object* v___x_133_; lean_object* v___x_134_; lean_object* v___x_135_; lean_object* v___x_136_; lean_object* v_toMulOneClass_137_; lean_object* v_toOne_138_; lean_object* v_toNatCast_139_; lean_object* v___x_141_; uint8_t v_isShared_142_; uint8_t v_isSharedCheck_146_; 
v_toNonUnitalNonAssocSemiring_131_ = lean_ctor_get(v_inst_130_, 0);
lean_inc_ref(v_toNonUnitalNonAssocSemiring_131_);
v___x_132_ = lp_mathlib_MulOpposite_instNonUnitalNonAssocSemiring___redArg(v_toNonUnitalNonAssocSemiring_131_);
lean_inc_ref(v_inst_130_);
v___x_133_ = lp_mathlib_NonAssocSemiring_toMulZeroOneClass___redArg(v_inst_130_);
v___x_134_ = lp_mathlib_MulOpposite_instMulZeroOneClass___redArg(v___x_133_);
v___x_135_ = lp_mathlib_NonAssocSemiring_toAddCommMonoidWithOne___redArg(v_inst_130_);
v___x_136_ = lp_mathlib_MulOpposite_instAddMonoidWithOne___redArg(v___x_135_);
v_toMulOneClass_137_ = lean_ctor_get(v___x_134_, 0);
lean_inc_ref(v_toMulOneClass_137_);
lean_dec_ref(v___x_134_);
v_toOne_138_ = lean_ctor_get(v_toMulOneClass_137_, 0);
lean_inc(v_toOne_138_);
lean_dec_ref(v_toMulOneClass_137_);
v_toNatCast_139_ = lean_ctor_get(v___x_136_, 0);
v_isSharedCheck_146_ = !lean_is_exclusive(v___x_136_);
if (v_isSharedCheck_146_ == 0)
{
lean_object* v_unused_147_; lean_object* v_unused_148_; 
v_unused_147_ = lean_ctor_get(v___x_136_, 2);
lean_dec(v_unused_147_);
v_unused_148_ = lean_ctor_get(v___x_136_, 1);
lean_dec(v_unused_148_);
v___x_141_ = v___x_136_;
v_isShared_142_ = v_isSharedCheck_146_;
goto v_resetjp_140_;
}
else
{
lean_inc(v_toNatCast_139_);
lean_dec(v___x_136_);
v___x_141_ = lean_box(0);
v_isShared_142_ = v_isSharedCheck_146_;
goto v_resetjp_140_;
}
v_resetjp_140_:
{
lean_object* v___x_144_; 
if (v_isShared_142_ == 0)
{
lean_ctor_set(v___x_141_, 2, v_toNatCast_139_);
lean_ctor_set(v___x_141_, 1, v_toOne_138_);
lean_ctor_set(v___x_141_, 0, v___x_132_);
v___x_144_ = v___x_141_;
goto v_reusejp_143_;
}
else
{
lean_object* v_reuseFailAlloc_145_; 
v_reuseFailAlloc_145_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v_reuseFailAlloc_145_, 0, v___x_132_);
lean_ctor_set(v_reuseFailAlloc_145_, 1, v_toOne_138_);
lean_ctor_set(v_reuseFailAlloc_145_, 2, v_toNatCast_139_);
v___x_144_ = v_reuseFailAlloc_145_;
goto v_reusejp_143_;
}
v_reusejp_143_:
{
return v___x_144_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulOpposite_instNonAssocSemiring(lean_object* v_R_149_, lean_object* v_inst_150_){
_start:
{
lean_object* v___x_151_; 
v___x_151_ = lp_mathlib_MulOpposite_instNonAssocSemiring___redArg(v_inst_150_);
return v___x_151_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulOpposite_instSemiring___redArg(lean_object* v_inst_152_){
_start:
{
lean_object* v___x_153_; lean_object* v___x_154_; lean_object* v___x_155_; lean_object* v___x_156_; lean_object* v___x_157_; lean_object* v___x_158_; lean_object* v_toMonoid_159_; lean_object* v_toAddCommMonoid_160_; lean_object* v_toMul_161_; lean_object* v_toOne_162_; lean_object* v_toNatCast_163_; lean_object* v___x_165_; uint8_t v_isShared_166_; uint8_t v_isSharedCheck_180_; 
v___x_153_ = lp_mathlib_Semiring_toNonUnitalSemiring___redArg(v_inst_152_);
v___x_154_ = lp_mathlib_MulOpposite_instNonUnitalNonAssocSemiring___redArg(v___x_153_);
lean_inc_ref(v_inst_152_);
v___x_155_ = lp_mathlib_Semiring_toNonAssocSemiring___redArg(v_inst_152_);
v___x_156_ = lp_mathlib_MulOpposite_instNonAssocSemiring___redArg(v___x_155_);
v___x_157_ = lp_mathlib_Semiring_toMonoidWithZero___redArg(v_inst_152_);
lean_dec_ref(v_inst_152_);
v___x_158_ = lp_mathlib_MulOpposite_instMonoidWithZero___redArg(v___x_157_);
v_toMonoid_159_ = lean_ctor_get(v___x_158_, 0);
lean_inc_ref(v_toMonoid_159_);
lean_dec_ref(v___x_158_);
v_toAddCommMonoid_160_ = lean_ctor_get(v___x_154_, 0);
lean_inc_ref(v_toAddCommMonoid_160_);
v_toMul_161_ = lean_ctor_get(v___x_154_, 1);
lean_inc(v_toMul_161_);
lean_dec_ref(v___x_154_);
v_toOne_162_ = lean_ctor_get(v___x_156_, 1);
v_toNatCast_163_ = lean_ctor_get(v___x_156_, 2);
v_isSharedCheck_180_ = !lean_is_exclusive(v___x_156_);
if (v_isSharedCheck_180_ == 0)
{
lean_object* v_unused_181_; 
v_unused_181_ = lean_ctor_get(v___x_156_, 0);
lean_dec(v_unused_181_);
v___x_165_ = v___x_156_;
v_isShared_166_ = v_isSharedCheck_180_;
goto v_resetjp_164_;
}
else
{
lean_inc(v_toNatCast_163_);
lean_inc(v_toOne_162_);
lean_dec(v___x_156_);
v___x_165_ = lean_box(0);
v_isShared_166_ = v_isSharedCheck_180_;
goto v_resetjp_164_;
}
v_resetjp_164_:
{
lean_object* v_toNPow_167_; lean_object* v___x_169_; uint8_t v_isShared_170_; uint8_t v_isSharedCheck_177_; 
v_toNPow_167_ = lean_ctor_get(v_toMonoid_159_, 2);
v_isSharedCheck_177_ = !lean_is_exclusive(v_toMonoid_159_);
if (v_isSharedCheck_177_ == 0)
{
lean_object* v_unused_178_; lean_object* v_unused_179_; 
v_unused_178_ = lean_ctor_get(v_toMonoid_159_, 1);
lean_dec(v_unused_178_);
v_unused_179_ = lean_ctor_get(v_toMonoid_159_, 0);
lean_dec(v_unused_179_);
v___x_169_ = v_toMonoid_159_;
v_isShared_170_ = v_isSharedCheck_177_;
goto v_resetjp_168_;
}
else
{
lean_inc(v_toNPow_167_);
lean_dec(v_toMonoid_159_);
v___x_169_ = lean_box(0);
v_isShared_170_ = v_isSharedCheck_177_;
goto v_resetjp_168_;
}
v_resetjp_168_:
{
lean_object* v___x_172_; 
if (v_isShared_170_ == 0)
{
lean_ctor_set(v___x_169_, 1, v_toMul_161_);
lean_ctor_set(v___x_169_, 0, v_toOne_162_);
v___x_172_ = v___x_169_;
goto v_reusejp_171_;
}
else
{
lean_object* v_reuseFailAlloc_176_; 
v_reuseFailAlloc_176_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v_reuseFailAlloc_176_, 0, v_toOne_162_);
lean_ctor_set(v_reuseFailAlloc_176_, 1, v_toMul_161_);
lean_ctor_set(v_reuseFailAlloc_176_, 2, v_toNPow_167_);
v___x_172_ = v_reuseFailAlloc_176_;
goto v_reusejp_171_;
}
v_reusejp_171_:
{
lean_object* v___x_174_; 
if (v_isShared_166_ == 0)
{
lean_ctor_set(v___x_165_, 1, v___x_172_);
lean_ctor_set(v___x_165_, 0, v_toAddCommMonoid_160_);
v___x_174_ = v___x_165_;
goto v_reusejp_173_;
}
else
{
lean_object* v_reuseFailAlloc_175_; 
v_reuseFailAlloc_175_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v_reuseFailAlloc_175_, 0, v_toAddCommMonoid_160_);
lean_ctor_set(v_reuseFailAlloc_175_, 1, v___x_172_);
lean_ctor_set(v_reuseFailAlloc_175_, 2, v_toNatCast_163_);
v___x_174_ = v_reuseFailAlloc_175_;
goto v_reusejp_173_;
}
v_reusejp_173_:
{
return v___x_174_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulOpposite_instSemiring(lean_object* v_R_182_, lean_object* v_inst_183_){
_start:
{
lean_object* v___x_184_; 
v___x_184_ = lp_mathlib_MulOpposite_instSemiring___redArg(v_inst_183_);
return v___x_184_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulOpposite_instNonUnitalCommSemiring___redArg(lean_object* v_inst_185_){
_start:
{
lean_object* v___x_186_; 
v___x_186_ = lp_mathlib_MulOpposite_instNonUnitalNonAssocSemiring___redArg(v_inst_185_);
return v___x_186_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulOpposite_instNonUnitalCommSemiring(lean_object* v_R_187_, lean_object* v_inst_188_){
_start:
{
lean_object* v___x_189_; 
v___x_189_ = lp_mathlib_MulOpposite_instNonUnitalNonAssocSemiring___redArg(v_inst_188_);
return v___x_189_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulOpposite_instCommSemiring___redArg(lean_object* v_inst_190_){
_start:
{
lean_object* v___x_191_; 
v___x_191_ = lp_mathlib_MulOpposite_instSemiring___redArg(v_inst_190_);
return v___x_191_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulOpposite_instCommSemiring(lean_object* v_R_192_, lean_object* v_inst_193_){
_start:
{
lean_object* v___x_194_; 
v___x_194_ = lp_mathlib_MulOpposite_instSemiring___redArg(v_inst_193_);
return v___x_194_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulOpposite_instNonUnitalNonAssocRing___redArg(lean_object* v_inst_195_){
_start:
{
lean_object* v_toAddCommGroup_196_; lean_object* v___x_197_; lean_object* v___x_198_; lean_object* v___x_199_; lean_object* v_toMul_200_; lean_object* v___x_202_; uint8_t v_isShared_203_; uint8_t v_isSharedCheck_207_; 
v_toAddCommGroup_196_ = lean_ctor_get(v_inst_195_, 0);
lean_inc_ref(v_toAddCommGroup_196_);
v___x_197_ = lp_mathlib_MulOpposite_instAddCommGroup___redArg(v_toAddCommGroup_196_);
v___x_198_ = lp_mathlib_NonUnitalNonAssocRing_toNonUnitalNonAssocSemiring___redArg(v_inst_195_);
v___x_199_ = lp_mathlib_MulOpposite_instNonUnitalNonAssocSemiring___redArg(v___x_198_);
v_toMul_200_ = lean_ctor_get(v___x_199_, 1);
v_isSharedCheck_207_ = !lean_is_exclusive(v___x_199_);
if (v_isSharedCheck_207_ == 0)
{
lean_object* v_unused_208_; 
v_unused_208_ = lean_ctor_get(v___x_199_, 0);
lean_dec(v_unused_208_);
v___x_202_ = v___x_199_;
v_isShared_203_ = v_isSharedCheck_207_;
goto v_resetjp_201_;
}
else
{
lean_inc(v_toMul_200_);
lean_dec(v___x_199_);
v___x_202_ = lean_box(0);
v_isShared_203_ = v_isSharedCheck_207_;
goto v_resetjp_201_;
}
v_resetjp_201_:
{
lean_object* v___x_205_; 
if (v_isShared_203_ == 0)
{
lean_ctor_set(v___x_202_, 0, v___x_197_);
v___x_205_ = v___x_202_;
goto v_reusejp_204_;
}
else
{
lean_object* v_reuseFailAlloc_206_; 
v_reuseFailAlloc_206_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_206_, 0, v___x_197_);
lean_ctor_set(v_reuseFailAlloc_206_, 1, v_toMul_200_);
v___x_205_ = v_reuseFailAlloc_206_;
goto v_reusejp_204_;
}
v_reusejp_204_:
{
return v___x_205_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulOpposite_instNonUnitalNonAssocRing(lean_object* v_R_209_, lean_object* v_inst_210_){
_start:
{
lean_object* v___x_211_; 
v___x_211_ = lp_mathlib_MulOpposite_instNonUnitalNonAssocRing___redArg(v_inst_210_);
return v___x_211_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulOpposite_instNonUnitalRing___redArg(lean_object* v_inst_212_){
_start:
{
lean_object* v___x_213_; 
v___x_213_ = lp_mathlib_MulOpposite_instNonUnitalNonAssocRing___redArg(v_inst_212_);
return v___x_213_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulOpposite_instNonUnitalRing(lean_object* v_R_214_, lean_object* v_inst_215_){
_start:
{
lean_object* v___x_216_; 
v___x_216_ = lp_mathlib_MulOpposite_instNonUnitalNonAssocRing___redArg(v_inst_215_);
return v___x_216_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulOpposite_instNonAssocRing___redArg(lean_object* v_inst_217_){
_start:
{
lean_object* v_toNonUnitalNonAssocRing_218_; lean_object* v___x_219_; lean_object* v___x_220_; lean_object* v___x_221_; lean_object* v___x_222_; lean_object* v___x_223_; lean_object* v_toOne_224_; lean_object* v_toNatCast_225_; lean_object* v_toIntCast_226_; lean_object* v___x_228_; uint8_t v_isShared_229_; uint8_t v_isSharedCheck_233_; 
v_toNonUnitalNonAssocRing_218_ = lean_ctor_get(v_inst_217_, 0);
lean_inc_ref(v_toNonUnitalNonAssocRing_218_);
v___x_219_ = lp_mathlib_MulOpposite_instNonUnitalNonAssocRing___redArg(v_toNonUnitalNonAssocRing_218_);
lean_inc_ref(v_inst_217_);
v___x_220_ = lp_mathlib_NonAssocRing_toNonAssocSemiring___redArg(v_inst_217_);
v___x_221_ = lp_mathlib_MulOpposite_instNonAssocSemiring___redArg(v___x_220_);
v___x_222_ = lp_mathlib_NonAssocRing_toAddCommGroupWithOne___redArg(v_inst_217_);
v___x_223_ = lp_mathlib_MulOpposite_instAddCommGroupWithOne___redArg(v___x_222_);
v_toOne_224_ = lean_ctor_get(v___x_221_, 1);
lean_inc(v_toOne_224_);
v_toNatCast_225_ = lean_ctor_get(v___x_221_, 2);
lean_inc(v_toNatCast_225_);
lean_dec_ref(v___x_221_);
v_toIntCast_226_ = lean_ctor_get(v___x_223_, 1);
v_isSharedCheck_233_ = !lean_is_exclusive(v___x_223_);
if (v_isSharedCheck_233_ == 0)
{
lean_object* v_unused_234_; lean_object* v_unused_235_; lean_object* v_unused_236_; 
v_unused_234_ = lean_ctor_get(v___x_223_, 3);
lean_dec(v_unused_234_);
v_unused_235_ = lean_ctor_get(v___x_223_, 2);
lean_dec(v_unused_235_);
v_unused_236_ = lean_ctor_get(v___x_223_, 0);
lean_dec(v_unused_236_);
v___x_228_ = v___x_223_;
v_isShared_229_ = v_isSharedCheck_233_;
goto v_resetjp_227_;
}
else
{
lean_inc(v_toIntCast_226_);
lean_dec(v___x_223_);
v___x_228_ = lean_box(0);
v_isShared_229_ = v_isSharedCheck_233_;
goto v_resetjp_227_;
}
v_resetjp_227_:
{
lean_object* v___x_231_; 
if (v_isShared_229_ == 0)
{
lean_ctor_set(v___x_228_, 3, v_toIntCast_226_);
lean_ctor_set(v___x_228_, 2, v_toNatCast_225_);
lean_ctor_set(v___x_228_, 1, v_toOne_224_);
lean_ctor_set(v___x_228_, 0, v___x_219_);
v___x_231_ = v___x_228_;
goto v_reusejp_230_;
}
else
{
lean_object* v_reuseFailAlloc_232_; 
v_reuseFailAlloc_232_ = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(v_reuseFailAlloc_232_, 0, v___x_219_);
lean_ctor_set(v_reuseFailAlloc_232_, 1, v_toOne_224_);
lean_ctor_set(v_reuseFailAlloc_232_, 2, v_toNatCast_225_);
lean_ctor_set(v_reuseFailAlloc_232_, 3, v_toIntCast_226_);
v___x_231_ = v_reuseFailAlloc_232_;
goto v_reusejp_230_;
}
v_reusejp_230_:
{
return v___x_231_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulOpposite_instNonAssocRing(lean_object* v_R_237_, lean_object* v_inst_238_){
_start:
{
lean_object* v___x_239_; 
v___x_239_ = lp_mathlib_MulOpposite_instNonAssocRing___redArg(v_inst_238_);
return v___x_239_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulOpposite_instRing___redArg(lean_object* v_inst_240_){
_start:
{
lean_object* v_toSemiring_241_; lean_object* v___x_242_; lean_object* v___x_243_; lean_object* v___x_245_; uint8_t v_isShared_246_; uint8_t v_isSharedCheck_257_; 
v_toSemiring_241_ = lean_ctor_get(v_inst_240_, 0);
lean_inc_ref(v_toSemiring_241_);
v___x_242_ = lp_mathlib_MulOpposite_instSemiring___redArg(v_toSemiring_241_);
v___x_243_ = lp_mathlib_Ring_toNonAssocRing___redArg(v_inst_240_);
v_isSharedCheck_257_ = !lean_is_exclusive(v_inst_240_);
if (v_isSharedCheck_257_ == 0)
{
lean_object* v_unused_258_; lean_object* v_unused_259_; lean_object* v_unused_260_; lean_object* v_unused_261_; lean_object* v_unused_262_; 
v_unused_258_ = lean_ctor_get(v_inst_240_, 4);
lean_dec(v_unused_258_);
v_unused_259_ = lean_ctor_get(v_inst_240_, 3);
lean_dec(v_unused_259_);
v_unused_260_ = lean_ctor_get(v_inst_240_, 2);
lean_dec(v_unused_260_);
v_unused_261_ = lean_ctor_get(v_inst_240_, 1);
lean_dec(v_unused_261_);
v_unused_262_ = lean_ctor_get(v_inst_240_, 0);
lean_dec(v_unused_262_);
v___x_245_ = v_inst_240_;
v_isShared_246_ = v_isSharedCheck_257_;
goto v_resetjp_244_;
}
else
{
lean_dec(v_inst_240_);
v___x_245_ = lean_box(0);
v_isShared_246_ = v_isSharedCheck_257_;
goto v_resetjp_244_;
}
v_resetjp_244_:
{
lean_object* v___x_247_; lean_object* v___x_248_; lean_object* v_toAddCommGroup_249_; lean_object* v_toIntCast_250_; lean_object* v_toNeg_251_; lean_object* v_toSub_252_; lean_object* v_toZSMul_253_; lean_object* v___x_255_; 
v___x_247_ = lp_mathlib_NonAssocRing_toAddCommGroupWithOne___redArg(v___x_243_);
v___x_248_ = lp_mathlib_MulOpposite_instAddCommGroupWithOne___redArg(v___x_247_);
v_toAddCommGroup_249_ = lean_ctor_get(v___x_248_, 0);
lean_inc_ref(v_toAddCommGroup_249_);
v_toIntCast_250_ = lean_ctor_get(v___x_248_, 1);
lean_inc(v_toIntCast_250_);
lean_dec_ref(v___x_248_);
v_toNeg_251_ = lean_ctor_get(v_toAddCommGroup_249_, 1);
lean_inc(v_toNeg_251_);
v_toSub_252_ = lean_ctor_get(v_toAddCommGroup_249_, 2);
lean_inc(v_toSub_252_);
v_toZSMul_253_ = lean_ctor_get(v_toAddCommGroup_249_, 3);
lean_inc(v_toZSMul_253_);
lean_dec_ref(v_toAddCommGroup_249_);
if (v_isShared_246_ == 0)
{
lean_ctor_set(v___x_245_, 4, v_toIntCast_250_);
lean_ctor_set(v___x_245_, 3, v_toZSMul_253_);
lean_ctor_set(v___x_245_, 2, v_toSub_252_);
lean_ctor_set(v___x_245_, 1, v_toNeg_251_);
lean_ctor_set(v___x_245_, 0, v___x_242_);
v___x_255_ = v___x_245_;
goto v_reusejp_254_;
}
else
{
lean_object* v_reuseFailAlloc_256_; 
v_reuseFailAlloc_256_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_256_, 0, v___x_242_);
lean_ctor_set(v_reuseFailAlloc_256_, 1, v_toNeg_251_);
lean_ctor_set(v_reuseFailAlloc_256_, 2, v_toSub_252_);
lean_ctor_set(v_reuseFailAlloc_256_, 3, v_toZSMul_253_);
lean_ctor_set(v_reuseFailAlloc_256_, 4, v_toIntCast_250_);
v___x_255_ = v_reuseFailAlloc_256_;
goto v_reusejp_254_;
}
v_reusejp_254_:
{
return v___x_255_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulOpposite_instRing(lean_object* v_R_263_, lean_object* v_inst_264_){
_start:
{
lean_object* v___x_265_; 
v___x_265_ = lp_mathlib_MulOpposite_instRing___redArg(v_inst_264_);
return v___x_265_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulOpposite_instNonUnitalCommRing___redArg(lean_object* v_inst_266_){
_start:
{
lean_object* v___x_267_; 
v___x_267_ = lp_mathlib_MulOpposite_instNonUnitalNonAssocRing___redArg(v_inst_266_);
return v___x_267_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulOpposite_instNonUnitalCommRing(lean_object* v_R_268_, lean_object* v_inst_269_){
_start:
{
lean_object* v___x_270_; 
v___x_270_ = lp_mathlib_MulOpposite_instNonUnitalNonAssocRing___redArg(v_inst_269_);
return v___x_270_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulOpposite_instCommRing___redArg(lean_object* v_inst_271_){
_start:
{
lean_object* v___x_272_; 
v___x_272_ = lp_mathlib_MulOpposite_instRing___redArg(v_inst_271_);
return v___x_272_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulOpposite_instCommRing(lean_object* v_R_273_, lean_object* v_inst_274_){
_start:
{
lean_object* v___x_275_; 
v___x_275_ = lp_mathlib_MulOpposite_instRing___redArg(v_inst_274_);
return v___x_275_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddOpposite_instDistrib___redArg(lean_object* v_inst_276_){
_start:
{
lean_object* v_toMul_277_; lean_object* v_toAdd_278_; lean_object* v___x_280_; uint8_t v_isShared_281_; uint8_t v_isSharedCheck_287_; 
v_toMul_277_ = lean_ctor_get(v_inst_276_, 0);
v_toAdd_278_ = lean_ctor_get(v_inst_276_, 1);
v_isSharedCheck_287_ = !lean_is_exclusive(v_inst_276_);
if (v_isSharedCheck_287_ == 0)
{
v___x_280_ = v_inst_276_;
v_isShared_281_ = v_isSharedCheck_287_;
goto v_resetjp_279_;
}
else
{
lean_inc(v_toAdd_278_);
lean_inc(v_toMul_277_);
lean_dec(v_inst_276_);
v___x_280_ = lean_box(0);
v_isShared_281_ = v_isSharedCheck_287_;
goto v_resetjp_279_;
}
v_resetjp_279_:
{
lean_object* v___f_282_; lean_object* v___f_283_; lean_object* v___x_285_; 
v___f_282_ = lean_alloc_closure((void*)(lp_mathlib_AddOpposite_instMul___redArg___lam__0), 3, 1);
lean_closure_set(v___f_282_, 0, v_toMul_277_);
v___f_283_ = lean_alloc_closure((void*)(lp_mathlib_MulOpposite_instMul___redArg___lam__0), 3, 1);
lean_closure_set(v___f_283_, 0, v_toAdd_278_);
if (v_isShared_281_ == 0)
{
lean_ctor_set(v___x_280_, 1, v___f_283_);
lean_ctor_set(v___x_280_, 0, v___f_282_);
v___x_285_ = v___x_280_;
goto v_reusejp_284_;
}
else
{
lean_object* v_reuseFailAlloc_286_; 
v_reuseFailAlloc_286_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_286_, 0, v___f_282_);
lean_ctor_set(v_reuseFailAlloc_286_, 1, v___f_283_);
v___x_285_ = v_reuseFailAlloc_286_;
goto v_reusejp_284_;
}
v_reusejp_284_:
{
return v___x_285_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddOpposite_instDistrib(lean_object* v_R_288_, lean_object* v_inst_289_){
_start:
{
lean_object* v___x_290_; 
v___x_290_ = lp_mathlib_AddOpposite_instDistrib___redArg(v_inst_289_);
return v___x_290_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddOpposite_instAddCommMonoidWithOne___redArg(lean_object* v_inst_291_){
_start:
{
lean_object* v_toNatCast_292_; lean_object* v_toAddMonoid_293_; lean_object* v_toOne_294_; lean_object* v___x_296_; uint8_t v_isShared_297_; uint8_t v_isSharedCheck_303_; 
v_toNatCast_292_ = lean_ctor_get(v_inst_291_, 0);
v_toAddMonoid_293_ = lean_ctor_get(v_inst_291_, 1);
v_toOne_294_ = lean_ctor_get(v_inst_291_, 2);
v_isSharedCheck_303_ = !lean_is_exclusive(v_inst_291_);
if (v_isSharedCheck_303_ == 0)
{
v___x_296_ = v_inst_291_;
v_isShared_297_ = v_isSharedCheck_303_;
goto v_resetjp_295_;
}
else
{
lean_inc(v_toOne_294_);
lean_inc(v_toAddMonoid_293_);
lean_inc(v_toNatCast_292_);
lean_dec(v_inst_291_);
v___x_296_ = lean_box(0);
v_isShared_297_ = v_isSharedCheck_303_;
goto v_resetjp_295_;
}
v_resetjp_295_:
{
lean_object* v___x_298_; lean_object* v___f_299_; lean_object* v___x_301_; 
v___x_298_ = lp_mathlib_AddOpposite_instAddMonoid___redArg(v_toAddMonoid_293_);
v___f_299_ = lean_alloc_closure((void*)(lp_mathlib_MulOpposite_instNatCast___redArg___lam__0), 2, 1);
lean_closure_set(v___f_299_, 0, v_toNatCast_292_);
if (v_isShared_297_ == 0)
{
lean_ctor_set(v___x_296_, 1, v___x_298_);
lean_ctor_set(v___x_296_, 0, v___f_299_);
v___x_301_ = v___x_296_;
goto v_reusejp_300_;
}
else
{
lean_object* v_reuseFailAlloc_302_; 
v_reuseFailAlloc_302_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v_reuseFailAlloc_302_, 0, v___f_299_);
lean_ctor_set(v_reuseFailAlloc_302_, 1, v___x_298_);
lean_ctor_set(v_reuseFailAlloc_302_, 2, v_toOne_294_);
v___x_301_ = v_reuseFailAlloc_302_;
goto v_reusejp_300_;
}
v_reusejp_300_:
{
return v___x_301_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddOpposite_instAddCommMonoidWithOne(lean_object* v_R_304_, lean_object* v_inst_305_){
_start:
{
lean_object* v___x_306_; 
v___x_306_ = lp_mathlib_AddOpposite_instAddCommMonoidWithOne___redArg(v_inst_305_);
return v___x_306_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddOpposite_instAddCommGroupWithOne___redArg(lean_object* v_inst_307_){
_start:
{
lean_object* v___x_308_; lean_object* v___x_309_; lean_object* v_toAddCommGroup_310_; lean_object* v___x_311_; lean_object* v___x_312_; lean_object* v___x_314_; uint8_t v_isShared_315_; uint8_t v_isSharedCheck_323_; 
v___x_308_ = lp_mathlib_AddCommGroupWithOne_toAddCommMonoidWithOne___redArg(v_inst_307_);
v___x_309_ = lp_mathlib_AddOpposite_instAddCommMonoidWithOne___redArg(v___x_308_);
v_toAddCommGroup_310_ = lean_ctor_get(v_inst_307_, 0);
lean_inc_ref(v_toAddCommGroup_310_);
v___x_311_ = lp_mathlib_AddOpposite_instSubNegMonoid___redArg(v_toAddCommGroup_310_);
v___x_312_ = lp_mathlib_AddCommGroupWithOne_toAddGroupWithOne___redArg(v_inst_307_);
v_isSharedCheck_323_ = !lean_is_exclusive(v_inst_307_);
if (v_isSharedCheck_323_ == 0)
{
lean_object* v_unused_324_; lean_object* v_unused_325_; lean_object* v_unused_326_; lean_object* v_unused_327_; 
v_unused_324_ = lean_ctor_get(v_inst_307_, 3);
lean_dec(v_unused_324_);
v_unused_325_ = lean_ctor_get(v_inst_307_, 2);
lean_dec(v_unused_325_);
v_unused_326_ = lean_ctor_get(v_inst_307_, 1);
lean_dec(v_unused_326_);
v_unused_327_ = lean_ctor_get(v_inst_307_, 0);
lean_dec(v_unused_327_);
v___x_314_ = v_inst_307_;
v_isShared_315_ = v_isSharedCheck_323_;
goto v_resetjp_313_;
}
else
{
lean_dec(v_inst_307_);
v___x_314_ = lean_box(0);
v_isShared_315_ = v_isSharedCheck_323_;
goto v_resetjp_313_;
}
v_resetjp_313_:
{
lean_object* v_toIntCast_316_; lean_object* v_toNatCast_317_; lean_object* v_toOne_318_; lean_object* v___f_319_; lean_object* v___x_321_; 
v_toIntCast_316_ = lean_ctor_get(v___x_312_, 0);
lean_inc(v_toIntCast_316_);
lean_dec_ref(v___x_312_);
v_toNatCast_317_ = lean_ctor_get(v___x_309_, 0);
lean_inc(v_toNatCast_317_);
v_toOne_318_ = lean_ctor_get(v___x_309_, 2);
lean_inc(v_toOne_318_);
lean_dec_ref(v___x_309_);
v___f_319_ = lean_alloc_closure((void*)(lp_mathlib_MulOpposite_instIntCast___redArg___lam__0), 2, 1);
lean_closure_set(v___f_319_, 0, v_toIntCast_316_);
if (v_isShared_315_ == 0)
{
lean_ctor_set(v___x_314_, 3, v_toOne_318_);
lean_ctor_set(v___x_314_, 2, v_toNatCast_317_);
lean_ctor_set(v___x_314_, 1, v___f_319_);
lean_ctor_set(v___x_314_, 0, v___x_311_);
v___x_321_ = v___x_314_;
goto v_reusejp_320_;
}
else
{
lean_object* v_reuseFailAlloc_322_; 
v_reuseFailAlloc_322_ = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(v_reuseFailAlloc_322_, 0, v___x_311_);
lean_ctor_set(v_reuseFailAlloc_322_, 1, v___f_319_);
lean_ctor_set(v_reuseFailAlloc_322_, 2, v_toNatCast_317_);
lean_ctor_set(v_reuseFailAlloc_322_, 3, v_toOne_318_);
v___x_321_ = v_reuseFailAlloc_322_;
goto v_reusejp_320_;
}
v_reusejp_320_:
{
return v___x_321_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddOpposite_instAddCommGroupWithOne(lean_object* v_R_328_, lean_object* v_inst_329_){
_start:
{
lean_object* v___x_330_; 
v___x_330_ = lp_mathlib_AddOpposite_instAddCommGroupWithOne___redArg(v_inst_329_);
return v___x_330_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddOpposite_instNonUnitalNonAssocSemiring___redArg(lean_object* v_inst_331_){
_start:
{
lean_object* v_toAddCommMonoid_332_; lean_object* v___x_333_; lean_object* v___x_334_; lean_object* v___x_335_; lean_object* v_toMul_336_; lean_object* v___x_338_; uint8_t v_isShared_339_; uint8_t v_isSharedCheck_343_; 
v_toAddCommMonoid_332_ = lean_ctor_get(v_inst_331_, 0);
lean_inc_ref(v_toAddCommMonoid_332_);
v___x_333_ = lp_mathlib_AddOpposite_instAddMonoid___redArg(v_toAddCommMonoid_332_);
v___x_334_ = lp_mathlib_NonUnitalNonAssocSemiring_toDistrib___redArg(v_inst_331_);
v___x_335_ = lp_mathlib_AddOpposite_instDistrib___redArg(v___x_334_);
v_toMul_336_ = lean_ctor_get(v___x_335_, 0);
v_isSharedCheck_343_ = !lean_is_exclusive(v___x_335_);
if (v_isSharedCheck_343_ == 0)
{
lean_object* v_unused_344_; 
v_unused_344_ = lean_ctor_get(v___x_335_, 1);
lean_dec(v_unused_344_);
v___x_338_ = v___x_335_;
v_isShared_339_ = v_isSharedCheck_343_;
goto v_resetjp_337_;
}
else
{
lean_inc(v_toMul_336_);
lean_dec(v___x_335_);
v___x_338_ = lean_box(0);
v_isShared_339_ = v_isSharedCheck_343_;
goto v_resetjp_337_;
}
v_resetjp_337_:
{
lean_object* v___x_341_; 
if (v_isShared_339_ == 0)
{
lean_ctor_set(v___x_338_, 1, v_toMul_336_);
lean_ctor_set(v___x_338_, 0, v___x_333_);
v___x_341_ = v___x_338_;
goto v_reusejp_340_;
}
else
{
lean_object* v_reuseFailAlloc_342_; 
v_reuseFailAlloc_342_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_342_, 0, v___x_333_);
lean_ctor_set(v_reuseFailAlloc_342_, 1, v_toMul_336_);
v___x_341_ = v_reuseFailAlloc_342_;
goto v_reusejp_340_;
}
v_reusejp_340_:
{
return v___x_341_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddOpposite_instNonUnitalNonAssocSemiring(lean_object* v_R_345_, lean_object* v_inst_346_){
_start:
{
lean_object* v___x_347_; 
v___x_347_ = lp_mathlib_AddOpposite_instNonUnitalNonAssocSemiring___redArg(v_inst_346_);
return v___x_347_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddOpposite_instNonUnitalSemiring___redArg(lean_object* v_inst_348_){
_start:
{
lean_object* v___x_349_; 
v___x_349_ = lp_mathlib_AddOpposite_instNonUnitalNonAssocSemiring___redArg(v_inst_348_);
return v___x_349_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddOpposite_instNonUnitalSemiring(lean_object* v_R_350_, lean_object* v_inst_351_){
_start:
{
lean_object* v___x_352_; 
v___x_352_ = lp_mathlib_AddOpposite_instNonUnitalNonAssocSemiring___redArg(v_inst_351_);
return v___x_352_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddOpposite_instNonAssocSemiring___redArg(lean_object* v_inst_353_){
_start:
{
lean_object* v_toNonUnitalNonAssocSemiring_354_; lean_object* v___x_355_; lean_object* v___x_356_; lean_object* v___x_357_; lean_object* v___x_358_; lean_object* v___x_359_; lean_object* v_toMulOneClass_360_; lean_object* v_toOne_361_; lean_object* v_toNatCast_362_; lean_object* v___x_364_; uint8_t v_isShared_365_; uint8_t v_isSharedCheck_369_; 
v_toNonUnitalNonAssocSemiring_354_ = lean_ctor_get(v_inst_353_, 0);
lean_inc_ref(v_toNonUnitalNonAssocSemiring_354_);
v___x_355_ = lp_mathlib_AddOpposite_instNonUnitalNonAssocSemiring___redArg(v_toNonUnitalNonAssocSemiring_354_);
lean_inc_ref(v_inst_353_);
v___x_356_ = lp_mathlib_NonAssocSemiring_toMulZeroOneClass___redArg(v_inst_353_);
v___x_357_ = lp_mathlib_AddOpposite_instMulZeroOneClass___redArg(v___x_356_);
v___x_358_ = lp_mathlib_NonAssocSemiring_toAddCommMonoidWithOne___redArg(v_inst_353_);
v___x_359_ = lp_mathlib_AddOpposite_instAddCommMonoidWithOne___redArg(v___x_358_);
v_toMulOneClass_360_ = lean_ctor_get(v___x_357_, 0);
lean_inc_ref(v_toMulOneClass_360_);
lean_dec_ref(v___x_357_);
v_toOne_361_ = lean_ctor_get(v_toMulOneClass_360_, 0);
lean_inc(v_toOne_361_);
lean_dec_ref(v_toMulOneClass_360_);
v_toNatCast_362_ = lean_ctor_get(v___x_359_, 0);
v_isSharedCheck_369_ = !lean_is_exclusive(v___x_359_);
if (v_isSharedCheck_369_ == 0)
{
lean_object* v_unused_370_; lean_object* v_unused_371_; 
v_unused_370_ = lean_ctor_get(v___x_359_, 2);
lean_dec(v_unused_370_);
v_unused_371_ = lean_ctor_get(v___x_359_, 1);
lean_dec(v_unused_371_);
v___x_364_ = v___x_359_;
v_isShared_365_ = v_isSharedCheck_369_;
goto v_resetjp_363_;
}
else
{
lean_inc(v_toNatCast_362_);
lean_dec(v___x_359_);
v___x_364_ = lean_box(0);
v_isShared_365_ = v_isSharedCheck_369_;
goto v_resetjp_363_;
}
v_resetjp_363_:
{
lean_object* v___x_367_; 
if (v_isShared_365_ == 0)
{
lean_ctor_set(v___x_364_, 2, v_toNatCast_362_);
lean_ctor_set(v___x_364_, 1, v_toOne_361_);
lean_ctor_set(v___x_364_, 0, v___x_355_);
v___x_367_ = v___x_364_;
goto v_reusejp_366_;
}
else
{
lean_object* v_reuseFailAlloc_368_; 
v_reuseFailAlloc_368_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v_reuseFailAlloc_368_, 0, v___x_355_);
lean_ctor_set(v_reuseFailAlloc_368_, 1, v_toOne_361_);
lean_ctor_set(v_reuseFailAlloc_368_, 2, v_toNatCast_362_);
v___x_367_ = v_reuseFailAlloc_368_;
goto v_reusejp_366_;
}
v_reusejp_366_:
{
return v___x_367_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddOpposite_instNonAssocSemiring(lean_object* v_R_372_, lean_object* v_inst_373_){
_start:
{
lean_object* v___x_374_; 
v___x_374_ = lp_mathlib_AddOpposite_instNonAssocSemiring___redArg(v_inst_373_);
return v___x_374_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddOpposite_instSemiring___redArg(lean_object* v_inst_375_){
_start:
{
lean_object* v___x_376_; lean_object* v___x_377_; lean_object* v___x_378_; lean_object* v___x_379_; lean_object* v___x_380_; lean_object* v___x_381_; lean_object* v_toMonoid_382_; lean_object* v_toAddCommMonoid_383_; lean_object* v_toMul_384_; lean_object* v_toOne_385_; lean_object* v_toNatCast_386_; lean_object* v___x_388_; uint8_t v_isShared_389_; uint8_t v_isSharedCheck_403_; 
v___x_376_ = lp_mathlib_Semiring_toNonUnitalSemiring___redArg(v_inst_375_);
v___x_377_ = lp_mathlib_AddOpposite_instNonUnitalNonAssocSemiring___redArg(v___x_376_);
lean_inc_ref(v_inst_375_);
v___x_378_ = lp_mathlib_Semiring_toNonAssocSemiring___redArg(v_inst_375_);
v___x_379_ = lp_mathlib_AddOpposite_instNonAssocSemiring___redArg(v___x_378_);
v___x_380_ = lp_mathlib_Semiring_toMonoidWithZero___redArg(v_inst_375_);
lean_dec_ref(v_inst_375_);
v___x_381_ = lp_mathlib_AddOpposite_instMonoidWithZero___redArg(v___x_380_);
v_toMonoid_382_ = lean_ctor_get(v___x_381_, 0);
lean_inc_ref(v_toMonoid_382_);
lean_dec_ref(v___x_381_);
v_toAddCommMonoid_383_ = lean_ctor_get(v___x_377_, 0);
lean_inc_ref(v_toAddCommMonoid_383_);
v_toMul_384_ = lean_ctor_get(v___x_377_, 1);
lean_inc(v_toMul_384_);
lean_dec_ref(v___x_377_);
v_toOne_385_ = lean_ctor_get(v___x_379_, 1);
v_toNatCast_386_ = lean_ctor_get(v___x_379_, 2);
v_isSharedCheck_403_ = !lean_is_exclusive(v___x_379_);
if (v_isSharedCheck_403_ == 0)
{
lean_object* v_unused_404_; 
v_unused_404_ = lean_ctor_get(v___x_379_, 0);
lean_dec(v_unused_404_);
v___x_388_ = v___x_379_;
v_isShared_389_ = v_isSharedCheck_403_;
goto v_resetjp_387_;
}
else
{
lean_inc(v_toNatCast_386_);
lean_inc(v_toOne_385_);
lean_dec(v___x_379_);
v___x_388_ = lean_box(0);
v_isShared_389_ = v_isSharedCheck_403_;
goto v_resetjp_387_;
}
v_resetjp_387_:
{
lean_object* v_toNPow_390_; lean_object* v___x_392_; uint8_t v_isShared_393_; uint8_t v_isSharedCheck_400_; 
v_toNPow_390_ = lean_ctor_get(v_toMonoid_382_, 2);
v_isSharedCheck_400_ = !lean_is_exclusive(v_toMonoid_382_);
if (v_isSharedCheck_400_ == 0)
{
lean_object* v_unused_401_; lean_object* v_unused_402_; 
v_unused_401_ = lean_ctor_get(v_toMonoid_382_, 1);
lean_dec(v_unused_401_);
v_unused_402_ = lean_ctor_get(v_toMonoid_382_, 0);
lean_dec(v_unused_402_);
v___x_392_ = v_toMonoid_382_;
v_isShared_393_ = v_isSharedCheck_400_;
goto v_resetjp_391_;
}
else
{
lean_inc(v_toNPow_390_);
lean_dec(v_toMonoid_382_);
v___x_392_ = lean_box(0);
v_isShared_393_ = v_isSharedCheck_400_;
goto v_resetjp_391_;
}
v_resetjp_391_:
{
lean_object* v___x_395_; 
if (v_isShared_393_ == 0)
{
lean_ctor_set(v___x_392_, 1, v_toMul_384_);
lean_ctor_set(v___x_392_, 0, v_toOne_385_);
v___x_395_ = v___x_392_;
goto v_reusejp_394_;
}
else
{
lean_object* v_reuseFailAlloc_399_; 
v_reuseFailAlloc_399_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v_reuseFailAlloc_399_, 0, v_toOne_385_);
lean_ctor_set(v_reuseFailAlloc_399_, 1, v_toMul_384_);
lean_ctor_set(v_reuseFailAlloc_399_, 2, v_toNPow_390_);
v___x_395_ = v_reuseFailAlloc_399_;
goto v_reusejp_394_;
}
v_reusejp_394_:
{
lean_object* v___x_397_; 
if (v_isShared_389_ == 0)
{
lean_ctor_set(v___x_388_, 1, v___x_395_);
lean_ctor_set(v___x_388_, 0, v_toAddCommMonoid_383_);
v___x_397_ = v___x_388_;
goto v_reusejp_396_;
}
else
{
lean_object* v_reuseFailAlloc_398_; 
v_reuseFailAlloc_398_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v_reuseFailAlloc_398_, 0, v_toAddCommMonoid_383_);
lean_ctor_set(v_reuseFailAlloc_398_, 1, v___x_395_);
lean_ctor_set(v_reuseFailAlloc_398_, 2, v_toNatCast_386_);
v___x_397_ = v_reuseFailAlloc_398_;
goto v_reusejp_396_;
}
v_reusejp_396_:
{
return v___x_397_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddOpposite_instSemiring(lean_object* v_R_405_, lean_object* v_inst_406_){
_start:
{
lean_object* v___x_407_; 
v___x_407_ = lp_mathlib_AddOpposite_instSemiring___redArg(v_inst_406_);
return v___x_407_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddOpposite_instNonUnitalCommSemiring___redArg(lean_object* v_inst_408_){
_start:
{
lean_object* v___x_409_; 
v___x_409_ = lp_mathlib_AddOpposite_instNonUnitalNonAssocSemiring___redArg(v_inst_408_);
return v___x_409_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddOpposite_instNonUnitalCommSemiring(lean_object* v_R_410_, lean_object* v_inst_411_){
_start:
{
lean_object* v___x_412_; 
v___x_412_ = lp_mathlib_AddOpposite_instNonUnitalNonAssocSemiring___redArg(v_inst_411_);
return v___x_412_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddOpposite_instCommSemiring___redArg(lean_object* v_inst_413_){
_start:
{
lean_object* v___x_414_; 
v___x_414_ = lp_mathlib_AddOpposite_instSemiring___redArg(v_inst_413_);
return v___x_414_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddOpposite_instCommSemiring(lean_object* v_R_415_, lean_object* v_inst_416_){
_start:
{
lean_object* v___x_417_; 
v___x_417_ = lp_mathlib_AddOpposite_instSemiring___redArg(v_inst_416_);
return v___x_417_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddOpposite_instNonUnitalNonAssocRing___redArg(lean_object* v_inst_418_){
_start:
{
lean_object* v_toAddCommGroup_419_; lean_object* v___x_420_; lean_object* v___x_421_; lean_object* v___x_422_; lean_object* v_toMul_423_; lean_object* v___x_425_; uint8_t v_isShared_426_; uint8_t v_isSharedCheck_430_; 
v_toAddCommGroup_419_ = lean_ctor_get(v_inst_418_, 0);
lean_inc_ref(v_toAddCommGroup_419_);
v___x_420_ = lp_mathlib_AddOpposite_instSubNegMonoid___redArg(v_toAddCommGroup_419_);
v___x_421_ = lp_mathlib_NonUnitalNonAssocRing_toNonUnitalNonAssocSemiring___redArg(v_inst_418_);
v___x_422_ = lp_mathlib_AddOpposite_instNonUnitalNonAssocSemiring___redArg(v___x_421_);
v_toMul_423_ = lean_ctor_get(v___x_422_, 1);
v_isSharedCheck_430_ = !lean_is_exclusive(v___x_422_);
if (v_isSharedCheck_430_ == 0)
{
lean_object* v_unused_431_; 
v_unused_431_ = lean_ctor_get(v___x_422_, 0);
lean_dec(v_unused_431_);
v___x_425_ = v___x_422_;
v_isShared_426_ = v_isSharedCheck_430_;
goto v_resetjp_424_;
}
else
{
lean_inc(v_toMul_423_);
lean_dec(v___x_422_);
v___x_425_ = lean_box(0);
v_isShared_426_ = v_isSharedCheck_430_;
goto v_resetjp_424_;
}
v_resetjp_424_:
{
lean_object* v___x_428_; 
if (v_isShared_426_ == 0)
{
lean_ctor_set(v___x_425_, 0, v___x_420_);
v___x_428_ = v___x_425_;
goto v_reusejp_427_;
}
else
{
lean_object* v_reuseFailAlloc_429_; 
v_reuseFailAlloc_429_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_429_, 0, v___x_420_);
lean_ctor_set(v_reuseFailAlloc_429_, 1, v_toMul_423_);
v___x_428_ = v_reuseFailAlloc_429_;
goto v_reusejp_427_;
}
v_reusejp_427_:
{
return v___x_428_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddOpposite_instNonUnitalNonAssocRing(lean_object* v_R_432_, lean_object* v_inst_433_){
_start:
{
lean_object* v___x_434_; 
v___x_434_ = lp_mathlib_AddOpposite_instNonUnitalNonAssocRing___redArg(v_inst_433_);
return v___x_434_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddOpposite_instNonUnitalRing___redArg(lean_object* v_inst_435_){
_start:
{
lean_object* v___x_436_; 
v___x_436_ = lp_mathlib_AddOpposite_instNonUnitalNonAssocRing___redArg(v_inst_435_);
return v___x_436_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddOpposite_instNonUnitalRing(lean_object* v_R_437_, lean_object* v_inst_438_){
_start:
{
lean_object* v___x_439_; 
v___x_439_ = lp_mathlib_AddOpposite_instNonUnitalNonAssocRing___redArg(v_inst_438_);
return v___x_439_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddOpposite_instNonAssocRing___redArg(lean_object* v_inst_440_){
_start:
{
lean_object* v_toNonUnitalNonAssocRing_441_; lean_object* v___x_442_; lean_object* v___x_443_; lean_object* v___x_444_; lean_object* v___x_445_; lean_object* v___x_446_; lean_object* v_toOne_447_; lean_object* v_toNatCast_448_; lean_object* v_toIntCast_449_; lean_object* v___x_451_; uint8_t v_isShared_452_; uint8_t v_isSharedCheck_456_; 
v_toNonUnitalNonAssocRing_441_ = lean_ctor_get(v_inst_440_, 0);
lean_inc_ref(v_toNonUnitalNonAssocRing_441_);
v___x_442_ = lp_mathlib_AddOpposite_instNonUnitalNonAssocRing___redArg(v_toNonUnitalNonAssocRing_441_);
lean_inc_ref(v_inst_440_);
v___x_443_ = lp_mathlib_NonAssocRing_toNonAssocSemiring___redArg(v_inst_440_);
v___x_444_ = lp_mathlib_AddOpposite_instNonAssocSemiring___redArg(v___x_443_);
v___x_445_ = lp_mathlib_NonAssocRing_toAddCommGroupWithOne___redArg(v_inst_440_);
v___x_446_ = lp_mathlib_AddOpposite_instAddCommGroupWithOne___redArg(v___x_445_);
v_toOne_447_ = lean_ctor_get(v___x_444_, 1);
lean_inc(v_toOne_447_);
v_toNatCast_448_ = lean_ctor_get(v___x_444_, 2);
lean_inc(v_toNatCast_448_);
lean_dec_ref(v___x_444_);
v_toIntCast_449_ = lean_ctor_get(v___x_446_, 1);
v_isSharedCheck_456_ = !lean_is_exclusive(v___x_446_);
if (v_isSharedCheck_456_ == 0)
{
lean_object* v_unused_457_; lean_object* v_unused_458_; lean_object* v_unused_459_; 
v_unused_457_ = lean_ctor_get(v___x_446_, 3);
lean_dec(v_unused_457_);
v_unused_458_ = lean_ctor_get(v___x_446_, 2);
lean_dec(v_unused_458_);
v_unused_459_ = lean_ctor_get(v___x_446_, 0);
lean_dec(v_unused_459_);
v___x_451_ = v___x_446_;
v_isShared_452_ = v_isSharedCheck_456_;
goto v_resetjp_450_;
}
else
{
lean_inc(v_toIntCast_449_);
lean_dec(v___x_446_);
v___x_451_ = lean_box(0);
v_isShared_452_ = v_isSharedCheck_456_;
goto v_resetjp_450_;
}
v_resetjp_450_:
{
lean_object* v___x_454_; 
if (v_isShared_452_ == 0)
{
lean_ctor_set(v___x_451_, 3, v_toIntCast_449_);
lean_ctor_set(v___x_451_, 2, v_toNatCast_448_);
lean_ctor_set(v___x_451_, 1, v_toOne_447_);
lean_ctor_set(v___x_451_, 0, v___x_442_);
v___x_454_ = v___x_451_;
goto v_reusejp_453_;
}
else
{
lean_object* v_reuseFailAlloc_455_; 
v_reuseFailAlloc_455_ = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(v_reuseFailAlloc_455_, 0, v___x_442_);
lean_ctor_set(v_reuseFailAlloc_455_, 1, v_toOne_447_);
lean_ctor_set(v_reuseFailAlloc_455_, 2, v_toNatCast_448_);
lean_ctor_set(v_reuseFailAlloc_455_, 3, v_toIntCast_449_);
v___x_454_ = v_reuseFailAlloc_455_;
goto v_reusejp_453_;
}
v_reusejp_453_:
{
return v___x_454_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddOpposite_instNonAssocRing(lean_object* v_R_460_, lean_object* v_inst_461_){
_start:
{
lean_object* v___x_462_; 
v___x_462_ = lp_mathlib_AddOpposite_instNonAssocRing___redArg(v_inst_461_);
return v___x_462_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddOpposite_instRing___redArg(lean_object* v_inst_463_){
_start:
{
lean_object* v_toSemiring_464_; lean_object* v___x_465_; lean_object* v___x_466_; lean_object* v___x_468_; uint8_t v_isShared_469_; uint8_t v_isSharedCheck_480_; 
v_toSemiring_464_ = lean_ctor_get(v_inst_463_, 0);
lean_inc_ref(v_toSemiring_464_);
v___x_465_ = lp_mathlib_AddOpposite_instSemiring___redArg(v_toSemiring_464_);
v___x_466_ = lp_mathlib_Ring_toNonAssocRing___redArg(v_inst_463_);
v_isSharedCheck_480_ = !lean_is_exclusive(v_inst_463_);
if (v_isSharedCheck_480_ == 0)
{
lean_object* v_unused_481_; lean_object* v_unused_482_; lean_object* v_unused_483_; lean_object* v_unused_484_; lean_object* v_unused_485_; 
v_unused_481_ = lean_ctor_get(v_inst_463_, 4);
lean_dec(v_unused_481_);
v_unused_482_ = lean_ctor_get(v_inst_463_, 3);
lean_dec(v_unused_482_);
v_unused_483_ = lean_ctor_get(v_inst_463_, 2);
lean_dec(v_unused_483_);
v_unused_484_ = lean_ctor_get(v_inst_463_, 1);
lean_dec(v_unused_484_);
v_unused_485_ = lean_ctor_get(v_inst_463_, 0);
lean_dec(v_unused_485_);
v___x_468_ = v_inst_463_;
v_isShared_469_ = v_isSharedCheck_480_;
goto v_resetjp_467_;
}
else
{
lean_dec(v_inst_463_);
v___x_468_ = lean_box(0);
v_isShared_469_ = v_isSharedCheck_480_;
goto v_resetjp_467_;
}
v_resetjp_467_:
{
lean_object* v___x_470_; lean_object* v___x_471_; lean_object* v_toAddCommGroup_472_; lean_object* v_toIntCast_473_; lean_object* v_toNeg_474_; lean_object* v_toSub_475_; lean_object* v_toZSMul_476_; lean_object* v___x_478_; 
v___x_470_ = lp_mathlib_NonAssocRing_toAddCommGroupWithOne___redArg(v___x_466_);
v___x_471_ = lp_mathlib_AddOpposite_instAddCommGroupWithOne___redArg(v___x_470_);
v_toAddCommGroup_472_ = lean_ctor_get(v___x_471_, 0);
lean_inc_ref(v_toAddCommGroup_472_);
v_toIntCast_473_ = lean_ctor_get(v___x_471_, 1);
lean_inc(v_toIntCast_473_);
lean_dec_ref(v___x_471_);
v_toNeg_474_ = lean_ctor_get(v_toAddCommGroup_472_, 1);
lean_inc(v_toNeg_474_);
v_toSub_475_ = lean_ctor_get(v_toAddCommGroup_472_, 2);
lean_inc(v_toSub_475_);
v_toZSMul_476_ = lean_ctor_get(v_toAddCommGroup_472_, 3);
lean_inc(v_toZSMul_476_);
lean_dec_ref(v_toAddCommGroup_472_);
if (v_isShared_469_ == 0)
{
lean_ctor_set(v___x_468_, 4, v_toIntCast_473_);
lean_ctor_set(v___x_468_, 3, v_toZSMul_476_);
lean_ctor_set(v___x_468_, 2, v_toSub_475_);
lean_ctor_set(v___x_468_, 1, v_toNeg_474_);
lean_ctor_set(v___x_468_, 0, v___x_465_);
v___x_478_ = v___x_468_;
goto v_reusejp_477_;
}
else
{
lean_object* v_reuseFailAlloc_479_; 
v_reuseFailAlloc_479_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_479_, 0, v___x_465_);
lean_ctor_set(v_reuseFailAlloc_479_, 1, v_toNeg_474_);
lean_ctor_set(v_reuseFailAlloc_479_, 2, v_toSub_475_);
lean_ctor_set(v_reuseFailAlloc_479_, 3, v_toZSMul_476_);
lean_ctor_set(v_reuseFailAlloc_479_, 4, v_toIntCast_473_);
v___x_478_ = v_reuseFailAlloc_479_;
goto v_reusejp_477_;
}
v_reusejp_477_:
{
return v___x_478_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddOpposite_instRing(lean_object* v_R_486_, lean_object* v_inst_487_){
_start:
{
lean_object* v___x_488_; 
v___x_488_ = lp_mathlib_AddOpposite_instRing___redArg(v_inst_487_);
return v___x_488_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddOpposite_instNonUnitalCommRing___redArg(lean_object* v_inst_489_){
_start:
{
lean_object* v___x_490_; 
v___x_490_ = lp_mathlib_AddOpposite_instNonUnitalNonAssocRing___redArg(v_inst_489_);
return v___x_490_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddOpposite_instNonUnitalCommRing(lean_object* v_R_491_, lean_object* v_inst_492_){
_start:
{
lean_object* v___x_493_; 
v___x_493_ = lp_mathlib_AddOpposite_instNonUnitalNonAssocRing___redArg(v_inst_492_);
return v___x_493_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddOpposite_instCommRing___redArg(lean_object* v_inst_494_){
_start:
{
lean_object* v___x_495_; 
v___x_495_ = lp_mathlib_AddOpposite_instRing___redArg(v_inst_494_);
return v___x_495_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddOpposite_instCommRing(lean_object* v_R_496_, lean_object* v_inst_497_){
_start:
{
lean_object* v___x_498_; 
v___x_498_ = lp_mathlib_AddOpposite_instRing___redArg(v_inst_497_);
return v___x_498_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalRingHom_toOpposite___redArg___lam__0(lean_object* v_f_499_, lean_object* v___y_500_){
_start:
{
lean_object* v___x_501_; 
v___x_501_ = lean_apply_1(v_f_499_, v___y_500_);
return v___x_501_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalRingHom_toOpposite___redArg(lean_object* v_f_503_){
_start:
{
lean_object* v___f_504_; lean_object* v___x_505_; lean_object* v___x_506_; 
v___f_504_ = lean_alloc_closure((void*)(lp_mathlib_NonUnitalRingHom_toOpposite___redArg___lam__0), 2, 1);
lean_closure_set(v___f_504_, 0, v_f_503_);
v___x_505_ = ((lean_object*)(lp_mathlib_NonUnitalRingHom_toOpposite___redArg___closed__0));
v___x_506_ = lean_alloc_closure((void*)(l_Function_comp), 6, 5);
lean_closure_set(v___x_506_, 0, lean_box(0));
lean_closure_set(v___x_506_, 1, lean_box(0));
lean_closure_set(v___x_506_, 2, lean_box(0));
lean_closure_set(v___x_506_, 3, v___x_505_);
lean_closure_set(v___x_506_, 4, v___f_504_);
return v___x_506_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalRingHom_toOpposite(lean_object* v_R_507_, lean_object* v_S_508_, lean_object* v_inst_509_, lean_object* v_inst_510_, lean_object* v_f_511_, lean_object* v_hf_512_){
_start:
{
lean_object* v___x_513_; 
v___x_513_ = lp_mathlib_NonUnitalRingHom_toOpposite___redArg(v_f_511_);
return v___x_513_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalRingHom_toOpposite___boxed(lean_object* v_R_514_, lean_object* v_S_515_, lean_object* v_inst_516_, lean_object* v_inst_517_, lean_object* v_f_518_, lean_object* v_hf_519_){
_start:
{
lean_object* v_res_520_; 
v_res_520_ = lp_mathlib_NonUnitalRingHom_toOpposite(v_R_514_, v_S_515_, v_inst_516_, v_inst_517_, v_f_518_, v_hf_519_);
lean_dec_ref(v_inst_517_);
lean_dec_ref(v_inst_516_);
return v_res_520_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalRingHom_fromOpposite___redArg(lean_object* v_f_522_){
_start:
{
lean_object* v___f_523_; lean_object* v___x_524_; lean_object* v___x_525_; 
v___f_523_ = lean_alloc_closure((void*)(lp_mathlib_NonUnitalRingHom_toOpposite___redArg___lam__0), 2, 1);
lean_closure_set(v___f_523_, 0, v_f_522_);
v___x_524_ = ((lean_object*)(lp_mathlib_NonUnitalRingHom_fromOpposite___redArg___closed__0));
v___x_525_ = lean_alloc_closure((void*)(l_Function_comp), 6, 5);
lean_closure_set(v___x_525_, 0, lean_box(0));
lean_closure_set(v___x_525_, 1, lean_box(0));
lean_closure_set(v___x_525_, 2, lean_box(0));
lean_closure_set(v___x_525_, 3, v___f_523_);
lean_closure_set(v___x_525_, 4, v___x_524_);
return v___x_525_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalRingHom_fromOpposite(lean_object* v_R_526_, lean_object* v_S_527_, lean_object* v_inst_528_, lean_object* v_inst_529_, lean_object* v_f_530_, lean_object* v_hf_531_){
_start:
{
lean_object* v___x_532_; 
v___x_532_ = lp_mathlib_NonUnitalRingHom_fromOpposite___redArg(v_f_530_);
return v___x_532_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalRingHom_fromOpposite___boxed(lean_object* v_R_533_, lean_object* v_S_534_, lean_object* v_inst_535_, lean_object* v_inst_536_, lean_object* v_f_537_, lean_object* v_hf_538_){
_start:
{
lean_object* v_res_539_; 
v_res_539_ = lp_mathlib_NonUnitalRingHom_fromOpposite(v_R_533_, v_S_534_, v_inst_535_, v_inst_536_, v_f_537_, v_hf_538_);
lean_dec_ref(v_inst_536_);
lean_dec_ref(v_inst_535_);
return v_res_539_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalRingHom_op___redArg___lam__0(lean_object* v___x_540_, lean_object* v___x_541_, lean_object* v_f_542_, lean_object* v___y_543_){
_start:
{
lean_object* v___x_544_; lean_object* v_toFun_545_; lean_object* v___x_546_; 
v___x_544_ = lp_mathlib_AddMonoidHom_mulOp(lean_box(0), lean_box(0), v___x_540_, v___x_541_);
v_toFun_545_ = lean_ctor_get(v___x_544_, 0);
lean_inc(v_toFun_545_);
lean_dec_ref(v___x_544_);
v___x_546_ = lean_apply_2(v_toFun_545_, v_f_542_, v___y_543_);
return v___x_546_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalRingHom_op___redArg___lam__0___boxed(lean_object* v___x_547_, lean_object* v___x_548_, lean_object* v_f_549_, lean_object* v___y_550_){
_start:
{
lean_object* v_res_551_; 
v_res_551_ = lp_mathlib_NonUnitalRingHom_op___redArg___lam__0(v___x_547_, v___x_548_, v_f_549_, v___y_550_);
lean_dec_ref(v___x_548_);
lean_dec_ref(v___x_547_);
return v_res_551_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalRingHom_op___redArg___lam__1(lean_object* v___x_552_, lean_object* v___x_553_, lean_object* v_f_554_, lean_object* v___y_555_){
_start:
{
lean_object* v___x_556_; lean_object* v_toFun_557_; lean_object* v___x_558_; 
v___x_556_ = lp_mathlib_AddMonoidHom_mulUnop___redArg(v___x_552_, v___x_553_);
v_toFun_557_ = lean_ctor_get(v___x_556_, 0);
lean_inc(v_toFun_557_);
lean_dec_ref(v___x_556_);
v___x_558_ = lean_apply_2(v_toFun_557_, v_f_554_, v___y_555_);
return v___x_558_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalRingHom_op___redArg___lam__1___boxed(lean_object* v___x_559_, lean_object* v___x_560_, lean_object* v_f_561_, lean_object* v___y_562_){
_start:
{
lean_object* v_res_563_; 
v_res_563_ = lp_mathlib_NonUnitalRingHom_op___redArg___lam__1(v___x_559_, v___x_560_, v_f_561_, v___y_562_);
lean_dec_ref(v___x_560_);
lean_dec_ref(v___x_559_);
return v_res_563_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalRingHom_op___redArg(lean_object* v_inst_564_, lean_object* v_inst_565_){
_start:
{
lean_object* v_toAddCommMonoid_566_; lean_object* v___x_567_; lean_object* v_toAddCommMonoid_568_; lean_object* v___x_570_; uint8_t v_isShared_571_; uint8_t v_isSharedCheck_578_; 
v_toAddCommMonoid_566_ = lean_ctor_get(v_inst_564_, 0);
v___x_567_ = lp_mathlib_AddMonoid_toAddZeroClass___redArg(v_toAddCommMonoid_566_);
v_toAddCommMonoid_568_ = lean_ctor_get(v_inst_565_, 0);
v_isSharedCheck_578_ = !lean_is_exclusive(v_inst_565_);
if (v_isSharedCheck_578_ == 0)
{
lean_object* v_unused_579_; 
v_unused_579_ = lean_ctor_get(v_inst_565_, 1);
lean_dec(v_unused_579_);
v___x_570_ = v_inst_565_;
v_isShared_571_ = v_isSharedCheck_578_;
goto v_resetjp_569_;
}
else
{
lean_inc(v_toAddCommMonoid_568_);
lean_dec(v_inst_565_);
v___x_570_ = lean_box(0);
v_isShared_571_ = v_isSharedCheck_578_;
goto v_resetjp_569_;
}
v_resetjp_569_:
{
lean_object* v___x_572_; lean_object* v___f_573_; lean_object* v___f_574_; lean_object* v___x_576_; 
v___x_572_ = lp_mathlib_AddMonoid_toAddZeroClass___redArg(v_toAddCommMonoid_568_);
lean_dec_ref(v_toAddCommMonoid_568_);
lean_inc_ref(v___x_572_);
lean_inc_ref(v___x_567_);
v___f_573_ = lean_alloc_closure((void*)(lp_mathlib_NonUnitalRingHom_op___redArg___lam__0___boxed), 4, 2);
lean_closure_set(v___f_573_, 0, v___x_567_);
lean_closure_set(v___f_573_, 1, v___x_572_);
v___f_574_ = lean_alloc_closure((void*)(lp_mathlib_NonUnitalRingHom_op___redArg___lam__1___boxed), 4, 2);
lean_closure_set(v___f_574_, 0, v___x_567_);
lean_closure_set(v___f_574_, 1, v___x_572_);
if (v_isShared_571_ == 0)
{
lean_ctor_set(v___x_570_, 1, v___f_574_);
lean_ctor_set(v___x_570_, 0, v___f_573_);
v___x_576_ = v___x_570_;
goto v_reusejp_575_;
}
else
{
lean_object* v_reuseFailAlloc_577_; 
v_reuseFailAlloc_577_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_577_, 0, v___f_573_);
lean_ctor_set(v_reuseFailAlloc_577_, 1, v___f_574_);
v___x_576_ = v_reuseFailAlloc_577_;
goto v_reusejp_575_;
}
v_reusejp_575_:
{
return v___x_576_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalRingHom_op___redArg___boxed(lean_object* v_inst_580_, lean_object* v_inst_581_){
_start:
{
lean_object* v_res_582_; 
v_res_582_ = lp_mathlib_NonUnitalRingHom_op___redArg(v_inst_580_, v_inst_581_);
lean_dec_ref(v_inst_580_);
return v_res_582_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalRingHom_op(lean_object* v_R_583_, lean_object* v_S_584_, lean_object* v_inst_585_, lean_object* v_inst_586_){
_start:
{
lean_object* v___x_587_; 
v___x_587_ = lp_mathlib_NonUnitalRingHom_op___redArg(v_inst_585_, v_inst_586_);
return v___x_587_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalRingHom_op___boxed(lean_object* v_R_588_, lean_object* v_S_589_, lean_object* v_inst_590_, lean_object* v_inst_591_){
_start:
{
lean_object* v_res_592_; 
v_res_592_ = lp_mathlib_NonUnitalRingHom_op(v_R_588_, v_S_589_, v_inst_590_, v_inst_591_);
lean_dec_ref(v_inst_590_);
return v_res_592_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalRingHom_unop___redArg(lean_object* v_inst_593_, lean_object* v_inst_594_){
_start:
{
lean_object* v___x_595_; lean_object* v___x_596_; 
v___x_595_ = lp_mathlib_NonUnitalRingHom_op___redArg(v_inst_593_, v_inst_594_);
v___x_596_ = lp_mathlib_Equiv_symm___redArg(v___x_595_);
return v___x_596_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalRingHom_unop___redArg___boxed(lean_object* v_inst_597_, lean_object* v_inst_598_){
_start:
{
lean_object* v_res_599_; 
v_res_599_ = lp_mathlib_NonUnitalRingHom_unop___redArg(v_inst_597_, v_inst_598_);
lean_dec_ref(v_inst_597_);
return v_res_599_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalRingHom_unop(lean_object* v_R_600_, lean_object* v_S_601_, lean_object* v_inst_602_, lean_object* v_inst_603_){
_start:
{
lean_object* v___x_604_; 
v___x_604_ = lp_mathlib_NonUnitalRingHom_unop___redArg(v_inst_602_, v_inst_603_);
return v___x_604_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalRingHom_unop___boxed(lean_object* v_R_605_, lean_object* v_S_606_, lean_object* v_inst_607_, lean_object* v_inst_608_){
_start:
{
lean_object* v_res_609_; 
v_res_609_ = lp_mathlib_NonUnitalRingHom_unop(v_R_605_, v_S_606_, v_inst_607_, v_inst_608_);
lean_dec_ref(v_inst_607_);
return v_res_609_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RingHom_toOpposite___redArg(lean_object* v_f_610_){
_start:
{
lean_object* v___f_611_; lean_object* v___x_612_; lean_object* v___x_613_; 
v___f_611_ = lean_alloc_closure((void*)(lp_mathlib_NonUnitalRingHom_toOpposite___redArg___lam__0), 2, 1);
lean_closure_set(v___f_611_, 0, v_f_610_);
v___x_612_ = ((lean_object*)(lp_mathlib_NonUnitalRingHom_toOpposite___redArg___closed__0));
v___x_613_ = lean_alloc_closure((void*)(l_Function_comp), 6, 5);
lean_closure_set(v___x_613_, 0, lean_box(0));
lean_closure_set(v___x_613_, 1, lean_box(0));
lean_closure_set(v___x_613_, 2, lean_box(0));
lean_closure_set(v___x_613_, 3, v___x_612_);
lean_closure_set(v___x_613_, 4, v___f_611_);
return v___x_613_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RingHom_toOpposite(lean_object* v_R_614_, lean_object* v_S_615_, lean_object* v_inst_616_, lean_object* v_inst_617_, lean_object* v_f_618_, lean_object* v_hf_619_){
_start:
{
lean_object* v___x_620_; 
v___x_620_ = lp_mathlib_RingHom_toOpposite___redArg(v_f_618_);
return v___x_620_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RingHom_toOpposite___boxed(lean_object* v_R_621_, lean_object* v_S_622_, lean_object* v_inst_623_, lean_object* v_inst_624_, lean_object* v_f_625_, lean_object* v_hf_626_){
_start:
{
lean_object* v_res_627_; 
v_res_627_ = lp_mathlib_RingHom_toOpposite(v_R_621_, v_S_622_, v_inst_623_, v_inst_624_, v_f_625_, v_hf_626_);
lean_dec_ref(v_inst_624_);
lean_dec_ref(v_inst_623_);
return v_res_627_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RingHom_fromOpposite___redArg(lean_object* v_f_628_){
_start:
{
lean_object* v___f_629_; lean_object* v___x_630_; lean_object* v___x_631_; 
v___f_629_ = lean_alloc_closure((void*)(lp_mathlib_NonUnitalRingHom_toOpposite___redArg___lam__0), 2, 1);
lean_closure_set(v___f_629_, 0, v_f_628_);
v___x_630_ = ((lean_object*)(lp_mathlib_NonUnitalRingHom_fromOpposite___redArg___closed__0));
v___x_631_ = lean_alloc_closure((void*)(l_Function_comp), 6, 5);
lean_closure_set(v___x_631_, 0, lean_box(0));
lean_closure_set(v___x_631_, 1, lean_box(0));
lean_closure_set(v___x_631_, 2, lean_box(0));
lean_closure_set(v___x_631_, 3, v___f_629_);
lean_closure_set(v___x_631_, 4, v___x_630_);
return v___x_631_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RingHom_fromOpposite(lean_object* v_R_632_, lean_object* v_S_633_, lean_object* v_inst_634_, lean_object* v_inst_635_, lean_object* v_f_636_, lean_object* v_hf_637_){
_start:
{
lean_object* v___x_638_; 
v___x_638_ = lp_mathlib_RingHom_fromOpposite___redArg(v_f_636_);
return v___x_638_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RingHom_fromOpposite___boxed(lean_object* v_R_639_, lean_object* v_S_640_, lean_object* v_inst_641_, lean_object* v_inst_642_, lean_object* v_f_643_, lean_object* v_hf_644_){
_start:
{
lean_object* v_res_645_; 
v_res_645_ = lp_mathlib_RingHom_fromOpposite(v_R_639_, v_S_640_, v_inst_641_, v_inst_642_, v_f_643_, v_hf_644_);
lean_dec_ref(v_inst_642_);
lean_dec_ref(v_inst_641_);
return v_res_645_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RingHom_op___redArg(lean_object* v_inst_646_, lean_object* v_inst_647_){
_start:
{
lean_object* v___x_648_; lean_object* v_toAddMonoid_649_; lean_object* v___x_650_; lean_object* v___x_651_; lean_object* v_toAddMonoid_652_; lean_object* v___x_653_; lean_object* v___f_654_; lean_object* v___f_655_; lean_object* v___x_656_; 
v___x_648_ = lp_mathlib_NonAssocSemiring_toAddCommMonoidWithOne___redArg(v_inst_646_);
v_toAddMonoid_649_ = lean_ctor_get(v___x_648_, 1);
lean_inc_ref(v_toAddMonoid_649_);
lean_dec_ref(v___x_648_);
v___x_650_ = lp_mathlib_AddMonoid_toAddZeroClass___redArg(v_toAddMonoid_649_);
lean_dec_ref(v_toAddMonoid_649_);
v___x_651_ = lp_mathlib_NonAssocSemiring_toAddCommMonoidWithOne___redArg(v_inst_647_);
v_toAddMonoid_652_ = lean_ctor_get(v___x_651_, 1);
lean_inc_ref(v_toAddMonoid_652_);
lean_dec_ref(v___x_651_);
v___x_653_ = lp_mathlib_AddMonoid_toAddZeroClass___redArg(v_toAddMonoid_652_);
lean_dec_ref(v_toAddMonoid_652_);
lean_inc_ref(v___x_653_);
lean_inc_ref(v___x_650_);
v___f_654_ = lean_alloc_closure((void*)(lp_mathlib_NonUnitalRingHom_op___redArg___lam__1___boxed), 4, 2);
lean_closure_set(v___f_654_, 0, v___x_650_);
lean_closure_set(v___f_654_, 1, v___x_653_);
v___f_655_ = lean_alloc_closure((void*)(lp_mathlib_NonUnitalRingHom_op___redArg___lam__0___boxed), 4, 2);
lean_closure_set(v___f_655_, 0, v___x_650_);
lean_closure_set(v___f_655_, 1, v___x_653_);
v___x_656_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_656_, 0, v___f_655_);
lean_ctor_set(v___x_656_, 1, v___f_654_);
return v___x_656_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RingHom_op(lean_object* v_R_657_, lean_object* v_S_658_, lean_object* v_inst_659_, lean_object* v_inst_660_){
_start:
{
lean_object* v___x_661_; 
v___x_661_ = lp_mathlib_RingHom_op___redArg(v_inst_659_, v_inst_660_);
return v___x_661_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RingHom_unop___redArg(lean_object* v_inst_662_, lean_object* v_inst_663_){
_start:
{
lean_object* v___x_664_; lean_object* v___x_665_; 
v___x_664_ = lp_mathlib_RingHom_op___redArg(v_inst_662_, v_inst_663_);
v___x_665_ = lp_mathlib_Equiv_symm___redArg(v___x_664_);
return v___x_665_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RingHom_unop(lean_object* v_R_666_, lean_object* v_S_667_, lean_object* v_inst_668_, lean_object* v_inst_669_){
_start:
{
lean_object* v___x_670_; 
v___x_670_ = lp_mathlib_RingHom_unop___redArg(v_inst_668_, v_inst_669_);
return v___x_670_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Group_Equiv_Opposite(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_GroupWithZero_Opposite(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Ring_Hom_Defs(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Data_Int_Cast_Basic(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Ring_Opposite(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Group_Equiv_Opposite(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_GroupWithZero_Opposite(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Ring_Hom_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Data_Int_Cast_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_Algebra_Ring_Opposite(uint8_t builtin) {
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
lean_object* initialize_mathlib_Mathlib_Algebra_Group_Equiv_Opposite(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_GroupWithZero_Opposite(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Ring_Hom_Defs(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Data_Int_Cast_Basic(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Algebra_Ring_Opposite(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Group_Equiv_Opposite(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_GroupWithZero_Opposite(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Ring_Hom_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_Int_Cast_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Ring_Opposite(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_Algebra_Ring_Opposite(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_Algebra_Ring_Opposite(builtin);
}
#ifdef __cplusplus
}
#endif
