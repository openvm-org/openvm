// Lean compiler output
// Module: Mathlib.Algebra.Ring.Pi
// Imports: public import Init public meta import Init public import Mathlib.Algebra.Group.Pi.Lemmas public import Mathlib.Algebra.GroupWithZero.Pi public import Mathlib.Algebra.Ring.CompTypeclasses public import Mathlib.Algebra.Ring.Hom.Defs
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
lean_object* lp_mathlib_Pi_involutiveNeg___redArg(lean_object*);
lean_object* lp_mathlib_Pi_addCommMonoid___redArg(lean_object*);
lean_object* lp_mathlib_NonUnitalNonAssocSemiring_toDistrib___redArg(lean_object*);
lean_object* lp_mathlib_Pi_instMul___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_NonUnitalNonAssocRing_toNonUnitalNonAssocSemiring___redArg(lean_object*);
lean_object* lp_mathlib_Pi_addCommGroup___redArg(lean_object*);
lean_object* lp_mathlib_Pi_evalMulHom___redArg___lam__0(lean_object*, lean_object*);
lean_object* lp_mathlib_Semiring_toNonUnitalSemiring___redArg(lean_object*);
lean_object* lp_mathlib_Semiring_toNonAssocSemiring___redArg(lean_object*);
lean_object* lp_mathlib_NonAssocSemiring_toMulZeroOneClass___redArg(lean_object*);
lean_object* lp_mathlib_Pi_mulZeroOneClass___redArg(lean_object*);
lean_object* lp_mathlib_NonAssocSemiring_toAddCommMonoidWithOne___redArg(lean_object*);
lean_object* lp_mathlib_Semiring_toMonoidWithZero___redArg(lean_object*);
lean_object* lp_mathlib_Pi_monoidWithZero___redArg(lean_object*);
lean_object* lp_mathlib_NonAssocRing_toNonAssocSemiring___redArg(lean_object*);
lean_object* lp_mathlib_Pi_addMonoid___redArg(lean_object*);
lean_object* lp_mathlib_Pi_instOne___redArg___lam__0(lean_object*, lean_object*);
lean_object* lp_mathlib_NonAssocRing_toAddCommGroupWithOne___redArg(lean_object*);
lean_object* lp_mathlib_AddCommGroupWithOne_toAddGroupWithOne___redArg(lean_object*);
lean_object* lp_mathlib_AddGroupWithOne_toAddGroup___redArg(lean_object*);
lean_object* lp_mathlib_Pi_addGroup___redArg(lean_object*);
lean_object* l_Function_const___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Ring_toAddCommGroup___redArg(lean_object*);
lean_object* lp_mathlib_Ring_toAddGroupWithOne___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Pi_distrib___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Pi_distrib___redArg___lam__1(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Pi_distrib___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Pi_distrib(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Pi_hasDistribNeg___redArg___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Pi_hasDistribNeg___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Pi_hasDistribNeg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Pi_hasDistribNeg___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Pi_addMonoidWithOne___redArg___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Pi_addMonoidWithOne___redArg___lam__1(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Pi_addMonoidWithOne___redArg___lam__2(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Pi_addMonoidWithOne___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Pi_addMonoidWithOne(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Pi_addGroupWithOne___redArg___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Pi_addGroupWithOne___redArg___lam__1(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Pi_addGroupWithOne___redArg___lam__2(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Pi_addGroupWithOne___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Pi_addGroupWithOne(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Pi_nonUnitalNonAssocSemiring___redArg___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Pi_nonUnitalNonAssocSemiring___redArg___lam__1(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Pi_nonUnitalNonAssocSemiring___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Pi_nonUnitalNonAssocSemiring(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Pi_nonUnitalSemiring___redArg___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Pi_nonUnitalSemiring___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Pi_nonUnitalSemiring(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Pi_nonAssocSemiring___redArg___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Pi_nonAssocSemiring___redArg___lam__1(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Pi_nonAssocSemiring___redArg___lam__2(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Pi_nonAssocSemiring___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Pi_nonAssocSemiring(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Pi_semiring___redArg___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Pi_semiring___redArg___lam__1(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Pi_semiring___redArg___lam__2(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Pi_semiring___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Pi_semiring(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Pi_nonUnitalCommSemiring___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Pi_nonUnitalCommSemiring(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Pi_commSemiring___redArg___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Pi_commSemiring___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Pi_commSemiring(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Pi_nonUnitalNonAssocRing___redArg___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Pi_nonUnitalNonAssocRing___redArg___lam__1(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Pi_nonUnitalNonAssocRing___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Pi_nonUnitalNonAssocRing(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Pi_nonUnitalRing___redArg___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Pi_nonUnitalRing___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Pi_nonUnitalRing(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Pi_nonAssocRing___redArg___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Pi_nonAssocRing___redArg___lam__1(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Pi_nonAssocRing___redArg___lam__2(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Pi_nonAssocRing___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Pi_nonAssocRing(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Pi_ring___redArg___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Pi_ring___redArg___lam__1(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Pi_ring___redArg___lam__2(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Pi_ring___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Pi_ring(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Pi_nonUnitalCommRing___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Pi_nonUnitalCommRing(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Pi_commRing___redArg___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Pi_commRing___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Pi_commRing(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalRingHom_pi___redArg___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalRingHom_pi___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalRingHom_pi(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalRingHom_pi___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Pi_nonUnitalRingHom___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Pi_nonUnitalRingHom(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Pi_nonUnitalRingHom___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Pi_evalNonUnitalRingHom___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Pi_evalNonUnitalRingHom(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Pi_evalNonUnitalRingHom___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_Pi_constNonUnitalRingHom___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*2, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Function_const___boxed, .m_arity = 4, .m_num_fixed = 2, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)(((size_t)(0) << 1) | 1))} };
static const lean_object* lp_mathlib_Pi_constNonUnitalRingHom___closed__0 = (const lean_object*)&lp_mathlib_Pi_constNonUnitalRingHom___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_Pi_constNonUnitalRingHom(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Pi_constNonUnitalRingHom___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalRingHom_compLeft___redArg___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalRingHom_compLeft___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalRingHom_compLeft(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalRingHom_compLeft___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RingHom_pi___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RingHom_pi(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RingHom_pi___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Pi_ringHom___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Pi_ringHom(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Pi_ringHom___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Pi_evalRingHom___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Pi_evalRingHom(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Pi_evalRingHom___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Pi_constRingHom(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Pi_constRingHom___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RingHom_compLeft___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RingHom_compLeft(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RingHom_compLeft___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Pi_distrib___redArg___lam__0(lean_object* v_inst_1_, lean_object* v_i_2_, lean_object* v___y_3_, lean_object* v___y_4_){
_start:
{
lean_object* v___x_5_; lean_object* v_toMul_6_; lean_object* v___x_7_; 
v___x_5_ = lean_apply_1(v_inst_1_, v_i_2_);
v_toMul_6_ = lean_ctor_get(v___x_5_, 0);
lean_inc(v_toMul_6_);
lean_dec_ref(v___x_5_);
v___x_7_ = lean_apply_2(v_toMul_6_, v___y_3_, v___y_4_);
return v___x_7_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Pi_distrib___redArg___lam__1(lean_object* v_inst_8_, lean_object* v_i_9_, lean_object* v___y_10_, lean_object* v___y_11_){
_start:
{
lean_object* v___x_12_; lean_object* v_toAdd_13_; lean_object* v___x_14_; 
v___x_12_ = lean_apply_1(v_inst_8_, v_i_9_);
v_toAdd_13_ = lean_ctor_get(v___x_12_, 1);
lean_inc(v_toAdd_13_);
lean_dec_ref(v___x_12_);
v___x_14_ = lean_apply_2(v_toAdd_13_, v___y_10_, v___y_11_);
return v___x_14_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Pi_distrib___redArg(lean_object* v_inst_15_){
_start:
{
lean_object* v___f_16_; lean_object* v___f_17_; lean_object* v___f_18_; lean_object* v___f_19_; lean_object* v___x_20_; 
lean_inc_ref(v_inst_15_);
v___f_16_ = lean_alloc_closure((void*)(lp_mathlib_Pi_distrib___redArg___lam__0), 4, 1);
lean_closure_set(v___f_16_, 0, v_inst_15_);
v___f_17_ = lean_alloc_closure((void*)(lp_mathlib_Pi_distrib___redArg___lam__1), 4, 1);
lean_closure_set(v___f_17_, 0, v_inst_15_);
v___f_18_ = lean_alloc_closure((void*)(lp_mathlib_Pi_instMul___redArg___lam__0), 4, 1);
lean_closure_set(v___f_18_, 0, v___f_16_);
v___f_19_ = lean_alloc_closure((void*)(lp_mathlib_Pi_instMul___redArg___lam__0), 4, 1);
lean_closure_set(v___f_19_, 0, v___f_17_);
v___x_20_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_20_, 0, v___f_18_);
lean_ctor_set(v___x_20_, 1, v___f_19_);
return v___x_20_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Pi_distrib(lean_object* v_I_21_, lean_object* v_f_22_, lean_object* v_inst_23_){
_start:
{
lean_object* v___x_24_; 
v___x_24_ = lp_mathlib_Pi_distrib___redArg(v_inst_23_);
return v___x_24_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Pi_hasDistribNeg___redArg___lam__0(lean_object* v_inst_25_, lean_object* v_i_26_, lean_object* v___y_27_){
_start:
{
lean_object* v___x_28_; 
v___x_28_ = lean_apply_2(v_inst_25_, v_i_26_, v___y_27_);
return v___x_28_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Pi_hasDistribNeg___redArg(lean_object* v_inst_29_){
_start:
{
lean_object* v___f_30_; lean_object* v___x_31_; 
v___f_30_ = lean_alloc_closure((void*)(lp_mathlib_Pi_hasDistribNeg___redArg___lam__0), 3, 1);
lean_closure_set(v___f_30_, 0, v_inst_29_);
v___x_31_ = lp_mathlib_Pi_involutiveNeg___redArg(v___f_30_);
return v___x_31_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Pi_hasDistribNeg(lean_object* v_I_32_, lean_object* v_f_33_, lean_object* v_inst_34_, lean_object* v_inst_35_){
_start:
{
lean_object* v___x_36_; 
v___x_36_ = lp_mathlib_Pi_hasDistribNeg___redArg(v_inst_35_);
return v___x_36_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Pi_hasDistribNeg___boxed(lean_object* v_I_37_, lean_object* v_f_38_, lean_object* v_inst_39_, lean_object* v_inst_40_){
_start:
{
lean_object* v_res_41_; 
v_res_41_ = lp_mathlib_Pi_hasDistribNeg(v_I_37_, v_f_38_, v_inst_39_, v_inst_40_);
lean_dec(v_inst_39_);
return v_res_41_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Pi_addMonoidWithOne___redArg___lam__0(lean_object* v_inst_42_, lean_object* v_n_43_, lean_object* v_x_44_){
_start:
{
lean_object* v___x_45_; lean_object* v_toNatCast_46_; lean_object* v___x_47_; 
v___x_45_ = lean_apply_1(v_inst_42_, v_x_44_);
v_toNatCast_46_ = lean_ctor_get(v___x_45_, 0);
lean_inc(v_toNatCast_46_);
lean_dec_ref(v___x_45_);
v___x_47_ = lean_apply_1(v_toNatCast_46_, v_n_43_);
return v___x_47_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Pi_addMonoidWithOne___redArg___lam__1(lean_object* v_inst_48_, lean_object* v_i_49_){
_start:
{
lean_object* v___x_50_; lean_object* v_toAddMonoid_51_; 
v___x_50_ = lean_apply_1(v_inst_48_, v_i_49_);
v_toAddMonoid_51_ = lean_ctor_get(v___x_50_, 1);
lean_inc_ref(v_toAddMonoid_51_);
lean_dec_ref(v___x_50_);
return v_toAddMonoid_51_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Pi_addMonoidWithOne___redArg___lam__2(lean_object* v_inst_52_, lean_object* v_i_53_){
_start:
{
lean_object* v___x_54_; lean_object* v_toOne_55_; 
v___x_54_ = lean_apply_1(v_inst_52_, v_i_53_);
v_toOne_55_ = lean_ctor_get(v___x_54_, 2);
lean_inc(v_toOne_55_);
lean_dec_ref(v___x_54_);
return v_toOne_55_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Pi_addMonoidWithOne___redArg(lean_object* v_inst_56_){
_start:
{
lean_object* v___f_57_; lean_object* v___f_58_; lean_object* v___f_59_; lean_object* v___x_60_; lean_object* v___f_61_; lean_object* v___x_62_; 
lean_inc_ref_n(v_inst_56_, 2);
v___f_57_ = lean_alloc_closure((void*)(lp_mathlib_Pi_addMonoidWithOne___redArg___lam__0), 3, 1);
lean_closure_set(v___f_57_, 0, v_inst_56_);
v___f_58_ = lean_alloc_closure((void*)(lp_mathlib_Pi_addMonoidWithOne___redArg___lam__1), 2, 1);
lean_closure_set(v___f_58_, 0, v_inst_56_);
v___f_59_ = lean_alloc_closure((void*)(lp_mathlib_Pi_addMonoidWithOne___redArg___lam__2), 2, 1);
lean_closure_set(v___f_59_, 0, v_inst_56_);
v___x_60_ = lp_mathlib_Pi_addMonoid___redArg(v___f_58_);
v___f_61_ = lean_alloc_closure((void*)(lp_mathlib_Pi_instOne___redArg___lam__0), 2, 1);
lean_closure_set(v___f_61_, 0, v___f_59_);
v___x_62_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_62_, 0, v___f_57_);
lean_ctor_set(v___x_62_, 1, v___x_60_);
lean_ctor_set(v___x_62_, 2, v___f_61_);
return v___x_62_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Pi_addMonoidWithOne(lean_object* v_I_63_, lean_object* v_f_64_, lean_object* v_inst_65_){
_start:
{
lean_object* v___x_66_; 
v___x_66_ = lp_mathlib_Pi_addMonoidWithOne___redArg(v_inst_65_);
return v___x_66_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Pi_addGroupWithOne___redArg___lam__0(lean_object* v_inst_67_, lean_object* v_i_68_){
_start:
{
lean_object* v___x_69_; lean_object* v___x_70_; 
v___x_69_ = lean_apply_1(v_inst_67_, v_i_68_);
v___x_70_ = lp_mathlib_AddGroupWithOne_toAddGroup___redArg(v___x_69_);
lean_dec_ref(v___x_69_);
return v___x_70_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Pi_addGroupWithOne___redArg___lam__1(lean_object* v_inst_71_, lean_object* v_i_72_){
_start:
{
lean_object* v___x_73_; lean_object* v_toAddMonoidWithOne_74_; 
v___x_73_ = lean_apply_1(v_inst_71_, v_i_72_);
v_toAddMonoidWithOne_74_ = lean_ctor_get(v___x_73_, 1);
lean_inc_ref(v_toAddMonoidWithOne_74_);
lean_dec_ref(v___x_73_);
return v_toAddMonoidWithOne_74_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Pi_addGroupWithOne___redArg___lam__2(lean_object* v_inst_75_, lean_object* v_n_76_, lean_object* v_x_77_){
_start:
{
lean_object* v___x_78_; lean_object* v_toIntCast_79_; lean_object* v___x_80_; 
v___x_78_ = lean_apply_1(v_inst_75_, v_x_77_);
v_toIntCast_79_ = lean_ctor_get(v___x_78_, 0);
lean_inc(v_toIntCast_79_);
lean_dec_ref(v___x_78_);
v___x_80_ = lean_apply_1(v_toIntCast_79_, v_n_76_);
return v___x_80_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Pi_addGroupWithOne___redArg(lean_object* v_inst_81_){
_start:
{
lean_object* v___f_82_; lean_object* v___x_83_; lean_object* v_toAddMonoid_84_; lean_object* v_toNeg_85_; lean_object* v_toSub_86_; lean_object* v_toZSMul_87_; lean_object* v___f_88_; lean_object* v___f_89_; lean_object* v___f_90_; lean_object* v___f_91_; lean_object* v___f_92_; lean_object* v___x_93_; lean_object* v___x_94_; 
lean_inc_ref_n(v_inst_81_, 2);
v___f_82_ = lean_alloc_closure((void*)(lp_mathlib_Pi_addGroupWithOne___redArg___lam__0), 2, 1);
lean_closure_set(v___f_82_, 0, v_inst_81_);
v___x_83_ = lp_mathlib_Pi_addGroup___redArg(v___f_82_);
v_toAddMonoid_84_ = lean_ctor_get(v___x_83_, 0);
lean_inc_ref(v_toAddMonoid_84_);
v_toNeg_85_ = lean_ctor_get(v___x_83_, 1);
lean_inc(v_toNeg_85_);
v_toSub_86_ = lean_ctor_get(v___x_83_, 2);
lean_inc(v_toSub_86_);
v_toZSMul_87_ = lean_ctor_get(v___x_83_, 3);
lean_inc(v_toZSMul_87_);
lean_dec_ref(v___x_83_);
v___f_88_ = lean_alloc_closure((void*)(lp_mathlib_Pi_addGroupWithOne___redArg___lam__1), 2, 1);
lean_closure_set(v___f_88_, 0, v_inst_81_);
v___f_89_ = lean_alloc_closure((void*)(lp_mathlib_Pi_addGroupWithOne___redArg___lam__2), 3, 1);
lean_closure_set(v___f_89_, 0, v_inst_81_);
lean_inc_ref(v___f_88_);
v___f_90_ = lean_alloc_closure((void*)(lp_mathlib_Pi_addMonoidWithOne___redArg___lam__0), 3, 1);
lean_closure_set(v___f_90_, 0, v___f_88_);
v___f_91_ = lean_alloc_closure((void*)(lp_mathlib_Pi_addMonoidWithOne___redArg___lam__2), 2, 1);
lean_closure_set(v___f_91_, 0, v___f_88_);
v___f_92_ = lean_alloc_closure((void*)(lp_mathlib_Pi_instOne___redArg___lam__0), 2, 1);
lean_closure_set(v___f_92_, 0, v___f_91_);
v___x_93_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_93_, 0, v___f_90_);
lean_ctor_set(v___x_93_, 1, v_toAddMonoid_84_);
lean_ctor_set(v___x_93_, 2, v___f_92_);
v___x_94_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v___x_94_, 0, v___f_89_);
lean_ctor_set(v___x_94_, 1, v___x_93_);
lean_ctor_set(v___x_94_, 2, v_toNeg_85_);
lean_ctor_set(v___x_94_, 3, v_toSub_86_);
lean_ctor_set(v___x_94_, 4, v_toZSMul_87_);
return v___x_94_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Pi_addGroupWithOne(lean_object* v_I_95_, lean_object* v_f_96_, lean_object* v_inst_97_){
_start:
{
lean_object* v___x_98_; 
v___x_98_ = lp_mathlib_Pi_addGroupWithOne___redArg(v_inst_97_);
return v___x_98_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Pi_nonUnitalNonAssocSemiring___redArg___lam__0(lean_object* v_inst_99_, lean_object* v_i_100_){
_start:
{
lean_object* v___x_101_; lean_object* v_toAddCommMonoid_102_; 
v___x_101_ = lean_apply_1(v_inst_99_, v_i_100_);
v_toAddCommMonoid_102_ = lean_ctor_get(v___x_101_, 0);
lean_inc_ref(v_toAddCommMonoid_102_);
lean_dec_ref(v___x_101_);
return v_toAddCommMonoid_102_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Pi_nonUnitalNonAssocSemiring___redArg___lam__1(lean_object* v_inst_103_, lean_object* v_i_104_){
_start:
{
lean_object* v___x_105_; lean_object* v___x_106_; 
v___x_105_ = lean_apply_1(v_inst_103_, v_i_104_);
v___x_106_ = lp_mathlib_NonUnitalNonAssocSemiring_toDistrib___redArg(v___x_105_);
return v___x_106_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Pi_nonUnitalNonAssocSemiring___redArg(lean_object* v_inst_107_){
_start:
{
lean_object* v___f_108_; lean_object* v___x_109_; lean_object* v_toZero_110_; lean_object* v_toNSMul_111_; lean_object* v___x_113_; uint8_t v_isShared_114_; uint8_t v_isSharedCheck_124_; 
lean_inc_ref(v_inst_107_);
v___f_108_ = lean_alloc_closure((void*)(lp_mathlib_Pi_nonUnitalNonAssocSemiring___redArg___lam__0), 2, 1);
lean_closure_set(v___f_108_, 0, v_inst_107_);
v___x_109_ = lp_mathlib_Pi_addCommMonoid___redArg(v___f_108_);
v_toZero_110_ = lean_ctor_get(v___x_109_, 0);
v_toNSMul_111_ = lean_ctor_get(v___x_109_, 2);
v_isSharedCheck_124_ = !lean_is_exclusive(v___x_109_);
if (v_isSharedCheck_124_ == 0)
{
lean_object* v_unused_125_; 
v_unused_125_ = lean_ctor_get(v___x_109_, 1);
lean_dec(v_unused_125_);
v___x_113_ = v___x_109_;
v_isShared_114_ = v_isSharedCheck_124_;
goto v_resetjp_112_;
}
else
{
lean_inc(v_toNSMul_111_);
lean_inc(v_toZero_110_);
lean_dec(v___x_109_);
v___x_113_ = lean_box(0);
v_isShared_114_ = v_isSharedCheck_124_;
goto v_resetjp_112_;
}
v_resetjp_112_:
{
lean_object* v___f_115_; lean_object* v___f_116_; lean_object* v___f_117_; lean_object* v___f_118_; lean_object* v___f_119_; lean_object* v___x_121_; 
v___f_115_ = lean_alloc_closure((void*)(lp_mathlib_Pi_nonUnitalNonAssocSemiring___redArg___lam__1), 2, 1);
lean_closure_set(v___f_115_, 0, v_inst_107_);
lean_inc_ref(v___f_115_);
v___f_116_ = lean_alloc_closure((void*)(lp_mathlib_Pi_distrib___redArg___lam__1), 4, 1);
lean_closure_set(v___f_116_, 0, v___f_115_);
v___f_117_ = lean_alloc_closure((void*)(lp_mathlib_Pi_instMul___redArg___lam__0), 4, 1);
lean_closure_set(v___f_117_, 0, v___f_116_);
v___f_118_ = lean_alloc_closure((void*)(lp_mathlib_Pi_distrib___redArg___lam__0), 4, 1);
lean_closure_set(v___f_118_, 0, v___f_115_);
v___f_119_ = lean_alloc_closure((void*)(lp_mathlib_Pi_instMul___redArg___lam__0), 4, 1);
lean_closure_set(v___f_119_, 0, v___f_118_);
if (v_isShared_114_ == 0)
{
lean_ctor_set(v___x_113_, 1, v___f_117_);
v___x_121_ = v___x_113_;
goto v_reusejp_120_;
}
else
{
lean_object* v_reuseFailAlloc_123_; 
v_reuseFailAlloc_123_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v_reuseFailAlloc_123_, 0, v_toZero_110_);
lean_ctor_set(v_reuseFailAlloc_123_, 1, v___f_117_);
lean_ctor_set(v_reuseFailAlloc_123_, 2, v_toNSMul_111_);
v___x_121_ = v_reuseFailAlloc_123_;
goto v_reusejp_120_;
}
v_reusejp_120_:
{
lean_object* v___x_122_; 
v___x_122_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_122_, 0, v___x_121_);
lean_ctor_set(v___x_122_, 1, v___f_119_);
return v___x_122_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Pi_nonUnitalNonAssocSemiring(lean_object* v_I_126_, lean_object* v_f_127_, lean_object* v_inst_128_){
_start:
{
lean_object* v___x_129_; 
v___x_129_ = lp_mathlib_Pi_nonUnitalNonAssocSemiring___redArg(v_inst_128_);
return v___x_129_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Pi_nonUnitalSemiring___redArg___lam__0(lean_object* v_inst_130_, lean_object* v_i_131_){
_start:
{
lean_object* v___x_132_; 
v___x_132_ = lean_apply_1(v_inst_130_, v_i_131_);
return v___x_132_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Pi_nonUnitalSemiring___redArg(lean_object* v_inst_133_){
_start:
{
lean_object* v___f_134_; lean_object* v___x_135_; 
v___f_134_ = lean_alloc_closure((void*)(lp_mathlib_Pi_nonUnitalSemiring___redArg___lam__0), 2, 1);
lean_closure_set(v___f_134_, 0, v_inst_133_);
v___x_135_ = lp_mathlib_Pi_nonUnitalNonAssocSemiring___redArg(v___f_134_);
return v___x_135_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Pi_nonUnitalSemiring(lean_object* v_I_136_, lean_object* v_f_137_, lean_object* v_inst_138_){
_start:
{
lean_object* v___x_139_; 
v___x_139_ = lp_mathlib_Pi_nonUnitalSemiring___redArg(v_inst_138_);
return v___x_139_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Pi_nonAssocSemiring___redArg___lam__0(lean_object* v_inst_140_, lean_object* v_i_141_){
_start:
{
lean_object* v___x_142_; lean_object* v_toNonUnitalNonAssocSemiring_143_; 
v___x_142_ = lean_apply_1(v_inst_140_, v_i_141_);
v_toNonUnitalNonAssocSemiring_143_ = lean_ctor_get(v___x_142_, 0);
lean_inc_ref(v_toNonUnitalNonAssocSemiring_143_);
lean_dec_ref(v___x_142_);
return v_toNonUnitalNonAssocSemiring_143_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Pi_nonAssocSemiring___redArg___lam__1(lean_object* v_inst_144_, lean_object* v_i_145_){
_start:
{
lean_object* v___x_146_; lean_object* v___x_147_; 
v___x_146_ = lean_apply_1(v_inst_144_, v_i_145_);
v___x_147_ = lp_mathlib_NonAssocSemiring_toMulZeroOneClass___redArg(v___x_146_);
return v___x_147_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Pi_nonAssocSemiring___redArg___lam__2(lean_object* v_inst_148_, lean_object* v_i_149_){
_start:
{
lean_object* v___x_150_; lean_object* v___x_151_; 
v___x_150_ = lean_apply_1(v_inst_148_, v_i_149_);
v___x_151_ = lp_mathlib_NonAssocSemiring_toAddCommMonoidWithOne___redArg(v___x_150_);
return v___x_151_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Pi_nonAssocSemiring___redArg(lean_object* v_inst_152_){
_start:
{
lean_object* v___f_153_; lean_object* v___f_154_; lean_object* v___x_155_; lean_object* v___x_156_; lean_object* v_toMulOneClass_157_; lean_object* v_toOne_158_; lean_object* v___f_159_; lean_object* v___f_160_; lean_object* v___x_161_; 
lean_inc_ref_n(v_inst_152_, 2);
v___f_153_ = lean_alloc_closure((void*)(lp_mathlib_Pi_nonAssocSemiring___redArg___lam__0), 2, 1);
lean_closure_set(v___f_153_, 0, v_inst_152_);
v___f_154_ = lean_alloc_closure((void*)(lp_mathlib_Pi_nonAssocSemiring___redArg___lam__1), 2, 1);
lean_closure_set(v___f_154_, 0, v_inst_152_);
v___x_155_ = lp_mathlib_Pi_nonUnitalNonAssocSemiring___redArg(v___f_153_);
v___x_156_ = lp_mathlib_Pi_mulZeroOneClass___redArg(v___f_154_);
v_toMulOneClass_157_ = lean_ctor_get(v___x_156_, 0);
lean_inc_ref(v_toMulOneClass_157_);
lean_dec_ref(v___x_156_);
v_toOne_158_ = lean_ctor_get(v_toMulOneClass_157_, 0);
lean_inc(v_toOne_158_);
lean_dec_ref(v_toMulOneClass_157_);
v___f_159_ = lean_alloc_closure((void*)(lp_mathlib_Pi_nonAssocSemiring___redArg___lam__2), 2, 1);
lean_closure_set(v___f_159_, 0, v_inst_152_);
v___f_160_ = lean_alloc_closure((void*)(lp_mathlib_Pi_addMonoidWithOne___redArg___lam__0), 3, 1);
lean_closure_set(v___f_160_, 0, v___f_159_);
v___x_161_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_161_, 0, v___x_155_);
lean_ctor_set(v___x_161_, 1, v_toOne_158_);
lean_ctor_set(v___x_161_, 2, v___f_160_);
return v___x_161_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Pi_nonAssocSemiring(lean_object* v_I_162_, lean_object* v_f_163_, lean_object* v_inst_164_){
_start:
{
lean_object* v___x_165_; 
v___x_165_ = lp_mathlib_Pi_nonAssocSemiring___redArg(v_inst_164_);
return v___x_165_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Pi_semiring___redArg___lam__0(lean_object* v_inst_166_, lean_object* v_i_167_){
_start:
{
lean_object* v___x_168_; lean_object* v___x_169_; 
v___x_168_ = lean_apply_1(v_inst_166_, v_i_167_);
v___x_169_ = lp_mathlib_Semiring_toNonUnitalSemiring___redArg(v___x_168_);
lean_dec_ref(v___x_168_);
return v___x_169_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Pi_semiring___redArg___lam__1(lean_object* v_inst_170_, lean_object* v_i_171_){
_start:
{
lean_object* v___x_172_; lean_object* v___x_173_; 
v___x_172_ = lean_apply_1(v_inst_170_, v_i_171_);
v___x_173_ = lp_mathlib_Semiring_toNonAssocSemiring___redArg(v___x_172_);
return v___x_173_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Pi_semiring___redArg___lam__2(lean_object* v_inst_174_, lean_object* v_i_175_){
_start:
{
lean_object* v___x_176_; lean_object* v___x_177_; 
v___x_176_ = lean_apply_1(v_inst_174_, v_i_175_);
v___x_177_ = lp_mathlib_Semiring_toMonoidWithZero___redArg(v___x_176_);
lean_dec_ref(v___x_176_);
return v___x_177_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Pi_semiring___redArg(lean_object* v_inst_178_){
_start:
{
lean_object* v___f_179_; lean_object* v___x_180_; lean_object* v_toAddCommMonoid_181_; lean_object* v_toMul_182_; lean_object* v___f_183_; lean_object* v___x_184_; lean_object* v_toOne_185_; lean_object* v_toNatCast_186_; lean_object* v___x_188_; uint8_t v_isShared_189_; uint8_t v_isSharedCheck_206_; 
lean_inc_ref_n(v_inst_178_, 2);
v___f_179_ = lean_alloc_closure((void*)(lp_mathlib_Pi_semiring___redArg___lam__0), 2, 1);
lean_closure_set(v___f_179_, 0, v_inst_178_);
v___x_180_ = lp_mathlib_Pi_nonUnitalSemiring___redArg(v___f_179_);
v_toAddCommMonoid_181_ = lean_ctor_get(v___x_180_, 0);
lean_inc_ref(v_toAddCommMonoid_181_);
v_toMul_182_ = lean_ctor_get(v___x_180_, 1);
lean_inc(v_toMul_182_);
lean_dec_ref(v___x_180_);
v___f_183_ = lean_alloc_closure((void*)(lp_mathlib_Pi_semiring___redArg___lam__1), 2, 1);
lean_closure_set(v___f_183_, 0, v_inst_178_);
v___x_184_ = lp_mathlib_Pi_nonAssocSemiring___redArg(v___f_183_);
v_toOne_185_ = lean_ctor_get(v___x_184_, 1);
v_toNatCast_186_ = lean_ctor_get(v___x_184_, 2);
v_isSharedCheck_206_ = !lean_is_exclusive(v___x_184_);
if (v_isSharedCheck_206_ == 0)
{
lean_object* v_unused_207_; 
v_unused_207_ = lean_ctor_get(v___x_184_, 0);
lean_dec(v_unused_207_);
v___x_188_ = v___x_184_;
v_isShared_189_ = v_isSharedCheck_206_;
goto v_resetjp_187_;
}
else
{
lean_inc(v_toNatCast_186_);
lean_inc(v_toOne_185_);
lean_dec(v___x_184_);
v___x_188_ = lean_box(0);
v_isShared_189_ = v_isSharedCheck_206_;
goto v_resetjp_187_;
}
v_resetjp_187_:
{
lean_object* v___f_190_; lean_object* v___x_191_; lean_object* v_toMonoid_192_; lean_object* v_toNPow_193_; lean_object* v___x_195_; uint8_t v_isShared_196_; uint8_t v_isSharedCheck_203_; 
v___f_190_ = lean_alloc_closure((void*)(lp_mathlib_Pi_semiring___redArg___lam__2), 2, 1);
lean_closure_set(v___f_190_, 0, v_inst_178_);
v___x_191_ = lp_mathlib_Pi_monoidWithZero___redArg(v___f_190_);
v_toMonoid_192_ = lean_ctor_get(v___x_191_, 0);
lean_inc_ref(v_toMonoid_192_);
lean_dec_ref(v___x_191_);
v_toNPow_193_ = lean_ctor_get(v_toMonoid_192_, 2);
v_isSharedCheck_203_ = !lean_is_exclusive(v_toMonoid_192_);
if (v_isSharedCheck_203_ == 0)
{
lean_object* v_unused_204_; lean_object* v_unused_205_; 
v_unused_204_ = lean_ctor_get(v_toMonoid_192_, 1);
lean_dec(v_unused_204_);
v_unused_205_ = lean_ctor_get(v_toMonoid_192_, 0);
lean_dec(v_unused_205_);
v___x_195_ = v_toMonoid_192_;
v_isShared_196_ = v_isSharedCheck_203_;
goto v_resetjp_194_;
}
else
{
lean_inc(v_toNPow_193_);
lean_dec(v_toMonoid_192_);
v___x_195_ = lean_box(0);
v_isShared_196_ = v_isSharedCheck_203_;
goto v_resetjp_194_;
}
v_resetjp_194_:
{
lean_object* v___x_198_; 
if (v_isShared_196_ == 0)
{
lean_ctor_set(v___x_195_, 1, v_toMul_182_);
lean_ctor_set(v___x_195_, 0, v_toOne_185_);
v___x_198_ = v___x_195_;
goto v_reusejp_197_;
}
else
{
lean_object* v_reuseFailAlloc_202_; 
v_reuseFailAlloc_202_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v_reuseFailAlloc_202_, 0, v_toOne_185_);
lean_ctor_set(v_reuseFailAlloc_202_, 1, v_toMul_182_);
lean_ctor_set(v_reuseFailAlloc_202_, 2, v_toNPow_193_);
v___x_198_ = v_reuseFailAlloc_202_;
goto v_reusejp_197_;
}
v_reusejp_197_:
{
lean_object* v___x_200_; 
if (v_isShared_189_ == 0)
{
lean_ctor_set(v___x_188_, 1, v___x_198_);
lean_ctor_set(v___x_188_, 0, v_toAddCommMonoid_181_);
v___x_200_ = v___x_188_;
goto v_reusejp_199_;
}
else
{
lean_object* v_reuseFailAlloc_201_; 
v_reuseFailAlloc_201_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v_reuseFailAlloc_201_, 0, v_toAddCommMonoid_181_);
lean_ctor_set(v_reuseFailAlloc_201_, 1, v___x_198_);
lean_ctor_set(v_reuseFailAlloc_201_, 2, v_toNatCast_186_);
v___x_200_ = v_reuseFailAlloc_201_;
goto v_reusejp_199_;
}
v_reusejp_199_:
{
return v___x_200_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Pi_semiring(lean_object* v_I_208_, lean_object* v_f_209_, lean_object* v_inst_210_){
_start:
{
lean_object* v___x_211_; 
v___x_211_ = lp_mathlib_Pi_semiring___redArg(v_inst_210_);
return v___x_211_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Pi_nonUnitalCommSemiring___redArg(lean_object* v_inst_212_){
_start:
{
lean_object* v___f_213_; lean_object* v___x_214_; 
v___f_213_ = lean_alloc_closure((void*)(lp_mathlib_Pi_nonUnitalSemiring___redArg___lam__0), 2, 1);
lean_closure_set(v___f_213_, 0, v_inst_212_);
v___x_214_ = lp_mathlib_Pi_nonUnitalSemiring___redArg(v___f_213_);
return v___x_214_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Pi_nonUnitalCommSemiring(lean_object* v_I_215_, lean_object* v_f_216_, lean_object* v_inst_217_){
_start:
{
lean_object* v___x_218_; 
v___x_218_ = lp_mathlib_Pi_nonUnitalCommSemiring___redArg(v_inst_217_);
return v___x_218_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Pi_commSemiring___redArg___lam__0(lean_object* v_inst_219_, lean_object* v_i_220_){
_start:
{
lean_object* v___x_221_; 
v___x_221_ = lean_apply_1(v_inst_219_, v_i_220_);
return v___x_221_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Pi_commSemiring___redArg(lean_object* v_inst_222_){
_start:
{
lean_object* v___f_223_; lean_object* v___x_224_; 
v___f_223_ = lean_alloc_closure((void*)(lp_mathlib_Pi_commSemiring___redArg___lam__0), 2, 1);
lean_closure_set(v___f_223_, 0, v_inst_222_);
v___x_224_ = lp_mathlib_Pi_semiring___redArg(v___f_223_);
return v___x_224_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Pi_commSemiring(lean_object* v_I_225_, lean_object* v_f_226_, lean_object* v_inst_227_){
_start:
{
lean_object* v___x_228_; 
v___x_228_ = lp_mathlib_Pi_commSemiring___redArg(v_inst_227_);
return v___x_228_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Pi_nonUnitalNonAssocRing___redArg___lam__0(lean_object* v_inst_229_, lean_object* v_i_230_){
_start:
{
lean_object* v___x_231_; lean_object* v_toAddCommGroup_232_; 
v___x_231_ = lean_apply_1(v_inst_229_, v_i_230_);
v_toAddCommGroup_232_ = lean_ctor_get(v___x_231_, 0);
lean_inc_ref(v_toAddCommGroup_232_);
lean_dec_ref(v___x_231_);
return v_toAddCommGroup_232_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Pi_nonUnitalNonAssocRing___redArg___lam__1(lean_object* v_inst_233_, lean_object* v_i_234_){
_start:
{
lean_object* v___x_235_; lean_object* v___x_236_; 
v___x_235_ = lean_apply_1(v_inst_233_, v_i_234_);
v___x_236_ = lp_mathlib_NonUnitalNonAssocRing_toNonUnitalNonAssocSemiring___redArg(v___x_235_);
return v___x_236_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Pi_nonUnitalNonAssocRing___redArg(lean_object* v_inst_237_){
_start:
{
lean_object* v___f_238_; lean_object* v___f_239_; lean_object* v___x_240_; lean_object* v___x_241_; lean_object* v_toMul_242_; lean_object* v___x_244_; uint8_t v_isShared_245_; uint8_t v_isSharedCheck_249_; 
lean_inc_ref(v_inst_237_);
v___f_238_ = lean_alloc_closure((void*)(lp_mathlib_Pi_nonUnitalNonAssocRing___redArg___lam__0), 2, 1);
lean_closure_set(v___f_238_, 0, v_inst_237_);
v___f_239_ = lean_alloc_closure((void*)(lp_mathlib_Pi_nonUnitalNonAssocRing___redArg___lam__1), 2, 1);
lean_closure_set(v___f_239_, 0, v_inst_237_);
v___x_240_ = lp_mathlib_Pi_addCommGroup___redArg(v___f_238_);
v___x_241_ = lp_mathlib_Pi_nonUnitalNonAssocSemiring___redArg(v___f_239_);
v_toMul_242_ = lean_ctor_get(v___x_241_, 1);
v_isSharedCheck_249_ = !lean_is_exclusive(v___x_241_);
if (v_isSharedCheck_249_ == 0)
{
lean_object* v_unused_250_; 
v_unused_250_ = lean_ctor_get(v___x_241_, 0);
lean_dec(v_unused_250_);
v___x_244_ = v___x_241_;
v_isShared_245_ = v_isSharedCheck_249_;
goto v_resetjp_243_;
}
else
{
lean_inc(v_toMul_242_);
lean_dec(v___x_241_);
v___x_244_ = lean_box(0);
v_isShared_245_ = v_isSharedCheck_249_;
goto v_resetjp_243_;
}
v_resetjp_243_:
{
lean_object* v___x_247_; 
if (v_isShared_245_ == 0)
{
lean_ctor_set(v___x_244_, 0, v___x_240_);
v___x_247_ = v___x_244_;
goto v_reusejp_246_;
}
else
{
lean_object* v_reuseFailAlloc_248_; 
v_reuseFailAlloc_248_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_248_, 0, v___x_240_);
lean_ctor_set(v_reuseFailAlloc_248_, 1, v_toMul_242_);
v___x_247_ = v_reuseFailAlloc_248_;
goto v_reusejp_246_;
}
v_reusejp_246_:
{
return v___x_247_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Pi_nonUnitalNonAssocRing(lean_object* v_I_251_, lean_object* v_f_252_, lean_object* v_inst_253_){
_start:
{
lean_object* v___x_254_; 
v___x_254_ = lp_mathlib_Pi_nonUnitalNonAssocRing___redArg(v_inst_253_);
return v___x_254_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Pi_nonUnitalRing___redArg___lam__0(lean_object* v_inst_255_, lean_object* v_i_256_){
_start:
{
lean_object* v___x_257_; 
v___x_257_ = lean_apply_1(v_inst_255_, v_i_256_);
return v___x_257_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Pi_nonUnitalRing___redArg(lean_object* v_inst_258_){
_start:
{
lean_object* v___f_259_; lean_object* v___x_260_; 
v___f_259_ = lean_alloc_closure((void*)(lp_mathlib_Pi_nonUnitalRing___redArg___lam__0), 2, 1);
lean_closure_set(v___f_259_, 0, v_inst_258_);
v___x_260_ = lp_mathlib_Pi_nonUnitalNonAssocRing___redArg(v___f_259_);
return v___x_260_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Pi_nonUnitalRing(lean_object* v_I_261_, lean_object* v_f_262_, lean_object* v_inst_263_){
_start:
{
lean_object* v___x_264_; 
v___x_264_ = lp_mathlib_Pi_nonUnitalRing___redArg(v_inst_263_);
return v___x_264_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Pi_nonAssocRing___redArg___lam__0(lean_object* v_inst_265_, lean_object* v_i_266_){
_start:
{
lean_object* v___x_267_; lean_object* v_toNonUnitalNonAssocRing_268_; 
v___x_267_ = lean_apply_1(v_inst_265_, v_i_266_);
v_toNonUnitalNonAssocRing_268_ = lean_ctor_get(v___x_267_, 0);
lean_inc_ref(v_toNonUnitalNonAssocRing_268_);
lean_dec_ref(v___x_267_);
return v_toNonUnitalNonAssocRing_268_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Pi_nonAssocRing___redArg___lam__1(lean_object* v_inst_269_, lean_object* v_i_270_){
_start:
{
lean_object* v___x_271_; lean_object* v___x_272_; 
v___x_271_ = lean_apply_1(v_inst_269_, v_i_270_);
v___x_272_ = lp_mathlib_NonAssocRing_toNonAssocSemiring___redArg(v___x_271_);
return v___x_272_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Pi_nonAssocRing___redArg___lam__2(lean_object* v_inst_273_, lean_object* v_i_274_){
_start:
{
lean_object* v___x_275_; lean_object* v___x_276_; lean_object* v___x_277_; 
v___x_275_ = lean_apply_1(v_inst_273_, v_i_274_);
v___x_276_ = lp_mathlib_NonAssocRing_toAddCommGroupWithOne___redArg(v___x_275_);
v___x_277_ = lp_mathlib_AddCommGroupWithOne_toAddGroupWithOne___redArg(v___x_276_);
lean_dec_ref(v___x_276_);
return v___x_277_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Pi_nonAssocRing___redArg(lean_object* v_inst_278_){
_start:
{
lean_object* v___f_279_; lean_object* v___f_280_; lean_object* v___x_281_; lean_object* v___x_282_; lean_object* v_toOne_283_; lean_object* v_toNatCast_284_; lean_object* v___f_285_; lean_object* v___x_286_; lean_object* v_toIntCast_287_; lean_object* v___x_288_; 
lean_inc_ref_n(v_inst_278_, 2);
v___f_279_ = lean_alloc_closure((void*)(lp_mathlib_Pi_nonAssocRing___redArg___lam__0), 2, 1);
lean_closure_set(v___f_279_, 0, v_inst_278_);
v___f_280_ = lean_alloc_closure((void*)(lp_mathlib_Pi_nonAssocRing___redArg___lam__1), 2, 1);
lean_closure_set(v___f_280_, 0, v_inst_278_);
v___x_281_ = lp_mathlib_Pi_nonUnitalNonAssocRing___redArg(v___f_279_);
v___x_282_ = lp_mathlib_Pi_nonAssocSemiring___redArg(v___f_280_);
v_toOne_283_ = lean_ctor_get(v___x_282_, 1);
lean_inc(v_toOne_283_);
v_toNatCast_284_ = lean_ctor_get(v___x_282_, 2);
lean_inc(v_toNatCast_284_);
lean_dec_ref(v___x_282_);
v___f_285_ = lean_alloc_closure((void*)(lp_mathlib_Pi_nonAssocRing___redArg___lam__2), 2, 1);
lean_closure_set(v___f_285_, 0, v_inst_278_);
v___x_286_ = lp_mathlib_Pi_addGroupWithOne___redArg(v___f_285_);
v_toIntCast_287_ = lean_ctor_get(v___x_286_, 0);
lean_inc(v_toIntCast_287_);
lean_dec_ref(v___x_286_);
v___x_288_ = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(v___x_288_, 0, v___x_281_);
lean_ctor_set(v___x_288_, 1, v_toOne_283_);
lean_ctor_set(v___x_288_, 2, v_toNatCast_284_);
lean_ctor_set(v___x_288_, 3, v_toIntCast_287_);
return v___x_288_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Pi_nonAssocRing(lean_object* v_I_289_, lean_object* v_f_290_, lean_object* v_inst_291_){
_start:
{
lean_object* v___x_292_; 
v___x_292_ = lp_mathlib_Pi_nonAssocRing___redArg(v_inst_291_);
return v___x_292_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Pi_ring___redArg___lam__0(lean_object* v_inst_293_, lean_object* v_i_294_){
_start:
{
lean_object* v___x_295_; lean_object* v_toSemiring_296_; 
v___x_295_ = lean_apply_1(v_inst_293_, v_i_294_);
v_toSemiring_296_ = lean_ctor_get(v___x_295_, 0);
lean_inc_ref(v_toSemiring_296_);
lean_dec_ref(v___x_295_);
return v_toSemiring_296_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Pi_ring___redArg___lam__1(lean_object* v_inst_297_, lean_object* v_i_298_){
_start:
{
lean_object* v___x_299_; lean_object* v___x_300_; 
v___x_299_ = lean_apply_1(v_inst_297_, v_i_298_);
v___x_300_ = lp_mathlib_Ring_toAddCommGroup___redArg(v___x_299_);
lean_dec_ref(v___x_299_);
return v___x_300_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Pi_ring___redArg___lam__2(lean_object* v_inst_301_, lean_object* v_i_302_){
_start:
{
lean_object* v___x_303_; lean_object* v___x_304_; 
v___x_303_ = lean_apply_1(v_inst_301_, v_i_302_);
v___x_304_ = lp_mathlib_Ring_toAddGroupWithOne___redArg(v___x_303_);
return v___x_304_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Pi_ring___redArg(lean_object* v_inst_305_){
_start:
{
lean_object* v___f_306_; lean_object* v___f_307_; lean_object* v___x_308_; lean_object* v___x_309_; lean_object* v_toNeg_310_; lean_object* v_toSub_311_; lean_object* v_toZSMul_312_; lean_object* v___f_313_; lean_object* v___x_314_; lean_object* v_toIntCast_315_; lean_object* v___x_317_; uint8_t v_isShared_318_; uint8_t v_isSharedCheck_322_; 
lean_inc_ref_n(v_inst_305_, 2);
v___f_306_ = lean_alloc_closure((void*)(lp_mathlib_Pi_ring___redArg___lam__0), 2, 1);
lean_closure_set(v___f_306_, 0, v_inst_305_);
v___f_307_ = lean_alloc_closure((void*)(lp_mathlib_Pi_ring___redArg___lam__1), 2, 1);
lean_closure_set(v___f_307_, 0, v_inst_305_);
v___x_308_ = lp_mathlib_Pi_semiring___redArg(v___f_306_);
v___x_309_ = lp_mathlib_Pi_addCommGroup___redArg(v___f_307_);
v_toNeg_310_ = lean_ctor_get(v___x_309_, 1);
lean_inc(v_toNeg_310_);
v_toSub_311_ = lean_ctor_get(v___x_309_, 2);
lean_inc(v_toSub_311_);
v_toZSMul_312_ = lean_ctor_get(v___x_309_, 3);
lean_inc(v_toZSMul_312_);
lean_dec_ref(v___x_309_);
v___f_313_ = lean_alloc_closure((void*)(lp_mathlib_Pi_ring___redArg___lam__2), 2, 1);
lean_closure_set(v___f_313_, 0, v_inst_305_);
v___x_314_ = lp_mathlib_Pi_addGroupWithOne___redArg(v___f_313_);
v_toIntCast_315_ = lean_ctor_get(v___x_314_, 0);
v_isSharedCheck_322_ = !lean_is_exclusive(v___x_314_);
if (v_isSharedCheck_322_ == 0)
{
lean_object* v_unused_323_; lean_object* v_unused_324_; lean_object* v_unused_325_; lean_object* v_unused_326_; 
v_unused_323_ = lean_ctor_get(v___x_314_, 4);
lean_dec(v_unused_323_);
v_unused_324_ = lean_ctor_get(v___x_314_, 3);
lean_dec(v_unused_324_);
v_unused_325_ = lean_ctor_get(v___x_314_, 2);
lean_dec(v_unused_325_);
v_unused_326_ = lean_ctor_get(v___x_314_, 1);
lean_dec(v_unused_326_);
v___x_317_ = v___x_314_;
v_isShared_318_ = v_isSharedCheck_322_;
goto v_resetjp_316_;
}
else
{
lean_inc(v_toIntCast_315_);
lean_dec(v___x_314_);
v___x_317_ = lean_box(0);
v_isShared_318_ = v_isSharedCheck_322_;
goto v_resetjp_316_;
}
v_resetjp_316_:
{
lean_object* v___x_320_; 
if (v_isShared_318_ == 0)
{
lean_ctor_set(v___x_317_, 4, v_toIntCast_315_);
lean_ctor_set(v___x_317_, 3, v_toZSMul_312_);
lean_ctor_set(v___x_317_, 2, v_toSub_311_);
lean_ctor_set(v___x_317_, 1, v_toNeg_310_);
lean_ctor_set(v___x_317_, 0, v___x_308_);
v___x_320_ = v___x_317_;
goto v_reusejp_319_;
}
else
{
lean_object* v_reuseFailAlloc_321_; 
v_reuseFailAlloc_321_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_321_, 0, v___x_308_);
lean_ctor_set(v_reuseFailAlloc_321_, 1, v_toNeg_310_);
lean_ctor_set(v_reuseFailAlloc_321_, 2, v_toSub_311_);
lean_ctor_set(v_reuseFailAlloc_321_, 3, v_toZSMul_312_);
lean_ctor_set(v_reuseFailAlloc_321_, 4, v_toIntCast_315_);
v___x_320_ = v_reuseFailAlloc_321_;
goto v_reusejp_319_;
}
v_reusejp_319_:
{
return v___x_320_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Pi_ring(lean_object* v_I_327_, lean_object* v_f_328_, lean_object* v_inst_329_){
_start:
{
lean_object* v___x_330_; 
v___x_330_ = lp_mathlib_Pi_ring___redArg(v_inst_329_);
return v___x_330_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Pi_nonUnitalCommRing___redArg(lean_object* v_inst_331_){
_start:
{
lean_object* v___f_332_; lean_object* v___x_333_; 
v___f_332_ = lean_alloc_closure((void*)(lp_mathlib_Pi_nonUnitalRing___redArg___lam__0), 2, 1);
lean_closure_set(v___f_332_, 0, v_inst_331_);
v___x_333_ = lp_mathlib_Pi_nonUnitalRing___redArg(v___f_332_);
return v___x_333_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Pi_nonUnitalCommRing(lean_object* v_I_334_, lean_object* v_f_335_, lean_object* v_inst_336_){
_start:
{
lean_object* v___x_337_; 
v___x_337_ = lp_mathlib_Pi_nonUnitalCommRing___redArg(v_inst_336_);
return v___x_337_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Pi_commRing___redArg___lam__0(lean_object* v_inst_338_, lean_object* v_i_339_){
_start:
{
lean_object* v___x_340_; 
v___x_340_ = lean_apply_1(v_inst_338_, v_i_339_);
return v___x_340_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Pi_commRing___redArg(lean_object* v_inst_341_){
_start:
{
lean_object* v___f_342_; lean_object* v___x_343_; 
v___f_342_ = lean_alloc_closure((void*)(lp_mathlib_Pi_commRing___redArg___lam__0), 2, 1);
lean_closure_set(v___f_342_, 0, v_inst_341_);
v___x_343_ = lp_mathlib_Pi_ring___redArg(v___f_342_);
return v___x_343_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Pi_commRing(lean_object* v_I_344_, lean_object* v_f_345_, lean_object* v_inst_346_){
_start:
{
lean_object* v___x_347_; 
v___x_347_ = lp_mathlib_Pi_commRing___redArg(v_inst_346_);
return v___x_347_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalRingHom_pi___redArg___lam__0(lean_object* v_g_348_, lean_object* v_x_349_, lean_object* v_b_350_){
_start:
{
lean_object* v___x_351_; 
v___x_351_ = lean_apply_2(v_g_348_, v_b_350_, v_x_349_);
return v___x_351_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalRingHom_pi___redArg(lean_object* v_g_352_){
_start:
{
lean_object* v___f_353_; 
v___f_353_ = lean_alloc_closure((void*)(lp_mathlib_NonUnitalRingHom_pi___redArg___lam__0), 3, 1);
lean_closure_set(v___f_353_, 0, v_g_352_);
return v___f_353_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalRingHom_pi(lean_object* v_I_354_, lean_object* v_f_355_, lean_object* v_00_u03b3_356_, lean_object* v_inst_357_, lean_object* v_inst_358_, lean_object* v_g_359_){
_start:
{
lean_object* v___f_360_; 
v___f_360_ = lean_alloc_closure((void*)(lp_mathlib_NonUnitalRingHom_pi___redArg___lam__0), 3, 1);
lean_closure_set(v___f_360_, 0, v_g_359_);
return v___f_360_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalRingHom_pi___boxed(lean_object* v_I_361_, lean_object* v_f_362_, lean_object* v_00_u03b3_363_, lean_object* v_inst_364_, lean_object* v_inst_365_, lean_object* v_g_366_){
_start:
{
lean_object* v_res_367_; 
v_res_367_ = lp_mathlib_NonUnitalRingHom_pi(v_I_361_, v_f_362_, v_00_u03b3_363_, v_inst_364_, v_inst_365_, v_g_366_);
lean_dec_ref(v_inst_365_);
lean_dec_ref(v_inst_364_);
return v_res_367_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Pi_nonUnitalRingHom___redArg(lean_object* v_g_368_){
_start:
{
lean_object* v___f_369_; 
v___f_369_ = lean_alloc_closure((void*)(lp_mathlib_NonUnitalRingHom_pi___redArg___lam__0), 3, 1);
lean_closure_set(v___f_369_, 0, v_g_368_);
return v___f_369_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Pi_nonUnitalRingHom(lean_object* v_I_370_, lean_object* v_f_371_, lean_object* v_00_u03b3_372_, lean_object* v_inst_373_, lean_object* v_inst_374_, lean_object* v_g_375_){
_start:
{
lean_object* v___f_376_; 
v___f_376_ = lean_alloc_closure((void*)(lp_mathlib_NonUnitalRingHom_pi___redArg___lam__0), 3, 1);
lean_closure_set(v___f_376_, 0, v_g_375_);
return v___f_376_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Pi_nonUnitalRingHom___boxed(lean_object* v_I_377_, lean_object* v_f_378_, lean_object* v_00_u03b3_379_, lean_object* v_inst_380_, lean_object* v_inst_381_, lean_object* v_g_382_){
_start:
{
lean_object* v_res_383_; 
v_res_383_ = lp_mathlib_Pi_nonUnitalRingHom(v_I_377_, v_f_378_, v_00_u03b3_379_, v_inst_380_, v_inst_381_, v_g_382_);
lean_dec_ref(v_inst_381_);
lean_dec_ref(v_inst_380_);
return v_res_383_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Pi_evalNonUnitalRingHom___redArg(lean_object* v_i_384_){
_start:
{
lean_object* v___f_385_; 
v___f_385_ = lean_alloc_closure((void*)(lp_mathlib_Pi_evalMulHom___redArg___lam__0), 2, 1);
lean_closure_set(v___f_385_, 0, v_i_384_);
return v___f_385_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Pi_evalNonUnitalRingHom(lean_object* v_I_386_, lean_object* v_f_387_, lean_object* v_inst_388_, lean_object* v_i_389_){
_start:
{
lean_object* v___f_390_; 
v___f_390_ = lean_alloc_closure((void*)(lp_mathlib_Pi_evalMulHom___redArg___lam__0), 2, 1);
lean_closure_set(v___f_390_, 0, v_i_389_);
return v___f_390_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Pi_evalNonUnitalRingHom___boxed(lean_object* v_I_391_, lean_object* v_f_392_, lean_object* v_inst_393_, lean_object* v_i_394_){
_start:
{
lean_object* v_res_395_; 
v_res_395_ = lp_mathlib_Pi_evalNonUnitalRingHom(v_I_391_, v_f_392_, v_inst_393_, v_i_394_);
lean_dec_ref(v_inst_393_);
return v_res_395_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Pi_constNonUnitalRingHom(lean_object* v_00_u03b1_397_, lean_object* v_00_u03b2_398_, lean_object* v_inst_399_){
_start:
{
lean_object* v___x_400_; 
v___x_400_ = ((lean_object*)(lp_mathlib_Pi_constNonUnitalRingHom___closed__0));
return v___x_400_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Pi_constNonUnitalRingHom___boxed(lean_object* v_00_u03b1_401_, lean_object* v_00_u03b2_402_, lean_object* v_inst_403_){
_start:
{
lean_object* v_res_404_; 
v_res_404_ = lp_mathlib_Pi_constNonUnitalRingHom(v_00_u03b1_401_, v_00_u03b2_402_, v_inst_403_);
lean_dec_ref(v_inst_403_);
return v_res_404_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalRingHom_compLeft___redArg___lam__0(lean_object* v_f_405_, lean_object* v_h_406_, lean_object* v___y_407_){
_start:
{
lean_object* v___x_408_; lean_object* v___x_409_; 
v___x_408_ = lean_apply_1(v_h_406_, v___y_407_);
v___x_409_ = lean_apply_1(v_f_405_, v___x_408_);
return v___x_409_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalRingHom_compLeft___redArg(lean_object* v_f_410_){
_start:
{
lean_object* v___f_411_; 
v___f_411_ = lean_alloc_closure((void*)(lp_mathlib_NonUnitalRingHom_compLeft___redArg___lam__0), 3, 1);
lean_closure_set(v___f_411_, 0, v_f_410_);
return v___f_411_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalRingHom_compLeft(lean_object* v_00_u03b1_412_, lean_object* v_00_u03b2_413_, lean_object* v_inst_414_, lean_object* v_inst_415_, lean_object* v_f_416_, lean_object* v_I_417_){
_start:
{
lean_object* v___f_418_; 
v___f_418_ = lean_alloc_closure((void*)(lp_mathlib_NonUnitalRingHom_compLeft___redArg___lam__0), 3, 1);
lean_closure_set(v___f_418_, 0, v_f_416_);
return v___f_418_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalRingHom_compLeft___boxed(lean_object* v_00_u03b1_419_, lean_object* v_00_u03b2_420_, lean_object* v_inst_421_, lean_object* v_inst_422_, lean_object* v_f_423_, lean_object* v_I_424_){
_start:
{
lean_object* v_res_425_; 
v_res_425_ = lp_mathlib_NonUnitalRingHom_compLeft(v_00_u03b1_419_, v_00_u03b2_420_, v_inst_421_, v_inst_422_, v_f_423_, v_I_424_);
lean_dec_ref(v_inst_422_);
lean_dec_ref(v_inst_421_);
return v_res_425_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RingHom_pi___redArg(lean_object* v_g_426_){
_start:
{
lean_object* v___f_427_; 
v___f_427_ = lean_alloc_closure((void*)(lp_mathlib_NonUnitalRingHom_pi___redArg___lam__0), 3, 1);
lean_closure_set(v___f_427_, 0, v_g_426_);
return v___f_427_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RingHom_pi(lean_object* v_I_428_, lean_object* v_f_429_, lean_object* v_00_u03b3_430_, lean_object* v_inst_431_, lean_object* v_inst_432_, lean_object* v_g_433_){
_start:
{
lean_object* v___f_434_; 
v___f_434_ = lean_alloc_closure((void*)(lp_mathlib_NonUnitalRingHom_pi___redArg___lam__0), 3, 1);
lean_closure_set(v___f_434_, 0, v_g_433_);
return v___f_434_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RingHom_pi___boxed(lean_object* v_I_435_, lean_object* v_f_436_, lean_object* v_00_u03b3_437_, lean_object* v_inst_438_, lean_object* v_inst_439_, lean_object* v_g_440_){
_start:
{
lean_object* v_res_441_; 
v_res_441_ = lp_mathlib_RingHom_pi(v_I_435_, v_f_436_, v_00_u03b3_437_, v_inst_438_, v_inst_439_, v_g_440_);
lean_dec_ref(v_inst_439_);
lean_dec_ref(v_inst_438_);
return v_res_441_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Pi_ringHom___redArg(lean_object* v_g_442_){
_start:
{
lean_object* v___f_443_; 
v___f_443_ = lean_alloc_closure((void*)(lp_mathlib_NonUnitalRingHom_pi___redArg___lam__0), 3, 1);
lean_closure_set(v___f_443_, 0, v_g_442_);
return v___f_443_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Pi_ringHom(lean_object* v_I_444_, lean_object* v_f_445_, lean_object* v_00_u03b3_446_, lean_object* v_inst_447_, lean_object* v_inst_448_, lean_object* v_g_449_){
_start:
{
lean_object* v___f_450_; 
v___f_450_ = lean_alloc_closure((void*)(lp_mathlib_NonUnitalRingHom_pi___redArg___lam__0), 3, 1);
lean_closure_set(v___f_450_, 0, v_g_449_);
return v___f_450_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Pi_ringHom___boxed(lean_object* v_I_451_, lean_object* v_f_452_, lean_object* v_00_u03b3_453_, lean_object* v_inst_454_, lean_object* v_inst_455_, lean_object* v_g_456_){
_start:
{
lean_object* v_res_457_; 
v_res_457_ = lp_mathlib_Pi_ringHom(v_I_451_, v_f_452_, v_00_u03b3_453_, v_inst_454_, v_inst_455_, v_g_456_);
lean_dec_ref(v_inst_455_);
lean_dec_ref(v_inst_454_);
return v_res_457_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Pi_evalRingHom___redArg(lean_object* v_i_458_){
_start:
{
lean_object* v___f_459_; 
v___f_459_ = lean_alloc_closure((void*)(lp_mathlib_Pi_evalMulHom___redArg___lam__0), 2, 1);
lean_closure_set(v___f_459_, 0, v_i_458_);
return v___f_459_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Pi_evalRingHom(lean_object* v_I_460_, lean_object* v_f_461_, lean_object* v_inst_462_, lean_object* v_i_463_){
_start:
{
lean_object* v___f_464_; 
v___f_464_ = lean_alloc_closure((void*)(lp_mathlib_Pi_evalMulHom___redArg___lam__0), 2, 1);
lean_closure_set(v___f_464_, 0, v_i_463_);
return v___f_464_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Pi_evalRingHom___boxed(lean_object* v_I_465_, lean_object* v_f_466_, lean_object* v_inst_467_, lean_object* v_i_468_){
_start:
{
lean_object* v_res_469_; 
v_res_469_ = lp_mathlib_Pi_evalRingHom(v_I_465_, v_f_466_, v_inst_467_, v_i_468_);
lean_dec_ref(v_inst_467_);
return v_res_469_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Pi_constRingHom(lean_object* v_00_u03b1_470_, lean_object* v_00_u03b2_471_, lean_object* v_inst_472_){
_start:
{
lean_object* v___x_473_; 
v___x_473_ = ((lean_object*)(lp_mathlib_Pi_constNonUnitalRingHom___closed__0));
return v___x_473_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Pi_constRingHom___boxed(lean_object* v_00_u03b1_474_, lean_object* v_00_u03b2_475_, lean_object* v_inst_476_){
_start:
{
lean_object* v_res_477_; 
v_res_477_ = lp_mathlib_Pi_constRingHom(v_00_u03b1_474_, v_00_u03b2_475_, v_inst_476_);
lean_dec_ref(v_inst_476_);
return v_res_477_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RingHom_compLeft___redArg(lean_object* v_f_478_){
_start:
{
lean_object* v___f_479_; 
v___f_479_ = lean_alloc_closure((void*)(lp_mathlib_NonUnitalRingHom_compLeft___redArg___lam__0), 3, 1);
lean_closure_set(v___f_479_, 0, v_f_478_);
return v___f_479_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RingHom_compLeft(lean_object* v_00_u03b1_480_, lean_object* v_00_u03b2_481_, lean_object* v_inst_482_, lean_object* v_inst_483_, lean_object* v_f_484_, lean_object* v_I_485_){
_start:
{
lean_object* v___f_486_; 
v___f_486_ = lean_alloc_closure((void*)(lp_mathlib_NonUnitalRingHom_compLeft___redArg___lam__0), 3, 1);
lean_closure_set(v___f_486_, 0, v_f_484_);
return v___f_486_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RingHom_compLeft___boxed(lean_object* v_00_u03b1_487_, lean_object* v_00_u03b2_488_, lean_object* v_inst_489_, lean_object* v_inst_490_, lean_object* v_f_491_, lean_object* v_I_492_){
_start:
{
lean_object* v_res_493_; 
v_res_493_ = lp_mathlib_RingHom_compLeft(v_00_u03b1_487_, v_00_u03b2_488_, v_inst_489_, v_inst_490_, v_f_491_, v_I_492_);
lean_dec_ref(v_inst_490_);
lean_dec_ref(v_inst_489_);
return v_res_493_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Group_Pi_Lemmas(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_GroupWithZero_Pi(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Ring_CompTypeclasses(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Ring_Hom_Defs(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Ring_Pi(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Group_Pi_Lemmas(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_GroupWithZero_Pi(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Ring_CompTypeclasses(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Ring_Hom_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_Algebra_Ring_Pi(uint8_t builtin) {
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
lean_object* initialize_mathlib_Mathlib_Algebra_Group_Pi_Lemmas(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_GroupWithZero_Pi(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Ring_CompTypeclasses(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Ring_Hom_Defs(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Algebra_Ring_Pi(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Group_Pi_Lemmas(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_GroupWithZero_Pi(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Ring_CompTypeclasses(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Ring_Hom_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Ring_Pi(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_Algebra_Ring_Pi(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_Algebra_Ring_Pi(builtin);
}
#ifdef __cplusplus
}
#endif
