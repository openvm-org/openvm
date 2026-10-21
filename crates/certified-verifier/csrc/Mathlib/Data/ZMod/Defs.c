// Lean compiler output
// Module: Mathlib.Data.ZMod.Defs
// Imports: public import Init public meta import Init public import Mathlib.Algebra.Group.Fin.Basic public import Mathlib.Algebra.NeZero public import Mathlib.Algebra.Ring.Int.Defs public import Mathlib.Algebra.Ring.GrindInstances public import Mathlib.Data.Nat.ModEq public import Mathlib.Data.Fintype.EquivFin public import Mathlib.Algebra.Ring.Nat
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
lean_object* l_Fin_mul___boxed(lean_object*, lean_object*, lean_object*);
uint8_t lean_int_dec_eq(lean_object*, lean_object*);
uint8_t lean_nat_dec_eq(lean_object*, lean_object*);
lean_object* lean_nat_to_int(lean_object*);
uint8_t lean_int_dec_lt(lean_object*, lean_object*);
lean_object* l_Int_repr(lean_object*);
lean_object* l_Repr_addAppParen(lean_object*, lean_object*);
lean_object* lean_nat_sub(lean_object*, lean_object*);
lean_object* l_Nat_reprFast(lean_object*);
lean_object* lean_int_mul(lean_object*, lean_object*);
lean_object* lp_mathlib_Fin_addCommMonoid___redArg(lean_object*);
lean_object* lp_mathlib_AddMonoid_toAddZeroClass___redArg(lean_object*);
lean_object* lp_mathlib_AddZeroClass_toAddZero___redArg(lean_object*);
lean_object* l_Fin_neg___lam__0___boxed(lean_object*, lean_object*);
lean_object* l_Fin_add___boxed(lean_object*, lean_object*, lean_object*);
lean_object* l_nsmulRec___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_zsmulRec___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Fin_mul(lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Fin_instAddMonoidWithOne___redArg(lean_object*);
lean_object* lp_mathlib_npowBinRec_go___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_NPow_ofPow___redArg___lam__0(lean_object*, lean_object*, lean_object*);
lean_object* lean_int_sub(lean_object*, lean_object*);
lean_object* l_Fin_sub(lean_object*, lean_object*, lean_object*);
lean_object* lean_nat_add(lean_object*, lean_object*);
lean_object* l_List_finRange(lean_object*);
lean_object* l_Fin_intCast___redArg(lean_object*, lean_object*);
lean_object* lp_mathlib_Fin_addCommGroup___redArg(lean_object*);
lean_object* lp_mathlib_Ring_toAddGroupWithOne___redArg(lean_object*);
extern lean_object* lp_mathlib_Fin_instUnique;
lean_object* lean_int_neg(lean_object*);
lean_object* lean_nat_mod(lean_object*, lean_object*);
lean_object* l_Int_pow(lean_object*, lean_object*);
lean_object* l_nsmulRec___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lean_int_add(lean_object*, lean_object*);
lean_object* l_Fin_add(lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_instMulZeroClassOfSemiring___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Fin_instCommSemigroup(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Fin_instHPowNatOfNeZero__mathlib___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Fin_instHPowNatOfNeZero__mathlib___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Fin_instHPowNatOfNeZero__mathlib___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Fin_instHPowNatOfNeZero__mathlib(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Fin_instPowNatOfNeZero__mathlib___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Fin_instPowNatOfNeZero__mathlib(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Fin_instDistrib(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Fin_instNonUnitalCommRing___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Fin_instNonUnitalCommRing(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Fin_instCommMonoid___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Fin_instCommMonoid(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Fin_instHasDistribNeg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Fin_instCommRing___redArg___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Fin_instCommRing___redArg___lam__0___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Fin_instCommRing___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Fin_instCommRing(lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_ZMod_decidableEq___aux__1(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ZMod_decidableEq___aux__1___boxed(lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_ZMod_decidableEq___aux__3___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ZMod_decidableEq___aux__3___redArg___boxed(lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_ZMod_decidableEq___aux__3(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ZMod_decidableEq___aux__3___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_ZMod_decidableEq(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ZMod_decidableEq___boxed(lean_object*, lean_object*, lean_object*);
static lean_once_cell_t lp_mathlib_ZMod_repr___aux__1___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_ZMod_repr___aux__1___closed__0;
LEAN_EXPORT lean_object* lp_mathlib_ZMod_repr___aux__1(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ZMod_repr___aux__1___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ZMod_repr___aux__3___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ZMod_repr___aux__3(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ZMod_repr___aux__3___boxed(lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_ZMod_repr___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_ZMod_repr___aux__1___boxed, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_ZMod_repr___closed__0 = (const lean_object*)&lp_mathlib_ZMod_repr___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_ZMod_repr(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ZMod_repr___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ZMod_instUnique;
LEAN_EXPORT lean_object* lp_mathlib_ZMod_fintype___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ZMod_fintype___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ZMod_fintype(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ZMod_fintype___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ZMod_commRing___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ZMod_commRing___lam__1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ZMod_commRing___lam__1___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ZMod_commRing___lam__2(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ZMod_commRing___lam__2___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ZMod_commRing___lam__3(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ZMod_commRing___lam__3___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ZMod_commRing___lam__4(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ZMod_commRing___lam__4___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ZMod_commRing___lam__5(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ZMod_commRing___lam__5___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ZMod_commRing___lam__6(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ZMod_commRing___lam__7(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ZMod_commRing___lam__7___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ZMod_commRing___lam__8(lean_object*, lean_object*);
static lean_once_cell_t lp_mathlib_ZMod_commRing___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_ZMod_commRing___closed__0;
LEAN_EXPORT lean_object* lp_mathlib_ZMod_commRing(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ZMod_inhabited(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Fin_instCommSemigroup(lean_object* v_n_1_){
_start:
{
lean_object* v___x_2_; 
v___x_2_ = lean_alloc_closure((void*)(l_Fin_mul___boxed), 3, 1);
lean_closure_set(v___x_2_, 0, v_n_1_);
return v___x_2_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Fin_instHPowNatOfNeZero__mathlib___redArg___lam__0(lean_object* v_n_3_, lean_object* v___x_4_, lean_object* v_a_5_, lean_object* v_m_6_){
_start:
{
lean_object* v___x_7_; lean_object* v_toOne_8_; lean_object* v___x_9_; 
v___x_7_ = lp_mathlib_Fin_instAddMonoidWithOne___redArg(v_n_3_);
v_toOne_8_ = lean_ctor_get(v___x_7_, 2);
lean_inc(v_toOne_8_);
lean_dec_ref(v___x_7_);
v___x_9_ = lp_mathlib_npowBinRec_go___redArg(v___x_4_, v_m_6_, v_toOne_8_, v_a_5_);
return v___x_9_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Fin_instHPowNatOfNeZero__mathlib___redArg___lam__0___boxed(lean_object* v_n_10_, lean_object* v___x_11_, lean_object* v_a_12_, lean_object* v_m_13_){
_start:
{
lean_object* v_res_14_; 
v_res_14_ = lp_mathlib_Fin_instHPowNatOfNeZero__mathlib___redArg___lam__0(v_n_10_, v___x_11_, v_a_12_, v_m_13_);
lean_dec(v_m_13_);
return v_res_14_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Fin_instHPowNatOfNeZero__mathlib___redArg(lean_object* v_n_15_){
_start:
{
lean_object* v___x_16_; lean_object* v___f_17_; 
lean_inc(v_n_15_);
v___x_16_ = lean_alloc_closure((void*)(l_Fin_mul___boxed), 3, 1);
lean_closure_set(v___x_16_, 0, v_n_15_);
v___f_17_ = lean_alloc_closure((void*)(lp_mathlib_Fin_instHPowNatOfNeZero__mathlib___redArg___lam__0___boxed), 4, 2);
lean_closure_set(v___f_17_, 0, v_n_15_);
lean_closure_set(v___f_17_, 1, v___x_16_);
return v___f_17_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Fin_instHPowNatOfNeZero__mathlib(lean_object* v_n_18_, lean_object* v_inst_19_){
_start:
{
lean_object* v___x_20_; 
v___x_20_ = lp_mathlib_Fin_instHPowNatOfNeZero__mathlib___redArg(v_n_18_);
return v___x_20_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Fin_instPowNatOfNeZero__mathlib___redArg(lean_object* v_n_21_){
_start:
{
lean_object* v___x_22_; lean_object* v___f_23_; 
lean_inc(v_n_21_);
v___x_22_ = lean_alloc_closure((void*)(l_Fin_mul___boxed), 3, 1);
lean_closure_set(v___x_22_, 0, v_n_21_);
v___f_23_ = lean_alloc_closure((void*)(lp_mathlib_Fin_instHPowNatOfNeZero__mathlib___redArg___lam__0___boxed), 4, 2);
lean_closure_set(v___f_23_, 0, v_n_21_);
lean_closure_set(v___f_23_, 1, v___x_22_);
return v___f_23_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Fin_instPowNatOfNeZero__mathlib(lean_object* v_n_24_, lean_object* v_inst_25_){
_start:
{
lean_object* v___x_26_; 
v___x_26_ = lp_mathlib_Fin_instPowNatOfNeZero__mathlib___redArg(v_n_24_);
return v___x_26_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Fin_instDistrib(lean_object* v_n_27_){
_start:
{
lean_object* v___x_28_; lean_object* v___x_29_; lean_object* v___x_30_; 
lean_inc(v_n_27_);
v___x_28_ = lean_alloc_closure((void*)(l_Fin_mul___boxed), 3, 1);
lean_closure_set(v___x_28_, 0, v_n_27_);
v___x_29_ = lean_alloc_closure((void*)(l_Fin_add___boxed), 3, 1);
lean_closure_set(v___x_29_, 0, v_n_27_);
v___x_30_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_30_, 0, v___x_28_);
lean_ctor_set(v___x_30_, 1, v___x_29_);
return v___x_30_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Fin_instNonUnitalCommRing___redArg(lean_object* v_n_31_){
_start:
{
lean_object* v___x_32_; lean_object* v___x_33_; lean_object* v___x_34_; 
lean_inc(v_n_31_);
v___x_32_ = lp_mathlib_Fin_addCommGroup___redArg(v_n_31_);
v___x_33_ = lean_alloc_closure((void*)(l_Fin_mul___boxed), 3, 1);
lean_closure_set(v___x_33_, 0, v_n_31_);
v___x_34_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_34_, 0, v___x_32_);
lean_ctor_set(v___x_34_, 1, v___x_33_);
return v___x_34_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Fin_instNonUnitalCommRing(lean_object* v_n_35_, lean_object* v_inst_36_){
_start:
{
lean_object* v___x_37_; 
v___x_37_ = lp_mathlib_Fin_instNonUnitalCommRing___redArg(v_n_35_);
return v___x_37_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Fin_instCommMonoid___redArg(lean_object* v_n_38_){
_start:
{
lean_object* v___x_39_; lean_object* v_toOne_40_; lean_object* v___x_42_; uint8_t v_isShared_43_; uint8_t v_isSharedCheck_50_; 
lean_inc(v_n_38_);
v___x_39_ = lp_mathlib_Fin_instAddMonoidWithOne___redArg(v_n_38_);
v_toOne_40_ = lean_ctor_get(v___x_39_, 2);
v_isSharedCheck_50_ = !lean_is_exclusive(v___x_39_);
if (v_isSharedCheck_50_ == 0)
{
lean_object* v_unused_51_; lean_object* v_unused_52_; 
v_unused_51_ = lean_ctor_get(v___x_39_, 1);
lean_dec(v_unused_51_);
v_unused_52_ = lean_ctor_get(v___x_39_, 0);
lean_dec(v_unused_52_);
v___x_42_ = v___x_39_;
v_isShared_43_ = v_isSharedCheck_50_;
goto v_resetjp_41_;
}
else
{
lean_inc(v_toOne_40_);
lean_dec(v___x_39_);
v___x_42_ = lean_box(0);
v_isShared_43_ = v_isSharedCheck_50_;
goto v_resetjp_41_;
}
v_resetjp_41_:
{
lean_object* v___x_44_; lean_object* v___x_45_; lean_object* v___f_46_; lean_object* v___x_48_; 
lean_inc(v_n_38_);
v___x_44_ = lean_alloc_closure((void*)(l_Fin_mul___boxed), 3, 1);
lean_closure_set(v___x_44_, 0, v_n_38_);
v___x_45_ = lp_mathlib_Fin_instPowNatOfNeZero__mathlib___redArg(v_n_38_);
v___f_46_ = lean_alloc_closure((void*)(lp_mathlib_NPow_ofPow___redArg___lam__0), 3, 1);
lean_closure_set(v___f_46_, 0, v___x_45_);
if (v_isShared_43_ == 0)
{
lean_ctor_set(v___x_42_, 2, v___f_46_);
lean_ctor_set(v___x_42_, 1, v___x_44_);
lean_ctor_set(v___x_42_, 0, v_toOne_40_);
v___x_48_ = v___x_42_;
goto v_reusejp_47_;
}
else
{
lean_object* v_reuseFailAlloc_49_; 
v_reuseFailAlloc_49_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v_reuseFailAlloc_49_, 0, v_toOne_40_);
lean_ctor_set(v_reuseFailAlloc_49_, 1, v___x_44_);
lean_ctor_set(v_reuseFailAlloc_49_, 2, v___f_46_);
v___x_48_ = v_reuseFailAlloc_49_;
goto v_reusejp_47_;
}
v_reusejp_47_:
{
return v___x_48_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Fin_instCommMonoid(lean_object* v_n_53_, lean_object* v_inst_54_){
_start:
{
lean_object* v___x_55_; 
v___x_55_ = lp_mathlib_Fin_instCommMonoid___redArg(v_n_53_);
return v___x_55_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Fin_instHasDistribNeg(lean_object* v_n_56_){
_start:
{
lean_object* v___f_57_; 
v___f_57_ = lean_alloc_closure((void*)(l_Fin_neg___lam__0___boxed), 2, 1);
lean_closure_set(v___f_57_, 0, v_n_56_);
return v___f_57_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Fin_instCommRing___redArg___lam__0(lean_object* v_n_58_, lean_object* v_n_59_){
_start:
{
lean_object* v___x_60_; 
v___x_60_ = l_Fin_intCast___redArg(v_n_58_, v_n_59_);
return v___x_60_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Fin_instCommRing___redArg___lam__0___boxed(lean_object* v_n_61_, lean_object* v_n_62_){
_start:
{
lean_object* v_res_63_; 
v_res_63_ = lp_mathlib_Fin_instCommRing___redArg___lam__0(v_n_61_, v_n_62_);
lean_dec(v_n_62_);
lean_dec(v_n_61_);
return v_res_63_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Fin_instCommRing___redArg(lean_object* v_n_64_){
_start:
{
lean_object* v___x_65_; lean_object* v_toAddMonoid_66_; lean_object* v_toNeg_67_; lean_object* v_toSub_68_; lean_object* v_toZSMul_69_; lean_object* v___x_70_; lean_object* v___x_71_; lean_object* v_toNatCast_72_; lean_object* v___x_74_; uint8_t v_isShared_75_; uint8_t v_isSharedCheck_81_; 
lean_inc_n(v_n_64_, 3);
v___x_65_ = lp_mathlib_Fin_addCommGroup___redArg(v_n_64_);
v_toAddMonoid_66_ = lean_ctor_get(v___x_65_, 0);
lean_inc_ref(v_toAddMonoid_66_);
v_toNeg_67_ = lean_ctor_get(v___x_65_, 1);
lean_inc(v_toNeg_67_);
v_toSub_68_ = lean_ctor_get(v___x_65_, 2);
lean_inc(v_toSub_68_);
v_toZSMul_69_ = lean_ctor_get(v___x_65_, 3);
lean_inc(v_toZSMul_69_);
lean_dec_ref(v___x_65_);
v___x_70_ = lp_mathlib_Fin_instCommMonoid___redArg(v_n_64_);
v___x_71_ = lp_mathlib_Fin_instAddMonoidWithOne___redArg(v_n_64_);
v_toNatCast_72_ = lean_ctor_get(v___x_71_, 0);
v_isSharedCheck_81_ = !lean_is_exclusive(v___x_71_);
if (v_isSharedCheck_81_ == 0)
{
lean_object* v_unused_82_; lean_object* v_unused_83_; 
v_unused_82_ = lean_ctor_get(v___x_71_, 2);
lean_dec(v_unused_82_);
v_unused_83_ = lean_ctor_get(v___x_71_, 1);
lean_dec(v_unused_83_);
v___x_74_ = v___x_71_;
v_isShared_75_ = v_isSharedCheck_81_;
goto v_resetjp_73_;
}
else
{
lean_inc(v_toNatCast_72_);
lean_dec(v___x_71_);
v___x_74_ = lean_box(0);
v_isShared_75_ = v_isSharedCheck_81_;
goto v_resetjp_73_;
}
v_resetjp_73_:
{
lean_object* v___f_76_; lean_object* v___x_78_; 
v___f_76_ = lean_alloc_closure((void*)(lp_mathlib_Fin_instCommRing___redArg___lam__0___boxed), 2, 1);
lean_closure_set(v___f_76_, 0, v_n_64_);
if (v_isShared_75_ == 0)
{
lean_ctor_set(v___x_74_, 2, v_toNatCast_72_);
lean_ctor_set(v___x_74_, 1, v___x_70_);
lean_ctor_set(v___x_74_, 0, v_toAddMonoid_66_);
v___x_78_ = v___x_74_;
goto v_reusejp_77_;
}
else
{
lean_object* v_reuseFailAlloc_80_; 
v_reuseFailAlloc_80_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v_reuseFailAlloc_80_, 0, v_toAddMonoid_66_);
lean_ctor_set(v_reuseFailAlloc_80_, 1, v___x_70_);
lean_ctor_set(v_reuseFailAlloc_80_, 2, v_toNatCast_72_);
v___x_78_ = v_reuseFailAlloc_80_;
goto v_reusejp_77_;
}
v_reusejp_77_:
{
lean_object* v___x_79_; 
v___x_79_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v___x_79_, 0, v___x_78_);
lean_ctor_set(v___x_79_, 1, v_toNeg_67_);
lean_ctor_set(v___x_79_, 2, v_toSub_68_);
lean_ctor_set(v___x_79_, 3, v_toZSMul_69_);
lean_ctor_set(v___x_79_, 4, v___f_76_);
return v___x_79_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Fin_instCommRing(lean_object* v_n_84_, lean_object* v_inst_85_){
_start:
{
lean_object* v___x_86_; 
v___x_86_ = lp_mathlib_Fin_instCommRing___redArg(v_n_84_);
return v___x_86_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_ZMod_decidableEq___aux__1(lean_object* v_a_87_, lean_object* v_b_88_){
_start:
{
uint8_t v___x_89_; 
v___x_89_ = lean_int_dec_eq(v_a_87_, v_b_88_);
return v___x_89_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ZMod_decidableEq___aux__1___boxed(lean_object* v_a_90_, lean_object* v_b_91_){
_start:
{
uint8_t v_res_92_; lean_object* v_r_93_; 
v_res_92_ = lp_mathlib_ZMod_decidableEq___aux__1(v_a_90_, v_b_91_);
lean_dec(v_b_91_);
lean_dec(v_a_90_);
v_r_93_ = lean_box(v_res_92_);
return v_r_93_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_ZMod_decidableEq___aux__3___redArg(lean_object* v_a_94_, lean_object* v_b_95_){
_start:
{
uint8_t v___x_96_; 
v___x_96_ = lean_nat_dec_eq(v_a_94_, v_b_95_);
return v___x_96_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ZMod_decidableEq___aux__3___redArg___boxed(lean_object* v_a_97_, lean_object* v_b_98_){
_start:
{
uint8_t v_res_99_; lean_object* v_r_100_; 
v_res_99_ = lp_mathlib_ZMod_decidableEq___aux__3___redArg(v_a_97_, v_b_98_);
lean_dec(v_b_98_);
lean_dec(v_a_97_);
v_r_100_ = lean_box(v_res_99_);
return v_r_100_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_ZMod_decidableEq___aux__3(lean_object* v_n_101_, lean_object* v_a_102_, lean_object* v_b_103_){
_start:
{
uint8_t v___x_104_; 
v___x_104_ = lean_nat_dec_eq(v_a_102_, v_b_103_);
return v___x_104_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ZMod_decidableEq___aux__3___boxed(lean_object* v_n_105_, lean_object* v_a_106_, lean_object* v_b_107_){
_start:
{
uint8_t v_res_108_; lean_object* v_r_109_; 
v_res_108_ = lp_mathlib_ZMod_decidableEq___aux__3(v_n_105_, v_a_106_, v_b_107_);
lean_dec(v_b_107_);
lean_dec(v_a_106_);
lean_dec(v_n_105_);
v_r_109_ = lean_box(v_res_108_);
return v_r_109_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_ZMod_decidableEq(lean_object* v_x_110_, lean_object* v_a_111_, lean_object* v_b_112_){
_start:
{
lean_object* v_zero_113_; uint8_t v_isZero_114_; 
v_zero_113_ = lean_unsigned_to_nat(0u);
v_isZero_114_ = lean_nat_dec_eq(v_x_110_, v_zero_113_);
if (v_isZero_114_ == 1)
{
uint8_t v___x_115_; 
v___x_115_ = lean_int_dec_eq(v_a_111_, v_b_112_);
return v___x_115_;
}
else
{
uint8_t v___x_116_; 
v___x_116_ = lean_nat_dec_eq(v_a_111_, v_b_112_);
return v___x_116_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_ZMod_decidableEq___boxed(lean_object* v_x_117_, lean_object* v_a_118_, lean_object* v_b_119_){
_start:
{
uint8_t v_res_120_; lean_object* v_r_121_; 
v_res_120_ = lp_mathlib_ZMod_decidableEq(v_x_117_, v_a_118_, v_b_119_);
lean_dec(v_b_119_);
lean_dec(v_a_118_);
lean_dec(v_x_117_);
v_r_121_ = lean_box(v_res_120_);
return v_r_121_;
}
}
static lean_object* _init_lp_mathlib_ZMod_repr___aux__1___closed__0(void){
_start:
{
lean_object* v___x_122_; lean_object* v___x_123_; 
v___x_122_ = lean_unsigned_to_nat(0u);
v___x_123_ = lean_nat_to_int(v___x_122_);
return v___x_123_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ZMod_repr___aux__1(lean_object* v_i_124_, lean_object* v_prec_125_){
_start:
{
lean_object* v___x_126_; uint8_t v___x_127_; 
v___x_126_ = lean_obj_once(&lp_mathlib_ZMod_repr___aux__1___closed__0, &lp_mathlib_ZMod_repr___aux__1___closed__0_once, _init_lp_mathlib_ZMod_repr___aux__1___closed__0);
v___x_127_ = lean_int_dec_lt(v_i_124_, v___x_126_);
if (v___x_127_ == 0)
{
lean_object* v___x_128_; lean_object* v___x_129_; 
v___x_128_ = l_Int_repr(v_i_124_);
v___x_129_ = lean_alloc_ctor(3, 1, 0);
lean_ctor_set(v___x_129_, 0, v___x_128_);
return v___x_129_;
}
else
{
lean_object* v___x_130_; lean_object* v___x_131_; lean_object* v___x_132_; 
v___x_130_ = l_Int_repr(v_i_124_);
v___x_131_ = lean_alloc_ctor(3, 1, 0);
lean_ctor_set(v___x_131_, 0, v___x_130_);
v___x_132_ = l_Repr_addAppParen(v___x_131_, v_prec_125_);
return v___x_132_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_ZMod_repr___aux__1___boxed(lean_object* v_i_133_, lean_object* v_prec_134_){
_start:
{
lean_object* v_res_135_; 
v_res_135_ = lp_mathlib_ZMod_repr___aux__1(v_i_133_, v_prec_134_);
lean_dec(v_prec_134_);
lean_dec(v_i_133_);
return v_res_135_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ZMod_repr___aux__3___redArg(lean_object* v_f_136_){
_start:
{
lean_object* v___x_137_; lean_object* v___x_138_; 
v___x_137_ = l_Nat_reprFast(v_f_136_);
v___x_138_ = lean_alloc_ctor(3, 1, 0);
lean_ctor_set(v___x_138_, 0, v___x_137_);
return v___x_138_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ZMod_repr___aux__3(lean_object* v_n_139_, lean_object* v_f_140_, lean_object* v_x_141_){
_start:
{
lean_object* v___x_142_; lean_object* v___x_143_; 
v___x_142_ = l_Nat_reprFast(v_f_140_);
v___x_143_ = lean_alloc_ctor(3, 1, 0);
lean_ctor_set(v___x_143_, 0, v___x_142_);
return v___x_143_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ZMod_repr___aux__3___boxed(lean_object* v_n_144_, lean_object* v_f_145_, lean_object* v_x_146_){
_start:
{
lean_object* v_res_147_; 
v_res_147_ = lp_mathlib_ZMod_repr___aux__3(v_n_144_, v_f_145_, v_x_146_);
lean_dec(v_x_146_);
lean_dec(v_n_144_);
return v_res_147_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ZMod_repr(lean_object* v_x_149_){
_start:
{
lean_object* v_zero_150_; uint8_t v_isZero_151_; 
v_zero_150_ = lean_unsigned_to_nat(0u);
v_isZero_151_ = lean_nat_dec_eq(v_x_149_, v_zero_150_);
if (v_isZero_151_ == 1)
{
lean_object* v___x_152_; 
v___x_152_ = ((lean_object*)(lp_mathlib_ZMod_repr___closed__0));
return v___x_152_;
}
else
{
lean_object* v_one_153_; lean_object* v_n_154_; lean_object* v___x_155_; 
v_one_153_ = lean_unsigned_to_nat(1u);
v_n_154_ = lean_nat_sub(v_x_149_, v_one_153_);
v___x_155_ = lean_alloc_closure((void*)(lp_mathlib_ZMod_repr___aux__3___boxed), 3, 1);
lean_closure_set(v___x_155_, 0, v_n_154_);
return v___x_155_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_ZMod_repr___boxed(lean_object* v_x_156_){
_start:
{
lean_object* v_res_157_; 
v_res_157_ = lp_mathlib_ZMod_repr(v_x_156_);
lean_dec(v_x_156_);
return v_res_157_;
}
}
static lean_object* _init_lp_mathlib_ZMod_instUnique(void){
_start:
{
lean_object* v___x_158_; 
v___x_158_ = lp_mathlib_Fin_instUnique;
return v___x_158_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ZMod_fintype___redArg(lean_object* v_x_159_){
_start:
{
lean_object* v_zero_160_; uint8_t v_isZero_161_; lean_object* v_one_162_; lean_object* v_n_163_; lean_object* v___x_164_; lean_object* v___x_165_; 
v_zero_160_ = lean_unsigned_to_nat(0u);
v_isZero_161_ = lean_nat_dec_eq(v_x_159_, v_zero_160_);
v_one_162_ = lean_unsigned_to_nat(1u);
v_n_163_ = lean_nat_sub(v_x_159_, v_one_162_);
v___x_164_ = lean_nat_add(v_n_163_, v_one_162_);
lean_dec(v_n_163_);
v___x_165_ = l_List_finRange(v___x_164_);
return v___x_165_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ZMod_fintype___redArg___boxed(lean_object* v_x_166_){
_start:
{
lean_object* v_res_167_; 
v_res_167_ = lp_mathlib_ZMod_fintype___redArg(v_x_166_);
lean_dec(v_x_166_);
return v_res_167_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ZMod_fintype(lean_object* v_x_168_, lean_object* v_x_169_){
_start:
{
lean_object* v___x_170_; 
v___x_170_ = lp_mathlib_ZMod_fintype___redArg(v_x_168_);
return v___x_170_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ZMod_fintype___boxed(lean_object* v_x_171_, lean_object* v_x_172_){
_start:
{
lean_object* v_res_173_; 
v_res_173_ = lp_mathlib_ZMod_fintype(v_x_171_, v_x_172_);
lean_dec(v_x_171_);
return v_res_173_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ZMod_commRing___lam__0(lean_object* v_n_174_, lean_object* v___y_175_){
_start:
{
lean_object* v_zero_176_; uint8_t v_isZero_177_; 
v_zero_176_ = lean_unsigned_to_nat(0u);
v_isZero_177_ = lean_nat_dec_eq(v_n_174_, v_zero_176_);
if (v_isZero_177_ == 1)
{
lean_dec(v_n_174_);
return v___y_175_;
}
else
{
lean_object* v___x_178_; lean_object* v___x_179_; lean_object* v_toIntCast_180_; lean_object* v___x_181_; 
v___x_178_ = lp_mathlib_Fin_instCommRing___redArg(v_n_174_);
v___x_179_ = lp_mathlib_Ring_toAddGroupWithOne___redArg(v___x_178_);
v_toIntCast_180_ = lean_ctor_get(v___x_179_, 0);
lean_inc(v_toIntCast_180_);
lean_dec_ref(v___x_179_);
v___x_181_ = lean_apply_1(v_toIntCast_180_, v___y_175_);
return v___x_181_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_ZMod_commRing___lam__1(lean_object* v_n_182_, lean_object* v___y_183_, lean_object* v___y_184_){
_start:
{
lean_object* v_zero_185_; uint8_t v_isZero_186_; 
v_zero_185_ = lean_unsigned_to_nat(0u);
v_isZero_186_ = lean_nat_dec_eq(v_n_182_, v_zero_185_);
if (v_isZero_186_ == 1)
{
lean_object* v___x_187_; 
lean_dec(v_n_182_);
v___x_187_ = lean_int_mul(v___y_183_, v___y_184_);
lean_dec(v___y_184_);
return v___x_187_;
}
else
{
lean_object* v___x_188_; lean_object* v___x_189_; lean_object* v___x_190_; lean_object* v_toZero_191_; lean_object* v___f_192_; lean_object* v___x_193_; lean_object* v___x_194_; lean_object* v___x_195_; 
lean_inc_n(v_n_182_, 2);
v___x_188_ = lp_mathlib_Fin_addCommMonoid___redArg(v_n_182_);
v___x_189_ = lp_mathlib_AddMonoid_toAddZeroClass___redArg(v___x_188_);
lean_dec_ref(v___x_188_);
v___x_190_ = lp_mathlib_AddZeroClass_toAddZero___redArg(v___x_189_);
v_toZero_191_ = lean_ctor_get(v___x_190_, 0);
lean_inc(v_toZero_191_);
lean_dec_ref(v___x_190_);
v___f_192_ = lean_alloc_closure((void*)(l_Fin_neg___lam__0___boxed), 2, 1);
lean_closure_set(v___f_192_, 0, v_n_182_);
v___x_193_ = lean_alloc_closure((void*)(l_Fin_add___boxed), 3, 1);
lean_closure_set(v___x_193_, 0, v_n_182_);
v___x_194_ = lean_alloc_closure((void*)(l_nsmulRec___boxed), 5, 3);
lean_closure_set(v___x_194_, 0, lean_box(0));
lean_closure_set(v___x_194_, 1, v_toZero_191_);
lean_closure_set(v___x_194_, 2, v___x_193_);
v___x_195_ = lp_mathlib_zsmulRec___redArg(v___f_192_, v___x_194_, v___y_183_, v___y_184_);
return v___x_195_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_ZMod_commRing___lam__1___boxed(lean_object* v_n_196_, lean_object* v___y_197_, lean_object* v___y_198_){
_start:
{
lean_object* v_res_199_; 
v_res_199_ = lp_mathlib_ZMod_commRing___lam__1(v_n_196_, v___y_197_, v___y_198_);
lean_dec(v___y_197_);
return v_res_199_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ZMod_commRing___lam__2(lean_object* v_n_200_, lean_object* v___y_201_, lean_object* v___y_202_){
_start:
{
lean_object* v_zero_203_; uint8_t v_isZero_204_; 
v_zero_203_ = lean_unsigned_to_nat(0u);
v_isZero_204_ = lean_nat_dec_eq(v_n_200_, v_zero_203_);
if (v_isZero_204_ == 1)
{
lean_object* v___x_205_; 
v___x_205_ = lean_int_sub(v___y_201_, v___y_202_);
return v___x_205_;
}
else
{
lean_object* v___x_206_; 
v___x_206_ = l_Fin_sub(v_n_200_, v___y_201_, v___y_202_);
return v___x_206_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_ZMod_commRing___lam__2___boxed(lean_object* v_n_207_, lean_object* v___y_208_, lean_object* v___y_209_){
_start:
{
lean_object* v_res_210_; 
v_res_210_ = lp_mathlib_ZMod_commRing___lam__2(v_n_207_, v___y_208_, v___y_209_);
lean_dec(v___y_209_);
lean_dec(v___y_208_);
lean_dec(v_n_207_);
return v_res_210_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ZMod_commRing___lam__3(lean_object* v_n_211_, lean_object* v___y_212_){
_start:
{
lean_object* v_zero_213_; uint8_t v_isZero_214_; 
v_zero_213_ = lean_unsigned_to_nat(0u);
v_isZero_214_ = lean_nat_dec_eq(v_n_211_, v_zero_213_);
if (v_isZero_214_ == 1)
{
lean_object* v___x_215_; 
v___x_215_ = lean_int_neg(v___y_212_);
return v___x_215_;
}
else
{
lean_object* v___x_216_; lean_object* v___x_217_; 
v___x_216_ = lean_nat_sub(v_n_211_, v___y_212_);
v___x_217_ = lean_nat_mod(v___x_216_, v_n_211_);
lean_dec(v___x_216_);
return v___x_217_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_ZMod_commRing___lam__3___boxed(lean_object* v_n_218_, lean_object* v___y_219_){
_start:
{
lean_object* v_res_220_; 
v_res_220_ = lp_mathlib_ZMod_commRing___lam__3(v_n_218_, v___y_219_);
lean_dec(v___y_219_);
lean_dec(v_n_218_);
return v_res_220_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ZMod_commRing___lam__4(lean_object* v_n_221_, lean_object* v___y_222_, lean_object* v___y_223_){
_start:
{
lean_object* v_zero_224_; uint8_t v_isZero_225_; 
v_zero_224_ = lean_unsigned_to_nat(0u);
v_isZero_225_ = lean_nat_dec_eq(v_n_221_, v_zero_224_);
if (v_isZero_225_ == 1)
{
lean_object* v___x_226_; 
lean_dec(v_n_221_);
v___x_226_ = l_Int_pow(v___y_223_, v___y_222_);
lean_dec(v___y_223_);
return v___x_226_;
}
else
{
lean_object* v___x_227_; lean_object* v_toOne_228_; lean_object* v___x_229_; lean_object* v___x_230_; 
lean_inc(v_n_221_);
v___x_227_ = lp_mathlib_Fin_instAddMonoidWithOne___redArg(v_n_221_);
v_toOne_228_ = lean_ctor_get(v___x_227_, 2);
lean_inc(v_toOne_228_);
lean_dec_ref(v___x_227_);
v___x_229_ = lean_alloc_closure((void*)(l_Fin_mul___boxed), 3, 1);
lean_closure_set(v___x_229_, 0, v_n_221_);
v___x_230_ = lp_mathlib_npowBinRec_go___redArg(v___x_229_, v___y_222_, v_toOne_228_, v___y_223_);
return v___x_230_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_ZMod_commRing___lam__4___boxed(lean_object* v_n_231_, lean_object* v___y_232_, lean_object* v___y_233_){
_start:
{
lean_object* v_res_234_; 
v_res_234_ = lp_mathlib_ZMod_commRing___lam__4(v_n_231_, v___y_232_, v___y_233_);
lean_dec(v___y_232_);
return v_res_234_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ZMod_commRing___lam__5(lean_object* v_n_235_, lean_object* v___y_236_, lean_object* v___y_237_){
_start:
{
lean_object* v_zero_238_; uint8_t v_isZero_239_; 
v_zero_238_ = lean_unsigned_to_nat(0u);
v_isZero_239_ = lean_nat_dec_eq(v_n_235_, v_zero_238_);
if (v_isZero_239_ == 1)
{
lean_object* v___x_240_; 
v___x_240_ = lean_int_mul(v___y_236_, v___y_237_);
return v___x_240_;
}
else
{
lean_object* v___x_241_; 
v___x_241_ = l_Fin_mul(v_n_235_, v___y_236_, v___y_237_);
return v___x_241_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_ZMod_commRing___lam__5___boxed(lean_object* v_n_242_, lean_object* v___y_243_, lean_object* v___y_244_){
_start:
{
lean_object* v_res_245_; 
v_res_245_ = lp_mathlib_ZMod_commRing___lam__5(v_n_242_, v___y_243_, v___y_244_);
lean_dec(v___y_244_);
lean_dec(v___y_243_);
lean_dec(v_n_242_);
return v_res_245_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ZMod_commRing___lam__6(lean_object* v_n_246_, lean_object* v___y_247_, lean_object* v___y_248_){
_start:
{
lean_object* v_zero_249_; uint8_t v_isZero_250_; 
v_zero_249_ = lean_unsigned_to_nat(0u);
v_isZero_250_ = lean_nat_dec_eq(v_n_246_, v_zero_249_);
if (v_isZero_250_ == 1)
{
lean_object* v___x_251_; lean_object* v___x_252_; 
lean_dec(v_n_246_);
v___x_251_ = lean_nat_to_int(v___y_247_);
v___x_252_ = lean_int_mul(v___x_251_, v___y_248_);
lean_dec(v___y_248_);
lean_dec(v___x_251_);
return v___x_252_;
}
else
{
lean_object* v___x_253_; lean_object* v___x_254_; lean_object* v___x_255_; 
v___x_253_ = lean_nat_mod(v_zero_249_, v_n_246_);
v___x_254_ = lean_alloc_closure((void*)(l_Fin_add___boxed), 3, 1);
lean_closure_set(v___x_254_, 0, v_n_246_);
v___x_255_ = l_nsmulRec___redArg(v___x_253_, v___x_254_, v___y_247_, v___y_248_);
lean_dec(v___y_247_);
lean_dec(v___x_253_);
return v___x_255_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_ZMod_commRing___lam__7(lean_object* v_n_256_, lean_object* v___y_257_, lean_object* v___y_258_){
_start:
{
lean_object* v_zero_259_; uint8_t v_isZero_260_; 
v_zero_259_ = lean_unsigned_to_nat(0u);
v_isZero_260_ = lean_nat_dec_eq(v_n_256_, v_zero_259_);
if (v_isZero_260_ == 1)
{
lean_object* v___x_261_; 
v___x_261_ = lean_int_add(v___y_257_, v___y_258_);
return v___x_261_;
}
else
{
lean_object* v___x_262_; 
v___x_262_ = l_Fin_add(v_n_256_, v___y_257_, v___y_258_);
return v___x_262_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_ZMod_commRing___lam__7___boxed(lean_object* v_n_263_, lean_object* v___y_264_, lean_object* v___y_265_){
_start:
{
lean_object* v_res_266_; 
v_res_266_ = lp_mathlib_ZMod_commRing___lam__7(v_n_263_, v___y_264_, v___y_265_);
lean_dec(v___y_265_);
lean_dec(v___y_264_);
lean_dec(v_n_263_);
return v_res_266_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ZMod_commRing___lam__8(lean_object* v_n_267_, lean_object* v___y_268_){
_start:
{
lean_object* v_zero_269_; uint8_t v_isZero_270_; 
v_zero_269_ = lean_unsigned_to_nat(0u);
v_isZero_270_ = lean_nat_dec_eq(v_n_267_, v_zero_269_);
if (v_isZero_270_ == 1)
{
lean_object* v___x_271_; 
lean_dec(v_n_267_);
v___x_271_ = lean_nat_to_int(v___y_268_);
return v___x_271_;
}
else
{
lean_object* v___x_272_; lean_object* v___x_273_; lean_object* v_toAddMonoidWithOne_274_; lean_object* v_toNatCast_275_; lean_object* v___x_276_; 
v___x_272_ = lp_mathlib_Fin_instCommRing___redArg(v_n_267_);
v___x_273_ = lp_mathlib_Ring_toAddGroupWithOne___redArg(v___x_272_);
v_toAddMonoidWithOne_274_ = lean_ctor_get(v___x_273_, 1);
lean_inc_ref(v_toAddMonoidWithOne_274_);
lean_dec_ref(v___x_273_);
v_toNatCast_275_ = lean_ctor_get(v_toAddMonoidWithOne_274_, 0);
lean_inc(v_toNatCast_275_);
lean_dec_ref(v_toAddMonoidWithOne_274_);
v___x_276_ = lean_apply_1(v_toNatCast_275_, v___y_268_);
return v___x_276_;
}
}
}
static lean_object* _init_lp_mathlib_ZMod_commRing___closed__0(void){
_start:
{
lean_object* v___x_277_; lean_object* v___x_278_; 
v___x_277_ = lean_unsigned_to_nat(1u);
v___x_278_ = lean_nat_to_int(v___x_277_);
return v___x_278_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ZMod_commRing(lean_object* v_n_279_){
_start:
{
lean_object* v___y_280_; lean_object* v___y_281_; lean_object* v___y_282_; lean_object* v___y_283_; lean_object* v___y_284_; lean_object* v___y_285_; lean_object* v___y_286_; lean_object* v___y_287_; lean_object* v___y_288_; lean_object* v___y_290_; lean_object* v___y_291_; lean_object* v___y_296_; lean_object* v_zero_303_; uint8_t v_isZero_304_; 
lean_inc_n(v_n_279_, 9);
v___y_280_ = lean_alloc_closure((void*)(lp_mathlib_ZMod_commRing___lam__0), 2, 1);
lean_closure_set(v___y_280_, 0, v_n_279_);
v___y_281_ = lean_alloc_closure((void*)(lp_mathlib_ZMod_commRing___lam__1___boxed), 3, 1);
lean_closure_set(v___y_281_, 0, v_n_279_);
v___y_282_ = lean_alloc_closure((void*)(lp_mathlib_ZMod_commRing___lam__2___boxed), 3, 1);
lean_closure_set(v___y_282_, 0, v_n_279_);
v___y_283_ = lean_alloc_closure((void*)(lp_mathlib_ZMod_commRing___lam__3___boxed), 2, 1);
lean_closure_set(v___y_283_, 0, v_n_279_);
v___y_284_ = lean_alloc_closure((void*)(lp_mathlib_ZMod_commRing___lam__4___boxed), 3, 1);
lean_closure_set(v___y_284_, 0, v_n_279_);
v___y_285_ = lean_alloc_closure((void*)(lp_mathlib_ZMod_commRing___lam__5___boxed), 3, 1);
lean_closure_set(v___y_285_, 0, v_n_279_);
v___y_286_ = lean_alloc_closure((void*)(lp_mathlib_ZMod_commRing___lam__6), 3, 1);
lean_closure_set(v___y_286_, 0, v_n_279_);
v___y_287_ = lean_alloc_closure((void*)(lp_mathlib_ZMod_commRing___lam__7___boxed), 3, 1);
lean_closure_set(v___y_287_, 0, v_n_279_);
v___y_288_ = lean_alloc_closure((void*)(lp_mathlib_ZMod_commRing___lam__8), 2, 1);
lean_closure_set(v___y_288_, 0, v_n_279_);
v_zero_303_ = lean_unsigned_to_nat(0u);
v_isZero_304_ = lean_nat_dec_eq(v_n_279_, v_zero_303_);
if (v_isZero_304_ == 1)
{
lean_object* v___x_305_; 
v___x_305_ = lean_obj_once(&lp_mathlib_ZMod_repr___aux__1___closed__0, &lp_mathlib_ZMod_repr___aux__1___closed__0_once, _init_lp_mathlib_ZMod_repr___aux__1___closed__0);
v___y_296_ = v___x_305_;
goto v___jp_295_;
}
else
{
lean_object* v___x_306_; 
v___x_306_ = lean_nat_mod(v_zero_303_, v_n_279_);
v___y_296_ = v___x_306_;
goto v___jp_295_;
}
v___jp_289_:
{
lean_object* v___x_292_; lean_object* v___x_293_; lean_object* v___x_294_; 
v___x_292_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_292_, 0, v___y_291_);
lean_ctor_set(v___x_292_, 1, v___y_285_);
lean_ctor_set(v___x_292_, 2, v___y_284_);
v___x_293_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_293_, 0, v___y_290_);
lean_ctor_set(v___x_293_, 1, v___x_292_);
lean_ctor_set(v___x_293_, 2, v___y_288_);
v___x_294_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v___x_294_, 0, v___x_293_);
lean_ctor_set(v___x_294_, 1, v___y_283_);
lean_ctor_set(v___x_294_, 2, v___y_282_);
lean_ctor_set(v___x_294_, 3, v___y_281_);
lean_ctor_set(v___x_294_, 4, v___y_280_);
return v___x_294_;
}
v___jp_295_:
{
lean_object* v___x_297_; lean_object* v_zero_298_; uint8_t v_isZero_299_; 
v___x_297_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_297_, 0, v___y_296_);
lean_ctor_set(v___x_297_, 1, v___y_287_);
lean_ctor_set(v___x_297_, 2, v___y_286_);
v_zero_298_ = lean_unsigned_to_nat(0u);
v_isZero_299_ = lean_nat_dec_eq(v_n_279_, v_zero_298_);
if (v_isZero_299_ == 1)
{
lean_object* v___x_300_; 
lean_dec(v_n_279_);
v___x_300_ = lean_obj_once(&lp_mathlib_ZMod_commRing___closed__0, &lp_mathlib_ZMod_commRing___closed__0_once, _init_lp_mathlib_ZMod_commRing___closed__0);
v___y_290_ = v___x_297_;
v___y_291_ = v___x_300_;
goto v___jp_289_;
}
else
{
lean_object* v___x_301_; lean_object* v___x_302_; 
v___x_301_ = lean_unsigned_to_nat(1u);
v___x_302_ = lean_nat_mod(v___x_301_, v_n_279_);
lean_dec(v_n_279_);
v___y_290_ = v___x_297_;
v___y_291_ = v___x_302_;
goto v___jp_289_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_ZMod_inhabited(lean_object* v_n_307_){
_start:
{
lean_object* v___x_308_; lean_object* v_toSemiring_309_; lean_object* v___x_310_; lean_object* v_toZero_311_; 
v___x_308_ = lp_mathlib_ZMod_commRing(v_n_307_);
v_toSemiring_309_ = lean_ctor_get(v___x_308_, 0);
lean_inc_ref(v_toSemiring_309_);
lean_dec_ref(v___x_308_);
v___x_310_ = lp_mathlib_instMulZeroClassOfSemiring___redArg(v_toSemiring_309_);
v_toZero_311_ = lean_ctor_get(v___x_310_, 1);
lean_inc(v_toZero_311_);
lean_dec_ref(v___x_310_);
return v_toZero_311_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Group_Fin_Basic(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_NeZero(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Ring_Int_Defs(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Ring_GrindInstances(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Data_Nat_ModEq(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Data_Fintype_EquivFin(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Ring_Nat(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_Data_ZMod_Defs(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Group_Fin_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_NeZero(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Ring_Int_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Ring_GrindInstances(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Data_Nat_ModEq(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Data_Fintype_EquivFin(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Ring_Nat(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
lp_mathlib_ZMod_instUnique = _init_lp_mathlib_ZMod_instUnique();
lean_mark_persistent(lp_mathlib_ZMod_instUnique);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_Data_ZMod_Defs(uint8_t builtin) {
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
lean_object* initialize_mathlib_Mathlib_Algebra_Group_Fin_Basic(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_NeZero(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Ring_Int_Defs(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Ring_GrindInstances(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Data_Nat_ModEq(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Data_Fintype_EquivFin(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Ring_Nat(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Data_ZMod_Defs(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Group_Fin_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_NeZero(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Ring_Int_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Ring_GrindInstances(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_Nat_ModEq(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_Fintype_EquivFin(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Ring_Nat(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Data_ZMod_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_Data_ZMod_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_Data_ZMod_Defs(builtin);
}
#ifdef __cplusplus
}
#endif
