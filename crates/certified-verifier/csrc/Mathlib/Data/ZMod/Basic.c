// Lean compiler output
// Module: Mathlib.Data.ZMod.Basic
// Imports: public import Init public meta import Init public import Mathlib.Algebra.CharP.Basic public import Mathlib.Algebra.GroupWithZero.Units.Fintype public import Mathlib.Algebra.Ring.Prod public import Mathlib.GroupTheory.GroupAction.SubMulAction public import Mathlib.GroupTheory.OrderOfElement public import Mathlib.Tactic.FinCases
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
lean_object* lp_mathlib_ZMod_commRing(lean_object*);
lean_object* lp_mathlib_Semiring_toNonAssocSemiring___redArg(lean_object*);
lean_object* lp_mathlib_NonAssocSemiring_toAddCommMonoidWithOne___redArg(lean_object*);
lean_object* lp_mathlib_Ring_toAddGroupWithOne___redArg(lean_object*);
lean_object* lean_nat_div(lean_object*, lean_object*);
uint8_t lean_nat_dec_eq(lean_object*, lean_object*);
lean_object* lp_mathlib_Equiv_refl(lean_object*);
lean_object* lean_nat_sub(lean_object*, lean_object*);
lean_object* lean_nat_add(lean_object*, lean_object*);
lean_object* lean_nat_abs(lean_object*);
lean_object* l_Int_sign(lean_object*);
lean_object* lp_mathlib_Nat_gcdA(lean_object*, lean_object*);
extern lean_object* lp_mathlib_Int_instCommRing;
lean_object* lp_mathlib_Equiv_subtypeEquivRight(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Int_castAddHom___redArg(lean_object*);
lean_object* lp_mathlib_AddMonoidHom_liftOfRightInverse___redArg(lean_object*, lean_object*);
lean_object* lp_mathlib_Equiv_trans___redArg(lean_object*, lean_object*);
lean_object* lp_mathlib_finCongr(lean_object*, lean_object*, lean_object*);
lean_object* lean_nat_mul(lean_object*, lean_object*);
lean_object* lp_mathlib_Nat_chineseRemainder_x27___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Monoid_toMulOneClass___redArg(lean_object*);
lean_object* lp_mathlib_MulOneClass_toMulOne___redArg(lean_object*);
static lean_once_cell_t lp_mathlib_ZMod_finEquiv___redArg___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_ZMod_finEquiv___redArg___closed__0;
LEAN_EXPORT lean_object* lp_mathlib_ZMod_finEquiv___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ZMod_finEquiv___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ZMod_finEquiv(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ZMod_finEquiv___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ZMod_val(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ZMod_val___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Data_ZMod_Basic_0__ZMod_val_match__1_splitter___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Data_ZMod_Basic_0__ZMod_val_match__1_splitter___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Data_ZMod_Basic_0__ZMod_val_match__1_splitter(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Data_ZMod_Basic_0__ZMod_val_match__1_splitter___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ZMod_cast___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ZMod_cast___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ZMod_cast(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ZMod_cast___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Data_ZMod_Basic_0__ZMod_match__1_splitter___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Data_ZMod_Basic_0__ZMod_match__1_splitter___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Data_ZMod_Basic_0__ZMod_match__1_splitter(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Data_ZMod_Basic_0__ZMod_match__1_splitter___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ZMod_castHom___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ZMod_castHom(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ZMod_castHom___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_once_cell_t lp_mathlib_MulEquiv_refl___at___00RingEquiv_refl___at___00ZMod_ringEquivCongr_spec__0_spec__0___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_MulEquiv_refl___at___00RingEquiv_refl___at___00ZMod_ringEquivCongr_spec__0_spec__0___closed__0;
LEAN_EXPORT lean_object* lp_mathlib_MulEquiv_refl___at___00RingEquiv_refl___at___00ZMod_ringEquivCongr_spec__0_spec__0;
LEAN_EXPORT lean_object* lp_mathlib_RingEquiv_refl___at___00ZMod_ringEquivCongr_spec__0;
LEAN_EXPORT lean_object* lp_mathlib_ZMod_ringEquivCongr___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ZMod_ringEquivCongr___redArg___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ZMod_ringEquivCongr(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ZMod_ringEquivCongr___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Int_cast___at___00ZMod_inv_spec__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ZMod_inv(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ZMod_inv___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ZMod_instInv(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Data_ZMod_Basic_0__Nat_xgcdAux_match__1_splitter___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Data_ZMod_Basic_0__Nat_xgcdAux_match__1_splitter___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Data_ZMod_Basic_0__Nat_xgcdAux_match__1_splitter(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Data_ZMod_Basic_0__Nat_xgcdAux_match__1_splitter___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Nat_cast___at___00ZMod_unitOfCoprime_spec__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ZMod_unitOfCoprime___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ZMod_unitOfCoprime(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ZMod_unitsEquivCoprime___redArg___lam__0(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ZMod_unitsEquivCoprime___redArg___lam__0___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ZMod_unitsEquivCoprime___redArg___lam__1(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ZMod_unitsEquivCoprime___redArg___lam__1___boxed(lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_ZMod_unitsEquivCoprime___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_ZMod_unitsEquivCoprime___redArg___lam__0___boxed, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_ZMod_unitsEquivCoprime___redArg___closed__0 = (const lean_object*)&lp_mathlib_ZMod_unitsEquivCoprime___redArg___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_ZMod_unitsEquivCoprime___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ZMod_unitsEquivCoprime(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RingHom_fst___at___00ZMod_chineseRemainder_spec__0___lam__0(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RingHom_fst___at___00ZMod_chineseRemainder_spec__0___lam__0___boxed(lean_object*);
static const lean_closure_object lp_mathlib_RingHom_fst___at___00ZMod_chineseRemainder_spec__0___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_RingHom_fst___at___00ZMod_chineseRemainder_spec__0___lam__0___boxed, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_RingHom_fst___at___00ZMod_chineseRemainder_spec__0___closed__0 = (const lean_object*)&lp_mathlib_RingHom_fst___at___00ZMod_chineseRemainder_spec__0___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_RingHom_fst___at___00ZMod_chineseRemainder_spec__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RingHom_fst___at___00ZMod_chineseRemainder_spec__0___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RingHom_snd___at___00ZMod_chineseRemainder_spec__2___lam__0(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RingHom_snd___at___00ZMod_chineseRemainder_spec__2___lam__0___boxed(lean_object*);
static const lean_closure_object lp_mathlib_RingHom_snd___at___00ZMod_chineseRemainder_spec__2___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_RingHom_snd___at___00ZMod_chineseRemainder_spec__2___lam__0___boxed, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_RingHom_snd___at___00ZMod_chineseRemainder_spec__2___closed__0 = (const lean_object*)&lp_mathlib_RingHom_snd___at___00ZMod_chineseRemainder_spec__2___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_RingHom_snd___at___00ZMod_chineseRemainder_spec__2(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RingHom_snd___at___00ZMod_chineseRemainder_spec__2___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Int_cast___at___00ZMod_cast___at___00ZMod_chineseRemainder_spec__1_spec__1(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ZMod_cast___at___00ZMod_chineseRemainder_spec__1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ZMod_cast___at___00ZMod_chineseRemainder_spec__1___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ZMod_chineseRemainder___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Nat_cast___at___00Nat_cast___at___00ZMod_cast___at___00ZMod_castHom___at___00ZMod_chineseRemainder_spec__3_spec__4_spec__6_spec__7(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Nat_cast___at___00ZMod_cast___at___00ZMod_castHom___at___00ZMod_chineseRemainder_spec__3_spec__4_spec__6(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Int_cast___at___00ZMod_cast___at___00ZMod_castHom___at___00ZMod_chineseRemainder_spec__3_spec__4_spec__5(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ZMod_cast___at___00ZMod_castHom___at___00ZMod_chineseRemainder_spec__3_spec__4(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ZMod_cast___at___00ZMod_castHom___at___00ZMod_chineseRemainder_spec__3_spec__4___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ZMod_castHom___at___00ZMod_chineseRemainder_spec__3___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ZMod_castHom___at___00ZMod_chineseRemainder_spec__3___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ZMod_castHom___at___00ZMod_chineseRemainder_spec__3___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ZMod_chineseRemainder___redArg___lam__1(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ZMod_chineseRemainder___redArg___lam__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ZMod_chineseRemainder___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ZMod_chineseRemainder(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ZMod_castHom___at___00ZMod_chineseRemainder_spec__3(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ZMod_castHom___at___00ZMod_chineseRemainder_spec__3___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_once_cell_t lp_mathlib_ZMod_instUniqueUnitsOfNatNat___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_ZMod_instUniqueUnitsOfNatNat___closed__0;
LEAN_EXPORT lean_object* lp_mathlib_ZMod_instUniqueUnitsOfNatNat;
static lean_once_cell_t lp_mathlib_ZMod_lift___redArg___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_ZMod_lift___redArg___closed__0;
static lean_once_cell_t lp_mathlib_ZMod_lift___redArg___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_ZMod_lift___redArg___closed__1;
LEAN_EXPORT lean_object* lp_mathlib_ZMod_lift___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ZMod_lift(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ZMod_lift___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddSubgroupClass_instZModSMul___redArg___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddSubgroupClass_instZModSMul___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddSubgroupClass_instZModSMul(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddSubgroupClass_instZModSMul___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddSubgroupClass_instZModModule___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddSubgroupClass_instZModModule(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddSubgroupClass_instZModModule___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Nat_residueClassesEquiv___redArg___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Nat_residueClassesEquiv___redArg___lam__0___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Nat_residueClassesEquiv___redArg___lam__1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Nat_residueClassesEquiv___redArg___lam__1___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Nat_residueClassesEquiv___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Nat_residueClassesEquiv(lean_object*, lean_object*);
static lean_object* _init_lp_mathlib_ZMod_finEquiv___redArg___closed__0(void){
_start:
{
lean_object* v___x_1_; 
v___x_1_ = lp_mathlib_Equiv_refl(lean_box(0));
return v___x_1_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ZMod_finEquiv___redArg(lean_object* v_x_2_){
_start:
{
lean_object* v_zero_3_; uint8_t v_isZero_4_; lean_object* v___x_5_; 
v_zero_3_ = lean_unsigned_to_nat(0u);
v_isZero_4_ = lean_nat_dec_eq(v_x_2_, v_zero_3_);
v___x_5_ = lean_obj_once(&lp_mathlib_ZMod_finEquiv___redArg___closed__0, &lp_mathlib_ZMod_finEquiv___redArg___closed__0_once, _init_lp_mathlib_ZMod_finEquiv___redArg___closed__0);
return v___x_5_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ZMod_finEquiv___redArg___boxed(lean_object* v_x_6_){
_start:
{
lean_object* v_res_7_; 
v_res_7_ = lp_mathlib_ZMod_finEquiv___redArg(v_x_6_);
lean_dec(v_x_6_);
return v_res_7_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ZMod_finEquiv(lean_object* v_x_8_, lean_object* v_x_9_){
_start:
{
lean_object* v___x_10_; 
v___x_10_ = lp_mathlib_ZMod_finEquiv___redArg(v_x_8_);
return v___x_10_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ZMod_finEquiv___boxed(lean_object* v_x_11_, lean_object* v_x_12_){
_start:
{
lean_object* v_res_13_; 
v_res_13_ = lp_mathlib_ZMod_finEquiv(v_x_11_, v_x_12_);
lean_dec(v_x_11_);
return v_res_13_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ZMod_val(lean_object* v_x_14_, lean_object* v_a_15_){
_start:
{
lean_object* v_zero_16_; uint8_t v_isZero_17_; 
v_zero_16_ = lean_unsigned_to_nat(0u);
v_isZero_17_ = lean_nat_dec_eq(v_x_14_, v_zero_16_);
if (v_isZero_17_ == 1)
{
lean_object* v___x_18_; 
v___x_18_ = lean_nat_abs(v_a_15_);
return v___x_18_;
}
else
{
lean_inc(v_a_15_);
return v_a_15_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_ZMod_val___boxed(lean_object* v_x_19_, lean_object* v_a_20_){
_start:
{
lean_object* v_res_21_; 
v_res_21_ = lp_mathlib_ZMod_val(v_x_19_, v_a_20_);
lean_dec(v_a_20_);
lean_dec(v_x_19_);
return v_res_21_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Data_ZMod_Basic_0__ZMod_val_match__1_splitter___redArg(lean_object* v_x_22_, lean_object* v_h__1_23_, lean_object* v_h__2_24_){
_start:
{
lean_object* v_zero_25_; uint8_t v_isZero_26_; 
v_zero_25_ = lean_unsigned_to_nat(0u);
v_isZero_26_ = lean_nat_dec_eq(v_x_22_, v_zero_25_);
if (v_isZero_26_ == 1)
{
lean_object* v___x_27_; lean_object* v___x_28_; 
lean_dec(v_h__2_24_);
v___x_27_ = lean_box(0);
v___x_28_ = lean_apply_1(v_h__1_23_, v___x_27_);
return v___x_28_;
}
else
{
lean_object* v_one_29_; lean_object* v_n_30_; lean_object* v___x_31_; 
lean_dec(v_h__1_23_);
v_one_29_ = lean_unsigned_to_nat(1u);
v_n_30_ = lean_nat_sub(v_x_22_, v_one_29_);
v___x_31_ = lean_apply_1(v_h__2_24_, v_n_30_);
return v___x_31_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Data_ZMod_Basic_0__ZMod_val_match__1_splitter___redArg___boxed(lean_object* v_x_32_, lean_object* v_h__1_33_, lean_object* v_h__2_34_){
_start:
{
lean_object* v_res_35_; 
v_res_35_ = lp_mathlib___private_Mathlib_Data_ZMod_Basic_0__ZMod_val_match__1_splitter___redArg(v_x_32_, v_h__1_33_, v_h__2_34_);
lean_dec(v_x_32_);
return v_res_35_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Data_ZMod_Basic_0__ZMod_val_match__1_splitter(lean_object* v_motive_36_, lean_object* v_x_37_, lean_object* v_h__1_38_, lean_object* v_h__2_39_){
_start:
{
lean_object* v_zero_40_; uint8_t v_isZero_41_; 
v_zero_40_ = lean_unsigned_to_nat(0u);
v_isZero_41_ = lean_nat_dec_eq(v_x_37_, v_zero_40_);
if (v_isZero_41_ == 1)
{
lean_object* v___x_42_; lean_object* v___x_43_; 
lean_dec(v_h__2_39_);
v___x_42_ = lean_box(0);
v___x_43_ = lean_apply_1(v_h__1_38_, v___x_42_);
return v___x_43_;
}
else
{
lean_object* v_one_44_; lean_object* v_n_45_; lean_object* v___x_46_; 
lean_dec(v_h__1_38_);
v_one_44_ = lean_unsigned_to_nat(1u);
v_n_45_ = lean_nat_sub(v_x_37_, v_one_44_);
v___x_46_ = lean_apply_1(v_h__2_39_, v_n_45_);
return v___x_46_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Data_ZMod_Basic_0__ZMod_val_match__1_splitter___boxed(lean_object* v_motive_47_, lean_object* v_x_48_, lean_object* v_h__1_49_, lean_object* v_h__2_50_){
_start:
{
lean_object* v_res_51_; 
v_res_51_ = lp_mathlib___private_Mathlib_Data_ZMod_Basic_0__ZMod_val_match__1_splitter(v_motive_47_, v_x_48_, v_h__1_49_, v_h__2_50_);
lean_dec(v_x_48_);
return v_res_51_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ZMod_cast___redArg(lean_object* v_inst_52_, lean_object* v_x_53_, lean_object* v_a_54_){
_start:
{
lean_object* v_toAddMonoidWithOne_55_; lean_object* v_toIntCast_56_; lean_object* v_toNatCast_57_; lean_object* v_zero_58_; uint8_t v_isZero_59_; 
v_toAddMonoidWithOne_55_ = lean_ctor_get(v_inst_52_, 1);
lean_inc_ref(v_toAddMonoidWithOne_55_);
v_toIntCast_56_ = lean_ctor_get(v_inst_52_, 0);
lean_inc(v_toIntCast_56_);
lean_dec_ref(v_inst_52_);
v_toNatCast_57_ = lean_ctor_get(v_toAddMonoidWithOne_55_, 0);
lean_inc(v_toNatCast_57_);
lean_dec_ref(v_toAddMonoidWithOne_55_);
v_zero_58_ = lean_unsigned_to_nat(0u);
v_isZero_59_ = lean_nat_dec_eq(v_x_53_, v_zero_58_);
if (v_isZero_59_ == 1)
{
lean_object* v___x_60_; 
lean_dec(v_toNatCast_57_);
v___x_60_ = lean_apply_1(v_toIntCast_56_, v_a_54_);
return v___x_60_;
}
else
{
lean_object* v_one_61_; lean_object* v_n_62_; lean_object* v___x_63_; lean_object* v___x_64_; lean_object* v___x_65_; 
lean_dec(v_toIntCast_56_);
v_one_61_ = lean_unsigned_to_nat(1u);
v_n_62_ = lean_nat_sub(v_x_53_, v_one_61_);
v___x_63_ = lean_nat_add(v_n_62_, v_one_61_);
lean_dec(v_n_62_);
v___x_64_ = lp_mathlib_ZMod_val(v___x_63_, v_a_54_);
lean_dec(v_a_54_);
lean_dec(v___x_63_);
v___x_65_ = lean_apply_1(v_toNatCast_57_, v___x_64_);
return v___x_65_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_ZMod_cast___redArg___boxed(lean_object* v_inst_66_, lean_object* v_x_67_, lean_object* v_a_68_){
_start:
{
lean_object* v_res_69_; 
v_res_69_ = lp_mathlib_ZMod_cast___redArg(v_inst_66_, v_x_67_, v_a_68_);
lean_dec(v_x_67_);
return v_res_69_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ZMod_cast(lean_object* v_R_70_, lean_object* v_inst_71_, lean_object* v_x_72_, lean_object* v_a_73_){
_start:
{
lean_object* v___x_74_; 
v___x_74_ = lp_mathlib_ZMod_cast___redArg(v_inst_71_, v_x_72_, v_a_73_);
return v___x_74_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ZMod_cast___boxed(lean_object* v_R_75_, lean_object* v_inst_76_, lean_object* v_x_77_, lean_object* v_a_78_){
_start:
{
lean_object* v_res_79_; 
v_res_79_ = lp_mathlib_ZMod_cast(v_R_75_, v_inst_76_, v_x_77_, v_a_78_);
lean_dec(v_x_77_);
return v_res_79_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Data_ZMod_Basic_0__ZMod_match__1_splitter___redArg(lean_object* v_x_80_, lean_object* v_h__1_81_, lean_object* v_h__2_82_){
_start:
{
lean_object* v_zero_83_; uint8_t v_isZero_84_; 
v_zero_83_ = lean_unsigned_to_nat(0u);
v_isZero_84_ = lean_nat_dec_eq(v_x_80_, v_zero_83_);
if (v_isZero_84_ == 1)
{
lean_object* v___x_85_; lean_object* v___x_86_; 
lean_dec(v_h__2_82_);
v___x_85_ = lean_box(0);
v___x_86_ = lean_apply_1(v_h__1_81_, v___x_85_);
return v___x_86_;
}
else
{
lean_object* v_one_87_; lean_object* v_n_88_; lean_object* v___x_89_; 
lean_dec(v_h__1_81_);
v_one_87_ = lean_unsigned_to_nat(1u);
v_n_88_ = lean_nat_sub(v_x_80_, v_one_87_);
v___x_89_ = lean_apply_1(v_h__2_82_, v_n_88_);
return v___x_89_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Data_ZMod_Basic_0__ZMod_match__1_splitter___redArg___boxed(lean_object* v_x_90_, lean_object* v_h__1_91_, lean_object* v_h__2_92_){
_start:
{
lean_object* v_res_93_; 
v_res_93_ = lp_mathlib___private_Mathlib_Data_ZMod_Basic_0__ZMod_match__1_splitter___redArg(v_x_90_, v_h__1_91_, v_h__2_92_);
lean_dec(v_x_90_);
return v_res_93_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Data_ZMod_Basic_0__ZMod_match__1_splitter(lean_object* v_motive_94_, lean_object* v_x_95_, lean_object* v_h__1_96_, lean_object* v_h__2_97_){
_start:
{
lean_object* v_zero_98_; uint8_t v_isZero_99_; 
v_zero_98_ = lean_unsigned_to_nat(0u);
v_isZero_99_ = lean_nat_dec_eq(v_x_95_, v_zero_98_);
if (v_isZero_99_ == 1)
{
lean_object* v___x_100_; lean_object* v___x_101_; 
lean_dec(v_h__2_97_);
v___x_100_ = lean_box(0);
v___x_101_ = lean_apply_1(v_h__1_96_, v___x_100_);
return v___x_101_;
}
else
{
lean_object* v_one_102_; lean_object* v_n_103_; lean_object* v___x_104_; 
lean_dec(v_h__1_96_);
v_one_102_ = lean_unsigned_to_nat(1u);
v_n_103_ = lean_nat_sub(v_x_95_, v_one_102_);
v___x_104_ = lean_apply_1(v_h__2_97_, v_n_103_);
return v___x_104_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Data_ZMod_Basic_0__ZMod_match__1_splitter___boxed(lean_object* v_motive_105_, lean_object* v_x_106_, lean_object* v_h__1_107_, lean_object* v_h__2_108_){
_start:
{
lean_object* v_res_109_; 
v_res_109_ = lp_mathlib___private_Mathlib_Data_ZMod_Basic_0__ZMod_match__1_splitter(v_motive_105_, v_x_106_, v_h__1_107_, v_h__2_108_);
lean_dec(v_x_106_);
return v_res_109_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ZMod_castHom___redArg(lean_object* v_n_110_, lean_object* v_inst_111_){
_start:
{
lean_object* v___x_112_; lean_object* v___x_113_; 
v___x_112_ = lp_mathlib_Ring_toAddGroupWithOne___redArg(v_inst_111_);
v___x_113_ = lean_alloc_closure((void*)(lp_mathlib_ZMod_cast___boxed), 4, 3);
lean_closure_set(v___x_113_, 0, lean_box(0));
lean_closure_set(v___x_113_, 1, v___x_112_);
lean_closure_set(v___x_113_, 2, v_n_110_);
return v___x_113_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ZMod_castHom(lean_object* v_n_114_, lean_object* v_m_115_, lean_object* v_h_116_, lean_object* v_R_117_, lean_object* v_inst_118_, lean_object* v_inst_119_){
_start:
{
lean_object* v___x_120_; 
v___x_120_ = lp_mathlib_ZMod_castHom___redArg(v_n_114_, v_inst_118_);
return v___x_120_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ZMod_castHom___boxed(lean_object* v_n_121_, lean_object* v_m_122_, lean_object* v_h_123_, lean_object* v_R_124_, lean_object* v_inst_125_, lean_object* v_inst_126_){
_start:
{
lean_object* v_res_127_; 
v_res_127_ = lp_mathlib_ZMod_castHom(v_n_121_, v_m_122_, v_h_123_, v_R_124_, v_inst_125_, v_inst_126_);
lean_dec(v_m_122_);
return v_res_127_;
}
}
static lean_object* _init_lp_mathlib_MulEquiv_refl___at___00RingEquiv_refl___at___00ZMod_ringEquivCongr_spec__0_spec__0___closed__0(void){
_start:
{
lean_object* v___x_128_; 
v___x_128_ = lp_mathlib_Equiv_refl(lean_box(0));
return v___x_128_;
}
}
static lean_object* _init_lp_mathlib_MulEquiv_refl___at___00RingEquiv_refl___at___00ZMod_ringEquivCongr_spec__0_spec__0(void){
_start:
{
lean_object* v___x_129_; 
v___x_129_ = lean_obj_once(&lp_mathlib_MulEquiv_refl___at___00RingEquiv_refl___at___00ZMod_ringEquivCongr_spec__0_spec__0___closed__0, &lp_mathlib_MulEquiv_refl___at___00RingEquiv_refl___at___00ZMod_ringEquivCongr_spec__0_spec__0___closed__0_once, _init_lp_mathlib_MulEquiv_refl___at___00RingEquiv_refl___at___00ZMod_ringEquivCongr_spec__0_spec__0___closed__0);
return v___x_129_;
}
}
static lean_object* _init_lp_mathlib_RingEquiv_refl___at___00ZMod_ringEquivCongr_spec__0(void){
_start:
{
lean_object* v___x_130_; 
v___x_130_ = lean_obj_once(&lp_mathlib_MulEquiv_refl___at___00RingEquiv_refl___at___00ZMod_ringEquivCongr_spec__0_spec__0___closed__0, &lp_mathlib_MulEquiv_refl___at___00RingEquiv_refl___at___00ZMod_ringEquivCongr_spec__0_spec__0___closed__0_once, _init_lp_mathlib_MulEquiv_refl___at___00RingEquiv_refl___at___00ZMod_ringEquivCongr_spec__0_spec__0___closed__0);
return v___x_130_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ZMod_ringEquivCongr___redArg(lean_object* v_m_131_, lean_object* v_n_132_){
_start:
{
lean_object* v_zero_133_; uint8_t v_isZero_134_; 
v_zero_133_ = lean_unsigned_to_nat(0u);
v_isZero_134_ = lean_nat_dec_eq(v_m_131_, v_zero_133_);
if (v_isZero_134_ == 1)
{
uint8_t v_isZero_135_; lean_object* v___x_136_; 
v_isZero_135_ = lean_nat_dec_eq(v_n_132_, v_zero_133_);
v___x_136_ = lean_obj_once(&lp_mathlib_MulEquiv_refl___at___00RingEquiv_refl___at___00ZMod_ringEquivCongr_spec__0_spec__0___closed__0, &lp_mathlib_MulEquiv_refl___at___00RingEquiv_refl___at___00ZMod_ringEquivCongr_spec__0_spec__0___closed__0_once, _init_lp_mathlib_MulEquiv_refl___at___00RingEquiv_refl___at___00ZMod_ringEquivCongr_spec__0_spec__0___closed__0);
return v___x_136_;
}
else
{
uint8_t v_isZero_137_; lean_object* v_one_138_; lean_object* v_n_139_; lean_object* v_n_140_; lean_object* v___x_141_; lean_object* v___x_142_; lean_object* v___x_143_; 
v_isZero_137_ = lean_nat_dec_eq(v_n_132_, v_zero_133_);
v_one_138_ = lean_unsigned_to_nat(1u);
v_n_139_ = lean_nat_sub(v_m_131_, v_one_138_);
v_n_140_ = lean_nat_sub(v_n_132_, v_one_138_);
v___x_141_ = lean_nat_add(v_n_139_, v_one_138_);
lean_dec(v_n_139_);
v___x_142_ = lean_nat_add(v_n_140_, v_one_138_);
lean_dec(v_n_140_);
v___x_143_ = lp_mathlib_finCongr(v___x_141_, v___x_142_, lean_box(0));
lean_dec(v___x_142_);
lean_dec(v___x_141_);
return v___x_143_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_ZMod_ringEquivCongr___redArg___boxed(lean_object* v_m_144_, lean_object* v_n_145_){
_start:
{
lean_object* v_res_146_; 
v_res_146_ = lp_mathlib_ZMod_ringEquivCongr___redArg(v_m_144_, v_n_145_);
lean_dec(v_n_145_);
lean_dec(v_m_144_);
return v_res_146_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ZMod_ringEquivCongr(lean_object* v_m_147_, lean_object* v_n_148_, lean_object* v_h_149_){
_start:
{
lean_object* v___x_150_; 
v___x_150_ = lp_mathlib_ZMod_ringEquivCongr___redArg(v_m_147_, v_n_148_);
return v___x_150_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ZMod_ringEquivCongr___boxed(lean_object* v_m_151_, lean_object* v_n_152_, lean_object* v_h_153_){
_start:
{
lean_object* v_res_154_; 
v_res_154_ = lp_mathlib_ZMod_ringEquivCongr(v_m_151_, v_n_152_, v_h_153_);
lean_dec(v_n_152_);
lean_dec(v_m_151_);
return v_res_154_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Int_cast___at___00ZMod_inv_spec__0(lean_object* v___x_155_, lean_object* v_a_156_){
_start:
{
lean_object* v___x_157_; lean_object* v___x_158_; lean_object* v_toIntCast_159_; lean_object* v___x_160_; 
v___x_157_ = lp_mathlib_ZMod_commRing(v___x_155_);
v___x_158_ = lp_mathlib_Ring_toAddGroupWithOne___redArg(v___x_157_);
v_toIntCast_159_ = lean_ctor_get(v___x_158_, 0);
lean_inc(v_toIntCast_159_);
lean_dec_ref(v___x_158_);
v___x_160_ = lean_apply_1(v_toIntCast_159_, v_a_156_);
return v___x_160_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ZMod_inv(lean_object* v_x_161_, lean_object* v_x_162_){
_start:
{
lean_object* v_zero_163_; uint8_t v_isZero_164_; 
v_zero_163_ = lean_unsigned_to_nat(0u);
v_isZero_164_ = lean_nat_dec_eq(v_x_161_, v_zero_163_);
if (v_isZero_164_ == 1)
{
lean_object* v___x_165_; 
v___x_165_ = l_Int_sign(v_x_162_);
return v___x_165_;
}
else
{
lean_object* v_one_166_; lean_object* v_n_167_; lean_object* v___x_168_; lean_object* v___x_169_; lean_object* v___x_170_; lean_object* v___x_171_; 
v_one_166_ = lean_unsigned_to_nat(1u);
v_n_167_ = lean_nat_sub(v_x_161_, v_one_166_);
v___x_168_ = lean_nat_add(v_n_167_, v_one_166_);
lean_dec(v_n_167_);
v___x_169_ = lp_mathlib_ZMod_val(v___x_168_, v_x_162_);
lean_inc(v___x_168_);
v___x_170_ = lp_mathlib_Nat_gcdA(v___x_169_, v___x_168_);
v___x_171_ = lp_mathlib_Int_cast___at___00ZMod_inv_spec__0(v___x_168_, v___x_170_);
return v___x_171_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_ZMod_inv___boxed(lean_object* v_x_172_, lean_object* v_x_173_){
_start:
{
lean_object* v_res_174_; 
v_res_174_ = lp_mathlib_ZMod_inv(v_x_172_, v_x_173_);
lean_dec(v_x_173_);
lean_dec(v_x_172_);
return v_res_174_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ZMod_instInv(lean_object* v_n_175_){
_start:
{
lean_object* v___x_176_; 
v___x_176_ = lean_alloc_closure((void*)(lp_mathlib_ZMod_inv___boxed), 2, 1);
lean_closure_set(v___x_176_, 0, v_n_175_);
return v___x_176_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Data_ZMod_Basic_0__Nat_xgcdAux_match__1_splitter___redArg(lean_object* v_n_177_, lean_object* v_ih_178_, lean_object* v_h__1_179_, lean_object* v_h__2_180_){
_start:
{
lean_object* v_zero_181_; uint8_t v_isZero_182_; 
v_zero_181_ = lean_unsigned_to_nat(0u);
v_isZero_182_ = lean_nat_dec_eq(v_n_177_, v_zero_181_);
if (v_isZero_182_ == 1)
{
lean_object* v___x_183_; 
lean_dec(v_h__2_180_);
v___x_183_ = lean_apply_1(v_h__1_179_, v_ih_178_);
return v___x_183_;
}
else
{
lean_object* v_one_184_; lean_object* v_n_185_; lean_object* v___x_186_; 
lean_dec(v_h__1_179_);
v_one_184_ = lean_unsigned_to_nat(1u);
v_n_185_ = lean_nat_sub(v_n_177_, v_one_184_);
v___x_186_ = lean_apply_2(v_h__2_180_, v_n_185_, v_ih_178_);
return v___x_186_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Data_ZMod_Basic_0__Nat_xgcdAux_match__1_splitter___redArg___boxed(lean_object* v_n_187_, lean_object* v_ih_188_, lean_object* v_h__1_189_, lean_object* v_h__2_190_){
_start:
{
lean_object* v_res_191_; 
v_res_191_ = lp_mathlib___private_Mathlib_Data_ZMod_Basic_0__Nat_xgcdAux_match__1_splitter___redArg(v_n_187_, v_ih_188_, v_h__1_189_, v_h__2_190_);
lean_dec(v_n_187_);
return v_res_191_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Data_ZMod_Basic_0__Nat_xgcdAux_match__1_splitter(lean_object* v_motive_192_, lean_object* v_n_193_, lean_object* v_ih_194_, lean_object* v_h__1_195_, lean_object* v_h__2_196_){
_start:
{
lean_object* v_zero_197_; uint8_t v_isZero_198_; 
v_zero_197_ = lean_unsigned_to_nat(0u);
v_isZero_198_ = lean_nat_dec_eq(v_n_193_, v_zero_197_);
if (v_isZero_198_ == 1)
{
lean_object* v___x_199_; 
lean_dec(v_h__2_196_);
v___x_199_ = lean_apply_1(v_h__1_195_, v_ih_194_);
return v___x_199_;
}
else
{
lean_object* v_one_200_; lean_object* v_n_201_; lean_object* v___x_202_; 
lean_dec(v_h__1_195_);
v_one_200_ = lean_unsigned_to_nat(1u);
v_n_201_ = lean_nat_sub(v_n_193_, v_one_200_);
v___x_202_ = lean_apply_2(v_h__2_196_, v_n_201_, v_ih_194_);
return v___x_202_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Data_ZMod_Basic_0__Nat_xgcdAux_match__1_splitter___boxed(lean_object* v_motive_203_, lean_object* v_n_204_, lean_object* v_ih_205_, lean_object* v_h__1_206_, lean_object* v_h__2_207_){
_start:
{
lean_object* v_res_208_; 
v_res_208_ = lp_mathlib___private_Mathlib_Data_ZMod_Basic_0__Nat_xgcdAux_match__1_splitter(v_motive_203_, v_n_204_, v_ih_205_, v_h__1_206_, v_h__2_207_);
lean_dec(v_n_204_);
return v_res_208_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Nat_cast___at___00ZMod_unitOfCoprime_spec__0(lean_object* v_n_209_, lean_object* v_a_210_){
_start:
{
lean_object* v___x_211_; lean_object* v___x_212_; lean_object* v_toAddMonoidWithOne_213_; lean_object* v_toNatCast_214_; lean_object* v___x_215_; 
v___x_211_ = lp_mathlib_ZMod_commRing(v_n_209_);
v___x_212_ = lp_mathlib_Ring_toAddGroupWithOne___redArg(v___x_211_);
v_toAddMonoidWithOne_213_ = lean_ctor_get(v___x_212_, 1);
lean_inc_ref(v_toAddMonoidWithOne_213_);
lean_dec_ref(v___x_212_);
v_toNatCast_214_ = lean_ctor_get(v_toAddMonoidWithOne_213_, 0);
lean_inc(v_toNatCast_214_);
lean_dec_ref(v_toAddMonoidWithOne_213_);
v___x_215_ = lean_apply_1(v_toNatCast_214_, v_a_210_);
return v___x_215_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ZMod_unitOfCoprime___redArg(lean_object* v_n_216_, lean_object* v_x_217_){
_start:
{
lean_object* v___x_218_; lean_object* v___x_219_; lean_object* v___x_220_; 
lean_inc(v_n_216_);
v___x_218_ = lp_mathlib_Nat_cast___at___00ZMod_unitOfCoprime_spec__0(v_n_216_, v_x_217_);
v___x_219_ = lp_mathlib_ZMod_inv(v_n_216_, v___x_218_);
lean_dec(v_n_216_);
v___x_220_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_220_, 0, v___x_218_);
lean_ctor_set(v___x_220_, 1, v___x_219_);
return v___x_220_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ZMod_unitOfCoprime(lean_object* v_n_221_, lean_object* v_x_222_, lean_object* v_h_223_){
_start:
{
lean_object* v___x_224_; 
v___x_224_ = lp_mathlib_ZMod_unitOfCoprime___redArg(v_n_221_, v_x_222_);
return v___x_224_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ZMod_unitsEquivCoprime___redArg___lam__0(lean_object* v_x_225_){
_start:
{
lean_object* v_val_226_; 
v_val_226_ = lean_ctor_get(v_x_225_, 0);
lean_inc(v_val_226_);
return v_val_226_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ZMod_unitsEquivCoprime___redArg___lam__0___boxed(lean_object* v_x_227_){
_start:
{
lean_object* v_res_228_; 
v_res_228_ = lp_mathlib_ZMod_unitsEquivCoprime___redArg___lam__0(v_x_227_);
lean_dec_ref(v_x_227_);
return v_res_228_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ZMod_unitsEquivCoprime___redArg___lam__1(lean_object* v_n_229_, lean_object* v_x_230_){
_start:
{
lean_object* v___x_231_; lean_object* v___x_232_; 
v___x_231_ = lp_mathlib_ZMod_val(v_n_229_, v_x_230_);
v___x_232_ = lp_mathlib_ZMod_unitOfCoprime___redArg(v_n_229_, v___x_231_);
return v___x_232_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ZMod_unitsEquivCoprime___redArg___lam__1___boxed(lean_object* v_n_233_, lean_object* v_x_234_){
_start:
{
lean_object* v_res_235_; 
v_res_235_ = lp_mathlib_ZMod_unitsEquivCoprime___redArg___lam__1(v_n_233_, v_x_234_);
lean_dec(v_x_234_);
return v_res_235_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ZMod_unitsEquivCoprime___redArg(lean_object* v_n_237_){
_start:
{
lean_object* v___f_238_; lean_object* v___f_239_; lean_object* v___x_240_; 
v___f_238_ = ((lean_object*)(lp_mathlib_ZMod_unitsEquivCoprime___redArg___closed__0));
v___f_239_ = lean_alloc_closure((void*)(lp_mathlib_ZMod_unitsEquivCoprime___redArg___lam__1___boxed), 2, 1);
lean_closure_set(v___f_239_, 0, v_n_237_);
v___x_240_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_240_, 0, v___f_238_);
lean_ctor_set(v___x_240_, 1, v___f_239_);
return v___x_240_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ZMod_unitsEquivCoprime(lean_object* v_n_241_, lean_object* v_inst_242_){
_start:
{
lean_object* v___x_243_; 
v___x_243_ = lp_mathlib_ZMod_unitsEquivCoprime___redArg(v_n_241_);
return v___x_243_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RingHom_fst___at___00ZMod_chineseRemainder_spec__0___lam__0(lean_object* v_self_244_){
_start:
{
lean_object* v_fst_245_; 
v_fst_245_ = lean_ctor_get(v_self_244_, 0);
lean_inc(v_fst_245_);
return v_fst_245_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RingHom_fst___at___00ZMod_chineseRemainder_spec__0___lam__0___boxed(lean_object* v_self_246_){
_start:
{
lean_object* v_res_247_; 
v_res_247_ = lp_mathlib_RingHom_fst___at___00ZMod_chineseRemainder_spec__0___lam__0(v_self_246_);
lean_dec_ref(v_self_246_);
return v_res_247_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RingHom_fst___at___00ZMod_chineseRemainder_spec__0(lean_object* v_m_249_, lean_object* v_n_250_){
_start:
{
lean_object* v___f_251_; 
v___f_251_ = ((lean_object*)(lp_mathlib_RingHom_fst___at___00ZMod_chineseRemainder_spec__0___closed__0));
return v___f_251_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RingHom_fst___at___00ZMod_chineseRemainder_spec__0___boxed(lean_object* v_m_252_, lean_object* v_n_253_){
_start:
{
lean_object* v_res_254_; 
v_res_254_ = lp_mathlib_RingHom_fst___at___00ZMod_chineseRemainder_spec__0(v_m_252_, v_n_253_);
lean_dec(v_n_253_);
lean_dec(v_m_252_);
return v_res_254_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RingHom_snd___at___00ZMod_chineseRemainder_spec__2___lam__0(lean_object* v_self_255_){
_start:
{
lean_object* v_snd_256_; 
v_snd_256_ = lean_ctor_get(v_self_255_, 1);
lean_inc(v_snd_256_);
return v_snd_256_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RingHom_snd___at___00ZMod_chineseRemainder_spec__2___lam__0___boxed(lean_object* v_self_257_){
_start:
{
lean_object* v_res_258_; 
v_res_258_ = lp_mathlib_RingHom_snd___at___00ZMod_chineseRemainder_spec__2___lam__0(v_self_257_);
lean_dec_ref(v_self_257_);
return v_res_258_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RingHom_snd___at___00ZMod_chineseRemainder_spec__2(lean_object* v_m_260_, lean_object* v_n_261_){
_start:
{
lean_object* v___f_262_; 
v___f_262_ = ((lean_object*)(lp_mathlib_RingHom_snd___at___00ZMod_chineseRemainder_spec__2___closed__0));
return v___f_262_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RingHom_snd___at___00ZMod_chineseRemainder_spec__2___boxed(lean_object* v_m_263_, lean_object* v_n_264_){
_start:
{
lean_object* v_res_265_; 
v_res_265_ = lp_mathlib_RingHom_snd___at___00ZMod_chineseRemainder_spec__2(v_m_263_, v_n_264_);
lean_dec(v_n_264_);
lean_dec(v_m_263_);
return v_res_265_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Int_cast___at___00ZMod_cast___at___00ZMod_chineseRemainder_spec__1_spec__1(lean_object* v___x_266_, lean_object* v_a_267_){
_start:
{
lean_object* v___x_268_; lean_object* v___x_269_; lean_object* v_toIntCast_270_; lean_object* v___x_271_; 
v___x_268_ = lp_mathlib_ZMod_commRing(v___x_266_);
v___x_269_ = lp_mathlib_Ring_toAddGroupWithOne___redArg(v___x_268_);
v_toIntCast_270_ = lean_ctor_get(v___x_269_, 0);
lean_inc(v_toIntCast_270_);
lean_dec_ref(v___x_269_);
v___x_271_ = lean_apply_1(v_toIntCast_270_, v_a_267_);
return v___x_271_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ZMod_cast___at___00ZMod_chineseRemainder_spec__1(lean_object* v___x_272_, lean_object* v_x_273_, lean_object* v_a_274_){
_start:
{
lean_object* v_zero_275_; uint8_t v_isZero_276_; 
v_zero_275_ = lean_unsigned_to_nat(0u);
v_isZero_276_ = lean_nat_dec_eq(v_x_273_, v_zero_275_);
if (v_isZero_276_ == 1)
{
lean_object* v___x_277_; 
v___x_277_ = lp_mathlib_Int_cast___at___00ZMod_cast___at___00ZMod_chineseRemainder_spec__1_spec__1(v___x_272_, v_a_274_);
return v___x_277_;
}
else
{
lean_object* v_one_278_; lean_object* v_n_279_; lean_object* v___x_280_; lean_object* v___x_281_; lean_object* v___x_282_; 
v_one_278_ = lean_unsigned_to_nat(1u);
v_n_279_ = lean_nat_sub(v_x_273_, v_one_278_);
v___x_280_ = lean_nat_add(v_n_279_, v_one_278_);
lean_dec(v_n_279_);
v___x_281_ = lp_mathlib_ZMod_val(v___x_280_, v_a_274_);
lean_dec(v_a_274_);
lean_dec(v___x_280_);
v___x_282_ = lp_mathlib_Nat_cast___at___00ZMod_unitOfCoprime_spec__0(v___x_272_, v___x_281_);
return v___x_282_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_ZMod_cast___at___00ZMod_chineseRemainder_spec__1___boxed(lean_object* v___x_283_, lean_object* v_x_284_, lean_object* v_a_285_){
_start:
{
lean_object* v_res_286_; 
v_res_286_ = lp_mathlib_ZMod_cast___at___00ZMod_chineseRemainder_spec__1(v___x_283_, v_x_284_, v_a_285_);
lean_dec(v_x_284_);
return v_res_286_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ZMod_chineseRemainder___redArg___lam__0(lean_object* v___x_287_, lean_object* v_m_288_, lean_object* v_n_289_, lean_object* v_x_290_){
_start:
{
lean_object* v___x_291_; uint8_t v___x_292_; 
v___x_291_ = lean_unsigned_to_nat(0u);
v___x_292_ = lean_nat_dec_eq(v___x_287_, v___x_291_);
if (v___x_292_ == 0)
{
lean_object* v_fst_293_; lean_object* v_snd_294_; lean_object* v___x_295_; lean_object* v___x_296_; lean_object* v___x_297_; lean_object* v___x_298_; 
v_fst_293_ = lean_ctor_get(v_x_290_, 0);
lean_inc(v_fst_293_);
v_snd_294_ = lean_ctor_get(v_x_290_, 1);
lean_inc(v_snd_294_);
lean_dec_ref(v_x_290_);
v___x_295_ = lp_mathlib_ZMod_val(v_m_288_, v_fst_293_);
lean_dec(v_fst_293_);
v___x_296_ = lp_mathlib_ZMod_val(v_n_289_, v_snd_294_);
lean_dec(v_snd_294_);
v___x_297_ = lp_mathlib_Nat_chineseRemainder_x27___redArg(v_n_289_, v_m_288_, v___x_295_, v___x_296_);
v___x_298_ = lp_mathlib_Nat_cast___at___00ZMod_unitOfCoprime_spec__0(v___x_287_, v___x_297_);
return v___x_298_;
}
else
{
lean_object* v___x_299_; uint8_t v___x_300_; 
v___x_299_ = lean_unsigned_to_nat(1u);
v___x_300_ = lean_nat_dec_eq(v_m_288_, v___x_299_);
if (v___x_300_ == 0)
{
lean_object* v_fst_301_; lean_object* v___x_302_; 
lean_dec(v_n_289_);
v_fst_301_ = lean_ctor_get(v_x_290_, 0);
lean_inc(v_fst_301_);
lean_dec_ref(v_x_290_);
v___x_302_ = lp_mathlib_ZMod_cast___at___00ZMod_chineseRemainder_spec__1(v___x_287_, v_m_288_, v_fst_301_);
lean_dec(v_m_288_);
return v___x_302_;
}
else
{
lean_object* v_snd_303_; lean_object* v___x_304_; 
lean_dec(v_m_288_);
v_snd_303_ = lean_ctor_get(v_x_290_, 1);
lean_inc(v_snd_303_);
lean_dec_ref(v_x_290_);
v___x_304_ = lp_mathlib_ZMod_cast___at___00ZMod_chineseRemainder_spec__1(v___x_287_, v_n_289_, v_snd_303_);
lean_dec(v_n_289_);
return v___x_304_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Nat_cast___at___00Nat_cast___at___00ZMod_cast___at___00ZMod_castHom___at___00ZMod_chineseRemainder_spec__3_spec__4_spec__6_spec__7(lean_object* v_m_305_, lean_object* v_a_306_){
_start:
{
lean_object* v___x_307_; lean_object* v_toSemiring_308_; lean_object* v___x_309_; lean_object* v___x_310_; lean_object* v_toNatCast_311_; lean_object* v___x_312_; 
v___x_307_ = lp_mathlib_ZMod_commRing(v_m_305_);
v_toSemiring_308_ = lean_ctor_get(v___x_307_, 0);
lean_inc_ref(v_toSemiring_308_);
lean_dec_ref(v___x_307_);
v___x_309_ = lp_mathlib_Semiring_toNonAssocSemiring___redArg(v_toSemiring_308_);
v___x_310_ = lp_mathlib_NonAssocSemiring_toAddCommMonoidWithOne___redArg(v___x_309_);
v_toNatCast_311_ = lean_ctor_get(v___x_310_, 0);
lean_inc(v_toNatCast_311_);
lean_dec_ref(v___x_310_);
v___x_312_ = lean_apply_1(v_toNatCast_311_, v_a_306_);
return v___x_312_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Nat_cast___at___00ZMod_cast___at___00ZMod_castHom___at___00ZMod_chineseRemainder_spec__3_spec__4_spec__6(lean_object* v_m_313_, lean_object* v_n_314_, lean_object* v_a_315_){
_start:
{
lean_object* v___x_316_; lean_object* v___x_317_; lean_object* v___x_318_; 
lean_inc(v_a_315_);
v___x_316_ = lp_mathlib_Nat_cast___at___00Nat_cast___at___00ZMod_cast___at___00ZMod_castHom___at___00ZMod_chineseRemainder_spec__3_spec__4_spec__6_spec__7(v_m_313_, v_a_315_);
v___x_317_ = lp_mathlib_Nat_cast___at___00Nat_cast___at___00ZMod_cast___at___00ZMod_castHom___at___00ZMod_chineseRemainder_spec__3_spec__4_spec__6_spec__7(v_n_314_, v_a_315_);
v___x_318_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_318_, 0, v___x_316_);
lean_ctor_set(v___x_318_, 1, v___x_317_);
return v___x_318_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Int_cast___at___00ZMod_cast___at___00ZMod_castHom___at___00ZMod_chineseRemainder_spec__3_spec__4_spec__5(lean_object* v_m_319_, lean_object* v_n_320_, lean_object* v_a_321_){
_start:
{
lean_object* v___x_322_; lean_object* v___x_323_; lean_object* v___x_324_; 
lean_inc(v_a_321_);
v___x_322_ = lp_mathlib_Int_cast___at___00ZMod_cast___at___00ZMod_chineseRemainder_spec__1_spec__1(v_m_319_, v_a_321_);
v___x_323_ = lp_mathlib_Int_cast___at___00ZMod_cast___at___00ZMod_chineseRemainder_spec__1_spec__1(v_n_320_, v_a_321_);
v___x_324_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_324_, 0, v___x_322_);
lean_ctor_set(v___x_324_, 1, v___x_323_);
return v___x_324_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ZMod_cast___at___00ZMod_castHom___at___00ZMod_chineseRemainder_spec__3_spec__4(lean_object* v_m_325_, lean_object* v_n_326_, lean_object* v_x_327_, lean_object* v_a_328_){
_start:
{
lean_object* v_zero_329_; uint8_t v_isZero_330_; 
v_zero_329_ = lean_unsigned_to_nat(0u);
v_isZero_330_ = lean_nat_dec_eq(v_x_327_, v_zero_329_);
if (v_isZero_330_ == 1)
{
lean_object* v___x_331_; 
v___x_331_ = lp_mathlib_Int_cast___at___00ZMod_cast___at___00ZMod_castHom___at___00ZMod_chineseRemainder_spec__3_spec__4_spec__5(v_m_325_, v_n_326_, v_a_328_);
return v___x_331_;
}
else
{
lean_object* v_one_332_; lean_object* v_n_333_; lean_object* v___x_334_; lean_object* v___x_335_; lean_object* v___x_336_; 
v_one_332_ = lean_unsigned_to_nat(1u);
v_n_333_ = lean_nat_sub(v_x_327_, v_one_332_);
v___x_334_ = lean_nat_add(v_n_333_, v_one_332_);
lean_dec(v_n_333_);
v___x_335_ = lp_mathlib_ZMod_val(v___x_334_, v_a_328_);
lean_dec(v_a_328_);
lean_dec(v___x_334_);
v___x_336_ = lp_mathlib_Nat_cast___at___00ZMod_cast___at___00ZMod_castHom___at___00ZMod_chineseRemainder_spec__3_spec__4_spec__6(v_m_325_, v_n_326_, v___x_335_);
return v___x_336_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_ZMod_cast___at___00ZMod_castHom___at___00ZMod_chineseRemainder_spec__3_spec__4___boxed(lean_object* v_m_337_, lean_object* v_n_338_, lean_object* v_x_339_, lean_object* v_a_340_){
_start:
{
lean_object* v_res_341_; 
v_res_341_ = lp_mathlib_ZMod_cast___at___00ZMod_castHom___at___00ZMod_chineseRemainder_spec__3_spec__4(v_m_337_, v_n_338_, v_x_339_, v_a_340_);
lean_dec(v_x_339_);
return v_res_341_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ZMod_castHom___at___00ZMod_chineseRemainder_spec__3___redArg___lam__0(lean_object* v_m_342_, lean_object* v_n_343_, lean_object* v_n_344_, lean_object* v___y_345_){
_start:
{
lean_object* v___x_346_; 
v___x_346_ = lp_mathlib_ZMod_cast___at___00ZMod_castHom___at___00ZMod_chineseRemainder_spec__3_spec__4(v_m_342_, v_n_343_, v_n_344_, v___y_345_);
return v___x_346_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ZMod_castHom___at___00ZMod_chineseRemainder_spec__3___redArg___lam__0___boxed(lean_object* v_m_347_, lean_object* v_n_348_, lean_object* v_n_349_, lean_object* v___y_350_){
_start:
{
lean_object* v_res_351_; 
v_res_351_ = lp_mathlib_ZMod_castHom___at___00ZMod_chineseRemainder_spec__3___redArg___lam__0(v_m_347_, v_n_348_, v_n_349_, v___y_350_);
lean_dec(v_n_349_);
return v_res_351_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ZMod_castHom___at___00ZMod_chineseRemainder_spec__3___redArg(lean_object* v_m_352_, lean_object* v_n_353_, lean_object* v_n_354_){
_start:
{
lean_object* v___f_355_; 
v___f_355_ = lean_alloc_closure((void*)(lp_mathlib_ZMod_castHom___at___00ZMod_chineseRemainder_spec__3___redArg___lam__0___boxed), 4, 3);
lean_closure_set(v___f_355_, 0, v_m_352_);
lean_closure_set(v___f_355_, 1, v_n_353_);
lean_closure_set(v___f_355_, 2, v_n_354_);
return v___f_355_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ZMod_chineseRemainder___redArg___lam__1(lean_object* v_m_356_, lean_object* v_n_357_, lean_object* v___x_358_, lean_object* v___y_359_){
_start:
{
lean_object* v___x_360_; 
v___x_360_ = lp_mathlib_ZMod_cast___at___00ZMod_castHom___at___00ZMod_chineseRemainder_spec__3_spec__4(v_m_356_, v_n_357_, v___x_358_, v___y_359_);
return v___x_360_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ZMod_chineseRemainder___redArg___lam__1___boxed(lean_object* v_m_361_, lean_object* v_n_362_, lean_object* v___x_363_, lean_object* v___y_364_){
_start:
{
lean_object* v_res_365_; 
v_res_365_ = lp_mathlib_ZMod_chineseRemainder___redArg___lam__1(v_m_361_, v_n_362_, v___x_363_, v___y_364_);
lean_dec(v___x_363_);
return v_res_365_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ZMod_chineseRemainder___redArg(lean_object* v_m_366_, lean_object* v_n_367_){
_start:
{
lean_object* v___x_368_; lean_object* v_inv__fun_369_; lean_object* v_to__fun_370_; lean_object* v___x_371_; 
v___x_368_ = lean_nat_mul(v_m_366_, v_n_367_);
lean_inc(v_n_367_);
lean_inc(v_m_366_);
lean_inc(v___x_368_);
v_inv__fun_369_ = lean_alloc_closure((void*)(lp_mathlib_ZMod_chineseRemainder___redArg___lam__0), 4, 3);
lean_closure_set(v_inv__fun_369_, 0, v___x_368_);
lean_closure_set(v_inv__fun_369_, 1, v_m_366_);
lean_closure_set(v_inv__fun_369_, 2, v_n_367_);
v_to__fun_370_ = lean_alloc_closure((void*)(lp_mathlib_ZMod_chineseRemainder___redArg___lam__1___boxed), 4, 3);
lean_closure_set(v_to__fun_370_, 0, v_m_366_);
lean_closure_set(v_to__fun_370_, 1, v_n_367_);
lean_closure_set(v_to__fun_370_, 2, v___x_368_);
v___x_371_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_371_, 0, v_to__fun_370_);
lean_ctor_set(v___x_371_, 1, v_inv__fun_369_);
return v___x_371_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ZMod_chineseRemainder(lean_object* v_m_372_, lean_object* v_n_373_, lean_object* v_h_374_){
_start:
{
lean_object* v___x_375_; 
v___x_375_ = lp_mathlib_ZMod_chineseRemainder___redArg(v_m_372_, v_n_373_);
return v___x_375_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ZMod_castHom___at___00ZMod_chineseRemainder_spec__3(lean_object* v_m_376_, lean_object* v_n_377_, lean_object* v_n_378_, lean_object* v_m_379_, lean_object* v_h_380_, lean_object* v_inst_381_){
_start:
{
lean_object* v___f_382_; 
v___f_382_ = lean_alloc_closure((void*)(lp_mathlib_ZMod_castHom___at___00ZMod_chineseRemainder_spec__3___redArg___lam__0___boxed), 4, 3);
lean_closure_set(v___f_382_, 0, v_m_376_);
lean_closure_set(v___f_382_, 1, v_n_377_);
lean_closure_set(v___f_382_, 2, v_n_378_);
return v___f_382_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ZMod_castHom___at___00ZMod_chineseRemainder_spec__3___boxed(lean_object* v_m_383_, lean_object* v_n_384_, lean_object* v_n_385_, lean_object* v_m_386_, lean_object* v_h_387_, lean_object* v_inst_388_){
_start:
{
lean_object* v_res_389_; 
v_res_389_ = lp_mathlib_ZMod_castHom___at___00ZMod_chineseRemainder_spec__3(v_m_383_, v_n_384_, v_n_385_, v_m_386_, v_h_387_, v_inst_388_);
lean_dec(v_m_386_);
return v_res_389_;
}
}
static lean_object* _init_lp_mathlib_ZMod_instUniqueUnitsOfNatNat___closed__0(void){
_start:
{
lean_object* v___x_390_; lean_object* v___x_391_; 
v___x_390_ = lean_unsigned_to_nat(2u);
v___x_391_ = lp_mathlib_ZMod_commRing(v___x_390_);
return v___x_391_;
}
}
static lean_object* _init_lp_mathlib_ZMod_instUniqueUnitsOfNatNat(void){
_start:
{
lean_object* v___x_392_; lean_object* v_toSemiring_393_; lean_object* v_toMonoid_394_; lean_object* v___x_395_; lean_object* v___x_396_; lean_object* v_toOne_397_; lean_object* v___x_399_; uint8_t v_isShared_400_; uint8_t v_isSharedCheck_404_; 
v___x_392_ = lean_obj_once(&lp_mathlib_ZMod_instUniqueUnitsOfNatNat___closed__0, &lp_mathlib_ZMod_instUniqueUnitsOfNatNat___closed__0_once, _init_lp_mathlib_ZMod_instUniqueUnitsOfNatNat___closed__0);
v_toSemiring_393_ = lean_ctor_get(v___x_392_, 0);
v_toMonoid_394_ = lean_ctor_get(v_toSemiring_393_, 1);
v___x_395_ = lp_mathlib_Monoid_toMulOneClass___redArg(v_toMonoid_394_);
v___x_396_ = lp_mathlib_MulOneClass_toMulOne___redArg(v___x_395_);
v_toOne_397_ = lean_ctor_get(v___x_396_, 0);
v_isSharedCheck_404_ = !lean_is_exclusive(v___x_396_);
if (v_isSharedCheck_404_ == 0)
{
lean_object* v_unused_405_; 
v_unused_405_ = lean_ctor_get(v___x_396_, 1);
lean_dec(v_unused_405_);
v___x_399_ = v___x_396_;
v_isShared_400_ = v_isSharedCheck_404_;
goto v_resetjp_398_;
}
else
{
lean_inc(v_toOne_397_);
lean_dec(v___x_396_);
v___x_399_ = lean_box(0);
v_isShared_400_ = v_isSharedCheck_404_;
goto v_resetjp_398_;
}
v_resetjp_398_:
{
lean_object* v___x_402_; 
lean_inc(v_toOne_397_);
if (v_isShared_400_ == 0)
{
lean_ctor_set(v___x_399_, 1, v_toOne_397_);
v___x_402_ = v___x_399_;
goto v_reusejp_401_;
}
else
{
lean_object* v_reuseFailAlloc_403_; 
v_reuseFailAlloc_403_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_403_, 0, v_toOne_397_);
lean_ctor_set(v_reuseFailAlloc_403_, 1, v_toOne_397_);
v___x_402_ = v_reuseFailAlloc_403_;
goto v_reusejp_401_;
}
v_reusejp_401_:
{
return v___x_402_;
}
}
}
}
static lean_object* _init_lp_mathlib_ZMod_lift___redArg___closed__0(void){
_start:
{
lean_object* v___x_406_; lean_object* v___x_407_; 
v___x_406_ = lp_mathlib_Int_instCommRing;
v___x_407_ = lp_mathlib_Ring_toAddGroupWithOne___redArg(v___x_406_);
return v___x_407_;
}
}
static lean_object* _init_lp_mathlib_ZMod_lift___redArg___closed__1(void){
_start:
{
lean_object* v___x_408_; 
v___x_408_ = lp_mathlib_Equiv_subtypeEquivRight(lean_box(0), lean_box(0), lean_box(0), lean_box(0));
return v___x_408_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ZMod_lift___redArg(lean_object* v_n_409_){
_start:
{
lean_object* v___x_410_; lean_object* v___x_411_; lean_object* v___x_412_; lean_object* v___x_413_; lean_object* v___x_414_; lean_object* v___x_415_; lean_object* v___x_416_; lean_object* v___x_417_; 
v___x_410_ = lean_obj_once(&lp_mathlib_ZMod_lift___redArg___closed__0, &lp_mathlib_ZMod_lift___redArg___closed__0_once, _init_lp_mathlib_ZMod_lift___redArg___closed__0);
v___x_411_ = lean_obj_once(&lp_mathlib_ZMod_lift___redArg___closed__1, &lp_mathlib_ZMod_lift___redArg___closed__1_once, _init_lp_mathlib_ZMod_lift___redArg___closed__1);
lean_inc(v_n_409_);
v___x_412_ = lp_mathlib_ZMod_commRing(v_n_409_);
v___x_413_ = lp_mathlib_Ring_toAddGroupWithOne___redArg(v___x_412_);
v___x_414_ = lp_mathlib_Int_castAddHom___redArg(v___x_413_);
v___x_415_ = lean_alloc_closure((void*)(lp_mathlib_ZMod_cast___boxed), 4, 3);
lean_closure_set(v___x_415_, 0, lean_box(0));
lean_closure_set(v___x_415_, 1, v___x_410_);
lean_closure_set(v___x_415_, 2, v_n_409_);
v___x_416_ = lp_mathlib_AddMonoidHom_liftOfRightInverse___redArg(v___x_414_, v___x_415_);
v___x_417_ = lp_mathlib_Equiv_trans___redArg(v___x_411_, v___x_416_);
return v___x_417_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ZMod_lift(lean_object* v_n_418_, lean_object* v_A_419_, lean_object* v_inst_420_){
_start:
{
lean_object* v___x_421_; 
v___x_421_ = lp_mathlib_ZMod_lift___redArg(v_n_418_);
return v___x_421_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ZMod_lift___boxed(lean_object* v_n_422_, lean_object* v_A_423_, lean_object* v_inst_424_){
_start:
{
lean_object* v_res_425_; 
v_res_425_ = lp_mathlib_ZMod_lift(v_n_422_, v_A_423_, v_inst_424_);
lean_dec_ref(v_inst_424_);
return v_res_425_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddSubgroupClass_instZModSMul___redArg___lam__0(lean_object* v_inst_426_, lean_object* v_a_427_, lean_object* v_x_428_){
_start:
{
lean_object* v___x_429_; 
v___x_429_ = lean_apply_2(v_inst_426_, v_a_427_, v_x_428_);
return v___x_429_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddSubgroupClass_instZModSMul___redArg(lean_object* v_inst_430_){
_start:
{
lean_object* v___f_431_; 
v___f_431_ = lean_alloc_closure((void*)(lp_mathlib_AddSubgroupClass_instZModSMul___redArg___lam__0), 3, 1);
lean_closure_set(v___f_431_, 0, v_inst_430_);
return v___f_431_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddSubgroupClass_instZModSMul(lean_object* v_n_432_, lean_object* v_S_433_, lean_object* v_G_434_, lean_object* v_inst_435_, lean_object* v_inst_436_, lean_object* v_inst_437_, lean_object* v_K_438_, lean_object* v_inst_439_){
_start:
{
lean_object* v___f_440_; 
v___f_440_ = lean_alloc_closure((void*)(lp_mathlib_AddSubgroupClass_instZModSMul___redArg___lam__0), 3, 1);
lean_closure_set(v___f_440_, 0, v_inst_439_);
return v___f_440_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddSubgroupClass_instZModSMul___boxed(lean_object* v_n_441_, lean_object* v_S_442_, lean_object* v_G_443_, lean_object* v_inst_444_, lean_object* v_inst_445_, lean_object* v_inst_446_, lean_object* v_K_447_, lean_object* v_inst_448_){
_start:
{
lean_object* v_res_449_; 
v_res_449_ = lp_mathlib_AddSubgroupClass_instZModSMul(v_n_441_, v_S_442_, v_G_443_, v_inst_444_, v_inst_445_, v_inst_446_, v_K_447_, v_inst_448_);
lean_dec(v_K_447_);
lean_dec_ref(v_inst_444_);
lean_dec(v_n_441_);
return v_res_449_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddSubgroupClass_instZModModule___redArg(lean_object* v_inst_450_){
_start:
{
lean_object* v___f_451_; 
v___f_451_ = lean_alloc_closure((void*)(lp_mathlib_AddSubgroupClass_instZModSMul___redArg___lam__0), 3, 1);
lean_closure_set(v___f_451_, 0, v_inst_450_);
return v___f_451_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddSubgroupClass_instZModModule(lean_object* v_n_452_, lean_object* v_S_453_, lean_object* v_G_454_, lean_object* v_inst_455_, lean_object* v_inst_456_, lean_object* v_inst_457_, lean_object* v_K_458_, lean_object* v_inst_459_){
_start:
{
lean_object* v___f_460_; 
v___f_460_ = lean_alloc_closure((void*)(lp_mathlib_AddSubgroupClass_instZModSMul___redArg___lam__0), 3, 1);
lean_closure_set(v___f_460_, 0, v_inst_459_);
return v___f_460_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddSubgroupClass_instZModModule___boxed(lean_object* v_n_461_, lean_object* v_S_462_, lean_object* v_G_463_, lean_object* v_inst_464_, lean_object* v_inst_465_, lean_object* v_inst_466_, lean_object* v_K_467_, lean_object* v_inst_468_){
_start:
{
lean_object* v_res_469_; 
v_res_469_ = lp_mathlib_AddSubgroupClass_instZModModule(v_n_461_, v_S_462_, v_G_463_, v_inst_464_, v_inst_465_, v_inst_466_, v_K_467_, v_inst_468_);
lean_dec(v_K_467_);
lean_dec_ref(v_inst_464_);
lean_dec(v_n_461_);
return v_res_469_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Nat_residueClassesEquiv___redArg___lam__0(lean_object* v_N_470_, lean_object* v_p_471_){
_start:
{
lean_object* v_fst_472_; lean_object* v_snd_473_; lean_object* v___x_474_; lean_object* v___x_475_; lean_object* v___x_476_; 
v_fst_472_ = lean_ctor_get(v_p_471_, 0);
v_snd_473_ = lean_ctor_get(v_p_471_, 1);
v___x_474_ = lp_mathlib_ZMod_val(v_N_470_, v_fst_472_);
v___x_475_ = lean_nat_mul(v_N_470_, v_snd_473_);
v___x_476_ = lean_nat_add(v___x_474_, v___x_475_);
lean_dec(v___x_475_);
lean_dec(v___x_474_);
return v___x_476_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Nat_residueClassesEquiv___redArg___lam__0___boxed(lean_object* v_N_477_, lean_object* v_p_478_){
_start:
{
lean_object* v_res_479_; 
v_res_479_ = lp_mathlib_Nat_residueClassesEquiv___redArg___lam__0(v_N_477_, v_p_478_);
lean_dec_ref(v_p_478_);
lean_dec(v_N_477_);
return v_res_479_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Nat_residueClassesEquiv___redArg___lam__1(lean_object* v_toNatCast_480_, lean_object* v_N_481_, lean_object* v_n_482_){
_start:
{
lean_object* v___x_483_; lean_object* v___x_484_; lean_object* v___x_485_; 
lean_inc(v_n_482_);
v___x_483_ = lean_apply_1(v_toNatCast_480_, v_n_482_);
v___x_484_ = lean_nat_div(v_n_482_, v_N_481_);
lean_dec(v_n_482_);
v___x_485_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_485_, 0, v___x_483_);
lean_ctor_set(v___x_485_, 1, v___x_484_);
return v___x_485_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Nat_residueClassesEquiv___redArg___lam__1___boxed(lean_object* v_toNatCast_486_, lean_object* v_N_487_, lean_object* v_n_488_){
_start:
{
lean_object* v_res_489_; 
v_res_489_ = lp_mathlib_Nat_residueClassesEquiv___redArg___lam__1(v_toNatCast_486_, v_N_487_, v_n_488_);
lean_dec(v_N_487_);
return v_res_489_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Nat_residueClassesEquiv___redArg(lean_object* v_N_490_){
_start:
{
lean_object* v___x_491_; lean_object* v___x_492_; lean_object* v_toAddMonoidWithOne_493_; lean_object* v_toNatCast_494_; lean_object* v___f_495_; lean_object* v___f_496_; lean_object* v___x_497_; 
lean_inc_n(v_N_490_, 2);
v___x_491_ = lp_mathlib_ZMod_commRing(v_N_490_);
v___x_492_ = lp_mathlib_Ring_toAddGroupWithOne___redArg(v___x_491_);
v_toAddMonoidWithOne_493_ = lean_ctor_get(v___x_492_, 1);
lean_inc_ref(v_toAddMonoidWithOne_493_);
lean_dec_ref(v___x_492_);
v_toNatCast_494_ = lean_ctor_get(v_toAddMonoidWithOne_493_, 0);
lean_inc(v_toNatCast_494_);
lean_dec_ref(v_toAddMonoidWithOne_493_);
v___f_495_ = lean_alloc_closure((void*)(lp_mathlib_Nat_residueClassesEquiv___redArg___lam__0___boxed), 2, 1);
lean_closure_set(v___f_495_, 0, v_N_490_);
v___f_496_ = lean_alloc_closure((void*)(lp_mathlib_Nat_residueClassesEquiv___redArg___lam__1___boxed), 3, 2);
lean_closure_set(v___f_496_, 0, v_toNatCast_494_);
lean_closure_set(v___f_496_, 1, v_N_490_);
v___x_497_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_497_, 0, v___f_496_);
lean_ctor_set(v___x_497_, 1, v___f_495_);
return v___x_497_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Nat_residueClassesEquiv(lean_object* v_N_498_, lean_object* v_inst_499_){
_start:
{
lean_object* v___x_500_; 
v___x_500_ = lp_mathlib_Nat_residueClassesEquiv___redArg(v_N_498_);
return v___x_500_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_CharP_Basic(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_GroupWithZero_Units_Fintype(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Ring_Prod(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_GroupTheory_GroupAction_SubMulAction(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_GroupTheory_OrderOfElement(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Tactic_FinCases(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_Data_ZMod_Basic(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_CharP_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_GroupWithZero_Units_Fintype(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Ring_Prod(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_GroupTheory_GroupAction_SubMulAction(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_GroupTheory_OrderOfElement(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Tactic_FinCases(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
lp_mathlib_MulEquiv_refl___at___00RingEquiv_refl___at___00ZMod_ringEquivCongr_spec__0_spec__0 = _init_lp_mathlib_MulEquiv_refl___at___00RingEquiv_refl___at___00ZMod_ringEquivCongr_spec__0_spec__0();
lean_mark_persistent(lp_mathlib_MulEquiv_refl___at___00RingEquiv_refl___at___00ZMod_ringEquivCongr_spec__0_spec__0);
lp_mathlib_RingEquiv_refl___at___00ZMod_ringEquivCongr_spec__0 = _init_lp_mathlib_RingEquiv_refl___at___00ZMod_ringEquivCongr_spec__0();
lean_mark_persistent(lp_mathlib_RingEquiv_refl___at___00ZMod_ringEquivCongr_spec__0);
lp_mathlib_ZMod_instUniqueUnitsOfNatNat = _init_lp_mathlib_ZMod_instUniqueUnitsOfNatNat();
lean_mark_persistent(lp_mathlib_ZMod_instUniqueUnitsOfNatNat);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_Data_ZMod_Basic(uint8_t builtin) {
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
lean_object* initialize_mathlib_Mathlib_Algebra_CharP_Basic(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_GroupWithZero_Units_Fintype(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Ring_Prod(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_GroupTheory_GroupAction_SubMulAction(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_GroupTheory_OrderOfElement(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Tactic_FinCases(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Data_ZMod_Basic(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_CharP_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_GroupWithZero_Units_Fintype(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Ring_Prod(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_GroupTheory_GroupAction_SubMulAction(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_GroupTheory_OrderOfElement(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Tactic_FinCases(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Data_ZMod_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_Data_ZMod_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_Data_ZMod_Basic(builtin);
}
#ifdef __cplusplus
}
#endif
