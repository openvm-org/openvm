// Lean compiler output
// Module: Mathlib.Algebra.Order.GroupWithZero.Canonical
// Imports: public import Init public meta import Init public import Mathlib.Algebra.Group.WithOne.Map public import Mathlib.Algebra.GroupWithZero.InjSurj public import Mathlib.Algebra.GroupWithZero.Regular public import Mathlib.Algebra.GroupWithZero.WithZero public import Mathlib.Algebra.Order.AddGroupWithTop public import Mathlib.Algebra.Order.Group.Defs public import Mathlib.Algebra.Order.Group.Int public import Mathlib.Algebra.Order.Group.Units public import Mathlib.Algebra.Order.GroupWithZero.Basic public import Mathlib.Algebra.Order.Monoid.OrderDual public import Mathlib.Algebra.Order.Monoid.TypeTags public import Mathlib.Data.Int.Basic public import Mathlib.Data.Set.Function
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
uint8_t l_Option_instDecidableEq___redArg(lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_LinearOrderedAddCommGroupWithTop_instLinearOrderedAddCommMonoidWithTop___redArg(lean_object*);
lean_object* lp_mathlib_OrderDual_instAddMonoid___redArg(lean_object*);
lean_object* lp_mathlib_Multiplicative_monoid___redArg(lean_object*);
lean_object* lp_mathlib_OrderDual_instLinearOrder___redArg(lean_object*);
lean_object* lp_mathlib_Multiplicative_linearOrder___redArg(lean_object*);
lean_object* lp_mathlib_Equiv_refl(lean_object*);
lean_object* lp_mathlib_Multiplicative_ofAdd(lean_object*);
lean_object* lp_mathlib_LinearOrderedAddCommGroupWithTop_toSubNegMonoid___redArg(lean_object*);
lean_object* lp_mathlib_OrderDual_instSubNegAddMonoid___redArg(lean_object*);
lean_object* lp_mathlib_Multiplicative_divInvMonoid___redArg(lean_object*);
lean_object* lp_mathlib_Additive_addMonoid___redArg(lean_object*);
lean_object* lp_mathlib_Additive_linearOrder___redArg(lean_object*);
lean_object* lp_mathlib_CommGroupWithZero_toGroupWithZero___redArg(lean_object*);
lean_object* lp_mathlib_GroupWithZero_toDivInvMonoid___redArg(lean_object*);
lean_object* lp_mathlib_Additive_subNegMonoid___redArg(lean_object*);
lean_object* lp_mathlib_WithZero_expEquiv___redArg(lean_object*);
lean_object* lp_mathlib_WithZero_map_u2082(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_WithZero_instCommMonoidWithZero___redArg(lean_object*);
lean_object* lp_mathlib_LinearOrder_toLattice___redArg(lean_object*);
lean_object* lp_mathlib_Lattice_toSemilatticeInf___redArg(lean_object*);
lean_object* lp_mathlib_SemilatticeInf_toMin___redArg___lam__0(lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_SemilatticeSup_toMax___redArg___lam__0(lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_WithZero_instCommGroupWithZero___redArg(lean_object*);
lean_object* lp_mathlib_OrderDual_instMonoid___redArg(lean_object*);
lean_object* lp_mathlib_WithZero_logEquiv___redArg(lean_object*);
lean_object* lp_mathlib_OrderDual_instDivInvMonoid___redArg(lean_object*);
extern lean_object* lp_mathlib_Int_instLinearOrder;
extern lean_object* lp_mathlib_Int_instAddCommGroup;
LEAN_EXPORT lean_object* lp_mathlib_LinearOrderedCommGroupWithZero_toCommGroupWithZero___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LinearOrderedCommGroupWithZero_toCommGroupWithZero(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_linearOrderedCommMonoidWithZero___redArg___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_linearOrderedCommMonoidWithZero___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_linearOrderedCommMonoidWithZero(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_linearOrderedCommMonoidWithZero___boxed(lean_object**);
LEAN_EXPORT lean_object* lp_mathlib_instLinearOrderedAddCommMonoidWithTopAdditiveOrderDual___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instLinearOrderedAddCommMonoidWithTopAdditiveOrderDual(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instLinearOrderedAddCommMonoidWithTopOrderDualAdditive___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instLinearOrderedAddCommMonoidWithTopOrderDualAdditive(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instLinearOrderedAddCommGroupWithTopAdditiveOrderDual___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instLinearOrderedAddCommGroupWithTopAdditiveOrderDual(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instLinearOrderedAddCommGroupWithTopOrderDualAdditive___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instLinearOrderedAddCommGroupWithTopOrderDualAdditive(lean_object*, lean_object*);
static lean_once_cell_t lp_mathlib_instLinearOrderedCommMonoidWithZeroMultiplicativeOrderDual___redArg___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_instLinearOrderedCommMonoidWithZeroMultiplicativeOrderDual___redArg___closed__0;
static lean_once_cell_t lp_mathlib_instLinearOrderedCommMonoidWithZeroMultiplicativeOrderDual___redArg___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_instLinearOrderedCommMonoidWithZeroMultiplicativeOrderDual___redArg___closed__1;
LEAN_EXPORT lean_object* lp_mathlib_instLinearOrderedCommMonoidWithZeroMultiplicativeOrderDual___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instLinearOrderedCommMonoidWithZeroMultiplicativeOrderDual(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instLinearOrderedCommGroupWithZeroMultiplicativeOrderDualOfLinearOrderedAddCommGroupWithTop___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instLinearOrderedCommGroupWithZeroMultiplicativeOrderDualOfLinearOrderedAddCommGroupWithTop___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instLinearOrderedCommGroupWithZeroMultiplicativeOrderDualOfLinearOrderedAddCommGroupWithTop(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instLinearOrderedCommGroupWithZeroMultiplicativeOrderDualOfLinearOrderedAddCommGroupWithTop___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_WithZero_instBot(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_WithZero_le(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_WithZero_instOrderBot(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_WithZero_instBoundedOrder___aux__1___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_WithZero_instBoundedOrder___aux__1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_WithZero_instBoundedOrder___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_WithZero_instBoundedOrder(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_WithZero_instLT(lean_object*, lean_object*);
static const lean_ctor_object lp_mathlib_WithZero_instPreorder___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_WithZero_instPreorder___closed__0 = (const lean_object*)&lp_mathlib_WithZero_instPreorder___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_WithZero_instPreorder(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_WithZero_instPreorder___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_WithZero_instPartialOrder___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_WithZero_instPartialOrder___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_WithZero_instPartialOrder(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_WithZero_instPartialOrder___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_WithZero_semilatticeSup___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_WithZero_semilatticeSup___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_WithZero_semilatticeSup___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_WithZero_semilatticeSup(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_WithZero_semilatticeInf___redArg___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_WithZero_semilatticeInf___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_WithZero_semilatticeInf(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_WithZero_instLattice___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_WithZero_instLattice(lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_WithZero_decidableEq___aux__1___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_WithZero_decidableEq___aux__1___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_WithZero_decidableEq___aux__1(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_WithZero_decidableEq___aux__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_WithZero_decidableEq___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_WithZero_decidableEq___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_WithZero_decidableEq(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_WithZero_decidableEq___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_WithZero_decidableLE___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_WithZero_decidableLE___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_WithZero_decidableLE(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_WithZero_decidableLE___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_WithZero_decidableLT___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_WithZero_decidableLT___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_WithZero_decidableLT(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_WithZero_decidableLT___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_WithZero_instLinearOrder___redArg___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_WithZero_instLinearOrder___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_WithZero_instLinearOrder___redArg___lam__1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_WithZero_instLinearOrder___redArg___lam__1___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_WithZero_instLinearOrder___redArg___lam__2(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_WithZero_instLinearOrder___redArg___lam__2___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_WithZero_instLinearOrder___redArg___lam__3(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_WithZero_instLinearOrder___redArg___lam__3___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_WithZero_instLinearOrder___redArg___lam__4(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_WithZero_instLinearOrder___redArg___lam__4___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_WithZero_instLinearOrder___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_WithZero_instLinearOrder(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_WithZero_instLinearOrderedCommMonoidWithZero___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_WithZero_instLinearOrderedCommMonoidWithZero(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_WithZero_instLinearOrderedCommGroupWithZero___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_WithZero_instLinearOrderedCommGroupWithZero(lean_object*, lean_object*, lean_object*, lean_object*);
static lean_once_cell_t lp_mathlib_WithZero_instLinearOrderedCommGroupWithZeroMultiplicativeInt___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_WithZero_instLinearOrderedCommGroupWithZeroMultiplicativeInt___closed__0;
static lean_once_cell_t lp_mathlib_WithZero_instLinearOrderedCommGroupWithZeroMultiplicativeInt___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_WithZero_instLinearOrderedCommGroupWithZeroMultiplicativeInt___closed__1;
static lean_once_cell_t lp_mathlib_WithZero_instLinearOrderedCommGroupWithZeroMultiplicativeInt___closed__2_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_WithZero_instLinearOrderedCommGroupWithZeroMultiplicativeInt___closed__2;
LEAN_EXPORT lean_object* lp_mathlib_WithZero_instLinearOrderedCommGroupWithZeroMultiplicativeInt;
LEAN_EXPORT lean_object* lp_mathlib_WithZero_expOrderIso___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_WithZero_expOrderIso(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_WithZero_expOrderIso___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_WithZero_logOrderIso___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_WithZero_logOrderIso(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_WithZero_logOrderIso___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LinearOrderedCommGroupWithZero_toCommGroupWithZero___redArg(lean_object* v_self_1_){
_start:
{
lean_object* v_toLinearOrderedCommMonoidWithZero_2_; lean_object* v_toInv_3_; lean_object* v_toDiv_4_; lean_object* v_toZPow_5_; lean_object* v___x_7_; uint8_t v_isShared_8_; uint8_t v_isSharedCheck_13_; 
v_toLinearOrderedCommMonoidWithZero_2_ = lean_ctor_get(v_self_1_, 0);
v_toInv_3_ = lean_ctor_get(v_self_1_, 1);
v_toDiv_4_ = lean_ctor_get(v_self_1_, 2);
v_toZPow_5_ = lean_ctor_get(v_self_1_, 3);
v_isSharedCheck_13_ = !lean_is_exclusive(v_self_1_);
if (v_isSharedCheck_13_ == 0)
{
v___x_7_ = v_self_1_;
v_isShared_8_ = v_isSharedCheck_13_;
goto v_resetjp_6_;
}
else
{
lean_inc(v_toZPow_5_);
lean_inc(v_toDiv_4_);
lean_inc(v_toInv_3_);
lean_inc(v_toLinearOrderedCommMonoidWithZero_2_);
lean_dec(v_self_1_);
v___x_7_ = lean_box(0);
v_isShared_8_ = v_isSharedCheck_13_;
goto v_resetjp_6_;
}
v_resetjp_6_:
{
lean_object* v_toCommMonoidWithZero_9_; lean_object* v___x_11_; 
v_toCommMonoidWithZero_9_ = lean_ctor_get(v_toLinearOrderedCommMonoidWithZero_2_, 0);
lean_inc_ref(v_toCommMonoidWithZero_9_);
lean_dec_ref(v_toLinearOrderedCommMonoidWithZero_2_);
if (v_isShared_8_ == 0)
{
lean_ctor_set(v___x_7_, 0, v_toCommMonoidWithZero_9_);
v___x_11_ = v___x_7_;
goto v_reusejp_10_;
}
else
{
lean_object* v_reuseFailAlloc_12_; 
v_reuseFailAlloc_12_ = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(v_reuseFailAlloc_12_, 0, v_toCommMonoidWithZero_9_);
lean_ctor_set(v_reuseFailAlloc_12_, 1, v_toInv_3_);
lean_ctor_set(v_reuseFailAlloc_12_, 2, v_toDiv_4_);
lean_ctor_set(v_reuseFailAlloc_12_, 3, v_toZPow_5_);
v___x_11_ = v_reuseFailAlloc_12_;
goto v_reusejp_10_;
}
v_reusejp_10_:
{
return v___x_11_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_LinearOrderedCommGroupWithZero_toCommGroupWithZero(lean_object* v_00_u03b1_14_, lean_object* v_self_15_){
_start:
{
lean_object* v___x_16_; 
v___x_16_ = lp_mathlib_LinearOrderedCommGroupWithZero_toCommGroupWithZero___redArg(v_self_15_);
return v___x_16_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_linearOrderedCommMonoidWithZero___redArg___lam__0(lean_object* v_inst_17_, lean_object* v_n_18_, lean_object* v_x_19_){
_start:
{
lean_object* v___x_20_; 
v___x_20_ = lean_apply_2(v_inst_17_, v_x_19_, v_n_18_);
return v___x_20_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_linearOrderedCommMonoidWithZero___redArg(lean_object* v_inst_21_, lean_object* v_inst_22_, lean_object* v_inst_23_, lean_object* v_inst_24_, lean_object* v_inst_25_, lean_object* v_inst_26_, lean_object* v_inst_27_, lean_object* v_inst_28_, lean_object* v_inst_29_, lean_object* v_inst_30_, lean_object* v_inst_31_, lean_object* v_inst_32_, lean_object* v_inst_33_){
_start:
{
lean_object* v___f_34_; lean_object* v___x_35_; lean_object* v___x_36_; lean_object* v___x_37_; lean_object* v___x_38_; lean_object* v___x_39_; 
v___f_34_ = lean_alloc_closure((void*)(lp_mathlib_Function_Injective_linearOrderedCommMonoidWithZero___redArg___lam__0), 3, 1);
lean_closure_set(v___f_34_, 0, v_inst_25_);
v___x_35_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_35_, 0, v_inst_26_);
lean_ctor_set(v___x_35_, 1, v_inst_27_);
v___x_36_ = lean_alloc_ctor(0, 7, 0);
lean_ctor_set(v___x_36_, 0, v___x_35_);
lean_ctor_set(v___x_36_, 1, v_inst_29_);
lean_ctor_set(v___x_36_, 2, v_inst_28_);
lean_ctor_set(v___x_36_, 3, v_inst_30_);
lean_ctor_set(v___x_36_, 4, v_inst_32_);
lean_ctor_set(v___x_36_, 5, v_inst_31_);
lean_ctor_set(v___x_36_, 6, v_inst_33_);
v___x_37_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_37_, 0, v_inst_23_);
lean_ctor_set(v___x_37_, 1, v_inst_24_);
lean_ctor_set(v___x_37_, 2, v___f_34_);
v___x_38_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_38_, 0, v___x_37_);
lean_ctor_set(v___x_38_, 1, v_inst_21_);
v___x_39_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_39_, 0, v___x_38_);
lean_ctor_set(v___x_39_, 1, v___x_36_);
lean_ctor_set(v___x_39_, 2, v_inst_22_);
return v___x_39_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_linearOrderedCommMonoidWithZero(lean_object* v_00_u03b1_40_, lean_object* v_inst_41_, lean_object* v_00_u03b2_42_, lean_object* v_inst_43_, lean_object* v_inst_44_, lean_object* v_inst_45_, lean_object* v_inst_46_, lean_object* v_inst_47_, lean_object* v_inst_48_, lean_object* v_inst_49_, lean_object* v_inst_50_, lean_object* v_inst_51_, lean_object* v_inst_52_, lean_object* v_inst_53_, lean_object* v_inst_54_, lean_object* v_inst_55_, lean_object* v_f_56_, lean_object* v_hf_57_, lean_object* v_zero_58_, lean_object* v_one_59_, lean_object* v_mul_60_, lean_object* v_npow_61_, lean_object* v_le_62_, lean_object* v_lt_63_, lean_object* v_hsup_64_, lean_object* v_hinf_65_, lean_object* v_bot_66_, lean_object* v_compare_67_){
_start:
{
lean_object* v___f_68_; lean_object* v___x_69_; lean_object* v___x_70_; lean_object* v___x_71_; lean_object* v___x_72_; lean_object* v___x_73_; 
v___f_68_ = lean_alloc_closure((void*)(lp_mathlib_Function_Injective_linearOrderedCommMonoidWithZero___redArg___lam__0), 3, 1);
lean_closure_set(v___f_68_, 0, v_inst_47_);
v___x_69_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_69_, 0, v_inst_48_);
lean_ctor_set(v___x_69_, 1, v_inst_49_);
v___x_70_ = lean_alloc_ctor(0, 7, 0);
lean_ctor_set(v___x_70_, 0, v___x_69_);
lean_ctor_set(v___x_70_, 1, v_inst_51_);
lean_ctor_set(v___x_70_, 2, v_inst_50_);
lean_ctor_set(v___x_70_, 3, v_inst_52_);
lean_ctor_set(v___x_70_, 4, v_inst_54_);
lean_ctor_set(v___x_70_, 5, v_inst_53_);
lean_ctor_set(v___x_70_, 6, v_inst_55_);
v___x_71_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_71_, 0, v_inst_45_);
lean_ctor_set(v___x_71_, 1, v_inst_46_);
lean_ctor_set(v___x_71_, 2, v___f_68_);
v___x_72_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_72_, 0, v___x_71_);
lean_ctor_set(v___x_72_, 1, v_inst_43_);
v___x_73_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_73_, 0, v___x_72_);
lean_ctor_set(v___x_73_, 1, v___x_70_);
lean_ctor_set(v___x_73_, 2, v_inst_44_);
return v___x_73_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_linearOrderedCommMonoidWithZero___boxed(lean_object** _args){
lean_object* v_00_u03b1_74_ = _args[0];
lean_object* v_inst_75_ = _args[1];
lean_object* v_00_u03b2_76_ = _args[2];
lean_object* v_inst_77_ = _args[3];
lean_object* v_inst_78_ = _args[4];
lean_object* v_inst_79_ = _args[5];
lean_object* v_inst_80_ = _args[6];
lean_object* v_inst_81_ = _args[7];
lean_object* v_inst_82_ = _args[8];
lean_object* v_inst_83_ = _args[9];
lean_object* v_inst_84_ = _args[10];
lean_object* v_inst_85_ = _args[11];
lean_object* v_inst_86_ = _args[12];
lean_object* v_inst_87_ = _args[13];
lean_object* v_inst_88_ = _args[14];
lean_object* v_inst_89_ = _args[15];
lean_object* v_f_90_ = _args[16];
lean_object* v_hf_91_ = _args[17];
lean_object* v_zero_92_ = _args[18];
lean_object* v_one_93_ = _args[19];
lean_object* v_mul_94_ = _args[20];
lean_object* v_npow_95_ = _args[21];
lean_object* v_le_96_ = _args[22];
lean_object* v_lt_97_ = _args[23];
lean_object* v_hsup_98_ = _args[24];
lean_object* v_hinf_99_ = _args[25];
lean_object* v_bot_100_ = _args[26];
lean_object* v_compare_101_ = _args[27];
_start:
{
lean_object* v_res_102_; 
v_res_102_ = lp_mathlib_Function_Injective_linearOrderedCommMonoidWithZero(v_00_u03b1_74_, v_inst_75_, v_00_u03b2_76_, v_inst_77_, v_inst_78_, v_inst_79_, v_inst_80_, v_inst_81_, v_inst_82_, v_inst_83_, v_inst_84_, v_inst_85_, v_inst_86_, v_inst_87_, v_inst_88_, v_inst_89_, v_f_90_, v_hf_91_, v_zero_92_, v_one_93_, v_mul_94_, v_npow_95_, v_le_96_, v_lt_97_, v_hsup_98_, v_hinf_99_, v_bot_100_, v_compare_101_);
lean_dec(v_f_90_);
lean_dec_ref(v_inst_75_);
return v_res_102_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instLinearOrderedAddCommMonoidWithTopAdditiveOrderDual___redArg(lean_object* v_inst_103_){
_start:
{
lean_object* v_toCommMonoidWithZero_104_; lean_object* v_toLinearOrder_105_; lean_object* v_toOrderBot_106_; lean_object* v___x_108_; uint8_t v_isShared_109_; uint8_t v_isSharedCheck_118_; 
v_toCommMonoidWithZero_104_ = lean_ctor_get(v_inst_103_, 0);
v_toLinearOrder_105_ = lean_ctor_get(v_inst_103_, 1);
v_toOrderBot_106_ = lean_ctor_get(v_inst_103_, 2);
v_isSharedCheck_118_ = !lean_is_exclusive(v_inst_103_);
if (v_isSharedCheck_118_ == 0)
{
v___x_108_ = v_inst_103_;
v_isShared_109_ = v_isSharedCheck_118_;
goto v_resetjp_107_;
}
else
{
lean_inc(v_toOrderBot_106_);
lean_inc(v_toLinearOrder_105_);
lean_inc(v_toCommMonoidWithZero_104_);
lean_dec(v_inst_103_);
v___x_108_ = lean_box(0);
v_isShared_109_ = v_isSharedCheck_118_;
goto v_resetjp_107_;
}
v_resetjp_107_:
{
lean_object* v_toCommMonoid_110_; lean_object* v___x_111_; lean_object* v___x_112_; lean_object* v___x_113_; lean_object* v___x_114_; lean_object* v___x_116_; 
v_toCommMonoid_110_ = lean_ctor_get(v_toCommMonoidWithZero_104_, 0);
lean_inc_ref(v_toCommMonoid_110_);
lean_dec_ref(v_toCommMonoidWithZero_104_);
v___x_111_ = lp_mathlib_OrderDual_instMonoid___redArg(v_toCommMonoid_110_);
v___x_112_ = lp_mathlib_Additive_addMonoid___redArg(v___x_111_);
v___x_113_ = lp_mathlib_OrderDual_instLinearOrder___redArg(v_toLinearOrder_105_);
v___x_114_ = lp_mathlib_Additive_linearOrder___redArg(v___x_113_);
if (v_isShared_109_ == 0)
{
lean_ctor_set(v___x_108_, 1, v___x_114_);
lean_ctor_set(v___x_108_, 0, v___x_112_);
v___x_116_ = v___x_108_;
goto v_reusejp_115_;
}
else
{
lean_object* v_reuseFailAlloc_117_; 
v_reuseFailAlloc_117_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v_reuseFailAlloc_117_, 0, v___x_112_);
lean_ctor_set(v_reuseFailAlloc_117_, 1, v___x_114_);
lean_ctor_set(v_reuseFailAlloc_117_, 2, v_toOrderBot_106_);
v___x_116_ = v_reuseFailAlloc_117_;
goto v_reusejp_115_;
}
v_reusejp_115_:
{
return v___x_116_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_instLinearOrderedAddCommMonoidWithTopAdditiveOrderDual(lean_object* v_00_u03b1_119_, lean_object* v_inst_120_){
_start:
{
lean_object* v___x_121_; 
v___x_121_ = lp_mathlib_instLinearOrderedAddCommMonoidWithTopAdditiveOrderDual___redArg(v_inst_120_);
return v___x_121_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instLinearOrderedAddCommMonoidWithTopOrderDualAdditive___redArg(lean_object* v_inst_122_){
_start:
{
lean_object* v_toCommMonoidWithZero_123_; lean_object* v_toLinearOrder_124_; lean_object* v_toOrderBot_125_; lean_object* v___x_127_; uint8_t v_isShared_128_; uint8_t v_isSharedCheck_137_; 
v_toCommMonoidWithZero_123_ = lean_ctor_get(v_inst_122_, 0);
v_toLinearOrder_124_ = lean_ctor_get(v_inst_122_, 1);
v_toOrderBot_125_ = lean_ctor_get(v_inst_122_, 2);
v_isSharedCheck_137_ = !lean_is_exclusive(v_inst_122_);
if (v_isSharedCheck_137_ == 0)
{
v___x_127_ = v_inst_122_;
v_isShared_128_ = v_isSharedCheck_137_;
goto v_resetjp_126_;
}
else
{
lean_inc(v_toOrderBot_125_);
lean_inc(v_toLinearOrder_124_);
lean_inc(v_toCommMonoidWithZero_123_);
lean_dec(v_inst_122_);
v___x_127_ = lean_box(0);
v_isShared_128_ = v_isSharedCheck_137_;
goto v_resetjp_126_;
}
v_resetjp_126_:
{
lean_object* v_toCommMonoid_129_; lean_object* v___x_130_; lean_object* v___x_131_; lean_object* v___x_132_; lean_object* v___x_133_; lean_object* v___x_135_; 
v_toCommMonoid_129_ = lean_ctor_get(v_toCommMonoidWithZero_123_, 0);
lean_inc_ref(v_toCommMonoid_129_);
lean_dec_ref(v_toCommMonoidWithZero_123_);
v___x_130_ = lp_mathlib_Additive_addMonoid___redArg(v_toCommMonoid_129_);
v___x_131_ = lp_mathlib_OrderDual_instAddMonoid___redArg(v___x_130_);
v___x_132_ = lp_mathlib_Additive_linearOrder___redArg(v_toLinearOrder_124_);
v___x_133_ = lp_mathlib_OrderDual_instLinearOrder___redArg(v___x_132_);
if (v_isShared_128_ == 0)
{
lean_ctor_set(v___x_127_, 1, v___x_133_);
lean_ctor_set(v___x_127_, 0, v___x_131_);
v___x_135_ = v___x_127_;
goto v_reusejp_134_;
}
else
{
lean_object* v_reuseFailAlloc_136_; 
v_reuseFailAlloc_136_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v_reuseFailAlloc_136_, 0, v___x_131_);
lean_ctor_set(v_reuseFailAlloc_136_, 1, v___x_133_);
lean_ctor_set(v_reuseFailAlloc_136_, 2, v_toOrderBot_125_);
v___x_135_ = v_reuseFailAlloc_136_;
goto v_reusejp_134_;
}
v_reusejp_134_:
{
return v___x_135_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_instLinearOrderedAddCommMonoidWithTopOrderDualAdditive(lean_object* v_00_u03b1_138_, lean_object* v_inst_139_){
_start:
{
lean_object* v___x_140_; 
v___x_140_ = lp_mathlib_instLinearOrderedAddCommMonoidWithTopOrderDualAdditive___redArg(v_inst_139_);
return v___x_140_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instLinearOrderedAddCommGroupWithTopAdditiveOrderDual___redArg(lean_object* v_inst_141_){
_start:
{
lean_object* v_toLinearOrderedCommMonoidWithZero_142_; lean_object* v_toCommMonoidWithZero_143_; lean_object* v_toLinearOrder_144_; lean_object* v_toOrderBot_145_; lean_object* v_toCommMonoid_146_; lean_object* v___x_147_; lean_object* v___x_148_; lean_object* v___x_149_; lean_object* v___x_150_; lean_object* v___x_151_; lean_object* v___x_152_; lean_object* v___x_153_; lean_object* v___x_154_; lean_object* v___x_155_; lean_object* v_toNeg_156_; lean_object* v_toSub_157_; lean_object* v_toZSMul_158_; lean_object* v___x_159_; 
v_toLinearOrderedCommMonoidWithZero_142_ = lean_ctor_get(v_inst_141_, 0);
v_toCommMonoidWithZero_143_ = lean_ctor_get(v_toLinearOrderedCommMonoidWithZero_142_, 0);
v_toLinearOrder_144_ = lean_ctor_get(v_toLinearOrderedCommMonoidWithZero_142_, 1);
v_toOrderBot_145_ = lean_ctor_get(v_toLinearOrderedCommMonoidWithZero_142_, 2);
lean_inc(v_toOrderBot_145_);
v_toCommMonoid_146_ = lean_ctor_get(v_toCommMonoidWithZero_143_, 0);
lean_inc_ref(v_toCommMonoid_146_);
v___x_147_ = lp_mathlib_OrderDual_instMonoid___redArg(v_toCommMonoid_146_);
v___x_148_ = lp_mathlib_Additive_addMonoid___redArg(v___x_147_);
lean_inc_ref(v_toLinearOrder_144_);
v___x_149_ = lp_mathlib_OrderDual_instLinearOrder___redArg(v_toLinearOrder_144_);
v___x_150_ = lp_mathlib_Additive_linearOrder___redArg(v___x_149_);
v___x_151_ = lp_mathlib_LinearOrderedCommGroupWithZero_toCommGroupWithZero___redArg(v_inst_141_);
v___x_152_ = lp_mathlib_CommGroupWithZero_toGroupWithZero___redArg(v___x_151_);
v___x_153_ = lp_mathlib_GroupWithZero_toDivInvMonoid___redArg(v___x_152_);
v___x_154_ = lp_mathlib_OrderDual_instDivInvMonoid___redArg(v___x_153_);
v___x_155_ = lp_mathlib_Additive_subNegMonoid___redArg(v___x_154_);
v_toNeg_156_ = lean_ctor_get(v___x_155_, 1);
lean_inc(v_toNeg_156_);
v_toSub_157_ = lean_ctor_get(v___x_155_, 2);
lean_inc(v_toSub_157_);
v_toZSMul_158_ = lean_ctor_get(v___x_155_, 3);
lean_inc(v_toZSMul_158_);
lean_dec_ref(v___x_155_);
v___x_159_ = lean_alloc_ctor(0, 6, 0);
lean_ctor_set(v___x_159_, 0, v___x_148_);
lean_ctor_set(v___x_159_, 1, v___x_150_);
lean_ctor_set(v___x_159_, 2, v_toOrderBot_145_);
lean_ctor_set(v___x_159_, 3, v_toNeg_156_);
lean_ctor_set(v___x_159_, 4, v_toSub_157_);
lean_ctor_set(v___x_159_, 5, v_toZSMul_158_);
return v___x_159_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instLinearOrderedAddCommGroupWithTopAdditiveOrderDual(lean_object* v_00_u03b1_160_, lean_object* v_inst_161_){
_start:
{
lean_object* v___x_162_; 
v___x_162_ = lp_mathlib_instLinearOrderedAddCommGroupWithTopAdditiveOrderDual___redArg(v_inst_161_);
return v___x_162_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instLinearOrderedAddCommGroupWithTopOrderDualAdditive___redArg(lean_object* v_inst_163_){
_start:
{
lean_object* v_toLinearOrderedCommMonoidWithZero_164_; lean_object* v_toCommMonoidWithZero_165_; lean_object* v_toLinearOrder_166_; lean_object* v_toOrderBot_167_; lean_object* v_toCommMonoid_168_; lean_object* v___x_169_; lean_object* v___x_170_; lean_object* v___x_171_; lean_object* v___x_172_; lean_object* v___x_173_; lean_object* v___x_174_; lean_object* v___x_175_; lean_object* v___x_176_; lean_object* v___x_177_; lean_object* v_toNeg_178_; lean_object* v_toSub_179_; lean_object* v_toZSMul_180_; lean_object* v___x_181_; 
v_toLinearOrderedCommMonoidWithZero_164_ = lean_ctor_get(v_inst_163_, 0);
v_toCommMonoidWithZero_165_ = lean_ctor_get(v_toLinearOrderedCommMonoidWithZero_164_, 0);
v_toLinearOrder_166_ = lean_ctor_get(v_toLinearOrderedCommMonoidWithZero_164_, 1);
v_toOrderBot_167_ = lean_ctor_get(v_toLinearOrderedCommMonoidWithZero_164_, 2);
lean_inc(v_toOrderBot_167_);
v_toCommMonoid_168_ = lean_ctor_get(v_toCommMonoidWithZero_165_, 0);
lean_inc_ref(v_toCommMonoid_168_);
v___x_169_ = lp_mathlib_Additive_addMonoid___redArg(v_toCommMonoid_168_);
v___x_170_ = lp_mathlib_OrderDual_instAddMonoid___redArg(v___x_169_);
lean_inc_ref(v_toLinearOrder_166_);
v___x_171_ = lp_mathlib_Additive_linearOrder___redArg(v_toLinearOrder_166_);
v___x_172_ = lp_mathlib_OrderDual_instLinearOrder___redArg(v___x_171_);
v___x_173_ = lp_mathlib_LinearOrderedCommGroupWithZero_toCommGroupWithZero___redArg(v_inst_163_);
v___x_174_ = lp_mathlib_CommGroupWithZero_toGroupWithZero___redArg(v___x_173_);
v___x_175_ = lp_mathlib_GroupWithZero_toDivInvMonoid___redArg(v___x_174_);
v___x_176_ = lp_mathlib_Additive_subNegMonoid___redArg(v___x_175_);
v___x_177_ = lp_mathlib_OrderDual_instSubNegAddMonoid___redArg(v___x_176_);
v_toNeg_178_ = lean_ctor_get(v___x_177_, 1);
lean_inc(v_toNeg_178_);
v_toSub_179_ = lean_ctor_get(v___x_177_, 2);
lean_inc(v_toSub_179_);
v_toZSMul_180_ = lean_ctor_get(v___x_177_, 3);
lean_inc(v_toZSMul_180_);
lean_dec_ref(v___x_177_);
v___x_181_ = lean_alloc_ctor(0, 6, 0);
lean_ctor_set(v___x_181_, 0, v___x_170_);
lean_ctor_set(v___x_181_, 1, v___x_172_);
lean_ctor_set(v___x_181_, 2, v_toOrderBot_167_);
lean_ctor_set(v___x_181_, 3, v_toNeg_178_);
lean_ctor_set(v___x_181_, 4, v_toSub_179_);
lean_ctor_set(v___x_181_, 5, v_toZSMul_180_);
return v___x_181_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instLinearOrderedAddCommGroupWithTopOrderDualAdditive(lean_object* v_00_u03b1_182_, lean_object* v_inst_183_){
_start:
{
lean_object* v___x_184_; 
v___x_184_ = lp_mathlib_instLinearOrderedAddCommGroupWithTopOrderDualAdditive___redArg(v_inst_183_);
return v___x_184_;
}
}
static lean_object* _init_lp_mathlib_instLinearOrderedCommMonoidWithZeroMultiplicativeOrderDual___redArg___closed__0(void){
_start:
{
lean_object* v___x_185_; 
v___x_185_ = lp_mathlib_Equiv_refl(lean_box(0));
return v___x_185_;
}
}
static lean_object* _init_lp_mathlib_instLinearOrderedCommMonoidWithZeroMultiplicativeOrderDual___redArg___closed__1(void){
_start:
{
lean_object* v___x_186_; 
v___x_186_ = lp_mathlib_Multiplicative_ofAdd(lean_box(0));
return v___x_186_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instLinearOrderedCommMonoidWithZeroMultiplicativeOrderDual___redArg(lean_object* v_inst_187_){
_start:
{
lean_object* v_toAddCommMonoid_188_; lean_object* v_toLinearOrder_189_; lean_object* v_toOrderTop_190_; lean_object* v___x_192_; uint8_t v_isShared_193_; uint8_t v_isSharedCheck_208_; 
v_toAddCommMonoid_188_ = lean_ctor_get(v_inst_187_, 0);
v_toLinearOrder_189_ = lean_ctor_get(v_inst_187_, 1);
v_toOrderTop_190_ = lean_ctor_get(v_inst_187_, 2);
v_isSharedCheck_208_ = !lean_is_exclusive(v_inst_187_);
if (v_isSharedCheck_208_ == 0)
{
v___x_192_ = v_inst_187_;
v_isShared_193_ = v_isSharedCheck_208_;
goto v_resetjp_191_;
}
else
{
lean_inc(v_toOrderTop_190_);
lean_inc(v_toLinearOrder_189_);
lean_inc(v_toAddCommMonoid_188_);
lean_dec(v_inst_187_);
v___x_192_ = lean_box(0);
v_isShared_193_ = v_isSharedCheck_208_;
goto v_resetjp_191_;
}
v_resetjp_191_:
{
lean_object* v___x_194_; lean_object* v___x_195_; lean_object* v___x_196_; lean_object* v___x_197_; lean_object* v___x_198_; lean_object* v_toFun_199_; lean_object* v___x_200_; lean_object* v_toFun_201_; lean_object* v___x_202_; lean_object* v___x_203_; lean_object* v___x_204_; lean_object* v___x_206_; 
v___x_194_ = lp_mathlib_OrderDual_instAddMonoid___redArg(v_toAddCommMonoid_188_);
v___x_195_ = lp_mathlib_Multiplicative_monoid___redArg(v___x_194_);
v___x_196_ = lp_mathlib_OrderDual_instLinearOrder___redArg(v_toLinearOrder_189_);
v___x_197_ = lp_mathlib_Multiplicative_linearOrder___redArg(v___x_196_);
v___x_198_ = lean_obj_once(&lp_mathlib_instLinearOrderedCommMonoidWithZeroMultiplicativeOrderDual___redArg___closed__0, &lp_mathlib_instLinearOrderedCommMonoidWithZeroMultiplicativeOrderDual___redArg___closed__0_once, _init_lp_mathlib_instLinearOrderedCommMonoidWithZeroMultiplicativeOrderDual___redArg___closed__0);
v_toFun_199_ = lean_ctor_get(v___x_198_, 0);
v___x_200_ = lean_obj_once(&lp_mathlib_instLinearOrderedCommMonoidWithZeroMultiplicativeOrderDual___redArg___closed__1, &lp_mathlib_instLinearOrderedCommMonoidWithZeroMultiplicativeOrderDual___redArg___closed__1_once, _init_lp_mathlib_instLinearOrderedCommMonoidWithZeroMultiplicativeOrderDual___redArg___closed__1);
v_toFun_201_ = lean_ctor_get(v___x_200_, 0);
lean_inc(v_toFun_199_);
lean_inc(v_toOrderTop_190_);
v___x_202_ = lean_apply_1(v_toFun_199_, v_toOrderTop_190_);
lean_inc(v_toFun_201_);
v___x_203_ = lean_apply_1(v_toFun_201_, v___x_202_);
v___x_204_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_204_, 0, v___x_195_);
lean_ctor_set(v___x_204_, 1, v___x_203_);
if (v_isShared_193_ == 0)
{
lean_ctor_set(v___x_192_, 1, v___x_197_);
lean_ctor_set(v___x_192_, 0, v___x_204_);
v___x_206_ = v___x_192_;
goto v_reusejp_205_;
}
else
{
lean_object* v_reuseFailAlloc_207_; 
v_reuseFailAlloc_207_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v_reuseFailAlloc_207_, 0, v___x_204_);
lean_ctor_set(v_reuseFailAlloc_207_, 1, v___x_197_);
lean_ctor_set(v_reuseFailAlloc_207_, 2, v_toOrderTop_190_);
v___x_206_ = v_reuseFailAlloc_207_;
goto v_reusejp_205_;
}
v_reusejp_205_:
{
return v___x_206_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_instLinearOrderedCommMonoidWithZeroMultiplicativeOrderDual(lean_object* v_00_u03b1_209_, lean_object* v_inst_210_){
_start:
{
lean_object* v___x_211_; 
v___x_211_ = lp_mathlib_instLinearOrderedCommMonoidWithZeroMultiplicativeOrderDual___redArg(v_inst_210_);
return v___x_211_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instLinearOrderedCommGroupWithZeroMultiplicativeOrderDualOfLinearOrderedAddCommGroupWithTop___redArg(lean_object* v_inst_212_){
_start:
{
lean_object* v___x_213_; lean_object* v___x_214_; lean_object* v___x_215_; lean_object* v___x_216_; lean_object* v___x_217_; lean_object* v_toInv_218_; lean_object* v_toDiv_219_; lean_object* v_toZPow_220_; lean_object* v___x_222_; uint8_t v_isShared_223_; uint8_t v_isSharedCheck_227_; 
v___x_213_ = lp_mathlib_LinearOrderedAddCommGroupWithTop_instLinearOrderedAddCommMonoidWithTop___redArg(v_inst_212_);
v___x_214_ = lp_mathlib_instLinearOrderedCommMonoidWithZeroMultiplicativeOrderDual___redArg(v___x_213_);
v___x_215_ = lp_mathlib_LinearOrderedAddCommGroupWithTop_toSubNegMonoid___redArg(v_inst_212_);
v___x_216_ = lp_mathlib_OrderDual_instSubNegAddMonoid___redArg(v___x_215_);
v___x_217_ = lp_mathlib_Multiplicative_divInvMonoid___redArg(v___x_216_);
v_toInv_218_ = lean_ctor_get(v___x_217_, 1);
v_toDiv_219_ = lean_ctor_get(v___x_217_, 2);
v_toZPow_220_ = lean_ctor_get(v___x_217_, 3);
v_isSharedCheck_227_ = !lean_is_exclusive(v___x_217_);
if (v_isSharedCheck_227_ == 0)
{
lean_object* v_unused_228_; 
v_unused_228_ = lean_ctor_get(v___x_217_, 0);
lean_dec(v_unused_228_);
v___x_222_ = v___x_217_;
v_isShared_223_ = v_isSharedCheck_227_;
goto v_resetjp_221_;
}
else
{
lean_inc(v_toZPow_220_);
lean_inc(v_toDiv_219_);
lean_inc(v_toInv_218_);
lean_dec(v___x_217_);
v___x_222_ = lean_box(0);
v_isShared_223_ = v_isSharedCheck_227_;
goto v_resetjp_221_;
}
v_resetjp_221_:
{
lean_object* v___x_225_; 
if (v_isShared_223_ == 0)
{
lean_ctor_set(v___x_222_, 0, v___x_214_);
v___x_225_ = v___x_222_;
goto v_reusejp_224_;
}
else
{
lean_object* v_reuseFailAlloc_226_; 
v_reuseFailAlloc_226_ = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(v_reuseFailAlloc_226_, 0, v___x_214_);
lean_ctor_set(v_reuseFailAlloc_226_, 1, v_toInv_218_);
lean_ctor_set(v_reuseFailAlloc_226_, 2, v_toDiv_219_);
lean_ctor_set(v_reuseFailAlloc_226_, 3, v_toZPow_220_);
v___x_225_ = v_reuseFailAlloc_226_;
goto v_reusejp_224_;
}
v_reusejp_224_:
{
return v___x_225_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_instLinearOrderedCommGroupWithZeroMultiplicativeOrderDualOfLinearOrderedAddCommGroupWithTop___redArg___boxed(lean_object* v_inst_229_){
_start:
{
lean_object* v_res_230_; 
v_res_230_ = lp_mathlib_instLinearOrderedCommGroupWithZeroMultiplicativeOrderDualOfLinearOrderedAddCommGroupWithTop___redArg(v_inst_229_);
lean_dec_ref(v_inst_229_);
return v_res_230_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instLinearOrderedCommGroupWithZeroMultiplicativeOrderDualOfLinearOrderedAddCommGroupWithTop(lean_object* v_00_u03b1_231_, lean_object* v_inst_232_){
_start:
{
lean_object* v___x_233_; 
v___x_233_ = lp_mathlib_instLinearOrderedCommGroupWithZeroMultiplicativeOrderDualOfLinearOrderedAddCommGroupWithTop___redArg(v_inst_232_);
return v___x_233_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instLinearOrderedCommGroupWithZeroMultiplicativeOrderDualOfLinearOrderedAddCommGroupWithTop___boxed(lean_object* v_00_u03b1_234_, lean_object* v_inst_235_){
_start:
{
lean_object* v_res_236_; 
v_res_236_ = lp_mathlib_instLinearOrderedCommGroupWithZeroMultiplicativeOrderDualOfLinearOrderedAddCommGroupWithTop(v_00_u03b1_234_, v_inst_235_);
lean_dec_ref(v_inst_235_);
return v_res_236_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_WithZero_instBot(lean_object* v_00_u03b1_237_){
_start:
{
lean_object* v___x_238_; 
v___x_238_ = lean_box(0);
return v___x_238_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_WithZero_le(lean_object* v_00_u03b1_239_, lean_object* v_inst_240_){
_start:
{
lean_object* v___x_241_; 
v___x_241_ = lean_box(0);
return v___x_241_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_WithZero_instOrderBot(lean_object* v_00_u03b1_242_, lean_object* v_inst_243_){
_start:
{
lean_object* v___x_244_; 
v___x_244_ = lean_box(0);
return v___x_244_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_WithZero_instBoundedOrder___aux__1___redArg(lean_object* v_inst_245_){
_start:
{
lean_object* v___x_246_; 
v___x_246_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_246_, 0, v_inst_245_);
return v___x_246_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_WithZero_instBoundedOrder___aux__1(lean_object* v_00_u03b1_247_, lean_object* v_inst_248_, lean_object* v_inst_249_){
_start:
{
lean_object* v___x_250_; 
v___x_250_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_250_, 0, v_inst_249_);
return v___x_250_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_WithZero_instBoundedOrder___redArg(lean_object* v_inst_251_){
_start:
{
lean_object* v___x_252_; lean_object* v___x_253_; lean_object* v___x_254_; 
v___x_252_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_252_, 0, v_inst_251_);
v___x_253_ = lean_box(0);
v___x_254_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_254_, 0, v___x_252_);
lean_ctor_set(v___x_254_, 1, v___x_253_);
return v___x_254_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_WithZero_instBoundedOrder(lean_object* v_00_u03b1_255_, lean_object* v_inst_256_, lean_object* v_inst_257_){
_start:
{
lean_object* v___x_258_; 
v___x_258_ = lp_mathlib_WithZero_instBoundedOrder___redArg(v_inst_257_);
return v___x_258_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_WithZero_instLT(lean_object* v_00_u03b1_259_, lean_object* v_inst_260_){
_start:
{
lean_object* v___x_261_; 
v___x_261_ = lean_box(0);
return v___x_261_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_WithZero_instPreorder(lean_object* v_00_u03b1_265_, lean_object* v_inst_266_){
_start:
{
lean_object* v___x_267_; 
v___x_267_ = ((lean_object*)(lp_mathlib_WithZero_instPreorder___closed__0));
return v___x_267_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_WithZero_instPreorder___boxed(lean_object* v_00_u03b1_268_, lean_object* v_inst_269_){
_start:
{
lean_object* v_res_270_; 
v_res_270_ = lp_mathlib_WithZero_instPreorder(v_00_u03b1_268_, v_inst_269_);
lean_dec_ref(v_inst_269_);
return v_res_270_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_WithZero_instPartialOrder___redArg(lean_object* v_inst_271_){
_start:
{
lean_object* v___x_272_; 
v___x_272_ = lp_mathlib_WithZero_instPreorder(lean_box(0), v_inst_271_);
return v___x_272_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_WithZero_instPartialOrder___redArg___boxed(lean_object* v_inst_273_){
_start:
{
lean_object* v_res_274_; 
v_res_274_ = lp_mathlib_WithZero_instPartialOrder___redArg(v_inst_273_);
lean_dec_ref(v_inst_273_);
return v_res_274_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_WithZero_instPartialOrder(lean_object* v_00_u03b1_275_, lean_object* v_inst_276_){
_start:
{
lean_object* v___x_277_; 
v___x_277_ = lp_mathlib_WithZero_instPreorder(lean_box(0), v_inst_276_);
return v___x_277_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_WithZero_instPartialOrder___boxed(lean_object* v_00_u03b1_278_, lean_object* v_inst_279_){
_start:
{
lean_object* v_res_280_; 
v_res_280_ = lp_mathlib_WithZero_instPartialOrder(v_00_u03b1_278_, v_inst_279_);
lean_dec_ref(v_inst_279_);
return v_res_280_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_WithZero_semilatticeSup___redArg___lam__0(lean_object* v___x_281_, lean_object* v_sup_282_, lean_object* v_x_283_, lean_object* v_x_284_){
_start:
{
if (lean_obj_tag(v_x_283_) == 0)
{
lean_dec(v_sup_282_);
if (lean_obj_tag(v_x_284_) == 0)
{
lean_inc(v___x_281_);
return v___x_281_;
}
else
{
return v_x_284_;
}
}
else
{
if (lean_obj_tag(v_x_284_) == 0)
{
lean_dec(v_sup_282_);
return v_x_283_;
}
else
{
lean_object* v_val_285_; lean_object* v_val_286_; lean_object* v___x_288_; uint8_t v_isShared_289_; uint8_t v_isSharedCheck_294_; 
v_val_285_ = lean_ctor_get(v_x_283_, 0);
lean_inc(v_val_285_);
lean_dec_ref_known(v_x_283_, 1);
v_val_286_ = lean_ctor_get(v_x_284_, 0);
v_isSharedCheck_294_ = !lean_is_exclusive(v_x_284_);
if (v_isSharedCheck_294_ == 0)
{
v___x_288_ = v_x_284_;
v_isShared_289_ = v_isSharedCheck_294_;
goto v_resetjp_287_;
}
else
{
lean_inc(v_val_286_);
lean_dec(v_x_284_);
v___x_288_ = lean_box(0);
v_isShared_289_ = v_isSharedCheck_294_;
goto v_resetjp_287_;
}
v_resetjp_287_:
{
lean_object* v___x_290_; lean_object* v___x_292_; 
v___x_290_ = lean_apply_2(v_sup_282_, v_val_285_, v_val_286_);
if (v_isShared_289_ == 0)
{
lean_ctor_set(v___x_288_, 0, v___x_290_);
v___x_292_ = v___x_288_;
goto v_reusejp_291_;
}
else
{
lean_object* v_reuseFailAlloc_293_; 
v_reuseFailAlloc_293_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_293_, 0, v___x_290_);
v___x_292_ = v_reuseFailAlloc_293_;
goto v_reusejp_291_;
}
v_reusejp_291_:
{
return v___x_292_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_WithZero_semilatticeSup___redArg___lam__0___boxed(lean_object* v___x_295_, lean_object* v_sup_296_, lean_object* v_x_297_, lean_object* v_x_298_){
_start:
{
lean_object* v_res_299_; 
v_res_299_ = lp_mathlib_WithZero_semilatticeSup___redArg___lam__0(v___x_295_, v_sup_296_, v_x_297_, v_x_298_);
lean_dec(v___x_295_);
return v_res_299_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_WithZero_semilatticeSup___redArg(lean_object* v_inst_300_){
_start:
{
lean_object* v_toPartialOrder_301_; lean_object* v_sup_302_; lean_object* v___x_304_; uint8_t v_isShared_305_; uint8_t v_isSharedCheck_312_; 
v_toPartialOrder_301_ = lean_ctor_get(v_inst_300_, 0);
v_sup_302_ = lean_ctor_get(v_inst_300_, 1);
v_isSharedCheck_312_ = !lean_is_exclusive(v_inst_300_);
if (v_isSharedCheck_312_ == 0)
{
v___x_304_ = v_inst_300_;
v_isShared_305_ = v_isSharedCheck_312_;
goto v_resetjp_303_;
}
else
{
lean_inc(v_sup_302_);
lean_inc(v_toPartialOrder_301_);
lean_dec(v_inst_300_);
v___x_304_ = lean_box(0);
v_isShared_305_ = v_isSharedCheck_312_;
goto v_resetjp_303_;
}
v_resetjp_303_:
{
lean_object* v___x_306_; lean_object* v___x_307_; lean_object* v___f_308_; lean_object* v___x_310_; 
v___x_306_ = lp_mathlib_WithZero_instPreorder(lean_box(0), v_toPartialOrder_301_);
lean_dec_ref(v_toPartialOrder_301_);
v___x_307_ = lean_box(0);
v___f_308_ = lean_alloc_closure((void*)(lp_mathlib_WithZero_semilatticeSup___redArg___lam__0___boxed), 4, 2);
lean_closure_set(v___f_308_, 0, v___x_307_);
lean_closure_set(v___f_308_, 1, v_sup_302_);
if (v_isShared_305_ == 0)
{
lean_ctor_set(v___x_304_, 1, v___f_308_);
lean_ctor_set(v___x_304_, 0, v___x_306_);
v___x_310_ = v___x_304_;
goto v_reusejp_309_;
}
else
{
lean_object* v_reuseFailAlloc_311_; 
v_reuseFailAlloc_311_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_311_, 0, v___x_306_);
lean_ctor_set(v_reuseFailAlloc_311_, 1, v___f_308_);
v___x_310_ = v_reuseFailAlloc_311_;
goto v_reusejp_309_;
}
v_reusejp_309_:
{
return v___x_310_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_WithZero_semilatticeSup(lean_object* v_00_u03b1_313_, lean_object* v_inst_314_){
_start:
{
lean_object* v___x_315_; 
v___x_315_ = lp_mathlib_WithZero_semilatticeSup___redArg(v_inst_314_);
return v___x_315_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_WithZero_semilatticeInf___redArg___lam__0(lean_object* v_inf_316_, lean_object* v_x1_317_, lean_object* v_x2_318_){
_start:
{
lean_object* v___x_319_; 
v___x_319_ = lean_apply_2(v_inf_316_, v_x1_317_, v_x2_318_);
return v___x_319_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_WithZero_semilatticeInf___redArg(lean_object* v_inst_320_){
_start:
{
lean_object* v_toPartialOrder_321_; lean_object* v_inf_322_; lean_object* v___x_324_; uint8_t v_isShared_325_; uint8_t v_isSharedCheck_332_; 
v_toPartialOrder_321_ = lean_ctor_get(v_inst_320_, 0);
v_inf_322_ = lean_ctor_get(v_inst_320_, 1);
v_isSharedCheck_332_ = !lean_is_exclusive(v_inst_320_);
if (v_isSharedCheck_332_ == 0)
{
v___x_324_ = v_inst_320_;
v_isShared_325_ = v_isSharedCheck_332_;
goto v_resetjp_323_;
}
else
{
lean_inc(v_inf_322_);
lean_inc(v_toPartialOrder_321_);
lean_dec(v_inst_320_);
v___x_324_ = lean_box(0);
v_isShared_325_ = v_isSharedCheck_332_;
goto v_resetjp_323_;
}
v_resetjp_323_:
{
lean_object* v___f_326_; lean_object* v___x_327_; lean_object* v___x_328_; lean_object* v___x_330_; 
v___f_326_ = lean_alloc_closure((void*)(lp_mathlib_WithZero_semilatticeInf___redArg___lam__0), 3, 1);
lean_closure_set(v___f_326_, 0, v_inf_322_);
v___x_327_ = lp_mathlib_WithZero_instPreorder(lean_box(0), v_toPartialOrder_321_);
lean_dec_ref(v_toPartialOrder_321_);
v___x_328_ = lean_alloc_closure((void*)(lp_mathlib_WithZero_map_u2082), 6, 4);
lean_closure_set(v___x_328_, 0, lean_box(0));
lean_closure_set(v___x_328_, 1, lean_box(0));
lean_closure_set(v___x_328_, 2, lean_box(0));
lean_closure_set(v___x_328_, 3, v___f_326_);
if (v_isShared_325_ == 0)
{
lean_ctor_set(v___x_324_, 1, v___x_328_);
lean_ctor_set(v___x_324_, 0, v___x_327_);
v___x_330_ = v___x_324_;
goto v_reusejp_329_;
}
else
{
lean_object* v_reuseFailAlloc_331_; 
v_reuseFailAlloc_331_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_331_, 0, v___x_327_);
lean_ctor_set(v_reuseFailAlloc_331_, 1, v___x_328_);
v___x_330_ = v_reuseFailAlloc_331_;
goto v_reusejp_329_;
}
v_reusejp_329_:
{
return v___x_330_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_WithZero_semilatticeInf(lean_object* v_00_u03b1_333_, lean_object* v_inst_334_){
_start:
{
lean_object* v___x_335_; 
v___x_335_ = lp_mathlib_WithZero_semilatticeInf___redArg(v_inst_334_);
return v___x_335_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_WithZero_instLattice___redArg(lean_object* v_inst_336_){
_start:
{
lean_object* v_toSemilatticeSup_337_; lean_object* v_inf_338_; lean_object* v___x_340_; uint8_t v_isShared_341_; uint8_t v_isSharedCheck_348_; 
v_toSemilatticeSup_337_ = lean_ctor_get(v_inst_336_, 0);
v_inf_338_ = lean_ctor_get(v_inst_336_, 1);
v_isSharedCheck_348_ = !lean_is_exclusive(v_inst_336_);
if (v_isSharedCheck_348_ == 0)
{
v___x_340_ = v_inst_336_;
v_isShared_341_ = v_isSharedCheck_348_;
goto v_resetjp_339_;
}
else
{
lean_inc(v_inf_338_);
lean_inc(v_toSemilatticeSup_337_);
lean_dec(v_inst_336_);
v___x_340_ = lean_box(0);
v_isShared_341_ = v_isSharedCheck_348_;
goto v_resetjp_339_;
}
v_resetjp_339_:
{
lean_object* v___f_342_; lean_object* v___x_343_; lean_object* v___x_344_; lean_object* v___x_346_; 
v___f_342_ = lean_alloc_closure((void*)(lp_mathlib_WithZero_semilatticeInf___redArg___lam__0), 3, 1);
lean_closure_set(v___f_342_, 0, v_inf_338_);
v___x_343_ = lp_mathlib_WithZero_semilatticeSup___redArg(v_toSemilatticeSup_337_);
v___x_344_ = lean_alloc_closure((void*)(lp_mathlib_WithZero_map_u2082), 6, 4);
lean_closure_set(v___x_344_, 0, lean_box(0));
lean_closure_set(v___x_344_, 1, lean_box(0));
lean_closure_set(v___x_344_, 2, lean_box(0));
lean_closure_set(v___x_344_, 3, v___f_342_);
if (v_isShared_341_ == 0)
{
lean_ctor_set(v___x_340_, 1, v___x_344_);
lean_ctor_set(v___x_340_, 0, v___x_343_);
v___x_346_ = v___x_340_;
goto v_reusejp_345_;
}
else
{
lean_object* v_reuseFailAlloc_347_; 
v_reuseFailAlloc_347_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_347_, 0, v___x_343_);
lean_ctor_set(v_reuseFailAlloc_347_, 1, v___x_344_);
v___x_346_ = v_reuseFailAlloc_347_;
goto v_reusejp_345_;
}
v_reusejp_345_:
{
return v___x_346_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_WithZero_instLattice(lean_object* v_00_u03b1_349_, lean_object* v_inst_350_){
_start:
{
lean_object* v___x_351_; 
v___x_351_ = lp_mathlib_WithZero_instLattice___redArg(v_inst_350_);
return v___x_351_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_WithZero_decidableEq___aux__1___redArg(lean_object* v_inst_352_, lean_object* v_a_353_, lean_object* v_b_354_){
_start:
{
uint8_t v___x_355_; 
v___x_355_ = l_Option_instDecidableEq___redArg(v_inst_352_, v_a_353_, v_b_354_);
return v___x_355_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_WithZero_decidableEq___aux__1___redArg___boxed(lean_object* v_inst_356_, lean_object* v_a_357_, lean_object* v_b_358_){
_start:
{
uint8_t v_res_359_; lean_object* v_r_360_; 
v_res_359_ = lp_mathlib_WithZero_decidableEq___aux__1___redArg(v_inst_356_, v_a_357_, v_b_358_);
v_r_360_ = lean_box(v_res_359_);
return v_r_360_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_WithZero_decidableEq___aux__1(lean_object* v_00_u03b1_361_, lean_object* v_inst_362_, lean_object* v_a_363_, lean_object* v_b_364_){
_start:
{
uint8_t v___x_365_; 
v___x_365_ = l_Option_instDecidableEq___redArg(v_inst_362_, v_a_363_, v_b_364_);
return v___x_365_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_WithZero_decidableEq___aux__1___boxed(lean_object* v_00_u03b1_366_, lean_object* v_inst_367_, lean_object* v_a_368_, lean_object* v_b_369_){
_start:
{
uint8_t v_res_370_; lean_object* v_r_371_; 
v_res_370_ = lp_mathlib_WithZero_decidableEq___aux__1(v_00_u03b1_366_, v_inst_367_, v_a_368_, v_b_369_);
v_r_371_ = lean_box(v_res_370_);
return v_r_371_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_WithZero_decidableEq___redArg(lean_object* v_inst_372_, lean_object* v_a_373_, lean_object* v_b_374_){
_start:
{
uint8_t v___x_375_; 
v___x_375_ = l_Option_instDecidableEq___redArg(v_inst_372_, v_a_373_, v_b_374_);
return v___x_375_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_WithZero_decidableEq___redArg___boxed(lean_object* v_inst_376_, lean_object* v_a_377_, lean_object* v_b_378_){
_start:
{
uint8_t v_res_379_; lean_object* v_r_380_; 
v_res_379_ = lp_mathlib_WithZero_decidableEq___redArg(v_inst_376_, v_a_377_, v_b_378_);
v_r_380_ = lean_box(v_res_379_);
return v_r_380_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_WithZero_decidableEq(lean_object* v_00_u03b1_381_, lean_object* v_inst_382_, lean_object* v_a_383_, lean_object* v_b_384_){
_start:
{
uint8_t v___x_385_; 
v___x_385_ = l_Option_instDecidableEq___redArg(v_inst_382_, v_a_383_, v_b_384_);
return v___x_385_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_WithZero_decidableEq___boxed(lean_object* v_00_u03b1_386_, lean_object* v_inst_387_, lean_object* v_a_388_, lean_object* v_b_389_){
_start:
{
uint8_t v_res_390_; lean_object* v_r_391_; 
v_res_390_ = lp_mathlib_WithZero_decidableEq(v_00_u03b1_386_, v_inst_387_, v_a_388_, v_b_389_);
v_r_391_ = lean_box(v_res_390_);
return v_r_391_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_WithZero_decidableLE___redArg(lean_object* v_inst_392_, lean_object* v_x_393_, lean_object* v_x_394_){
_start:
{
uint8_t v___x_395_; 
v___x_395_ = 1;
if (lean_obj_tag(v_x_393_) == 0)
{
lean_dec(v_x_394_);
lean_dec_ref(v_inst_392_);
return v___x_395_;
}
else
{
lean_object* v_val_396_; uint8_t v___x_397_; 
v_val_396_ = lean_ctor_get(v_x_393_, 0);
lean_inc(v_val_396_);
lean_dec_ref_known(v_x_393_, 1);
v___x_397_ = 0;
if (lean_obj_tag(v_x_394_) == 0)
{
lean_dec(v_val_396_);
lean_dec_ref(v_inst_392_);
return v___x_397_;
}
else
{
lean_object* v_val_398_; lean_object* v___x_399_; uint8_t v___x_400_; 
v_val_398_ = lean_ctor_get(v_x_394_, 0);
lean_inc(v_val_398_);
lean_dec_ref_known(v_x_394_, 1);
v___x_399_ = lean_apply_2(v_inst_392_, v_val_396_, v_val_398_);
v___x_400_ = lean_unbox(v___x_399_);
if (v___x_400_ == 0)
{
return v___x_397_;
}
else
{
return v___x_395_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_WithZero_decidableLE___redArg___boxed(lean_object* v_inst_401_, lean_object* v_x_402_, lean_object* v_x_403_){
_start:
{
uint8_t v_res_404_; lean_object* v_r_405_; 
v_res_404_ = lp_mathlib_WithZero_decidableLE___redArg(v_inst_401_, v_x_402_, v_x_403_);
v_r_405_ = lean_box(v_res_404_);
return v_r_405_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_WithZero_decidableLE(lean_object* v_00_u03b1_406_, lean_object* v_inst_407_, lean_object* v_inst_408_, lean_object* v_x_409_, lean_object* v_x_410_){
_start:
{
uint8_t v___x_411_; 
v___x_411_ = lp_mathlib_WithZero_decidableLE___redArg(v_inst_408_, v_x_409_, v_x_410_);
return v___x_411_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_WithZero_decidableLE___boxed(lean_object* v_00_u03b1_412_, lean_object* v_inst_413_, lean_object* v_inst_414_, lean_object* v_x_415_, lean_object* v_x_416_){
_start:
{
uint8_t v_res_417_; lean_object* v_r_418_; 
v_res_417_ = lp_mathlib_WithZero_decidableLE(v_00_u03b1_412_, v_inst_413_, v_inst_414_, v_x_415_, v_x_416_);
lean_dec_ref(v_inst_413_);
v_r_418_ = lean_box(v_res_417_);
return v_r_418_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_WithZero_decidableLT___redArg(lean_object* v_inst_419_, lean_object* v_x_420_, lean_object* v_x_421_){
_start:
{
uint8_t v___x_422_; 
v___x_422_ = 0;
if (lean_obj_tag(v_x_421_) == 0)
{
lean_dec(v_x_420_);
lean_dec_ref(v_inst_419_);
return v___x_422_;
}
else
{
lean_object* v_val_423_; uint8_t v___x_424_; 
v_val_423_ = lean_ctor_get(v_x_421_, 0);
lean_inc(v_val_423_);
lean_dec_ref_known(v_x_421_, 1);
v___x_424_ = 1;
if (lean_obj_tag(v_x_420_) == 0)
{
lean_dec(v_val_423_);
lean_dec_ref(v_inst_419_);
return v___x_424_;
}
else
{
lean_object* v_val_425_; lean_object* v___x_426_; uint8_t v___x_427_; 
v_val_425_ = lean_ctor_get(v_x_420_, 0);
lean_inc(v_val_425_);
lean_dec_ref_known(v_x_420_, 1);
v___x_426_ = lean_apply_2(v_inst_419_, v_val_425_, v_val_423_);
v___x_427_ = lean_unbox(v___x_426_);
if (v___x_427_ == 0)
{
return v___x_422_;
}
else
{
return v___x_424_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_WithZero_decidableLT___redArg___boxed(lean_object* v_inst_428_, lean_object* v_x_429_, lean_object* v_x_430_){
_start:
{
uint8_t v_res_431_; lean_object* v_r_432_; 
v_res_431_ = lp_mathlib_WithZero_decidableLT___redArg(v_inst_428_, v_x_429_, v_x_430_);
v_r_432_ = lean_box(v_res_431_);
return v_r_432_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_WithZero_decidableLT(lean_object* v_00_u03b1_433_, lean_object* v_inst_434_, lean_object* v_inst_435_, lean_object* v_x_436_, lean_object* v_x_437_){
_start:
{
uint8_t v___x_438_; 
v___x_438_ = lp_mathlib_WithZero_decidableLT___redArg(v_inst_435_, v_x_436_, v_x_437_);
return v___x_438_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_WithZero_decidableLT___boxed(lean_object* v_00_u03b1_439_, lean_object* v_inst_440_, lean_object* v_inst_441_, lean_object* v_x_442_, lean_object* v_x_443_){
_start:
{
uint8_t v_res_444_; lean_object* v_r_445_; 
v_res_444_ = lp_mathlib_WithZero_decidableLT(v_00_u03b1_439_, v_inst_440_, v_inst_441_, v_x_442_, v_x_443_);
lean_dec_ref(v_inst_440_);
v_r_445_ = lean_box(v_res_444_);
return v_r_445_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_WithZero_instLinearOrder___redArg___lam__0(lean_object* v_inst_446_, lean_object* v_a_447_, lean_object* v_b_448_){
_start:
{
lean_object* v_toDecidableEq_449_; lean_object* v___x_450_; uint8_t v___x_451_; 
v_toDecidableEq_449_ = lean_ctor_get(v_inst_446_, 5);
lean_inc_ref(v_toDecidableEq_449_);
lean_dec_ref(v_inst_446_);
v___x_450_ = lean_apply_2(v_toDecidableEq_449_, v_a_447_, v_b_448_);
v___x_451_ = lean_unbox(v___x_450_);
return v___x_451_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_WithZero_instLinearOrder___redArg___lam__0___boxed(lean_object* v_inst_452_, lean_object* v_a_453_, lean_object* v_b_454_){
_start:
{
uint8_t v_res_455_; lean_object* v_r_456_; 
v_res_455_ = lp_mathlib_WithZero_instLinearOrder___redArg___lam__0(v_inst_452_, v_a_453_, v_b_454_);
v_r_456_ = lean_box(v_res_455_);
return v_r_456_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_WithZero_instLinearOrder___redArg___lam__1(lean_object* v___f_457_, lean_object* v_a_458_, lean_object* v_b_459_){
_start:
{
uint8_t v___x_460_; 
v___x_460_ = l_Option_instDecidableEq___redArg(v___f_457_, v_a_458_, v_b_459_);
return v___x_460_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_WithZero_instLinearOrder___redArg___lam__1___boxed(lean_object* v___f_461_, lean_object* v_a_462_, lean_object* v_b_463_){
_start:
{
uint8_t v_res_464_; lean_object* v_r_465_; 
v_res_464_ = lp_mathlib_WithZero_instLinearOrder___redArg___lam__1(v___f_461_, v_a_462_, v_b_463_);
v_r_465_ = lean_box(v_res_464_);
return v_r_465_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_WithZero_instLinearOrder___redArg___lam__2(lean_object* v_inst_466_, lean_object* v_a_467_, lean_object* v_b_468_){
_start:
{
lean_object* v_toDecidableLT_469_; uint8_t v___x_470_; 
v_toDecidableLT_469_ = lean_ctor_get(v_inst_466_, 6);
lean_inc_ref(v_toDecidableLT_469_);
lean_dec_ref(v_inst_466_);
v___x_470_ = lp_mathlib_WithZero_decidableLT___redArg(v_toDecidableLT_469_, v_a_467_, v_b_468_);
return v___x_470_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_WithZero_instLinearOrder___redArg___lam__2___boxed(lean_object* v_inst_471_, lean_object* v_a_472_, lean_object* v_b_473_){
_start:
{
uint8_t v_res_474_; lean_object* v_r_475_; 
v_res_474_ = lp_mathlib_WithZero_instLinearOrder___redArg___lam__2(v_inst_471_, v_a_472_, v_b_473_);
v_r_475_ = lean_box(v_res_474_);
return v_r_475_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_WithZero_instLinearOrder___redArg___lam__3(lean_object* v___f_476_, lean_object* v___f_477_, lean_object* v_a_478_, lean_object* v_b_479_){
_start:
{
lean_object* v___x_480_; uint8_t v___x_481_; 
lean_inc(v_b_479_);
lean_inc(v_a_478_);
v___x_480_ = lean_apply_2(v___f_476_, v_a_478_, v_b_479_);
v___x_481_ = lean_unbox(v___x_480_);
if (v___x_481_ == 0)
{
uint8_t v___x_482_; 
v___x_482_ = l_Option_instDecidableEq___redArg(v___f_477_, v_a_478_, v_b_479_);
if (v___x_482_ == 0)
{
uint8_t v___x_483_; 
v___x_483_ = 2;
return v___x_483_;
}
else
{
uint8_t v___x_484_; 
v___x_484_ = 1;
return v___x_484_;
}
}
else
{
uint8_t v___x_485_; 
lean_dec(v_b_479_);
lean_dec(v_a_478_);
lean_dec_ref(v___f_477_);
v___x_485_ = 0;
return v___x_485_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_WithZero_instLinearOrder___redArg___lam__3___boxed(lean_object* v___f_486_, lean_object* v___f_487_, lean_object* v_a_488_, lean_object* v_b_489_){
_start:
{
uint8_t v_res_490_; lean_object* v_r_491_; 
v_res_490_ = lp_mathlib_WithZero_instLinearOrder___redArg___lam__3(v___f_486_, v___f_487_, v_a_488_, v_b_489_);
v_r_491_ = lean_box(v_res_490_);
return v_r_491_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_WithZero_instLinearOrder___redArg___lam__4(lean_object* v_inst_492_, lean_object* v_a_493_, lean_object* v_b_494_){
_start:
{
lean_object* v_toDecidableLE_495_; uint8_t v___x_496_; 
v_toDecidableLE_495_ = lean_ctor_get(v_inst_492_, 4);
lean_inc_ref(v_toDecidableLE_495_);
lean_dec_ref(v_inst_492_);
v___x_496_ = lp_mathlib_WithZero_decidableLE___redArg(v_toDecidableLE_495_, v_a_493_, v_b_494_);
return v___x_496_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_WithZero_instLinearOrder___redArg___lam__4___boxed(lean_object* v_inst_497_, lean_object* v_a_498_, lean_object* v_b_499_){
_start:
{
uint8_t v_res_500_; lean_object* v_r_501_; 
v_res_500_ = lp_mathlib_WithZero_instLinearOrder___redArg___lam__4(v_inst_497_, v_a_498_, v_b_499_);
v_r_501_ = lean_box(v_res_500_);
return v_r_501_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_WithZero_instLinearOrder___redArg(lean_object* v_inst_502_){
_start:
{
lean_object* v___x_503_; lean_object* v___x_504_; lean_object* v___x_505_; lean_object* v_toPartialOrder_506_; lean_object* v_toSemilatticeSup_507_; lean_object* v___f_508_; lean_object* v___f_509_; lean_object* v___f_510_; lean_object* v___f_511_; lean_object* v___f_512_; lean_object* v___f_513_; lean_object* v___f_514_; lean_object* v___x_515_; 
v___x_503_ = lp_mathlib_LinearOrder_toLattice___redArg(v_inst_502_);
v___x_504_ = lp_mathlib_WithZero_instLattice___redArg(v___x_503_);
lean_inc_ref(v___x_504_);
v___x_505_ = lp_mathlib_Lattice_toSemilatticeInf___redArg(v___x_504_);
v_toPartialOrder_506_ = lean_ctor_get(v___x_505_, 0);
lean_inc_ref(v_toPartialOrder_506_);
v_toSemilatticeSup_507_ = lean_ctor_get(v___x_504_, 0);
lean_inc_ref(v_toSemilatticeSup_507_);
lean_dec_ref(v___x_504_);
lean_inc_ref_n(v_inst_502_, 2);
v___f_508_ = lean_alloc_closure((void*)(lp_mathlib_WithZero_instLinearOrder___redArg___lam__0___boxed), 3, 1);
lean_closure_set(v___f_508_, 0, v_inst_502_);
lean_inc_ref(v___f_508_);
v___f_509_ = lean_alloc_closure((void*)(lp_mathlib_WithZero_instLinearOrder___redArg___lam__1___boxed), 3, 1);
lean_closure_set(v___f_509_, 0, v___f_508_);
v___f_510_ = lean_alloc_closure((void*)(lp_mathlib_WithZero_instLinearOrder___redArg___lam__2___boxed), 3, 1);
lean_closure_set(v___f_510_, 0, v_inst_502_);
lean_inc_ref(v___f_510_);
v___f_511_ = lean_alloc_closure((void*)(lp_mathlib_WithZero_instLinearOrder___redArg___lam__3___boxed), 4, 2);
lean_closure_set(v___f_511_, 0, v___f_510_);
lean_closure_set(v___f_511_, 1, v___f_508_);
v___f_512_ = lean_alloc_closure((void*)(lp_mathlib_WithZero_instLinearOrder___redArg___lam__4___boxed), 3, 1);
lean_closure_set(v___f_512_, 0, v_inst_502_);
v___f_513_ = lean_alloc_closure((void*)(lp_mathlib_SemilatticeInf_toMin___redArg___lam__0), 3, 1);
lean_closure_set(v___f_513_, 0, v___x_505_);
v___f_514_ = lean_alloc_closure((void*)(lp_mathlib_SemilatticeSup_toMax___redArg___lam__0), 3, 1);
lean_closure_set(v___f_514_, 0, v_toSemilatticeSup_507_);
v___x_515_ = lean_alloc_ctor(0, 7, 0);
lean_ctor_set(v___x_515_, 0, v_toPartialOrder_506_);
lean_ctor_set(v___x_515_, 1, v___f_513_);
lean_ctor_set(v___x_515_, 2, v___f_514_);
lean_ctor_set(v___x_515_, 3, v___f_511_);
lean_ctor_set(v___x_515_, 4, v___f_512_);
lean_ctor_set(v___x_515_, 5, v___f_509_);
lean_ctor_set(v___x_515_, 6, v___f_510_);
return v___x_515_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_WithZero_instLinearOrder(lean_object* v_00_u03b1_516_, lean_object* v_inst_517_){
_start:
{
lean_object* v___x_518_; 
v___x_518_ = lp_mathlib_WithZero_instLinearOrder___redArg(v_inst_517_);
return v___x_518_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_WithZero_instLinearOrderedCommMonoidWithZero___redArg(lean_object* v_inst_519_, lean_object* v_inst_520_){
_start:
{
lean_object* v___x_521_; lean_object* v___x_522_; lean_object* v___x_523_; lean_object* v___x_524_; 
v___x_521_ = lp_mathlib_WithZero_instCommMonoidWithZero___redArg(v_inst_519_);
v___x_522_ = lp_mathlib_WithZero_instLinearOrder___redArg(v_inst_520_);
v___x_523_ = lean_box(0);
v___x_524_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_524_, 0, v___x_521_);
lean_ctor_set(v___x_524_, 1, v___x_522_);
lean_ctor_set(v___x_524_, 2, v___x_523_);
return v___x_524_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_WithZero_instLinearOrderedCommMonoidWithZero(lean_object* v_00_u03b1_525_, lean_object* v_inst_526_, lean_object* v_inst_527_, lean_object* v_inst_528_){
_start:
{
lean_object* v___x_529_; 
v___x_529_ = lp_mathlib_WithZero_instLinearOrderedCommMonoidWithZero___redArg(v_inst_526_, v_inst_527_);
return v___x_529_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_WithZero_instLinearOrderedCommGroupWithZero___redArg(lean_object* v_inst_530_, lean_object* v_inst_531_){
_start:
{
lean_object* v_toMonoid_532_; lean_object* v___x_533_; lean_object* v___x_534_; lean_object* v_toInv_535_; lean_object* v_toDiv_536_; lean_object* v_toZPow_537_; lean_object* v___x_539_; uint8_t v_isShared_540_; uint8_t v_isSharedCheck_544_; 
v_toMonoid_532_ = lean_ctor_get(v_inst_530_, 0);
lean_inc_ref(v_toMonoid_532_);
v___x_533_ = lp_mathlib_WithZero_instLinearOrderedCommMonoidWithZero___redArg(v_toMonoid_532_, v_inst_531_);
v___x_534_ = lp_mathlib_WithZero_instCommGroupWithZero___redArg(v_inst_530_);
v_toInv_535_ = lean_ctor_get(v___x_534_, 1);
v_toDiv_536_ = lean_ctor_get(v___x_534_, 2);
v_toZPow_537_ = lean_ctor_get(v___x_534_, 3);
v_isSharedCheck_544_ = !lean_is_exclusive(v___x_534_);
if (v_isSharedCheck_544_ == 0)
{
lean_object* v_unused_545_; 
v_unused_545_ = lean_ctor_get(v___x_534_, 0);
lean_dec(v_unused_545_);
v___x_539_ = v___x_534_;
v_isShared_540_ = v_isSharedCheck_544_;
goto v_resetjp_538_;
}
else
{
lean_inc(v_toZPow_537_);
lean_inc(v_toDiv_536_);
lean_inc(v_toInv_535_);
lean_dec(v___x_534_);
v___x_539_ = lean_box(0);
v_isShared_540_ = v_isSharedCheck_544_;
goto v_resetjp_538_;
}
v_resetjp_538_:
{
lean_object* v___x_542_; 
if (v_isShared_540_ == 0)
{
lean_ctor_set(v___x_539_, 0, v___x_533_);
v___x_542_ = v___x_539_;
goto v_reusejp_541_;
}
else
{
lean_object* v_reuseFailAlloc_543_; 
v_reuseFailAlloc_543_ = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(v_reuseFailAlloc_543_, 0, v___x_533_);
lean_ctor_set(v_reuseFailAlloc_543_, 1, v_toInv_535_);
lean_ctor_set(v_reuseFailAlloc_543_, 2, v_toDiv_536_);
lean_ctor_set(v_reuseFailAlloc_543_, 3, v_toZPow_537_);
v___x_542_ = v_reuseFailAlloc_543_;
goto v_reusejp_541_;
}
v_reusejp_541_:
{
return v___x_542_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_WithZero_instLinearOrderedCommGroupWithZero(lean_object* v_00_u03b1_546_, lean_object* v_inst_547_, lean_object* v_inst_548_, lean_object* v_inst_549_){
_start:
{
lean_object* v___x_550_; 
v___x_550_ = lp_mathlib_WithZero_instLinearOrderedCommGroupWithZero___redArg(v_inst_547_, v_inst_548_);
return v___x_550_;
}
}
static lean_object* _init_lp_mathlib_WithZero_instLinearOrderedCommGroupWithZeroMultiplicativeInt___closed__0(void){
_start:
{
lean_object* v___x_551_; lean_object* v___x_552_; 
v___x_551_ = lp_mathlib_Int_instAddCommGroup;
v___x_552_ = lp_mathlib_Multiplicative_divInvMonoid___redArg(v___x_551_);
return v___x_552_;
}
}
static lean_object* _init_lp_mathlib_WithZero_instLinearOrderedCommGroupWithZeroMultiplicativeInt___closed__1(void){
_start:
{
lean_object* v___x_553_; lean_object* v___x_554_; 
v___x_553_ = lp_mathlib_Int_instLinearOrder;
v___x_554_ = lp_mathlib_Multiplicative_linearOrder___redArg(v___x_553_);
return v___x_554_;
}
}
static lean_object* _init_lp_mathlib_WithZero_instLinearOrderedCommGroupWithZeroMultiplicativeInt___closed__2(void){
_start:
{
lean_object* v___x_555_; lean_object* v___x_556_; lean_object* v___x_557_; 
v___x_555_ = lean_obj_once(&lp_mathlib_WithZero_instLinearOrderedCommGroupWithZeroMultiplicativeInt___closed__1, &lp_mathlib_WithZero_instLinearOrderedCommGroupWithZeroMultiplicativeInt___closed__1_once, _init_lp_mathlib_WithZero_instLinearOrderedCommGroupWithZeroMultiplicativeInt___closed__1);
v___x_556_ = lean_obj_once(&lp_mathlib_WithZero_instLinearOrderedCommGroupWithZeroMultiplicativeInt___closed__0, &lp_mathlib_WithZero_instLinearOrderedCommGroupWithZeroMultiplicativeInt___closed__0_once, _init_lp_mathlib_WithZero_instLinearOrderedCommGroupWithZeroMultiplicativeInt___closed__0);
v___x_557_ = lp_mathlib_WithZero_instLinearOrderedCommGroupWithZero___redArg(v___x_556_, v___x_555_);
return v___x_557_;
}
}
static lean_object* _init_lp_mathlib_WithZero_instLinearOrderedCommGroupWithZeroMultiplicativeInt(void){
_start:
{
lean_object* v___x_558_; 
v___x_558_ = lean_obj_once(&lp_mathlib_WithZero_instLinearOrderedCommGroupWithZeroMultiplicativeInt___closed__2, &lp_mathlib_WithZero_instLinearOrderedCommGroupWithZeroMultiplicativeInt___closed__2_once, _init_lp_mathlib_WithZero_instLinearOrderedCommGroupWithZeroMultiplicativeInt___closed__2);
return v___x_558_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_WithZero_expOrderIso___redArg(lean_object* v_inst_559_){
_start:
{
lean_object* v___x_560_; 
v___x_560_ = lp_mathlib_WithZero_expEquiv___redArg(v_inst_559_);
return v___x_560_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_WithZero_expOrderIso(lean_object* v_G_561_, lean_object* v_inst_562_, lean_object* v_inst_563_){
_start:
{
lean_object* v___x_564_; 
v___x_564_ = lp_mathlib_WithZero_expEquiv___redArg(v_inst_563_);
return v___x_564_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_WithZero_expOrderIso___boxed(lean_object* v_G_565_, lean_object* v_inst_566_, lean_object* v_inst_567_){
_start:
{
lean_object* v_res_568_; 
v_res_568_ = lp_mathlib_WithZero_expOrderIso(v_G_565_, v_inst_566_, v_inst_567_);
lean_dec_ref(v_inst_566_);
return v_res_568_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_WithZero_logOrderIso___redArg(lean_object* v_inst_569_){
_start:
{
lean_object* v___x_570_; 
v___x_570_ = lp_mathlib_WithZero_logEquiv___redArg(v_inst_569_);
return v___x_570_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_WithZero_logOrderIso(lean_object* v_G_571_, lean_object* v_inst_572_, lean_object* v_inst_573_){
_start:
{
lean_object* v___x_574_; 
v___x_574_ = lp_mathlib_WithZero_logEquiv___redArg(v_inst_573_);
return v___x_574_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_WithZero_logOrderIso___boxed(lean_object* v_G_575_, lean_object* v_inst_576_, lean_object* v_inst_577_){
_start:
{
lean_object* v_res_578_; 
v_res_578_ = lp_mathlib_WithZero_logOrderIso(v_G_575_, v_inst_576_, v_inst_577_);
lean_dec_ref(v_inst_576_);
return v_res_578_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Group_WithOne_Map(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_GroupWithZero_InjSurj(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_GroupWithZero_Regular(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_GroupWithZero_WithZero(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Order_AddGroupWithTop(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Order_Group_Defs(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Order_Group_Int(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Order_Group_Units(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Order_GroupWithZero_Basic(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Order_Monoid_OrderDual(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Order_Monoid_TypeTags(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Data_Int_Basic(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Data_Set_Function(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Order_GroupWithZero_Canonical(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Group_WithOne_Map(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_GroupWithZero_InjSurj(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_GroupWithZero_Regular(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_GroupWithZero_WithZero(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Order_AddGroupWithTop(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Order_Group_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Order_Group_Int(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Order_Group_Units(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Order_GroupWithZero_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Order_Monoid_OrderDual(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Order_Monoid_TypeTags(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Data_Int_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Data_Set_Function(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
lp_mathlib_WithZero_instLinearOrderedCommGroupWithZeroMultiplicativeInt = _init_lp_mathlib_WithZero_instLinearOrderedCommGroupWithZeroMultiplicativeInt();
lean_mark_persistent(lp_mathlib_WithZero_instLinearOrderedCommGroupWithZeroMultiplicativeInt);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_Algebra_Order_GroupWithZero_Canonical(uint8_t builtin) {
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
lean_object* initialize_mathlib_Mathlib_Algebra_Group_WithOne_Map(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_GroupWithZero_InjSurj(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_GroupWithZero_Regular(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_GroupWithZero_WithZero(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Order_AddGroupWithTop(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Order_Group_Defs(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Order_Group_Int(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Order_Group_Units(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Order_GroupWithZero_Basic(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Order_Monoid_OrderDual(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Order_Monoid_TypeTags(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Data_Int_Basic(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Data_Set_Function(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Algebra_Order_GroupWithZero_Canonical(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Group_WithOne_Map(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_GroupWithZero_InjSurj(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_GroupWithZero_Regular(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_GroupWithZero_WithZero(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Order_AddGroupWithTop(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Order_Group_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Order_Group_Int(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Order_Group_Units(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Order_GroupWithZero_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Order_Monoid_OrderDual(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Order_Monoid_TypeTags(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_Int_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_Set_Function(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Order_GroupWithZero_Canonical(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_Algebra_Order_GroupWithZero_Canonical(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_Algebra_Order_GroupWithZero_Canonical(builtin);
}
#ifdef __cplusplus
}
#endif
