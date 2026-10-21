// Lean compiler output
// Module: Mathlib.Algebra.Order.Monoid.Unbundled.WithTop
// Imports: public import Init public meta import Init public import Mathlib.Algebra.CharZero.Defs public import Mathlib.Algebra.Group.Hom.Defs public import Mathlib.Algebra.Group.Equiv.Defs public import Mathlib.Algebra.Order.Monoid.Unbundled.ExistsOfLE public import Mathlib.Algebra.Order.ZeroLEOne public import Mathlib.Order.WithBot
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
lean_object* lp_mathlib_AddMonoid_toAddZeroClass___redArg(lean_object*);
lean_object* lp_mathlib_AddZeroClass_toAddZero___redArg(lean_object*);
lean_object* lp_mathlib_WithBot_map_u2082(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
uint8_t lean_nat_dec_eq(lean_object*, lean_object*);
lean_object* lp_mathlib_WithTop_map(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_WithTop_map_u2082(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lean_nat_sub(lean_object*, lean_object*);
lean_object* lp_mathlib_WithTop_some(lean_object*, lean_object*);
lean_object* lp_mathlib_WithBot_map(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Equiv_withBotCongr___redArg(lean_object*);
lean_object* lp_mathlib_Equiv_withTopCongr___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_WithTop_one___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_WithTop_one(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_WithTop_zero___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_WithTop_zero(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_WithTop_add___redArg___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_WithTop_add___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_WithTop_add(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_WithTop_addSemigroup___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_WithTop_addSemigroup(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_WithTop_addCommSemigroup___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_WithTop_addCommSemigroup(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_WithTop_addZeroClass___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_WithTop_addZeroClass(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Algebra_Order_Monoid_Unbundled_WithTop_0__WithTop_addMonoid_match__1_splitter___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Algebra_Order_Monoid_Unbundled_WithTop_0__WithTop_addMonoid_match__1_splitter(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_WithTop_addMonoid___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_WithTop_addMonoid___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_WithTop_addMonoid___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_WithTop_addMonoid(lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_WithTop_addHom___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*1, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_WithTop_some, .m_arity = 2, .m_num_fixed = 1, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1))} };
static const lean_object* lp_mathlib_WithTop_addHom___closed__0 = (const lean_object*)&lp_mathlib_WithTop_addHom___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_WithTop_addHom(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_WithTop_addHom___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_WithTop_addCommMonoid___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_WithTop_addCommMonoid(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_WithTop_natCast___redArg___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_WithTop_natCast___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_WithTop_natCast(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_WithTop_addMonoidWithOne___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_WithTop_addMonoidWithOne(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_WithTop_addCommMonoidWithOne___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_WithTop_addCommMonoidWithOne(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OneHom_withTopMap___redArg___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OneHom_withTopMap___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OneHom_withTopMap(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OneHom_withTopMap___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ZeroHom_withTopMap___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ZeroHom_withTopMap(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ZeroHom_withTopMap___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddHom_withTopMap___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddHom_withTopMap(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddHom_withTopMap___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddMonoidHom_withTopMap___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddMonoidHom_withTopMap(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddMonoidHom_withTopMap___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_WithBot_one___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_WithBot_one(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_WithBot_zero___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_WithBot_zero(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_WithBot_add___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_WithBot_add(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_WithBot_addSemigroup___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_WithBot_addSemigroup(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_WithBot_addCommSemigroup___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_WithBot_addCommSemigroup(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_WithBot_addZeroClass___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_WithBot_addZeroClass(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_WithBot_addMonoid___aux__4___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_WithBot_addMonoid___aux__4(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_WithBot_addMonoid___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_WithBot_addMonoid(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_WithBot_addHom(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_WithBot_addHom___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_WithBot_addCommMonoid___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_WithBot_addCommMonoid(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_WithBot_instNatCast___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_WithBot_instNatCast(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_WithBot_addMonoidWithOne___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_WithBot_addMonoidWithOne(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_WithBot_addCommMonoidWithOne___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_WithBot_addCommMonoidWithOne(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OneHom_withBotMap___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OneHom_withBotMap(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OneHom_withBotMap___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ZeroHom_withBotMap___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ZeroHom_withBotMap(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ZeroHom_withBotMap___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddHom_withBotMap___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddHom_withBotMap(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddHom_withBotMap___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddMonoidHom_withBotMap___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddMonoidHom_withBotMap(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddMonoidHom_withBotMap___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddEquiv_withBotCongr___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddEquiv_withBotCongr(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddEquiv_withBotCongr___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddEquiv_withTopCongr___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddEquiv_withTopCongr(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddEquiv_withTopCongr___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_WithTop_one___redArg(lean_object* v_inst_1_){
_start:
{
lean_object* v___x_2_; 
v___x_2_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_2_, 0, v_inst_1_);
return v___x_2_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_WithTop_one(lean_object* v_00_u03b1_3_, lean_object* v_inst_4_){
_start:
{
lean_object* v___x_5_; 
v___x_5_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_5_, 0, v_inst_4_);
return v___x_5_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_WithTop_zero___redArg(lean_object* v_inst_6_){
_start:
{
lean_object* v___x_7_; 
v___x_7_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_7_, 0, v_inst_6_);
return v___x_7_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_WithTop_zero(lean_object* v_00_u03b1_8_, lean_object* v_inst_9_){
_start:
{
lean_object* v___x_10_; 
v___x_10_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_10_, 0, v_inst_9_);
return v___x_10_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_WithTop_add___redArg___lam__0(lean_object* v_inst_11_, lean_object* v_x1_12_, lean_object* v_x2_13_){
_start:
{
lean_object* v___x_14_; 
v___x_14_ = lean_apply_2(v_inst_11_, v_x1_12_, v_x2_13_);
return v___x_14_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_WithTop_add___redArg(lean_object* v_inst_15_){
_start:
{
lean_object* v___f_16_; lean_object* v___x_17_; 
v___f_16_ = lean_alloc_closure((void*)(lp_mathlib_WithTop_add___redArg___lam__0), 3, 1);
lean_closure_set(v___f_16_, 0, v_inst_15_);
v___x_17_ = lean_alloc_closure((void*)(lp_mathlib_WithTop_map_u2082), 6, 4);
lean_closure_set(v___x_17_, 0, lean_box(0));
lean_closure_set(v___x_17_, 1, lean_box(0));
lean_closure_set(v___x_17_, 2, lean_box(0));
lean_closure_set(v___x_17_, 3, v___f_16_);
return v___x_17_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_WithTop_add(lean_object* v_00_u03b1_18_, lean_object* v_inst_19_){
_start:
{
lean_object* v___x_20_; 
v___x_20_ = lp_mathlib_WithTop_add___redArg(v_inst_19_);
return v___x_20_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_WithTop_addSemigroup___redArg(lean_object* v_inst_21_){
_start:
{
lean_object* v___x_22_; 
v___x_22_ = lp_mathlib_WithTop_add___redArg(v_inst_21_);
return v___x_22_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_WithTop_addSemigroup(lean_object* v_00_u03b1_23_, lean_object* v_inst_24_){
_start:
{
lean_object* v___x_25_; 
v___x_25_ = lp_mathlib_WithTop_add___redArg(v_inst_24_);
return v___x_25_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_WithTop_addCommSemigroup___redArg(lean_object* v_inst_26_){
_start:
{
lean_object* v___x_27_; 
v___x_27_ = lp_mathlib_WithTop_add___redArg(v_inst_26_);
return v___x_27_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_WithTop_addCommSemigroup(lean_object* v_00_u03b1_28_, lean_object* v_inst_29_){
_start:
{
lean_object* v___x_30_; 
v___x_30_ = lp_mathlib_WithTop_add___redArg(v_inst_29_);
return v___x_30_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_WithTop_addZeroClass___redArg(lean_object* v_inst_31_){
_start:
{
lean_object* v___x_32_; lean_object* v_toZero_33_; lean_object* v_toAdd_34_; lean_object* v___x_36_; uint8_t v_isShared_37_; uint8_t v_isSharedCheck_43_; 
v___x_32_ = lp_mathlib_AddZeroClass_toAddZero___redArg(v_inst_31_);
v_toZero_33_ = lean_ctor_get(v___x_32_, 0);
v_toAdd_34_ = lean_ctor_get(v___x_32_, 1);
v_isSharedCheck_43_ = !lean_is_exclusive(v___x_32_);
if (v_isSharedCheck_43_ == 0)
{
v___x_36_ = v___x_32_;
v_isShared_37_ = v_isSharedCheck_43_;
goto v_resetjp_35_;
}
else
{
lean_inc(v_toAdd_34_);
lean_inc(v_toZero_33_);
lean_dec(v___x_32_);
v___x_36_ = lean_box(0);
v_isShared_37_ = v_isSharedCheck_43_;
goto v_resetjp_35_;
}
v_resetjp_35_:
{
lean_object* v___x_38_; lean_object* v___x_39_; lean_object* v___x_41_; 
v___x_38_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_38_, 0, v_toZero_33_);
v___x_39_ = lp_mathlib_WithTop_add___redArg(v_toAdd_34_);
if (v_isShared_37_ == 0)
{
lean_ctor_set(v___x_36_, 1, v___x_39_);
lean_ctor_set(v___x_36_, 0, v___x_38_);
v___x_41_ = v___x_36_;
goto v_reusejp_40_;
}
else
{
lean_object* v_reuseFailAlloc_42_; 
v_reuseFailAlloc_42_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_42_, 0, v___x_38_);
lean_ctor_set(v_reuseFailAlloc_42_, 1, v___x_39_);
v___x_41_ = v_reuseFailAlloc_42_;
goto v_reusejp_40_;
}
v_reusejp_40_:
{
return v___x_41_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_WithTop_addZeroClass(lean_object* v_00_u03b1_44_, lean_object* v_inst_45_){
_start:
{
lean_object* v___x_46_; 
v___x_46_ = lp_mathlib_WithTop_addZeroClass___redArg(v_inst_45_);
return v___x_46_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Algebra_Order_Monoid_Unbundled_WithTop_0__WithTop_addMonoid_match__1_splitter___redArg(lean_object* v_a_47_, lean_object* v_n_48_, lean_object* v_h__1_49_, lean_object* v_h__2_50_, lean_object* v_h__3_51_){
_start:
{
if (lean_obj_tag(v_a_47_) == 0)
{
lean_object* v_zero_52_; uint8_t v_isZero_53_; 
lean_dec(v_h__1_49_);
v_zero_52_ = lean_unsigned_to_nat(0u);
v_isZero_53_ = lean_nat_dec_eq(v_n_48_, v_zero_52_);
if (v_isZero_53_ == 1)
{
lean_object* v___x_54_; lean_object* v___x_55_; 
lean_dec(v_h__3_51_);
lean_dec(v_n_48_);
v___x_54_ = lean_box(0);
v___x_55_ = lean_apply_1(v_h__2_50_, v___x_54_);
return v___x_55_;
}
else
{
lean_object* v_one_56_; lean_object* v_n_57_; lean_object* v___x_58_; 
lean_dec(v_h__2_50_);
v_one_56_ = lean_unsigned_to_nat(1u);
v_n_57_ = lean_nat_sub(v_n_48_, v_one_56_);
lean_dec(v_n_48_);
v___x_58_ = lean_apply_1(v_h__3_51_, v_n_57_);
return v___x_58_;
}
}
else
{
lean_object* v_val_59_; lean_object* v___x_60_; 
lean_dec(v_h__3_51_);
lean_dec(v_h__2_50_);
v_val_59_ = lean_ctor_get(v_a_47_, 0);
lean_inc(v_val_59_);
lean_dec_ref_known(v_a_47_, 1);
v___x_60_ = lean_apply_2(v_h__1_49_, v_val_59_, v_n_48_);
return v___x_60_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Algebra_Order_Monoid_Unbundled_WithTop_0__WithTop_addMonoid_match__1_splitter(lean_object* v_00_u03b1_61_, lean_object* v_motive_62_, lean_object* v_a_63_, lean_object* v_n_64_, lean_object* v_h__1_65_, lean_object* v_h__2_66_, lean_object* v_h__3_67_){
_start:
{
if (lean_obj_tag(v_a_63_) == 0)
{
lean_object* v_zero_68_; uint8_t v_isZero_69_; 
lean_dec(v_h__1_65_);
v_zero_68_ = lean_unsigned_to_nat(0u);
v_isZero_69_ = lean_nat_dec_eq(v_n_64_, v_zero_68_);
if (v_isZero_69_ == 1)
{
lean_object* v___x_70_; lean_object* v___x_71_; 
lean_dec(v_h__3_67_);
lean_dec(v_n_64_);
v___x_70_ = lean_box(0);
v___x_71_ = lean_apply_1(v_h__2_66_, v___x_70_);
return v___x_71_;
}
else
{
lean_object* v_one_72_; lean_object* v_n_73_; lean_object* v___x_74_; 
lean_dec(v_h__2_66_);
v_one_72_ = lean_unsigned_to_nat(1u);
v_n_73_ = lean_nat_sub(v_n_64_, v_one_72_);
lean_dec(v_n_64_);
v___x_74_ = lean_apply_1(v_h__3_67_, v_n_73_);
return v___x_74_;
}
}
else
{
lean_object* v_val_75_; lean_object* v___x_76_; 
lean_dec(v_h__3_67_);
lean_dec(v_h__2_66_);
v_val_75_ = lean_ctor_get(v_a_63_, 0);
lean_inc(v_val_75_);
lean_dec_ref_known(v_a_63_, 1);
v___x_76_ = lean_apply_2(v_h__1_65_, v_val_75_, v_n_64_);
return v___x_76_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_WithTop_addMonoid___redArg___lam__0(lean_object* v_toZero_77_, lean_object* v___x_78_, lean_object* v_toNSMul_79_, lean_object* v_n_80_, lean_object* v_a_81_){
_start:
{
if (lean_obj_tag(v_a_81_) == 0)
{
lean_object* v_zero_82_; uint8_t v_isZero_83_; 
lean_dec(v_toNSMul_79_);
v_zero_82_ = lean_unsigned_to_nat(0u);
v_isZero_83_ = lean_nat_dec_eq(v_n_80_, v_zero_82_);
lean_dec(v_n_80_);
if (v_isZero_83_ == 1)
{
lean_object* v___x_84_; 
v___x_84_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_84_, 0, v_toZero_77_);
return v___x_84_;
}
else
{
lean_dec(v_toZero_77_);
lean_inc(v___x_78_);
return v___x_78_;
}
}
else
{
lean_object* v_val_85_; lean_object* v___x_87_; uint8_t v_isShared_88_; uint8_t v_isSharedCheck_93_; 
lean_dec(v_toZero_77_);
v_val_85_ = lean_ctor_get(v_a_81_, 0);
v_isSharedCheck_93_ = !lean_is_exclusive(v_a_81_);
if (v_isSharedCheck_93_ == 0)
{
v___x_87_ = v_a_81_;
v_isShared_88_ = v_isSharedCheck_93_;
goto v_resetjp_86_;
}
else
{
lean_inc(v_val_85_);
lean_dec(v_a_81_);
v___x_87_ = lean_box(0);
v_isShared_88_ = v_isSharedCheck_93_;
goto v_resetjp_86_;
}
v_resetjp_86_:
{
lean_object* v___x_89_; lean_object* v___x_91_; 
v___x_89_ = lean_apply_2(v_toNSMul_79_, v_n_80_, v_val_85_);
if (v_isShared_88_ == 0)
{
lean_ctor_set(v___x_87_, 0, v___x_89_);
v___x_91_ = v___x_87_;
goto v_reusejp_90_;
}
else
{
lean_object* v_reuseFailAlloc_92_; 
v_reuseFailAlloc_92_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_92_, 0, v___x_89_);
v___x_91_ = v_reuseFailAlloc_92_;
goto v_reusejp_90_;
}
v_reusejp_90_:
{
return v___x_91_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_WithTop_addMonoid___redArg___lam__0___boxed(lean_object* v_toZero_94_, lean_object* v___x_95_, lean_object* v_toNSMul_96_, lean_object* v_n_97_, lean_object* v_a_98_){
_start:
{
lean_object* v_res_99_; 
v_res_99_ = lp_mathlib_WithTop_addMonoid___redArg___lam__0(v_toZero_94_, v___x_95_, v_toNSMul_96_, v_n_97_, v_a_98_);
lean_dec(v___x_95_);
return v_res_99_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_WithTop_addMonoid___redArg(lean_object* v_inst_100_){
_start:
{
lean_object* v_toAdd_101_; lean_object* v_toNSMul_102_; lean_object* v___x_103_; lean_object* v___x_105_; uint8_t v_isShared_106_; uint8_t v_isSharedCheck_117_; 
v_toAdd_101_ = lean_ctor_get(v_inst_100_, 1);
lean_inc(v_toAdd_101_);
v_toNSMul_102_ = lean_ctor_get(v_inst_100_, 2);
lean_inc(v_toNSMul_102_);
v___x_103_ = lp_mathlib_AddMonoid_toAddZeroClass___redArg(v_inst_100_);
v_isSharedCheck_117_ = !lean_is_exclusive(v_inst_100_);
if (v_isSharedCheck_117_ == 0)
{
lean_object* v_unused_118_; lean_object* v_unused_119_; lean_object* v_unused_120_; 
v_unused_118_ = lean_ctor_get(v_inst_100_, 2);
lean_dec(v_unused_118_);
v_unused_119_ = lean_ctor_get(v_inst_100_, 1);
lean_dec(v_unused_119_);
v_unused_120_ = lean_ctor_get(v_inst_100_, 0);
lean_dec(v_unused_120_);
v___x_105_ = v_inst_100_;
v_isShared_106_ = v_isSharedCheck_117_;
goto v_resetjp_104_;
}
else
{
lean_dec(v_inst_100_);
v___x_105_ = lean_box(0);
v_isShared_106_ = v_isSharedCheck_117_;
goto v_resetjp_104_;
}
v_resetjp_104_:
{
lean_object* v___x_107_; lean_object* v_toZero_108_; lean_object* v___x_109_; lean_object* v_toZero_110_; lean_object* v___x_111_; lean_object* v___x_112_; lean_object* v___f_113_; lean_object* v___x_115_; 
lean_inc_ref(v___x_103_);
v___x_107_ = lp_mathlib_WithTop_addZeroClass___redArg(v___x_103_);
v_toZero_108_ = lean_ctor_get(v___x_107_, 0);
lean_inc(v_toZero_108_);
lean_dec_ref(v___x_107_);
v___x_109_ = lp_mathlib_AddZeroClass_toAddZero___redArg(v___x_103_);
v_toZero_110_ = lean_ctor_get(v___x_109_, 0);
lean_inc(v_toZero_110_);
lean_dec_ref(v___x_109_);
v___x_111_ = lp_mathlib_WithTop_add___redArg(v_toAdd_101_);
v___x_112_ = lean_box(0);
v___f_113_ = lean_alloc_closure((void*)(lp_mathlib_WithTop_addMonoid___redArg___lam__0___boxed), 5, 3);
lean_closure_set(v___f_113_, 0, v_toZero_110_);
lean_closure_set(v___f_113_, 1, v___x_112_);
lean_closure_set(v___f_113_, 2, v_toNSMul_102_);
if (v_isShared_106_ == 0)
{
lean_ctor_set(v___x_105_, 2, v___f_113_);
lean_ctor_set(v___x_105_, 1, v___x_111_);
lean_ctor_set(v___x_105_, 0, v_toZero_108_);
v___x_115_ = v___x_105_;
goto v_reusejp_114_;
}
else
{
lean_object* v_reuseFailAlloc_116_; 
v_reuseFailAlloc_116_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v_reuseFailAlloc_116_, 0, v_toZero_108_);
lean_ctor_set(v_reuseFailAlloc_116_, 1, v___x_111_);
lean_ctor_set(v_reuseFailAlloc_116_, 2, v___f_113_);
v___x_115_ = v_reuseFailAlloc_116_;
goto v_reusejp_114_;
}
v_reusejp_114_:
{
return v___x_115_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_WithTop_addMonoid(lean_object* v_00_u03b1_121_, lean_object* v_inst_122_){
_start:
{
lean_object* v___x_123_; 
v___x_123_ = lp_mathlib_WithTop_addMonoid___redArg(v_inst_122_);
return v___x_123_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_WithTop_addHom(lean_object* v_00_u03b1_125_, lean_object* v_inst_126_){
_start:
{
lean_object* v___x_127_; 
v___x_127_ = ((lean_object*)(lp_mathlib_WithTop_addHom___closed__0));
return v___x_127_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_WithTop_addHom___boxed(lean_object* v_00_u03b1_128_, lean_object* v_inst_129_){
_start:
{
lean_object* v_res_130_; 
v_res_130_ = lp_mathlib_WithTop_addHom(v_00_u03b1_128_, v_inst_129_);
lean_dec_ref(v_inst_129_);
return v_res_130_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_WithTop_addCommMonoid___redArg(lean_object* v_inst_131_){
_start:
{
lean_object* v___x_132_; 
v___x_132_ = lp_mathlib_WithTop_addMonoid___redArg(v_inst_131_);
return v___x_132_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_WithTop_addCommMonoid(lean_object* v_00_u03b1_133_, lean_object* v_inst_134_){
_start:
{
lean_object* v___x_135_; 
v___x_135_ = lp_mathlib_WithTop_addMonoid___redArg(v_inst_134_);
return v___x_135_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_WithTop_natCast___redArg___lam__0(lean_object* v_inst_136_, lean_object* v_n_137_){
_start:
{
lean_object* v___x_138_; lean_object* v___x_139_; 
v___x_138_ = lean_apply_1(v_inst_136_, v_n_137_);
v___x_139_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_139_, 0, v___x_138_);
return v___x_139_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_WithTop_natCast___redArg(lean_object* v_inst_140_){
_start:
{
lean_object* v___f_141_; 
v___f_141_ = lean_alloc_closure((void*)(lp_mathlib_WithTop_natCast___redArg___lam__0), 2, 1);
lean_closure_set(v___f_141_, 0, v_inst_140_);
return v___f_141_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_WithTop_natCast(lean_object* v_00_u03b1_142_, lean_object* v_inst_143_){
_start:
{
lean_object* v___f_144_; 
v___f_144_ = lean_alloc_closure((void*)(lp_mathlib_WithTop_natCast___redArg___lam__0), 2, 1);
lean_closure_set(v___f_144_, 0, v_inst_143_);
return v___f_144_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_WithTop_addMonoidWithOne___redArg(lean_object* v_inst_145_){
_start:
{
lean_object* v_toNatCast_146_; lean_object* v_toAddMonoid_147_; lean_object* v_toOne_148_; lean_object* v___x_150_; uint8_t v_isShared_151_; uint8_t v_isSharedCheck_158_; 
v_toNatCast_146_ = lean_ctor_get(v_inst_145_, 0);
v_toAddMonoid_147_ = lean_ctor_get(v_inst_145_, 1);
v_toOne_148_ = lean_ctor_get(v_inst_145_, 2);
v_isSharedCheck_158_ = !lean_is_exclusive(v_inst_145_);
if (v_isSharedCheck_158_ == 0)
{
v___x_150_ = v_inst_145_;
v_isShared_151_ = v_isSharedCheck_158_;
goto v_resetjp_149_;
}
else
{
lean_inc(v_toOne_148_);
lean_inc(v_toAddMonoid_147_);
lean_inc(v_toNatCast_146_);
lean_dec(v_inst_145_);
v___x_150_ = lean_box(0);
v_isShared_151_ = v_isSharedCheck_158_;
goto v_resetjp_149_;
}
v_resetjp_149_:
{
lean_object* v___f_152_; lean_object* v___x_153_; lean_object* v___x_154_; lean_object* v___x_156_; 
v___f_152_ = lean_alloc_closure((void*)(lp_mathlib_WithTop_natCast___redArg___lam__0), 2, 1);
lean_closure_set(v___f_152_, 0, v_toNatCast_146_);
v___x_153_ = lp_mathlib_WithTop_addMonoid___redArg(v_toAddMonoid_147_);
v___x_154_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_154_, 0, v_toOne_148_);
if (v_isShared_151_ == 0)
{
lean_ctor_set(v___x_150_, 2, v___x_154_);
lean_ctor_set(v___x_150_, 1, v___x_153_);
lean_ctor_set(v___x_150_, 0, v___f_152_);
v___x_156_ = v___x_150_;
goto v_reusejp_155_;
}
else
{
lean_object* v_reuseFailAlloc_157_; 
v_reuseFailAlloc_157_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v_reuseFailAlloc_157_, 0, v___f_152_);
lean_ctor_set(v_reuseFailAlloc_157_, 1, v___x_153_);
lean_ctor_set(v_reuseFailAlloc_157_, 2, v___x_154_);
v___x_156_ = v_reuseFailAlloc_157_;
goto v_reusejp_155_;
}
v_reusejp_155_:
{
return v___x_156_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_WithTop_addMonoidWithOne(lean_object* v_00_u03b1_159_, lean_object* v_inst_160_){
_start:
{
lean_object* v___x_161_; 
v___x_161_ = lp_mathlib_WithTop_addMonoidWithOne___redArg(v_inst_160_);
return v___x_161_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_WithTop_addCommMonoidWithOne___redArg(lean_object* v_inst_162_){
_start:
{
lean_object* v___x_163_; 
v___x_163_ = lp_mathlib_WithTop_addMonoidWithOne___redArg(v_inst_162_);
return v___x_163_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_WithTop_addCommMonoidWithOne(lean_object* v_00_u03b1_164_, lean_object* v_inst_165_){
_start:
{
lean_object* v___x_166_; 
v___x_166_ = lp_mathlib_WithTop_addMonoidWithOne___redArg(v_inst_165_);
return v___x_166_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OneHom_withTopMap___redArg___lam__0(lean_object* v_f_167_, lean_object* v___y_168_){
_start:
{
lean_object* v___x_169_; 
v___x_169_ = lean_apply_1(v_f_167_, v___y_168_);
return v___x_169_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OneHom_withTopMap___redArg(lean_object* v_f_170_){
_start:
{
lean_object* v___f_171_; lean_object* v___x_172_; 
v___f_171_ = lean_alloc_closure((void*)(lp_mathlib_OneHom_withTopMap___redArg___lam__0), 2, 1);
lean_closure_set(v___f_171_, 0, v_f_170_);
v___x_172_ = lean_alloc_closure((void*)(lp_mathlib_WithTop_map), 4, 3);
lean_closure_set(v___x_172_, 0, lean_box(0));
lean_closure_set(v___x_172_, 1, lean_box(0));
lean_closure_set(v___x_172_, 2, v___f_171_);
return v___x_172_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OneHom_withTopMap(lean_object* v_M_173_, lean_object* v_N_174_, lean_object* v_inst_175_, lean_object* v_inst_176_, lean_object* v_f_177_){
_start:
{
lean_object* v___x_178_; 
v___x_178_ = lp_mathlib_OneHom_withTopMap___redArg(v_f_177_);
return v___x_178_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OneHom_withTopMap___boxed(lean_object* v_M_179_, lean_object* v_N_180_, lean_object* v_inst_181_, lean_object* v_inst_182_, lean_object* v_f_183_){
_start:
{
lean_object* v_res_184_; 
v_res_184_ = lp_mathlib_OneHom_withTopMap(v_M_179_, v_N_180_, v_inst_181_, v_inst_182_, v_f_183_);
lean_dec(v_inst_182_);
lean_dec(v_inst_181_);
return v_res_184_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ZeroHom_withTopMap___redArg(lean_object* v_f_185_){
_start:
{
lean_object* v___f_186_; lean_object* v___x_187_; 
v___f_186_ = lean_alloc_closure((void*)(lp_mathlib_OneHom_withTopMap___redArg___lam__0), 2, 1);
lean_closure_set(v___f_186_, 0, v_f_185_);
v___x_187_ = lean_alloc_closure((void*)(lp_mathlib_WithTop_map), 4, 3);
lean_closure_set(v___x_187_, 0, lean_box(0));
lean_closure_set(v___x_187_, 1, lean_box(0));
lean_closure_set(v___x_187_, 2, v___f_186_);
return v___x_187_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ZeroHom_withTopMap(lean_object* v_M_188_, lean_object* v_N_189_, lean_object* v_inst_190_, lean_object* v_inst_191_, lean_object* v_f_192_){
_start:
{
lean_object* v___x_193_; 
v___x_193_ = lp_mathlib_ZeroHom_withTopMap___redArg(v_f_192_);
return v___x_193_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ZeroHom_withTopMap___boxed(lean_object* v_M_194_, lean_object* v_N_195_, lean_object* v_inst_196_, lean_object* v_inst_197_, lean_object* v_f_198_){
_start:
{
lean_object* v_res_199_; 
v_res_199_ = lp_mathlib_ZeroHom_withTopMap(v_M_194_, v_N_195_, v_inst_196_, v_inst_197_, v_f_198_);
lean_dec(v_inst_197_);
lean_dec(v_inst_196_);
return v_res_199_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddHom_withTopMap___redArg(lean_object* v_f_200_){
_start:
{
lean_object* v___f_201_; lean_object* v___x_202_; 
v___f_201_ = lean_alloc_closure((void*)(lp_mathlib_OneHom_withTopMap___redArg___lam__0), 2, 1);
lean_closure_set(v___f_201_, 0, v_f_200_);
v___x_202_ = lean_alloc_closure((void*)(lp_mathlib_WithTop_map), 4, 3);
lean_closure_set(v___x_202_, 0, lean_box(0));
lean_closure_set(v___x_202_, 1, lean_box(0));
lean_closure_set(v___x_202_, 2, v___f_201_);
return v___x_202_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddHom_withTopMap(lean_object* v_M_203_, lean_object* v_N_204_, lean_object* v_inst_205_, lean_object* v_inst_206_, lean_object* v_f_207_){
_start:
{
lean_object* v___x_208_; 
v___x_208_ = lp_mathlib_AddHom_withTopMap___redArg(v_f_207_);
return v___x_208_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddHom_withTopMap___boxed(lean_object* v_M_209_, lean_object* v_N_210_, lean_object* v_inst_211_, lean_object* v_inst_212_, lean_object* v_f_213_){
_start:
{
lean_object* v_res_214_; 
v_res_214_ = lp_mathlib_AddHom_withTopMap(v_M_209_, v_N_210_, v_inst_211_, v_inst_212_, v_f_213_);
lean_dec(v_inst_212_);
lean_dec(v_inst_211_);
return v_res_214_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddMonoidHom_withTopMap___redArg(lean_object* v_f_215_){
_start:
{
lean_object* v___f_216_; lean_object* v___x_217_; 
v___f_216_ = lean_alloc_closure((void*)(lp_mathlib_OneHom_withTopMap___redArg___lam__0), 2, 1);
lean_closure_set(v___f_216_, 0, v_f_215_);
v___x_217_ = lean_alloc_closure((void*)(lp_mathlib_WithTop_map), 4, 3);
lean_closure_set(v___x_217_, 0, lean_box(0));
lean_closure_set(v___x_217_, 1, lean_box(0));
lean_closure_set(v___x_217_, 2, v___f_216_);
return v___x_217_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddMonoidHom_withTopMap(lean_object* v_M_218_, lean_object* v_N_219_, lean_object* v_inst_220_, lean_object* v_inst_221_, lean_object* v_f_222_){
_start:
{
lean_object* v___x_223_; 
v___x_223_ = lp_mathlib_AddMonoidHom_withTopMap___redArg(v_f_222_);
return v___x_223_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddMonoidHom_withTopMap___boxed(lean_object* v_M_224_, lean_object* v_N_225_, lean_object* v_inst_226_, lean_object* v_inst_227_, lean_object* v_f_228_){
_start:
{
lean_object* v_res_229_; 
v_res_229_ = lp_mathlib_AddMonoidHom_withTopMap(v_M_224_, v_N_225_, v_inst_226_, v_inst_227_, v_f_228_);
lean_dec_ref(v_inst_227_);
lean_dec_ref(v_inst_226_);
return v_res_229_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_WithBot_one___redArg(lean_object* v_inst_230_){
_start:
{
lean_object* v___x_231_; 
v___x_231_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_231_, 0, v_inst_230_);
return v___x_231_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_WithBot_one(lean_object* v_00_u03b1_232_, lean_object* v_inst_233_){
_start:
{
lean_object* v___x_234_; 
v___x_234_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_234_, 0, v_inst_233_);
return v___x_234_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_WithBot_zero___redArg(lean_object* v_inst_235_){
_start:
{
lean_object* v___x_236_; 
v___x_236_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_236_, 0, v_inst_235_);
return v___x_236_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_WithBot_zero(lean_object* v_00_u03b1_237_, lean_object* v_inst_238_){
_start:
{
lean_object* v___x_239_; 
v___x_239_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_239_, 0, v_inst_238_);
return v___x_239_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_WithBot_add___redArg(lean_object* v_inst_240_){
_start:
{
lean_object* v___f_241_; lean_object* v___x_242_; 
v___f_241_ = lean_alloc_closure((void*)(lp_mathlib_WithTop_add___redArg___lam__0), 3, 1);
lean_closure_set(v___f_241_, 0, v_inst_240_);
v___x_242_ = lean_alloc_closure((void*)(lp_mathlib_WithBot_map_u2082), 6, 4);
lean_closure_set(v___x_242_, 0, lean_box(0));
lean_closure_set(v___x_242_, 1, lean_box(0));
lean_closure_set(v___x_242_, 2, lean_box(0));
lean_closure_set(v___x_242_, 3, v___f_241_);
return v___x_242_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_WithBot_add(lean_object* v_00_u03b1_243_, lean_object* v_inst_244_){
_start:
{
lean_object* v___x_245_; 
v___x_245_ = lp_mathlib_WithBot_add___redArg(v_inst_244_);
return v___x_245_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_WithBot_addSemigroup___redArg(lean_object* v_inst_246_){
_start:
{
lean_object* v___x_247_; 
v___x_247_ = lp_mathlib_WithBot_add___redArg(v_inst_246_);
return v___x_247_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_WithBot_addSemigroup(lean_object* v_00_u03b1_248_, lean_object* v_inst_249_){
_start:
{
lean_object* v___x_250_; 
v___x_250_ = lp_mathlib_WithBot_add___redArg(v_inst_249_);
return v___x_250_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_WithBot_addCommSemigroup___redArg(lean_object* v_inst_251_){
_start:
{
lean_object* v___x_252_; 
v___x_252_ = lp_mathlib_WithBot_add___redArg(v_inst_251_);
return v___x_252_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_WithBot_addCommSemigroup(lean_object* v_00_u03b1_253_, lean_object* v_inst_254_){
_start:
{
lean_object* v___x_255_; 
v___x_255_ = lp_mathlib_WithBot_add___redArg(v_inst_254_);
return v___x_255_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_WithBot_addZeroClass___redArg(lean_object* v_inst_256_){
_start:
{
lean_object* v___x_257_; lean_object* v_toZero_258_; lean_object* v_toAdd_259_; lean_object* v___x_261_; uint8_t v_isShared_262_; uint8_t v_isSharedCheck_268_; 
v___x_257_ = lp_mathlib_AddZeroClass_toAddZero___redArg(v_inst_256_);
v_toZero_258_ = lean_ctor_get(v___x_257_, 0);
v_toAdd_259_ = lean_ctor_get(v___x_257_, 1);
v_isSharedCheck_268_ = !lean_is_exclusive(v___x_257_);
if (v_isSharedCheck_268_ == 0)
{
v___x_261_ = v___x_257_;
v_isShared_262_ = v_isSharedCheck_268_;
goto v_resetjp_260_;
}
else
{
lean_inc(v_toAdd_259_);
lean_inc(v_toZero_258_);
lean_dec(v___x_257_);
v___x_261_ = lean_box(0);
v_isShared_262_ = v_isSharedCheck_268_;
goto v_resetjp_260_;
}
v_resetjp_260_:
{
lean_object* v___x_263_; lean_object* v___x_264_; lean_object* v___x_266_; 
v___x_263_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_263_, 0, v_toZero_258_);
v___x_264_ = lp_mathlib_WithBot_add___redArg(v_toAdd_259_);
if (v_isShared_262_ == 0)
{
lean_ctor_set(v___x_261_, 1, v___x_264_);
lean_ctor_set(v___x_261_, 0, v___x_263_);
v___x_266_ = v___x_261_;
goto v_reusejp_265_;
}
else
{
lean_object* v_reuseFailAlloc_267_; 
v_reuseFailAlloc_267_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_267_, 0, v___x_263_);
lean_ctor_set(v_reuseFailAlloc_267_, 1, v___x_264_);
v___x_266_ = v_reuseFailAlloc_267_;
goto v_reusejp_265_;
}
v_reusejp_265_:
{
return v___x_266_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_WithBot_addZeroClass(lean_object* v_00_u03b1_269_, lean_object* v_inst_270_){
_start:
{
lean_object* v___x_271_; 
v___x_271_ = lp_mathlib_WithBot_addZeroClass___redArg(v_inst_270_);
return v___x_271_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_WithBot_addMonoid___aux__4___redArg(lean_object* v_inst_272_, lean_object* v_n_273_, lean_object* v_a_274_){
_start:
{
lean_object* v_toNSMul_275_; lean_object* v___x_276_; lean_object* v___x_277_; 
v_toNSMul_275_ = lean_ctor_get(v_inst_272_, 2);
lean_inc(v_toNSMul_275_);
v___x_276_ = lp_mathlib_AddMonoid_toAddZeroClass___redArg(v_inst_272_);
lean_dec_ref(v_inst_272_);
v___x_277_ = lp_mathlib_AddZeroClass_toAddZero___redArg(v___x_276_);
if (lean_obj_tag(v_a_274_) == 0)
{
lean_object* v_toZero_278_; lean_object* v_zero_279_; uint8_t v_isZero_280_; 
lean_dec(v_toNSMul_275_);
v_toZero_278_ = lean_ctor_get(v___x_277_, 0);
lean_inc(v_toZero_278_);
lean_dec_ref(v___x_277_);
v_zero_279_ = lean_unsigned_to_nat(0u);
v_isZero_280_ = lean_nat_dec_eq(v_n_273_, v_zero_279_);
lean_dec(v_n_273_);
if (v_isZero_280_ == 1)
{
lean_object* v___x_281_; 
v___x_281_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_281_, 0, v_toZero_278_);
return v___x_281_;
}
else
{
lean_object* v___x_282_; 
lean_dec(v_toZero_278_);
v___x_282_ = lean_box(0);
return v___x_282_;
}
}
else
{
lean_object* v_val_283_; lean_object* v___x_285_; uint8_t v_isShared_286_; uint8_t v_isSharedCheck_291_; 
lean_dec_ref(v___x_277_);
v_val_283_ = lean_ctor_get(v_a_274_, 0);
v_isSharedCheck_291_ = !lean_is_exclusive(v_a_274_);
if (v_isSharedCheck_291_ == 0)
{
v___x_285_ = v_a_274_;
v_isShared_286_ = v_isSharedCheck_291_;
goto v_resetjp_284_;
}
else
{
lean_inc(v_val_283_);
lean_dec(v_a_274_);
v___x_285_ = lean_box(0);
v_isShared_286_ = v_isSharedCheck_291_;
goto v_resetjp_284_;
}
v_resetjp_284_:
{
lean_object* v___x_287_; lean_object* v___x_289_; 
v___x_287_ = lean_apply_2(v_toNSMul_275_, v_n_273_, v_val_283_);
if (v_isShared_286_ == 0)
{
lean_ctor_set(v___x_285_, 0, v___x_287_);
v___x_289_ = v___x_285_;
goto v_reusejp_288_;
}
else
{
lean_object* v_reuseFailAlloc_290_; 
v_reuseFailAlloc_290_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_290_, 0, v___x_287_);
v___x_289_ = v_reuseFailAlloc_290_;
goto v_reusejp_288_;
}
v_reusejp_288_:
{
return v___x_289_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_WithBot_addMonoid___aux__4(lean_object* v_00_u03b1_292_, lean_object* v_inst_293_, lean_object* v_n_294_, lean_object* v_a_295_){
_start:
{
lean_object* v_toNSMul_296_; lean_object* v___x_297_; lean_object* v___x_298_; 
v_toNSMul_296_ = lean_ctor_get(v_inst_293_, 2);
lean_inc(v_toNSMul_296_);
v___x_297_ = lp_mathlib_AddMonoid_toAddZeroClass___redArg(v_inst_293_);
lean_dec_ref(v_inst_293_);
v___x_298_ = lp_mathlib_AddZeroClass_toAddZero___redArg(v___x_297_);
if (lean_obj_tag(v_a_295_) == 0)
{
lean_object* v_toZero_299_; lean_object* v_zero_300_; uint8_t v_isZero_301_; 
lean_dec(v_toNSMul_296_);
v_toZero_299_ = lean_ctor_get(v___x_298_, 0);
lean_inc(v_toZero_299_);
lean_dec_ref(v___x_298_);
v_zero_300_ = lean_unsigned_to_nat(0u);
v_isZero_301_ = lean_nat_dec_eq(v_n_294_, v_zero_300_);
lean_dec(v_n_294_);
if (v_isZero_301_ == 1)
{
lean_object* v___x_302_; 
v___x_302_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_302_, 0, v_toZero_299_);
return v___x_302_;
}
else
{
lean_object* v___x_303_; 
lean_dec(v_toZero_299_);
v___x_303_ = lean_box(0);
return v___x_303_;
}
}
else
{
lean_object* v_val_304_; lean_object* v___x_306_; uint8_t v_isShared_307_; uint8_t v_isSharedCheck_312_; 
lean_dec_ref(v___x_298_);
v_val_304_ = lean_ctor_get(v_a_295_, 0);
v_isSharedCheck_312_ = !lean_is_exclusive(v_a_295_);
if (v_isSharedCheck_312_ == 0)
{
v___x_306_ = v_a_295_;
v_isShared_307_ = v_isSharedCheck_312_;
goto v_resetjp_305_;
}
else
{
lean_inc(v_val_304_);
lean_dec(v_a_295_);
v___x_306_ = lean_box(0);
v_isShared_307_ = v_isSharedCheck_312_;
goto v_resetjp_305_;
}
v_resetjp_305_:
{
lean_object* v___x_308_; lean_object* v___x_310_; 
v___x_308_ = lean_apply_2(v_toNSMul_296_, v_n_294_, v_val_304_);
if (v_isShared_307_ == 0)
{
lean_ctor_set(v___x_306_, 0, v___x_308_);
v___x_310_ = v___x_306_;
goto v_reusejp_309_;
}
else
{
lean_object* v_reuseFailAlloc_311_; 
v_reuseFailAlloc_311_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_311_, 0, v___x_308_);
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
}
LEAN_EXPORT lean_object* lp_mathlib_WithBot_addMonoid___redArg(lean_object* v_inst_313_){
_start:
{
lean_object* v___x_314_; lean_object* v___x_315_; lean_object* v_toZero_316_; lean_object* v_toAdd_317_; lean_object* v___x_318_; lean_object* v___x_319_; lean_object* v___x_320_; lean_object* v___x_321_; 
v___x_314_ = lp_mathlib_AddMonoid_toAddZeroClass___redArg(v_inst_313_);
v___x_315_ = lp_mathlib_AddZeroClass_toAddZero___redArg(v___x_314_);
v_toZero_316_ = lean_ctor_get(v___x_315_, 0);
lean_inc(v_toZero_316_);
lean_dec_ref(v___x_315_);
v_toAdd_317_ = lean_ctor_get(v_inst_313_, 1);
v___x_318_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_318_, 0, v_toZero_316_);
lean_inc(v_toAdd_317_);
v___x_319_ = lp_mathlib_WithBot_add___redArg(v_toAdd_317_);
v___x_320_ = lean_alloc_closure((void*)(lp_mathlib_WithBot_addMonoid___aux__4), 4, 2);
lean_closure_set(v___x_320_, 0, lean_box(0));
lean_closure_set(v___x_320_, 1, v_inst_313_);
v___x_321_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_321_, 0, v___x_318_);
lean_ctor_set(v___x_321_, 1, v___x_319_);
lean_ctor_set(v___x_321_, 2, v___x_320_);
return v___x_321_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_WithBot_addMonoid(lean_object* v_00_u03b1_322_, lean_object* v_inst_323_){
_start:
{
lean_object* v___x_324_; 
v___x_324_ = lp_mathlib_WithBot_addMonoid___redArg(v_inst_323_);
return v___x_324_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_WithBot_addHom(lean_object* v_00_u03b1_325_, lean_object* v_inst_326_){
_start:
{
lean_object* v___x_327_; 
v___x_327_ = ((lean_object*)(lp_mathlib_WithTop_addHom___closed__0));
return v___x_327_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_WithBot_addHom___boxed(lean_object* v_00_u03b1_328_, lean_object* v_inst_329_){
_start:
{
lean_object* v_res_330_; 
v_res_330_ = lp_mathlib_WithBot_addHom(v_00_u03b1_328_, v_inst_329_);
lean_dec_ref(v_inst_329_);
return v_res_330_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_WithBot_addCommMonoid___redArg(lean_object* v_inst_331_){
_start:
{
lean_object* v___x_332_; 
v___x_332_ = lp_mathlib_WithBot_addMonoid___redArg(v_inst_331_);
return v___x_332_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_WithBot_addCommMonoid(lean_object* v_00_u03b1_333_, lean_object* v_inst_334_){
_start:
{
lean_object* v___x_335_; 
v___x_335_ = lp_mathlib_WithBot_addMonoid___redArg(v_inst_334_);
return v___x_335_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_WithBot_instNatCast___redArg(lean_object* v_inst_336_){
_start:
{
lean_object* v___f_337_; 
v___f_337_ = lean_alloc_closure((void*)(lp_mathlib_WithTop_natCast___redArg___lam__0), 2, 1);
lean_closure_set(v___f_337_, 0, v_inst_336_);
return v___f_337_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_WithBot_instNatCast(lean_object* v_00_u03b1_338_, lean_object* v_inst_339_){
_start:
{
lean_object* v___f_340_; 
v___f_340_ = lean_alloc_closure((void*)(lp_mathlib_WithTop_natCast___redArg___lam__0), 2, 1);
lean_closure_set(v___f_340_, 0, v_inst_339_);
return v___f_340_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_WithBot_addMonoidWithOne___redArg(lean_object* v_inst_341_){
_start:
{
lean_object* v_toNatCast_342_; lean_object* v_toAddMonoid_343_; lean_object* v_toOne_344_; lean_object* v___x_346_; uint8_t v_isShared_347_; uint8_t v_isSharedCheck_354_; 
v_toNatCast_342_ = lean_ctor_get(v_inst_341_, 0);
v_toAddMonoid_343_ = lean_ctor_get(v_inst_341_, 1);
v_toOne_344_ = lean_ctor_get(v_inst_341_, 2);
v_isSharedCheck_354_ = !lean_is_exclusive(v_inst_341_);
if (v_isSharedCheck_354_ == 0)
{
v___x_346_ = v_inst_341_;
v_isShared_347_ = v_isSharedCheck_354_;
goto v_resetjp_345_;
}
else
{
lean_inc(v_toOne_344_);
lean_inc(v_toAddMonoid_343_);
lean_inc(v_toNatCast_342_);
lean_dec(v_inst_341_);
v___x_346_ = lean_box(0);
v_isShared_347_ = v_isSharedCheck_354_;
goto v_resetjp_345_;
}
v_resetjp_345_:
{
lean_object* v___f_348_; lean_object* v___x_349_; lean_object* v___x_350_; lean_object* v___x_352_; 
v___f_348_ = lean_alloc_closure((void*)(lp_mathlib_WithTop_natCast___redArg___lam__0), 2, 1);
lean_closure_set(v___f_348_, 0, v_toNatCast_342_);
v___x_349_ = lp_mathlib_WithBot_addMonoid___redArg(v_toAddMonoid_343_);
v___x_350_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_350_, 0, v_toOne_344_);
if (v_isShared_347_ == 0)
{
lean_ctor_set(v___x_346_, 2, v___x_350_);
lean_ctor_set(v___x_346_, 1, v___x_349_);
lean_ctor_set(v___x_346_, 0, v___f_348_);
v___x_352_ = v___x_346_;
goto v_reusejp_351_;
}
else
{
lean_object* v_reuseFailAlloc_353_; 
v_reuseFailAlloc_353_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v_reuseFailAlloc_353_, 0, v___f_348_);
lean_ctor_set(v_reuseFailAlloc_353_, 1, v___x_349_);
lean_ctor_set(v_reuseFailAlloc_353_, 2, v___x_350_);
v___x_352_ = v_reuseFailAlloc_353_;
goto v_reusejp_351_;
}
v_reusejp_351_:
{
return v___x_352_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_WithBot_addMonoidWithOne(lean_object* v_00_u03b1_355_, lean_object* v_inst_356_){
_start:
{
lean_object* v___x_357_; 
v___x_357_ = lp_mathlib_WithBot_addMonoidWithOne___redArg(v_inst_356_);
return v___x_357_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_WithBot_addCommMonoidWithOne___redArg(lean_object* v_inst_358_){
_start:
{
lean_object* v___x_359_; 
v___x_359_ = lp_mathlib_WithBot_addMonoidWithOne___redArg(v_inst_358_);
return v___x_359_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_WithBot_addCommMonoidWithOne(lean_object* v_00_u03b1_360_, lean_object* v_inst_361_){
_start:
{
lean_object* v___x_362_; 
v___x_362_ = lp_mathlib_WithBot_addMonoidWithOne___redArg(v_inst_361_);
return v___x_362_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OneHom_withBotMap___redArg(lean_object* v_f_363_){
_start:
{
lean_object* v___f_364_; lean_object* v___x_365_; 
v___f_364_ = lean_alloc_closure((void*)(lp_mathlib_OneHom_withTopMap___redArg___lam__0), 2, 1);
lean_closure_set(v___f_364_, 0, v_f_363_);
v___x_365_ = lean_alloc_closure((void*)(lp_mathlib_WithBot_map), 4, 3);
lean_closure_set(v___x_365_, 0, lean_box(0));
lean_closure_set(v___x_365_, 1, lean_box(0));
lean_closure_set(v___x_365_, 2, v___f_364_);
return v___x_365_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OneHom_withBotMap(lean_object* v_M_366_, lean_object* v_N_367_, lean_object* v_inst_368_, lean_object* v_inst_369_, lean_object* v_f_370_){
_start:
{
lean_object* v___x_371_; 
v___x_371_ = lp_mathlib_OneHom_withBotMap___redArg(v_f_370_);
return v___x_371_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OneHom_withBotMap___boxed(lean_object* v_M_372_, lean_object* v_N_373_, lean_object* v_inst_374_, lean_object* v_inst_375_, lean_object* v_f_376_){
_start:
{
lean_object* v_res_377_; 
v_res_377_ = lp_mathlib_OneHom_withBotMap(v_M_372_, v_N_373_, v_inst_374_, v_inst_375_, v_f_376_);
lean_dec(v_inst_375_);
lean_dec(v_inst_374_);
return v_res_377_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ZeroHom_withBotMap___redArg(lean_object* v_f_378_){
_start:
{
lean_object* v___f_379_; lean_object* v___x_380_; 
v___f_379_ = lean_alloc_closure((void*)(lp_mathlib_OneHom_withTopMap___redArg___lam__0), 2, 1);
lean_closure_set(v___f_379_, 0, v_f_378_);
v___x_380_ = lean_alloc_closure((void*)(lp_mathlib_WithBot_map), 4, 3);
lean_closure_set(v___x_380_, 0, lean_box(0));
lean_closure_set(v___x_380_, 1, lean_box(0));
lean_closure_set(v___x_380_, 2, v___f_379_);
return v___x_380_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ZeroHom_withBotMap(lean_object* v_M_381_, lean_object* v_N_382_, lean_object* v_inst_383_, lean_object* v_inst_384_, lean_object* v_f_385_){
_start:
{
lean_object* v___x_386_; 
v___x_386_ = lp_mathlib_ZeroHom_withBotMap___redArg(v_f_385_);
return v___x_386_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ZeroHom_withBotMap___boxed(lean_object* v_M_387_, lean_object* v_N_388_, lean_object* v_inst_389_, lean_object* v_inst_390_, lean_object* v_f_391_){
_start:
{
lean_object* v_res_392_; 
v_res_392_ = lp_mathlib_ZeroHom_withBotMap(v_M_387_, v_N_388_, v_inst_389_, v_inst_390_, v_f_391_);
lean_dec(v_inst_390_);
lean_dec(v_inst_389_);
return v_res_392_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddHom_withBotMap___redArg(lean_object* v_f_393_){
_start:
{
lean_object* v___f_394_; lean_object* v___x_395_; 
v___f_394_ = lean_alloc_closure((void*)(lp_mathlib_OneHom_withTopMap___redArg___lam__0), 2, 1);
lean_closure_set(v___f_394_, 0, v_f_393_);
v___x_395_ = lean_alloc_closure((void*)(lp_mathlib_WithBot_map), 4, 3);
lean_closure_set(v___x_395_, 0, lean_box(0));
lean_closure_set(v___x_395_, 1, lean_box(0));
lean_closure_set(v___x_395_, 2, v___f_394_);
return v___x_395_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddHom_withBotMap(lean_object* v_M_396_, lean_object* v_N_397_, lean_object* v_inst_398_, lean_object* v_inst_399_, lean_object* v_f_400_){
_start:
{
lean_object* v___x_401_; 
v___x_401_ = lp_mathlib_AddHom_withBotMap___redArg(v_f_400_);
return v___x_401_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddHom_withBotMap___boxed(lean_object* v_M_402_, lean_object* v_N_403_, lean_object* v_inst_404_, lean_object* v_inst_405_, lean_object* v_f_406_){
_start:
{
lean_object* v_res_407_; 
v_res_407_ = lp_mathlib_AddHom_withBotMap(v_M_402_, v_N_403_, v_inst_404_, v_inst_405_, v_f_406_);
lean_dec(v_inst_405_);
lean_dec(v_inst_404_);
return v_res_407_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddMonoidHom_withBotMap___redArg(lean_object* v_f_408_){
_start:
{
lean_object* v___f_409_; lean_object* v___x_410_; 
v___f_409_ = lean_alloc_closure((void*)(lp_mathlib_OneHom_withTopMap___redArg___lam__0), 2, 1);
lean_closure_set(v___f_409_, 0, v_f_408_);
v___x_410_ = lean_alloc_closure((void*)(lp_mathlib_WithBot_map), 4, 3);
lean_closure_set(v___x_410_, 0, lean_box(0));
lean_closure_set(v___x_410_, 1, lean_box(0));
lean_closure_set(v___x_410_, 2, v___f_409_);
return v___x_410_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddMonoidHom_withBotMap(lean_object* v_M_411_, lean_object* v_N_412_, lean_object* v_inst_413_, lean_object* v_inst_414_, lean_object* v_f_415_){
_start:
{
lean_object* v___x_416_; 
v___x_416_ = lp_mathlib_AddMonoidHom_withBotMap___redArg(v_f_415_);
return v___x_416_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddMonoidHom_withBotMap___boxed(lean_object* v_M_417_, lean_object* v_N_418_, lean_object* v_inst_419_, lean_object* v_inst_420_, lean_object* v_f_421_){
_start:
{
lean_object* v_res_422_; 
v_res_422_ = lp_mathlib_AddMonoidHom_withBotMap(v_M_417_, v_N_418_, v_inst_419_, v_inst_420_, v_f_421_);
lean_dec_ref(v_inst_420_);
lean_dec_ref(v_inst_419_);
return v_res_422_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddEquiv_withBotCongr___redArg(lean_object* v_e_423_){
_start:
{
lean_object* v___x_424_; 
v___x_424_ = lp_mathlib_Equiv_withBotCongr___redArg(v_e_423_);
return v___x_424_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddEquiv_withBotCongr(lean_object* v_00_u03b1_425_, lean_object* v_00_u03b2_426_, lean_object* v_inst_427_, lean_object* v_inst_428_, lean_object* v_e_429_){
_start:
{
lean_object* v___x_430_; 
v___x_430_ = lp_mathlib_Equiv_withBotCongr___redArg(v_e_429_);
return v___x_430_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddEquiv_withBotCongr___boxed(lean_object* v_00_u03b1_431_, lean_object* v_00_u03b2_432_, lean_object* v_inst_433_, lean_object* v_inst_434_, lean_object* v_e_435_){
_start:
{
lean_object* v_res_436_; 
v_res_436_ = lp_mathlib_AddEquiv_withBotCongr(v_00_u03b1_431_, v_00_u03b2_432_, v_inst_433_, v_inst_434_, v_e_435_);
lean_dec(v_inst_434_);
lean_dec(v_inst_433_);
return v_res_436_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddEquiv_withTopCongr___redArg(lean_object* v_e_437_){
_start:
{
lean_object* v___x_438_; 
v___x_438_ = lp_mathlib_Equiv_withTopCongr___redArg(v_e_437_);
return v___x_438_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddEquiv_withTopCongr(lean_object* v_00_u03b1_439_, lean_object* v_00_u03b2_440_, lean_object* v_inst_441_, lean_object* v_inst_442_, lean_object* v_e_443_){
_start:
{
lean_object* v___x_444_; 
v___x_444_ = lp_mathlib_Equiv_withTopCongr___redArg(v_e_443_);
return v___x_444_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddEquiv_withTopCongr___boxed(lean_object* v_00_u03b1_445_, lean_object* v_00_u03b2_446_, lean_object* v_inst_447_, lean_object* v_inst_448_, lean_object* v_e_449_){
_start:
{
lean_object* v_res_450_; 
v_res_450_ = lp_mathlib_AddEquiv_withTopCongr(v_00_u03b1_445_, v_00_u03b2_446_, v_inst_447_, v_inst_448_, v_e_449_);
lean_dec(v_inst_448_);
lean_dec(v_inst_447_);
return v_res_450_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_CharZero_Defs(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Group_Hom_Defs(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Group_Equiv_Defs(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Order_Monoid_Unbundled_ExistsOfLE(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Order_ZeroLEOne(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Order_WithBot(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Order_Monoid_Unbundled_WithTop(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_CharZero_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Group_Hom_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Group_Equiv_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Order_Monoid_Unbundled_ExistsOfLE(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Order_ZeroLEOne(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Order_WithBot(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_Algebra_Order_Monoid_Unbundled_WithTop(uint8_t builtin) {
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
lean_object* initialize_mathlib_Mathlib_Algebra_CharZero_Defs(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Group_Hom_Defs(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Group_Equiv_Defs(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Order_Monoid_Unbundled_ExistsOfLE(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Order_ZeroLEOne(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Order_WithBot(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Algebra_Order_Monoid_Unbundled_WithTop(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_CharZero_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Group_Hom_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Group_Equiv_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Order_Monoid_Unbundled_ExistsOfLE(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Order_ZeroLEOne(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Order_WithBot(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Order_Monoid_Unbundled_WithTop(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_Algebra_Order_Monoid_Unbundled_WithTop(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_Algebra_Order_Monoid_Unbundled_WithTop(builtin);
}
#ifdef __cplusplus
}
#endif
