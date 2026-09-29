// Lean compiler output
// Module: Mathlib.Order.BooleanAlgebra.Basic
// Imports: public import Init public meta import Init public import Mathlib.Order.BooleanAlgebra.Defs public import Mathlib.Tactic.GRewrite
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
lean_object* lp_mathlib_OrderDual_instLattice___redArg(lean_object*);
lean_object* lp_mathlib_BiheytingAlgebra_toCoheytingAlgebra___redArg(lean_object*);
lean_object* lp_mathlib_OrderDual_instHeytingAlgebra___redArg(lean_object*);
lean_object* lp_mathlib_OrderDual_instGeneralizedCoheytingAlgebra___redArg(lean_object*);
lean_object* lp_mathlib_Prod_instLattice___redArg(lean_object*, lean_object*);
lean_object* lp_mathlib_Prod_instHeytingAlgebra___redArg(lean_object*, lean_object*);
lean_object* lp_mathlib_Prod_instSDiff___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Equiv_symm___redArg(lean_object*);
lean_object* lp_mathlib_Function_Injective_semilatticeInf___redArg(lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Lattice_toSemilatticeInf___redArg(lean_object*);
lean_object* lp_mathlib_Pi_instDistribLattice___redArg(lean_object*);
lean_object* lp_mathlib_Pi_instSDiff___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Pi_instBotForall___redArg___lam__0(lean_object*, lean_object*);
lean_object* lp_mathlib_Pi_instHeytingAlgebra___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_GeneralizedBooleanAlgebra_toOrderBot___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_GeneralizedBooleanAlgebra_toOrderBot___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_GeneralizedBooleanAlgebra_toOrderBot(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_GeneralizedBooleanAlgebra_toOrderBot___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_GeneralizedBooleanAlgebra_toGeneralizedCoheytingAlgebra___redArg___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_GeneralizedBooleanAlgebra_toGeneralizedCoheytingAlgebra___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_GeneralizedBooleanAlgebra_toGeneralizedCoheytingAlgebra(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Prod_instGeneralizedBooleanAlgebra___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Prod_instGeneralizedBooleanAlgebra(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Pi_instGeneralizedBooleanAlgebra___redArg___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Pi_instGeneralizedBooleanAlgebra___redArg___lam__1(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Pi_instGeneralizedBooleanAlgebra___redArg___lam__2(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Pi_instGeneralizedBooleanAlgebra___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Pi_instGeneralizedBooleanAlgebra(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_GeneralizedBooleanAlgebra_toBooleanAlgebra___redArg___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_GeneralizedBooleanAlgebra_toBooleanAlgebra___redArg___lam__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_GeneralizedBooleanAlgebra_toBooleanAlgebra___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_GeneralizedBooleanAlgebra_toBooleanAlgebra(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_BooleanAlgebra_toGeneralizedBooleanAlgebra___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_BooleanAlgebra_toGeneralizedBooleanAlgebra___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_BooleanAlgebra_toGeneralizedBooleanAlgebra(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_BooleanAlgebra_toGeneralizedBooleanAlgebra___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_BooleanAlgebra_toBiheytingAlgebra___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_BooleanAlgebra_toBiheytingAlgebra___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_BooleanAlgebra_toBiheytingAlgebra(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_BooleanAlgebra_toBiheytingAlgebra___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderDual_instBooleanAlgebra___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderDual_instBooleanAlgebra(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Prod_instBooleanAlgebra___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Prod_instBooleanAlgebra(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Pi_instBooleanAlgebra___redArg___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Pi_instBooleanAlgebra___redArg___lam__1(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Pi_instBooleanAlgebra___redArg___lam__2(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Pi_instBooleanAlgebra___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Pi_instBooleanAlgebra(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_generalizedBooleanAlgebra___redArg___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_generalizedBooleanAlgebra___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_generalizedBooleanAlgebra(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_generalizedBooleanAlgebra___boxed(lean_object**);
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_booleanAlgebra___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_booleanAlgebra(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_booleanAlgebra___boxed(lean_object**);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_generalizedBooleanAlgebra___redArg___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_generalizedBooleanAlgebra___redArg___lam__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_generalizedBooleanAlgebra___redArg___lam__3(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_generalizedBooleanAlgebra___redArg___lam__2(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_generalizedBooleanAlgebra___redArg___lam__5(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_generalizedBooleanAlgebra___redArg___lam__7(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_Equiv_generalizedBooleanAlgebra___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Equiv_generalizedBooleanAlgebra___redArg___lam__0, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Equiv_generalizedBooleanAlgebra___redArg___closed__0 = (const lean_object*)&lp_mathlib_Equiv_generalizedBooleanAlgebra___redArg___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_Equiv_generalizedBooleanAlgebra___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_generalizedBooleanAlgebra(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_booleanAlgebra___redArg___lam__12(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_booleanAlgebra___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_booleanAlgebra___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_booleanAlgebra(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_GeneralizedBooleanAlgebra_toOrderBot___redArg(lean_object* v_inst_1_){
_start:
{
lean_object* v_toBot_2_; 
v_toBot_2_ = lean_ctor_get(v_inst_1_, 2);
lean_inc(v_toBot_2_);
return v_toBot_2_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_GeneralizedBooleanAlgebra_toOrderBot___redArg___boxed(lean_object* v_inst_3_){
_start:
{
lean_object* v_res_4_; 
v_res_4_ = lp_mathlib_GeneralizedBooleanAlgebra_toOrderBot___redArg(v_inst_3_);
lean_dec_ref(v_inst_3_);
return v_res_4_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_GeneralizedBooleanAlgebra_toOrderBot(lean_object* v_00_u03b1_5_, lean_object* v_inst_6_){
_start:
{
lean_object* v_toBot_7_; 
v_toBot_7_ = lean_ctor_get(v_inst_6_, 2);
lean_inc(v_toBot_7_);
return v_toBot_7_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_GeneralizedBooleanAlgebra_toOrderBot___boxed(lean_object* v_00_u03b1_8_, lean_object* v_inst_9_){
_start:
{
lean_object* v_res_10_; 
v_res_10_ = lp_mathlib_GeneralizedBooleanAlgebra_toOrderBot(v_00_u03b1_8_, v_inst_9_);
lean_dec_ref(v_inst_9_);
return v_res_10_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_GeneralizedBooleanAlgebra_toGeneralizedCoheytingAlgebra___redArg___lam__0(lean_object* v_toSDiff_11_, lean_object* v_x1_12_, lean_object* v_x2_13_){
_start:
{
lean_object* v___x_14_; 
v___x_14_ = lean_apply_2(v_toSDiff_11_, v_x1_12_, v_x2_13_);
return v___x_14_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_GeneralizedBooleanAlgebra_toGeneralizedCoheytingAlgebra___redArg(lean_object* v_inst_15_){
_start:
{
lean_object* v_toDistribLattice_16_; lean_object* v_toSDiff_17_; lean_object* v_toBot_18_; lean_object* v___x_20_; uint8_t v_isShared_21_; uint8_t v_isSharedCheck_26_; 
v_toDistribLattice_16_ = lean_ctor_get(v_inst_15_, 0);
v_toSDiff_17_ = lean_ctor_get(v_inst_15_, 1);
v_toBot_18_ = lean_ctor_get(v_inst_15_, 2);
v_isSharedCheck_26_ = !lean_is_exclusive(v_inst_15_);
if (v_isSharedCheck_26_ == 0)
{
v___x_20_ = v_inst_15_;
v_isShared_21_ = v_isSharedCheck_26_;
goto v_resetjp_19_;
}
else
{
lean_inc(v_toBot_18_);
lean_inc(v_toSDiff_17_);
lean_inc(v_toDistribLattice_16_);
lean_dec(v_inst_15_);
v___x_20_ = lean_box(0);
v_isShared_21_ = v_isSharedCheck_26_;
goto v_resetjp_19_;
}
v_resetjp_19_:
{
lean_object* v___f_22_; lean_object* v___x_24_; 
v___f_22_ = lean_alloc_closure((void*)(lp_mathlib_GeneralizedBooleanAlgebra_toGeneralizedCoheytingAlgebra___redArg___lam__0), 3, 1);
lean_closure_set(v___f_22_, 0, v_toSDiff_17_);
if (v_isShared_21_ == 0)
{
lean_ctor_set(v___x_20_, 2, v___f_22_);
lean_ctor_set(v___x_20_, 1, v_toBot_18_);
v___x_24_ = v___x_20_;
goto v_reusejp_23_;
}
else
{
lean_object* v_reuseFailAlloc_25_; 
v_reuseFailAlloc_25_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v_reuseFailAlloc_25_, 0, v_toDistribLattice_16_);
lean_ctor_set(v_reuseFailAlloc_25_, 1, v_toBot_18_);
lean_ctor_set(v_reuseFailAlloc_25_, 2, v___f_22_);
v___x_24_ = v_reuseFailAlloc_25_;
goto v_reusejp_23_;
}
v_reusejp_23_:
{
return v___x_24_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_GeneralizedBooleanAlgebra_toGeneralizedCoheytingAlgebra(lean_object* v_00_u03b1_27_, lean_object* v_inst_28_){
_start:
{
lean_object* v___x_29_; 
v___x_29_ = lp_mathlib_GeneralizedBooleanAlgebra_toGeneralizedCoheytingAlgebra___redArg(v_inst_28_);
return v___x_29_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Prod_instGeneralizedBooleanAlgebra___redArg(lean_object* v_inst_30_, lean_object* v_inst_31_){
_start:
{
lean_object* v_toDistribLattice_32_; lean_object* v_toSDiff_33_; lean_object* v_toBot_34_; lean_object* v_toDistribLattice_35_; lean_object* v_toSDiff_36_; lean_object* v_toBot_37_; lean_object* v___x_39_; uint8_t v_isShared_40_; uint8_t v_isSharedCheck_47_; 
v_toDistribLattice_32_ = lean_ctor_get(v_inst_30_, 0);
lean_inc_ref(v_toDistribLattice_32_);
v_toSDiff_33_ = lean_ctor_get(v_inst_30_, 1);
lean_inc(v_toSDiff_33_);
v_toBot_34_ = lean_ctor_get(v_inst_30_, 2);
lean_inc(v_toBot_34_);
lean_dec_ref(v_inst_30_);
v_toDistribLattice_35_ = lean_ctor_get(v_inst_31_, 0);
v_toSDiff_36_ = lean_ctor_get(v_inst_31_, 1);
v_toBot_37_ = lean_ctor_get(v_inst_31_, 2);
v_isSharedCheck_47_ = !lean_is_exclusive(v_inst_31_);
if (v_isSharedCheck_47_ == 0)
{
v___x_39_ = v_inst_31_;
v_isShared_40_ = v_isSharedCheck_47_;
goto v_resetjp_38_;
}
else
{
lean_inc(v_toBot_37_);
lean_inc(v_toSDiff_36_);
lean_inc(v_toDistribLattice_35_);
lean_dec(v_inst_31_);
v___x_39_ = lean_box(0);
v_isShared_40_ = v_isSharedCheck_47_;
goto v_resetjp_38_;
}
v_resetjp_38_:
{
lean_object* v___x_41_; lean_object* v___f_42_; lean_object* v___x_43_; lean_object* v___x_45_; 
v___x_41_ = lp_mathlib_Prod_instLattice___redArg(v_toDistribLattice_32_, v_toDistribLattice_35_);
v___f_42_ = lean_alloc_closure((void*)(lp_mathlib_Prod_instSDiff___redArg___lam__0), 4, 2);
lean_closure_set(v___f_42_, 0, v_toSDiff_33_);
lean_closure_set(v___f_42_, 1, v_toSDiff_36_);
v___x_43_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_43_, 0, v_toBot_34_);
lean_ctor_set(v___x_43_, 1, v_toBot_37_);
if (v_isShared_40_ == 0)
{
lean_ctor_set(v___x_39_, 2, v___x_43_);
lean_ctor_set(v___x_39_, 1, v___f_42_);
lean_ctor_set(v___x_39_, 0, v___x_41_);
v___x_45_ = v___x_39_;
goto v_reusejp_44_;
}
else
{
lean_object* v_reuseFailAlloc_46_; 
v_reuseFailAlloc_46_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v_reuseFailAlloc_46_, 0, v___x_41_);
lean_ctor_set(v_reuseFailAlloc_46_, 1, v___f_42_);
lean_ctor_set(v_reuseFailAlloc_46_, 2, v___x_43_);
v___x_45_ = v_reuseFailAlloc_46_;
goto v_reusejp_44_;
}
v_reusejp_44_:
{
return v___x_45_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Prod_instGeneralizedBooleanAlgebra(lean_object* v_00_u03b1_48_, lean_object* v_00_u03b2_49_, lean_object* v_inst_50_, lean_object* v_inst_51_){
_start:
{
lean_object* v___x_52_; 
v___x_52_ = lp_mathlib_Prod_instGeneralizedBooleanAlgebra___redArg(v_inst_50_, v_inst_51_);
return v___x_52_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Pi_instGeneralizedBooleanAlgebra___redArg___lam__0(lean_object* v_inst_53_, lean_object* v_i_54_){
_start:
{
lean_object* v___x_55_; lean_object* v_toDistribLattice_56_; 
v___x_55_ = lean_apply_1(v_inst_53_, v_i_54_);
v_toDistribLattice_56_ = lean_ctor_get(v___x_55_, 0);
lean_inc_ref(v_toDistribLattice_56_);
lean_dec_ref(v___x_55_);
return v_toDistribLattice_56_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Pi_instGeneralizedBooleanAlgebra___redArg___lam__1(lean_object* v_inst_57_, lean_object* v_i_58_, lean_object* v___y_59_, lean_object* v___y_60_){
_start:
{
lean_object* v___x_61_; lean_object* v_toSDiff_62_; lean_object* v___x_63_; 
v___x_61_ = lean_apply_1(v_inst_57_, v_i_58_);
v_toSDiff_62_ = lean_ctor_get(v___x_61_, 1);
lean_inc(v_toSDiff_62_);
lean_dec_ref(v___x_61_);
v___x_63_ = lean_apply_2(v_toSDiff_62_, v___y_59_, v___y_60_);
return v___x_63_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Pi_instGeneralizedBooleanAlgebra___redArg___lam__2(lean_object* v_inst_64_, lean_object* v_i_65_){
_start:
{
lean_object* v___x_66_; lean_object* v_toBot_67_; 
v___x_66_ = lean_apply_1(v_inst_64_, v_i_65_);
v_toBot_67_ = lean_ctor_get(v___x_66_, 2);
lean_inc(v_toBot_67_);
lean_dec_ref(v___x_66_);
return v_toBot_67_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Pi_instGeneralizedBooleanAlgebra___redArg(lean_object* v_inst_68_){
_start:
{
lean_object* v___f_69_; lean_object* v___f_70_; lean_object* v___f_71_; lean_object* v___x_72_; lean_object* v___f_73_; lean_object* v___f_74_; lean_object* v___x_75_; 
lean_inc_ref_n(v_inst_68_, 2);
v___f_69_ = lean_alloc_closure((void*)(lp_mathlib_Pi_instGeneralizedBooleanAlgebra___redArg___lam__0), 2, 1);
lean_closure_set(v___f_69_, 0, v_inst_68_);
v___f_70_ = lean_alloc_closure((void*)(lp_mathlib_Pi_instGeneralizedBooleanAlgebra___redArg___lam__1), 4, 1);
lean_closure_set(v___f_70_, 0, v_inst_68_);
v___f_71_ = lean_alloc_closure((void*)(lp_mathlib_Pi_instGeneralizedBooleanAlgebra___redArg___lam__2), 2, 1);
lean_closure_set(v___f_71_, 0, v_inst_68_);
v___x_72_ = lp_mathlib_Pi_instDistribLattice___redArg(v___f_69_);
v___f_73_ = lean_alloc_closure((void*)(lp_mathlib_Pi_instSDiff___redArg___lam__0), 4, 1);
lean_closure_set(v___f_73_, 0, v___f_70_);
v___f_74_ = lean_alloc_closure((void*)(lp_mathlib_Pi_instBotForall___redArg___lam__0), 2, 1);
lean_closure_set(v___f_74_, 0, v___f_71_);
v___x_75_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_75_, 0, v___x_72_);
lean_ctor_set(v___x_75_, 1, v___f_73_);
lean_ctor_set(v___x_75_, 2, v___f_74_);
return v___x_75_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Pi_instGeneralizedBooleanAlgebra(lean_object* v_00_u03b9_76_, lean_object* v_00_u03b1_77_, lean_object* v_inst_78_){
_start:
{
lean_object* v___x_79_; 
v___x_79_ = lp_mathlib_Pi_instGeneralizedBooleanAlgebra___redArg(v_inst_78_);
return v___x_79_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_GeneralizedBooleanAlgebra_toBooleanAlgebra___redArg___lam__0(lean_object* v_toSDiff_80_, lean_object* v_inst_81_, lean_object* v_a_82_){
_start:
{
lean_object* v___x_83_; 
v___x_83_ = lean_apply_2(v_toSDiff_80_, v_inst_81_, v_a_82_);
return v___x_83_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_GeneralizedBooleanAlgebra_toBooleanAlgebra___redArg___lam__1(lean_object* v_toSemilatticeSup_84_, lean_object* v_toSDiff_85_, lean_object* v_inst_86_, lean_object* v_x_87_, lean_object* v_y_88_){
_start:
{
lean_object* v_sup_89_; lean_object* v___x_90_; lean_object* v___x_91_; 
v_sup_89_ = lean_ctor_get(v_toSemilatticeSup_84_, 1);
lean_inc(v_sup_89_);
lean_dec_ref(v_toSemilatticeSup_84_);
v___x_90_ = lean_apply_2(v_toSDiff_85_, v_inst_86_, v_x_87_);
v___x_91_ = lean_apply_2(v_sup_89_, v_y_88_, v___x_90_);
return v___x_91_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_GeneralizedBooleanAlgebra_toBooleanAlgebra___redArg(lean_object* v_inst_92_, lean_object* v_inst_93_){
_start:
{
lean_object* v_toDistribLattice_94_; lean_object* v_toSDiff_95_; lean_object* v_toBot_96_; lean_object* v_toSemilatticeSup_97_; lean_object* v___f_98_; lean_object* v___f_99_; lean_object* v___x_100_; 
v_toDistribLattice_94_ = lean_ctor_get(v_inst_92_, 0);
lean_inc_ref(v_toDistribLattice_94_);
v_toSDiff_95_ = lean_ctor_get(v_inst_92_, 1);
lean_inc_n(v_toSDiff_95_, 3);
v_toBot_96_ = lean_ctor_get(v_inst_92_, 2);
lean_inc(v_toBot_96_);
lean_dec_ref(v_inst_92_);
v_toSemilatticeSup_97_ = lean_ctor_get(v_toDistribLattice_94_, 0);
lean_inc_n(v_inst_93_, 2);
v___f_98_ = lean_alloc_closure((void*)(lp_mathlib_GeneralizedBooleanAlgebra_toBooleanAlgebra___redArg___lam__0), 3, 2);
lean_closure_set(v___f_98_, 0, v_toSDiff_95_);
lean_closure_set(v___f_98_, 1, v_inst_93_);
lean_inc_ref(v_toSemilatticeSup_97_);
v___f_99_ = lean_alloc_closure((void*)(lp_mathlib_GeneralizedBooleanAlgebra_toBooleanAlgebra___redArg___lam__1), 5, 3);
lean_closure_set(v___f_99_, 0, v_toSemilatticeSup_97_);
lean_closure_set(v___f_99_, 1, v_toSDiff_95_);
lean_closure_set(v___f_99_, 2, v_inst_93_);
v___x_100_ = lean_alloc_ctor(0, 6, 0);
lean_ctor_set(v___x_100_, 0, v_toDistribLattice_94_);
lean_ctor_set(v___x_100_, 1, v___f_98_);
lean_ctor_set(v___x_100_, 2, v_toSDiff_95_);
lean_ctor_set(v___x_100_, 3, v___f_99_);
lean_ctor_set(v___x_100_, 4, v_inst_93_);
lean_ctor_set(v___x_100_, 5, v_toBot_96_);
return v___x_100_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_GeneralizedBooleanAlgebra_toBooleanAlgebra(lean_object* v_00_u03b1_101_, lean_object* v_inst_102_, lean_object* v_inst_103_){
_start:
{
lean_object* v_toDistribLattice_104_; lean_object* v_toSDiff_105_; lean_object* v_toBot_106_; lean_object* v_toSemilatticeSup_107_; lean_object* v___f_108_; lean_object* v___f_109_; lean_object* v___x_110_; 
v_toDistribLattice_104_ = lean_ctor_get(v_inst_102_, 0);
lean_inc_ref(v_toDistribLattice_104_);
v_toSDiff_105_ = lean_ctor_get(v_inst_102_, 1);
lean_inc_n(v_toSDiff_105_, 3);
v_toBot_106_ = lean_ctor_get(v_inst_102_, 2);
lean_inc(v_toBot_106_);
lean_dec_ref(v_inst_102_);
v_toSemilatticeSup_107_ = lean_ctor_get(v_toDistribLattice_104_, 0);
lean_inc_n(v_inst_103_, 2);
v___f_108_ = lean_alloc_closure((void*)(lp_mathlib_GeneralizedBooleanAlgebra_toBooleanAlgebra___redArg___lam__0), 3, 2);
lean_closure_set(v___f_108_, 0, v_toSDiff_105_);
lean_closure_set(v___f_108_, 1, v_inst_103_);
lean_inc_ref(v_toSemilatticeSup_107_);
v___f_109_ = lean_alloc_closure((void*)(lp_mathlib_GeneralizedBooleanAlgebra_toBooleanAlgebra___redArg___lam__1), 5, 3);
lean_closure_set(v___f_109_, 0, v_toSemilatticeSup_107_);
lean_closure_set(v___f_109_, 1, v_toSDiff_105_);
lean_closure_set(v___f_109_, 2, v_inst_103_);
v___x_110_ = lean_alloc_ctor(0, 6, 0);
lean_ctor_set(v___x_110_, 0, v_toDistribLattice_104_);
lean_ctor_set(v___x_110_, 1, v___f_108_);
lean_ctor_set(v___x_110_, 2, v_toSDiff_105_);
lean_ctor_set(v___x_110_, 3, v___f_109_);
lean_ctor_set(v___x_110_, 4, v_inst_103_);
lean_ctor_set(v___x_110_, 5, v_toBot_106_);
return v___x_110_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_BooleanAlgebra_toGeneralizedBooleanAlgebra___redArg(lean_object* v_inst_111_){
_start:
{
lean_object* v_toDistribLattice_112_; lean_object* v_toSDiff_113_; lean_object* v_toBot_114_; lean_object* v___x_115_; 
v_toDistribLattice_112_ = lean_ctor_get(v_inst_111_, 0);
v_toSDiff_113_ = lean_ctor_get(v_inst_111_, 2);
v_toBot_114_ = lean_ctor_get(v_inst_111_, 5);
lean_inc(v_toBot_114_);
lean_inc(v_toSDiff_113_);
lean_inc_ref(v_toDistribLattice_112_);
v___x_115_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_115_, 0, v_toDistribLattice_112_);
lean_ctor_set(v___x_115_, 1, v_toSDiff_113_);
lean_ctor_set(v___x_115_, 2, v_toBot_114_);
return v___x_115_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_BooleanAlgebra_toGeneralizedBooleanAlgebra___redArg___boxed(lean_object* v_inst_116_){
_start:
{
lean_object* v_res_117_; 
v_res_117_ = lp_mathlib_BooleanAlgebra_toGeneralizedBooleanAlgebra___redArg(v_inst_116_);
lean_dec_ref(v_inst_116_);
return v_res_117_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_BooleanAlgebra_toGeneralizedBooleanAlgebra(lean_object* v_00_u03b1_118_, lean_object* v_inst_119_){
_start:
{
lean_object* v___x_120_; 
v___x_120_ = lp_mathlib_BooleanAlgebra_toGeneralizedBooleanAlgebra___redArg(v_inst_119_);
return v___x_120_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_BooleanAlgebra_toGeneralizedBooleanAlgebra___boxed(lean_object* v_00_u03b1_121_, lean_object* v_inst_122_){
_start:
{
lean_object* v_res_123_; 
v_res_123_ = lp_mathlib_BooleanAlgebra_toGeneralizedBooleanAlgebra(v_00_u03b1_121_, v_inst_122_);
lean_dec_ref(v_inst_122_);
return v_res_123_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_BooleanAlgebra_toBiheytingAlgebra___redArg(lean_object* v_inst_124_){
_start:
{
lean_object* v_toDistribLattice_125_; lean_object* v_toCompl_126_; lean_object* v_toSDiff_127_; lean_object* v_toHImp_128_; lean_object* v_toTop_129_; lean_object* v_toBot_130_; lean_object* v___x_131_; lean_object* v___x_132_; lean_object* v___x_133_; 
v_toDistribLattice_125_ = lean_ctor_get(v_inst_124_, 0);
v_toCompl_126_ = lean_ctor_get(v_inst_124_, 1);
v_toSDiff_127_ = lean_ctor_get(v_inst_124_, 2);
v_toHImp_128_ = lean_ctor_get(v_inst_124_, 3);
v_toTop_129_ = lean_ctor_get(v_inst_124_, 4);
v_toBot_130_ = lean_ctor_get(v_inst_124_, 5);
lean_inc(v_toHImp_128_);
lean_inc(v_toTop_129_);
lean_inc_ref(v_toDistribLattice_125_);
v___x_131_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_131_, 0, v_toDistribLattice_125_);
lean_ctor_set(v___x_131_, 1, v_toTop_129_);
lean_ctor_set(v___x_131_, 2, v_toHImp_128_);
lean_inc_n(v_toCompl_126_, 2);
lean_inc(v_toBot_130_);
v___x_132_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_132_, 0, v___x_131_);
lean_ctor_set(v___x_132_, 1, v_toBot_130_);
lean_ctor_set(v___x_132_, 2, v_toCompl_126_);
lean_inc(v_toSDiff_127_);
v___x_133_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_133_, 0, v___x_132_);
lean_ctor_set(v___x_133_, 1, v_toSDiff_127_);
lean_ctor_set(v___x_133_, 2, v_toCompl_126_);
return v___x_133_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_BooleanAlgebra_toBiheytingAlgebra___redArg___boxed(lean_object* v_inst_134_){
_start:
{
lean_object* v_res_135_; 
v_res_135_ = lp_mathlib_BooleanAlgebra_toBiheytingAlgebra___redArg(v_inst_134_);
lean_dec_ref(v_inst_134_);
return v_res_135_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_BooleanAlgebra_toBiheytingAlgebra(lean_object* v_00_u03b1_136_, lean_object* v_inst_137_){
_start:
{
lean_object* v___x_138_; 
v___x_138_ = lp_mathlib_BooleanAlgebra_toBiheytingAlgebra___redArg(v_inst_137_);
return v___x_138_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_BooleanAlgebra_toBiheytingAlgebra___boxed(lean_object* v_00_u03b1_139_, lean_object* v_inst_140_){
_start:
{
lean_object* v_res_141_; 
v_res_141_ = lp_mathlib_BooleanAlgebra_toBiheytingAlgebra(v_00_u03b1_139_, v_inst_140_);
lean_dec_ref(v_inst_140_);
return v_res_141_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderDual_instBooleanAlgebra___redArg(lean_object* v_inst_142_){
_start:
{
lean_object* v_toDistribLattice_143_; lean_object* v___x_144_; lean_object* v___x_145_; lean_object* v___x_147_; uint8_t v_isShared_148_; uint8_t v_isSharedCheck_163_; 
v_toDistribLattice_143_ = lean_ctor_get(v_inst_142_, 0);
lean_inc_ref(v_toDistribLattice_143_);
v___x_144_ = lp_mathlib_OrderDual_instLattice___redArg(v_toDistribLattice_143_);
v___x_145_ = lp_mathlib_BooleanAlgebra_toBiheytingAlgebra___redArg(v_inst_142_);
v_isSharedCheck_163_ = !lean_is_exclusive(v_inst_142_);
if (v_isSharedCheck_163_ == 0)
{
lean_object* v_unused_164_; lean_object* v_unused_165_; lean_object* v_unused_166_; lean_object* v_unused_167_; lean_object* v_unused_168_; lean_object* v_unused_169_; 
v_unused_164_ = lean_ctor_get(v_inst_142_, 5);
lean_dec(v_unused_164_);
v_unused_165_ = lean_ctor_get(v_inst_142_, 4);
lean_dec(v_unused_165_);
v_unused_166_ = lean_ctor_get(v_inst_142_, 3);
lean_dec(v_unused_166_);
v_unused_167_ = lean_ctor_get(v_inst_142_, 2);
lean_dec(v_unused_167_);
v_unused_168_ = lean_ctor_get(v_inst_142_, 1);
lean_dec(v_unused_168_);
v_unused_169_ = lean_ctor_get(v_inst_142_, 0);
lean_dec(v_unused_169_);
v___x_147_ = v_inst_142_;
v_isShared_148_ = v_isSharedCheck_163_;
goto v_resetjp_146_;
}
else
{
lean_dec(v_inst_142_);
v___x_147_ = lean_box(0);
v_isShared_148_ = v_isSharedCheck_163_;
goto v_resetjp_146_;
}
v_resetjp_146_:
{
lean_object* v___x_149_; lean_object* v___x_150_; lean_object* v_toHeytingAlgebra_151_; lean_object* v_toGeneralizedHeytingAlgebra_152_; lean_object* v_toOrderBot_153_; lean_object* v_toCompl_154_; lean_object* v_toGeneralizedHeytingAlgebra_155_; lean_object* v___x_156_; lean_object* v_toSDiff_157_; lean_object* v_toOrderTop_158_; lean_object* v_toHImp_159_; lean_object* v___x_161_; 
lean_inc_ref(v___x_145_);
v___x_149_ = lp_mathlib_BiheytingAlgebra_toCoheytingAlgebra___redArg(v___x_145_);
v___x_150_ = lp_mathlib_OrderDual_instHeytingAlgebra___redArg(v___x_149_);
v_toHeytingAlgebra_151_ = lean_ctor_get(v___x_145_, 0);
lean_inc_ref(v_toHeytingAlgebra_151_);
lean_dec_ref(v___x_145_);
v_toGeneralizedHeytingAlgebra_152_ = lean_ctor_get(v___x_150_, 0);
lean_inc_ref(v_toGeneralizedHeytingAlgebra_152_);
v_toOrderBot_153_ = lean_ctor_get(v___x_150_, 1);
lean_inc(v_toOrderBot_153_);
v_toCompl_154_ = lean_ctor_get(v___x_150_, 2);
lean_inc(v_toCompl_154_);
lean_dec_ref(v___x_150_);
v_toGeneralizedHeytingAlgebra_155_ = lean_ctor_get(v_toHeytingAlgebra_151_, 0);
lean_inc_ref(v_toGeneralizedHeytingAlgebra_155_);
lean_dec_ref(v_toHeytingAlgebra_151_);
v___x_156_ = lp_mathlib_OrderDual_instGeneralizedCoheytingAlgebra___redArg(v_toGeneralizedHeytingAlgebra_155_);
v_toSDiff_157_ = lean_ctor_get(v___x_156_, 2);
lean_inc(v_toSDiff_157_);
lean_dec_ref(v___x_156_);
v_toOrderTop_158_ = lean_ctor_get(v_toGeneralizedHeytingAlgebra_152_, 1);
lean_inc(v_toOrderTop_158_);
v_toHImp_159_ = lean_ctor_get(v_toGeneralizedHeytingAlgebra_152_, 2);
lean_inc(v_toHImp_159_);
lean_dec_ref(v_toGeneralizedHeytingAlgebra_152_);
if (v_isShared_148_ == 0)
{
lean_ctor_set(v___x_147_, 5, v_toOrderBot_153_);
lean_ctor_set(v___x_147_, 4, v_toOrderTop_158_);
lean_ctor_set(v___x_147_, 3, v_toHImp_159_);
lean_ctor_set(v___x_147_, 2, v_toSDiff_157_);
lean_ctor_set(v___x_147_, 1, v_toCompl_154_);
lean_ctor_set(v___x_147_, 0, v___x_144_);
v___x_161_ = v___x_147_;
goto v_reusejp_160_;
}
else
{
lean_object* v_reuseFailAlloc_162_; 
v_reuseFailAlloc_162_ = lean_alloc_ctor(0, 6, 0);
lean_ctor_set(v_reuseFailAlloc_162_, 0, v___x_144_);
lean_ctor_set(v_reuseFailAlloc_162_, 1, v_toCompl_154_);
lean_ctor_set(v_reuseFailAlloc_162_, 2, v_toSDiff_157_);
lean_ctor_set(v_reuseFailAlloc_162_, 3, v_toHImp_159_);
lean_ctor_set(v_reuseFailAlloc_162_, 4, v_toOrderTop_158_);
lean_ctor_set(v_reuseFailAlloc_162_, 5, v_toOrderBot_153_);
v___x_161_ = v_reuseFailAlloc_162_;
goto v_reusejp_160_;
}
v_reusejp_160_:
{
return v___x_161_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderDual_instBooleanAlgebra(lean_object* v_00_u03b1_170_, lean_object* v_inst_171_){
_start:
{
lean_object* v___x_172_; 
v___x_172_ = lp_mathlib_OrderDual_instBooleanAlgebra___redArg(v_inst_171_);
return v___x_172_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Prod_instBooleanAlgebra___redArg(lean_object* v_inst_173_, lean_object* v_inst_174_){
_start:
{
lean_object* v_toDistribLattice_175_; lean_object* v_toSDiff_176_; lean_object* v_toDistribLattice_177_; lean_object* v_toSDiff_178_; lean_object* v___x_179_; lean_object* v___x_180_; lean_object* v_toHeytingAlgebra_181_; lean_object* v___x_182_; lean_object* v___x_184_; uint8_t v_isShared_185_; uint8_t v_isSharedCheck_197_; 
v_toDistribLattice_175_ = lean_ctor_get(v_inst_173_, 0);
v_toSDiff_176_ = lean_ctor_get(v_inst_173_, 2);
lean_inc(v_toSDiff_176_);
v_toDistribLattice_177_ = lean_ctor_get(v_inst_174_, 0);
v_toSDiff_178_ = lean_ctor_get(v_inst_174_, 2);
lean_inc(v_toSDiff_178_);
lean_inc_ref(v_toDistribLattice_177_);
lean_inc_ref(v_toDistribLattice_175_);
v___x_179_ = lp_mathlib_Prod_instLattice___redArg(v_toDistribLattice_175_, v_toDistribLattice_177_);
v___x_180_ = lp_mathlib_BooleanAlgebra_toBiheytingAlgebra___redArg(v_inst_173_);
lean_dec_ref(v_inst_173_);
v_toHeytingAlgebra_181_ = lean_ctor_get(v___x_180_, 0);
lean_inc_ref(v_toHeytingAlgebra_181_);
lean_dec_ref(v___x_180_);
v___x_182_ = lp_mathlib_BooleanAlgebra_toBiheytingAlgebra___redArg(v_inst_174_);
v_isSharedCheck_197_ = !lean_is_exclusive(v_inst_174_);
if (v_isSharedCheck_197_ == 0)
{
lean_object* v_unused_198_; lean_object* v_unused_199_; lean_object* v_unused_200_; lean_object* v_unused_201_; lean_object* v_unused_202_; lean_object* v_unused_203_; 
v_unused_198_ = lean_ctor_get(v_inst_174_, 5);
lean_dec(v_unused_198_);
v_unused_199_ = lean_ctor_get(v_inst_174_, 4);
lean_dec(v_unused_199_);
v_unused_200_ = lean_ctor_get(v_inst_174_, 3);
lean_dec(v_unused_200_);
v_unused_201_ = lean_ctor_get(v_inst_174_, 2);
lean_dec(v_unused_201_);
v_unused_202_ = lean_ctor_get(v_inst_174_, 1);
lean_dec(v_unused_202_);
v_unused_203_ = lean_ctor_get(v_inst_174_, 0);
lean_dec(v_unused_203_);
v___x_184_ = v_inst_174_;
v_isShared_185_ = v_isSharedCheck_197_;
goto v_resetjp_183_;
}
else
{
lean_dec(v_inst_174_);
v___x_184_ = lean_box(0);
v_isShared_185_ = v_isSharedCheck_197_;
goto v_resetjp_183_;
}
v_resetjp_183_:
{
lean_object* v_toHeytingAlgebra_186_; lean_object* v___x_187_; lean_object* v_toGeneralizedHeytingAlgebra_188_; lean_object* v_toOrderBot_189_; lean_object* v_toCompl_190_; lean_object* v_toOrderTop_191_; lean_object* v_toHImp_192_; lean_object* v___f_193_; lean_object* v___x_195_; 
v_toHeytingAlgebra_186_ = lean_ctor_get(v___x_182_, 0);
lean_inc_ref(v_toHeytingAlgebra_186_);
lean_dec_ref(v___x_182_);
v___x_187_ = lp_mathlib_Prod_instHeytingAlgebra___redArg(v_toHeytingAlgebra_181_, v_toHeytingAlgebra_186_);
v_toGeneralizedHeytingAlgebra_188_ = lean_ctor_get(v___x_187_, 0);
lean_inc_ref(v_toGeneralizedHeytingAlgebra_188_);
v_toOrderBot_189_ = lean_ctor_get(v___x_187_, 1);
lean_inc(v_toOrderBot_189_);
v_toCompl_190_ = lean_ctor_get(v___x_187_, 2);
lean_inc(v_toCompl_190_);
lean_dec_ref(v___x_187_);
v_toOrderTop_191_ = lean_ctor_get(v_toGeneralizedHeytingAlgebra_188_, 1);
lean_inc(v_toOrderTop_191_);
v_toHImp_192_ = lean_ctor_get(v_toGeneralizedHeytingAlgebra_188_, 2);
lean_inc(v_toHImp_192_);
lean_dec_ref(v_toGeneralizedHeytingAlgebra_188_);
v___f_193_ = lean_alloc_closure((void*)(lp_mathlib_Prod_instSDiff___redArg___lam__0), 4, 2);
lean_closure_set(v___f_193_, 0, v_toSDiff_176_);
lean_closure_set(v___f_193_, 1, v_toSDiff_178_);
if (v_isShared_185_ == 0)
{
lean_ctor_set(v___x_184_, 5, v_toOrderBot_189_);
lean_ctor_set(v___x_184_, 4, v_toOrderTop_191_);
lean_ctor_set(v___x_184_, 3, v_toHImp_192_);
lean_ctor_set(v___x_184_, 2, v___f_193_);
lean_ctor_set(v___x_184_, 1, v_toCompl_190_);
lean_ctor_set(v___x_184_, 0, v___x_179_);
v___x_195_ = v___x_184_;
goto v_reusejp_194_;
}
else
{
lean_object* v_reuseFailAlloc_196_; 
v_reuseFailAlloc_196_ = lean_alloc_ctor(0, 6, 0);
lean_ctor_set(v_reuseFailAlloc_196_, 0, v___x_179_);
lean_ctor_set(v_reuseFailAlloc_196_, 1, v_toCompl_190_);
lean_ctor_set(v_reuseFailAlloc_196_, 2, v___f_193_);
lean_ctor_set(v_reuseFailAlloc_196_, 3, v_toHImp_192_);
lean_ctor_set(v_reuseFailAlloc_196_, 4, v_toOrderTop_191_);
lean_ctor_set(v_reuseFailAlloc_196_, 5, v_toOrderBot_189_);
v___x_195_ = v_reuseFailAlloc_196_;
goto v_reusejp_194_;
}
v_reusejp_194_:
{
return v___x_195_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Prod_instBooleanAlgebra(lean_object* v_00_u03b1_204_, lean_object* v_00_u03b2_205_, lean_object* v_inst_206_, lean_object* v_inst_207_){
_start:
{
lean_object* v___x_208_; 
v___x_208_ = lp_mathlib_Prod_instBooleanAlgebra___redArg(v_inst_206_, v_inst_207_);
return v___x_208_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Pi_instBooleanAlgebra___redArg___lam__0(lean_object* v_inst_209_, lean_object* v_i_210_){
_start:
{
lean_object* v___x_211_; lean_object* v_toDistribLattice_212_; 
v___x_211_ = lean_apply_1(v_inst_209_, v_i_210_);
v_toDistribLattice_212_ = lean_ctor_get(v___x_211_, 0);
lean_inc_ref(v_toDistribLattice_212_);
lean_dec_ref(v___x_211_);
return v_toDistribLattice_212_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Pi_instBooleanAlgebra___redArg___lam__1(lean_object* v_inst_213_, lean_object* v_i_214_){
_start:
{
lean_object* v___x_215_; lean_object* v___x_216_; lean_object* v_toHeytingAlgebra_217_; 
v___x_215_ = lean_apply_1(v_inst_213_, v_i_214_);
v___x_216_ = lp_mathlib_BooleanAlgebra_toBiheytingAlgebra___redArg(v___x_215_);
lean_dec_ref(v___x_215_);
v_toHeytingAlgebra_217_ = lean_ctor_get(v___x_216_, 0);
lean_inc_ref(v_toHeytingAlgebra_217_);
lean_dec_ref(v___x_216_);
return v_toHeytingAlgebra_217_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Pi_instBooleanAlgebra___redArg___lam__2(lean_object* v_inst_218_, lean_object* v_i_219_, lean_object* v___y_220_, lean_object* v___y_221_){
_start:
{
lean_object* v___x_222_; lean_object* v_toSDiff_223_; lean_object* v___x_224_; 
v___x_222_ = lean_apply_1(v_inst_218_, v_i_219_);
v_toSDiff_223_ = lean_ctor_get(v___x_222_, 2);
lean_inc(v_toSDiff_223_);
lean_dec_ref(v___x_222_);
v___x_224_ = lean_apply_2(v_toSDiff_223_, v___y_220_, v___y_221_);
return v___x_224_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Pi_instBooleanAlgebra___redArg(lean_object* v_inst_225_){
_start:
{
lean_object* v___f_226_; lean_object* v___f_227_; lean_object* v___x_228_; lean_object* v___x_229_; lean_object* v_toGeneralizedHeytingAlgebra_230_; lean_object* v_toOrderBot_231_; lean_object* v_toCompl_232_; lean_object* v_toOrderTop_233_; lean_object* v_toHImp_234_; lean_object* v___f_235_; lean_object* v___f_236_; lean_object* v___x_237_; 
lean_inc_ref_n(v_inst_225_, 2);
v___f_226_ = lean_alloc_closure((void*)(lp_mathlib_Pi_instBooleanAlgebra___redArg___lam__0), 2, 1);
lean_closure_set(v___f_226_, 0, v_inst_225_);
v___f_227_ = lean_alloc_closure((void*)(lp_mathlib_Pi_instBooleanAlgebra___redArg___lam__1), 2, 1);
lean_closure_set(v___f_227_, 0, v_inst_225_);
v___x_228_ = lp_mathlib_Pi_instDistribLattice___redArg(v___f_226_);
v___x_229_ = lp_mathlib_Pi_instHeytingAlgebra___redArg(v___f_227_);
v_toGeneralizedHeytingAlgebra_230_ = lean_ctor_get(v___x_229_, 0);
lean_inc_ref(v_toGeneralizedHeytingAlgebra_230_);
v_toOrderBot_231_ = lean_ctor_get(v___x_229_, 1);
lean_inc(v_toOrderBot_231_);
v_toCompl_232_ = lean_ctor_get(v___x_229_, 2);
lean_inc(v_toCompl_232_);
lean_dec_ref(v___x_229_);
v_toOrderTop_233_ = lean_ctor_get(v_toGeneralizedHeytingAlgebra_230_, 1);
lean_inc(v_toOrderTop_233_);
v_toHImp_234_ = lean_ctor_get(v_toGeneralizedHeytingAlgebra_230_, 2);
lean_inc(v_toHImp_234_);
lean_dec_ref(v_toGeneralizedHeytingAlgebra_230_);
v___f_235_ = lean_alloc_closure((void*)(lp_mathlib_Pi_instBooleanAlgebra___redArg___lam__2), 4, 1);
lean_closure_set(v___f_235_, 0, v_inst_225_);
v___f_236_ = lean_alloc_closure((void*)(lp_mathlib_Pi_instSDiff___redArg___lam__0), 4, 1);
lean_closure_set(v___f_236_, 0, v___f_235_);
v___x_237_ = lean_alloc_ctor(0, 6, 0);
lean_ctor_set(v___x_237_, 0, v___x_228_);
lean_ctor_set(v___x_237_, 1, v_toCompl_232_);
lean_ctor_set(v___x_237_, 2, v___f_236_);
lean_ctor_set(v___x_237_, 3, v_toHImp_234_);
lean_ctor_set(v___x_237_, 4, v_toOrderTop_233_);
lean_ctor_set(v___x_237_, 5, v_toOrderBot_231_);
return v___x_237_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Pi_instBooleanAlgebra(lean_object* v_00_u03b9_238_, lean_object* v_00_u03b1_239_, lean_object* v_inst_240_){
_start:
{
lean_object* v___x_241_; 
v___x_241_ = lp_mathlib_Pi_instBooleanAlgebra___redArg(v_inst_240_);
return v___x_241_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_generalizedBooleanAlgebra___redArg___lam__0(lean_object* v_inst_242_, lean_object* v_a_243_, lean_object* v_b_244_){
_start:
{
lean_object* v___x_245_; 
v___x_245_ = lean_apply_2(v_inst_242_, v_a_243_, v_b_244_);
return v___x_245_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_generalizedBooleanAlgebra___redArg(lean_object* v_inst_246_, lean_object* v_inst_247_, lean_object* v_inst_248_, lean_object* v_inst_249_, lean_object* v_inst_250_, lean_object* v_inst_251_){
_start:
{
lean_object* v___f_252_; lean_object* v___f_253_; lean_object* v___x_254_; lean_object* v___x_255_; lean_object* v___x_256_; lean_object* v___x_257_; 
v___f_252_ = lean_alloc_closure((void*)(lp_mathlib_Function_Injective_generalizedBooleanAlgebra___redArg___lam__0), 3, 1);
lean_closure_set(v___f_252_, 0, v_inst_246_);
v___f_253_ = lean_alloc_closure((void*)(lp_mathlib_Function_Injective_generalizedBooleanAlgebra___redArg___lam__0), 3, 1);
lean_closure_set(v___f_253_, 0, v_inst_247_);
v___x_254_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_254_, 0, v_inst_248_);
lean_ctor_set(v___x_254_, 1, v_inst_249_);
v___x_255_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_255_, 0, v___x_254_);
lean_ctor_set(v___x_255_, 1, v___f_252_);
v___x_256_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_256_, 0, v___x_255_);
lean_ctor_set(v___x_256_, 1, v___f_253_);
v___x_257_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_257_, 0, v___x_256_);
lean_ctor_set(v___x_257_, 1, v_inst_251_);
lean_ctor_set(v___x_257_, 2, v_inst_250_);
return v___x_257_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_generalizedBooleanAlgebra(lean_object* v_00_u03b1_258_, lean_object* v_00_u03b2_259_, lean_object* v_inst_260_, lean_object* v_inst_261_, lean_object* v_inst_262_, lean_object* v_inst_263_, lean_object* v_inst_264_, lean_object* v_inst_265_, lean_object* v_inst_266_, lean_object* v_f_267_, lean_object* v_hf_268_, lean_object* v_le_269_, lean_object* v_lt_270_, lean_object* v_map__sup_271_, lean_object* v_map__inf_272_, lean_object* v_map__bot_273_, lean_object* v_map__sdiff_274_){
_start:
{
lean_object* v___f_275_; lean_object* v___f_276_; lean_object* v___x_277_; lean_object* v___x_278_; lean_object* v___x_279_; lean_object* v___x_280_; 
v___f_275_ = lean_alloc_closure((void*)(lp_mathlib_Function_Injective_generalizedBooleanAlgebra___redArg___lam__0), 3, 1);
lean_closure_set(v___f_275_, 0, v_inst_260_);
v___f_276_ = lean_alloc_closure((void*)(lp_mathlib_Function_Injective_generalizedBooleanAlgebra___redArg___lam__0), 3, 1);
lean_closure_set(v___f_276_, 0, v_inst_261_);
v___x_277_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_277_, 0, v_inst_262_);
lean_ctor_set(v___x_277_, 1, v_inst_263_);
v___x_278_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_278_, 0, v___x_277_);
lean_ctor_set(v___x_278_, 1, v___f_275_);
v___x_279_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_279_, 0, v___x_278_);
lean_ctor_set(v___x_279_, 1, v___f_276_);
v___x_280_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_280_, 0, v___x_279_);
lean_ctor_set(v___x_280_, 1, v_inst_265_);
lean_ctor_set(v___x_280_, 2, v_inst_264_);
return v___x_280_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_generalizedBooleanAlgebra___boxed(lean_object** _args){
lean_object* v_00_u03b1_281_ = _args[0];
lean_object* v_00_u03b2_282_ = _args[1];
lean_object* v_inst_283_ = _args[2];
lean_object* v_inst_284_ = _args[3];
lean_object* v_inst_285_ = _args[4];
lean_object* v_inst_286_ = _args[5];
lean_object* v_inst_287_ = _args[6];
lean_object* v_inst_288_ = _args[7];
lean_object* v_inst_289_ = _args[8];
lean_object* v_f_290_ = _args[9];
lean_object* v_hf_291_ = _args[10];
lean_object* v_le_292_ = _args[11];
lean_object* v_lt_293_ = _args[12];
lean_object* v_map__sup_294_ = _args[13];
lean_object* v_map__inf_295_ = _args[14];
lean_object* v_map__bot_296_ = _args[15];
lean_object* v_map__sdiff_297_ = _args[16];
_start:
{
lean_object* v_res_298_; 
v_res_298_ = lp_mathlib_Function_Injective_generalizedBooleanAlgebra(v_00_u03b1_281_, v_00_u03b2_282_, v_inst_283_, v_inst_284_, v_inst_285_, v_inst_286_, v_inst_287_, v_inst_288_, v_inst_289_, v_f_290_, v_hf_291_, v_le_292_, v_lt_293_, v_map__sup_294_, v_map__inf_295_, v_map__bot_296_, v_map__sdiff_297_);
lean_dec(v_f_290_);
lean_dec_ref(v_inst_289_);
return v_res_298_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_booleanAlgebra___redArg(lean_object* v_inst_299_, lean_object* v_inst_300_, lean_object* v_inst_301_, lean_object* v_inst_302_, lean_object* v_inst_303_, lean_object* v_inst_304_, lean_object* v_inst_305_, lean_object* v_inst_306_, lean_object* v_inst_307_){
_start:
{
lean_object* v___f_308_; lean_object* v___f_309_; lean_object* v___x_310_; lean_object* v___x_311_; lean_object* v___x_312_; lean_object* v___x_313_; 
v___f_308_ = lean_alloc_closure((void*)(lp_mathlib_Function_Injective_generalizedBooleanAlgebra___redArg___lam__0), 3, 1);
lean_closure_set(v___f_308_, 0, v_inst_299_);
v___f_309_ = lean_alloc_closure((void*)(lp_mathlib_Function_Injective_generalizedBooleanAlgebra___redArg___lam__0), 3, 1);
lean_closure_set(v___f_309_, 0, v_inst_300_);
v___x_310_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_310_, 0, v_inst_301_);
lean_ctor_set(v___x_310_, 1, v_inst_302_);
v___x_311_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_311_, 0, v___x_310_);
lean_ctor_set(v___x_311_, 1, v___f_308_);
v___x_312_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_312_, 0, v___x_311_);
lean_ctor_set(v___x_312_, 1, v___f_309_);
v___x_313_ = lean_alloc_ctor(0, 6, 0);
lean_ctor_set(v___x_313_, 0, v___x_312_);
lean_ctor_set(v___x_313_, 1, v_inst_305_);
lean_ctor_set(v___x_313_, 2, v_inst_306_);
lean_ctor_set(v___x_313_, 3, v_inst_307_);
lean_ctor_set(v___x_313_, 4, v_inst_303_);
lean_ctor_set(v___x_313_, 5, v_inst_304_);
return v___x_313_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_booleanAlgebra(lean_object* v_00_u03b1_314_, lean_object* v_00_u03b2_315_, lean_object* v_inst_316_, lean_object* v_inst_317_, lean_object* v_inst_318_, lean_object* v_inst_319_, lean_object* v_inst_320_, lean_object* v_inst_321_, lean_object* v_inst_322_, lean_object* v_inst_323_, lean_object* v_inst_324_, lean_object* v_inst_325_, lean_object* v_f_326_, lean_object* v_hf_327_, lean_object* v_le_328_, lean_object* v_lt_329_, lean_object* v_map__sup_330_, lean_object* v_map__inf_331_, lean_object* v_map__top_332_, lean_object* v_map__bot_333_, lean_object* v_map__compl_334_, lean_object* v_map__sdiff_335_, lean_object* v_map__himp_336_){
_start:
{
lean_object* v___f_337_; lean_object* v___f_338_; lean_object* v___x_339_; lean_object* v___x_340_; lean_object* v___x_341_; lean_object* v___x_342_; 
v___f_337_ = lean_alloc_closure((void*)(lp_mathlib_Function_Injective_generalizedBooleanAlgebra___redArg___lam__0), 3, 1);
lean_closure_set(v___f_337_, 0, v_inst_316_);
v___f_338_ = lean_alloc_closure((void*)(lp_mathlib_Function_Injective_generalizedBooleanAlgebra___redArg___lam__0), 3, 1);
lean_closure_set(v___f_338_, 0, v_inst_317_);
v___x_339_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_339_, 0, v_inst_318_);
lean_ctor_set(v___x_339_, 1, v_inst_319_);
v___x_340_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_340_, 0, v___x_339_);
lean_ctor_set(v___x_340_, 1, v___f_337_);
v___x_341_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_341_, 0, v___x_340_);
lean_ctor_set(v___x_341_, 1, v___f_338_);
v___x_342_ = lean_alloc_ctor(0, 6, 0);
lean_ctor_set(v___x_342_, 0, v___x_341_);
lean_ctor_set(v___x_342_, 1, v_inst_322_);
lean_ctor_set(v___x_342_, 2, v_inst_323_);
lean_ctor_set(v___x_342_, 3, v_inst_324_);
lean_ctor_set(v___x_342_, 4, v_inst_320_);
lean_ctor_set(v___x_342_, 5, v_inst_321_);
return v___x_342_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_booleanAlgebra___boxed(lean_object** _args){
lean_object* v_00_u03b1_343_ = _args[0];
lean_object* v_00_u03b2_344_ = _args[1];
lean_object* v_inst_345_ = _args[2];
lean_object* v_inst_346_ = _args[3];
lean_object* v_inst_347_ = _args[4];
lean_object* v_inst_348_ = _args[5];
lean_object* v_inst_349_ = _args[6];
lean_object* v_inst_350_ = _args[7];
lean_object* v_inst_351_ = _args[8];
lean_object* v_inst_352_ = _args[9];
lean_object* v_inst_353_ = _args[10];
lean_object* v_inst_354_ = _args[11];
lean_object* v_f_355_ = _args[12];
lean_object* v_hf_356_ = _args[13];
lean_object* v_le_357_ = _args[14];
lean_object* v_lt_358_ = _args[15];
lean_object* v_map__sup_359_ = _args[16];
lean_object* v_map__inf_360_ = _args[17];
lean_object* v_map__top_361_ = _args[18];
lean_object* v_map__bot_362_ = _args[19];
lean_object* v_map__compl_363_ = _args[20];
lean_object* v_map__sdiff_364_ = _args[21];
lean_object* v_map__himp_365_ = _args[22];
_start:
{
lean_object* v_res_366_; 
v_res_366_ = lp_mathlib_Function_Injective_booleanAlgebra(v_00_u03b1_343_, v_00_u03b2_344_, v_inst_345_, v_inst_346_, v_inst_347_, v_inst_348_, v_inst_349_, v_inst_350_, v_inst_351_, v_inst_352_, v_inst_353_, v_inst_354_, v_f_355_, v_hf_356_, v_le_357_, v_lt_358_, v_map__sup_359_, v_map__inf_360_, v_map__top_361_, v_map__bot_362_, v_map__compl_363_, v_map__sdiff_364_, v_map__himp_365_);
lean_dec(v_f_355_);
lean_dec_ref(v_inst_354_);
return v_res_366_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_generalizedBooleanAlgebra___redArg___lam__0(lean_object* v_self_367_, lean_object* v___y_368_){
_start:
{
lean_object* v_toFun_369_; lean_object* v___x_370_; 
v_toFun_369_ = lean_ctor_get(v_self_367_, 0);
lean_inc(v_toFun_369_);
lean_dec_ref(v_self_367_);
v___x_370_ = lean_apply_1(v_toFun_369_, v___y_368_);
return v___x_370_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_generalizedBooleanAlgebra___redArg___lam__1(lean_object* v___f_371_, lean_object* v_e_372_, lean_object* v_inf_373_, lean_object* v_toFun_374_, lean_object* v_a_375_, lean_object* v_b_376_){
_start:
{
lean_object* v___x_377_; lean_object* v___x_378_; lean_object* v___x_379_; lean_object* v___x_380_; 
lean_inc(v___f_371_);
lean_inc_ref(v_e_372_);
v___x_377_ = lean_apply_2(v___f_371_, v_e_372_, v_a_375_);
v___x_378_ = lean_apply_2(v___f_371_, v_e_372_, v_b_376_);
v___x_379_ = lean_apply_2(v_inf_373_, v___x_377_, v___x_378_);
v___x_380_ = lean_apply_1(v_toFun_374_, v___x_379_);
return v___x_380_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_generalizedBooleanAlgebra___redArg___lam__3(lean_object* v_min_381_, lean_object* v_a_382_, lean_object* v_b_383_){
_start:
{
lean_object* v___x_384_; 
v___x_384_ = lean_apply_2(v_min_381_, v_a_382_, v_b_383_);
return v___x_384_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_generalizedBooleanAlgebra___redArg___lam__2(lean_object* v_toSemilatticeSup_385_, lean_object* v___f_386_, lean_object* v_e_387_, lean_object* v_toFun_388_, lean_object* v_a_389_, lean_object* v_b_390_){
_start:
{
lean_object* v_sup_391_; lean_object* v___x_392_; lean_object* v___x_393_; lean_object* v___x_394_; lean_object* v___x_395_; 
v_sup_391_ = lean_ctor_get(v_toSemilatticeSup_385_, 1);
lean_inc(v_sup_391_);
lean_dec_ref(v_toSemilatticeSup_385_);
lean_inc(v___f_386_);
lean_inc_ref(v_e_387_);
v___x_392_ = lean_apply_2(v___f_386_, v_e_387_, v_a_389_);
v___x_393_ = lean_apply_2(v___f_386_, v_e_387_, v_b_390_);
v___x_394_ = lean_apply_2(v_sup_391_, v___x_392_, v___x_393_);
v___x_395_ = lean_apply_1(v_toFun_388_, v___x_394_);
return v___x_395_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_generalizedBooleanAlgebra___redArg___lam__5(lean_object* v___f_396_, lean_object* v_a_397_, lean_object* v_b_398_){
_start:
{
lean_object* v___x_399_; 
v___x_399_ = lean_apply_2(v___f_396_, v_a_397_, v_b_398_);
return v___x_399_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_generalizedBooleanAlgebra___redArg___lam__7(lean_object* v___f_400_, lean_object* v_e_401_, lean_object* v_toSDiff_402_, lean_object* v_toFun_403_, lean_object* v_a_404_, lean_object* v_b_405_){
_start:
{
lean_object* v___x_406_; lean_object* v___x_407_; lean_object* v___x_408_; lean_object* v___x_409_; 
lean_inc(v___f_400_);
lean_inc_ref(v_e_401_);
v___x_406_ = lean_apply_2(v___f_400_, v_e_401_, v_a_404_);
v___x_407_ = lean_apply_2(v___f_400_, v_e_401_, v_b_405_);
v___x_408_ = lean_apply_2(v_toSDiff_402_, v___x_406_, v___x_407_);
v___x_409_ = lean_apply_1(v_toFun_403_, v___x_408_);
return v___x_409_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_generalizedBooleanAlgebra___redArg(lean_object* v_e_411_, lean_object* v_inst_412_){
_start:
{
lean_object* v_toDistribLattice_413_; lean_object* v_toSDiff_414_; lean_object* v_toBot_415_; lean_object* v___x_417_; uint8_t v_isShared_418_; uint8_t v_isSharedCheck_508_; 
v_toDistribLattice_413_ = lean_ctor_get(v_inst_412_, 0);
v_toSDiff_414_ = lean_ctor_get(v_inst_412_, 1);
v_toBot_415_ = lean_ctor_get(v_inst_412_, 2);
v_isSharedCheck_508_ = !lean_is_exclusive(v_inst_412_);
if (v_isSharedCheck_508_ == 0)
{
v___x_417_ = v_inst_412_;
v_isShared_418_ = v_isSharedCheck_508_;
goto v_resetjp_416_;
}
else
{
lean_inc(v_toBot_415_);
lean_inc(v_toSDiff_414_);
lean_inc(v_toDistribLattice_413_);
lean_dec(v_inst_412_);
v___x_417_ = lean_box(0);
v_isShared_418_ = v_isSharedCheck_508_;
goto v_resetjp_416_;
}
v_resetjp_416_:
{
lean_object* v___x_419_; lean_object* v_toFun_420_; lean_object* v___x_422_; uint8_t v_isShared_423_; uint8_t v_isSharedCheck_506_; 
lean_inc_ref(v_e_411_);
v___x_419_ = lp_mathlib_Equiv_symm___redArg(v_e_411_);
v_toFun_420_ = lean_ctor_get(v___x_419_, 0);
v_isSharedCheck_506_ = !lean_is_exclusive(v___x_419_);
if (v_isSharedCheck_506_ == 0)
{
lean_object* v_unused_507_; 
v_unused_507_ = lean_ctor_get(v___x_419_, 1);
lean_dec(v_unused_507_);
v___x_422_ = v___x_419_;
v_isShared_423_ = v_isSharedCheck_506_;
goto v_resetjp_421_;
}
else
{
lean_inc(v_toFun_420_);
lean_dec(v___x_419_);
v___x_422_ = lean_box(0);
v_isShared_423_ = v_isSharedCheck_506_;
goto v_resetjp_421_;
}
v_resetjp_421_:
{
lean_object* v_toSemilatticeSup_424_; lean_object* v_inf_425_; lean_object* v___x_427_; uint8_t v_isShared_428_; uint8_t v_isSharedCheck_505_; 
v_toSemilatticeSup_424_ = lean_ctor_get(v_toDistribLattice_413_, 0);
v_inf_425_ = lean_ctor_get(v_toDistribLattice_413_, 1);
v_isSharedCheck_505_ = !lean_is_exclusive(v_toDistribLattice_413_);
if (v_isSharedCheck_505_ == 0)
{
v___x_427_ = v_toDistribLattice_413_;
v_isShared_428_ = v_isSharedCheck_505_;
goto v_resetjp_426_;
}
else
{
lean_inc(v_inf_425_);
lean_inc(v_toSemilatticeSup_424_);
lean_dec(v_toDistribLattice_413_);
v___x_427_ = lean_box(0);
v_isShared_428_ = v_isSharedCheck_505_;
goto v_resetjp_426_;
}
v_resetjp_426_:
{
lean_object* v___f_429_; lean_object* v_min_430_; lean_object* v_le_431_; lean_object* v_lt_432_; lean_object* v_semilatticeInf_433_; lean_object* v_toPartialOrder_434_; lean_object* v___x_436_; uint8_t v_isShared_437_; uint8_t v_isSharedCheck_503_; 
v___f_429_ = ((lean_object*)(lp_mathlib_Equiv_generalizedBooleanAlgebra___redArg___closed__0));
lean_inc(v_toFun_420_);
lean_inc_ref(v_e_411_);
v_min_430_ = lean_alloc_closure((void*)(lp_mathlib_Equiv_generalizedBooleanAlgebra___redArg___lam__1), 6, 4);
lean_closure_set(v_min_430_, 0, v___f_429_);
lean_closure_set(v_min_430_, 1, v_e_411_);
lean_closure_set(v_min_430_, 2, v_inf_425_);
lean_closure_set(v_min_430_, 3, v_toFun_420_);
v_le_431_ = lean_box(0);
v_lt_432_ = lean_box(0);
lean_inc_ref(v_min_430_);
v_semilatticeInf_433_ = lp_mathlib_Function_Injective_semilatticeInf___redArg(v_min_430_, v_le_431_, v_lt_432_);
v_toPartialOrder_434_ = lean_ctor_get(v_semilatticeInf_433_, 0);
v_isSharedCheck_503_ = !lean_is_exclusive(v_semilatticeInf_433_);
if (v_isSharedCheck_503_ == 0)
{
lean_object* v_unused_504_; 
v_unused_504_ = lean_ctor_get(v_semilatticeInf_433_, 1);
lean_dec(v_unused_504_);
v___x_436_ = v_semilatticeInf_433_;
v_isShared_437_ = v_isSharedCheck_503_;
goto v_resetjp_435_;
}
else
{
lean_inc(v_toPartialOrder_434_);
lean_dec(v_semilatticeInf_433_);
v___x_436_ = lean_box(0);
v_isShared_437_ = v_isSharedCheck_503_;
goto v_resetjp_435_;
}
v_resetjp_435_:
{
lean_object* v_toLE_438_; lean_object* v_toLT_439_; lean_object* v___x_441_; uint8_t v_isShared_442_; uint8_t v_isSharedCheck_502_; 
v_toLE_438_ = lean_ctor_get(v_toPartialOrder_434_, 0);
v_toLT_439_ = lean_ctor_get(v_toPartialOrder_434_, 1);
v_isSharedCheck_502_ = !lean_is_exclusive(v_toPartialOrder_434_);
if (v_isSharedCheck_502_ == 0)
{
v___x_441_ = v_toPartialOrder_434_;
v_isShared_442_ = v_isSharedCheck_502_;
goto v_resetjp_440_;
}
else
{
lean_inc(v_toLT_439_);
lean_inc(v_toLE_438_);
lean_dec(v_toPartialOrder_434_);
v___x_441_ = lean_box(0);
v_isShared_442_ = v_isSharedCheck_502_;
goto v_resetjp_440_;
}
v_resetjp_440_:
{
lean_object* v___f_443_; lean_object* v___f_444_; lean_object* v___x_446_; 
v___f_443_ = lean_alloc_closure((void*)(lp_mathlib_Equiv_generalizedBooleanAlgebra___redArg___lam__3), 3, 1);
lean_closure_set(v___f_443_, 0, v_min_430_);
lean_inc(v_toFun_420_);
lean_inc_ref(v_e_411_);
v___f_444_ = lean_alloc_closure((void*)(lp_mathlib_Equiv_generalizedBooleanAlgebra___redArg___lam__2), 6, 4);
lean_closure_set(v___f_444_, 0, v_toSemilatticeSup_424_);
lean_closure_set(v___f_444_, 1, v___f_429_);
lean_closure_set(v___f_444_, 2, v_e_411_);
lean_closure_set(v___f_444_, 3, v_toFun_420_);
if (v_isShared_442_ == 0)
{
v___x_446_ = v___x_441_;
goto v_reusejp_445_;
}
else
{
lean_object* v_reuseFailAlloc_501_; 
v_reuseFailAlloc_501_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_501_, 0, v_toLE_438_);
lean_ctor_set(v_reuseFailAlloc_501_, 1, v_toLT_439_);
v___x_446_ = v_reuseFailAlloc_501_;
goto v_reusejp_445_;
}
v_reusejp_445_:
{
lean_object* v___x_448_; 
lean_inc_ref(v___f_444_);
if (v_isShared_437_ == 0)
{
lean_ctor_set(v___x_436_, 1, v___f_444_);
lean_ctor_set(v___x_436_, 0, v___x_446_);
v___x_448_ = v___x_436_;
goto v_reusejp_447_;
}
else
{
lean_object* v_reuseFailAlloc_500_; 
v_reuseFailAlloc_500_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_500_, 0, v___x_446_);
lean_ctor_set(v_reuseFailAlloc_500_, 1, v___f_444_);
v___x_448_ = v_reuseFailAlloc_500_;
goto v_reusejp_447_;
}
v_reusejp_447_:
{
lean_object* v_lattice_450_; 
lean_inc_ref(v___f_443_);
if (v_isShared_428_ == 0)
{
lean_ctor_set(v___x_427_, 1, v___f_443_);
lean_ctor_set(v___x_427_, 0, v___x_448_);
v_lattice_450_ = v___x_427_;
goto v_reusejp_449_;
}
else
{
lean_object* v_reuseFailAlloc_499_; 
v_reuseFailAlloc_499_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_499_, 0, v___x_448_);
lean_ctor_set(v_reuseFailAlloc_499_, 1, v___f_443_);
v_lattice_450_ = v_reuseFailAlloc_499_;
goto v_reusejp_449_;
}
v_reusejp_449_:
{
lean_object* v___x_451_; lean_object* v_toPartialOrder_452_; lean_object* v___x_454_; uint8_t v_isShared_455_; uint8_t v_isSharedCheck_497_; 
v___x_451_ = lp_mathlib_Lattice_toSemilatticeInf___redArg(v_lattice_450_);
v_toPartialOrder_452_ = lean_ctor_get(v___x_451_, 0);
v_isSharedCheck_497_ = !lean_is_exclusive(v___x_451_);
if (v_isSharedCheck_497_ == 0)
{
lean_object* v_unused_498_; 
v_unused_498_ = lean_ctor_get(v___x_451_, 1);
lean_dec(v_unused_498_);
v___x_454_ = v___x_451_;
v_isShared_455_ = v_isSharedCheck_497_;
goto v_resetjp_453_;
}
else
{
lean_inc(v_toPartialOrder_452_);
lean_dec(v___x_451_);
v___x_454_ = lean_box(0);
v_isShared_455_ = v_isSharedCheck_497_;
goto v_resetjp_453_;
}
v_resetjp_453_:
{
lean_object* v_toLE_456_; lean_object* v_toLT_457_; lean_object* v___x_459_; uint8_t v_isShared_460_; uint8_t v_isSharedCheck_496_; 
v_toLE_456_ = lean_ctor_get(v_toPartialOrder_452_, 0);
v_toLT_457_ = lean_ctor_get(v_toPartialOrder_452_, 1);
v_isSharedCheck_496_ = !lean_is_exclusive(v_toPartialOrder_452_);
if (v_isSharedCheck_496_ == 0)
{
v___x_459_ = v_toPartialOrder_452_;
v_isShared_460_ = v_isSharedCheck_496_;
goto v_resetjp_458_;
}
else
{
lean_inc(v_toLT_457_);
lean_inc(v_toLE_456_);
lean_dec(v_toPartialOrder_452_);
v___x_459_ = lean_box(0);
v_isShared_460_ = v_isSharedCheck_496_;
goto v_resetjp_458_;
}
v_resetjp_458_:
{
lean_object* v___f_461_; lean_object* v___x_463_; 
v___f_461_ = lean_alloc_closure((void*)(lp_mathlib_Equiv_generalizedBooleanAlgebra___redArg___lam__5), 3, 1);
lean_closure_set(v___f_461_, 0, v___f_444_);
if (v_isShared_460_ == 0)
{
v___x_463_ = v___x_459_;
goto v_reusejp_462_;
}
else
{
lean_object* v_reuseFailAlloc_495_; 
v_reuseFailAlloc_495_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_495_, 0, v_toLE_456_);
lean_ctor_set(v_reuseFailAlloc_495_, 1, v_toLT_457_);
v___x_463_ = v_reuseFailAlloc_495_;
goto v_reusejp_462_;
}
v_reusejp_462_:
{
lean_object* v___x_465_; 
lean_inc_ref(v___f_461_);
if (v_isShared_455_ == 0)
{
lean_ctor_set(v___x_454_, 1, v___f_461_);
lean_ctor_set(v___x_454_, 0, v___x_463_);
v___x_465_ = v___x_454_;
goto v_reusejp_464_;
}
else
{
lean_object* v_reuseFailAlloc_494_; 
v_reuseFailAlloc_494_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_494_, 0, v___x_463_);
lean_ctor_set(v_reuseFailAlloc_494_, 1, v___f_461_);
v___x_465_ = v_reuseFailAlloc_494_;
goto v_reusejp_464_;
}
v_reusejp_464_:
{
lean_object* v___x_467_; 
lean_inc_ref(v___f_443_);
if (v_isShared_423_ == 0)
{
lean_ctor_set(v___x_422_, 1, v___f_443_);
lean_ctor_set(v___x_422_, 0, v___x_465_);
v___x_467_ = v___x_422_;
goto v_reusejp_466_;
}
else
{
lean_object* v_reuseFailAlloc_493_; 
v_reuseFailAlloc_493_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_493_, 0, v___x_465_);
lean_ctor_set(v_reuseFailAlloc_493_, 1, v___f_443_);
v___x_467_ = v_reuseFailAlloc_493_;
goto v_reusejp_466_;
}
v_reusejp_466_:
{
lean_object* v___x_468_; lean_object* v_toPartialOrder_469_; lean_object* v___x_471_; uint8_t v_isShared_472_; uint8_t v_isSharedCheck_491_; 
v___x_468_ = lp_mathlib_Lattice_toSemilatticeInf___redArg(v___x_467_);
v_toPartialOrder_469_ = lean_ctor_get(v___x_468_, 0);
v_isSharedCheck_491_ = !lean_is_exclusive(v___x_468_);
if (v_isSharedCheck_491_ == 0)
{
lean_object* v_unused_492_; 
v_unused_492_ = lean_ctor_get(v___x_468_, 1);
lean_dec(v_unused_492_);
v___x_471_ = v___x_468_;
v_isShared_472_ = v_isSharedCheck_491_;
goto v_resetjp_470_;
}
else
{
lean_inc(v_toPartialOrder_469_);
lean_dec(v___x_468_);
v___x_471_ = lean_box(0);
v_isShared_472_ = v_isSharedCheck_491_;
goto v_resetjp_470_;
}
v_resetjp_470_:
{
lean_object* v_toLE_473_; lean_object* v_toLT_474_; lean_object* v___x_476_; uint8_t v_isShared_477_; uint8_t v_isSharedCheck_490_; 
v_toLE_473_ = lean_ctor_get(v_toPartialOrder_469_, 0);
v_toLT_474_ = lean_ctor_get(v_toPartialOrder_469_, 1);
v_isSharedCheck_490_ = !lean_is_exclusive(v_toPartialOrder_469_);
if (v_isSharedCheck_490_ == 0)
{
v___x_476_ = v_toPartialOrder_469_;
v_isShared_477_ = v_isSharedCheck_490_;
goto v_resetjp_475_;
}
else
{
lean_inc(v_toLT_474_);
lean_inc(v_toLE_473_);
lean_dec(v_toPartialOrder_469_);
v___x_476_ = lean_box(0);
v_isShared_477_ = v_isSharedCheck_490_;
goto v_resetjp_475_;
}
v_resetjp_475_:
{
lean_object* v_sdiff_478_; lean_object* v_bot_479_; lean_object* v___x_481_; 
lean_inc(v_toFun_420_);
v_sdiff_478_ = lean_alloc_closure((void*)(lp_mathlib_Equiv_generalizedBooleanAlgebra___redArg___lam__7), 6, 4);
lean_closure_set(v_sdiff_478_, 0, v___f_429_);
lean_closure_set(v_sdiff_478_, 1, v_e_411_);
lean_closure_set(v_sdiff_478_, 2, v_toSDiff_414_);
lean_closure_set(v_sdiff_478_, 3, v_toFun_420_);
v_bot_479_ = lean_apply_1(v_toFun_420_, v_toBot_415_);
if (v_isShared_477_ == 0)
{
v___x_481_ = v___x_476_;
goto v_reusejp_480_;
}
else
{
lean_object* v_reuseFailAlloc_489_; 
v_reuseFailAlloc_489_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_489_, 0, v_toLE_473_);
lean_ctor_set(v_reuseFailAlloc_489_, 1, v_toLT_474_);
v___x_481_ = v_reuseFailAlloc_489_;
goto v_reusejp_480_;
}
v_reusejp_480_:
{
lean_object* v___x_483_; 
if (v_isShared_472_ == 0)
{
lean_ctor_set(v___x_471_, 1, v___f_461_);
lean_ctor_set(v___x_471_, 0, v___x_481_);
v___x_483_ = v___x_471_;
goto v_reusejp_482_;
}
else
{
lean_object* v_reuseFailAlloc_488_; 
v_reuseFailAlloc_488_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_488_, 0, v___x_481_);
lean_ctor_set(v_reuseFailAlloc_488_, 1, v___f_461_);
v___x_483_ = v_reuseFailAlloc_488_;
goto v_reusejp_482_;
}
v_reusejp_482_:
{
lean_object* v___x_484_; lean_object* v___x_486_; 
v___x_484_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_484_, 0, v___x_483_);
lean_ctor_set(v___x_484_, 1, v___f_443_);
if (v_isShared_418_ == 0)
{
lean_ctor_set(v___x_417_, 2, v_bot_479_);
lean_ctor_set(v___x_417_, 1, v_sdiff_478_);
lean_ctor_set(v___x_417_, 0, v___x_484_);
v___x_486_ = v___x_417_;
goto v_reusejp_485_;
}
else
{
lean_object* v_reuseFailAlloc_487_; 
v_reuseFailAlloc_487_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v_reuseFailAlloc_487_, 0, v___x_484_);
lean_ctor_set(v_reuseFailAlloc_487_, 1, v_sdiff_478_);
lean_ctor_set(v_reuseFailAlloc_487_, 2, v_bot_479_);
v___x_486_ = v_reuseFailAlloc_487_;
goto v_reusejp_485_;
}
v_reusejp_485_:
{
return v___x_486_;
}
}
}
}
}
}
}
}
}
}
}
}
}
}
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_generalizedBooleanAlgebra(lean_object* v_00_u03b1_509_, lean_object* v_00_u03b2_510_, lean_object* v_e_511_, lean_object* v_inst_512_){
_start:
{
lean_object* v_toDistribLattice_513_; lean_object* v_toSDiff_514_; lean_object* v_toBot_515_; lean_object* v___x_517_; uint8_t v_isShared_518_; uint8_t v_isSharedCheck_608_; 
v_toDistribLattice_513_ = lean_ctor_get(v_inst_512_, 0);
v_toSDiff_514_ = lean_ctor_get(v_inst_512_, 1);
v_toBot_515_ = lean_ctor_get(v_inst_512_, 2);
v_isSharedCheck_608_ = !lean_is_exclusive(v_inst_512_);
if (v_isSharedCheck_608_ == 0)
{
v___x_517_ = v_inst_512_;
v_isShared_518_ = v_isSharedCheck_608_;
goto v_resetjp_516_;
}
else
{
lean_inc(v_toBot_515_);
lean_inc(v_toSDiff_514_);
lean_inc(v_toDistribLattice_513_);
lean_dec(v_inst_512_);
v___x_517_ = lean_box(0);
v_isShared_518_ = v_isSharedCheck_608_;
goto v_resetjp_516_;
}
v_resetjp_516_:
{
lean_object* v___x_519_; lean_object* v_toFun_520_; lean_object* v___x_522_; uint8_t v_isShared_523_; uint8_t v_isSharedCheck_606_; 
lean_inc_ref(v_e_511_);
v___x_519_ = lp_mathlib_Equiv_symm___redArg(v_e_511_);
v_toFun_520_ = lean_ctor_get(v___x_519_, 0);
v_isSharedCheck_606_ = !lean_is_exclusive(v___x_519_);
if (v_isSharedCheck_606_ == 0)
{
lean_object* v_unused_607_; 
v_unused_607_ = lean_ctor_get(v___x_519_, 1);
lean_dec(v_unused_607_);
v___x_522_ = v___x_519_;
v_isShared_523_ = v_isSharedCheck_606_;
goto v_resetjp_521_;
}
else
{
lean_inc(v_toFun_520_);
lean_dec(v___x_519_);
v___x_522_ = lean_box(0);
v_isShared_523_ = v_isSharedCheck_606_;
goto v_resetjp_521_;
}
v_resetjp_521_:
{
lean_object* v_toSemilatticeSup_524_; lean_object* v_inf_525_; lean_object* v___x_527_; uint8_t v_isShared_528_; uint8_t v_isSharedCheck_605_; 
v_toSemilatticeSup_524_ = lean_ctor_get(v_toDistribLattice_513_, 0);
v_inf_525_ = lean_ctor_get(v_toDistribLattice_513_, 1);
v_isSharedCheck_605_ = !lean_is_exclusive(v_toDistribLattice_513_);
if (v_isSharedCheck_605_ == 0)
{
v___x_527_ = v_toDistribLattice_513_;
v_isShared_528_ = v_isSharedCheck_605_;
goto v_resetjp_526_;
}
else
{
lean_inc(v_inf_525_);
lean_inc(v_toSemilatticeSup_524_);
lean_dec(v_toDistribLattice_513_);
v___x_527_ = lean_box(0);
v_isShared_528_ = v_isSharedCheck_605_;
goto v_resetjp_526_;
}
v_resetjp_526_:
{
lean_object* v___f_529_; lean_object* v_min_530_; lean_object* v_le_531_; lean_object* v_lt_532_; lean_object* v_semilatticeInf_533_; lean_object* v_toPartialOrder_534_; lean_object* v___x_536_; uint8_t v_isShared_537_; uint8_t v_isSharedCheck_603_; 
v___f_529_ = ((lean_object*)(lp_mathlib_Equiv_generalizedBooleanAlgebra___redArg___closed__0));
lean_inc(v_toFun_520_);
lean_inc_ref(v_e_511_);
v_min_530_ = lean_alloc_closure((void*)(lp_mathlib_Equiv_generalizedBooleanAlgebra___redArg___lam__1), 6, 4);
lean_closure_set(v_min_530_, 0, v___f_529_);
lean_closure_set(v_min_530_, 1, v_e_511_);
lean_closure_set(v_min_530_, 2, v_inf_525_);
lean_closure_set(v_min_530_, 3, v_toFun_520_);
v_le_531_ = lean_box(0);
v_lt_532_ = lean_box(0);
lean_inc_ref(v_min_530_);
v_semilatticeInf_533_ = lp_mathlib_Function_Injective_semilatticeInf___redArg(v_min_530_, v_le_531_, v_lt_532_);
v_toPartialOrder_534_ = lean_ctor_get(v_semilatticeInf_533_, 0);
v_isSharedCheck_603_ = !lean_is_exclusive(v_semilatticeInf_533_);
if (v_isSharedCheck_603_ == 0)
{
lean_object* v_unused_604_; 
v_unused_604_ = lean_ctor_get(v_semilatticeInf_533_, 1);
lean_dec(v_unused_604_);
v___x_536_ = v_semilatticeInf_533_;
v_isShared_537_ = v_isSharedCheck_603_;
goto v_resetjp_535_;
}
else
{
lean_inc(v_toPartialOrder_534_);
lean_dec(v_semilatticeInf_533_);
v___x_536_ = lean_box(0);
v_isShared_537_ = v_isSharedCheck_603_;
goto v_resetjp_535_;
}
v_resetjp_535_:
{
lean_object* v_toLE_538_; lean_object* v_toLT_539_; lean_object* v___x_541_; uint8_t v_isShared_542_; uint8_t v_isSharedCheck_602_; 
v_toLE_538_ = lean_ctor_get(v_toPartialOrder_534_, 0);
v_toLT_539_ = lean_ctor_get(v_toPartialOrder_534_, 1);
v_isSharedCheck_602_ = !lean_is_exclusive(v_toPartialOrder_534_);
if (v_isSharedCheck_602_ == 0)
{
v___x_541_ = v_toPartialOrder_534_;
v_isShared_542_ = v_isSharedCheck_602_;
goto v_resetjp_540_;
}
else
{
lean_inc(v_toLT_539_);
lean_inc(v_toLE_538_);
lean_dec(v_toPartialOrder_534_);
v___x_541_ = lean_box(0);
v_isShared_542_ = v_isSharedCheck_602_;
goto v_resetjp_540_;
}
v_resetjp_540_:
{
lean_object* v___f_543_; lean_object* v___f_544_; lean_object* v___x_546_; 
v___f_543_ = lean_alloc_closure((void*)(lp_mathlib_Equiv_generalizedBooleanAlgebra___redArg___lam__3), 3, 1);
lean_closure_set(v___f_543_, 0, v_min_530_);
lean_inc(v_toFun_520_);
lean_inc_ref(v_e_511_);
v___f_544_ = lean_alloc_closure((void*)(lp_mathlib_Equiv_generalizedBooleanAlgebra___redArg___lam__2), 6, 4);
lean_closure_set(v___f_544_, 0, v_toSemilatticeSup_524_);
lean_closure_set(v___f_544_, 1, v___f_529_);
lean_closure_set(v___f_544_, 2, v_e_511_);
lean_closure_set(v___f_544_, 3, v_toFun_520_);
if (v_isShared_542_ == 0)
{
v___x_546_ = v___x_541_;
goto v_reusejp_545_;
}
else
{
lean_object* v_reuseFailAlloc_601_; 
v_reuseFailAlloc_601_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_601_, 0, v_toLE_538_);
lean_ctor_set(v_reuseFailAlloc_601_, 1, v_toLT_539_);
v___x_546_ = v_reuseFailAlloc_601_;
goto v_reusejp_545_;
}
v_reusejp_545_:
{
lean_object* v___x_548_; 
lean_inc_ref(v___f_544_);
if (v_isShared_537_ == 0)
{
lean_ctor_set(v___x_536_, 1, v___f_544_);
lean_ctor_set(v___x_536_, 0, v___x_546_);
v___x_548_ = v___x_536_;
goto v_reusejp_547_;
}
else
{
lean_object* v_reuseFailAlloc_600_; 
v_reuseFailAlloc_600_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_600_, 0, v___x_546_);
lean_ctor_set(v_reuseFailAlloc_600_, 1, v___f_544_);
v___x_548_ = v_reuseFailAlloc_600_;
goto v_reusejp_547_;
}
v_reusejp_547_:
{
lean_object* v_lattice_550_; 
lean_inc_ref(v___f_543_);
if (v_isShared_528_ == 0)
{
lean_ctor_set(v___x_527_, 1, v___f_543_);
lean_ctor_set(v___x_527_, 0, v___x_548_);
v_lattice_550_ = v___x_527_;
goto v_reusejp_549_;
}
else
{
lean_object* v_reuseFailAlloc_599_; 
v_reuseFailAlloc_599_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_599_, 0, v___x_548_);
lean_ctor_set(v_reuseFailAlloc_599_, 1, v___f_543_);
v_lattice_550_ = v_reuseFailAlloc_599_;
goto v_reusejp_549_;
}
v_reusejp_549_:
{
lean_object* v___x_551_; lean_object* v_toPartialOrder_552_; lean_object* v___x_554_; uint8_t v_isShared_555_; uint8_t v_isSharedCheck_597_; 
v___x_551_ = lp_mathlib_Lattice_toSemilatticeInf___redArg(v_lattice_550_);
v_toPartialOrder_552_ = lean_ctor_get(v___x_551_, 0);
v_isSharedCheck_597_ = !lean_is_exclusive(v___x_551_);
if (v_isSharedCheck_597_ == 0)
{
lean_object* v_unused_598_; 
v_unused_598_ = lean_ctor_get(v___x_551_, 1);
lean_dec(v_unused_598_);
v___x_554_ = v___x_551_;
v_isShared_555_ = v_isSharedCheck_597_;
goto v_resetjp_553_;
}
else
{
lean_inc(v_toPartialOrder_552_);
lean_dec(v___x_551_);
v___x_554_ = lean_box(0);
v_isShared_555_ = v_isSharedCheck_597_;
goto v_resetjp_553_;
}
v_resetjp_553_:
{
lean_object* v_toLE_556_; lean_object* v_toLT_557_; lean_object* v___x_559_; uint8_t v_isShared_560_; uint8_t v_isSharedCheck_596_; 
v_toLE_556_ = lean_ctor_get(v_toPartialOrder_552_, 0);
v_toLT_557_ = lean_ctor_get(v_toPartialOrder_552_, 1);
v_isSharedCheck_596_ = !lean_is_exclusive(v_toPartialOrder_552_);
if (v_isSharedCheck_596_ == 0)
{
v___x_559_ = v_toPartialOrder_552_;
v_isShared_560_ = v_isSharedCheck_596_;
goto v_resetjp_558_;
}
else
{
lean_inc(v_toLT_557_);
lean_inc(v_toLE_556_);
lean_dec(v_toPartialOrder_552_);
v___x_559_ = lean_box(0);
v_isShared_560_ = v_isSharedCheck_596_;
goto v_resetjp_558_;
}
v_resetjp_558_:
{
lean_object* v___f_561_; lean_object* v___x_563_; 
v___f_561_ = lean_alloc_closure((void*)(lp_mathlib_Equiv_generalizedBooleanAlgebra___redArg___lam__5), 3, 1);
lean_closure_set(v___f_561_, 0, v___f_544_);
if (v_isShared_560_ == 0)
{
v___x_563_ = v___x_559_;
goto v_reusejp_562_;
}
else
{
lean_object* v_reuseFailAlloc_595_; 
v_reuseFailAlloc_595_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_595_, 0, v_toLE_556_);
lean_ctor_set(v_reuseFailAlloc_595_, 1, v_toLT_557_);
v___x_563_ = v_reuseFailAlloc_595_;
goto v_reusejp_562_;
}
v_reusejp_562_:
{
lean_object* v___x_565_; 
lean_inc_ref(v___f_561_);
if (v_isShared_555_ == 0)
{
lean_ctor_set(v___x_554_, 1, v___f_561_);
lean_ctor_set(v___x_554_, 0, v___x_563_);
v___x_565_ = v___x_554_;
goto v_reusejp_564_;
}
else
{
lean_object* v_reuseFailAlloc_594_; 
v_reuseFailAlloc_594_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_594_, 0, v___x_563_);
lean_ctor_set(v_reuseFailAlloc_594_, 1, v___f_561_);
v___x_565_ = v_reuseFailAlloc_594_;
goto v_reusejp_564_;
}
v_reusejp_564_:
{
lean_object* v___x_567_; 
lean_inc_ref(v___f_543_);
if (v_isShared_523_ == 0)
{
lean_ctor_set(v___x_522_, 1, v___f_543_);
lean_ctor_set(v___x_522_, 0, v___x_565_);
v___x_567_ = v___x_522_;
goto v_reusejp_566_;
}
else
{
lean_object* v_reuseFailAlloc_593_; 
v_reuseFailAlloc_593_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_593_, 0, v___x_565_);
lean_ctor_set(v_reuseFailAlloc_593_, 1, v___f_543_);
v___x_567_ = v_reuseFailAlloc_593_;
goto v_reusejp_566_;
}
v_reusejp_566_:
{
lean_object* v___x_568_; lean_object* v_toPartialOrder_569_; lean_object* v___x_571_; uint8_t v_isShared_572_; uint8_t v_isSharedCheck_591_; 
v___x_568_ = lp_mathlib_Lattice_toSemilatticeInf___redArg(v___x_567_);
v_toPartialOrder_569_ = lean_ctor_get(v___x_568_, 0);
v_isSharedCheck_591_ = !lean_is_exclusive(v___x_568_);
if (v_isSharedCheck_591_ == 0)
{
lean_object* v_unused_592_; 
v_unused_592_ = lean_ctor_get(v___x_568_, 1);
lean_dec(v_unused_592_);
v___x_571_ = v___x_568_;
v_isShared_572_ = v_isSharedCheck_591_;
goto v_resetjp_570_;
}
else
{
lean_inc(v_toPartialOrder_569_);
lean_dec(v___x_568_);
v___x_571_ = lean_box(0);
v_isShared_572_ = v_isSharedCheck_591_;
goto v_resetjp_570_;
}
v_resetjp_570_:
{
lean_object* v_toLE_573_; lean_object* v_toLT_574_; lean_object* v___x_576_; uint8_t v_isShared_577_; uint8_t v_isSharedCheck_590_; 
v_toLE_573_ = lean_ctor_get(v_toPartialOrder_569_, 0);
v_toLT_574_ = lean_ctor_get(v_toPartialOrder_569_, 1);
v_isSharedCheck_590_ = !lean_is_exclusive(v_toPartialOrder_569_);
if (v_isSharedCheck_590_ == 0)
{
v___x_576_ = v_toPartialOrder_569_;
v_isShared_577_ = v_isSharedCheck_590_;
goto v_resetjp_575_;
}
else
{
lean_inc(v_toLT_574_);
lean_inc(v_toLE_573_);
lean_dec(v_toPartialOrder_569_);
v___x_576_ = lean_box(0);
v_isShared_577_ = v_isSharedCheck_590_;
goto v_resetjp_575_;
}
v_resetjp_575_:
{
lean_object* v_sdiff_578_; lean_object* v_bot_579_; lean_object* v___x_581_; 
lean_inc(v_toFun_520_);
v_sdiff_578_ = lean_alloc_closure((void*)(lp_mathlib_Equiv_generalizedBooleanAlgebra___redArg___lam__7), 6, 4);
lean_closure_set(v_sdiff_578_, 0, v___f_529_);
lean_closure_set(v_sdiff_578_, 1, v_e_511_);
lean_closure_set(v_sdiff_578_, 2, v_toSDiff_514_);
lean_closure_set(v_sdiff_578_, 3, v_toFun_520_);
v_bot_579_ = lean_apply_1(v_toFun_520_, v_toBot_515_);
if (v_isShared_577_ == 0)
{
v___x_581_ = v___x_576_;
goto v_reusejp_580_;
}
else
{
lean_object* v_reuseFailAlloc_589_; 
v_reuseFailAlloc_589_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_589_, 0, v_toLE_573_);
lean_ctor_set(v_reuseFailAlloc_589_, 1, v_toLT_574_);
v___x_581_ = v_reuseFailAlloc_589_;
goto v_reusejp_580_;
}
v_reusejp_580_:
{
lean_object* v___x_583_; 
if (v_isShared_572_ == 0)
{
lean_ctor_set(v___x_571_, 1, v___f_561_);
lean_ctor_set(v___x_571_, 0, v___x_581_);
v___x_583_ = v___x_571_;
goto v_reusejp_582_;
}
else
{
lean_object* v_reuseFailAlloc_588_; 
v_reuseFailAlloc_588_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_588_, 0, v___x_581_);
lean_ctor_set(v_reuseFailAlloc_588_, 1, v___f_561_);
v___x_583_ = v_reuseFailAlloc_588_;
goto v_reusejp_582_;
}
v_reusejp_582_:
{
lean_object* v___x_584_; lean_object* v___x_586_; 
v___x_584_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_584_, 0, v___x_583_);
lean_ctor_set(v___x_584_, 1, v___f_543_);
if (v_isShared_518_ == 0)
{
lean_ctor_set(v___x_517_, 2, v_bot_579_);
lean_ctor_set(v___x_517_, 1, v_sdiff_578_);
lean_ctor_set(v___x_517_, 0, v___x_584_);
v___x_586_ = v___x_517_;
goto v_reusejp_585_;
}
else
{
lean_object* v_reuseFailAlloc_587_; 
v_reuseFailAlloc_587_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v_reuseFailAlloc_587_, 0, v___x_584_);
lean_ctor_set(v_reuseFailAlloc_587_, 1, v_sdiff_578_);
lean_ctor_set(v_reuseFailAlloc_587_, 2, v_bot_579_);
v___x_586_ = v_reuseFailAlloc_587_;
goto v_reusejp_585_;
}
v_reusejp_585_:
{
return v___x_586_;
}
}
}
}
}
}
}
}
}
}
}
}
}
}
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_booleanAlgebra___redArg___lam__12(lean_object* v_e_609_, lean_object* v_toCompl_610_, lean_object* v_toFun_611_, lean_object* v_a_612_){
_start:
{
lean_object* v_toFun_613_; lean_object* v___x_614_; lean_object* v___x_615_; lean_object* v___x_616_; 
v_toFun_613_ = lean_ctor_get(v_e_609_, 0);
lean_inc(v_toFun_613_);
lean_dec_ref(v_e_609_);
v___x_614_ = lean_apply_1(v_toFun_613_, v_a_612_);
v___x_615_ = lean_apply_1(v_toCompl_610_, v___x_614_);
v___x_616_ = lean_apply_1(v_toFun_611_, v___x_615_);
return v___x_616_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_booleanAlgebra___redArg___lam__0(lean_object* v___f_617_, lean_object* v_e_618_, lean_object* v_toHImp_619_, lean_object* v_toFun_620_, lean_object* v_a_621_, lean_object* v_b_622_){
_start:
{
lean_object* v___x_623_; lean_object* v___x_624_; lean_object* v___x_625_; lean_object* v___x_626_; 
lean_inc(v___f_617_);
lean_inc_ref(v_e_618_);
v___x_623_ = lean_apply_2(v___f_617_, v_e_618_, v_a_621_);
v___x_624_ = lean_apply_2(v___f_617_, v_e_618_, v_b_622_);
v___x_625_ = lean_apply_2(v_toHImp_619_, v___x_623_, v___x_624_);
v___x_626_ = lean_apply_1(v_toFun_620_, v___x_625_);
return v___x_626_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_booleanAlgebra___redArg(lean_object* v_e_627_, lean_object* v_inst_628_){
_start:
{
lean_object* v_toCompl_629_; lean_object* v_toHImp_630_; lean_object* v_toTop_631_; lean_object* v___x_632_; lean_object* v_toFun_633_; lean_object* v___x_635_; uint8_t v_isShared_636_; uint8_t v_isSharedCheck_765_; 
v_toCompl_629_ = lean_ctor_get(v_inst_628_, 1);
lean_inc(v_toCompl_629_);
v_toHImp_630_ = lean_ctor_get(v_inst_628_, 3);
lean_inc(v_toHImp_630_);
v_toTop_631_ = lean_ctor_get(v_inst_628_, 4);
lean_inc(v_toTop_631_);
lean_inc_ref(v_e_627_);
v___x_632_ = lp_mathlib_Equiv_symm___redArg(v_e_627_);
v_toFun_633_ = lean_ctor_get(v___x_632_, 0);
v_isSharedCheck_765_ = !lean_is_exclusive(v___x_632_);
if (v_isSharedCheck_765_ == 0)
{
lean_object* v_unused_766_; 
v_unused_766_ = lean_ctor_get(v___x_632_, 1);
lean_dec(v_unused_766_);
v___x_635_ = v___x_632_;
v_isShared_636_ = v_isSharedCheck_765_;
goto v_resetjp_634_;
}
else
{
lean_inc(v_toFun_633_);
lean_dec(v___x_632_);
v___x_635_ = lean_box(0);
v_isShared_636_ = v_isSharedCheck_765_;
goto v_resetjp_634_;
}
v_resetjp_634_:
{
lean_object* v___x_637_; lean_object* v___x_639_; uint8_t v_isShared_640_; uint8_t v_isSharedCheck_758_; 
v___x_637_ = lp_mathlib_BooleanAlgebra_toGeneralizedBooleanAlgebra___redArg(v_inst_628_);
v_isSharedCheck_758_ = !lean_is_exclusive(v_inst_628_);
if (v_isSharedCheck_758_ == 0)
{
lean_object* v_unused_759_; lean_object* v_unused_760_; lean_object* v_unused_761_; lean_object* v_unused_762_; lean_object* v_unused_763_; lean_object* v_unused_764_; 
v_unused_759_ = lean_ctor_get(v_inst_628_, 5);
lean_dec(v_unused_759_);
v_unused_760_ = lean_ctor_get(v_inst_628_, 4);
lean_dec(v_unused_760_);
v_unused_761_ = lean_ctor_get(v_inst_628_, 3);
lean_dec(v_unused_761_);
v_unused_762_ = lean_ctor_get(v_inst_628_, 2);
lean_dec(v_unused_762_);
v_unused_763_ = lean_ctor_get(v_inst_628_, 1);
lean_dec(v_unused_763_);
v_unused_764_ = lean_ctor_get(v_inst_628_, 0);
lean_dec(v_unused_764_);
v___x_639_ = v_inst_628_;
v_isShared_640_ = v_isSharedCheck_758_;
goto v_resetjp_638_;
}
else
{
lean_dec(v_inst_628_);
v___x_639_ = lean_box(0);
v_isShared_640_ = v_isSharedCheck_758_;
goto v_resetjp_638_;
}
v_resetjp_638_:
{
lean_object* v_toDistribLattice_641_; lean_object* v_toSDiff_642_; lean_object* v_toBot_643_; lean_object* v___x_645_; uint8_t v_isShared_646_; uint8_t v_isSharedCheck_757_; 
v_toDistribLattice_641_ = lean_ctor_get(v___x_637_, 0);
v_toSDiff_642_ = lean_ctor_get(v___x_637_, 1);
v_toBot_643_ = lean_ctor_get(v___x_637_, 2);
v_isSharedCheck_757_ = !lean_is_exclusive(v___x_637_);
if (v_isSharedCheck_757_ == 0)
{
v___x_645_ = v___x_637_;
v_isShared_646_ = v_isSharedCheck_757_;
goto v_resetjp_644_;
}
else
{
lean_inc(v_toBot_643_);
lean_inc(v_toSDiff_642_);
lean_inc(v_toDistribLattice_641_);
lean_dec(v___x_637_);
v___x_645_ = lean_box(0);
v_isShared_646_ = v_isSharedCheck_757_;
goto v_resetjp_644_;
}
v_resetjp_644_:
{
lean_object* v_toSemilatticeSup_647_; lean_object* v_inf_648_; lean_object* v___x_650_; uint8_t v_isShared_651_; uint8_t v_isSharedCheck_756_; 
v_toSemilatticeSup_647_ = lean_ctor_get(v_toDistribLattice_641_, 0);
v_inf_648_ = lean_ctor_get(v_toDistribLattice_641_, 1);
v_isSharedCheck_756_ = !lean_is_exclusive(v_toDistribLattice_641_);
if (v_isSharedCheck_756_ == 0)
{
v___x_650_ = v_toDistribLattice_641_;
v_isShared_651_ = v_isSharedCheck_756_;
goto v_resetjp_649_;
}
else
{
lean_inc(v_inf_648_);
lean_inc(v_toSemilatticeSup_647_);
lean_dec(v_toDistribLattice_641_);
v___x_650_ = lean_box(0);
v_isShared_651_ = v_isSharedCheck_756_;
goto v_resetjp_649_;
}
v_resetjp_649_:
{
lean_object* v___f_652_; lean_object* v_min_653_; lean_object* v_le_654_; lean_object* v_lt_655_; lean_object* v_semilatticeInf_656_; lean_object* v_toPartialOrder_657_; lean_object* v___x_659_; uint8_t v_isShared_660_; uint8_t v_isSharedCheck_754_; 
v___f_652_ = ((lean_object*)(lp_mathlib_Equiv_generalizedBooleanAlgebra___redArg___closed__0));
lean_inc(v_toFun_633_);
lean_inc_ref(v_e_627_);
v_min_653_ = lean_alloc_closure((void*)(lp_mathlib_Equiv_generalizedBooleanAlgebra___redArg___lam__1), 6, 4);
lean_closure_set(v_min_653_, 0, v___f_652_);
lean_closure_set(v_min_653_, 1, v_e_627_);
lean_closure_set(v_min_653_, 2, v_inf_648_);
lean_closure_set(v_min_653_, 3, v_toFun_633_);
v_le_654_ = lean_box(0);
v_lt_655_ = lean_box(0);
lean_inc_ref(v_min_653_);
v_semilatticeInf_656_ = lp_mathlib_Function_Injective_semilatticeInf___redArg(v_min_653_, v_le_654_, v_lt_655_);
v_toPartialOrder_657_ = lean_ctor_get(v_semilatticeInf_656_, 0);
v_isSharedCheck_754_ = !lean_is_exclusive(v_semilatticeInf_656_);
if (v_isSharedCheck_754_ == 0)
{
lean_object* v_unused_755_; 
v_unused_755_ = lean_ctor_get(v_semilatticeInf_656_, 1);
lean_dec(v_unused_755_);
v___x_659_ = v_semilatticeInf_656_;
v_isShared_660_ = v_isSharedCheck_754_;
goto v_resetjp_658_;
}
else
{
lean_inc(v_toPartialOrder_657_);
lean_dec(v_semilatticeInf_656_);
v___x_659_ = lean_box(0);
v_isShared_660_ = v_isSharedCheck_754_;
goto v_resetjp_658_;
}
v_resetjp_658_:
{
lean_object* v_toLE_661_; lean_object* v_toLT_662_; lean_object* v___x_664_; uint8_t v_isShared_665_; uint8_t v_isSharedCheck_753_; 
v_toLE_661_ = lean_ctor_get(v_toPartialOrder_657_, 0);
v_toLT_662_ = lean_ctor_get(v_toPartialOrder_657_, 1);
v_isSharedCheck_753_ = !lean_is_exclusive(v_toPartialOrder_657_);
if (v_isSharedCheck_753_ == 0)
{
v___x_664_ = v_toPartialOrder_657_;
v_isShared_665_ = v_isSharedCheck_753_;
goto v_resetjp_663_;
}
else
{
lean_inc(v_toLT_662_);
lean_inc(v_toLE_661_);
lean_dec(v_toPartialOrder_657_);
v___x_664_ = lean_box(0);
v_isShared_665_ = v_isSharedCheck_753_;
goto v_resetjp_663_;
}
v_resetjp_663_:
{
lean_object* v___f_666_; lean_object* v___f_667_; lean_object* v___x_669_; 
v___f_666_ = lean_alloc_closure((void*)(lp_mathlib_Equiv_generalizedBooleanAlgebra___redArg___lam__3), 3, 1);
lean_closure_set(v___f_666_, 0, v_min_653_);
lean_inc(v_toFun_633_);
lean_inc_ref(v_e_627_);
v___f_667_ = lean_alloc_closure((void*)(lp_mathlib_Equiv_generalizedBooleanAlgebra___redArg___lam__2), 6, 4);
lean_closure_set(v___f_667_, 0, v_toSemilatticeSup_647_);
lean_closure_set(v___f_667_, 1, v___f_652_);
lean_closure_set(v___f_667_, 2, v_e_627_);
lean_closure_set(v___f_667_, 3, v_toFun_633_);
if (v_isShared_665_ == 0)
{
v___x_669_ = v___x_664_;
goto v_reusejp_668_;
}
else
{
lean_object* v_reuseFailAlloc_752_; 
v_reuseFailAlloc_752_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_752_, 0, v_toLE_661_);
lean_ctor_set(v_reuseFailAlloc_752_, 1, v_toLT_662_);
v___x_669_ = v_reuseFailAlloc_752_;
goto v_reusejp_668_;
}
v_reusejp_668_:
{
lean_object* v___x_671_; 
lean_inc_ref(v___f_667_);
if (v_isShared_660_ == 0)
{
lean_ctor_set(v___x_659_, 1, v___f_667_);
lean_ctor_set(v___x_659_, 0, v___x_669_);
v___x_671_ = v___x_659_;
goto v_reusejp_670_;
}
else
{
lean_object* v_reuseFailAlloc_751_; 
v_reuseFailAlloc_751_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_751_, 0, v___x_669_);
lean_ctor_set(v_reuseFailAlloc_751_, 1, v___f_667_);
v___x_671_ = v_reuseFailAlloc_751_;
goto v_reusejp_670_;
}
v_reusejp_670_:
{
lean_object* v_lattice_673_; 
lean_inc_ref(v___f_666_);
if (v_isShared_651_ == 0)
{
lean_ctor_set(v___x_650_, 1, v___f_666_);
lean_ctor_set(v___x_650_, 0, v___x_671_);
v_lattice_673_ = v___x_650_;
goto v_reusejp_672_;
}
else
{
lean_object* v_reuseFailAlloc_750_; 
v_reuseFailAlloc_750_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_750_, 0, v___x_671_);
lean_ctor_set(v_reuseFailAlloc_750_, 1, v___f_666_);
v_lattice_673_ = v_reuseFailAlloc_750_;
goto v_reusejp_672_;
}
v_reusejp_672_:
{
lean_object* v___x_674_; lean_object* v_toPartialOrder_675_; lean_object* v___x_677_; uint8_t v_isShared_678_; uint8_t v_isSharedCheck_748_; 
v___x_674_ = lp_mathlib_Lattice_toSemilatticeInf___redArg(v_lattice_673_);
v_toPartialOrder_675_ = lean_ctor_get(v___x_674_, 0);
v_isSharedCheck_748_ = !lean_is_exclusive(v___x_674_);
if (v_isSharedCheck_748_ == 0)
{
lean_object* v_unused_749_; 
v_unused_749_ = lean_ctor_get(v___x_674_, 1);
lean_dec(v_unused_749_);
v___x_677_ = v___x_674_;
v_isShared_678_ = v_isSharedCheck_748_;
goto v_resetjp_676_;
}
else
{
lean_inc(v_toPartialOrder_675_);
lean_dec(v___x_674_);
v___x_677_ = lean_box(0);
v_isShared_678_ = v_isSharedCheck_748_;
goto v_resetjp_676_;
}
v_resetjp_676_:
{
lean_object* v_toLE_679_; lean_object* v_toLT_680_; lean_object* v___x_682_; uint8_t v_isShared_683_; uint8_t v_isSharedCheck_747_; 
v_toLE_679_ = lean_ctor_get(v_toPartialOrder_675_, 0);
v_toLT_680_ = lean_ctor_get(v_toPartialOrder_675_, 1);
v_isSharedCheck_747_ = !lean_is_exclusive(v_toPartialOrder_675_);
if (v_isSharedCheck_747_ == 0)
{
v___x_682_ = v_toPartialOrder_675_;
v_isShared_683_ = v_isSharedCheck_747_;
goto v_resetjp_681_;
}
else
{
lean_inc(v_toLT_680_);
lean_inc(v_toLE_679_);
lean_dec(v_toPartialOrder_675_);
v___x_682_ = lean_box(0);
v_isShared_683_ = v_isSharedCheck_747_;
goto v_resetjp_681_;
}
v_resetjp_681_:
{
lean_object* v___f_684_; lean_object* v___x_686_; 
v___f_684_ = lean_alloc_closure((void*)(lp_mathlib_Equiv_generalizedBooleanAlgebra___redArg___lam__5), 3, 1);
lean_closure_set(v___f_684_, 0, v___f_667_);
if (v_isShared_683_ == 0)
{
v___x_686_ = v___x_682_;
goto v_reusejp_685_;
}
else
{
lean_object* v_reuseFailAlloc_746_; 
v_reuseFailAlloc_746_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_746_, 0, v_toLE_679_);
lean_ctor_set(v_reuseFailAlloc_746_, 1, v_toLT_680_);
v___x_686_ = v_reuseFailAlloc_746_;
goto v_reusejp_685_;
}
v_reusejp_685_:
{
lean_object* v___x_688_; 
lean_inc_ref(v___f_684_);
if (v_isShared_678_ == 0)
{
lean_ctor_set(v___x_677_, 1, v___f_684_);
lean_ctor_set(v___x_677_, 0, v___x_686_);
v___x_688_ = v___x_677_;
goto v_reusejp_687_;
}
else
{
lean_object* v_reuseFailAlloc_745_; 
v_reuseFailAlloc_745_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_745_, 0, v___x_686_);
lean_ctor_set(v_reuseFailAlloc_745_, 1, v___f_684_);
v___x_688_ = v_reuseFailAlloc_745_;
goto v_reusejp_687_;
}
v_reusejp_687_:
{
lean_object* v___x_690_; 
lean_inc_ref(v___f_666_);
if (v_isShared_636_ == 0)
{
lean_ctor_set(v___x_635_, 1, v___f_666_);
lean_ctor_set(v___x_635_, 0, v___x_688_);
v___x_690_ = v___x_635_;
goto v_reusejp_689_;
}
else
{
lean_object* v_reuseFailAlloc_744_; 
v_reuseFailAlloc_744_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_744_, 0, v___x_688_);
lean_ctor_set(v_reuseFailAlloc_744_, 1, v___f_666_);
v___x_690_ = v_reuseFailAlloc_744_;
goto v_reusejp_689_;
}
v_reusejp_689_:
{
lean_object* v___x_691_; lean_object* v_toPartialOrder_692_; lean_object* v___x_694_; uint8_t v_isShared_695_; uint8_t v_isSharedCheck_742_; 
v___x_691_ = lp_mathlib_Lattice_toSemilatticeInf___redArg(v___x_690_);
v_toPartialOrder_692_ = lean_ctor_get(v___x_691_, 0);
v_isSharedCheck_742_ = !lean_is_exclusive(v___x_691_);
if (v_isSharedCheck_742_ == 0)
{
lean_object* v_unused_743_; 
v_unused_743_ = lean_ctor_get(v___x_691_, 1);
lean_dec(v_unused_743_);
v___x_694_ = v___x_691_;
v_isShared_695_ = v_isSharedCheck_742_;
goto v_resetjp_693_;
}
else
{
lean_inc(v_toPartialOrder_692_);
lean_dec(v___x_691_);
v___x_694_ = lean_box(0);
v_isShared_695_ = v_isSharedCheck_742_;
goto v_resetjp_693_;
}
v_resetjp_693_:
{
lean_object* v_toLE_696_; lean_object* v_toLT_697_; lean_object* v___x_699_; uint8_t v_isShared_700_; uint8_t v_isSharedCheck_741_; 
v_toLE_696_ = lean_ctor_get(v_toPartialOrder_692_, 0);
v_toLT_697_ = lean_ctor_get(v_toPartialOrder_692_, 1);
v_isSharedCheck_741_ = !lean_is_exclusive(v_toPartialOrder_692_);
if (v_isSharedCheck_741_ == 0)
{
v___x_699_ = v_toPartialOrder_692_;
v_isShared_700_ = v_isSharedCheck_741_;
goto v_resetjp_698_;
}
else
{
lean_inc(v_toLT_697_);
lean_inc(v_toLE_696_);
lean_dec(v_toPartialOrder_692_);
v___x_699_ = lean_box(0);
v_isShared_700_ = v_isSharedCheck_741_;
goto v_resetjp_698_;
}
v_resetjp_698_:
{
lean_object* v_top_701_; lean_object* v_sdiff_702_; lean_object* v_bot_703_; lean_object* v___x_705_; 
lean_inc_n(v_toFun_633_, 3);
v_top_701_ = lean_apply_1(v_toFun_633_, v_toTop_631_);
lean_inc_ref(v_e_627_);
v_sdiff_702_ = lean_alloc_closure((void*)(lp_mathlib_Equiv_generalizedBooleanAlgebra___redArg___lam__7), 6, 4);
lean_closure_set(v_sdiff_702_, 0, v___f_652_);
lean_closure_set(v_sdiff_702_, 1, v_e_627_);
lean_closure_set(v_sdiff_702_, 2, v_toSDiff_642_);
lean_closure_set(v_sdiff_702_, 3, v_toFun_633_);
v_bot_703_ = lean_apply_1(v_toFun_633_, v_toBot_643_);
if (v_isShared_700_ == 0)
{
v___x_705_ = v___x_699_;
goto v_reusejp_704_;
}
else
{
lean_object* v_reuseFailAlloc_740_; 
v_reuseFailAlloc_740_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_740_, 0, v_toLE_696_);
lean_ctor_set(v_reuseFailAlloc_740_, 1, v_toLT_697_);
v___x_705_ = v_reuseFailAlloc_740_;
goto v_reusejp_704_;
}
v_reusejp_704_:
{
lean_object* v___x_707_; 
lean_inc_ref(v___f_684_);
if (v_isShared_695_ == 0)
{
lean_ctor_set(v___x_694_, 1, v___f_684_);
lean_ctor_set(v___x_694_, 0, v___x_705_);
v___x_707_ = v___x_694_;
goto v_reusejp_706_;
}
else
{
lean_object* v_reuseFailAlloc_739_; 
v_reuseFailAlloc_739_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_739_, 0, v___x_705_);
lean_ctor_set(v_reuseFailAlloc_739_, 1, v___f_684_);
v___x_707_ = v_reuseFailAlloc_739_;
goto v_reusejp_706_;
}
v_reusejp_706_:
{
lean_object* v___x_708_; lean_object* v_generalizedBooleanAlgebra_710_; 
lean_inc_ref(v___f_666_);
v___x_708_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_708_, 0, v___x_707_);
lean_ctor_set(v___x_708_, 1, v___f_666_);
lean_inc(v_bot_703_);
lean_inc_ref(v_sdiff_702_);
if (v_isShared_646_ == 0)
{
lean_ctor_set(v___x_645_, 2, v_bot_703_);
lean_ctor_set(v___x_645_, 1, v_sdiff_702_);
lean_ctor_set(v___x_645_, 0, v___x_708_);
v_generalizedBooleanAlgebra_710_ = v___x_645_;
goto v_reusejp_709_;
}
else
{
lean_object* v_reuseFailAlloc_738_; 
v_reuseFailAlloc_738_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v_reuseFailAlloc_738_, 0, v___x_708_);
lean_ctor_set(v_reuseFailAlloc_738_, 1, v_sdiff_702_);
lean_ctor_set(v_reuseFailAlloc_738_, 2, v_bot_703_);
v_generalizedBooleanAlgebra_710_ = v_reuseFailAlloc_738_;
goto v_reusejp_709_;
}
v_reusejp_709_:
{
lean_object* v___x_711_; lean_object* v_toLattice_712_; lean_object* v___x_713_; lean_object* v_toPartialOrder_714_; lean_object* v___x_716_; uint8_t v_isShared_717_; uint8_t v_isSharedCheck_736_; 
v___x_711_ = lp_mathlib_GeneralizedBooleanAlgebra_toGeneralizedCoheytingAlgebra___redArg(v_generalizedBooleanAlgebra_710_);
v_toLattice_712_ = lean_ctor_get(v___x_711_, 0);
lean_inc_ref(v_toLattice_712_);
lean_dec_ref(v___x_711_);
v___x_713_ = lp_mathlib_Lattice_toSemilatticeInf___redArg(v_toLattice_712_);
v_toPartialOrder_714_ = lean_ctor_get(v___x_713_, 0);
v_isSharedCheck_736_ = !lean_is_exclusive(v___x_713_);
if (v_isSharedCheck_736_ == 0)
{
lean_object* v_unused_737_; 
v_unused_737_ = lean_ctor_get(v___x_713_, 1);
lean_dec(v_unused_737_);
v___x_716_ = v___x_713_;
v_isShared_717_ = v_isSharedCheck_736_;
goto v_resetjp_715_;
}
else
{
lean_inc(v_toPartialOrder_714_);
lean_dec(v___x_713_);
v___x_716_ = lean_box(0);
v_isShared_717_ = v_isSharedCheck_736_;
goto v_resetjp_715_;
}
v_resetjp_715_:
{
lean_object* v_toLE_718_; lean_object* v_toLT_719_; lean_object* v___x_721_; uint8_t v_isShared_722_; uint8_t v_isSharedCheck_735_; 
v_toLE_718_ = lean_ctor_get(v_toPartialOrder_714_, 0);
v_toLT_719_ = lean_ctor_get(v_toPartialOrder_714_, 1);
v_isSharedCheck_735_ = !lean_is_exclusive(v_toPartialOrder_714_);
if (v_isSharedCheck_735_ == 0)
{
v___x_721_ = v_toPartialOrder_714_;
v_isShared_722_ = v_isSharedCheck_735_;
goto v_resetjp_720_;
}
else
{
lean_inc(v_toLT_719_);
lean_inc(v_toLE_718_);
lean_dec(v_toPartialOrder_714_);
v___x_721_ = lean_box(0);
v_isShared_722_ = v_isSharedCheck_735_;
goto v_resetjp_720_;
}
v_resetjp_720_:
{
lean_object* v_compl_723_; lean_object* v_himp_724_; lean_object* v___x_726_; 
lean_inc(v_toFun_633_);
lean_inc_ref(v_e_627_);
v_compl_723_ = lean_alloc_closure((void*)(lp_mathlib_Equiv_booleanAlgebra___redArg___lam__12), 4, 3);
lean_closure_set(v_compl_723_, 0, v_e_627_);
lean_closure_set(v_compl_723_, 1, v_toCompl_629_);
lean_closure_set(v_compl_723_, 2, v_toFun_633_);
v_himp_724_ = lean_alloc_closure((void*)(lp_mathlib_Equiv_booleanAlgebra___redArg___lam__0), 6, 4);
lean_closure_set(v_himp_724_, 0, v___f_652_);
lean_closure_set(v_himp_724_, 1, v_e_627_);
lean_closure_set(v_himp_724_, 2, v_toHImp_630_);
lean_closure_set(v_himp_724_, 3, v_toFun_633_);
if (v_isShared_722_ == 0)
{
v___x_726_ = v___x_721_;
goto v_reusejp_725_;
}
else
{
lean_object* v_reuseFailAlloc_734_; 
v_reuseFailAlloc_734_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_734_, 0, v_toLE_718_);
lean_ctor_set(v_reuseFailAlloc_734_, 1, v_toLT_719_);
v___x_726_ = v_reuseFailAlloc_734_;
goto v_reusejp_725_;
}
v_reusejp_725_:
{
lean_object* v___x_728_; 
if (v_isShared_717_ == 0)
{
lean_ctor_set(v___x_716_, 1, v___f_684_);
lean_ctor_set(v___x_716_, 0, v___x_726_);
v___x_728_ = v___x_716_;
goto v_reusejp_727_;
}
else
{
lean_object* v_reuseFailAlloc_733_; 
v_reuseFailAlloc_733_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_733_, 0, v___x_726_);
lean_ctor_set(v_reuseFailAlloc_733_, 1, v___f_684_);
v___x_728_ = v_reuseFailAlloc_733_;
goto v_reusejp_727_;
}
v_reusejp_727_:
{
lean_object* v___x_729_; lean_object* v___x_731_; 
v___x_729_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_729_, 0, v___x_728_);
lean_ctor_set(v___x_729_, 1, v___f_666_);
if (v_isShared_640_ == 0)
{
lean_ctor_set(v___x_639_, 5, v_bot_703_);
lean_ctor_set(v___x_639_, 4, v_top_701_);
lean_ctor_set(v___x_639_, 3, v_himp_724_);
lean_ctor_set(v___x_639_, 2, v_sdiff_702_);
lean_ctor_set(v___x_639_, 1, v_compl_723_);
lean_ctor_set(v___x_639_, 0, v___x_729_);
v___x_731_ = v___x_639_;
goto v_reusejp_730_;
}
else
{
lean_object* v_reuseFailAlloc_732_; 
v_reuseFailAlloc_732_ = lean_alloc_ctor(0, 6, 0);
lean_ctor_set(v_reuseFailAlloc_732_, 0, v___x_729_);
lean_ctor_set(v_reuseFailAlloc_732_, 1, v_compl_723_);
lean_ctor_set(v_reuseFailAlloc_732_, 2, v_sdiff_702_);
lean_ctor_set(v_reuseFailAlloc_732_, 3, v_himp_724_);
lean_ctor_set(v_reuseFailAlloc_732_, 4, v_top_701_);
lean_ctor_set(v_reuseFailAlloc_732_, 5, v_bot_703_);
v___x_731_ = v_reuseFailAlloc_732_;
goto v_reusejp_730_;
}
v_reusejp_730_:
{
return v___x_731_;
}
}
}
}
}
}
}
}
}
}
}
}
}
}
}
}
}
}
}
}
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_booleanAlgebra(lean_object* v_00_u03b1_767_, lean_object* v_00_u03b2_768_, lean_object* v_e_769_, lean_object* v_inst_770_){
_start:
{
lean_object* v_toCompl_771_; lean_object* v_toHImp_772_; lean_object* v_toTop_773_; lean_object* v___x_774_; lean_object* v_toFun_775_; lean_object* v___x_777_; uint8_t v_isShared_778_; uint8_t v_isSharedCheck_907_; 
v_toCompl_771_ = lean_ctor_get(v_inst_770_, 1);
lean_inc(v_toCompl_771_);
v_toHImp_772_ = lean_ctor_get(v_inst_770_, 3);
lean_inc(v_toHImp_772_);
v_toTop_773_ = lean_ctor_get(v_inst_770_, 4);
lean_inc(v_toTop_773_);
lean_inc_ref(v_e_769_);
v___x_774_ = lp_mathlib_Equiv_symm___redArg(v_e_769_);
v_toFun_775_ = lean_ctor_get(v___x_774_, 0);
v_isSharedCheck_907_ = !lean_is_exclusive(v___x_774_);
if (v_isSharedCheck_907_ == 0)
{
lean_object* v_unused_908_; 
v_unused_908_ = lean_ctor_get(v___x_774_, 1);
lean_dec(v_unused_908_);
v___x_777_ = v___x_774_;
v_isShared_778_ = v_isSharedCheck_907_;
goto v_resetjp_776_;
}
else
{
lean_inc(v_toFun_775_);
lean_dec(v___x_774_);
v___x_777_ = lean_box(0);
v_isShared_778_ = v_isSharedCheck_907_;
goto v_resetjp_776_;
}
v_resetjp_776_:
{
lean_object* v___x_779_; lean_object* v___x_781_; uint8_t v_isShared_782_; uint8_t v_isSharedCheck_900_; 
v___x_779_ = lp_mathlib_BooleanAlgebra_toGeneralizedBooleanAlgebra___redArg(v_inst_770_);
v_isSharedCheck_900_ = !lean_is_exclusive(v_inst_770_);
if (v_isSharedCheck_900_ == 0)
{
lean_object* v_unused_901_; lean_object* v_unused_902_; lean_object* v_unused_903_; lean_object* v_unused_904_; lean_object* v_unused_905_; lean_object* v_unused_906_; 
v_unused_901_ = lean_ctor_get(v_inst_770_, 5);
lean_dec(v_unused_901_);
v_unused_902_ = lean_ctor_get(v_inst_770_, 4);
lean_dec(v_unused_902_);
v_unused_903_ = lean_ctor_get(v_inst_770_, 3);
lean_dec(v_unused_903_);
v_unused_904_ = lean_ctor_get(v_inst_770_, 2);
lean_dec(v_unused_904_);
v_unused_905_ = lean_ctor_get(v_inst_770_, 1);
lean_dec(v_unused_905_);
v_unused_906_ = lean_ctor_get(v_inst_770_, 0);
lean_dec(v_unused_906_);
v___x_781_ = v_inst_770_;
v_isShared_782_ = v_isSharedCheck_900_;
goto v_resetjp_780_;
}
else
{
lean_dec(v_inst_770_);
v___x_781_ = lean_box(0);
v_isShared_782_ = v_isSharedCheck_900_;
goto v_resetjp_780_;
}
v_resetjp_780_:
{
lean_object* v_toDistribLattice_783_; lean_object* v_toSDiff_784_; lean_object* v_toBot_785_; lean_object* v___x_787_; uint8_t v_isShared_788_; uint8_t v_isSharedCheck_899_; 
v_toDistribLattice_783_ = lean_ctor_get(v___x_779_, 0);
v_toSDiff_784_ = lean_ctor_get(v___x_779_, 1);
v_toBot_785_ = lean_ctor_get(v___x_779_, 2);
v_isSharedCheck_899_ = !lean_is_exclusive(v___x_779_);
if (v_isSharedCheck_899_ == 0)
{
v___x_787_ = v___x_779_;
v_isShared_788_ = v_isSharedCheck_899_;
goto v_resetjp_786_;
}
else
{
lean_inc(v_toBot_785_);
lean_inc(v_toSDiff_784_);
lean_inc(v_toDistribLattice_783_);
lean_dec(v___x_779_);
v___x_787_ = lean_box(0);
v_isShared_788_ = v_isSharedCheck_899_;
goto v_resetjp_786_;
}
v_resetjp_786_:
{
lean_object* v_toSemilatticeSup_789_; lean_object* v_inf_790_; lean_object* v___x_792_; uint8_t v_isShared_793_; uint8_t v_isSharedCheck_898_; 
v_toSemilatticeSup_789_ = lean_ctor_get(v_toDistribLattice_783_, 0);
v_inf_790_ = lean_ctor_get(v_toDistribLattice_783_, 1);
v_isSharedCheck_898_ = !lean_is_exclusive(v_toDistribLattice_783_);
if (v_isSharedCheck_898_ == 0)
{
v___x_792_ = v_toDistribLattice_783_;
v_isShared_793_ = v_isSharedCheck_898_;
goto v_resetjp_791_;
}
else
{
lean_inc(v_inf_790_);
lean_inc(v_toSemilatticeSup_789_);
lean_dec(v_toDistribLattice_783_);
v___x_792_ = lean_box(0);
v_isShared_793_ = v_isSharedCheck_898_;
goto v_resetjp_791_;
}
v_resetjp_791_:
{
lean_object* v___f_794_; lean_object* v_min_795_; lean_object* v_le_796_; lean_object* v_lt_797_; lean_object* v_semilatticeInf_798_; lean_object* v_toPartialOrder_799_; lean_object* v___x_801_; uint8_t v_isShared_802_; uint8_t v_isSharedCheck_896_; 
v___f_794_ = ((lean_object*)(lp_mathlib_Equiv_generalizedBooleanAlgebra___redArg___closed__0));
lean_inc(v_toFun_775_);
lean_inc_ref(v_e_769_);
v_min_795_ = lean_alloc_closure((void*)(lp_mathlib_Equiv_generalizedBooleanAlgebra___redArg___lam__1), 6, 4);
lean_closure_set(v_min_795_, 0, v___f_794_);
lean_closure_set(v_min_795_, 1, v_e_769_);
lean_closure_set(v_min_795_, 2, v_inf_790_);
lean_closure_set(v_min_795_, 3, v_toFun_775_);
v_le_796_ = lean_box(0);
v_lt_797_ = lean_box(0);
lean_inc_ref(v_min_795_);
v_semilatticeInf_798_ = lp_mathlib_Function_Injective_semilatticeInf___redArg(v_min_795_, v_le_796_, v_lt_797_);
v_toPartialOrder_799_ = lean_ctor_get(v_semilatticeInf_798_, 0);
v_isSharedCheck_896_ = !lean_is_exclusive(v_semilatticeInf_798_);
if (v_isSharedCheck_896_ == 0)
{
lean_object* v_unused_897_; 
v_unused_897_ = lean_ctor_get(v_semilatticeInf_798_, 1);
lean_dec(v_unused_897_);
v___x_801_ = v_semilatticeInf_798_;
v_isShared_802_ = v_isSharedCheck_896_;
goto v_resetjp_800_;
}
else
{
lean_inc(v_toPartialOrder_799_);
lean_dec(v_semilatticeInf_798_);
v___x_801_ = lean_box(0);
v_isShared_802_ = v_isSharedCheck_896_;
goto v_resetjp_800_;
}
v_resetjp_800_:
{
lean_object* v_toLE_803_; lean_object* v_toLT_804_; lean_object* v___x_806_; uint8_t v_isShared_807_; uint8_t v_isSharedCheck_895_; 
v_toLE_803_ = lean_ctor_get(v_toPartialOrder_799_, 0);
v_toLT_804_ = lean_ctor_get(v_toPartialOrder_799_, 1);
v_isSharedCheck_895_ = !lean_is_exclusive(v_toPartialOrder_799_);
if (v_isSharedCheck_895_ == 0)
{
v___x_806_ = v_toPartialOrder_799_;
v_isShared_807_ = v_isSharedCheck_895_;
goto v_resetjp_805_;
}
else
{
lean_inc(v_toLT_804_);
lean_inc(v_toLE_803_);
lean_dec(v_toPartialOrder_799_);
v___x_806_ = lean_box(0);
v_isShared_807_ = v_isSharedCheck_895_;
goto v_resetjp_805_;
}
v_resetjp_805_:
{
lean_object* v___f_808_; lean_object* v___f_809_; lean_object* v___x_811_; 
v___f_808_ = lean_alloc_closure((void*)(lp_mathlib_Equiv_generalizedBooleanAlgebra___redArg___lam__3), 3, 1);
lean_closure_set(v___f_808_, 0, v_min_795_);
lean_inc(v_toFun_775_);
lean_inc_ref(v_e_769_);
v___f_809_ = lean_alloc_closure((void*)(lp_mathlib_Equiv_generalizedBooleanAlgebra___redArg___lam__2), 6, 4);
lean_closure_set(v___f_809_, 0, v_toSemilatticeSup_789_);
lean_closure_set(v___f_809_, 1, v___f_794_);
lean_closure_set(v___f_809_, 2, v_e_769_);
lean_closure_set(v___f_809_, 3, v_toFun_775_);
if (v_isShared_807_ == 0)
{
v___x_811_ = v___x_806_;
goto v_reusejp_810_;
}
else
{
lean_object* v_reuseFailAlloc_894_; 
v_reuseFailAlloc_894_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_894_, 0, v_toLE_803_);
lean_ctor_set(v_reuseFailAlloc_894_, 1, v_toLT_804_);
v___x_811_ = v_reuseFailAlloc_894_;
goto v_reusejp_810_;
}
v_reusejp_810_:
{
lean_object* v___x_813_; 
lean_inc_ref(v___f_809_);
if (v_isShared_802_ == 0)
{
lean_ctor_set(v___x_801_, 1, v___f_809_);
lean_ctor_set(v___x_801_, 0, v___x_811_);
v___x_813_ = v___x_801_;
goto v_reusejp_812_;
}
else
{
lean_object* v_reuseFailAlloc_893_; 
v_reuseFailAlloc_893_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_893_, 0, v___x_811_);
lean_ctor_set(v_reuseFailAlloc_893_, 1, v___f_809_);
v___x_813_ = v_reuseFailAlloc_893_;
goto v_reusejp_812_;
}
v_reusejp_812_:
{
lean_object* v_lattice_815_; 
lean_inc_ref(v___f_808_);
if (v_isShared_793_ == 0)
{
lean_ctor_set(v___x_792_, 1, v___f_808_);
lean_ctor_set(v___x_792_, 0, v___x_813_);
v_lattice_815_ = v___x_792_;
goto v_reusejp_814_;
}
else
{
lean_object* v_reuseFailAlloc_892_; 
v_reuseFailAlloc_892_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_892_, 0, v___x_813_);
lean_ctor_set(v_reuseFailAlloc_892_, 1, v___f_808_);
v_lattice_815_ = v_reuseFailAlloc_892_;
goto v_reusejp_814_;
}
v_reusejp_814_:
{
lean_object* v___x_816_; lean_object* v_toPartialOrder_817_; lean_object* v___x_819_; uint8_t v_isShared_820_; uint8_t v_isSharedCheck_890_; 
v___x_816_ = lp_mathlib_Lattice_toSemilatticeInf___redArg(v_lattice_815_);
v_toPartialOrder_817_ = lean_ctor_get(v___x_816_, 0);
v_isSharedCheck_890_ = !lean_is_exclusive(v___x_816_);
if (v_isSharedCheck_890_ == 0)
{
lean_object* v_unused_891_; 
v_unused_891_ = lean_ctor_get(v___x_816_, 1);
lean_dec(v_unused_891_);
v___x_819_ = v___x_816_;
v_isShared_820_ = v_isSharedCheck_890_;
goto v_resetjp_818_;
}
else
{
lean_inc(v_toPartialOrder_817_);
lean_dec(v___x_816_);
v___x_819_ = lean_box(0);
v_isShared_820_ = v_isSharedCheck_890_;
goto v_resetjp_818_;
}
v_resetjp_818_:
{
lean_object* v_toLE_821_; lean_object* v_toLT_822_; lean_object* v___x_824_; uint8_t v_isShared_825_; uint8_t v_isSharedCheck_889_; 
v_toLE_821_ = lean_ctor_get(v_toPartialOrder_817_, 0);
v_toLT_822_ = lean_ctor_get(v_toPartialOrder_817_, 1);
v_isSharedCheck_889_ = !lean_is_exclusive(v_toPartialOrder_817_);
if (v_isSharedCheck_889_ == 0)
{
v___x_824_ = v_toPartialOrder_817_;
v_isShared_825_ = v_isSharedCheck_889_;
goto v_resetjp_823_;
}
else
{
lean_inc(v_toLT_822_);
lean_inc(v_toLE_821_);
lean_dec(v_toPartialOrder_817_);
v___x_824_ = lean_box(0);
v_isShared_825_ = v_isSharedCheck_889_;
goto v_resetjp_823_;
}
v_resetjp_823_:
{
lean_object* v___f_826_; lean_object* v___x_828_; 
v___f_826_ = lean_alloc_closure((void*)(lp_mathlib_Equiv_generalizedBooleanAlgebra___redArg___lam__5), 3, 1);
lean_closure_set(v___f_826_, 0, v___f_809_);
if (v_isShared_825_ == 0)
{
v___x_828_ = v___x_824_;
goto v_reusejp_827_;
}
else
{
lean_object* v_reuseFailAlloc_888_; 
v_reuseFailAlloc_888_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_888_, 0, v_toLE_821_);
lean_ctor_set(v_reuseFailAlloc_888_, 1, v_toLT_822_);
v___x_828_ = v_reuseFailAlloc_888_;
goto v_reusejp_827_;
}
v_reusejp_827_:
{
lean_object* v___x_830_; 
lean_inc_ref(v___f_826_);
if (v_isShared_820_ == 0)
{
lean_ctor_set(v___x_819_, 1, v___f_826_);
lean_ctor_set(v___x_819_, 0, v___x_828_);
v___x_830_ = v___x_819_;
goto v_reusejp_829_;
}
else
{
lean_object* v_reuseFailAlloc_887_; 
v_reuseFailAlloc_887_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_887_, 0, v___x_828_);
lean_ctor_set(v_reuseFailAlloc_887_, 1, v___f_826_);
v___x_830_ = v_reuseFailAlloc_887_;
goto v_reusejp_829_;
}
v_reusejp_829_:
{
lean_object* v___x_832_; 
lean_inc_ref(v___f_808_);
if (v_isShared_778_ == 0)
{
lean_ctor_set(v___x_777_, 1, v___f_808_);
lean_ctor_set(v___x_777_, 0, v___x_830_);
v___x_832_ = v___x_777_;
goto v_reusejp_831_;
}
else
{
lean_object* v_reuseFailAlloc_886_; 
v_reuseFailAlloc_886_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_886_, 0, v___x_830_);
lean_ctor_set(v_reuseFailAlloc_886_, 1, v___f_808_);
v___x_832_ = v_reuseFailAlloc_886_;
goto v_reusejp_831_;
}
v_reusejp_831_:
{
lean_object* v___x_833_; lean_object* v_toPartialOrder_834_; lean_object* v___x_836_; uint8_t v_isShared_837_; uint8_t v_isSharedCheck_884_; 
v___x_833_ = lp_mathlib_Lattice_toSemilatticeInf___redArg(v___x_832_);
v_toPartialOrder_834_ = lean_ctor_get(v___x_833_, 0);
v_isSharedCheck_884_ = !lean_is_exclusive(v___x_833_);
if (v_isSharedCheck_884_ == 0)
{
lean_object* v_unused_885_; 
v_unused_885_ = lean_ctor_get(v___x_833_, 1);
lean_dec(v_unused_885_);
v___x_836_ = v___x_833_;
v_isShared_837_ = v_isSharedCheck_884_;
goto v_resetjp_835_;
}
else
{
lean_inc(v_toPartialOrder_834_);
lean_dec(v___x_833_);
v___x_836_ = lean_box(0);
v_isShared_837_ = v_isSharedCheck_884_;
goto v_resetjp_835_;
}
v_resetjp_835_:
{
lean_object* v_toLE_838_; lean_object* v_toLT_839_; lean_object* v___x_841_; uint8_t v_isShared_842_; uint8_t v_isSharedCheck_883_; 
v_toLE_838_ = lean_ctor_get(v_toPartialOrder_834_, 0);
v_toLT_839_ = lean_ctor_get(v_toPartialOrder_834_, 1);
v_isSharedCheck_883_ = !lean_is_exclusive(v_toPartialOrder_834_);
if (v_isSharedCheck_883_ == 0)
{
v___x_841_ = v_toPartialOrder_834_;
v_isShared_842_ = v_isSharedCheck_883_;
goto v_resetjp_840_;
}
else
{
lean_inc(v_toLT_839_);
lean_inc(v_toLE_838_);
lean_dec(v_toPartialOrder_834_);
v___x_841_ = lean_box(0);
v_isShared_842_ = v_isSharedCheck_883_;
goto v_resetjp_840_;
}
v_resetjp_840_:
{
lean_object* v_top_843_; lean_object* v_sdiff_844_; lean_object* v_bot_845_; lean_object* v___x_847_; 
lean_inc_n(v_toFun_775_, 3);
v_top_843_ = lean_apply_1(v_toFun_775_, v_toTop_773_);
lean_inc_ref(v_e_769_);
v_sdiff_844_ = lean_alloc_closure((void*)(lp_mathlib_Equiv_generalizedBooleanAlgebra___redArg___lam__7), 6, 4);
lean_closure_set(v_sdiff_844_, 0, v___f_794_);
lean_closure_set(v_sdiff_844_, 1, v_e_769_);
lean_closure_set(v_sdiff_844_, 2, v_toSDiff_784_);
lean_closure_set(v_sdiff_844_, 3, v_toFun_775_);
v_bot_845_ = lean_apply_1(v_toFun_775_, v_toBot_785_);
if (v_isShared_842_ == 0)
{
v___x_847_ = v___x_841_;
goto v_reusejp_846_;
}
else
{
lean_object* v_reuseFailAlloc_882_; 
v_reuseFailAlloc_882_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_882_, 0, v_toLE_838_);
lean_ctor_set(v_reuseFailAlloc_882_, 1, v_toLT_839_);
v___x_847_ = v_reuseFailAlloc_882_;
goto v_reusejp_846_;
}
v_reusejp_846_:
{
lean_object* v___x_849_; 
lean_inc_ref(v___f_826_);
if (v_isShared_837_ == 0)
{
lean_ctor_set(v___x_836_, 1, v___f_826_);
lean_ctor_set(v___x_836_, 0, v___x_847_);
v___x_849_ = v___x_836_;
goto v_reusejp_848_;
}
else
{
lean_object* v_reuseFailAlloc_881_; 
v_reuseFailAlloc_881_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_881_, 0, v___x_847_);
lean_ctor_set(v_reuseFailAlloc_881_, 1, v___f_826_);
v___x_849_ = v_reuseFailAlloc_881_;
goto v_reusejp_848_;
}
v_reusejp_848_:
{
lean_object* v___x_850_; lean_object* v_generalizedBooleanAlgebra_852_; 
lean_inc_ref(v___f_808_);
v___x_850_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_850_, 0, v___x_849_);
lean_ctor_set(v___x_850_, 1, v___f_808_);
lean_inc(v_bot_845_);
lean_inc_ref(v_sdiff_844_);
if (v_isShared_788_ == 0)
{
lean_ctor_set(v___x_787_, 2, v_bot_845_);
lean_ctor_set(v___x_787_, 1, v_sdiff_844_);
lean_ctor_set(v___x_787_, 0, v___x_850_);
v_generalizedBooleanAlgebra_852_ = v___x_787_;
goto v_reusejp_851_;
}
else
{
lean_object* v_reuseFailAlloc_880_; 
v_reuseFailAlloc_880_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v_reuseFailAlloc_880_, 0, v___x_850_);
lean_ctor_set(v_reuseFailAlloc_880_, 1, v_sdiff_844_);
lean_ctor_set(v_reuseFailAlloc_880_, 2, v_bot_845_);
v_generalizedBooleanAlgebra_852_ = v_reuseFailAlloc_880_;
goto v_reusejp_851_;
}
v_reusejp_851_:
{
lean_object* v___x_853_; lean_object* v_toLattice_854_; lean_object* v___x_855_; lean_object* v_toPartialOrder_856_; lean_object* v___x_858_; uint8_t v_isShared_859_; uint8_t v_isSharedCheck_878_; 
v___x_853_ = lp_mathlib_GeneralizedBooleanAlgebra_toGeneralizedCoheytingAlgebra___redArg(v_generalizedBooleanAlgebra_852_);
v_toLattice_854_ = lean_ctor_get(v___x_853_, 0);
lean_inc_ref(v_toLattice_854_);
lean_dec_ref(v___x_853_);
v___x_855_ = lp_mathlib_Lattice_toSemilatticeInf___redArg(v_toLattice_854_);
v_toPartialOrder_856_ = lean_ctor_get(v___x_855_, 0);
v_isSharedCheck_878_ = !lean_is_exclusive(v___x_855_);
if (v_isSharedCheck_878_ == 0)
{
lean_object* v_unused_879_; 
v_unused_879_ = lean_ctor_get(v___x_855_, 1);
lean_dec(v_unused_879_);
v___x_858_ = v___x_855_;
v_isShared_859_ = v_isSharedCheck_878_;
goto v_resetjp_857_;
}
else
{
lean_inc(v_toPartialOrder_856_);
lean_dec(v___x_855_);
v___x_858_ = lean_box(0);
v_isShared_859_ = v_isSharedCheck_878_;
goto v_resetjp_857_;
}
v_resetjp_857_:
{
lean_object* v_toLE_860_; lean_object* v_toLT_861_; lean_object* v___x_863_; uint8_t v_isShared_864_; uint8_t v_isSharedCheck_877_; 
v_toLE_860_ = lean_ctor_get(v_toPartialOrder_856_, 0);
v_toLT_861_ = lean_ctor_get(v_toPartialOrder_856_, 1);
v_isSharedCheck_877_ = !lean_is_exclusive(v_toPartialOrder_856_);
if (v_isSharedCheck_877_ == 0)
{
v___x_863_ = v_toPartialOrder_856_;
v_isShared_864_ = v_isSharedCheck_877_;
goto v_resetjp_862_;
}
else
{
lean_inc(v_toLT_861_);
lean_inc(v_toLE_860_);
lean_dec(v_toPartialOrder_856_);
v___x_863_ = lean_box(0);
v_isShared_864_ = v_isSharedCheck_877_;
goto v_resetjp_862_;
}
v_resetjp_862_:
{
lean_object* v_compl_865_; lean_object* v_himp_866_; lean_object* v___x_868_; 
lean_inc(v_toFun_775_);
lean_inc_ref(v_e_769_);
v_compl_865_ = lean_alloc_closure((void*)(lp_mathlib_Equiv_booleanAlgebra___redArg___lam__12), 4, 3);
lean_closure_set(v_compl_865_, 0, v_e_769_);
lean_closure_set(v_compl_865_, 1, v_toCompl_771_);
lean_closure_set(v_compl_865_, 2, v_toFun_775_);
v_himp_866_ = lean_alloc_closure((void*)(lp_mathlib_Equiv_booleanAlgebra___redArg___lam__0), 6, 4);
lean_closure_set(v_himp_866_, 0, v___f_794_);
lean_closure_set(v_himp_866_, 1, v_e_769_);
lean_closure_set(v_himp_866_, 2, v_toHImp_772_);
lean_closure_set(v_himp_866_, 3, v_toFun_775_);
if (v_isShared_864_ == 0)
{
v___x_868_ = v___x_863_;
goto v_reusejp_867_;
}
else
{
lean_object* v_reuseFailAlloc_876_; 
v_reuseFailAlloc_876_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_876_, 0, v_toLE_860_);
lean_ctor_set(v_reuseFailAlloc_876_, 1, v_toLT_861_);
v___x_868_ = v_reuseFailAlloc_876_;
goto v_reusejp_867_;
}
v_reusejp_867_:
{
lean_object* v___x_870_; 
if (v_isShared_859_ == 0)
{
lean_ctor_set(v___x_858_, 1, v___f_826_);
lean_ctor_set(v___x_858_, 0, v___x_868_);
v___x_870_ = v___x_858_;
goto v_reusejp_869_;
}
else
{
lean_object* v_reuseFailAlloc_875_; 
v_reuseFailAlloc_875_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_875_, 0, v___x_868_);
lean_ctor_set(v_reuseFailAlloc_875_, 1, v___f_826_);
v___x_870_ = v_reuseFailAlloc_875_;
goto v_reusejp_869_;
}
v_reusejp_869_:
{
lean_object* v___x_871_; lean_object* v___x_873_; 
v___x_871_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_871_, 0, v___x_870_);
lean_ctor_set(v___x_871_, 1, v___f_808_);
if (v_isShared_782_ == 0)
{
lean_ctor_set(v___x_781_, 5, v_bot_845_);
lean_ctor_set(v___x_781_, 4, v_top_843_);
lean_ctor_set(v___x_781_, 3, v_himp_866_);
lean_ctor_set(v___x_781_, 2, v_sdiff_844_);
lean_ctor_set(v___x_781_, 1, v_compl_865_);
lean_ctor_set(v___x_781_, 0, v___x_871_);
v___x_873_ = v___x_781_;
goto v_reusejp_872_;
}
else
{
lean_object* v_reuseFailAlloc_874_; 
v_reuseFailAlloc_874_ = lean_alloc_ctor(0, 6, 0);
lean_ctor_set(v_reuseFailAlloc_874_, 0, v___x_871_);
lean_ctor_set(v_reuseFailAlloc_874_, 1, v_compl_865_);
lean_ctor_set(v_reuseFailAlloc_874_, 2, v_sdiff_844_);
lean_ctor_set(v_reuseFailAlloc_874_, 3, v_himp_866_);
lean_ctor_set(v_reuseFailAlloc_874_, 4, v_top_843_);
lean_ctor_set(v_reuseFailAlloc_874_, 5, v_bot_845_);
v___x_873_ = v_reuseFailAlloc_874_;
goto v_reusejp_872_;
}
v_reusejp_872_:
{
return v___x_873_;
}
}
}
}
}
}
}
}
}
}
}
}
}
}
}
}
}
}
}
}
}
}
}
}
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Order_BooleanAlgebra_Defs(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Tactic_GRewrite(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_Order_BooleanAlgebra_Basic(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Order_BooleanAlgebra_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Tactic_GRewrite(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_Order_BooleanAlgebra_Basic(uint8_t builtin) {
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
lean_object* initialize_mathlib_Mathlib_Order_BooleanAlgebra_Defs(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Tactic_GRewrite(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Order_BooleanAlgebra_Basic(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Order_BooleanAlgebra_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Tactic_GRewrite(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Order_BooleanAlgebra_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_Order_BooleanAlgebra_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_Order_BooleanAlgebra_Basic(builtin);
}
#ifdef __cplusplus
}
#endif
