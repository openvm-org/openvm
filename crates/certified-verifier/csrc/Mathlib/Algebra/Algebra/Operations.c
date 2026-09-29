// Lean compiler output
// Module: Mathlib.Algebra.Algebra.Operations
// Imports: public import Init public meta import Init public import Mathlib.Algebra.Algebra.Bilinear public import Mathlib.Algebra.Algebra.Opposite public import Mathlib.Algebra.Group.Pointwise.Finset.Basic public import Mathlib.Algebra.Group.Pointwise.Set.BigOperators public import Mathlib.Algebra.Module.Submodule.Finsupp public import Mathlib.Algebra.Ring.NonZeroDivisors public import Mathlib.Algebra.Ring.Submonoid.Pointwise public import Mathlib.Data.Set.Semiring public import Mathlib.GroupTheory.GroupAction.SubMulAction.Pointwise
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
lean_object* lp_mathlib_Submodule_pointwiseAddCommMonoid(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Submodule_pointwiseAdd___lam__0(lean_object*, lean_object*);
lean_object* lp_mathlib_Nat_unaryCast___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_AddMonoid_toAddZeroClass___redArg(lean_object*);
lean_object* lp_mathlib_AddSubmonoid_instCompleteLattice___redArg(lean_object*);
lean_object* lp_mathlib_CompleteLattice_toConditionallyCompleteLattice___redArg(lean_object*);
lean_object* lp_mathlib_ConditionallyCompleteLattice_toConditionallyCompletePartialOrder___redArg(lean_object*);
lean_object* l_npowRec___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Submodule_pointwiseDistribMulAction___lam__0___boxed(lean_object*, lean_object*);
lean_object* lp_mathlib_Submodule_pointwiseNeg___lam__0(lean_object*);
lean_object* lp_mathlib_NPow_ofPow___redArg___lam__0(lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Submodule_completeLattice___redArg(lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_NonUnitalRingHom_id___lam__0___boxed(lean_object*);
lean_object* lp_mathlib_AlgHom_toLinearMap___redArg___lam__0(lean_object*, lean_object*);
lean_object* lp_mathlib_Submodule_map___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Submodule_one(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Submodule_one___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_Submodule_instAddCommMonoidWithOne___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Submodule_pointwiseAdd___lam__0, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Submodule_instAddCommMonoidWithOne___redArg___closed__0 = (const lean_object*)&lp_mathlib_Submodule_instAddCommMonoidWithOne___redArg___closed__0_value;
static const lean_closure_object lp_mathlib_Submodule_instAddCommMonoidWithOne___redArg___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*4, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Nat_unaryCast___boxed, .m_arity = 5, .m_num_fixed = 4, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Submodule_instAddCommMonoidWithOne___redArg___closed__0_value)} };
static const lean_object* lp_mathlib_Submodule_instAddCommMonoidWithOne___redArg___closed__1 = (const lean_object*)&lp_mathlib_Submodule_instAddCommMonoidWithOne___redArg___closed__1_value;
LEAN_EXPORT lean_object* lp_mathlib_Submodule_instAddCommMonoidWithOne___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Submodule_instAddCommMonoidWithOne___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Submodule_instAddCommMonoidWithOne(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Submodule_instAddCommMonoidWithOne___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Submodule_instSMul___redArg___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Submodule_instSMul___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Submodule_instSMul___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Submodule_instSMul(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Submodule_instSMul___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Submodule_mul___redArg___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Submodule_mul___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Submodule_mul___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Submodule_mul(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Submodule_mul___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Submodule_instNonUnitalSemiring___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Submodule_instNonUnitalSemiring___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Submodule_instNonUnitalSemiring(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Submodule_instNonUnitalSemiring___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Submodule_instPowNat___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Submodule_instPowNat___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Submodule_instPowNat___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Submodule_instPowNat(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Submodule_instPowNat___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_Submodule_hasDistribPointwiseNeg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Submodule_pointwiseNeg___lam__0, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Submodule_hasDistribPointwiseNeg___closed__0 = (const lean_object*)&lp_mathlib_Submodule_hasDistribPointwiseNeg___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_Submodule_hasDistribPointwiseNeg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Submodule_hasDistribPointwiseNeg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Submodule_idemSemiring___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Submodule_idemSemiring___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Submodule_idemSemiring(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Submodule_idemSemiring___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_Submodule_mapHom___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_NonUnitalRingHom_id___lam__0___boxed, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Submodule_mapHom___redArg___closed__0 = (const lean_object*)&lp_mathlib_Submodule_mapHom___redArg___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_Submodule_mapHom___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Submodule_mapHom(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Submodule_equivOpposite___lam__0(lean_object*);
static const lean_closure_object lp_mathlib_Submodule_equivOpposite___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Submodule_equivOpposite___lam__0, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Submodule_equivOpposite___closed__0 = (const lean_object*)&lp_mathlib_Submodule_equivOpposite___closed__0_value;
static const lean_ctor_object lp_mathlib_Submodule_equivOpposite___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_Submodule_equivOpposite___closed__0_value),((lean_object*)&lp_mathlib_Submodule_equivOpposite___closed__0_value)}};
static const lean_object* lp_mathlib_Submodule_equivOpposite___closed__1 = (const lean_object*)&lp_mathlib_Submodule_equivOpposite___closed__1_value;
LEAN_EXPORT lean_object* lp_mathlib_Submodule_equivOpposite(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Submodule_equivOpposite___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_Submodule_pointwiseMulSemiringAction___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Submodule_pointwiseDistribMulAction___lam__0___boxed, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Submodule_pointwiseMulSemiringAction___closed__0 = (const lean_object*)&lp_mathlib_Submodule_pointwiseMulSemiringAction___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_Submodule_pointwiseMulSemiringAction(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Submodule_pointwiseMulSemiringAction___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Submodule_instIdemCommSemiring___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Submodule_instIdemCommSemiring___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Submodule_instIdemCommSemiring(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Submodule_instIdemCommSemiring___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Submodule_instDiv___lam__0(lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_Submodule_instDiv___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Submodule_instDiv___lam__0, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Submodule_instDiv___closed__0 = (const lean_object*)&lp_mathlib_Submodule_instDiv___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_Submodule_instDiv(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Submodule_instDiv___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Submodule_one(lean_object* v_R_1_, lean_object* v_inst_2_, lean_object* v_A_3_, lean_object* v_inst_4_, lean_object* v_inst_5_){
_start:
{
lean_object* v___x_6_; 
v___x_6_ = lean_box(0);
return v___x_6_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Submodule_one___boxed(lean_object* v_R_7_, lean_object* v_inst_8_, lean_object* v_A_9_, lean_object* v_inst_10_, lean_object* v_inst_11_){
_start:
{
lean_object* v_res_12_; 
v_res_12_ = lp_mathlib_Submodule_one(v_R_7_, v_inst_8_, v_A_9_, v_inst_10_, v_inst_11_);
lean_dec(v_inst_11_);
lean_dec_ref(v_inst_10_);
lean_dec_ref(v_inst_8_);
return v_res_12_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Submodule_instAddCommMonoidWithOne___redArg(lean_object* v_inst_17_, lean_object* v_inst_18_, lean_object* v_inst_19_){
_start:
{
lean_object* v_toAddCommMonoid_20_; lean_object* v___x_22_; uint8_t v_isShared_23_; uint8_t v_isSharedCheck_30_; 
v_toAddCommMonoid_20_ = lean_ctor_get(v_inst_18_, 0);
v_isSharedCheck_30_ = !lean_is_exclusive(v_inst_18_);
if (v_isSharedCheck_30_ == 0)
{
lean_object* v_unused_31_; lean_object* v_unused_32_; 
v_unused_31_ = lean_ctor_get(v_inst_18_, 2);
lean_dec(v_unused_31_);
v_unused_32_ = lean_ctor_get(v_inst_18_, 1);
lean_dec(v_unused_32_);
v___x_22_ = v_inst_18_;
v_isShared_23_ = v_isSharedCheck_30_;
goto v_resetjp_21_;
}
else
{
lean_inc(v_toAddCommMonoid_20_);
lean_dec(v_inst_18_);
v___x_22_ = lean_box(0);
v_isShared_23_ = v_isSharedCheck_30_;
goto v_resetjp_21_;
}
v_resetjp_21_:
{
lean_object* v___x_24_; lean_object* v___x_25_; lean_object* v___x_26_; lean_object* v___x_28_; 
v___x_24_ = lean_box(0);
v___x_25_ = lp_mathlib_Submodule_pointwiseAddCommMonoid(lean_box(0), lean_box(0), v_inst_17_, v_toAddCommMonoid_20_, v_inst_19_);
lean_dec_ref(v_toAddCommMonoid_20_);
v___x_26_ = ((lean_object*)(lp_mathlib_Submodule_instAddCommMonoidWithOne___redArg___closed__1));
if (v_isShared_23_ == 0)
{
lean_ctor_set(v___x_22_, 2, v___x_24_);
lean_ctor_set(v___x_22_, 1, v___x_25_);
lean_ctor_set(v___x_22_, 0, v___x_26_);
v___x_28_ = v___x_22_;
goto v_reusejp_27_;
}
else
{
lean_object* v_reuseFailAlloc_29_; 
v_reuseFailAlloc_29_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v_reuseFailAlloc_29_, 0, v___x_26_);
lean_ctor_set(v_reuseFailAlloc_29_, 1, v___x_25_);
lean_ctor_set(v_reuseFailAlloc_29_, 2, v___x_24_);
v___x_28_ = v_reuseFailAlloc_29_;
goto v_reusejp_27_;
}
v_reusejp_27_:
{
return v___x_28_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Submodule_instAddCommMonoidWithOne___redArg___boxed(lean_object* v_inst_33_, lean_object* v_inst_34_, lean_object* v_inst_35_){
_start:
{
lean_object* v_res_36_; 
v_res_36_ = lp_mathlib_Submodule_instAddCommMonoidWithOne___redArg(v_inst_33_, v_inst_34_, v_inst_35_);
lean_dec(v_inst_35_);
lean_dec_ref(v_inst_33_);
return v_res_36_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Submodule_instAddCommMonoidWithOne(lean_object* v_R_37_, lean_object* v_inst_38_, lean_object* v_A_39_, lean_object* v_inst_40_, lean_object* v_inst_41_){
_start:
{
lean_object* v___x_42_; 
v___x_42_ = lp_mathlib_Submodule_instAddCommMonoidWithOne___redArg(v_inst_38_, v_inst_40_, v_inst_41_);
return v___x_42_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Submodule_instAddCommMonoidWithOne___boxed(lean_object* v_R_43_, lean_object* v_inst_44_, lean_object* v_A_45_, lean_object* v_inst_46_, lean_object* v_inst_47_){
_start:
{
lean_object* v_res_48_; 
v_res_48_ = lp_mathlib_Submodule_instAddCommMonoidWithOne(v_R_43_, v_inst_44_, v_A_45_, v_inst_46_, v_inst_47_);
lean_dec(v_inst_47_);
lean_dec_ref(v_inst_44_);
return v_res_48_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Submodule_instSMul___redArg___lam__0(lean_object* v_inst_49_, lean_object* v_A_x27_50_, lean_object* v_M_x27_51_){
_start:
{
lean_object* v___x_52_; lean_object* v___x_53_; lean_object* v___x_54_; lean_object* v___x_55_; lean_object* v_toConditionallyCompletePartialOrderSup_56_; lean_object* v_toSupSet_57_; lean_object* v___x_58_; 
v___x_52_ = lp_mathlib_AddMonoid_toAddZeroClass___redArg(v_inst_49_);
v___x_53_ = lp_mathlib_AddSubmonoid_instCompleteLattice___redArg(v___x_52_);
lean_dec_ref(v___x_52_);
v___x_54_ = lp_mathlib_CompleteLattice_toConditionallyCompleteLattice___redArg(v___x_53_);
v___x_55_ = lp_mathlib_ConditionallyCompleteLattice_toConditionallyCompletePartialOrder___redArg(v___x_54_);
v_toConditionallyCompletePartialOrderSup_56_ = lean_ctor_get(v___x_55_, 0);
lean_inc_ref(v_toConditionallyCompletePartialOrderSup_56_);
lean_dec_ref(v___x_55_);
v_toSupSet_57_ = lean_ctor_get(v_toConditionallyCompletePartialOrderSup_56_, 1);
lean_inc(v_toSupSet_57_);
lean_dec_ref(v_toConditionallyCompletePartialOrderSup_56_);
v___x_58_ = lean_apply_1(v_toSupSet_57_, lean_box(0));
return v___x_58_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Submodule_instSMul___redArg___lam__0___boxed(lean_object* v_inst_59_, lean_object* v_A_x27_60_, lean_object* v_M_x27_61_){
_start:
{
lean_object* v_res_62_; 
v_res_62_ = lp_mathlib_Submodule_instSMul___redArg___lam__0(v_inst_59_, v_A_x27_60_, v_M_x27_61_);
lean_dec_ref(v_inst_59_);
return v_res_62_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Submodule_instSMul___redArg(lean_object* v_inst_63_){
_start:
{
lean_object* v___f_64_; 
v___f_64_ = lean_alloc_closure((void*)(lp_mathlib_Submodule_instSMul___redArg___lam__0___boxed), 3, 1);
lean_closure_set(v___f_64_, 0, v_inst_63_);
return v___f_64_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Submodule_instSMul(lean_object* v_R_65_, lean_object* v_inst_66_, lean_object* v_A_67_, lean_object* v_inst_68_, lean_object* v_inst_69_, lean_object* v_M_70_, lean_object* v_inst_71_, lean_object* v_inst_72_, lean_object* v_inst_73_, lean_object* v_inst_74_){
_start:
{
lean_object* v___f_75_; 
v___f_75_ = lean_alloc_closure((void*)(lp_mathlib_Submodule_instSMul___redArg___lam__0___boxed), 3, 1);
lean_closure_set(v___f_75_, 0, v_inst_71_);
return v___f_75_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Submodule_instSMul___boxed(lean_object* v_R_76_, lean_object* v_inst_77_, lean_object* v_A_78_, lean_object* v_inst_79_, lean_object* v_inst_80_, lean_object* v_M_81_, lean_object* v_inst_82_, lean_object* v_inst_83_, lean_object* v_inst_84_, lean_object* v_inst_85_){
_start:
{
lean_object* v_res_86_; 
v_res_86_ = lp_mathlib_Submodule_instSMul(v_R_76_, v_inst_77_, v_A_78_, v_inst_79_, v_inst_80_, v_M_81_, v_inst_82_, v_inst_83_, v_inst_84_, v_inst_85_);
lean_dec(v_inst_84_);
lean_dec(v_inst_83_);
lean_dec(v_inst_80_);
lean_dec_ref(v_inst_79_);
lean_dec_ref(v_inst_77_);
return v_res_86_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Submodule_mul___redArg___lam__0(lean_object* v_toAddCommMonoid_87_, lean_object* v_x1_88_, lean_object* v_x2_89_){
_start:
{
lean_object* v___x_90_; lean_object* v___x_91_; lean_object* v___x_92_; lean_object* v___x_93_; lean_object* v_toConditionallyCompletePartialOrderSup_94_; lean_object* v_toSupSet_95_; lean_object* v___x_96_; 
v___x_90_ = lp_mathlib_AddMonoid_toAddZeroClass___redArg(v_toAddCommMonoid_87_);
v___x_91_ = lp_mathlib_AddSubmonoid_instCompleteLattice___redArg(v___x_90_);
lean_dec_ref(v___x_90_);
v___x_92_ = lp_mathlib_CompleteLattice_toConditionallyCompleteLattice___redArg(v___x_91_);
v___x_93_ = lp_mathlib_ConditionallyCompleteLattice_toConditionallyCompletePartialOrder___redArg(v___x_92_);
v_toConditionallyCompletePartialOrderSup_94_ = lean_ctor_get(v___x_93_, 0);
lean_inc_ref(v_toConditionallyCompletePartialOrderSup_94_);
lean_dec_ref(v___x_93_);
v_toSupSet_95_ = lean_ctor_get(v_toConditionallyCompletePartialOrderSup_94_, 1);
lean_inc(v_toSupSet_95_);
lean_dec_ref(v_toConditionallyCompletePartialOrderSup_94_);
v___x_96_ = lean_apply_1(v_toSupSet_95_, lean_box(0));
return v___x_96_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Submodule_mul___redArg___lam__0___boxed(lean_object* v_toAddCommMonoid_97_, lean_object* v_x1_98_, lean_object* v_x2_99_){
_start:
{
lean_object* v_res_100_; 
v_res_100_ = lp_mathlib_Submodule_mul___redArg___lam__0(v_toAddCommMonoid_97_, v_x1_98_, v_x2_99_);
lean_dec_ref(v_toAddCommMonoid_97_);
return v_res_100_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Submodule_mul___redArg(lean_object* v_inst_101_){
_start:
{
lean_object* v_toAddCommMonoid_102_; lean_object* v___f_103_; 
v_toAddCommMonoid_102_ = lean_ctor_get(v_inst_101_, 0);
lean_inc_ref(v_toAddCommMonoid_102_);
lean_dec_ref(v_inst_101_);
v___f_103_ = lean_alloc_closure((void*)(lp_mathlib_Submodule_mul___redArg___lam__0___boxed), 3, 1);
lean_closure_set(v___f_103_, 0, v_toAddCommMonoid_102_);
return v___f_103_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Submodule_mul(lean_object* v_R_104_, lean_object* v_inst_105_, lean_object* v_A_106_, lean_object* v_inst_107_, lean_object* v_inst_108_, lean_object* v_inst_109_){
_start:
{
lean_object* v___x_110_; 
v___x_110_ = lp_mathlib_Submodule_mul___redArg(v_inst_107_);
return v___x_110_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Submodule_mul___boxed(lean_object* v_R_111_, lean_object* v_inst_112_, lean_object* v_A_113_, lean_object* v_inst_114_, lean_object* v_inst_115_, lean_object* v_inst_116_){
_start:
{
lean_object* v_res_117_; 
v_res_117_ = lp_mathlib_Submodule_mul(v_R_111_, v_inst_112_, v_A_113_, v_inst_114_, v_inst_115_, v_inst_116_);
lean_dec(v_inst_115_);
lean_dec_ref(v_inst_112_);
return v_res_117_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Submodule_instNonUnitalSemiring___redArg(lean_object* v_inst_118_, lean_object* v_inst_119_, lean_object* v_inst_120_){
_start:
{
lean_object* v_toAddCommMonoid_121_; lean_object* v___f_122_; lean_object* v___x_123_; lean_object* v___x_124_; 
v_toAddCommMonoid_121_ = lean_ctor_get(v_inst_119_, 0);
lean_inc_ref_n(v_toAddCommMonoid_121_, 2);
lean_dec_ref(v_inst_119_);
v___f_122_ = lean_alloc_closure((void*)(lp_mathlib_Submodule_mul___redArg___lam__0___boxed), 3, 1);
lean_closure_set(v___f_122_, 0, v_toAddCommMonoid_121_);
v___x_123_ = lp_mathlib_Submodule_pointwiseAddCommMonoid(lean_box(0), lean_box(0), v_inst_118_, v_toAddCommMonoid_121_, v_inst_120_);
lean_dec_ref(v_toAddCommMonoid_121_);
v___x_124_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_124_, 0, v___x_123_);
lean_ctor_set(v___x_124_, 1, v___f_122_);
return v___x_124_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Submodule_instNonUnitalSemiring___redArg___boxed(lean_object* v_inst_125_, lean_object* v_inst_126_, lean_object* v_inst_127_){
_start:
{
lean_object* v_res_128_; 
v_res_128_ = lp_mathlib_Submodule_instNonUnitalSemiring___redArg(v_inst_125_, v_inst_126_, v_inst_127_);
lean_dec(v_inst_127_);
lean_dec_ref(v_inst_125_);
return v_res_128_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Submodule_instNonUnitalSemiring(lean_object* v_R_129_, lean_object* v_inst_130_, lean_object* v_A_131_, lean_object* v_inst_132_, lean_object* v_inst_133_, lean_object* v_inst_134_){
_start:
{
lean_object* v___x_135_; 
v___x_135_ = lp_mathlib_Submodule_instNonUnitalSemiring___redArg(v_inst_130_, v_inst_132_, v_inst_133_);
return v___x_135_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Submodule_instNonUnitalSemiring___boxed(lean_object* v_R_136_, lean_object* v_inst_137_, lean_object* v_A_138_, lean_object* v_inst_139_, lean_object* v_inst_140_, lean_object* v_inst_141_){
_start:
{
lean_object* v_res_142_; 
v_res_142_ = lp_mathlib_Submodule_instNonUnitalSemiring(v_R_136_, v_inst_137_, v_A_138_, v_inst_139_, v_inst_140_, v_inst_141_);
lean_dec(v_inst_140_);
lean_dec_ref(v_inst_137_);
return v_res_142_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Submodule_instPowNat___redArg___lam__0(lean_object* v_inst_143_, lean_object* v___x_144_, lean_object* v_s_145_, lean_object* v_n_146_){
_start:
{
lean_object* v___x_147_; lean_object* v___x_148_; 
v___x_147_ = lp_mathlib_Submodule_mul___redArg(v_inst_143_);
v___x_148_ = l_npowRec___redArg(v___x_144_, v___x_147_, v_n_146_, v_s_145_);
return v___x_148_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Submodule_instPowNat___redArg___lam__0___boxed(lean_object* v_inst_149_, lean_object* v___x_150_, lean_object* v_s_151_, lean_object* v_n_152_){
_start:
{
lean_object* v_res_153_; 
v_res_153_ = lp_mathlib_Submodule_instPowNat___redArg___lam__0(v_inst_149_, v___x_150_, v_s_151_, v_n_152_);
lean_dec(v_n_152_);
return v_res_153_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Submodule_instPowNat___redArg(lean_object* v_inst_154_){
_start:
{
lean_object* v___x_155_; lean_object* v___f_156_; 
v___x_155_ = lean_box(0);
v___f_156_ = lean_alloc_closure((void*)(lp_mathlib_Submodule_instPowNat___redArg___lam__0___boxed), 4, 2);
lean_closure_set(v___f_156_, 0, v_inst_154_);
lean_closure_set(v___f_156_, 1, v___x_155_);
return v___f_156_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Submodule_instPowNat(lean_object* v_R_157_, lean_object* v_inst_158_, lean_object* v_A_159_, lean_object* v_inst_160_, lean_object* v_inst_161_, lean_object* v_inst_162_){
_start:
{
lean_object* v___x_163_; 
v___x_163_ = lp_mathlib_Submodule_instPowNat___redArg(v_inst_160_);
return v___x_163_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Submodule_instPowNat___boxed(lean_object* v_R_164_, lean_object* v_inst_165_, lean_object* v_A_166_, lean_object* v_inst_167_, lean_object* v_inst_168_, lean_object* v_inst_169_){
_start:
{
lean_object* v_res_170_; 
v_res_170_ = lp_mathlib_Submodule_instPowNat(v_R_164_, v_inst_165_, v_A_166_, v_inst_167_, v_inst_168_, v_inst_169_);
lean_dec(v_inst_168_);
lean_dec_ref(v_inst_165_);
return v_res_170_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Submodule_hasDistribPointwiseNeg(lean_object* v_R_172_, lean_object* v_inst_173_, lean_object* v_A_174_, lean_object* v_inst_175_, lean_object* v_inst_176_){
_start:
{
lean_object* v___f_177_; 
v___f_177_ = ((lean_object*)(lp_mathlib_Submodule_hasDistribPointwiseNeg___closed__0));
return v___f_177_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Submodule_hasDistribPointwiseNeg___boxed(lean_object* v_R_178_, lean_object* v_inst_179_, lean_object* v_A_180_, lean_object* v_inst_181_, lean_object* v_inst_182_){
_start:
{
lean_object* v_res_183_; 
v_res_183_ = lp_mathlib_Submodule_hasDistribPointwiseNeg(v_R_178_, v_inst_179_, v_A_180_, v_inst_181_, v_inst_182_);
lean_dec_ref(v_inst_182_);
lean_dec_ref(v_inst_181_);
lean_dec_ref(v_inst_179_);
return v_res_183_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Submodule_idemSemiring___redArg(lean_object* v_inst_184_, lean_object* v_inst_185_, lean_object* v_inst_186_){
_start:
{
lean_object* v_toAddCommMonoid_187_; lean_object* v_toSMul_188_; lean_object* v___x_189_; lean_object* v___x_190_; lean_object* v_toNatCast_191_; lean_object* v_toOne_192_; lean_object* v___x_194_; uint8_t v_isShared_195_; uint8_t v_isSharedCheck_218_; 
v_toAddCommMonoid_187_ = lean_ctor_get(v_inst_185_, 0);
lean_inc_ref(v_toAddCommMonoid_187_);
v_toSMul_188_ = lean_ctor_get(v_inst_186_, 0);
v___x_189_ = lp_mathlib_Submodule_pointwiseAddCommMonoid(lean_box(0), lean_box(0), v_inst_184_, v_toAddCommMonoid_187_, v_toSMul_188_);
lean_inc_ref(v_inst_185_);
v___x_190_ = lp_mathlib_Submodule_instAddCommMonoidWithOne___redArg(v_inst_184_, v_inst_185_, v_toSMul_188_);
v_toNatCast_191_ = lean_ctor_get(v___x_190_, 0);
v_toOne_192_ = lean_ctor_get(v___x_190_, 2);
v_isSharedCheck_218_ = !lean_is_exclusive(v___x_190_);
if (v_isSharedCheck_218_ == 0)
{
lean_object* v_unused_219_; 
v_unused_219_ = lean_ctor_get(v___x_190_, 1);
lean_dec(v_unused_219_);
v___x_194_ = v___x_190_;
v_isShared_195_ = v_isSharedCheck_218_;
goto v_resetjp_193_;
}
else
{
lean_inc(v_toOne_192_);
lean_inc(v_toNatCast_191_);
lean_dec(v___x_190_);
v___x_194_ = lean_box(0);
v_isShared_195_ = v_isSharedCheck_218_;
goto v_resetjp_193_;
}
v_resetjp_193_:
{
lean_object* v___x_196_; lean_object* v_toMul_197_; lean_object* v___x_198_; lean_object* v___f_199_; lean_object* v___x_201_; 
lean_inc_ref(v_inst_185_);
v___x_196_ = lp_mathlib_Submodule_instNonUnitalSemiring___redArg(v_inst_184_, v_inst_185_, v_toSMul_188_);
v_toMul_197_ = lean_ctor_get(v___x_196_, 1);
lean_inc(v_toMul_197_);
lean_dec_ref(v___x_196_);
v___x_198_ = lp_mathlib_Submodule_instPowNat___redArg(v_inst_185_);
v___f_199_ = lean_alloc_closure((void*)(lp_mathlib_NPow_ofPow___redArg___lam__0), 3, 1);
lean_closure_set(v___f_199_, 0, v___x_198_);
if (v_isShared_195_ == 0)
{
lean_ctor_set(v___x_194_, 2, v___f_199_);
lean_ctor_set(v___x_194_, 1, v_toMul_197_);
lean_ctor_set(v___x_194_, 0, v_toOne_192_);
v___x_201_ = v___x_194_;
goto v_reusejp_200_;
}
else
{
lean_object* v_reuseFailAlloc_217_; 
v_reuseFailAlloc_217_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v_reuseFailAlloc_217_, 0, v_toOne_192_);
lean_ctor_set(v_reuseFailAlloc_217_, 1, v_toMul_197_);
lean_ctor_set(v_reuseFailAlloc_217_, 2, v___f_199_);
v___x_201_ = v_reuseFailAlloc_217_;
goto v_reusejp_200_;
}
v_reusejp_200_:
{
lean_object* v___x_202_; lean_object* v___x_203_; lean_object* v___x_204_; lean_object* v_toLattice_205_; lean_object* v___x_207_; uint8_t v_isShared_208_; uint8_t v_isSharedCheck_214_; 
v___x_202_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_202_, 0, v___x_189_);
lean_ctor_set(v___x_202_, 1, v___x_201_);
lean_ctor_set(v___x_202_, 2, v_toNatCast_191_);
v___x_203_ = lp_mathlib_Submodule_completeLattice___redArg(v_inst_184_, v_toAddCommMonoid_187_, v_toSMul_188_);
lean_dec_ref(v_toAddCommMonoid_187_);
v___x_204_ = lp_mathlib_CompleteLattice_toConditionallyCompleteLattice___redArg(v___x_203_);
v_toLattice_205_ = lean_ctor_get(v___x_204_, 0);
v_isSharedCheck_214_ = !lean_is_exclusive(v___x_204_);
if (v_isSharedCheck_214_ == 0)
{
lean_object* v_unused_215_; lean_object* v_unused_216_; 
v_unused_215_ = lean_ctor_get(v___x_204_, 2);
lean_dec(v_unused_215_);
v_unused_216_ = lean_ctor_get(v___x_204_, 1);
lean_dec(v_unused_216_);
v___x_207_ = v___x_204_;
v_isShared_208_ = v_isSharedCheck_214_;
goto v_resetjp_206_;
}
else
{
lean_inc(v_toLattice_205_);
lean_dec(v___x_204_);
v___x_207_ = lean_box(0);
v_isShared_208_ = v_isSharedCheck_214_;
goto v_resetjp_206_;
}
v_resetjp_206_:
{
lean_object* v_toSemilatticeSup_209_; lean_object* v___x_210_; lean_object* v___x_212_; 
v_toSemilatticeSup_209_ = lean_ctor_get(v_toLattice_205_, 0);
lean_inc_ref(v_toSemilatticeSup_209_);
lean_dec_ref(v_toLattice_205_);
v___x_210_ = lean_box(0);
if (v_isShared_208_ == 0)
{
lean_ctor_set(v___x_207_, 2, v___x_210_);
lean_ctor_set(v___x_207_, 1, v_toSemilatticeSup_209_);
lean_ctor_set(v___x_207_, 0, v___x_202_);
v___x_212_ = v___x_207_;
goto v_reusejp_211_;
}
else
{
lean_object* v_reuseFailAlloc_213_; 
v_reuseFailAlloc_213_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v_reuseFailAlloc_213_, 0, v___x_202_);
lean_ctor_set(v_reuseFailAlloc_213_, 1, v_toSemilatticeSup_209_);
lean_ctor_set(v_reuseFailAlloc_213_, 2, v___x_210_);
v___x_212_ = v_reuseFailAlloc_213_;
goto v_reusejp_211_;
}
v_reusejp_211_:
{
return v___x_212_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Submodule_idemSemiring___redArg___boxed(lean_object* v_inst_220_, lean_object* v_inst_221_, lean_object* v_inst_222_){
_start:
{
lean_object* v_res_223_; 
v_res_223_ = lp_mathlib_Submodule_idemSemiring___redArg(v_inst_220_, v_inst_221_, v_inst_222_);
lean_dec_ref(v_inst_222_);
lean_dec_ref(v_inst_220_);
return v_res_223_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Submodule_idemSemiring(lean_object* v_R_224_, lean_object* v_inst_225_, lean_object* v_A_226_, lean_object* v_inst_227_, lean_object* v_inst_228_){
_start:
{
lean_object* v___x_229_; 
v___x_229_ = lp_mathlib_Submodule_idemSemiring___redArg(v_inst_225_, v_inst_227_, v_inst_228_);
return v___x_229_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Submodule_idemSemiring___boxed(lean_object* v_R_230_, lean_object* v_inst_231_, lean_object* v_A_232_, lean_object* v_inst_233_, lean_object* v_inst_234_){
_start:
{
lean_object* v_res_235_; 
v_res_235_ = lp_mathlib_Submodule_idemSemiring(v_R_230_, v_inst_231_, v_A_232_, v_inst_233_, v_inst_234_);
lean_dec_ref(v_inst_234_);
lean_dec_ref(v_inst_231_);
return v_res_235_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Submodule_mapHom___redArg(lean_object* v_inst_237_, lean_object* v_inst_238_, lean_object* v_inst_239_, lean_object* v_inst_240_, lean_object* v_inst_241_, lean_object* v_f_242_){
_start:
{
lean_object* v_toAddCommMonoid_243_; lean_object* v_toAddCommMonoid_244_; lean_object* v_toSMul_245_; lean_object* v_toSMul_246_; lean_object* v___f_247_; lean_object* v___f_248_; lean_object* v___x_249_; 
v_toAddCommMonoid_243_ = lean_ctor_get(v_inst_238_, 0);
lean_inc_ref(v_toAddCommMonoid_243_);
lean_dec_ref(v_inst_238_);
v_toAddCommMonoid_244_ = lean_ctor_get(v_inst_240_, 0);
lean_inc_ref(v_toAddCommMonoid_244_);
lean_dec_ref(v_inst_240_);
v_toSMul_245_ = lean_ctor_get(v_inst_239_, 0);
lean_inc(v_toSMul_245_);
lean_dec_ref(v_inst_239_);
v_toSMul_246_ = lean_ctor_get(v_inst_241_, 0);
lean_inc(v_toSMul_246_);
lean_dec_ref(v_inst_241_);
v___f_247_ = ((lean_object*)(lp_mathlib_Submodule_mapHom___redArg___closed__0));
v___f_248_ = lean_alloc_closure((void*)(lp_mathlib_AlgHom_toLinearMap___redArg___lam__0), 2, 1);
lean_closure_set(v___f_248_, 0, v_f_242_);
lean_inc_ref(v_inst_237_);
v___x_249_ = lean_alloc_closure((void*)(lp_mathlib_Submodule_map___boxed), 14, 13);
lean_closure_set(v___x_249_, 0, lean_box(0));
lean_closure_set(v___x_249_, 1, lean_box(0));
lean_closure_set(v___x_249_, 2, lean_box(0));
lean_closure_set(v___x_249_, 3, lean_box(0));
lean_closure_set(v___x_249_, 4, v_inst_237_);
lean_closure_set(v___x_249_, 5, v_inst_237_);
lean_closure_set(v___x_249_, 6, v_toAddCommMonoid_243_);
lean_closure_set(v___x_249_, 7, v_toAddCommMonoid_244_);
lean_closure_set(v___x_249_, 8, v_toSMul_245_);
lean_closure_set(v___x_249_, 9, v_toSMul_246_);
lean_closure_set(v___x_249_, 10, v___f_247_);
lean_closure_set(v___x_249_, 11, lean_box(0));
lean_closure_set(v___x_249_, 12, v___f_248_);
return v___x_249_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Submodule_mapHom(lean_object* v_R_250_, lean_object* v_inst_251_, lean_object* v_A_252_, lean_object* v_inst_253_, lean_object* v_inst_254_, lean_object* v_A_x27_255_, lean_object* v_inst_256_, lean_object* v_inst_257_, lean_object* v_f_258_){
_start:
{
lean_object* v___x_259_; 
v___x_259_ = lp_mathlib_Submodule_mapHom___redArg(v_inst_251_, v_inst_253_, v_inst_254_, v_inst_256_, v_inst_257_, v_f_258_);
return v___x_259_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Submodule_equivOpposite___lam__0(lean_object* v_p_260_){
_start:
{
lean_object* v___x_261_; 
v___x_261_ = lean_box(0);
return v___x_261_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Submodule_equivOpposite(lean_object* v_R_265_, lean_object* v_inst_266_, lean_object* v_A_267_, lean_object* v_inst_268_, lean_object* v_inst_269_){
_start:
{
lean_object* v___x_270_; 
v___x_270_ = ((lean_object*)(lp_mathlib_Submodule_equivOpposite___closed__1));
return v___x_270_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Submodule_equivOpposite___boxed(lean_object* v_R_271_, lean_object* v_inst_272_, lean_object* v_A_273_, lean_object* v_inst_274_, lean_object* v_inst_275_){
_start:
{
lean_object* v_res_276_; 
v_res_276_ = lp_mathlib_Submodule_equivOpposite(v_R_271_, v_inst_272_, v_A_273_, v_inst_274_, v_inst_275_);
lean_dec_ref(v_inst_275_);
lean_dec_ref(v_inst_274_);
lean_dec_ref(v_inst_272_);
return v_res_276_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Submodule_pointwiseMulSemiringAction(lean_object* v_R_278_, lean_object* v_inst_279_, lean_object* v_A_280_, lean_object* v_inst_281_, lean_object* v_inst_282_, lean_object* v_00_u03b1_283_, lean_object* v_inst_284_, lean_object* v_inst_285_, lean_object* v_inst_286_){
_start:
{
lean_object* v___f_287_; 
v___f_287_ = ((lean_object*)(lp_mathlib_Submodule_pointwiseMulSemiringAction___closed__0));
return v___f_287_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Submodule_pointwiseMulSemiringAction___boxed(lean_object* v_R_288_, lean_object* v_inst_289_, lean_object* v_A_290_, lean_object* v_inst_291_, lean_object* v_inst_292_, lean_object* v_00_u03b1_293_, lean_object* v_inst_294_, lean_object* v_inst_295_, lean_object* v_inst_296_){
_start:
{
lean_object* v_res_297_; 
v_res_297_ = lp_mathlib_Submodule_pointwiseMulSemiringAction(v_R_288_, v_inst_289_, v_A_290_, v_inst_291_, v_inst_292_, v_00_u03b1_293_, v_inst_294_, v_inst_295_, v_inst_296_);
lean_dec(v_inst_295_);
lean_dec_ref(v_inst_294_);
lean_dec_ref(v_inst_292_);
lean_dec_ref(v_inst_291_);
lean_dec_ref(v_inst_289_);
return v_res_297_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Submodule_instIdemCommSemiring___redArg(lean_object* v_inst_298_, lean_object* v_inst_299_, lean_object* v_inst_300_){
_start:
{
lean_object* v___x_301_; lean_object* v_toSemiring_302_; lean_object* v_toSemilatticeSup_303_; lean_object* v_toOrderBot_304_; lean_object* v___x_306_; uint8_t v_isShared_307_; uint8_t v_isSharedCheck_311_; 
v___x_301_ = lp_mathlib_Submodule_idemSemiring___redArg(v_inst_298_, v_inst_299_, v_inst_300_);
v_toSemiring_302_ = lean_ctor_get(v___x_301_, 0);
v_toSemilatticeSup_303_ = lean_ctor_get(v___x_301_, 1);
v_toOrderBot_304_ = lean_ctor_get(v___x_301_, 2);
v_isSharedCheck_311_ = !lean_is_exclusive(v___x_301_);
if (v_isSharedCheck_311_ == 0)
{
v___x_306_ = v___x_301_;
v_isShared_307_ = v_isSharedCheck_311_;
goto v_resetjp_305_;
}
else
{
lean_inc(v_toOrderBot_304_);
lean_inc(v_toSemilatticeSup_303_);
lean_inc(v_toSemiring_302_);
lean_dec(v___x_301_);
v___x_306_ = lean_box(0);
v_isShared_307_ = v_isSharedCheck_311_;
goto v_resetjp_305_;
}
v_resetjp_305_:
{
lean_object* v___x_309_; 
if (v_isShared_307_ == 0)
{
v___x_309_ = v___x_306_;
goto v_reusejp_308_;
}
else
{
lean_object* v_reuseFailAlloc_310_; 
v_reuseFailAlloc_310_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v_reuseFailAlloc_310_, 0, v_toSemiring_302_);
lean_ctor_set(v_reuseFailAlloc_310_, 1, v_toSemilatticeSup_303_);
lean_ctor_set(v_reuseFailAlloc_310_, 2, v_toOrderBot_304_);
v___x_309_ = v_reuseFailAlloc_310_;
goto v_reusejp_308_;
}
v_reusejp_308_:
{
return v___x_309_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Submodule_instIdemCommSemiring___redArg___boxed(lean_object* v_inst_312_, lean_object* v_inst_313_, lean_object* v_inst_314_){
_start:
{
lean_object* v_res_315_; 
v_res_315_ = lp_mathlib_Submodule_instIdemCommSemiring___redArg(v_inst_312_, v_inst_313_, v_inst_314_);
lean_dec_ref(v_inst_314_);
lean_dec_ref(v_inst_312_);
return v_res_315_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Submodule_instIdemCommSemiring(lean_object* v_R_316_, lean_object* v_inst_317_, lean_object* v_A_318_, lean_object* v_inst_319_, lean_object* v_inst_320_){
_start:
{
lean_object* v___x_321_; 
v___x_321_ = lp_mathlib_Submodule_instIdemCommSemiring___redArg(v_inst_317_, v_inst_319_, v_inst_320_);
return v___x_321_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Submodule_instIdemCommSemiring___boxed(lean_object* v_R_322_, lean_object* v_inst_323_, lean_object* v_A_324_, lean_object* v_inst_325_, lean_object* v_inst_326_){
_start:
{
lean_object* v_res_327_; 
v_res_327_ = lp_mathlib_Submodule_instIdemCommSemiring(v_R_322_, v_inst_323_, v_A_324_, v_inst_325_, v_inst_326_);
lean_dec_ref(v_inst_326_);
lean_dec_ref(v_inst_323_);
return v_res_327_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Submodule_instDiv___lam__0(lean_object* v_I_328_, lean_object* v_J_329_){
_start:
{
lean_object* v___x_330_; 
v___x_330_ = lean_box(0);
return v___x_330_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Submodule_instDiv(lean_object* v_R_332_, lean_object* v_inst_333_, lean_object* v_A_334_, lean_object* v_inst_335_, lean_object* v_inst_336_){
_start:
{
lean_object* v___f_337_; 
v___f_337_ = ((lean_object*)(lp_mathlib_Submodule_instDiv___closed__0));
return v___f_337_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Submodule_instDiv___boxed(lean_object* v_R_338_, lean_object* v_inst_339_, lean_object* v_A_340_, lean_object* v_inst_341_, lean_object* v_inst_342_){
_start:
{
lean_object* v_res_343_; 
v_res_343_ = lp_mathlib_Submodule_instDiv(v_R_338_, v_inst_339_, v_A_340_, v_inst_341_, v_inst_342_);
lean_dec_ref(v_inst_342_);
lean_dec_ref(v_inst_341_);
lean_dec_ref(v_inst_339_);
return v_res_343_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Algebra_Bilinear(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Algebra_Opposite(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Group_Pointwise_Finset_Basic(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Group_Pointwise_Set_BigOperators(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Module_Submodule_Finsupp(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Ring_NonZeroDivisors(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Ring_Submonoid_Pointwise(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Data_Set_Semiring(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_GroupTheory_GroupAction_SubMulAction_Pointwise(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Algebra_Operations(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Algebra_Bilinear(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Algebra_Opposite(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Group_Pointwise_Finset_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Group_Pointwise_Set_BigOperators(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Module_Submodule_Finsupp(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Ring_NonZeroDivisors(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Ring_Submonoid_Pointwise(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Data_Set_Semiring(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_GroupTheory_GroupAction_SubMulAction_Pointwise(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_Algebra_Algebra_Operations(uint8_t builtin) {
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
lean_object* initialize_mathlib_Mathlib_Algebra_Algebra_Bilinear(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Algebra_Opposite(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Group_Pointwise_Finset_Basic(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Group_Pointwise_Set_BigOperators(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Module_Submodule_Finsupp(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Ring_NonZeroDivisors(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Ring_Submonoid_Pointwise(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Data_Set_Semiring(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_GroupTheory_GroupAction_SubMulAction_Pointwise(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Algebra_Algebra_Operations(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Algebra_Bilinear(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Algebra_Opposite(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Group_Pointwise_Finset_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Group_Pointwise_Set_BigOperators(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Module_Submodule_Finsupp(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Ring_NonZeroDivisors(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Ring_Submonoid_Pointwise(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_Set_Semiring(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_GroupTheory_GroupAction_SubMulAction_Pointwise(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Algebra_Operations(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_Algebra_Algebra_Operations(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_Algebra_Algebra_Operations(builtin);
}
#ifdef __cplusplus
}
#endif
