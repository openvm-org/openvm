// Lean compiler output
// Module: Mathlib.Algebra.Ring.Submonoid.Pointwise
// Imports: public import Init public meta import Init public import Mathlib.Algebra.Group.Submonoid.Pointwise public import Mathlib.Algebra.Module.Defs public import Mathlib.Data.Nat.Cast.Basic
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
lean_object* lp_mathlib_AddSubmonoid_neg___lam__0(lean_object*);
lean_object* lp_mathlib_AddMonoid_toAddZeroClass___redArg(lean_object*);
lean_object* lp_mathlib_AddSubmonoid_instCompleteLattice___redArg(lean_object*);
lean_object* lp_mathlib_CompleteLattice_toConditionallyCompleteLattice___redArg(lean_object*);
lean_object* lp_mathlib_ConditionallyCompleteLattice_toConditionallyCompletePartialOrder___redArg(lean_object*);
lean_object* lp_mathlib_Semiring_toNonAssocSemiring___redArg(lean_object*);
lean_object* lp_mathlib_npowBinRecAuto___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddSubmonoid_one(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddSubmonoid_one___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddSubmonoid_smul___redArg___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddSubmonoid_smul___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddSubmonoid_smul___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddSubmonoid_smul(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddSubmonoid_smul___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddSubmonoid_mul___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddSubmonoid_mul___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddSubmonoid_mul(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddSubmonoid_mul___boxed(lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_AddSubmonoid_hasDistribNeg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_AddSubmonoid_neg___lam__0, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_AddSubmonoid_hasDistribNeg___closed__0 = (const lean_object*)&lp_mathlib_AddSubmonoid_hasDistribNeg___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_AddSubmonoid_hasDistribNeg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddSubmonoid_hasDistribNeg___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddSubmonoid_mulOneClass___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddSubmonoid_mulOneClass___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddSubmonoid_mulOneClass(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddSubmonoid_mulOneClass___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddSubmonoid_semigroup___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddSubmonoid_semigroup___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddSubmonoid_semigroup(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddSubmonoid_semigroup___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddSubmonoid_monoid___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddSubmonoid_monoid(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddSubmonoid_one(lean_object* v_R_1_, lean_object* v_inst_2_){
_start:
{
lean_object* v___x_3_; 
v___x_3_ = lean_box(0);
return v___x_3_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddSubmonoid_one___boxed(lean_object* v_R_4_, lean_object* v_inst_5_){
_start:
{
lean_object* v_res_6_; 
v_res_6_ = lp_mathlib_AddSubmonoid_one(v_R_4_, v_inst_5_);
lean_dec_ref(v_inst_5_);
return v_res_6_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddSubmonoid_smul___redArg___lam__0(lean_object* v_toSupSet_7_, lean_object* v_M_8_, lean_object* v_N_9_){
_start:
{
lean_object* v___x_10_; 
v___x_10_ = lean_apply_1(v_toSupSet_7_, lean_box(0));
return v___x_10_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddSubmonoid_smul___redArg(lean_object* v_inst_11_){
_start:
{
lean_object* v___x_12_; lean_object* v___x_13_; lean_object* v___x_14_; lean_object* v___x_15_; lean_object* v_toConditionallyCompletePartialOrderSup_16_; lean_object* v_toSupSet_17_; lean_object* v___f_18_; 
v___x_12_ = lp_mathlib_AddMonoid_toAddZeroClass___redArg(v_inst_11_);
v___x_13_ = lp_mathlib_AddSubmonoid_instCompleteLattice___redArg(v___x_12_);
lean_dec_ref(v___x_12_);
v___x_14_ = lp_mathlib_CompleteLattice_toConditionallyCompleteLattice___redArg(v___x_13_);
v___x_15_ = lp_mathlib_ConditionallyCompleteLattice_toConditionallyCompletePartialOrder___redArg(v___x_14_);
v_toConditionallyCompletePartialOrderSup_16_ = lean_ctor_get(v___x_15_, 0);
lean_inc_ref(v_toConditionallyCompletePartialOrderSup_16_);
lean_dec_ref(v___x_15_);
v_toSupSet_17_ = lean_ctor_get(v_toConditionallyCompletePartialOrderSup_16_, 1);
lean_inc(v_toSupSet_17_);
lean_dec_ref(v_toConditionallyCompletePartialOrderSup_16_);
v___f_18_ = lean_alloc_closure((void*)(lp_mathlib_AddSubmonoid_smul___redArg___lam__0), 3, 1);
lean_closure_set(v___f_18_, 0, v_toSupSet_17_);
return v___f_18_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddSubmonoid_smul___redArg___boxed(lean_object* v_inst_19_){
_start:
{
lean_object* v_res_20_; 
v_res_20_ = lp_mathlib_AddSubmonoid_smul___redArg(v_inst_19_);
lean_dec_ref(v_inst_19_);
return v_res_20_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddSubmonoid_smul(lean_object* v_R_21_, lean_object* v_A_22_, lean_object* v_inst_23_, lean_object* v_inst_24_, lean_object* v_inst_25_){
_start:
{
lean_object* v___x_26_; 
v___x_26_ = lp_mathlib_AddSubmonoid_smul___redArg(v_inst_24_);
return v___x_26_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddSubmonoid_smul___boxed(lean_object* v_R_27_, lean_object* v_A_28_, lean_object* v_inst_29_, lean_object* v_inst_30_, lean_object* v_inst_31_){
_start:
{
lean_object* v_res_32_; 
v_res_32_ = lp_mathlib_AddSubmonoid_smul(v_R_27_, v_A_28_, v_inst_29_, v_inst_30_, v_inst_31_);
lean_dec(v_inst_31_);
lean_dec_ref(v_inst_30_);
lean_dec_ref(v_inst_29_);
return v_res_32_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddSubmonoid_mul___redArg(lean_object* v_inst_33_){
_start:
{
lean_object* v_toAddCommMonoid_34_; lean_object* v___x_35_; lean_object* v___x_36_; lean_object* v___x_37_; lean_object* v___x_38_; lean_object* v_toConditionallyCompletePartialOrderSup_39_; lean_object* v_toSupSet_40_; lean_object* v___f_41_; 
v_toAddCommMonoid_34_ = lean_ctor_get(v_inst_33_, 0);
v___x_35_ = lp_mathlib_AddMonoid_toAddZeroClass___redArg(v_toAddCommMonoid_34_);
v___x_36_ = lp_mathlib_AddSubmonoid_instCompleteLattice___redArg(v___x_35_);
lean_dec_ref(v___x_35_);
v___x_37_ = lp_mathlib_CompleteLattice_toConditionallyCompleteLattice___redArg(v___x_36_);
v___x_38_ = lp_mathlib_ConditionallyCompleteLattice_toConditionallyCompletePartialOrder___redArg(v___x_37_);
v_toConditionallyCompletePartialOrderSup_39_ = lean_ctor_get(v___x_38_, 0);
lean_inc_ref(v_toConditionallyCompletePartialOrderSup_39_);
lean_dec_ref(v___x_38_);
v_toSupSet_40_ = lean_ctor_get(v_toConditionallyCompletePartialOrderSup_39_, 1);
lean_inc(v_toSupSet_40_);
lean_dec_ref(v_toConditionallyCompletePartialOrderSup_39_);
v___f_41_ = lean_alloc_closure((void*)(lp_mathlib_AddSubmonoid_smul___redArg___lam__0), 3, 1);
lean_closure_set(v___f_41_, 0, v_toSupSet_40_);
return v___f_41_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddSubmonoid_mul___redArg___boxed(lean_object* v_inst_42_){
_start:
{
lean_object* v_res_43_; 
v_res_43_ = lp_mathlib_AddSubmonoid_mul___redArg(v_inst_42_);
lean_dec_ref(v_inst_42_);
return v_res_43_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddSubmonoid_mul(lean_object* v_R_44_, lean_object* v_inst_45_){
_start:
{
lean_object* v___x_46_; 
v___x_46_ = lp_mathlib_AddSubmonoid_mul___redArg(v_inst_45_);
return v___x_46_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddSubmonoid_mul___boxed(lean_object* v_R_47_, lean_object* v_inst_48_){
_start:
{
lean_object* v_res_49_; 
v_res_49_ = lp_mathlib_AddSubmonoid_mul(v_R_47_, v_inst_48_);
lean_dec_ref(v_inst_48_);
return v_res_49_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddSubmonoid_hasDistribNeg(lean_object* v_R_51_, lean_object* v_inst_52_){
_start:
{
lean_object* v___f_53_; 
v___f_53_ = ((lean_object*)(lp_mathlib_AddSubmonoid_hasDistribNeg___closed__0));
return v___f_53_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddSubmonoid_hasDistribNeg___boxed(lean_object* v_R_54_, lean_object* v_inst_55_){
_start:
{
lean_object* v_res_56_; 
v_res_56_ = lp_mathlib_AddSubmonoid_hasDistribNeg(v_R_54_, v_inst_55_);
lean_dec_ref(v_inst_55_);
return v_res_56_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddSubmonoid_mulOneClass___redArg(lean_object* v_inst_57_){
_start:
{
lean_object* v_toNonUnitalNonAssocSemiring_58_; lean_object* v___x_59_; lean_object* v___x_60_; lean_object* v___x_61_; 
v_toNonUnitalNonAssocSemiring_58_ = lean_ctor_get(v_inst_57_, 0);
v___x_59_ = lean_box(0);
v___x_60_ = lp_mathlib_AddSubmonoid_mul___redArg(v_toNonUnitalNonAssocSemiring_58_);
v___x_61_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_61_, 0, v___x_59_);
lean_ctor_set(v___x_61_, 1, v___x_60_);
return v___x_61_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddSubmonoid_mulOneClass___redArg___boxed(lean_object* v_inst_62_){
_start:
{
lean_object* v_res_63_; 
v_res_63_ = lp_mathlib_AddSubmonoid_mulOneClass___redArg(v_inst_62_);
lean_dec_ref(v_inst_62_);
return v_res_63_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddSubmonoid_mulOneClass(lean_object* v_R_64_, lean_object* v_inst_65_){
_start:
{
lean_object* v___x_66_; 
v___x_66_ = lp_mathlib_AddSubmonoid_mulOneClass___redArg(v_inst_65_);
return v___x_66_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddSubmonoid_mulOneClass___boxed(lean_object* v_R_67_, lean_object* v_inst_68_){
_start:
{
lean_object* v_res_69_; 
v_res_69_ = lp_mathlib_AddSubmonoid_mulOneClass(v_R_67_, v_inst_68_);
lean_dec_ref(v_inst_68_);
return v_res_69_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddSubmonoid_semigroup___redArg(lean_object* v_inst_70_){
_start:
{
lean_object* v___x_71_; 
v___x_71_ = lp_mathlib_AddSubmonoid_mul___redArg(v_inst_70_);
return v___x_71_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddSubmonoid_semigroup___redArg___boxed(lean_object* v_inst_72_){
_start:
{
lean_object* v_res_73_; 
v_res_73_ = lp_mathlib_AddSubmonoid_semigroup___redArg(v_inst_72_);
lean_dec_ref(v_inst_72_);
return v_res_73_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddSubmonoid_semigroup(lean_object* v_R_74_, lean_object* v_inst_75_){
_start:
{
lean_object* v___x_76_; 
v___x_76_ = lp_mathlib_AddSubmonoid_mul___redArg(v_inst_75_);
return v___x_76_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddSubmonoid_semigroup___boxed(lean_object* v_R_77_, lean_object* v_inst_78_){
_start:
{
lean_object* v_res_79_; 
v_res_79_ = lp_mathlib_AddSubmonoid_semigroup(v_R_77_, v_inst_78_);
lean_dec_ref(v_inst_78_);
return v_res_79_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddSubmonoid_monoid___redArg(lean_object* v_inst_80_){
_start:
{
lean_object* v___x_81_; lean_object* v_toNonUnitalNonAssocSemiring_82_; lean_object* v___x_84_; uint8_t v_isShared_85_; uint8_t v_isSharedCheck_92_; 
v___x_81_ = lp_mathlib_Semiring_toNonAssocSemiring___redArg(v_inst_80_);
v_toNonUnitalNonAssocSemiring_82_ = lean_ctor_get(v___x_81_, 0);
v_isSharedCheck_92_ = !lean_is_exclusive(v___x_81_);
if (v_isSharedCheck_92_ == 0)
{
lean_object* v_unused_93_; lean_object* v_unused_94_; 
v_unused_93_ = lean_ctor_get(v___x_81_, 2);
lean_dec(v_unused_93_);
v_unused_94_ = lean_ctor_get(v___x_81_, 1);
lean_dec(v_unused_94_);
v___x_84_ = v___x_81_;
v_isShared_85_ = v_isSharedCheck_92_;
goto v_resetjp_83_;
}
else
{
lean_inc(v_toNonUnitalNonAssocSemiring_82_);
lean_dec(v___x_81_);
v___x_84_ = lean_box(0);
v_isShared_85_ = v_isSharedCheck_92_;
goto v_resetjp_83_;
}
v_resetjp_83_:
{
lean_object* v___x_86_; lean_object* v___x_87_; lean_object* v___x_88_; lean_object* v___x_90_; 
v___x_86_ = lean_box(0);
v___x_87_ = lp_mathlib_AddSubmonoid_mul___redArg(v_toNonUnitalNonAssocSemiring_82_);
lean_dec_ref(v_toNonUnitalNonAssocSemiring_82_);
lean_inc_ref(v___x_87_);
v___x_88_ = lean_alloc_closure((void*)(lp_mathlib_npowBinRecAuto___boxed), 5, 3);
lean_closure_set(v___x_88_, 0, lean_box(0));
lean_closure_set(v___x_88_, 1, v___x_87_);
lean_closure_set(v___x_88_, 2, v___x_86_);
if (v_isShared_85_ == 0)
{
lean_ctor_set(v___x_84_, 2, v___x_88_);
lean_ctor_set(v___x_84_, 1, v___x_87_);
lean_ctor_set(v___x_84_, 0, v___x_86_);
v___x_90_ = v___x_84_;
goto v_reusejp_89_;
}
else
{
lean_object* v_reuseFailAlloc_91_; 
v_reuseFailAlloc_91_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v_reuseFailAlloc_91_, 0, v___x_86_);
lean_ctor_set(v_reuseFailAlloc_91_, 1, v___x_87_);
lean_ctor_set(v_reuseFailAlloc_91_, 2, v___x_88_);
v___x_90_ = v_reuseFailAlloc_91_;
goto v_reusejp_89_;
}
v_reusejp_89_:
{
return v___x_90_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddSubmonoid_monoid(lean_object* v_R_95_, lean_object* v_inst_96_){
_start:
{
lean_object* v___x_97_; 
v___x_97_ = lp_mathlib_AddSubmonoid_monoid___redArg(v_inst_96_);
return v___x_97_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Group_Submonoid_Pointwise(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Module_Defs(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Data_Nat_Cast_Basic(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Ring_Submonoid_Pointwise(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Group_Submonoid_Pointwise(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Module_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Data_Nat_Cast_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_Algebra_Ring_Submonoid_Pointwise(uint8_t builtin) {
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
lean_object* initialize_mathlib_Mathlib_Algebra_Group_Submonoid_Pointwise(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Module_Defs(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Data_Nat_Cast_Basic(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Algebra_Ring_Submonoid_Pointwise(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Group_Submonoid_Pointwise(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Module_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_Nat_Cast_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Ring_Submonoid_Pointwise(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_Algebra_Ring_Submonoid_Pointwise(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_Algebra_Ring_Submonoid_Pointwise(builtin);
}
#ifdef __cplusplus
}
#endif
