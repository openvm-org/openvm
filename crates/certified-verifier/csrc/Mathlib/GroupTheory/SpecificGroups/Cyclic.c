// Lean compiler output
// Module: Mathlib.GroupTheory.SpecificGroups.Cyclic
// Imports: public import Init public meta import Init public import Mathlib.Algebra.Group.Subgroup.ZPowers.Lemmas public import Mathlib.Algebra.Group.TypeTags.Finite public import Mathlib.Algebra.Order.Hom.TypeTags public import Mathlib.Data.Nat.Totient public import Mathlib.Data.ZMod.Aut public import Mathlib.GroupTheory.Exponent public import Mathlib.GroupTheory.SpecificGroups.Cyclic.Basic public import Mathlib.GroupTheory.Subgroup.Simple public import Mathlib.Tactic.Group public import Mathlib.Tactic.IntervalCases
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
LEAN_EXPORT lean_object* lp_mathlib_commGroupOfCyclicCenterQuotient___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_commGroupOfCyclicCenterQuotient___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_commGroupOfCyclicCenterQuotient(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_commGroupOfCyclicCenterQuotient___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_addCommGroupOfAddCyclicCenterQuotient___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_addCommGroupOfAddCyclicCenterQuotient___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_addCommGroupOfAddCyclicCenterQuotient(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_addCommGroupOfAddCyclicCenterQuotient___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_commGroupOfCyclicCenterQuotient___redArg(lean_object* v_inst_1_){
_start:
{
lean_inc_ref(v_inst_1_);
return v_inst_1_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_commGroupOfCyclicCenterQuotient___redArg___boxed(lean_object* v_inst_2_){
_start:
{
lean_object* v_res_3_; 
v_res_3_ = lp_mathlib_commGroupOfCyclicCenterQuotient___redArg(v_inst_2_);
lean_dec_ref(v_inst_2_);
return v_res_3_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_commGroupOfCyclicCenterQuotient(lean_object* v_G_4_, lean_object* v_G_x27_5_, lean_object* v_inst_6_, lean_object* v_inst_7_, lean_object* v_inst_8_, lean_object* v_f_9_, lean_object* v_hf_10_){
_start:
{
lean_inc_ref(v_inst_6_);
return v_inst_6_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_commGroupOfCyclicCenterQuotient___boxed(lean_object* v_G_11_, lean_object* v_G_x27_12_, lean_object* v_inst_13_, lean_object* v_inst_14_, lean_object* v_inst_15_, lean_object* v_f_16_, lean_object* v_hf_17_){
_start:
{
lean_object* v_res_18_; 
v_res_18_ = lp_mathlib_commGroupOfCyclicCenterQuotient(v_G_11_, v_G_x27_12_, v_inst_13_, v_inst_14_, v_inst_15_, v_f_16_, v_hf_17_);
lean_dec(v_f_16_);
lean_dec_ref(v_inst_14_);
lean_dec_ref(v_inst_13_);
return v_res_18_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_addCommGroupOfAddCyclicCenterQuotient___redArg(lean_object* v_inst_19_){
_start:
{
lean_inc_ref(v_inst_19_);
return v_inst_19_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_addCommGroupOfAddCyclicCenterQuotient___redArg___boxed(lean_object* v_inst_20_){
_start:
{
lean_object* v_res_21_; 
v_res_21_ = lp_mathlib_addCommGroupOfAddCyclicCenterQuotient___redArg(v_inst_20_);
lean_dec_ref(v_inst_20_);
return v_res_21_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_addCommGroupOfAddCyclicCenterQuotient(lean_object* v_G_22_, lean_object* v_G_x27_23_, lean_object* v_inst_24_, lean_object* v_inst_25_, lean_object* v_inst_26_, lean_object* v_f_27_, lean_object* v_hf_28_){
_start:
{
lean_inc_ref(v_inst_24_);
return v_inst_24_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_addCommGroupOfAddCyclicCenterQuotient___boxed(lean_object* v_G_29_, lean_object* v_G_x27_30_, lean_object* v_inst_31_, lean_object* v_inst_32_, lean_object* v_inst_33_, lean_object* v_f_34_, lean_object* v_hf_35_){
_start:
{
lean_object* v_res_36_; 
v_res_36_ = lp_mathlib_addCommGroupOfAddCyclicCenterQuotient(v_G_29_, v_G_x27_30_, v_inst_31_, v_inst_32_, v_inst_33_, v_f_34_, v_hf_35_);
lean_dec(v_f_34_);
lean_dec_ref(v_inst_32_);
lean_dec_ref(v_inst_31_);
return v_res_36_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Group_Subgroup_ZPowers_Lemmas(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Group_TypeTags_Finite(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Order_Hom_TypeTags(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Data_Nat_Totient(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Data_ZMod_Aut(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_GroupTheory_Exponent(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_GroupTheory_SpecificGroups_Cyclic_Basic(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_GroupTheory_Subgroup_Simple(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Tactic_Group(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Tactic_IntervalCases(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_GroupTheory_SpecificGroups_Cyclic(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Group_Subgroup_ZPowers_Lemmas(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Group_TypeTags_Finite(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Order_Hom_TypeTags(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Data_Nat_Totient(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Data_ZMod_Aut(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_GroupTheory_Exponent(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_GroupTheory_SpecificGroups_Cyclic_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_GroupTheory_Subgroup_Simple(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Tactic_Group(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Tactic_IntervalCases(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_GroupTheory_SpecificGroups_Cyclic(uint8_t builtin) {
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
lean_object* initialize_mathlib_Mathlib_Algebra_Group_Subgroup_ZPowers_Lemmas(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Group_TypeTags_Finite(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Order_Hom_TypeTags(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Data_Nat_Totient(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Data_ZMod_Aut(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_GroupTheory_Exponent(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_GroupTheory_SpecificGroups_Cyclic_Basic(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_GroupTheory_Subgroup_Simple(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Tactic_Group(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Tactic_IntervalCases(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_GroupTheory_SpecificGroups_Cyclic(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Group_Subgroup_ZPowers_Lemmas(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Group_TypeTags_Finite(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Order_Hom_TypeTags(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_Nat_Totient(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_ZMod_Aut(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_GroupTheory_Exponent(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_GroupTheory_SpecificGroups_Cyclic_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_GroupTheory_Subgroup_Simple(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Tactic_Group(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Tactic_IntervalCases(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_GroupTheory_SpecificGroups_Cyclic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_GroupTheory_SpecificGroups_Cyclic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_GroupTheory_SpecificGroups_Cyclic(builtin);
}
#ifdef __cplusplus
}
#endif
