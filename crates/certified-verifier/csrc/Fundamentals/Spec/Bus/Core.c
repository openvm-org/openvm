// Lean compiler output
// Module: Fundamentals.Spec.Bus.Core
// Imports: public import Init public meta import Init public import Mathlib.Algebra.Field.Defs public import Mathlib.Algebra.BigOperators.Group.Finset.Basic
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
lean_object* lp_mathlib_Field_toSemifield___redArg(lean_object*);
lean_object* lp_mathlib_instDistribOfSemiring___redArg(lean_object*);
lean_object* lp_mathlib_instMulZeroClassOfSemiring___redArg(lean_object*);
uint8_t l_instDecidableEqList___redArg(lean_object*, lean_object*, lean_object*);
lean_object* l_List_filterTR_loop___redArg(lean_object*, lean_object*, lean_object*);
lean_object* l_List_mapTR_loop___redArg(lean_object*, lean_object*, lean_object*);
lean_object* l_List_sum___redArg(lean_object*, lean_object*, lean_object*);
uint8_t l_List_all___redArg(lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_swirl_x2dfv_Fundamentals_Bus_instDecidableEqBusEvent___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_Bus_instDecidableEqBusEvent___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_swirl_x2dfv_Fundamentals_Bus_instDecidableEqBusEvent(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_Bus_instDecidableEqBusEvent___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_Bus_BusEvent_executableMultiplicity___redArg___lam__0(lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_Bus_BusEvent_executableMultiplicity___redArg___lam__0___boxed(lean_object*);
LEAN_EXPORT uint8_t lp_swirl_x2dfv_Fundamentals_Bus_BusEvent_executableMultiplicity___redArg___lam__1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_Bus_BusEvent_executableMultiplicity___redArg___lam__1___boxed(lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_swirl_x2dfv_Fundamentals_Bus_BusEvent_executableMultiplicity___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_swirl_x2dfv_Fundamentals_Bus_BusEvent_executableMultiplicity___redArg___lam__0___boxed, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_swirl_x2dfv_Fundamentals_Bus_BusEvent_executableMultiplicity___redArg___closed__0 = (const lean_object*)&lp_swirl_x2dfv_Fundamentals_Bus_BusEvent_executableMultiplicity___redArg___closed__0_value;
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_Bus_BusEvent_executableMultiplicity___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_Bus_BusEvent_executableMultiplicity___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_Bus_BusEvent_executableMultiplicity(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_Bus_BusEvent_executableMultiplicity___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_swirl_x2dfv_Fundamentals_Bus_BusEvent_balanceCheck___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_Bus_BusEvent_balanceCheck___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_swirl_x2dfv_Fundamentals_Bus_BusEvent_balanceCheck___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_Bus_BusEvent_balanceCheck___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_swirl_x2dfv_Fundamentals_Bus_BusEvent_balanceCheck(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_Bus_BusEvent_balanceCheck___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_swirl_x2dfv_Fundamentals_Bus_instDecidableEqBusEvent___redArg(lean_object* v_inst_1_, lean_object* v_e_u2081_2_, lean_object* v_e_u2082_3_){
_start:
{
lean_object* v_mult_4_; lean_object* v_msg_5_; lean_object* v_mult_6_; lean_object* v_msg_7_; lean_object* v___x_8_; uint8_t v___x_9_; 
v_mult_4_ = lean_ctor_get(v_e_u2081_2_, 0);
lean_inc(v_mult_4_);
v_msg_5_ = lean_ctor_get(v_e_u2081_2_, 1);
lean_inc(v_msg_5_);
lean_dec_ref(v_e_u2081_2_);
v_mult_6_ = lean_ctor_get(v_e_u2082_3_, 0);
lean_inc(v_mult_6_);
v_msg_7_ = lean_ctor_get(v_e_u2082_3_, 1);
lean_inc(v_msg_7_);
lean_dec_ref(v_e_u2082_3_);
lean_inc_ref(v_inst_1_);
v___x_8_ = lean_apply_2(v_inst_1_, v_mult_4_, v_mult_6_);
v___x_9_ = lean_unbox(v___x_8_);
if (v___x_9_ == 0)
{
uint8_t v___x_10_; 
lean_dec(v_msg_7_);
lean_dec(v_msg_5_);
lean_dec_ref(v_inst_1_);
v___x_10_ = lean_unbox(v___x_8_);
return v___x_10_;
}
else
{
uint8_t v___x_11_; 
v___x_11_ = l_instDecidableEqList___redArg(v_inst_1_, v_msg_5_, v_msg_7_);
return v___x_11_;
}
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_Bus_instDecidableEqBusEvent___redArg___boxed(lean_object* v_inst_12_, lean_object* v_e_u2081_13_, lean_object* v_e_u2082_14_){
_start:
{
uint8_t v_res_15_; lean_object* v_r_16_; 
v_res_15_ = lp_swirl_x2dfv_Fundamentals_Bus_instDecidableEqBusEvent___redArg(v_inst_12_, v_e_u2081_13_, v_e_u2082_14_);
v_r_16_ = lean_box(v_res_15_);
return v_r_16_;
}
}
LEAN_EXPORT uint8_t lp_swirl_x2dfv_Fundamentals_Bus_instDecidableEqBusEvent(lean_object* v_F_17_, lean_object* v_inst_18_, lean_object* v_inst_19_, lean_object* v_e_u2081_20_, lean_object* v_e_u2082_21_){
_start:
{
uint8_t v___x_22_; 
v___x_22_ = lp_swirl_x2dfv_Fundamentals_Bus_instDecidableEqBusEvent___redArg(v_inst_19_, v_e_u2081_20_, v_e_u2082_21_);
return v___x_22_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_Bus_instDecidableEqBusEvent___boxed(lean_object* v_F_23_, lean_object* v_inst_24_, lean_object* v_inst_25_, lean_object* v_e_u2081_26_, lean_object* v_e_u2082_27_){
_start:
{
uint8_t v_res_28_; lean_object* v_r_29_; 
v_res_28_ = lp_swirl_x2dfv_Fundamentals_Bus_instDecidableEqBusEvent(v_F_23_, v_inst_24_, v_inst_25_, v_e_u2081_26_, v_e_u2082_27_);
lean_dec_ref(v_inst_24_);
v_r_29_ = lean_box(v_res_28_);
return v_r_29_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_Bus_BusEvent_executableMultiplicity___redArg___lam__0(lean_object* v_self_30_){
_start:
{
lean_object* v_mult_31_; 
v_mult_31_ = lean_ctor_get(v_self_30_, 0);
lean_inc(v_mult_31_);
return v_mult_31_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_Bus_BusEvent_executableMultiplicity___redArg___lam__0___boxed(lean_object* v_self_32_){
_start:
{
lean_object* v_res_33_; 
v_res_33_ = lp_swirl_x2dfv_Fundamentals_Bus_BusEvent_executableMultiplicity___redArg___lam__0(v_self_32_);
lean_dec_ref(v_self_32_);
return v_res_33_;
}
}
LEAN_EXPORT uint8_t lp_swirl_x2dfv_Fundamentals_Bus_BusEvent_executableMultiplicity___redArg___lam__1(lean_object* v_inst_34_, lean_object* v_msg_35_, lean_object* v_event_36_){
_start:
{
lean_object* v_msg_37_; uint8_t v___x_38_; 
v_msg_37_ = lean_ctor_get(v_event_36_, 1);
lean_inc(v_msg_37_);
lean_dec_ref(v_event_36_);
v___x_38_ = l_instDecidableEqList___redArg(v_inst_34_, v_msg_37_, v_msg_35_);
return v___x_38_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_Bus_BusEvent_executableMultiplicity___redArg___lam__1___boxed(lean_object* v_inst_39_, lean_object* v_msg_40_, lean_object* v_event_41_){
_start:
{
uint8_t v_res_42_; lean_object* v_r_43_; 
v_res_42_ = lp_swirl_x2dfv_Fundamentals_Bus_BusEvent_executableMultiplicity___redArg___lam__1(v_inst_39_, v_msg_40_, v_event_41_);
v_r_43_ = lean_box(v_res_42_);
return v_r_43_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_Bus_BusEvent_executableMultiplicity___redArg(lean_object* v_inst_45_, lean_object* v_inst_46_, lean_object* v_events_47_, lean_object* v_msg_48_){
_start:
{
lean_object* v___x_49_; lean_object* v_toCommSemiring_50_; lean_object* v___x_51_; lean_object* v_toAdd_52_; lean_object* v___x_53_; lean_object* v_toZero_54_; lean_object* v___f_55_; lean_object* v___f_56_; lean_object* v___x_57_; lean_object* v___x_58_; lean_object* v___x_59_; lean_object* v___x_60_; 
v___x_49_ = lp_mathlib_Field_toSemifield___redArg(v_inst_45_);
v_toCommSemiring_50_ = lean_ctor_get(v___x_49_, 0);
lean_inc_ref_n(v_toCommSemiring_50_, 2);
lean_dec_ref(v___x_49_);
v___x_51_ = lp_mathlib_instDistribOfSemiring___redArg(v_toCommSemiring_50_);
v_toAdd_52_ = lean_ctor_get(v___x_51_, 1);
lean_inc(v_toAdd_52_);
lean_dec_ref(v___x_51_);
v___x_53_ = lp_mathlib_instMulZeroClassOfSemiring___redArg(v_toCommSemiring_50_);
v_toZero_54_ = lean_ctor_get(v___x_53_, 1);
lean_inc(v_toZero_54_);
lean_dec_ref(v___x_53_);
v___f_55_ = ((lean_object*)(lp_swirl_x2dfv_Fundamentals_Bus_BusEvent_executableMultiplicity___redArg___closed__0));
v___f_56_ = lean_alloc_closure((void*)(lp_swirl_x2dfv_Fundamentals_Bus_BusEvent_executableMultiplicity___redArg___lam__1___boxed), 3, 2);
lean_closure_set(v___f_56_, 0, v_inst_46_);
lean_closure_set(v___f_56_, 1, v_msg_48_);
v___x_57_ = lean_box(0);
v___x_58_ = l_List_filterTR_loop___redArg(v___f_56_, v_events_47_, v___x_57_);
v___x_59_ = l_List_mapTR_loop___redArg(v___f_55_, v___x_58_, v___x_57_);
v___x_60_ = l_List_sum___redArg(v_toAdd_52_, v_toZero_54_, v___x_59_);
lean_dec(v_toZero_54_);
return v___x_60_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_Bus_BusEvent_executableMultiplicity___redArg___boxed(lean_object* v_inst_61_, lean_object* v_inst_62_, lean_object* v_events_63_, lean_object* v_msg_64_){
_start:
{
lean_object* v_res_65_; 
v_res_65_ = lp_swirl_x2dfv_Fundamentals_Bus_BusEvent_executableMultiplicity___redArg(v_inst_61_, v_inst_62_, v_events_63_, v_msg_64_);
lean_dec_ref(v_inst_61_);
return v_res_65_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_Bus_BusEvent_executableMultiplicity(lean_object* v_F_66_, lean_object* v_inst_67_, lean_object* v_inst_68_, lean_object* v_events_69_, lean_object* v_msg_70_){
_start:
{
lean_object* v___x_71_; 
v___x_71_ = lp_swirl_x2dfv_Fundamentals_Bus_BusEvent_executableMultiplicity___redArg(v_inst_67_, v_inst_68_, v_events_69_, v_msg_70_);
return v___x_71_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_Bus_BusEvent_executableMultiplicity___boxed(lean_object* v_F_72_, lean_object* v_inst_73_, lean_object* v_inst_74_, lean_object* v_events_75_, lean_object* v_msg_76_){
_start:
{
lean_object* v_res_77_; 
v_res_77_ = lp_swirl_x2dfv_Fundamentals_Bus_BusEvent_executableMultiplicity(v_F_72_, v_inst_73_, v_inst_74_, v_events_75_, v_msg_76_);
lean_dec_ref(v_inst_73_);
return v_res_77_;
}
}
LEAN_EXPORT uint8_t lp_swirl_x2dfv_Fundamentals_Bus_BusEvent_balanceCheck___redArg___lam__0(lean_object* v_inst_78_, lean_object* v_inst_79_, lean_object* v_events_80_, lean_object* v_toZero_81_, lean_object* v_event_82_){
_start:
{
lean_object* v_msg_83_; lean_object* v___x_84_; lean_object* v___x_85_; uint8_t v___x_86_; 
v_msg_83_ = lean_ctor_get(v_event_82_, 1);
lean_inc(v_msg_83_);
lean_dec_ref(v_event_82_);
lean_inc_ref(v_inst_79_);
v___x_84_ = lp_swirl_x2dfv_Fundamentals_Bus_BusEvent_executableMultiplicity___redArg(v_inst_78_, v_inst_79_, v_events_80_, v_msg_83_);
v___x_85_ = lean_apply_2(v_inst_79_, v___x_84_, v_toZero_81_);
v___x_86_ = lean_unbox(v___x_85_);
return v___x_86_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_Bus_BusEvent_balanceCheck___redArg___lam__0___boxed(lean_object* v_inst_87_, lean_object* v_inst_88_, lean_object* v_events_89_, lean_object* v_toZero_90_, lean_object* v_event_91_){
_start:
{
uint8_t v_res_92_; lean_object* v_r_93_; 
v_res_92_ = lp_swirl_x2dfv_Fundamentals_Bus_BusEvent_balanceCheck___redArg___lam__0(v_inst_87_, v_inst_88_, v_events_89_, v_toZero_90_, v_event_91_);
lean_dec_ref(v_inst_87_);
v_r_93_ = lean_box(v_res_92_);
return v_r_93_;
}
}
LEAN_EXPORT uint8_t lp_swirl_x2dfv_Fundamentals_Bus_BusEvent_balanceCheck___redArg(lean_object* v_inst_94_, lean_object* v_inst_95_, lean_object* v_events_96_){
_start:
{
lean_object* v___x_97_; lean_object* v_toCommSemiring_98_; lean_object* v___x_99_; lean_object* v_toZero_100_; lean_object* v___f_101_; uint8_t v___x_102_; 
v___x_97_ = lp_mathlib_Field_toSemifield___redArg(v_inst_94_);
v_toCommSemiring_98_ = lean_ctor_get(v___x_97_, 0);
lean_inc_ref(v_toCommSemiring_98_);
lean_dec_ref(v___x_97_);
v___x_99_ = lp_mathlib_instMulZeroClassOfSemiring___redArg(v_toCommSemiring_98_);
v_toZero_100_ = lean_ctor_get(v___x_99_, 1);
lean_inc(v_toZero_100_);
lean_dec_ref(v___x_99_);
lean_inc(v_events_96_);
v___f_101_ = lean_alloc_closure((void*)(lp_swirl_x2dfv_Fundamentals_Bus_BusEvent_balanceCheck___redArg___lam__0___boxed), 5, 4);
lean_closure_set(v___f_101_, 0, v_inst_94_);
lean_closure_set(v___f_101_, 1, v_inst_95_);
lean_closure_set(v___f_101_, 2, v_events_96_);
lean_closure_set(v___f_101_, 3, v_toZero_100_);
v___x_102_ = l_List_all___redArg(v_events_96_, v___f_101_);
return v___x_102_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_Bus_BusEvent_balanceCheck___redArg___boxed(lean_object* v_inst_103_, lean_object* v_inst_104_, lean_object* v_events_105_){
_start:
{
uint8_t v_res_106_; lean_object* v_r_107_; 
v_res_106_ = lp_swirl_x2dfv_Fundamentals_Bus_BusEvent_balanceCheck___redArg(v_inst_103_, v_inst_104_, v_events_105_);
v_r_107_ = lean_box(v_res_106_);
return v_r_107_;
}
}
LEAN_EXPORT uint8_t lp_swirl_x2dfv_Fundamentals_Bus_BusEvent_balanceCheck(lean_object* v_F_108_, lean_object* v_inst_109_, lean_object* v_inst_110_, lean_object* v_events_111_){
_start:
{
uint8_t v___x_112_; 
v___x_112_ = lp_swirl_x2dfv_Fundamentals_Bus_BusEvent_balanceCheck___redArg(v_inst_109_, v_inst_110_, v_events_111_);
return v___x_112_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_Bus_BusEvent_balanceCheck___boxed(lean_object* v_F_113_, lean_object* v_inst_114_, lean_object* v_inst_115_, lean_object* v_events_116_){
_start:
{
uint8_t v_res_117_; lean_object* v_r_118_; 
v_res_117_ = lp_swirl_x2dfv_Fundamentals_Bus_BusEvent_balanceCheck(v_F_113_, v_inst_114_, v_inst_115_, v_events_116_);
v_r_118_ = lean_box(v_res_117_);
return v_r_118_;
}
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Field_Defs(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_BigOperators_Group_Finset_Basic(uint8_t builtin);
void lean_initialize();
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_swirl_x2dfv_Fundamentals_Spec_Bus_Core(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
lean_initialize();
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Field_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_BigOperators_Group_Finset_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
#ifdef __cplusplus
}
#endif
