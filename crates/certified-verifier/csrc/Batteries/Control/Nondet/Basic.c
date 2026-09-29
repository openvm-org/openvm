// Lean compiler output
// Module: Batteries.Control.Nondet.Basic
// Imports: public import Init public meta import Init public import Batteries.Tactic.Lint.Misc public import Batteries.Data.MLList.Basic import Lean.Util.MonadBacktrack
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
lean_object* lp_batteries_MLList_append___redArg(lean_object*, lean_object*, lean_object*);
lean_object* lp_batteries___private_Batteries_Data_MLList_Basic_0__MLList_unconsImpl(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_batteries_AlternativeMonad_toMonad___redArg(lean_object*);
lean_object* lp_batteries_MLList_head___redArg(lean_object*, lean_object*);
lean_object* lp_batteries_MLList_force___redArg(lean_object*, lean_object*);
lean_object* lp_batteries_MLList_singletonM___redArg(lean_object*, lean_object*);
lean_object* l_Function_comp(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Function_const___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_List_mapTR_loop___redArg(lean_object*, lean_object*, lean_object*);
lean_object* lp_batteries_MLList_ofListM___redArg(lean_object*, lean_object*);
lean_object* lp_batteries_MLList_map___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Nondet_nil(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Nondet_nil___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Nondet_instInhabited(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Nondet_instInhabited___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Nondet_squash___redArg___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Nondet_squash___redArg___lam__1(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Nondet_squash___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Nondet_squash(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Nondet_squash___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Nondet_bind___redArg___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Nondet_bind___redArg___lam__0___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Nondet_bind___redArg___lam__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Nondet_bind___redArg___lam__2(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Nondet_bind___redArg___lam__4(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Nondet_bind___redArg___lam__3(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Nondet_bind___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Nondet_bind(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Nondet_singletonM___redArg___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Nondet_singletonM___redArg___lam__1(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Nondet_singletonM___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Nondet_singletonM(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Nondet_singleton___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Nondet_singleton(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Nondet_instAlternativeMonad___redArg___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Nondet_instAlternativeMonad___redArg___lam__0___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Nondet_instAlternativeMonad___redArg___lam__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Nondet_instAlternativeMonad___redArg___lam__2(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Nondet_instAlternativeMonad___redArg___lam__3(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Nondet_instAlternativeMonad___redArg___lam__4(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Nondet_instAlternativeMonad___redArg___lam__5(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Nondet_instAlternativeMonad___redArg___lam__7(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Nondet_instAlternativeMonad___redArg___lam__6(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Nondet_instAlternativeMonad___redArg___lam__8(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Nondet_instAlternativeMonad___redArg___lam__9(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Nondet_instAlternativeMonad___redArg___lam__10(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Nondet_instAlternativeMonad___redArg___lam__10___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Nondet_instAlternativeMonad___redArg___lam__11(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Nondet_instAlternativeMonad___redArg___lam__12(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Nondet_instAlternativeMonad___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Nondet_instAlternativeMonad(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Nondet_instMonadLift___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Nondet_instMonadLift(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Nondet_ofListM___redArg___lam__1(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Nondet_ofListM___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Nondet_ofListM___redArg___lam__2(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Nondet_ofListM___redArg___lam__3(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Nondet_ofListM___redArg___lam__4(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Nondet_ofListM___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Nondet_ofListM(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Nondet_ofList___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Nondet_ofList(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Nondet_mapM___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Nondet_mapM___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Nondet_mapM(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Nondet_map___redArg___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Nondet_map___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Nondet_map(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Nondet_ofOptionM___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Nondet_ofOptionM___redArg___lam__1(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Nondet_ofOptionM___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Nondet_ofOptionM(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Nondet_ofOption___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Nondet_ofOption(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Nondet_filterMapM___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Nondet_filterMapM___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Nondet_filterMapM(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Nondet_filterMap___redArg___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Nondet_filterMap___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Nondet_filterMap(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Nondet_filterM___redArg___lam__0(lean_object*, lean_object*, uint8_t);
LEAN_EXPORT lean_object* lp_batteries_Nondet_filterM___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Nondet_filterM___redArg___lam__1(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Nondet_filterM___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Nondet_filterM(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Nondet_filter___redArg___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Nondet_filter___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Nondet_filter(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Nondet_iterate___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Nondet_iterate___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Nondet_iterate(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Nondet_toMLList_x27___redArg___lam__0(lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Nondet_toMLList_x27___redArg___lam__0___boxed(lean_object*);
static const lean_closure_object lp_batteries_Nondet_toMLList_x27___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_batteries_Nondet_toMLList_x27___redArg___lam__0___boxed, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_batteries_Nondet_toMLList_x27___redArg___closed__0 = (const lean_object*)&lp_batteries_Nondet_toMLList_x27___redArg___closed__0_value;
LEAN_EXPORT lean_object* lp_batteries_Nondet_toMLList_x27___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Nondet_toMLList_x27(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Nondet_toMLList_x27___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Nondet_toList___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Nondet_toList(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Nondet_toList___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Nondet_toList_x27___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Nondet_toList_x27(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Nondet_toList_x27___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Nondet_head___redArg___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Nondet_head___redArg___lam__1(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Nondet_head___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Nondet_head(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Nondet_firstM___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Nondet_firstM(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_instMonadBacktrackUnitId__batteries___lam__0(lean_object*, lean_object*);
static const lean_closure_object lp_batteries_instMonadBacktrackUnitId__batteries___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*1, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_batteries_instMonadBacktrackUnitId__batteries___lam__0, .m_arity = 2, .m_num_fixed = 1, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1))} };
static const lean_object* lp_batteries_instMonadBacktrackUnitId__batteries___closed__0 = (const lean_object*)&lp_batteries_instMonadBacktrackUnitId__batteries___closed__0_value;
static const lean_ctor_object lp_batteries_instMonadBacktrackUnitId__batteries___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_batteries_instMonadBacktrackUnitId__batteries___closed__0_value)}};
static const lean_object* lp_batteries_instMonadBacktrackUnitId__batteries___closed__1 = (const lean_object*)&lp_batteries_instMonadBacktrackUnitId__batteries___closed__1_value;
LEAN_EXPORT const lean_object* lp_batteries_instMonadBacktrackUnitId__batteries = (const lean_object*)&lp_batteries_instMonadBacktrackUnitId__batteries___closed__1_value;
LEAN_EXPORT lean_object* lp_batteries_Nondet_nil(lean_object* v_00_u03c3_1_, lean_object* v_m_2_, lean_object* v_inst_3_, lean_object* v_00_u03b1_4_){
_start:
{
lean_object* v___x_5_; 
v___x_5_ = lean_box(0);
return v___x_5_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Nondet_nil___boxed(lean_object* v_00_u03c3_6_, lean_object* v_m_7_, lean_object* v_inst_8_, lean_object* v_00_u03b1_9_){
_start:
{
lean_object* v_res_10_; 
v_res_10_ = lp_batteries_Nondet_nil(v_00_u03c3_6_, v_m_7_, v_inst_8_, v_00_u03b1_9_);
lean_dec_ref(v_inst_8_);
return v_res_10_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Nondet_instInhabited(lean_object* v_00_u03c3_11_, lean_object* v_m_12_, lean_object* v_inst_13_, lean_object* v_00_u03b1_14_){
_start:
{
lean_object* v___x_15_; 
v___x_15_ = lean_box(0);
return v___x_15_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Nondet_instInhabited___boxed(lean_object* v_00_u03c3_16_, lean_object* v_m_17_, lean_object* v_inst_18_, lean_object* v_00_u03b1_19_){
_start:
{
lean_object* v_res_20_; 
v_res_20_ = lp_batteries_Nondet_instInhabited(v_00_u03c3_16_, v_m_17_, v_inst_18_, v_00_u03b1_19_);
lean_dec_ref(v_inst_18_);
return v_res_20_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Nondet_squash___redArg___lam__0(lean_object* v_toPure_21_, lean_object* v_____do__lift_22_){
_start:
{
lean_object* v___x_23_; 
v___x_23_ = lean_apply_2(v_toPure_21_, lean_box(0), v_____do__lift_22_);
return v___x_23_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Nondet_squash___redArg___lam__1(lean_object* v_L_24_, lean_object* v_toBind_25_, lean_object* v___f_26_, lean_object* v_x_27_){
_start:
{
lean_object* v___x_28_; lean_object* v___x_29_; lean_object* v___x_30_; 
v___x_28_ = lean_box(0);
v___x_29_ = lean_apply_1(v_L_24_, v___x_28_);
v___x_30_ = lean_apply_4(v_toBind_25_, lean_box(0), lean_box(0), v___x_29_, v___f_26_);
return v___x_30_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Nondet_squash___redArg(lean_object* v_inst_31_, lean_object* v_L_32_){
_start:
{
lean_object* v_toApplicative_33_; lean_object* v_toBind_34_; lean_object* v_toPure_35_; lean_object* v___f_36_; lean_object* v___f_37_; lean_object* v___x_38_; 
v_toApplicative_33_ = lean_ctor_get(v_inst_31_, 0);
lean_inc_ref(v_toApplicative_33_);
v_toBind_34_ = lean_ctor_get(v_inst_31_, 1);
lean_inc(v_toBind_34_);
lean_dec_ref(v_inst_31_);
v_toPure_35_ = lean_ctor_get(v_toApplicative_33_, 1);
lean_inc(v_toPure_35_);
lean_dec_ref(v_toApplicative_33_);
v___f_36_ = lean_alloc_closure((void*)(lp_batteries_Nondet_squash___redArg___lam__0), 2, 1);
lean_closure_set(v___f_36_, 0, v_toPure_35_);
v___f_37_ = lean_alloc_closure((void*)(lp_batteries_Nondet_squash___redArg___lam__1), 4, 3);
lean_closure_set(v___f_37_, 0, v_L_32_);
lean_closure_set(v___f_37_, 1, v_toBind_34_);
lean_closure_set(v___f_37_, 2, v___f_36_);
v___x_38_ = lean_alloc_ctor(3, 1, 0);
lean_ctor_set(v___x_38_, 0, v___f_37_);
return v___x_38_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Nondet_squash(lean_object* v_00_u03c3_39_, lean_object* v_m_40_, lean_object* v_inst_41_, lean_object* v_inst_42_, lean_object* v_00_u03b1_43_, lean_object* v_L_44_){
_start:
{
lean_object* v___x_45_; 
v___x_45_ = lp_batteries_Nondet_squash___redArg(v_inst_41_, v_L_44_);
return v___x_45_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Nondet_squash___boxed(lean_object* v_00_u03c3_46_, lean_object* v_m_47_, lean_object* v_inst_48_, lean_object* v_inst_49_, lean_object* v_00_u03b1_50_, lean_object* v_L_51_){
_start:
{
lean_object* v_res_52_; 
v_res_52_ = lp_batteries_Nondet_squash(v_00_u03c3_46_, v_m_47_, v_inst_48_, v_inst_49_, v_00_u03b1_50_, v_L_51_);
lean_dec_ref(v_inst_49_);
return v_res_52_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Nondet_bind___redArg___lam__0(lean_object* v_r_53_, lean_object* v_x_54_){
_start:
{
lean_inc(v_r_53_);
return v_r_53_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Nondet_bind___redArg___lam__0___boxed(lean_object* v_r_55_, lean_object* v_x_56_){
_start:
{
lean_object* v_res_57_; 
v_res_57_ = lp_batteries_Nondet_bind___redArg___lam__0(v_r_55_, v_x_56_);
lean_dec(v_r_55_);
return v_res_57_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Nondet_bind___redArg___lam__1(lean_object* v_toPure_58_, lean_object* v_r_59_, lean_object* v_inst_60_, lean_object* v___f_61_, lean_object* v_____do__lift_62_){
_start:
{
if (lean_obj_tag(v_____do__lift_62_) == 0)
{
lean_object* v___x_63_; 
lean_dec(v___f_61_);
lean_dec_ref(v_inst_60_);
v___x_63_ = lean_apply_2(v_toPure_58_, lean_box(0), v_r_59_);
return v___x_63_;
}
else
{
lean_object* v_val_64_; lean_object* v_fst_65_; lean_object* v_snd_66_; lean_object* v___x_68_; uint8_t v_isShared_69_; uint8_t v_isSharedCheck_75_; 
lean_dec(v_r_59_);
v_val_64_ = lean_ctor_get(v_____do__lift_62_, 0);
lean_inc(v_val_64_);
lean_dec_ref_known(v_____do__lift_62_, 1);
v_fst_65_ = lean_ctor_get(v_val_64_, 0);
v_snd_66_ = lean_ctor_get(v_val_64_, 1);
v_isSharedCheck_75_ = !lean_is_exclusive(v_val_64_);
if (v_isSharedCheck_75_ == 0)
{
v___x_68_ = v_val_64_;
v_isShared_69_ = v_isSharedCheck_75_;
goto v_resetjp_67_;
}
else
{
lean_inc(v_snd_66_);
lean_inc(v_fst_65_);
lean_dec(v_val_64_);
v___x_68_ = lean_box(0);
v_isShared_69_ = v_isSharedCheck_75_;
goto v_resetjp_67_;
}
v_resetjp_67_:
{
lean_object* v___x_70_; lean_object* v___x_72_; 
v___x_70_ = lp_batteries_MLList_append___redArg(v_inst_60_, v_snd_66_, v___f_61_);
if (v_isShared_69_ == 0)
{
lean_ctor_set_tag(v___x_68_, 1);
lean_ctor_set(v___x_68_, 1, v___x_70_);
v___x_72_ = v___x_68_;
goto v_reusejp_71_;
}
else
{
lean_object* v_reuseFailAlloc_74_; 
v_reuseFailAlloc_74_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_74_, 0, v_fst_65_);
lean_ctor_set(v_reuseFailAlloc_74_, 1, v___x_70_);
v___x_72_ = v_reuseFailAlloc_74_;
goto v_reusejp_71_;
}
v_reusejp_71_:
{
lean_object* v___x_73_; 
v___x_73_ = lean_apply_2(v_toPure_58_, lean_box(0), v___x_72_);
return v___x_73_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_batteries_Nondet_bind___redArg___lam__2(lean_object* v_f_76_, lean_object* v_fst_77_, lean_object* v_inst_78_, lean_object* v_toBind_79_, lean_object* v___f_80_, lean_object* v_____r_81_){
_start:
{
lean_object* v___x_82_; lean_object* v___x_83_; lean_object* v___x_84_; 
v___x_82_ = lean_apply_1(v_f_76_, v_fst_77_);
v___x_83_ = lp_batteries___private_Batteries_Data_MLList_Basic_0__MLList_unconsImpl(lean_box(0), lean_box(0), v_inst_78_, v___x_82_);
v___x_84_ = lean_apply_4(v_toBind_79_, lean_box(0), lean_box(0), v___x_83_, v___f_80_);
return v___x_84_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Nondet_bind___redArg___lam__4(lean_object* v_inst_85_, lean_object* v_L_86_, lean_object* v_toBind_87_, lean_object* v___f_88_, lean_object* v_x_89_){
_start:
{
lean_object* v___x_90_; lean_object* v___x_91_; 
v___x_90_ = lp_batteries___private_Batteries_Data_MLList_Basic_0__MLList_unconsImpl(lean_box(0), lean_box(0), v_inst_85_, v_L_86_);
v___x_91_ = lean_apply_4(v_toBind_87_, lean_box(0), lean_box(0), v___x_90_, v___f_88_);
return v___x_91_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Nondet_bind___redArg___lam__3(lean_object* v_toPure_92_, lean_object* v_inst_93_, lean_object* v_inst_94_, lean_object* v_f_95_, lean_object* v_toBind_96_, lean_object* v_____do__lift_97_){
_start:
{
if (lean_obj_tag(v_____do__lift_97_) == 0)
{
lean_object* v___x_98_; lean_object* v___x_99_; 
lean_dec(v_toBind_96_);
lean_dec(v_f_95_);
lean_dec_ref(v_inst_94_);
lean_dec_ref(v_inst_93_);
v___x_98_ = lean_box(0);
v___x_99_ = lean_apply_2(v_toPure_92_, lean_box(0), v___x_98_);
return v___x_99_;
}
else
{
lean_object* v_val_100_; lean_object* v_fst_101_; lean_object* v_snd_102_; lean_object* v_fst_103_; lean_object* v_snd_104_; lean_object* v_restoreState_105_; lean_object* v_r_106_; lean_object* v___f_107_; lean_object* v___f_108_; lean_object* v___f_109_; lean_object* v___x_110_; lean_object* v___x_111_; 
v_val_100_ = lean_ctor_get(v_____do__lift_97_, 0);
lean_inc(v_val_100_);
lean_dec_ref_known(v_____do__lift_97_, 1);
v_fst_101_ = lean_ctor_get(v_val_100_, 0);
lean_inc(v_fst_101_);
v_snd_102_ = lean_ctor_get(v_val_100_, 1);
lean_inc(v_snd_102_);
lean_dec(v_val_100_);
v_fst_103_ = lean_ctor_get(v_fst_101_, 0);
lean_inc(v_fst_103_);
v_snd_104_ = lean_ctor_get(v_fst_101_, 1);
lean_inc(v_snd_104_);
lean_dec(v_fst_101_);
v_restoreState_105_ = lean_ctor_get(v_inst_93_, 1);
lean_inc(v_restoreState_105_);
lean_inc(v_f_95_);
lean_inc_ref_n(v_inst_94_, 2);
v_r_106_ = lp_batteries_Nondet_bind___redArg(v_inst_94_, v_inst_93_, v_snd_102_, v_f_95_);
lean_inc(v_r_106_);
v___f_107_ = lean_alloc_closure((void*)(lp_batteries_Nondet_bind___redArg___lam__0___boxed), 2, 1);
lean_closure_set(v___f_107_, 0, v_r_106_);
v___f_108_ = lean_alloc_closure((void*)(lp_batteries_Nondet_bind___redArg___lam__1), 5, 4);
lean_closure_set(v___f_108_, 0, v_toPure_92_);
lean_closure_set(v___f_108_, 1, v_r_106_);
lean_closure_set(v___f_108_, 2, v_inst_94_);
lean_closure_set(v___f_108_, 3, v___f_107_);
lean_inc(v_toBind_96_);
v___f_109_ = lean_alloc_closure((void*)(lp_batteries_Nondet_bind___redArg___lam__2), 6, 5);
lean_closure_set(v___f_109_, 0, v_f_95_);
lean_closure_set(v___f_109_, 1, v_fst_103_);
lean_closure_set(v___f_109_, 2, v_inst_94_);
lean_closure_set(v___f_109_, 3, v_toBind_96_);
lean_closure_set(v___f_109_, 4, v___f_108_);
v___x_110_ = lean_apply_1(v_restoreState_105_, v_snd_104_);
v___x_111_ = lean_apply_4(v_toBind_96_, lean_box(0), lean_box(0), v___x_110_, v___f_109_);
return v___x_111_;
}
}
}
LEAN_EXPORT lean_object* lp_batteries_Nondet_bind___redArg(lean_object* v_inst_112_, lean_object* v_inst_113_, lean_object* v_L_114_, lean_object* v_f_115_){
_start:
{
lean_object* v_toApplicative_116_; lean_object* v_toBind_117_; lean_object* v_toPure_118_; lean_object* v___f_119_; lean_object* v___f_120_; lean_object* v___x_121_; 
v_toApplicative_116_ = lean_ctor_get(v_inst_112_, 0);
v_toBind_117_ = lean_ctor_get(v_inst_112_, 1);
v_toPure_118_ = lean_ctor_get(v_toApplicative_116_, 1);
lean_inc_n(v_toBind_117_, 2);
lean_inc_ref_n(v_inst_112_, 2);
lean_inc(v_toPure_118_);
v___f_119_ = lean_alloc_closure((void*)(lp_batteries_Nondet_bind___redArg___lam__3), 6, 5);
lean_closure_set(v___f_119_, 0, v_toPure_118_);
lean_closure_set(v___f_119_, 1, v_inst_113_);
lean_closure_set(v___f_119_, 2, v_inst_112_);
lean_closure_set(v___f_119_, 3, v_f_115_);
lean_closure_set(v___f_119_, 4, v_toBind_117_);
v___f_120_ = lean_alloc_closure((void*)(lp_batteries_Nondet_bind___redArg___lam__4), 5, 4);
lean_closure_set(v___f_120_, 0, v_inst_112_);
lean_closure_set(v___f_120_, 1, v_L_114_);
lean_closure_set(v___f_120_, 2, v_toBind_117_);
lean_closure_set(v___f_120_, 3, v___f_119_);
v___x_121_ = lp_batteries_Nondet_squash___redArg(v_inst_112_, v___f_120_);
return v___x_121_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Nondet_bind(lean_object* v_00_u03c3_122_, lean_object* v_m_123_, lean_object* v_inst_124_, lean_object* v_inst_125_, lean_object* v_00_u03b1_126_, lean_object* v_00_u03b2_127_, lean_object* v_L_128_, lean_object* v_f_129_){
_start:
{
lean_object* v___x_130_; 
v___x_130_ = lp_batteries_Nondet_bind___redArg(v_inst_124_, v_inst_125_, v_L_128_, v_f_129_);
return v___x_130_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Nondet_singletonM___redArg___lam__0(lean_object* v_a_131_, lean_object* v_toPure_132_, lean_object* v_____do__lift_133_){
_start:
{
lean_object* v___x_134_; lean_object* v___x_135_; 
v___x_134_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_134_, 0, v_a_131_);
lean_ctor_set(v___x_134_, 1, v_____do__lift_133_);
v___x_135_ = lean_apply_2(v_toPure_132_, lean_box(0), v___x_134_);
return v___x_135_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Nondet_singletonM___redArg___lam__1(lean_object* v_inst_136_, lean_object* v_toPure_137_, lean_object* v_toBind_138_, lean_object* v_a_139_){
_start:
{
lean_object* v_saveState_140_; lean_object* v___f_141_; lean_object* v___x_142_; 
v_saveState_140_ = lean_ctor_get(v_inst_136_, 0);
lean_inc(v_saveState_140_);
lean_dec_ref(v_inst_136_);
v___f_141_ = lean_alloc_closure((void*)(lp_batteries_Nondet_singletonM___redArg___lam__0), 3, 2);
lean_closure_set(v___f_141_, 0, v_a_139_);
lean_closure_set(v___f_141_, 1, v_toPure_137_);
v___x_142_ = lean_apply_4(v_toBind_138_, lean_box(0), lean_box(0), v_saveState_140_, v___f_141_);
return v___x_142_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Nondet_singletonM___redArg(lean_object* v_inst_143_, lean_object* v_inst_144_, lean_object* v_x_145_){
_start:
{
lean_object* v_toApplicative_146_; lean_object* v_toBind_147_; lean_object* v_toPure_148_; lean_object* v___f_149_; lean_object* v___x_150_; lean_object* v___x_151_; 
v_toApplicative_146_ = lean_ctor_get(v_inst_143_, 0);
v_toBind_147_ = lean_ctor_get(v_inst_143_, 1);
v_toPure_148_ = lean_ctor_get(v_toApplicative_146_, 1);
lean_inc_n(v_toBind_147_, 2);
lean_inc(v_toPure_148_);
v___f_149_ = lean_alloc_closure((void*)(lp_batteries_Nondet_singletonM___redArg___lam__1), 4, 3);
lean_closure_set(v___f_149_, 0, v_inst_144_);
lean_closure_set(v___f_149_, 1, v_toPure_148_);
lean_closure_set(v___f_149_, 2, v_toBind_147_);
v___x_150_ = lean_apply_4(v_toBind_147_, lean_box(0), lean_box(0), v_x_145_, v___f_149_);
v___x_151_ = lp_batteries_MLList_singletonM___redArg(v_inst_143_, v___x_150_);
return v___x_151_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Nondet_singletonM(lean_object* v_00_u03c3_152_, lean_object* v_m_153_, lean_object* v_inst_154_, lean_object* v_inst_155_, lean_object* v_00_u03b1_156_, lean_object* v_x_157_){
_start:
{
lean_object* v___x_158_; 
v___x_158_ = lp_batteries_Nondet_singletonM___redArg(v_inst_154_, v_inst_155_, v_x_157_);
return v___x_158_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Nondet_singleton___redArg(lean_object* v_inst_159_, lean_object* v_inst_160_, lean_object* v_x_161_){
_start:
{
lean_object* v_toApplicative_162_; lean_object* v_toPure_163_; lean_object* v___x_164_; lean_object* v___x_165_; 
v_toApplicative_162_ = lean_ctor_get(v_inst_159_, 0);
v_toPure_163_ = lean_ctor_get(v_toApplicative_162_, 1);
lean_inc(v_toPure_163_);
v___x_164_ = lean_apply_2(v_toPure_163_, lean_box(0), v_x_161_);
v___x_165_ = lp_batteries_Nondet_singletonM___redArg(v_inst_159_, v_inst_160_, v___x_164_);
return v___x_165_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Nondet_singleton(lean_object* v_00_u03c3_166_, lean_object* v_m_167_, lean_object* v_inst_168_, lean_object* v_inst_169_, lean_object* v_00_u03b1_170_, lean_object* v_x_171_){
_start:
{
lean_object* v___x_172_; 
v___x_172_ = lp_batteries_Nondet_singleton___redArg(v_inst_168_, v_inst_169_, v_x_171_);
return v___x_172_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Nondet_instAlternativeMonad___redArg___lam__0(lean_object* v_y_173_, lean_object* v_x_174_){
_start:
{
lean_object* v___x_175_; lean_object* v___x_176_; 
v___x_175_ = lean_box(0);
v___x_176_ = lean_apply_1(v_y_173_, v___x_175_);
return v___x_176_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Nondet_instAlternativeMonad___redArg___lam__0___boxed(lean_object* v_y_177_, lean_object* v_x_178_){
_start:
{
lean_object* v_res_179_; 
v_res_179_ = lp_batteries_Nondet_instAlternativeMonad___redArg___lam__0(v_y_177_, v_x_178_);
lean_dec(v_x_178_);
return v_res_179_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Nondet_instAlternativeMonad___redArg___lam__1(lean_object* v_inst_180_, lean_object* v_inst_181_, lean_object* v_00_u03b1_182_, lean_object* v_00_u03b2_183_, lean_object* v_x_184_, lean_object* v_y_185_){
_start:
{
lean_object* v___f_186_; lean_object* v___x_187_; 
v___f_186_ = lean_alloc_closure((void*)(lp_batteries_Nondet_instAlternativeMonad___redArg___lam__0___boxed), 2, 1);
lean_closure_set(v___f_186_, 0, v_y_185_);
v___x_187_ = lp_batteries_Nondet_bind___redArg(v_inst_180_, v_inst_181_, v_x_184_, v___f_186_);
return v___x_187_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Nondet_instAlternativeMonad___redArg___lam__2(lean_object* v_y_188_, lean_object* v_x_189_){
_start:
{
lean_object* v___x_190_; lean_object* v___x_191_; 
v___x_190_ = lean_box(0);
v___x_191_ = lean_apply_1(v_y_188_, v___x_190_);
return v___x_191_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Nondet_instAlternativeMonad___redArg___lam__3(lean_object* v_inst_192_, lean_object* v_00_u03b1_193_, lean_object* v_x_194_, lean_object* v_y_195_){
_start:
{
lean_object* v___f_196_; lean_object* v___x_197_; 
v___f_196_ = lean_alloc_closure((void*)(lp_batteries_Nondet_instAlternativeMonad___redArg___lam__2), 2, 1);
lean_closure_set(v___f_196_, 0, v_y_195_);
v___x_197_ = lp_batteries_MLList_append___redArg(v_inst_192_, v_x_194_, v___f_196_);
return v___x_197_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Nondet_instAlternativeMonad___redArg___lam__4(lean_object* v_toPure_198_, lean_object* v_inst_199_, lean_object* v_inst_200_, lean_object* v___y_201_){
_start:
{
lean_object* v___x_202_; lean_object* v___x_203_; 
v___x_202_ = lean_apply_2(v_toPure_198_, lean_box(0), v___y_201_);
v___x_203_ = lp_batteries_Nondet_singletonM___redArg(v_inst_199_, v_inst_200_, v___x_202_);
return v___x_203_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Nondet_instAlternativeMonad___redArg___lam__5(lean_object* v___f_204_, lean_object* v_inst_205_, lean_object* v_inst_206_, lean_object* v_00_u03b1_207_, lean_object* v_00_u03b2_208_, lean_object* v_f_209_, lean_object* v_x_210_){
_start:
{
lean_object* v___x_211_; lean_object* v___x_212_; 
v___x_211_ = lean_alloc_closure((void*)(l_Function_comp), 6, 5);
lean_closure_set(v___x_211_, 0, lean_box(0));
lean_closure_set(v___x_211_, 1, lean_box(0));
lean_closure_set(v___x_211_, 2, lean_box(0));
lean_closure_set(v___x_211_, 3, v___f_204_);
lean_closure_set(v___x_211_, 4, v_f_209_);
v___x_212_ = lp_batteries_Nondet_bind___redArg(v_inst_205_, v_inst_206_, v_x_210_, v___x_211_);
return v___x_212_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Nondet_instAlternativeMonad___redArg___lam__7(lean_object* v___f_213_, lean_object* v_inst_214_, lean_object* v_inst_215_, lean_object* v_00_u03b1_216_, lean_object* v_00_u03b2_217_, lean_object* v___y_218_, lean_object* v___y_219_){
_start:
{
lean_object* v___x_220_; lean_object* v___x_221_; lean_object* v___x_222_; 
v___x_220_ = lean_alloc_closure((void*)(l_Function_const___boxed), 4, 3);
lean_closure_set(v___x_220_, 0, lean_box(0));
lean_closure_set(v___x_220_, 1, lean_box(0));
lean_closure_set(v___x_220_, 2, v___y_218_);
v___x_221_ = lean_alloc_closure((void*)(l_Function_comp), 6, 5);
lean_closure_set(v___x_221_, 0, lean_box(0));
lean_closure_set(v___x_221_, 1, lean_box(0));
lean_closure_set(v___x_221_, 2, lean_box(0));
lean_closure_set(v___x_221_, 3, v___f_213_);
lean_closure_set(v___x_221_, 4, v___x_220_);
v___x_222_ = lp_batteries_Nondet_bind___redArg(v_inst_214_, v_inst_215_, v___y_219_, v___x_221_);
return v___x_222_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Nondet_instAlternativeMonad___redArg___lam__6(lean_object* v_toPure_223_, lean_object* v_inst_224_, lean_object* v_inst_225_, lean_object* v_00_u03b1_226_, lean_object* v_a_227_){
_start:
{
lean_object* v___x_228_; lean_object* v___x_229_; 
v___x_228_ = lean_apply_2(v_toPure_223_, lean_box(0), v_a_227_);
v___x_229_ = lp_batteries_Nondet_singletonM___redArg(v_inst_224_, v_inst_225_, v___x_228_);
return v___x_229_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Nondet_instAlternativeMonad___redArg___lam__8(lean_object* v_x_230_, lean_object* v___f_231_, lean_object* v_inst_232_, lean_object* v_inst_233_, lean_object* v_y_234_){
_start:
{
lean_object* v___x_235_; lean_object* v___x_236_; lean_object* v___x_237_; lean_object* v___x_238_; lean_object* v___x_239_; 
v___x_235_ = lean_box(0);
v___x_236_ = lean_apply_1(v_x_230_, v___x_235_);
v___x_237_ = lean_apply_1(v___f_231_, lean_box(0));
v___x_238_ = lean_alloc_closure((void*)(l_Function_comp), 6, 5);
lean_closure_set(v___x_238_, 0, lean_box(0));
lean_closure_set(v___x_238_, 1, lean_box(0));
lean_closure_set(v___x_238_, 2, lean_box(0));
lean_closure_set(v___x_238_, 3, v___x_237_);
lean_closure_set(v___x_238_, 4, v_y_234_);
v___x_239_ = lp_batteries_Nondet_bind___redArg(v_inst_232_, v_inst_233_, v___x_236_, v___x_238_);
return v___x_239_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Nondet_instAlternativeMonad___redArg___lam__9(lean_object* v___f_240_, lean_object* v_inst_241_, lean_object* v_inst_242_, lean_object* v_00_u03b1_243_, lean_object* v_00_u03b2_244_, lean_object* v_f_245_, lean_object* v_x_246_){
_start:
{
lean_object* v___f_247_; lean_object* v___x_248_; 
lean_inc_ref(v_inst_242_);
lean_inc_ref(v_inst_241_);
v___f_247_ = lean_alloc_closure((void*)(lp_batteries_Nondet_instAlternativeMonad___redArg___lam__8), 5, 4);
lean_closure_set(v___f_247_, 0, v_x_246_);
lean_closure_set(v___f_247_, 1, v___f_240_);
lean_closure_set(v___f_247_, 2, v_inst_241_);
lean_closure_set(v___f_247_, 3, v_inst_242_);
v___x_248_ = lp_batteries_Nondet_bind___redArg(v_inst_241_, v_inst_242_, v_f_245_, v___f_247_);
return v___x_248_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Nondet_instAlternativeMonad___redArg___lam__10(lean_object* v___f_249_, lean_object* v_a_250_, lean_object* v_x_251_){
_start:
{
lean_object* v___x_252_; 
v___x_252_ = lean_apply_2(v___f_249_, lean_box(0), v_a_250_);
return v___x_252_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Nondet_instAlternativeMonad___redArg___lam__10___boxed(lean_object* v___f_253_, lean_object* v_a_254_, lean_object* v_x_255_){
_start:
{
lean_object* v_res_256_; 
v_res_256_ = lp_batteries_Nondet_instAlternativeMonad___redArg___lam__10(v___f_253_, v_a_254_, v_x_255_);
lean_dec(v_x_255_);
return v_res_256_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Nondet_instAlternativeMonad___redArg___lam__11(lean_object* v___f_257_, lean_object* v_y_258_, lean_object* v_inst_259_, lean_object* v_inst_260_, lean_object* v_a_261_){
_start:
{
lean_object* v___f_262_; lean_object* v___x_263_; lean_object* v___x_264_; lean_object* v___x_265_; 
v___f_262_ = lean_alloc_closure((void*)(lp_batteries_Nondet_instAlternativeMonad___redArg___lam__10___boxed), 3, 2);
lean_closure_set(v___f_262_, 0, v___f_257_);
lean_closure_set(v___f_262_, 1, v_a_261_);
v___x_263_ = lean_box(0);
v___x_264_ = lean_apply_1(v_y_258_, v___x_263_);
v___x_265_ = lp_batteries_Nondet_bind___redArg(v_inst_259_, v_inst_260_, v___x_264_, v___f_262_);
return v___x_265_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Nondet_instAlternativeMonad___redArg___lam__12(lean_object* v___f_266_, lean_object* v_inst_267_, lean_object* v_inst_268_, lean_object* v_00_u03b1_269_, lean_object* v_00_u03b2_270_, lean_object* v_x_271_, lean_object* v_y_272_){
_start:
{
lean_object* v___f_273_; lean_object* v___x_274_; 
lean_inc_ref(v_inst_268_);
lean_inc_ref(v_inst_267_);
v___f_273_ = lean_alloc_closure((void*)(lp_batteries_Nondet_instAlternativeMonad___redArg___lam__11), 5, 4);
lean_closure_set(v___f_273_, 0, v___f_266_);
lean_closure_set(v___f_273_, 1, v_y_272_);
lean_closure_set(v___f_273_, 2, v_inst_267_);
lean_closure_set(v___f_273_, 3, v_inst_268_);
v___x_274_ = lp_batteries_Nondet_bind___redArg(v_inst_267_, v_inst_268_, v_x_271_, v___f_273_);
return v___x_274_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Nondet_instAlternativeMonad___redArg(lean_object* v_inst_275_, lean_object* v_inst_276_){
_start:
{
lean_object* v_toApplicative_277_; lean_object* v_toPure_278_; lean_object* v___x_280_; uint8_t v_isShared_281_; uint8_t v_isSharedCheck_298_; 
v_toApplicative_277_ = lean_ctor_get(v_inst_275_, 0);
lean_inc_ref(v_toApplicative_277_);
v_toPure_278_ = lean_ctor_get(v_toApplicative_277_, 1);
v_isSharedCheck_298_ = !lean_is_exclusive(v_toApplicative_277_);
if (v_isSharedCheck_298_ == 0)
{
lean_object* v_unused_299_; lean_object* v_unused_300_; lean_object* v_unused_301_; lean_object* v_unused_302_; 
v_unused_299_ = lean_ctor_get(v_toApplicative_277_, 4);
lean_dec(v_unused_299_);
v_unused_300_ = lean_ctor_get(v_toApplicative_277_, 3);
lean_dec(v_unused_300_);
v_unused_301_ = lean_ctor_get(v_toApplicative_277_, 2);
lean_dec(v_unused_301_);
v_unused_302_ = lean_ctor_get(v_toApplicative_277_, 0);
lean_dec(v_unused_302_);
v___x_280_ = v_toApplicative_277_;
v_isShared_281_ = v_isSharedCheck_298_;
goto v_resetjp_279_;
}
else
{
lean_inc(v_toPure_278_);
lean_dec(v_toApplicative_277_);
v___x_280_ = lean_box(0);
v_isShared_281_ = v_isSharedCheck_298_;
goto v_resetjp_279_;
}
v_resetjp_279_:
{
lean_object* v___f_282_; lean_object* v___f_283_; lean_object* v___f_284_; lean_object* v___f_285_; lean_object* v___f_286_; lean_object* v___f_287_; lean_object* v___f_288_; lean_object* v___f_289_; lean_object* v___x_290_; lean_object* v___x_292_; 
lean_inc_ref_n(v_inst_276_, 7);
lean_inc_ref_n(v_inst_275_, 8);
v___f_282_ = lean_alloc_closure((void*)(lp_batteries_Nondet_instAlternativeMonad___redArg___lam__1), 6, 2);
lean_closure_set(v___f_282_, 0, v_inst_275_);
lean_closure_set(v___f_282_, 1, v_inst_276_);
v___f_283_ = lean_alloc_closure((void*)(lp_batteries_Nondet_instAlternativeMonad___redArg___lam__3), 4, 1);
lean_closure_set(v___f_283_, 0, v_inst_275_);
lean_inc(v_toPure_278_);
v___f_284_ = lean_alloc_closure((void*)(lp_batteries_Nondet_instAlternativeMonad___redArg___lam__4), 4, 3);
lean_closure_set(v___f_284_, 0, v_toPure_278_);
lean_closure_set(v___f_284_, 1, v_inst_275_);
lean_closure_set(v___f_284_, 2, v_inst_276_);
lean_inc_ref(v___f_284_);
v___f_285_ = lean_alloc_closure((void*)(lp_batteries_Nondet_instAlternativeMonad___redArg___lam__5), 7, 3);
lean_closure_set(v___f_285_, 0, v___f_284_);
lean_closure_set(v___f_285_, 1, v_inst_275_);
lean_closure_set(v___f_285_, 2, v_inst_276_);
v___f_286_ = lean_alloc_closure((void*)(lp_batteries_Nondet_instAlternativeMonad___redArg___lam__7), 7, 3);
lean_closure_set(v___f_286_, 0, v___f_284_);
lean_closure_set(v___f_286_, 1, v_inst_275_);
lean_closure_set(v___f_286_, 2, v_inst_276_);
v___f_287_ = lean_alloc_closure((void*)(lp_batteries_Nondet_instAlternativeMonad___redArg___lam__6), 5, 3);
lean_closure_set(v___f_287_, 0, v_toPure_278_);
lean_closure_set(v___f_287_, 1, v_inst_275_);
lean_closure_set(v___f_287_, 2, v_inst_276_);
lean_inc_ref_n(v___f_287_, 2);
v___f_288_ = lean_alloc_closure((void*)(lp_batteries_Nondet_instAlternativeMonad___redArg___lam__9), 7, 3);
lean_closure_set(v___f_288_, 0, v___f_287_);
lean_closure_set(v___f_288_, 1, v_inst_275_);
lean_closure_set(v___f_288_, 2, v_inst_276_);
v___f_289_ = lean_alloc_closure((void*)(lp_batteries_Nondet_instAlternativeMonad___redArg___lam__12), 7, 3);
lean_closure_set(v___f_289_, 0, v___f_287_);
lean_closure_set(v___f_289_, 1, v_inst_275_);
lean_closure_set(v___f_289_, 2, v_inst_276_);
v___x_290_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_290_, 0, v___f_285_);
lean_ctor_set(v___x_290_, 1, v___f_286_);
if (v_isShared_281_ == 0)
{
lean_ctor_set(v___x_280_, 4, v___f_282_);
lean_ctor_set(v___x_280_, 3, v___f_289_);
lean_ctor_set(v___x_280_, 2, v___f_288_);
lean_ctor_set(v___x_280_, 1, v___f_287_);
lean_ctor_set(v___x_280_, 0, v___x_290_);
v___x_292_ = v___x_280_;
goto v_reusejp_291_;
}
else
{
lean_object* v_reuseFailAlloc_297_; 
v_reuseFailAlloc_297_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_297_, 0, v___x_290_);
lean_ctor_set(v_reuseFailAlloc_297_, 1, v___f_287_);
lean_ctor_set(v_reuseFailAlloc_297_, 2, v___f_288_);
lean_ctor_set(v_reuseFailAlloc_297_, 3, v___f_289_);
lean_ctor_set(v_reuseFailAlloc_297_, 4, v___f_282_);
v___x_292_ = v_reuseFailAlloc_297_;
goto v_reusejp_291_;
}
v_reusejp_291_:
{
lean_object* v___x_293_; lean_object* v___x_294_; lean_object* v___x_295_; lean_object* v___x_296_; 
lean_inc_ref(v_inst_276_);
v___x_293_ = lean_alloc_closure((void*)(lp_batteries_Nondet_nil___boxed), 4, 3);
lean_closure_set(v___x_293_, 0, lean_box(0));
lean_closure_set(v___x_293_, 1, lean_box(0));
lean_closure_set(v___x_293_, 2, v_inst_276_);
v___x_294_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_294_, 0, v___x_292_);
lean_ctor_set(v___x_294_, 1, v___x_293_);
lean_ctor_set(v___x_294_, 2, v___f_283_);
v___x_295_ = lean_alloc_closure((void*)(lp_batteries_Nondet_bind), 8, 4);
lean_closure_set(v___x_295_, 0, lean_box(0));
lean_closure_set(v___x_295_, 1, lean_box(0));
lean_closure_set(v___x_295_, 2, v_inst_275_);
lean_closure_set(v___x_295_, 3, v_inst_276_);
v___x_296_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_296_, 0, v___x_294_);
lean_ctor_set(v___x_296_, 1, v___x_295_);
return v___x_296_;
}
}
}
}
LEAN_EXPORT lean_object* lp_batteries_Nondet_instAlternativeMonad(lean_object* v_00_u03c3_303_, lean_object* v_m_304_, lean_object* v_inst_305_, lean_object* v_inst_306_){
_start:
{
lean_object* v___x_307_; 
v___x_307_ = lp_batteries_Nondet_instAlternativeMonad___redArg(v_inst_305_, v_inst_306_);
return v___x_307_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Nondet_instMonadLift___redArg(lean_object* v_inst_308_, lean_object* v_inst_309_){
_start:
{
lean_object* v___x_310_; 
v___x_310_ = lean_alloc_closure((void*)(lp_batteries_Nondet_singletonM), 6, 4);
lean_closure_set(v___x_310_, 0, lean_box(0));
lean_closure_set(v___x_310_, 1, lean_box(0));
lean_closure_set(v___x_310_, 2, v_inst_308_);
lean_closure_set(v___x_310_, 3, v_inst_309_);
return v___x_310_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Nondet_instMonadLift(lean_object* v_00_u03c3_311_, lean_object* v_m_312_, lean_object* v_inst_313_, lean_object* v_inst_314_){
_start:
{
lean_object* v___x_315_; 
v___x_315_ = lean_alloc_closure((void*)(lp_batteries_Nondet_singletonM), 6, 4);
lean_closure_set(v___x_315_, 0, lean_box(0));
lean_closure_set(v___x_315_, 1, lean_box(0));
lean_closure_set(v___x_315_, 2, v_inst_313_);
lean_closure_set(v___x_315_, 3, v_inst_314_);
return v___x_315_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Nondet_ofListM___redArg___lam__1(lean_object* v_toPure_316_, lean_object* v_toBind_317_, lean_object* v_saveState_318_, lean_object* v_a_319_){
_start:
{
lean_object* v___f_320_; lean_object* v___x_321_; 
v___f_320_ = lean_alloc_closure((void*)(lp_batteries_Nondet_singletonM___redArg___lam__0), 3, 2);
lean_closure_set(v___f_320_, 0, v_a_319_);
lean_closure_set(v___f_320_, 1, v_toPure_316_);
v___x_321_ = lean_apply_4(v_toBind_317_, lean_box(0), lean_box(0), v_saveState_318_, v___f_320_);
return v___x_321_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Nondet_ofListM___redArg___lam__0(lean_object* v_toBind_322_, lean_object* v_x_323_, lean_object* v___f_324_, lean_object* v_____r_325_){
_start:
{
lean_object* v___x_326_; 
v___x_326_ = lean_apply_4(v_toBind_322_, lean_box(0), lean_box(0), v_x_323_, v___f_324_);
return v___x_326_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Nondet_ofListM___redArg___lam__2(lean_object* v_toBind_327_, lean_object* v___f_328_, lean_object* v_restoreState_329_, lean_object* v_s_330_, lean_object* v_x_331_){
_start:
{
lean_object* v___f_332_; lean_object* v___x_333_; lean_object* v___x_334_; 
lean_inc(v_toBind_327_);
v___f_332_ = lean_alloc_closure((void*)(lp_batteries_Nondet_ofListM___redArg___lam__0), 4, 3);
lean_closure_set(v___f_332_, 0, v_toBind_327_);
lean_closure_set(v___f_332_, 1, v_x_331_);
lean_closure_set(v___f_332_, 2, v___f_328_);
v___x_333_ = lean_apply_1(v_restoreState_329_, v_s_330_);
v___x_334_ = lean_apply_4(v_toBind_327_, lean_box(0), lean_box(0), v___x_333_, v___f_332_);
return v___x_334_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Nondet_ofListM___redArg___lam__3(lean_object* v_toBind_335_, lean_object* v___f_336_, lean_object* v_restoreState_337_, lean_object* v_L_338_, lean_object* v_inst_339_, lean_object* v_toPure_340_, lean_object* v_s_341_){
_start:
{
lean_object* v___f_342_; lean_object* v___x_343_; lean_object* v___x_344_; lean_object* v___x_345_; lean_object* v___x_346_; 
v___f_342_ = lean_alloc_closure((void*)(lp_batteries_Nondet_ofListM___redArg___lam__2), 5, 4);
lean_closure_set(v___f_342_, 0, v_toBind_335_);
lean_closure_set(v___f_342_, 1, v___f_336_);
lean_closure_set(v___f_342_, 2, v_restoreState_337_);
lean_closure_set(v___f_342_, 3, v_s_341_);
v___x_343_ = lean_box(0);
v___x_344_ = l_List_mapTR_loop___redArg(v___f_342_, v_L_338_, v___x_343_);
v___x_345_ = lp_batteries_MLList_ofListM___redArg(v_inst_339_, v___x_344_);
v___x_346_ = lean_apply_2(v_toPure_340_, lean_box(0), v___x_345_);
return v___x_346_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Nondet_ofListM___redArg___lam__4(lean_object* v_inst_347_, lean_object* v_toPure_348_, lean_object* v_toBind_349_, lean_object* v_L_350_, lean_object* v_inst_351_, lean_object* v_x_352_){
_start:
{
lean_object* v_saveState_353_; lean_object* v_restoreState_354_; lean_object* v___f_355_; lean_object* v___f_356_; lean_object* v___x_357_; 
v_saveState_353_ = lean_ctor_get(v_inst_347_, 0);
lean_inc_n(v_saveState_353_, 2);
v_restoreState_354_ = lean_ctor_get(v_inst_347_, 1);
lean_inc(v_restoreState_354_);
lean_dec_ref(v_inst_347_);
lean_inc_n(v_toBind_349_, 2);
lean_inc(v_toPure_348_);
v___f_355_ = lean_alloc_closure((void*)(lp_batteries_Nondet_ofListM___redArg___lam__1), 4, 3);
lean_closure_set(v___f_355_, 0, v_toPure_348_);
lean_closure_set(v___f_355_, 1, v_toBind_349_);
lean_closure_set(v___f_355_, 2, v_saveState_353_);
v___f_356_ = lean_alloc_closure((void*)(lp_batteries_Nondet_ofListM___redArg___lam__3), 7, 6);
lean_closure_set(v___f_356_, 0, v_toBind_349_);
lean_closure_set(v___f_356_, 1, v___f_355_);
lean_closure_set(v___f_356_, 2, v_restoreState_354_);
lean_closure_set(v___f_356_, 3, v_L_350_);
lean_closure_set(v___f_356_, 4, v_inst_351_);
lean_closure_set(v___f_356_, 5, v_toPure_348_);
v___x_357_ = lean_apply_4(v_toBind_349_, lean_box(0), lean_box(0), v_saveState_353_, v___f_356_);
return v___x_357_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Nondet_ofListM___redArg(lean_object* v_inst_358_, lean_object* v_inst_359_, lean_object* v_L_360_){
_start:
{
lean_object* v_toApplicative_361_; lean_object* v_toBind_362_; lean_object* v_toPure_363_; lean_object* v___f_364_; lean_object* v___x_365_; 
v_toApplicative_361_ = lean_ctor_get(v_inst_358_, 0);
v_toBind_362_ = lean_ctor_get(v_inst_358_, 1);
v_toPure_363_ = lean_ctor_get(v_toApplicative_361_, 1);
lean_inc_ref(v_inst_358_);
lean_inc(v_toBind_362_);
lean_inc(v_toPure_363_);
v___f_364_ = lean_alloc_closure((void*)(lp_batteries_Nondet_ofListM___redArg___lam__4), 6, 5);
lean_closure_set(v___f_364_, 0, v_inst_359_);
lean_closure_set(v___f_364_, 1, v_toPure_363_);
lean_closure_set(v___f_364_, 2, v_toBind_362_);
lean_closure_set(v___f_364_, 3, v_L_360_);
lean_closure_set(v___f_364_, 4, v_inst_358_);
v___x_365_ = lp_batteries_Nondet_squash___redArg(v_inst_358_, v___f_364_);
return v___x_365_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Nondet_ofListM(lean_object* v_00_u03c3_366_, lean_object* v_m_367_, lean_object* v_inst_368_, lean_object* v_inst_369_, lean_object* v_00_u03b1_370_, lean_object* v_L_371_){
_start:
{
lean_object* v___x_372_; 
v___x_372_ = lp_batteries_Nondet_ofListM___redArg(v_inst_368_, v_inst_369_, v_L_371_);
return v___x_372_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Nondet_ofList___redArg(lean_object* v_inst_373_, lean_object* v_inst_374_, lean_object* v_L_375_){
_start:
{
lean_object* v_toApplicative_376_; lean_object* v_toPure_377_; lean_object* v___x_378_; lean_object* v___x_379_; lean_object* v___x_380_; lean_object* v___x_381_; 
v_toApplicative_376_ = lean_ctor_get(v_inst_373_, 0);
v_toPure_377_ = lean_ctor_get(v_toApplicative_376_, 1);
lean_inc(v_toPure_377_);
v___x_378_ = lean_apply_1(v_toPure_377_, lean_box(0));
v___x_379_ = lean_box(0);
v___x_380_ = l_List_mapTR_loop___redArg(v___x_378_, v_L_375_, v___x_379_);
v___x_381_ = lp_batteries_Nondet_ofListM___redArg(v_inst_373_, v_inst_374_, v___x_380_);
return v___x_381_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Nondet_ofList(lean_object* v_00_u03c3_382_, lean_object* v_m_383_, lean_object* v_inst_384_, lean_object* v_inst_385_, lean_object* v_00_u03b1_386_, lean_object* v_L_387_){
_start:
{
lean_object* v___x_388_; 
v___x_388_ = lp_batteries_Nondet_ofList___redArg(v_inst_384_, v_inst_385_, v_L_387_);
return v___x_388_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Nondet_mapM___redArg___lam__0(lean_object* v_f_389_, lean_object* v_inst_390_, lean_object* v_inst_391_, lean_object* v_a_392_){
_start:
{
lean_object* v___x_393_; lean_object* v___x_394_; 
v___x_393_ = lean_apply_1(v_f_389_, v_a_392_);
v___x_394_ = lp_batteries_Nondet_singletonM___redArg(v_inst_390_, v_inst_391_, v___x_393_);
return v___x_394_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Nondet_mapM___redArg(lean_object* v_inst_395_, lean_object* v_inst_396_, lean_object* v_f_397_, lean_object* v_L_398_){
_start:
{
lean_object* v___f_399_; lean_object* v___x_400_; 
lean_inc_ref(v_inst_396_);
lean_inc_ref(v_inst_395_);
v___f_399_ = lean_alloc_closure((void*)(lp_batteries_Nondet_mapM___redArg___lam__0), 4, 3);
lean_closure_set(v___f_399_, 0, v_f_397_);
lean_closure_set(v___f_399_, 1, v_inst_395_);
lean_closure_set(v___f_399_, 2, v_inst_396_);
v___x_400_ = lp_batteries_Nondet_bind___redArg(v_inst_395_, v_inst_396_, v_L_398_, v___f_399_);
return v___x_400_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Nondet_mapM(lean_object* v_00_u03c3_401_, lean_object* v_m_402_, lean_object* v_inst_403_, lean_object* v_inst_404_, lean_object* v_00_u03b1_405_, lean_object* v_00_u03b2_406_, lean_object* v_f_407_, lean_object* v_L_408_){
_start:
{
lean_object* v___x_409_; 
v___x_409_ = lp_batteries_Nondet_mapM___redArg(v_inst_403_, v_inst_404_, v_f_407_, v_L_408_);
return v___x_409_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Nondet_map___redArg___lam__0(lean_object* v_f_410_, lean_object* v_toPure_411_, lean_object* v_a_412_){
_start:
{
lean_object* v___x_413_; lean_object* v___x_414_; 
v___x_413_ = lean_apply_1(v_f_410_, v_a_412_);
v___x_414_ = lean_apply_2(v_toPure_411_, lean_box(0), v___x_413_);
return v___x_414_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Nondet_map___redArg(lean_object* v_inst_415_, lean_object* v_inst_416_, lean_object* v_f_417_, lean_object* v_L_418_){
_start:
{
lean_object* v_toApplicative_419_; lean_object* v_toPure_420_; lean_object* v___f_421_; lean_object* v___x_422_; 
v_toApplicative_419_ = lean_ctor_get(v_inst_415_, 0);
v_toPure_420_ = lean_ctor_get(v_toApplicative_419_, 1);
lean_inc(v_toPure_420_);
v___f_421_ = lean_alloc_closure((void*)(lp_batteries_Nondet_map___redArg___lam__0), 3, 2);
lean_closure_set(v___f_421_, 0, v_f_417_);
lean_closure_set(v___f_421_, 1, v_toPure_420_);
v___x_422_ = lp_batteries_Nondet_mapM___redArg(v_inst_415_, v_inst_416_, v___f_421_, v_L_418_);
return v___x_422_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Nondet_map(lean_object* v_00_u03c3_423_, lean_object* v_m_424_, lean_object* v_inst_425_, lean_object* v_inst_426_, lean_object* v_00_u03b1_427_, lean_object* v_00_u03b2_428_, lean_object* v_f_429_, lean_object* v_L_430_){
_start:
{
lean_object* v___x_431_; 
v___x_431_ = lp_batteries_Nondet_map___redArg(v_inst_425_, v_inst_426_, v_f_429_, v_L_430_);
return v___x_431_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Nondet_ofOptionM___redArg___lam__0(lean_object* v_toPure_432_, lean_object* v_inst_433_, lean_object* v_inst_434_, lean_object* v_____do__lift_435_){
_start:
{
if (lean_obj_tag(v_____do__lift_435_) == 0)
{
lean_object* v___x_436_; lean_object* v___x_437_; 
lean_dec_ref(v_inst_434_);
lean_dec_ref(v_inst_433_);
v___x_436_ = lean_box(0);
v___x_437_ = lean_apply_2(v_toPure_432_, lean_box(0), v___x_436_);
return v___x_437_;
}
else
{
lean_object* v_val_438_; lean_object* v___x_439_; lean_object* v___x_440_; 
v_val_438_ = lean_ctor_get(v_____do__lift_435_, 0);
lean_inc(v_val_438_);
lean_dec_ref_known(v_____do__lift_435_, 1);
v___x_439_ = lp_batteries_Nondet_singleton___redArg(v_inst_433_, v_inst_434_, v_val_438_);
v___x_440_ = lean_apply_2(v_toPure_432_, lean_box(0), v___x_439_);
return v___x_440_;
}
}
}
LEAN_EXPORT lean_object* lp_batteries_Nondet_ofOptionM___redArg___lam__1(lean_object* v_toBind_441_, lean_object* v_x_442_, lean_object* v___f_443_, lean_object* v_x_444_){
_start:
{
lean_object* v___x_445_; 
v___x_445_ = lean_apply_4(v_toBind_441_, lean_box(0), lean_box(0), v_x_442_, v___f_443_);
return v___x_445_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Nondet_ofOptionM___redArg(lean_object* v_inst_446_, lean_object* v_inst_447_, lean_object* v_x_448_){
_start:
{
lean_object* v_toApplicative_449_; lean_object* v_toBind_450_; lean_object* v_toPure_451_; lean_object* v___f_452_; lean_object* v___f_453_; lean_object* v___x_454_; 
v_toApplicative_449_ = lean_ctor_get(v_inst_446_, 0);
v_toBind_450_ = lean_ctor_get(v_inst_446_, 1);
v_toPure_451_ = lean_ctor_get(v_toApplicative_449_, 1);
lean_inc_ref(v_inst_446_);
lean_inc(v_toPure_451_);
v___f_452_ = lean_alloc_closure((void*)(lp_batteries_Nondet_ofOptionM___redArg___lam__0), 4, 3);
lean_closure_set(v___f_452_, 0, v_toPure_451_);
lean_closure_set(v___f_452_, 1, v_inst_446_);
lean_closure_set(v___f_452_, 2, v_inst_447_);
lean_inc(v_toBind_450_);
v___f_453_ = lean_alloc_closure((void*)(lp_batteries_Nondet_ofOptionM___redArg___lam__1), 4, 3);
lean_closure_set(v___f_453_, 0, v_toBind_450_);
lean_closure_set(v___f_453_, 1, v_x_448_);
lean_closure_set(v___f_453_, 2, v___f_452_);
v___x_454_ = lp_batteries_Nondet_squash___redArg(v_inst_446_, v___f_453_);
return v___x_454_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Nondet_ofOptionM(lean_object* v_00_u03c3_455_, lean_object* v_m_456_, lean_object* v_inst_457_, lean_object* v_inst_458_, lean_object* v_00_u03b1_459_, lean_object* v_x_460_){
_start:
{
lean_object* v___x_461_; 
v___x_461_ = lp_batteries_Nondet_ofOptionM___redArg(v_inst_457_, v_inst_458_, v_x_460_);
return v___x_461_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Nondet_ofOption___redArg(lean_object* v_inst_462_, lean_object* v_inst_463_, lean_object* v_x_464_){
_start:
{
lean_object* v_toApplicative_465_; lean_object* v_toPure_466_; lean_object* v___x_467_; lean_object* v___x_468_; 
v_toApplicative_465_ = lean_ctor_get(v_inst_462_, 0);
v_toPure_466_ = lean_ctor_get(v_toApplicative_465_, 1);
lean_inc(v_toPure_466_);
v___x_467_ = lean_apply_2(v_toPure_466_, lean_box(0), v_x_464_);
v___x_468_ = lp_batteries_Nondet_ofOptionM___redArg(v_inst_462_, v_inst_463_, v___x_467_);
return v___x_468_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Nondet_ofOption(lean_object* v_00_u03c3_469_, lean_object* v_m_470_, lean_object* v_inst_471_, lean_object* v_inst_472_, lean_object* v_00_u03b1_473_, lean_object* v_x_474_){
_start:
{
lean_object* v___x_475_; 
v___x_475_ = lp_batteries_Nondet_ofOption___redArg(v_inst_471_, v_inst_472_, v_x_474_);
return v___x_475_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Nondet_filterMapM___redArg___lam__0(lean_object* v_f_476_, lean_object* v_inst_477_, lean_object* v_inst_478_, lean_object* v_a_479_){
_start:
{
lean_object* v___x_480_; lean_object* v___x_481_; 
v___x_480_ = lean_apply_1(v_f_476_, v_a_479_);
v___x_481_ = lp_batteries_Nondet_ofOptionM___redArg(v_inst_477_, v_inst_478_, v___x_480_);
return v___x_481_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Nondet_filterMapM___redArg(lean_object* v_inst_482_, lean_object* v_inst_483_, lean_object* v_f_484_, lean_object* v_L_485_){
_start:
{
lean_object* v___f_486_; lean_object* v___x_487_; 
lean_inc_ref(v_inst_483_);
lean_inc_ref(v_inst_482_);
v___f_486_ = lean_alloc_closure((void*)(lp_batteries_Nondet_filterMapM___redArg___lam__0), 4, 3);
lean_closure_set(v___f_486_, 0, v_f_484_);
lean_closure_set(v___f_486_, 1, v_inst_482_);
lean_closure_set(v___f_486_, 2, v_inst_483_);
v___x_487_ = lp_batteries_Nondet_bind___redArg(v_inst_482_, v_inst_483_, v_L_485_, v___f_486_);
return v___x_487_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Nondet_filterMapM(lean_object* v_00_u03c3_488_, lean_object* v_m_489_, lean_object* v_inst_490_, lean_object* v_inst_491_, lean_object* v_00_u03b1_492_, lean_object* v_00_u03b2_493_, lean_object* v_f_494_, lean_object* v_L_495_){
_start:
{
lean_object* v___x_496_; 
v___x_496_ = lp_batteries_Nondet_filterMapM___redArg(v_inst_490_, v_inst_491_, v_f_494_, v_L_495_);
return v___x_496_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Nondet_filterMap___redArg___lam__0(lean_object* v_f_497_, lean_object* v_toPure_498_, lean_object* v_a_499_){
_start:
{
lean_object* v___x_500_; lean_object* v___x_501_; 
v___x_500_ = lean_apply_1(v_f_497_, v_a_499_);
v___x_501_ = lean_apply_2(v_toPure_498_, lean_box(0), v___x_500_);
return v___x_501_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Nondet_filterMap___redArg(lean_object* v_inst_502_, lean_object* v_inst_503_, lean_object* v_f_504_, lean_object* v_L_505_){
_start:
{
lean_object* v_toApplicative_506_; lean_object* v_toPure_507_; lean_object* v___f_508_; lean_object* v___x_509_; 
v_toApplicative_506_ = lean_ctor_get(v_inst_502_, 0);
v_toPure_507_ = lean_ctor_get(v_toApplicative_506_, 1);
lean_inc(v_toPure_507_);
v___f_508_ = lean_alloc_closure((void*)(lp_batteries_Nondet_filterMap___redArg___lam__0), 3, 2);
lean_closure_set(v___f_508_, 0, v_f_504_);
lean_closure_set(v___f_508_, 1, v_toPure_507_);
v___x_509_ = lp_batteries_Nondet_filterMapM___redArg(v_inst_502_, v_inst_503_, v___f_508_, v_L_505_);
return v___x_509_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Nondet_filterMap(lean_object* v_00_u03c3_510_, lean_object* v_m_511_, lean_object* v_inst_512_, lean_object* v_inst_513_, lean_object* v_00_u03b1_514_, lean_object* v_00_u03b2_515_, lean_object* v_f_516_, lean_object* v_L_517_){
_start:
{
lean_object* v___x_518_; 
v___x_518_ = lp_batteries_Nondet_filterMap___redArg(v_inst_512_, v_inst_513_, v_f_516_, v_L_517_);
return v___x_518_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Nondet_filterM___redArg___lam__0(lean_object* v_toApplicative_519_, lean_object* v_a_520_, uint8_t v_____do__lift_521_){
_start:
{
if (v_____do__lift_521_ == 0)
{
lean_object* v_toPure_522_; lean_object* v___x_523_; lean_object* v___x_524_; 
lean_dec(v_a_520_);
v_toPure_522_ = lean_ctor_get(v_toApplicative_519_, 1);
lean_inc(v_toPure_522_);
lean_dec_ref(v_toApplicative_519_);
v___x_523_ = lean_box(0);
v___x_524_ = lean_apply_2(v_toPure_522_, lean_box(0), v___x_523_);
return v___x_524_;
}
else
{
lean_object* v_toPure_525_; lean_object* v___x_526_; lean_object* v___x_527_; 
v_toPure_525_ = lean_ctor_get(v_toApplicative_519_, 1);
lean_inc(v_toPure_525_);
lean_dec_ref(v_toApplicative_519_);
v___x_526_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_526_, 0, v_a_520_);
v___x_527_ = lean_apply_2(v_toPure_525_, lean_box(0), v___x_526_);
return v___x_527_;
}
}
}
LEAN_EXPORT lean_object* lp_batteries_Nondet_filterM___redArg___lam__0___boxed(lean_object* v_toApplicative_528_, lean_object* v_a_529_, lean_object* v_____do__lift_530_){
_start:
{
uint8_t v_____do__lift_69__boxed_531_; lean_object* v_res_532_; 
v_____do__lift_69__boxed_531_ = lean_unbox(v_____do__lift_530_);
v_res_532_ = lp_batteries_Nondet_filterM___redArg___lam__0(v_toApplicative_528_, v_a_529_, v_____do__lift_69__boxed_531_);
return v_res_532_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Nondet_filterM___redArg___lam__1(lean_object* v_toApplicative_533_, lean_object* v_p_534_, lean_object* v_toBind_535_, lean_object* v_a_536_){
_start:
{
lean_object* v___f_537_; lean_object* v___x_538_; lean_object* v___x_539_; 
lean_inc(v_a_536_);
v___f_537_ = lean_alloc_closure((void*)(lp_batteries_Nondet_filterM___redArg___lam__0___boxed), 3, 2);
lean_closure_set(v___f_537_, 0, v_toApplicative_533_);
lean_closure_set(v___f_537_, 1, v_a_536_);
v___x_538_ = lean_apply_1(v_p_534_, v_a_536_);
v___x_539_ = lean_apply_4(v_toBind_535_, lean_box(0), lean_box(0), v___x_538_, v___f_537_);
return v___x_539_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Nondet_filterM___redArg(lean_object* v_inst_540_, lean_object* v_inst_541_, lean_object* v_p_542_, lean_object* v_L_543_){
_start:
{
lean_object* v_toApplicative_544_; lean_object* v_toBind_545_; lean_object* v___f_546_; lean_object* v___x_547_; 
v_toApplicative_544_ = lean_ctor_get(v_inst_540_, 0);
v_toBind_545_ = lean_ctor_get(v_inst_540_, 1);
lean_inc(v_toBind_545_);
lean_inc_ref(v_toApplicative_544_);
v___f_546_ = lean_alloc_closure((void*)(lp_batteries_Nondet_filterM___redArg___lam__1), 4, 3);
lean_closure_set(v___f_546_, 0, v_toApplicative_544_);
lean_closure_set(v___f_546_, 1, v_p_542_);
lean_closure_set(v___f_546_, 2, v_toBind_545_);
v___x_547_ = lp_batteries_Nondet_filterMapM___redArg(v_inst_540_, v_inst_541_, v___f_546_, v_L_543_);
return v___x_547_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Nondet_filterM(lean_object* v_00_u03c3_548_, lean_object* v_m_549_, lean_object* v_inst_550_, lean_object* v_inst_551_, lean_object* v_00_u03b1_552_, lean_object* v_p_553_, lean_object* v_L_554_){
_start:
{
lean_object* v___x_555_; 
v___x_555_ = lp_batteries_Nondet_filterM___redArg(v_inst_550_, v_inst_551_, v_p_553_, v_L_554_);
return v___x_555_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Nondet_filter___redArg___lam__0(lean_object* v_p_556_, lean_object* v_toPure_557_, lean_object* v_a_558_){
_start:
{
lean_object* v___x_559_; lean_object* v___x_560_; 
v___x_559_ = lean_apply_1(v_p_556_, v_a_558_);
v___x_560_ = lean_apply_2(v_toPure_557_, lean_box(0), v___x_559_);
return v___x_560_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Nondet_filter___redArg(lean_object* v_inst_561_, lean_object* v_inst_562_, lean_object* v_p_563_, lean_object* v_L_564_){
_start:
{
lean_object* v_toApplicative_565_; lean_object* v_toPure_566_; lean_object* v___f_567_; lean_object* v___x_568_; 
v_toApplicative_565_ = lean_ctor_get(v_inst_561_, 0);
v_toPure_566_ = lean_ctor_get(v_toApplicative_565_, 1);
lean_inc(v_toPure_566_);
v___f_567_ = lean_alloc_closure((void*)(lp_batteries_Nondet_filter___redArg___lam__0), 3, 2);
lean_closure_set(v___f_567_, 0, v_p_563_);
lean_closure_set(v___f_567_, 1, v_toPure_566_);
v___x_568_ = lp_batteries_Nondet_filterM___redArg(v_inst_561_, v_inst_562_, v___f_567_, v_L_564_);
return v___x_568_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Nondet_filter(lean_object* v_00_u03c3_569_, lean_object* v_m_570_, lean_object* v_inst_571_, lean_object* v_inst_572_, lean_object* v_00_u03b1_573_, lean_object* v_p_574_, lean_object* v_L_575_){
_start:
{
lean_object* v___x_576_; 
v___x_576_ = lp_batteries_Nondet_filter___redArg(v_inst_571_, v_inst_572_, v_p_574_, v_L_575_);
return v___x_576_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Nondet_iterate___redArg___lam__0(lean_object* v_f_577_, lean_object* v_a_578_, lean_object* v_inst_579_, lean_object* v_inst_580_, lean_object* v_x_581_){
_start:
{
lean_object* v___x_582_; lean_object* v___x_583_; lean_object* v___x_584_; 
lean_inc(v_f_577_);
v___x_582_ = lean_apply_1(v_f_577_, v_a_578_);
lean_inc_ref(v_inst_580_);
lean_inc_ref(v_inst_579_);
v___x_583_ = lean_alloc_closure((void*)(lp_batteries_Nondet_iterate___redArg), 4, 3);
lean_closure_set(v___x_583_, 0, v_inst_579_);
lean_closure_set(v___x_583_, 1, v_inst_580_);
lean_closure_set(v___x_583_, 2, v_f_577_);
v___x_584_ = lp_batteries_Nondet_bind___redArg(v_inst_579_, v_inst_580_, v___x_582_, v___x_583_);
return v___x_584_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Nondet_iterate___redArg(lean_object* v_inst_585_, lean_object* v_inst_586_, lean_object* v_f_587_, lean_object* v_a_588_){
_start:
{
lean_object* v___f_589_; lean_object* v___x_590_; lean_object* v___x_591_; 
lean_inc_ref(v_inst_586_);
lean_inc_ref_n(v_inst_585_, 2);
lean_inc(v_a_588_);
v___f_589_ = lean_alloc_closure((void*)(lp_batteries_Nondet_iterate___redArg___lam__0), 5, 4);
lean_closure_set(v___f_589_, 0, v_f_587_);
lean_closure_set(v___f_589_, 1, v_a_588_);
lean_closure_set(v___f_589_, 2, v_inst_585_);
lean_closure_set(v___f_589_, 3, v_inst_586_);
v___x_590_ = lp_batteries_Nondet_singleton___redArg(v_inst_585_, v_inst_586_, v_a_588_);
v___x_591_ = lp_batteries_MLList_append___redArg(v_inst_585_, v___x_590_, v___f_589_);
return v___x_591_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Nondet_iterate(lean_object* v_00_u03c3_592_, lean_object* v_m_593_, lean_object* v_inst_594_, lean_object* v_inst_595_, lean_object* v_00_u03b1_596_, lean_object* v_f_597_, lean_object* v_a_598_){
_start:
{
lean_object* v___x_599_; 
v___x_599_ = lp_batteries_Nondet_iterate___redArg(v_inst_594_, v_inst_595_, v_f_597_, v_a_598_);
return v___x_599_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Nondet_toMLList_x27___redArg___lam__0(lean_object* v_x_600_){
_start:
{
lean_object* v_fst_601_; 
v_fst_601_ = lean_ctor_get(v_x_600_, 0);
lean_inc(v_fst_601_);
return v_fst_601_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Nondet_toMLList_x27___redArg___lam__0___boxed(lean_object* v_x_602_){
_start:
{
lean_object* v_res_603_; 
v_res_603_ = lp_batteries_Nondet_toMLList_x27___redArg___lam__0(v_x_602_);
lean_dec_ref(v_x_602_);
return v_res_603_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Nondet_toMLList_x27___redArg(lean_object* v_inst_605_, lean_object* v_L_606_){
_start:
{
lean_object* v___f_607_; lean_object* v___x_608_; 
v___f_607_ = ((lean_object*)(lp_batteries_Nondet_toMLList_x27___redArg___closed__0));
v___x_608_ = lp_batteries_MLList_map___redArg(v_inst_605_, v___f_607_, v_L_606_);
return v___x_608_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Nondet_toMLList_x27(lean_object* v_00_u03c3_609_, lean_object* v_m_610_, lean_object* v_inst_611_, lean_object* v_inst_612_, lean_object* v_00_u03b1_613_, lean_object* v_L_614_){
_start:
{
lean_object* v___x_615_; 
v___x_615_ = lp_batteries_Nondet_toMLList_x27___redArg(v_inst_611_, v_L_614_);
return v___x_615_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Nondet_toMLList_x27___boxed(lean_object* v_00_u03c3_616_, lean_object* v_m_617_, lean_object* v_inst_618_, lean_object* v_inst_619_, lean_object* v_00_u03b1_620_, lean_object* v_L_621_){
_start:
{
lean_object* v_res_622_; 
v_res_622_ = lp_batteries_Nondet_toMLList_x27(v_00_u03c3_616_, v_m_617_, v_inst_618_, v_inst_619_, v_00_u03b1_620_, v_L_621_);
lean_dec_ref(v_inst_619_);
return v_res_622_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Nondet_toList___redArg(lean_object* v_inst_623_, lean_object* v_L_624_){
_start:
{
lean_object* v___x_625_; 
v___x_625_ = lp_batteries_MLList_force___redArg(v_inst_623_, v_L_624_);
return v___x_625_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Nondet_toList(lean_object* v_00_u03c3_626_, lean_object* v_m_627_, lean_object* v_inst_628_, lean_object* v_inst_629_, lean_object* v_00_u03b1_630_, lean_object* v_L_631_){
_start:
{
lean_object* v___x_632_; 
v___x_632_ = lp_batteries_MLList_force___redArg(v_inst_628_, v_L_631_);
return v___x_632_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Nondet_toList___boxed(lean_object* v_00_u03c3_633_, lean_object* v_m_634_, lean_object* v_inst_635_, lean_object* v_inst_636_, lean_object* v_00_u03b1_637_, lean_object* v_L_638_){
_start:
{
lean_object* v_res_639_; 
v_res_639_ = lp_batteries_Nondet_toList(v_00_u03c3_633_, v_m_634_, v_inst_635_, v_inst_636_, v_00_u03b1_637_, v_L_638_);
lean_dec_ref(v_inst_636_);
return v_res_639_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Nondet_toList_x27___redArg(lean_object* v_inst_640_, lean_object* v_L_641_){
_start:
{
lean_object* v___f_642_; lean_object* v___x_643_; lean_object* v___x_644_; 
v___f_642_ = ((lean_object*)(lp_batteries_Nondet_toMLList_x27___redArg___closed__0));
lean_inc_ref(v_inst_640_);
v___x_643_ = lp_batteries_MLList_map___redArg(v_inst_640_, v___f_642_, v_L_641_);
v___x_644_ = lp_batteries_MLList_force___redArg(v_inst_640_, v___x_643_);
return v___x_644_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Nondet_toList_x27(lean_object* v_00_u03c3_645_, lean_object* v_m_646_, lean_object* v_inst_647_, lean_object* v_inst_648_, lean_object* v_00_u03b1_649_, lean_object* v_L_650_){
_start:
{
lean_object* v___x_651_; 
v___x_651_ = lp_batteries_Nondet_toList_x27___redArg(v_inst_647_, v_L_650_);
return v___x_651_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Nondet_toList_x27___boxed(lean_object* v_00_u03c3_652_, lean_object* v_m_653_, lean_object* v_inst_654_, lean_object* v_inst_655_, lean_object* v_00_u03b1_656_, lean_object* v_L_657_){
_start:
{
lean_object* v_res_658_; 
v_res_658_ = lp_batteries_Nondet_toList_x27(v_00_u03c3_652_, v_m_653_, v_inst_654_, v_inst_655_, v_00_u03b1_656_, v_L_657_);
lean_dec_ref(v_inst_655_);
return v_res_658_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Nondet_head___redArg___lam__0(lean_object* v_toPure_659_, lean_object* v_fst_660_, lean_object* v_____r_661_){
_start:
{
lean_object* v___x_662_; 
v___x_662_ = lean_apply_2(v_toPure_659_, lean_box(0), v_fst_660_);
return v___x_662_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Nondet_head___redArg___lam__1(lean_object* v_inst_663_, lean_object* v_toPure_664_, lean_object* v_toBind_665_, lean_object* v_____x_666_){
_start:
{
lean_object* v_fst_667_; lean_object* v_snd_668_; lean_object* v_restoreState_669_; lean_object* v___f_670_; lean_object* v___x_671_; lean_object* v___x_672_; 
v_fst_667_ = lean_ctor_get(v_____x_666_, 0);
lean_inc(v_fst_667_);
v_snd_668_ = lean_ctor_get(v_____x_666_, 1);
lean_inc(v_snd_668_);
lean_dec_ref(v_____x_666_);
v_restoreState_669_ = lean_ctor_get(v_inst_663_, 1);
lean_inc(v_restoreState_669_);
lean_dec_ref(v_inst_663_);
v___f_670_ = lean_alloc_closure((void*)(lp_batteries_Nondet_head___redArg___lam__0), 3, 2);
lean_closure_set(v___f_670_, 0, v_toPure_664_);
lean_closure_set(v___f_670_, 1, v_fst_667_);
v___x_671_ = lean_apply_1(v_restoreState_669_, v_snd_668_);
v___x_672_ = lean_apply_4(v_toBind_665_, lean_box(0), lean_box(0), v___x_671_, v___f_670_);
return v___x_672_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Nondet_head___redArg(lean_object* v_inst_673_, lean_object* v_inst_674_, lean_object* v_L_675_){
_start:
{
lean_object* v___x_676_; lean_object* v_toAlternative_677_; lean_object* v_toApplicative_678_; lean_object* v_toBind_679_; lean_object* v_toPure_680_; lean_object* v___x_681_; lean_object* v___f_682_; lean_object* v___x_683_; 
lean_inc_ref(v_inst_673_);
v___x_676_ = lp_batteries_AlternativeMonad_toMonad___redArg(v_inst_673_);
v_toAlternative_677_ = lean_ctor_get(v_inst_673_, 0);
v_toApplicative_678_ = lean_ctor_get(v_toAlternative_677_, 0);
v_toBind_679_ = lean_ctor_get(v___x_676_, 1);
lean_inc_n(v_toBind_679_, 2);
lean_dec_ref(v___x_676_);
v_toPure_680_ = lean_ctor_get(v_toApplicative_678_, 1);
lean_inc(v_toPure_680_);
v___x_681_ = lp_batteries_MLList_head___redArg(v_inst_673_, v_L_675_);
v___f_682_ = lean_alloc_closure((void*)(lp_batteries_Nondet_head___redArg___lam__1), 4, 3);
lean_closure_set(v___f_682_, 0, v_inst_674_);
lean_closure_set(v___f_682_, 1, v_toPure_680_);
lean_closure_set(v___f_682_, 2, v_toBind_679_);
v___x_683_ = lean_apply_4(v_toBind_679_, lean_box(0), lean_box(0), v___x_681_, v___f_682_);
return v___x_683_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Nondet_head(lean_object* v_00_u03c3_684_, lean_object* v_m_685_, lean_object* v_inst_686_, lean_object* v_inst_687_, lean_object* v_00_u03b1_688_, lean_object* v_L_689_){
_start:
{
lean_object* v___x_690_; 
v___x_690_ = lp_batteries_Nondet_head___redArg(v_inst_686_, v_inst_687_, v_L_689_);
return v___x_690_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Nondet_firstM___redArg(lean_object* v_inst_691_, lean_object* v_inst_692_, lean_object* v_L_693_, lean_object* v_f_694_){
_start:
{
lean_object* v___x_695_; lean_object* v___x_696_; lean_object* v___x_697_; 
lean_inc_ref(v_inst_691_);
v___x_695_ = lp_batteries_AlternativeMonad_toMonad___redArg(v_inst_691_);
lean_inc_ref(v_inst_692_);
v___x_696_ = lp_batteries_Nondet_filterMapM___redArg(v___x_695_, v_inst_692_, v_f_694_, v_L_693_);
v___x_697_ = lp_batteries_Nondet_head___redArg(v_inst_691_, v_inst_692_, v___x_696_);
return v___x_697_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Nondet_firstM(lean_object* v_00_u03c3_698_, lean_object* v_m_699_, lean_object* v_inst_700_, lean_object* v_inst_701_, lean_object* v_00_u03b1_702_, lean_object* v_00_u03b2_703_, lean_object* v_L_704_, lean_object* v_f_705_){
_start:
{
lean_object* v___x_706_; 
v___x_706_ = lp_batteries_Nondet_firstM___redArg(v_inst_700_, v_inst_701_, v_L_704_, v_f_705_);
return v___x_706_;
}
}
LEAN_EXPORT lean_object* lp_batteries_instMonadBacktrackUnitId__batteries___lam__0(lean_object* v___x_707_, lean_object* v_x_708_){
_start:
{
return v___x_707_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_batteries_Batteries_Tactic_Lint_Misc(uint8_t builtin);
lean_object* runtime_initialize_batteries_Batteries_Data_MLList_Basic(uint8_t builtin);
lean_object* runtime_initialize_Lean_Util_MonadBacktrack(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_batteries_Batteries_Control_Nondet_Basic(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_batteries_Batteries_Tactic_Lint_Misc(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_batteries_Batteries_Data_MLList_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_Lean_Util_MonadBacktrack(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_batteries_Batteries_Control_Nondet_Basic(uint8_t builtin) {
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
lean_object* initialize_batteries_Batteries_Tactic_Lint_Misc(uint8_t builtin);
lean_object* initialize_batteries_Batteries_Data_MLList_Basic(uint8_t builtin);
lean_object* initialize_Lean_Util_MonadBacktrack(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_batteries_Batteries_Control_Nondet_Basic(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_batteries_Batteries_Tactic_Lint_Misc(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_batteries_Batteries_Data_MLList_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Lean_Util_MonadBacktrack(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_batteries_Batteries_Control_Nondet_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_batteries_Batteries_Control_Nondet_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_batteries_Batteries_Control_Nondet_Basic(builtin);
}
#ifdef __cplusplus
}
#endif
