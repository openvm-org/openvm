// Lean compiler output
// Module: Mathlib.Algebra.Group.Action.End
// Imports: public import Init public meta import Init public import Mathlib.Algebra.Group.Action.Basic public import Mathlib.Algebra.Group.Action.Hom public import Mathlib.Algebra.Group.End
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
lean_object* lp_mathlib_Additive_vadd___redArg___lam__0(lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_AddAction_compHom___redArg(lean_object*, lean_object*);
lean_object* lp_mathlib_Equiv_Perm_permGroup(lean_object*);
lean_object* lp_mathlib_arrowAction___redArg(lean_object*, lean_object*);
lean_object* lp_mathlib_Pi_monoid___redArg(lean_object*);
lean_object* lp_mathlib_MulDistribMulAction_toMonoidHom___redArg___lam__0(lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_MulAction_toPerm___redArg(lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_SMul_comp_smul(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_MulAction_toPerm___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_AddMonoid_toAddZeroClass___redArg(lean_object*);
lean_object* lp_mathlib_Monoid_toMulOneClass___redArg(lean_object*);
lean_object* lp_mathlib_Multiplicative_divInvMonoid___redArg(lean_object*);
lean_object* lp_mathlib_MonoidHom_toAdditiveRight___redArg(lean_object*, lean_object*);
lean_object* lp_mathlib_Multiplicative_smul___redArg___lam__0(lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_instMonoidEnd(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Function_End_applyMulAction___lam__0(lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_Function_End_applyMulAction___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Function_End_applyMulAction___lam__0, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Function_End_applyMulAction___closed__0 = (const lean_object*)&lp_mathlib_Function_End_applyMulAction___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_Function_End_applyMulAction(lean_object*);
static const lean_closure_object lp_mathlib_Function_End_applyAddAction___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*1, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Additive_vadd___redArg___lam__0, .m_arity = 3, .m_num_fixed = 1, .m_objs = {((lean_object*)&lp_mathlib_Function_End_applyMulAction___closed__0_value)} };
static const lean_object* lp_mathlib_Function_End_applyAddAction___closed__0 = (const lean_object*)&lp_mathlib_Function_End_applyAddAction___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_Function_End_applyAddAction(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_Perm_applyMulAction___lam__0(lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_Equiv_Perm_applyMulAction___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Equiv_Perm_applyMulAction___lam__0, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Equiv_Perm_applyMulAction___closed__0 = (const lean_object*)&lp_mathlib_Equiv_Perm_applyMulAction___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_Equiv_Perm_applyMulAction(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulAut_applyMulAction___lam__0(lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_MulAut_applyMulAction___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_MulAut_applyMulAction___lam__0, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_MulAut_applyMulAction___closed__0 = (const lean_object*)&lp_mathlib_MulAut_applyMulAction___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_MulAut_applyMulAction(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulAut_applyMulAction___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddAut_applyAddAction(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddAut_applyAddAction___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulAut_applyMulDistribMulAction(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulAut_applyMulDistribMulAction___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddAut_applyAddDistribAddAction(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddAut_applyAddDistribAddAction___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulAction_toEndHom___redArg___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulAction_toEndHom___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulAction_toEndHom(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulAction_toEndHom___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulAction_ofEndHom___redArg___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulAction_ofEndHom___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulAction_ofEndHom(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulAction_ofEndHom___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
static lean_once_cell_t lp_mathlib_AddAction_toEndHom___redArg___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_AddAction_toEndHom___redArg___closed__0;
static lean_once_cell_t lp_mathlib_AddAction_toEndHom___redArg___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_AddAction_toEndHom___redArg___closed__1;
LEAN_EXPORT lean_object* lp_mathlib_AddAction_toEndHom___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddAction_toEndHom___redArg___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddAction_toEndHom(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddAction_toEndHom___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
static lean_once_cell_t lp_mathlib_AddAction_ofEndHom___redArg___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_AddAction_ofEndHom___redArg___closed__0;
LEAN_EXPORT lean_object* lp_mathlib_AddAction_ofEndHom___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddAction_ofEndHom(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddAction_ofEndHom___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulAction_toPermHom___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulAction_toPermHom(lean_object*, lean_object*, lean_object*, lean_object*);
static lean_once_cell_t lp_mathlib_AddAction_toPermHom___redArg___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_AddAction_toPermHom___redArg___closed__0;
LEAN_EXPORT lean_object* lp_mathlib_AddAction_toPermHom___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddAction_toPermHom(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulDistribMulAction_toMulEquiv___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulDistribMulAction_toMulEquiv___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulDistribMulAction_toMulEquiv(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulDistribMulAction_toMulEquiv___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulDistribMulAction_toMulAut___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulDistribMulAction_toMulAut(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_mulAutArrow___redArg___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_mulAutArrow___redArg___lam__0___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_mulAutArrow___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_mulAutArrow(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Function_End_applyMulAction___lam__0(lean_object* v_x1_1_, lean_object* v_x2_2_){
_start:
{
lean_object* v___x_3_; 
v___x_3_ = lean_apply_1(v_x1_1_, v_x2_2_);
return v___x_3_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_End_applyMulAction(lean_object* v_00_u03b1_5_){
_start:
{
lean_object* v___f_6_; 
v___f_6_ = ((lean_object*)(lp_mathlib_Function_End_applyMulAction___closed__0));
return v___f_6_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_End_applyAddAction(lean_object* v_00_u03b1_9_){
_start:
{
lean_object* v___f_10_; 
v___f_10_ = ((lean_object*)(lp_mathlib_Function_End_applyAddAction___closed__0));
return v___f_10_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_Perm_applyMulAction___lam__0(lean_object* v_f_11_, lean_object* v_a_12_){
_start:
{
lean_object* v_toFun_13_; lean_object* v___x_14_; 
v_toFun_13_ = lean_ctor_get(v_f_11_, 0);
lean_inc(v_toFun_13_);
lean_dec_ref(v_f_11_);
v___x_14_ = lean_apply_1(v_toFun_13_, v_a_12_);
return v___x_14_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_Perm_applyMulAction(lean_object* v_00_u03b1_16_){
_start:
{
lean_object* v___f_17_; 
v___f_17_ = ((lean_object*)(lp_mathlib_Equiv_Perm_applyMulAction___closed__0));
return v___f_17_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulAut_applyMulAction___lam__0(lean_object* v_x1_18_, lean_object* v_x2_19_){
_start:
{
lean_object* v_toFun_20_; lean_object* v___x_21_; 
v_toFun_20_ = lean_ctor_get(v_x1_18_, 0);
lean_inc(v_toFun_20_);
lean_dec_ref(v_x1_18_);
v___x_21_ = lean_apply_1(v_toFun_20_, v_x2_19_);
return v___x_21_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulAut_applyMulAction(lean_object* v_M_23_, lean_object* v_inst_24_){
_start:
{
lean_object* v___f_25_; 
v___f_25_ = ((lean_object*)(lp_mathlib_MulAut_applyMulAction___closed__0));
return v___f_25_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulAut_applyMulAction___boxed(lean_object* v_M_26_, lean_object* v_inst_27_){
_start:
{
lean_object* v_res_28_; 
v_res_28_ = lp_mathlib_MulAut_applyMulAction(v_M_26_, v_inst_27_);
lean_dec_ref(v_inst_27_);
return v_res_28_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddAut_applyAddAction(lean_object* v_M_29_, lean_object* v_inst_30_){
_start:
{
lean_object* v___f_31_; 
v___f_31_ = ((lean_object*)(lp_mathlib_MulAut_applyMulAction___closed__0));
return v___f_31_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddAut_applyAddAction___boxed(lean_object* v_M_32_, lean_object* v_inst_33_){
_start:
{
lean_object* v_res_34_; 
v_res_34_ = lp_mathlib_AddAut_applyAddAction(v_M_32_, v_inst_33_);
lean_dec_ref(v_inst_33_);
return v_res_34_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulAut_applyMulDistribMulAction(lean_object* v_M_35_, lean_object* v_inst_36_){
_start:
{
lean_object* v___f_37_; 
v___f_37_ = ((lean_object*)(lp_mathlib_MulAut_applyMulAction___closed__0));
return v___f_37_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulAut_applyMulDistribMulAction___boxed(lean_object* v_M_38_, lean_object* v_inst_39_){
_start:
{
lean_object* v_res_40_; 
v_res_40_ = lp_mathlib_MulAut_applyMulDistribMulAction(v_M_38_, v_inst_39_);
lean_dec_ref(v_inst_39_);
return v_res_40_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddAut_applyAddDistribAddAction(lean_object* v_M_41_, lean_object* v_inst_42_){
_start:
{
lean_object* v___f_43_; 
v___f_43_ = ((lean_object*)(lp_mathlib_MulAut_applyMulAction___closed__0));
return v___f_43_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddAut_applyAddDistribAddAction___boxed(lean_object* v_M_44_, lean_object* v_inst_45_){
_start:
{
lean_object* v_res_46_; 
v_res_46_ = lp_mathlib_AddAut_applyAddDistribAddAction(v_M_44_, v_inst_45_);
lean_dec_ref(v_inst_45_);
return v_res_46_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulAction_toEndHom___redArg___lam__0(lean_object* v_inst_47_, lean_object* v_x1_48_, lean_object* v_x2_49_){
_start:
{
lean_object* v___x_50_; 
v___x_50_ = lean_apply_2(v_inst_47_, v_x1_48_, v_x2_49_);
return v___x_50_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulAction_toEndHom___redArg(lean_object* v_inst_51_){
_start:
{
lean_object* v___f_52_; 
v___f_52_ = lean_alloc_closure((void*)(lp_mathlib_MulAction_toEndHom___redArg___lam__0), 3, 1);
lean_closure_set(v___f_52_, 0, v_inst_51_);
return v___f_52_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulAction_toEndHom(lean_object* v_M_53_, lean_object* v_00_u03b1_54_, lean_object* v_inst_55_, lean_object* v_inst_56_){
_start:
{
lean_object* v___f_57_; 
v___f_57_ = lean_alloc_closure((void*)(lp_mathlib_MulAction_toEndHom___redArg___lam__0), 3, 1);
lean_closure_set(v___f_57_, 0, v_inst_56_);
return v___f_57_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulAction_toEndHom___boxed(lean_object* v_M_58_, lean_object* v_00_u03b1_59_, lean_object* v_inst_60_, lean_object* v_inst_61_){
_start:
{
lean_object* v_res_62_; 
v_res_62_ = lp_mathlib_MulAction_toEndHom(v_M_58_, v_00_u03b1_59_, v_inst_60_, v_inst_61_);
lean_dec_ref(v_inst_60_);
return v_res_62_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulAction_ofEndHom___redArg___lam__0(lean_object* v_f_63_, lean_object* v___y_64_, lean_object* v___y_65_){
_start:
{
lean_object* v___x_66_; 
v___x_66_ = lean_apply_2(v_f_63_, v___y_64_, v___y_65_);
return v___x_66_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulAction_ofEndHom___redArg(lean_object* v_f_67_){
_start:
{
lean_object* v___f_68_; lean_object* v___f_69_; lean_object* v___x_70_; 
v___f_68_ = lean_alloc_closure((void*)(lp_mathlib_MulAction_ofEndHom___redArg___lam__0), 3, 1);
lean_closure_set(v___f_68_, 0, v_f_67_);
v___f_69_ = ((lean_object*)(lp_mathlib_Function_End_applyMulAction___closed__0));
v___x_70_ = lean_alloc_closure((void*)(lp_mathlib_SMul_comp_smul), 7, 5);
lean_closure_set(v___x_70_, 0, lean_box(0));
lean_closure_set(v___x_70_, 1, lean_box(0));
lean_closure_set(v___x_70_, 2, lean_box(0));
lean_closure_set(v___x_70_, 3, v___f_69_);
lean_closure_set(v___x_70_, 4, v___f_68_);
return v___x_70_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulAction_ofEndHom(lean_object* v_M_71_, lean_object* v_00_u03b1_72_, lean_object* v_inst_73_, lean_object* v_f_74_){
_start:
{
lean_object* v___f_75_; lean_object* v___f_76_; lean_object* v___x_77_; 
v___f_75_ = lean_alloc_closure((void*)(lp_mathlib_MulAction_ofEndHom___redArg___lam__0), 3, 1);
lean_closure_set(v___f_75_, 0, v_f_74_);
v___f_76_ = ((lean_object*)(lp_mathlib_Function_End_applyMulAction___closed__0));
v___x_77_ = lean_alloc_closure((void*)(lp_mathlib_SMul_comp_smul), 7, 5);
lean_closure_set(v___x_77_, 0, lean_box(0));
lean_closure_set(v___x_77_, 1, lean_box(0));
lean_closure_set(v___x_77_, 2, lean_box(0));
lean_closure_set(v___x_77_, 3, v___f_76_);
lean_closure_set(v___x_77_, 4, v___f_75_);
return v___x_77_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulAction_ofEndHom___boxed(lean_object* v_M_78_, lean_object* v_00_u03b1_79_, lean_object* v_inst_80_, lean_object* v_f_81_){
_start:
{
lean_object* v_res_82_; 
v_res_82_ = lp_mathlib_MulAction_ofEndHom(v_M_78_, v_00_u03b1_79_, v_inst_80_, v_f_81_);
lean_dec_ref(v_inst_80_);
return v_res_82_;
}
}
static lean_object* _init_lp_mathlib_AddAction_toEndHom___redArg___closed__0(void){
_start:
{
lean_object* v___x_83_; 
v___x_83_ = lp_mathlib_instMonoidEnd(lean_box(0));
return v___x_83_;
}
}
static lean_object* _init_lp_mathlib_AddAction_toEndHom___redArg___closed__1(void){
_start:
{
lean_object* v___x_84_; lean_object* v___x_85_; 
v___x_84_ = lean_obj_once(&lp_mathlib_AddAction_toEndHom___redArg___closed__0, &lp_mathlib_AddAction_toEndHom___redArg___closed__0_once, _init_lp_mathlib_AddAction_toEndHom___redArg___closed__0);
v___x_85_ = lp_mathlib_Monoid_toMulOneClass___redArg(v___x_84_);
return v___x_85_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddAction_toEndHom___redArg(lean_object* v_inst_86_, lean_object* v_inst_87_){
_start:
{
lean_object* v___x_88_; lean_object* v___x_89_; lean_object* v___x_90_; lean_object* v_toFun_91_; lean_object* v___f_92_; lean_object* v___f_93_; lean_object* v___x_94_; 
v___x_88_ = lp_mathlib_AddMonoid_toAddZeroClass___redArg(v_inst_86_);
v___x_89_ = lean_obj_once(&lp_mathlib_AddAction_toEndHom___redArg___closed__1, &lp_mathlib_AddAction_toEndHom___redArg___closed__1_once, _init_lp_mathlib_AddAction_toEndHom___redArg___closed__1);
v___x_90_ = lp_mathlib_MonoidHom_toAdditiveRight___redArg(v___x_88_, v___x_89_);
lean_dec_ref(v___x_88_);
v_toFun_91_ = lean_ctor_get(v___x_90_, 0);
lean_inc(v_toFun_91_);
lean_dec_ref(v___x_90_);
v___f_92_ = lean_alloc_closure((void*)(lp_mathlib_Multiplicative_smul___redArg___lam__0), 3, 1);
lean_closure_set(v___f_92_, 0, v_inst_87_);
v___f_93_ = lean_alloc_closure((void*)(lp_mathlib_MulAction_toEndHom___redArg___lam__0), 3, 1);
lean_closure_set(v___f_93_, 0, v___f_92_);
v___x_94_ = lean_apply_1(v_toFun_91_, v___f_93_);
return v___x_94_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddAction_toEndHom___redArg___boxed(lean_object* v_inst_95_, lean_object* v_inst_96_){
_start:
{
lean_object* v_res_97_; 
v_res_97_ = lp_mathlib_AddAction_toEndHom___redArg(v_inst_95_, v_inst_96_);
lean_dec_ref(v_inst_95_);
return v_res_97_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddAction_toEndHom(lean_object* v_M_98_, lean_object* v_00_u03b1_99_, lean_object* v_inst_100_, lean_object* v_inst_101_){
_start:
{
lean_object* v___x_102_; 
v___x_102_ = lp_mathlib_AddAction_toEndHom___redArg(v_inst_100_, v_inst_101_);
return v___x_102_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddAction_toEndHom___boxed(lean_object* v_M_103_, lean_object* v_00_u03b1_104_, lean_object* v_inst_105_, lean_object* v_inst_106_){
_start:
{
lean_object* v_res_107_; 
v_res_107_ = lp_mathlib_AddAction_toEndHom(v_M_103_, v_00_u03b1_104_, v_inst_105_, v_inst_106_);
lean_dec_ref(v_inst_105_);
return v_res_107_;
}
}
static lean_object* _init_lp_mathlib_AddAction_ofEndHom___redArg___closed__0(void){
_start:
{
lean_object* v___x_108_; 
v___x_108_ = lp_mathlib_Function_End_applyAddAction(lean_box(0));
return v___x_108_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddAction_ofEndHom___redArg(lean_object* v_f_109_){
_start:
{
lean_object* v___x_110_; lean_object* v___x_111_; 
v___x_110_ = lean_obj_once(&lp_mathlib_AddAction_ofEndHom___redArg___closed__0, &lp_mathlib_AddAction_ofEndHom___redArg___closed__0_once, _init_lp_mathlib_AddAction_ofEndHom___redArg___closed__0);
v___x_111_ = lp_mathlib_AddAction_compHom___redArg(v___x_110_, v_f_109_);
return v___x_111_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddAction_ofEndHom(lean_object* v_M_112_, lean_object* v_00_u03b1_113_, lean_object* v_inst_114_, lean_object* v_f_115_){
_start:
{
lean_object* v___x_116_; lean_object* v___x_117_; 
v___x_116_ = lean_obj_once(&lp_mathlib_AddAction_ofEndHom___redArg___closed__0, &lp_mathlib_AddAction_ofEndHom___redArg___closed__0_once, _init_lp_mathlib_AddAction_ofEndHom___redArg___closed__0);
v___x_117_ = lp_mathlib_AddAction_compHom___redArg(v___x_116_, v_f_115_);
return v___x_117_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddAction_ofEndHom___boxed(lean_object* v_M_118_, lean_object* v_00_u03b1_119_, lean_object* v_inst_120_, lean_object* v_f_121_){
_start:
{
lean_object* v_res_122_; 
v_res_122_ = lp_mathlib_AddAction_ofEndHom(v_M_118_, v_00_u03b1_119_, v_inst_120_, v_f_121_);
lean_dec_ref(v_inst_120_);
return v_res_122_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulAction_toPermHom___redArg(lean_object* v_inst_123_, lean_object* v_inst_124_){
_start:
{
lean_object* v___x_125_; 
v___x_125_ = lean_alloc_closure((void*)(lp_mathlib_MulAction_toPerm___boxed), 5, 4);
lean_closure_set(v___x_125_, 0, lean_box(0));
lean_closure_set(v___x_125_, 1, lean_box(0));
lean_closure_set(v___x_125_, 2, v_inst_123_);
lean_closure_set(v___x_125_, 3, v_inst_124_);
return v___x_125_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulAction_toPermHom(lean_object* v_G_126_, lean_object* v_00_u03b1_127_, lean_object* v_inst_128_, lean_object* v_inst_129_){
_start:
{
lean_object* v___x_130_; 
v___x_130_ = lean_alloc_closure((void*)(lp_mathlib_MulAction_toPerm___boxed), 5, 4);
lean_closure_set(v___x_130_, 0, lean_box(0));
lean_closure_set(v___x_130_, 1, lean_box(0));
lean_closure_set(v___x_130_, 2, v_inst_128_);
lean_closure_set(v___x_130_, 3, v_inst_129_);
return v___x_130_;
}
}
static lean_object* _init_lp_mathlib_AddAction_toPermHom___redArg___closed__0(void){
_start:
{
lean_object* v___x_131_; 
v___x_131_ = lp_mathlib_Equiv_Perm_permGroup(lean_box(0));
return v___x_131_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddAction_toPermHom___redArg(lean_object* v_inst_132_, lean_object* v_inst_133_){
_start:
{
lean_object* v_toAddMonoid_134_; lean_object* v___x_135_; lean_object* v___x_136_; lean_object* v_toMonoid_137_; lean_object* v___x_138_; lean_object* v___x_139_; lean_object* v___x_140_; lean_object* v_toFun_141_; lean_object* v___f_142_; lean_object* v___x_143_; lean_object* v___x_144_; 
v_toAddMonoid_134_ = lean_ctor_get(v_inst_132_, 0);
v___x_135_ = lp_mathlib_AddMonoid_toAddZeroClass___redArg(v_toAddMonoid_134_);
v___x_136_ = lean_obj_once(&lp_mathlib_AddAction_toPermHom___redArg___closed__0, &lp_mathlib_AddAction_toPermHom___redArg___closed__0_once, _init_lp_mathlib_AddAction_toPermHom___redArg___closed__0);
v_toMonoid_137_ = lean_ctor_get(v___x_136_, 0);
v___x_138_ = lp_mathlib_Monoid_toMulOneClass___redArg(v_toMonoid_137_);
v___x_139_ = lp_mathlib_Multiplicative_divInvMonoid___redArg(v_inst_132_);
v___x_140_ = lp_mathlib_MonoidHom_toAdditiveRight___redArg(v___x_135_, v___x_138_);
lean_dec_ref(v___x_138_);
lean_dec_ref(v___x_135_);
v_toFun_141_ = lean_ctor_get(v___x_140_, 0);
lean_inc(v_toFun_141_);
lean_dec_ref(v___x_140_);
v___f_142_ = lean_alloc_closure((void*)(lp_mathlib_Multiplicative_smul___redArg___lam__0), 3, 1);
lean_closure_set(v___f_142_, 0, v_inst_133_);
v___x_143_ = lean_alloc_closure((void*)(lp_mathlib_MulAction_toPerm___boxed), 5, 4);
lean_closure_set(v___x_143_, 0, lean_box(0));
lean_closure_set(v___x_143_, 1, lean_box(0));
lean_closure_set(v___x_143_, 2, v___x_139_);
lean_closure_set(v___x_143_, 3, v___f_142_);
v___x_144_ = lean_apply_1(v_toFun_141_, v___x_143_);
return v___x_144_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddAction_toPermHom(lean_object* v_G_145_, lean_object* v_00_u03b1_146_, lean_object* v_inst_147_, lean_object* v_inst_148_){
_start:
{
lean_object* v___x_149_; 
v___x_149_ = lp_mathlib_AddAction_toPermHom___redArg(v_inst_147_, v_inst_148_);
return v___x_149_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulDistribMulAction_toMulEquiv___redArg(lean_object* v_inst_150_, lean_object* v_inst_151_, lean_object* v_x_152_){
_start:
{
lean_object* v___f_153_; lean_object* v___x_154_; lean_object* v_invFun_155_; lean_object* v___x_157_; uint8_t v_isShared_158_; uint8_t v_isSharedCheck_162_; 
lean_inc(v_x_152_);
lean_inc(v_inst_151_);
v___f_153_ = lean_alloc_closure((void*)(lp_mathlib_MulDistribMulAction_toMonoidHom___redArg___lam__0), 3, 2);
lean_closure_set(v___f_153_, 0, v_inst_151_);
lean_closure_set(v___f_153_, 1, v_x_152_);
v___x_154_ = lp_mathlib_MulAction_toPerm___redArg(v_inst_150_, v_inst_151_, v_x_152_);
v_invFun_155_ = lean_ctor_get(v___x_154_, 1);
v_isSharedCheck_162_ = !lean_is_exclusive(v___x_154_);
if (v_isSharedCheck_162_ == 0)
{
lean_object* v_unused_163_; 
v_unused_163_ = lean_ctor_get(v___x_154_, 0);
lean_dec(v_unused_163_);
v___x_157_ = v___x_154_;
v_isShared_158_ = v_isSharedCheck_162_;
goto v_resetjp_156_;
}
else
{
lean_inc(v_invFun_155_);
lean_dec(v___x_154_);
v___x_157_ = lean_box(0);
v_isShared_158_ = v_isSharedCheck_162_;
goto v_resetjp_156_;
}
v_resetjp_156_:
{
lean_object* v___x_160_; 
if (v_isShared_158_ == 0)
{
lean_ctor_set(v___x_157_, 0, v___f_153_);
v___x_160_ = v___x_157_;
goto v_reusejp_159_;
}
else
{
lean_object* v_reuseFailAlloc_161_; 
v_reuseFailAlloc_161_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_161_, 0, v___f_153_);
lean_ctor_set(v_reuseFailAlloc_161_, 1, v_invFun_155_);
v___x_160_ = v_reuseFailAlloc_161_;
goto v_reusejp_159_;
}
v_reusejp_159_:
{
return v___x_160_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulDistribMulAction_toMulEquiv___redArg___boxed(lean_object* v_inst_164_, lean_object* v_inst_165_, lean_object* v_x_166_){
_start:
{
lean_object* v_res_167_; 
v_res_167_ = lp_mathlib_MulDistribMulAction_toMulEquiv___redArg(v_inst_164_, v_inst_165_, v_x_166_);
lean_dec_ref(v_inst_164_);
return v_res_167_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulDistribMulAction_toMulEquiv(lean_object* v_G_168_, lean_object* v_M_169_, lean_object* v_inst_170_, lean_object* v_inst_171_, lean_object* v_inst_172_, lean_object* v_x_173_){
_start:
{
lean_object* v___x_174_; 
v___x_174_ = lp_mathlib_MulDistribMulAction_toMulEquiv___redArg(v_inst_170_, v_inst_172_, v_x_173_);
return v___x_174_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulDistribMulAction_toMulEquiv___boxed(lean_object* v_G_175_, lean_object* v_M_176_, lean_object* v_inst_177_, lean_object* v_inst_178_, lean_object* v_inst_179_, lean_object* v_x_180_){
_start:
{
lean_object* v_res_181_; 
v_res_181_ = lp_mathlib_MulDistribMulAction_toMulEquiv(v_G_175_, v_M_176_, v_inst_177_, v_inst_178_, v_inst_179_, v_x_180_);
lean_dec_ref(v_inst_178_);
lean_dec_ref(v_inst_177_);
return v_res_181_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulDistribMulAction_toMulAut___redArg(lean_object* v_inst_182_, lean_object* v_inst_183_, lean_object* v_inst_184_){
_start:
{
lean_object* v___x_185_; 
v___x_185_ = lean_alloc_closure((void*)(lp_mathlib_MulDistribMulAction_toMulEquiv___boxed), 6, 5);
lean_closure_set(v___x_185_, 0, lean_box(0));
lean_closure_set(v___x_185_, 1, lean_box(0));
lean_closure_set(v___x_185_, 2, v_inst_182_);
lean_closure_set(v___x_185_, 3, v_inst_183_);
lean_closure_set(v___x_185_, 4, v_inst_184_);
return v___x_185_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulDistribMulAction_toMulAut(lean_object* v_G_186_, lean_object* v_M_187_, lean_object* v_inst_188_, lean_object* v_inst_189_, lean_object* v_inst_190_){
_start:
{
lean_object* v___x_191_; 
v___x_191_ = lean_alloc_closure((void*)(lp_mathlib_MulDistribMulAction_toMulEquiv___boxed), 6, 5);
lean_closure_set(v___x_191_, 0, lean_box(0));
lean_closure_set(v___x_191_, 1, lean_box(0));
lean_closure_set(v___x_191_, 2, v_inst_188_);
lean_closure_set(v___x_191_, 3, v_inst_189_);
lean_closure_set(v___x_191_, 4, v_inst_190_);
return v___x_191_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_mulAutArrow___redArg___lam__0(lean_object* v_inst_192_, lean_object* v_i_193_){
_start:
{
lean_inc_ref(v_inst_192_);
return v_inst_192_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_mulAutArrow___redArg___lam__0___boxed(lean_object* v_inst_194_, lean_object* v_i_195_){
_start:
{
lean_object* v_res_196_; 
v_res_196_ = lp_mathlib_mulAutArrow___redArg___lam__0(v_inst_194_, v_i_195_);
lean_dec(v_i_195_);
lean_dec_ref(v_inst_194_);
return v_res_196_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_mulAutArrow___redArg(lean_object* v_inst_197_, lean_object* v_inst_198_, lean_object* v_inst_199_){
_start:
{
lean_object* v___f_200_; lean_object* v___x_201_; lean_object* v___x_202_; lean_object* v___x_203_; 
v___f_200_ = lean_alloc_closure((void*)(lp_mathlib_mulAutArrow___redArg___lam__0___boxed), 2, 1);
lean_closure_set(v___f_200_, 0, v_inst_199_);
v___x_201_ = lp_mathlib_arrowAction___redArg(v_inst_197_, v_inst_198_);
v___x_202_ = lp_mathlib_Pi_monoid___redArg(v___f_200_);
v___x_203_ = lean_alloc_closure((void*)(lp_mathlib_MulDistribMulAction_toMulEquiv___boxed), 6, 5);
lean_closure_set(v___x_203_, 0, lean_box(0));
lean_closure_set(v___x_203_, 1, lean_box(0));
lean_closure_set(v___x_203_, 2, v_inst_197_);
lean_closure_set(v___x_203_, 3, v___x_202_);
lean_closure_set(v___x_203_, 4, v___x_201_);
return v___x_203_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_mulAutArrow(lean_object* v_G_204_, lean_object* v_M_205_, lean_object* v_A_206_, lean_object* v_inst_207_, lean_object* v_inst_208_, lean_object* v_inst_209_){
_start:
{
lean_object* v___x_210_; 
v___x_210_ = lp_mathlib_mulAutArrow___redArg(v_inst_207_, v_inst_208_, v_inst_209_);
return v___x_210_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Group_Action_Basic(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Group_Action_Hom(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Group_End(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Group_Action_End(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Group_Action_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Group_Action_Hom(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Group_End(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_Algebra_Group_Action_End(uint8_t builtin) {
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
lean_object* initialize_mathlib_Mathlib_Algebra_Group_Action_Basic(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Group_Action_Hom(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Group_End(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Algebra_Group_Action_End(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Group_Action_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Group_Action_Hom(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Group_End(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Group_Action_End(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_Algebra_Group_Action_End(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_Algebra_Group_Action_End(builtin);
}
#ifdef __cplusplus
}
#endif
