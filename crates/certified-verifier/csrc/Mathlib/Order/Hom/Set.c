// Lean compiler output
// Module: Mathlib.Order.Hom.Set
// Imports: public import Init public meta import Init public import Mathlib.Logic.Equiv.Set public import Mathlib.Order.Hom.Basic public import Mathlib.Order.Interval.Set.Defs public import Mathlib.Order.WellFounded public import Mathlib.Tactic.MinImports
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
lean_object* lp_mathlib_Equiv_toEmbedding___redArg___lam__0(lean_object*, lean_object*);
lean_object* lp_mathlib_Equiv_symm___redArg(lean_object*);
lean_object* lp_mathlib_Equiv_subtypeEquivProp(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Equiv_Set_univ(lean_object*);
lean_object* lp_mathlib_Equiv_refl(lean_object*);
lean_object* l_Function_comp(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Set_sumEquiv___lam__0(lean_object*);
static const lean_closure_object lp_mathlib_Set_sumEquiv___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Set_sumEquiv___lam__0, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Set_sumEquiv___closed__0 = (const lean_object*)&lp_mathlib_Set_sumEquiv___closed__0_value;
static const lean_ctor_object lp_mathlib_Set_sumEquiv___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_Set_sumEquiv___closed__0_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_Set_sumEquiv___closed__1 = (const lean_object*)&lp_mathlib_Set_sumEquiv___closed__1_value;
LEAN_EXPORT lean_object* lp_mathlib_Set_sumEquiv(lean_object*, lean_object*);
static lean_once_cell_t lp_mathlib_Set_orderIsoOfEq___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Set_orderIsoOfEq___closed__0;
LEAN_EXPORT lean_object* lp_mathlib_Set_orderIsoOfEq(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Set_orderIsoOfEq___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderIso_Set_congr(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderIso_Set_congr___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderIso_setCongr(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderIso_setCongr___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_once_cell_t lp_mathlib_OrderIso_Set_univ___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_OrderIso_Set_univ___closed__0;
LEAN_EXPORT lean_object* lp_mathlib_OrderIso_Set_univ(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderIso_Set_univ___boxed(lean_object*, lean_object*);
static lean_once_cell_t lp_mathlib_OrderIso_unique__of__wellFoundedLT___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_OrderIso_unique__of__wellFoundedLT___closed__0;
LEAN_EXPORT lean_object* lp_mathlib_OrderIso_unique__of__wellFoundedLT(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderIso_unique__of__wellFoundedLT___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderIso_unique__of__wellFoundedGT(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderIso_unique__of__wellFoundedGT___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderIso_Iic___redArg___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderIso_Iic___redArg___lam__1(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderIso_Iic___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderIso_Iic(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderIso_Iic___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderIso_Ici___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderIso_Ici(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderIso_Ici___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderIso_Icc___redArg___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderIso_Icc___redArg___lam__1(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderIso_Icc___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderIso_Icc(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderIso_Icc___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderIso_compl___redArg___lam__0(lean_object*, lean_object*);
static lean_once_cell_t lp_mathlib_OrderIso_compl___redArg___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_OrderIso_compl___redArg___closed__0;
LEAN_EXPORT lean_object* lp_mathlib_OrderIso_compl___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderIso_compl(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Set_sumEquiv___lam__0(lean_object* v_s_1_){
_start:
{
lean_object* v___x_2_; 
v___x_2_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_2_, 0, lean_box(0));
lean_ctor_set(v___x_2_, 1, lean_box(0));
return v___x_2_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Set_sumEquiv(lean_object* v_00_u03b1_6_, lean_object* v_00_u03b2_7_){
_start:
{
lean_object* v___x_8_; 
v___x_8_ = ((lean_object*)(lp_mathlib_Set_sumEquiv___closed__1));
return v___x_8_;
}
}
static lean_object* _init_lp_mathlib_Set_orderIsoOfEq___closed__0(void){
_start:
{
lean_object* v___x_9_; 
v___x_9_ = lp_mathlib_Equiv_subtypeEquivProp(lean_box(0), lean_box(0), lean_box(0), lean_box(0));
return v___x_9_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Set_orderIsoOfEq(lean_object* v_00_u03b1_10_, lean_object* v_inst_11_, lean_object* v_s_12_, lean_object* v_t_13_, lean_object* v_h_14_){
_start:
{
lean_object* v___x_15_; 
v___x_15_ = lean_obj_once(&lp_mathlib_Set_orderIsoOfEq___closed__0, &lp_mathlib_Set_orderIsoOfEq___closed__0_once, _init_lp_mathlib_Set_orderIsoOfEq___closed__0);
return v___x_15_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Set_orderIsoOfEq___boxed(lean_object* v_00_u03b1_16_, lean_object* v_inst_17_, lean_object* v_s_18_, lean_object* v_t_19_, lean_object* v_h_20_){
_start:
{
lean_object* v_res_21_; 
v_res_21_ = lp_mathlib_Set_orderIsoOfEq(v_00_u03b1_16_, v_inst_17_, v_s_18_, v_t_19_, v_h_20_);
lean_dec_ref(v_inst_17_);
return v_res_21_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderIso_Set_congr(lean_object* v_00_u03b1_22_, lean_object* v_inst_23_, lean_object* v_s_24_, lean_object* v_t_25_, lean_object* v_h_26_){
_start:
{
lean_object* v___x_27_; 
v___x_27_ = lean_obj_once(&lp_mathlib_Set_orderIsoOfEq___closed__0, &lp_mathlib_Set_orderIsoOfEq___closed__0_once, _init_lp_mathlib_Set_orderIsoOfEq___closed__0);
return v___x_27_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderIso_Set_congr___boxed(lean_object* v_00_u03b1_28_, lean_object* v_inst_29_, lean_object* v_s_30_, lean_object* v_t_31_, lean_object* v_h_32_){
_start:
{
lean_object* v_res_33_; 
v_res_33_ = lp_mathlib_OrderIso_Set_congr(v_00_u03b1_28_, v_inst_29_, v_s_30_, v_t_31_, v_h_32_);
lean_dec_ref(v_inst_29_);
return v_res_33_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderIso_setCongr(lean_object* v_00_u03b1_34_, lean_object* v_inst_35_, lean_object* v_s_36_, lean_object* v_t_37_, lean_object* v_h_38_){
_start:
{
lean_object* v___x_39_; 
v___x_39_ = lean_obj_once(&lp_mathlib_Set_orderIsoOfEq___closed__0, &lp_mathlib_Set_orderIsoOfEq___closed__0_once, _init_lp_mathlib_Set_orderIsoOfEq___closed__0);
return v___x_39_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderIso_setCongr___boxed(lean_object* v_00_u03b1_40_, lean_object* v_inst_41_, lean_object* v_s_42_, lean_object* v_t_43_, lean_object* v_h_44_){
_start:
{
lean_object* v_res_45_; 
v_res_45_ = lp_mathlib_OrderIso_setCongr(v_00_u03b1_40_, v_inst_41_, v_s_42_, v_t_43_, v_h_44_);
lean_dec_ref(v_inst_41_);
return v_res_45_;
}
}
static lean_object* _init_lp_mathlib_OrderIso_Set_univ___closed__0(void){
_start:
{
lean_object* v___x_46_; 
v___x_46_ = lp_mathlib_Equiv_Set_univ(lean_box(0));
return v___x_46_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderIso_Set_univ(lean_object* v_00_u03b1_47_, lean_object* v_inst_48_){
_start:
{
lean_object* v___x_49_; 
v___x_49_ = lean_obj_once(&lp_mathlib_OrderIso_Set_univ___closed__0, &lp_mathlib_OrderIso_Set_univ___closed__0_once, _init_lp_mathlib_OrderIso_Set_univ___closed__0);
return v___x_49_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderIso_Set_univ___boxed(lean_object* v_00_u03b1_50_, lean_object* v_inst_51_){
_start:
{
lean_object* v_res_52_; 
v_res_52_ = lp_mathlib_OrderIso_Set_univ(v_00_u03b1_50_, v_inst_51_);
lean_dec_ref(v_inst_51_);
return v_res_52_;
}
}
static lean_object* _init_lp_mathlib_OrderIso_unique__of__wellFoundedLT___closed__0(void){
_start:
{
lean_object* v___x_53_; 
v___x_53_ = lp_mathlib_Equiv_refl(lean_box(0));
return v___x_53_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderIso_unique__of__wellFoundedLT(lean_object* v_00_u03b1_54_, lean_object* v_inst_55_, lean_object* v_inst_56_){
_start:
{
lean_object* v___x_57_; 
v___x_57_ = lean_obj_once(&lp_mathlib_OrderIso_unique__of__wellFoundedLT___closed__0, &lp_mathlib_OrderIso_unique__of__wellFoundedLT___closed__0_once, _init_lp_mathlib_OrderIso_unique__of__wellFoundedLT___closed__0);
return v___x_57_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderIso_unique__of__wellFoundedLT___boxed(lean_object* v_00_u03b1_58_, lean_object* v_inst_59_, lean_object* v_inst_60_){
_start:
{
lean_object* v_res_61_; 
v_res_61_ = lp_mathlib_OrderIso_unique__of__wellFoundedLT(v_00_u03b1_58_, v_inst_59_, v_inst_60_);
lean_dec_ref(v_inst_59_);
return v_res_61_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderIso_unique__of__wellFoundedGT(lean_object* v_00_u03b1_62_, lean_object* v_inst_63_, lean_object* v_inst_64_){
_start:
{
lean_object* v___x_65_; 
v___x_65_ = lean_obj_once(&lp_mathlib_OrderIso_unique__of__wellFoundedLT___closed__0, &lp_mathlib_OrderIso_unique__of__wellFoundedLT___closed__0_once, _init_lp_mathlib_OrderIso_unique__of__wellFoundedLT___closed__0);
return v___x_65_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderIso_unique__of__wellFoundedGT___boxed(lean_object* v_00_u03b1_66_, lean_object* v_inst_67_, lean_object* v_inst_68_){
_start:
{
lean_object* v_res_69_; 
v_res_69_ = lp_mathlib_OrderIso_unique__of__wellFoundedGT(v_00_u03b1_66_, v_inst_67_, v_inst_68_);
lean_dec_ref(v_inst_67_);
return v_res_69_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderIso_Iic___redArg___lam__0(lean_object* v_e_70_, lean_object* v_y_71_){
_start:
{
lean_object* v___x_72_; 
v___x_72_ = lp_mathlib_Equiv_toEmbedding___redArg___lam__0(v_e_70_, v_y_71_);
return v___x_72_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderIso_Iic___redArg___lam__1(lean_object* v_e_73_, lean_object* v_y_74_){
_start:
{
lean_object* v___x_75_; lean_object* v___x_76_; 
v___x_75_ = lp_mathlib_Equiv_symm___redArg(v_e_73_);
v___x_76_ = lp_mathlib_Equiv_toEmbedding___redArg___lam__0(v___x_75_, v_y_74_);
return v___x_76_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderIso_Iic___redArg(lean_object* v_e_77_){
_start:
{
lean_object* v___f_78_; lean_object* v___f_79_; lean_object* v___x_80_; 
lean_inc_ref(v_e_77_);
v___f_78_ = lean_alloc_closure((void*)(lp_mathlib_OrderIso_Iic___redArg___lam__0), 2, 1);
lean_closure_set(v___f_78_, 0, v_e_77_);
v___f_79_ = lean_alloc_closure((void*)(lp_mathlib_OrderIso_Iic___redArg___lam__1), 2, 1);
lean_closure_set(v___f_79_, 0, v_e_77_);
v___x_80_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_80_, 0, v___f_78_);
lean_ctor_set(v___x_80_, 1, v___f_79_);
return v___x_80_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderIso_Iic(lean_object* v_00_u03b1_81_, lean_object* v_00_u03b2_82_, lean_object* v_inst_83_, lean_object* v_inst_84_, lean_object* v_e_85_, lean_object* v_x_86_){
_start:
{
lean_object* v___x_87_; 
v___x_87_ = lp_mathlib_OrderIso_Iic___redArg(v_e_85_);
return v___x_87_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderIso_Iic___boxed(lean_object* v_00_u03b1_88_, lean_object* v_00_u03b2_89_, lean_object* v_inst_90_, lean_object* v_inst_91_, lean_object* v_e_92_, lean_object* v_x_93_){
_start:
{
lean_object* v_res_94_; 
v_res_94_ = lp_mathlib_OrderIso_Iic(v_00_u03b1_88_, v_00_u03b2_89_, v_inst_90_, v_inst_91_, v_e_92_, v_x_93_);
lean_dec(v_x_93_);
lean_dec_ref(v_inst_91_);
lean_dec_ref(v_inst_90_);
return v_res_94_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderIso_Ici___redArg(lean_object* v_e_95_){
_start:
{
lean_object* v___f_96_; lean_object* v___f_97_; lean_object* v___x_98_; 
lean_inc_ref(v_e_95_);
v___f_96_ = lean_alloc_closure((void*)(lp_mathlib_OrderIso_Iic___redArg___lam__0), 2, 1);
lean_closure_set(v___f_96_, 0, v_e_95_);
v___f_97_ = lean_alloc_closure((void*)(lp_mathlib_OrderIso_Iic___redArg___lam__1), 2, 1);
lean_closure_set(v___f_97_, 0, v_e_95_);
v___x_98_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_98_, 0, v___f_96_);
lean_ctor_set(v___x_98_, 1, v___f_97_);
return v___x_98_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderIso_Ici(lean_object* v_00_u03b1_99_, lean_object* v_00_u03b2_100_, lean_object* v_inst_101_, lean_object* v_inst_102_, lean_object* v_e_103_, lean_object* v_x_104_){
_start:
{
lean_object* v___x_105_; 
v___x_105_ = lp_mathlib_OrderIso_Ici___redArg(v_e_103_);
return v___x_105_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderIso_Ici___boxed(lean_object* v_00_u03b1_106_, lean_object* v_00_u03b2_107_, lean_object* v_inst_108_, lean_object* v_inst_109_, lean_object* v_e_110_, lean_object* v_x_111_){
_start:
{
lean_object* v_res_112_; 
v_res_112_ = lp_mathlib_OrderIso_Ici(v_00_u03b1_106_, v_00_u03b2_107_, v_inst_108_, v_inst_109_, v_e_110_, v_x_111_);
lean_dec(v_x_111_);
lean_dec_ref(v_inst_109_);
lean_dec_ref(v_inst_108_);
return v_res_112_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderIso_Icc___redArg___lam__0(lean_object* v_e_113_, lean_object* v_z_114_){
_start:
{
lean_object* v___x_115_; 
v___x_115_ = lp_mathlib_Equiv_toEmbedding___redArg___lam__0(v_e_113_, v_z_114_);
return v___x_115_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderIso_Icc___redArg___lam__1(lean_object* v_e_116_, lean_object* v_z_117_){
_start:
{
lean_object* v___x_118_; lean_object* v___x_119_; 
v___x_118_ = lp_mathlib_Equiv_symm___redArg(v_e_116_);
v___x_119_ = lp_mathlib_Equiv_toEmbedding___redArg___lam__0(v___x_118_, v_z_117_);
return v___x_119_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderIso_Icc___redArg(lean_object* v_e_120_){
_start:
{
lean_object* v___f_121_; lean_object* v___f_122_; lean_object* v___x_123_; 
lean_inc_ref(v_e_120_);
v___f_121_ = lean_alloc_closure((void*)(lp_mathlib_OrderIso_Icc___redArg___lam__0), 2, 1);
lean_closure_set(v___f_121_, 0, v_e_120_);
v___f_122_ = lean_alloc_closure((void*)(lp_mathlib_OrderIso_Icc___redArg___lam__1), 2, 1);
lean_closure_set(v___f_122_, 0, v_e_120_);
v___x_123_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_123_, 0, v___f_121_);
lean_ctor_set(v___x_123_, 1, v___f_122_);
return v___x_123_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderIso_Icc(lean_object* v_00_u03b1_124_, lean_object* v_00_u03b2_125_, lean_object* v_inst_126_, lean_object* v_inst_127_, lean_object* v_e_128_, lean_object* v_x_129_, lean_object* v_y_130_){
_start:
{
lean_object* v___x_131_; 
v___x_131_ = lp_mathlib_OrderIso_Icc___redArg(v_e_128_);
return v___x_131_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderIso_Icc___boxed(lean_object* v_00_u03b1_132_, lean_object* v_00_u03b2_133_, lean_object* v_inst_134_, lean_object* v_inst_135_, lean_object* v_e_136_, lean_object* v_x_137_, lean_object* v_y_138_){
_start:
{
lean_object* v_res_139_; 
v_res_139_ = lp_mathlib_OrderIso_Icc(v_00_u03b1_132_, v_00_u03b2_133_, v_inst_134_, v_inst_135_, v_e_136_, v_x_137_, v_y_138_);
lean_dec(v_y_138_);
lean_dec(v_x_137_);
lean_dec_ref(v_inst_135_);
lean_dec_ref(v_inst_134_);
return v_res_139_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderIso_compl___redArg___lam__0(lean_object* v___x_140_, lean_object* v___y_141_){
_start:
{
lean_object* v_toFun_142_; lean_object* v___x_143_; 
v_toFun_142_ = lean_ctor_get(v___x_140_, 0);
lean_inc(v_toFun_142_);
lean_dec_ref(v___x_140_);
v___x_143_ = lean_apply_1(v_toFun_142_, v___y_141_);
return v___x_143_;
}
}
static lean_object* _init_lp_mathlib_OrderIso_compl___redArg___closed__0(void){
_start:
{
lean_object* v___x_144_; lean_object* v___f_145_; 
v___x_144_ = lean_obj_once(&lp_mathlib_OrderIso_unique__of__wellFoundedLT___closed__0, &lp_mathlib_OrderIso_unique__of__wellFoundedLT___closed__0_once, _init_lp_mathlib_OrderIso_unique__of__wellFoundedLT___closed__0);
v___f_145_ = lean_alloc_closure((void*)(lp_mathlib_OrderIso_compl___redArg___lam__0), 2, 1);
lean_closure_set(v___f_145_, 0, v___x_144_);
return v___f_145_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderIso_compl___redArg(lean_object* v_inst_146_){
_start:
{
lean_object* v_toCompl_147_; lean_object* v___f_148_; lean_object* v___x_149_; lean_object* v___x_150_; lean_object* v___x_151_; 
v_toCompl_147_ = lean_ctor_get(v_inst_146_, 1);
lean_inc_n(v_toCompl_147_, 2);
lean_dec_ref(v_inst_146_);
v___f_148_ = lean_obj_once(&lp_mathlib_OrderIso_compl___redArg___closed__0, &lp_mathlib_OrderIso_compl___redArg___closed__0_once, _init_lp_mathlib_OrderIso_compl___redArg___closed__0);
v___x_149_ = lean_alloc_closure((void*)(l_Function_comp), 6, 5);
lean_closure_set(v___x_149_, 0, lean_box(0));
lean_closure_set(v___x_149_, 1, lean_box(0));
lean_closure_set(v___x_149_, 2, lean_box(0));
lean_closure_set(v___x_149_, 3, v___f_148_);
lean_closure_set(v___x_149_, 4, v_toCompl_147_);
v___x_150_ = lean_alloc_closure((void*)(l_Function_comp), 6, 5);
lean_closure_set(v___x_150_, 0, lean_box(0));
lean_closure_set(v___x_150_, 1, lean_box(0));
lean_closure_set(v___x_150_, 2, lean_box(0));
lean_closure_set(v___x_150_, 3, v_toCompl_147_);
lean_closure_set(v___x_150_, 4, v___f_148_);
v___x_151_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_151_, 0, v___x_149_);
lean_ctor_set(v___x_151_, 1, v___x_150_);
return v___x_151_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderIso_compl(lean_object* v_00_u03b1_152_, lean_object* v_inst_153_){
_start:
{
lean_object* v___x_154_; 
v___x_154_ = lp_mathlib_OrderIso_compl___redArg(v_inst_153_);
return v___x_154_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Logic_Equiv_Set(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Order_Hom_Basic(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Order_Interval_Set_Defs(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Order_WellFounded(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Tactic_MinImports(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_Order_Hom_Set(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Logic_Equiv_Set(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Order_Hom_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Order_Interval_Set_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Order_WellFounded(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Tactic_MinImports(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_Order_Hom_Set(uint8_t builtin) {
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
lean_object* initialize_mathlib_Mathlib_Logic_Equiv_Set(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Order_Hom_Basic(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Order_Interval_Set_Defs(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Order_WellFounded(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Tactic_MinImports(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Order_Hom_Set(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Logic_Equiv_Set(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Order_Hom_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Order_Interval_Set_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Order_WellFounded(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Tactic_MinImports(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Order_Hom_Set(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_Order_Hom_Set(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_Order_Hom_Set(builtin);
}
#ifdef __cplusplus
}
#endif
