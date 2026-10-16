// Lean compiler output
// Module: Mathlib.Data.Set.Finite.Basic
// Imports: public import Init public meta import Init public import Mathlib.Data.Fintype.EquivFin public import Mathlib.Tactic.Nontriviality
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
lean_object* lp_mathlib_Set_toFinset___redArg(lean_object*);
lean_object* lp_mathlib_Multiset_filter___redArg(lean_object*, lean_object*);
lean_object* lp_mathlib_Fintype_subtype___redArg(lean_object*);
lean_object* lp_mathlib_Equiv_refl(lean_object*);
lean_object* lp_mathlib_Equiv_Set_univ(lean_object*);
lean_object* lp_mathlib_Equiv_symm___redArg(lean_object*);
lean_object* lp_mathlib_Fintype_ofEquiv___redArg(lean_object*, lean_object*);
lean_object* lean_nat_add(lean_object*, lean_object*);
lean_object* l_List_range(lean_object*);
uint8_t lp_mathlib_Set_decidableCompl___redArg(uint8_t);
lean_object* lp_mathlib_Finset_image___redArg(lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Finset_map___redArg(lean_object*, lean_object*);
lean_object* lp_mathlib_Multiset_attach___redArg(lean_object*);
lean_object* lp_mathlib_Multiset_ndinsert___redArg(lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Multiset_ndinter___redArg(lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Multiset_sub___redArg(lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Multiset_filterMap___redArg(lean_object*, lean_object*);
lean_object* lp_mathlib_Equiv_subtypeEquiv___redArg(lean_object*);
lean_object* lp_mathlib_Multiset_ndunion___redArg(lean_object*, lean_object*, lean_object*);
static lean_once_cell_t lp_mathlib_Set_Finite_subtypeEquivToFinset___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Set_Finite_subtypeEquivToFinset___closed__0;
static lean_once_cell_t lp_mathlib_Set_Finite_subtypeEquivToFinset___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Set_Finite_subtypeEquivToFinset___closed__1;
LEAN_EXPORT lean_object* lp_mathlib_Set_Finite_subtypeEquivToFinset(lean_object*, lean_object*, lean_object*);
static lean_once_cell_t lp_mathlib_Set_fintypeUniv___redArg___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Set_fintypeUniv___redArg___closed__0;
static lean_once_cell_t lp_mathlib_Set_fintypeUniv___redArg___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Set_fintypeUniv___redArg___closed__1;
LEAN_EXPORT lean_object* lp_mathlib_Set_fintypeUniv___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Set_fintypeUniv(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Set_fintypeTop___redArg___lam__0(lean_object*, lean_object*);
static lean_once_cell_t lp_mathlib_Set_fintypeTop___redArg___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Set_fintypeTop___redArg___closed__0;
LEAN_EXPORT lean_object* lp_mathlib_Set_fintypeTop___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Set_fintypeTop(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Set_fintypeUnion___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Set_fintypeUnion(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Set_fintypeSep___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Set_fintypeSep(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Set_fintypeInter___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Set_fintypeInter(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Set_fintypeInterOfLeft___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Set_fintypeInterOfLeft(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Set_fintypeInterOfRight___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Set_fintypeInterOfRight(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Set_fintypeSubset___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Set_fintypeSubset(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Set_fintypeDiff___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Set_fintypeDiff(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Set_fintypeDiffLeft___redArg___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Set_fintypeDiffLeft___redArg___lam__0___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Set_fintypeDiffLeft___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Set_fintypeDiffLeft(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_once_cell_t lp_mathlib_Set_fintypeEmpty___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Set_fintypeEmpty___closed__0;
LEAN_EXPORT lean_object* lp_mathlib_Set_fintypeEmpty(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Set_fintypeSingleton___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Set_fintypeSingleton(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Set_fintypeInsert___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Set_fintypeInsert(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Set_fintypeInsertOfNotMem___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Set_fintypeInsertOfNotMem(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Set_fintypeInsertOfMem___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Set_fintypeInsertOfMem(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Set_fintypeInsertOfMem___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Set_fintypeInsert_x27___redArg(lean_object*, uint8_t, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Set_fintypeInsert_x27___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Set_fintypeInsert_x27(lean_object*, lean_object*, lean_object*, uint8_t, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Set_fintypeInsert_x27___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Set_fintypeImage___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Set_fintypeImage(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Set_fintypeOfFintypeImage___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Set_fintypeOfFintypeImage(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Set_fintypeOfFintypeImage___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Set_fintypeMap___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Set_fintypeMap(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Set_fintypeLTNat(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Set_fintypeLENat(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Set_fintypeLENat___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Set_Nat_fintypeIio(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Set_fintypeMemFinset___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Set_fintypeMemFinset(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Set_Finite_inhabited(lean_object*);
static lean_object* _init_lp_mathlib_Set_Finite_subtypeEquivToFinset___closed__0(void){
_start:
{
lean_object* v___x_1_; 
v___x_1_ = lp_mathlib_Equiv_refl(lean_box(0));
return v___x_1_;
}
}
static lean_object* _init_lp_mathlib_Set_Finite_subtypeEquivToFinset___closed__1(void){
_start:
{
lean_object* v___x_2_; lean_object* v___x_3_; 
v___x_2_ = lean_obj_once(&lp_mathlib_Set_Finite_subtypeEquivToFinset___closed__0, &lp_mathlib_Set_Finite_subtypeEquivToFinset___closed__0_once, _init_lp_mathlib_Set_Finite_subtypeEquivToFinset___closed__0);
v___x_3_ = lp_mathlib_Equiv_subtypeEquiv___redArg(v___x_2_);
return v___x_3_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Set_Finite_subtypeEquivToFinset(lean_object* v_00_u03b1_4_, lean_object* v_s_5_, lean_object* v_hs_6_){
_start:
{
lean_object* v___x_7_; 
v___x_7_ = lean_obj_once(&lp_mathlib_Set_Finite_subtypeEquivToFinset___closed__1, &lp_mathlib_Set_Finite_subtypeEquivToFinset___closed__1_once, _init_lp_mathlib_Set_Finite_subtypeEquivToFinset___closed__1);
return v___x_7_;
}
}
static lean_object* _init_lp_mathlib_Set_fintypeUniv___redArg___closed__0(void){
_start:
{
lean_object* v___x_8_; 
v___x_8_ = lp_mathlib_Equiv_Set_univ(lean_box(0));
return v___x_8_;
}
}
static lean_object* _init_lp_mathlib_Set_fintypeUniv___redArg___closed__1(void){
_start:
{
lean_object* v___x_9_; lean_object* v___x_10_; 
v___x_9_ = lean_obj_once(&lp_mathlib_Set_fintypeUniv___redArg___closed__0, &lp_mathlib_Set_fintypeUniv___redArg___closed__0_once, _init_lp_mathlib_Set_fintypeUniv___redArg___closed__0);
v___x_10_ = lp_mathlib_Equiv_symm___redArg(v___x_9_);
return v___x_10_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Set_fintypeUniv___redArg(lean_object* v_inst_11_){
_start:
{
lean_object* v___x_12_; lean_object* v___x_13_; 
v___x_12_ = lean_obj_once(&lp_mathlib_Set_fintypeUniv___redArg___closed__1, &lp_mathlib_Set_fintypeUniv___redArg___closed__1_once, _init_lp_mathlib_Set_fintypeUniv___redArg___closed__1);
v___x_13_ = lp_mathlib_Fintype_ofEquiv___redArg(v_inst_11_, v___x_12_);
return v___x_13_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Set_fintypeUniv(lean_object* v_00_u03b1_14_, lean_object* v_inst_15_){
_start:
{
lean_object* v___x_16_; 
v___x_16_ = lp_mathlib_Set_fintypeUniv___redArg(v_inst_15_);
return v___x_16_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Set_fintypeTop___redArg___lam__0(lean_object* v___x_17_, lean_object* v___y_18_){
_start:
{
lean_object* v_toFun_19_; lean_object* v___x_20_; 
v_toFun_19_ = lean_ctor_get(v___x_17_, 0);
lean_inc(v_toFun_19_);
lean_dec_ref(v___x_17_);
v___x_20_ = lean_apply_1(v_toFun_19_, v___y_18_);
return v___x_20_;
}
}
static lean_object* _init_lp_mathlib_Set_fintypeTop___redArg___closed__0(void){
_start:
{
lean_object* v___x_21_; lean_object* v___f_22_; 
v___x_21_ = lean_obj_once(&lp_mathlib_Set_fintypeUniv___redArg___closed__1, &lp_mathlib_Set_fintypeUniv___redArg___closed__1_once, _init_lp_mathlib_Set_fintypeUniv___redArg___closed__1);
v___f_22_ = lean_alloc_closure((void*)(lp_mathlib_Set_fintypeTop___redArg___lam__0), 2, 1);
lean_closure_set(v___f_22_, 0, v___x_21_);
return v___f_22_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Set_fintypeTop___redArg(lean_object* v_inst_23_){
_start:
{
lean_object* v___f_24_; lean_object* v___x_25_; 
v___f_24_ = lean_obj_once(&lp_mathlib_Set_fintypeTop___redArg___closed__0, &lp_mathlib_Set_fintypeTop___redArg___closed__0_once, _init_lp_mathlib_Set_fintypeTop___redArg___closed__0);
v___x_25_ = lp_mathlib_Finset_map___redArg(v___f_24_, v_inst_23_);
return v___x_25_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Set_fintypeTop(lean_object* v_00_u03b1_26_, lean_object* v_inst_27_){
_start:
{
lean_object* v___x_28_; 
v___x_28_ = lp_mathlib_Set_fintypeTop___redArg(v_inst_27_);
return v___x_28_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Set_fintypeUnion___redArg(lean_object* v_inst_29_, lean_object* v_inst_30_, lean_object* v_inst_31_){
_start:
{
lean_object* v___x_32_; lean_object* v___x_33_; lean_object* v___x_34_; lean_object* v___x_35_; 
v___x_32_ = lp_mathlib_Set_toFinset___redArg(v_inst_30_);
v___x_33_ = lp_mathlib_Set_toFinset___redArg(v_inst_31_);
v___x_34_ = lp_mathlib_Multiset_ndunion___redArg(v_inst_29_, v___x_32_, v___x_33_);
v___x_35_ = lp_mathlib_Fintype_subtype___redArg(v___x_34_);
return v___x_35_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Set_fintypeUnion(lean_object* v_00_u03b1_36_, lean_object* v_inst_37_, lean_object* v_s_38_, lean_object* v_t_39_, lean_object* v_inst_40_, lean_object* v_inst_41_){
_start:
{
lean_object* v___x_42_; 
v___x_42_ = lp_mathlib_Set_fintypeUnion___redArg(v_inst_37_, v_inst_40_, v_inst_41_);
return v___x_42_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Set_fintypeSep___redArg(lean_object* v_inst_43_, lean_object* v_inst_44_){
_start:
{
lean_object* v___x_45_; lean_object* v___x_46_; lean_object* v___x_47_; 
v___x_45_ = lp_mathlib_Set_toFinset___redArg(v_inst_43_);
v___x_46_ = lp_mathlib_Multiset_filter___redArg(v_inst_44_, v___x_45_);
v___x_47_ = lp_mathlib_Fintype_subtype___redArg(v___x_46_);
return v___x_47_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Set_fintypeSep(lean_object* v_00_u03b1_48_, lean_object* v_s_49_, lean_object* v_p_50_, lean_object* v_inst_51_, lean_object* v_inst_52_){
_start:
{
lean_object* v___x_53_; 
v___x_53_ = lp_mathlib_Set_fintypeSep___redArg(v_inst_51_, v_inst_52_);
return v___x_53_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Set_fintypeInter___redArg(lean_object* v_inst_54_, lean_object* v_inst_55_, lean_object* v_inst_56_){
_start:
{
lean_object* v___x_57_; lean_object* v___x_58_; lean_object* v___x_59_; lean_object* v___x_60_; 
v___x_57_ = lp_mathlib_Set_toFinset___redArg(v_inst_55_);
v___x_58_ = lp_mathlib_Set_toFinset___redArg(v_inst_56_);
v___x_59_ = lp_mathlib_Multiset_ndinter___redArg(v_inst_54_, v___x_57_, v___x_58_);
v___x_60_ = lp_mathlib_Fintype_subtype___redArg(v___x_59_);
return v___x_60_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Set_fintypeInter(lean_object* v_00_u03b1_61_, lean_object* v_s_62_, lean_object* v_t_63_, lean_object* v_inst_64_, lean_object* v_inst_65_, lean_object* v_inst_66_){
_start:
{
lean_object* v___x_67_; 
v___x_67_ = lp_mathlib_Set_fintypeInter___redArg(v_inst_64_, v_inst_65_, v_inst_66_);
return v___x_67_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Set_fintypeInterOfLeft___redArg(lean_object* v_inst_68_, lean_object* v_inst_69_){
_start:
{
lean_object* v___x_70_; lean_object* v___x_71_; lean_object* v___x_72_; 
v___x_70_ = lp_mathlib_Set_toFinset___redArg(v_inst_68_);
v___x_71_ = lp_mathlib_Multiset_filter___redArg(v_inst_69_, v___x_70_);
v___x_72_ = lp_mathlib_Fintype_subtype___redArg(v___x_71_);
return v___x_72_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Set_fintypeInterOfLeft(lean_object* v_00_u03b1_73_, lean_object* v_s_74_, lean_object* v_t_75_, lean_object* v_inst_76_, lean_object* v_inst_77_){
_start:
{
lean_object* v___x_78_; 
v___x_78_ = lp_mathlib_Set_fintypeInterOfLeft___redArg(v_inst_76_, v_inst_77_);
return v___x_78_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Set_fintypeInterOfRight___redArg(lean_object* v_inst_79_, lean_object* v_inst_80_){
_start:
{
lean_object* v___x_81_; lean_object* v___x_82_; lean_object* v___x_83_; 
v___x_81_ = lp_mathlib_Set_toFinset___redArg(v_inst_79_);
v___x_82_ = lp_mathlib_Multiset_filter___redArg(v_inst_80_, v___x_81_);
v___x_83_ = lp_mathlib_Fintype_subtype___redArg(v___x_82_);
return v___x_83_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Set_fintypeInterOfRight(lean_object* v_00_u03b1_84_, lean_object* v_s_85_, lean_object* v_t_86_, lean_object* v_inst_87_, lean_object* v_inst_88_){
_start:
{
lean_object* v___x_89_; 
v___x_89_ = lp_mathlib_Set_fintypeInterOfRight___redArg(v_inst_87_, v_inst_88_);
return v___x_89_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Set_fintypeSubset___redArg(lean_object* v_inst_90_, lean_object* v_inst_91_){
_start:
{
lean_object* v___x_92_; 
v___x_92_ = lp_mathlib_Set_fintypeInterOfLeft___redArg(v_inst_90_, v_inst_91_);
return v___x_92_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Set_fintypeSubset(lean_object* v_00_u03b1_93_, lean_object* v_s_94_, lean_object* v_t_95_, lean_object* v_inst_96_, lean_object* v_inst_97_, lean_object* v_h_98_){
_start:
{
lean_object* v___x_99_; 
v___x_99_ = lp_mathlib_Set_fintypeInterOfLeft___redArg(v_inst_96_, v_inst_97_);
return v___x_99_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Set_fintypeDiff___redArg(lean_object* v_inst_100_, lean_object* v_inst_101_, lean_object* v_inst_102_){
_start:
{
lean_object* v___x_103_; lean_object* v___x_104_; lean_object* v___x_105_; lean_object* v___x_106_; 
v___x_103_ = lp_mathlib_Set_toFinset___redArg(v_inst_101_);
v___x_104_ = lp_mathlib_Set_toFinset___redArg(v_inst_102_);
v___x_105_ = lp_mathlib_Multiset_sub___redArg(v_inst_100_, v___x_103_, v___x_104_);
v___x_106_ = lp_mathlib_Fintype_subtype___redArg(v___x_105_);
return v___x_106_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Set_fintypeDiff(lean_object* v_00_u03b1_107_, lean_object* v_inst_108_, lean_object* v_s_109_, lean_object* v_t_110_, lean_object* v_inst_111_, lean_object* v_inst_112_){
_start:
{
lean_object* v___x_113_; 
v___x_113_ = lp_mathlib_Set_fintypeDiff___redArg(v_inst_108_, v_inst_111_, v_inst_112_);
return v___x_113_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Set_fintypeDiffLeft___redArg___lam__0(lean_object* v_inst_114_, lean_object* v_a_115_){
_start:
{
lean_object* v___x_116_; uint8_t v___x_117_; uint8_t v___x_118_; 
v___x_116_ = lean_apply_1(v_inst_114_, v_a_115_);
v___x_117_ = lean_unbox(v___x_116_);
v___x_118_ = lp_mathlib_Set_decidableCompl___redArg(v___x_117_);
return v___x_118_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Set_fintypeDiffLeft___redArg___lam__0___boxed(lean_object* v_inst_119_, lean_object* v_a_120_){
_start:
{
uint8_t v_res_121_; lean_object* v_r_122_; 
v_res_121_ = lp_mathlib_Set_fintypeDiffLeft___redArg___lam__0(v_inst_119_, v_a_120_);
v_r_122_ = lean_box(v_res_121_);
return v_r_122_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Set_fintypeDiffLeft___redArg(lean_object* v_inst_123_, lean_object* v_inst_124_){
_start:
{
lean_object* v___f_125_; lean_object* v___x_126_; 
v___f_125_ = lean_alloc_closure((void*)(lp_mathlib_Set_fintypeDiffLeft___redArg___lam__0___boxed), 2, 1);
lean_closure_set(v___f_125_, 0, v_inst_124_);
v___x_126_ = lp_mathlib_Set_fintypeSep___redArg(v_inst_123_, v___f_125_);
return v___x_126_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Set_fintypeDiffLeft(lean_object* v_00_u03b1_127_, lean_object* v_s_128_, lean_object* v_t_129_, lean_object* v_inst_130_, lean_object* v_inst_131_){
_start:
{
lean_object* v___x_132_; 
v___x_132_ = lp_mathlib_Set_fintypeDiffLeft___redArg(v_inst_130_, v_inst_131_);
return v___x_132_;
}
}
static lean_object* _init_lp_mathlib_Set_fintypeEmpty___closed__0(void){
_start:
{
lean_object* v___x_133_; lean_object* v___x_134_; 
v___x_133_ = lean_box(0);
v___x_134_ = lp_mathlib_Fintype_subtype___redArg(v___x_133_);
return v___x_134_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Set_fintypeEmpty(lean_object* v_00_u03b1_135_){
_start:
{
lean_object* v___x_136_; 
v___x_136_ = lean_obj_once(&lp_mathlib_Set_fintypeEmpty___closed__0, &lp_mathlib_Set_fintypeEmpty___closed__0_once, _init_lp_mathlib_Set_fintypeEmpty___closed__0);
return v___x_136_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Set_fintypeSingleton___redArg(lean_object* v_a_137_){
_start:
{
lean_object* v___x_138_; lean_object* v___x_139_; lean_object* v___x_140_; 
v___x_138_ = lean_box(0);
v___x_139_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_139_, 0, v_a_137_);
lean_ctor_set(v___x_139_, 1, v___x_138_);
v___x_140_ = lp_mathlib_Fintype_subtype___redArg(v___x_139_);
return v___x_140_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Set_fintypeSingleton(lean_object* v_00_u03b1_141_, lean_object* v_a_142_){
_start:
{
lean_object* v___x_143_; 
v___x_143_ = lp_mathlib_Set_fintypeSingleton___redArg(v_a_142_);
return v___x_143_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Set_fintypeInsert___redArg(lean_object* v_a_144_, lean_object* v_inst_145_, lean_object* v_inst_146_){
_start:
{
lean_object* v___x_147_; lean_object* v___x_148_; lean_object* v___x_149_; 
v___x_147_ = lp_mathlib_Set_toFinset___redArg(v_inst_146_);
v___x_148_ = lp_mathlib_Multiset_ndinsert___redArg(v_inst_145_, v_a_144_, v___x_147_);
v___x_149_ = lp_mathlib_Fintype_subtype___redArg(v___x_148_);
return v___x_149_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Set_fintypeInsert(lean_object* v_00_u03b1_150_, lean_object* v_a_151_, lean_object* v_s_152_, lean_object* v_inst_153_, lean_object* v_inst_154_){
_start:
{
lean_object* v___x_155_; 
v___x_155_ = lp_mathlib_Set_fintypeInsert___redArg(v_a_151_, v_inst_153_, v_inst_154_);
return v___x_155_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Set_fintypeInsertOfNotMem___redArg(lean_object* v_a_156_, lean_object* v_inst_157_){
_start:
{
lean_object* v___x_158_; lean_object* v___x_159_; lean_object* v___x_160_; 
v___x_158_ = lp_mathlib_Set_toFinset___redArg(v_inst_157_);
v___x_159_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_159_, 0, v_a_156_);
lean_ctor_set(v___x_159_, 1, v___x_158_);
v___x_160_ = lp_mathlib_Fintype_subtype___redArg(v___x_159_);
return v___x_160_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Set_fintypeInsertOfNotMem(lean_object* v_00_u03b1_161_, lean_object* v_a_162_, lean_object* v_s_163_, lean_object* v_inst_164_, lean_object* v_h_165_){
_start:
{
lean_object* v___x_166_; 
v___x_166_ = lp_mathlib_Set_fintypeInsertOfNotMem___redArg(v_a_162_, v_inst_164_);
return v___x_166_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Set_fintypeInsertOfMem___redArg(lean_object* v_inst_167_){
_start:
{
lean_object* v___x_168_; lean_object* v___x_169_; 
v___x_168_ = lp_mathlib_Set_toFinset___redArg(v_inst_167_);
v___x_169_ = lp_mathlib_Fintype_subtype___redArg(v___x_168_);
return v___x_169_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Set_fintypeInsertOfMem(lean_object* v_00_u03b1_170_, lean_object* v_a_171_, lean_object* v_s_172_, lean_object* v_inst_173_, lean_object* v_h_174_){
_start:
{
lean_object* v___x_175_; 
v___x_175_ = lp_mathlib_Set_fintypeInsertOfMem___redArg(v_inst_173_);
return v___x_175_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Set_fintypeInsertOfMem___boxed(lean_object* v_00_u03b1_176_, lean_object* v_a_177_, lean_object* v_s_178_, lean_object* v_inst_179_, lean_object* v_h_180_){
_start:
{
lean_object* v_res_181_; 
v_res_181_ = lp_mathlib_Set_fintypeInsertOfMem(v_00_u03b1_176_, v_a_177_, v_s_178_, v_inst_179_, v_h_180_);
lean_dec(v_a_177_);
return v_res_181_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Set_fintypeInsert_x27___redArg(lean_object* v_a_182_, uint8_t v_inst_183_, lean_object* v_inst_184_){
_start:
{
if (v_inst_183_ == 0)
{
lean_object* v___x_185_; 
v___x_185_ = lp_mathlib_Set_fintypeInsertOfNotMem___redArg(v_a_182_, v_inst_184_);
return v___x_185_;
}
else
{
lean_object* v___x_186_; 
lean_dec(v_a_182_);
v___x_186_ = lp_mathlib_Set_fintypeInsertOfMem___redArg(v_inst_184_);
return v___x_186_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Set_fintypeInsert_x27___redArg___boxed(lean_object* v_a_187_, lean_object* v_inst_188_, lean_object* v_inst_189_){
_start:
{
uint8_t v_inst_10__boxed_190_; lean_object* v_res_191_; 
v_inst_10__boxed_190_ = lean_unbox(v_inst_188_);
v_res_191_ = lp_mathlib_Set_fintypeInsert_x27___redArg(v_a_187_, v_inst_10__boxed_190_, v_inst_189_);
return v_res_191_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Set_fintypeInsert_x27(lean_object* v_00_u03b1_192_, lean_object* v_a_193_, lean_object* v_s_194_, uint8_t v_inst_195_, lean_object* v_inst_196_){
_start:
{
lean_object* v___x_197_; 
v___x_197_ = lp_mathlib_Set_fintypeInsert_x27___redArg(v_a_193_, v_inst_195_, v_inst_196_);
return v___x_197_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Set_fintypeInsert_x27___boxed(lean_object* v_00_u03b1_198_, lean_object* v_a_199_, lean_object* v_s_200_, lean_object* v_inst_201_, lean_object* v_inst_202_){
_start:
{
uint8_t v_inst_20__boxed_203_; lean_object* v_res_204_; 
v_inst_20__boxed_203_ = lean_unbox(v_inst_201_);
v_res_204_ = lp_mathlib_Set_fintypeInsert_x27(v_00_u03b1_198_, v_a_199_, v_s_200_, v_inst_20__boxed_203_, v_inst_202_);
return v_res_204_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Set_fintypeImage___redArg(lean_object* v_inst_205_, lean_object* v_f_206_, lean_object* v_inst_207_){
_start:
{
lean_object* v___x_208_; lean_object* v___x_209_; lean_object* v___x_210_; 
v___x_208_ = lp_mathlib_Set_toFinset___redArg(v_inst_207_);
v___x_209_ = lp_mathlib_Finset_image___redArg(v_inst_205_, v_f_206_, v___x_208_);
v___x_210_ = lp_mathlib_Fintype_subtype___redArg(v___x_209_);
return v___x_210_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Set_fintypeImage(lean_object* v_00_u03b1_211_, lean_object* v_00_u03b2_212_, lean_object* v_inst_213_, lean_object* v_s_214_, lean_object* v_f_215_, lean_object* v_inst_216_){
_start:
{
lean_object* v___x_217_; 
v___x_217_ = lp_mathlib_Set_fintypeImage___redArg(v_inst_213_, v_f_215_, v_inst_216_);
return v___x_217_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Set_fintypeOfFintypeImage___redArg(lean_object* v_g_218_, lean_object* v_inst_219_){
_start:
{
lean_object* v___x_220_; lean_object* v___x_221_; lean_object* v___x_222_; 
v___x_220_ = lp_mathlib_Set_toFinset___redArg(v_inst_219_);
v___x_221_ = lp_mathlib_Multiset_filterMap___redArg(v_g_218_, v___x_220_);
v___x_222_ = lp_mathlib_Fintype_subtype___redArg(v___x_221_);
return v___x_222_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Set_fintypeOfFintypeImage(lean_object* v_00_u03b1_223_, lean_object* v_00_u03b2_224_, lean_object* v_s_225_, lean_object* v_f_226_, lean_object* v_g_227_, lean_object* v_I_228_, lean_object* v_inst_229_){
_start:
{
lean_object* v___x_230_; 
v___x_230_ = lp_mathlib_Set_fintypeOfFintypeImage___redArg(v_g_227_, v_inst_229_);
return v___x_230_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Set_fintypeOfFintypeImage___boxed(lean_object* v_00_u03b1_231_, lean_object* v_00_u03b2_232_, lean_object* v_s_233_, lean_object* v_f_234_, lean_object* v_g_235_, lean_object* v_I_236_, lean_object* v_inst_237_){
_start:
{
lean_object* v_res_238_; 
v_res_238_ = lp_mathlib_Set_fintypeOfFintypeImage(v_00_u03b1_231_, v_00_u03b2_232_, v_s_233_, v_f_234_, v_g_235_, v_I_236_, v_inst_237_);
lean_dec(v_f_234_);
return v_res_238_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Set_fintypeMap___redArg(lean_object* v_inst_239_, lean_object* v_f_240_, lean_object* v_inst_241_){
_start:
{
lean_object* v___x_242_; 
v___x_242_ = lp_mathlib_Set_fintypeImage___redArg(v_inst_239_, v_f_240_, v_inst_241_);
return v___x_242_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Set_fintypeMap(lean_object* v_00_u03b1_243_, lean_object* v_00_u03b2_244_, lean_object* v_inst_245_, lean_object* v_s_246_, lean_object* v_f_247_, lean_object* v_inst_248_){
_start:
{
lean_object* v___x_249_; 
v___x_249_ = lp_mathlib_Set_fintypeImage___redArg(v_inst_245_, v_f_247_, v_inst_248_);
return v___x_249_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Set_fintypeLTNat(lean_object* v_n_250_){
_start:
{
lean_object* v___x_251_; lean_object* v___x_252_; 
v___x_251_ = l_List_range(v_n_250_);
v___x_252_ = lp_mathlib_Fintype_subtype___redArg(v___x_251_);
return v___x_252_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Set_fintypeLENat(lean_object* v_n_253_){
_start:
{
lean_object* v___x_254_; lean_object* v___x_255_; lean_object* v___x_256_; 
v___x_254_ = lean_unsigned_to_nat(1u);
v___x_255_ = lean_nat_add(v_n_253_, v___x_254_);
v___x_256_ = lp_mathlib_Set_fintypeLTNat(v___x_255_);
return v___x_256_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Set_fintypeLENat___boxed(lean_object* v_n_257_){
_start:
{
lean_object* v_res_258_; 
v_res_258_ = lp_mathlib_Set_fintypeLENat(v_n_257_);
lean_dec(v_n_257_);
return v_res_258_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Set_Nat_fintypeIio(lean_object* v_n_259_){
_start:
{
lean_object* v___x_260_; 
v___x_260_ = lp_mathlib_Set_fintypeLTNat(v_n_259_);
return v___x_260_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Set_fintypeMemFinset___redArg(lean_object* v_s_261_){
_start:
{
lean_object* v___x_262_; 
v___x_262_ = lp_mathlib_Multiset_attach___redArg(v_s_261_);
return v___x_262_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Set_fintypeMemFinset(lean_object* v_00_u03b1_263_, lean_object* v_s_264_){
_start:
{
lean_object* v___x_265_; 
v___x_265_ = lp_mathlib_Multiset_attach___redArg(v_s_264_);
return v___x_265_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Set_Finite_inhabited(lean_object* v_00_u03b1_266_){
_start:
{
lean_object* v___x_267_; 
v___x_267_ = lean_box(0);
return v___x_267_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Data_Fintype_EquivFin(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Tactic_Nontriviality(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_Data_Set_Finite_Basic(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Data_Fintype_EquivFin(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Tactic_Nontriviality(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_Data_Set_Finite_Basic(uint8_t builtin) {
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
lean_object* initialize_mathlib_Mathlib_Data_Fintype_EquivFin(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Tactic_Nontriviality(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Data_Set_Finite_Basic(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_Fintype_EquivFin(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Tactic_Nontriviality(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Data_Set_Finite_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_Data_Set_Finite_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_Data_Set_Finite_Basic(builtin);
}
#ifdef __cplusplus
}
#endif
