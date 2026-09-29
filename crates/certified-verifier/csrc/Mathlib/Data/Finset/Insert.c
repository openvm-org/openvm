// Lean compiler output
// Module: Mathlib.Data.Finset.Insert
// Imports: public import Init public meta import Init public import Mathlib.Data.Finset.Attr public import Mathlib.Data.Finset.Dedup public import Mathlib.Data.Finset.Empty public import Mathlib.Data.Multiset.FinsetOps public import Mathlib.Util.Delaborators
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
lean_object* lp_mathlib_Multiset_ndinsert___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Finset_instSingleton___lam__0(lean_object*);
static const lean_closure_object lp_mathlib_Finset_instSingleton___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Finset_instSingleton___lam__0, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Finset_instSingleton___closed__0 = (const lean_object*)&lp_mathlib_Finset_instSingleton___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_Finset_instSingleton(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Finset_instUniqueOfIsEmpty(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Finset_instUniqueSubtypeMemSingleton___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Finset_instUniqueSubtypeMemSingleton___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Finset_instUniqueSubtypeMemSingleton(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Finset_instUniqueSubtypeMemSingleton___boxed(lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Finset_Nontrivial_instDecidablePred___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Finset_Nontrivial_instDecidablePred___redArg___boxed(lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Finset_Nontrivial_instDecidablePred(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Finset_Nontrivial_instDecidablePred___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Finset_cons___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Finset_cons(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Finset_consPiProd___redArg___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Finset_consPiProd___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Finset_consPiProd(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Finset_consPiProd___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Finset_prodPiCons___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Finset_prodPiCons(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Finset_prodPiCons___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Finset_consPiProdEquiv___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Finset_consPiProdEquiv(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Finset_instInsert___redArg___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Finset_instInsert___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Finset_instInsert(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Finset_subtypeInsertEquivOption___redArg___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Finset_subtypeInsertEquivOption___redArg___lam__1(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Finset_subtypeInsertEquivOption___redArg___lam__1___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Finset_subtypeInsertEquivOption___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Finset_subtypeInsertEquivOption(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Finset_subtypeInsertEquivOption___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Finset_insertPiProd___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Finset_insertPiProd(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Finset_insertPiProd___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Finset_prodPiInsert___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Finset_prodPiInsert(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Finset_prodPiInsert___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Finset_insertPiProdEquiv___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Finset_insertPiProdEquiv(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Finset_instSingleton___lam__0(lean_object* v_a_1_){
_start:
{
lean_object* v___x_2_; lean_object* v___x_3_; 
v___x_2_ = lean_box(0);
v___x_3_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_3_, 0, v_a_1_);
lean_ctor_set(v___x_3_, 1, v___x_2_);
return v___x_3_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Finset_instSingleton(lean_object* v_00_u03b1_5_){
_start:
{
lean_object* v___f_6_; 
v___f_6_ = ((lean_object*)(lp_mathlib_Finset_instSingleton___closed__0));
return v___f_6_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Finset_instUniqueOfIsEmpty(lean_object* v_00_u03b1_7_, lean_object* v_inst_8_){
_start:
{
lean_object* v___x_9_; 
v___x_9_ = lean_box(0);
return v___x_9_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Finset_instUniqueSubtypeMemSingleton___redArg(lean_object* v_i_10_){
_start:
{
lean_inc(v_i_10_);
return v_i_10_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Finset_instUniqueSubtypeMemSingleton___redArg___boxed(lean_object* v_i_11_){
_start:
{
lean_object* v_res_12_; 
v_res_12_ = lp_mathlib_Finset_instUniqueSubtypeMemSingleton___redArg(v_i_11_);
lean_dec(v_i_11_);
return v_res_12_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Finset_instUniqueSubtypeMemSingleton(lean_object* v_00_u03b1_13_, lean_object* v_i_14_){
_start:
{
lean_inc(v_i_14_);
return v_i_14_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Finset_instUniqueSubtypeMemSingleton___boxed(lean_object* v_00_u03b1_15_, lean_object* v_i_16_){
_start:
{
lean_object* v_res_17_; 
v_res_17_ = lp_mathlib_Finset_instUniqueSubtypeMemSingleton(v_00_u03b1_15_, v_i_16_);
lean_dec(v_i_16_);
return v_res_17_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Finset_Nontrivial_instDecidablePred___redArg(lean_object* v_s_18_){
_start:
{
if (lean_obj_tag(v_s_18_) == 0)
{
uint8_t v___x_19_; 
v___x_19_ = 0;
return v___x_19_;
}
else
{
lean_object* v_tail_20_; 
v_tail_20_ = lean_ctor_get(v_s_18_, 1);
if (lean_obj_tag(v_tail_20_) == 0)
{
uint8_t v___x_21_; 
v___x_21_ = 0;
return v___x_21_;
}
else
{
uint8_t v___x_22_; 
v___x_22_ = 1;
return v___x_22_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Finset_Nontrivial_instDecidablePred___redArg___boxed(lean_object* v_s_23_){
_start:
{
uint8_t v_res_24_; lean_object* v_r_25_; 
v_res_24_ = lp_mathlib_Finset_Nontrivial_instDecidablePred___redArg(v_s_23_);
lean_dec(v_s_23_);
v_r_25_ = lean_box(v_res_24_);
return v_r_25_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Finset_Nontrivial_instDecidablePred(lean_object* v_00_u03b1_26_, lean_object* v_s_27_){
_start:
{
uint8_t v___x_28_; 
v___x_28_ = lp_mathlib_Finset_Nontrivial_instDecidablePred___redArg(v_s_27_);
return v___x_28_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Finset_Nontrivial_instDecidablePred___boxed(lean_object* v_00_u03b1_29_, lean_object* v_s_30_){
_start:
{
uint8_t v_res_31_; lean_object* v_r_32_; 
v_res_31_ = lp_mathlib_Finset_Nontrivial_instDecidablePred(v_00_u03b1_29_, v_s_30_);
lean_dec(v_s_30_);
v_r_32_ = lean_box(v_res_31_);
return v_r_32_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Finset_cons___redArg(lean_object* v_a_33_, lean_object* v_s_34_){
_start:
{
lean_object* v___x_35_; 
v___x_35_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_35_, 0, v_a_33_);
lean_ctor_set(v___x_35_, 1, v_s_34_);
return v___x_35_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Finset_cons(lean_object* v_00_u03b1_36_, lean_object* v_a_37_, lean_object* v_s_38_, lean_object* v_h_39_){
_start:
{
lean_object* v___x_40_; 
v___x_40_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_40_, 0, v_a_37_);
lean_ctor_set(v___x_40_, 1, v_s_38_);
return v___x_40_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Finset_consPiProd___redArg___lam__0(lean_object* v_x_41_, lean_object* v_i_42_, lean_object* v_hi_43_){
_start:
{
lean_object* v___x_44_; 
v___x_44_ = lean_apply_2(v_x_41_, v_i_42_, lean_box(0));
return v___x_44_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Finset_consPiProd___redArg(lean_object* v_a_45_, lean_object* v_x_46_){
_start:
{
lean_object* v___f_47_; lean_object* v___x_48_; lean_object* v___x_49_; 
lean_inc(v_x_46_);
v___f_47_ = lean_alloc_closure((void*)(lp_mathlib_Finset_consPiProd___redArg___lam__0), 3, 1);
lean_closure_set(v___f_47_, 0, v_x_46_);
v___x_48_ = lean_apply_2(v_x_46_, v_a_45_, lean_box(0));
v___x_49_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_49_, 0, v___x_48_);
lean_ctor_set(v___x_49_, 1, v___f_47_);
return v___x_49_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Finset_consPiProd(lean_object* v_00_u03b1_50_, lean_object* v_s_51_, lean_object* v_a_52_, lean_object* v_f_53_, lean_object* v_has_54_, lean_object* v_x_55_){
_start:
{
lean_object* v___x_56_; 
v___x_56_ = lp_mathlib_Finset_consPiProd___redArg(v_a_52_, v_x_55_);
return v___x_56_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Finset_consPiProd___boxed(lean_object* v_00_u03b1_57_, lean_object* v_s_58_, lean_object* v_a_59_, lean_object* v_f_60_, lean_object* v_has_61_, lean_object* v_x_62_){
_start:
{
lean_object* v_res_63_; 
v_res_63_ = lp_mathlib_Finset_consPiProd(v_00_u03b1_57_, v_s_58_, v_a_59_, v_f_60_, v_has_61_, v_x_62_);
lean_dec(v_s_58_);
return v_res_63_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Finset_prodPiCons___redArg(lean_object* v_inst_64_, lean_object* v_a_65_, lean_object* v_x_66_, lean_object* v_i_67_){
_start:
{
lean_object* v___x_68_; uint8_t v___x_69_; 
lean_inc(v_i_67_);
v___x_68_ = lean_apply_2(v_inst_64_, v_i_67_, v_a_65_);
v___x_69_ = lean_unbox(v___x_68_);
if (v___x_69_ == 0)
{
lean_object* v_snd_70_; lean_object* v___x_71_; 
v_snd_70_ = lean_ctor_get(v_x_66_, 1);
lean_inc(v_snd_70_);
lean_dec_ref(v_x_66_);
v___x_71_ = lean_apply_2(v_snd_70_, v_i_67_, lean_box(0));
return v___x_71_;
}
else
{
lean_object* v_fst_72_; 
lean_dec(v_i_67_);
v_fst_72_ = lean_ctor_get(v_x_66_, 0);
lean_inc(v_fst_72_);
lean_dec_ref(v_x_66_);
return v_fst_72_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Finset_prodPiCons(lean_object* v_00_u03b1_73_, lean_object* v_s_74_, lean_object* v_inst_75_, lean_object* v_f_76_, lean_object* v_a_77_, lean_object* v_has_78_, lean_object* v_x_79_, lean_object* v_i_80_, lean_object* v_hi_81_){
_start:
{
lean_object* v___x_82_; 
v___x_82_ = lp_mathlib_Finset_prodPiCons___redArg(v_inst_75_, v_a_77_, v_x_79_, v_i_80_);
return v___x_82_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Finset_prodPiCons___boxed(lean_object* v_00_u03b1_83_, lean_object* v_s_84_, lean_object* v_inst_85_, lean_object* v_f_86_, lean_object* v_a_87_, lean_object* v_has_88_, lean_object* v_x_89_, lean_object* v_i_90_, lean_object* v_hi_91_){
_start:
{
lean_object* v_res_92_; 
v_res_92_ = lp_mathlib_Finset_prodPiCons(v_00_u03b1_83_, v_s_84_, v_inst_85_, v_f_86_, v_a_87_, v_has_88_, v_x_89_, v_i_90_, v_hi_91_);
lean_dec(v_s_84_);
return v_res_92_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Finset_consPiProdEquiv___redArg(lean_object* v_inst_93_, lean_object* v_s_94_, lean_object* v_a_95_){
_start:
{
lean_object* v___x_96_; lean_object* v___x_97_; lean_object* v___x_98_; 
lean_inc(v_a_95_);
lean_inc(v_s_94_);
v___x_96_ = lean_alloc_closure((void*)(lp_mathlib_Finset_consPiProd___boxed), 6, 5);
lean_closure_set(v___x_96_, 0, lean_box(0));
lean_closure_set(v___x_96_, 1, v_s_94_);
lean_closure_set(v___x_96_, 2, v_a_95_);
lean_closure_set(v___x_96_, 3, lean_box(0));
lean_closure_set(v___x_96_, 4, lean_box(0));
v___x_97_ = lean_alloc_closure((void*)(lp_mathlib_Finset_prodPiCons___boxed), 9, 6);
lean_closure_set(v___x_97_, 0, lean_box(0));
lean_closure_set(v___x_97_, 1, v_s_94_);
lean_closure_set(v___x_97_, 2, v_inst_93_);
lean_closure_set(v___x_97_, 3, lean_box(0));
lean_closure_set(v___x_97_, 4, v_a_95_);
lean_closure_set(v___x_97_, 5, lean_box(0));
v___x_98_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_98_, 0, v___x_96_);
lean_ctor_set(v___x_98_, 1, v___x_97_);
return v___x_98_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Finset_consPiProdEquiv(lean_object* v_00_u03b1_99_, lean_object* v_inst_100_, lean_object* v_s_101_, lean_object* v_f_102_, lean_object* v_a_103_, lean_object* v_has_104_){
_start:
{
lean_object* v___x_105_; 
v___x_105_ = lp_mathlib_Finset_consPiProdEquiv___redArg(v_inst_100_, v_s_101_, v_a_103_);
return v___x_105_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Finset_instInsert___redArg___lam__0(lean_object* v_inst_106_, lean_object* v_a_107_, lean_object* v_s_108_){
_start:
{
lean_object* v___x_109_; 
v___x_109_ = lp_mathlib_Multiset_ndinsert___redArg(v_inst_106_, v_a_107_, v_s_108_);
return v___x_109_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Finset_instInsert___redArg(lean_object* v_inst_110_){
_start:
{
lean_object* v___f_111_; 
v___f_111_ = lean_alloc_closure((void*)(lp_mathlib_Finset_instInsert___redArg___lam__0), 3, 1);
lean_closure_set(v___f_111_, 0, v_inst_110_);
return v___f_111_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Finset_instInsert(lean_object* v_00_u03b1_112_, lean_object* v_inst_113_){
_start:
{
lean_object* v___f_114_; 
v___f_114_ = lean_alloc_closure((void*)(lp_mathlib_Finset_instInsert___redArg___lam__0), 3, 1);
lean_closure_set(v___f_114_, 0, v_inst_113_);
return v___f_114_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Finset_subtypeInsertEquivOption___redArg___lam__0(lean_object* v_inst_115_, lean_object* v_x_116_, lean_object* v_y_117_){
_start:
{
lean_object* v___x_118_; uint8_t v___x_119_; 
lean_inc(v_y_117_);
v___x_118_ = lean_apply_2(v_inst_115_, v_y_117_, v_x_116_);
v___x_119_ = lean_unbox(v___x_118_);
if (v___x_119_ == 0)
{
lean_object* v___x_120_; 
v___x_120_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_120_, 0, v_y_117_);
return v___x_120_;
}
else
{
lean_object* v___x_121_; 
lean_dec(v_y_117_);
v___x_121_ = lean_box(0);
return v___x_121_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Finset_subtypeInsertEquivOption___redArg___lam__1(lean_object* v_x_122_, lean_object* v_y_123_){
_start:
{
if (lean_obj_tag(v_y_123_) == 0)
{
lean_inc(v_x_122_);
return v_x_122_;
}
else
{
lean_object* v_val_124_; 
v_val_124_ = lean_ctor_get(v_y_123_, 0);
lean_inc(v_val_124_);
return v_val_124_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Finset_subtypeInsertEquivOption___redArg___lam__1___boxed(lean_object* v_x_125_, lean_object* v_y_126_){
_start:
{
lean_object* v_res_127_; 
v_res_127_ = lp_mathlib_Finset_subtypeInsertEquivOption___redArg___lam__1(v_x_125_, v_y_126_);
lean_dec(v_y_126_);
lean_dec(v_x_125_);
return v_res_127_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Finset_subtypeInsertEquivOption___redArg(lean_object* v_inst_128_, lean_object* v_x_129_){
_start:
{
lean_object* v___f_130_; lean_object* v___f_131_; lean_object* v___x_132_; 
lean_inc(v_x_129_);
v___f_130_ = lean_alloc_closure((void*)(lp_mathlib_Finset_subtypeInsertEquivOption___redArg___lam__0), 3, 2);
lean_closure_set(v___f_130_, 0, v_inst_128_);
lean_closure_set(v___f_130_, 1, v_x_129_);
v___f_131_ = lean_alloc_closure((void*)(lp_mathlib_Finset_subtypeInsertEquivOption___redArg___lam__1___boxed), 2, 1);
lean_closure_set(v___f_131_, 0, v_x_129_);
v___x_132_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_132_, 0, v___f_130_);
lean_ctor_set(v___x_132_, 1, v___f_131_);
return v___x_132_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Finset_subtypeInsertEquivOption(lean_object* v_00_u03b1_133_, lean_object* v_inst_134_, lean_object* v_t_135_, lean_object* v_x_136_, lean_object* v_h_137_){
_start:
{
lean_object* v___x_138_; 
v___x_138_ = lp_mathlib_Finset_subtypeInsertEquivOption___redArg(v_inst_134_, v_x_136_);
return v___x_138_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Finset_subtypeInsertEquivOption___boxed(lean_object* v_00_u03b1_139_, lean_object* v_inst_140_, lean_object* v_t_141_, lean_object* v_x_142_, lean_object* v_h_143_){
_start:
{
lean_object* v_res_144_; 
v_res_144_ = lp_mathlib_Finset_subtypeInsertEquivOption(v_00_u03b1_139_, v_inst_140_, v_t_141_, v_x_142_, v_h_143_);
lean_dec(v_t_141_);
return v_res_144_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Finset_insertPiProd___redArg(lean_object* v_a_145_, lean_object* v_x_146_){
_start:
{
lean_object* v___f_147_; lean_object* v___x_148_; lean_object* v___x_149_; 
lean_inc(v_x_146_);
v___f_147_ = lean_alloc_closure((void*)(lp_mathlib_Finset_consPiProd___redArg___lam__0), 3, 1);
lean_closure_set(v___f_147_, 0, v_x_146_);
v___x_148_ = lean_apply_2(v_x_146_, v_a_145_, lean_box(0));
v___x_149_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_149_, 0, v___x_148_);
lean_ctor_set(v___x_149_, 1, v___f_147_);
return v___x_149_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Finset_insertPiProd(lean_object* v_00_u03b1_150_, lean_object* v_inst_151_, lean_object* v_s_152_, lean_object* v_a_153_, lean_object* v_f_154_, lean_object* v_x_155_){
_start:
{
lean_object* v___x_156_; 
v___x_156_ = lp_mathlib_Finset_insertPiProd___redArg(v_a_153_, v_x_155_);
return v___x_156_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Finset_insertPiProd___boxed(lean_object* v_00_u03b1_157_, lean_object* v_inst_158_, lean_object* v_s_159_, lean_object* v_a_160_, lean_object* v_f_161_, lean_object* v_x_162_){
_start:
{
lean_object* v_res_163_; 
v_res_163_ = lp_mathlib_Finset_insertPiProd(v_00_u03b1_157_, v_inst_158_, v_s_159_, v_a_160_, v_f_161_, v_x_162_);
lean_dec(v_s_159_);
lean_dec_ref(v_inst_158_);
return v_res_163_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Finset_prodPiInsert___redArg(lean_object* v_inst_164_, lean_object* v_a_165_, lean_object* v_x_166_, lean_object* v_i_167_){
_start:
{
lean_object* v___x_168_; uint8_t v___x_169_; 
lean_inc(v_i_167_);
v___x_168_ = lean_apply_2(v_inst_164_, v_i_167_, v_a_165_);
v___x_169_ = lean_unbox(v___x_168_);
if (v___x_169_ == 0)
{
lean_object* v_snd_170_; lean_object* v___x_171_; 
v_snd_170_ = lean_ctor_get(v_x_166_, 1);
lean_inc(v_snd_170_);
lean_dec_ref(v_x_166_);
v___x_171_ = lean_apply_2(v_snd_170_, v_i_167_, lean_box(0));
return v___x_171_;
}
else
{
lean_object* v_fst_172_; 
lean_dec(v_i_167_);
v_fst_172_ = lean_ctor_get(v_x_166_, 0);
lean_inc(v_fst_172_);
lean_dec_ref(v_x_166_);
return v_fst_172_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Finset_prodPiInsert(lean_object* v_00_u03b1_173_, lean_object* v_inst_174_, lean_object* v_s_175_, lean_object* v_f_176_, lean_object* v_a_177_, lean_object* v_x_178_, lean_object* v_i_179_, lean_object* v_hi_180_){
_start:
{
lean_object* v___x_181_; 
v___x_181_ = lp_mathlib_Finset_prodPiInsert___redArg(v_inst_174_, v_a_177_, v_x_178_, v_i_179_);
return v___x_181_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Finset_prodPiInsert___boxed(lean_object* v_00_u03b1_182_, lean_object* v_inst_183_, lean_object* v_s_184_, lean_object* v_f_185_, lean_object* v_a_186_, lean_object* v_x_187_, lean_object* v_i_188_, lean_object* v_hi_189_){
_start:
{
lean_object* v_res_190_; 
v_res_190_ = lp_mathlib_Finset_prodPiInsert(v_00_u03b1_182_, v_inst_183_, v_s_184_, v_f_185_, v_a_186_, v_x_187_, v_i_188_, v_hi_189_);
lean_dec(v_s_184_);
return v_res_190_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Finset_insertPiProdEquiv___redArg(lean_object* v_inst_191_, lean_object* v_s_192_, lean_object* v_a_193_){
_start:
{
lean_object* v___x_194_; lean_object* v___x_195_; lean_object* v___x_196_; 
lean_inc(v_a_193_);
lean_inc(v_s_192_);
lean_inc_ref(v_inst_191_);
v___x_194_ = lean_alloc_closure((void*)(lp_mathlib_Finset_insertPiProd___boxed), 6, 5);
lean_closure_set(v___x_194_, 0, lean_box(0));
lean_closure_set(v___x_194_, 1, v_inst_191_);
lean_closure_set(v___x_194_, 2, v_s_192_);
lean_closure_set(v___x_194_, 3, v_a_193_);
lean_closure_set(v___x_194_, 4, lean_box(0));
v___x_195_ = lean_alloc_closure((void*)(lp_mathlib_Finset_prodPiInsert___boxed), 8, 5);
lean_closure_set(v___x_195_, 0, lean_box(0));
lean_closure_set(v___x_195_, 1, v_inst_191_);
lean_closure_set(v___x_195_, 2, v_s_192_);
lean_closure_set(v___x_195_, 3, lean_box(0));
lean_closure_set(v___x_195_, 4, v_a_193_);
v___x_196_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_196_, 0, v___x_194_);
lean_ctor_set(v___x_196_, 1, v___x_195_);
return v___x_196_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Finset_insertPiProdEquiv(lean_object* v_00_u03b1_197_, lean_object* v_inst_198_, lean_object* v_s_199_, lean_object* v_f_200_, lean_object* v_a_201_, lean_object* v_has_202_){
_start:
{
lean_object* v___x_203_; 
v___x_203_ = lp_mathlib_Finset_insertPiProdEquiv___redArg(v_inst_198_, v_s_199_, v_a_201_);
return v___x_203_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Data_Finset_Attr(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Data_Finset_Dedup(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Data_Finset_Empty(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Data_Multiset_FinsetOps(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Util_Delaborators(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_Data_Finset_Insert(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Data_Finset_Attr(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Data_Finset_Dedup(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Data_Finset_Empty(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Data_Multiset_FinsetOps(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Util_Delaborators(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_Data_Finset_Insert(uint8_t builtin) {
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
lean_object* initialize_mathlib_Mathlib_Data_Finset_Attr(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Data_Finset_Dedup(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Data_Finset_Empty(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Data_Multiset_FinsetOps(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Util_Delaborators(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Data_Finset_Insert(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_Finset_Attr(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_Finset_Dedup(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_Finset_Empty(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_Multiset_FinsetOps(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Util_Delaborators(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Data_Finset_Insert(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_Data_Finset_Insert(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_Data_Finset_Insert(builtin);
}
#ifdef __cplusplus
}
#endif
