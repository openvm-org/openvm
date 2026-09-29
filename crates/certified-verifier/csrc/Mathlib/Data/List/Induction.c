// Lean compiler output
// Module: Mathlib.Data.List.Induction
// Imports: public import Init public meta import Init public import Mathlib.Init
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
lean_object* lean_array_mk(lean_object*);
lean_object* lean_array_pop(lean_object*);
lean_object* lean_array_to_list(lean_object*);
lean_object* l_List_getLast___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_reverseRec___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_reverseRec___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_reverseRec(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_reverseRec___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Data_List_Induction_0__List_reverseRec_match__1_splitter___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Data_List_Induction_0__List_reverseRec_match__1_splitter(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_reverseRecOn___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_reverseRecOn___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_reverseRecOn(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_reverseRecOn___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_bidirectionalRec___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_bidirectionalRec___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_bidirectionalRec(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_bidirectionalRec___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Data_List_Induction_0__List_bidirectionalRec_match__1_splitter___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Data_List_Induction_0__List_bidirectionalRec_match__1_splitter(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_bidirectionalRecOn___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_bidirectionalRecOn___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_bidirectionalRecOn(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_bidirectionalRecOn___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_recNeNil___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_recNeNil(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_recOnNeNil___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_recOnNeNil(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_twoStepInduction___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_twoStepInduction___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_twoStepInduction(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_reverseRec___redArg(lean_object* v_nil_1_, lean_object* v_append__singleton_2_, lean_object* v_x_3_){
_start:
{
if (lean_obj_tag(v_x_3_) == 0)
{
lean_dec(v_append__singleton_2_);
lean_inc(v_nil_1_);
return v_nil_1_;
}
else
{
lean_object* v___x_4_; lean_object* v___x_5_; lean_object* v___x_6_; lean_object* v___x_7_; lean_object* v___x_8_; lean_object* v___x_9_; 
lean_inc_ref(v_x_3_);
v___x_4_ = lean_array_mk(v_x_3_);
v___x_5_ = lean_array_pop(v___x_4_);
v___x_6_ = lean_array_to_list(v___x_5_);
v___x_7_ = l_List_getLast___redArg(v_x_3_);
lean_dec_ref_known(v_x_3_, 2);
lean_inc(v___x_6_);
lean_inc(v_append__singleton_2_);
v___x_8_ = lp_mathlib_List_reverseRec___redArg(v_nil_1_, v_append__singleton_2_, v___x_6_);
v___x_9_ = lean_apply_3(v_append__singleton_2_, v___x_6_, v___x_7_, v___x_8_);
return v___x_9_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_reverseRec___redArg___boxed(lean_object* v_nil_10_, lean_object* v_append__singleton_11_, lean_object* v_x_12_){
_start:
{
lean_object* v_res_13_; 
v_res_13_ = lp_mathlib_List_reverseRec___redArg(v_nil_10_, v_append__singleton_11_, v_x_12_);
lean_dec(v_nil_10_);
return v_res_13_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_reverseRec(lean_object* v_00_u03b1_14_, lean_object* v_motive_15_, lean_object* v_nil_16_, lean_object* v_append__singleton_17_, lean_object* v_x_18_){
_start:
{
lean_object* v___x_19_; 
v___x_19_ = lp_mathlib_List_reverseRec___redArg(v_nil_16_, v_append__singleton_17_, v_x_18_);
return v___x_19_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_reverseRec___boxed(lean_object* v_00_u03b1_20_, lean_object* v_motive_21_, lean_object* v_nil_22_, lean_object* v_append__singleton_23_, lean_object* v_x_24_){
_start:
{
lean_object* v_res_25_; 
v_res_25_ = lp_mathlib_List_reverseRec(v_00_u03b1_20_, v_motive_21_, v_nil_22_, v_append__singleton_23_, v_x_24_);
lean_dec(v_nil_22_);
return v_res_25_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Data_List_Induction_0__List_reverseRec_match__1_splitter___redArg(lean_object* v_x_26_, lean_object* v_h__1_27_, lean_object* v_h__2_28_){
_start:
{
if (lean_obj_tag(v_x_26_) == 0)
{
lean_object* v___x_29_; lean_object* v___x_30_; 
lean_dec(v_h__2_28_);
v___x_29_ = lean_box(0);
v___x_30_ = lean_apply_1(v_h__1_27_, v___x_29_);
return v___x_30_;
}
else
{
lean_object* v_head_31_; lean_object* v_tail_32_; lean_object* v___x_33_; 
lean_dec(v_h__1_27_);
v_head_31_ = lean_ctor_get(v_x_26_, 0);
lean_inc(v_head_31_);
v_tail_32_ = lean_ctor_get(v_x_26_, 1);
lean_inc(v_tail_32_);
lean_dec_ref_known(v_x_26_, 2);
v___x_33_ = lean_apply_2(v_h__2_28_, v_head_31_, v_tail_32_);
return v___x_33_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Data_List_Induction_0__List_reverseRec_match__1_splitter(lean_object* v_00_u03b1_34_, lean_object* v_motive_35_, lean_object* v_x_36_, lean_object* v_h__1_37_, lean_object* v_h__2_38_){
_start:
{
if (lean_obj_tag(v_x_36_) == 0)
{
lean_object* v___x_39_; lean_object* v___x_40_; 
lean_dec(v_h__2_38_);
v___x_39_ = lean_box(0);
v___x_40_ = lean_apply_1(v_h__1_37_, v___x_39_);
return v___x_40_;
}
else
{
lean_object* v_head_41_; lean_object* v_tail_42_; lean_object* v___x_43_; 
lean_dec(v_h__1_37_);
v_head_41_ = lean_ctor_get(v_x_36_, 0);
lean_inc(v_head_41_);
v_tail_42_ = lean_ctor_get(v_x_36_, 1);
lean_inc(v_tail_42_);
lean_dec_ref_known(v_x_36_, 2);
v___x_43_ = lean_apply_2(v_h__2_38_, v_head_41_, v_tail_42_);
return v___x_43_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_reverseRecOn___redArg(lean_object* v_l_44_, lean_object* v_nil_45_, lean_object* v_append__singleton_46_){
_start:
{
lean_object* v___x_47_; 
v___x_47_ = lp_mathlib_List_reverseRec___redArg(v_nil_45_, v_append__singleton_46_, v_l_44_);
return v___x_47_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_reverseRecOn___redArg___boxed(lean_object* v_l_48_, lean_object* v_nil_49_, lean_object* v_append__singleton_50_){
_start:
{
lean_object* v_res_51_; 
v_res_51_ = lp_mathlib_List_reverseRecOn___redArg(v_l_48_, v_nil_49_, v_append__singleton_50_);
lean_dec(v_nil_49_);
return v_res_51_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_reverseRecOn(lean_object* v_00_u03b1_52_, lean_object* v_motive_53_, lean_object* v_l_54_, lean_object* v_nil_55_, lean_object* v_append__singleton_56_){
_start:
{
lean_object* v___x_57_; 
v___x_57_ = lp_mathlib_List_reverseRec___redArg(v_nil_55_, v_append__singleton_56_, v_l_54_);
return v___x_57_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_reverseRecOn___boxed(lean_object* v_00_u03b1_58_, lean_object* v_motive_59_, lean_object* v_l_60_, lean_object* v_nil_61_, lean_object* v_append__singleton_62_){
_start:
{
lean_object* v_res_63_; 
v_res_63_ = lp_mathlib_List_reverseRecOn(v_00_u03b1_58_, v_motive_59_, v_l_60_, v_nil_61_, v_append__singleton_62_);
lean_dec(v_nil_61_);
return v_res_63_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_bidirectionalRec___redArg(lean_object* v_nil_64_, lean_object* v_singleton_65_, lean_object* v_cons__append_66_, lean_object* v_x_67_){
_start:
{
if (lean_obj_tag(v_x_67_) == 0)
{
lean_dec(v_cons__append_66_);
lean_dec(v_singleton_65_);
lean_inc(v_nil_64_);
return v_nil_64_;
}
else
{
lean_object* v_tail_68_; 
v_tail_68_ = lean_ctor_get(v_x_67_, 1);
if (lean_obj_tag(v_tail_68_) == 0)
{
lean_object* v_head_69_; lean_object* v___x_70_; 
lean_dec(v_cons__append_66_);
v_head_69_ = lean_ctor_get(v_x_67_, 0);
lean_inc(v_head_69_);
lean_dec_ref_known(v_x_67_, 2);
v___x_70_ = lean_apply_1(v_singleton_65_, v_head_69_);
return v___x_70_;
}
else
{
lean_object* v_head_71_; lean_object* v___x_72_; lean_object* v___x_73_; lean_object* v___x_74_; lean_object* v___x_75_; lean_object* v___x_76_; lean_object* v___x_77_; 
lean_inc_ref_n(v_tail_68_, 2);
v_head_71_ = lean_ctor_get(v_x_67_, 0);
lean_inc(v_head_71_);
lean_dec_ref_known(v_x_67_, 2);
v___x_72_ = lean_array_mk(v_tail_68_);
v___x_73_ = lean_array_pop(v___x_72_);
v___x_74_ = lean_array_to_list(v___x_73_);
v___x_75_ = l_List_getLast___redArg(v_tail_68_);
lean_dec_ref_known(v_tail_68_, 2);
lean_inc(v___x_74_);
lean_inc(v_cons__append_66_);
v___x_76_ = lp_mathlib_List_bidirectionalRec___redArg(v_nil_64_, v_singleton_65_, v_cons__append_66_, v___x_74_);
v___x_77_ = lean_apply_4(v_cons__append_66_, v_head_71_, v___x_74_, v___x_75_, v___x_76_);
return v___x_77_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_bidirectionalRec___redArg___boxed(lean_object* v_nil_78_, lean_object* v_singleton_79_, lean_object* v_cons__append_80_, lean_object* v_x_81_){
_start:
{
lean_object* v_res_82_; 
v_res_82_ = lp_mathlib_List_bidirectionalRec___redArg(v_nil_78_, v_singleton_79_, v_cons__append_80_, v_x_81_);
lean_dec(v_nil_78_);
return v_res_82_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_bidirectionalRec(lean_object* v_00_u03b1_83_, lean_object* v_motive_84_, lean_object* v_nil_85_, lean_object* v_singleton_86_, lean_object* v_cons__append_87_, lean_object* v_x_88_){
_start:
{
lean_object* v___x_89_; 
v___x_89_ = lp_mathlib_List_bidirectionalRec___redArg(v_nil_85_, v_singleton_86_, v_cons__append_87_, v_x_88_);
return v___x_89_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_bidirectionalRec___boxed(lean_object* v_00_u03b1_90_, lean_object* v_motive_91_, lean_object* v_nil_92_, lean_object* v_singleton_93_, lean_object* v_cons__append_94_, lean_object* v_x_95_){
_start:
{
lean_object* v_res_96_; 
v_res_96_ = lp_mathlib_List_bidirectionalRec(v_00_u03b1_90_, v_motive_91_, v_nil_92_, v_singleton_93_, v_cons__append_94_, v_x_95_);
lean_dec(v_nil_92_);
return v_res_96_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Data_List_Induction_0__List_bidirectionalRec_match__1_splitter___redArg(lean_object* v_x_97_, lean_object* v_h__1_98_, lean_object* v_h__2_99_, lean_object* v_h__3_100_){
_start:
{
if (lean_obj_tag(v_x_97_) == 0)
{
lean_object* v___x_101_; lean_object* v___x_102_; 
lean_dec(v_h__3_100_);
lean_dec(v_h__2_99_);
v___x_101_ = lean_box(0);
v___x_102_ = lean_apply_1(v_h__1_98_, v___x_101_);
return v___x_102_;
}
else
{
lean_object* v_tail_103_; 
lean_dec(v_h__1_98_);
v_tail_103_ = lean_ctor_get(v_x_97_, 1);
if (lean_obj_tag(v_tail_103_) == 0)
{
lean_object* v_head_104_; lean_object* v___x_105_; 
lean_dec(v_h__3_100_);
v_head_104_ = lean_ctor_get(v_x_97_, 0);
lean_inc(v_head_104_);
lean_dec_ref_known(v_x_97_, 2);
v___x_105_ = lean_apply_1(v_h__2_99_, v_head_104_);
return v___x_105_;
}
else
{
lean_object* v_head_106_; lean_object* v_head_107_; lean_object* v_tail_108_; lean_object* v___x_109_; 
lean_inc_ref(v_tail_103_);
lean_dec(v_h__2_99_);
v_head_106_ = lean_ctor_get(v_x_97_, 0);
lean_inc(v_head_106_);
lean_dec_ref_known(v_x_97_, 2);
v_head_107_ = lean_ctor_get(v_tail_103_, 0);
lean_inc(v_head_107_);
v_tail_108_ = lean_ctor_get(v_tail_103_, 1);
lean_inc(v_tail_108_);
lean_dec_ref_known(v_tail_103_, 2);
v___x_109_ = lean_apply_3(v_h__3_100_, v_head_106_, v_head_107_, v_tail_108_);
return v___x_109_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Data_List_Induction_0__List_bidirectionalRec_match__1_splitter(lean_object* v_00_u03b1_110_, lean_object* v_motive_111_, lean_object* v_x_112_, lean_object* v_h__1_113_, lean_object* v_h__2_114_, lean_object* v_h__3_115_){
_start:
{
if (lean_obj_tag(v_x_112_) == 0)
{
lean_object* v___x_116_; lean_object* v___x_117_; 
lean_dec(v_h__3_115_);
lean_dec(v_h__2_114_);
v___x_116_ = lean_box(0);
v___x_117_ = lean_apply_1(v_h__1_113_, v___x_116_);
return v___x_117_;
}
else
{
lean_object* v_tail_118_; 
lean_dec(v_h__1_113_);
v_tail_118_ = lean_ctor_get(v_x_112_, 1);
if (lean_obj_tag(v_tail_118_) == 0)
{
lean_object* v_head_119_; lean_object* v___x_120_; 
lean_dec(v_h__3_115_);
v_head_119_ = lean_ctor_get(v_x_112_, 0);
lean_inc(v_head_119_);
lean_dec_ref_known(v_x_112_, 2);
v___x_120_ = lean_apply_1(v_h__2_114_, v_head_119_);
return v___x_120_;
}
else
{
lean_object* v_head_121_; lean_object* v_head_122_; lean_object* v_tail_123_; lean_object* v___x_124_; 
lean_inc_ref(v_tail_118_);
lean_dec(v_h__2_114_);
v_head_121_ = lean_ctor_get(v_x_112_, 0);
lean_inc(v_head_121_);
lean_dec_ref_known(v_x_112_, 2);
v_head_122_ = lean_ctor_get(v_tail_118_, 0);
lean_inc(v_head_122_);
v_tail_123_ = lean_ctor_get(v_tail_118_, 1);
lean_inc(v_tail_123_);
lean_dec_ref_known(v_tail_118_, 2);
v___x_124_ = lean_apply_3(v_h__3_115_, v_head_121_, v_head_122_, v_tail_123_);
return v___x_124_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_bidirectionalRecOn___redArg(lean_object* v_l_125_, lean_object* v_H0_126_, lean_object* v_H1_127_, lean_object* v_Hn_128_){
_start:
{
lean_object* v___x_129_; 
v___x_129_ = lp_mathlib_List_bidirectionalRec___redArg(v_H0_126_, v_H1_127_, v_Hn_128_, v_l_125_);
return v___x_129_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_bidirectionalRecOn___redArg___boxed(lean_object* v_l_130_, lean_object* v_H0_131_, lean_object* v_H1_132_, lean_object* v_Hn_133_){
_start:
{
lean_object* v_res_134_; 
v_res_134_ = lp_mathlib_List_bidirectionalRecOn___redArg(v_l_130_, v_H0_131_, v_H1_132_, v_Hn_133_);
lean_dec(v_H0_131_);
return v_res_134_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_bidirectionalRecOn(lean_object* v_00_u03b1_135_, lean_object* v_C_136_, lean_object* v_l_137_, lean_object* v_H0_138_, lean_object* v_H1_139_, lean_object* v_Hn_140_){
_start:
{
lean_object* v___x_141_; 
v___x_141_ = lp_mathlib_List_bidirectionalRec___redArg(v_H0_138_, v_H1_139_, v_Hn_140_, v_l_137_);
return v___x_141_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_bidirectionalRecOn___boxed(lean_object* v_00_u03b1_142_, lean_object* v_C_143_, lean_object* v_l_144_, lean_object* v_H0_145_, lean_object* v_H1_146_, lean_object* v_Hn_147_){
_start:
{
lean_object* v_res_148_; 
v_res_148_ = lp_mathlib_List_bidirectionalRecOn(v_00_u03b1_142_, v_C_143_, v_l_144_, v_H0_145_, v_H1_146_, v_Hn_147_);
lean_dec(v_H0_145_);
return v_res_148_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_recNeNil___redArg(lean_object* v_singleton_149_, lean_object* v_cons_150_, lean_object* v_l_151_){
_start:
{
lean_object* v_tail_152_; 
v_tail_152_ = lean_ctor_get(v_l_151_, 1);
if (lean_obj_tag(v_tail_152_) == 0)
{
lean_object* v_head_153_; lean_object* v___x_154_; 
lean_dec(v_cons_150_);
v_head_153_ = lean_ctor_get(v_l_151_, 0);
lean_inc(v_head_153_);
lean_dec(v_l_151_);
v___x_154_ = lean_apply_1(v_singleton_149_, v_head_153_);
return v___x_154_;
}
else
{
lean_object* v_head_155_; lean_object* v___x_156_; lean_object* v___x_157_; 
lean_inc_ref_n(v_tail_152_, 2);
v_head_155_ = lean_ctor_get(v_l_151_, 0);
lean_inc(v_head_155_);
lean_dec(v_l_151_);
lean_inc(v_cons_150_);
v___x_156_ = lp_mathlib_List_recNeNil___redArg(v_singleton_149_, v_cons_150_, v_tail_152_);
v___x_157_ = lean_apply_4(v_cons_150_, v_head_155_, v_tail_152_, lean_box(0), v___x_156_);
return v___x_157_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_recNeNil(lean_object* v_00_u03b1_158_, lean_object* v_motive_159_, lean_object* v_singleton_160_, lean_object* v_cons_161_, lean_object* v_l_162_, lean_object* v_h_163_){
_start:
{
lean_object* v___x_164_; 
v___x_164_ = lp_mathlib_List_recNeNil___redArg(v_singleton_160_, v_cons_161_, v_l_162_);
return v___x_164_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_recOnNeNil___redArg(lean_object* v_l_165_, lean_object* v_singleton_166_, lean_object* v_cons_167_){
_start:
{
lean_object* v___x_168_; 
v___x_168_ = lp_mathlib_List_recNeNil___redArg(v_singleton_166_, v_cons_167_, v_l_165_);
return v___x_168_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_recOnNeNil(lean_object* v_00_u03b1_169_, lean_object* v_motive_170_, lean_object* v_l_171_, lean_object* v_h_172_, lean_object* v_singleton_173_, lean_object* v_cons_174_){
_start:
{
lean_object* v___x_175_; 
v___x_175_ = lp_mathlib_List_recNeNil___redArg(v_singleton_173_, v_cons_174_, v_l_171_);
return v___x_175_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_twoStepInduction___redArg(lean_object* v_nil_176_, lean_object* v_singleton_177_, lean_object* v_cons__cons_178_, lean_object* v_l_179_){
_start:
{
if (lean_obj_tag(v_l_179_) == 0)
{
lean_dec(v_cons__cons_178_);
lean_dec(v_singleton_177_);
return v_nil_176_;
}
else
{
lean_object* v_tail_180_; 
v_tail_180_ = lean_ctor_get(v_l_179_, 1);
if (lean_obj_tag(v_tail_180_) == 0)
{
lean_object* v_head_181_; lean_object* v___x_182_; 
lean_dec(v_cons__cons_178_);
lean_dec(v_nil_176_);
v_head_181_ = lean_ctor_get(v_l_179_, 0);
lean_inc(v_head_181_);
lean_dec_ref_known(v_l_179_, 2);
v___x_182_ = lean_apply_1(v_singleton_177_, v_head_181_);
return v___x_182_;
}
else
{
lean_object* v_head_183_; lean_object* v_head_184_; lean_object* v_tail_185_; lean_object* v___f_186_; lean_object* v___x_187_; lean_object* v___x_188_; 
lean_inc_ref(v_tail_180_);
v_head_183_ = lean_ctor_get(v_l_179_, 0);
lean_inc(v_head_183_);
lean_dec_ref_known(v_l_179_, 2);
v_head_184_ = lean_ctor_get(v_tail_180_, 0);
lean_inc(v_head_184_);
v_tail_185_ = lean_ctor_get(v_tail_180_, 1);
lean_inc_n(v_tail_185_, 3);
lean_dec_ref_known(v_tail_180_, 2);
lean_inc_n(v_cons__cons_178_, 2);
lean_inc(v_singleton_177_);
lean_inc(v_nil_176_);
v___f_186_ = lean_alloc_closure((void*)(lp_mathlib_List_twoStepInduction___redArg___lam__0), 5, 4);
lean_closure_set(v___f_186_, 0, v_tail_185_);
lean_closure_set(v___f_186_, 1, v_nil_176_);
lean_closure_set(v___f_186_, 2, v_singleton_177_);
lean_closure_set(v___f_186_, 3, v_cons__cons_178_);
v___x_187_ = lp_mathlib_List_twoStepInduction___redArg(v_nil_176_, v_singleton_177_, v_cons__cons_178_, v_tail_185_);
v___x_188_ = lean_apply_5(v_cons__cons_178_, v_head_183_, v_head_184_, v_tail_185_, v___x_187_, v___f_186_);
return v___x_188_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_twoStepInduction___redArg___lam__0(lean_object* v_tail_189_, lean_object* v_nil_190_, lean_object* v_singleton_191_, lean_object* v_cons__cons_192_, lean_object* v_y_193_){
_start:
{
lean_object* v___x_194_; lean_object* v___x_195_; 
v___x_194_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_194_, 0, v_y_193_);
lean_ctor_set(v___x_194_, 1, v_tail_189_);
v___x_195_ = lp_mathlib_List_twoStepInduction___redArg(v_nil_190_, v_singleton_191_, v_cons__cons_192_, v___x_194_);
return v___x_195_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_twoStepInduction(lean_object* v_00_u03b1_196_, lean_object* v_motive_197_, lean_object* v_nil_198_, lean_object* v_singleton_199_, lean_object* v_cons__cons_200_, lean_object* v_l_201_){
_start:
{
lean_object* v___x_202_; 
v___x_202_ = lp_mathlib_List_twoStepInduction___redArg(v_nil_198_, v_singleton_199_, v_cons__cons_200_, v_l_201_);
return v___x_202_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Init(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_Data_List_Induction(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_Data_List_Induction(uint8_t builtin) {
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
lean_object* initialize_mathlib_Mathlib_Init(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Data_List_Induction(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Data_List_Induction(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_Data_List_Induction(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_Data_List_Induction(builtin);
}
#ifdef __cplusplus
}
#endif
