// Lean compiler output
// Module: Mathlib.Data.List.Sublists
// Imports: public import Init public meta import Init public import Mathlib.Data.Nat.Choose.Basic public import Mathlib.Data.List.Perm.Basic public import Mathlib.Data.List.Lex public import Mathlib.Data.List.Induction public import Mathlib.Data.List.Nodup public import Mathlib.Data.Prod.Basic public import Mathlib.Tactic.Finiteness.Attr
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
lean_object* l_List_appendTR___redArg(lean_object*, lean_object*);
uint8_t lean_nat_dec_eq(lean_object*, lean_object*);
lean_object* lean_nat_sub(lean_object*, lean_object*);
lean_object* lean_nat_add(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_foldl___at___00List_sublists_x27Aux_spec__0___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_sublists_x27Aux___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_sublists_x27Aux(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_foldl___at___00List_sublists_x27Aux_spec__0(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_foldl___at___00List_sublistsAux_spec__0___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_sublistsAux___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_sublistsAux(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_foldl___at___00List_sublistsAux_spec__0(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_sublistsLenAux___redArg___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_sublistsLenAux___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_sublistsLenAux(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_sublistsLen___redArg___lam__0(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_sublistsLen___redArg___lam__0___boxed(lean_object*);
static const lean_closure_object lp_mathlib_List_sublistsLen___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_List_sublistsLen___redArg___lam__0___boxed, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_List_sublistsLen___redArg___closed__0 = (const lean_object*)&lp_mathlib_List_sublistsLen___redArg___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_List_sublistsLen___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_sublistsLen(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Data_List_Sublists_0__List_sublistsLenAux_match__1_splitter___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Data_List_Sublists_0__List_sublistsLenAux_match__1_splitter___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Data_List_Sublists_0__List_sublistsLenAux_match__1_splitter(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Data_List_Sublists_0__List_sublistsLenAux_match__1_splitter___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_foldl___at___00List_sublists_x27Aux_spec__0___redArg(lean_object* v_a_1_, lean_object* v_x_2_, lean_object* v_x_3_){
_start:
{
if (lean_obj_tag(v_x_3_) == 0)
{
lean_dec(v_a_1_);
return v_x_2_;
}
else
{
lean_object* v_head_4_; lean_object* v_tail_5_; lean_object* v___x_7_; uint8_t v_isShared_8_; uint8_t v_isSharedCheck_16_; 
v_head_4_ = lean_ctor_get(v_x_3_, 0);
v_tail_5_ = lean_ctor_get(v_x_3_, 1);
v_isSharedCheck_16_ = !lean_is_exclusive(v_x_3_);
if (v_isSharedCheck_16_ == 0)
{
v___x_7_ = v_x_3_;
v_isShared_8_ = v_isSharedCheck_16_;
goto v_resetjp_6_;
}
else
{
lean_inc(v_tail_5_);
lean_inc(v_head_4_);
lean_dec(v_x_3_);
v___x_7_ = lean_box(0);
v_isShared_8_ = v_isSharedCheck_16_;
goto v_resetjp_6_;
}
v_resetjp_6_:
{
lean_object* v___x_10_; 
lean_inc(v_a_1_);
if (v_isShared_8_ == 0)
{
lean_ctor_set(v___x_7_, 1, v_head_4_);
lean_ctor_set(v___x_7_, 0, v_a_1_);
v___x_10_ = v___x_7_;
goto v_reusejp_9_;
}
else
{
lean_object* v_reuseFailAlloc_15_; 
v_reuseFailAlloc_15_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_15_, 0, v_a_1_);
lean_ctor_set(v_reuseFailAlloc_15_, 1, v_head_4_);
v___x_10_ = v_reuseFailAlloc_15_;
goto v_reusejp_9_;
}
v_reusejp_9_:
{
lean_object* v___x_11_; lean_object* v___x_12_; lean_object* v___x_13_; 
v___x_11_ = lean_box(0);
v___x_12_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_12_, 0, v___x_10_);
lean_ctor_set(v___x_12_, 1, v___x_11_);
v___x_13_ = l_List_appendTR___redArg(v_x_2_, v___x_12_);
v_x_2_ = v___x_13_;
v_x_3_ = v_tail_5_;
goto _start;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_sublists_x27Aux___redArg(lean_object* v_a_17_, lean_object* v_r_u2081_18_, lean_object* v_r_u2082_19_){
_start:
{
lean_object* v___x_20_; 
v___x_20_ = lp_mathlib_List_foldl___at___00List_sublists_x27Aux_spec__0___redArg(v_a_17_, v_r_u2082_19_, v_r_u2081_18_);
return v___x_20_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_sublists_x27Aux(lean_object* v_00_u03b1_21_, lean_object* v_a_22_, lean_object* v_r_u2081_23_, lean_object* v_r_u2082_24_){
_start:
{
lean_object* v___x_25_; 
v___x_25_ = lp_mathlib_List_foldl___at___00List_sublists_x27Aux_spec__0___redArg(v_a_22_, v_r_u2082_24_, v_r_u2081_23_);
return v___x_25_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_foldl___at___00List_sublists_x27Aux_spec__0(lean_object* v_00_u03b1_26_, lean_object* v_a_27_, lean_object* v_x_28_, lean_object* v_x_29_){
_start:
{
lean_object* v___x_30_; 
v___x_30_ = lp_mathlib_List_foldl___at___00List_sublists_x27Aux_spec__0___redArg(v_a_27_, v_x_28_, v_x_29_);
return v___x_30_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_foldl___at___00List_sublistsAux_spec__0___redArg(lean_object* v_a_31_, lean_object* v_x_32_, lean_object* v_x_33_){
_start:
{
if (lean_obj_tag(v_x_33_) == 0)
{
lean_dec(v_a_31_);
return v_x_32_;
}
else
{
lean_object* v_head_34_; lean_object* v_tail_35_; lean_object* v___x_37_; uint8_t v_isShared_38_; uint8_t v_isSharedCheck_47_; 
v_head_34_ = lean_ctor_get(v_x_33_, 0);
v_tail_35_ = lean_ctor_get(v_x_33_, 1);
v_isSharedCheck_47_ = !lean_is_exclusive(v_x_33_);
if (v_isSharedCheck_47_ == 0)
{
v___x_37_ = v_x_33_;
v_isShared_38_ = v_isSharedCheck_47_;
goto v_resetjp_36_;
}
else
{
lean_inc(v_tail_35_);
lean_inc(v_head_34_);
lean_dec(v_x_33_);
v___x_37_ = lean_box(0);
v_isShared_38_ = v_isSharedCheck_47_;
goto v_resetjp_36_;
}
v_resetjp_36_:
{
lean_object* v___x_40_; 
lean_inc(v_head_34_);
lean_inc(v_a_31_);
if (v_isShared_38_ == 0)
{
lean_ctor_set(v___x_37_, 1, v_head_34_);
lean_ctor_set(v___x_37_, 0, v_a_31_);
v___x_40_ = v___x_37_;
goto v_reusejp_39_;
}
else
{
lean_object* v_reuseFailAlloc_46_; 
v_reuseFailAlloc_46_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_46_, 0, v_a_31_);
lean_ctor_set(v_reuseFailAlloc_46_, 1, v_head_34_);
v___x_40_ = v_reuseFailAlloc_46_;
goto v_reusejp_39_;
}
v_reusejp_39_:
{
lean_object* v___x_41_; lean_object* v___x_42_; lean_object* v___x_43_; lean_object* v___x_44_; 
v___x_41_ = lean_box(0);
v___x_42_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_42_, 0, v___x_40_);
lean_ctor_set(v___x_42_, 1, v___x_41_);
v___x_43_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_43_, 0, v_head_34_);
lean_ctor_set(v___x_43_, 1, v___x_42_);
v___x_44_ = l_List_appendTR___redArg(v_x_32_, v___x_43_);
v_x_32_ = v___x_44_;
v_x_33_ = v_tail_35_;
goto _start;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_sublistsAux___redArg(lean_object* v_a_48_, lean_object* v_r_49_){
_start:
{
lean_object* v___x_50_; lean_object* v___x_51_; 
v___x_50_ = lean_box(0);
v___x_51_ = lp_mathlib_List_foldl___at___00List_sublistsAux_spec__0___redArg(v_a_48_, v___x_50_, v_r_49_);
return v___x_51_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_sublistsAux(lean_object* v_00_u03b1_52_, lean_object* v_a_53_, lean_object* v_r_54_){
_start:
{
lean_object* v___x_55_; 
v___x_55_ = lp_mathlib_List_sublistsAux___redArg(v_a_53_, v_r_54_);
return v___x_55_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_foldl___at___00List_sublistsAux_spec__0(lean_object* v_00_u03b1_56_, lean_object* v_a_57_, lean_object* v_x_58_, lean_object* v_x_59_){
_start:
{
lean_object* v___x_60_; 
v___x_60_ = lp_mathlib_List_foldl___at___00List_sublistsAux_spec__0___redArg(v_a_57_, v_x_58_, v_x_59_);
return v___x_60_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_sublistsLenAux___redArg___lam__0(lean_object* v_head_61_, lean_object* v_x_62_, lean_object* v___y_63_){
_start:
{
lean_object* v___x_64_; lean_object* v___x_65_; 
v___x_64_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_64_, 0, v_head_61_);
lean_ctor_set(v___x_64_, 1, v___y_63_);
v___x_65_ = lean_apply_1(v_x_62_, v___x_64_);
return v___x_65_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_sublistsLenAux___redArg(lean_object* v_x_66_, lean_object* v_x_67_, lean_object* v_x_68_, lean_object* v_x_69_){
_start:
{
lean_object* v_zero_70_; uint8_t v_isZero_71_; 
v_zero_70_ = lean_unsigned_to_nat(0u);
v_isZero_71_ = lean_nat_dec_eq(v_x_66_, v_zero_70_);
if (v_isZero_71_ == 1)
{
lean_object* v___x_72_; lean_object* v___x_73_; lean_object* v___x_74_; 
lean_dec(v_x_67_);
lean_dec(v_x_66_);
v___x_72_ = lean_box(0);
v___x_73_ = lean_apply_1(v_x_68_, v___x_72_);
v___x_74_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_74_, 0, v___x_73_);
lean_ctor_set(v___x_74_, 1, v_x_69_);
return v___x_74_;
}
else
{
if (lean_obj_tag(v_x_67_) == 0)
{
lean_dec(v_x_68_);
lean_dec(v_x_66_);
return v_x_69_;
}
else
{
lean_object* v_head_75_; lean_object* v_tail_76_; lean_object* v_one_77_; lean_object* v_n_78_; lean_object* v___f_79_; lean_object* v___x_80_; lean_object* v___x_81_; 
v_head_75_ = lean_ctor_get(v_x_67_, 0);
lean_inc(v_head_75_);
v_tail_76_ = lean_ctor_get(v_x_67_, 1);
lean_inc_n(v_tail_76_, 2);
lean_dec_ref_known(v_x_67_, 2);
v_one_77_ = lean_unsigned_to_nat(1u);
v_n_78_ = lean_nat_sub(v_x_66_, v_one_77_);
lean_dec(v_x_66_);
lean_inc(v_x_68_);
v___f_79_ = lean_alloc_closure((void*)(lp_mathlib_List_sublistsLenAux___redArg___lam__0), 3, 2);
lean_closure_set(v___f_79_, 0, v_head_75_);
lean_closure_set(v___f_79_, 1, v_x_68_);
v___x_80_ = lean_nat_add(v_n_78_, v_one_77_);
v___x_81_ = lp_mathlib_List_sublistsLenAux___redArg(v_n_78_, v_tail_76_, v___f_79_, v_x_69_);
v_x_66_ = v___x_80_;
v_x_67_ = v_tail_76_;
v_x_69_ = v___x_81_;
goto _start;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_sublistsLenAux(lean_object* v_00_u03b1_83_, lean_object* v_00_u03b2_84_, lean_object* v_x_85_, lean_object* v_x_86_, lean_object* v_x_87_, lean_object* v_x_88_){
_start:
{
lean_object* v___x_89_; 
v___x_89_ = lp_mathlib_List_sublistsLenAux___redArg(v_x_85_, v_x_86_, v_x_87_, v_x_88_);
return v___x_89_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_sublistsLen___redArg___lam__0(lean_object* v___y_90_){
_start:
{
lean_inc(v___y_90_);
return v___y_90_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_sublistsLen___redArg___lam__0___boxed(lean_object* v___y_91_){
_start:
{
lean_object* v_res_92_; 
v_res_92_ = lp_mathlib_List_sublistsLen___redArg___lam__0(v___y_91_);
lean_dec(v___y_91_);
return v_res_92_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_sublistsLen___redArg(lean_object* v_n_94_, lean_object* v_l_95_){
_start:
{
lean_object* v___f_96_; lean_object* v___x_97_; lean_object* v___x_98_; 
v___f_96_ = ((lean_object*)(lp_mathlib_List_sublistsLen___redArg___closed__0));
v___x_97_ = lean_box(0);
v___x_98_ = lp_mathlib_List_sublistsLenAux___redArg(v_n_94_, v_l_95_, v___f_96_, v___x_97_);
return v___x_98_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_sublistsLen(lean_object* v_00_u03b1_99_, lean_object* v_n_100_, lean_object* v_l_101_){
_start:
{
lean_object* v___x_102_; 
v___x_102_ = lp_mathlib_List_sublistsLen___redArg(v_n_100_, v_l_101_);
return v___x_102_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Data_List_Sublists_0__List_sublistsLenAux_match__1_splitter___redArg(lean_object* v_x_103_, lean_object* v_x_104_, lean_object* v_x_105_, lean_object* v_x_106_, lean_object* v_h__1_107_, lean_object* v_h__2_108_, lean_object* v_h__3_109_){
_start:
{
lean_object* v_zero_110_; uint8_t v_isZero_111_; 
v_zero_110_ = lean_unsigned_to_nat(0u);
v_isZero_111_ = lean_nat_dec_eq(v_x_103_, v_zero_110_);
if (v_isZero_111_ == 1)
{
lean_object* v___x_112_; 
lean_dec(v_h__3_109_);
lean_dec(v_h__2_108_);
v___x_112_ = lean_apply_3(v_h__1_107_, v_x_104_, v_x_105_, v_x_106_);
return v___x_112_;
}
else
{
lean_object* v_one_113_; lean_object* v_n_114_; 
lean_dec(v_h__1_107_);
v_one_113_ = lean_unsigned_to_nat(1u);
v_n_114_ = lean_nat_sub(v_x_103_, v_one_113_);
if (lean_obj_tag(v_x_104_) == 0)
{
lean_object* v___x_115_; 
lean_dec(v_h__3_109_);
v___x_115_ = lean_apply_3(v_h__2_108_, v_n_114_, v_x_105_, v_x_106_);
return v___x_115_;
}
else
{
lean_object* v_head_116_; lean_object* v_tail_117_; lean_object* v___x_118_; 
lean_dec(v_h__2_108_);
v_head_116_ = lean_ctor_get(v_x_104_, 0);
lean_inc(v_head_116_);
v_tail_117_ = lean_ctor_get(v_x_104_, 1);
lean_inc(v_tail_117_);
lean_dec_ref_known(v_x_104_, 2);
v___x_118_ = lean_apply_5(v_h__3_109_, v_n_114_, v_head_116_, v_tail_117_, v_x_105_, v_x_106_);
return v___x_118_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Data_List_Sublists_0__List_sublistsLenAux_match__1_splitter___redArg___boxed(lean_object* v_x_119_, lean_object* v_x_120_, lean_object* v_x_121_, lean_object* v_x_122_, lean_object* v_h__1_123_, lean_object* v_h__2_124_, lean_object* v_h__3_125_){
_start:
{
lean_object* v_res_126_; 
v_res_126_ = lp_mathlib___private_Mathlib_Data_List_Sublists_0__List_sublistsLenAux_match__1_splitter___redArg(v_x_119_, v_x_120_, v_x_121_, v_x_122_, v_h__1_123_, v_h__2_124_, v_h__3_125_);
lean_dec(v_x_119_);
return v_res_126_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Data_List_Sublists_0__List_sublistsLenAux_match__1_splitter(lean_object* v_00_u03b1_127_, lean_object* v_00_u03b2_128_, lean_object* v_motive_129_, lean_object* v_x_130_, lean_object* v_x_131_, lean_object* v_x_132_, lean_object* v_x_133_, lean_object* v_h__1_134_, lean_object* v_h__2_135_, lean_object* v_h__3_136_){
_start:
{
lean_object* v_zero_137_; uint8_t v_isZero_138_; 
v_zero_137_ = lean_unsigned_to_nat(0u);
v_isZero_138_ = lean_nat_dec_eq(v_x_130_, v_zero_137_);
if (v_isZero_138_ == 1)
{
lean_object* v___x_139_; 
lean_dec(v_h__3_136_);
lean_dec(v_h__2_135_);
v___x_139_ = lean_apply_3(v_h__1_134_, v_x_131_, v_x_132_, v_x_133_);
return v___x_139_;
}
else
{
lean_object* v_one_140_; lean_object* v_n_141_; 
lean_dec(v_h__1_134_);
v_one_140_ = lean_unsigned_to_nat(1u);
v_n_141_ = lean_nat_sub(v_x_130_, v_one_140_);
if (lean_obj_tag(v_x_131_) == 0)
{
lean_object* v___x_142_; 
lean_dec(v_h__3_136_);
v___x_142_ = lean_apply_3(v_h__2_135_, v_n_141_, v_x_132_, v_x_133_);
return v___x_142_;
}
else
{
lean_object* v_head_143_; lean_object* v_tail_144_; lean_object* v___x_145_; 
lean_dec(v_h__2_135_);
v_head_143_ = lean_ctor_get(v_x_131_, 0);
lean_inc(v_head_143_);
v_tail_144_ = lean_ctor_get(v_x_131_, 1);
lean_inc(v_tail_144_);
lean_dec_ref_known(v_x_131_, 2);
v___x_145_ = lean_apply_5(v_h__3_136_, v_n_141_, v_head_143_, v_tail_144_, v_x_132_, v_x_133_);
return v___x_145_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Data_List_Sublists_0__List_sublistsLenAux_match__1_splitter___boxed(lean_object* v_00_u03b1_146_, lean_object* v_00_u03b2_147_, lean_object* v_motive_148_, lean_object* v_x_149_, lean_object* v_x_150_, lean_object* v_x_151_, lean_object* v_x_152_, lean_object* v_h__1_153_, lean_object* v_h__2_154_, lean_object* v_h__3_155_){
_start:
{
lean_object* v_res_156_; 
v_res_156_ = lp_mathlib___private_Mathlib_Data_List_Sublists_0__List_sublistsLenAux_match__1_splitter(v_00_u03b1_146_, v_00_u03b2_147_, v_motive_148_, v_x_149_, v_x_150_, v_x_151_, v_x_152_, v_h__1_153_, v_h__2_154_, v_h__3_155_);
lean_dec(v_x_149_);
return v_res_156_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Data_Nat_Choose_Basic(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Data_List_Perm_Basic(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Data_List_Lex(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Data_List_Induction(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Data_List_Nodup(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Data_Prod_Basic(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Tactic_Finiteness_Attr(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_Data_List_Sublists(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Data_Nat_Choose_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Data_List_Perm_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Data_List_Lex(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Data_List_Induction(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Data_List_Nodup(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Data_Prod_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Tactic_Finiteness_Attr(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_Data_List_Sublists(uint8_t builtin) {
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
lean_object* initialize_mathlib_Mathlib_Data_Nat_Choose_Basic(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Data_List_Perm_Basic(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Data_List_Lex(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Data_List_Induction(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Data_List_Nodup(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Data_Prod_Basic(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Tactic_Finiteness_Attr(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Data_List_Sublists(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_Nat_Choose_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_List_Perm_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_List_Lex(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_List_Induction(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_List_Nodup(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_Prod_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Tactic_Finiteness_Attr(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Data_List_Sublists(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_Data_List_Sublists(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_Data_List_Sublists(builtin);
}
#ifdef __cplusplus
}
#endif
