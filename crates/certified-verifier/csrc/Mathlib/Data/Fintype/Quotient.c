// Lean compiler output
// Module: Mathlib.Data.Fintype.Quotient
// Imports: public import Init public meta import Init public import Mathlib.Data.List.Pi public import Mathlib.Data.Fintype.Defs
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
lean_object* lp_mathlib_Equiv_subtypeQuotientEquivQuotientSubtype(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_List_Pi_tail___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_List_Pi_cons___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Quotient_eval___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Quotient_listChoice___redArg___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Quotient_listChoice___redArg___lam__0___boxed(lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_Quotient_listChoice___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Quotient_listChoice___redArg___lam__0___boxed, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Quotient_listChoice___redArg___closed__0 = (const lean_object*)&lp_mathlib_Quotient_listChoice___redArg___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_Quotient_listChoice___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Quotient_listChoice(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Quotient_listChoice___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Data_Fintype_Quotient_0__Quotient_listChoice_match__3_splitter___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Data_Fintype_Quotient_0__Quotient_listChoice_match__3_splitter(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Data_Fintype_Quotient_0__Quotient_listChoice_match__3_splitter___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Data_Fintype_Quotient_0__Quotient_listChoice_match__1_splitter(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Data_Fintype_Quotient_0__Quotient_listChoice_match__1_splitter___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Quotient_finChoice___redArg___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Quotient_finChoice___redArg___lam__1(lean_object*, lean_object*, lean_object*, lean_object*);
static lean_once_cell_t lp_mathlib_Quotient_finChoice___redArg___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Quotient_finChoice___redArg___closed__0;
LEAN_EXPORT lean_object* lp_mathlib_Quotient_finChoice___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Quotient_finChoice(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Quotient_finChoice___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Quotient_finLiftOn___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Quotient_finLiftOn(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Quotient_finLiftOn___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Quotient_finChoiceEquiv___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Quotient_finChoiceEquiv(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Quotient_finHRecOn___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Quotient_finHRecOn(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Quotient_finHRecOn___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Quotient_finRecOn___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Quotient_finRecOn(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Quotient_finRecOn___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Trunc_finChoice___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Trunc_finChoice(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Trunc_finLiftOn___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Trunc_finLiftOn(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Trunc_finChoiceEquiv___redArg___lam__0(lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_Trunc_finChoiceEquiv___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Trunc_finChoiceEquiv___redArg___lam__0, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Trunc_finChoiceEquiv___redArg___closed__0 = (const lean_object*)&lp_mathlib_Trunc_finChoiceEquiv___redArg___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_Trunc_finChoiceEquiv___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Trunc_finChoiceEquiv(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Trunc_finRecOn___redArg___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Trunc_finRecOn___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Trunc_finRecOn(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Quotient_listChoice___redArg___lam__0(lean_object* v_a_1_, lean_object* v_a_2_){
_start:
{
lean_internal_panic_unreachable();
}
}
LEAN_EXPORT lean_object* lp_mathlib_Quotient_listChoice___redArg___lam__0___boxed(lean_object* v_a_3_, lean_object* v_a_4_){
_start:
{
lean_object* v_res_5_; 
v_res_5_ = lp_mathlib_Quotient_listChoice___redArg___lam__0(v_a_3_, v_a_4_);
lean_dec(v_a_3_);
return v_res_5_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Quotient_listChoice___redArg(lean_object* v_inst_7_, lean_object* v_l_8_, lean_object* v_q_9_){
_start:
{
if (lean_obj_tag(v_l_8_) == 0)
{
lean_object* v___f_10_; 
lean_dec(v_q_9_);
lean_dec_ref(v_inst_7_);
v___f_10_ = ((lean_object*)(lp_mathlib_Quotient_listChoice___redArg___closed__0));
return v___f_10_;
}
else
{
lean_object* v_head_11_; lean_object* v_tail_12_; lean_object* v___x_13_; lean_object* v___x_14_; lean_object* v___x_15_; lean_object* v___x_16_; 
v_head_11_ = lean_ctor_get(v_l_8_, 0);
lean_inc_n(v_head_11_, 3);
v_tail_12_ = lean_ctor_get(v_l_8_, 1);
lean_inc_n(v_tail_12_, 3);
lean_dec_ref_known(v_l_8_, 2);
lean_inc(v_q_9_);
v___x_13_ = lean_apply_2(v_q_9_, v_head_11_, lean_box(0));
v___x_14_ = lean_alloc_closure((void*)(lp_mathlib_List_Pi_tail___boxed), 7, 5);
lean_closure_set(v___x_14_, 0, lean_box(0));
lean_closure_set(v___x_14_, 1, lean_box(0));
lean_closure_set(v___x_14_, 2, v_head_11_);
lean_closure_set(v___x_14_, 3, v_tail_12_);
lean_closure_set(v___x_14_, 4, v_q_9_);
lean_inc_ref(v_inst_7_);
v___x_15_ = lp_mathlib_Quotient_listChoice___redArg(v_inst_7_, v_tail_12_, v___x_14_);
v___x_16_ = lean_alloc_closure((void*)(lp_mathlib_List_Pi_cons___boxed), 9, 7);
lean_closure_set(v___x_16_, 0, lean_box(0));
lean_closure_set(v___x_16_, 1, v_inst_7_);
lean_closure_set(v___x_16_, 2, lean_box(0));
lean_closure_set(v___x_16_, 3, v_head_11_);
lean_closure_set(v___x_16_, 4, v_tail_12_);
lean_closure_set(v___x_16_, 5, v___x_13_);
lean_closure_set(v___x_16_, 6, v___x_15_);
return v___x_16_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Quotient_listChoice(lean_object* v_00_u03b9_17_, lean_object* v_inst_18_, lean_object* v_00_u03b1_19_, lean_object* v_S_20_, lean_object* v_l_21_, lean_object* v_q_22_){
_start:
{
lean_object* v___x_23_; 
v___x_23_ = lp_mathlib_Quotient_listChoice___redArg(v_inst_18_, v_l_21_, v_q_22_);
return v___x_23_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Quotient_listChoice___boxed(lean_object* v_00_u03b9_24_, lean_object* v_inst_25_, lean_object* v_00_u03b1_26_, lean_object* v_S_27_, lean_object* v_l_28_, lean_object* v_q_29_){
_start:
{
lean_object* v_res_30_; 
v_res_30_ = lp_mathlib_Quotient_listChoice(v_00_u03b9_24_, v_inst_25_, v_00_u03b1_26_, v_S_27_, v_l_28_, v_q_29_);
lean_dec_ref(v_S_27_);
return v_res_30_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Data_Fintype_Quotient_0__Quotient_listChoice_match__3_splitter___redArg(lean_object* v_l_31_, lean_object* v_q_32_, lean_object* v_h__1_33_, lean_object* v_h__2_34_){
_start:
{
if (lean_obj_tag(v_l_31_) == 0)
{
lean_object* v___x_35_; 
lean_dec(v_h__2_34_);
v___x_35_ = lean_apply_1(v_h__1_33_, v_q_32_);
return v___x_35_;
}
else
{
lean_object* v_head_36_; lean_object* v_tail_37_; lean_object* v___x_38_; 
lean_dec(v_h__1_33_);
v_head_36_ = lean_ctor_get(v_l_31_, 0);
lean_inc(v_head_36_);
v_tail_37_ = lean_ctor_get(v_l_31_, 1);
lean_inc(v_tail_37_);
lean_dec_ref_known(v_l_31_, 2);
v___x_38_ = lean_apply_3(v_h__2_34_, v_head_36_, v_tail_37_, v_q_32_);
return v___x_38_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Data_Fintype_Quotient_0__Quotient_listChoice_match__3_splitter(lean_object* v_00_u03b9_39_, lean_object* v_00_u03b1_40_, lean_object* v_S_41_, lean_object* v_motive_42_, lean_object* v_l_43_, lean_object* v_q_44_, lean_object* v_h__1_45_, lean_object* v_h__2_46_){
_start:
{
if (lean_obj_tag(v_l_43_) == 0)
{
lean_object* v___x_47_; 
lean_dec(v_h__2_46_);
v___x_47_ = lean_apply_1(v_h__1_45_, v_q_44_);
return v___x_47_;
}
else
{
lean_object* v_head_48_; lean_object* v_tail_49_; lean_object* v___x_50_; 
lean_dec(v_h__1_45_);
v_head_48_ = lean_ctor_get(v_l_43_, 0);
lean_inc(v_head_48_);
v_tail_49_ = lean_ctor_get(v_l_43_, 1);
lean_inc(v_tail_49_);
lean_dec_ref_known(v_l_43_, 2);
v___x_50_ = lean_apply_3(v_h__2_46_, v_head_48_, v_tail_49_, v_q_44_);
return v___x_50_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Data_Fintype_Quotient_0__Quotient_listChoice_match__3_splitter___boxed(lean_object* v_00_u03b9_51_, lean_object* v_00_u03b1_52_, lean_object* v_S_53_, lean_object* v_motive_54_, lean_object* v_l_55_, lean_object* v_q_56_, lean_object* v_h__1_57_, lean_object* v_h__2_58_){
_start:
{
lean_object* v_res_59_; 
v_res_59_ = lp_mathlib___private_Mathlib_Data_Fintype_Quotient_0__Quotient_listChoice_match__3_splitter(v_00_u03b9_51_, v_00_u03b1_52_, v_S_53_, v_motive_54_, v_l_55_, v_q_56_, v_h__1_57_, v_h__2_58_);
lean_dec_ref(v_S_53_);
return v_res_59_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Data_Fintype_Quotient_0__Quotient_listChoice_match__1_splitter(lean_object* v_00_u03b9_60_, lean_object* v_motive_61_, lean_object* v_a_62_, lean_object* v_a_63_){
_start:
{
lean_internal_panic_unreachable();
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Data_Fintype_Quotient_0__Quotient_listChoice_match__1_splitter___boxed(lean_object* v_00_u03b9_64_, lean_object* v_motive_65_, lean_object* v_a_66_, lean_object* v_a_67_){
_start:
{
lean_object* v_res_68_; 
v_res_68_ = lp_mathlib___private_Mathlib_Data_Fintype_Quotient_0__Quotient_listChoice_match__1_splitter(v_00_u03b9_64_, v_motive_65_, v_a_66_, v_a_67_);
lean_dec(v_a_66_);
return v_res_68_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Quotient_finChoice___redArg___lam__0(lean_object* v_q_69_, lean_object* v_i_70_, lean_object* v_x_71_){
_start:
{
lean_object* v___x_72_; 
v___x_72_ = lean_apply_1(v_q_69_, v_i_70_);
return v___x_72_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Quotient_finChoice___redArg___lam__1(lean_object* v_inst_73_, lean_object* v_e_74_, lean_object* v___f_75_, lean_object* v___y_76_){
_start:
{
lean_object* v___x_67__overap_77_; lean_object* v___x_78_; 
v___x_67__overap_77_ = lp_mathlib_Quotient_listChoice___redArg(v_inst_73_, v_e_74_, v___f_75_);
v___x_78_ = lean_apply_2(v___x_67__overap_77_, v___y_76_, lean_box(0));
return v___x_78_;
}
}
static lean_object* _init_lp_mathlib_Quotient_finChoice___redArg___closed__0(void){
_start:
{
lean_object* v___x_79_; lean_object* v___x_80_; 
v___x_79_ = lean_box(0);
v___x_80_ = lp_mathlib_Equiv_subtypeQuotientEquivQuotientSubtype(lean_box(0), lean_box(0), v___x_79_, v___x_79_, lean_box(0), lean_box(0), lean_box(0));
return v___x_80_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Quotient_finChoice___redArg(lean_object* v_inst_81_, lean_object* v_inst_82_, lean_object* v_q_83_){
_start:
{
lean_object* v___x_84_; lean_object* v_toFun_85_; lean_object* v___f_86_; lean_object* v_e_87_; lean_object* v___f_88_; 
v___x_84_ = lean_obj_once(&lp_mathlib_Quotient_finChoice___redArg___closed__0, &lp_mathlib_Quotient_finChoice___redArg___closed__0_once, _init_lp_mathlib_Quotient_finChoice___redArg___closed__0);
v_toFun_85_ = lean_ctor_get(v___x_84_, 0);
v___f_86_ = lean_alloc_closure((void*)(lp_mathlib_Quotient_finChoice___redArg___lam__0), 3, 1);
lean_closure_set(v___f_86_, 0, v_q_83_);
lean_inc(v_toFun_85_);
v_e_87_ = lean_apply_1(v_toFun_85_, v_inst_81_);
v___f_88_ = lean_alloc_closure((void*)(lp_mathlib_Quotient_finChoice___redArg___lam__1), 4, 3);
lean_closure_set(v___f_88_, 0, v_inst_82_);
lean_closure_set(v___f_88_, 1, v_e_87_);
lean_closure_set(v___f_88_, 2, v___f_86_);
return v___f_88_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Quotient_finChoice(lean_object* v_00_u03b9_89_, lean_object* v_inst_90_, lean_object* v_inst_91_, lean_object* v_00_u03b1_92_, lean_object* v_S_93_, lean_object* v_q_94_){
_start:
{
lean_object* v___x_95_; 
v___x_95_ = lp_mathlib_Quotient_finChoice___redArg(v_inst_90_, v_inst_91_, v_q_94_);
return v___x_95_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Quotient_finChoice___boxed(lean_object* v_00_u03b9_96_, lean_object* v_inst_97_, lean_object* v_inst_98_, lean_object* v_00_u03b1_99_, lean_object* v_S_100_, lean_object* v_q_101_){
_start:
{
lean_object* v_res_102_; 
v_res_102_ = lp_mathlib_Quotient_finChoice(v_00_u03b9_96_, v_inst_97_, v_inst_98_, v_00_u03b1_99_, v_S_100_, v_q_101_);
lean_dec_ref(v_S_100_);
return v_res_102_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Quotient_finLiftOn___redArg(lean_object* v_inst_103_, lean_object* v_inst_104_, lean_object* v_q_105_, lean_object* v_f_106_){
_start:
{
lean_object* v___x_107_; lean_object* v___x_108_; 
v___x_107_ = lp_mathlib_Quotient_finChoice___redArg(v_inst_103_, v_inst_104_, v_q_105_);
v___x_108_ = lean_apply_1(v_f_106_, v___x_107_);
return v___x_108_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Quotient_finLiftOn(lean_object* v_00_u03b9_109_, lean_object* v_inst_110_, lean_object* v_inst_111_, lean_object* v_00_u03b1_112_, lean_object* v_S_113_, lean_object* v_00_u03b2_114_, lean_object* v_q_115_, lean_object* v_f_116_, lean_object* v_h_117_){
_start:
{
lean_object* v___x_118_; 
v___x_118_ = lp_mathlib_Quotient_finLiftOn___redArg(v_inst_110_, v_inst_111_, v_q_115_, v_f_116_);
return v___x_118_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Quotient_finLiftOn___boxed(lean_object* v_00_u03b9_119_, lean_object* v_inst_120_, lean_object* v_inst_121_, lean_object* v_00_u03b1_122_, lean_object* v_S_123_, lean_object* v_00_u03b2_124_, lean_object* v_q_125_, lean_object* v_f_126_, lean_object* v_h_127_){
_start:
{
lean_object* v_res_128_; 
v_res_128_ = lp_mathlib_Quotient_finLiftOn(v_00_u03b9_119_, v_inst_120_, v_inst_121_, v_00_u03b1_122_, v_S_123_, v_00_u03b2_124_, v_q_125_, v_f_126_, v_h_127_);
lean_dec_ref(v_S_123_);
return v_res_128_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Quotient_finChoiceEquiv___redArg(lean_object* v_inst_129_, lean_object* v_inst_130_, lean_object* v_S_131_){
_start:
{
lean_object* v___x_132_; lean_object* v___x_133_; lean_object* v___x_134_; 
lean_inc_ref(v_S_131_);
v___x_132_ = lean_alloc_closure((void*)(lp_mathlib_Quotient_finChoice___boxed), 6, 5);
lean_closure_set(v___x_132_, 0, lean_box(0));
lean_closure_set(v___x_132_, 1, v_inst_129_);
lean_closure_set(v___x_132_, 2, v_inst_130_);
lean_closure_set(v___x_132_, 3, lean_box(0));
lean_closure_set(v___x_132_, 4, v_S_131_);
v___x_133_ = lean_alloc_closure((void*)(lp_mathlib_Quotient_eval___boxed), 5, 3);
lean_closure_set(v___x_133_, 0, lean_box(0));
lean_closure_set(v___x_133_, 1, lean_box(0));
lean_closure_set(v___x_133_, 2, v_S_131_);
v___x_134_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_134_, 0, v___x_132_);
lean_ctor_set(v___x_134_, 1, v___x_133_);
return v___x_134_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Quotient_finChoiceEquiv(lean_object* v_00_u03b9_135_, lean_object* v_inst_136_, lean_object* v_inst_137_, lean_object* v_00_u03b1_138_, lean_object* v_S_139_){
_start:
{
lean_object* v___x_140_; 
v___x_140_ = lp_mathlib_Quotient_finChoiceEquiv___redArg(v_inst_136_, v_inst_137_, v_S_139_);
return v___x_140_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Quotient_finHRecOn___redArg(lean_object* v_inst_141_, lean_object* v_inst_142_, lean_object* v_q_143_, lean_object* v_f_144_){
_start:
{
lean_object* v___x_145_; lean_object* v___x_146_; 
v___x_145_ = lp_mathlib_Quotient_finChoice___redArg(v_inst_141_, v_inst_142_, v_q_143_);
v___x_146_ = lean_apply_1(v_f_144_, v___x_145_);
return v___x_146_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Quotient_finHRecOn(lean_object* v_00_u03b9_147_, lean_object* v_inst_148_, lean_object* v_inst_149_, lean_object* v_00_u03b1_150_, lean_object* v_S_151_, lean_object* v_C_152_, lean_object* v_q_153_, lean_object* v_f_154_, lean_object* v_h_155_){
_start:
{
lean_object* v___x_156_; 
v___x_156_ = lp_mathlib_Quotient_finHRecOn___redArg(v_inst_148_, v_inst_149_, v_q_153_, v_f_154_);
return v___x_156_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Quotient_finHRecOn___boxed(lean_object* v_00_u03b9_157_, lean_object* v_inst_158_, lean_object* v_inst_159_, lean_object* v_00_u03b1_160_, lean_object* v_S_161_, lean_object* v_C_162_, lean_object* v_q_163_, lean_object* v_f_164_, lean_object* v_h_165_){
_start:
{
lean_object* v_res_166_; 
v_res_166_ = lp_mathlib_Quotient_finHRecOn(v_00_u03b9_157_, v_inst_158_, v_inst_159_, v_00_u03b1_160_, v_S_161_, v_C_162_, v_q_163_, v_f_164_, v_h_165_);
lean_dec_ref(v_S_161_);
return v_res_166_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Quotient_finRecOn___redArg(lean_object* v_inst_167_, lean_object* v_inst_168_, lean_object* v_q_169_, lean_object* v_f_170_){
_start:
{
lean_object* v___x_171_; 
v___x_171_ = lp_mathlib_Quotient_finHRecOn___redArg(v_inst_167_, v_inst_168_, v_q_169_, v_f_170_);
return v___x_171_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Quotient_finRecOn(lean_object* v_00_u03b9_172_, lean_object* v_inst_173_, lean_object* v_inst_174_, lean_object* v_00_u03b1_175_, lean_object* v_S_176_, lean_object* v_C_177_, lean_object* v_q_178_, lean_object* v_f_179_, lean_object* v_h_180_){
_start:
{
lean_object* v___x_181_; 
v___x_181_ = lp_mathlib_Quotient_finHRecOn___redArg(v_inst_173_, v_inst_174_, v_q_178_, v_f_179_);
return v___x_181_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Quotient_finRecOn___boxed(lean_object* v_00_u03b9_182_, lean_object* v_inst_183_, lean_object* v_inst_184_, lean_object* v_00_u03b1_185_, lean_object* v_S_186_, lean_object* v_C_187_, lean_object* v_q_188_, lean_object* v_f_189_, lean_object* v_h_190_){
_start:
{
lean_object* v_res_191_; 
v_res_191_ = lp_mathlib_Quotient_finRecOn(v_00_u03b9_182_, v_inst_183_, v_inst_184_, v_00_u03b1_185_, v_S_186_, v_C_187_, v_q_188_, v_f_189_, v_h_190_);
lean_dec_ref(v_S_186_);
return v_res_191_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Trunc_finChoice___redArg(lean_object* v_inst_192_, lean_object* v_inst_193_, lean_object* v_q_194_){
_start:
{
lean_object* v___x_195_; 
v___x_195_ = lp_mathlib_Quotient_finChoice___redArg(v_inst_193_, v_inst_192_, v_q_194_);
return v___x_195_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Trunc_finChoice(lean_object* v_00_u03b9_196_, lean_object* v_inst_197_, lean_object* v_inst_198_, lean_object* v_00_u03b1_199_, lean_object* v_q_200_){
_start:
{
lean_object* v___x_201_; 
v___x_201_ = lp_mathlib_Quotient_finChoice___redArg(v_inst_198_, v_inst_197_, v_q_200_);
return v___x_201_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Trunc_finLiftOn___redArg(lean_object* v_inst_202_, lean_object* v_inst_203_, lean_object* v_q_204_, lean_object* v_f_205_){
_start:
{
lean_object* v___x_206_; 
v___x_206_ = lp_mathlib_Quotient_finLiftOn___redArg(v_inst_203_, v_inst_202_, v_q_204_, v_f_205_);
return v___x_206_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Trunc_finLiftOn(lean_object* v_00_u03b9_207_, lean_object* v_inst_208_, lean_object* v_inst_209_, lean_object* v_00_u03b1_210_, lean_object* v_00_u03b2_211_, lean_object* v_q_212_, lean_object* v_f_213_, lean_object* v_h_214_){
_start:
{
lean_object* v___x_215_; 
v___x_215_ = lp_mathlib_Quotient_finLiftOn___redArg(v_inst_209_, v_inst_208_, v_q_212_, v_f_213_);
return v___x_215_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Trunc_finChoiceEquiv___redArg___lam__0(lean_object* v_q_216_, lean_object* v_i_217_){
_start:
{
lean_object* v___x_218_; 
v___x_218_ = lean_apply_1(v_q_216_, v_i_217_);
return v___x_218_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Trunc_finChoiceEquiv___redArg(lean_object* v_inst_220_, lean_object* v_inst_221_){
_start:
{
lean_object* v___f_222_; lean_object* v___x_223_; lean_object* v___x_224_; 
v___f_222_ = ((lean_object*)(lp_mathlib_Trunc_finChoiceEquiv___redArg___closed__0));
v___x_223_ = lean_alloc_closure((void*)(lp_mathlib_Trunc_finChoice), 5, 4);
lean_closure_set(v___x_223_, 0, lean_box(0));
lean_closure_set(v___x_223_, 1, v_inst_220_);
lean_closure_set(v___x_223_, 2, v_inst_221_);
lean_closure_set(v___x_223_, 3, lean_box(0));
v___x_224_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_224_, 0, v___x_223_);
lean_ctor_set(v___x_224_, 1, v___f_222_);
return v___x_224_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Trunc_finChoiceEquiv(lean_object* v_00_u03b9_225_, lean_object* v_inst_226_, lean_object* v_inst_227_, lean_object* v_00_u03b1_228_){
_start:
{
lean_object* v___x_229_; 
v___x_229_ = lp_mathlib_Trunc_finChoiceEquiv___redArg(v_inst_226_, v_inst_227_);
return v___x_229_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Trunc_finRecOn___redArg___lam__0(lean_object* v_f_230_, lean_object* v_x_231_){
_start:
{
lean_object* v___x_232_; 
v___x_232_ = lean_apply_1(v_f_230_, v_x_231_);
return v___x_232_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Trunc_finRecOn___redArg(lean_object* v_inst_233_, lean_object* v_inst_234_, lean_object* v_q_235_, lean_object* v_f_236_){
_start:
{
lean_object* v___f_237_; lean_object* v___x_238_; 
v___f_237_ = lean_alloc_closure((void*)(lp_mathlib_Trunc_finRecOn___redArg___lam__0), 2, 1);
lean_closure_set(v___f_237_, 0, v_f_236_);
v___x_238_ = lp_mathlib_Quotient_finHRecOn___redArg(v_inst_234_, v_inst_233_, v_q_235_, v___f_237_);
return v___x_238_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Trunc_finRecOn(lean_object* v_00_u03b9_239_, lean_object* v_inst_240_, lean_object* v_inst_241_, lean_object* v_00_u03b1_242_, lean_object* v_C_243_, lean_object* v_q_244_, lean_object* v_f_245_, lean_object* v_h_246_){
_start:
{
lean_object* v___x_247_; 
v___x_247_ = lp_mathlib_Trunc_finRecOn___redArg(v_inst_240_, v_inst_241_, v_q_244_, v_f_245_);
return v___x_247_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Data_List_Pi(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Data_Fintype_Defs(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_Data_Fintype_Quotient(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Data_List_Pi(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Data_Fintype_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_Data_Fintype_Quotient(uint8_t builtin) {
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
lean_object* initialize_mathlib_Mathlib_Data_List_Pi(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Data_Fintype_Defs(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Data_Fintype_Quotient(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_List_Pi(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_Fintype_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Data_Fintype_Quotient(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_Data_Fintype_Quotient(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_Data_Fintype_Quotient(builtin);
}
#ifdef __cplusplus
}
#endif
