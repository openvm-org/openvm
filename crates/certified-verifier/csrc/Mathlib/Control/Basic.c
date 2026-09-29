// Lean compiler output
// Module: Mathlib.Control.Basic
// Imports: public import Init public meta import Init public import Mathlib.Control.Combinators public import Mathlib.Tactic.CasesM public import Mathlib.Tactic.Attr.Core import Mathlib.Tactic.Attr.Register
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
lean_object* l_Function_comp(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Function_const___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_zipWithM___redArg___lam__0(lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_zipWithM___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_zipWithM___redArg___lam__0, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_zipWithM___redArg___closed__0 = (const lean_object*)&lp_mathlib_zipWithM___redArg___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_zipWithM___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_zipWithM___redArg___lam__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_zipWithM(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_zipWithM_x27___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_zipWithM_x27___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_zipWithM_x27(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_mapAccumRM___redArg___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_mapAccumRM___redArg___lam__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_mapAccumRM___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_mapAccumRM(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_mapAccumLM___redArg___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_mapAccumLM___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_mapAccumLM___redArg___lam__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_mapAccumLM(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_succeeds___redArg___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_succeeds___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_succeeds(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_tryM___redArg___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_tryM___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_tryM(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_try_x3f___redArg___lam__0(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_try_x3f___redArg___lam__1(lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_try_x3f___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_try_x3f___redArg___lam__0, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_try_x3f___redArg___closed__0 = (const lean_object*)&lp_mathlib_try_x3f___redArg___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_try_x3f___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_try_x3f(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Sum_bind___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Sum_bind(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Sum_instMonad__mathlib___lam__0(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Sum_instMonad__mathlib___lam__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Sum_instMonad__mathlib___lam__3(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Sum_instMonad__mathlib___lam__2(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Sum_instMonad__mathlib___lam__5(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Sum_instMonad__mathlib___lam__4(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Sum_instMonad__mathlib___lam__6(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Sum_instMonad__mathlib___lam__6___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Sum_instMonad__mathlib___lam__7(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Sum_instMonad__mathlib___lam__8(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Sum_instMonad__mathlib___lam__9(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Sum_instMonad__mathlib___lam__9___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Sum_instMonad__mathlib___lam__10(lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_Sum_instMonad__mathlib___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Sum_instMonad__mathlib___lam__0, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Sum_instMonad__mathlib___closed__0 = (const lean_object*)&lp_mathlib_Sum_instMonad__mathlib___closed__0_value;
static const lean_closure_object lp_mathlib_Sum_instMonad__mathlib___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*1, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Sum_instMonad__mathlib___lam__1, .m_arity = 5, .m_num_fixed = 1, .m_objs = {((lean_object*)&lp_mathlib_Sum_instMonad__mathlib___closed__0_value)} };
static const lean_object* lp_mathlib_Sum_instMonad__mathlib___closed__1 = (const lean_object*)&lp_mathlib_Sum_instMonad__mathlib___closed__1_value;
static const lean_closure_object lp_mathlib_Sum_instMonad__mathlib___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*1, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Sum_instMonad__mathlib___lam__3, .m_arity = 5, .m_num_fixed = 1, .m_objs = {((lean_object*)&lp_mathlib_Sum_instMonad__mathlib___closed__0_value)} };
static const lean_object* lp_mathlib_Sum_instMonad__mathlib___closed__2 = (const lean_object*)&lp_mathlib_Sum_instMonad__mathlib___closed__2_value;
static const lean_closure_object lp_mathlib_Sum_instMonad__mathlib___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Sum_instMonad__mathlib___lam__2, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Sum_instMonad__mathlib___closed__3 = (const lean_object*)&lp_mathlib_Sum_instMonad__mathlib___closed__3_value;
static const lean_closure_object lp_mathlib_Sum_instMonad__mathlib___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*1, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Sum_instMonad__mathlib___lam__4, .m_arity = 5, .m_num_fixed = 1, .m_objs = {((lean_object*)&lp_mathlib_Sum_instMonad__mathlib___closed__0_value)} };
static const lean_object* lp_mathlib_Sum_instMonad__mathlib___closed__4 = (const lean_object*)&lp_mathlib_Sum_instMonad__mathlib___closed__4_value;
static const lean_closure_object lp_mathlib_Sum_instMonad__mathlib___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Sum_instMonad__mathlib___lam__8, .m_arity = 4, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Sum_instMonad__mathlib___closed__5 = (const lean_object*)&lp_mathlib_Sum_instMonad__mathlib___closed__5_value;
static const lean_closure_object lp_mathlib_Sum_instMonad__mathlib___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Sum_instMonad__mathlib___lam__10, .m_arity = 4, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Sum_instMonad__mathlib___closed__6 = (const lean_object*)&lp_mathlib_Sum_instMonad__mathlib___closed__6_value;
static const lean_ctor_object lp_mathlib_Sum_instMonad__mathlib___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_Sum_instMonad__mathlib___closed__1_value),((lean_object*)&lp_mathlib_Sum_instMonad__mathlib___closed__2_value)}};
static const lean_object* lp_mathlib_Sum_instMonad__mathlib___closed__7 = (const lean_object*)&lp_mathlib_Sum_instMonad__mathlib___closed__7_value;
static const lean_ctor_object lp_mathlib_Sum_instMonad__mathlib___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*5 + 0, .m_other = 5, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_Sum_instMonad__mathlib___closed__7_value),((lean_object*)&lp_mathlib_Sum_instMonad__mathlib___closed__3_value),((lean_object*)&lp_mathlib_Sum_instMonad__mathlib___closed__4_value),((lean_object*)&lp_mathlib_Sum_instMonad__mathlib___closed__5_value),((lean_object*)&lp_mathlib_Sum_instMonad__mathlib___closed__6_value)}};
static const lean_object* lp_mathlib_Sum_instMonad__mathlib___closed__8 = (const lean_object*)&lp_mathlib_Sum_instMonad__mathlib___closed__8_value;
static const lean_closure_object lp_mathlib_Sum_instMonad__mathlib___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*1, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Sum_bind, .m_arity = 5, .m_num_fixed = 1, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1))} };
static const lean_object* lp_mathlib_Sum_instMonad__mathlib___closed__9 = (const lean_object*)&lp_mathlib_Sum_instMonad__mathlib___closed__9_value;
static const lean_ctor_object lp_mathlib_Sum_instMonad__mathlib___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_Sum_instMonad__mathlib___closed__8_value),((lean_object*)&lp_mathlib_Sum_instMonad__mathlib___closed__9_value)}};
static const lean_object* lp_mathlib_Sum_instMonad__mathlib___closed__10 = (const lean_object*)&lp_mathlib_Sum_instMonad__mathlib___closed__10_value;
LEAN_EXPORT lean_object* lp_mathlib_Sum_instMonad__mathlib(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_zipWithM___redArg___lam__0(lean_object* v_x1_1_, lean_object* v_x2_2_){
_start:
{
lean_object* v___x_3_; 
v___x_3_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_3_, 0, v_x1_1_);
lean_ctor_set(v___x_3_, 1, v_x2_2_);
return v___x_3_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_zipWithM___redArg(lean_object* v_inst_5_, lean_object* v_f_6_, lean_object* v_x_7_, lean_object* v_x_8_){
_start:
{
lean_object* v_toFunctor_9_; lean_object* v_toPure_10_; lean_object* v_toSeq_11_; 
v_toFunctor_9_ = lean_ctor_get(v_inst_5_, 0);
v_toPure_10_ = lean_ctor_get(v_inst_5_, 1);
v_toSeq_11_ = lean_ctor_get(v_inst_5_, 2);
lean_inc(v_toSeq_11_);
if (lean_obj_tag(v_x_7_) == 1)
{
if (lean_obj_tag(v_x_8_) == 1)
{
lean_object* v_head_15_; lean_object* v_tail_16_; lean_object* v_head_17_; lean_object* v_tail_18_; lean_object* v_map_19_; lean_object* v___f_20_; lean_object* v___f_21_; lean_object* v___x_22_; lean_object* v___x_23_; lean_object* v___x_24_; 
v_head_15_ = lean_ctor_get(v_x_7_, 0);
lean_inc(v_head_15_);
v_tail_16_ = lean_ctor_get(v_x_7_, 1);
lean_inc(v_tail_16_);
lean_dec_ref_known(v_x_7_, 2);
v_head_17_ = lean_ctor_get(v_x_8_, 0);
lean_inc(v_head_17_);
v_tail_18_ = lean_ctor_get(v_x_8_, 1);
lean_inc(v_tail_18_);
lean_dec_ref_known(v_x_8_, 2);
v_map_19_ = lean_ctor_get(v_toFunctor_9_, 0);
lean_inc(v_map_19_);
v___f_20_ = ((lean_object*)(lp_mathlib_zipWithM___redArg___closed__0));
lean_inc(v_f_6_);
v___f_21_ = lean_alloc_closure((void*)(lp_mathlib_zipWithM___redArg___lam__1), 5, 4);
lean_closure_set(v___f_21_, 0, v_inst_5_);
lean_closure_set(v___f_21_, 1, v_f_6_);
lean_closure_set(v___f_21_, 2, v_tail_16_);
lean_closure_set(v___f_21_, 3, v_tail_18_);
v___x_22_ = lean_apply_2(v_f_6_, v_head_15_, v_head_17_);
v___x_23_ = lean_apply_4(v_map_19_, lean_box(0), lean_box(0), v___f_20_, v___x_22_);
v___x_24_ = lean_apply_4(v_toSeq_11_, lean_box(0), lean_box(0), v___x_23_, v___f_21_);
return v___x_24_;
}
else
{
lean_inc(v_toPure_10_);
lean_dec_ref_known(v_x_7_, 2);
lean_dec(v_toSeq_11_);
lean_dec(v_x_8_);
lean_dec(v_f_6_);
lean_dec_ref(v_inst_5_);
goto v___jp_12_;
}
}
else
{
lean_inc(v_toPure_10_);
lean_dec(v_toSeq_11_);
lean_dec(v_x_8_);
lean_dec(v_x_7_);
lean_dec(v_f_6_);
lean_dec_ref(v_inst_5_);
goto v___jp_12_;
}
v___jp_12_:
{
lean_object* v___x_13_; lean_object* v___x_14_; 
v___x_13_ = lean_box(0);
v___x_14_ = lean_apply_2(v_toPure_10_, lean_box(0), v___x_13_);
return v___x_14_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_zipWithM___redArg___lam__1(lean_object* v_inst_25_, lean_object* v_f_26_, lean_object* v_tail_27_, lean_object* v_tail_28_, lean_object* v_x_29_){
_start:
{
lean_object* v___x_30_; 
v___x_30_ = lp_mathlib_zipWithM___redArg(v_inst_25_, v_f_26_, v_tail_27_, v_tail_28_);
return v___x_30_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_zipWithM(lean_object* v_F_31_, lean_object* v_inst_32_, lean_object* v_00_u03b1_u2081_33_, lean_object* v_00_u03b1_u2082_34_, lean_object* v_00_u03c6_35_, lean_object* v_f_36_, lean_object* v_x_37_, lean_object* v_x_38_){
_start:
{
lean_object* v___x_39_; 
v___x_39_ = lp_mathlib_zipWithM___redArg(v_inst_32_, v_f_36_, v_x_37_, v_x_38_);
return v___x_39_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_zipWithM_x27___redArg(lean_object* v_inst_40_, lean_object* v_f_41_, lean_object* v_x_42_, lean_object* v_x_43_){
_start:
{
if (lean_obj_tag(v_x_42_) == 0)
{
lean_object* v_toPure_44_; lean_object* v___x_45_; lean_object* v___x_46_; 
lean_dec(v_x_43_);
lean_dec(v_f_41_);
v_toPure_44_ = lean_ctor_get(v_inst_40_, 1);
lean_inc(v_toPure_44_);
lean_dec_ref(v_inst_40_);
v___x_45_ = lean_box(0);
v___x_46_ = lean_apply_2(v_toPure_44_, lean_box(0), v___x_45_);
return v___x_46_;
}
else
{
if (lean_obj_tag(v_x_43_) == 0)
{
lean_object* v_toPure_47_; lean_object* v___x_48_; lean_object* v___x_49_; 
lean_dec_ref_known(v_x_42_, 2);
lean_dec(v_f_41_);
v_toPure_47_ = lean_ctor_get(v_inst_40_, 1);
lean_inc(v_toPure_47_);
lean_dec_ref(v_inst_40_);
v___x_48_ = lean_box(0);
v___x_49_ = lean_apply_2(v_toPure_47_, lean_box(0), v___x_48_);
return v___x_49_;
}
else
{
lean_object* v_toSeqRight_50_; lean_object* v_head_51_; lean_object* v_tail_52_; lean_object* v_head_53_; lean_object* v_tail_54_; lean_object* v___f_55_; lean_object* v___x_56_; lean_object* v___x_57_; 
v_toSeqRight_50_ = lean_ctor_get(v_inst_40_, 4);
lean_inc(v_toSeqRight_50_);
v_head_51_ = lean_ctor_get(v_x_42_, 0);
lean_inc(v_head_51_);
v_tail_52_ = lean_ctor_get(v_x_42_, 1);
lean_inc(v_tail_52_);
lean_dec_ref_known(v_x_42_, 2);
v_head_53_ = lean_ctor_get(v_x_43_, 0);
lean_inc(v_head_53_);
v_tail_54_ = lean_ctor_get(v_x_43_, 1);
lean_inc(v_tail_54_);
lean_dec_ref_known(v_x_43_, 2);
lean_inc(v_f_41_);
v___f_55_ = lean_alloc_closure((void*)(lp_mathlib_zipWithM_x27___redArg___lam__0), 5, 4);
lean_closure_set(v___f_55_, 0, v_inst_40_);
lean_closure_set(v___f_55_, 1, v_f_41_);
lean_closure_set(v___f_55_, 2, v_tail_52_);
lean_closure_set(v___f_55_, 3, v_tail_54_);
v___x_56_ = lean_apply_2(v_f_41_, v_head_51_, v_head_53_);
v___x_57_ = lean_apply_4(v_toSeqRight_50_, lean_box(0), lean_box(0), v___x_56_, v___f_55_);
return v___x_57_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_zipWithM_x27___redArg___lam__0(lean_object* v_inst_58_, lean_object* v_f_59_, lean_object* v_tail_60_, lean_object* v_tail_61_, lean_object* v_x_62_){
_start:
{
lean_object* v___x_63_; 
v___x_63_ = lp_mathlib_zipWithM_x27___redArg(v_inst_58_, v_f_59_, v_tail_60_, v_tail_61_);
return v___x_63_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_zipWithM_x27(lean_object* v_00_u03b1_64_, lean_object* v_00_u03b2_65_, lean_object* v_00_u03b3_66_, lean_object* v_F_67_, lean_object* v_inst_68_, lean_object* v_f_69_, lean_object* v_x_70_, lean_object* v_x_71_){
_start:
{
lean_object* v___x_72_; 
v___x_72_ = lp_mathlib_zipWithM_x27___redArg(v_inst_68_, v_f_69_, v_x_70_, v_x_71_);
return v___x_72_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_mapAccumRM___redArg___lam__0(lean_object* v_snd_73_, lean_object* v_toPure_74_, lean_object* v_____x_75_){
_start:
{
lean_object* v_fst_76_; lean_object* v_snd_77_; lean_object* v___x_79_; uint8_t v_isShared_80_; uint8_t v_isSharedCheck_86_; 
v_fst_76_ = lean_ctor_get(v_____x_75_, 0);
v_snd_77_ = lean_ctor_get(v_____x_75_, 1);
v_isSharedCheck_86_ = !lean_is_exclusive(v_____x_75_);
if (v_isSharedCheck_86_ == 0)
{
v___x_79_ = v_____x_75_;
v_isShared_80_ = v_isSharedCheck_86_;
goto v_resetjp_78_;
}
else
{
lean_inc(v_snd_77_);
lean_inc(v_fst_76_);
lean_dec(v_____x_75_);
v___x_79_ = lean_box(0);
v_isShared_80_ = v_isSharedCheck_86_;
goto v_resetjp_78_;
}
v_resetjp_78_:
{
lean_object* v___x_81_; lean_object* v___x_83_; 
v___x_81_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_81_, 0, v_snd_77_);
lean_ctor_set(v___x_81_, 1, v_snd_73_);
if (v_isShared_80_ == 0)
{
lean_ctor_set(v___x_79_, 1, v___x_81_);
v___x_83_ = v___x_79_;
goto v_reusejp_82_;
}
else
{
lean_object* v_reuseFailAlloc_85_; 
v_reuseFailAlloc_85_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_85_, 0, v_fst_76_);
lean_ctor_set(v_reuseFailAlloc_85_, 1, v___x_81_);
v___x_83_ = v_reuseFailAlloc_85_;
goto v_reusejp_82_;
}
v_reusejp_82_:
{
lean_object* v___x_84_; 
v___x_84_ = lean_apply_2(v_toPure_74_, lean_box(0), v___x_83_);
return v___x_84_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_mapAccumRM___redArg___lam__1(lean_object* v_toPure_87_, lean_object* v_f_88_, lean_object* v_head_89_, lean_object* v_toBind_90_, lean_object* v_____x_91_){
_start:
{
lean_object* v_fst_92_; lean_object* v_snd_93_; lean_object* v___f_94_; lean_object* v___x_95_; lean_object* v___x_96_; 
v_fst_92_ = lean_ctor_get(v_____x_91_, 0);
lean_inc(v_fst_92_);
v_snd_93_ = lean_ctor_get(v_____x_91_, 1);
lean_inc(v_snd_93_);
lean_dec_ref(v_____x_91_);
v___f_94_ = lean_alloc_closure((void*)(lp_mathlib_List_mapAccumRM___redArg___lam__0), 3, 2);
lean_closure_set(v___f_94_, 0, v_snd_93_);
lean_closure_set(v___f_94_, 1, v_toPure_87_);
v___x_95_ = lean_apply_2(v_f_88_, v_head_89_, v_fst_92_);
v___x_96_ = lean_apply_4(v_toBind_90_, lean_box(0), lean_box(0), v___x_95_, v___f_94_);
return v___x_96_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_mapAccumRM___redArg(lean_object* v_inst_97_, lean_object* v_f_98_, lean_object* v_x_99_, lean_object* v_x_100_){
_start:
{
if (lean_obj_tag(v_x_100_) == 0)
{
lean_object* v_toApplicative_101_; lean_object* v___x_103_; uint8_t v_isShared_104_; uint8_t v_isSharedCheck_111_; 
v_toApplicative_101_ = lean_ctor_get(v_inst_97_, 0);
lean_inc_ref(v_toApplicative_101_);
lean_dec(v_f_98_);
v_isSharedCheck_111_ = !lean_is_exclusive(v_inst_97_);
if (v_isSharedCheck_111_ == 0)
{
lean_object* v_unused_112_; lean_object* v_unused_113_; 
v_unused_112_ = lean_ctor_get(v_inst_97_, 1);
lean_dec(v_unused_112_);
v_unused_113_ = lean_ctor_get(v_inst_97_, 0);
lean_dec(v_unused_113_);
v___x_103_ = v_inst_97_;
v_isShared_104_ = v_isSharedCheck_111_;
goto v_resetjp_102_;
}
else
{
lean_dec(v_inst_97_);
v___x_103_ = lean_box(0);
v_isShared_104_ = v_isSharedCheck_111_;
goto v_resetjp_102_;
}
v_resetjp_102_:
{
lean_object* v_toPure_105_; lean_object* v___x_106_; lean_object* v___x_108_; 
v_toPure_105_ = lean_ctor_get(v_toApplicative_101_, 1);
lean_inc(v_toPure_105_);
lean_dec_ref(v_toApplicative_101_);
v___x_106_ = lean_box(0);
if (v_isShared_104_ == 0)
{
lean_ctor_set(v___x_103_, 1, v___x_106_);
lean_ctor_set(v___x_103_, 0, v_x_99_);
v___x_108_ = v___x_103_;
goto v_reusejp_107_;
}
else
{
lean_object* v_reuseFailAlloc_110_; 
v_reuseFailAlloc_110_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_110_, 0, v_x_99_);
lean_ctor_set(v_reuseFailAlloc_110_, 1, v___x_106_);
v___x_108_ = v_reuseFailAlloc_110_;
goto v_reusejp_107_;
}
v_reusejp_107_:
{
lean_object* v___x_109_; 
v___x_109_ = lean_apply_2(v_toPure_105_, lean_box(0), v___x_108_);
return v___x_109_;
}
}
}
else
{
lean_object* v_toApplicative_114_; lean_object* v_toBind_115_; lean_object* v_toPure_116_; lean_object* v_head_117_; lean_object* v_tail_118_; lean_object* v___f_119_; lean_object* v___x_120_; lean_object* v___x_121_; 
v_toApplicative_114_ = lean_ctor_get(v_inst_97_, 0);
v_toBind_115_ = lean_ctor_get(v_inst_97_, 1);
lean_inc_n(v_toBind_115_, 2);
v_toPure_116_ = lean_ctor_get(v_toApplicative_114_, 1);
v_head_117_ = lean_ctor_get(v_x_100_, 0);
lean_inc(v_head_117_);
v_tail_118_ = lean_ctor_get(v_x_100_, 1);
lean_inc(v_tail_118_);
lean_dec_ref_known(v_x_100_, 2);
lean_inc(v_f_98_);
lean_inc(v_toPure_116_);
v___f_119_ = lean_alloc_closure((void*)(lp_mathlib_List_mapAccumRM___redArg___lam__1), 5, 4);
lean_closure_set(v___f_119_, 0, v_toPure_116_);
lean_closure_set(v___f_119_, 1, v_f_98_);
lean_closure_set(v___f_119_, 2, v_head_117_);
lean_closure_set(v___f_119_, 3, v_toBind_115_);
v___x_120_ = lp_mathlib_List_mapAccumRM___redArg(v_inst_97_, v_f_98_, v_x_99_, v_tail_118_);
v___x_121_ = lean_apply_4(v_toBind_115_, lean_box(0), lean_box(0), v___x_120_, v___f_119_);
return v___x_121_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_mapAccumRM(lean_object* v_00_u03b1_122_, lean_object* v_00_u03b2_x27_123_, lean_object* v_00_u03b3_x27_124_, lean_object* v_m_x27_125_, lean_object* v_inst_126_, lean_object* v_f_127_, lean_object* v_x_128_, lean_object* v_x_129_){
_start:
{
lean_object* v___x_130_; 
v___x_130_ = lp_mathlib_List_mapAccumRM___redArg(v_inst_126_, v_f_127_, v_x_128_, v_x_129_);
return v___x_130_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_mapAccumLM___redArg___lam__0(lean_object* v_snd_131_, lean_object* v_toPure_132_, lean_object* v_____x_133_){
_start:
{
lean_object* v_fst_134_; lean_object* v_snd_135_; lean_object* v___x_137_; uint8_t v_isShared_138_; uint8_t v_isSharedCheck_144_; 
v_fst_134_ = lean_ctor_get(v_____x_133_, 0);
v_snd_135_ = lean_ctor_get(v_____x_133_, 1);
v_isSharedCheck_144_ = !lean_is_exclusive(v_____x_133_);
if (v_isSharedCheck_144_ == 0)
{
v___x_137_ = v_____x_133_;
v_isShared_138_ = v_isSharedCheck_144_;
goto v_resetjp_136_;
}
else
{
lean_inc(v_snd_135_);
lean_inc(v_fst_134_);
lean_dec(v_____x_133_);
v___x_137_ = lean_box(0);
v_isShared_138_ = v_isSharedCheck_144_;
goto v_resetjp_136_;
}
v_resetjp_136_:
{
lean_object* v___x_139_; lean_object* v___x_141_; 
v___x_139_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_139_, 0, v_snd_131_);
lean_ctor_set(v___x_139_, 1, v_snd_135_);
if (v_isShared_138_ == 0)
{
lean_ctor_set(v___x_137_, 1, v___x_139_);
v___x_141_ = v___x_137_;
goto v_reusejp_140_;
}
else
{
lean_object* v_reuseFailAlloc_143_; 
v_reuseFailAlloc_143_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_143_, 0, v_fst_134_);
lean_ctor_set(v_reuseFailAlloc_143_, 1, v___x_139_);
v___x_141_ = v_reuseFailAlloc_143_;
goto v_reusejp_140_;
}
v_reusejp_140_:
{
lean_object* v___x_142_; 
v___x_142_ = lean_apply_2(v_toPure_132_, lean_box(0), v___x_141_);
return v___x_142_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_mapAccumLM___redArg(lean_object* v_inst_145_, lean_object* v_f_146_, lean_object* v_x_147_, lean_object* v_x_148_){
_start:
{
if (lean_obj_tag(v_x_148_) == 0)
{
lean_object* v_toApplicative_149_; lean_object* v___x_151_; uint8_t v_isShared_152_; uint8_t v_isSharedCheck_159_; 
v_toApplicative_149_ = lean_ctor_get(v_inst_145_, 0);
lean_inc_ref(v_toApplicative_149_);
lean_dec(v_f_146_);
v_isSharedCheck_159_ = !lean_is_exclusive(v_inst_145_);
if (v_isSharedCheck_159_ == 0)
{
lean_object* v_unused_160_; lean_object* v_unused_161_; 
v_unused_160_ = lean_ctor_get(v_inst_145_, 1);
lean_dec(v_unused_160_);
v_unused_161_ = lean_ctor_get(v_inst_145_, 0);
lean_dec(v_unused_161_);
v___x_151_ = v_inst_145_;
v_isShared_152_ = v_isSharedCheck_159_;
goto v_resetjp_150_;
}
else
{
lean_dec(v_inst_145_);
v___x_151_ = lean_box(0);
v_isShared_152_ = v_isSharedCheck_159_;
goto v_resetjp_150_;
}
v_resetjp_150_:
{
lean_object* v_toPure_153_; lean_object* v___x_154_; lean_object* v___x_156_; 
v_toPure_153_ = lean_ctor_get(v_toApplicative_149_, 1);
lean_inc(v_toPure_153_);
lean_dec_ref(v_toApplicative_149_);
v___x_154_ = lean_box(0);
if (v_isShared_152_ == 0)
{
lean_ctor_set(v___x_151_, 1, v___x_154_);
lean_ctor_set(v___x_151_, 0, v_x_147_);
v___x_156_ = v___x_151_;
goto v_reusejp_155_;
}
else
{
lean_object* v_reuseFailAlloc_158_; 
v_reuseFailAlloc_158_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_158_, 0, v_x_147_);
lean_ctor_set(v_reuseFailAlloc_158_, 1, v___x_154_);
v___x_156_ = v_reuseFailAlloc_158_;
goto v_reusejp_155_;
}
v_reusejp_155_:
{
lean_object* v___x_157_; 
v___x_157_ = lean_apply_2(v_toPure_153_, lean_box(0), v___x_156_);
return v___x_157_;
}
}
}
else
{
lean_object* v_toApplicative_162_; lean_object* v_toBind_163_; lean_object* v_toPure_164_; lean_object* v_head_165_; lean_object* v_tail_166_; lean_object* v___f_167_; lean_object* v___x_168_; lean_object* v___x_169_; 
v_toApplicative_162_ = lean_ctor_get(v_inst_145_, 0);
v_toBind_163_ = lean_ctor_get(v_inst_145_, 1);
lean_inc_n(v_toBind_163_, 2);
v_toPure_164_ = lean_ctor_get(v_toApplicative_162_, 1);
lean_inc(v_toPure_164_);
v_head_165_ = lean_ctor_get(v_x_148_, 0);
lean_inc(v_head_165_);
v_tail_166_ = lean_ctor_get(v_x_148_, 1);
lean_inc(v_tail_166_);
lean_dec_ref_known(v_x_148_, 2);
lean_inc(v_f_146_);
v___f_167_ = lean_alloc_closure((void*)(lp_mathlib_List_mapAccumLM___redArg___lam__1), 6, 5);
lean_closure_set(v___f_167_, 0, v_toPure_164_);
lean_closure_set(v___f_167_, 1, v_inst_145_);
lean_closure_set(v___f_167_, 2, v_f_146_);
lean_closure_set(v___f_167_, 3, v_tail_166_);
lean_closure_set(v___f_167_, 4, v_toBind_163_);
v___x_168_ = lean_apply_2(v_f_146_, v_x_147_, v_head_165_);
v___x_169_ = lean_apply_4(v_toBind_163_, lean_box(0), lean_box(0), v___x_168_, v___f_167_);
return v___x_169_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_mapAccumLM___redArg___lam__1(lean_object* v_toPure_170_, lean_object* v_inst_171_, lean_object* v_f_172_, lean_object* v_tail_173_, lean_object* v_toBind_174_, lean_object* v_____x_175_){
_start:
{
lean_object* v_fst_176_; lean_object* v_snd_177_; lean_object* v___f_178_; lean_object* v___x_179_; lean_object* v___x_180_; 
v_fst_176_ = lean_ctor_get(v_____x_175_, 0);
lean_inc(v_fst_176_);
v_snd_177_ = lean_ctor_get(v_____x_175_, 1);
lean_inc(v_snd_177_);
lean_dec_ref(v_____x_175_);
v___f_178_ = lean_alloc_closure((void*)(lp_mathlib_List_mapAccumLM___redArg___lam__0), 3, 2);
lean_closure_set(v___f_178_, 0, v_snd_177_);
lean_closure_set(v___f_178_, 1, v_toPure_170_);
v___x_179_ = lp_mathlib_List_mapAccumLM___redArg(v_inst_171_, v_f_172_, v_fst_176_, v_tail_173_);
v___x_180_ = lean_apply_4(v_toBind_174_, lean_box(0), lean_box(0), v___x_179_, v___f_178_);
return v___x_180_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_mapAccumLM(lean_object* v_00_u03b1_181_, lean_object* v_00_u03b2_x27_182_, lean_object* v_00_u03b3_x27_183_, lean_object* v_m_x27_184_, lean_object* v_inst_185_, lean_object* v_f_186_, lean_object* v_x_187_, lean_object* v_x_188_){
_start:
{
lean_object* v___x_189_; 
v___x_189_ = lp_mathlib_List_mapAccumLM___redArg(v_inst_185_, v_f_186_, v_x_187_, v_x_188_);
return v___x_189_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_succeeds___redArg___lam__0(lean_object* v_toPure_190_, lean_object* v_x_191_){
_start:
{
uint8_t v___x_192_; lean_object* v___x_193_; lean_object* v___x_194_; 
v___x_192_ = 0;
v___x_193_ = lean_box(v___x_192_);
v___x_194_ = lean_apply_2(v_toPure_190_, lean_box(0), v___x_193_);
return v___x_194_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_succeeds___redArg(lean_object* v_inst_195_, lean_object* v_x_196_){
_start:
{
lean_object* v_toApplicative_197_; lean_object* v_toFunctor_198_; lean_object* v_orElse_199_; lean_object* v_toPure_200_; lean_object* v_mapConst_201_; lean_object* v___f_202_; uint8_t v___x_203_; lean_object* v___x_204_; lean_object* v___x_205_; lean_object* v___x_206_; 
v_toApplicative_197_ = lean_ctor_get(v_inst_195_, 0);
lean_inc_ref(v_toApplicative_197_);
v_toFunctor_198_ = lean_ctor_get(v_toApplicative_197_, 0);
lean_inc_ref(v_toFunctor_198_);
v_orElse_199_ = lean_ctor_get(v_inst_195_, 2);
lean_inc(v_orElse_199_);
lean_dec_ref(v_inst_195_);
v_toPure_200_ = lean_ctor_get(v_toApplicative_197_, 1);
lean_inc(v_toPure_200_);
lean_dec_ref(v_toApplicative_197_);
v_mapConst_201_ = lean_ctor_get(v_toFunctor_198_, 1);
lean_inc(v_mapConst_201_);
lean_dec_ref(v_toFunctor_198_);
v___f_202_ = lean_alloc_closure((void*)(lp_mathlib_succeeds___redArg___lam__0), 2, 1);
lean_closure_set(v___f_202_, 0, v_toPure_200_);
v___x_203_ = 1;
v___x_204_ = lean_box(v___x_203_);
v___x_205_ = lean_apply_4(v_mapConst_201_, lean_box(0), lean_box(0), v___x_204_, v_x_196_);
v___x_206_ = lean_apply_3(v_orElse_199_, lean_box(0), v___x_205_, v___f_202_);
return v___x_206_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_succeeds(lean_object* v_F_207_, lean_object* v_inst_208_, lean_object* v_00_u03b1_209_, lean_object* v_x_210_){
_start:
{
lean_object* v___x_211_; 
v___x_211_ = lp_mathlib_succeeds___redArg(v_inst_208_, v_x_210_);
return v___x_211_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_tryM___redArg___lam__0(lean_object* v_toPure_212_, lean_object* v___x_213_, lean_object* v_x_214_){
_start:
{
lean_object* v___x_215_; 
v___x_215_ = lean_apply_2(v_toPure_212_, lean_box(0), v___x_213_);
return v___x_215_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_tryM___redArg(lean_object* v_inst_216_, lean_object* v_x_217_){
_start:
{
lean_object* v_toApplicative_218_; lean_object* v_toFunctor_219_; lean_object* v_orElse_220_; lean_object* v_toPure_221_; lean_object* v_mapConst_222_; lean_object* v___x_223_; lean_object* v___f_224_; lean_object* v___x_225_; lean_object* v___x_226_; 
v_toApplicative_218_ = lean_ctor_get(v_inst_216_, 0);
lean_inc_ref(v_toApplicative_218_);
v_toFunctor_219_ = lean_ctor_get(v_toApplicative_218_, 0);
lean_inc_ref(v_toFunctor_219_);
v_orElse_220_ = lean_ctor_get(v_inst_216_, 2);
lean_inc(v_orElse_220_);
lean_dec_ref(v_inst_216_);
v_toPure_221_ = lean_ctor_get(v_toApplicative_218_, 1);
lean_inc(v_toPure_221_);
lean_dec_ref(v_toApplicative_218_);
v_mapConst_222_ = lean_ctor_get(v_toFunctor_219_, 1);
lean_inc(v_mapConst_222_);
lean_dec_ref(v_toFunctor_219_);
v___x_223_ = lean_box(0);
v___f_224_ = lean_alloc_closure((void*)(lp_mathlib_tryM___redArg___lam__0), 3, 2);
lean_closure_set(v___f_224_, 0, v_toPure_221_);
lean_closure_set(v___f_224_, 1, v___x_223_);
v___x_225_ = lean_apply_4(v_mapConst_222_, lean_box(0), lean_box(0), v___x_223_, v_x_217_);
v___x_226_ = lean_apply_3(v_orElse_220_, lean_box(0), v___x_225_, v___f_224_);
return v___x_226_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_tryM(lean_object* v_F_227_, lean_object* v_inst_228_, lean_object* v_00_u03b1_229_, lean_object* v_x_230_){
_start:
{
lean_object* v___x_231_; 
v___x_231_ = lp_mathlib_tryM___redArg(v_inst_228_, v_x_230_);
return v___x_231_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_try_x3f___redArg___lam__0(lean_object* v_val_232_){
_start:
{
lean_object* v___x_233_; 
v___x_233_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_233_, 0, v_val_232_);
return v___x_233_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_try_x3f___redArg___lam__1(lean_object* v_toPure_234_, lean_object* v_x_235_){
_start:
{
lean_object* v___x_236_; lean_object* v___x_237_; 
v___x_236_ = lean_box(0);
v___x_237_ = lean_apply_2(v_toPure_234_, lean_box(0), v___x_236_);
return v___x_237_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_try_x3f___redArg(lean_object* v_inst_239_, lean_object* v_x_240_){
_start:
{
lean_object* v_toApplicative_241_; lean_object* v_toFunctor_242_; lean_object* v_orElse_243_; lean_object* v_toPure_244_; lean_object* v_map_245_; lean_object* v___f_246_; lean_object* v___f_247_; lean_object* v___x_248_; lean_object* v___x_249_; 
v_toApplicative_241_ = lean_ctor_get(v_inst_239_, 0);
lean_inc_ref(v_toApplicative_241_);
v_toFunctor_242_ = lean_ctor_get(v_toApplicative_241_, 0);
lean_inc_ref(v_toFunctor_242_);
v_orElse_243_ = lean_ctor_get(v_inst_239_, 2);
lean_inc(v_orElse_243_);
lean_dec_ref(v_inst_239_);
v_toPure_244_ = lean_ctor_get(v_toApplicative_241_, 1);
lean_inc(v_toPure_244_);
lean_dec_ref(v_toApplicative_241_);
v_map_245_ = lean_ctor_get(v_toFunctor_242_, 0);
lean_inc(v_map_245_);
lean_dec_ref(v_toFunctor_242_);
v___f_246_ = ((lean_object*)(lp_mathlib_try_x3f___redArg___closed__0));
v___f_247_ = lean_alloc_closure((void*)(lp_mathlib_try_x3f___redArg___lam__1), 2, 1);
lean_closure_set(v___f_247_, 0, v_toPure_244_);
v___x_248_ = lean_apply_4(v_map_245_, lean_box(0), lean_box(0), v___f_246_, v_x_240_);
v___x_249_ = lean_apply_3(v_orElse_243_, lean_box(0), v___x_248_, v___f_247_);
return v___x_249_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_try_x3f(lean_object* v_F_250_, lean_object* v_inst_251_, lean_object* v_00_u03b1_252_, lean_object* v_x_253_){
_start:
{
lean_object* v___x_254_; 
v___x_254_ = lp_mathlib_try_x3f___redArg(v_inst_251_, v_x_253_);
return v___x_254_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Sum_bind___redArg(lean_object* v_x_255_, lean_object* v_x_256_){
_start:
{
if (lean_obj_tag(v_x_255_) == 0)
{
lean_object* v_val_257_; lean_object* v___x_259_; uint8_t v_isShared_260_; uint8_t v_isSharedCheck_264_; 
lean_dec_ref(v_x_256_);
v_val_257_ = lean_ctor_get(v_x_255_, 0);
v_isSharedCheck_264_ = !lean_is_exclusive(v_x_255_);
if (v_isSharedCheck_264_ == 0)
{
v___x_259_ = v_x_255_;
v_isShared_260_ = v_isSharedCheck_264_;
goto v_resetjp_258_;
}
else
{
lean_inc(v_val_257_);
lean_dec(v_x_255_);
v___x_259_ = lean_box(0);
v_isShared_260_ = v_isSharedCheck_264_;
goto v_resetjp_258_;
}
v_resetjp_258_:
{
lean_object* v___x_262_; 
if (v_isShared_260_ == 0)
{
v___x_262_ = v___x_259_;
goto v_reusejp_261_;
}
else
{
lean_object* v_reuseFailAlloc_263_; 
v_reuseFailAlloc_263_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_263_, 0, v_val_257_);
v___x_262_ = v_reuseFailAlloc_263_;
goto v_reusejp_261_;
}
v_reusejp_261_:
{
return v___x_262_;
}
}
}
else
{
lean_object* v_val_265_; lean_object* v___x_266_; 
v_val_265_ = lean_ctor_get(v_x_255_, 0);
lean_inc(v_val_265_);
lean_dec_ref_known(v_x_255_, 1);
v___x_266_ = lean_apply_1(v_x_256_, v_val_265_);
return v___x_266_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Sum_bind(lean_object* v_e_267_, lean_object* v_00_u03b1_268_, lean_object* v_00_u03b2_269_, lean_object* v_x_270_, lean_object* v_x_271_){
_start:
{
lean_object* v___x_272_; 
v___x_272_ = lp_mathlib_Sum_bind___redArg(v_x_270_, v_x_271_);
return v___x_272_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Sum_instMonad__mathlib___lam__0(lean_object* v_val_273_){
_start:
{
lean_object* v___x_274_; 
v___x_274_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_274_, 0, v_val_273_);
return v___x_274_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Sum_instMonad__mathlib___lam__1(lean_object* v___f_275_, lean_object* v_00_u03b1_276_, lean_object* v_00_u03b2_277_, lean_object* v_f_278_, lean_object* v_x_279_){
_start:
{
lean_object* v___x_280_; lean_object* v___x_281_; 
v___x_280_ = lean_alloc_closure((void*)(l_Function_comp), 6, 5);
lean_closure_set(v___x_280_, 0, lean_box(0));
lean_closure_set(v___x_280_, 1, lean_box(0));
lean_closure_set(v___x_280_, 2, lean_box(0));
lean_closure_set(v___x_280_, 3, v___f_275_);
lean_closure_set(v___x_280_, 4, v_f_278_);
v___x_281_ = lp_mathlib_Sum_bind___redArg(v_x_279_, v___x_280_);
return v___x_281_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Sum_instMonad__mathlib___lam__3(lean_object* v___f_282_, lean_object* v_00_u03b1_283_, lean_object* v_00_u03b2_284_, lean_object* v___y_285_, lean_object* v___y_286_){
_start:
{
lean_object* v___x_287_; lean_object* v___x_288_; lean_object* v___x_289_; 
v___x_287_ = lean_alloc_closure((void*)(l_Function_const___boxed), 4, 3);
lean_closure_set(v___x_287_, 0, lean_box(0));
lean_closure_set(v___x_287_, 1, lean_box(0));
lean_closure_set(v___x_287_, 2, v___y_285_);
v___x_288_ = lean_alloc_closure((void*)(l_Function_comp), 6, 5);
lean_closure_set(v___x_288_, 0, lean_box(0));
lean_closure_set(v___x_288_, 1, lean_box(0));
lean_closure_set(v___x_288_, 2, lean_box(0));
lean_closure_set(v___x_288_, 3, v___f_282_);
lean_closure_set(v___x_288_, 4, v___x_287_);
v___x_289_ = lp_mathlib_Sum_bind___redArg(v___y_286_, v___x_288_);
return v___x_289_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Sum_instMonad__mathlib___lam__2(lean_object* v_00_u03b2_290_, lean_object* v_val_291_){
_start:
{
lean_object* v___x_292_; 
v___x_292_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_292_, 0, v_val_291_);
return v___x_292_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Sum_instMonad__mathlib___lam__5(lean_object* v_x_293_, lean_object* v___f_294_, lean_object* v_y_295_){
_start:
{
lean_object* v___x_296_; lean_object* v___x_297_; lean_object* v___x_298_; lean_object* v___x_299_; 
v___x_296_ = lean_box(0);
v___x_297_ = lean_apply_1(v_x_293_, v___x_296_);
v___x_298_ = lean_alloc_closure((void*)(l_Function_comp), 6, 5);
lean_closure_set(v___x_298_, 0, lean_box(0));
lean_closure_set(v___x_298_, 1, lean_box(0));
lean_closure_set(v___x_298_, 2, lean_box(0));
lean_closure_set(v___x_298_, 3, v___f_294_);
lean_closure_set(v___x_298_, 4, v_y_295_);
v___x_299_ = lp_mathlib_Sum_bind___redArg(v___x_297_, v___x_298_);
return v___x_299_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Sum_instMonad__mathlib___lam__4(lean_object* v___f_300_, lean_object* v_00_u03b1_301_, lean_object* v_00_u03b2_302_, lean_object* v_f_303_, lean_object* v_x_304_){
_start:
{
lean_object* v___f_305_; lean_object* v___x_306_; 
v___f_305_ = lean_alloc_closure((void*)(lp_mathlib_Sum_instMonad__mathlib___lam__5), 3, 2);
lean_closure_set(v___f_305_, 0, v_x_304_);
lean_closure_set(v___f_305_, 1, v___f_300_);
v___x_306_ = lp_mathlib_Sum_bind___redArg(v_f_303_, v___f_305_);
return v___x_306_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Sum_instMonad__mathlib___lam__6(lean_object* v_a_307_, lean_object* v_x_308_){
_start:
{
lean_object* v___x_309_; 
v___x_309_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_309_, 0, v_a_307_);
return v___x_309_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Sum_instMonad__mathlib___lam__6___boxed(lean_object* v_a_310_, lean_object* v_x_311_){
_start:
{
lean_object* v_res_312_; 
v_res_312_ = lp_mathlib_Sum_instMonad__mathlib___lam__6(v_a_310_, v_x_311_);
lean_dec(v_x_311_);
return v_res_312_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Sum_instMonad__mathlib___lam__7(lean_object* v_y_313_, lean_object* v_a_314_){
_start:
{
lean_object* v___f_315_; lean_object* v___x_316_; lean_object* v___x_317_; lean_object* v___x_318_; 
v___f_315_ = lean_alloc_closure((void*)(lp_mathlib_Sum_instMonad__mathlib___lam__6___boxed), 2, 1);
lean_closure_set(v___f_315_, 0, v_a_314_);
v___x_316_ = lean_box(0);
v___x_317_ = lean_apply_1(v_y_313_, v___x_316_);
v___x_318_ = lp_mathlib_Sum_bind___redArg(v___x_317_, v___f_315_);
return v___x_318_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Sum_instMonad__mathlib___lam__8(lean_object* v_00_u03b1_319_, lean_object* v_00_u03b2_320_, lean_object* v_x_321_, lean_object* v_y_322_){
_start:
{
lean_object* v___f_323_; lean_object* v___x_324_; 
v___f_323_ = lean_alloc_closure((void*)(lp_mathlib_Sum_instMonad__mathlib___lam__7), 2, 1);
lean_closure_set(v___f_323_, 0, v_y_322_);
v___x_324_ = lp_mathlib_Sum_bind___redArg(v_x_321_, v___f_323_);
return v___x_324_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Sum_instMonad__mathlib___lam__9(lean_object* v_y_325_, lean_object* v_x_326_){
_start:
{
lean_object* v___x_327_; lean_object* v___x_328_; 
v___x_327_ = lean_box(0);
v___x_328_ = lean_apply_1(v_y_325_, v___x_327_);
return v___x_328_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Sum_instMonad__mathlib___lam__9___boxed(lean_object* v_y_329_, lean_object* v_x_330_){
_start:
{
lean_object* v_res_331_; 
v_res_331_ = lp_mathlib_Sum_instMonad__mathlib___lam__9(v_y_329_, v_x_330_);
lean_dec(v_x_330_);
return v_res_331_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Sum_instMonad__mathlib___lam__10(lean_object* v_00_u03b1_332_, lean_object* v_00_u03b2_333_, lean_object* v_x_334_, lean_object* v_y_335_){
_start:
{
lean_object* v___f_336_; lean_object* v___x_337_; 
v___f_336_ = lean_alloc_closure((void*)(lp_mathlib_Sum_instMonad__mathlib___lam__9___boxed), 2, 1);
lean_closure_set(v___f_336_, 0, v_y_335_);
v___x_337_ = lp_mathlib_Sum_bind___redArg(v_x_334_, v___f_336_);
return v___x_337_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Sum_instMonad__mathlib(lean_object* v_e_361_){
_start:
{
lean_object* v___x_362_; 
v___x_362_ = ((lean_object*)(lp_mathlib_Sum_instMonad__mathlib___closed__10));
return v___x_362_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Control_Combinators(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Tactic_CasesM(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Tactic_Attr_Core(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Tactic_Attr_Register(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_Control_Basic(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Control_Combinators(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Tactic_CasesM(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Tactic_Attr_Core(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Tactic_Attr_Register(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_Control_Basic(uint8_t builtin) {
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
lean_object* initialize_mathlib_Mathlib_Control_Combinators(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Tactic_CasesM(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Tactic_Attr_Core(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Tactic_Attr_Register(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Control_Basic(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Control_Combinators(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Tactic_CasesM(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Tactic_Attr_Core(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Tactic_Attr_Register(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Control_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_Control_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_Control_Basic(builtin);
}
#ifdef __cplusplus
}
#endif
