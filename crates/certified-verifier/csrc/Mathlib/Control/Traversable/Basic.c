// Lean compiler output
// Module: Mathlib.Control.Traversable.Basic
// Imports: public import Init public meta import Init public import Mathlib.Data.Option.Defs public import Mathlib.Control.Functor public import Batteries.Data.List.Basic public import Mathlib.Control.Basic import Mathlib.Tactic.Attr.Register
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
lean_object* lp_mathlib_Option_traverse___redArg(lean_object*, lean_object*, lean_object*);
lean_object* l_instFunctorOption___lam__0(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Option_map(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_batteries_List_traverse___redArg(lean_object*, lean_object*, lean_object*);
extern lean_object* l_List_instFunctor;
lean_object* lp_mathlib_Sum_instMonad__mathlib(lean_object*);
lean_object* l_Id_instMonad___lam__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Id_instMonad___lam__0(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_id___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ApplicativeTransformation_instCoeFunForallForall___lam__0(lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_ApplicativeTransformation_instCoeFunForallForall___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_ApplicativeTransformation_instCoeFunForallForall___lam__0, .m_arity = 3, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_ApplicativeTransformation_instCoeFunForallForall___closed__0 = (const lean_object*)&lp_mathlib_ApplicativeTransformation_instCoeFunForallForall___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_ApplicativeTransformation_instCoeFunForallForall(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ApplicativeTransformation_instCoeFunForallForall___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ApplicativeTransformation_idTransformation___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ApplicativeTransformation_idTransformation___lam__0___boxed(lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_ApplicativeTransformation_idTransformation___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_ApplicativeTransformation_idTransformation___lam__0___boxed, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_ApplicativeTransformation_idTransformation___closed__0 = (const lean_object*)&lp_mathlib_ApplicativeTransformation_idTransformation___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_ApplicativeTransformation_idTransformation(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ApplicativeTransformation_idTransformation___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ApplicativeTransformation_instInhabited(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ApplicativeTransformation_instInhabited___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ApplicativeTransformation_comp___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ApplicativeTransformation_comp___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ApplicativeTransformation_comp(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ApplicativeTransformation_comp___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_sequence___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*1, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_id___boxed, .m_arity = 2, .m_num_fixed = 1, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1))} };
static const lean_object* lp_mathlib_sequence___redArg___closed__0 = (const lean_object*)&lp_mathlib_sequence___redArg___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_sequence___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_sequence(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instTraversableId___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instTraversableId___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_instTraversableId___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_instTraversableId___lam__0___boxed, .m_arity = 6, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_instTraversableId___closed__0 = (const lean_object*)&lp_mathlib_instTraversableId___closed__0_value;
static const lean_closure_object lp_mathlib_instTraversableId___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Id_instMonad___lam__0, .m_arity = 4, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_instTraversableId___closed__1 = (const lean_object*)&lp_mathlib_instTraversableId___closed__1_value;
static const lean_closure_object lp_mathlib_instTraversableId___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Id_instMonad___lam__1___boxed, .m_arity = 4, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_instTraversableId___closed__2 = (const lean_object*)&lp_mathlib_instTraversableId___closed__2_value;
static const lean_ctor_object lp_mathlib_instTraversableId___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_instTraversableId___closed__1_value),((lean_object*)&lp_mathlib_instTraversableId___closed__2_value)}};
static const lean_object* lp_mathlib_instTraversableId___closed__3 = (const lean_object*)&lp_mathlib_instTraversableId___closed__3_value;
static const lean_ctor_object lp_mathlib_instTraversableId___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_instTraversableId___closed__3_value),((lean_object*)&lp_mathlib_instTraversableId___closed__0_value)}};
static const lean_object* lp_mathlib_instTraversableId___closed__4 = (const lean_object*)&lp_mathlib_instTraversableId___closed__4_value;
LEAN_EXPORT const lean_object* lp_mathlib_instTraversableId = (const lean_object*)&lp_mathlib_instTraversableId___closed__4_value;
LEAN_EXPORT lean_object* lp_mathlib_instTraversableOption___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_instTraversableOption___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_instTraversableOption___lam__0, .m_arity = 6, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_instTraversableOption___closed__0 = (const lean_object*)&lp_mathlib_instTraversableOption___closed__0_value;
static const lean_closure_object lp_mathlib_instTraversableOption___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_instFunctorOption___lam__0, .m_arity = 4, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_instTraversableOption___closed__1 = (const lean_object*)&lp_mathlib_instTraversableOption___closed__1_value;
static const lean_closure_object lp_mathlib_instTraversableOption___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Option_map, .m_arity = 4, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_instTraversableOption___closed__2 = (const lean_object*)&lp_mathlib_instTraversableOption___closed__2_value;
static const lean_ctor_object lp_mathlib_instTraversableOption___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_instTraversableOption___closed__2_value),((lean_object*)&lp_mathlib_instTraversableOption___closed__1_value)}};
static const lean_object* lp_mathlib_instTraversableOption___closed__3 = (const lean_object*)&lp_mathlib_instTraversableOption___closed__3_value;
static const lean_ctor_object lp_mathlib_instTraversableOption___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_instTraversableOption___closed__3_value),((lean_object*)&lp_mathlib_instTraversableOption___closed__0_value)}};
static const lean_object* lp_mathlib_instTraversableOption___closed__4 = (const lean_object*)&lp_mathlib_instTraversableOption___closed__4_value;
LEAN_EXPORT const lean_object* lp_mathlib_instTraversableOption = (const lean_object*)&lp_mathlib_instTraversableOption___closed__4_value;
LEAN_EXPORT lean_object* lp_mathlib_instTraversableList___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_instTraversableList___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_instTraversableList___lam__0, .m_arity = 6, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_instTraversableList___closed__0 = (const lean_object*)&lp_mathlib_instTraversableList___closed__0_value;
static lean_once_cell_t lp_mathlib_instTraversableList___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_instTraversableList___closed__1;
LEAN_EXPORT lean_object* lp_mathlib_instTraversableList;
LEAN_EXPORT lean_object* lp_mathlib_Sum_traverse___redArg___lam__0(lean_object*);
static const lean_closure_object lp_mathlib_Sum_traverse___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Sum_traverse___redArg___lam__0, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Sum_traverse___redArg___closed__0 = (const lean_object*)&lp_mathlib_Sum_traverse___redArg___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_Sum_traverse___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Sum_traverse(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_once_cell_t lp_mathlib_instTraversableSum___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_instTraversableSum___closed__0;
static const lean_closure_object lp_mathlib_instTraversableSum___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*1, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Sum_traverse, .m_arity = 7, .m_num_fixed = 1, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1))} };
static const lean_object* lp_mathlib_instTraversableSum___closed__1 = (const lean_object*)&lp_mathlib_instTraversableSum___closed__1_value;
LEAN_EXPORT lean_object* lp_mathlib_instTraversableSum(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ApplicativeTransformation_instCoeFunForallForall___lam__0(lean_object* v_00_u03b7_1_, lean_object* v_00_u03b1_2_, lean_object* v___y_3_){
_start:
{
lean_object* v___x_4_; 
v___x_4_ = lean_apply_2(v_00_u03b7_1_, lean_box(0), v___y_3_);
return v___x_4_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ApplicativeTransformation_instCoeFunForallForall(lean_object* v_F_6_, lean_object* v_inst_7_, lean_object* v_G_8_, lean_object* v_inst_9_){
_start:
{
lean_object* v___f_10_; 
v___f_10_ = ((lean_object*)(lp_mathlib_ApplicativeTransformation_instCoeFunForallForall___closed__0));
return v___f_10_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ApplicativeTransformation_instCoeFunForallForall___boxed(lean_object* v_F_11_, lean_object* v_inst_12_, lean_object* v_G_13_, lean_object* v_inst_14_){
_start:
{
lean_object* v_res_15_; 
v_res_15_ = lp_mathlib_ApplicativeTransformation_instCoeFunForallForall(v_F_11_, v_inst_12_, v_G_13_, v_inst_14_);
lean_dec_ref(v_inst_14_);
lean_dec_ref(v_inst_12_);
return v_res_15_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ApplicativeTransformation_idTransformation___lam__0(lean_object* v_x_16_, lean_object* v___y_17_){
_start:
{
lean_inc(v___y_17_);
return v___y_17_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ApplicativeTransformation_idTransformation___lam__0___boxed(lean_object* v_x_18_, lean_object* v___y_19_){
_start:
{
lean_object* v_res_20_; 
v_res_20_ = lp_mathlib_ApplicativeTransformation_idTransformation___lam__0(v_x_18_, v___y_19_);
lean_dec(v___y_19_);
return v_res_20_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ApplicativeTransformation_idTransformation(lean_object* v_F_22_, lean_object* v_inst_23_){
_start:
{
lean_object* v___f_24_; 
v___f_24_ = ((lean_object*)(lp_mathlib_ApplicativeTransformation_idTransformation___closed__0));
return v___f_24_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ApplicativeTransformation_idTransformation___boxed(lean_object* v_F_25_, lean_object* v_inst_26_){
_start:
{
lean_object* v_res_27_; 
v_res_27_ = lp_mathlib_ApplicativeTransformation_idTransformation(v_F_25_, v_inst_26_);
lean_dec_ref(v_inst_26_);
return v_res_27_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ApplicativeTransformation_instInhabited(lean_object* v_F_28_, lean_object* v_inst_29_){
_start:
{
lean_object* v___f_30_; 
v___f_30_ = ((lean_object*)(lp_mathlib_ApplicativeTransformation_idTransformation___closed__0));
return v___f_30_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ApplicativeTransformation_instInhabited___boxed(lean_object* v_F_31_, lean_object* v_inst_32_){
_start:
{
lean_object* v_res_33_; 
v_res_33_ = lp_mathlib_ApplicativeTransformation_instInhabited(v_F_31_, v_inst_32_);
lean_dec_ref(v_inst_32_);
return v_res_33_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ApplicativeTransformation_comp___redArg___lam__0(lean_object* v_00_u03b7_34_, lean_object* v_00_u03b7_x27_35_, lean_object* v_x_36_, lean_object* v_x_37_){
_start:
{
lean_object* v___x_38_; lean_object* v___x_39_; 
v___x_38_ = lean_apply_2(v_00_u03b7_34_, lean_box(0), v_x_37_);
v___x_39_ = lean_apply_2(v_00_u03b7_x27_35_, lean_box(0), v___x_38_);
return v___x_39_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ApplicativeTransformation_comp___redArg(lean_object* v_00_u03b7_x27_40_, lean_object* v_00_u03b7_41_){
_start:
{
lean_object* v___f_42_; 
v___f_42_ = lean_alloc_closure((void*)(lp_mathlib_ApplicativeTransformation_comp___redArg___lam__0), 4, 2);
lean_closure_set(v___f_42_, 0, v_00_u03b7_41_);
lean_closure_set(v___f_42_, 1, v_00_u03b7_x27_40_);
return v___f_42_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ApplicativeTransformation_comp(lean_object* v_F_43_, lean_object* v_inst_44_, lean_object* v_G_45_, lean_object* v_inst_46_, lean_object* v_H_47_, lean_object* v_inst_48_, lean_object* v_00_u03b7_x27_49_, lean_object* v_00_u03b7_50_){
_start:
{
lean_object* v___f_51_; 
v___f_51_ = lean_alloc_closure((void*)(lp_mathlib_ApplicativeTransformation_comp___redArg___lam__0), 4, 2);
lean_closure_set(v___f_51_, 0, v_00_u03b7_50_);
lean_closure_set(v___f_51_, 1, v_00_u03b7_x27_49_);
return v___f_51_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ApplicativeTransformation_comp___boxed(lean_object* v_F_52_, lean_object* v_inst_53_, lean_object* v_G_54_, lean_object* v_inst_55_, lean_object* v_H_56_, lean_object* v_inst_57_, lean_object* v_00_u03b7_x27_58_, lean_object* v_00_u03b7_59_){
_start:
{
lean_object* v_res_60_; 
v_res_60_ = lp_mathlib_ApplicativeTransformation_comp(v_F_52_, v_inst_53_, v_G_54_, v_inst_55_, v_H_56_, v_inst_57_, v_00_u03b7_x27_58_, v_00_u03b7_59_);
lean_dec_ref(v_inst_57_);
lean_dec_ref(v_inst_55_);
lean_dec_ref(v_inst_53_);
return v_res_60_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_sequence___redArg(lean_object* v_inst_62_, lean_object* v_inst_63_, lean_object* v_a_64_){
_start:
{
lean_object* v_traverse_65_; lean_object* v___x_66_; lean_object* v___x_67_; 
v_traverse_65_ = lean_ctor_get(v_inst_63_, 1);
lean_inc(v_traverse_65_);
lean_dec_ref(v_inst_63_);
v___x_66_ = ((lean_object*)(lp_mathlib_sequence___redArg___closed__0));
v___x_67_ = lean_apply_6(v_traverse_65_, lean_box(0), v_inst_62_, lean_box(0), lean_box(0), v___x_66_, v_a_64_);
return v___x_67_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_sequence(lean_object* v_t_68_, lean_object* v_00_u03b1_69_, lean_object* v_f_70_, lean_object* v_inst_71_, lean_object* v_inst_72_, lean_object* v_a_73_){
_start:
{
lean_object* v___x_74_; 
v___x_74_ = lp_mathlib_sequence___redArg(v_inst_71_, v_inst_72_, v_a_73_);
return v___x_74_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instTraversableId___lam__0(lean_object* v_m_75_, lean_object* v_inst_76_, lean_object* v_00_u03b1_77_, lean_object* v_00_u03b2_78_, lean_object* v___y_79_, lean_object* v___y_80_){
_start:
{
lean_object* v___x_81_; 
v___x_81_ = lean_apply_1(v___y_79_, v___y_80_);
return v___x_81_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instTraversableId___lam__0___boxed(lean_object* v_m_82_, lean_object* v_inst_83_, lean_object* v_00_u03b1_84_, lean_object* v_00_u03b2_85_, lean_object* v___y_86_, lean_object* v___y_87_){
_start:
{
lean_object* v_res_88_; 
v_res_88_ = lp_mathlib_instTraversableId___lam__0(v_m_82_, v_inst_83_, v_00_u03b1_84_, v_00_u03b2_85_, v___y_86_, v___y_87_);
lean_dec_ref(v_inst_83_);
return v_res_88_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instTraversableOption___lam__0(lean_object* v_m_99_, lean_object* v_inst_100_, lean_object* v_00_u03b1_101_, lean_object* v_00_u03b2_102_, lean_object* v___y_103_, lean_object* v___y_104_){
_start:
{
lean_object* v___x_105_; 
v___x_105_ = lp_mathlib_Option_traverse___redArg(v_inst_100_, v___y_103_, v___y_104_);
return v___x_105_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instTraversableList___lam__0(lean_object* v_m_116_, lean_object* v_inst_117_, lean_object* v_00_u03b1_118_, lean_object* v_00_u03b2_119_, lean_object* v___y_120_, lean_object* v___y_121_){
_start:
{
lean_object* v___x_122_; 
v___x_122_ = lp_batteries_List_traverse___redArg(v_inst_117_, v___y_120_, v___y_121_);
return v___x_122_;
}
}
static lean_object* _init_lp_mathlib_instTraversableList___closed__1(void){
_start:
{
lean_object* v___f_124_; lean_object* v___x_125_; lean_object* v___x_126_; 
v___f_124_ = ((lean_object*)(lp_mathlib_instTraversableList___closed__0));
v___x_125_ = l_List_instFunctor;
v___x_126_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_126_, 0, v___x_125_);
lean_ctor_set(v___x_126_, 1, v___f_124_);
return v___x_126_;
}
}
static lean_object* _init_lp_mathlib_instTraversableList(void){
_start:
{
lean_object* v___x_127_; 
v___x_127_ = lean_obj_once(&lp_mathlib_instTraversableList___closed__1, &lp_mathlib_instTraversableList___closed__1_once, _init_lp_mathlib_instTraversableList___closed__1);
return v___x_127_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Sum_traverse___redArg___lam__0(lean_object* v_val_128_){
_start:
{
lean_object* v___x_129_; 
v___x_129_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_129_, 0, v_val_128_);
return v___x_129_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Sum_traverse___redArg(lean_object* v_inst_131_, lean_object* v_f_132_, lean_object* v_x_133_){
_start:
{
if (lean_obj_tag(v_x_133_) == 0)
{
lean_object* v_toPure_134_; lean_object* v_val_135_; lean_object* v___x_137_; uint8_t v_isShared_138_; uint8_t v_isSharedCheck_143_; 
lean_dec(v_f_132_);
v_toPure_134_ = lean_ctor_get(v_inst_131_, 1);
lean_inc(v_toPure_134_);
lean_dec_ref(v_inst_131_);
v_val_135_ = lean_ctor_get(v_x_133_, 0);
v_isSharedCheck_143_ = !lean_is_exclusive(v_x_133_);
if (v_isSharedCheck_143_ == 0)
{
v___x_137_ = v_x_133_;
v_isShared_138_ = v_isSharedCheck_143_;
goto v_resetjp_136_;
}
else
{
lean_inc(v_val_135_);
lean_dec(v_x_133_);
v___x_137_ = lean_box(0);
v_isShared_138_ = v_isSharedCheck_143_;
goto v_resetjp_136_;
}
v_resetjp_136_:
{
lean_object* v___x_140_; 
if (v_isShared_138_ == 0)
{
v___x_140_ = v___x_137_;
goto v_reusejp_139_;
}
else
{
lean_object* v_reuseFailAlloc_142_; 
v_reuseFailAlloc_142_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_142_, 0, v_val_135_);
v___x_140_ = v_reuseFailAlloc_142_;
goto v_reusejp_139_;
}
v_reusejp_139_:
{
lean_object* v___x_141_; 
v___x_141_ = lean_apply_2(v_toPure_134_, lean_box(0), v___x_140_);
return v___x_141_;
}
}
}
else
{
lean_object* v_toFunctor_144_; lean_object* v_val_145_; lean_object* v_map_146_; lean_object* v___f_147_; lean_object* v___x_148_; lean_object* v___x_149_; 
v_toFunctor_144_ = lean_ctor_get(v_inst_131_, 0);
lean_inc_ref(v_toFunctor_144_);
lean_dec_ref(v_inst_131_);
v_val_145_ = lean_ctor_get(v_x_133_, 0);
lean_inc(v_val_145_);
lean_dec_ref_known(v_x_133_, 1);
v_map_146_ = lean_ctor_get(v_toFunctor_144_, 0);
lean_inc(v_map_146_);
lean_dec_ref(v_toFunctor_144_);
v___f_147_ = ((lean_object*)(lp_mathlib_Sum_traverse___redArg___closed__0));
v___x_148_ = lean_apply_1(v_f_132_, v_val_145_);
v___x_149_ = lean_apply_4(v_map_146_, lean_box(0), lean_box(0), v___f_147_, v___x_148_);
return v___x_149_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Sum_traverse(lean_object* v_00_u03c3_150_, lean_object* v_F_151_, lean_object* v_inst_152_, lean_object* v_00_u03b1_153_, lean_object* v_00_u03b2_154_, lean_object* v_f_155_, lean_object* v_x_156_){
_start:
{
lean_object* v___x_157_; 
v___x_157_ = lp_mathlib_Sum_traverse___redArg(v_inst_152_, v_f_155_, v_x_156_);
return v___x_157_;
}
}
static lean_object* _init_lp_mathlib_instTraversableSum___closed__0(void){
_start:
{
lean_object* v___x_158_; 
v___x_158_ = lp_mathlib_Sum_instMonad__mathlib(lean_box(0));
return v___x_158_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instTraversableSum(lean_object* v_00_u03c3_160_){
_start:
{
lean_object* v___x_161_; lean_object* v_toApplicative_162_; lean_object* v_toFunctor_163_; lean_object* v___x_164_; lean_object* v___x_165_; 
v___x_161_ = lean_obj_once(&lp_mathlib_instTraversableSum___closed__0, &lp_mathlib_instTraversableSum___closed__0_once, _init_lp_mathlib_instTraversableSum___closed__0);
v_toApplicative_162_ = lean_ctor_get(v___x_161_, 0);
v_toFunctor_163_ = lean_ctor_get(v_toApplicative_162_, 0);
v___x_164_ = ((lean_object*)(lp_mathlib_instTraversableSum___closed__1));
lean_inc_ref(v_toFunctor_163_);
v___x_165_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_165_, 0, v_toFunctor_163_);
lean_ctor_set(v___x_165_, 1, v___x_164_);
return v___x_165_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Data_Option_Defs(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Control_Functor(uint8_t builtin);
lean_object* runtime_initialize_batteries_Batteries_Data_List_Basic(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Control_Basic(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Tactic_Attr_Register(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_Control_Traversable_Basic(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Data_Option_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Control_Functor(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_batteries_Batteries_Data_List_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Control_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Tactic_Attr_Register(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
lp_mathlib_instTraversableList = _init_lp_mathlib_instTraversableList();
lean_mark_persistent(lp_mathlib_instTraversableList);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_Control_Traversable_Basic(uint8_t builtin) {
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
lean_object* initialize_mathlib_Mathlib_Data_Option_Defs(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Control_Functor(uint8_t builtin);
lean_object* initialize_batteries_Batteries_Data_List_Basic(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Control_Basic(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Tactic_Attr_Register(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Control_Traversable_Basic(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_Option_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Control_Functor(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_batteries_Batteries_Data_List_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Control_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Tactic_Attr_Register(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Control_Traversable_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_Control_Traversable_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_Control_Traversable_Basic(builtin);
}
#ifdef __cplusplus
}
#endif
