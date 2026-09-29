// Lean compiler output
// Module: Mathlib.LinearAlgebra.LinearIndependent.Defs
// Imports: public import Init public meta import Init public import Mathlib.Algebra.Order.Sub.Basic public import Mathlib.LinearAlgebra.Finsupp.LinearCombination public meta import Mathlib.Lean.Expr.ExtraRecognizers
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
lean_object* l_Lean_Name_mkStr3(lean_object*, lean_object*, lean_object*);
extern lean_object* l_Lean_pp_funBinderTypes;
lean_object* lp_mathlib_Lean_PrettyPrinter_Delaborator_OptionsPerPos_setBool(lean_object*, lean_object*, lean_object*, uint8_t);
lean_object* l_Lean_PrettyPrinter_Delaborator_delab(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Expr_getAppNumArgs(lean_object*);
lean_object* lean_nat_sub(lean_object*, lean_object*);
lean_object* l_Lean_Expr_getRevArg_x21(lean_object*, lean_object*);
lean_object* lp_mathlib_Lean_Expr_coeTypeSet_x3f(lean_object*);
uint8_t l_Lean_Expr_isLambda(lean_object*);
extern lean_object* l_Lean_instInhabitedExpr;
lean_object* l_Lean_Expr_sort___override(lean_object*);
lean_object* lean_mk_array(lean_object*, lean_object*);
lean_object* l___private_Lean_Expr_0__Lean_Expr_getAppArgsAux(lean_object*, lean_object*, lean_object*);
lean_object* lean_array_get_size(lean_object*);
lean_object* l_Lean_SubExpr_Pos_pushNaryArg(lean_object*, lean_object*, lean_object*);
lean_object* lean_array_get(lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_PrettyPrinter_Delaborator_failure___redArg();
lean_object* l_Lean_Name_mkStr1(lean_object*);
uint8_t l_Lean_Expr_isAppOfArity(lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_PrettyPrinter_Delaborator_withOptionAtCurrPos___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_getPPNotation___boxed(lean_object*);
lean_object* l_Lean_getPPAnalysisSkip___boxed(lean_object*);
lean_object* l_Lean_PrettyPrinter_Delaborator_whenNotPPOption___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_PrettyPrinter_Delaborator_whenPPOption(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_getExpr___at___00delabLinearIndependent_spec__0___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_getExpr___at___00delabLinearIndependent_spec__0___redArg___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_getExpr___at___00delabLinearIndependent_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_getExpr___at___00delabLinearIndependent_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_getPos___at___00delabLinearIndependent_spec__1___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_getPos___at___00delabLinearIndependent_spec__1___redArg___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_getPos___at___00delabLinearIndependent_spec__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_getPos___at___00delabLinearIndependent_spec__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_delabLinearIndependent___lam__0___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "namedArg"};
static const lean_object* lp_mathlib_delabLinearIndependent___lam__0___closed__0 = (const lean_object*)&lp_mathlib_delabLinearIndependent___lam__0___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_delabLinearIndependent___lam__0(lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_delabLinearIndependent___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_delabLinearIndependent___lam__1(uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_delabLinearIndependent___lam__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_once_cell_t lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_withNaryArg___at___00delabLinearIndependent_spec__2___redArg___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_withNaryArg___at___00delabLinearIndependent_spec__2___redArg___closed__0;
LEAN_EXPORT lean_object* lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_withNaryArg___at___00delabLinearIndependent_spec__2___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_withNaryArg___at___00delabLinearIndependent_spec__2___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_delabLinearIndependent___lam__2___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 18, .m_capacity = 18, .m_length = 17, .m_data = "LinearIndependent"};
static const lean_object* lp_mathlib_delabLinearIndependent___lam__2___closed__0 = (const lean_object*)&lp_mathlib_delabLinearIndependent___lam__2___closed__0_value;
static const lean_ctor_object lp_mathlib_delabLinearIndependent___lam__2___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_delabLinearIndependent___lam__2___closed__0_value),LEAN_SCALAR_PTR_LITERAL(239, 48, 142, 46, 96, 99, 79, 164)}};
static const lean_object* lp_mathlib_delabLinearIndependent___lam__2___closed__1 = (const lean_object*)&lp_mathlib_delabLinearIndependent___lam__2___closed__1_value;
LEAN_EXPORT lean_object* lp_mathlib_delabLinearIndependent___lam__2(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_delabLinearIndependent___lam__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_delabLinearIndependent___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Lean_getPPNotation___boxed, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_delabLinearIndependent___closed__0 = (const lean_object*)&lp_mathlib_delabLinearIndependent___closed__0_value;
static const lean_closure_object lp_mathlib_delabLinearIndependent___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Lean_getPPAnalysisSkip___boxed, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_delabLinearIndependent___closed__1 = (const lean_object*)&lp_mathlib_delabLinearIndependent___closed__1_value;
static const lean_string_object lp_mathlib_delabLinearIndependent___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 3, .m_capacity = 3, .m_length = 2, .m_data = "pp"};
static const lean_object* lp_mathlib_delabLinearIndependent___closed__2 = (const lean_object*)&lp_mathlib_delabLinearIndependent___closed__2_value;
static const lean_string_object lp_mathlib_delabLinearIndependent___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "analysis"};
static const lean_object* lp_mathlib_delabLinearIndependent___closed__3 = (const lean_object*)&lp_mathlib_delabLinearIndependent___closed__3_value;
static const lean_string_object lp_mathlib_delabLinearIndependent___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "skip"};
static const lean_object* lp_mathlib_delabLinearIndependent___closed__4 = (const lean_object*)&lp_mathlib_delabLinearIndependent___closed__4_value;
static const lean_ctor_object lp_mathlib_delabLinearIndependent___closed__5_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_delabLinearIndependent___closed__2_value),LEAN_SCALAR_PTR_LITERAL(249, 51, 192, 169, 230, 180, 160, 93)}};
static const lean_ctor_object lp_mathlib_delabLinearIndependent___closed__5_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_delabLinearIndependent___closed__5_value_aux_0),((lean_object*)&lp_mathlib_delabLinearIndependent___closed__3_value),LEAN_SCALAR_PTR_LITERAL(246, 193, 108, 174, 20, 188, 4, 229)}};
static const lean_ctor_object lp_mathlib_delabLinearIndependent___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_delabLinearIndependent___closed__5_value_aux_1),((lean_object*)&lp_mathlib_delabLinearIndependent___closed__4_value),LEAN_SCALAR_PTR_LITERAL(132, 156, 177, 169, 136, 73, 208, 253)}};
static const lean_object* lp_mathlib_delabLinearIndependent___closed__5 = (const lean_object*)&lp_mathlib_delabLinearIndependent___closed__5_value;
static const lean_closure_object lp_mathlib_delabLinearIndependent___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*3, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_delabLinearIndependent___lam__0___boxed, .m_arity = 10, .m_num_fixed = 3, .m_objs = {((lean_object*)&lp_mathlib_delabLinearIndependent___closed__2_value),((lean_object*)&lp_mathlib_delabLinearIndependent___closed__3_value),((lean_object*)(((size_t)(1) << 1) | 1))} };
static const lean_object* lp_mathlib_delabLinearIndependent___closed__6 = (const lean_object*)&lp_mathlib_delabLinearIndependent___closed__6_value;
static const lean_closure_object lp_mathlib_delabLinearIndependent___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*1, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_delabLinearIndependent___lam__1___boxed, .m_arity = 8, .m_num_fixed = 1, .m_objs = {((lean_object*)(((size_t)(1) << 1) | 1))} };
static const lean_object* lp_mathlib_delabLinearIndependent___closed__7 = (const lean_object*)&lp_mathlib_delabLinearIndependent___closed__7_value;
static const lean_closure_object lp_mathlib_delabLinearIndependent___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*2, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_delabLinearIndependent___lam__2___boxed, .m_arity = 9, .m_num_fixed = 2, .m_objs = {((lean_object*)&lp_mathlib_delabLinearIndependent___closed__6_value),((lean_object*)&lp_mathlib_delabLinearIndependent___closed__7_value)} };
static const lean_object* lp_mathlib_delabLinearIndependent___closed__8 = (const lean_object*)&lp_mathlib_delabLinearIndependent___closed__8_value;
static const lean_ctor_object lp_mathlib_delabLinearIndependent___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*0 + 8, .m_other = 0, .m_tag = 1}, .m_objs = {LEAN_SCALAR_PTR_LITERAL(1, 0, 0, 0, 0, 0, 0, 0)}};
static const lean_object* lp_mathlib_delabLinearIndependent___closed__9 = (const lean_object*)&lp_mathlib_delabLinearIndependent___closed__9_value;
static const lean_closure_object lp_mathlib_delabLinearIndependent___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*4, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Lean_PrettyPrinter_Delaborator_withOptionAtCurrPos___boxed, .m_arity = 11, .m_num_fixed = 4, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_delabLinearIndependent___closed__5_value),((lean_object*)&lp_mathlib_delabLinearIndependent___closed__9_value),((lean_object*)&lp_mathlib_delabLinearIndependent___closed__8_value)} };
static const lean_object* lp_mathlib_delabLinearIndependent___closed__10 = (const lean_object*)&lp_mathlib_delabLinearIndependent___closed__10_value;
static const lean_closure_object lp_mathlib_delabLinearIndependent___closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*2, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Lean_PrettyPrinter_Delaborator_whenNotPPOption___boxed, .m_arity = 9, .m_num_fixed = 2, .m_objs = {((lean_object*)&lp_mathlib_delabLinearIndependent___closed__1_value),((lean_object*)&lp_mathlib_delabLinearIndependent___closed__10_value)} };
static const lean_object* lp_mathlib_delabLinearIndependent___closed__11 = (const lean_object*)&lp_mathlib_delabLinearIndependent___closed__11_value;
LEAN_EXPORT lean_object* lp_mathlib_delabLinearIndependent(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_delabLinearIndependent___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_withNaryArg___at___00delabLinearIndependent_spec__2(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_withNaryArg___at___00delabLinearIndependent_spec__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_getExpr___at___00delabLinearIndependent_spec__0___redArg(lean_object* v___y_1_){
_start:
{
lean_object* v_subExpr_3_; lean_object* v_expr_4_; lean_object* v___x_5_; 
v_subExpr_3_ = lean_ctor_get(v___y_1_, 3);
v_expr_4_ = lean_ctor_get(v_subExpr_3_, 0);
lean_inc_ref(v_expr_4_);
v___x_5_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_5_, 0, v_expr_4_);
return v___x_5_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_getExpr___at___00delabLinearIndependent_spec__0___redArg___boxed(lean_object* v___y_6_, lean_object* v___y_7_){
_start:
{
lean_object* v_res_8_; 
v_res_8_ = lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_getExpr___at___00delabLinearIndependent_spec__0___redArg(v___y_6_);
lean_dec_ref(v___y_6_);
return v_res_8_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_getExpr___at___00delabLinearIndependent_spec__0(lean_object* v___y_9_, lean_object* v___y_10_, lean_object* v___y_11_, lean_object* v___y_12_, lean_object* v___y_13_, lean_object* v___y_14_){
_start:
{
lean_object* v___x_16_; 
v___x_16_ = lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_getExpr___at___00delabLinearIndependent_spec__0___redArg(v___y_9_);
return v___x_16_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_getExpr___at___00delabLinearIndependent_spec__0___boxed(lean_object* v___y_17_, lean_object* v___y_18_, lean_object* v___y_19_, lean_object* v___y_20_, lean_object* v___y_21_, lean_object* v___y_22_, lean_object* v___y_23_){
_start:
{
lean_object* v_res_24_; 
v_res_24_ = lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_getExpr___at___00delabLinearIndependent_spec__0(v___y_17_, v___y_18_, v___y_19_, v___y_20_, v___y_21_, v___y_22_);
lean_dec(v___y_22_);
lean_dec_ref(v___y_21_);
lean_dec(v___y_20_);
lean_dec_ref(v___y_19_);
lean_dec(v___y_18_);
lean_dec_ref(v___y_17_);
return v_res_24_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_getPos___at___00delabLinearIndependent_spec__1___redArg(lean_object* v___y_25_){
_start:
{
lean_object* v_subExpr_27_; lean_object* v_pos_28_; lean_object* v___x_29_; 
v_subExpr_27_ = lean_ctor_get(v___y_25_, 3);
v_pos_28_ = lean_ctor_get(v_subExpr_27_, 1);
lean_inc(v_pos_28_);
v___x_29_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_29_, 0, v_pos_28_);
return v___x_29_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_getPos___at___00delabLinearIndependent_spec__1___redArg___boxed(lean_object* v___y_30_, lean_object* v___y_31_){
_start:
{
lean_object* v_res_32_; 
v_res_32_ = lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_getPos___at___00delabLinearIndependent_spec__1___redArg(v___y_30_);
lean_dec_ref(v___y_30_);
return v_res_32_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_getPos___at___00delabLinearIndependent_spec__1(lean_object* v___y_33_, lean_object* v___y_34_, lean_object* v___y_35_, lean_object* v___y_36_, lean_object* v___y_37_, lean_object* v___y_38_){
_start:
{
lean_object* v___x_40_; 
v___x_40_ = lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_getPos___at___00delabLinearIndependent_spec__1___redArg(v___y_33_);
return v___x_40_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_getPos___at___00delabLinearIndependent_spec__1___boxed(lean_object* v___y_41_, lean_object* v___y_42_, lean_object* v___y_43_, lean_object* v___y_44_, lean_object* v___y_45_, lean_object* v___y_46_, lean_object* v___y_47_){
_start:
{
lean_object* v_res_48_; 
v_res_48_ = lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_getPos___at___00delabLinearIndependent_spec__1(v___y_41_, v___y_42_, v___y_43_, v___y_44_, v___y_45_, v___y_46_);
lean_dec(v___y_46_);
lean_dec_ref(v___y_45_);
lean_dec(v___y_44_);
lean_dec_ref(v___y_43_);
lean_dec(v___y_42_);
lean_dec_ref(v___y_41_);
return v_res_48_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_delabLinearIndependent___lam__0(lean_object* v___x_50_, lean_object* v___x_51_, uint8_t v___x_52_, lean_object* v___y_53_, lean_object* v___y_54_, lean_object* v___y_55_, lean_object* v___y_56_, lean_object* v___y_57_, lean_object* v___y_58_){
_start:
{
lean_object* v___x_60_; lean_object* v_a_61_; lean_object* v___x_63_; uint8_t v_isShared_64_; uint8_t v_isSharedCheck_72_; 
v___x_60_ = lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_getPos___at___00delabLinearIndependent_spec__1___redArg(v___y_53_);
v_a_61_ = lean_ctor_get(v___x_60_, 0);
v_isSharedCheck_72_ = !lean_is_exclusive(v___x_60_);
if (v_isSharedCheck_72_ == 0)
{
v___x_63_ = v___x_60_;
v_isShared_64_ = v_isSharedCheck_72_;
goto v_resetjp_62_;
}
else
{
lean_inc(v_a_61_);
lean_dec(v___x_60_);
v___x_63_ = lean_box(0);
v_isShared_64_ = v_isSharedCheck_72_;
goto v_resetjp_62_;
}
v_resetjp_62_:
{
lean_object* v_optionsPerPos_65_; lean_object* v___x_66_; lean_object* v___x_67_; lean_object* v___x_68_; lean_object* v___x_70_; 
v_optionsPerPos_65_ = lean_ctor_get(v___y_53_, 0);
v___x_66_ = ((lean_object*)(lp_mathlib_delabLinearIndependent___lam__0___closed__0));
v___x_67_ = l_Lean_Name_mkStr3(v___x_50_, v___x_51_, v___x_66_);
lean_inc(v_optionsPerPos_65_);
v___x_68_ = lp_mathlib_Lean_PrettyPrinter_Delaborator_OptionsPerPos_setBool(v_optionsPerPos_65_, v_a_61_, v___x_67_, v___x_52_);
if (v_isShared_64_ == 0)
{
lean_ctor_set(v___x_63_, 0, v___x_68_);
v___x_70_ = v___x_63_;
goto v_reusejp_69_;
}
else
{
lean_object* v_reuseFailAlloc_71_; 
v_reuseFailAlloc_71_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_71_, 0, v___x_68_);
v___x_70_ = v_reuseFailAlloc_71_;
goto v_reusejp_69_;
}
v_reusejp_69_:
{
return v___x_70_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_delabLinearIndependent___lam__0___boxed(lean_object* v___x_73_, lean_object* v___x_74_, lean_object* v___x_75_, lean_object* v___y_76_, lean_object* v___y_77_, lean_object* v___y_78_, lean_object* v___y_79_, lean_object* v___y_80_, lean_object* v___y_81_, lean_object* v___y_82_){
_start:
{
uint8_t v___x_5052__boxed_83_; lean_object* v_res_84_; 
v___x_5052__boxed_83_ = lean_unbox(v___x_75_);
v_res_84_ = lp_mathlib_delabLinearIndependent___lam__0(v___x_73_, v___x_74_, v___x_5052__boxed_83_, v___y_76_, v___y_77_, v___y_78_, v___y_79_, v___y_80_, v___y_81_);
lean_dec(v___y_81_);
lean_dec_ref(v___y_80_);
lean_dec(v___y_79_);
lean_dec_ref(v___y_78_);
lean_dec(v___y_77_);
lean_dec_ref(v___y_76_);
return v_res_84_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_delabLinearIndependent___lam__1(uint8_t v___x_85_, lean_object* v___y_86_, lean_object* v___y_87_, lean_object* v___y_88_, lean_object* v___y_89_, lean_object* v___y_90_, lean_object* v___y_91_){
_start:
{
lean_object* v___x_93_; lean_object* v_a_94_; lean_object* v___x_96_; uint8_t v_isShared_97_; uint8_t v_isSharedCheck_105_; 
v___x_93_ = lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_getPos___at___00delabLinearIndependent_spec__1___redArg(v___y_86_);
v_a_94_ = lean_ctor_get(v___x_93_, 0);
v_isSharedCheck_105_ = !lean_is_exclusive(v___x_93_);
if (v_isSharedCheck_105_ == 0)
{
v___x_96_ = v___x_93_;
v_isShared_97_ = v_isSharedCheck_105_;
goto v_resetjp_95_;
}
else
{
lean_inc(v_a_94_);
lean_dec(v___x_93_);
v___x_96_ = lean_box(0);
v_isShared_97_ = v_isSharedCheck_105_;
goto v_resetjp_95_;
}
v_resetjp_95_:
{
lean_object* v_optionsPerPos_98_; lean_object* v___x_99_; lean_object* v_name_100_; lean_object* v___x_101_; lean_object* v___x_103_; 
v_optionsPerPos_98_ = lean_ctor_get(v___y_86_, 0);
v___x_99_ = l_Lean_pp_funBinderTypes;
v_name_100_ = lean_ctor_get(v___x_99_, 0);
lean_inc(v_name_100_);
lean_inc(v_optionsPerPos_98_);
v___x_101_ = lp_mathlib_Lean_PrettyPrinter_Delaborator_OptionsPerPos_setBool(v_optionsPerPos_98_, v_a_94_, v_name_100_, v___x_85_);
if (v_isShared_97_ == 0)
{
lean_ctor_set(v___x_96_, 0, v___x_101_);
v___x_103_ = v___x_96_;
goto v_reusejp_102_;
}
else
{
lean_object* v_reuseFailAlloc_104_; 
v_reuseFailAlloc_104_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_104_, 0, v___x_101_);
v___x_103_ = v_reuseFailAlloc_104_;
goto v_reusejp_102_;
}
v_reusejp_102_:
{
return v___x_103_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_delabLinearIndependent___lam__1___boxed(lean_object* v___x_106_, lean_object* v___y_107_, lean_object* v___y_108_, lean_object* v___y_109_, lean_object* v___y_110_, lean_object* v___y_111_, lean_object* v___y_112_, lean_object* v___y_113_){
_start:
{
uint8_t v___x_5106__boxed_114_; lean_object* v_res_115_; 
v___x_5106__boxed_114_ = lean_unbox(v___x_106_);
v_res_115_ = lp_mathlib_delabLinearIndependent___lam__1(v___x_5106__boxed_114_, v___y_107_, v___y_108_, v___y_109_, v___y_110_, v___y_111_, v___y_112_);
lean_dec(v___y_112_);
lean_dec_ref(v___y_111_);
lean_dec(v___y_110_);
lean_dec_ref(v___y_109_);
lean_dec(v___y_108_);
lean_dec_ref(v___y_107_);
return v_res_115_;
}
}
static lean_object* _init_lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_withNaryArg___at___00delabLinearIndependent_spec__2___redArg___closed__0(void){
_start:
{
lean_object* v___x_116_; lean_object* v_dummy_117_; 
v___x_116_ = lean_box(0);
v_dummy_117_ = l_Lean_Expr_sort___override(v___x_116_);
return v_dummy_117_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_withNaryArg___at___00delabLinearIndependent_spec__2___redArg(lean_object* v_argIdx_118_, lean_object* v_x_119_, lean_object* v___y_120_, lean_object* v___y_121_, lean_object* v___y_122_, lean_object* v___y_123_, lean_object* v___y_124_, lean_object* v___y_125_){
_start:
{
lean_object* v___x_127_; lean_object* v_a_128_; lean_object* v___x_129_; lean_object* v_a_130_; lean_object* v_optionsPerPos_131_; lean_object* v_currNamespace_132_; lean_object* v_openDecls_133_; uint8_t v_inPattern_134_; lean_object* v_depth_135_; lean_object* v_lctxInitIndices_136_; lean_object* v_nargs_137_; lean_object* v___x_138_; lean_object* v_dummy_139_; lean_object* v___x_140_; lean_object* v___x_141_; lean_object* v___x_142_; lean_object* v_args_143_; lean_object* v___x_144_; lean_object* v_newPos_145_; lean_object* v___x_146_; lean_object* v___x_147_; lean_object* v___x_148_; lean_object* v___x_149_; 
v___x_127_ = lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_getExpr___at___00delabLinearIndependent_spec__0___redArg(v___y_120_);
v_a_128_ = lean_ctor_get(v___x_127_, 0);
lean_inc(v_a_128_);
lean_dec_ref(v___x_127_);
v___x_129_ = lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_getPos___at___00delabLinearIndependent_spec__1___redArg(v___y_120_);
v_a_130_ = lean_ctor_get(v___x_129_, 0);
lean_inc(v_a_130_);
lean_dec_ref(v___x_129_);
v_optionsPerPos_131_ = lean_ctor_get(v___y_120_, 0);
v_currNamespace_132_ = lean_ctor_get(v___y_120_, 1);
v_openDecls_133_ = lean_ctor_get(v___y_120_, 2);
v_inPattern_134_ = lean_ctor_get_uint8(v___y_120_, sizeof(void*)*6);
v_depth_135_ = lean_ctor_get(v___y_120_, 4);
v_lctxInitIndices_136_ = lean_ctor_get(v___y_120_, 5);
v_nargs_137_ = l_Lean_Expr_getAppNumArgs(v_a_128_);
v___x_138_ = l_Lean_instInhabitedExpr;
v_dummy_139_ = lean_obj_once(&lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_withNaryArg___at___00delabLinearIndependent_spec__2___redArg___closed__0, &lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_withNaryArg___at___00delabLinearIndependent_spec__2___redArg___closed__0_once, _init_lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_withNaryArg___at___00delabLinearIndependent_spec__2___redArg___closed__0);
lean_inc(v_nargs_137_);
v___x_140_ = lean_mk_array(v_nargs_137_, v_dummy_139_);
v___x_141_ = lean_unsigned_to_nat(1u);
v___x_142_ = lean_nat_sub(v_nargs_137_, v___x_141_);
lean_dec(v_nargs_137_);
v_args_143_ = l___private_Lean_Expr_0__Lean_Expr_getAppArgsAux(v_a_128_, v___x_140_, v___x_142_);
v___x_144_ = lean_array_get_size(v_args_143_);
v_newPos_145_ = l_Lean_SubExpr_Pos_pushNaryArg(v___x_144_, v_argIdx_118_, v_a_130_);
lean_dec(v_a_130_);
v___x_146_ = lean_array_get(v___x_138_, v_args_143_, v_argIdx_118_);
lean_dec_ref(v_args_143_);
v___x_147_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_147_, 0, v___x_146_);
lean_ctor_set(v___x_147_, 1, v_newPos_145_);
lean_inc(v_lctxInitIndices_136_);
lean_inc(v_depth_135_);
lean_inc(v_openDecls_133_);
lean_inc(v_currNamespace_132_);
lean_inc(v_optionsPerPos_131_);
v___x_148_ = lean_alloc_ctor(0, 6, 1);
lean_ctor_set(v___x_148_, 0, v_optionsPerPos_131_);
lean_ctor_set(v___x_148_, 1, v_currNamespace_132_);
lean_ctor_set(v___x_148_, 2, v_openDecls_133_);
lean_ctor_set(v___x_148_, 3, v___x_147_);
lean_ctor_set(v___x_148_, 4, v_depth_135_);
lean_ctor_set(v___x_148_, 5, v_lctxInitIndices_136_);
lean_ctor_set_uint8(v___x_148_, sizeof(void*)*6, v_inPattern_134_);
lean_inc(v___y_125_);
lean_inc_ref(v___y_124_);
lean_inc(v___y_123_);
lean_inc_ref(v___y_122_);
lean_inc(v___y_121_);
v___x_149_ = lean_apply_7(v_x_119_, v___x_148_, v___y_121_, v___y_122_, v___y_123_, v___y_124_, v___y_125_, lean_box(0));
return v___x_149_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_withNaryArg___at___00delabLinearIndependent_spec__2___redArg___boxed(lean_object* v_argIdx_150_, lean_object* v_x_151_, lean_object* v___y_152_, lean_object* v___y_153_, lean_object* v___y_154_, lean_object* v___y_155_, lean_object* v___y_156_, lean_object* v___y_157_, lean_object* v___y_158_){
_start:
{
lean_object* v_res_159_; 
v_res_159_ = lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_withNaryArg___at___00delabLinearIndependent_spec__2___redArg(v_argIdx_150_, v_x_151_, v___y_152_, v___y_153_, v___y_154_, v___y_155_, v___y_156_, v___y_157_);
lean_dec(v___y_157_);
lean_dec_ref(v___y_156_);
lean_dec(v___y_155_);
lean_dec_ref(v___y_154_);
lean_dec(v___y_153_);
lean_dec_ref(v___y_152_);
lean_dec(v_argIdx_150_);
return v_res_159_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_delabLinearIndependent___lam__2(lean_object* v___f_163_, lean_object* v___f_164_, lean_object* v___y_165_, lean_object* v___y_166_, lean_object* v___y_167_, lean_object* v___y_168_, lean_object* v___y_169_, lean_object* v___y_170_){
_start:
{
lean_object* v_optionsPerPos_173_; lean_object* v___y_174_; lean_object* v___y_175_; lean_object* v___y_176_; lean_object* v___y_177_; lean_object* v___y_178_; lean_object* v___y_179_; lean_object* v___x_188_; lean_object* v_a_189_; lean_object* v___x_223_; lean_object* v___x_224_; uint8_t v___x_225_; 
v___x_188_ = lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_getExpr___at___00delabLinearIndependent_spec__0___redArg(v___y_165_);
v_a_189_ = lean_ctor_get(v___x_188_, 0);
lean_inc(v_a_189_);
lean_dec_ref(v___x_188_);
v___x_223_ = ((lean_object*)(lp_mathlib_delabLinearIndependent___lam__2___closed__1));
v___x_224_ = lean_unsigned_to_nat(7u);
v___x_225_ = l_Lean_Expr_isAppOfArity(v_a_189_, v___x_223_, v___x_224_);
if (v___x_225_ == 0)
{
lean_object* v___x_226_; 
v___x_226_ = l_Lean_PrettyPrinter_Delaborator_failure___redArg();
if (lean_obj_tag(v___x_226_) == 0)
{
lean_dec_ref_known(v___x_226_, 1);
goto v___jp_190_;
}
else
{
lean_object* v_a_227_; lean_object* v___x_229_; uint8_t v_isShared_230_; uint8_t v_isSharedCheck_234_; 
lean_dec(v_a_189_);
lean_dec_ref(v___f_164_);
lean_dec_ref(v___f_163_);
v_a_227_ = lean_ctor_get(v___x_226_, 0);
v_isSharedCheck_234_ = !lean_is_exclusive(v___x_226_);
if (v_isSharedCheck_234_ == 0)
{
v___x_229_ = v___x_226_;
v_isShared_230_ = v_isSharedCheck_234_;
goto v_resetjp_228_;
}
else
{
lean_inc(v_a_227_);
lean_dec(v___x_226_);
v___x_229_ = lean_box(0);
v_isShared_230_ = v_isSharedCheck_234_;
goto v_resetjp_228_;
}
v_resetjp_228_:
{
lean_object* v___x_232_; 
if (v_isShared_230_ == 0)
{
v___x_232_ = v___x_229_;
goto v_reusejp_231_;
}
else
{
lean_object* v_reuseFailAlloc_233_; 
v_reuseFailAlloc_233_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_233_, 0, v_a_227_);
v___x_232_ = v_reuseFailAlloc_233_;
goto v_reusejp_231_;
}
v_reusejp_231_:
{
return v___x_232_;
}
}
}
}
else
{
goto v___jp_190_;
}
v___jp_172_:
{
lean_object* v_currNamespace_180_; lean_object* v_openDecls_181_; uint8_t v_inPattern_182_; lean_object* v_subExpr_183_; lean_object* v_depth_184_; lean_object* v_lctxInitIndices_185_; lean_object* v___x_186_; lean_object* v___x_187_; 
v_currNamespace_180_ = lean_ctor_get(v___y_174_, 1);
v_openDecls_181_ = lean_ctor_get(v___y_174_, 2);
v_inPattern_182_ = lean_ctor_get_uint8(v___y_174_, sizeof(void*)*6);
v_subExpr_183_ = lean_ctor_get(v___y_174_, 3);
v_depth_184_ = lean_ctor_get(v___y_174_, 4);
v_lctxInitIndices_185_ = lean_ctor_get(v___y_174_, 5);
lean_inc(v_lctxInitIndices_185_);
lean_inc(v_depth_184_);
lean_inc_ref(v_subExpr_183_);
lean_inc(v_openDecls_181_);
lean_inc(v_currNamespace_180_);
v___x_186_ = lean_alloc_ctor(0, 6, 1);
lean_ctor_set(v___x_186_, 0, v_optionsPerPos_173_);
lean_ctor_set(v___x_186_, 1, v_currNamespace_180_);
lean_ctor_set(v___x_186_, 2, v_openDecls_181_);
lean_ctor_set(v___x_186_, 3, v_subExpr_183_);
lean_ctor_set(v___x_186_, 4, v_depth_184_);
lean_ctor_set(v___x_186_, 5, v_lctxInitIndices_185_);
lean_ctor_set_uint8(v___x_186_, sizeof(void*)*6, v_inPattern_182_);
v___x_187_ = l_Lean_PrettyPrinter_Delaborator_delab(v___x_186_, v___y_175_, v___y_176_, v___y_177_, v___y_178_, v___y_179_);
lean_dec_ref_known(v___x_186_, 6);
return v___x_187_;
}
v___jp_190_:
{
lean_object* v___x_191_; lean_object* v___x_192_; lean_object* v___x_193_; lean_object* v___x_194_; lean_object* v___x_195_; 
v___x_191_ = l_Lean_Expr_getAppNumArgs(v_a_189_);
v___x_192_ = lean_unsigned_to_nat(1u);
v___x_193_ = lean_nat_sub(v___x_191_, v___x_192_);
v___x_194_ = l_Lean_Expr_getRevArg_x21(v_a_189_, v___x_193_);
v___x_195_ = lp_mathlib_Lean_Expr_coeTypeSet_x3f(v___x_194_);
lean_dec_ref(v___x_194_);
if (lean_obj_tag(v___x_195_) == 1)
{
lean_object* v___x_196_; lean_object* v___x_197_; lean_object* v___x_198_; lean_object* v___x_199_; uint8_t v___x_200_; 
lean_dec_ref_known(v___x_195_, 1);
v___x_196_ = lean_unsigned_to_nat(3u);
v___x_197_ = lean_nat_sub(v___x_191_, v___x_196_);
lean_dec(v___x_191_);
v___x_198_ = lean_nat_sub(v___x_197_, v___x_192_);
lean_dec(v___x_197_);
v___x_199_ = l_Lean_Expr_getRevArg_x21(v_a_189_, v___x_198_);
lean_dec(v_a_189_);
v___x_200_ = l_Lean_Expr_isLambda(v___x_199_);
lean_dec_ref(v___x_199_);
if (v___x_200_ == 0)
{
lean_object* v___x_201_; lean_object* v___x_202_; 
lean_dec_ref(v___f_164_);
v___x_201_ = lean_unsigned_to_nat(0u);
v___x_202_ = lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_withNaryArg___at___00delabLinearIndependent_spec__2___redArg(v___x_201_, v___f_163_, v___y_165_, v___y_166_, v___y_167_, v___y_168_, v___y_169_, v___y_170_);
if (lean_obj_tag(v___x_202_) == 0)
{
lean_object* v_a_203_; 
v_a_203_ = lean_ctor_get(v___x_202_, 0);
lean_inc(v_a_203_);
lean_dec_ref_known(v___x_202_, 1);
v_optionsPerPos_173_ = v_a_203_;
v___y_174_ = v___y_165_;
v___y_175_ = v___y_166_;
v___y_176_ = v___y_167_;
v___y_177_ = v___y_168_;
v___y_178_ = v___y_169_;
v___y_179_ = v___y_170_;
goto v___jp_172_;
}
else
{
lean_object* v_a_204_; lean_object* v___x_206_; uint8_t v_isShared_207_; uint8_t v_isSharedCheck_211_; 
v_a_204_ = lean_ctor_get(v___x_202_, 0);
v_isSharedCheck_211_ = !lean_is_exclusive(v___x_202_);
if (v_isSharedCheck_211_ == 0)
{
v___x_206_ = v___x_202_;
v_isShared_207_ = v_isSharedCheck_211_;
goto v_resetjp_205_;
}
else
{
lean_inc(v_a_204_);
lean_dec(v___x_202_);
v___x_206_ = lean_box(0);
v_isShared_207_ = v_isSharedCheck_211_;
goto v_resetjp_205_;
}
v_resetjp_205_:
{
lean_object* v___x_209_; 
if (v_isShared_207_ == 0)
{
v___x_209_ = v___x_206_;
goto v_reusejp_208_;
}
else
{
lean_object* v_reuseFailAlloc_210_; 
v_reuseFailAlloc_210_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_210_, 0, v_a_204_);
v___x_209_ = v_reuseFailAlloc_210_;
goto v_reusejp_208_;
}
v_reusejp_208_:
{
return v___x_209_;
}
}
}
}
else
{
lean_object* v___x_212_; 
lean_dec_ref(v___f_163_);
v___x_212_ = lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_withNaryArg___at___00delabLinearIndependent_spec__2___redArg(v___x_196_, v___f_164_, v___y_165_, v___y_166_, v___y_167_, v___y_168_, v___y_169_, v___y_170_);
if (lean_obj_tag(v___x_212_) == 0)
{
lean_object* v_a_213_; 
v_a_213_ = lean_ctor_get(v___x_212_, 0);
lean_inc(v_a_213_);
lean_dec_ref_known(v___x_212_, 1);
v_optionsPerPos_173_ = v_a_213_;
v___y_174_ = v___y_165_;
v___y_175_ = v___y_166_;
v___y_176_ = v___y_167_;
v___y_177_ = v___y_168_;
v___y_178_ = v___y_169_;
v___y_179_ = v___y_170_;
goto v___jp_172_;
}
else
{
lean_object* v_a_214_; lean_object* v___x_216_; uint8_t v_isShared_217_; uint8_t v_isSharedCheck_221_; 
v_a_214_ = lean_ctor_get(v___x_212_, 0);
v_isSharedCheck_221_ = !lean_is_exclusive(v___x_212_);
if (v_isSharedCheck_221_ == 0)
{
v___x_216_ = v___x_212_;
v_isShared_217_ = v_isSharedCheck_221_;
goto v_resetjp_215_;
}
else
{
lean_inc(v_a_214_);
lean_dec(v___x_212_);
v___x_216_ = lean_box(0);
v_isShared_217_ = v_isSharedCheck_221_;
goto v_resetjp_215_;
}
v_resetjp_215_:
{
lean_object* v___x_219_; 
if (v_isShared_217_ == 0)
{
v___x_219_ = v___x_216_;
goto v_reusejp_218_;
}
else
{
lean_object* v_reuseFailAlloc_220_; 
v_reuseFailAlloc_220_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_220_, 0, v_a_214_);
v___x_219_ = v_reuseFailAlloc_220_;
goto v_reusejp_218_;
}
v_reusejp_218_:
{
return v___x_219_;
}
}
}
}
}
else
{
lean_object* v___x_222_; 
lean_dec(v___x_195_);
lean_dec(v___x_191_);
lean_dec(v_a_189_);
lean_dec_ref(v___f_164_);
lean_dec_ref(v___f_163_);
v___x_222_ = l_Lean_PrettyPrinter_Delaborator_failure___redArg();
return v___x_222_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_delabLinearIndependent___lam__2___boxed(lean_object* v___f_235_, lean_object* v___f_236_, lean_object* v___y_237_, lean_object* v___y_238_, lean_object* v___y_239_, lean_object* v___y_240_, lean_object* v___y_241_, lean_object* v___y_242_, lean_object* v___y_243_){
_start:
{
lean_object* v_res_244_; 
v_res_244_ = lp_mathlib_delabLinearIndependent___lam__2(v___f_235_, v___f_236_, v___y_237_, v___y_238_, v___y_239_, v___y_240_, v___y_241_, v___y_242_);
lean_dec(v___y_242_);
lean_dec_ref(v___y_241_);
lean_dec(v___y_240_);
lean_dec_ref(v___y_239_);
lean_dec(v___y_238_);
lean_dec_ref(v___y_237_);
return v_res_244_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_delabLinearIndependent(lean_object* v_a_274_, lean_object* v_a_275_, lean_object* v_a_276_, lean_object* v_a_277_, lean_object* v_a_278_, lean_object* v_a_279_){
_start:
{
lean_object* v___x_281_; lean_object* v___x_282_; lean_object* v___x_283_; 
v___x_281_ = ((lean_object*)(lp_mathlib_delabLinearIndependent___closed__0));
v___x_282_ = ((lean_object*)(lp_mathlib_delabLinearIndependent___closed__11));
v___x_283_ = l_Lean_PrettyPrinter_Delaborator_whenPPOption(v___x_281_, v___x_282_, v_a_274_, v_a_275_, v_a_276_, v_a_277_, v_a_278_, v_a_279_);
return v___x_283_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_delabLinearIndependent___boxed(lean_object* v_a_284_, lean_object* v_a_285_, lean_object* v_a_286_, lean_object* v_a_287_, lean_object* v_a_288_, lean_object* v_a_289_, lean_object* v_a_290_){
_start:
{
lean_object* v_res_291_; 
v_res_291_ = lp_mathlib_delabLinearIndependent(v_a_284_, v_a_285_, v_a_286_, v_a_287_, v_a_288_, v_a_289_);
lean_dec(v_a_289_);
lean_dec_ref(v_a_288_);
lean_dec(v_a_287_);
lean_dec_ref(v_a_286_);
lean_dec(v_a_285_);
lean_dec_ref(v_a_284_);
return v_res_291_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_withNaryArg___at___00delabLinearIndependent_spec__2(lean_object* v_00_u03b1_292_, lean_object* v_argIdx_293_, lean_object* v_x_294_, lean_object* v___y_295_, lean_object* v___y_296_, lean_object* v___y_297_, lean_object* v___y_298_, lean_object* v___y_299_, lean_object* v___y_300_){
_start:
{
lean_object* v___x_302_; 
v___x_302_ = lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_withNaryArg___at___00delabLinearIndependent_spec__2___redArg(v_argIdx_293_, v_x_294_, v___y_295_, v___y_296_, v___y_297_, v___y_298_, v___y_299_, v___y_300_);
return v___x_302_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_withNaryArg___at___00delabLinearIndependent_spec__2___boxed(lean_object* v_00_u03b1_303_, lean_object* v_argIdx_304_, lean_object* v_x_305_, lean_object* v___y_306_, lean_object* v___y_307_, lean_object* v___y_308_, lean_object* v___y_309_, lean_object* v___y_310_, lean_object* v___y_311_, lean_object* v___y_312_){
_start:
{
lean_object* v_res_313_; 
v_res_313_ = lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_withNaryArg___at___00delabLinearIndependent_spec__2(v_00_u03b1_303_, v_argIdx_304_, v_x_305_, v___y_306_, v___y_307_, v___y_308_, v___y_309_, v___y_310_, v___y_311_);
lean_dec(v___y_311_);
lean_dec_ref(v___y_310_);
lean_dec(v___y_309_);
lean_dec_ref(v___y_308_);
lean_dec(v___y_307_);
lean_dec_ref(v___y_306_);
lean_dec(v_argIdx_304_);
return v_res_313_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Order_Sub_Basic(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_LinearAlgebra_Finsupp_LinearCombination(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_LinearAlgebra_LinearIndependent_Defs(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Order_Sub_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_LinearAlgebra_Finsupp_LinearCombination(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Lean_Expr_ExtraRecognizers(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_LinearAlgebra_LinearIndependent_Defs(uint8_t builtin) {
lean_object * res;
if (_G_meta_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_meta_initialized = true;
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Lean_Expr_ExtraRecognizers(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Order_Sub_Basic(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_LinearAlgebra_Finsupp_LinearCombination(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Lean_Expr_ExtraRecognizers(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_LinearAlgebra_LinearIndependent_Defs(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Order_Sub_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_LinearAlgebra_Finsupp_LinearCombination(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Lean_Expr_ExtraRecognizers(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_LinearAlgebra_LinearIndependent_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_LinearAlgebra_LinearIndependent_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_LinearAlgebra_LinearIndependent_Defs(builtin);
}
#ifdef __cplusplus
}
#endif
