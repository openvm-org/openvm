// Lean compiler output
// Module: Mathlib.Tactic.ClearExclamation
// Imports: public import Init public meta import Init public import Mathlib.Init public meta import Lean.Elab.Tactic.ElabTerm
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
extern lean_object* l_Lean_Elab_unsupportedSyntaxExceptionId;
lean_object* l_Lean_Name_mkStr3(lean_object*, lean_object*, lean_object*);
uint8_t l_Lean_Syntax_isOfKind(lean_object*, lean_object*);
lean_object* l_Lean_Syntax_getArg(lean_object*, lean_object*);
lean_object* l_Lean_Syntax_getArgs(lean_object*);
lean_object* l_Lean_Elab_Tactic_getFVarIds(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Elab_Tactic_getMainGoal___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
size_t lean_array_size(lean_object*);
uint8_t lean_usize_dec_lt(size_t, size_t);
lean_object* lean_array_uget(lean_object*, size_t);
lean_object* lean_array_uset(lean_object*, size_t, lean_object*);
lean_object* l_Lean_Expr_fvar___override(lean_object*);
size_t lean_usize_add(size_t, size_t);
lean_object* l_Lean_Meta_collectForwardDeps(lean_object*, uint8_t, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Expr_fvarId_x21(lean_object*);
lean_object* l_Lean_MVarId_tryClearMany(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Elab_Tactic_replaceMainGoal___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Elab_Tactic_withMainContext___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Name_mkStr1(lean_object*);
static const lean_string_object lp_mathlib_Mathlib_Tactic_clear_x21___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "Mathlib"};
static const lean_object* lp_mathlib_Mathlib_Tactic_clear_x21___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_clear_x21___closed__0_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_clear_x21___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "Tactic"};
static const lean_object* lp_mathlib_Mathlib_Tactic_clear_x21___closed__1 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_clear_x21___closed__1_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_clear_x21___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "clear!"};
static const lean_object* lp_mathlib_Mathlib_Tactic_clear_x21___closed__2 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_clear_x21___closed__2_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_clear_x21___closed__3_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_clear_x21___closed__0_value),LEAN_SCALAR_PTR_LITERAL(118, 213, 161, 2, 73, 184, 31, 228)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_clear_x21___closed__3_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_clear_x21___closed__3_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic_clear_x21___closed__1_value),LEAN_SCALAR_PTR_LITERAL(139, 222, 98, 232, 116, 132, 69, 249)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_clear_x21___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_clear_x21___closed__3_value_aux_1),((lean_object*)&lp_mathlib_Mathlib_Tactic_clear_x21___closed__2_value),LEAN_SCALAR_PTR_LITERAL(205, 200, 179, 253, 49, 249, 24, 166)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_clear_x21___closed__3 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_clear_x21___closed__3_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_clear_x21___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "andthen"};
static const lean_object* lp_mathlib_Mathlib_Tactic_clear_x21___closed__4 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_clear_x21___closed__4_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_clear_x21___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_clear_x21___closed__4_value),LEAN_SCALAR_PTR_LITERAL(40, 255, 78, 30, 143, 119, 117, 174)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_clear_x21___closed__5 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_clear_x21___closed__5_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_clear_x21___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 8, .m_other = 1, .m_tag = 6}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_clear_x21___closed__2_value),LEAN_SCALAR_PTR_LITERAL(0, 0, 0, 0, 0, 0, 0, 0)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_clear_x21___closed__6 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_clear_x21___closed__6_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_clear_x21___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "many"};
static const lean_object* lp_mathlib_Mathlib_Tactic_clear_x21___closed__7 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_clear_x21___closed__7_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_clear_x21___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_clear_x21___closed__7_value),LEAN_SCALAR_PTR_LITERAL(41, 35, 40, 86, 189, 97, 244, 31)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_clear_x21___closed__8 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_clear_x21___closed__8_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_clear_x21___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "ppSpace"};
static const lean_object* lp_mathlib_Mathlib_Tactic_clear_x21___closed__9 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_clear_x21___closed__9_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_clear_x21___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_clear_x21___closed__9_value),LEAN_SCALAR_PTR_LITERAL(207, 47, 58, 43, 30, 240, 125, 246)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_clear_x21___closed__10 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_clear_x21___closed__10_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_clear_x21___closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_clear_x21___closed__10_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_clear_x21___closed__11 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_clear_x21___closed__11_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_clear_x21___closed__12_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "colGt"};
static const lean_object* lp_mathlib_Mathlib_Tactic_clear_x21___closed__12 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_clear_x21___closed__12_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_clear_x21___closed__13_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_clear_x21___closed__12_value),LEAN_SCALAR_PTR_LITERAL(185, 236, 32, 153, 169, 213, 53, 244)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_clear_x21___closed__13 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_clear_x21___closed__13_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_clear_x21___closed__14_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_clear_x21___closed__13_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_clear_x21___closed__14 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_clear_x21___closed__14_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_clear_x21___closed__15_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_clear_x21___closed__5_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_clear_x21___closed__11_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_clear_x21___closed__14_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_clear_x21___closed__15 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_clear_x21___closed__15_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_clear_x21___closed__16_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "ident"};
static const lean_object* lp_mathlib_Mathlib_Tactic_clear_x21___closed__16 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_clear_x21___closed__16_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_clear_x21___closed__17_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_clear_x21___closed__16_value),LEAN_SCALAR_PTR_LITERAL(52, 159, 208, 51, 14, 60, 6, 71)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_clear_x21___closed__17 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_clear_x21___closed__17_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_clear_x21___closed__18_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_clear_x21___closed__17_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_clear_x21___closed__18 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_clear_x21___closed__18_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_clear_x21___closed__19_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_clear_x21___closed__5_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_clear_x21___closed__15_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_clear_x21___closed__18_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_clear_x21___closed__19 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_clear_x21___closed__19_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_clear_x21___closed__20_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_clear_x21___closed__8_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_clear_x21___closed__19_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_clear_x21___closed__20 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_clear_x21___closed__20_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_clear_x21___closed__21_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_clear_x21___closed__5_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_clear_x21___closed__6_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_clear_x21___closed__20_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_clear_x21___closed__21 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_clear_x21___closed__21_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_clear_x21___closed__22_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 3}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_clear_x21___closed__3_value),((lean_object*)(((size_t)(1022) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_clear_x21___closed__21_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_clear_x21___closed__22 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_clear_x21___closed__22_value;
LEAN_EXPORT const lean_object* lp_mathlib_Mathlib_Tactic_clear_x21 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_clear_x21___closed__22_value;
static lean_once_cell_t lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic___aux__Mathlib__Tactic__ClearExclamation______elabRules__Mathlib__Tactic__clear_x21__1_spec__0___redArg___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic___aux__Mathlib__Tactic__ClearExclamation______elabRules__Mathlib__Tactic__clear_x21__1_spec__0___redArg___closed__0;
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic___aux__Mathlib__Tactic__ClearExclamation______elabRules__Mathlib__Tactic__clear_x21__1_spec__0___redArg();
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic___aux__Mathlib__Tactic__ClearExclamation______elabRules__Mathlib__Tactic__clear_x21__1_spec__0___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic___aux__Mathlib__Tactic__ClearExclamation______elabRules__Mathlib__Tactic__clear_x21__1_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic___aux__Mathlib__Tactic__ClearExclamation______elabRules__Mathlib__Tactic__clear_x21__1_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Mathlib_Tactic___aux__Mathlib__Tactic__ClearExclamation______elabRules__Mathlib__Tactic__clear_x21__1_spec__1(size_t, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Mathlib_Tactic___aux__Mathlib__Tactic__ClearExclamation______elabRules__Mathlib__Tactic__clear_x21__1_spec__1___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Mathlib_Tactic___aux__Mathlib__Tactic__ClearExclamation______elabRules__Mathlib__Tactic__clear_x21__1_spec__2(size_t, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Mathlib_Tactic___aux__Mathlib__Tactic__ClearExclamation______elabRules__Mathlib__Tactic__clear_x21__1_spec__2___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__ClearExclamation______elabRules__Mathlib__Tactic__clear_x21__1___lam__0(lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__ClearExclamation______elabRules__Mathlib__Tactic__clear_x21__1___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__ClearExclamation______elabRules__Mathlib__Tactic__clear_x21__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__ClearExclamation______elabRules__Mathlib__Tactic__clear_x21__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_object* _init_lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic___aux__Mathlib__Tactic__ClearExclamation______elabRules__Mathlib__Tactic__clear_x21__1_spec__0___redArg___closed__0(void){
_start:
{
lean_object* v___x_52_; lean_object* v___x_53_; lean_object* v___x_54_; 
v___x_52_ = lean_box(0);
v___x_53_ = l_Lean_Elab_unsupportedSyntaxExceptionId;
v___x_54_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_54_, 0, v___x_53_);
lean_ctor_set(v___x_54_, 1, v___x_52_);
return v___x_54_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic___aux__Mathlib__Tactic__ClearExclamation______elabRules__Mathlib__Tactic__clear_x21__1_spec__0___redArg(){
_start:
{
lean_object* v___x_56_; lean_object* v___x_57_; 
v___x_56_ = lean_obj_once(&lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic___aux__Mathlib__Tactic__ClearExclamation______elabRules__Mathlib__Tactic__clear_x21__1_spec__0___redArg___closed__0, &lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic___aux__Mathlib__Tactic__ClearExclamation______elabRules__Mathlib__Tactic__clear_x21__1_spec__0___redArg___closed__0_once, _init_lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic___aux__Mathlib__Tactic__ClearExclamation______elabRules__Mathlib__Tactic__clear_x21__1_spec__0___redArg___closed__0);
v___x_57_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_57_, 0, v___x_56_);
return v___x_57_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic___aux__Mathlib__Tactic__ClearExclamation______elabRules__Mathlib__Tactic__clear_x21__1_spec__0___redArg___boxed(lean_object* v___y_58_){
_start:
{
lean_object* v_res_59_; 
v_res_59_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic___aux__Mathlib__Tactic__ClearExclamation______elabRules__Mathlib__Tactic__clear_x21__1_spec__0___redArg();
return v_res_59_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic___aux__Mathlib__Tactic__ClearExclamation______elabRules__Mathlib__Tactic__clear_x21__1_spec__0(lean_object* v_00_u03b1_60_, lean_object* v___y_61_, lean_object* v___y_62_, lean_object* v___y_63_, lean_object* v___y_64_, lean_object* v___y_65_, lean_object* v___y_66_, lean_object* v___y_67_, lean_object* v___y_68_){
_start:
{
lean_object* v___x_70_; 
v___x_70_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic___aux__Mathlib__Tactic__ClearExclamation______elabRules__Mathlib__Tactic__clear_x21__1_spec__0___redArg();
return v___x_70_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic___aux__Mathlib__Tactic__ClearExclamation______elabRules__Mathlib__Tactic__clear_x21__1_spec__0___boxed(lean_object* v_00_u03b1_71_, lean_object* v___y_72_, lean_object* v___y_73_, lean_object* v___y_74_, lean_object* v___y_75_, lean_object* v___y_76_, lean_object* v___y_77_, lean_object* v___y_78_, lean_object* v___y_79_, lean_object* v___y_80_){
_start:
{
lean_object* v_res_81_; 
v_res_81_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic___aux__Mathlib__Tactic__ClearExclamation______elabRules__Mathlib__Tactic__clear_x21__1_spec__0(v_00_u03b1_71_, v___y_72_, v___y_73_, v___y_74_, v___y_75_, v___y_76_, v___y_77_, v___y_78_, v___y_79_);
lean_dec(v___y_79_);
lean_dec_ref(v___y_78_);
lean_dec(v___y_77_);
lean_dec_ref(v___y_76_);
lean_dec(v___y_75_);
lean_dec_ref(v___y_74_);
lean_dec(v___y_73_);
lean_dec_ref(v___y_72_);
return v_res_81_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Mathlib_Tactic___aux__Mathlib__Tactic__ClearExclamation______elabRules__Mathlib__Tactic__clear_x21__1_spec__1(size_t v_sz_82_, size_t v_i_83_, lean_object* v_bs_84_){
_start:
{
uint8_t v___x_85_; 
v___x_85_ = lean_usize_dec_lt(v_i_83_, v_sz_82_);
if (v___x_85_ == 0)
{
return v_bs_84_;
}
else
{
lean_object* v_v_86_; lean_object* v___x_87_; lean_object* v_bs_x27_88_; lean_object* v___x_89_; size_t v___x_90_; size_t v___x_91_; lean_object* v___x_92_; 
v_v_86_ = lean_array_uget(v_bs_84_, v_i_83_);
v___x_87_ = lean_unsigned_to_nat(0u);
v_bs_x27_88_ = lean_array_uset(v_bs_84_, v_i_83_, v___x_87_);
v___x_89_ = l_Lean_Expr_fvar___override(v_v_86_);
v___x_90_ = ((size_t)1ULL);
v___x_91_ = lean_usize_add(v_i_83_, v___x_90_);
v___x_92_ = lean_array_uset(v_bs_x27_88_, v_i_83_, v___x_89_);
v_i_83_ = v___x_91_;
v_bs_84_ = v___x_92_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Mathlib_Tactic___aux__Mathlib__Tactic__ClearExclamation______elabRules__Mathlib__Tactic__clear_x21__1_spec__1___boxed(lean_object* v_sz_94_, lean_object* v_i_95_, lean_object* v_bs_96_){
_start:
{
size_t v_sz_boxed_97_; size_t v_i_boxed_98_; lean_object* v_res_99_; 
v_sz_boxed_97_ = lean_unbox_usize(v_sz_94_);
lean_dec(v_sz_94_);
v_i_boxed_98_ = lean_unbox_usize(v_i_95_);
lean_dec(v_i_95_);
v_res_99_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Mathlib_Tactic___aux__Mathlib__Tactic__ClearExclamation______elabRules__Mathlib__Tactic__clear_x21__1_spec__1(v_sz_boxed_97_, v_i_boxed_98_, v_bs_96_);
return v_res_99_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Mathlib_Tactic___aux__Mathlib__Tactic__ClearExclamation______elabRules__Mathlib__Tactic__clear_x21__1_spec__2(size_t v_sz_100_, size_t v_i_101_, lean_object* v_bs_102_){
_start:
{
uint8_t v___x_103_; 
v___x_103_ = lean_usize_dec_lt(v_i_101_, v_sz_100_);
if (v___x_103_ == 0)
{
return v_bs_102_;
}
else
{
lean_object* v_v_104_; lean_object* v___x_105_; lean_object* v_bs_x27_106_; lean_object* v___x_107_; size_t v___x_108_; size_t v___x_109_; lean_object* v___x_110_; 
v_v_104_ = lean_array_uget(v_bs_102_, v_i_101_);
v___x_105_ = lean_unsigned_to_nat(0u);
v_bs_x27_106_ = lean_array_uset(v_bs_102_, v_i_101_, v___x_105_);
v___x_107_ = l_Lean_Expr_fvarId_x21(v_v_104_);
lean_dec(v_v_104_);
v___x_108_ = ((size_t)1ULL);
v___x_109_ = lean_usize_add(v_i_101_, v___x_108_);
v___x_110_ = lean_array_uset(v_bs_x27_106_, v_i_101_, v___x_107_);
v_i_101_ = v___x_109_;
v_bs_102_ = v___x_110_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Mathlib_Tactic___aux__Mathlib__Tactic__ClearExclamation______elabRules__Mathlib__Tactic__clear_x21__1_spec__2___boxed(lean_object* v_sz_112_, lean_object* v_i_113_, lean_object* v_bs_114_){
_start:
{
size_t v_sz_boxed_115_; size_t v_i_boxed_116_; lean_object* v_res_117_; 
v_sz_boxed_115_ = lean_unbox_usize(v_sz_112_);
lean_dec(v_sz_112_);
v_i_boxed_116_ = lean_unbox_usize(v_i_113_);
lean_dec(v_i_113_);
v_res_117_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Mathlib_Tactic___aux__Mathlib__Tactic__ClearExclamation______elabRules__Mathlib__Tactic__clear_x21__1_spec__2(v_sz_boxed_115_, v_i_boxed_116_, v_bs_114_);
return v_res_117_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__ClearExclamation______elabRules__Mathlib__Tactic__clear_x21__1___lam__0(lean_object* v_a_118_, uint8_t v___x_119_, lean_object* v___y_120_, lean_object* v___y_121_, lean_object* v___y_122_, lean_object* v___y_123_, lean_object* v___y_124_, lean_object* v___y_125_, lean_object* v___y_126_, lean_object* v___y_127_){
_start:
{
lean_object* v___x_129_; 
v___x_129_ = l_Lean_Elab_Tactic_getMainGoal___redArg(v___y_121_, v___y_124_, v___y_125_, v___y_126_, v___y_127_);
if (lean_obj_tag(v___x_129_) == 0)
{
lean_object* v_a_130_; size_t v_sz_131_; size_t v___x_132_; lean_object* v___x_133_; lean_object* v___x_134_; 
v_a_130_ = lean_ctor_get(v___x_129_, 0);
lean_inc(v_a_130_);
lean_dec_ref_known(v___x_129_, 1);
v_sz_131_ = lean_array_size(v_a_118_);
v___x_132_ = ((size_t)0ULL);
v___x_133_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Mathlib_Tactic___aux__Mathlib__Tactic__ClearExclamation______elabRules__Mathlib__Tactic__clear_x21__1_spec__1(v_sz_131_, v___x_132_, v_a_118_);
v___x_134_ = l_Lean_Meta_collectForwardDeps(v___x_133_, v___x_119_, v___x_119_, v___y_124_, v___y_125_, v___y_126_, v___y_127_);
if (lean_obj_tag(v___x_134_) == 0)
{
lean_object* v_a_135_; size_t v_sz_136_; lean_object* v___x_137_; lean_object* v___x_138_; 
v_a_135_ = lean_ctor_get(v___x_134_, 0);
lean_inc(v_a_135_);
lean_dec_ref_known(v___x_134_, 1);
v_sz_136_ = lean_array_size(v_a_135_);
v___x_137_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Mathlib_Tactic___aux__Mathlib__Tactic__ClearExclamation______elabRules__Mathlib__Tactic__clear_x21__1_spec__2(v_sz_136_, v___x_132_, v_a_135_);
v___x_138_ = l_Lean_MVarId_tryClearMany(v_a_130_, v___x_137_, v___y_124_, v___y_125_, v___y_126_, v___y_127_);
lean_dec_ref(v___x_137_);
if (lean_obj_tag(v___x_138_) == 0)
{
lean_object* v_a_139_; lean_object* v___x_140_; lean_object* v___x_141_; lean_object* v___x_142_; 
v_a_139_ = lean_ctor_get(v___x_138_, 0);
lean_inc(v_a_139_);
lean_dec_ref_known(v___x_138_, 1);
v___x_140_ = lean_box(0);
v___x_141_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_141_, 0, v_a_139_);
lean_ctor_set(v___x_141_, 1, v___x_140_);
v___x_142_ = l_Lean_Elab_Tactic_replaceMainGoal___redArg(v___x_141_, v___y_121_, v___y_124_, v___y_125_, v___y_126_, v___y_127_);
return v___x_142_;
}
else
{
lean_object* v_a_143_; lean_object* v___x_145_; uint8_t v_isShared_146_; uint8_t v_isSharedCheck_150_; 
v_a_143_ = lean_ctor_get(v___x_138_, 0);
v_isSharedCheck_150_ = !lean_is_exclusive(v___x_138_);
if (v_isSharedCheck_150_ == 0)
{
v___x_145_ = v___x_138_;
v_isShared_146_ = v_isSharedCheck_150_;
goto v_resetjp_144_;
}
else
{
lean_inc(v_a_143_);
lean_dec(v___x_138_);
v___x_145_ = lean_box(0);
v_isShared_146_ = v_isSharedCheck_150_;
goto v_resetjp_144_;
}
v_resetjp_144_:
{
lean_object* v___x_148_; 
if (v_isShared_146_ == 0)
{
v___x_148_ = v___x_145_;
goto v_reusejp_147_;
}
else
{
lean_object* v_reuseFailAlloc_149_; 
v_reuseFailAlloc_149_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_149_, 0, v_a_143_);
v___x_148_ = v_reuseFailAlloc_149_;
goto v_reusejp_147_;
}
v_reusejp_147_:
{
return v___x_148_;
}
}
}
}
else
{
lean_object* v_a_151_; lean_object* v___x_153_; uint8_t v_isShared_154_; uint8_t v_isSharedCheck_158_; 
lean_dec(v_a_130_);
v_a_151_ = lean_ctor_get(v___x_134_, 0);
v_isSharedCheck_158_ = !lean_is_exclusive(v___x_134_);
if (v_isSharedCheck_158_ == 0)
{
v___x_153_ = v___x_134_;
v_isShared_154_ = v_isSharedCheck_158_;
goto v_resetjp_152_;
}
else
{
lean_inc(v_a_151_);
lean_dec(v___x_134_);
v___x_153_ = lean_box(0);
v_isShared_154_ = v_isSharedCheck_158_;
goto v_resetjp_152_;
}
v_resetjp_152_:
{
lean_object* v___x_156_; 
if (v_isShared_154_ == 0)
{
v___x_156_ = v___x_153_;
goto v_reusejp_155_;
}
else
{
lean_object* v_reuseFailAlloc_157_; 
v_reuseFailAlloc_157_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_157_, 0, v_a_151_);
v___x_156_ = v_reuseFailAlloc_157_;
goto v_reusejp_155_;
}
v_reusejp_155_:
{
return v___x_156_;
}
}
}
}
else
{
lean_object* v_a_159_; lean_object* v___x_161_; uint8_t v_isShared_162_; uint8_t v_isSharedCheck_166_; 
lean_dec_ref(v_a_118_);
v_a_159_ = lean_ctor_get(v___x_129_, 0);
v_isSharedCheck_166_ = !lean_is_exclusive(v___x_129_);
if (v_isSharedCheck_166_ == 0)
{
v___x_161_ = v___x_129_;
v_isShared_162_ = v_isSharedCheck_166_;
goto v_resetjp_160_;
}
else
{
lean_inc(v_a_159_);
lean_dec(v___x_129_);
v___x_161_ = lean_box(0);
v_isShared_162_ = v_isSharedCheck_166_;
goto v_resetjp_160_;
}
v_resetjp_160_:
{
lean_object* v___x_164_; 
if (v_isShared_162_ == 0)
{
v___x_164_ = v___x_161_;
goto v_reusejp_163_;
}
else
{
lean_object* v_reuseFailAlloc_165_; 
v_reuseFailAlloc_165_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_165_, 0, v_a_159_);
v___x_164_ = v_reuseFailAlloc_165_;
goto v_reusejp_163_;
}
v_reusejp_163_:
{
return v___x_164_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__ClearExclamation______elabRules__Mathlib__Tactic__clear_x21__1___lam__0___boxed(lean_object* v_a_167_, lean_object* v___x_168_, lean_object* v___y_169_, lean_object* v___y_170_, lean_object* v___y_171_, lean_object* v___y_172_, lean_object* v___y_173_, lean_object* v___y_174_, lean_object* v___y_175_, lean_object* v___y_176_, lean_object* v___y_177_){
_start:
{
uint8_t v___x_1371__boxed_178_; lean_object* v_res_179_; 
v___x_1371__boxed_178_ = lean_unbox(v___x_168_);
v_res_179_ = lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__ClearExclamation______elabRules__Mathlib__Tactic__clear_x21__1___lam__0(v_a_167_, v___x_1371__boxed_178_, v___y_169_, v___y_170_, v___y_171_, v___y_172_, v___y_173_, v___y_174_, v___y_175_, v___y_176_);
lean_dec(v___y_176_);
lean_dec_ref(v___y_175_);
lean_dec(v___y_174_);
lean_dec_ref(v___y_173_);
lean_dec(v___y_172_);
lean_dec_ref(v___y_171_);
lean_dec(v___y_170_);
lean_dec_ref(v___y_169_);
return v_res_179_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__ClearExclamation______elabRules__Mathlib__Tactic__clear_x21__1(lean_object* v_x_180_, lean_object* v_a_181_, lean_object* v_a_182_, lean_object* v_a_183_, lean_object* v_a_184_, lean_object* v_a_185_, lean_object* v_a_186_, lean_object* v_a_187_, lean_object* v_a_188_){
_start:
{
lean_object* v___x_190_; uint8_t v___x_191_; 
v___x_190_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_clear_x21___closed__3));
lean_inc(v_x_180_);
v___x_191_ = l_Lean_Syntax_isOfKind(v_x_180_, v___x_190_);
if (v___x_191_ == 0)
{
lean_object* v___x_192_; 
lean_dec(v_x_180_);
v___x_192_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic___aux__Mathlib__Tactic__ClearExclamation______elabRules__Mathlib__Tactic__clear_x21__1_spec__0___redArg();
return v___x_192_;
}
else
{
lean_object* v___x_193_; lean_object* v___x_194_; lean_object* v_hs_195_; lean_object* v___x_196_; 
v___x_193_ = lean_unsigned_to_nat(1u);
v___x_194_ = l_Lean_Syntax_getArg(v_x_180_, v___x_193_);
lean_dec(v_x_180_);
v_hs_195_ = l_Lean_Syntax_getArgs(v___x_194_);
lean_dec(v___x_194_);
v___x_196_ = l_Lean_Elab_Tactic_getFVarIds(v_hs_195_, v_a_181_, v_a_182_, v_a_183_, v_a_184_, v_a_185_, v_a_186_, v_a_187_, v_a_188_);
if (lean_obj_tag(v___x_196_) == 0)
{
lean_object* v_a_197_; lean_object* v___x_198_; lean_object* v___f_199_; lean_object* v___x_200_; 
v_a_197_ = lean_ctor_get(v___x_196_, 0);
lean_inc(v_a_197_);
lean_dec_ref_known(v___x_196_, 1);
v___x_198_ = lean_box(v___x_191_);
v___f_199_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__ClearExclamation______elabRules__Mathlib__Tactic__clear_x21__1___lam__0___boxed), 11, 2);
lean_closure_set(v___f_199_, 0, v_a_197_);
lean_closure_set(v___f_199_, 1, v___x_198_);
v___x_200_ = l_Lean_Elab_Tactic_withMainContext___redArg(v___f_199_, v_a_181_, v_a_182_, v_a_183_, v_a_184_, v_a_185_, v_a_186_, v_a_187_, v_a_188_);
return v___x_200_;
}
else
{
lean_object* v_a_201_; lean_object* v___x_203_; uint8_t v_isShared_204_; uint8_t v_isSharedCheck_208_; 
v_a_201_ = lean_ctor_get(v___x_196_, 0);
v_isSharedCheck_208_ = !lean_is_exclusive(v___x_196_);
if (v_isSharedCheck_208_ == 0)
{
v___x_203_ = v___x_196_;
v_isShared_204_ = v_isSharedCheck_208_;
goto v_resetjp_202_;
}
else
{
lean_inc(v_a_201_);
lean_dec(v___x_196_);
v___x_203_ = lean_box(0);
v_isShared_204_ = v_isSharedCheck_208_;
goto v_resetjp_202_;
}
v_resetjp_202_:
{
lean_object* v___x_206_; 
if (v_isShared_204_ == 0)
{
v___x_206_ = v___x_203_;
goto v_reusejp_205_;
}
else
{
lean_object* v_reuseFailAlloc_207_; 
v_reuseFailAlloc_207_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_207_, 0, v_a_201_);
v___x_206_ = v_reuseFailAlloc_207_;
goto v_reusejp_205_;
}
v_reusejp_205_:
{
return v___x_206_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__ClearExclamation______elabRules__Mathlib__Tactic__clear_x21__1___boxed(lean_object* v_x_209_, lean_object* v_a_210_, lean_object* v_a_211_, lean_object* v_a_212_, lean_object* v_a_213_, lean_object* v_a_214_, lean_object* v_a_215_, lean_object* v_a_216_, lean_object* v_a_217_, lean_object* v_a_218_){
_start:
{
lean_object* v_res_219_; 
v_res_219_ = lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__ClearExclamation______elabRules__Mathlib__Tactic__clear_x21__1(v_x_209_, v_a_210_, v_a_211_, v_a_212_, v_a_213_, v_a_214_, v_a_215_, v_a_216_, v_a_217_);
lean_dec(v_a_217_);
lean_dec_ref(v_a_216_);
lean_dec(v_a_215_);
lean_dec_ref(v_a_214_);
lean_dec(v_a_213_);
lean_dec_ref(v_a_212_);
lean_dec(v_a_211_);
lean_dec_ref(v_a_210_);
return v_res_219_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Init(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_Tactic_ClearExclamation(uint8_t builtin) {
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
lean_object* runtime_initialize_Lean_Elab_Tactic_ElabTerm(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_Tactic_ClearExclamation(uint8_t builtin) {
lean_object * res;
if (_G_meta_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_meta_initialized = true;
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_Lean_Elab_Tactic_ElabTerm(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Init(uint8_t builtin);
lean_object* initialize_Lean_Elab_Tactic_ElabTerm(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Tactic_ClearExclamation(uint8_t builtin) {
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
res = initialize_Lean_Elab_Tactic_ElabTerm(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Tactic_ClearExclamation(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_Tactic_ClearExclamation(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_Tactic_ClearExclamation(builtin);
}
#ifdef __cplusplus
}
#endif
