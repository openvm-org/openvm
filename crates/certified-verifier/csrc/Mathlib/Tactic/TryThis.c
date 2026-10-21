// Lean compiler output
// Module: Mathlib.Tactic.TryThis
// Imports: public import Init public meta import Init public import Mathlib.Init public meta import Lean.Meta.Tactic.TryThis
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
lean_object* l_Lean_Name_mkStr1(lean_object*);
lean_object* l_Lean_Name_mkStr3(lean_object*, lean_object*, lean_object*);
uint8_t l_Lean_Syntax_isOfKind(lean_object*, lean_object*);
lean_object* l_Lean_Syntax_getArg(lean_object*, lean_object*);
extern lean_object* l_Lean_MessageData_nil;
lean_object* l_Lean_Meta_Tactic_TryThis_addSuggestion(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Elab_Tactic_evalTactic(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_TSyntax_getString(lean_object*);
lean_object* l_Lean_Syntax_getOptional_x3f(lean_object*);
static const lean_string_object lp_mathlib_Mathlib_Tactic_tacticTry__this_____00__closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "Mathlib"};
static const lean_object* lp_mathlib_Mathlib_Tactic_tacticTry__this_____00__closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_tacticTry__this_____00__closed__0_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_tacticTry__this_____00__closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "Tactic"};
static const lean_object* lp_mathlib_Mathlib_Tactic_tacticTry__this_____00__closed__1 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_tacticTry__this_____00__closed__1_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_tacticTry__this_____00__closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 17, .m_capacity = 17, .m_length = 16, .m_data = "tacticTry_this__"};
static const lean_object* lp_mathlib_Mathlib_Tactic_tacticTry__this_____00__closed__2 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_tacticTry__this_____00__closed__2_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_tacticTry__this_____00__closed__3_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_tacticTry__this_____00__closed__0_value),LEAN_SCALAR_PTR_LITERAL(118, 213, 161, 2, 73, 184, 31, 228)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_tacticTry__this_____00__closed__3_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_tacticTry__this_____00__closed__3_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic_tacticTry__this_____00__closed__1_value),LEAN_SCALAR_PTR_LITERAL(139, 222, 98, 232, 116, 132, 69, 249)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_tacticTry__this_____00__closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_tacticTry__this_____00__closed__3_value_aux_1),((lean_object*)&lp_mathlib_Mathlib_Tactic_tacticTry__this_____00__closed__2_value),LEAN_SCALAR_PTR_LITERAL(40, 25, 74, 169, 214, 21, 160, 3)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_tacticTry__this_____00__closed__3 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_tacticTry__this_____00__closed__3_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_tacticTry__this_____00__closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "andthen"};
static const lean_object* lp_mathlib_Mathlib_Tactic_tacticTry__this_____00__closed__4 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_tacticTry__this_____00__closed__4_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_tacticTry__this_____00__closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_tacticTry__this_____00__closed__4_value),LEAN_SCALAR_PTR_LITERAL(40, 255, 78, 30, 143, 119, 117, 174)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_tacticTry__this_____00__closed__5 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_tacticTry__this_____00__closed__5_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_tacticTry__this_____00__closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "try_this"};
static const lean_object* lp_mathlib_Mathlib_Tactic_tacticTry__this_____00__closed__6 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_tacticTry__this_____00__closed__6_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_tacticTry__this_____00__closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 8, .m_other = 1, .m_tag = 6}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_tacticTry__this_____00__closed__6_value),LEAN_SCALAR_PTR_LITERAL(0, 0, 0, 0, 0, 0, 0, 0)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_tacticTry__this_____00__closed__7 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_tacticTry__this_____00__closed__7_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_tacticTry__this_____00__closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "tactic"};
static const lean_object* lp_mathlib_Mathlib_Tactic_tacticTry__this_____00__closed__8 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_tacticTry__this_____00__closed__8_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_tacticTry__this_____00__closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_tacticTry__this_____00__closed__8_value),LEAN_SCALAR_PTR_LITERAL(99, 76, 33, 121, 85, 143, 17, 224)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_tacticTry__this_____00__closed__9 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_tacticTry__this_____00__closed__9_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_tacticTry__this_____00__closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 7}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_tacticTry__this_____00__closed__9_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_Mathlib_Tactic_tacticTry__this_____00__closed__10 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_tacticTry__this_____00__closed__10_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_tacticTry__this_____00__closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_tacticTry__this_____00__closed__5_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_tacticTry__this_____00__closed__7_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_tacticTry__this_____00__closed__10_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_tacticTry__this_____00__closed__11 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_tacticTry__this_____00__closed__11_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_tacticTry__this_____00__closed__12_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "optional"};
static const lean_object* lp_mathlib_Mathlib_Tactic_tacticTry__this_____00__closed__12 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_tacticTry__this_____00__closed__12_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_tacticTry__this_____00__closed__13_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_tacticTry__this_____00__closed__12_value),LEAN_SCALAR_PTR_LITERAL(233, 141, 154, 50, 143, 135, 42, 252)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_tacticTry__this_____00__closed__13 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_tacticTry__this_____00__closed__13_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_tacticTry__this_____00__closed__14_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = "str"};
static const lean_object* lp_mathlib_Mathlib_Tactic_tacticTry__this_____00__closed__14 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_tacticTry__this_____00__closed__14_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_tacticTry__this_____00__closed__15_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_tacticTry__this_____00__closed__14_value),LEAN_SCALAR_PTR_LITERAL(255, 188, 142, 1, 190, 33, 34, 128)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_tacticTry__this_____00__closed__15 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_tacticTry__this_____00__closed__15_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_tacticTry__this_____00__closed__16_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_tacticTry__this_____00__closed__15_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_tacticTry__this_____00__closed__16 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_tacticTry__this_____00__closed__16_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_tacticTry__this_____00__closed__17_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_tacticTry__this_____00__closed__13_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_tacticTry__this_____00__closed__16_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_tacticTry__this_____00__closed__17 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_tacticTry__this_____00__closed__17_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_tacticTry__this_____00__closed__18_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_tacticTry__this_____00__closed__5_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_tacticTry__this_____00__closed__11_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_tacticTry__this_____00__closed__17_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_tacticTry__this_____00__closed__18 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_tacticTry__this_____00__closed__18_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_tacticTry__this_____00__closed__19_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 3}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_tacticTry__this_____00__closed__3_value),((lean_object*)(((size_t)(1022) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_tacticTry__this_____00__closed__18_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_tacticTry__this_____00__closed__19 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_tacticTry__this_____00__closed__19_value;
LEAN_EXPORT const lean_object* lp_mathlib_Mathlib_Tactic_tacticTry__this____ = (const lean_object*)&lp_mathlib_Mathlib_Tactic_tacticTry__this_____00__closed__19_value;
static lean_once_cell_t lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic___aux__Mathlib__Tactic__TryThis______elabRules__Mathlib__Tactic__tacticTry__this______1_spec__0___redArg___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic___aux__Mathlib__Tactic__TryThis______elabRules__Mathlib__Tactic__tacticTry__this______1_spec__0___redArg___closed__0;
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic___aux__Mathlib__Tactic__TryThis______elabRules__Mathlib__Tactic__tacticTry__this______1_spec__0___redArg();
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic___aux__Mathlib__Tactic__TryThis______elabRules__Mathlib__Tactic__tacticTry__this______1_spec__0___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic___aux__Mathlib__Tactic__TryThis______elabRules__Mathlib__Tactic__tacticTry__this______1_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic___aux__Mathlib__Tactic__TryThis______elabRules__Mathlib__Tactic__tacticTry__this______1_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__TryThis______elabRules__Mathlib__Tactic__tacticTry__this______1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 9, .m_data = "Try this:"};
static const lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__TryThis______elabRules__Mathlib__Tactic__tacticTry__this______1___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__TryThis______elabRules__Mathlib__Tactic__tacticTry__this______1___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__TryThis______elabRules__Mathlib__Tactic__tacticTry__this______1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__TryThis______elabRules__Mathlib__Tactic__tacticTry__this______1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Mathlib_Tactic_convTry__this_____00__closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 15, .m_capacity = 15, .m_length = 14, .m_data = "convTry_this__"};
static const lean_object* lp_mathlib_Mathlib_Tactic_convTry__this_____00__closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_convTry__this_____00__closed__0_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_convTry__this_____00__closed__1_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_tacticTry__this_____00__closed__0_value),LEAN_SCALAR_PTR_LITERAL(118, 213, 161, 2, 73, 184, 31, 228)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_convTry__this_____00__closed__1_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_convTry__this_____00__closed__1_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic_tacticTry__this_____00__closed__1_value),LEAN_SCALAR_PTR_LITERAL(139, 222, 98, 232, 116, 132, 69, 249)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_convTry__this_____00__closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_convTry__this_____00__closed__1_value_aux_1),((lean_object*)&lp_mathlib_Mathlib_Tactic_convTry__this_____00__closed__0_value),LEAN_SCALAR_PTR_LITERAL(0, 180, 235, 220, 39, 148, 50, 68)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_convTry__this_____00__closed__1 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_convTry__this_____00__closed__1_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_convTry__this_____00__closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "conv"};
static const lean_object* lp_mathlib_Mathlib_Tactic_convTry__this_____00__closed__2 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_convTry__this_____00__closed__2_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_convTry__this_____00__closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_convTry__this_____00__closed__2_value),LEAN_SCALAR_PTR_LITERAL(232, 67, 39, 189, 45, 247, 54, 81)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_convTry__this_____00__closed__3 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_convTry__this_____00__closed__3_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_convTry__this_____00__closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 7}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_convTry__this_____00__closed__3_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_Mathlib_Tactic_convTry__this_____00__closed__4 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_convTry__this_____00__closed__4_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_convTry__this_____00__closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_tacticTry__this_____00__closed__5_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_tacticTry__this_____00__closed__7_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_convTry__this_____00__closed__4_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_convTry__this_____00__closed__5 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_convTry__this_____00__closed__5_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_convTry__this_____00__closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_tacticTry__this_____00__closed__5_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_convTry__this_____00__closed__5_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_tacticTry__this_____00__closed__17_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_convTry__this_____00__closed__6 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_convTry__this_____00__closed__6_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_convTry__this_____00__closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 3}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_convTry__this_____00__closed__1_value),((lean_object*)(((size_t)(1022) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_convTry__this_____00__closed__6_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_convTry__this_____00__closed__7 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_convTry__this_____00__closed__7_value;
LEAN_EXPORT const lean_object* lp_mathlib_Mathlib_Tactic_convTry__this____ = (const lean_object*)&lp_mathlib_Mathlib_Tactic_convTry__this_____00__closed__7_value;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__TryThis______elabRules__Mathlib__Tactic__convTry__this______1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__TryThis______elabRules__Mathlib__Tactic__convTry__this______1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_object* _init_lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic___aux__Mathlib__Tactic__TryThis______elabRules__Mathlib__Tactic__tacticTry__this______1_spec__0___redArg___closed__0(void){
_start:
{
lean_object* v___x_45_; lean_object* v___x_46_; lean_object* v___x_47_; 
v___x_45_ = lean_box(0);
v___x_46_ = l_Lean_Elab_unsupportedSyntaxExceptionId;
v___x_47_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_47_, 0, v___x_46_);
lean_ctor_set(v___x_47_, 1, v___x_45_);
return v___x_47_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic___aux__Mathlib__Tactic__TryThis______elabRules__Mathlib__Tactic__tacticTry__this______1_spec__0___redArg(){
_start:
{
lean_object* v___x_49_; lean_object* v___x_50_; 
v___x_49_ = lean_obj_once(&lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic___aux__Mathlib__Tactic__TryThis______elabRules__Mathlib__Tactic__tacticTry__this______1_spec__0___redArg___closed__0, &lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic___aux__Mathlib__Tactic__TryThis______elabRules__Mathlib__Tactic__tacticTry__this______1_spec__0___redArg___closed__0_once, _init_lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic___aux__Mathlib__Tactic__TryThis______elabRules__Mathlib__Tactic__tacticTry__this______1_spec__0___redArg___closed__0);
v___x_50_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_50_, 0, v___x_49_);
return v___x_50_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic___aux__Mathlib__Tactic__TryThis______elabRules__Mathlib__Tactic__tacticTry__this______1_spec__0___redArg___boxed(lean_object* v___y_51_){
_start:
{
lean_object* v_res_52_; 
v_res_52_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic___aux__Mathlib__Tactic__TryThis______elabRules__Mathlib__Tactic__tacticTry__this______1_spec__0___redArg();
return v_res_52_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic___aux__Mathlib__Tactic__TryThis______elabRules__Mathlib__Tactic__tacticTry__this______1_spec__0(lean_object* v_00_u03b1_53_, lean_object* v___y_54_, lean_object* v___y_55_, lean_object* v___y_56_, lean_object* v___y_57_, lean_object* v___y_58_, lean_object* v___y_59_, lean_object* v___y_60_, lean_object* v___y_61_){
_start:
{
lean_object* v___x_63_; 
v___x_63_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic___aux__Mathlib__Tactic__TryThis______elabRules__Mathlib__Tactic__tacticTry__this______1_spec__0___redArg();
return v___x_63_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic___aux__Mathlib__Tactic__TryThis______elabRules__Mathlib__Tactic__tacticTry__this______1_spec__0___boxed(lean_object* v_00_u03b1_64_, lean_object* v___y_65_, lean_object* v___y_66_, lean_object* v___y_67_, lean_object* v___y_68_, lean_object* v___y_69_, lean_object* v___y_70_, lean_object* v___y_71_, lean_object* v___y_72_, lean_object* v___y_73_){
_start:
{
lean_object* v_res_74_; 
v_res_74_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic___aux__Mathlib__Tactic__TryThis______elabRules__Mathlib__Tactic__tacticTry__this______1_spec__0(v_00_u03b1_64_, v___y_65_, v___y_66_, v___y_67_, v___y_68_, v___y_69_, v___y_70_, v___y_71_, v___y_72_);
lean_dec(v___y_72_);
lean_dec_ref(v___y_71_);
lean_dec(v___y_70_);
lean_dec_ref(v___y_69_);
lean_dec(v___y_68_);
lean_dec_ref(v___y_67_);
lean_dec(v___y_66_);
lean_dec_ref(v___y_65_);
return v_res_74_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__TryThis______elabRules__Mathlib__Tactic__tacticTry__this______1(lean_object* v_x_76_, lean_object* v_a_77_, lean_object* v_a_78_, lean_object* v_a_79_, lean_object* v_a_80_, lean_object* v_a_81_, lean_object* v_a_82_, lean_object* v_a_83_, lean_object* v_a_84_){
_start:
{
lean_object* v___x_86_; uint8_t v___x_87_; 
v___x_86_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_tacticTry__this_____00__closed__3));
lean_inc(v_x_76_);
v___x_87_ = l_Lean_Syntax_isOfKind(v_x_76_, v___x_86_);
if (v___x_87_ == 0)
{
lean_object* v___x_88_; 
lean_dec(v_x_76_);
v___x_88_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic___aux__Mathlib__Tactic__TryThis______elabRules__Mathlib__Tactic__tacticTry__this______1_spec__0___redArg();
return v___x_88_;
}
else
{
lean_object* v___x_89_; lean_object* v_tk_90_; lean_object* v___y_92_; lean_object* v___y_93_; lean_object* v___y_94_; lean_object* v___y_95_; lean_object* v___x_103_; lean_object* v_tac_104_; lean_object* v___y_106_; lean_object* v___x_121_; lean_object* v___x_122_; lean_object* v___x_123_; 
v___x_89_ = lean_unsigned_to_nat(0u);
v_tk_90_ = l_Lean_Syntax_getArg(v_x_76_, v___x_89_);
v___x_103_ = lean_unsigned_to_nat(1u);
v_tac_104_ = l_Lean_Syntax_getArg(v_x_76_, v___x_103_);
v___x_121_ = lean_unsigned_to_nat(2u);
v___x_122_ = l_Lean_Syntax_getArg(v_x_76_, v___x_121_);
lean_dec(v_x_76_);
v___x_123_ = l_Lean_Syntax_getOptional_x3f(v___x_122_);
lean_dec(v___x_122_);
if (lean_obj_tag(v___x_123_) == 0)
{
lean_object* v___x_124_; 
v___x_124_ = lean_box(0);
v___y_106_ = v___x_124_;
goto v___jp_105_;
}
else
{
lean_object* v_val_125_; lean_object* v___x_127_; uint8_t v_isShared_128_; uint8_t v_isSharedCheck_132_; 
v_val_125_ = lean_ctor_get(v___x_123_, 0);
v_isSharedCheck_132_ = !lean_is_exclusive(v___x_123_);
if (v_isSharedCheck_132_ == 0)
{
v___x_127_ = v___x_123_;
v_isShared_128_ = v_isSharedCheck_132_;
goto v_resetjp_126_;
}
else
{
lean_inc(v_val_125_);
lean_dec(v___x_123_);
v___x_127_ = lean_box(0);
v_isShared_128_ = v_isSharedCheck_132_;
goto v_resetjp_126_;
}
v_resetjp_126_:
{
lean_object* v___x_130_; 
if (v_isShared_128_ == 0)
{
v___x_130_ = v___x_127_;
goto v_reusejp_129_;
}
else
{
lean_object* v_reuseFailAlloc_131_; 
v_reuseFailAlloc_131_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_131_, 0, v_val_125_);
v___x_130_ = v_reuseFailAlloc_131_;
goto v_reusejp_129_;
}
v_reusejp_129_:
{
v___y_106_ = v___x_130_;
goto v___jp_105_;
}
}
}
v___jp_91_:
{
lean_object* v___x_96_; lean_object* v___x_97_; lean_object* v___x_98_; lean_object* v___x_99_; uint8_t v___x_100_; lean_object* v___x_101_; lean_object* v___x_102_; 
v___x_96_ = lean_box(0);
lean_inc(v___y_93_);
v___x_97_ = lean_alloc_ctor(0, 6, 0);
lean_ctor_set(v___x_97_, 0, v___y_92_);
lean_ctor_set(v___x_97_, 1, v___y_93_);
lean_ctor_set(v___x_97_, 2, v___y_95_);
lean_ctor_set(v___x_97_, 3, v___x_96_);
lean_ctor_set(v___x_97_, 4, v___x_96_);
lean_ctor_set(v___x_97_, 5, v___x_96_);
lean_inc(v___y_94_);
v___x_98_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_98_, 0, v___y_94_);
v___x_99_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__TryThis______elabRules__Mathlib__Tactic__tacticTry__this______1___closed__0));
v___x_100_ = 4;
v___x_101_ = l_Lean_MessageData_nil;
v___x_102_ = l_Lean_Meta_Tactic_TryThis_addSuggestion(v_tk_90_, v___x_97_, v___x_98_, v___x_99_, v___y_93_, v___x_100_, v___x_101_, v_a_83_, v_a_84_);
return v___x_102_;
}
v___jp_105_:
{
lean_object* v___x_107_; 
lean_inc(v_tac_104_);
v___x_107_ = l_Lean_Elab_Tactic_evalTactic(v_tac_104_, v_a_77_, v_a_78_, v_a_79_, v_a_80_, v_a_81_, v_a_82_, v_a_83_, v_a_84_);
if (lean_obj_tag(v___x_107_) == 0)
{
lean_object* v_ref_108_; lean_object* v___x_109_; lean_object* v___x_110_; lean_object* v___x_111_; 
lean_dec_ref_known(v___x_107_, 1);
v_ref_108_ = lean_ctor_get(v_a_83_, 5);
v___x_109_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_tacticTry__this_____00__closed__9));
v___x_110_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_110_, 0, v___x_109_);
lean_ctor_set(v___x_110_, 1, v_tac_104_);
v___x_111_ = lean_box(0);
if (lean_obj_tag(v___y_106_) == 0)
{
v___y_92_ = v___x_110_;
v___y_93_ = v___x_111_;
v___y_94_ = v_ref_108_;
v___y_95_ = v___x_111_;
goto v___jp_91_;
}
else
{
lean_object* v_val_112_; lean_object* v___x_114_; uint8_t v_isShared_115_; uint8_t v_isSharedCheck_120_; 
v_val_112_ = lean_ctor_get(v___y_106_, 0);
v_isSharedCheck_120_ = !lean_is_exclusive(v___y_106_);
if (v_isSharedCheck_120_ == 0)
{
v___x_114_ = v___y_106_;
v_isShared_115_ = v_isSharedCheck_120_;
goto v_resetjp_113_;
}
else
{
lean_inc(v_val_112_);
lean_dec(v___y_106_);
v___x_114_ = lean_box(0);
v_isShared_115_ = v_isSharedCheck_120_;
goto v_resetjp_113_;
}
v_resetjp_113_:
{
lean_object* v___x_116_; lean_object* v___x_118_; 
v___x_116_ = l_Lean_TSyntax_getString(v_val_112_);
lean_dec(v_val_112_);
if (v_isShared_115_ == 0)
{
lean_ctor_set(v___x_114_, 0, v___x_116_);
v___x_118_ = v___x_114_;
goto v_reusejp_117_;
}
else
{
lean_object* v_reuseFailAlloc_119_; 
v_reuseFailAlloc_119_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_119_, 0, v___x_116_);
v___x_118_ = v_reuseFailAlloc_119_;
goto v_reusejp_117_;
}
v_reusejp_117_:
{
v___y_92_ = v___x_110_;
v___y_93_ = v___x_111_;
v___y_94_ = v_ref_108_;
v___y_95_ = v___x_118_;
goto v___jp_91_;
}
}
}
}
else
{
lean_dec(v___y_106_);
lean_dec(v_tac_104_);
lean_dec(v_tk_90_);
return v___x_107_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__TryThis______elabRules__Mathlib__Tactic__tacticTry__this______1___boxed(lean_object* v_x_133_, lean_object* v_a_134_, lean_object* v_a_135_, lean_object* v_a_136_, lean_object* v_a_137_, lean_object* v_a_138_, lean_object* v_a_139_, lean_object* v_a_140_, lean_object* v_a_141_, lean_object* v_a_142_){
_start:
{
lean_object* v_res_143_; 
v_res_143_ = lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__TryThis______elabRules__Mathlib__Tactic__tacticTry__this______1(v_x_133_, v_a_134_, v_a_135_, v_a_136_, v_a_137_, v_a_138_, v_a_139_, v_a_140_, v_a_141_);
lean_dec(v_a_141_);
lean_dec_ref(v_a_140_);
lean_dec(v_a_139_);
lean_dec_ref(v_a_138_);
lean_dec(v_a_137_);
lean_dec_ref(v_a_136_);
lean_dec(v_a_135_);
lean_dec_ref(v_a_134_);
return v_res_143_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__TryThis______elabRules__Mathlib__Tactic__convTry__this______1(lean_object* v_x_168_, lean_object* v_a_169_, lean_object* v_a_170_, lean_object* v_a_171_, lean_object* v_a_172_, lean_object* v_a_173_, lean_object* v_a_174_, lean_object* v_a_175_, lean_object* v_a_176_){
_start:
{
lean_object* v___x_178_; uint8_t v___x_179_; 
v___x_178_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_convTry__this_____00__closed__1));
lean_inc(v_x_168_);
v___x_179_ = l_Lean_Syntax_isOfKind(v_x_168_, v___x_178_);
if (v___x_179_ == 0)
{
lean_object* v___x_180_; 
lean_dec(v_x_168_);
v___x_180_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic___aux__Mathlib__Tactic__TryThis______elabRules__Mathlib__Tactic__tacticTry__this______1_spec__0___redArg();
return v___x_180_;
}
else
{
lean_object* v___x_181_; lean_object* v_tk_182_; lean_object* v___y_184_; lean_object* v___y_185_; lean_object* v___y_186_; lean_object* v___y_187_; lean_object* v___x_195_; lean_object* v_tac_196_; lean_object* v___y_198_; lean_object* v___x_213_; lean_object* v___x_214_; lean_object* v___x_215_; 
v___x_181_ = lean_unsigned_to_nat(0u);
v_tk_182_ = l_Lean_Syntax_getArg(v_x_168_, v___x_181_);
v___x_195_ = lean_unsigned_to_nat(1u);
v_tac_196_ = l_Lean_Syntax_getArg(v_x_168_, v___x_195_);
v___x_213_ = lean_unsigned_to_nat(2u);
v___x_214_ = l_Lean_Syntax_getArg(v_x_168_, v___x_213_);
lean_dec(v_x_168_);
v___x_215_ = l_Lean_Syntax_getOptional_x3f(v___x_214_);
lean_dec(v___x_214_);
if (lean_obj_tag(v___x_215_) == 0)
{
lean_object* v___x_216_; 
v___x_216_ = lean_box(0);
v___y_198_ = v___x_216_;
goto v___jp_197_;
}
else
{
lean_object* v_val_217_; lean_object* v___x_219_; uint8_t v_isShared_220_; uint8_t v_isSharedCheck_224_; 
v_val_217_ = lean_ctor_get(v___x_215_, 0);
v_isSharedCheck_224_ = !lean_is_exclusive(v___x_215_);
if (v_isSharedCheck_224_ == 0)
{
v___x_219_ = v___x_215_;
v_isShared_220_ = v_isSharedCheck_224_;
goto v_resetjp_218_;
}
else
{
lean_inc(v_val_217_);
lean_dec(v___x_215_);
v___x_219_ = lean_box(0);
v_isShared_220_ = v_isSharedCheck_224_;
goto v_resetjp_218_;
}
v_resetjp_218_:
{
lean_object* v___x_222_; 
if (v_isShared_220_ == 0)
{
v___x_222_ = v___x_219_;
goto v_reusejp_221_;
}
else
{
lean_object* v_reuseFailAlloc_223_; 
v_reuseFailAlloc_223_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_223_, 0, v_val_217_);
v___x_222_ = v_reuseFailAlloc_223_;
goto v_reusejp_221_;
}
v_reusejp_221_:
{
v___y_198_ = v___x_222_;
goto v___jp_197_;
}
}
}
v___jp_183_:
{
lean_object* v___x_188_; lean_object* v___x_189_; lean_object* v___x_190_; lean_object* v___x_191_; uint8_t v___x_192_; lean_object* v___x_193_; lean_object* v___x_194_; 
v___x_188_ = lean_box(0);
lean_inc(v___y_185_);
v___x_189_ = lean_alloc_ctor(0, 6, 0);
lean_ctor_set(v___x_189_, 0, v___y_186_);
lean_ctor_set(v___x_189_, 1, v___y_185_);
lean_ctor_set(v___x_189_, 2, v___y_187_);
lean_ctor_set(v___x_189_, 3, v___x_188_);
lean_ctor_set(v___x_189_, 4, v___x_188_);
lean_ctor_set(v___x_189_, 5, v___x_188_);
lean_inc(v___y_184_);
v___x_190_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_190_, 0, v___y_184_);
v___x_191_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__TryThis______elabRules__Mathlib__Tactic__tacticTry__this______1___closed__0));
v___x_192_ = 4;
v___x_193_ = l_Lean_MessageData_nil;
v___x_194_ = l_Lean_Meta_Tactic_TryThis_addSuggestion(v_tk_182_, v___x_189_, v___x_190_, v___x_191_, v___y_185_, v___x_192_, v___x_193_, v_a_175_, v_a_176_);
return v___x_194_;
}
v___jp_197_:
{
lean_object* v___x_199_; 
lean_inc(v_tac_196_);
v___x_199_ = l_Lean_Elab_Tactic_evalTactic(v_tac_196_, v_a_169_, v_a_170_, v_a_171_, v_a_172_, v_a_173_, v_a_174_, v_a_175_, v_a_176_);
if (lean_obj_tag(v___x_199_) == 0)
{
lean_object* v_ref_200_; lean_object* v___x_201_; lean_object* v___x_202_; lean_object* v___x_203_; 
lean_dec_ref_known(v___x_199_, 1);
v_ref_200_ = lean_ctor_get(v_a_175_, 5);
v___x_201_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_convTry__this_____00__closed__3));
v___x_202_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_202_, 0, v___x_201_);
lean_ctor_set(v___x_202_, 1, v_tac_196_);
v___x_203_ = lean_box(0);
if (lean_obj_tag(v___y_198_) == 0)
{
v___y_184_ = v_ref_200_;
v___y_185_ = v___x_203_;
v___y_186_ = v___x_202_;
v___y_187_ = v___x_203_;
goto v___jp_183_;
}
else
{
lean_object* v_val_204_; lean_object* v___x_206_; uint8_t v_isShared_207_; uint8_t v_isSharedCheck_212_; 
v_val_204_ = lean_ctor_get(v___y_198_, 0);
v_isSharedCheck_212_ = !lean_is_exclusive(v___y_198_);
if (v_isSharedCheck_212_ == 0)
{
v___x_206_ = v___y_198_;
v_isShared_207_ = v_isSharedCheck_212_;
goto v_resetjp_205_;
}
else
{
lean_inc(v_val_204_);
lean_dec(v___y_198_);
v___x_206_ = lean_box(0);
v_isShared_207_ = v_isSharedCheck_212_;
goto v_resetjp_205_;
}
v_resetjp_205_:
{
lean_object* v___x_208_; lean_object* v___x_210_; 
v___x_208_ = l_Lean_TSyntax_getString(v_val_204_);
lean_dec(v_val_204_);
if (v_isShared_207_ == 0)
{
lean_ctor_set(v___x_206_, 0, v___x_208_);
v___x_210_ = v___x_206_;
goto v_reusejp_209_;
}
else
{
lean_object* v_reuseFailAlloc_211_; 
v_reuseFailAlloc_211_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_211_, 0, v___x_208_);
v___x_210_ = v_reuseFailAlloc_211_;
goto v_reusejp_209_;
}
v_reusejp_209_:
{
v___y_184_ = v_ref_200_;
v___y_185_ = v___x_203_;
v___y_186_ = v___x_202_;
v___y_187_ = v___x_210_;
goto v___jp_183_;
}
}
}
}
else
{
lean_dec(v___y_198_);
lean_dec(v_tac_196_);
lean_dec(v_tk_182_);
return v___x_199_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__TryThis______elabRules__Mathlib__Tactic__convTry__this______1___boxed(lean_object* v_x_225_, lean_object* v_a_226_, lean_object* v_a_227_, lean_object* v_a_228_, lean_object* v_a_229_, lean_object* v_a_230_, lean_object* v_a_231_, lean_object* v_a_232_, lean_object* v_a_233_, lean_object* v_a_234_){
_start:
{
lean_object* v_res_235_; 
v_res_235_ = lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__TryThis______elabRules__Mathlib__Tactic__convTry__this______1(v_x_225_, v_a_226_, v_a_227_, v_a_228_, v_a_229_, v_a_230_, v_a_231_, v_a_232_, v_a_233_);
lean_dec(v_a_233_);
lean_dec_ref(v_a_232_);
lean_dec(v_a_231_);
lean_dec_ref(v_a_230_);
lean_dec(v_a_229_);
lean_dec_ref(v_a_228_);
lean_dec(v_a_227_);
lean_dec_ref(v_a_226_);
return v_res_235_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Init(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_Tactic_TryThis(uint8_t builtin) {
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
lean_object* runtime_initialize_Lean_Meta_Tactic_TryThis(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_Tactic_TryThis(uint8_t builtin) {
lean_object * res;
if (_G_meta_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_meta_initialized = true;
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_Lean_Meta_Tactic_TryThis(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Init(uint8_t builtin);
lean_object* initialize_Lean_Meta_Tactic_TryThis(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Tactic_TryThis(uint8_t builtin) {
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
res = initialize_Lean_Meta_Tactic_TryThis(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Tactic_TryThis(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_Tactic_TryThis(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_Tactic_TryThis(builtin);
}
#ifdef __cplusplus
}
#endif
