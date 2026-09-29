// Lean compiler output
// Module: Mathlib.GroupTheory.GroupAction.DomAct.Basic
// Imports: public import Init public meta import Init public import Mathlib.Algebra.Group.Action.Basic public import Mathlib.Algebra.Group.Opposite public import Mathlib.Algebra.Group.Pi.Lemmas public import Mathlib.Algebra.GroupWithZero.Action.Hom public import Mathlib.Algebra.Ring.Defs public meta import Mathlib.Tactic.ToDual
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
lean_object* lp_mathlib_MulOpposite_opEquiv(lean_object*);
lean_object* lp_mathlib_Equiv_symm___redArg(lean_object*);
lean_object* lp_mathlib_MulDistribMulAction_toMonoidHom___redArg___lam__0(lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_OneHom_comp___redArg___lam__0(lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Name_mkStr1(lean_object*);
lean_object* lp_mathlib_SMulZeroClass_toZeroHom___redArg___lam__0(lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_AddOpposite_opEquiv(lean_object*);
lean_object* l_Lean_Name_mkStr4(lean_object*, lean_object*, lean_object*, lean_object*);
uint8_t l_Lean_Syntax_isOfKind(lean_object*, lean_object*);
lean_object* l_Lean_Syntax_getArg(lean_object*, lean_object*);
uint8_t l_Lean_Syntax_matchesNull(lean_object*, lean_object*);
lean_object* l_Lean_replaceRef(lean_object*, lean_object*);
lean_object* l_Lean_SourceInfo_fromRef(lean_object*, uint8_t);
lean_object* l_Lean_Syntax_node2(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_String_toRawSubstring_x27(lean_object*);
lean_object* l_Lean_addMacroScope(lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Syntax_node1(lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_term___u1d48_u1d50_u1d43___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 15, .m_capacity = 15, .m_length = 8, .m_data = "term_ᵈᵐᵃ"};
static const lean_object* lp_mathlib_term___u1d48_u1d50_u1d43___closed__0 = (const lean_object*)&lp_mathlib_term___u1d48_u1d50_u1d43___closed__0_value;
static const lean_ctor_object lp_mathlib_term___u1d48_u1d50_u1d43___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_term___u1d48_u1d50_u1d43___closed__0_value),LEAN_SCALAR_PTR_LITERAL(212, 170, 147, 202, 167, 68, 33, 198)}};
static const lean_object* lp_mathlib_term___u1d48_u1d50_u1d43___closed__1 = (const lean_object*)&lp_mathlib_term___u1d48_u1d50_u1d43___closed__1_value;
static const lean_string_object lp_mathlib_term___u1d48_u1d50_u1d43___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 3, .m_data = "ᵈᵐᵃ"};
static const lean_object* lp_mathlib_term___u1d48_u1d50_u1d43___closed__2 = (const lean_object*)&lp_mathlib_term___u1d48_u1d50_u1d43___closed__2_value;
static const lean_ctor_object lp_mathlib_term___u1d48_u1d50_u1d43___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 5}, .m_objs = {((lean_object*)&lp_mathlib_term___u1d48_u1d50_u1d43___closed__2_value)}};
static const lean_object* lp_mathlib_term___u1d48_u1d50_u1d43___closed__3 = (const lean_object*)&lp_mathlib_term___u1d48_u1d50_u1d43___closed__3_value;
static const lean_ctor_object lp_mathlib_term___u1d48_u1d50_u1d43___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*4 + 0, .m_other = 4, .m_tag = 4}, .m_objs = {((lean_object*)&lp_mathlib_term___u1d48_u1d50_u1d43___closed__1_value),((lean_object*)(((size_t)(1024) << 1) | 1)),((lean_object*)(((size_t)(1024) << 1) | 1)),((lean_object*)&lp_mathlib_term___u1d48_u1d50_u1d43___closed__3_value)}};
static const lean_object* lp_mathlib_term___u1d48_u1d50_u1d43___closed__4 = (const lean_object*)&lp_mathlib_term___u1d48_u1d50_u1d43___closed__4_value;
LEAN_EXPORT const lean_object* lp_mathlib_term___u1d48_u1d50_u1d43 = (const lean_object*)&lp_mathlib_term___u1d48_u1d50_u1d43___closed__4_value;
static const lean_string_object lp_mathlib___aux__Mathlib__GroupTheory__GroupAction__DomAct__Basic______macroRules__term___u1d48_u1d50_u1d43__1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "Lean"};
static const lean_object* lp_mathlib___aux__Mathlib__GroupTheory__GroupAction__DomAct__Basic______macroRules__term___u1d48_u1d50_u1d43__1___closed__0 = (const lean_object*)&lp_mathlib___aux__Mathlib__GroupTheory__GroupAction__DomAct__Basic______macroRules__term___u1d48_u1d50_u1d43__1___closed__0_value;
static const lean_string_object lp_mathlib___aux__Mathlib__GroupTheory__GroupAction__DomAct__Basic______macroRules__term___u1d48_u1d50_u1d43__1___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "Parser"};
static const lean_object* lp_mathlib___aux__Mathlib__GroupTheory__GroupAction__DomAct__Basic______macroRules__term___u1d48_u1d50_u1d43__1___closed__1 = (const lean_object*)&lp_mathlib___aux__Mathlib__GroupTheory__GroupAction__DomAct__Basic______macroRules__term___u1d48_u1d50_u1d43__1___closed__1_value;
static const lean_string_object lp_mathlib___aux__Mathlib__GroupTheory__GroupAction__DomAct__Basic______macroRules__term___u1d48_u1d50_u1d43__1___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "Term"};
static const lean_object* lp_mathlib___aux__Mathlib__GroupTheory__GroupAction__DomAct__Basic______macroRules__term___u1d48_u1d50_u1d43__1___closed__2 = (const lean_object*)&lp_mathlib___aux__Mathlib__GroupTheory__GroupAction__DomAct__Basic______macroRules__term___u1d48_u1d50_u1d43__1___closed__2_value;
static const lean_string_object lp_mathlib___aux__Mathlib__GroupTheory__GroupAction__DomAct__Basic______macroRules__term___u1d48_u1d50_u1d43__1___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = "app"};
static const lean_object* lp_mathlib___aux__Mathlib__GroupTheory__GroupAction__DomAct__Basic______macroRules__term___u1d48_u1d50_u1d43__1___closed__3 = (const lean_object*)&lp_mathlib___aux__Mathlib__GroupTheory__GroupAction__DomAct__Basic______macroRules__term___u1d48_u1d50_u1d43__1___closed__3_value;
static const lean_ctor_object lp_mathlib___aux__Mathlib__GroupTheory__GroupAction__DomAct__Basic______macroRules__term___u1d48_u1d50_u1d43__1___closed__4_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___aux__Mathlib__GroupTheory__GroupAction__DomAct__Basic______macroRules__term___u1d48_u1d50_u1d43__1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib___aux__Mathlib__GroupTheory__GroupAction__DomAct__Basic______macroRules__term___u1d48_u1d50_u1d43__1___closed__4_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___aux__Mathlib__GroupTheory__GroupAction__DomAct__Basic______macroRules__term___u1d48_u1d50_u1d43__1___closed__4_value_aux_0),((lean_object*)&lp_mathlib___aux__Mathlib__GroupTheory__GroupAction__DomAct__Basic______macroRules__term___u1d48_u1d50_u1d43__1___closed__1_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib___aux__Mathlib__GroupTheory__GroupAction__DomAct__Basic______macroRules__term___u1d48_u1d50_u1d43__1___closed__4_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___aux__Mathlib__GroupTheory__GroupAction__DomAct__Basic______macroRules__term___u1d48_u1d50_u1d43__1___closed__4_value_aux_1),((lean_object*)&lp_mathlib___aux__Mathlib__GroupTheory__GroupAction__DomAct__Basic______macroRules__term___u1d48_u1d50_u1d43__1___closed__2_value),LEAN_SCALAR_PTR_LITERAL(75, 170, 162, 138, 136, 204, 251, 229)}};
static const lean_ctor_object lp_mathlib___aux__Mathlib__GroupTheory__GroupAction__DomAct__Basic______macroRules__term___u1d48_u1d50_u1d43__1___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___aux__Mathlib__GroupTheory__GroupAction__DomAct__Basic______macroRules__term___u1d48_u1d50_u1d43__1___closed__4_value_aux_2),((lean_object*)&lp_mathlib___aux__Mathlib__GroupTheory__GroupAction__DomAct__Basic______macroRules__term___u1d48_u1d50_u1d43__1___closed__3_value),LEAN_SCALAR_PTR_LITERAL(69, 118, 10, 41, 220, 156, 243, 179)}};
static const lean_object* lp_mathlib___aux__Mathlib__GroupTheory__GroupAction__DomAct__Basic______macroRules__term___u1d48_u1d50_u1d43__1___closed__4 = (const lean_object*)&lp_mathlib___aux__Mathlib__GroupTheory__GroupAction__DomAct__Basic______macroRules__term___u1d48_u1d50_u1d43__1___closed__4_value;
static const lean_string_object lp_mathlib___aux__Mathlib__GroupTheory__GroupAction__DomAct__Basic______macroRules__term___u1d48_u1d50_u1d43__1___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 9, .m_data = "DomMulAct"};
static const lean_object* lp_mathlib___aux__Mathlib__GroupTheory__GroupAction__DomAct__Basic______macroRules__term___u1d48_u1d50_u1d43__1___closed__5 = (const lean_object*)&lp_mathlib___aux__Mathlib__GroupTheory__GroupAction__DomAct__Basic______macroRules__term___u1d48_u1d50_u1d43__1___closed__5_value;
static lean_once_cell_t lp_mathlib___aux__Mathlib__GroupTheory__GroupAction__DomAct__Basic______macroRules__term___u1d48_u1d50_u1d43__1___closed__6_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___aux__Mathlib__GroupTheory__GroupAction__DomAct__Basic______macroRules__term___u1d48_u1d50_u1d43__1___closed__6;
static const lean_ctor_object lp_mathlib___aux__Mathlib__GroupTheory__GroupAction__DomAct__Basic______macroRules__term___u1d48_u1d50_u1d43__1___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___aux__Mathlib__GroupTheory__GroupAction__DomAct__Basic______macroRules__term___u1d48_u1d50_u1d43__1___closed__5_value),LEAN_SCALAR_PTR_LITERAL(85, 162, 71, 164, 174, 221, 23, 7)}};
static const lean_object* lp_mathlib___aux__Mathlib__GroupTheory__GroupAction__DomAct__Basic______macroRules__term___u1d48_u1d50_u1d43__1___closed__7 = (const lean_object*)&lp_mathlib___aux__Mathlib__GroupTheory__GroupAction__DomAct__Basic______macroRules__term___u1d48_u1d50_u1d43__1___closed__7_value;
static const lean_ctor_object lp_mathlib___aux__Mathlib__GroupTheory__GroupAction__DomAct__Basic______macroRules__term___u1d48_u1d50_u1d43__1___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___aux__Mathlib__GroupTheory__GroupAction__DomAct__Basic______macroRules__term___u1d48_u1d50_u1d43__1___closed__7_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib___aux__Mathlib__GroupTheory__GroupAction__DomAct__Basic______macroRules__term___u1d48_u1d50_u1d43__1___closed__8 = (const lean_object*)&lp_mathlib___aux__Mathlib__GroupTheory__GroupAction__DomAct__Basic______macroRules__term___u1d48_u1d50_u1d43__1___closed__8_value;
static const lean_ctor_object lp_mathlib___aux__Mathlib__GroupTheory__GroupAction__DomAct__Basic______macroRules__term___u1d48_u1d50_u1d43__1___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___aux__Mathlib__GroupTheory__GroupAction__DomAct__Basic______macroRules__term___u1d48_u1d50_u1d43__1___closed__8_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib___aux__Mathlib__GroupTheory__GroupAction__DomAct__Basic______macroRules__term___u1d48_u1d50_u1d43__1___closed__9 = (const lean_object*)&lp_mathlib___aux__Mathlib__GroupTheory__GroupAction__DomAct__Basic______macroRules__term___u1d48_u1d50_u1d43__1___closed__9_value;
static const lean_string_object lp_mathlib___aux__Mathlib__GroupTheory__GroupAction__DomAct__Basic______macroRules__term___u1d48_u1d50_u1d43__1___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "null"};
static const lean_object* lp_mathlib___aux__Mathlib__GroupTheory__GroupAction__DomAct__Basic______macroRules__term___u1d48_u1d50_u1d43__1___closed__10 = (const lean_object*)&lp_mathlib___aux__Mathlib__GroupTheory__GroupAction__DomAct__Basic______macroRules__term___u1d48_u1d50_u1d43__1___closed__10_value;
static const lean_ctor_object lp_mathlib___aux__Mathlib__GroupTheory__GroupAction__DomAct__Basic______macroRules__term___u1d48_u1d50_u1d43__1___closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___aux__Mathlib__GroupTheory__GroupAction__DomAct__Basic______macroRules__term___u1d48_u1d50_u1d43__1___closed__10_value),LEAN_SCALAR_PTR_LITERAL(24, 58, 49, 223, 146, 207, 197, 136)}};
static const lean_object* lp_mathlib___aux__Mathlib__GroupTheory__GroupAction__DomAct__Basic______macroRules__term___u1d48_u1d50_u1d43__1___closed__11 = (const lean_object*)&lp_mathlib___aux__Mathlib__GroupTheory__GroupAction__DomAct__Basic______macroRules__term___u1d48_u1d50_u1d43__1___closed__11_value;
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__GroupTheory__GroupAction__DomAct__Basic______macroRules__term___u1d48_u1d50_u1d43__1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__GroupTheory__GroupAction__DomAct__Basic______macroRules__term___u1d48_u1d50_u1d43__1___boxed(lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib___aux__Mathlib__GroupTheory__GroupAction__DomAct__Basic______unexpand__DomMulAct__1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "ident"};
static const lean_object* lp_mathlib___aux__Mathlib__GroupTheory__GroupAction__DomAct__Basic______unexpand__DomMulAct__1___closed__0 = (const lean_object*)&lp_mathlib___aux__Mathlib__GroupTheory__GroupAction__DomAct__Basic______unexpand__DomMulAct__1___closed__0_value;
static const lean_ctor_object lp_mathlib___aux__Mathlib__GroupTheory__GroupAction__DomAct__Basic______unexpand__DomMulAct__1___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___aux__Mathlib__GroupTheory__GroupAction__DomAct__Basic______unexpand__DomMulAct__1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(52, 159, 208, 51, 14, 60, 6, 71)}};
static const lean_object* lp_mathlib___aux__Mathlib__GroupTheory__GroupAction__DomAct__Basic______unexpand__DomMulAct__1___closed__1 = (const lean_object*)&lp_mathlib___aux__Mathlib__GroupTheory__GroupAction__DomAct__Basic______unexpand__DomMulAct__1___closed__1_value;
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__GroupTheory__GroupAction__DomAct__Basic______unexpand__DomMulAct__1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__GroupTheory__GroupAction__DomAct__Basic______unexpand__DomMulAct__1___boxed(lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_term___u1d48_u1d43_u1d43___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 15, .m_capacity = 15, .m_length = 8, .m_data = "term_ᵈᵃᵃ"};
static const lean_object* lp_mathlib_term___u1d48_u1d43_u1d43___closed__0 = (const lean_object*)&lp_mathlib_term___u1d48_u1d43_u1d43___closed__0_value;
static const lean_ctor_object lp_mathlib_term___u1d48_u1d43_u1d43___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_term___u1d48_u1d43_u1d43___closed__0_value),LEAN_SCALAR_PTR_LITERAL(40, 90, 12, 128, 174, 117, 212, 135)}};
static const lean_object* lp_mathlib_term___u1d48_u1d43_u1d43___closed__1 = (const lean_object*)&lp_mathlib_term___u1d48_u1d43_u1d43___closed__1_value;
static const lean_string_object lp_mathlib_term___u1d48_u1d43_u1d43___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 3, .m_data = "ᵈᵃᵃ"};
static const lean_object* lp_mathlib_term___u1d48_u1d43_u1d43___closed__2 = (const lean_object*)&lp_mathlib_term___u1d48_u1d43_u1d43___closed__2_value;
static const lean_ctor_object lp_mathlib_term___u1d48_u1d43_u1d43___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 5}, .m_objs = {((lean_object*)&lp_mathlib_term___u1d48_u1d43_u1d43___closed__2_value)}};
static const lean_object* lp_mathlib_term___u1d48_u1d43_u1d43___closed__3 = (const lean_object*)&lp_mathlib_term___u1d48_u1d43_u1d43___closed__3_value;
static const lean_ctor_object lp_mathlib_term___u1d48_u1d43_u1d43___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*4 + 0, .m_other = 4, .m_tag = 4}, .m_objs = {((lean_object*)&lp_mathlib_term___u1d48_u1d43_u1d43___closed__1_value),((lean_object*)(((size_t)(1024) << 1) | 1)),((lean_object*)(((size_t)(1024) << 1) | 1)),((lean_object*)&lp_mathlib_term___u1d48_u1d43_u1d43___closed__3_value)}};
static const lean_object* lp_mathlib_term___u1d48_u1d43_u1d43___closed__4 = (const lean_object*)&lp_mathlib_term___u1d48_u1d43_u1d43___closed__4_value;
LEAN_EXPORT const lean_object* lp_mathlib_term___u1d48_u1d43_u1d43 = (const lean_object*)&lp_mathlib_term___u1d48_u1d43_u1d43___closed__4_value;
static const lean_string_object lp_mathlib___aux__Mathlib__GroupTheory__GroupAction__DomAct__Basic______macroRules__term___u1d48_u1d43_u1d43__1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 9, .m_data = "DomAddAct"};
static const lean_object* lp_mathlib___aux__Mathlib__GroupTheory__GroupAction__DomAct__Basic______macroRules__term___u1d48_u1d43_u1d43__1___closed__0 = (const lean_object*)&lp_mathlib___aux__Mathlib__GroupTheory__GroupAction__DomAct__Basic______macroRules__term___u1d48_u1d43_u1d43__1___closed__0_value;
static lean_once_cell_t lp_mathlib___aux__Mathlib__GroupTheory__GroupAction__DomAct__Basic______macroRules__term___u1d48_u1d43_u1d43__1___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___aux__Mathlib__GroupTheory__GroupAction__DomAct__Basic______macroRules__term___u1d48_u1d43_u1d43__1___closed__1;
static const lean_ctor_object lp_mathlib___aux__Mathlib__GroupTheory__GroupAction__DomAct__Basic______macroRules__term___u1d48_u1d43_u1d43__1___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___aux__Mathlib__GroupTheory__GroupAction__DomAct__Basic______macroRules__term___u1d48_u1d43_u1d43__1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(1, 99, 147, 127, 212, 9, 39, 94)}};
static const lean_object* lp_mathlib___aux__Mathlib__GroupTheory__GroupAction__DomAct__Basic______macroRules__term___u1d48_u1d43_u1d43__1___closed__2 = (const lean_object*)&lp_mathlib___aux__Mathlib__GroupTheory__GroupAction__DomAct__Basic______macroRules__term___u1d48_u1d43_u1d43__1___closed__2_value;
static const lean_ctor_object lp_mathlib___aux__Mathlib__GroupTheory__GroupAction__DomAct__Basic______macroRules__term___u1d48_u1d43_u1d43__1___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___aux__Mathlib__GroupTheory__GroupAction__DomAct__Basic______macroRules__term___u1d48_u1d43_u1d43__1___closed__2_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib___aux__Mathlib__GroupTheory__GroupAction__DomAct__Basic______macroRules__term___u1d48_u1d43_u1d43__1___closed__3 = (const lean_object*)&lp_mathlib___aux__Mathlib__GroupTheory__GroupAction__DomAct__Basic______macroRules__term___u1d48_u1d43_u1d43__1___closed__3_value;
static const lean_ctor_object lp_mathlib___aux__Mathlib__GroupTheory__GroupAction__DomAct__Basic______macroRules__term___u1d48_u1d43_u1d43__1___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___aux__Mathlib__GroupTheory__GroupAction__DomAct__Basic______macroRules__term___u1d48_u1d43_u1d43__1___closed__3_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib___aux__Mathlib__GroupTheory__GroupAction__DomAct__Basic______macroRules__term___u1d48_u1d43_u1d43__1___closed__4 = (const lean_object*)&lp_mathlib___aux__Mathlib__GroupTheory__GroupAction__DomAct__Basic______macroRules__term___u1d48_u1d43_u1d43__1___closed__4_value;
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__GroupTheory__GroupAction__DomAct__Basic______macroRules__term___u1d48_u1d43_u1d43__1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__GroupTheory__GroupAction__DomAct__Basic______macroRules__term___u1d48_u1d43_u1d43__1___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__GroupTheory__GroupAction__DomAct__Basic______unexpand__DomAddAct__1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__GroupTheory__GroupAction__DomAct__Basic______unexpand__DomAddAct__1___boxed(lean_object*, lean_object*, lean_object*);
static lean_once_cell_t lp_mathlib_DomMulAct_mk___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_DomMulAct_mk___closed__0;
LEAN_EXPORT lean_object* lp_mathlib_DomMulAct_mk(lean_object*);
static lean_once_cell_t lp_mathlib_DomAddAct_mk___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_DomAddAct_mk___closed__0;
LEAN_EXPORT lean_object* lp_mathlib_DomAddAct_mk(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DomMulAct_instMulOfMulOpposite___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DomMulAct_instMulOfMulOpposite___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DomMulAct_instMulOfMulOpposite(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DomMulAct_instMulOfMulOpposite___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DomAddAct_instAddOfAddOpposite___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DomAddAct_instAddOfAddOpposite___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DomAddAct_instAddOfAddOpposite(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DomAddAct_instAddOfAddOpposite___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DomMulAct_instOneOfMulOpposite___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DomMulAct_instOneOfMulOpposite___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DomMulAct_instOneOfMulOpposite(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DomMulAct_instOneOfMulOpposite___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DomAddAct_instZeroOfAddOpposite___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DomAddAct_instZeroOfAddOpposite___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DomAddAct_instZeroOfAddOpposite(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DomAddAct_instZeroOfAddOpposite___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DomMulAct_instInvOfMulOpposite___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DomMulAct_instInvOfMulOpposite___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DomMulAct_instInvOfMulOpposite(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DomMulAct_instInvOfMulOpposite___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DomAddAct_instNegOfAddOpposite___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DomAddAct_instNegOfAddOpposite___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DomAddAct_instNegOfAddOpposite(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DomAddAct_instNegOfAddOpposite___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DomMulAct_instSemigroupOfMulOpposite___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DomMulAct_instSemigroupOfMulOpposite___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DomMulAct_instSemigroupOfMulOpposite(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DomMulAct_instSemigroupOfMulOpposite___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DomAddAct_instAddSemigroupOfAddOpposite___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DomAddAct_instAddSemigroupOfAddOpposite___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DomAddAct_instAddSemigroupOfAddOpposite(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DomAddAct_instAddSemigroupOfAddOpposite___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DomMulAct_instCommSemigroupOfMulOpposite___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DomMulAct_instCommSemigroupOfMulOpposite___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DomMulAct_instCommSemigroupOfMulOpposite(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DomMulAct_instCommSemigroupOfMulOpposite___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DomAddAct_instAddCommSemigroupOfAddOpposite___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DomAddAct_instAddCommSemigroupOfAddOpposite___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DomAddAct_instAddCommSemigroupOfAddOpposite(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DomAddAct_instAddCommSemigroupOfAddOpposite___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DomMulAct_instLeftCancelSemigroupOfMulOpposite___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DomMulAct_instLeftCancelSemigroupOfMulOpposite___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DomMulAct_instLeftCancelSemigroupOfMulOpposite(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DomMulAct_instLeftCancelSemigroupOfMulOpposite___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DomAddAct_instAddLeftCancelSemigroupOfAddOpposite___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DomAddAct_instAddLeftCancelSemigroupOfAddOpposite___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DomAddAct_instAddLeftCancelSemigroupOfAddOpposite(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DomAddAct_instAddLeftCancelSemigroupOfAddOpposite___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DomMulAct_instRightCancelSemigroupOfMulOpposite___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DomMulAct_instRightCancelSemigroupOfMulOpposite___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DomMulAct_instRightCancelSemigroupOfMulOpposite(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DomMulAct_instRightCancelSemigroupOfMulOpposite___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DomAddAct_instAddRightCancelSemigroupOfAddOpposite___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DomAddAct_instAddRightCancelSemigroupOfAddOpposite___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DomAddAct_instAddRightCancelSemigroupOfAddOpposite(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DomAddAct_instAddRightCancelSemigroupOfAddOpposite___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DomMulAct_instMulOneClassOfMulOpposite___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DomMulAct_instMulOneClassOfMulOpposite___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DomMulAct_instMulOneClassOfMulOpposite(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DomMulAct_instMulOneClassOfMulOpposite___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DomAddAct_instAddZeroClassOfAddOpposite___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DomAddAct_instAddZeroClassOfAddOpposite___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DomAddAct_instAddZeroClassOfAddOpposite(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DomAddAct_instAddZeroClassOfAddOpposite___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DomMulAct_instMonoidOfMulOpposite___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DomMulAct_instMonoidOfMulOpposite___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DomMulAct_instMonoidOfMulOpposite(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DomMulAct_instMonoidOfMulOpposite___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DomAddAct_instAddMonoidOfAddOpposite___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DomAddAct_instAddMonoidOfAddOpposite___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DomAddAct_instAddMonoidOfAddOpposite(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DomAddAct_instAddMonoidOfAddOpposite___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DomMulAct_instCommMonoidOfMulOpposite___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DomMulAct_instCommMonoidOfMulOpposite___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DomMulAct_instCommMonoidOfMulOpposite(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DomMulAct_instCommMonoidOfMulOpposite___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DomAddAct_instAddCommMonoidOfAddOpposite___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DomAddAct_instAddCommMonoidOfAddOpposite___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DomAddAct_instAddCommMonoidOfAddOpposite(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DomAddAct_instAddCommMonoidOfAddOpposite___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DomMulAct_instLeftCancelMonoidOfMulOpposite___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DomMulAct_instLeftCancelMonoidOfMulOpposite___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DomMulAct_instLeftCancelMonoidOfMulOpposite(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DomMulAct_instLeftCancelMonoidOfMulOpposite___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DomAddAct_instAddLeftCancelMonoidOfAddOpposite___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DomAddAct_instAddLeftCancelMonoidOfAddOpposite___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DomAddAct_instAddLeftCancelMonoidOfAddOpposite(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DomAddAct_instAddLeftCancelMonoidOfAddOpposite___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DomMulAct_instRightCancelMonoidOfMulOpposite___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DomMulAct_instRightCancelMonoidOfMulOpposite___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DomMulAct_instRightCancelMonoidOfMulOpposite(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DomMulAct_instRightCancelMonoidOfMulOpposite___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DomAddAct_instAddRightCancelMonoidOfAddOpposite___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DomAddAct_instAddRightCancelMonoidOfAddOpposite___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DomAddAct_instAddRightCancelMonoidOfAddOpposite(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DomAddAct_instAddRightCancelMonoidOfAddOpposite___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DomMulAct_instCancelMonoidOfMulOpposite___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DomMulAct_instCancelMonoidOfMulOpposite___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DomMulAct_instCancelMonoidOfMulOpposite(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DomMulAct_instCancelMonoidOfMulOpposite___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DomAddAct_instAddCancelMonoidOfAddOpposite___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DomAddAct_instAddCancelMonoidOfAddOpposite___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DomAddAct_instAddCancelMonoidOfAddOpposite(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DomAddAct_instAddCancelMonoidOfAddOpposite___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DomMulAct_instCancelCommMonoidOfMulOpposite___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DomMulAct_instCancelCommMonoidOfMulOpposite___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DomMulAct_instCancelCommMonoidOfMulOpposite(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DomMulAct_instCancelCommMonoidOfMulOpposite___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DomAddAct_instAddCancelCommMonoidOfAddOpposite___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DomAddAct_instAddCancelCommMonoidOfAddOpposite___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DomAddAct_instAddCancelCommMonoidOfAddOpposite(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DomAddAct_instAddCancelCommMonoidOfAddOpposite___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DomMulAct_instInvolutiveInvOfMulOpposite___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DomMulAct_instInvolutiveInvOfMulOpposite___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DomMulAct_instInvolutiveInvOfMulOpposite(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DomMulAct_instInvolutiveInvOfMulOpposite___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DomAddAct_instInvolutiveNegOfAddOpposite___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DomAddAct_instInvolutiveNegOfAddOpposite___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DomAddAct_instInvolutiveNegOfAddOpposite(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DomAddAct_instInvolutiveNegOfAddOpposite___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DomMulAct_instDivInvMonoidOfMulOpposite___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DomMulAct_instDivInvMonoidOfMulOpposite___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DomMulAct_instDivInvMonoidOfMulOpposite(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DomMulAct_instDivInvMonoidOfMulOpposite___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DomAddAct_instSubNegAddMonoidOfAddOpposite___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DomAddAct_instSubNegAddMonoidOfAddOpposite___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DomAddAct_instSubNegAddMonoidOfAddOpposite(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DomAddAct_instSubNegAddMonoidOfAddOpposite___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DomMulAct_instInvOneClassOfMulOpposite___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DomMulAct_instInvOneClassOfMulOpposite___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DomMulAct_instInvOneClassOfMulOpposite(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DomMulAct_instInvOneClassOfMulOpposite___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DomAddAct_instNegZeroClassOfAddOpposite___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DomAddAct_instNegZeroClassOfAddOpposite___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DomAddAct_instNegZeroClassOfAddOpposite(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DomAddAct_instNegZeroClassOfAddOpposite___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DomMulAct_instDivInvOneMonoidOfMulOpposite___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DomMulAct_instDivInvOneMonoidOfMulOpposite___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DomMulAct_instDivInvOneMonoidOfMulOpposite(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DomMulAct_instDivInvOneMonoidOfMulOpposite___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DomAddAct_instSubNegZeroMonoidOfAddOpposite___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DomAddAct_instSubNegZeroMonoidOfAddOpposite___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DomAddAct_instSubNegZeroMonoidOfAddOpposite(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DomAddAct_instSubNegZeroMonoidOfAddOpposite___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DomMulAct_instDivisionMonoidOfMulOpposite___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DomMulAct_instDivisionMonoidOfMulOpposite___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DomMulAct_instDivisionMonoidOfMulOpposite(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DomMulAct_instDivisionMonoidOfMulOpposite___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DomAddAct_instSubtractionMonoidOfAddOpposite___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DomAddAct_instSubtractionMonoidOfAddOpposite___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DomAddAct_instSubtractionMonoidOfAddOpposite(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DomAddAct_instSubtractionMonoidOfAddOpposite___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DomMulAct_instDivisionCommMonoidOfMulOpposite___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DomMulAct_instDivisionCommMonoidOfMulOpposite___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DomMulAct_instDivisionCommMonoidOfMulOpposite(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DomMulAct_instDivisionCommMonoidOfMulOpposite___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DomAddAct_instDivisionAddCommMonoidOfAddOpposite___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DomAddAct_instDivisionAddCommMonoidOfAddOpposite___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DomAddAct_instDivisionAddCommMonoidOfAddOpposite(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DomAddAct_instDivisionAddCommMonoidOfAddOpposite___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DomMulAct_instGroupOfMulOpposite___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DomMulAct_instGroupOfMulOpposite___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DomMulAct_instGroupOfMulOpposite(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DomMulAct_instGroupOfMulOpposite___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DomAddAct_instAddGroupOfAddOpposite___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DomAddAct_instAddGroupOfAddOpposite___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DomAddAct_instAddGroupOfAddOpposite(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DomAddAct_instAddGroupOfAddOpposite___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DomMulAct_instCommGroupOfMulOpposite___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DomMulAct_instCommGroupOfMulOpposite___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DomMulAct_instCommGroupOfMulOpposite(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DomMulAct_instCommGroupOfMulOpposite___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DomAddAct_instAddCommGroupOfAddOpposite___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DomAddAct_instAddCommGroupOfAddOpposite___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DomAddAct_instAddCommGroupOfAddOpposite(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DomAddAct_instAddCommGroupOfAddOpposite___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DomMulAct_instNonAssocSemiringOfMulOpposite___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DomMulAct_instNonAssocSemiringOfMulOpposite___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DomMulAct_instNonAssocSemiringOfMulOpposite(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DomMulAct_instNonAssocSemiringOfMulOpposite___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DomAddAct_instNonAssocSemiringOfAddOpposite___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DomAddAct_instNonAssocSemiringOfAddOpposite___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DomAddAct_instNonAssocSemiringOfAddOpposite(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DomAddAct_instNonAssocSemiringOfAddOpposite___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DomMulAct_instNonUnitalSemiringOfMulOpposite___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DomMulAct_instNonUnitalSemiringOfMulOpposite___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DomMulAct_instNonUnitalSemiringOfMulOpposite(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DomMulAct_instNonUnitalSemiringOfMulOpposite___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DomAddAct_instNonUnitalSemiringOfAddOpposite___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DomAddAct_instNonUnitalSemiringOfAddOpposite___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DomAddAct_instNonUnitalSemiringOfAddOpposite(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DomAddAct_instNonUnitalSemiringOfAddOpposite___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DomMulAct_instSemiringOfMulOpposite___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DomMulAct_instSemiringOfMulOpposite___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DomMulAct_instSemiringOfMulOpposite(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DomMulAct_instSemiringOfMulOpposite___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DomAddAct_instSemiringOfAddOpposite___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DomAddAct_instSemiringOfAddOpposite___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DomAddAct_instSemiringOfAddOpposite(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DomAddAct_instSemiringOfAddOpposite___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DomMulAct_instRingOfMulOpposite___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DomMulAct_instRingOfMulOpposite___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DomMulAct_instRingOfMulOpposite(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DomMulAct_instRingOfMulOpposite___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DomAddAct_instRingOfAddOpposite___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DomAddAct_instRingOfAddOpposite___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DomAddAct_instRingOfAddOpposite(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DomAddAct_instRingOfAddOpposite___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DomMulAct_instCommRingOfMulOpposite___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DomMulAct_instCommRingOfMulOpposite___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DomMulAct_instCommRingOfMulOpposite(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DomMulAct_instCommRingOfMulOpposite___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DomAddAct_instCommRingOfAddOpposite___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DomAddAct_instCommRingOfAddOpposite___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DomAddAct_instCommRingOfAddOpposite(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DomAddAct_instCommRingOfAddOpposite___boxed(lean_object*, lean_object*);
static lean_once_cell_t lp_mathlib_DomMulAct_instSMulForall___redArg___lam__0___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_DomMulAct_instSMulForall___redArg___lam__0___closed__0;
LEAN_EXPORT lean_object* lp_mathlib_DomMulAct_instSMulForall___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DomMulAct_instSMulForall___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DomMulAct_instSMulForall(lean_object*, lean_object*, lean_object*, lean_object*);
static lean_once_cell_t lp_mathlib_DomAddAct_instVAddForall___redArg___lam__0___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_DomAddAct_instVAddForall___redArg___lam__0___closed__0;
LEAN_EXPORT lean_object* lp_mathlib_DomAddAct_instVAddForall___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DomAddAct_instVAddForall___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DomAddAct_instVAddForall(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DomMulAct_instSMulZeroClassForallOfSMul___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DomMulAct_instSMulZeroClassForallOfSMul(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DomMulAct_instSMulZeroClassForallOfSMul___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DomMulAct_instDistribSMulForallOfSMul___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DomMulAct_instDistribSMulForallOfSMul(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DomMulAct_instDistribSMulForallOfSMul___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DomMulAct_instMulActionForall___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DomMulAct_instMulActionForall(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DomMulAct_instMulActionForall___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DomAddAct_instAddActionForall___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DomAddAct_instAddActionForall(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DomAddAct_instAddActionForall___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DomMulAct_instDistribMulActionForallOfMulAction___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DomMulAct_instDistribMulActionForallOfMulAction(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DomMulAct_instDistribMulActionForallOfMulAction___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DomMulAct_instMulDistribMulActionForallOfMulAction___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DomMulAct_instMulDistribMulActionForallOfMulAction(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DomMulAct_instMulDistribMulActionForallOfMulAction___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DomMulAct_instSMulMonoidHom___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DomMulAct_instSMulMonoidHom___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DomMulAct_instSMulMonoidHom(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DomMulAct_instSMulMonoidHom___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DomMulAct_instMulActionMonoidHom___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DomMulAct_instMulActionMonoidHom(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DomMulAct_instMulActionMonoidHom___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DomMulAct_instSMulAddMonoidHom___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DomMulAct_instSMulAddMonoidHom___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DomMulAct_instSMulAddMonoidHom(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DomMulAct_instSMulAddMonoidHom___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DomMulAct_instMulActionAddMonoidHomOfDistribMulAction___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DomMulAct_instMulActionAddMonoidHomOfDistribMulAction(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DomMulAct_instMulActionAddMonoidHomOfDistribMulAction___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DomMulAct_instDistribMulActionAddMonoidHom___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DomMulAct_instDistribMulActionAddMonoidHom(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DomMulAct_instDistribMulActionAddMonoidHom___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DomMulAct_instMulDistribMulActionMonoidHom___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DomMulAct_instMulDistribMulActionMonoidHom(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DomMulAct_instMulDistribMulActionMonoidHom___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_object* _init_lp_mathlib___aux__Mathlib__GroupTheory__GroupAction__DomAct__Basic______macroRules__term___u1d48_u1d50_u1d43__1___closed__6(void){
_start:
{
lean_object* v___x_22_; lean_object* v___x_23_; 
v___x_22_ = ((lean_object*)(lp_mathlib___aux__Mathlib__GroupTheory__GroupAction__DomAct__Basic______macroRules__term___u1d48_u1d50_u1d43__1___closed__5));
v___x_23_ = l_String_toRawSubstring_x27(v___x_22_);
return v___x_23_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__GroupTheory__GroupAction__DomAct__Basic______macroRules__term___u1d48_u1d50_u1d43__1(lean_object* v_x_35_, lean_object* v_a_36_, lean_object* v_a_37_){
_start:
{
lean_object* v___x_38_; uint8_t v___x_39_; 
v___x_38_ = ((lean_object*)(lp_mathlib_term___u1d48_u1d50_u1d43___closed__1));
lean_inc(v_x_35_);
v___x_39_ = l_Lean_Syntax_isOfKind(v_x_35_, v___x_38_);
if (v___x_39_ == 0)
{
lean_object* v___x_40_; lean_object* v___x_41_; 
lean_dec(v_x_35_);
v___x_40_ = lean_box(1);
v___x_41_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_41_, 0, v___x_40_);
lean_ctor_set(v___x_41_, 1, v_a_37_);
return v___x_41_;
}
else
{
lean_object* v_quotContext_42_; lean_object* v_currMacroScope_43_; lean_object* v_ref_44_; lean_object* v___x_45_; lean_object* v___x_46_; uint8_t v___x_47_; lean_object* v___x_48_; lean_object* v___x_49_; lean_object* v___x_50_; lean_object* v___x_51_; lean_object* v___x_52_; lean_object* v___x_53_; lean_object* v___x_54_; lean_object* v___x_55_; lean_object* v___x_56_; lean_object* v___x_57_; lean_object* v___x_58_; 
v_quotContext_42_ = lean_ctor_get(v_a_36_, 1);
v_currMacroScope_43_ = lean_ctor_get(v_a_36_, 2);
v_ref_44_ = lean_ctor_get(v_a_36_, 5);
v___x_45_ = lean_unsigned_to_nat(0u);
v___x_46_ = l_Lean_Syntax_getArg(v_x_35_, v___x_45_);
lean_dec(v_x_35_);
v___x_47_ = 0;
v___x_48_ = l_Lean_SourceInfo_fromRef(v_ref_44_, v___x_47_);
v___x_49_ = ((lean_object*)(lp_mathlib___aux__Mathlib__GroupTheory__GroupAction__DomAct__Basic______macroRules__term___u1d48_u1d50_u1d43__1___closed__4));
v___x_50_ = lean_obj_once(&lp_mathlib___aux__Mathlib__GroupTheory__GroupAction__DomAct__Basic______macroRules__term___u1d48_u1d50_u1d43__1___closed__6, &lp_mathlib___aux__Mathlib__GroupTheory__GroupAction__DomAct__Basic______macroRules__term___u1d48_u1d50_u1d43__1___closed__6_once, _init_lp_mathlib___aux__Mathlib__GroupTheory__GroupAction__DomAct__Basic______macroRules__term___u1d48_u1d50_u1d43__1___closed__6);
v___x_51_ = ((lean_object*)(lp_mathlib___aux__Mathlib__GroupTheory__GroupAction__DomAct__Basic______macroRules__term___u1d48_u1d50_u1d43__1___closed__7));
lean_inc(v_currMacroScope_43_);
lean_inc(v_quotContext_42_);
v___x_52_ = l_Lean_addMacroScope(v_quotContext_42_, v___x_51_, v_currMacroScope_43_);
v___x_53_ = ((lean_object*)(lp_mathlib___aux__Mathlib__GroupTheory__GroupAction__DomAct__Basic______macroRules__term___u1d48_u1d50_u1d43__1___closed__9));
lean_inc_n(v___x_48_, 2);
v___x_54_ = lean_alloc_ctor(3, 4, 0);
lean_ctor_set(v___x_54_, 0, v___x_48_);
lean_ctor_set(v___x_54_, 1, v___x_50_);
lean_ctor_set(v___x_54_, 2, v___x_52_);
lean_ctor_set(v___x_54_, 3, v___x_53_);
v___x_55_ = ((lean_object*)(lp_mathlib___aux__Mathlib__GroupTheory__GroupAction__DomAct__Basic______macroRules__term___u1d48_u1d50_u1d43__1___closed__11));
v___x_56_ = l_Lean_Syntax_node1(v___x_48_, v___x_55_, v___x_46_);
v___x_57_ = l_Lean_Syntax_node2(v___x_48_, v___x_49_, v___x_54_, v___x_56_);
v___x_58_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_58_, 0, v___x_57_);
lean_ctor_set(v___x_58_, 1, v_a_37_);
return v___x_58_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__GroupTheory__GroupAction__DomAct__Basic______macroRules__term___u1d48_u1d50_u1d43__1___boxed(lean_object* v_x_59_, lean_object* v_a_60_, lean_object* v_a_61_){
_start:
{
lean_object* v_res_62_; 
v_res_62_ = lp_mathlib___aux__Mathlib__GroupTheory__GroupAction__DomAct__Basic______macroRules__term___u1d48_u1d50_u1d43__1(v_x_59_, v_a_60_, v_a_61_);
lean_dec_ref(v_a_60_);
return v_res_62_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__GroupTheory__GroupAction__DomAct__Basic______unexpand__DomMulAct__1(lean_object* v_x_66_, lean_object* v_a_67_, lean_object* v_a_68_){
_start:
{
lean_object* v___x_69_; uint8_t v___x_70_; 
v___x_69_ = ((lean_object*)(lp_mathlib___aux__Mathlib__GroupTheory__GroupAction__DomAct__Basic______macroRules__term___u1d48_u1d50_u1d43__1___closed__4));
lean_inc(v_x_66_);
v___x_70_ = l_Lean_Syntax_isOfKind(v_x_66_, v___x_69_);
if (v___x_70_ == 0)
{
lean_object* v___x_71_; lean_object* v___x_72_; 
lean_dec(v_x_66_);
v___x_71_ = lean_box(0);
v___x_72_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_72_, 0, v___x_71_);
lean_ctor_set(v___x_72_, 1, v_a_68_);
return v___x_72_;
}
else
{
lean_object* v___x_73_; lean_object* v___x_74_; lean_object* v___x_75_; uint8_t v___x_76_; 
v___x_73_ = lean_unsigned_to_nat(0u);
v___x_74_ = l_Lean_Syntax_getArg(v_x_66_, v___x_73_);
v___x_75_ = ((lean_object*)(lp_mathlib___aux__Mathlib__GroupTheory__GroupAction__DomAct__Basic______unexpand__DomMulAct__1___closed__1));
lean_inc(v___x_74_);
v___x_76_ = l_Lean_Syntax_isOfKind(v___x_74_, v___x_75_);
if (v___x_76_ == 0)
{
lean_object* v___x_77_; lean_object* v___x_78_; 
lean_dec(v___x_74_);
lean_dec(v_x_66_);
v___x_77_ = lean_box(0);
v___x_78_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_78_, 0, v___x_77_);
lean_ctor_set(v___x_78_, 1, v_a_68_);
return v___x_78_;
}
else
{
lean_object* v___x_79_; lean_object* v___x_80_; uint8_t v___x_81_; 
v___x_79_ = lean_unsigned_to_nat(1u);
v___x_80_ = l_Lean_Syntax_getArg(v_x_66_, v___x_79_);
lean_dec(v_x_66_);
lean_inc(v___x_80_);
v___x_81_ = l_Lean_Syntax_matchesNull(v___x_80_, v___x_79_);
if (v___x_81_ == 0)
{
lean_object* v___x_82_; lean_object* v___x_83_; 
lean_dec(v___x_80_);
lean_dec(v___x_74_);
v___x_82_ = lean_box(0);
v___x_83_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_83_, 0, v___x_82_);
lean_ctor_set(v___x_83_, 1, v_a_68_);
return v___x_83_;
}
else
{
lean_object* v___x_84_; lean_object* v_ref_85_; uint8_t v___x_86_; lean_object* v___x_87_; lean_object* v___x_88_; lean_object* v___x_89_; lean_object* v___x_90_; lean_object* v___x_91_; lean_object* v___x_92_; 
v___x_84_ = l_Lean_Syntax_getArg(v___x_80_, v___x_73_);
lean_dec(v___x_80_);
v_ref_85_ = l_Lean_replaceRef(v___x_74_, v_a_67_);
lean_dec(v___x_74_);
v___x_86_ = 0;
v___x_87_ = l_Lean_SourceInfo_fromRef(v_ref_85_, v___x_86_);
lean_dec(v_ref_85_);
v___x_88_ = ((lean_object*)(lp_mathlib_term___u1d48_u1d50_u1d43___closed__1));
v___x_89_ = ((lean_object*)(lp_mathlib_term___u1d48_u1d50_u1d43___closed__2));
lean_inc(v___x_87_);
v___x_90_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_90_, 0, v___x_87_);
lean_ctor_set(v___x_90_, 1, v___x_89_);
v___x_91_ = l_Lean_Syntax_node2(v___x_87_, v___x_88_, v___x_84_, v___x_90_);
v___x_92_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_92_, 0, v___x_91_);
lean_ctor_set(v___x_92_, 1, v_a_68_);
return v___x_92_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__GroupTheory__GroupAction__DomAct__Basic______unexpand__DomMulAct__1___boxed(lean_object* v_x_93_, lean_object* v_a_94_, lean_object* v_a_95_){
_start:
{
lean_object* v_res_96_; 
v_res_96_ = lp_mathlib___aux__Mathlib__GroupTheory__GroupAction__DomAct__Basic______unexpand__DomMulAct__1(v_x_93_, v_a_94_, v_a_95_);
lean_dec(v_a_94_);
return v_res_96_;
}
}
static lean_object* _init_lp_mathlib___aux__Mathlib__GroupTheory__GroupAction__DomAct__Basic______macroRules__term___u1d48_u1d43_u1d43__1___closed__1(void){
_start:
{
lean_object* v___x_109_; lean_object* v___x_110_; 
v___x_109_ = ((lean_object*)(lp_mathlib___aux__Mathlib__GroupTheory__GroupAction__DomAct__Basic______macroRules__term___u1d48_u1d43_u1d43__1___closed__0));
v___x_110_ = l_String_toRawSubstring_x27(v___x_109_);
return v___x_110_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__GroupTheory__GroupAction__DomAct__Basic______macroRules__term___u1d48_u1d43_u1d43__1(lean_object* v_x_119_, lean_object* v_a_120_, lean_object* v_a_121_){
_start:
{
lean_object* v___x_122_; uint8_t v___x_123_; 
v___x_122_ = ((lean_object*)(lp_mathlib_term___u1d48_u1d43_u1d43___closed__1));
lean_inc(v_x_119_);
v___x_123_ = l_Lean_Syntax_isOfKind(v_x_119_, v___x_122_);
if (v___x_123_ == 0)
{
lean_object* v___x_124_; lean_object* v___x_125_; 
lean_dec(v_x_119_);
v___x_124_ = lean_box(1);
v___x_125_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_125_, 0, v___x_124_);
lean_ctor_set(v___x_125_, 1, v_a_121_);
return v___x_125_;
}
else
{
lean_object* v_quotContext_126_; lean_object* v_currMacroScope_127_; lean_object* v_ref_128_; lean_object* v___x_129_; lean_object* v___x_130_; uint8_t v___x_131_; lean_object* v___x_132_; lean_object* v___x_133_; lean_object* v___x_134_; lean_object* v___x_135_; lean_object* v___x_136_; lean_object* v___x_137_; lean_object* v___x_138_; lean_object* v___x_139_; lean_object* v___x_140_; lean_object* v___x_141_; lean_object* v___x_142_; 
v_quotContext_126_ = lean_ctor_get(v_a_120_, 1);
v_currMacroScope_127_ = lean_ctor_get(v_a_120_, 2);
v_ref_128_ = lean_ctor_get(v_a_120_, 5);
v___x_129_ = lean_unsigned_to_nat(0u);
v___x_130_ = l_Lean_Syntax_getArg(v_x_119_, v___x_129_);
lean_dec(v_x_119_);
v___x_131_ = 0;
v___x_132_ = l_Lean_SourceInfo_fromRef(v_ref_128_, v___x_131_);
v___x_133_ = ((lean_object*)(lp_mathlib___aux__Mathlib__GroupTheory__GroupAction__DomAct__Basic______macroRules__term___u1d48_u1d50_u1d43__1___closed__4));
v___x_134_ = lean_obj_once(&lp_mathlib___aux__Mathlib__GroupTheory__GroupAction__DomAct__Basic______macroRules__term___u1d48_u1d43_u1d43__1___closed__1, &lp_mathlib___aux__Mathlib__GroupTheory__GroupAction__DomAct__Basic______macroRules__term___u1d48_u1d43_u1d43__1___closed__1_once, _init_lp_mathlib___aux__Mathlib__GroupTheory__GroupAction__DomAct__Basic______macroRules__term___u1d48_u1d43_u1d43__1___closed__1);
v___x_135_ = ((lean_object*)(lp_mathlib___aux__Mathlib__GroupTheory__GroupAction__DomAct__Basic______macroRules__term___u1d48_u1d43_u1d43__1___closed__2));
lean_inc(v_currMacroScope_127_);
lean_inc(v_quotContext_126_);
v___x_136_ = l_Lean_addMacroScope(v_quotContext_126_, v___x_135_, v_currMacroScope_127_);
v___x_137_ = ((lean_object*)(lp_mathlib___aux__Mathlib__GroupTheory__GroupAction__DomAct__Basic______macroRules__term___u1d48_u1d43_u1d43__1___closed__4));
lean_inc_n(v___x_132_, 2);
v___x_138_ = lean_alloc_ctor(3, 4, 0);
lean_ctor_set(v___x_138_, 0, v___x_132_);
lean_ctor_set(v___x_138_, 1, v___x_134_);
lean_ctor_set(v___x_138_, 2, v___x_136_);
lean_ctor_set(v___x_138_, 3, v___x_137_);
v___x_139_ = ((lean_object*)(lp_mathlib___aux__Mathlib__GroupTheory__GroupAction__DomAct__Basic______macroRules__term___u1d48_u1d50_u1d43__1___closed__11));
v___x_140_ = l_Lean_Syntax_node1(v___x_132_, v___x_139_, v___x_130_);
v___x_141_ = l_Lean_Syntax_node2(v___x_132_, v___x_133_, v___x_138_, v___x_140_);
v___x_142_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_142_, 0, v___x_141_);
lean_ctor_set(v___x_142_, 1, v_a_121_);
return v___x_142_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__GroupTheory__GroupAction__DomAct__Basic______macroRules__term___u1d48_u1d43_u1d43__1___boxed(lean_object* v_x_143_, lean_object* v_a_144_, lean_object* v_a_145_){
_start:
{
lean_object* v_res_146_; 
v_res_146_ = lp_mathlib___aux__Mathlib__GroupTheory__GroupAction__DomAct__Basic______macroRules__term___u1d48_u1d43_u1d43__1(v_x_143_, v_a_144_, v_a_145_);
lean_dec_ref(v_a_144_);
return v_res_146_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__GroupTheory__GroupAction__DomAct__Basic______unexpand__DomAddAct__1(lean_object* v_x_147_, lean_object* v_a_148_, lean_object* v_a_149_){
_start:
{
lean_object* v___x_150_; uint8_t v___x_151_; 
v___x_150_ = ((lean_object*)(lp_mathlib___aux__Mathlib__GroupTheory__GroupAction__DomAct__Basic______macroRules__term___u1d48_u1d50_u1d43__1___closed__4));
lean_inc(v_x_147_);
v___x_151_ = l_Lean_Syntax_isOfKind(v_x_147_, v___x_150_);
if (v___x_151_ == 0)
{
lean_object* v___x_152_; lean_object* v___x_153_; 
lean_dec(v_x_147_);
v___x_152_ = lean_box(0);
v___x_153_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_153_, 0, v___x_152_);
lean_ctor_set(v___x_153_, 1, v_a_149_);
return v___x_153_;
}
else
{
lean_object* v___x_154_; lean_object* v___x_155_; lean_object* v___x_156_; uint8_t v___x_157_; 
v___x_154_ = lean_unsigned_to_nat(0u);
v___x_155_ = l_Lean_Syntax_getArg(v_x_147_, v___x_154_);
v___x_156_ = ((lean_object*)(lp_mathlib___aux__Mathlib__GroupTheory__GroupAction__DomAct__Basic______unexpand__DomMulAct__1___closed__1));
lean_inc(v___x_155_);
v___x_157_ = l_Lean_Syntax_isOfKind(v___x_155_, v___x_156_);
if (v___x_157_ == 0)
{
lean_object* v___x_158_; lean_object* v___x_159_; 
lean_dec(v___x_155_);
lean_dec(v_x_147_);
v___x_158_ = lean_box(0);
v___x_159_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_159_, 0, v___x_158_);
lean_ctor_set(v___x_159_, 1, v_a_149_);
return v___x_159_;
}
else
{
lean_object* v___x_160_; lean_object* v___x_161_; uint8_t v___x_162_; 
v___x_160_ = lean_unsigned_to_nat(1u);
v___x_161_ = l_Lean_Syntax_getArg(v_x_147_, v___x_160_);
lean_dec(v_x_147_);
lean_inc(v___x_161_);
v___x_162_ = l_Lean_Syntax_matchesNull(v___x_161_, v___x_160_);
if (v___x_162_ == 0)
{
lean_object* v___x_163_; lean_object* v___x_164_; 
lean_dec(v___x_161_);
lean_dec(v___x_155_);
v___x_163_ = lean_box(0);
v___x_164_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_164_, 0, v___x_163_);
lean_ctor_set(v___x_164_, 1, v_a_149_);
return v___x_164_;
}
else
{
lean_object* v___x_165_; lean_object* v_ref_166_; uint8_t v___x_167_; lean_object* v___x_168_; lean_object* v___x_169_; lean_object* v___x_170_; lean_object* v___x_171_; lean_object* v___x_172_; lean_object* v___x_173_; 
v___x_165_ = l_Lean_Syntax_getArg(v___x_161_, v___x_154_);
lean_dec(v___x_161_);
v_ref_166_ = l_Lean_replaceRef(v___x_155_, v_a_148_);
lean_dec(v___x_155_);
v___x_167_ = 0;
v___x_168_ = l_Lean_SourceInfo_fromRef(v_ref_166_, v___x_167_);
lean_dec(v_ref_166_);
v___x_169_ = ((lean_object*)(lp_mathlib_term___u1d48_u1d43_u1d43___closed__1));
v___x_170_ = ((lean_object*)(lp_mathlib_term___u1d48_u1d43_u1d43___closed__2));
lean_inc(v___x_168_);
v___x_171_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_171_, 0, v___x_168_);
lean_ctor_set(v___x_171_, 1, v___x_170_);
v___x_172_ = l_Lean_Syntax_node2(v___x_168_, v___x_169_, v___x_165_, v___x_171_);
v___x_173_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_173_, 0, v___x_172_);
lean_ctor_set(v___x_173_, 1, v_a_149_);
return v___x_173_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__GroupTheory__GroupAction__DomAct__Basic______unexpand__DomAddAct__1___boxed(lean_object* v_x_174_, lean_object* v_a_175_, lean_object* v_a_176_){
_start:
{
lean_object* v_res_177_; 
v_res_177_ = lp_mathlib___aux__Mathlib__GroupTheory__GroupAction__DomAct__Basic______unexpand__DomAddAct__1(v_x_174_, v_a_175_, v_a_176_);
lean_dec(v_a_175_);
return v_res_177_;
}
}
static lean_object* _init_lp_mathlib_DomMulAct_mk___closed__0(void){
_start:
{
lean_object* v___x_178_; 
v___x_178_ = lp_mathlib_MulOpposite_opEquiv(lean_box(0));
return v___x_178_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DomMulAct_mk(lean_object* v___y_179_){
_start:
{
lean_object* v___x_180_; 
v___x_180_ = lean_obj_once(&lp_mathlib_DomMulAct_mk___closed__0, &lp_mathlib_DomMulAct_mk___closed__0_once, _init_lp_mathlib_DomMulAct_mk___closed__0);
return v___x_180_;
}
}
static lean_object* _init_lp_mathlib_DomAddAct_mk___closed__0(void){
_start:
{
lean_object* v___x_181_; 
v___x_181_ = lp_mathlib_AddOpposite_opEquiv(lean_box(0));
return v___x_181_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DomAddAct_mk(lean_object* v___y_182_){
_start:
{
lean_object* v___x_183_; 
v___x_183_ = lean_obj_once(&lp_mathlib_DomAddAct_mk___closed__0, &lp_mathlib_DomAddAct_mk___closed__0_once, _init_lp_mathlib_DomAddAct_mk___closed__0);
return v___x_183_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DomMulAct_instMulOfMulOpposite___redArg(lean_object* v_inst_184_){
_start:
{
lean_inc(v_inst_184_);
return v_inst_184_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DomMulAct_instMulOfMulOpposite___redArg___boxed(lean_object* v_inst_185_){
_start:
{
lean_object* v_res_186_; 
v_res_186_ = lp_mathlib_DomMulAct_instMulOfMulOpposite___redArg(v_inst_185_);
lean_dec(v_inst_185_);
return v_res_186_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DomMulAct_instMulOfMulOpposite(lean_object* v_M_187_, lean_object* v_inst_188_){
_start:
{
lean_inc(v_inst_188_);
return v_inst_188_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DomMulAct_instMulOfMulOpposite___boxed(lean_object* v_M_189_, lean_object* v_inst_190_){
_start:
{
lean_object* v_res_191_; 
v_res_191_ = lp_mathlib_DomMulAct_instMulOfMulOpposite(v_M_189_, v_inst_190_);
lean_dec(v_inst_190_);
return v_res_191_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DomAddAct_instAddOfAddOpposite___redArg(lean_object* v_inst_192_){
_start:
{
lean_inc(v_inst_192_);
return v_inst_192_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DomAddAct_instAddOfAddOpposite___redArg___boxed(lean_object* v_inst_193_){
_start:
{
lean_object* v_res_194_; 
v_res_194_ = lp_mathlib_DomAddAct_instAddOfAddOpposite___redArg(v_inst_193_);
lean_dec(v_inst_193_);
return v_res_194_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DomAddAct_instAddOfAddOpposite(lean_object* v_M_195_, lean_object* v_inst_196_){
_start:
{
lean_inc(v_inst_196_);
return v_inst_196_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DomAddAct_instAddOfAddOpposite___boxed(lean_object* v_M_197_, lean_object* v_inst_198_){
_start:
{
lean_object* v_res_199_; 
v_res_199_ = lp_mathlib_DomAddAct_instAddOfAddOpposite(v_M_197_, v_inst_198_);
lean_dec(v_inst_198_);
return v_res_199_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DomMulAct_instOneOfMulOpposite___redArg(lean_object* v_inst_200_){
_start:
{
lean_inc(v_inst_200_);
return v_inst_200_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DomMulAct_instOneOfMulOpposite___redArg___boxed(lean_object* v_inst_201_){
_start:
{
lean_object* v_res_202_; 
v_res_202_ = lp_mathlib_DomMulAct_instOneOfMulOpposite___redArg(v_inst_201_);
lean_dec(v_inst_201_);
return v_res_202_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DomMulAct_instOneOfMulOpposite(lean_object* v_M_203_, lean_object* v_inst_204_){
_start:
{
lean_inc(v_inst_204_);
return v_inst_204_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DomMulAct_instOneOfMulOpposite___boxed(lean_object* v_M_205_, lean_object* v_inst_206_){
_start:
{
lean_object* v_res_207_; 
v_res_207_ = lp_mathlib_DomMulAct_instOneOfMulOpposite(v_M_205_, v_inst_206_);
lean_dec(v_inst_206_);
return v_res_207_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DomAddAct_instZeroOfAddOpposite___redArg(lean_object* v_inst_208_){
_start:
{
lean_inc(v_inst_208_);
return v_inst_208_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DomAddAct_instZeroOfAddOpposite___redArg___boxed(lean_object* v_inst_209_){
_start:
{
lean_object* v_res_210_; 
v_res_210_ = lp_mathlib_DomAddAct_instZeroOfAddOpposite___redArg(v_inst_209_);
lean_dec(v_inst_209_);
return v_res_210_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DomAddAct_instZeroOfAddOpposite(lean_object* v_M_211_, lean_object* v_inst_212_){
_start:
{
lean_inc(v_inst_212_);
return v_inst_212_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DomAddAct_instZeroOfAddOpposite___boxed(lean_object* v_M_213_, lean_object* v_inst_214_){
_start:
{
lean_object* v_res_215_; 
v_res_215_ = lp_mathlib_DomAddAct_instZeroOfAddOpposite(v_M_213_, v_inst_214_);
lean_dec(v_inst_214_);
return v_res_215_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DomMulAct_instInvOfMulOpposite___redArg(lean_object* v_inst_216_){
_start:
{
lean_inc(v_inst_216_);
return v_inst_216_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DomMulAct_instInvOfMulOpposite___redArg___boxed(lean_object* v_inst_217_){
_start:
{
lean_object* v_res_218_; 
v_res_218_ = lp_mathlib_DomMulAct_instInvOfMulOpposite___redArg(v_inst_217_);
lean_dec(v_inst_217_);
return v_res_218_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DomMulAct_instInvOfMulOpposite(lean_object* v_M_219_, lean_object* v_inst_220_){
_start:
{
lean_inc(v_inst_220_);
return v_inst_220_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DomMulAct_instInvOfMulOpposite___boxed(lean_object* v_M_221_, lean_object* v_inst_222_){
_start:
{
lean_object* v_res_223_; 
v_res_223_ = lp_mathlib_DomMulAct_instInvOfMulOpposite(v_M_221_, v_inst_222_);
lean_dec(v_inst_222_);
return v_res_223_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DomAddAct_instNegOfAddOpposite___redArg(lean_object* v_inst_224_){
_start:
{
lean_inc(v_inst_224_);
return v_inst_224_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DomAddAct_instNegOfAddOpposite___redArg___boxed(lean_object* v_inst_225_){
_start:
{
lean_object* v_res_226_; 
v_res_226_ = lp_mathlib_DomAddAct_instNegOfAddOpposite___redArg(v_inst_225_);
lean_dec(v_inst_225_);
return v_res_226_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DomAddAct_instNegOfAddOpposite(lean_object* v_M_227_, lean_object* v_inst_228_){
_start:
{
lean_inc(v_inst_228_);
return v_inst_228_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DomAddAct_instNegOfAddOpposite___boxed(lean_object* v_M_229_, lean_object* v_inst_230_){
_start:
{
lean_object* v_res_231_; 
v_res_231_ = lp_mathlib_DomAddAct_instNegOfAddOpposite(v_M_229_, v_inst_230_);
lean_dec(v_inst_230_);
return v_res_231_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DomMulAct_instSemigroupOfMulOpposite___redArg(lean_object* v_inst_232_){
_start:
{
lean_inc(v_inst_232_);
return v_inst_232_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DomMulAct_instSemigroupOfMulOpposite___redArg___boxed(lean_object* v_inst_233_){
_start:
{
lean_object* v_res_234_; 
v_res_234_ = lp_mathlib_DomMulAct_instSemigroupOfMulOpposite___redArg(v_inst_233_);
lean_dec(v_inst_233_);
return v_res_234_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DomMulAct_instSemigroupOfMulOpposite(lean_object* v_M_235_, lean_object* v_inst_236_){
_start:
{
lean_inc(v_inst_236_);
return v_inst_236_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DomMulAct_instSemigroupOfMulOpposite___boxed(lean_object* v_M_237_, lean_object* v_inst_238_){
_start:
{
lean_object* v_res_239_; 
v_res_239_ = lp_mathlib_DomMulAct_instSemigroupOfMulOpposite(v_M_237_, v_inst_238_);
lean_dec(v_inst_238_);
return v_res_239_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DomAddAct_instAddSemigroupOfAddOpposite___redArg(lean_object* v_inst_240_){
_start:
{
lean_inc(v_inst_240_);
return v_inst_240_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DomAddAct_instAddSemigroupOfAddOpposite___redArg___boxed(lean_object* v_inst_241_){
_start:
{
lean_object* v_res_242_; 
v_res_242_ = lp_mathlib_DomAddAct_instAddSemigroupOfAddOpposite___redArg(v_inst_241_);
lean_dec(v_inst_241_);
return v_res_242_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DomAddAct_instAddSemigroupOfAddOpposite(lean_object* v_M_243_, lean_object* v_inst_244_){
_start:
{
lean_inc(v_inst_244_);
return v_inst_244_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DomAddAct_instAddSemigroupOfAddOpposite___boxed(lean_object* v_M_245_, lean_object* v_inst_246_){
_start:
{
lean_object* v_res_247_; 
v_res_247_ = lp_mathlib_DomAddAct_instAddSemigroupOfAddOpposite(v_M_245_, v_inst_246_);
lean_dec(v_inst_246_);
return v_res_247_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DomMulAct_instCommSemigroupOfMulOpposite___redArg(lean_object* v_inst_248_){
_start:
{
lean_inc(v_inst_248_);
return v_inst_248_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DomMulAct_instCommSemigroupOfMulOpposite___redArg___boxed(lean_object* v_inst_249_){
_start:
{
lean_object* v_res_250_; 
v_res_250_ = lp_mathlib_DomMulAct_instCommSemigroupOfMulOpposite___redArg(v_inst_249_);
lean_dec(v_inst_249_);
return v_res_250_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DomMulAct_instCommSemigroupOfMulOpposite(lean_object* v_M_251_, lean_object* v_inst_252_){
_start:
{
lean_inc(v_inst_252_);
return v_inst_252_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DomMulAct_instCommSemigroupOfMulOpposite___boxed(lean_object* v_M_253_, lean_object* v_inst_254_){
_start:
{
lean_object* v_res_255_; 
v_res_255_ = lp_mathlib_DomMulAct_instCommSemigroupOfMulOpposite(v_M_253_, v_inst_254_);
lean_dec(v_inst_254_);
return v_res_255_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DomAddAct_instAddCommSemigroupOfAddOpposite___redArg(lean_object* v_inst_256_){
_start:
{
lean_inc(v_inst_256_);
return v_inst_256_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DomAddAct_instAddCommSemigroupOfAddOpposite___redArg___boxed(lean_object* v_inst_257_){
_start:
{
lean_object* v_res_258_; 
v_res_258_ = lp_mathlib_DomAddAct_instAddCommSemigroupOfAddOpposite___redArg(v_inst_257_);
lean_dec(v_inst_257_);
return v_res_258_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DomAddAct_instAddCommSemigroupOfAddOpposite(lean_object* v_M_259_, lean_object* v_inst_260_){
_start:
{
lean_inc(v_inst_260_);
return v_inst_260_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DomAddAct_instAddCommSemigroupOfAddOpposite___boxed(lean_object* v_M_261_, lean_object* v_inst_262_){
_start:
{
lean_object* v_res_263_; 
v_res_263_ = lp_mathlib_DomAddAct_instAddCommSemigroupOfAddOpposite(v_M_261_, v_inst_262_);
lean_dec(v_inst_262_);
return v_res_263_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DomMulAct_instLeftCancelSemigroupOfMulOpposite___redArg(lean_object* v_inst_264_){
_start:
{
lean_inc(v_inst_264_);
return v_inst_264_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DomMulAct_instLeftCancelSemigroupOfMulOpposite___redArg___boxed(lean_object* v_inst_265_){
_start:
{
lean_object* v_res_266_; 
v_res_266_ = lp_mathlib_DomMulAct_instLeftCancelSemigroupOfMulOpposite___redArg(v_inst_265_);
lean_dec(v_inst_265_);
return v_res_266_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DomMulAct_instLeftCancelSemigroupOfMulOpposite(lean_object* v_M_267_, lean_object* v_inst_268_){
_start:
{
lean_inc(v_inst_268_);
return v_inst_268_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DomMulAct_instLeftCancelSemigroupOfMulOpposite___boxed(lean_object* v_M_269_, lean_object* v_inst_270_){
_start:
{
lean_object* v_res_271_; 
v_res_271_ = lp_mathlib_DomMulAct_instLeftCancelSemigroupOfMulOpposite(v_M_269_, v_inst_270_);
lean_dec(v_inst_270_);
return v_res_271_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DomAddAct_instAddLeftCancelSemigroupOfAddOpposite___redArg(lean_object* v_inst_272_){
_start:
{
lean_inc(v_inst_272_);
return v_inst_272_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DomAddAct_instAddLeftCancelSemigroupOfAddOpposite___redArg___boxed(lean_object* v_inst_273_){
_start:
{
lean_object* v_res_274_; 
v_res_274_ = lp_mathlib_DomAddAct_instAddLeftCancelSemigroupOfAddOpposite___redArg(v_inst_273_);
lean_dec(v_inst_273_);
return v_res_274_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DomAddAct_instAddLeftCancelSemigroupOfAddOpposite(lean_object* v_M_275_, lean_object* v_inst_276_){
_start:
{
lean_inc(v_inst_276_);
return v_inst_276_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DomAddAct_instAddLeftCancelSemigroupOfAddOpposite___boxed(lean_object* v_M_277_, lean_object* v_inst_278_){
_start:
{
lean_object* v_res_279_; 
v_res_279_ = lp_mathlib_DomAddAct_instAddLeftCancelSemigroupOfAddOpposite(v_M_277_, v_inst_278_);
lean_dec(v_inst_278_);
return v_res_279_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DomMulAct_instRightCancelSemigroupOfMulOpposite___redArg(lean_object* v_inst_280_){
_start:
{
lean_inc(v_inst_280_);
return v_inst_280_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DomMulAct_instRightCancelSemigroupOfMulOpposite___redArg___boxed(lean_object* v_inst_281_){
_start:
{
lean_object* v_res_282_; 
v_res_282_ = lp_mathlib_DomMulAct_instRightCancelSemigroupOfMulOpposite___redArg(v_inst_281_);
lean_dec(v_inst_281_);
return v_res_282_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DomMulAct_instRightCancelSemigroupOfMulOpposite(lean_object* v_M_283_, lean_object* v_inst_284_){
_start:
{
lean_inc(v_inst_284_);
return v_inst_284_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DomMulAct_instRightCancelSemigroupOfMulOpposite___boxed(lean_object* v_M_285_, lean_object* v_inst_286_){
_start:
{
lean_object* v_res_287_; 
v_res_287_ = lp_mathlib_DomMulAct_instRightCancelSemigroupOfMulOpposite(v_M_285_, v_inst_286_);
lean_dec(v_inst_286_);
return v_res_287_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DomAddAct_instAddRightCancelSemigroupOfAddOpposite___redArg(lean_object* v_inst_288_){
_start:
{
lean_inc(v_inst_288_);
return v_inst_288_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DomAddAct_instAddRightCancelSemigroupOfAddOpposite___redArg___boxed(lean_object* v_inst_289_){
_start:
{
lean_object* v_res_290_; 
v_res_290_ = lp_mathlib_DomAddAct_instAddRightCancelSemigroupOfAddOpposite___redArg(v_inst_289_);
lean_dec(v_inst_289_);
return v_res_290_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DomAddAct_instAddRightCancelSemigroupOfAddOpposite(lean_object* v_M_291_, lean_object* v_inst_292_){
_start:
{
lean_inc(v_inst_292_);
return v_inst_292_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DomAddAct_instAddRightCancelSemigroupOfAddOpposite___boxed(lean_object* v_M_293_, lean_object* v_inst_294_){
_start:
{
lean_object* v_res_295_; 
v_res_295_ = lp_mathlib_DomAddAct_instAddRightCancelSemigroupOfAddOpposite(v_M_293_, v_inst_294_);
lean_dec(v_inst_294_);
return v_res_295_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DomMulAct_instMulOneClassOfMulOpposite___redArg(lean_object* v_inst_296_){
_start:
{
lean_inc_ref(v_inst_296_);
return v_inst_296_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DomMulAct_instMulOneClassOfMulOpposite___redArg___boxed(lean_object* v_inst_297_){
_start:
{
lean_object* v_res_298_; 
v_res_298_ = lp_mathlib_DomMulAct_instMulOneClassOfMulOpposite___redArg(v_inst_297_);
lean_dec_ref(v_inst_297_);
return v_res_298_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DomMulAct_instMulOneClassOfMulOpposite(lean_object* v_M_299_, lean_object* v_inst_300_){
_start:
{
lean_inc_ref(v_inst_300_);
return v_inst_300_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DomMulAct_instMulOneClassOfMulOpposite___boxed(lean_object* v_M_301_, lean_object* v_inst_302_){
_start:
{
lean_object* v_res_303_; 
v_res_303_ = lp_mathlib_DomMulAct_instMulOneClassOfMulOpposite(v_M_301_, v_inst_302_);
lean_dec_ref(v_inst_302_);
return v_res_303_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DomAddAct_instAddZeroClassOfAddOpposite___redArg(lean_object* v_inst_304_){
_start:
{
lean_inc_ref(v_inst_304_);
return v_inst_304_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DomAddAct_instAddZeroClassOfAddOpposite___redArg___boxed(lean_object* v_inst_305_){
_start:
{
lean_object* v_res_306_; 
v_res_306_ = lp_mathlib_DomAddAct_instAddZeroClassOfAddOpposite___redArg(v_inst_305_);
lean_dec_ref(v_inst_305_);
return v_res_306_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DomAddAct_instAddZeroClassOfAddOpposite(lean_object* v_M_307_, lean_object* v_inst_308_){
_start:
{
lean_inc_ref(v_inst_308_);
return v_inst_308_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DomAddAct_instAddZeroClassOfAddOpposite___boxed(lean_object* v_M_309_, lean_object* v_inst_310_){
_start:
{
lean_object* v_res_311_; 
v_res_311_ = lp_mathlib_DomAddAct_instAddZeroClassOfAddOpposite(v_M_309_, v_inst_310_);
lean_dec_ref(v_inst_310_);
return v_res_311_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DomMulAct_instMonoidOfMulOpposite___redArg(lean_object* v_inst_312_){
_start:
{
lean_inc_ref(v_inst_312_);
return v_inst_312_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DomMulAct_instMonoidOfMulOpposite___redArg___boxed(lean_object* v_inst_313_){
_start:
{
lean_object* v_res_314_; 
v_res_314_ = lp_mathlib_DomMulAct_instMonoidOfMulOpposite___redArg(v_inst_313_);
lean_dec_ref(v_inst_313_);
return v_res_314_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DomMulAct_instMonoidOfMulOpposite(lean_object* v_M_315_, lean_object* v_inst_316_){
_start:
{
lean_inc_ref(v_inst_316_);
return v_inst_316_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DomMulAct_instMonoidOfMulOpposite___boxed(lean_object* v_M_317_, lean_object* v_inst_318_){
_start:
{
lean_object* v_res_319_; 
v_res_319_ = lp_mathlib_DomMulAct_instMonoidOfMulOpposite(v_M_317_, v_inst_318_);
lean_dec_ref(v_inst_318_);
return v_res_319_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DomAddAct_instAddMonoidOfAddOpposite___redArg(lean_object* v_inst_320_){
_start:
{
lean_inc_ref(v_inst_320_);
return v_inst_320_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DomAddAct_instAddMonoidOfAddOpposite___redArg___boxed(lean_object* v_inst_321_){
_start:
{
lean_object* v_res_322_; 
v_res_322_ = lp_mathlib_DomAddAct_instAddMonoidOfAddOpposite___redArg(v_inst_321_);
lean_dec_ref(v_inst_321_);
return v_res_322_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DomAddAct_instAddMonoidOfAddOpposite(lean_object* v_M_323_, lean_object* v_inst_324_){
_start:
{
lean_inc_ref(v_inst_324_);
return v_inst_324_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DomAddAct_instAddMonoidOfAddOpposite___boxed(lean_object* v_M_325_, lean_object* v_inst_326_){
_start:
{
lean_object* v_res_327_; 
v_res_327_ = lp_mathlib_DomAddAct_instAddMonoidOfAddOpposite(v_M_325_, v_inst_326_);
lean_dec_ref(v_inst_326_);
return v_res_327_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DomMulAct_instCommMonoidOfMulOpposite___redArg(lean_object* v_inst_328_){
_start:
{
lean_inc_ref(v_inst_328_);
return v_inst_328_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DomMulAct_instCommMonoidOfMulOpposite___redArg___boxed(lean_object* v_inst_329_){
_start:
{
lean_object* v_res_330_; 
v_res_330_ = lp_mathlib_DomMulAct_instCommMonoidOfMulOpposite___redArg(v_inst_329_);
lean_dec_ref(v_inst_329_);
return v_res_330_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DomMulAct_instCommMonoidOfMulOpposite(lean_object* v_M_331_, lean_object* v_inst_332_){
_start:
{
lean_inc_ref(v_inst_332_);
return v_inst_332_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DomMulAct_instCommMonoidOfMulOpposite___boxed(lean_object* v_M_333_, lean_object* v_inst_334_){
_start:
{
lean_object* v_res_335_; 
v_res_335_ = lp_mathlib_DomMulAct_instCommMonoidOfMulOpposite(v_M_333_, v_inst_334_);
lean_dec_ref(v_inst_334_);
return v_res_335_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DomAddAct_instAddCommMonoidOfAddOpposite___redArg(lean_object* v_inst_336_){
_start:
{
lean_inc_ref(v_inst_336_);
return v_inst_336_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DomAddAct_instAddCommMonoidOfAddOpposite___redArg___boxed(lean_object* v_inst_337_){
_start:
{
lean_object* v_res_338_; 
v_res_338_ = lp_mathlib_DomAddAct_instAddCommMonoidOfAddOpposite___redArg(v_inst_337_);
lean_dec_ref(v_inst_337_);
return v_res_338_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DomAddAct_instAddCommMonoidOfAddOpposite(lean_object* v_M_339_, lean_object* v_inst_340_){
_start:
{
lean_inc_ref(v_inst_340_);
return v_inst_340_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DomAddAct_instAddCommMonoidOfAddOpposite___boxed(lean_object* v_M_341_, lean_object* v_inst_342_){
_start:
{
lean_object* v_res_343_; 
v_res_343_ = lp_mathlib_DomAddAct_instAddCommMonoidOfAddOpposite(v_M_341_, v_inst_342_);
lean_dec_ref(v_inst_342_);
return v_res_343_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DomMulAct_instLeftCancelMonoidOfMulOpposite___redArg(lean_object* v_inst_344_){
_start:
{
lean_inc_ref(v_inst_344_);
return v_inst_344_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DomMulAct_instLeftCancelMonoidOfMulOpposite___redArg___boxed(lean_object* v_inst_345_){
_start:
{
lean_object* v_res_346_; 
v_res_346_ = lp_mathlib_DomMulAct_instLeftCancelMonoidOfMulOpposite___redArg(v_inst_345_);
lean_dec_ref(v_inst_345_);
return v_res_346_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DomMulAct_instLeftCancelMonoidOfMulOpposite(lean_object* v_M_347_, lean_object* v_inst_348_){
_start:
{
lean_inc_ref(v_inst_348_);
return v_inst_348_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DomMulAct_instLeftCancelMonoidOfMulOpposite___boxed(lean_object* v_M_349_, lean_object* v_inst_350_){
_start:
{
lean_object* v_res_351_; 
v_res_351_ = lp_mathlib_DomMulAct_instLeftCancelMonoidOfMulOpposite(v_M_349_, v_inst_350_);
lean_dec_ref(v_inst_350_);
return v_res_351_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DomAddAct_instAddLeftCancelMonoidOfAddOpposite___redArg(lean_object* v_inst_352_){
_start:
{
lean_inc_ref(v_inst_352_);
return v_inst_352_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DomAddAct_instAddLeftCancelMonoidOfAddOpposite___redArg___boxed(lean_object* v_inst_353_){
_start:
{
lean_object* v_res_354_; 
v_res_354_ = lp_mathlib_DomAddAct_instAddLeftCancelMonoidOfAddOpposite___redArg(v_inst_353_);
lean_dec_ref(v_inst_353_);
return v_res_354_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DomAddAct_instAddLeftCancelMonoidOfAddOpposite(lean_object* v_M_355_, lean_object* v_inst_356_){
_start:
{
lean_inc_ref(v_inst_356_);
return v_inst_356_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DomAddAct_instAddLeftCancelMonoidOfAddOpposite___boxed(lean_object* v_M_357_, lean_object* v_inst_358_){
_start:
{
lean_object* v_res_359_; 
v_res_359_ = lp_mathlib_DomAddAct_instAddLeftCancelMonoidOfAddOpposite(v_M_357_, v_inst_358_);
lean_dec_ref(v_inst_358_);
return v_res_359_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DomMulAct_instRightCancelMonoidOfMulOpposite___redArg(lean_object* v_inst_360_){
_start:
{
lean_inc_ref(v_inst_360_);
return v_inst_360_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DomMulAct_instRightCancelMonoidOfMulOpposite___redArg___boxed(lean_object* v_inst_361_){
_start:
{
lean_object* v_res_362_; 
v_res_362_ = lp_mathlib_DomMulAct_instRightCancelMonoidOfMulOpposite___redArg(v_inst_361_);
lean_dec_ref(v_inst_361_);
return v_res_362_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DomMulAct_instRightCancelMonoidOfMulOpposite(lean_object* v_M_363_, lean_object* v_inst_364_){
_start:
{
lean_inc_ref(v_inst_364_);
return v_inst_364_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DomMulAct_instRightCancelMonoidOfMulOpposite___boxed(lean_object* v_M_365_, lean_object* v_inst_366_){
_start:
{
lean_object* v_res_367_; 
v_res_367_ = lp_mathlib_DomMulAct_instRightCancelMonoidOfMulOpposite(v_M_365_, v_inst_366_);
lean_dec_ref(v_inst_366_);
return v_res_367_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DomAddAct_instAddRightCancelMonoidOfAddOpposite___redArg(lean_object* v_inst_368_){
_start:
{
lean_inc_ref(v_inst_368_);
return v_inst_368_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DomAddAct_instAddRightCancelMonoidOfAddOpposite___redArg___boxed(lean_object* v_inst_369_){
_start:
{
lean_object* v_res_370_; 
v_res_370_ = lp_mathlib_DomAddAct_instAddRightCancelMonoidOfAddOpposite___redArg(v_inst_369_);
lean_dec_ref(v_inst_369_);
return v_res_370_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DomAddAct_instAddRightCancelMonoidOfAddOpposite(lean_object* v_M_371_, lean_object* v_inst_372_){
_start:
{
lean_inc_ref(v_inst_372_);
return v_inst_372_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DomAddAct_instAddRightCancelMonoidOfAddOpposite___boxed(lean_object* v_M_373_, lean_object* v_inst_374_){
_start:
{
lean_object* v_res_375_; 
v_res_375_ = lp_mathlib_DomAddAct_instAddRightCancelMonoidOfAddOpposite(v_M_373_, v_inst_374_);
lean_dec_ref(v_inst_374_);
return v_res_375_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DomMulAct_instCancelMonoidOfMulOpposite___redArg(lean_object* v_inst_376_){
_start:
{
lean_inc_ref(v_inst_376_);
return v_inst_376_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DomMulAct_instCancelMonoidOfMulOpposite___redArg___boxed(lean_object* v_inst_377_){
_start:
{
lean_object* v_res_378_; 
v_res_378_ = lp_mathlib_DomMulAct_instCancelMonoidOfMulOpposite___redArg(v_inst_377_);
lean_dec_ref(v_inst_377_);
return v_res_378_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DomMulAct_instCancelMonoidOfMulOpposite(lean_object* v_M_379_, lean_object* v_inst_380_){
_start:
{
lean_inc_ref(v_inst_380_);
return v_inst_380_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DomMulAct_instCancelMonoidOfMulOpposite___boxed(lean_object* v_M_381_, lean_object* v_inst_382_){
_start:
{
lean_object* v_res_383_; 
v_res_383_ = lp_mathlib_DomMulAct_instCancelMonoidOfMulOpposite(v_M_381_, v_inst_382_);
lean_dec_ref(v_inst_382_);
return v_res_383_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DomAddAct_instAddCancelMonoidOfAddOpposite___redArg(lean_object* v_inst_384_){
_start:
{
lean_inc_ref(v_inst_384_);
return v_inst_384_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DomAddAct_instAddCancelMonoidOfAddOpposite___redArg___boxed(lean_object* v_inst_385_){
_start:
{
lean_object* v_res_386_; 
v_res_386_ = lp_mathlib_DomAddAct_instAddCancelMonoidOfAddOpposite___redArg(v_inst_385_);
lean_dec_ref(v_inst_385_);
return v_res_386_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DomAddAct_instAddCancelMonoidOfAddOpposite(lean_object* v_M_387_, lean_object* v_inst_388_){
_start:
{
lean_inc_ref(v_inst_388_);
return v_inst_388_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DomAddAct_instAddCancelMonoidOfAddOpposite___boxed(lean_object* v_M_389_, lean_object* v_inst_390_){
_start:
{
lean_object* v_res_391_; 
v_res_391_ = lp_mathlib_DomAddAct_instAddCancelMonoidOfAddOpposite(v_M_389_, v_inst_390_);
lean_dec_ref(v_inst_390_);
return v_res_391_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DomMulAct_instCancelCommMonoidOfMulOpposite___redArg(lean_object* v_inst_392_){
_start:
{
lean_inc_ref(v_inst_392_);
return v_inst_392_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DomMulAct_instCancelCommMonoidOfMulOpposite___redArg___boxed(lean_object* v_inst_393_){
_start:
{
lean_object* v_res_394_; 
v_res_394_ = lp_mathlib_DomMulAct_instCancelCommMonoidOfMulOpposite___redArg(v_inst_393_);
lean_dec_ref(v_inst_393_);
return v_res_394_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DomMulAct_instCancelCommMonoidOfMulOpposite(lean_object* v_M_395_, lean_object* v_inst_396_){
_start:
{
lean_inc_ref(v_inst_396_);
return v_inst_396_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DomMulAct_instCancelCommMonoidOfMulOpposite___boxed(lean_object* v_M_397_, lean_object* v_inst_398_){
_start:
{
lean_object* v_res_399_; 
v_res_399_ = lp_mathlib_DomMulAct_instCancelCommMonoidOfMulOpposite(v_M_397_, v_inst_398_);
lean_dec_ref(v_inst_398_);
return v_res_399_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DomAddAct_instAddCancelCommMonoidOfAddOpposite___redArg(lean_object* v_inst_400_){
_start:
{
lean_inc_ref(v_inst_400_);
return v_inst_400_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DomAddAct_instAddCancelCommMonoidOfAddOpposite___redArg___boxed(lean_object* v_inst_401_){
_start:
{
lean_object* v_res_402_; 
v_res_402_ = lp_mathlib_DomAddAct_instAddCancelCommMonoidOfAddOpposite___redArg(v_inst_401_);
lean_dec_ref(v_inst_401_);
return v_res_402_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DomAddAct_instAddCancelCommMonoidOfAddOpposite(lean_object* v_M_403_, lean_object* v_inst_404_){
_start:
{
lean_inc_ref(v_inst_404_);
return v_inst_404_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DomAddAct_instAddCancelCommMonoidOfAddOpposite___boxed(lean_object* v_M_405_, lean_object* v_inst_406_){
_start:
{
lean_object* v_res_407_; 
v_res_407_ = lp_mathlib_DomAddAct_instAddCancelCommMonoidOfAddOpposite(v_M_405_, v_inst_406_);
lean_dec_ref(v_inst_406_);
return v_res_407_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DomMulAct_instInvolutiveInvOfMulOpposite___redArg(lean_object* v_inst_408_){
_start:
{
lean_inc(v_inst_408_);
return v_inst_408_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DomMulAct_instInvolutiveInvOfMulOpposite___redArg___boxed(lean_object* v_inst_409_){
_start:
{
lean_object* v_res_410_; 
v_res_410_ = lp_mathlib_DomMulAct_instInvolutiveInvOfMulOpposite___redArg(v_inst_409_);
lean_dec(v_inst_409_);
return v_res_410_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DomMulAct_instInvolutiveInvOfMulOpposite(lean_object* v_M_411_, lean_object* v_inst_412_){
_start:
{
lean_inc(v_inst_412_);
return v_inst_412_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DomMulAct_instInvolutiveInvOfMulOpposite___boxed(lean_object* v_M_413_, lean_object* v_inst_414_){
_start:
{
lean_object* v_res_415_; 
v_res_415_ = lp_mathlib_DomMulAct_instInvolutiveInvOfMulOpposite(v_M_413_, v_inst_414_);
lean_dec(v_inst_414_);
return v_res_415_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DomAddAct_instInvolutiveNegOfAddOpposite___redArg(lean_object* v_inst_416_){
_start:
{
lean_inc(v_inst_416_);
return v_inst_416_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DomAddAct_instInvolutiveNegOfAddOpposite___redArg___boxed(lean_object* v_inst_417_){
_start:
{
lean_object* v_res_418_; 
v_res_418_ = lp_mathlib_DomAddAct_instInvolutiveNegOfAddOpposite___redArg(v_inst_417_);
lean_dec(v_inst_417_);
return v_res_418_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DomAddAct_instInvolutiveNegOfAddOpposite(lean_object* v_M_419_, lean_object* v_inst_420_){
_start:
{
lean_inc(v_inst_420_);
return v_inst_420_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DomAddAct_instInvolutiveNegOfAddOpposite___boxed(lean_object* v_M_421_, lean_object* v_inst_422_){
_start:
{
lean_object* v_res_423_; 
v_res_423_ = lp_mathlib_DomAddAct_instInvolutiveNegOfAddOpposite(v_M_421_, v_inst_422_);
lean_dec(v_inst_422_);
return v_res_423_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DomMulAct_instDivInvMonoidOfMulOpposite___redArg(lean_object* v_inst_424_){
_start:
{
lean_inc_ref(v_inst_424_);
return v_inst_424_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DomMulAct_instDivInvMonoidOfMulOpposite___redArg___boxed(lean_object* v_inst_425_){
_start:
{
lean_object* v_res_426_; 
v_res_426_ = lp_mathlib_DomMulAct_instDivInvMonoidOfMulOpposite___redArg(v_inst_425_);
lean_dec_ref(v_inst_425_);
return v_res_426_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DomMulAct_instDivInvMonoidOfMulOpposite(lean_object* v_M_427_, lean_object* v_inst_428_){
_start:
{
lean_inc_ref(v_inst_428_);
return v_inst_428_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DomMulAct_instDivInvMonoidOfMulOpposite___boxed(lean_object* v_M_429_, lean_object* v_inst_430_){
_start:
{
lean_object* v_res_431_; 
v_res_431_ = lp_mathlib_DomMulAct_instDivInvMonoidOfMulOpposite(v_M_429_, v_inst_430_);
lean_dec_ref(v_inst_430_);
return v_res_431_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DomAddAct_instSubNegAddMonoidOfAddOpposite___redArg(lean_object* v_inst_432_){
_start:
{
lean_inc_ref(v_inst_432_);
return v_inst_432_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DomAddAct_instSubNegAddMonoidOfAddOpposite___redArg___boxed(lean_object* v_inst_433_){
_start:
{
lean_object* v_res_434_; 
v_res_434_ = lp_mathlib_DomAddAct_instSubNegAddMonoidOfAddOpposite___redArg(v_inst_433_);
lean_dec_ref(v_inst_433_);
return v_res_434_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DomAddAct_instSubNegAddMonoidOfAddOpposite(lean_object* v_M_435_, lean_object* v_inst_436_){
_start:
{
lean_inc_ref(v_inst_436_);
return v_inst_436_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DomAddAct_instSubNegAddMonoidOfAddOpposite___boxed(lean_object* v_M_437_, lean_object* v_inst_438_){
_start:
{
lean_object* v_res_439_; 
v_res_439_ = lp_mathlib_DomAddAct_instSubNegAddMonoidOfAddOpposite(v_M_437_, v_inst_438_);
lean_dec_ref(v_inst_438_);
return v_res_439_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DomMulAct_instInvOneClassOfMulOpposite___redArg(lean_object* v_inst_440_){
_start:
{
lean_inc_ref(v_inst_440_);
return v_inst_440_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DomMulAct_instInvOneClassOfMulOpposite___redArg___boxed(lean_object* v_inst_441_){
_start:
{
lean_object* v_res_442_; 
v_res_442_ = lp_mathlib_DomMulAct_instInvOneClassOfMulOpposite___redArg(v_inst_441_);
lean_dec_ref(v_inst_441_);
return v_res_442_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DomMulAct_instInvOneClassOfMulOpposite(lean_object* v_M_443_, lean_object* v_inst_444_){
_start:
{
lean_inc_ref(v_inst_444_);
return v_inst_444_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DomMulAct_instInvOneClassOfMulOpposite___boxed(lean_object* v_M_445_, lean_object* v_inst_446_){
_start:
{
lean_object* v_res_447_; 
v_res_447_ = lp_mathlib_DomMulAct_instInvOneClassOfMulOpposite(v_M_445_, v_inst_446_);
lean_dec_ref(v_inst_446_);
return v_res_447_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DomAddAct_instNegZeroClassOfAddOpposite___redArg(lean_object* v_inst_448_){
_start:
{
lean_inc_ref(v_inst_448_);
return v_inst_448_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DomAddAct_instNegZeroClassOfAddOpposite___redArg___boxed(lean_object* v_inst_449_){
_start:
{
lean_object* v_res_450_; 
v_res_450_ = lp_mathlib_DomAddAct_instNegZeroClassOfAddOpposite___redArg(v_inst_449_);
lean_dec_ref(v_inst_449_);
return v_res_450_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DomAddAct_instNegZeroClassOfAddOpposite(lean_object* v_M_451_, lean_object* v_inst_452_){
_start:
{
lean_inc_ref(v_inst_452_);
return v_inst_452_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DomAddAct_instNegZeroClassOfAddOpposite___boxed(lean_object* v_M_453_, lean_object* v_inst_454_){
_start:
{
lean_object* v_res_455_; 
v_res_455_ = lp_mathlib_DomAddAct_instNegZeroClassOfAddOpposite(v_M_453_, v_inst_454_);
lean_dec_ref(v_inst_454_);
return v_res_455_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DomMulAct_instDivInvOneMonoidOfMulOpposite___redArg(lean_object* v_inst_456_){
_start:
{
lean_inc_ref(v_inst_456_);
return v_inst_456_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DomMulAct_instDivInvOneMonoidOfMulOpposite___redArg___boxed(lean_object* v_inst_457_){
_start:
{
lean_object* v_res_458_; 
v_res_458_ = lp_mathlib_DomMulAct_instDivInvOneMonoidOfMulOpposite___redArg(v_inst_457_);
lean_dec_ref(v_inst_457_);
return v_res_458_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DomMulAct_instDivInvOneMonoidOfMulOpposite(lean_object* v_M_459_, lean_object* v_inst_460_){
_start:
{
lean_inc_ref(v_inst_460_);
return v_inst_460_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DomMulAct_instDivInvOneMonoidOfMulOpposite___boxed(lean_object* v_M_461_, lean_object* v_inst_462_){
_start:
{
lean_object* v_res_463_; 
v_res_463_ = lp_mathlib_DomMulAct_instDivInvOneMonoidOfMulOpposite(v_M_461_, v_inst_462_);
lean_dec_ref(v_inst_462_);
return v_res_463_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DomAddAct_instSubNegZeroMonoidOfAddOpposite___redArg(lean_object* v_inst_464_){
_start:
{
lean_inc_ref(v_inst_464_);
return v_inst_464_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DomAddAct_instSubNegZeroMonoidOfAddOpposite___redArg___boxed(lean_object* v_inst_465_){
_start:
{
lean_object* v_res_466_; 
v_res_466_ = lp_mathlib_DomAddAct_instSubNegZeroMonoidOfAddOpposite___redArg(v_inst_465_);
lean_dec_ref(v_inst_465_);
return v_res_466_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DomAddAct_instSubNegZeroMonoidOfAddOpposite(lean_object* v_M_467_, lean_object* v_inst_468_){
_start:
{
lean_inc_ref(v_inst_468_);
return v_inst_468_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DomAddAct_instSubNegZeroMonoidOfAddOpposite___boxed(lean_object* v_M_469_, lean_object* v_inst_470_){
_start:
{
lean_object* v_res_471_; 
v_res_471_ = lp_mathlib_DomAddAct_instSubNegZeroMonoidOfAddOpposite(v_M_469_, v_inst_470_);
lean_dec_ref(v_inst_470_);
return v_res_471_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DomMulAct_instDivisionMonoidOfMulOpposite___redArg(lean_object* v_inst_472_){
_start:
{
lean_inc_ref(v_inst_472_);
return v_inst_472_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DomMulAct_instDivisionMonoidOfMulOpposite___redArg___boxed(lean_object* v_inst_473_){
_start:
{
lean_object* v_res_474_; 
v_res_474_ = lp_mathlib_DomMulAct_instDivisionMonoidOfMulOpposite___redArg(v_inst_473_);
lean_dec_ref(v_inst_473_);
return v_res_474_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DomMulAct_instDivisionMonoidOfMulOpposite(lean_object* v_M_475_, lean_object* v_inst_476_){
_start:
{
lean_inc_ref(v_inst_476_);
return v_inst_476_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DomMulAct_instDivisionMonoidOfMulOpposite___boxed(lean_object* v_M_477_, lean_object* v_inst_478_){
_start:
{
lean_object* v_res_479_; 
v_res_479_ = lp_mathlib_DomMulAct_instDivisionMonoidOfMulOpposite(v_M_477_, v_inst_478_);
lean_dec_ref(v_inst_478_);
return v_res_479_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DomAddAct_instSubtractionMonoidOfAddOpposite___redArg(lean_object* v_inst_480_){
_start:
{
lean_inc_ref(v_inst_480_);
return v_inst_480_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DomAddAct_instSubtractionMonoidOfAddOpposite___redArg___boxed(lean_object* v_inst_481_){
_start:
{
lean_object* v_res_482_; 
v_res_482_ = lp_mathlib_DomAddAct_instSubtractionMonoidOfAddOpposite___redArg(v_inst_481_);
lean_dec_ref(v_inst_481_);
return v_res_482_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DomAddAct_instSubtractionMonoidOfAddOpposite(lean_object* v_M_483_, lean_object* v_inst_484_){
_start:
{
lean_inc_ref(v_inst_484_);
return v_inst_484_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DomAddAct_instSubtractionMonoidOfAddOpposite___boxed(lean_object* v_M_485_, lean_object* v_inst_486_){
_start:
{
lean_object* v_res_487_; 
v_res_487_ = lp_mathlib_DomAddAct_instSubtractionMonoidOfAddOpposite(v_M_485_, v_inst_486_);
lean_dec_ref(v_inst_486_);
return v_res_487_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DomMulAct_instDivisionCommMonoidOfMulOpposite___redArg(lean_object* v_inst_488_){
_start:
{
lean_inc_ref(v_inst_488_);
return v_inst_488_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DomMulAct_instDivisionCommMonoidOfMulOpposite___redArg___boxed(lean_object* v_inst_489_){
_start:
{
lean_object* v_res_490_; 
v_res_490_ = lp_mathlib_DomMulAct_instDivisionCommMonoidOfMulOpposite___redArg(v_inst_489_);
lean_dec_ref(v_inst_489_);
return v_res_490_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DomMulAct_instDivisionCommMonoidOfMulOpposite(lean_object* v_M_491_, lean_object* v_inst_492_){
_start:
{
lean_inc_ref(v_inst_492_);
return v_inst_492_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DomMulAct_instDivisionCommMonoidOfMulOpposite___boxed(lean_object* v_M_493_, lean_object* v_inst_494_){
_start:
{
lean_object* v_res_495_; 
v_res_495_ = lp_mathlib_DomMulAct_instDivisionCommMonoidOfMulOpposite(v_M_493_, v_inst_494_);
lean_dec_ref(v_inst_494_);
return v_res_495_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DomAddAct_instDivisionAddCommMonoidOfAddOpposite___redArg(lean_object* v_inst_496_){
_start:
{
lean_inc_ref(v_inst_496_);
return v_inst_496_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DomAddAct_instDivisionAddCommMonoidOfAddOpposite___redArg___boxed(lean_object* v_inst_497_){
_start:
{
lean_object* v_res_498_; 
v_res_498_ = lp_mathlib_DomAddAct_instDivisionAddCommMonoidOfAddOpposite___redArg(v_inst_497_);
lean_dec_ref(v_inst_497_);
return v_res_498_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DomAddAct_instDivisionAddCommMonoidOfAddOpposite(lean_object* v_M_499_, lean_object* v_inst_500_){
_start:
{
lean_inc_ref(v_inst_500_);
return v_inst_500_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DomAddAct_instDivisionAddCommMonoidOfAddOpposite___boxed(lean_object* v_M_501_, lean_object* v_inst_502_){
_start:
{
lean_object* v_res_503_; 
v_res_503_ = lp_mathlib_DomAddAct_instDivisionAddCommMonoidOfAddOpposite(v_M_501_, v_inst_502_);
lean_dec_ref(v_inst_502_);
return v_res_503_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DomMulAct_instGroupOfMulOpposite___redArg(lean_object* v_inst_504_){
_start:
{
lean_inc_ref(v_inst_504_);
return v_inst_504_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DomMulAct_instGroupOfMulOpposite___redArg___boxed(lean_object* v_inst_505_){
_start:
{
lean_object* v_res_506_; 
v_res_506_ = lp_mathlib_DomMulAct_instGroupOfMulOpposite___redArg(v_inst_505_);
lean_dec_ref(v_inst_505_);
return v_res_506_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DomMulAct_instGroupOfMulOpposite(lean_object* v_M_507_, lean_object* v_inst_508_){
_start:
{
lean_inc_ref(v_inst_508_);
return v_inst_508_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DomMulAct_instGroupOfMulOpposite___boxed(lean_object* v_M_509_, lean_object* v_inst_510_){
_start:
{
lean_object* v_res_511_; 
v_res_511_ = lp_mathlib_DomMulAct_instGroupOfMulOpposite(v_M_509_, v_inst_510_);
lean_dec_ref(v_inst_510_);
return v_res_511_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DomAddAct_instAddGroupOfAddOpposite___redArg(lean_object* v_inst_512_){
_start:
{
lean_inc_ref(v_inst_512_);
return v_inst_512_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DomAddAct_instAddGroupOfAddOpposite___redArg___boxed(lean_object* v_inst_513_){
_start:
{
lean_object* v_res_514_; 
v_res_514_ = lp_mathlib_DomAddAct_instAddGroupOfAddOpposite___redArg(v_inst_513_);
lean_dec_ref(v_inst_513_);
return v_res_514_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DomAddAct_instAddGroupOfAddOpposite(lean_object* v_M_515_, lean_object* v_inst_516_){
_start:
{
lean_inc_ref(v_inst_516_);
return v_inst_516_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DomAddAct_instAddGroupOfAddOpposite___boxed(lean_object* v_M_517_, lean_object* v_inst_518_){
_start:
{
lean_object* v_res_519_; 
v_res_519_ = lp_mathlib_DomAddAct_instAddGroupOfAddOpposite(v_M_517_, v_inst_518_);
lean_dec_ref(v_inst_518_);
return v_res_519_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DomMulAct_instCommGroupOfMulOpposite___redArg(lean_object* v_inst_520_){
_start:
{
lean_inc_ref(v_inst_520_);
return v_inst_520_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DomMulAct_instCommGroupOfMulOpposite___redArg___boxed(lean_object* v_inst_521_){
_start:
{
lean_object* v_res_522_; 
v_res_522_ = lp_mathlib_DomMulAct_instCommGroupOfMulOpposite___redArg(v_inst_521_);
lean_dec_ref(v_inst_521_);
return v_res_522_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DomMulAct_instCommGroupOfMulOpposite(lean_object* v_M_523_, lean_object* v_inst_524_){
_start:
{
lean_inc_ref(v_inst_524_);
return v_inst_524_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DomMulAct_instCommGroupOfMulOpposite___boxed(lean_object* v_M_525_, lean_object* v_inst_526_){
_start:
{
lean_object* v_res_527_; 
v_res_527_ = lp_mathlib_DomMulAct_instCommGroupOfMulOpposite(v_M_525_, v_inst_526_);
lean_dec_ref(v_inst_526_);
return v_res_527_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DomAddAct_instAddCommGroupOfAddOpposite___redArg(lean_object* v_inst_528_){
_start:
{
lean_inc_ref(v_inst_528_);
return v_inst_528_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DomAddAct_instAddCommGroupOfAddOpposite___redArg___boxed(lean_object* v_inst_529_){
_start:
{
lean_object* v_res_530_; 
v_res_530_ = lp_mathlib_DomAddAct_instAddCommGroupOfAddOpposite___redArg(v_inst_529_);
lean_dec_ref(v_inst_529_);
return v_res_530_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DomAddAct_instAddCommGroupOfAddOpposite(lean_object* v_M_531_, lean_object* v_inst_532_){
_start:
{
lean_inc_ref(v_inst_532_);
return v_inst_532_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DomAddAct_instAddCommGroupOfAddOpposite___boxed(lean_object* v_M_533_, lean_object* v_inst_534_){
_start:
{
lean_object* v_res_535_; 
v_res_535_ = lp_mathlib_DomAddAct_instAddCommGroupOfAddOpposite(v_M_533_, v_inst_534_);
lean_dec_ref(v_inst_534_);
return v_res_535_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DomMulAct_instNonAssocSemiringOfMulOpposite___redArg(lean_object* v_inst_536_){
_start:
{
lean_inc_ref(v_inst_536_);
return v_inst_536_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DomMulAct_instNonAssocSemiringOfMulOpposite___redArg___boxed(lean_object* v_inst_537_){
_start:
{
lean_object* v_res_538_; 
v_res_538_ = lp_mathlib_DomMulAct_instNonAssocSemiringOfMulOpposite___redArg(v_inst_537_);
lean_dec_ref(v_inst_537_);
return v_res_538_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DomMulAct_instNonAssocSemiringOfMulOpposite(lean_object* v_M_539_, lean_object* v_inst_540_){
_start:
{
lean_inc_ref(v_inst_540_);
return v_inst_540_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DomMulAct_instNonAssocSemiringOfMulOpposite___boxed(lean_object* v_M_541_, lean_object* v_inst_542_){
_start:
{
lean_object* v_res_543_; 
v_res_543_ = lp_mathlib_DomMulAct_instNonAssocSemiringOfMulOpposite(v_M_541_, v_inst_542_);
lean_dec_ref(v_inst_542_);
return v_res_543_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DomAddAct_instNonAssocSemiringOfAddOpposite___redArg(lean_object* v_inst_544_){
_start:
{
lean_inc_ref(v_inst_544_);
return v_inst_544_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DomAddAct_instNonAssocSemiringOfAddOpposite___redArg___boxed(lean_object* v_inst_545_){
_start:
{
lean_object* v_res_546_; 
v_res_546_ = lp_mathlib_DomAddAct_instNonAssocSemiringOfAddOpposite___redArg(v_inst_545_);
lean_dec_ref(v_inst_545_);
return v_res_546_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DomAddAct_instNonAssocSemiringOfAddOpposite(lean_object* v_M_547_, lean_object* v_inst_548_){
_start:
{
lean_inc_ref(v_inst_548_);
return v_inst_548_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DomAddAct_instNonAssocSemiringOfAddOpposite___boxed(lean_object* v_M_549_, lean_object* v_inst_550_){
_start:
{
lean_object* v_res_551_; 
v_res_551_ = lp_mathlib_DomAddAct_instNonAssocSemiringOfAddOpposite(v_M_549_, v_inst_550_);
lean_dec_ref(v_inst_550_);
return v_res_551_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DomMulAct_instNonUnitalSemiringOfMulOpposite___redArg(lean_object* v_inst_552_){
_start:
{
lean_inc_ref(v_inst_552_);
return v_inst_552_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DomMulAct_instNonUnitalSemiringOfMulOpposite___redArg___boxed(lean_object* v_inst_553_){
_start:
{
lean_object* v_res_554_; 
v_res_554_ = lp_mathlib_DomMulAct_instNonUnitalSemiringOfMulOpposite___redArg(v_inst_553_);
lean_dec_ref(v_inst_553_);
return v_res_554_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DomMulAct_instNonUnitalSemiringOfMulOpposite(lean_object* v_M_555_, lean_object* v_inst_556_){
_start:
{
lean_inc_ref(v_inst_556_);
return v_inst_556_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DomMulAct_instNonUnitalSemiringOfMulOpposite___boxed(lean_object* v_M_557_, lean_object* v_inst_558_){
_start:
{
lean_object* v_res_559_; 
v_res_559_ = lp_mathlib_DomMulAct_instNonUnitalSemiringOfMulOpposite(v_M_557_, v_inst_558_);
lean_dec_ref(v_inst_558_);
return v_res_559_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DomAddAct_instNonUnitalSemiringOfAddOpposite___redArg(lean_object* v_inst_560_){
_start:
{
lean_inc_ref(v_inst_560_);
return v_inst_560_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DomAddAct_instNonUnitalSemiringOfAddOpposite___redArg___boxed(lean_object* v_inst_561_){
_start:
{
lean_object* v_res_562_; 
v_res_562_ = lp_mathlib_DomAddAct_instNonUnitalSemiringOfAddOpposite___redArg(v_inst_561_);
lean_dec_ref(v_inst_561_);
return v_res_562_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DomAddAct_instNonUnitalSemiringOfAddOpposite(lean_object* v_M_563_, lean_object* v_inst_564_){
_start:
{
lean_inc_ref(v_inst_564_);
return v_inst_564_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DomAddAct_instNonUnitalSemiringOfAddOpposite___boxed(lean_object* v_M_565_, lean_object* v_inst_566_){
_start:
{
lean_object* v_res_567_; 
v_res_567_ = lp_mathlib_DomAddAct_instNonUnitalSemiringOfAddOpposite(v_M_565_, v_inst_566_);
lean_dec_ref(v_inst_566_);
return v_res_567_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DomMulAct_instSemiringOfMulOpposite___redArg(lean_object* v_inst_568_){
_start:
{
lean_inc_ref(v_inst_568_);
return v_inst_568_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DomMulAct_instSemiringOfMulOpposite___redArg___boxed(lean_object* v_inst_569_){
_start:
{
lean_object* v_res_570_; 
v_res_570_ = lp_mathlib_DomMulAct_instSemiringOfMulOpposite___redArg(v_inst_569_);
lean_dec_ref(v_inst_569_);
return v_res_570_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DomMulAct_instSemiringOfMulOpposite(lean_object* v_M_571_, lean_object* v_inst_572_){
_start:
{
lean_inc_ref(v_inst_572_);
return v_inst_572_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DomMulAct_instSemiringOfMulOpposite___boxed(lean_object* v_M_573_, lean_object* v_inst_574_){
_start:
{
lean_object* v_res_575_; 
v_res_575_ = lp_mathlib_DomMulAct_instSemiringOfMulOpposite(v_M_573_, v_inst_574_);
lean_dec_ref(v_inst_574_);
return v_res_575_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DomAddAct_instSemiringOfAddOpposite___redArg(lean_object* v_inst_576_){
_start:
{
lean_inc_ref(v_inst_576_);
return v_inst_576_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DomAddAct_instSemiringOfAddOpposite___redArg___boxed(lean_object* v_inst_577_){
_start:
{
lean_object* v_res_578_; 
v_res_578_ = lp_mathlib_DomAddAct_instSemiringOfAddOpposite___redArg(v_inst_577_);
lean_dec_ref(v_inst_577_);
return v_res_578_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DomAddAct_instSemiringOfAddOpposite(lean_object* v_M_579_, lean_object* v_inst_580_){
_start:
{
lean_inc_ref(v_inst_580_);
return v_inst_580_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DomAddAct_instSemiringOfAddOpposite___boxed(lean_object* v_M_581_, lean_object* v_inst_582_){
_start:
{
lean_object* v_res_583_; 
v_res_583_ = lp_mathlib_DomAddAct_instSemiringOfAddOpposite(v_M_581_, v_inst_582_);
lean_dec_ref(v_inst_582_);
return v_res_583_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DomMulAct_instRingOfMulOpposite___redArg(lean_object* v_inst_584_){
_start:
{
lean_inc_ref(v_inst_584_);
return v_inst_584_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DomMulAct_instRingOfMulOpposite___redArg___boxed(lean_object* v_inst_585_){
_start:
{
lean_object* v_res_586_; 
v_res_586_ = lp_mathlib_DomMulAct_instRingOfMulOpposite___redArg(v_inst_585_);
lean_dec_ref(v_inst_585_);
return v_res_586_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DomMulAct_instRingOfMulOpposite(lean_object* v_M_587_, lean_object* v_inst_588_){
_start:
{
lean_inc_ref(v_inst_588_);
return v_inst_588_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DomMulAct_instRingOfMulOpposite___boxed(lean_object* v_M_589_, lean_object* v_inst_590_){
_start:
{
lean_object* v_res_591_; 
v_res_591_ = lp_mathlib_DomMulAct_instRingOfMulOpposite(v_M_589_, v_inst_590_);
lean_dec_ref(v_inst_590_);
return v_res_591_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DomAddAct_instRingOfAddOpposite___redArg(lean_object* v_inst_592_){
_start:
{
lean_inc_ref(v_inst_592_);
return v_inst_592_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DomAddAct_instRingOfAddOpposite___redArg___boxed(lean_object* v_inst_593_){
_start:
{
lean_object* v_res_594_; 
v_res_594_ = lp_mathlib_DomAddAct_instRingOfAddOpposite___redArg(v_inst_593_);
lean_dec_ref(v_inst_593_);
return v_res_594_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DomAddAct_instRingOfAddOpposite(lean_object* v_M_595_, lean_object* v_inst_596_){
_start:
{
lean_inc_ref(v_inst_596_);
return v_inst_596_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DomAddAct_instRingOfAddOpposite___boxed(lean_object* v_M_597_, lean_object* v_inst_598_){
_start:
{
lean_object* v_res_599_; 
v_res_599_ = lp_mathlib_DomAddAct_instRingOfAddOpposite(v_M_597_, v_inst_598_);
lean_dec_ref(v_inst_598_);
return v_res_599_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DomMulAct_instCommRingOfMulOpposite___redArg(lean_object* v_inst_600_){
_start:
{
lean_inc_ref(v_inst_600_);
return v_inst_600_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DomMulAct_instCommRingOfMulOpposite___redArg___boxed(lean_object* v_inst_601_){
_start:
{
lean_object* v_res_602_; 
v_res_602_ = lp_mathlib_DomMulAct_instCommRingOfMulOpposite___redArg(v_inst_601_);
lean_dec_ref(v_inst_601_);
return v_res_602_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DomMulAct_instCommRingOfMulOpposite(lean_object* v_M_603_, lean_object* v_inst_604_){
_start:
{
lean_inc_ref(v_inst_604_);
return v_inst_604_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DomMulAct_instCommRingOfMulOpposite___boxed(lean_object* v_M_605_, lean_object* v_inst_606_){
_start:
{
lean_object* v_res_607_; 
v_res_607_ = lp_mathlib_DomMulAct_instCommRingOfMulOpposite(v_M_605_, v_inst_606_);
lean_dec_ref(v_inst_606_);
return v_res_607_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DomAddAct_instCommRingOfAddOpposite___redArg(lean_object* v_inst_608_){
_start:
{
lean_inc_ref(v_inst_608_);
return v_inst_608_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DomAddAct_instCommRingOfAddOpposite___redArg___boxed(lean_object* v_inst_609_){
_start:
{
lean_object* v_res_610_; 
v_res_610_ = lp_mathlib_DomAddAct_instCommRingOfAddOpposite___redArg(v_inst_609_);
lean_dec_ref(v_inst_609_);
return v_res_610_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DomAddAct_instCommRingOfAddOpposite(lean_object* v_M_611_, lean_object* v_inst_612_){
_start:
{
lean_inc_ref(v_inst_612_);
return v_inst_612_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DomAddAct_instCommRingOfAddOpposite___boxed(lean_object* v_M_613_, lean_object* v_inst_614_){
_start:
{
lean_object* v_res_615_; 
v_res_615_ = lp_mathlib_DomAddAct_instCommRingOfAddOpposite(v_M_613_, v_inst_614_);
lean_dec_ref(v_inst_614_);
return v_res_615_;
}
}
static lean_object* _init_lp_mathlib_DomMulAct_instSMulForall___redArg___lam__0___closed__0(void){
_start:
{
lean_object* v___x_616_; lean_object* v___x_617_; 
v___x_616_ = lean_obj_once(&lp_mathlib_DomMulAct_mk___closed__0, &lp_mathlib_DomMulAct_mk___closed__0_once, _init_lp_mathlib_DomMulAct_mk___closed__0);
v___x_617_ = lp_mathlib_Equiv_symm___redArg(v___x_616_);
return v___x_617_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DomMulAct_instSMulForall___redArg___lam__0(lean_object* v_inst_618_, lean_object* v_c_619_, lean_object* v_f_620_, lean_object* v_a_621_){
_start:
{
lean_object* v___x_622_; lean_object* v_toFun_623_; lean_object* v___x_624_; lean_object* v___x_625_; lean_object* v___x_626_; 
v___x_622_ = lean_obj_once(&lp_mathlib_DomMulAct_instSMulForall___redArg___lam__0___closed__0, &lp_mathlib_DomMulAct_instSMulForall___redArg___lam__0___closed__0_once, _init_lp_mathlib_DomMulAct_instSMulForall___redArg___lam__0___closed__0);
v_toFun_623_ = lean_ctor_get(v___x_622_, 0);
lean_inc(v_toFun_623_);
v___x_624_ = lean_apply_1(v_toFun_623_, v_c_619_);
v___x_625_ = lean_apply_2(v_inst_618_, v___x_624_, v_a_621_);
v___x_626_ = lean_apply_1(v_f_620_, v___x_625_);
return v___x_626_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DomMulAct_instSMulForall___redArg(lean_object* v_inst_627_){
_start:
{
lean_object* v___f_628_; 
v___f_628_ = lean_alloc_closure((void*)(lp_mathlib_DomMulAct_instSMulForall___redArg___lam__0), 4, 1);
lean_closure_set(v___f_628_, 0, v_inst_627_);
return v___f_628_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DomMulAct_instSMulForall(lean_object* v_M_629_, lean_object* v_00_u03b2_630_, lean_object* v_00_u03b1_631_, lean_object* v_inst_632_){
_start:
{
lean_object* v___f_633_; 
v___f_633_ = lean_alloc_closure((void*)(lp_mathlib_DomMulAct_instSMulForall___redArg___lam__0), 4, 1);
lean_closure_set(v___f_633_, 0, v_inst_632_);
return v___f_633_;
}
}
static lean_object* _init_lp_mathlib_DomAddAct_instVAddForall___redArg___lam__0___closed__0(void){
_start:
{
lean_object* v___x_634_; lean_object* v___x_635_; 
v___x_634_ = lean_obj_once(&lp_mathlib_DomAddAct_mk___closed__0, &lp_mathlib_DomAddAct_mk___closed__0_once, _init_lp_mathlib_DomAddAct_mk___closed__0);
v___x_635_ = lp_mathlib_Equiv_symm___redArg(v___x_634_);
return v___x_635_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DomAddAct_instVAddForall___redArg___lam__0(lean_object* v_inst_636_, lean_object* v_c_637_, lean_object* v_f_638_, lean_object* v_a_639_){
_start:
{
lean_object* v___x_640_; lean_object* v_toFun_641_; lean_object* v___x_642_; lean_object* v___x_643_; lean_object* v___x_644_; 
v___x_640_ = lean_obj_once(&lp_mathlib_DomAddAct_instVAddForall___redArg___lam__0___closed__0, &lp_mathlib_DomAddAct_instVAddForall___redArg___lam__0___closed__0_once, _init_lp_mathlib_DomAddAct_instVAddForall___redArg___lam__0___closed__0);
v_toFun_641_ = lean_ctor_get(v___x_640_, 0);
lean_inc(v_toFun_641_);
v___x_642_ = lean_apply_1(v_toFun_641_, v_c_637_);
v___x_643_ = lean_apply_2(v_inst_636_, v___x_642_, v_a_639_);
v___x_644_ = lean_apply_1(v_f_638_, v___x_643_);
return v___x_644_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DomAddAct_instVAddForall___redArg(lean_object* v_inst_645_){
_start:
{
lean_object* v___f_646_; 
v___f_646_ = lean_alloc_closure((void*)(lp_mathlib_DomAddAct_instVAddForall___redArg___lam__0), 4, 1);
lean_closure_set(v___f_646_, 0, v_inst_645_);
return v___f_646_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DomAddAct_instVAddForall(lean_object* v_M_647_, lean_object* v_00_u03b2_648_, lean_object* v_00_u03b1_649_, lean_object* v_inst_650_){
_start:
{
lean_object* v___f_651_; 
v___f_651_ = lean_alloc_closure((void*)(lp_mathlib_DomAddAct_instVAddForall___redArg___lam__0), 4, 1);
lean_closure_set(v___f_651_, 0, v_inst_650_);
return v___f_651_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DomMulAct_instSMulZeroClassForallOfSMul___redArg(lean_object* v_inst_652_){
_start:
{
lean_object* v___f_653_; 
v___f_653_ = lean_alloc_closure((void*)(lp_mathlib_DomMulAct_instSMulForall___redArg___lam__0), 4, 1);
lean_closure_set(v___f_653_, 0, v_inst_652_);
return v___f_653_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DomMulAct_instSMulZeroClassForallOfSMul(lean_object* v_M_654_, lean_object* v_00_u03b2_655_, lean_object* v_00_u03b1_656_, lean_object* v_inst_657_, lean_object* v_inst_658_){
_start:
{
lean_object* v___f_659_; 
v___f_659_ = lean_alloc_closure((void*)(lp_mathlib_DomMulAct_instSMulForall___redArg___lam__0), 4, 1);
lean_closure_set(v___f_659_, 0, v_inst_657_);
return v___f_659_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DomMulAct_instSMulZeroClassForallOfSMul___boxed(lean_object* v_M_660_, lean_object* v_00_u03b2_661_, lean_object* v_00_u03b1_662_, lean_object* v_inst_663_, lean_object* v_inst_664_){
_start:
{
lean_object* v_res_665_; 
v_res_665_ = lp_mathlib_DomMulAct_instSMulZeroClassForallOfSMul(v_M_660_, v_00_u03b2_661_, v_00_u03b1_662_, v_inst_663_, v_inst_664_);
lean_dec(v_inst_664_);
return v_res_665_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DomMulAct_instDistribSMulForallOfSMul___redArg(lean_object* v_inst_666_){
_start:
{
lean_object* v___f_667_; 
v___f_667_ = lean_alloc_closure((void*)(lp_mathlib_DomMulAct_instSMulForall___redArg___lam__0), 4, 1);
lean_closure_set(v___f_667_, 0, v_inst_666_);
return v___f_667_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DomMulAct_instDistribSMulForallOfSMul(lean_object* v_M_668_, lean_object* v_00_u03b1_669_, lean_object* v_A_670_, lean_object* v_inst_671_, lean_object* v_inst_672_){
_start:
{
lean_object* v___f_673_; 
v___f_673_ = lean_alloc_closure((void*)(lp_mathlib_DomMulAct_instSMulForall___redArg___lam__0), 4, 1);
lean_closure_set(v___f_673_, 0, v_inst_671_);
return v___f_673_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DomMulAct_instDistribSMulForallOfSMul___boxed(lean_object* v_M_674_, lean_object* v_00_u03b1_675_, lean_object* v_A_676_, lean_object* v_inst_677_, lean_object* v_inst_678_){
_start:
{
lean_object* v_res_679_; 
v_res_679_ = lp_mathlib_DomMulAct_instDistribSMulForallOfSMul(v_M_674_, v_00_u03b1_675_, v_A_676_, v_inst_677_, v_inst_678_);
lean_dec_ref(v_inst_678_);
return v_res_679_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DomMulAct_instMulActionForall___redArg(lean_object* v_inst_680_){
_start:
{
lean_object* v___f_681_; 
v___f_681_ = lean_alloc_closure((void*)(lp_mathlib_DomMulAct_instSMulForall___redArg___lam__0), 4, 1);
lean_closure_set(v___f_681_, 0, v_inst_680_);
return v___f_681_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DomMulAct_instMulActionForall(lean_object* v_M_682_, lean_object* v_00_u03b2_683_, lean_object* v_00_u03b1_684_, lean_object* v_inst_685_, lean_object* v_inst_686_){
_start:
{
lean_object* v___f_687_; 
v___f_687_ = lean_alloc_closure((void*)(lp_mathlib_DomMulAct_instSMulForall___redArg___lam__0), 4, 1);
lean_closure_set(v___f_687_, 0, v_inst_686_);
return v___f_687_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DomMulAct_instMulActionForall___boxed(lean_object* v_M_688_, lean_object* v_00_u03b2_689_, lean_object* v_00_u03b1_690_, lean_object* v_inst_691_, lean_object* v_inst_692_){
_start:
{
lean_object* v_res_693_; 
v_res_693_ = lp_mathlib_DomMulAct_instMulActionForall(v_M_688_, v_00_u03b2_689_, v_00_u03b1_690_, v_inst_691_, v_inst_692_);
lean_dec_ref(v_inst_691_);
return v_res_693_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DomAddAct_instAddActionForall___redArg(lean_object* v_inst_694_){
_start:
{
lean_object* v___f_695_; 
v___f_695_ = lean_alloc_closure((void*)(lp_mathlib_DomAddAct_instVAddForall___redArg___lam__0), 4, 1);
lean_closure_set(v___f_695_, 0, v_inst_694_);
return v___f_695_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DomAddAct_instAddActionForall(lean_object* v_M_696_, lean_object* v_00_u03b2_697_, lean_object* v_00_u03b1_698_, lean_object* v_inst_699_, lean_object* v_inst_700_){
_start:
{
lean_object* v___f_701_; 
v___f_701_ = lean_alloc_closure((void*)(lp_mathlib_DomAddAct_instVAddForall___redArg___lam__0), 4, 1);
lean_closure_set(v___f_701_, 0, v_inst_700_);
return v___f_701_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DomAddAct_instAddActionForall___boxed(lean_object* v_M_702_, lean_object* v_00_u03b2_703_, lean_object* v_00_u03b1_704_, lean_object* v_inst_705_, lean_object* v_inst_706_){
_start:
{
lean_object* v_res_707_; 
v_res_707_ = lp_mathlib_DomAddAct_instAddActionForall(v_M_702_, v_00_u03b2_703_, v_00_u03b1_704_, v_inst_705_, v_inst_706_);
lean_dec_ref(v_inst_705_);
return v_res_707_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DomMulAct_instDistribMulActionForallOfMulAction___redArg(lean_object* v_inst_708_){
_start:
{
lean_object* v___f_709_; 
v___f_709_ = lean_alloc_closure((void*)(lp_mathlib_DomMulAct_instSMulForall___redArg___lam__0), 4, 1);
lean_closure_set(v___f_709_, 0, v_inst_708_);
return v___f_709_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DomMulAct_instDistribMulActionForallOfMulAction(lean_object* v_M_710_, lean_object* v_00_u03b1_711_, lean_object* v_A_712_, lean_object* v_inst_713_, lean_object* v_inst_714_, lean_object* v_inst_715_){
_start:
{
lean_object* v___f_716_; 
v___f_716_ = lean_alloc_closure((void*)(lp_mathlib_DomMulAct_instSMulForall___redArg___lam__0), 4, 1);
lean_closure_set(v___f_716_, 0, v_inst_714_);
return v___f_716_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DomMulAct_instDistribMulActionForallOfMulAction___boxed(lean_object* v_M_717_, lean_object* v_00_u03b1_718_, lean_object* v_A_719_, lean_object* v_inst_720_, lean_object* v_inst_721_, lean_object* v_inst_722_){
_start:
{
lean_object* v_res_723_; 
v_res_723_ = lp_mathlib_DomMulAct_instDistribMulActionForallOfMulAction(v_M_717_, v_00_u03b1_718_, v_A_719_, v_inst_720_, v_inst_721_, v_inst_722_);
lean_dec_ref(v_inst_722_);
lean_dec_ref(v_inst_720_);
return v_res_723_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DomMulAct_instMulDistribMulActionForallOfMulAction___redArg(lean_object* v_inst_724_){
_start:
{
lean_object* v___f_725_; 
v___f_725_ = lean_alloc_closure((void*)(lp_mathlib_DomMulAct_instSMulForall___redArg___lam__0), 4, 1);
lean_closure_set(v___f_725_, 0, v_inst_724_);
return v___f_725_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DomMulAct_instMulDistribMulActionForallOfMulAction(lean_object* v_M_726_, lean_object* v_00_u03b1_727_, lean_object* v_A_728_, lean_object* v_inst_729_, lean_object* v_inst_730_, lean_object* v_inst_731_){
_start:
{
lean_object* v___f_732_; 
v___f_732_ = lean_alloc_closure((void*)(lp_mathlib_DomMulAct_instSMulForall___redArg___lam__0), 4, 1);
lean_closure_set(v___f_732_, 0, v_inst_730_);
return v___f_732_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DomMulAct_instMulDistribMulActionForallOfMulAction___boxed(lean_object* v_M_733_, lean_object* v_00_u03b1_734_, lean_object* v_A_735_, lean_object* v_inst_736_, lean_object* v_inst_737_, lean_object* v_inst_738_){
_start:
{
lean_object* v_res_739_; 
v_res_739_ = lp_mathlib_DomMulAct_instMulDistribMulActionForallOfMulAction(v_M_733_, v_00_u03b1_734_, v_A_735_, v_inst_736_, v_inst_737_, v_inst_738_);
lean_dec_ref(v_inst_738_);
lean_dec_ref(v_inst_736_);
return v_res_739_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DomMulAct_instSMulMonoidHom___redArg___lam__0(lean_object* v_inst_740_, lean_object* v_c_741_, lean_object* v_f_742_, lean_object* v___y_743_){
_start:
{
lean_object* v___x_744_; lean_object* v_toFun_745_; lean_object* v___x_746_; lean_object* v___f_747_; lean_object* v___x_748_; 
v___x_744_ = lean_obj_once(&lp_mathlib_DomMulAct_instSMulForall___redArg___lam__0___closed__0, &lp_mathlib_DomMulAct_instSMulForall___redArg___lam__0___closed__0_once, _init_lp_mathlib_DomMulAct_instSMulForall___redArg___lam__0___closed__0);
v_toFun_745_ = lean_ctor_get(v___x_744_, 0);
lean_inc(v_toFun_745_);
v___x_746_ = lean_apply_1(v_toFun_745_, v_c_741_);
v___f_747_ = lean_alloc_closure((void*)(lp_mathlib_MulDistribMulAction_toMonoidHom___redArg___lam__0), 3, 2);
lean_closure_set(v___f_747_, 0, v_inst_740_);
lean_closure_set(v___f_747_, 1, v___x_746_);
v___x_748_ = lp_mathlib_OneHom_comp___redArg___lam__0(v___f_747_, v_f_742_, v___y_743_);
return v___x_748_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DomMulAct_instSMulMonoidHom___redArg(lean_object* v_inst_749_){
_start:
{
lean_object* v___f_750_; 
v___f_750_ = lean_alloc_closure((void*)(lp_mathlib_DomMulAct_instSMulMonoidHom___redArg___lam__0), 4, 1);
lean_closure_set(v___f_750_, 0, v_inst_749_);
return v___f_750_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DomMulAct_instSMulMonoidHom(lean_object* v_M_751_, lean_object* v_A_752_, lean_object* v_B_753_, lean_object* v_inst_754_, lean_object* v_inst_755_, lean_object* v_inst_756_, lean_object* v_inst_757_){
_start:
{
lean_object* v___f_758_; 
v___f_758_ = lean_alloc_closure((void*)(lp_mathlib_DomMulAct_instSMulMonoidHom___redArg___lam__0), 4, 1);
lean_closure_set(v___f_758_, 0, v_inst_756_);
return v___f_758_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DomMulAct_instSMulMonoidHom___boxed(lean_object* v_M_759_, lean_object* v_A_760_, lean_object* v_B_761_, lean_object* v_inst_762_, lean_object* v_inst_763_, lean_object* v_inst_764_, lean_object* v_inst_765_){
_start:
{
lean_object* v_res_766_; 
v_res_766_ = lp_mathlib_DomMulAct_instSMulMonoidHom(v_M_759_, v_A_760_, v_B_761_, v_inst_762_, v_inst_763_, v_inst_764_, v_inst_765_);
lean_dec_ref(v_inst_765_);
lean_dec_ref(v_inst_763_);
lean_dec_ref(v_inst_762_);
return v_res_766_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DomMulAct_instMulActionMonoidHom___redArg(lean_object* v_inst_767_){
_start:
{
lean_object* v___f_768_; 
v___f_768_ = lean_alloc_closure((void*)(lp_mathlib_DomMulAct_instSMulMonoidHom___redArg___lam__0), 4, 1);
lean_closure_set(v___f_768_, 0, v_inst_767_);
return v___f_768_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DomMulAct_instMulActionMonoidHom(lean_object* v_M_769_, lean_object* v_A_770_, lean_object* v_B_771_, lean_object* v_inst_772_, lean_object* v_inst_773_, lean_object* v_inst_774_, lean_object* v_inst_775_){
_start:
{
lean_object* v___f_776_; 
v___f_776_ = lean_alloc_closure((void*)(lp_mathlib_DomMulAct_instSMulMonoidHom___redArg___lam__0), 4, 1);
lean_closure_set(v___f_776_, 0, v_inst_774_);
return v___f_776_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DomMulAct_instMulActionMonoidHom___boxed(lean_object* v_M_777_, lean_object* v_A_778_, lean_object* v_B_779_, lean_object* v_inst_780_, lean_object* v_inst_781_, lean_object* v_inst_782_, lean_object* v_inst_783_){
_start:
{
lean_object* v_res_784_; 
v_res_784_ = lp_mathlib_DomMulAct_instMulActionMonoidHom(v_M_777_, v_A_778_, v_B_779_, v_inst_780_, v_inst_781_, v_inst_782_, v_inst_783_);
lean_dec_ref(v_inst_783_);
lean_dec_ref(v_inst_781_);
lean_dec_ref(v_inst_780_);
return v_res_784_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DomMulAct_instSMulAddMonoidHom___redArg___lam__0(lean_object* v_inst_785_, lean_object* v_c_786_, lean_object* v_f_787_, lean_object* v___y_788_){
_start:
{
lean_object* v___x_789_; lean_object* v_toFun_790_; lean_object* v___x_791_; lean_object* v___f_792_; lean_object* v___x_793_; 
v___x_789_ = lean_obj_once(&lp_mathlib_DomMulAct_instSMulForall___redArg___lam__0___closed__0, &lp_mathlib_DomMulAct_instSMulForall___redArg___lam__0___closed__0_once, _init_lp_mathlib_DomMulAct_instSMulForall___redArg___lam__0___closed__0);
v_toFun_790_ = lean_ctor_get(v___x_789_, 0);
lean_inc(v_toFun_790_);
v___x_791_ = lean_apply_1(v_toFun_790_, v_c_786_);
v___f_792_ = lean_alloc_closure((void*)(lp_mathlib_SMulZeroClass_toZeroHom___redArg___lam__0), 3, 2);
lean_closure_set(v___f_792_, 0, v_inst_785_);
lean_closure_set(v___f_792_, 1, v___x_791_);
v___x_793_ = lp_mathlib_OneHom_comp___redArg___lam__0(v___f_792_, v_f_787_, v___y_788_);
return v___x_793_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DomMulAct_instSMulAddMonoidHom___redArg(lean_object* v_inst_794_){
_start:
{
lean_object* v___f_795_; 
v___f_795_ = lean_alloc_closure((void*)(lp_mathlib_DomMulAct_instSMulAddMonoidHom___redArg___lam__0), 4, 1);
lean_closure_set(v___f_795_, 0, v_inst_794_);
return v___f_795_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DomMulAct_instSMulAddMonoidHom(lean_object* v_A_796_, lean_object* v_B_797_, lean_object* v_M_798_, lean_object* v_inst_799_, lean_object* v_inst_800_, lean_object* v_inst_801_){
_start:
{
lean_object* v___f_802_; 
v___f_802_ = lean_alloc_closure((void*)(lp_mathlib_DomMulAct_instSMulAddMonoidHom___redArg___lam__0), 4, 1);
lean_closure_set(v___f_802_, 0, v_inst_800_);
return v___f_802_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DomMulAct_instSMulAddMonoidHom___boxed(lean_object* v_A_803_, lean_object* v_B_804_, lean_object* v_M_805_, lean_object* v_inst_806_, lean_object* v_inst_807_, lean_object* v_inst_808_){
_start:
{
lean_object* v_res_809_; 
v_res_809_ = lp_mathlib_DomMulAct_instSMulAddMonoidHom(v_A_803_, v_B_804_, v_M_805_, v_inst_806_, v_inst_807_, v_inst_808_);
lean_dec_ref(v_inst_808_);
lean_dec_ref(v_inst_806_);
return v_res_809_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DomMulAct_instMulActionAddMonoidHomOfDistribMulAction___redArg(lean_object* v_inst_810_){
_start:
{
lean_object* v___f_811_; 
v___f_811_ = lean_alloc_closure((void*)(lp_mathlib_DomMulAct_instSMulAddMonoidHom___redArg___lam__0), 4, 1);
lean_closure_set(v___f_811_, 0, v_inst_810_);
return v___f_811_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DomMulAct_instMulActionAddMonoidHomOfDistribMulAction(lean_object* v_A_812_, lean_object* v_M_813_, lean_object* v_B_814_, lean_object* v_inst_815_, lean_object* v_inst_816_, lean_object* v_inst_817_, lean_object* v_inst_818_){
_start:
{
lean_object* v___f_819_; 
v___f_819_ = lean_alloc_closure((void*)(lp_mathlib_DomMulAct_instSMulAddMonoidHom___redArg___lam__0), 4, 1);
lean_closure_set(v___f_819_, 0, v_inst_817_);
return v___f_819_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DomMulAct_instMulActionAddMonoidHomOfDistribMulAction___boxed(lean_object* v_A_820_, lean_object* v_M_821_, lean_object* v_B_822_, lean_object* v_inst_823_, lean_object* v_inst_824_, lean_object* v_inst_825_, lean_object* v_inst_826_){
_start:
{
lean_object* v_res_827_; 
v_res_827_ = lp_mathlib_DomMulAct_instMulActionAddMonoidHomOfDistribMulAction(v_A_820_, v_M_821_, v_B_822_, v_inst_823_, v_inst_824_, v_inst_825_, v_inst_826_);
lean_dec_ref(v_inst_826_);
lean_dec_ref(v_inst_824_);
lean_dec_ref(v_inst_823_);
return v_res_827_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DomMulAct_instDistribMulActionAddMonoidHom___redArg(lean_object* v_inst_828_){
_start:
{
lean_object* v___f_829_; 
v___f_829_ = lean_alloc_closure((void*)(lp_mathlib_DomMulAct_instSMulAddMonoidHom___redArg___lam__0), 4, 1);
lean_closure_set(v___f_829_, 0, v_inst_828_);
return v___f_829_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DomMulAct_instDistribMulActionAddMonoidHom(lean_object* v_A_830_, lean_object* v_M_831_, lean_object* v_B_832_, lean_object* v_inst_833_, lean_object* v_inst_834_, lean_object* v_inst_835_, lean_object* v_inst_836_){
_start:
{
lean_object* v___f_837_; 
v___f_837_ = lean_alloc_closure((void*)(lp_mathlib_DomMulAct_instSMulAddMonoidHom___redArg___lam__0), 4, 1);
lean_closure_set(v___f_837_, 0, v_inst_835_);
return v___f_837_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DomMulAct_instDistribMulActionAddMonoidHom___boxed(lean_object* v_A_838_, lean_object* v_M_839_, lean_object* v_B_840_, lean_object* v_inst_841_, lean_object* v_inst_842_, lean_object* v_inst_843_, lean_object* v_inst_844_){
_start:
{
lean_object* v_res_845_; 
v_res_845_ = lp_mathlib_DomMulAct_instDistribMulActionAddMonoidHom(v_A_838_, v_M_839_, v_B_840_, v_inst_841_, v_inst_842_, v_inst_843_, v_inst_844_);
lean_dec_ref(v_inst_844_);
lean_dec_ref(v_inst_842_);
lean_dec_ref(v_inst_841_);
return v_res_845_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DomMulAct_instMulDistribMulActionMonoidHom___redArg(lean_object* v_inst_846_){
_start:
{
lean_object* v___f_847_; 
v___f_847_ = lean_alloc_closure((void*)(lp_mathlib_DomMulAct_instSMulMonoidHom___redArg___lam__0), 4, 1);
lean_closure_set(v___f_847_, 0, v_inst_846_);
return v___f_847_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DomMulAct_instMulDistribMulActionMonoidHom(lean_object* v_A_848_, lean_object* v_M_849_, lean_object* v_B_850_, lean_object* v_inst_851_, lean_object* v_inst_852_, lean_object* v_inst_853_, lean_object* v_inst_854_){
_start:
{
lean_object* v___f_855_; 
v___f_855_ = lean_alloc_closure((void*)(lp_mathlib_DomMulAct_instSMulMonoidHom___redArg___lam__0), 4, 1);
lean_closure_set(v___f_855_, 0, v_inst_853_);
return v___f_855_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DomMulAct_instMulDistribMulActionMonoidHom___boxed(lean_object* v_A_856_, lean_object* v_M_857_, lean_object* v_B_858_, lean_object* v_inst_859_, lean_object* v_inst_860_, lean_object* v_inst_861_, lean_object* v_inst_862_){
_start:
{
lean_object* v_res_863_; 
v_res_863_ = lp_mathlib_DomMulAct_instMulDistribMulActionMonoidHom(v_A_856_, v_M_857_, v_B_858_, v_inst_859_, v_inst_860_, v_inst_861_, v_inst_862_);
lean_dec_ref(v_inst_862_);
lean_dec_ref(v_inst_860_);
lean_dec_ref(v_inst_859_);
return v_res_863_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Group_Action_Basic(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Group_Opposite(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Group_Pi_Lemmas(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_GroupWithZero_Action_Hom(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Ring_Defs(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_GroupTheory_GroupAction_DomAct_Basic(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Group_Action_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Group_Opposite(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Group_Pi_Lemmas(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_GroupWithZero_Action_Hom(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Ring_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Tactic_ToDual(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_GroupTheory_GroupAction_DomAct_Basic(uint8_t builtin) {
lean_object * res;
if (_G_meta_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_meta_initialized = true;
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Tactic_ToDual(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Group_Action_Basic(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Group_Opposite(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Group_Pi_Lemmas(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_GroupWithZero_Action_Hom(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Ring_Defs(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Tactic_ToDual(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_GroupTheory_GroupAction_DomAct_Basic(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Group_Action_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Group_Opposite(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Group_Pi_Lemmas(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_GroupWithZero_Action_Hom(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Ring_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Tactic_ToDual(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_GroupTheory_GroupAction_DomAct_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_GroupTheory_GroupAction_DomAct_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_GroupTheory_GroupAction_DomAct_Basic(builtin);
}
#ifdef __cplusplus
}
#endif
