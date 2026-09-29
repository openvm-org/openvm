// Lean compiler output
// Module: Mathlib.Algebra.GroupWithZero.Units.Basic
// Imports: public import Init public meta import Init public import Mathlib.Algebra.Group.Units.Basic public import Mathlib.Algebra.GroupWithZero.Basic public import Mathlib.Data.Nat.Basic public import Mathlib.Lean.Meta.CongrTheorems public import Mathlib.Tactic.Contrapose public import Mathlib.Tactic.Spread public import Mathlib.Tactic.Convert public import Mathlib.Tactic.Nontriviality
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
lean_object* lp_mathlib_GroupWithZero_toDivInvMonoid___redArg(lean_object*);
lean_object* lp_mathlib_DivInvOneMonoid_toInvOneClass___redArg(lean_object*);
lean_object* l_Lean_Name_mkStr4(lean_object*, lean_object*, lean_object*, lean_object*);
uint8_t l_Lean_Syntax_isOfKind(lean_object*, lean_object*);
lean_object* l_Lean_Syntax_getArg(lean_object*, lean_object*);
lean_object* l_Lean_Name_mkStr1(lean_object*);
uint8_t l_Lean_Syntax_matchesNull(lean_object*, lean_object*);
lean_object* l_Lean_replaceRef(lean_object*, lean_object*);
lean_object* l_Lean_SourceInfo_fromRef(lean_object*, uint8_t);
lean_object* l_Lean_Name_mkStr2(lean_object*, lean_object*);
lean_object* l_Lean_Syntax_node2(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_String_toRawSubstring_x27(lean_object*);
lean_object* l_Lean_addMacroScope(lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Syntax_node1(lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Ring_term___u207b_xb9_u02b3___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "Ring"};
static const lean_object* lp_mathlib_Ring_term___u207b_xb9_u02b3___closed__0 = (const lean_object*)&lp_mathlib_Ring_term___u207b_xb9_u02b3___closed__0_value;
static const lean_string_object lp_mathlib_Ring_term___u207b_xb9_u02b3___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 13, .m_capacity = 13, .m_length = 8, .m_data = "term_⁻¹ʳ"};
static const lean_object* lp_mathlib_Ring_term___u207b_xb9_u02b3___closed__1 = (const lean_object*)&lp_mathlib_Ring_term___u207b_xb9_u02b3___closed__1_value;
static const lean_ctor_object lp_mathlib_Ring_term___u207b_xb9_u02b3___closed__2_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Ring_term___u207b_xb9_u02b3___closed__0_value),LEAN_SCALAR_PTR_LITERAL(151, 9, 120, 97, 235, 184, 251, 227)}};
static const lean_ctor_object lp_mathlib_Ring_term___u207b_xb9_u02b3___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Ring_term___u207b_xb9_u02b3___closed__2_value_aux_0),((lean_object*)&lp_mathlib_Ring_term___u207b_xb9_u02b3___closed__1_value),LEAN_SCALAR_PTR_LITERAL(176, 171, 160, 8, 192, 1, 207, 13)}};
static const lean_object* lp_mathlib_Ring_term___u207b_xb9_u02b3___closed__2 = (const lean_object*)&lp_mathlib_Ring_term___u207b_xb9_u02b3___closed__2_value;
static const lean_string_object lp_mathlib_Ring_term___u207b_xb9_u02b3___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 3, .m_data = "⁻¹ʳ"};
static const lean_object* lp_mathlib_Ring_term___u207b_xb9_u02b3___closed__3 = (const lean_object*)&lp_mathlib_Ring_term___u207b_xb9_u02b3___closed__3_value;
static const lean_ctor_object lp_mathlib_Ring_term___u207b_xb9_u02b3___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 5}, .m_objs = {((lean_object*)&lp_mathlib_Ring_term___u207b_xb9_u02b3___closed__3_value)}};
static const lean_object* lp_mathlib_Ring_term___u207b_xb9_u02b3___closed__4 = (const lean_object*)&lp_mathlib_Ring_term___u207b_xb9_u02b3___closed__4_value;
static const lean_ctor_object lp_mathlib_Ring_term___u207b_xb9_u02b3___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*4 + 0, .m_other = 4, .m_tag = 4}, .m_objs = {((lean_object*)&lp_mathlib_Ring_term___u207b_xb9_u02b3___closed__2_value),((lean_object*)(((size_t)(1024) << 1) | 1)),((lean_object*)(((size_t)(1024) << 1) | 1)),((lean_object*)&lp_mathlib_Ring_term___u207b_xb9_u02b3___closed__4_value)}};
static const lean_object* lp_mathlib_Ring_term___u207b_xb9_u02b3___closed__5 = (const lean_object*)&lp_mathlib_Ring_term___u207b_xb9_u02b3___closed__5_value;
LEAN_EXPORT const lean_object* lp_mathlib_Ring_term___u207b_xb9_u02b3 = (const lean_object*)&lp_mathlib_Ring_term___u207b_xb9_u02b3___closed__5_value;
static const lean_string_object lp_mathlib_Ring___aux__Mathlib__Algebra__GroupWithZero__Units__Basic______macroRules__Ring__term___u207b_xb9_u02b3__1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "Lean"};
static const lean_object* lp_mathlib_Ring___aux__Mathlib__Algebra__GroupWithZero__Units__Basic______macroRules__Ring__term___u207b_xb9_u02b3__1___closed__0 = (const lean_object*)&lp_mathlib_Ring___aux__Mathlib__Algebra__GroupWithZero__Units__Basic______macroRules__Ring__term___u207b_xb9_u02b3__1___closed__0_value;
static const lean_string_object lp_mathlib_Ring___aux__Mathlib__Algebra__GroupWithZero__Units__Basic______macroRules__Ring__term___u207b_xb9_u02b3__1___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "Parser"};
static const lean_object* lp_mathlib_Ring___aux__Mathlib__Algebra__GroupWithZero__Units__Basic______macroRules__Ring__term___u207b_xb9_u02b3__1___closed__1 = (const lean_object*)&lp_mathlib_Ring___aux__Mathlib__Algebra__GroupWithZero__Units__Basic______macroRules__Ring__term___u207b_xb9_u02b3__1___closed__1_value;
static const lean_string_object lp_mathlib_Ring___aux__Mathlib__Algebra__GroupWithZero__Units__Basic______macroRules__Ring__term___u207b_xb9_u02b3__1___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "Term"};
static const lean_object* lp_mathlib_Ring___aux__Mathlib__Algebra__GroupWithZero__Units__Basic______macroRules__Ring__term___u207b_xb9_u02b3__1___closed__2 = (const lean_object*)&lp_mathlib_Ring___aux__Mathlib__Algebra__GroupWithZero__Units__Basic______macroRules__Ring__term___u207b_xb9_u02b3__1___closed__2_value;
static const lean_string_object lp_mathlib_Ring___aux__Mathlib__Algebra__GroupWithZero__Units__Basic______macroRules__Ring__term___u207b_xb9_u02b3__1___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = "app"};
static const lean_object* lp_mathlib_Ring___aux__Mathlib__Algebra__GroupWithZero__Units__Basic______macroRules__Ring__term___u207b_xb9_u02b3__1___closed__3 = (const lean_object*)&lp_mathlib_Ring___aux__Mathlib__Algebra__GroupWithZero__Units__Basic______macroRules__Ring__term___u207b_xb9_u02b3__1___closed__3_value;
static const lean_ctor_object lp_mathlib_Ring___aux__Mathlib__Algebra__GroupWithZero__Units__Basic______macroRules__Ring__term___u207b_xb9_u02b3__1___closed__4_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Ring___aux__Mathlib__Algebra__GroupWithZero__Units__Basic______macroRules__Ring__term___u207b_xb9_u02b3__1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib_Ring___aux__Mathlib__Algebra__GroupWithZero__Units__Basic______macroRules__Ring__term___u207b_xb9_u02b3__1___closed__4_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Ring___aux__Mathlib__Algebra__GroupWithZero__Units__Basic______macroRules__Ring__term___u207b_xb9_u02b3__1___closed__4_value_aux_0),((lean_object*)&lp_mathlib_Ring___aux__Mathlib__Algebra__GroupWithZero__Units__Basic______macroRules__Ring__term___u207b_xb9_u02b3__1___closed__1_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib_Ring___aux__Mathlib__Algebra__GroupWithZero__Units__Basic______macroRules__Ring__term___u207b_xb9_u02b3__1___closed__4_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Ring___aux__Mathlib__Algebra__GroupWithZero__Units__Basic______macroRules__Ring__term___u207b_xb9_u02b3__1___closed__4_value_aux_1),((lean_object*)&lp_mathlib_Ring___aux__Mathlib__Algebra__GroupWithZero__Units__Basic______macroRules__Ring__term___u207b_xb9_u02b3__1___closed__2_value),LEAN_SCALAR_PTR_LITERAL(75, 170, 162, 138, 136, 204, 251, 229)}};
static const lean_ctor_object lp_mathlib_Ring___aux__Mathlib__Algebra__GroupWithZero__Units__Basic______macroRules__Ring__term___u207b_xb9_u02b3__1___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Ring___aux__Mathlib__Algebra__GroupWithZero__Units__Basic______macroRules__Ring__term___u207b_xb9_u02b3__1___closed__4_value_aux_2),((lean_object*)&lp_mathlib_Ring___aux__Mathlib__Algebra__GroupWithZero__Units__Basic______macroRules__Ring__term___u207b_xb9_u02b3__1___closed__3_value),LEAN_SCALAR_PTR_LITERAL(69, 118, 10, 41, 220, 156, 243, 179)}};
static const lean_object* lp_mathlib_Ring___aux__Mathlib__Algebra__GroupWithZero__Units__Basic______macroRules__Ring__term___u207b_xb9_u02b3__1___closed__4 = (const lean_object*)&lp_mathlib_Ring___aux__Mathlib__Algebra__GroupWithZero__Units__Basic______macroRules__Ring__term___u207b_xb9_u02b3__1___closed__4_value;
static const lean_string_object lp_mathlib_Ring___aux__Mathlib__Algebra__GroupWithZero__Units__Basic______macroRules__Ring__term___u207b_xb9_u02b3__1___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "inverse"};
static const lean_object* lp_mathlib_Ring___aux__Mathlib__Algebra__GroupWithZero__Units__Basic______macroRules__Ring__term___u207b_xb9_u02b3__1___closed__5 = (const lean_object*)&lp_mathlib_Ring___aux__Mathlib__Algebra__GroupWithZero__Units__Basic______macroRules__Ring__term___u207b_xb9_u02b3__1___closed__5_value;
static lean_once_cell_t lp_mathlib_Ring___aux__Mathlib__Algebra__GroupWithZero__Units__Basic______macroRules__Ring__term___u207b_xb9_u02b3__1___closed__6_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Ring___aux__Mathlib__Algebra__GroupWithZero__Units__Basic______macroRules__Ring__term___u207b_xb9_u02b3__1___closed__6;
static const lean_ctor_object lp_mathlib_Ring___aux__Mathlib__Algebra__GroupWithZero__Units__Basic______macroRules__Ring__term___u207b_xb9_u02b3__1___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Ring___aux__Mathlib__Algebra__GroupWithZero__Units__Basic______macroRules__Ring__term___u207b_xb9_u02b3__1___closed__5_value),LEAN_SCALAR_PTR_LITERAL(200, 187, 129, 77, 144, 131, 82, 1)}};
static const lean_object* lp_mathlib_Ring___aux__Mathlib__Algebra__GroupWithZero__Units__Basic______macroRules__Ring__term___u207b_xb9_u02b3__1___closed__7 = (const lean_object*)&lp_mathlib_Ring___aux__Mathlib__Algebra__GroupWithZero__Units__Basic______macroRules__Ring__term___u207b_xb9_u02b3__1___closed__7_value;
static const lean_ctor_object lp_mathlib_Ring___aux__Mathlib__Algebra__GroupWithZero__Units__Basic______macroRules__Ring__term___u207b_xb9_u02b3__1___closed__8_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Ring_term___u207b_xb9_u02b3___closed__0_value),LEAN_SCALAR_PTR_LITERAL(151, 9, 120, 97, 235, 184, 251, 227)}};
static const lean_ctor_object lp_mathlib_Ring___aux__Mathlib__Algebra__GroupWithZero__Units__Basic______macroRules__Ring__term___u207b_xb9_u02b3__1___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Ring___aux__Mathlib__Algebra__GroupWithZero__Units__Basic______macroRules__Ring__term___u207b_xb9_u02b3__1___closed__8_value_aux_0),((lean_object*)&lp_mathlib_Ring___aux__Mathlib__Algebra__GroupWithZero__Units__Basic______macroRules__Ring__term___u207b_xb9_u02b3__1___closed__5_value),LEAN_SCALAR_PTR_LITERAL(212, 249, 100, 229, 88, 159, 173, 94)}};
static const lean_object* lp_mathlib_Ring___aux__Mathlib__Algebra__GroupWithZero__Units__Basic______macroRules__Ring__term___u207b_xb9_u02b3__1___closed__8 = (const lean_object*)&lp_mathlib_Ring___aux__Mathlib__Algebra__GroupWithZero__Units__Basic______macroRules__Ring__term___u207b_xb9_u02b3__1___closed__8_value;
static const lean_ctor_object lp_mathlib_Ring___aux__Mathlib__Algebra__GroupWithZero__Units__Basic______macroRules__Ring__term___u207b_xb9_u02b3__1___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Ring___aux__Mathlib__Algebra__GroupWithZero__Units__Basic______macroRules__Ring__term___u207b_xb9_u02b3__1___closed__8_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_Ring___aux__Mathlib__Algebra__GroupWithZero__Units__Basic______macroRules__Ring__term___u207b_xb9_u02b3__1___closed__9 = (const lean_object*)&lp_mathlib_Ring___aux__Mathlib__Algebra__GroupWithZero__Units__Basic______macroRules__Ring__term___u207b_xb9_u02b3__1___closed__9_value;
static const lean_ctor_object lp_mathlib_Ring___aux__Mathlib__Algebra__GroupWithZero__Units__Basic______macroRules__Ring__term___u207b_xb9_u02b3__1___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Ring___aux__Mathlib__Algebra__GroupWithZero__Units__Basic______macroRules__Ring__term___u207b_xb9_u02b3__1___closed__9_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_Ring___aux__Mathlib__Algebra__GroupWithZero__Units__Basic______macroRules__Ring__term___u207b_xb9_u02b3__1___closed__10 = (const lean_object*)&lp_mathlib_Ring___aux__Mathlib__Algebra__GroupWithZero__Units__Basic______macroRules__Ring__term___u207b_xb9_u02b3__1___closed__10_value;
static const lean_string_object lp_mathlib_Ring___aux__Mathlib__Algebra__GroupWithZero__Units__Basic______macroRules__Ring__term___u207b_xb9_u02b3__1___closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "null"};
static const lean_object* lp_mathlib_Ring___aux__Mathlib__Algebra__GroupWithZero__Units__Basic______macroRules__Ring__term___u207b_xb9_u02b3__1___closed__11 = (const lean_object*)&lp_mathlib_Ring___aux__Mathlib__Algebra__GroupWithZero__Units__Basic______macroRules__Ring__term___u207b_xb9_u02b3__1___closed__11_value;
static const lean_ctor_object lp_mathlib_Ring___aux__Mathlib__Algebra__GroupWithZero__Units__Basic______macroRules__Ring__term___u207b_xb9_u02b3__1___closed__12_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Ring___aux__Mathlib__Algebra__GroupWithZero__Units__Basic______macroRules__Ring__term___u207b_xb9_u02b3__1___closed__11_value),LEAN_SCALAR_PTR_LITERAL(24, 58, 49, 223, 146, 207, 197, 136)}};
static const lean_object* lp_mathlib_Ring___aux__Mathlib__Algebra__GroupWithZero__Units__Basic______macroRules__Ring__term___u207b_xb9_u02b3__1___closed__12 = (const lean_object*)&lp_mathlib_Ring___aux__Mathlib__Algebra__GroupWithZero__Units__Basic______macroRules__Ring__term___u207b_xb9_u02b3__1___closed__12_value;
LEAN_EXPORT lean_object* lp_mathlib_Ring___aux__Mathlib__Algebra__GroupWithZero__Units__Basic______macroRules__Ring__term___u207b_xb9_u02b3__1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Ring___aux__Mathlib__Algebra__GroupWithZero__Units__Basic______macroRules__Ring__term___u207b_xb9_u02b3__1___boxed(lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Ring___aux__Mathlib__Algebra__GroupWithZero__Units__Basic______unexpand__Ring__inverse__1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "ident"};
static const lean_object* lp_mathlib_Ring___aux__Mathlib__Algebra__GroupWithZero__Units__Basic______unexpand__Ring__inverse__1___closed__0 = (const lean_object*)&lp_mathlib_Ring___aux__Mathlib__Algebra__GroupWithZero__Units__Basic______unexpand__Ring__inverse__1___closed__0_value;
static const lean_ctor_object lp_mathlib_Ring___aux__Mathlib__Algebra__GroupWithZero__Units__Basic______unexpand__Ring__inverse__1___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Ring___aux__Mathlib__Algebra__GroupWithZero__Units__Basic______unexpand__Ring__inverse__1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(52, 159, 208, 51, 14, 60, 6, 71)}};
static const lean_object* lp_mathlib_Ring___aux__Mathlib__Algebra__GroupWithZero__Units__Basic______unexpand__Ring__inverse__1___closed__1 = (const lean_object*)&lp_mathlib_Ring___aux__Mathlib__Algebra__GroupWithZero__Units__Basic______unexpand__Ring__inverse__1___closed__1_value;
LEAN_EXPORT lean_object* lp_mathlib_Ring___aux__Mathlib__Algebra__GroupWithZero__Units__Basic______unexpand__Ring__inverse__1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Ring___aux__Mathlib__Algebra__GroupWithZero__Units__Basic______unexpand__Ring__inverse__1___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Units_mk0___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Units_mk0(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CommGroupWithZero_toDivisionCommMonoid___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CommGroupWithZero_toDivisionCommMonoid(lean_object*, lean_object*);
static lean_object* _init_lp_mathlib_Ring___aux__Mathlib__Algebra__GroupWithZero__Units__Basic______macroRules__Ring__term___u207b_xb9_u02b3__1___closed__6(void){
_start:
{
lean_object* v___x_24_; lean_object* v___x_25_; 
v___x_24_ = ((lean_object*)(lp_mathlib_Ring___aux__Mathlib__Algebra__GroupWithZero__Units__Basic______macroRules__Ring__term___u207b_xb9_u02b3__1___closed__5));
v___x_25_ = l_String_toRawSubstring_x27(v___x_24_);
return v___x_25_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Ring___aux__Mathlib__Algebra__GroupWithZero__Units__Basic______macroRules__Ring__term___u207b_xb9_u02b3__1(lean_object* v_x_40_, lean_object* v_a_41_, lean_object* v_a_42_){
_start:
{
lean_object* v___x_43_; uint8_t v___x_44_; 
v___x_43_ = ((lean_object*)(lp_mathlib_Ring_term___u207b_xb9_u02b3___closed__2));
lean_inc(v_x_40_);
v___x_44_ = l_Lean_Syntax_isOfKind(v_x_40_, v___x_43_);
if (v___x_44_ == 0)
{
lean_object* v___x_45_; lean_object* v___x_46_; 
lean_dec(v_x_40_);
v___x_45_ = lean_box(1);
v___x_46_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_46_, 0, v___x_45_);
lean_ctor_set(v___x_46_, 1, v_a_42_);
return v___x_46_;
}
else
{
lean_object* v_quotContext_47_; lean_object* v_currMacroScope_48_; lean_object* v_ref_49_; lean_object* v___x_50_; lean_object* v___x_51_; uint8_t v___x_52_; lean_object* v___x_53_; lean_object* v___x_54_; lean_object* v___x_55_; lean_object* v___x_56_; lean_object* v___x_57_; lean_object* v___x_58_; lean_object* v___x_59_; lean_object* v___x_60_; lean_object* v___x_61_; lean_object* v___x_62_; lean_object* v___x_63_; 
v_quotContext_47_ = lean_ctor_get(v_a_41_, 1);
v_currMacroScope_48_ = lean_ctor_get(v_a_41_, 2);
v_ref_49_ = lean_ctor_get(v_a_41_, 5);
v___x_50_ = lean_unsigned_to_nat(0u);
v___x_51_ = l_Lean_Syntax_getArg(v_x_40_, v___x_50_);
lean_dec(v_x_40_);
v___x_52_ = 0;
v___x_53_ = l_Lean_SourceInfo_fromRef(v_ref_49_, v___x_52_);
v___x_54_ = ((lean_object*)(lp_mathlib_Ring___aux__Mathlib__Algebra__GroupWithZero__Units__Basic______macroRules__Ring__term___u207b_xb9_u02b3__1___closed__4));
v___x_55_ = lean_obj_once(&lp_mathlib_Ring___aux__Mathlib__Algebra__GroupWithZero__Units__Basic______macroRules__Ring__term___u207b_xb9_u02b3__1___closed__6, &lp_mathlib_Ring___aux__Mathlib__Algebra__GroupWithZero__Units__Basic______macroRules__Ring__term___u207b_xb9_u02b3__1___closed__6_once, _init_lp_mathlib_Ring___aux__Mathlib__Algebra__GroupWithZero__Units__Basic______macroRules__Ring__term___u207b_xb9_u02b3__1___closed__6);
v___x_56_ = ((lean_object*)(lp_mathlib_Ring___aux__Mathlib__Algebra__GroupWithZero__Units__Basic______macroRules__Ring__term___u207b_xb9_u02b3__1___closed__7));
lean_inc(v_currMacroScope_48_);
lean_inc(v_quotContext_47_);
v___x_57_ = l_Lean_addMacroScope(v_quotContext_47_, v___x_56_, v_currMacroScope_48_);
v___x_58_ = ((lean_object*)(lp_mathlib_Ring___aux__Mathlib__Algebra__GroupWithZero__Units__Basic______macroRules__Ring__term___u207b_xb9_u02b3__1___closed__10));
lean_inc_n(v___x_53_, 2);
v___x_59_ = lean_alloc_ctor(3, 4, 0);
lean_ctor_set(v___x_59_, 0, v___x_53_);
lean_ctor_set(v___x_59_, 1, v___x_55_);
lean_ctor_set(v___x_59_, 2, v___x_57_);
lean_ctor_set(v___x_59_, 3, v___x_58_);
v___x_60_ = ((lean_object*)(lp_mathlib_Ring___aux__Mathlib__Algebra__GroupWithZero__Units__Basic______macroRules__Ring__term___u207b_xb9_u02b3__1___closed__12));
v___x_61_ = l_Lean_Syntax_node1(v___x_53_, v___x_60_, v___x_51_);
v___x_62_ = l_Lean_Syntax_node2(v___x_53_, v___x_54_, v___x_59_, v___x_61_);
v___x_63_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_63_, 0, v___x_62_);
lean_ctor_set(v___x_63_, 1, v_a_42_);
return v___x_63_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Ring___aux__Mathlib__Algebra__GroupWithZero__Units__Basic______macroRules__Ring__term___u207b_xb9_u02b3__1___boxed(lean_object* v_x_64_, lean_object* v_a_65_, lean_object* v_a_66_){
_start:
{
lean_object* v_res_67_; 
v_res_67_ = lp_mathlib_Ring___aux__Mathlib__Algebra__GroupWithZero__Units__Basic______macroRules__Ring__term___u207b_xb9_u02b3__1(v_x_64_, v_a_65_, v_a_66_);
lean_dec_ref(v_a_65_);
return v_res_67_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Ring___aux__Mathlib__Algebra__GroupWithZero__Units__Basic______unexpand__Ring__inverse__1(lean_object* v_x_71_, lean_object* v_a_72_, lean_object* v_a_73_){
_start:
{
lean_object* v___x_74_; uint8_t v___x_75_; 
v___x_74_ = ((lean_object*)(lp_mathlib_Ring___aux__Mathlib__Algebra__GroupWithZero__Units__Basic______macroRules__Ring__term___u207b_xb9_u02b3__1___closed__4));
lean_inc(v_x_71_);
v___x_75_ = l_Lean_Syntax_isOfKind(v_x_71_, v___x_74_);
if (v___x_75_ == 0)
{
lean_object* v___x_76_; lean_object* v___x_77_; 
lean_dec(v_x_71_);
v___x_76_ = lean_box(0);
v___x_77_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_77_, 0, v___x_76_);
lean_ctor_set(v___x_77_, 1, v_a_73_);
return v___x_77_;
}
else
{
lean_object* v___x_78_; lean_object* v___x_79_; lean_object* v___x_80_; uint8_t v___x_81_; 
v___x_78_ = lean_unsigned_to_nat(0u);
v___x_79_ = l_Lean_Syntax_getArg(v_x_71_, v___x_78_);
v___x_80_ = ((lean_object*)(lp_mathlib_Ring___aux__Mathlib__Algebra__GroupWithZero__Units__Basic______unexpand__Ring__inverse__1___closed__1));
lean_inc(v___x_79_);
v___x_81_ = l_Lean_Syntax_isOfKind(v___x_79_, v___x_80_);
if (v___x_81_ == 0)
{
lean_object* v___x_82_; lean_object* v___x_83_; 
lean_dec(v___x_79_);
lean_dec(v_x_71_);
v___x_82_ = lean_box(0);
v___x_83_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_83_, 0, v___x_82_);
lean_ctor_set(v___x_83_, 1, v_a_73_);
return v___x_83_;
}
else
{
lean_object* v___x_84_; lean_object* v___x_85_; uint8_t v___x_86_; 
v___x_84_ = lean_unsigned_to_nat(1u);
v___x_85_ = l_Lean_Syntax_getArg(v_x_71_, v___x_84_);
lean_dec(v_x_71_);
lean_inc(v___x_85_);
v___x_86_ = l_Lean_Syntax_matchesNull(v___x_85_, v___x_84_);
if (v___x_86_ == 0)
{
lean_object* v___x_87_; lean_object* v___x_88_; 
lean_dec(v___x_85_);
lean_dec(v___x_79_);
v___x_87_ = lean_box(0);
v___x_88_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_88_, 0, v___x_87_);
lean_ctor_set(v___x_88_, 1, v_a_73_);
return v___x_88_;
}
else
{
lean_object* v___x_89_; lean_object* v_ref_90_; uint8_t v___x_91_; lean_object* v___x_92_; lean_object* v___x_93_; lean_object* v___x_94_; lean_object* v___x_95_; lean_object* v___x_96_; lean_object* v___x_97_; 
v___x_89_ = l_Lean_Syntax_getArg(v___x_85_, v___x_78_);
lean_dec(v___x_85_);
v_ref_90_ = l_Lean_replaceRef(v___x_79_, v_a_72_);
lean_dec(v___x_79_);
v___x_91_ = 0;
v___x_92_ = l_Lean_SourceInfo_fromRef(v_ref_90_, v___x_91_);
lean_dec(v_ref_90_);
v___x_93_ = ((lean_object*)(lp_mathlib_Ring_term___u207b_xb9_u02b3___closed__2));
v___x_94_ = ((lean_object*)(lp_mathlib_Ring_term___u207b_xb9_u02b3___closed__3));
lean_inc(v___x_92_);
v___x_95_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_95_, 0, v___x_92_);
lean_ctor_set(v___x_95_, 1, v___x_94_);
v___x_96_ = l_Lean_Syntax_node2(v___x_92_, v___x_93_, v___x_89_, v___x_95_);
v___x_97_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_97_, 0, v___x_96_);
lean_ctor_set(v___x_97_, 1, v_a_73_);
return v___x_97_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Ring___aux__Mathlib__Algebra__GroupWithZero__Units__Basic______unexpand__Ring__inverse__1___boxed(lean_object* v_x_98_, lean_object* v_a_99_, lean_object* v_a_100_){
_start:
{
lean_object* v_res_101_; 
v_res_101_ = lp_mathlib_Ring___aux__Mathlib__Algebra__GroupWithZero__Units__Basic______unexpand__Ring__inverse__1(v_x_98_, v_a_99_, v_a_100_);
lean_dec(v_a_99_);
return v_res_101_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Units_mk0___redArg(lean_object* v_inst_102_, lean_object* v_a_103_){
_start:
{
lean_object* v___x_104_; lean_object* v___x_105_; lean_object* v_toInv_106_; lean_object* v___x_108_; uint8_t v_isShared_109_; uint8_t v_isSharedCheck_114_; 
v___x_104_ = lp_mathlib_GroupWithZero_toDivInvMonoid___redArg(v_inst_102_);
v___x_105_ = lp_mathlib_DivInvOneMonoid_toInvOneClass___redArg(v___x_104_);
lean_dec_ref(v___x_104_);
v_toInv_106_ = lean_ctor_get(v___x_105_, 1);
v_isSharedCheck_114_ = !lean_is_exclusive(v___x_105_);
if (v_isSharedCheck_114_ == 0)
{
lean_object* v_unused_115_; 
v_unused_115_ = lean_ctor_get(v___x_105_, 0);
lean_dec(v_unused_115_);
v___x_108_ = v___x_105_;
v_isShared_109_ = v_isSharedCheck_114_;
goto v_resetjp_107_;
}
else
{
lean_inc(v_toInv_106_);
lean_dec(v___x_105_);
v___x_108_ = lean_box(0);
v_isShared_109_ = v_isSharedCheck_114_;
goto v_resetjp_107_;
}
v_resetjp_107_:
{
lean_object* v___x_110_; lean_object* v___x_112_; 
lean_inc(v_a_103_);
v___x_110_ = lean_apply_1(v_toInv_106_, v_a_103_);
if (v_isShared_109_ == 0)
{
lean_ctor_set(v___x_108_, 1, v___x_110_);
lean_ctor_set(v___x_108_, 0, v_a_103_);
v___x_112_ = v___x_108_;
goto v_reusejp_111_;
}
else
{
lean_object* v_reuseFailAlloc_113_; 
v_reuseFailAlloc_113_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_113_, 0, v_a_103_);
lean_ctor_set(v_reuseFailAlloc_113_, 1, v___x_110_);
v___x_112_ = v_reuseFailAlloc_113_;
goto v_reusejp_111_;
}
v_reusejp_111_:
{
return v___x_112_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Units_mk0(lean_object* v_G_u2080_116_, lean_object* v_inst_117_, lean_object* v_a_118_, lean_object* v_ha_119_){
_start:
{
lean_object* v___x_120_; 
v___x_120_ = lp_mathlib_Units_mk0___redArg(v_inst_117_, v_a_118_);
return v___x_120_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CommGroupWithZero_toDivisionCommMonoid___redArg(lean_object* v_inst_121_){
_start:
{
lean_object* v_toCommMonoidWithZero_122_; lean_object* v_toInv_123_; lean_object* v_toDiv_124_; lean_object* v_toZPow_125_; lean_object* v___x_127_; uint8_t v_isShared_128_; uint8_t v_isSharedCheck_133_; 
v_toCommMonoidWithZero_122_ = lean_ctor_get(v_inst_121_, 0);
v_toInv_123_ = lean_ctor_get(v_inst_121_, 1);
v_toDiv_124_ = lean_ctor_get(v_inst_121_, 2);
v_toZPow_125_ = lean_ctor_get(v_inst_121_, 3);
v_isSharedCheck_133_ = !lean_is_exclusive(v_inst_121_);
if (v_isSharedCheck_133_ == 0)
{
v___x_127_ = v_inst_121_;
v_isShared_128_ = v_isSharedCheck_133_;
goto v_resetjp_126_;
}
else
{
lean_inc(v_toZPow_125_);
lean_inc(v_toDiv_124_);
lean_inc(v_toInv_123_);
lean_inc(v_toCommMonoidWithZero_122_);
lean_dec(v_inst_121_);
v___x_127_ = lean_box(0);
v_isShared_128_ = v_isSharedCheck_133_;
goto v_resetjp_126_;
}
v_resetjp_126_:
{
lean_object* v_toCommMonoid_129_; lean_object* v___x_131_; 
v_toCommMonoid_129_ = lean_ctor_get(v_toCommMonoidWithZero_122_, 0);
lean_inc_ref(v_toCommMonoid_129_);
lean_dec_ref(v_toCommMonoidWithZero_122_);
if (v_isShared_128_ == 0)
{
lean_ctor_set(v___x_127_, 0, v_toCommMonoid_129_);
v___x_131_ = v___x_127_;
goto v_reusejp_130_;
}
else
{
lean_object* v_reuseFailAlloc_132_; 
v_reuseFailAlloc_132_ = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(v_reuseFailAlloc_132_, 0, v_toCommMonoid_129_);
lean_ctor_set(v_reuseFailAlloc_132_, 1, v_toInv_123_);
lean_ctor_set(v_reuseFailAlloc_132_, 2, v_toDiv_124_);
lean_ctor_set(v_reuseFailAlloc_132_, 3, v_toZPow_125_);
v___x_131_ = v_reuseFailAlloc_132_;
goto v_reusejp_130_;
}
v_reusejp_130_:
{
return v___x_131_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_CommGroupWithZero_toDivisionCommMonoid(lean_object* v_G_u2080_134_, lean_object* v_inst_135_){
_start:
{
lean_object* v___x_136_; 
v___x_136_ = lp_mathlib_CommGroupWithZero_toDivisionCommMonoid___redArg(v_inst_135_);
return v___x_136_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Group_Units_Basic(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_GroupWithZero_Basic(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Data_Nat_Basic(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Lean_Meta_CongrTheorems(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Tactic_Contrapose(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Tactic_Spread(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Tactic_Convert(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Tactic_Nontriviality(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_Algebra_GroupWithZero_Units_Basic(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Group_Units_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_GroupWithZero_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Data_Nat_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Lean_Meta_CongrTheorems(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Tactic_Contrapose(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Tactic_Spread(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Tactic_Convert(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Tactic_Nontriviality(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_Algebra_GroupWithZero_Units_Basic(uint8_t builtin) {
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
lean_object* initialize_mathlib_Mathlib_Algebra_Group_Units_Basic(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_GroupWithZero_Basic(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Data_Nat_Basic(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Lean_Meta_CongrTheorems(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Tactic_Contrapose(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Tactic_Spread(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Tactic_Convert(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Tactic_Nontriviality(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Algebra_GroupWithZero_Units_Basic(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Group_Units_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_GroupWithZero_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_Nat_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Lean_Meta_CongrTheorems(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Tactic_Contrapose(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Tactic_Spread(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Tactic_Convert(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Tactic_Nontriviality(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_GroupWithZero_Units_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_Algebra_GroupWithZero_Units_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_Algebra_GroupWithZero_Units_Basic(builtin);
}
#ifdef __cplusplus
}
#endif
