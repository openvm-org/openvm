// Lean compiler output
// Module: Mathlib.Algebra.Opposites
// Imports: public import Init public meta import Init public import Mathlib.Algebra.Group.DivInvMonoid public import Mathlib.Algebra.Notation.Defs public import Mathlib.Logic.Equiv.Defs
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
lean_object* l_Lean_Name_mkStr1(lean_object*);
uint8_t l_Lean_Syntax_isOfKind(lean_object*, lean_object*);
lean_object* l_Lean_Syntax_getArg(lean_object*, lean_object*);
lean_object* l_Lean_SourceInfo_fromRef(lean_object*, uint8_t);
lean_object* l_Lean_Name_mkStr4(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_String_toRawSubstring_x27(lean_object*);
lean_object* l_Lean_addMacroScope(lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Syntax_node1(lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Syntax_node2(lean_object*, lean_object*, lean_object*, lean_object*);
uint8_t lp_mathlib_Function_Injective_decidableEq___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
uint8_t l_Lean_Syntax_matchesNull(lean_object*, lean_object*);
lean_object* l_Lean_replaceRef(lean_object*, lean_object*);
static const lean_string_object lp_mathlib_term___u1d50_u1d52_u1d56___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 15, .m_capacity = 15, .m_length = 8, .m_data = "term_ᵐᵒᵖ"};
static const lean_object* lp_mathlib_term___u1d50_u1d52_u1d56___closed__0 = (const lean_object*)&lp_mathlib_term___u1d50_u1d52_u1d56___closed__0_value;
static const lean_ctor_object lp_mathlib_term___u1d50_u1d52_u1d56___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_term___u1d50_u1d52_u1d56___closed__0_value),LEAN_SCALAR_PTR_LITERAL(201, 76, 162, 62, 157, 57, 222, 234)}};
static const lean_object* lp_mathlib_term___u1d50_u1d52_u1d56___closed__1 = (const lean_object*)&lp_mathlib_term___u1d50_u1d52_u1d56___closed__1_value;
static const lean_string_object lp_mathlib_term___u1d50_u1d52_u1d56___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 3, .m_data = "ᵐᵒᵖ"};
static const lean_object* lp_mathlib_term___u1d50_u1d52_u1d56___closed__2 = (const lean_object*)&lp_mathlib_term___u1d50_u1d52_u1d56___closed__2_value;
static const lean_ctor_object lp_mathlib_term___u1d50_u1d52_u1d56___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 5}, .m_objs = {((lean_object*)&lp_mathlib_term___u1d50_u1d52_u1d56___closed__2_value)}};
static const lean_object* lp_mathlib_term___u1d50_u1d52_u1d56___closed__3 = (const lean_object*)&lp_mathlib_term___u1d50_u1d52_u1d56___closed__3_value;
static const lean_ctor_object lp_mathlib_term___u1d50_u1d52_u1d56___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*4 + 0, .m_other = 4, .m_tag = 4}, .m_objs = {((lean_object*)&lp_mathlib_term___u1d50_u1d52_u1d56___closed__1_value),((lean_object*)(((size_t)(1024) << 1) | 1)),((lean_object*)(((size_t)(1024) << 1) | 1)),((lean_object*)&lp_mathlib_term___u1d50_u1d52_u1d56___closed__3_value)}};
static const lean_object* lp_mathlib_term___u1d50_u1d52_u1d56___closed__4 = (const lean_object*)&lp_mathlib_term___u1d50_u1d52_u1d56___closed__4_value;
LEAN_EXPORT const lean_object* lp_mathlib_term___u1d50_u1d52_u1d56 = (const lean_object*)&lp_mathlib_term___u1d50_u1d52_u1d56___closed__4_value;
static const lean_string_object lp_mathlib___aux__Mathlib__Algebra__Opposites______macroRules__term___u1d50_u1d52_u1d56__1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "Lean"};
static const lean_object* lp_mathlib___aux__Mathlib__Algebra__Opposites______macroRules__term___u1d50_u1d52_u1d56__1___closed__0 = (const lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Opposites______macroRules__term___u1d50_u1d52_u1d56__1___closed__0_value;
static const lean_string_object lp_mathlib___aux__Mathlib__Algebra__Opposites______macroRules__term___u1d50_u1d52_u1d56__1___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "Parser"};
static const lean_object* lp_mathlib___aux__Mathlib__Algebra__Opposites______macroRules__term___u1d50_u1d52_u1d56__1___closed__1 = (const lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Opposites______macroRules__term___u1d50_u1d52_u1d56__1___closed__1_value;
static const lean_string_object lp_mathlib___aux__Mathlib__Algebra__Opposites______macroRules__term___u1d50_u1d52_u1d56__1___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "Term"};
static const lean_object* lp_mathlib___aux__Mathlib__Algebra__Opposites______macroRules__term___u1d50_u1d52_u1d56__1___closed__2 = (const lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Opposites______macroRules__term___u1d50_u1d52_u1d56__1___closed__2_value;
static const lean_string_object lp_mathlib___aux__Mathlib__Algebra__Opposites______macroRules__term___u1d50_u1d52_u1d56__1___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = "app"};
static const lean_object* lp_mathlib___aux__Mathlib__Algebra__Opposites______macroRules__term___u1d50_u1d52_u1d56__1___closed__3 = (const lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Opposites______macroRules__term___u1d50_u1d52_u1d56__1___closed__3_value;
static const lean_ctor_object lp_mathlib___aux__Mathlib__Algebra__Opposites______macroRules__term___u1d50_u1d52_u1d56__1___closed__4_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Opposites______macroRules__term___u1d50_u1d52_u1d56__1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib___aux__Mathlib__Algebra__Opposites______macroRules__term___u1d50_u1d52_u1d56__1___closed__4_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Opposites______macroRules__term___u1d50_u1d52_u1d56__1___closed__4_value_aux_0),((lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Opposites______macroRules__term___u1d50_u1d52_u1d56__1___closed__1_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib___aux__Mathlib__Algebra__Opposites______macroRules__term___u1d50_u1d52_u1d56__1___closed__4_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Opposites______macroRules__term___u1d50_u1d52_u1d56__1___closed__4_value_aux_1),((lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Opposites______macroRules__term___u1d50_u1d52_u1d56__1___closed__2_value),LEAN_SCALAR_PTR_LITERAL(75, 170, 162, 138, 136, 204, 251, 229)}};
static const lean_ctor_object lp_mathlib___aux__Mathlib__Algebra__Opposites______macroRules__term___u1d50_u1d52_u1d56__1___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Opposites______macroRules__term___u1d50_u1d52_u1d56__1___closed__4_value_aux_2),((lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Opposites______macroRules__term___u1d50_u1d52_u1d56__1___closed__3_value),LEAN_SCALAR_PTR_LITERAL(69, 118, 10, 41, 220, 156, 243, 179)}};
static const lean_object* lp_mathlib___aux__Mathlib__Algebra__Opposites______macroRules__term___u1d50_u1d52_u1d56__1___closed__4 = (const lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Opposites______macroRules__term___u1d50_u1d52_u1d56__1___closed__4_value;
static const lean_string_object lp_mathlib___aux__Mathlib__Algebra__Opposites______macroRules__term___u1d50_u1d52_u1d56__1___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 12, .m_capacity = 12, .m_length = 11, .m_data = "MulOpposite"};
static const lean_object* lp_mathlib___aux__Mathlib__Algebra__Opposites______macroRules__term___u1d50_u1d52_u1d56__1___closed__5 = (const lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Opposites______macroRules__term___u1d50_u1d52_u1d56__1___closed__5_value;
static lean_once_cell_t lp_mathlib___aux__Mathlib__Algebra__Opposites______macroRules__term___u1d50_u1d52_u1d56__1___closed__6_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___aux__Mathlib__Algebra__Opposites______macroRules__term___u1d50_u1d52_u1d56__1___closed__6;
static const lean_ctor_object lp_mathlib___aux__Mathlib__Algebra__Opposites______macroRules__term___u1d50_u1d52_u1d56__1___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Opposites______macroRules__term___u1d50_u1d52_u1d56__1___closed__5_value),LEAN_SCALAR_PTR_LITERAL(167, 55, 13, 115, 157, 142, 229, 51)}};
static const lean_object* lp_mathlib___aux__Mathlib__Algebra__Opposites______macroRules__term___u1d50_u1d52_u1d56__1___closed__7 = (const lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Opposites______macroRules__term___u1d50_u1d52_u1d56__1___closed__7_value;
static const lean_ctor_object lp_mathlib___aux__Mathlib__Algebra__Opposites______macroRules__term___u1d50_u1d52_u1d56__1___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Opposites______macroRules__term___u1d50_u1d52_u1d56__1___closed__7_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib___aux__Mathlib__Algebra__Opposites______macroRules__term___u1d50_u1d52_u1d56__1___closed__8 = (const lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Opposites______macroRules__term___u1d50_u1d52_u1d56__1___closed__8_value;
static const lean_ctor_object lp_mathlib___aux__Mathlib__Algebra__Opposites______macroRules__term___u1d50_u1d52_u1d56__1___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Opposites______macroRules__term___u1d50_u1d52_u1d56__1___closed__8_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib___aux__Mathlib__Algebra__Opposites______macroRules__term___u1d50_u1d52_u1d56__1___closed__9 = (const lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Opposites______macroRules__term___u1d50_u1d52_u1d56__1___closed__9_value;
static const lean_string_object lp_mathlib___aux__Mathlib__Algebra__Opposites______macroRules__term___u1d50_u1d52_u1d56__1___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "null"};
static const lean_object* lp_mathlib___aux__Mathlib__Algebra__Opposites______macroRules__term___u1d50_u1d52_u1d56__1___closed__10 = (const lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Opposites______macroRules__term___u1d50_u1d52_u1d56__1___closed__10_value;
static const lean_ctor_object lp_mathlib___aux__Mathlib__Algebra__Opposites______macroRules__term___u1d50_u1d52_u1d56__1___closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Opposites______macroRules__term___u1d50_u1d52_u1d56__1___closed__10_value),LEAN_SCALAR_PTR_LITERAL(24, 58, 49, 223, 146, 207, 197, 136)}};
static const lean_object* lp_mathlib___aux__Mathlib__Algebra__Opposites______macroRules__term___u1d50_u1d52_u1d56__1___closed__11 = (const lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Opposites______macroRules__term___u1d50_u1d52_u1d56__1___closed__11_value;
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Algebra__Opposites______macroRules__term___u1d50_u1d52_u1d56__1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Algebra__Opposites______macroRules__term___u1d50_u1d52_u1d56__1___boxed(lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib___aux__Mathlib__Algebra__Opposites______unexpand__MulOpposite__1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "ident"};
static const lean_object* lp_mathlib___aux__Mathlib__Algebra__Opposites______unexpand__MulOpposite__1___closed__0 = (const lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Opposites______unexpand__MulOpposite__1___closed__0_value;
static const lean_ctor_object lp_mathlib___aux__Mathlib__Algebra__Opposites______unexpand__MulOpposite__1___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Opposites______unexpand__MulOpposite__1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(52, 159, 208, 51, 14, 60, 6, 71)}};
static const lean_object* lp_mathlib___aux__Mathlib__Algebra__Opposites______unexpand__MulOpposite__1___closed__1 = (const lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Opposites______unexpand__MulOpposite__1___closed__1_value;
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Algebra__Opposites______unexpand__MulOpposite__1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Algebra__Opposites______unexpand__MulOpposite__1___boxed(lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_term___u1d43_u1d52_u1d56___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 15, .m_capacity = 15, .m_length = 8, .m_data = "term_ᵃᵒᵖ"};
static const lean_object* lp_mathlib_term___u1d43_u1d52_u1d56___closed__0 = (const lean_object*)&lp_mathlib_term___u1d43_u1d52_u1d56___closed__0_value;
static const lean_ctor_object lp_mathlib_term___u1d43_u1d52_u1d56___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_term___u1d43_u1d52_u1d56___closed__0_value),LEAN_SCALAR_PTR_LITERAL(14, 224, 235, 241, 35, 148, 197, 157)}};
static const lean_object* lp_mathlib_term___u1d43_u1d52_u1d56___closed__1 = (const lean_object*)&lp_mathlib_term___u1d43_u1d52_u1d56___closed__1_value;
static const lean_string_object lp_mathlib_term___u1d43_u1d52_u1d56___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 3, .m_data = "ᵃᵒᵖ"};
static const lean_object* lp_mathlib_term___u1d43_u1d52_u1d56___closed__2 = (const lean_object*)&lp_mathlib_term___u1d43_u1d52_u1d56___closed__2_value;
static const lean_ctor_object lp_mathlib_term___u1d43_u1d52_u1d56___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 5}, .m_objs = {((lean_object*)&lp_mathlib_term___u1d43_u1d52_u1d56___closed__2_value)}};
static const lean_object* lp_mathlib_term___u1d43_u1d52_u1d56___closed__3 = (const lean_object*)&lp_mathlib_term___u1d43_u1d52_u1d56___closed__3_value;
static const lean_ctor_object lp_mathlib_term___u1d43_u1d52_u1d56___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*4 + 0, .m_other = 4, .m_tag = 4}, .m_objs = {((lean_object*)&lp_mathlib_term___u1d43_u1d52_u1d56___closed__1_value),((lean_object*)(((size_t)(1024) << 1) | 1)),((lean_object*)(((size_t)(1024) << 1) | 1)),((lean_object*)&lp_mathlib_term___u1d43_u1d52_u1d56___closed__3_value)}};
static const lean_object* lp_mathlib_term___u1d43_u1d52_u1d56___closed__4 = (const lean_object*)&lp_mathlib_term___u1d43_u1d52_u1d56___closed__4_value;
LEAN_EXPORT const lean_object* lp_mathlib_term___u1d43_u1d52_u1d56 = (const lean_object*)&lp_mathlib_term___u1d43_u1d52_u1d56___closed__4_value;
static const lean_string_object lp_mathlib___aux__Mathlib__Algebra__Opposites______macroRules__term___u1d43_u1d52_u1d56__1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 12, .m_capacity = 12, .m_length = 11, .m_data = "AddOpposite"};
static const lean_object* lp_mathlib___aux__Mathlib__Algebra__Opposites______macroRules__term___u1d43_u1d52_u1d56__1___closed__0 = (const lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Opposites______macroRules__term___u1d43_u1d52_u1d56__1___closed__0_value;
static lean_once_cell_t lp_mathlib___aux__Mathlib__Algebra__Opposites______macroRules__term___u1d43_u1d52_u1d56__1___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___aux__Mathlib__Algebra__Opposites______macroRules__term___u1d43_u1d52_u1d56__1___closed__1;
static const lean_ctor_object lp_mathlib___aux__Mathlib__Algebra__Opposites______macroRules__term___u1d43_u1d52_u1d56__1___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Opposites______macroRules__term___u1d43_u1d52_u1d56__1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(248, 170, 135, 100, 51, 49, 73, 76)}};
static const lean_object* lp_mathlib___aux__Mathlib__Algebra__Opposites______macroRules__term___u1d43_u1d52_u1d56__1___closed__2 = (const lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Opposites______macroRules__term___u1d43_u1d52_u1d56__1___closed__2_value;
static const lean_ctor_object lp_mathlib___aux__Mathlib__Algebra__Opposites______macroRules__term___u1d43_u1d52_u1d56__1___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Opposites______macroRules__term___u1d43_u1d52_u1d56__1___closed__2_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib___aux__Mathlib__Algebra__Opposites______macroRules__term___u1d43_u1d52_u1d56__1___closed__3 = (const lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Opposites______macroRules__term___u1d43_u1d52_u1d56__1___closed__3_value;
static const lean_ctor_object lp_mathlib___aux__Mathlib__Algebra__Opposites______macroRules__term___u1d43_u1d52_u1d56__1___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Opposites______macroRules__term___u1d43_u1d52_u1d56__1___closed__3_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib___aux__Mathlib__Algebra__Opposites______macroRules__term___u1d43_u1d52_u1d56__1___closed__4 = (const lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Opposites______macroRules__term___u1d43_u1d52_u1d56__1___closed__4_value;
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Algebra__Opposites______macroRules__term___u1d43_u1d52_u1d56__1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Algebra__Opposites______macroRules__term___u1d43_u1d52_u1d56__1___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Algebra__Opposites______unexpand__AddOpposite__1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Algebra__Opposites______unexpand__AddOpposite__1___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulOpposite_op___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulOpposite_op___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulOpposite_op(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulOpposite_op___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddOpposite_op___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddOpposite_op___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddOpposite_op(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddOpposite_op___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulOpposite_unop___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulOpposite_unop___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulOpposite_unop(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulOpposite_unop___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddOpposite_unop___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddOpposite_unop___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddOpposite_unop(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddOpposite_unop___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulOpposite_rec_x27___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulOpposite_rec_x27(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddOpposite_rec_x27___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddOpposite_rec_x27(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulOpposite_opEquiv___lam__0(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulOpposite_opEquiv___lam__0___boxed(lean_object*);
static const lean_closure_object lp_mathlib_MulOpposite_opEquiv___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_MulOpposite_opEquiv___lam__0___boxed, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_MulOpposite_opEquiv___closed__0 = (const lean_object*)&lp_mathlib_MulOpposite_opEquiv___closed__0_value;
static const lean_ctor_object lp_mathlib_MulOpposite_opEquiv___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_MulOpposite_opEquiv___closed__0_value),((lean_object*)&lp_mathlib_MulOpposite_opEquiv___closed__0_value)}};
static const lean_object* lp_mathlib_MulOpposite_opEquiv___closed__1 = (const lean_object*)&lp_mathlib_MulOpposite_opEquiv___closed__1_value;
LEAN_EXPORT lean_object* lp_mathlib_MulOpposite_opEquiv(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddOpposite_opEquiv(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulOpposite_instInhabited___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulOpposite_instInhabited___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulOpposite_instInhabited(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulOpposite_instInhabited___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddOpposite_instInhabited___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddOpposite_instInhabited___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddOpposite_instInhabited(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddOpposite_instInhabited___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulOpposite_instUnique___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulOpposite_instUnique___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulOpposite_instUnique(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulOpposite_instUnique___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddOpposite_instUnique___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddOpposite_instUnique___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddOpposite_instUnique(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddOpposite_instUnique___boxed(lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_MulOpposite_instDecidableEq___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*1, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_MulOpposite_unop___boxed, .m_arity = 2, .m_num_fixed = 1, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1))} };
static const lean_object* lp_mathlib_MulOpposite_instDecidableEq___redArg___closed__0 = (const lean_object*)&lp_mathlib_MulOpposite_instDecidableEq___redArg___closed__0_value;
LEAN_EXPORT uint8_t lp_mathlib_MulOpposite_instDecidableEq___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulOpposite_instDecidableEq___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_MulOpposite_instDecidableEq(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulOpposite_instDecidableEq___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_AddOpposite_instDecidableEq___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*1, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_AddOpposite_unop___boxed, .m_arity = 2, .m_num_fixed = 1, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1))} };
static const lean_object* lp_mathlib_AddOpposite_instDecidableEq___redArg___closed__0 = (const lean_object*)&lp_mathlib_AddOpposite_instDecidableEq___redArg___closed__0_value;
LEAN_EXPORT uint8_t lp_mathlib_AddOpposite_instDecidableEq___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddOpposite_instDecidableEq___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_AddOpposite_instDecidableEq(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddOpposite_instDecidableEq___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulOpposite_instZero___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulOpposite_instZero___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulOpposite_instZero(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulOpposite_instZero___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulOpposite_instOne___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulOpposite_instOne___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulOpposite_instOne(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulOpposite_instOne___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddOpposite_instZero___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddOpposite_instZero___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddOpposite_instZero(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddOpposite_instZero___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulOpposite_instAdd___redArg___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulOpposite_instAdd___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulOpposite_instAdd(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulOpposite_instSub___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulOpposite_instSub(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulOpposite_instNeg___redArg___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulOpposite_instNeg___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulOpposite_instNeg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulOpposite_instInvolutiveNeg___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulOpposite_instInvolutiveNeg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulOpposite_instMul___redArg___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulOpposite_instMul___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulOpposite_instMul(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddOpposite_instAdd___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddOpposite_instAdd(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulOpposite_instInv___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulOpposite_instInv(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddOpposite_instNeg___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddOpposite_instNeg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulOpposite_instInvolutiveInv___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulOpposite_instInvolutiveInv(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddOpposite_instInvolutiveNeg___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddOpposite_instInvolutiveNeg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulOpposite_instSMul___redArg___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulOpposite_instSMul___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulOpposite_instSMul(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddOpposite_instVAdd___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddOpposite_instVAdd(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddOpposite_instOne___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddOpposite_instOne___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddOpposite_instOne(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddOpposite_instOne___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddOpposite_instMul___redArg___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddOpposite_instMul___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddOpposite_instMul(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddOpposite_instInv___redArg___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddOpposite_instInv___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddOpposite_instInv(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddOpposite_instInvolutiveInv___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddOpposite_instInvolutiveInv(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddOpposite_instDiv___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddOpposite_instDiv(lean_object*, lean_object*);
static lean_object* _init_lp_mathlib___aux__Mathlib__Algebra__Opposites______macroRules__term___u1d50_u1d52_u1d56__1___closed__6(void){
_start:
{
lean_object* v___x_22_; lean_object* v___x_23_; 
v___x_22_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Algebra__Opposites______macroRules__term___u1d50_u1d52_u1d56__1___closed__5));
v___x_23_ = l_String_toRawSubstring_x27(v___x_22_);
return v___x_23_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Algebra__Opposites______macroRules__term___u1d50_u1d52_u1d56__1(lean_object* v_x_35_, lean_object* v_a_36_, lean_object* v_a_37_){
_start:
{
lean_object* v___x_38_; uint8_t v___x_39_; 
v___x_38_ = ((lean_object*)(lp_mathlib_term___u1d50_u1d52_u1d56___closed__1));
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
v___x_49_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Algebra__Opposites______macroRules__term___u1d50_u1d52_u1d56__1___closed__4));
v___x_50_ = lean_obj_once(&lp_mathlib___aux__Mathlib__Algebra__Opposites______macroRules__term___u1d50_u1d52_u1d56__1___closed__6, &lp_mathlib___aux__Mathlib__Algebra__Opposites______macroRules__term___u1d50_u1d52_u1d56__1___closed__6_once, _init_lp_mathlib___aux__Mathlib__Algebra__Opposites______macroRules__term___u1d50_u1d52_u1d56__1___closed__6);
v___x_51_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Algebra__Opposites______macroRules__term___u1d50_u1d52_u1d56__1___closed__7));
lean_inc(v_currMacroScope_43_);
lean_inc(v_quotContext_42_);
v___x_52_ = l_Lean_addMacroScope(v_quotContext_42_, v___x_51_, v_currMacroScope_43_);
v___x_53_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Algebra__Opposites______macroRules__term___u1d50_u1d52_u1d56__1___closed__9));
lean_inc_n(v___x_48_, 2);
v___x_54_ = lean_alloc_ctor(3, 4, 0);
lean_ctor_set(v___x_54_, 0, v___x_48_);
lean_ctor_set(v___x_54_, 1, v___x_50_);
lean_ctor_set(v___x_54_, 2, v___x_52_);
lean_ctor_set(v___x_54_, 3, v___x_53_);
v___x_55_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Algebra__Opposites______macroRules__term___u1d50_u1d52_u1d56__1___closed__11));
v___x_56_ = l_Lean_Syntax_node1(v___x_48_, v___x_55_, v___x_46_);
v___x_57_ = l_Lean_Syntax_node2(v___x_48_, v___x_49_, v___x_54_, v___x_56_);
v___x_58_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_58_, 0, v___x_57_);
lean_ctor_set(v___x_58_, 1, v_a_37_);
return v___x_58_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Algebra__Opposites______macroRules__term___u1d50_u1d52_u1d56__1___boxed(lean_object* v_x_59_, lean_object* v_a_60_, lean_object* v_a_61_){
_start:
{
lean_object* v_res_62_; 
v_res_62_ = lp_mathlib___aux__Mathlib__Algebra__Opposites______macroRules__term___u1d50_u1d52_u1d56__1(v_x_59_, v_a_60_, v_a_61_);
lean_dec_ref(v_a_60_);
return v_res_62_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Algebra__Opposites______unexpand__MulOpposite__1(lean_object* v_x_66_, lean_object* v_a_67_, lean_object* v_a_68_){
_start:
{
lean_object* v___x_69_; uint8_t v___x_70_; 
v___x_69_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Algebra__Opposites______macroRules__term___u1d50_u1d52_u1d56__1___closed__4));
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
v___x_75_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Algebra__Opposites______unexpand__MulOpposite__1___closed__1));
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
v___x_88_ = ((lean_object*)(lp_mathlib_term___u1d50_u1d52_u1d56___closed__1));
v___x_89_ = ((lean_object*)(lp_mathlib_term___u1d50_u1d52_u1d56___closed__2));
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
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Algebra__Opposites______unexpand__MulOpposite__1___boxed(lean_object* v_x_93_, lean_object* v_a_94_, lean_object* v_a_95_){
_start:
{
lean_object* v_res_96_; 
v_res_96_ = lp_mathlib___aux__Mathlib__Algebra__Opposites______unexpand__MulOpposite__1(v_x_93_, v_a_94_, v_a_95_);
lean_dec(v_a_94_);
return v_res_96_;
}
}
static lean_object* _init_lp_mathlib___aux__Mathlib__Algebra__Opposites______macroRules__term___u1d43_u1d52_u1d56__1___closed__1(void){
_start:
{
lean_object* v___x_109_; lean_object* v___x_110_; 
v___x_109_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Algebra__Opposites______macroRules__term___u1d43_u1d52_u1d56__1___closed__0));
v___x_110_ = l_String_toRawSubstring_x27(v___x_109_);
return v___x_110_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Algebra__Opposites______macroRules__term___u1d43_u1d52_u1d56__1(lean_object* v_x_119_, lean_object* v_a_120_, lean_object* v_a_121_){
_start:
{
lean_object* v___x_122_; uint8_t v___x_123_; 
v___x_122_ = ((lean_object*)(lp_mathlib_term___u1d43_u1d52_u1d56___closed__1));
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
v___x_133_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Algebra__Opposites______macroRules__term___u1d50_u1d52_u1d56__1___closed__4));
v___x_134_ = lean_obj_once(&lp_mathlib___aux__Mathlib__Algebra__Opposites______macroRules__term___u1d43_u1d52_u1d56__1___closed__1, &lp_mathlib___aux__Mathlib__Algebra__Opposites______macroRules__term___u1d43_u1d52_u1d56__1___closed__1_once, _init_lp_mathlib___aux__Mathlib__Algebra__Opposites______macroRules__term___u1d43_u1d52_u1d56__1___closed__1);
v___x_135_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Algebra__Opposites______macroRules__term___u1d43_u1d52_u1d56__1___closed__2));
lean_inc(v_currMacroScope_127_);
lean_inc(v_quotContext_126_);
v___x_136_ = l_Lean_addMacroScope(v_quotContext_126_, v___x_135_, v_currMacroScope_127_);
v___x_137_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Algebra__Opposites______macroRules__term___u1d43_u1d52_u1d56__1___closed__4));
lean_inc_n(v___x_132_, 2);
v___x_138_ = lean_alloc_ctor(3, 4, 0);
lean_ctor_set(v___x_138_, 0, v___x_132_);
lean_ctor_set(v___x_138_, 1, v___x_134_);
lean_ctor_set(v___x_138_, 2, v___x_136_);
lean_ctor_set(v___x_138_, 3, v___x_137_);
v___x_139_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Algebra__Opposites______macroRules__term___u1d50_u1d52_u1d56__1___closed__11));
v___x_140_ = l_Lean_Syntax_node1(v___x_132_, v___x_139_, v___x_130_);
v___x_141_ = l_Lean_Syntax_node2(v___x_132_, v___x_133_, v___x_138_, v___x_140_);
v___x_142_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_142_, 0, v___x_141_);
lean_ctor_set(v___x_142_, 1, v_a_121_);
return v___x_142_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Algebra__Opposites______macroRules__term___u1d43_u1d52_u1d56__1___boxed(lean_object* v_x_143_, lean_object* v_a_144_, lean_object* v_a_145_){
_start:
{
lean_object* v_res_146_; 
v_res_146_ = lp_mathlib___aux__Mathlib__Algebra__Opposites______macroRules__term___u1d43_u1d52_u1d56__1(v_x_143_, v_a_144_, v_a_145_);
lean_dec_ref(v_a_144_);
return v_res_146_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Algebra__Opposites______unexpand__AddOpposite__1(lean_object* v_x_147_, lean_object* v_a_148_, lean_object* v_a_149_){
_start:
{
lean_object* v___x_150_; uint8_t v___x_151_; 
v___x_150_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Algebra__Opposites______macroRules__term___u1d50_u1d52_u1d56__1___closed__4));
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
v___x_156_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Algebra__Opposites______unexpand__MulOpposite__1___closed__1));
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
v___x_169_ = ((lean_object*)(lp_mathlib_term___u1d43_u1d52_u1d56___closed__1));
v___x_170_ = ((lean_object*)(lp_mathlib_term___u1d43_u1d52_u1d56___closed__2));
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
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Algebra__Opposites______unexpand__AddOpposite__1___boxed(lean_object* v_x_174_, lean_object* v_a_175_, lean_object* v_a_176_){
_start:
{
lean_object* v_res_177_; 
v_res_177_ = lp_mathlib___aux__Mathlib__Algebra__Opposites______unexpand__AddOpposite__1(v_x_174_, v_a_175_, v_a_176_);
lean_dec(v_a_175_);
return v_res_177_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulOpposite_op___redArg(lean_object* v_unop_x27_178_){
_start:
{
lean_inc(v_unop_x27_178_);
return v_unop_x27_178_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulOpposite_op___redArg___boxed(lean_object* v_unop_x27_179_){
_start:
{
lean_object* v_res_180_; 
v_res_180_ = lp_mathlib_MulOpposite_op___redArg(v_unop_x27_179_);
lean_dec(v_unop_x27_179_);
return v_res_180_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulOpposite_op(lean_object* v_00_u03b1_181_, lean_object* v_unop_x27_182_){
_start:
{
lean_inc(v_unop_x27_182_);
return v_unop_x27_182_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulOpposite_op___boxed(lean_object* v_00_u03b1_183_, lean_object* v_unop_x27_184_){
_start:
{
lean_object* v_res_185_; 
v_res_185_ = lp_mathlib_MulOpposite_op(v_00_u03b1_183_, v_unop_x27_184_);
lean_dec(v_unop_x27_184_);
return v_res_185_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddOpposite_op___redArg(lean_object* v_unop_x27_186_){
_start:
{
lean_inc(v_unop_x27_186_);
return v_unop_x27_186_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddOpposite_op___redArg___boxed(lean_object* v_unop_x27_187_){
_start:
{
lean_object* v_res_188_; 
v_res_188_ = lp_mathlib_AddOpposite_op___redArg(v_unop_x27_187_);
lean_dec(v_unop_x27_187_);
return v_res_188_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddOpposite_op(lean_object* v_00_u03b1_189_, lean_object* v_unop_x27_190_){
_start:
{
lean_inc(v_unop_x27_190_);
return v_unop_x27_190_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddOpposite_op___boxed(lean_object* v_00_u03b1_191_, lean_object* v_unop_x27_192_){
_start:
{
lean_object* v_res_193_; 
v_res_193_ = lp_mathlib_AddOpposite_op(v_00_u03b1_191_, v_unop_x27_192_);
lean_dec(v_unop_x27_192_);
return v_res_193_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulOpposite_unop___redArg(lean_object* v_self_194_){
_start:
{
lean_inc(v_self_194_);
return v_self_194_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulOpposite_unop___redArg___boxed(lean_object* v_self_195_){
_start:
{
lean_object* v_res_196_; 
v_res_196_ = lp_mathlib_MulOpposite_unop___redArg(v_self_195_);
lean_dec(v_self_195_);
return v_res_196_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulOpposite_unop(lean_object* v_00_u03b1_197_, lean_object* v_self_198_){
_start:
{
lean_inc(v_self_198_);
return v_self_198_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulOpposite_unop___boxed(lean_object* v_00_u03b1_199_, lean_object* v_self_200_){
_start:
{
lean_object* v_res_201_; 
v_res_201_ = lp_mathlib_MulOpposite_unop(v_00_u03b1_199_, v_self_200_);
lean_dec(v_self_200_);
return v_res_201_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddOpposite_unop___redArg(lean_object* v_self_202_){
_start:
{
lean_inc(v_self_202_);
return v_self_202_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddOpposite_unop___redArg___boxed(lean_object* v_self_203_){
_start:
{
lean_object* v_res_204_; 
v_res_204_ = lp_mathlib_AddOpposite_unop___redArg(v_self_203_);
lean_dec(v_self_203_);
return v_res_204_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddOpposite_unop(lean_object* v_00_u03b1_205_, lean_object* v_self_206_){
_start:
{
lean_inc(v_self_206_);
return v_self_206_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddOpposite_unop___boxed(lean_object* v_00_u03b1_207_, lean_object* v_self_208_){
_start:
{
lean_object* v_res_209_; 
v_res_209_ = lp_mathlib_AddOpposite_unop(v_00_u03b1_207_, v_self_208_);
lean_dec(v_self_208_);
return v_res_209_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulOpposite_rec_x27___redArg(lean_object* v_h_210_, lean_object* v_X_211_){
_start:
{
lean_object* v___x_212_; 
v___x_212_ = lean_apply_1(v_h_210_, v_X_211_);
return v___x_212_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulOpposite_rec_x27(lean_object* v_00_u03b1_213_, lean_object* v_F_214_, lean_object* v_h_215_, lean_object* v_X_216_){
_start:
{
lean_object* v___x_217_; 
v___x_217_ = lean_apply_1(v_h_215_, v_X_216_);
return v___x_217_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddOpposite_rec_x27___redArg(lean_object* v_h_218_, lean_object* v_X_219_){
_start:
{
lean_object* v___x_220_; 
v___x_220_ = lean_apply_1(v_h_218_, v_X_219_);
return v___x_220_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddOpposite_rec_x27(lean_object* v_00_u03b1_221_, lean_object* v_F_222_, lean_object* v_h_223_, lean_object* v_X_224_){
_start:
{
lean_object* v___x_225_; 
v___x_225_ = lean_apply_1(v_h_223_, v_X_224_);
return v___x_225_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulOpposite_opEquiv___lam__0(lean_object* v___y_226_){
_start:
{
lean_inc(v___y_226_);
return v___y_226_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulOpposite_opEquiv___lam__0___boxed(lean_object* v___y_227_){
_start:
{
lean_object* v_res_228_; 
v_res_228_ = lp_mathlib_MulOpposite_opEquiv___lam__0(v___y_227_);
lean_dec(v___y_227_);
return v_res_228_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulOpposite_opEquiv(lean_object* v_00_u03b1_232_){
_start:
{
lean_object* v___x_233_; 
v___x_233_ = ((lean_object*)(lp_mathlib_MulOpposite_opEquiv___closed__1));
return v___x_233_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddOpposite_opEquiv(lean_object* v_00_u03b1_234_){
_start:
{
lean_object* v___x_235_; 
v___x_235_ = ((lean_object*)(lp_mathlib_MulOpposite_opEquiv___closed__1));
return v___x_235_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulOpposite_instInhabited___redArg(lean_object* v_inst_236_){
_start:
{
lean_inc(v_inst_236_);
return v_inst_236_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulOpposite_instInhabited___redArg___boxed(lean_object* v_inst_237_){
_start:
{
lean_object* v_res_238_; 
v_res_238_ = lp_mathlib_MulOpposite_instInhabited___redArg(v_inst_237_);
lean_dec(v_inst_237_);
return v_res_238_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulOpposite_instInhabited(lean_object* v_00_u03b1_239_, lean_object* v_inst_240_){
_start:
{
lean_inc(v_inst_240_);
return v_inst_240_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulOpposite_instInhabited___boxed(lean_object* v_00_u03b1_241_, lean_object* v_inst_242_){
_start:
{
lean_object* v_res_243_; 
v_res_243_ = lp_mathlib_MulOpposite_instInhabited(v_00_u03b1_241_, v_inst_242_);
lean_dec(v_inst_242_);
return v_res_243_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddOpposite_instInhabited___redArg(lean_object* v_inst_244_){
_start:
{
lean_inc(v_inst_244_);
return v_inst_244_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddOpposite_instInhabited___redArg___boxed(lean_object* v_inst_245_){
_start:
{
lean_object* v_res_246_; 
v_res_246_ = lp_mathlib_AddOpposite_instInhabited___redArg(v_inst_245_);
lean_dec(v_inst_245_);
return v_res_246_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddOpposite_instInhabited(lean_object* v_00_u03b1_247_, lean_object* v_inst_248_){
_start:
{
lean_inc(v_inst_248_);
return v_inst_248_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddOpposite_instInhabited___boxed(lean_object* v_00_u03b1_249_, lean_object* v_inst_250_){
_start:
{
lean_object* v_res_251_; 
v_res_251_ = lp_mathlib_AddOpposite_instInhabited(v_00_u03b1_249_, v_inst_250_);
lean_dec(v_inst_250_);
return v_res_251_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulOpposite_instUnique___redArg(lean_object* v_inst_252_){
_start:
{
lean_inc(v_inst_252_);
return v_inst_252_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulOpposite_instUnique___redArg___boxed(lean_object* v_inst_253_){
_start:
{
lean_object* v_res_254_; 
v_res_254_ = lp_mathlib_MulOpposite_instUnique___redArg(v_inst_253_);
lean_dec(v_inst_253_);
return v_res_254_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulOpposite_instUnique(lean_object* v_00_u03b1_255_, lean_object* v_inst_256_){
_start:
{
lean_inc(v_inst_256_);
return v_inst_256_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulOpposite_instUnique___boxed(lean_object* v_00_u03b1_257_, lean_object* v_inst_258_){
_start:
{
lean_object* v_res_259_; 
v_res_259_ = lp_mathlib_MulOpposite_instUnique(v_00_u03b1_257_, v_inst_258_);
lean_dec(v_inst_258_);
return v_res_259_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddOpposite_instUnique___redArg(lean_object* v_inst_260_){
_start:
{
lean_inc(v_inst_260_);
return v_inst_260_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddOpposite_instUnique___redArg___boxed(lean_object* v_inst_261_){
_start:
{
lean_object* v_res_262_; 
v_res_262_ = lp_mathlib_AddOpposite_instUnique___redArg(v_inst_261_);
lean_dec(v_inst_261_);
return v_res_262_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddOpposite_instUnique(lean_object* v_00_u03b1_263_, lean_object* v_inst_264_){
_start:
{
lean_inc(v_inst_264_);
return v_inst_264_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddOpposite_instUnique___boxed(lean_object* v_00_u03b1_265_, lean_object* v_inst_266_){
_start:
{
lean_object* v_res_267_; 
v_res_267_ = lp_mathlib_AddOpposite_instUnique(v_00_u03b1_265_, v_inst_266_);
lean_dec(v_inst_266_);
return v_res_267_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_MulOpposite_instDecidableEq___redArg(lean_object* v_inst_269_, lean_object* v_a_270_, lean_object* v_b_271_){
_start:
{
lean_object* v___x_272_; uint8_t v___x_273_; 
v___x_272_ = ((lean_object*)(lp_mathlib_MulOpposite_instDecidableEq___redArg___closed__0));
v___x_273_ = lp_mathlib_Function_Injective_decidableEq___redArg(v___x_272_, v_inst_269_, v_a_270_, v_b_271_);
return v___x_273_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulOpposite_instDecidableEq___redArg___boxed(lean_object* v_inst_274_, lean_object* v_a_275_, lean_object* v_b_276_){
_start:
{
uint8_t v_res_277_; lean_object* v_r_278_; 
v_res_277_ = lp_mathlib_MulOpposite_instDecidableEq___redArg(v_inst_274_, v_a_275_, v_b_276_);
v_r_278_ = lean_box(v_res_277_);
return v_r_278_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_MulOpposite_instDecidableEq(lean_object* v_00_u03b1_279_, lean_object* v_inst_280_, lean_object* v_a_281_, lean_object* v_b_282_){
_start:
{
uint8_t v___x_283_; 
v___x_283_ = lp_mathlib_MulOpposite_instDecidableEq___redArg(v_inst_280_, v_a_281_, v_b_282_);
return v___x_283_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulOpposite_instDecidableEq___boxed(lean_object* v_00_u03b1_284_, lean_object* v_inst_285_, lean_object* v_a_286_, lean_object* v_b_287_){
_start:
{
uint8_t v_res_288_; lean_object* v_r_289_; 
v_res_288_ = lp_mathlib_MulOpposite_instDecidableEq(v_00_u03b1_284_, v_inst_285_, v_a_286_, v_b_287_);
v_r_289_ = lean_box(v_res_288_);
return v_r_289_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_AddOpposite_instDecidableEq___redArg(lean_object* v_inst_291_, lean_object* v_a_292_, lean_object* v_b_293_){
_start:
{
lean_object* v___x_294_; uint8_t v___x_295_; 
v___x_294_ = ((lean_object*)(lp_mathlib_AddOpposite_instDecidableEq___redArg___closed__0));
v___x_295_ = lp_mathlib_Function_Injective_decidableEq___redArg(v___x_294_, v_inst_291_, v_a_292_, v_b_293_);
return v___x_295_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddOpposite_instDecidableEq___redArg___boxed(lean_object* v_inst_296_, lean_object* v_a_297_, lean_object* v_b_298_){
_start:
{
uint8_t v_res_299_; lean_object* v_r_300_; 
v_res_299_ = lp_mathlib_AddOpposite_instDecidableEq___redArg(v_inst_296_, v_a_297_, v_b_298_);
v_r_300_ = lean_box(v_res_299_);
return v_r_300_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_AddOpposite_instDecidableEq(lean_object* v_00_u03b1_301_, lean_object* v_inst_302_, lean_object* v_a_303_, lean_object* v_b_304_){
_start:
{
uint8_t v___x_305_; 
v___x_305_ = lp_mathlib_AddOpposite_instDecidableEq___redArg(v_inst_302_, v_a_303_, v_b_304_);
return v___x_305_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddOpposite_instDecidableEq___boxed(lean_object* v_00_u03b1_306_, lean_object* v_inst_307_, lean_object* v_a_308_, lean_object* v_b_309_){
_start:
{
uint8_t v_res_310_; lean_object* v_r_311_; 
v_res_310_ = lp_mathlib_AddOpposite_instDecidableEq(v_00_u03b1_306_, v_inst_307_, v_a_308_, v_b_309_);
v_r_311_ = lean_box(v_res_310_);
return v_r_311_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulOpposite_instZero___redArg(lean_object* v_inst_312_){
_start:
{
lean_inc(v_inst_312_);
return v_inst_312_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulOpposite_instZero___redArg___boxed(lean_object* v_inst_313_){
_start:
{
lean_object* v_res_314_; 
v_res_314_ = lp_mathlib_MulOpposite_instZero___redArg(v_inst_313_);
lean_dec(v_inst_313_);
return v_res_314_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulOpposite_instZero(lean_object* v_00_u03b1_315_, lean_object* v_inst_316_){
_start:
{
lean_inc(v_inst_316_);
return v_inst_316_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulOpposite_instZero___boxed(lean_object* v_00_u03b1_317_, lean_object* v_inst_318_){
_start:
{
lean_object* v_res_319_; 
v_res_319_ = lp_mathlib_MulOpposite_instZero(v_00_u03b1_317_, v_inst_318_);
lean_dec(v_inst_318_);
return v_res_319_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulOpposite_instOne___redArg(lean_object* v_inst_320_){
_start:
{
lean_inc(v_inst_320_);
return v_inst_320_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulOpposite_instOne___redArg___boxed(lean_object* v_inst_321_){
_start:
{
lean_object* v_res_322_; 
v_res_322_ = lp_mathlib_MulOpposite_instOne___redArg(v_inst_321_);
lean_dec(v_inst_321_);
return v_res_322_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulOpposite_instOne(lean_object* v_00_u03b1_323_, lean_object* v_inst_324_){
_start:
{
lean_inc(v_inst_324_);
return v_inst_324_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulOpposite_instOne___boxed(lean_object* v_00_u03b1_325_, lean_object* v_inst_326_){
_start:
{
lean_object* v_res_327_; 
v_res_327_ = lp_mathlib_MulOpposite_instOne(v_00_u03b1_325_, v_inst_326_);
lean_dec(v_inst_326_);
return v_res_327_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddOpposite_instZero___redArg(lean_object* v_inst_328_){
_start:
{
lean_inc(v_inst_328_);
return v_inst_328_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddOpposite_instZero___redArg___boxed(lean_object* v_inst_329_){
_start:
{
lean_object* v_res_330_; 
v_res_330_ = lp_mathlib_AddOpposite_instZero___redArg(v_inst_329_);
lean_dec(v_inst_329_);
return v_res_330_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddOpposite_instZero(lean_object* v_00_u03b1_331_, lean_object* v_inst_332_){
_start:
{
lean_inc(v_inst_332_);
return v_inst_332_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddOpposite_instZero___boxed(lean_object* v_00_u03b1_333_, lean_object* v_inst_334_){
_start:
{
lean_object* v_res_335_; 
v_res_335_ = lp_mathlib_AddOpposite_instZero(v_00_u03b1_333_, v_inst_334_);
lean_dec(v_inst_334_);
return v_res_335_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulOpposite_instAdd___redArg___lam__0(lean_object* v_inst_336_, lean_object* v_x_337_, lean_object* v_y_338_){
_start:
{
lean_object* v___x_339_; 
v___x_339_ = lean_apply_2(v_inst_336_, v_x_337_, v_y_338_);
return v___x_339_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulOpposite_instAdd___redArg(lean_object* v_inst_340_){
_start:
{
lean_object* v___f_341_; 
v___f_341_ = lean_alloc_closure((void*)(lp_mathlib_MulOpposite_instAdd___redArg___lam__0), 3, 1);
lean_closure_set(v___f_341_, 0, v_inst_340_);
return v___f_341_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulOpposite_instAdd(lean_object* v_00_u03b1_342_, lean_object* v_inst_343_){
_start:
{
lean_object* v___f_344_; 
v___f_344_ = lean_alloc_closure((void*)(lp_mathlib_MulOpposite_instAdd___redArg___lam__0), 3, 1);
lean_closure_set(v___f_344_, 0, v_inst_343_);
return v___f_344_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulOpposite_instSub___redArg(lean_object* v_inst_345_){
_start:
{
lean_object* v___f_346_; 
v___f_346_ = lean_alloc_closure((void*)(lp_mathlib_MulOpposite_instAdd___redArg___lam__0), 3, 1);
lean_closure_set(v___f_346_, 0, v_inst_345_);
return v___f_346_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulOpposite_instSub(lean_object* v_00_u03b1_347_, lean_object* v_inst_348_){
_start:
{
lean_object* v___f_349_; 
v___f_349_ = lean_alloc_closure((void*)(lp_mathlib_MulOpposite_instAdd___redArg___lam__0), 3, 1);
lean_closure_set(v___f_349_, 0, v_inst_348_);
return v___f_349_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulOpposite_instNeg___redArg___lam__0(lean_object* v_inst_350_, lean_object* v_x_351_){
_start:
{
lean_object* v___x_352_; 
v___x_352_ = lean_apply_1(v_inst_350_, v_x_351_);
return v___x_352_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulOpposite_instNeg___redArg(lean_object* v_inst_353_){
_start:
{
lean_object* v___f_354_; 
v___f_354_ = lean_alloc_closure((void*)(lp_mathlib_MulOpposite_instNeg___redArg___lam__0), 2, 1);
lean_closure_set(v___f_354_, 0, v_inst_353_);
return v___f_354_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulOpposite_instNeg(lean_object* v_00_u03b1_355_, lean_object* v_inst_356_){
_start:
{
lean_object* v___f_357_; 
v___f_357_ = lean_alloc_closure((void*)(lp_mathlib_MulOpposite_instNeg___redArg___lam__0), 2, 1);
lean_closure_set(v___f_357_, 0, v_inst_356_);
return v___f_357_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulOpposite_instInvolutiveNeg___redArg(lean_object* v_inst_358_){
_start:
{
lean_object* v___f_359_; 
v___f_359_ = lean_alloc_closure((void*)(lp_mathlib_MulOpposite_instNeg___redArg___lam__0), 2, 1);
lean_closure_set(v___f_359_, 0, v_inst_358_);
return v___f_359_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulOpposite_instInvolutiveNeg(lean_object* v_00_u03b1_360_, lean_object* v_inst_361_){
_start:
{
lean_object* v___f_362_; 
v___f_362_ = lean_alloc_closure((void*)(lp_mathlib_MulOpposite_instNeg___redArg___lam__0), 2, 1);
lean_closure_set(v___f_362_, 0, v_inst_361_);
return v___f_362_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulOpposite_instMul___redArg___lam__0(lean_object* v_inst_363_, lean_object* v_x_364_, lean_object* v_y_365_){
_start:
{
lean_object* v___x_366_; 
v___x_366_ = lean_apply_2(v_inst_363_, v_y_365_, v_x_364_);
return v___x_366_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulOpposite_instMul___redArg(lean_object* v_inst_367_){
_start:
{
lean_object* v___f_368_; 
v___f_368_ = lean_alloc_closure((void*)(lp_mathlib_MulOpposite_instMul___redArg___lam__0), 3, 1);
lean_closure_set(v___f_368_, 0, v_inst_367_);
return v___f_368_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulOpposite_instMul(lean_object* v_00_u03b1_369_, lean_object* v_inst_370_){
_start:
{
lean_object* v___f_371_; 
v___f_371_ = lean_alloc_closure((void*)(lp_mathlib_MulOpposite_instMul___redArg___lam__0), 3, 1);
lean_closure_set(v___f_371_, 0, v_inst_370_);
return v___f_371_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddOpposite_instAdd___redArg(lean_object* v_inst_372_){
_start:
{
lean_object* v___f_373_; 
v___f_373_ = lean_alloc_closure((void*)(lp_mathlib_MulOpposite_instMul___redArg___lam__0), 3, 1);
lean_closure_set(v___f_373_, 0, v_inst_372_);
return v___f_373_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddOpposite_instAdd(lean_object* v_00_u03b1_374_, lean_object* v_inst_375_){
_start:
{
lean_object* v___f_376_; 
v___f_376_ = lean_alloc_closure((void*)(lp_mathlib_MulOpposite_instMul___redArg___lam__0), 3, 1);
lean_closure_set(v___f_376_, 0, v_inst_375_);
return v___f_376_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulOpposite_instInv___redArg(lean_object* v_inst_377_){
_start:
{
lean_object* v___f_378_; 
v___f_378_ = lean_alloc_closure((void*)(lp_mathlib_MulOpposite_instNeg___redArg___lam__0), 2, 1);
lean_closure_set(v___f_378_, 0, v_inst_377_);
return v___f_378_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulOpposite_instInv(lean_object* v_00_u03b1_379_, lean_object* v_inst_380_){
_start:
{
lean_object* v___f_381_; 
v___f_381_ = lean_alloc_closure((void*)(lp_mathlib_MulOpposite_instNeg___redArg___lam__0), 2, 1);
lean_closure_set(v___f_381_, 0, v_inst_380_);
return v___f_381_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddOpposite_instNeg___redArg(lean_object* v_inst_382_){
_start:
{
lean_object* v___f_383_; 
v___f_383_ = lean_alloc_closure((void*)(lp_mathlib_MulOpposite_instNeg___redArg___lam__0), 2, 1);
lean_closure_set(v___f_383_, 0, v_inst_382_);
return v___f_383_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddOpposite_instNeg(lean_object* v_00_u03b1_384_, lean_object* v_inst_385_){
_start:
{
lean_object* v___f_386_; 
v___f_386_ = lean_alloc_closure((void*)(lp_mathlib_MulOpposite_instNeg___redArg___lam__0), 2, 1);
lean_closure_set(v___f_386_, 0, v_inst_385_);
return v___f_386_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulOpposite_instInvolutiveInv___redArg(lean_object* v_inst_387_){
_start:
{
lean_object* v___f_388_; 
v___f_388_ = lean_alloc_closure((void*)(lp_mathlib_MulOpposite_instNeg___redArg___lam__0), 2, 1);
lean_closure_set(v___f_388_, 0, v_inst_387_);
return v___f_388_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulOpposite_instInvolutiveInv(lean_object* v_00_u03b1_389_, lean_object* v_inst_390_){
_start:
{
lean_object* v___f_391_; 
v___f_391_ = lean_alloc_closure((void*)(lp_mathlib_MulOpposite_instNeg___redArg___lam__0), 2, 1);
lean_closure_set(v___f_391_, 0, v_inst_390_);
return v___f_391_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddOpposite_instInvolutiveNeg___redArg(lean_object* v_inst_392_){
_start:
{
lean_object* v___f_393_; 
v___f_393_ = lean_alloc_closure((void*)(lp_mathlib_MulOpposite_instNeg___redArg___lam__0), 2, 1);
lean_closure_set(v___f_393_, 0, v_inst_392_);
return v___f_393_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddOpposite_instInvolutiveNeg(lean_object* v_00_u03b1_394_, lean_object* v_inst_395_){
_start:
{
lean_object* v___f_396_; 
v___f_396_ = lean_alloc_closure((void*)(lp_mathlib_MulOpposite_instNeg___redArg___lam__0), 2, 1);
lean_closure_set(v___f_396_, 0, v_inst_395_);
return v___f_396_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulOpposite_instSMul___redArg___lam__0(lean_object* v_inst_397_, lean_object* v_c_398_, lean_object* v_x_399_){
_start:
{
lean_object* v___x_400_; 
v___x_400_ = lean_apply_2(v_inst_397_, v_c_398_, v_x_399_);
return v___x_400_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulOpposite_instSMul___redArg(lean_object* v_inst_401_){
_start:
{
lean_object* v___f_402_; 
v___f_402_ = lean_alloc_closure((void*)(lp_mathlib_MulOpposite_instSMul___redArg___lam__0), 3, 1);
lean_closure_set(v___f_402_, 0, v_inst_401_);
return v___f_402_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulOpposite_instSMul(lean_object* v_00_u03b1_403_, lean_object* v_00_u03b2_404_, lean_object* v_inst_405_){
_start:
{
lean_object* v___f_406_; 
v___f_406_ = lean_alloc_closure((void*)(lp_mathlib_MulOpposite_instSMul___redArg___lam__0), 3, 1);
lean_closure_set(v___f_406_, 0, v_inst_405_);
return v___f_406_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddOpposite_instVAdd___redArg(lean_object* v_inst_407_){
_start:
{
lean_object* v___f_408_; 
v___f_408_ = lean_alloc_closure((void*)(lp_mathlib_MulOpposite_instSMul___redArg___lam__0), 3, 1);
lean_closure_set(v___f_408_, 0, v_inst_407_);
return v___f_408_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddOpposite_instVAdd(lean_object* v_00_u03b1_409_, lean_object* v_00_u03b2_410_, lean_object* v_inst_411_){
_start:
{
lean_object* v___f_412_; 
v___f_412_ = lean_alloc_closure((void*)(lp_mathlib_MulOpposite_instSMul___redArg___lam__0), 3, 1);
lean_closure_set(v___f_412_, 0, v_inst_411_);
return v___f_412_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddOpposite_instOne___redArg(lean_object* v_inst_413_){
_start:
{
lean_inc(v_inst_413_);
return v_inst_413_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddOpposite_instOne___redArg___boxed(lean_object* v_inst_414_){
_start:
{
lean_object* v_res_415_; 
v_res_415_ = lp_mathlib_AddOpposite_instOne___redArg(v_inst_414_);
lean_dec(v_inst_414_);
return v_res_415_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddOpposite_instOne(lean_object* v_00_u03b1_416_, lean_object* v_inst_417_){
_start:
{
lean_inc(v_inst_417_);
return v_inst_417_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddOpposite_instOne___boxed(lean_object* v_00_u03b1_418_, lean_object* v_inst_419_){
_start:
{
lean_object* v_res_420_; 
v_res_420_ = lp_mathlib_AddOpposite_instOne(v_00_u03b1_418_, v_inst_419_);
lean_dec(v_inst_419_);
return v_res_420_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddOpposite_instMul___redArg___lam__0(lean_object* v_inst_421_, lean_object* v_a_422_, lean_object* v_b_423_){
_start:
{
lean_object* v___x_424_; 
v___x_424_ = lean_apply_2(v_inst_421_, v_a_422_, v_b_423_);
return v___x_424_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddOpposite_instMul___redArg(lean_object* v_inst_425_){
_start:
{
lean_object* v___f_426_; 
v___f_426_ = lean_alloc_closure((void*)(lp_mathlib_AddOpposite_instMul___redArg___lam__0), 3, 1);
lean_closure_set(v___f_426_, 0, v_inst_425_);
return v___f_426_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddOpposite_instMul(lean_object* v_00_u03b1_427_, lean_object* v_inst_428_){
_start:
{
lean_object* v___f_429_; 
v___f_429_ = lean_alloc_closure((void*)(lp_mathlib_AddOpposite_instMul___redArg___lam__0), 3, 1);
lean_closure_set(v___f_429_, 0, v_inst_428_);
return v___f_429_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddOpposite_instInv___redArg___lam__0(lean_object* v_inst_430_, lean_object* v_a_431_){
_start:
{
lean_object* v___x_432_; 
v___x_432_ = lean_apply_1(v_inst_430_, v_a_431_);
return v___x_432_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddOpposite_instInv___redArg(lean_object* v_inst_433_){
_start:
{
lean_object* v___f_434_; 
v___f_434_ = lean_alloc_closure((void*)(lp_mathlib_AddOpposite_instInv___redArg___lam__0), 2, 1);
lean_closure_set(v___f_434_, 0, v_inst_433_);
return v___f_434_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddOpposite_instInv(lean_object* v_00_u03b1_435_, lean_object* v_inst_436_){
_start:
{
lean_object* v___f_437_; 
v___f_437_ = lean_alloc_closure((void*)(lp_mathlib_AddOpposite_instInv___redArg___lam__0), 2, 1);
lean_closure_set(v___f_437_, 0, v_inst_436_);
return v___f_437_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddOpposite_instInvolutiveInv___redArg(lean_object* v_inst_438_){
_start:
{
lean_object* v___f_439_; 
v___f_439_ = lean_alloc_closure((void*)(lp_mathlib_AddOpposite_instInv___redArg___lam__0), 2, 1);
lean_closure_set(v___f_439_, 0, v_inst_438_);
return v___f_439_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddOpposite_instInvolutiveInv(lean_object* v_00_u03b1_440_, lean_object* v_inst_441_){
_start:
{
lean_object* v___f_442_; 
v___f_442_ = lean_alloc_closure((void*)(lp_mathlib_AddOpposite_instInv___redArg___lam__0), 2, 1);
lean_closure_set(v___f_442_, 0, v_inst_441_);
return v___f_442_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddOpposite_instDiv___redArg(lean_object* v_inst_443_){
_start:
{
lean_object* v___f_444_; 
v___f_444_ = lean_alloc_closure((void*)(lp_mathlib_AddOpposite_instMul___redArg___lam__0), 3, 1);
lean_closure_set(v___f_444_, 0, v_inst_443_);
return v___f_444_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddOpposite_instDiv(lean_object* v_00_u03b1_445_, lean_object* v_inst_446_){
_start:
{
lean_object* v___f_447_; 
v___f_447_ = lean_alloc_closure((void*)(lp_mathlib_AddOpposite_instMul___redArg___lam__0), 3, 1);
lean_closure_set(v___f_447_, 0, v_inst_446_);
return v___f_447_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Group_DivInvMonoid(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Notation_Defs(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Logic_Equiv_Defs(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Opposites(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Group_DivInvMonoid(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Notation_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Logic_Equiv_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_Algebra_Opposites(uint8_t builtin) {
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
lean_object* initialize_mathlib_Mathlib_Algebra_Group_DivInvMonoid(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Notation_Defs(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Logic_Equiv_Defs(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Algebra_Opposites(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Group_DivInvMonoid(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Notation_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Logic_Equiv_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Opposites(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_Algebra_Opposites(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_Algebra_Opposites(builtin);
}
#ifdef __cplusplus
}
#endif
