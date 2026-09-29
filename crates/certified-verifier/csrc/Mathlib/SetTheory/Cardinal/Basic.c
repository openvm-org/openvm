// Lean compiler output
// Module: Mathlib.SetTheory.Cardinal.Basic
// Imports: public import Init public meta import Init public import Mathlib.Basic.Countable.Small public import Mathlib.Basic.UnivLE public import Mathlib.Data.Fintype.BigOperators public import Mathlib.Data.Fintype.Powerset public import Mathlib.Data.Nat.Cast.Order.Basic public import Mathlib.Data.Set.Countable public import Mathlib.Logic.Small.Set public import Mathlib.SetTheory.Cardinal.Order
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
lean_object* l_Lean_Name_mkStr2(lean_object*, lean_object*);
lean_object* l_Lean_Name_mkStr4(lean_object*, lean_object*, lean_object*, lean_object*);
uint8_t l_Lean_Syntax_isOfKind(lean_object*, lean_object*);
lean_object* l_Lean_Syntax_getArg(lean_object*, lean_object*);
uint8_t l_Lean_Syntax_matchesNull(lean_object*, lean_object*);
lean_object* l_Lean_replaceRef(lean_object*, lean_object*);
lean_object* l_Lean_SourceInfo_fromRef(lean_object*, uint8_t);
lean_object* l_Lean_Syntax_node3(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_String_toRawSubstring_x27(lean_object*);
lean_object* l_Lean_addMacroScope(lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Syntax_node2(lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Cardinal_term___x5e_x3c___00__closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "Cardinal"};
static const lean_object* lp_mathlib_Cardinal_term___x5e_x3c___00__closed__0 = (const lean_object*)&lp_mathlib_Cardinal_term___x5e_x3c___00__closed__0_value;
static const lean_string_object lp_mathlib_Cardinal_term___x5e_x3c___00__closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "term_^<_"};
static const lean_object* lp_mathlib_Cardinal_term___x5e_x3c___00__closed__1 = (const lean_object*)&lp_mathlib_Cardinal_term___x5e_x3c___00__closed__1_value;
static const lean_ctor_object lp_mathlib_Cardinal_term___x5e_x3c___00__closed__2_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Cardinal_term___x5e_x3c___00__closed__0_value),LEAN_SCALAR_PTR_LITERAL(247, 176, 10, 15, 146, 217, 130, 31)}};
static const lean_ctor_object lp_mathlib_Cardinal_term___x5e_x3c___00__closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Cardinal_term___x5e_x3c___00__closed__2_value_aux_0),((lean_object*)&lp_mathlib_Cardinal_term___x5e_x3c___00__closed__1_value),LEAN_SCALAR_PTR_LITERAL(25, 240, 252, 126, 146, 7, 97, 194)}};
static const lean_object* lp_mathlib_Cardinal_term___x5e_x3c___00__closed__2 = (const lean_object*)&lp_mathlib_Cardinal_term___x5e_x3c___00__closed__2_value;
static const lean_string_object lp_mathlib_Cardinal_term___x5e_x3c___00__closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "andthen"};
static const lean_object* lp_mathlib_Cardinal_term___x5e_x3c___00__closed__3 = (const lean_object*)&lp_mathlib_Cardinal_term___x5e_x3c___00__closed__3_value;
static const lean_ctor_object lp_mathlib_Cardinal_term___x5e_x3c___00__closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Cardinal_term___x5e_x3c___00__closed__3_value),LEAN_SCALAR_PTR_LITERAL(40, 255, 78, 30, 143, 119, 117, 174)}};
static const lean_object* lp_mathlib_Cardinal_term___x5e_x3c___00__closed__4 = (const lean_object*)&lp_mathlib_Cardinal_term___x5e_x3c___00__closed__4_value;
static const lean_string_object lp_mathlib_Cardinal_term___x5e_x3c___00__closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = " ^< "};
static const lean_object* lp_mathlib_Cardinal_term___x5e_x3c___00__closed__5 = (const lean_object*)&lp_mathlib_Cardinal_term___x5e_x3c___00__closed__5_value;
static const lean_ctor_object lp_mathlib_Cardinal_term___x5e_x3c___00__closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 5}, .m_objs = {((lean_object*)&lp_mathlib_Cardinal_term___x5e_x3c___00__closed__5_value)}};
static const lean_object* lp_mathlib_Cardinal_term___x5e_x3c___00__closed__6 = (const lean_object*)&lp_mathlib_Cardinal_term___x5e_x3c___00__closed__6_value;
static const lean_string_object lp_mathlib_Cardinal_term___x5e_x3c___00__closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "term"};
static const lean_object* lp_mathlib_Cardinal_term___x5e_x3c___00__closed__7 = (const lean_object*)&lp_mathlib_Cardinal_term___x5e_x3c___00__closed__7_value;
static const lean_ctor_object lp_mathlib_Cardinal_term___x5e_x3c___00__closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Cardinal_term___x5e_x3c___00__closed__7_value),LEAN_SCALAR_PTR_LITERAL(187, 230, 181, 162, 253, 146, 122, 119)}};
static const lean_object* lp_mathlib_Cardinal_term___x5e_x3c___00__closed__8 = (const lean_object*)&lp_mathlib_Cardinal_term___x5e_x3c___00__closed__8_value;
static const lean_ctor_object lp_mathlib_Cardinal_term___x5e_x3c___00__closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 7}, .m_objs = {((lean_object*)&lp_mathlib_Cardinal_term___x5e_x3c___00__closed__8_value),((lean_object*)(((size_t)(81) << 1) | 1))}};
static const lean_object* lp_mathlib_Cardinal_term___x5e_x3c___00__closed__9 = (const lean_object*)&lp_mathlib_Cardinal_term___x5e_x3c___00__closed__9_value;
static const lean_ctor_object lp_mathlib_Cardinal_term___x5e_x3c___00__closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib_Cardinal_term___x5e_x3c___00__closed__4_value),((lean_object*)&lp_mathlib_Cardinal_term___x5e_x3c___00__closed__6_value),((lean_object*)&lp_mathlib_Cardinal_term___x5e_x3c___00__closed__9_value)}};
static const lean_object* lp_mathlib_Cardinal_term___x5e_x3c___00__closed__10 = (const lean_object*)&lp_mathlib_Cardinal_term___x5e_x3c___00__closed__10_value;
static const lean_ctor_object lp_mathlib_Cardinal_term___x5e_x3c___00__closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*4 + 0, .m_other = 4, .m_tag = 4}, .m_objs = {((lean_object*)&lp_mathlib_Cardinal_term___x5e_x3c___00__closed__2_value),((lean_object*)(((size_t)(80) << 1) | 1)),((lean_object*)(((size_t)(80) << 1) | 1)),((lean_object*)&lp_mathlib_Cardinal_term___x5e_x3c___00__closed__10_value)}};
static const lean_object* lp_mathlib_Cardinal_term___x5e_x3c___00__closed__11 = (const lean_object*)&lp_mathlib_Cardinal_term___x5e_x3c___00__closed__11_value;
LEAN_EXPORT const lean_object* lp_mathlib_Cardinal_term___x5e_x3c__ = (const lean_object*)&lp_mathlib_Cardinal_term___x5e_x3c___00__closed__11_value;
static const lean_string_object lp_mathlib_Cardinal___aux__Mathlib__SetTheory__Cardinal__Basic______macroRules__Cardinal__term___x5e_x3c____1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "Lean"};
static const lean_object* lp_mathlib_Cardinal___aux__Mathlib__SetTheory__Cardinal__Basic______macroRules__Cardinal__term___x5e_x3c____1___closed__0 = (const lean_object*)&lp_mathlib_Cardinal___aux__Mathlib__SetTheory__Cardinal__Basic______macroRules__Cardinal__term___x5e_x3c____1___closed__0_value;
static const lean_string_object lp_mathlib_Cardinal___aux__Mathlib__SetTheory__Cardinal__Basic______macroRules__Cardinal__term___x5e_x3c____1___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "Parser"};
static const lean_object* lp_mathlib_Cardinal___aux__Mathlib__SetTheory__Cardinal__Basic______macroRules__Cardinal__term___x5e_x3c____1___closed__1 = (const lean_object*)&lp_mathlib_Cardinal___aux__Mathlib__SetTheory__Cardinal__Basic______macroRules__Cardinal__term___x5e_x3c____1___closed__1_value;
static const lean_string_object lp_mathlib_Cardinal___aux__Mathlib__SetTheory__Cardinal__Basic______macroRules__Cardinal__term___x5e_x3c____1___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "Term"};
static const lean_object* lp_mathlib_Cardinal___aux__Mathlib__SetTheory__Cardinal__Basic______macroRules__Cardinal__term___x5e_x3c____1___closed__2 = (const lean_object*)&lp_mathlib_Cardinal___aux__Mathlib__SetTheory__Cardinal__Basic______macroRules__Cardinal__term___x5e_x3c____1___closed__2_value;
static const lean_string_object lp_mathlib_Cardinal___aux__Mathlib__SetTheory__Cardinal__Basic______macroRules__Cardinal__term___x5e_x3c____1___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = "app"};
static const lean_object* lp_mathlib_Cardinal___aux__Mathlib__SetTheory__Cardinal__Basic______macroRules__Cardinal__term___x5e_x3c____1___closed__3 = (const lean_object*)&lp_mathlib_Cardinal___aux__Mathlib__SetTheory__Cardinal__Basic______macroRules__Cardinal__term___x5e_x3c____1___closed__3_value;
static const lean_ctor_object lp_mathlib_Cardinal___aux__Mathlib__SetTheory__Cardinal__Basic______macroRules__Cardinal__term___x5e_x3c____1___closed__4_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Cardinal___aux__Mathlib__SetTheory__Cardinal__Basic______macroRules__Cardinal__term___x5e_x3c____1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib_Cardinal___aux__Mathlib__SetTheory__Cardinal__Basic______macroRules__Cardinal__term___x5e_x3c____1___closed__4_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Cardinal___aux__Mathlib__SetTheory__Cardinal__Basic______macroRules__Cardinal__term___x5e_x3c____1___closed__4_value_aux_0),((lean_object*)&lp_mathlib_Cardinal___aux__Mathlib__SetTheory__Cardinal__Basic______macroRules__Cardinal__term___x5e_x3c____1___closed__1_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib_Cardinal___aux__Mathlib__SetTheory__Cardinal__Basic______macroRules__Cardinal__term___x5e_x3c____1___closed__4_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Cardinal___aux__Mathlib__SetTheory__Cardinal__Basic______macroRules__Cardinal__term___x5e_x3c____1___closed__4_value_aux_1),((lean_object*)&lp_mathlib_Cardinal___aux__Mathlib__SetTheory__Cardinal__Basic______macroRules__Cardinal__term___x5e_x3c____1___closed__2_value),LEAN_SCALAR_PTR_LITERAL(75, 170, 162, 138, 136, 204, 251, 229)}};
static const lean_ctor_object lp_mathlib_Cardinal___aux__Mathlib__SetTheory__Cardinal__Basic______macroRules__Cardinal__term___x5e_x3c____1___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Cardinal___aux__Mathlib__SetTheory__Cardinal__Basic______macroRules__Cardinal__term___x5e_x3c____1___closed__4_value_aux_2),((lean_object*)&lp_mathlib_Cardinal___aux__Mathlib__SetTheory__Cardinal__Basic______macroRules__Cardinal__term___x5e_x3c____1___closed__3_value),LEAN_SCALAR_PTR_LITERAL(69, 118, 10, 41, 220, 156, 243, 179)}};
static const lean_object* lp_mathlib_Cardinal___aux__Mathlib__SetTheory__Cardinal__Basic______macroRules__Cardinal__term___x5e_x3c____1___closed__4 = (const lean_object*)&lp_mathlib_Cardinal___aux__Mathlib__SetTheory__Cardinal__Basic______macroRules__Cardinal__term___x5e_x3c____1___closed__4_value;
static const lean_string_object lp_mathlib_Cardinal___aux__Mathlib__SetTheory__Cardinal__Basic______macroRules__Cardinal__term___x5e_x3c____1___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "powerlt"};
static const lean_object* lp_mathlib_Cardinal___aux__Mathlib__SetTheory__Cardinal__Basic______macroRules__Cardinal__term___x5e_x3c____1___closed__5 = (const lean_object*)&lp_mathlib_Cardinal___aux__Mathlib__SetTheory__Cardinal__Basic______macroRules__Cardinal__term___x5e_x3c____1___closed__5_value;
static lean_once_cell_t lp_mathlib_Cardinal___aux__Mathlib__SetTheory__Cardinal__Basic______macroRules__Cardinal__term___x5e_x3c____1___closed__6_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Cardinal___aux__Mathlib__SetTheory__Cardinal__Basic______macroRules__Cardinal__term___x5e_x3c____1___closed__6;
static const lean_ctor_object lp_mathlib_Cardinal___aux__Mathlib__SetTheory__Cardinal__Basic______macroRules__Cardinal__term___x5e_x3c____1___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Cardinal___aux__Mathlib__SetTheory__Cardinal__Basic______macroRules__Cardinal__term___x5e_x3c____1___closed__5_value),LEAN_SCALAR_PTR_LITERAL(194, 19, 169, 135, 136, 208, 184, 175)}};
static const lean_object* lp_mathlib_Cardinal___aux__Mathlib__SetTheory__Cardinal__Basic______macroRules__Cardinal__term___x5e_x3c____1___closed__7 = (const lean_object*)&lp_mathlib_Cardinal___aux__Mathlib__SetTheory__Cardinal__Basic______macroRules__Cardinal__term___x5e_x3c____1___closed__7_value;
static const lean_ctor_object lp_mathlib_Cardinal___aux__Mathlib__SetTheory__Cardinal__Basic______macroRules__Cardinal__term___x5e_x3c____1___closed__8_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Cardinal_term___x5e_x3c___00__closed__0_value),LEAN_SCALAR_PTR_LITERAL(247, 176, 10, 15, 146, 217, 130, 31)}};
static const lean_ctor_object lp_mathlib_Cardinal___aux__Mathlib__SetTheory__Cardinal__Basic______macroRules__Cardinal__term___x5e_x3c____1___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Cardinal___aux__Mathlib__SetTheory__Cardinal__Basic______macroRules__Cardinal__term___x5e_x3c____1___closed__8_value_aux_0),((lean_object*)&lp_mathlib_Cardinal___aux__Mathlib__SetTheory__Cardinal__Basic______macroRules__Cardinal__term___x5e_x3c____1___closed__5_value),LEAN_SCALAR_PTR_LITERAL(174, 192, 212, 197, 9, 93, 219, 99)}};
static const lean_object* lp_mathlib_Cardinal___aux__Mathlib__SetTheory__Cardinal__Basic______macroRules__Cardinal__term___x5e_x3c____1___closed__8 = (const lean_object*)&lp_mathlib_Cardinal___aux__Mathlib__SetTheory__Cardinal__Basic______macroRules__Cardinal__term___x5e_x3c____1___closed__8_value;
static const lean_ctor_object lp_mathlib_Cardinal___aux__Mathlib__SetTheory__Cardinal__Basic______macroRules__Cardinal__term___x5e_x3c____1___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Cardinal___aux__Mathlib__SetTheory__Cardinal__Basic______macroRules__Cardinal__term___x5e_x3c____1___closed__8_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_Cardinal___aux__Mathlib__SetTheory__Cardinal__Basic______macroRules__Cardinal__term___x5e_x3c____1___closed__9 = (const lean_object*)&lp_mathlib_Cardinal___aux__Mathlib__SetTheory__Cardinal__Basic______macroRules__Cardinal__term___x5e_x3c____1___closed__9_value;
static const lean_ctor_object lp_mathlib_Cardinal___aux__Mathlib__SetTheory__Cardinal__Basic______macroRules__Cardinal__term___x5e_x3c____1___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Cardinal___aux__Mathlib__SetTheory__Cardinal__Basic______macroRules__Cardinal__term___x5e_x3c____1___closed__9_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_Cardinal___aux__Mathlib__SetTheory__Cardinal__Basic______macroRules__Cardinal__term___x5e_x3c____1___closed__10 = (const lean_object*)&lp_mathlib_Cardinal___aux__Mathlib__SetTheory__Cardinal__Basic______macroRules__Cardinal__term___x5e_x3c____1___closed__10_value;
static const lean_string_object lp_mathlib_Cardinal___aux__Mathlib__SetTheory__Cardinal__Basic______macroRules__Cardinal__term___x5e_x3c____1___closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "null"};
static const lean_object* lp_mathlib_Cardinal___aux__Mathlib__SetTheory__Cardinal__Basic______macroRules__Cardinal__term___x5e_x3c____1___closed__11 = (const lean_object*)&lp_mathlib_Cardinal___aux__Mathlib__SetTheory__Cardinal__Basic______macroRules__Cardinal__term___x5e_x3c____1___closed__11_value;
static const lean_ctor_object lp_mathlib_Cardinal___aux__Mathlib__SetTheory__Cardinal__Basic______macroRules__Cardinal__term___x5e_x3c____1___closed__12_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Cardinal___aux__Mathlib__SetTheory__Cardinal__Basic______macroRules__Cardinal__term___x5e_x3c____1___closed__11_value),LEAN_SCALAR_PTR_LITERAL(24, 58, 49, 223, 146, 207, 197, 136)}};
static const lean_object* lp_mathlib_Cardinal___aux__Mathlib__SetTheory__Cardinal__Basic______macroRules__Cardinal__term___x5e_x3c____1___closed__12 = (const lean_object*)&lp_mathlib_Cardinal___aux__Mathlib__SetTheory__Cardinal__Basic______macroRules__Cardinal__term___x5e_x3c____1___closed__12_value;
LEAN_EXPORT lean_object* lp_mathlib_Cardinal___aux__Mathlib__SetTheory__Cardinal__Basic______macroRules__Cardinal__term___x5e_x3c____1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Cardinal___aux__Mathlib__SetTheory__Cardinal__Basic______macroRules__Cardinal__term___x5e_x3c____1___boxed(lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Cardinal___aux__Mathlib__SetTheory__Cardinal__Basic______unexpand__Cardinal__powerlt__1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "ident"};
static const lean_object* lp_mathlib_Cardinal___aux__Mathlib__SetTheory__Cardinal__Basic______unexpand__Cardinal__powerlt__1___closed__0 = (const lean_object*)&lp_mathlib_Cardinal___aux__Mathlib__SetTheory__Cardinal__Basic______unexpand__Cardinal__powerlt__1___closed__0_value;
static const lean_ctor_object lp_mathlib_Cardinal___aux__Mathlib__SetTheory__Cardinal__Basic______unexpand__Cardinal__powerlt__1___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Cardinal___aux__Mathlib__SetTheory__Cardinal__Basic______unexpand__Cardinal__powerlt__1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(52, 159, 208, 51, 14, 60, 6, 71)}};
static const lean_object* lp_mathlib_Cardinal___aux__Mathlib__SetTheory__Cardinal__Basic______unexpand__Cardinal__powerlt__1___closed__1 = (const lean_object*)&lp_mathlib_Cardinal___aux__Mathlib__SetTheory__Cardinal__Basic______unexpand__Cardinal__powerlt__1___closed__1_value;
LEAN_EXPORT lean_object* lp_mathlib_Cardinal___aux__Mathlib__SetTheory__Cardinal__Basic______unexpand__Cardinal__powerlt__1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Cardinal___aux__Mathlib__SetTheory__Cardinal__Basic______unexpand__Cardinal__powerlt__1___boxed(lean_object*, lean_object*, lean_object*);
static lean_object* _init_lp_mathlib_Cardinal___aux__Mathlib__SetTheory__Cardinal__Basic______macroRules__Cardinal__term___x5e_x3c____1___closed__6(void){
_start:
{
lean_object* v___x_37_; lean_object* v___x_38_; 
v___x_37_ = ((lean_object*)(lp_mathlib_Cardinal___aux__Mathlib__SetTheory__Cardinal__Basic______macroRules__Cardinal__term___x5e_x3c____1___closed__5));
v___x_38_ = l_String_toRawSubstring_x27(v___x_37_);
return v___x_38_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Cardinal___aux__Mathlib__SetTheory__Cardinal__Basic______macroRules__Cardinal__term___x5e_x3c____1(lean_object* v_x_53_, lean_object* v_a_54_, lean_object* v_a_55_){
_start:
{
lean_object* v___x_56_; uint8_t v___x_57_; 
v___x_56_ = ((lean_object*)(lp_mathlib_Cardinal_term___x5e_x3c___00__closed__2));
lean_inc(v_x_53_);
v___x_57_ = l_Lean_Syntax_isOfKind(v_x_53_, v___x_56_);
if (v___x_57_ == 0)
{
lean_object* v___x_58_; lean_object* v___x_59_; 
lean_dec(v_x_53_);
v___x_58_ = lean_box(1);
v___x_59_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_59_, 0, v___x_58_);
lean_ctor_set(v___x_59_, 1, v_a_55_);
return v___x_59_;
}
else
{
lean_object* v_quotContext_60_; lean_object* v_currMacroScope_61_; lean_object* v_ref_62_; lean_object* v___x_63_; lean_object* v___x_64_; lean_object* v___x_65_; lean_object* v___x_66_; uint8_t v___x_67_; lean_object* v___x_68_; lean_object* v___x_69_; lean_object* v___x_70_; lean_object* v___x_71_; lean_object* v___x_72_; lean_object* v___x_73_; lean_object* v___x_74_; lean_object* v___x_75_; lean_object* v___x_76_; lean_object* v___x_77_; lean_object* v___x_78_; 
v_quotContext_60_ = lean_ctor_get(v_a_54_, 1);
v_currMacroScope_61_ = lean_ctor_get(v_a_54_, 2);
v_ref_62_ = lean_ctor_get(v_a_54_, 5);
v___x_63_ = lean_unsigned_to_nat(0u);
v___x_64_ = l_Lean_Syntax_getArg(v_x_53_, v___x_63_);
v___x_65_ = lean_unsigned_to_nat(2u);
v___x_66_ = l_Lean_Syntax_getArg(v_x_53_, v___x_65_);
lean_dec(v_x_53_);
v___x_67_ = 0;
v___x_68_ = l_Lean_SourceInfo_fromRef(v_ref_62_, v___x_67_);
v___x_69_ = ((lean_object*)(lp_mathlib_Cardinal___aux__Mathlib__SetTheory__Cardinal__Basic______macroRules__Cardinal__term___x5e_x3c____1___closed__4));
v___x_70_ = lean_obj_once(&lp_mathlib_Cardinal___aux__Mathlib__SetTheory__Cardinal__Basic______macroRules__Cardinal__term___x5e_x3c____1___closed__6, &lp_mathlib_Cardinal___aux__Mathlib__SetTheory__Cardinal__Basic______macroRules__Cardinal__term___x5e_x3c____1___closed__6_once, _init_lp_mathlib_Cardinal___aux__Mathlib__SetTheory__Cardinal__Basic______macroRules__Cardinal__term___x5e_x3c____1___closed__6);
v___x_71_ = ((lean_object*)(lp_mathlib_Cardinal___aux__Mathlib__SetTheory__Cardinal__Basic______macroRules__Cardinal__term___x5e_x3c____1___closed__7));
lean_inc(v_currMacroScope_61_);
lean_inc(v_quotContext_60_);
v___x_72_ = l_Lean_addMacroScope(v_quotContext_60_, v___x_71_, v_currMacroScope_61_);
v___x_73_ = ((lean_object*)(lp_mathlib_Cardinal___aux__Mathlib__SetTheory__Cardinal__Basic______macroRules__Cardinal__term___x5e_x3c____1___closed__10));
lean_inc_n(v___x_68_, 2);
v___x_74_ = lean_alloc_ctor(3, 4, 0);
lean_ctor_set(v___x_74_, 0, v___x_68_);
lean_ctor_set(v___x_74_, 1, v___x_70_);
lean_ctor_set(v___x_74_, 2, v___x_72_);
lean_ctor_set(v___x_74_, 3, v___x_73_);
v___x_75_ = ((lean_object*)(lp_mathlib_Cardinal___aux__Mathlib__SetTheory__Cardinal__Basic______macroRules__Cardinal__term___x5e_x3c____1___closed__12));
v___x_76_ = l_Lean_Syntax_node2(v___x_68_, v___x_75_, v___x_64_, v___x_66_);
v___x_77_ = l_Lean_Syntax_node2(v___x_68_, v___x_69_, v___x_74_, v___x_76_);
v___x_78_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_78_, 0, v___x_77_);
lean_ctor_set(v___x_78_, 1, v_a_55_);
return v___x_78_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Cardinal___aux__Mathlib__SetTheory__Cardinal__Basic______macroRules__Cardinal__term___x5e_x3c____1___boxed(lean_object* v_x_79_, lean_object* v_a_80_, lean_object* v_a_81_){
_start:
{
lean_object* v_res_82_; 
v_res_82_ = lp_mathlib_Cardinal___aux__Mathlib__SetTheory__Cardinal__Basic______macroRules__Cardinal__term___x5e_x3c____1(v_x_79_, v_a_80_, v_a_81_);
lean_dec_ref(v_a_80_);
return v_res_82_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Cardinal___aux__Mathlib__SetTheory__Cardinal__Basic______unexpand__Cardinal__powerlt__1(lean_object* v_x_86_, lean_object* v_a_87_, lean_object* v_a_88_){
_start:
{
lean_object* v___x_89_; uint8_t v___x_90_; 
v___x_89_ = ((lean_object*)(lp_mathlib_Cardinal___aux__Mathlib__SetTheory__Cardinal__Basic______macroRules__Cardinal__term___x5e_x3c____1___closed__4));
lean_inc(v_x_86_);
v___x_90_ = l_Lean_Syntax_isOfKind(v_x_86_, v___x_89_);
if (v___x_90_ == 0)
{
lean_object* v___x_91_; lean_object* v___x_92_; 
lean_dec(v_x_86_);
v___x_91_ = lean_box(0);
v___x_92_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_92_, 0, v___x_91_);
lean_ctor_set(v___x_92_, 1, v_a_88_);
return v___x_92_;
}
else
{
lean_object* v___x_93_; lean_object* v___x_94_; lean_object* v___x_95_; uint8_t v___x_96_; 
v___x_93_ = lean_unsigned_to_nat(0u);
v___x_94_ = l_Lean_Syntax_getArg(v_x_86_, v___x_93_);
v___x_95_ = ((lean_object*)(lp_mathlib_Cardinal___aux__Mathlib__SetTheory__Cardinal__Basic______unexpand__Cardinal__powerlt__1___closed__1));
lean_inc(v___x_94_);
v___x_96_ = l_Lean_Syntax_isOfKind(v___x_94_, v___x_95_);
if (v___x_96_ == 0)
{
lean_object* v___x_97_; lean_object* v___x_98_; 
lean_dec(v___x_94_);
lean_dec(v_x_86_);
v___x_97_ = lean_box(0);
v___x_98_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_98_, 0, v___x_97_);
lean_ctor_set(v___x_98_, 1, v_a_88_);
return v___x_98_;
}
else
{
lean_object* v___x_99_; lean_object* v___x_100_; lean_object* v___x_101_; uint8_t v___x_102_; 
v___x_99_ = lean_unsigned_to_nat(1u);
v___x_100_ = l_Lean_Syntax_getArg(v_x_86_, v___x_99_);
lean_dec(v_x_86_);
v___x_101_ = lean_unsigned_to_nat(2u);
lean_inc(v___x_100_);
v___x_102_ = l_Lean_Syntax_matchesNull(v___x_100_, v___x_101_);
if (v___x_102_ == 0)
{
lean_object* v___x_103_; lean_object* v___x_104_; 
lean_dec(v___x_100_);
lean_dec(v___x_94_);
v___x_103_ = lean_box(0);
v___x_104_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_104_, 0, v___x_103_);
lean_ctor_set(v___x_104_, 1, v_a_88_);
return v___x_104_;
}
else
{
lean_object* v___x_105_; lean_object* v___x_106_; lean_object* v_ref_107_; uint8_t v___x_108_; lean_object* v___x_109_; lean_object* v___x_110_; lean_object* v___x_111_; lean_object* v___x_112_; lean_object* v___x_113_; lean_object* v___x_114_; 
v___x_105_ = l_Lean_Syntax_getArg(v___x_100_, v___x_93_);
v___x_106_ = l_Lean_Syntax_getArg(v___x_100_, v___x_99_);
lean_dec(v___x_100_);
v_ref_107_ = l_Lean_replaceRef(v___x_94_, v_a_87_);
lean_dec(v___x_94_);
v___x_108_ = 0;
v___x_109_ = l_Lean_SourceInfo_fromRef(v_ref_107_, v___x_108_);
lean_dec(v_ref_107_);
v___x_110_ = ((lean_object*)(lp_mathlib_Cardinal_term___x5e_x3c___00__closed__2));
v___x_111_ = ((lean_object*)(lp_mathlib_Cardinal_term___x5e_x3c___00__closed__5));
lean_inc(v___x_109_);
v___x_112_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_112_, 0, v___x_109_);
lean_ctor_set(v___x_112_, 1, v___x_111_);
v___x_113_ = l_Lean_Syntax_node3(v___x_109_, v___x_110_, v___x_105_, v___x_112_, v___x_106_);
v___x_114_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_114_, 0, v___x_113_);
lean_ctor_set(v___x_114_, 1, v_a_88_);
return v___x_114_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Cardinal___aux__Mathlib__SetTheory__Cardinal__Basic______unexpand__Cardinal__powerlt__1___boxed(lean_object* v_x_115_, lean_object* v_a_116_, lean_object* v_a_117_){
_start:
{
lean_object* v_res_118_; 
v_res_118_ = lp_mathlib_Cardinal___aux__Mathlib__SetTheory__Cardinal__Basic______unexpand__Cardinal__powerlt__1(v_x_115_, v_a_116_, v_a_117_);
lean_dec(v_a_116_);
return v_res_118_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Basic_Countable_Small(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Basic_UnivLE(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Data_Fintype_BigOperators(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Data_Fintype_Powerset(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Data_Nat_Cast_Order_Basic(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Data_Set_Countable(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Logic_Small_Set(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_SetTheory_Cardinal_Order(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_SetTheory_Cardinal_Basic(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Basic_Countable_Small(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Basic_UnivLE(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Data_Fintype_BigOperators(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Data_Fintype_Powerset(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Data_Nat_Cast_Order_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Data_Set_Countable(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Logic_Small_Set(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_SetTheory_Cardinal_Order(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_SetTheory_Cardinal_Basic(uint8_t builtin) {
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
lean_object* initialize_mathlib_Mathlib_Basic_Countable_Small(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Basic_UnivLE(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Data_Fintype_BigOperators(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Data_Fintype_Powerset(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Data_Nat_Cast_Order_Basic(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Data_Set_Countable(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Logic_Small_Set(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_SetTheory_Cardinal_Order(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_SetTheory_Cardinal_Basic(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Basic_Countable_Small(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Basic_UnivLE(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_Fintype_BigOperators(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_Fintype_Powerset(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_Nat_Cast_Order_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_Set_Countable(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Logic_Small_Set(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_SetTheory_Cardinal_Order(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_SetTheory_Cardinal_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_SetTheory_Cardinal_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_SetTheory_Cardinal_Basic(builtin);
}
#ifdef __cplusplus
}
#endif
