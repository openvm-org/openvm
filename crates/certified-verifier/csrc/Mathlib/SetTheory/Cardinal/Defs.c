// Lean compiler output
// Module: Mathlib.SetTheory.Cardinal.Defs
// Imports: public import Init public meta import Init public import Mathlib.Basic.IsEmpty.Basic public import Mathlib.Data.ULift public import Mathlib.Tactic.CrossRefAttribute public import Mathlib.Tactic.PPWithUniv public import Mathlib.Util.Delaborators
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
lean_object* l_Lean_Name_mkStr2(lean_object*, lean_object*);
lean_object* l_Lean_Name_mkStr4(lean_object*, lean_object*, lean_object*, lean_object*);
uint8_t l_Lean_Syntax_isOfKind(lean_object*, lean_object*);
lean_object* l_Lean_Syntax_getArg(lean_object*, lean_object*);
lean_object* l_Lean_Name_mkStr1(lean_object*);
uint8_t l_Lean_Syntax_matchesNull(lean_object*, lean_object*);
lean_object* l_Lean_replaceRef(lean_object*, lean_object*);
lean_object* l_Lean_SourceInfo_fromRef(lean_object*, uint8_t);
lean_object* l_Lean_Syntax_node2(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Syntax_node1(lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Equiv_sumCongr___redArg(lean_object*, lean_object*);
lean_object* l_String_toRawSubstring_x27(lean_object*);
lean_object* l_Lean_addMacroScope(lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Equiv_arrowCongr___redArg(lean_object*, lean_object*);
lean_object* lp_mathlib_Equiv_prodCongr___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Cardinal_isEquivalent;
LEAN_EXPORT lean_object* lp_mathlib_Cardinal_mk(lean_object*);
static const lean_string_object lp_mathlib_Cardinal_term_x23___00__closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "Cardinal"};
static const lean_object* lp_mathlib_Cardinal_term_x23___00__closed__0 = (const lean_object*)&lp_mathlib_Cardinal_term_x23___00__closed__0_value;
static const lean_string_object lp_mathlib_Cardinal_term_x23___00__closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "term#_"};
static const lean_object* lp_mathlib_Cardinal_term_x23___00__closed__1 = (const lean_object*)&lp_mathlib_Cardinal_term_x23___00__closed__1_value;
static const lean_ctor_object lp_mathlib_Cardinal_term_x23___00__closed__2_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Cardinal_term_x23___00__closed__0_value),LEAN_SCALAR_PTR_LITERAL(247, 176, 10, 15, 146, 217, 130, 31)}};
static const lean_ctor_object lp_mathlib_Cardinal_term_x23___00__closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Cardinal_term_x23___00__closed__2_value_aux_0),((lean_object*)&lp_mathlib_Cardinal_term_x23___00__closed__1_value),LEAN_SCALAR_PTR_LITERAL(181, 46, 127, 253, 19, 4, 128, 143)}};
static const lean_object* lp_mathlib_Cardinal_term_x23___00__closed__2 = (const lean_object*)&lp_mathlib_Cardinal_term_x23___00__closed__2_value;
static const lean_string_object lp_mathlib_Cardinal_term_x23___00__closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "andthen"};
static const lean_object* lp_mathlib_Cardinal_term_x23___00__closed__3 = (const lean_object*)&lp_mathlib_Cardinal_term_x23___00__closed__3_value;
static const lean_ctor_object lp_mathlib_Cardinal_term_x23___00__closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Cardinal_term_x23___00__closed__3_value),LEAN_SCALAR_PTR_LITERAL(40, 255, 78, 30, 143, 119, 117, 174)}};
static const lean_object* lp_mathlib_Cardinal_term_x23___00__closed__4 = (const lean_object*)&lp_mathlib_Cardinal_term_x23___00__closed__4_value;
static const lean_string_object lp_mathlib_Cardinal_term_x23___00__closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = "#"};
static const lean_object* lp_mathlib_Cardinal_term_x23___00__closed__5 = (const lean_object*)&lp_mathlib_Cardinal_term_x23___00__closed__5_value;
static const lean_ctor_object lp_mathlib_Cardinal_term_x23___00__closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 5}, .m_objs = {((lean_object*)&lp_mathlib_Cardinal_term_x23___00__closed__5_value)}};
static const lean_object* lp_mathlib_Cardinal_term_x23___00__closed__6 = (const lean_object*)&lp_mathlib_Cardinal_term_x23___00__closed__6_value;
static const lean_string_object lp_mathlib_Cardinal_term_x23___00__closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "term"};
static const lean_object* lp_mathlib_Cardinal_term_x23___00__closed__7 = (const lean_object*)&lp_mathlib_Cardinal_term_x23___00__closed__7_value;
static const lean_ctor_object lp_mathlib_Cardinal_term_x23___00__closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Cardinal_term_x23___00__closed__7_value),LEAN_SCALAR_PTR_LITERAL(187, 230, 181, 162, 253, 146, 122, 119)}};
static const lean_object* lp_mathlib_Cardinal_term_x23___00__closed__8 = (const lean_object*)&lp_mathlib_Cardinal_term_x23___00__closed__8_value;
static const lean_ctor_object lp_mathlib_Cardinal_term_x23___00__closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 7}, .m_objs = {((lean_object*)&lp_mathlib_Cardinal_term_x23___00__closed__8_value),((lean_object*)(((size_t)(1024) << 1) | 1))}};
static const lean_object* lp_mathlib_Cardinal_term_x23___00__closed__9 = (const lean_object*)&lp_mathlib_Cardinal_term_x23___00__closed__9_value;
static const lean_ctor_object lp_mathlib_Cardinal_term_x23___00__closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib_Cardinal_term_x23___00__closed__4_value),((lean_object*)&lp_mathlib_Cardinal_term_x23___00__closed__6_value),((lean_object*)&lp_mathlib_Cardinal_term_x23___00__closed__9_value)}};
static const lean_object* lp_mathlib_Cardinal_term_x23___00__closed__10 = (const lean_object*)&lp_mathlib_Cardinal_term_x23___00__closed__10_value;
static const lean_ctor_object lp_mathlib_Cardinal_term_x23___00__closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 3}, .m_objs = {((lean_object*)&lp_mathlib_Cardinal_term_x23___00__closed__2_value),((lean_object*)(((size_t)(1024) << 1) | 1)),((lean_object*)&lp_mathlib_Cardinal_term_x23___00__closed__10_value)}};
static const lean_object* lp_mathlib_Cardinal_term_x23___00__closed__11 = (const lean_object*)&lp_mathlib_Cardinal_term_x23___00__closed__11_value;
LEAN_EXPORT const lean_object* lp_mathlib_Cardinal_term_x23__ = (const lean_object*)&lp_mathlib_Cardinal_term_x23___00__closed__11_value;
static const lean_string_object lp_mathlib_Cardinal___aux__Mathlib__SetTheory__Cardinal__Defs______macroRules__Cardinal__term_x23____1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "Lean"};
static const lean_object* lp_mathlib_Cardinal___aux__Mathlib__SetTheory__Cardinal__Defs______macroRules__Cardinal__term_x23____1___closed__0 = (const lean_object*)&lp_mathlib_Cardinal___aux__Mathlib__SetTheory__Cardinal__Defs______macroRules__Cardinal__term_x23____1___closed__0_value;
static const lean_string_object lp_mathlib_Cardinal___aux__Mathlib__SetTheory__Cardinal__Defs______macroRules__Cardinal__term_x23____1___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "Parser"};
static const lean_object* lp_mathlib_Cardinal___aux__Mathlib__SetTheory__Cardinal__Defs______macroRules__Cardinal__term_x23____1___closed__1 = (const lean_object*)&lp_mathlib_Cardinal___aux__Mathlib__SetTheory__Cardinal__Defs______macroRules__Cardinal__term_x23____1___closed__1_value;
static const lean_string_object lp_mathlib_Cardinal___aux__Mathlib__SetTheory__Cardinal__Defs______macroRules__Cardinal__term_x23____1___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "Term"};
static const lean_object* lp_mathlib_Cardinal___aux__Mathlib__SetTheory__Cardinal__Defs______macroRules__Cardinal__term_x23____1___closed__2 = (const lean_object*)&lp_mathlib_Cardinal___aux__Mathlib__SetTheory__Cardinal__Defs______macroRules__Cardinal__term_x23____1___closed__2_value;
static const lean_string_object lp_mathlib_Cardinal___aux__Mathlib__SetTheory__Cardinal__Defs______macroRules__Cardinal__term_x23____1___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = "app"};
static const lean_object* lp_mathlib_Cardinal___aux__Mathlib__SetTheory__Cardinal__Defs______macroRules__Cardinal__term_x23____1___closed__3 = (const lean_object*)&lp_mathlib_Cardinal___aux__Mathlib__SetTheory__Cardinal__Defs______macroRules__Cardinal__term_x23____1___closed__3_value;
static const lean_ctor_object lp_mathlib_Cardinal___aux__Mathlib__SetTheory__Cardinal__Defs______macroRules__Cardinal__term_x23____1___closed__4_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Cardinal___aux__Mathlib__SetTheory__Cardinal__Defs______macroRules__Cardinal__term_x23____1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib_Cardinal___aux__Mathlib__SetTheory__Cardinal__Defs______macroRules__Cardinal__term_x23____1___closed__4_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Cardinal___aux__Mathlib__SetTheory__Cardinal__Defs______macroRules__Cardinal__term_x23____1___closed__4_value_aux_0),((lean_object*)&lp_mathlib_Cardinal___aux__Mathlib__SetTheory__Cardinal__Defs______macroRules__Cardinal__term_x23____1___closed__1_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib_Cardinal___aux__Mathlib__SetTheory__Cardinal__Defs______macroRules__Cardinal__term_x23____1___closed__4_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Cardinal___aux__Mathlib__SetTheory__Cardinal__Defs______macroRules__Cardinal__term_x23____1___closed__4_value_aux_1),((lean_object*)&lp_mathlib_Cardinal___aux__Mathlib__SetTheory__Cardinal__Defs______macroRules__Cardinal__term_x23____1___closed__2_value),LEAN_SCALAR_PTR_LITERAL(75, 170, 162, 138, 136, 204, 251, 229)}};
static const lean_ctor_object lp_mathlib_Cardinal___aux__Mathlib__SetTheory__Cardinal__Defs______macroRules__Cardinal__term_x23____1___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Cardinal___aux__Mathlib__SetTheory__Cardinal__Defs______macroRules__Cardinal__term_x23____1___closed__4_value_aux_2),((lean_object*)&lp_mathlib_Cardinal___aux__Mathlib__SetTheory__Cardinal__Defs______macroRules__Cardinal__term_x23____1___closed__3_value),LEAN_SCALAR_PTR_LITERAL(69, 118, 10, 41, 220, 156, 243, 179)}};
static const lean_object* lp_mathlib_Cardinal___aux__Mathlib__SetTheory__Cardinal__Defs______macroRules__Cardinal__term_x23____1___closed__4 = (const lean_object*)&lp_mathlib_Cardinal___aux__Mathlib__SetTheory__Cardinal__Defs______macroRules__Cardinal__term_x23____1___closed__4_value;
static const lean_string_object lp_mathlib_Cardinal___aux__Mathlib__SetTheory__Cardinal__Defs______macroRules__Cardinal__term_x23____1___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 12, .m_capacity = 12, .m_length = 11, .m_data = "Cardinal.mk"};
static const lean_object* lp_mathlib_Cardinal___aux__Mathlib__SetTheory__Cardinal__Defs______macroRules__Cardinal__term_x23____1___closed__5 = (const lean_object*)&lp_mathlib_Cardinal___aux__Mathlib__SetTheory__Cardinal__Defs______macroRules__Cardinal__term_x23____1___closed__5_value;
static lean_once_cell_t lp_mathlib_Cardinal___aux__Mathlib__SetTheory__Cardinal__Defs______macroRules__Cardinal__term_x23____1___closed__6_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Cardinal___aux__Mathlib__SetTheory__Cardinal__Defs______macroRules__Cardinal__term_x23____1___closed__6;
static const lean_string_object lp_mathlib_Cardinal___aux__Mathlib__SetTheory__Cardinal__Defs______macroRules__Cardinal__term_x23____1___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 3, .m_capacity = 3, .m_length = 2, .m_data = "mk"};
static const lean_object* lp_mathlib_Cardinal___aux__Mathlib__SetTheory__Cardinal__Defs______macroRules__Cardinal__term_x23____1___closed__7 = (const lean_object*)&lp_mathlib_Cardinal___aux__Mathlib__SetTheory__Cardinal__Defs______macroRules__Cardinal__term_x23____1___closed__7_value;
static const lean_ctor_object lp_mathlib_Cardinal___aux__Mathlib__SetTheory__Cardinal__Defs______macroRules__Cardinal__term_x23____1___closed__8_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Cardinal_term_x23___00__closed__0_value),LEAN_SCALAR_PTR_LITERAL(247, 176, 10, 15, 146, 217, 130, 31)}};
static const lean_ctor_object lp_mathlib_Cardinal___aux__Mathlib__SetTheory__Cardinal__Defs______macroRules__Cardinal__term_x23____1___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Cardinal___aux__Mathlib__SetTheory__Cardinal__Defs______macroRules__Cardinal__term_x23____1___closed__8_value_aux_0),((lean_object*)&lp_mathlib_Cardinal___aux__Mathlib__SetTheory__Cardinal__Defs______macroRules__Cardinal__term_x23____1___closed__7_value),LEAN_SCALAR_PTR_LITERAL(27, 221, 33, 162, 29, 30, 168, 68)}};
static const lean_object* lp_mathlib_Cardinal___aux__Mathlib__SetTheory__Cardinal__Defs______macroRules__Cardinal__term_x23____1___closed__8 = (const lean_object*)&lp_mathlib_Cardinal___aux__Mathlib__SetTheory__Cardinal__Defs______macroRules__Cardinal__term_x23____1___closed__8_value;
static const lean_ctor_object lp_mathlib_Cardinal___aux__Mathlib__SetTheory__Cardinal__Defs______macroRules__Cardinal__term_x23____1___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Cardinal___aux__Mathlib__SetTheory__Cardinal__Defs______macroRules__Cardinal__term_x23____1___closed__8_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_Cardinal___aux__Mathlib__SetTheory__Cardinal__Defs______macroRules__Cardinal__term_x23____1___closed__9 = (const lean_object*)&lp_mathlib_Cardinal___aux__Mathlib__SetTheory__Cardinal__Defs______macroRules__Cardinal__term_x23____1___closed__9_value;
static const lean_ctor_object lp_mathlib_Cardinal___aux__Mathlib__SetTheory__Cardinal__Defs______macroRules__Cardinal__term_x23____1___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Cardinal___aux__Mathlib__SetTheory__Cardinal__Defs______macroRules__Cardinal__term_x23____1___closed__9_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_Cardinal___aux__Mathlib__SetTheory__Cardinal__Defs______macroRules__Cardinal__term_x23____1___closed__10 = (const lean_object*)&lp_mathlib_Cardinal___aux__Mathlib__SetTheory__Cardinal__Defs______macroRules__Cardinal__term_x23____1___closed__10_value;
static const lean_string_object lp_mathlib_Cardinal___aux__Mathlib__SetTheory__Cardinal__Defs______macroRules__Cardinal__term_x23____1___closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "null"};
static const lean_object* lp_mathlib_Cardinal___aux__Mathlib__SetTheory__Cardinal__Defs______macroRules__Cardinal__term_x23____1___closed__11 = (const lean_object*)&lp_mathlib_Cardinal___aux__Mathlib__SetTheory__Cardinal__Defs______macroRules__Cardinal__term_x23____1___closed__11_value;
static const lean_ctor_object lp_mathlib_Cardinal___aux__Mathlib__SetTheory__Cardinal__Defs______macroRules__Cardinal__term_x23____1___closed__12_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Cardinal___aux__Mathlib__SetTheory__Cardinal__Defs______macroRules__Cardinal__term_x23____1___closed__11_value),LEAN_SCALAR_PTR_LITERAL(24, 58, 49, 223, 146, 207, 197, 136)}};
static const lean_object* lp_mathlib_Cardinal___aux__Mathlib__SetTheory__Cardinal__Defs______macroRules__Cardinal__term_x23____1___closed__12 = (const lean_object*)&lp_mathlib_Cardinal___aux__Mathlib__SetTheory__Cardinal__Defs______macroRules__Cardinal__term_x23____1___closed__12_value;
LEAN_EXPORT lean_object* lp_mathlib_Cardinal___aux__Mathlib__SetTheory__Cardinal__Defs______macroRules__Cardinal__term_x23____1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Cardinal___aux__Mathlib__SetTheory__Cardinal__Defs______macroRules__Cardinal__term_x23____1___boxed(lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Cardinal___aux__Mathlib__SetTheory__Cardinal__Defs______unexpand__Cardinal__mk__1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "ident"};
static const lean_object* lp_mathlib_Cardinal___aux__Mathlib__SetTheory__Cardinal__Defs______unexpand__Cardinal__mk__1___closed__0 = (const lean_object*)&lp_mathlib_Cardinal___aux__Mathlib__SetTheory__Cardinal__Defs______unexpand__Cardinal__mk__1___closed__0_value;
static const lean_ctor_object lp_mathlib_Cardinal___aux__Mathlib__SetTheory__Cardinal__Defs______unexpand__Cardinal__mk__1___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Cardinal___aux__Mathlib__SetTheory__Cardinal__Defs______unexpand__Cardinal__mk__1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(52, 159, 208, 51, 14, 60, 6, 71)}};
static const lean_object* lp_mathlib_Cardinal___aux__Mathlib__SetTheory__Cardinal__Defs______unexpand__Cardinal__mk__1___closed__1 = (const lean_object*)&lp_mathlib_Cardinal___aux__Mathlib__SetTheory__Cardinal__Defs______unexpand__Cardinal__mk__1___closed__1_value;
LEAN_EXPORT lean_object* lp_mathlib_Cardinal___aux__Mathlib__SetTheory__Cardinal__Defs______unexpand__Cardinal__mk__1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Cardinal___aux__Mathlib__SetTheory__Cardinal__Defs______unexpand__Cardinal__mk__1___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Cardinal_map(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Cardinal_map___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Quotient_map_u2082___at___00Cardinal_map_u2082_spec__0(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Quotient_map_u2082___at___00Cardinal_map_u2082_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Cardinal_map_u2082___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Cardinal_map_u2082___redArg___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Cardinal_map_u2082(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Cardinal_map_u2082___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Cardinal_lift(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Cardinal_lift___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Cardinal_instZero;
LEAN_EXPORT lean_object* lp_mathlib_Cardinal_instInhabited;
LEAN_EXPORT lean_object* lp_mathlib_Cardinal_instOne;
LEAN_EXPORT lean_object* lp_mathlib_Cardinal_instAdd___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_Cardinal_instAdd___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Cardinal_instAdd___lam__0, .m_arity = 6, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Cardinal_instAdd___closed__0 = (const lean_object*)&lp_mathlib_Cardinal_instAdd___closed__0_value;
static const lean_closure_object lp_mathlib_Cardinal_instAdd___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*2, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Cardinal_map_u2082___boxed, .m_arity = 4, .m_num_fixed = 2, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Cardinal_instAdd___closed__0_value)} };
static const lean_object* lp_mathlib_Cardinal_instAdd___closed__1 = (const lean_object*)&lp_mathlib_Cardinal_instAdd___closed__1_value;
LEAN_EXPORT const lean_object* lp_mathlib_Cardinal_instAdd = (const lean_object*)&lp_mathlib_Cardinal_instAdd___closed__1_value;
LEAN_EXPORT lean_object* lp_mathlib_Cardinal_instNatCast___lam__0(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Cardinal_instNatCast___lam__0___boxed(lean_object*);
static const lean_closure_object lp_mathlib_Cardinal_instNatCast___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Cardinal_instNatCast___lam__0___boxed, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Cardinal_instNatCast___closed__0 = (const lean_object*)&lp_mathlib_Cardinal_instNatCast___closed__0_value;
LEAN_EXPORT const lean_object* lp_mathlib_Cardinal_instNatCast = (const lean_object*)&lp_mathlib_Cardinal_instNatCast___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_Cardinal_instMul___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_Cardinal_instMul___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Cardinal_instMul___lam__0, .m_arity = 6, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Cardinal_instMul___closed__0 = (const lean_object*)&lp_mathlib_Cardinal_instMul___closed__0_value;
static const lean_closure_object lp_mathlib_Cardinal_instMul___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*2, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Cardinal_map_u2082___boxed, .m_arity = 4, .m_num_fixed = 2, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Cardinal_instMul___closed__0_value)} };
static const lean_object* lp_mathlib_Cardinal_instMul___closed__1 = (const lean_object*)&lp_mathlib_Cardinal_instMul___closed__1_value;
LEAN_EXPORT const lean_object* lp_mathlib_Cardinal_instMul = (const lean_object*)&lp_mathlib_Cardinal_instMul___closed__1_value;
LEAN_EXPORT lean_object* lp_mathlib_Cardinal_instPowCardinal___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_Cardinal_instPowCardinal___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Cardinal_instPowCardinal___lam__0, .m_arity = 6, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Cardinal_instPowCardinal___closed__0 = (const lean_object*)&lp_mathlib_Cardinal_instPowCardinal___closed__0_value;
static const lean_closure_object lp_mathlib_Cardinal_instPowCardinal___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*2, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Cardinal_map_u2082___boxed, .m_arity = 4, .m_num_fixed = 2, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Cardinal_instPowCardinal___closed__0_value)} };
static const lean_object* lp_mathlib_Cardinal_instPowCardinal___closed__1 = (const lean_object*)&lp_mathlib_Cardinal_instPowCardinal___closed__1_value;
LEAN_EXPORT const lean_object* lp_mathlib_Cardinal_instPowCardinal = (const lean_object*)&lp_mathlib_Cardinal_instPowCardinal___closed__1_value;
LEAN_EXPORT lean_object* lp_mathlib_Cardinal_sum(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Cardinal_sum___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Cardinal_prod(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Cardinal_prod___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Cardinal_aleph0;
static const lean_string_object lp_mathlib_Cardinal_term_u2135_u2080___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 11, .m_capacity = 11, .m_length = 6, .m_data = "termℵ₀"};
static const lean_object* lp_mathlib_Cardinal_term_u2135_u2080___closed__0 = (const lean_object*)&lp_mathlib_Cardinal_term_u2135_u2080___closed__0_value;
static const lean_ctor_object lp_mathlib_Cardinal_term_u2135_u2080___closed__1_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Cardinal_term_x23___00__closed__0_value),LEAN_SCALAR_PTR_LITERAL(247, 176, 10, 15, 146, 217, 130, 31)}};
static const lean_ctor_object lp_mathlib_Cardinal_term_u2135_u2080___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Cardinal_term_u2135_u2080___closed__1_value_aux_0),((lean_object*)&lp_mathlib_Cardinal_term_u2135_u2080___closed__0_value),LEAN_SCALAR_PTR_LITERAL(47, 181, 254, 208, 110, 112, 23, 34)}};
static const lean_object* lp_mathlib_Cardinal_term_u2135_u2080___closed__1 = (const lean_object*)&lp_mathlib_Cardinal_term_u2135_u2080___closed__1_value;
static const lean_string_object lp_mathlib_Cardinal_term_u2135_u2080___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 2, .m_data = "ℵ₀"};
static const lean_object* lp_mathlib_Cardinal_term_u2135_u2080___closed__2 = (const lean_object*)&lp_mathlib_Cardinal_term_u2135_u2080___closed__2_value;
static const lean_ctor_object lp_mathlib_Cardinal_term_u2135_u2080___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 5}, .m_objs = {((lean_object*)&lp_mathlib_Cardinal_term_u2135_u2080___closed__2_value)}};
static const lean_object* lp_mathlib_Cardinal_term_u2135_u2080___closed__3 = (const lean_object*)&lp_mathlib_Cardinal_term_u2135_u2080___closed__3_value;
static const lean_ctor_object lp_mathlib_Cardinal_term_u2135_u2080___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 3}, .m_objs = {((lean_object*)&lp_mathlib_Cardinal_term_u2135_u2080___closed__1_value),((lean_object*)(((size_t)(1024) << 1) | 1)),((lean_object*)&lp_mathlib_Cardinal_term_u2135_u2080___closed__3_value)}};
static const lean_object* lp_mathlib_Cardinal_term_u2135_u2080___closed__4 = (const lean_object*)&lp_mathlib_Cardinal_term_u2135_u2080___closed__4_value;
LEAN_EXPORT const lean_object* lp_mathlib_Cardinal_term_u2135_u2080 = (const lean_object*)&lp_mathlib_Cardinal_term_u2135_u2080___closed__4_value;
static const lean_string_object lp_mathlib_Cardinal___aux__Mathlib__SetTheory__Cardinal__Defs______macroRules__Cardinal__term_u2135_u2080__1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 16, .m_capacity = 16, .m_length = 15, .m_data = "Cardinal.aleph0"};
static const lean_object* lp_mathlib_Cardinal___aux__Mathlib__SetTheory__Cardinal__Defs______macroRules__Cardinal__term_u2135_u2080__1___closed__0 = (const lean_object*)&lp_mathlib_Cardinal___aux__Mathlib__SetTheory__Cardinal__Defs______macroRules__Cardinal__term_u2135_u2080__1___closed__0_value;
static lean_once_cell_t lp_mathlib_Cardinal___aux__Mathlib__SetTheory__Cardinal__Defs______macroRules__Cardinal__term_u2135_u2080__1___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Cardinal___aux__Mathlib__SetTheory__Cardinal__Defs______macroRules__Cardinal__term_u2135_u2080__1___closed__1;
static const lean_string_object lp_mathlib_Cardinal___aux__Mathlib__SetTheory__Cardinal__Defs______macroRules__Cardinal__term_u2135_u2080__1___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "aleph0"};
static const lean_object* lp_mathlib_Cardinal___aux__Mathlib__SetTheory__Cardinal__Defs______macroRules__Cardinal__term_u2135_u2080__1___closed__2 = (const lean_object*)&lp_mathlib_Cardinal___aux__Mathlib__SetTheory__Cardinal__Defs______macroRules__Cardinal__term_u2135_u2080__1___closed__2_value;
static const lean_ctor_object lp_mathlib_Cardinal___aux__Mathlib__SetTheory__Cardinal__Defs______macroRules__Cardinal__term_u2135_u2080__1___closed__3_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Cardinal_term_x23___00__closed__0_value),LEAN_SCALAR_PTR_LITERAL(247, 176, 10, 15, 146, 217, 130, 31)}};
static const lean_ctor_object lp_mathlib_Cardinal___aux__Mathlib__SetTheory__Cardinal__Defs______macroRules__Cardinal__term_u2135_u2080__1___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Cardinal___aux__Mathlib__SetTheory__Cardinal__Defs______macroRules__Cardinal__term_u2135_u2080__1___closed__3_value_aux_0),((lean_object*)&lp_mathlib_Cardinal___aux__Mathlib__SetTheory__Cardinal__Defs______macroRules__Cardinal__term_u2135_u2080__1___closed__2_value),LEAN_SCALAR_PTR_LITERAL(83, 248, 157, 206, 147, 33, 25, 101)}};
static const lean_object* lp_mathlib_Cardinal___aux__Mathlib__SetTheory__Cardinal__Defs______macroRules__Cardinal__term_u2135_u2080__1___closed__3 = (const lean_object*)&lp_mathlib_Cardinal___aux__Mathlib__SetTheory__Cardinal__Defs______macroRules__Cardinal__term_u2135_u2080__1___closed__3_value;
static const lean_ctor_object lp_mathlib_Cardinal___aux__Mathlib__SetTheory__Cardinal__Defs______macroRules__Cardinal__term_u2135_u2080__1___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Cardinal___aux__Mathlib__SetTheory__Cardinal__Defs______macroRules__Cardinal__term_u2135_u2080__1___closed__3_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_Cardinal___aux__Mathlib__SetTheory__Cardinal__Defs______macroRules__Cardinal__term_u2135_u2080__1___closed__4 = (const lean_object*)&lp_mathlib_Cardinal___aux__Mathlib__SetTheory__Cardinal__Defs______macroRules__Cardinal__term_u2135_u2080__1___closed__4_value;
static const lean_ctor_object lp_mathlib_Cardinal___aux__Mathlib__SetTheory__Cardinal__Defs______macroRules__Cardinal__term_u2135_u2080__1___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Cardinal___aux__Mathlib__SetTheory__Cardinal__Defs______macroRules__Cardinal__term_u2135_u2080__1___closed__4_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_Cardinal___aux__Mathlib__SetTheory__Cardinal__Defs______macroRules__Cardinal__term_u2135_u2080__1___closed__5 = (const lean_object*)&lp_mathlib_Cardinal___aux__Mathlib__SetTheory__Cardinal__Defs______macroRules__Cardinal__term_u2135_u2080__1___closed__5_value;
LEAN_EXPORT lean_object* lp_mathlib_Cardinal___aux__Mathlib__SetTheory__Cardinal__Defs______macroRules__Cardinal__term_u2135_u2080__1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Cardinal___aux__Mathlib__SetTheory__Cardinal__Defs______macroRules__Cardinal__term_u2135_u2080__1___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Cardinal___aux__Mathlib__SetTheory__Cardinal__Defs______unexpand__Cardinal__aleph0__1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Cardinal___aux__Mathlib__SetTheory__Cardinal__Defs______unexpand__Cardinal__aleph0__1___boxed(lean_object*, lean_object*, lean_object*);
static lean_object* _init_lp_mathlib_Cardinal_isEquivalent(void){
_start:
{
lean_object* v___x_1_; 
v___x_1_ = lean_box(0);
return v___x_1_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Cardinal_mk(lean_object* v_a_2_){
_start:
{
lean_object* v___x_3_; 
v___x_3_ = lean_box(0);
return v___x_3_;
}
}
static lean_object* _init_lp_mathlib_Cardinal___aux__Mathlib__SetTheory__Cardinal__Defs______macroRules__Cardinal__term_x23____1___closed__6(void){
_start:
{
lean_object* v___x_40_; lean_object* v___x_41_; 
v___x_40_ = ((lean_object*)(lp_mathlib_Cardinal___aux__Mathlib__SetTheory__Cardinal__Defs______macroRules__Cardinal__term_x23____1___closed__5));
v___x_41_ = l_String_toRawSubstring_x27(v___x_40_);
return v___x_41_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Cardinal___aux__Mathlib__SetTheory__Cardinal__Defs______macroRules__Cardinal__term_x23____1(lean_object* v_x_55_, lean_object* v_a_56_, lean_object* v_a_57_){
_start:
{
lean_object* v___x_58_; uint8_t v___x_59_; 
v___x_58_ = ((lean_object*)(lp_mathlib_Cardinal_term_x23___00__closed__2));
lean_inc(v_x_55_);
v___x_59_ = l_Lean_Syntax_isOfKind(v_x_55_, v___x_58_);
if (v___x_59_ == 0)
{
lean_object* v___x_60_; lean_object* v___x_61_; 
lean_dec(v_x_55_);
v___x_60_ = lean_box(1);
v___x_61_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_61_, 0, v___x_60_);
lean_ctor_set(v___x_61_, 1, v_a_57_);
return v___x_61_;
}
else
{
lean_object* v_quotContext_62_; lean_object* v_currMacroScope_63_; lean_object* v_ref_64_; lean_object* v___x_65_; lean_object* v___x_66_; uint8_t v___x_67_; lean_object* v___x_68_; lean_object* v___x_69_; lean_object* v___x_70_; lean_object* v___x_71_; lean_object* v___x_72_; lean_object* v___x_73_; lean_object* v___x_74_; lean_object* v___x_75_; lean_object* v___x_76_; lean_object* v___x_77_; lean_object* v___x_78_; 
v_quotContext_62_ = lean_ctor_get(v_a_56_, 1);
v_currMacroScope_63_ = lean_ctor_get(v_a_56_, 2);
v_ref_64_ = lean_ctor_get(v_a_56_, 5);
v___x_65_ = lean_unsigned_to_nat(1u);
v___x_66_ = l_Lean_Syntax_getArg(v_x_55_, v___x_65_);
lean_dec(v_x_55_);
v___x_67_ = 0;
v___x_68_ = l_Lean_SourceInfo_fromRef(v_ref_64_, v___x_67_);
v___x_69_ = ((lean_object*)(lp_mathlib_Cardinal___aux__Mathlib__SetTheory__Cardinal__Defs______macroRules__Cardinal__term_x23____1___closed__4));
v___x_70_ = lean_obj_once(&lp_mathlib_Cardinal___aux__Mathlib__SetTheory__Cardinal__Defs______macroRules__Cardinal__term_x23____1___closed__6, &lp_mathlib_Cardinal___aux__Mathlib__SetTheory__Cardinal__Defs______macroRules__Cardinal__term_x23____1___closed__6_once, _init_lp_mathlib_Cardinal___aux__Mathlib__SetTheory__Cardinal__Defs______macroRules__Cardinal__term_x23____1___closed__6);
v___x_71_ = ((lean_object*)(lp_mathlib_Cardinal___aux__Mathlib__SetTheory__Cardinal__Defs______macroRules__Cardinal__term_x23____1___closed__8));
lean_inc(v_currMacroScope_63_);
lean_inc(v_quotContext_62_);
v___x_72_ = l_Lean_addMacroScope(v_quotContext_62_, v___x_71_, v_currMacroScope_63_);
v___x_73_ = ((lean_object*)(lp_mathlib_Cardinal___aux__Mathlib__SetTheory__Cardinal__Defs______macroRules__Cardinal__term_x23____1___closed__10));
lean_inc_n(v___x_68_, 2);
v___x_74_ = lean_alloc_ctor(3, 4, 0);
lean_ctor_set(v___x_74_, 0, v___x_68_);
lean_ctor_set(v___x_74_, 1, v___x_70_);
lean_ctor_set(v___x_74_, 2, v___x_72_);
lean_ctor_set(v___x_74_, 3, v___x_73_);
v___x_75_ = ((lean_object*)(lp_mathlib_Cardinal___aux__Mathlib__SetTheory__Cardinal__Defs______macroRules__Cardinal__term_x23____1___closed__12));
v___x_76_ = l_Lean_Syntax_node1(v___x_68_, v___x_75_, v___x_66_);
v___x_77_ = l_Lean_Syntax_node2(v___x_68_, v___x_69_, v___x_74_, v___x_76_);
v___x_78_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_78_, 0, v___x_77_);
lean_ctor_set(v___x_78_, 1, v_a_57_);
return v___x_78_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Cardinal___aux__Mathlib__SetTheory__Cardinal__Defs______macroRules__Cardinal__term_x23____1___boxed(lean_object* v_x_79_, lean_object* v_a_80_, lean_object* v_a_81_){
_start:
{
lean_object* v_res_82_; 
v_res_82_ = lp_mathlib_Cardinal___aux__Mathlib__SetTheory__Cardinal__Defs______macroRules__Cardinal__term_x23____1(v_x_79_, v_a_80_, v_a_81_);
lean_dec_ref(v_a_80_);
return v_res_82_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Cardinal___aux__Mathlib__SetTheory__Cardinal__Defs______unexpand__Cardinal__mk__1(lean_object* v_x_86_, lean_object* v_a_87_, lean_object* v_a_88_){
_start:
{
lean_object* v___x_89_; uint8_t v___x_90_; 
v___x_89_ = ((lean_object*)(lp_mathlib_Cardinal___aux__Mathlib__SetTheory__Cardinal__Defs______macroRules__Cardinal__term_x23____1___closed__4));
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
v___x_95_ = ((lean_object*)(lp_mathlib_Cardinal___aux__Mathlib__SetTheory__Cardinal__Defs______unexpand__Cardinal__mk__1___closed__1));
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
lean_object* v___x_99_; lean_object* v___x_100_; uint8_t v___x_101_; 
v___x_99_ = lean_unsigned_to_nat(1u);
v___x_100_ = l_Lean_Syntax_getArg(v_x_86_, v___x_99_);
lean_dec(v_x_86_);
lean_inc(v___x_100_);
v___x_101_ = l_Lean_Syntax_matchesNull(v___x_100_, v___x_99_);
if (v___x_101_ == 0)
{
lean_object* v___x_102_; lean_object* v___x_103_; 
lean_dec(v___x_100_);
lean_dec(v___x_94_);
v___x_102_ = lean_box(0);
v___x_103_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_103_, 0, v___x_102_);
lean_ctor_set(v___x_103_, 1, v_a_88_);
return v___x_103_;
}
else
{
lean_object* v___x_104_; lean_object* v_ref_105_; uint8_t v___x_106_; lean_object* v___x_107_; lean_object* v___x_108_; lean_object* v___x_109_; lean_object* v___x_110_; lean_object* v___x_111_; lean_object* v___x_112_; 
v___x_104_ = l_Lean_Syntax_getArg(v___x_100_, v___x_93_);
lean_dec(v___x_100_);
v_ref_105_ = l_Lean_replaceRef(v___x_94_, v_a_87_);
lean_dec(v___x_94_);
v___x_106_ = 0;
v___x_107_ = l_Lean_SourceInfo_fromRef(v_ref_105_, v___x_106_);
lean_dec(v_ref_105_);
v___x_108_ = ((lean_object*)(lp_mathlib_Cardinal_term_x23___00__closed__2));
v___x_109_ = ((lean_object*)(lp_mathlib_Cardinal_term_x23___00__closed__5));
lean_inc(v___x_107_);
v___x_110_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_110_, 0, v___x_107_);
lean_ctor_set(v___x_110_, 1, v___x_109_);
v___x_111_ = l_Lean_Syntax_node2(v___x_107_, v___x_108_, v___x_110_, v___x_104_);
v___x_112_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_112_, 0, v___x_111_);
lean_ctor_set(v___x_112_, 1, v_a_88_);
return v___x_112_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Cardinal___aux__Mathlib__SetTheory__Cardinal__Defs______unexpand__Cardinal__mk__1___boxed(lean_object* v_x_113_, lean_object* v_a_114_, lean_object* v_a_115_){
_start:
{
lean_object* v_res_116_; 
v_res_116_ = lp_mathlib_Cardinal___aux__Mathlib__SetTheory__Cardinal__Defs______unexpand__Cardinal__mk__1(v_x_113_, v_a_114_, v_a_115_);
lean_dec(v_a_114_);
return v_res_116_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Cardinal_map(lean_object* v_f_117_, lean_object* v_hf_118_, lean_object* v_a_119_){
_start:
{
lean_object* v___x_120_; 
v___x_120_ = lean_box(0);
return v___x_120_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Cardinal_map___boxed(lean_object* v_f_121_, lean_object* v_hf_122_, lean_object* v_a_123_){
_start:
{
lean_object* v_res_124_; 
v_res_124_ = lp_mathlib_Cardinal_map(v_f_121_, v_hf_122_, v_a_123_);
lean_dec(v_a_123_);
lean_dec_ref(v_hf_122_);
return v_res_124_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Quotient_map_u2082___at___00Cardinal_map_u2082_spec__0(lean_object* v_f_125_, lean_object* v_h_126_, lean_object* v_q_u2081_127_, lean_object* v_q_u2082_128_){
_start:
{
lean_object* v___x_129_; 
v___x_129_ = lean_box(0);
return v___x_129_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Quotient_map_u2082___at___00Cardinal_map_u2082_spec__0___boxed(lean_object* v_f_130_, lean_object* v_h_131_, lean_object* v_q_u2081_132_, lean_object* v_q_u2082_133_){
_start:
{
lean_object* v_res_134_; 
v_res_134_ = lp_mathlib_Quotient_map_u2082___at___00Cardinal_map_u2082_spec__0(v_f_130_, v_h_131_, v_q_u2081_132_, v_q_u2082_133_);
lean_dec(v_q_u2082_133_);
lean_dec(v_q_u2081_132_);
return v_res_134_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Cardinal_map_u2082___redArg(lean_object* v_a_135_, lean_object* v_a_136_){
_start:
{
lean_object* v___x_137_; 
v___x_137_ = lean_box(0);
return v___x_137_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Cardinal_map_u2082___redArg___boxed(lean_object* v_a_138_, lean_object* v_a_139_){
_start:
{
lean_object* v_res_140_; 
v_res_140_ = lp_mathlib_Cardinal_map_u2082___redArg(v_a_138_, v_a_139_);
lean_dec(v_a_139_);
lean_dec(v_a_138_);
return v_res_140_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Cardinal_map_u2082(lean_object* v_f_141_, lean_object* v_hf_142_, lean_object* v_a_143_, lean_object* v_a_144_){
_start:
{
lean_object* v___x_145_; 
v___x_145_ = lean_box(0);
return v___x_145_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Cardinal_map_u2082___boxed(lean_object* v_f_146_, lean_object* v_hf_147_, lean_object* v_a_148_, lean_object* v_a_149_){
_start:
{
lean_object* v_res_150_; 
v_res_150_ = lp_mathlib_Cardinal_map_u2082(v_f_146_, v_hf_147_, v_a_148_, v_a_149_);
lean_dec(v_a_149_);
lean_dec(v_a_148_);
lean_dec_ref(v_hf_147_);
return v_res_150_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Cardinal_lift(lean_object* v_c_151_){
_start:
{
lean_object* v___x_152_; 
v___x_152_ = lean_box(0);
return v___x_152_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Cardinal_lift___boxed(lean_object* v_c_153_){
_start:
{
lean_object* v_res_154_; 
v_res_154_ = lp_mathlib_Cardinal_lift(v_c_153_);
lean_dec(v_c_153_);
return v_res_154_;
}
}
static lean_object* _init_lp_mathlib_Cardinal_instZero(void){
_start:
{
lean_object* v___x_155_; 
v___x_155_ = lean_box(0);
return v___x_155_;
}
}
static lean_object* _init_lp_mathlib_Cardinal_instInhabited(void){
_start:
{
lean_object* v___x_156_; 
v___x_156_ = lean_box(0);
return v___x_156_;
}
}
static lean_object* _init_lp_mathlib_Cardinal_instOne(void){
_start:
{
lean_object* v___x_157_; 
v___x_157_ = lean_box(0);
return v___x_157_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Cardinal_instAdd___lam__0(lean_object* v_x_158_, lean_object* v_x_159_, lean_object* v_x_160_, lean_object* v_x_161_, lean_object* v___y_162_, lean_object* v___y_163_){
_start:
{
lean_object* v___x_164_; 
v___x_164_ = lp_mathlib_Equiv_sumCongr___redArg(v___y_162_, v___y_163_);
return v___x_164_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Cardinal_instNatCast___lam__0(lean_object* v_n_169_){
_start:
{
lean_object* v___x_170_; 
v___x_170_ = lean_box(0);
return v___x_170_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Cardinal_instNatCast___lam__0___boxed(lean_object* v_n_171_){
_start:
{
lean_object* v_res_172_; 
v_res_172_ = lp_mathlib_Cardinal_instNatCast___lam__0(v_n_171_);
lean_dec(v_n_171_);
return v_res_172_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Cardinal_instMul___lam__0(lean_object* v_x_175_, lean_object* v_x_176_, lean_object* v_x_177_, lean_object* v_x_178_, lean_object* v___y_179_, lean_object* v___y_180_){
_start:
{
lean_object* v___x_181_; 
v___x_181_ = lp_mathlib_Equiv_prodCongr___redArg(v___y_179_, v___y_180_);
return v___x_181_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Cardinal_instPowCardinal___lam__0(lean_object* v_x_186_, lean_object* v_x_187_, lean_object* v_x_188_, lean_object* v_x_189_, lean_object* v_e_u2081_190_, lean_object* v_e_u2082_191_){
_start:
{
lean_object* v___x_192_; 
v___x_192_ = lp_mathlib_Equiv_arrowCongr___redArg(v_e_u2082_191_, v_e_u2081_190_);
return v___x_192_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Cardinal_sum(lean_object* v_00_u03b9_197_, lean_object* v_f_198_){
_start:
{
lean_object* v___x_199_; 
v___x_199_ = lean_box(0);
return v___x_199_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Cardinal_sum___boxed(lean_object* v_00_u03b9_200_, lean_object* v_f_201_){
_start:
{
lean_object* v_res_202_; 
v_res_202_ = lp_mathlib_Cardinal_sum(v_00_u03b9_200_, v_f_201_);
lean_dec(v_f_201_);
return v_res_202_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Cardinal_prod(lean_object* v_00_u03b9_203_, lean_object* v_f_204_){
_start:
{
lean_object* v___x_205_; 
v___x_205_ = lean_box(0);
return v___x_205_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Cardinal_prod___boxed(lean_object* v_00_u03b9_206_, lean_object* v_f_207_){
_start:
{
lean_object* v_res_208_; 
v_res_208_ = lp_mathlib_Cardinal_prod(v_00_u03b9_206_, v_f_207_);
lean_dec(v_f_207_);
return v_res_208_;
}
}
static lean_object* _init_lp_mathlib_Cardinal_aleph0(void){
_start:
{
lean_object* v___x_209_; 
v___x_209_ = lean_box(0);
return v___x_209_;
}
}
static lean_object* _init_lp_mathlib_Cardinal___aux__Mathlib__SetTheory__Cardinal__Defs______macroRules__Cardinal__term_u2135_u2080__1___closed__1(void){
_start:
{
lean_object* v___x_223_; lean_object* v___x_224_; 
v___x_223_ = ((lean_object*)(lp_mathlib_Cardinal___aux__Mathlib__SetTheory__Cardinal__Defs______macroRules__Cardinal__term_u2135_u2080__1___closed__0));
v___x_224_ = l_String_toRawSubstring_x27(v___x_223_);
return v___x_224_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Cardinal___aux__Mathlib__SetTheory__Cardinal__Defs______macroRules__Cardinal__term_u2135_u2080__1(lean_object* v_x_235_, lean_object* v_a_236_, lean_object* v_a_237_){
_start:
{
lean_object* v___x_238_; uint8_t v___x_239_; 
v___x_238_ = ((lean_object*)(lp_mathlib_Cardinal_term_u2135_u2080___closed__1));
v___x_239_ = l_Lean_Syntax_isOfKind(v_x_235_, v___x_238_);
if (v___x_239_ == 0)
{
lean_object* v___x_240_; lean_object* v___x_241_; 
v___x_240_ = lean_box(1);
v___x_241_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_241_, 0, v___x_240_);
lean_ctor_set(v___x_241_, 1, v_a_237_);
return v___x_241_;
}
else
{
lean_object* v_quotContext_242_; lean_object* v_currMacroScope_243_; lean_object* v_ref_244_; uint8_t v___x_245_; lean_object* v___x_246_; lean_object* v___x_247_; lean_object* v___x_248_; lean_object* v___x_249_; lean_object* v___x_250_; lean_object* v___x_251_; lean_object* v___x_252_; 
v_quotContext_242_ = lean_ctor_get(v_a_236_, 1);
v_currMacroScope_243_ = lean_ctor_get(v_a_236_, 2);
v_ref_244_ = lean_ctor_get(v_a_236_, 5);
v___x_245_ = 0;
v___x_246_ = l_Lean_SourceInfo_fromRef(v_ref_244_, v___x_245_);
v___x_247_ = lean_obj_once(&lp_mathlib_Cardinal___aux__Mathlib__SetTheory__Cardinal__Defs______macroRules__Cardinal__term_u2135_u2080__1___closed__1, &lp_mathlib_Cardinal___aux__Mathlib__SetTheory__Cardinal__Defs______macroRules__Cardinal__term_u2135_u2080__1___closed__1_once, _init_lp_mathlib_Cardinal___aux__Mathlib__SetTheory__Cardinal__Defs______macroRules__Cardinal__term_u2135_u2080__1___closed__1);
v___x_248_ = ((lean_object*)(lp_mathlib_Cardinal___aux__Mathlib__SetTheory__Cardinal__Defs______macroRules__Cardinal__term_u2135_u2080__1___closed__3));
lean_inc(v_currMacroScope_243_);
lean_inc(v_quotContext_242_);
v___x_249_ = l_Lean_addMacroScope(v_quotContext_242_, v___x_248_, v_currMacroScope_243_);
v___x_250_ = ((lean_object*)(lp_mathlib_Cardinal___aux__Mathlib__SetTheory__Cardinal__Defs______macroRules__Cardinal__term_u2135_u2080__1___closed__5));
v___x_251_ = lean_alloc_ctor(3, 4, 0);
lean_ctor_set(v___x_251_, 0, v___x_246_);
lean_ctor_set(v___x_251_, 1, v___x_247_);
lean_ctor_set(v___x_251_, 2, v___x_249_);
lean_ctor_set(v___x_251_, 3, v___x_250_);
v___x_252_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_252_, 0, v___x_251_);
lean_ctor_set(v___x_252_, 1, v_a_237_);
return v___x_252_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Cardinal___aux__Mathlib__SetTheory__Cardinal__Defs______macroRules__Cardinal__term_u2135_u2080__1___boxed(lean_object* v_x_253_, lean_object* v_a_254_, lean_object* v_a_255_){
_start:
{
lean_object* v_res_256_; 
v_res_256_ = lp_mathlib_Cardinal___aux__Mathlib__SetTheory__Cardinal__Defs______macroRules__Cardinal__term_u2135_u2080__1(v_x_253_, v_a_254_, v_a_255_);
lean_dec_ref(v_a_254_);
return v_res_256_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Cardinal___aux__Mathlib__SetTheory__Cardinal__Defs______unexpand__Cardinal__aleph0__1(lean_object* v_x_257_, lean_object* v_a_258_, lean_object* v_a_259_){
_start:
{
lean_object* v___x_260_; uint8_t v___x_261_; 
v___x_260_ = ((lean_object*)(lp_mathlib_Cardinal___aux__Mathlib__SetTheory__Cardinal__Defs______unexpand__Cardinal__mk__1___closed__1));
lean_inc(v_x_257_);
v___x_261_ = l_Lean_Syntax_isOfKind(v_x_257_, v___x_260_);
if (v___x_261_ == 0)
{
lean_object* v___x_262_; lean_object* v___x_263_; 
lean_dec(v_x_257_);
v___x_262_ = lean_box(0);
v___x_263_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_263_, 0, v___x_262_);
lean_ctor_set(v___x_263_, 1, v_a_259_);
return v___x_263_;
}
else
{
lean_object* v_ref_264_; uint8_t v___x_265_; lean_object* v___x_266_; lean_object* v___x_267_; lean_object* v___x_268_; lean_object* v___x_269_; lean_object* v___x_270_; lean_object* v___x_271_; 
v_ref_264_ = l_Lean_replaceRef(v_x_257_, v_a_258_);
lean_dec(v_x_257_);
v___x_265_ = 0;
v___x_266_ = l_Lean_SourceInfo_fromRef(v_ref_264_, v___x_265_);
lean_dec(v_ref_264_);
v___x_267_ = ((lean_object*)(lp_mathlib_Cardinal_term_u2135_u2080___closed__1));
v___x_268_ = ((lean_object*)(lp_mathlib_Cardinal_term_u2135_u2080___closed__2));
lean_inc(v___x_266_);
v___x_269_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_269_, 0, v___x_266_);
lean_ctor_set(v___x_269_, 1, v___x_268_);
v___x_270_ = l_Lean_Syntax_node1(v___x_266_, v___x_267_, v___x_269_);
v___x_271_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_271_, 0, v___x_270_);
lean_ctor_set(v___x_271_, 1, v_a_259_);
return v___x_271_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Cardinal___aux__Mathlib__SetTheory__Cardinal__Defs______unexpand__Cardinal__aleph0__1___boxed(lean_object* v_x_272_, lean_object* v_a_273_, lean_object* v_a_274_){
_start:
{
lean_object* v_res_275_; 
v_res_275_ = lp_mathlib_Cardinal___aux__Mathlib__SetTheory__Cardinal__Defs______unexpand__Cardinal__aleph0__1(v_x_272_, v_a_273_, v_a_274_);
lean_dec(v_a_273_);
return v_res_275_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Basic_IsEmpty_Basic(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Data_ULift(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Tactic_CrossRefAttribute(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Tactic_PPWithUniv(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Util_Delaborators(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_SetTheory_Cardinal_Defs(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Basic_IsEmpty_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Data_ULift(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Tactic_CrossRefAttribute(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Tactic_PPWithUniv(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Util_Delaborators(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
lp_mathlib_Cardinal_isEquivalent = _init_lp_mathlib_Cardinal_isEquivalent();
lean_mark_persistent(lp_mathlib_Cardinal_isEquivalent);
lp_mathlib_Cardinal_instZero = _init_lp_mathlib_Cardinal_instZero();
lean_mark_persistent(lp_mathlib_Cardinal_instZero);
lp_mathlib_Cardinal_instInhabited = _init_lp_mathlib_Cardinal_instInhabited();
lean_mark_persistent(lp_mathlib_Cardinal_instInhabited);
lp_mathlib_Cardinal_instOne = _init_lp_mathlib_Cardinal_instOne();
lean_mark_persistent(lp_mathlib_Cardinal_instOne);
lp_mathlib_Cardinal_aleph0 = _init_lp_mathlib_Cardinal_aleph0();
lean_mark_persistent(lp_mathlib_Cardinal_aleph0);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_SetTheory_Cardinal_Defs(uint8_t builtin) {
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
lean_object* initialize_mathlib_Mathlib_Basic_IsEmpty_Basic(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Data_ULift(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Tactic_CrossRefAttribute(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Tactic_PPWithUniv(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Util_Delaborators(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_SetTheory_Cardinal_Defs(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Basic_IsEmpty_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_ULift(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Tactic_CrossRefAttribute(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Tactic_PPWithUniv(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Util_Delaborators(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_SetTheory_Cardinal_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_SetTheory_Cardinal_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_SetTheory_Cardinal_Defs(builtin);
}
#ifdef __cplusplus
}
#endif
