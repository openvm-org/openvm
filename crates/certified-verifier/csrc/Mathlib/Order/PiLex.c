// Lean compiler output
// Module: Mathlib.Order.PiLex
// Imports: public import Init public meta import Init public import Mathlib.Order.Lex public import Mathlib.Order.WellFounded public import Mathlib.Tactic.Common public import Mathlib.Tactic.Attr.Core
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
lean_object* lp_mathlib_Equiv_refl(lean_object*);
lean_object* l_String_toRawSubstring_x27(lean_object*);
lean_object* l_Lean_Name_mkStr4(lean_object*, lean_object*, lean_object*, lean_object*);
extern lean_object* lp_batteries_Batteries_ExtendedBinder_extBinders;
lean_object* l_Lean_Name_mkStr2(lean_object*, lean_object*);
uint8_t l_Lean_Syntax_isOfKind(lean_object*, lean_object*);
lean_object* l_Lean_Syntax_getArg(lean_object*, lean_object*);
lean_object* l_Lean_SourceInfo_fromRef(lean_object*, uint8_t);
lean_object* l_Lean_Name_mkStr3(lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_addMacroScope(lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Syntax_node1(lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Syntax_node2(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Array_mkArray0(lean_object*);
lean_object* l_Lean_Syntax_node5(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Syntax_node3(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lean_mk_empty_array_with_capacity(lean_object*);
lean_object* lean_array_push(lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Pi_term_u03a0_u2097___x2c___00__closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 3, .m_capacity = 3, .m_length = 2, .m_data = "Pi"};
static const lean_object* lp_mathlib_Pi_term_u03a0_u2097___x2c___00__closed__0 = (const lean_object*)&lp_mathlib_Pi_term_u03a0_u2097___x2c___00__closed__0_value;
static const lean_string_object lp_mathlib_Pi_term_u03a0_u2097___x2c___00__closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 13, .m_capacity = 13, .m_length = 9, .m_data = "termΠₗ_,_"};
static const lean_object* lp_mathlib_Pi_term_u03a0_u2097___x2c___00__closed__1 = (const lean_object*)&lp_mathlib_Pi_term_u03a0_u2097___x2c___00__closed__1_value;
static const lean_ctor_object lp_mathlib_Pi_term_u03a0_u2097___x2c___00__closed__2_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Pi_term_u03a0_u2097___x2c___00__closed__0_value),LEAN_SCALAR_PTR_LITERAL(236, 177, 129, 149, 110, 155, 170, 220)}};
static const lean_ctor_object lp_mathlib_Pi_term_u03a0_u2097___x2c___00__closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Pi_term_u03a0_u2097___x2c___00__closed__2_value_aux_0),((lean_object*)&lp_mathlib_Pi_term_u03a0_u2097___x2c___00__closed__1_value),LEAN_SCALAR_PTR_LITERAL(108, 18, 23, 224, 4, 17, 217, 79)}};
static const lean_object* lp_mathlib_Pi_term_u03a0_u2097___x2c___00__closed__2 = (const lean_object*)&lp_mathlib_Pi_term_u03a0_u2097___x2c___00__closed__2_value;
static const lean_string_object lp_mathlib_Pi_term_u03a0_u2097___x2c___00__closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "andthen"};
static const lean_object* lp_mathlib_Pi_term_u03a0_u2097___x2c___00__closed__3 = (const lean_object*)&lp_mathlib_Pi_term_u03a0_u2097___x2c___00__closed__3_value;
static const lean_ctor_object lp_mathlib_Pi_term_u03a0_u2097___x2c___00__closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Pi_term_u03a0_u2097___x2c___00__closed__3_value),LEAN_SCALAR_PTR_LITERAL(40, 255, 78, 30, 143, 119, 117, 174)}};
static const lean_object* lp_mathlib_Pi_term_u03a0_u2097___x2c___00__closed__4 = (const lean_object*)&lp_mathlib_Pi_term_u03a0_u2097___x2c___00__closed__4_value;
static const lean_string_object lp_mathlib_Pi_term_u03a0_u2097___x2c___00__closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 2, .m_data = "Πₗ"};
static const lean_object* lp_mathlib_Pi_term_u03a0_u2097___x2c___00__closed__5 = (const lean_object*)&lp_mathlib_Pi_term_u03a0_u2097___x2c___00__closed__5_value;
static const lean_ctor_object lp_mathlib_Pi_term_u03a0_u2097___x2c___00__closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 5}, .m_objs = {((lean_object*)&lp_mathlib_Pi_term_u03a0_u2097___x2c___00__closed__5_value)}};
static const lean_object* lp_mathlib_Pi_term_u03a0_u2097___x2c___00__closed__6 = (const lean_object*)&lp_mathlib_Pi_term_u03a0_u2097___x2c___00__closed__6_value;
static lean_once_cell_t lp_mathlib_Pi_term_u03a0_u2097___x2c___00__closed__7_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Pi_term_u03a0_u2097___x2c___00__closed__7;
static const lean_string_object lp_mathlib_Pi_term_u03a0_u2097___x2c___00__closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 3, .m_capacity = 3, .m_length = 2, .m_data = ", "};
static const lean_object* lp_mathlib_Pi_term_u03a0_u2097___x2c___00__closed__8 = (const lean_object*)&lp_mathlib_Pi_term_u03a0_u2097___x2c___00__closed__8_value;
static const lean_ctor_object lp_mathlib_Pi_term_u03a0_u2097___x2c___00__closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 5}, .m_objs = {((lean_object*)&lp_mathlib_Pi_term_u03a0_u2097___x2c___00__closed__8_value)}};
static const lean_object* lp_mathlib_Pi_term_u03a0_u2097___x2c___00__closed__9 = (const lean_object*)&lp_mathlib_Pi_term_u03a0_u2097___x2c___00__closed__9_value;
static lean_once_cell_t lp_mathlib_Pi_term_u03a0_u2097___x2c___00__closed__10_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Pi_term_u03a0_u2097___x2c___00__closed__10;
static const lean_string_object lp_mathlib_Pi_term_u03a0_u2097___x2c___00__closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "term"};
static const lean_object* lp_mathlib_Pi_term_u03a0_u2097___x2c___00__closed__11 = (const lean_object*)&lp_mathlib_Pi_term_u03a0_u2097___x2c___00__closed__11_value;
static const lean_ctor_object lp_mathlib_Pi_term_u03a0_u2097___x2c___00__closed__12_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Pi_term_u03a0_u2097___x2c___00__closed__11_value),LEAN_SCALAR_PTR_LITERAL(187, 230, 181, 162, 253, 146, 122, 119)}};
static const lean_object* lp_mathlib_Pi_term_u03a0_u2097___x2c___00__closed__12 = (const lean_object*)&lp_mathlib_Pi_term_u03a0_u2097___x2c___00__closed__12_value;
static const lean_ctor_object lp_mathlib_Pi_term_u03a0_u2097___x2c___00__closed__13_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 7}, .m_objs = {((lean_object*)&lp_mathlib_Pi_term_u03a0_u2097___x2c___00__closed__12_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_Pi_term_u03a0_u2097___x2c___00__closed__13 = (const lean_object*)&lp_mathlib_Pi_term_u03a0_u2097___x2c___00__closed__13_value;
static lean_once_cell_t lp_mathlib_Pi_term_u03a0_u2097___x2c___00__closed__14_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Pi_term_u03a0_u2097___x2c___00__closed__14;
static lean_once_cell_t lp_mathlib_Pi_term_u03a0_u2097___x2c___00__closed__15_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Pi_term_u03a0_u2097___x2c___00__closed__15;
LEAN_EXPORT lean_object* lp_mathlib_Pi_term_u03a0_u2097___x2c__;
static const lean_string_object lp_mathlib_Pi___aux__Mathlib__Order__PiLex______macroRules__Pi__term_u03a0_u2097___x2c____1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "Mathlib"};
static const lean_object* lp_mathlib_Pi___aux__Mathlib__Order__PiLex______macroRules__Pi__term_u03a0_u2097___x2c____1___closed__0 = (const lean_object*)&lp_mathlib_Pi___aux__Mathlib__Order__PiLex______macroRules__Pi__term_u03a0_u2097___x2c____1___closed__0_value;
static const lean_string_object lp_mathlib_Pi___aux__Mathlib__Order__PiLex______macroRules__Pi__term_u03a0_u2097___x2c____1___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 9, .m_data = "Notation3"};
static const lean_object* lp_mathlib_Pi___aux__Mathlib__Order__PiLex______macroRules__Pi__term_u03a0_u2097___x2c____1___closed__1 = (const lean_object*)&lp_mathlib_Pi___aux__Mathlib__Order__PiLex______macroRules__Pi__term_u03a0_u2097___x2c____1___closed__1_value;
static const lean_string_object lp_mathlib_Pi___aux__Mathlib__Order__PiLex______macroRules__Pi__term_u03a0_u2097___x2c____1___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 29, .m_capacity = 29, .m_length = 28, .m_data = "termExpand_binders%(_=>_)_,_"};
static const lean_object* lp_mathlib_Pi___aux__Mathlib__Order__PiLex______macroRules__Pi__term_u03a0_u2097___x2c____1___closed__2 = (const lean_object*)&lp_mathlib_Pi___aux__Mathlib__Order__PiLex______macroRules__Pi__term_u03a0_u2097___x2c____1___closed__2_value;
static const lean_ctor_object lp_mathlib_Pi___aux__Mathlib__Order__PiLex______macroRules__Pi__term_u03a0_u2097___x2c____1___closed__3_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Pi___aux__Mathlib__Order__PiLex______macroRules__Pi__term_u03a0_u2097___x2c____1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(118, 213, 161, 2, 73, 184, 31, 228)}};
static const lean_ctor_object lp_mathlib_Pi___aux__Mathlib__Order__PiLex______macroRules__Pi__term_u03a0_u2097___x2c____1___closed__3_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Pi___aux__Mathlib__Order__PiLex______macroRules__Pi__term_u03a0_u2097___x2c____1___closed__3_value_aux_0),((lean_object*)&lp_mathlib_Pi___aux__Mathlib__Order__PiLex______macroRules__Pi__term_u03a0_u2097___x2c____1___closed__1_value),LEAN_SCALAR_PTR_LITERAL(187, 176, 22, 214, 10, 13, 147, 22)}};
static const lean_ctor_object lp_mathlib_Pi___aux__Mathlib__Order__PiLex______macroRules__Pi__term_u03a0_u2097___x2c____1___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Pi___aux__Mathlib__Order__PiLex______macroRules__Pi__term_u03a0_u2097___x2c____1___closed__3_value_aux_1),((lean_object*)&lp_mathlib_Pi___aux__Mathlib__Order__PiLex______macroRules__Pi__term_u03a0_u2097___x2c____1___closed__2_value),LEAN_SCALAR_PTR_LITERAL(120, 7, 237, 26, 3, 243, 131, 214)}};
static const lean_object* lp_mathlib_Pi___aux__Mathlib__Order__PiLex______macroRules__Pi__term_u03a0_u2097___x2c____1___closed__3 = (const lean_object*)&lp_mathlib_Pi___aux__Mathlib__Order__PiLex______macroRules__Pi__term_u03a0_u2097___x2c____1___closed__3_value;
static const lean_string_object lp_mathlib_Pi___aux__Mathlib__Order__PiLex______macroRules__Pi__term_u03a0_u2097___x2c____1___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 16, .m_capacity = 16, .m_length = 15, .m_data = "expand_binders%"};
static const lean_object* lp_mathlib_Pi___aux__Mathlib__Order__PiLex______macroRules__Pi__term_u03a0_u2097___x2c____1___closed__4 = (const lean_object*)&lp_mathlib_Pi___aux__Mathlib__Order__PiLex______macroRules__Pi__term_u03a0_u2097___x2c____1___closed__4_value;
static const lean_string_object lp_mathlib_Pi___aux__Mathlib__Order__PiLex______macroRules__Pi__term_u03a0_u2097___x2c____1___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = "("};
static const lean_object* lp_mathlib_Pi___aux__Mathlib__Order__PiLex______macroRules__Pi__term_u03a0_u2097___x2c____1___closed__5 = (const lean_object*)&lp_mathlib_Pi___aux__Mathlib__Order__PiLex______macroRules__Pi__term_u03a0_u2097___x2c____1___closed__5_value;
static const lean_string_object lp_mathlib_Pi___aux__Mathlib__Order__PiLex______macroRules__Pi__term_u03a0_u2097___x2c____1___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = "p"};
static const lean_object* lp_mathlib_Pi___aux__Mathlib__Order__PiLex______macroRules__Pi__term_u03a0_u2097___x2c____1___closed__6 = (const lean_object*)&lp_mathlib_Pi___aux__Mathlib__Order__PiLex______macroRules__Pi__term_u03a0_u2097___x2c____1___closed__6_value;
static lean_once_cell_t lp_mathlib_Pi___aux__Mathlib__Order__PiLex______macroRules__Pi__term_u03a0_u2097___x2c____1___closed__7_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Pi___aux__Mathlib__Order__PiLex______macroRules__Pi__term_u03a0_u2097___x2c____1___closed__7;
static const lean_ctor_object lp_mathlib_Pi___aux__Mathlib__Order__PiLex______macroRules__Pi__term_u03a0_u2097___x2c____1___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Pi___aux__Mathlib__Order__PiLex______macroRules__Pi__term_u03a0_u2097___x2c____1___closed__6_value),LEAN_SCALAR_PTR_LITERAL(34, 153, 146, 175, 179, 220, 230, 134)}};
static const lean_object* lp_mathlib_Pi___aux__Mathlib__Order__PiLex______macroRules__Pi__term_u03a0_u2097___x2c____1___closed__8 = (const lean_object*)&lp_mathlib_Pi___aux__Mathlib__Order__PiLex______macroRules__Pi__term_u03a0_u2097___x2c____1___closed__8_value;
static const lean_string_object lp_mathlib_Pi___aux__Mathlib__Order__PiLex______macroRules__Pi__term_u03a0_u2097___x2c____1___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 3, .m_capacity = 3, .m_length = 2, .m_data = "=>"};
static const lean_object* lp_mathlib_Pi___aux__Mathlib__Order__PiLex______macroRules__Pi__term_u03a0_u2097___x2c____1___closed__9 = (const lean_object*)&lp_mathlib_Pi___aux__Mathlib__Order__PiLex______macroRules__Pi__term_u03a0_u2097___x2c____1___closed__9_value;
static const lean_string_object lp_mathlib_Pi___aux__Mathlib__Order__PiLex______macroRules__Pi__term_u03a0_u2097___x2c____1___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "Lean"};
static const lean_object* lp_mathlib_Pi___aux__Mathlib__Order__PiLex______macroRules__Pi__term_u03a0_u2097___x2c____1___closed__10 = (const lean_object*)&lp_mathlib_Pi___aux__Mathlib__Order__PiLex______macroRules__Pi__term_u03a0_u2097___x2c____1___closed__10_value;
static const lean_string_object lp_mathlib_Pi___aux__Mathlib__Order__PiLex______macroRules__Pi__term_u03a0_u2097___x2c____1___closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "Parser"};
static const lean_object* lp_mathlib_Pi___aux__Mathlib__Order__PiLex______macroRules__Pi__term_u03a0_u2097___x2c____1___closed__11 = (const lean_object*)&lp_mathlib_Pi___aux__Mathlib__Order__PiLex______macroRules__Pi__term_u03a0_u2097___x2c____1___closed__11_value;
static const lean_string_object lp_mathlib_Pi___aux__Mathlib__Order__PiLex______macroRules__Pi__term_u03a0_u2097___x2c____1___closed__12_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "Term"};
static const lean_object* lp_mathlib_Pi___aux__Mathlib__Order__PiLex______macroRules__Pi__term_u03a0_u2097___x2c____1___closed__12 = (const lean_object*)&lp_mathlib_Pi___aux__Mathlib__Order__PiLex______macroRules__Pi__term_u03a0_u2097___x2c____1___closed__12_value;
static const lean_string_object lp_mathlib_Pi___aux__Mathlib__Order__PiLex______macroRules__Pi__term_u03a0_u2097___x2c____1___closed__13_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = "app"};
static const lean_object* lp_mathlib_Pi___aux__Mathlib__Order__PiLex______macroRules__Pi__term_u03a0_u2097___x2c____1___closed__13 = (const lean_object*)&lp_mathlib_Pi___aux__Mathlib__Order__PiLex______macroRules__Pi__term_u03a0_u2097___x2c____1___closed__13_value;
static const lean_ctor_object lp_mathlib_Pi___aux__Mathlib__Order__PiLex______macroRules__Pi__term_u03a0_u2097___x2c____1___closed__14_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Pi___aux__Mathlib__Order__PiLex______macroRules__Pi__term_u03a0_u2097___x2c____1___closed__10_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib_Pi___aux__Mathlib__Order__PiLex______macroRules__Pi__term_u03a0_u2097___x2c____1___closed__14_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Pi___aux__Mathlib__Order__PiLex______macroRules__Pi__term_u03a0_u2097___x2c____1___closed__14_value_aux_0),((lean_object*)&lp_mathlib_Pi___aux__Mathlib__Order__PiLex______macroRules__Pi__term_u03a0_u2097___x2c____1___closed__11_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib_Pi___aux__Mathlib__Order__PiLex______macroRules__Pi__term_u03a0_u2097___x2c____1___closed__14_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Pi___aux__Mathlib__Order__PiLex______macroRules__Pi__term_u03a0_u2097___x2c____1___closed__14_value_aux_1),((lean_object*)&lp_mathlib_Pi___aux__Mathlib__Order__PiLex______macroRules__Pi__term_u03a0_u2097___x2c____1___closed__12_value),LEAN_SCALAR_PTR_LITERAL(75, 170, 162, 138, 136, 204, 251, 229)}};
static const lean_ctor_object lp_mathlib_Pi___aux__Mathlib__Order__PiLex______macroRules__Pi__term_u03a0_u2097___x2c____1___closed__14_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Pi___aux__Mathlib__Order__PiLex______macroRules__Pi__term_u03a0_u2097___x2c____1___closed__14_value_aux_2),((lean_object*)&lp_mathlib_Pi___aux__Mathlib__Order__PiLex______macroRules__Pi__term_u03a0_u2097___x2c____1___closed__13_value),LEAN_SCALAR_PTR_LITERAL(69, 118, 10, 41, 220, 156, 243, 179)}};
static const lean_object* lp_mathlib_Pi___aux__Mathlib__Order__PiLex______macroRules__Pi__term_u03a0_u2097___x2c____1___closed__14 = (const lean_object*)&lp_mathlib_Pi___aux__Mathlib__Order__PiLex______macroRules__Pi__term_u03a0_u2097___x2c____1___closed__14_value;
static const lean_string_object lp_mathlib_Pi___aux__Mathlib__Order__PiLex______macroRules__Pi__term_u03a0_u2097___x2c____1___closed__15_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = "Lex"};
static const lean_object* lp_mathlib_Pi___aux__Mathlib__Order__PiLex______macroRules__Pi__term_u03a0_u2097___x2c____1___closed__15 = (const lean_object*)&lp_mathlib_Pi___aux__Mathlib__Order__PiLex______macroRules__Pi__term_u03a0_u2097___x2c____1___closed__15_value;
static lean_once_cell_t lp_mathlib_Pi___aux__Mathlib__Order__PiLex______macroRules__Pi__term_u03a0_u2097___x2c____1___closed__16_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Pi___aux__Mathlib__Order__PiLex______macroRules__Pi__term_u03a0_u2097___x2c____1___closed__16;
static const lean_ctor_object lp_mathlib_Pi___aux__Mathlib__Order__PiLex______macroRules__Pi__term_u03a0_u2097___x2c____1___closed__17_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Pi___aux__Mathlib__Order__PiLex______macroRules__Pi__term_u03a0_u2097___x2c____1___closed__15_value),LEAN_SCALAR_PTR_LITERAL(47, 205, 122, 164, 96, 181, 7, 42)}};
static const lean_object* lp_mathlib_Pi___aux__Mathlib__Order__PiLex______macroRules__Pi__term_u03a0_u2097___x2c____1___closed__17 = (const lean_object*)&lp_mathlib_Pi___aux__Mathlib__Order__PiLex______macroRules__Pi__term_u03a0_u2097___x2c____1___closed__17_value;
static const lean_ctor_object lp_mathlib_Pi___aux__Mathlib__Order__PiLex______macroRules__Pi__term_u03a0_u2097___x2c____1___closed__18_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Pi___aux__Mathlib__Order__PiLex______macroRules__Pi__term_u03a0_u2097___x2c____1___closed__17_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_Pi___aux__Mathlib__Order__PiLex______macroRules__Pi__term_u03a0_u2097___x2c____1___closed__18 = (const lean_object*)&lp_mathlib_Pi___aux__Mathlib__Order__PiLex______macroRules__Pi__term_u03a0_u2097___x2c____1___closed__18_value;
static const lean_ctor_object lp_mathlib_Pi___aux__Mathlib__Order__PiLex______macroRules__Pi__term_u03a0_u2097___x2c____1___closed__19_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_Pi___aux__Mathlib__Order__PiLex______macroRules__Pi__term_u03a0_u2097___x2c____1___closed__17_value)}};
static const lean_object* lp_mathlib_Pi___aux__Mathlib__Order__PiLex______macroRules__Pi__term_u03a0_u2097___x2c____1___closed__19 = (const lean_object*)&lp_mathlib_Pi___aux__Mathlib__Order__PiLex______macroRules__Pi__term_u03a0_u2097___x2c____1___closed__19_value;
static const lean_ctor_object lp_mathlib_Pi___aux__Mathlib__Order__PiLex______macroRules__Pi__term_u03a0_u2097___x2c____1___closed__20_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Pi___aux__Mathlib__Order__PiLex______macroRules__Pi__term_u03a0_u2097___x2c____1___closed__19_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_Pi___aux__Mathlib__Order__PiLex______macroRules__Pi__term_u03a0_u2097___x2c____1___closed__20 = (const lean_object*)&lp_mathlib_Pi___aux__Mathlib__Order__PiLex______macroRules__Pi__term_u03a0_u2097___x2c____1___closed__20_value;
static const lean_ctor_object lp_mathlib_Pi___aux__Mathlib__Order__PiLex______macroRules__Pi__term_u03a0_u2097___x2c____1___closed__21_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Pi___aux__Mathlib__Order__PiLex______macroRules__Pi__term_u03a0_u2097___x2c____1___closed__18_value),((lean_object*)&lp_mathlib_Pi___aux__Mathlib__Order__PiLex______macroRules__Pi__term_u03a0_u2097___x2c____1___closed__20_value)}};
static const lean_object* lp_mathlib_Pi___aux__Mathlib__Order__PiLex______macroRules__Pi__term_u03a0_u2097___x2c____1___closed__21 = (const lean_object*)&lp_mathlib_Pi___aux__Mathlib__Order__PiLex______macroRules__Pi__term_u03a0_u2097___x2c____1___closed__21_value;
static const lean_string_object lp_mathlib_Pi___aux__Mathlib__Order__PiLex______macroRules__Pi__term_u03a0_u2097___x2c____1___closed__22_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "null"};
static const lean_object* lp_mathlib_Pi___aux__Mathlib__Order__PiLex______macroRules__Pi__term_u03a0_u2097___x2c____1___closed__22 = (const lean_object*)&lp_mathlib_Pi___aux__Mathlib__Order__PiLex______macroRules__Pi__term_u03a0_u2097___x2c____1___closed__22_value;
static const lean_ctor_object lp_mathlib_Pi___aux__Mathlib__Order__PiLex______macroRules__Pi__term_u03a0_u2097___x2c____1___closed__23_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Pi___aux__Mathlib__Order__PiLex______macroRules__Pi__term_u03a0_u2097___x2c____1___closed__22_value),LEAN_SCALAR_PTR_LITERAL(24, 58, 49, 223, 146, 207, 197, 136)}};
static const lean_object* lp_mathlib_Pi___aux__Mathlib__Order__PiLex______macroRules__Pi__term_u03a0_u2097___x2c____1___closed__23 = (const lean_object*)&lp_mathlib_Pi___aux__Mathlib__Order__PiLex______macroRules__Pi__term_u03a0_u2097___x2c____1___closed__23_value;
static const lean_string_object lp_mathlib_Pi___aux__Mathlib__Order__PiLex______macroRules__Pi__term_u03a0_u2097___x2c____1___closed__24_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "paren"};
static const lean_object* lp_mathlib_Pi___aux__Mathlib__Order__PiLex______macroRules__Pi__term_u03a0_u2097___x2c____1___closed__24 = (const lean_object*)&lp_mathlib_Pi___aux__Mathlib__Order__PiLex______macroRules__Pi__term_u03a0_u2097___x2c____1___closed__24_value;
static const lean_ctor_object lp_mathlib_Pi___aux__Mathlib__Order__PiLex______macroRules__Pi__term_u03a0_u2097___x2c____1___closed__25_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Pi___aux__Mathlib__Order__PiLex______macroRules__Pi__term_u03a0_u2097___x2c____1___closed__10_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib_Pi___aux__Mathlib__Order__PiLex______macroRules__Pi__term_u03a0_u2097___x2c____1___closed__25_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Pi___aux__Mathlib__Order__PiLex______macroRules__Pi__term_u03a0_u2097___x2c____1___closed__25_value_aux_0),((lean_object*)&lp_mathlib_Pi___aux__Mathlib__Order__PiLex______macroRules__Pi__term_u03a0_u2097___x2c____1___closed__11_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib_Pi___aux__Mathlib__Order__PiLex______macroRules__Pi__term_u03a0_u2097___x2c____1___closed__25_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Pi___aux__Mathlib__Order__PiLex______macroRules__Pi__term_u03a0_u2097___x2c____1___closed__25_value_aux_1),((lean_object*)&lp_mathlib_Pi___aux__Mathlib__Order__PiLex______macroRules__Pi__term_u03a0_u2097___x2c____1___closed__12_value),LEAN_SCALAR_PTR_LITERAL(75, 170, 162, 138, 136, 204, 251, 229)}};
static const lean_ctor_object lp_mathlib_Pi___aux__Mathlib__Order__PiLex______macroRules__Pi__term_u03a0_u2097___x2c____1___closed__25_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Pi___aux__Mathlib__Order__PiLex______macroRules__Pi__term_u03a0_u2097___x2c____1___closed__25_value_aux_2),((lean_object*)&lp_mathlib_Pi___aux__Mathlib__Order__PiLex______macroRules__Pi__term_u03a0_u2097___x2c____1___closed__24_value),LEAN_SCALAR_PTR_LITERAL(124, 9, 161, 194, 227, 100, 20, 110)}};
static const lean_object* lp_mathlib_Pi___aux__Mathlib__Order__PiLex______macroRules__Pi__term_u03a0_u2097___x2c____1___closed__25 = (const lean_object*)&lp_mathlib_Pi___aux__Mathlib__Order__PiLex______macroRules__Pi__term_u03a0_u2097___x2c____1___closed__25_value;
static const lean_string_object lp_mathlib_Pi___aux__Mathlib__Order__PiLex______macroRules__Pi__term_u03a0_u2097___x2c____1___closed__26_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 15, .m_capacity = 15, .m_length = 14, .m_data = "hygienicLParen"};
static const lean_object* lp_mathlib_Pi___aux__Mathlib__Order__PiLex______macroRules__Pi__term_u03a0_u2097___x2c____1___closed__26 = (const lean_object*)&lp_mathlib_Pi___aux__Mathlib__Order__PiLex______macroRules__Pi__term_u03a0_u2097___x2c____1___closed__26_value;
static const lean_ctor_object lp_mathlib_Pi___aux__Mathlib__Order__PiLex______macroRules__Pi__term_u03a0_u2097___x2c____1___closed__27_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Pi___aux__Mathlib__Order__PiLex______macroRules__Pi__term_u03a0_u2097___x2c____1___closed__10_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib_Pi___aux__Mathlib__Order__PiLex______macroRules__Pi__term_u03a0_u2097___x2c____1___closed__27_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Pi___aux__Mathlib__Order__PiLex______macroRules__Pi__term_u03a0_u2097___x2c____1___closed__27_value_aux_0),((lean_object*)&lp_mathlib_Pi___aux__Mathlib__Order__PiLex______macroRules__Pi__term_u03a0_u2097___x2c____1___closed__11_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib_Pi___aux__Mathlib__Order__PiLex______macroRules__Pi__term_u03a0_u2097___x2c____1___closed__27_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Pi___aux__Mathlib__Order__PiLex______macroRules__Pi__term_u03a0_u2097___x2c____1___closed__27_value_aux_1),((lean_object*)&lp_mathlib_Pi___aux__Mathlib__Order__PiLex______macroRules__Pi__term_u03a0_u2097___x2c____1___closed__12_value),LEAN_SCALAR_PTR_LITERAL(75, 170, 162, 138, 136, 204, 251, 229)}};
static const lean_ctor_object lp_mathlib_Pi___aux__Mathlib__Order__PiLex______macroRules__Pi__term_u03a0_u2097___x2c____1___closed__27_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Pi___aux__Mathlib__Order__PiLex______macroRules__Pi__term_u03a0_u2097___x2c____1___closed__27_value_aux_2),((lean_object*)&lp_mathlib_Pi___aux__Mathlib__Order__PiLex______macroRules__Pi__term_u03a0_u2097___x2c____1___closed__26_value),LEAN_SCALAR_PTR_LITERAL(41, 104, 206, 51, 21, 254, 100, 101)}};
static const lean_object* lp_mathlib_Pi___aux__Mathlib__Order__PiLex______macroRules__Pi__term_u03a0_u2097___x2c____1___closed__27 = (const lean_object*)&lp_mathlib_Pi___aux__Mathlib__Order__PiLex______macroRules__Pi__term_u03a0_u2097___x2c____1___closed__27_value;
static const lean_string_object lp_mathlib_Pi___aux__Mathlib__Order__PiLex______macroRules__Pi__term_u03a0_u2097___x2c____1___closed__28_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 12, .m_capacity = 12, .m_length = 11, .m_data = "hygieneInfo"};
static const lean_object* lp_mathlib_Pi___aux__Mathlib__Order__PiLex______macroRules__Pi__term_u03a0_u2097___x2c____1___closed__28 = (const lean_object*)&lp_mathlib_Pi___aux__Mathlib__Order__PiLex______macroRules__Pi__term_u03a0_u2097___x2c____1___closed__28_value;
static const lean_ctor_object lp_mathlib_Pi___aux__Mathlib__Order__PiLex______macroRules__Pi__term_u03a0_u2097___x2c____1___closed__29_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Pi___aux__Mathlib__Order__PiLex______macroRules__Pi__term_u03a0_u2097___x2c____1___closed__28_value),LEAN_SCALAR_PTR_LITERAL(27, 64, 36, 144, 170, 151, 255, 136)}};
static const lean_object* lp_mathlib_Pi___aux__Mathlib__Order__PiLex______macroRules__Pi__term_u03a0_u2097___x2c____1___closed__29 = (const lean_object*)&lp_mathlib_Pi___aux__Mathlib__Order__PiLex______macroRules__Pi__term_u03a0_u2097___x2c____1___closed__29_value;
static const lean_string_object lp_mathlib_Pi___aux__Mathlib__Order__PiLex______macroRules__Pi__term_u03a0_u2097___x2c____1___closed__30_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 1, .m_capacity = 1, .m_length = 0, .m_data = ""};
static const lean_object* lp_mathlib_Pi___aux__Mathlib__Order__PiLex______macroRules__Pi__term_u03a0_u2097___x2c____1___closed__30 = (const lean_object*)&lp_mathlib_Pi___aux__Mathlib__Order__PiLex______macroRules__Pi__term_u03a0_u2097___x2c____1___closed__30_value;
static lean_once_cell_t lp_mathlib_Pi___aux__Mathlib__Order__PiLex______macroRules__Pi__term_u03a0_u2097___x2c____1___closed__31_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Pi___aux__Mathlib__Order__PiLex______macroRules__Pi__term_u03a0_u2097___x2c____1___closed__31;
static const lean_ctor_object lp_mathlib_Pi___aux__Mathlib__Order__PiLex______macroRules__Pi__term_u03a0_u2097___x2c____1___closed__32_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Pi_term_u03a0_u2097___x2c___00__closed__0_value),LEAN_SCALAR_PTR_LITERAL(236, 177, 129, 149, 110, 155, 170, 220)}};
static const lean_object* lp_mathlib_Pi___aux__Mathlib__Order__PiLex______macroRules__Pi__term_u03a0_u2097___x2c____1___closed__32 = (const lean_object*)&lp_mathlib_Pi___aux__Mathlib__Order__PiLex______macroRules__Pi__term_u03a0_u2097___x2c____1___closed__32_value;
static const lean_ctor_object lp_mathlib_Pi___aux__Mathlib__Order__PiLex______macroRules__Pi__term_u03a0_u2097___x2c____1___closed__33_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_Pi___aux__Mathlib__Order__PiLex______macroRules__Pi__term_u03a0_u2097___x2c____1___closed__32_value)}};
static const lean_object* lp_mathlib_Pi___aux__Mathlib__Order__PiLex______macroRules__Pi__term_u03a0_u2097___x2c____1___closed__33 = (const lean_object*)&lp_mathlib_Pi___aux__Mathlib__Order__PiLex______macroRules__Pi__term_u03a0_u2097___x2c____1___closed__33_value;
static const lean_ctor_object lp_mathlib_Pi___aux__Mathlib__Order__PiLex______macroRules__Pi__term_u03a0_u2097___x2c____1___closed__34_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Pi___aux__Mathlib__Order__PiLex______macroRules__Pi__term_u03a0_u2097___x2c____1___closed__33_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_Pi___aux__Mathlib__Order__PiLex______macroRules__Pi__term_u03a0_u2097___x2c____1___closed__34 = (const lean_object*)&lp_mathlib_Pi___aux__Mathlib__Order__PiLex______macroRules__Pi__term_u03a0_u2097___x2c____1___closed__34_value;
static const lean_string_object lp_mathlib_Pi___aux__Mathlib__Order__PiLex______macroRules__Pi__term_u03a0_u2097___x2c____1___closed__35_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "forall"};
static const lean_object* lp_mathlib_Pi___aux__Mathlib__Order__PiLex______macroRules__Pi__term_u03a0_u2097___x2c____1___closed__35 = (const lean_object*)&lp_mathlib_Pi___aux__Mathlib__Order__PiLex______macroRules__Pi__term_u03a0_u2097___x2c____1___closed__35_value;
static const lean_ctor_object lp_mathlib_Pi___aux__Mathlib__Order__PiLex______macroRules__Pi__term_u03a0_u2097___x2c____1___closed__36_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Pi___aux__Mathlib__Order__PiLex______macroRules__Pi__term_u03a0_u2097___x2c____1___closed__10_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib_Pi___aux__Mathlib__Order__PiLex______macroRules__Pi__term_u03a0_u2097___x2c____1___closed__36_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Pi___aux__Mathlib__Order__PiLex______macroRules__Pi__term_u03a0_u2097___x2c____1___closed__36_value_aux_0),((lean_object*)&lp_mathlib_Pi___aux__Mathlib__Order__PiLex______macroRules__Pi__term_u03a0_u2097___x2c____1___closed__11_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib_Pi___aux__Mathlib__Order__PiLex______macroRules__Pi__term_u03a0_u2097___x2c____1___closed__36_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Pi___aux__Mathlib__Order__PiLex______macroRules__Pi__term_u03a0_u2097___x2c____1___closed__36_value_aux_1),((lean_object*)&lp_mathlib_Pi___aux__Mathlib__Order__PiLex______macroRules__Pi__term_u03a0_u2097___x2c____1___closed__12_value),LEAN_SCALAR_PTR_LITERAL(75, 170, 162, 138, 136, 204, 251, 229)}};
static const lean_ctor_object lp_mathlib_Pi___aux__Mathlib__Order__PiLex______macroRules__Pi__term_u03a0_u2097___x2c____1___closed__36_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Pi___aux__Mathlib__Order__PiLex______macroRules__Pi__term_u03a0_u2097___x2c____1___closed__36_value_aux_2),((lean_object*)&lp_mathlib_Pi___aux__Mathlib__Order__PiLex______macroRules__Pi__term_u03a0_u2097___x2c____1___closed__35_value),LEAN_SCALAR_PTR_LITERAL(195, 142, 115, 15, 55, 103, 31, 115)}};
static const lean_object* lp_mathlib_Pi___aux__Mathlib__Order__PiLex______macroRules__Pi__term_u03a0_u2097___x2c____1___closed__36 = (const lean_object*)&lp_mathlib_Pi___aux__Mathlib__Order__PiLex______macroRules__Pi__term_u03a0_u2097___x2c____1___closed__36_value;
static const lean_string_object lp_mathlib_Pi___aux__Mathlib__Order__PiLex______macroRules__Pi__term_u03a0_u2097___x2c____1___closed__37_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 1, .m_data = "∀"};
static const lean_object* lp_mathlib_Pi___aux__Mathlib__Order__PiLex______macroRules__Pi__term_u03a0_u2097___x2c____1___closed__37 = (const lean_object*)&lp_mathlib_Pi___aux__Mathlib__Order__PiLex______macroRules__Pi__term_u03a0_u2097___x2c____1___closed__37_value;
static const lean_string_object lp_mathlib_Pi___aux__Mathlib__Order__PiLex______macroRules__Pi__term_u03a0_u2097___x2c____1___closed__38_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = "i"};
static const lean_object* lp_mathlib_Pi___aux__Mathlib__Order__PiLex______macroRules__Pi__term_u03a0_u2097___x2c____1___closed__38 = (const lean_object*)&lp_mathlib_Pi___aux__Mathlib__Order__PiLex______macroRules__Pi__term_u03a0_u2097___x2c____1___closed__38_value;
static lean_once_cell_t lp_mathlib_Pi___aux__Mathlib__Order__PiLex______macroRules__Pi__term_u03a0_u2097___x2c____1___closed__39_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Pi___aux__Mathlib__Order__PiLex______macroRules__Pi__term_u03a0_u2097___x2c____1___closed__39;
static const lean_ctor_object lp_mathlib_Pi___aux__Mathlib__Order__PiLex______macroRules__Pi__term_u03a0_u2097___x2c____1___closed__40_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Pi___aux__Mathlib__Order__PiLex______macroRules__Pi__term_u03a0_u2097___x2c____1___closed__38_value),LEAN_SCALAR_PTR_LITERAL(14, 215, 4, 153, 96, 18, 167, 14)}};
static const lean_object* lp_mathlib_Pi___aux__Mathlib__Order__PiLex______macroRules__Pi__term_u03a0_u2097___x2c____1___closed__40 = (const lean_object*)&lp_mathlib_Pi___aux__Mathlib__Order__PiLex______macroRules__Pi__term_u03a0_u2097___x2c____1___closed__40_value;
static lean_once_cell_t lp_mathlib_Pi___aux__Mathlib__Order__PiLex______macroRules__Pi__term_u03a0_u2097___x2c____1___closed__41_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Pi___aux__Mathlib__Order__PiLex______macroRules__Pi__term_u03a0_u2097___x2c____1___closed__41;
static const lean_string_object lp_mathlib_Pi___aux__Mathlib__Order__PiLex______macroRules__Pi__term_u03a0_u2097___x2c____1___closed__42_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = ","};
static const lean_object* lp_mathlib_Pi___aux__Mathlib__Order__PiLex______macroRules__Pi__term_u03a0_u2097___x2c____1___closed__42 = (const lean_object*)&lp_mathlib_Pi___aux__Mathlib__Order__PiLex______macroRules__Pi__term_u03a0_u2097___x2c____1___closed__42_value;
static const lean_string_object lp_mathlib_Pi___aux__Mathlib__Order__PiLex______macroRules__Pi__term_u03a0_u2097___x2c____1___closed__43_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = ")"};
static const lean_object* lp_mathlib_Pi___aux__Mathlib__Order__PiLex______macroRules__Pi__term_u03a0_u2097___x2c____1___closed__43 = (const lean_object*)&lp_mathlib_Pi___aux__Mathlib__Order__PiLex______macroRules__Pi__term_u03a0_u2097___x2c____1___closed__43_value;
LEAN_EXPORT lean_object* lp_mathlib_Pi___aux__Mathlib__Order__PiLex______macroRules__Pi__term_u03a0_u2097___x2c____1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Pi___aux__Mathlib__Order__PiLex______macroRules__Pi__term_u03a0_u2097___x2c____1___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Pi_instLTLexForall(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Pi_instLTLexForall___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Pi_instLTColexForall(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Pi_instLTColexForall___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_ctor_object lp_mathlib_Pi_instPartialOrderLexForallOfLinearOrder___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_Pi_instPartialOrderLexForallOfLinearOrder___closed__0 = (const lean_object*)&lp_mathlib_Pi_instPartialOrderLexForallOfLinearOrder___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_Pi_instPartialOrderLexForallOfLinearOrder(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Pi_instPartialOrderLexForallOfLinearOrder___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Pi_instPartialOrderColexForallOfLinearOrder(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Pi_instPartialOrderColexForallOfLinearOrder___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Pi_instOrderBotLexForallOfWellFoundedLT___redArg___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Pi_instOrderBotLexForallOfWellFoundedLT___redArg___lam__1(lean_object*, lean_object*, lean_object*);
static lean_once_cell_t lp_mathlib_Pi_instOrderBotLexForallOfWellFoundedLT___redArg___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Pi_instOrderBotLexForallOfWellFoundedLT___redArg___closed__0;
LEAN_EXPORT lean_object* lp_mathlib_Pi_instOrderBotLexForallOfWellFoundedLT___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Pi_instOrderBotLexForallOfWellFoundedLT(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Pi_instOrderBotLexForallOfWellFoundedLT___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Pi_instOrderBotColexForallOfWellFoundedGT___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Pi_instOrderBotColexForallOfWellFoundedGT(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Pi_instOrderBotColexForallOfWellFoundedGT___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Pi_instOrderTopLexForallOfWellFoundedLT___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Pi_instOrderTopLexForallOfWellFoundedLT(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Pi_instOrderTopLexForallOfWellFoundedLT___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Pi_instOrderTopColexForallOfWellFoundedGT___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Pi_instOrderTopColexForallOfWellFoundedGT(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Pi_instOrderTopColexForallOfWellFoundedGT___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Pi_instBoundedOrderLexForallOfWellFoundedLT___redArg___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Pi_instBoundedOrderLexForallOfWellFoundedLT___redArg___lam__1(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Pi_instBoundedOrderLexForallOfWellFoundedLT___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Pi_instBoundedOrderLexForallOfWellFoundedLT(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Pi_instBoundedOrderLexForallOfWellFoundedLT___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Pi_instBoundedOrderColexForallOfWellFoundedGT___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Pi_instBoundedOrderColexForallOfWellFoundedGT(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Pi_instBoundedOrderColexForallOfWellFoundedGT___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_object* _init_lp_mathlib_Pi_term_u03a0_u2097___x2c___00__closed__7(void){
_start:
{
lean_object* v___x_12_; lean_object* v___x_13_; lean_object* v___x_14_; lean_object* v___x_15_; 
v___x_12_ = lp_batteries_Batteries_ExtendedBinder_extBinders;
v___x_13_ = ((lean_object*)(lp_mathlib_Pi_term_u03a0_u2097___x2c___00__closed__6));
v___x_14_ = ((lean_object*)(lp_mathlib_Pi_term_u03a0_u2097___x2c___00__closed__4));
v___x_15_ = lean_alloc_ctor(2, 3, 0);
lean_ctor_set(v___x_15_, 0, v___x_14_);
lean_ctor_set(v___x_15_, 1, v___x_13_);
lean_ctor_set(v___x_15_, 2, v___x_12_);
return v___x_15_;
}
}
static lean_object* _init_lp_mathlib_Pi_term_u03a0_u2097___x2c___00__closed__10(void){
_start:
{
lean_object* v___x_19_; lean_object* v___x_20_; lean_object* v___x_21_; lean_object* v___x_22_; 
v___x_19_ = ((lean_object*)(lp_mathlib_Pi_term_u03a0_u2097___x2c___00__closed__9));
v___x_20_ = lean_obj_once(&lp_mathlib_Pi_term_u03a0_u2097___x2c___00__closed__7, &lp_mathlib_Pi_term_u03a0_u2097___x2c___00__closed__7_once, _init_lp_mathlib_Pi_term_u03a0_u2097___x2c___00__closed__7);
v___x_21_ = ((lean_object*)(lp_mathlib_Pi_term_u03a0_u2097___x2c___00__closed__4));
v___x_22_ = lean_alloc_ctor(2, 3, 0);
lean_ctor_set(v___x_22_, 0, v___x_21_);
lean_ctor_set(v___x_22_, 1, v___x_20_);
lean_ctor_set(v___x_22_, 2, v___x_19_);
return v___x_22_;
}
}
static lean_object* _init_lp_mathlib_Pi_term_u03a0_u2097___x2c___00__closed__14(void){
_start:
{
lean_object* v___x_29_; lean_object* v___x_30_; lean_object* v___x_31_; lean_object* v___x_32_; 
v___x_29_ = ((lean_object*)(lp_mathlib_Pi_term_u03a0_u2097___x2c___00__closed__13));
v___x_30_ = lean_obj_once(&lp_mathlib_Pi_term_u03a0_u2097___x2c___00__closed__10, &lp_mathlib_Pi_term_u03a0_u2097___x2c___00__closed__10_once, _init_lp_mathlib_Pi_term_u03a0_u2097___x2c___00__closed__10);
v___x_31_ = ((lean_object*)(lp_mathlib_Pi_term_u03a0_u2097___x2c___00__closed__4));
v___x_32_ = lean_alloc_ctor(2, 3, 0);
lean_ctor_set(v___x_32_, 0, v___x_31_);
lean_ctor_set(v___x_32_, 1, v___x_30_);
lean_ctor_set(v___x_32_, 2, v___x_29_);
return v___x_32_;
}
}
static lean_object* _init_lp_mathlib_Pi_term_u03a0_u2097___x2c___00__closed__15(void){
_start:
{
lean_object* v___x_33_; lean_object* v___x_34_; lean_object* v___x_35_; lean_object* v___x_36_; 
v___x_33_ = lean_obj_once(&lp_mathlib_Pi_term_u03a0_u2097___x2c___00__closed__14, &lp_mathlib_Pi_term_u03a0_u2097___x2c___00__closed__14_once, _init_lp_mathlib_Pi_term_u03a0_u2097___x2c___00__closed__14);
v___x_34_ = lean_unsigned_to_nat(1022u);
v___x_35_ = ((lean_object*)(lp_mathlib_Pi_term_u03a0_u2097___x2c___00__closed__2));
v___x_36_ = lean_alloc_ctor(3, 3, 0);
lean_ctor_set(v___x_36_, 0, v___x_35_);
lean_ctor_set(v___x_36_, 1, v___x_34_);
lean_ctor_set(v___x_36_, 2, v___x_33_);
return v___x_36_;
}
}
static lean_object* _init_lp_mathlib_Pi_term_u03a0_u2097___x2c__(void){
_start:
{
lean_object* v___x_37_; 
v___x_37_ = lean_obj_once(&lp_mathlib_Pi_term_u03a0_u2097___x2c___00__closed__15, &lp_mathlib_Pi_term_u03a0_u2097___x2c___00__closed__15_once, _init_lp_mathlib_Pi_term_u03a0_u2097___x2c___00__closed__15);
return v___x_37_;
}
}
static lean_object* _init_lp_mathlib_Pi___aux__Mathlib__Order__PiLex______macroRules__Pi__term_u03a0_u2097___x2c____1___closed__7(void){
_start:
{
lean_object* v___x_48_; lean_object* v___x_49_; 
v___x_48_ = ((lean_object*)(lp_mathlib_Pi___aux__Mathlib__Order__PiLex______macroRules__Pi__term_u03a0_u2097___x2c____1___closed__6));
v___x_49_ = l_String_toRawSubstring_x27(v___x_48_);
return v___x_49_;
}
}
static lean_object* _init_lp_mathlib_Pi___aux__Mathlib__Order__PiLex______macroRules__Pi__term_u03a0_u2097___x2c____1___closed__16(void){
_start:
{
lean_object* v___x_63_; lean_object* v___x_64_; 
v___x_63_ = ((lean_object*)(lp_mathlib_Pi___aux__Mathlib__Order__PiLex______macroRules__Pi__term_u03a0_u2097___x2c____1___closed__15));
v___x_64_ = l_String_toRawSubstring_x27(v___x_63_);
return v___x_64_;
}
}
static lean_object* _init_lp_mathlib_Pi___aux__Mathlib__Order__PiLex______macroRules__Pi__term_u03a0_u2097___x2c____1___closed__31(void){
_start:
{
lean_object* v___x_97_; lean_object* v___x_98_; 
v___x_97_ = ((lean_object*)(lp_mathlib_Pi___aux__Mathlib__Order__PiLex______macroRules__Pi__term_u03a0_u2097___x2c____1___closed__30));
v___x_98_ = l_String_toRawSubstring_x27(v___x_97_);
return v___x_98_;
}
}
static lean_object* _init_lp_mathlib_Pi___aux__Mathlib__Order__PiLex______macroRules__Pi__term_u03a0_u2097___x2c____1___closed__39(void){
_start:
{
lean_object* v___x_114_; lean_object* v___x_115_; 
v___x_114_ = ((lean_object*)(lp_mathlib_Pi___aux__Mathlib__Order__PiLex______macroRules__Pi__term_u03a0_u2097___x2c____1___closed__38));
v___x_115_ = l_String_toRawSubstring_x27(v___x_114_);
return v___x_115_;
}
}
static lean_object* _init_lp_mathlib_Pi___aux__Mathlib__Order__PiLex______macroRules__Pi__term_u03a0_u2097___x2c____1___closed__41(void){
_start:
{
lean_object* v___x_118_; 
v___x_118_ = l_Array_mkArray0(lean_box(0));
return v___x_118_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Pi___aux__Mathlib__Order__PiLex______macroRules__Pi__term_u03a0_u2097___x2c____1(lean_object* v_x_121_, lean_object* v_a_122_, lean_object* v_a_123_){
_start:
{
lean_object* v___x_124_; uint8_t v___x_125_; 
v___x_124_ = ((lean_object*)(lp_mathlib_Pi_term_u03a0_u2097___x2c___00__closed__2));
lean_inc(v_x_121_);
v___x_125_ = l_Lean_Syntax_isOfKind(v_x_121_, v___x_124_);
if (v___x_125_ == 0)
{
lean_object* v___x_126_; lean_object* v___x_127_; 
lean_dec(v_x_121_);
v___x_126_ = lean_box(1);
v___x_127_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_127_, 0, v___x_126_);
lean_ctor_set(v___x_127_, 1, v_a_123_);
return v___x_127_;
}
else
{
lean_object* v_quotContext_128_; lean_object* v_currMacroScope_129_; lean_object* v_ref_130_; lean_object* v___x_131_; lean_object* v___x_132_; lean_object* v___x_133_; lean_object* v___x_134_; uint8_t v___x_135_; lean_object* v___x_136_; lean_object* v___x_137_; lean_object* v___x_138_; lean_object* v___x_139_; lean_object* v___x_140_; lean_object* v___x_141_; lean_object* v___x_142_; lean_object* v___x_143_; lean_object* v___x_144_; lean_object* v___x_145_; lean_object* v___x_146_; lean_object* v___x_147_; lean_object* v___x_148_; lean_object* v___x_149_; lean_object* v___x_150_; lean_object* v___x_151_; lean_object* v___x_152_; lean_object* v___x_153_; lean_object* v___x_154_; lean_object* v___x_155_; lean_object* v___x_156_; lean_object* v___x_157_; lean_object* v___x_158_; lean_object* v___x_159_; lean_object* v___x_160_; lean_object* v___x_161_; lean_object* v___x_162_; lean_object* v___x_163_; lean_object* v___x_164_; lean_object* v___x_165_; lean_object* v___x_166_; lean_object* v___x_167_; lean_object* v___x_168_; lean_object* v___x_169_; lean_object* v___x_170_; lean_object* v___x_171_; lean_object* v___x_172_; lean_object* v___x_173_; lean_object* v___x_174_; lean_object* v___x_175_; lean_object* v___x_176_; lean_object* v___x_177_; lean_object* v___x_178_; lean_object* v___x_179_; lean_object* v___x_180_; lean_object* v___x_181_; lean_object* v___x_182_; lean_object* v___x_183_; lean_object* v___x_184_; lean_object* v___x_185_; lean_object* v___x_186_; lean_object* v___x_187_; lean_object* v___x_188_; lean_object* v___x_189_; lean_object* v___x_190_; lean_object* v___x_191_; lean_object* v___x_192_; lean_object* v___x_193_; lean_object* v___x_194_; lean_object* v___x_195_; lean_object* v___x_196_; lean_object* v___x_197_; 
v_quotContext_128_ = lean_ctor_get(v_a_122_, 1);
v_currMacroScope_129_ = lean_ctor_get(v_a_122_, 2);
v_ref_130_ = lean_ctor_get(v_a_122_, 5);
v___x_131_ = lean_unsigned_to_nat(1u);
v___x_132_ = l_Lean_Syntax_getArg(v_x_121_, v___x_131_);
v___x_133_ = lean_unsigned_to_nat(3u);
v___x_134_ = l_Lean_Syntax_getArg(v_x_121_, v___x_133_);
lean_dec(v_x_121_);
v___x_135_ = 0;
v___x_136_ = l_Lean_SourceInfo_fromRef(v_ref_130_, v___x_135_);
v___x_137_ = ((lean_object*)(lp_mathlib_Pi___aux__Mathlib__Order__PiLex______macroRules__Pi__term_u03a0_u2097___x2c____1___closed__3));
v___x_138_ = ((lean_object*)(lp_mathlib_Pi___aux__Mathlib__Order__PiLex______macroRules__Pi__term_u03a0_u2097___x2c____1___closed__4));
lean_inc_n(v___x_136_, 19);
v___x_139_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_139_, 0, v___x_136_);
lean_ctor_set(v___x_139_, 1, v___x_138_);
v___x_140_ = ((lean_object*)(lp_mathlib_Pi___aux__Mathlib__Order__PiLex______macroRules__Pi__term_u03a0_u2097___x2c____1___closed__5));
v___x_141_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_141_, 0, v___x_136_);
lean_ctor_set(v___x_141_, 1, v___x_140_);
v___x_142_ = lean_obj_once(&lp_mathlib_Pi___aux__Mathlib__Order__PiLex______macroRules__Pi__term_u03a0_u2097___x2c____1___closed__7, &lp_mathlib_Pi___aux__Mathlib__Order__PiLex______macroRules__Pi__term_u03a0_u2097___x2c____1___closed__7_once, _init_lp_mathlib_Pi___aux__Mathlib__Order__PiLex______macroRules__Pi__term_u03a0_u2097___x2c____1___closed__7);
v___x_143_ = ((lean_object*)(lp_mathlib_Pi___aux__Mathlib__Order__PiLex______macroRules__Pi__term_u03a0_u2097___x2c____1___closed__8));
lean_inc_n(v_currMacroScope_129_, 4);
lean_inc_n(v_quotContext_128_, 4);
v___x_144_ = l_Lean_addMacroScope(v_quotContext_128_, v___x_143_, v_currMacroScope_129_);
v___x_145_ = lean_box(0);
v___x_146_ = lean_alloc_ctor(3, 4, 0);
lean_ctor_set(v___x_146_, 0, v___x_136_);
lean_ctor_set(v___x_146_, 1, v___x_142_);
lean_ctor_set(v___x_146_, 2, v___x_144_);
lean_ctor_set(v___x_146_, 3, v___x_145_);
v___x_147_ = ((lean_object*)(lp_mathlib_Pi___aux__Mathlib__Order__PiLex______macroRules__Pi__term_u03a0_u2097___x2c____1___closed__9));
v___x_148_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_148_, 0, v___x_136_);
lean_ctor_set(v___x_148_, 1, v___x_147_);
v___x_149_ = ((lean_object*)(lp_mathlib_Pi___aux__Mathlib__Order__PiLex______macroRules__Pi__term_u03a0_u2097___x2c____1___closed__14));
v___x_150_ = lean_obj_once(&lp_mathlib_Pi___aux__Mathlib__Order__PiLex______macroRules__Pi__term_u03a0_u2097___x2c____1___closed__16, &lp_mathlib_Pi___aux__Mathlib__Order__PiLex______macroRules__Pi__term_u03a0_u2097___x2c____1___closed__16_once, _init_lp_mathlib_Pi___aux__Mathlib__Order__PiLex______macroRules__Pi__term_u03a0_u2097___x2c____1___closed__16);
v___x_151_ = ((lean_object*)(lp_mathlib_Pi___aux__Mathlib__Order__PiLex______macroRules__Pi__term_u03a0_u2097___x2c____1___closed__17));
v___x_152_ = l_Lean_addMacroScope(v_quotContext_128_, v___x_151_, v_currMacroScope_129_);
v___x_153_ = ((lean_object*)(lp_mathlib_Pi___aux__Mathlib__Order__PiLex______macroRules__Pi__term_u03a0_u2097___x2c____1___closed__21));
v___x_154_ = lean_alloc_ctor(3, 4, 0);
lean_ctor_set(v___x_154_, 0, v___x_136_);
lean_ctor_set(v___x_154_, 1, v___x_150_);
lean_ctor_set(v___x_154_, 2, v___x_152_);
lean_ctor_set(v___x_154_, 3, v___x_153_);
v___x_155_ = ((lean_object*)(lp_mathlib_Pi___aux__Mathlib__Order__PiLex______macroRules__Pi__term_u03a0_u2097___x2c____1___closed__23));
v___x_156_ = ((lean_object*)(lp_mathlib_Pi___aux__Mathlib__Order__PiLex______macroRules__Pi__term_u03a0_u2097___x2c____1___closed__25));
v___x_157_ = ((lean_object*)(lp_mathlib_Pi___aux__Mathlib__Order__PiLex______macroRules__Pi__term_u03a0_u2097___x2c____1___closed__27));
v___x_158_ = ((lean_object*)(lp_mathlib_Pi___aux__Mathlib__Order__PiLex______macroRules__Pi__term_u03a0_u2097___x2c____1___closed__29));
v___x_159_ = lean_obj_once(&lp_mathlib_Pi___aux__Mathlib__Order__PiLex______macroRules__Pi__term_u03a0_u2097___x2c____1___closed__31, &lp_mathlib_Pi___aux__Mathlib__Order__PiLex______macroRules__Pi__term_u03a0_u2097___x2c____1___closed__31_once, _init_lp_mathlib_Pi___aux__Mathlib__Order__PiLex______macroRules__Pi__term_u03a0_u2097___x2c____1___closed__31);
v___x_160_ = lean_box(0);
v___x_161_ = l_Lean_addMacroScope(v_quotContext_128_, v___x_160_, v_currMacroScope_129_);
v___x_162_ = ((lean_object*)(lp_mathlib_Pi___aux__Mathlib__Order__PiLex______macroRules__Pi__term_u03a0_u2097___x2c____1___closed__34));
v___x_163_ = lean_alloc_ctor(3, 4, 0);
lean_ctor_set(v___x_163_, 0, v___x_136_);
lean_ctor_set(v___x_163_, 1, v___x_159_);
lean_ctor_set(v___x_163_, 2, v___x_161_);
lean_ctor_set(v___x_163_, 3, v___x_162_);
v___x_164_ = l_Lean_Syntax_node1(v___x_136_, v___x_158_, v___x_163_);
lean_inc_ref(v___x_141_);
v___x_165_ = l_Lean_Syntax_node2(v___x_136_, v___x_157_, v___x_141_, v___x_164_);
v___x_166_ = ((lean_object*)(lp_mathlib_Pi___aux__Mathlib__Order__PiLex______macroRules__Pi__term_u03a0_u2097___x2c____1___closed__36));
v___x_167_ = ((lean_object*)(lp_mathlib_Pi___aux__Mathlib__Order__PiLex______macroRules__Pi__term_u03a0_u2097___x2c____1___closed__37));
v___x_168_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_168_, 0, v___x_136_);
lean_ctor_set(v___x_168_, 1, v___x_167_);
v___x_169_ = lean_obj_once(&lp_mathlib_Pi___aux__Mathlib__Order__PiLex______macroRules__Pi__term_u03a0_u2097___x2c____1___closed__39, &lp_mathlib_Pi___aux__Mathlib__Order__PiLex______macroRules__Pi__term_u03a0_u2097___x2c____1___closed__39_once, _init_lp_mathlib_Pi___aux__Mathlib__Order__PiLex______macroRules__Pi__term_u03a0_u2097___x2c____1___closed__39);
v___x_170_ = ((lean_object*)(lp_mathlib_Pi___aux__Mathlib__Order__PiLex______macroRules__Pi__term_u03a0_u2097___x2c____1___closed__40));
v___x_171_ = l_Lean_addMacroScope(v_quotContext_128_, v___x_170_, v_currMacroScope_129_);
v___x_172_ = lean_alloc_ctor(3, 4, 0);
lean_ctor_set(v___x_172_, 0, v___x_136_);
lean_ctor_set(v___x_172_, 1, v___x_169_);
lean_ctor_set(v___x_172_, 2, v___x_171_);
lean_ctor_set(v___x_172_, 3, v___x_145_);
v___x_173_ = l_Lean_Syntax_node1(v___x_136_, v___x_155_, v___x_172_);
v___x_174_ = lean_obj_once(&lp_mathlib_Pi___aux__Mathlib__Order__PiLex______macroRules__Pi__term_u03a0_u2097___x2c____1___closed__41, &lp_mathlib_Pi___aux__Mathlib__Order__PiLex______macroRules__Pi__term_u03a0_u2097___x2c____1___closed__41_once, _init_lp_mathlib_Pi___aux__Mathlib__Order__PiLex______macroRules__Pi__term_u03a0_u2097___x2c____1___closed__41);
v___x_175_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_175_, 0, v___x_136_);
lean_ctor_set(v___x_175_, 1, v___x_155_);
lean_ctor_set(v___x_175_, 2, v___x_174_);
v___x_176_ = ((lean_object*)(lp_mathlib_Pi___aux__Mathlib__Order__PiLex______macroRules__Pi__term_u03a0_u2097___x2c____1___closed__42));
v___x_177_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_177_, 0, v___x_136_);
lean_ctor_set(v___x_177_, 1, v___x_176_);
lean_inc(v___x_173_);
lean_inc_ref(v___x_146_);
v___x_178_ = l_Lean_Syntax_node2(v___x_136_, v___x_149_, v___x_146_, v___x_173_);
lean_inc_ref(v___x_177_);
v___x_179_ = l_Lean_Syntax_node5(v___x_136_, v___x_166_, v___x_168_, v___x_173_, v___x_175_, v___x_177_, v___x_178_);
v___x_180_ = ((lean_object*)(lp_mathlib_Pi___aux__Mathlib__Order__PiLex______macroRules__Pi__term_u03a0_u2097___x2c____1___closed__43));
v___x_181_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_181_, 0, v___x_136_);
lean_ctor_set(v___x_181_, 1, v___x_180_);
lean_inc_ref(v___x_181_);
v___x_182_ = l_Lean_Syntax_node3(v___x_136_, v___x_156_, v___x_165_, v___x_179_, v___x_181_);
v___x_183_ = l_Lean_Syntax_node1(v___x_136_, v___x_155_, v___x_182_);
v___x_184_ = l_Lean_Syntax_node2(v___x_136_, v___x_149_, v___x_154_, v___x_183_);
v___x_185_ = lean_unsigned_to_nat(9u);
v___x_186_ = lean_mk_empty_array_with_capacity(v___x_185_);
v___x_187_ = lean_array_push(v___x_186_, v___x_139_);
v___x_188_ = lean_array_push(v___x_187_, v___x_141_);
v___x_189_ = lean_array_push(v___x_188_, v___x_146_);
v___x_190_ = lean_array_push(v___x_189_, v___x_148_);
v___x_191_ = lean_array_push(v___x_190_, v___x_184_);
v___x_192_ = lean_array_push(v___x_191_, v___x_181_);
v___x_193_ = lean_array_push(v___x_192_, v___x_132_);
v___x_194_ = lean_array_push(v___x_193_, v___x_177_);
v___x_195_ = lean_array_push(v___x_194_, v___x_134_);
v___x_196_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_196_, 0, v___x_136_);
lean_ctor_set(v___x_196_, 1, v___x_137_);
lean_ctor_set(v___x_196_, 2, v___x_195_);
v___x_197_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_197_, 0, v___x_196_);
lean_ctor_set(v___x_197_, 1, v_a_123_);
return v___x_197_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Pi___aux__Mathlib__Order__PiLex______macroRules__Pi__term_u03a0_u2097___x2c____1___boxed(lean_object* v_x_198_, lean_object* v_a_199_, lean_object* v_a_200_){
_start:
{
lean_object* v_res_201_; 
v_res_201_ = lp_mathlib_Pi___aux__Mathlib__Order__PiLex______macroRules__Pi__term_u03a0_u2097___x2c____1(v_x_198_, v_a_199_, v_a_200_);
lean_dec_ref(v_a_199_);
return v_res_201_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Pi_instLTLexForall(lean_object* v_00_u03b9_202_, lean_object* v_00_u03b2_203_, lean_object* v_inst_204_, lean_object* v_inst_205_){
_start:
{
lean_object* v___x_206_; 
v___x_206_ = lean_box(0);
return v___x_206_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Pi_instLTLexForall___boxed(lean_object* v_00_u03b9_207_, lean_object* v_00_u03b2_208_, lean_object* v_inst_209_, lean_object* v_inst_210_){
_start:
{
lean_object* v_res_211_; 
v_res_211_ = lp_mathlib_Pi_instLTLexForall(v_00_u03b9_207_, v_00_u03b2_208_, v_inst_209_, v_inst_210_);
lean_dec_ref(v_inst_210_);
return v_res_211_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Pi_instLTColexForall(lean_object* v_00_u03b9_212_, lean_object* v_00_u03b2_213_, lean_object* v_inst_214_, lean_object* v_inst_215_){
_start:
{
lean_object* v___x_216_; 
v___x_216_ = lean_box(0);
return v___x_216_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Pi_instLTColexForall___boxed(lean_object* v_00_u03b9_217_, lean_object* v_00_u03b2_218_, lean_object* v_inst_219_, lean_object* v_inst_220_){
_start:
{
lean_object* v_res_221_; 
v_res_221_ = lp_mathlib_Pi_instLTColexForall(v_00_u03b9_217_, v_00_u03b2_218_, v_inst_219_, v_inst_220_);
lean_dec_ref(v_inst_220_);
return v_res_221_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Pi_instPartialOrderLexForallOfLinearOrder(lean_object* v_00_u03b9_225_, lean_object* v_00_u03b2_226_, lean_object* v_inst_227_, lean_object* v_inst_228_){
_start:
{
lean_object* v___x_229_; 
v___x_229_ = ((lean_object*)(lp_mathlib_Pi_instPartialOrderLexForallOfLinearOrder___closed__0));
return v___x_229_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Pi_instPartialOrderLexForallOfLinearOrder___boxed(lean_object* v_00_u03b9_230_, lean_object* v_00_u03b2_231_, lean_object* v_inst_232_, lean_object* v_inst_233_){
_start:
{
lean_object* v_res_234_; 
v_res_234_ = lp_mathlib_Pi_instPartialOrderLexForallOfLinearOrder(v_00_u03b9_230_, v_00_u03b2_231_, v_inst_232_, v_inst_233_);
lean_dec_ref(v_inst_233_);
lean_dec_ref(v_inst_232_);
return v_res_234_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Pi_instPartialOrderColexForallOfLinearOrder(lean_object* v_00_u03b9_235_, lean_object* v_00_u03b2_236_, lean_object* v_inst_237_, lean_object* v_inst_238_){
_start:
{
lean_object* v___x_239_; 
v___x_239_ = ((lean_object*)(lp_mathlib_Pi_instPartialOrderLexForallOfLinearOrder___closed__0));
return v___x_239_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Pi_instPartialOrderColexForallOfLinearOrder___boxed(lean_object* v_00_u03b9_240_, lean_object* v_00_u03b2_241_, lean_object* v_inst_242_, lean_object* v_inst_243_){
_start:
{
lean_object* v_res_244_; 
v_res_244_ = lp_mathlib_Pi_instPartialOrderColexForallOfLinearOrder(v_00_u03b9_240_, v_00_u03b2_241_, v_inst_242_, v_inst_243_);
lean_dec_ref(v_inst_243_);
lean_dec_ref(v_inst_242_);
return v_res_244_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Pi_instOrderBotLexForallOfWellFoundedLT___redArg___lam__0(lean_object* v_inst_245_, lean_object* v_x_246_){
_start:
{
lean_object* v___x_247_; 
v___x_247_ = lean_apply_1(v_inst_245_, v_x_246_);
return v___x_247_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Pi_instOrderBotLexForallOfWellFoundedLT___redArg___lam__1(lean_object* v___x_248_, lean_object* v___f_249_, lean_object* v___y_250_){
_start:
{
lean_object* v_toFun_251_; lean_object* v___x_252_; 
v_toFun_251_ = lean_ctor_get(v___x_248_, 0);
lean_inc(v_toFun_251_);
lean_dec_ref(v___x_248_);
v___x_252_ = lean_apply_2(v_toFun_251_, v___f_249_, v___y_250_);
return v___x_252_;
}
}
static lean_object* _init_lp_mathlib_Pi_instOrderBotLexForallOfWellFoundedLT___redArg___closed__0(void){
_start:
{
lean_object* v___x_253_; 
v___x_253_ = lp_mathlib_Equiv_refl(lean_box(0));
return v___x_253_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Pi_instOrderBotLexForallOfWellFoundedLT___redArg(lean_object* v_inst_254_){
_start:
{
lean_object* v___f_255_; lean_object* v___x_256_; lean_object* v___f_257_; 
v___f_255_ = lean_alloc_closure((void*)(lp_mathlib_Pi_instOrderBotLexForallOfWellFoundedLT___redArg___lam__0), 2, 1);
lean_closure_set(v___f_255_, 0, v_inst_254_);
v___x_256_ = lean_obj_once(&lp_mathlib_Pi_instOrderBotLexForallOfWellFoundedLT___redArg___closed__0, &lp_mathlib_Pi_instOrderBotLexForallOfWellFoundedLT___redArg___closed__0_once, _init_lp_mathlib_Pi_instOrderBotLexForallOfWellFoundedLT___redArg___closed__0);
v___f_257_ = lean_alloc_closure((void*)(lp_mathlib_Pi_instOrderBotLexForallOfWellFoundedLT___redArg___lam__1), 3, 2);
lean_closure_set(v___f_257_, 0, v___x_256_);
lean_closure_set(v___f_257_, 1, v___f_255_);
return v___f_257_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Pi_instOrderBotLexForallOfWellFoundedLT(lean_object* v_00_u03b9_258_, lean_object* v_00_u03b2_259_, lean_object* v_inst_260_, lean_object* v_inst_261_, lean_object* v_inst_262_, lean_object* v_inst_263_){
_start:
{
lean_object* v___x_264_; 
v___x_264_ = lp_mathlib_Pi_instOrderBotLexForallOfWellFoundedLT___redArg(v_inst_263_);
return v___x_264_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Pi_instOrderBotLexForallOfWellFoundedLT___boxed(lean_object* v_00_u03b9_265_, lean_object* v_00_u03b2_266_, lean_object* v_inst_267_, lean_object* v_inst_268_, lean_object* v_inst_269_, lean_object* v_inst_270_){
_start:
{
lean_object* v_res_271_; 
v_res_271_ = lp_mathlib_Pi_instOrderBotLexForallOfWellFoundedLT(v_00_u03b9_265_, v_00_u03b2_266_, v_inst_267_, v_inst_268_, v_inst_269_, v_inst_270_);
lean_dec_ref(v_inst_269_);
lean_dec_ref(v_inst_267_);
return v_res_271_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Pi_instOrderBotColexForallOfWellFoundedGT___redArg(lean_object* v_inst_272_){
_start:
{
lean_object* v___f_273_; lean_object* v___x_274_; lean_object* v___f_275_; 
v___f_273_ = lean_alloc_closure((void*)(lp_mathlib_Pi_instOrderBotLexForallOfWellFoundedLT___redArg___lam__0), 2, 1);
lean_closure_set(v___f_273_, 0, v_inst_272_);
v___x_274_ = lean_obj_once(&lp_mathlib_Pi_instOrderBotLexForallOfWellFoundedLT___redArg___closed__0, &lp_mathlib_Pi_instOrderBotLexForallOfWellFoundedLT___redArg___closed__0_once, _init_lp_mathlib_Pi_instOrderBotLexForallOfWellFoundedLT___redArg___closed__0);
v___f_275_ = lean_alloc_closure((void*)(lp_mathlib_Pi_instOrderBotLexForallOfWellFoundedLT___redArg___lam__1), 3, 2);
lean_closure_set(v___f_275_, 0, v___x_274_);
lean_closure_set(v___f_275_, 1, v___f_273_);
return v___f_275_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Pi_instOrderBotColexForallOfWellFoundedGT(lean_object* v_00_u03b9_276_, lean_object* v_00_u03b2_277_, lean_object* v_inst_278_, lean_object* v_inst_279_, lean_object* v_inst_280_, lean_object* v_inst_281_){
_start:
{
lean_object* v___x_282_; 
v___x_282_ = lp_mathlib_Pi_instOrderBotColexForallOfWellFoundedGT___redArg(v_inst_281_);
return v___x_282_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Pi_instOrderBotColexForallOfWellFoundedGT___boxed(lean_object* v_00_u03b9_283_, lean_object* v_00_u03b2_284_, lean_object* v_inst_285_, lean_object* v_inst_286_, lean_object* v_inst_287_, lean_object* v_inst_288_){
_start:
{
lean_object* v_res_289_; 
v_res_289_ = lp_mathlib_Pi_instOrderBotColexForallOfWellFoundedGT(v_00_u03b9_283_, v_00_u03b2_284_, v_inst_285_, v_inst_286_, v_inst_287_, v_inst_288_);
lean_dec_ref(v_inst_287_);
lean_dec_ref(v_inst_285_);
return v_res_289_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Pi_instOrderTopLexForallOfWellFoundedLT___redArg(lean_object* v_inst_290_){
_start:
{
lean_object* v___f_291_; lean_object* v___x_292_; lean_object* v___f_293_; 
v___f_291_ = lean_alloc_closure((void*)(lp_mathlib_Pi_instOrderBotLexForallOfWellFoundedLT___redArg___lam__0), 2, 1);
lean_closure_set(v___f_291_, 0, v_inst_290_);
v___x_292_ = lean_obj_once(&lp_mathlib_Pi_instOrderBotLexForallOfWellFoundedLT___redArg___closed__0, &lp_mathlib_Pi_instOrderBotLexForallOfWellFoundedLT___redArg___closed__0_once, _init_lp_mathlib_Pi_instOrderBotLexForallOfWellFoundedLT___redArg___closed__0);
v___f_293_ = lean_alloc_closure((void*)(lp_mathlib_Pi_instOrderBotLexForallOfWellFoundedLT___redArg___lam__1), 3, 2);
lean_closure_set(v___f_293_, 0, v___x_292_);
lean_closure_set(v___f_293_, 1, v___f_291_);
return v___f_293_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Pi_instOrderTopLexForallOfWellFoundedLT(lean_object* v_00_u03b9_294_, lean_object* v_00_u03b2_295_, lean_object* v_inst_296_, lean_object* v_inst_297_, lean_object* v_inst_298_, lean_object* v_inst_299_){
_start:
{
lean_object* v___x_300_; 
v___x_300_ = lp_mathlib_Pi_instOrderTopLexForallOfWellFoundedLT___redArg(v_inst_299_);
return v___x_300_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Pi_instOrderTopLexForallOfWellFoundedLT___boxed(lean_object* v_00_u03b9_301_, lean_object* v_00_u03b2_302_, lean_object* v_inst_303_, lean_object* v_inst_304_, lean_object* v_inst_305_, lean_object* v_inst_306_){
_start:
{
lean_object* v_res_307_; 
v_res_307_ = lp_mathlib_Pi_instOrderTopLexForallOfWellFoundedLT(v_00_u03b9_301_, v_00_u03b2_302_, v_inst_303_, v_inst_304_, v_inst_305_, v_inst_306_);
lean_dec_ref(v_inst_305_);
lean_dec_ref(v_inst_303_);
return v_res_307_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Pi_instOrderTopColexForallOfWellFoundedGT___redArg(lean_object* v_inst_308_){
_start:
{
lean_object* v___f_309_; lean_object* v___x_310_; lean_object* v___f_311_; 
v___f_309_ = lean_alloc_closure((void*)(lp_mathlib_Pi_instOrderBotLexForallOfWellFoundedLT___redArg___lam__0), 2, 1);
lean_closure_set(v___f_309_, 0, v_inst_308_);
v___x_310_ = lean_obj_once(&lp_mathlib_Pi_instOrderBotLexForallOfWellFoundedLT___redArg___closed__0, &lp_mathlib_Pi_instOrderBotLexForallOfWellFoundedLT___redArg___closed__0_once, _init_lp_mathlib_Pi_instOrderBotLexForallOfWellFoundedLT___redArg___closed__0);
v___f_311_ = lean_alloc_closure((void*)(lp_mathlib_Pi_instOrderBotLexForallOfWellFoundedLT___redArg___lam__1), 3, 2);
lean_closure_set(v___f_311_, 0, v___x_310_);
lean_closure_set(v___f_311_, 1, v___f_309_);
return v___f_311_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Pi_instOrderTopColexForallOfWellFoundedGT(lean_object* v_00_u03b9_312_, lean_object* v_00_u03b2_313_, lean_object* v_inst_314_, lean_object* v_inst_315_, lean_object* v_inst_316_, lean_object* v_inst_317_){
_start:
{
lean_object* v___x_318_; 
v___x_318_ = lp_mathlib_Pi_instOrderTopColexForallOfWellFoundedGT___redArg(v_inst_317_);
return v___x_318_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Pi_instOrderTopColexForallOfWellFoundedGT___boxed(lean_object* v_00_u03b9_319_, lean_object* v_00_u03b2_320_, lean_object* v_inst_321_, lean_object* v_inst_322_, lean_object* v_inst_323_, lean_object* v_inst_324_){
_start:
{
lean_object* v_res_325_; 
v_res_325_ = lp_mathlib_Pi_instOrderTopColexForallOfWellFoundedGT(v_00_u03b9_319_, v_00_u03b2_320_, v_inst_321_, v_inst_322_, v_inst_323_, v_inst_324_);
lean_dec_ref(v_inst_323_);
lean_dec_ref(v_inst_321_);
return v_res_325_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Pi_instBoundedOrderLexForallOfWellFoundedLT___redArg___lam__0(lean_object* v_inst_326_, lean_object* v_a_327_){
_start:
{
lean_object* v___x_328_; lean_object* v_toOrderTop_329_; 
v___x_328_ = lean_apply_1(v_inst_326_, v_a_327_);
v_toOrderTop_329_ = lean_ctor_get(v___x_328_, 0);
lean_inc(v_toOrderTop_329_);
lean_dec_ref(v___x_328_);
return v_toOrderTop_329_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Pi_instBoundedOrderLexForallOfWellFoundedLT___redArg___lam__1(lean_object* v_inst_330_, lean_object* v_a_331_){
_start:
{
lean_object* v___x_332_; lean_object* v_toOrderBot_333_; 
v___x_332_ = lean_apply_1(v_inst_330_, v_a_331_);
v_toOrderBot_333_ = lean_ctor_get(v___x_332_, 1);
lean_inc(v_toOrderBot_333_);
lean_dec_ref(v___x_332_);
return v_toOrderBot_333_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Pi_instBoundedOrderLexForallOfWellFoundedLT___redArg(lean_object* v_inst_334_){
_start:
{
lean_object* v___f_335_; lean_object* v___f_336_; lean_object* v___x_337_; lean_object* v___x_338_; lean_object* v___x_339_; 
lean_inc_ref(v_inst_334_);
v___f_335_ = lean_alloc_closure((void*)(lp_mathlib_Pi_instBoundedOrderLexForallOfWellFoundedLT___redArg___lam__0), 2, 1);
lean_closure_set(v___f_335_, 0, v_inst_334_);
v___f_336_ = lean_alloc_closure((void*)(lp_mathlib_Pi_instBoundedOrderLexForallOfWellFoundedLT___redArg___lam__1), 2, 1);
lean_closure_set(v___f_336_, 0, v_inst_334_);
v___x_337_ = lp_mathlib_Pi_instOrderTopLexForallOfWellFoundedLT___redArg(v___f_335_);
v___x_338_ = lp_mathlib_Pi_instOrderBotLexForallOfWellFoundedLT___redArg(v___f_336_);
v___x_339_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_339_, 0, v___x_337_);
lean_ctor_set(v___x_339_, 1, v___x_338_);
return v___x_339_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Pi_instBoundedOrderLexForallOfWellFoundedLT(lean_object* v_00_u03b9_340_, lean_object* v_00_u03b2_341_, lean_object* v_inst_342_, lean_object* v_inst_343_, lean_object* v_inst_344_, lean_object* v_inst_345_){
_start:
{
lean_object* v___x_346_; 
v___x_346_ = lp_mathlib_Pi_instBoundedOrderLexForallOfWellFoundedLT___redArg(v_inst_345_);
return v___x_346_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Pi_instBoundedOrderLexForallOfWellFoundedLT___boxed(lean_object* v_00_u03b9_347_, lean_object* v_00_u03b2_348_, lean_object* v_inst_349_, lean_object* v_inst_350_, lean_object* v_inst_351_, lean_object* v_inst_352_){
_start:
{
lean_object* v_res_353_; 
v_res_353_ = lp_mathlib_Pi_instBoundedOrderLexForallOfWellFoundedLT(v_00_u03b9_347_, v_00_u03b2_348_, v_inst_349_, v_inst_350_, v_inst_351_, v_inst_352_);
lean_dec_ref(v_inst_351_);
lean_dec_ref(v_inst_349_);
return v_res_353_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Pi_instBoundedOrderColexForallOfWellFoundedGT___redArg(lean_object* v_inst_354_){
_start:
{
lean_object* v___f_355_; lean_object* v___f_356_; lean_object* v___x_357_; lean_object* v___x_358_; lean_object* v___x_359_; 
lean_inc_ref(v_inst_354_);
v___f_355_ = lean_alloc_closure((void*)(lp_mathlib_Pi_instBoundedOrderLexForallOfWellFoundedLT___redArg___lam__0), 2, 1);
lean_closure_set(v___f_355_, 0, v_inst_354_);
v___f_356_ = lean_alloc_closure((void*)(lp_mathlib_Pi_instBoundedOrderLexForallOfWellFoundedLT___redArg___lam__1), 2, 1);
lean_closure_set(v___f_356_, 0, v_inst_354_);
v___x_357_ = lp_mathlib_Pi_instOrderTopColexForallOfWellFoundedGT___redArg(v___f_355_);
v___x_358_ = lp_mathlib_Pi_instOrderBotColexForallOfWellFoundedGT___redArg(v___f_356_);
v___x_359_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_359_, 0, v___x_357_);
lean_ctor_set(v___x_359_, 1, v___x_358_);
return v___x_359_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Pi_instBoundedOrderColexForallOfWellFoundedGT(lean_object* v_00_u03b9_360_, lean_object* v_00_u03b2_361_, lean_object* v_inst_362_, lean_object* v_inst_363_, lean_object* v_inst_364_, lean_object* v_inst_365_){
_start:
{
lean_object* v___x_366_; 
v___x_366_ = lp_mathlib_Pi_instBoundedOrderColexForallOfWellFoundedGT___redArg(v_inst_365_);
return v___x_366_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Pi_instBoundedOrderColexForallOfWellFoundedGT___boxed(lean_object* v_00_u03b9_367_, lean_object* v_00_u03b2_368_, lean_object* v_inst_369_, lean_object* v_inst_370_, lean_object* v_inst_371_, lean_object* v_inst_372_){
_start:
{
lean_object* v_res_373_; 
v_res_373_ = lp_mathlib_Pi_instBoundedOrderColexForallOfWellFoundedGT(v_00_u03b9_367_, v_00_u03b2_368_, v_inst_369_, v_inst_370_, v_inst_371_, v_inst_372_);
lean_dec_ref(v_inst_371_);
lean_dec_ref(v_inst_369_);
return v_res_373_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Order_Lex(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Order_WellFounded(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Tactic_Common(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Tactic_Attr_Core(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_Order_PiLex(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Order_Lex(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Order_WellFounded(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Tactic_Common(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Tactic_Attr_Core(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_Order_PiLex(uint8_t builtin) {
lean_object * res;
if (_G_meta_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_meta_initialized = true;
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
lp_mathlib_Pi_term_u03a0_u2097___x2c__ = _init_lp_mathlib_Pi_term_u03a0_u2097___x2c__();
lean_mark_persistent(lp_mathlib_Pi_term_u03a0_u2097___x2c__);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Order_Lex(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Order_WellFounded(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Tactic_Common(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Tactic_Attr_Core(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Order_PiLex(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Order_Lex(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Order_WellFounded(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Tactic_Common(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Tactic_Attr_Core(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Order_PiLex(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_Order_PiLex(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_Order_PiLex(builtin);
}
#ifdef __cplusplus
}
#endif
