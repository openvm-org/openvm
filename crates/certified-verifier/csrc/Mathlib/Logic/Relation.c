// Lean compiler output
// Module: Mathlib.Logic.Relation
// Imports: public import Init public meta import Init public import Mathlib.Logic.Relator public import Mathlib.Tactic.Use public import Mathlib.Tactic.MkIffOfInductiveProp public import Mathlib.Tactic.SimpRw public import Mathlib.Order.Defs.Prop public import Mathlib.Order.Defs.Unbundled public import Batteries.Logic public import Batteries.Tactic.Trans
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
lean_object* l_Lean_Name_mkStr4(lean_object*, lean_object*, lean_object*, lean_object*);
uint8_t l_Lean_Syntax_isOfKind(lean_object*, lean_object*);
lean_object* l_Lean_Syntax_getArg(lean_object*, lean_object*);
lean_object* l_Lean_Name_mkStr1(lean_object*);
uint8_t l_Lean_Syntax_matchesNull(lean_object*, lean_object*);
lean_object* l_Lean_replaceRef(lean_object*, lean_object*);
lean_object* l_Lean_SourceInfo_fromRef(lean_object*, uint8_t);
lean_object* l_Lean_Name_str___override(lean_object*, lean_object*);
lean_object* l_Lean_Name_num___override(lean_object*, lean_object*);
lean_object* l_Lean_Syntax_node3(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_String_toRawSubstring_x27(lean_object*);
lean_object* l_Lean_Name_mkStr2(lean_object*, lean_object*);
lean_object* l_Lean_addMacroScope(lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Syntax_node2(lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib___private_Mathlib_Logic_Relation_0__Relation_term___u2218r___00__closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "_private"};
static const lean_object* lp_mathlib___private_Mathlib_Logic_Relation_0__Relation_term___u2218r___00__closed__0 = (const lean_object*)&lp_mathlib___private_Mathlib_Logic_Relation_0__Relation_term___u2218r___00__closed__0_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Logic_Relation_0__Relation_term___u2218r___00__closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Logic_Relation_0__Relation_term___u2218r___00__closed__0_value),LEAN_SCALAR_PTR_LITERAL(103, 214, 75, 80, 34, 198, 193, 153)}};
static const lean_object* lp_mathlib___private_Mathlib_Logic_Relation_0__Relation_term___u2218r___00__closed__1 = (const lean_object*)&lp_mathlib___private_Mathlib_Logic_Relation_0__Relation_term___u2218r___00__closed__1_value;
static const lean_string_object lp_mathlib___private_Mathlib_Logic_Relation_0__Relation_term___u2218r___00__closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "Mathlib"};
static const lean_object* lp_mathlib___private_Mathlib_Logic_Relation_0__Relation_term___u2218r___00__closed__2 = (const lean_object*)&lp_mathlib___private_Mathlib_Logic_Relation_0__Relation_term___u2218r___00__closed__2_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Logic_Relation_0__Relation_term___u2218r___00__closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Logic_Relation_0__Relation_term___u2218r___00__closed__1_value),((lean_object*)&lp_mathlib___private_Mathlib_Logic_Relation_0__Relation_term___u2218r___00__closed__2_value),LEAN_SCALAR_PTR_LITERAL(234, 232, 174, 134, 127, 136, 69, 92)}};
static const lean_object* lp_mathlib___private_Mathlib_Logic_Relation_0__Relation_term___u2218r___00__closed__3 = (const lean_object*)&lp_mathlib___private_Mathlib_Logic_Relation_0__Relation_term___u2218r___00__closed__3_value;
static const lean_string_object lp_mathlib___private_Mathlib_Logic_Relation_0__Relation_term___u2218r___00__closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "Logic"};
static const lean_object* lp_mathlib___private_Mathlib_Logic_Relation_0__Relation_term___u2218r___00__closed__4 = (const lean_object*)&lp_mathlib___private_Mathlib_Logic_Relation_0__Relation_term___u2218r___00__closed__4_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Logic_Relation_0__Relation_term___u2218r___00__closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Logic_Relation_0__Relation_term___u2218r___00__closed__3_value),((lean_object*)&lp_mathlib___private_Mathlib_Logic_Relation_0__Relation_term___u2218r___00__closed__4_value),LEAN_SCALAR_PTR_LITERAL(222, 164, 153, 31, 25, 197, 191, 150)}};
static const lean_object* lp_mathlib___private_Mathlib_Logic_Relation_0__Relation_term___u2218r___00__closed__5 = (const lean_object*)&lp_mathlib___private_Mathlib_Logic_Relation_0__Relation_term___u2218r___00__closed__5_value;
static const lean_string_object lp_mathlib___private_Mathlib_Logic_Relation_0__Relation_term___u2218r___00__closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "Relation"};
static const lean_object* lp_mathlib___private_Mathlib_Logic_Relation_0__Relation_term___u2218r___00__closed__6 = (const lean_object*)&lp_mathlib___private_Mathlib_Logic_Relation_0__Relation_term___u2218r___00__closed__6_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Logic_Relation_0__Relation_term___u2218r___00__closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Logic_Relation_0__Relation_term___u2218r___00__closed__5_value),((lean_object*)&lp_mathlib___private_Mathlib_Logic_Relation_0__Relation_term___u2218r___00__closed__6_value),LEAN_SCALAR_PTR_LITERAL(194, 39, 194, 215, 0, 48, 38, 169)}};
static const lean_object* lp_mathlib___private_Mathlib_Logic_Relation_0__Relation_term___u2218r___00__closed__7 = (const lean_object*)&lp_mathlib___private_Mathlib_Logic_Relation_0__Relation_term___u2218r___00__closed__7_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Logic_Relation_0__Relation_term___u2218r___00__closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Logic_Relation_0__Relation_term___u2218r___00__closed__7_value),((lean_object*)(((size_t)(0) << 1) | 1)),LEAN_SCALAR_PTR_LITERAL(163, 71, 91, 240, 215, 38, 238, 201)}};
static const lean_object* lp_mathlib___private_Mathlib_Logic_Relation_0__Relation_term___u2218r___00__closed__8 = (const lean_object*)&lp_mathlib___private_Mathlib_Logic_Relation_0__Relation_term___u2218r___00__closed__8_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Logic_Relation_0__Relation_term___u2218r___00__closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Logic_Relation_0__Relation_term___u2218r___00__closed__8_value),((lean_object*)&lp_mathlib___private_Mathlib_Logic_Relation_0__Relation_term___u2218r___00__closed__6_value),LEAN_SCALAR_PTR_LITERAL(83, 163, 85, 25, 27, 234, 19, 80)}};
static const lean_object* lp_mathlib___private_Mathlib_Logic_Relation_0__Relation_term___u2218r___00__closed__9 = (const lean_object*)&lp_mathlib___private_Mathlib_Logic_Relation_0__Relation_term___u2218r___00__closed__9_value;
static const lean_string_object lp_mathlib___private_Mathlib_Logic_Relation_0__Relation_term___u2218r___00__closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 11, .m_capacity = 11, .m_length = 8, .m_data = "term_∘r_"};
static const lean_object* lp_mathlib___private_Mathlib_Logic_Relation_0__Relation_term___u2218r___00__closed__10 = (const lean_object*)&lp_mathlib___private_Mathlib_Logic_Relation_0__Relation_term___u2218r___00__closed__10_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Logic_Relation_0__Relation_term___u2218r___00__closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Logic_Relation_0__Relation_term___u2218r___00__closed__9_value),((lean_object*)&lp_mathlib___private_Mathlib_Logic_Relation_0__Relation_term___u2218r___00__closed__10_value),LEAN_SCALAR_PTR_LITERAL(176, 207, 70, 51, 42, 52, 157, 9)}};
static const lean_object* lp_mathlib___private_Mathlib_Logic_Relation_0__Relation_term___u2218r___00__closed__11 = (const lean_object*)&lp_mathlib___private_Mathlib_Logic_Relation_0__Relation_term___u2218r___00__closed__11_value;
static const lean_string_object lp_mathlib___private_Mathlib_Logic_Relation_0__Relation_term___u2218r___00__closed__12_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "andthen"};
static const lean_object* lp_mathlib___private_Mathlib_Logic_Relation_0__Relation_term___u2218r___00__closed__12 = (const lean_object*)&lp_mathlib___private_Mathlib_Logic_Relation_0__Relation_term___u2218r___00__closed__12_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Logic_Relation_0__Relation_term___u2218r___00__closed__13_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Logic_Relation_0__Relation_term___u2218r___00__closed__12_value),LEAN_SCALAR_PTR_LITERAL(40, 255, 78, 30, 143, 119, 117, 174)}};
static const lean_object* lp_mathlib___private_Mathlib_Logic_Relation_0__Relation_term___u2218r___00__closed__13 = (const lean_object*)&lp_mathlib___private_Mathlib_Logic_Relation_0__Relation_term___u2218r___00__closed__13_value;
static const lean_string_object lp_mathlib___private_Mathlib_Logic_Relation_0__Relation_term___u2218r___00__closed__14_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 4, .m_data = " ∘r "};
static const lean_object* lp_mathlib___private_Mathlib_Logic_Relation_0__Relation_term___u2218r___00__closed__14 = (const lean_object*)&lp_mathlib___private_Mathlib_Logic_Relation_0__Relation_term___u2218r___00__closed__14_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Logic_Relation_0__Relation_term___u2218r___00__closed__15_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 5}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Logic_Relation_0__Relation_term___u2218r___00__closed__14_value)}};
static const lean_object* lp_mathlib___private_Mathlib_Logic_Relation_0__Relation_term___u2218r___00__closed__15 = (const lean_object*)&lp_mathlib___private_Mathlib_Logic_Relation_0__Relation_term___u2218r___00__closed__15_value;
static const lean_string_object lp_mathlib___private_Mathlib_Logic_Relation_0__Relation_term___u2218r___00__closed__16_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "term"};
static const lean_object* lp_mathlib___private_Mathlib_Logic_Relation_0__Relation_term___u2218r___00__closed__16 = (const lean_object*)&lp_mathlib___private_Mathlib_Logic_Relation_0__Relation_term___u2218r___00__closed__16_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Logic_Relation_0__Relation_term___u2218r___00__closed__17_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Logic_Relation_0__Relation_term___u2218r___00__closed__16_value),LEAN_SCALAR_PTR_LITERAL(187, 230, 181, 162, 253, 146, 122, 119)}};
static const lean_object* lp_mathlib___private_Mathlib_Logic_Relation_0__Relation_term___u2218r___00__closed__17 = (const lean_object*)&lp_mathlib___private_Mathlib_Logic_Relation_0__Relation_term___u2218r___00__closed__17_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Logic_Relation_0__Relation_term___u2218r___00__closed__18_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 7}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Logic_Relation_0__Relation_term___u2218r___00__closed__17_value),((lean_object*)(((size_t)(80) << 1) | 1))}};
static const lean_object* lp_mathlib___private_Mathlib_Logic_Relation_0__Relation_term___u2218r___00__closed__18 = (const lean_object*)&lp_mathlib___private_Mathlib_Logic_Relation_0__Relation_term___u2218r___00__closed__18_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Logic_Relation_0__Relation_term___u2218r___00__closed__19_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Logic_Relation_0__Relation_term___u2218r___00__closed__13_value),((lean_object*)&lp_mathlib___private_Mathlib_Logic_Relation_0__Relation_term___u2218r___00__closed__15_value),((lean_object*)&lp_mathlib___private_Mathlib_Logic_Relation_0__Relation_term___u2218r___00__closed__18_value)}};
static const lean_object* lp_mathlib___private_Mathlib_Logic_Relation_0__Relation_term___u2218r___00__closed__19 = (const lean_object*)&lp_mathlib___private_Mathlib_Logic_Relation_0__Relation_term___u2218r___00__closed__19_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Logic_Relation_0__Relation_term___u2218r___00__closed__20_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*4 + 0, .m_other = 4, .m_tag = 4}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Logic_Relation_0__Relation_term___u2218r___00__closed__11_value),((lean_object*)(((size_t)(80) << 1) | 1)),((lean_object*)(((size_t)(81) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Logic_Relation_0__Relation_term___u2218r___00__closed__19_value)}};
static const lean_object* lp_mathlib___private_Mathlib_Logic_Relation_0__Relation_term___u2218r___00__closed__20 = (const lean_object*)&lp_mathlib___private_Mathlib_Logic_Relation_0__Relation_term___u2218r___00__closed__20_value;
LEAN_EXPORT const lean_object* lp_mathlib___private_Mathlib_Logic_Relation_0__Relation_term___u2218r__ = (const lean_object*)&lp_mathlib___private_Mathlib_Logic_Relation_0__Relation_term___u2218r___00__closed__20_value;
static const lean_string_object lp_mathlib___private_Mathlib_Logic_Relation_0__Relation___aux__Mathlib__Logic__Relation______macroRules____private__Mathlib__Logic__Relation__0__Relation__term___u2218r____1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "Lean"};
static const lean_object* lp_mathlib___private_Mathlib_Logic_Relation_0__Relation___aux__Mathlib__Logic__Relation______macroRules____private__Mathlib__Logic__Relation__0__Relation__term___u2218r____1___closed__0 = (const lean_object*)&lp_mathlib___private_Mathlib_Logic_Relation_0__Relation___aux__Mathlib__Logic__Relation______macroRules____private__Mathlib__Logic__Relation__0__Relation__term___u2218r____1___closed__0_value;
static const lean_string_object lp_mathlib___private_Mathlib_Logic_Relation_0__Relation___aux__Mathlib__Logic__Relation______macroRules____private__Mathlib__Logic__Relation__0__Relation__term___u2218r____1___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "Parser"};
static const lean_object* lp_mathlib___private_Mathlib_Logic_Relation_0__Relation___aux__Mathlib__Logic__Relation______macroRules____private__Mathlib__Logic__Relation__0__Relation__term___u2218r____1___closed__1 = (const lean_object*)&lp_mathlib___private_Mathlib_Logic_Relation_0__Relation___aux__Mathlib__Logic__Relation______macroRules____private__Mathlib__Logic__Relation__0__Relation__term___u2218r____1___closed__1_value;
static const lean_string_object lp_mathlib___private_Mathlib_Logic_Relation_0__Relation___aux__Mathlib__Logic__Relation______macroRules____private__Mathlib__Logic__Relation__0__Relation__term___u2218r____1___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "Term"};
static const lean_object* lp_mathlib___private_Mathlib_Logic_Relation_0__Relation___aux__Mathlib__Logic__Relation______macroRules____private__Mathlib__Logic__Relation__0__Relation__term___u2218r____1___closed__2 = (const lean_object*)&lp_mathlib___private_Mathlib_Logic_Relation_0__Relation___aux__Mathlib__Logic__Relation______macroRules____private__Mathlib__Logic__Relation__0__Relation__term___u2218r____1___closed__2_value;
static const lean_string_object lp_mathlib___private_Mathlib_Logic_Relation_0__Relation___aux__Mathlib__Logic__Relation______macroRules____private__Mathlib__Logic__Relation__0__Relation__term___u2218r____1___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = "app"};
static const lean_object* lp_mathlib___private_Mathlib_Logic_Relation_0__Relation___aux__Mathlib__Logic__Relation______macroRules____private__Mathlib__Logic__Relation__0__Relation__term___u2218r____1___closed__3 = (const lean_object*)&lp_mathlib___private_Mathlib_Logic_Relation_0__Relation___aux__Mathlib__Logic__Relation______macroRules____private__Mathlib__Logic__Relation__0__Relation__term___u2218r____1___closed__3_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Logic_Relation_0__Relation___aux__Mathlib__Logic__Relation______macroRules____private__Mathlib__Logic__Relation__0__Relation__term___u2218r____1___closed__4_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Logic_Relation_0__Relation___aux__Mathlib__Logic__Relation______macroRules____private__Mathlib__Logic__Relation__0__Relation__term___u2218r____1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Logic_Relation_0__Relation___aux__Mathlib__Logic__Relation______macroRules____private__Mathlib__Logic__Relation__0__Relation__term___u2218r____1___closed__4_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Logic_Relation_0__Relation___aux__Mathlib__Logic__Relation______macroRules____private__Mathlib__Logic__Relation__0__Relation__term___u2218r____1___closed__4_value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Logic_Relation_0__Relation___aux__Mathlib__Logic__Relation______macroRules____private__Mathlib__Logic__Relation__0__Relation__term___u2218r____1___closed__1_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Logic_Relation_0__Relation___aux__Mathlib__Logic__Relation______macroRules____private__Mathlib__Logic__Relation__0__Relation__term___u2218r____1___closed__4_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Logic_Relation_0__Relation___aux__Mathlib__Logic__Relation______macroRules____private__Mathlib__Logic__Relation__0__Relation__term___u2218r____1___closed__4_value_aux_1),((lean_object*)&lp_mathlib___private_Mathlib_Logic_Relation_0__Relation___aux__Mathlib__Logic__Relation______macroRules____private__Mathlib__Logic__Relation__0__Relation__term___u2218r____1___closed__2_value),LEAN_SCALAR_PTR_LITERAL(75, 170, 162, 138, 136, 204, 251, 229)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Logic_Relation_0__Relation___aux__Mathlib__Logic__Relation______macroRules____private__Mathlib__Logic__Relation__0__Relation__term___u2218r____1___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Logic_Relation_0__Relation___aux__Mathlib__Logic__Relation______macroRules____private__Mathlib__Logic__Relation__0__Relation__term___u2218r____1___closed__4_value_aux_2),((lean_object*)&lp_mathlib___private_Mathlib_Logic_Relation_0__Relation___aux__Mathlib__Logic__Relation______macroRules____private__Mathlib__Logic__Relation__0__Relation__term___u2218r____1___closed__3_value),LEAN_SCALAR_PTR_LITERAL(69, 118, 10, 41, 220, 156, 243, 179)}};
static const lean_object* lp_mathlib___private_Mathlib_Logic_Relation_0__Relation___aux__Mathlib__Logic__Relation______macroRules____private__Mathlib__Logic__Relation__0__Relation__term___u2218r____1___closed__4 = (const lean_object*)&lp_mathlib___private_Mathlib_Logic_Relation_0__Relation___aux__Mathlib__Logic__Relation______macroRules____private__Mathlib__Logic__Relation__0__Relation__term___u2218r____1___closed__4_value;
static const lean_string_object lp_mathlib___private_Mathlib_Logic_Relation_0__Relation___aux__Mathlib__Logic__Relation______macroRules____private__Mathlib__Logic__Relation__0__Relation__term___u2218r____1___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 14, .m_capacity = 14, .m_length = 13, .m_data = "Relation.Comp"};
static const lean_object* lp_mathlib___private_Mathlib_Logic_Relation_0__Relation___aux__Mathlib__Logic__Relation______macroRules____private__Mathlib__Logic__Relation__0__Relation__term___u2218r____1___closed__5 = (const lean_object*)&lp_mathlib___private_Mathlib_Logic_Relation_0__Relation___aux__Mathlib__Logic__Relation______macroRules____private__Mathlib__Logic__Relation__0__Relation__term___u2218r____1___closed__5_value;
static lean_once_cell_t lp_mathlib___private_Mathlib_Logic_Relation_0__Relation___aux__Mathlib__Logic__Relation______macroRules____private__Mathlib__Logic__Relation__0__Relation__term___u2218r____1___closed__6_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Logic_Relation_0__Relation___aux__Mathlib__Logic__Relation______macroRules____private__Mathlib__Logic__Relation__0__Relation__term___u2218r____1___closed__6;
static const lean_string_object lp_mathlib___private_Mathlib_Logic_Relation_0__Relation___aux__Mathlib__Logic__Relation______macroRules____private__Mathlib__Logic__Relation__0__Relation__term___u2218r____1___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "Comp"};
static const lean_object* lp_mathlib___private_Mathlib_Logic_Relation_0__Relation___aux__Mathlib__Logic__Relation______macroRules____private__Mathlib__Logic__Relation__0__Relation__term___u2218r____1___closed__7 = (const lean_object*)&lp_mathlib___private_Mathlib_Logic_Relation_0__Relation___aux__Mathlib__Logic__Relation______macroRules____private__Mathlib__Logic__Relation__0__Relation__term___u2218r____1___closed__7_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Logic_Relation_0__Relation___aux__Mathlib__Logic__Relation______macroRules____private__Mathlib__Logic__Relation__0__Relation__term___u2218r____1___closed__8_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Logic_Relation_0__Relation_term___u2218r___00__closed__6_value),LEAN_SCALAR_PTR_LITERAL(75, 52, 124, 254, 211, 241, 202, 221)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Logic_Relation_0__Relation___aux__Mathlib__Logic__Relation______macroRules____private__Mathlib__Logic__Relation__0__Relation__term___u2218r____1___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Logic_Relation_0__Relation___aux__Mathlib__Logic__Relation______macroRules____private__Mathlib__Logic__Relation__0__Relation__term___u2218r____1___closed__8_value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Logic_Relation_0__Relation___aux__Mathlib__Logic__Relation______macroRules____private__Mathlib__Logic__Relation__0__Relation__term___u2218r____1___closed__7_value),LEAN_SCALAR_PTR_LITERAL(208, 226, 194, 184, 8, 138, 89, 143)}};
static const lean_object* lp_mathlib___private_Mathlib_Logic_Relation_0__Relation___aux__Mathlib__Logic__Relation______macroRules____private__Mathlib__Logic__Relation__0__Relation__term___u2218r____1___closed__8 = (const lean_object*)&lp_mathlib___private_Mathlib_Logic_Relation_0__Relation___aux__Mathlib__Logic__Relation______macroRules____private__Mathlib__Logic__Relation__0__Relation__term___u2218r____1___closed__8_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Logic_Relation_0__Relation___aux__Mathlib__Logic__Relation______macroRules____private__Mathlib__Logic__Relation__0__Relation__term___u2218r____1___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Logic_Relation_0__Relation___aux__Mathlib__Logic__Relation______macroRules____private__Mathlib__Logic__Relation__0__Relation__term___u2218r____1___closed__8_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib___private_Mathlib_Logic_Relation_0__Relation___aux__Mathlib__Logic__Relation______macroRules____private__Mathlib__Logic__Relation__0__Relation__term___u2218r____1___closed__9 = (const lean_object*)&lp_mathlib___private_Mathlib_Logic_Relation_0__Relation___aux__Mathlib__Logic__Relation______macroRules____private__Mathlib__Logic__Relation__0__Relation__term___u2218r____1___closed__9_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Logic_Relation_0__Relation___aux__Mathlib__Logic__Relation______macroRules____private__Mathlib__Logic__Relation__0__Relation__term___u2218r____1___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Logic_Relation_0__Relation___aux__Mathlib__Logic__Relation______macroRules____private__Mathlib__Logic__Relation__0__Relation__term___u2218r____1___closed__9_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib___private_Mathlib_Logic_Relation_0__Relation___aux__Mathlib__Logic__Relation______macroRules____private__Mathlib__Logic__Relation__0__Relation__term___u2218r____1___closed__10 = (const lean_object*)&lp_mathlib___private_Mathlib_Logic_Relation_0__Relation___aux__Mathlib__Logic__Relation______macroRules____private__Mathlib__Logic__Relation__0__Relation__term___u2218r____1___closed__10_value;
static const lean_string_object lp_mathlib___private_Mathlib_Logic_Relation_0__Relation___aux__Mathlib__Logic__Relation______macroRules____private__Mathlib__Logic__Relation__0__Relation__term___u2218r____1___closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "null"};
static const lean_object* lp_mathlib___private_Mathlib_Logic_Relation_0__Relation___aux__Mathlib__Logic__Relation______macroRules____private__Mathlib__Logic__Relation__0__Relation__term___u2218r____1___closed__11 = (const lean_object*)&lp_mathlib___private_Mathlib_Logic_Relation_0__Relation___aux__Mathlib__Logic__Relation______macroRules____private__Mathlib__Logic__Relation__0__Relation__term___u2218r____1___closed__11_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Logic_Relation_0__Relation___aux__Mathlib__Logic__Relation______macroRules____private__Mathlib__Logic__Relation__0__Relation__term___u2218r____1___closed__12_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Logic_Relation_0__Relation___aux__Mathlib__Logic__Relation______macroRules____private__Mathlib__Logic__Relation__0__Relation__term___u2218r____1___closed__11_value),LEAN_SCALAR_PTR_LITERAL(24, 58, 49, 223, 146, 207, 197, 136)}};
static const lean_object* lp_mathlib___private_Mathlib_Logic_Relation_0__Relation___aux__Mathlib__Logic__Relation______macroRules____private__Mathlib__Logic__Relation__0__Relation__term___u2218r____1___closed__12 = (const lean_object*)&lp_mathlib___private_Mathlib_Logic_Relation_0__Relation___aux__Mathlib__Logic__Relation______macroRules____private__Mathlib__Logic__Relation__0__Relation__term___u2218r____1___closed__12_value;
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Logic_Relation_0__Relation___aux__Mathlib__Logic__Relation______macroRules____private__Mathlib__Logic__Relation__0__Relation__term___u2218r____1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Logic_Relation_0__Relation___aux__Mathlib__Logic__Relation______macroRules____private__Mathlib__Logic__Relation__0__Relation__term___u2218r____1___boxed(lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib___private_Mathlib_Logic_Relation_0__Relation___aux__Mathlib__Logic__Relation______unexpand__Relation__Comp__1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "ident"};
static const lean_object* lp_mathlib___private_Mathlib_Logic_Relation_0__Relation___aux__Mathlib__Logic__Relation______unexpand__Relation__Comp__1___closed__0 = (const lean_object*)&lp_mathlib___private_Mathlib_Logic_Relation_0__Relation___aux__Mathlib__Logic__Relation______unexpand__Relation__Comp__1___closed__0_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Logic_Relation_0__Relation___aux__Mathlib__Logic__Relation______unexpand__Relation__Comp__1___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Logic_Relation_0__Relation___aux__Mathlib__Logic__Relation______unexpand__Relation__Comp__1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(52, 159, 208, 51, 14, 60, 6, 71)}};
static const lean_object* lp_mathlib___private_Mathlib_Logic_Relation_0__Relation___aux__Mathlib__Logic__Relation______unexpand__Relation__Comp__1___closed__1 = (const lean_object*)&lp_mathlib___private_Mathlib_Logic_Relation_0__Relation___aux__Mathlib__Logic__Relation______unexpand__Relation__Comp__1___closed__1_value;
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Logic_Relation_0__Relation___aux__Mathlib__Logic__Relation______unexpand__Relation__Comp__1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Logic_Relation_0__Relation___aux__Mathlib__Logic__Relation______unexpand__Relation__Comp__1___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Relation_instDecidableMapOfExistsAndEq___redArg(uint8_t);
LEAN_EXPORT lean_object* lp_mathlib_Relation_instDecidableMapOfExistsAndEq___redArg___boxed(lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Relation_instDecidableMapOfExistsAndEq(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, uint8_t);
LEAN_EXPORT lean_object* lp_mathlib_Relation_instDecidableMapOfExistsAndEq___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Relation_SymmGen_decidableRel___aux__1___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Relation_SymmGen_decidableRel___aux__1___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Relation_SymmGen_decidableRel___aux__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Relation_SymmGen_decidableRel___aux__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Relation_SymmGen_decidableRel___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Relation_SymmGen_decidableRel___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Relation_SymmGen_decidableRel(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Relation_SymmGen_decidableRel___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Relation_instTransTransGen__mathlib(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Relation_instTransTransGen__mathlib__1(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Relation_instTransTransGenReflTransGen(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Relation_instTransReflTransGenTransGen(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Relation_instTransReflTransGen(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Relation_instTransReflTransGen__1(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Relation_EqvGen_setoid(lean_object*, lean_object*);
static lean_object* _init_lp_mathlib___private_Mathlib_Logic_Relation_0__Relation___aux__Mathlib__Logic__Relation______macroRules____private__Mathlib__Logic__Relation__0__Relation__term___u2218r____1___closed__6(void){
_start:
{
lean_object* v___x_59_; lean_object* v___x_60_; 
v___x_59_ = ((lean_object*)(lp_mathlib___private_Mathlib_Logic_Relation_0__Relation___aux__Mathlib__Logic__Relation______macroRules____private__Mathlib__Logic__Relation__0__Relation__term___u2218r____1___closed__5));
v___x_60_ = l_String_toRawSubstring_x27(v___x_59_);
return v___x_60_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Logic_Relation_0__Relation___aux__Mathlib__Logic__Relation______macroRules____private__Mathlib__Logic__Relation__0__Relation__term___u2218r____1(lean_object* v_x_74_, lean_object* v_a_75_, lean_object* v_a_76_){
_start:
{
lean_object* v___x_77_; lean_object* v___x_78_; uint8_t v___x_79_; 
v___x_77_ = lean_unsigned_to_nat(0u);
v___x_78_ = ((lean_object*)(lp_mathlib___private_Mathlib_Logic_Relation_0__Relation_term___u2218r___00__closed__11));
lean_inc(v_x_74_);
v___x_79_ = l_Lean_Syntax_isOfKind(v_x_74_, v___x_78_);
if (v___x_79_ == 0)
{
lean_object* v___x_80_; lean_object* v___x_81_; 
lean_dec(v_x_74_);
v___x_80_ = lean_box(1);
v___x_81_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_81_, 0, v___x_80_);
lean_ctor_set(v___x_81_, 1, v_a_76_);
return v___x_81_;
}
else
{
lean_object* v_quotContext_82_; lean_object* v_currMacroScope_83_; lean_object* v_ref_84_; lean_object* v___x_85_; lean_object* v___x_86_; lean_object* v___x_87_; uint8_t v___x_88_; lean_object* v___x_89_; lean_object* v___x_90_; lean_object* v___x_91_; lean_object* v___x_92_; lean_object* v___x_93_; lean_object* v___x_94_; lean_object* v___x_95_; lean_object* v___x_96_; lean_object* v___x_97_; lean_object* v___x_98_; lean_object* v___x_99_; 
v_quotContext_82_ = lean_ctor_get(v_a_75_, 1);
v_currMacroScope_83_ = lean_ctor_get(v_a_75_, 2);
v_ref_84_ = lean_ctor_get(v_a_75_, 5);
v___x_85_ = l_Lean_Syntax_getArg(v_x_74_, v___x_77_);
v___x_86_ = lean_unsigned_to_nat(2u);
v___x_87_ = l_Lean_Syntax_getArg(v_x_74_, v___x_86_);
lean_dec(v_x_74_);
v___x_88_ = 0;
v___x_89_ = l_Lean_SourceInfo_fromRef(v_ref_84_, v___x_88_);
v___x_90_ = ((lean_object*)(lp_mathlib___private_Mathlib_Logic_Relation_0__Relation___aux__Mathlib__Logic__Relation______macroRules____private__Mathlib__Logic__Relation__0__Relation__term___u2218r____1___closed__4));
v___x_91_ = lean_obj_once(&lp_mathlib___private_Mathlib_Logic_Relation_0__Relation___aux__Mathlib__Logic__Relation______macroRules____private__Mathlib__Logic__Relation__0__Relation__term___u2218r____1___closed__6, &lp_mathlib___private_Mathlib_Logic_Relation_0__Relation___aux__Mathlib__Logic__Relation______macroRules____private__Mathlib__Logic__Relation__0__Relation__term___u2218r____1___closed__6_once, _init_lp_mathlib___private_Mathlib_Logic_Relation_0__Relation___aux__Mathlib__Logic__Relation______macroRules____private__Mathlib__Logic__Relation__0__Relation__term___u2218r____1___closed__6);
v___x_92_ = ((lean_object*)(lp_mathlib___private_Mathlib_Logic_Relation_0__Relation___aux__Mathlib__Logic__Relation______macroRules____private__Mathlib__Logic__Relation__0__Relation__term___u2218r____1___closed__8));
lean_inc(v_currMacroScope_83_);
lean_inc(v_quotContext_82_);
v___x_93_ = l_Lean_addMacroScope(v_quotContext_82_, v___x_92_, v_currMacroScope_83_);
v___x_94_ = ((lean_object*)(lp_mathlib___private_Mathlib_Logic_Relation_0__Relation___aux__Mathlib__Logic__Relation______macroRules____private__Mathlib__Logic__Relation__0__Relation__term___u2218r____1___closed__10));
lean_inc_n(v___x_89_, 2);
v___x_95_ = lean_alloc_ctor(3, 4, 0);
lean_ctor_set(v___x_95_, 0, v___x_89_);
lean_ctor_set(v___x_95_, 1, v___x_91_);
lean_ctor_set(v___x_95_, 2, v___x_93_);
lean_ctor_set(v___x_95_, 3, v___x_94_);
v___x_96_ = ((lean_object*)(lp_mathlib___private_Mathlib_Logic_Relation_0__Relation___aux__Mathlib__Logic__Relation______macroRules____private__Mathlib__Logic__Relation__0__Relation__term___u2218r____1___closed__12));
v___x_97_ = l_Lean_Syntax_node2(v___x_89_, v___x_96_, v___x_85_, v___x_87_);
v___x_98_ = l_Lean_Syntax_node2(v___x_89_, v___x_90_, v___x_95_, v___x_97_);
v___x_99_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_99_, 0, v___x_98_);
lean_ctor_set(v___x_99_, 1, v_a_76_);
return v___x_99_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Logic_Relation_0__Relation___aux__Mathlib__Logic__Relation______macroRules____private__Mathlib__Logic__Relation__0__Relation__term___u2218r____1___boxed(lean_object* v_x_100_, lean_object* v_a_101_, lean_object* v_a_102_){
_start:
{
lean_object* v_res_103_; 
v_res_103_ = lp_mathlib___private_Mathlib_Logic_Relation_0__Relation___aux__Mathlib__Logic__Relation______macroRules____private__Mathlib__Logic__Relation__0__Relation__term___u2218r____1(v_x_100_, v_a_101_, v_a_102_);
lean_dec_ref(v_a_101_);
return v_res_103_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Logic_Relation_0__Relation___aux__Mathlib__Logic__Relation______unexpand__Relation__Comp__1(lean_object* v_x_107_, lean_object* v_a_108_, lean_object* v_a_109_){
_start:
{
lean_object* v___x_110_; uint8_t v___x_111_; 
v___x_110_ = ((lean_object*)(lp_mathlib___private_Mathlib_Logic_Relation_0__Relation___aux__Mathlib__Logic__Relation______macroRules____private__Mathlib__Logic__Relation__0__Relation__term___u2218r____1___closed__4));
lean_inc(v_x_107_);
v___x_111_ = l_Lean_Syntax_isOfKind(v_x_107_, v___x_110_);
if (v___x_111_ == 0)
{
lean_object* v___x_112_; lean_object* v___x_113_; 
lean_dec(v_x_107_);
v___x_112_ = lean_box(0);
v___x_113_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_113_, 0, v___x_112_);
lean_ctor_set(v___x_113_, 1, v_a_109_);
return v___x_113_;
}
else
{
lean_object* v___x_114_; lean_object* v___x_115_; lean_object* v___x_116_; uint8_t v___x_117_; 
v___x_114_ = lean_unsigned_to_nat(0u);
v___x_115_ = l_Lean_Syntax_getArg(v_x_107_, v___x_114_);
v___x_116_ = ((lean_object*)(lp_mathlib___private_Mathlib_Logic_Relation_0__Relation___aux__Mathlib__Logic__Relation______unexpand__Relation__Comp__1___closed__1));
lean_inc(v___x_115_);
v___x_117_ = l_Lean_Syntax_isOfKind(v___x_115_, v___x_116_);
if (v___x_117_ == 0)
{
lean_object* v___x_118_; lean_object* v___x_119_; 
lean_dec(v___x_115_);
lean_dec(v_x_107_);
v___x_118_ = lean_box(0);
v___x_119_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_119_, 0, v___x_118_);
lean_ctor_set(v___x_119_, 1, v_a_109_);
return v___x_119_;
}
else
{
lean_object* v___x_120_; lean_object* v___x_121_; lean_object* v___x_122_; uint8_t v___x_123_; 
v___x_120_ = lean_unsigned_to_nat(1u);
v___x_121_ = l_Lean_Syntax_getArg(v_x_107_, v___x_120_);
lean_dec(v_x_107_);
v___x_122_ = lean_unsigned_to_nat(2u);
lean_inc(v___x_121_);
v___x_123_ = l_Lean_Syntax_matchesNull(v___x_121_, v___x_122_);
if (v___x_123_ == 0)
{
lean_object* v___x_124_; lean_object* v___x_125_; 
lean_dec(v___x_121_);
lean_dec(v___x_115_);
v___x_124_ = lean_box(0);
v___x_125_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_125_, 0, v___x_124_);
lean_ctor_set(v___x_125_, 1, v_a_109_);
return v___x_125_;
}
else
{
lean_object* v___x_126_; lean_object* v___x_127_; lean_object* v_ref_128_; uint8_t v___x_129_; lean_object* v___x_130_; lean_object* v___x_131_; lean_object* v___x_132_; lean_object* v___x_133_; lean_object* v___x_134_; lean_object* v___x_135_; 
v___x_126_ = l_Lean_Syntax_getArg(v___x_121_, v___x_114_);
v___x_127_ = l_Lean_Syntax_getArg(v___x_121_, v___x_120_);
lean_dec(v___x_121_);
v_ref_128_ = l_Lean_replaceRef(v___x_115_, v_a_108_);
lean_dec(v___x_115_);
v___x_129_ = 0;
v___x_130_ = l_Lean_SourceInfo_fromRef(v_ref_128_, v___x_129_);
lean_dec(v_ref_128_);
v___x_131_ = ((lean_object*)(lp_mathlib___private_Mathlib_Logic_Relation_0__Relation_term___u2218r___00__closed__11));
v___x_132_ = ((lean_object*)(lp_mathlib___private_Mathlib_Logic_Relation_0__Relation_term___u2218r___00__closed__14));
lean_inc(v___x_130_);
v___x_133_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_133_, 0, v___x_130_);
lean_ctor_set(v___x_133_, 1, v___x_132_);
v___x_134_ = l_Lean_Syntax_node3(v___x_130_, v___x_131_, v___x_126_, v___x_133_, v___x_127_);
v___x_135_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_135_, 0, v___x_134_);
lean_ctor_set(v___x_135_, 1, v_a_109_);
return v___x_135_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Logic_Relation_0__Relation___aux__Mathlib__Logic__Relation______unexpand__Relation__Comp__1___boxed(lean_object* v_x_136_, lean_object* v_a_137_, lean_object* v_a_138_){
_start:
{
lean_object* v_res_139_; 
v_res_139_ = lp_mathlib___private_Mathlib_Logic_Relation_0__Relation___aux__Mathlib__Logic__Relation______unexpand__Relation__Comp__1(v_x_136_, v_a_137_, v_a_138_);
lean_dec(v_a_137_);
return v_res_139_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Relation_instDecidableMapOfExistsAndEq___redArg(uint8_t v_inst_140_){
_start:
{
return v_inst_140_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Relation_instDecidableMapOfExistsAndEq___redArg___boxed(lean_object* v_inst_141_){
_start:
{
uint8_t v_inst_5__boxed_142_; uint8_t v_res_143_; lean_object* v_r_144_; 
v_inst_5__boxed_142_ = lean_unbox(v_inst_141_);
v_res_143_ = lp_mathlib_Relation_instDecidableMapOfExistsAndEq___redArg(v_inst_5__boxed_142_);
v_r_144_ = lean_box(v_res_143_);
return v_r_144_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Relation_instDecidableMapOfExistsAndEq(lean_object* v_00_u03b1_145_, lean_object* v_00_u03b2_146_, lean_object* v_00_u03b3_147_, lean_object* v_00_u03b4_148_, lean_object* v_r_149_, lean_object* v_f_150_, lean_object* v_g_151_, lean_object* v_c_152_, lean_object* v_d_153_, uint8_t v_inst_154_){
_start:
{
return v_inst_154_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Relation_instDecidableMapOfExistsAndEq___boxed(lean_object* v_00_u03b1_155_, lean_object* v_00_u03b2_156_, lean_object* v_00_u03b3_157_, lean_object* v_00_u03b4_158_, lean_object* v_r_159_, lean_object* v_f_160_, lean_object* v_g_161_, lean_object* v_c_162_, lean_object* v_d_163_, lean_object* v_inst_164_){
_start:
{
uint8_t v_inst_8__boxed_165_; uint8_t v_res_166_; lean_object* v_r_167_; 
v_inst_8__boxed_165_ = lean_unbox(v_inst_164_);
v_res_166_ = lp_mathlib_Relation_instDecidableMapOfExistsAndEq(v_00_u03b1_155_, v_00_u03b2_156_, v_00_u03b3_157_, v_00_u03b4_158_, v_r_159_, v_f_160_, v_g_161_, v_c_162_, v_d_163_, v_inst_8__boxed_165_);
lean_dec(v_d_163_);
lean_dec(v_c_162_);
lean_dec(v_g_161_);
lean_dec(v_f_160_);
v_r_167_ = lean_box(v_res_166_);
return v_r_167_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Relation_SymmGen_decidableRel___aux__1___redArg(lean_object* v_inst_168_, lean_object* v_x_169_, lean_object* v_x_170_){
_start:
{
lean_object* v___x_171_; lean_object* v___x_172_; uint8_t v___x_173_; 
lean_inc_ref(v_inst_168_);
lean_inc(v_x_169_);
lean_inc(v_x_170_);
v___x_171_ = lean_apply_2(v_inst_168_, v_x_170_, v_x_169_);
v___x_172_ = lean_apply_2(v_inst_168_, v_x_169_, v_x_170_);
v___x_173_ = lean_unbox(v___x_172_);
if (v___x_173_ == 0)
{
uint8_t v___x_174_; 
v___x_174_ = lean_unbox(v___x_171_);
return v___x_174_;
}
else
{
uint8_t v___x_175_; 
v___x_175_ = lean_unbox(v___x_172_);
return v___x_175_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Relation_SymmGen_decidableRel___aux__1___redArg___boxed(lean_object* v_inst_176_, lean_object* v_x_177_, lean_object* v_x_178_){
_start:
{
uint8_t v_res_179_; lean_object* v_r_180_; 
v_res_179_ = lp_mathlib_Relation_SymmGen_decidableRel___aux__1___redArg(v_inst_176_, v_x_177_, v_x_178_);
v_r_180_ = lean_box(v_res_179_);
return v_r_180_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Relation_SymmGen_decidableRel___aux__1(lean_object* v_00_u03b1_181_, lean_object* v_r_182_, lean_object* v_inst_183_, lean_object* v_x_184_, lean_object* v_x_185_){
_start:
{
uint8_t v___x_186_; 
v___x_186_ = lp_mathlib_Relation_SymmGen_decidableRel___aux__1___redArg(v_inst_183_, v_x_184_, v_x_185_);
return v___x_186_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Relation_SymmGen_decidableRel___aux__1___boxed(lean_object* v_00_u03b1_187_, lean_object* v_r_188_, lean_object* v_inst_189_, lean_object* v_x_190_, lean_object* v_x_191_){
_start:
{
uint8_t v_res_192_; lean_object* v_r_193_; 
v_res_192_ = lp_mathlib_Relation_SymmGen_decidableRel___aux__1(v_00_u03b1_187_, v_r_188_, v_inst_189_, v_x_190_, v_x_191_);
v_r_193_ = lean_box(v_res_192_);
return v_r_193_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Relation_SymmGen_decidableRel___redArg(lean_object* v_inst_194_, lean_object* v_x_195_, lean_object* v_x_196_){
_start:
{
uint8_t v___x_197_; 
v___x_197_ = lp_mathlib_Relation_SymmGen_decidableRel___aux__1___redArg(v_inst_194_, v_x_195_, v_x_196_);
return v___x_197_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Relation_SymmGen_decidableRel___redArg___boxed(lean_object* v_inst_198_, lean_object* v_x_199_, lean_object* v_x_200_){
_start:
{
uint8_t v_res_201_; lean_object* v_r_202_; 
v_res_201_ = lp_mathlib_Relation_SymmGen_decidableRel___redArg(v_inst_198_, v_x_199_, v_x_200_);
v_r_202_ = lean_box(v_res_201_);
return v_r_202_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Relation_SymmGen_decidableRel(lean_object* v_00_u03b1_203_, lean_object* v_r_204_, lean_object* v_inst_205_, lean_object* v_x_206_, lean_object* v_x_207_){
_start:
{
uint8_t v___x_208_; 
v___x_208_ = lp_mathlib_Relation_SymmGen_decidableRel___aux__1___redArg(v_inst_205_, v_x_206_, v_x_207_);
return v___x_208_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Relation_SymmGen_decidableRel___boxed(lean_object* v_00_u03b1_209_, lean_object* v_r_210_, lean_object* v_inst_211_, lean_object* v_x_212_, lean_object* v_x_213_){
_start:
{
uint8_t v_res_214_; lean_object* v_r_215_; 
v_res_214_ = lp_mathlib_Relation_SymmGen_decidableRel(v_00_u03b1_209_, v_r_210_, v_inst_211_, v_x_212_, v_x_213_);
v_r_215_ = lean_box(v_res_214_);
return v_r_215_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Relation_instTransTransGen__mathlib(lean_object* v_00_u03b1_216_, lean_object* v_r_217_){
_start:
{
lean_object* v___x_218_; 
v___x_218_ = lean_box(0);
return v___x_218_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Relation_instTransTransGen__mathlib__1(lean_object* v_00_u03b1_219_, lean_object* v_r_220_){
_start:
{
lean_object* v___x_221_; 
v___x_221_ = lean_box(0);
return v___x_221_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Relation_instTransTransGenReflTransGen(lean_object* v_00_u03b1_222_, lean_object* v_r_223_){
_start:
{
lean_object* v___x_224_; 
v___x_224_ = lean_box(0);
return v___x_224_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Relation_instTransReflTransGenTransGen(lean_object* v_00_u03b1_225_, lean_object* v_r_226_){
_start:
{
lean_object* v___x_227_; 
v___x_227_ = lean_box(0);
return v___x_227_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Relation_instTransReflTransGen(lean_object* v_00_u03b1_228_, lean_object* v_r_229_){
_start:
{
lean_object* v___x_230_; 
v___x_230_ = lean_box(0);
return v___x_230_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Relation_instTransReflTransGen__1(lean_object* v_00_u03b1_231_, lean_object* v_r_232_){
_start:
{
lean_object* v___x_233_; 
v___x_233_ = lean_box(0);
return v___x_233_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Relation_EqvGen_setoid(lean_object* v_00_u03b1_234_, lean_object* v_r_235_){
_start:
{
lean_object* v___x_236_; 
v___x_236_ = lean_box(0);
return v___x_236_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Logic_Relator(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Tactic_Use(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Tactic_MkIffOfInductiveProp(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Tactic_SimpRw(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Order_Defs_Prop(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Order_Defs_Unbundled(uint8_t builtin);
lean_object* runtime_initialize_batteries_Batteries_Logic(uint8_t builtin);
lean_object* runtime_initialize_batteries_Batteries_Tactic_Trans(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_Logic_Relation(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Logic_Relator(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Tactic_Use(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Tactic_MkIffOfInductiveProp(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Tactic_SimpRw(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Order_Defs_Prop(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Order_Defs_Unbundled(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_batteries_Batteries_Logic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_batteries_Batteries_Tactic_Trans(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_Logic_Relation(uint8_t builtin) {
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
lean_object* initialize_mathlib_Mathlib_Logic_Relator(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Tactic_Use(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Tactic_MkIffOfInductiveProp(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Tactic_SimpRw(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Order_Defs_Prop(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Order_Defs_Unbundled(uint8_t builtin);
lean_object* initialize_batteries_Batteries_Logic(uint8_t builtin);
lean_object* initialize_batteries_Batteries_Tactic_Trans(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Logic_Relation(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Logic_Relator(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Tactic_Use(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Tactic_MkIffOfInductiveProp(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Tactic_SimpRw(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Order_Defs_Prop(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Order_Defs_Unbundled(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_batteries_Batteries_Logic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_batteries_Batteries_Tactic_Trans(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Logic_Relation(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_Logic_Relation(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_Logic_Relation(builtin);
}
#ifdef __cplusplus
}
#endif
