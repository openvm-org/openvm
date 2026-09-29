// Lean compiler output
// Module: Mathlib.Basic.ExistsUnique
// Imports: public import Init public meta import Init public import Mathlib.Init
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
lean_object* l_Lean_Name_mkStr3(lean_object*, lean_object*, lean_object*);
uint8_t l_Lean_Syntax_isOfKind(lean_object*, lean_object*);
lean_object* l_Lean_Syntax_getArg(lean_object*, lean_object*);
lean_object* l_Lean_expandExplicitBinders(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Name_mkStr2(lean_object*, lean_object*);
uint8_t l_Lean_Syntax_matchesNull(lean_object*, lean_object*);
uint8_t l_Lean_Syntax_isNone(lean_object*);
lean_object* l_Lean_Macro_throwErrorAt___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
uint8_t l_List_decidableBAll___redArg(lean_object*, lean_object*);
lean_object* l_Lean_Name_mkStr4(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_SourceInfo_fromRef(lean_object*, uint8_t);
lean_object* l_String_toRawSubstring_x27(lean_object*);
lean_object* l_Lean_addMacroScope(lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Syntax_node1(lean_object*, lean_object*, lean_object*);
lean_object* l_Array_mkArray0(lean_object*);
lean_object* l_Lean_Syntax_node2(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Syntax_node3(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Syntax_node4(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
extern lean_object* l_Lean_binderIdent;
extern lean_object* l_Lean_explicitBinders;
uint8_t l_Lean_Syntax_matchesIdent(lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Mathlib_Notation_isExplicitBinderSingular___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "Lean"};
static const lean_object* lp_mathlib_Mathlib_Notation_isExplicitBinderSingular___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Notation_isExplicitBinderSingular___closed__0_value;
static const lean_string_object lp_mathlib_Mathlib_Notation_isExplicitBinderSingular___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 16, .m_capacity = 16, .m_length = 15, .m_data = "explicitBinders"};
static const lean_object* lp_mathlib_Mathlib_Notation_isExplicitBinderSingular___closed__1 = (const lean_object*)&lp_mathlib_Mathlib_Notation_isExplicitBinderSingular___closed__1_value;
static const lean_ctor_object lp_mathlib_Mathlib_Notation_isExplicitBinderSingular___closed__2_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Notation_isExplicitBinderSingular___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib_Mathlib_Notation_isExplicitBinderSingular___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Notation_isExplicitBinderSingular___closed__2_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Notation_isExplicitBinderSingular___closed__1_value),LEAN_SCALAR_PTR_LITERAL(167, 149, 127, 13, 202, 239, 226, 94)}};
static const lean_object* lp_mathlib_Mathlib_Notation_isExplicitBinderSingular___closed__2 = (const lean_object*)&lp_mathlib_Mathlib_Notation_isExplicitBinderSingular___closed__2_value;
static const lean_string_object lp_mathlib_Mathlib_Notation_isExplicitBinderSingular___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 27, .m_capacity = 27, .m_length = 26, .m_data = "unbracketedExplicitBinders"};
static const lean_object* lp_mathlib_Mathlib_Notation_isExplicitBinderSingular___closed__3 = (const lean_object*)&lp_mathlib_Mathlib_Notation_isExplicitBinderSingular___closed__3_value;
static const lean_ctor_object lp_mathlib_Mathlib_Notation_isExplicitBinderSingular___closed__4_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Notation_isExplicitBinderSingular___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib_Mathlib_Notation_isExplicitBinderSingular___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Notation_isExplicitBinderSingular___closed__4_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Notation_isExplicitBinderSingular___closed__3_value),LEAN_SCALAR_PTR_LITERAL(187, 220, 119, 82, 242, 112, 119, 200)}};
static const lean_object* lp_mathlib_Mathlib_Notation_isExplicitBinderSingular___closed__4 = (const lean_object*)&lp_mathlib_Mathlib_Notation_isExplicitBinderSingular___closed__4_value;
static const lean_string_object lp_mathlib_Mathlib_Notation_isExplicitBinderSingular___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 25, .m_capacity = 25, .m_length = 24, .m_data = "bracketedExplicitBinders"};
static const lean_object* lp_mathlib_Mathlib_Notation_isExplicitBinderSingular___closed__5 = (const lean_object*)&lp_mathlib_Mathlib_Notation_isExplicitBinderSingular___closed__5_value;
static const lean_ctor_object lp_mathlib_Mathlib_Notation_isExplicitBinderSingular___closed__6_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Notation_isExplicitBinderSingular___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib_Mathlib_Notation_isExplicitBinderSingular___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Notation_isExplicitBinderSingular___closed__6_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Notation_isExplicitBinderSingular___closed__5_value),LEAN_SCALAR_PTR_LITERAL(22, 65, 7, 186, 44, 89, 152, 79)}};
static const lean_object* lp_mathlib_Mathlib_Notation_isExplicitBinderSingular___closed__6 = (const lean_object*)&lp_mathlib_Mathlib_Notation_isExplicitBinderSingular___closed__6_value;
static const lean_string_object lp_mathlib_Mathlib_Notation_isExplicitBinderSingular___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 12, .m_capacity = 12, .m_length = 11, .m_data = "binderIdent"};
static const lean_object* lp_mathlib_Mathlib_Notation_isExplicitBinderSingular___closed__7 = (const lean_object*)&lp_mathlib_Mathlib_Notation_isExplicitBinderSingular___closed__7_value;
static const lean_ctor_object lp_mathlib_Mathlib_Notation_isExplicitBinderSingular___closed__8_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Notation_isExplicitBinderSingular___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib_Mathlib_Notation_isExplicitBinderSingular___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Notation_isExplicitBinderSingular___closed__8_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Notation_isExplicitBinderSingular___closed__7_value),LEAN_SCALAR_PTR_LITERAL(37, 194, 68, 106, 254, 181, 31, 191)}};
static const lean_object* lp_mathlib_Mathlib_Notation_isExplicitBinderSingular___closed__8 = (const lean_object*)&lp_mathlib_Mathlib_Notation_isExplicitBinderSingular___closed__8_value;
LEAN_EXPORT uint8_t lp_mathlib_Mathlib_Notation_isExplicitBinderSingular(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Notation_isExplicitBinderSingular___boxed(lean_object*);
static const lean_string_object lp_mathlib_Mathlib_Notation_term_u2203_x21___x2c___00__closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "Mathlib"};
static const lean_object* lp_mathlib_Mathlib_Notation_term_u2203_x21___x2c___00__closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Notation_term_u2203_x21___x2c___00__closed__0_value;
static const lean_string_object lp_mathlib_Mathlib_Notation_term_u2203_x21___x2c___00__closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "Notation"};
static const lean_object* lp_mathlib_Mathlib_Notation_term_u2203_x21___x2c___00__closed__1 = (const lean_object*)&lp_mathlib_Mathlib_Notation_term_u2203_x21___x2c___00__closed__1_value;
static const lean_string_object lp_mathlib_Mathlib_Notation_term_u2203_x21___x2c___00__closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 12, .m_capacity = 12, .m_length = 9, .m_data = "term∃!_,_"};
static const lean_object* lp_mathlib_Mathlib_Notation_term_u2203_x21___x2c___00__closed__2 = (const lean_object*)&lp_mathlib_Mathlib_Notation_term_u2203_x21___x2c___00__closed__2_value;
static const lean_ctor_object lp_mathlib_Mathlib_Notation_term_u2203_x21___x2c___00__closed__3_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Notation_term_u2203_x21___x2c___00__closed__0_value),LEAN_SCALAR_PTR_LITERAL(118, 213, 161, 2, 73, 184, 31, 228)}};
static const lean_ctor_object lp_mathlib_Mathlib_Notation_term_u2203_x21___x2c___00__closed__3_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Notation_term_u2203_x21___x2c___00__closed__3_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Notation_term_u2203_x21___x2c___00__closed__1_value),LEAN_SCALAR_PTR_LITERAL(6, 103, 244, 232, 62, 27, 250, 92)}};
static const lean_ctor_object lp_mathlib_Mathlib_Notation_term_u2203_x21___x2c___00__closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Notation_term_u2203_x21___x2c___00__closed__3_value_aux_1),((lean_object*)&lp_mathlib_Mathlib_Notation_term_u2203_x21___x2c___00__closed__2_value),LEAN_SCALAR_PTR_LITERAL(15, 215, 94, 62, 48, 9, 239, 228)}};
static const lean_object* lp_mathlib_Mathlib_Notation_term_u2203_x21___x2c___00__closed__3 = (const lean_object*)&lp_mathlib_Mathlib_Notation_term_u2203_x21___x2c___00__closed__3_value;
static const lean_string_object lp_mathlib_Mathlib_Notation_term_u2203_x21___x2c___00__closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "andthen"};
static const lean_object* lp_mathlib_Mathlib_Notation_term_u2203_x21___x2c___00__closed__4 = (const lean_object*)&lp_mathlib_Mathlib_Notation_term_u2203_x21___x2c___00__closed__4_value;
static const lean_ctor_object lp_mathlib_Mathlib_Notation_term_u2203_x21___x2c___00__closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Notation_term_u2203_x21___x2c___00__closed__4_value),LEAN_SCALAR_PTR_LITERAL(40, 255, 78, 30, 143, 119, 117, 174)}};
static const lean_object* lp_mathlib_Mathlib_Notation_term_u2203_x21___x2c___00__closed__5 = (const lean_object*)&lp_mathlib_Mathlib_Notation_term_u2203_x21___x2c___00__closed__5_value;
static const lean_string_object lp_mathlib_Mathlib_Notation_term_u2203_x21___x2c___00__closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 2, .m_data = "∃!"};
static const lean_object* lp_mathlib_Mathlib_Notation_term_u2203_x21___x2c___00__closed__6 = (const lean_object*)&lp_mathlib_Mathlib_Notation_term_u2203_x21___x2c___00__closed__6_value;
static const lean_ctor_object lp_mathlib_Mathlib_Notation_term_u2203_x21___x2c___00__closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 5}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Notation_term_u2203_x21___x2c___00__closed__6_value)}};
static const lean_object* lp_mathlib_Mathlib_Notation_term_u2203_x21___x2c___00__closed__7 = (const lean_object*)&lp_mathlib_Mathlib_Notation_term_u2203_x21___x2c___00__closed__7_value;
static lean_once_cell_t lp_mathlib_Mathlib_Notation_term_u2203_x21___x2c___00__closed__8_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Notation_term_u2203_x21___x2c___00__closed__8;
static const lean_string_object lp_mathlib_Mathlib_Notation_term_u2203_x21___x2c___00__closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 3, .m_capacity = 3, .m_length = 2, .m_data = ", "};
static const lean_object* lp_mathlib_Mathlib_Notation_term_u2203_x21___x2c___00__closed__9 = (const lean_object*)&lp_mathlib_Mathlib_Notation_term_u2203_x21___x2c___00__closed__9_value;
static const lean_ctor_object lp_mathlib_Mathlib_Notation_term_u2203_x21___x2c___00__closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 5}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Notation_term_u2203_x21___x2c___00__closed__9_value)}};
static const lean_object* lp_mathlib_Mathlib_Notation_term_u2203_x21___x2c___00__closed__10 = (const lean_object*)&lp_mathlib_Mathlib_Notation_term_u2203_x21___x2c___00__closed__10_value;
static lean_once_cell_t lp_mathlib_Mathlib_Notation_term_u2203_x21___x2c___00__closed__11_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Notation_term_u2203_x21___x2c___00__closed__11;
static const lean_string_object lp_mathlib_Mathlib_Notation_term_u2203_x21___x2c___00__closed__12_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "term"};
static const lean_object* lp_mathlib_Mathlib_Notation_term_u2203_x21___x2c___00__closed__12 = (const lean_object*)&lp_mathlib_Mathlib_Notation_term_u2203_x21___x2c___00__closed__12_value;
static const lean_ctor_object lp_mathlib_Mathlib_Notation_term_u2203_x21___x2c___00__closed__13_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Notation_term_u2203_x21___x2c___00__closed__12_value),LEAN_SCALAR_PTR_LITERAL(187, 230, 181, 162, 253, 146, 122, 119)}};
static const lean_object* lp_mathlib_Mathlib_Notation_term_u2203_x21___x2c___00__closed__13 = (const lean_object*)&lp_mathlib_Mathlib_Notation_term_u2203_x21___x2c___00__closed__13_value;
static const lean_ctor_object lp_mathlib_Mathlib_Notation_term_u2203_x21___x2c___00__closed__14_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 7}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Notation_term_u2203_x21___x2c___00__closed__13_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_Mathlib_Notation_term_u2203_x21___x2c___00__closed__14 = (const lean_object*)&lp_mathlib_Mathlib_Notation_term_u2203_x21___x2c___00__closed__14_value;
static lean_once_cell_t lp_mathlib_Mathlib_Notation_term_u2203_x21___x2c___00__closed__15_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Notation_term_u2203_x21___x2c___00__closed__15;
static lean_once_cell_t lp_mathlib_Mathlib_Notation_term_u2203_x21___x2c___00__closed__16_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Notation_term_u2203_x21___x2c___00__closed__16;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Notation_term_u2203_x21___x2c__;
static const lean_string_object lp_mathlib_Mathlib_Notation___aux__Mathlib__Basic__ExistsUnique______macroRules__Mathlib__Notation__term_u2203_x21___x2c____1___lam__0___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 13, .m_capacity = 13, .m_length = 12, .m_data = "ExistsUnique"};
static const lean_object* lp_mathlib_Mathlib_Notation___aux__Mathlib__Basic__ExistsUnique______macroRules__Mathlib__Notation__term_u2203_x21___x2c____1___lam__0___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Notation___aux__Mathlib__Basic__ExistsUnique______macroRules__Mathlib__Notation__term_u2203_x21___x2c____1___lam__0___closed__0_value;
static const lean_ctor_object lp_mathlib_Mathlib_Notation___aux__Mathlib__Basic__ExistsUnique______macroRules__Mathlib__Notation__term_u2203_x21___x2c____1___lam__0___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Notation___aux__Mathlib__Basic__ExistsUnique______macroRules__Mathlib__Notation__term_u2203_x21___x2c____1___lam__0___closed__0_value),LEAN_SCALAR_PTR_LITERAL(186, 203, 5, 233, 1, 71, 199, 158)}};
static const lean_object* lp_mathlib_Mathlib_Notation___aux__Mathlib__Basic__ExistsUnique______macroRules__Mathlib__Notation__term_u2203_x21___x2c____1___lam__0___closed__1 = (const lean_object*)&lp_mathlib_Mathlib_Notation___aux__Mathlib__Basic__ExistsUnique______macroRules__Mathlib__Notation__term_u2203_x21___x2c____1___lam__0___closed__1_value;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Notation___aux__Mathlib__Basic__ExistsUnique______macroRules__Mathlib__Notation__term_u2203_x21___x2c____1___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Notation___aux__Mathlib__Basic__ExistsUnique______macroRules__Mathlib__Notation__term_u2203_x21___x2c____1___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Mathlib_Notation___aux__Mathlib__Basic__ExistsUnique______macroRules__Mathlib__Notation__term_u2203_x21___x2c____1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 324, .m_capacity = 324, .m_length = 312, .m_data = "The `ExistsUnique` notation should not be used with more than one binder.\n\nThe reason for this is that `∃! (x : α), ∃! (y : β), p x y` has a completely different meaning from `∃! q : α × β, p q.1 q.2`. To prevent confusion, this notation requires that you be explicit and use one with the correct interpretation."};
static const lean_object* lp_mathlib_Mathlib_Notation___aux__Mathlib__Basic__ExistsUnique______macroRules__Mathlib__Notation__term_u2203_x21___x2c____1___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Notation___aux__Mathlib__Basic__ExistsUnique______macroRules__Mathlib__Notation__term_u2203_x21___x2c____1___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Notation___aux__Mathlib__Basic__ExistsUnique______macroRules__Mathlib__Notation__term_u2203_x21___x2c____1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Notation___aux__Mathlib__Basic__ExistsUnique______macroRules__Mathlib__Notation__term_u2203_x21___x2c____1___boxed(lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Mathlib_Notation_unexpandExistsUnique___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "Parser"};
static const lean_object* lp_mathlib_Mathlib_Notation_unexpandExistsUnique___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Notation_unexpandExistsUnique___closed__0_value;
static const lean_string_object lp_mathlib_Mathlib_Notation_unexpandExistsUnique___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "Term"};
static const lean_object* lp_mathlib_Mathlib_Notation_unexpandExistsUnique___closed__1 = (const lean_object*)&lp_mathlib_Mathlib_Notation_unexpandExistsUnique___closed__1_value;
static const lean_string_object lp_mathlib_Mathlib_Notation_unexpandExistsUnique___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = "app"};
static const lean_object* lp_mathlib_Mathlib_Notation_unexpandExistsUnique___closed__2 = (const lean_object*)&lp_mathlib_Mathlib_Notation_unexpandExistsUnique___closed__2_value;
static const lean_ctor_object lp_mathlib_Mathlib_Notation_unexpandExistsUnique___closed__3_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Notation_isExplicitBinderSingular___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib_Mathlib_Notation_unexpandExistsUnique___closed__3_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Notation_unexpandExistsUnique___closed__3_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Notation_unexpandExistsUnique___closed__0_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib_Mathlib_Notation_unexpandExistsUnique___closed__3_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Notation_unexpandExistsUnique___closed__3_value_aux_1),((lean_object*)&lp_mathlib_Mathlib_Notation_unexpandExistsUnique___closed__1_value),LEAN_SCALAR_PTR_LITERAL(75, 170, 162, 138, 136, 204, 251, 229)}};
static const lean_ctor_object lp_mathlib_Mathlib_Notation_unexpandExistsUnique___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Notation_unexpandExistsUnique___closed__3_value_aux_2),((lean_object*)&lp_mathlib_Mathlib_Notation_unexpandExistsUnique___closed__2_value),LEAN_SCALAR_PTR_LITERAL(69, 118, 10, 41, 220, 156, 243, 179)}};
static const lean_object* lp_mathlib_Mathlib_Notation_unexpandExistsUnique___closed__3 = (const lean_object*)&lp_mathlib_Mathlib_Notation_unexpandExistsUnique___closed__3_value;
static const lean_string_object lp_mathlib_Mathlib_Notation_unexpandExistsUnique___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = "fun"};
static const lean_object* lp_mathlib_Mathlib_Notation_unexpandExistsUnique___closed__4 = (const lean_object*)&lp_mathlib_Mathlib_Notation_unexpandExistsUnique___closed__4_value;
static const lean_ctor_object lp_mathlib_Mathlib_Notation_unexpandExistsUnique___closed__5_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Notation_isExplicitBinderSingular___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib_Mathlib_Notation_unexpandExistsUnique___closed__5_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Notation_unexpandExistsUnique___closed__5_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Notation_unexpandExistsUnique___closed__0_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib_Mathlib_Notation_unexpandExistsUnique___closed__5_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Notation_unexpandExistsUnique___closed__5_value_aux_1),((lean_object*)&lp_mathlib_Mathlib_Notation_unexpandExistsUnique___closed__1_value),LEAN_SCALAR_PTR_LITERAL(75, 170, 162, 138, 136, 204, 251, 229)}};
static const lean_ctor_object lp_mathlib_Mathlib_Notation_unexpandExistsUnique___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Notation_unexpandExistsUnique___closed__5_value_aux_2),((lean_object*)&lp_mathlib_Mathlib_Notation_unexpandExistsUnique___closed__4_value),LEAN_SCALAR_PTR_LITERAL(249, 155, 133, 242, 71, 132, 191, 97)}};
static const lean_object* lp_mathlib_Mathlib_Notation_unexpandExistsUnique___closed__5 = (const lean_object*)&lp_mathlib_Mathlib_Notation_unexpandExistsUnique___closed__5_value;
static const lean_string_object lp_mathlib_Mathlib_Notation_unexpandExistsUnique___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "basicFun"};
static const lean_object* lp_mathlib_Mathlib_Notation_unexpandExistsUnique___closed__6 = (const lean_object*)&lp_mathlib_Mathlib_Notation_unexpandExistsUnique___closed__6_value;
static const lean_ctor_object lp_mathlib_Mathlib_Notation_unexpandExistsUnique___closed__7_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Notation_isExplicitBinderSingular___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib_Mathlib_Notation_unexpandExistsUnique___closed__7_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Notation_unexpandExistsUnique___closed__7_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Notation_unexpandExistsUnique___closed__0_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib_Mathlib_Notation_unexpandExistsUnique___closed__7_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Notation_unexpandExistsUnique___closed__7_value_aux_1),((lean_object*)&lp_mathlib_Mathlib_Notation_unexpandExistsUnique___closed__1_value),LEAN_SCALAR_PTR_LITERAL(75, 170, 162, 138, 136, 204, 251, 229)}};
static const lean_ctor_object lp_mathlib_Mathlib_Notation_unexpandExistsUnique___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Notation_unexpandExistsUnique___closed__7_value_aux_2),((lean_object*)&lp_mathlib_Mathlib_Notation_unexpandExistsUnique___closed__6_value),LEAN_SCALAR_PTR_LITERAL(209, 134, 40, 160, 122, 195, 31, 223)}};
static const lean_object* lp_mathlib_Mathlib_Notation_unexpandExistsUnique___closed__7 = (const lean_object*)&lp_mathlib_Mathlib_Notation_unexpandExistsUnique___closed__7_value;
static const lean_string_object lp_mathlib_Mathlib_Notation_unexpandExistsUnique___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "ident"};
static const lean_object* lp_mathlib_Mathlib_Notation_unexpandExistsUnique___closed__8 = (const lean_object*)&lp_mathlib_Mathlib_Notation_unexpandExistsUnique___closed__8_value;
static const lean_ctor_object lp_mathlib_Mathlib_Notation_unexpandExistsUnique___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Notation_unexpandExistsUnique___closed__8_value),LEAN_SCALAR_PTR_LITERAL(52, 159, 208, 51, 14, 60, 6, 71)}};
static const lean_object* lp_mathlib_Mathlib_Notation_unexpandExistsUnique___closed__9 = (const lean_object*)&lp_mathlib_Mathlib_Notation_unexpandExistsUnique___closed__9_value;
static const lean_string_object lp_mathlib_Mathlib_Notation_unexpandExistsUnique___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 15, .m_capacity = 15, .m_length = 14, .m_data = "typeAscription"};
static const lean_object* lp_mathlib_Mathlib_Notation_unexpandExistsUnique___closed__10 = (const lean_object*)&lp_mathlib_Mathlib_Notation_unexpandExistsUnique___closed__10_value;
static const lean_ctor_object lp_mathlib_Mathlib_Notation_unexpandExistsUnique___closed__11_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Notation_isExplicitBinderSingular___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib_Mathlib_Notation_unexpandExistsUnique___closed__11_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Notation_unexpandExistsUnique___closed__11_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Notation_unexpandExistsUnique___closed__0_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib_Mathlib_Notation_unexpandExistsUnique___closed__11_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Notation_unexpandExistsUnique___closed__11_value_aux_1),((lean_object*)&lp_mathlib_Mathlib_Notation_unexpandExistsUnique___closed__1_value),LEAN_SCALAR_PTR_LITERAL(75, 170, 162, 138, 136, 204, 251, 229)}};
static const lean_ctor_object lp_mathlib_Mathlib_Notation_unexpandExistsUnique___closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Notation_unexpandExistsUnique___closed__11_value_aux_2),((lean_object*)&lp_mathlib_Mathlib_Notation_unexpandExistsUnique___closed__10_value),LEAN_SCALAR_PTR_LITERAL(247, 209, 88, 141, 5, 195, 49, 74)}};
static const lean_object* lp_mathlib_Mathlib_Notation_unexpandExistsUnique___closed__11 = (const lean_object*)&lp_mathlib_Mathlib_Notation_unexpandExistsUnique___closed__11_value;
static const lean_string_object lp_mathlib_Mathlib_Notation_unexpandExistsUnique___closed__12_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 15, .m_capacity = 15, .m_length = 14, .m_data = "hygienicLParen"};
static const lean_object* lp_mathlib_Mathlib_Notation_unexpandExistsUnique___closed__12 = (const lean_object*)&lp_mathlib_Mathlib_Notation_unexpandExistsUnique___closed__12_value;
static const lean_ctor_object lp_mathlib_Mathlib_Notation_unexpandExistsUnique___closed__13_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Notation_isExplicitBinderSingular___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib_Mathlib_Notation_unexpandExistsUnique___closed__13_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Notation_unexpandExistsUnique___closed__13_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Notation_unexpandExistsUnique___closed__0_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib_Mathlib_Notation_unexpandExistsUnique___closed__13_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Notation_unexpandExistsUnique___closed__13_value_aux_1),((lean_object*)&lp_mathlib_Mathlib_Notation_unexpandExistsUnique___closed__1_value),LEAN_SCALAR_PTR_LITERAL(75, 170, 162, 138, 136, 204, 251, 229)}};
static const lean_ctor_object lp_mathlib_Mathlib_Notation_unexpandExistsUnique___closed__13_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Notation_unexpandExistsUnique___closed__13_value_aux_2),((lean_object*)&lp_mathlib_Mathlib_Notation_unexpandExistsUnique___closed__12_value),LEAN_SCALAR_PTR_LITERAL(41, 104, 206, 51, 21, 254, 100, 101)}};
static const lean_object* lp_mathlib_Mathlib_Notation_unexpandExistsUnique___closed__13 = (const lean_object*)&lp_mathlib_Mathlib_Notation_unexpandExistsUnique___closed__13_value;
static const lean_string_object lp_mathlib_Mathlib_Notation_unexpandExistsUnique___closed__14_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 12, .m_capacity = 12, .m_length = 11, .m_data = "hygieneInfo"};
static const lean_object* lp_mathlib_Mathlib_Notation_unexpandExistsUnique___closed__14 = (const lean_object*)&lp_mathlib_Mathlib_Notation_unexpandExistsUnique___closed__14_value;
static const lean_ctor_object lp_mathlib_Mathlib_Notation_unexpandExistsUnique___closed__15_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Notation_unexpandExistsUnique___closed__14_value),LEAN_SCALAR_PTR_LITERAL(27, 64, 36, 144, 170, 151, 255, 136)}};
static const lean_object* lp_mathlib_Mathlib_Notation_unexpandExistsUnique___closed__15 = (const lean_object*)&lp_mathlib_Mathlib_Notation_unexpandExistsUnique___closed__15_value;
static const lean_string_object lp_mathlib_Mathlib_Notation_unexpandExistsUnique___closed__16_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "null"};
static const lean_object* lp_mathlib_Mathlib_Notation_unexpandExistsUnique___closed__16 = (const lean_object*)&lp_mathlib_Mathlib_Notation_unexpandExistsUnique___closed__16_value;
static const lean_ctor_object lp_mathlib_Mathlib_Notation_unexpandExistsUnique___closed__17_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Notation_unexpandExistsUnique___closed__16_value),LEAN_SCALAR_PTR_LITERAL(24, 58, 49, 223, 146, 207, 197, 136)}};
static const lean_object* lp_mathlib_Mathlib_Notation_unexpandExistsUnique___closed__17 = (const lean_object*)&lp_mathlib_Mathlib_Notation_unexpandExistsUnique___closed__17_value;
static const lean_string_object lp_mathlib_Mathlib_Notation_unexpandExistsUnique___closed__18_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = ":"};
static const lean_object* lp_mathlib_Mathlib_Notation_unexpandExistsUnique___closed__18 = (const lean_object*)&lp_mathlib_Mathlib_Notation_unexpandExistsUnique___closed__18_value;
static const lean_string_object lp_mathlib_Mathlib_Notation_unexpandExistsUnique___closed__19_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = ","};
static const lean_object* lp_mathlib_Mathlib_Notation_unexpandExistsUnique___closed__19 = (const lean_object*)&lp_mathlib_Mathlib_Notation_unexpandExistsUnique___closed__19_value;
static lean_once_cell_t lp_mathlib_Mathlib_Notation_unexpandExistsUnique___closed__20_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Notation_unexpandExistsUnique___closed__20;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Notation_unexpandExistsUnique(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Notation_unexpandExistsUnique___boxed(lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Mathlib_Notation_term_u2203_x21_____x2c___00__closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 13, .m_capacity = 13, .m_length = 10, .m_data = "term∃!__,_"};
static const lean_object* lp_mathlib_Mathlib_Notation_term_u2203_x21_____x2c___00__closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Notation_term_u2203_x21_____x2c___00__closed__0_value;
static const lean_ctor_object lp_mathlib_Mathlib_Notation_term_u2203_x21_____x2c___00__closed__1_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Notation_term_u2203_x21___x2c___00__closed__0_value),LEAN_SCALAR_PTR_LITERAL(118, 213, 161, 2, 73, 184, 31, 228)}};
static const lean_ctor_object lp_mathlib_Mathlib_Notation_term_u2203_x21_____x2c___00__closed__1_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Notation_term_u2203_x21_____x2c___00__closed__1_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Notation_term_u2203_x21___x2c___00__closed__1_value),LEAN_SCALAR_PTR_LITERAL(6, 103, 244, 232, 62, 27, 250, 92)}};
static const lean_ctor_object lp_mathlib_Mathlib_Notation_term_u2203_x21_____x2c___00__closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Notation_term_u2203_x21_____x2c___00__closed__1_value_aux_1),((lean_object*)&lp_mathlib_Mathlib_Notation_term_u2203_x21_____x2c___00__closed__0_value),LEAN_SCALAR_PTR_LITERAL(251, 61, 76, 94, 150, 214, 153, 124)}};
static const lean_object* lp_mathlib_Mathlib_Notation_term_u2203_x21_____x2c___00__closed__1 = (const lean_object*)&lp_mathlib_Mathlib_Notation_term_u2203_x21_____x2c___00__closed__1_value;
static const lean_string_object lp_mathlib_Mathlib_Notation_term_u2203_x21_____x2c___00__closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 3, .m_data = "∃! "};
static const lean_object* lp_mathlib_Mathlib_Notation_term_u2203_x21_____x2c___00__closed__2 = (const lean_object*)&lp_mathlib_Mathlib_Notation_term_u2203_x21_____x2c___00__closed__2_value;
static const lean_ctor_object lp_mathlib_Mathlib_Notation_term_u2203_x21_____x2c___00__closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 5}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Notation_term_u2203_x21_____x2c___00__closed__2_value)}};
static const lean_object* lp_mathlib_Mathlib_Notation_term_u2203_x21_____x2c___00__closed__3 = (const lean_object*)&lp_mathlib_Mathlib_Notation_term_u2203_x21_____x2c___00__closed__3_value;
static lean_once_cell_t lp_mathlib_Mathlib_Notation_term_u2203_x21_____x2c___00__closed__4_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Notation_term_u2203_x21_____x2c___00__closed__4;
static const lean_string_object lp_mathlib_Mathlib_Notation_term_u2203_x21_____x2c___00__closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 11, .m_capacity = 11, .m_length = 10, .m_data = "binderPred"};
static const lean_object* lp_mathlib_Mathlib_Notation_term_u2203_x21_____x2c___00__closed__5 = (const lean_object*)&lp_mathlib_Mathlib_Notation_term_u2203_x21_____x2c___00__closed__5_value;
static const lean_ctor_object lp_mathlib_Mathlib_Notation_term_u2203_x21_____x2c___00__closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Notation_term_u2203_x21_____x2c___00__closed__5_value),LEAN_SCALAR_PTR_LITERAL(218, 134, 142, 164, 134, 201, 62, 191)}};
static const lean_object* lp_mathlib_Mathlib_Notation_term_u2203_x21_____x2c___00__closed__6 = (const lean_object*)&lp_mathlib_Mathlib_Notation_term_u2203_x21_____x2c___00__closed__6_value;
static const lean_ctor_object lp_mathlib_Mathlib_Notation_term_u2203_x21_____x2c___00__closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 7}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Notation_term_u2203_x21_____x2c___00__closed__6_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_Mathlib_Notation_term_u2203_x21_____x2c___00__closed__7 = (const lean_object*)&lp_mathlib_Mathlib_Notation_term_u2203_x21_____x2c___00__closed__7_value;
static lean_once_cell_t lp_mathlib_Mathlib_Notation_term_u2203_x21_____x2c___00__closed__8_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Notation_term_u2203_x21_____x2c___00__closed__8;
static lean_once_cell_t lp_mathlib_Mathlib_Notation_term_u2203_x21_____x2c___00__closed__9_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Notation_term_u2203_x21_____x2c___00__closed__9;
static lean_once_cell_t lp_mathlib_Mathlib_Notation_term_u2203_x21_____x2c___00__closed__10_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Notation_term_u2203_x21_____x2c___00__closed__10;
static lean_once_cell_t lp_mathlib_Mathlib_Notation_term_u2203_x21_____x2c___00__closed__11_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Notation_term_u2203_x21_____x2c___00__closed__11;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Notation_term_u2203_x21_____x2c__;
static const lean_string_object lp_mathlib_Mathlib_Notation___aux__Mathlib__Basic__ExistsUnique______macroRules__Mathlib__Notation__term_u2203_x21_____x2c____1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "hole"};
static const lean_object* lp_mathlib_Mathlib_Notation___aux__Mathlib__Basic__ExistsUnique______macroRules__Mathlib__Notation__term_u2203_x21_____x2c____1___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Notation___aux__Mathlib__Basic__ExistsUnique______macroRules__Mathlib__Notation__term_u2203_x21_____x2c____1___closed__0_value;
static const lean_ctor_object lp_mathlib_Mathlib_Notation___aux__Mathlib__Basic__ExistsUnique______macroRules__Mathlib__Notation__term_u2203_x21_____x2c____1___closed__1_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Notation_isExplicitBinderSingular___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib_Mathlib_Notation___aux__Mathlib__Basic__ExistsUnique______macroRules__Mathlib__Notation__term_u2203_x21_____x2c____1___closed__1_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Notation___aux__Mathlib__Basic__ExistsUnique______macroRules__Mathlib__Notation__term_u2203_x21_____x2c____1___closed__1_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Notation_unexpandExistsUnique___closed__0_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib_Mathlib_Notation___aux__Mathlib__Basic__ExistsUnique______macroRules__Mathlib__Notation__term_u2203_x21_____x2c____1___closed__1_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Notation___aux__Mathlib__Basic__ExistsUnique______macroRules__Mathlib__Notation__term_u2203_x21_____x2c____1___closed__1_value_aux_1),((lean_object*)&lp_mathlib_Mathlib_Notation_unexpandExistsUnique___closed__1_value),LEAN_SCALAR_PTR_LITERAL(75, 170, 162, 138, 136, 204, 251, 229)}};
static const lean_ctor_object lp_mathlib_Mathlib_Notation___aux__Mathlib__Basic__ExistsUnique______macroRules__Mathlib__Notation__term_u2203_x21_____x2c____1___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Notation___aux__Mathlib__Basic__ExistsUnique______macroRules__Mathlib__Notation__term_u2203_x21_____x2c____1___closed__1_value_aux_2),((lean_object*)&lp_mathlib_Mathlib_Notation___aux__Mathlib__Basic__ExistsUnique______macroRules__Mathlib__Notation__term_u2203_x21_____x2c____1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(135, 134, 219, 115, 97, 130, 74, 55)}};
static const lean_object* lp_mathlib_Mathlib_Notation___aux__Mathlib__Basic__ExistsUnique______macroRules__Mathlib__Notation__term_u2203_x21_____x2c____1___closed__1 = (const lean_object*)&lp_mathlib_Mathlib_Notation___aux__Mathlib__Basic__ExistsUnique______macroRules__Mathlib__Notation__term_u2203_x21_____x2c____1___closed__1_value;
static const lean_string_object lp_mathlib_Mathlib_Notation___aux__Mathlib__Basic__ExistsUnique______macroRules__Mathlib__Notation__term_u2203_x21_____x2c____1___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = "x"};
static const lean_object* lp_mathlib_Mathlib_Notation___aux__Mathlib__Basic__ExistsUnique______macroRules__Mathlib__Notation__term_u2203_x21_____x2c____1___closed__2 = (const lean_object*)&lp_mathlib_Mathlib_Notation___aux__Mathlib__Basic__ExistsUnique______macroRules__Mathlib__Notation__term_u2203_x21_____x2c____1___closed__2_value;
static lean_once_cell_t lp_mathlib_Mathlib_Notation___aux__Mathlib__Basic__ExistsUnique______macroRules__Mathlib__Notation__term_u2203_x21_____x2c____1___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Notation___aux__Mathlib__Basic__ExistsUnique______macroRules__Mathlib__Notation__term_u2203_x21_____x2c____1___closed__3;
static const lean_ctor_object lp_mathlib_Mathlib_Notation___aux__Mathlib__Basic__ExistsUnique______macroRules__Mathlib__Notation__term_u2203_x21_____x2c____1___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Notation___aux__Mathlib__Basic__ExistsUnique______macroRules__Mathlib__Notation__term_u2203_x21_____x2c____1___closed__2_value),LEAN_SCALAR_PTR_LITERAL(243, 101, 181, 186, 114, 114, 131, 189)}};
static const lean_object* lp_mathlib_Mathlib_Notation___aux__Mathlib__Basic__ExistsUnique______macroRules__Mathlib__Notation__term_u2203_x21_____x2c____1___closed__4 = (const lean_object*)&lp_mathlib_Mathlib_Notation___aux__Mathlib__Basic__ExistsUnique______macroRules__Mathlib__Notation__term_u2203_x21_____x2c____1___closed__4_value;
static const lean_string_object lp_mathlib_Mathlib_Notation___aux__Mathlib__Basic__ExistsUnique______macroRules__Mathlib__Notation__term_u2203_x21_____x2c____1___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 7, .m_data = "term_∧_"};
static const lean_object* lp_mathlib_Mathlib_Notation___aux__Mathlib__Basic__ExistsUnique______macroRules__Mathlib__Notation__term_u2203_x21_____x2c____1___closed__5 = (const lean_object*)&lp_mathlib_Mathlib_Notation___aux__Mathlib__Basic__ExistsUnique______macroRules__Mathlib__Notation__term_u2203_x21_____x2c____1___closed__5_value;
static const lean_ctor_object lp_mathlib_Mathlib_Notation___aux__Mathlib__Basic__ExistsUnique______macroRules__Mathlib__Notation__term_u2203_x21_____x2c____1___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Notation___aux__Mathlib__Basic__ExistsUnique______macroRules__Mathlib__Notation__term_u2203_x21_____x2c____1___closed__5_value),LEAN_SCALAR_PTR_LITERAL(213, 224, 85, 99, 168, 124, 84, 223)}};
static const lean_object* lp_mathlib_Mathlib_Notation___aux__Mathlib__Basic__ExistsUnique______macroRules__Mathlib__Notation__term_u2203_x21_____x2c____1___closed__6 = (const lean_object*)&lp_mathlib_Mathlib_Notation___aux__Mathlib__Basic__ExistsUnique______macroRules__Mathlib__Notation__term_u2203_x21_____x2c____1___closed__6_value;
static const lean_string_object lp_mathlib_Mathlib_Notation___aux__Mathlib__Basic__ExistsUnique______macroRules__Mathlib__Notation__term_u2203_x21_____x2c____1___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 29, .m_capacity = 29, .m_length = 28, .m_data = "termSatisfies_binder_pred%__"};
static const lean_object* lp_mathlib_Mathlib_Notation___aux__Mathlib__Basic__ExistsUnique______macroRules__Mathlib__Notation__term_u2203_x21_____x2c____1___closed__7 = (const lean_object*)&lp_mathlib_Mathlib_Notation___aux__Mathlib__Basic__ExistsUnique______macroRules__Mathlib__Notation__term_u2203_x21_____x2c____1___closed__7_value;
static const lean_ctor_object lp_mathlib_Mathlib_Notation___aux__Mathlib__Basic__ExistsUnique______macroRules__Mathlib__Notation__term_u2203_x21_____x2c____1___closed__8_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Notation_isExplicitBinderSingular___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib_Mathlib_Notation___aux__Mathlib__Basic__ExistsUnique______macroRules__Mathlib__Notation__term_u2203_x21_____x2c____1___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Notation___aux__Mathlib__Basic__ExistsUnique______macroRules__Mathlib__Notation__term_u2203_x21_____x2c____1___closed__8_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Notation___aux__Mathlib__Basic__ExistsUnique______macroRules__Mathlib__Notation__term_u2203_x21_____x2c____1___closed__7_value),LEAN_SCALAR_PTR_LITERAL(35, 32, 166, 185, 227, 132, 228, 81)}};
static const lean_object* lp_mathlib_Mathlib_Notation___aux__Mathlib__Basic__ExistsUnique______macroRules__Mathlib__Notation__term_u2203_x21_____x2c____1___closed__8 = (const lean_object*)&lp_mathlib_Mathlib_Notation___aux__Mathlib__Basic__ExistsUnique______macroRules__Mathlib__Notation__term_u2203_x21_____x2c____1___closed__8_value;
static const lean_string_object lp_mathlib_Mathlib_Notation___aux__Mathlib__Basic__ExistsUnique______macroRules__Mathlib__Notation__term_u2203_x21_____x2c____1___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 23, .m_capacity = 23, .m_length = 22, .m_data = "satisfies_binder_pred%"};
static const lean_object* lp_mathlib_Mathlib_Notation___aux__Mathlib__Basic__ExistsUnique______macroRules__Mathlib__Notation__term_u2203_x21_____x2c____1___closed__9 = (const lean_object*)&lp_mathlib_Mathlib_Notation___aux__Mathlib__Basic__ExistsUnique______macroRules__Mathlib__Notation__term_u2203_x21_____x2c____1___closed__9_value;
static const lean_string_object lp_mathlib_Mathlib_Notation___aux__Mathlib__Basic__ExistsUnique______macroRules__Mathlib__Notation__term_u2203_x21_____x2c____1___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 1, .m_data = "∧"};
static const lean_object* lp_mathlib_Mathlib_Notation___aux__Mathlib__Basic__ExistsUnique______macroRules__Mathlib__Notation__term_u2203_x21_____x2c____1___closed__10 = (const lean_object*)&lp_mathlib_Mathlib_Notation___aux__Mathlib__Basic__ExistsUnique______macroRules__Mathlib__Notation__term_u2203_x21_____x2c____1___closed__10_value;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Notation___aux__Mathlib__Basic__ExistsUnique______macroRules__Mathlib__Notation__term_u2203_x21_____x2c____1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Notation___aux__Mathlib__Basic__ExistsUnique______macroRules__Mathlib__Notation__term_u2203_x21_____x2c____1___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_List_decidableBExistsUnique___redArg___lam__0(lean_object*, uint8_t, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_decidableBExistsUnique___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_List_decidableBExistsUnique___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_decidableBExistsUnique___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_List_decidableBExistsUnique(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_decidableBExistsUnique___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Mathlib_Notation_isExplicitBinderSingular(lean_object* v_xs_18_){
_start:
{
lean_object* v___x_19_; uint8_t v___x_20_; 
v___x_19_ = ((lean_object*)(lp_mathlib_Mathlib_Notation_isExplicitBinderSingular___closed__2));
lean_inc(v_xs_18_);
v___x_20_ = l_Lean_Syntax_isOfKind(v_xs_18_, v___x_19_);
if (v___x_20_ == 0)
{
lean_dec(v_xs_18_);
return v___x_20_;
}
else
{
lean_object* v___x_21_; lean_object* v___x_22_; lean_object* v___x_23_; uint8_t v___x_24_; 
v___x_21_ = lean_unsigned_to_nat(0u);
v___x_22_ = l_Lean_Syntax_getArg(v_xs_18_, v___x_21_);
lean_dec(v_xs_18_);
v___x_23_ = ((lean_object*)(lp_mathlib_Mathlib_Notation_isExplicitBinderSingular___closed__4));
lean_inc(v___x_22_);
v___x_24_ = l_Lean_Syntax_isOfKind(v___x_22_, v___x_23_);
if (v___x_24_ == 0)
{
lean_object* v___x_25_; uint8_t v___x_26_; 
v___x_25_ = lean_unsigned_to_nat(1u);
lean_inc(v___x_22_);
v___x_26_ = l_Lean_Syntax_matchesNull(v___x_22_, v___x_25_);
if (v___x_26_ == 0)
{
lean_dec(v___x_22_);
return v___x_26_;
}
else
{
lean_object* v___x_27_; lean_object* v___x_28_; uint8_t v___x_29_; 
v___x_27_ = l_Lean_Syntax_getArg(v___x_22_, v___x_21_);
lean_dec(v___x_22_);
v___x_28_ = ((lean_object*)(lp_mathlib_Mathlib_Notation_isExplicitBinderSingular___closed__6));
lean_inc(v___x_27_);
v___x_29_ = l_Lean_Syntax_isOfKind(v___x_27_, v___x_28_);
if (v___x_29_ == 0)
{
lean_dec(v___x_27_);
return v___x_29_;
}
else
{
lean_object* v___x_30_; uint8_t v___x_31_; 
v___x_30_ = l_Lean_Syntax_getArg(v___x_27_, v___x_25_);
lean_dec(v___x_27_);
lean_inc(v___x_30_);
v___x_31_ = l_Lean_Syntax_matchesNull(v___x_30_, v___x_25_);
if (v___x_31_ == 0)
{
lean_dec(v___x_30_);
return v___x_31_;
}
else
{
lean_object* v___x_32_; lean_object* v___x_33_; uint8_t v___x_34_; 
v___x_32_ = l_Lean_Syntax_getArg(v___x_30_, v___x_21_);
lean_dec(v___x_30_);
v___x_33_ = ((lean_object*)(lp_mathlib_Mathlib_Notation_isExplicitBinderSingular___closed__8));
v___x_34_ = l_Lean_Syntax_isOfKind(v___x_32_, v___x_33_);
if (v___x_34_ == 0)
{
return v___x_34_;
}
else
{
return v___x_20_;
}
}
}
}
}
else
{
lean_object* v___x_35_; lean_object* v___x_36_; uint8_t v___x_37_; 
v___x_35_ = l_Lean_Syntax_getArg(v___x_22_, v___x_21_);
v___x_36_ = lean_unsigned_to_nat(1u);
lean_inc(v___x_35_);
v___x_37_ = l_Lean_Syntax_matchesNull(v___x_35_, v___x_36_);
if (v___x_37_ == 0)
{
lean_dec(v___x_35_);
lean_dec(v___x_22_);
return v___x_37_;
}
else
{
lean_object* v___x_38_; lean_object* v___x_39_; uint8_t v___x_40_; 
v___x_38_ = l_Lean_Syntax_getArg(v___x_35_, v___x_21_);
lean_dec(v___x_35_);
v___x_39_ = ((lean_object*)(lp_mathlib_Mathlib_Notation_isExplicitBinderSingular___closed__8));
v___x_40_ = l_Lean_Syntax_isOfKind(v___x_38_, v___x_39_);
if (v___x_40_ == 0)
{
lean_dec(v___x_22_);
return v___x_40_;
}
else
{
lean_object* v___x_41_; uint8_t v___x_42_; 
v___x_41_ = l_Lean_Syntax_getArg(v___x_22_, v___x_36_);
lean_dec(v___x_22_);
v___x_42_ = l_Lean_Syntax_isNone(v___x_41_);
if (v___x_42_ == 0)
{
lean_object* v___x_43_; uint8_t v___x_44_; 
v___x_43_ = lean_unsigned_to_nat(2u);
v___x_44_ = l_Lean_Syntax_matchesNull(v___x_41_, v___x_43_);
if (v___x_44_ == 0)
{
return v___x_44_;
}
else
{
return v___x_20_;
}
}
else
{
lean_dec(v___x_41_);
return v___x_20_;
}
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Notation_isExplicitBinderSingular___boxed(lean_object* v_xs_45_){
_start:
{
uint8_t v_res_46_; lean_object* v_r_47_; 
v_res_46_ = lp_mathlib_Mathlib_Notation_isExplicitBinderSingular(v_xs_45_);
v_r_47_ = lean_box(v_res_46_);
return v_r_47_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Notation_term_u2203_x21___x2c___00__closed__8(void){
_start:
{
lean_object* v___x_61_; lean_object* v___x_62_; lean_object* v___x_63_; lean_object* v___x_64_; 
v___x_61_ = l_Lean_explicitBinders;
v___x_62_ = ((lean_object*)(lp_mathlib_Mathlib_Notation_term_u2203_x21___x2c___00__closed__7));
v___x_63_ = ((lean_object*)(lp_mathlib_Mathlib_Notation_term_u2203_x21___x2c___00__closed__5));
v___x_64_ = lean_alloc_ctor(2, 3, 0);
lean_ctor_set(v___x_64_, 0, v___x_63_);
lean_ctor_set(v___x_64_, 1, v___x_62_);
lean_ctor_set(v___x_64_, 2, v___x_61_);
return v___x_64_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Notation_term_u2203_x21___x2c___00__closed__11(void){
_start:
{
lean_object* v___x_68_; lean_object* v___x_69_; lean_object* v___x_70_; lean_object* v___x_71_; 
v___x_68_ = ((lean_object*)(lp_mathlib_Mathlib_Notation_term_u2203_x21___x2c___00__closed__10));
v___x_69_ = lean_obj_once(&lp_mathlib_Mathlib_Notation_term_u2203_x21___x2c___00__closed__8, &lp_mathlib_Mathlib_Notation_term_u2203_x21___x2c___00__closed__8_once, _init_lp_mathlib_Mathlib_Notation_term_u2203_x21___x2c___00__closed__8);
v___x_70_ = ((lean_object*)(lp_mathlib_Mathlib_Notation_term_u2203_x21___x2c___00__closed__5));
v___x_71_ = lean_alloc_ctor(2, 3, 0);
lean_ctor_set(v___x_71_, 0, v___x_70_);
lean_ctor_set(v___x_71_, 1, v___x_69_);
lean_ctor_set(v___x_71_, 2, v___x_68_);
return v___x_71_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Notation_term_u2203_x21___x2c___00__closed__15(void){
_start:
{
lean_object* v___x_78_; lean_object* v___x_79_; lean_object* v___x_80_; lean_object* v___x_81_; 
v___x_78_ = ((lean_object*)(lp_mathlib_Mathlib_Notation_term_u2203_x21___x2c___00__closed__14));
v___x_79_ = lean_obj_once(&lp_mathlib_Mathlib_Notation_term_u2203_x21___x2c___00__closed__11, &lp_mathlib_Mathlib_Notation_term_u2203_x21___x2c___00__closed__11_once, _init_lp_mathlib_Mathlib_Notation_term_u2203_x21___x2c___00__closed__11);
v___x_80_ = ((lean_object*)(lp_mathlib_Mathlib_Notation_term_u2203_x21___x2c___00__closed__5));
v___x_81_ = lean_alloc_ctor(2, 3, 0);
lean_ctor_set(v___x_81_, 0, v___x_80_);
lean_ctor_set(v___x_81_, 1, v___x_79_);
lean_ctor_set(v___x_81_, 2, v___x_78_);
return v___x_81_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Notation_term_u2203_x21___x2c___00__closed__16(void){
_start:
{
lean_object* v___x_82_; lean_object* v___x_83_; lean_object* v___x_84_; lean_object* v___x_85_; 
v___x_82_ = lean_obj_once(&lp_mathlib_Mathlib_Notation_term_u2203_x21___x2c___00__closed__15, &lp_mathlib_Mathlib_Notation_term_u2203_x21___x2c___00__closed__15_once, _init_lp_mathlib_Mathlib_Notation_term_u2203_x21___x2c___00__closed__15);
v___x_83_ = lean_unsigned_to_nat(1022u);
v___x_84_ = ((lean_object*)(lp_mathlib_Mathlib_Notation_term_u2203_x21___x2c___00__closed__3));
v___x_85_ = lean_alloc_ctor(3, 3, 0);
lean_ctor_set(v___x_85_, 0, v___x_84_);
lean_ctor_set(v___x_85_, 1, v___x_83_);
lean_ctor_set(v___x_85_, 2, v___x_82_);
return v___x_85_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Notation_term_u2203_x21___x2c__(void){
_start:
{
lean_object* v___x_86_; 
v___x_86_ = lean_obj_once(&lp_mathlib_Mathlib_Notation_term_u2203_x21___x2c___00__closed__16, &lp_mathlib_Mathlib_Notation_term_u2203_x21___x2c___00__closed__16_once, _init_lp_mathlib_Mathlib_Notation_term_u2203_x21___x2c___00__closed__16);
return v___x_86_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Notation___aux__Mathlib__Basic__ExistsUnique______macroRules__Mathlib__Notation__term_u2203_x21___x2c____1___lam__0(lean_object* v_xs_90_, lean_object* v___x_91_, lean_object* v_____r_92_, lean_object* v___y_93_, lean_object* v___y_94_){
_start:
{
lean_object* v___x_95_; lean_object* v___x_96_; 
v___x_95_ = ((lean_object*)(lp_mathlib_Mathlib_Notation___aux__Mathlib__Basic__ExistsUnique______macroRules__Mathlib__Notation__term_u2203_x21___x2c____1___lam__0___closed__1));
v___x_96_ = l_Lean_expandExplicitBinders(v___x_95_, v_xs_90_, v___x_91_, v___y_93_, v___y_94_);
if (lean_obj_tag(v___x_96_) == 0)
{
lean_object* v_a_97_; lean_object* v_a_98_; lean_object* v___x_100_; uint8_t v_isShared_101_; uint8_t v_isSharedCheck_105_; 
v_a_97_ = lean_ctor_get(v___x_96_, 0);
v_a_98_ = lean_ctor_get(v___x_96_, 1);
v_isSharedCheck_105_ = !lean_is_exclusive(v___x_96_);
if (v_isSharedCheck_105_ == 0)
{
v___x_100_ = v___x_96_;
v_isShared_101_ = v_isSharedCheck_105_;
goto v_resetjp_99_;
}
else
{
lean_inc(v_a_98_);
lean_inc(v_a_97_);
lean_dec(v___x_96_);
v___x_100_ = lean_box(0);
v_isShared_101_ = v_isSharedCheck_105_;
goto v_resetjp_99_;
}
v_resetjp_99_:
{
lean_object* v___x_103_; 
if (v_isShared_101_ == 0)
{
v___x_103_ = v___x_100_;
goto v_reusejp_102_;
}
else
{
lean_object* v_reuseFailAlloc_104_; 
v_reuseFailAlloc_104_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_104_, 0, v_a_97_);
lean_ctor_set(v_reuseFailAlloc_104_, 1, v_a_98_);
v___x_103_ = v_reuseFailAlloc_104_;
goto v_reusejp_102_;
}
v_reusejp_102_:
{
return v___x_103_;
}
}
}
else
{
lean_object* v_a_106_; lean_object* v_a_107_; lean_object* v___x_109_; uint8_t v_isShared_110_; uint8_t v_isSharedCheck_114_; 
v_a_106_ = lean_ctor_get(v___x_96_, 0);
v_a_107_ = lean_ctor_get(v___x_96_, 1);
v_isSharedCheck_114_ = !lean_is_exclusive(v___x_96_);
if (v_isSharedCheck_114_ == 0)
{
v___x_109_ = v___x_96_;
v_isShared_110_ = v_isSharedCheck_114_;
goto v_resetjp_108_;
}
else
{
lean_inc(v_a_107_);
lean_inc(v_a_106_);
lean_dec(v___x_96_);
v___x_109_ = lean_box(0);
v_isShared_110_ = v_isSharedCheck_114_;
goto v_resetjp_108_;
}
v_resetjp_108_:
{
lean_object* v___x_112_; 
if (v_isShared_110_ == 0)
{
v___x_112_ = v___x_109_;
goto v_reusejp_111_;
}
else
{
lean_object* v_reuseFailAlloc_113_; 
v_reuseFailAlloc_113_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_113_, 0, v_a_106_);
lean_ctor_set(v_reuseFailAlloc_113_, 1, v_a_107_);
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
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Notation___aux__Mathlib__Basic__ExistsUnique______macroRules__Mathlib__Notation__term_u2203_x21___x2c____1___lam__0___boxed(lean_object* v_xs_115_, lean_object* v___x_116_, lean_object* v_____r_117_, lean_object* v___y_118_, lean_object* v___y_119_){
_start:
{
lean_object* v_res_120_; 
v_res_120_ = lp_mathlib_Mathlib_Notation___aux__Mathlib__Basic__ExistsUnique______macroRules__Mathlib__Notation__term_u2203_x21___x2c____1___lam__0(v_xs_115_, v___x_116_, v_____r_117_, v___y_118_, v___y_119_);
lean_dec_ref(v___y_118_);
lean_dec(v_xs_115_);
return v_res_120_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Notation___aux__Mathlib__Basic__ExistsUnique______macroRules__Mathlib__Notation__term_u2203_x21___x2c____1(lean_object* v_x_122_, lean_object* v_a_123_, lean_object* v_a_124_){
_start:
{
lean_object* v___y_126_; lean_object* v___x_145_; uint8_t v___x_146_; 
v___x_145_ = ((lean_object*)(lp_mathlib_Mathlib_Notation_term_u2203_x21___x2c___00__closed__3));
lean_inc(v_x_122_);
v___x_146_ = l_Lean_Syntax_isOfKind(v_x_122_, v___x_145_);
if (v___x_146_ == 0)
{
lean_object* v___x_147_; lean_object* v___x_148_; 
lean_dec(v_x_122_);
v___x_147_ = lean_box(1);
v___x_148_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_148_, 0, v___x_147_);
lean_ctor_set(v___x_148_, 1, v_a_124_);
return v___x_148_;
}
else
{
lean_object* v___x_149_; lean_object* v_xs_150_; lean_object* v___x_151_; lean_object* v___x_152_; uint8_t v___x_156_; 
v___x_149_ = lean_unsigned_to_nat(1u);
v_xs_150_ = l_Lean_Syntax_getArg(v_x_122_, v___x_149_);
v___x_151_ = lean_unsigned_to_nat(3u);
v___x_152_ = l_Lean_Syntax_getArg(v_x_122_, v___x_151_);
lean_dec(v_x_122_);
lean_inc(v_xs_150_);
v___x_156_ = lp_mathlib_Mathlib_Notation_isExplicitBinderSingular(v_xs_150_);
if (v___x_156_ == 0)
{
if (v___x_146_ == 0)
{
goto v___jp_153_;
}
else
{
lean_object* v___x_157_; lean_object* v___x_158_; 
v___x_157_ = ((lean_object*)(lp_mathlib_Mathlib_Notation___aux__Mathlib__Basic__ExistsUnique______macroRules__Mathlib__Notation__term_u2203_x21___x2c____1___closed__0));
v___x_158_ = l_Lean_Macro_throwErrorAt___redArg(v_xs_150_, v___x_157_, v_a_123_, v_a_124_);
if (lean_obj_tag(v___x_158_) == 0)
{
lean_object* v_a_159_; lean_object* v_a_160_; lean_object* v___x_161_; 
v_a_159_ = lean_ctor_get(v___x_158_, 0);
lean_inc(v_a_159_);
v_a_160_ = lean_ctor_get(v___x_158_, 1);
lean_inc(v_a_160_);
lean_dec_ref_known(v___x_158_, 2);
v___x_161_ = lp_mathlib_Mathlib_Notation___aux__Mathlib__Basic__ExistsUnique______macroRules__Mathlib__Notation__term_u2203_x21___x2c____1___lam__0(v_xs_150_, v___x_152_, v_a_159_, v_a_123_, v_a_160_);
lean_dec(v_xs_150_);
v___y_126_ = v___x_161_;
goto v___jp_125_;
}
else
{
lean_object* v_a_162_; lean_object* v_a_163_; lean_object* v___x_165_; uint8_t v_isShared_166_; uint8_t v_isSharedCheck_170_; 
lean_dec(v___x_152_);
lean_dec(v_xs_150_);
v_a_162_ = lean_ctor_get(v___x_158_, 0);
v_a_163_ = lean_ctor_get(v___x_158_, 1);
v_isSharedCheck_170_ = !lean_is_exclusive(v___x_158_);
if (v_isSharedCheck_170_ == 0)
{
v___x_165_ = v___x_158_;
v_isShared_166_ = v_isSharedCheck_170_;
goto v_resetjp_164_;
}
else
{
lean_inc(v_a_163_);
lean_inc(v_a_162_);
lean_dec(v___x_158_);
v___x_165_ = lean_box(0);
v_isShared_166_ = v_isSharedCheck_170_;
goto v_resetjp_164_;
}
v_resetjp_164_:
{
lean_object* v___x_168_; 
if (v_isShared_166_ == 0)
{
v___x_168_ = v___x_165_;
goto v_reusejp_167_;
}
else
{
lean_object* v_reuseFailAlloc_169_; 
v_reuseFailAlloc_169_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_169_, 0, v_a_162_);
lean_ctor_set(v_reuseFailAlloc_169_, 1, v_a_163_);
v___x_168_ = v_reuseFailAlloc_169_;
goto v_reusejp_167_;
}
v_reusejp_167_:
{
return v___x_168_;
}
}
}
}
}
else
{
goto v___jp_153_;
}
v___jp_153_:
{
lean_object* v___x_154_; lean_object* v___x_155_; 
v___x_154_ = lean_box(0);
v___x_155_ = lp_mathlib_Mathlib_Notation___aux__Mathlib__Basic__ExistsUnique______macroRules__Mathlib__Notation__term_u2203_x21___x2c____1___lam__0(v_xs_150_, v___x_152_, v___x_154_, v_a_123_, v_a_124_);
lean_dec(v_xs_150_);
v___y_126_ = v___x_155_;
goto v___jp_125_;
}
}
v___jp_125_:
{
if (lean_obj_tag(v___y_126_) == 0)
{
lean_object* v_a_127_; lean_object* v_a_128_; lean_object* v___x_130_; uint8_t v_isShared_131_; uint8_t v_isSharedCheck_135_; 
v_a_127_ = lean_ctor_get(v___y_126_, 0);
v_a_128_ = lean_ctor_get(v___y_126_, 1);
v_isSharedCheck_135_ = !lean_is_exclusive(v___y_126_);
if (v_isSharedCheck_135_ == 0)
{
v___x_130_ = v___y_126_;
v_isShared_131_ = v_isSharedCheck_135_;
goto v_resetjp_129_;
}
else
{
lean_inc(v_a_128_);
lean_inc(v_a_127_);
lean_dec(v___y_126_);
v___x_130_ = lean_box(0);
v_isShared_131_ = v_isSharedCheck_135_;
goto v_resetjp_129_;
}
v_resetjp_129_:
{
lean_object* v___x_133_; 
if (v_isShared_131_ == 0)
{
v___x_133_ = v___x_130_;
goto v_reusejp_132_;
}
else
{
lean_object* v_reuseFailAlloc_134_; 
v_reuseFailAlloc_134_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_134_, 0, v_a_127_);
lean_ctor_set(v_reuseFailAlloc_134_, 1, v_a_128_);
v___x_133_ = v_reuseFailAlloc_134_;
goto v_reusejp_132_;
}
v_reusejp_132_:
{
return v___x_133_;
}
}
}
else
{
lean_object* v_a_136_; lean_object* v_a_137_; lean_object* v___x_139_; uint8_t v_isShared_140_; uint8_t v_isSharedCheck_144_; 
v_a_136_ = lean_ctor_get(v___y_126_, 0);
v_a_137_ = lean_ctor_get(v___y_126_, 1);
v_isSharedCheck_144_ = !lean_is_exclusive(v___y_126_);
if (v_isSharedCheck_144_ == 0)
{
v___x_139_ = v___y_126_;
v_isShared_140_ = v_isSharedCheck_144_;
goto v_resetjp_138_;
}
else
{
lean_inc(v_a_137_);
lean_inc(v_a_136_);
lean_dec(v___y_126_);
v___x_139_ = lean_box(0);
v_isShared_140_ = v_isSharedCheck_144_;
goto v_resetjp_138_;
}
v_resetjp_138_:
{
lean_object* v___x_142_; 
if (v_isShared_140_ == 0)
{
v___x_142_ = v___x_139_;
goto v_reusejp_141_;
}
else
{
lean_object* v_reuseFailAlloc_143_; 
v_reuseFailAlloc_143_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_143_, 0, v_a_136_);
lean_ctor_set(v_reuseFailAlloc_143_, 1, v_a_137_);
v___x_142_ = v_reuseFailAlloc_143_;
goto v_reusejp_141_;
}
v_reusejp_141_:
{
return v___x_142_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Notation___aux__Mathlib__Basic__ExistsUnique______macroRules__Mathlib__Notation__term_u2203_x21___x2c____1___boxed(lean_object* v_x_171_, lean_object* v_a_172_, lean_object* v_a_173_){
_start:
{
lean_object* v_res_174_; 
v_res_174_ = lp_mathlib_Mathlib_Notation___aux__Mathlib__Basic__ExistsUnique______macroRules__Mathlib__Notation__term_u2203_x21___x2c____1(v_x_171_, v_a_172_, v_a_173_);
lean_dec_ref(v_a_172_);
return v_res_174_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Notation_unexpandExistsUnique___closed__20(void){
_start:
{
lean_object* v___x_218_; 
v___x_218_ = l_Array_mkArray0(lean_box(0));
return v___x_218_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Notation_unexpandExistsUnique(lean_object* v_x_219_, lean_object* v_a_220_, lean_object* v_a_221_){
_start:
{
lean_object* v___x_222_; uint8_t v___x_223_; 
v___x_222_ = ((lean_object*)(lp_mathlib_Mathlib_Notation_unexpandExistsUnique___closed__3));
lean_inc(v_x_219_);
v___x_223_ = l_Lean_Syntax_isOfKind(v_x_219_, v___x_222_);
if (v___x_223_ == 0)
{
lean_object* v___x_224_; lean_object* v___x_225_; 
lean_dec(v_x_219_);
v___x_224_ = lean_box(0);
v___x_225_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_225_, 0, v___x_224_);
lean_ctor_set(v___x_225_, 1, v_a_221_);
return v___x_225_;
}
else
{
lean_object* v___x_226_; lean_object* v___x_227_; uint8_t v___x_228_; 
v___x_226_ = lean_unsigned_to_nat(1u);
v___x_227_ = l_Lean_Syntax_getArg(v_x_219_, v___x_226_);
lean_dec(v_x_219_);
lean_inc(v___x_227_);
v___x_228_ = l_Lean_Syntax_matchesNull(v___x_227_, v___x_226_);
if (v___x_228_ == 0)
{
lean_object* v___x_229_; lean_object* v___x_230_; 
lean_dec(v___x_227_);
v___x_229_ = lean_box(0);
v___x_230_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_230_, 0, v___x_229_);
lean_ctor_set(v___x_230_, 1, v_a_221_);
return v___x_230_;
}
else
{
lean_object* v___x_231_; lean_object* v___x_232_; lean_object* v___x_233_; uint8_t v___x_234_; 
v___x_231_ = lean_unsigned_to_nat(0u);
v___x_232_ = l_Lean_Syntax_getArg(v___x_227_, v___x_231_);
lean_dec(v___x_227_);
v___x_233_ = ((lean_object*)(lp_mathlib_Mathlib_Notation_unexpandExistsUnique___closed__5));
lean_inc(v___x_232_);
v___x_234_ = l_Lean_Syntax_isOfKind(v___x_232_, v___x_233_);
if (v___x_234_ == 0)
{
lean_object* v___x_235_; lean_object* v___x_236_; 
lean_dec(v___x_232_);
v___x_235_ = lean_box(0);
v___x_236_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_236_, 0, v___x_235_);
lean_ctor_set(v___x_236_, 1, v_a_221_);
return v___x_236_;
}
else
{
lean_object* v___x_237_; lean_object* v___x_238_; uint8_t v___x_239_; 
v___x_237_ = l_Lean_Syntax_getArg(v___x_232_, v___x_226_);
lean_dec(v___x_232_);
v___x_238_ = ((lean_object*)(lp_mathlib_Mathlib_Notation_unexpandExistsUnique___closed__7));
lean_inc(v___x_237_);
v___x_239_ = l_Lean_Syntax_isOfKind(v___x_237_, v___x_238_);
if (v___x_239_ == 0)
{
lean_object* v___x_240_; lean_object* v___x_241_; 
lean_dec(v___x_237_);
v___x_240_ = lean_box(0);
v___x_241_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_241_, 0, v___x_240_);
lean_ctor_set(v___x_241_, 1, v_a_221_);
return v___x_241_;
}
else
{
lean_object* v___x_242_; uint8_t v___x_243_; 
v___x_242_ = l_Lean_Syntax_getArg(v___x_237_, v___x_231_);
lean_inc(v___x_242_);
v___x_243_ = l_Lean_Syntax_matchesNull(v___x_242_, v___x_226_);
if (v___x_243_ == 0)
{
lean_object* v___x_244_; lean_object* v___x_245_; 
lean_dec(v___x_242_);
lean_dec(v___x_237_);
v___x_244_ = lean_box(0);
v___x_245_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_245_, 0, v___x_244_);
lean_ctor_set(v___x_245_, 1, v_a_221_);
return v___x_245_;
}
else
{
lean_object* v___x_246_; lean_object* v___x_247_; uint8_t v___x_248_; 
v___x_246_ = l_Lean_Syntax_getArg(v___x_242_, v___x_231_);
lean_dec(v___x_242_);
v___x_247_ = ((lean_object*)(lp_mathlib_Mathlib_Notation_unexpandExistsUnique___closed__9));
lean_inc(v___x_246_);
v___x_248_ = l_Lean_Syntax_isOfKind(v___x_246_, v___x_247_);
if (v___x_248_ == 0)
{
lean_object* v___x_249_; uint8_t v___x_250_; 
v___x_249_ = ((lean_object*)(lp_mathlib_Mathlib_Notation_unexpandExistsUnique___closed__11));
lean_inc(v___x_246_);
v___x_250_ = l_Lean_Syntax_isOfKind(v___x_246_, v___x_249_);
if (v___x_250_ == 0)
{
lean_object* v___x_251_; lean_object* v___x_252_; 
lean_dec(v___x_246_);
lean_dec(v___x_237_);
v___x_251_ = lean_box(0);
v___x_252_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_252_, 0, v___x_251_);
lean_ctor_set(v___x_252_, 1, v_a_221_);
return v___x_252_;
}
else
{
lean_object* v___x_253_; lean_object* v___x_254_; uint8_t v___x_255_; 
v___x_253_ = l_Lean_Syntax_getArg(v___x_246_, v___x_231_);
v___x_254_ = ((lean_object*)(lp_mathlib_Mathlib_Notation_unexpandExistsUnique___closed__13));
lean_inc(v___x_253_);
v___x_255_ = l_Lean_Syntax_isOfKind(v___x_253_, v___x_254_);
if (v___x_255_ == 0)
{
lean_object* v___x_256_; lean_object* v___x_257_; 
lean_dec(v___x_253_);
lean_dec(v___x_246_);
lean_dec(v___x_237_);
v___x_256_ = lean_box(0);
v___x_257_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_257_, 0, v___x_256_);
lean_ctor_set(v___x_257_, 1, v_a_221_);
return v___x_257_;
}
else
{
lean_object* v___x_258_; lean_object* v___x_259_; uint8_t v___x_260_; 
v___x_258_ = l_Lean_Syntax_getArg(v___x_253_, v___x_226_);
lean_dec(v___x_253_);
v___x_259_ = ((lean_object*)(lp_mathlib_Mathlib_Notation_unexpandExistsUnique___closed__15));
lean_inc(v___x_258_);
v___x_260_ = l_Lean_Syntax_isOfKind(v___x_258_, v___x_259_);
if (v___x_260_ == 0)
{
lean_object* v___x_261_; lean_object* v___x_262_; 
lean_dec(v___x_258_);
lean_dec(v___x_246_);
lean_dec(v___x_237_);
v___x_261_ = lean_box(0);
v___x_262_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_262_, 0, v___x_261_);
lean_ctor_set(v___x_262_, 1, v_a_221_);
return v___x_262_;
}
else
{
lean_object* v___x_263_; lean_object* v___x_264_; uint8_t v___x_265_; 
v___x_263_ = l_Lean_Syntax_getArg(v___x_258_, v___x_231_);
lean_dec(v___x_258_);
v___x_264_ = lean_box(0);
v___x_265_ = l_Lean_Syntax_matchesIdent(v___x_263_, v___x_264_);
lean_dec(v___x_263_);
if (v___x_265_ == 0)
{
lean_object* v___x_266_; lean_object* v___x_267_; 
lean_dec(v___x_246_);
lean_dec(v___x_237_);
v___x_266_ = lean_box(0);
v___x_267_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_267_, 0, v___x_266_);
lean_ctor_set(v___x_267_, 1, v_a_221_);
return v___x_267_;
}
else
{
lean_object* v___x_268_; uint8_t v___x_269_; 
v___x_268_ = l_Lean_Syntax_getArg(v___x_246_, v___x_226_);
lean_inc(v___x_268_);
v___x_269_ = l_Lean_Syntax_isOfKind(v___x_268_, v___x_247_);
if (v___x_269_ == 0)
{
lean_object* v___x_270_; lean_object* v___x_271_; 
lean_dec(v___x_268_);
lean_dec(v___x_246_);
lean_dec(v___x_237_);
v___x_270_ = lean_box(0);
v___x_271_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_271_, 0, v___x_270_);
lean_ctor_set(v___x_271_, 1, v_a_221_);
return v___x_271_;
}
else
{
lean_object* v___x_272_; lean_object* v___x_273_; uint8_t v___x_274_; 
v___x_272_ = lean_unsigned_to_nat(3u);
v___x_273_ = l_Lean_Syntax_getArg(v___x_246_, v___x_272_);
lean_dec(v___x_246_);
lean_inc(v___x_273_);
v___x_274_ = l_Lean_Syntax_matchesNull(v___x_273_, v___x_226_);
if (v___x_274_ == 0)
{
lean_object* v___x_275_; lean_object* v___x_276_; 
lean_dec(v___x_273_);
lean_dec(v___x_268_);
lean_dec(v___x_237_);
v___x_275_ = lean_box(0);
v___x_276_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_276_, 0, v___x_275_);
lean_ctor_set(v___x_276_, 1, v_a_221_);
return v___x_276_;
}
else
{
lean_object* v___x_277_; uint8_t v___x_278_; 
v___x_277_ = l_Lean_Syntax_getArg(v___x_237_, v___x_226_);
v___x_278_ = l_Lean_Syntax_matchesNull(v___x_277_, v___x_231_);
if (v___x_278_ == 0)
{
lean_object* v___x_279_; lean_object* v___x_280_; 
lean_dec(v___x_273_);
lean_dec(v___x_268_);
lean_dec(v___x_237_);
v___x_279_ = lean_box(0);
v___x_280_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_280_, 0, v___x_279_);
lean_ctor_set(v___x_280_, 1, v_a_221_);
return v___x_280_;
}
else
{
lean_object* v___x_281_; lean_object* v___x_282_; lean_object* v___x_283_; lean_object* v___x_284_; lean_object* v___x_285_; lean_object* v___x_286_; lean_object* v___x_287_; lean_object* v___x_288_; lean_object* v___x_289_; lean_object* v___x_290_; lean_object* v___x_291_; lean_object* v___x_292_; lean_object* v___x_293_; lean_object* v___x_294_; lean_object* v___x_295_; lean_object* v___x_296_; lean_object* v___x_297_; lean_object* v___x_298_; lean_object* v___x_299_; lean_object* v___x_300_; lean_object* v___x_301_; 
v___x_281_ = l_Lean_Syntax_getArg(v___x_273_, v___x_231_);
lean_dec(v___x_273_);
v___x_282_ = l_Lean_Syntax_getArg(v___x_237_, v___x_272_);
lean_dec(v___x_237_);
v___x_283_ = l_Lean_SourceInfo_fromRef(v_a_220_, v___x_248_);
v___x_284_ = ((lean_object*)(lp_mathlib_Mathlib_Notation_term_u2203_x21___x2c___00__closed__3));
v___x_285_ = ((lean_object*)(lp_mathlib_Mathlib_Notation_term_u2203_x21___x2c___00__closed__6));
lean_inc_n(v___x_283_, 8);
v___x_286_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_286_, 0, v___x_283_);
lean_ctor_set(v___x_286_, 1, v___x_285_);
v___x_287_ = ((lean_object*)(lp_mathlib_Mathlib_Notation_isExplicitBinderSingular___closed__2));
v___x_288_ = ((lean_object*)(lp_mathlib_Mathlib_Notation_isExplicitBinderSingular___closed__4));
v___x_289_ = ((lean_object*)(lp_mathlib_Mathlib_Notation_unexpandExistsUnique___closed__17));
v___x_290_ = ((lean_object*)(lp_mathlib_Mathlib_Notation_isExplicitBinderSingular___closed__8));
v___x_291_ = l_Lean_Syntax_node1(v___x_283_, v___x_290_, v___x_268_);
v___x_292_ = l_Lean_Syntax_node1(v___x_283_, v___x_289_, v___x_291_);
v___x_293_ = ((lean_object*)(lp_mathlib_Mathlib_Notation_unexpandExistsUnique___closed__18));
v___x_294_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_294_, 0, v___x_283_);
lean_ctor_set(v___x_294_, 1, v___x_293_);
v___x_295_ = l_Lean_Syntax_node2(v___x_283_, v___x_289_, v___x_294_, v___x_281_);
v___x_296_ = l_Lean_Syntax_node2(v___x_283_, v___x_288_, v___x_292_, v___x_295_);
v___x_297_ = l_Lean_Syntax_node1(v___x_283_, v___x_287_, v___x_296_);
v___x_298_ = ((lean_object*)(lp_mathlib_Mathlib_Notation_unexpandExistsUnique___closed__19));
v___x_299_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_299_, 0, v___x_283_);
lean_ctor_set(v___x_299_, 1, v___x_298_);
v___x_300_ = l_Lean_Syntax_node4(v___x_283_, v___x_284_, v___x_286_, v___x_297_, v___x_299_, v___x_282_);
v___x_301_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_301_, 0, v___x_300_);
lean_ctor_set(v___x_301_, 1, v_a_221_);
return v___x_301_;
}
}
}
}
}
}
}
}
else
{
lean_object* v___x_302_; uint8_t v___x_303_; 
v___x_302_ = l_Lean_Syntax_getArg(v___x_237_, v___x_226_);
v___x_303_ = l_Lean_Syntax_matchesNull(v___x_302_, v___x_231_);
if (v___x_303_ == 0)
{
lean_object* v___x_304_; lean_object* v___x_305_; 
lean_dec(v___x_246_);
lean_dec(v___x_237_);
v___x_304_ = lean_box(0);
v___x_305_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_305_, 0, v___x_304_);
lean_ctor_set(v___x_305_, 1, v_a_221_);
return v___x_305_;
}
else
{
lean_object* v___x_306_; lean_object* v___x_307_; uint8_t v___x_308_; lean_object* v___x_309_; lean_object* v___x_310_; lean_object* v___x_311_; lean_object* v___x_312_; lean_object* v___x_313_; lean_object* v___x_314_; lean_object* v___x_315_; lean_object* v___x_316_; lean_object* v___x_317_; lean_object* v___x_318_; lean_object* v___x_319_; lean_object* v___x_320_; lean_object* v___x_321_; lean_object* v___x_322_; lean_object* v___x_323_; lean_object* v___x_324_; lean_object* v___x_325_; lean_object* v___x_326_; 
v___x_306_ = lean_unsigned_to_nat(3u);
v___x_307_ = l_Lean_Syntax_getArg(v___x_237_, v___x_306_);
lean_dec(v___x_237_);
v___x_308_ = 0;
v___x_309_ = l_Lean_SourceInfo_fromRef(v_a_220_, v___x_308_);
v___x_310_ = ((lean_object*)(lp_mathlib_Mathlib_Notation_term_u2203_x21___x2c___00__closed__3));
v___x_311_ = ((lean_object*)(lp_mathlib_Mathlib_Notation_term_u2203_x21___x2c___00__closed__6));
lean_inc_n(v___x_309_, 7);
v___x_312_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_312_, 0, v___x_309_);
lean_ctor_set(v___x_312_, 1, v___x_311_);
v___x_313_ = ((lean_object*)(lp_mathlib_Mathlib_Notation_isExplicitBinderSingular___closed__2));
v___x_314_ = ((lean_object*)(lp_mathlib_Mathlib_Notation_isExplicitBinderSingular___closed__4));
v___x_315_ = ((lean_object*)(lp_mathlib_Mathlib_Notation_unexpandExistsUnique___closed__17));
v___x_316_ = ((lean_object*)(lp_mathlib_Mathlib_Notation_isExplicitBinderSingular___closed__8));
v___x_317_ = l_Lean_Syntax_node1(v___x_309_, v___x_316_, v___x_246_);
v___x_318_ = l_Lean_Syntax_node1(v___x_309_, v___x_315_, v___x_317_);
v___x_319_ = lean_obj_once(&lp_mathlib_Mathlib_Notation_unexpandExistsUnique___closed__20, &lp_mathlib_Mathlib_Notation_unexpandExistsUnique___closed__20_once, _init_lp_mathlib_Mathlib_Notation_unexpandExistsUnique___closed__20);
v___x_320_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_320_, 0, v___x_309_);
lean_ctor_set(v___x_320_, 1, v___x_315_);
lean_ctor_set(v___x_320_, 2, v___x_319_);
v___x_321_ = l_Lean_Syntax_node2(v___x_309_, v___x_314_, v___x_318_, v___x_320_);
v___x_322_ = l_Lean_Syntax_node1(v___x_309_, v___x_313_, v___x_321_);
v___x_323_ = ((lean_object*)(lp_mathlib_Mathlib_Notation_unexpandExistsUnique___closed__19));
v___x_324_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_324_, 0, v___x_309_);
lean_ctor_set(v___x_324_, 1, v___x_323_);
v___x_325_ = l_Lean_Syntax_node4(v___x_309_, v___x_310_, v___x_312_, v___x_322_, v___x_324_, v___x_307_);
v___x_326_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_326_, 0, v___x_325_);
lean_ctor_set(v___x_326_, 1, v_a_221_);
return v___x_326_;
}
}
}
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Notation_unexpandExistsUnique___boxed(lean_object* v_x_327_, lean_object* v_a_328_, lean_object* v_a_329_){
_start:
{
lean_object* v_res_330_; 
v_res_330_ = lp_mathlib_Mathlib_Notation_unexpandExistsUnique(v_x_327_, v_a_328_, v_a_329_);
lean_dec(v_a_328_);
return v_res_330_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Notation_term_u2203_x21_____x2c___00__closed__4(void){
_start:
{
lean_object* v___x_339_; lean_object* v___x_340_; lean_object* v___x_341_; lean_object* v___x_342_; 
v___x_339_ = l_Lean_binderIdent;
v___x_340_ = ((lean_object*)(lp_mathlib_Mathlib_Notation_term_u2203_x21_____x2c___00__closed__3));
v___x_341_ = ((lean_object*)(lp_mathlib_Mathlib_Notation_term_u2203_x21___x2c___00__closed__5));
v___x_342_ = lean_alloc_ctor(2, 3, 0);
lean_ctor_set(v___x_342_, 0, v___x_341_);
lean_ctor_set(v___x_342_, 1, v___x_340_);
lean_ctor_set(v___x_342_, 2, v___x_339_);
return v___x_342_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Notation_term_u2203_x21_____x2c___00__closed__8(void){
_start:
{
lean_object* v___x_349_; lean_object* v___x_350_; lean_object* v___x_351_; lean_object* v___x_352_; 
v___x_349_ = ((lean_object*)(lp_mathlib_Mathlib_Notation_term_u2203_x21_____x2c___00__closed__7));
v___x_350_ = lean_obj_once(&lp_mathlib_Mathlib_Notation_term_u2203_x21_____x2c___00__closed__4, &lp_mathlib_Mathlib_Notation_term_u2203_x21_____x2c___00__closed__4_once, _init_lp_mathlib_Mathlib_Notation_term_u2203_x21_____x2c___00__closed__4);
v___x_351_ = ((lean_object*)(lp_mathlib_Mathlib_Notation_term_u2203_x21___x2c___00__closed__5));
v___x_352_ = lean_alloc_ctor(2, 3, 0);
lean_ctor_set(v___x_352_, 0, v___x_351_);
lean_ctor_set(v___x_352_, 1, v___x_350_);
lean_ctor_set(v___x_352_, 2, v___x_349_);
return v___x_352_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Notation_term_u2203_x21_____x2c___00__closed__9(void){
_start:
{
lean_object* v___x_353_; lean_object* v___x_354_; lean_object* v___x_355_; lean_object* v___x_356_; 
v___x_353_ = ((lean_object*)(lp_mathlib_Mathlib_Notation_term_u2203_x21___x2c___00__closed__10));
v___x_354_ = lean_obj_once(&lp_mathlib_Mathlib_Notation_term_u2203_x21_____x2c___00__closed__8, &lp_mathlib_Mathlib_Notation_term_u2203_x21_____x2c___00__closed__8_once, _init_lp_mathlib_Mathlib_Notation_term_u2203_x21_____x2c___00__closed__8);
v___x_355_ = ((lean_object*)(lp_mathlib_Mathlib_Notation_term_u2203_x21___x2c___00__closed__5));
v___x_356_ = lean_alloc_ctor(2, 3, 0);
lean_ctor_set(v___x_356_, 0, v___x_355_);
lean_ctor_set(v___x_356_, 1, v___x_354_);
lean_ctor_set(v___x_356_, 2, v___x_353_);
return v___x_356_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Notation_term_u2203_x21_____x2c___00__closed__10(void){
_start:
{
lean_object* v___x_357_; lean_object* v___x_358_; lean_object* v___x_359_; lean_object* v___x_360_; 
v___x_357_ = ((lean_object*)(lp_mathlib_Mathlib_Notation_term_u2203_x21___x2c___00__closed__14));
v___x_358_ = lean_obj_once(&lp_mathlib_Mathlib_Notation_term_u2203_x21_____x2c___00__closed__9, &lp_mathlib_Mathlib_Notation_term_u2203_x21_____x2c___00__closed__9_once, _init_lp_mathlib_Mathlib_Notation_term_u2203_x21_____x2c___00__closed__9);
v___x_359_ = ((lean_object*)(lp_mathlib_Mathlib_Notation_term_u2203_x21___x2c___00__closed__5));
v___x_360_ = lean_alloc_ctor(2, 3, 0);
lean_ctor_set(v___x_360_, 0, v___x_359_);
lean_ctor_set(v___x_360_, 1, v___x_358_);
lean_ctor_set(v___x_360_, 2, v___x_357_);
return v___x_360_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Notation_term_u2203_x21_____x2c___00__closed__11(void){
_start:
{
lean_object* v___x_361_; lean_object* v___x_362_; lean_object* v___x_363_; lean_object* v___x_364_; 
v___x_361_ = lean_obj_once(&lp_mathlib_Mathlib_Notation_term_u2203_x21_____x2c___00__closed__10, &lp_mathlib_Mathlib_Notation_term_u2203_x21_____x2c___00__closed__10_once, _init_lp_mathlib_Mathlib_Notation_term_u2203_x21_____x2c___00__closed__10);
v___x_362_ = lean_unsigned_to_nat(1022u);
v___x_363_ = ((lean_object*)(lp_mathlib_Mathlib_Notation_term_u2203_x21_____x2c___00__closed__1));
v___x_364_ = lean_alloc_ctor(3, 3, 0);
lean_ctor_set(v___x_364_, 0, v___x_363_);
lean_ctor_set(v___x_364_, 1, v___x_362_);
lean_ctor_set(v___x_364_, 2, v___x_361_);
return v___x_364_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Notation_term_u2203_x21_____x2c__(void){
_start:
{
lean_object* v___x_365_; 
v___x_365_ = lean_obj_once(&lp_mathlib_Mathlib_Notation_term_u2203_x21_____x2c___00__closed__11, &lp_mathlib_Mathlib_Notation_term_u2203_x21_____x2c___00__closed__11_once, _init_lp_mathlib_Mathlib_Notation_term_u2203_x21_____x2c___00__closed__11);
return v___x_365_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Notation___aux__Mathlib__Basic__ExistsUnique______macroRules__Mathlib__Notation__term_u2203_x21_____x2c____1___closed__3(void){
_start:
{
lean_object* v___x_373_; lean_object* v___x_374_; 
v___x_373_ = ((lean_object*)(lp_mathlib_Mathlib_Notation___aux__Mathlib__Basic__ExistsUnique______macroRules__Mathlib__Notation__term_u2203_x21_____x2c____1___closed__2));
v___x_374_ = l_String_toRawSubstring_x27(v___x_373_);
return v___x_374_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Notation___aux__Mathlib__Basic__ExistsUnique______macroRules__Mathlib__Notation__term_u2203_x21_____x2c____1(lean_object* v_x_386_, lean_object* v_a_387_, lean_object* v_a_388_){
_start:
{
lean_object* v___x_389_; uint8_t v___x_390_; 
v___x_389_ = ((lean_object*)(lp_mathlib_Mathlib_Notation_term_u2203_x21_____x2c___00__closed__1));
lean_inc(v_x_386_);
v___x_390_ = l_Lean_Syntax_isOfKind(v_x_386_, v___x_389_);
if (v___x_390_ == 0)
{
lean_object* v___x_391_; lean_object* v___x_392_; 
lean_dec(v_x_386_);
v___x_391_ = lean_box(1);
v___x_392_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_392_, 0, v___x_391_);
lean_ctor_set(v___x_392_, 1, v_a_388_);
return v___x_392_;
}
else
{
lean_object* v___x_393_; lean_object* v___x_394_; lean_object* v___x_395_; uint8_t v___x_396_; 
v___x_393_ = lean_unsigned_to_nat(1u);
v___x_394_ = l_Lean_Syntax_getArg(v_x_386_, v___x_393_);
v___x_395_ = ((lean_object*)(lp_mathlib_Mathlib_Notation_isExplicitBinderSingular___closed__8));
lean_inc(v___x_394_);
v___x_396_ = l_Lean_Syntax_isOfKind(v___x_394_, v___x_395_);
if (v___x_396_ == 0)
{
lean_object* v___x_397_; lean_object* v___x_398_; 
lean_dec(v___x_394_);
lean_dec(v_x_386_);
v___x_397_ = lean_box(1);
v___x_398_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_398_, 0, v___x_397_);
lean_ctor_set(v___x_398_, 1, v_a_388_);
return v___x_398_;
}
else
{
lean_object* v___x_399_; lean_object* v___x_400_; lean_object* v___x_401_; uint8_t v___x_402_; 
v___x_399_ = lean_unsigned_to_nat(0u);
v___x_400_ = l_Lean_Syntax_getArg(v___x_394_, v___x_399_);
lean_dec(v___x_394_);
v___x_401_ = ((lean_object*)(lp_mathlib_Mathlib_Notation_unexpandExistsUnique___closed__9));
lean_inc(v___x_400_);
v___x_402_ = l_Lean_Syntax_isOfKind(v___x_400_, v___x_401_);
if (v___x_402_ == 0)
{
lean_object* v___x_403_; uint8_t v___x_404_; 
v___x_403_ = ((lean_object*)(lp_mathlib_Mathlib_Notation___aux__Mathlib__Basic__ExistsUnique______macroRules__Mathlib__Notation__term_u2203_x21_____x2c____1___closed__1));
v___x_404_ = l_Lean_Syntax_isOfKind(v___x_400_, v___x_403_);
if (v___x_404_ == 0)
{
lean_object* v___x_405_; lean_object* v___x_406_; 
lean_dec(v_x_386_);
v___x_405_ = lean_box(1);
v___x_406_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_406_, 0, v___x_405_);
lean_ctor_set(v___x_406_, 1, v_a_388_);
return v___x_406_;
}
else
{
lean_object* v_quotContext_407_; lean_object* v_currMacroScope_408_; lean_object* v_ref_409_; lean_object* v___x_410_; lean_object* v___x_411_; lean_object* v___x_412_; lean_object* v___x_413_; lean_object* v___x_414_; lean_object* v___x_415_; lean_object* v___x_416_; lean_object* v___x_417_; lean_object* v___x_418_; lean_object* v___x_419_; lean_object* v___x_420_; lean_object* v___x_421_; lean_object* v___x_422_; lean_object* v___x_423_; lean_object* v___x_424_; lean_object* v___x_425_; lean_object* v___x_426_; lean_object* v___x_427_; lean_object* v___x_428_; lean_object* v___x_429_; lean_object* v___x_430_; lean_object* v___x_431_; lean_object* v___x_432_; lean_object* v___x_433_; lean_object* v___x_434_; lean_object* v___x_435_; lean_object* v___x_436_; lean_object* v___x_437_; lean_object* v___x_438_; lean_object* v___x_439_; lean_object* v___x_440_; lean_object* v___x_441_; lean_object* v___x_442_; lean_object* v___x_443_; 
v_quotContext_407_ = lean_ctor_get(v_a_387_, 1);
v_currMacroScope_408_ = lean_ctor_get(v_a_387_, 2);
v_ref_409_ = lean_ctor_get(v_a_387_, 5);
v___x_410_ = lean_unsigned_to_nat(2u);
v___x_411_ = l_Lean_Syntax_getArg(v_x_386_, v___x_410_);
v___x_412_ = lean_unsigned_to_nat(4u);
v___x_413_ = l_Lean_Syntax_getArg(v_x_386_, v___x_412_);
lean_dec(v_x_386_);
v___x_414_ = l_Lean_SourceInfo_fromRef(v_ref_409_, v___x_402_);
v___x_415_ = ((lean_object*)(lp_mathlib_Mathlib_Notation_term_u2203_x21___x2c___00__closed__3));
v___x_416_ = ((lean_object*)(lp_mathlib_Mathlib_Notation_term_u2203_x21___x2c___00__closed__6));
lean_inc_n(v___x_414_, 12);
v___x_417_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_417_, 0, v___x_414_);
lean_ctor_set(v___x_417_, 1, v___x_416_);
v___x_418_ = ((lean_object*)(lp_mathlib_Mathlib_Notation_isExplicitBinderSingular___closed__2));
v___x_419_ = ((lean_object*)(lp_mathlib_Mathlib_Notation_isExplicitBinderSingular___closed__4));
v___x_420_ = ((lean_object*)(lp_mathlib_Mathlib_Notation_unexpandExistsUnique___closed__17));
v___x_421_ = lean_obj_once(&lp_mathlib_Mathlib_Notation___aux__Mathlib__Basic__ExistsUnique______macroRules__Mathlib__Notation__term_u2203_x21_____x2c____1___closed__3, &lp_mathlib_Mathlib_Notation___aux__Mathlib__Basic__ExistsUnique______macroRules__Mathlib__Notation__term_u2203_x21_____x2c____1___closed__3_once, _init_lp_mathlib_Mathlib_Notation___aux__Mathlib__Basic__ExistsUnique______macroRules__Mathlib__Notation__term_u2203_x21_____x2c____1___closed__3);
v___x_422_ = ((lean_object*)(lp_mathlib_Mathlib_Notation___aux__Mathlib__Basic__ExistsUnique______macroRules__Mathlib__Notation__term_u2203_x21_____x2c____1___closed__4));
lean_inc(v_currMacroScope_408_);
lean_inc(v_quotContext_407_);
v___x_423_ = l_Lean_addMacroScope(v_quotContext_407_, v___x_422_, v_currMacroScope_408_);
v___x_424_ = lean_box(0);
v___x_425_ = lean_alloc_ctor(3, 4, 0);
lean_ctor_set(v___x_425_, 0, v___x_414_);
lean_ctor_set(v___x_425_, 1, v___x_421_);
lean_ctor_set(v___x_425_, 2, v___x_423_);
lean_ctor_set(v___x_425_, 3, v___x_424_);
lean_inc_ref(v___x_425_);
v___x_426_ = l_Lean_Syntax_node1(v___x_414_, v___x_395_, v___x_425_);
v___x_427_ = l_Lean_Syntax_node1(v___x_414_, v___x_420_, v___x_426_);
v___x_428_ = lean_obj_once(&lp_mathlib_Mathlib_Notation_unexpandExistsUnique___closed__20, &lp_mathlib_Mathlib_Notation_unexpandExistsUnique___closed__20_once, _init_lp_mathlib_Mathlib_Notation_unexpandExistsUnique___closed__20);
v___x_429_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_429_, 0, v___x_414_);
lean_ctor_set(v___x_429_, 1, v___x_420_);
lean_ctor_set(v___x_429_, 2, v___x_428_);
v___x_430_ = l_Lean_Syntax_node2(v___x_414_, v___x_419_, v___x_427_, v___x_429_);
v___x_431_ = l_Lean_Syntax_node1(v___x_414_, v___x_418_, v___x_430_);
v___x_432_ = ((lean_object*)(lp_mathlib_Mathlib_Notation_unexpandExistsUnique___closed__19));
v___x_433_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_433_, 0, v___x_414_);
lean_ctor_set(v___x_433_, 1, v___x_432_);
v___x_434_ = ((lean_object*)(lp_mathlib_Mathlib_Notation___aux__Mathlib__Basic__ExistsUnique______macroRules__Mathlib__Notation__term_u2203_x21_____x2c____1___closed__6));
v___x_435_ = ((lean_object*)(lp_mathlib_Mathlib_Notation___aux__Mathlib__Basic__ExistsUnique______macroRules__Mathlib__Notation__term_u2203_x21_____x2c____1___closed__8));
v___x_436_ = ((lean_object*)(lp_mathlib_Mathlib_Notation___aux__Mathlib__Basic__ExistsUnique______macroRules__Mathlib__Notation__term_u2203_x21_____x2c____1___closed__9));
v___x_437_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_437_, 0, v___x_414_);
lean_ctor_set(v___x_437_, 1, v___x_436_);
v___x_438_ = l_Lean_Syntax_node3(v___x_414_, v___x_435_, v___x_437_, v___x_425_, v___x_411_);
v___x_439_ = ((lean_object*)(lp_mathlib_Mathlib_Notation___aux__Mathlib__Basic__ExistsUnique______macroRules__Mathlib__Notation__term_u2203_x21_____x2c____1___closed__10));
v___x_440_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_440_, 0, v___x_414_);
lean_ctor_set(v___x_440_, 1, v___x_439_);
v___x_441_ = l_Lean_Syntax_node3(v___x_414_, v___x_434_, v___x_438_, v___x_440_, v___x_413_);
v___x_442_ = l_Lean_Syntax_node4(v___x_414_, v___x_415_, v___x_417_, v___x_431_, v___x_433_, v___x_441_);
v___x_443_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_443_, 0, v___x_442_);
lean_ctor_set(v___x_443_, 1, v_a_388_);
return v___x_443_;
}
}
else
{
lean_object* v_ref_444_; lean_object* v___x_445_; lean_object* v___x_446_; lean_object* v___x_447_; lean_object* v___x_448_; uint8_t v___x_449_; lean_object* v___x_450_; lean_object* v___x_451_; lean_object* v___x_452_; lean_object* v___x_453_; lean_object* v___x_454_; lean_object* v___x_455_; lean_object* v___x_456_; lean_object* v___x_457_; lean_object* v___x_458_; lean_object* v___x_459_; lean_object* v___x_460_; lean_object* v___x_461_; lean_object* v___x_462_; lean_object* v___x_463_; lean_object* v___x_464_; lean_object* v___x_465_; lean_object* v___x_466_; lean_object* v___x_467_; lean_object* v___x_468_; lean_object* v___x_469_; lean_object* v___x_470_; lean_object* v___x_471_; lean_object* v___x_472_; lean_object* v___x_473_; lean_object* v___x_474_; 
v_ref_444_ = lean_ctor_get(v_a_387_, 5);
v___x_445_ = lean_unsigned_to_nat(2u);
v___x_446_ = l_Lean_Syntax_getArg(v_x_386_, v___x_445_);
v___x_447_ = lean_unsigned_to_nat(4u);
v___x_448_ = l_Lean_Syntax_getArg(v_x_386_, v___x_447_);
lean_dec(v_x_386_);
v___x_449_ = 0;
v___x_450_ = l_Lean_SourceInfo_fromRef(v_ref_444_, v___x_449_);
v___x_451_ = ((lean_object*)(lp_mathlib_Mathlib_Notation_term_u2203_x21___x2c___00__closed__3));
v___x_452_ = ((lean_object*)(lp_mathlib_Mathlib_Notation_term_u2203_x21___x2c___00__closed__6));
lean_inc_n(v___x_450_, 11);
v___x_453_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_453_, 0, v___x_450_);
lean_ctor_set(v___x_453_, 1, v___x_452_);
v___x_454_ = ((lean_object*)(lp_mathlib_Mathlib_Notation_isExplicitBinderSingular___closed__2));
v___x_455_ = ((lean_object*)(lp_mathlib_Mathlib_Notation_isExplicitBinderSingular___closed__4));
v___x_456_ = ((lean_object*)(lp_mathlib_Mathlib_Notation_unexpandExistsUnique___closed__17));
lean_inc(v___x_400_);
v___x_457_ = l_Lean_Syntax_node1(v___x_450_, v___x_395_, v___x_400_);
v___x_458_ = l_Lean_Syntax_node1(v___x_450_, v___x_456_, v___x_457_);
v___x_459_ = lean_obj_once(&lp_mathlib_Mathlib_Notation_unexpandExistsUnique___closed__20, &lp_mathlib_Mathlib_Notation_unexpandExistsUnique___closed__20_once, _init_lp_mathlib_Mathlib_Notation_unexpandExistsUnique___closed__20);
v___x_460_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_460_, 0, v___x_450_);
lean_ctor_set(v___x_460_, 1, v___x_456_);
lean_ctor_set(v___x_460_, 2, v___x_459_);
v___x_461_ = l_Lean_Syntax_node2(v___x_450_, v___x_455_, v___x_458_, v___x_460_);
v___x_462_ = l_Lean_Syntax_node1(v___x_450_, v___x_454_, v___x_461_);
v___x_463_ = ((lean_object*)(lp_mathlib_Mathlib_Notation_unexpandExistsUnique___closed__19));
v___x_464_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_464_, 0, v___x_450_);
lean_ctor_set(v___x_464_, 1, v___x_463_);
v___x_465_ = ((lean_object*)(lp_mathlib_Mathlib_Notation___aux__Mathlib__Basic__ExistsUnique______macroRules__Mathlib__Notation__term_u2203_x21_____x2c____1___closed__6));
v___x_466_ = ((lean_object*)(lp_mathlib_Mathlib_Notation___aux__Mathlib__Basic__ExistsUnique______macroRules__Mathlib__Notation__term_u2203_x21_____x2c____1___closed__8));
v___x_467_ = ((lean_object*)(lp_mathlib_Mathlib_Notation___aux__Mathlib__Basic__ExistsUnique______macroRules__Mathlib__Notation__term_u2203_x21_____x2c____1___closed__9));
v___x_468_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_468_, 0, v___x_450_);
lean_ctor_set(v___x_468_, 1, v___x_467_);
v___x_469_ = l_Lean_Syntax_node3(v___x_450_, v___x_466_, v___x_468_, v___x_400_, v___x_446_);
v___x_470_ = ((lean_object*)(lp_mathlib_Mathlib_Notation___aux__Mathlib__Basic__ExistsUnique______macroRules__Mathlib__Notation__term_u2203_x21_____x2c____1___closed__10));
v___x_471_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_471_, 0, v___x_450_);
lean_ctor_set(v___x_471_, 1, v___x_470_);
v___x_472_ = l_Lean_Syntax_node3(v___x_450_, v___x_465_, v___x_469_, v___x_471_, v___x_448_);
v___x_473_ = l_Lean_Syntax_node4(v___x_450_, v___x_451_, v___x_453_, v___x_462_, v___x_464_, v___x_472_);
v___x_474_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_474_, 0, v___x_473_);
lean_ctor_set(v___x_474_, 1, v_a_388_);
return v___x_474_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Notation___aux__Mathlib__Basic__ExistsUnique______macroRules__Mathlib__Notation__term_u2203_x21_____x2c____1___boxed(lean_object* v_x_475_, lean_object* v_a_476_, lean_object* v_a_477_){
_start:
{
lean_object* v_res_478_; 
v_res_478_ = lp_mathlib_Mathlib_Notation___aux__Mathlib__Basic__ExistsUnique______macroRules__Mathlib__Notation__term_u2203_x21_____x2c____1(v_x_475_, v_a_476_, v_a_477_);
lean_dec_ref(v_a_476_);
return v_res_478_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_List_decidableBExistsUnique___redArg___lam__0(lean_object* v_inst_479_, uint8_t v___x_480_, lean_object* v_inst_481_, lean_object* v_head_482_, lean_object* v_a_483_){
_start:
{
lean_object* v___x_484_; uint8_t v___x_485_; 
lean_inc(v_a_483_);
v___x_484_ = lean_apply_1(v_inst_479_, v_a_483_);
v___x_485_ = lean_unbox(v___x_484_);
if (v___x_485_ == 0)
{
lean_dec(v_a_483_);
lean_dec(v_head_482_);
lean_dec_ref(v_inst_481_);
return v___x_480_;
}
else
{
lean_object* v___x_486_; uint8_t v___x_487_; 
v___x_486_ = lean_apply_2(v_inst_481_, v_head_482_, v_a_483_);
v___x_487_ = lean_unbox(v___x_486_);
return v___x_487_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_decidableBExistsUnique___redArg___lam__0___boxed(lean_object* v_inst_488_, lean_object* v___x_489_, lean_object* v_inst_490_, lean_object* v_head_491_, lean_object* v_a_492_){
_start:
{
uint8_t v___x_58__boxed_493_; uint8_t v_res_494_; lean_object* v_r_495_; 
v___x_58__boxed_493_ = lean_unbox(v___x_489_);
v_res_494_ = lp_mathlib_List_decidableBExistsUnique___redArg___lam__0(v_inst_488_, v___x_58__boxed_493_, v_inst_490_, v_head_491_, v_a_492_);
v_r_495_ = lean_box(v_res_494_);
return v_r_495_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_List_decidableBExistsUnique___redArg(lean_object* v_inst_496_, lean_object* v_inst_497_, lean_object* v_x_498_){
_start:
{
if (lean_obj_tag(v_x_498_) == 0)
{
uint8_t v___x_499_; 
lean_dec_ref(v_inst_497_);
lean_dec_ref(v_inst_496_);
v___x_499_ = 0;
return v___x_499_;
}
else
{
lean_object* v_head_500_; lean_object* v_tail_501_; lean_object* v___x_502_; uint8_t v___x_503_; 
v_head_500_ = lean_ctor_get(v_x_498_, 0);
lean_inc_n(v_head_500_, 2);
v_tail_501_ = lean_ctor_get(v_x_498_, 1);
lean_inc(v_tail_501_);
lean_dec_ref_known(v_x_498_, 2);
lean_inc_ref(v_inst_497_);
v___x_502_ = lean_apply_1(v_inst_497_, v_head_500_);
v___x_503_ = lean_unbox(v___x_502_);
if (v___x_503_ == 0)
{
lean_dec(v_head_500_);
v_x_498_ = v_tail_501_;
goto _start;
}
else
{
lean_object* v___f_505_; uint8_t v___x_506_; 
v___f_505_ = lean_alloc_closure((void*)(lp_mathlib_List_decidableBExistsUnique___redArg___lam__0___boxed), 5, 4);
lean_closure_set(v___f_505_, 0, v_inst_497_);
lean_closure_set(v___f_505_, 1, v___x_502_);
lean_closure_set(v___f_505_, 2, v_inst_496_);
lean_closure_set(v___f_505_, 3, v_head_500_);
v___x_506_ = l_List_decidableBAll___redArg(v___f_505_, v_tail_501_);
return v___x_506_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_decidableBExistsUnique___redArg___boxed(lean_object* v_inst_507_, lean_object* v_inst_508_, lean_object* v_x_509_){
_start:
{
uint8_t v_res_510_; lean_object* v_r_511_; 
v_res_510_ = lp_mathlib_List_decidableBExistsUnique___redArg(v_inst_507_, v_inst_508_, v_x_509_);
v_r_511_ = lean_box(v_res_510_);
return v_r_511_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_List_decidableBExistsUnique(lean_object* v_00_u03b1_512_, lean_object* v_inst_513_, lean_object* v_p_514_, lean_object* v_inst_515_, lean_object* v_x_516_){
_start:
{
uint8_t v___x_517_; 
v___x_517_ = lp_mathlib_List_decidableBExistsUnique___redArg(v_inst_513_, v_inst_515_, v_x_516_);
return v___x_517_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_decidableBExistsUnique___boxed(lean_object* v_00_u03b1_518_, lean_object* v_inst_519_, lean_object* v_p_520_, lean_object* v_inst_521_, lean_object* v_x_522_){
_start:
{
uint8_t v_res_523_; lean_object* v_r_524_; 
v_res_523_ = lp_mathlib_List_decidableBExistsUnique(v_00_u03b1_518_, v_inst_519_, v_p_520_, v_inst_521_, v_x_522_);
v_r_524_ = lean_box(v_res_523_);
return v_r_524_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Init(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_Basic_ExistsUnique(uint8_t builtin) {
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
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_Basic_ExistsUnique(uint8_t builtin) {
lean_object * res;
if (_G_meta_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_meta_initialized = true;
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
lp_mathlib_Mathlib_Notation_term_u2203_x21___x2c__ = _init_lp_mathlib_Mathlib_Notation_term_u2203_x21___x2c__();
lean_mark_persistent(lp_mathlib_Mathlib_Notation_term_u2203_x21___x2c__);
lp_mathlib_Mathlib_Notation_term_u2203_x21_____x2c__ = _init_lp_mathlib_Mathlib_Notation_term_u2203_x21_____x2c__();
lean_mark_persistent(lp_mathlib_Mathlib_Notation_term_u2203_x21_____x2c__);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Init(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Basic_ExistsUnique(uint8_t builtin) {
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
res = runtime_initialize_mathlib_Mathlib_Basic_ExistsUnique(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_Basic_ExistsUnique(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_Basic_ExistsUnique(builtin);
}
#ifdef __cplusplus
}
#endif
