// Lean compiler output
// Module: Mathlib.Data.List.Duplicate
// Imports: public import Init public meta import Init public import Mathlib.Data.List.Nodup
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
lean_object* l_Lean_Name_str___override(lean_object*, lean_object*);
lean_object* l_Lean_Name_num___override(lean_object*, lean_object*);
uint8_t l_Lean_Syntax_isOfKind(lean_object*, lean_object*);
lean_object* l_Lean_Syntax_getArg(lean_object*, lean_object*);
lean_object* l_Lean_SourceInfo_fromRef(lean_object*, uint8_t);
lean_object* l_Lean_Name_mkStr4(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_String_toRawSubstring_x27(lean_object*);
lean_object* l_Lean_Name_mkStr2(lean_object*, lean_object*);
lean_object* l_Lean_addMacroScope(lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Name_mkStr1(lean_object*);
lean_object* l_Lean_Syntax_node2(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_instBEqOfDecidableEq___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*);
uint8_t l_List_elem___redArg(lean_object*, lean_object*, lean_object*);
uint8_t l_Lean_Syntax_matchesNull(lean_object*, lean_object*);
lean_object* l_Lean_replaceRef(lean_object*, lean_object*);
lean_object* l_Lean_Syntax_node3(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib___private_Mathlib_Data_List_Duplicate_0__List_term___u2208_x2b___00__closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "_private"};
static const lean_object* lp_mathlib___private_Mathlib_Data_List_Duplicate_0__List_term___u2208_x2b___00__closed__0 = (const lean_object*)&lp_mathlib___private_Mathlib_Data_List_Duplicate_0__List_term___u2208_x2b___00__closed__0_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Data_List_Duplicate_0__List_term___u2208_x2b___00__closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Data_List_Duplicate_0__List_term___u2208_x2b___00__closed__0_value),LEAN_SCALAR_PTR_LITERAL(103, 214, 75, 80, 34, 198, 193, 153)}};
static const lean_object* lp_mathlib___private_Mathlib_Data_List_Duplicate_0__List_term___u2208_x2b___00__closed__1 = (const lean_object*)&lp_mathlib___private_Mathlib_Data_List_Duplicate_0__List_term___u2208_x2b___00__closed__1_value;
static const lean_string_object lp_mathlib___private_Mathlib_Data_List_Duplicate_0__List_term___u2208_x2b___00__closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "Mathlib"};
static const lean_object* lp_mathlib___private_Mathlib_Data_List_Duplicate_0__List_term___u2208_x2b___00__closed__2 = (const lean_object*)&lp_mathlib___private_Mathlib_Data_List_Duplicate_0__List_term___u2208_x2b___00__closed__2_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Data_List_Duplicate_0__List_term___u2208_x2b___00__closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Data_List_Duplicate_0__List_term___u2208_x2b___00__closed__1_value),((lean_object*)&lp_mathlib___private_Mathlib_Data_List_Duplicate_0__List_term___u2208_x2b___00__closed__2_value),LEAN_SCALAR_PTR_LITERAL(234, 232, 174, 134, 127, 136, 69, 92)}};
static const lean_object* lp_mathlib___private_Mathlib_Data_List_Duplicate_0__List_term___u2208_x2b___00__closed__3 = (const lean_object*)&lp_mathlib___private_Mathlib_Data_List_Duplicate_0__List_term___u2208_x2b___00__closed__3_value;
static const lean_string_object lp_mathlib___private_Mathlib_Data_List_Duplicate_0__List_term___u2208_x2b___00__closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "Data"};
static const lean_object* lp_mathlib___private_Mathlib_Data_List_Duplicate_0__List_term___u2208_x2b___00__closed__4 = (const lean_object*)&lp_mathlib___private_Mathlib_Data_List_Duplicate_0__List_term___u2208_x2b___00__closed__4_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Data_List_Duplicate_0__List_term___u2208_x2b___00__closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Data_List_Duplicate_0__List_term___u2208_x2b___00__closed__3_value),((lean_object*)&lp_mathlib___private_Mathlib_Data_List_Duplicate_0__List_term___u2208_x2b___00__closed__4_value),LEAN_SCALAR_PTR_LITERAL(149, 48, 214, 5, 44, 128, 44, 12)}};
static const lean_object* lp_mathlib___private_Mathlib_Data_List_Duplicate_0__List_term___u2208_x2b___00__closed__5 = (const lean_object*)&lp_mathlib___private_Mathlib_Data_List_Duplicate_0__List_term___u2208_x2b___00__closed__5_value;
static const lean_string_object lp_mathlib___private_Mathlib_Data_List_Duplicate_0__List_term___u2208_x2b___00__closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "List"};
static const lean_object* lp_mathlib___private_Mathlib_Data_List_Duplicate_0__List_term___u2208_x2b___00__closed__6 = (const lean_object*)&lp_mathlib___private_Mathlib_Data_List_Duplicate_0__List_term___u2208_x2b___00__closed__6_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Data_List_Duplicate_0__List_term___u2208_x2b___00__closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Data_List_Duplicate_0__List_term___u2208_x2b___00__closed__5_value),((lean_object*)&lp_mathlib___private_Mathlib_Data_List_Duplicate_0__List_term___u2208_x2b___00__closed__6_value),LEAN_SCALAR_PTR_LITERAL(123, 214, 204, 50, 8, 169, 217, 159)}};
static const lean_object* lp_mathlib___private_Mathlib_Data_List_Duplicate_0__List_term___u2208_x2b___00__closed__7 = (const lean_object*)&lp_mathlib___private_Mathlib_Data_List_Duplicate_0__List_term___u2208_x2b___00__closed__7_value;
static const lean_string_object lp_mathlib___private_Mathlib_Data_List_Duplicate_0__List_term___u2208_x2b___00__closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 9, .m_data = "Duplicate"};
static const lean_object* lp_mathlib___private_Mathlib_Data_List_Duplicate_0__List_term___u2208_x2b___00__closed__8 = (const lean_object*)&lp_mathlib___private_Mathlib_Data_List_Duplicate_0__List_term___u2208_x2b___00__closed__8_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Data_List_Duplicate_0__List_term___u2208_x2b___00__closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Data_List_Duplicate_0__List_term___u2208_x2b___00__closed__7_value),((lean_object*)&lp_mathlib___private_Mathlib_Data_List_Duplicate_0__List_term___u2208_x2b___00__closed__8_value),LEAN_SCALAR_PTR_LITERAL(111, 33, 155, 216, 50, 101, 21, 30)}};
static const lean_object* lp_mathlib___private_Mathlib_Data_List_Duplicate_0__List_term___u2208_x2b___00__closed__9 = (const lean_object*)&lp_mathlib___private_Mathlib_Data_List_Duplicate_0__List_term___u2208_x2b___00__closed__9_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Data_List_Duplicate_0__List_term___u2208_x2b___00__closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Data_List_Duplicate_0__List_term___u2208_x2b___00__closed__9_value),((lean_object*)(((size_t)(0) << 1) | 1)),LEAN_SCALAR_PTR_LITERAL(130, 131, 140, 109, 141, 139, 87, 130)}};
static const lean_object* lp_mathlib___private_Mathlib_Data_List_Duplicate_0__List_term___u2208_x2b___00__closed__10 = (const lean_object*)&lp_mathlib___private_Mathlib_Data_List_Duplicate_0__List_term___u2208_x2b___00__closed__10_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Data_List_Duplicate_0__List_term___u2208_x2b___00__closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Data_List_Duplicate_0__List_term___u2208_x2b___00__closed__10_value),((lean_object*)&lp_mathlib___private_Mathlib_Data_List_Duplicate_0__List_term___u2208_x2b___00__closed__6_value),LEAN_SCALAR_PTR_LITERAL(184, 29, 55, 241, 190, 26, 120, 253)}};
static const lean_object* lp_mathlib___private_Mathlib_Data_List_Duplicate_0__List_term___u2208_x2b___00__closed__11 = (const lean_object*)&lp_mathlib___private_Mathlib_Data_List_Duplicate_0__List_term___u2208_x2b___00__closed__11_value;
static const lean_string_object lp_mathlib___private_Mathlib_Data_List_Duplicate_0__List_term___u2208_x2b___00__closed__12_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 11, .m_capacity = 11, .m_length = 8, .m_data = "term_∈+_"};
static const lean_object* lp_mathlib___private_Mathlib_Data_List_Duplicate_0__List_term___u2208_x2b___00__closed__12 = (const lean_object*)&lp_mathlib___private_Mathlib_Data_List_Duplicate_0__List_term___u2208_x2b___00__closed__12_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Data_List_Duplicate_0__List_term___u2208_x2b___00__closed__13_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Data_List_Duplicate_0__List_term___u2208_x2b___00__closed__11_value),((lean_object*)&lp_mathlib___private_Mathlib_Data_List_Duplicate_0__List_term___u2208_x2b___00__closed__12_value),LEAN_SCALAR_PTR_LITERAL(55, 20, 54, 173, 37, 81, 182, 160)}};
static const lean_object* lp_mathlib___private_Mathlib_Data_List_Duplicate_0__List_term___u2208_x2b___00__closed__13 = (const lean_object*)&lp_mathlib___private_Mathlib_Data_List_Duplicate_0__List_term___u2208_x2b___00__closed__13_value;
static const lean_string_object lp_mathlib___private_Mathlib_Data_List_Duplicate_0__List_term___u2208_x2b___00__closed__14_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "andthen"};
static const lean_object* lp_mathlib___private_Mathlib_Data_List_Duplicate_0__List_term___u2208_x2b___00__closed__14 = (const lean_object*)&lp_mathlib___private_Mathlib_Data_List_Duplicate_0__List_term___u2208_x2b___00__closed__14_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Data_List_Duplicate_0__List_term___u2208_x2b___00__closed__15_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Data_List_Duplicate_0__List_term___u2208_x2b___00__closed__14_value),LEAN_SCALAR_PTR_LITERAL(40, 255, 78, 30, 143, 119, 117, 174)}};
static const lean_object* lp_mathlib___private_Mathlib_Data_List_Duplicate_0__List_term___u2208_x2b___00__closed__15 = (const lean_object*)&lp_mathlib___private_Mathlib_Data_List_Duplicate_0__List_term___u2208_x2b___00__closed__15_value;
static const lean_string_object lp_mathlib___private_Mathlib_Data_List_Duplicate_0__List_term___u2208_x2b___00__closed__16_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 4, .m_data = " ∈+ "};
static const lean_object* lp_mathlib___private_Mathlib_Data_List_Duplicate_0__List_term___u2208_x2b___00__closed__16 = (const lean_object*)&lp_mathlib___private_Mathlib_Data_List_Duplicate_0__List_term___u2208_x2b___00__closed__16_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Data_List_Duplicate_0__List_term___u2208_x2b___00__closed__17_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 5}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Data_List_Duplicate_0__List_term___u2208_x2b___00__closed__16_value)}};
static const lean_object* lp_mathlib___private_Mathlib_Data_List_Duplicate_0__List_term___u2208_x2b___00__closed__17 = (const lean_object*)&lp_mathlib___private_Mathlib_Data_List_Duplicate_0__List_term___u2208_x2b___00__closed__17_value;
static const lean_string_object lp_mathlib___private_Mathlib_Data_List_Duplicate_0__List_term___u2208_x2b___00__closed__18_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "term"};
static const lean_object* lp_mathlib___private_Mathlib_Data_List_Duplicate_0__List_term___u2208_x2b___00__closed__18 = (const lean_object*)&lp_mathlib___private_Mathlib_Data_List_Duplicate_0__List_term___u2208_x2b___00__closed__18_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Data_List_Duplicate_0__List_term___u2208_x2b___00__closed__19_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Data_List_Duplicate_0__List_term___u2208_x2b___00__closed__18_value),LEAN_SCALAR_PTR_LITERAL(187, 230, 181, 162, 253, 146, 122, 119)}};
static const lean_object* lp_mathlib___private_Mathlib_Data_List_Duplicate_0__List_term___u2208_x2b___00__closed__19 = (const lean_object*)&lp_mathlib___private_Mathlib_Data_List_Duplicate_0__List_term___u2208_x2b___00__closed__19_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Data_List_Duplicate_0__List_term___u2208_x2b___00__closed__20_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 7}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Data_List_Duplicate_0__List_term___u2208_x2b___00__closed__19_value),((lean_object*)(((size_t)(51) << 1) | 1))}};
static const lean_object* lp_mathlib___private_Mathlib_Data_List_Duplicate_0__List_term___u2208_x2b___00__closed__20 = (const lean_object*)&lp_mathlib___private_Mathlib_Data_List_Duplicate_0__List_term___u2208_x2b___00__closed__20_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Data_List_Duplicate_0__List_term___u2208_x2b___00__closed__21_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Data_List_Duplicate_0__List_term___u2208_x2b___00__closed__15_value),((lean_object*)&lp_mathlib___private_Mathlib_Data_List_Duplicate_0__List_term___u2208_x2b___00__closed__17_value),((lean_object*)&lp_mathlib___private_Mathlib_Data_List_Duplicate_0__List_term___u2208_x2b___00__closed__20_value)}};
static const lean_object* lp_mathlib___private_Mathlib_Data_List_Duplicate_0__List_term___u2208_x2b___00__closed__21 = (const lean_object*)&lp_mathlib___private_Mathlib_Data_List_Duplicate_0__List_term___u2208_x2b___00__closed__21_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Data_List_Duplicate_0__List_term___u2208_x2b___00__closed__22_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*4 + 0, .m_other = 4, .m_tag = 4}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Data_List_Duplicate_0__List_term___u2208_x2b___00__closed__13_value),((lean_object*)(((size_t)(50) << 1) | 1)),((lean_object*)(((size_t)(50) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Data_List_Duplicate_0__List_term___u2208_x2b___00__closed__21_value)}};
static const lean_object* lp_mathlib___private_Mathlib_Data_List_Duplicate_0__List_term___u2208_x2b___00__closed__22 = (const lean_object*)&lp_mathlib___private_Mathlib_Data_List_Duplicate_0__List_term___u2208_x2b___00__closed__22_value;
LEAN_EXPORT const lean_object* lp_mathlib___private_Mathlib_Data_List_Duplicate_0__List_term___u2208_x2b__ = (const lean_object*)&lp_mathlib___private_Mathlib_Data_List_Duplicate_0__List_term___u2208_x2b___00__closed__22_value;
static const lean_string_object lp_mathlib___private_Mathlib_Data_List_Duplicate_0__List___aux__Mathlib__Data__List__Duplicate______macroRules____private__Mathlib__Data__List__Duplicate__0__List__term___u2208_x2b____1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "Lean"};
static const lean_object* lp_mathlib___private_Mathlib_Data_List_Duplicate_0__List___aux__Mathlib__Data__List__Duplicate______macroRules____private__Mathlib__Data__List__Duplicate__0__List__term___u2208_x2b____1___closed__0 = (const lean_object*)&lp_mathlib___private_Mathlib_Data_List_Duplicate_0__List___aux__Mathlib__Data__List__Duplicate______macroRules____private__Mathlib__Data__List__Duplicate__0__List__term___u2208_x2b____1___closed__0_value;
static const lean_string_object lp_mathlib___private_Mathlib_Data_List_Duplicate_0__List___aux__Mathlib__Data__List__Duplicate______macroRules____private__Mathlib__Data__List__Duplicate__0__List__term___u2208_x2b____1___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "Parser"};
static const lean_object* lp_mathlib___private_Mathlib_Data_List_Duplicate_0__List___aux__Mathlib__Data__List__Duplicate______macroRules____private__Mathlib__Data__List__Duplicate__0__List__term___u2208_x2b____1___closed__1 = (const lean_object*)&lp_mathlib___private_Mathlib_Data_List_Duplicate_0__List___aux__Mathlib__Data__List__Duplicate______macroRules____private__Mathlib__Data__List__Duplicate__0__List__term___u2208_x2b____1___closed__1_value;
static const lean_string_object lp_mathlib___private_Mathlib_Data_List_Duplicate_0__List___aux__Mathlib__Data__List__Duplicate______macroRules____private__Mathlib__Data__List__Duplicate__0__List__term___u2208_x2b____1___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "Term"};
static const lean_object* lp_mathlib___private_Mathlib_Data_List_Duplicate_0__List___aux__Mathlib__Data__List__Duplicate______macroRules____private__Mathlib__Data__List__Duplicate__0__List__term___u2208_x2b____1___closed__2 = (const lean_object*)&lp_mathlib___private_Mathlib_Data_List_Duplicate_0__List___aux__Mathlib__Data__List__Duplicate______macroRules____private__Mathlib__Data__List__Duplicate__0__List__term___u2208_x2b____1___closed__2_value;
static const lean_string_object lp_mathlib___private_Mathlib_Data_List_Duplicate_0__List___aux__Mathlib__Data__List__Duplicate______macroRules____private__Mathlib__Data__List__Duplicate__0__List__term___u2208_x2b____1___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = "app"};
static const lean_object* lp_mathlib___private_Mathlib_Data_List_Duplicate_0__List___aux__Mathlib__Data__List__Duplicate______macroRules____private__Mathlib__Data__List__Duplicate__0__List__term___u2208_x2b____1___closed__3 = (const lean_object*)&lp_mathlib___private_Mathlib_Data_List_Duplicate_0__List___aux__Mathlib__Data__List__Duplicate______macroRules____private__Mathlib__Data__List__Duplicate__0__List__term___u2208_x2b____1___closed__3_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Data_List_Duplicate_0__List___aux__Mathlib__Data__List__Duplicate______macroRules____private__Mathlib__Data__List__Duplicate__0__List__term___u2208_x2b____1___closed__4_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Data_List_Duplicate_0__List___aux__Mathlib__Data__List__Duplicate______macroRules____private__Mathlib__Data__List__Duplicate__0__List__term___u2208_x2b____1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Data_List_Duplicate_0__List___aux__Mathlib__Data__List__Duplicate______macroRules____private__Mathlib__Data__List__Duplicate__0__List__term___u2208_x2b____1___closed__4_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Data_List_Duplicate_0__List___aux__Mathlib__Data__List__Duplicate______macroRules____private__Mathlib__Data__List__Duplicate__0__List__term___u2208_x2b____1___closed__4_value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Data_List_Duplicate_0__List___aux__Mathlib__Data__List__Duplicate______macroRules____private__Mathlib__Data__List__Duplicate__0__List__term___u2208_x2b____1___closed__1_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Data_List_Duplicate_0__List___aux__Mathlib__Data__List__Duplicate______macroRules____private__Mathlib__Data__List__Duplicate__0__List__term___u2208_x2b____1___closed__4_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Data_List_Duplicate_0__List___aux__Mathlib__Data__List__Duplicate______macroRules____private__Mathlib__Data__List__Duplicate__0__List__term___u2208_x2b____1___closed__4_value_aux_1),((lean_object*)&lp_mathlib___private_Mathlib_Data_List_Duplicate_0__List___aux__Mathlib__Data__List__Duplicate______macroRules____private__Mathlib__Data__List__Duplicate__0__List__term___u2208_x2b____1___closed__2_value),LEAN_SCALAR_PTR_LITERAL(75, 170, 162, 138, 136, 204, 251, 229)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Data_List_Duplicate_0__List___aux__Mathlib__Data__List__Duplicate______macroRules____private__Mathlib__Data__List__Duplicate__0__List__term___u2208_x2b____1___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Data_List_Duplicate_0__List___aux__Mathlib__Data__List__Duplicate______macroRules____private__Mathlib__Data__List__Duplicate__0__List__term___u2208_x2b____1___closed__4_value_aux_2),((lean_object*)&lp_mathlib___private_Mathlib_Data_List_Duplicate_0__List___aux__Mathlib__Data__List__Duplicate______macroRules____private__Mathlib__Data__List__Duplicate__0__List__term___u2208_x2b____1___closed__3_value),LEAN_SCALAR_PTR_LITERAL(69, 118, 10, 41, 220, 156, 243, 179)}};
static const lean_object* lp_mathlib___private_Mathlib_Data_List_Duplicate_0__List___aux__Mathlib__Data__List__Duplicate______macroRules____private__Mathlib__Data__List__Duplicate__0__List__term___u2208_x2b____1___closed__4 = (const lean_object*)&lp_mathlib___private_Mathlib_Data_List_Duplicate_0__List___aux__Mathlib__Data__List__Duplicate______macroRules____private__Mathlib__Data__List__Duplicate__0__List__term___u2208_x2b____1___closed__4_value;
static const lean_string_object lp_mathlib___private_Mathlib_Data_List_Duplicate_0__List___aux__Mathlib__Data__List__Duplicate______macroRules____private__Mathlib__Data__List__Duplicate__0__List__term___u2208_x2b____1___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 15, .m_capacity = 15, .m_length = 14, .m_data = "List.Duplicate"};
static const lean_object* lp_mathlib___private_Mathlib_Data_List_Duplicate_0__List___aux__Mathlib__Data__List__Duplicate______macroRules____private__Mathlib__Data__List__Duplicate__0__List__term___u2208_x2b____1___closed__5 = (const lean_object*)&lp_mathlib___private_Mathlib_Data_List_Duplicate_0__List___aux__Mathlib__Data__List__Duplicate______macroRules____private__Mathlib__Data__List__Duplicate__0__List__term___u2208_x2b____1___closed__5_value;
static lean_once_cell_t lp_mathlib___private_Mathlib_Data_List_Duplicate_0__List___aux__Mathlib__Data__List__Duplicate______macroRules____private__Mathlib__Data__List__Duplicate__0__List__term___u2208_x2b____1___closed__6_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Data_List_Duplicate_0__List___aux__Mathlib__Data__List__Duplicate______macroRules____private__Mathlib__Data__List__Duplicate__0__List__term___u2208_x2b____1___closed__6;
static const lean_ctor_object lp_mathlib___private_Mathlib_Data_List_Duplicate_0__List___aux__Mathlib__Data__List__Duplicate______macroRules____private__Mathlib__Data__List__Duplicate__0__List__term___u2208_x2b____1___closed__7_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Data_List_Duplicate_0__List_term___u2208_x2b___00__closed__6_value),LEAN_SCALAR_PTR_LITERAL(245, 188, 225, 225, 165, 5, 251, 132)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Data_List_Duplicate_0__List___aux__Mathlib__Data__List__Duplicate______macroRules____private__Mathlib__Data__List__Duplicate__0__List__term___u2208_x2b____1___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Data_List_Duplicate_0__List___aux__Mathlib__Data__List__Duplicate______macroRules____private__Mathlib__Data__List__Duplicate__0__List__term___u2208_x2b____1___closed__7_value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Data_List_Duplicate_0__List_term___u2208_x2b___00__closed__8_value),LEAN_SCALAR_PTR_LITERAL(193, 40, 194, 90, 152, 150, 185, 134)}};
static const lean_object* lp_mathlib___private_Mathlib_Data_List_Duplicate_0__List___aux__Mathlib__Data__List__Duplicate______macroRules____private__Mathlib__Data__List__Duplicate__0__List__term___u2208_x2b____1___closed__7 = (const lean_object*)&lp_mathlib___private_Mathlib_Data_List_Duplicate_0__List___aux__Mathlib__Data__List__Duplicate______macroRules____private__Mathlib__Data__List__Duplicate__0__List__term___u2208_x2b____1___closed__7_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Data_List_Duplicate_0__List___aux__Mathlib__Data__List__Duplicate______macroRules____private__Mathlib__Data__List__Duplicate__0__List__term___u2208_x2b____1___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Data_List_Duplicate_0__List___aux__Mathlib__Data__List__Duplicate______macroRules____private__Mathlib__Data__List__Duplicate__0__List__term___u2208_x2b____1___closed__7_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib___private_Mathlib_Data_List_Duplicate_0__List___aux__Mathlib__Data__List__Duplicate______macroRules____private__Mathlib__Data__List__Duplicate__0__List__term___u2208_x2b____1___closed__8 = (const lean_object*)&lp_mathlib___private_Mathlib_Data_List_Duplicate_0__List___aux__Mathlib__Data__List__Duplicate______macroRules____private__Mathlib__Data__List__Duplicate__0__List__term___u2208_x2b____1___closed__8_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Data_List_Duplicate_0__List___aux__Mathlib__Data__List__Duplicate______macroRules____private__Mathlib__Data__List__Duplicate__0__List__term___u2208_x2b____1___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Data_List_Duplicate_0__List___aux__Mathlib__Data__List__Duplicate______macroRules____private__Mathlib__Data__List__Duplicate__0__List__term___u2208_x2b____1___closed__7_value)}};
static const lean_object* lp_mathlib___private_Mathlib_Data_List_Duplicate_0__List___aux__Mathlib__Data__List__Duplicate______macroRules____private__Mathlib__Data__List__Duplicate__0__List__term___u2208_x2b____1___closed__9 = (const lean_object*)&lp_mathlib___private_Mathlib_Data_List_Duplicate_0__List___aux__Mathlib__Data__List__Duplicate______macroRules____private__Mathlib__Data__List__Duplicate__0__List__term___u2208_x2b____1___closed__9_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Data_List_Duplicate_0__List___aux__Mathlib__Data__List__Duplicate______macroRules____private__Mathlib__Data__List__Duplicate__0__List__term___u2208_x2b____1___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Data_List_Duplicate_0__List___aux__Mathlib__Data__List__Duplicate______macroRules____private__Mathlib__Data__List__Duplicate__0__List__term___u2208_x2b____1___closed__9_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib___private_Mathlib_Data_List_Duplicate_0__List___aux__Mathlib__Data__List__Duplicate______macroRules____private__Mathlib__Data__List__Duplicate__0__List__term___u2208_x2b____1___closed__10 = (const lean_object*)&lp_mathlib___private_Mathlib_Data_List_Duplicate_0__List___aux__Mathlib__Data__List__Duplicate______macroRules____private__Mathlib__Data__List__Duplicate__0__List__term___u2208_x2b____1___closed__10_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Data_List_Duplicate_0__List___aux__Mathlib__Data__List__Duplicate______macroRules____private__Mathlib__Data__List__Duplicate__0__List__term___u2208_x2b____1___closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Data_List_Duplicate_0__List___aux__Mathlib__Data__List__Duplicate______macroRules____private__Mathlib__Data__List__Duplicate__0__List__term___u2208_x2b____1___closed__8_value),((lean_object*)&lp_mathlib___private_Mathlib_Data_List_Duplicate_0__List___aux__Mathlib__Data__List__Duplicate______macroRules____private__Mathlib__Data__List__Duplicate__0__List__term___u2208_x2b____1___closed__10_value)}};
static const lean_object* lp_mathlib___private_Mathlib_Data_List_Duplicate_0__List___aux__Mathlib__Data__List__Duplicate______macroRules____private__Mathlib__Data__List__Duplicate__0__List__term___u2208_x2b____1___closed__11 = (const lean_object*)&lp_mathlib___private_Mathlib_Data_List_Duplicate_0__List___aux__Mathlib__Data__List__Duplicate______macroRules____private__Mathlib__Data__List__Duplicate__0__List__term___u2208_x2b____1___closed__11_value;
static const lean_string_object lp_mathlib___private_Mathlib_Data_List_Duplicate_0__List___aux__Mathlib__Data__List__Duplicate______macroRules____private__Mathlib__Data__List__Duplicate__0__List__term___u2208_x2b____1___closed__12_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "null"};
static const lean_object* lp_mathlib___private_Mathlib_Data_List_Duplicate_0__List___aux__Mathlib__Data__List__Duplicate______macroRules____private__Mathlib__Data__List__Duplicate__0__List__term___u2208_x2b____1___closed__12 = (const lean_object*)&lp_mathlib___private_Mathlib_Data_List_Duplicate_0__List___aux__Mathlib__Data__List__Duplicate______macroRules____private__Mathlib__Data__List__Duplicate__0__List__term___u2208_x2b____1___closed__12_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Data_List_Duplicate_0__List___aux__Mathlib__Data__List__Duplicate______macroRules____private__Mathlib__Data__List__Duplicate__0__List__term___u2208_x2b____1___closed__13_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Data_List_Duplicate_0__List___aux__Mathlib__Data__List__Duplicate______macroRules____private__Mathlib__Data__List__Duplicate__0__List__term___u2208_x2b____1___closed__12_value),LEAN_SCALAR_PTR_LITERAL(24, 58, 49, 223, 146, 207, 197, 136)}};
static const lean_object* lp_mathlib___private_Mathlib_Data_List_Duplicate_0__List___aux__Mathlib__Data__List__Duplicate______macroRules____private__Mathlib__Data__List__Duplicate__0__List__term___u2208_x2b____1___closed__13 = (const lean_object*)&lp_mathlib___private_Mathlib_Data_List_Duplicate_0__List___aux__Mathlib__Data__List__Duplicate______macroRules____private__Mathlib__Data__List__Duplicate__0__List__term___u2208_x2b____1___closed__13_value;
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Data_List_Duplicate_0__List___aux__Mathlib__Data__List__Duplicate______macroRules____private__Mathlib__Data__List__Duplicate__0__List__term___u2208_x2b____1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Data_List_Duplicate_0__List___aux__Mathlib__Data__List__Duplicate______macroRules____private__Mathlib__Data__List__Duplicate__0__List__term___u2208_x2b____1___boxed(lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib___private_Mathlib_Data_List_Duplicate_0__List___aux__Mathlib__Data__List__Duplicate______unexpand__List__Duplicate__1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "ident"};
static const lean_object* lp_mathlib___private_Mathlib_Data_List_Duplicate_0__List___aux__Mathlib__Data__List__Duplicate______unexpand__List__Duplicate__1___closed__0 = (const lean_object*)&lp_mathlib___private_Mathlib_Data_List_Duplicate_0__List___aux__Mathlib__Data__List__Duplicate______unexpand__List__Duplicate__1___closed__0_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Data_List_Duplicate_0__List___aux__Mathlib__Data__List__Duplicate______unexpand__List__Duplicate__1___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Data_List_Duplicate_0__List___aux__Mathlib__Data__List__Duplicate______unexpand__List__Duplicate__1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(52, 159, 208, 51, 14, 60, 6, 71)}};
static const lean_object* lp_mathlib___private_Mathlib_Data_List_Duplicate_0__List___aux__Mathlib__Data__List__Duplicate______unexpand__List__Duplicate__1___closed__1 = (const lean_object*)&lp_mathlib___private_Mathlib_Data_List_Duplicate_0__List___aux__Mathlib__Data__List__Duplicate______unexpand__List__Duplicate__1___closed__1_value;
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Data_List_Duplicate_0__List___aux__Mathlib__Data__List__Duplicate______unexpand__List__Duplicate__1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Data_List_Duplicate_0__List___aux__Mathlib__Data__List__Duplicate______unexpand__List__Duplicate__1___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_List_decidableDuplicate___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_decidableDuplicate___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_List_decidableDuplicate(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_decidableDuplicate___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
static lean_object* _init_lp_mathlib___private_Mathlib_Data_List_Duplicate_0__List___aux__Mathlib__Data__List__Duplicate______macroRules____private__Mathlib__Data__List__Duplicate__0__List__term___u2208_x2b____1___closed__6(void){
_start:
{
lean_object* v___x_62_; lean_object* v___x_63_; 
v___x_62_ = ((lean_object*)(lp_mathlib___private_Mathlib_Data_List_Duplicate_0__List___aux__Mathlib__Data__List__Duplicate______macroRules____private__Mathlib__Data__List__Duplicate__0__List__term___u2208_x2b____1___closed__5));
v___x_63_ = l_String_toRawSubstring_x27(v___x_62_);
return v___x_63_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Data_List_Duplicate_0__List___aux__Mathlib__Data__List__Duplicate______macroRules____private__Mathlib__Data__List__Duplicate__0__List__term___u2208_x2b____1(lean_object* v_x_81_, lean_object* v_a_82_, lean_object* v_a_83_){
_start:
{
lean_object* v___x_84_; lean_object* v___x_85_; uint8_t v___x_86_; 
v___x_84_ = lean_unsigned_to_nat(0u);
v___x_85_ = ((lean_object*)(lp_mathlib___private_Mathlib_Data_List_Duplicate_0__List_term___u2208_x2b___00__closed__13));
lean_inc(v_x_81_);
v___x_86_ = l_Lean_Syntax_isOfKind(v_x_81_, v___x_85_);
if (v___x_86_ == 0)
{
lean_object* v___x_87_; lean_object* v___x_88_; 
lean_dec(v_x_81_);
v___x_87_ = lean_box(1);
v___x_88_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_88_, 0, v___x_87_);
lean_ctor_set(v___x_88_, 1, v_a_83_);
return v___x_88_;
}
else
{
lean_object* v_quotContext_89_; lean_object* v_currMacroScope_90_; lean_object* v_ref_91_; lean_object* v___x_92_; lean_object* v___x_93_; lean_object* v___x_94_; uint8_t v___x_95_; lean_object* v___x_96_; lean_object* v___x_97_; lean_object* v___x_98_; lean_object* v___x_99_; lean_object* v___x_100_; lean_object* v___x_101_; lean_object* v___x_102_; lean_object* v___x_103_; lean_object* v___x_104_; lean_object* v___x_105_; lean_object* v___x_106_; 
v_quotContext_89_ = lean_ctor_get(v_a_82_, 1);
v_currMacroScope_90_ = lean_ctor_get(v_a_82_, 2);
v_ref_91_ = lean_ctor_get(v_a_82_, 5);
v___x_92_ = l_Lean_Syntax_getArg(v_x_81_, v___x_84_);
v___x_93_ = lean_unsigned_to_nat(2u);
v___x_94_ = l_Lean_Syntax_getArg(v_x_81_, v___x_93_);
lean_dec(v_x_81_);
v___x_95_ = 0;
v___x_96_ = l_Lean_SourceInfo_fromRef(v_ref_91_, v___x_95_);
v___x_97_ = ((lean_object*)(lp_mathlib___private_Mathlib_Data_List_Duplicate_0__List___aux__Mathlib__Data__List__Duplicate______macroRules____private__Mathlib__Data__List__Duplicate__0__List__term___u2208_x2b____1___closed__4));
v___x_98_ = lean_obj_once(&lp_mathlib___private_Mathlib_Data_List_Duplicate_0__List___aux__Mathlib__Data__List__Duplicate______macroRules____private__Mathlib__Data__List__Duplicate__0__List__term___u2208_x2b____1___closed__6, &lp_mathlib___private_Mathlib_Data_List_Duplicate_0__List___aux__Mathlib__Data__List__Duplicate______macroRules____private__Mathlib__Data__List__Duplicate__0__List__term___u2208_x2b____1___closed__6_once, _init_lp_mathlib___private_Mathlib_Data_List_Duplicate_0__List___aux__Mathlib__Data__List__Duplicate______macroRules____private__Mathlib__Data__List__Duplicate__0__List__term___u2208_x2b____1___closed__6);
v___x_99_ = ((lean_object*)(lp_mathlib___private_Mathlib_Data_List_Duplicate_0__List___aux__Mathlib__Data__List__Duplicate______macroRules____private__Mathlib__Data__List__Duplicate__0__List__term___u2208_x2b____1___closed__7));
lean_inc(v_currMacroScope_90_);
lean_inc(v_quotContext_89_);
v___x_100_ = l_Lean_addMacroScope(v_quotContext_89_, v___x_99_, v_currMacroScope_90_);
v___x_101_ = ((lean_object*)(lp_mathlib___private_Mathlib_Data_List_Duplicate_0__List___aux__Mathlib__Data__List__Duplicate______macroRules____private__Mathlib__Data__List__Duplicate__0__List__term___u2208_x2b____1___closed__11));
lean_inc_n(v___x_96_, 2);
v___x_102_ = lean_alloc_ctor(3, 4, 0);
lean_ctor_set(v___x_102_, 0, v___x_96_);
lean_ctor_set(v___x_102_, 1, v___x_98_);
lean_ctor_set(v___x_102_, 2, v___x_100_);
lean_ctor_set(v___x_102_, 3, v___x_101_);
v___x_103_ = ((lean_object*)(lp_mathlib___private_Mathlib_Data_List_Duplicate_0__List___aux__Mathlib__Data__List__Duplicate______macroRules____private__Mathlib__Data__List__Duplicate__0__List__term___u2208_x2b____1___closed__13));
v___x_104_ = l_Lean_Syntax_node2(v___x_96_, v___x_103_, v___x_92_, v___x_94_);
v___x_105_ = l_Lean_Syntax_node2(v___x_96_, v___x_97_, v___x_102_, v___x_104_);
v___x_106_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_106_, 0, v___x_105_);
lean_ctor_set(v___x_106_, 1, v_a_83_);
return v___x_106_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Data_List_Duplicate_0__List___aux__Mathlib__Data__List__Duplicate______macroRules____private__Mathlib__Data__List__Duplicate__0__List__term___u2208_x2b____1___boxed(lean_object* v_x_107_, lean_object* v_a_108_, lean_object* v_a_109_){
_start:
{
lean_object* v_res_110_; 
v_res_110_ = lp_mathlib___private_Mathlib_Data_List_Duplicate_0__List___aux__Mathlib__Data__List__Duplicate______macroRules____private__Mathlib__Data__List__Duplicate__0__List__term___u2208_x2b____1(v_x_107_, v_a_108_, v_a_109_);
lean_dec_ref(v_a_108_);
return v_res_110_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Data_List_Duplicate_0__List___aux__Mathlib__Data__List__Duplicate______unexpand__List__Duplicate__1(lean_object* v_x_114_, lean_object* v_a_115_, lean_object* v_a_116_){
_start:
{
lean_object* v___x_117_; uint8_t v___x_118_; 
v___x_117_ = ((lean_object*)(lp_mathlib___private_Mathlib_Data_List_Duplicate_0__List___aux__Mathlib__Data__List__Duplicate______macroRules____private__Mathlib__Data__List__Duplicate__0__List__term___u2208_x2b____1___closed__4));
lean_inc(v_x_114_);
v___x_118_ = l_Lean_Syntax_isOfKind(v_x_114_, v___x_117_);
if (v___x_118_ == 0)
{
lean_object* v___x_119_; lean_object* v___x_120_; 
lean_dec(v_x_114_);
v___x_119_ = lean_box(0);
v___x_120_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_120_, 0, v___x_119_);
lean_ctor_set(v___x_120_, 1, v_a_116_);
return v___x_120_;
}
else
{
lean_object* v___x_121_; lean_object* v___x_122_; lean_object* v___x_123_; uint8_t v___x_124_; 
v___x_121_ = lean_unsigned_to_nat(0u);
v___x_122_ = l_Lean_Syntax_getArg(v_x_114_, v___x_121_);
v___x_123_ = ((lean_object*)(lp_mathlib___private_Mathlib_Data_List_Duplicate_0__List___aux__Mathlib__Data__List__Duplicate______unexpand__List__Duplicate__1___closed__1));
lean_inc(v___x_122_);
v___x_124_ = l_Lean_Syntax_isOfKind(v___x_122_, v___x_123_);
if (v___x_124_ == 0)
{
lean_object* v___x_125_; lean_object* v___x_126_; 
lean_dec(v___x_122_);
lean_dec(v_x_114_);
v___x_125_ = lean_box(0);
v___x_126_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_126_, 0, v___x_125_);
lean_ctor_set(v___x_126_, 1, v_a_116_);
return v___x_126_;
}
else
{
lean_object* v___x_127_; lean_object* v___x_128_; lean_object* v___x_129_; uint8_t v___x_130_; 
v___x_127_ = lean_unsigned_to_nat(1u);
v___x_128_ = l_Lean_Syntax_getArg(v_x_114_, v___x_127_);
lean_dec(v_x_114_);
v___x_129_ = lean_unsigned_to_nat(2u);
lean_inc(v___x_128_);
v___x_130_ = l_Lean_Syntax_matchesNull(v___x_128_, v___x_129_);
if (v___x_130_ == 0)
{
lean_object* v___x_131_; lean_object* v___x_132_; 
lean_dec(v___x_128_);
lean_dec(v___x_122_);
v___x_131_ = lean_box(0);
v___x_132_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_132_, 0, v___x_131_);
lean_ctor_set(v___x_132_, 1, v_a_116_);
return v___x_132_;
}
else
{
lean_object* v___x_133_; lean_object* v___x_134_; lean_object* v_ref_135_; uint8_t v___x_136_; lean_object* v___x_137_; lean_object* v___x_138_; lean_object* v___x_139_; lean_object* v___x_140_; lean_object* v___x_141_; lean_object* v___x_142_; 
v___x_133_ = l_Lean_Syntax_getArg(v___x_128_, v___x_121_);
v___x_134_ = l_Lean_Syntax_getArg(v___x_128_, v___x_127_);
lean_dec(v___x_128_);
v_ref_135_ = l_Lean_replaceRef(v___x_122_, v_a_115_);
lean_dec(v___x_122_);
v___x_136_ = 0;
v___x_137_ = l_Lean_SourceInfo_fromRef(v_ref_135_, v___x_136_);
lean_dec(v_ref_135_);
v___x_138_ = ((lean_object*)(lp_mathlib___private_Mathlib_Data_List_Duplicate_0__List_term___u2208_x2b___00__closed__13));
v___x_139_ = ((lean_object*)(lp_mathlib___private_Mathlib_Data_List_Duplicate_0__List_term___u2208_x2b___00__closed__16));
lean_inc(v___x_137_);
v___x_140_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_140_, 0, v___x_137_);
lean_ctor_set(v___x_140_, 1, v___x_139_);
v___x_141_ = l_Lean_Syntax_node3(v___x_137_, v___x_138_, v___x_133_, v___x_140_, v___x_134_);
v___x_142_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_142_, 0, v___x_141_);
lean_ctor_set(v___x_142_, 1, v_a_116_);
return v___x_142_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Data_List_Duplicate_0__List___aux__Mathlib__Data__List__Duplicate______unexpand__List__Duplicate__1___boxed(lean_object* v_x_143_, lean_object* v_a_144_, lean_object* v_a_145_){
_start:
{
lean_object* v_res_146_; 
v_res_146_ = lp_mathlib___private_Mathlib_Data_List_Duplicate_0__List___aux__Mathlib__Data__List__Duplicate______unexpand__List__Duplicate__1(v_x_143_, v_a_144_, v_a_145_);
lean_dec(v_a_144_);
return v_res_146_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_List_decidableDuplicate___redArg(lean_object* v_inst_147_, lean_object* v_x_148_, lean_object* v_x_149_){
_start:
{
if (lean_obj_tag(v_x_149_) == 0)
{
uint8_t v___x_150_; 
lean_dec(v_x_148_);
lean_dec_ref(v_inst_147_);
v___x_150_ = 0;
return v___x_150_;
}
else
{
lean_object* v_head_151_; lean_object* v_tail_152_; lean_object* v___f_153_; lean_object* v___x_154_; uint8_t v___x_155_; 
v_head_151_ = lean_ctor_get(v_x_149_, 0);
lean_inc(v_head_151_);
v_tail_152_ = lean_ctor_get(v_x_149_, 1);
lean_inc_n(v_tail_152_, 2);
lean_dec_ref_known(v_x_149_, 2);
lean_inc_ref_n(v_inst_147_, 2);
v___f_153_ = lean_alloc_closure((void*)(l_instBEqOfDecidableEq___redArg___lam__0___boxed), 3, 1);
lean_closure_set(v___f_153_, 0, v_inst_147_);
lean_inc_n(v_x_148_, 2);
v___x_154_ = lean_apply_2(v_inst_147_, v_head_151_, v_x_148_);
v___x_155_ = lp_mathlib_List_decidableDuplicate___redArg(v_inst_147_, v_x_148_, v_tail_152_);
if (v___x_155_ == 0)
{
uint8_t v___x_156_; 
v___x_156_ = lean_unbox(v___x_154_);
if (v___x_156_ == 0)
{
uint8_t v___x_157_; 
lean_dec_ref(v___f_153_);
lean_dec(v_tail_152_);
lean_dec(v_x_148_);
v___x_157_ = lean_unbox(v___x_154_);
return v___x_157_;
}
else
{
uint8_t v___x_158_; 
v___x_158_ = l_List_elem___redArg(v___f_153_, v_x_148_, v_tail_152_);
return v___x_158_;
}
}
else
{
lean_dec_ref(v___f_153_);
lean_dec(v_tail_152_);
lean_dec(v_x_148_);
return v___x_155_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_decidableDuplicate___redArg___boxed(lean_object* v_inst_159_, lean_object* v_x_160_, lean_object* v_x_161_){
_start:
{
uint8_t v_res_162_; lean_object* v_r_163_; 
v_res_162_ = lp_mathlib_List_decidableDuplicate___redArg(v_inst_159_, v_x_160_, v_x_161_);
v_r_163_ = lean_box(v_res_162_);
return v_r_163_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_List_decidableDuplicate(lean_object* v_00_u03b1_164_, lean_object* v_inst_165_, lean_object* v_x_166_, lean_object* v_x_167_){
_start:
{
uint8_t v___x_168_; 
v___x_168_ = lp_mathlib_List_decidableDuplicate___redArg(v_inst_165_, v_x_166_, v_x_167_);
return v___x_168_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_decidableDuplicate___boxed(lean_object* v_00_u03b1_169_, lean_object* v_inst_170_, lean_object* v_x_171_, lean_object* v_x_172_){
_start:
{
uint8_t v_res_173_; lean_object* v_r_174_; 
v_res_173_ = lp_mathlib_List_decidableDuplicate(v_00_u03b1_169_, v_inst_170_, v_x_171_, v_x_172_);
v_r_174_ = lean_box(v_res_173_);
return v_r_174_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Data_List_Nodup(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_Data_List_Duplicate(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Data_List_Nodup(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_Data_List_Duplicate(uint8_t builtin) {
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
lean_object* initialize_mathlib_Mathlib_Data_List_Nodup(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Data_List_Duplicate(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_List_Nodup(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Data_List_Duplicate(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_Data_List_Duplicate(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_Data_List_Duplicate(builtin);
}
#ifdef __cplusplus
}
#endif
