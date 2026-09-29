// Lean compiler output
// Module: Mathlib.Data.List.Perm.Basic
// Imports: public import Init public meta import Init public import Batteries.Data.List.Perm public import Mathlib.Logic.Relation public import Mathlib.Data.List.Forall2 public import Mathlib.Data.List.InsertIdx public import Mathlib.Logic.OpClass
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
lean_object* l_Lean_Name_str___override(lean_object*, lean_object*);
lean_object* l_Lean_Name_num___override(lean_object*, lean_object*);
lean_object* l_Lean_Name_mkStr2(lean_object*, lean_object*);
lean_object* l_Lean_Name_mkStr4(lean_object*, lean_object*, lean_object*, lean_object*);
uint8_t l_Lean_Syntax_isOfKind(lean_object*, lean_object*);
lean_object* l_Lean_Syntax_getArg(lean_object*, lean_object*);
uint8_t l_Lean_Syntax_matchesNull(lean_object*, lean_object*);
lean_object* l_Lean_replaceRef(lean_object*, lean_object*);
lean_object* l_Lean_SourceInfo_fromRef(lean_object*, uint8_t);
lean_object* l_Lean_Syntax_node3(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
uint8_t l_Lean_Syntax_matchesIdent(lean_object*, lean_object*);
lean_object* l_String_toRawSubstring_x27(lean_object*);
lean_object* l_Lean_addMacroScope(lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Syntax_node2(lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib___private_Mathlib_Data_List_Perm_Basic_0__List_term___u2218r___00__closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "_private"};
static const lean_object* lp_mathlib___private_Mathlib_Data_List_Perm_Basic_0__List_term___u2218r___00__closed__0 = (const lean_object*)&lp_mathlib___private_Mathlib_Data_List_Perm_Basic_0__List_term___u2218r___00__closed__0_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Data_List_Perm_Basic_0__List_term___u2218r___00__closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Data_List_Perm_Basic_0__List_term___u2218r___00__closed__0_value),LEAN_SCALAR_PTR_LITERAL(103, 214, 75, 80, 34, 198, 193, 153)}};
static const lean_object* lp_mathlib___private_Mathlib_Data_List_Perm_Basic_0__List_term___u2218r___00__closed__1 = (const lean_object*)&lp_mathlib___private_Mathlib_Data_List_Perm_Basic_0__List_term___u2218r___00__closed__1_value;
static const lean_string_object lp_mathlib___private_Mathlib_Data_List_Perm_Basic_0__List_term___u2218r___00__closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "Mathlib"};
static const lean_object* lp_mathlib___private_Mathlib_Data_List_Perm_Basic_0__List_term___u2218r___00__closed__2 = (const lean_object*)&lp_mathlib___private_Mathlib_Data_List_Perm_Basic_0__List_term___u2218r___00__closed__2_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Data_List_Perm_Basic_0__List_term___u2218r___00__closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Data_List_Perm_Basic_0__List_term___u2218r___00__closed__1_value),((lean_object*)&lp_mathlib___private_Mathlib_Data_List_Perm_Basic_0__List_term___u2218r___00__closed__2_value),LEAN_SCALAR_PTR_LITERAL(234, 232, 174, 134, 127, 136, 69, 92)}};
static const lean_object* lp_mathlib___private_Mathlib_Data_List_Perm_Basic_0__List_term___u2218r___00__closed__3 = (const lean_object*)&lp_mathlib___private_Mathlib_Data_List_Perm_Basic_0__List_term___u2218r___00__closed__3_value;
static const lean_string_object lp_mathlib___private_Mathlib_Data_List_Perm_Basic_0__List_term___u2218r___00__closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "Data"};
static const lean_object* lp_mathlib___private_Mathlib_Data_List_Perm_Basic_0__List_term___u2218r___00__closed__4 = (const lean_object*)&lp_mathlib___private_Mathlib_Data_List_Perm_Basic_0__List_term___u2218r___00__closed__4_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Data_List_Perm_Basic_0__List_term___u2218r___00__closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Data_List_Perm_Basic_0__List_term___u2218r___00__closed__3_value),((lean_object*)&lp_mathlib___private_Mathlib_Data_List_Perm_Basic_0__List_term___u2218r___00__closed__4_value),LEAN_SCALAR_PTR_LITERAL(149, 48, 214, 5, 44, 128, 44, 12)}};
static const lean_object* lp_mathlib___private_Mathlib_Data_List_Perm_Basic_0__List_term___u2218r___00__closed__5 = (const lean_object*)&lp_mathlib___private_Mathlib_Data_List_Perm_Basic_0__List_term___u2218r___00__closed__5_value;
static const lean_string_object lp_mathlib___private_Mathlib_Data_List_Perm_Basic_0__List_term___u2218r___00__closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "List"};
static const lean_object* lp_mathlib___private_Mathlib_Data_List_Perm_Basic_0__List_term___u2218r___00__closed__6 = (const lean_object*)&lp_mathlib___private_Mathlib_Data_List_Perm_Basic_0__List_term___u2218r___00__closed__6_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Data_List_Perm_Basic_0__List_term___u2218r___00__closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Data_List_Perm_Basic_0__List_term___u2218r___00__closed__5_value),((lean_object*)&lp_mathlib___private_Mathlib_Data_List_Perm_Basic_0__List_term___u2218r___00__closed__6_value),LEAN_SCALAR_PTR_LITERAL(123, 214, 204, 50, 8, 169, 217, 159)}};
static const lean_object* lp_mathlib___private_Mathlib_Data_List_Perm_Basic_0__List_term___u2218r___00__closed__7 = (const lean_object*)&lp_mathlib___private_Mathlib_Data_List_Perm_Basic_0__List_term___u2218r___00__closed__7_value;
static const lean_string_object lp_mathlib___private_Mathlib_Data_List_Perm_Basic_0__List_term___u2218r___00__closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "Perm"};
static const lean_object* lp_mathlib___private_Mathlib_Data_List_Perm_Basic_0__List_term___u2218r___00__closed__8 = (const lean_object*)&lp_mathlib___private_Mathlib_Data_List_Perm_Basic_0__List_term___u2218r___00__closed__8_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Data_List_Perm_Basic_0__List_term___u2218r___00__closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Data_List_Perm_Basic_0__List_term___u2218r___00__closed__7_value),((lean_object*)&lp_mathlib___private_Mathlib_Data_List_Perm_Basic_0__List_term___u2218r___00__closed__8_value),LEAN_SCALAR_PTR_LITERAL(29, 210, 49, 125, 219, 36, 54, 227)}};
static const lean_object* lp_mathlib___private_Mathlib_Data_List_Perm_Basic_0__List_term___u2218r___00__closed__9 = (const lean_object*)&lp_mathlib___private_Mathlib_Data_List_Perm_Basic_0__List_term___u2218r___00__closed__9_value;
static const lean_string_object lp_mathlib___private_Mathlib_Data_List_Perm_Basic_0__List_term___u2218r___00__closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "Basic"};
static const lean_object* lp_mathlib___private_Mathlib_Data_List_Perm_Basic_0__List_term___u2218r___00__closed__10 = (const lean_object*)&lp_mathlib___private_Mathlib_Data_List_Perm_Basic_0__List_term___u2218r___00__closed__10_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Data_List_Perm_Basic_0__List_term___u2218r___00__closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Data_List_Perm_Basic_0__List_term___u2218r___00__closed__9_value),((lean_object*)&lp_mathlib___private_Mathlib_Data_List_Perm_Basic_0__List_term___u2218r___00__closed__10_value),LEAN_SCALAR_PTR_LITERAL(122, 167, 27, 46, 48, 32, 131, 93)}};
static const lean_object* lp_mathlib___private_Mathlib_Data_List_Perm_Basic_0__List_term___u2218r___00__closed__11 = (const lean_object*)&lp_mathlib___private_Mathlib_Data_List_Perm_Basic_0__List_term___u2218r___00__closed__11_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Data_List_Perm_Basic_0__List_term___u2218r___00__closed__12_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Data_List_Perm_Basic_0__List_term___u2218r___00__closed__11_value),((lean_object*)(((size_t)(0) << 1) | 1)),LEAN_SCALAR_PTR_LITERAL(27, 120, 182, 22, 207, 8, 141, 39)}};
static const lean_object* lp_mathlib___private_Mathlib_Data_List_Perm_Basic_0__List_term___u2218r___00__closed__12 = (const lean_object*)&lp_mathlib___private_Mathlib_Data_List_Perm_Basic_0__List_term___u2218r___00__closed__12_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Data_List_Perm_Basic_0__List_term___u2218r___00__closed__13_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Data_List_Perm_Basic_0__List_term___u2218r___00__closed__12_value),((lean_object*)&lp_mathlib___private_Mathlib_Data_List_Perm_Basic_0__List_term___u2218r___00__closed__6_value),LEAN_SCALAR_PTR_LITERAL(213, 93, 75, 82, 195, 153, 122, 56)}};
static const lean_object* lp_mathlib___private_Mathlib_Data_List_Perm_Basic_0__List_term___u2218r___00__closed__13 = (const lean_object*)&lp_mathlib___private_Mathlib_Data_List_Perm_Basic_0__List_term___u2218r___00__closed__13_value;
static const lean_string_object lp_mathlib___private_Mathlib_Data_List_Perm_Basic_0__List_term___u2218r___00__closed__14_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 11, .m_capacity = 11, .m_length = 8, .m_data = "term_∘r_"};
static const lean_object* lp_mathlib___private_Mathlib_Data_List_Perm_Basic_0__List_term___u2218r___00__closed__14 = (const lean_object*)&lp_mathlib___private_Mathlib_Data_List_Perm_Basic_0__List_term___u2218r___00__closed__14_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Data_List_Perm_Basic_0__List_term___u2218r___00__closed__15_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Data_List_Perm_Basic_0__List_term___u2218r___00__closed__13_value),((lean_object*)&lp_mathlib___private_Mathlib_Data_List_Perm_Basic_0__List_term___u2218r___00__closed__14_value),LEAN_SCALAR_PTR_LITERAL(174, 172, 167, 219, 73, 192, 160, 15)}};
static const lean_object* lp_mathlib___private_Mathlib_Data_List_Perm_Basic_0__List_term___u2218r___00__closed__15 = (const lean_object*)&lp_mathlib___private_Mathlib_Data_List_Perm_Basic_0__List_term___u2218r___00__closed__15_value;
static const lean_string_object lp_mathlib___private_Mathlib_Data_List_Perm_Basic_0__List_term___u2218r___00__closed__16_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "andthen"};
static const lean_object* lp_mathlib___private_Mathlib_Data_List_Perm_Basic_0__List_term___u2218r___00__closed__16 = (const lean_object*)&lp_mathlib___private_Mathlib_Data_List_Perm_Basic_0__List_term___u2218r___00__closed__16_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Data_List_Perm_Basic_0__List_term___u2218r___00__closed__17_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Data_List_Perm_Basic_0__List_term___u2218r___00__closed__16_value),LEAN_SCALAR_PTR_LITERAL(40, 255, 78, 30, 143, 119, 117, 174)}};
static const lean_object* lp_mathlib___private_Mathlib_Data_List_Perm_Basic_0__List_term___u2218r___00__closed__17 = (const lean_object*)&lp_mathlib___private_Mathlib_Data_List_Perm_Basic_0__List_term___u2218r___00__closed__17_value;
static const lean_string_object lp_mathlib___private_Mathlib_Data_List_Perm_Basic_0__List_term___u2218r___00__closed__18_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 4, .m_data = " ∘r "};
static const lean_object* lp_mathlib___private_Mathlib_Data_List_Perm_Basic_0__List_term___u2218r___00__closed__18 = (const lean_object*)&lp_mathlib___private_Mathlib_Data_List_Perm_Basic_0__List_term___u2218r___00__closed__18_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Data_List_Perm_Basic_0__List_term___u2218r___00__closed__19_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 5}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Data_List_Perm_Basic_0__List_term___u2218r___00__closed__18_value)}};
static const lean_object* lp_mathlib___private_Mathlib_Data_List_Perm_Basic_0__List_term___u2218r___00__closed__19 = (const lean_object*)&lp_mathlib___private_Mathlib_Data_List_Perm_Basic_0__List_term___u2218r___00__closed__19_value;
static const lean_string_object lp_mathlib___private_Mathlib_Data_List_Perm_Basic_0__List_term___u2218r___00__closed__20_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "term"};
static const lean_object* lp_mathlib___private_Mathlib_Data_List_Perm_Basic_0__List_term___u2218r___00__closed__20 = (const lean_object*)&lp_mathlib___private_Mathlib_Data_List_Perm_Basic_0__List_term___u2218r___00__closed__20_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Data_List_Perm_Basic_0__List_term___u2218r___00__closed__21_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Data_List_Perm_Basic_0__List_term___u2218r___00__closed__20_value),LEAN_SCALAR_PTR_LITERAL(187, 230, 181, 162, 253, 146, 122, 119)}};
static const lean_object* lp_mathlib___private_Mathlib_Data_List_Perm_Basic_0__List_term___u2218r___00__closed__21 = (const lean_object*)&lp_mathlib___private_Mathlib_Data_List_Perm_Basic_0__List_term___u2218r___00__closed__21_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Data_List_Perm_Basic_0__List_term___u2218r___00__closed__22_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 7}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Data_List_Perm_Basic_0__List_term___u2218r___00__closed__21_value),((lean_object*)(((size_t)(80) << 1) | 1))}};
static const lean_object* lp_mathlib___private_Mathlib_Data_List_Perm_Basic_0__List_term___u2218r___00__closed__22 = (const lean_object*)&lp_mathlib___private_Mathlib_Data_List_Perm_Basic_0__List_term___u2218r___00__closed__22_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Data_List_Perm_Basic_0__List_term___u2218r___00__closed__23_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Data_List_Perm_Basic_0__List_term___u2218r___00__closed__17_value),((lean_object*)&lp_mathlib___private_Mathlib_Data_List_Perm_Basic_0__List_term___u2218r___00__closed__19_value),((lean_object*)&lp_mathlib___private_Mathlib_Data_List_Perm_Basic_0__List_term___u2218r___00__closed__22_value)}};
static const lean_object* lp_mathlib___private_Mathlib_Data_List_Perm_Basic_0__List_term___u2218r___00__closed__23 = (const lean_object*)&lp_mathlib___private_Mathlib_Data_List_Perm_Basic_0__List_term___u2218r___00__closed__23_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Data_List_Perm_Basic_0__List_term___u2218r___00__closed__24_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*4 + 0, .m_other = 4, .m_tag = 4}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Data_List_Perm_Basic_0__List_term___u2218r___00__closed__15_value),((lean_object*)(((size_t)(80) << 1) | 1)),((lean_object*)(((size_t)(81) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Data_List_Perm_Basic_0__List_term___u2218r___00__closed__23_value)}};
static const lean_object* lp_mathlib___private_Mathlib_Data_List_Perm_Basic_0__List_term___u2218r___00__closed__24 = (const lean_object*)&lp_mathlib___private_Mathlib_Data_List_Perm_Basic_0__List_term___u2218r___00__closed__24_value;
LEAN_EXPORT const lean_object* lp_mathlib___private_Mathlib_Data_List_Perm_Basic_0__List_term___u2218r__ = (const lean_object*)&lp_mathlib___private_Mathlib_Data_List_Perm_Basic_0__List_term___u2218r___00__closed__24_value;
static const lean_string_object lp_mathlib___private_Mathlib_Data_List_Perm_Basic_0__List___aux__Mathlib__Data__List__Perm__Basic______macroRules____private__Mathlib__Data__List__Perm__Basic__0__List__term___u2218r____1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "Lean"};
static const lean_object* lp_mathlib___private_Mathlib_Data_List_Perm_Basic_0__List___aux__Mathlib__Data__List__Perm__Basic______macroRules____private__Mathlib__Data__List__Perm__Basic__0__List__term___u2218r____1___closed__0 = (const lean_object*)&lp_mathlib___private_Mathlib_Data_List_Perm_Basic_0__List___aux__Mathlib__Data__List__Perm__Basic______macroRules____private__Mathlib__Data__List__Perm__Basic__0__List__term___u2218r____1___closed__0_value;
static const lean_string_object lp_mathlib___private_Mathlib_Data_List_Perm_Basic_0__List___aux__Mathlib__Data__List__Perm__Basic______macroRules____private__Mathlib__Data__List__Perm__Basic__0__List__term___u2218r____1___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "Parser"};
static const lean_object* lp_mathlib___private_Mathlib_Data_List_Perm_Basic_0__List___aux__Mathlib__Data__List__Perm__Basic______macroRules____private__Mathlib__Data__List__Perm__Basic__0__List__term___u2218r____1___closed__1 = (const lean_object*)&lp_mathlib___private_Mathlib_Data_List_Perm_Basic_0__List___aux__Mathlib__Data__List__Perm__Basic______macroRules____private__Mathlib__Data__List__Perm__Basic__0__List__term___u2218r____1___closed__1_value;
static const lean_string_object lp_mathlib___private_Mathlib_Data_List_Perm_Basic_0__List___aux__Mathlib__Data__List__Perm__Basic______macroRules____private__Mathlib__Data__List__Perm__Basic__0__List__term___u2218r____1___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "Term"};
static const lean_object* lp_mathlib___private_Mathlib_Data_List_Perm_Basic_0__List___aux__Mathlib__Data__List__Perm__Basic______macroRules____private__Mathlib__Data__List__Perm__Basic__0__List__term___u2218r____1___closed__2 = (const lean_object*)&lp_mathlib___private_Mathlib_Data_List_Perm_Basic_0__List___aux__Mathlib__Data__List__Perm__Basic______macroRules____private__Mathlib__Data__List__Perm__Basic__0__List__term___u2218r____1___closed__2_value;
static const lean_string_object lp_mathlib___private_Mathlib_Data_List_Perm_Basic_0__List___aux__Mathlib__Data__List__Perm__Basic______macroRules____private__Mathlib__Data__List__Perm__Basic__0__List__term___u2218r____1___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = "app"};
static const lean_object* lp_mathlib___private_Mathlib_Data_List_Perm_Basic_0__List___aux__Mathlib__Data__List__Perm__Basic______macroRules____private__Mathlib__Data__List__Perm__Basic__0__List__term___u2218r____1___closed__3 = (const lean_object*)&lp_mathlib___private_Mathlib_Data_List_Perm_Basic_0__List___aux__Mathlib__Data__List__Perm__Basic______macroRules____private__Mathlib__Data__List__Perm__Basic__0__List__term___u2218r____1___closed__3_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Data_List_Perm_Basic_0__List___aux__Mathlib__Data__List__Perm__Basic______macroRules____private__Mathlib__Data__List__Perm__Basic__0__List__term___u2218r____1___closed__4_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Data_List_Perm_Basic_0__List___aux__Mathlib__Data__List__Perm__Basic______macroRules____private__Mathlib__Data__List__Perm__Basic__0__List__term___u2218r____1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Data_List_Perm_Basic_0__List___aux__Mathlib__Data__List__Perm__Basic______macroRules____private__Mathlib__Data__List__Perm__Basic__0__List__term___u2218r____1___closed__4_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Data_List_Perm_Basic_0__List___aux__Mathlib__Data__List__Perm__Basic______macroRules____private__Mathlib__Data__List__Perm__Basic__0__List__term___u2218r____1___closed__4_value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Data_List_Perm_Basic_0__List___aux__Mathlib__Data__List__Perm__Basic______macroRules____private__Mathlib__Data__List__Perm__Basic__0__List__term___u2218r____1___closed__1_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Data_List_Perm_Basic_0__List___aux__Mathlib__Data__List__Perm__Basic______macroRules____private__Mathlib__Data__List__Perm__Basic__0__List__term___u2218r____1___closed__4_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Data_List_Perm_Basic_0__List___aux__Mathlib__Data__List__Perm__Basic______macroRules____private__Mathlib__Data__List__Perm__Basic__0__List__term___u2218r____1___closed__4_value_aux_1),((lean_object*)&lp_mathlib___private_Mathlib_Data_List_Perm_Basic_0__List___aux__Mathlib__Data__List__Perm__Basic______macroRules____private__Mathlib__Data__List__Perm__Basic__0__List__term___u2218r____1___closed__2_value),LEAN_SCALAR_PTR_LITERAL(75, 170, 162, 138, 136, 204, 251, 229)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Data_List_Perm_Basic_0__List___aux__Mathlib__Data__List__Perm__Basic______macroRules____private__Mathlib__Data__List__Perm__Basic__0__List__term___u2218r____1___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Data_List_Perm_Basic_0__List___aux__Mathlib__Data__List__Perm__Basic______macroRules____private__Mathlib__Data__List__Perm__Basic__0__List__term___u2218r____1___closed__4_value_aux_2),((lean_object*)&lp_mathlib___private_Mathlib_Data_List_Perm_Basic_0__List___aux__Mathlib__Data__List__Perm__Basic______macroRules____private__Mathlib__Data__List__Perm__Basic__0__List__term___u2218r____1___closed__3_value),LEAN_SCALAR_PTR_LITERAL(69, 118, 10, 41, 220, 156, 243, 179)}};
static const lean_object* lp_mathlib___private_Mathlib_Data_List_Perm_Basic_0__List___aux__Mathlib__Data__List__Perm__Basic______macroRules____private__Mathlib__Data__List__Perm__Basic__0__List__term___u2218r____1___closed__4 = (const lean_object*)&lp_mathlib___private_Mathlib_Data_List_Perm_Basic_0__List___aux__Mathlib__Data__List__Perm__Basic______macroRules____private__Mathlib__Data__List__Perm__Basic__0__List__term___u2218r____1___closed__4_value;
static const lean_string_object lp_mathlib___private_Mathlib_Data_List_Perm_Basic_0__List___aux__Mathlib__Data__List__Perm__Basic______macroRules____private__Mathlib__Data__List__Perm__Basic__0__List__term___u2218r____1___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 14, .m_capacity = 14, .m_length = 13, .m_data = "Relation.Comp"};
static const lean_object* lp_mathlib___private_Mathlib_Data_List_Perm_Basic_0__List___aux__Mathlib__Data__List__Perm__Basic______macroRules____private__Mathlib__Data__List__Perm__Basic__0__List__term___u2218r____1___closed__5 = (const lean_object*)&lp_mathlib___private_Mathlib_Data_List_Perm_Basic_0__List___aux__Mathlib__Data__List__Perm__Basic______macroRules____private__Mathlib__Data__List__Perm__Basic__0__List__term___u2218r____1___closed__5_value;
static lean_once_cell_t lp_mathlib___private_Mathlib_Data_List_Perm_Basic_0__List___aux__Mathlib__Data__List__Perm__Basic______macroRules____private__Mathlib__Data__List__Perm__Basic__0__List__term___u2218r____1___closed__6_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Data_List_Perm_Basic_0__List___aux__Mathlib__Data__List__Perm__Basic______macroRules____private__Mathlib__Data__List__Perm__Basic__0__List__term___u2218r____1___closed__6;
static const lean_string_object lp_mathlib___private_Mathlib_Data_List_Perm_Basic_0__List___aux__Mathlib__Data__List__Perm__Basic______macroRules____private__Mathlib__Data__List__Perm__Basic__0__List__term___u2218r____1___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "Relation"};
static const lean_object* lp_mathlib___private_Mathlib_Data_List_Perm_Basic_0__List___aux__Mathlib__Data__List__Perm__Basic______macroRules____private__Mathlib__Data__List__Perm__Basic__0__List__term___u2218r____1___closed__7 = (const lean_object*)&lp_mathlib___private_Mathlib_Data_List_Perm_Basic_0__List___aux__Mathlib__Data__List__Perm__Basic______macroRules____private__Mathlib__Data__List__Perm__Basic__0__List__term___u2218r____1___closed__7_value;
static const lean_string_object lp_mathlib___private_Mathlib_Data_List_Perm_Basic_0__List___aux__Mathlib__Data__List__Perm__Basic______macroRules____private__Mathlib__Data__List__Perm__Basic__0__List__term___u2218r____1___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "Comp"};
static const lean_object* lp_mathlib___private_Mathlib_Data_List_Perm_Basic_0__List___aux__Mathlib__Data__List__Perm__Basic______macroRules____private__Mathlib__Data__List__Perm__Basic__0__List__term___u2218r____1___closed__8 = (const lean_object*)&lp_mathlib___private_Mathlib_Data_List_Perm_Basic_0__List___aux__Mathlib__Data__List__Perm__Basic______macroRules____private__Mathlib__Data__List__Perm__Basic__0__List__term___u2218r____1___closed__8_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Data_List_Perm_Basic_0__List___aux__Mathlib__Data__List__Perm__Basic______macroRules____private__Mathlib__Data__List__Perm__Basic__0__List__term___u2218r____1___closed__9_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Data_List_Perm_Basic_0__List___aux__Mathlib__Data__List__Perm__Basic______macroRules____private__Mathlib__Data__List__Perm__Basic__0__List__term___u2218r____1___closed__7_value),LEAN_SCALAR_PTR_LITERAL(75, 52, 124, 254, 211, 241, 202, 221)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Data_List_Perm_Basic_0__List___aux__Mathlib__Data__List__Perm__Basic______macroRules____private__Mathlib__Data__List__Perm__Basic__0__List__term___u2218r____1___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Data_List_Perm_Basic_0__List___aux__Mathlib__Data__List__Perm__Basic______macroRules____private__Mathlib__Data__List__Perm__Basic__0__List__term___u2218r____1___closed__9_value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Data_List_Perm_Basic_0__List___aux__Mathlib__Data__List__Perm__Basic______macroRules____private__Mathlib__Data__List__Perm__Basic__0__List__term___u2218r____1___closed__8_value),LEAN_SCALAR_PTR_LITERAL(208, 226, 194, 184, 8, 138, 89, 143)}};
static const lean_object* lp_mathlib___private_Mathlib_Data_List_Perm_Basic_0__List___aux__Mathlib__Data__List__Perm__Basic______macroRules____private__Mathlib__Data__List__Perm__Basic__0__List__term___u2218r____1___closed__9 = (const lean_object*)&lp_mathlib___private_Mathlib_Data_List_Perm_Basic_0__List___aux__Mathlib__Data__List__Perm__Basic______macroRules____private__Mathlib__Data__List__Perm__Basic__0__List__term___u2218r____1___closed__9_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Data_List_Perm_Basic_0__List___aux__Mathlib__Data__List__Perm__Basic______macroRules____private__Mathlib__Data__List__Perm__Basic__0__List__term___u2218r____1___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Data_List_Perm_Basic_0__List___aux__Mathlib__Data__List__Perm__Basic______macroRules____private__Mathlib__Data__List__Perm__Basic__0__List__term___u2218r____1___closed__9_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib___private_Mathlib_Data_List_Perm_Basic_0__List___aux__Mathlib__Data__List__Perm__Basic______macroRules____private__Mathlib__Data__List__Perm__Basic__0__List__term___u2218r____1___closed__10 = (const lean_object*)&lp_mathlib___private_Mathlib_Data_List_Perm_Basic_0__List___aux__Mathlib__Data__List__Perm__Basic______macroRules____private__Mathlib__Data__List__Perm__Basic__0__List__term___u2218r____1___closed__10_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Data_List_Perm_Basic_0__List___aux__Mathlib__Data__List__Perm__Basic______macroRules____private__Mathlib__Data__List__Perm__Basic__0__List__term___u2218r____1___closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Data_List_Perm_Basic_0__List___aux__Mathlib__Data__List__Perm__Basic______macroRules____private__Mathlib__Data__List__Perm__Basic__0__List__term___u2218r____1___closed__10_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib___private_Mathlib_Data_List_Perm_Basic_0__List___aux__Mathlib__Data__List__Perm__Basic______macroRules____private__Mathlib__Data__List__Perm__Basic__0__List__term___u2218r____1___closed__11 = (const lean_object*)&lp_mathlib___private_Mathlib_Data_List_Perm_Basic_0__List___aux__Mathlib__Data__List__Perm__Basic______macroRules____private__Mathlib__Data__List__Perm__Basic__0__List__term___u2218r____1___closed__11_value;
static const lean_string_object lp_mathlib___private_Mathlib_Data_List_Perm_Basic_0__List___aux__Mathlib__Data__List__Perm__Basic______macroRules____private__Mathlib__Data__List__Perm__Basic__0__List__term___u2218r____1___closed__12_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "null"};
static const lean_object* lp_mathlib___private_Mathlib_Data_List_Perm_Basic_0__List___aux__Mathlib__Data__List__Perm__Basic______macroRules____private__Mathlib__Data__List__Perm__Basic__0__List__term___u2218r____1___closed__12 = (const lean_object*)&lp_mathlib___private_Mathlib_Data_List_Perm_Basic_0__List___aux__Mathlib__Data__List__Perm__Basic______macroRules____private__Mathlib__Data__List__Perm__Basic__0__List__term___u2218r____1___closed__12_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Data_List_Perm_Basic_0__List___aux__Mathlib__Data__List__Perm__Basic______macroRules____private__Mathlib__Data__List__Perm__Basic__0__List__term___u2218r____1___closed__13_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Data_List_Perm_Basic_0__List___aux__Mathlib__Data__List__Perm__Basic______macroRules____private__Mathlib__Data__List__Perm__Basic__0__List__term___u2218r____1___closed__12_value),LEAN_SCALAR_PTR_LITERAL(24, 58, 49, 223, 146, 207, 197, 136)}};
static const lean_object* lp_mathlib___private_Mathlib_Data_List_Perm_Basic_0__List___aux__Mathlib__Data__List__Perm__Basic______macroRules____private__Mathlib__Data__List__Perm__Basic__0__List__term___u2218r____1___closed__13 = (const lean_object*)&lp_mathlib___private_Mathlib_Data_List_Perm_Basic_0__List___aux__Mathlib__Data__List__Perm__Basic______macroRules____private__Mathlib__Data__List__Perm__Basic__0__List__term___u2218r____1___closed__13_value;
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Data_List_Perm_Basic_0__List___aux__Mathlib__Data__List__Perm__Basic______macroRules____private__Mathlib__Data__List__Perm__Basic__0__List__term___u2218r____1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Data_List_Perm_Basic_0__List___aux__Mathlib__Data__List__Perm__Basic______macroRules____private__Mathlib__Data__List__Perm__Basic__0__List__term___u2218r____1___boxed(lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib___private_Mathlib_Data_List_Perm_Basic_0__List___aux__Mathlib__Data__List__Perm__Basic______unexpand__Relation__Comp__1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "ident"};
static const lean_object* lp_mathlib___private_Mathlib_Data_List_Perm_Basic_0__List___aux__Mathlib__Data__List__Perm__Basic______unexpand__Relation__Comp__1___closed__0 = (const lean_object*)&lp_mathlib___private_Mathlib_Data_List_Perm_Basic_0__List___aux__Mathlib__Data__List__Perm__Basic______unexpand__Relation__Comp__1___closed__0_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Data_List_Perm_Basic_0__List___aux__Mathlib__Data__List__Perm__Basic______unexpand__Relation__Comp__1___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Data_List_Perm_Basic_0__List___aux__Mathlib__Data__List__Perm__Basic______unexpand__Relation__Comp__1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(52, 159, 208, 51, 14, 60, 6, 71)}};
static const lean_object* lp_mathlib___private_Mathlib_Data_List_Perm_Basic_0__List___aux__Mathlib__Data__List__Perm__Basic______unexpand__Relation__Comp__1___closed__1 = (const lean_object*)&lp_mathlib___private_Mathlib_Data_List_Perm_Basic_0__List___aux__Mathlib__Data__List__Perm__Basic______unexpand__Relation__Comp__1___closed__1_value;
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Data_List_Perm_Basic_0__List___aux__Mathlib__Data__List__Perm__Basic______unexpand__Relation__Comp__1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Data_List_Perm_Basic_0__List___aux__Mathlib__Data__List__Perm__Basic______unexpand__Relation__Comp__1___boxed(lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib___private_Mathlib_Data_List_Perm_Basic_0__List_term___x2a___00__closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "term_*_"};
static const lean_object* lp_mathlib___private_Mathlib_Data_List_Perm_Basic_0__List_term___x2a___00__closed__0 = (const lean_object*)&lp_mathlib___private_Mathlib_Data_List_Perm_Basic_0__List_term___x2a___00__closed__0_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Data_List_Perm_Basic_0__List_term___x2a___00__closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Data_List_Perm_Basic_0__List_term___u2218r___00__closed__13_value),((lean_object*)&lp_mathlib___private_Mathlib_Data_List_Perm_Basic_0__List_term___x2a___00__closed__0_value),LEAN_SCALAR_PTR_LITERAL(96, 115, 126, 195, 163, 10, 193, 22)}};
static const lean_object* lp_mathlib___private_Mathlib_Data_List_Perm_Basic_0__List_term___x2a___00__closed__1 = (const lean_object*)&lp_mathlib___private_Mathlib_Data_List_Perm_Basic_0__List_term___x2a___00__closed__1_value;
static const lean_string_object lp_mathlib___private_Mathlib_Data_List_Perm_Basic_0__List_term___x2a___00__closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = " * "};
static const lean_object* lp_mathlib___private_Mathlib_Data_List_Perm_Basic_0__List_term___x2a___00__closed__2 = (const lean_object*)&lp_mathlib___private_Mathlib_Data_List_Perm_Basic_0__List_term___x2a___00__closed__2_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Data_List_Perm_Basic_0__List_term___x2a___00__closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 5}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Data_List_Perm_Basic_0__List_term___x2a___00__closed__2_value)}};
static const lean_object* lp_mathlib___private_Mathlib_Data_List_Perm_Basic_0__List_term___x2a___00__closed__3 = (const lean_object*)&lp_mathlib___private_Mathlib_Data_List_Perm_Basic_0__List_term___x2a___00__closed__3_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Data_List_Perm_Basic_0__List_term___x2a___00__closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 7}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Data_List_Perm_Basic_0__List_term___u2218r___00__closed__21_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib___private_Mathlib_Data_List_Perm_Basic_0__List_term___x2a___00__closed__4 = (const lean_object*)&lp_mathlib___private_Mathlib_Data_List_Perm_Basic_0__List_term___x2a___00__closed__4_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Data_List_Perm_Basic_0__List_term___x2a___00__closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Data_List_Perm_Basic_0__List_term___u2218r___00__closed__17_value),((lean_object*)&lp_mathlib___private_Mathlib_Data_List_Perm_Basic_0__List_term___x2a___00__closed__3_value),((lean_object*)&lp_mathlib___private_Mathlib_Data_List_Perm_Basic_0__List_term___x2a___00__closed__4_value)}};
static const lean_object* lp_mathlib___private_Mathlib_Data_List_Perm_Basic_0__List_term___x2a___00__closed__5 = (const lean_object*)&lp_mathlib___private_Mathlib_Data_List_Perm_Basic_0__List_term___x2a___00__closed__5_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Data_List_Perm_Basic_0__List_term___x2a___00__closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*4 + 0, .m_other = 4, .m_tag = 4}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Data_List_Perm_Basic_0__List_term___x2a___00__closed__1_value),((lean_object*)(((size_t)(1022) << 1) | 1)),((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Data_List_Perm_Basic_0__List_term___x2a___00__closed__5_value)}};
static const lean_object* lp_mathlib___private_Mathlib_Data_List_Perm_Basic_0__List_term___x2a___00__closed__6 = (const lean_object*)&lp_mathlib___private_Mathlib_Data_List_Perm_Basic_0__List_term___x2a___00__closed__6_value;
LEAN_EXPORT const lean_object* lp_mathlib___private_Mathlib_Data_List_Perm_Basic_0__List_term___x2a__ = (const lean_object*)&lp_mathlib___private_Mathlib_Data_List_Perm_Basic_0__List_term___x2a___00__closed__6_value;
static const lean_string_object lp_mathlib___private_Mathlib_Data_List_Perm_Basic_0__List___aux__Mathlib__Data__List__Perm__Basic______macroRules____private__Mathlib__Data__List__Perm__Basic__0__List__term___x2a____1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 3, .m_capacity = 3, .m_length = 2, .m_data = "op"};
static const lean_object* lp_mathlib___private_Mathlib_Data_List_Perm_Basic_0__List___aux__Mathlib__Data__List__Perm__Basic______macroRules____private__Mathlib__Data__List__Perm__Basic__0__List__term___x2a____1___closed__0 = (const lean_object*)&lp_mathlib___private_Mathlib_Data_List_Perm_Basic_0__List___aux__Mathlib__Data__List__Perm__Basic______macroRules____private__Mathlib__Data__List__Perm__Basic__0__List__term___x2a____1___closed__0_value;
static lean_once_cell_t lp_mathlib___private_Mathlib_Data_List_Perm_Basic_0__List___aux__Mathlib__Data__List__Perm__Basic______macroRules____private__Mathlib__Data__List__Perm__Basic__0__List__term___x2a____1___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Data_List_Perm_Basic_0__List___aux__Mathlib__Data__List__Perm__Basic______macroRules____private__Mathlib__Data__List__Perm__Basic__0__List__term___x2a____1___closed__1;
static const lean_ctor_object lp_mathlib___private_Mathlib_Data_List_Perm_Basic_0__List___aux__Mathlib__Data__List__Perm__Basic______macroRules____private__Mathlib__Data__List__Perm__Basic__0__List__term___x2a____1___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Data_List_Perm_Basic_0__List___aux__Mathlib__Data__List__Perm__Basic______macroRules____private__Mathlib__Data__List__Perm__Basic__0__List__term___x2a____1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(234, 254, 5, 67, 7, 92, 131, 164)}};
static const lean_object* lp_mathlib___private_Mathlib_Data_List_Perm_Basic_0__List___aux__Mathlib__Data__List__Perm__Basic______macroRules____private__Mathlib__Data__List__Perm__Basic__0__List__term___x2a____1___closed__2 = (const lean_object*)&lp_mathlib___private_Mathlib_Data_List_Perm_Basic_0__List___aux__Mathlib__Data__List__Perm__Basic______macroRules____private__Mathlib__Data__List__Perm__Basic__0__List__term___x2a____1___closed__2_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Data_List_Perm_Basic_0__List___aux__Mathlib__Data__List__Perm__Basic______macroRules____private__Mathlib__Data__List__Perm__Basic__0__List__term___x2a____1___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Data_List_Perm_Basic_0__List___aux__Mathlib__Data__List__Perm__Basic______macroRules____private__Mathlib__Data__List__Perm__Basic__0__List__term___x2a____1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(234, 254, 5, 67, 7, 92, 131, 164)}};
static const lean_object* lp_mathlib___private_Mathlib_Data_List_Perm_Basic_0__List___aux__Mathlib__Data__List__Perm__Basic______macroRules____private__Mathlib__Data__List__Perm__Basic__0__List__term___x2a____1___closed__3 = (const lean_object*)&lp_mathlib___private_Mathlib_Data_List_Perm_Basic_0__List___aux__Mathlib__Data__List__Perm__Basic______macroRules____private__Mathlib__Data__List__Perm__Basic__0__List__term___x2a____1___closed__3_value;
static const lean_string_object lp_mathlib___private_Mathlib_Data_List_Perm_Basic_0__List___aux__Mathlib__Data__List__Perm__Basic______macroRules____private__Mathlib__Data__List__Perm__Basic__0__List__term___x2a____1___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 3, .m_capacity = 3, .m_length = 2, .m_data = "_@"};
static const lean_object* lp_mathlib___private_Mathlib_Data_List_Perm_Basic_0__List___aux__Mathlib__Data__List__Perm__Basic______macroRules____private__Mathlib__Data__List__Perm__Basic__0__List__term___x2a____1___closed__4 = (const lean_object*)&lp_mathlib___private_Mathlib_Data_List_Perm_Basic_0__List___aux__Mathlib__Data__List__Perm__Basic______macroRules____private__Mathlib__Data__List__Perm__Basic__0__List__term___x2a____1___closed__4_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Data_List_Perm_Basic_0__List___aux__Mathlib__Data__List__Perm__Basic______macroRules____private__Mathlib__Data__List__Perm__Basic__0__List__term___x2a____1___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Data_List_Perm_Basic_0__List___aux__Mathlib__Data__List__Perm__Basic______macroRules____private__Mathlib__Data__List__Perm__Basic__0__List__term___x2a____1___closed__3_value),((lean_object*)&lp_mathlib___private_Mathlib_Data_List_Perm_Basic_0__List___aux__Mathlib__Data__List__Perm__Basic______macroRules____private__Mathlib__Data__List__Perm__Basic__0__List__term___x2a____1___closed__4_value),LEAN_SCALAR_PTR_LITERAL(195, 225, 22, 247, 238, 155, 51, 65)}};
static const lean_object* lp_mathlib___private_Mathlib_Data_List_Perm_Basic_0__List___aux__Mathlib__Data__List__Perm__Basic______macroRules____private__Mathlib__Data__List__Perm__Basic__0__List__term___x2a____1___closed__5 = (const lean_object*)&lp_mathlib___private_Mathlib_Data_List_Perm_Basic_0__List___aux__Mathlib__Data__List__Perm__Basic______macroRules____private__Mathlib__Data__List__Perm__Basic__0__List__term___x2a____1___closed__5_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Data_List_Perm_Basic_0__List___aux__Mathlib__Data__List__Perm__Basic______macroRules____private__Mathlib__Data__List__Perm__Basic__0__List__term___x2a____1___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Data_List_Perm_Basic_0__List___aux__Mathlib__Data__List__Perm__Basic______macroRules____private__Mathlib__Data__List__Perm__Basic__0__List__term___x2a____1___closed__5_value),((lean_object*)&lp_mathlib___private_Mathlib_Data_List_Perm_Basic_0__List_term___u2218r___00__closed__2_value),LEAN_SCALAR_PTR_LITERAL(94, 254, 163, 115, 152, 147, 149, 111)}};
static const lean_object* lp_mathlib___private_Mathlib_Data_List_Perm_Basic_0__List___aux__Mathlib__Data__List__Perm__Basic______macroRules____private__Mathlib__Data__List__Perm__Basic__0__List__term___x2a____1___closed__6 = (const lean_object*)&lp_mathlib___private_Mathlib_Data_List_Perm_Basic_0__List___aux__Mathlib__Data__List__Perm__Basic______macroRules____private__Mathlib__Data__List__Perm__Basic__0__List__term___x2a____1___closed__6_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Data_List_Perm_Basic_0__List___aux__Mathlib__Data__List__Perm__Basic______macroRules____private__Mathlib__Data__List__Perm__Basic__0__List__term___x2a____1___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Data_List_Perm_Basic_0__List___aux__Mathlib__Data__List__Perm__Basic______macroRules____private__Mathlib__Data__List__Perm__Basic__0__List__term___x2a____1___closed__6_value),((lean_object*)&lp_mathlib___private_Mathlib_Data_List_Perm_Basic_0__List_term___u2218r___00__closed__4_value),LEAN_SCALAR_PTR_LITERAL(89, 231, 153, 27, 227, 65, 114, 171)}};
static const lean_object* lp_mathlib___private_Mathlib_Data_List_Perm_Basic_0__List___aux__Mathlib__Data__List__Perm__Basic______macroRules____private__Mathlib__Data__List__Perm__Basic__0__List__term___x2a____1___closed__7 = (const lean_object*)&lp_mathlib___private_Mathlib_Data_List_Perm_Basic_0__List___aux__Mathlib__Data__List__Perm__Basic______macroRules____private__Mathlib__Data__List__Perm__Basic__0__List__term___x2a____1___closed__7_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Data_List_Perm_Basic_0__List___aux__Mathlib__Data__List__Perm__Basic______macroRules____private__Mathlib__Data__List__Perm__Basic__0__List__term___x2a____1___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Data_List_Perm_Basic_0__List___aux__Mathlib__Data__List__Perm__Basic______macroRules____private__Mathlib__Data__List__Perm__Basic__0__List__term___x2a____1___closed__7_value),((lean_object*)&lp_mathlib___private_Mathlib_Data_List_Perm_Basic_0__List_term___u2218r___00__closed__6_value),LEAN_SCALAR_PTR_LITERAL(191, 56, 101, 248, 251, 171, 78, 36)}};
static const lean_object* lp_mathlib___private_Mathlib_Data_List_Perm_Basic_0__List___aux__Mathlib__Data__List__Perm__Basic______macroRules____private__Mathlib__Data__List__Perm__Basic__0__List__term___x2a____1___closed__8 = (const lean_object*)&lp_mathlib___private_Mathlib_Data_List_Perm_Basic_0__List___aux__Mathlib__Data__List__Perm__Basic______macroRules____private__Mathlib__Data__List__Perm__Basic__0__List__term___x2a____1___closed__8_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Data_List_Perm_Basic_0__List___aux__Mathlib__Data__List__Perm__Basic______macroRules____private__Mathlib__Data__List__Perm__Basic__0__List__term___x2a____1___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Data_List_Perm_Basic_0__List___aux__Mathlib__Data__List__Perm__Basic______macroRules____private__Mathlib__Data__List__Perm__Basic__0__List__term___x2a____1___closed__8_value),((lean_object*)&lp_mathlib___private_Mathlib_Data_List_Perm_Basic_0__List_term___u2218r___00__closed__8_value),LEAN_SCALAR_PTR_LITERAL(177, 251, 237, 15, 48, 6, 86, 249)}};
static const lean_object* lp_mathlib___private_Mathlib_Data_List_Perm_Basic_0__List___aux__Mathlib__Data__List__Perm__Basic______macroRules____private__Mathlib__Data__List__Perm__Basic__0__List__term___x2a____1___closed__9 = (const lean_object*)&lp_mathlib___private_Mathlib_Data_List_Perm_Basic_0__List___aux__Mathlib__Data__List__Perm__Basic______macroRules____private__Mathlib__Data__List__Perm__Basic__0__List__term___x2a____1___closed__9_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Data_List_Perm_Basic_0__List___aux__Mathlib__Data__List__Perm__Basic______macroRules____private__Mathlib__Data__List__Perm__Basic__0__List__term___x2a____1___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Data_List_Perm_Basic_0__List___aux__Mathlib__Data__List__Perm__Basic______macroRules____private__Mathlib__Data__List__Perm__Basic__0__List__term___x2a____1___closed__9_value),((lean_object*)&lp_mathlib___private_Mathlib_Data_List_Perm_Basic_0__List_term___u2218r___00__closed__10_value),LEAN_SCALAR_PTR_LITERAL(150, 250, 44, 104, 137, 159, 245, 209)}};
static const lean_object* lp_mathlib___private_Mathlib_Data_List_Perm_Basic_0__List___aux__Mathlib__Data__List__Perm__Basic______macroRules____private__Mathlib__Data__List__Perm__Basic__0__List__term___x2a____1___closed__10 = (const lean_object*)&lp_mathlib___private_Mathlib_Data_List_Perm_Basic_0__List___aux__Mathlib__Data__List__Perm__Basic______macroRules____private__Mathlib__Data__List__Perm__Basic__0__List__term___x2a____1___closed__10_value;
static lean_once_cell_t lp_mathlib___private_Mathlib_Data_List_Perm_Basic_0__List___aux__Mathlib__Data__List__Perm__Basic______macroRules____private__Mathlib__Data__List__Perm__Basic__0__List__term___x2a____1___closed__11_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Data_List_Perm_Basic_0__List___aux__Mathlib__Data__List__Perm__Basic______macroRules____private__Mathlib__Data__List__Perm__Basic__0__List__term___x2a____1___closed__11;
static const lean_string_object lp_mathlib___private_Mathlib_Data_List_Perm_Basic_0__List___aux__Mathlib__Data__List__Perm__Basic______macroRules____private__Mathlib__Data__List__Perm__Basic__0__List__term___x2a____1___closed__12_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "_hygCtx"};
static const lean_object* lp_mathlib___private_Mathlib_Data_List_Perm_Basic_0__List___aux__Mathlib__Data__List__Perm__Basic______macroRules____private__Mathlib__Data__List__Perm__Basic__0__List__term___x2a____1___closed__12 = (const lean_object*)&lp_mathlib___private_Mathlib_Data_List_Perm_Basic_0__List___aux__Mathlib__Data__List__Perm__Basic______macroRules____private__Mathlib__Data__List__Perm__Basic__0__List__term___x2a____1___closed__12_value;
static lean_once_cell_t lp_mathlib___private_Mathlib_Data_List_Perm_Basic_0__List___aux__Mathlib__Data__List__Perm__Basic______macroRules____private__Mathlib__Data__List__Perm__Basic__0__List__term___x2a____1___closed__13_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Data_List_Perm_Basic_0__List___aux__Mathlib__Data__List__Perm__Basic______macroRules____private__Mathlib__Data__List__Perm__Basic__0__List__term___x2a____1___closed__13;
static const lean_string_object lp_mathlib___private_Mathlib_Data_List_Perm_Basic_0__List___aux__Mathlib__Data__List__Perm__Basic______macroRules____private__Mathlib__Data__List__Perm__Basic__0__List__term___x2a____1___closed__14_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "_hyg"};
static const lean_object* lp_mathlib___private_Mathlib_Data_List_Perm_Basic_0__List___aux__Mathlib__Data__List__Perm__Basic______macroRules____private__Mathlib__Data__List__Perm__Basic__0__List__term___x2a____1___closed__14 = (const lean_object*)&lp_mathlib___private_Mathlib_Data_List_Perm_Basic_0__List___aux__Mathlib__Data__List__Perm__Basic______macroRules____private__Mathlib__Data__List__Perm__Basic__0__List__term___x2a____1___closed__14_value;
static lean_once_cell_t lp_mathlib___private_Mathlib_Data_List_Perm_Basic_0__List___aux__Mathlib__Data__List__Perm__Basic______macroRules____private__Mathlib__Data__List__Perm__Basic__0__List__term___x2a____1___closed__15_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Data_List_Perm_Basic_0__List___aux__Mathlib__Data__List__Perm__Basic______macroRules____private__Mathlib__Data__List__Perm__Basic__0__List__term___x2a____1___closed__15;
static lean_once_cell_t lp_mathlib___private_Mathlib_Data_List_Perm_Basic_0__List___aux__Mathlib__Data__List__Perm__Basic______macroRules____private__Mathlib__Data__List__Perm__Basic__0__List__term___x2a____1___closed__16_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Data_List_Perm_Basic_0__List___aux__Mathlib__Data__List__Perm__Basic______macroRules____private__Mathlib__Data__List__Perm__Basic__0__List__term___x2a____1___closed__16;
static lean_once_cell_t lp_mathlib___private_Mathlib_Data_List_Perm_Basic_0__List___aux__Mathlib__Data__List__Perm__Basic______macroRules____private__Mathlib__Data__List__Perm__Basic__0__List__term___x2a____1___closed__17_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Data_List_Perm_Basic_0__List___aux__Mathlib__Data__List__Perm__Basic______macroRules____private__Mathlib__Data__List__Perm__Basic__0__List__term___x2a____1___closed__17;
static lean_once_cell_t lp_mathlib___private_Mathlib_Data_List_Perm_Basic_0__List___aux__Mathlib__Data__List__Perm__Basic______macroRules____private__Mathlib__Data__List__Perm__Basic__0__List__term___x2a____1___closed__18_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Data_List_Perm_Basic_0__List___aux__Mathlib__Data__List__Perm__Basic______macroRules____private__Mathlib__Data__List__Perm__Basic__0__List__term___x2a____1___closed__18;
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Data_List_Perm_Basic_0__List___aux__Mathlib__Data__List__Perm__Basic______macroRules____private__Mathlib__Data__List__Perm__Basic__0__List__term___x2a____1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Data_List_Perm_Basic_0__List___aux__Mathlib__Data__List__Perm__Basic______macroRules____private__Mathlib__Data__List__Perm__Basic__0__List__term___x2a____1___boxed(lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib___private_Mathlib_Data_List_Perm_Basic_0__List_term___x3c_x2a_x3e___00__closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 9, .m_data = "term_<*>_"};
static const lean_object* lp_mathlib___private_Mathlib_Data_List_Perm_Basic_0__List_term___x3c_x2a_x3e___00__closed__0 = (const lean_object*)&lp_mathlib___private_Mathlib_Data_List_Perm_Basic_0__List_term___x3c_x2a_x3e___00__closed__0_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Data_List_Perm_Basic_0__List_term___x3c_x2a_x3e___00__closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Data_List_Perm_Basic_0__List_term___u2218r___00__closed__13_value),((lean_object*)&lp_mathlib___private_Mathlib_Data_List_Perm_Basic_0__List_term___x3c_x2a_x3e___00__closed__0_value),LEAN_SCALAR_PTR_LITERAL(112, 192, 201, 206, 93, 8, 209, 181)}};
static const lean_object* lp_mathlib___private_Mathlib_Data_List_Perm_Basic_0__List_term___x3c_x2a_x3e___00__closed__1 = (const lean_object*)&lp_mathlib___private_Mathlib_Data_List_Perm_Basic_0__List_term___x3c_x2a_x3e___00__closed__1_value;
static const lean_string_object lp_mathlib___private_Mathlib_Data_List_Perm_Basic_0__List_term___x3c_x2a_x3e___00__closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = " <*> "};
static const lean_object* lp_mathlib___private_Mathlib_Data_List_Perm_Basic_0__List_term___x3c_x2a_x3e___00__closed__2 = (const lean_object*)&lp_mathlib___private_Mathlib_Data_List_Perm_Basic_0__List_term___x3c_x2a_x3e___00__closed__2_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Data_List_Perm_Basic_0__List_term___x3c_x2a_x3e___00__closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 5}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Data_List_Perm_Basic_0__List_term___x3c_x2a_x3e___00__closed__2_value)}};
static const lean_object* lp_mathlib___private_Mathlib_Data_List_Perm_Basic_0__List_term___x3c_x2a_x3e___00__closed__3 = (const lean_object*)&lp_mathlib___private_Mathlib_Data_List_Perm_Basic_0__List_term___x3c_x2a_x3e___00__closed__3_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Data_List_Perm_Basic_0__List_term___x3c_x2a_x3e___00__closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Data_List_Perm_Basic_0__List_term___u2218r___00__closed__17_value),((lean_object*)&lp_mathlib___private_Mathlib_Data_List_Perm_Basic_0__List_term___x3c_x2a_x3e___00__closed__3_value),((lean_object*)&lp_mathlib___private_Mathlib_Data_List_Perm_Basic_0__List_term___x2a___00__closed__4_value)}};
static const lean_object* lp_mathlib___private_Mathlib_Data_List_Perm_Basic_0__List_term___x3c_x2a_x3e___00__closed__4 = (const lean_object*)&lp_mathlib___private_Mathlib_Data_List_Perm_Basic_0__List_term___x3c_x2a_x3e___00__closed__4_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Data_List_Perm_Basic_0__List_term___x3c_x2a_x3e___00__closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*4 + 0, .m_other = 4, .m_tag = 4}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Data_List_Perm_Basic_0__List_term___x3c_x2a_x3e___00__closed__1_value),((lean_object*)(((size_t)(1022) << 1) | 1)),((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Data_List_Perm_Basic_0__List_term___x3c_x2a_x3e___00__closed__4_value)}};
static const lean_object* lp_mathlib___private_Mathlib_Data_List_Perm_Basic_0__List_term___x3c_x2a_x3e___00__closed__5 = (const lean_object*)&lp_mathlib___private_Mathlib_Data_List_Perm_Basic_0__List_term___x3c_x2a_x3e___00__closed__5_value;
LEAN_EXPORT const lean_object* lp_mathlib___private_Mathlib_Data_List_Perm_Basic_0__List_term___x3c_x2a_x3e__ = (const lean_object*)&lp_mathlib___private_Mathlib_Data_List_Perm_Basic_0__List_term___x3c_x2a_x3e___00__closed__5_value;
static const lean_string_object lp_mathlib___private_Mathlib_Data_List_Perm_Basic_0__List___aux__Mathlib__Data__List__Perm__Basic______macroRules____private__Mathlib__Data__List__Perm__Basic__0__List__term___x3c_x2a_x3e____1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "foldl"};
static const lean_object* lp_mathlib___private_Mathlib_Data_List_Perm_Basic_0__List___aux__Mathlib__Data__List__Perm__Basic______macroRules____private__Mathlib__Data__List__Perm__Basic__0__List__term___x3c_x2a_x3e____1___closed__0 = (const lean_object*)&lp_mathlib___private_Mathlib_Data_List_Perm_Basic_0__List___aux__Mathlib__Data__List__Perm__Basic______macroRules____private__Mathlib__Data__List__Perm__Basic__0__List__term___x3c_x2a_x3e____1___closed__0_value;
static lean_once_cell_t lp_mathlib___private_Mathlib_Data_List_Perm_Basic_0__List___aux__Mathlib__Data__List__Perm__Basic______macroRules____private__Mathlib__Data__List__Perm__Basic__0__List__term___x3c_x2a_x3e____1___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Data_List_Perm_Basic_0__List___aux__Mathlib__Data__List__Perm__Basic______macroRules____private__Mathlib__Data__List__Perm__Basic__0__List__term___x3c_x2a_x3e____1___closed__1;
static const lean_ctor_object lp_mathlib___private_Mathlib_Data_List_Perm_Basic_0__List___aux__Mathlib__Data__List__Perm__Basic______macroRules____private__Mathlib__Data__List__Perm__Basic__0__List__term___x3c_x2a_x3e____1___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Data_List_Perm_Basic_0__List___aux__Mathlib__Data__List__Perm__Basic______macroRules____private__Mathlib__Data__List__Perm__Basic__0__List__term___x3c_x2a_x3e____1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(227, 231, 20, 29, 131, 212, 172, 68)}};
static const lean_object* lp_mathlib___private_Mathlib_Data_List_Perm_Basic_0__List___aux__Mathlib__Data__List__Perm__Basic______macroRules____private__Mathlib__Data__List__Perm__Basic__0__List__term___x3c_x2a_x3e____1___closed__2 = (const lean_object*)&lp_mathlib___private_Mathlib_Data_List_Perm_Basic_0__List___aux__Mathlib__Data__List__Perm__Basic______macroRules____private__Mathlib__Data__List__Perm__Basic__0__List__term___x3c_x2a_x3e____1___closed__2_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Data_List_Perm_Basic_0__List___aux__Mathlib__Data__List__Perm__Basic______macroRules____private__Mathlib__Data__List__Perm__Basic__0__List__term___x3c_x2a_x3e____1___closed__3_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Data_List_Perm_Basic_0__List_term___u2218r___00__closed__6_value),LEAN_SCALAR_PTR_LITERAL(245, 188, 225, 225, 165, 5, 251, 132)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Data_List_Perm_Basic_0__List___aux__Mathlib__Data__List__Perm__Basic______macroRules____private__Mathlib__Data__List__Perm__Basic__0__List__term___x3c_x2a_x3e____1___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Data_List_Perm_Basic_0__List___aux__Mathlib__Data__List__Perm__Basic______macroRules____private__Mathlib__Data__List__Perm__Basic__0__List__term___x3c_x2a_x3e____1___closed__3_value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Data_List_Perm_Basic_0__List___aux__Mathlib__Data__List__Perm__Basic______macroRules____private__Mathlib__Data__List__Perm__Basic__0__List__term___x3c_x2a_x3e____1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(77, 114, 232, 9, 217, 91, 70, 100)}};
static const lean_object* lp_mathlib___private_Mathlib_Data_List_Perm_Basic_0__List___aux__Mathlib__Data__List__Perm__Basic______macroRules____private__Mathlib__Data__List__Perm__Basic__0__List__term___x3c_x2a_x3e____1___closed__3 = (const lean_object*)&lp_mathlib___private_Mathlib_Data_List_Perm_Basic_0__List___aux__Mathlib__Data__List__Perm__Basic______macroRules____private__Mathlib__Data__List__Perm__Basic__0__List__term___x3c_x2a_x3e____1___closed__3_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Data_List_Perm_Basic_0__List___aux__Mathlib__Data__List__Perm__Basic______macroRules____private__Mathlib__Data__List__Perm__Basic__0__List__term___x3c_x2a_x3e____1___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Data_List_Perm_Basic_0__List___aux__Mathlib__Data__List__Perm__Basic______macroRules____private__Mathlib__Data__List__Perm__Basic__0__List__term___x3c_x2a_x3e____1___closed__3_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib___private_Mathlib_Data_List_Perm_Basic_0__List___aux__Mathlib__Data__List__Perm__Basic______macroRules____private__Mathlib__Data__List__Perm__Basic__0__List__term___x3c_x2a_x3e____1___closed__4 = (const lean_object*)&lp_mathlib___private_Mathlib_Data_List_Perm_Basic_0__List___aux__Mathlib__Data__List__Perm__Basic______macroRules____private__Mathlib__Data__List__Perm__Basic__0__List__term___x3c_x2a_x3e____1___closed__4_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Data_List_Perm_Basic_0__List___aux__Mathlib__Data__List__Perm__Basic______macroRules____private__Mathlib__Data__List__Perm__Basic__0__List__term___x3c_x2a_x3e____1___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Data_List_Perm_Basic_0__List___aux__Mathlib__Data__List__Perm__Basic______macroRules____private__Mathlib__Data__List__Perm__Basic__0__List__term___x3c_x2a_x3e____1___closed__3_value)}};
static const lean_object* lp_mathlib___private_Mathlib_Data_List_Perm_Basic_0__List___aux__Mathlib__Data__List__Perm__Basic______macroRules____private__Mathlib__Data__List__Perm__Basic__0__List__term___x3c_x2a_x3e____1___closed__5 = (const lean_object*)&lp_mathlib___private_Mathlib_Data_List_Perm_Basic_0__List___aux__Mathlib__Data__List__Perm__Basic______macroRules____private__Mathlib__Data__List__Perm__Basic__0__List__term___x3c_x2a_x3e____1___closed__5_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Data_List_Perm_Basic_0__List___aux__Mathlib__Data__List__Perm__Basic______macroRules____private__Mathlib__Data__List__Perm__Basic__0__List__term___x3c_x2a_x3e____1___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Data_List_Perm_Basic_0__List___aux__Mathlib__Data__List__Perm__Basic______macroRules____private__Mathlib__Data__List__Perm__Basic__0__List__term___x3c_x2a_x3e____1___closed__5_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib___private_Mathlib_Data_List_Perm_Basic_0__List___aux__Mathlib__Data__List__Perm__Basic______macroRules____private__Mathlib__Data__List__Perm__Basic__0__List__term___x3c_x2a_x3e____1___closed__6 = (const lean_object*)&lp_mathlib___private_Mathlib_Data_List_Perm_Basic_0__List___aux__Mathlib__Data__List__Perm__Basic______macroRules____private__Mathlib__Data__List__Perm__Basic__0__List__term___x3c_x2a_x3e____1___closed__6_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Data_List_Perm_Basic_0__List___aux__Mathlib__Data__List__Perm__Basic______macroRules____private__Mathlib__Data__List__Perm__Basic__0__List__term___x3c_x2a_x3e____1___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Data_List_Perm_Basic_0__List___aux__Mathlib__Data__List__Perm__Basic______macroRules____private__Mathlib__Data__List__Perm__Basic__0__List__term___x3c_x2a_x3e____1___closed__4_value),((lean_object*)&lp_mathlib___private_Mathlib_Data_List_Perm_Basic_0__List___aux__Mathlib__Data__List__Perm__Basic______macroRules____private__Mathlib__Data__List__Perm__Basic__0__List__term___x3c_x2a_x3e____1___closed__6_value)}};
static const lean_object* lp_mathlib___private_Mathlib_Data_List_Perm_Basic_0__List___aux__Mathlib__Data__List__Perm__Basic______macroRules____private__Mathlib__Data__List__Perm__Basic__0__List__term___x3c_x2a_x3e____1___closed__7 = (const lean_object*)&lp_mathlib___private_Mathlib_Data_List_Perm_Basic_0__List___aux__Mathlib__Data__List__Perm__Basic______macroRules____private__Mathlib__Data__List__Perm__Basic__0__List__term___x3c_x2a_x3e____1___closed__7_value;
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Data_List_Perm_Basic_0__List___aux__Mathlib__Data__List__Perm__Basic______macroRules____private__Mathlib__Data__List__Perm__Basic__0__List__term___x3c_x2a_x3e____1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Data_List_Perm_Basic_0__List___aux__Mathlib__Data__List__Perm__Basic______macroRules____private__Mathlib__Data__List__Perm__Basic__0__List__term___x3c_x2a_x3e____1___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Data_List_Perm_Basic_0__List___aux__Mathlib__Data__List__Perm__Basic______unexpand__List__foldl__1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Data_List_Perm_Basic_0__List___aux__Mathlib__Data__List__Perm__Basic______unexpand__List__foldl__1___boxed(lean_object*, lean_object*, lean_object*);
static lean_object* _init_lp_mathlib___private_Mathlib_Data_List_Perm_Basic_0__List___aux__Mathlib__Data__List__Perm__Basic______macroRules____private__Mathlib__Data__List__Perm__Basic__0__List__term___u2218r____1___closed__6(void){
_start:
{
lean_object* v___x_67_; lean_object* v___x_68_; 
v___x_67_ = ((lean_object*)(lp_mathlib___private_Mathlib_Data_List_Perm_Basic_0__List___aux__Mathlib__Data__List__Perm__Basic______macroRules____private__Mathlib__Data__List__Perm__Basic__0__List__term___u2218r____1___closed__5));
v___x_68_ = l_String_toRawSubstring_x27(v___x_67_);
return v___x_68_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Data_List_Perm_Basic_0__List___aux__Mathlib__Data__List__Perm__Basic______macroRules____private__Mathlib__Data__List__Perm__Basic__0__List__term___u2218r____1(lean_object* v_x_83_, lean_object* v_a_84_, lean_object* v_a_85_){
_start:
{
lean_object* v___x_86_; lean_object* v___x_87_; uint8_t v___x_88_; 
v___x_86_ = lean_unsigned_to_nat(0u);
v___x_87_ = ((lean_object*)(lp_mathlib___private_Mathlib_Data_List_Perm_Basic_0__List_term___u2218r___00__closed__15));
lean_inc(v_x_83_);
v___x_88_ = l_Lean_Syntax_isOfKind(v_x_83_, v___x_87_);
if (v___x_88_ == 0)
{
lean_object* v___x_89_; lean_object* v___x_90_; 
lean_dec(v_x_83_);
v___x_89_ = lean_box(1);
v___x_90_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_90_, 0, v___x_89_);
lean_ctor_set(v___x_90_, 1, v_a_85_);
return v___x_90_;
}
else
{
lean_object* v_quotContext_91_; lean_object* v_currMacroScope_92_; lean_object* v_ref_93_; lean_object* v___x_94_; lean_object* v___x_95_; lean_object* v___x_96_; uint8_t v___x_97_; lean_object* v___x_98_; lean_object* v___x_99_; lean_object* v___x_100_; lean_object* v___x_101_; lean_object* v___x_102_; lean_object* v___x_103_; lean_object* v___x_104_; lean_object* v___x_105_; lean_object* v___x_106_; lean_object* v___x_107_; lean_object* v___x_108_; 
v_quotContext_91_ = lean_ctor_get(v_a_84_, 1);
v_currMacroScope_92_ = lean_ctor_get(v_a_84_, 2);
v_ref_93_ = lean_ctor_get(v_a_84_, 5);
v___x_94_ = l_Lean_Syntax_getArg(v_x_83_, v___x_86_);
v___x_95_ = lean_unsigned_to_nat(2u);
v___x_96_ = l_Lean_Syntax_getArg(v_x_83_, v___x_95_);
lean_dec(v_x_83_);
v___x_97_ = 0;
v___x_98_ = l_Lean_SourceInfo_fromRef(v_ref_93_, v___x_97_);
v___x_99_ = ((lean_object*)(lp_mathlib___private_Mathlib_Data_List_Perm_Basic_0__List___aux__Mathlib__Data__List__Perm__Basic______macroRules____private__Mathlib__Data__List__Perm__Basic__0__List__term___u2218r____1___closed__4));
v___x_100_ = lean_obj_once(&lp_mathlib___private_Mathlib_Data_List_Perm_Basic_0__List___aux__Mathlib__Data__List__Perm__Basic______macroRules____private__Mathlib__Data__List__Perm__Basic__0__List__term___u2218r____1___closed__6, &lp_mathlib___private_Mathlib_Data_List_Perm_Basic_0__List___aux__Mathlib__Data__List__Perm__Basic______macroRules____private__Mathlib__Data__List__Perm__Basic__0__List__term___u2218r____1___closed__6_once, _init_lp_mathlib___private_Mathlib_Data_List_Perm_Basic_0__List___aux__Mathlib__Data__List__Perm__Basic______macroRules____private__Mathlib__Data__List__Perm__Basic__0__List__term___u2218r____1___closed__6);
v___x_101_ = ((lean_object*)(lp_mathlib___private_Mathlib_Data_List_Perm_Basic_0__List___aux__Mathlib__Data__List__Perm__Basic______macroRules____private__Mathlib__Data__List__Perm__Basic__0__List__term___u2218r____1___closed__9));
lean_inc(v_currMacroScope_92_);
lean_inc(v_quotContext_91_);
v___x_102_ = l_Lean_addMacroScope(v_quotContext_91_, v___x_101_, v_currMacroScope_92_);
v___x_103_ = ((lean_object*)(lp_mathlib___private_Mathlib_Data_List_Perm_Basic_0__List___aux__Mathlib__Data__List__Perm__Basic______macroRules____private__Mathlib__Data__List__Perm__Basic__0__List__term___u2218r____1___closed__11));
lean_inc_n(v___x_98_, 2);
v___x_104_ = lean_alloc_ctor(3, 4, 0);
lean_ctor_set(v___x_104_, 0, v___x_98_);
lean_ctor_set(v___x_104_, 1, v___x_100_);
lean_ctor_set(v___x_104_, 2, v___x_102_);
lean_ctor_set(v___x_104_, 3, v___x_103_);
v___x_105_ = ((lean_object*)(lp_mathlib___private_Mathlib_Data_List_Perm_Basic_0__List___aux__Mathlib__Data__List__Perm__Basic______macroRules____private__Mathlib__Data__List__Perm__Basic__0__List__term___u2218r____1___closed__13));
v___x_106_ = l_Lean_Syntax_node2(v___x_98_, v___x_105_, v___x_94_, v___x_96_);
v___x_107_ = l_Lean_Syntax_node2(v___x_98_, v___x_99_, v___x_104_, v___x_106_);
v___x_108_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_108_, 0, v___x_107_);
lean_ctor_set(v___x_108_, 1, v_a_85_);
return v___x_108_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Data_List_Perm_Basic_0__List___aux__Mathlib__Data__List__Perm__Basic______macroRules____private__Mathlib__Data__List__Perm__Basic__0__List__term___u2218r____1___boxed(lean_object* v_x_109_, lean_object* v_a_110_, lean_object* v_a_111_){
_start:
{
lean_object* v_res_112_; 
v_res_112_ = lp_mathlib___private_Mathlib_Data_List_Perm_Basic_0__List___aux__Mathlib__Data__List__Perm__Basic______macroRules____private__Mathlib__Data__List__Perm__Basic__0__List__term___u2218r____1(v_x_109_, v_a_110_, v_a_111_);
lean_dec_ref(v_a_110_);
return v_res_112_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Data_List_Perm_Basic_0__List___aux__Mathlib__Data__List__Perm__Basic______unexpand__Relation__Comp__1(lean_object* v_x_116_, lean_object* v_a_117_, lean_object* v_a_118_){
_start:
{
lean_object* v___x_119_; uint8_t v___x_120_; 
v___x_119_ = ((lean_object*)(lp_mathlib___private_Mathlib_Data_List_Perm_Basic_0__List___aux__Mathlib__Data__List__Perm__Basic______macroRules____private__Mathlib__Data__List__Perm__Basic__0__List__term___u2218r____1___closed__4));
lean_inc(v_x_116_);
v___x_120_ = l_Lean_Syntax_isOfKind(v_x_116_, v___x_119_);
if (v___x_120_ == 0)
{
lean_object* v___x_121_; lean_object* v___x_122_; 
lean_dec(v_x_116_);
v___x_121_ = lean_box(0);
v___x_122_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_122_, 0, v___x_121_);
lean_ctor_set(v___x_122_, 1, v_a_118_);
return v___x_122_;
}
else
{
lean_object* v___x_123_; lean_object* v___x_124_; lean_object* v___x_125_; uint8_t v___x_126_; 
v___x_123_ = lean_unsigned_to_nat(0u);
v___x_124_ = l_Lean_Syntax_getArg(v_x_116_, v___x_123_);
v___x_125_ = ((lean_object*)(lp_mathlib___private_Mathlib_Data_List_Perm_Basic_0__List___aux__Mathlib__Data__List__Perm__Basic______unexpand__Relation__Comp__1___closed__1));
lean_inc(v___x_124_);
v___x_126_ = l_Lean_Syntax_isOfKind(v___x_124_, v___x_125_);
if (v___x_126_ == 0)
{
lean_object* v___x_127_; lean_object* v___x_128_; 
lean_dec(v___x_124_);
lean_dec(v_x_116_);
v___x_127_ = lean_box(0);
v___x_128_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_128_, 0, v___x_127_);
lean_ctor_set(v___x_128_, 1, v_a_118_);
return v___x_128_;
}
else
{
lean_object* v___x_129_; lean_object* v___x_130_; lean_object* v___x_131_; uint8_t v___x_132_; 
v___x_129_ = lean_unsigned_to_nat(1u);
v___x_130_ = l_Lean_Syntax_getArg(v_x_116_, v___x_129_);
lean_dec(v_x_116_);
v___x_131_ = lean_unsigned_to_nat(2u);
lean_inc(v___x_130_);
v___x_132_ = l_Lean_Syntax_matchesNull(v___x_130_, v___x_131_);
if (v___x_132_ == 0)
{
lean_object* v___x_133_; lean_object* v___x_134_; 
lean_dec(v___x_130_);
lean_dec(v___x_124_);
v___x_133_ = lean_box(0);
v___x_134_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_134_, 0, v___x_133_);
lean_ctor_set(v___x_134_, 1, v_a_118_);
return v___x_134_;
}
else
{
lean_object* v___x_135_; lean_object* v___x_136_; lean_object* v_ref_137_; uint8_t v___x_138_; lean_object* v___x_139_; lean_object* v___x_140_; lean_object* v___x_141_; lean_object* v___x_142_; lean_object* v___x_143_; lean_object* v___x_144_; 
v___x_135_ = l_Lean_Syntax_getArg(v___x_130_, v___x_123_);
v___x_136_ = l_Lean_Syntax_getArg(v___x_130_, v___x_129_);
lean_dec(v___x_130_);
v_ref_137_ = l_Lean_replaceRef(v___x_124_, v_a_117_);
lean_dec(v___x_124_);
v___x_138_ = 0;
v___x_139_ = l_Lean_SourceInfo_fromRef(v_ref_137_, v___x_138_);
lean_dec(v_ref_137_);
v___x_140_ = ((lean_object*)(lp_mathlib___private_Mathlib_Data_List_Perm_Basic_0__List_term___u2218r___00__closed__15));
v___x_141_ = ((lean_object*)(lp_mathlib___private_Mathlib_Data_List_Perm_Basic_0__List_term___u2218r___00__closed__18));
lean_inc(v___x_139_);
v___x_142_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_142_, 0, v___x_139_);
lean_ctor_set(v___x_142_, 1, v___x_141_);
v___x_143_ = l_Lean_Syntax_node3(v___x_139_, v___x_140_, v___x_135_, v___x_142_, v___x_136_);
v___x_144_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_144_, 0, v___x_143_);
lean_ctor_set(v___x_144_, 1, v_a_118_);
return v___x_144_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Data_List_Perm_Basic_0__List___aux__Mathlib__Data__List__Perm__Basic______unexpand__Relation__Comp__1___boxed(lean_object* v_x_145_, lean_object* v_a_146_, lean_object* v_a_147_){
_start:
{
lean_object* v_res_148_; 
v_res_148_ = lp_mathlib___private_Mathlib_Data_List_Perm_Basic_0__List___aux__Mathlib__Data__List__Perm__Basic______unexpand__Relation__Comp__1(v_x_145_, v_a_146_, v_a_147_);
lean_dec(v_a_146_);
return v_res_148_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Data_List_Perm_Basic_0__List___aux__Mathlib__Data__List__Perm__Basic______macroRules____private__Mathlib__Data__List__Perm__Basic__0__List__term___x2a____1___closed__1(void){
_start:
{
lean_object* v___x_170_; lean_object* v___x_171_; 
v___x_170_ = ((lean_object*)(lp_mathlib___private_Mathlib_Data_List_Perm_Basic_0__List___aux__Mathlib__Data__List__Perm__Basic______macroRules____private__Mathlib__Data__List__Perm__Basic__0__List__term___x2a____1___closed__0));
v___x_171_ = l_String_toRawSubstring_x27(v___x_170_);
return v___x_171_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Data_List_Perm_Basic_0__List___aux__Mathlib__Data__List__Perm__Basic______macroRules____private__Mathlib__Data__List__Perm__Basic__0__List__term___x2a____1___closed__11(void){
_start:
{
lean_object* v___x_196_; lean_object* v___x_197_; lean_object* v___x_198_; 
v___x_196_ = lean_unsigned_to_nat(2785942168u);
v___x_197_ = ((lean_object*)(lp_mathlib___private_Mathlib_Data_List_Perm_Basic_0__List___aux__Mathlib__Data__List__Perm__Basic______macroRules____private__Mathlib__Data__List__Perm__Basic__0__List__term___x2a____1___closed__10));
v___x_198_ = l_Lean_Name_num___override(v___x_197_, v___x_196_);
return v___x_198_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Data_List_Perm_Basic_0__List___aux__Mathlib__Data__List__Perm__Basic______macroRules____private__Mathlib__Data__List__Perm__Basic__0__List__term___x2a____1___closed__13(void){
_start:
{
lean_object* v___x_200_; lean_object* v___x_201_; lean_object* v___x_202_; 
v___x_200_ = ((lean_object*)(lp_mathlib___private_Mathlib_Data_List_Perm_Basic_0__List___aux__Mathlib__Data__List__Perm__Basic______macroRules____private__Mathlib__Data__List__Perm__Basic__0__List__term___x2a____1___closed__12));
v___x_201_ = lean_obj_once(&lp_mathlib___private_Mathlib_Data_List_Perm_Basic_0__List___aux__Mathlib__Data__List__Perm__Basic______macroRules____private__Mathlib__Data__List__Perm__Basic__0__List__term___x2a____1___closed__11, &lp_mathlib___private_Mathlib_Data_List_Perm_Basic_0__List___aux__Mathlib__Data__List__Perm__Basic______macroRules____private__Mathlib__Data__List__Perm__Basic__0__List__term___x2a____1___closed__11_once, _init_lp_mathlib___private_Mathlib_Data_List_Perm_Basic_0__List___aux__Mathlib__Data__List__Perm__Basic______macroRules____private__Mathlib__Data__List__Perm__Basic__0__List__term___x2a____1___closed__11);
v___x_202_ = l_Lean_Name_str___override(v___x_201_, v___x_200_);
return v___x_202_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Data_List_Perm_Basic_0__List___aux__Mathlib__Data__List__Perm__Basic______macroRules____private__Mathlib__Data__List__Perm__Basic__0__List__term___x2a____1___closed__15(void){
_start:
{
lean_object* v___x_204_; lean_object* v___x_205_; lean_object* v___x_206_; 
v___x_204_ = ((lean_object*)(lp_mathlib___private_Mathlib_Data_List_Perm_Basic_0__List___aux__Mathlib__Data__List__Perm__Basic______macroRules____private__Mathlib__Data__List__Perm__Basic__0__List__term___x2a____1___closed__14));
v___x_205_ = lean_obj_once(&lp_mathlib___private_Mathlib_Data_List_Perm_Basic_0__List___aux__Mathlib__Data__List__Perm__Basic______macroRules____private__Mathlib__Data__List__Perm__Basic__0__List__term___x2a____1___closed__13, &lp_mathlib___private_Mathlib_Data_List_Perm_Basic_0__List___aux__Mathlib__Data__List__Perm__Basic______macroRules____private__Mathlib__Data__List__Perm__Basic__0__List__term___x2a____1___closed__13_once, _init_lp_mathlib___private_Mathlib_Data_List_Perm_Basic_0__List___aux__Mathlib__Data__List__Perm__Basic______macroRules____private__Mathlib__Data__List__Perm__Basic__0__List__term___x2a____1___closed__13);
v___x_206_ = l_Lean_Name_str___override(v___x_205_, v___x_204_);
return v___x_206_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Data_List_Perm_Basic_0__List___aux__Mathlib__Data__List__Perm__Basic______macroRules____private__Mathlib__Data__List__Perm__Basic__0__List__term___x2a____1___closed__16(void){
_start:
{
lean_object* v___x_207_; lean_object* v___x_208_; lean_object* v___x_209_; 
v___x_207_ = lean_unsigned_to_nat(16u);
v___x_208_ = lean_obj_once(&lp_mathlib___private_Mathlib_Data_List_Perm_Basic_0__List___aux__Mathlib__Data__List__Perm__Basic______macroRules____private__Mathlib__Data__List__Perm__Basic__0__List__term___x2a____1___closed__15, &lp_mathlib___private_Mathlib_Data_List_Perm_Basic_0__List___aux__Mathlib__Data__List__Perm__Basic______macroRules____private__Mathlib__Data__List__Perm__Basic__0__List__term___x2a____1___closed__15_once, _init_lp_mathlib___private_Mathlib_Data_List_Perm_Basic_0__List___aux__Mathlib__Data__List__Perm__Basic______macroRules____private__Mathlib__Data__List__Perm__Basic__0__List__term___x2a____1___closed__15);
v___x_209_ = l_Lean_Name_num___override(v___x_208_, v___x_207_);
return v___x_209_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Data_List_Perm_Basic_0__List___aux__Mathlib__Data__List__Perm__Basic______macroRules____private__Mathlib__Data__List__Perm__Basic__0__List__term___x2a____1___closed__17(void){
_start:
{
lean_object* v___x_210_; lean_object* v___x_211_; lean_object* v___x_212_; 
v___x_210_ = lean_box(0);
v___x_211_ = lean_obj_once(&lp_mathlib___private_Mathlib_Data_List_Perm_Basic_0__List___aux__Mathlib__Data__List__Perm__Basic______macroRules____private__Mathlib__Data__List__Perm__Basic__0__List__term___x2a____1___closed__16, &lp_mathlib___private_Mathlib_Data_List_Perm_Basic_0__List___aux__Mathlib__Data__List__Perm__Basic______macroRules____private__Mathlib__Data__List__Perm__Basic__0__List__term___x2a____1___closed__16_once, _init_lp_mathlib___private_Mathlib_Data_List_Perm_Basic_0__List___aux__Mathlib__Data__List__Perm__Basic______macroRules____private__Mathlib__Data__List__Perm__Basic__0__List__term___x2a____1___closed__16);
v___x_212_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_212_, 0, v___x_211_);
lean_ctor_set(v___x_212_, 1, v___x_210_);
return v___x_212_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Data_List_Perm_Basic_0__List___aux__Mathlib__Data__List__Perm__Basic______macroRules____private__Mathlib__Data__List__Perm__Basic__0__List__term___x2a____1___closed__18(void){
_start:
{
lean_object* v___x_213_; lean_object* v___x_214_; lean_object* v___x_215_; 
v___x_213_ = lean_box(0);
v___x_214_ = lean_obj_once(&lp_mathlib___private_Mathlib_Data_List_Perm_Basic_0__List___aux__Mathlib__Data__List__Perm__Basic______macroRules____private__Mathlib__Data__List__Perm__Basic__0__List__term___x2a____1___closed__17, &lp_mathlib___private_Mathlib_Data_List_Perm_Basic_0__List___aux__Mathlib__Data__List__Perm__Basic______macroRules____private__Mathlib__Data__List__Perm__Basic__0__List__term___x2a____1___closed__17_once, _init_lp_mathlib___private_Mathlib_Data_List_Perm_Basic_0__List___aux__Mathlib__Data__List__Perm__Basic______macroRules____private__Mathlib__Data__List__Perm__Basic__0__List__term___x2a____1___closed__17);
v___x_215_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_215_, 0, v___x_214_);
lean_ctor_set(v___x_215_, 1, v___x_213_);
return v___x_215_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Data_List_Perm_Basic_0__List___aux__Mathlib__Data__List__Perm__Basic______macroRules____private__Mathlib__Data__List__Perm__Basic__0__List__term___x2a____1(lean_object* v_x_216_, lean_object* v_a_217_, lean_object* v_a_218_){
_start:
{
lean_object* v___x_219_; lean_object* v___x_220_; uint8_t v___x_221_; 
v___x_219_ = lean_unsigned_to_nat(0u);
v___x_220_ = ((lean_object*)(lp_mathlib___private_Mathlib_Data_List_Perm_Basic_0__List_term___x2a___00__closed__1));
lean_inc(v_x_216_);
v___x_221_ = l_Lean_Syntax_isOfKind(v_x_216_, v___x_220_);
if (v___x_221_ == 0)
{
lean_object* v___x_222_; lean_object* v___x_223_; 
lean_dec(v_x_216_);
v___x_222_ = lean_box(1);
v___x_223_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_223_, 0, v___x_222_);
lean_ctor_set(v___x_223_, 1, v_a_218_);
return v___x_223_;
}
else
{
lean_object* v_quotContext_224_; lean_object* v_currMacroScope_225_; lean_object* v_ref_226_; lean_object* v___x_227_; lean_object* v___x_228_; lean_object* v___x_229_; uint8_t v___x_230_; lean_object* v___x_231_; lean_object* v___x_232_; lean_object* v___x_233_; lean_object* v___x_234_; lean_object* v___x_235_; lean_object* v___x_236_; lean_object* v___x_237_; lean_object* v___x_238_; lean_object* v___x_239_; lean_object* v___x_240_; lean_object* v___x_241_; 
v_quotContext_224_ = lean_ctor_get(v_a_217_, 1);
v_currMacroScope_225_ = lean_ctor_get(v_a_217_, 2);
v_ref_226_ = lean_ctor_get(v_a_217_, 5);
v___x_227_ = l_Lean_Syntax_getArg(v_x_216_, v___x_219_);
v___x_228_ = lean_unsigned_to_nat(2u);
v___x_229_ = l_Lean_Syntax_getArg(v_x_216_, v___x_228_);
lean_dec(v_x_216_);
v___x_230_ = 0;
v___x_231_ = l_Lean_SourceInfo_fromRef(v_ref_226_, v___x_230_);
v___x_232_ = ((lean_object*)(lp_mathlib___private_Mathlib_Data_List_Perm_Basic_0__List___aux__Mathlib__Data__List__Perm__Basic______macroRules____private__Mathlib__Data__List__Perm__Basic__0__List__term___u2218r____1___closed__4));
v___x_233_ = lean_obj_once(&lp_mathlib___private_Mathlib_Data_List_Perm_Basic_0__List___aux__Mathlib__Data__List__Perm__Basic______macroRules____private__Mathlib__Data__List__Perm__Basic__0__List__term___x2a____1___closed__1, &lp_mathlib___private_Mathlib_Data_List_Perm_Basic_0__List___aux__Mathlib__Data__List__Perm__Basic______macroRules____private__Mathlib__Data__List__Perm__Basic__0__List__term___x2a____1___closed__1_once, _init_lp_mathlib___private_Mathlib_Data_List_Perm_Basic_0__List___aux__Mathlib__Data__List__Perm__Basic______macroRules____private__Mathlib__Data__List__Perm__Basic__0__List__term___x2a____1___closed__1);
v___x_234_ = ((lean_object*)(lp_mathlib___private_Mathlib_Data_List_Perm_Basic_0__List___aux__Mathlib__Data__List__Perm__Basic______macroRules____private__Mathlib__Data__List__Perm__Basic__0__List__term___x2a____1___closed__2));
lean_inc(v_currMacroScope_225_);
lean_inc(v_quotContext_224_);
v___x_235_ = l_Lean_addMacroScope(v_quotContext_224_, v___x_234_, v_currMacroScope_225_);
v___x_236_ = lean_obj_once(&lp_mathlib___private_Mathlib_Data_List_Perm_Basic_0__List___aux__Mathlib__Data__List__Perm__Basic______macroRules____private__Mathlib__Data__List__Perm__Basic__0__List__term___x2a____1___closed__18, &lp_mathlib___private_Mathlib_Data_List_Perm_Basic_0__List___aux__Mathlib__Data__List__Perm__Basic______macroRules____private__Mathlib__Data__List__Perm__Basic__0__List__term___x2a____1___closed__18_once, _init_lp_mathlib___private_Mathlib_Data_List_Perm_Basic_0__List___aux__Mathlib__Data__List__Perm__Basic______macroRules____private__Mathlib__Data__List__Perm__Basic__0__List__term___x2a____1___closed__18);
lean_inc_n(v___x_231_, 2);
v___x_237_ = lean_alloc_ctor(3, 4, 0);
lean_ctor_set(v___x_237_, 0, v___x_231_);
lean_ctor_set(v___x_237_, 1, v___x_233_);
lean_ctor_set(v___x_237_, 2, v___x_235_);
lean_ctor_set(v___x_237_, 3, v___x_236_);
v___x_238_ = ((lean_object*)(lp_mathlib___private_Mathlib_Data_List_Perm_Basic_0__List___aux__Mathlib__Data__List__Perm__Basic______macroRules____private__Mathlib__Data__List__Perm__Basic__0__List__term___u2218r____1___closed__13));
v___x_239_ = l_Lean_Syntax_node2(v___x_231_, v___x_238_, v___x_227_, v___x_229_);
v___x_240_ = l_Lean_Syntax_node2(v___x_231_, v___x_232_, v___x_237_, v___x_239_);
v___x_241_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_241_, 0, v___x_240_);
lean_ctor_set(v___x_241_, 1, v_a_218_);
return v___x_241_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Data_List_Perm_Basic_0__List___aux__Mathlib__Data__List__Perm__Basic______macroRules____private__Mathlib__Data__List__Perm__Basic__0__List__term___x2a____1___boxed(lean_object* v_x_242_, lean_object* v_a_243_, lean_object* v_a_244_){
_start:
{
lean_object* v_res_245_; 
v_res_245_ = lp_mathlib___private_Mathlib_Data_List_Perm_Basic_0__List___aux__Mathlib__Data__List__Perm__Basic______macroRules____private__Mathlib__Data__List__Perm__Basic__0__List__term___x2a____1(v_x_242_, v_a_243_, v_a_244_);
lean_dec_ref(v_a_243_);
return v_res_245_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Data_List_Perm_Basic_0__List___aux__Mathlib__Data__List__Perm__Basic______macroRules____private__Mathlib__Data__List__Perm__Basic__0__List__term___x3c_x2a_x3e____1___closed__1(void){
_start:
{
lean_object* v___x_264_; lean_object* v___x_265_; 
v___x_264_ = ((lean_object*)(lp_mathlib___private_Mathlib_Data_List_Perm_Basic_0__List___aux__Mathlib__Data__List__Perm__Basic______macroRules____private__Mathlib__Data__List__Perm__Basic__0__List__term___x3c_x2a_x3e____1___closed__0));
v___x_265_ = l_String_toRawSubstring_x27(v___x_264_);
return v___x_265_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Data_List_Perm_Basic_0__List___aux__Mathlib__Data__List__Perm__Basic______macroRules____private__Mathlib__Data__List__Perm__Basic__0__List__term___x3c_x2a_x3e____1(lean_object* v_x_282_, lean_object* v_a_283_, lean_object* v_a_284_){
_start:
{
lean_object* v___x_285_; lean_object* v___x_286_; uint8_t v___x_287_; 
v___x_285_ = lean_unsigned_to_nat(0u);
v___x_286_ = ((lean_object*)(lp_mathlib___private_Mathlib_Data_List_Perm_Basic_0__List_term___x3c_x2a_x3e___00__closed__1));
lean_inc(v_x_282_);
v___x_287_ = l_Lean_Syntax_isOfKind(v_x_282_, v___x_286_);
if (v___x_287_ == 0)
{
lean_object* v___x_288_; lean_object* v___x_289_; 
lean_dec(v_x_282_);
v___x_288_ = lean_box(1);
v___x_289_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_289_, 0, v___x_288_);
lean_ctor_set(v___x_289_, 1, v_a_284_);
return v___x_289_;
}
else
{
lean_object* v_quotContext_290_; lean_object* v_currMacroScope_291_; lean_object* v_ref_292_; lean_object* v___x_293_; lean_object* v___x_294_; lean_object* v___x_295_; uint8_t v___x_296_; lean_object* v___x_297_; lean_object* v___x_298_; lean_object* v___x_299_; lean_object* v___x_300_; lean_object* v___x_301_; lean_object* v___x_302_; lean_object* v___x_303_; lean_object* v___x_304_; lean_object* v___x_305_; lean_object* v___x_306_; lean_object* v___x_307_; lean_object* v___x_308_; lean_object* v___x_309_; lean_object* v___x_310_; lean_object* v___x_311_; lean_object* v___x_312_; 
v_quotContext_290_ = lean_ctor_get(v_a_283_, 1);
v_currMacroScope_291_ = lean_ctor_get(v_a_283_, 2);
v_ref_292_ = lean_ctor_get(v_a_283_, 5);
v___x_293_ = l_Lean_Syntax_getArg(v_x_282_, v___x_285_);
v___x_294_ = lean_unsigned_to_nat(2u);
v___x_295_ = l_Lean_Syntax_getArg(v_x_282_, v___x_294_);
lean_dec(v_x_282_);
v___x_296_ = 0;
v___x_297_ = l_Lean_SourceInfo_fromRef(v_ref_292_, v___x_296_);
v___x_298_ = ((lean_object*)(lp_mathlib___private_Mathlib_Data_List_Perm_Basic_0__List___aux__Mathlib__Data__List__Perm__Basic______macroRules____private__Mathlib__Data__List__Perm__Basic__0__List__term___u2218r____1___closed__4));
v___x_299_ = lean_obj_once(&lp_mathlib___private_Mathlib_Data_List_Perm_Basic_0__List___aux__Mathlib__Data__List__Perm__Basic______macroRules____private__Mathlib__Data__List__Perm__Basic__0__List__term___x3c_x2a_x3e____1___closed__1, &lp_mathlib___private_Mathlib_Data_List_Perm_Basic_0__List___aux__Mathlib__Data__List__Perm__Basic______macroRules____private__Mathlib__Data__List__Perm__Basic__0__List__term___x3c_x2a_x3e____1___closed__1_once, _init_lp_mathlib___private_Mathlib_Data_List_Perm_Basic_0__List___aux__Mathlib__Data__List__Perm__Basic______macroRules____private__Mathlib__Data__List__Perm__Basic__0__List__term___x3c_x2a_x3e____1___closed__1);
v___x_300_ = ((lean_object*)(lp_mathlib___private_Mathlib_Data_List_Perm_Basic_0__List___aux__Mathlib__Data__List__Perm__Basic______macroRules____private__Mathlib__Data__List__Perm__Basic__0__List__term___x3c_x2a_x3e____1___closed__2));
lean_inc_n(v_currMacroScope_291_, 2);
lean_inc_n(v_quotContext_290_, 2);
v___x_301_ = l_Lean_addMacroScope(v_quotContext_290_, v___x_300_, v_currMacroScope_291_);
v___x_302_ = ((lean_object*)(lp_mathlib___private_Mathlib_Data_List_Perm_Basic_0__List___aux__Mathlib__Data__List__Perm__Basic______macroRules____private__Mathlib__Data__List__Perm__Basic__0__List__term___x3c_x2a_x3e____1___closed__7));
lean_inc_n(v___x_297_, 3);
v___x_303_ = lean_alloc_ctor(3, 4, 0);
lean_ctor_set(v___x_303_, 0, v___x_297_);
lean_ctor_set(v___x_303_, 1, v___x_299_);
lean_ctor_set(v___x_303_, 2, v___x_301_);
lean_ctor_set(v___x_303_, 3, v___x_302_);
v___x_304_ = ((lean_object*)(lp_mathlib___private_Mathlib_Data_List_Perm_Basic_0__List___aux__Mathlib__Data__List__Perm__Basic______macroRules____private__Mathlib__Data__List__Perm__Basic__0__List__term___u2218r____1___closed__13));
v___x_305_ = lean_obj_once(&lp_mathlib___private_Mathlib_Data_List_Perm_Basic_0__List___aux__Mathlib__Data__List__Perm__Basic______macroRules____private__Mathlib__Data__List__Perm__Basic__0__List__term___x2a____1___closed__1, &lp_mathlib___private_Mathlib_Data_List_Perm_Basic_0__List___aux__Mathlib__Data__List__Perm__Basic______macroRules____private__Mathlib__Data__List__Perm__Basic__0__List__term___x2a____1___closed__1_once, _init_lp_mathlib___private_Mathlib_Data_List_Perm_Basic_0__List___aux__Mathlib__Data__List__Perm__Basic______macroRules____private__Mathlib__Data__List__Perm__Basic__0__List__term___x2a____1___closed__1);
v___x_306_ = ((lean_object*)(lp_mathlib___private_Mathlib_Data_List_Perm_Basic_0__List___aux__Mathlib__Data__List__Perm__Basic______macroRules____private__Mathlib__Data__List__Perm__Basic__0__List__term___x2a____1___closed__2));
v___x_307_ = l_Lean_addMacroScope(v_quotContext_290_, v___x_306_, v_currMacroScope_291_);
v___x_308_ = lean_obj_once(&lp_mathlib___private_Mathlib_Data_List_Perm_Basic_0__List___aux__Mathlib__Data__List__Perm__Basic______macroRules____private__Mathlib__Data__List__Perm__Basic__0__List__term___x2a____1___closed__18, &lp_mathlib___private_Mathlib_Data_List_Perm_Basic_0__List___aux__Mathlib__Data__List__Perm__Basic______macroRules____private__Mathlib__Data__List__Perm__Basic__0__List__term___x2a____1___closed__18_once, _init_lp_mathlib___private_Mathlib_Data_List_Perm_Basic_0__List___aux__Mathlib__Data__List__Perm__Basic______macroRules____private__Mathlib__Data__List__Perm__Basic__0__List__term___x2a____1___closed__18);
v___x_309_ = lean_alloc_ctor(3, 4, 0);
lean_ctor_set(v___x_309_, 0, v___x_297_);
lean_ctor_set(v___x_309_, 1, v___x_305_);
lean_ctor_set(v___x_309_, 2, v___x_307_);
lean_ctor_set(v___x_309_, 3, v___x_308_);
v___x_310_ = l_Lean_Syntax_node3(v___x_297_, v___x_304_, v___x_309_, v___x_295_, v___x_293_);
v___x_311_ = l_Lean_Syntax_node2(v___x_297_, v___x_298_, v___x_303_, v___x_310_);
v___x_312_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_312_, 0, v___x_311_);
lean_ctor_set(v___x_312_, 1, v_a_284_);
return v___x_312_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Data_List_Perm_Basic_0__List___aux__Mathlib__Data__List__Perm__Basic______macroRules____private__Mathlib__Data__List__Perm__Basic__0__List__term___x3c_x2a_x3e____1___boxed(lean_object* v_x_313_, lean_object* v_a_314_, lean_object* v_a_315_){
_start:
{
lean_object* v_res_316_; 
v_res_316_ = lp_mathlib___private_Mathlib_Data_List_Perm_Basic_0__List___aux__Mathlib__Data__List__Perm__Basic______macroRules____private__Mathlib__Data__List__Perm__Basic__0__List__term___x3c_x2a_x3e____1(v_x_313_, v_a_314_, v_a_315_);
lean_dec_ref(v_a_314_);
return v_res_316_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Data_List_Perm_Basic_0__List___aux__Mathlib__Data__List__Perm__Basic______unexpand__List__foldl__1(lean_object* v_x_317_, lean_object* v_a_318_, lean_object* v_a_319_){
_start:
{
lean_object* v___x_320_; uint8_t v___x_321_; 
v___x_320_ = ((lean_object*)(lp_mathlib___private_Mathlib_Data_List_Perm_Basic_0__List___aux__Mathlib__Data__List__Perm__Basic______macroRules____private__Mathlib__Data__List__Perm__Basic__0__List__term___u2218r____1___closed__4));
lean_inc(v_x_317_);
v___x_321_ = l_Lean_Syntax_isOfKind(v_x_317_, v___x_320_);
if (v___x_321_ == 0)
{
lean_object* v___x_322_; lean_object* v___x_323_; 
lean_dec(v_x_317_);
v___x_322_ = lean_box(0);
v___x_323_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_323_, 0, v___x_322_);
lean_ctor_set(v___x_323_, 1, v_a_319_);
return v___x_323_;
}
else
{
lean_object* v___x_324_; lean_object* v___x_325_; lean_object* v___x_326_; uint8_t v___x_327_; 
v___x_324_ = lean_unsigned_to_nat(0u);
v___x_325_ = l_Lean_Syntax_getArg(v_x_317_, v___x_324_);
v___x_326_ = ((lean_object*)(lp_mathlib___private_Mathlib_Data_List_Perm_Basic_0__List___aux__Mathlib__Data__List__Perm__Basic______unexpand__Relation__Comp__1___closed__1));
lean_inc(v___x_325_);
v___x_327_ = l_Lean_Syntax_isOfKind(v___x_325_, v___x_326_);
if (v___x_327_ == 0)
{
lean_object* v___x_328_; lean_object* v___x_329_; 
lean_dec(v___x_325_);
lean_dec(v_x_317_);
v___x_328_ = lean_box(0);
v___x_329_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_329_, 0, v___x_328_);
lean_ctor_set(v___x_329_, 1, v_a_319_);
return v___x_329_;
}
else
{
lean_object* v___x_330_; lean_object* v___x_331_; lean_object* v___x_332_; uint8_t v___x_333_; 
v___x_330_ = lean_unsigned_to_nat(1u);
v___x_331_ = l_Lean_Syntax_getArg(v_x_317_, v___x_330_);
lean_dec(v_x_317_);
v___x_332_ = lean_unsigned_to_nat(3u);
lean_inc(v___x_331_);
v___x_333_ = l_Lean_Syntax_matchesNull(v___x_331_, v___x_332_);
if (v___x_333_ == 0)
{
lean_object* v___x_334_; lean_object* v___x_335_; 
lean_dec(v___x_331_);
lean_dec(v___x_325_);
v___x_334_ = lean_box(0);
v___x_335_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_335_, 0, v___x_334_);
lean_ctor_set(v___x_335_, 1, v_a_319_);
return v___x_335_;
}
else
{
lean_object* v___x_336_; lean_object* v___x_337_; uint8_t v___x_338_; 
v___x_336_ = l_Lean_Syntax_getArg(v___x_331_, v___x_324_);
v___x_337_ = ((lean_object*)(lp_mathlib___private_Mathlib_Data_List_Perm_Basic_0__List___aux__Mathlib__Data__List__Perm__Basic______macroRules____private__Mathlib__Data__List__Perm__Basic__0__List__term___x2a____1___closed__2));
v___x_338_ = l_Lean_Syntax_matchesIdent(v___x_336_, v___x_337_);
lean_dec(v___x_336_);
if (v___x_338_ == 0)
{
lean_object* v___x_339_; lean_object* v___x_340_; 
lean_dec(v___x_331_);
lean_dec(v___x_325_);
v___x_339_ = lean_box(0);
v___x_340_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_340_, 0, v___x_339_);
lean_ctor_set(v___x_340_, 1, v_a_319_);
return v___x_340_;
}
else
{
lean_object* v___x_341_; lean_object* v___x_342_; lean_object* v___x_343_; lean_object* v_ref_344_; uint8_t v___x_345_; lean_object* v___x_346_; lean_object* v___x_347_; lean_object* v___x_348_; lean_object* v___x_349_; lean_object* v___x_350_; lean_object* v___x_351_; 
v___x_341_ = l_Lean_Syntax_getArg(v___x_331_, v___x_330_);
v___x_342_ = lean_unsigned_to_nat(2u);
v___x_343_ = l_Lean_Syntax_getArg(v___x_331_, v___x_342_);
lean_dec(v___x_331_);
v_ref_344_ = l_Lean_replaceRef(v___x_325_, v_a_318_);
lean_dec(v___x_325_);
v___x_345_ = 0;
v___x_346_ = l_Lean_SourceInfo_fromRef(v_ref_344_, v___x_345_);
lean_dec(v_ref_344_);
v___x_347_ = ((lean_object*)(lp_mathlib___private_Mathlib_Data_List_Perm_Basic_0__List_term___x3c_x2a_x3e___00__closed__1));
v___x_348_ = ((lean_object*)(lp_mathlib___private_Mathlib_Data_List_Perm_Basic_0__List_term___x3c_x2a_x3e___00__closed__2));
lean_inc(v___x_346_);
v___x_349_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_349_, 0, v___x_346_);
lean_ctor_set(v___x_349_, 1, v___x_348_);
v___x_350_ = l_Lean_Syntax_node3(v___x_346_, v___x_347_, v___x_343_, v___x_349_, v___x_341_);
v___x_351_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_351_, 0, v___x_350_);
lean_ctor_set(v___x_351_, 1, v_a_319_);
return v___x_351_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Data_List_Perm_Basic_0__List___aux__Mathlib__Data__List__Perm__Basic______unexpand__List__foldl__1___boxed(lean_object* v_x_352_, lean_object* v_a_353_, lean_object* v_a_354_){
_start:
{
lean_object* v_res_355_; 
v_res_355_ = lp_mathlib___private_Mathlib_Data_List_Perm_Basic_0__List___aux__Mathlib__Data__List__Perm__Basic______unexpand__List__foldl__1(v_x_352_, v_a_353_, v_a_354_);
lean_dec(v_a_353_);
return v_res_355_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_batteries_Batteries_Data_List_Perm(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Logic_Relation(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Data_List_Forall2(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Data_List_InsertIdx(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Logic_OpClass(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_Data_List_Perm_Basic(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_batteries_Batteries_Data_List_Perm(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Logic_Relation(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Data_List_Forall2(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Data_List_InsertIdx(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Logic_OpClass(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_Data_List_Perm_Basic(uint8_t builtin) {
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
lean_object* initialize_batteries_Batteries_Data_List_Perm(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Logic_Relation(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Data_List_Forall2(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Data_List_InsertIdx(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Logic_OpClass(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Data_List_Perm_Basic(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_batteries_Batteries_Data_List_Perm(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Logic_Relation(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_List_Forall2(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_List_InsertIdx(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Logic_OpClass(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Data_List_Perm_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_Data_List_Perm_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_Data_List_Perm_Basic(builtin);
}
#ifdef __cplusplus
}
#endif
