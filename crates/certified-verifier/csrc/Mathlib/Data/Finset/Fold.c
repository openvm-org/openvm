// Lean compiler output
// Module: Mathlib.Data.Finset.Fold
// Imports: public import Init public meta import Init public import Mathlib.Data.Finset.Basic public import Mathlib.Data.Finset.Image public import Mathlib.Data.Multiset.Fold public import Mathlib.Data.Finset.Lattice.Lemmas
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
lean_object* l_Lean_Name_mkStr4(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Name_mkStr1(lean_object*);
uint8_t l_Lean_Syntax_isOfKind(lean_object*, lean_object*);
lean_object* l_Lean_Syntax_getArg(lean_object*, lean_object*);
lean_object* l_Lean_SourceInfo_fromRef(lean_object*, uint8_t);
lean_object* l_String_toRawSubstring_x27(lean_object*);
lean_object* l_Lean_addMacroScope(lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Syntax_node2(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Multiset_map___redArg(lean_object*, lean_object*);
lean_object* l_List_foldrTR___redArg(lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib___private_Mathlib_Data_Finset_Fold_0__Finset_term___x2a___00__closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "_private"};
static const lean_object* lp_mathlib___private_Mathlib_Data_Finset_Fold_0__Finset_term___x2a___00__closed__0 = (const lean_object*)&lp_mathlib___private_Mathlib_Data_Finset_Fold_0__Finset_term___x2a___00__closed__0_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Data_Finset_Fold_0__Finset_term___x2a___00__closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Data_Finset_Fold_0__Finset_term___x2a___00__closed__0_value),LEAN_SCALAR_PTR_LITERAL(103, 214, 75, 80, 34, 198, 193, 153)}};
static const lean_object* lp_mathlib___private_Mathlib_Data_Finset_Fold_0__Finset_term___x2a___00__closed__1 = (const lean_object*)&lp_mathlib___private_Mathlib_Data_Finset_Fold_0__Finset_term___x2a___00__closed__1_value;
static const lean_string_object lp_mathlib___private_Mathlib_Data_Finset_Fold_0__Finset_term___x2a___00__closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "Mathlib"};
static const lean_object* lp_mathlib___private_Mathlib_Data_Finset_Fold_0__Finset_term___x2a___00__closed__2 = (const lean_object*)&lp_mathlib___private_Mathlib_Data_Finset_Fold_0__Finset_term___x2a___00__closed__2_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Data_Finset_Fold_0__Finset_term___x2a___00__closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Data_Finset_Fold_0__Finset_term___x2a___00__closed__1_value),((lean_object*)&lp_mathlib___private_Mathlib_Data_Finset_Fold_0__Finset_term___x2a___00__closed__2_value),LEAN_SCALAR_PTR_LITERAL(234, 232, 174, 134, 127, 136, 69, 92)}};
static const lean_object* lp_mathlib___private_Mathlib_Data_Finset_Fold_0__Finset_term___x2a___00__closed__3 = (const lean_object*)&lp_mathlib___private_Mathlib_Data_Finset_Fold_0__Finset_term___x2a___00__closed__3_value;
static const lean_string_object lp_mathlib___private_Mathlib_Data_Finset_Fold_0__Finset_term___x2a___00__closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "Data"};
static const lean_object* lp_mathlib___private_Mathlib_Data_Finset_Fold_0__Finset_term___x2a___00__closed__4 = (const lean_object*)&lp_mathlib___private_Mathlib_Data_Finset_Fold_0__Finset_term___x2a___00__closed__4_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Data_Finset_Fold_0__Finset_term___x2a___00__closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Data_Finset_Fold_0__Finset_term___x2a___00__closed__3_value),((lean_object*)&lp_mathlib___private_Mathlib_Data_Finset_Fold_0__Finset_term___x2a___00__closed__4_value),LEAN_SCALAR_PTR_LITERAL(149, 48, 214, 5, 44, 128, 44, 12)}};
static const lean_object* lp_mathlib___private_Mathlib_Data_Finset_Fold_0__Finset_term___x2a___00__closed__5 = (const lean_object*)&lp_mathlib___private_Mathlib_Data_Finset_Fold_0__Finset_term___x2a___00__closed__5_value;
static const lean_string_object lp_mathlib___private_Mathlib_Data_Finset_Fold_0__Finset_term___x2a___00__closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "Finset"};
static const lean_object* lp_mathlib___private_Mathlib_Data_Finset_Fold_0__Finset_term___x2a___00__closed__6 = (const lean_object*)&lp_mathlib___private_Mathlib_Data_Finset_Fold_0__Finset_term___x2a___00__closed__6_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Data_Finset_Fold_0__Finset_term___x2a___00__closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Data_Finset_Fold_0__Finset_term___x2a___00__closed__5_value),((lean_object*)&lp_mathlib___private_Mathlib_Data_Finset_Fold_0__Finset_term___x2a___00__closed__6_value),LEAN_SCALAR_PTR_LITERAL(57, 113, 226, 246, 136, 115, 60, 58)}};
static const lean_object* lp_mathlib___private_Mathlib_Data_Finset_Fold_0__Finset_term___x2a___00__closed__7 = (const lean_object*)&lp_mathlib___private_Mathlib_Data_Finset_Fold_0__Finset_term___x2a___00__closed__7_value;
static const lean_string_object lp_mathlib___private_Mathlib_Data_Finset_Fold_0__Finset_term___x2a___00__closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "Fold"};
static const lean_object* lp_mathlib___private_Mathlib_Data_Finset_Fold_0__Finset_term___x2a___00__closed__8 = (const lean_object*)&lp_mathlib___private_Mathlib_Data_Finset_Fold_0__Finset_term___x2a___00__closed__8_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Data_Finset_Fold_0__Finset_term___x2a___00__closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Data_Finset_Fold_0__Finset_term___x2a___00__closed__7_value),((lean_object*)&lp_mathlib___private_Mathlib_Data_Finset_Fold_0__Finset_term___x2a___00__closed__8_value),LEAN_SCALAR_PTR_LITERAL(59, 179, 112, 76, 219, 22, 28, 13)}};
static const lean_object* lp_mathlib___private_Mathlib_Data_Finset_Fold_0__Finset_term___x2a___00__closed__9 = (const lean_object*)&lp_mathlib___private_Mathlib_Data_Finset_Fold_0__Finset_term___x2a___00__closed__9_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Data_Finset_Fold_0__Finset_term___x2a___00__closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Data_Finset_Fold_0__Finset_term___x2a___00__closed__9_value),((lean_object*)(((size_t)(0) << 1) | 1)),LEAN_SCALAR_PTR_LITERAL(70, 37, 128, 126, 88, 6, 98, 135)}};
static const lean_object* lp_mathlib___private_Mathlib_Data_Finset_Fold_0__Finset_term___x2a___00__closed__10 = (const lean_object*)&lp_mathlib___private_Mathlib_Data_Finset_Fold_0__Finset_term___x2a___00__closed__10_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Data_Finset_Fold_0__Finset_term___x2a___00__closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Data_Finset_Fold_0__Finset_term___x2a___00__closed__10_value),((lean_object*)&lp_mathlib___private_Mathlib_Data_Finset_Fold_0__Finset_term___x2a___00__closed__6_value),LEAN_SCALAR_PTR_LITERAL(62, 42, 160, 132, 217, 207, 237, 74)}};
static const lean_object* lp_mathlib___private_Mathlib_Data_Finset_Fold_0__Finset_term___x2a___00__closed__11 = (const lean_object*)&lp_mathlib___private_Mathlib_Data_Finset_Fold_0__Finset_term___x2a___00__closed__11_value;
static const lean_string_object lp_mathlib___private_Mathlib_Data_Finset_Fold_0__Finset_term___x2a___00__closed__12_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "term_*_"};
static const lean_object* lp_mathlib___private_Mathlib_Data_Finset_Fold_0__Finset_term___x2a___00__closed__12 = (const lean_object*)&lp_mathlib___private_Mathlib_Data_Finset_Fold_0__Finset_term___x2a___00__closed__12_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Data_Finset_Fold_0__Finset_term___x2a___00__closed__13_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Data_Finset_Fold_0__Finset_term___x2a___00__closed__11_value),((lean_object*)&lp_mathlib___private_Mathlib_Data_Finset_Fold_0__Finset_term___x2a___00__closed__12_value),LEAN_SCALAR_PTR_LITERAL(103, 139, 4, 43, 170, 131, 125, 167)}};
static const lean_object* lp_mathlib___private_Mathlib_Data_Finset_Fold_0__Finset_term___x2a___00__closed__13 = (const lean_object*)&lp_mathlib___private_Mathlib_Data_Finset_Fold_0__Finset_term___x2a___00__closed__13_value;
static const lean_string_object lp_mathlib___private_Mathlib_Data_Finset_Fold_0__Finset_term___x2a___00__closed__14_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "andthen"};
static const lean_object* lp_mathlib___private_Mathlib_Data_Finset_Fold_0__Finset_term___x2a___00__closed__14 = (const lean_object*)&lp_mathlib___private_Mathlib_Data_Finset_Fold_0__Finset_term___x2a___00__closed__14_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Data_Finset_Fold_0__Finset_term___x2a___00__closed__15_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Data_Finset_Fold_0__Finset_term___x2a___00__closed__14_value),LEAN_SCALAR_PTR_LITERAL(40, 255, 78, 30, 143, 119, 117, 174)}};
static const lean_object* lp_mathlib___private_Mathlib_Data_Finset_Fold_0__Finset_term___x2a___00__closed__15 = (const lean_object*)&lp_mathlib___private_Mathlib_Data_Finset_Fold_0__Finset_term___x2a___00__closed__15_value;
static const lean_string_object lp_mathlib___private_Mathlib_Data_Finset_Fold_0__Finset_term___x2a___00__closed__16_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = " * "};
static const lean_object* lp_mathlib___private_Mathlib_Data_Finset_Fold_0__Finset_term___x2a___00__closed__16 = (const lean_object*)&lp_mathlib___private_Mathlib_Data_Finset_Fold_0__Finset_term___x2a___00__closed__16_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Data_Finset_Fold_0__Finset_term___x2a___00__closed__17_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 5}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Data_Finset_Fold_0__Finset_term___x2a___00__closed__16_value)}};
static const lean_object* lp_mathlib___private_Mathlib_Data_Finset_Fold_0__Finset_term___x2a___00__closed__17 = (const lean_object*)&lp_mathlib___private_Mathlib_Data_Finset_Fold_0__Finset_term___x2a___00__closed__17_value;
static const lean_string_object lp_mathlib___private_Mathlib_Data_Finset_Fold_0__Finset_term___x2a___00__closed__18_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "term"};
static const lean_object* lp_mathlib___private_Mathlib_Data_Finset_Fold_0__Finset_term___x2a___00__closed__18 = (const lean_object*)&lp_mathlib___private_Mathlib_Data_Finset_Fold_0__Finset_term___x2a___00__closed__18_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Data_Finset_Fold_0__Finset_term___x2a___00__closed__19_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Data_Finset_Fold_0__Finset_term___x2a___00__closed__18_value),LEAN_SCALAR_PTR_LITERAL(187, 230, 181, 162, 253, 146, 122, 119)}};
static const lean_object* lp_mathlib___private_Mathlib_Data_Finset_Fold_0__Finset_term___x2a___00__closed__19 = (const lean_object*)&lp_mathlib___private_Mathlib_Data_Finset_Fold_0__Finset_term___x2a___00__closed__19_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Data_Finset_Fold_0__Finset_term___x2a___00__closed__20_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 7}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Data_Finset_Fold_0__Finset_term___x2a___00__closed__19_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib___private_Mathlib_Data_Finset_Fold_0__Finset_term___x2a___00__closed__20 = (const lean_object*)&lp_mathlib___private_Mathlib_Data_Finset_Fold_0__Finset_term___x2a___00__closed__20_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Data_Finset_Fold_0__Finset_term___x2a___00__closed__21_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Data_Finset_Fold_0__Finset_term___x2a___00__closed__15_value),((lean_object*)&lp_mathlib___private_Mathlib_Data_Finset_Fold_0__Finset_term___x2a___00__closed__17_value),((lean_object*)&lp_mathlib___private_Mathlib_Data_Finset_Fold_0__Finset_term___x2a___00__closed__20_value)}};
static const lean_object* lp_mathlib___private_Mathlib_Data_Finset_Fold_0__Finset_term___x2a___00__closed__21 = (const lean_object*)&lp_mathlib___private_Mathlib_Data_Finset_Fold_0__Finset_term___x2a___00__closed__21_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Data_Finset_Fold_0__Finset_term___x2a___00__closed__22_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*4 + 0, .m_other = 4, .m_tag = 4}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Data_Finset_Fold_0__Finset_term___x2a___00__closed__13_value),((lean_object*)(((size_t)(1022) << 1) | 1)),((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Data_Finset_Fold_0__Finset_term___x2a___00__closed__21_value)}};
static const lean_object* lp_mathlib___private_Mathlib_Data_Finset_Fold_0__Finset_term___x2a___00__closed__22 = (const lean_object*)&lp_mathlib___private_Mathlib_Data_Finset_Fold_0__Finset_term___x2a___00__closed__22_value;
LEAN_EXPORT const lean_object* lp_mathlib___private_Mathlib_Data_Finset_Fold_0__Finset_term___x2a__ = (const lean_object*)&lp_mathlib___private_Mathlib_Data_Finset_Fold_0__Finset_term___x2a___00__closed__22_value;
static const lean_string_object lp_mathlib___private_Mathlib_Data_Finset_Fold_0__Finset___aux__Mathlib__Data__Finset__Fold______macroRules____private__Mathlib__Data__Finset__Fold__0__Finset__term___x2a____1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "Lean"};
static const lean_object* lp_mathlib___private_Mathlib_Data_Finset_Fold_0__Finset___aux__Mathlib__Data__Finset__Fold______macroRules____private__Mathlib__Data__Finset__Fold__0__Finset__term___x2a____1___closed__0 = (const lean_object*)&lp_mathlib___private_Mathlib_Data_Finset_Fold_0__Finset___aux__Mathlib__Data__Finset__Fold______macroRules____private__Mathlib__Data__Finset__Fold__0__Finset__term___x2a____1___closed__0_value;
static const lean_string_object lp_mathlib___private_Mathlib_Data_Finset_Fold_0__Finset___aux__Mathlib__Data__Finset__Fold______macroRules____private__Mathlib__Data__Finset__Fold__0__Finset__term___x2a____1___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "Parser"};
static const lean_object* lp_mathlib___private_Mathlib_Data_Finset_Fold_0__Finset___aux__Mathlib__Data__Finset__Fold______macroRules____private__Mathlib__Data__Finset__Fold__0__Finset__term___x2a____1___closed__1 = (const lean_object*)&lp_mathlib___private_Mathlib_Data_Finset_Fold_0__Finset___aux__Mathlib__Data__Finset__Fold______macroRules____private__Mathlib__Data__Finset__Fold__0__Finset__term___x2a____1___closed__1_value;
static const lean_string_object lp_mathlib___private_Mathlib_Data_Finset_Fold_0__Finset___aux__Mathlib__Data__Finset__Fold______macroRules____private__Mathlib__Data__Finset__Fold__0__Finset__term___x2a____1___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "Term"};
static const lean_object* lp_mathlib___private_Mathlib_Data_Finset_Fold_0__Finset___aux__Mathlib__Data__Finset__Fold______macroRules____private__Mathlib__Data__Finset__Fold__0__Finset__term___x2a____1___closed__2 = (const lean_object*)&lp_mathlib___private_Mathlib_Data_Finset_Fold_0__Finset___aux__Mathlib__Data__Finset__Fold______macroRules____private__Mathlib__Data__Finset__Fold__0__Finset__term___x2a____1___closed__2_value;
static const lean_string_object lp_mathlib___private_Mathlib_Data_Finset_Fold_0__Finset___aux__Mathlib__Data__Finset__Fold______macroRules____private__Mathlib__Data__Finset__Fold__0__Finset__term___x2a____1___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = "app"};
static const lean_object* lp_mathlib___private_Mathlib_Data_Finset_Fold_0__Finset___aux__Mathlib__Data__Finset__Fold______macroRules____private__Mathlib__Data__Finset__Fold__0__Finset__term___x2a____1___closed__3 = (const lean_object*)&lp_mathlib___private_Mathlib_Data_Finset_Fold_0__Finset___aux__Mathlib__Data__Finset__Fold______macroRules____private__Mathlib__Data__Finset__Fold__0__Finset__term___x2a____1___closed__3_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Data_Finset_Fold_0__Finset___aux__Mathlib__Data__Finset__Fold______macroRules____private__Mathlib__Data__Finset__Fold__0__Finset__term___x2a____1___closed__4_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Data_Finset_Fold_0__Finset___aux__Mathlib__Data__Finset__Fold______macroRules____private__Mathlib__Data__Finset__Fold__0__Finset__term___x2a____1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Data_Finset_Fold_0__Finset___aux__Mathlib__Data__Finset__Fold______macroRules____private__Mathlib__Data__Finset__Fold__0__Finset__term___x2a____1___closed__4_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Data_Finset_Fold_0__Finset___aux__Mathlib__Data__Finset__Fold______macroRules____private__Mathlib__Data__Finset__Fold__0__Finset__term___x2a____1___closed__4_value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Data_Finset_Fold_0__Finset___aux__Mathlib__Data__Finset__Fold______macroRules____private__Mathlib__Data__Finset__Fold__0__Finset__term___x2a____1___closed__1_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Data_Finset_Fold_0__Finset___aux__Mathlib__Data__Finset__Fold______macroRules____private__Mathlib__Data__Finset__Fold__0__Finset__term___x2a____1___closed__4_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Data_Finset_Fold_0__Finset___aux__Mathlib__Data__Finset__Fold______macroRules____private__Mathlib__Data__Finset__Fold__0__Finset__term___x2a____1___closed__4_value_aux_1),((lean_object*)&lp_mathlib___private_Mathlib_Data_Finset_Fold_0__Finset___aux__Mathlib__Data__Finset__Fold______macroRules____private__Mathlib__Data__Finset__Fold__0__Finset__term___x2a____1___closed__2_value),LEAN_SCALAR_PTR_LITERAL(75, 170, 162, 138, 136, 204, 251, 229)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Data_Finset_Fold_0__Finset___aux__Mathlib__Data__Finset__Fold______macroRules____private__Mathlib__Data__Finset__Fold__0__Finset__term___x2a____1___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Data_Finset_Fold_0__Finset___aux__Mathlib__Data__Finset__Fold______macroRules____private__Mathlib__Data__Finset__Fold__0__Finset__term___x2a____1___closed__4_value_aux_2),((lean_object*)&lp_mathlib___private_Mathlib_Data_Finset_Fold_0__Finset___aux__Mathlib__Data__Finset__Fold______macroRules____private__Mathlib__Data__Finset__Fold__0__Finset__term___x2a____1___closed__3_value),LEAN_SCALAR_PTR_LITERAL(69, 118, 10, 41, 220, 156, 243, 179)}};
static const lean_object* lp_mathlib___private_Mathlib_Data_Finset_Fold_0__Finset___aux__Mathlib__Data__Finset__Fold______macroRules____private__Mathlib__Data__Finset__Fold__0__Finset__term___x2a____1___closed__4 = (const lean_object*)&lp_mathlib___private_Mathlib_Data_Finset_Fold_0__Finset___aux__Mathlib__Data__Finset__Fold______macroRules____private__Mathlib__Data__Finset__Fold__0__Finset__term___x2a____1___closed__4_value;
static const lean_string_object lp_mathlib___private_Mathlib_Data_Finset_Fold_0__Finset___aux__Mathlib__Data__Finset__Fold______macroRules____private__Mathlib__Data__Finset__Fold__0__Finset__term___x2a____1___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 3, .m_capacity = 3, .m_length = 2, .m_data = "op"};
static const lean_object* lp_mathlib___private_Mathlib_Data_Finset_Fold_0__Finset___aux__Mathlib__Data__Finset__Fold______macroRules____private__Mathlib__Data__Finset__Fold__0__Finset__term___x2a____1___closed__5 = (const lean_object*)&lp_mathlib___private_Mathlib_Data_Finset_Fold_0__Finset___aux__Mathlib__Data__Finset__Fold______macroRules____private__Mathlib__Data__Finset__Fold__0__Finset__term___x2a____1___closed__5_value;
static lean_once_cell_t lp_mathlib___private_Mathlib_Data_Finset_Fold_0__Finset___aux__Mathlib__Data__Finset__Fold______macroRules____private__Mathlib__Data__Finset__Fold__0__Finset__term___x2a____1___closed__6_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Data_Finset_Fold_0__Finset___aux__Mathlib__Data__Finset__Fold______macroRules____private__Mathlib__Data__Finset__Fold__0__Finset__term___x2a____1___closed__6;
static const lean_ctor_object lp_mathlib___private_Mathlib_Data_Finset_Fold_0__Finset___aux__Mathlib__Data__Finset__Fold______macroRules____private__Mathlib__Data__Finset__Fold__0__Finset__term___x2a____1___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Data_Finset_Fold_0__Finset___aux__Mathlib__Data__Finset__Fold______macroRules____private__Mathlib__Data__Finset__Fold__0__Finset__term___x2a____1___closed__5_value),LEAN_SCALAR_PTR_LITERAL(234, 254, 5, 67, 7, 92, 131, 164)}};
static const lean_object* lp_mathlib___private_Mathlib_Data_Finset_Fold_0__Finset___aux__Mathlib__Data__Finset__Fold______macroRules____private__Mathlib__Data__Finset__Fold__0__Finset__term___x2a____1___closed__7 = (const lean_object*)&lp_mathlib___private_Mathlib_Data_Finset_Fold_0__Finset___aux__Mathlib__Data__Finset__Fold______macroRules____private__Mathlib__Data__Finset__Fold__0__Finset__term___x2a____1___closed__7_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Data_Finset_Fold_0__Finset___aux__Mathlib__Data__Finset__Fold______macroRules____private__Mathlib__Data__Finset__Fold__0__Finset__term___x2a____1___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Data_Finset_Fold_0__Finset___aux__Mathlib__Data__Finset__Fold______macroRules____private__Mathlib__Data__Finset__Fold__0__Finset__term___x2a____1___closed__5_value),LEAN_SCALAR_PTR_LITERAL(234, 254, 5, 67, 7, 92, 131, 164)}};
static const lean_object* lp_mathlib___private_Mathlib_Data_Finset_Fold_0__Finset___aux__Mathlib__Data__Finset__Fold______macroRules____private__Mathlib__Data__Finset__Fold__0__Finset__term___x2a____1___closed__8 = (const lean_object*)&lp_mathlib___private_Mathlib_Data_Finset_Fold_0__Finset___aux__Mathlib__Data__Finset__Fold______macroRules____private__Mathlib__Data__Finset__Fold__0__Finset__term___x2a____1___closed__8_value;
static const lean_string_object lp_mathlib___private_Mathlib_Data_Finset_Fold_0__Finset___aux__Mathlib__Data__Finset__Fold______macroRules____private__Mathlib__Data__Finset__Fold__0__Finset__term___x2a____1___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 3, .m_capacity = 3, .m_length = 2, .m_data = "_@"};
static const lean_object* lp_mathlib___private_Mathlib_Data_Finset_Fold_0__Finset___aux__Mathlib__Data__Finset__Fold______macroRules____private__Mathlib__Data__Finset__Fold__0__Finset__term___x2a____1___closed__9 = (const lean_object*)&lp_mathlib___private_Mathlib_Data_Finset_Fold_0__Finset___aux__Mathlib__Data__Finset__Fold______macroRules____private__Mathlib__Data__Finset__Fold__0__Finset__term___x2a____1___closed__9_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Data_Finset_Fold_0__Finset___aux__Mathlib__Data__Finset__Fold______macroRules____private__Mathlib__Data__Finset__Fold__0__Finset__term___x2a____1___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Data_Finset_Fold_0__Finset___aux__Mathlib__Data__Finset__Fold______macroRules____private__Mathlib__Data__Finset__Fold__0__Finset__term___x2a____1___closed__8_value),((lean_object*)&lp_mathlib___private_Mathlib_Data_Finset_Fold_0__Finset___aux__Mathlib__Data__Finset__Fold______macroRules____private__Mathlib__Data__Finset__Fold__0__Finset__term___x2a____1___closed__9_value),LEAN_SCALAR_PTR_LITERAL(195, 225, 22, 247, 238, 155, 51, 65)}};
static const lean_object* lp_mathlib___private_Mathlib_Data_Finset_Fold_0__Finset___aux__Mathlib__Data__Finset__Fold______macroRules____private__Mathlib__Data__Finset__Fold__0__Finset__term___x2a____1___closed__10 = (const lean_object*)&lp_mathlib___private_Mathlib_Data_Finset_Fold_0__Finset___aux__Mathlib__Data__Finset__Fold______macroRules____private__Mathlib__Data__Finset__Fold__0__Finset__term___x2a____1___closed__10_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Data_Finset_Fold_0__Finset___aux__Mathlib__Data__Finset__Fold______macroRules____private__Mathlib__Data__Finset__Fold__0__Finset__term___x2a____1___closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Data_Finset_Fold_0__Finset___aux__Mathlib__Data__Finset__Fold______macroRules____private__Mathlib__Data__Finset__Fold__0__Finset__term___x2a____1___closed__10_value),((lean_object*)&lp_mathlib___private_Mathlib_Data_Finset_Fold_0__Finset_term___x2a___00__closed__2_value),LEAN_SCALAR_PTR_LITERAL(94, 254, 163, 115, 152, 147, 149, 111)}};
static const lean_object* lp_mathlib___private_Mathlib_Data_Finset_Fold_0__Finset___aux__Mathlib__Data__Finset__Fold______macroRules____private__Mathlib__Data__Finset__Fold__0__Finset__term___x2a____1___closed__11 = (const lean_object*)&lp_mathlib___private_Mathlib_Data_Finset_Fold_0__Finset___aux__Mathlib__Data__Finset__Fold______macroRules____private__Mathlib__Data__Finset__Fold__0__Finset__term___x2a____1___closed__11_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Data_Finset_Fold_0__Finset___aux__Mathlib__Data__Finset__Fold______macroRules____private__Mathlib__Data__Finset__Fold__0__Finset__term___x2a____1___closed__12_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Data_Finset_Fold_0__Finset___aux__Mathlib__Data__Finset__Fold______macroRules____private__Mathlib__Data__Finset__Fold__0__Finset__term___x2a____1___closed__11_value),((lean_object*)&lp_mathlib___private_Mathlib_Data_Finset_Fold_0__Finset_term___x2a___00__closed__4_value),LEAN_SCALAR_PTR_LITERAL(89, 231, 153, 27, 227, 65, 114, 171)}};
static const lean_object* lp_mathlib___private_Mathlib_Data_Finset_Fold_0__Finset___aux__Mathlib__Data__Finset__Fold______macroRules____private__Mathlib__Data__Finset__Fold__0__Finset__term___x2a____1___closed__12 = (const lean_object*)&lp_mathlib___private_Mathlib_Data_Finset_Fold_0__Finset___aux__Mathlib__Data__Finset__Fold______macroRules____private__Mathlib__Data__Finset__Fold__0__Finset__term___x2a____1___closed__12_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Data_Finset_Fold_0__Finset___aux__Mathlib__Data__Finset__Fold______macroRules____private__Mathlib__Data__Finset__Fold__0__Finset__term___x2a____1___closed__13_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Data_Finset_Fold_0__Finset___aux__Mathlib__Data__Finset__Fold______macroRules____private__Mathlib__Data__Finset__Fold__0__Finset__term___x2a____1___closed__12_value),((lean_object*)&lp_mathlib___private_Mathlib_Data_Finset_Fold_0__Finset_term___x2a___00__closed__6_value),LEAN_SCALAR_PTR_LITERAL(77, 106, 74, 194, 76, 195, 158, 10)}};
static const lean_object* lp_mathlib___private_Mathlib_Data_Finset_Fold_0__Finset___aux__Mathlib__Data__Finset__Fold______macroRules____private__Mathlib__Data__Finset__Fold__0__Finset__term___x2a____1___closed__13 = (const lean_object*)&lp_mathlib___private_Mathlib_Data_Finset_Fold_0__Finset___aux__Mathlib__Data__Finset__Fold______macroRules____private__Mathlib__Data__Finset__Fold__0__Finset__term___x2a____1___closed__13_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Data_Finset_Fold_0__Finset___aux__Mathlib__Data__Finset__Fold______macroRules____private__Mathlib__Data__Finset__Fold__0__Finset__term___x2a____1___closed__14_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Data_Finset_Fold_0__Finset___aux__Mathlib__Data__Finset__Fold______macroRules____private__Mathlib__Data__Finset__Fold__0__Finset__term___x2a____1___closed__13_value),((lean_object*)&lp_mathlib___private_Mathlib_Data_Finset_Fold_0__Finset_term___x2a___00__closed__8_value),LEAN_SCALAR_PTR_LITERAL(23, 43, 212, 82, 205, 41, 12, 252)}};
static const lean_object* lp_mathlib___private_Mathlib_Data_Finset_Fold_0__Finset___aux__Mathlib__Data__Finset__Fold______macroRules____private__Mathlib__Data__Finset__Fold__0__Finset__term___x2a____1___closed__14 = (const lean_object*)&lp_mathlib___private_Mathlib_Data_Finset_Fold_0__Finset___aux__Mathlib__Data__Finset__Fold______macroRules____private__Mathlib__Data__Finset__Fold__0__Finset__term___x2a____1___closed__14_value;
static lean_once_cell_t lp_mathlib___private_Mathlib_Data_Finset_Fold_0__Finset___aux__Mathlib__Data__Finset__Fold______macroRules____private__Mathlib__Data__Finset__Fold__0__Finset__term___x2a____1___closed__15_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Data_Finset_Fold_0__Finset___aux__Mathlib__Data__Finset__Fold______macroRules____private__Mathlib__Data__Finset__Fold__0__Finset__term___x2a____1___closed__15;
static const lean_string_object lp_mathlib___private_Mathlib_Data_Finset_Fold_0__Finset___aux__Mathlib__Data__Finset__Fold______macroRules____private__Mathlib__Data__Finset__Fold__0__Finset__term___x2a____1___closed__16_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "_hygCtx"};
static const lean_object* lp_mathlib___private_Mathlib_Data_Finset_Fold_0__Finset___aux__Mathlib__Data__Finset__Fold______macroRules____private__Mathlib__Data__Finset__Fold__0__Finset__term___x2a____1___closed__16 = (const lean_object*)&lp_mathlib___private_Mathlib_Data_Finset_Fold_0__Finset___aux__Mathlib__Data__Finset__Fold______macroRules____private__Mathlib__Data__Finset__Fold__0__Finset__term___x2a____1___closed__16_value;
static lean_once_cell_t lp_mathlib___private_Mathlib_Data_Finset_Fold_0__Finset___aux__Mathlib__Data__Finset__Fold______macroRules____private__Mathlib__Data__Finset__Fold__0__Finset__term___x2a____1___closed__17_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Data_Finset_Fold_0__Finset___aux__Mathlib__Data__Finset__Fold______macroRules____private__Mathlib__Data__Finset__Fold__0__Finset__term___x2a____1___closed__17;
static const lean_string_object lp_mathlib___private_Mathlib_Data_Finset_Fold_0__Finset___aux__Mathlib__Data__Finset__Fold______macroRules____private__Mathlib__Data__Finset__Fold__0__Finset__term___x2a____1___closed__18_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "_hyg"};
static const lean_object* lp_mathlib___private_Mathlib_Data_Finset_Fold_0__Finset___aux__Mathlib__Data__Finset__Fold______macroRules____private__Mathlib__Data__Finset__Fold__0__Finset__term___x2a____1___closed__18 = (const lean_object*)&lp_mathlib___private_Mathlib_Data_Finset_Fold_0__Finset___aux__Mathlib__Data__Finset__Fold______macroRules____private__Mathlib__Data__Finset__Fold__0__Finset__term___x2a____1___closed__18_value;
static lean_once_cell_t lp_mathlib___private_Mathlib_Data_Finset_Fold_0__Finset___aux__Mathlib__Data__Finset__Fold______macroRules____private__Mathlib__Data__Finset__Fold__0__Finset__term___x2a____1___closed__19_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Data_Finset_Fold_0__Finset___aux__Mathlib__Data__Finset__Fold______macroRules____private__Mathlib__Data__Finset__Fold__0__Finset__term___x2a____1___closed__19;
static lean_once_cell_t lp_mathlib___private_Mathlib_Data_Finset_Fold_0__Finset___aux__Mathlib__Data__Finset__Fold______macroRules____private__Mathlib__Data__Finset__Fold__0__Finset__term___x2a____1___closed__20_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Data_Finset_Fold_0__Finset___aux__Mathlib__Data__Finset__Fold______macroRules____private__Mathlib__Data__Finset__Fold__0__Finset__term___x2a____1___closed__20;
static lean_once_cell_t lp_mathlib___private_Mathlib_Data_Finset_Fold_0__Finset___aux__Mathlib__Data__Finset__Fold______macroRules____private__Mathlib__Data__Finset__Fold__0__Finset__term___x2a____1___closed__21_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Data_Finset_Fold_0__Finset___aux__Mathlib__Data__Finset__Fold______macroRules____private__Mathlib__Data__Finset__Fold__0__Finset__term___x2a____1___closed__21;
static lean_once_cell_t lp_mathlib___private_Mathlib_Data_Finset_Fold_0__Finset___aux__Mathlib__Data__Finset__Fold______macroRules____private__Mathlib__Data__Finset__Fold__0__Finset__term___x2a____1___closed__22_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Data_Finset_Fold_0__Finset___aux__Mathlib__Data__Finset__Fold______macroRules____private__Mathlib__Data__Finset__Fold__0__Finset__term___x2a____1___closed__22;
static const lean_string_object lp_mathlib___private_Mathlib_Data_Finset_Fold_0__Finset___aux__Mathlib__Data__Finset__Fold______macroRules____private__Mathlib__Data__Finset__Fold__0__Finset__term___x2a____1___closed__23_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "null"};
static const lean_object* lp_mathlib___private_Mathlib_Data_Finset_Fold_0__Finset___aux__Mathlib__Data__Finset__Fold______macroRules____private__Mathlib__Data__Finset__Fold__0__Finset__term___x2a____1___closed__23 = (const lean_object*)&lp_mathlib___private_Mathlib_Data_Finset_Fold_0__Finset___aux__Mathlib__Data__Finset__Fold______macroRules____private__Mathlib__Data__Finset__Fold__0__Finset__term___x2a____1___closed__23_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Data_Finset_Fold_0__Finset___aux__Mathlib__Data__Finset__Fold______macroRules____private__Mathlib__Data__Finset__Fold__0__Finset__term___x2a____1___closed__24_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Data_Finset_Fold_0__Finset___aux__Mathlib__Data__Finset__Fold______macroRules____private__Mathlib__Data__Finset__Fold__0__Finset__term___x2a____1___closed__23_value),LEAN_SCALAR_PTR_LITERAL(24, 58, 49, 223, 146, 207, 197, 136)}};
static const lean_object* lp_mathlib___private_Mathlib_Data_Finset_Fold_0__Finset___aux__Mathlib__Data__Finset__Fold______macroRules____private__Mathlib__Data__Finset__Fold__0__Finset__term___x2a____1___closed__24 = (const lean_object*)&lp_mathlib___private_Mathlib_Data_Finset_Fold_0__Finset___aux__Mathlib__Data__Finset__Fold______macroRules____private__Mathlib__Data__Finset__Fold__0__Finset__term___x2a____1___closed__24_value;
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Data_Finset_Fold_0__Finset___aux__Mathlib__Data__Finset__Fold______macroRules____private__Mathlib__Data__Finset__Fold__0__Finset__term___x2a____1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Data_Finset_Fold_0__Finset___aux__Mathlib__Data__Finset__Fold______macroRules____private__Mathlib__Data__Finset__Fold__0__Finset__term___x2a____1___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Finset_fold___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Finset_fold(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_object* _init_lp_mathlib___private_Mathlib_Data_Finset_Fold_0__Finset___aux__Mathlib__Data__Finset__Fold______macroRules____private__Mathlib__Data__Finset__Fold__0__Finset__term___x2a____1___closed__6(void){
_start:
{
lean_object* v___x_63_; lean_object* v___x_64_; 
v___x_63_ = ((lean_object*)(lp_mathlib___private_Mathlib_Data_Finset_Fold_0__Finset___aux__Mathlib__Data__Finset__Fold______macroRules____private__Mathlib__Data__Finset__Fold__0__Finset__term___x2a____1___closed__5));
v___x_64_ = l_String_toRawSubstring_x27(v___x_63_);
return v___x_64_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Data_Finset_Fold_0__Finset___aux__Mathlib__Data__Finset__Fold______macroRules____private__Mathlib__Data__Finset__Fold__0__Finset__term___x2a____1___closed__15(void){
_start:
{
lean_object* v___x_86_; lean_object* v___x_87_; lean_object* v___x_88_; 
v___x_86_ = lean_unsigned_to_nat(3822096415u);
v___x_87_ = ((lean_object*)(lp_mathlib___private_Mathlib_Data_Finset_Fold_0__Finset___aux__Mathlib__Data__Finset__Fold______macroRules____private__Mathlib__Data__Finset__Fold__0__Finset__term___x2a____1___closed__14));
v___x_88_ = l_Lean_Name_num___override(v___x_87_, v___x_86_);
return v___x_88_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Data_Finset_Fold_0__Finset___aux__Mathlib__Data__Finset__Fold______macroRules____private__Mathlib__Data__Finset__Fold__0__Finset__term___x2a____1___closed__17(void){
_start:
{
lean_object* v___x_90_; lean_object* v___x_91_; lean_object* v___x_92_; 
v___x_90_ = ((lean_object*)(lp_mathlib___private_Mathlib_Data_Finset_Fold_0__Finset___aux__Mathlib__Data__Finset__Fold______macroRules____private__Mathlib__Data__Finset__Fold__0__Finset__term___x2a____1___closed__16));
v___x_91_ = lean_obj_once(&lp_mathlib___private_Mathlib_Data_Finset_Fold_0__Finset___aux__Mathlib__Data__Finset__Fold______macroRules____private__Mathlib__Data__Finset__Fold__0__Finset__term___x2a____1___closed__15, &lp_mathlib___private_Mathlib_Data_Finset_Fold_0__Finset___aux__Mathlib__Data__Finset__Fold______macroRules____private__Mathlib__Data__Finset__Fold__0__Finset__term___x2a____1___closed__15_once, _init_lp_mathlib___private_Mathlib_Data_Finset_Fold_0__Finset___aux__Mathlib__Data__Finset__Fold______macroRules____private__Mathlib__Data__Finset__Fold__0__Finset__term___x2a____1___closed__15);
v___x_92_ = l_Lean_Name_str___override(v___x_91_, v___x_90_);
return v___x_92_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Data_Finset_Fold_0__Finset___aux__Mathlib__Data__Finset__Fold______macroRules____private__Mathlib__Data__Finset__Fold__0__Finset__term___x2a____1___closed__19(void){
_start:
{
lean_object* v___x_94_; lean_object* v___x_95_; lean_object* v___x_96_; 
v___x_94_ = ((lean_object*)(lp_mathlib___private_Mathlib_Data_Finset_Fold_0__Finset___aux__Mathlib__Data__Finset__Fold______macroRules____private__Mathlib__Data__Finset__Fold__0__Finset__term___x2a____1___closed__18));
v___x_95_ = lean_obj_once(&lp_mathlib___private_Mathlib_Data_Finset_Fold_0__Finset___aux__Mathlib__Data__Finset__Fold______macroRules____private__Mathlib__Data__Finset__Fold__0__Finset__term___x2a____1___closed__17, &lp_mathlib___private_Mathlib_Data_Finset_Fold_0__Finset___aux__Mathlib__Data__Finset__Fold______macroRules____private__Mathlib__Data__Finset__Fold__0__Finset__term___x2a____1___closed__17_once, _init_lp_mathlib___private_Mathlib_Data_Finset_Fold_0__Finset___aux__Mathlib__Data__Finset__Fold______macroRules____private__Mathlib__Data__Finset__Fold__0__Finset__term___x2a____1___closed__17);
v___x_96_ = l_Lean_Name_str___override(v___x_95_, v___x_94_);
return v___x_96_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Data_Finset_Fold_0__Finset___aux__Mathlib__Data__Finset__Fold______macroRules____private__Mathlib__Data__Finset__Fold__0__Finset__term___x2a____1___closed__20(void){
_start:
{
lean_object* v___x_97_; lean_object* v___x_98_; lean_object* v___x_99_; 
v___x_97_ = lean_unsigned_to_nat(15u);
v___x_98_ = lean_obj_once(&lp_mathlib___private_Mathlib_Data_Finset_Fold_0__Finset___aux__Mathlib__Data__Finset__Fold______macroRules____private__Mathlib__Data__Finset__Fold__0__Finset__term___x2a____1___closed__19, &lp_mathlib___private_Mathlib_Data_Finset_Fold_0__Finset___aux__Mathlib__Data__Finset__Fold______macroRules____private__Mathlib__Data__Finset__Fold__0__Finset__term___x2a____1___closed__19_once, _init_lp_mathlib___private_Mathlib_Data_Finset_Fold_0__Finset___aux__Mathlib__Data__Finset__Fold______macroRules____private__Mathlib__Data__Finset__Fold__0__Finset__term___x2a____1___closed__19);
v___x_99_ = l_Lean_Name_num___override(v___x_98_, v___x_97_);
return v___x_99_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Data_Finset_Fold_0__Finset___aux__Mathlib__Data__Finset__Fold______macroRules____private__Mathlib__Data__Finset__Fold__0__Finset__term___x2a____1___closed__21(void){
_start:
{
lean_object* v___x_100_; lean_object* v___x_101_; lean_object* v___x_102_; 
v___x_100_ = lean_box(0);
v___x_101_ = lean_obj_once(&lp_mathlib___private_Mathlib_Data_Finset_Fold_0__Finset___aux__Mathlib__Data__Finset__Fold______macroRules____private__Mathlib__Data__Finset__Fold__0__Finset__term___x2a____1___closed__20, &lp_mathlib___private_Mathlib_Data_Finset_Fold_0__Finset___aux__Mathlib__Data__Finset__Fold______macroRules____private__Mathlib__Data__Finset__Fold__0__Finset__term___x2a____1___closed__20_once, _init_lp_mathlib___private_Mathlib_Data_Finset_Fold_0__Finset___aux__Mathlib__Data__Finset__Fold______macroRules____private__Mathlib__Data__Finset__Fold__0__Finset__term___x2a____1___closed__20);
v___x_102_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_102_, 0, v___x_101_);
lean_ctor_set(v___x_102_, 1, v___x_100_);
return v___x_102_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Data_Finset_Fold_0__Finset___aux__Mathlib__Data__Finset__Fold______macroRules____private__Mathlib__Data__Finset__Fold__0__Finset__term___x2a____1___closed__22(void){
_start:
{
lean_object* v___x_103_; lean_object* v___x_104_; lean_object* v___x_105_; 
v___x_103_ = lean_box(0);
v___x_104_ = lean_obj_once(&lp_mathlib___private_Mathlib_Data_Finset_Fold_0__Finset___aux__Mathlib__Data__Finset__Fold______macroRules____private__Mathlib__Data__Finset__Fold__0__Finset__term___x2a____1___closed__21, &lp_mathlib___private_Mathlib_Data_Finset_Fold_0__Finset___aux__Mathlib__Data__Finset__Fold______macroRules____private__Mathlib__Data__Finset__Fold__0__Finset__term___x2a____1___closed__21_once, _init_lp_mathlib___private_Mathlib_Data_Finset_Fold_0__Finset___aux__Mathlib__Data__Finset__Fold______macroRules____private__Mathlib__Data__Finset__Fold__0__Finset__term___x2a____1___closed__21);
v___x_105_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_105_, 0, v___x_104_);
lean_ctor_set(v___x_105_, 1, v___x_103_);
return v___x_105_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Data_Finset_Fold_0__Finset___aux__Mathlib__Data__Finset__Fold______macroRules____private__Mathlib__Data__Finset__Fold__0__Finset__term___x2a____1(lean_object* v_x_109_, lean_object* v_a_110_, lean_object* v_a_111_){
_start:
{
lean_object* v___x_112_; lean_object* v___x_113_; uint8_t v___x_114_; 
v___x_112_ = lean_unsigned_to_nat(0u);
v___x_113_ = ((lean_object*)(lp_mathlib___private_Mathlib_Data_Finset_Fold_0__Finset_term___x2a___00__closed__13));
lean_inc(v_x_109_);
v___x_114_ = l_Lean_Syntax_isOfKind(v_x_109_, v___x_113_);
if (v___x_114_ == 0)
{
lean_object* v___x_115_; lean_object* v___x_116_; 
lean_dec(v_x_109_);
v___x_115_ = lean_box(1);
v___x_116_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_116_, 0, v___x_115_);
lean_ctor_set(v___x_116_, 1, v_a_111_);
return v___x_116_;
}
else
{
lean_object* v_quotContext_117_; lean_object* v_currMacroScope_118_; lean_object* v_ref_119_; lean_object* v___x_120_; lean_object* v___x_121_; lean_object* v___x_122_; uint8_t v___x_123_; lean_object* v___x_124_; lean_object* v___x_125_; lean_object* v___x_126_; lean_object* v___x_127_; lean_object* v___x_128_; lean_object* v___x_129_; lean_object* v___x_130_; lean_object* v___x_131_; lean_object* v___x_132_; lean_object* v___x_133_; lean_object* v___x_134_; 
v_quotContext_117_ = lean_ctor_get(v_a_110_, 1);
v_currMacroScope_118_ = lean_ctor_get(v_a_110_, 2);
v_ref_119_ = lean_ctor_get(v_a_110_, 5);
v___x_120_ = l_Lean_Syntax_getArg(v_x_109_, v___x_112_);
v___x_121_ = lean_unsigned_to_nat(2u);
v___x_122_ = l_Lean_Syntax_getArg(v_x_109_, v___x_121_);
lean_dec(v_x_109_);
v___x_123_ = 0;
v___x_124_ = l_Lean_SourceInfo_fromRef(v_ref_119_, v___x_123_);
v___x_125_ = ((lean_object*)(lp_mathlib___private_Mathlib_Data_Finset_Fold_0__Finset___aux__Mathlib__Data__Finset__Fold______macroRules____private__Mathlib__Data__Finset__Fold__0__Finset__term___x2a____1___closed__4));
v___x_126_ = lean_obj_once(&lp_mathlib___private_Mathlib_Data_Finset_Fold_0__Finset___aux__Mathlib__Data__Finset__Fold______macroRules____private__Mathlib__Data__Finset__Fold__0__Finset__term___x2a____1___closed__6, &lp_mathlib___private_Mathlib_Data_Finset_Fold_0__Finset___aux__Mathlib__Data__Finset__Fold______macroRules____private__Mathlib__Data__Finset__Fold__0__Finset__term___x2a____1___closed__6_once, _init_lp_mathlib___private_Mathlib_Data_Finset_Fold_0__Finset___aux__Mathlib__Data__Finset__Fold______macroRules____private__Mathlib__Data__Finset__Fold__0__Finset__term___x2a____1___closed__6);
v___x_127_ = ((lean_object*)(lp_mathlib___private_Mathlib_Data_Finset_Fold_0__Finset___aux__Mathlib__Data__Finset__Fold______macroRules____private__Mathlib__Data__Finset__Fold__0__Finset__term___x2a____1___closed__7));
lean_inc(v_currMacroScope_118_);
lean_inc(v_quotContext_117_);
v___x_128_ = l_Lean_addMacroScope(v_quotContext_117_, v___x_127_, v_currMacroScope_118_);
v___x_129_ = lean_obj_once(&lp_mathlib___private_Mathlib_Data_Finset_Fold_0__Finset___aux__Mathlib__Data__Finset__Fold______macroRules____private__Mathlib__Data__Finset__Fold__0__Finset__term___x2a____1___closed__22, &lp_mathlib___private_Mathlib_Data_Finset_Fold_0__Finset___aux__Mathlib__Data__Finset__Fold______macroRules____private__Mathlib__Data__Finset__Fold__0__Finset__term___x2a____1___closed__22_once, _init_lp_mathlib___private_Mathlib_Data_Finset_Fold_0__Finset___aux__Mathlib__Data__Finset__Fold______macroRules____private__Mathlib__Data__Finset__Fold__0__Finset__term___x2a____1___closed__22);
lean_inc_n(v___x_124_, 2);
v___x_130_ = lean_alloc_ctor(3, 4, 0);
lean_ctor_set(v___x_130_, 0, v___x_124_);
lean_ctor_set(v___x_130_, 1, v___x_126_);
lean_ctor_set(v___x_130_, 2, v___x_128_);
lean_ctor_set(v___x_130_, 3, v___x_129_);
v___x_131_ = ((lean_object*)(lp_mathlib___private_Mathlib_Data_Finset_Fold_0__Finset___aux__Mathlib__Data__Finset__Fold______macroRules____private__Mathlib__Data__Finset__Fold__0__Finset__term___x2a____1___closed__24));
v___x_132_ = l_Lean_Syntax_node2(v___x_124_, v___x_131_, v___x_120_, v___x_122_);
v___x_133_ = l_Lean_Syntax_node2(v___x_124_, v___x_125_, v___x_130_, v___x_132_);
v___x_134_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_134_, 0, v___x_133_);
lean_ctor_set(v___x_134_, 1, v_a_111_);
return v___x_134_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Data_Finset_Fold_0__Finset___aux__Mathlib__Data__Finset__Fold______macroRules____private__Mathlib__Data__Finset__Fold__0__Finset__term___x2a____1___boxed(lean_object* v_x_135_, lean_object* v_a_136_, lean_object* v_a_137_){
_start:
{
lean_object* v_res_138_; 
v_res_138_ = lp_mathlib___private_Mathlib_Data_Finset_Fold_0__Finset___aux__Mathlib__Data__Finset__Fold______macroRules____private__Mathlib__Data__Finset__Fold__0__Finset__term___x2a____1(v_x_135_, v_a_136_, v_a_137_);
lean_dec_ref(v_a_136_);
return v_res_138_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Finset_fold___redArg(lean_object* v_op_139_, lean_object* v_b_140_, lean_object* v_f_141_, lean_object* v_s_142_){
_start:
{
lean_object* v___x_143_; lean_object* v___x_144_; 
v___x_143_ = lp_mathlib_Multiset_map___redArg(v_f_141_, v_s_142_);
v___x_144_ = l_List_foldrTR___redArg(v_op_139_, v_b_140_, v___x_143_);
return v___x_144_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Finset_fold(lean_object* v_00_u03b1_145_, lean_object* v_00_u03b2_146_, lean_object* v_op_147_, lean_object* v_hc_148_, lean_object* v_ha_149_, lean_object* v_b_150_, lean_object* v_f_151_, lean_object* v_s_152_){
_start:
{
lean_object* v___x_153_; 
v___x_153_ = lp_mathlib_Finset_fold___redArg(v_op_147_, v_b_150_, v_f_151_, v_s_152_);
return v___x_153_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Data_Finset_Basic(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Data_Finset_Image(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Data_Multiset_Fold(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Data_Finset_Lattice_Lemmas(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_Data_Finset_Fold(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Data_Finset_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Data_Finset_Image(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Data_Multiset_Fold(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Data_Finset_Lattice_Lemmas(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_Data_Finset_Fold(uint8_t builtin) {
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
lean_object* initialize_mathlib_Mathlib_Data_Finset_Basic(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Data_Finset_Image(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Data_Multiset_Fold(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Data_Finset_Lattice_Lemmas(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Data_Finset_Fold(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_Finset_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_Finset_Image(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_Multiset_Fold(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_Finset_Lattice_Lemmas(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Data_Finset_Fold(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_Data_Finset_Fold(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_Data_Finset_Fold(builtin);
}
#ifdef __cplusplus
}
#endif
