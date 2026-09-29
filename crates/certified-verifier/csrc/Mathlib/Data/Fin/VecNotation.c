// Lean compiler output
// Module: Mathlib.Data.Fin.VecNotation
// Imports: public import Init public meta import Init public import Mathlib.Data.Fin.Tuple.Basic import Mathlib.Data.Set.Image
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
lean_object* lean_nat_add(lean_object*, lean_object*);
lean_object* l_Lean_Name_mkStr2(lean_object*, lean_object*);
lean_object* l_Lean_Expr_const___override(lean_object*, lean_object*);
lean_object* l_Lean_Expr_app___override(lean_object*, lean_object*);
uint8_t lean_nat_dec_lt(lean_object*, lean_object*);
lean_object* lean_nat_sub(lean_object*, lean_object*);
lean_object* l_Lean_mkNatLit(lean_object*);
lean_object* l_Lean_Name_mkStr1(lean_object*);
lean_object* l_Lean_Expr_lit___override(lean_object*);
lean_object* l_Fin_succ___redArg(lean_object*);
lean_object* l_Fin_addCases___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_List_finRange(lean_object*);
lean_object* l_List_mapTR_loop___redArg(lean_object*, lean_object*, lean_object*);
lean_object* l_Std_Format_joinSep___redArg(lean_object*, lean_object*, lean_object*);
lean_object* lean_string_length(lean_object*);
lean_object* lean_nat_to_int(lean_object*);
lean_object* l_Lean_Meta_whnfR(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Expr_cleanupAnnotations(lean_object*);
uint8_t l_Lean_Expr_isApp(lean_object*);
lean_object* l_Lean_Expr_appFnCleanup___redArg(lean_object*);
uint8_t l_Lean_Expr_isConstOf(lean_object*, lean_object*);
lean_object* l_Lean_Expr_int_x3f(lean_object*);
lean_object* l_Lean_Meta_instantiateMVarsIfMVarApp___redArg(lean_object*, lean_object*);
lean_object* l_Lean_Meta_whnfD(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_List_lengthTR___redArg(lean_object*);
lean_object* lp_Qq_Qq_synthInstanceQ___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_mkRawNatLit(lean_object*);
lean_object* l_List_get___redArg(lean_object*, lean_object*);
lean_object* lean_int_add(lean_object*, lean_object*);
lean_object* lean_int_emod(lean_object*, lean_object*);
lean_object* l_Int_toNat(lean_object*);
uint8_t lean_int_dec_le(lean_object*, lean_object*);
uint8_t lean_int_dec_lt(lean_object*, lean_object*);
lean_object* l_Lean_Meta_isOffset_x3f(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lean_nat_mod(lean_object*, lean_object*);
lean_object* l_Fin_cases___redArg(lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Name_mkStr4(lean_object*, lean_object*, lean_object*, lean_object*);
uint8_t l_Lean_Syntax_isOfKind(lean_object*, lean_object*);
lean_object* l_Lean_Syntax_getArg(lean_object*, lean_object*);
uint8_t l_Lean_Syntax_matchesNull(lean_object*, lean_object*);
lean_object* l_Lean_Syntax_getNumArgs(lean_object*);
uint8_t lean_nat_dec_le(lean_object*, lean_object*);
lean_object* l_Lean_SourceInfo_fromRef(lean_object*, uint8_t);
lean_object* l_Lean_Syntax_node1(lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Syntax_node3(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Syntax_getArgs(lean_object*);
lean_object* l_Array_extract___redArg(lean_object*, lean_object*, lean_object*);
lean_object* l_Array_mkArray4___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Array_append___redArg(lean_object*, lean_object*);
lean_object* l_Std_instToFormatFormat___lam__0___boxed(lean_object*);
lean_object* l_String_toRawSubstring_x27(lean_object*);
lean_object* l_Lean_addMacroScope(lean_object*, lean_object*, lean_object*);
lean_object* l_Array_mkArray0(lean_object*);
lean_object* l_Lean_Syntax_node2(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Expr_forallE___override(lean_object*, lean_object*, lean_object*, uint8_t);
LEAN_EXPORT lean_object* lp_mathlib_Matrix_vecEmpty(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Matrix_vecEmpty___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Matrix_vecCons___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Matrix_vecCons___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Matrix_vecCons(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Matrix_vecCons___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Matrix_vecNotation___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "Matrix"};
static const lean_object* lp_mathlib_Matrix_vecNotation___closed__0 = (const lean_object*)&lp_mathlib_Matrix_vecNotation___closed__0_value;
static const lean_string_object lp_mathlib_Matrix_vecNotation___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 12, .m_capacity = 12, .m_length = 11, .m_data = "vecNotation"};
static const lean_object* lp_mathlib_Matrix_vecNotation___closed__1 = (const lean_object*)&lp_mathlib_Matrix_vecNotation___closed__1_value;
static const lean_ctor_object lp_mathlib_Matrix_vecNotation___closed__2_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Matrix_vecNotation___closed__0_value),LEAN_SCALAR_PTR_LITERAL(214, 58, 148, 51, 223, 251, 16, 41)}};
static const lean_ctor_object lp_mathlib_Matrix_vecNotation___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Matrix_vecNotation___closed__2_value_aux_0),((lean_object*)&lp_mathlib_Matrix_vecNotation___closed__1_value),LEAN_SCALAR_PTR_LITERAL(88, 205, 234, 165, 252, 156, 117, 174)}};
static const lean_object* lp_mathlib_Matrix_vecNotation___closed__2 = (const lean_object*)&lp_mathlib_Matrix_vecNotation___closed__2_value;
static const lean_string_object lp_mathlib_Matrix_vecNotation___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "andthen"};
static const lean_object* lp_mathlib_Matrix_vecNotation___closed__3 = (const lean_object*)&lp_mathlib_Matrix_vecNotation___closed__3_value;
static const lean_ctor_object lp_mathlib_Matrix_vecNotation___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Matrix_vecNotation___closed__3_value),LEAN_SCALAR_PTR_LITERAL(40, 255, 78, 30, 143, 119, 117, 174)}};
static const lean_object* lp_mathlib_Matrix_vecNotation___closed__4 = (const lean_object*)&lp_mathlib_Matrix_vecNotation___closed__4_value;
static const lean_string_object lp_mathlib_Matrix_vecNotation___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 3, .m_capacity = 3, .m_length = 2, .m_data = "!["};
static const lean_object* lp_mathlib_Matrix_vecNotation___closed__5 = (const lean_object*)&lp_mathlib_Matrix_vecNotation___closed__5_value;
static const lean_ctor_object lp_mathlib_Matrix_vecNotation___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 5}, .m_objs = {((lean_object*)&lp_mathlib_Matrix_vecNotation___closed__5_value)}};
static const lean_object* lp_mathlib_Matrix_vecNotation___closed__6 = (const lean_object*)&lp_mathlib_Matrix_vecNotation___closed__6_value;
static const lean_string_object lp_mathlib_Matrix_vecNotation___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "term"};
static const lean_object* lp_mathlib_Matrix_vecNotation___closed__7 = (const lean_object*)&lp_mathlib_Matrix_vecNotation___closed__7_value;
static const lean_ctor_object lp_mathlib_Matrix_vecNotation___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Matrix_vecNotation___closed__7_value),LEAN_SCALAR_PTR_LITERAL(187, 230, 181, 162, 253, 146, 122, 119)}};
static const lean_object* lp_mathlib_Matrix_vecNotation___closed__8 = (const lean_object*)&lp_mathlib_Matrix_vecNotation___closed__8_value;
static const lean_ctor_object lp_mathlib_Matrix_vecNotation___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 7}, .m_objs = {((lean_object*)&lp_mathlib_Matrix_vecNotation___closed__8_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_Matrix_vecNotation___closed__9 = (const lean_object*)&lp_mathlib_Matrix_vecNotation___closed__9_value;
static const lean_string_object lp_mathlib_Matrix_vecNotation___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = ","};
static const lean_object* lp_mathlib_Matrix_vecNotation___closed__10 = (const lean_object*)&lp_mathlib_Matrix_vecNotation___closed__10_value;
static const lean_string_object lp_mathlib_Matrix_vecNotation___closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 3, .m_capacity = 3, .m_length = 2, .m_data = ", "};
static const lean_object* lp_mathlib_Matrix_vecNotation___closed__11 = (const lean_object*)&lp_mathlib_Matrix_vecNotation___closed__11_value;
static const lean_ctor_object lp_mathlib_Matrix_vecNotation___closed__12_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 5}, .m_objs = {((lean_object*)&lp_mathlib_Matrix_vecNotation___closed__11_value)}};
static const lean_object* lp_mathlib_Matrix_vecNotation___closed__12 = (const lean_object*)&lp_mathlib_Matrix_vecNotation___closed__12_value;
static const lean_ctor_object lp_mathlib_Matrix_vecNotation___closed__13_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 8, .m_other = 3, .m_tag = 10}, .m_objs = {((lean_object*)&lp_mathlib_Matrix_vecNotation___closed__9_value),((lean_object*)&lp_mathlib_Matrix_vecNotation___closed__10_value),((lean_object*)&lp_mathlib_Matrix_vecNotation___closed__12_value),LEAN_SCALAR_PTR_LITERAL(0, 0, 0, 0, 0, 0, 0, 0)}};
static const lean_object* lp_mathlib_Matrix_vecNotation___closed__13 = (const lean_object*)&lp_mathlib_Matrix_vecNotation___closed__13_value;
static const lean_ctor_object lp_mathlib_Matrix_vecNotation___closed__14_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib_Matrix_vecNotation___closed__4_value),((lean_object*)&lp_mathlib_Matrix_vecNotation___closed__6_value),((lean_object*)&lp_mathlib_Matrix_vecNotation___closed__13_value)}};
static const lean_object* lp_mathlib_Matrix_vecNotation___closed__14 = (const lean_object*)&lp_mathlib_Matrix_vecNotation___closed__14_value;
static const lean_string_object lp_mathlib_Matrix_vecNotation___closed__15_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = "]"};
static const lean_object* lp_mathlib_Matrix_vecNotation___closed__15 = (const lean_object*)&lp_mathlib_Matrix_vecNotation___closed__15_value;
static const lean_ctor_object lp_mathlib_Matrix_vecNotation___closed__16_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 5}, .m_objs = {((lean_object*)&lp_mathlib_Matrix_vecNotation___closed__15_value)}};
static const lean_object* lp_mathlib_Matrix_vecNotation___closed__16 = (const lean_object*)&lp_mathlib_Matrix_vecNotation___closed__16_value;
static const lean_ctor_object lp_mathlib_Matrix_vecNotation___closed__17_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib_Matrix_vecNotation___closed__4_value),((lean_object*)&lp_mathlib_Matrix_vecNotation___closed__14_value),((lean_object*)&lp_mathlib_Matrix_vecNotation___closed__16_value)}};
static const lean_object* lp_mathlib_Matrix_vecNotation___closed__17 = (const lean_object*)&lp_mathlib_Matrix_vecNotation___closed__17_value;
static const lean_ctor_object lp_mathlib_Matrix_vecNotation___closed__18_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 3}, .m_objs = {((lean_object*)&lp_mathlib_Matrix_vecNotation___closed__2_value),((lean_object*)(((size_t)(1024) << 1) | 1)),((lean_object*)&lp_mathlib_Matrix_vecNotation___closed__17_value)}};
static const lean_object* lp_mathlib_Matrix_vecNotation___closed__18 = (const lean_object*)&lp_mathlib_Matrix_vecNotation___closed__18_value;
LEAN_EXPORT const lean_object* lp_mathlib_Matrix_vecNotation = (const lean_object*)&lp_mathlib_Matrix_vecNotation___closed__18_value;
static const lean_string_object lp_mathlib_Matrix___aux__Mathlib__Data__Fin__VecNotation______macroRules__Matrix__vecNotation__1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "vecEmpty"};
static const lean_object* lp_mathlib_Matrix___aux__Mathlib__Data__Fin__VecNotation______macroRules__Matrix__vecNotation__1___closed__0 = (const lean_object*)&lp_mathlib_Matrix___aux__Mathlib__Data__Fin__VecNotation______macroRules__Matrix__vecNotation__1___closed__0_value;
static lean_once_cell_t lp_mathlib_Matrix___aux__Mathlib__Data__Fin__VecNotation______macroRules__Matrix__vecNotation__1___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Matrix___aux__Mathlib__Data__Fin__VecNotation______macroRules__Matrix__vecNotation__1___closed__1;
static const lean_ctor_object lp_mathlib_Matrix___aux__Mathlib__Data__Fin__VecNotation______macroRules__Matrix__vecNotation__1___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Matrix___aux__Mathlib__Data__Fin__VecNotation______macroRules__Matrix__vecNotation__1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(164, 150, 255, 150, 173, 14, 60, 155)}};
static const lean_object* lp_mathlib_Matrix___aux__Mathlib__Data__Fin__VecNotation______macroRules__Matrix__vecNotation__1___closed__2 = (const lean_object*)&lp_mathlib_Matrix___aux__Mathlib__Data__Fin__VecNotation______macroRules__Matrix__vecNotation__1___closed__2_value;
static const lean_ctor_object lp_mathlib_Matrix___aux__Mathlib__Data__Fin__VecNotation______macroRules__Matrix__vecNotation__1___closed__3_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Matrix_vecNotation___closed__0_value),LEAN_SCALAR_PTR_LITERAL(214, 58, 148, 51, 223, 251, 16, 41)}};
static const lean_ctor_object lp_mathlib_Matrix___aux__Mathlib__Data__Fin__VecNotation______macroRules__Matrix__vecNotation__1___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Matrix___aux__Mathlib__Data__Fin__VecNotation______macroRules__Matrix__vecNotation__1___closed__3_value_aux_0),((lean_object*)&lp_mathlib_Matrix___aux__Mathlib__Data__Fin__VecNotation______macroRules__Matrix__vecNotation__1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(109, 122, 58, 8, 245, 198, 21, 173)}};
static const lean_object* lp_mathlib_Matrix___aux__Mathlib__Data__Fin__VecNotation______macroRules__Matrix__vecNotation__1___closed__3 = (const lean_object*)&lp_mathlib_Matrix___aux__Mathlib__Data__Fin__VecNotation______macroRules__Matrix__vecNotation__1___closed__3_value;
static const lean_ctor_object lp_mathlib_Matrix___aux__Mathlib__Data__Fin__VecNotation______macroRules__Matrix__vecNotation__1___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Matrix___aux__Mathlib__Data__Fin__VecNotation______macroRules__Matrix__vecNotation__1___closed__3_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_Matrix___aux__Mathlib__Data__Fin__VecNotation______macroRules__Matrix__vecNotation__1___closed__4 = (const lean_object*)&lp_mathlib_Matrix___aux__Mathlib__Data__Fin__VecNotation______macroRules__Matrix__vecNotation__1___closed__4_value;
static const lean_ctor_object lp_mathlib_Matrix___aux__Mathlib__Data__Fin__VecNotation______macroRules__Matrix__vecNotation__1___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Matrix___aux__Mathlib__Data__Fin__VecNotation______macroRules__Matrix__vecNotation__1___closed__4_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_Matrix___aux__Mathlib__Data__Fin__VecNotation______macroRules__Matrix__vecNotation__1___closed__5 = (const lean_object*)&lp_mathlib_Matrix___aux__Mathlib__Data__Fin__VecNotation______macroRules__Matrix__vecNotation__1___closed__5_value;
static const lean_string_object lp_mathlib_Matrix___aux__Mathlib__Data__Fin__VecNotation______macroRules__Matrix__vecNotation__1___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "Lean"};
static const lean_object* lp_mathlib_Matrix___aux__Mathlib__Data__Fin__VecNotation______macroRules__Matrix__vecNotation__1___closed__6 = (const lean_object*)&lp_mathlib_Matrix___aux__Mathlib__Data__Fin__VecNotation______macroRules__Matrix__vecNotation__1___closed__6_value;
static const lean_string_object lp_mathlib_Matrix___aux__Mathlib__Data__Fin__VecNotation______macroRules__Matrix__vecNotation__1___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "Parser"};
static const lean_object* lp_mathlib_Matrix___aux__Mathlib__Data__Fin__VecNotation______macroRules__Matrix__vecNotation__1___closed__7 = (const lean_object*)&lp_mathlib_Matrix___aux__Mathlib__Data__Fin__VecNotation______macroRules__Matrix__vecNotation__1___closed__7_value;
static const lean_string_object lp_mathlib_Matrix___aux__Mathlib__Data__Fin__VecNotation______macroRules__Matrix__vecNotation__1___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "Term"};
static const lean_object* lp_mathlib_Matrix___aux__Mathlib__Data__Fin__VecNotation______macroRules__Matrix__vecNotation__1___closed__8 = (const lean_object*)&lp_mathlib_Matrix___aux__Mathlib__Data__Fin__VecNotation______macroRules__Matrix__vecNotation__1___closed__8_value;
static const lean_string_object lp_mathlib_Matrix___aux__Mathlib__Data__Fin__VecNotation______macroRules__Matrix__vecNotation__1___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = "app"};
static const lean_object* lp_mathlib_Matrix___aux__Mathlib__Data__Fin__VecNotation______macroRules__Matrix__vecNotation__1___closed__9 = (const lean_object*)&lp_mathlib_Matrix___aux__Mathlib__Data__Fin__VecNotation______macroRules__Matrix__vecNotation__1___closed__9_value;
static const lean_ctor_object lp_mathlib_Matrix___aux__Mathlib__Data__Fin__VecNotation______macroRules__Matrix__vecNotation__1___closed__10_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Matrix___aux__Mathlib__Data__Fin__VecNotation______macroRules__Matrix__vecNotation__1___closed__6_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib_Matrix___aux__Mathlib__Data__Fin__VecNotation______macroRules__Matrix__vecNotation__1___closed__10_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Matrix___aux__Mathlib__Data__Fin__VecNotation______macroRules__Matrix__vecNotation__1___closed__10_value_aux_0),((lean_object*)&lp_mathlib_Matrix___aux__Mathlib__Data__Fin__VecNotation______macroRules__Matrix__vecNotation__1___closed__7_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib_Matrix___aux__Mathlib__Data__Fin__VecNotation______macroRules__Matrix__vecNotation__1___closed__10_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Matrix___aux__Mathlib__Data__Fin__VecNotation______macroRules__Matrix__vecNotation__1___closed__10_value_aux_1),((lean_object*)&lp_mathlib_Matrix___aux__Mathlib__Data__Fin__VecNotation______macroRules__Matrix__vecNotation__1___closed__8_value),LEAN_SCALAR_PTR_LITERAL(75, 170, 162, 138, 136, 204, 251, 229)}};
static const lean_ctor_object lp_mathlib_Matrix___aux__Mathlib__Data__Fin__VecNotation______macroRules__Matrix__vecNotation__1___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Matrix___aux__Mathlib__Data__Fin__VecNotation______macroRules__Matrix__vecNotation__1___closed__10_value_aux_2),((lean_object*)&lp_mathlib_Matrix___aux__Mathlib__Data__Fin__VecNotation______macroRules__Matrix__vecNotation__1___closed__9_value),LEAN_SCALAR_PTR_LITERAL(69, 118, 10, 41, 220, 156, 243, 179)}};
static const lean_object* lp_mathlib_Matrix___aux__Mathlib__Data__Fin__VecNotation______macroRules__Matrix__vecNotation__1___closed__10 = (const lean_object*)&lp_mathlib_Matrix___aux__Mathlib__Data__Fin__VecNotation______macroRules__Matrix__vecNotation__1___closed__10_value;
static const lean_string_object lp_mathlib_Matrix___aux__Mathlib__Data__Fin__VecNotation______macroRules__Matrix__vecNotation__1___closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "vecCons"};
static const lean_object* lp_mathlib_Matrix___aux__Mathlib__Data__Fin__VecNotation______macroRules__Matrix__vecNotation__1___closed__11 = (const lean_object*)&lp_mathlib_Matrix___aux__Mathlib__Data__Fin__VecNotation______macroRules__Matrix__vecNotation__1___closed__11_value;
static lean_once_cell_t lp_mathlib_Matrix___aux__Mathlib__Data__Fin__VecNotation______macroRules__Matrix__vecNotation__1___closed__12_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Matrix___aux__Mathlib__Data__Fin__VecNotation______macroRules__Matrix__vecNotation__1___closed__12;
static const lean_ctor_object lp_mathlib_Matrix___aux__Mathlib__Data__Fin__VecNotation______macroRules__Matrix__vecNotation__1___closed__13_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Matrix___aux__Mathlib__Data__Fin__VecNotation______macroRules__Matrix__vecNotation__1___closed__11_value),LEAN_SCALAR_PTR_LITERAL(171, 67, 145, 46, 14, 68, 248, 224)}};
static const lean_object* lp_mathlib_Matrix___aux__Mathlib__Data__Fin__VecNotation______macroRules__Matrix__vecNotation__1___closed__13 = (const lean_object*)&lp_mathlib_Matrix___aux__Mathlib__Data__Fin__VecNotation______macroRules__Matrix__vecNotation__1___closed__13_value;
static const lean_ctor_object lp_mathlib_Matrix___aux__Mathlib__Data__Fin__VecNotation______macroRules__Matrix__vecNotation__1___closed__14_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Matrix_vecNotation___closed__0_value),LEAN_SCALAR_PTR_LITERAL(214, 58, 148, 51, 223, 251, 16, 41)}};
static const lean_ctor_object lp_mathlib_Matrix___aux__Mathlib__Data__Fin__VecNotation______macroRules__Matrix__vecNotation__1___closed__14_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Matrix___aux__Mathlib__Data__Fin__VecNotation______macroRules__Matrix__vecNotation__1___closed__14_value_aux_0),((lean_object*)&lp_mathlib_Matrix___aux__Mathlib__Data__Fin__VecNotation______macroRules__Matrix__vecNotation__1___closed__11_value),LEAN_SCALAR_PTR_LITERAL(186, 245, 9, 82, 94, 240, 46, 227)}};
static const lean_object* lp_mathlib_Matrix___aux__Mathlib__Data__Fin__VecNotation______macroRules__Matrix__vecNotation__1___closed__14 = (const lean_object*)&lp_mathlib_Matrix___aux__Mathlib__Data__Fin__VecNotation______macroRules__Matrix__vecNotation__1___closed__14_value;
static const lean_ctor_object lp_mathlib_Matrix___aux__Mathlib__Data__Fin__VecNotation______macroRules__Matrix__vecNotation__1___closed__15_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Matrix___aux__Mathlib__Data__Fin__VecNotation______macroRules__Matrix__vecNotation__1___closed__14_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_Matrix___aux__Mathlib__Data__Fin__VecNotation______macroRules__Matrix__vecNotation__1___closed__15 = (const lean_object*)&lp_mathlib_Matrix___aux__Mathlib__Data__Fin__VecNotation______macroRules__Matrix__vecNotation__1___closed__15_value;
static const lean_ctor_object lp_mathlib_Matrix___aux__Mathlib__Data__Fin__VecNotation______macroRules__Matrix__vecNotation__1___closed__16_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Matrix___aux__Mathlib__Data__Fin__VecNotation______macroRules__Matrix__vecNotation__1___closed__15_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_Matrix___aux__Mathlib__Data__Fin__VecNotation______macroRules__Matrix__vecNotation__1___closed__16 = (const lean_object*)&lp_mathlib_Matrix___aux__Mathlib__Data__Fin__VecNotation______macroRules__Matrix__vecNotation__1___closed__16_value;
static const lean_string_object lp_mathlib_Matrix___aux__Mathlib__Data__Fin__VecNotation______macroRules__Matrix__vecNotation__1___closed__17_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "null"};
static const lean_object* lp_mathlib_Matrix___aux__Mathlib__Data__Fin__VecNotation______macroRules__Matrix__vecNotation__1___closed__17 = (const lean_object*)&lp_mathlib_Matrix___aux__Mathlib__Data__Fin__VecNotation______macroRules__Matrix__vecNotation__1___closed__17_value;
static const lean_ctor_object lp_mathlib_Matrix___aux__Mathlib__Data__Fin__VecNotation______macroRules__Matrix__vecNotation__1___closed__18_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Matrix___aux__Mathlib__Data__Fin__VecNotation______macroRules__Matrix__vecNotation__1___closed__17_value),LEAN_SCALAR_PTR_LITERAL(24, 58, 49, 223, 146, 207, 197, 136)}};
static const lean_object* lp_mathlib_Matrix___aux__Mathlib__Data__Fin__VecNotation______macroRules__Matrix__vecNotation__1___closed__18 = (const lean_object*)&lp_mathlib_Matrix___aux__Mathlib__Data__Fin__VecNotation______macroRules__Matrix__vecNotation__1___closed__18_value;
static lean_once_cell_t lp_mathlib_Matrix___aux__Mathlib__Data__Fin__VecNotation______macroRules__Matrix__vecNotation__1___closed__19_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Matrix___aux__Mathlib__Data__Fin__VecNotation______macroRules__Matrix__vecNotation__1___closed__19;
LEAN_EXPORT lean_object* lp_mathlib_Matrix___aux__Mathlib__Data__Fin__VecNotation______macroRules__Matrix__vecNotation__1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Matrix___aux__Mathlib__Data__Fin__VecNotation______macroRules__Matrix__vecNotation__1___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Matrix_vecConsUnexpander(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Matrix_vecConsUnexpander___boxed(lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Matrix_vecEmptyUnexpander___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "ident"};
static const lean_object* lp_mathlib_Matrix_vecEmptyUnexpander___closed__0 = (const lean_object*)&lp_mathlib_Matrix_vecEmptyUnexpander___closed__0_value;
static const lean_ctor_object lp_mathlib_Matrix_vecEmptyUnexpander___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Matrix_vecEmptyUnexpander___closed__0_value),LEAN_SCALAR_PTR_LITERAL(52, 159, 208, 51, 14, 60, 6, 71)}};
static const lean_object* lp_mathlib_Matrix_vecEmptyUnexpander___closed__1 = (const lean_object*)&lp_mathlib_Matrix_vecEmptyUnexpander___closed__1_value;
LEAN_EXPORT lean_object* lp_mathlib_Matrix_vecEmptyUnexpander(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Matrix_vecEmptyUnexpander___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Matrix_vecHead___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Matrix_vecHead___redArg___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Matrix_vecHead(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Matrix_vecHead___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Matrix_vecTail___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Matrix_vecTail___redArg___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Matrix_vecTail(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Matrix_vecTail___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_PiFin_hasRepr___redArg___lam__0(lean_object*, lean_object*, lean_object*);
static const lean_ctor_object lp_mathlib_PiFin_hasRepr___redArg___lam__1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_mathlib_Matrix_vecNotation___closed__10_value)}};
static const lean_object* lp_mathlib_PiFin_hasRepr___redArg___lam__1___closed__0 = (const lean_object*)&lp_mathlib_PiFin_hasRepr___redArg___lam__1___closed__0_value;
static const lean_ctor_object lp_mathlib_PiFin_hasRepr___redArg___lam__1___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 5}, .m_objs = {((lean_object*)&lp_mathlib_PiFin_hasRepr___redArg___lam__1___closed__0_value),((lean_object*)(((size_t)(1) << 1) | 1))}};
static const lean_object* lp_mathlib_PiFin_hasRepr___redArg___lam__1___closed__1 = (const lean_object*)&lp_mathlib_PiFin_hasRepr___redArg___lam__1___closed__1_value;
static lean_once_cell_t lp_mathlib_PiFin_hasRepr___redArg___lam__1___closed__2_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_PiFin_hasRepr___redArg___lam__1___closed__2;
static lean_once_cell_t lp_mathlib_PiFin_hasRepr___redArg___lam__1___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_PiFin_hasRepr___redArg___lam__1___closed__3;
static const lean_ctor_object lp_mathlib_PiFin_hasRepr___redArg___lam__1___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_mathlib_Matrix_vecNotation___closed__5_value)}};
static const lean_object* lp_mathlib_PiFin_hasRepr___redArg___lam__1___closed__4 = (const lean_object*)&lp_mathlib_PiFin_hasRepr___redArg___lam__1___closed__4_value;
static const lean_ctor_object lp_mathlib_PiFin_hasRepr___redArg___lam__1___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_mathlib_Matrix_vecNotation___closed__15_value)}};
static const lean_object* lp_mathlib_PiFin_hasRepr___redArg___lam__1___closed__5 = (const lean_object*)&lp_mathlib_PiFin_hasRepr___redArg___lam__1___closed__5_value;
LEAN_EXPORT lean_object* lp_mathlib_PiFin_hasRepr___redArg___lam__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_PiFin_hasRepr___redArg___lam__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_PiFin_hasRepr___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Std_instToFormatFormat___lam__0___boxed, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_PiFin_hasRepr___redArg___closed__0 = (const lean_object*)&lp_mathlib_PiFin_hasRepr___redArg___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_PiFin_hasRepr___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_PiFin_hasRepr(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Matrix_matchVecConsPrefix(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Matrix_matchVecConsPrefix___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Nat_cast___at___00Matrix_cons__val_spec__0(lean_object*);
static const lean_ctor_object lp_mathlib_Matrix_cons__val___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 2}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_Matrix_cons__val___redArg___closed__0 = (const lean_object*)&lp_mathlib_Matrix_cons__val___redArg___closed__0_value;
static const lean_string_object lp_mathlib_Matrix_cons__val___redArg___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "NeZero"};
static const lean_object* lp_mathlib_Matrix_cons__val___redArg___closed__1 = (const lean_object*)&lp_mathlib_Matrix_cons__val___redArg___closed__1_value;
static const lean_ctor_object lp_mathlib_Matrix_cons__val___redArg___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Matrix_cons__val___redArg___closed__1_value),LEAN_SCALAR_PTR_LITERAL(82, 249, 173, 83, 51, 144, 28, 211)}};
static const lean_object* lp_mathlib_Matrix_cons__val___redArg___closed__2 = (const lean_object*)&lp_mathlib_Matrix_cons__val___redArg___closed__2_value;
static const lean_ctor_object lp_mathlib_Matrix_cons__val___redArg___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_Matrix_cons__val___redArg___closed__3 = (const lean_object*)&lp_mathlib_Matrix_cons__val___redArg___closed__3_value;
static lean_once_cell_t lp_mathlib_Matrix_cons__val___redArg___closed__4_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Matrix_cons__val___redArg___closed__4;
static const lean_string_object lp_mathlib_Matrix_cons__val___redArg___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = "Nat"};
static const lean_object* lp_mathlib_Matrix_cons__val___redArg___closed__5 = (const lean_object*)&lp_mathlib_Matrix_cons__val___redArg___closed__5_value;
static const lean_ctor_object lp_mathlib_Matrix_cons__val___redArg___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Matrix_cons__val___redArg___closed__5_value),LEAN_SCALAR_PTR_LITERAL(155, 221, 223, 104, 58, 13, 204, 158)}};
static const lean_object* lp_mathlib_Matrix_cons__val___redArg___closed__6 = (const lean_object*)&lp_mathlib_Matrix_cons__val___redArg___closed__6_value;
static lean_once_cell_t lp_mathlib_Matrix_cons__val___redArg___closed__7_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Matrix_cons__val___redArg___closed__7;
static lean_once_cell_t lp_mathlib_Matrix_cons__val___redArg___closed__8_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Matrix_cons__val___redArg___closed__8;
static const lean_string_object lp_mathlib_Matrix_cons__val___redArg___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "Zero"};
static const lean_object* lp_mathlib_Matrix_cons__val___redArg___closed__9 = (const lean_object*)&lp_mathlib_Matrix_cons__val___redArg___closed__9_value;
static const lean_string_object lp_mathlib_Matrix_cons__val___redArg___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "ofOfNat0"};
static const lean_object* lp_mathlib_Matrix_cons__val___redArg___closed__10 = (const lean_object*)&lp_mathlib_Matrix_cons__val___redArg___closed__10_value;
static const lean_ctor_object lp_mathlib_Matrix_cons__val___redArg___closed__11_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Matrix_cons__val___redArg___closed__9_value),LEAN_SCALAR_PTR_LITERAL(192, 171, 244, 106, 217, 72, 118, 253)}};
static const lean_ctor_object lp_mathlib_Matrix_cons__val___redArg___closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Matrix_cons__val___redArg___closed__11_value_aux_0),((lean_object*)&lp_mathlib_Matrix_cons__val___redArg___closed__10_value),LEAN_SCALAR_PTR_LITERAL(5, 143, 143, 98, 82, 180, 92, 57)}};
static const lean_object* lp_mathlib_Matrix_cons__val___redArg___closed__11 = (const lean_object*)&lp_mathlib_Matrix_cons__val___redArg___closed__11_value;
static lean_once_cell_t lp_mathlib_Matrix_cons__val___redArg___closed__12_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Matrix_cons__val___redArg___closed__12;
static lean_once_cell_t lp_mathlib_Matrix_cons__val___redArg___closed__13_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Matrix_cons__val___redArg___closed__13;
static const lean_string_object lp_mathlib_Matrix_cons__val___redArg___closed__14_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 13, .m_capacity = 13, .m_length = 12, .m_data = "instOfNatNat"};
static const lean_object* lp_mathlib_Matrix_cons__val___redArg___closed__14 = (const lean_object*)&lp_mathlib_Matrix_cons__val___redArg___closed__14_value;
static const lean_ctor_object lp_mathlib_Matrix_cons__val___redArg___closed__15_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Matrix_cons__val___redArg___closed__14_value),LEAN_SCALAR_PTR_LITERAL(217, 8, 172, 44, 179, 254, 147, 95)}};
static const lean_object* lp_mathlib_Matrix_cons__val___redArg___closed__15 = (const lean_object*)&lp_mathlib_Matrix_cons__val___redArg___closed__15_value;
static lean_once_cell_t lp_mathlib_Matrix_cons__val___redArg___closed__16_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Matrix_cons__val___redArg___closed__16;
static const lean_ctor_object lp_mathlib_Matrix_cons__val___redArg___closed__17_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 0}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_Matrix_cons__val___redArg___closed__17 = (const lean_object*)&lp_mathlib_Matrix_cons__val___redArg___closed__17_value;
static lean_once_cell_t lp_mathlib_Matrix_cons__val___redArg___closed__18_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Matrix_cons__val___redArg___closed__18;
static lean_once_cell_t lp_mathlib_Matrix_cons__val___redArg___closed__19_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Matrix_cons__val___redArg___closed__19;
static lean_once_cell_t lp_mathlib_Matrix_cons__val___redArg___closed__20_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Matrix_cons__val___redArg___closed__20;
static lean_once_cell_t lp_mathlib_Matrix_cons__val___redArg___closed__21_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Matrix_cons__val___redArg___closed__21;
static const lean_string_object lp_mathlib_Matrix_cons__val___redArg___closed__22_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = "Fin"};
static const lean_object* lp_mathlib_Matrix_cons__val___redArg___closed__22 = (const lean_object*)&lp_mathlib_Matrix_cons__val___redArg___closed__22_value;
static const lean_ctor_object lp_mathlib_Matrix_cons__val___redArg___closed__23_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Matrix_cons__val___redArg___closed__22_value),LEAN_SCALAR_PTR_LITERAL(62, 91, 162, 2, 110, 238, 123, 219)}};
static const lean_object* lp_mathlib_Matrix_cons__val___redArg___closed__23 = (const lean_object*)&lp_mathlib_Matrix_cons__val___redArg___closed__23_value;
static lean_once_cell_t lp_mathlib_Matrix_cons__val___redArg___closed__24_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Matrix_cons__val___redArg___closed__24;
static const lean_string_object lp_mathlib_Matrix_cons__val___redArg___closed__25_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "OfNat"};
static const lean_object* lp_mathlib_Matrix_cons__val___redArg___closed__25 = (const lean_object*)&lp_mathlib_Matrix_cons__val___redArg___closed__25_value;
static const lean_string_object lp_mathlib_Matrix_cons__val___redArg___closed__26_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "ofNat"};
static const lean_object* lp_mathlib_Matrix_cons__val___redArg___closed__26 = (const lean_object*)&lp_mathlib_Matrix_cons__val___redArg___closed__26_value;
static const lean_ctor_object lp_mathlib_Matrix_cons__val___redArg___closed__27_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Matrix_cons__val___redArg___closed__25_value),LEAN_SCALAR_PTR_LITERAL(135, 241, 166, 108, 243, 216, 193, 244)}};
static const lean_ctor_object lp_mathlib_Matrix_cons__val___redArg___closed__27_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Matrix_cons__val___redArg___closed__27_value_aux_0),((lean_object*)&lp_mathlib_Matrix_cons__val___redArg___closed__26_value),LEAN_SCALAR_PTR_LITERAL(2, 108, 58, 34, 100, 49, 50, 216)}};
static const lean_object* lp_mathlib_Matrix_cons__val___redArg___closed__27 = (const lean_object*)&lp_mathlib_Matrix_cons__val___redArg___closed__27_value;
static lean_once_cell_t lp_mathlib_Matrix_cons__val___redArg___closed__28_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Matrix_cons__val___redArg___closed__28;
static const lean_string_object lp_mathlib_Matrix_cons__val___redArg___closed__29_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 9, .m_data = "instOfNat"};
static const lean_object* lp_mathlib_Matrix_cons__val___redArg___closed__29 = (const lean_object*)&lp_mathlib_Matrix_cons__val___redArg___closed__29_value;
static const lean_ctor_object lp_mathlib_Matrix_cons__val___redArg___closed__30_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Matrix_cons__val___redArg___closed__22_value),LEAN_SCALAR_PTR_LITERAL(62, 91, 162, 2, 110, 238, 123, 219)}};
static const lean_ctor_object lp_mathlib_Matrix_cons__val___redArg___closed__30_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Matrix_cons__val___redArg___closed__30_value_aux_0),((lean_object*)&lp_mathlib_Matrix_cons__val___redArg___closed__29_value),LEAN_SCALAR_PTR_LITERAL(92, 84, 52, 176, 228, 163, 228, 83)}};
static const lean_object* lp_mathlib_Matrix_cons__val___redArg___closed__30 = (const lean_object*)&lp_mathlib_Matrix_cons__val___redArg___closed__30_value;
static lean_once_cell_t lp_mathlib_Matrix_cons__val___redArg___closed__31_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Matrix_cons__val___redArg___closed__31;
static lean_once_cell_t lp_mathlib_Matrix_cons__val___redArg___closed__32_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Matrix_cons__val___redArg___closed__32;
static const lean_string_object lp_mathlib_Matrix_cons__val___redArg___closed__33_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "HAdd"};
static const lean_object* lp_mathlib_Matrix_cons__val___redArg___closed__33 = (const lean_object*)&lp_mathlib_Matrix_cons__val___redArg___closed__33_value;
static const lean_string_object lp_mathlib_Matrix_cons__val___redArg___closed__34_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "hAdd"};
static const lean_object* lp_mathlib_Matrix_cons__val___redArg___closed__34 = (const lean_object*)&lp_mathlib_Matrix_cons__val___redArg___closed__34_value;
static const lean_ctor_object lp_mathlib_Matrix_cons__val___redArg___closed__35_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Matrix_cons__val___redArg___closed__33_value),LEAN_SCALAR_PTR_LITERAL(221, 239, 47, 196, 170, 166, 59, 144)}};
static const lean_ctor_object lp_mathlib_Matrix_cons__val___redArg___closed__35_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Matrix_cons__val___redArg___closed__35_value_aux_0),((lean_object*)&lp_mathlib_Matrix_cons__val___redArg___closed__34_value),LEAN_SCALAR_PTR_LITERAL(134, 172, 115, 219, 189, 252, 56, 148)}};
static const lean_object* lp_mathlib_Matrix_cons__val___redArg___closed__35 = (const lean_object*)&lp_mathlib_Matrix_cons__val___redArg___closed__35_value;
static const lean_ctor_object lp_mathlib_Matrix_cons__val___redArg___closed__36_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Matrix_cons__val___redArg___closed__3_value)}};
static const lean_object* lp_mathlib_Matrix_cons__val___redArg___closed__36 = (const lean_object*)&lp_mathlib_Matrix_cons__val___redArg___closed__36_value;
static const lean_ctor_object lp_mathlib_Matrix_cons__val___redArg___closed__37_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Matrix_cons__val___redArg___closed__36_value)}};
static const lean_object* lp_mathlib_Matrix_cons__val___redArg___closed__37 = (const lean_object*)&lp_mathlib_Matrix_cons__val___redArg___closed__37_value;
static lean_once_cell_t lp_mathlib_Matrix_cons__val___redArg___closed__38_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Matrix_cons__val___redArg___closed__38;
static lean_once_cell_t lp_mathlib_Matrix_cons__val___redArg___closed__39_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Matrix_cons__val___redArg___closed__39;
static lean_once_cell_t lp_mathlib_Matrix_cons__val___redArg___closed__40_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Matrix_cons__val___redArg___closed__40;
static lean_once_cell_t lp_mathlib_Matrix_cons__val___redArg___closed__41_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Matrix_cons__val___redArg___closed__41;
static const lean_string_object lp_mathlib_Matrix_cons__val___redArg___closed__42_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "instHAdd"};
static const lean_object* lp_mathlib_Matrix_cons__val___redArg___closed__42 = (const lean_object*)&lp_mathlib_Matrix_cons__val___redArg___closed__42_value;
static const lean_ctor_object lp_mathlib_Matrix_cons__val___redArg___closed__43_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Matrix_cons__val___redArg___closed__42_value),LEAN_SCALAR_PTR_LITERAL(229, 81, 239, 34, 203, 244, 36, 133)}};
static const lean_object* lp_mathlib_Matrix_cons__val___redArg___closed__43 = (const lean_object*)&lp_mathlib_Matrix_cons__val___redArg___closed__43_value;
static lean_once_cell_t lp_mathlib_Matrix_cons__val___redArg___closed__44_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Matrix_cons__val___redArg___closed__44;
static lean_once_cell_t lp_mathlib_Matrix_cons__val___redArg___closed__45_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Matrix_cons__val___redArg___closed__45;
static const lean_string_object lp_mathlib_Matrix_cons__val___redArg___closed__46_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 11, .m_capacity = 11, .m_length = 10, .m_data = "instAddNat"};
static const lean_object* lp_mathlib_Matrix_cons__val___redArg___closed__46 = (const lean_object*)&lp_mathlib_Matrix_cons__val___redArg___closed__46_value;
static const lean_ctor_object lp_mathlib_Matrix_cons__val___redArg___closed__47_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Matrix_cons__val___redArg___closed__46_value),LEAN_SCALAR_PTR_LITERAL(228, 164, 175, 25, 228, 165, 175, 183)}};
static const lean_object* lp_mathlib_Matrix_cons__val___redArg___closed__47 = (const lean_object*)&lp_mathlib_Matrix_cons__val___redArg___closed__47_value;
static lean_once_cell_t lp_mathlib_Matrix_cons__val___redArg___closed__48_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Matrix_cons__val___redArg___closed__48;
static lean_once_cell_t lp_mathlib_Matrix_cons__val___redArg___closed__49_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Matrix_cons__val___redArg___closed__49;
static lean_once_cell_t lp_mathlib_Matrix_cons__val___redArg___closed__50_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Matrix_cons__val___redArg___closed__50;
static lean_once_cell_t lp_mathlib_Matrix_cons__val___redArg___closed__51_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Matrix_cons__val___redArg___closed__51;
LEAN_EXPORT lean_object* lp_mathlib_Matrix_cons__val___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Matrix_cons__val___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Matrix_cons__val(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Matrix_cons__val___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Data_Fin_VecNotation_0__PiFin_mkLiteralQ_loop(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Data_Fin_VecNotation_0__PiFin_mkLiteralQ_loop___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_PiFin_mkLiteralQ(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_PiFin_mkLiteralQ___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_PiFin_toExpr___redArg___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_PiFin_toExpr___redArg___lam__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_PiFin_toExpr___redArg___lam__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_PiFin_toExpr___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_PiFin_toExpr(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Matrix_vecAppend___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Matrix_vecAppend___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Matrix_vecAppend(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Matrix_vecAppend___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Matrix_vecAlt0___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Matrix_vecAlt0___redArg___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Matrix_vecAlt0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Matrix_vecAlt0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Matrix_vecAlt1___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Matrix_vecAlt1___redArg___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Matrix_vecAlt1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Matrix_vecAlt1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Matrix_vecEmpty(lean_object* v_00_u03b1_1_, lean_object* v_a_2_){
_start:
{
lean_internal_panic_unreachable();
}
}
LEAN_EXPORT lean_object* lp_mathlib_Matrix_vecEmpty___boxed(lean_object* v_00_u03b1_3_, lean_object* v_a_4_){
_start:
{
lean_object* v_res_5_; 
v_res_5_ = lp_mathlib_Matrix_vecEmpty(v_00_u03b1_3_, v_a_4_);
lean_dec(v_a_4_);
return v_res_5_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Matrix_vecCons___redArg(lean_object* v_h_6_, lean_object* v_t_7_, lean_object* v_i_8_){
_start:
{
lean_object* v___x_9_; 
v___x_9_ = l_Fin_cases___redArg(v_h_6_, v_t_7_, v_i_8_);
return v___x_9_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Matrix_vecCons___redArg___boxed(lean_object* v_h_10_, lean_object* v_t_11_, lean_object* v_i_12_){
_start:
{
lean_object* v_res_13_; 
v_res_13_ = lp_mathlib_Matrix_vecCons___redArg(v_h_10_, v_t_11_, v_i_12_);
lean_dec(v_i_12_);
lean_dec(v_h_10_);
return v_res_13_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Matrix_vecCons(lean_object* v_00_u03b1_14_, lean_object* v_n_15_, lean_object* v_h_16_, lean_object* v_t_17_, lean_object* v_i_18_){
_start:
{
lean_object* v___x_19_; 
v___x_19_ = l_Fin_cases___redArg(v_h_16_, v_t_17_, v_i_18_);
return v___x_19_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Matrix_vecCons___boxed(lean_object* v_00_u03b1_20_, lean_object* v_n_21_, lean_object* v_h_22_, lean_object* v_t_23_, lean_object* v_i_24_){
_start:
{
lean_object* v_res_25_; 
v_res_25_ = lp_mathlib_Matrix_vecCons(v_00_u03b1_20_, v_n_21_, v_h_22_, v_t_23_, v_i_24_);
lean_dec(v_i_24_);
lean_dec(v_h_22_);
lean_dec(v_n_21_);
return v_res_25_;
}
}
static lean_object* _init_lp_mathlib_Matrix___aux__Mathlib__Data__Fin__VecNotation______macroRules__Matrix__vecNotation__1___closed__1(void){
_start:
{
lean_object* v___x_69_; lean_object* v___x_70_; 
v___x_69_ = ((lean_object*)(lp_mathlib_Matrix___aux__Mathlib__Data__Fin__VecNotation______macroRules__Matrix__vecNotation__1___closed__0));
v___x_70_ = l_String_toRawSubstring_x27(v___x_69_);
return v___x_70_;
}
}
static lean_object* _init_lp_mathlib_Matrix___aux__Mathlib__Data__Fin__VecNotation______macroRules__Matrix__vecNotation__1___closed__12(void){
_start:
{
lean_object* v___x_92_; lean_object* v___x_93_; 
v___x_92_ = ((lean_object*)(lp_mathlib_Matrix___aux__Mathlib__Data__Fin__VecNotation______macroRules__Matrix__vecNotation__1___closed__11));
v___x_93_ = l_String_toRawSubstring_x27(v___x_92_);
return v___x_93_;
}
}
static lean_object* _init_lp_mathlib_Matrix___aux__Mathlib__Data__Fin__VecNotation______macroRules__Matrix__vecNotation__1___closed__19(void){
_start:
{
lean_object* v___x_108_; 
v___x_108_ = l_Array_mkArray0(lean_box(0));
return v___x_108_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Matrix___aux__Mathlib__Data__Fin__VecNotation______macroRules__Matrix__vecNotation__1(lean_object* v_x_109_, lean_object* v_a_110_, lean_object* v_a_111_){
_start:
{
lean_object* v___x_112_; uint8_t v___x_113_; 
v___x_112_ = ((lean_object*)(lp_mathlib_Matrix_vecNotation___closed__2));
lean_inc(v_x_109_);
v___x_113_ = l_Lean_Syntax_isOfKind(v_x_109_, v___x_112_);
if (v___x_113_ == 0)
{
lean_object* v___x_114_; lean_object* v___x_115_; 
lean_dec(v_x_109_);
v___x_114_ = lean_box(1);
v___x_115_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_115_, 0, v___x_114_);
lean_ctor_set(v___x_115_, 1, v_a_111_);
return v___x_115_;
}
else
{
lean_object* v___x_116_; lean_object* v___x_117_; lean_object* v___x_118_; lean_object* v___x_119_; lean_object* v___x_120_; uint8_t v___x_121_; 
v___x_116_ = lean_unsigned_to_nat(0u);
v___x_117_ = lean_unsigned_to_nat(1u);
v___x_118_ = l_Lean_Syntax_getArg(v_x_109_, v___x_117_);
lean_dec(v_x_109_);
v___x_119_ = lean_unsigned_to_nat(2u);
v___x_120_ = l_Lean_Syntax_getNumArgs(v___x_118_);
v___x_121_ = lean_nat_dec_le(v___x_119_, v___x_120_);
if (v___x_121_ == 0)
{
uint8_t v___x_122_; 
lean_dec(v___x_120_);
lean_inc(v___x_118_);
v___x_122_ = l_Lean_Syntax_matchesNull(v___x_118_, v___x_117_);
if (v___x_122_ == 0)
{
uint8_t v___x_123_; 
v___x_123_ = l_Lean_Syntax_matchesNull(v___x_118_, v___x_116_);
if (v___x_123_ == 0)
{
lean_object* v___x_124_; lean_object* v___x_125_; 
v___x_124_ = lean_box(1);
v___x_125_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_125_, 0, v___x_124_);
lean_ctor_set(v___x_125_, 1, v_a_111_);
return v___x_125_;
}
else
{
lean_object* v_quotContext_126_; lean_object* v_currMacroScope_127_; lean_object* v_ref_128_; lean_object* v___x_129_; lean_object* v___x_130_; lean_object* v___x_131_; lean_object* v___x_132_; lean_object* v___x_133_; lean_object* v___x_134_; lean_object* v___x_135_; 
v_quotContext_126_ = lean_ctor_get(v_a_110_, 1);
v_currMacroScope_127_ = lean_ctor_get(v_a_110_, 2);
v_ref_128_ = lean_ctor_get(v_a_110_, 5);
v___x_129_ = l_Lean_SourceInfo_fromRef(v_ref_128_, v___x_122_);
v___x_130_ = lean_obj_once(&lp_mathlib_Matrix___aux__Mathlib__Data__Fin__VecNotation______macroRules__Matrix__vecNotation__1___closed__1, &lp_mathlib_Matrix___aux__Mathlib__Data__Fin__VecNotation______macroRules__Matrix__vecNotation__1___closed__1_once, _init_lp_mathlib_Matrix___aux__Mathlib__Data__Fin__VecNotation______macroRules__Matrix__vecNotation__1___closed__1);
v___x_131_ = ((lean_object*)(lp_mathlib_Matrix___aux__Mathlib__Data__Fin__VecNotation______macroRules__Matrix__vecNotation__1___closed__2));
lean_inc(v_currMacroScope_127_);
lean_inc(v_quotContext_126_);
v___x_132_ = l_Lean_addMacroScope(v_quotContext_126_, v___x_131_, v_currMacroScope_127_);
v___x_133_ = ((lean_object*)(lp_mathlib_Matrix___aux__Mathlib__Data__Fin__VecNotation______macroRules__Matrix__vecNotation__1___closed__5));
v___x_134_ = lean_alloc_ctor(3, 4, 0);
lean_ctor_set(v___x_134_, 0, v___x_129_);
lean_ctor_set(v___x_134_, 1, v___x_130_);
lean_ctor_set(v___x_134_, 2, v___x_132_);
lean_ctor_set(v___x_134_, 3, v___x_133_);
v___x_135_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_135_, 0, v___x_134_);
lean_ctor_set(v___x_135_, 1, v_a_111_);
return v___x_135_;
}
}
else
{
lean_object* v_quotContext_136_; lean_object* v_currMacroScope_137_; lean_object* v_ref_138_; lean_object* v___x_139_; lean_object* v___x_140_; lean_object* v___x_141_; lean_object* v___x_142_; lean_object* v___x_143_; lean_object* v___x_144_; lean_object* v___x_145_; lean_object* v___x_146_; lean_object* v___x_147_; lean_object* v___x_148_; lean_object* v___x_149_; lean_object* v___x_150_; lean_object* v___x_151_; lean_object* v___x_152_; lean_object* v___x_153_; lean_object* v___x_154_; lean_object* v___x_155_; lean_object* v___x_156_; lean_object* v___x_157_; 
v_quotContext_136_ = lean_ctor_get(v_a_110_, 1);
v_currMacroScope_137_ = lean_ctor_get(v_a_110_, 2);
v_ref_138_ = lean_ctor_get(v_a_110_, 5);
v___x_139_ = l_Lean_Syntax_getArg(v___x_118_, v___x_116_);
lean_dec(v___x_118_);
v___x_140_ = l_Lean_SourceInfo_fromRef(v_ref_138_, v___x_121_);
v___x_141_ = ((lean_object*)(lp_mathlib_Matrix___aux__Mathlib__Data__Fin__VecNotation______macroRules__Matrix__vecNotation__1___closed__10));
v___x_142_ = lean_obj_once(&lp_mathlib_Matrix___aux__Mathlib__Data__Fin__VecNotation______macroRules__Matrix__vecNotation__1___closed__12, &lp_mathlib_Matrix___aux__Mathlib__Data__Fin__VecNotation______macroRules__Matrix__vecNotation__1___closed__12_once, _init_lp_mathlib_Matrix___aux__Mathlib__Data__Fin__VecNotation______macroRules__Matrix__vecNotation__1___closed__12);
v___x_143_ = ((lean_object*)(lp_mathlib_Matrix___aux__Mathlib__Data__Fin__VecNotation______macroRules__Matrix__vecNotation__1___closed__13));
lean_inc(v_currMacroScope_137_);
lean_inc(v_quotContext_136_);
v___x_144_ = l_Lean_addMacroScope(v_quotContext_136_, v___x_143_, v_currMacroScope_137_);
v___x_145_ = ((lean_object*)(lp_mathlib_Matrix___aux__Mathlib__Data__Fin__VecNotation______macroRules__Matrix__vecNotation__1___closed__16));
lean_inc_n(v___x_140_, 6);
v___x_146_ = lean_alloc_ctor(3, 4, 0);
lean_ctor_set(v___x_146_, 0, v___x_140_);
lean_ctor_set(v___x_146_, 1, v___x_142_);
lean_ctor_set(v___x_146_, 2, v___x_144_);
lean_ctor_set(v___x_146_, 3, v___x_145_);
v___x_147_ = ((lean_object*)(lp_mathlib_Matrix___aux__Mathlib__Data__Fin__VecNotation______macroRules__Matrix__vecNotation__1___closed__18));
v___x_148_ = ((lean_object*)(lp_mathlib_Matrix_vecNotation___closed__5));
v___x_149_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_149_, 0, v___x_140_);
lean_ctor_set(v___x_149_, 1, v___x_148_);
v___x_150_ = lean_obj_once(&lp_mathlib_Matrix___aux__Mathlib__Data__Fin__VecNotation______macroRules__Matrix__vecNotation__1___closed__19, &lp_mathlib_Matrix___aux__Mathlib__Data__Fin__VecNotation______macroRules__Matrix__vecNotation__1___closed__19_once, _init_lp_mathlib_Matrix___aux__Mathlib__Data__Fin__VecNotation______macroRules__Matrix__vecNotation__1___closed__19);
v___x_151_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_151_, 0, v___x_140_);
lean_ctor_set(v___x_151_, 1, v___x_147_);
lean_ctor_set(v___x_151_, 2, v___x_150_);
v___x_152_ = ((lean_object*)(lp_mathlib_Matrix_vecNotation___closed__15));
v___x_153_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_153_, 0, v___x_140_);
lean_ctor_set(v___x_153_, 1, v___x_152_);
v___x_154_ = l_Lean_Syntax_node3(v___x_140_, v___x_112_, v___x_149_, v___x_151_, v___x_153_);
v___x_155_ = l_Lean_Syntax_node2(v___x_140_, v___x_147_, v___x_139_, v___x_154_);
v___x_156_ = l_Lean_Syntax_node2(v___x_140_, v___x_141_, v___x_146_, v___x_155_);
v___x_157_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_157_, 0, v___x_156_);
lean_ctor_set(v___x_157_, 1, v_a_111_);
return v___x_157_;
}
}
else
{
lean_object* v_quotContext_158_; lean_object* v_currMacroScope_159_; lean_object* v_ref_160_; lean_object* v___x_161_; lean_object* v___x_162_; lean_object* v___x_163_; lean_object* v___x_164_; lean_object* v___x_165_; lean_object* v___x_166_; lean_object* v___x_167_; uint8_t v___x_168_; lean_object* v___x_169_; lean_object* v___x_170_; lean_object* v___x_171_; lean_object* v___x_172_; lean_object* v___x_173_; lean_object* v___x_174_; lean_object* v___x_175_; lean_object* v___x_176_; lean_object* v___x_177_; lean_object* v___x_178_; lean_object* v___x_179_; lean_object* v___x_180_; lean_object* v___x_181_; lean_object* v___x_182_; lean_object* v___x_183_; lean_object* v___x_184_; lean_object* v___x_185_; lean_object* v___x_186_; 
v_quotContext_158_ = lean_ctor_get(v_a_110_, 1);
v_currMacroScope_159_ = lean_ctor_get(v_a_110_, 2);
v_ref_160_ = lean_ctor_get(v_a_110_, 5);
v___x_161_ = l_Lean_Syntax_getArg(v___x_118_, v___x_116_);
v___x_162_ = l_Lean_Syntax_getArgs(v___x_118_);
lean_dec(v___x_118_);
v___x_163_ = l_Array_extract___redArg(v___x_162_, v___x_119_, v___x_120_);
lean_dec_ref(v___x_162_);
v___x_164_ = ((lean_object*)(lp_mathlib_Matrix___aux__Mathlib__Data__Fin__VecNotation______macroRules__Matrix__vecNotation__1___closed__18));
v___x_165_ = lean_box(2);
v___x_166_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_166_, 0, v___x_165_);
lean_ctor_set(v___x_166_, 1, v___x_164_);
lean_ctor_set(v___x_166_, 2, v___x_163_);
v___x_167_ = l_Lean_Syntax_getArgs(v___x_166_);
lean_dec_ref_known(v___x_166_, 3);
v___x_168_ = 0;
v___x_169_ = l_Lean_SourceInfo_fromRef(v_ref_160_, v___x_168_);
v___x_170_ = ((lean_object*)(lp_mathlib_Matrix___aux__Mathlib__Data__Fin__VecNotation______macroRules__Matrix__vecNotation__1___closed__10));
v___x_171_ = lean_obj_once(&lp_mathlib_Matrix___aux__Mathlib__Data__Fin__VecNotation______macroRules__Matrix__vecNotation__1___closed__12, &lp_mathlib_Matrix___aux__Mathlib__Data__Fin__VecNotation______macroRules__Matrix__vecNotation__1___closed__12_once, _init_lp_mathlib_Matrix___aux__Mathlib__Data__Fin__VecNotation______macroRules__Matrix__vecNotation__1___closed__12);
v___x_172_ = ((lean_object*)(lp_mathlib_Matrix___aux__Mathlib__Data__Fin__VecNotation______macroRules__Matrix__vecNotation__1___closed__13));
lean_inc(v_currMacroScope_159_);
lean_inc(v_quotContext_158_);
v___x_173_ = l_Lean_addMacroScope(v_quotContext_158_, v___x_172_, v_currMacroScope_159_);
v___x_174_ = ((lean_object*)(lp_mathlib_Matrix___aux__Mathlib__Data__Fin__VecNotation______macroRules__Matrix__vecNotation__1___closed__16));
lean_inc_n(v___x_169_, 6);
v___x_175_ = lean_alloc_ctor(3, 4, 0);
lean_ctor_set(v___x_175_, 0, v___x_169_);
lean_ctor_set(v___x_175_, 1, v___x_171_);
lean_ctor_set(v___x_175_, 2, v___x_173_);
lean_ctor_set(v___x_175_, 3, v___x_174_);
v___x_176_ = ((lean_object*)(lp_mathlib_Matrix_vecNotation___closed__5));
v___x_177_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_177_, 0, v___x_169_);
lean_ctor_set(v___x_177_, 1, v___x_176_);
v___x_178_ = lean_obj_once(&lp_mathlib_Matrix___aux__Mathlib__Data__Fin__VecNotation______macroRules__Matrix__vecNotation__1___closed__19, &lp_mathlib_Matrix___aux__Mathlib__Data__Fin__VecNotation______macroRules__Matrix__vecNotation__1___closed__19_once, _init_lp_mathlib_Matrix___aux__Mathlib__Data__Fin__VecNotation______macroRules__Matrix__vecNotation__1___closed__19);
v___x_179_ = l_Array_append___redArg(v___x_178_, v___x_167_);
lean_dec_ref(v___x_167_);
v___x_180_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_180_, 0, v___x_169_);
lean_ctor_set(v___x_180_, 1, v___x_164_);
lean_ctor_set(v___x_180_, 2, v___x_179_);
v___x_181_ = ((lean_object*)(lp_mathlib_Matrix_vecNotation___closed__15));
v___x_182_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_182_, 0, v___x_169_);
lean_ctor_set(v___x_182_, 1, v___x_181_);
v___x_183_ = l_Lean_Syntax_node3(v___x_169_, v___x_112_, v___x_177_, v___x_180_, v___x_182_);
v___x_184_ = l_Lean_Syntax_node2(v___x_169_, v___x_164_, v___x_161_, v___x_183_);
v___x_185_ = l_Lean_Syntax_node2(v___x_169_, v___x_170_, v___x_175_, v___x_184_);
v___x_186_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_186_, 0, v___x_185_);
lean_ctor_set(v___x_186_, 1, v_a_111_);
return v___x_186_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Matrix___aux__Mathlib__Data__Fin__VecNotation______macroRules__Matrix__vecNotation__1___boxed(lean_object* v_x_187_, lean_object* v_a_188_, lean_object* v_a_189_){
_start:
{
lean_object* v_res_190_; 
v_res_190_ = lp_mathlib_Matrix___aux__Mathlib__Data__Fin__VecNotation______macroRules__Matrix__vecNotation__1(v_x_187_, v_a_188_, v_a_189_);
lean_dec_ref(v_a_188_);
return v_res_190_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Matrix_vecConsUnexpander(lean_object* v_x_191_, lean_object* v_a_192_, lean_object* v_a_193_){
_start:
{
lean_object* v___x_194_; uint8_t v___x_195_; 
v___x_194_ = ((lean_object*)(lp_mathlib_Matrix___aux__Mathlib__Data__Fin__VecNotation______macroRules__Matrix__vecNotation__1___closed__10));
lean_inc(v_x_191_);
v___x_195_ = l_Lean_Syntax_isOfKind(v_x_191_, v___x_194_);
if (v___x_195_ == 0)
{
lean_object* v___x_196_; lean_object* v___x_197_; 
lean_dec(v_x_191_);
v___x_196_ = lean_box(0);
v___x_197_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_197_, 0, v___x_196_);
lean_ctor_set(v___x_197_, 1, v_a_193_);
return v___x_197_;
}
else
{
lean_object* v___x_198_; lean_object* v___x_199_; lean_object* v___x_200_; uint8_t v___x_201_; 
v___x_198_ = lean_unsigned_to_nat(1u);
v___x_199_ = l_Lean_Syntax_getArg(v_x_191_, v___x_198_);
lean_dec(v_x_191_);
v___x_200_ = lean_unsigned_to_nat(2u);
lean_inc(v___x_199_);
v___x_201_ = l_Lean_Syntax_matchesNull(v___x_199_, v___x_200_);
if (v___x_201_ == 0)
{
lean_object* v___x_202_; lean_object* v___x_203_; 
lean_dec(v___x_199_);
v___x_202_ = lean_box(0);
v___x_203_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_203_, 0, v___x_202_);
lean_ctor_set(v___x_203_, 1, v_a_193_);
return v___x_203_;
}
else
{
lean_object* v___x_204_; lean_object* v___x_205_; uint8_t v___x_206_; 
v___x_204_ = l_Lean_Syntax_getArg(v___x_199_, v___x_198_);
v___x_205_ = ((lean_object*)(lp_mathlib_Matrix_vecNotation___closed__2));
lean_inc(v___x_204_);
v___x_206_ = l_Lean_Syntax_isOfKind(v___x_204_, v___x_205_);
if (v___x_206_ == 0)
{
lean_object* v___x_207_; lean_object* v___x_208_; 
lean_dec(v___x_204_);
lean_dec(v___x_199_);
v___x_207_ = lean_box(0);
v___x_208_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_208_, 0, v___x_207_);
lean_ctor_set(v___x_208_, 1, v_a_193_);
return v___x_208_;
}
else
{
lean_object* v___x_209_; lean_object* v___x_210_; lean_object* v___x_211_; lean_object* v___x_212_; uint8_t v___x_213_; 
v___x_209_ = lean_unsigned_to_nat(0u);
v___x_210_ = l_Lean_Syntax_getArg(v___x_199_, v___x_209_);
lean_dec(v___x_199_);
v___x_211_ = l_Lean_Syntax_getArg(v___x_204_, v___x_198_);
lean_dec(v___x_204_);
v___x_212_ = l_Lean_Syntax_getNumArgs(v___x_211_);
v___x_213_ = lean_nat_dec_le(v___x_200_, v___x_212_);
if (v___x_213_ == 0)
{
uint8_t v___x_214_; 
lean_dec(v___x_212_);
lean_inc(v___x_211_);
v___x_214_ = l_Lean_Syntax_matchesNull(v___x_211_, v___x_198_);
if (v___x_214_ == 0)
{
uint8_t v___x_215_; 
v___x_215_ = l_Lean_Syntax_matchesNull(v___x_211_, v___x_209_);
if (v___x_215_ == 0)
{
lean_object* v___x_216_; lean_object* v___x_217_; 
lean_dec(v___x_210_);
v___x_216_ = lean_box(0);
v___x_217_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_217_, 0, v___x_216_);
lean_ctor_set(v___x_217_, 1, v_a_193_);
return v___x_217_;
}
else
{
lean_object* v___x_218_; lean_object* v___x_219_; lean_object* v___x_220_; lean_object* v___x_221_; lean_object* v___x_222_; lean_object* v___x_223_; lean_object* v___x_224_; lean_object* v___x_225_; lean_object* v___x_226_; 
v___x_218_ = l_Lean_SourceInfo_fromRef(v_a_192_, v___x_214_);
v___x_219_ = ((lean_object*)(lp_mathlib_Matrix_vecNotation___closed__5));
lean_inc_n(v___x_218_, 3);
v___x_220_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_220_, 0, v___x_218_);
lean_ctor_set(v___x_220_, 1, v___x_219_);
v___x_221_ = ((lean_object*)(lp_mathlib_Matrix___aux__Mathlib__Data__Fin__VecNotation______macroRules__Matrix__vecNotation__1___closed__18));
v___x_222_ = l_Lean_Syntax_node1(v___x_218_, v___x_221_, v___x_210_);
v___x_223_ = ((lean_object*)(lp_mathlib_Matrix_vecNotation___closed__15));
v___x_224_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_224_, 0, v___x_218_);
lean_ctor_set(v___x_224_, 1, v___x_223_);
v___x_225_ = l_Lean_Syntax_node3(v___x_218_, v___x_205_, v___x_220_, v___x_222_, v___x_224_);
v___x_226_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_226_, 0, v___x_225_);
lean_ctor_set(v___x_226_, 1, v_a_193_);
return v___x_226_;
}
}
else
{
lean_object* v___x_227_; lean_object* v___x_228_; lean_object* v___x_229_; lean_object* v___x_230_; lean_object* v___x_231_; lean_object* v___x_232_; lean_object* v___x_233_; lean_object* v___x_234_; lean_object* v___x_235_; lean_object* v___x_236_; lean_object* v___x_237_; lean_object* v___x_238_; 
v___x_227_ = l_Lean_Syntax_getArg(v___x_211_, v___x_209_);
lean_dec(v___x_211_);
v___x_228_ = l_Lean_SourceInfo_fromRef(v_a_192_, v___x_213_);
v___x_229_ = ((lean_object*)(lp_mathlib_Matrix_vecNotation___closed__5));
lean_inc_n(v___x_228_, 4);
v___x_230_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_230_, 0, v___x_228_);
lean_ctor_set(v___x_230_, 1, v___x_229_);
v___x_231_ = ((lean_object*)(lp_mathlib_Matrix___aux__Mathlib__Data__Fin__VecNotation______macroRules__Matrix__vecNotation__1___closed__18));
v___x_232_ = ((lean_object*)(lp_mathlib_Matrix_vecNotation___closed__10));
v___x_233_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_233_, 0, v___x_228_);
lean_ctor_set(v___x_233_, 1, v___x_232_);
v___x_234_ = l_Lean_Syntax_node3(v___x_228_, v___x_231_, v___x_210_, v___x_233_, v___x_227_);
v___x_235_ = ((lean_object*)(lp_mathlib_Matrix_vecNotation___closed__15));
v___x_236_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_236_, 0, v___x_228_);
lean_ctor_set(v___x_236_, 1, v___x_235_);
v___x_237_ = l_Lean_Syntax_node3(v___x_228_, v___x_205_, v___x_230_, v___x_234_, v___x_236_);
v___x_238_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_238_, 0, v___x_237_);
lean_ctor_set(v___x_238_, 1, v_a_193_);
return v___x_238_;
}
}
else
{
lean_object* v___x_239_; lean_object* v___x_240_; lean_object* v___x_241_; lean_object* v___x_242_; lean_object* v___x_243_; lean_object* v___x_244_; lean_object* v___x_245_; lean_object* v___x_246_; uint8_t v___x_247_; lean_object* v___x_248_; lean_object* v___x_249_; lean_object* v___x_250_; lean_object* v___x_251_; lean_object* v___x_252_; lean_object* v___x_253_; lean_object* v___x_254_; lean_object* v___x_255_; lean_object* v___x_256_; lean_object* v___x_257_; lean_object* v___x_258_; 
v___x_239_ = l_Lean_Syntax_getArg(v___x_211_, v___x_209_);
v___x_240_ = l_Lean_Syntax_getArgs(v___x_211_);
lean_dec(v___x_211_);
v___x_241_ = l_Array_extract___redArg(v___x_240_, v___x_200_, v___x_212_);
lean_dec_ref(v___x_240_);
v___x_242_ = ((lean_object*)(lp_mathlib_Matrix___aux__Mathlib__Data__Fin__VecNotation______macroRules__Matrix__vecNotation__1___closed__18));
v___x_243_ = lean_box(2);
v___x_244_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_244_, 0, v___x_243_);
lean_ctor_set(v___x_244_, 1, v___x_242_);
lean_ctor_set(v___x_244_, 2, v___x_241_);
v___x_245_ = ((lean_object*)(lp_mathlib_Matrix_vecNotation___closed__10));
v___x_246_ = l_Lean_Syntax_getArgs(v___x_244_);
lean_dec_ref_known(v___x_244_, 3);
v___x_247_ = 0;
v___x_248_ = l_Lean_SourceInfo_fromRef(v_a_192_, v___x_247_);
v___x_249_ = ((lean_object*)(lp_mathlib_Matrix_vecNotation___closed__5));
lean_inc_n(v___x_248_, 4);
v___x_250_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_250_, 0, v___x_248_);
lean_ctor_set(v___x_250_, 1, v___x_249_);
v___x_251_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_251_, 0, v___x_248_);
lean_ctor_set(v___x_251_, 1, v___x_245_);
lean_inc_ref(v___x_251_);
v___x_252_ = l_Array_mkArray4___redArg(v___x_210_, v___x_251_, v___x_239_, v___x_251_);
v___x_253_ = l_Array_append___redArg(v___x_252_, v___x_246_);
lean_dec_ref(v___x_246_);
v___x_254_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_254_, 0, v___x_248_);
lean_ctor_set(v___x_254_, 1, v___x_242_);
lean_ctor_set(v___x_254_, 2, v___x_253_);
v___x_255_ = ((lean_object*)(lp_mathlib_Matrix_vecNotation___closed__15));
v___x_256_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_256_, 0, v___x_248_);
lean_ctor_set(v___x_256_, 1, v___x_255_);
v___x_257_ = l_Lean_Syntax_node3(v___x_248_, v___x_205_, v___x_250_, v___x_254_, v___x_256_);
v___x_258_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_258_, 0, v___x_257_);
lean_ctor_set(v___x_258_, 1, v_a_193_);
return v___x_258_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Matrix_vecConsUnexpander___boxed(lean_object* v_x_259_, lean_object* v_a_260_, lean_object* v_a_261_){
_start:
{
lean_object* v_res_262_; 
v_res_262_ = lp_mathlib_Matrix_vecConsUnexpander(v_x_259_, v_a_260_, v_a_261_);
lean_dec(v_a_260_);
return v_res_262_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Matrix_vecEmptyUnexpander(lean_object* v_x_266_, lean_object* v_a_267_, lean_object* v_a_268_){
_start:
{
lean_object* v___x_269_; uint8_t v___x_270_; 
v___x_269_ = ((lean_object*)(lp_mathlib_Matrix_vecEmptyUnexpander___closed__1));
v___x_270_ = l_Lean_Syntax_isOfKind(v_x_266_, v___x_269_);
if (v___x_270_ == 0)
{
lean_object* v___x_271_; lean_object* v___x_272_; 
v___x_271_ = lean_box(0);
v___x_272_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_272_, 0, v___x_271_);
lean_ctor_set(v___x_272_, 1, v_a_268_);
return v___x_272_;
}
else
{
uint8_t v___x_273_; lean_object* v___x_274_; lean_object* v___x_275_; lean_object* v___x_276_; lean_object* v___x_277_; lean_object* v___x_278_; lean_object* v___x_279_; lean_object* v___x_280_; lean_object* v___x_281_; lean_object* v___x_282_; lean_object* v___x_283_; lean_object* v___x_284_; 
v___x_273_ = 0;
v___x_274_ = l_Lean_SourceInfo_fromRef(v_a_267_, v___x_273_);
v___x_275_ = ((lean_object*)(lp_mathlib_Matrix_vecNotation___closed__2));
v___x_276_ = ((lean_object*)(lp_mathlib_Matrix_vecNotation___closed__5));
lean_inc_n(v___x_274_, 3);
v___x_277_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_277_, 0, v___x_274_);
lean_ctor_set(v___x_277_, 1, v___x_276_);
v___x_278_ = ((lean_object*)(lp_mathlib_Matrix___aux__Mathlib__Data__Fin__VecNotation______macroRules__Matrix__vecNotation__1___closed__18));
v___x_279_ = lean_obj_once(&lp_mathlib_Matrix___aux__Mathlib__Data__Fin__VecNotation______macroRules__Matrix__vecNotation__1___closed__19, &lp_mathlib_Matrix___aux__Mathlib__Data__Fin__VecNotation______macroRules__Matrix__vecNotation__1___closed__19_once, _init_lp_mathlib_Matrix___aux__Mathlib__Data__Fin__VecNotation______macroRules__Matrix__vecNotation__1___closed__19);
v___x_280_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_280_, 0, v___x_274_);
lean_ctor_set(v___x_280_, 1, v___x_278_);
lean_ctor_set(v___x_280_, 2, v___x_279_);
v___x_281_ = ((lean_object*)(lp_mathlib_Matrix_vecNotation___closed__15));
v___x_282_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_282_, 0, v___x_274_);
lean_ctor_set(v___x_282_, 1, v___x_281_);
v___x_283_ = l_Lean_Syntax_node3(v___x_274_, v___x_275_, v___x_277_, v___x_280_, v___x_282_);
v___x_284_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_284_, 0, v___x_283_);
lean_ctor_set(v___x_284_, 1, v_a_268_);
return v___x_284_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Matrix_vecEmptyUnexpander___boxed(lean_object* v_x_285_, lean_object* v_a_286_, lean_object* v_a_287_){
_start:
{
lean_object* v_res_288_; 
v_res_288_ = lp_mathlib_Matrix_vecEmptyUnexpander(v_x_285_, v_a_286_, v_a_287_);
lean_dec(v_a_286_);
return v_res_288_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Matrix_vecHead___redArg(lean_object* v_n_289_, lean_object* v_v_290_){
_start:
{
lean_object* v___x_291_; lean_object* v___x_292_; lean_object* v___x_293_; lean_object* v___x_294_; lean_object* v___x_295_; 
v___x_291_ = lean_unsigned_to_nat(1u);
v___x_292_ = lean_nat_add(v_n_289_, v___x_291_);
v___x_293_ = lean_unsigned_to_nat(0u);
v___x_294_ = lean_nat_mod(v___x_293_, v___x_292_);
lean_dec(v___x_292_);
v___x_295_ = lean_apply_1(v_v_290_, v___x_294_);
return v___x_295_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Matrix_vecHead___redArg___boxed(lean_object* v_n_296_, lean_object* v_v_297_){
_start:
{
lean_object* v_res_298_; 
v_res_298_ = lp_mathlib_Matrix_vecHead___redArg(v_n_296_, v_v_297_);
lean_dec(v_n_296_);
return v_res_298_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Matrix_vecHead(lean_object* v_00_u03b1_299_, lean_object* v_n_300_, lean_object* v_v_301_){
_start:
{
lean_object* v___x_302_; 
v___x_302_ = lp_mathlib_Matrix_vecHead___redArg(v_n_300_, v_v_301_);
return v___x_302_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Matrix_vecHead___boxed(lean_object* v_00_u03b1_303_, lean_object* v_n_304_, lean_object* v_v_305_){
_start:
{
lean_object* v_res_306_; 
v_res_306_ = lp_mathlib_Matrix_vecHead(v_00_u03b1_303_, v_n_304_, v_v_305_);
lean_dec(v_n_304_);
return v_res_306_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Matrix_vecTail___redArg(lean_object* v_v_307_, lean_object* v_a_308_){
_start:
{
lean_object* v___x_309_; lean_object* v___x_310_; 
v___x_309_ = l_Fin_succ___redArg(v_a_308_);
v___x_310_ = lean_apply_1(v_v_307_, v___x_309_);
return v___x_310_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Matrix_vecTail___redArg___boxed(lean_object* v_v_311_, lean_object* v_a_312_){
_start:
{
lean_object* v_res_313_; 
v_res_313_ = lp_mathlib_Matrix_vecTail___redArg(v_v_311_, v_a_312_);
lean_dec(v_a_312_);
return v_res_313_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Matrix_vecTail(lean_object* v_00_u03b1_314_, lean_object* v_n_315_, lean_object* v_v_316_, lean_object* v_a_317_){
_start:
{
lean_object* v___x_318_; 
v___x_318_ = lp_mathlib_Matrix_vecTail___redArg(v_v_316_, v_a_317_);
return v___x_318_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Matrix_vecTail___boxed(lean_object* v_00_u03b1_319_, lean_object* v_n_320_, lean_object* v_v_321_, lean_object* v_a_322_){
_start:
{
lean_object* v_res_323_; 
v_res_323_ = lp_mathlib_Matrix_vecTail(v_00_u03b1_319_, v_n_320_, v_v_321_, v_a_322_);
lean_dec(v_a_322_);
lean_dec(v_n_320_);
return v_res_323_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_PiFin_hasRepr___redArg___lam__0(lean_object* v_f_324_, lean_object* v_inst_325_, lean_object* v_n_326_){
_start:
{
lean_object* v___x_327_; lean_object* v___x_328_; lean_object* v___x_329_; 
v___x_327_ = lean_apply_1(v_f_324_, v_n_326_);
v___x_328_ = lean_unsigned_to_nat(0u);
v___x_329_ = lean_apply_2(v_inst_325_, v___x_327_, v___x_328_);
return v___x_329_;
}
}
static lean_object* _init_lp_mathlib_PiFin_hasRepr___redArg___lam__1___closed__2(void){
_start:
{
lean_object* v___x_335_; lean_object* v___x_336_; 
v___x_335_ = ((lean_object*)(lp_mathlib_Matrix_vecNotation___closed__5));
v___x_336_ = lean_string_length(v___x_335_);
return v___x_336_;
}
}
static lean_object* _init_lp_mathlib_PiFin_hasRepr___redArg___lam__1___closed__3(void){
_start:
{
lean_object* v___x_337_; lean_object* v___x_338_; 
v___x_337_ = lean_obj_once(&lp_mathlib_PiFin_hasRepr___redArg___lam__1___closed__2, &lp_mathlib_PiFin_hasRepr___redArg___lam__1___closed__2_once, _init_lp_mathlib_PiFin_hasRepr___redArg___lam__1___closed__2);
v___x_338_ = lean_nat_to_int(v___x_337_);
return v___x_338_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_PiFin_hasRepr___redArg___lam__1(lean_object* v_inst_343_, lean_object* v_n_344_, lean_object* v___f_345_, lean_object* v_f_346_, lean_object* v_x_347_){
_start:
{
lean_object* v___f_348_; lean_object* v___x_349_; lean_object* v___x_350_; lean_object* v___x_351_; lean_object* v___x_352_; lean_object* v___x_353_; lean_object* v___x_354_; lean_object* v___x_355_; lean_object* v___x_356_; lean_object* v___x_357_; lean_object* v___x_358_; lean_object* v___x_359_; uint8_t v___x_360_; lean_object* v___x_361_; 
v___f_348_ = lean_alloc_closure((void*)(lp_mathlib_PiFin_hasRepr___redArg___lam__0), 3, 2);
lean_closure_set(v___f_348_, 0, v_f_346_);
lean_closure_set(v___f_348_, 1, v_inst_343_);
v___x_349_ = l_List_finRange(v_n_344_);
v___x_350_ = lean_box(0);
v___x_351_ = l_List_mapTR_loop___redArg(v___f_348_, v___x_349_, v___x_350_);
v___x_352_ = ((lean_object*)(lp_mathlib_PiFin_hasRepr___redArg___lam__1___closed__1));
v___x_353_ = l_Std_Format_joinSep___redArg(v___f_345_, v___x_351_, v___x_352_);
v___x_354_ = lean_obj_once(&lp_mathlib_PiFin_hasRepr___redArg___lam__1___closed__3, &lp_mathlib_PiFin_hasRepr___redArg___lam__1___closed__3_once, _init_lp_mathlib_PiFin_hasRepr___redArg___lam__1___closed__3);
v___x_355_ = ((lean_object*)(lp_mathlib_PiFin_hasRepr___redArg___lam__1___closed__4));
v___x_356_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_356_, 0, v___x_355_);
lean_ctor_set(v___x_356_, 1, v___x_353_);
v___x_357_ = ((lean_object*)(lp_mathlib_PiFin_hasRepr___redArg___lam__1___closed__5));
v___x_358_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_358_, 0, v___x_356_);
lean_ctor_set(v___x_358_, 1, v___x_357_);
v___x_359_ = lean_alloc_ctor(4, 2, 0);
lean_ctor_set(v___x_359_, 0, v___x_354_);
lean_ctor_set(v___x_359_, 1, v___x_358_);
v___x_360_ = 0;
v___x_361_ = lean_alloc_ctor(6, 1, 1);
lean_ctor_set(v___x_361_, 0, v___x_359_);
lean_ctor_set_uint8(v___x_361_, sizeof(void*)*1, v___x_360_);
return v___x_361_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_PiFin_hasRepr___redArg___lam__1___boxed(lean_object* v_inst_362_, lean_object* v_n_363_, lean_object* v___f_364_, lean_object* v_f_365_, lean_object* v_x_366_){
_start:
{
lean_object* v_res_367_; 
v_res_367_ = lp_mathlib_PiFin_hasRepr___redArg___lam__1(v_inst_362_, v_n_363_, v___f_364_, v_f_365_, v_x_366_);
lean_dec(v_x_366_);
return v_res_367_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_PiFin_hasRepr___redArg(lean_object* v_n_369_, lean_object* v_inst_370_){
_start:
{
lean_object* v___f_371_; lean_object* v___f_372_; 
v___f_371_ = ((lean_object*)(lp_mathlib_PiFin_hasRepr___redArg___closed__0));
v___f_372_ = lean_alloc_closure((void*)(lp_mathlib_PiFin_hasRepr___redArg___lam__1___boxed), 5, 3);
lean_closure_set(v___f_372_, 0, v_inst_370_);
lean_closure_set(v___f_372_, 1, v_n_369_);
lean_closure_set(v___f_372_, 2, v___f_371_);
return v___f_372_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_PiFin_hasRepr(lean_object* v_00_u03b1_373_, lean_object* v_n_374_, lean_object* v_inst_375_){
_start:
{
lean_object* v___x_376_; 
v___x_376_ = lp_mathlib_PiFin_hasRepr___redArg(v_n_374_, v_inst_375_);
return v___x_376_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Matrix_matchVecConsPrefix(lean_object* v_n_377_, lean_object* v_e_378_, lean_object* v_a_379_, lean_object* v_a_380_, lean_object* v_a_381_, lean_object* v_a_382_){
_start:
{
lean_object* v___x_384_; 
lean_inc_ref(v_e_378_);
v___x_384_ = l_Lean_Meta_whnfR(v_e_378_, v_a_379_, v_a_380_, v_a_381_, v_a_382_);
if (lean_obj_tag(v___x_384_) == 0)
{
lean_object* v_a_385_; lean_object* v___x_386_; 
v_a_385_ = lean_ctor_get(v___x_384_, 0);
lean_inc(v_a_385_);
lean_dec_ref_known(v___x_384_, 1);
v___x_386_ = l_Lean_Meta_instantiateMVarsIfMVarApp___redArg(v_a_385_, v_a_380_);
if (lean_obj_tag(v___x_386_) == 0)
{
lean_object* v_a_387_; lean_object* v___x_389_; uint8_t v_isShared_390_; uint8_t v_isSharedCheck_431_; 
v_a_387_ = lean_ctor_get(v___x_386_, 0);
v_isSharedCheck_431_ = !lean_is_exclusive(v___x_386_);
if (v_isSharedCheck_431_ == 0)
{
v___x_389_ = v___x_386_;
v_isShared_390_ = v_isSharedCheck_431_;
goto v_resetjp_388_;
}
else
{
lean_inc(v_a_387_);
lean_dec(v___x_386_);
v___x_389_ = lean_box(0);
v_isShared_390_ = v_isSharedCheck_431_;
goto v_resetjp_388_;
}
v_resetjp_388_:
{
lean_object* v___x_398_; uint8_t v___x_399_; 
v___x_398_ = l_Lean_Expr_cleanupAnnotations(v_a_387_);
v___x_399_ = l_Lean_Expr_isApp(v___x_398_);
if (v___x_399_ == 0)
{
lean_dec_ref(v___x_398_);
goto v___jp_391_;
}
else
{
lean_object* v_arg_400_; lean_object* v___x_401_; uint8_t v___x_402_; 
v_arg_400_ = lean_ctor_get(v___x_398_, 1);
lean_inc_ref(v_arg_400_);
v___x_401_ = l_Lean_Expr_appFnCleanup___redArg(v___x_398_);
v___x_402_ = l_Lean_Expr_isApp(v___x_401_);
if (v___x_402_ == 0)
{
lean_dec_ref(v___x_401_);
lean_dec_ref(v_arg_400_);
goto v___jp_391_;
}
else
{
lean_object* v_arg_403_; lean_object* v___x_404_; uint8_t v___x_405_; 
v_arg_403_ = lean_ctor_get(v___x_401_, 1);
lean_inc_ref(v_arg_403_);
v___x_404_ = l_Lean_Expr_appFnCleanup___redArg(v___x_401_);
v___x_405_ = l_Lean_Expr_isApp(v___x_404_);
if (v___x_405_ == 0)
{
lean_dec_ref(v___x_404_);
lean_dec_ref(v_arg_403_);
lean_dec_ref(v_arg_400_);
goto v___jp_391_;
}
else
{
lean_object* v_arg_406_; lean_object* v___x_407_; uint8_t v___x_408_; 
v_arg_406_ = lean_ctor_get(v___x_404_, 1);
lean_inc_ref(v_arg_406_);
v___x_407_ = l_Lean_Expr_appFnCleanup___redArg(v___x_404_);
v___x_408_ = l_Lean_Expr_isApp(v___x_407_);
if (v___x_408_ == 0)
{
lean_dec_ref(v___x_407_);
lean_dec_ref(v_arg_406_);
lean_dec_ref(v_arg_403_);
lean_dec_ref(v_arg_400_);
goto v___jp_391_;
}
else
{
lean_object* v___x_409_; lean_object* v___x_410_; uint8_t v___x_411_; 
v___x_409_ = l_Lean_Expr_appFnCleanup___redArg(v___x_407_);
v___x_410_ = ((lean_object*)(lp_mathlib_Matrix___aux__Mathlib__Data__Fin__VecNotation______macroRules__Matrix__vecNotation__1___closed__14));
v___x_411_ = l_Lean_Expr_isConstOf(v___x_409_, v___x_410_);
lean_dec_ref(v___x_409_);
if (v___x_411_ == 0)
{
lean_dec_ref(v_arg_406_);
lean_dec_ref(v_arg_403_);
lean_dec_ref(v_arg_400_);
goto v___jp_391_;
}
else
{
lean_object* v___x_412_; 
lean_del_object(v___x_389_);
lean_dec_ref(v_e_378_);
lean_dec_ref(v_n_377_);
v___x_412_ = lp_mathlib_Matrix_matchVecConsPrefix(v_arg_406_, v_arg_400_, v_a_379_, v_a_380_, v_a_381_, v_a_382_);
if (lean_obj_tag(v___x_412_) == 0)
{
lean_object* v_a_413_; lean_object* v___x_415_; uint8_t v_isShared_416_; uint8_t v_isSharedCheck_430_; 
v_a_413_ = lean_ctor_get(v___x_412_, 0);
v_isSharedCheck_430_ = !lean_is_exclusive(v___x_412_);
if (v_isSharedCheck_430_ == 0)
{
v___x_415_ = v___x_412_;
v_isShared_416_ = v_isSharedCheck_430_;
goto v_resetjp_414_;
}
else
{
lean_inc(v_a_413_);
lean_dec(v___x_412_);
v___x_415_ = lean_box(0);
v_isShared_416_ = v_isSharedCheck_430_;
goto v_resetjp_414_;
}
v_resetjp_414_:
{
lean_object* v_fst_417_; lean_object* v_snd_418_; lean_object* v___x_420_; uint8_t v_isShared_421_; uint8_t v_isSharedCheck_429_; 
v_fst_417_ = lean_ctor_get(v_a_413_, 0);
v_snd_418_ = lean_ctor_get(v_a_413_, 1);
v_isSharedCheck_429_ = !lean_is_exclusive(v_a_413_);
if (v_isSharedCheck_429_ == 0)
{
v___x_420_ = v_a_413_;
v_isShared_421_ = v_isSharedCheck_429_;
goto v_resetjp_419_;
}
else
{
lean_inc(v_snd_418_);
lean_inc(v_fst_417_);
lean_dec(v_a_413_);
v___x_420_ = lean_box(0);
v_isShared_421_ = v_isSharedCheck_429_;
goto v_resetjp_419_;
}
v_resetjp_419_:
{
lean_object* v___x_422_; lean_object* v___x_424_; 
v___x_422_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_422_, 0, v_arg_403_);
lean_ctor_set(v___x_422_, 1, v_fst_417_);
if (v_isShared_421_ == 0)
{
lean_ctor_set(v___x_420_, 0, v___x_422_);
v___x_424_ = v___x_420_;
goto v_reusejp_423_;
}
else
{
lean_object* v_reuseFailAlloc_428_; 
v_reuseFailAlloc_428_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_428_, 0, v___x_422_);
lean_ctor_set(v_reuseFailAlloc_428_, 1, v_snd_418_);
v___x_424_ = v_reuseFailAlloc_428_;
goto v_reusejp_423_;
}
v_reusejp_423_:
{
lean_object* v___x_426_; 
if (v_isShared_416_ == 0)
{
lean_ctor_set(v___x_415_, 0, v___x_424_);
v___x_426_ = v___x_415_;
goto v_reusejp_425_;
}
else
{
lean_object* v_reuseFailAlloc_427_; 
v_reuseFailAlloc_427_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_427_, 0, v___x_424_);
v___x_426_ = v_reuseFailAlloc_427_;
goto v_reusejp_425_;
}
v_reusejp_425_:
{
return v___x_426_;
}
}
}
}
}
else
{
lean_dec_ref(v_arg_403_);
return v___x_412_;
}
}
}
}
}
}
v___jp_391_:
{
lean_object* v___x_392_; lean_object* v___x_393_; lean_object* v___x_394_; lean_object* v___x_396_; 
v___x_392_ = lean_box(0);
v___x_393_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_393_, 0, v_n_377_);
lean_ctor_set(v___x_393_, 1, v_e_378_);
v___x_394_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_394_, 0, v___x_392_);
lean_ctor_set(v___x_394_, 1, v___x_393_);
if (v_isShared_390_ == 0)
{
lean_ctor_set(v___x_389_, 0, v___x_394_);
v___x_396_ = v___x_389_;
goto v_reusejp_395_;
}
else
{
lean_object* v_reuseFailAlloc_397_; 
v_reuseFailAlloc_397_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_397_, 0, v___x_394_);
v___x_396_ = v_reuseFailAlloc_397_;
goto v_reusejp_395_;
}
v_reusejp_395_:
{
return v___x_396_;
}
}
}
}
else
{
lean_object* v_a_432_; lean_object* v___x_434_; uint8_t v_isShared_435_; uint8_t v_isSharedCheck_439_; 
lean_dec_ref(v_e_378_);
lean_dec_ref(v_n_377_);
v_a_432_ = lean_ctor_get(v___x_386_, 0);
v_isSharedCheck_439_ = !lean_is_exclusive(v___x_386_);
if (v_isSharedCheck_439_ == 0)
{
v___x_434_ = v___x_386_;
v_isShared_435_ = v_isSharedCheck_439_;
goto v_resetjp_433_;
}
else
{
lean_inc(v_a_432_);
lean_dec(v___x_386_);
v___x_434_ = lean_box(0);
v_isShared_435_ = v_isSharedCheck_439_;
goto v_resetjp_433_;
}
v_resetjp_433_:
{
lean_object* v___x_437_; 
if (v_isShared_435_ == 0)
{
v___x_437_ = v___x_434_;
goto v_reusejp_436_;
}
else
{
lean_object* v_reuseFailAlloc_438_; 
v_reuseFailAlloc_438_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_438_, 0, v_a_432_);
v___x_437_ = v_reuseFailAlloc_438_;
goto v_reusejp_436_;
}
v_reusejp_436_:
{
return v___x_437_;
}
}
}
}
else
{
lean_object* v_a_440_; lean_object* v___x_442_; uint8_t v_isShared_443_; uint8_t v_isSharedCheck_447_; 
lean_dec_ref(v_e_378_);
lean_dec_ref(v_n_377_);
v_a_440_ = lean_ctor_get(v___x_384_, 0);
v_isSharedCheck_447_ = !lean_is_exclusive(v___x_384_);
if (v_isSharedCheck_447_ == 0)
{
v___x_442_ = v___x_384_;
v_isShared_443_ = v_isSharedCheck_447_;
goto v_resetjp_441_;
}
else
{
lean_inc(v_a_440_);
lean_dec(v___x_384_);
v___x_442_ = lean_box(0);
v_isShared_443_ = v_isSharedCheck_447_;
goto v_resetjp_441_;
}
v_resetjp_441_:
{
lean_object* v___x_445_; 
if (v_isShared_443_ == 0)
{
v___x_445_ = v___x_442_;
goto v_reusejp_444_;
}
else
{
lean_object* v_reuseFailAlloc_446_; 
v_reuseFailAlloc_446_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_446_, 0, v_a_440_);
v___x_445_ = v_reuseFailAlloc_446_;
goto v_reusejp_444_;
}
v_reusejp_444_:
{
return v___x_445_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Matrix_matchVecConsPrefix___boxed(lean_object* v_n_448_, lean_object* v_e_449_, lean_object* v_a_450_, lean_object* v_a_451_, lean_object* v_a_452_, lean_object* v_a_453_, lean_object* v_a_454_){
_start:
{
lean_object* v_res_455_; 
v_res_455_ = lp_mathlib_Matrix_matchVecConsPrefix(v_n_448_, v_e_449_, v_a_450_, v_a_451_, v_a_452_, v_a_453_);
lean_dec(v_a_453_);
lean_dec_ref(v_a_452_);
lean_dec(v_a_451_);
lean_dec_ref(v_a_450_);
return v_res_455_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Nat_cast___at___00Matrix_cons__val_spec__0(lean_object* v_a_456_){
_start:
{
lean_object* v___x_457_; 
v___x_457_ = lean_nat_to_int(v_a_456_);
return v___x_457_;
}
}
static lean_object* _init_lp_mathlib_Matrix_cons__val___redArg___closed__4(void){
_start:
{
lean_object* v___x_466_; lean_object* v___x_467_; lean_object* v___x_468_; 
v___x_466_ = ((lean_object*)(lp_mathlib_Matrix_cons__val___redArg___closed__3));
v___x_467_ = ((lean_object*)(lp_mathlib_Matrix_cons__val___redArg___closed__2));
v___x_468_ = l_Lean_Expr_const___override(v___x_467_, v___x_466_);
return v___x_468_;
}
}
static lean_object* _init_lp_mathlib_Matrix_cons__val___redArg___closed__7(void){
_start:
{
lean_object* v___x_472_; lean_object* v___x_473_; lean_object* v___x_474_; 
v___x_472_ = lean_box(0);
v___x_473_ = ((lean_object*)(lp_mathlib_Matrix_cons__val___redArg___closed__6));
v___x_474_ = l_Lean_Expr_const___override(v___x_473_, v___x_472_);
return v___x_474_;
}
}
static lean_object* _init_lp_mathlib_Matrix_cons__val___redArg___closed__8(void){
_start:
{
lean_object* v___x_475_; lean_object* v___x_476_; lean_object* v___x_477_; 
v___x_475_ = lean_obj_once(&lp_mathlib_Matrix_cons__val___redArg___closed__7, &lp_mathlib_Matrix_cons__val___redArg___closed__7_once, _init_lp_mathlib_Matrix_cons__val___redArg___closed__7);
v___x_476_ = lean_obj_once(&lp_mathlib_Matrix_cons__val___redArg___closed__4, &lp_mathlib_Matrix_cons__val___redArg___closed__4_once, _init_lp_mathlib_Matrix_cons__val___redArg___closed__4);
v___x_477_ = l_Lean_Expr_app___override(v___x_476_, v___x_475_);
return v___x_477_;
}
}
static lean_object* _init_lp_mathlib_Matrix_cons__val___redArg___closed__12(void){
_start:
{
lean_object* v___x_483_; lean_object* v___x_484_; lean_object* v___x_485_; 
v___x_483_ = ((lean_object*)(lp_mathlib_Matrix_cons__val___redArg___closed__3));
v___x_484_ = ((lean_object*)(lp_mathlib_Matrix_cons__val___redArg___closed__11));
v___x_485_ = l_Lean_Expr_const___override(v___x_484_, v___x_483_);
return v___x_485_;
}
}
static lean_object* _init_lp_mathlib_Matrix_cons__val___redArg___closed__13(void){
_start:
{
lean_object* v___x_486_; lean_object* v___x_487_; lean_object* v___x_488_; 
v___x_486_ = lean_obj_once(&lp_mathlib_Matrix_cons__val___redArg___closed__7, &lp_mathlib_Matrix_cons__val___redArg___closed__7_once, _init_lp_mathlib_Matrix_cons__val___redArg___closed__7);
v___x_487_ = lean_obj_once(&lp_mathlib_Matrix_cons__val___redArg___closed__12, &lp_mathlib_Matrix_cons__val___redArg___closed__12_once, _init_lp_mathlib_Matrix_cons__val___redArg___closed__12);
v___x_488_ = l_Lean_Expr_app___override(v___x_487_, v___x_486_);
return v___x_488_;
}
}
static lean_object* _init_lp_mathlib_Matrix_cons__val___redArg___closed__16(void){
_start:
{
lean_object* v___x_492_; lean_object* v___x_493_; lean_object* v___x_494_; 
v___x_492_ = lean_box(0);
v___x_493_ = ((lean_object*)(lp_mathlib_Matrix_cons__val___redArg___closed__15));
v___x_494_ = l_Lean_Expr_const___override(v___x_493_, v___x_492_);
return v___x_494_;
}
}
static lean_object* _init_lp_mathlib_Matrix_cons__val___redArg___closed__18(void){
_start:
{
lean_object* v___x_497_; lean_object* v___x_498_; 
v___x_497_ = ((lean_object*)(lp_mathlib_Matrix_cons__val___redArg___closed__17));
v___x_498_ = l_Lean_Expr_lit___override(v___x_497_);
return v___x_498_;
}
}
static lean_object* _init_lp_mathlib_Matrix_cons__val___redArg___closed__19(void){
_start:
{
lean_object* v___x_499_; lean_object* v___x_500_; lean_object* v___x_501_; 
v___x_499_ = lean_obj_once(&lp_mathlib_Matrix_cons__val___redArg___closed__18, &lp_mathlib_Matrix_cons__val___redArg___closed__18_once, _init_lp_mathlib_Matrix_cons__val___redArg___closed__18);
v___x_500_ = lean_obj_once(&lp_mathlib_Matrix_cons__val___redArg___closed__16, &lp_mathlib_Matrix_cons__val___redArg___closed__16_once, _init_lp_mathlib_Matrix_cons__val___redArg___closed__16);
v___x_501_ = l_Lean_Expr_app___override(v___x_500_, v___x_499_);
return v___x_501_;
}
}
static lean_object* _init_lp_mathlib_Matrix_cons__val___redArg___closed__20(void){
_start:
{
lean_object* v___x_502_; lean_object* v___x_503_; lean_object* v___x_504_; 
v___x_502_ = lean_obj_once(&lp_mathlib_Matrix_cons__val___redArg___closed__19, &lp_mathlib_Matrix_cons__val___redArg___closed__19_once, _init_lp_mathlib_Matrix_cons__val___redArg___closed__19);
v___x_503_ = lean_obj_once(&lp_mathlib_Matrix_cons__val___redArg___closed__13, &lp_mathlib_Matrix_cons__val___redArg___closed__13_once, _init_lp_mathlib_Matrix_cons__val___redArg___closed__13);
v___x_504_ = l_Lean_Expr_app___override(v___x_503_, v___x_502_);
return v___x_504_;
}
}
static lean_object* _init_lp_mathlib_Matrix_cons__val___redArg___closed__21(void){
_start:
{
lean_object* v___x_505_; lean_object* v___x_506_; lean_object* v___x_507_; 
v___x_505_ = lean_obj_once(&lp_mathlib_Matrix_cons__val___redArg___closed__20, &lp_mathlib_Matrix_cons__val___redArg___closed__20_once, _init_lp_mathlib_Matrix_cons__val___redArg___closed__20);
v___x_506_ = lean_obj_once(&lp_mathlib_Matrix_cons__val___redArg___closed__8, &lp_mathlib_Matrix_cons__val___redArg___closed__8_once, _init_lp_mathlib_Matrix_cons__val___redArg___closed__8);
v___x_507_ = l_Lean_Expr_app___override(v___x_506_, v___x_505_);
return v___x_507_;
}
}
static lean_object* _init_lp_mathlib_Matrix_cons__val___redArg___closed__24(void){
_start:
{
lean_object* v___x_511_; lean_object* v___x_512_; lean_object* v___x_513_; 
v___x_511_ = lean_box(0);
v___x_512_ = ((lean_object*)(lp_mathlib_Matrix_cons__val___redArg___closed__23));
v___x_513_ = l_Lean_Expr_const___override(v___x_512_, v___x_511_);
return v___x_513_;
}
}
static lean_object* _init_lp_mathlib_Matrix_cons__val___redArg___closed__28(void){
_start:
{
lean_object* v___x_519_; lean_object* v___x_520_; lean_object* v___x_521_; 
v___x_519_ = ((lean_object*)(lp_mathlib_Matrix_cons__val___redArg___closed__3));
v___x_520_ = ((lean_object*)(lp_mathlib_Matrix_cons__val___redArg___closed__27));
v___x_521_ = l_Lean_Expr_const___override(v___x_520_, v___x_519_);
return v___x_521_;
}
}
static lean_object* _init_lp_mathlib_Matrix_cons__val___redArg___closed__31(void){
_start:
{
lean_object* v___x_526_; lean_object* v___x_527_; lean_object* v___x_528_; 
v___x_526_ = lean_box(0);
v___x_527_ = ((lean_object*)(lp_mathlib_Matrix_cons__val___redArg___closed__30));
v___x_528_ = l_Lean_Expr_const___override(v___x_527_, v___x_526_);
return v___x_528_;
}
}
static lean_object* _init_lp_mathlib_Matrix_cons__val___redArg___closed__32(void){
_start:
{
lean_object* v___x_529_; lean_object* v___x_530_; 
v___x_529_ = lean_unsigned_to_nat(0u);
v___x_530_ = lean_nat_to_int(v___x_529_);
return v___x_530_;
}
}
static lean_object* _init_lp_mathlib_Matrix_cons__val___redArg___closed__38(void){
_start:
{
lean_object* v___x_542_; lean_object* v___x_543_; lean_object* v___x_544_; 
v___x_542_ = ((lean_object*)(lp_mathlib_Matrix_cons__val___redArg___closed__37));
v___x_543_ = ((lean_object*)(lp_mathlib_Matrix_cons__val___redArg___closed__35));
v___x_544_ = l_Lean_Expr_const___override(v___x_543_, v___x_542_);
return v___x_544_;
}
}
static lean_object* _init_lp_mathlib_Matrix_cons__val___redArg___closed__39(void){
_start:
{
lean_object* v___x_545_; lean_object* v___x_546_; lean_object* v___x_547_; 
v___x_545_ = lean_obj_once(&lp_mathlib_Matrix_cons__val___redArg___closed__7, &lp_mathlib_Matrix_cons__val___redArg___closed__7_once, _init_lp_mathlib_Matrix_cons__val___redArg___closed__7);
v___x_546_ = lean_obj_once(&lp_mathlib_Matrix_cons__val___redArg___closed__38, &lp_mathlib_Matrix_cons__val___redArg___closed__38_once, _init_lp_mathlib_Matrix_cons__val___redArg___closed__38);
v___x_547_ = l_Lean_Expr_app___override(v___x_546_, v___x_545_);
return v___x_547_;
}
}
static lean_object* _init_lp_mathlib_Matrix_cons__val___redArg___closed__40(void){
_start:
{
lean_object* v___x_548_; lean_object* v___x_549_; lean_object* v___x_550_; 
v___x_548_ = lean_obj_once(&lp_mathlib_Matrix_cons__val___redArg___closed__7, &lp_mathlib_Matrix_cons__val___redArg___closed__7_once, _init_lp_mathlib_Matrix_cons__val___redArg___closed__7);
v___x_549_ = lean_obj_once(&lp_mathlib_Matrix_cons__val___redArg___closed__39, &lp_mathlib_Matrix_cons__val___redArg___closed__39_once, _init_lp_mathlib_Matrix_cons__val___redArg___closed__39);
v___x_550_ = l_Lean_Expr_app___override(v___x_549_, v___x_548_);
return v___x_550_;
}
}
static lean_object* _init_lp_mathlib_Matrix_cons__val___redArg___closed__41(void){
_start:
{
lean_object* v___x_551_; lean_object* v___x_552_; lean_object* v___x_553_; 
v___x_551_ = lean_obj_once(&lp_mathlib_Matrix_cons__val___redArg___closed__7, &lp_mathlib_Matrix_cons__val___redArg___closed__7_once, _init_lp_mathlib_Matrix_cons__val___redArg___closed__7);
v___x_552_ = lean_obj_once(&lp_mathlib_Matrix_cons__val___redArg___closed__40, &lp_mathlib_Matrix_cons__val___redArg___closed__40_once, _init_lp_mathlib_Matrix_cons__val___redArg___closed__40);
v___x_553_ = l_Lean_Expr_app___override(v___x_552_, v___x_551_);
return v___x_553_;
}
}
static lean_object* _init_lp_mathlib_Matrix_cons__val___redArg___closed__44(void){
_start:
{
lean_object* v___x_557_; lean_object* v___x_558_; lean_object* v___x_559_; 
v___x_557_ = ((lean_object*)(lp_mathlib_Matrix_cons__val___redArg___closed__3));
v___x_558_ = ((lean_object*)(lp_mathlib_Matrix_cons__val___redArg___closed__43));
v___x_559_ = l_Lean_Expr_const___override(v___x_558_, v___x_557_);
return v___x_559_;
}
}
static lean_object* _init_lp_mathlib_Matrix_cons__val___redArg___closed__45(void){
_start:
{
lean_object* v___x_560_; lean_object* v___x_561_; lean_object* v___x_562_; 
v___x_560_ = lean_obj_once(&lp_mathlib_Matrix_cons__val___redArg___closed__7, &lp_mathlib_Matrix_cons__val___redArg___closed__7_once, _init_lp_mathlib_Matrix_cons__val___redArg___closed__7);
v___x_561_ = lean_obj_once(&lp_mathlib_Matrix_cons__val___redArg___closed__44, &lp_mathlib_Matrix_cons__val___redArg___closed__44_once, _init_lp_mathlib_Matrix_cons__val___redArg___closed__44);
v___x_562_ = l_Lean_Expr_app___override(v___x_561_, v___x_560_);
return v___x_562_;
}
}
static lean_object* _init_lp_mathlib_Matrix_cons__val___redArg___closed__48(void){
_start:
{
lean_object* v___x_566_; lean_object* v___x_567_; lean_object* v___x_568_; 
v___x_566_ = lean_box(0);
v___x_567_ = ((lean_object*)(lp_mathlib_Matrix_cons__val___redArg___closed__47));
v___x_568_ = l_Lean_Expr_const___override(v___x_567_, v___x_566_);
return v___x_568_;
}
}
static lean_object* _init_lp_mathlib_Matrix_cons__val___redArg___closed__49(void){
_start:
{
lean_object* v___x_569_; lean_object* v___x_570_; lean_object* v___x_571_; 
v___x_569_ = lean_obj_once(&lp_mathlib_Matrix_cons__val___redArg___closed__48, &lp_mathlib_Matrix_cons__val___redArg___closed__48_once, _init_lp_mathlib_Matrix_cons__val___redArg___closed__48);
v___x_570_ = lean_obj_once(&lp_mathlib_Matrix_cons__val___redArg___closed__45, &lp_mathlib_Matrix_cons__val___redArg___closed__45_once, _init_lp_mathlib_Matrix_cons__val___redArg___closed__45);
v___x_571_ = l_Lean_Expr_app___override(v___x_570_, v___x_569_);
return v___x_571_;
}
}
static lean_object* _init_lp_mathlib_Matrix_cons__val___redArg___closed__50(void){
_start:
{
lean_object* v___x_572_; lean_object* v___x_573_; lean_object* v___x_574_; 
v___x_572_ = lean_obj_once(&lp_mathlib_Matrix_cons__val___redArg___closed__49, &lp_mathlib_Matrix_cons__val___redArg___closed__49_once, _init_lp_mathlib_Matrix_cons__val___redArg___closed__49);
v___x_573_ = lean_obj_once(&lp_mathlib_Matrix_cons__val___redArg___closed__41, &lp_mathlib_Matrix_cons__val___redArg___closed__41_once, _init_lp_mathlib_Matrix_cons__val___redArg___closed__41);
v___x_574_ = l_Lean_Expr_app___override(v___x_573_, v___x_572_);
return v___x_574_;
}
}
static lean_object* _init_lp_mathlib_Matrix_cons__val___redArg___closed__51(void){
_start:
{
lean_object* v___x_575_; lean_object* v___x_576_; lean_object* v___x_577_; 
v___x_575_ = lean_obj_once(&lp_mathlib_Matrix_cons__val___redArg___closed__7, &lp_mathlib_Matrix_cons__val___redArg___closed__7_once, _init_lp_mathlib_Matrix_cons__val___redArg___closed__7);
v___x_576_ = lean_obj_once(&lp_mathlib_Matrix_cons__val___redArg___closed__28, &lp_mathlib_Matrix_cons__val___redArg___closed__28_once, _init_lp_mathlib_Matrix_cons__val___redArg___closed__28);
v___x_577_ = l_Lean_Expr_app___override(v___x_576_, v___x_575_);
return v___x_577_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Matrix_cons__val___redArg(lean_object* v_e_578_, lean_object* v_a_579_, lean_object* v_a_580_, lean_object* v_a_581_, lean_object* v_a_582_){
_start:
{
lean_object* v___x_587_; 
v___x_587_ = l_Lean_Meta_whnfR(v_e_578_, v_a_579_, v_a_580_, v_a_581_, v_a_582_);
if (lean_obj_tag(v___x_587_) == 0)
{
lean_object* v_a_588_; lean_object* v___x_590_; uint8_t v_isShared_591_; uint8_t v_isSharedCheck_769_; 
v_a_588_ = lean_ctor_get(v___x_587_, 0);
v_isSharedCheck_769_ = !lean_is_exclusive(v___x_587_);
if (v_isSharedCheck_769_ == 0)
{
v___x_590_ = v___x_587_;
v_isShared_591_ = v_isSharedCheck_769_;
goto v_resetjp_589_;
}
else
{
lean_inc(v_a_588_);
lean_dec(v___x_587_);
v___x_590_ = lean_box(0);
v_isShared_591_ = v_isSharedCheck_769_;
goto v_resetjp_589_;
}
v_resetjp_589_:
{
lean_object* v___x_597_; uint8_t v___x_598_; 
v___x_597_ = l_Lean_Expr_cleanupAnnotations(v_a_588_);
v___x_598_ = l_Lean_Expr_isApp(v___x_597_);
if (v___x_598_ == 0)
{
lean_dec_ref(v___x_597_);
goto v___jp_592_;
}
else
{
lean_object* v_arg_599_; lean_object* v___x_600_; uint8_t v___x_601_; 
v_arg_599_ = lean_ctor_get(v___x_597_, 1);
lean_inc_ref(v_arg_599_);
v___x_600_ = l_Lean_Expr_appFnCleanup___redArg(v___x_597_);
v___x_601_ = l_Lean_Expr_isApp(v___x_600_);
if (v___x_601_ == 0)
{
lean_dec_ref(v___x_600_);
lean_dec_ref(v_arg_599_);
goto v___jp_592_;
}
else
{
lean_object* v_arg_602_; lean_object* v___x_603_; uint8_t v___x_604_; 
v_arg_602_ = lean_ctor_get(v___x_600_, 1);
lean_inc_ref(v_arg_602_);
v___x_603_ = l_Lean_Expr_appFnCleanup___redArg(v___x_600_);
v___x_604_ = l_Lean_Expr_isApp(v___x_603_);
if (v___x_604_ == 0)
{
lean_dec_ref(v___x_603_);
lean_dec_ref(v_arg_602_);
lean_dec_ref(v_arg_599_);
goto v___jp_592_;
}
else
{
lean_object* v_arg_605_; lean_object* v___x_606_; uint8_t v___x_607_; 
v_arg_605_ = lean_ctor_get(v___x_603_, 1);
lean_inc_ref(v_arg_605_);
v___x_606_ = l_Lean_Expr_appFnCleanup___redArg(v___x_603_);
v___x_607_ = l_Lean_Expr_isApp(v___x_606_);
if (v___x_607_ == 0)
{
lean_dec_ref(v___x_606_);
lean_dec_ref(v_arg_605_);
lean_dec_ref(v_arg_602_);
lean_dec_ref(v_arg_599_);
goto v___jp_592_;
}
else
{
lean_object* v_arg_608_; lean_object* v___x_609_; uint8_t v___x_610_; 
v_arg_608_ = lean_ctor_get(v___x_606_, 1);
lean_inc_ref(v_arg_608_);
v___x_609_ = l_Lean_Expr_appFnCleanup___redArg(v___x_606_);
v___x_610_ = l_Lean_Expr_isApp(v___x_609_);
if (v___x_610_ == 0)
{
lean_dec_ref(v___x_609_);
lean_dec_ref(v_arg_608_);
lean_dec_ref(v_arg_605_);
lean_dec_ref(v_arg_602_);
lean_dec_ref(v_arg_599_);
goto v___jp_592_;
}
else
{
lean_object* v___x_611_; lean_object* v___x_612_; uint8_t v___x_613_; 
v___x_611_ = l_Lean_Expr_appFnCleanup___redArg(v___x_609_);
v___x_612_ = ((lean_object*)(lp_mathlib_Matrix___aux__Mathlib__Data__Fin__VecNotation______macroRules__Matrix__vecNotation__1___closed__14));
v___x_613_ = l_Lean_Expr_isConstOf(v___x_611_, v___x_612_);
lean_dec_ref(v___x_611_);
if (v___x_613_ == 0)
{
lean_dec_ref(v_arg_608_);
lean_dec_ref(v_arg_605_);
lean_dec_ref(v_arg_602_);
lean_dec_ref(v_arg_599_);
goto v___jp_592_;
}
else
{
lean_object* v___x_614_; 
lean_del_object(v___x_590_);
v___x_614_ = l_Lean_Expr_int_x3f(v_arg_599_);
if (lean_obj_tag(v___x_614_) == 1)
{
lean_object* v_val_615_; lean_object* v___x_617_; uint8_t v_isShared_618_; uint8_t v_isSharedCheck_766_; 
v_val_615_ = lean_ctor_get(v___x_614_, 0);
v_isSharedCheck_766_ = !lean_is_exclusive(v___x_614_);
if (v_isSharedCheck_766_ == 0)
{
v___x_617_ = v___x_614_;
v_isShared_618_ = v_isSharedCheck_766_;
goto v_resetjp_616_;
}
else
{
lean_inc(v_val_615_);
lean_dec(v___x_614_);
v___x_617_ = lean_box(0);
v_isShared_618_ = v_isSharedCheck_766_;
goto v_resetjp_616_;
}
v_resetjp_616_:
{
lean_object* v___x_619_; 
v___x_619_ = lp_mathlib_Matrix_matchVecConsPrefix(v_arg_608_, v_arg_602_, v_a_579_, v_a_580_, v_a_581_, v_a_582_);
if (lean_obj_tag(v___x_619_) == 0)
{
lean_object* v_a_620_; lean_object* v_snd_621_; lean_object* v_fst_622_; lean_object* v_fst_623_; lean_object* v_snd_624_; lean_object* v___x_626_; uint8_t v_isShared_627_; uint8_t v_isSharedCheck_757_; 
v_a_620_ = lean_ctor_get(v___x_619_, 0);
lean_inc(v_a_620_);
lean_dec_ref_known(v___x_619_, 1);
v_snd_621_ = lean_ctor_get(v_a_620_, 1);
lean_inc(v_snd_621_);
v_fst_622_ = lean_ctor_get(v_a_620_, 0);
lean_inc(v_fst_622_);
lean_dec(v_a_620_);
v_fst_623_ = lean_ctor_get(v_snd_621_, 0);
v_snd_624_ = lean_ctor_get(v_snd_621_, 1);
v_isSharedCheck_757_ = !lean_is_exclusive(v_snd_621_);
if (v_isSharedCheck_757_ == 0)
{
v___x_626_ = v_snd_621_;
v_isShared_627_ = v_isSharedCheck_757_;
goto v_resetjp_625_;
}
else
{
lean_inc(v_snd_624_);
lean_inc(v_fst_623_);
lean_dec(v_snd_621_);
v___x_626_ = lean_box(0);
v_isShared_627_ = v_isSharedCheck_757_;
goto v_resetjp_625_;
}
v_resetjp_625_:
{
lean_object* v___x_628_; 
lean_inc(v_fst_623_);
v___x_628_ = l_Lean_Meta_whnfD(v_fst_623_, v_a_579_, v_a_580_, v_a_581_, v_a_582_);
if (lean_obj_tag(v___x_628_) == 0)
{
lean_object* v_a_629_; lean_object* v___x_631_; uint8_t v_isShared_632_; uint8_t v_isSharedCheck_748_; 
v_a_629_ = lean_ctor_get(v___x_628_, 0);
v_isSharedCheck_748_ = !lean_is_exclusive(v___x_628_);
if (v_isSharedCheck_748_ == 0)
{
v___x_631_ = v___x_628_;
v_isShared_632_ = v_isSharedCheck_748_;
goto v_resetjp_630_;
}
else
{
lean_inc(v_a_629_);
lean_dec(v___x_628_);
v___x_631_ = lean_box(0);
v_isShared_632_ = v_isSharedCheck_748_;
goto v_resetjp_630_;
}
v_resetjp_630_:
{
lean_object* v___x_634_; 
if (v_isShared_627_ == 0)
{
lean_ctor_set_tag(v___x_626_, 1);
lean_ctor_set(v___x_626_, 1, v_fst_622_);
lean_ctor_set(v___x_626_, 0, v_arg_605_);
v___x_634_ = v___x_626_;
goto v_reusejp_633_;
}
else
{
lean_object* v_reuseFailAlloc_747_; 
v_reuseFailAlloc_747_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_747_, 0, v_arg_605_);
lean_ctor_set(v_reuseFailAlloc_747_, 1, v_fst_622_);
v___x_634_ = v_reuseFailAlloc_747_;
goto v_reusejp_633_;
}
v_reusejp_633_:
{
lean_object* v___y_636_; lean_object* v_wrapped__i_637_; lean_object* v___y_638_; lean_object* v___y_639_; lean_object* v___y_640_; lean_object* v___y_641_; lean_object* v_fst_689_; lean_object* v_snd_690_; lean_object* v___y_691_; lean_object* v___y_692_; lean_object* v___y_693_; lean_object* v___y_694_; lean_object* v_fst_702_; uint8_t v_fst_703_; lean_object* v_snd_704_; lean_object* v___y_705_; lean_object* v___y_706_; lean_object* v___y_707_; lean_object* v___y_708_; lean_object* v___y_718_; lean_object* v___y_719_; lean_object* v___y_720_; lean_object* v___y_721_; 
if (lean_obj_tag(v_a_629_) == 9)
{
lean_object* v_a_740_; 
v_a_740_ = lean_ctor_get(v_a_629_, 0);
if (lean_obj_tag(v_a_740_) == 0)
{
lean_object* v_val_741_; lean_object* v___x_742_; lean_object* v___x_743_; lean_object* v___x_744_; lean_object* v___x_745_; lean_object* v___x_746_; 
lean_dec(v_fst_623_);
v_val_741_ = lean_ctor_get(v_a_740_, 0);
lean_inc(v_val_741_);
v___x_742_ = lean_obj_once(&lp_mathlib_Matrix_cons__val___redArg___closed__51, &lp_mathlib_Matrix_cons__val___redArg___closed__51_once, _init_lp_mathlib_Matrix_cons__val___redArg___closed__51);
lean_inc_ref(v_a_629_);
v___x_743_ = l_Lean_Expr_app___override(v___x_742_, v_a_629_);
v___x_744_ = lean_obj_once(&lp_mathlib_Matrix_cons__val___redArg___closed__16, &lp_mathlib_Matrix_cons__val___redArg___closed__16_once, _init_lp_mathlib_Matrix_cons__val___redArg___closed__16);
v___x_745_ = l_Lean_Expr_app___override(v___x_744_, v_a_629_);
v___x_746_ = l_Lean_Expr_app___override(v___x_743_, v___x_745_);
v_fst_689_ = v_val_741_;
v_snd_690_ = v___x_746_;
v___y_691_ = v_a_579_;
v___y_692_ = v_a_580_;
v___y_693_ = v_a_581_;
v___y_694_ = v_a_582_;
goto v___jp_688_;
}
else
{
v___y_718_ = v_a_579_;
v___y_719_ = v_a_580_;
v___y_720_ = v_a_581_;
v___y_721_ = v_a_582_;
goto v___jp_717_;
}
}
else
{
v___y_718_ = v_a_579_;
v___y_719_ = v_a_580_;
v___y_720_ = v_a_581_;
v___y_721_ = v_a_582_;
goto v___jp_717_;
}
v___jp_635_:
{
lean_object* v___x_642_; uint8_t v___x_643_; 
v___x_642_ = l_List_lengthTR___redArg(v___x_634_);
v___x_643_ = lean_nat_dec_lt(v_wrapped__i_637_, v___x_642_);
if (v___x_643_ == 0)
{
lean_object* v___x_644_; lean_object* v___x_645_; lean_object* v___x_646_; 
lean_dec_ref(v___x_634_);
lean_del_object(v___x_631_);
v___x_644_ = lean_obj_once(&lp_mathlib_Matrix_cons__val___redArg___closed__21, &lp_mathlib_Matrix_cons__val___redArg___closed__21_once, _init_lp_mathlib_Matrix_cons__val___redArg___closed__21);
lean_inc_ref(v___y_636_);
v___x_645_ = l_Lean_Expr_app___override(v___x_644_, v___y_636_);
v___x_646_ = lp_Qq_Qq_synthInstanceQ___redArg(v___x_645_, v___y_638_, v___y_639_, v___y_640_, v___y_641_);
if (lean_obj_tag(v___x_646_) == 0)
{
lean_object* v_a_647_; lean_object* v___x_649_; uint8_t v_isShared_650_; uint8_t v_isSharedCheck_671_; 
v_a_647_ = lean_ctor_get(v___x_646_, 0);
v_isSharedCheck_671_ = !lean_is_exclusive(v___x_646_);
if (v_isSharedCheck_671_ == 0)
{
v___x_649_ = v___x_646_;
v_isShared_650_ = v_isSharedCheck_671_;
goto v_resetjp_648_;
}
else
{
lean_inc(v_a_647_);
lean_dec(v___x_646_);
v___x_649_ = lean_box(0);
v_isShared_650_ = v_isSharedCheck_671_;
goto v_resetjp_648_;
}
v_resetjp_648_:
{
lean_object* v___x_651_; lean_object* v___x_652_; lean_object* v___x_653_; lean_object* v___x_654_; lean_object* v___x_655_; lean_object* v___x_656_; lean_object* v___x_657_; lean_object* v___x_658_; lean_object* v___x_659_; lean_object* v___x_660_; lean_object* v___x_661_; lean_object* v___x_662_; lean_object* v___x_663_; lean_object* v___x_665_; 
v___x_651_ = lean_nat_sub(v_wrapped__i_637_, v___x_642_);
lean_dec(v___x_642_);
lean_dec(v_wrapped__i_637_);
v___x_652_ = l_Lean_mkRawNatLit(v___x_651_);
v___x_653_ = lean_obj_once(&lp_mathlib_Matrix_cons__val___redArg___closed__24, &lp_mathlib_Matrix_cons__val___redArg___closed__24_once, _init_lp_mathlib_Matrix_cons__val___redArg___closed__24);
lean_inc_ref(v___y_636_);
v___x_654_ = l_Lean_Expr_app___override(v___x_653_, v___y_636_);
v___x_655_ = lean_obj_once(&lp_mathlib_Matrix_cons__val___redArg___closed__28, &lp_mathlib_Matrix_cons__val___redArg___closed__28_once, _init_lp_mathlib_Matrix_cons__val___redArg___closed__28);
v___x_656_ = l_Lean_Expr_app___override(v___x_655_, v___x_654_);
lean_inc_ref(v___x_652_);
v___x_657_ = l_Lean_Expr_app___override(v___x_656_, v___x_652_);
v___x_658_ = lean_obj_once(&lp_mathlib_Matrix_cons__val___redArg___closed__31, &lp_mathlib_Matrix_cons__val___redArg___closed__31_once, _init_lp_mathlib_Matrix_cons__val___redArg___closed__31);
v___x_659_ = l_Lean_Expr_app___override(v___x_658_, v___y_636_);
v___x_660_ = l_Lean_Expr_app___override(v___x_659_, v_a_647_);
v___x_661_ = l_Lean_Expr_app___override(v___x_660_, v___x_652_);
v___x_662_ = l_Lean_Expr_app___override(v___x_657_, v___x_661_);
v___x_663_ = l_Lean_Expr_app___override(v_snd_624_, v___x_662_);
if (v_isShared_618_ == 0)
{
lean_ctor_set(v___x_617_, 0, v___x_663_);
v___x_665_ = v___x_617_;
goto v_reusejp_664_;
}
else
{
lean_object* v_reuseFailAlloc_670_; 
v_reuseFailAlloc_670_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_670_, 0, v___x_663_);
v___x_665_ = v_reuseFailAlloc_670_;
goto v_reusejp_664_;
}
v_reusejp_664_:
{
lean_object* v___x_666_; lean_object* v___x_668_; 
v___x_666_ = lean_alloc_ctor(2, 1, 0);
lean_ctor_set(v___x_666_, 0, v___x_665_);
if (v_isShared_650_ == 0)
{
lean_ctor_set(v___x_649_, 0, v___x_666_);
v___x_668_ = v___x_649_;
goto v_reusejp_667_;
}
else
{
lean_object* v_reuseFailAlloc_669_; 
v_reuseFailAlloc_669_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_669_, 0, v___x_666_);
v___x_668_ = v_reuseFailAlloc_669_;
goto v_reusejp_667_;
}
v_reusejp_667_:
{
return v___x_668_;
}
}
}
}
else
{
lean_object* v_a_672_; lean_object* v___x_674_; uint8_t v_isShared_675_; uint8_t v_isSharedCheck_679_; 
lean_dec(v___x_642_);
lean_dec(v_wrapped__i_637_);
lean_dec_ref(v___y_636_);
lean_dec(v_snd_624_);
lean_del_object(v___x_617_);
v_a_672_ = lean_ctor_get(v___x_646_, 0);
v_isSharedCheck_679_ = !lean_is_exclusive(v___x_646_);
if (v_isSharedCheck_679_ == 0)
{
v___x_674_ = v___x_646_;
v_isShared_675_ = v_isSharedCheck_679_;
goto v_resetjp_673_;
}
else
{
lean_inc(v_a_672_);
lean_dec(v___x_646_);
v___x_674_ = lean_box(0);
v_isShared_675_ = v_isSharedCheck_679_;
goto v_resetjp_673_;
}
v_resetjp_673_:
{
lean_object* v___x_677_; 
if (v_isShared_675_ == 0)
{
v___x_677_ = v___x_674_;
goto v_reusejp_676_;
}
else
{
lean_object* v_reuseFailAlloc_678_; 
v_reuseFailAlloc_678_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_678_, 0, v_a_672_);
v___x_677_ = v_reuseFailAlloc_678_;
goto v_reusejp_676_;
}
v_reusejp_676_:
{
return v___x_677_;
}
}
}
}
else
{
lean_object* v___x_680_; lean_object* v___x_682_; 
lean_dec(v___x_642_);
lean_dec_ref(v___y_636_);
lean_dec(v_snd_624_);
v___x_680_ = l_List_get___redArg(v___x_634_, v_wrapped__i_637_);
lean_dec_ref(v___x_634_);
if (v_isShared_618_ == 0)
{
lean_ctor_set(v___x_617_, 0, v___x_680_);
v___x_682_ = v___x_617_;
goto v_reusejp_681_;
}
else
{
lean_object* v_reuseFailAlloc_687_; 
v_reuseFailAlloc_687_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_687_, 0, v___x_680_);
v___x_682_ = v_reuseFailAlloc_687_;
goto v_reusejp_681_;
}
v_reusejp_681_:
{
lean_object* v___x_683_; lean_object* v___x_685_; 
v___x_683_ = lean_alloc_ctor(2, 1, 0);
lean_ctor_set(v___x_683_, 0, v___x_682_);
if (v_isShared_632_ == 0)
{
lean_ctor_set(v___x_631_, 0, v___x_683_);
v___x_685_ = v___x_631_;
goto v_reusejp_684_;
}
else
{
lean_object* v_reuseFailAlloc_686_; 
v_reuseFailAlloc_686_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_686_, 0, v___x_683_);
v___x_685_ = v_reuseFailAlloc_686_;
goto v_reusejp_684_;
}
v_reusejp_684_:
{
return v___x_685_;
}
}
}
}
v___jp_688_:
{
lean_object* v___x_695_; lean_object* v___x_696_; lean_object* v___x_697_; lean_object* v___x_698_; lean_object* v___x_699_; lean_object* v___x_700_; 
v___x_695_ = l_List_lengthTR___redArg(v___x_634_);
v___x_696_ = lean_nat_to_int(v___x_695_);
v___x_697_ = lean_nat_to_int(v_fst_689_);
v___x_698_ = lean_int_add(v___x_696_, v___x_697_);
lean_dec(v___x_697_);
lean_dec(v___x_696_);
v___x_699_ = lean_int_emod(v_val_615_, v___x_698_);
lean_dec(v___x_698_);
lean_dec(v_val_615_);
v___x_700_ = l_Int_toNat(v___x_699_);
lean_dec(v___x_699_);
v___y_636_ = v_snd_690_;
v_wrapped__i_637_ = v___x_700_;
v___y_638_ = v___y_691_;
v___y_639_ = v___y_692_;
v___y_640_ = v___y_693_;
v___y_641_ = v___y_694_;
goto v___jp_635_;
}
v___jp_701_:
{
if (v_fst_703_ == 0)
{
v_fst_689_ = v_fst_702_;
v_snd_690_ = v_snd_704_;
v___y_691_ = v___y_705_;
v___y_692_ = v___y_706_;
v___y_693_ = v___y_707_;
v___y_694_ = v___y_708_;
goto v___jp_688_;
}
else
{
lean_object* v___x_709_; uint8_t v___x_710_; 
v___x_709_ = lean_obj_once(&lp_mathlib_Matrix_cons__val___redArg___closed__32, &lp_mathlib_Matrix_cons__val___redArg___closed__32_once, _init_lp_mathlib_Matrix_cons__val___redArg___closed__32);
v___x_710_ = lean_int_dec_le(v___x_709_, v_val_615_);
if (v___x_710_ == 0)
{
lean_dec_ref(v_snd_704_);
lean_dec(v_fst_702_);
lean_dec_ref(v___x_634_);
lean_del_object(v___x_631_);
lean_dec(v_snd_624_);
lean_del_object(v___x_617_);
lean_dec(v_val_615_);
goto v___jp_584_;
}
else
{
lean_object* v___x_711_; lean_object* v___x_712_; lean_object* v___x_713_; lean_object* v___x_714_; uint8_t v___x_715_; 
v___x_711_ = l_List_lengthTR___redArg(v___x_634_);
v___x_712_ = lean_nat_to_int(v___x_711_);
v___x_713_ = lean_nat_to_int(v_fst_702_);
v___x_714_ = lean_int_add(v___x_712_, v___x_713_);
lean_dec(v___x_713_);
lean_dec(v___x_712_);
v___x_715_ = lean_int_dec_lt(v_val_615_, v___x_714_);
lean_dec(v___x_714_);
if (v___x_715_ == 0)
{
lean_dec_ref(v_snd_704_);
lean_dec_ref(v___x_634_);
lean_del_object(v___x_631_);
lean_dec(v_snd_624_);
lean_del_object(v___x_617_);
lean_dec(v_val_615_);
goto v___jp_584_;
}
else
{
lean_object* v___x_716_; 
v___x_716_ = l_Int_toNat(v_val_615_);
lean_dec(v_val_615_);
v___y_636_ = v_snd_704_;
v_wrapped__i_637_ = v___x_716_;
v___y_638_ = v___y_705_;
v___y_639_ = v___y_706_;
v___y_640_ = v___y_707_;
v___y_641_ = v___y_708_;
goto v___jp_635_;
}
}
}
}
v___jp_717_:
{
lean_object* v___x_722_; 
v___x_722_ = l_Lean_Meta_isOffset_x3f(v_a_629_, v___y_718_, v___y_719_, v___y_720_, v___y_721_);
if (lean_obj_tag(v___x_722_) == 0)
{
lean_object* v_a_723_; 
v_a_723_ = lean_ctor_get(v___x_722_, 0);
lean_inc(v_a_723_);
lean_dec_ref_known(v___x_722_, 1);
if (lean_obj_tag(v_a_723_) == 1)
{
lean_object* v_val_724_; lean_object* v_fst_725_; lean_object* v_snd_726_; lean_object* v___x_727_; lean_object* v___x_728_; lean_object* v___x_729_; lean_object* v___x_730_; 
lean_dec(v_fst_623_);
v_val_724_ = lean_ctor_get(v_a_723_, 0);
lean_inc(v_val_724_);
lean_dec_ref_known(v_a_723_, 1);
v_fst_725_ = lean_ctor_get(v_val_724_, 0);
lean_inc(v_fst_725_);
v_snd_726_ = lean_ctor_get(v_val_724_, 1);
lean_inc_n(v_snd_726_, 2);
lean_dec(v_val_724_);
v___x_727_ = lean_obj_once(&lp_mathlib_Matrix_cons__val___redArg___closed__50, &lp_mathlib_Matrix_cons__val___redArg___closed__50_once, _init_lp_mathlib_Matrix_cons__val___redArg___closed__50);
v___x_728_ = l_Lean_Expr_app___override(v___x_727_, v_fst_725_);
v___x_729_ = l_Lean_mkNatLit(v_snd_726_);
v___x_730_ = l_Lean_Expr_app___override(v___x_728_, v___x_729_);
v_fst_702_ = v_snd_726_;
v_fst_703_ = v___x_613_;
v_snd_704_ = v___x_730_;
v___y_705_ = v___y_718_;
v___y_706_ = v___y_719_;
v___y_707_ = v___y_720_;
v___y_708_ = v___y_721_;
goto v___jp_701_;
}
else
{
lean_object* v___x_731_; 
lean_dec(v_a_723_);
v___x_731_ = lean_unsigned_to_nat(0u);
v_fst_702_ = v___x_731_;
v_fst_703_ = v___x_613_;
v_snd_704_ = v_fst_623_;
v___y_705_ = v___y_718_;
v___y_706_ = v___y_719_;
v___y_707_ = v___y_720_;
v___y_708_ = v___y_721_;
goto v___jp_701_;
}
}
else
{
lean_object* v_a_732_; lean_object* v___x_734_; uint8_t v_isShared_735_; uint8_t v_isSharedCheck_739_; 
lean_dec_ref(v___x_634_);
lean_del_object(v___x_631_);
lean_dec(v_snd_624_);
lean_dec(v_fst_623_);
lean_del_object(v___x_617_);
lean_dec(v_val_615_);
v_a_732_ = lean_ctor_get(v___x_722_, 0);
v_isSharedCheck_739_ = !lean_is_exclusive(v___x_722_);
if (v_isSharedCheck_739_ == 0)
{
v___x_734_ = v___x_722_;
v_isShared_735_ = v_isSharedCheck_739_;
goto v_resetjp_733_;
}
else
{
lean_inc(v_a_732_);
lean_dec(v___x_722_);
v___x_734_ = lean_box(0);
v_isShared_735_ = v_isSharedCheck_739_;
goto v_resetjp_733_;
}
v_resetjp_733_:
{
lean_object* v___x_737_; 
if (v_isShared_735_ == 0)
{
v___x_737_ = v___x_734_;
goto v_reusejp_736_;
}
else
{
lean_object* v_reuseFailAlloc_738_; 
v_reuseFailAlloc_738_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_738_, 0, v_a_732_);
v___x_737_ = v_reuseFailAlloc_738_;
goto v_reusejp_736_;
}
v_reusejp_736_:
{
return v___x_737_;
}
}
}
}
}
}
}
else
{
lean_object* v_a_749_; lean_object* v___x_751_; uint8_t v_isShared_752_; uint8_t v_isSharedCheck_756_; 
lean_del_object(v___x_626_);
lean_dec(v_snd_624_);
lean_dec(v_fst_623_);
lean_dec(v_fst_622_);
lean_del_object(v___x_617_);
lean_dec(v_val_615_);
lean_dec_ref(v_arg_605_);
v_a_749_ = lean_ctor_get(v___x_628_, 0);
v_isSharedCheck_756_ = !lean_is_exclusive(v___x_628_);
if (v_isSharedCheck_756_ == 0)
{
v___x_751_ = v___x_628_;
v_isShared_752_ = v_isSharedCheck_756_;
goto v_resetjp_750_;
}
else
{
lean_inc(v_a_749_);
lean_dec(v___x_628_);
v___x_751_ = lean_box(0);
v_isShared_752_ = v_isSharedCheck_756_;
goto v_resetjp_750_;
}
v_resetjp_750_:
{
lean_object* v___x_754_; 
if (v_isShared_752_ == 0)
{
v___x_754_ = v___x_751_;
goto v_reusejp_753_;
}
else
{
lean_object* v_reuseFailAlloc_755_; 
v_reuseFailAlloc_755_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_755_, 0, v_a_749_);
v___x_754_ = v_reuseFailAlloc_755_;
goto v_reusejp_753_;
}
v_reusejp_753_:
{
return v___x_754_;
}
}
}
}
}
else
{
lean_object* v_a_758_; lean_object* v___x_760_; uint8_t v_isShared_761_; uint8_t v_isSharedCheck_765_; 
lean_del_object(v___x_617_);
lean_dec(v_val_615_);
lean_dec_ref(v_arg_605_);
v_a_758_ = lean_ctor_get(v___x_619_, 0);
v_isSharedCheck_765_ = !lean_is_exclusive(v___x_619_);
if (v_isSharedCheck_765_ == 0)
{
v___x_760_ = v___x_619_;
v_isShared_761_ = v_isSharedCheck_765_;
goto v_resetjp_759_;
}
else
{
lean_inc(v_a_758_);
lean_dec(v___x_619_);
v___x_760_ = lean_box(0);
v_isShared_761_ = v_isSharedCheck_765_;
goto v_resetjp_759_;
}
v_resetjp_759_:
{
lean_object* v___x_763_; 
if (v_isShared_761_ == 0)
{
v___x_763_ = v___x_760_;
goto v_reusejp_762_;
}
else
{
lean_object* v_reuseFailAlloc_764_; 
v_reuseFailAlloc_764_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_764_, 0, v_a_758_);
v___x_763_ = v_reuseFailAlloc_764_;
goto v_reusejp_762_;
}
v_reusejp_762_:
{
return v___x_763_;
}
}
}
}
}
else
{
lean_object* v___x_767_; lean_object* v___x_768_; 
lean_dec(v___x_614_);
lean_dec_ref(v_arg_608_);
lean_dec_ref(v_arg_605_);
lean_dec_ref(v_arg_602_);
v___x_767_ = ((lean_object*)(lp_mathlib_Matrix_cons__val___redArg___closed__0));
v___x_768_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_768_, 0, v___x_767_);
return v___x_768_;
}
}
}
}
}
}
}
v___jp_592_:
{
lean_object* v___x_593_; lean_object* v___x_595_; 
v___x_593_ = ((lean_object*)(lp_mathlib_Matrix_cons__val___redArg___closed__0));
if (v_isShared_591_ == 0)
{
lean_ctor_set(v___x_590_, 0, v___x_593_);
v___x_595_ = v___x_590_;
goto v_reusejp_594_;
}
else
{
lean_object* v_reuseFailAlloc_596_; 
v_reuseFailAlloc_596_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_596_, 0, v___x_593_);
v___x_595_ = v_reuseFailAlloc_596_;
goto v_reusejp_594_;
}
v_reusejp_594_:
{
return v___x_595_;
}
}
}
}
else
{
lean_object* v_a_770_; lean_object* v___x_772_; uint8_t v_isShared_773_; uint8_t v_isSharedCheck_777_; 
v_a_770_ = lean_ctor_get(v___x_587_, 0);
v_isSharedCheck_777_ = !lean_is_exclusive(v___x_587_);
if (v_isSharedCheck_777_ == 0)
{
v___x_772_ = v___x_587_;
v_isShared_773_ = v_isSharedCheck_777_;
goto v_resetjp_771_;
}
else
{
lean_inc(v_a_770_);
lean_dec(v___x_587_);
v___x_772_ = lean_box(0);
v_isShared_773_ = v_isSharedCheck_777_;
goto v_resetjp_771_;
}
v_resetjp_771_:
{
lean_object* v___x_775_; 
if (v_isShared_773_ == 0)
{
v___x_775_ = v___x_772_;
goto v_reusejp_774_;
}
else
{
lean_object* v_reuseFailAlloc_776_; 
v_reuseFailAlloc_776_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_776_, 0, v_a_770_);
v___x_775_ = v_reuseFailAlloc_776_;
goto v_reusejp_774_;
}
v_reusejp_774_:
{
return v___x_775_;
}
}
}
v___jp_584_:
{
lean_object* v___x_585_; lean_object* v___x_586_; 
v___x_585_ = ((lean_object*)(lp_mathlib_Matrix_cons__val___redArg___closed__0));
v___x_586_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_586_, 0, v___x_585_);
return v___x_586_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Matrix_cons__val___redArg___boxed(lean_object* v_e_778_, lean_object* v_a_779_, lean_object* v_a_780_, lean_object* v_a_781_, lean_object* v_a_782_, lean_object* v_a_783_){
_start:
{
lean_object* v_res_784_; 
v_res_784_ = lp_mathlib_Matrix_cons__val___redArg(v_e_778_, v_a_779_, v_a_780_, v_a_781_, v_a_782_);
lean_dec(v_a_782_);
lean_dec_ref(v_a_781_);
lean_dec(v_a_780_);
lean_dec_ref(v_a_779_);
return v_res_784_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Matrix_cons__val(lean_object* v_e_785_, lean_object* v_a_786_, lean_object* v_a_787_, lean_object* v_a_788_, lean_object* v_a_789_, lean_object* v_a_790_, lean_object* v_a_791_, lean_object* v_a_792_){
_start:
{
lean_object* v___x_794_; 
v___x_794_ = lp_mathlib_Matrix_cons__val___redArg(v_e_785_, v_a_789_, v_a_790_, v_a_791_, v_a_792_);
return v___x_794_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Matrix_cons__val___boxed(lean_object* v_e_795_, lean_object* v_a_796_, lean_object* v_a_797_, lean_object* v_a_798_, lean_object* v_a_799_, lean_object* v_a_800_, lean_object* v_a_801_, lean_object* v_a_802_, lean_object* v_a_803_){
_start:
{
lean_object* v_res_804_; 
v_res_804_ = lp_mathlib_Matrix_cons__val(v_e_795_, v_a_796_, v_a_797_, v_a_798_, v_a_799_, v_a_800_, v_a_801_, v_a_802_);
lean_dec(v_a_802_);
lean_dec_ref(v_a_801_);
lean_dec(v_a_800_);
lean_dec_ref(v_a_799_);
lean_dec(v_a_798_);
lean_dec_ref(v_a_797_);
lean_dec(v_a_796_);
return v_res_804_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Data_Fin_VecNotation_0__PiFin_mkLiteralQ_loop(lean_object* v_u_805_, lean_object* v_00_u03b1_806_, lean_object* v_n_807_, lean_object* v_elems_808_, lean_object* v_i_809_, lean_object* v_rest_810_){
_start:
{
uint8_t v___x_811_; 
v___x_811_ = lean_nat_dec_lt(v_i_809_, v_n_807_);
if (v___x_811_ == 0)
{
lean_dec(v_i_809_);
lean_dec_ref(v_elems_808_);
lean_dec_ref(v_00_u03b1_806_);
lean_dec(v_u_805_);
return v_rest_810_;
}
else
{
lean_object* v___x_812_; lean_object* v___x_813_; lean_object* v___x_814_; lean_object* v_a_815_; lean_object* v___x_816_; lean_object* v___x_817_; lean_object* v___x_818_; lean_object* v___x_819_; lean_object* v___x_820_; lean_object* v___x_821_; lean_object* v___x_822_; lean_object* v___x_823_; lean_object* v___x_824_; 
v___x_812_ = lean_unsigned_to_nat(1u);
v___x_813_ = lean_nat_add(v_i_809_, v___x_812_);
v___x_814_ = lean_nat_sub(v_n_807_, v___x_813_);
lean_inc_ref(v_elems_808_);
v_a_815_ = lean_apply_1(v_elems_808_, v___x_814_);
v___x_816_ = lean_box(0);
v___x_817_ = ((lean_object*)(lp_mathlib_Matrix___aux__Mathlib__Data__Fin__VecNotation______macroRules__Matrix__vecNotation__1___closed__14));
lean_inc(v_u_805_);
v___x_818_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_818_, 0, v_u_805_);
lean_ctor_set(v___x_818_, 1, v___x_816_);
v___x_819_ = l_Lean_Expr_const___override(v___x_817_, v___x_818_);
lean_inc_ref(v_00_u03b1_806_);
v___x_820_ = l_Lean_Expr_app___override(v___x_819_, v_00_u03b1_806_);
v___x_821_ = l_Lean_mkNatLit(v_i_809_);
v___x_822_ = l_Lean_Expr_app___override(v___x_820_, v___x_821_);
v___x_823_ = l_Lean_Expr_app___override(v___x_822_, v_a_815_);
v___x_824_ = l_Lean_Expr_app___override(v___x_823_, v_rest_810_);
v_i_809_ = v___x_813_;
v_rest_810_ = v___x_824_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Data_Fin_VecNotation_0__PiFin_mkLiteralQ_loop___boxed(lean_object* v_u_826_, lean_object* v_00_u03b1_827_, lean_object* v_n_828_, lean_object* v_elems_829_, lean_object* v_i_830_, lean_object* v_rest_831_){
_start:
{
lean_object* v_res_832_; 
v_res_832_ = lp_mathlib___private_Mathlib_Data_Fin_VecNotation_0__PiFin_mkLiteralQ_loop(v_u_826_, v_00_u03b1_827_, v_n_828_, v_elems_829_, v_i_830_, v_rest_831_);
lean_dec(v_n_828_);
return v_res_832_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_PiFin_mkLiteralQ(lean_object* v_u_833_, lean_object* v_00_u03b1_834_, lean_object* v_n_835_, lean_object* v_elems_836_){
_start:
{
lean_object* v___x_837_; lean_object* v___x_838_; lean_object* v___x_839_; lean_object* v___x_840_; lean_object* v___x_841_; lean_object* v___x_842_; lean_object* v___x_843_; 
v___x_837_ = lean_unsigned_to_nat(0u);
v___x_838_ = lean_box(0);
v___x_839_ = ((lean_object*)(lp_mathlib_Matrix___aux__Mathlib__Data__Fin__VecNotation______macroRules__Matrix__vecNotation__1___closed__3));
lean_inc(v_u_833_);
v___x_840_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_840_, 0, v_u_833_);
lean_ctor_set(v___x_840_, 1, v___x_838_);
v___x_841_ = l_Lean_Expr_const___override(v___x_839_, v___x_840_);
lean_inc_ref(v_00_u03b1_834_);
v___x_842_ = l_Lean_Expr_app___override(v___x_841_, v_00_u03b1_834_);
v___x_843_ = lp_mathlib___private_Mathlib_Data_Fin_VecNotation_0__PiFin_mkLiteralQ_loop(v_u_833_, v_00_u03b1_834_, v_n_835_, v_elems_836_, v___x_837_, v___x_842_);
return v___x_843_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_PiFin_mkLiteralQ___boxed(lean_object* v_u_844_, lean_object* v_00_u03b1_845_, lean_object* v_n_846_, lean_object* v_elems_847_){
_start:
{
lean_object* v_res_848_; 
v_res_848_ = lp_mathlib_PiFin_mkLiteralQ(v_u_844_, v_00_u03b1_845_, v_n_846_, v_elems_847_);
lean_dec(v_n_846_);
return v_res_848_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_PiFin_toExpr___redArg___lam__0(lean_object* v_v_849_, lean_object* v_toExpr_850_, lean_object* v_i_851_){
_start:
{
lean_object* v___x_852_; lean_object* v_this_853_; 
v___x_852_ = lean_apply_1(v_v_849_, v_i_851_);
v_this_853_ = lean_apply_1(v_toExpr_850_, v___x_852_);
return v_this_853_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_PiFin_toExpr___redArg___lam__1(lean_object* v_toExpr_854_, lean_object* v_inst_855_, lean_object* v_toTypeExpr_856_, lean_object* v_n_857_, lean_object* v_v_858_){
_start:
{
lean_object* v___f_859_; lean_object* v___x_860_; 
v___f_859_ = lean_alloc_closure((void*)(lp_mathlib_PiFin_toExpr___redArg___lam__0), 3, 2);
lean_closure_set(v___f_859_, 0, v_v_858_);
lean_closure_set(v___f_859_, 1, v_toExpr_854_);
v___x_860_ = lp_mathlib_PiFin_mkLiteralQ(v_inst_855_, v_toTypeExpr_856_, v_n_857_, v___f_859_);
return v___x_860_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_PiFin_toExpr___redArg___lam__1___boxed(lean_object* v_toExpr_861_, lean_object* v_inst_862_, lean_object* v_toTypeExpr_863_, lean_object* v_n_864_, lean_object* v_v_865_){
_start:
{
lean_object* v_res_866_; 
v_res_866_ = lp_mathlib_PiFin_toExpr___redArg___lam__1(v_toExpr_861_, v_inst_862_, v_toTypeExpr_863_, v_n_864_, v_v_865_);
lean_dec(v_n_864_);
return v_res_866_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_PiFin_toExpr___redArg(lean_object* v_inst_867_, lean_object* v_inst_868_, lean_object* v_n_869_){
_start:
{
lean_object* v_toExpr_870_; lean_object* v_toTypeExpr_871_; lean_object* v___x_873_; uint8_t v_isShared_874_; uint8_t v_isSharedCheck_885_; 
v_toExpr_870_ = lean_ctor_get(v_inst_868_, 0);
v_toTypeExpr_871_ = lean_ctor_get(v_inst_868_, 1);
v_isSharedCheck_885_ = !lean_is_exclusive(v_inst_868_);
if (v_isSharedCheck_885_ == 0)
{
v___x_873_ = v_inst_868_;
v_isShared_874_ = v_isSharedCheck_885_;
goto v_resetjp_872_;
}
else
{
lean_inc(v_toTypeExpr_871_);
lean_inc(v_toExpr_870_);
lean_dec(v_inst_868_);
v___x_873_ = lean_box(0);
v_isShared_874_ = v_isSharedCheck_885_;
goto v_resetjp_872_;
}
v_resetjp_872_:
{
lean_object* v___f_875_; lean_object* v___x_876_; lean_object* v___x_877_; lean_object* v___x_878_; lean_object* v___x_879_; uint8_t v___x_880_; lean_object* v_toTypeExpr_881_; lean_object* v___x_883_; 
lean_inc(v_n_869_);
lean_inc_ref(v_toTypeExpr_871_);
v___f_875_ = lean_alloc_closure((void*)(lp_mathlib_PiFin_toExpr___redArg___lam__1___boxed), 5, 4);
lean_closure_set(v___f_875_, 0, v_toExpr_870_);
lean_closure_set(v___f_875_, 1, v_inst_867_);
lean_closure_set(v___f_875_, 2, v_toTypeExpr_871_);
lean_closure_set(v___f_875_, 3, v_n_869_);
v___x_876_ = lean_box(0);
v___x_877_ = lean_obj_once(&lp_mathlib_Matrix_cons__val___redArg___closed__24, &lp_mathlib_Matrix_cons__val___redArg___closed__24_once, _init_lp_mathlib_Matrix_cons__val___redArg___closed__24);
v___x_878_ = l_Lean_mkNatLit(v_n_869_);
v___x_879_ = l_Lean_Expr_app___override(v___x_877_, v___x_878_);
v___x_880_ = 0;
v_toTypeExpr_881_ = l_Lean_Expr_forallE___override(v___x_876_, v___x_879_, v_toTypeExpr_871_, v___x_880_);
if (v_isShared_874_ == 0)
{
lean_ctor_set(v___x_873_, 1, v_toTypeExpr_881_);
lean_ctor_set(v___x_873_, 0, v___f_875_);
v___x_883_ = v___x_873_;
goto v_reusejp_882_;
}
else
{
lean_object* v_reuseFailAlloc_884_; 
v_reuseFailAlloc_884_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_884_, 0, v___f_875_);
lean_ctor_set(v_reuseFailAlloc_884_, 1, v_toTypeExpr_881_);
v___x_883_ = v_reuseFailAlloc_884_;
goto v_reusejp_882_;
}
v_reusejp_882_:
{
return v___x_883_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_PiFin_toExpr(lean_object* v_00_u03b1_886_, lean_object* v_inst_887_, lean_object* v_inst_888_, lean_object* v_n_889_){
_start:
{
lean_object* v___x_890_; 
v___x_890_ = lp_mathlib_PiFin_toExpr___redArg(v_inst_887_, v_inst_888_, v_n_889_);
return v___x_890_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Matrix_vecAppend___redArg(lean_object* v_m_891_, lean_object* v_u_892_, lean_object* v_v_893_, lean_object* v_a_894_){
_start:
{
lean_object* v___x_895_; 
v___x_895_ = l_Fin_addCases___redArg(v_m_891_, v_u_892_, v_v_893_, v_a_894_);
return v___x_895_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Matrix_vecAppend___redArg___boxed(lean_object* v_m_896_, lean_object* v_u_897_, lean_object* v_v_898_, lean_object* v_a_899_){
_start:
{
lean_object* v_res_900_; 
v_res_900_ = lp_mathlib_Matrix_vecAppend___redArg(v_m_896_, v_u_897_, v_v_898_, v_a_899_);
lean_dec(v_m_896_);
return v_res_900_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Matrix_vecAppend(lean_object* v_m_901_, lean_object* v_n_902_, lean_object* v_00_u03b1_903_, lean_object* v_o_904_, lean_object* v_ho_905_, lean_object* v_u_906_, lean_object* v_v_907_, lean_object* v_a_908_){
_start:
{
lean_object* v___x_909_; 
v___x_909_ = l_Fin_addCases___redArg(v_m_901_, v_u_906_, v_v_907_, v_a_908_);
return v___x_909_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Matrix_vecAppend___boxed(lean_object* v_m_910_, lean_object* v_n_911_, lean_object* v_00_u03b1_912_, lean_object* v_o_913_, lean_object* v_ho_914_, lean_object* v_u_915_, lean_object* v_v_916_, lean_object* v_a_917_){
_start:
{
lean_object* v_res_918_; 
v_res_918_ = lp_mathlib_Matrix_vecAppend(v_m_910_, v_n_911_, v_00_u03b1_912_, v_o_913_, v_ho_914_, v_u_915_, v_v_916_, v_a_917_);
lean_dec(v_o_913_);
lean_dec(v_n_911_);
lean_dec(v_m_910_);
return v_res_918_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Matrix_vecAlt0___redArg(lean_object* v_v_919_, lean_object* v_k_920_){
_start:
{
lean_object* v___x_921_; lean_object* v___x_922_; 
v___x_921_ = lean_nat_add(v_k_920_, v_k_920_);
v___x_922_ = lean_apply_1(v_v_919_, v___x_921_);
return v___x_922_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Matrix_vecAlt0___redArg___boxed(lean_object* v_v_923_, lean_object* v_k_924_){
_start:
{
lean_object* v_res_925_; 
v_res_925_ = lp_mathlib_Matrix_vecAlt0___redArg(v_v_923_, v_k_924_);
lean_dec(v_k_924_);
return v_res_925_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Matrix_vecAlt0(lean_object* v_00_u03b1_926_, lean_object* v_m_927_, lean_object* v_n_928_, lean_object* v_hm_929_, lean_object* v_v_930_, lean_object* v_k_931_){
_start:
{
lean_object* v___x_932_; 
v___x_932_ = lp_mathlib_Matrix_vecAlt0___redArg(v_v_930_, v_k_931_);
return v___x_932_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Matrix_vecAlt0___boxed(lean_object* v_00_u03b1_933_, lean_object* v_m_934_, lean_object* v_n_935_, lean_object* v_hm_936_, lean_object* v_v_937_, lean_object* v_k_938_){
_start:
{
lean_object* v_res_939_; 
v_res_939_ = lp_mathlib_Matrix_vecAlt0(v_00_u03b1_933_, v_m_934_, v_n_935_, v_hm_936_, v_v_937_, v_k_938_);
lean_dec(v_k_938_);
lean_dec(v_n_935_);
lean_dec(v_m_934_);
return v_res_939_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Matrix_vecAlt1___redArg(lean_object* v_v_940_, lean_object* v_k_941_){
_start:
{
lean_object* v___x_942_; lean_object* v___x_943_; lean_object* v___x_944_; lean_object* v___x_945_; 
v___x_942_ = lean_nat_add(v_k_941_, v_k_941_);
v___x_943_ = lean_unsigned_to_nat(1u);
v___x_944_ = lean_nat_add(v___x_942_, v___x_943_);
lean_dec(v___x_942_);
v___x_945_ = lean_apply_1(v_v_940_, v___x_944_);
return v___x_945_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Matrix_vecAlt1___redArg___boxed(lean_object* v_v_946_, lean_object* v_k_947_){
_start:
{
lean_object* v_res_948_; 
v_res_948_ = lp_mathlib_Matrix_vecAlt1___redArg(v_v_946_, v_k_947_);
lean_dec(v_k_947_);
return v_res_948_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Matrix_vecAlt1(lean_object* v_00_u03b1_949_, lean_object* v_m_950_, lean_object* v_n_951_, lean_object* v_hm_952_, lean_object* v_v_953_, lean_object* v_k_954_){
_start:
{
lean_object* v___x_955_; 
v___x_955_ = lp_mathlib_Matrix_vecAlt1___redArg(v_v_953_, v_k_954_);
return v___x_955_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Matrix_vecAlt1___boxed(lean_object* v_00_u03b1_956_, lean_object* v_m_957_, lean_object* v_n_958_, lean_object* v_hm_959_, lean_object* v_v_960_, lean_object* v_k_961_){
_start:
{
lean_object* v_res_962_; 
v_res_962_ = lp_mathlib_Matrix_vecAlt1(v_00_u03b1_956_, v_m_957_, v_n_958_, v_hm_959_, v_v_960_, v_k_961_);
lean_dec(v_k_961_);
lean_dec(v_n_958_);
lean_dec(v_m_957_);
return v_res_962_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Data_Fin_Tuple_Basic(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Data_Set_Image(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_Data_Fin_VecNotation(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Data_Fin_Tuple_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Data_Set_Image(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_Data_Fin_VecNotation(uint8_t builtin) {
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
lean_object* initialize_mathlib_Mathlib_Data_Fin_Tuple_Basic(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Data_Set_Image(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Data_Fin_VecNotation(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_Fin_Tuple_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_Set_Image(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Data_Fin_VecNotation(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_Data_Fin_VecNotation(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_Data_Fin_VecNotation(builtin);
}
#ifdef __cplusplus
}
#endif
