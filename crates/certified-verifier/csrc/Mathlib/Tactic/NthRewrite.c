// Lean compiler output
// Module: Mathlib.Tactic.NthRewrite
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
lean_object* l_Lean_Name_mkStr4(lean_object*, lean_object*, lean_object*, lean_object*);
extern lean_object* l_Lean_Parser_Tactic_optConfig;
lean_object* l_Lean_Name_mkStr3(lean_object*, lean_object*, lean_object*);
uint8_t l_Lean_Syntax_isOfKind(lean_object*, lean_object*);
lean_object* l_Lean_Syntax_getArg(lean_object*, lean_object*);
lean_object* l_Array_append___redArg(lean_object*, lean_object*);
lean_object* l_Lean_Syntax_node4(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Syntax_getArgs(lean_object*);
lean_object* l_Lean_SourceInfo_fromRef(lean_object*, uint8_t);
lean_object* l_Array_mkArray0(lean_object*);
lean_object* l_Lean_Parser_Tactic_getConfigItems(lean_object*);
size_t lean_array_size(lean_object*);
uint8_t lean_usize_dec_lt(size_t, size_t);
lean_object* lean_array_uget(lean_object*, size_t);
lean_object* lean_array_uset(lean_object*, size_t, lean_object*);
size_t lean_usize_add(size_t, size_t);
lean_object* l_String_toRawSubstring_x27(lean_object*);
lean_object* l_Lean_addMacroScope(lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Syntax_node2(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_mkAtom(lean_object*);
lean_object* l_Lean_mkSepArray(lean_object*, lean_object*);
lean_object* l_Lean_Syntax_node3(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Syntax_node1(lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Syntax_node5(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lean_array_push(lean_object*, lean_object*);
lean_object* lean_mk_empty_array_with_capacity(lean_object*);
lean_object* l_Lean_Syntax_getOptional_x3f(lean_object*);
extern lean_object* l_Lean_Parser_Tactic_location;
extern lean_object* l_Lean_Parser_Tactic_rwRuleSeq;
static const lean_string_object lp_mathlib_Mathlib_Tactic_tacticNth__rewrite___________00__closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "Mathlib"};
static const lean_object* lp_mathlib_Mathlib_Tactic_tacticNth__rewrite___________00__closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_tacticNth__rewrite___________00__closed__0_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_tacticNth__rewrite___________00__closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "Tactic"};
static const lean_object* lp_mathlib_Mathlib_Tactic_tacticNth__rewrite___________00__closed__1 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_tacticNth__rewrite___________00__closed__1_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_tacticNth__rewrite___________00__closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 23, .m_capacity = 23, .m_length = 22, .m_data = "tacticNth_rewrite_____"};
static const lean_object* lp_mathlib_Mathlib_Tactic_tacticNth__rewrite___________00__closed__2 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_tacticNth__rewrite___________00__closed__2_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_tacticNth__rewrite___________00__closed__3_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_tacticNth__rewrite___________00__closed__0_value),LEAN_SCALAR_PTR_LITERAL(118, 213, 161, 2, 73, 184, 31, 228)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_tacticNth__rewrite___________00__closed__3_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_tacticNth__rewrite___________00__closed__3_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic_tacticNth__rewrite___________00__closed__1_value),LEAN_SCALAR_PTR_LITERAL(139, 222, 98, 232, 116, 132, 69, 249)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_tacticNth__rewrite___________00__closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_tacticNth__rewrite___________00__closed__3_value_aux_1),((lean_object*)&lp_mathlib_Mathlib_Tactic_tacticNth__rewrite___________00__closed__2_value),LEAN_SCALAR_PTR_LITERAL(23, 71, 213, 251, 78, 54, 112, 9)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_tacticNth__rewrite___________00__closed__3 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_tacticNth__rewrite___________00__closed__3_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_tacticNth__rewrite___________00__closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "andthen"};
static const lean_object* lp_mathlib_Mathlib_Tactic_tacticNth__rewrite___________00__closed__4 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_tacticNth__rewrite___________00__closed__4_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_tacticNth__rewrite___________00__closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_tacticNth__rewrite___________00__closed__4_value),LEAN_SCALAR_PTR_LITERAL(40, 255, 78, 30, 143, 119, 117, 174)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_tacticNth__rewrite___________00__closed__5 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_tacticNth__rewrite___________00__closed__5_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_tacticNth__rewrite___________00__closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 12, .m_capacity = 12, .m_length = 11, .m_data = "nth_rewrite"};
static const lean_object* lp_mathlib_Mathlib_Tactic_tacticNth__rewrite___________00__closed__6 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_tacticNth__rewrite___________00__closed__6_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_tacticNth__rewrite___________00__closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 8, .m_other = 1, .m_tag = 6}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_tacticNth__rewrite___________00__closed__6_value),LEAN_SCALAR_PTR_LITERAL(0, 0, 0, 0, 0, 0, 0, 0)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_tacticNth__rewrite___________00__closed__7 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_tacticNth__rewrite___________00__closed__7_value;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_tacticNth__rewrite___________00__closed__8_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_tacticNth__rewrite___________00__closed__8;
static const lean_string_object lp_mathlib_Mathlib_Tactic_tacticNth__rewrite___________00__closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "group"};
static const lean_object* lp_mathlib_Mathlib_Tactic_tacticNth__rewrite___________00__closed__9 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_tacticNth__rewrite___________00__closed__9_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_tacticNth__rewrite___________00__closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_tacticNth__rewrite___________00__closed__9_value),LEAN_SCALAR_PTR_LITERAL(206, 113, 20, 57, 188, 177, 187, 30)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_tacticNth__rewrite___________00__closed__10 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_tacticNth__rewrite___________00__closed__10_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_tacticNth__rewrite___________00__closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "ppSpace"};
static const lean_object* lp_mathlib_Mathlib_Tactic_tacticNth__rewrite___________00__closed__11 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_tacticNth__rewrite___________00__closed__11_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_tacticNth__rewrite___________00__closed__12_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_tacticNth__rewrite___________00__closed__11_value),LEAN_SCALAR_PTR_LITERAL(207, 47, 58, 43, 30, 240, 125, 246)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_tacticNth__rewrite___________00__closed__12 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_tacticNth__rewrite___________00__closed__12_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_tacticNth__rewrite___________00__closed__13_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_tacticNth__rewrite___________00__closed__12_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_tacticNth__rewrite___________00__closed__13 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_tacticNth__rewrite___________00__closed__13_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_tacticNth__rewrite___________00__closed__14_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_tacticNth__rewrite___________00__closed__10_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_tacticNth__rewrite___________00__closed__13_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_tacticNth__rewrite___________00__closed__14 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_tacticNth__rewrite___________00__closed__14_value;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_tacticNth__rewrite___________00__closed__15_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_tacticNth__rewrite___________00__closed__15;
static const lean_string_object lp_mathlib_Mathlib_Tactic_tacticNth__rewrite___________00__closed__16_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "many1"};
static const lean_object* lp_mathlib_Mathlib_Tactic_tacticNth__rewrite___________00__closed__16 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_tacticNth__rewrite___________00__closed__16_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_tacticNth__rewrite___________00__closed__17_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_tacticNth__rewrite___________00__closed__16_value),LEAN_SCALAR_PTR_LITERAL(55, 136, 52, 6, 12, 19, 78, 239)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_tacticNth__rewrite___________00__closed__17 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_tacticNth__rewrite___________00__closed__17_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_tacticNth__rewrite___________00__closed__18_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = "num"};
static const lean_object* lp_mathlib_Mathlib_Tactic_tacticNth__rewrite___________00__closed__18 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_tacticNth__rewrite___________00__closed__18_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_tacticNth__rewrite___________00__closed__19_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_tacticNth__rewrite___________00__closed__18_value),LEAN_SCALAR_PTR_LITERAL(227, 68, 22, 222, 47, 51, 204, 84)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_tacticNth__rewrite___________00__closed__19 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_tacticNth__rewrite___________00__closed__19_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_tacticNth__rewrite___________00__closed__20_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_tacticNth__rewrite___________00__closed__19_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_tacticNth__rewrite___________00__closed__20 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_tacticNth__rewrite___________00__closed__20_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_tacticNth__rewrite___________00__closed__21_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_tacticNth__rewrite___________00__closed__17_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_tacticNth__rewrite___________00__closed__20_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_tacticNth__rewrite___________00__closed__21 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_tacticNth__rewrite___________00__closed__21_value;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_tacticNth__rewrite___________00__closed__22_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_tacticNth__rewrite___________00__closed__22;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_tacticNth__rewrite___________00__closed__23_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_tacticNth__rewrite___________00__closed__23;
static const lean_string_object lp_mathlib_Mathlib_Tactic_tacticNth__rewrite___________00__closed__24_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "optional"};
static const lean_object* lp_mathlib_Mathlib_Tactic_tacticNth__rewrite___________00__closed__24 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_tacticNth__rewrite___________00__closed__24_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_tacticNth__rewrite___________00__closed__25_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_tacticNth__rewrite___________00__closed__24_value),LEAN_SCALAR_PTR_LITERAL(233, 141, 154, 50, 143, 135, 42, 252)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_tacticNth__rewrite___________00__closed__25 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_tacticNth__rewrite___________00__closed__25_value;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_tacticNth__rewrite___________00__closed__26_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_tacticNth__rewrite___________00__closed__26;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_tacticNth__rewrite___________00__closed__27_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_tacticNth__rewrite___________00__closed__27;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_tacticNth__rewrite___________00__closed__28_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_tacticNth__rewrite___________00__closed__28;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_tacticNth__rewrite__________;
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Mathlib_Tactic___aux__Mathlib__Tactic__NthRewrite______macroRules__Mathlib__Tactic__tacticNth__rewrite____________1_spec__0(size_t, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Mathlib_Tactic___aux__Mathlib__Tactic__NthRewrite______macroRules__Mathlib__Tactic__tacticNth__rewrite____________1_spec__0___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Mathlib_Tactic___aux__Mathlib__Tactic__NthRewrite______macroRules__Mathlib__Tactic__tacticNth__rewrite____________1_spec__1(size_t, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Mathlib_Tactic___aux__Mathlib__Tactic__NthRewrite______macroRules__Mathlib__Tactic__tacticNth__rewrite____________1_spec__1___boxed(lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__NthRewrite______macroRules__Mathlib__Tactic__tacticNth__rewrite____________1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "Lean"};
static const lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__NthRewrite______macroRules__Mathlib__Tactic__tacticNth__rewrite____________1___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__NthRewrite______macroRules__Mathlib__Tactic__tacticNth__rewrite____________1___closed__0_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__NthRewrite______macroRules__Mathlib__Tactic__tacticNth__rewrite____________1___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "Parser"};
static const lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__NthRewrite______macroRules__Mathlib__Tactic__tacticNth__rewrite____________1___closed__1 = (const lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__NthRewrite______macroRules__Mathlib__Tactic__tacticNth__rewrite____________1___closed__1_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__NthRewrite______macroRules__Mathlib__Tactic__tacticNth__rewrite____________1___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 9, .m_data = "optConfig"};
static const lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__NthRewrite______macroRules__Mathlib__Tactic__tacticNth__rewrite____________1___closed__2 = (const lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__NthRewrite______macroRules__Mathlib__Tactic__tacticNth__rewrite____________1___closed__2_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__NthRewrite______macroRules__Mathlib__Tactic__tacticNth__rewrite____________1___closed__3_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__NthRewrite______macroRules__Mathlib__Tactic__tacticNth__rewrite____________1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__NthRewrite______macroRules__Mathlib__Tactic__tacticNth__rewrite____________1___closed__3_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__NthRewrite______macroRules__Mathlib__Tactic__tacticNth__rewrite____________1___closed__3_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__NthRewrite______macroRules__Mathlib__Tactic__tacticNth__rewrite____________1___closed__1_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__NthRewrite______macroRules__Mathlib__Tactic__tacticNth__rewrite____________1___closed__3_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__NthRewrite______macroRules__Mathlib__Tactic__tacticNth__rewrite____________1___closed__3_value_aux_1),((lean_object*)&lp_mathlib_Mathlib_Tactic_tacticNth__rewrite___________00__closed__1_value),LEAN_SCALAR_PTR_LITERAL(166, 58, 35, 182, 187, 130, 147, 254)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__NthRewrite______macroRules__Mathlib__Tactic__tacticNth__rewrite____________1___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__NthRewrite______macroRules__Mathlib__Tactic__tacticNth__rewrite____________1___closed__3_value_aux_2),((lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__NthRewrite______macroRules__Mathlib__Tactic__tacticNth__rewrite____________1___closed__2_value),LEAN_SCALAR_PTR_LITERAL(137, 208, 10, 74, 108, 50, 106, 48)}};
static const lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__NthRewrite______macroRules__Mathlib__Tactic__tacticNth__rewrite____________1___closed__3 = (const lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__NthRewrite______macroRules__Mathlib__Tactic__tacticNth__rewrite____________1___closed__3_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__NthRewrite______macroRules__Mathlib__Tactic__tacticNth__rewrite____________1___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 11, .m_capacity = 11, .m_length = 10, .m_data = "rewriteSeq"};
static const lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__NthRewrite______macroRules__Mathlib__Tactic__tacticNth__rewrite____________1___closed__4 = (const lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__NthRewrite______macroRules__Mathlib__Tactic__tacticNth__rewrite____________1___closed__4_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__NthRewrite______macroRules__Mathlib__Tactic__tacticNth__rewrite____________1___closed__5_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__NthRewrite______macroRules__Mathlib__Tactic__tacticNth__rewrite____________1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__NthRewrite______macroRules__Mathlib__Tactic__tacticNth__rewrite____________1___closed__5_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__NthRewrite______macroRules__Mathlib__Tactic__tacticNth__rewrite____________1___closed__5_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__NthRewrite______macroRules__Mathlib__Tactic__tacticNth__rewrite____________1___closed__1_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__NthRewrite______macroRules__Mathlib__Tactic__tacticNth__rewrite____________1___closed__5_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__NthRewrite______macroRules__Mathlib__Tactic__tacticNth__rewrite____________1___closed__5_value_aux_1),((lean_object*)&lp_mathlib_Mathlib_Tactic_tacticNth__rewrite___________00__closed__1_value),LEAN_SCALAR_PTR_LITERAL(166, 58, 35, 182, 187, 130, 147, 254)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__NthRewrite______macroRules__Mathlib__Tactic__tacticNth__rewrite____________1___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__NthRewrite______macroRules__Mathlib__Tactic__tacticNth__rewrite____________1___closed__5_value_aux_2),((lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__NthRewrite______macroRules__Mathlib__Tactic__tacticNth__rewrite____________1___closed__4_value),LEAN_SCALAR_PTR_LITERAL(197, 231, 198, 107, 115, 169, 96, 174)}};
static const lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__NthRewrite______macroRules__Mathlib__Tactic__tacticNth__rewrite____________1___closed__5 = (const lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__NthRewrite______macroRules__Mathlib__Tactic__tacticNth__rewrite____________1___closed__5_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__NthRewrite______macroRules__Mathlib__Tactic__tacticNth__rewrite____________1___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "rewrite"};
static const lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__NthRewrite______macroRules__Mathlib__Tactic__tacticNth__rewrite____________1___closed__6 = (const lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__NthRewrite______macroRules__Mathlib__Tactic__tacticNth__rewrite____________1___closed__6_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__NthRewrite______macroRules__Mathlib__Tactic__tacticNth__rewrite____________1___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "null"};
static const lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__NthRewrite______macroRules__Mathlib__Tactic__tacticNth__rewrite____________1___closed__7 = (const lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__NthRewrite______macroRules__Mathlib__Tactic__tacticNth__rewrite____________1___closed__7_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__NthRewrite______macroRules__Mathlib__Tactic__tacticNth__rewrite____________1___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__NthRewrite______macroRules__Mathlib__Tactic__tacticNth__rewrite____________1___closed__7_value),LEAN_SCALAR_PTR_LITERAL(24, 58, 49, 223, 146, 207, 197, 136)}};
static const lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__NthRewrite______macroRules__Mathlib__Tactic__tacticNth__rewrite____________1___closed__8 = (const lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__NthRewrite______macroRules__Mathlib__Tactic__tacticNth__rewrite____________1___closed__8_value;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__NthRewrite______macroRules__Mathlib__Tactic__tacticNth__rewrite____________1___closed__9_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__NthRewrite______macroRules__Mathlib__Tactic__tacticNth__rewrite____________1___closed__9;
static const lean_string_object lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__NthRewrite______macroRules__Mathlib__Tactic__tacticNth__rewrite____________1___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 11, .m_capacity = 11, .m_length = 10, .m_data = "configItem"};
static const lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__NthRewrite______macroRules__Mathlib__Tactic__tacticNth__rewrite____________1___closed__10 = (const lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__NthRewrite______macroRules__Mathlib__Tactic__tacticNth__rewrite____________1___closed__10_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__NthRewrite______macroRules__Mathlib__Tactic__tacticNth__rewrite____________1___closed__11_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__NthRewrite______macroRules__Mathlib__Tactic__tacticNth__rewrite____________1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__NthRewrite______macroRules__Mathlib__Tactic__tacticNth__rewrite____________1___closed__11_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__NthRewrite______macroRules__Mathlib__Tactic__tacticNth__rewrite____________1___closed__11_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__NthRewrite______macroRules__Mathlib__Tactic__tacticNth__rewrite____________1___closed__1_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__NthRewrite______macroRules__Mathlib__Tactic__tacticNth__rewrite____________1___closed__11_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__NthRewrite______macroRules__Mathlib__Tactic__tacticNth__rewrite____________1___closed__11_value_aux_1),((lean_object*)&lp_mathlib_Mathlib_Tactic_tacticNth__rewrite___________00__closed__1_value),LEAN_SCALAR_PTR_LITERAL(166, 58, 35, 182, 187, 130, 147, 254)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__NthRewrite______macroRules__Mathlib__Tactic__tacticNth__rewrite____________1___closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__NthRewrite______macroRules__Mathlib__Tactic__tacticNth__rewrite____________1___closed__11_value_aux_2),((lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__NthRewrite______macroRules__Mathlib__Tactic__tacticNth__rewrite____________1___closed__10_value),LEAN_SCALAR_PTR_LITERAL(205, 9, 236, 192, 59, 252, 178, 140)}};
static const lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__NthRewrite______macroRules__Mathlib__Tactic__tacticNth__rewrite____________1___closed__11 = (const lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__NthRewrite______macroRules__Mathlib__Tactic__tacticNth__rewrite____________1___closed__11_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__NthRewrite______macroRules__Mathlib__Tactic__tacticNth__rewrite____________1___closed__12_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 14, .m_capacity = 14, .m_length = 13, .m_data = "valConfigItem"};
static const lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__NthRewrite______macroRules__Mathlib__Tactic__tacticNth__rewrite____________1___closed__12 = (const lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__NthRewrite______macroRules__Mathlib__Tactic__tacticNth__rewrite____________1___closed__12_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__NthRewrite______macroRules__Mathlib__Tactic__tacticNth__rewrite____________1___closed__13_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__NthRewrite______macroRules__Mathlib__Tactic__tacticNth__rewrite____________1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__NthRewrite______macroRules__Mathlib__Tactic__tacticNth__rewrite____________1___closed__13_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__NthRewrite______macroRules__Mathlib__Tactic__tacticNth__rewrite____________1___closed__13_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__NthRewrite______macroRules__Mathlib__Tactic__tacticNth__rewrite____________1___closed__1_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__NthRewrite______macroRules__Mathlib__Tactic__tacticNth__rewrite____________1___closed__13_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__NthRewrite______macroRules__Mathlib__Tactic__tacticNth__rewrite____________1___closed__13_value_aux_1),((lean_object*)&lp_mathlib_Mathlib_Tactic_tacticNth__rewrite___________00__closed__1_value),LEAN_SCALAR_PTR_LITERAL(166, 58, 35, 182, 187, 130, 147, 254)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__NthRewrite______macroRules__Mathlib__Tactic__tacticNth__rewrite____________1___closed__13_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__NthRewrite______macroRules__Mathlib__Tactic__tacticNth__rewrite____________1___closed__13_value_aux_2),((lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__NthRewrite______macroRules__Mathlib__Tactic__tacticNth__rewrite____________1___closed__12_value),LEAN_SCALAR_PTR_LITERAL(135, 67, 19, 169, 17, 95, 109, 188)}};
static const lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__NthRewrite______macroRules__Mathlib__Tactic__tacticNth__rewrite____________1___closed__13 = (const lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__NthRewrite______macroRules__Mathlib__Tactic__tacticNth__rewrite____________1___closed__13_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__NthRewrite______macroRules__Mathlib__Tactic__tacticNth__rewrite____________1___closed__14_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = "("};
static const lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__NthRewrite______macroRules__Mathlib__Tactic__tacticNth__rewrite____________1___closed__14 = (const lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__NthRewrite______macroRules__Mathlib__Tactic__tacticNth__rewrite____________1___closed__14_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__NthRewrite______macroRules__Mathlib__Tactic__tacticNth__rewrite____________1___closed__15_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "occs"};
static const lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__NthRewrite______macroRules__Mathlib__Tactic__tacticNth__rewrite____________1___closed__15 = (const lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__NthRewrite______macroRules__Mathlib__Tactic__tacticNth__rewrite____________1___closed__15_value;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__NthRewrite______macroRules__Mathlib__Tactic__tacticNth__rewrite____________1___closed__16_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__NthRewrite______macroRules__Mathlib__Tactic__tacticNth__rewrite____________1___closed__16;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__NthRewrite______macroRules__Mathlib__Tactic__tacticNth__rewrite____________1___closed__17_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__NthRewrite______macroRules__Mathlib__Tactic__tacticNth__rewrite____________1___closed__15_value),LEAN_SCALAR_PTR_LITERAL(84, 3, 67, 129, 86, 149, 50, 122)}};
static const lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__NthRewrite______macroRules__Mathlib__Tactic__tacticNth__rewrite____________1___closed__17 = (const lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__NthRewrite______macroRules__Mathlib__Tactic__tacticNth__rewrite____________1___closed__17_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__NthRewrite______macroRules__Mathlib__Tactic__tacticNth__rewrite____________1___closed__18_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 3, .m_capacity = 3, .m_length = 2, .m_data = ":="};
static const lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__NthRewrite______macroRules__Mathlib__Tactic__tacticNth__rewrite____________1___closed__18 = (const lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__NthRewrite______macroRules__Mathlib__Tactic__tacticNth__rewrite____________1___closed__18_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__NthRewrite______macroRules__Mathlib__Tactic__tacticNth__rewrite____________1___closed__19_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "Term"};
static const lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__NthRewrite______macroRules__Mathlib__Tactic__tacticNth__rewrite____________1___closed__19 = (const lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__NthRewrite______macroRules__Mathlib__Tactic__tacticNth__rewrite____________1___closed__19_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__NthRewrite______macroRules__Mathlib__Tactic__tacticNth__rewrite____________1___closed__20_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = "app"};
static const lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__NthRewrite______macroRules__Mathlib__Tactic__tacticNth__rewrite____________1___closed__20 = (const lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__NthRewrite______macroRules__Mathlib__Tactic__tacticNth__rewrite____________1___closed__20_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__NthRewrite______macroRules__Mathlib__Tactic__tacticNth__rewrite____________1___closed__21_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__NthRewrite______macroRules__Mathlib__Tactic__tacticNth__rewrite____________1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__NthRewrite______macroRules__Mathlib__Tactic__tacticNth__rewrite____________1___closed__21_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__NthRewrite______macroRules__Mathlib__Tactic__tacticNth__rewrite____________1___closed__21_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__NthRewrite______macroRules__Mathlib__Tactic__tacticNth__rewrite____________1___closed__1_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__NthRewrite______macroRules__Mathlib__Tactic__tacticNth__rewrite____________1___closed__21_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__NthRewrite______macroRules__Mathlib__Tactic__tacticNth__rewrite____________1___closed__21_value_aux_1),((lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__NthRewrite______macroRules__Mathlib__Tactic__tacticNth__rewrite____________1___closed__19_value),LEAN_SCALAR_PTR_LITERAL(75, 170, 162, 138, 136, 204, 251, 229)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__NthRewrite______macroRules__Mathlib__Tactic__tacticNth__rewrite____________1___closed__21_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__NthRewrite______macroRules__Mathlib__Tactic__tacticNth__rewrite____________1___closed__21_value_aux_2),((lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__NthRewrite______macroRules__Mathlib__Tactic__tacticNth__rewrite____________1___closed__20_value),LEAN_SCALAR_PTR_LITERAL(69, 118, 10, 41, 220, 156, 243, 179)}};
static const lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__NthRewrite______macroRules__Mathlib__Tactic__tacticNth__rewrite____________1___closed__21 = (const lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__NthRewrite______macroRules__Mathlib__Tactic__tacticNth__rewrite____________1___closed__21_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__NthRewrite______macroRules__Mathlib__Tactic__tacticNth__rewrite____________1___closed__22_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "dotIdent"};
static const lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__NthRewrite______macroRules__Mathlib__Tactic__tacticNth__rewrite____________1___closed__22 = (const lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__NthRewrite______macroRules__Mathlib__Tactic__tacticNth__rewrite____________1___closed__22_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__NthRewrite______macroRules__Mathlib__Tactic__tacticNth__rewrite____________1___closed__23_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__NthRewrite______macroRules__Mathlib__Tactic__tacticNth__rewrite____________1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__NthRewrite______macroRules__Mathlib__Tactic__tacticNth__rewrite____________1___closed__23_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__NthRewrite______macroRules__Mathlib__Tactic__tacticNth__rewrite____________1___closed__23_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__NthRewrite______macroRules__Mathlib__Tactic__tacticNth__rewrite____________1___closed__1_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__NthRewrite______macroRules__Mathlib__Tactic__tacticNth__rewrite____________1___closed__23_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__NthRewrite______macroRules__Mathlib__Tactic__tacticNth__rewrite____________1___closed__23_value_aux_1),((lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__NthRewrite______macroRules__Mathlib__Tactic__tacticNth__rewrite____________1___closed__19_value),LEAN_SCALAR_PTR_LITERAL(75, 170, 162, 138, 136, 204, 251, 229)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__NthRewrite______macroRules__Mathlib__Tactic__tacticNth__rewrite____________1___closed__23_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__NthRewrite______macroRules__Mathlib__Tactic__tacticNth__rewrite____________1___closed__23_value_aux_2),((lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__NthRewrite______macroRules__Mathlib__Tactic__tacticNth__rewrite____________1___closed__22_value),LEAN_SCALAR_PTR_LITERAL(173, 139, 76, 218, 89, 59, 213, 196)}};
static const lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__NthRewrite______macroRules__Mathlib__Tactic__tacticNth__rewrite____________1___closed__23 = (const lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__NthRewrite______macroRules__Mathlib__Tactic__tacticNth__rewrite____________1___closed__23_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__NthRewrite______macroRules__Mathlib__Tactic__tacticNth__rewrite____________1___closed__24_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = "."};
static const lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__NthRewrite______macroRules__Mathlib__Tactic__tacticNth__rewrite____________1___closed__24 = (const lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__NthRewrite______macroRules__Mathlib__Tactic__tacticNth__rewrite____________1___closed__24_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__NthRewrite______macroRules__Mathlib__Tactic__tacticNth__rewrite____________1___closed__25_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = "pos"};
static const lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__NthRewrite______macroRules__Mathlib__Tactic__tacticNth__rewrite____________1___closed__25 = (const lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__NthRewrite______macroRules__Mathlib__Tactic__tacticNth__rewrite____________1___closed__25_value;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__NthRewrite______macroRules__Mathlib__Tactic__tacticNth__rewrite____________1___closed__26_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__NthRewrite______macroRules__Mathlib__Tactic__tacticNth__rewrite____________1___closed__26;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__NthRewrite______macroRules__Mathlib__Tactic__tacticNth__rewrite____________1___closed__27_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__NthRewrite______macroRules__Mathlib__Tactic__tacticNth__rewrite____________1___closed__25_value),LEAN_SCALAR_PTR_LITERAL(175, 67, 188, 228, 198, 126, 180, 88)}};
static const lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__NthRewrite______macroRules__Mathlib__Tactic__tacticNth__rewrite____________1___closed__27 = (const lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__NthRewrite______macroRules__Mathlib__Tactic__tacticNth__rewrite____________1___closed__27_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__NthRewrite______macroRules__Mathlib__Tactic__tacticNth__rewrite____________1___closed__28_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "term[_]"};
static const lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__NthRewrite______macroRules__Mathlib__Tactic__tacticNth__rewrite____________1___closed__28 = (const lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__NthRewrite______macroRules__Mathlib__Tactic__tacticNth__rewrite____________1___closed__28_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__NthRewrite______macroRules__Mathlib__Tactic__tacticNth__rewrite____________1___closed__29_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__NthRewrite______macroRules__Mathlib__Tactic__tacticNth__rewrite____________1___closed__28_value),LEAN_SCALAR_PTR_LITERAL(86, 147, 168, 74, 195, 98, 232, 161)}};
static const lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__NthRewrite______macroRules__Mathlib__Tactic__tacticNth__rewrite____________1___closed__29 = (const lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__NthRewrite______macroRules__Mathlib__Tactic__tacticNth__rewrite____________1___closed__29_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__NthRewrite______macroRules__Mathlib__Tactic__tacticNth__rewrite____________1___closed__30_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = "["};
static const lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__NthRewrite______macroRules__Mathlib__Tactic__tacticNth__rewrite____________1___closed__30 = (const lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__NthRewrite______macroRules__Mathlib__Tactic__tacticNth__rewrite____________1___closed__30_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__NthRewrite______macroRules__Mathlib__Tactic__tacticNth__rewrite____________1___closed__31_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = ","};
static const lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__NthRewrite______macroRules__Mathlib__Tactic__tacticNth__rewrite____________1___closed__31 = (const lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__NthRewrite______macroRules__Mathlib__Tactic__tacticNth__rewrite____________1___closed__31_value;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__NthRewrite______macroRules__Mathlib__Tactic__tacticNth__rewrite____________1___closed__32_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__NthRewrite______macroRules__Mathlib__Tactic__tacticNth__rewrite____________1___closed__32;
static const lean_string_object lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__NthRewrite______macroRules__Mathlib__Tactic__tacticNth__rewrite____________1___closed__33_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = "]"};
static const lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__NthRewrite______macroRules__Mathlib__Tactic__tacticNth__rewrite____________1___closed__33 = (const lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__NthRewrite______macroRules__Mathlib__Tactic__tacticNth__rewrite____________1___closed__33_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__NthRewrite______macroRules__Mathlib__Tactic__tacticNth__rewrite____________1___closed__34_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = ")"};
static const lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__NthRewrite______macroRules__Mathlib__Tactic__tacticNth__rewrite____________1___closed__34 = (const lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__NthRewrite______macroRules__Mathlib__Tactic__tacticNth__rewrite____________1___closed__34_value;
static const lean_array_object lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__NthRewrite______macroRules__Mathlib__Tactic__tacticNth__rewrite____________1___closed__35_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_array_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 246}, .m_size = 0, .m_capacity = 0, .m_data = {}};
static const lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__NthRewrite______macroRules__Mathlib__Tactic__tacticNth__rewrite____________1___closed__35 = (const lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__NthRewrite______macroRules__Mathlib__Tactic__tacticNth__rewrite____________1___closed__35_value;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__NthRewrite______macroRules__Mathlib__Tactic__tacticNth__rewrite____________1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__NthRewrite______macroRules__Mathlib__Tactic__tacticNth__rewrite____________1___boxed(lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Mathlib_Tactic_tacticNth__rw___________00__closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 18, .m_capacity = 18, .m_length = 17, .m_data = "tacticNth_rw_____"};
static const lean_object* lp_mathlib_Mathlib_Tactic_tacticNth__rw___________00__closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_tacticNth__rw___________00__closed__0_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_tacticNth__rw___________00__closed__1_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_tacticNth__rewrite___________00__closed__0_value),LEAN_SCALAR_PTR_LITERAL(118, 213, 161, 2, 73, 184, 31, 228)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_tacticNth__rw___________00__closed__1_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_tacticNth__rw___________00__closed__1_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic_tacticNth__rewrite___________00__closed__1_value),LEAN_SCALAR_PTR_LITERAL(139, 222, 98, 232, 116, 132, 69, 249)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_tacticNth__rw___________00__closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_tacticNth__rw___________00__closed__1_value_aux_1),((lean_object*)&lp_mathlib_Mathlib_Tactic_tacticNth__rw___________00__closed__0_value),LEAN_SCALAR_PTR_LITERAL(183, 0, 15, 176, 14, 11, 28, 46)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_tacticNth__rw___________00__closed__1 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_tacticNth__rw___________00__closed__1_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_tacticNth__rw___________00__closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "nth_rw"};
static const lean_object* lp_mathlib_Mathlib_Tactic_tacticNth__rw___________00__closed__2 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_tacticNth__rw___________00__closed__2_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_tacticNth__rw___________00__closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 8, .m_other = 1, .m_tag = 6}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_tacticNth__rw___________00__closed__2_value),LEAN_SCALAR_PTR_LITERAL(0, 0, 0, 0, 0, 0, 0, 0)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_tacticNth__rw___________00__closed__3 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_tacticNth__rw___________00__closed__3_value;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_tacticNth__rw___________00__closed__4_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_tacticNth__rw___________00__closed__4;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_tacticNth__rw___________00__closed__5_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_tacticNth__rw___________00__closed__5;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_tacticNth__rw___________00__closed__6_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_tacticNth__rw___________00__closed__6;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_tacticNth__rw___________00__closed__7_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_tacticNth__rw___________00__closed__7;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_tacticNth__rw___________00__closed__8_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_tacticNth__rw___________00__closed__8;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_tacticNth__rw___________00__closed__9_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_tacticNth__rw___________00__closed__9;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_tacticNth__rw__________;
static const lean_string_object lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__NthRewrite______macroRules__Mathlib__Tactic__tacticNth__rw____________1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "rwSeq"};
static const lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__NthRewrite______macroRules__Mathlib__Tactic__tacticNth__rw____________1___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__NthRewrite______macroRules__Mathlib__Tactic__tacticNth__rw____________1___closed__0_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__NthRewrite______macroRules__Mathlib__Tactic__tacticNth__rw____________1___closed__1_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__NthRewrite______macroRules__Mathlib__Tactic__tacticNth__rewrite____________1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__NthRewrite______macroRules__Mathlib__Tactic__tacticNth__rw____________1___closed__1_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__NthRewrite______macroRules__Mathlib__Tactic__tacticNth__rw____________1___closed__1_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__NthRewrite______macroRules__Mathlib__Tactic__tacticNth__rewrite____________1___closed__1_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__NthRewrite______macroRules__Mathlib__Tactic__tacticNth__rw____________1___closed__1_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__NthRewrite______macroRules__Mathlib__Tactic__tacticNth__rw____________1___closed__1_value_aux_1),((lean_object*)&lp_mathlib_Mathlib_Tactic_tacticNth__rewrite___________00__closed__1_value),LEAN_SCALAR_PTR_LITERAL(166, 58, 35, 182, 187, 130, 147, 254)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__NthRewrite______macroRules__Mathlib__Tactic__tacticNth__rw____________1___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__NthRewrite______macroRules__Mathlib__Tactic__tacticNth__rw____________1___closed__1_value_aux_2),((lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__NthRewrite______macroRules__Mathlib__Tactic__tacticNth__rw____________1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(50, 16, 185, 246, 153, 187, 181, 153)}};
static const lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__NthRewrite______macroRules__Mathlib__Tactic__tacticNth__rw____________1___closed__1 = (const lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__NthRewrite______macroRules__Mathlib__Tactic__tacticNth__rw____________1___closed__1_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__NthRewrite______macroRules__Mathlib__Tactic__tacticNth__rw____________1___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 3, .m_capacity = 3, .m_length = 2, .m_data = "rw"};
static const lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__NthRewrite______macroRules__Mathlib__Tactic__tacticNth__rw____________1___closed__2 = (const lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__NthRewrite______macroRules__Mathlib__Tactic__tacticNth__rw____________1___closed__2_value;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__NthRewrite______macroRules__Mathlib__Tactic__tacticNth__rw____________1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__NthRewrite______macroRules__Mathlib__Tactic__tacticNth__rw____________1___boxed(lean_object*, lean_object*, lean_object*);
static lean_object* _init_lp_mathlib_Mathlib_Tactic_tacticNth__rewrite___________00__closed__8(void){
_start:
{
lean_object* v___x_15_; lean_object* v___x_16_; lean_object* v___x_17_; lean_object* v___x_18_; 
v___x_15_ = l_Lean_Parser_Tactic_optConfig;
v___x_16_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_tacticNth__rewrite___________00__closed__7));
v___x_17_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_tacticNth__rewrite___________00__closed__5));
v___x_18_ = lean_alloc_ctor(2, 3, 0);
lean_ctor_set(v___x_18_, 0, v___x_17_);
lean_ctor_set(v___x_18_, 1, v___x_16_);
lean_ctor_set(v___x_18_, 2, v___x_15_);
return v___x_18_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_tacticNth__rewrite___________00__closed__15(void){
_start:
{
lean_object* v___x_30_; lean_object* v___x_31_; lean_object* v___x_32_; lean_object* v___x_33_; 
v___x_30_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_tacticNth__rewrite___________00__closed__14));
v___x_31_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_tacticNth__rewrite___________00__closed__8, &lp_mathlib_Mathlib_Tactic_tacticNth__rewrite___________00__closed__8_once, _init_lp_mathlib_Mathlib_Tactic_tacticNth__rewrite___________00__closed__8);
v___x_32_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_tacticNth__rewrite___________00__closed__5));
v___x_33_ = lean_alloc_ctor(2, 3, 0);
lean_ctor_set(v___x_33_, 0, v___x_32_);
lean_ctor_set(v___x_33_, 1, v___x_31_);
lean_ctor_set(v___x_33_, 2, v___x_30_);
return v___x_33_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_tacticNth__rewrite___________00__closed__22(void){
_start:
{
lean_object* v___x_45_; lean_object* v___x_46_; lean_object* v___x_47_; lean_object* v___x_48_; 
v___x_45_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_tacticNth__rewrite___________00__closed__21));
v___x_46_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_tacticNth__rewrite___________00__closed__15, &lp_mathlib_Mathlib_Tactic_tacticNth__rewrite___________00__closed__15_once, _init_lp_mathlib_Mathlib_Tactic_tacticNth__rewrite___________00__closed__15);
v___x_47_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_tacticNth__rewrite___________00__closed__5));
v___x_48_ = lean_alloc_ctor(2, 3, 0);
lean_ctor_set(v___x_48_, 0, v___x_47_);
lean_ctor_set(v___x_48_, 1, v___x_46_);
lean_ctor_set(v___x_48_, 2, v___x_45_);
return v___x_48_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_tacticNth__rewrite___________00__closed__23(void){
_start:
{
lean_object* v___x_49_; lean_object* v___x_50_; lean_object* v___x_51_; lean_object* v___x_52_; 
v___x_49_ = l_Lean_Parser_Tactic_rwRuleSeq;
v___x_50_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_tacticNth__rewrite___________00__closed__22, &lp_mathlib_Mathlib_Tactic_tacticNth__rewrite___________00__closed__22_once, _init_lp_mathlib_Mathlib_Tactic_tacticNth__rewrite___________00__closed__22);
v___x_51_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_tacticNth__rewrite___________00__closed__5));
v___x_52_ = lean_alloc_ctor(2, 3, 0);
lean_ctor_set(v___x_52_, 0, v___x_51_);
lean_ctor_set(v___x_52_, 1, v___x_50_);
lean_ctor_set(v___x_52_, 2, v___x_49_);
return v___x_52_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_tacticNth__rewrite___________00__closed__26(void){
_start:
{
lean_object* v___x_56_; lean_object* v___x_57_; lean_object* v___x_58_; 
v___x_56_ = l_Lean_Parser_Tactic_location;
v___x_57_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_tacticNth__rewrite___________00__closed__25));
v___x_58_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_58_, 0, v___x_57_);
lean_ctor_set(v___x_58_, 1, v___x_56_);
return v___x_58_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_tacticNth__rewrite___________00__closed__27(void){
_start:
{
lean_object* v___x_59_; lean_object* v___x_60_; lean_object* v___x_61_; lean_object* v___x_62_; 
v___x_59_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_tacticNth__rewrite___________00__closed__26, &lp_mathlib_Mathlib_Tactic_tacticNth__rewrite___________00__closed__26_once, _init_lp_mathlib_Mathlib_Tactic_tacticNth__rewrite___________00__closed__26);
v___x_60_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_tacticNth__rewrite___________00__closed__23, &lp_mathlib_Mathlib_Tactic_tacticNth__rewrite___________00__closed__23_once, _init_lp_mathlib_Mathlib_Tactic_tacticNth__rewrite___________00__closed__23);
v___x_61_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_tacticNth__rewrite___________00__closed__5));
v___x_62_ = lean_alloc_ctor(2, 3, 0);
lean_ctor_set(v___x_62_, 0, v___x_61_);
lean_ctor_set(v___x_62_, 1, v___x_60_);
lean_ctor_set(v___x_62_, 2, v___x_59_);
return v___x_62_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_tacticNth__rewrite___________00__closed__28(void){
_start:
{
lean_object* v___x_63_; lean_object* v___x_64_; lean_object* v___x_65_; lean_object* v___x_66_; 
v___x_63_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_tacticNth__rewrite___________00__closed__27, &lp_mathlib_Mathlib_Tactic_tacticNth__rewrite___________00__closed__27_once, _init_lp_mathlib_Mathlib_Tactic_tacticNth__rewrite___________00__closed__27);
v___x_64_ = lean_unsigned_to_nat(1022u);
v___x_65_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_tacticNth__rewrite___________00__closed__3));
v___x_66_ = lean_alloc_ctor(3, 3, 0);
lean_ctor_set(v___x_66_, 0, v___x_65_);
lean_ctor_set(v___x_66_, 1, v___x_64_);
lean_ctor_set(v___x_66_, 2, v___x_63_);
return v___x_66_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_tacticNth__rewrite__________(void){
_start:
{
lean_object* v___x_67_; 
v___x_67_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_tacticNth__rewrite___________00__closed__28, &lp_mathlib_Mathlib_Tactic_tacticNth__rewrite___________00__closed__28_once, _init_lp_mathlib_Mathlib_Tactic_tacticNth__rewrite___________00__closed__28);
return v___x_67_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Mathlib_Tactic___aux__Mathlib__Tactic__NthRewrite______macroRules__Mathlib__Tactic__tacticNth__rewrite____________1_spec__0(size_t v_sz_68_, size_t v_i_69_, lean_object* v_bs_70_){
_start:
{
uint8_t v___x_71_; 
v___x_71_ = lean_usize_dec_lt(v_i_69_, v_sz_68_);
if (v___x_71_ == 0)
{
return v_bs_70_;
}
else
{
lean_object* v_v_72_; lean_object* v___x_73_; lean_object* v_bs_x27_74_; size_t v___x_75_; size_t v___x_76_; lean_object* v___x_77_; 
v_v_72_ = lean_array_uget(v_bs_70_, v_i_69_);
v___x_73_ = lean_unsigned_to_nat(0u);
v_bs_x27_74_ = lean_array_uset(v_bs_70_, v_i_69_, v___x_73_);
v___x_75_ = ((size_t)1ULL);
v___x_76_ = lean_usize_add(v_i_69_, v___x_75_);
v___x_77_ = lean_array_uset(v_bs_x27_74_, v_i_69_, v_v_72_);
v_i_69_ = v___x_76_;
v_bs_70_ = v___x_77_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Mathlib_Tactic___aux__Mathlib__Tactic__NthRewrite______macroRules__Mathlib__Tactic__tacticNth__rewrite____________1_spec__0___boxed(lean_object* v_sz_79_, lean_object* v_i_80_, lean_object* v_bs_81_){
_start:
{
size_t v_sz_boxed_82_; size_t v_i_boxed_83_; lean_object* v_res_84_; 
v_sz_boxed_82_ = lean_unbox_usize(v_sz_79_);
lean_dec(v_sz_79_);
v_i_boxed_83_ = lean_unbox_usize(v_i_80_);
lean_dec(v_i_80_);
v_res_84_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Mathlib_Tactic___aux__Mathlib__Tactic__NthRewrite______macroRules__Mathlib__Tactic__tacticNth__rewrite____________1_spec__0(v_sz_boxed_82_, v_i_boxed_83_, v_bs_81_);
return v_res_84_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Mathlib_Tactic___aux__Mathlib__Tactic__NthRewrite______macroRules__Mathlib__Tactic__tacticNth__rewrite____________1_spec__1(size_t v_sz_85_, size_t v_i_86_, lean_object* v_bs_87_){
_start:
{
uint8_t v___x_88_; 
v___x_88_ = lean_usize_dec_lt(v_i_86_, v_sz_85_);
if (v___x_88_ == 0)
{
return v_bs_87_;
}
else
{
lean_object* v_v_89_; lean_object* v___x_90_; lean_object* v_bs_x27_91_; size_t v___x_92_; size_t v___x_93_; lean_object* v___x_94_; 
v_v_89_ = lean_array_uget(v_bs_87_, v_i_86_);
v___x_90_ = lean_unsigned_to_nat(0u);
v_bs_x27_91_ = lean_array_uset(v_bs_87_, v_i_86_, v___x_90_);
v___x_92_ = ((size_t)1ULL);
v___x_93_ = lean_usize_add(v_i_86_, v___x_92_);
v___x_94_ = lean_array_uset(v_bs_x27_91_, v_i_86_, v_v_89_);
v_i_86_ = v___x_93_;
v_bs_87_ = v___x_94_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Mathlib_Tactic___aux__Mathlib__Tactic__NthRewrite______macroRules__Mathlib__Tactic__tacticNth__rewrite____________1_spec__1___boxed(lean_object* v_sz_96_, lean_object* v_i_97_, lean_object* v_bs_98_){
_start:
{
size_t v_sz_boxed_99_; size_t v_i_boxed_100_; lean_object* v_res_101_; 
v_sz_boxed_99_ = lean_unbox_usize(v_sz_96_);
lean_dec(v_sz_96_);
v_i_boxed_100_ = lean_unbox_usize(v_i_97_);
lean_dec(v_i_97_);
v_res_101_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Mathlib_Tactic___aux__Mathlib__Tactic__NthRewrite______macroRules__Mathlib__Tactic__tacticNth__rewrite____________1_spec__1(v_sz_boxed_99_, v_i_boxed_100_, v_bs_98_);
return v_res_101_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__NthRewrite______macroRules__Mathlib__Tactic__tacticNth__rewrite____________1___closed__9(void){
_start:
{
lean_object* v___x_120_; 
v___x_120_ = l_Array_mkArray0(lean_box(0));
return v___x_120_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__NthRewrite______macroRules__Mathlib__Tactic__tacticNth__rewrite____________1___closed__16(void){
_start:
{
lean_object* v___x_135_; lean_object* v___x_136_; 
v___x_135_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__NthRewrite______macroRules__Mathlib__Tactic__tacticNth__rewrite____________1___closed__15));
v___x_136_ = l_String_toRawSubstring_x27(v___x_135_);
return v___x_136_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__NthRewrite______macroRules__Mathlib__Tactic__tacticNth__rewrite____________1___closed__26(void){
_start:
{
lean_object* v___x_155_; lean_object* v___x_156_; 
v___x_155_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__NthRewrite______macroRules__Mathlib__Tactic__tacticNth__rewrite____________1___closed__25));
v___x_156_ = l_String_toRawSubstring_x27(v___x_155_);
return v___x_156_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__NthRewrite______macroRules__Mathlib__Tactic__tacticNth__rewrite____________1___closed__32(void){
_start:
{
lean_object* v___x_164_; lean_object* v___x_165_; 
v___x_164_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__NthRewrite______macroRules__Mathlib__Tactic__tacticNth__rewrite____________1___closed__31));
v___x_165_ = l_Lean_mkAtom(v___x_164_);
return v___x_165_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__NthRewrite______macroRules__Mathlib__Tactic__tacticNth__rewrite____________1(lean_object* v_x_170_, lean_object* v_a_171_, lean_object* v_a_172_){
_start:
{
lean_object* v___x_173_; uint8_t v___x_174_; 
v___x_173_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_tacticNth__rewrite___________00__closed__3));
lean_inc(v_x_170_);
v___x_174_ = l_Lean_Syntax_isOfKind(v_x_170_, v___x_173_);
if (v___x_174_ == 0)
{
lean_object* v___x_175_; lean_object* v___x_176_; 
lean_dec(v_x_170_);
v___x_175_ = lean_box(1);
v___x_176_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_176_, 0, v___x_175_);
lean_ctor_set(v___x_176_, 1, v_a_172_);
return v___x_176_;
}
else
{
lean_object* v___x_177_; lean_object* v___x_178_; lean_object* v___x_179_; lean_object* v___x_180_; lean_object* v___x_181_; lean_object* v___x_182_; lean_object* v___y_184_; lean_object* v___y_185_; lean_object* v___y_186_; lean_object* v___y_187_; lean_object* v___y_188_; lean_object* v___y_189_; lean_object* v___y_190_; lean_object* v___y_196_; lean_object* v___x_261_; lean_object* v___x_262_; lean_object* v___x_263_; 
v___x_177_ = lean_unsigned_to_nat(1u);
v___x_178_ = l_Lean_Syntax_getArg(v_x_170_, v___x_177_);
v___x_179_ = lean_unsigned_to_nat(3u);
v___x_180_ = l_Lean_Syntax_getArg(v_x_170_, v___x_179_);
v___x_181_ = lean_unsigned_to_nat(4u);
v___x_182_ = l_Lean_Syntax_getArg(v_x_170_, v___x_181_);
v___x_261_ = lean_unsigned_to_nat(5u);
v___x_262_ = l_Lean_Syntax_getArg(v_x_170_, v___x_261_);
lean_dec(v_x_170_);
v___x_263_ = l_Lean_Syntax_getOptional_x3f(v___x_262_);
lean_dec(v___x_262_);
if (lean_obj_tag(v___x_263_) == 0)
{
lean_object* v___x_264_; 
v___x_264_ = lean_box(0);
v___y_196_ = v___x_264_;
goto v___jp_195_;
}
else
{
lean_object* v_val_265_; lean_object* v___x_267_; uint8_t v_isShared_268_; uint8_t v_isSharedCheck_272_; 
v_val_265_ = lean_ctor_get(v___x_263_, 0);
v_isSharedCheck_272_ = !lean_is_exclusive(v___x_263_);
if (v_isSharedCheck_272_ == 0)
{
v___x_267_ = v___x_263_;
v_isShared_268_ = v_isSharedCheck_272_;
goto v_resetjp_266_;
}
else
{
lean_inc(v_val_265_);
lean_dec(v___x_263_);
v___x_267_ = lean_box(0);
v_isShared_268_ = v_isSharedCheck_272_;
goto v_resetjp_266_;
}
v_resetjp_266_:
{
lean_object* v___x_270_; 
if (v_isShared_268_ == 0)
{
v___x_270_ = v___x_267_;
goto v_reusejp_269_;
}
else
{
lean_object* v_reuseFailAlloc_271_; 
v_reuseFailAlloc_271_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_271_, 0, v_val_265_);
v___x_270_ = v_reuseFailAlloc_271_;
goto v_reusejp_269_;
}
v_reusejp_269_:
{
v___y_196_ = v___x_270_;
goto v___jp_195_;
}
}
}
v___jp_183_:
{
lean_object* v___x_191_; lean_object* v___x_192_; lean_object* v___x_193_; lean_object* v___x_194_; 
v___x_191_ = l_Array_append___redArg(v___y_187_, v___y_190_);
lean_dec_ref(v___y_190_);
lean_inc(v___y_189_);
v___x_192_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_192_, 0, v___y_189_);
lean_ctor_set(v___x_192_, 1, v___y_185_);
lean_ctor_set(v___x_192_, 2, v___x_191_);
lean_inc(v___y_188_);
v___x_193_ = l_Lean_Syntax_node4(v___y_189_, v___y_188_, v___y_184_, v___y_186_, v___x_182_, v___x_192_);
v___x_194_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_194_, 0, v___x_193_);
lean_ctor_set(v___x_194_, 1, v_a_172_);
return v___x_194_;
}
v___jp_195_:
{
lean_object* v_quotContext_197_; lean_object* v_currMacroScope_198_; lean_object* v_ref_199_; lean_object* v_nums_200_; lean_object* v___x_201_; uint8_t v___x_202_; lean_object* v___x_203_; lean_object* v___x_204_; lean_object* v___x_205_; lean_object* v___x_206_; lean_object* v___x_207_; lean_object* v___x_208_; lean_object* v___x_209_; size_t v_sz_210_; size_t v___x_211_; lean_object* v___x_212_; lean_object* v___x_213_; lean_object* v___x_214_; lean_object* v___x_215_; lean_object* v___x_216_; lean_object* v___x_217_; lean_object* v___x_218_; lean_object* v___x_219_; lean_object* v___x_220_; lean_object* v___x_221_; lean_object* v___x_222_; lean_object* v___x_223_; lean_object* v___x_224_; lean_object* v___x_225_; lean_object* v___x_226_; lean_object* v___x_227_; lean_object* v___x_228_; lean_object* v___x_229_; lean_object* v___x_230_; lean_object* v___x_231_; lean_object* v___x_232_; lean_object* v___x_233_; lean_object* v___x_234_; lean_object* v___x_235_; lean_object* v___x_236_; size_t v_sz_237_; lean_object* v___x_238_; size_t v_sz_239_; lean_object* v___x_240_; lean_object* v___x_241_; lean_object* v___x_242_; lean_object* v___x_243_; lean_object* v___x_244_; lean_object* v___x_245_; lean_object* v___x_246_; lean_object* v___x_247_; lean_object* v___x_248_; lean_object* v___x_249_; lean_object* v___x_250_; lean_object* v___x_251_; lean_object* v___x_252_; lean_object* v___x_253_; lean_object* v___x_254_; lean_object* v___x_255_; lean_object* v___x_256_; 
v_quotContext_197_ = lean_ctor_get(v_a_171_, 1);
v_currMacroScope_198_ = lean_ctor_get(v_a_171_, 2);
v_ref_199_ = lean_ctor_get(v_a_171_, 5);
v_nums_200_ = l_Lean_Syntax_getArgs(v___x_180_);
lean_dec(v___x_180_);
v___x_201_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__NthRewrite______macroRules__Mathlib__Tactic__tacticNth__rewrite____________1___closed__3));
v___x_202_ = 0;
v___x_203_ = l_Lean_SourceInfo_fromRef(v_ref_199_, v___x_202_);
v___x_204_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__NthRewrite______macroRules__Mathlib__Tactic__tacticNth__rewrite____________1___closed__5));
v___x_205_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__NthRewrite______macroRules__Mathlib__Tactic__tacticNth__rewrite____________1___closed__6));
lean_inc_n(v___x_203_, 18);
v___x_206_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_206_, 0, v___x_203_);
lean_ctor_set(v___x_206_, 1, v___x_205_);
v___x_207_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__NthRewrite______macroRules__Mathlib__Tactic__tacticNth__rewrite____________1___closed__8));
v___x_208_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__NthRewrite______macroRules__Mathlib__Tactic__tacticNth__rewrite____________1___closed__9, &lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__NthRewrite______macroRules__Mathlib__Tactic__tacticNth__rewrite____________1___closed__9_once, _init_lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__NthRewrite______macroRules__Mathlib__Tactic__tacticNth__rewrite____________1___closed__9);
v___x_209_ = l_Lean_Parser_Tactic_getConfigItems(v___x_178_);
v_sz_210_ = lean_array_size(v___x_209_);
v___x_211_ = ((size_t)0ULL);
v___x_212_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Mathlib_Tactic___aux__Mathlib__Tactic__NthRewrite______macroRules__Mathlib__Tactic__tacticNth__rewrite____________1_spec__0(v_sz_210_, v___x_211_, v___x_209_);
v___x_213_ = l_Array_append___redArg(v___x_208_, v___x_212_);
lean_dec_ref(v___x_212_);
v___x_214_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__NthRewrite______macroRules__Mathlib__Tactic__tacticNth__rewrite____________1___closed__11));
v___x_215_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__NthRewrite______macroRules__Mathlib__Tactic__tacticNth__rewrite____________1___closed__13));
v___x_216_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__NthRewrite______macroRules__Mathlib__Tactic__tacticNth__rewrite____________1___closed__14));
v___x_217_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_217_, 0, v___x_203_);
lean_ctor_set(v___x_217_, 1, v___x_216_);
v___x_218_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__NthRewrite______macroRules__Mathlib__Tactic__tacticNth__rewrite____________1___closed__16, &lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__NthRewrite______macroRules__Mathlib__Tactic__tacticNth__rewrite____________1___closed__16_once, _init_lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__NthRewrite______macroRules__Mathlib__Tactic__tacticNth__rewrite____________1___closed__16);
v___x_219_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__NthRewrite______macroRules__Mathlib__Tactic__tacticNth__rewrite____________1___closed__17));
lean_inc_n(v_currMacroScope_198_, 2);
lean_inc_n(v_quotContext_197_, 2);
v___x_220_ = l_Lean_addMacroScope(v_quotContext_197_, v___x_219_, v_currMacroScope_198_);
v___x_221_ = lean_box(0);
v___x_222_ = lean_alloc_ctor(3, 4, 0);
lean_ctor_set(v___x_222_, 0, v___x_203_);
lean_ctor_set(v___x_222_, 1, v___x_218_);
lean_ctor_set(v___x_222_, 2, v___x_220_);
lean_ctor_set(v___x_222_, 3, v___x_221_);
v___x_223_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__NthRewrite______macroRules__Mathlib__Tactic__tacticNth__rewrite____________1___closed__18));
v___x_224_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_224_, 0, v___x_203_);
lean_ctor_set(v___x_224_, 1, v___x_223_);
v___x_225_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__NthRewrite______macroRules__Mathlib__Tactic__tacticNth__rewrite____________1___closed__21));
v___x_226_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__NthRewrite______macroRules__Mathlib__Tactic__tacticNth__rewrite____________1___closed__23));
v___x_227_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__NthRewrite______macroRules__Mathlib__Tactic__tacticNth__rewrite____________1___closed__24));
v___x_228_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_228_, 0, v___x_203_);
lean_ctor_set(v___x_228_, 1, v___x_227_);
v___x_229_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__NthRewrite______macroRules__Mathlib__Tactic__tacticNth__rewrite____________1___closed__26, &lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__NthRewrite______macroRules__Mathlib__Tactic__tacticNth__rewrite____________1___closed__26_once, _init_lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__NthRewrite______macroRules__Mathlib__Tactic__tacticNth__rewrite____________1___closed__26);
v___x_230_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__NthRewrite______macroRules__Mathlib__Tactic__tacticNth__rewrite____________1___closed__27));
v___x_231_ = l_Lean_addMacroScope(v_quotContext_197_, v___x_230_, v_currMacroScope_198_);
v___x_232_ = lean_alloc_ctor(3, 4, 0);
lean_ctor_set(v___x_232_, 0, v___x_203_);
lean_ctor_set(v___x_232_, 1, v___x_229_);
lean_ctor_set(v___x_232_, 2, v___x_231_);
lean_ctor_set(v___x_232_, 3, v___x_221_);
v___x_233_ = l_Lean_Syntax_node2(v___x_203_, v___x_226_, v___x_228_, v___x_232_);
v___x_234_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__NthRewrite______macroRules__Mathlib__Tactic__tacticNth__rewrite____________1___closed__29));
v___x_235_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__NthRewrite______macroRules__Mathlib__Tactic__tacticNth__rewrite____________1___closed__30));
v___x_236_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_236_, 0, v___x_203_);
lean_ctor_set(v___x_236_, 1, v___x_235_);
v_sz_237_ = lean_array_size(v_nums_200_);
v___x_238_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Mathlib_Tactic___aux__Mathlib__Tactic__NthRewrite______macroRules__Mathlib__Tactic__tacticNth__rewrite____________1_spec__1(v_sz_237_, v___x_211_, v_nums_200_);
v_sz_239_ = lean_array_size(v___x_238_);
v___x_240_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Mathlib_Tactic___aux__Mathlib__Tactic__NthRewrite______macroRules__Mathlib__Tactic__tacticNth__rewrite____________1_spec__0(v_sz_239_, v___x_211_, v___x_238_);
v___x_241_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__NthRewrite______macroRules__Mathlib__Tactic__tacticNth__rewrite____________1___closed__32, &lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__NthRewrite______macroRules__Mathlib__Tactic__tacticNth__rewrite____________1___closed__32_once, _init_lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__NthRewrite______macroRules__Mathlib__Tactic__tacticNth__rewrite____________1___closed__32);
v___x_242_ = l_Lean_mkSepArray(v___x_240_, v___x_241_);
lean_dec_ref(v___x_240_);
v___x_243_ = l_Array_append___redArg(v___x_208_, v___x_242_);
lean_dec_ref(v___x_242_);
v___x_244_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_244_, 0, v___x_203_);
lean_ctor_set(v___x_244_, 1, v___x_207_);
lean_ctor_set(v___x_244_, 2, v___x_243_);
v___x_245_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__NthRewrite______macroRules__Mathlib__Tactic__tacticNth__rewrite____________1___closed__33));
v___x_246_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_246_, 0, v___x_203_);
lean_ctor_set(v___x_246_, 1, v___x_245_);
v___x_247_ = l_Lean_Syntax_node3(v___x_203_, v___x_234_, v___x_236_, v___x_244_, v___x_246_);
v___x_248_ = l_Lean_Syntax_node1(v___x_203_, v___x_207_, v___x_247_);
v___x_249_ = l_Lean_Syntax_node2(v___x_203_, v___x_225_, v___x_233_, v___x_248_);
v___x_250_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__NthRewrite______macroRules__Mathlib__Tactic__tacticNth__rewrite____________1___closed__34));
v___x_251_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_251_, 0, v___x_203_);
lean_ctor_set(v___x_251_, 1, v___x_250_);
v___x_252_ = l_Lean_Syntax_node5(v___x_203_, v___x_215_, v___x_217_, v___x_222_, v___x_224_, v___x_249_, v___x_251_);
v___x_253_ = l_Lean_Syntax_node1(v___x_203_, v___x_214_, v___x_252_);
v___x_254_ = lean_array_push(v___x_213_, v___x_253_);
v___x_255_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_255_, 0, v___x_203_);
lean_ctor_set(v___x_255_, 1, v___x_207_);
lean_ctor_set(v___x_255_, 2, v___x_254_);
v___x_256_ = l_Lean_Syntax_node1(v___x_203_, v___x_201_, v___x_255_);
if (lean_obj_tag(v___y_196_) == 0)
{
lean_object* v___x_257_; 
v___x_257_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__NthRewrite______macroRules__Mathlib__Tactic__tacticNth__rewrite____________1___closed__35));
v___y_184_ = v___x_206_;
v___y_185_ = v___x_207_;
v___y_186_ = v___x_256_;
v___y_187_ = v___x_208_;
v___y_188_ = v___x_204_;
v___y_189_ = v___x_203_;
v___y_190_ = v___x_257_;
goto v___jp_183_;
}
else
{
lean_object* v_val_258_; lean_object* v___x_259_; lean_object* v___x_260_; 
v_val_258_ = lean_ctor_get(v___y_196_, 0);
lean_inc(v_val_258_);
lean_dec_ref_known(v___y_196_, 1);
v___x_259_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__NthRewrite______macroRules__Mathlib__Tactic__tacticNth__rewrite____________1___closed__35));
v___x_260_ = lean_array_push(v___x_259_, v_val_258_);
v___y_184_ = v___x_206_;
v___y_185_ = v___x_207_;
v___y_186_ = v___x_256_;
v___y_187_ = v___x_208_;
v___y_188_ = v___x_204_;
v___y_189_ = v___x_203_;
v___y_190_ = v___x_260_;
goto v___jp_183_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__NthRewrite______macroRules__Mathlib__Tactic__tacticNth__rewrite____________1___boxed(lean_object* v_x_273_, lean_object* v_a_274_, lean_object* v_a_275_){
_start:
{
lean_object* v_res_276_; 
v_res_276_ = lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__NthRewrite______macroRules__Mathlib__Tactic__tacticNth__rewrite____________1(v_x_273_, v_a_274_, v_a_275_);
lean_dec_ref(v_a_274_);
return v_res_276_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_tacticNth__rw___________00__closed__4(void){
_start:
{
lean_object* v___x_286_; lean_object* v___x_287_; lean_object* v___x_288_; lean_object* v___x_289_; 
v___x_286_ = l_Lean_Parser_Tactic_optConfig;
v___x_287_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_tacticNth__rw___________00__closed__3));
v___x_288_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_tacticNth__rewrite___________00__closed__5));
v___x_289_ = lean_alloc_ctor(2, 3, 0);
lean_ctor_set(v___x_289_, 0, v___x_288_);
lean_ctor_set(v___x_289_, 1, v___x_287_);
lean_ctor_set(v___x_289_, 2, v___x_286_);
return v___x_289_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_tacticNth__rw___________00__closed__5(void){
_start:
{
lean_object* v___x_290_; lean_object* v___x_291_; lean_object* v___x_292_; lean_object* v___x_293_; 
v___x_290_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_tacticNth__rewrite___________00__closed__14));
v___x_291_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_tacticNth__rw___________00__closed__4, &lp_mathlib_Mathlib_Tactic_tacticNth__rw___________00__closed__4_once, _init_lp_mathlib_Mathlib_Tactic_tacticNth__rw___________00__closed__4);
v___x_292_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_tacticNth__rewrite___________00__closed__5));
v___x_293_ = lean_alloc_ctor(2, 3, 0);
lean_ctor_set(v___x_293_, 0, v___x_292_);
lean_ctor_set(v___x_293_, 1, v___x_291_);
lean_ctor_set(v___x_293_, 2, v___x_290_);
return v___x_293_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_tacticNth__rw___________00__closed__6(void){
_start:
{
lean_object* v___x_294_; lean_object* v___x_295_; lean_object* v___x_296_; lean_object* v___x_297_; 
v___x_294_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_tacticNth__rewrite___________00__closed__21));
v___x_295_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_tacticNth__rw___________00__closed__5, &lp_mathlib_Mathlib_Tactic_tacticNth__rw___________00__closed__5_once, _init_lp_mathlib_Mathlib_Tactic_tacticNth__rw___________00__closed__5);
v___x_296_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_tacticNth__rewrite___________00__closed__5));
v___x_297_ = lean_alloc_ctor(2, 3, 0);
lean_ctor_set(v___x_297_, 0, v___x_296_);
lean_ctor_set(v___x_297_, 1, v___x_295_);
lean_ctor_set(v___x_297_, 2, v___x_294_);
return v___x_297_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_tacticNth__rw___________00__closed__7(void){
_start:
{
lean_object* v___x_298_; lean_object* v___x_299_; lean_object* v___x_300_; lean_object* v___x_301_; 
v___x_298_ = l_Lean_Parser_Tactic_rwRuleSeq;
v___x_299_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_tacticNth__rw___________00__closed__6, &lp_mathlib_Mathlib_Tactic_tacticNth__rw___________00__closed__6_once, _init_lp_mathlib_Mathlib_Tactic_tacticNth__rw___________00__closed__6);
v___x_300_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_tacticNth__rewrite___________00__closed__5));
v___x_301_ = lean_alloc_ctor(2, 3, 0);
lean_ctor_set(v___x_301_, 0, v___x_300_);
lean_ctor_set(v___x_301_, 1, v___x_299_);
lean_ctor_set(v___x_301_, 2, v___x_298_);
return v___x_301_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_tacticNth__rw___________00__closed__8(void){
_start:
{
lean_object* v___x_302_; lean_object* v___x_303_; lean_object* v___x_304_; lean_object* v___x_305_; 
v___x_302_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_tacticNth__rewrite___________00__closed__26, &lp_mathlib_Mathlib_Tactic_tacticNth__rewrite___________00__closed__26_once, _init_lp_mathlib_Mathlib_Tactic_tacticNth__rewrite___________00__closed__26);
v___x_303_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_tacticNth__rw___________00__closed__7, &lp_mathlib_Mathlib_Tactic_tacticNth__rw___________00__closed__7_once, _init_lp_mathlib_Mathlib_Tactic_tacticNth__rw___________00__closed__7);
v___x_304_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_tacticNth__rewrite___________00__closed__5));
v___x_305_ = lean_alloc_ctor(2, 3, 0);
lean_ctor_set(v___x_305_, 0, v___x_304_);
lean_ctor_set(v___x_305_, 1, v___x_303_);
lean_ctor_set(v___x_305_, 2, v___x_302_);
return v___x_305_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_tacticNth__rw___________00__closed__9(void){
_start:
{
lean_object* v___x_306_; lean_object* v___x_307_; lean_object* v___x_308_; lean_object* v___x_309_; 
v___x_306_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_tacticNth__rw___________00__closed__8, &lp_mathlib_Mathlib_Tactic_tacticNth__rw___________00__closed__8_once, _init_lp_mathlib_Mathlib_Tactic_tacticNth__rw___________00__closed__8);
v___x_307_ = lean_unsigned_to_nat(1022u);
v___x_308_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_tacticNth__rw___________00__closed__1));
v___x_309_ = lean_alloc_ctor(3, 3, 0);
lean_ctor_set(v___x_309_, 0, v___x_308_);
lean_ctor_set(v___x_309_, 1, v___x_307_);
lean_ctor_set(v___x_309_, 2, v___x_306_);
return v___x_309_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_tacticNth__rw__________(void){
_start:
{
lean_object* v___x_310_; 
v___x_310_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_tacticNth__rw___________00__closed__9, &lp_mathlib_Mathlib_Tactic_tacticNth__rw___________00__closed__9_once, _init_lp_mathlib_Mathlib_Tactic_tacticNth__rw___________00__closed__9);
return v___x_310_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__NthRewrite______macroRules__Mathlib__Tactic__tacticNth__rw____________1(lean_object* v_x_318_, lean_object* v_a_319_, lean_object* v_a_320_){
_start:
{
lean_object* v___x_321_; uint8_t v___x_322_; 
v___x_321_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_tacticNth__rw___________00__closed__1));
lean_inc(v_x_318_);
v___x_322_ = l_Lean_Syntax_isOfKind(v_x_318_, v___x_321_);
if (v___x_322_ == 0)
{
lean_object* v___x_323_; lean_object* v___x_324_; 
lean_dec(v_x_318_);
v___x_323_ = lean_box(1);
v___x_324_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_324_, 0, v___x_323_);
lean_ctor_set(v___x_324_, 1, v_a_320_);
return v___x_324_;
}
else
{
lean_object* v___x_325_; lean_object* v___x_326_; lean_object* v___x_327_; lean_object* v___x_328_; lean_object* v___x_329_; lean_object* v___x_330_; lean_object* v___y_332_; lean_object* v___y_333_; lean_object* v___y_334_; lean_object* v___y_335_; lean_object* v___y_336_; lean_object* v___y_337_; lean_object* v___y_338_; lean_object* v___y_344_; lean_object* v___x_409_; lean_object* v___x_410_; lean_object* v___x_411_; 
v___x_325_ = lean_unsigned_to_nat(1u);
v___x_326_ = l_Lean_Syntax_getArg(v_x_318_, v___x_325_);
v___x_327_ = lean_unsigned_to_nat(3u);
v___x_328_ = l_Lean_Syntax_getArg(v_x_318_, v___x_327_);
v___x_329_ = lean_unsigned_to_nat(4u);
v___x_330_ = l_Lean_Syntax_getArg(v_x_318_, v___x_329_);
v___x_409_ = lean_unsigned_to_nat(5u);
v___x_410_ = l_Lean_Syntax_getArg(v_x_318_, v___x_409_);
lean_dec(v_x_318_);
v___x_411_ = l_Lean_Syntax_getOptional_x3f(v___x_410_);
lean_dec(v___x_410_);
if (lean_obj_tag(v___x_411_) == 0)
{
lean_object* v___x_412_; 
v___x_412_ = lean_box(0);
v___y_344_ = v___x_412_;
goto v___jp_343_;
}
else
{
lean_object* v_val_413_; lean_object* v___x_415_; uint8_t v_isShared_416_; uint8_t v_isSharedCheck_420_; 
v_val_413_ = lean_ctor_get(v___x_411_, 0);
v_isSharedCheck_420_ = !lean_is_exclusive(v___x_411_);
if (v_isSharedCheck_420_ == 0)
{
v___x_415_ = v___x_411_;
v_isShared_416_ = v_isSharedCheck_420_;
goto v_resetjp_414_;
}
else
{
lean_inc(v_val_413_);
lean_dec(v___x_411_);
v___x_415_ = lean_box(0);
v_isShared_416_ = v_isSharedCheck_420_;
goto v_resetjp_414_;
}
v_resetjp_414_:
{
lean_object* v___x_418_; 
if (v_isShared_416_ == 0)
{
v___x_418_ = v___x_415_;
goto v_reusejp_417_;
}
else
{
lean_object* v_reuseFailAlloc_419_; 
v_reuseFailAlloc_419_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_419_, 0, v_val_413_);
v___x_418_ = v_reuseFailAlloc_419_;
goto v_reusejp_417_;
}
v_reusejp_417_:
{
v___y_344_ = v___x_418_;
goto v___jp_343_;
}
}
}
v___jp_331_:
{
lean_object* v___x_339_; lean_object* v___x_340_; lean_object* v___x_341_; lean_object* v___x_342_; 
v___x_339_ = l_Array_append___redArg(v___y_332_, v___y_338_);
lean_dec_ref(v___y_338_);
lean_inc(v___y_334_);
v___x_340_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_340_, 0, v___y_334_);
lean_ctor_set(v___x_340_, 1, v___y_333_);
lean_ctor_set(v___x_340_, 2, v___x_339_);
lean_inc(v___y_335_);
v___x_341_ = l_Lean_Syntax_node4(v___y_334_, v___y_335_, v___y_336_, v___y_337_, v___x_330_, v___x_340_);
v___x_342_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_342_, 0, v___x_341_);
lean_ctor_set(v___x_342_, 1, v_a_320_);
return v___x_342_;
}
v___jp_343_:
{
lean_object* v_quotContext_345_; lean_object* v_currMacroScope_346_; lean_object* v_ref_347_; lean_object* v_nums_348_; lean_object* v___x_349_; uint8_t v___x_350_; lean_object* v___x_351_; lean_object* v___x_352_; lean_object* v___x_353_; lean_object* v___x_354_; lean_object* v___x_355_; lean_object* v___x_356_; lean_object* v___x_357_; size_t v_sz_358_; size_t v___x_359_; lean_object* v___x_360_; lean_object* v___x_361_; lean_object* v___x_362_; lean_object* v___x_363_; lean_object* v___x_364_; lean_object* v___x_365_; lean_object* v___x_366_; lean_object* v___x_367_; lean_object* v___x_368_; lean_object* v___x_369_; lean_object* v___x_370_; lean_object* v___x_371_; lean_object* v___x_372_; lean_object* v___x_373_; lean_object* v___x_374_; lean_object* v___x_375_; lean_object* v___x_376_; lean_object* v___x_377_; lean_object* v___x_378_; lean_object* v___x_379_; lean_object* v___x_380_; lean_object* v___x_381_; lean_object* v___x_382_; lean_object* v___x_383_; lean_object* v___x_384_; size_t v_sz_385_; lean_object* v___x_386_; size_t v_sz_387_; lean_object* v___x_388_; lean_object* v___x_389_; lean_object* v___x_390_; lean_object* v___x_391_; lean_object* v___x_392_; lean_object* v___x_393_; lean_object* v___x_394_; lean_object* v___x_395_; lean_object* v___x_396_; lean_object* v___x_397_; lean_object* v___x_398_; lean_object* v___x_399_; lean_object* v___x_400_; lean_object* v___x_401_; lean_object* v___x_402_; lean_object* v___x_403_; lean_object* v___x_404_; 
v_quotContext_345_ = lean_ctor_get(v_a_319_, 1);
v_currMacroScope_346_ = lean_ctor_get(v_a_319_, 2);
v_ref_347_ = lean_ctor_get(v_a_319_, 5);
v_nums_348_ = l_Lean_Syntax_getArgs(v___x_328_);
lean_dec(v___x_328_);
v___x_349_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__NthRewrite______macroRules__Mathlib__Tactic__tacticNth__rewrite____________1___closed__3));
v___x_350_ = 0;
v___x_351_ = l_Lean_SourceInfo_fromRef(v_ref_347_, v___x_350_);
v___x_352_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__NthRewrite______macroRules__Mathlib__Tactic__tacticNth__rw____________1___closed__1));
v___x_353_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__NthRewrite______macroRules__Mathlib__Tactic__tacticNth__rw____________1___closed__2));
lean_inc_n(v___x_351_, 18);
v___x_354_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_354_, 0, v___x_351_);
lean_ctor_set(v___x_354_, 1, v___x_353_);
v___x_355_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__NthRewrite______macroRules__Mathlib__Tactic__tacticNth__rewrite____________1___closed__8));
v___x_356_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__NthRewrite______macroRules__Mathlib__Tactic__tacticNth__rewrite____________1___closed__9, &lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__NthRewrite______macroRules__Mathlib__Tactic__tacticNth__rewrite____________1___closed__9_once, _init_lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__NthRewrite______macroRules__Mathlib__Tactic__tacticNth__rewrite____________1___closed__9);
v___x_357_ = l_Lean_Parser_Tactic_getConfigItems(v___x_326_);
v_sz_358_ = lean_array_size(v___x_357_);
v___x_359_ = ((size_t)0ULL);
v___x_360_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Mathlib_Tactic___aux__Mathlib__Tactic__NthRewrite______macroRules__Mathlib__Tactic__tacticNth__rewrite____________1_spec__0(v_sz_358_, v___x_359_, v___x_357_);
v___x_361_ = l_Array_append___redArg(v___x_356_, v___x_360_);
lean_dec_ref(v___x_360_);
v___x_362_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__NthRewrite______macroRules__Mathlib__Tactic__tacticNth__rewrite____________1___closed__11));
v___x_363_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__NthRewrite______macroRules__Mathlib__Tactic__tacticNth__rewrite____________1___closed__13));
v___x_364_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__NthRewrite______macroRules__Mathlib__Tactic__tacticNth__rewrite____________1___closed__14));
v___x_365_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_365_, 0, v___x_351_);
lean_ctor_set(v___x_365_, 1, v___x_364_);
v___x_366_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__NthRewrite______macroRules__Mathlib__Tactic__tacticNth__rewrite____________1___closed__16, &lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__NthRewrite______macroRules__Mathlib__Tactic__tacticNth__rewrite____________1___closed__16_once, _init_lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__NthRewrite______macroRules__Mathlib__Tactic__tacticNth__rewrite____________1___closed__16);
v___x_367_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__NthRewrite______macroRules__Mathlib__Tactic__tacticNth__rewrite____________1___closed__17));
lean_inc_n(v_currMacroScope_346_, 2);
lean_inc_n(v_quotContext_345_, 2);
v___x_368_ = l_Lean_addMacroScope(v_quotContext_345_, v___x_367_, v_currMacroScope_346_);
v___x_369_ = lean_box(0);
v___x_370_ = lean_alloc_ctor(3, 4, 0);
lean_ctor_set(v___x_370_, 0, v___x_351_);
lean_ctor_set(v___x_370_, 1, v___x_366_);
lean_ctor_set(v___x_370_, 2, v___x_368_);
lean_ctor_set(v___x_370_, 3, v___x_369_);
v___x_371_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__NthRewrite______macroRules__Mathlib__Tactic__tacticNth__rewrite____________1___closed__18));
v___x_372_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_372_, 0, v___x_351_);
lean_ctor_set(v___x_372_, 1, v___x_371_);
v___x_373_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__NthRewrite______macroRules__Mathlib__Tactic__tacticNth__rewrite____________1___closed__21));
v___x_374_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__NthRewrite______macroRules__Mathlib__Tactic__tacticNth__rewrite____________1___closed__23));
v___x_375_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__NthRewrite______macroRules__Mathlib__Tactic__tacticNth__rewrite____________1___closed__24));
v___x_376_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_376_, 0, v___x_351_);
lean_ctor_set(v___x_376_, 1, v___x_375_);
v___x_377_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__NthRewrite______macroRules__Mathlib__Tactic__tacticNth__rewrite____________1___closed__26, &lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__NthRewrite______macroRules__Mathlib__Tactic__tacticNth__rewrite____________1___closed__26_once, _init_lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__NthRewrite______macroRules__Mathlib__Tactic__tacticNth__rewrite____________1___closed__26);
v___x_378_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__NthRewrite______macroRules__Mathlib__Tactic__tacticNth__rewrite____________1___closed__27));
v___x_379_ = l_Lean_addMacroScope(v_quotContext_345_, v___x_378_, v_currMacroScope_346_);
v___x_380_ = lean_alloc_ctor(3, 4, 0);
lean_ctor_set(v___x_380_, 0, v___x_351_);
lean_ctor_set(v___x_380_, 1, v___x_377_);
lean_ctor_set(v___x_380_, 2, v___x_379_);
lean_ctor_set(v___x_380_, 3, v___x_369_);
v___x_381_ = l_Lean_Syntax_node2(v___x_351_, v___x_374_, v___x_376_, v___x_380_);
v___x_382_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__NthRewrite______macroRules__Mathlib__Tactic__tacticNth__rewrite____________1___closed__29));
v___x_383_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__NthRewrite______macroRules__Mathlib__Tactic__tacticNth__rewrite____________1___closed__30));
v___x_384_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_384_, 0, v___x_351_);
lean_ctor_set(v___x_384_, 1, v___x_383_);
v_sz_385_ = lean_array_size(v_nums_348_);
v___x_386_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Mathlib_Tactic___aux__Mathlib__Tactic__NthRewrite______macroRules__Mathlib__Tactic__tacticNth__rewrite____________1_spec__1(v_sz_385_, v___x_359_, v_nums_348_);
v_sz_387_ = lean_array_size(v___x_386_);
v___x_388_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Mathlib_Tactic___aux__Mathlib__Tactic__NthRewrite______macroRules__Mathlib__Tactic__tacticNth__rewrite____________1_spec__0(v_sz_387_, v___x_359_, v___x_386_);
v___x_389_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__NthRewrite______macroRules__Mathlib__Tactic__tacticNth__rewrite____________1___closed__32, &lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__NthRewrite______macroRules__Mathlib__Tactic__tacticNth__rewrite____________1___closed__32_once, _init_lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__NthRewrite______macroRules__Mathlib__Tactic__tacticNth__rewrite____________1___closed__32);
v___x_390_ = l_Lean_mkSepArray(v___x_388_, v___x_389_);
lean_dec_ref(v___x_388_);
v___x_391_ = l_Array_append___redArg(v___x_356_, v___x_390_);
lean_dec_ref(v___x_390_);
v___x_392_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_392_, 0, v___x_351_);
lean_ctor_set(v___x_392_, 1, v___x_355_);
lean_ctor_set(v___x_392_, 2, v___x_391_);
v___x_393_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__NthRewrite______macroRules__Mathlib__Tactic__tacticNth__rewrite____________1___closed__33));
v___x_394_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_394_, 0, v___x_351_);
lean_ctor_set(v___x_394_, 1, v___x_393_);
v___x_395_ = l_Lean_Syntax_node3(v___x_351_, v___x_382_, v___x_384_, v___x_392_, v___x_394_);
v___x_396_ = l_Lean_Syntax_node1(v___x_351_, v___x_355_, v___x_395_);
v___x_397_ = l_Lean_Syntax_node2(v___x_351_, v___x_373_, v___x_381_, v___x_396_);
v___x_398_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__NthRewrite______macroRules__Mathlib__Tactic__tacticNth__rewrite____________1___closed__34));
v___x_399_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_399_, 0, v___x_351_);
lean_ctor_set(v___x_399_, 1, v___x_398_);
v___x_400_ = l_Lean_Syntax_node5(v___x_351_, v___x_363_, v___x_365_, v___x_370_, v___x_372_, v___x_397_, v___x_399_);
v___x_401_ = l_Lean_Syntax_node1(v___x_351_, v___x_362_, v___x_400_);
v___x_402_ = lean_array_push(v___x_361_, v___x_401_);
v___x_403_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_403_, 0, v___x_351_);
lean_ctor_set(v___x_403_, 1, v___x_355_);
lean_ctor_set(v___x_403_, 2, v___x_402_);
v___x_404_ = l_Lean_Syntax_node1(v___x_351_, v___x_349_, v___x_403_);
if (lean_obj_tag(v___y_344_) == 0)
{
lean_object* v___x_405_; 
v___x_405_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__NthRewrite______macroRules__Mathlib__Tactic__tacticNth__rewrite____________1___closed__35));
v___y_332_ = v___x_356_;
v___y_333_ = v___x_355_;
v___y_334_ = v___x_351_;
v___y_335_ = v___x_352_;
v___y_336_ = v___x_354_;
v___y_337_ = v___x_404_;
v___y_338_ = v___x_405_;
goto v___jp_331_;
}
else
{
lean_object* v_val_406_; lean_object* v___x_407_; lean_object* v___x_408_; 
v_val_406_ = lean_ctor_get(v___y_344_, 0);
lean_inc(v_val_406_);
lean_dec_ref_known(v___y_344_, 1);
v___x_407_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__NthRewrite______macroRules__Mathlib__Tactic__tacticNth__rewrite____________1___closed__35));
v___x_408_ = lean_array_push(v___x_407_, v_val_406_);
v___y_332_ = v___x_356_;
v___y_333_ = v___x_355_;
v___y_334_ = v___x_351_;
v___y_335_ = v___x_352_;
v___y_336_ = v___x_354_;
v___y_337_ = v___x_404_;
v___y_338_ = v___x_408_;
goto v___jp_331_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__NthRewrite______macroRules__Mathlib__Tactic__tacticNth__rw____________1___boxed(lean_object* v_x_421_, lean_object* v_a_422_, lean_object* v_a_423_){
_start:
{
lean_object* v_res_424_; 
v_res_424_ = lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__NthRewrite______macroRules__Mathlib__Tactic__tacticNth__rw____________1(v_x_421_, v_a_422_, v_a_423_);
lean_dec_ref(v_a_422_);
return v_res_424_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Init(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_Tactic_NthRewrite(uint8_t builtin) {
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
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_Tactic_NthRewrite(uint8_t builtin) {
lean_object * res;
if (_G_meta_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_meta_initialized = true;
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
lp_mathlib_Mathlib_Tactic_tacticNth__rewrite__________ = _init_lp_mathlib_Mathlib_Tactic_tacticNth__rewrite__________();
lean_mark_persistent(lp_mathlib_Mathlib_Tactic_tacticNth__rewrite__________);
lp_mathlib_Mathlib_Tactic_tacticNth__rw__________ = _init_lp_mathlib_Mathlib_Tactic_tacticNth__rw__________();
lean_mark_persistent(lp_mathlib_Mathlib_Tactic_tacticNth__rw__________);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Init(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Tactic_NthRewrite(uint8_t builtin) {
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
res = runtime_initialize_mathlib_Mathlib_Tactic_NthRewrite(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_Tactic_NthRewrite(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_Tactic_NthRewrite(builtin);
}
#ifdef __cplusplus
}
#endif
