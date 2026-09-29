// Lean compiler output
// Module: Mathlib.Logic.Equiv.Defs
// Imports: public import Init public meta import Init public import Mathlib.Basic.Unique public import Mathlib.Data.FunLike.Equiv public import Mathlib.Data.Quot public import Mathlib.Data.Subtype public import Mathlib.Tactic.Simps
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
lean_object* l_Lean_mkAtom(lean_object*);
lean_object* lean_mk_empty_array_with_capacity(lean_object*);
lean_object* lean_array_push(lean_object*, lean_object*);
lean_object* l_Lean_Name_mkStr4(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Name_mkStr1(lean_object*);
lean_object* lean_nat_mod(lean_object*, lean_object*);
lean_object* l_Lean_Name_mkStr5(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_uniqueElim___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
uint8_t lp_mathlib_Function_Injective_decidableEq___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
uint8_t l_Lean_Syntax_isOfKind(lean_object*, lean_object*);
lean_object* l_Lean_Syntax_getArg(lean_object*, lean_object*);
uint8_t l_Lean_Syntax_matchesNull(lean_object*, lean_object*);
lean_object* l_Lean_replaceRef(lean_object*, lean_object*);
lean_object* l_Lean_SourceInfo_fromRef(lean_object*, uint8_t);
lean_object* l_Lean_Syntax_node3(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_String_toRawSubstring_x27(lean_object*);
lean_object* l_Lean_addMacroScope(lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Syntax_node2(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Quot_map(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
uint8_t lean_nat_dec_eq(lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Equiv_left__inv___autoParam___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "Lean"};
static const lean_object* lp_mathlib_Equiv_left__inv___autoParam___closed__0 = (const lean_object*)&lp_mathlib_Equiv_left__inv___autoParam___closed__0_value;
static const lean_string_object lp_mathlib_Equiv_left__inv___autoParam___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "Parser"};
static const lean_object* lp_mathlib_Equiv_left__inv___autoParam___closed__1 = (const lean_object*)&lp_mathlib_Equiv_left__inv___autoParam___closed__1_value;
static const lean_string_object lp_mathlib_Equiv_left__inv___autoParam___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "Tactic"};
static const lean_object* lp_mathlib_Equiv_left__inv___autoParam___closed__2 = (const lean_object*)&lp_mathlib_Equiv_left__inv___autoParam___closed__2_value;
static const lean_string_object lp_mathlib_Equiv_left__inv___autoParam___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 9, .m_data = "tacticSeq"};
static const lean_object* lp_mathlib_Equiv_left__inv___autoParam___closed__3 = (const lean_object*)&lp_mathlib_Equiv_left__inv___autoParam___closed__3_value;
static const lean_ctor_object lp_mathlib_Equiv_left__inv___autoParam___closed__4_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Equiv_left__inv___autoParam___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib_Equiv_left__inv___autoParam___closed__4_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Equiv_left__inv___autoParam___closed__4_value_aux_0),((lean_object*)&lp_mathlib_Equiv_left__inv___autoParam___closed__1_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib_Equiv_left__inv___autoParam___closed__4_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Equiv_left__inv___autoParam___closed__4_value_aux_1),((lean_object*)&lp_mathlib_Equiv_left__inv___autoParam___closed__2_value),LEAN_SCALAR_PTR_LITERAL(166, 58, 35, 182, 187, 130, 147, 254)}};
static const lean_ctor_object lp_mathlib_Equiv_left__inv___autoParam___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Equiv_left__inv___autoParam___closed__4_value_aux_2),((lean_object*)&lp_mathlib_Equiv_left__inv___autoParam___closed__3_value),LEAN_SCALAR_PTR_LITERAL(212, 140, 85, 215, 241, 69, 7, 118)}};
static const lean_object* lp_mathlib_Equiv_left__inv___autoParam___closed__4 = (const lean_object*)&lp_mathlib_Equiv_left__inv___autoParam___closed__4_value;
static const lean_array_object lp_mathlib_Equiv_left__inv___autoParam___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_array_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 246}, .m_size = 0, .m_capacity = 0, .m_data = {}};
static const lean_object* lp_mathlib_Equiv_left__inv___autoParam___closed__5 = (const lean_object*)&lp_mathlib_Equiv_left__inv___autoParam___closed__5_value;
static const lean_string_object lp_mathlib_Equiv_left__inv___autoParam___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 19, .m_capacity = 19, .m_length = 18, .m_data = "tacticSeq1Indented"};
static const lean_object* lp_mathlib_Equiv_left__inv___autoParam___closed__6 = (const lean_object*)&lp_mathlib_Equiv_left__inv___autoParam___closed__6_value;
static const lean_ctor_object lp_mathlib_Equiv_left__inv___autoParam___closed__7_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Equiv_left__inv___autoParam___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib_Equiv_left__inv___autoParam___closed__7_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Equiv_left__inv___autoParam___closed__7_value_aux_0),((lean_object*)&lp_mathlib_Equiv_left__inv___autoParam___closed__1_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib_Equiv_left__inv___autoParam___closed__7_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Equiv_left__inv___autoParam___closed__7_value_aux_1),((lean_object*)&lp_mathlib_Equiv_left__inv___autoParam___closed__2_value),LEAN_SCALAR_PTR_LITERAL(166, 58, 35, 182, 187, 130, 147, 254)}};
static const lean_ctor_object lp_mathlib_Equiv_left__inv___autoParam___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Equiv_left__inv___autoParam___closed__7_value_aux_2),((lean_object*)&lp_mathlib_Equiv_left__inv___autoParam___closed__6_value),LEAN_SCALAR_PTR_LITERAL(223, 90, 160, 238, 133, 180, 23, 239)}};
static const lean_object* lp_mathlib_Equiv_left__inv___autoParam___closed__7 = (const lean_object*)&lp_mathlib_Equiv_left__inv___autoParam___closed__7_value;
static const lean_string_object lp_mathlib_Equiv_left__inv___autoParam___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "null"};
static const lean_object* lp_mathlib_Equiv_left__inv___autoParam___closed__8 = (const lean_object*)&lp_mathlib_Equiv_left__inv___autoParam___closed__8_value;
static const lean_ctor_object lp_mathlib_Equiv_left__inv___autoParam___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Equiv_left__inv___autoParam___closed__8_value),LEAN_SCALAR_PTR_LITERAL(24, 58, 49, 223, 146, 207, 197, 136)}};
static const lean_object* lp_mathlib_Equiv_left__inv___autoParam___closed__9 = (const lean_object*)&lp_mathlib_Equiv_left__inv___autoParam___closed__9_value;
static const lean_string_object lp_mathlib_Equiv_left__inv___autoParam___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "intro"};
static const lean_object* lp_mathlib_Equiv_left__inv___autoParam___closed__10 = (const lean_object*)&lp_mathlib_Equiv_left__inv___autoParam___closed__10_value;
static const lean_ctor_object lp_mathlib_Equiv_left__inv___autoParam___closed__11_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Equiv_left__inv___autoParam___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib_Equiv_left__inv___autoParam___closed__11_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Equiv_left__inv___autoParam___closed__11_value_aux_0),((lean_object*)&lp_mathlib_Equiv_left__inv___autoParam___closed__1_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib_Equiv_left__inv___autoParam___closed__11_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Equiv_left__inv___autoParam___closed__11_value_aux_1),((lean_object*)&lp_mathlib_Equiv_left__inv___autoParam___closed__2_value),LEAN_SCALAR_PTR_LITERAL(166, 58, 35, 182, 187, 130, 147, 254)}};
static const lean_ctor_object lp_mathlib_Equiv_left__inv___autoParam___closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Equiv_left__inv___autoParam___closed__11_value_aux_2),((lean_object*)&lp_mathlib_Equiv_left__inv___autoParam___closed__10_value),LEAN_SCALAR_PTR_LITERAL(41, 145, 9, 18, 75, 146, 159, 78)}};
static const lean_object* lp_mathlib_Equiv_left__inv___autoParam___closed__11 = (const lean_object*)&lp_mathlib_Equiv_left__inv___autoParam___closed__11_value;
static lean_once_cell_t lp_mathlib_Equiv_left__inv___autoParam___closed__12_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Equiv_left__inv___autoParam___closed__12;
static lean_once_cell_t lp_mathlib_Equiv_left__inv___autoParam___closed__13_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Equiv_left__inv___autoParam___closed__13;
static const lean_ctor_object lp_mathlib_Equiv_left__inv___autoParam___closed__14_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(2) << 1) | 1)),((lean_object*)&lp_mathlib_Equiv_left__inv___autoParam___closed__9_value),((lean_object*)&lp_mathlib_Equiv_left__inv___autoParam___closed__5_value)}};
static const lean_object* lp_mathlib_Equiv_left__inv___autoParam___closed__14 = (const lean_object*)&lp_mathlib_Equiv_left__inv___autoParam___closed__14_value;
static lean_once_cell_t lp_mathlib_Equiv_left__inv___autoParam___closed__15_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Equiv_left__inv___autoParam___closed__15;
static lean_once_cell_t lp_mathlib_Equiv_left__inv___autoParam___closed__16_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Equiv_left__inv___autoParam___closed__16;
static lean_once_cell_t lp_mathlib_Equiv_left__inv___autoParam___closed__17_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Equiv_left__inv___autoParam___closed__17;
static const lean_string_object lp_mathlib_Equiv_left__inv___autoParam___closed__18_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = ";"};
static const lean_object* lp_mathlib_Equiv_left__inv___autoParam___closed__18 = (const lean_object*)&lp_mathlib_Equiv_left__inv___autoParam___closed__18_value;
static lean_once_cell_t lp_mathlib_Equiv_left__inv___autoParam___closed__19_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Equiv_left__inv___autoParam___closed__19;
static lean_once_cell_t lp_mathlib_Equiv_left__inv___autoParam___closed__20_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Equiv_left__inv___autoParam___closed__20;
static const lean_string_object lp_mathlib_Equiv_left__inv___autoParam___closed__21_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "first"};
static const lean_object* lp_mathlib_Equiv_left__inv___autoParam___closed__21 = (const lean_object*)&lp_mathlib_Equiv_left__inv___autoParam___closed__21_value;
static const lean_ctor_object lp_mathlib_Equiv_left__inv___autoParam___closed__22_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Equiv_left__inv___autoParam___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib_Equiv_left__inv___autoParam___closed__22_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Equiv_left__inv___autoParam___closed__22_value_aux_0),((lean_object*)&lp_mathlib_Equiv_left__inv___autoParam___closed__1_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib_Equiv_left__inv___autoParam___closed__22_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Equiv_left__inv___autoParam___closed__22_value_aux_1),((lean_object*)&lp_mathlib_Equiv_left__inv___autoParam___closed__2_value),LEAN_SCALAR_PTR_LITERAL(166, 58, 35, 182, 187, 130, 147, 254)}};
static const lean_ctor_object lp_mathlib_Equiv_left__inv___autoParam___closed__22_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Equiv_left__inv___autoParam___closed__22_value_aux_2),((lean_object*)&lp_mathlib_Equiv_left__inv___autoParam___closed__21_value),LEAN_SCALAR_PTR_LITERAL(59, 232, 35, 17, 172, 62, 48, 174)}};
static const lean_object* lp_mathlib_Equiv_left__inv___autoParam___closed__22 = (const lean_object*)&lp_mathlib_Equiv_left__inv___autoParam___closed__22_value;
static lean_once_cell_t lp_mathlib_Equiv_left__inv___autoParam___closed__23_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Equiv_left__inv___autoParam___closed__23;
static lean_once_cell_t lp_mathlib_Equiv_left__inv___autoParam___closed__24_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Equiv_left__inv___autoParam___closed__24;
static const lean_string_object lp_mathlib_Equiv_left__inv___autoParam___closed__25_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "group"};
static const lean_object* lp_mathlib_Equiv_left__inv___autoParam___closed__25 = (const lean_object*)&lp_mathlib_Equiv_left__inv___autoParam___closed__25_value;
static const lean_ctor_object lp_mathlib_Equiv_left__inv___autoParam___closed__26_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Equiv_left__inv___autoParam___closed__25_value),LEAN_SCALAR_PTR_LITERAL(206, 113, 20, 57, 188, 177, 187, 30)}};
static const lean_object* lp_mathlib_Equiv_left__inv___autoParam___closed__26 = (const lean_object*)&lp_mathlib_Equiv_left__inv___autoParam___closed__26_value;
static const lean_string_object lp_mathlib_Equiv_left__inv___autoParam___closed__27_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = "|"};
static const lean_object* lp_mathlib_Equiv_left__inv___autoParam___closed__27 = (const lean_object*)&lp_mathlib_Equiv_left__inv___autoParam___closed__27_value;
static lean_once_cell_t lp_mathlib_Equiv_left__inv___autoParam___closed__28_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Equiv_left__inv___autoParam___closed__28;
static lean_once_cell_t lp_mathlib_Equiv_left__inv___autoParam___closed__29_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Equiv_left__inv___autoParam___closed__29;
static const lean_string_object lp_mathlib_Equiv_left__inv___autoParam___closed__30_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 9, .m_data = "tacticRfl"};
static const lean_object* lp_mathlib_Equiv_left__inv___autoParam___closed__30 = (const lean_object*)&lp_mathlib_Equiv_left__inv___autoParam___closed__30_value;
static const lean_ctor_object lp_mathlib_Equiv_left__inv___autoParam___closed__31_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Equiv_left__inv___autoParam___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib_Equiv_left__inv___autoParam___closed__31_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Equiv_left__inv___autoParam___closed__31_value_aux_0),((lean_object*)&lp_mathlib_Equiv_left__inv___autoParam___closed__1_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib_Equiv_left__inv___autoParam___closed__31_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Equiv_left__inv___autoParam___closed__31_value_aux_1),((lean_object*)&lp_mathlib_Equiv_left__inv___autoParam___closed__2_value),LEAN_SCALAR_PTR_LITERAL(166, 58, 35, 182, 187, 130, 147, 254)}};
static const lean_ctor_object lp_mathlib_Equiv_left__inv___autoParam___closed__31_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Equiv_left__inv___autoParam___closed__31_value_aux_2),((lean_object*)&lp_mathlib_Equiv_left__inv___autoParam___closed__30_value),LEAN_SCALAR_PTR_LITERAL(201, 188, 173, 198, 169, 252, 183, 45)}};
static const lean_object* lp_mathlib_Equiv_left__inv___autoParam___closed__31 = (const lean_object*)&lp_mathlib_Equiv_left__inv___autoParam___closed__31_value;
static const lean_string_object lp_mathlib_Equiv_left__inv___autoParam___closed__32_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = "rfl"};
static const lean_object* lp_mathlib_Equiv_left__inv___autoParam___closed__32 = (const lean_object*)&lp_mathlib_Equiv_left__inv___autoParam___closed__32_value;
static lean_once_cell_t lp_mathlib_Equiv_left__inv___autoParam___closed__33_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Equiv_left__inv___autoParam___closed__33;
static lean_once_cell_t lp_mathlib_Equiv_left__inv___autoParam___closed__34_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Equiv_left__inv___autoParam___closed__34;
static lean_once_cell_t lp_mathlib_Equiv_left__inv___autoParam___closed__35_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Equiv_left__inv___autoParam___closed__35;
static lean_once_cell_t lp_mathlib_Equiv_left__inv___autoParam___closed__36_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Equiv_left__inv___autoParam___closed__36;
static lean_once_cell_t lp_mathlib_Equiv_left__inv___autoParam___closed__37_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Equiv_left__inv___autoParam___closed__37;
static lean_once_cell_t lp_mathlib_Equiv_left__inv___autoParam___closed__38_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Equiv_left__inv___autoParam___closed__38;
static lean_once_cell_t lp_mathlib_Equiv_left__inv___autoParam___closed__39_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Equiv_left__inv___autoParam___closed__39;
static lean_once_cell_t lp_mathlib_Equiv_left__inv___autoParam___closed__40_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Equiv_left__inv___autoParam___closed__40;
static lean_once_cell_t lp_mathlib_Equiv_left__inv___autoParam___closed__41_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Equiv_left__inv___autoParam___closed__41;
static lean_once_cell_t lp_mathlib_Equiv_left__inv___autoParam___closed__42_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Equiv_left__inv___autoParam___closed__42;
static lean_once_cell_t lp_mathlib_Equiv_left__inv___autoParam___closed__43_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Equiv_left__inv___autoParam___closed__43;
static lean_once_cell_t lp_mathlib_Equiv_left__inv___autoParam___closed__44_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Equiv_left__inv___autoParam___closed__44;
static const lean_string_object lp_mathlib_Equiv_left__inv___autoParam___closed__45_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 12, .m_capacity = 12, .m_length = 11, .m_data = "tactic_<;>_"};
static const lean_object* lp_mathlib_Equiv_left__inv___autoParam___closed__45 = (const lean_object*)&lp_mathlib_Equiv_left__inv___autoParam___closed__45_value;
static const lean_ctor_object lp_mathlib_Equiv_left__inv___autoParam___closed__46_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Equiv_left__inv___autoParam___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib_Equiv_left__inv___autoParam___closed__46_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Equiv_left__inv___autoParam___closed__46_value_aux_0),((lean_object*)&lp_mathlib_Equiv_left__inv___autoParam___closed__1_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib_Equiv_left__inv___autoParam___closed__46_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Equiv_left__inv___autoParam___closed__46_value_aux_1),((lean_object*)&lp_mathlib_Equiv_left__inv___autoParam___closed__2_value),LEAN_SCALAR_PTR_LITERAL(166, 58, 35, 182, 187, 130, 147, 254)}};
static const lean_ctor_object lp_mathlib_Equiv_left__inv___autoParam___closed__46_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Equiv_left__inv___autoParam___closed__46_value_aux_2),((lean_object*)&lp_mathlib_Equiv_left__inv___autoParam___closed__45_value),LEAN_SCALAR_PTR_LITERAL(31, 118, 44, 159, 195, 11, 47, 176)}};
static const lean_object* lp_mathlib_Equiv_left__inv___autoParam___closed__46 = (const lean_object*)&lp_mathlib_Equiv_left__inv___autoParam___closed__46_value;
static const lean_string_object lp_mathlib_Equiv_left__inv___autoParam___closed__47_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "Elab"};
static const lean_object* lp_mathlib_Equiv_left__inv___autoParam___closed__47 = (const lean_object*)&lp_mathlib_Equiv_left__inv___autoParam___closed__47_value;
static const lean_string_object lp_mathlib_Equiv_left__inv___autoParam___closed__48_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = "Ext"};
static const lean_object* lp_mathlib_Equiv_left__inv___autoParam___closed__48 = (const lean_object*)&lp_mathlib_Equiv_left__inv___autoParam___closed__48_value;
static const lean_string_object lp_mathlib_Equiv_left__inv___autoParam___closed__49_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = "ext"};
static const lean_object* lp_mathlib_Equiv_left__inv___autoParam___closed__49 = (const lean_object*)&lp_mathlib_Equiv_left__inv___autoParam___closed__49_value;
static const lean_ctor_object lp_mathlib_Equiv_left__inv___autoParam___closed__50_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Equiv_left__inv___autoParam___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib_Equiv_left__inv___autoParam___closed__50_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Equiv_left__inv___autoParam___closed__50_value_aux_0),((lean_object*)&lp_mathlib_Equiv_left__inv___autoParam___closed__47_value),LEAN_SCALAR_PTR_LITERAL(52, 247, 248, 201, 92, 23, 188, 159)}};
static const lean_ctor_object lp_mathlib_Equiv_left__inv___autoParam___closed__50_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Equiv_left__inv___autoParam___closed__50_value_aux_1),((lean_object*)&lp_mathlib_Equiv_left__inv___autoParam___closed__2_value),LEAN_SCALAR_PTR_LITERAL(161, 230, 229, 85, 182, 144, 182, 176)}};
static const lean_ctor_object lp_mathlib_Equiv_left__inv___autoParam___closed__50_value_aux_3 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Equiv_left__inv___autoParam___closed__50_value_aux_2),((lean_object*)&lp_mathlib_Equiv_left__inv___autoParam___closed__48_value),LEAN_SCALAR_PTR_LITERAL(49, 70, 231, 255, 233, 213, 189, 46)}};
static const lean_ctor_object lp_mathlib_Equiv_left__inv___autoParam___closed__50_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Equiv_left__inv___autoParam___closed__50_value_aux_3),((lean_object*)&lp_mathlib_Equiv_left__inv___autoParam___closed__49_value),LEAN_SCALAR_PTR_LITERAL(243, 8, 190, 90, 148, 21, 56, 73)}};
static const lean_object* lp_mathlib_Equiv_left__inv___autoParam___closed__50 = (const lean_object*)&lp_mathlib_Equiv_left__inv___autoParam___closed__50_value;
static lean_once_cell_t lp_mathlib_Equiv_left__inv___autoParam___closed__51_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Equiv_left__inv___autoParam___closed__51;
static lean_once_cell_t lp_mathlib_Equiv_left__inv___autoParam___closed__52_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Equiv_left__inv___autoParam___closed__52;
static lean_once_cell_t lp_mathlib_Equiv_left__inv___autoParam___closed__53_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Equiv_left__inv___autoParam___closed__53;
static lean_once_cell_t lp_mathlib_Equiv_left__inv___autoParam___closed__54_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Equiv_left__inv___autoParam___closed__54;
static lean_once_cell_t lp_mathlib_Equiv_left__inv___autoParam___closed__55_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Equiv_left__inv___autoParam___closed__55;
static lean_once_cell_t lp_mathlib_Equiv_left__inv___autoParam___closed__56_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Equiv_left__inv___autoParam___closed__56;
static const lean_string_object lp_mathlib_Equiv_left__inv___autoParam___closed__57_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = "<;>"};
static const lean_object* lp_mathlib_Equiv_left__inv___autoParam___closed__57 = (const lean_object*)&lp_mathlib_Equiv_left__inv___autoParam___closed__57_value;
static lean_once_cell_t lp_mathlib_Equiv_left__inv___autoParam___closed__58_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Equiv_left__inv___autoParam___closed__58;
static lean_once_cell_t lp_mathlib_Equiv_left__inv___autoParam___closed__59_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Equiv_left__inv___autoParam___closed__59;
static lean_once_cell_t lp_mathlib_Equiv_left__inv___autoParam___closed__60_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Equiv_left__inv___autoParam___closed__60;
static lean_once_cell_t lp_mathlib_Equiv_left__inv___autoParam___closed__61_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Equiv_left__inv___autoParam___closed__61;
static lean_once_cell_t lp_mathlib_Equiv_left__inv___autoParam___closed__62_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Equiv_left__inv___autoParam___closed__62;
static lean_once_cell_t lp_mathlib_Equiv_left__inv___autoParam___closed__63_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Equiv_left__inv___autoParam___closed__63;
static lean_once_cell_t lp_mathlib_Equiv_left__inv___autoParam___closed__64_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Equiv_left__inv___autoParam___closed__64;
static lean_once_cell_t lp_mathlib_Equiv_left__inv___autoParam___closed__65_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Equiv_left__inv___autoParam___closed__65;
static lean_once_cell_t lp_mathlib_Equiv_left__inv___autoParam___closed__66_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Equiv_left__inv___autoParam___closed__66;
static lean_once_cell_t lp_mathlib_Equiv_left__inv___autoParam___closed__67_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Equiv_left__inv___autoParam___closed__67;
static lean_once_cell_t lp_mathlib_Equiv_left__inv___autoParam___closed__68_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Equiv_left__inv___autoParam___closed__68;
static lean_once_cell_t lp_mathlib_Equiv_left__inv___autoParam___closed__69_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Equiv_left__inv___autoParam___closed__69;
static lean_once_cell_t lp_mathlib_Equiv_left__inv___autoParam___closed__70_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Equiv_left__inv___autoParam___closed__70;
static lean_once_cell_t lp_mathlib_Equiv_left__inv___autoParam___closed__71_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Equiv_left__inv___autoParam___closed__71;
static lean_once_cell_t lp_mathlib_Equiv_left__inv___autoParam___closed__72_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Equiv_left__inv___autoParam___closed__72;
static lean_once_cell_t lp_mathlib_Equiv_left__inv___autoParam___closed__73_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Equiv_left__inv___autoParam___closed__73;
static lean_once_cell_t lp_mathlib_Equiv_left__inv___autoParam___closed__74_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Equiv_left__inv___autoParam___closed__74;
static lean_once_cell_t lp_mathlib_Equiv_left__inv___autoParam___closed__75_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Equiv_left__inv___autoParam___closed__75;
static lean_once_cell_t lp_mathlib_Equiv_left__inv___autoParam___closed__76_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Equiv_left__inv___autoParam___closed__76;
static lean_once_cell_t lp_mathlib_Equiv_left__inv___autoParam___closed__77_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Equiv_left__inv___autoParam___closed__77;
static lean_once_cell_t lp_mathlib_Equiv_left__inv___autoParam___closed__78_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Equiv_left__inv___autoParam___closed__78;
static lean_once_cell_t lp_mathlib_Equiv_left__inv___autoParam___closed__79_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Equiv_left__inv___autoParam___closed__79;
LEAN_EXPORT lean_object* lp_mathlib_Equiv_left__inv___autoParam;
LEAN_EXPORT lean_object* lp_mathlib_Equiv_right__inv___autoParam;
static const lean_string_object lp_mathlib_term___u2243___00__closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 7, .m_data = "term_≃_"};
static const lean_object* lp_mathlib_term___u2243___00__closed__0 = (const lean_object*)&lp_mathlib_term___u2243___00__closed__0_value;
static const lean_ctor_object lp_mathlib_term___u2243___00__closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_term___u2243___00__closed__0_value),LEAN_SCALAR_PTR_LITERAL(172, 92, 84, 66, 87, 25, 24, 137)}};
static const lean_object* lp_mathlib_term___u2243___00__closed__1 = (const lean_object*)&lp_mathlib_term___u2243___00__closed__1_value;
static const lean_string_object lp_mathlib_term___u2243___00__closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "andthen"};
static const lean_object* lp_mathlib_term___u2243___00__closed__2 = (const lean_object*)&lp_mathlib_term___u2243___00__closed__2_value;
static const lean_ctor_object lp_mathlib_term___u2243___00__closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_term___u2243___00__closed__2_value),LEAN_SCALAR_PTR_LITERAL(40, 255, 78, 30, 143, 119, 117, 174)}};
static const lean_object* lp_mathlib_term___u2243___00__closed__3 = (const lean_object*)&lp_mathlib_term___u2243___00__closed__3_value;
static const lean_string_object lp_mathlib_term___u2243___00__closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 3, .m_data = " ≃ "};
static const lean_object* lp_mathlib_term___u2243___00__closed__4 = (const lean_object*)&lp_mathlib_term___u2243___00__closed__4_value;
static const lean_ctor_object lp_mathlib_term___u2243___00__closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 5}, .m_objs = {((lean_object*)&lp_mathlib_term___u2243___00__closed__4_value)}};
static const lean_object* lp_mathlib_term___u2243___00__closed__5 = (const lean_object*)&lp_mathlib_term___u2243___00__closed__5_value;
static const lean_string_object lp_mathlib_term___u2243___00__closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "term"};
static const lean_object* lp_mathlib_term___u2243___00__closed__6 = (const lean_object*)&lp_mathlib_term___u2243___00__closed__6_value;
static const lean_ctor_object lp_mathlib_term___u2243___00__closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_term___u2243___00__closed__6_value),LEAN_SCALAR_PTR_LITERAL(187, 230, 181, 162, 253, 146, 122, 119)}};
static const lean_object* lp_mathlib_term___u2243___00__closed__7 = (const lean_object*)&lp_mathlib_term___u2243___00__closed__7_value;
static const lean_ctor_object lp_mathlib_term___u2243___00__closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 7}, .m_objs = {((lean_object*)&lp_mathlib_term___u2243___00__closed__7_value),((lean_object*)(((size_t)(26) << 1) | 1))}};
static const lean_object* lp_mathlib_term___u2243___00__closed__8 = (const lean_object*)&lp_mathlib_term___u2243___00__closed__8_value;
static const lean_ctor_object lp_mathlib_term___u2243___00__closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib_term___u2243___00__closed__3_value),((lean_object*)&lp_mathlib_term___u2243___00__closed__5_value),((lean_object*)&lp_mathlib_term___u2243___00__closed__8_value)}};
static const lean_object* lp_mathlib_term___u2243___00__closed__9 = (const lean_object*)&lp_mathlib_term___u2243___00__closed__9_value;
static const lean_ctor_object lp_mathlib_term___u2243___00__closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*4 + 0, .m_other = 4, .m_tag = 4}, .m_objs = {((lean_object*)&lp_mathlib_term___u2243___00__closed__1_value),((lean_object*)(((size_t)(25) << 1) | 1)),((lean_object*)(((size_t)(25) << 1) | 1)),((lean_object*)&lp_mathlib_term___u2243___00__closed__9_value)}};
static const lean_object* lp_mathlib_term___u2243___00__closed__10 = (const lean_object*)&lp_mathlib_term___u2243___00__closed__10_value;
LEAN_EXPORT const lean_object* lp_mathlib_term___u2243__ = (const lean_object*)&lp_mathlib_term___u2243___00__closed__10_value;
static const lean_string_object lp_mathlib___aux__Mathlib__Logic__Equiv__Defs______macroRules__term___u2243____1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "Term"};
static const lean_object* lp_mathlib___aux__Mathlib__Logic__Equiv__Defs______macroRules__term___u2243____1___closed__0 = (const lean_object*)&lp_mathlib___aux__Mathlib__Logic__Equiv__Defs______macroRules__term___u2243____1___closed__0_value;
static const lean_string_object lp_mathlib___aux__Mathlib__Logic__Equiv__Defs______macroRules__term___u2243____1___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = "app"};
static const lean_object* lp_mathlib___aux__Mathlib__Logic__Equiv__Defs______macroRules__term___u2243____1___closed__1 = (const lean_object*)&lp_mathlib___aux__Mathlib__Logic__Equiv__Defs______macroRules__term___u2243____1___closed__1_value;
static const lean_ctor_object lp_mathlib___aux__Mathlib__Logic__Equiv__Defs______macroRules__term___u2243____1___closed__2_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Equiv_left__inv___autoParam___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib___aux__Mathlib__Logic__Equiv__Defs______macroRules__term___u2243____1___closed__2_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___aux__Mathlib__Logic__Equiv__Defs______macroRules__term___u2243____1___closed__2_value_aux_0),((lean_object*)&lp_mathlib_Equiv_left__inv___autoParam___closed__1_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib___aux__Mathlib__Logic__Equiv__Defs______macroRules__term___u2243____1___closed__2_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___aux__Mathlib__Logic__Equiv__Defs______macroRules__term___u2243____1___closed__2_value_aux_1),((lean_object*)&lp_mathlib___aux__Mathlib__Logic__Equiv__Defs______macroRules__term___u2243____1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(75, 170, 162, 138, 136, 204, 251, 229)}};
static const lean_ctor_object lp_mathlib___aux__Mathlib__Logic__Equiv__Defs______macroRules__term___u2243____1___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___aux__Mathlib__Logic__Equiv__Defs______macroRules__term___u2243____1___closed__2_value_aux_2),((lean_object*)&lp_mathlib___aux__Mathlib__Logic__Equiv__Defs______macroRules__term___u2243____1___closed__1_value),LEAN_SCALAR_PTR_LITERAL(69, 118, 10, 41, 220, 156, 243, 179)}};
static const lean_object* lp_mathlib___aux__Mathlib__Logic__Equiv__Defs______macroRules__term___u2243____1___closed__2 = (const lean_object*)&lp_mathlib___aux__Mathlib__Logic__Equiv__Defs______macroRules__term___u2243____1___closed__2_value;
static const lean_string_object lp_mathlib___aux__Mathlib__Logic__Equiv__Defs______macroRules__term___u2243____1___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "Equiv"};
static const lean_object* lp_mathlib___aux__Mathlib__Logic__Equiv__Defs______macroRules__term___u2243____1___closed__3 = (const lean_object*)&lp_mathlib___aux__Mathlib__Logic__Equiv__Defs______macroRules__term___u2243____1___closed__3_value;
static lean_once_cell_t lp_mathlib___aux__Mathlib__Logic__Equiv__Defs______macroRules__term___u2243____1___closed__4_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___aux__Mathlib__Logic__Equiv__Defs______macroRules__term___u2243____1___closed__4;
static const lean_ctor_object lp_mathlib___aux__Mathlib__Logic__Equiv__Defs______macroRules__term___u2243____1___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___aux__Mathlib__Logic__Equiv__Defs______macroRules__term___u2243____1___closed__3_value),LEAN_SCALAR_PTR_LITERAL(0, 253, 123, 237, 128, 91, 245, 83)}};
static const lean_object* lp_mathlib___aux__Mathlib__Logic__Equiv__Defs______macroRules__term___u2243____1___closed__5 = (const lean_object*)&lp_mathlib___aux__Mathlib__Logic__Equiv__Defs______macroRules__term___u2243____1___closed__5_value;
static const lean_ctor_object lp_mathlib___aux__Mathlib__Logic__Equiv__Defs______macroRules__term___u2243____1___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___aux__Mathlib__Logic__Equiv__Defs______macroRules__term___u2243____1___closed__5_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib___aux__Mathlib__Logic__Equiv__Defs______macroRules__term___u2243____1___closed__6 = (const lean_object*)&lp_mathlib___aux__Mathlib__Logic__Equiv__Defs______macroRules__term___u2243____1___closed__6_value;
static const lean_ctor_object lp_mathlib___aux__Mathlib__Logic__Equiv__Defs______macroRules__term___u2243____1___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib___aux__Mathlib__Logic__Equiv__Defs______macroRules__term___u2243____1___closed__5_value)}};
static const lean_object* lp_mathlib___aux__Mathlib__Logic__Equiv__Defs______macroRules__term___u2243____1___closed__7 = (const lean_object*)&lp_mathlib___aux__Mathlib__Logic__Equiv__Defs______macroRules__term___u2243____1___closed__7_value;
static const lean_ctor_object lp_mathlib___aux__Mathlib__Logic__Equiv__Defs______macroRules__term___u2243____1___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___aux__Mathlib__Logic__Equiv__Defs______macroRules__term___u2243____1___closed__7_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib___aux__Mathlib__Logic__Equiv__Defs______macroRules__term___u2243____1___closed__8 = (const lean_object*)&lp_mathlib___aux__Mathlib__Logic__Equiv__Defs______macroRules__term___u2243____1___closed__8_value;
static const lean_ctor_object lp_mathlib___aux__Mathlib__Logic__Equiv__Defs______macroRules__term___u2243____1___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___aux__Mathlib__Logic__Equiv__Defs______macroRules__term___u2243____1___closed__6_value),((lean_object*)&lp_mathlib___aux__Mathlib__Logic__Equiv__Defs______macroRules__term___u2243____1___closed__8_value)}};
static const lean_object* lp_mathlib___aux__Mathlib__Logic__Equiv__Defs______macroRules__term___u2243____1___closed__9 = (const lean_object*)&lp_mathlib___aux__Mathlib__Logic__Equiv__Defs______macroRules__term___u2243____1___closed__9_value;
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Logic__Equiv__Defs______macroRules__term___u2243____1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Logic__Equiv__Defs______macroRules__term___u2243____1___boxed(lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib___aux__Mathlib__Logic__Equiv__Defs______unexpand__Equiv__1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "ident"};
static const lean_object* lp_mathlib___aux__Mathlib__Logic__Equiv__Defs______unexpand__Equiv__1___closed__0 = (const lean_object*)&lp_mathlib___aux__Mathlib__Logic__Equiv__Defs______unexpand__Equiv__1___closed__0_value;
static const lean_ctor_object lp_mathlib___aux__Mathlib__Logic__Equiv__Defs______unexpand__Equiv__1___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___aux__Mathlib__Logic__Equiv__Defs______unexpand__Equiv__1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(52, 159, 208, 51, 14, 60, 6, 71)}};
static const lean_object* lp_mathlib___aux__Mathlib__Logic__Equiv__Defs______unexpand__Equiv__1___closed__1 = (const lean_object*)&lp_mathlib___aux__Mathlib__Logic__Equiv__Defs______unexpand__Equiv__1___closed__1_value;
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Logic__Equiv__Defs______unexpand__Equiv__1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Logic__Equiv__Defs______unexpand__Equiv__1___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_EquivLike_toEquiv___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_EquivLike_toEquiv(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instCoeTCEquivOfEquivLike___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instCoeTCEquivOfEquivLike(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_refl___lam__0(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_refl___lam__0___boxed(lean_object*);
static const lean_closure_object lp_mathlib_Equiv_refl___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Equiv_refl___lam__0___boxed, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Equiv_refl___closed__0 = (const lean_object*)&lp_mathlib_Equiv_refl___closed__0_value;
static const lean_ctor_object lp_mathlib_Equiv_refl___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_Equiv_refl___closed__0_value),((lean_object*)&lp_mathlib_Equiv_refl___closed__0_value)}};
static const lean_object* lp_mathlib_Equiv_refl___closed__1 = (const lean_object*)&lp_mathlib_Equiv_refl___closed__1_value;
LEAN_EXPORT lean_object* lp_mathlib_Equiv_refl(lean_object*);
static lean_once_cell_t lp_mathlib_Equiv_inhabited_x27___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Equiv_inhabited_x27___closed__0;
LEAN_EXPORT lean_object* lp_mathlib_Equiv_inhabited_x27(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_symm___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_symm(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_Simps_symm__apply___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_Simps_symm__apply(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_trans___redArg___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_trans___redArg___lam__1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_trans___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_trans(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_Equiv_instTrans___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Equiv_trans, .m_arity = 5, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Equiv_instTrans___closed__0 = (const lean_object*)&lp_mathlib_Equiv_instTrans___closed__0_value;
LEAN_EXPORT const lean_object* lp_mathlib_Equiv_instTrans = (const lean_object*)&lp_mathlib_Equiv_instTrans___closed__0_value;
static const lean_closure_object lp_mathlib_Equiv_symmEquiv___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*2, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Equiv_symm, .m_arity = 3, .m_num_fixed = 2, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)(((size_t)(0) << 1) | 1))} };
static const lean_object* lp_mathlib_Equiv_symmEquiv___closed__0 = (const lean_object*)&lp_mathlib_Equiv_symmEquiv___closed__0_value;
static const lean_ctor_object lp_mathlib_Equiv_symmEquiv___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_Equiv_symmEquiv___closed__0_value),((lean_object*)&lp_mathlib_Equiv_symmEquiv___closed__0_value)}};
static const lean_object* lp_mathlib_Equiv_symmEquiv___closed__1 = (const lean_object*)&lp_mathlib_Equiv_symmEquiv___closed__1_value;
LEAN_EXPORT lean_object* lp_mathlib_Equiv_symmEquiv(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_permUnique(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_decidableEq___redArg___lam__0(lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Equiv_decidableEq___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_decidableEq___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Equiv_decidableEq(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_decidableEq___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_inhabited___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_inhabited(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_unique___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_unique(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_cast___lam__0(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_cast___lam__0___boxed(lean_object*);
static const lean_closure_object lp_mathlib_Equiv_cast___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Equiv_cast___lam__0___boxed, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Equiv_cast___closed__0 = (const lean_object*)&lp_mathlib_Equiv_cast___closed__0_value;
static const lean_ctor_object lp_mathlib_Equiv_cast___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_Equiv_cast___closed__0_value),((lean_object*)&lp_mathlib_Equiv_cast___closed__0_value)}};
static const lean_object* lp_mathlib_Equiv_cast___closed__1 = (const lean_object*)&lp_mathlib_Equiv_cast___closed__1_value;
LEAN_EXPORT lean_object* lp_mathlib_Equiv_cast(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_equivCongr___redArg___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_equivCongr___redArg___lam__1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_equivCongr___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_equivCongr(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_permCongr___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_permCongr(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_equivOfIsEmpty___lam__0(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_equivOfIsEmpty___lam__0___boxed(lean_object*);
static const lean_closure_object lp_mathlib_Equiv_equivOfIsEmpty___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Equiv_equivOfIsEmpty___lam__0___boxed, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Equiv_equivOfIsEmpty___closed__0 = (const lean_object*)&lp_mathlib_Equiv_equivOfIsEmpty___closed__0_value;
static const lean_ctor_object lp_mathlib_Equiv_equivOfIsEmpty___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_Equiv_equivOfIsEmpty___closed__0_value),((lean_object*)&lp_mathlib_Equiv_equivOfIsEmpty___closed__0_value)}};
static const lean_object* lp_mathlib_Equiv_equivOfIsEmpty___closed__1 = (const lean_object*)&lp_mathlib_Equiv_equivOfIsEmpty___closed__1_value;
LEAN_EXPORT lean_object* lp_mathlib_Equiv_equivOfIsEmpty(lean_object*, lean_object*, lean_object*, lean_object*);
static lean_once_cell_t lp_mathlib_Equiv_equivEmpty___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Equiv_equivEmpty___closed__0;
LEAN_EXPORT lean_object* lp_mathlib_Equiv_equivEmpty(lean_object*, lean_object*);
static lean_once_cell_t lp_mathlib_Equiv_equivPEmpty___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Equiv_equivPEmpty___closed__0;
LEAN_EXPORT lean_object* lp_mathlib_Equiv_equivPEmpty(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_equivEmptyEquiv___lam__0(lean_object*);
static const lean_closure_object lp_mathlib_Equiv_equivEmptyEquiv___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Equiv_equivEmptyEquiv___lam__0, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Equiv_equivEmptyEquiv___closed__0 = (const lean_object*)&lp_mathlib_Equiv_equivEmptyEquiv___closed__0_value;
static const lean_ctor_object lp_mathlib_Equiv_equivEmptyEquiv___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Equiv_equivEmptyEquiv___closed__0_value)}};
static const lean_object* lp_mathlib_Equiv_equivEmptyEquiv___closed__1 = (const lean_object*)&lp_mathlib_Equiv_equivEmptyEquiv___closed__1_value;
LEAN_EXPORT lean_object* lp_mathlib_Equiv_equivEmptyEquiv(lean_object*);
static lean_once_cell_t lp_mathlib_Equiv_propEquivPEmpty___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Equiv_propEquivPEmpty___closed__0;
LEAN_EXPORT lean_object* lp_mathlib_Equiv_propEquivPEmpty(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_ofUnique___redArg___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_ofUnique___redArg___lam__0___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_ofUnique___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_ofUnique(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_equivPUnit___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_equivPUnit(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_ofUnique___at___00Equiv_equivPUnit___at___00Equiv_propEquivPUnit_spec__0_spec__0___lam__1(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_ofUnique___at___00Equiv_equivPUnit___at___00Equiv_propEquivPUnit_spec__0_spec__0___lam__0(lean_object*);
static const lean_closure_object lp_mathlib_Equiv_ofUnique___at___00Equiv_equivPUnit___at___00Equiv_propEquivPUnit_spec__0_spec__0___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Equiv_ofUnique___at___00Equiv_equivPUnit___at___00Equiv_propEquivPUnit_spec__0_spec__0___lam__0, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Equiv_ofUnique___at___00Equiv_equivPUnit___at___00Equiv_propEquivPUnit_spec__0_spec__0___closed__0 = (const lean_object*)&lp_mathlib_Equiv_ofUnique___at___00Equiv_equivPUnit___at___00Equiv_propEquivPUnit_spec__0_spec__0___closed__0_value;
static const lean_ctor_object lp_mathlib_Equiv_ofUnique___at___00Equiv_equivPUnit___at___00Equiv_propEquivPUnit_spec__0_spec__0___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_Equiv_ofUnique___at___00Equiv_equivPUnit___at___00Equiv_propEquivPUnit_spec__0_spec__0___closed__0_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_Equiv_ofUnique___at___00Equiv_equivPUnit___at___00Equiv_propEquivPUnit_spec__0_spec__0___closed__1 = (const lean_object*)&lp_mathlib_Equiv_ofUnique___at___00Equiv_equivPUnit___at___00Equiv_propEquivPUnit_spec__0_spec__0___closed__1_value;
LEAN_EXPORT const lean_object* lp_mathlib_Equiv_ofUnique___at___00Equiv_equivPUnit___at___00Equiv_propEquivPUnit_spec__0_spec__0 = (const lean_object*)&lp_mathlib_Equiv_ofUnique___at___00Equiv_equivPUnit___at___00Equiv_propEquivPUnit_spec__0_spec__0___closed__1_value;
LEAN_EXPORT lean_object* lp_mathlib_Equiv_propEquivPUnit(lean_object*, lean_object*);
LEAN_EXPORT const lean_object* lp_mathlib_Equiv_equivPUnit___at___00Equiv_propEquivPUnit_spec__0 = (const lean_object*)&lp_mathlib_Equiv_ofUnique___at___00Equiv_equivPUnit___at___00Equiv_propEquivPUnit_spec__0_spec__0___closed__1_value;
LEAN_EXPORT lean_object* lp_mathlib_Equiv_ulift___lam__0(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_ulift___lam__0___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_ulift___lam__1(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_ulift___lam__1___boxed(lean_object*);
static const lean_closure_object lp_mathlib_Equiv_ulift___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Equiv_ulift___lam__0___boxed, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Equiv_ulift___closed__0 = (const lean_object*)&lp_mathlib_Equiv_ulift___closed__0_value;
static const lean_closure_object lp_mathlib_Equiv_ulift___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Equiv_ulift___lam__1___boxed, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Equiv_ulift___closed__1 = (const lean_object*)&lp_mathlib_Equiv_ulift___closed__1_value;
static const lean_ctor_object lp_mathlib_Equiv_ulift___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_Equiv_ulift___closed__0_value),((lean_object*)&lp_mathlib_Equiv_ulift___closed__1_value)}};
static const lean_object* lp_mathlib_Equiv_ulift___closed__2 = (const lean_object*)&lp_mathlib_Equiv_ulift___closed__2_value;
LEAN_EXPORT lean_object* lp_mathlib_Equiv_ulift(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_plift(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_ofIff(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_arrowCongr___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_arrowCongr___redArg___lam__1(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_arrowCongr___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_arrowCongr(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_arrowCongr_x27___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_arrowCongr_x27(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_conj___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_conj(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_punitEquivPUnit___lam__0(lean_object*);
static const lean_closure_object lp_mathlib_Equiv_punitEquivPUnit___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Equiv_punitEquivPUnit___lam__0, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Equiv_punitEquivPUnit___closed__0 = (const lean_object*)&lp_mathlib_Equiv_punitEquivPUnit___closed__0_value;
static const lean_ctor_object lp_mathlib_Equiv_punitEquivPUnit___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_Equiv_punitEquivPUnit___closed__0_value),((lean_object*)&lp_mathlib_Equiv_punitEquivPUnit___closed__0_value)}};
static const lean_object* lp_mathlib_Equiv_punitEquivPUnit___closed__1 = (const lean_object*)&lp_mathlib_Equiv_punitEquivPUnit___closed__1_value;
LEAN_EXPORT const lean_object* lp_mathlib_Equiv_punitEquivPUnit = (const lean_object*)&lp_mathlib_Equiv_punitEquivPUnit___closed__1_value;
LEAN_EXPORT lean_object* lp_mathlib_Equiv_arrowPUnitEquivPUnit___lam__0(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_arrowPUnitEquivPUnit___lam__0___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_arrowPUnitEquivPUnit___lam__1(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_arrowPUnitEquivPUnit___lam__1___boxed(lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_Equiv_arrowPUnitEquivPUnit___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Equiv_arrowPUnitEquivPUnit___lam__0___boxed, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Equiv_arrowPUnitEquivPUnit___closed__0 = (const lean_object*)&lp_mathlib_Equiv_arrowPUnitEquivPUnit___closed__0_value;
static const lean_closure_object lp_mathlib_Equiv_arrowPUnitEquivPUnit___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Equiv_arrowPUnitEquivPUnit___lam__1___boxed, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Equiv_arrowPUnitEquivPUnit___closed__1 = (const lean_object*)&lp_mathlib_Equiv_arrowPUnitEquivPUnit___closed__1_value;
static const lean_ctor_object lp_mathlib_Equiv_arrowPUnitEquivPUnit___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_Equiv_arrowPUnitEquivPUnit___closed__0_value),((lean_object*)&lp_mathlib_Equiv_arrowPUnitEquivPUnit___closed__1_value)}};
static const lean_object* lp_mathlib_Equiv_arrowPUnitEquivPUnit___closed__2 = (const lean_object*)&lp_mathlib_Equiv_arrowPUnitEquivPUnit___closed__2_value;
LEAN_EXPORT lean_object* lp_mathlib_Equiv_arrowPUnitEquivPUnit(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_piUnique___redArg___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_piUnique___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_piUnique(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_funUnique___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_funUnique(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_piUnique___at___00Equiv_punitArrowEquiv_spec__0___lam__0(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_piUnique___at___00Equiv_punitArrowEquiv_spec__0___lam__1(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_piUnique___at___00Equiv_punitArrowEquiv_spec__0___lam__1___boxed(lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_Equiv_piUnique___at___00Equiv_punitArrowEquiv_spec__0___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Equiv_piUnique___at___00Equiv_punitArrowEquiv_spec__0___lam__0, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Equiv_piUnique___at___00Equiv_punitArrowEquiv_spec__0___closed__0 = (const lean_object*)&lp_mathlib_Equiv_piUnique___at___00Equiv_punitArrowEquiv_spec__0___closed__0_value;
static const lean_closure_object lp_mathlib_Equiv_piUnique___at___00Equiv_punitArrowEquiv_spec__0___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Equiv_piUnique___at___00Equiv_punitArrowEquiv_spec__0___lam__1___boxed, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Equiv_piUnique___at___00Equiv_punitArrowEquiv_spec__0___closed__1 = (const lean_object*)&lp_mathlib_Equiv_piUnique___at___00Equiv_punitArrowEquiv_spec__0___closed__1_value;
static const lean_ctor_object lp_mathlib_Equiv_piUnique___at___00Equiv_punitArrowEquiv_spec__0___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_Equiv_piUnique___at___00Equiv_punitArrowEquiv_spec__0___closed__0_value),((lean_object*)&lp_mathlib_Equiv_piUnique___at___00Equiv_punitArrowEquiv_spec__0___closed__1_value)}};
static const lean_object* lp_mathlib_Equiv_piUnique___at___00Equiv_punitArrowEquiv_spec__0___closed__2 = (const lean_object*)&lp_mathlib_Equiv_piUnique___at___00Equiv_punitArrowEquiv_spec__0___closed__2_value;
LEAN_EXPORT lean_object* lp_mathlib_Equiv_piUnique___at___00Equiv_punitArrowEquiv_spec__0(lean_object*);
static lean_once_cell_t lp_mathlib_Equiv_punitArrowEquiv___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Equiv_punitArrowEquiv___closed__0;
LEAN_EXPORT lean_object* lp_mathlib_Equiv_punitArrowEquiv(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_piUnique___at___00Equiv_trueArrowEquiv_spec__0___lam__0(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_piUnique___at___00Equiv_trueArrowEquiv_spec__0___lam__1(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_piUnique___at___00Equiv_trueArrowEquiv_spec__0___lam__1___boxed(lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_Equiv_piUnique___at___00Equiv_trueArrowEquiv_spec__0___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Equiv_piUnique___at___00Equiv_trueArrowEquiv_spec__0___lam__0, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Equiv_piUnique___at___00Equiv_trueArrowEquiv_spec__0___closed__0 = (const lean_object*)&lp_mathlib_Equiv_piUnique___at___00Equiv_trueArrowEquiv_spec__0___closed__0_value;
static const lean_closure_object lp_mathlib_Equiv_piUnique___at___00Equiv_trueArrowEquiv_spec__0___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Equiv_piUnique___at___00Equiv_trueArrowEquiv_spec__0___lam__1___boxed, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Equiv_piUnique___at___00Equiv_trueArrowEquiv_spec__0___closed__1 = (const lean_object*)&lp_mathlib_Equiv_piUnique___at___00Equiv_trueArrowEquiv_spec__0___closed__1_value;
static const lean_ctor_object lp_mathlib_Equiv_piUnique___at___00Equiv_trueArrowEquiv_spec__0___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_Equiv_piUnique___at___00Equiv_trueArrowEquiv_spec__0___closed__0_value),((lean_object*)&lp_mathlib_Equiv_piUnique___at___00Equiv_trueArrowEquiv_spec__0___closed__1_value)}};
static const lean_object* lp_mathlib_Equiv_piUnique___at___00Equiv_trueArrowEquiv_spec__0___closed__2 = (const lean_object*)&lp_mathlib_Equiv_piUnique___at___00Equiv_trueArrowEquiv_spec__0___closed__2_value;
LEAN_EXPORT lean_object* lp_mathlib_Equiv_piUnique___at___00Equiv_trueArrowEquiv_spec__0(lean_object*);
static lean_once_cell_t lp_mathlib_Equiv_trueArrowEquiv___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Equiv_trueArrowEquiv___closed__0;
LEAN_EXPORT lean_object* lp_mathlib_Equiv_trueArrowEquiv(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_arrowPUnitOfIsEmpty___lam__0(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_arrowPUnitOfIsEmpty___lam__0___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_arrowPUnitOfIsEmpty___lam__1(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_arrowPUnitOfIsEmpty___lam__1___boxed(lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_Equiv_arrowPUnitOfIsEmpty___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Equiv_arrowPUnitOfIsEmpty___lam__0___boxed, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Equiv_arrowPUnitOfIsEmpty___closed__0 = (const lean_object*)&lp_mathlib_Equiv_arrowPUnitOfIsEmpty___closed__0_value;
static const lean_closure_object lp_mathlib_Equiv_arrowPUnitOfIsEmpty___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Equiv_arrowPUnitOfIsEmpty___lam__1___boxed, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Equiv_arrowPUnitOfIsEmpty___closed__1 = (const lean_object*)&lp_mathlib_Equiv_arrowPUnitOfIsEmpty___closed__1_value;
static const lean_ctor_object lp_mathlib_Equiv_arrowPUnitOfIsEmpty___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_Equiv_arrowPUnitOfIsEmpty___closed__0_value),((lean_object*)&lp_mathlib_Equiv_arrowPUnitOfIsEmpty___closed__1_value)}};
static const lean_object* lp_mathlib_Equiv_arrowPUnitOfIsEmpty___closed__2 = (const lean_object*)&lp_mathlib_Equiv_arrowPUnitOfIsEmpty___closed__2_value;
LEAN_EXPORT lean_object* lp_mathlib_Equiv_arrowPUnitOfIsEmpty(lean_object*, lean_object*, lean_object*);
static lean_once_cell_t lp_mathlib_Equiv_emptyArrowEquivPUnit___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Equiv_emptyArrowEquivPUnit___closed__0;
LEAN_EXPORT lean_object* lp_mathlib_Equiv_emptyArrowEquivPUnit(lean_object*);
static lean_once_cell_t lp_mathlib_Equiv_pemptyArrowEquivPUnit___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Equiv_pemptyArrowEquivPUnit___closed__0;
LEAN_EXPORT lean_object* lp_mathlib_Equiv_pemptyArrowEquivPUnit(lean_object*);
static lean_once_cell_t lp_mathlib_Equiv_falseArrowEquivPUnit___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Equiv_falseArrowEquivPUnit___closed__0;
LEAN_EXPORT lean_object* lp_mathlib_Equiv_falseArrowEquivPUnit(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_psigmaEquivSigma___lam__0(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_psigmaEquivSigma___lam__1(lean_object*);
static const lean_closure_object lp_mathlib_Equiv_psigmaEquivSigma___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Equiv_psigmaEquivSigma___lam__0, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Equiv_psigmaEquivSigma___closed__0 = (const lean_object*)&lp_mathlib_Equiv_psigmaEquivSigma___closed__0_value;
static const lean_closure_object lp_mathlib_Equiv_psigmaEquivSigma___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Equiv_psigmaEquivSigma___lam__1, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Equiv_psigmaEquivSigma___closed__1 = (const lean_object*)&lp_mathlib_Equiv_psigmaEquivSigma___closed__1_value;
static const lean_ctor_object lp_mathlib_Equiv_psigmaEquivSigma___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_Equiv_psigmaEquivSigma___closed__0_value),((lean_object*)&lp_mathlib_Equiv_psigmaEquivSigma___closed__1_value)}};
static const lean_object* lp_mathlib_Equiv_psigmaEquivSigma___closed__2 = (const lean_object*)&lp_mathlib_Equiv_psigmaEquivSigma___closed__2_value;
LEAN_EXPORT lean_object* lp_mathlib_Equiv_psigmaEquivSigma(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_psigmaEquivSigmaPLift(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_psigmaCongrRight___redArg___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_psigmaCongrRight___redArg___lam__1(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_psigmaCongrRight___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_psigmaCongrRight(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_sigmaCongrRight___redArg___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_sigmaCongrRight___redArg___lam__1(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_sigmaCongrRight___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_sigmaCongrRight(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_psigmaEquivSubtype___lam__0(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_psigmaEquivSubtype___lam__0___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_psigmaEquivSubtype___lam__1(lean_object*);
static const lean_closure_object lp_mathlib_Equiv_psigmaEquivSubtype___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Equiv_psigmaEquivSubtype___lam__0___boxed, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Equiv_psigmaEquivSubtype___closed__0 = (const lean_object*)&lp_mathlib_Equiv_psigmaEquivSubtype___closed__0_value;
static const lean_closure_object lp_mathlib_Equiv_psigmaEquivSubtype___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Equiv_psigmaEquivSubtype___lam__1, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Equiv_psigmaEquivSubtype___closed__1 = (const lean_object*)&lp_mathlib_Equiv_psigmaEquivSubtype___closed__1_value;
static const lean_ctor_object lp_mathlib_Equiv_psigmaEquivSubtype___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_Equiv_psigmaEquivSubtype___closed__0_value),((lean_object*)&lp_mathlib_Equiv_psigmaEquivSubtype___closed__1_value)}};
static const lean_object* lp_mathlib_Equiv_psigmaEquivSubtype___closed__2 = (const lean_object*)&lp_mathlib_Equiv_psigmaEquivSubtype___closed__2_value;
LEAN_EXPORT lean_object* lp_mathlib_Equiv_psigmaEquivSubtype(lean_object*, lean_object*);
static lean_once_cell_t lp_mathlib_Equiv_sigmaPLiftEquivSubtype___lam__0___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Equiv_sigmaPLiftEquivSubtype___lam__0___closed__0;
LEAN_EXPORT lean_object* lp_mathlib_Equiv_sigmaPLiftEquivSubtype___lam__0(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_sigmaPLiftEquivSubtype___lam__0___boxed(lean_object*);
static const lean_closure_object lp_mathlib_Equiv_sigmaPLiftEquivSubtype___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Equiv_sigmaPLiftEquivSubtype___lam__0___boxed, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Equiv_sigmaPLiftEquivSubtype___closed__0 = (const lean_object*)&lp_mathlib_Equiv_sigmaPLiftEquivSubtype___closed__0_value;
static lean_once_cell_t lp_mathlib_Equiv_sigmaPLiftEquivSubtype___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Equiv_sigmaPLiftEquivSubtype___closed__1;
static lean_once_cell_t lp_mathlib_Equiv_sigmaPLiftEquivSubtype___closed__2_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Equiv_sigmaPLiftEquivSubtype___closed__2;
static lean_once_cell_t lp_mathlib_Equiv_sigmaPLiftEquivSubtype___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Equiv_sigmaPLiftEquivSubtype___closed__3;
static lean_once_cell_t lp_mathlib_Equiv_sigmaPLiftEquivSubtype___closed__4_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Equiv_sigmaPLiftEquivSubtype___closed__4;
static lean_once_cell_t lp_mathlib_Equiv_sigmaPLiftEquivSubtype___closed__5_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Equiv_sigmaPLiftEquivSubtype___closed__5;
static lean_once_cell_t lp_mathlib_Equiv_sigmaPLiftEquivSubtype___closed__6_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Equiv_sigmaPLiftEquivSubtype___closed__6;
LEAN_EXPORT lean_object* lp_mathlib_Equiv_sigmaPLiftEquivSubtype(lean_object*, lean_object*);
static lean_once_cell_t lp_mathlib_Equiv_sigmaULiftPLiftEquivSubtype___lam__0___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Equiv_sigmaULiftPLiftEquivSubtype___lam__0___closed__0;
LEAN_EXPORT lean_object* lp_mathlib_Equiv_sigmaULiftPLiftEquivSubtype___lam__0(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_sigmaULiftPLiftEquivSubtype___lam__0___boxed(lean_object*);
static const lean_closure_object lp_mathlib_Equiv_sigmaULiftPLiftEquivSubtype___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Equiv_sigmaULiftPLiftEquivSubtype___lam__0___boxed, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Equiv_sigmaULiftPLiftEquivSubtype___closed__0 = (const lean_object*)&lp_mathlib_Equiv_sigmaULiftPLiftEquivSubtype___closed__0_value;
static lean_once_cell_t lp_mathlib_Equiv_sigmaULiftPLiftEquivSubtype___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Equiv_sigmaULiftPLiftEquivSubtype___closed__1;
static lean_once_cell_t lp_mathlib_Equiv_sigmaULiftPLiftEquivSubtype___closed__2_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Equiv_sigmaULiftPLiftEquivSubtype___closed__2;
static lean_once_cell_t lp_mathlib_Equiv_sigmaULiftPLiftEquivSubtype___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Equiv_sigmaULiftPLiftEquivSubtype___closed__3;
LEAN_EXPORT lean_object* lp_mathlib_Equiv_sigmaULiftPLiftEquivSubtype(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_Perm_sigmaCongrRight___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_Perm_sigmaCongrRight(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_functionSwap___lam__0(lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_Equiv_functionSwap___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Equiv_functionSwap___lam__0, .m_arity = 3, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Equiv_functionSwap___closed__0 = (const lean_object*)&lp_mathlib_Equiv_functionSwap___closed__0_value;
static const lean_ctor_object lp_mathlib_Equiv_functionSwap___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_Equiv_functionSwap___closed__0_value),((lean_object*)&lp_mathlib_Equiv_functionSwap___closed__0_value)}};
static const lean_object* lp_mathlib_Equiv_functionSwap___closed__1 = (const lean_object*)&lp_mathlib_Equiv_functionSwap___closed__1_value;
LEAN_EXPORT lean_object* lp_mathlib_Equiv_functionSwap(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_sigmaCongrLeft___redArg___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_sigmaCongrLeft___redArg___lam__1(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_sigmaCongrLeft___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_sigmaCongrLeft(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_sigmaCongrLeft_x27___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_sigmaCongrLeft_x27(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_sigmaCongr___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_sigmaCongr(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_sigmaEquivProd___lam__0(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_sigmaEquivProd___lam__1(lean_object*);
static const lean_closure_object lp_mathlib_Equiv_sigmaEquivProd___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Equiv_sigmaEquivProd___lam__0, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Equiv_sigmaEquivProd___closed__0 = (const lean_object*)&lp_mathlib_Equiv_sigmaEquivProd___closed__0_value;
static const lean_closure_object lp_mathlib_Equiv_sigmaEquivProd___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Equiv_sigmaEquivProd___lam__1, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Equiv_sigmaEquivProd___closed__1 = (const lean_object*)&lp_mathlib_Equiv_sigmaEquivProd___closed__1_value;
static const lean_ctor_object lp_mathlib_Equiv_sigmaEquivProd___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_Equiv_sigmaEquivProd___closed__0_value),((lean_object*)&lp_mathlib_Equiv_sigmaEquivProd___closed__1_value)}};
static const lean_object* lp_mathlib_Equiv_sigmaEquivProd___closed__2 = (const lean_object*)&lp_mathlib_Equiv_sigmaEquivProd___closed__2_value;
LEAN_EXPORT lean_object* lp_mathlib_Equiv_sigmaEquivProd(lean_object*, lean_object*);
static lean_once_cell_t lp_mathlib_Equiv_sigmaEquivProdOfEquiv___redArg___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Equiv_sigmaEquivProdOfEquiv___redArg___closed__0;
LEAN_EXPORT lean_object* lp_mathlib_Equiv_sigmaEquivProdOfEquiv___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_sigmaEquivProdOfEquiv(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_sigmaAssoc___lam__0(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_sigmaAssoc___lam__1(lean_object*);
static const lean_closure_object lp_mathlib_Equiv_sigmaAssoc___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Equiv_sigmaAssoc___lam__0, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Equiv_sigmaAssoc___closed__0 = (const lean_object*)&lp_mathlib_Equiv_sigmaAssoc___closed__0_value;
static const lean_closure_object lp_mathlib_Equiv_sigmaAssoc___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Equiv_sigmaAssoc___lam__1, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Equiv_sigmaAssoc___closed__1 = (const lean_object*)&lp_mathlib_Equiv_sigmaAssoc___closed__1_value;
static const lean_ctor_object lp_mathlib_Equiv_sigmaAssoc___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_Equiv_sigmaAssoc___closed__0_value),((lean_object*)&lp_mathlib_Equiv_sigmaAssoc___closed__1_value)}};
static const lean_object* lp_mathlib_Equiv_sigmaAssoc___closed__2 = (const lean_object*)&lp_mathlib_Equiv_sigmaAssoc___closed__2_value;
LEAN_EXPORT lean_object* lp_mathlib_Equiv_sigmaAssoc(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_pSigmaAssoc___lam__0(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_pSigmaAssoc___lam__1(lean_object*);
static const lean_closure_object lp_mathlib_Equiv_pSigmaAssoc___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Equiv_pSigmaAssoc___lam__0, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Equiv_pSigmaAssoc___closed__0 = (const lean_object*)&lp_mathlib_Equiv_pSigmaAssoc___closed__0_value;
static const lean_closure_object lp_mathlib_Equiv_pSigmaAssoc___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Equiv_pSigmaAssoc___lam__1, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Equiv_pSigmaAssoc___closed__1 = (const lean_object*)&lp_mathlib_Equiv_pSigmaAssoc___closed__1_value;
static const lean_ctor_object lp_mathlib_Equiv_pSigmaAssoc___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_Equiv_pSigmaAssoc___closed__0_value),((lean_object*)&lp_mathlib_Equiv_pSigmaAssoc___closed__1_value)}};
static const lean_object* lp_mathlib_Equiv_pSigmaAssoc___closed__2 = (const lean_object*)&lp_mathlib_Equiv_pSigmaAssoc___closed__2_value;
LEAN_EXPORT lean_object* lp_mathlib_Equiv_pSigmaAssoc(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Quot_congr___redArg___lam__1(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Quot_congr___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Quot_congr(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_once_cell_t lp_mathlib_Quot_congrRight___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Quot_congrRight___closed__0;
LEAN_EXPORT lean_object* lp_mathlib_Quot_congrRight(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Quot_congrLeft___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Quot_congrLeft(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Quotient_congr___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Quotient_congr(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_once_cell_t lp_mathlib_Quotient_congrRight___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Quotient_congrRight___closed__0;
LEAN_EXPORT lean_object* lp_mathlib_Quotient_congrRight(lean_object*, lean_object*, lean_object*, lean_object*);
static lean_once_cell_t lp_mathlib_finZeroEquiv___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_finZeroEquiv___closed__0;
LEAN_EXPORT lean_object* lp_mathlib_finZeroEquiv;
static lean_once_cell_t lp_mathlib_finZeroEquiv_x27___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_finZeroEquiv_x27___closed__0;
LEAN_EXPORT lean_object* lp_mathlib_finZeroEquiv_x27;
static lean_once_cell_t lp_mathlib_Equiv_ofUnique___at___00Equiv_equivPUnit___at___00finOneEquiv_spec__0_spec__0___lam__1___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Equiv_ofUnique___at___00Equiv_equivPUnit___at___00finOneEquiv_spec__0_spec__0___lam__1___closed__0;
LEAN_EXPORT lean_object* lp_mathlib_Equiv_ofUnique___at___00Equiv_equivPUnit___at___00finOneEquiv_spec__0_spec__0___lam__1(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_ofUnique___at___00Equiv_equivPUnit___at___00finOneEquiv_spec__0_spec__0___lam__0(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_ofUnique___at___00Equiv_equivPUnit___at___00finOneEquiv_spec__0_spec__0___lam__0___boxed(lean_object*);
static const lean_closure_object lp_mathlib_Equiv_ofUnique___at___00Equiv_equivPUnit___at___00finOneEquiv_spec__0_spec__0___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Equiv_ofUnique___at___00Equiv_equivPUnit___at___00finOneEquiv_spec__0_spec__0___lam__0___boxed, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Equiv_ofUnique___at___00Equiv_equivPUnit___at___00finOneEquiv_spec__0_spec__0___closed__0 = (const lean_object*)&lp_mathlib_Equiv_ofUnique___at___00Equiv_equivPUnit___at___00finOneEquiv_spec__0_spec__0___closed__0_value;
static const lean_closure_object lp_mathlib_Equiv_ofUnique___at___00Equiv_equivPUnit___at___00finOneEquiv_spec__0_spec__0___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Equiv_ofUnique___at___00Equiv_equivPUnit___at___00finOneEquiv_spec__0_spec__0___lam__1, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Equiv_ofUnique___at___00Equiv_equivPUnit___at___00finOneEquiv_spec__0_spec__0___closed__1 = (const lean_object*)&lp_mathlib_Equiv_ofUnique___at___00Equiv_equivPUnit___at___00finOneEquiv_spec__0_spec__0___closed__1_value;
static const lean_ctor_object lp_mathlib_Equiv_ofUnique___at___00Equiv_equivPUnit___at___00finOneEquiv_spec__0_spec__0___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_Equiv_ofUnique___at___00Equiv_equivPUnit___at___00finOneEquiv_spec__0_spec__0___closed__0_value),((lean_object*)&lp_mathlib_Equiv_ofUnique___at___00Equiv_equivPUnit___at___00finOneEquiv_spec__0_spec__0___closed__1_value)}};
static const lean_object* lp_mathlib_Equiv_ofUnique___at___00Equiv_equivPUnit___at___00finOneEquiv_spec__0_spec__0___closed__2 = (const lean_object*)&lp_mathlib_Equiv_ofUnique___at___00Equiv_equivPUnit___at___00finOneEquiv_spec__0_spec__0___closed__2_value;
LEAN_EXPORT const lean_object* lp_mathlib_Equiv_ofUnique___at___00Equiv_equivPUnit___at___00finOneEquiv_spec__0_spec__0 = (const lean_object*)&lp_mathlib_Equiv_ofUnique___at___00Equiv_equivPUnit___at___00finOneEquiv_spec__0_spec__0___closed__2_value;
LEAN_EXPORT const lean_object* lp_mathlib_finOneEquiv = (const lean_object*)&lp_mathlib_Equiv_ofUnique___at___00Equiv_equivPUnit___at___00finOneEquiv_spec__0_spec__0___closed__2_value;
LEAN_EXPORT const lean_object* lp_mathlib_Equiv_equivPUnit___at___00finOneEquiv_spec__0 = (const lean_object*)&lp_mathlib_Equiv_ofUnique___at___00Equiv_equivPUnit___at___00finOneEquiv_spec__0_spec__0___closed__2_value;
static lean_once_cell_t lp_mathlib_finTwoEquiv___lam__0___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_finTwoEquiv___lam__0___closed__0;
LEAN_EXPORT uint8_t lp_mathlib_finTwoEquiv___lam__0(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_finTwoEquiv___lam__0___boxed(lean_object*);
static lean_once_cell_t lp_mathlib_finTwoEquiv___lam__1___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_finTwoEquiv___lam__1___closed__0;
LEAN_EXPORT lean_object* lp_mathlib_finTwoEquiv___lam__1(uint8_t);
LEAN_EXPORT lean_object* lp_mathlib_finTwoEquiv___lam__1___boxed(lean_object*);
static const lean_closure_object lp_mathlib_finTwoEquiv___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_finTwoEquiv___lam__0___boxed, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_finTwoEquiv___closed__0 = (const lean_object*)&lp_mathlib_finTwoEquiv___closed__0_value;
static const lean_closure_object lp_mathlib_finTwoEquiv___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_finTwoEquiv___lam__1___boxed, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_finTwoEquiv___closed__1 = (const lean_object*)&lp_mathlib_finTwoEquiv___closed__1_value;
static const lean_ctor_object lp_mathlib_finTwoEquiv___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_finTwoEquiv___closed__0_value),((lean_object*)&lp_mathlib_finTwoEquiv___closed__1_value)}};
static const lean_object* lp_mathlib_finTwoEquiv___closed__2 = (const lean_object*)&lp_mathlib_finTwoEquiv___closed__2_value;
LEAN_EXPORT const lean_object* lp_mathlib_finTwoEquiv = (const lean_object*)&lp_mathlib_finTwoEquiv___closed__2_value;
LEAN_EXPORT lean_object* lp_mathlib_Equiv_sumIsLeft___lam__0(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_sumIsLeft___lam__0___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_sumIsLeft___lam__1(lean_object*);
static const lean_closure_object lp_mathlib_Equiv_sumIsLeft___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Equiv_sumIsLeft___lam__0___boxed, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Equiv_sumIsLeft___closed__0 = (const lean_object*)&lp_mathlib_Equiv_sumIsLeft___closed__0_value;
static const lean_closure_object lp_mathlib_Equiv_sumIsLeft___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Equiv_sumIsLeft___lam__1, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Equiv_sumIsLeft___closed__1 = (const lean_object*)&lp_mathlib_Equiv_sumIsLeft___closed__1_value;
static const lean_ctor_object lp_mathlib_Equiv_sumIsLeft___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_Equiv_sumIsLeft___closed__0_value),((lean_object*)&lp_mathlib_Equiv_sumIsLeft___closed__1_value)}};
static const lean_object* lp_mathlib_Equiv_sumIsLeft___closed__2 = (const lean_object*)&lp_mathlib_Equiv_sumIsLeft___closed__2_value;
LEAN_EXPORT lean_object* lp_mathlib_Equiv_sumIsLeft(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_sumIsRight___lam__0(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_sumIsRight___lam__0___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_sumIsRight___lam__1(lean_object*);
static const lean_closure_object lp_mathlib_Equiv_sumIsRight___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Equiv_sumIsRight___lam__0___boxed, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Equiv_sumIsRight___closed__0 = (const lean_object*)&lp_mathlib_Equiv_sumIsRight___closed__0_value;
static const lean_closure_object lp_mathlib_Equiv_sumIsRight___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Equiv_sumIsRight___lam__1, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Equiv_sumIsRight___closed__1 = (const lean_object*)&lp_mathlib_Equiv_sumIsRight___closed__1_value;
static const lean_ctor_object lp_mathlib_Equiv_sumIsRight___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_Equiv_sumIsRight___closed__0_value),((lean_object*)&lp_mathlib_Equiv_sumIsRight___closed__1_value)}};
static const lean_object* lp_mathlib_Equiv_sumIsRight___closed__2 = (const lean_object*)&lp_mathlib_Equiv_sumIsRight___closed__2_value;
LEAN_EXPORT lean_object* lp_mathlib_Equiv_sumIsRight(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_le(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_le___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_lt(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_lt___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_max___redArg___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_max___redArg___lam__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_Equiv_max___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Equiv_max___redArg___lam__0, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Equiv_max___redArg___closed__0 = (const lean_object*)&lp_mathlib_Equiv_max___redArg___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_Equiv_max___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_max(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_min___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_min(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Equiv_ord___redArg___lam__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_ord___redArg___lam__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_ord___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_ord(lean_object*, lean_object*, lean_object*, lean_object*);
static lean_object* _init_lp_mathlib_Equiv_left__inv___autoParam___closed__12(void){
_start:
{
lean_object* v___x_27_; lean_object* v___x_28_; 
v___x_27_ = ((lean_object*)(lp_mathlib_Equiv_left__inv___autoParam___closed__10));
v___x_28_ = l_Lean_mkAtom(v___x_27_);
return v___x_28_;
}
}
static lean_object* _init_lp_mathlib_Equiv_left__inv___autoParam___closed__13(void){
_start:
{
lean_object* v___x_29_; lean_object* v___x_30_; lean_object* v___x_31_; 
v___x_29_ = lean_obj_once(&lp_mathlib_Equiv_left__inv___autoParam___closed__12, &lp_mathlib_Equiv_left__inv___autoParam___closed__12_once, _init_lp_mathlib_Equiv_left__inv___autoParam___closed__12);
v___x_30_ = ((lean_object*)(lp_mathlib_Equiv_left__inv___autoParam___closed__5));
v___x_31_ = lean_array_push(v___x_30_, v___x_29_);
return v___x_31_;
}
}
static lean_object* _init_lp_mathlib_Equiv_left__inv___autoParam___closed__15(void){
_start:
{
lean_object* v___x_36_; lean_object* v___x_37_; lean_object* v___x_38_; 
v___x_36_ = ((lean_object*)(lp_mathlib_Equiv_left__inv___autoParam___closed__14));
v___x_37_ = lean_obj_once(&lp_mathlib_Equiv_left__inv___autoParam___closed__13, &lp_mathlib_Equiv_left__inv___autoParam___closed__13_once, _init_lp_mathlib_Equiv_left__inv___autoParam___closed__13);
v___x_38_ = lean_array_push(v___x_37_, v___x_36_);
return v___x_38_;
}
}
static lean_object* _init_lp_mathlib_Equiv_left__inv___autoParam___closed__16(void){
_start:
{
lean_object* v___x_39_; lean_object* v___x_40_; lean_object* v___x_41_; lean_object* v___x_42_; 
v___x_39_ = lean_obj_once(&lp_mathlib_Equiv_left__inv___autoParam___closed__15, &lp_mathlib_Equiv_left__inv___autoParam___closed__15_once, _init_lp_mathlib_Equiv_left__inv___autoParam___closed__15);
v___x_40_ = ((lean_object*)(lp_mathlib_Equiv_left__inv___autoParam___closed__11));
v___x_41_ = lean_box(2);
v___x_42_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_42_, 0, v___x_41_);
lean_ctor_set(v___x_42_, 1, v___x_40_);
lean_ctor_set(v___x_42_, 2, v___x_39_);
return v___x_42_;
}
}
static lean_object* _init_lp_mathlib_Equiv_left__inv___autoParam___closed__17(void){
_start:
{
lean_object* v___x_43_; lean_object* v___x_44_; lean_object* v___x_45_; 
v___x_43_ = lean_obj_once(&lp_mathlib_Equiv_left__inv___autoParam___closed__16, &lp_mathlib_Equiv_left__inv___autoParam___closed__16_once, _init_lp_mathlib_Equiv_left__inv___autoParam___closed__16);
v___x_44_ = ((lean_object*)(lp_mathlib_Equiv_left__inv___autoParam___closed__5));
v___x_45_ = lean_array_push(v___x_44_, v___x_43_);
return v___x_45_;
}
}
static lean_object* _init_lp_mathlib_Equiv_left__inv___autoParam___closed__19(void){
_start:
{
lean_object* v___x_47_; lean_object* v___x_48_; 
v___x_47_ = ((lean_object*)(lp_mathlib_Equiv_left__inv___autoParam___closed__18));
v___x_48_ = l_Lean_mkAtom(v___x_47_);
return v___x_48_;
}
}
static lean_object* _init_lp_mathlib_Equiv_left__inv___autoParam___closed__20(void){
_start:
{
lean_object* v___x_49_; lean_object* v___x_50_; lean_object* v___x_51_; 
v___x_49_ = lean_obj_once(&lp_mathlib_Equiv_left__inv___autoParam___closed__19, &lp_mathlib_Equiv_left__inv___autoParam___closed__19_once, _init_lp_mathlib_Equiv_left__inv___autoParam___closed__19);
v___x_50_ = lean_obj_once(&lp_mathlib_Equiv_left__inv___autoParam___closed__17, &lp_mathlib_Equiv_left__inv___autoParam___closed__17_once, _init_lp_mathlib_Equiv_left__inv___autoParam___closed__17);
v___x_51_ = lean_array_push(v___x_50_, v___x_49_);
return v___x_51_;
}
}
static lean_object* _init_lp_mathlib_Equiv_left__inv___autoParam___closed__23(void){
_start:
{
lean_object* v___x_58_; lean_object* v___x_59_; 
v___x_58_ = ((lean_object*)(lp_mathlib_Equiv_left__inv___autoParam___closed__21));
v___x_59_ = l_Lean_mkAtom(v___x_58_);
return v___x_59_;
}
}
static lean_object* _init_lp_mathlib_Equiv_left__inv___autoParam___closed__24(void){
_start:
{
lean_object* v___x_60_; lean_object* v___x_61_; lean_object* v___x_62_; 
v___x_60_ = lean_obj_once(&lp_mathlib_Equiv_left__inv___autoParam___closed__23, &lp_mathlib_Equiv_left__inv___autoParam___closed__23_once, _init_lp_mathlib_Equiv_left__inv___autoParam___closed__23);
v___x_61_ = ((lean_object*)(lp_mathlib_Equiv_left__inv___autoParam___closed__5));
v___x_62_ = lean_array_push(v___x_61_, v___x_60_);
return v___x_62_;
}
}
static lean_object* _init_lp_mathlib_Equiv_left__inv___autoParam___closed__28(void){
_start:
{
lean_object* v___x_67_; lean_object* v___x_68_; 
v___x_67_ = ((lean_object*)(lp_mathlib_Equiv_left__inv___autoParam___closed__27));
v___x_68_ = l_Lean_mkAtom(v___x_67_);
return v___x_68_;
}
}
static lean_object* _init_lp_mathlib_Equiv_left__inv___autoParam___closed__29(void){
_start:
{
lean_object* v___x_69_; lean_object* v___x_70_; lean_object* v___x_71_; 
v___x_69_ = lean_obj_once(&lp_mathlib_Equiv_left__inv___autoParam___closed__28, &lp_mathlib_Equiv_left__inv___autoParam___closed__28_once, _init_lp_mathlib_Equiv_left__inv___autoParam___closed__28);
v___x_70_ = ((lean_object*)(lp_mathlib_Equiv_left__inv___autoParam___closed__5));
v___x_71_ = lean_array_push(v___x_70_, v___x_69_);
return v___x_71_;
}
}
static lean_object* _init_lp_mathlib_Equiv_left__inv___autoParam___closed__33(void){
_start:
{
lean_object* v___x_79_; lean_object* v___x_80_; 
v___x_79_ = ((lean_object*)(lp_mathlib_Equiv_left__inv___autoParam___closed__32));
v___x_80_ = l_Lean_mkAtom(v___x_79_);
return v___x_80_;
}
}
static lean_object* _init_lp_mathlib_Equiv_left__inv___autoParam___closed__34(void){
_start:
{
lean_object* v___x_81_; lean_object* v___x_82_; lean_object* v___x_83_; 
v___x_81_ = lean_obj_once(&lp_mathlib_Equiv_left__inv___autoParam___closed__33, &lp_mathlib_Equiv_left__inv___autoParam___closed__33_once, _init_lp_mathlib_Equiv_left__inv___autoParam___closed__33);
v___x_82_ = ((lean_object*)(lp_mathlib_Equiv_left__inv___autoParam___closed__5));
v___x_83_ = lean_array_push(v___x_82_, v___x_81_);
return v___x_83_;
}
}
static lean_object* _init_lp_mathlib_Equiv_left__inv___autoParam___closed__35(void){
_start:
{
lean_object* v___x_84_; lean_object* v___x_85_; lean_object* v___x_86_; lean_object* v___x_87_; 
v___x_84_ = lean_obj_once(&lp_mathlib_Equiv_left__inv___autoParam___closed__34, &lp_mathlib_Equiv_left__inv___autoParam___closed__34_once, _init_lp_mathlib_Equiv_left__inv___autoParam___closed__34);
v___x_85_ = ((lean_object*)(lp_mathlib_Equiv_left__inv___autoParam___closed__31));
v___x_86_ = lean_box(2);
v___x_87_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_87_, 0, v___x_86_);
lean_ctor_set(v___x_87_, 1, v___x_85_);
lean_ctor_set(v___x_87_, 2, v___x_84_);
return v___x_87_;
}
}
static lean_object* _init_lp_mathlib_Equiv_left__inv___autoParam___closed__36(void){
_start:
{
lean_object* v___x_88_; lean_object* v___x_89_; lean_object* v___x_90_; 
v___x_88_ = lean_obj_once(&lp_mathlib_Equiv_left__inv___autoParam___closed__35, &lp_mathlib_Equiv_left__inv___autoParam___closed__35_once, _init_lp_mathlib_Equiv_left__inv___autoParam___closed__35);
v___x_89_ = ((lean_object*)(lp_mathlib_Equiv_left__inv___autoParam___closed__5));
v___x_90_ = lean_array_push(v___x_89_, v___x_88_);
return v___x_90_;
}
}
static lean_object* _init_lp_mathlib_Equiv_left__inv___autoParam___closed__37(void){
_start:
{
lean_object* v___x_91_; lean_object* v___x_92_; lean_object* v___x_93_; lean_object* v___x_94_; 
v___x_91_ = lean_obj_once(&lp_mathlib_Equiv_left__inv___autoParam___closed__36, &lp_mathlib_Equiv_left__inv___autoParam___closed__36_once, _init_lp_mathlib_Equiv_left__inv___autoParam___closed__36);
v___x_92_ = ((lean_object*)(lp_mathlib_Equiv_left__inv___autoParam___closed__9));
v___x_93_ = lean_box(2);
v___x_94_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_94_, 0, v___x_93_);
lean_ctor_set(v___x_94_, 1, v___x_92_);
lean_ctor_set(v___x_94_, 2, v___x_91_);
return v___x_94_;
}
}
static lean_object* _init_lp_mathlib_Equiv_left__inv___autoParam___closed__38(void){
_start:
{
lean_object* v___x_95_; lean_object* v___x_96_; lean_object* v___x_97_; 
v___x_95_ = lean_obj_once(&lp_mathlib_Equiv_left__inv___autoParam___closed__37, &lp_mathlib_Equiv_left__inv___autoParam___closed__37_once, _init_lp_mathlib_Equiv_left__inv___autoParam___closed__37);
v___x_96_ = ((lean_object*)(lp_mathlib_Equiv_left__inv___autoParam___closed__5));
v___x_97_ = lean_array_push(v___x_96_, v___x_95_);
return v___x_97_;
}
}
static lean_object* _init_lp_mathlib_Equiv_left__inv___autoParam___closed__39(void){
_start:
{
lean_object* v___x_98_; lean_object* v___x_99_; lean_object* v___x_100_; lean_object* v___x_101_; 
v___x_98_ = lean_obj_once(&lp_mathlib_Equiv_left__inv___autoParam___closed__38, &lp_mathlib_Equiv_left__inv___autoParam___closed__38_once, _init_lp_mathlib_Equiv_left__inv___autoParam___closed__38);
v___x_99_ = ((lean_object*)(lp_mathlib_Equiv_left__inv___autoParam___closed__7));
v___x_100_ = lean_box(2);
v___x_101_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_101_, 0, v___x_100_);
lean_ctor_set(v___x_101_, 1, v___x_99_);
lean_ctor_set(v___x_101_, 2, v___x_98_);
return v___x_101_;
}
}
static lean_object* _init_lp_mathlib_Equiv_left__inv___autoParam___closed__40(void){
_start:
{
lean_object* v___x_102_; lean_object* v___x_103_; lean_object* v___x_104_; 
v___x_102_ = lean_obj_once(&lp_mathlib_Equiv_left__inv___autoParam___closed__39, &lp_mathlib_Equiv_left__inv___autoParam___closed__39_once, _init_lp_mathlib_Equiv_left__inv___autoParam___closed__39);
v___x_103_ = ((lean_object*)(lp_mathlib_Equiv_left__inv___autoParam___closed__5));
v___x_104_ = lean_array_push(v___x_103_, v___x_102_);
return v___x_104_;
}
}
static lean_object* _init_lp_mathlib_Equiv_left__inv___autoParam___closed__41(void){
_start:
{
lean_object* v___x_105_; lean_object* v___x_106_; lean_object* v___x_107_; lean_object* v___x_108_; 
v___x_105_ = lean_obj_once(&lp_mathlib_Equiv_left__inv___autoParam___closed__40, &lp_mathlib_Equiv_left__inv___autoParam___closed__40_once, _init_lp_mathlib_Equiv_left__inv___autoParam___closed__40);
v___x_106_ = ((lean_object*)(lp_mathlib_Equiv_left__inv___autoParam___closed__4));
v___x_107_ = lean_box(2);
v___x_108_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_108_, 0, v___x_107_);
lean_ctor_set(v___x_108_, 1, v___x_106_);
lean_ctor_set(v___x_108_, 2, v___x_105_);
return v___x_108_;
}
}
static lean_object* _init_lp_mathlib_Equiv_left__inv___autoParam___closed__42(void){
_start:
{
lean_object* v___x_109_; lean_object* v___x_110_; lean_object* v___x_111_; 
v___x_109_ = lean_obj_once(&lp_mathlib_Equiv_left__inv___autoParam___closed__41, &lp_mathlib_Equiv_left__inv___autoParam___closed__41_once, _init_lp_mathlib_Equiv_left__inv___autoParam___closed__41);
v___x_110_ = lean_obj_once(&lp_mathlib_Equiv_left__inv___autoParam___closed__29, &lp_mathlib_Equiv_left__inv___autoParam___closed__29_once, _init_lp_mathlib_Equiv_left__inv___autoParam___closed__29);
v___x_111_ = lean_array_push(v___x_110_, v___x_109_);
return v___x_111_;
}
}
static lean_object* _init_lp_mathlib_Equiv_left__inv___autoParam___closed__43(void){
_start:
{
lean_object* v___x_112_; lean_object* v___x_113_; lean_object* v___x_114_; lean_object* v___x_115_; 
v___x_112_ = lean_obj_once(&lp_mathlib_Equiv_left__inv___autoParam___closed__42, &lp_mathlib_Equiv_left__inv___autoParam___closed__42_once, _init_lp_mathlib_Equiv_left__inv___autoParam___closed__42);
v___x_113_ = ((lean_object*)(lp_mathlib_Equiv_left__inv___autoParam___closed__26));
v___x_114_ = lean_box(2);
v___x_115_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_115_, 0, v___x_114_);
lean_ctor_set(v___x_115_, 1, v___x_113_);
lean_ctor_set(v___x_115_, 2, v___x_112_);
return v___x_115_;
}
}
static lean_object* _init_lp_mathlib_Equiv_left__inv___autoParam___closed__44(void){
_start:
{
lean_object* v___x_116_; lean_object* v___x_117_; lean_object* v___x_118_; 
v___x_116_ = lean_obj_once(&lp_mathlib_Equiv_left__inv___autoParam___closed__43, &lp_mathlib_Equiv_left__inv___autoParam___closed__43_once, _init_lp_mathlib_Equiv_left__inv___autoParam___closed__43);
v___x_117_ = ((lean_object*)(lp_mathlib_Equiv_left__inv___autoParam___closed__5));
v___x_118_ = lean_array_push(v___x_117_, v___x_116_);
return v___x_118_;
}
}
static lean_object* _init_lp_mathlib_Equiv_left__inv___autoParam___closed__51(void){
_start:
{
lean_object* v___x_134_; lean_object* v___x_135_; 
v___x_134_ = ((lean_object*)(lp_mathlib_Equiv_left__inv___autoParam___closed__49));
v___x_135_ = l_Lean_mkAtom(v___x_134_);
return v___x_135_;
}
}
static lean_object* _init_lp_mathlib_Equiv_left__inv___autoParam___closed__52(void){
_start:
{
lean_object* v___x_136_; lean_object* v___x_137_; lean_object* v___x_138_; 
v___x_136_ = lean_obj_once(&lp_mathlib_Equiv_left__inv___autoParam___closed__51, &lp_mathlib_Equiv_left__inv___autoParam___closed__51_once, _init_lp_mathlib_Equiv_left__inv___autoParam___closed__51);
v___x_137_ = ((lean_object*)(lp_mathlib_Equiv_left__inv___autoParam___closed__5));
v___x_138_ = lean_array_push(v___x_137_, v___x_136_);
return v___x_138_;
}
}
static lean_object* _init_lp_mathlib_Equiv_left__inv___autoParam___closed__53(void){
_start:
{
lean_object* v___x_139_; lean_object* v___x_140_; lean_object* v___x_141_; 
v___x_139_ = ((lean_object*)(lp_mathlib_Equiv_left__inv___autoParam___closed__14));
v___x_140_ = lean_obj_once(&lp_mathlib_Equiv_left__inv___autoParam___closed__52, &lp_mathlib_Equiv_left__inv___autoParam___closed__52_once, _init_lp_mathlib_Equiv_left__inv___autoParam___closed__52);
v___x_141_ = lean_array_push(v___x_140_, v___x_139_);
return v___x_141_;
}
}
static lean_object* _init_lp_mathlib_Equiv_left__inv___autoParam___closed__54(void){
_start:
{
lean_object* v___x_142_; lean_object* v___x_143_; lean_object* v___x_144_; 
v___x_142_ = ((lean_object*)(lp_mathlib_Equiv_left__inv___autoParam___closed__14));
v___x_143_ = lean_obj_once(&lp_mathlib_Equiv_left__inv___autoParam___closed__53, &lp_mathlib_Equiv_left__inv___autoParam___closed__53_once, _init_lp_mathlib_Equiv_left__inv___autoParam___closed__53);
v___x_144_ = lean_array_push(v___x_143_, v___x_142_);
return v___x_144_;
}
}
static lean_object* _init_lp_mathlib_Equiv_left__inv___autoParam___closed__55(void){
_start:
{
lean_object* v___x_145_; lean_object* v___x_146_; lean_object* v___x_147_; lean_object* v___x_148_; 
v___x_145_ = lean_obj_once(&lp_mathlib_Equiv_left__inv___autoParam___closed__54, &lp_mathlib_Equiv_left__inv___autoParam___closed__54_once, _init_lp_mathlib_Equiv_left__inv___autoParam___closed__54);
v___x_146_ = ((lean_object*)(lp_mathlib_Equiv_left__inv___autoParam___closed__50));
v___x_147_ = lean_box(2);
v___x_148_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_148_, 0, v___x_147_);
lean_ctor_set(v___x_148_, 1, v___x_146_);
lean_ctor_set(v___x_148_, 2, v___x_145_);
return v___x_148_;
}
}
static lean_object* _init_lp_mathlib_Equiv_left__inv___autoParam___closed__56(void){
_start:
{
lean_object* v___x_149_; lean_object* v___x_150_; lean_object* v___x_151_; 
v___x_149_ = lean_obj_once(&lp_mathlib_Equiv_left__inv___autoParam___closed__55, &lp_mathlib_Equiv_left__inv___autoParam___closed__55_once, _init_lp_mathlib_Equiv_left__inv___autoParam___closed__55);
v___x_150_ = ((lean_object*)(lp_mathlib_Equiv_left__inv___autoParam___closed__5));
v___x_151_ = lean_array_push(v___x_150_, v___x_149_);
return v___x_151_;
}
}
static lean_object* _init_lp_mathlib_Equiv_left__inv___autoParam___closed__58(void){
_start:
{
lean_object* v___x_153_; lean_object* v___x_154_; 
v___x_153_ = ((lean_object*)(lp_mathlib_Equiv_left__inv___autoParam___closed__57));
v___x_154_ = l_Lean_mkAtom(v___x_153_);
return v___x_154_;
}
}
static lean_object* _init_lp_mathlib_Equiv_left__inv___autoParam___closed__59(void){
_start:
{
lean_object* v___x_155_; lean_object* v___x_156_; lean_object* v___x_157_; 
v___x_155_ = lean_obj_once(&lp_mathlib_Equiv_left__inv___autoParam___closed__58, &lp_mathlib_Equiv_left__inv___autoParam___closed__58_once, _init_lp_mathlib_Equiv_left__inv___autoParam___closed__58);
v___x_156_ = lean_obj_once(&lp_mathlib_Equiv_left__inv___autoParam___closed__56, &lp_mathlib_Equiv_left__inv___autoParam___closed__56_once, _init_lp_mathlib_Equiv_left__inv___autoParam___closed__56);
v___x_157_ = lean_array_push(v___x_156_, v___x_155_);
return v___x_157_;
}
}
static lean_object* _init_lp_mathlib_Equiv_left__inv___autoParam___closed__60(void){
_start:
{
lean_object* v___x_158_; lean_object* v___x_159_; lean_object* v___x_160_; 
v___x_158_ = lean_obj_once(&lp_mathlib_Equiv_left__inv___autoParam___closed__35, &lp_mathlib_Equiv_left__inv___autoParam___closed__35_once, _init_lp_mathlib_Equiv_left__inv___autoParam___closed__35);
v___x_159_ = lean_obj_once(&lp_mathlib_Equiv_left__inv___autoParam___closed__59, &lp_mathlib_Equiv_left__inv___autoParam___closed__59_once, _init_lp_mathlib_Equiv_left__inv___autoParam___closed__59);
v___x_160_ = lean_array_push(v___x_159_, v___x_158_);
return v___x_160_;
}
}
static lean_object* _init_lp_mathlib_Equiv_left__inv___autoParam___closed__61(void){
_start:
{
lean_object* v___x_161_; lean_object* v___x_162_; lean_object* v___x_163_; lean_object* v___x_164_; 
v___x_161_ = lean_obj_once(&lp_mathlib_Equiv_left__inv___autoParam___closed__60, &lp_mathlib_Equiv_left__inv___autoParam___closed__60_once, _init_lp_mathlib_Equiv_left__inv___autoParam___closed__60);
v___x_162_ = ((lean_object*)(lp_mathlib_Equiv_left__inv___autoParam___closed__46));
v___x_163_ = lean_box(2);
v___x_164_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_164_, 0, v___x_163_);
lean_ctor_set(v___x_164_, 1, v___x_162_);
lean_ctor_set(v___x_164_, 2, v___x_161_);
return v___x_164_;
}
}
static lean_object* _init_lp_mathlib_Equiv_left__inv___autoParam___closed__62(void){
_start:
{
lean_object* v___x_165_; lean_object* v___x_166_; lean_object* v___x_167_; 
v___x_165_ = lean_obj_once(&lp_mathlib_Equiv_left__inv___autoParam___closed__61, &lp_mathlib_Equiv_left__inv___autoParam___closed__61_once, _init_lp_mathlib_Equiv_left__inv___autoParam___closed__61);
v___x_166_ = ((lean_object*)(lp_mathlib_Equiv_left__inv___autoParam___closed__5));
v___x_167_ = lean_array_push(v___x_166_, v___x_165_);
return v___x_167_;
}
}
static lean_object* _init_lp_mathlib_Equiv_left__inv___autoParam___closed__63(void){
_start:
{
lean_object* v___x_168_; lean_object* v___x_169_; lean_object* v___x_170_; lean_object* v___x_171_; 
v___x_168_ = lean_obj_once(&lp_mathlib_Equiv_left__inv___autoParam___closed__62, &lp_mathlib_Equiv_left__inv___autoParam___closed__62_once, _init_lp_mathlib_Equiv_left__inv___autoParam___closed__62);
v___x_169_ = ((lean_object*)(lp_mathlib_Equiv_left__inv___autoParam___closed__9));
v___x_170_ = lean_box(2);
v___x_171_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_171_, 0, v___x_170_);
lean_ctor_set(v___x_171_, 1, v___x_169_);
lean_ctor_set(v___x_171_, 2, v___x_168_);
return v___x_171_;
}
}
static lean_object* _init_lp_mathlib_Equiv_left__inv___autoParam___closed__64(void){
_start:
{
lean_object* v___x_172_; lean_object* v___x_173_; lean_object* v___x_174_; 
v___x_172_ = lean_obj_once(&lp_mathlib_Equiv_left__inv___autoParam___closed__63, &lp_mathlib_Equiv_left__inv___autoParam___closed__63_once, _init_lp_mathlib_Equiv_left__inv___autoParam___closed__63);
v___x_173_ = ((lean_object*)(lp_mathlib_Equiv_left__inv___autoParam___closed__5));
v___x_174_ = lean_array_push(v___x_173_, v___x_172_);
return v___x_174_;
}
}
static lean_object* _init_lp_mathlib_Equiv_left__inv___autoParam___closed__65(void){
_start:
{
lean_object* v___x_175_; lean_object* v___x_176_; lean_object* v___x_177_; lean_object* v___x_178_; 
v___x_175_ = lean_obj_once(&lp_mathlib_Equiv_left__inv___autoParam___closed__64, &lp_mathlib_Equiv_left__inv___autoParam___closed__64_once, _init_lp_mathlib_Equiv_left__inv___autoParam___closed__64);
v___x_176_ = ((lean_object*)(lp_mathlib_Equiv_left__inv___autoParam___closed__7));
v___x_177_ = lean_box(2);
v___x_178_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_178_, 0, v___x_177_);
lean_ctor_set(v___x_178_, 1, v___x_176_);
lean_ctor_set(v___x_178_, 2, v___x_175_);
return v___x_178_;
}
}
static lean_object* _init_lp_mathlib_Equiv_left__inv___autoParam___closed__66(void){
_start:
{
lean_object* v___x_179_; lean_object* v___x_180_; lean_object* v___x_181_; 
v___x_179_ = lean_obj_once(&lp_mathlib_Equiv_left__inv___autoParam___closed__65, &lp_mathlib_Equiv_left__inv___autoParam___closed__65_once, _init_lp_mathlib_Equiv_left__inv___autoParam___closed__65);
v___x_180_ = ((lean_object*)(lp_mathlib_Equiv_left__inv___autoParam___closed__5));
v___x_181_ = lean_array_push(v___x_180_, v___x_179_);
return v___x_181_;
}
}
static lean_object* _init_lp_mathlib_Equiv_left__inv___autoParam___closed__67(void){
_start:
{
lean_object* v___x_182_; lean_object* v___x_183_; lean_object* v___x_184_; lean_object* v___x_185_; 
v___x_182_ = lean_obj_once(&lp_mathlib_Equiv_left__inv___autoParam___closed__66, &lp_mathlib_Equiv_left__inv___autoParam___closed__66_once, _init_lp_mathlib_Equiv_left__inv___autoParam___closed__66);
v___x_183_ = ((lean_object*)(lp_mathlib_Equiv_left__inv___autoParam___closed__4));
v___x_184_ = lean_box(2);
v___x_185_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_185_, 0, v___x_184_);
lean_ctor_set(v___x_185_, 1, v___x_183_);
lean_ctor_set(v___x_185_, 2, v___x_182_);
return v___x_185_;
}
}
static lean_object* _init_lp_mathlib_Equiv_left__inv___autoParam___closed__68(void){
_start:
{
lean_object* v___x_186_; lean_object* v___x_187_; lean_object* v___x_188_; 
v___x_186_ = lean_obj_once(&lp_mathlib_Equiv_left__inv___autoParam___closed__67, &lp_mathlib_Equiv_left__inv___autoParam___closed__67_once, _init_lp_mathlib_Equiv_left__inv___autoParam___closed__67);
v___x_187_ = lean_obj_once(&lp_mathlib_Equiv_left__inv___autoParam___closed__29, &lp_mathlib_Equiv_left__inv___autoParam___closed__29_once, _init_lp_mathlib_Equiv_left__inv___autoParam___closed__29);
v___x_188_ = lean_array_push(v___x_187_, v___x_186_);
return v___x_188_;
}
}
static lean_object* _init_lp_mathlib_Equiv_left__inv___autoParam___closed__69(void){
_start:
{
lean_object* v___x_189_; lean_object* v___x_190_; lean_object* v___x_191_; lean_object* v___x_192_; 
v___x_189_ = lean_obj_once(&lp_mathlib_Equiv_left__inv___autoParam___closed__68, &lp_mathlib_Equiv_left__inv___autoParam___closed__68_once, _init_lp_mathlib_Equiv_left__inv___autoParam___closed__68);
v___x_190_ = ((lean_object*)(lp_mathlib_Equiv_left__inv___autoParam___closed__26));
v___x_191_ = lean_box(2);
v___x_192_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_192_, 0, v___x_191_);
lean_ctor_set(v___x_192_, 1, v___x_190_);
lean_ctor_set(v___x_192_, 2, v___x_189_);
return v___x_192_;
}
}
static lean_object* _init_lp_mathlib_Equiv_left__inv___autoParam___closed__70(void){
_start:
{
lean_object* v___x_193_; lean_object* v___x_194_; lean_object* v___x_195_; 
v___x_193_ = lean_obj_once(&lp_mathlib_Equiv_left__inv___autoParam___closed__69, &lp_mathlib_Equiv_left__inv___autoParam___closed__69_once, _init_lp_mathlib_Equiv_left__inv___autoParam___closed__69);
v___x_194_ = lean_obj_once(&lp_mathlib_Equiv_left__inv___autoParam___closed__44, &lp_mathlib_Equiv_left__inv___autoParam___closed__44_once, _init_lp_mathlib_Equiv_left__inv___autoParam___closed__44);
v___x_195_ = lean_array_push(v___x_194_, v___x_193_);
return v___x_195_;
}
}
static lean_object* _init_lp_mathlib_Equiv_left__inv___autoParam___closed__71(void){
_start:
{
lean_object* v___x_196_; lean_object* v___x_197_; lean_object* v___x_198_; lean_object* v___x_199_; 
v___x_196_ = lean_obj_once(&lp_mathlib_Equiv_left__inv___autoParam___closed__70, &lp_mathlib_Equiv_left__inv___autoParam___closed__70_once, _init_lp_mathlib_Equiv_left__inv___autoParam___closed__70);
v___x_197_ = ((lean_object*)(lp_mathlib_Equiv_left__inv___autoParam___closed__9));
v___x_198_ = lean_box(2);
v___x_199_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_199_, 0, v___x_198_);
lean_ctor_set(v___x_199_, 1, v___x_197_);
lean_ctor_set(v___x_199_, 2, v___x_196_);
return v___x_199_;
}
}
static lean_object* _init_lp_mathlib_Equiv_left__inv___autoParam___closed__72(void){
_start:
{
lean_object* v___x_200_; lean_object* v___x_201_; lean_object* v___x_202_; 
v___x_200_ = lean_obj_once(&lp_mathlib_Equiv_left__inv___autoParam___closed__71, &lp_mathlib_Equiv_left__inv___autoParam___closed__71_once, _init_lp_mathlib_Equiv_left__inv___autoParam___closed__71);
v___x_201_ = lean_obj_once(&lp_mathlib_Equiv_left__inv___autoParam___closed__24, &lp_mathlib_Equiv_left__inv___autoParam___closed__24_once, _init_lp_mathlib_Equiv_left__inv___autoParam___closed__24);
v___x_202_ = lean_array_push(v___x_201_, v___x_200_);
return v___x_202_;
}
}
static lean_object* _init_lp_mathlib_Equiv_left__inv___autoParam___closed__73(void){
_start:
{
lean_object* v___x_203_; lean_object* v___x_204_; lean_object* v___x_205_; lean_object* v___x_206_; 
v___x_203_ = lean_obj_once(&lp_mathlib_Equiv_left__inv___autoParam___closed__72, &lp_mathlib_Equiv_left__inv___autoParam___closed__72_once, _init_lp_mathlib_Equiv_left__inv___autoParam___closed__72);
v___x_204_ = ((lean_object*)(lp_mathlib_Equiv_left__inv___autoParam___closed__22));
v___x_205_ = lean_box(2);
v___x_206_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_206_, 0, v___x_205_);
lean_ctor_set(v___x_206_, 1, v___x_204_);
lean_ctor_set(v___x_206_, 2, v___x_203_);
return v___x_206_;
}
}
static lean_object* _init_lp_mathlib_Equiv_left__inv___autoParam___closed__74(void){
_start:
{
lean_object* v___x_207_; lean_object* v___x_208_; lean_object* v___x_209_; 
v___x_207_ = lean_obj_once(&lp_mathlib_Equiv_left__inv___autoParam___closed__73, &lp_mathlib_Equiv_left__inv___autoParam___closed__73_once, _init_lp_mathlib_Equiv_left__inv___autoParam___closed__73);
v___x_208_ = lean_obj_once(&lp_mathlib_Equiv_left__inv___autoParam___closed__20, &lp_mathlib_Equiv_left__inv___autoParam___closed__20_once, _init_lp_mathlib_Equiv_left__inv___autoParam___closed__20);
v___x_209_ = lean_array_push(v___x_208_, v___x_207_);
return v___x_209_;
}
}
static lean_object* _init_lp_mathlib_Equiv_left__inv___autoParam___closed__75(void){
_start:
{
lean_object* v___x_210_; lean_object* v___x_211_; lean_object* v___x_212_; lean_object* v___x_213_; 
v___x_210_ = lean_obj_once(&lp_mathlib_Equiv_left__inv___autoParam___closed__74, &lp_mathlib_Equiv_left__inv___autoParam___closed__74_once, _init_lp_mathlib_Equiv_left__inv___autoParam___closed__74);
v___x_211_ = ((lean_object*)(lp_mathlib_Equiv_left__inv___autoParam___closed__9));
v___x_212_ = lean_box(2);
v___x_213_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_213_, 0, v___x_212_);
lean_ctor_set(v___x_213_, 1, v___x_211_);
lean_ctor_set(v___x_213_, 2, v___x_210_);
return v___x_213_;
}
}
static lean_object* _init_lp_mathlib_Equiv_left__inv___autoParam___closed__76(void){
_start:
{
lean_object* v___x_214_; lean_object* v___x_215_; lean_object* v___x_216_; 
v___x_214_ = lean_obj_once(&lp_mathlib_Equiv_left__inv___autoParam___closed__75, &lp_mathlib_Equiv_left__inv___autoParam___closed__75_once, _init_lp_mathlib_Equiv_left__inv___autoParam___closed__75);
v___x_215_ = ((lean_object*)(lp_mathlib_Equiv_left__inv___autoParam___closed__5));
v___x_216_ = lean_array_push(v___x_215_, v___x_214_);
return v___x_216_;
}
}
static lean_object* _init_lp_mathlib_Equiv_left__inv___autoParam___closed__77(void){
_start:
{
lean_object* v___x_217_; lean_object* v___x_218_; lean_object* v___x_219_; lean_object* v___x_220_; 
v___x_217_ = lean_obj_once(&lp_mathlib_Equiv_left__inv___autoParam___closed__76, &lp_mathlib_Equiv_left__inv___autoParam___closed__76_once, _init_lp_mathlib_Equiv_left__inv___autoParam___closed__76);
v___x_218_ = ((lean_object*)(lp_mathlib_Equiv_left__inv___autoParam___closed__7));
v___x_219_ = lean_box(2);
v___x_220_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_220_, 0, v___x_219_);
lean_ctor_set(v___x_220_, 1, v___x_218_);
lean_ctor_set(v___x_220_, 2, v___x_217_);
return v___x_220_;
}
}
static lean_object* _init_lp_mathlib_Equiv_left__inv___autoParam___closed__78(void){
_start:
{
lean_object* v___x_221_; lean_object* v___x_222_; lean_object* v___x_223_; 
v___x_221_ = lean_obj_once(&lp_mathlib_Equiv_left__inv___autoParam___closed__77, &lp_mathlib_Equiv_left__inv___autoParam___closed__77_once, _init_lp_mathlib_Equiv_left__inv___autoParam___closed__77);
v___x_222_ = ((lean_object*)(lp_mathlib_Equiv_left__inv___autoParam___closed__5));
v___x_223_ = lean_array_push(v___x_222_, v___x_221_);
return v___x_223_;
}
}
static lean_object* _init_lp_mathlib_Equiv_left__inv___autoParam___closed__79(void){
_start:
{
lean_object* v___x_224_; lean_object* v___x_225_; lean_object* v___x_226_; lean_object* v___x_227_; 
v___x_224_ = lean_obj_once(&lp_mathlib_Equiv_left__inv___autoParam___closed__78, &lp_mathlib_Equiv_left__inv___autoParam___closed__78_once, _init_lp_mathlib_Equiv_left__inv___autoParam___closed__78);
v___x_225_ = ((lean_object*)(lp_mathlib_Equiv_left__inv___autoParam___closed__4));
v___x_226_ = lean_box(2);
v___x_227_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_227_, 0, v___x_226_);
lean_ctor_set(v___x_227_, 1, v___x_225_);
lean_ctor_set(v___x_227_, 2, v___x_224_);
return v___x_227_;
}
}
static lean_object* _init_lp_mathlib_Equiv_left__inv___autoParam(void){
_start:
{
lean_object* v___x_228_; 
v___x_228_ = lean_obj_once(&lp_mathlib_Equiv_left__inv___autoParam___closed__79, &lp_mathlib_Equiv_left__inv___autoParam___closed__79_once, _init_lp_mathlib_Equiv_left__inv___autoParam___closed__79);
return v___x_228_;
}
}
static lean_object* _init_lp_mathlib_Equiv_right__inv___autoParam(void){
_start:
{
lean_object* v___x_229_; 
v___x_229_ = lean_obj_once(&lp_mathlib_Equiv_left__inv___autoParam___closed__79, &lp_mathlib_Equiv_left__inv___autoParam___closed__79_once, _init_lp_mathlib_Equiv_left__inv___autoParam___closed__79);
return v___x_229_;
}
}
static lean_object* _init_lp_mathlib___aux__Mathlib__Logic__Equiv__Defs______macroRules__term___u2243____1___closed__4(void){
_start:
{
lean_object* v___x_262_; lean_object* v___x_263_; 
v___x_262_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Logic__Equiv__Defs______macroRules__term___u2243____1___closed__3));
v___x_263_ = l_String_toRawSubstring_x27(v___x_262_);
return v___x_263_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Logic__Equiv__Defs______macroRules__term___u2243____1(lean_object* v_x_277_, lean_object* v_a_278_, lean_object* v_a_279_){
_start:
{
lean_object* v___x_280_; uint8_t v___x_281_; 
v___x_280_ = ((lean_object*)(lp_mathlib_term___u2243___00__closed__1));
lean_inc(v_x_277_);
v___x_281_ = l_Lean_Syntax_isOfKind(v_x_277_, v___x_280_);
if (v___x_281_ == 0)
{
lean_object* v___x_282_; lean_object* v___x_283_; 
lean_dec(v_x_277_);
v___x_282_ = lean_box(1);
v___x_283_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_283_, 0, v___x_282_);
lean_ctor_set(v___x_283_, 1, v_a_279_);
return v___x_283_;
}
else
{
lean_object* v_quotContext_284_; lean_object* v_currMacroScope_285_; lean_object* v_ref_286_; lean_object* v___x_287_; lean_object* v___x_288_; lean_object* v___x_289_; lean_object* v___x_290_; uint8_t v___x_291_; lean_object* v___x_292_; lean_object* v___x_293_; lean_object* v___x_294_; lean_object* v___x_295_; lean_object* v___x_296_; lean_object* v___x_297_; lean_object* v___x_298_; lean_object* v___x_299_; lean_object* v___x_300_; lean_object* v___x_301_; lean_object* v___x_302_; 
v_quotContext_284_ = lean_ctor_get(v_a_278_, 1);
v_currMacroScope_285_ = lean_ctor_get(v_a_278_, 2);
v_ref_286_ = lean_ctor_get(v_a_278_, 5);
v___x_287_ = lean_unsigned_to_nat(0u);
v___x_288_ = l_Lean_Syntax_getArg(v_x_277_, v___x_287_);
v___x_289_ = lean_unsigned_to_nat(2u);
v___x_290_ = l_Lean_Syntax_getArg(v_x_277_, v___x_289_);
lean_dec(v_x_277_);
v___x_291_ = 0;
v___x_292_ = l_Lean_SourceInfo_fromRef(v_ref_286_, v___x_291_);
v___x_293_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Logic__Equiv__Defs______macroRules__term___u2243____1___closed__2));
v___x_294_ = lean_obj_once(&lp_mathlib___aux__Mathlib__Logic__Equiv__Defs______macroRules__term___u2243____1___closed__4, &lp_mathlib___aux__Mathlib__Logic__Equiv__Defs______macroRules__term___u2243____1___closed__4_once, _init_lp_mathlib___aux__Mathlib__Logic__Equiv__Defs______macroRules__term___u2243____1___closed__4);
v___x_295_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Logic__Equiv__Defs______macroRules__term___u2243____1___closed__5));
lean_inc(v_currMacroScope_285_);
lean_inc(v_quotContext_284_);
v___x_296_ = l_Lean_addMacroScope(v_quotContext_284_, v___x_295_, v_currMacroScope_285_);
v___x_297_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Logic__Equiv__Defs______macroRules__term___u2243____1___closed__9));
lean_inc_n(v___x_292_, 2);
v___x_298_ = lean_alloc_ctor(3, 4, 0);
lean_ctor_set(v___x_298_, 0, v___x_292_);
lean_ctor_set(v___x_298_, 1, v___x_294_);
lean_ctor_set(v___x_298_, 2, v___x_296_);
lean_ctor_set(v___x_298_, 3, v___x_297_);
v___x_299_ = ((lean_object*)(lp_mathlib_Equiv_left__inv___autoParam___closed__9));
v___x_300_ = l_Lean_Syntax_node2(v___x_292_, v___x_299_, v___x_288_, v___x_290_);
v___x_301_ = l_Lean_Syntax_node2(v___x_292_, v___x_293_, v___x_298_, v___x_300_);
v___x_302_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_302_, 0, v___x_301_);
lean_ctor_set(v___x_302_, 1, v_a_279_);
return v___x_302_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Logic__Equiv__Defs______macroRules__term___u2243____1___boxed(lean_object* v_x_303_, lean_object* v_a_304_, lean_object* v_a_305_){
_start:
{
lean_object* v_res_306_; 
v_res_306_ = lp_mathlib___aux__Mathlib__Logic__Equiv__Defs______macroRules__term___u2243____1(v_x_303_, v_a_304_, v_a_305_);
lean_dec_ref(v_a_304_);
return v_res_306_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Logic__Equiv__Defs______unexpand__Equiv__1(lean_object* v_x_310_, lean_object* v_a_311_, lean_object* v_a_312_){
_start:
{
lean_object* v___x_313_; uint8_t v___x_314_; 
v___x_313_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Logic__Equiv__Defs______macroRules__term___u2243____1___closed__2));
lean_inc(v_x_310_);
v___x_314_ = l_Lean_Syntax_isOfKind(v_x_310_, v___x_313_);
if (v___x_314_ == 0)
{
lean_object* v___x_315_; lean_object* v___x_316_; 
lean_dec(v_x_310_);
v___x_315_ = lean_box(0);
v___x_316_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_316_, 0, v___x_315_);
lean_ctor_set(v___x_316_, 1, v_a_312_);
return v___x_316_;
}
else
{
lean_object* v___x_317_; lean_object* v___x_318_; lean_object* v___x_319_; uint8_t v___x_320_; 
v___x_317_ = lean_unsigned_to_nat(0u);
v___x_318_ = l_Lean_Syntax_getArg(v_x_310_, v___x_317_);
v___x_319_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Logic__Equiv__Defs______unexpand__Equiv__1___closed__1));
lean_inc(v___x_318_);
v___x_320_ = l_Lean_Syntax_isOfKind(v___x_318_, v___x_319_);
if (v___x_320_ == 0)
{
lean_object* v___x_321_; lean_object* v___x_322_; 
lean_dec(v___x_318_);
lean_dec(v_x_310_);
v___x_321_ = lean_box(0);
v___x_322_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_322_, 0, v___x_321_);
lean_ctor_set(v___x_322_, 1, v_a_312_);
return v___x_322_;
}
else
{
lean_object* v___x_323_; lean_object* v___x_324_; lean_object* v___x_325_; uint8_t v___x_326_; 
v___x_323_ = lean_unsigned_to_nat(1u);
v___x_324_ = l_Lean_Syntax_getArg(v_x_310_, v___x_323_);
lean_dec(v_x_310_);
v___x_325_ = lean_unsigned_to_nat(2u);
lean_inc(v___x_324_);
v___x_326_ = l_Lean_Syntax_matchesNull(v___x_324_, v___x_325_);
if (v___x_326_ == 0)
{
lean_object* v___x_327_; lean_object* v___x_328_; 
lean_dec(v___x_324_);
lean_dec(v___x_318_);
v___x_327_ = lean_box(0);
v___x_328_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_328_, 0, v___x_327_);
lean_ctor_set(v___x_328_, 1, v_a_312_);
return v___x_328_;
}
else
{
lean_object* v___x_329_; lean_object* v___x_330_; lean_object* v_ref_331_; uint8_t v___x_332_; lean_object* v___x_333_; lean_object* v___x_334_; lean_object* v___x_335_; lean_object* v___x_336_; lean_object* v___x_337_; lean_object* v___x_338_; 
v___x_329_ = l_Lean_Syntax_getArg(v___x_324_, v___x_317_);
v___x_330_ = l_Lean_Syntax_getArg(v___x_324_, v___x_323_);
lean_dec(v___x_324_);
v_ref_331_ = l_Lean_replaceRef(v___x_318_, v_a_311_);
lean_dec(v___x_318_);
v___x_332_ = 0;
v___x_333_ = l_Lean_SourceInfo_fromRef(v_ref_331_, v___x_332_);
lean_dec(v_ref_331_);
v___x_334_ = ((lean_object*)(lp_mathlib_term___u2243___00__closed__1));
v___x_335_ = ((lean_object*)(lp_mathlib_term___u2243___00__closed__4));
lean_inc(v___x_333_);
v___x_336_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_336_, 0, v___x_333_);
lean_ctor_set(v___x_336_, 1, v___x_335_);
v___x_337_ = l_Lean_Syntax_node3(v___x_333_, v___x_334_, v___x_329_, v___x_336_, v___x_330_);
v___x_338_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_338_, 0, v___x_337_);
lean_ctor_set(v___x_338_, 1, v_a_312_);
return v___x_338_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Logic__Equiv__Defs______unexpand__Equiv__1___boxed(lean_object* v_x_339_, lean_object* v_a_340_, lean_object* v_a_341_){
_start:
{
lean_object* v_res_342_; 
v_res_342_ = lp_mathlib___aux__Mathlib__Logic__Equiv__Defs______unexpand__Equiv__1(v_x_339_, v_a_340_, v_a_341_);
lean_dec(v_a_340_);
return v_res_342_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_EquivLike_toEquiv___redArg(lean_object* v_inst_343_, lean_object* v_f_344_){
_start:
{
lean_object* v_coe_345_; lean_object* v_inv_346_; lean_object* v___x_348_; uint8_t v_isShared_349_; uint8_t v_isSharedCheck_355_; 
v_coe_345_ = lean_ctor_get(v_inst_343_, 0);
v_inv_346_ = lean_ctor_get(v_inst_343_, 1);
v_isSharedCheck_355_ = !lean_is_exclusive(v_inst_343_);
if (v_isSharedCheck_355_ == 0)
{
v___x_348_ = v_inst_343_;
v_isShared_349_ = v_isSharedCheck_355_;
goto v_resetjp_347_;
}
else
{
lean_inc(v_inv_346_);
lean_inc(v_coe_345_);
lean_dec(v_inst_343_);
v___x_348_ = lean_box(0);
v_isShared_349_ = v_isSharedCheck_355_;
goto v_resetjp_347_;
}
v_resetjp_347_:
{
lean_object* v___x_350_; lean_object* v___x_351_; lean_object* v___x_353_; 
lean_inc(v_f_344_);
v___x_350_ = lean_apply_1(v_coe_345_, v_f_344_);
v___x_351_ = lean_apply_1(v_inv_346_, v_f_344_);
if (v_isShared_349_ == 0)
{
lean_ctor_set(v___x_348_, 1, v___x_351_);
lean_ctor_set(v___x_348_, 0, v___x_350_);
v___x_353_ = v___x_348_;
goto v_reusejp_352_;
}
else
{
lean_object* v_reuseFailAlloc_354_; 
v_reuseFailAlloc_354_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_354_, 0, v___x_350_);
lean_ctor_set(v_reuseFailAlloc_354_, 1, v___x_351_);
v___x_353_ = v_reuseFailAlloc_354_;
goto v_reusejp_352_;
}
v_reusejp_352_:
{
return v___x_353_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_EquivLike_toEquiv(lean_object* v_00_u03b1_356_, lean_object* v_00_u03b2_357_, lean_object* v_F_358_, lean_object* v_inst_359_, lean_object* v_f_360_){
_start:
{
lean_object* v___x_361_; 
v___x_361_ = lp_mathlib_EquivLike_toEquiv___redArg(v_inst_359_, v_f_360_);
return v___x_361_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instCoeTCEquivOfEquivLike___redArg(lean_object* v_inst_362_){
_start:
{
lean_object* v___x_363_; 
v___x_363_ = lean_alloc_closure((void*)(lp_mathlib_EquivLike_toEquiv), 5, 4);
lean_closure_set(v___x_363_, 0, lean_box(0));
lean_closure_set(v___x_363_, 1, lean_box(0));
lean_closure_set(v___x_363_, 2, lean_box(0));
lean_closure_set(v___x_363_, 3, v_inst_362_);
return v___x_363_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instCoeTCEquivOfEquivLike(lean_object* v_00_u03b1_364_, lean_object* v_00_u03b2_365_, lean_object* v_F_366_, lean_object* v_inst_367_){
_start:
{
lean_object* v___x_368_; 
v___x_368_ = lean_alloc_closure((void*)(lp_mathlib_EquivLike_toEquiv), 5, 4);
lean_closure_set(v___x_368_, 0, lean_box(0));
lean_closure_set(v___x_368_, 1, lean_box(0));
lean_closure_set(v___x_368_, 2, lean_box(0));
lean_closure_set(v___x_368_, 3, v_inst_367_);
return v___x_368_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_refl___lam__0(lean_object* v___y_369_){
_start:
{
lean_inc(v___y_369_);
return v___y_369_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_refl___lam__0___boxed(lean_object* v___y_370_){
_start:
{
lean_object* v_res_371_; 
v_res_371_ = lp_mathlib_Equiv_refl___lam__0(v___y_370_);
lean_dec(v___y_370_);
return v_res_371_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_refl(lean_object* v_00_u03b1_375_){
_start:
{
lean_object* v___x_376_; 
v___x_376_ = ((lean_object*)(lp_mathlib_Equiv_refl___closed__1));
return v___x_376_;
}
}
static lean_object* _init_lp_mathlib_Equiv_inhabited_x27___closed__0(void){
_start:
{
lean_object* v___x_377_; 
v___x_377_ = lp_mathlib_Equiv_refl(lean_box(0));
return v___x_377_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_inhabited_x27(lean_object* v_00_u03b1_378_){
_start:
{
lean_object* v___x_379_; 
v___x_379_ = lean_obj_once(&lp_mathlib_Equiv_inhabited_x27___closed__0, &lp_mathlib_Equiv_inhabited_x27___closed__0_once, _init_lp_mathlib_Equiv_inhabited_x27___closed__0);
return v___x_379_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_symm___redArg(lean_object* v_e_380_){
_start:
{
lean_object* v_toFun_381_; lean_object* v_invFun_382_; lean_object* v___x_384_; uint8_t v_isShared_385_; uint8_t v_isSharedCheck_389_; 
v_toFun_381_ = lean_ctor_get(v_e_380_, 0);
v_invFun_382_ = lean_ctor_get(v_e_380_, 1);
v_isSharedCheck_389_ = !lean_is_exclusive(v_e_380_);
if (v_isSharedCheck_389_ == 0)
{
v___x_384_ = v_e_380_;
v_isShared_385_ = v_isSharedCheck_389_;
goto v_resetjp_383_;
}
else
{
lean_inc(v_invFun_382_);
lean_inc(v_toFun_381_);
lean_dec(v_e_380_);
v___x_384_ = lean_box(0);
v_isShared_385_ = v_isSharedCheck_389_;
goto v_resetjp_383_;
}
v_resetjp_383_:
{
lean_object* v___x_387_; 
if (v_isShared_385_ == 0)
{
lean_ctor_set(v___x_384_, 1, v_toFun_381_);
lean_ctor_set(v___x_384_, 0, v_invFun_382_);
v___x_387_ = v___x_384_;
goto v_reusejp_386_;
}
else
{
lean_object* v_reuseFailAlloc_388_; 
v_reuseFailAlloc_388_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_388_, 0, v_invFun_382_);
lean_ctor_set(v_reuseFailAlloc_388_, 1, v_toFun_381_);
v___x_387_ = v_reuseFailAlloc_388_;
goto v_reusejp_386_;
}
v_reusejp_386_:
{
return v___x_387_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_symm(lean_object* v_00_u03b1_390_, lean_object* v_00_u03b2_391_, lean_object* v_e_392_){
_start:
{
lean_object* v___x_393_; 
v___x_393_ = lp_mathlib_Equiv_symm___redArg(v_e_392_);
return v___x_393_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_Simps_symm__apply___redArg(lean_object* v_e_394_, lean_object* v_a_395_){
_start:
{
lean_object* v___x_396_; lean_object* v_toFun_397_; lean_object* v___x_398_; 
v___x_396_ = lp_mathlib_Equiv_symm___redArg(v_e_394_);
v_toFun_397_ = lean_ctor_get(v___x_396_, 0);
lean_inc(v_toFun_397_);
lean_dec_ref(v___x_396_);
v___x_398_ = lean_apply_1(v_toFun_397_, v_a_395_);
return v___x_398_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_Simps_symm__apply(lean_object* v_00_u03b1_399_, lean_object* v_00_u03b2_400_, lean_object* v_e_401_, lean_object* v_a_402_){
_start:
{
lean_object* v___x_403_; 
v___x_403_ = lp_mathlib_Equiv_Simps_symm__apply___redArg(v_e_401_, v_a_402_);
return v___x_403_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_trans___redArg___lam__0(lean_object* v_e_u2081_404_, lean_object* v_e_u2082_405_, lean_object* v___y_406_){
_start:
{
lean_object* v_toFun_407_; lean_object* v_toFun_408_; lean_object* v___x_409_; lean_object* v___x_410_; 
v_toFun_407_ = lean_ctor_get(v_e_u2081_404_, 0);
lean_inc(v_toFun_407_);
lean_dec_ref(v_e_u2081_404_);
v_toFun_408_ = lean_ctor_get(v_e_u2082_405_, 0);
lean_inc(v_toFun_408_);
lean_dec_ref(v_e_u2082_405_);
v___x_409_ = lean_apply_1(v_toFun_407_, v___y_406_);
v___x_410_ = lean_apply_1(v_toFun_408_, v___x_409_);
return v___x_410_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_trans___redArg___lam__1(lean_object* v___x_411_, lean_object* v___x_412_, lean_object* v___y_413_){
_start:
{
lean_object* v_toFun_414_; lean_object* v_toFun_415_; lean_object* v___x_416_; lean_object* v___x_417_; 
v_toFun_414_ = lean_ctor_get(v___x_411_, 0);
lean_inc(v_toFun_414_);
lean_dec_ref(v___x_411_);
v_toFun_415_ = lean_ctor_get(v___x_412_, 0);
lean_inc(v_toFun_415_);
lean_dec_ref(v___x_412_);
v___x_416_ = lean_apply_1(v_toFun_414_, v___y_413_);
v___x_417_ = lean_apply_1(v_toFun_415_, v___x_416_);
return v___x_417_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_trans___redArg(lean_object* v_e_u2081_418_, lean_object* v_e_u2082_419_){
_start:
{
lean_object* v___f_420_; lean_object* v___x_421_; lean_object* v___x_422_; lean_object* v___f_423_; lean_object* v___x_424_; 
lean_inc_ref(v_e_u2082_419_);
lean_inc_ref(v_e_u2081_418_);
v___f_420_ = lean_alloc_closure((void*)(lp_mathlib_Equiv_trans___redArg___lam__0), 3, 2);
lean_closure_set(v___f_420_, 0, v_e_u2081_418_);
lean_closure_set(v___f_420_, 1, v_e_u2082_419_);
v___x_421_ = lp_mathlib_Equiv_symm___redArg(v_e_u2081_418_);
v___x_422_ = lp_mathlib_Equiv_symm___redArg(v_e_u2082_419_);
v___f_423_ = lean_alloc_closure((void*)(lp_mathlib_Equiv_trans___redArg___lam__1), 3, 2);
lean_closure_set(v___f_423_, 0, v___x_422_);
lean_closure_set(v___f_423_, 1, v___x_421_);
v___x_424_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_424_, 0, v___f_420_);
lean_ctor_set(v___x_424_, 1, v___f_423_);
return v___x_424_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_trans(lean_object* v_00_u03b1_425_, lean_object* v_00_u03b2_426_, lean_object* v_00_u03b3_427_, lean_object* v_e_u2081_428_, lean_object* v_e_u2082_429_){
_start:
{
lean_object* v___x_430_; 
v___x_430_ = lp_mathlib_Equiv_trans___redArg(v_e_u2081_428_, v_e_u2082_429_);
return v___x_430_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_symmEquiv(lean_object* v_00_u03b1_436_, lean_object* v_00_u03b2_437_){
_start:
{
lean_object* v___x_438_; 
v___x_438_ = ((lean_object*)(lp_mathlib_Equiv_symmEquiv___closed__1));
return v___x_438_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_permUnique(lean_object* v_00_u03b1_439_, lean_object* v_inst_440_){
_start:
{
lean_object* v___x_441_; 
v___x_441_ = lean_obj_once(&lp_mathlib_Equiv_inhabited_x27___closed__0, &lp_mathlib_Equiv_inhabited_x27___closed__0_once, _init_lp_mathlib_Equiv_inhabited_x27___closed__0);
return v___x_441_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_decidableEq___redArg___lam__0(lean_object* v_e_442_, lean_object* v___y_443_){
_start:
{
lean_object* v_toFun_444_; lean_object* v___x_445_; 
v_toFun_444_ = lean_ctor_get(v_e_442_, 0);
lean_inc(v_toFun_444_);
lean_dec_ref(v_e_442_);
v___x_445_ = lean_apply_1(v_toFun_444_, v___y_443_);
return v___x_445_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Equiv_decidableEq___redArg(lean_object* v_e_446_, lean_object* v_inst_447_, lean_object* v_a_448_, lean_object* v_b_449_){
_start:
{
lean_object* v___f_450_; uint8_t v___x_451_; 
v___f_450_ = lean_alloc_closure((void*)(lp_mathlib_Equiv_decidableEq___redArg___lam__0), 2, 1);
lean_closure_set(v___f_450_, 0, v_e_446_);
v___x_451_ = lp_mathlib_Function_Injective_decidableEq___redArg(v___f_450_, v_inst_447_, v_a_448_, v_b_449_);
return v___x_451_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_decidableEq___redArg___boxed(lean_object* v_e_452_, lean_object* v_inst_453_, lean_object* v_a_454_, lean_object* v_b_455_){
_start:
{
uint8_t v_res_456_; lean_object* v_r_457_; 
v_res_456_ = lp_mathlib_Equiv_decidableEq___redArg(v_e_452_, v_inst_453_, v_a_454_, v_b_455_);
v_r_457_ = lean_box(v_res_456_);
return v_r_457_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Equiv_decidableEq(lean_object* v_00_u03b1_458_, lean_object* v_00_u03b2_459_, lean_object* v_e_460_, lean_object* v_inst_461_, lean_object* v_a_462_, lean_object* v_b_463_){
_start:
{
lean_object* v___f_464_; uint8_t v___x_465_; 
v___f_464_ = lean_alloc_closure((void*)(lp_mathlib_Equiv_decidableEq___redArg___lam__0), 2, 1);
lean_closure_set(v___f_464_, 0, v_e_460_);
v___x_465_ = lp_mathlib_Function_Injective_decidableEq___redArg(v___f_464_, v_inst_461_, v_a_462_, v_b_463_);
return v___x_465_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_decidableEq___boxed(lean_object* v_00_u03b1_466_, lean_object* v_00_u03b2_467_, lean_object* v_e_468_, lean_object* v_inst_469_, lean_object* v_a_470_, lean_object* v_b_471_){
_start:
{
uint8_t v_res_472_; lean_object* v_r_473_; 
v_res_472_ = lp_mathlib_Equiv_decidableEq(v_00_u03b1_466_, v_00_u03b2_467_, v_e_468_, v_inst_469_, v_a_470_, v_b_471_);
v_r_473_ = lean_box(v_res_472_);
return v_r_473_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_inhabited___redArg(lean_object* v_inst_474_, lean_object* v_e_475_){
_start:
{
lean_object* v___x_476_; lean_object* v_toFun_477_; lean_object* v___x_478_; 
v___x_476_ = lp_mathlib_Equiv_symm___redArg(v_e_475_);
v_toFun_477_ = lean_ctor_get(v___x_476_, 0);
lean_inc(v_toFun_477_);
lean_dec_ref(v___x_476_);
v___x_478_ = lean_apply_1(v_toFun_477_, v_inst_474_);
return v___x_478_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_inhabited(lean_object* v_00_u03b1_479_, lean_object* v_00_u03b2_480_, lean_object* v_inst_481_, lean_object* v_e_482_){
_start:
{
lean_object* v___x_483_; lean_object* v_toFun_484_; lean_object* v___x_485_; 
v___x_483_ = lp_mathlib_Equiv_symm___redArg(v_e_482_);
v_toFun_484_ = lean_ctor_get(v___x_483_, 0);
lean_inc(v_toFun_484_);
lean_dec_ref(v___x_483_);
v___x_485_ = lean_apply_1(v_toFun_484_, v_inst_481_);
return v___x_485_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_unique___redArg(lean_object* v_inst_486_, lean_object* v_e_487_){
_start:
{
lean_object* v___x_488_; lean_object* v_toFun_489_; lean_object* v___x_490_; 
v___x_488_ = lp_mathlib_Equiv_symm___redArg(v_e_487_);
v_toFun_489_ = lean_ctor_get(v___x_488_, 0);
lean_inc(v_toFun_489_);
lean_dec_ref(v___x_488_);
v___x_490_ = lean_apply_1(v_toFun_489_, v_inst_486_);
return v___x_490_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_unique(lean_object* v_00_u03b1_491_, lean_object* v_00_u03b2_492_, lean_object* v_inst_493_, lean_object* v_e_494_){
_start:
{
lean_object* v___x_495_; lean_object* v_toFun_496_; lean_object* v___x_497_; 
v___x_495_ = lp_mathlib_Equiv_symm___redArg(v_e_494_);
v_toFun_496_ = lean_ctor_get(v___x_495_, 0);
lean_inc(v_toFun_496_);
lean_dec_ref(v___x_495_);
v___x_497_ = lean_apply_1(v_toFun_496_, v_inst_493_);
return v___x_497_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_cast___lam__0(lean_object* v_a_498_){
_start:
{
lean_inc(v_a_498_);
return v_a_498_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_cast___lam__0___boxed(lean_object* v_a_499_){
_start:
{
lean_object* v_res_500_; 
v_res_500_ = lp_mathlib_Equiv_cast___lam__0(v_a_499_);
lean_dec(v_a_499_);
return v_res_500_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_cast(lean_object* v_00_u03b1_504_, lean_object* v_00_u03b2_505_, lean_object* v_h_506_){
_start:
{
lean_object* v___x_507_; 
v___x_507_ = ((lean_object*)(lp_mathlib_Equiv_cast___closed__1));
return v___x_507_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_equivCongr___redArg___lam__0(lean_object* v_ab_508_, lean_object* v_cd_509_, lean_object* v_ac_510_){
_start:
{
lean_object* v___x_511_; lean_object* v___x_512_; lean_object* v___x_513_; 
v___x_511_ = lp_mathlib_Equiv_symm___redArg(v_ab_508_);
v___x_512_ = lp_mathlib_Equiv_trans___redArg(v___x_511_, v_ac_510_);
v___x_513_ = lp_mathlib_Equiv_trans___redArg(v___x_512_, v_cd_509_);
return v___x_513_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_equivCongr___redArg___lam__1(lean_object* v_cd_514_, lean_object* v_ab_515_, lean_object* v_bd_516_){
_start:
{
lean_object* v___x_517_; lean_object* v___x_518_; lean_object* v___x_519_; 
v___x_517_ = lp_mathlib_Equiv_symm___redArg(v_cd_514_);
v___x_518_ = lp_mathlib_Equiv_trans___redArg(v_bd_516_, v___x_517_);
v___x_519_ = lp_mathlib_Equiv_trans___redArg(v_ab_515_, v___x_518_);
return v___x_519_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_equivCongr___redArg(lean_object* v_ab_520_, lean_object* v_cd_521_){
_start:
{
lean_object* v___f_522_; lean_object* v___f_523_; lean_object* v___x_524_; 
lean_inc_ref(v_cd_521_);
lean_inc_ref(v_ab_520_);
v___f_522_ = lean_alloc_closure((void*)(lp_mathlib_Equiv_equivCongr___redArg___lam__0), 3, 2);
lean_closure_set(v___f_522_, 0, v_ab_520_);
lean_closure_set(v___f_522_, 1, v_cd_521_);
v___f_523_ = lean_alloc_closure((void*)(lp_mathlib_Equiv_equivCongr___redArg___lam__1), 3, 2);
lean_closure_set(v___f_523_, 0, v_cd_521_);
lean_closure_set(v___f_523_, 1, v_ab_520_);
v___x_524_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_524_, 0, v___f_522_);
lean_ctor_set(v___x_524_, 1, v___f_523_);
return v___x_524_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_equivCongr(lean_object* v_00_u03b1_525_, lean_object* v_00_u03b2_526_, lean_object* v_00_u03b3_527_, lean_object* v_00_u03b4_528_, lean_object* v_ab_529_, lean_object* v_cd_530_){
_start:
{
lean_object* v___x_531_; 
v___x_531_ = lp_mathlib_Equiv_equivCongr___redArg(v_ab_529_, v_cd_530_);
return v___x_531_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_permCongr___redArg(lean_object* v_e_532_){
_start:
{
lean_object* v___x_533_; 
lean_inc_ref(v_e_532_);
v___x_533_ = lp_mathlib_Equiv_equivCongr___redArg(v_e_532_, v_e_532_);
return v___x_533_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_permCongr(lean_object* v_00_u03b1_x27_534_, lean_object* v_00_u03b2_x27_535_, lean_object* v_e_536_){
_start:
{
lean_object* v___x_537_; 
lean_inc_ref(v_e_536_);
v___x_537_ = lp_mathlib_Equiv_equivCongr___redArg(v_e_536_, v_e_536_);
return v___x_537_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_equivOfIsEmpty___lam__0(lean_object* v_a_538_){
_start:
{
lean_internal_panic_unreachable();
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_equivOfIsEmpty___lam__0___boxed(lean_object* v_a_539_){
_start:
{
lean_object* v_res_540_; 
v_res_540_ = lp_mathlib_Equiv_equivOfIsEmpty___lam__0(v_a_539_);
lean_dec(v_a_539_);
return v_res_540_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_equivOfIsEmpty(lean_object* v_00_u03b1_544_, lean_object* v_00_u03b2_545_, lean_object* v_inst_546_, lean_object* v_inst_547_){
_start:
{
lean_object* v___x_548_; 
v___x_548_ = ((lean_object*)(lp_mathlib_Equiv_equivOfIsEmpty___closed__1));
return v___x_548_;
}
}
static lean_object* _init_lp_mathlib_Equiv_equivEmpty___closed__0(void){
_start:
{
lean_object* v___x_549_; 
v___x_549_ = lp_mathlib_Equiv_equivOfIsEmpty(lean_box(0), lean_box(0), lean_box(0), lean_box(0));
return v___x_549_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_equivEmpty(lean_object* v_00_u03b1_550_, lean_object* v_inst_551_){
_start:
{
lean_object* v___x_552_; 
v___x_552_ = lean_obj_once(&lp_mathlib_Equiv_equivEmpty___closed__0, &lp_mathlib_Equiv_equivEmpty___closed__0_once, _init_lp_mathlib_Equiv_equivEmpty___closed__0);
return v___x_552_;
}
}
static lean_object* _init_lp_mathlib_Equiv_equivPEmpty___closed__0(void){
_start:
{
lean_object* v___x_553_; 
v___x_553_ = lp_mathlib_Equiv_equivOfIsEmpty(lean_box(0), lean_box(0), lean_box(0), lean_box(0));
return v___x_553_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_equivPEmpty(lean_object* v_00_u03b1_554_, lean_object* v_inst_555_){
_start:
{
lean_object* v___x_556_; 
v___x_556_ = lean_obj_once(&lp_mathlib_Equiv_equivPEmpty___closed__0, &lp_mathlib_Equiv_equivPEmpty___closed__0_once, _init_lp_mathlib_Equiv_equivPEmpty___closed__0);
return v___x_556_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_equivEmptyEquiv___lam__0(lean_object* v___y_557_){
_start:
{
lean_object* v___x_558_; 
v___x_558_ = lean_obj_once(&lp_mathlib_Equiv_equivEmpty___closed__0, &lp_mathlib_Equiv_equivEmpty___closed__0_once, _init_lp_mathlib_Equiv_equivEmpty___closed__0);
return v___x_558_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_equivEmptyEquiv(lean_object* v_00_u03b1_562_){
_start:
{
lean_object* v___x_563_; 
v___x_563_ = ((lean_object*)(lp_mathlib_Equiv_equivEmptyEquiv___closed__1));
return v___x_563_;
}
}
static lean_object* _init_lp_mathlib_Equiv_propEquivPEmpty___closed__0(void){
_start:
{
lean_object* v___x_564_; 
v___x_564_ = lp_mathlib_Equiv_equivOfIsEmpty(lean_box(0), lean_box(0), lean_box(0), lean_box(0));
return v___x_564_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_propEquivPEmpty(lean_object* v_p_565_, lean_object* v_h_566_){
_start:
{
lean_object* v___x_567_; 
v___x_567_ = lean_obj_once(&lp_mathlib_Equiv_propEquivPEmpty___closed__0, &lp_mathlib_Equiv_propEquivPEmpty___closed__0_once, _init_lp_mathlib_Equiv_propEquivPEmpty___closed__0);
return v___x_567_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_ofUnique___redArg___lam__0(lean_object* v_inst_568_, lean_object* v_x_569_){
_start:
{
lean_inc(v_inst_568_);
return v_inst_568_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_ofUnique___redArg___lam__0___boxed(lean_object* v_inst_570_, lean_object* v_x_571_){
_start:
{
lean_object* v_res_572_; 
v_res_572_ = lp_mathlib_Equiv_ofUnique___redArg___lam__0(v_inst_570_, v_x_571_);
lean_dec(v_x_571_);
lean_dec(v_inst_570_);
return v_res_572_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_ofUnique___redArg(lean_object* v_inst_573_, lean_object* v_inst_574_){
_start:
{
lean_object* v___f_575_; lean_object* v___f_576_; lean_object* v___x_577_; 
v___f_575_ = lean_alloc_closure((void*)(lp_mathlib_Equiv_ofUnique___redArg___lam__0___boxed), 2, 1);
lean_closure_set(v___f_575_, 0, v_inst_574_);
v___f_576_ = lean_alloc_closure((void*)(lp_mathlib_Equiv_ofUnique___redArg___lam__0___boxed), 2, 1);
lean_closure_set(v___f_576_, 0, v_inst_573_);
v___x_577_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_577_, 0, v___f_575_);
lean_ctor_set(v___x_577_, 1, v___f_576_);
return v___x_577_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_ofUnique(lean_object* v_00_u03b1_578_, lean_object* v_00_u03b2_579_, lean_object* v_inst_580_, lean_object* v_inst_581_){
_start:
{
lean_object* v___x_582_; 
v___x_582_ = lp_mathlib_Equiv_ofUnique___redArg(v_inst_580_, v_inst_581_);
return v___x_582_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_equivPUnit___redArg(lean_object* v_inst_583_){
_start:
{
lean_object* v___x_584_; lean_object* v___x_585_; 
v___x_584_ = lean_box(0);
v___x_585_ = lp_mathlib_Equiv_ofUnique___redArg(v_inst_583_, v___x_584_);
return v___x_585_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_equivPUnit(lean_object* v_00_u03b1_586_, lean_object* v_inst_587_){
_start:
{
lean_object* v___x_588_; 
v___x_588_ = lp_mathlib_Equiv_equivPUnit___redArg(v_inst_587_);
return v___x_588_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_ofUnique___at___00Equiv_equivPUnit___at___00Equiv_propEquivPUnit_spec__0_spec__0___lam__1(lean_object* v_x_589_){
_start:
{
lean_object* v___x_590_; 
v___x_590_ = lean_box(0);
return v___x_590_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_ofUnique___at___00Equiv_equivPUnit___at___00Equiv_propEquivPUnit_spec__0_spec__0___lam__0(lean_object* v_x_591_){
_start:
{
lean_object* v___x_592_; 
v___x_592_ = lean_box(0);
return v___x_592_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_propEquivPUnit(lean_object* v_p_597_, lean_object* v_h_598_){
_start:
{
lean_object* v___x_599_; 
v___x_599_ = ((lean_object*)(lp_mathlib_Equiv_ofUnique___at___00Equiv_equivPUnit___at___00Equiv_propEquivPUnit_spec__0_spec__0));
return v___x_599_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_ulift___lam__0(lean_object* v_self_601_){
_start:
{
lean_inc(v_self_601_);
return v_self_601_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_ulift___lam__0___boxed(lean_object* v_self_602_){
_start:
{
lean_object* v_res_603_; 
v_res_603_ = lp_mathlib_Equiv_ulift___lam__0(v_self_602_);
lean_dec(v_self_602_);
return v_res_603_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_ulift___lam__1(lean_object* v_down_604_){
_start:
{
lean_inc(v_down_604_);
return v_down_604_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_ulift___lam__1___boxed(lean_object* v_down_605_){
_start:
{
lean_object* v_res_606_; 
v_res_606_ = lp_mathlib_Equiv_ulift___lam__1(v_down_605_);
lean_dec(v_down_605_);
return v_res_606_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_ulift(lean_object* v_00_u03b1_612_){
_start:
{
lean_object* v___x_613_; 
v___x_613_ = ((lean_object*)(lp_mathlib_Equiv_ulift___closed__2));
return v___x_613_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_plift(lean_object* v_00_u03b1_614_){
_start:
{
lean_object* v___x_615_; 
v___x_615_ = ((lean_object*)(lp_mathlib_Equiv_ulift___closed__2));
return v___x_615_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_ofIff(lean_object* v_P_616_, lean_object* v_Q_617_, lean_object* v_h_618_){
_start:
{
lean_object* v___x_619_; 
v___x_619_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_619_, 0, lean_box(0));
lean_ctor_set(v___x_619_, 1, lean_box(0));
return v___x_619_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_arrowCongr___redArg___lam__0(lean_object* v_e_u2081_620_, lean_object* v_e_u2082_621_, lean_object* v_f_622_, lean_object* v___y_623_){
_start:
{
lean_object* v___x_624_; lean_object* v_toFun_625_; lean_object* v_toFun_626_; lean_object* v___x_627_; lean_object* v___x_628_; lean_object* v___x_629_; 
v___x_624_ = lp_mathlib_Equiv_symm___redArg(v_e_u2081_620_);
v_toFun_625_ = lean_ctor_get(v___x_624_, 0);
lean_inc(v_toFun_625_);
lean_dec_ref(v___x_624_);
v_toFun_626_ = lean_ctor_get(v_e_u2082_621_, 0);
lean_inc(v_toFun_626_);
lean_dec_ref(v_e_u2082_621_);
v___x_627_ = lean_apply_1(v_toFun_625_, v___y_623_);
v___x_628_ = lean_apply_1(v_f_622_, v___x_627_);
v___x_629_ = lean_apply_1(v_toFun_626_, v___x_628_);
return v___x_629_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_arrowCongr___redArg___lam__1(lean_object* v_e_u2081_630_, lean_object* v_e_u2082_631_, lean_object* v_f_632_, lean_object* v___y_633_){
_start:
{
lean_object* v_toFun_634_; lean_object* v___x_635_; lean_object* v_toFun_636_; lean_object* v___x_637_; lean_object* v___x_638_; lean_object* v___x_639_; 
v_toFun_634_ = lean_ctor_get(v_e_u2081_630_, 0);
lean_inc(v_toFun_634_);
lean_dec_ref(v_e_u2081_630_);
v___x_635_ = lp_mathlib_Equiv_symm___redArg(v_e_u2082_631_);
v_toFun_636_ = lean_ctor_get(v___x_635_, 0);
lean_inc(v_toFun_636_);
lean_dec_ref(v___x_635_);
v___x_637_ = lean_apply_1(v_toFun_634_, v___y_633_);
v___x_638_ = lean_apply_1(v_f_632_, v___x_637_);
v___x_639_ = lean_apply_1(v_toFun_636_, v___x_638_);
return v___x_639_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_arrowCongr___redArg(lean_object* v_e_u2081_640_, lean_object* v_e_u2082_641_){
_start:
{
lean_object* v___f_642_; lean_object* v___f_643_; lean_object* v___x_644_; 
lean_inc_ref(v_e_u2082_641_);
lean_inc_ref(v_e_u2081_640_);
v___f_642_ = lean_alloc_closure((void*)(lp_mathlib_Equiv_arrowCongr___redArg___lam__0), 4, 2);
lean_closure_set(v___f_642_, 0, v_e_u2081_640_);
lean_closure_set(v___f_642_, 1, v_e_u2082_641_);
v___f_643_ = lean_alloc_closure((void*)(lp_mathlib_Equiv_arrowCongr___redArg___lam__1), 4, 2);
lean_closure_set(v___f_643_, 0, v_e_u2081_640_);
lean_closure_set(v___f_643_, 1, v_e_u2082_641_);
v___x_644_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_644_, 0, v___f_642_);
lean_ctor_set(v___x_644_, 1, v___f_643_);
return v___x_644_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_arrowCongr(lean_object* v_00_u03b1_u2081_645_, lean_object* v_00_u03b2_u2081_646_, lean_object* v_00_u03b1_u2082_647_, lean_object* v_00_u03b2_u2082_648_, lean_object* v_e_u2081_649_, lean_object* v_e_u2082_650_){
_start:
{
lean_object* v___x_651_; 
v___x_651_ = lp_mathlib_Equiv_arrowCongr___redArg(v_e_u2081_649_, v_e_u2082_650_);
return v___x_651_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_arrowCongr_x27___redArg(lean_object* v_h_u03b1_652_, lean_object* v_h_u03b2_653_){
_start:
{
lean_object* v___x_654_; 
v___x_654_ = lp_mathlib_Equiv_arrowCongr___redArg(v_h_u03b1_652_, v_h_u03b2_653_);
return v___x_654_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_arrowCongr_x27(lean_object* v_00_u03b1_u2081_655_, lean_object* v_00_u03b2_u2081_656_, lean_object* v_00_u03b1_u2082_657_, lean_object* v_00_u03b2_u2082_658_, lean_object* v_h_u03b1_659_, lean_object* v_h_u03b2_660_){
_start:
{
lean_object* v___x_661_; 
v___x_661_ = lp_mathlib_Equiv_arrowCongr___redArg(v_h_u03b1_659_, v_h_u03b2_660_);
return v___x_661_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_conj___redArg(lean_object* v_e_662_){
_start:
{
lean_object* v___x_663_; 
lean_inc_ref(v_e_662_);
v___x_663_ = lp_mathlib_Equiv_arrowCongr___redArg(v_e_662_, v_e_662_);
return v___x_663_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_conj(lean_object* v_00_u03b1_664_, lean_object* v_00_u03b2_665_, lean_object* v_e_666_){
_start:
{
lean_object* v___x_667_; 
lean_inc_ref(v_e_666_);
v___x_667_ = lp_mathlib_Equiv_arrowCongr___redArg(v_e_666_, v_e_666_);
return v___x_667_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_punitEquivPUnit___lam__0(lean_object* v_x_668_){
_start:
{
lean_object* v___x_669_; 
v___x_669_ = lean_box(0);
return v___x_669_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_arrowPUnitEquivPUnit___lam__0(lean_object* v_x_674_){
_start:
{
lean_object* v___x_675_; 
v___x_675_ = lean_box(0);
return v___x_675_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_arrowPUnitEquivPUnit___lam__0___boxed(lean_object* v_x_676_){
_start:
{
lean_object* v_res_677_; 
v_res_677_ = lp_mathlib_Equiv_arrowPUnitEquivPUnit___lam__0(v_x_676_);
lean_dec_ref(v_x_676_);
return v_res_677_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_arrowPUnitEquivPUnit___lam__1(lean_object* v_x_678_, lean_object* v_x_679_){
_start:
{
lean_object* v___x_680_; 
v___x_680_ = lean_box(0);
return v___x_680_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_arrowPUnitEquivPUnit___lam__1___boxed(lean_object* v_x_681_, lean_object* v_x_682_){
_start:
{
lean_object* v_res_683_; 
v_res_683_ = lp_mathlib_Equiv_arrowPUnitEquivPUnit___lam__1(v_x_681_, v_x_682_);
lean_dec(v_x_682_);
return v_res_683_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_arrowPUnitEquivPUnit(lean_object* v_00_u03b1_689_){
_start:
{
lean_object* v___x_690_; 
v___x_690_ = ((lean_object*)(lp_mathlib_Equiv_arrowPUnitEquivPUnit___closed__2));
return v___x_690_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_piUnique___redArg___lam__0(lean_object* v_inst_691_, lean_object* v_f_692_){
_start:
{
lean_object* v___x_693_; 
v___x_693_ = lean_apply_1(v_f_692_, v_inst_691_);
return v___x_693_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_piUnique___redArg(lean_object* v_inst_694_){
_start:
{
lean_object* v___f_695_; lean_object* v___x_696_; lean_object* v___x_697_; 
lean_inc(v_inst_694_);
v___f_695_ = lean_alloc_closure((void*)(lp_mathlib_Equiv_piUnique___redArg___lam__0), 2, 1);
lean_closure_set(v___f_695_, 0, v_inst_694_);
v___x_696_ = lean_alloc_closure((void*)(lp_mathlib_uniqueElim___boxed), 5, 3);
lean_closure_set(v___x_696_, 0, lean_box(0));
lean_closure_set(v___x_696_, 1, lean_box(0));
lean_closure_set(v___x_696_, 2, v_inst_694_);
v___x_697_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_697_, 0, v___f_695_);
lean_ctor_set(v___x_697_, 1, v___x_696_);
return v___x_697_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_piUnique(lean_object* v_00_u03b1_698_, lean_object* v_inst_699_, lean_object* v_00_u03b2_700_){
_start:
{
lean_object* v___x_701_; 
v___x_701_ = lp_mathlib_Equiv_piUnique___redArg(v_inst_699_);
return v___x_701_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_funUnique___redArg(lean_object* v_inst_702_){
_start:
{
lean_object* v___x_703_; 
v___x_703_ = lp_mathlib_Equiv_piUnique___redArg(v_inst_702_);
return v___x_703_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_funUnique(lean_object* v_00_u03b1_704_, lean_object* v_00_u03b2_705_, lean_object* v_inst_706_){
_start:
{
lean_object* v___x_707_; 
v___x_707_ = lp_mathlib_Equiv_piUnique___redArg(v_inst_706_);
return v___x_707_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_piUnique___at___00Equiv_punitArrowEquiv_spec__0___lam__0(lean_object* v_f_708_){
_start:
{
lean_object* v___x_709_; lean_object* v___x_710_; 
v___x_709_ = lean_box(0);
v___x_710_ = lean_apply_1(v_f_708_, v___x_709_);
return v___x_710_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_piUnique___at___00Equiv_punitArrowEquiv_spec__0___lam__1(lean_object* v___y_711_, lean_object* v___y_712_){
_start:
{
lean_inc(v___y_711_);
return v___y_711_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_piUnique___at___00Equiv_punitArrowEquiv_spec__0___lam__1___boxed(lean_object* v___y_713_, lean_object* v___y_714_){
_start:
{
lean_object* v_res_715_; 
v_res_715_ = lp_mathlib_Equiv_piUnique___at___00Equiv_punitArrowEquiv_spec__0___lam__1(v___y_713_, v___y_714_);
lean_dec(v___y_713_);
return v_res_715_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_piUnique___at___00Equiv_punitArrowEquiv_spec__0(lean_object* v_00_u03b2_721_){
_start:
{
lean_object* v___x_722_; 
v___x_722_ = ((lean_object*)(lp_mathlib_Equiv_piUnique___at___00Equiv_punitArrowEquiv_spec__0___closed__2));
return v___x_722_;
}
}
static lean_object* _init_lp_mathlib_Equiv_punitArrowEquiv___closed__0(void){
_start:
{
lean_object* v___x_723_; 
v___x_723_ = lp_mathlib_Equiv_piUnique___at___00Equiv_punitArrowEquiv_spec__0(lean_box(0));
return v___x_723_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_punitArrowEquiv(lean_object* v_00_u03b1_724_){
_start:
{
lean_object* v___x_725_; 
v___x_725_ = lean_obj_once(&lp_mathlib_Equiv_punitArrowEquiv___closed__0, &lp_mathlib_Equiv_punitArrowEquiv___closed__0_once, _init_lp_mathlib_Equiv_punitArrowEquiv___closed__0);
return v___x_725_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_piUnique___at___00Equiv_trueArrowEquiv_spec__0___lam__0(lean_object* v_f_726_){
_start:
{
lean_object* v___x_727_; 
v___x_727_ = lean_apply_1(v_f_726_, lean_box(0));
return v___x_727_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_piUnique___at___00Equiv_trueArrowEquiv_spec__0___lam__1(lean_object* v___y_728_, lean_object* v___y_729_){
_start:
{
lean_inc(v___y_728_);
return v___y_728_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_piUnique___at___00Equiv_trueArrowEquiv_spec__0___lam__1___boxed(lean_object* v___y_730_, lean_object* v___y_731_){
_start:
{
lean_object* v_res_732_; 
v_res_732_ = lp_mathlib_Equiv_piUnique___at___00Equiv_trueArrowEquiv_spec__0___lam__1(v___y_730_, v___y_731_);
lean_dec(v___y_730_);
return v_res_732_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_piUnique___at___00Equiv_trueArrowEquiv_spec__0(lean_object* v_00_u03b2_738_){
_start:
{
lean_object* v___x_739_; 
v___x_739_ = ((lean_object*)(lp_mathlib_Equiv_piUnique___at___00Equiv_trueArrowEquiv_spec__0___closed__2));
return v___x_739_;
}
}
static lean_object* _init_lp_mathlib_Equiv_trueArrowEquiv___closed__0(void){
_start:
{
lean_object* v___x_740_; 
v___x_740_ = lp_mathlib_Equiv_piUnique___at___00Equiv_trueArrowEquiv_spec__0(lean_box(0));
return v___x_740_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_trueArrowEquiv(lean_object* v_00_u03b1_741_){
_start:
{
lean_object* v___x_742_; 
v___x_742_ = lean_obj_once(&lp_mathlib_Equiv_trueArrowEquiv___closed__0, &lp_mathlib_Equiv_trueArrowEquiv___closed__0_once, _init_lp_mathlib_Equiv_trueArrowEquiv___closed__0);
return v___x_742_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_arrowPUnitOfIsEmpty___lam__0(lean_object* v_x_743_){
_start:
{
lean_object* v___x_744_; 
v___x_744_ = lean_box(0);
return v___x_744_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_arrowPUnitOfIsEmpty___lam__0___boxed(lean_object* v_x_745_){
_start:
{
lean_object* v_res_746_; 
v_res_746_ = lp_mathlib_Equiv_arrowPUnitOfIsEmpty___lam__0(v_x_745_);
lean_dec(v_x_745_);
return v_res_746_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_arrowPUnitOfIsEmpty___lam__1(lean_object* v_x_747_, lean_object* v_a_748_){
_start:
{
lean_internal_panic_unreachable();
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_arrowPUnitOfIsEmpty___lam__1___boxed(lean_object* v_x_749_, lean_object* v_a_750_){
_start:
{
lean_object* v_res_751_; 
v_res_751_ = lp_mathlib_Equiv_arrowPUnitOfIsEmpty___lam__1(v_x_749_, v_a_750_);
lean_dec(v_a_750_);
return v_res_751_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_arrowPUnitOfIsEmpty(lean_object* v_00_u03b1_757_, lean_object* v_00_u03b2_758_, lean_object* v_inst_759_){
_start:
{
lean_object* v___x_760_; 
v___x_760_ = ((lean_object*)(lp_mathlib_Equiv_arrowPUnitOfIsEmpty___closed__2));
return v___x_760_;
}
}
static lean_object* _init_lp_mathlib_Equiv_emptyArrowEquivPUnit___closed__0(void){
_start:
{
lean_object* v___x_761_; 
v___x_761_ = lp_mathlib_Equiv_arrowPUnitOfIsEmpty(lean_box(0), lean_box(0), lean_box(0));
return v___x_761_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_emptyArrowEquivPUnit(lean_object* v_00_u03b1_762_){
_start:
{
lean_object* v___x_763_; 
v___x_763_ = lean_obj_once(&lp_mathlib_Equiv_emptyArrowEquivPUnit___closed__0, &lp_mathlib_Equiv_emptyArrowEquivPUnit___closed__0_once, _init_lp_mathlib_Equiv_emptyArrowEquivPUnit___closed__0);
return v___x_763_;
}
}
static lean_object* _init_lp_mathlib_Equiv_pemptyArrowEquivPUnit___closed__0(void){
_start:
{
lean_object* v___x_764_; 
v___x_764_ = lp_mathlib_Equiv_arrowPUnitOfIsEmpty(lean_box(0), lean_box(0), lean_box(0));
return v___x_764_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_pemptyArrowEquivPUnit(lean_object* v_00_u03b1_765_){
_start:
{
lean_object* v___x_766_; 
v___x_766_ = lean_obj_once(&lp_mathlib_Equiv_pemptyArrowEquivPUnit___closed__0, &lp_mathlib_Equiv_pemptyArrowEquivPUnit___closed__0_once, _init_lp_mathlib_Equiv_pemptyArrowEquivPUnit___closed__0);
return v___x_766_;
}
}
static lean_object* _init_lp_mathlib_Equiv_falseArrowEquivPUnit___closed__0(void){
_start:
{
lean_object* v___x_767_; 
v___x_767_ = lp_mathlib_Equiv_arrowPUnitOfIsEmpty(lean_box(0), lean_box(0), lean_box(0));
return v___x_767_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_falseArrowEquivPUnit(lean_object* v_00_u03b1_768_){
_start:
{
lean_object* v___x_769_; 
v___x_769_ = lean_obj_once(&lp_mathlib_Equiv_falseArrowEquivPUnit___closed__0, &lp_mathlib_Equiv_falseArrowEquivPUnit___closed__0_once, _init_lp_mathlib_Equiv_falseArrowEquivPUnit___closed__0);
return v___x_769_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_psigmaEquivSigma___lam__0(lean_object* v_a_770_){
_start:
{
lean_object* v_fst_771_; lean_object* v_snd_772_; lean_object* v___x_774_; uint8_t v_isShared_775_; uint8_t v_isSharedCheck_779_; 
v_fst_771_ = lean_ctor_get(v_a_770_, 0);
v_snd_772_ = lean_ctor_get(v_a_770_, 1);
v_isSharedCheck_779_ = !lean_is_exclusive(v_a_770_);
if (v_isSharedCheck_779_ == 0)
{
v___x_774_ = v_a_770_;
v_isShared_775_ = v_isSharedCheck_779_;
goto v_resetjp_773_;
}
else
{
lean_inc(v_snd_772_);
lean_inc(v_fst_771_);
lean_dec(v_a_770_);
v___x_774_ = lean_box(0);
v_isShared_775_ = v_isSharedCheck_779_;
goto v_resetjp_773_;
}
v_resetjp_773_:
{
lean_object* v___x_777_; 
if (v_isShared_775_ == 0)
{
v___x_777_ = v___x_774_;
goto v_reusejp_776_;
}
else
{
lean_object* v_reuseFailAlloc_778_; 
v_reuseFailAlloc_778_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_778_, 0, v_fst_771_);
lean_ctor_set(v_reuseFailAlloc_778_, 1, v_snd_772_);
v___x_777_ = v_reuseFailAlloc_778_;
goto v_reusejp_776_;
}
v_reusejp_776_:
{
return v___x_777_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_psigmaEquivSigma___lam__1(lean_object* v_a_780_){
_start:
{
lean_object* v_fst_781_; lean_object* v_snd_782_; lean_object* v___x_784_; uint8_t v_isShared_785_; uint8_t v_isSharedCheck_789_; 
v_fst_781_ = lean_ctor_get(v_a_780_, 0);
v_snd_782_ = lean_ctor_get(v_a_780_, 1);
v_isSharedCheck_789_ = !lean_is_exclusive(v_a_780_);
if (v_isSharedCheck_789_ == 0)
{
v___x_784_ = v_a_780_;
v_isShared_785_ = v_isSharedCheck_789_;
goto v_resetjp_783_;
}
else
{
lean_inc(v_snd_782_);
lean_inc(v_fst_781_);
lean_dec(v_a_780_);
v___x_784_ = lean_box(0);
v_isShared_785_ = v_isSharedCheck_789_;
goto v_resetjp_783_;
}
v_resetjp_783_:
{
lean_object* v___x_787_; 
if (v_isShared_785_ == 0)
{
v___x_787_ = v___x_784_;
goto v_reusejp_786_;
}
else
{
lean_object* v_reuseFailAlloc_788_; 
v_reuseFailAlloc_788_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_788_, 0, v_fst_781_);
lean_ctor_set(v_reuseFailAlloc_788_, 1, v_snd_782_);
v___x_787_ = v_reuseFailAlloc_788_;
goto v_reusejp_786_;
}
v_reusejp_786_:
{
return v___x_787_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_psigmaEquivSigma(lean_object* v_00_u03b1_795_, lean_object* v_00_u03b2_796_){
_start:
{
lean_object* v___x_797_; 
v___x_797_ = ((lean_object*)(lp_mathlib_Equiv_psigmaEquivSigma___closed__2));
return v___x_797_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_psigmaEquivSigmaPLift(lean_object* v_00_u03b1_798_, lean_object* v_00_u03b2_799_){
_start:
{
lean_object* v___x_800_; 
v___x_800_ = ((lean_object*)(lp_mathlib_Equiv_psigmaEquivSigma___closed__2));
return v___x_800_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_psigmaCongrRight___redArg___lam__0(lean_object* v_F_801_, lean_object* v_a_802_){
_start:
{
lean_object* v_fst_803_; lean_object* v_snd_804_; lean_object* v___x_806_; uint8_t v_isShared_807_; uint8_t v_isSharedCheck_814_; 
v_fst_803_ = lean_ctor_get(v_a_802_, 0);
v_snd_804_ = lean_ctor_get(v_a_802_, 1);
v_isSharedCheck_814_ = !lean_is_exclusive(v_a_802_);
if (v_isSharedCheck_814_ == 0)
{
v___x_806_ = v_a_802_;
v_isShared_807_ = v_isSharedCheck_814_;
goto v_resetjp_805_;
}
else
{
lean_inc(v_snd_804_);
lean_inc(v_fst_803_);
lean_dec(v_a_802_);
v___x_806_ = lean_box(0);
v_isShared_807_ = v_isSharedCheck_814_;
goto v_resetjp_805_;
}
v_resetjp_805_:
{
lean_object* v___x_808_; lean_object* v_toFun_809_; lean_object* v___x_810_; lean_object* v___x_812_; 
lean_inc(v_fst_803_);
v___x_808_ = lean_apply_1(v_F_801_, v_fst_803_);
v_toFun_809_ = lean_ctor_get(v___x_808_, 0);
lean_inc(v_toFun_809_);
lean_dec_ref(v___x_808_);
v___x_810_ = lean_apply_1(v_toFun_809_, v_snd_804_);
if (v_isShared_807_ == 0)
{
lean_ctor_set(v___x_806_, 1, v___x_810_);
v___x_812_ = v___x_806_;
goto v_reusejp_811_;
}
else
{
lean_object* v_reuseFailAlloc_813_; 
v_reuseFailAlloc_813_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_813_, 0, v_fst_803_);
lean_ctor_set(v_reuseFailAlloc_813_, 1, v___x_810_);
v___x_812_ = v_reuseFailAlloc_813_;
goto v_reusejp_811_;
}
v_reusejp_811_:
{
return v___x_812_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_psigmaCongrRight___redArg___lam__1(lean_object* v_F_815_, lean_object* v_a_816_){
_start:
{
lean_object* v_fst_817_; lean_object* v_snd_818_; lean_object* v___x_820_; uint8_t v_isShared_821_; uint8_t v_isSharedCheck_829_; 
v_fst_817_ = lean_ctor_get(v_a_816_, 0);
v_snd_818_ = lean_ctor_get(v_a_816_, 1);
v_isSharedCheck_829_ = !lean_is_exclusive(v_a_816_);
if (v_isSharedCheck_829_ == 0)
{
v___x_820_ = v_a_816_;
v_isShared_821_ = v_isSharedCheck_829_;
goto v_resetjp_819_;
}
else
{
lean_inc(v_snd_818_);
lean_inc(v_fst_817_);
lean_dec(v_a_816_);
v___x_820_ = lean_box(0);
v_isShared_821_ = v_isSharedCheck_829_;
goto v_resetjp_819_;
}
v_resetjp_819_:
{
lean_object* v___x_822_; lean_object* v___x_823_; lean_object* v_toFun_824_; lean_object* v___x_825_; lean_object* v___x_827_; 
lean_inc(v_fst_817_);
v___x_822_ = lean_apply_1(v_F_815_, v_fst_817_);
v___x_823_ = lp_mathlib_Equiv_symm___redArg(v___x_822_);
v_toFun_824_ = lean_ctor_get(v___x_823_, 0);
lean_inc(v_toFun_824_);
lean_dec_ref(v___x_823_);
v___x_825_ = lean_apply_1(v_toFun_824_, v_snd_818_);
if (v_isShared_821_ == 0)
{
lean_ctor_set(v___x_820_, 1, v___x_825_);
v___x_827_ = v___x_820_;
goto v_reusejp_826_;
}
else
{
lean_object* v_reuseFailAlloc_828_; 
v_reuseFailAlloc_828_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_828_, 0, v_fst_817_);
lean_ctor_set(v_reuseFailAlloc_828_, 1, v___x_825_);
v___x_827_ = v_reuseFailAlloc_828_;
goto v_reusejp_826_;
}
v_reusejp_826_:
{
return v___x_827_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_psigmaCongrRight___redArg(lean_object* v_F_830_){
_start:
{
lean_object* v___f_831_; lean_object* v___f_832_; lean_object* v___x_833_; 
lean_inc_ref(v_F_830_);
v___f_831_ = lean_alloc_closure((void*)(lp_mathlib_Equiv_psigmaCongrRight___redArg___lam__0), 2, 1);
lean_closure_set(v___f_831_, 0, v_F_830_);
v___f_832_ = lean_alloc_closure((void*)(lp_mathlib_Equiv_psigmaCongrRight___redArg___lam__1), 2, 1);
lean_closure_set(v___f_832_, 0, v_F_830_);
v___x_833_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_833_, 0, v___f_831_);
lean_ctor_set(v___x_833_, 1, v___f_832_);
return v___x_833_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_psigmaCongrRight(lean_object* v_00_u03b1_834_, lean_object* v_00_u03b2_u2081_835_, lean_object* v_00_u03b2_u2082_836_, lean_object* v_F_837_){
_start:
{
lean_object* v___x_838_; 
v___x_838_ = lp_mathlib_Equiv_psigmaCongrRight___redArg(v_F_837_);
return v___x_838_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_sigmaCongrRight___redArg___lam__0(lean_object* v_F_839_, lean_object* v_a_840_){
_start:
{
lean_object* v_fst_841_; lean_object* v_snd_842_; lean_object* v___x_844_; uint8_t v_isShared_845_; uint8_t v_isSharedCheck_852_; 
v_fst_841_ = lean_ctor_get(v_a_840_, 0);
v_snd_842_ = lean_ctor_get(v_a_840_, 1);
v_isSharedCheck_852_ = !lean_is_exclusive(v_a_840_);
if (v_isSharedCheck_852_ == 0)
{
v___x_844_ = v_a_840_;
v_isShared_845_ = v_isSharedCheck_852_;
goto v_resetjp_843_;
}
else
{
lean_inc(v_snd_842_);
lean_inc(v_fst_841_);
lean_dec(v_a_840_);
v___x_844_ = lean_box(0);
v_isShared_845_ = v_isSharedCheck_852_;
goto v_resetjp_843_;
}
v_resetjp_843_:
{
lean_object* v___x_846_; lean_object* v_toFun_847_; lean_object* v___x_848_; lean_object* v___x_850_; 
lean_inc(v_fst_841_);
v___x_846_ = lean_apply_1(v_F_839_, v_fst_841_);
v_toFun_847_ = lean_ctor_get(v___x_846_, 0);
lean_inc(v_toFun_847_);
lean_dec_ref(v___x_846_);
v___x_848_ = lean_apply_1(v_toFun_847_, v_snd_842_);
if (v_isShared_845_ == 0)
{
lean_ctor_set(v___x_844_, 1, v___x_848_);
v___x_850_ = v___x_844_;
goto v_reusejp_849_;
}
else
{
lean_object* v_reuseFailAlloc_851_; 
v_reuseFailAlloc_851_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_851_, 0, v_fst_841_);
lean_ctor_set(v_reuseFailAlloc_851_, 1, v___x_848_);
v___x_850_ = v_reuseFailAlloc_851_;
goto v_reusejp_849_;
}
v_reusejp_849_:
{
return v___x_850_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_sigmaCongrRight___redArg___lam__1(lean_object* v_F_853_, lean_object* v_a_854_){
_start:
{
lean_object* v_fst_855_; lean_object* v_snd_856_; lean_object* v___x_858_; uint8_t v_isShared_859_; uint8_t v_isSharedCheck_867_; 
v_fst_855_ = lean_ctor_get(v_a_854_, 0);
v_snd_856_ = lean_ctor_get(v_a_854_, 1);
v_isSharedCheck_867_ = !lean_is_exclusive(v_a_854_);
if (v_isSharedCheck_867_ == 0)
{
v___x_858_ = v_a_854_;
v_isShared_859_ = v_isSharedCheck_867_;
goto v_resetjp_857_;
}
else
{
lean_inc(v_snd_856_);
lean_inc(v_fst_855_);
lean_dec(v_a_854_);
v___x_858_ = lean_box(0);
v_isShared_859_ = v_isSharedCheck_867_;
goto v_resetjp_857_;
}
v_resetjp_857_:
{
lean_object* v___x_860_; lean_object* v___x_861_; lean_object* v_toFun_862_; lean_object* v___x_863_; lean_object* v___x_865_; 
lean_inc(v_fst_855_);
v___x_860_ = lean_apply_1(v_F_853_, v_fst_855_);
v___x_861_ = lp_mathlib_Equiv_symm___redArg(v___x_860_);
v_toFun_862_ = lean_ctor_get(v___x_861_, 0);
lean_inc(v_toFun_862_);
lean_dec_ref(v___x_861_);
v___x_863_ = lean_apply_1(v_toFun_862_, v_snd_856_);
if (v_isShared_859_ == 0)
{
lean_ctor_set(v___x_858_, 1, v___x_863_);
v___x_865_ = v___x_858_;
goto v_reusejp_864_;
}
else
{
lean_object* v_reuseFailAlloc_866_; 
v_reuseFailAlloc_866_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_866_, 0, v_fst_855_);
lean_ctor_set(v_reuseFailAlloc_866_, 1, v___x_863_);
v___x_865_ = v_reuseFailAlloc_866_;
goto v_reusejp_864_;
}
v_reusejp_864_:
{
return v___x_865_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_sigmaCongrRight___redArg(lean_object* v_F_868_){
_start:
{
lean_object* v___f_869_; lean_object* v___f_870_; lean_object* v___x_871_; 
lean_inc_ref(v_F_868_);
v___f_869_ = lean_alloc_closure((void*)(lp_mathlib_Equiv_sigmaCongrRight___redArg___lam__0), 2, 1);
lean_closure_set(v___f_869_, 0, v_F_868_);
v___f_870_ = lean_alloc_closure((void*)(lp_mathlib_Equiv_sigmaCongrRight___redArg___lam__1), 2, 1);
lean_closure_set(v___f_870_, 0, v_F_868_);
v___x_871_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_871_, 0, v___f_869_);
lean_ctor_set(v___x_871_, 1, v___f_870_);
return v___x_871_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_sigmaCongrRight(lean_object* v_00_u03b1_872_, lean_object* v_00_u03b2_u2081_873_, lean_object* v_00_u03b2_u2082_874_, lean_object* v_F_875_){
_start:
{
lean_object* v___x_876_; 
v___x_876_ = lp_mathlib_Equiv_sigmaCongrRight___redArg(v_F_875_);
return v___x_876_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_psigmaEquivSubtype___lam__0(lean_object* v_x_877_){
_start:
{
lean_object* v_fst_878_; 
v_fst_878_ = lean_ctor_get(v_x_877_, 0);
lean_inc(v_fst_878_);
return v_fst_878_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_psigmaEquivSubtype___lam__0___boxed(lean_object* v_x_879_){
_start:
{
lean_object* v_res_880_; 
v_res_880_ = lp_mathlib_Equiv_psigmaEquivSubtype___lam__0(v_x_879_);
lean_dec_ref(v_x_879_);
return v_res_880_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_psigmaEquivSubtype___lam__1(lean_object* v_x_881_){
_start:
{
lean_object* v___x_882_; 
v___x_882_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_882_, 0, v_x_881_);
lean_ctor_set(v___x_882_, 1, lean_box(0));
return v___x_882_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_psigmaEquivSubtype(lean_object* v_00_u03b1_888_, lean_object* v_P_889_){
_start:
{
lean_object* v___x_890_; 
v___x_890_ = ((lean_object*)(lp_mathlib_Equiv_psigmaEquivSubtype___closed__2));
return v___x_890_;
}
}
static lean_object* _init_lp_mathlib_Equiv_sigmaPLiftEquivSubtype___lam__0___closed__0(void){
_start:
{
lean_object* v___x_891_; 
v___x_891_ = lp_mathlib_Equiv_plift(lean_box(0));
return v___x_891_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_sigmaPLiftEquivSubtype___lam__0(lean_object* v_x_892_){
_start:
{
lean_object* v___x_893_; 
v___x_893_ = lean_obj_once(&lp_mathlib_Equiv_sigmaPLiftEquivSubtype___lam__0___closed__0, &lp_mathlib_Equiv_sigmaPLiftEquivSubtype___lam__0___closed__0_once, _init_lp_mathlib_Equiv_sigmaPLiftEquivSubtype___lam__0___closed__0);
return v___x_893_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_sigmaPLiftEquivSubtype___lam__0___boxed(lean_object* v_x_894_){
_start:
{
lean_object* v_res_895_; 
v_res_895_ = lp_mathlib_Equiv_sigmaPLiftEquivSubtype___lam__0(v_x_894_);
lean_dec(v_x_894_);
return v_res_895_;
}
}
static lean_object* _init_lp_mathlib_Equiv_sigmaPLiftEquivSubtype___closed__1(void){
_start:
{
lean_object* v___x_897_; 
v___x_897_ = lp_mathlib_Equiv_psigmaEquivSigma(lean_box(0), lean_box(0));
return v___x_897_;
}
}
static lean_object* _init_lp_mathlib_Equiv_sigmaPLiftEquivSubtype___closed__2(void){
_start:
{
lean_object* v___x_898_; lean_object* v___x_899_; 
v___x_898_ = lean_obj_once(&lp_mathlib_Equiv_sigmaPLiftEquivSubtype___closed__1, &lp_mathlib_Equiv_sigmaPLiftEquivSubtype___closed__1_once, _init_lp_mathlib_Equiv_sigmaPLiftEquivSubtype___closed__1);
v___x_899_ = lp_mathlib_Equiv_symm___redArg(v___x_898_);
return v___x_899_;
}
}
static lean_object* _init_lp_mathlib_Equiv_sigmaPLiftEquivSubtype___closed__3(void){
_start:
{
lean_object* v___f_900_; lean_object* v___x_901_; 
v___f_900_ = ((lean_object*)(lp_mathlib_Equiv_sigmaPLiftEquivSubtype___closed__0));
v___x_901_ = lp_mathlib_Equiv_psigmaCongrRight___redArg(v___f_900_);
return v___x_901_;
}
}
static lean_object* _init_lp_mathlib_Equiv_sigmaPLiftEquivSubtype___closed__4(void){
_start:
{
lean_object* v___x_902_; lean_object* v___x_903_; lean_object* v___x_904_; 
v___x_902_ = lean_obj_once(&lp_mathlib_Equiv_sigmaPLiftEquivSubtype___closed__3, &lp_mathlib_Equiv_sigmaPLiftEquivSubtype___closed__3_once, _init_lp_mathlib_Equiv_sigmaPLiftEquivSubtype___closed__3);
v___x_903_ = lean_obj_once(&lp_mathlib_Equiv_sigmaPLiftEquivSubtype___closed__2, &lp_mathlib_Equiv_sigmaPLiftEquivSubtype___closed__2_once, _init_lp_mathlib_Equiv_sigmaPLiftEquivSubtype___closed__2);
v___x_904_ = lp_mathlib_Equiv_trans___redArg(v___x_903_, v___x_902_);
return v___x_904_;
}
}
static lean_object* _init_lp_mathlib_Equiv_sigmaPLiftEquivSubtype___closed__5(void){
_start:
{
lean_object* v___x_905_; 
v___x_905_ = lp_mathlib_Equiv_psigmaEquivSubtype(lean_box(0), lean_box(0));
return v___x_905_;
}
}
static lean_object* _init_lp_mathlib_Equiv_sigmaPLiftEquivSubtype___closed__6(void){
_start:
{
lean_object* v___x_906_; lean_object* v___x_907_; lean_object* v___x_908_; 
v___x_906_ = lean_obj_once(&lp_mathlib_Equiv_sigmaPLiftEquivSubtype___closed__5, &lp_mathlib_Equiv_sigmaPLiftEquivSubtype___closed__5_once, _init_lp_mathlib_Equiv_sigmaPLiftEquivSubtype___closed__5);
v___x_907_ = lean_obj_once(&lp_mathlib_Equiv_sigmaPLiftEquivSubtype___closed__4, &lp_mathlib_Equiv_sigmaPLiftEquivSubtype___closed__4_once, _init_lp_mathlib_Equiv_sigmaPLiftEquivSubtype___closed__4);
v___x_908_ = lp_mathlib_Equiv_trans___redArg(v___x_907_, v___x_906_);
return v___x_908_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_sigmaPLiftEquivSubtype(lean_object* v_00_u03b1_909_, lean_object* v_P_910_){
_start:
{
lean_object* v___x_911_; 
v___x_911_ = lean_obj_once(&lp_mathlib_Equiv_sigmaPLiftEquivSubtype___closed__6, &lp_mathlib_Equiv_sigmaPLiftEquivSubtype___closed__6_once, _init_lp_mathlib_Equiv_sigmaPLiftEquivSubtype___closed__6);
return v___x_911_;
}
}
static lean_object* _init_lp_mathlib_Equiv_sigmaULiftPLiftEquivSubtype___lam__0___closed__0(void){
_start:
{
lean_object* v___x_912_; 
v___x_912_ = lp_mathlib_Equiv_ulift(lean_box(0));
return v___x_912_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_sigmaULiftPLiftEquivSubtype___lam__0(lean_object* v_x_913_){
_start:
{
lean_object* v___x_914_; 
v___x_914_ = lean_obj_once(&lp_mathlib_Equiv_sigmaULiftPLiftEquivSubtype___lam__0___closed__0, &lp_mathlib_Equiv_sigmaULiftPLiftEquivSubtype___lam__0___closed__0_once, _init_lp_mathlib_Equiv_sigmaULiftPLiftEquivSubtype___lam__0___closed__0);
return v___x_914_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_sigmaULiftPLiftEquivSubtype___lam__0___boxed(lean_object* v_x_915_){
_start:
{
lean_object* v_res_916_; 
v_res_916_ = lp_mathlib_Equiv_sigmaULiftPLiftEquivSubtype___lam__0(v_x_915_);
lean_dec(v_x_915_);
return v_res_916_;
}
}
static lean_object* _init_lp_mathlib_Equiv_sigmaULiftPLiftEquivSubtype___closed__1(void){
_start:
{
lean_object* v___f_918_; lean_object* v___x_919_; 
v___f_918_ = ((lean_object*)(lp_mathlib_Equiv_sigmaULiftPLiftEquivSubtype___closed__0));
v___x_919_ = lp_mathlib_Equiv_sigmaCongrRight___redArg(v___f_918_);
return v___x_919_;
}
}
static lean_object* _init_lp_mathlib_Equiv_sigmaULiftPLiftEquivSubtype___closed__2(void){
_start:
{
lean_object* v___x_920_; 
v___x_920_ = lp_mathlib_Equiv_sigmaPLiftEquivSubtype(lean_box(0), lean_box(0));
return v___x_920_;
}
}
static lean_object* _init_lp_mathlib_Equiv_sigmaULiftPLiftEquivSubtype___closed__3(void){
_start:
{
lean_object* v___x_921_; lean_object* v___x_922_; lean_object* v___x_923_; 
v___x_921_ = lean_obj_once(&lp_mathlib_Equiv_sigmaULiftPLiftEquivSubtype___closed__2, &lp_mathlib_Equiv_sigmaULiftPLiftEquivSubtype___closed__2_once, _init_lp_mathlib_Equiv_sigmaULiftPLiftEquivSubtype___closed__2);
v___x_922_ = lean_obj_once(&lp_mathlib_Equiv_sigmaULiftPLiftEquivSubtype___closed__1, &lp_mathlib_Equiv_sigmaULiftPLiftEquivSubtype___closed__1_once, _init_lp_mathlib_Equiv_sigmaULiftPLiftEquivSubtype___closed__1);
v___x_923_ = lp_mathlib_Equiv_trans___redArg(v___x_922_, v___x_921_);
return v___x_923_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_sigmaULiftPLiftEquivSubtype(lean_object* v_00_u03b1_924_, lean_object* v_P_925_){
_start:
{
lean_object* v___x_926_; 
v___x_926_ = lean_obj_once(&lp_mathlib_Equiv_sigmaULiftPLiftEquivSubtype___closed__3, &lp_mathlib_Equiv_sigmaULiftPLiftEquivSubtype___closed__3_once, _init_lp_mathlib_Equiv_sigmaULiftPLiftEquivSubtype___closed__3);
return v___x_926_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_Perm_sigmaCongrRight___redArg(lean_object* v_F_927_){
_start:
{
lean_object* v___x_928_; 
v___x_928_ = lp_mathlib_Equiv_sigmaCongrRight___redArg(v_F_927_);
return v___x_928_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_Perm_sigmaCongrRight(lean_object* v_00_u03b1_929_, lean_object* v_00_u03b2_930_, lean_object* v_F_931_){
_start:
{
lean_object* v___x_932_; 
v___x_932_ = lp_mathlib_Equiv_sigmaCongrRight___redArg(v_F_931_);
return v___x_932_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_functionSwap___lam__0(lean_object* v___y_933_, lean_object* v___y_934_, lean_object* v___y_935_){
_start:
{
lean_object* v___x_936_; 
v___x_936_ = lean_apply_2(v___y_933_, v___y_935_, v___y_934_);
return v___x_936_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_functionSwap(lean_object* v_00_u03b1_940_, lean_object* v_00_u03b2_941_, lean_object* v_00_u03b3_942_){
_start:
{
lean_object* v___x_943_; 
v___x_943_ = ((lean_object*)(lp_mathlib_Equiv_functionSwap___closed__1));
return v___x_943_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_sigmaCongrLeft___redArg___lam__0(lean_object* v_e_944_, lean_object* v_a_945_){
_start:
{
lean_object* v_fst_946_; lean_object* v_snd_947_; lean_object* v___x_949_; uint8_t v_isShared_950_; uint8_t v_isSharedCheck_956_; 
v_fst_946_ = lean_ctor_get(v_a_945_, 0);
v_snd_947_ = lean_ctor_get(v_a_945_, 1);
v_isSharedCheck_956_ = !lean_is_exclusive(v_a_945_);
if (v_isSharedCheck_956_ == 0)
{
v___x_949_ = v_a_945_;
v_isShared_950_ = v_isSharedCheck_956_;
goto v_resetjp_948_;
}
else
{
lean_inc(v_snd_947_);
lean_inc(v_fst_946_);
lean_dec(v_a_945_);
v___x_949_ = lean_box(0);
v_isShared_950_ = v_isSharedCheck_956_;
goto v_resetjp_948_;
}
v_resetjp_948_:
{
lean_object* v_toFun_951_; lean_object* v___x_952_; lean_object* v___x_954_; 
v_toFun_951_ = lean_ctor_get(v_e_944_, 0);
lean_inc(v_toFun_951_);
lean_dec_ref(v_e_944_);
v___x_952_ = lean_apply_1(v_toFun_951_, v_fst_946_);
if (v_isShared_950_ == 0)
{
lean_ctor_set(v___x_949_, 0, v___x_952_);
v___x_954_ = v___x_949_;
goto v_reusejp_953_;
}
else
{
lean_object* v_reuseFailAlloc_955_; 
v_reuseFailAlloc_955_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_955_, 0, v___x_952_);
lean_ctor_set(v_reuseFailAlloc_955_, 1, v_snd_947_);
v___x_954_ = v_reuseFailAlloc_955_;
goto v_reusejp_953_;
}
v_reusejp_953_:
{
return v___x_954_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_sigmaCongrLeft___redArg___lam__1(lean_object* v_e_957_, lean_object* v_a_958_){
_start:
{
lean_object* v_fst_959_; lean_object* v_snd_960_; lean_object* v___x_962_; uint8_t v_isShared_963_; uint8_t v_isSharedCheck_970_; 
v_fst_959_ = lean_ctor_get(v_a_958_, 0);
v_snd_960_ = lean_ctor_get(v_a_958_, 1);
v_isSharedCheck_970_ = !lean_is_exclusive(v_a_958_);
if (v_isSharedCheck_970_ == 0)
{
v___x_962_ = v_a_958_;
v_isShared_963_ = v_isSharedCheck_970_;
goto v_resetjp_961_;
}
else
{
lean_inc(v_snd_960_);
lean_inc(v_fst_959_);
lean_dec(v_a_958_);
v___x_962_ = lean_box(0);
v_isShared_963_ = v_isSharedCheck_970_;
goto v_resetjp_961_;
}
v_resetjp_961_:
{
lean_object* v___x_964_; lean_object* v_toFun_965_; lean_object* v___x_966_; lean_object* v___x_968_; 
v___x_964_ = lp_mathlib_Equiv_symm___redArg(v_e_957_);
v_toFun_965_ = lean_ctor_get(v___x_964_, 0);
lean_inc(v_toFun_965_);
lean_dec_ref(v___x_964_);
v___x_966_ = lean_apply_1(v_toFun_965_, v_fst_959_);
if (v_isShared_963_ == 0)
{
lean_ctor_set(v___x_962_, 0, v___x_966_);
v___x_968_ = v___x_962_;
goto v_reusejp_967_;
}
else
{
lean_object* v_reuseFailAlloc_969_; 
v_reuseFailAlloc_969_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_969_, 0, v___x_966_);
lean_ctor_set(v_reuseFailAlloc_969_, 1, v_snd_960_);
v___x_968_ = v_reuseFailAlloc_969_;
goto v_reusejp_967_;
}
v_reusejp_967_:
{
return v___x_968_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_sigmaCongrLeft___redArg(lean_object* v_e_971_){
_start:
{
lean_object* v___f_972_; lean_object* v___f_973_; lean_object* v___x_974_; 
lean_inc_ref(v_e_971_);
v___f_972_ = lean_alloc_closure((void*)(lp_mathlib_Equiv_sigmaCongrLeft___redArg___lam__0), 2, 1);
lean_closure_set(v___f_972_, 0, v_e_971_);
v___f_973_ = lean_alloc_closure((void*)(lp_mathlib_Equiv_sigmaCongrLeft___redArg___lam__1), 2, 1);
lean_closure_set(v___f_973_, 0, v_e_971_);
v___x_974_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_974_, 0, v___f_972_);
lean_ctor_set(v___x_974_, 1, v___f_973_);
return v___x_974_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_sigmaCongrLeft(lean_object* v_00_u03b1_u2081_975_, lean_object* v_00_u03b1_u2082_976_, lean_object* v_00_u03b2_977_, lean_object* v_e_978_){
_start:
{
lean_object* v___x_979_; 
v___x_979_ = lp_mathlib_Equiv_sigmaCongrLeft___redArg(v_e_978_);
return v___x_979_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_sigmaCongrLeft_x27___redArg(lean_object* v_f_980_){
_start:
{
lean_object* v___x_981_; lean_object* v___x_982_; lean_object* v___x_983_; 
v___x_981_ = lp_mathlib_Equiv_symm___redArg(v_f_980_);
v___x_982_ = lp_mathlib_Equiv_sigmaCongrLeft___redArg(v___x_981_);
v___x_983_ = lp_mathlib_Equiv_symm___redArg(v___x_982_);
return v___x_983_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_sigmaCongrLeft_x27(lean_object* v_00_u03b1_u2081_984_, lean_object* v_00_u03b1_u2082_985_, lean_object* v_00_u03b2_986_, lean_object* v_f_987_){
_start:
{
lean_object* v___x_988_; 
v___x_988_ = lp_mathlib_Equiv_sigmaCongrLeft_x27___redArg(v_f_987_);
return v___x_988_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_sigmaCongr___redArg(lean_object* v_f_989_, lean_object* v_F_990_){
_start:
{
lean_object* v___x_991_; lean_object* v___x_992_; lean_object* v___x_993_; 
v___x_991_ = lp_mathlib_Equiv_sigmaCongrRight___redArg(v_F_990_);
v___x_992_ = lp_mathlib_Equiv_sigmaCongrLeft___redArg(v_f_989_);
v___x_993_ = lp_mathlib_Equiv_trans___redArg(v___x_991_, v___x_992_);
return v___x_993_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_sigmaCongr(lean_object* v_00_u03b1_u2081_994_, lean_object* v_00_u03b1_u2082_995_, lean_object* v_00_u03b2_u2081_996_, lean_object* v_00_u03b2_u2082_997_, lean_object* v_f_998_, lean_object* v_F_999_){
_start:
{
lean_object* v___x_1000_; 
v___x_1000_ = lp_mathlib_Equiv_sigmaCongr___redArg(v_f_998_, v_F_999_);
return v___x_1000_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_sigmaEquivProd___lam__0(lean_object* v_a_1001_){
_start:
{
lean_object* v_fst_1002_; lean_object* v_snd_1003_; lean_object* v___x_1005_; uint8_t v_isShared_1006_; uint8_t v_isSharedCheck_1010_; 
v_fst_1002_ = lean_ctor_get(v_a_1001_, 0);
v_snd_1003_ = lean_ctor_get(v_a_1001_, 1);
v_isSharedCheck_1010_ = !lean_is_exclusive(v_a_1001_);
if (v_isSharedCheck_1010_ == 0)
{
v___x_1005_ = v_a_1001_;
v_isShared_1006_ = v_isSharedCheck_1010_;
goto v_resetjp_1004_;
}
else
{
lean_inc(v_snd_1003_);
lean_inc(v_fst_1002_);
lean_dec(v_a_1001_);
v___x_1005_ = lean_box(0);
v_isShared_1006_ = v_isSharedCheck_1010_;
goto v_resetjp_1004_;
}
v_resetjp_1004_:
{
lean_object* v___x_1008_; 
if (v_isShared_1006_ == 0)
{
v___x_1008_ = v___x_1005_;
goto v_reusejp_1007_;
}
else
{
lean_object* v_reuseFailAlloc_1009_; 
v_reuseFailAlloc_1009_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1009_, 0, v_fst_1002_);
lean_ctor_set(v_reuseFailAlloc_1009_, 1, v_snd_1003_);
v___x_1008_ = v_reuseFailAlloc_1009_;
goto v_reusejp_1007_;
}
v_reusejp_1007_:
{
return v___x_1008_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_sigmaEquivProd___lam__1(lean_object* v_a_1011_){
_start:
{
lean_object* v_fst_1012_; lean_object* v_snd_1013_; lean_object* v___x_1015_; uint8_t v_isShared_1016_; uint8_t v_isSharedCheck_1020_; 
v_fst_1012_ = lean_ctor_get(v_a_1011_, 0);
v_snd_1013_ = lean_ctor_get(v_a_1011_, 1);
v_isSharedCheck_1020_ = !lean_is_exclusive(v_a_1011_);
if (v_isSharedCheck_1020_ == 0)
{
v___x_1015_ = v_a_1011_;
v_isShared_1016_ = v_isSharedCheck_1020_;
goto v_resetjp_1014_;
}
else
{
lean_inc(v_snd_1013_);
lean_inc(v_fst_1012_);
lean_dec(v_a_1011_);
v___x_1015_ = lean_box(0);
v_isShared_1016_ = v_isSharedCheck_1020_;
goto v_resetjp_1014_;
}
v_resetjp_1014_:
{
lean_object* v___x_1018_; 
if (v_isShared_1016_ == 0)
{
v___x_1018_ = v___x_1015_;
goto v_reusejp_1017_;
}
else
{
lean_object* v_reuseFailAlloc_1019_; 
v_reuseFailAlloc_1019_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1019_, 0, v_fst_1012_);
lean_ctor_set(v_reuseFailAlloc_1019_, 1, v_snd_1013_);
v___x_1018_ = v_reuseFailAlloc_1019_;
goto v_reusejp_1017_;
}
v_reusejp_1017_:
{
return v___x_1018_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_sigmaEquivProd(lean_object* v_00_u03b1_1026_, lean_object* v_00_u03b2_1027_){
_start:
{
lean_object* v___x_1028_; 
v___x_1028_ = ((lean_object*)(lp_mathlib_Equiv_sigmaEquivProd___closed__2));
return v___x_1028_;
}
}
static lean_object* _init_lp_mathlib_Equiv_sigmaEquivProdOfEquiv___redArg___closed__0(void){
_start:
{
lean_object* v___x_1029_; 
v___x_1029_ = lp_mathlib_Equiv_sigmaEquivProd(lean_box(0), lean_box(0));
return v___x_1029_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_sigmaEquivProdOfEquiv___redArg(lean_object* v_F_1030_){
_start:
{
lean_object* v___x_1031_; lean_object* v___x_1032_; lean_object* v___x_1033_; 
v___x_1031_ = lp_mathlib_Equiv_sigmaCongrRight___redArg(v_F_1030_);
v___x_1032_ = lean_obj_once(&lp_mathlib_Equiv_sigmaEquivProdOfEquiv___redArg___closed__0, &lp_mathlib_Equiv_sigmaEquivProdOfEquiv___redArg___closed__0_once, _init_lp_mathlib_Equiv_sigmaEquivProdOfEquiv___redArg___closed__0);
v___x_1033_ = lp_mathlib_Equiv_trans___redArg(v___x_1031_, v___x_1032_);
return v___x_1033_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_sigmaEquivProdOfEquiv(lean_object* v_00_u03b1_1034_, lean_object* v_00_u03b2_1035_, lean_object* v_00_u03b2_u2081_1036_, lean_object* v_F_1037_){
_start:
{
lean_object* v___x_1038_; 
v___x_1038_ = lp_mathlib_Equiv_sigmaEquivProdOfEquiv___redArg(v_F_1037_);
return v___x_1038_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_sigmaAssoc___lam__0(lean_object* v_x_1039_){
_start:
{
lean_object* v_fst_1040_; lean_object* v_snd_1041_; lean_object* v___x_1043_; uint8_t v_isShared_1044_; uint8_t v_isSharedCheck_1057_; 
v_fst_1040_ = lean_ctor_get(v_x_1039_, 0);
v_snd_1041_ = lean_ctor_get(v_x_1039_, 1);
v_isSharedCheck_1057_ = !lean_is_exclusive(v_x_1039_);
if (v_isSharedCheck_1057_ == 0)
{
v___x_1043_ = v_x_1039_;
v_isShared_1044_ = v_isSharedCheck_1057_;
goto v_resetjp_1042_;
}
else
{
lean_inc(v_snd_1041_);
lean_inc(v_fst_1040_);
lean_dec(v_x_1039_);
v___x_1043_ = lean_box(0);
v_isShared_1044_ = v_isSharedCheck_1057_;
goto v_resetjp_1042_;
}
v_resetjp_1042_:
{
lean_object* v_fst_1045_; lean_object* v_snd_1046_; lean_object* v___x_1048_; uint8_t v_isShared_1049_; uint8_t v_isSharedCheck_1056_; 
v_fst_1045_ = lean_ctor_get(v_fst_1040_, 0);
v_snd_1046_ = lean_ctor_get(v_fst_1040_, 1);
v_isSharedCheck_1056_ = !lean_is_exclusive(v_fst_1040_);
if (v_isSharedCheck_1056_ == 0)
{
v___x_1048_ = v_fst_1040_;
v_isShared_1049_ = v_isSharedCheck_1056_;
goto v_resetjp_1047_;
}
else
{
lean_inc(v_snd_1046_);
lean_inc(v_fst_1045_);
lean_dec(v_fst_1040_);
v___x_1048_ = lean_box(0);
v_isShared_1049_ = v_isSharedCheck_1056_;
goto v_resetjp_1047_;
}
v_resetjp_1047_:
{
lean_object* v___x_1051_; 
if (v_isShared_1049_ == 0)
{
lean_ctor_set(v___x_1048_, 1, v_snd_1041_);
lean_ctor_set(v___x_1048_, 0, v_snd_1046_);
v___x_1051_ = v___x_1048_;
goto v_reusejp_1050_;
}
else
{
lean_object* v_reuseFailAlloc_1055_; 
v_reuseFailAlloc_1055_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1055_, 0, v_snd_1046_);
lean_ctor_set(v_reuseFailAlloc_1055_, 1, v_snd_1041_);
v___x_1051_ = v_reuseFailAlloc_1055_;
goto v_reusejp_1050_;
}
v_reusejp_1050_:
{
lean_object* v___x_1053_; 
if (v_isShared_1044_ == 0)
{
lean_ctor_set(v___x_1043_, 1, v___x_1051_);
lean_ctor_set(v___x_1043_, 0, v_fst_1045_);
v___x_1053_ = v___x_1043_;
goto v_reusejp_1052_;
}
else
{
lean_object* v_reuseFailAlloc_1054_; 
v_reuseFailAlloc_1054_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1054_, 0, v_fst_1045_);
lean_ctor_set(v_reuseFailAlloc_1054_, 1, v___x_1051_);
v___x_1053_ = v_reuseFailAlloc_1054_;
goto v_reusejp_1052_;
}
v_reusejp_1052_:
{
return v___x_1053_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_sigmaAssoc___lam__1(lean_object* v_x_1058_){
_start:
{
lean_object* v_snd_1059_; lean_object* v_fst_1060_; lean_object* v___x_1062_; uint8_t v_isShared_1063_; uint8_t v_isSharedCheck_1076_; 
v_snd_1059_ = lean_ctor_get(v_x_1058_, 1);
v_fst_1060_ = lean_ctor_get(v_x_1058_, 0);
v_isSharedCheck_1076_ = !lean_is_exclusive(v_x_1058_);
if (v_isSharedCheck_1076_ == 0)
{
v___x_1062_ = v_x_1058_;
v_isShared_1063_ = v_isSharedCheck_1076_;
goto v_resetjp_1061_;
}
else
{
lean_inc(v_snd_1059_);
lean_inc(v_fst_1060_);
lean_dec(v_x_1058_);
v___x_1062_ = lean_box(0);
v_isShared_1063_ = v_isSharedCheck_1076_;
goto v_resetjp_1061_;
}
v_resetjp_1061_:
{
lean_object* v_fst_1064_; lean_object* v_snd_1065_; lean_object* v___x_1067_; uint8_t v_isShared_1068_; uint8_t v_isSharedCheck_1075_; 
v_fst_1064_ = lean_ctor_get(v_snd_1059_, 0);
v_snd_1065_ = lean_ctor_get(v_snd_1059_, 1);
v_isSharedCheck_1075_ = !lean_is_exclusive(v_snd_1059_);
if (v_isSharedCheck_1075_ == 0)
{
v___x_1067_ = v_snd_1059_;
v_isShared_1068_ = v_isSharedCheck_1075_;
goto v_resetjp_1066_;
}
else
{
lean_inc(v_snd_1065_);
lean_inc(v_fst_1064_);
lean_dec(v_snd_1059_);
v___x_1067_ = lean_box(0);
v_isShared_1068_ = v_isSharedCheck_1075_;
goto v_resetjp_1066_;
}
v_resetjp_1066_:
{
lean_object* v___x_1070_; 
if (v_isShared_1068_ == 0)
{
lean_ctor_set(v___x_1067_, 1, v_fst_1064_);
lean_ctor_set(v___x_1067_, 0, v_fst_1060_);
v___x_1070_ = v___x_1067_;
goto v_reusejp_1069_;
}
else
{
lean_object* v_reuseFailAlloc_1074_; 
v_reuseFailAlloc_1074_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1074_, 0, v_fst_1060_);
lean_ctor_set(v_reuseFailAlloc_1074_, 1, v_fst_1064_);
v___x_1070_ = v_reuseFailAlloc_1074_;
goto v_reusejp_1069_;
}
v_reusejp_1069_:
{
lean_object* v___x_1072_; 
if (v_isShared_1063_ == 0)
{
lean_ctor_set(v___x_1062_, 1, v_snd_1065_);
lean_ctor_set(v___x_1062_, 0, v___x_1070_);
v___x_1072_ = v___x_1062_;
goto v_reusejp_1071_;
}
else
{
lean_object* v_reuseFailAlloc_1073_; 
v_reuseFailAlloc_1073_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1073_, 0, v___x_1070_);
lean_ctor_set(v_reuseFailAlloc_1073_, 1, v_snd_1065_);
v___x_1072_ = v_reuseFailAlloc_1073_;
goto v_reusejp_1071_;
}
v_reusejp_1071_:
{
return v___x_1072_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_sigmaAssoc(lean_object* v_00_u03b1_1082_, lean_object* v_00_u03b2_1083_, lean_object* v_00_u03b3_1084_){
_start:
{
lean_object* v___x_1085_; 
v___x_1085_ = ((lean_object*)(lp_mathlib_Equiv_sigmaAssoc___closed__2));
return v___x_1085_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_pSigmaAssoc___lam__0(lean_object* v_x_1086_){
_start:
{
lean_object* v_fst_1087_; lean_object* v_snd_1088_; lean_object* v___x_1090_; uint8_t v_isShared_1091_; uint8_t v_isSharedCheck_1104_; 
v_fst_1087_ = lean_ctor_get(v_x_1086_, 0);
v_snd_1088_ = lean_ctor_get(v_x_1086_, 1);
v_isSharedCheck_1104_ = !lean_is_exclusive(v_x_1086_);
if (v_isSharedCheck_1104_ == 0)
{
v___x_1090_ = v_x_1086_;
v_isShared_1091_ = v_isSharedCheck_1104_;
goto v_resetjp_1089_;
}
else
{
lean_inc(v_snd_1088_);
lean_inc(v_fst_1087_);
lean_dec(v_x_1086_);
v___x_1090_ = lean_box(0);
v_isShared_1091_ = v_isSharedCheck_1104_;
goto v_resetjp_1089_;
}
v_resetjp_1089_:
{
lean_object* v_fst_1092_; lean_object* v_snd_1093_; lean_object* v___x_1095_; uint8_t v_isShared_1096_; uint8_t v_isSharedCheck_1103_; 
v_fst_1092_ = lean_ctor_get(v_fst_1087_, 0);
v_snd_1093_ = lean_ctor_get(v_fst_1087_, 1);
v_isSharedCheck_1103_ = !lean_is_exclusive(v_fst_1087_);
if (v_isSharedCheck_1103_ == 0)
{
v___x_1095_ = v_fst_1087_;
v_isShared_1096_ = v_isSharedCheck_1103_;
goto v_resetjp_1094_;
}
else
{
lean_inc(v_snd_1093_);
lean_inc(v_fst_1092_);
lean_dec(v_fst_1087_);
v___x_1095_ = lean_box(0);
v_isShared_1096_ = v_isSharedCheck_1103_;
goto v_resetjp_1094_;
}
v_resetjp_1094_:
{
lean_object* v___x_1098_; 
if (v_isShared_1096_ == 0)
{
lean_ctor_set(v___x_1095_, 1, v_snd_1088_);
lean_ctor_set(v___x_1095_, 0, v_snd_1093_);
v___x_1098_ = v___x_1095_;
goto v_reusejp_1097_;
}
else
{
lean_object* v_reuseFailAlloc_1102_; 
v_reuseFailAlloc_1102_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1102_, 0, v_snd_1093_);
lean_ctor_set(v_reuseFailAlloc_1102_, 1, v_snd_1088_);
v___x_1098_ = v_reuseFailAlloc_1102_;
goto v_reusejp_1097_;
}
v_reusejp_1097_:
{
lean_object* v___x_1100_; 
if (v_isShared_1091_ == 0)
{
lean_ctor_set(v___x_1090_, 1, v___x_1098_);
lean_ctor_set(v___x_1090_, 0, v_fst_1092_);
v___x_1100_ = v___x_1090_;
goto v_reusejp_1099_;
}
else
{
lean_object* v_reuseFailAlloc_1101_; 
v_reuseFailAlloc_1101_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1101_, 0, v_fst_1092_);
lean_ctor_set(v_reuseFailAlloc_1101_, 1, v___x_1098_);
v___x_1100_ = v_reuseFailAlloc_1101_;
goto v_reusejp_1099_;
}
v_reusejp_1099_:
{
return v___x_1100_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_pSigmaAssoc___lam__1(lean_object* v_x_1105_){
_start:
{
lean_object* v_snd_1106_; lean_object* v_fst_1107_; lean_object* v___x_1109_; uint8_t v_isShared_1110_; uint8_t v_isSharedCheck_1123_; 
v_snd_1106_ = lean_ctor_get(v_x_1105_, 1);
v_fst_1107_ = lean_ctor_get(v_x_1105_, 0);
v_isSharedCheck_1123_ = !lean_is_exclusive(v_x_1105_);
if (v_isSharedCheck_1123_ == 0)
{
v___x_1109_ = v_x_1105_;
v_isShared_1110_ = v_isSharedCheck_1123_;
goto v_resetjp_1108_;
}
else
{
lean_inc(v_snd_1106_);
lean_inc(v_fst_1107_);
lean_dec(v_x_1105_);
v___x_1109_ = lean_box(0);
v_isShared_1110_ = v_isSharedCheck_1123_;
goto v_resetjp_1108_;
}
v_resetjp_1108_:
{
lean_object* v_fst_1111_; lean_object* v_snd_1112_; lean_object* v___x_1114_; uint8_t v_isShared_1115_; uint8_t v_isSharedCheck_1122_; 
v_fst_1111_ = lean_ctor_get(v_snd_1106_, 0);
v_snd_1112_ = lean_ctor_get(v_snd_1106_, 1);
v_isSharedCheck_1122_ = !lean_is_exclusive(v_snd_1106_);
if (v_isSharedCheck_1122_ == 0)
{
v___x_1114_ = v_snd_1106_;
v_isShared_1115_ = v_isSharedCheck_1122_;
goto v_resetjp_1113_;
}
else
{
lean_inc(v_snd_1112_);
lean_inc(v_fst_1111_);
lean_dec(v_snd_1106_);
v___x_1114_ = lean_box(0);
v_isShared_1115_ = v_isSharedCheck_1122_;
goto v_resetjp_1113_;
}
v_resetjp_1113_:
{
lean_object* v___x_1117_; 
if (v_isShared_1115_ == 0)
{
lean_ctor_set(v___x_1114_, 1, v_fst_1111_);
lean_ctor_set(v___x_1114_, 0, v_fst_1107_);
v___x_1117_ = v___x_1114_;
goto v_reusejp_1116_;
}
else
{
lean_object* v_reuseFailAlloc_1121_; 
v_reuseFailAlloc_1121_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1121_, 0, v_fst_1107_);
lean_ctor_set(v_reuseFailAlloc_1121_, 1, v_fst_1111_);
v___x_1117_ = v_reuseFailAlloc_1121_;
goto v_reusejp_1116_;
}
v_reusejp_1116_:
{
lean_object* v___x_1119_; 
if (v_isShared_1110_ == 0)
{
lean_ctor_set(v___x_1109_, 1, v_snd_1112_);
lean_ctor_set(v___x_1109_, 0, v___x_1117_);
v___x_1119_ = v___x_1109_;
goto v_reusejp_1118_;
}
else
{
lean_object* v_reuseFailAlloc_1120_; 
v_reuseFailAlloc_1120_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1120_, 0, v___x_1117_);
lean_ctor_set(v_reuseFailAlloc_1120_, 1, v_snd_1112_);
v___x_1119_ = v_reuseFailAlloc_1120_;
goto v_reusejp_1118_;
}
v_reusejp_1118_:
{
return v___x_1119_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_pSigmaAssoc(lean_object* v_00_u03b1_1129_, lean_object* v_00_u03b2_1130_, lean_object* v_00_u03b3_1131_){
_start:
{
lean_object* v___x_1132_; 
v___x_1132_ = ((lean_object*)(lp_mathlib_Equiv_pSigmaAssoc___closed__2));
return v___x_1132_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Quot_congr___redArg___lam__1(lean_object* v___x_1133_, lean_object* v___y_1134_){
_start:
{
lean_object* v_toFun_1135_; lean_object* v___x_1136_; 
v_toFun_1135_ = lean_ctor_get(v___x_1133_, 0);
lean_inc(v_toFun_1135_);
lean_dec_ref(v___x_1133_);
v___x_1136_ = lean_apply_1(v_toFun_1135_, v___y_1134_);
return v___x_1136_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Quot_congr___redArg(lean_object* v_e_1137_){
_start:
{
lean_object* v___f_1138_; lean_object* v___x_1139_; lean_object* v___x_1140_; lean_object* v___f_1141_; lean_object* v___x_1142_; lean_object* v___x_1143_; 
lean_inc_ref(v_e_1137_);
v___f_1138_ = lean_alloc_closure((void*)(lp_mathlib_Equiv_decidableEq___redArg___lam__0), 2, 1);
lean_closure_set(v___f_1138_, 0, v_e_1137_);
v___x_1139_ = lean_alloc_closure((void*)(lp_mathlib_Quot_map), 7, 6);
lean_closure_set(v___x_1139_, 0, lean_box(0));
lean_closure_set(v___x_1139_, 1, lean_box(0));
lean_closure_set(v___x_1139_, 2, lean_box(0));
lean_closure_set(v___x_1139_, 3, lean_box(0));
lean_closure_set(v___x_1139_, 4, v___f_1138_);
lean_closure_set(v___x_1139_, 5, lean_box(0));
v___x_1140_ = lp_mathlib_Equiv_symm___redArg(v_e_1137_);
v___f_1141_ = lean_alloc_closure((void*)(lp_mathlib_Quot_congr___redArg___lam__1), 2, 1);
lean_closure_set(v___f_1141_, 0, v___x_1140_);
v___x_1142_ = lean_alloc_closure((void*)(lp_mathlib_Quot_map), 7, 6);
lean_closure_set(v___x_1142_, 0, lean_box(0));
lean_closure_set(v___x_1142_, 1, lean_box(0));
lean_closure_set(v___x_1142_, 2, lean_box(0));
lean_closure_set(v___x_1142_, 3, lean_box(0));
lean_closure_set(v___x_1142_, 4, v___f_1141_);
lean_closure_set(v___x_1142_, 5, lean_box(0));
v___x_1143_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1143_, 0, v___x_1139_);
lean_ctor_set(v___x_1143_, 1, v___x_1142_);
return v___x_1143_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Quot_congr(lean_object* v_00_u03b1_1144_, lean_object* v_00_u03b2_1145_, lean_object* v_ra_1146_, lean_object* v_rb_1147_, lean_object* v_e_1148_, lean_object* v_eq_1149_){
_start:
{
lean_object* v___x_1150_; 
v___x_1150_ = lp_mathlib_Quot_congr___redArg(v_e_1148_);
return v___x_1150_;
}
}
static lean_object* _init_lp_mathlib_Quot_congrRight___closed__0(void){
_start:
{
lean_object* v___x_1151_; lean_object* v___x_1152_; 
v___x_1151_ = lean_obj_once(&lp_mathlib_Equiv_inhabited_x27___closed__0, &lp_mathlib_Equiv_inhabited_x27___closed__0_once, _init_lp_mathlib_Equiv_inhabited_x27___closed__0);
v___x_1152_ = lp_mathlib_Quot_congr___redArg(v___x_1151_);
return v___x_1152_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Quot_congrRight(lean_object* v_00_u03b1_1153_, lean_object* v_r_1154_, lean_object* v_r_x27_1155_, lean_object* v_eq_1156_){
_start:
{
lean_object* v___x_1157_; 
v___x_1157_ = lean_obj_once(&lp_mathlib_Quot_congrRight___closed__0, &lp_mathlib_Quot_congrRight___closed__0_once, _init_lp_mathlib_Quot_congrRight___closed__0);
return v___x_1157_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Quot_congrLeft___redArg(lean_object* v_e_1158_){
_start:
{
lean_object* v___x_1159_; 
v___x_1159_ = lp_mathlib_Quot_congr___redArg(v_e_1158_);
return v___x_1159_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Quot_congrLeft(lean_object* v_00_u03b1_1160_, lean_object* v_00_u03b2_1161_, lean_object* v_r_1162_, lean_object* v_e_1163_){
_start:
{
lean_object* v___x_1164_; 
v___x_1164_ = lp_mathlib_Quot_congr___redArg(v_e_1163_);
return v___x_1164_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Quotient_congr___redArg(lean_object* v_e_1165_){
_start:
{
lean_object* v___x_1166_; 
v___x_1166_ = lp_mathlib_Quot_congr___redArg(v_e_1165_);
return v___x_1166_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Quotient_congr(lean_object* v_00_u03b1_1167_, lean_object* v_00_u03b2_1168_, lean_object* v_ra_1169_, lean_object* v_rb_1170_, lean_object* v_e_1171_, lean_object* v_eq_1172_){
_start:
{
lean_object* v___x_1173_; 
v___x_1173_ = lp_mathlib_Quot_congr___redArg(v_e_1171_);
return v___x_1173_;
}
}
static lean_object* _init_lp_mathlib_Quotient_congrRight___closed__0(void){
_start:
{
lean_object* v___x_1174_; 
v___x_1174_ = lp_mathlib_Quot_congrRight(lean_box(0), lean_box(0), lean_box(0), lean_box(0));
return v___x_1174_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Quotient_congrRight(lean_object* v_00_u03b1_1175_, lean_object* v_r_1176_, lean_object* v_r_x27_1177_, lean_object* v_eq_1178_){
_start:
{
lean_object* v___x_1179_; 
v___x_1179_ = lean_obj_once(&lp_mathlib_Quotient_congrRight___closed__0, &lp_mathlib_Quotient_congrRight___closed__0_once, _init_lp_mathlib_Quotient_congrRight___closed__0);
return v___x_1179_;
}
}
static lean_object* _init_lp_mathlib_finZeroEquiv___closed__0(void){
_start:
{
lean_object* v___x_1180_; 
v___x_1180_ = lp_mathlib_Equiv_equivOfIsEmpty(lean_box(0), lean_box(0), lean_box(0), lean_box(0));
return v___x_1180_;
}
}
static lean_object* _init_lp_mathlib_finZeroEquiv(void){
_start:
{
lean_object* v___x_1181_; 
v___x_1181_ = lean_obj_once(&lp_mathlib_finZeroEquiv___closed__0, &lp_mathlib_finZeroEquiv___closed__0_once, _init_lp_mathlib_finZeroEquiv___closed__0);
return v___x_1181_;
}
}
static lean_object* _init_lp_mathlib_finZeroEquiv_x27___closed__0(void){
_start:
{
lean_object* v___x_1182_; 
v___x_1182_ = lp_mathlib_Equiv_equivOfIsEmpty(lean_box(0), lean_box(0), lean_box(0), lean_box(0));
return v___x_1182_;
}
}
static lean_object* _init_lp_mathlib_finZeroEquiv_x27(void){
_start:
{
lean_object* v___x_1183_; 
v___x_1183_ = lean_obj_once(&lp_mathlib_finZeroEquiv_x27___closed__0, &lp_mathlib_finZeroEquiv_x27___closed__0_once, _init_lp_mathlib_finZeroEquiv_x27___closed__0);
return v___x_1183_;
}
}
static lean_object* _init_lp_mathlib_Equiv_ofUnique___at___00Equiv_equivPUnit___at___00finOneEquiv_spec__0_spec__0___lam__1___closed__0(void){
_start:
{
lean_object* v___x_1184_; lean_object* v___x_1185_; lean_object* v___x_1186_; 
v___x_1184_ = lean_unsigned_to_nat(1u);
v___x_1185_ = lean_unsigned_to_nat(0u);
v___x_1186_ = lean_nat_mod(v___x_1185_, v___x_1184_);
return v___x_1186_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_ofUnique___at___00Equiv_equivPUnit___at___00finOneEquiv_spec__0_spec__0___lam__1(lean_object* v_x_1187_){
_start:
{
lean_object* v___x_1188_; 
v___x_1188_ = lean_obj_once(&lp_mathlib_Equiv_ofUnique___at___00Equiv_equivPUnit___at___00finOneEquiv_spec__0_spec__0___lam__1___closed__0, &lp_mathlib_Equiv_ofUnique___at___00Equiv_equivPUnit___at___00finOneEquiv_spec__0_spec__0___lam__1___closed__0_once, _init_lp_mathlib_Equiv_ofUnique___at___00Equiv_equivPUnit___at___00finOneEquiv_spec__0_spec__0___lam__1___closed__0);
return v___x_1188_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_ofUnique___at___00Equiv_equivPUnit___at___00finOneEquiv_spec__0_spec__0___lam__0(lean_object* v_x_1189_){
_start:
{
lean_object* v___x_1190_; 
v___x_1190_ = lean_box(0);
return v___x_1190_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_ofUnique___at___00Equiv_equivPUnit___at___00finOneEquiv_spec__0_spec__0___lam__0___boxed(lean_object* v_x_1191_){
_start:
{
lean_object* v_res_1192_; 
v_res_1192_ = lp_mathlib_Equiv_ofUnique___at___00Equiv_equivPUnit___at___00finOneEquiv_spec__0_spec__0___lam__0(v_x_1191_);
lean_dec(v_x_1191_);
return v_res_1192_;
}
}
static lean_object* _init_lp_mathlib_finTwoEquiv___lam__0___closed__0(void){
_start:
{
lean_object* v___x_1201_; lean_object* v___x_1202_; lean_object* v___x_1203_; 
v___x_1201_ = lean_unsigned_to_nat(2u);
v___x_1202_ = lean_unsigned_to_nat(1u);
v___x_1203_ = lean_nat_mod(v___x_1202_, v___x_1201_);
return v___x_1203_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_finTwoEquiv___lam__0(lean_object* v_i_1204_){
_start:
{
lean_object* v___x_1205_; uint8_t v___x_1206_; 
v___x_1205_ = lean_obj_once(&lp_mathlib_finTwoEquiv___lam__0___closed__0, &lp_mathlib_finTwoEquiv___lam__0___closed__0_once, _init_lp_mathlib_finTwoEquiv___lam__0___closed__0);
v___x_1206_ = lean_nat_dec_eq(v_i_1204_, v___x_1205_);
return v___x_1206_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_finTwoEquiv___lam__0___boxed(lean_object* v_i_1207_){
_start:
{
uint8_t v_res_1208_; lean_object* v_r_1209_; 
v_res_1208_ = lp_mathlib_finTwoEquiv___lam__0(v_i_1207_);
lean_dec(v_i_1207_);
v_r_1209_ = lean_box(v_res_1208_);
return v_r_1209_;
}
}
static lean_object* _init_lp_mathlib_finTwoEquiv___lam__1___closed__0(void){
_start:
{
lean_object* v___x_1210_; lean_object* v___x_1211_; lean_object* v___x_1212_; 
v___x_1210_ = lean_unsigned_to_nat(2u);
v___x_1211_ = lean_unsigned_to_nat(0u);
v___x_1212_ = lean_nat_mod(v___x_1211_, v___x_1210_);
return v___x_1212_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_finTwoEquiv___lam__1(uint8_t v_b_1213_){
_start:
{
if (v_b_1213_ == 0)
{
lean_object* v___x_1214_; 
v___x_1214_ = lean_obj_once(&lp_mathlib_finTwoEquiv___lam__1___closed__0, &lp_mathlib_finTwoEquiv___lam__1___closed__0_once, _init_lp_mathlib_finTwoEquiv___lam__1___closed__0);
return v___x_1214_;
}
else
{
lean_object* v___x_1215_; 
v___x_1215_ = lean_obj_once(&lp_mathlib_finTwoEquiv___lam__0___closed__0, &lp_mathlib_finTwoEquiv___lam__0___closed__0_once, _init_lp_mathlib_finTwoEquiv___lam__0___closed__0);
return v___x_1215_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_finTwoEquiv___lam__1___boxed(lean_object* v_b_1216_){
_start:
{
uint8_t v_b_boxed_1217_; lean_object* v_res_1218_; 
v_b_boxed_1217_ = lean_unbox(v_b_1216_);
v_res_1218_ = lp_mathlib_finTwoEquiv___lam__1(v_b_boxed_1217_);
return v_res_1218_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_sumIsLeft___lam__0(lean_object* v_x_1225_){
_start:
{
lean_object* v_val_1226_; 
v_val_1226_ = lean_ctor_get(v_x_1225_, 0);
lean_inc(v_val_1226_);
return v_val_1226_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_sumIsLeft___lam__0___boxed(lean_object* v_x_1227_){
_start:
{
lean_object* v_res_1228_; 
v_res_1228_ = lp_mathlib_Equiv_sumIsLeft___lam__0(v_x_1227_);
lean_dec_ref(v_x_1227_);
return v_res_1228_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_sumIsLeft___lam__1(lean_object* v_a_1229_){
_start:
{
lean_object* v___x_1230_; 
v___x_1230_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1230_, 0, v_a_1229_);
return v___x_1230_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_sumIsLeft(lean_object* v_00_u03b1_1236_, lean_object* v_00_u03b2_1237_){
_start:
{
lean_object* v___x_1238_; 
v___x_1238_ = ((lean_object*)(lp_mathlib_Equiv_sumIsLeft___closed__2));
return v___x_1238_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_sumIsRight___lam__0(lean_object* v_x_1239_){
_start:
{
lean_object* v_val_1240_; 
v_val_1240_ = lean_ctor_get(v_x_1239_, 0);
lean_inc(v_val_1240_);
return v_val_1240_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_sumIsRight___lam__0___boxed(lean_object* v_x_1241_){
_start:
{
lean_object* v_res_1242_; 
v_res_1242_ = lp_mathlib_Equiv_sumIsRight___lam__0(v_x_1241_);
lean_dec_ref(v_x_1241_);
return v_res_1242_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_sumIsRight___lam__1(lean_object* v_b_1243_){
_start:
{
lean_object* v___x_1244_; 
v___x_1244_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_1244_, 0, v_b_1243_);
return v___x_1244_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_sumIsRight(lean_object* v_00_u03b1_1250_, lean_object* v_00_u03b2_1251_){
_start:
{
lean_object* v___x_1252_; 
v___x_1252_ = ((lean_object*)(lp_mathlib_Equiv_sumIsRight___closed__2));
return v___x_1252_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_le(lean_object* v_00_u03b1_1253_, lean_object* v_00_u03b2_1254_, lean_object* v_e_1255_, lean_object* v_inst_1256_){
_start:
{
lean_object* v___x_1257_; 
v___x_1257_ = lean_box(0);
return v___x_1257_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_le___boxed(lean_object* v_00_u03b1_1258_, lean_object* v_00_u03b2_1259_, lean_object* v_e_1260_, lean_object* v_inst_1261_){
_start:
{
lean_object* v_res_1262_; 
v_res_1262_ = lp_mathlib_Equiv_le(v_00_u03b1_1258_, v_00_u03b2_1259_, v_e_1260_, v_inst_1261_);
lean_dec_ref(v_e_1260_);
return v_res_1262_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_lt(lean_object* v_00_u03b1_1263_, lean_object* v_00_u03b2_1264_, lean_object* v_e_1265_, lean_object* v_inst_1266_){
_start:
{
lean_object* v___x_1267_; 
v___x_1267_ = lean_box(0);
return v___x_1267_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_lt___boxed(lean_object* v_00_u03b1_1268_, lean_object* v_00_u03b2_1269_, lean_object* v_e_1270_, lean_object* v_inst_1271_){
_start:
{
lean_object* v_res_1272_; 
v_res_1272_ = lp_mathlib_Equiv_lt(v_00_u03b1_1268_, v_00_u03b2_1269_, v_e_1270_, v_inst_1271_);
lean_dec_ref(v_e_1270_);
return v_res_1272_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_max___redArg___lam__0(lean_object* v_self_1273_, lean_object* v___y_1274_){
_start:
{
lean_object* v_toFun_1275_; lean_object* v___x_1276_; 
v_toFun_1275_ = lean_ctor_get(v_self_1273_, 0);
lean_inc(v_toFun_1275_);
lean_dec_ref(v_self_1273_);
v___x_1276_ = lean_apply_1(v_toFun_1275_, v___y_1274_);
return v___x_1276_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_max___redArg___lam__1(lean_object* v_e_1277_, lean_object* v___f_1278_, lean_object* v_inst_1279_, lean_object* v_a_1280_, lean_object* v_b_1281_){
_start:
{
lean_object* v___x_1282_; lean_object* v_toFun_1283_; lean_object* v___x_1284_; lean_object* v___x_1285_; lean_object* v___x_1286_; lean_object* v___x_1287_; 
lean_inc_ref_n(v_e_1277_, 2);
v___x_1282_ = lp_mathlib_Equiv_symm___redArg(v_e_1277_);
v_toFun_1283_ = lean_ctor_get(v___x_1282_, 0);
lean_inc(v_toFun_1283_);
lean_dec_ref(v___x_1282_);
lean_inc(v___f_1278_);
v___x_1284_ = lean_apply_2(v___f_1278_, v_e_1277_, v_a_1280_);
v___x_1285_ = lean_apply_2(v___f_1278_, v_e_1277_, v_b_1281_);
v___x_1286_ = lean_apply_2(v_inst_1279_, v___x_1284_, v___x_1285_);
v___x_1287_ = lean_apply_1(v_toFun_1283_, v___x_1286_);
return v___x_1287_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_max___redArg(lean_object* v_e_1289_, lean_object* v_inst_1290_){
_start:
{
lean_object* v___f_1291_; lean_object* v___f_1292_; 
v___f_1291_ = ((lean_object*)(lp_mathlib_Equiv_max___redArg___closed__0));
v___f_1292_ = lean_alloc_closure((void*)(lp_mathlib_Equiv_max___redArg___lam__1), 5, 3);
lean_closure_set(v___f_1292_, 0, v_e_1289_);
lean_closure_set(v___f_1292_, 1, v___f_1291_);
lean_closure_set(v___f_1292_, 2, v_inst_1290_);
return v___f_1292_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_max(lean_object* v_00_u03b1_1293_, lean_object* v_00_u03b2_1294_, lean_object* v_e_1295_, lean_object* v_inst_1296_){
_start:
{
lean_object* v___f_1297_; lean_object* v___f_1298_; 
v___f_1297_ = ((lean_object*)(lp_mathlib_Equiv_max___redArg___closed__0));
v___f_1298_ = lean_alloc_closure((void*)(lp_mathlib_Equiv_max___redArg___lam__1), 5, 3);
lean_closure_set(v___f_1298_, 0, v_e_1295_);
lean_closure_set(v___f_1298_, 1, v___f_1297_);
lean_closure_set(v___f_1298_, 2, v_inst_1296_);
return v___f_1298_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_min___redArg(lean_object* v_e_1299_, lean_object* v_inst_1300_){
_start:
{
lean_object* v___f_1301_; lean_object* v___f_1302_; 
v___f_1301_ = ((lean_object*)(lp_mathlib_Equiv_max___redArg___closed__0));
v___f_1302_ = lean_alloc_closure((void*)(lp_mathlib_Equiv_max___redArg___lam__1), 5, 3);
lean_closure_set(v___f_1302_, 0, v_e_1299_);
lean_closure_set(v___f_1302_, 1, v___f_1301_);
lean_closure_set(v___f_1302_, 2, v_inst_1300_);
return v___f_1302_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_min(lean_object* v_00_u03b1_1303_, lean_object* v_00_u03b2_1304_, lean_object* v_e_1305_, lean_object* v_inst_1306_){
_start:
{
lean_object* v___f_1307_; lean_object* v___f_1308_; 
v___f_1307_ = ((lean_object*)(lp_mathlib_Equiv_max___redArg___closed__0));
v___f_1308_ = lean_alloc_closure((void*)(lp_mathlib_Equiv_max___redArg___lam__1), 5, 3);
lean_closure_set(v___f_1308_, 0, v_e_1305_);
lean_closure_set(v___f_1308_, 1, v___f_1307_);
lean_closure_set(v___f_1308_, 2, v_inst_1306_);
return v___f_1308_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Equiv_ord___redArg___lam__1(lean_object* v___f_1309_, lean_object* v_e_1310_, lean_object* v_inst_1311_, lean_object* v_a_1312_, lean_object* v_b_1313_){
_start:
{
lean_object* v___x_1314_; lean_object* v___x_1315_; lean_object* v___x_1316_; uint8_t v___x_1317_; 
lean_inc(v___f_1309_);
lean_inc_ref(v_e_1310_);
v___x_1314_ = lean_apply_2(v___f_1309_, v_e_1310_, v_a_1312_);
v___x_1315_ = lean_apply_2(v___f_1309_, v_e_1310_, v_b_1313_);
v___x_1316_ = lean_apply_2(v_inst_1311_, v___x_1314_, v___x_1315_);
v___x_1317_ = lean_unbox(v___x_1316_);
return v___x_1317_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_ord___redArg___lam__1___boxed(lean_object* v___f_1318_, lean_object* v_e_1319_, lean_object* v_inst_1320_, lean_object* v_a_1321_, lean_object* v_b_1322_){
_start:
{
uint8_t v_res_1323_; lean_object* v_r_1324_; 
v_res_1323_ = lp_mathlib_Equiv_ord___redArg___lam__1(v___f_1318_, v_e_1319_, v_inst_1320_, v_a_1321_, v_b_1322_);
v_r_1324_ = lean_box(v_res_1323_);
return v_r_1324_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_ord___redArg(lean_object* v_e_1325_, lean_object* v_inst_1326_){
_start:
{
lean_object* v___f_1327_; lean_object* v___f_1328_; 
v___f_1327_ = ((lean_object*)(lp_mathlib_Equiv_max___redArg___closed__0));
v___f_1328_ = lean_alloc_closure((void*)(lp_mathlib_Equiv_ord___redArg___lam__1___boxed), 5, 3);
lean_closure_set(v___f_1328_, 0, v___f_1327_);
lean_closure_set(v___f_1328_, 1, v_e_1325_);
lean_closure_set(v___f_1328_, 2, v_inst_1326_);
return v___f_1328_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_ord(lean_object* v_00_u03b1_1329_, lean_object* v_00_u03b2_1330_, lean_object* v_e_1331_, lean_object* v_inst_1332_){
_start:
{
lean_object* v___f_1333_; lean_object* v___f_1334_; 
v___f_1333_ = ((lean_object*)(lp_mathlib_Equiv_max___redArg___closed__0));
v___f_1334_ = lean_alloc_closure((void*)(lp_mathlib_Equiv_ord___redArg___lam__1___boxed), 5, 3);
lean_closure_set(v___f_1334_, 0, v___f_1333_);
lean_closure_set(v___f_1334_, 1, v_e_1331_);
lean_closure_set(v___f_1334_, 2, v_inst_1332_);
return v___f_1334_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Basic_Unique(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Data_FunLike_Equiv(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Data_Quot(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Data_Subtype(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Tactic_Simps(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_Logic_Equiv_Defs(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Basic_Unique(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Data_FunLike_Equiv(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Data_Quot(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Data_Subtype(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Tactic_Simps(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
lp_mathlib_finZeroEquiv = _init_lp_mathlib_finZeroEquiv();
lean_mark_persistent(lp_mathlib_finZeroEquiv);
lp_mathlib_finZeroEquiv_x27 = _init_lp_mathlib_finZeroEquiv_x27();
lean_mark_persistent(lp_mathlib_finZeroEquiv_x27);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_Logic_Equiv_Defs(uint8_t builtin) {
lean_object * res;
if (_G_meta_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_meta_initialized = true;
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
lp_mathlib_Equiv_left__inv___autoParam = _init_lp_mathlib_Equiv_left__inv___autoParam();
lean_mark_persistent(lp_mathlib_Equiv_left__inv___autoParam);
lp_mathlib_Equiv_right__inv___autoParam = _init_lp_mathlib_Equiv_right__inv___autoParam();
lean_mark_persistent(lp_mathlib_Equiv_right__inv___autoParam);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Basic_Unique(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Data_FunLike_Equiv(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Data_Quot(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Data_Subtype(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Tactic_Simps(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Logic_Equiv_Defs(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Basic_Unique(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_FunLike_Equiv(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_Quot(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_Subtype(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Tactic_Simps(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Logic_Equiv_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_Logic_Equiv_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_Logic_Equiv_Defs(builtin);
}
#ifdef __cplusplus
}
#endif
