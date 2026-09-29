// Lean compiler output
// Module: Mathlib.Data.Fin.Basic
// Imports: public import Init public meta import Init public import Mathlib.Data.Int.DivMod public import Mathlib.Data.Nat.Init public import Mathlib.Logic.Equiv.Defs public import Mathlib.Tactic.Common public import Batteries.Data.Fin.Basic public import Mathlib.Tactic.Attr.Core
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
lean_object* l_Array_mkArray0(lean_object*);
lean_object* l_Lean_Name_mkStr2(lean_object*, lean_object*);
lean_object* lean_nat_add(lean_object*, lean_object*);
lean_object* l_Fin_instInhabited___redArg(lean_object*);
lean_object* l_Lean_Name_mkStr1(lean_object*);
lean_object* l_Lean_Name_mkStr4(lean_object*, lean_object*, lean_object*, lean_object*);
uint8_t l_Lean_Syntax_isOfKind(lean_object*, lean_object*);
lean_object* l_Lean_SourceInfo_fromRef(lean_object*, uint8_t);
lean_object* l_Lean_Syntax_node1(lean_object*, lean_object*, lean_object*);
lean_object* l_String_toRawSubstring_x27(lean_object*);
lean_object* l_Lean_addMacroScope(lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Syntax_node3(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Syntax_node5(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Syntax_node2(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Syntax_node6(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_finZeroElim(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_finZeroElim___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Fin_rec0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Fin_rec0___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Fin_equivSubtype___lam__0(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Fin_equivSubtype___lam__0___boxed(lean_object*);
static const lean_closure_object lp_mathlib_Fin_equivSubtype___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Fin_equivSubtype___lam__0___boxed, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Fin_equivSubtype___closed__0 = (const lean_object*)&lp_mathlib_Fin_equivSubtype___closed__0_value;
static const lean_ctor_object lp_mathlib_Fin_equivSubtype___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_Fin_equivSubtype___closed__0_value),((lean_object*)&lp_mathlib_Fin_equivSubtype___closed__0_value)}};
static const lean_object* lp_mathlib_Fin_equivSubtype___closed__1 = (const lean_object*)&lp_mathlib_Fin_equivSubtype___closed__1_value;
LEAN_EXPORT lean_object* lp_mathlib_Fin_equivSubtype(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Fin_equivSubtype___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Fin_instWellFoundedRelation__mathlib(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Fin_instWellFoundedRelation__mathlib___boxed(lean_object*);
static const lean_string_object lp_mathlib_Fin_tacticFin__omega___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = "Fin"};
static const lean_object* lp_mathlib_Fin_tacticFin__omega___closed__0 = (const lean_object*)&lp_mathlib_Fin_tacticFin__omega___closed__0_value;
static const lean_string_object lp_mathlib_Fin_tacticFin__omega___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 16, .m_capacity = 16, .m_length = 15, .m_data = "tacticFin_omega"};
static const lean_object* lp_mathlib_Fin_tacticFin__omega___closed__1 = (const lean_object*)&lp_mathlib_Fin_tacticFin__omega___closed__1_value;
static const lean_ctor_object lp_mathlib_Fin_tacticFin__omega___closed__2_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Fin_tacticFin__omega___closed__0_value),LEAN_SCALAR_PTR_LITERAL(62, 91, 162, 2, 110, 238, 123, 219)}};
static const lean_ctor_object lp_mathlib_Fin_tacticFin__omega___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Fin_tacticFin__omega___closed__2_value_aux_0),((lean_object*)&lp_mathlib_Fin_tacticFin__omega___closed__1_value),LEAN_SCALAR_PTR_LITERAL(68, 105, 166, 74, 80, 103, 130, 10)}};
static const lean_object* lp_mathlib_Fin_tacticFin__omega___closed__2 = (const lean_object*)&lp_mathlib_Fin_tacticFin__omega___closed__2_value;
static const lean_string_object lp_mathlib_Fin_tacticFin__omega___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 9, .m_data = "fin_omega"};
static const lean_object* lp_mathlib_Fin_tacticFin__omega___closed__3 = (const lean_object*)&lp_mathlib_Fin_tacticFin__omega___closed__3_value;
static const lean_ctor_object lp_mathlib_Fin_tacticFin__omega___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 8, .m_other = 1, .m_tag = 6}, .m_objs = {((lean_object*)&lp_mathlib_Fin_tacticFin__omega___closed__3_value),LEAN_SCALAR_PTR_LITERAL(0, 0, 0, 0, 0, 0, 0, 0)}};
static const lean_object* lp_mathlib_Fin_tacticFin__omega___closed__4 = (const lean_object*)&lp_mathlib_Fin_tacticFin__omega___closed__4_value;
static const lean_ctor_object lp_mathlib_Fin_tacticFin__omega___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 3}, .m_objs = {((lean_object*)&lp_mathlib_Fin_tacticFin__omega___closed__2_value),((lean_object*)(((size_t)(1024) << 1) | 1)),((lean_object*)&lp_mathlib_Fin_tacticFin__omega___closed__4_value)}};
static const lean_object* lp_mathlib_Fin_tacticFin__omega___closed__5 = (const lean_object*)&lp_mathlib_Fin_tacticFin__omega___closed__5_value;
LEAN_EXPORT const lean_object* lp_mathlib_Fin_tacticFin__omega = (const lean_object*)&lp_mathlib_Fin_tacticFin__omega___closed__5_value;
static const lean_string_object lp_mathlib_Fin___aux__Mathlib__Data__Fin__Basic______macroRules__Fin__tacticFin__omega__1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "Lean"};
static const lean_object* lp_mathlib_Fin___aux__Mathlib__Data__Fin__Basic______macroRules__Fin__tacticFin__omega__1___closed__0 = (const lean_object*)&lp_mathlib_Fin___aux__Mathlib__Data__Fin__Basic______macroRules__Fin__tacticFin__omega__1___closed__0_value;
static const lean_string_object lp_mathlib_Fin___aux__Mathlib__Data__Fin__Basic______macroRules__Fin__tacticFin__omega__1___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "Parser"};
static const lean_object* lp_mathlib_Fin___aux__Mathlib__Data__Fin__Basic______macroRules__Fin__tacticFin__omega__1___closed__1 = (const lean_object*)&lp_mathlib_Fin___aux__Mathlib__Data__Fin__Basic______macroRules__Fin__tacticFin__omega__1___closed__1_value;
static const lean_string_object lp_mathlib_Fin___aux__Mathlib__Data__Fin__Basic______macroRules__Fin__tacticFin__omega__1___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "Tactic"};
static const lean_object* lp_mathlib_Fin___aux__Mathlib__Data__Fin__Basic______macroRules__Fin__tacticFin__omega__1___closed__2 = (const lean_object*)&lp_mathlib_Fin___aux__Mathlib__Data__Fin__Basic______macroRules__Fin__tacticFin__omega__1___closed__2_value;
static const lean_string_object lp_mathlib_Fin___aux__Mathlib__Data__Fin__Basic______macroRules__Fin__tacticFin__omega__1___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 19, .m_capacity = 19, .m_length = 18, .m_data = "tacticSeqBracketed"};
static const lean_object* lp_mathlib_Fin___aux__Mathlib__Data__Fin__Basic______macroRules__Fin__tacticFin__omega__1___closed__3 = (const lean_object*)&lp_mathlib_Fin___aux__Mathlib__Data__Fin__Basic______macroRules__Fin__tacticFin__omega__1___closed__3_value;
static const lean_ctor_object lp_mathlib_Fin___aux__Mathlib__Data__Fin__Basic______macroRules__Fin__tacticFin__omega__1___closed__4_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Fin___aux__Mathlib__Data__Fin__Basic______macroRules__Fin__tacticFin__omega__1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib_Fin___aux__Mathlib__Data__Fin__Basic______macroRules__Fin__tacticFin__omega__1___closed__4_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Fin___aux__Mathlib__Data__Fin__Basic______macroRules__Fin__tacticFin__omega__1___closed__4_value_aux_0),((lean_object*)&lp_mathlib_Fin___aux__Mathlib__Data__Fin__Basic______macroRules__Fin__tacticFin__omega__1___closed__1_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib_Fin___aux__Mathlib__Data__Fin__Basic______macroRules__Fin__tacticFin__omega__1___closed__4_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Fin___aux__Mathlib__Data__Fin__Basic______macroRules__Fin__tacticFin__omega__1___closed__4_value_aux_1),((lean_object*)&lp_mathlib_Fin___aux__Mathlib__Data__Fin__Basic______macroRules__Fin__tacticFin__omega__1___closed__2_value),LEAN_SCALAR_PTR_LITERAL(166, 58, 35, 182, 187, 130, 147, 254)}};
static const lean_ctor_object lp_mathlib_Fin___aux__Mathlib__Data__Fin__Basic______macroRules__Fin__tacticFin__omega__1___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Fin___aux__Mathlib__Data__Fin__Basic______macroRules__Fin__tacticFin__omega__1___closed__4_value_aux_2),((lean_object*)&lp_mathlib_Fin___aux__Mathlib__Data__Fin__Basic______macroRules__Fin__tacticFin__omega__1___closed__3_value),LEAN_SCALAR_PTR_LITERAL(142, 80, 121, 250, 245, 54, 71, 145)}};
static const lean_object* lp_mathlib_Fin___aux__Mathlib__Data__Fin__Basic______macroRules__Fin__tacticFin__omega__1___closed__4 = (const lean_object*)&lp_mathlib_Fin___aux__Mathlib__Data__Fin__Basic______macroRules__Fin__tacticFin__omega__1___closed__4_value;
static const lean_string_object lp_mathlib_Fin___aux__Mathlib__Data__Fin__Basic______macroRules__Fin__tacticFin__omega__1___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = "{"};
static const lean_object* lp_mathlib_Fin___aux__Mathlib__Data__Fin__Basic______macroRules__Fin__tacticFin__omega__1___closed__5 = (const lean_object*)&lp_mathlib_Fin___aux__Mathlib__Data__Fin__Basic______macroRules__Fin__tacticFin__omega__1___closed__5_value;
static const lean_string_object lp_mathlib_Fin___aux__Mathlib__Data__Fin__Basic______macroRules__Fin__tacticFin__omega__1___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "null"};
static const lean_object* lp_mathlib_Fin___aux__Mathlib__Data__Fin__Basic______macroRules__Fin__tacticFin__omega__1___closed__6 = (const lean_object*)&lp_mathlib_Fin___aux__Mathlib__Data__Fin__Basic______macroRules__Fin__tacticFin__omega__1___closed__6_value;
static const lean_ctor_object lp_mathlib_Fin___aux__Mathlib__Data__Fin__Basic______macroRules__Fin__tacticFin__omega__1___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Fin___aux__Mathlib__Data__Fin__Basic______macroRules__Fin__tacticFin__omega__1___closed__6_value),LEAN_SCALAR_PTR_LITERAL(24, 58, 49, 223, 146, 207, 197, 136)}};
static const lean_object* lp_mathlib_Fin___aux__Mathlib__Data__Fin__Basic______macroRules__Fin__tacticFin__omega__1___closed__7 = (const lean_object*)&lp_mathlib_Fin___aux__Mathlib__Data__Fin__Basic______macroRules__Fin__tacticFin__omega__1___closed__7_value;
static const lean_string_object lp_mathlib_Fin___aux__Mathlib__Data__Fin__Basic______macroRules__Fin__tacticFin__omega__1___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 11, .m_capacity = 11, .m_length = 10, .m_data = "tacticTry_"};
static const lean_object* lp_mathlib_Fin___aux__Mathlib__Data__Fin__Basic______macroRules__Fin__tacticFin__omega__1___closed__8 = (const lean_object*)&lp_mathlib_Fin___aux__Mathlib__Data__Fin__Basic______macroRules__Fin__tacticFin__omega__1___closed__8_value;
static const lean_ctor_object lp_mathlib_Fin___aux__Mathlib__Data__Fin__Basic______macroRules__Fin__tacticFin__omega__1___closed__9_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Fin___aux__Mathlib__Data__Fin__Basic______macroRules__Fin__tacticFin__omega__1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib_Fin___aux__Mathlib__Data__Fin__Basic______macroRules__Fin__tacticFin__omega__1___closed__9_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Fin___aux__Mathlib__Data__Fin__Basic______macroRules__Fin__tacticFin__omega__1___closed__9_value_aux_0),((lean_object*)&lp_mathlib_Fin___aux__Mathlib__Data__Fin__Basic______macroRules__Fin__tacticFin__omega__1___closed__1_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib_Fin___aux__Mathlib__Data__Fin__Basic______macroRules__Fin__tacticFin__omega__1___closed__9_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Fin___aux__Mathlib__Data__Fin__Basic______macroRules__Fin__tacticFin__omega__1___closed__9_value_aux_1),((lean_object*)&lp_mathlib_Fin___aux__Mathlib__Data__Fin__Basic______macroRules__Fin__tacticFin__omega__1___closed__2_value),LEAN_SCALAR_PTR_LITERAL(166, 58, 35, 182, 187, 130, 147, 254)}};
static const lean_ctor_object lp_mathlib_Fin___aux__Mathlib__Data__Fin__Basic______macroRules__Fin__tacticFin__omega__1___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Fin___aux__Mathlib__Data__Fin__Basic______macroRules__Fin__tacticFin__omega__1___closed__9_value_aux_2),((lean_object*)&lp_mathlib_Fin___aux__Mathlib__Data__Fin__Basic______macroRules__Fin__tacticFin__omega__1___closed__8_value),LEAN_SCALAR_PTR_LITERAL(34, 109, 187, 155, 23, 130, 33, 152)}};
static const lean_object* lp_mathlib_Fin___aux__Mathlib__Data__Fin__Basic______macroRules__Fin__tacticFin__omega__1___closed__9 = (const lean_object*)&lp_mathlib_Fin___aux__Mathlib__Data__Fin__Basic______macroRules__Fin__tacticFin__omega__1___closed__9_value;
static const lean_string_object lp_mathlib_Fin___aux__Mathlib__Data__Fin__Basic______macroRules__Fin__tacticFin__omega__1___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = "try"};
static const lean_object* lp_mathlib_Fin___aux__Mathlib__Data__Fin__Basic______macroRules__Fin__tacticFin__omega__1___closed__10 = (const lean_object*)&lp_mathlib_Fin___aux__Mathlib__Data__Fin__Basic______macroRules__Fin__tacticFin__omega__1___closed__10_value;
static const lean_string_object lp_mathlib_Fin___aux__Mathlib__Data__Fin__Basic______macroRules__Fin__tacticFin__omega__1___closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 9, .m_data = "tacticSeq"};
static const lean_object* lp_mathlib_Fin___aux__Mathlib__Data__Fin__Basic______macroRules__Fin__tacticFin__omega__1___closed__11 = (const lean_object*)&lp_mathlib_Fin___aux__Mathlib__Data__Fin__Basic______macroRules__Fin__tacticFin__omega__1___closed__11_value;
static const lean_ctor_object lp_mathlib_Fin___aux__Mathlib__Data__Fin__Basic______macroRules__Fin__tacticFin__omega__1___closed__12_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Fin___aux__Mathlib__Data__Fin__Basic______macroRules__Fin__tacticFin__omega__1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib_Fin___aux__Mathlib__Data__Fin__Basic______macroRules__Fin__tacticFin__omega__1___closed__12_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Fin___aux__Mathlib__Data__Fin__Basic______macroRules__Fin__tacticFin__omega__1___closed__12_value_aux_0),((lean_object*)&lp_mathlib_Fin___aux__Mathlib__Data__Fin__Basic______macroRules__Fin__tacticFin__omega__1___closed__1_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib_Fin___aux__Mathlib__Data__Fin__Basic______macroRules__Fin__tacticFin__omega__1___closed__12_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Fin___aux__Mathlib__Data__Fin__Basic______macroRules__Fin__tacticFin__omega__1___closed__12_value_aux_1),((lean_object*)&lp_mathlib_Fin___aux__Mathlib__Data__Fin__Basic______macroRules__Fin__tacticFin__omega__1___closed__2_value),LEAN_SCALAR_PTR_LITERAL(166, 58, 35, 182, 187, 130, 147, 254)}};
static const lean_ctor_object lp_mathlib_Fin___aux__Mathlib__Data__Fin__Basic______macroRules__Fin__tacticFin__omega__1___closed__12_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Fin___aux__Mathlib__Data__Fin__Basic______macroRules__Fin__tacticFin__omega__1___closed__12_value_aux_2),((lean_object*)&lp_mathlib_Fin___aux__Mathlib__Data__Fin__Basic______macroRules__Fin__tacticFin__omega__1___closed__11_value),LEAN_SCALAR_PTR_LITERAL(212, 140, 85, 215, 241, 69, 7, 118)}};
static const lean_object* lp_mathlib_Fin___aux__Mathlib__Data__Fin__Basic______macroRules__Fin__tacticFin__omega__1___closed__12 = (const lean_object*)&lp_mathlib_Fin___aux__Mathlib__Data__Fin__Basic______macroRules__Fin__tacticFin__omega__1___closed__12_value;
static const lean_string_object lp_mathlib_Fin___aux__Mathlib__Data__Fin__Basic______macroRules__Fin__tacticFin__omega__1___closed__13_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 19, .m_capacity = 19, .m_length = 18, .m_data = "tacticSeq1Indented"};
static const lean_object* lp_mathlib_Fin___aux__Mathlib__Data__Fin__Basic______macroRules__Fin__tacticFin__omega__1___closed__13 = (const lean_object*)&lp_mathlib_Fin___aux__Mathlib__Data__Fin__Basic______macroRules__Fin__tacticFin__omega__1___closed__13_value;
static const lean_ctor_object lp_mathlib_Fin___aux__Mathlib__Data__Fin__Basic______macroRules__Fin__tacticFin__omega__1___closed__14_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Fin___aux__Mathlib__Data__Fin__Basic______macroRules__Fin__tacticFin__omega__1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib_Fin___aux__Mathlib__Data__Fin__Basic______macroRules__Fin__tacticFin__omega__1___closed__14_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Fin___aux__Mathlib__Data__Fin__Basic______macroRules__Fin__tacticFin__omega__1___closed__14_value_aux_0),((lean_object*)&lp_mathlib_Fin___aux__Mathlib__Data__Fin__Basic______macroRules__Fin__tacticFin__omega__1___closed__1_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib_Fin___aux__Mathlib__Data__Fin__Basic______macroRules__Fin__tacticFin__omega__1___closed__14_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Fin___aux__Mathlib__Data__Fin__Basic______macroRules__Fin__tacticFin__omega__1___closed__14_value_aux_1),((lean_object*)&lp_mathlib_Fin___aux__Mathlib__Data__Fin__Basic______macroRules__Fin__tacticFin__omega__1___closed__2_value),LEAN_SCALAR_PTR_LITERAL(166, 58, 35, 182, 187, 130, 147, 254)}};
static const lean_ctor_object lp_mathlib_Fin___aux__Mathlib__Data__Fin__Basic______macroRules__Fin__tacticFin__omega__1___closed__14_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Fin___aux__Mathlib__Data__Fin__Basic______macroRules__Fin__tacticFin__omega__1___closed__14_value_aux_2),((lean_object*)&lp_mathlib_Fin___aux__Mathlib__Data__Fin__Basic______macroRules__Fin__tacticFin__omega__1___closed__13_value),LEAN_SCALAR_PTR_LITERAL(223, 90, 160, 238, 133, 180, 23, 239)}};
static const lean_object* lp_mathlib_Fin___aux__Mathlib__Data__Fin__Basic______macroRules__Fin__tacticFin__omega__1___closed__14 = (const lean_object*)&lp_mathlib_Fin___aux__Mathlib__Data__Fin__Basic______macroRules__Fin__tacticFin__omega__1___closed__14_value;
static const lean_string_object lp_mathlib_Fin___aux__Mathlib__Data__Fin__Basic______macroRules__Fin__tacticFin__omega__1___closed__15_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "simp"};
static const lean_object* lp_mathlib_Fin___aux__Mathlib__Data__Fin__Basic______macroRules__Fin__tacticFin__omega__1___closed__15 = (const lean_object*)&lp_mathlib_Fin___aux__Mathlib__Data__Fin__Basic______macroRules__Fin__tacticFin__omega__1___closed__15_value;
static const lean_ctor_object lp_mathlib_Fin___aux__Mathlib__Data__Fin__Basic______macroRules__Fin__tacticFin__omega__1___closed__16_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Fin___aux__Mathlib__Data__Fin__Basic______macroRules__Fin__tacticFin__omega__1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib_Fin___aux__Mathlib__Data__Fin__Basic______macroRules__Fin__tacticFin__omega__1___closed__16_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Fin___aux__Mathlib__Data__Fin__Basic______macroRules__Fin__tacticFin__omega__1___closed__16_value_aux_0),((lean_object*)&lp_mathlib_Fin___aux__Mathlib__Data__Fin__Basic______macroRules__Fin__tacticFin__omega__1___closed__1_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib_Fin___aux__Mathlib__Data__Fin__Basic______macroRules__Fin__tacticFin__omega__1___closed__16_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Fin___aux__Mathlib__Data__Fin__Basic______macroRules__Fin__tacticFin__omega__1___closed__16_value_aux_1),((lean_object*)&lp_mathlib_Fin___aux__Mathlib__Data__Fin__Basic______macroRules__Fin__tacticFin__omega__1___closed__2_value),LEAN_SCALAR_PTR_LITERAL(166, 58, 35, 182, 187, 130, 147, 254)}};
static const lean_ctor_object lp_mathlib_Fin___aux__Mathlib__Data__Fin__Basic______macroRules__Fin__tacticFin__omega__1___closed__16_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Fin___aux__Mathlib__Data__Fin__Basic______macroRules__Fin__tacticFin__omega__1___closed__16_value_aux_2),((lean_object*)&lp_mathlib_Fin___aux__Mathlib__Data__Fin__Basic______macroRules__Fin__tacticFin__omega__1___closed__15_value),LEAN_SCALAR_PTR_LITERAL(50, 13, 241, 145, 67, 153, 105, 177)}};
static const lean_object* lp_mathlib_Fin___aux__Mathlib__Data__Fin__Basic______macroRules__Fin__tacticFin__omega__1___closed__16 = (const lean_object*)&lp_mathlib_Fin___aux__Mathlib__Data__Fin__Basic______macroRules__Fin__tacticFin__omega__1___closed__16_value;
static const lean_string_object lp_mathlib_Fin___aux__Mathlib__Data__Fin__Basic______macroRules__Fin__tacticFin__omega__1___closed__17_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 9, .m_data = "optConfig"};
static const lean_object* lp_mathlib_Fin___aux__Mathlib__Data__Fin__Basic______macroRules__Fin__tacticFin__omega__1___closed__17 = (const lean_object*)&lp_mathlib_Fin___aux__Mathlib__Data__Fin__Basic______macroRules__Fin__tacticFin__omega__1___closed__17_value;
static const lean_ctor_object lp_mathlib_Fin___aux__Mathlib__Data__Fin__Basic______macroRules__Fin__tacticFin__omega__1___closed__18_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Fin___aux__Mathlib__Data__Fin__Basic______macroRules__Fin__tacticFin__omega__1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib_Fin___aux__Mathlib__Data__Fin__Basic______macroRules__Fin__tacticFin__omega__1___closed__18_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Fin___aux__Mathlib__Data__Fin__Basic______macroRules__Fin__tacticFin__omega__1___closed__18_value_aux_0),((lean_object*)&lp_mathlib_Fin___aux__Mathlib__Data__Fin__Basic______macroRules__Fin__tacticFin__omega__1___closed__1_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib_Fin___aux__Mathlib__Data__Fin__Basic______macroRules__Fin__tacticFin__omega__1___closed__18_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Fin___aux__Mathlib__Data__Fin__Basic______macroRules__Fin__tacticFin__omega__1___closed__18_value_aux_1),((lean_object*)&lp_mathlib_Fin___aux__Mathlib__Data__Fin__Basic______macroRules__Fin__tacticFin__omega__1___closed__2_value),LEAN_SCALAR_PTR_LITERAL(166, 58, 35, 182, 187, 130, 147, 254)}};
static const lean_ctor_object lp_mathlib_Fin___aux__Mathlib__Data__Fin__Basic______macroRules__Fin__tacticFin__omega__1___closed__18_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Fin___aux__Mathlib__Data__Fin__Basic______macroRules__Fin__tacticFin__omega__1___closed__18_value_aux_2),((lean_object*)&lp_mathlib_Fin___aux__Mathlib__Data__Fin__Basic______macroRules__Fin__tacticFin__omega__1___closed__17_value),LEAN_SCALAR_PTR_LITERAL(137, 208, 10, 74, 108, 50, 106, 48)}};
static const lean_object* lp_mathlib_Fin___aux__Mathlib__Data__Fin__Basic______macroRules__Fin__tacticFin__omega__1___closed__18 = (const lean_object*)&lp_mathlib_Fin___aux__Mathlib__Data__Fin__Basic______macroRules__Fin__tacticFin__omega__1___closed__18_value;
static lean_once_cell_t lp_mathlib_Fin___aux__Mathlib__Data__Fin__Basic______macroRules__Fin__tacticFin__omega__1___closed__19_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Fin___aux__Mathlib__Data__Fin__Basic______macroRules__Fin__tacticFin__omega__1___closed__19;
static const lean_string_object lp_mathlib_Fin___aux__Mathlib__Data__Fin__Basic______macroRules__Fin__tacticFin__omega__1___closed__20_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "only"};
static const lean_object* lp_mathlib_Fin___aux__Mathlib__Data__Fin__Basic______macroRules__Fin__tacticFin__omega__1___closed__20 = (const lean_object*)&lp_mathlib_Fin___aux__Mathlib__Data__Fin__Basic______macroRules__Fin__tacticFin__omega__1___closed__20_value;
static const lean_string_object lp_mathlib_Fin___aux__Mathlib__Data__Fin__Basic______macroRules__Fin__tacticFin__omega__1___closed__21_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = "["};
static const lean_object* lp_mathlib_Fin___aux__Mathlib__Data__Fin__Basic______macroRules__Fin__tacticFin__omega__1___closed__21 = (const lean_object*)&lp_mathlib_Fin___aux__Mathlib__Data__Fin__Basic______macroRules__Fin__tacticFin__omega__1___closed__21_value;
static const lean_string_object lp_mathlib_Fin___aux__Mathlib__Data__Fin__Basic______macroRules__Fin__tacticFin__omega__1___closed__22_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 9, .m_data = "simpLemma"};
static const lean_object* lp_mathlib_Fin___aux__Mathlib__Data__Fin__Basic______macroRules__Fin__tacticFin__omega__1___closed__22 = (const lean_object*)&lp_mathlib_Fin___aux__Mathlib__Data__Fin__Basic______macroRules__Fin__tacticFin__omega__1___closed__22_value;
static const lean_ctor_object lp_mathlib_Fin___aux__Mathlib__Data__Fin__Basic______macroRules__Fin__tacticFin__omega__1___closed__23_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Fin___aux__Mathlib__Data__Fin__Basic______macroRules__Fin__tacticFin__omega__1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib_Fin___aux__Mathlib__Data__Fin__Basic______macroRules__Fin__tacticFin__omega__1___closed__23_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Fin___aux__Mathlib__Data__Fin__Basic______macroRules__Fin__tacticFin__omega__1___closed__23_value_aux_0),((lean_object*)&lp_mathlib_Fin___aux__Mathlib__Data__Fin__Basic______macroRules__Fin__tacticFin__omega__1___closed__1_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib_Fin___aux__Mathlib__Data__Fin__Basic______macroRules__Fin__tacticFin__omega__1___closed__23_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Fin___aux__Mathlib__Data__Fin__Basic______macroRules__Fin__tacticFin__omega__1___closed__23_value_aux_1),((lean_object*)&lp_mathlib_Fin___aux__Mathlib__Data__Fin__Basic______macroRules__Fin__tacticFin__omega__1___closed__2_value),LEAN_SCALAR_PTR_LITERAL(166, 58, 35, 182, 187, 130, 147, 254)}};
static const lean_ctor_object lp_mathlib_Fin___aux__Mathlib__Data__Fin__Basic______macroRules__Fin__tacticFin__omega__1___closed__23_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Fin___aux__Mathlib__Data__Fin__Basic______macroRules__Fin__tacticFin__omega__1___closed__23_value_aux_2),((lean_object*)&lp_mathlib_Fin___aux__Mathlib__Data__Fin__Basic______macroRules__Fin__tacticFin__omega__1___closed__22_value),LEAN_SCALAR_PTR_LITERAL(38, 215, 101, 250, 181, 108, 118, 102)}};
static const lean_object* lp_mathlib_Fin___aux__Mathlib__Data__Fin__Basic______macroRules__Fin__tacticFin__omega__1___closed__23 = (const lean_object*)&lp_mathlib_Fin___aux__Mathlib__Data__Fin__Basic______macroRules__Fin__tacticFin__omega__1___closed__23_value;
static lean_once_cell_t lp_mathlib_Fin___aux__Mathlib__Data__Fin__Basic______macroRules__Fin__tacticFin__omega__1___closed__24_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Fin___aux__Mathlib__Data__Fin__Basic______macroRules__Fin__tacticFin__omega__1___closed__24;
static const lean_ctor_object lp_mathlib_Fin___aux__Mathlib__Data__Fin__Basic______macroRules__Fin__tacticFin__omega__1___closed__25_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Fin_tacticFin__omega___closed__3_value),LEAN_SCALAR_PTR_LITERAL(180, 240, 18, 190, 174, 37, 227, 30)}};
static const lean_object* lp_mathlib_Fin___aux__Mathlib__Data__Fin__Basic______macroRules__Fin__tacticFin__omega__1___closed__25 = (const lean_object*)&lp_mathlib_Fin___aux__Mathlib__Data__Fin__Basic______macroRules__Fin__tacticFin__omega__1___closed__25_value;
static const lean_string_object lp_mathlib_Fin___aux__Mathlib__Data__Fin__Basic______macroRules__Fin__tacticFin__omega__1___closed__26_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = ","};
static const lean_object* lp_mathlib_Fin___aux__Mathlib__Data__Fin__Basic______macroRules__Fin__tacticFin__omega__1___closed__26 = (const lean_object*)&lp_mathlib_Fin___aux__Mathlib__Data__Fin__Basic______macroRules__Fin__tacticFin__omega__1___closed__26_value;
static const lean_string_object lp_mathlib_Fin___aux__Mathlib__Data__Fin__Basic______macroRules__Fin__tacticFin__omega__1___closed__27_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 1, .m_data = "←"};
static const lean_object* lp_mathlib_Fin___aux__Mathlib__Data__Fin__Basic______macroRules__Fin__tacticFin__omega__1___closed__27 = (const lean_object*)&lp_mathlib_Fin___aux__Mathlib__Data__Fin__Basic______macroRules__Fin__tacticFin__omega__1___closed__27_value;
static const lean_string_object lp_mathlib_Fin___aux__Mathlib__Data__Fin__Basic______macroRules__Fin__tacticFin__omega__1___closed__28_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 13, .m_capacity = 13, .m_length = 12, .m_data = "Int.ofNat_lt"};
static const lean_object* lp_mathlib_Fin___aux__Mathlib__Data__Fin__Basic______macroRules__Fin__tacticFin__omega__1___closed__28 = (const lean_object*)&lp_mathlib_Fin___aux__Mathlib__Data__Fin__Basic______macroRules__Fin__tacticFin__omega__1___closed__28_value;
static lean_once_cell_t lp_mathlib_Fin___aux__Mathlib__Data__Fin__Basic______macroRules__Fin__tacticFin__omega__1___closed__29_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Fin___aux__Mathlib__Data__Fin__Basic______macroRules__Fin__tacticFin__omega__1___closed__29;
static const lean_string_object lp_mathlib_Fin___aux__Mathlib__Data__Fin__Basic______macroRules__Fin__tacticFin__omega__1___closed__30_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = "Int"};
static const lean_object* lp_mathlib_Fin___aux__Mathlib__Data__Fin__Basic______macroRules__Fin__tacticFin__omega__1___closed__30 = (const lean_object*)&lp_mathlib_Fin___aux__Mathlib__Data__Fin__Basic______macroRules__Fin__tacticFin__omega__1___closed__30_value;
static const lean_string_object lp_mathlib_Fin___aux__Mathlib__Data__Fin__Basic______macroRules__Fin__tacticFin__omega__1___closed__31_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "ofNat_lt"};
static const lean_object* lp_mathlib_Fin___aux__Mathlib__Data__Fin__Basic______macroRules__Fin__tacticFin__omega__1___closed__31 = (const lean_object*)&lp_mathlib_Fin___aux__Mathlib__Data__Fin__Basic______macroRules__Fin__tacticFin__omega__1___closed__31_value;
static const lean_ctor_object lp_mathlib_Fin___aux__Mathlib__Data__Fin__Basic______macroRules__Fin__tacticFin__omega__1___closed__32_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Fin___aux__Mathlib__Data__Fin__Basic______macroRules__Fin__tacticFin__omega__1___closed__30_value),LEAN_SCALAR_PTR_LITERAL(61, 25, 98, 154, 117, 127, 69, 97)}};
static const lean_ctor_object lp_mathlib_Fin___aux__Mathlib__Data__Fin__Basic______macroRules__Fin__tacticFin__omega__1___closed__32_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Fin___aux__Mathlib__Data__Fin__Basic______macroRules__Fin__tacticFin__omega__1___closed__32_value_aux_0),((lean_object*)&lp_mathlib_Fin___aux__Mathlib__Data__Fin__Basic______macroRules__Fin__tacticFin__omega__1___closed__31_value),LEAN_SCALAR_PTR_LITERAL(76, 147, 85, 174, 105, 11, 58, 76)}};
static const lean_object* lp_mathlib_Fin___aux__Mathlib__Data__Fin__Basic______macroRules__Fin__tacticFin__omega__1___closed__32 = (const lean_object*)&lp_mathlib_Fin___aux__Mathlib__Data__Fin__Basic______macroRules__Fin__tacticFin__omega__1___closed__32_value;
static const lean_ctor_object lp_mathlib_Fin___aux__Mathlib__Data__Fin__Basic______macroRules__Fin__tacticFin__omega__1___closed__33_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Fin___aux__Mathlib__Data__Fin__Basic______macroRules__Fin__tacticFin__omega__1___closed__32_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_Fin___aux__Mathlib__Data__Fin__Basic______macroRules__Fin__tacticFin__omega__1___closed__33 = (const lean_object*)&lp_mathlib_Fin___aux__Mathlib__Data__Fin__Basic______macroRules__Fin__tacticFin__omega__1___closed__33_value;
static const lean_ctor_object lp_mathlib_Fin___aux__Mathlib__Data__Fin__Basic______macroRules__Fin__tacticFin__omega__1___closed__34_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Fin___aux__Mathlib__Data__Fin__Basic______macroRules__Fin__tacticFin__omega__1___closed__33_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_Fin___aux__Mathlib__Data__Fin__Basic______macroRules__Fin__tacticFin__omega__1___closed__34 = (const lean_object*)&lp_mathlib_Fin___aux__Mathlib__Data__Fin__Basic______macroRules__Fin__tacticFin__omega__1___closed__34_value;
static const lean_string_object lp_mathlib_Fin___aux__Mathlib__Data__Fin__Basic______macroRules__Fin__tacticFin__omega__1___closed__35_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 13, .m_capacity = 13, .m_length = 12, .m_data = "Int.ofNat_le"};
static const lean_object* lp_mathlib_Fin___aux__Mathlib__Data__Fin__Basic______macroRules__Fin__tacticFin__omega__1___closed__35 = (const lean_object*)&lp_mathlib_Fin___aux__Mathlib__Data__Fin__Basic______macroRules__Fin__tacticFin__omega__1___closed__35_value;
static lean_once_cell_t lp_mathlib_Fin___aux__Mathlib__Data__Fin__Basic______macroRules__Fin__tacticFin__omega__1___closed__36_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Fin___aux__Mathlib__Data__Fin__Basic______macroRules__Fin__tacticFin__omega__1___closed__36;
static const lean_string_object lp_mathlib_Fin___aux__Mathlib__Data__Fin__Basic______macroRules__Fin__tacticFin__omega__1___closed__37_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "ofNat_le"};
static const lean_object* lp_mathlib_Fin___aux__Mathlib__Data__Fin__Basic______macroRules__Fin__tacticFin__omega__1___closed__37 = (const lean_object*)&lp_mathlib_Fin___aux__Mathlib__Data__Fin__Basic______macroRules__Fin__tacticFin__omega__1___closed__37_value;
static const lean_ctor_object lp_mathlib_Fin___aux__Mathlib__Data__Fin__Basic______macroRules__Fin__tacticFin__omega__1___closed__38_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Fin___aux__Mathlib__Data__Fin__Basic______macroRules__Fin__tacticFin__omega__1___closed__30_value),LEAN_SCALAR_PTR_LITERAL(61, 25, 98, 154, 117, 127, 69, 97)}};
static const lean_ctor_object lp_mathlib_Fin___aux__Mathlib__Data__Fin__Basic______macroRules__Fin__tacticFin__omega__1___closed__38_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Fin___aux__Mathlib__Data__Fin__Basic______macroRules__Fin__tacticFin__omega__1___closed__38_value_aux_0),((lean_object*)&lp_mathlib_Fin___aux__Mathlib__Data__Fin__Basic______macroRules__Fin__tacticFin__omega__1___closed__37_value),LEAN_SCALAR_PTR_LITERAL(245, 197, 242, 32, 94, 230, 188, 209)}};
static const lean_object* lp_mathlib_Fin___aux__Mathlib__Data__Fin__Basic______macroRules__Fin__tacticFin__omega__1___closed__38 = (const lean_object*)&lp_mathlib_Fin___aux__Mathlib__Data__Fin__Basic______macroRules__Fin__tacticFin__omega__1___closed__38_value;
static const lean_ctor_object lp_mathlib_Fin___aux__Mathlib__Data__Fin__Basic______macroRules__Fin__tacticFin__omega__1___closed__39_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Fin___aux__Mathlib__Data__Fin__Basic______macroRules__Fin__tacticFin__omega__1___closed__38_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_Fin___aux__Mathlib__Data__Fin__Basic______macroRules__Fin__tacticFin__omega__1___closed__39 = (const lean_object*)&lp_mathlib_Fin___aux__Mathlib__Data__Fin__Basic______macroRules__Fin__tacticFin__omega__1___closed__39_value;
static const lean_ctor_object lp_mathlib_Fin___aux__Mathlib__Data__Fin__Basic______macroRules__Fin__tacticFin__omega__1___closed__40_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Fin___aux__Mathlib__Data__Fin__Basic______macroRules__Fin__tacticFin__omega__1___closed__39_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_Fin___aux__Mathlib__Data__Fin__Basic______macroRules__Fin__tacticFin__omega__1___closed__40 = (const lean_object*)&lp_mathlib_Fin___aux__Mathlib__Data__Fin__Basic______macroRules__Fin__tacticFin__omega__1___closed__40_value;
static const lean_string_object lp_mathlib_Fin___aux__Mathlib__Data__Fin__Basic______macroRules__Fin__tacticFin__omega__1___closed__41_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = "]"};
static const lean_object* lp_mathlib_Fin___aux__Mathlib__Data__Fin__Basic______macroRules__Fin__tacticFin__omega__1___closed__41 = (const lean_object*)&lp_mathlib_Fin___aux__Mathlib__Data__Fin__Basic______macroRules__Fin__tacticFin__omega__1___closed__41_value;
static const lean_string_object lp_mathlib_Fin___aux__Mathlib__Data__Fin__Basic______macroRules__Fin__tacticFin__omega__1___closed__42_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "location"};
static const lean_object* lp_mathlib_Fin___aux__Mathlib__Data__Fin__Basic______macroRules__Fin__tacticFin__omega__1___closed__42 = (const lean_object*)&lp_mathlib_Fin___aux__Mathlib__Data__Fin__Basic______macroRules__Fin__tacticFin__omega__1___closed__42_value;
static const lean_ctor_object lp_mathlib_Fin___aux__Mathlib__Data__Fin__Basic______macroRules__Fin__tacticFin__omega__1___closed__43_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Fin___aux__Mathlib__Data__Fin__Basic______macroRules__Fin__tacticFin__omega__1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib_Fin___aux__Mathlib__Data__Fin__Basic______macroRules__Fin__tacticFin__omega__1___closed__43_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Fin___aux__Mathlib__Data__Fin__Basic______macroRules__Fin__tacticFin__omega__1___closed__43_value_aux_0),((lean_object*)&lp_mathlib_Fin___aux__Mathlib__Data__Fin__Basic______macroRules__Fin__tacticFin__omega__1___closed__1_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib_Fin___aux__Mathlib__Data__Fin__Basic______macroRules__Fin__tacticFin__omega__1___closed__43_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Fin___aux__Mathlib__Data__Fin__Basic______macroRules__Fin__tacticFin__omega__1___closed__43_value_aux_1),((lean_object*)&lp_mathlib_Fin___aux__Mathlib__Data__Fin__Basic______macroRules__Fin__tacticFin__omega__1___closed__2_value),LEAN_SCALAR_PTR_LITERAL(166, 58, 35, 182, 187, 130, 147, 254)}};
static const lean_ctor_object lp_mathlib_Fin___aux__Mathlib__Data__Fin__Basic______macroRules__Fin__tacticFin__omega__1___closed__43_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Fin___aux__Mathlib__Data__Fin__Basic______macroRules__Fin__tacticFin__omega__1___closed__43_value_aux_2),((lean_object*)&lp_mathlib_Fin___aux__Mathlib__Data__Fin__Basic______macroRules__Fin__tacticFin__omega__1___closed__42_value),LEAN_SCALAR_PTR_LITERAL(124, 82, 43, 228, 241, 102, 135, 24)}};
static const lean_object* lp_mathlib_Fin___aux__Mathlib__Data__Fin__Basic______macroRules__Fin__tacticFin__omega__1___closed__43 = (const lean_object*)&lp_mathlib_Fin___aux__Mathlib__Data__Fin__Basic______macroRules__Fin__tacticFin__omega__1___closed__43_value;
static const lean_string_object lp_mathlib_Fin___aux__Mathlib__Data__Fin__Basic______macroRules__Fin__tacticFin__omega__1___closed__44_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 3, .m_capacity = 3, .m_length = 2, .m_data = "at"};
static const lean_object* lp_mathlib_Fin___aux__Mathlib__Data__Fin__Basic______macroRules__Fin__tacticFin__omega__1___closed__44 = (const lean_object*)&lp_mathlib_Fin___aux__Mathlib__Data__Fin__Basic______macroRules__Fin__tacticFin__omega__1___closed__44_value;
static const lean_string_object lp_mathlib_Fin___aux__Mathlib__Data__Fin__Basic______macroRules__Fin__tacticFin__omega__1___closed__45_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 17, .m_capacity = 17, .m_length = 16, .m_data = "locationWildcard"};
static const lean_object* lp_mathlib_Fin___aux__Mathlib__Data__Fin__Basic______macroRules__Fin__tacticFin__omega__1___closed__45 = (const lean_object*)&lp_mathlib_Fin___aux__Mathlib__Data__Fin__Basic______macroRules__Fin__tacticFin__omega__1___closed__45_value;
static const lean_ctor_object lp_mathlib_Fin___aux__Mathlib__Data__Fin__Basic______macroRules__Fin__tacticFin__omega__1___closed__46_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Fin___aux__Mathlib__Data__Fin__Basic______macroRules__Fin__tacticFin__omega__1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib_Fin___aux__Mathlib__Data__Fin__Basic______macroRules__Fin__tacticFin__omega__1___closed__46_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Fin___aux__Mathlib__Data__Fin__Basic______macroRules__Fin__tacticFin__omega__1___closed__46_value_aux_0),((lean_object*)&lp_mathlib_Fin___aux__Mathlib__Data__Fin__Basic______macroRules__Fin__tacticFin__omega__1___closed__1_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib_Fin___aux__Mathlib__Data__Fin__Basic______macroRules__Fin__tacticFin__omega__1___closed__46_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Fin___aux__Mathlib__Data__Fin__Basic______macroRules__Fin__tacticFin__omega__1___closed__46_value_aux_1),((lean_object*)&lp_mathlib_Fin___aux__Mathlib__Data__Fin__Basic______macroRules__Fin__tacticFin__omega__1___closed__2_value),LEAN_SCALAR_PTR_LITERAL(166, 58, 35, 182, 187, 130, 147, 254)}};
static const lean_ctor_object lp_mathlib_Fin___aux__Mathlib__Data__Fin__Basic______macroRules__Fin__tacticFin__omega__1___closed__46_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Fin___aux__Mathlib__Data__Fin__Basic______macroRules__Fin__tacticFin__omega__1___closed__46_value_aux_2),((lean_object*)&lp_mathlib_Fin___aux__Mathlib__Data__Fin__Basic______macroRules__Fin__tacticFin__omega__1___closed__45_value),LEAN_SCALAR_PTR_LITERAL(134, 218, 71, 35, 220, 118, 132, 17)}};
static const lean_object* lp_mathlib_Fin___aux__Mathlib__Data__Fin__Basic______macroRules__Fin__tacticFin__omega__1___closed__46 = (const lean_object*)&lp_mathlib_Fin___aux__Mathlib__Data__Fin__Basic______macroRules__Fin__tacticFin__omega__1___closed__46_value;
static const lean_string_object lp_mathlib_Fin___aux__Mathlib__Data__Fin__Basic______macroRules__Fin__tacticFin__omega__1___closed__47_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = "*"};
static const lean_object* lp_mathlib_Fin___aux__Mathlib__Data__Fin__Basic______macroRules__Fin__tacticFin__omega__1___closed__47 = (const lean_object*)&lp_mathlib_Fin___aux__Mathlib__Data__Fin__Basic______macroRules__Fin__tacticFin__omega__1___closed__47_value;
static const lean_string_object lp_mathlib_Fin___aux__Mathlib__Data__Fin__Basic______macroRules__Fin__tacticFin__omega__1___closed__48_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "omega"};
static const lean_object* lp_mathlib_Fin___aux__Mathlib__Data__Fin__Basic______macroRules__Fin__tacticFin__omega__1___closed__48 = (const lean_object*)&lp_mathlib_Fin___aux__Mathlib__Data__Fin__Basic______macroRules__Fin__tacticFin__omega__1___closed__48_value;
static const lean_ctor_object lp_mathlib_Fin___aux__Mathlib__Data__Fin__Basic______macroRules__Fin__tacticFin__omega__1___closed__49_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Fin___aux__Mathlib__Data__Fin__Basic______macroRules__Fin__tacticFin__omega__1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib_Fin___aux__Mathlib__Data__Fin__Basic______macroRules__Fin__tacticFin__omega__1___closed__49_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Fin___aux__Mathlib__Data__Fin__Basic______macroRules__Fin__tacticFin__omega__1___closed__49_value_aux_0),((lean_object*)&lp_mathlib_Fin___aux__Mathlib__Data__Fin__Basic______macroRules__Fin__tacticFin__omega__1___closed__1_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib_Fin___aux__Mathlib__Data__Fin__Basic______macroRules__Fin__tacticFin__omega__1___closed__49_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Fin___aux__Mathlib__Data__Fin__Basic______macroRules__Fin__tacticFin__omega__1___closed__49_value_aux_1),((lean_object*)&lp_mathlib_Fin___aux__Mathlib__Data__Fin__Basic______macroRules__Fin__tacticFin__omega__1___closed__2_value),LEAN_SCALAR_PTR_LITERAL(166, 58, 35, 182, 187, 130, 147, 254)}};
static const lean_ctor_object lp_mathlib_Fin___aux__Mathlib__Data__Fin__Basic______macroRules__Fin__tacticFin__omega__1___closed__49_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Fin___aux__Mathlib__Data__Fin__Basic______macroRules__Fin__tacticFin__omega__1___closed__49_value_aux_2),((lean_object*)&lp_mathlib_Fin___aux__Mathlib__Data__Fin__Basic______macroRules__Fin__tacticFin__omega__1___closed__48_value),LEAN_SCALAR_PTR_LITERAL(138, 49, 229, 237, 137, 52, 176, 206)}};
static const lean_object* lp_mathlib_Fin___aux__Mathlib__Data__Fin__Basic______macroRules__Fin__tacticFin__omega__1___closed__49 = (const lean_object*)&lp_mathlib_Fin___aux__Mathlib__Data__Fin__Basic______macroRules__Fin__tacticFin__omega__1___closed__49_value;
static const lean_string_object lp_mathlib_Fin___aux__Mathlib__Data__Fin__Basic______macroRules__Fin__tacticFin__omega__1___closed__50_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = "}"};
static const lean_object* lp_mathlib_Fin___aux__Mathlib__Data__Fin__Basic______macroRules__Fin__tacticFin__omega__1___closed__50 = (const lean_object*)&lp_mathlib_Fin___aux__Mathlib__Data__Fin__Basic______macroRules__Fin__tacticFin__omega__1___closed__50_value;
LEAN_EXPORT lean_object* lp_mathlib_Fin___aux__Mathlib__Data__Fin__Basic______macroRules__Fin__tacticFin__omega__1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Fin___aux__Mathlib__Data__Fin__Basic______macroRules__Fin__tacticFin__omega__1___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Fin_inhabitedFinOneAdd(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Fin_inhabitedFinOneAdd___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_finZeroElim(lean_object* v_00_u03b1_1_, lean_object* v_x_2_){
_start:
{
lean_internal_panic_unreachable();
}
}
LEAN_EXPORT lean_object* lp_mathlib_finZeroElim___boxed(lean_object* v_00_u03b1_3_, lean_object* v_x_4_){
_start:
{
lean_object* v_res_5_; 
v_res_5_ = lp_mathlib_finZeroElim(v_00_u03b1_3_, v_x_4_);
lean_dec(v_x_4_);
return v_res_5_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Fin_rec0(lean_object* v_00_u03b1_6_, lean_object* v_i_7_){
_start:
{
lean_internal_panic_unreachable();
}
}
LEAN_EXPORT lean_object* lp_mathlib_Fin_rec0___boxed(lean_object* v_00_u03b1_8_, lean_object* v_i_9_){
_start:
{
lean_object* v_res_10_; 
v_res_10_ = lp_mathlib_Fin_rec0(v_00_u03b1_8_, v_i_9_);
lean_dec(v_i_9_);
return v_res_10_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Fin_equivSubtype___lam__0(lean_object* v_a_11_){
_start:
{
lean_inc(v_a_11_);
return v_a_11_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Fin_equivSubtype___lam__0___boxed(lean_object* v_a_12_){
_start:
{
lean_object* v_res_13_; 
v_res_13_ = lp_mathlib_Fin_equivSubtype___lam__0(v_a_12_);
lean_dec(v_a_12_);
return v_res_13_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Fin_equivSubtype(lean_object* v_n_17_){
_start:
{
lean_object* v___x_18_; 
v___x_18_ = ((lean_object*)(lp_mathlib_Fin_equivSubtype___closed__1));
return v___x_18_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Fin_equivSubtype___boxed(lean_object* v_n_19_){
_start:
{
lean_object* v_res_20_; 
v_res_20_ = lp_mathlib_Fin_equivSubtype(v_n_19_);
lean_dec(v_n_19_);
return v_res_20_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Fin_instWellFoundedRelation__mathlib(lean_object* v_n_21_){
_start:
{
lean_object* v___x_22_; 
v___x_22_ = lean_box(0);
return v___x_22_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Fin_instWellFoundedRelation__mathlib___boxed(lean_object* v_n_23_){
_start:
{
lean_object* v_res_24_; 
v_res_24_ = lp_mathlib_Fin_instWellFoundedRelation__mathlib(v_n_23_);
lean_dec(v_n_23_);
return v_res_24_;
}
}
static lean_object* _init_lp_mathlib_Fin___aux__Mathlib__Data__Fin__Basic______macroRules__Fin__tacticFin__omega__1___closed__19(void){
_start:
{
lean_object* v___x_83_; 
v___x_83_ = l_Array_mkArray0(lean_box(0));
return v___x_83_;
}
}
static lean_object* _init_lp_mathlib_Fin___aux__Mathlib__Data__Fin__Basic______macroRules__Fin__tacticFin__omega__1___closed__24(void){
_start:
{
lean_object* v___x_92_; lean_object* v___x_93_; 
v___x_92_ = ((lean_object*)(lp_mathlib_Fin_tacticFin__omega___closed__3));
v___x_93_ = l_String_toRawSubstring_x27(v___x_92_);
return v___x_93_;
}
}
static lean_object* _init_lp_mathlib_Fin___aux__Mathlib__Data__Fin__Basic______macroRules__Fin__tacticFin__omega__1___closed__29(void){
_start:
{
lean_object* v___x_99_; lean_object* v___x_100_; 
v___x_99_ = ((lean_object*)(lp_mathlib_Fin___aux__Mathlib__Data__Fin__Basic______macroRules__Fin__tacticFin__omega__1___closed__28));
v___x_100_ = l_String_toRawSubstring_x27(v___x_99_);
return v___x_100_;
}
}
static lean_object* _init_lp_mathlib_Fin___aux__Mathlib__Data__Fin__Basic______macroRules__Fin__tacticFin__omega__1___closed__36(void){
_start:
{
lean_object* v___x_113_; lean_object* v___x_114_; 
v___x_113_ = ((lean_object*)(lp_mathlib_Fin___aux__Mathlib__Data__Fin__Basic______macroRules__Fin__tacticFin__omega__1___closed__35));
v___x_114_ = l_String_toRawSubstring_x27(v___x_113_);
return v___x_114_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Fin___aux__Mathlib__Data__Fin__Basic______macroRules__Fin__tacticFin__omega__1(lean_object* v_x_147_, lean_object* v_a_148_, lean_object* v_a_149_){
_start:
{
lean_object* v___x_150_; uint8_t v___x_151_; 
v___x_150_ = ((lean_object*)(lp_mathlib_Fin_tacticFin__omega___closed__2));
v___x_151_ = l_Lean_Syntax_isOfKind(v_x_147_, v___x_150_);
if (v___x_151_ == 0)
{
lean_object* v___x_152_; lean_object* v___x_153_; 
v___x_152_ = lean_box(1);
v___x_153_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_153_, 0, v___x_152_);
lean_ctor_set(v___x_153_, 1, v_a_149_);
return v___x_153_;
}
else
{
lean_object* v_quotContext_154_; lean_object* v_currMacroScope_155_; lean_object* v_ref_156_; uint8_t v___x_157_; lean_object* v___x_158_; lean_object* v___x_159_; lean_object* v___x_160_; lean_object* v___x_161_; lean_object* v___x_162_; lean_object* v___x_163_; lean_object* v___x_164_; lean_object* v___x_165_; lean_object* v___x_166_; lean_object* v___x_167_; lean_object* v___x_168_; lean_object* v___x_169_; lean_object* v___x_170_; lean_object* v___x_171_; lean_object* v___x_172_; lean_object* v___x_173_; lean_object* v___x_174_; lean_object* v___x_175_; lean_object* v___x_176_; lean_object* v___x_177_; lean_object* v___x_178_; lean_object* v___x_179_; lean_object* v___x_180_; lean_object* v___x_181_; lean_object* v___x_182_; lean_object* v___x_183_; lean_object* v___x_184_; lean_object* v___x_185_; lean_object* v___x_186_; lean_object* v___x_187_; lean_object* v___x_188_; lean_object* v___x_189_; lean_object* v___x_190_; lean_object* v___x_191_; lean_object* v___x_192_; lean_object* v___x_193_; lean_object* v___x_194_; lean_object* v___x_195_; lean_object* v___x_196_; lean_object* v___x_197_; lean_object* v___x_198_; lean_object* v___x_199_; lean_object* v___x_200_; lean_object* v___x_201_; lean_object* v___x_202_; lean_object* v___x_203_; lean_object* v___x_204_; lean_object* v___x_205_; lean_object* v___x_206_; lean_object* v___x_207_; lean_object* v___x_208_; lean_object* v___x_209_; lean_object* v___x_210_; lean_object* v___x_211_; lean_object* v___x_212_; lean_object* v___x_213_; lean_object* v___x_214_; lean_object* v___x_215_; lean_object* v___x_216_; lean_object* v___x_217_; lean_object* v___x_218_; lean_object* v___x_219_; lean_object* v___x_220_; lean_object* v___x_221_; lean_object* v___x_222_; lean_object* v___x_223_; lean_object* v___x_224_; lean_object* v___x_225_; lean_object* v___x_226_; lean_object* v___x_227_; lean_object* v___x_228_; lean_object* v___x_229_; lean_object* v___x_230_; 
v_quotContext_154_ = lean_ctor_get(v_a_148_, 1);
v_currMacroScope_155_ = lean_ctor_get(v_a_148_, 2);
v_ref_156_ = lean_ctor_get(v_a_148_, 5);
v___x_157_ = 0;
v___x_158_ = l_Lean_SourceInfo_fromRef(v_ref_156_, v___x_157_);
v___x_159_ = ((lean_object*)(lp_mathlib_Fin___aux__Mathlib__Data__Fin__Basic______macroRules__Fin__tacticFin__omega__1___closed__4));
v___x_160_ = ((lean_object*)(lp_mathlib_Fin___aux__Mathlib__Data__Fin__Basic______macroRules__Fin__tacticFin__omega__1___closed__5));
lean_inc_n(v___x_158_, 34);
v___x_161_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_161_, 0, v___x_158_);
lean_ctor_set(v___x_161_, 1, v___x_160_);
v___x_162_ = ((lean_object*)(lp_mathlib_Fin___aux__Mathlib__Data__Fin__Basic______macroRules__Fin__tacticFin__omega__1___closed__7));
v___x_163_ = ((lean_object*)(lp_mathlib_Fin___aux__Mathlib__Data__Fin__Basic______macroRules__Fin__tacticFin__omega__1___closed__9));
v___x_164_ = ((lean_object*)(lp_mathlib_Fin___aux__Mathlib__Data__Fin__Basic______macroRules__Fin__tacticFin__omega__1___closed__10));
v___x_165_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_165_, 0, v___x_158_);
lean_ctor_set(v___x_165_, 1, v___x_164_);
v___x_166_ = ((lean_object*)(lp_mathlib_Fin___aux__Mathlib__Data__Fin__Basic______macroRules__Fin__tacticFin__omega__1___closed__12));
v___x_167_ = ((lean_object*)(lp_mathlib_Fin___aux__Mathlib__Data__Fin__Basic______macroRules__Fin__tacticFin__omega__1___closed__14));
v___x_168_ = ((lean_object*)(lp_mathlib_Fin___aux__Mathlib__Data__Fin__Basic______macroRules__Fin__tacticFin__omega__1___closed__15));
v___x_169_ = ((lean_object*)(lp_mathlib_Fin___aux__Mathlib__Data__Fin__Basic______macroRules__Fin__tacticFin__omega__1___closed__16));
v___x_170_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_170_, 0, v___x_158_);
lean_ctor_set(v___x_170_, 1, v___x_168_);
v___x_171_ = ((lean_object*)(lp_mathlib_Fin___aux__Mathlib__Data__Fin__Basic______macroRules__Fin__tacticFin__omega__1___closed__18));
v___x_172_ = lean_obj_once(&lp_mathlib_Fin___aux__Mathlib__Data__Fin__Basic______macroRules__Fin__tacticFin__omega__1___closed__19, &lp_mathlib_Fin___aux__Mathlib__Data__Fin__Basic______macroRules__Fin__tacticFin__omega__1___closed__19_once, _init_lp_mathlib_Fin___aux__Mathlib__Data__Fin__Basic______macroRules__Fin__tacticFin__omega__1___closed__19);
v___x_173_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_173_, 0, v___x_158_);
lean_ctor_set(v___x_173_, 1, v___x_162_);
lean_ctor_set(v___x_173_, 2, v___x_172_);
lean_inc_ref_n(v___x_173_, 6);
v___x_174_ = l_Lean_Syntax_node1(v___x_158_, v___x_171_, v___x_173_);
v___x_175_ = ((lean_object*)(lp_mathlib_Fin___aux__Mathlib__Data__Fin__Basic______macroRules__Fin__tacticFin__omega__1___closed__20));
v___x_176_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_176_, 0, v___x_158_);
lean_ctor_set(v___x_176_, 1, v___x_175_);
v___x_177_ = l_Lean_Syntax_node1(v___x_158_, v___x_162_, v___x_176_);
v___x_178_ = ((lean_object*)(lp_mathlib_Fin___aux__Mathlib__Data__Fin__Basic______macroRules__Fin__tacticFin__omega__1___closed__21));
v___x_179_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_179_, 0, v___x_158_);
lean_ctor_set(v___x_179_, 1, v___x_178_);
v___x_180_ = ((lean_object*)(lp_mathlib_Fin___aux__Mathlib__Data__Fin__Basic______macroRules__Fin__tacticFin__omega__1___closed__23));
v___x_181_ = lean_obj_once(&lp_mathlib_Fin___aux__Mathlib__Data__Fin__Basic______macroRules__Fin__tacticFin__omega__1___closed__24, &lp_mathlib_Fin___aux__Mathlib__Data__Fin__Basic______macroRules__Fin__tacticFin__omega__1___closed__24_once, _init_lp_mathlib_Fin___aux__Mathlib__Data__Fin__Basic______macroRules__Fin__tacticFin__omega__1___closed__24);
v___x_182_ = ((lean_object*)(lp_mathlib_Fin___aux__Mathlib__Data__Fin__Basic______macroRules__Fin__tacticFin__omega__1___closed__25));
lean_inc_n(v_currMacroScope_155_, 3);
lean_inc_n(v_quotContext_154_, 3);
v___x_183_ = l_Lean_addMacroScope(v_quotContext_154_, v___x_182_, v_currMacroScope_155_);
v___x_184_ = lean_box(0);
v___x_185_ = lean_alloc_ctor(3, 4, 0);
lean_ctor_set(v___x_185_, 0, v___x_158_);
lean_ctor_set(v___x_185_, 1, v___x_181_);
lean_ctor_set(v___x_185_, 2, v___x_183_);
lean_ctor_set(v___x_185_, 3, v___x_184_);
v___x_186_ = l_Lean_Syntax_node3(v___x_158_, v___x_180_, v___x_173_, v___x_173_, v___x_185_);
v___x_187_ = ((lean_object*)(lp_mathlib_Fin___aux__Mathlib__Data__Fin__Basic______macroRules__Fin__tacticFin__omega__1___closed__26));
v___x_188_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_188_, 0, v___x_158_);
lean_ctor_set(v___x_188_, 1, v___x_187_);
v___x_189_ = ((lean_object*)(lp_mathlib_Fin___aux__Mathlib__Data__Fin__Basic______macroRules__Fin__tacticFin__omega__1___closed__27));
v___x_190_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_190_, 0, v___x_158_);
lean_ctor_set(v___x_190_, 1, v___x_189_);
v___x_191_ = l_Lean_Syntax_node1(v___x_158_, v___x_162_, v___x_190_);
v___x_192_ = lean_obj_once(&lp_mathlib_Fin___aux__Mathlib__Data__Fin__Basic______macroRules__Fin__tacticFin__omega__1___closed__29, &lp_mathlib_Fin___aux__Mathlib__Data__Fin__Basic______macroRules__Fin__tacticFin__omega__1___closed__29_once, _init_lp_mathlib_Fin___aux__Mathlib__Data__Fin__Basic______macroRules__Fin__tacticFin__omega__1___closed__29);
v___x_193_ = ((lean_object*)(lp_mathlib_Fin___aux__Mathlib__Data__Fin__Basic______macroRules__Fin__tacticFin__omega__1___closed__32));
v___x_194_ = l_Lean_addMacroScope(v_quotContext_154_, v___x_193_, v_currMacroScope_155_);
v___x_195_ = ((lean_object*)(lp_mathlib_Fin___aux__Mathlib__Data__Fin__Basic______macroRules__Fin__tacticFin__omega__1___closed__34));
v___x_196_ = lean_alloc_ctor(3, 4, 0);
lean_ctor_set(v___x_196_, 0, v___x_158_);
lean_ctor_set(v___x_196_, 1, v___x_192_);
lean_ctor_set(v___x_196_, 2, v___x_194_);
lean_ctor_set(v___x_196_, 3, v___x_195_);
lean_inc(v___x_191_);
v___x_197_ = l_Lean_Syntax_node3(v___x_158_, v___x_180_, v___x_173_, v___x_191_, v___x_196_);
v___x_198_ = lean_obj_once(&lp_mathlib_Fin___aux__Mathlib__Data__Fin__Basic______macroRules__Fin__tacticFin__omega__1___closed__36, &lp_mathlib_Fin___aux__Mathlib__Data__Fin__Basic______macroRules__Fin__tacticFin__omega__1___closed__36_once, _init_lp_mathlib_Fin___aux__Mathlib__Data__Fin__Basic______macroRules__Fin__tacticFin__omega__1___closed__36);
v___x_199_ = ((lean_object*)(lp_mathlib_Fin___aux__Mathlib__Data__Fin__Basic______macroRules__Fin__tacticFin__omega__1___closed__38));
v___x_200_ = l_Lean_addMacroScope(v_quotContext_154_, v___x_199_, v_currMacroScope_155_);
v___x_201_ = ((lean_object*)(lp_mathlib_Fin___aux__Mathlib__Data__Fin__Basic______macroRules__Fin__tacticFin__omega__1___closed__40));
v___x_202_ = lean_alloc_ctor(3, 4, 0);
lean_ctor_set(v___x_202_, 0, v___x_158_);
lean_ctor_set(v___x_202_, 1, v___x_198_);
lean_ctor_set(v___x_202_, 2, v___x_200_);
lean_ctor_set(v___x_202_, 3, v___x_201_);
v___x_203_ = l_Lean_Syntax_node3(v___x_158_, v___x_180_, v___x_173_, v___x_191_, v___x_202_);
lean_inc_ref(v___x_188_);
v___x_204_ = l_Lean_Syntax_node5(v___x_158_, v___x_162_, v___x_186_, v___x_188_, v___x_197_, v___x_188_, v___x_203_);
v___x_205_ = ((lean_object*)(lp_mathlib_Fin___aux__Mathlib__Data__Fin__Basic______macroRules__Fin__tacticFin__omega__1___closed__41));
v___x_206_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_206_, 0, v___x_158_);
lean_ctor_set(v___x_206_, 1, v___x_205_);
v___x_207_ = l_Lean_Syntax_node3(v___x_158_, v___x_162_, v___x_179_, v___x_204_, v___x_206_);
v___x_208_ = ((lean_object*)(lp_mathlib_Fin___aux__Mathlib__Data__Fin__Basic______macroRules__Fin__tacticFin__omega__1___closed__43));
v___x_209_ = ((lean_object*)(lp_mathlib_Fin___aux__Mathlib__Data__Fin__Basic______macroRules__Fin__tacticFin__omega__1___closed__44));
v___x_210_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_210_, 0, v___x_158_);
lean_ctor_set(v___x_210_, 1, v___x_209_);
v___x_211_ = ((lean_object*)(lp_mathlib_Fin___aux__Mathlib__Data__Fin__Basic______macroRules__Fin__tacticFin__omega__1___closed__46));
v___x_212_ = ((lean_object*)(lp_mathlib_Fin___aux__Mathlib__Data__Fin__Basic______macroRules__Fin__tacticFin__omega__1___closed__47));
v___x_213_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_213_, 0, v___x_158_);
lean_ctor_set(v___x_213_, 1, v___x_212_);
v___x_214_ = l_Lean_Syntax_node1(v___x_158_, v___x_211_, v___x_213_);
v___x_215_ = l_Lean_Syntax_node2(v___x_158_, v___x_208_, v___x_210_, v___x_214_);
v___x_216_ = l_Lean_Syntax_node1(v___x_158_, v___x_162_, v___x_215_);
lean_inc(v___x_174_);
v___x_217_ = l_Lean_Syntax_node6(v___x_158_, v___x_169_, v___x_170_, v___x_174_, v___x_173_, v___x_177_, v___x_207_, v___x_216_);
v___x_218_ = l_Lean_Syntax_node1(v___x_158_, v___x_162_, v___x_217_);
v___x_219_ = l_Lean_Syntax_node1(v___x_158_, v___x_167_, v___x_218_);
v___x_220_ = l_Lean_Syntax_node1(v___x_158_, v___x_166_, v___x_219_);
v___x_221_ = l_Lean_Syntax_node2(v___x_158_, v___x_163_, v___x_165_, v___x_220_);
v___x_222_ = ((lean_object*)(lp_mathlib_Fin___aux__Mathlib__Data__Fin__Basic______macroRules__Fin__tacticFin__omega__1___closed__48));
v___x_223_ = ((lean_object*)(lp_mathlib_Fin___aux__Mathlib__Data__Fin__Basic______macroRules__Fin__tacticFin__omega__1___closed__49));
v___x_224_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_224_, 0, v___x_158_);
lean_ctor_set(v___x_224_, 1, v___x_222_);
v___x_225_ = l_Lean_Syntax_node2(v___x_158_, v___x_223_, v___x_224_, v___x_174_);
v___x_226_ = l_Lean_Syntax_node3(v___x_158_, v___x_162_, v___x_221_, v___x_173_, v___x_225_);
v___x_227_ = ((lean_object*)(lp_mathlib_Fin___aux__Mathlib__Data__Fin__Basic______macroRules__Fin__tacticFin__omega__1___closed__50));
v___x_228_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_228_, 0, v___x_158_);
lean_ctor_set(v___x_228_, 1, v___x_227_);
v___x_229_ = l_Lean_Syntax_node3(v___x_158_, v___x_159_, v___x_161_, v___x_226_, v___x_228_);
v___x_230_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_230_, 0, v___x_229_);
lean_ctor_set(v___x_230_, 1, v_a_149_);
return v___x_230_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Fin___aux__Mathlib__Data__Fin__Basic______macroRules__Fin__tacticFin__omega__1___boxed(lean_object* v_x_231_, lean_object* v_a_232_, lean_object* v_a_233_){
_start:
{
lean_object* v_res_234_; 
v_res_234_ = lp_mathlib_Fin___aux__Mathlib__Data__Fin__Basic______macroRules__Fin__tacticFin__omega__1(v_x_231_, v_a_232_, v_a_233_);
lean_dec_ref(v_a_232_);
return v_res_234_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Fin_inhabitedFinOneAdd(lean_object* v_n_235_){
_start:
{
lean_object* v___x_236_; lean_object* v___x_237_; lean_object* v___x_238_; 
v___x_236_ = lean_unsigned_to_nat(1u);
v___x_237_ = lean_nat_add(v___x_236_, v_n_235_);
v___x_238_ = l_Fin_instInhabited___redArg(v___x_237_);
lean_dec(v___x_237_);
return v___x_238_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Fin_inhabitedFinOneAdd___boxed(lean_object* v_n_239_){
_start:
{
lean_object* v_res_240_; 
v_res_240_ = lp_mathlib_Fin_inhabitedFinOneAdd(v_n_239_);
lean_dec(v_n_239_);
return v_res_240_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Data_Int_DivMod(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Data_Nat_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Logic_Equiv_Defs(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Tactic_Common(uint8_t builtin);
lean_object* runtime_initialize_batteries_Batteries_Data_Fin_Basic(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Tactic_Attr_Core(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_Data_Fin_Basic(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Data_Int_DivMod(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Data_Nat_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Logic_Equiv_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Tactic_Common(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_batteries_Batteries_Data_Fin_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Tactic_Attr_Core(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_Data_Fin_Basic(uint8_t builtin) {
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
lean_object* initialize_mathlib_Mathlib_Data_Int_DivMod(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Data_Nat_Init(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Logic_Equiv_Defs(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Tactic_Common(uint8_t builtin);
lean_object* initialize_batteries_Batteries_Data_Fin_Basic(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Tactic_Attr_Core(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Data_Fin_Basic(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_Int_DivMod(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_Nat_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Logic_Equiv_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Tactic_Common(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_batteries_Batteries_Data_Fin_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Tactic_Attr_Core(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Data_Fin_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_Data_Fin_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_Data_Fin_Basic(builtin);
}
#ifdef __cplusplus
}
#endif
