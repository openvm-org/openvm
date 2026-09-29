// Lean compiler output
// Module: Mathlib.Data.Vector.Basic
// Imports: public import Init public meta import Init public import Mathlib.Data.Vector.Defs public import Mathlib.Data.List.Nodup public import Mathlib.Control.Applicative public import Mathlib.Control.Traversable.Basic public import Mathlib.Algebra.BigOperators.Group.List.Basic public import Batteries.Data.Fin.Lemmas public import Mathlib.Data.Fin.SuccPred
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
lean_object* l_Nat_recCompiled___redArg(lean_object*, lean_object*, lean_object*);
lean_object* l_String_toRawSubstring_x27(lean_object*);
uint8_t lean_nat_dec_eq(lean_object*, lean_object*);
lean_object* lean_nat_sub(lean_object*, lean_object*);
lean_object* l_Function_const___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_List_Vector_map___redArg(lean_object*, lean_object*);
lean_object* l_Lean_Name_mkStr3(lean_object*, lean_object*, lean_object*);
lean_object* lean_nat_add(lean_object*, lean_object*);
uint8_t l_Lean_Syntax_isOfKind(lean_object*, lean_object*);
lean_object* l_Lean_Syntax_getArg(lean_object*, lean_object*);
lean_object* l_Lean_SourceInfo_fromRef(lean_object*, uint8_t);
lean_object* l_Lean_Name_mkStr4(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Name_mkStr2(lean_object*, lean_object*);
lean_object* l_Lean_addMacroScope(lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Name_mkStr1(lean_object*);
lean_object* l_Lean_Syntax_node2(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Fin_succ___redArg(lean_object*);
lean_object* lean_nat_mod(lean_object*, lean_object*);
lean_object* lp_mathlib_List_Vector_ofFn___redArg(lean_object*, lean_object*);
lean_object* lean_array_mk(lean_object*);
lean_object* l_List_reverse___redArg(lean_object*);
lean_object* l_List_lengthTR___redArg(lean_object*);
lean_object* lp_mathlib_List_Vector_cons___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
uint8_t l_Lean_Syntax_matchesNull(lean_object*, lean_object*);
lean_object* l_Lean_replaceRef(lean_object*, lean_object*);
lean_object* l_Lean_Syntax_node3(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lean_mk_empty_array_with_capacity(lean_object*);
lean_object* l___private_Init_Data_List_Impl_0__List_setTR_go(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l___private_Init_Data_List_Impl_0__List_insertIdxTR_go(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_List_Vector_get___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_List_Vector_ofFn___boxed(lean_object*, lean_object*, lean_object*);
lean_object* l_List_get___redArg(lean_object*, lean_object*);
static const lean_string_object lp_mathlib_List_Vector_term___x3a_x3a_u1d65___00__closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "List"};
static const lean_object* lp_mathlib_List_Vector_term___x3a_x3a_u1d65___00__closed__0 = (const lean_object*)&lp_mathlib_List_Vector_term___x3a_x3a_u1d65___00__closed__0_value;
static const lean_string_object lp_mathlib_List_Vector_term___x3a_x3a_u1d65___00__closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "Vector"};
static const lean_object* lp_mathlib_List_Vector_term___x3a_x3a_u1d65___00__closed__1 = (const lean_object*)&lp_mathlib_List_Vector_term___x3a_x3a_u1d65___00__closed__1_value;
static const lean_string_object lp_mathlib_List_Vector_term___x3a_x3a_u1d65___00__closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 12, .m_capacity = 12, .m_length = 9, .m_data = "term_::ᵥ_"};
static const lean_object* lp_mathlib_List_Vector_term___x3a_x3a_u1d65___00__closed__2 = (const lean_object*)&lp_mathlib_List_Vector_term___x3a_x3a_u1d65___00__closed__2_value;
static const lean_ctor_object lp_mathlib_List_Vector_term___x3a_x3a_u1d65___00__closed__3_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_List_Vector_term___x3a_x3a_u1d65___00__closed__0_value),LEAN_SCALAR_PTR_LITERAL(245, 188, 225, 225, 165, 5, 251, 132)}};
static const lean_ctor_object lp_mathlib_List_Vector_term___x3a_x3a_u1d65___00__closed__3_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_List_Vector_term___x3a_x3a_u1d65___00__closed__3_value_aux_0),((lean_object*)&lp_mathlib_List_Vector_term___x3a_x3a_u1d65___00__closed__1_value),LEAN_SCALAR_PTR_LITERAL(191, 151, 237, 53, 167, 179, 230, 56)}};
static const lean_ctor_object lp_mathlib_List_Vector_term___x3a_x3a_u1d65___00__closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_List_Vector_term___x3a_x3a_u1d65___00__closed__3_value_aux_1),((lean_object*)&lp_mathlib_List_Vector_term___x3a_x3a_u1d65___00__closed__2_value),LEAN_SCALAR_PTR_LITERAL(154, 138, 9, 14, 207, 96, 187, 34)}};
static const lean_object* lp_mathlib_List_Vector_term___x3a_x3a_u1d65___00__closed__3 = (const lean_object*)&lp_mathlib_List_Vector_term___x3a_x3a_u1d65___00__closed__3_value;
static const lean_string_object lp_mathlib_List_Vector_term___x3a_x3a_u1d65___00__closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "andthen"};
static const lean_object* lp_mathlib_List_Vector_term___x3a_x3a_u1d65___00__closed__4 = (const lean_object*)&lp_mathlib_List_Vector_term___x3a_x3a_u1d65___00__closed__4_value;
static const lean_ctor_object lp_mathlib_List_Vector_term___x3a_x3a_u1d65___00__closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_List_Vector_term___x3a_x3a_u1d65___00__closed__4_value),LEAN_SCALAR_PTR_LITERAL(40, 255, 78, 30, 143, 119, 117, 174)}};
static const lean_object* lp_mathlib_List_Vector_term___x3a_x3a_u1d65___00__closed__5 = (const lean_object*)&lp_mathlib_List_Vector_term___x3a_x3a_u1d65___00__closed__5_value;
static const lean_string_object lp_mathlib_List_Vector_term___x3a_x3a_u1d65___00__closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 5, .m_data = " ::ᵥ "};
static const lean_object* lp_mathlib_List_Vector_term___x3a_x3a_u1d65___00__closed__6 = (const lean_object*)&lp_mathlib_List_Vector_term___x3a_x3a_u1d65___00__closed__6_value;
static const lean_ctor_object lp_mathlib_List_Vector_term___x3a_x3a_u1d65___00__closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 5}, .m_objs = {((lean_object*)&lp_mathlib_List_Vector_term___x3a_x3a_u1d65___00__closed__6_value)}};
static const lean_object* lp_mathlib_List_Vector_term___x3a_x3a_u1d65___00__closed__7 = (const lean_object*)&lp_mathlib_List_Vector_term___x3a_x3a_u1d65___00__closed__7_value;
static const lean_string_object lp_mathlib_List_Vector_term___x3a_x3a_u1d65___00__closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "term"};
static const lean_object* lp_mathlib_List_Vector_term___x3a_x3a_u1d65___00__closed__8 = (const lean_object*)&lp_mathlib_List_Vector_term___x3a_x3a_u1d65___00__closed__8_value;
static const lean_ctor_object lp_mathlib_List_Vector_term___x3a_x3a_u1d65___00__closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_List_Vector_term___x3a_x3a_u1d65___00__closed__8_value),LEAN_SCALAR_PTR_LITERAL(187, 230, 181, 162, 253, 146, 122, 119)}};
static const lean_object* lp_mathlib_List_Vector_term___x3a_x3a_u1d65___00__closed__9 = (const lean_object*)&lp_mathlib_List_Vector_term___x3a_x3a_u1d65___00__closed__9_value;
static const lean_ctor_object lp_mathlib_List_Vector_term___x3a_x3a_u1d65___00__closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 7}, .m_objs = {((lean_object*)&lp_mathlib_List_Vector_term___x3a_x3a_u1d65___00__closed__9_value),((lean_object*)(((size_t)(67) << 1) | 1))}};
static const lean_object* lp_mathlib_List_Vector_term___x3a_x3a_u1d65___00__closed__10 = (const lean_object*)&lp_mathlib_List_Vector_term___x3a_x3a_u1d65___00__closed__10_value;
static const lean_ctor_object lp_mathlib_List_Vector_term___x3a_x3a_u1d65___00__closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib_List_Vector_term___x3a_x3a_u1d65___00__closed__5_value),((lean_object*)&lp_mathlib_List_Vector_term___x3a_x3a_u1d65___00__closed__7_value),((lean_object*)&lp_mathlib_List_Vector_term___x3a_x3a_u1d65___00__closed__10_value)}};
static const lean_object* lp_mathlib_List_Vector_term___x3a_x3a_u1d65___00__closed__11 = (const lean_object*)&lp_mathlib_List_Vector_term___x3a_x3a_u1d65___00__closed__11_value;
static const lean_ctor_object lp_mathlib_List_Vector_term___x3a_x3a_u1d65___00__closed__12_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*4 + 0, .m_other = 4, .m_tag = 4}, .m_objs = {((lean_object*)&lp_mathlib_List_Vector_term___x3a_x3a_u1d65___00__closed__3_value),((lean_object*)(((size_t)(67) << 1) | 1)),((lean_object*)(((size_t)(68) << 1) | 1)),((lean_object*)&lp_mathlib_List_Vector_term___x3a_x3a_u1d65___00__closed__11_value)}};
static const lean_object* lp_mathlib_List_Vector_term___x3a_x3a_u1d65___00__closed__12 = (const lean_object*)&lp_mathlib_List_Vector_term___x3a_x3a_u1d65___00__closed__12_value;
LEAN_EXPORT const lean_object* lp_mathlib_List_Vector_term___x3a_x3a_u1d65__ = (const lean_object*)&lp_mathlib_List_Vector_term___x3a_x3a_u1d65___00__closed__12_value;
static const lean_string_object lp_mathlib_List_Vector___aux__Mathlib__Data__Vector__Basic______macroRules__List__Vector__term___x3a_x3a_u1d65____1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "Lean"};
static const lean_object* lp_mathlib_List_Vector___aux__Mathlib__Data__Vector__Basic______macroRules__List__Vector__term___x3a_x3a_u1d65____1___closed__0 = (const lean_object*)&lp_mathlib_List_Vector___aux__Mathlib__Data__Vector__Basic______macroRules__List__Vector__term___x3a_x3a_u1d65____1___closed__0_value;
static const lean_string_object lp_mathlib_List_Vector___aux__Mathlib__Data__Vector__Basic______macroRules__List__Vector__term___x3a_x3a_u1d65____1___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "Parser"};
static const lean_object* lp_mathlib_List_Vector___aux__Mathlib__Data__Vector__Basic______macroRules__List__Vector__term___x3a_x3a_u1d65____1___closed__1 = (const lean_object*)&lp_mathlib_List_Vector___aux__Mathlib__Data__Vector__Basic______macroRules__List__Vector__term___x3a_x3a_u1d65____1___closed__1_value;
static const lean_string_object lp_mathlib_List_Vector___aux__Mathlib__Data__Vector__Basic______macroRules__List__Vector__term___x3a_x3a_u1d65____1___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "Term"};
static const lean_object* lp_mathlib_List_Vector___aux__Mathlib__Data__Vector__Basic______macroRules__List__Vector__term___x3a_x3a_u1d65____1___closed__2 = (const lean_object*)&lp_mathlib_List_Vector___aux__Mathlib__Data__Vector__Basic______macroRules__List__Vector__term___x3a_x3a_u1d65____1___closed__2_value;
static const lean_string_object lp_mathlib_List_Vector___aux__Mathlib__Data__Vector__Basic______macroRules__List__Vector__term___x3a_x3a_u1d65____1___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = "app"};
static const lean_object* lp_mathlib_List_Vector___aux__Mathlib__Data__Vector__Basic______macroRules__List__Vector__term___x3a_x3a_u1d65____1___closed__3 = (const lean_object*)&lp_mathlib_List_Vector___aux__Mathlib__Data__Vector__Basic______macroRules__List__Vector__term___x3a_x3a_u1d65____1___closed__3_value;
static const lean_ctor_object lp_mathlib_List_Vector___aux__Mathlib__Data__Vector__Basic______macroRules__List__Vector__term___x3a_x3a_u1d65____1___closed__4_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_List_Vector___aux__Mathlib__Data__Vector__Basic______macroRules__List__Vector__term___x3a_x3a_u1d65____1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib_List_Vector___aux__Mathlib__Data__Vector__Basic______macroRules__List__Vector__term___x3a_x3a_u1d65____1___closed__4_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_List_Vector___aux__Mathlib__Data__Vector__Basic______macroRules__List__Vector__term___x3a_x3a_u1d65____1___closed__4_value_aux_0),((lean_object*)&lp_mathlib_List_Vector___aux__Mathlib__Data__Vector__Basic______macroRules__List__Vector__term___x3a_x3a_u1d65____1___closed__1_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib_List_Vector___aux__Mathlib__Data__Vector__Basic______macroRules__List__Vector__term___x3a_x3a_u1d65____1___closed__4_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_List_Vector___aux__Mathlib__Data__Vector__Basic______macroRules__List__Vector__term___x3a_x3a_u1d65____1___closed__4_value_aux_1),((lean_object*)&lp_mathlib_List_Vector___aux__Mathlib__Data__Vector__Basic______macroRules__List__Vector__term___x3a_x3a_u1d65____1___closed__2_value),LEAN_SCALAR_PTR_LITERAL(75, 170, 162, 138, 136, 204, 251, 229)}};
static const lean_ctor_object lp_mathlib_List_Vector___aux__Mathlib__Data__Vector__Basic______macroRules__List__Vector__term___x3a_x3a_u1d65____1___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_List_Vector___aux__Mathlib__Data__Vector__Basic______macroRules__List__Vector__term___x3a_x3a_u1d65____1___closed__4_value_aux_2),((lean_object*)&lp_mathlib_List_Vector___aux__Mathlib__Data__Vector__Basic______macroRules__List__Vector__term___x3a_x3a_u1d65____1___closed__3_value),LEAN_SCALAR_PTR_LITERAL(69, 118, 10, 41, 220, 156, 243, 179)}};
static const lean_object* lp_mathlib_List_Vector___aux__Mathlib__Data__Vector__Basic______macroRules__List__Vector__term___x3a_x3a_u1d65____1___closed__4 = (const lean_object*)&lp_mathlib_List_Vector___aux__Mathlib__Data__Vector__Basic______macroRules__List__Vector__term___x3a_x3a_u1d65____1___closed__4_value;
static const lean_string_object lp_mathlib_List_Vector___aux__Mathlib__Data__Vector__Basic______macroRules__List__Vector__term___x3a_x3a_u1d65____1___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 12, .m_capacity = 12, .m_length = 11, .m_data = "Vector.cons"};
static const lean_object* lp_mathlib_List_Vector___aux__Mathlib__Data__Vector__Basic______macroRules__List__Vector__term___x3a_x3a_u1d65____1___closed__5 = (const lean_object*)&lp_mathlib_List_Vector___aux__Mathlib__Data__Vector__Basic______macroRules__List__Vector__term___x3a_x3a_u1d65____1___closed__5_value;
static lean_once_cell_t lp_mathlib_List_Vector___aux__Mathlib__Data__Vector__Basic______macroRules__List__Vector__term___x3a_x3a_u1d65____1___closed__6_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_List_Vector___aux__Mathlib__Data__Vector__Basic______macroRules__List__Vector__term___x3a_x3a_u1d65____1___closed__6;
static const lean_string_object lp_mathlib_List_Vector___aux__Mathlib__Data__Vector__Basic______macroRules__List__Vector__term___x3a_x3a_u1d65____1___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "cons"};
static const lean_object* lp_mathlib_List_Vector___aux__Mathlib__Data__Vector__Basic______macroRules__List__Vector__term___x3a_x3a_u1d65____1___closed__7 = (const lean_object*)&lp_mathlib_List_Vector___aux__Mathlib__Data__Vector__Basic______macroRules__List__Vector__term___x3a_x3a_u1d65____1___closed__7_value;
static const lean_ctor_object lp_mathlib_List_Vector___aux__Mathlib__Data__Vector__Basic______macroRules__List__Vector__term___x3a_x3a_u1d65____1___closed__8_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_List_Vector_term___x3a_x3a_u1d65___00__closed__1_value),LEAN_SCALAR_PTR_LITERAL(209, 122, 98, 30, 71, 224, 237, 30)}};
static const lean_ctor_object lp_mathlib_List_Vector___aux__Mathlib__Data__Vector__Basic______macroRules__List__Vector__term___x3a_x3a_u1d65____1___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_List_Vector___aux__Mathlib__Data__Vector__Basic______macroRules__List__Vector__term___x3a_x3a_u1d65____1___closed__8_value_aux_0),((lean_object*)&lp_mathlib_List_Vector___aux__Mathlib__Data__Vector__Basic______macroRules__List__Vector__term___x3a_x3a_u1d65____1___closed__7_value),LEAN_SCALAR_PTR_LITERAL(86, 153, 61, 143, 8, 88, 148, 207)}};
static const lean_object* lp_mathlib_List_Vector___aux__Mathlib__Data__Vector__Basic______macroRules__List__Vector__term___x3a_x3a_u1d65____1___closed__8 = (const lean_object*)&lp_mathlib_List_Vector___aux__Mathlib__Data__Vector__Basic______macroRules__List__Vector__term___x3a_x3a_u1d65____1___closed__8_value;
static const lean_ctor_object lp_mathlib_List_Vector___aux__Mathlib__Data__Vector__Basic______macroRules__List__Vector__term___x3a_x3a_u1d65____1___closed__9_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_List_Vector_term___x3a_x3a_u1d65___00__closed__0_value),LEAN_SCALAR_PTR_LITERAL(245, 188, 225, 225, 165, 5, 251, 132)}};
static const lean_ctor_object lp_mathlib_List_Vector___aux__Mathlib__Data__Vector__Basic______macroRules__List__Vector__term___x3a_x3a_u1d65____1___closed__9_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_List_Vector___aux__Mathlib__Data__Vector__Basic______macroRules__List__Vector__term___x3a_x3a_u1d65____1___closed__9_value_aux_0),((lean_object*)&lp_mathlib_List_Vector_term___x3a_x3a_u1d65___00__closed__1_value),LEAN_SCALAR_PTR_LITERAL(191, 151, 237, 53, 167, 179, 230, 56)}};
static const lean_ctor_object lp_mathlib_List_Vector___aux__Mathlib__Data__Vector__Basic______macroRules__List__Vector__term___x3a_x3a_u1d65____1___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_List_Vector___aux__Mathlib__Data__Vector__Basic______macroRules__List__Vector__term___x3a_x3a_u1d65____1___closed__9_value_aux_1),((lean_object*)&lp_mathlib_List_Vector___aux__Mathlib__Data__Vector__Basic______macroRules__List__Vector__term___x3a_x3a_u1d65____1___closed__7_value),LEAN_SCALAR_PTR_LITERAL(80, 220, 132, 186, 253, 230, 113, 239)}};
static const lean_object* lp_mathlib_List_Vector___aux__Mathlib__Data__Vector__Basic______macroRules__List__Vector__term___x3a_x3a_u1d65____1___closed__9 = (const lean_object*)&lp_mathlib_List_Vector___aux__Mathlib__Data__Vector__Basic______macroRules__List__Vector__term___x3a_x3a_u1d65____1___closed__9_value;
static const lean_ctor_object lp_mathlib_List_Vector___aux__Mathlib__Data__Vector__Basic______macroRules__List__Vector__term___x3a_x3a_u1d65____1___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_List_Vector___aux__Mathlib__Data__Vector__Basic______macroRules__List__Vector__term___x3a_x3a_u1d65____1___closed__9_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_List_Vector___aux__Mathlib__Data__Vector__Basic______macroRules__List__Vector__term___x3a_x3a_u1d65____1___closed__10 = (const lean_object*)&lp_mathlib_List_Vector___aux__Mathlib__Data__Vector__Basic______macroRules__List__Vector__term___x3a_x3a_u1d65____1___closed__10_value;
static const lean_ctor_object lp_mathlib_List_Vector___aux__Mathlib__Data__Vector__Basic______macroRules__List__Vector__term___x3a_x3a_u1d65____1___closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_List_Vector___aux__Mathlib__Data__Vector__Basic______macroRules__List__Vector__term___x3a_x3a_u1d65____1___closed__9_value)}};
static const lean_object* lp_mathlib_List_Vector___aux__Mathlib__Data__Vector__Basic______macroRules__List__Vector__term___x3a_x3a_u1d65____1___closed__11 = (const lean_object*)&lp_mathlib_List_Vector___aux__Mathlib__Data__Vector__Basic______macroRules__List__Vector__term___x3a_x3a_u1d65____1___closed__11_value;
static const lean_ctor_object lp_mathlib_List_Vector___aux__Mathlib__Data__Vector__Basic______macroRules__List__Vector__term___x3a_x3a_u1d65____1___closed__12_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_List_Vector___aux__Mathlib__Data__Vector__Basic______macroRules__List__Vector__term___x3a_x3a_u1d65____1___closed__11_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_List_Vector___aux__Mathlib__Data__Vector__Basic______macroRules__List__Vector__term___x3a_x3a_u1d65____1___closed__12 = (const lean_object*)&lp_mathlib_List_Vector___aux__Mathlib__Data__Vector__Basic______macroRules__List__Vector__term___x3a_x3a_u1d65____1___closed__12_value;
static const lean_ctor_object lp_mathlib_List_Vector___aux__Mathlib__Data__Vector__Basic______macroRules__List__Vector__term___x3a_x3a_u1d65____1___closed__13_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_List_Vector___aux__Mathlib__Data__Vector__Basic______macroRules__List__Vector__term___x3a_x3a_u1d65____1___closed__10_value),((lean_object*)&lp_mathlib_List_Vector___aux__Mathlib__Data__Vector__Basic______macroRules__List__Vector__term___x3a_x3a_u1d65____1___closed__12_value)}};
static const lean_object* lp_mathlib_List_Vector___aux__Mathlib__Data__Vector__Basic______macroRules__List__Vector__term___x3a_x3a_u1d65____1___closed__13 = (const lean_object*)&lp_mathlib_List_Vector___aux__Mathlib__Data__Vector__Basic______macroRules__List__Vector__term___x3a_x3a_u1d65____1___closed__13_value;
static const lean_string_object lp_mathlib_List_Vector___aux__Mathlib__Data__Vector__Basic______macroRules__List__Vector__term___x3a_x3a_u1d65____1___closed__14_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "null"};
static const lean_object* lp_mathlib_List_Vector___aux__Mathlib__Data__Vector__Basic______macroRules__List__Vector__term___x3a_x3a_u1d65____1___closed__14 = (const lean_object*)&lp_mathlib_List_Vector___aux__Mathlib__Data__Vector__Basic______macroRules__List__Vector__term___x3a_x3a_u1d65____1___closed__14_value;
static const lean_ctor_object lp_mathlib_List_Vector___aux__Mathlib__Data__Vector__Basic______macroRules__List__Vector__term___x3a_x3a_u1d65____1___closed__15_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_List_Vector___aux__Mathlib__Data__Vector__Basic______macroRules__List__Vector__term___x3a_x3a_u1d65____1___closed__14_value),LEAN_SCALAR_PTR_LITERAL(24, 58, 49, 223, 146, 207, 197, 136)}};
static const lean_object* lp_mathlib_List_Vector___aux__Mathlib__Data__Vector__Basic______macroRules__List__Vector__term___x3a_x3a_u1d65____1___closed__15 = (const lean_object*)&lp_mathlib_List_Vector___aux__Mathlib__Data__Vector__Basic______macroRules__List__Vector__term___x3a_x3a_u1d65____1___closed__15_value;
LEAN_EXPORT lean_object* lp_mathlib_List_Vector___aux__Mathlib__Data__Vector__Basic______macroRules__List__Vector__term___x3a_x3a_u1d65____1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_Vector___aux__Mathlib__Data__Vector__Basic______macroRules__List__Vector__term___x3a_x3a_u1d65____1___boxed(lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_List_Vector___aux__Mathlib__Data__Vector__Basic______unexpand__List__Vector__cons__1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "ident"};
static const lean_object* lp_mathlib_List_Vector___aux__Mathlib__Data__Vector__Basic______unexpand__List__Vector__cons__1___closed__0 = (const lean_object*)&lp_mathlib_List_Vector___aux__Mathlib__Data__Vector__Basic______unexpand__List__Vector__cons__1___closed__0_value;
static const lean_ctor_object lp_mathlib_List_Vector___aux__Mathlib__Data__Vector__Basic______unexpand__List__Vector__cons__1___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_List_Vector___aux__Mathlib__Data__Vector__Basic______unexpand__List__Vector__cons__1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(52, 159, 208, 51, 14, 60, 6, 71)}};
static const lean_object* lp_mathlib_List_Vector___aux__Mathlib__Data__Vector__Basic______unexpand__List__Vector__cons__1___closed__1 = (const lean_object*)&lp_mathlib_List_Vector___aux__Mathlib__Data__Vector__Basic______unexpand__List__Vector__cons__1___closed__1_value;
LEAN_EXPORT lean_object* lp_mathlib_List_Vector___aux__Mathlib__Data__Vector__Basic______unexpand__List__Vector__cons__1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_Vector___aux__Mathlib__Data__Vector__Basic______unexpand__List__Vector__cons__1___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_Vector_instInhabited___redArg___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_Vector_instInhabited___redArg___lam__0___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_Vector_instInhabited___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_Vector_instInhabited___redArg___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_Vector_instInhabited(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_Vector_instInhabited___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Data_Vector_Basic_0__List_Vector_ofFn_match__1_splitter___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Data_Vector_Basic_0__List_Vector_ofFn_match__1_splitter___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Data_Vector_Basic_0__List_Vector_ofFn_match__1_splitter(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Data_Vector_Basic_0__List_Vector_ofFn_match__1_splitter___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Data_Vector_Basic_0__List_Vector_elim_match__3_splitter___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Data_Vector_Basic_0__List_Vector_elim_match__3_splitter(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Data_Vector_Basic_0__List_Vector_elim_match__3_splitter___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_vectorEquivFin___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_vectorEquivFin(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_Vector_reverse___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_Vector_reverse(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_Vector_reverse___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_Vector_last___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_Vector_last___redArg___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_Vector_last(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_Vector_last___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_List_Scan_Basic_0__List_scanAuxM_go___at___00List_Vector_scanl_spec__0___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_Vector_scanl___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_Vector_scanl(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_Vector_scanl___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_List_Scan_Basic_0__List_scanAuxM_go___at___00List_Vector_scanl_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_Vector_mOfFn___redArg___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_Vector_mOfFn___redArg___lam__0___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_Vector_mOfFn___redArg___lam__1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_Vector_mOfFn___redArg___lam__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_Vector_mOfFn___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_Vector_mOfFn___redArg___lam__2(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_Vector_mOfFn___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_Vector_mOfFn(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_Vector_mOfFn___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Data_Vector_Basic_0__List_Vector_mOfFn_match__1_splitter___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Data_Vector_Basic_0__List_Vector_mOfFn_match__1_splitter___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Data_Vector_Basic_0__List_Vector_mOfFn_match__1_splitter(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Data_Vector_Basic_0__List_Vector_mOfFn_match__1_splitter___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_Vector_mmap___redArg___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_Vector_mmap___redArg___lam__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_Vector_mmap___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_Vector_mmap___redArg___lam__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_Vector_mmap___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_Vector_mmap(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_Vector_mmap___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_Vector_inductionOn___redArg___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_Vector_inductionOn___redArg___lam__0___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_Vector_inductionOn___redArg___lam__1(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_Vector_inductionOn___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_Vector_inductionOn___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_Vector_inductionOn(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_Vector_inductionOn___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_Vector_inductionOn_u2082___redArg___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_Vector_inductionOn_u2082___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_Vector_inductionOn_u2082___redArg___lam__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_Vector_inductionOn_u2082___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_Vector_inductionOn_u2082___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_Vector_inductionOn_u2082(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_Vector_inductionOn_u2082___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_Vector_inductionOn_u2083___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_Vector_inductionOn_u2083___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_Vector_inductionOn_u2083___redArg___lam__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_Vector_inductionOn_u2083___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_Vector_inductionOn_u2083___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_Vector_inductionOn_u2083(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_Vector_inductionOn_u2083___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_Vector_casesOn___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_Vector_casesOn___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_Vector_casesOn___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_Vector_casesOn___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_Vector_casesOn(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_Vector_casesOn___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_Vector_casesOn_u2082___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_Vector_casesOn_u2082___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_Vector_casesOn_u2082___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_Vector_casesOn_u2082___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_Vector_casesOn_u2082(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_Vector_casesOn_u2082___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_Vector_casesOn_u2083___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_Vector_casesOn_u2083___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_Vector_casesOn_u2083___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_Vector_casesOn_u2083___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_Vector_casesOn_u2083(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_Vector_casesOn_u2083___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_Vector_toArray___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_Vector_toArray(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_Vector_toArray___boxed(lean_object*, lean_object*, lean_object*);
static const lean_array_object lp_mathlib_List_Vector_insertIdx___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_array_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 246}, .m_size = 0, .m_capacity = 0, .m_data = {}};
static const lean_object* lp_mathlib_List_Vector_insertIdx___redArg___closed__0 = (const lean_object*)&lp_mathlib_List_Vector_insertIdx___redArg___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_List_Vector_insertIdx___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_Vector_insertIdx(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_Vector_insertIdx___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_Vector_set___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_Vector_set(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_Vector_set___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Data_Vector_Basic_0__List_Vector_traverseAux___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Data_Vector_Basic_0__List_Vector_traverseAux___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Data_Vector_Basic_0__List_Vector_traverseAux(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_Vector_traverse___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_Vector_traverse(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_Vector_traverse___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_Vector_instTraversableFlipNat___lam__0(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_Vector_instTraversableFlipNat___lam__1(lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_List_Vector_instTraversableFlipNat___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_List_Vector_instTraversableFlipNat___lam__0, .m_arity = 4, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_List_Vector_instTraversableFlipNat___closed__0 = (const lean_object*)&lp_mathlib_List_Vector_instTraversableFlipNat___closed__0_value;
static const lean_closure_object lp_mathlib_List_Vector_instTraversableFlipNat___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_List_Vector_instTraversableFlipNat___lam__1, .m_arity = 4, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_List_Vector_instTraversableFlipNat___closed__1 = (const lean_object*)&lp_mathlib_List_Vector_instTraversableFlipNat___closed__1_value;
static const lean_ctor_object lp_mathlib_List_Vector_instTraversableFlipNat___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_List_Vector_instTraversableFlipNat___closed__0_value),((lean_object*)&lp_mathlib_List_Vector_instTraversableFlipNat___closed__1_value)}};
static const lean_object* lp_mathlib_List_Vector_instTraversableFlipNat___closed__2 = (const lean_object*)&lp_mathlib_List_Vector_instTraversableFlipNat___closed__2_value;
LEAN_EXPORT lean_object* lp_mathlib_List_Vector_instTraversableFlipNat(lean_object*);
static lean_object* _init_lp_mathlib_List_Vector___aux__Mathlib__Data__Vector__Basic______macroRules__List__Vector__term___x3a_x3a_u1d65____1___closed__6(void){
_start:
{
lean_object* v___x_40_; lean_object* v___x_41_; 
v___x_40_ = ((lean_object*)(lp_mathlib_List_Vector___aux__Mathlib__Data__Vector__Basic______macroRules__List__Vector__term___x3a_x3a_u1d65____1___closed__5));
v___x_41_ = l_String_toRawSubstring_x27(v___x_40_);
return v___x_41_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_Vector___aux__Mathlib__Data__Vector__Basic______macroRules__List__Vector__term___x3a_x3a_u1d65____1(lean_object* v_x_64_, lean_object* v_a_65_, lean_object* v_a_66_){
_start:
{
lean_object* v___x_67_; uint8_t v___x_68_; 
v___x_67_ = ((lean_object*)(lp_mathlib_List_Vector_term___x3a_x3a_u1d65___00__closed__3));
lean_inc(v_x_64_);
v___x_68_ = l_Lean_Syntax_isOfKind(v_x_64_, v___x_67_);
if (v___x_68_ == 0)
{
lean_object* v___x_69_; lean_object* v___x_70_; 
lean_dec(v_x_64_);
v___x_69_ = lean_box(1);
v___x_70_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_70_, 0, v___x_69_);
lean_ctor_set(v___x_70_, 1, v_a_66_);
return v___x_70_;
}
else
{
lean_object* v_quotContext_71_; lean_object* v_currMacroScope_72_; lean_object* v_ref_73_; lean_object* v___x_74_; lean_object* v___x_75_; lean_object* v___x_76_; lean_object* v___x_77_; uint8_t v___x_78_; lean_object* v___x_79_; lean_object* v___x_80_; lean_object* v___x_81_; lean_object* v___x_82_; lean_object* v___x_83_; lean_object* v___x_84_; lean_object* v___x_85_; lean_object* v___x_86_; lean_object* v___x_87_; lean_object* v___x_88_; lean_object* v___x_89_; 
v_quotContext_71_ = lean_ctor_get(v_a_65_, 1);
v_currMacroScope_72_ = lean_ctor_get(v_a_65_, 2);
v_ref_73_ = lean_ctor_get(v_a_65_, 5);
v___x_74_ = lean_unsigned_to_nat(0u);
v___x_75_ = l_Lean_Syntax_getArg(v_x_64_, v___x_74_);
v___x_76_ = lean_unsigned_to_nat(2u);
v___x_77_ = l_Lean_Syntax_getArg(v_x_64_, v___x_76_);
lean_dec(v_x_64_);
v___x_78_ = 0;
v___x_79_ = l_Lean_SourceInfo_fromRef(v_ref_73_, v___x_78_);
v___x_80_ = ((lean_object*)(lp_mathlib_List_Vector___aux__Mathlib__Data__Vector__Basic______macroRules__List__Vector__term___x3a_x3a_u1d65____1___closed__4));
v___x_81_ = lean_obj_once(&lp_mathlib_List_Vector___aux__Mathlib__Data__Vector__Basic______macroRules__List__Vector__term___x3a_x3a_u1d65____1___closed__6, &lp_mathlib_List_Vector___aux__Mathlib__Data__Vector__Basic______macroRules__List__Vector__term___x3a_x3a_u1d65____1___closed__6_once, _init_lp_mathlib_List_Vector___aux__Mathlib__Data__Vector__Basic______macroRules__List__Vector__term___x3a_x3a_u1d65____1___closed__6);
v___x_82_ = ((lean_object*)(lp_mathlib_List_Vector___aux__Mathlib__Data__Vector__Basic______macroRules__List__Vector__term___x3a_x3a_u1d65____1___closed__8));
lean_inc(v_currMacroScope_72_);
lean_inc(v_quotContext_71_);
v___x_83_ = l_Lean_addMacroScope(v_quotContext_71_, v___x_82_, v_currMacroScope_72_);
v___x_84_ = ((lean_object*)(lp_mathlib_List_Vector___aux__Mathlib__Data__Vector__Basic______macroRules__List__Vector__term___x3a_x3a_u1d65____1___closed__13));
lean_inc_n(v___x_79_, 2);
v___x_85_ = lean_alloc_ctor(3, 4, 0);
lean_ctor_set(v___x_85_, 0, v___x_79_);
lean_ctor_set(v___x_85_, 1, v___x_81_);
lean_ctor_set(v___x_85_, 2, v___x_83_);
lean_ctor_set(v___x_85_, 3, v___x_84_);
v___x_86_ = ((lean_object*)(lp_mathlib_List_Vector___aux__Mathlib__Data__Vector__Basic______macroRules__List__Vector__term___x3a_x3a_u1d65____1___closed__15));
v___x_87_ = l_Lean_Syntax_node2(v___x_79_, v___x_86_, v___x_75_, v___x_77_);
v___x_88_ = l_Lean_Syntax_node2(v___x_79_, v___x_80_, v___x_85_, v___x_87_);
v___x_89_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_89_, 0, v___x_88_);
lean_ctor_set(v___x_89_, 1, v_a_66_);
return v___x_89_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_Vector___aux__Mathlib__Data__Vector__Basic______macroRules__List__Vector__term___x3a_x3a_u1d65____1___boxed(lean_object* v_x_90_, lean_object* v_a_91_, lean_object* v_a_92_){
_start:
{
lean_object* v_res_93_; 
v_res_93_ = lp_mathlib_List_Vector___aux__Mathlib__Data__Vector__Basic______macroRules__List__Vector__term___x3a_x3a_u1d65____1(v_x_90_, v_a_91_, v_a_92_);
lean_dec_ref(v_a_91_);
return v_res_93_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_Vector___aux__Mathlib__Data__Vector__Basic______unexpand__List__Vector__cons__1(lean_object* v_x_97_, lean_object* v_a_98_, lean_object* v_a_99_){
_start:
{
lean_object* v___x_100_; uint8_t v___x_101_; 
v___x_100_ = ((lean_object*)(lp_mathlib_List_Vector___aux__Mathlib__Data__Vector__Basic______macroRules__List__Vector__term___x3a_x3a_u1d65____1___closed__4));
lean_inc(v_x_97_);
v___x_101_ = l_Lean_Syntax_isOfKind(v_x_97_, v___x_100_);
if (v___x_101_ == 0)
{
lean_object* v___x_102_; lean_object* v___x_103_; 
lean_dec(v_x_97_);
v___x_102_ = lean_box(0);
v___x_103_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_103_, 0, v___x_102_);
lean_ctor_set(v___x_103_, 1, v_a_99_);
return v___x_103_;
}
else
{
lean_object* v___x_104_; lean_object* v___x_105_; lean_object* v___x_106_; uint8_t v___x_107_; 
v___x_104_ = lean_unsigned_to_nat(0u);
v___x_105_ = l_Lean_Syntax_getArg(v_x_97_, v___x_104_);
v___x_106_ = ((lean_object*)(lp_mathlib_List_Vector___aux__Mathlib__Data__Vector__Basic______unexpand__List__Vector__cons__1___closed__1));
lean_inc(v___x_105_);
v___x_107_ = l_Lean_Syntax_isOfKind(v___x_105_, v___x_106_);
if (v___x_107_ == 0)
{
lean_object* v___x_108_; lean_object* v___x_109_; 
lean_dec(v___x_105_);
lean_dec(v_x_97_);
v___x_108_ = lean_box(0);
v___x_109_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_109_, 0, v___x_108_);
lean_ctor_set(v___x_109_, 1, v_a_99_);
return v___x_109_;
}
else
{
lean_object* v___x_110_; lean_object* v___x_111_; lean_object* v___x_112_; uint8_t v___x_113_; 
v___x_110_ = lean_unsigned_to_nat(1u);
v___x_111_ = l_Lean_Syntax_getArg(v_x_97_, v___x_110_);
lean_dec(v_x_97_);
v___x_112_ = lean_unsigned_to_nat(2u);
lean_inc(v___x_111_);
v___x_113_ = l_Lean_Syntax_matchesNull(v___x_111_, v___x_112_);
if (v___x_113_ == 0)
{
lean_object* v___x_114_; lean_object* v___x_115_; 
lean_dec(v___x_111_);
lean_dec(v___x_105_);
v___x_114_ = lean_box(0);
v___x_115_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_115_, 0, v___x_114_);
lean_ctor_set(v___x_115_, 1, v_a_99_);
return v___x_115_;
}
else
{
lean_object* v___x_116_; lean_object* v___x_117_; lean_object* v_ref_118_; uint8_t v___x_119_; lean_object* v___x_120_; lean_object* v___x_121_; lean_object* v___x_122_; lean_object* v___x_123_; lean_object* v___x_124_; lean_object* v___x_125_; 
v___x_116_ = l_Lean_Syntax_getArg(v___x_111_, v___x_104_);
v___x_117_ = l_Lean_Syntax_getArg(v___x_111_, v___x_110_);
lean_dec(v___x_111_);
v_ref_118_ = l_Lean_replaceRef(v___x_105_, v_a_98_);
lean_dec(v___x_105_);
v___x_119_ = 0;
v___x_120_ = l_Lean_SourceInfo_fromRef(v_ref_118_, v___x_119_);
lean_dec(v_ref_118_);
v___x_121_ = ((lean_object*)(lp_mathlib_List_Vector_term___x3a_x3a_u1d65___00__closed__3));
v___x_122_ = ((lean_object*)(lp_mathlib_List_Vector_term___x3a_x3a_u1d65___00__closed__6));
lean_inc(v___x_120_);
v___x_123_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_123_, 0, v___x_120_);
lean_ctor_set(v___x_123_, 1, v___x_122_);
v___x_124_ = l_Lean_Syntax_node3(v___x_120_, v___x_121_, v___x_116_, v___x_123_, v___x_117_);
v___x_125_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_125_, 0, v___x_124_);
lean_ctor_set(v___x_125_, 1, v_a_99_);
return v___x_125_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_Vector___aux__Mathlib__Data__Vector__Basic______unexpand__List__Vector__cons__1___boxed(lean_object* v_x_126_, lean_object* v_a_127_, lean_object* v_a_128_){
_start:
{
lean_object* v_res_129_; 
v_res_129_ = lp_mathlib_List_Vector___aux__Mathlib__Data__Vector__Basic______unexpand__List__Vector__cons__1(v_x_126_, v_a_127_, v_a_128_);
lean_dec(v_a_127_);
return v_res_129_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_Vector_instInhabited___redArg___lam__0(lean_object* v_inst_130_, lean_object* v_x_131_){
_start:
{
lean_inc(v_inst_130_);
return v_inst_130_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_Vector_instInhabited___redArg___lam__0___boxed(lean_object* v_inst_132_, lean_object* v_x_133_){
_start:
{
lean_object* v_res_134_; 
v_res_134_ = lp_mathlib_List_Vector_instInhabited___redArg___lam__0(v_inst_132_, v_x_133_);
lean_dec(v_x_133_);
lean_dec(v_inst_132_);
return v_res_134_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_Vector_instInhabited___redArg(lean_object* v_n_135_, lean_object* v_inst_136_){
_start:
{
lean_object* v___f_137_; lean_object* v___x_138_; 
v___f_137_ = lean_alloc_closure((void*)(lp_mathlib_List_Vector_instInhabited___redArg___lam__0___boxed), 2, 1);
lean_closure_set(v___f_137_, 0, v_inst_136_);
v___x_138_ = lp_mathlib_List_Vector_ofFn___redArg(v_n_135_, v___f_137_);
return v___x_138_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_Vector_instInhabited___redArg___boxed(lean_object* v_n_139_, lean_object* v_inst_140_){
_start:
{
lean_object* v_res_141_; 
v_res_141_ = lp_mathlib_List_Vector_instInhabited___redArg(v_n_139_, v_inst_140_);
lean_dec(v_n_139_);
return v_res_141_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_Vector_instInhabited(lean_object* v_00_u03b1_142_, lean_object* v_n_143_, lean_object* v_inst_144_){
_start:
{
lean_object* v___x_145_; 
v___x_145_ = lp_mathlib_List_Vector_instInhabited___redArg(v_n_143_, v_inst_144_);
return v___x_145_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_Vector_instInhabited___boxed(lean_object* v_00_u03b1_146_, lean_object* v_n_147_, lean_object* v_inst_148_){
_start:
{
lean_object* v_res_149_; 
v_res_149_ = lp_mathlib_List_Vector_instInhabited(v_00_u03b1_146_, v_n_147_, v_inst_148_);
lean_dec(v_n_147_);
return v_res_149_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Data_Vector_Basic_0__List_Vector_ofFn_match__1_splitter___redArg(lean_object* v_x_150_, lean_object* v_x_151_, lean_object* v_h__1_152_, lean_object* v_h__2_153_){
_start:
{
lean_object* v_zero_154_; uint8_t v_isZero_155_; 
v_zero_154_ = lean_unsigned_to_nat(0u);
v_isZero_155_ = lean_nat_dec_eq(v_x_150_, v_zero_154_);
if (v_isZero_155_ == 1)
{
lean_object* v___x_156_; 
lean_dec(v_h__2_153_);
v___x_156_ = lean_apply_1(v_h__1_152_, v_x_151_);
return v___x_156_;
}
else
{
lean_object* v_one_157_; lean_object* v_n_158_; lean_object* v___x_159_; 
lean_dec(v_h__1_152_);
v_one_157_ = lean_unsigned_to_nat(1u);
v_n_158_ = lean_nat_sub(v_x_150_, v_one_157_);
v___x_159_ = lean_apply_2(v_h__2_153_, v_n_158_, v_x_151_);
return v___x_159_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Data_Vector_Basic_0__List_Vector_ofFn_match__1_splitter___redArg___boxed(lean_object* v_x_160_, lean_object* v_x_161_, lean_object* v_h__1_162_, lean_object* v_h__2_163_){
_start:
{
lean_object* v_res_164_; 
v_res_164_ = lp_mathlib___private_Mathlib_Data_Vector_Basic_0__List_Vector_ofFn_match__1_splitter___redArg(v_x_160_, v_x_161_, v_h__1_162_, v_h__2_163_);
lean_dec(v_x_160_);
return v_res_164_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Data_Vector_Basic_0__List_Vector_ofFn_match__1_splitter(lean_object* v_00_u03b1_165_, lean_object* v_motive_166_, lean_object* v_x_167_, lean_object* v_x_168_, lean_object* v_h__1_169_, lean_object* v_h__2_170_){
_start:
{
lean_object* v_zero_171_; uint8_t v_isZero_172_; 
v_zero_171_ = lean_unsigned_to_nat(0u);
v_isZero_172_ = lean_nat_dec_eq(v_x_167_, v_zero_171_);
if (v_isZero_172_ == 1)
{
lean_object* v___x_173_; 
lean_dec(v_h__2_170_);
v___x_173_ = lean_apply_1(v_h__1_169_, v_x_168_);
return v___x_173_;
}
else
{
lean_object* v_one_174_; lean_object* v_n_175_; lean_object* v___x_176_; 
lean_dec(v_h__1_169_);
v_one_174_ = lean_unsigned_to_nat(1u);
v_n_175_ = lean_nat_sub(v_x_167_, v_one_174_);
v___x_176_ = lean_apply_2(v_h__2_170_, v_n_175_, v_x_168_);
return v___x_176_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Data_Vector_Basic_0__List_Vector_ofFn_match__1_splitter___boxed(lean_object* v_00_u03b1_177_, lean_object* v_motive_178_, lean_object* v_x_179_, lean_object* v_x_180_, lean_object* v_h__1_181_, lean_object* v_h__2_182_){
_start:
{
lean_object* v_res_183_; 
v_res_183_ = lp_mathlib___private_Mathlib_Data_Vector_Basic_0__List_Vector_ofFn_match__1_splitter(v_00_u03b1_177_, v_motive_178_, v_x_179_, v_x_180_, v_h__1_181_, v_h__2_182_);
lean_dec(v_x_179_);
return v_res_183_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Data_Vector_Basic_0__List_Vector_elim_match__3_splitter___redArg(lean_object* v_x_184_, lean_object* v_h__1_185_){
_start:
{
lean_object* v___x_186_; 
v___x_186_ = lean_apply_2(v_h__1_185_, v_x_184_, lean_box(0));
return v___x_186_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Data_Vector_Basic_0__List_Vector_elim_match__3_splitter(lean_object* v_00_u03b1_187_, lean_object* v_n_188_, lean_object* v_motive_189_, lean_object* v_x_190_, lean_object* v_h__1_191_){
_start:
{
lean_object* v___x_192_; 
v___x_192_ = lean_apply_2(v_h__1_191_, v_x_190_, lean_box(0));
return v___x_192_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Data_Vector_Basic_0__List_Vector_elim_match__3_splitter___boxed(lean_object* v_00_u03b1_193_, lean_object* v_n_194_, lean_object* v_motive_195_, lean_object* v_x_196_, lean_object* v_h__1_197_){
_start:
{
lean_object* v_res_198_; 
v_res_198_ = lp_mathlib___private_Mathlib_Data_Vector_Basic_0__List_Vector_elim_match__3_splitter(v_00_u03b1_193_, v_n_194_, v_motive_195_, v_x_196_, v_h__1_197_);
lean_dec(v_n_194_);
return v_res_198_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_vectorEquivFin___redArg(lean_object* v_n_199_){
_start:
{
lean_object* v___x_200_; lean_object* v___x_201_; lean_object* v___x_202_; 
lean_inc(v_n_199_);
v___x_200_ = lean_alloc_closure((void*)(lp_mathlib_List_Vector_get___boxed), 4, 2);
lean_closure_set(v___x_200_, 0, lean_box(0));
lean_closure_set(v___x_200_, 1, v_n_199_);
v___x_201_ = lean_alloc_closure((void*)(lp_mathlib_List_Vector_ofFn___boxed), 3, 2);
lean_closure_set(v___x_201_, 0, lean_box(0));
lean_closure_set(v___x_201_, 1, v_n_199_);
v___x_202_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_202_, 0, v___x_200_);
lean_ctor_set(v___x_202_, 1, v___x_201_);
return v___x_202_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_vectorEquivFin(lean_object* v_00_u03b1_203_, lean_object* v_n_204_){
_start:
{
lean_object* v___x_205_; 
v___x_205_ = lp_mathlib_Equiv_vectorEquivFin___redArg(v_n_204_);
return v___x_205_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_Vector_reverse___redArg(lean_object* v_v_206_){
_start:
{
lean_object* v___x_207_; 
v___x_207_ = l_List_reverse___redArg(v_v_206_);
return v___x_207_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_Vector_reverse(lean_object* v_00_u03b1_208_, lean_object* v_n_209_, lean_object* v_v_210_){
_start:
{
lean_object* v___x_211_; 
v___x_211_ = l_List_reverse___redArg(v_v_210_);
return v___x_211_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_Vector_reverse___boxed(lean_object* v_00_u03b1_212_, lean_object* v_n_213_, lean_object* v_v_214_){
_start:
{
lean_object* v_res_215_; 
v_res_215_ = lp_mathlib_List_Vector_reverse(v_00_u03b1_212_, v_n_213_, v_v_214_);
lean_dec(v_n_213_);
return v_res_215_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_Vector_last___redArg(lean_object* v_n_216_, lean_object* v_v_217_){
_start:
{
lean_object* v___x_218_; 
v___x_218_ = l_List_get___redArg(v_v_217_, v_n_216_);
return v___x_218_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_Vector_last___redArg___boxed(lean_object* v_n_219_, lean_object* v_v_220_){
_start:
{
lean_object* v_res_221_; 
v_res_221_ = lp_mathlib_List_Vector_last___redArg(v_n_219_, v_v_220_);
lean_dec(v_v_220_);
return v_res_221_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_Vector_last(lean_object* v_00_u03b1_222_, lean_object* v_n_223_, lean_object* v_v_224_){
_start:
{
lean_object* v___x_225_; 
v___x_225_ = l_List_get___redArg(v_v_224_, v_n_223_);
return v___x_225_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_Vector_last___boxed(lean_object* v_00_u03b1_226_, lean_object* v_n_227_, lean_object* v_v_228_){
_start:
{
lean_object* v_res_229_; 
v_res_229_ = lp_mathlib_List_Vector_last(v_00_u03b1_226_, v_n_227_, v_v_228_);
lean_dec(v_v_228_);
return v_res_229_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_List_Scan_Basic_0__List_scanAuxM_go___at___00List_Vector_scanl_spec__0___redArg(lean_object* v_f_230_, lean_object* v_a_231_, lean_object* v_a_232_, lean_object* v_a_233_){
_start:
{
if (lean_obj_tag(v_a_231_) == 0)
{
lean_object* v___x_234_; 
lean_dec(v_f_230_);
v___x_234_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_234_, 0, v_a_232_);
lean_ctor_set(v___x_234_, 1, v_a_233_);
return v___x_234_;
}
else
{
lean_object* v_head_235_; lean_object* v_tail_236_; lean_object* v___x_238_; uint8_t v_isShared_239_; uint8_t v_isSharedCheck_245_; 
v_head_235_ = lean_ctor_get(v_a_231_, 0);
v_tail_236_ = lean_ctor_get(v_a_231_, 1);
v_isSharedCheck_245_ = !lean_is_exclusive(v_a_231_);
if (v_isSharedCheck_245_ == 0)
{
v___x_238_ = v_a_231_;
v_isShared_239_ = v_isSharedCheck_245_;
goto v_resetjp_237_;
}
else
{
lean_inc(v_tail_236_);
lean_inc(v_head_235_);
lean_dec(v_a_231_);
v___x_238_ = lean_box(0);
v_isShared_239_ = v_isSharedCheck_245_;
goto v_resetjp_237_;
}
v_resetjp_237_:
{
lean_object* v___x_240_; lean_object* v___x_242_; 
lean_inc(v_f_230_);
lean_inc(v_a_232_);
v___x_240_ = lean_apply_2(v_f_230_, v_a_232_, v_head_235_);
if (v_isShared_239_ == 0)
{
lean_ctor_set(v___x_238_, 1, v_a_233_);
lean_ctor_set(v___x_238_, 0, v_a_232_);
v___x_242_ = v___x_238_;
goto v_reusejp_241_;
}
else
{
lean_object* v_reuseFailAlloc_244_; 
v_reuseFailAlloc_244_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_244_, 0, v_a_232_);
lean_ctor_set(v_reuseFailAlloc_244_, 1, v_a_233_);
v___x_242_ = v_reuseFailAlloc_244_;
goto v_reusejp_241_;
}
v_reusejp_241_:
{
v_a_231_ = v_tail_236_;
v_a_232_ = v___x_240_;
v_a_233_ = v___x_242_;
goto _start;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_Vector_scanl___redArg(lean_object* v_f_246_, lean_object* v_b_247_, lean_object* v_v_248_){
_start:
{
lean_object* v___x_249_; lean_object* v___x_250_; lean_object* v___x_251_; 
v___x_249_ = lean_box(0);
v___x_250_ = lp_mathlib___private_Init_Data_List_Scan_Basic_0__List_scanAuxM_go___at___00List_Vector_scanl_spec__0___redArg(v_f_246_, v_v_248_, v_b_247_, v___x_249_);
v___x_251_ = l_List_reverse___redArg(v___x_250_);
return v___x_251_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_Vector_scanl(lean_object* v_00_u03b1_252_, lean_object* v_n_253_, lean_object* v_00_u03b2_254_, lean_object* v_f_255_, lean_object* v_b_256_, lean_object* v_v_257_){
_start:
{
lean_object* v___x_258_; 
v___x_258_ = lp_mathlib_List_Vector_scanl___redArg(v_f_255_, v_b_256_, v_v_257_);
return v___x_258_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_Vector_scanl___boxed(lean_object* v_00_u03b1_259_, lean_object* v_n_260_, lean_object* v_00_u03b2_261_, lean_object* v_f_262_, lean_object* v_b_263_, lean_object* v_v_264_){
_start:
{
lean_object* v_res_265_; 
v_res_265_ = lp_mathlib_List_Vector_scanl(v_00_u03b1_259_, v_n_260_, v_00_u03b2_261_, v_f_262_, v_b_263_, v_v_264_);
lean_dec(v_n_260_);
return v_res_265_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_List_Scan_Basic_0__List_scanAuxM_go___at___00List_Vector_scanl_spec__0(lean_object* v_00_u03b2_266_, lean_object* v_00_u03b1_267_, lean_object* v_f_268_, lean_object* v_a_269_, lean_object* v_a_270_, lean_object* v_a_271_){
_start:
{
lean_object* v___x_272_; 
v___x_272_ = lp_mathlib___private_Init_Data_List_Scan_Basic_0__List_scanAuxM_go___at___00List_Vector_scanl_spec__0___redArg(v_f_268_, v_a_269_, v_a_270_, v_a_271_);
return v___x_272_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_Vector_mOfFn___redArg___lam__0(lean_object* v_x_273_, lean_object* v_i_274_){
_start:
{
lean_object* v___x_275_; lean_object* v___x_276_; 
v___x_275_ = l_Fin_succ___redArg(v_i_274_);
v___x_276_ = lean_apply_1(v_x_273_, v___x_275_);
return v___x_276_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_Vector_mOfFn___redArg___lam__0___boxed(lean_object* v_x_277_, lean_object* v_i_278_){
_start:
{
lean_object* v_res_279_; 
v_res_279_ = lp_mathlib_List_Vector_mOfFn___redArg___lam__0(v_x_277_, v_i_278_);
lean_dec(v_i_278_);
return v_res_279_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_Vector_mOfFn___redArg___lam__1(lean_object* v_a_280_, lean_object* v_toPure_281_, lean_object* v_v_282_){
_start:
{
lean_object* v___x_283_; lean_object* v___x_284_; 
v___x_283_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_283_, 0, v_a_280_);
lean_ctor_set(v___x_283_, 1, v_v_282_);
v___x_284_ = lean_apply_2(v_toPure_281_, lean_box(0), v___x_283_);
return v___x_284_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_Vector_mOfFn___redArg___lam__2___boxed(lean_object* v_toPure_285_, lean_object* v_inst_286_, lean_object* v_n_287_, lean_object* v___f_288_, lean_object* v_toBind_289_, lean_object* v_a_290_){
_start:
{
lean_object* v_res_291_; 
v_res_291_ = lp_mathlib_List_Vector_mOfFn___redArg___lam__2(v_toPure_285_, v_inst_286_, v_n_287_, v___f_288_, v_toBind_289_, v_a_290_);
lean_dec(v_n_287_);
return v_res_291_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_Vector_mOfFn___redArg(lean_object* v_inst_292_, lean_object* v_x_293_, lean_object* v_x_294_){
_start:
{
lean_object* v_toApplicative_295_; lean_object* v_toBind_296_; lean_object* v_toPure_297_; lean_object* v_zero_298_; uint8_t v_isZero_299_; 
v_toApplicative_295_ = lean_ctor_get(v_inst_292_, 0);
v_toBind_296_ = lean_ctor_get(v_inst_292_, 1);
lean_inc(v_toBind_296_);
v_toPure_297_ = lean_ctor_get(v_toApplicative_295_, 1);
lean_inc(v_toPure_297_);
v_zero_298_ = lean_unsigned_to_nat(0u);
v_isZero_299_ = lean_nat_dec_eq(v_x_293_, v_zero_298_);
if (v_isZero_299_ == 1)
{
lean_object* v___x_300_; lean_object* v___x_301_; 
lean_dec(v_toBind_296_);
lean_dec(v_x_294_);
lean_dec_ref(v_inst_292_);
v___x_300_ = lean_box(0);
v___x_301_ = lean_apply_2(v_toPure_297_, lean_box(0), v___x_300_);
return v___x_301_;
}
else
{
lean_object* v___f_302_; lean_object* v_one_303_; lean_object* v_n_304_; lean_object* v___f_305_; lean_object* v___x_306_; lean_object* v___x_307_; lean_object* v___x_308_; lean_object* v___x_309_; 
lean_inc(v_x_294_);
v___f_302_ = lean_alloc_closure((void*)(lp_mathlib_List_Vector_mOfFn___redArg___lam__0___boxed), 2, 1);
lean_closure_set(v___f_302_, 0, v_x_294_);
v_one_303_ = lean_unsigned_to_nat(1u);
v_n_304_ = lean_nat_sub(v_x_293_, v_one_303_);
lean_inc(v_toBind_296_);
lean_inc(v_n_304_);
v___f_305_ = lean_alloc_closure((void*)(lp_mathlib_List_Vector_mOfFn___redArg___lam__2___boxed), 6, 5);
lean_closure_set(v___f_305_, 0, v_toPure_297_);
lean_closure_set(v___f_305_, 1, v_inst_292_);
lean_closure_set(v___f_305_, 2, v_n_304_);
lean_closure_set(v___f_305_, 3, v___f_302_);
lean_closure_set(v___f_305_, 4, v_toBind_296_);
v___x_306_ = lean_nat_add(v_n_304_, v_one_303_);
lean_dec(v_n_304_);
v___x_307_ = lean_nat_mod(v_zero_298_, v___x_306_);
lean_dec(v___x_306_);
v___x_308_ = lean_apply_1(v_x_294_, v___x_307_);
v___x_309_ = lean_apply_4(v_toBind_296_, lean_box(0), lean_box(0), v___x_308_, v___f_305_);
return v___x_309_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_Vector_mOfFn___redArg___lam__2(lean_object* v_toPure_310_, lean_object* v_inst_311_, lean_object* v_n_312_, lean_object* v___f_313_, lean_object* v_toBind_314_, lean_object* v_a_315_){
_start:
{
lean_object* v___f_316_; lean_object* v___x_317_; lean_object* v___x_318_; 
v___f_316_ = lean_alloc_closure((void*)(lp_mathlib_List_Vector_mOfFn___redArg___lam__1), 3, 2);
lean_closure_set(v___f_316_, 0, v_a_315_);
lean_closure_set(v___f_316_, 1, v_toPure_310_);
v___x_317_ = lp_mathlib_List_Vector_mOfFn___redArg(v_inst_311_, v_n_312_, v___f_313_);
v___x_318_ = lean_apply_4(v_toBind_314_, lean_box(0), lean_box(0), v___x_317_, v___f_316_);
return v___x_318_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_Vector_mOfFn___redArg___boxed(lean_object* v_inst_319_, lean_object* v_x_320_, lean_object* v_x_321_){
_start:
{
lean_object* v_res_322_; 
v_res_322_ = lp_mathlib_List_Vector_mOfFn___redArg(v_inst_319_, v_x_320_, v_x_321_);
lean_dec(v_x_320_);
return v_res_322_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_Vector_mOfFn(lean_object* v_m_323_, lean_object* v_inst_324_, lean_object* v_00_u03b1_325_, lean_object* v_x_326_, lean_object* v_x_327_){
_start:
{
lean_object* v___x_328_; 
v___x_328_ = lp_mathlib_List_Vector_mOfFn___redArg(v_inst_324_, v_x_326_, v_x_327_);
return v___x_328_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_Vector_mOfFn___boxed(lean_object* v_m_329_, lean_object* v_inst_330_, lean_object* v_00_u03b1_331_, lean_object* v_x_332_, lean_object* v_x_333_){
_start:
{
lean_object* v_res_334_; 
v_res_334_ = lp_mathlib_List_Vector_mOfFn(v_m_329_, v_inst_330_, v_00_u03b1_331_, v_x_332_, v_x_333_);
lean_dec(v_x_332_);
return v_res_334_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Data_Vector_Basic_0__List_Vector_mOfFn_match__1_splitter___redArg(lean_object* v_x_335_, lean_object* v_x_336_, lean_object* v_h__1_337_, lean_object* v_h__2_338_){
_start:
{
lean_object* v_zero_339_; uint8_t v_isZero_340_; 
v_zero_339_ = lean_unsigned_to_nat(0u);
v_isZero_340_ = lean_nat_dec_eq(v_x_335_, v_zero_339_);
if (v_isZero_340_ == 1)
{
lean_object* v___x_341_; 
lean_dec(v_h__2_338_);
v___x_341_ = lean_apply_1(v_h__1_337_, v_x_336_);
return v___x_341_;
}
else
{
lean_object* v_one_342_; lean_object* v_n_343_; lean_object* v___x_344_; 
lean_dec(v_h__1_337_);
v_one_342_ = lean_unsigned_to_nat(1u);
v_n_343_ = lean_nat_sub(v_x_335_, v_one_342_);
v___x_344_ = lean_apply_2(v_h__2_338_, v_n_343_, v_x_336_);
return v___x_344_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Data_Vector_Basic_0__List_Vector_mOfFn_match__1_splitter___redArg___boxed(lean_object* v_x_345_, lean_object* v_x_346_, lean_object* v_h__1_347_, lean_object* v_h__2_348_){
_start:
{
lean_object* v_res_349_; 
v_res_349_ = lp_mathlib___private_Mathlib_Data_Vector_Basic_0__List_Vector_mOfFn_match__1_splitter___redArg(v_x_345_, v_x_346_, v_h__1_347_, v_h__2_348_);
lean_dec(v_x_345_);
return v_res_349_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Data_Vector_Basic_0__List_Vector_mOfFn_match__1_splitter(lean_object* v_m_350_, lean_object* v_00_u03b1_351_, lean_object* v_motive_352_, lean_object* v_x_353_, lean_object* v_x_354_, lean_object* v_h__1_355_, lean_object* v_h__2_356_){
_start:
{
lean_object* v_zero_357_; uint8_t v_isZero_358_; 
v_zero_357_ = lean_unsigned_to_nat(0u);
v_isZero_358_ = lean_nat_dec_eq(v_x_353_, v_zero_357_);
if (v_isZero_358_ == 1)
{
lean_object* v___x_359_; 
lean_dec(v_h__2_356_);
v___x_359_ = lean_apply_1(v_h__1_355_, v_x_354_);
return v___x_359_;
}
else
{
lean_object* v_one_360_; lean_object* v_n_361_; lean_object* v___x_362_; 
lean_dec(v_h__1_355_);
v_one_360_ = lean_unsigned_to_nat(1u);
v_n_361_ = lean_nat_sub(v_x_353_, v_one_360_);
v___x_362_ = lean_apply_2(v_h__2_356_, v_n_361_, v_x_354_);
return v___x_362_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Data_Vector_Basic_0__List_Vector_mOfFn_match__1_splitter___boxed(lean_object* v_m_363_, lean_object* v_00_u03b1_364_, lean_object* v_motive_365_, lean_object* v_x_366_, lean_object* v_x_367_, lean_object* v_h__1_368_, lean_object* v_h__2_369_){
_start:
{
lean_object* v_res_370_; 
v_res_370_ = lp_mathlib___private_Mathlib_Data_Vector_Basic_0__List_Vector_mOfFn_match__1_splitter(v_m_363_, v_00_u03b1_364_, v_motive_365_, v_x_366_, v_x_367_, v_h__1_368_, v_h__2_369_);
lean_dec(v_x_366_);
return v_res_370_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_Vector_mmap___redArg___lam__0(lean_object* v_h_x27_371_, lean_object* v_toPure_372_, lean_object* v_t_x27_373_){
_start:
{
lean_object* v___x_374_; lean_object* v___x_375_; 
v___x_374_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_374_, 0, v_h_x27_371_);
lean_ctor_set(v___x_374_, 1, v_t_x27_373_);
v___x_375_ = lean_apply_2(v_toPure_372_, lean_box(0), v___x_374_);
return v___x_375_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_Vector_mmap___redArg___lam__1___boxed(lean_object* v_toPure_376_, lean_object* v_n_377_, lean_object* v_inst_378_, lean_object* v_f_379_, lean_object* v_tail_380_, lean_object* v_toBind_381_, lean_object* v_h_x27_382_){
_start:
{
lean_object* v_res_383_; 
v_res_383_ = lp_mathlib_List_Vector_mmap___redArg___lam__1(v_toPure_376_, v_n_377_, v_inst_378_, v_f_379_, v_tail_380_, v_toBind_381_, v_h_x27_382_);
lean_dec(v_n_377_);
return v_res_383_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_Vector_mmap___redArg(lean_object* v_inst_384_, lean_object* v_f_385_, lean_object* v_x_386_, lean_object* v_x_387_){
_start:
{
lean_object* v_toApplicative_388_; lean_object* v_toBind_389_; lean_object* v_toPure_390_; lean_object* v_zero_391_; uint8_t v_isZero_392_; 
v_toApplicative_388_ = lean_ctor_get(v_inst_384_, 0);
v_toBind_389_ = lean_ctor_get(v_inst_384_, 1);
lean_inc(v_toBind_389_);
v_toPure_390_ = lean_ctor_get(v_toApplicative_388_, 1);
lean_inc(v_toPure_390_);
v_zero_391_ = lean_unsigned_to_nat(0u);
v_isZero_392_ = lean_nat_dec_eq(v_x_386_, v_zero_391_);
if (v_isZero_392_ == 1)
{
lean_object* v___x_393_; lean_object* v___x_394_; 
lean_dec(v_toBind_389_);
lean_dec(v_x_387_);
lean_dec(v_f_385_);
lean_dec_ref(v_inst_384_);
v___x_393_ = lean_box(0);
v___x_394_ = lean_apply_2(v_toPure_390_, lean_box(0), v___x_393_);
return v___x_394_;
}
else
{
lean_object* v_head_395_; lean_object* v_tail_396_; lean_object* v_one_397_; lean_object* v_n_398_; lean_object* v___f_399_; lean_object* v___x_400_; lean_object* v___x_401_; 
v_head_395_ = lean_ctor_get(v_x_387_, 0);
lean_inc(v_head_395_);
v_tail_396_ = lean_ctor_get(v_x_387_, 1);
lean_inc(v_tail_396_);
lean_dec(v_x_387_);
v_one_397_ = lean_unsigned_to_nat(1u);
v_n_398_ = lean_nat_sub(v_x_386_, v_one_397_);
lean_inc(v_toBind_389_);
lean_inc(v_f_385_);
v___f_399_ = lean_alloc_closure((void*)(lp_mathlib_List_Vector_mmap___redArg___lam__1___boxed), 7, 6);
lean_closure_set(v___f_399_, 0, v_toPure_390_);
lean_closure_set(v___f_399_, 1, v_n_398_);
lean_closure_set(v___f_399_, 2, v_inst_384_);
lean_closure_set(v___f_399_, 3, v_f_385_);
lean_closure_set(v___f_399_, 4, v_tail_396_);
lean_closure_set(v___f_399_, 5, v_toBind_389_);
v___x_400_ = lean_apply_1(v_f_385_, v_head_395_);
v___x_401_ = lean_apply_4(v_toBind_389_, lean_box(0), lean_box(0), v___x_400_, v___f_399_);
return v___x_401_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_Vector_mmap___redArg___lam__1(lean_object* v_toPure_402_, lean_object* v_n_403_, lean_object* v_inst_404_, lean_object* v_f_405_, lean_object* v_tail_406_, lean_object* v_toBind_407_, lean_object* v_h_x27_408_){
_start:
{
lean_object* v___f_409_; lean_object* v___x_410_; lean_object* v___x_411_; lean_object* v___x_412_; lean_object* v___x_413_; lean_object* v___x_414_; 
v___f_409_ = lean_alloc_closure((void*)(lp_mathlib_List_Vector_mmap___redArg___lam__0), 3, 2);
lean_closure_set(v___f_409_, 0, v_h_x27_408_);
lean_closure_set(v___f_409_, 1, v_toPure_402_);
v___x_410_ = lean_unsigned_to_nat(1u);
v___x_411_ = lean_nat_add(v_n_403_, v___x_410_);
v___x_412_ = lean_nat_sub(v___x_411_, v___x_410_);
lean_dec(v___x_411_);
v___x_413_ = lp_mathlib_List_Vector_mmap___redArg(v_inst_404_, v_f_405_, v___x_412_, v_tail_406_);
lean_dec(v___x_412_);
v___x_414_ = lean_apply_4(v_toBind_407_, lean_box(0), lean_box(0), v___x_413_, v___f_409_);
return v___x_414_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_Vector_mmap___redArg___boxed(lean_object* v_inst_415_, lean_object* v_f_416_, lean_object* v_x_417_, lean_object* v_x_418_){
_start:
{
lean_object* v_res_419_; 
v_res_419_ = lp_mathlib_List_Vector_mmap___redArg(v_inst_415_, v_f_416_, v_x_417_, v_x_418_);
lean_dec(v_x_417_);
return v_res_419_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_Vector_mmap(lean_object* v_m_420_, lean_object* v_inst_421_, lean_object* v_00_u03b1_422_, lean_object* v_00_u03b2_423_, lean_object* v_f_424_, lean_object* v_x_425_, lean_object* v_x_426_){
_start:
{
lean_object* v___x_427_; 
v___x_427_ = lp_mathlib_List_Vector_mmap___redArg(v_inst_421_, v_f_424_, v_x_425_, v_x_426_);
return v___x_427_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_Vector_mmap___boxed(lean_object* v_m_428_, lean_object* v_inst_429_, lean_object* v_00_u03b1_430_, lean_object* v_00_u03b2_431_, lean_object* v_f_432_, lean_object* v_x_433_, lean_object* v_x_434_){
_start:
{
lean_object* v_res_435_; 
v_res_435_ = lp_mathlib_List_Vector_mmap(v_m_428_, v_inst_429_, v_00_u03b1_430_, v_00_u03b2_431_, v_f_432_, v_x_433_, v_x_434_);
lean_dec(v_x_433_);
return v_res_435_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_Vector_inductionOn___redArg___lam__0(lean_object* v_nil_436_, lean_object* v_v_437_){
_start:
{
lean_inc(v_nil_436_);
return v_nil_436_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_Vector_inductionOn___redArg___lam__0___boxed(lean_object* v_nil_438_, lean_object* v_v_439_){
_start:
{
lean_object* v_res_440_; 
v_res_440_ = lp_mathlib_List_Vector_inductionOn___redArg___lam__0(v_nil_438_, v_v_439_);
lean_dec(v_v_439_);
lean_dec(v_nil_438_);
return v_res_440_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_Vector_inductionOn___redArg___lam__1(lean_object* v_cons_441_, lean_object* v_n_442_, lean_object* v_ih_443_, lean_object* v_v_444_){
_start:
{
lean_object* v_a_445_; lean_object* v_v_446_; lean_object* v___x_447_; lean_object* v___x_448_; 
v_a_445_ = lean_ctor_get(v_v_444_, 0);
lean_inc(v_a_445_);
v_v_446_ = lean_ctor_get(v_v_444_, 1);
lean_inc_n(v_v_446_, 2);
lean_dec(v_v_444_);
v___x_447_ = lean_apply_1(v_ih_443_, v_v_446_);
v___x_448_ = lean_apply_4(v_cons_441_, v_n_442_, v_a_445_, v_v_446_, v___x_447_);
return v___x_448_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_Vector_inductionOn___redArg(lean_object* v_n_449_, lean_object* v_v_450_, lean_object* v_nil_451_, lean_object* v_cons_452_){
_start:
{
lean_object* v___f_453_; lean_object* v___f_454_; lean_object* v___x_98__overap_455_; lean_object* v___x_456_; 
v___f_453_ = lean_alloc_closure((void*)(lp_mathlib_List_Vector_inductionOn___redArg___lam__0___boxed), 2, 1);
lean_closure_set(v___f_453_, 0, v_nil_451_);
v___f_454_ = lean_alloc_closure((void*)(lp_mathlib_List_Vector_inductionOn___redArg___lam__1), 4, 1);
lean_closure_set(v___f_454_, 0, v_cons_452_);
v___x_98__overap_455_ = l_Nat_recCompiled___redArg(v___f_453_, v___f_454_, v_n_449_);
lean_dec_ref(v___f_453_);
v___x_456_ = lean_apply_1(v___x_98__overap_455_, v_v_450_);
return v___x_456_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_Vector_inductionOn___redArg___boxed(lean_object* v_n_457_, lean_object* v_v_458_, lean_object* v_nil_459_, lean_object* v_cons_460_){
_start:
{
lean_object* v_res_461_; 
v_res_461_ = lp_mathlib_List_Vector_inductionOn___redArg(v_n_457_, v_v_458_, v_nil_459_, v_cons_460_);
lean_dec(v_n_457_);
return v_res_461_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_Vector_inductionOn(lean_object* v_00_u03b1_462_, lean_object* v_C_463_, lean_object* v_n_464_, lean_object* v_v_465_, lean_object* v_nil_466_, lean_object* v_cons_467_){
_start:
{
lean_object* v___x_468_; 
v___x_468_ = lp_mathlib_List_Vector_inductionOn___redArg(v_n_464_, v_v_465_, v_nil_466_, v_cons_467_);
return v___x_468_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_Vector_inductionOn___boxed(lean_object* v_00_u03b1_469_, lean_object* v_C_470_, lean_object* v_n_471_, lean_object* v_v_472_, lean_object* v_nil_473_, lean_object* v_cons_474_){
_start:
{
lean_object* v_res_475_; 
v_res_475_ = lp_mathlib_List_Vector_inductionOn(v_00_u03b1_469_, v_C_470_, v_n_471_, v_v_472_, v_nil_473_, v_cons_474_);
lean_dec(v_n_471_);
return v_res_475_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_Vector_inductionOn_u2082___redArg___lam__0(lean_object* v_nil_476_, lean_object* v_v_477_, lean_object* v_w_478_){
_start:
{
lean_inc(v_nil_476_);
return v_nil_476_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_Vector_inductionOn_u2082___redArg___lam__0___boxed(lean_object* v_nil_479_, lean_object* v_v_480_, lean_object* v_w_481_){
_start:
{
lean_object* v_res_482_; 
v_res_482_ = lp_mathlib_List_Vector_inductionOn_u2082___redArg___lam__0(v_nil_479_, v_v_480_, v_w_481_);
lean_dec(v_w_481_);
lean_dec(v_v_480_);
lean_dec(v_nil_479_);
return v_res_482_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_Vector_inductionOn_u2082___redArg___lam__1(lean_object* v_cons_483_, lean_object* v_n_484_, lean_object* v_ih_485_, lean_object* v_v_486_, lean_object* v_w_487_){
_start:
{
lean_object* v_a_488_; lean_object* v_v_489_; lean_object* v_b_490_; lean_object* v_w_491_; lean_object* v___x_492_; lean_object* v___x_493_; 
v_a_488_ = lean_ctor_get(v_v_486_, 0);
lean_inc(v_a_488_);
v_v_489_ = lean_ctor_get(v_v_486_, 1);
lean_inc_n(v_v_489_, 2);
lean_dec(v_v_486_);
v_b_490_ = lean_ctor_get(v_w_487_, 0);
lean_inc(v_b_490_);
v_w_491_ = lean_ctor_get(v_w_487_, 1);
lean_inc_n(v_w_491_, 2);
lean_dec(v_w_487_);
v___x_492_ = lean_apply_2(v_ih_485_, v_v_489_, v_w_491_);
v___x_493_ = lean_apply_6(v_cons_483_, v_n_484_, v_a_488_, v_b_490_, v_v_489_, v_w_491_, v___x_492_);
return v___x_493_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_Vector_inductionOn_u2082___redArg(lean_object* v_n_494_, lean_object* v_v_495_, lean_object* v_w_496_, lean_object* v_nil_497_, lean_object* v_cons_498_){
_start:
{
lean_object* v___f_499_; lean_object* v___f_500_; lean_object* v___x_274__overap_501_; lean_object* v___x_502_; 
v___f_499_ = lean_alloc_closure((void*)(lp_mathlib_List_Vector_inductionOn_u2082___redArg___lam__0___boxed), 3, 1);
lean_closure_set(v___f_499_, 0, v_nil_497_);
v___f_500_ = lean_alloc_closure((void*)(lp_mathlib_List_Vector_inductionOn_u2082___redArg___lam__1), 5, 1);
lean_closure_set(v___f_500_, 0, v_cons_498_);
v___x_274__overap_501_ = l_Nat_recCompiled___redArg(v___f_499_, v___f_500_, v_n_494_);
lean_dec_ref(v___f_499_);
v___x_502_ = lean_apply_2(v___x_274__overap_501_, v_v_495_, v_w_496_);
return v___x_502_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_Vector_inductionOn_u2082___redArg___boxed(lean_object* v_n_503_, lean_object* v_v_504_, lean_object* v_w_505_, lean_object* v_nil_506_, lean_object* v_cons_507_){
_start:
{
lean_object* v_res_508_; 
v_res_508_ = lp_mathlib_List_Vector_inductionOn_u2082___redArg(v_n_503_, v_v_504_, v_w_505_, v_nil_506_, v_cons_507_);
lean_dec(v_n_503_);
return v_res_508_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_Vector_inductionOn_u2082(lean_object* v_00_u03b1_509_, lean_object* v_n_510_, lean_object* v_00_u03b2_511_, lean_object* v_C_512_, lean_object* v_v_513_, lean_object* v_w_514_, lean_object* v_nil_515_, lean_object* v_cons_516_){
_start:
{
lean_object* v___x_517_; 
v___x_517_ = lp_mathlib_List_Vector_inductionOn_u2082___redArg(v_n_510_, v_v_513_, v_w_514_, v_nil_515_, v_cons_516_);
return v___x_517_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_Vector_inductionOn_u2082___boxed(lean_object* v_00_u03b1_518_, lean_object* v_n_519_, lean_object* v_00_u03b2_520_, lean_object* v_C_521_, lean_object* v_v_522_, lean_object* v_w_523_, lean_object* v_nil_524_, lean_object* v_cons_525_){
_start:
{
lean_object* v_res_526_; 
v_res_526_ = lp_mathlib_List_Vector_inductionOn_u2082(v_00_u03b1_518_, v_n_519_, v_00_u03b2_520_, v_C_521_, v_v_522_, v_w_523_, v_nil_524_, v_cons_525_);
lean_dec(v_n_519_);
return v_res_526_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_Vector_inductionOn_u2083___redArg___lam__0(lean_object* v_nil_527_, lean_object* v_u_528_, lean_object* v_v_529_, lean_object* v_w_530_){
_start:
{
lean_inc(v_nil_527_);
return v_nil_527_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_Vector_inductionOn_u2083___redArg___lam__0___boxed(lean_object* v_nil_531_, lean_object* v_u_532_, lean_object* v_v_533_, lean_object* v_w_534_){
_start:
{
lean_object* v_res_535_; 
v_res_535_ = lp_mathlib_List_Vector_inductionOn_u2083___redArg___lam__0(v_nil_531_, v_u_532_, v_v_533_, v_w_534_);
lean_dec(v_w_534_);
lean_dec(v_v_533_);
lean_dec(v_u_532_);
lean_dec(v_nil_531_);
return v_res_535_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_Vector_inductionOn_u2083___redArg___lam__1(lean_object* v_cons_536_, lean_object* v_n_537_, lean_object* v_ih_538_, lean_object* v_u_539_, lean_object* v_v_540_, lean_object* v_w_541_){
_start:
{
lean_object* v_a_542_; lean_object* v_u_543_; lean_object* v_b_544_; lean_object* v_v_545_; lean_object* v_c_546_; lean_object* v_w_547_; lean_object* v___x_548_; lean_object* v___x_549_; 
v_a_542_ = lean_ctor_get(v_u_539_, 0);
lean_inc(v_a_542_);
v_u_543_ = lean_ctor_get(v_u_539_, 1);
lean_inc_n(v_u_543_, 2);
lean_dec(v_u_539_);
v_b_544_ = lean_ctor_get(v_v_540_, 0);
lean_inc(v_b_544_);
v_v_545_ = lean_ctor_get(v_v_540_, 1);
lean_inc_n(v_v_545_, 2);
lean_dec(v_v_540_);
v_c_546_ = lean_ctor_get(v_w_541_, 0);
lean_inc(v_c_546_);
v_w_547_ = lean_ctor_get(v_w_541_, 1);
lean_inc_n(v_w_547_, 2);
lean_dec(v_w_541_);
v___x_548_ = lean_apply_3(v_ih_538_, v_u_543_, v_v_545_, v_w_547_);
v___x_549_ = lean_apply_8(v_cons_536_, v_n_537_, v_a_542_, v_b_544_, v_c_546_, v_u_543_, v_v_545_, v_w_547_, v___x_548_);
return v___x_549_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_Vector_inductionOn_u2083___redArg(lean_object* v_n_550_, lean_object* v_u_551_, lean_object* v_v_552_, lean_object* v_w_553_, lean_object* v_nil_554_, lean_object* v_cons_555_){
_start:
{
lean_object* v___f_556_; lean_object* v___f_557_; lean_object* v___x_536__overap_558_; lean_object* v___x_559_; 
v___f_556_ = lean_alloc_closure((void*)(lp_mathlib_List_Vector_inductionOn_u2083___redArg___lam__0___boxed), 4, 1);
lean_closure_set(v___f_556_, 0, v_nil_554_);
v___f_557_ = lean_alloc_closure((void*)(lp_mathlib_List_Vector_inductionOn_u2083___redArg___lam__1), 6, 1);
lean_closure_set(v___f_557_, 0, v_cons_555_);
v___x_536__overap_558_ = l_Nat_recCompiled___redArg(v___f_556_, v___f_557_, v_n_550_);
lean_dec_ref(v___f_556_);
v___x_559_ = lean_apply_3(v___x_536__overap_558_, v_u_551_, v_v_552_, v_w_553_);
return v___x_559_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_Vector_inductionOn_u2083___redArg___boxed(lean_object* v_n_560_, lean_object* v_u_561_, lean_object* v_v_562_, lean_object* v_w_563_, lean_object* v_nil_564_, lean_object* v_cons_565_){
_start:
{
lean_object* v_res_566_; 
v_res_566_ = lp_mathlib_List_Vector_inductionOn_u2083___redArg(v_n_560_, v_u_561_, v_v_562_, v_w_563_, v_nil_564_, v_cons_565_);
lean_dec(v_n_560_);
return v_res_566_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_Vector_inductionOn_u2083(lean_object* v_00_u03b1_567_, lean_object* v_n_568_, lean_object* v_00_u03b2_569_, lean_object* v_00_u03b3_570_, lean_object* v_C_571_, lean_object* v_u_572_, lean_object* v_v_573_, lean_object* v_w_574_, lean_object* v_nil_575_, lean_object* v_cons_576_){
_start:
{
lean_object* v___x_577_; 
v___x_577_ = lp_mathlib_List_Vector_inductionOn_u2083___redArg(v_n_568_, v_u_572_, v_v_573_, v_w_574_, v_nil_575_, v_cons_576_);
return v___x_577_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_Vector_inductionOn_u2083___boxed(lean_object* v_00_u03b1_578_, lean_object* v_n_579_, lean_object* v_00_u03b2_580_, lean_object* v_00_u03b3_581_, lean_object* v_C_582_, lean_object* v_u_583_, lean_object* v_v_584_, lean_object* v_w_585_, lean_object* v_nil_586_, lean_object* v_cons_587_){
_start:
{
lean_object* v_res_588_; 
v_res_588_ = lp_mathlib_List_Vector_inductionOn_u2083(v_00_u03b1_578_, v_n_579_, v_00_u03b2_580_, v_00_u03b3_581_, v_C_582_, v_u_583_, v_v_584_, v_w_585_, v_nil_586_, v_cons_587_);
lean_dec(v_n_579_);
return v_res_588_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_Vector_casesOn___redArg___lam__0(lean_object* v_cons_589_, lean_object* v_x_590_, lean_object* v_hd_591_, lean_object* v_tl_592_, lean_object* v_x_593_){
_start:
{
lean_object* v___x_594_; 
v___x_594_ = lean_apply_3(v_cons_589_, v_x_590_, v_hd_591_, v_tl_592_);
return v___x_594_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_Vector_casesOn___redArg___lam__0___boxed(lean_object* v_cons_595_, lean_object* v_x_596_, lean_object* v_hd_597_, lean_object* v_tl_598_, lean_object* v_x_599_){
_start:
{
lean_object* v_res_600_; 
v_res_600_ = lp_mathlib_List_Vector_casesOn___redArg___lam__0(v_cons_595_, v_x_596_, v_hd_597_, v_tl_598_, v_x_599_);
lean_dec(v_x_599_);
return v_res_600_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_Vector_casesOn___redArg(lean_object* v_m_601_, lean_object* v_v_602_, lean_object* v_nil_603_, lean_object* v_cons_604_){
_start:
{
lean_object* v___f_605_; lean_object* v___x_606_; 
v___f_605_ = lean_alloc_closure((void*)(lp_mathlib_List_Vector_casesOn___redArg___lam__0___boxed), 5, 1);
lean_closure_set(v___f_605_, 0, v_cons_604_);
v___x_606_ = lp_mathlib_List_Vector_inductionOn___redArg(v_m_601_, v_v_602_, v_nil_603_, v___f_605_);
return v___x_606_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_Vector_casesOn___redArg___boxed(lean_object* v_m_607_, lean_object* v_v_608_, lean_object* v_nil_609_, lean_object* v_cons_610_){
_start:
{
lean_object* v_res_611_; 
v_res_611_ = lp_mathlib_List_Vector_casesOn___redArg(v_m_607_, v_v_608_, v_nil_609_, v_cons_610_);
lean_dec(v_m_607_);
return v_res_611_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_Vector_casesOn(lean_object* v_00_u03b1_612_, lean_object* v_m_613_, lean_object* v_motive_614_, lean_object* v_v_615_, lean_object* v_nil_616_, lean_object* v_cons_617_){
_start:
{
lean_object* v___x_618_; 
v___x_618_ = lp_mathlib_List_Vector_casesOn___redArg(v_m_613_, v_v_615_, v_nil_616_, v_cons_617_);
return v___x_618_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_Vector_casesOn___boxed(lean_object* v_00_u03b1_619_, lean_object* v_m_620_, lean_object* v_motive_621_, lean_object* v_v_622_, lean_object* v_nil_623_, lean_object* v_cons_624_){
_start:
{
lean_object* v_res_625_; 
v_res_625_ = lp_mathlib_List_Vector_casesOn(v_00_u03b1_619_, v_m_620_, v_motive_621_, v_v_622_, v_nil_623_, v_cons_624_);
lean_dec(v_m_620_);
return v_res_625_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_Vector_casesOn_u2082___redArg___lam__0(lean_object* v_cons_626_, lean_object* v_x_627_, lean_object* v_x_628_, lean_object* v_y_629_, lean_object* v_xs_630_, lean_object* v_ys_631_, lean_object* v_x_632_){
_start:
{
lean_object* v___x_633_; 
v___x_633_ = lean_apply_5(v_cons_626_, v_x_627_, v_x_628_, v_y_629_, v_xs_630_, v_ys_631_);
return v___x_633_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_Vector_casesOn_u2082___redArg___lam__0___boxed(lean_object* v_cons_634_, lean_object* v_x_635_, lean_object* v_x_636_, lean_object* v_y_637_, lean_object* v_xs_638_, lean_object* v_ys_639_, lean_object* v_x_640_){
_start:
{
lean_object* v_res_641_; 
v_res_641_ = lp_mathlib_List_Vector_casesOn_u2082___redArg___lam__0(v_cons_634_, v_x_635_, v_x_636_, v_y_637_, v_xs_638_, v_ys_639_, v_x_640_);
lean_dec(v_x_640_);
return v_res_641_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_Vector_casesOn_u2082___redArg(lean_object* v_m_642_, lean_object* v_v_u2081_643_, lean_object* v_v_u2082_644_, lean_object* v_nil_645_, lean_object* v_cons_646_){
_start:
{
lean_object* v___f_647_; lean_object* v___x_648_; 
v___f_647_ = lean_alloc_closure((void*)(lp_mathlib_List_Vector_casesOn_u2082___redArg___lam__0___boxed), 7, 1);
lean_closure_set(v___f_647_, 0, v_cons_646_);
v___x_648_ = lp_mathlib_List_Vector_inductionOn_u2082___redArg(v_m_642_, v_v_u2081_643_, v_v_u2082_644_, v_nil_645_, v___f_647_);
return v___x_648_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_Vector_casesOn_u2082___redArg___boxed(lean_object* v_m_649_, lean_object* v_v_u2081_650_, lean_object* v_v_u2082_651_, lean_object* v_nil_652_, lean_object* v_cons_653_){
_start:
{
lean_object* v_res_654_; 
v_res_654_ = lp_mathlib_List_Vector_casesOn_u2082___redArg(v_m_649_, v_v_u2081_650_, v_v_u2082_651_, v_nil_652_, v_cons_653_);
lean_dec(v_m_649_);
return v_res_654_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_Vector_casesOn_u2082(lean_object* v_00_u03b1_655_, lean_object* v_m_656_, lean_object* v_00_u03b2_657_, lean_object* v_motive_658_, lean_object* v_v_u2081_659_, lean_object* v_v_u2082_660_, lean_object* v_nil_661_, lean_object* v_cons_662_){
_start:
{
lean_object* v___x_663_; 
v___x_663_ = lp_mathlib_List_Vector_casesOn_u2082___redArg(v_m_656_, v_v_u2081_659_, v_v_u2082_660_, v_nil_661_, v_cons_662_);
return v___x_663_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_Vector_casesOn_u2082___boxed(lean_object* v_00_u03b1_664_, lean_object* v_m_665_, lean_object* v_00_u03b2_666_, lean_object* v_motive_667_, lean_object* v_v_u2081_668_, lean_object* v_v_u2082_669_, lean_object* v_nil_670_, lean_object* v_cons_671_){
_start:
{
lean_object* v_res_672_; 
v_res_672_ = lp_mathlib_List_Vector_casesOn_u2082(v_00_u03b1_664_, v_m_665_, v_00_u03b2_666_, v_motive_667_, v_v_u2081_668_, v_v_u2082_669_, v_nil_670_, v_cons_671_);
lean_dec(v_m_665_);
return v_res_672_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_Vector_casesOn_u2083___redArg___lam__0(lean_object* v_cons_673_, lean_object* v_x_674_, lean_object* v_x_675_, lean_object* v_y_676_, lean_object* v_z_677_, lean_object* v_xs_678_, lean_object* v_ys_679_, lean_object* v_zs_680_, lean_object* v_x_681_){
_start:
{
lean_object* v___x_682_; 
v___x_682_ = lean_apply_7(v_cons_673_, v_x_674_, v_x_675_, v_y_676_, v_z_677_, v_xs_678_, v_ys_679_, v_zs_680_);
return v___x_682_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_Vector_casesOn_u2083___redArg___lam__0___boxed(lean_object* v_cons_683_, lean_object* v_x_684_, lean_object* v_x_685_, lean_object* v_y_686_, lean_object* v_z_687_, lean_object* v_xs_688_, lean_object* v_ys_689_, lean_object* v_zs_690_, lean_object* v_x_691_){
_start:
{
lean_object* v_res_692_; 
v_res_692_ = lp_mathlib_List_Vector_casesOn_u2083___redArg___lam__0(v_cons_683_, v_x_684_, v_x_685_, v_y_686_, v_z_687_, v_xs_688_, v_ys_689_, v_zs_690_, v_x_691_);
lean_dec(v_x_691_);
return v_res_692_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_Vector_casesOn_u2083___redArg(lean_object* v_m_693_, lean_object* v_v_u2081_694_, lean_object* v_v_u2082_695_, lean_object* v_v_u2083_696_, lean_object* v_nil_697_, lean_object* v_cons_698_){
_start:
{
lean_object* v___f_699_; lean_object* v___x_700_; 
v___f_699_ = lean_alloc_closure((void*)(lp_mathlib_List_Vector_casesOn_u2083___redArg___lam__0___boxed), 9, 1);
lean_closure_set(v___f_699_, 0, v_cons_698_);
v___x_700_ = lp_mathlib_List_Vector_inductionOn_u2083___redArg(v_m_693_, v_v_u2081_694_, v_v_u2082_695_, v_v_u2083_696_, v_nil_697_, v___f_699_);
return v___x_700_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_Vector_casesOn_u2083___redArg___boxed(lean_object* v_m_701_, lean_object* v_v_u2081_702_, lean_object* v_v_u2082_703_, lean_object* v_v_u2083_704_, lean_object* v_nil_705_, lean_object* v_cons_706_){
_start:
{
lean_object* v_res_707_; 
v_res_707_ = lp_mathlib_List_Vector_casesOn_u2083___redArg(v_m_701_, v_v_u2081_702_, v_v_u2082_703_, v_v_u2083_704_, v_nil_705_, v_cons_706_);
lean_dec(v_m_701_);
return v_res_707_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_Vector_casesOn_u2083(lean_object* v_00_u03b1_708_, lean_object* v_m_709_, lean_object* v_00_u03b2_710_, lean_object* v_00_u03b3_711_, lean_object* v_motive_712_, lean_object* v_v_u2081_713_, lean_object* v_v_u2082_714_, lean_object* v_v_u2083_715_, lean_object* v_nil_716_, lean_object* v_cons_717_){
_start:
{
lean_object* v___x_718_; 
v___x_718_ = lp_mathlib_List_Vector_casesOn_u2083___redArg(v_m_709_, v_v_u2081_713_, v_v_u2082_714_, v_v_u2083_715_, v_nil_716_, v_cons_717_);
return v___x_718_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_Vector_casesOn_u2083___boxed(lean_object* v_00_u03b1_719_, lean_object* v_m_720_, lean_object* v_00_u03b2_721_, lean_object* v_00_u03b3_722_, lean_object* v_motive_723_, lean_object* v_v_u2081_724_, lean_object* v_v_u2082_725_, lean_object* v_v_u2083_726_, lean_object* v_nil_727_, lean_object* v_cons_728_){
_start:
{
lean_object* v_res_729_; 
v_res_729_ = lp_mathlib_List_Vector_casesOn_u2083(v_00_u03b1_719_, v_m_720_, v_00_u03b2_721_, v_00_u03b3_722_, v_motive_723_, v_v_u2081_724_, v_v_u2082_725_, v_v_u2083_726_, v_nil_727_, v_cons_728_);
lean_dec(v_m_720_);
return v_res_729_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_Vector_toArray___redArg(lean_object* v_x_730_){
_start:
{
lean_object* v___x_731_; 
v___x_731_ = lean_array_mk(v_x_730_);
return v___x_731_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_Vector_toArray(lean_object* v_00_u03b1_732_, lean_object* v_n_733_, lean_object* v_x_734_){
_start:
{
lean_object* v___x_735_; 
v___x_735_ = lean_array_mk(v_x_734_);
return v___x_735_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_Vector_toArray___boxed(lean_object* v_00_u03b1_736_, lean_object* v_n_737_, lean_object* v_x_738_){
_start:
{
lean_object* v_res_739_; 
v_res_739_ = lp_mathlib_List_Vector_toArray(v_00_u03b1_736_, v_n_737_, v_x_738_);
lean_dec(v_n_737_);
return v_res_739_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_Vector_insertIdx___redArg(lean_object* v_a_742_, lean_object* v_i_743_, lean_object* v_v_744_){
_start:
{
lean_object* v___x_745_; lean_object* v___x_746_; 
v___x_745_ = ((lean_object*)(lp_mathlib_List_Vector_insertIdx___redArg___closed__0));
v___x_746_ = l___private_Init_Data_List_Impl_0__List_insertIdxTR_go(lean_box(0), v_a_742_, v_i_743_, v_v_744_, v___x_745_);
return v___x_746_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_Vector_insertIdx(lean_object* v_00_u03b1_747_, lean_object* v_n_748_, lean_object* v_a_749_, lean_object* v_i_750_, lean_object* v_v_751_){
_start:
{
lean_object* v___x_752_; 
v___x_752_ = lp_mathlib_List_Vector_insertIdx___redArg(v_a_749_, v_i_750_, v_v_751_);
return v___x_752_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_Vector_insertIdx___boxed(lean_object* v_00_u03b1_753_, lean_object* v_n_754_, lean_object* v_a_755_, lean_object* v_i_756_, lean_object* v_v_757_){
_start:
{
lean_object* v_res_758_; 
v_res_758_ = lp_mathlib_List_Vector_insertIdx(v_00_u03b1_753_, v_n_754_, v_a_755_, v_i_756_, v_v_757_);
lean_dec(v_n_754_);
return v_res_758_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_Vector_set___redArg(lean_object* v_v_759_, lean_object* v_i_760_, lean_object* v_a_761_){
_start:
{
lean_object* v___x_762_; lean_object* v___x_763_; 
v___x_762_ = ((lean_object*)(lp_mathlib_List_Vector_insertIdx___redArg___closed__0));
lean_inc(v_v_759_);
v___x_763_ = l___private_Init_Data_List_Impl_0__List_setTR_go(lean_box(0), v_v_759_, v_a_761_, v_v_759_, v_i_760_, v___x_762_);
lean_dec(v_v_759_);
return v___x_763_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_Vector_set(lean_object* v_00_u03b1_764_, lean_object* v_n_765_, lean_object* v_v_766_, lean_object* v_i_767_, lean_object* v_a_768_){
_start:
{
lean_object* v___x_769_; 
v___x_769_ = lp_mathlib_List_Vector_set___redArg(v_v_766_, v_i_767_, v_a_768_);
return v___x_769_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_Vector_set___boxed(lean_object* v_00_u03b1_770_, lean_object* v_n_771_, lean_object* v_v_772_, lean_object* v_i_773_, lean_object* v_a_774_){
_start:
{
lean_object* v_res_775_; 
v_res_775_ = lp_mathlib_List_Vector_set(v_00_u03b1_770_, v_n_771_, v_v_772_, v_i_773_, v_a_774_);
lean_dec(v_n_771_);
return v_res_775_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Data_Vector_Basic_0__List_Vector_traverseAux___redArg(lean_object* v_inst_776_, lean_object* v_f_777_, lean_object* v_x_778_){
_start:
{
if (lean_obj_tag(v_x_778_) == 0)
{
lean_object* v_toPure_779_; lean_object* v___x_780_; lean_object* v___x_781_; 
lean_dec(v_f_777_);
v_toPure_779_ = lean_ctor_get(v_inst_776_, 1);
lean_inc(v_toPure_779_);
lean_dec_ref(v_inst_776_);
v___x_780_ = lean_box(0);
v___x_781_ = lean_apply_2(v_toPure_779_, lean_box(0), v___x_780_);
return v___x_781_;
}
else
{
lean_object* v_toFunctor_782_; lean_object* v_toSeq_783_; lean_object* v_head_784_; lean_object* v_tail_785_; lean_object* v_map_786_; lean_object* v___f_787_; lean_object* v___x_788_; lean_object* v___x_789_; lean_object* v___x_790_; lean_object* v___x_791_; lean_object* v___x_792_; 
v_toFunctor_782_ = lean_ctor_get(v_inst_776_, 0);
v_toSeq_783_ = lean_ctor_get(v_inst_776_, 2);
lean_inc(v_toSeq_783_);
v_head_784_ = lean_ctor_get(v_x_778_, 0);
lean_inc(v_head_784_);
v_tail_785_ = lean_ctor_get(v_x_778_, 1);
lean_inc_n(v_tail_785_, 2);
lean_dec_ref_known(v_x_778_, 2);
v_map_786_ = lean_ctor_get(v_toFunctor_782_, 0);
lean_inc(v_map_786_);
lean_inc(v_f_777_);
v___f_787_ = lean_alloc_closure((void*)(lp_mathlib___private_Mathlib_Data_Vector_Basic_0__List_Vector_traverseAux___redArg___lam__0), 4, 3);
lean_closure_set(v___f_787_, 0, v_inst_776_);
lean_closure_set(v___f_787_, 1, v_f_777_);
lean_closure_set(v___f_787_, 2, v_tail_785_);
v___x_788_ = l_List_lengthTR___redArg(v_tail_785_);
lean_dec(v_tail_785_);
v___x_789_ = lean_alloc_closure((void*)(lp_mathlib_List_Vector_cons___boxed), 4, 2);
lean_closure_set(v___x_789_, 0, lean_box(0));
lean_closure_set(v___x_789_, 1, v___x_788_);
v___x_790_ = lean_apply_1(v_f_777_, v_head_784_);
v___x_791_ = lean_apply_4(v_map_786_, lean_box(0), lean_box(0), v___x_789_, v___x_790_);
v___x_792_ = lean_apply_4(v_toSeq_783_, lean_box(0), lean_box(0), v___x_791_, v___f_787_);
return v___x_792_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Data_Vector_Basic_0__List_Vector_traverseAux___redArg___lam__0(lean_object* v_inst_793_, lean_object* v_f_794_, lean_object* v_tail_795_, lean_object* v_x_796_){
_start:
{
lean_object* v___x_797_; 
v___x_797_ = lp_mathlib___private_Mathlib_Data_Vector_Basic_0__List_Vector_traverseAux___redArg(v_inst_793_, v_f_794_, v_tail_795_);
return v___x_797_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Data_Vector_Basic_0__List_Vector_traverseAux(lean_object* v_F_798_, lean_object* v_inst_799_, lean_object* v_00_u03b1_800_, lean_object* v_00_u03b2_801_, lean_object* v_f_802_, lean_object* v_x_803_){
_start:
{
lean_object* v___x_804_; 
v___x_804_ = lp_mathlib___private_Mathlib_Data_Vector_Basic_0__List_Vector_traverseAux___redArg(v_inst_799_, v_f_802_, v_x_803_);
return v___x_804_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_Vector_traverse___redArg(lean_object* v_inst_805_, lean_object* v_f_806_, lean_object* v_x_807_){
_start:
{
lean_object* v___x_808_; 
v___x_808_ = lp_mathlib___private_Mathlib_Data_Vector_Basic_0__List_Vector_traverseAux___redArg(v_inst_805_, v_f_806_, v_x_807_);
return v___x_808_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_Vector_traverse(lean_object* v_n_809_, lean_object* v_F_810_, lean_object* v_inst_811_, lean_object* v_00_u03b1_812_, lean_object* v_00_u03b2_813_, lean_object* v_f_814_, lean_object* v_x_815_){
_start:
{
lean_object* v___x_816_; 
v___x_816_ = lp_mathlib___private_Mathlib_Data_Vector_Basic_0__List_Vector_traverseAux___redArg(v_inst_811_, v_f_814_, v_x_815_);
return v___x_816_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_Vector_traverse___boxed(lean_object* v_n_817_, lean_object* v_F_818_, lean_object* v_inst_819_, lean_object* v_00_u03b1_820_, lean_object* v_00_u03b2_821_, lean_object* v_f_822_, lean_object* v_x_823_){
_start:
{
lean_object* v_res_824_; 
v_res_824_ = lp_mathlib_List_Vector_traverse(v_n_817_, v_F_818_, v_inst_819_, v_00_u03b1_820_, v_00_u03b2_821_, v_f_822_, v_x_823_);
lean_dec(v_n_817_);
return v_res_824_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_Vector_instTraversableFlipNat___lam__0(lean_object* v_00_u03b1_825_, lean_object* v_00_u03b2_826_, lean_object* v___y_827_, lean_object* v___y_828_){
_start:
{
lean_object* v___x_829_; 
v___x_829_ = lp_mathlib_List_Vector_map___redArg(v___y_827_, v___y_828_);
return v___x_829_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_Vector_instTraversableFlipNat___lam__1(lean_object* v_00_u03b1_830_, lean_object* v_00_u03b2_831_, lean_object* v___y_832_, lean_object* v___y_833_){
_start:
{
lean_object* v___x_834_; lean_object* v___x_835_; 
v___x_834_ = lean_alloc_closure((void*)(l_Function_const___boxed), 4, 3);
lean_closure_set(v___x_834_, 0, lean_box(0));
lean_closure_set(v___x_834_, 1, lean_box(0));
lean_closure_set(v___x_834_, 2, v___y_832_);
v___x_835_ = lp_mathlib_List_Vector_map___redArg(v___x_834_, v___y_833_);
return v___x_835_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_Vector_instTraversableFlipNat(lean_object* v_n_841_){
_start:
{
lean_object* v___x_842_; lean_object* v___x_843_; lean_object* v___x_844_; 
v___x_842_ = ((lean_object*)(lp_mathlib_List_Vector_instTraversableFlipNat___closed__2));
v___x_843_ = lean_alloc_closure((void*)(lp_mathlib_List_Vector_traverse___boxed), 7, 1);
lean_closure_set(v___x_843_, 0, v_n_841_);
v___x_844_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_844_, 0, v___x_842_);
lean_ctor_set(v___x_844_, 1, v___x_843_);
return v___x_844_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Data_Vector_Defs(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Data_List_Nodup(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Control_Applicative(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Control_Traversable_Basic(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_BigOperators_Group_List_Basic(uint8_t builtin);
lean_object* runtime_initialize_batteries_Batteries_Data_Fin_Lemmas(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Data_Fin_SuccPred(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_Data_Vector_Basic(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Data_Vector_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Data_List_Nodup(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Control_Applicative(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Control_Traversable_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_BigOperators_Group_List_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_batteries_Batteries_Data_Fin_Lemmas(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Data_Fin_SuccPred(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_Data_Vector_Basic(uint8_t builtin) {
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
lean_object* initialize_mathlib_Mathlib_Data_Vector_Defs(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Data_List_Nodup(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Control_Applicative(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Control_Traversable_Basic(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_BigOperators_Group_List_Basic(uint8_t builtin);
lean_object* initialize_batteries_Batteries_Data_Fin_Lemmas(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Data_Fin_SuccPred(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Data_Vector_Basic(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_Vector_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_List_Nodup(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Control_Applicative(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Control_Traversable_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_BigOperators_Group_List_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_batteries_Batteries_Data_Fin_Lemmas(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_Fin_SuccPred(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Data_Vector_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_Data_Vector_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_Data_Vector_Basic(builtin);
}
#ifdef __cplusplus
}
#endif
