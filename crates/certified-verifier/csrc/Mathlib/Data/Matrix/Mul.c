// Lean compiler output
// Module: Mathlib.Data.Matrix.Mul
// Imports: public import Init public meta import Init public import Mathlib.Algebra.BigOperators.GroupWithZero.Action public import Mathlib.Algebra.BigOperators.Ring.Finset public import Mathlib.Algebra.Regular.Basic public import Mathlib.Algebra.Ring.Subsemiring.Defs public import Mathlib.Data.Fintype.BigOperators public import Mathlib.Data.Matrix.Diagonal public import Mathlib.Algebra.Order.BigOperators.Group.Finset
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
lean_object* l_Lean_Name_mkStr2(lean_object*, lean_object*);
lean_object* lp_mathlib_NonUnitalNonAssocSemiring_toDistrib___redArg(lean_object*);
lean_object* lp_mathlib_Finset_sum___redArg(lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_AddMonoid_toAddZeroClass___redArg(lean_object*);
lean_object* lp_mathlib_AddZeroClass_toAddZero___redArg(lean_object*);
lean_object* lp_mathlib_Matrix_one___redArg(lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Matrix_addMonoid___redArg(lean_object*);
lean_object* l_Lean_Name_mkStr4(lean_object*, lean_object*, lean_object*, lean_object*);
uint8_t l_Lean_Syntax_isOfKind(lean_object*, lean_object*);
lean_object* l_Lean_Syntax_getArg(lean_object*, lean_object*);
uint8_t l_Lean_Syntax_matchesNull(lean_object*, lean_object*);
lean_object* l_Lean_replaceRef(lean_object*, lean_object*);
lean_object* l_Lean_SourceInfo_fromRef(lean_object*, uint8_t);
lean_object* l_Lean_Syntax_node3(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Semiring_toNonUnitalSemiring___redArg(lean_object*);
lean_object* lp_mathlib_Semiring_toNonAssocSemiring___redArg(lean_object*);
lean_object* lp_mathlib_NonAssocSemiring_toAddCommMonoidWithOne___redArg(lean_object*);
lean_object* lp_mathlib_Matrix_instAddCommMonoidWithOne___redArg(lean_object*, lean_object*);
lean_object* lp_mathlib_npowBinRecAuto___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_NonUnitalRing_toNonUnitalSemiring___redArg(lean_object*);
lean_object* lp_mathlib_Matrix_addGroup___redArg(lean_object*);
lean_object* l_String_toRawSubstring_x27(lean_object*);
lean_object* lp_mathlib_NonUnitalNonAssocRing_toNonUnitalNonAssocSemiring___redArg(lean_object*);
lean_object* l_Lean_addMacroScope(lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Syntax_node2(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Ring_toNonAssocRing___redArg(lean_object*);
lean_object* lp_mathlib_NonAssocRing_toAddCommGroupWithOne___redArg(lean_object*);
lean_object* lp_mathlib_Matrix_instAddCommGroupWithOne___redArg(lean_object*, lean_object*);
lean_object* lp_mathlib_Equiv_refl(lean_object*);
lean_object* lp_mathlib_NonAssocRing_toNonAssocSemiring___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_dotProduct___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_dotProduct___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_dotProduct___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_dotProduct(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_dotProduct___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_term___u2b1d_u1d65___00__closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 13, .m_capacity = 13, .m_length = 8, .m_data = "term_⬝ᵥ_"};
static const lean_object* lp_mathlib_term___u2b1d_u1d65___00__closed__0 = (const lean_object*)&lp_mathlib_term___u2b1d_u1d65___00__closed__0_value;
static const lean_ctor_object lp_mathlib_term___u2b1d_u1d65___00__closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_term___u2b1d_u1d65___00__closed__0_value),LEAN_SCALAR_PTR_LITERAL(2, 12, 48, 192, 30, 103, 94, 131)}};
static const lean_object* lp_mathlib_term___u2b1d_u1d65___00__closed__1 = (const lean_object*)&lp_mathlib_term___u2b1d_u1d65___00__closed__1_value;
static const lean_string_object lp_mathlib_term___u2b1d_u1d65___00__closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "andthen"};
static const lean_object* lp_mathlib_term___u2b1d_u1d65___00__closed__2 = (const lean_object*)&lp_mathlib_term___u2b1d_u1d65___00__closed__2_value;
static const lean_ctor_object lp_mathlib_term___u2b1d_u1d65___00__closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_term___u2b1d_u1d65___00__closed__2_value),LEAN_SCALAR_PTR_LITERAL(40, 255, 78, 30, 143, 119, 117, 174)}};
static const lean_object* lp_mathlib_term___u2b1d_u1d65___00__closed__3 = (const lean_object*)&lp_mathlib_term___u2b1d_u1d65___00__closed__3_value;
static const lean_string_object lp_mathlib_term___u2b1d_u1d65___00__closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 4, .m_data = " ⬝ᵥ "};
static const lean_object* lp_mathlib_term___u2b1d_u1d65___00__closed__4 = (const lean_object*)&lp_mathlib_term___u2b1d_u1d65___00__closed__4_value;
static const lean_ctor_object lp_mathlib_term___u2b1d_u1d65___00__closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 5}, .m_objs = {((lean_object*)&lp_mathlib_term___u2b1d_u1d65___00__closed__4_value)}};
static const lean_object* lp_mathlib_term___u2b1d_u1d65___00__closed__5 = (const lean_object*)&lp_mathlib_term___u2b1d_u1d65___00__closed__5_value;
static const lean_string_object lp_mathlib_term___u2b1d_u1d65___00__closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "term"};
static const lean_object* lp_mathlib_term___u2b1d_u1d65___00__closed__6 = (const lean_object*)&lp_mathlib_term___u2b1d_u1d65___00__closed__6_value;
static const lean_ctor_object lp_mathlib_term___u2b1d_u1d65___00__closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_term___u2b1d_u1d65___00__closed__6_value),LEAN_SCALAR_PTR_LITERAL(187, 230, 181, 162, 253, 146, 122, 119)}};
static const lean_object* lp_mathlib_term___u2b1d_u1d65___00__closed__7 = (const lean_object*)&lp_mathlib_term___u2b1d_u1d65___00__closed__7_value;
static const lean_ctor_object lp_mathlib_term___u2b1d_u1d65___00__closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 7}, .m_objs = {((lean_object*)&lp_mathlib_term___u2b1d_u1d65___00__closed__7_value),((lean_object*)(((size_t)(73) << 1) | 1))}};
static const lean_object* lp_mathlib_term___u2b1d_u1d65___00__closed__8 = (const lean_object*)&lp_mathlib_term___u2b1d_u1d65___00__closed__8_value;
static const lean_ctor_object lp_mathlib_term___u2b1d_u1d65___00__closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib_term___u2b1d_u1d65___00__closed__3_value),((lean_object*)&lp_mathlib_term___u2b1d_u1d65___00__closed__5_value),((lean_object*)&lp_mathlib_term___u2b1d_u1d65___00__closed__8_value)}};
static const lean_object* lp_mathlib_term___u2b1d_u1d65___00__closed__9 = (const lean_object*)&lp_mathlib_term___u2b1d_u1d65___00__closed__9_value;
static const lean_ctor_object lp_mathlib_term___u2b1d_u1d65___00__closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*4 + 0, .m_other = 4, .m_tag = 4}, .m_objs = {((lean_object*)&lp_mathlib_term___u2b1d_u1d65___00__closed__1_value),((lean_object*)(((size_t)(72) << 1) | 1)),((lean_object*)(((size_t)(72) << 1) | 1)),((lean_object*)&lp_mathlib_term___u2b1d_u1d65___00__closed__9_value)}};
static const lean_object* lp_mathlib_term___u2b1d_u1d65___00__closed__10 = (const lean_object*)&lp_mathlib_term___u2b1d_u1d65___00__closed__10_value;
LEAN_EXPORT const lean_object* lp_mathlib_term___u2b1d_u1d65__ = (const lean_object*)&lp_mathlib_term___u2b1d_u1d65___00__closed__10_value;
static const lean_string_object lp_mathlib___aux__Mathlib__Data__Matrix__Mul______macroRules__term___u2b1d_u1d65____1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "Lean"};
static const lean_object* lp_mathlib___aux__Mathlib__Data__Matrix__Mul______macroRules__term___u2b1d_u1d65____1___closed__0 = (const lean_object*)&lp_mathlib___aux__Mathlib__Data__Matrix__Mul______macroRules__term___u2b1d_u1d65____1___closed__0_value;
static const lean_string_object lp_mathlib___aux__Mathlib__Data__Matrix__Mul______macroRules__term___u2b1d_u1d65____1___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "Parser"};
static const lean_object* lp_mathlib___aux__Mathlib__Data__Matrix__Mul______macroRules__term___u2b1d_u1d65____1___closed__1 = (const lean_object*)&lp_mathlib___aux__Mathlib__Data__Matrix__Mul______macroRules__term___u2b1d_u1d65____1___closed__1_value;
static const lean_string_object lp_mathlib___aux__Mathlib__Data__Matrix__Mul______macroRules__term___u2b1d_u1d65____1___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "Term"};
static const lean_object* lp_mathlib___aux__Mathlib__Data__Matrix__Mul______macroRules__term___u2b1d_u1d65____1___closed__2 = (const lean_object*)&lp_mathlib___aux__Mathlib__Data__Matrix__Mul______macroRules__term___u2b1d_u1d65____1___closed__2_value;
static const lean_string_object lp_mathlib___aux__Mathlib__Data__Matrix__Mul______macroRules__term___u2b1d_u1d65____1___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = "app"};
static const lean_object* lp_mathlib___aux__Mathlib__Data__Matrix__Mul______macroRules__term___u2b1d_u1d65____1___closed__3 = (const lean_object*)&lp_mathlib___aux__Mathlib__Data__Matrix__Mul______macroRules__term___u2b1d_u1d65____1___closed__3_value;
static const lean_ctor_object lp_mathlib___aux__Mathlib__Data__Matrix__Mul______macroRules__term___u2b1d_u1d65____1___closed__4_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___aux__Mathlib__Data__Matrix__Mul______macroRules__term___u2b1d_u1d65____1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib___aux__Mathlib__Data__Matrix__Mul______macroRules__term___u2b1d_u1d65____1___closed__4_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___aux__Mathlib__Data__Matrix__Mul______macroRules__term___u2b1d_u1d65____1___closed__4_value_aux_0),((lean_object*)&lp_mathlib___aux__Mathlib__Data__Matrix__Mul______macroRules__term___u2b1d_u1d65____1___closed__1_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib___aux__Mathlib__Data__Matrix__Mul______macroRules__term___u2b1d_u1d65____1___closed__4_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___aux__Mathlib__Data__Matrix__Mul______macroRules__term___u2b1d_u1d65____1___closed__4_value_aux_1),((lean_object*)&lp_mathlib___aux__Mathlib__Data__Matrix__Mul______macroRules__term___u2b1d_u1d65____1___closed__2_value),LEAN_SCALAR_PTR_LITERAL(75, 170, 162, 138, 136, 204, 251, 229)}};
static const lean_ctor_object lp_mathlib___aux__Mathlib__Data__Matrix__Mul______macroRules__term___u2b1d_u1d65____1___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___aux__Mathlib__Data__Matrix__Mul______macroRules__term___u2b1d_u1d65____1___closed__4_value_aux_2),((lean_object*)&lp_mathlib___aux__Mathlib__Data__Matrix__Mul______macroRules__term___u2b1d_u1d65____1___closed__3_value),LEAN_SCALAR_PTR_LITERAL(69, 118, 10, 41, 220, 156, 243, 179)}};
static const lean_object* lp_mathlib___aux__Mathlib__Data__Matrix__Mul______macroRules__term___u2b1d_u1d65____1___closed__4 = (const lean_object*)&lp_mathlib___aux__Mathlib__Data__Matrix__Mul______macroRules__term___u2b1d_u1d65____1___closed__4_value;
static const lean_string_object lp_mathlib___aux__Mathlib__Data__Matrix__Mul______macroRules__term___u2b1d_u1d65____1___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 11, .m_capacity = 11, .m_length = 10, .m_data = "dotProduct"};
static const lean_object* lp_mathlib___aux__Mathlib__Data__Matrix__Mul______macroRules__term___u2b1d_u1d65____1___closed__5 = (const lean_object*)&lp_mathlib___aux__Mathlib__Data__Matrix__Mul______macroRules__term___u2b1d_u1d65____1___closed__5_value;
static lean_once_cell_t lp_mathlib___aux__Mathlib__Data__Matrix__Mul______macroRules__term___u2b1d_u1d65____1___closed__6_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___aux__Mathlib__Data__Matrix__Mul______macroRules__term___u2b1d_u1d65____1___closed__6;
static const lean_ctor_object lp_mathlib___aux__Mathlib__Data__Matrix__Mul______macroRules__term___u2b1d_u1d65____1___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___aux__Mathlib__Data__Matrix__Mul______macroRules__term___u2b1d_u1d65____1___closed__5_value),LEAN_SCALAR_PTR_LITERAL(238, 192, 250, 227, 154, 254, 137, 32)}};
static const lean_object* lp_mathlib___aux__Mathlib__Data__Matrix__Mul______macroRules__term___u2b1d_u1d65____1___closed__7 = (const lean_object*)&lp_mathlib___aux__Mathlib__Data__Matrix__Mul______macroRules__term___u2b1d_u1d65____1___closed__7_value;
static const lean_ctor_object lp_mathlib___aux__Mathlib__Data__Matrix__Mul______macroRules__term___u2b1d_u1d65____1___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___aux__Mathlib__Data__Matrix__Mul______macroRules__term___u2b1d_u1d65____1___closed__7_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib___aux__Mathlib__Data__Matrix__Mul______macroRules__term___u2b1d_u1d65____1___closed__8 = (const lean_object*)&lp_mathlib___aux__Mathlib__Data__Matrix__Mul______macroRules__term___u2b1d_u1d65____1___closed__8_value;
static const lean_ctor_object lp_mathlib___aux__Mathlib__Data__Matrix__Mul______macroRules__term___u2b1d_u1d65____1___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___aux__Mathlib__Data__Matrix__Mul______macroRules__term___u2b1d_u1d65____1___closed__8_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib___aux__Mathlib__Data__Matrix__Mul______macroRules__term___u2b1d_u1d65____1___closed__9 = (const lean_object*)&lp_mathlib___aux__Mathlib__Data__Matrix__Mul______macroRules__term___u2b1d_u1d65____1___closed__9_value;
static const lean_string_object lp_mathlib___aux__Mathlib__Data__Matrix__Mul______macroRules__term___u2b1d_u1d65____1___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "null"};
static const lean_object* lp_mathlib___aux__Mathlib__Data__Matrix__Mul______macroRules__term___u2b1d_u1d65____1___closed__10 = (const lean_object*)&lp_mathlib___aux__Mathlib__Data__Matrix__Mul______macroRules__term___u2b1d_u1d65____1___closed__10_value;
static const lean_ctor_object lp_mathlib___aux__Mathlib__Data__Matrix__Mul______macroRules__term___u2b1d_u1d65____1___closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___aux__Mathlib__Data__Matrix__Mul______macroRules__term___u2b1d_u1d65____1___closed__10_value),LEAN_SCALAR_PTR_LITERAL(24, 58, 49, 223, 146, 207, 197, 136)}};
static const lean_object* lp_mathlib___aux__Mathlib__Data__Matrix__Mul______macroRules__term___u2b1d_u1d65____1___closed__11 = (const lean_object*)&lp_mathlib___aux__Mathlib__Data__Matrix__Mul______macroRules__term___u2b1d_u1d65____1___closed__11_value;
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Data__Matrix__Mul______macroRules__term___u2b1d_u1d65____1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Data__Matrix__Mul______macroRules__term___u2b1d_u1d65____1___boxed(lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib___aux__Mathlib__Data__Matrix__Mul______unexpand__dotProduct__1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "ident"};
static const lean_object* lp_mathlib___aux__Mathlib__Data__Matrix__Mul______unexpand__dotProduct__1___closed__0 = (const lean_object*)&lp_mathlib___aux__Mathlib__Data__Matrix__Mul______unexpand__dotProduct__1___closed__0_value;
static const lean_ctor_object lp_mathlib___aux__Mathlib__Data__Matrix__Mul______unexpand__dotProduct__1___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___aux__Mathlib__Data__Matrix__Mul______unexpand__dotProduct__1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(52, 159, 208, 51, 14, 60, 6, 71)}};
static const lean_object* lp_mathlib___aux__Mathlib__Data__Matrix__Mul______unexpand__dotProduct__1___closed__1 = (const lean_object*)&lp_mathlib___aux__Mathlib__Data__Matrix__Mul______unexpand__dotProduct__1___closed__1_value;
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Data__Matrix__Mul______unexpand__dotProduct__1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Data__Matrix__Mul______unexpand__dotProduct__1___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Matrix_instHMulOfFintypeOfMulOfAddCommMonoid___redArg___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Matrix_instHMulOfFintypeOfMulOfAddCommMonoid___redArg___lam__1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Matrix_instHMulOfFintypeOfMulOfAddCommMonoid___redArg___lam__2(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Matrix_instHMulOfFintypeOfMulOfAddCommMonoid___redArg___lam__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Matrix_instHMulOfFintypeOfMulOfAddCommMonoid___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Matrix_instHMulOfFintypeOfMulOfAddCommMonoid(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Matrix_instMulOfFintypeOfAddCommMonoid___redArg___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Matrix_instMulOfFintypeOfAddCommMonoid___redArg___lam__1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Matrix_instMulOfFintypeOfAddCommMonoid___redArg___lam__2(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Matrix_instMulOfFintypeOfAddCommMonoid___redArg___lam__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Matrix_instMulOfFintypeOfAddCommMonoid___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Matrix_instMulOfFintypeOfAddCommMonoid(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Matrix_instMulOneOfFintypeOfDecidableEqOfAddCommMonoid___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Matrix_instMulOneOfFintypeOfDecidableEqOfAddCommMonoid(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Matrix_nonUnitalNonAssocSemiring___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Matrix_nonUnitalNonAssocSemiring(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Matrix_addMonoidHomMulLeft___redArg___lam__1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Matrix_addMonoidHomMulLeft___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Matrix_addMonoidHomMulLeft___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Matrix_addMonoidHomMulLeft___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Matrix_addMonoidHomMulLeft(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Matrix_addMonoidHomMulRight___redArg___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Matrix_addMonoidHomMulRight___redArg___lam__1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Matrix_addMonoidHomMulRight___redArg___lam__2(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Matrix_addMonoidHomMulRight___redArg___lam__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Matrix_addMonoidHomMulRight___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Matrix_addMonoidHomMulRight(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Matrix_nonAssocSemiring___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Matrix_nonAssocSemiring(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Matrix_nonUnitalSemiring___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Matrix_nonUnitalSemiring(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Matrix_semiring___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Matrix_semiring(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Matrix_nonUnitalNonAssocRing___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Matrix_nonUnitalNonAssocRing(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Matrix_instNonUnitalRing___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Matrix_instNonUnitalRing(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Matrix_instNonAssocRing___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Matrix_instNonAssocRing(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Matrix_instRing___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Matrix_instRing(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Matrix_vecMulVec___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_once_cell_t lp_mathlib_Matrix_vecMulVec___redArg___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Matrix_vecMulVec___redArg___closed__0;
LEAN_EXPORT lean_object* lp_mathlib_Matrix_vecMulVec___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Matrix_vecMulVec(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Matrix_mulVec___redArg___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Matrix_mulVec___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Matrix_mulVec(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Matrix_term___x2a_u1d65___00__closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "Matrix"};
static const lean_object* lp_mathlib_Matrix_term___x2a_u1d65___00__closed__0 = (const lean_object*)&lp_mathlib_Matrix_term___x2a_u1d65___00__closed__0_value;
static const lean_string_object lp_mathlib_Matrix_term___x2a_u1d65___00__closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 11, .m_capacity = 11, .m_length = 8, .m_data = "term_*ᵥ_"};
static const lean_object* lp_mathlib_Matrix_term___x2a_u1d65___00__closed__1 = (const lean_object*)&lp_mathlib_Matrix_term___x2a_u1d65___00__closed__1_value;
static const lean_ctor_object lp_mathlib_Matrix_term___x2a_u1d65___00__closed__2_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Matrix_term___x2a_u1d65___00__closed__0_value),LEAN_SCALAR_PTR_LITERAL(214, 58, 148, 51, 223, 251, 16, 41)}};
static const lean_ctor_object lp_mathlib_Matrix_term___x2a_u1d65___00__closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Matrix_term___x2a_u1d65___00__closed__2_value_aux_0),((lean_object*)&lp_mathlib_Matrix_term___x2a_u1d65___00__closed__1_value),LEAN_SCALAR_PTR_LITERAL(62, 72, 185, 168, 148, 158, 127, 24)}};
static const lean_object* lp_mathlib_Matrix_term___x2a_u1d65___00__closed__2 = (const lean_object*)&lp_mathlib_Matrix_term___x2a_u1d65___00__closed__2_value;
static const lean_string_object lp_mathlib_Matrix_term___x2a_u1d65___00__closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 4, .m_data = " *ᵥ "};
static const lean_object* lp_mathlib_Matrix_term___x2a_u1d65___00__closed__3 = (const lean_object*)&lp_mathlib_Matrix_term___x2a_u1d65___00__closed__3_value;
static const lean_ctor_object lp_mathlib_Matrix_term___x2a_u1d65___00__closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 5}, .m_objs = {((lean_object*)&lp_mathlib_Matrix_term___x2a_u1d65___00__closed__3_value)}};
static const lean_object* lp_mathlib_Matrix_term___x2a_u1d65___00__closed__4 = (const lean_object*)&lp_mathlib_Matrix_term___x2a_u1d65___00__closed__4_value;
static const lean_ctor_object lp_mathlib_Matrix_term___x2a_u1d65___00__closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib_term___u2b1d_u1d65___00__closed__3_value),((lean_object*)&lp_mathlib_Matrix_term___x2a_u1d65___00__closed__4_value),((lean_object*)&lp_mathlib_term___u2b1d_u1d65___00__closed__8_value)}};
static const lean_object* lp_mathlib_Matrix_term___x2a_u1d65___00__closed__5 = (const lean_object*)&lp_mathlib_Matrix_term___x2a_u1d65___00__closed__5_value;
static const lean_ctor_object lp_mathlib_Matrix_term___x2a_u1d65___00__closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*4 + 0, .m_other = 4, .m_tag = 4}, .m_objs = {((lean_object*)&lp_mathlib_Matrix_term___x2a_u1d65___00__closed__2_value),((lean_object*)(((size_t)(73) << 1) | 1)),((lean_object*)(((size_t)(74) << 1) | 1)),((lean_object*)&lp_mathlib_Matrix_term___x2a_u1d65___00__closed__5_value)}};
static const lean_object* lp_mathlib_Matrix_term___x2a_u1d65___00__closed__6 = (const lean_object*)&lp_mathlib_Matrix_term___x2a_u1d65___00__closed__6_value;
LEAN_EXPORT const lean_object* lp_mathlib_Matrix_term___x2a_u1d65__ = (const lean_object*)&lp_mathlib_Matrix_term___x2a_u1d65___00__closed__6_value;
static const lean_string_object lp_mathlib_Matrix___aux__Mathlib__Data__Matrix__Mul______macroRules__Matrix__term___x2a_u1d65____1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 14, .m_capacity = 14, .m_length = 13, .m_data = "Matrix.mulVec"};
static const lean_object* lp_mathlib_Matrix___aux__Mathlib__Data__Matrix__Mul______macroRules__Matrix__term___x2a_u1d65____1___closed__0 = (const lean_object*)&lp_mathlib_Matrix___aux__Mathlib__Data__Matrix__Mul______macroRules__Matrix__term___x2a_u1d65____1___closed__0_value;
static lean_once_cell_t lp_mathlib_Matrix___aux__Mathlib__Data__Matrix__Mul______macroRules__Matrix__term___x2a_u1d65____1___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Matrix___aux__Mathlib__Data__Matrix__Mul______macroRules__Matrix__term___x2a_u1d65____1___closed__1;
static const lean_string_object lp_mathlib_Matrix___aux__Mathlib__Data__Matrix__Mul______macroRules__Matrix__term___x2a_u1d65____1___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "mulVec"};
static const lean_object* lp_mathlib_Matrix___aux__Mathlib__Data__Matrix__Mul______macroRules__Matrix__term___x2a_u1d65____1___closed__2 = (const lean_object*)&lp_mathlib_Matrix___aux__Mathlib__Data__Matrix__Mul______macroRules__Matrix__term___x2a_u1d65____1___closed__2_value;
static const lean_ctor_object lp_mathlib_Matrix___aux__Mathlib__Data__Matrix__Mul______macroRules__Matrix__term___x2a_u1d65____1___closed__3_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Matrix_term___x2a_u1d65___00__closed__0_value),LEAN_SCALAR_PTR_LITERAL(214, 58, 148, 51, 223, 251, 16, 41)}};
static const lean_ctor_object lp_mathlib_Matrix___aux__Mathlib__Data__Matrix__Mul______macroRules__Matrix__term___x2a_u1d65____1___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Matrix___aux__Mathlib__Data__Matrix__Mul______macroRules__Matrix__term___x2a_u1d65____1___closed__3_value_aux_0),((lean_object*)&lp_mathlib_Matrix___aux__Mathlib__Data__Matrix__Mul______macroRules__Matrix__term___x2a_u1d65____1___closed__2_value),LEAN_SCALAR_PTR_LITERAL(61, 53, 205, 186, 190, 255, 44, 222)}};
static const lean_object* lp_mathlib_Matrix___aux__Mathlib__Data__Matrix__Mul______macroRules__Matrix__term___x2a_u1d65____1___closed__3 = (const lean_object*)&lp_mathlib_Matrix___aux__Mathlib__Data__Matrix__Mul______macroRules__Matrix__term___x2a_u1d65____1___closed__3_value;
static const lean_ctor_object lp_mathlib_Matrix___aux__Mathlib__Data__Matrix__Mul______macroRules__Matrix__term___x2a_u1d65____1___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Matrix___aux__Mathlib__Data__Matrix__Mul______macroRules__Matrix__term___x2a_u1d65____1___closed__3_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_Matrix___aux__Mathlib__Data__Matrix__Mul______macroRules__Matrix__term___x2a_u1d65____1___closed__4 = (const lean_object*)&lp_mathlib_Matrix___aux__Mathlib__Data__Matrix__Mul______macroRules__Matrix__term___x2a_u1d65____1___closed__4_value;
static const lean_ctor_object lp_mathlib_Matrix___aux__Mathlib__Data__Matrix__Mul______macroRules__Matrix__term___x2a_u1d65____1___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Matrix___aux__Mathlib__Data__Matrix__Mul______macroRules__Matrix__term___x2a_u1d65____1___closed__4_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_Matrix___aux__Mathlib__Data__Matrix__Mul______macroRules__Matrix__term___x2a_u1d65____1___closed__5 = (const lean_object*)&lp_mathlib_Matrix___aux__Mathlib__Data__Matrix__Mul______macroRules__Matrix__term___x2a_u1d65____1___closed__5_value;
LEAN_EXPORT lean_object* lp_mathlib_Matrix___aux__Mathlib__Data__Matrix__Mul______macroRules__Matrix__term___x2a_u1d65____1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Matrix___aux__Mathlib__Data__Matrix__Mul______macroRules__Matrix__term___x2a_u1d65____1___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Matrix___aux__Mathlib__Data__Matrix__Mul______unexpand__Matrix__mulVec__1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Matrix___aux__Mathlib__Data__Matrix__Mul______unexpand__Matrix__mulVec__1___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Matrix_vecMul___redArg___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Matrix_vecMul___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Matrix_vecMul(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Matrix_term___u1d65_x2a___00__closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 11, .m_capacity = 11, .m_length = 8, .m_data = "term_ᵥ*_"};
static const lean_object* lp_mathlib_Matrix_term___u1d65_x2a___00__closed__0 = (const lean_object*)&lp_mathlib_Matrix_term___u1d65_x2a___00__closed__0_value;
static const lean_ctor_object lp_mathlib_Matrix_term___u1d65_x2a___00__closed__1_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Matrix_term___x2a_u1d65___00__closed__0_value),LEAN_SCALAR_PTR_LITERAL(214, 58, 148, 51, 223, 251, 16, 41)}};
static const lean_ctor_object lp_mathlib_Matrix_term___u1d65_x2a___00__closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Matrix_term___u1d65_x2a___00__closed__1_value_aux_0),((lean_object*)&lp_mathlib_Matrix_term___u1d65_x2a___00__closed__0_value),LEAN_SCALAR_PTR_LITERAL(237, 169, 169, 118, 52, 225, 228, 68)}};
static const lean_object* lp_mathlib_Matrix_term___u1d65_x2a___00__closed__1 = (const lean_object*)&lp_mathlib_Matrix_term___u1d65_x2a___00__closed__1_value;
static const lean_string_object lp_mathlib_Matrix_term___u1d65_x2a___00__closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 4, .m_data = " ᵥ* "};
static const lean_object* lp_mathlib_Matrix_term___u1d65_x2a___00__closed__2 = (const lean_object*)&lp_mathlib_Matrix_term___u1d65_x2a___00__closed__2_value;
static const lean_ctor_object lp_mathlib_Matrix_term___u1d65_x2a___00__closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 5}, .m_objs = {((lean_object*)&lp_mathlib_Matrix_term___u1d65_x2a___00__closed__2_value)}};
static const lean_object* lp_mathlib_Matrix_term___u1d65_x2a___00__closed__3 = (const lean_object*)&lp_mathlib_Matrix_term___u1d65_x2a___00__closed__3_value;
static const lean_ctor_object lp_mathlib_Matrix_term___u1d65_x2a___00__closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 7}, .m_objs = {((lean_object*)&lp_mathlib_term___u2b1d_u1d65___00__closed__7_value),((lean_object*)(((size_t)(74) << 1) | 1))}};
static const lean_object* lp_mathlib_Matrix_term___u1d65_x2a___00__closed__4 = (const lean_object*)&lp_mathlib_Matrix_term___u1d65_x2a___00__closed__4_value;
static const lean_ctor_object lp_mathlib_Matrix_term___u1d65_x2a___00__closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib_term___u2b1d_u1d65___00__closed__3_value),((lean_object*)&lp_mathlib_Matrix_term___u1d65_x2a___00__closed__3_value),((lean_object*)&lp_mathlib_Matrix_term___u1d65_x2a___00__closed__4_value)}};
static const lean_object* lp_mathlib_Matrix_term___u1d65_x2a___00__closed__5 = (const lean_object*)&lp_mathlib_Matrix_term___u1d65_x2a___00__closed__5_value;
static const lean_ctor_object lp_mathlib_Matrix_term___u1d65_x2a___00__closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*4 + 0, .m_other = 4, .m_tag = 4}, .m_objs = {((lean_object*)&lp_mathlib_Matrix_term___u1d65_x2a___00__closed__1_value),((lean_object*)(((size_t)(73) << 1) | 1)),((lean_object*)(((size_t)(73) << 1) | 1)),((lean_object*)&lp_mathlib_Matrix_term___u1d65_x2a___00__closed__5_value)}};
static const lean_object* lp_mathlib_Matrix_term___u1d65_x2a___00__closed__6 = (const lean_object*)&lp_mathlib_Matrix_term___u1d65_x2a___00__closed__6_value;
LEAN_EXPORT const lean_object* lp_mathlib_Matrix_term___u1d65_x2a__ = (const lean_object*)&lp_mathlib_Matrix_term___u1d65_x2a___00__closed__6_value;
static const lean_string_object lp_mathlib_Matrix___aux__Mathlib__Data__Matrix__Mul______macroRules__Matrix__term___u1d65_x2a____1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 14, .m_capacity = 14, .m_length = 13, .m_data = "Matrix.vecMul"};
static const lean_object* lp_mathlib_Matrix___aux__Mathlib__Data__Matrix__Mul______macroRules__Matrix__term___u1d65_x2a____1___closed__0 = (const lean_object*)&lp_mathlib_Matrix___aux__Mathlib__Data__Matrix__Mul______macroRules__Matrix__term___u1d65_x2a____1___closed__0_value;
static lean_once_cell_t lp_mathlib_Matrix___aux__Mathlib__Data__Matrix__Mul______macroRules__Matrix__term___u1d65_x2a____1___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Matrix___aux__Mathlib__Data__Matrix__Mul______macroRules__Matrix__term___u1d65_x2a____1___closed__1;
static const lean_string_object lp_mathlib_Matrix___aux__Mathlib__Data__Matrix__Mul______macroRules__Matrix__term___u1d65_x2a____1___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "vecMul"};
static const lean_object* lp_mathlib_Matrix___aux__Mathlib__Data__Matrix__Mul______macroRules__Matrix__term___u1d65_x2a____1___closed__2 = (const lean_object*)&lp_mathlib_Matrix___aux__Mathlib__Data__Matrix__Mul______macroRules__Matrix__term___u1d65_x2a____1___closed__2_value;
static const lean_ctor_object lp_mathlib_Matrix___aux__Mathlib__Data__Matrix__Mul______macroRules__Matrix__term___u1d65_x2a____1___closed__3_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Matrix_term___x2a_u1d65___00__closed__0_value),LEAN_SCALAR_PTR_LITERAL(214, 58, 148, 51, 223, 251, 16, 41)}};
static const lean_ctor_object lp_mathlib_Matrix___aux__Mathlib__Data__Matrix__Mul______macroRules__Matrix__term___u1d65_x2a____1___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Matrix___aux__Mathlib__Data__Matrix__Mul______macroRules__Matrix__term___u1d65_x2a____1___closed__3_value_aux_0),((lean_object*)&lp_mathlib_Matrix___aux__Mathlib__Data__Matrix__Mul______macroRules__Matrix__term___u1d65_x2a____1___closed__2_value),LEAN_SCALAR_PTR_LITERAL(50, 172, 253, 134, 252, 140, 63, 56)}};
static const lean_object* lp_mathlib_Matrix___aux__Mathlib__Data__Matrix__Mul______macroRules__Matrix__term___u1d65_x2a____1___closed__3 = (const lean_object*)&lp_mathlib_Matrix___aux__Mathlib__Data__Matrix__Mul______macroRules__Matrix__term___u1d65_x2a____1___closed__3_value;
static const lean_ctor_object lp_mathlib_Matrix___aux__Mathlib__Data__Matrix__Mul______macroRules__Matrix__term___u1d65_x2a____1___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Matrix___aux__Mathlib__Data__Matrix__Mul______macroRules__Matrix__term___u1d65_x2a____1___closed__3_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_Matrix___aux__Mathlib__Data__Matrix__Mul______macroRules__Matrix__term___u1d65_x2a____1___closed__4 = (const lean_object*)&lp_mathlib_Matrix___aux__Mathlib__Data__Matrix__Mul______macroRules__Matrix__term___u1d65_x2a____1___closed__4_value;
static const lean_ctor_object lp_mathlib_Matrix___aux__Mathlib__Data__Matrix__Mul______macroRules__Matrix__term___u1d65_x2a____1___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Matrix___aux__Mathlib__Data__Matrix__Mul______macroRules__Matrix__term___u1d65_x2a____1___closed__4_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_Matrix___aux__Mathlib__Data__Matrix__Mul______macroRules__Matrix__term___u1d65_x2a____1___closed__5 = (const lean_object*)&lp_mathlib_Matrix___aux__Mathlib__Data__Matrix__Mul______macroRules__Matrix__term___u1d65_x2a____1___closed__5_value;
LEAN_EXPORT lean_object* lp_mathlib_Matrix___aux__Mathlib__Data__Matrix__Mul______macroRules__Matrix__term___u1d65_x2a____1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Matrix___aux__Mathlib__Data__Matrix__Mul______macroRules__Matrix__term___u1d65_x2a____1___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Matrix___aux__Mathlib__Data__Matrix__Mul______unexpand__Matrix__vecMul__1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Matrix___aux__Mathlib__Data__Matrix__Mul______unexpand__Matrix__vecMul__1___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Matrix_mulVec_addMonoidHomLeft___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Matrix_mulVec_addMonoidHomLeft___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Matrix_mulVec_addMonoidHomLeft(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_dotProduct___redArg___lam__0(lean_object* v_v_1_, lean_object* v_w_2_, lean_object* v_inst_3_, lean_object* v_i_4_){
_start:
{
lean_object* v___x_5_; lean_object* v___x_6_; lean_object* v___x_7_; 
lean_inc(v_i_4_);
v___x_5_ = lean_apply_1(v_v_1_, v_i_4_);
v___x_6_ = lean_apply_1(v_w_2_, v_i_4_);
v___x_7_ = lean_apply_2(v_inst_3_, v___x_5_, v___x_6_);
return v___x_7_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_dotProduct___redArg(lean_object* v_inst_8_, lean_object* v_inst_9_, lean_object* v_inst_10_, lean_object* v_v_11_, lean_object* v_w_12_){
_start:
{
lean_object* v___f_13_; lean_object* v___x_14_; 
v___f_13_ = lean_alloc_closure((void*)(lp_mathlib_dotProduct___redArg___lam__0), 4, 3);
lean_closure_set(v___f_13_, 0, v_v_11_);
lean_closure_set(v___f_13_, 1, v_w_12_);
lean_closure_set(v___f_13_, 2, v_inst_9_);
v___x_14_ = lp_mathlib_Finset_sum___redArg(v_inst_10_, v_inst_8_, v___f_13_);
return v___x_14_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_dotProduct___redArg___boxed(lean_object* v_inst_15_, lean_object* v_inst_16_, lean_object* v_inst_17_, lean_object* v_v_18_, lean_object* v_w_19_){
_start:
{
lean_object* v_res_20_; 
v_res_20_ = lp_mathlib_dotProduct___redArg(v_inst_15_, v_inst_16_, v_inst_17_, v_v_18_, v_w_19_);
lean_dec_ref(v_inst_17_);
return v_res_20_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_dotProduct(lean_object* v_m_21_, lean_object* v_00_u03b1_22_, lean_object* v_inst_23_, lean_object* v_inst_24_, lean_object* v_inst_25_, lean_object* v_v_26_, lean_object* v_w_27_){
_start:
{
lean_object* v___x_28_; 
v___x_28_ = lp_mathlib_dotProduct___redArg(v_inst_23_, v_inst_24_, v_inst_25_, v_v_26_, v_w_27_);
return v___x_28_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_dotProduct___boxed(lean_object* v_m_29_, lean_object* v_00_u03b1_30_, lean_object* v_inst_31_, lean_object* v_inst_32_, lean_object* v_inst_33_, lean_object* v_v_34_, lean_object* v_w_35_){
_start:
{
lean_object* v_res_36_; 
v_res_36_ = lp_mathlib_dotProduct(v_m_29_, v_00_u03b1_30_, v_inst_31_, v_inst_32_, v_inst_33_, v_v_34_, v_w_35_);
lean_dec_ref(v_inst_33_);
return v_res_36_;
}
}
static lean_object* _init_lp_mathlib___aux__Mathlib__Data__Matrix__Mul______macroRules__term___u2b1d_u1d65____1___closed__6(void){
_start:
{
lean_object* v___x_71_; lean_object* v___x_72_; 
v___x_71_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Data__Matrix__Mul______macroRules__term___u2b1d_u1d65____1___closed__5));
v___x_72_ = l_String_toRawSubstring_x27(v___x_71_);
return v___x_72_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Data__Matrix__Mul______macroRules__term___u2b1d_u1d65____1(lean_object* v_x_84_, lean_object* v_a_85_, lean_object* v_a_86_){
_start:
{
lean_object* v___x_87_; uint8_t v___x_88_; 
v___x_87_ = ((lean_object*)(lp_mathlib_term___u2b1d_u1d65___00__closed__1));
lean_inc(v_x_84_);
v___x_88_ = l_Lean_Syntax_isOfKind(v_x_84_, v___x_87_);
if (v___x_88_ == 0)
{
lean_object* v___x_89_; lean_object* v___x_90_; 
lean_dec(v_x_84_);
v___x_89_ = lean_box(1);
v___x_90_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_90_, 0, v___x_89_);
lean_ctor_set(v___x_90_, 1, v_a_86_);
return v___x_90_;
}
else
{
lean_object* v_quotContext_91_; lean_object* v_currMacroScope_92_; lean_object* v_ref_93_; lean_object* v___x_94_; lean_object* v___x_95_; lean_object* v___x_96_; lean_object* v___x_97_; uint8_t v___x_98_; lean_object* v___x_99_; lean_object* v___x_100_; lean_object* v___x_101_; lean_object* v___x_102_; lean_object* v___x_103_; lean_object* v___x_104_; lean_object* v___x_105_; lean_object* v___x_106_; lean_object* v___x_107_; lean_object* v___x_108_; lean_object* v___x_109_; 
v_quotContext_91_ = lean_ctor_get(v_a_85_, 1);
v_currMacroScope_92_ = lean_ctor_get(v_a_85_, 2);
v_ref_93_ = lean_ctor_get(v_a_85_, 5);
v___x_94_ = lean_unsigned_to_nat(0u);
v___x_95_ = l_Lean_Syntax_getArg(v_x_84_, v___x_94_);
v___x_96_ = lean_unsigned_to_nat(2u);
v___x_97_ = l_Lean_Syntax_getArg(v_x_84_, v___x_96_);
lean_dec(v_x_84_);
v___x_98_ = 0;
v___x_99_ = l_Lean_SourceInfo_fromRef(v_ref_93_, v___x_98_);
v___x_100_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Data__Matrix__Mul______macroRules__term___u2b1d_u1d65____1___closed__4));
v___x_101_ = lean_obj_once(&lp_mathlib___aux__Mathlib__Data__Matrix__Mul______macroRules__term___u2b1d_u1d65____1___closed__6, &lp_mathlib___aux__Mathlib__Data__Matrix__Mul______macroRules__term___u2b1d_u1d65____1___closed__6_once, _init_lp_mathlib___aux__Mathlib__Data__Matrix__Mul______macroRules__term___u2b1d_u1d65____1___closed__6);
v___x_102_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Data__Matrix__Mul______macroRules__term___u2b1d_u1d65____1___closed__7));
lean_inc(v_currMacroScope_92_);
lean_inc(v_quotContext_91_);
v___x_103_ = l_Lean_addMacroScope(v_quotContext_91_, v___x_102_, v_currMacroScope_92_);
v___x_104_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Data__Matrix__Mul______macroRules__term___u2b1d_u1d65____1___closed__9));
lean_inc_n(v___x_99_, 2);
v___x_105_ = lean_alloc_ctor(3, 4, 0);
lean_ctor_set(v___x_105_, 0, v___x_99_);
lean_ctor_set(v___x_105_, 1, v___x_101_);
lean_ctor_set(v___x_105_, 2, v___x_103_);
lean_ctor_set(v___x_105_, 3, v___x_104_);
v___x_106_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Data__Matrix__Mul______macroRules__term___u2b1d_u1d65____1___closed__11));
v___x_107_ = l_Lean_Syntax_node2(v___x_99_, v___x_106_, v___x_95_, v___x_97_);
v___x_108_ = l_Lean_Syntax_node2(v___x_99_, v___x_100_, v___x_105_, v___x_107_);
v___x_109_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_109_, 0, v___x_108_);
lean_ctor_set(v___x_109_, 1, v_a_86_);
return v___x_109_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Data__Matrix__Mul______macroRules__term___u2b1d_u1d65____1___boxed(lean_object* v_x_110_, lean_object* v_a_111_, lean_object* v_a_112_){
_start:
{
lean_object* v_res_113_; 
v_res_113_ = lp_mathlib___aux__Mathlib__Data__Matrix__Mul______macroRules__term___u2b1d_u1d65____1(v_x_110_, v_a_111_, v_a_112_);
lean_dec_ref(v_a_111_);
return v_res_113_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Data__Matrix__Mul______unexpand__dotProduct__1(lean_object* v_x_117_, lean_object* v_a_118_, lean_object* v_a_119_){
_start:
{
lean_object* v___x_120_; uint8_t v___x_121_; 
v___x_120_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Data__Matrix__Mul______macroRules__term___u2b1d_u1d65____1___closed__4));
lean_inc(v_x_117_);
v___x_121_ = l_Lean_Syntax_isOfKind(v_x_117_, v___x_120_);
if (v___x_121_ == 0)
{
lean_object* v___x_122_; lean_object* v___x_123_; 
lean_dec(v_x_117_);
v___x_122_ = lean_box(0);
v___x_123_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_123_, 0, v___x_122_);
lean_ctor_set(v___x_123_, 1, v_a_119_);
return v___x_123_;
}
else
{
lean_object* v___x_124_; lean_object* v___x_125_; lean_object* v___x_126_; uint8_t v___x_127_; 
v___x_124_ = lean_unsigned_to_nat(0u);
v___x_125_ = l_Lean_Syntax_getArg(v_x_117_, v___x_124_);
v___x_126_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Data__Matrix__Mul______unexpand__dotProduct__1___closed__1));
lean_inc(v___x_125_);
v___x_127_ = l_Lean_Syntax_isOfKind(v___x_125_, v___x_126_);
if (v___x_127_ == 0)
{
lean_object* v___x_128_; lean_object* v___x_129_; 
lean_dec(v___x_125_);
lean_dec(v_x_117_);
v___x_128_ = lean_box(0);
v___x_129_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_129_, 0, v___x_128_);
lean_ctor_set(v___x_129_, 1, v_a_119_);
return v___x_129_;
}
else
{
lean_object* v___x_130_; lean_object* v___x_131_; lean_object* v___x_132_; uint8_t v___x_133_; 
v___x_130_ = lean_unsigned_to_nat(1u);
v___x_131_ = l_Lean_Syntax_getArg(v_x_117_, v___x_130_);
lean_dec(v_x_117_);
v___x_132_ = lean_unsigned_to_nat(2u);
lean_inc(v___x_131_);
v___x_133_ = l_Lean_Syntax_matchesNull(v___x_131_, v___x_132_);
if (v___x_133_ == 0)
{
lean_object* v___x_134_; lean_object* v___x_135_; 
lean_dec(v___x_131_);
lean_dec(v___x_125_);
v___x_134_ = lean_box(0);
v___x_135_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_135_, 0, v___x_134_);
lean_ctor_set(v___x_135_, 1, v_a_119_);
return v___x_135_;
}
else
{
lean_object* v___x_136_; lean_object* v___x_137_; lean_object* v_ref_138_; uint8_t v___x_139_; lean_object* v___x_140_; lean_object* v___x_141_; lean_object* v___x_142_; lean_object* v___x_143_; lean_object* v___x_144_; lean_object* v___x_145_; 
v___x_136_ = l_Lean_Syntax_getArg(v___x_131_, v___x_124_);
v___x_137_ = l_Lean_Syntax_getArg(v___x_131_, v___x_130_);
lean_dec(v___x_131_);
v_ref_138_ = l_Lean_replaceRef(v___x_125_, v_a_118_);
lean_dec(v___x_125_);
v___x_139_ = 0;
v___x_140_ = l_Lean_SourceInfo_fromRef(v_ref_138_, v___x_139_);
lean_dec(v_ref_138_);
v___x_141_ = ((lean_object*)(lp_mathlib_term___u2b1d_u1d65___00__closed__1));
v___x_142_ = ((lean_object*)(lp_mathlib_term___u2b1d_u1d65___00__closed__4));
lean_inc(v___x_140_);
v___x_143_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_143_, 0, v___x_140_);
lean_ctor_set(v___x_143_, 1, v___x_142_);
v___x_144_ = l_Lean_Syntax_node3(v___x_140_, v___x_141_, v___x_136_, v___x_143_, v___x_137_);
v___x_145_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_145_, 0, v___x_144_);
lean_ctor_set(v___x_145_, 1, v_a_119_);
return v___x_145_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Data__Matrix__Mul______unexpand__dotProduct__1___boxed(lean_object* v_x_146_, lean_object* v_a_147_, lean_object* v_a_148_){
_start:
{
lean_object* v_res_149_; 
v_res_149_ = lp_mathlib___aux__Mathlib__Data__Matrix__Mul______unexpand__dotProduct__1(v_x_146_, v_a_147_, v_a_148_);
lean_dec(v_a_147_);
return v_res_149_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Matrix_instHMulOfFintypeOfMulOfAddCommMonoid___redArg___lam__0(lean_object* v_N_150_, lean_object* v_k_151_, lean_object* v_j_152_){
_start:
{
lean_object* v___x_153_; 
v___x_153_ = lean_apply_2(v_N_150_, v_j_152_, v_k_151_);
return v___x_153_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Matrix_instHMulOfFintypeOfMulOfAddCommMonoid___redArg___lam__1(lean_object* v_M_154_, lean_object* v_i_155_, lean_object* v_j_156_){
_start:
{
lean_object* v___x_157_; 
v___x_157_ = lean_apply_2(v_M_154_, v_i_155_, v_j_156_);
return v___x_157_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Matrix_instHMulOfFintypeOfMulOfAddCommMonoid___redArg___lam__2(lean_object* v_inst_158_, lean_object* v_inst_159_, lean_object* v_inst_160_, lean_object* v_M_161_, lean_object* v_N_162_, lean_object* v_i_163_, lean_object* v_k_164_){
_start:
{
lean_object* v___f_165_; lean_object* v___f_166_; lean_object* v___x_167_; 
v___f_165_ = lean_alloc_closure((void*)(lp_mathlib_Matrix_instHMulOfFintypeOfMulOfAddCommMonoid___redArg___lam__0), 3, 2);
lean_closure_set(v___f_165_, 0, v_N_162_);
lean_closure_set(v___f_165_, 1, v_k_164_);
v___f_166_ = lean_alloc_closure((void*)(lp_mathlib_Matrix_instHMulOfFintypeOfMulOfAddCommMonoid___redArg___lam__1), 3, 2);
lean_closure_set(v___f_166_, 0, v_M_161_);
lean_closure_set(v___f_166_, 1, v_i_163_);
v___x_167_ = lp_mathlib_dotProduct___redArg(v_inst_158_, v_inst_159_, v_inst_160_, v___f_166_, v___f_165_);
return v___x_167_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Matrix_instHMulOfFintypeOfMulOfAddCommMonoid___redArg___lam__2___boxed(lean_object* v_inst_168_, lean_object* v_inst_169_, lean_object* v_inst_170_, lean_object* v_M_171_, lean_object* v_N_172_, lean_object* v_i_173_, lean_object* v_k_174_){
_start:
{
lean_object* v_res_175_; 
v_res_175_ = lp_mathlib_Matrix_instHMulOfFintypeOfMulOfAddCommMonoid___redArg___lam__2(v_inst_168_, v_inst_169_, v_inst_170_, v_M_171_, v_N_172_, v_i_173_, v_k_174_);
lean_dec_ref(v_inst_170_);
return v_res_175_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Matrix_instHMulOfFintypeOfMulOfAddCommMonoid___redArg(lean_object* v_inst_176_, lean_object* v_inst_177_, lean_object* v_inst_178_){
_start:
{
lean_object* v___f_179_; 
v___f_179_ = lean_alloc_closure((void*)(lp_mathlib_Matrix_instHMulOfFintypeOfMulOfAddCommMonoid___redArg___lam__2___boxed), 7, 3);
lean_closure_set(v___f_179_, 0, v_inst_176_);
lean_closure_set(v___f_179_, 1, v_inst_177_);
lean_closure_set(v___f_179_, 2, v_inst_178_);
return v___f_179_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Matrix_instHMulOfFintypeOfMulOfAddCommMonoid(lean_object* v_l_180_, lean_object* v_m_181_, lean_object* v_n_182_, lean_object* v_00_u03b1_183_, lean_object* v_inst_184_, lean_object* v_inst_185_, lean_object* v_inst_186_){
_start:
{
lean_object* v___f_187_; 
v___f_187_ = lean_alloc_closure((void*)(lp_mathlib_Matrix_instHMulOfFintypeOfMulOfAddCommMonoid___redArg___lam__2___boxed), 7, 3);
lean_closure_set(v___f_187_, 0, v_inst_184_);
lean_closure_set(v___f_187_, 1, v_inst_185_);
lean_closure_set(v___f_187_, 2, v_inst_186_);
return v___f_187_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Matrix_instMulOfFintypeOfAddCommMonoid___redArg___lam__0(lean_object* v_N_188_, lean_object* v___y_189_, lean_object* v_j_190_){
_start:
{
lean_object* v___x_191_; 
v___x_191_ = lean_apply_2(v_N_188_, v_j_190_, v___y_189_);
return v___x_191_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Matrix_instMulOfFintypeOfAddCommMonoid___redArg___lam__1(lean_object* v_M_192_, lean_object* v___y_193_, lean_object* v_j_194_){
_start:
{
lean_object* v___x_195_; 
v___x_195_ = lean_apply_2(v_M_192_, v___y_193_, v_j_194_);
return v___x_195_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Matrix_instMulOfFintypeOfAddCommMonoid___redArg___lam__2(lean_object* v_inst_196_, lean_object* v_inst_197_, lean_object* v_inst_198_, lean_object* v_M_199_, lean_object* v_N_200_, lean_object* v___y_201_, lean_object* v___y_202_){
_start:
{
lean_object* v___f_203_; lean_object* v___f_204_; lean_object* v___x_205_; 
v___f_203_ = lean_alloc_closure((void*)(lp_mathlib_Matrix_instMulOfFintypeOfAddCommMonoid___redArg___lam__0), 3, 2);
lean_closure_set(v___f_203_, 0, v_N_200_);
lean_closure_set(v___f_203_, 1, v___y_202_);
v___f_204_ = lean_alloc_closure((void*)(lp_mathlib_Matrix_instMulOfFintypeOfAddCommMonoid___redArg___lam__1), 3, 2);
lean_closure_set(v___f_204_, 0, v_M_199_);
lean_closure_set(v___f_204_, 1, v___y_201_);
v___x_205_ = lp_mathlib_dotProduct___redArg(v_inst_196_, v_inst_197_, v_inst_198_, v___f_204_, v___f_203_);
return v___x_205_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Matrix_instMulOfFintypeOfAddCommMonoid___redArg___lam__2___boxed(lean_object* v_inst_206_, lean_object* v_inst_207_, lean_object* v_inst_208_, lean_object* v_M_209_, lean_object* v_N_210_, lean_object* v___y_211_, lean_object* v___y_212_){
_start:
{
lean_object* v_res_213_; 
v_res_213_ = lp_mathlib_Matrix_instMulOfFintypeOfAddCommMonoid___redArg___lam__2(v_inst_206_, v_inst_207_, v_inst_208_, v_M_209_, v_N_210_, v___y_211_, v___y_212_);
lean_dec_ref(v_inst_208_);
return v_res_213_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Matrix_instMulOfFintypeOfAddCommMonoid___redArg(lean_object* v_inst_214_, lean_object* v_inst_215_, lean_object* v_inst_216_){
_start:
{
lean_object* v___f_217_; 
v___f_217_ = lean_alloc_closure((void*)(lp_mathlib_Matrix_instMulOfFintypeOfAddCommMonoid___redArg___lam__2___boxed), 7, 3);
lean_closure_set(v___f_217_, 0, v_inst_214_);
lean_closure_set(v___f_217_, 1, v_inst_215_);
lean_closure_set(v___f_217_, 2, v_inst_216_);
return v___f_217_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Matrix_instMulOfFintypeOfAddCommMonoid(lean_object* v_n_218_, lean_object* v_00_u03b1_219_, lean_object* v_inst_220_, lean_object* v_inst_221_, lean_object* v_inst_222_){
_start:
{
lean_object* v___f_223_; 
v___f_223_ = lean_alloc_closure((void*)(lp_mathlib_Matrix_instMulOfFintypeOfAddCommMonoid___redArg___lam__2___boxed), 7, 3);
lean_closure_set(v___f_223_, 0, v_inst_220_);
lean_closure_set(v___f_223_, 1, v_inst_221_);
lean_closure_set(v___f_223_, 2, v_inst_222_);
return v___f_223_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Matrix_instMulOneOfFintypeOfDecidableEqOfAddCommMonoid___redArg(lean_object* v_inst_224_, lean_object* v_inst_225_, lean_object* v_inst_226_, lean_object* v_inst_227_){
_start:
{
lean_object* v___x_228_; lean_object* v___x_229_; lean_object* v_toZero_230_; lean_object* v_toOne_231_; lean_object* v_toMul_232_; lean_object* v___x_234_; uint8_t v_isShared_235_; uint8_t v_isSharedCheck_241_; 
v___x_228_ = lp_mathlib_AddMonoid_toAddZeroClass___redArg(v_inst_227_);
v___x_229_ = lp_mathlib_AddZeroClass_toAddZero___redArg(v___x_228_);
v_toZero_230_ = lean_ctor_get(v___x_229_, 0);
lean_inc(v_toZero_230_);
lean_dec_ref(v___x_229_);
v_toOne_231_ = lean_ctor_get(v_inst_226_, 0);
v_toMul_232_ = lean_ctor_get(v_inst_226_, 1);
v_isSharedCheck_241_ = !lean_is_exclusive(v_inst_226_);
if (v_isSharedCheck_241_ == 0)
{
v___x_234_ = v_inst_226_;
v_isShared_235_ = v_isSharedCheck_241_;
goto v_resetjp_233_;
}
else
{
lean_inc(v_toMul_232_);
lean_inc(v_toOne_231_);
lean_dec(v_inst_226_);
v___x_234_ = lean_box(0);
v_isShared_235_ = v_isSharedCheck_241_;
goto v_resetjp_233_;
}
v_resetjp_233_:
{
lean_object* v___x_236_; lean_object* v___f_237_; lean_object* v___x_239_; 
v___x_236_ = lp_mathlib_Matrix_one___redArg(v_inst_225_, v_toZero_230_, v_toOne_231_);
v___f_237_ = lean_alloc_closure((void*)(lp_mathlib_Matrix_instMulOfFintypeOfAddCommMonoid___redArg___lam__2___boxed), 7, 3);
lean_closure_set(v___f_237_, 0, v_inst_224_);
lean_closure_set(v___f_237_, 1, v_toMul_232_);
lean_closure_set(v___f_237_, 2, v_inst_227_);
if (v_isShared_235_ == 0)
{
lean_ctor_set(v___x_234_, 1, v___f_237_);
lean_ctor_set(v___x_234_, 0, v___x_236_);
v___x_239_ = v___x_234_;
goto v_reusejp_238_;
}
else
{
lean_object* v_reuseFailAlloc_240_; 
v_reuseFailAlloc_240_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_240_, 0, v___x_236_);
lean_ctor_set(v_reuseFailAlloc_240_, 1, v___f_237_);
v___x_239_ = v_reuseFailAlloc_240_;
goto v_reusejp_238_;
}
v_reusejp_238_:
{
return v___x_239_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Matrix_instMulOneOfFintypeOfDecidableEqOfAddCommMonoid(lean_object* v_n_242_, lean_object* v_00_u03b1_243_, lean_object* v_inst_244_, lean_object* v_inst_245_, lean_object* v_inst_246_, lean_object* v_inst_247_){
_start:
{
lean_object* v___x_248_; 
v___x_248_ = lp_mathlib_Matrix_instMulOneOfFintypeOfDecidableEqOfAddCommMonoid___redArg(v_inst_244_, v_inst_245_, v_inst_246_, v_inst_247_);
return v___x_248_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Matrix_nonUnitalNonAssocSemiring___redArg(lean_object* v_inst_249_, lean_object* v_inst_250_){
_start:
{
lean_object* v_toAddCommMonoid_251_; lean_object* v___x_252_; lean_object* v___x_253_; lean_object* v_toMul_254_; lean_object* v___x_256_; uint8_t v_isShared_257_; uint8_t v_isSharedCheck_262_; 
v_toAddCommMonoid_251_ = lean_ctor_get(v_inst_249_, 0);
lean_inc_ref_n(v_toAddCommMonoid_251_, 2);
v___x_252_ = lp_mathlib_Matrix_addMonoid___redArg(v_toAddCommMonoid_251_);
v___x_253_ = lp_mathlib_NonUnitalNonAssocSemiring_toDistrib___redArg(v_inst_249_);
v_toMul_254_ = lean_ctor_get(v___x_253_, 0);
v_isSharedCheck_262_ = !lean_is_exclusive(v___x_253_);
if (v_isSharedCheck_262_ == 0)
{
lean_object* v_unused_263_; 
v_unused_263_ = lean_ctor_get(v___x_253_, 1);
lean_dec(v_unused_263_);
v___x_256_ = v___x_253_;
v_isShared_257_ = v_isSharedCheck_262_;
goto v_resetjp_255_;
}
else
{
lean_inc(v_toMul_254_);
lean_dec(v___x_253_);
v___x_256_ = lean_box(0);
v_isShared_257_ = v_isSharedCheck_262_;
goto v_resetjp_255_;
}
v_resetjp_255_:
{
lean_object* v___f_258_; lean_object* v___x_260_; 
v___f_258_ = lean_alloc_closure((void*)(lp_mathlib_Matrix_instMulOfFintypeOfAddCommMonoid___redArg___lam__2___boxed), 7, 3);
lean_closure_set(v___f_258_, 0, v_inst_250_);
lean_closure_set(v___f_258_, 1, v_toMul_254_);
lean_closure_set(v___f_258_, 2, v_toAddCommMonoid_251_);
if (v_isShared_257_ == 0)
{
lean_ctor_set(v___x_256_, 1, v___f_258_);
lean_ctor_set(v___x_256_, 0, v___x_252_);
v___x_260_ = v___x_256_;
goto v_reusejp_259_;
}
else
{
lean_object* v_reuseFailAlloc_261_; 
v_reuseFailAlloc_261_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_261_, 0, v___x_252_);
lean_ctor_set(v_reuseFailAlloc_261_, 1, v___f_258_);
v___x_260_ = v_reuseFailAlloc_261_;
goto v_reusejp_259_;
}
v_reusejp_259_:
{
return v___x_260_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Matrix_nonUnitalNonAssocSemiring(lean_object* v_n_264_, lean_object* v_00_u03b1_265_, lean_object* v_inst_266_, lean_object* v_inst_267_){
_start:
{
lean_object* v___x_268_; 
v___x_268_ = lp_mathlib_Matrix_nonUnitalNonAssocSemiring___redArg(v_inst_266_, v_inst_267_);
return v___x_268_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Matrix_addMonoidHomMulLeft___redArg___lam__1(lean_object* v_x_269_, lean_object* v___y_270_, lean_object* v_j_271_){
_start:
{
lean_object* v___x_272_; 
v___x_272_ = lean_apply_2(v_x_269_, v_j_271_, v___y_270_);
return v___x_272_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Matrix_addMonoidHomMulLeft___redArg___lam__0(lean_object* v_M_273_, lean_object* v_inst_274_, lean_object* v_toMul_275_, lean_object* v_toAddCommMonoid_276_, lean_object* v_x_277_, lean_object* v___y_278_, lean_object* v___y_279_){
_start:
{
lean_object* v___f_280_; lean_object* v___f_281_; lean_object* v___x_282_; 
v___f_280_ = lean_alloc_closure((void*)(lp_mathlib_Matrix_instMulOfFintypeOfAddCommMonoid___redArg___lam__1), 3, 2);
lean_closure_set(v___f_280_, 0, v_M_273_);
lean_closure_set(v___f_280_, 1, v___y_278_);
v___f_281_ = lean_alloc_closure((void*)(lp_mathlib_Matrix_addMonoidHomMulLeft___redArg___lam__1), 3, 2);
lean_closure_set(v___f_281_, 0, v_x_277_);
lean_closure_set(v___f_281_, 1, v___y_279_);
v___x_282_ = lp_mathlib_dotProduct___redArg(v_inst_274_, v_toMul_275_, v_toAddCommMonoid_276_, v___f_280_, v___f_281_);
return v___x_282_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Matrix_addMonoidHomMulLeft___redArg___lam__0___boxed(lean_object* v_M_283_, lean_object* v_inst_284_, lean_object* v_toMul_285_, lean_object* v_toAddCommMonoid_286_, lean_object* v_x_287_, lean_object* v___y_288_, lean_object* v___y_289_){
_start:
{
lean_object* v_res_290_; 
v_res_290_ = lp_mathlib_Matrix_addMonoidHomMulLeft___redArg___lam__0(v_M_283_, v_inst_284_, v_toMul_285_, v_toAddCommMonoid_286_, v_x_287_, v___y_288_, v___y_289_);
lean_dec_ref(v_toAddCommMonoid_286_);
return v_res_290_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Matrix_addMonoidHomMulLeft___redArg(lean_object* v_inst_291_, lean_object* v_inst_292_, lean_object* v_M_293_){
_start:
{
lean_object* v_toAddCommMonoid_294_; lean_object* v___x_295_; lean_object* v_toMul_296_; lean_object* v___f_297_; 
v_toAddCommMonoid_294_ = lean_ctor_get(v_inst_291_, 0);
lean_inc_ref(v_toAddCommMonoid_294_);
v___x_295_ = lp_mathlib_NonUnitalNonAssocSemiring_toDistrib___redArg(v_inst_291_);
v_toMul_296_ = lean_ctor_get(v___x_295_, 0);
lean_inc(v_toMul_296_);
lean_dec_ref(v___x_295_);
v___f_297_ = lean_alloc_closure((void*)(lp_mathlib_Matrix_addMonoidHomMulLeft___redArg___lam__0___boxed), 7, 4);
lean_closure_set(v___f_297_, 0, v_M_293_);
lean_closure_set(v___f_297_, 1, v_inst_292_);
lean_closure_set(v___f_297_, 2, v_toMul_296_);
lean_closure_set(v___f_297_, 3, v_toAddCommMonoid_294_);
return v___f_297_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Matrix_addMonoidHomMulLeft(lean_object* v_l_298_, lean_object* v_m_299_, lean_object* v_n_300_, lean_object* v_00_u03b1_301_, lean_object* v_inst_302_, lean_object* v_inst_303_, lean_object* v_M_304_){
_start:
{
lean_object* v___x_305_; 
v___x_305_ = lp_mathlib_Matrix_addMonoidHomMulLeft___redArg(v_inst_302_, v_inst_303_, v_M_304_);
return v___x_305_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Matrix_addMonoidHomMulRight___redArg___lam__0(lean_object* v_M_306_, lean_object* v___y_307_, lean_object* v_j_308_){
_start:
{
lean_object* v___x_309_; 
v___x_309_ = lean_apply_2(v_M_306_, v_j_308_, v___y_307_);
return v___x_309_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Matrix_addMonoidHomMulRight___redArg___lam__1(lean_object* v_x_310_, lean_object* v___y_311_, lean_object* v_j_312_){
_start:
{
lean_object* v___x_313_; 
v___x_313_ = lean_apply_2(v_x_310_, v___y_311_, v_j_312_);
return v___x_313_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Matrix_addMonoidHomMulRight___redArg___lam__2(lean_object* v_M_314_, lean_object* v_inst_315_, lean_object* v_toMul_316_, lean_object* v_toAddCommMonoid_317_, lean_object* v_x_318_, lean_object* v___y_319_, lean_object* v___y_320_){
_start:
{
lean_object* v___f_321_; lean_object* v___f_322_; lean_object* v___x_323_; 
v___f_321_ = lean_alloc_closure((void*)(lp_mathlib_Matrix_addMonoidHomMulRight___redArg___lam__0), 3, 2);
lean_closure_set(v___f_321_, 0, v_M_314_);
lean_closure_set(v___f_321_, 1, v___y_320_);
v___f_322_ = lean_alloc_closure((void*)(lp_mathlib_Matrix_addMonoidHomMulRight___redArg___lam__1), 3, 2);
lean_closure_set(v___f_322_, 0, v_x_318_);
lean_closure_set(v___f_322_, 1, v___y_319_);
v___x_323_ = lp_mathlib_dotProduct___redArg(v_inst_315_, v_toMul_316_, v_toAddCommMonoid_317_, v___f_322_, v___f_321_);
return v___x_323_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Matrix_addMonoidHomMulRight___redArg___lam__2___boxed(lean_object* v_M_324_, lean_object* v_inst_325_, lean_object* v_toMul_326_, lean_object* v_toAddCommMonoid_327_, lean_object* v_x_328_, lean_object* v___y_329_, lean_object* v___y_330_){
_start:
{
lean_object* v_res_331_; 
v_res_331_ = lp_mathlib_Matrix_addMonoidHomMulRight___redArg___lam__2(v_M_324_, v_inst_325_, v_toMul_326_, v_toAddCommMonoid_327_, v_x_328_, v___y_329_, v___y_330_);
lean_dec_ref(v_toAddCommMonoid_327_);
return v_res_331_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Matrix_addMonoidHomMulRight___redArg(lean_object* v_inst_332_, lean_object* v_inst_333_, lean_object* v_M_334_){
_start:
{
lean_object* v_toAddCommMonoid_335_; lean_object* v___x_336_; lean_object* v_toMul_337_; lean_object* v___f_338_; 
v_toAddCommMonoid_335_ = lean_ctor_get(v_inst_332_, 0);
lean_inc_ref(v_toAddCommMonoid_335_);
v___x_336_ = lp_mathlib_NonUnitalNonAssocSemiring_toDistrib___redArg(v_inst_332_);
v_toMul_337_ = lean_ctor_get(v___x_336_, 0);
lean_inc(v_toMul_337_);
lean_dec_ref(v___x_336_);
v___f_338_ = lean_alloc_closure((void*)(lp_mathlib_Matrix_addMonoidHomMulRight___redArg___lam__2___boxed), 7, 4);
lean_closure_set(v___f_338_, 0, v_M_334_);
lean_closure_set(v___f_338_, 1, v_inst_333_);
lean_closure_set(v___f_338_, 2, v_toMul_337_);
lean_closure_set(v___f_338_, 3, v_toAddCommMonoid_335_);
return v___f_338_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Matrix_addMonoidHomMulRight(lean_object* v_l_339_, lean_object* v_m_340_, lean_object* v_n_341_, lean_object* v_00_u03b1_342_, lean_object* v_inst_343_, lean_object* v_inst_344_, lean_object* v_M_345_){
_start:
{
lean_object* v___x_346_; 
v___x_346_ = lp_mathlib_Matrix_addMonoidHomMulRight___redArg(v_inst_343_, v_inst_344_, v_M_345_);
return v___x_346_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Matrix_nonAssocSemiring___redArg(lean_object* v_inst_347_, lean_object* v_inst_348_, lean_object* v_inst_349_){
_start:
{
lean_object* v_toNonUnitalNonAssocSemiring_350_; lean_object* v___x_351_; lean_object* v___x_352_; lean_object* v___x_353_; lean_object* v_toNatCast_354_; lean_object* v_toOne_355_; lean_object* v___x_357_; uint8_t v_isShared_358_; uint8_t v_isSharedCheck_362_; 
v_toNonUnitalNonAssocSemiring_350_ = lean_ctor_get(v_inst_347_, 0);
lean_inc_ref(v_toNonUnitalNonAssocSemiring_350_);
v___x_351_ = lp_mathlib_Matrix_nonUnitalNonAssocSemiring___redArg(v_toNonUnitalNonAssocSemiring_350_, v_inst_348_);
v___x_352_ = lp_mathlib_NonAssocSemiring_toAddCommMonoidWithOne___redArg(v_inst_347_);
v___x_353_ = lp_mathlib_Matrix_instAddCommMonoidWithOne___redArg(v_inst_349_, v___x_352_);
v_toNatCast_354_ = lean_ctor_get(v___x_353_, 0);
v_toOne_355_ = lean_ctor_get(v___x_353_, 2);
v_isSharedCheck_362_ = !lean_is_exclusive(v___x_353_);
if (v_isSharedCheck_362_ == 0)
{
lean_object* v_unused_363_; 
v_unused_363_ = lean_ctor_get(v___x_353_, 1);
lean_dec(v_unused_363_);
v___x_357_ = v___x_353_;
v_isShared_358_ = v_isSharedCheck_362_;
goto v_resetjp_356_;
}
else
{
lean_inc(v_toOne_355_);
lean_inc(v_toNatCast_354_);
lean_dec(v___x_353_);
v___x_357_ = lean_box(0);
v_isShared_358_ = v_isSharedCheck_362_;
goto v_resetjp_356_;
}
v_resetjp_356_:
{
lean_object* v___x_360_; 
if (v_isShared_358_ == 0)
{
lean_ctor_set(v___x_357_, 2, v_toNatCast_354_);
lean_ctor_set(v___x_357_, 1, v_toOne_355_);
lean_ctor_set(v___x_357_, 0, v___x_351_);
v___x_360_ = v___x_357_;
goto v_reusejp_359_;
}
else
{
lean_object* v_reuseFailAlloc_361_; 
v_reuseFailAlloc_361_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v_reuseFailAlloc_361_, 0, v___x_351_);
lean_ctor_set(v_reuseFailAlloc_361_, 1, v_toOne_355_);
lean_ctor_set(v_reuseFailAlloc_361_, 2, v_toNatCast_354_);
v___x_360_ = v_reuseFailAlloc_361_;
goto v_reusejp_359_;
}
v_reusejp_359_:
{
return v___x_360_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Matrix_nonAssocSemiring(lean_object* v_n_364_, lean_object* v_00_u03b1_365_, lean_object* v_inst_366_, lean_object* v_inst_367_, lean_object* v_inst_368_){
_start:
{
lean_object* v___x_369_; 
v___x_369_ = lp_mathlib_Matrix_nonAssocSemiring___redArg(v_inst_366_, v_inst_367_, v_inst_368_);
return v___x_369_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Matrix_nonUnitalSemiring___redArg(lean_object* v_inst_370_, lean_object* v_inst_371_){
_start:
{
lean_object* v___x_372_; 
v___x_372_ = lp_mathlib_Matrix_nonUnitalNonAssocSemiring___redArg(v_inst_370_, v_inst_371_);
return v___x_372_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Matrix_nonUnitalSemiring(lean_object* v_n_373_, lean_object* v_00_u03b1_374_, lean_object* v_inst_375_, lean_object* v_inst_376_){
_start:
{
lean_object* v___x_377_; 
v___x_377_ = lp_mathlib_Matrix_nonUnitalNonAssocSemiring___redArg(v_inst_375_, v_inst_376_);
return v___x_377_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Matrix_semiring___redArg(lean_object* v_inst_378_, lean_object* v_inst_379_, lean_object* v_inst_380_){
_start:
{
lean_object* v___x_381_; lean_object* v___x_382_; lean_object* v___x_383_; lean_object* v___x_384_; lean_object* v_toAddCommMonoid_385_; lean_object* v_toMul_386_; lean_object* v_toOne_387_; lean_object* v_toNatCast_388_; lean_object* v___x_390_; uint8_t v_isShared_391_; uint8_t v_isSharedCheck_397_; 
v___x_381_ = lp_mathlib_Semiring_toNonUnitalSemiring___redArg(v_inst_378_);
lean_inc(v_inst_379_);
v___x_382_ = lp_mathlib_Matrix_nonUnitalNonAssocSemiring___redArg(v___x_381_, v_inst_379_);
v___x_383_ = lp_mathlib_Semiring_toNonAssocSemiring___redArg(v_inst_378_);
v___x_384_ = lp_mathlib_Matrix_nonAssocSemiring___redArg(v___x_383_, v_inst_379_, v_inst_380_);
v_toAddCommMonoid_385_ = lean_ctor_get(v___x_382_, 0);
lean_inc_ref(v_toAddCommMonoid_385_);
v_toMul_386_ = lean_ctor_get(v___x_382_, 1);
lean_inc(v_toMul_386_);
lean_dec_ref(v___x_382_);
v_toOne_387_ = lean_ctor_get(v___x_384_, 1);
v_toNatCast_388_ = lean_ctor_get(v___x_384_, 2);
v_isSharedCheck_397_ = !lean_is_exclusive(v___x_384_);
if (v_isSharedCheck_397_ == 0)
{
lean_object* v_unused_398_; 
v_unused_398_ = lean_ctor_get(v___x_384_, 0);
lean_dec(v_unused_398_);
v___x_390_ = v___x_384_;
v_isShared_391_ = v_isSharedCheck_397_;
goto v_resetjp_389_;
}
else
{
lean_inc(v_toNatCast_388_);
lean_inc(v_toOne_387_);
lean_dec(v___x_384_);
v___x_390_ = lean_box(0);
v_isShared_391_ = v_isSharedCheck_397_;
goto v_resetjp_389_;
}
v_resetjp_389_:
{
lean_object* v___x_392_; lean_object* v___x_394_; 
lean_inc(v_toOne_387_);
lean_inc(v_toMul_386_);
v___x_392_ = lean_alloc_closure((void*)(lp_mathlib_npowBinRecAuto___boxed), 5, 3);
lean_closure_set(v___x_392_, 0, lean_box(0));
lean_closure_set(v___x_392_, 1, v_toMul_386_);
lean_closure_set(v___x_392_, 2, v_toOne_387_);
if (v_isShared_391_ == 0)
{
lean_ctor_set(v___x_390_, 2, v___x_392_);
lean_ctor_set(v___x_390_, 1, v_toMul_386_);
lean_ctor_set(v___x_390_, 0, v_toOne_387_);
v___x_394_ = v___x_390_;
goto v_reusejp_393_;
}
else
{
lean_object* v_reuseFailAlloc_396_; 
v_reuseFailAlloc_396_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v_reuseFailAlloc_396_, 0, v_toOne_387_);
lean_ctor_set(v_reuseFailAlloc_396_, 1, v_toMul_386_);
lean_ctor_set(v_reuseFailAlloc_396_, 2, v___x_392_);
v___x_394_ = v_reuseFailAlloc_396_;
goto v_reusejp_393_;
}
v_reusejp_393_:
{
lean_object* v___x_395_; 
v___x_395_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_395_, 0, v_toAddCommMonoid_385_);
lean_ctor_set(v___x_395_, 1, v___x_394_);
lean_ctor_set(v___x_395_, 2, v_toNatCast_388_);
return v___x_395_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Matrix_semiring(lean_object* v_n_399_, lean_object* v_00_u03b1_400_, lean_object* v_inst_401_, lean_object* v_inst_402_, lean_object* v_inst_403_){
_start:
{
lean_object* v___x_404_; 
v___x_404_ = lp_mathlib_Matrix_semiring___redArg(v_inst_401_, v_inst_402_, v_inst_403_);
return v___x_404_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Matrix_nonUnitalNonAssocRing___redArg(lean_object* v_inst_405_, lean_object* v_inst_406_){
_start:
{
lean_object* v___x_407_; lean_object* v___x_408_; lean_object* v_toAddCommGroup_409_; lean_object* v___x_411_; uint8_t v_isShared_412_; uint8_t v_isSharedCheck_430_; 
lean_inc_ref(v_inst_405_);
v___x_407_ = lp_mathlib_NonUnitalNonAssocRing_toNonUnitalNonAssocSemiring___redArg(v_inst_405_);
v___x_408_ = lp_mathlib_Matrix_nonUnitalNonAssocSemiring___redArg(v___x_407_, v_inst_406_);
v_toAddCommGroup_409_ = lean_ctor_get(v_inst_405_, 0);
v_isSharedCheck_430_ = !lean_is_exclusive(v_inst_405_);
if (v_isSharedCheck_430_ == 0)
{
lean_object* v_unused_431_; 
v_unused_431_ = lean_ctor_get(v_inst_405_, 1);
lean_dec(v_unused_431_);
v___x_411_ = v_inst_405_;
v_isShared_412_ = v_isSharedCheck_430_;
goto v_resetjp_410_;
}
else
{
lean_inc(v_toAddCommGroup_409_);
lean_dec(v_inst_405_);
v___x_411_ = lean_box(0);
v_isShared_412_ = v_isSharedCheck_430_;
goto v_resetjp_410_;
}
v_resetjp_410_:
{
lean_object* v___x_413_; lean_object* v_toAddCommMonoid_414_; lean_object* v_toMul_415_; lean_object* v_toNeg_416_; lean_object* v_toSub_417_; lean_object* v_toZSMul_418_; lean_object* v___x_420_; uint8_t v_isShared_421_; uint8_t v_isSharedCheck_428_; 
v___x_413_ = lp_mathlib_Matrix_addGroup___redArg(v_toAddCommGroup_409_);
v_toAddCommMonoid_414_ = lean_ctor_get(v___x_408_, 0);
lean_inc_ref(v_toAddCommMonoid_414_);
v_toMul_415_ = lean_ctor_get(v___x_408_, 1);
lean_inc(v_toMul_415_);
lean_dec_ref(v___x_408_);
v_toNeg_416_ = lean_ctor_get(v___x_413_, 1);
v_toSub_417_ = lean_ctor_get(v___x_413_, 2);
v_toZSMul_418_ = lean_ctor_get(v___x_413_, 3);
v_isSharedCheck_428_ = !lean_is_exclusive(v___x_413_);
if (v_isSharedCheck_428_ == 0)
{
lean_object* v_unused_429_; 
v_unused_429_ = lean_ctor_get(v___x_413_, 0);
lean_dec(v_unused_429_);
v___x_420_ = v___x_413_;
v_isShared_421_ = v_isSharedCheck_428_;
goto v_resetjp_419_;
}
else
{
lean_inc(v_toZSMul_418_);
lean_inc(v_toSub_417_);
lean_inc(v_toNeg_416_);
lean_dec(v___x_413_);
v___x_420_ = lean_box(0);
v_isShared_421_ = v_isSharedCheck_428_;
goto v_resetjp_419_;
}
v_resetjp_419_:
{
lean_object* v___x_423_; 
if (v_isShared_421_ == 0)
{
lean_ctor_set(v___x_420_, 0, v_toAddCommMonoid_414_);
v___x_423_ = v___x_420_;
goto v_reusejp_422_;
}
else
{
lean_object* v_reuseFailAlloc_427_; 
v_reuseFailAlloc_427_ = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(v_reuseFailAlloc_427_, 0, v_toAddCommMonoid_414_);
lean_ctor_set(v_reuseFailAlloc_427_, 1, v_toNeg_416_);
lean_ctor_set(v_reuseFailAlloc_427_, 2, v_toSub_417_);
lean_ctor_set(v_reuseFailAlloc_427_, 3, v_toZSMul_418_);
v___x_423_ = v_reuseFailAlloc_427_;
goto v_reusejp_422_;
}
v_reusejp_422_:
{
lean_object* v___x_425_; 
if (v_isShared_412_ == 0)
{
lean_ctor_set(v___x_411_, 1, v_toMul_415_);
lean_ctor_set(v___x_411_, 0, v___x_423_);
v___x_425_ = v___x_411_;
goto v_reusejp_424_;
}
else
{
lean_object* v_reuseFailAlloc_426_; 
v_reuseFailAlloc_426_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_426_, 0, v___x_423_);
lean_ctor_set(v_reuseFailAlloc_426_, 1, v_toMul_415_);
v___x_425_ = v_reuseFailAlloc_426_;
goto v_reusejp_424_;
}
v_reusejp_424_:
{
return v___x_425_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Matrix_nonUnitalNonAssocRing(lean_object* v_n_432_, lean_object* v_00_u03b1_433_, lean_object* v_inst_434_, lean_object* v_inst_435_){
_start:
{
lean_object* v___x_436_; 
v___x_436_ = lp_mathlib_Matrix_nonUnitalNonAssocRing___redArg(v_inst_434_, v_inst_435_);
return v___x_436_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Matrix_instNonUnitalRing___redArg(lean_object* v_inst_437_, lean_object* v_inst_438_){
_start:
{
lean_object* v___x_439_; lean_object* v___x_440_; lean_object* v_toAddCommGroup_441_; lean_object* v___x_443_; uint8_t v_isShared_444_; uint8_t v_isSharedCheck_462_; 
lean_inc_ref(v_inst_438_);
v___x_439_ = lp_mathlib_NonUnitalRing_toNonUnitalSemiring___redArg(v_inst_438_);
v___x_440_ = lp_mathlib_Matrix_nonUnitalNonAssocSemiring___redArg(v___x_439_, v_inst_437_);
v_toAddCommGroup_441_ = lean_ctor_get(v_inst_438_, 0);
v_isSharedCheck_462_ = !lean_is_exclusive(v_inst_438_);
if (v_isSharedCheck_462_ == 0)
{
lean_object* v_unused_463_; 
v_unused_463_ = lean_ctor_get(v_inst_438_, 1);
lean_dec(v_unused_463_);
v___x_443_ = v_inst_438_;
v_isShared_444_ = v_isSharedCheck_462_;
goto v_resetjp_442_;
}
else
{
lean_inc(v_toAddCommGroup_441_);
lean_dec(v_inst_438_);
v___x_443_ = lean_box(0);
v_isShared_444_ = v_isSharedCheck_462_;
goto v_resetjp_442_;
}
v_resetjp_442_:
{
lean_object* v___x_445_; lean_object* v_toAddCommMonoid_446_; lean_object* v_toMul_447_; lean_object* v_toNeg_448_; lean_object* v_toSub_449_; lean_object* v_toZSMul_450_; lean_object* v___x_452_; uint8_t v_isShared_453_; uint8_t v_isSharedCheck_460_; 
v___x_445_ = lp_mathlib_Matrix_addGroup___redArg(v_toAddCommGroup_441_);
v_toAddCommMonoid_446_ = lean_ctor_get(v___x_440_, 0);
lean_inc_ref(v_toAddCommMonoid_446_);
v_toMul_447_ = lean_ctor_get(v___x_440_, 1);
lean_inc(v_toMul_447_);
lean_dec_ref(v___x_440_);
v_toNeg_448_ = lean_ctor_get(v___x_445_, 1);
v_toSub_449_ = lean_ctor_get(v___x_445_, 2);
v_toZSMul_450_ = lean_ctor_get(v___x_445_, 3);
v_isSharedCheck_460_ = !lean_is_exclusive(v___x_445_);
if (v_isSharedCheck_460_ == 0)
{
lean_object* v_unused_461_; 
v_unused_461_ = lean_ctor_get(v___x_445_, 0);
lean_dec(v_unused_461_);
v___x_452_ = v___x_445_;
v_isShared_453_ = v_isSharedCheck_460_;
goto v_resetjp_451_;
}
else
{
lean_inc(v_toZSMul_450_);
lean_inc(v_toSub_449_);
lean_inc(v_toNeg_448_);
lean_dec(v___x_445_);
v___x_452_ = lean_box(0);
v_isShared_453_ = v_isSharedCheck_460_;
goto v_resetjp_451_;
}
v_resetjp_451_:
{
lean_object* v___x_455_; 
if (v_isShared_453_ == 0)
{
lean_ctor_set(v___x_452_, 0, v_toAddCommMonoid_446_);
v___x_455_ = v___x_452_;
goto v_reusejp_454_;
}
else
{
lean_object* v_reuseFailAlloc_459_; 
v_reuseFailAlloc_459_ = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(v_reuseFailAlloc_459_, 0, v_toAddCommMonoid_446_);
lean_ctor_set(v_reuseFailAlloc_459_, 1, v_toNeg_448_);
lean_ctor_set(v_reuseFailAlloc_459_, 2, v_toSub_449_);
lean_ctor_set(v_reuseFailAlloc_459_, 3, v_toZSMul_450_);
v___x_455_ = v_reuseFailAlloc_459_;
goto v_reusejp_454_;
}
v_reusejp_454_:
{
lean_object* v___x_457_; 
if (v_isShared_444_ == 0)
{
lean_ctor_set(v___x_443_, 1, v_toMul_447_);
lean_ctor_set(v___x_443_, 0, v___x_455_);
v___x_457_ = v___x_443_;
goto v_reusejp_456_;
}
else
{
lean_object* v_reuseFailAlloc_458_; 
v_reuseFailAlloc_458_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_458_, 0, v___x_455_);
lean_ctor_set(v_reuseFailAlloc_458_, 1, v_toMul_447_);
v___x_457_ = v_reuseFailAlloc_458_;
goto v_reusejp_456_;
}
v_reusejp_456_:
{
return v___x_457_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Matrix_instNonUnitalRing(lean_object* v_n_464_, lean_object* v_00_u03b1_465_, lean_object* v_inst_466_, lean_object* v_inst_467_){
_start:
{
lean_object* v___x_468_; 
v___x_468_ = lp_mathlib_Matrix_instNonUnitalRing___redArg(v_inst_466_, v_inst_467_);
return v___x_468_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Matrix_instNonAssocRing___redArg(lean_object* v_inst_469_, lean_object* v_inst_470_, lean_object* v_inst_471_){
_start:
{
lean_object* v___x_472_; lean_object* v___x_473_; lean_object* v___x_474_; lean_object* v___x_475_; lean_object* v_toNonUnitalNonAssocSemiring_476_; lean_object* v_toAddCommGroup_477_; lean_object* v_toOne_478_; lean_object* v_toNatCast_479_; lean_object* v_toAddCommMonoid_480_; lean_object* v_toMul_481_; lean_object* v___x_483_; uint8_t v_isShared_484_; uint8_t v_isSharedCheck_510_; 
lean_inc_ref(v_inst_471_);
v___x_472_ = lp_mathlib_NonAssocRing_toNonAssocSemiring___redArg(v_inst_471_);
lean_inc_ref(v_inst_470_);
v___x_473_ = lp_mathlib_Matrix_nonAssocSemiring___redArg(v___x_472_, v_inst_469_, v_inst_470_);
v___x_474_ = lp_mathlib_NonAssocRing_toAddCommGroupWithOne___redArg(v_inst_471_);
v___x_475_ = lp_mathlib_Matrix_instAddCommGroupWithOne___redArg(v_inst_470_, v___x_474_);
v_toNonUnitalNonAssocSemiring_476_ = lean_ctor_get(v___x_473_, 0);
lean_inc_ref(v_toNonUnitalNonAssocSemiring_476_);
v_toAddCommGroup_477_ = lean_ctor_get(v___x_475_, 0);
lean_inc_ref(v_toAddCommGroup_477_);
v_toOne_478_ = lean_ctor_get(v___x_473_, 1);
lean_inc(v_toOne_478_);
v_toNatCast_479_ = lean_ctor_get(v___x_473_, 2);
lean_inc(v_toNatCast_479_);
lean_dec_ref(v___x_473_);
v_toAddCommMonoid_480_ = lean_ctor_get(v_toNonUnitalNonAssocSemiring_476_, 0);
v_toMul_481_ = lean_ctor_get(v_toNonUnitalNonAssocSemiring_476_, 1);
v_isSharedCheck_510_ = !lean_is_exclusive(v_toNonUnitalNonAssocSemiring_476_);
if (v_isSharedCheck_510_ == 0)
{
v___x_483_ = v_toNonUnitalNonAssocSemiring_476_;
v_isShared_484_ = v_isSharedCheck_510_;
goto v_resetjp_482_;
}
else
{
lean_inc(v_toMul_481_);
lean_inc(v_toAddCommMonoid_480_);
lean_dec(v_toNonUnitalNonAssocSemiring_476_);
v___x_483_ = lean_box(0);
v_isShared_484_ = v_isSharedCheck_510_;
goto v_resetjp_482_;
}
v_resetjp_482_:
{
lean_object* v_toIntCast_485_; lean_object* v___x_487_; uint8_t v_isShared_488_; uint8_t v_isSharedCheck_506_; 
v_toIntCast_485_ = lean_ctor_get(v___x_475_, 1);
v_isSharedCheck_506_ = !lean_is_exclusive(v___x_475_);
if (v_isSharedCheck_506_ == 0)
{
lean_object* v_unused_507_; lean_object* v_unused_508_; lean_object* v_unused_509_; 
v_unused_507_ = lean_ctor_get(v___x_475_, 3);
lean_dec(v_unused_507_);
v_unused_508_ = lean_ctor_get(v___x_475_, 2);
lean_dec(v_unused_508_);
v_unused_509_ = lean_ctor_get(v___x_475_, 0);
lean_dec(v_unused_509_);
v___x_487_ = v___x_475_;
v_isShared_488_ = v_isSharedCheck_506_;
goto v_resetjp_486_;
}
else
{
lean_inc(v_toIntCast_485_);
lean_dec(v___x_475_);
v___x_487_ = lean_box(0);
v_isShared_488_ = v_isSharedCheck_506_;
goto v_resetjp_486_;
}
v_resetjp_486_:
{
lean_object* v_toNeg_489_; lean_object* v_toSub_490_; lean_object* v_toZSMul_491_; lean_object* v___x_493_; uint8_t v_isShared_494_; uint8_t v_isSharedCheck_504_; 
v_toNeg_489_ = lean_ctor_get(v_toAddCommGroup_477_, 1);
v_toSub_490_ = lean_ctor_get(v_toAddCommGroup_477_, 2);
v_toZSMul_491_ = lean_ctor_get(v_toAddCommGroup_477_, 3);
v_isSharedCheck_504_ = !lean_is_exclusive(v_toAddCommGroup_477_);
if (v_isSharedCheck_504_ == 0)
{
lean_object* v_unused_505_; 
v_unused_505_ = lean_ctor_get(v_toAddCommGroup_477_, 0);
lean_dec(v_unused_505_);
v___x_493_ = v_toAddCommGroup_477_;
v_isShared_494_ = v_isSharedCheck_504_;
goto v_resetjp_492_;
}
else
{
lean_inc(v_toZSMul_491_);
lean_inc(v_toSub_490_);
lean_inc(v_toNeg_489_);
lean_dec(v_toAddCommGroup_477_);
v___x_493_ = lean_box(0);
v_isShared_494_ = v_isSharedCheck_504_;
goto v_resetjp_492_;
}
v_resetjp_492_:
{
lean_object* v___x_496_; 
if (v_isShared_494_ == 0)
{
lean_ctor_set(v___x_493_, 0, v_toAddCommMonoid_480_);
v___x_496_ = v___x_493_;
goto v_reusejp_495_;
}
else
{
lean_object* v_reuseFailAlloc_503_; 
v_reuseFailAlloc_503_ = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(v_reuseFailAlloc_503_, 0, v_toAddCommMonoid_480_);
lean_ctor_set(v_reuseFailAlloc_503_, 1, v_toNeg_489_);
lean_ctor_set(v_reuseFailAlloc_503_, 2, v_toSub_490_);
lean_ctor_set(v_reuseFailAlloc_503_, 3, v_toZSMul_491_);
v___x_496_ = v_reuseFailAlloc_503_;
goto v_reusejp_495_;
}
v_reusejp_495_:
{
lean_object* v___x_498_; 
if (v_isShared_484_ == 0)
{
lean_ctor_set(v___x_483_, 0, v___x_496_);
v___x_498_ = v___x_483_;
goto v_reusejp_497_;
}
else
{
lean_object* v_reuseFailAlloc_502_; 
v_reuseFailAlloc_502_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_502_, 0, v___x_496_);
lean_ctor_set(v_reuseFailAlloc_502_, 1, v_toMul_481_);
v___x_498_ = v_reuseFailAlloc_502_;
goto v_reusejp_497_;
}
v_reusejp_497_:
{
lean_object* v___x_500_; 
if (v_isShared_488_ == 0)
{
lean_ctor_set(v___x_487_, 3, v_toIntCast_485_);
lean_ctor_set(v___x_487_, 2, v_toNatCast_479_);
lean_ctor_set(v___x_487_, 1, v_toOne_478_);
lean_ctor_set(v___x_487_, 0, v___x_498_);
v___x_500_ = v___x_487_;
goto v_reusejp_499_;
}
else
{
lean_object* v_reuseFailAlloc_501_; 
v_reuseFailAlloc_501_ = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(v_reuseFailAlloc_501_, 0, v___x_498_);
lean_ctor_set(v_reuseFailAlloc_501_, 1, v_toOne_478_);
lean_ctor_set(v_reuseFailAlloc_501_, 2, v_toNatCast_479_);
lean_ctor_set(v_reuseFailAlloc_501_, 3, v_toIntCast_485_);
v___x_500_ = v_reuseFailAlloc_501_;
goto v_reusejp_499_;
}
v_reusejp_499_:
{
return v___x_500_;
}
}
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Matrix_instNonAssocRing(lean_object* v_n_511_, lean_object* v_00_u03b1_512_, lean_object* v_inst_513_, lean_object* v_inst_514_, lean_object* v_inst_515_){
_start:
{
lean_object* v___x_516_; 
v___x_516_ = lp_mathlib_Matrix_instNonAssocRing___redArg(v_inst_513_, v_inst_514_, v_inst_515_);
return v___x_516_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Matrix_instRing___redArg(lean_object* v_inst_517_, lean_object* v_inst_518_, lean_object* v_inst_519_){
_start:
{
lean_object* v_toSemiring_520_; lean_object* v___x_521_; lean_object* v___x_522_; lean_object* v___x_524_; uint8_t v_isShared_525_; uint8_t v_isSharedCheck_536_; 
v_toSemiring_520_ = lean_ctor_get(v_inst_519_, 0);
lean_inc_ref(v_inst_518_);
lean_inc_ref(v_toSemiring_520_);
v___x_521_ = lp_mathlib_Matrix_semiring___redArg(v_toSemiring_520_, v_inst_517_, v_inst_518_);
v___x_522_ = lp_mathlib_Ring_toNonAssocRing___redArg(v_inst_519_);
v_isSharedCheck_536_ = !lean_is_exclusive(v_inst_519_);
if (v_isSharedCheck_536_ == 0)
{
lean_object* v_unused_537_; lean_object* v_unused_538_; lean_object* v_unused_539_; lean_object* v_unused_540_; lean_object* v_unused_541_; 
v_unused_537_ = lean_ctor_get(v_inst_519_, 4);
lean_dec(v_unused_537_);
v_unused_538_ = lean_ctor_get(v_inst_519_, 3);
lean_dec(v_unused_538_);
v_unused_539_ = lean_ctor_get(v_inst_519_, 2);
lean_dec(v_unused_539_);
v_unused_540_ = lean_ctor_get(v_inst_519_, 1);
lean_dec(v_unused_540_);
v_unused_541_ = lean_ctor_get(v_inst_519_, 0);
lean_dec(v_unused_541_);
v___x_524_ = v_inst_519_;
v_isShared_525_ = v_isSharedCheck_536_;
goto v_resetjp_523_;
}
else
{
lean_dec(v_inst_519_);
v___x_524_ = lean_box(0);
v_isShared_525_ = v_isSharedCheck_536_;
goto v_resetjp_523_;
}
v_resetjp_523_:
{
lean_object* v___x_526_; lean_object* v___x_527_; lean_object* v_toAddCommGroup_528_; lean_object* v_toIntCast_529_; lean_object* v_toNeg_530_; lean_object* v_toSub_531_; lean_object* v_toZSMul_532_; lean_object* v___x_534_; 
v___x_526_ = lp_mathlib_NonAssocRing_toAddCommGroupWithOne___redArg(v___x_522_);
v___x_527_ = lp_mathlib_Matrix_instAddCommGroupWithOne___redArg(v_inst_518_, v___x_526_);
v_toAddCommGroup_528_ = lean_ctor_get(v___x_527_, 0);
lean_inc_ref(v_toAddCommGroup_528_);
v_toIntCast_529_ = lean_ctor_get(v___x_527_, 1);
lean_inc(v_toIntCast_529_);
lean_dec_ref(v___x_527_);
v_toNeg_530_ = lean_ctor_get(v_toAddCommGroup_528_, 1);
lean_inc(v_toNeg_530_);
v_toSub_531_ = lean_ctor_get(v_toAddCommGroup_528_, 2);
lean_inc(v_toSub_531_);
v_toZSMul_532_ = lean_ctor_get(v_toAddCommGroup_528_, 3);
lean_inc(v_toZSMul_532_);
lean_dec_ref(v_toAddCommGroup_528_);
if (v_isShared_525_ == 0)
{
lean_ctor_set(v___x_524_, 4, v_toIntCast_529_);
lean_ctor_set(v___x_524_, 3, v_toZSMul_532_);
lean_ctor_set(v___x_524_, 2, v_toSub_531_);
lean_ctor_set(v___x_524_, 1, v_toNeg_530_);
lean_ctor_set(v___x_524_, 0, v___x_521_);
v___x_534_ = v___x_524_;
goto v_reusejp_533_;
}
else
{
lean_object* v_reuseFailAlloc_535_; 
v_reuseFailAlloc_535_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_535_, 0, v___x_521_);
lean_ctor_set(v_reuseFailAlloc_535_, 1, v_toNeg_530_);
lean_ctor_set(v_reuseFailAlloc_535_, 2, v_toSub_531_);
lean_ctor_set(v_reuseFailAlloc_535_, 3, v_toZSMul_532_);
lean_ctor_set(v_reuseFailAlloc_535_, 4, v_toIntCast_529_);
v___x_534_ = v_reuseFailAlloc_535_;
goto v_reusejp_533_;
}
v_reusejp_533_:
{
return v___x_534_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Matrix_instRing(lean_object* v_n_542_, lean_object* v_00_u03b1_543_, lean_object* v_inst_544_, lean_object* v_inst_545_, lean_object* v_inst_546_){
_start:
{
lean_object* v___x_547_; 
v___x_547_ = lp_mathlib_Matrix_instRing___redArg(v_inst_544_, v_inst_545_, v_inst_546_);
return v___x_547_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Matrix_vecMulVec___redArg___lam__0(lean_object* v_w_548_, lean_object* v_v_549_, lean_object* v_inst_550_, lean_object* v_x_551_, lean_object* v_y_552_){
_start:
{
lean_object* v___x_553_; lean_object* v___x_554_; lean_object* v___x_555_; 
v___x_553_ = lean_apply_1(v_w_548_, v_x_551_);
v___x_554_ = lean_apply_1(v_v_549_, v_y_552_);
v___x_555_ = lean_apply_2(v_inst_550_, v___x_553_, v___x_554_);
return v___x_555_;
}
}
static lean_object* _init_lp_mathlib_Matrix_vecMulVec___redArg___closed__0(void){
_start:
{
lean_object* v___x_556_; 
v___x_556_ = lp_mathlib_Equiv_refl(lean_box(0));
return v___x_556_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Matrix_vecMulVec___redArg(lean_object* v_inst_557_, lean_object* v_w_558_, lean_object* v_v_559_, lean_object* v_a_560_, lean_object* v_a_561_){
_start:
{
lean_object* v___x_562_; lean_object* v_toFun_563_; lean_object* v___f_564_; lean_object* v___x_565_; 
v___x_562_ = lean_obj_once(&lp_mathlib_Matrix_vecMulVec___redArg___closed__0, &lp_mathlib_Matrix_vecMulVec___redArg___closed__0_once, _init_lp_mathlib_Matrix_vecMulVec___redArg___closed__0);
v_toFun_563_ = lean_ctor_get(v___x_562_, 0);
v___f_564_ = lean_alloc_closure((void*)(lp_mathlib_Matrix_vecMulVec___redArg___lam__0), 5, 3);
lean_closure_set(v___f_564_, 0, v_w_558_);
lean_closure_set(v___f_564_, 1, v_v_559_);
lean_closure_set(v___f_564_, 2, v_inst_557_);
lean_inc(v_toFun_563_);
v___x_565_ = lean_apply_3(v_toFun_563_, v___f_564_, v_a_560_, v_a_561_);
return v___x_565_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Matrix_vecMulVec(lean_object* v_m_566_, lean_object* v_n_567_, lean_object* v_00_u03b1_568_, lean_object* v_inst_569_, lean_object* v_w_570_, lean_object* v_v_571_, lean_object* v_a_572_, lean_object* v_a_573_){
_start:
{
lean_object* v___x_574_; 
v___x_574_ = lp_mathlib_Matrix_vecMulVec___redArg(v_inst_569_, v_w_570_, v_v_571_, v_a_572_, v_a_573_);
return v___x_574_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Matrix_mulVec___redArg___lam__0(lean_object* v_M_575_, lean_object* v_x_576_, lean_object* v_j_577_){
_start:
{
lean_object* v___x_578_; 
v___x_578_ = lean_apply_2(v_M_575_, v_x_576_, v_j_577_);
return v___x_578_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Matrix_mulVec___redArg(lean_object* v_inst_579_, lean_object* v_inst_580_, lean_object* v_M_581_, lean_object* v_v_582_, lean_object* v_x_583_){
_start:
{
lean_object* v___x_584_; lean_object* v_toMul_585_; lean_object* v_toAddCommMonoid_586_; lean_object* v___f_587_; lean_object* v___x_588_; 
lean_inc_ref(v_inst_579_);
v___x_584_ = lp_mathlib_NonUnitalNonAssocSemiring_toDistrib___redArg(v_inst_579_);
v_toMul_585_ = lean_ctor_get(v___x_584_, 0);
lean_inc(v_toMul_585_);
lean_dec_ref(v___x_584_);
v_toAddCommMonoid_586_ = lean_ctor_get(v_inst_579_, 0);
lean_inc_ref(v_toAddCommMonoid_586_);
lean_dec_ref(v_inst_579_);
v___f_587_ = lean_alloc_closure((void*)(lp_mathlib_Matrix_mulVec___redArg___lam__0), 3, 2);
lean_closure_set(v___f_587_, 0, v_M_581_);
lean_closure_set(v___f_587_, 1, v_x_583_);
v___x_588_ = lp_mathlib_dotProduct___redArg(v_inst_580_, v_toMul_585_, v_toAddCommMonoid_586_, v___f_587_, v_v_582_);
lean_dec_ref(v_toAddCommMonoid_586_);
return v___x_588_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Matrix_mulVec(lean_object* v_m_589_, lean_object* v_n_590_, lean_object* v_00_u03b1_591_, lean_object* v_inst_592_, lean_object* v_inst_593_, lean_object* v_M_594_, lean_object* v_v_595_, lean_object* v_x_596_){
_start:
{
lean_object* v___x_597_; 
v___x_597_ = lp_mathlib_Matrix_mulVec___redArg(v_inst_592_, v_inst_593_, v_M_594_, v_v_595_, v_x_596_);
return v___x_597_;
}
}
static lean_object* _init_lp_mathlib_Matrix___aux__Mathlib__Data__Matrix__Mul______macroRules__Matrix__term___x2a_u1d65____1___closed__1(void){
_start:
{
lean_object* v___x_617_; lean_object* v___x_618_; 
v___x_617_ = ((lean_object*)(lp_mathlib_Matrix___aux__Mathlib__Data__Matrix__Mul______macroRules__Matrix__term___x2a_u1d65____1___closed__0));
v___x_618_ = l_String_toRawSubstring_x27(v___x_617_);
return v___x_618_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Matrix___aux__Mathlib__Data__Matrix__Mul______macroRules__Matrix__term___x2a_u1d65____1(lean_object* v_x_629_, lean_object* v_a_630_, lean_object* v_a_631_){
_start:
{
lean_object* v___x_632_; uint8_t v___x_633_; 
v___x_632_ = ((lean_object*)(lp_mathlib_Matrix_term___x2a_u1d65___00__closed__2));
lean_inc(v_x_629_);
v___x_633_ = l_Lean_Syntax_isOfKind(v_x_629_, v___x_632_);
if (v___x_633_ == 0)
{
lean_object* v___x_634_; lean_object* v___x_635_; 
lean_dec(v_x_629_);
v___x_634_ = lean_box(1);
v___x_635_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_635_, 0, v___x_634_);
lean_ctor_set(v___x_635_, 1, v_a_631_);
return v___x_635_;
}
else
{
lean_object* v_quotContext_636_; lean_object* v_currMacroScope_637_; lean_object* v_ref_638_; lean_object* v___x_639_; lean_object* v___x_640_; lean_object* v___x_641_; lean_object* v___x_642_; uint8_t v___x_643_; lean_object* v___x_644_; lean_object* v___x_645_; lean_object* v___x_646_; lean_object* v___x_647_; lean_object* v___x_648_; lean_object* v___x_649_; lean_object* v___x_650_; lean_object* v___x_651_; lean_object* v___x_652_; lean_object* v___x_653_; lean_object* v___x_654_; 
v_quotContext_636_ = lean_ctor_get(v_a_630_, 1);
v_currMacroScope_637_ = lean_ctor_get(v_a_630_, 2);
v_ref_638_ = lean_ctor_get(v_a_630_, 5);
v___x_639_ = lean_unsigned_to_nat(0u);
v___x_640_ = l_Lean_Syntax_getArg(v_x_629_, v___x_639_);
v___x_641_ = lean_unsigned_to_nat(2u);
v___x_642_ = l_Lean_Syntax_getArg(v_x_629_, v___x_641_);
lean_dec(v_x_629_);
v___x_643_ = 0;
v___x_644_ = l_Lean_SourceInfo_fromRef(v_ref_638_, v___x_643_);
v___x_645_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Data__Matrix__Mul______macroRules__term___u2b1d_u1d65____1___closed__4));
v___x_646_ = lean_obj_once(&lp_mathlib_Matrix___aux__Mathlib__Data__Matrix__Mul______macroRules__Matrix__term___x2a_u1d65____1___closed__1, &lp_mathlib_Matrix___aux__Mathlib__Data__Matrix__Mul______macroRules__Matrix__term___x2a_u1d65____1___closed__1_once, _init_lp_mathlib_Matrix___aux__Mathlib__Data__Matrix__Mul______macroRules__Matrix__term___x2a_u1d65____1___closed__1);
v___x_647_ = ((lean_object*)(lp_mathlib_Matrix___aux__Mathlib__Data__Matrix__Mul______macroRules__Matrix__term___x2a_u1d65____1___closed__3));
lean_inc(v_currMacroScope_637_);
lean_inc(v_quotContext_636_);
v___x_648_ = l_Lean_addMacroScope(v_quotContext_636_, v___x_647_, v_currMacroScope_637_);
v___x_649_ = ((lean_object*)(lp_mathlib_Matrix___aux__Mathlib__Data__Matrix__Mul______macroRules__Matrix__term___x2a_u1d65____1___closed__5));
lean_inc_n(v___x_644_, 2);
v___x_650_ = lean_alloc_ctor(3, 4, 0);
lean_ctor_set(v___x_650_, 0, v___x_644_);
lean_ctor_set(v___x_650_, 1, v___x_646_);
lean_ctor_set(v___x_650_, 2, v___x_648_);
lean_ctor_set(v___x_650_, 3, v___x_649_);
v___x_651_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Data__Matrix__Mul______macroRules__term___u2b1d_u1d65____1___closed__11));
v___x_652_ = l_Lean_Syntax_node2(v___x_644_, v___x_651_, v___x_640_, v___x_642_);
v___x_653_ = l_Lean_Syntax_node2(v___x_644_, v___x_645_, v___x_650_, v___x_652_);
v___x_654_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_654_, 0, v___x_653_);
lean_ctor_set(v___x_654_, 1, v_a_631_);
return v___x_654_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Matrix___aux__Mathlib__Data__Matrix__Mul______macroRules__Matrix__term___x2a_u1d65____1___boxed(lean_object* v_x_655_, lean_object* v_a_656_, lean_object* v_a_657_){
_start:
{
lean_object* v_res_658_; 
v_res_658_ = lp_mathlib_Matrix___aux__Mathlib__Data__Matrix__Mul______macroRules__Matrix__term___x2a_u1d65____1(v_x_655_, v_a_656_, v_a_657_);
lean_dec_ref(v_a_656_);
return v_res_658_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Matrix___aux__Mathlib__Data__Matrix__Mul______unexpand__Matrix__mulVec__1(lean_object* v_x_659_, lean_object* v_a_660_, lean_object* v_a_661_){
_start:
{
lean_object* v___x_662_; uint8_t v___x_663_; 
v___x_662_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Data__Matrix__Mul______macroRules__term___u2b1d_u1d65____1___closed__4));
lean_inc(v_x_659_);
v___x_663_ = l_Lean_Syntax_isOfKind(v_x_659_, v___x_662_);
if (v___x_663_ == 0)
{
lean_object* v___x_664_; lean_object* v___x_665_; 
lean_dec(v_x_659_);
v___x_664_ = lean_box(0);
v___x_665_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_665_, 0, v___x_664_);
lean_ctor_set(v___x_665_, 1, v_a_661_);
return v___x_665_;
}
else
{
lean_object* v___x_666_; lean_object* v___x_667_; lean_object* v___x_668_; uint8_t v___x_669_; 
v___x_666_ = lean_unsigned_to_nat(0u);
v___x_667_ = l_Lean_Syntax_getArg(v_x_659_, v___x_666_);
v___x_668_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Data__Matrix__Mul______unexpand__dotProduct__1___closed__1));
lean_inc(v___x_667_);
v___x_669_ = l_Lean_Syntax_isOfKind(v___x_667_, v___x_668_);
if (v___x_669_ == 0)
{
lean_object* v___x_670_; lean_object* v___x_671_; 
lean_dec(v___x_667_);
lean_dec(v_x_659_);
v___x_670_ = lean_box(0);
v___x_671_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_671_, 0, v___x_670_);
lean_ctor_set(v___x_671_, 1, v_a_661_);
return v___x_671_;
}
else
{
lean_object* v___x_672_; lean_object* v___x_673_; lean_object* v___x_674_; uint8_t v___x_675_; 
v___x_672_ = lean_unsigned_to_nat(1u);
v___x_673_ = l_Lean_Syntax_getArg(v_x_659_, v___x_672_);
lean_dec(v_x_659_);
v___x_674_ = lean_unsigned_to_nat(2u);
lean_inc(v___x_673_);
v___x_675_ = l_Lean_Syntax_matchesNull(v___x_673_, v___x_674_);
if (v___x_675_ == 0)
{
lean_object* v___x_676_; lean_object* v___x_677_; 
lean_dec(v___x_673_);
lean_dec(v___x_667_);
v___x_676_ = lean_box(0);
v___x_677_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_677_, 0, v___x_676_);
lean_ctor_set(v___x_677_, 1, v_a_661_);
return v___x_677_;
}
else
{
lean_object* v___x_678_; lean_object* v___x_679_; lean_object* v_ref_680_; uint8_t v___x_681_; lean_object* v___x_682_; lean_object* v___x_683_; lean_object* v___x_684_; lean_object* v___x_685_; lean_object* v___x_686_; lean_object* v___x_687_; 
v___x_678_ = l_Lean_Syntax_getArg(v___x_673_, v___x_666_);
v___x_679_ = l_Lean_Syntax_getArg(v___x_673_, v___x_672_);
lean_dec(v___x_673_);
v_ref_680_ = l_Lean_replaceRef(v___x_667_, v_a_660_);
lean_dec(v___x_667_);
v___x_681_ = 0;
v___x_682_ = l_Lean_SourceInfo_fromRef(v_ref_680_, v___x_681_);
lean_dec(v_ref_680_);
v___x_683_ = ((lean_object*)(lp_mathlib_Matrix_term___x2a_u1d65___00__closed__2));
v___x_684_ = ((lean_object*)(lp_mathlib_Matrix_term___x2a_u1d65___00__closed__3));
lean_inc(v___x_682_);
v___x_685_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_685_, 0, v___x_682_);
lean_ctor_set(v___x_685_, 1, v___x_684_);
v___x_686_ = l_Lean_Syntax_node3(v___x_682_, v___x_683_, v___x_678_, v___x_685_, v___x_679_);
v___x_687_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_687_, 0, v___x_686_);
lean_ctor_set(v___x_687_, 1, v_a_661_);
return v___x_687_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Matrix___aux__Mathlib__Data__Matrix__Mul______unexpand__Matrix__mulVec__1___boxed(lean_object* v_x_688_, lean_object* v_a_689_, lean_object* v_a_690_){
_start:
{
lean_object* v_res_691_; 
v_res_691_ = lp_mathlib_Matrix___aux__Mathlib__Data__Matrix__Mul______unexpand__Matrix__mulVec__1(v_x_688_, v_a_689_, v_a_690_);
lean_dec(v_a_689_);
return v_res_691_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Matrix_vecMul___redArg___lam__0(lean_object* v_M_692_, lean_object* v_x_693_, lean_object* v_i_694_){
_start:
{
lean_object* v___x_695_; 
v___x_695_ = lean_apply_2(v_M_692_, v_i_694_, v_x_693_);
return v___x_695_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Matrix_vecMul___redArg(lean_object* v_inst_696_, lean_object* v_inst_697_, lean_object* v_v_698_, lean_object* v_M_699_, lean_object* v_x_700_){
_start:
{
lean_object* v___x_701_; lean_object* v_toMul_702_; lean_object* v_toAddCommMonoid_703_; lean_object* v___f_704_; lean_object* v___x_705_; 
lean_inc_ref(v_inst_696_);
v___x_701_ = lp_mathlib_NonUnitalNonAssocSemiring_toDistrib___redArg(v_inst_696_);
v_toMul_702_ = lean_ctor_get(v___x_701_, 0);
lean_inc(v_toMul_702_);
lean_dec_ref(v___x_701_);
v_toAddCommMonoid_703_ = lean_ctor_get(v_inst_696_, 0);
lean_inc_ref(v_toAddCommMonoid_703_);
lean_dec_ref(v_inst_696_);
v___f_704_ = lean_alloc_closure((void*)(lp_mathlib_Matrix_vecMul___redArg___lam__0), 3, 2);
lean_closure_set(v___f_704_, 0, v_M_699_);
lean_closure_set(v___f_704_, 1, v_x_700_);
v___x_705_ = lp_mathlib_dotProduct___redArg(v_inst_697_, v_toMul_702_, v_toAddCommMonoid_703_, v_v_698_, v___f_704_);
lean_dec_ref(v_toAddCommMonoid_703_);
return v___x_705_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Matrix_vecMul(lean_object* v_m_706_, lean_object* v_n_707_, lean_object* v_00_u03b1_708_, lean_object* v_inst_709_, lean_object* v_inst_710_, lean_object* v_v_711_, lean_object* v_M_712_, lean_object* v_x_713_){
_start:
{
lean_object* v___x_714_; 
v___x_714_ = lp_mathlib_Matrix_vecMul___redArg(v_inst_709_, v_inst_710_, v_v_711_, v_M_712_, v_x_713_);
return v___x_714_;
}
}
static lean_object* _init_lp_mathlib_Matrix___aux__Mathlib__Data__Matrix__Mul______macroRules__Matrix__term___u1d65_x2a____1___closed__1(void){
_start:
{
lean_object* v___x_735_; lean_object* v___x_736_; 
v___x_735_ = ((lean_object*)(lp_mathlib_Matrix___aux__Mathlib__Data__Matrix__Mul______macroRules__Matrix__term___u1d65_x2a____1___closed__0));
v___x_736_ = l_String_toRawSubstring_x27(v___x_735_);
return v___x_736_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Matrix___aux__Mathlib__Data__Matrix__Mul______macroRules__Matrix__term___u1d65_x2a____1(lean_object* v_x_747_, lean_object* v_a_748_, lean_object* v_a_749_){
_start:
{
lean_object* v___x_750_; uint8_t v___x_751_; 
v___x_750_ = ((lean_object*)(lp_mathlib_Matrix_term___u1d65_x2a___00__closed__1));
lean_inc(v_x_747_);
v___x_751_ = l_Lean_Syntax_isOfKind(v_x_747_, v___x_750_);
if (v___x_751_ == 0)
{
lean_object* v___x_752_; lean_object* v___x_753_; 
lean_dec(v_x_747_);
v___x_752_ = lean_box(1);
v___x_753_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_753_, 0, v___x_752_);
lean_ctor_set(v___x_753_, 1, v_a_749_);
return v___x_753_;
}
else
{
lean_object* v_quotContext_754_; lean_object* v_currMacroScope_755_; lean_object* v_ref_756_; lean_object* v___x_757_; lean_object* v___x_758_; lean_object* v___x_759_; lean_object* v___x_760_; uint8_t v___x_761_; lean_object* v___x_762_; lean_object* v___x_763_; lean_object* v___x_764_; lean_object* v___x_765_; lean_object* v___x_766_; lean_object* v___x_767_; lean_object* v___x_768_; lean_object* v___x_769_; lean_object* v___x_770_; lean_object* v___x_771_; lean_object* v___x_772_; 
v_quotContext_754_ = lean_ctor_get(v_a_748_, 1);
v_currMacroScope_755_ = lean_ctor_get(v_a_748_, 2);
v_ref_756_ = lean_ctor_get(v_a_748_, 5);
v___x_757_ = lean_unsigned_to_nat(0u);
v___x_758_ = l_Lean_Syntax_getArg(v_x_747_, v___x_757_);
v___x_759_ = lean_unsigned_to_nat(2u);
v___x_760_ = l_Lean_Syntax_getArg(v_x_747_, v___x_759_);
lean_dec(v_x_747_);
v___x_761_ = 0;
v___x_762_ = l_Lean_SourceInfo_fromRef(v_ref_756_, v___x_761_);
v___x_763_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Data__Matrix__Mul______macroRules__term___u2b1d_u1d65____1___closed__4));
v___x_764_ = lean_obj_once(&lp_mathlib_Matrix___aux__Mathlib__Data__Matrix__Mul______macroRules__Matrix__term___u1d65_x2a____1___closed__1, &lp_mathlib_Matrix___aux__Mathlib__Data__Matrix__Mul______macroRules__Matrix__term___u1d65_x2a____1___closed__1_once, _init_lp_mathlib_Matrix___aux__Mathlib__Data__Matrix__Mul______macroRules__Matrix__term___u1d65_x2a____1___closed__1);
v___x_765_ = ((lean_object*)(lp_mathlib_Matrix___aux__Mathlib__Data__Matrix__Mul______macroRules__Matrix__term___u1d65_x2a____1___closed__3));
lean_inc(v_currMacroScope_755_);
lean_inc(v_quotContext_754_);
v___x_766_ = l_Lean_addMacroScope(v_quotContext_754_, v___x_765_, v_currMacroScope_755_);
v___x_767_ = ((lean_object*)(lp_mathlib_Matrix___aux__Mathlib__Data__Matrix__Mul______macroRules__Matrix__term___u1d65_x2a____1___closed__5));
lean_inc_n(v___x_762_, 2);
v___x_768_ = lean_alloc_ctor(3, 4, 0);
lean_ctor_set(v___x_768_, 0, v___x_762_);
lean_ctor_set(v___x_768_, 1, v___x_764_);
lean_ctor_set(v___x_768_, 2, v___x_766_);
lean_ctor_set(v___x_768_, 3, v___x_767_);
v___x_769_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Data__Matrix__Mul______macroRules__term___u2b1d_u1d65____1___closed__11));
v___x_770_ = l_Lean_Syntax_node2(v___x_762_, v___x_769_, v___x_758_, v___x_760_);
v___x_771_ = l_Lean_Syntax_node2(v___x_762_, v___x_763_, v___x_768_, v___x_770_);
v___x_772_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_772_, 0, v___x_771_);
lean_ctor_set(v___x_772_, 1, v_a_749_);
return v___x_772_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Matrix___aux__Mathlib__Data__Matrix__Mul______macroRules__Matrix__term___u1d65_x2a____1___boxed(lean_object* v_x_773_, lean_object* v_a_774_, lean_object* v_a_775_){
_start:
{
lean_object* v_res_776_; 
v_res_776_ = lp_mathlib_Matrix___aux__Mathlib__Data__Matrix__Mul______macroRules__Matrix__term___u1d65_x2a____1(v_x_773_, v_a_774_, v_a_775_);
lean_dec_ref(v_a_774_);
return v_res_776_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Matrix___aux__Mathlib__Data__Matrix__Mul______unexpand__Matrix__vecMul__1(lean_object* v_x_777_, lean_object* v_a_778_, lean_object* v_a_779_){
_start:
{
lean_object* v___x_780_; uint8_t v___x_781_; 
v___x_780_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Data__Matrix__Mul______macroRules__term___u2b1d_u1d65____1___closed__4));
lean_inc(v_x_777_);
v___x_781_ = l_Lean_Syntax_isOfKind(v_x_777_, v___x_780_);
if (v___x_781_ == 0)
{
lean_object* v___x_782_; lean_object* v___x_783_; 
lean_dec(v_x_777_);
v___x_782_ = lean_box(0);
v___x_783_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_783_, 0, v___x_782_);
lean_ctor_set(v___x_783_, 1, v_a_779_);
return v___x_783_;
}
else
{
lean_object* v___x_784_; lean_object* v___x_785_; lean_object* v___x_786_; uint8_t v___x_787_; 
v___x_784_ = lean_unsigned_to_nat(0u);
v___x_785_ = l_Lean_Syntax_getArg(v_x_777_, v___x_784_);
v___x_786_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Data__Matrix__Mul______unexpand__dotProduct__1___closed__1));
lean_inc(v___x_785_);
v___x_787_ = l_Lean_Syntax_isOfKind(v___x_785_, v___x_786_);
if (v___x_787_ == 0)
{
lean_object* v___x_788_; lean_object* v___x_789_; 
lean_dec(v___x_785_);
lean_dec(v_x_777_);
v___x_788_ = lean_box(0);
v___x_789_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_789_, 0, v___x_788_);
lean_ctor_set(v___x_789_, 1, v_a_779_);
return v___x_789_;
}
else
{
lean_object* v___x_790_; lean_object* v___x_791_; lean_object* v___x_792_; uint8_t v___x_793_; 
v___x_790_ = lean_unsigned_to_nat(1u);
v___x_791_ = l_Lean_Syntax_getArg(v_x_777_, v___x_790_);
lean_dec(v_x_777_);
v___x_792_ = lean_unsigned_to_nat(2u);
lean_inc(v___x_791_);
v___x_793_ = l_Lean_Syntax_matchesNull(v___x_791_, v___x_792_);
if (v___x_793_ == 0)
{
lean_object* v___x_794_; lean_object* v___x_795_; 
lean_dec(v___x_791_);
lean_dec(v___x_785_);
v___x_794_ = lean_box(0);
v___x_795_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_795_, 0, v___x_794_);
lean_ctor_set(v___x_795_, 1, v_a_779_);
return v___x_795_;
}
else
{
lean_object* v___x_796_; lean_object* v___x_797_; lean_object* v_ref_798_; uint8_t v___x_799_; lean_object* v___x_800_; lean_object* v___x_801_; lean_object* v___x_802_; lean_object* v___x_803_; lean_object* v___x_804_; lean_object* v___x_805_; 
v___x_796_ = l_Lean_Syntax_getArg(v___x_791_, v___x_784_);
v___x_797_ = l_Lean_Syntax_getArg(v___x_791_, v___x_790_);
lean_dec(v___x_791_);
v_ref_798_ = l_Lean_replaceRef(v___x_785_, v_a_778_);
lean_dec(v___x_785_);
v___x_799_ = 0;
v___x_800_ = l_Lean_SourceInfo_fromRef(v_ref_798_, v___x_799_);
lean_dec(v_ref_798_);
v___x_801_ = ((lean_object*)(lp_mathlib_Matrix_term___u1d65_x2a___00__closed__1));
v___x_802_ = ((lean_object*)(lp_mathlib_Matrix_term___u1d65_x2a___00__closed__2));
lean_inc(v___x_800_);
v___x_803_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_803_, 0, v___x_800_);
lean_ctor_set(v___x_803_, 1, v___x_802_);
v___x_804_ = l_Lean_Syntax_node3(v___x_800_, v___x_801_, v___x_796_, v___x_803_, v___x_797_);
v___x_805_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_805_, 0, v___x_804_);
lean_ctor_set(v___x_805_, 1, v_a_779_);
return v___x_805_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Matrix___aux__Mathlib__Data__Matrix__Mul______unexpand__Matrix__vecMul__1___boxed(lean_object* v_x_806_, lean_object* v_a_807_, lean_object* v_a_808_){
_start:
{
lean_object* v_res_809_; 
v_res_809_ = lp_mathlib_Matrix___aux__Mathlib__Data__Matrix__Mul______unexpand__Matrix__vecMul__1(v_x_806_, v_a_807_, v_a_808_);
lean_dec(v_a_807_);
return v_res_809_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Matrix_mulVec_addMonoidHomLeft___redArg___lam__0(lean_object* v_inst_810_, lean_object* v_inst_811_, lean_object* v_v_812_, lean_object* v_M_813_, lean_object* v___y_814_){
_start:
{
lean_object* v___x_815_; 
v___x_815_ = lp_mathlib_Matrix_mulVec___redArg(v_inst_810_, v_inst_811_, v_M_813_, v_v_812_, v___y_814_);
return v___x_815_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Matrix_mulVec_addMonoidHomLeft___redArg(lean_object* v_inst_816_, lean_object* v_inst_817_, lean_object* v_v_818_){
_start:
{
lean_object* v___f_819_; 
v___f_819_ = lean_alloc_closure((void*)(lp_mathlib_Matrix_mulVec_addMonoidHomLeft___redArg___lam__0), 5, 3);
lean_closure_set(v___f_819_, 0, v_inst_816_);
lean_closure_set(v___f_819_, 1, v_inst_817_);
lean_closure_set(v___f_819_, 2, v_v_818_);
return v___f_819_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Matrix_mulVec_addMonoidHomLeft(lean_object* v_m_820_, lean_object* v_n_821_, lean_object* v_00_u03b1_822_, lean_object* v_inst_823_, lean_object* v_inst_824_, lean_object* v_v_825_){
_start:
{
lean_object* v___f_826_; 
v___f_826_ = lean_alloc_closure((void*)(lp_mathlib_Matrix_mulVec_addMonoidHomLeft___redArg___lam__0), 5, 3);
lean_closure_set(v___f_826_, 0, v_inst_823_);
lean_closure_set(v___f_826_, 1, v_inst_824_);
lean_closure_set(v___f_826_, 2, v_v_825_);
return v___f_826_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_BigOperators_GroupWithZero_Action(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_BigOperators_Ring_Finset(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Regular_Basic(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Ring_Subsemiring_Defs(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Data_Fintype_BigOperators(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Data_Matrix_Diagonal(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Order_BigOperators_Group_Finset(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_Data_Matrix_Mul(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_BigOperators_GroupWithZero_Action(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_BigOperators_Ring_Finset(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Regular_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Ring_Subsemiring_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Data_Fintype_BigOperators(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Data_Matrix_Diagonal(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Order_BigOperators_Group_Finset(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_Data_Matrix_Mul(uint8_t builtin) {
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
lean_object* initialize_mathlib_Mathlib_Algebra_BigOperators_GroupWithZero_Action(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_BigOperators_Ring_Finset(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Regular_Basic(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Ring_Subsemiring_Defs(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Data_Fintype_BigOperators(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Data_Matrix_Diagonal(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Order_BigOperators_Group_Finset(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Data_Matrix_Mul(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_BigOperators_GroupWithZero_Action(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_BigOperators_Ring_Finset(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Regular_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Ring_Subsemiring_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_Fintype_BigOperators(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_Matrix_Diagonal(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Order_BigOperators_Group_Finset(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Data_Matrix_Mul(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_Data_Matrix_Mul(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_Data_Matrix_Mul(builtin);
}
#ifdef __cplusplus
}
#endif
