// Lean compiler output
// Module: Mathlib.LinearAlgebra.Matrix.Defs
// Imports: public import Init public meta import Init public import Batteries.Data.Fin.Lemmas public import Mathlib.Algebra.Module.Pi public import Mathlib.Basic.Nontrivial.Basic public import Mathlib.Tactic.CrossRefAttribute public import Mathlib.Tactic.Attr.Core
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
lean_object* lp_mathlib_AddMonoid_toAddZeroClass___redArg(lean_object*);
lean_object* lp_mathlib_AddZeroClass_toAddZero___redArg(lean_object*);
lean_object* lp_mathlib_NSMul_toSMul___redArg___lam__0(lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Name_mkStr2(lean_object*, lean_object*);
lean_object* l_Lean_Name_mkStr1(lean_object*);
lean_object* lp_mathlib_Equiv_refl(lean_object*);
lean_object* lp_mathlib_SubNegZeroMonoid_toNegZeroClass___redArg(lean_object*);
lean_object* lp_mathlib_ZSMul_toSMul___redArg___lam__0(lean_object*, lean_object*, lean_object*);
lean_object* lean_nat_add(lean_object*, lean_object*);
lean_object* l_id___boxed(lean_object*, lean_object*);
lean_object* l_Fin_castAdd___boxed(lean_object*, lean_object*, lean_object*);
lean_object* l_Fin_natAdd___boxed(lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Name_mkStr4(lean_object*, lean_object*, lean_object*, lean_object*);
uint8_t l_Lean_Syntax_isOfKind(lean_object*, lean_object*);
lean_object* l_Lean_Syntax_getArg(lean_object*, lean_object*);
uint8_t l_Lean_Syntax_matchesNull(lean_object*, lean_object*);
lean_object* l_Lean_replaceRef(lean_object*, lean_object*);
lean_object* l_Lean_SourceInfo_fromRef(lean_object*, uint8_t);
lean_object* l_Lean_Syntax_node2(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_String_toRawSubstring_x27(lean_object*);
lean_object* l_Lean_addMacroScope(lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Syntax_node1(lean_object*, lean_object*, lean_object*);
lean_object* lp_batteries_Fin_mkDivMod___redArg(lean_object*, lean_object*, lean_object*);
lean_object* lean_array_fget_borrowed(lean_object*, lean_object*);
lean_object* lp_mathlib_Equiv_symm___redArg(lean_object*);
static lean_once_cell_t lp_mathlib_Matrix_of___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Matrix_of___closed__0;
LEAN_EXPORT lean_object* lp_mathlib_Matrix_of(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Matrix_ofArray___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Matrix_ofArray___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Matrix_ofArray(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Matrix_ofArray___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Matrix_map___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Matrix_map___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Matrix_map(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Matrix_transpose___redArg___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Matrix_transpose___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Matrix_transpose(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Matrix_term___u1d40___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "Matrix"};
static const lean_object* lp_mathlib_Matrix_term___u1d40___closed__0 = (const lean_object*)&lp_mathlib_Matrix_term___u1d40___closed__0_value;
static const lean_string_object lp_mathlib_Matrix_term___u1d40___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 6, .m_data = "term_ᵀ"};
static const lean_object* lp_mathlib_Matrix_term___u1d40___closed__1 = (const lean_object*)&lp_mathlib_Matrix_term___u1d40___closed__1_value;
static const lean_ctor_object lp_mathlib_Matrix_term___u1d40___closed__2_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Matrix_term___u1d40___closed__0_value),LEAN_SCALAR_PTR_LITERAL(214, 58, 148, 51, 223, 251, 16, 41)}};
static const lean_ctor_object lp_mathlib_Matrix_term___u1d40___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Matrix_term___u1d40___closed__2_value_aux_0),((lean_object*)&lp_mathlib_Matrix_term___u1d40___closed__1_value),LEAN_SCALAR_PTR_LITERAL(146, 67, 52, 19, 246, 239, 73, 49)}};
static const lean_object* lp_mathlib_Matrix_term___u1d40___closed__2 = (const lean_object*)&lp_mathlib_Matrix_term___u1d40___closed__2_value;
static const lean_string_object lp_mathlib_Matrix_term___u1d40___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 1, .m_data = "ᵀ"};
static const lean_object* lp_mathlib_Matrix_term___u1d40___closed__3 = (const lean_object*)&lp_mathlib_Matrix_term___u1d40___closed__3_value;
static const lean_ctor_object lp_mathlib_Matrix_term___u1d40___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 5}, .m_objs = {((lean_object*)&lp_mathlib_Matrix_term___u1d40___closed__3_value)}};
static const lean_object* lp_mathlib_Matrix_term___u1d40___closed__4 = (const lean_object*)&lp_mathlib_Matrix_term___u1d40___closed__4_value;
static const lean_ctor_object lp_mathlib_Matrix_term___u1d40___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*4 + 0, .m_other = 4, .m_tag = 4}, .m_objs = {((lean_object*)&lp_mathlib_Matrix_term___u1d40___closed__2_value),((lean_object*)(((size_t)(1024) << 1) | 1)),((lean_object*)(((size_t)(1024) << 1) | 1)),((lean_object*)&lp_mathlib_Matrix_term___u1d40___closed__4_value)}};
static const lean_object* lp_mathlib_Matrix_term___u1d40___closed__5 = (const lean_object*)&lp_mathlib_Matrix_term___u1d40___closed__5_value;
LEAN_EXPORT const lean_object* lp_mathlib_Matrix_term___u1d40 = (const lean_object*)&lp_mathlib_Matrix_term___u1d40___closed__5_value;
static const lean_string_object lp_mathlib_Matrix___aux__Mathlib__LinearAlgebra__Matrix__Defs______macroRules__Matrix__term___u1d40__1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "Lean"};
static const lean_object* lp_mathlib_Matrix___aux__Mathlib__LinearAlgebra__Matrix__Defs______macroRules__Matrix__term___u1d40__1___closed__0 = (const lean_object*)&lp_mathlib_Matrix___aux__Mathlib__LinearAlgebra__Matrix__Defs______macroRules__Matrix__term___u1d40__1___closed__0_value;
static const lean_string_object lp_mathlib_Matrix___aux__Mathlib__LinearAlgebra__Matrix__Defs______macroRules__Matrix__term___u1d40__1___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "Parser"};
static const lean_object* lp_mathlib_Matrix___aux__Mathlib__LinearAlgebra__Matrix__Defs______macroRules__Matrix__term___u1d40__1___closed__1 = (const lean_object*)&lp_mathlib_Matrix___aux__Mathlib__LinearAlgebra__Matrix__Defs______macroRules__Matrix__term___u1d40__1___closed__1_value;
static const lean_string_object lp_mathlib_Matrix___aux__Mathlib__LinearAlgebra__Matrix__Defs______macroRules__Matrix__term___u1d40__1___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "Term"};
static const lean_object* lp_mathlib_Matrix___aux__Mathlib__LinearAlgebra__Matrix__Defs______macroRules__Matrix__term___u1d40__1___closed__2 = (const lean_object*)&lp_mathlib_Matrix___aux__Mathlib__LinearAlgebra__Matrix__Defs______macroRules__Matrix__term___u1d40__1___closed__2_value;
static const lean_string_object lp_mathlib_Matrix___aux__Mathlib__LinearAlgebra__Matrix__Defs______macroRules__Matrix__term___u1d40__1___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = "app"};
static const lean_object* lp_mathlib_Matrix___aux__Mathlib__LinearAlgebra__Matrix__Defs______macroRules__Matrix__term___u1d40__1___closed__3 = (const lean_object*)&lp_mathlib_Matrix___aux__Mathlib__LinearAlgebra__Matrix__Defs______macroRules__Matrix__term___u1d40__1___closed__3_value;
static const lean_ctor_object lp_mathlib_Matrix___aux__Mathlib__LinearAlgebra__Matrix__Defs______macroRules__Matrix__term___u1d40__1___closed__4_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Matrix___aux__Mathlib__LinearAlgebra__Matrix__Defs______macroRules__Matrix__term___u1d40__1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib_Matrix___aux__Mathlib__LinearAlgebra__Matrix__Defs______macroRules__Matrix__term___u1d40__1___closed__4_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Matrix___aux__Mathlib__LinearAlgebra__Matrix__Defs______macroRules__Matrix__term___u1d40__1___closed__4_value_aux_0),((lean_object*)&lp_mathlib_Matrix___aux__Mathlib__LinearAlgebra__Matrix__Defs______macroRules__Matrix__term___u1d40__1___closed__1_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib_Matrix___aux__Mathlib__LinearAlgebra__Matrix__Defs______macroRules__Matrix__term___u1d40__1___closed__4_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Matrix___aux__Mathlib__LinearAlgebra__Matrix__Defs______macroRules__Matrix__term___u1d40__1___closed__4_value_aux_1),((lean_object*)&lp_mathlib_Matrix___aux__Mathlib__LinearAlgebra__Matrix__Defs______macroRules__Matrix__term___u1d40__1___closed__2_value),LEAN_SCALAR_PTR_LITERAL(75, 170, 162, 138, 136, 204, 251, 229)}};
static const lean_ctor_object lp_mathlib_Matrix___aux__Mathlib__LinearAlgebra__Matrix__Defs______macroRules__Matrix__term___u1d40__1___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Matrix___aux__Mathlib__LinearAlgebra__Matrix__Defs______macroRules__Matrix__term___u1d40__1___closed__4_value_aux_2),((lean_object*)&lp_mathlib_Matrix___aux__Mathlib__LinearAlgebra__Matrix__Defs______macroRules__Matrix__term___u1d40__1___closed__3_value),LEAN_SCALAR_PTR_LITERAL(69, 118, 10, 41, 220, 156, 243, 179)}};
static const lean_object* lp_mathlib_Matrix___aux__Mathlib__LinearAlgebra__Matrix__Defs______macroRules__Matrix__term___u1d40__1___closed__4 = (const lean_object*)&lp_mathlib_Matrix___aux__Mathlib__LinearAlgebra__Matrix__Defs______macroRules__Matrix__term___u1d40__1___closed__4_value;
static const lean_string_object lp_mathlib_Matrix___aux__Mathlib__LinearAlgebra__Matrix__Defs______macroRules__Matrix__term___u1d40__1___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 17, .m_capacity = 17, .m_length = 16, .m_data = "Matrix.transpose"};
static const lean_object* lp_mathlib_Matrix___aux__Mathlib__LinearAlgebra__Matrix__Defs______macroRules__Matrix__term___u1d40__1___closed__5 = (const lean_object*)&lp_mathlib_Matrix___aux__Mathlib__LinearAlgebra__Matrix__Defs______macroRules__Matrix__term___u1d40__1___closed__5_value;
static lean_once_cell_t lp_mathlib_Matrix___aux__Mathlib__LinearAlgebra__Matrix__Defs______macroRules__Matrix__term___u1d40__1___closed__6_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Matrix___aux__Mathlib__LinearAlgebra__Matrix__Defs______macroRules__Matrix__term___u1d40__1___closed__6;
static const lean_string_object lp_mathlib_Matrix___aux__Mathlib__LinearAlgebra__Matrix__Defs______macroRules__Matrix__term___u1d40__1___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 9, .m_data = "transpose"};
static const lean_object* lp_mathlib_Matrix___aux__Mathlib__LinearAlgebra__Matrix__Defs______macroRules__Matrix__term___u1d40__1___closed__7 = (const lean_object*)&lp_mathlib_Matrix___aux__Mathlib__LinearAlgebra__Matrix__Defs______macroRules__Matrix__term___u1d40__1___closed__7_value;
static const lean_ctor_object lp_mathlib_Matrix___aux__Mathlib__LinearAlgebra__Matrix__Defs______macroRules__Matrix__term___u1d40__1___closed__8_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Matrix_term___u1d40___closed__0_value),LEAN_SCALAR_PTR_LITERAL(214, 58, 148, 51, 223, 251, 16, 41)}};
static const lean_ctor_object lp_mathlib_Matrix___aux__Mathlib__LinearAlgebra__Matrix__Defs______macroRules__Matrix__term___u1d40__1___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Matrix___aux__Mathlib__LinearAlgebra__Matrix__Defs______macroRules__Matrix__term___u1d40__1___closed__8_value_aux_0),((lean_object*)&lp_mathlib_Matrix___aux__Mathlib__LinearAlgebra__Matrix__Defs______macroRules__Matrix__term___u1d40__1___closed__7_value),LEAN_SCALAR_PTR_LITERAL(90, 37, 134, 78, 33, 8, 231, 154)}};
static const lean_object* lp_mathlib_Matrix___aux__Mathlib__LinearAlgebra__Matrix__Defs______macroRules__Matrix__term___u1d40__1___closed__8 = (const lean_object*)&lp_mathlib_Matrix___aux__Mathlib__LinearAlgebra__Matrix__Defs______macroRules__Matrix__term___u1d40__1___closed__8_value;
static const lean_ctor_object lp_mathlib_Matrix___aux__Mathlib__LinearAlgebra__Matrix__Defs______macroRules__Matrix__term___u1d40__1___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Matrix___aux__Mathlib__LinearAlgebra__Matrix__Defs______macroRules__Matrix__term___u1d40__1___closed__8_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_Matrix___aux__Mathlib__LinearAlgebra__Matrix__Defs______macroRules__Matrix__term___u1d40__1___closed__9 = (const lean_object*)&lp_mathlib_Matrix___aux__Mathlib__LinearAlgebra__Matrix__Defs______macroRules__Matrix__term___u1d40__1___closed__9_value;
static const lean_ctor_object lp_mathlib_Matrix___aux__Mathlib__LinearAlgebra__Matrix__Defs______macroRules__Matrix__term___u1d40__1___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Matrix___aux__Mathlib__LinearAlgebra__Matrix__Defs______macroRules__Matrix__term___u1d40__1___closed__9_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_Matrix___aux__Mathlib__LinearAlgebra__Matrix__Defs______macroRules__Matrix__term___u1d40__1___closed__10 = (const lean_object*)&lp_mathlib_Matrix___aux__Mathlib__LinearAlgebra__Matrix__Defs______macroRules__Matrix__term___u1d40__1___closed__10_value;
static const lean_string_object lp_mathlib_Matrix___aux__Mathlib__LinearAlgebra__Matrix__Defs______macroRules__Matrix__term___u1d40__1___closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "null"};
static const lean_object* lp_mathlib_Matrix___aux__Mathlib__LinearAlgebra__Matrix__Defs______macroRules__Matrix__term___u1d40__1___closed__11 = (const lean_object*)&lp_mathlib_Matrix___aux__Mathlib__LinearAlgebra__Matrix__Defs______macroRules__Matrix__term___u1d40__1___closed__11_value;
static const lean_ctor_object lp_mathlib_Matrix___aux__Mathlib__LinearAlgebra__Matrix__Defs______macroRules__Matrix__term___u1d40__1___closed__12_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Matrix___aux__Mathlib__LinearAlgebra__Matrix__Defs______macroRules__Matrix__term___u1d40__1___closed__11_value),LEAN_SCALAR_PTR_LITERAL(24, 58, 49, 223, 146, 207, 197, 136)}};
static const lean_object* lp_mathlib_Matrix___aux__Mathlib__LinearAlgebra__Matrix__Defs______macroRules__Matrix__term___u1d40__1___closed__12 = (const lean_object*)&lp_mathlib_Matrix___aux__Mathlib__LinearAlgebra__Matrix__Defs______macroRules__Matrix__term___u1d40__1___closed__12_value;
LEAN_EXPORT lean_object* lp_mathlib_Matrix___aux__Mathlib__LinearAlgebra__Matrix__Defs______macroRules__Matrix__term___u1d40__1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Matrix___aux__Mathlib__LinearAlgebra__Matrix__Defs______macroRules__Matrix__term___u1d40__1___boxed(lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Matrix___aux__Mathlib__LinearAlgebra__Matrix__Defs______unexpand__Matrix__transpose__1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "ident"};
static const lean_object* lp_mathlib_Matrix___aux__Mathlib__LinearAlgebra__Matrix__Defs______unexpand__Matrix__transpose__1___closed__0 = (const lean_object*)&lp_mathlib_Matrix___aux__Mathlib__LinearAlgebra__Matrix__Defs______unexpand__Matrix__transpose__1___closed__0_value;
static const lean_ctor_object lp_mathlib_Matrix___aux__Mathlib__LinearAlgebra__Matrix__Defs______unexpand__Matrix__transpose__1___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Matrix___aux__Mathlib__LinearAlgebra__Matrix__Defs______unexpand__Matrix__transpose__1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(52, 159, 208, 51, 14, 60, 6, 71)}};
static const lean_object* lp_mathlib_Matrix___aux__Mathlib__LinearAlgebra__Matrix__Defs______unexpand__Matrix__transpose__1___closed__1 = (const lean_object*)&lp_mathlib_Matrix___aux__Mathlib__LinearAlgebra__Matrix__Defs______unexpand__Matrix__transpose__1___closed__1_value;
LEAN_EXPORT lean_object* lp_mathlib_Matrix___aux__Mathlib__LinearAlgebra__Matrix__Defs______unexpand__Matrix__transpose__1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Matrix___aux__Mathlib__LinearAlgebra__Matrix__Defs______unexpand__Matrix__transpose__1___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Matrix_inhabited___aux__1___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Matrix_inhabited___aux__1___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Matrix_inhabited___aux__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Matrix_inhabited___aux__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Matrix_inhabited___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Matrix_inhabited(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Matrix_add___aux__1___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Matrix_add___aux__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Matrix_add___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Matrix_add(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Matrix_smul___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Matrix_smul___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Matrix_smul(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Matrix_addSemigroup___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Matrix_addSemigroup(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Matrix_addCommSemigroup___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Matrix_addCommSemigroup(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Matrix_zero___aux__1___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Matrix_zero___aux__1___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Matrix_zero___aux__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Matrix_zero___aux__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Matrix_zero___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Matrix_zero(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Matrix_addZeroClass___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Matrix_addZeroClass(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Matrix_addMonoid___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Matrix_addMonoid(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Matrix_addCommMonoid___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Matrix_addCommMonoid(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Matrix_neg___aux__1___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Matrix_neg___aux__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Matrix_neg___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Matrix_neg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Matrix_involutiveNeg___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Matrix_involutiveNeg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Matrix_sub___aux__1___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Matrix_sub___aux__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Matrix_sub___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Matrix_sub(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Matrix_addGroup___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Matrix_addGroup(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Matrix_addCommGroup___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Matrix_addCommGroup(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Matrix_unique___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Matrix_unique(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Matrix_mulAction___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Matrix_mulAction(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Matrix_mulAction___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Matrix_distribMulAction___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Matrix_distribMulAction(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Matrix_distribMulAction___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Matrix_module___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Matrix_module(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Matrix_module___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Matrix_instAddCommMagma___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Matrix_instAddCommMagma(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Matrix_instAddLeftCancelSemigroup___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Matrix_instAddLeftCancelSemigroup(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Matrix_instAddRightCancelSemigroup___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Matrix_instAddRightCancelSemigroup(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Matrix_instAddLeftCancelMonoid___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Matrix_instAddLeftCancelMonoid(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Matrix_instAddRightCancelMonoid___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Matrix_instAddRightCancelMonoid(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Matrix_instAddCancelMonoid___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Matrix_instAddCancelMonoid(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Matrix_instAddCancelCommMonoid___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Matrix_instAddCancelCommMonoid(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Matrix_ofAddEquiv(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Matrix_ofAddEquiv___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Matrix_submatrix___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Matrix_submatrix___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Matrix_submatrix(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Matrix_reindex___redArg___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Matrix_reindex___redArg___lam__2(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Matrix_reindex___redArg___lam__1(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Matrix_reindex___redArg___lam__3(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Matrix_reindex___redArg___lam__4(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Matrix_reindex___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Matrix_reindex(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_Matrix_subLeft___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*1, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_id___boxed, .m_arity = 2, .m_num_fixed = 1, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1))} };
static const lean_object* lp_mathlib_Matrix_subLeft___redArg___closed__0 = (const lean_object*)&lp_mathlib_Matrix_subLeft___redArg___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_Matrix_subLeft___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Matrix_subLeft(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Matrix_subLeft___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Matrix_subRight___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Matrix_subRight(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Matrix_subRight___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Matrix_subUp___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Matrix_subUp(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Matrix_subUp___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Matrix_subDown___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Matrix_subDown(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Matrix_subDown___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Matrix_subUpRight___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Matrix_subUpRight(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Matrix_subDownRight___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Matrix_subDownRight(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Matrix_subUpLeft___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Matrix_subUpLeft(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Matrix_subDownLeft___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Matrix_subDownLeft(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Matrix_row___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Matrix_row(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Matrix_col___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Matrix_col(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_object* _init_lp_mathlib_Matrix_of___closed__0(void){
_start:
{
lean_object* v___x_1_; 
v___x_1_ = lp_mathlib_Equiv_refl(lean_box(0));
return v___x_1_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Matrix_of(lean_object* v_m_2_, lean_object* v_n_3_, lean_object* v_00_u03b1_4_){
_start:
{
lean_object* v___x_5_; 
v___x_5_ = lean_obj_once(&lp_mathlib_Matrix_of___closed__0, &lp_mathlib_Matrix_of___closed__0_once, _init_lp_mathlib_Matrix_of___closed__0);
return v___x_5_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Matrix_ofArray___redArg(lean_object* v_n_6_, lean_object* v_A_7_, lean_object* v_i_8_, lean_object* v_j_9_){
_start:
{
lean_object* v___x_10_; lean_object* v___x_11_; 
v___x_10_ = lp_batteries_Fin_mkDivMod___redArg(v_n_6_, v_i_8_, v_j_9_);
v___x_11_ = lean_array_fget_borrowed(v_A_7_, v___x_10_);
lean_dec(v___x_10_);
lean_inc(v___x_11_);
return v___x_11_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Matrix_ofArray___redArg___boxed(lean_object* v_n_12_, lean_object* v_A_13_, lean_object* v_i_14_, lean_object* v_j_15_){
_start:
{
lean_object* v_res_16_; 
v_res_16_ = lp_mathlib_Matrix_ofArray___redArg(v_n_12_, v_A_13_, v_i_14_, v_j_15_);
lean_dec(v_j_15_);
lean_dec(v_i_14_);
lean_dec_ref(v_A_13_);
lean_dec(v_n_12_);
return v_res_16_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Matrix_ofArray(lean_object* v_R_17_, lean_object* v_m_18_, lean_object* v_n_19_, lean_object* v_A_20_, lean_object* v_hA_21_, lean_object* v_i_22_, lean_object* v_j_23_){
_start:
{
lean_object* v___x_24_; 
v___x_24_ = lp_mathlib_Matrix_ofArray___redArg(v_n_19_, v_A_20_, v_i_22_, v_j_23_);
return v___x_24_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Matrix_ofArray___boxed(lean_object* v_R_25_, lean_object* v_m_26_, lean_object* v_n_27_, lean_object* v_A_28_, lean_object* v_hA_29_, lean_object* v_i_30_, lean_object* v_j_31_){
_start:
{
lean_object* v_res_32_; 
v_res_32_ = lp_mathlib_Matrix_ofArray(v_R_25_, v_m_26_, v_n_27_, v_A_28_, v_hA_29_, v_i_30_, v_j_31_);
lean_dec(v_j_31_);
lean_dec(v_i_30_);
lean_dec_ref(v_A_28_);
lean_dec(v_n_27_);
lean_dec(v_m_26_);
return v_res_32_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Matrix_map___redArg___lam__0(lean_object* v_M_33_, lean_object* v_f_34_, lean_object* v_i_35_, lean_object* v_j_36_){
_start:
{
lean_object* v___x_37_; lean_object* v___x_38_; 
v___x_37_ = lean_apply_2(v_M_33_, v_i_35_, v_j_36_);
v___x_38_ = lean_apply_1(v_f_34_, v___x_37_);
return v___x_38_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Matrix_map___redArg(lean_object* v_M_39_, lean_object* v_f_40_, lean_object* v_a_41_, lean_object* v_a_42_){
_start:
{
lean_object* v___x_43_; lean_object* v_toFun_44_; lean_object* v___f_45_; lean_object* v___x_46_; 
v___x_43_ = lean_obj_once(&lp_mathlib_Matrix_of___closed__0, &lp_mathlib_Matrix_of___closed__0_once, _init_lp_mathlib_Matrix_of___closed__0);
v_toFun_44_ = lean_ctor_get(v___x_43_, 0);
v___f_45_ = lean_alloc_closure((void*)(lp_mathlib_Matrix_map___redArg___lam__0), 4, 2);
lean_closure_set(v___f_45_, 0, v_M_39_);
lean_closure_set(v___f_45_, 1, v_f_40_);
lean_inc(v_toFun_44_);
v___x_46_ = lean_apply_3(v_toFun_44_, v___f_45_, v_a_41_, v_a_42_);
return v___x_46_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Matrix_map(lean_object* v_m_47_, lean_object* v_n_48_, lean_object* v_00_u03b1_49_, lean_object* v_00_u03b2_50_, lean_object* v_M_51_, lean_object* v_f_52_, lean_object* v_a_53_, lean_object* v_a_54_){
_start:
{
lean_object* v___x_55_; 
v___x_55_ = lp_mathlib_Matrix_map___redArg(v_M_51_, v_f_52_, v_a_53_, v_a_54_);
return v___x_55_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Matrix_transpose___redArg___lam__0(lean_object* v_M_56_, lean_object* v_x_57_, lean_object* v_y_58_){
_start:
{
lean_object* v___x_59_; 
v___x_59_ = lean_apply_2(v_M_56_, v_y_58_, v_x_57_);
return v___x_59_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Matrix_transpose___redArg(lean_object* v_M_60_, lean_object* v_a_61_, lean_object* v_a_62_){
_start:
{
lean_object* v___x_63_; lean_object* v_toFun_64_; lean_object* v___f_65_; lean_object* v___x_66_; 
v___x_63_ = lean_obj_once(&lp_mathlib_Matrix_of___closed__0, &lp_mathlib_Matrix_of___closed__0_once, _init_lp_mathlib_Matrix_of___closed__0);
v_toFun_64_ = lean_ctor_get(v___x_63_, 0);
v___f_65_ = lean_alloc_closure((void*)(lp_mathlib_Matrix_transpose___redArg___lam__0), 3, 1);
lean_closure_set(v___f_65_, 0, v_M_60_);
lean_inc(v_toFun_64_);
v___x_66_ = lean_apply_3(v_toFun_64_, v___f_65_, v_a_61_, v_a_62_);
return v___x_66_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Matrix_transpose(lean_object* v_m_67_, lean_object* v_n_68_, lean_object* v_00_u03b1_69_, lean_object* v_M_70_, lean_object* v_a_71_, lean_object* v_a_72_){
_start:
{
lean_object* v___x_73_; 
v___x_73_ = lp_mathlib_Matrix_transpose___redArg(v_M_70_, v_a_71_, v_a_72_);
return v___x_73_;
}
}
static lean_object* _init_lp_mathlib_Matrix___aux__Mathlib__LinearAlgebra__Matrix__Defs______macroRules__Matrix__term___u1d40__1___closed__6(void){
_start:
{
lean_object* v___x_97_; lean_object* v___x_98_; 
v___x_97_ = ((lean_object*)(lp_mathlib_Matrix___aux__Mathlib__LinearAlgebra__Matrix__Defs______macroRules__Matrix__term___u1d40__1___closed__5));
v___x_98_ = l_String_toRawSubstring_x27(v___x_97_);
return v___x_98_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Matrix___aux__Mathlib__LinearAlgebra__Matrix__Defs______macroRules__Matrix__term___u1d40__1(lean_object* v_x_112_, lean_object* v_a_113_, lean_object* v_a_114_){
_start:
{
lean_object* v___x_115_; uint8_t v___x_116_; 
v___x_115_ = ((lean_object*)(lp_mathlib_Matrix_term___u1d40___closed__2));
lean_inc(v_x_112_);
v___x_116_ = l_Lean_Syntax_isOfKind(v_x_112_, v___x_115_);
if (v___x_116_ == 0)
{
lean_object* v___x_117_; lean_object* v___x_118_; 
lean_dec(v_x_112_);
v___x_117_ = lean_box(1);
v___x_118_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_118_, 0, v___x_117_);
lean_ctor_set(v___x_118_, 1, v_a_114_);
return v___x_118_;
}
else
{
lean_object* v_quotContext_119_; lean_object* v_currMacroScope_120_; lean_object* v_ref_121_; lean_object* v___x_122_; lean_object* v___x_123_; uint8_t v___x_124_; lean_object* v___x_125_; lean_object* v___x_126_; lean_object* v___x_127_; lean_object* v___x_128_; lean_object* v___x_129_; lean_object* v___x_130_; lean_object* v___x_131_; lean_object* v___x_132_; lean_object* v___x_133_; lean_object* v___x_134_; lean_object* v___x_135_; 
v_quotContext_119_ = lean_ctor_get(v_a_113_, 1);
v_currMacroScope_120_ = lean_ctor_get(v_a_113_, 2);
v_ref_121_ = lean_ctor_get(v_a_113_, 5);
v___x_122_ = lean_unsigned_to_nat(0u);
v___x_123_ = l_Lean_Syntax_getArg(v_x_112_, v___x_122_);
lean_dec(v_x_112_);
v___x_124_ = 0;
v___x_125_ = l_Lean_SourceInfo_fromRef(v_ref_121_, v___x_124_);
v___x_126_ = ((lean_object*)(lp_mathlib_Matrix___aux__Mathlib__LinearAlgebra__Matrix__Defs______macroRules__Matrix__term___u1d40__1___closed__4));
v___x_127_ = lean_obj_once(&lp_mathlib_Matrix___aux__Mathlib__LinearAlgebra__Matrix__Defs______macroRules__Matrix__term___u1d40__1___closed__6, &lp_mathlib_Matrix___aux__Mathlib__LinearAlgebra__Matrix__Defs______macroRules__Matrix__term___u1d40__1___closed__6_once, _init_lp_mathlib_Matrix___aux__Mathlib__LinearAlgebra__Matrix__Defs______macroRules__Matrix__term___u1d40__1___closed__6);
v___x_128_ = ((lean_object*)(lp_mathlib_Matrix___aux__Mathlib__LinearAlgebra__Matrix__Defs______macroRules__Matrix__term___u1d40__1___closed__8));
lean_inc(v_currMacroScope_120_);
lean_inc(v_quotContext_119_);
v___x_129_ = l_Lean_addMacroScope(v_quotContext_119_, v___x_128_, v_currMacroScope_120_);
v___x_130_ = ((lean_object*)(lp_mathlib_Matrix___aux__Mathlib__LinearAlgebra__Matrix__Defs______macroRules__Matrix__term___u1d40__1___closed__10));
lean_inc_n(v___x_125_, 2);
v___x_131_ = lean_alloc_ctor(3, 4, 0);
lean_ctor_set(v___x_131_, 0, v___x_125_);
lean_ctor_set(v___x_131_, 1, v___x_127_);
lean_ctor_set(v___x_131_, 2, v___x_129_);
lean_ctor_set(v___x_131_, 3, v___x_130_);
v___x_132_ = ((lean_object*)(lp_mathlib_Matrix___aux__Mathlib__LinearAlgebra__Matrix__Defs______macroRules__Matrix__term___u1d40__1___closed__12));
v___x_133_ = l_Lean_Syntax_node1(v___x_125_, v___x_132_, v___x_123_);
v___x_134_ = l_Lean_Syntax_node2(v___x_125_, v___x_126_, v___x_131_, v___x_133_);
v___x_135_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_135_, 0, v___x_134_);
lean_ctor_set(v___x_135_, 1, v_a_114_);
return v___x_135_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Matrix___aux__Mathlib__LinearAlgebra__Matrix__Defs______macroRules__Matrix__term___u1d40__1___boxed(lean_object* v_x_136_, lean_object* v_a_137_, lean_object* v_a_138_){
_start:
{
lean_object* v_res_139_; 
v_res_139_ = lp_mathlib_Matrix___aux__Mathlib__LinearAlgebra__Matrix__Defs______macroRules__Matrix__term___u1d40__1(v_x_136_, v_a_137_, v_a_138_);
lean_dec_ref(v_a_137_);
return v_res_139_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Matrix___aux__Mathlib__LinearAlgebra__Matrix__Defs______unexpand__Matrix__transpose__1(lean_object* v_x_143_, lean_object* v_a_144_, lean_object* v_a_145_){
_start:
{
lean_object* v___x_146_; uint8_t v___x_147_; 
v___x_146_ = ((lean_object*)(lp_mathlib_Matrix___aux__Mathlib__LinearAlgebra__Matrix__Defs______macroRules__Matrix__term___u1d40__1___closed__4));
lean_inc(v_x_143_);
v___x_147_ = l_Lean_Syntax_isOfKind(v_x_143_, v___x_146_);
if (v___x_147_ == 0)
{
lean_object* v___x_148_; lean_object* v___x_149_; 
lean_dec(v_x_143_);
v___x_148_ = lean_box(0);
v___x_149_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_149_, 0, v___x_148_);
lean_ctor_set(v___x_149_, 1, v_a_145_);
return v___x_149_;
}
else
{
lean_object* v___x_150_; lean_object* v___x_151_; lean_object* v___x_152_; uint8_t v___x_153_; 
v___x_150_ = lean_unsigned_to_nat(0u);
v___x_151_ = l_Lean_Syntax_getArg(v_x_143_, v___x_150_);
v___x_152_ = ((lean_object*)(lp_mathlib_Matrix___aux__Mathlib__LinearAlgebra__Matrix__Defs______unexpand__Matrix__transpose__1___closed__1));
lean_inc(v___x_151_);
v___x_153_ = l_Lean_Syntax_isOfKind(v___x_151_, v___x_152_);
if (v___x_153_ == 0)
{
lean_object* v___x_154_; lean_object* v___x_155_; 
lean_dec(v___x_151_);
lean_dec(v_x_143_);
v___x_154_ = lean_box(0);
v___x_155_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_155_, 0, v___x_154_);
lean_ctor_set(v___x_155_, 1, v_a_145_);
return v___x_155_;
}
else
{
lean_object* v___x_156_; lean_object* v___x_157_; uint8_t v___x_158_; 
v___x_156_ = lean_unsigned_to_nat(1u);
v___x_157_ = l_Lean_Syntax_getArg(v_x_143_, v___x_156_);
lean_dec(v_x_143_);
lean_inc(v___x_157_);
v___x_158_ = l_Lean_Syntax_matchesNull(v___x_157_, v___x_156_);
if (v___x_158_ == 0)
{
lean_object* v___x_159_; lean_object* v___x_160_; 
lean_dec(v___x_157_);
lean_dec(v___x_151_);
v___x_159_ = lean_box(0);
v___x_160_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_160_, 0, v___x_159_);
lean_ctor_set(v___x_160_, 1, v_a_145_);
return v___x_160_;
}
else
{
lean_object* v___x_161_; lean_object* v_ref_162_; uint8_t v___x_163_; lean_object* v___x_164_; lean_object* v___x_165_; lean_object* v___x_166_; lean_object* v___x_167_; lean_object* v___x_168_; lean_object* v___x_169_; 
v___x_161_ = l_Lean_Syntax_getArg(v___x_157_, v___x_150_);
lean_dec(v___x_157_);
v_ref_162_ = l_Lean_replaceRef(v___x_151_, v_a_144_);
lean_dec(v___x_151_);
v___x_163_ = 0;
v___x_164_ = l_Lean_SourceInfo_fromRef(v_ref_162_, v___x_163_);
lean_dec(v_ref_162_);
v___x_165_ = ((lean_object*)(lp_mathlib_Matrix_term___u1d40___closed__2));
v___x_166_ = ((lean_object*)(lp_mathlib_Matrix_term___u1d40___closed__3));
lean_inc(v___x_164_);
v___x_167_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_167_, 0, v___x_164_);
lean_ctor_set(v___x_167_, 1, v___x_166_);
v___x_168_ = l_Lean_Syntax_node2(v___x_164_, v___x_165_, v___x_161_, v___x_167_);
v___x_169_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_169_, 0, v___x_168_);
lean_ctor_set(v___x_169_, 1, v_a_145_);
return v___x_169_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Matrix___aux__Mathlib__LinearAlgebra__Matrix__Defs______unexpand__Matrix__transpose__1___boxed(lean_object* v_x_170_, lean_object* v_a_171_, lean_object* v_a_172_){
_start:
{
lean_object* v_res_173_; 
v_res_173_ = lp_mathlib_Matrix___aux__Mathlib__LinearAlgebra__Matrix__Defs______unexpand__Matrix__transpose__1(v_x_170_, v_a_171_, v_a_172_);
lean_dec(v_a_171_);
return v_res_173_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Matrix_inhabited___aux__1___redArg(lean_object* v_inst_174_){
_start:
{
lean_inc(v_inst_174_);
return v_inst_174_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Matrix_inhabited___aux__1___redArg___boxed(lean_object* v_inst_175_){
_start:
{
lean_object* v_res_176_; 
v_res_176_ = lp_mathlib_Matrix_inhabited___aux__1___redArg(v_inst_175_);
lean_dec(v_inst_175_);
return v_res_176_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Matrix_inhabited___aux__1(lean_object* v_m_177_, lean_object* v_n_178_, lean_object* v_00_u03b1_179_, lean_object* v_inst_180_, lean_object* v_x_181_, lean_object* v_a_182_){
_start:
{
lean_inc(v_inst_180_);
return v_inst_180_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Matrix_inhabited___aux__1___boxed(lean_object* v_m_183_, lean_object* v_n_184_, lean_object* v_00_u03b1_185_, lean_object* v_inst_186_, lean_object* v_x_187_, lean_object* v_a_188_){
_start:
{
lean_object* v_res_189_; 
v_res_189_ = lp_mathlib_Matrix_inhabited___aux__1(v_m_183_, v_n_184_, v_00_u03b1_185_, v_inst_186_, v_x_187_, v_a_188_);
lean_dec(v_a_188_);
lean_dec(v_x_187_);
lean_dec(v_inst_186_);
return v_res_189_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Matrix_inhabited___redArg(lean_object* v_inst_190_){
_start:
{
lean_object* v___x_191_; 
v___x_191_ = lean_alloc_closure((void*)(lp_mathlib_Matrix_inhabited___aux__1___boxed), 6, 4);
lean_closure_set(v___x_191_, 0, lean_box(0));
lean_closure_set(v___x_191_, 1, lean_box(0));
lean_closure_set(v___x_191_, 2, lean_box(0));
lean_closure_set(v___x_191_, 3, v_inst_190_);
return v___x_191_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Matrix_inhabited(lean_object* v_m_192_, lean_object* v_n_193_, lean_object* v_00_u03b1_194_, lean_object* v_inst_195_){
_start:
{
lean_object* v___x_196_; 
v___x_196_ = lean_alloc_closure((void*)(lp_mathlib_Matrix_inhabited___aux__1___boxed), 6, 4);
lean_closure_set(v___x_196_, 0, lean_box(0));
lean_closure_set(v___x_196_, 1, lean_box(0));
lean_closure_set(v___x_196_, 2, lean_box(0));
lean_closure_set(v___x_196_, 3, v_inst_195_);
return v___x_196_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Matrix_add___aux__1___redArg(lean_object* v_inst_197_, lean_object* v_f_198_, lean_object* v_g_199_, lean_object* v_i_200_, lean_object* v_a_201_){
_start:
{
lean_object* v___x_202_; lean_object* v___x_203_; lean_object* v___x_204_; 
lean_inc(v_a_201_);
lean_inc(v_i_200_);
v___x_202_ = lean_apply_2(v_f_198_, v_i_200_, v_a_201_);
v___x_203_ = lean_apply_2(v_g_199_, v_i_200_, v_a_201_);
v___x_204_ = lean_apply_2(v_inst_197_, v___x_202_, v___x_203_);
return v___x_204_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Matrix_add___aux__1(lean_object* v_m_205_, lean_object* v_n_206_, lean_object* v_00_u03b1_207_, lean_object* v_inst_208_, lean_object* v_f_209_, lean_object* v_g_210_, lean_object* v_i_211_, lean_object* v_a_212_){
_start:
{
lean_object* v___x_213_; lean_object* v___x_214_; lean_object* v___x_215_; 
lean_inc(v_a_212_);
lean_inc(v_i_211_);
v___x_213_ = lean_apply_2(v_f_209_, v_i_211_, v_a_212_);
v___x_214_ = lean_apply_2(v_g_210_, v_i_211_, v_a_212_);
v___x_215_ = lean_apply_2(v_inst_208_, v___x_213_, v___x_214_);
return v___x_215_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Matrix_add___redArg(lean_object* v_inst_216_){
_start:
{
lean_object* v___x_217_; 
v___x_217_ = lean_alloc_closure((void*)(lp_mathlib_Matrix_add___aux__1), 8, 4);
lean_closure_set(v___x_217_, 0, lean_box(0));
lean_closure_set(v___x_217_, 1, lean_box(0));
lean_closure_set(v___x_217_, 2, lean_box(0));
lean_closure_set(v___x_217_, 3, v_inst_216_);
return v___x_217_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Matrix_add(lean_object* v_m_218_, lean_object* v_n_219_, lean_object* v_00_u03b1_220_, lean_object* v_inst_221_){
_start:
{
lean_object* v___x_222_; 
v___x_222_ = lean_alloc_closure((void*)(lp_mathlib_Matrix_add___aux__1), 8, 4);
lean_closure_set(v___x_222_, 0, lean_box(0));
lean_closure_set(v___x_222_, 1, lean_box(0));
lean_closure_set(v___x_222_, 2, lean_box(0));
lean_closure_set(v___x_222_, 3, v_inst_221_);
return v___x_222_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Matrix_smul___redArg___lam__0(lean_object* v_inst_223_, lean_object* v_a_224_, lean_object* v_b_225_, lean_object* v_i_226_, lean_object* v___y_227_){
_start:
{
lean_object* v___x_228_; lean_object* v___x_229_; 
v___x_228_ = lean_apply_2(v_b_225_, v_i_226_, v___y_227_);
v___x_229_ = lean_apply_2(v_inst_223_, v_a_224_, v___x_228_);
return v___x_229_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Matrix_smul___redArg(lean_object* v_inst_230_){
_start:
{
lean_object* v___f_231_; 
v___f_231_ = lean_alloc_closure((void*)(lp_mathlib_Matrix_smul___redArg___lam__0), 5, 1);
lean_closure_set(v___f_231_, 0, v_inst_230_);
return v___f_231_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Matrix_smul(lean_object* v_m_232_, lean_object* v_n_233_, lean_object* v_R_234_, lean_object* v_00_u03b1_235_, lean_object* v_inst_236_){
_start:
{
lean_object* v___f_237_; 
v___f_237_ = lean_alloc_closure((void*)(lp_mathlib_Matrix_smul___redArg___lam__0), 5, 1);
lean_closure_set(v___f_237_, 0, v_inst_236_);
return v___f_237_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Matrix_addSemigroup___redArg(lean_object* v_inst_238_){
_start:
{
lean_object* v___x_239_; 
v___x_239_ = lean_alloc_closure((void*)(lp_mathlib_Matrix_add___aux__1), 8, 4);
lean_closure_set(v___x_239_, 0, lean_box(0));
lean_closure_set(v___x_239_, 1, lean_box(0));
lean_closure_set(v___x_239_, 2, lean_box(0));
lean_closure_set(v___x_239_, 3, v_inst_238_);
return v___x_239_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Matrix_addSemigroup(lean_object* v_m_240_, lean_object* v_n_241_, lean_object* v_00_u03b1_242_, lean_object* v_inst_243_){
_start:
{
lean_object* v___x_244_; 
v___x_244_ = lean_alloc_closure((void*)(lp_mathlib_Matrix_add___aux__1), 8, 4);
lean_closure_set(v___x_244_, 0, lean_box(0));
lean_closure_set(v___x_244_, 1, lean_box(0));
lean_closure_set(v___x_244_, 2, lean_box(0));
lean_closure_set(v___x_244_, 3, v_inst_243_);
return v___x_244_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Matrix_addCommSemigroup___redArg(lean_object* v_inst_245_){
_start:
{
lean_object* v___x_246_; 
v___x_246_ = lean_alloc_closure((void*)(lp_mathlib_Matrix_add___aux__1), 8, 4);
lean_closure_set(v___x_246_, 0, lean_box(0));
lean_closure_set(v___x_246_, 1, lean_box(0));
lean_closure_set(v___x_246_, 2, lean_box(0));
lean_closure_set(v___x_246_, 3, v_inst_245_);
return v___x_246_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Matrix_addCommSemigroup(lean_object* v_m_247_, lean_object* v_n_248_, lean_object* v_00_u03b1_249_, lean_object* v_inst_250_){
_start:
{
lean_object* v___x_251_; 
v___x_251_ = lean_alloc_closure((void*)(lp_mathlib_Matrix_add___aux__1), 8, 4);
lean_closure_set(v___x_251_, 0, lean_box(0));
lean_closure_set(v___x_251_, 1, lean_box(0));
lean_closure_set(v___x_251_, 2, lean_box(0));
lean_closure_set(v___x_251_, 3, v_inst_250_);
return v___x_251_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Matrix_zero___aux__1___redArg(lean_object* v_inst_252_){
_start:
{
lean_inc(v_inst_252_);
return v_inst_252_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Matrix_zero___aux__1___redArg___boxed(lean_object* v_inst_253_){
_start:
{
lean_object* v_res_254_; 
v_res_254_ = lp_mathlib_Matrix_zero___aux__1___redArg(v_inst_253_);
lean_dec(v_inst_253_);
return v_res_254_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Matrix_zero___aux__1(lean_object* v_m_255_, lean_object* v_n_256_, lean_object* v_00_u03b1_257_, lean_object* v_inst_258_, lean_object* v_x_259_, lean_object* v_a_260_){
_start:
{
lean_inc(v_inst_258_);
return v_inst_258_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Matrix_zero___aux__1___boxed(lean_object* v_m_261_, lean_object* v_n_262_, lean_object* v_00_u03b1_263_, lean_object* v_inst_264_, lean_object* v_x_265_, lean_object* v_a_266_){
_start:
{
lean_object* v_res_267_; 
v_res_267_ = lp_mathlib_Matrix_zero___aux__1(v_m_261_, v_n_262_, v_00_u03b1_263_, v_inst_264_, v_x_265_, v_a_266_);
lean_dec(v_a_266_);
lean_dec(v_x_265_);
lean_dec(v_inst_264_);
return v_res_267_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Matrix_zero___redArg(lean_object* v_inst_268_){
_start:
{
lean_object* v___x_269_; 
v___x_269_ = lean_alloc_closure((void*)(lp_mathlib_Matrix_zero___aux__1___boxed), 6, 4);
lean_closure_set(v___x_269_, 0, lean_box(0));
lean_closure_set(v___x_269_, 1, lean_box(0));
lean_closure_set(v___x_269_, 2, lean_box(0));
lean_closure_set(v___x_269_, 3, v_inst_268_);
return v___x_269_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Matrix_zero(lean_object* v_m_270_, lean_object* v_n_271_, lean_object* v_00_u03b1_272_, lean_object* v_inst_273_){
_start:
{
lean_object* v___x_274_; 
v___x_274_ = lean_alloc_closure((void*)(lp_mathlib_Matrix_zero___aux__1___boxed), 6, 4);
lean_closure_set(v___x_274_, 0, lean_box(0));
lean_closure_set(v___x_274_, 1, lean_box(0));
lean_closure_set(v___x_274_, 2, lean_box(0));
lean_closure_set(v___x_274_, 3, v_inst_273_);
return v___x_274_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Matrix_addZeroClass___redArg(lean_object* v_inst_275_){
_start:
{
lean_object* v___x_276_; lean_object* v_toZero_277_; lean_object* v_toAdd_278_; lean_object* v___x_280_; uint8_t v_isShared_281_; uint8_t v_isSharedCheck_287_; 
v___x_276_ = lp_mathlib_AddZeroClass_toAddZero___redArg(v_inst_275_);
v_toZero_277_ = lean_ctor_get(v___x_276_, 0);
v_toAdd_278_ = lean_ctor_get(v___x_276_, 1);
v_isSharedCheck_287_ = !lean_is_exclusive(v___x_276_);
if (v_isSharedCheck_287_ == 0)
{
v___x_280_ = v___x_276_;
v_isShared_281_ = v_isSharedCheck_287_;
goto v_resetjp_279_;
}
else
{
lean_inc(v_toAdd_278_);
lean_inc(v_toZero_277_);
lean_dec(v___x_276_);
v___x_280_ = lean_box(0);
v_isShared_281_ = v_isSharedCheck_287_;
goto v_resetjp_279_;
}
v_resetjp_279_:
{
lean_object* v___x_282_; lean_object* v___x_283_; lean_object* v___x_285_; 
v___x_282_ = lean_alloc_closure((void*)(lp_mathlib_Matrix_zero___aux__1___boxed), 6, 4);
lean_closure_set(v___x_282_, 0, lean_box(0));
lean_closure_set(v___x_282_, 1, lean_box(0));
lean_closure_set(v___x_282_, 2, lean_box(0));
lean_closure_set(v___x_282_, 3, v_toZero_277_);
v___x_283_ = lean_alloc_closure((void*)(lp_mathlib_Matrix_add___aux__1), 8, 4);
lean_closure_set(v___x_283_, 0, lean_box(0));
lean_closure_set(v___x_283_, 1, lean_box(0));
lean_closure_set(v___x_283_, 2, lean_box(0));
lean_closure_set(v___x_283_, 3, v_toAdd_278_);
if (v_isShared_281_ == 0)
{
lean_ctor_set(v___x_280_, 1, v___x_283_);
lean_ctor_set(v___x_280_, 0, v___x_282_);
v___x_285_ = v___x_280_;
goto v_reusejp_284_;
}
else
{
lean_object* v_reuseFailAlloc_286_; 
v_reuseFailAlloc_286_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_286_, 0, v___x_282_);
lean_ctor_set(v_reuseFailAlloc_286_, 1, v___x_283_);
v___x_285_ = v_reuseFailAlloc_286_;
goto v_reusejp_284_;
}
v_reusejp_284_:
{
return v___x_285_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Matrix_addZeroClass(lean_object* v_m_288_, lean_object* v_n_289_, lean_object* v_00_u03b1_290_, lean_object* v_inst_291_){
_start:
{
lean_object* v___x_292_; 
v___x_292_ = lp_mathlib_Matrix_addZeroClass___redArg(v_inst_291_);
return v___x_292_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Matrix_addMonoid___redArg(lean_object* v_inst_293_){
_start:
{
lean_object* v___x_294_; lean_object* v___x_295_; lean_object* v_toZero_296_; lean_object* v_toAdd_297_; lean_object* v_toNSMul_298_; lean_object* v___x_300_; uint8_t v_isShared_301_; uint8_t v_isSharedCheck_312_; 
v___x_294_ = lp_mathlib_AddMonoid_toAddZeroClass___redArg(v_inst_293_);
v___x_295_ = lp_mathlib_AddZeroClass_toAddZero___redArg(v___x_294_);
v_toZero_296_ = lean_ctor_get(v___x_295_, 0);
lean_inc(v_toZero_296_);
lean_dec_ref(v___x_295_);
v_toAdd_297_ = lean_ctor_get(v_inst_293_, 1);
v_toNSMul_298_ = lean_ctor_get(v_inst_293_, 2);
v_isSharedCheck_312_ = !lean_is_exclusive(v_inst_293_);
if (v_isSharedCheck_312_ == 0)
{
lean_object* v_unused_313_; 
v_unused_313_ = lean_ctor_get(v_inst_293_, 0);
lean_dec(v_unused_313_);
v___x_300_ = v_inst_293_;
v_isShared_301_ = v_isSharedCheck_312_;
goto v_resetjp_299_;
}
else
{
lean_inc(v_toNSMul_298_);
lean_inc(v_toAdd_297_);
lean_dec(v_inst_293_);
v___x_300_ = lean_box(0);
v_isShared_301_ = v_isSharedCheck_312_;
goto v_resetjp_299_;
}
v_resetjp_299_:
{
lean_object* v___x_302_; lean_object* v___x_303_; lean_object* v___f_304_; lean_object* v___f_305_; lean_object* v___f_306_; lean_object* v___f_307_; lean_object* v___f_308_; lean_object* v___x_310_; 
v___x_302_ = lean_alloc_closure((void*)(lp_mathlib_Matrix_zero___aux__1___boxed), 6, 4);
lean_closure_set(v___x_302_, 0, lean_box(0));
lean_closure_set(v___x_302_, 1, lean_box(0));
lean_closure_set(v___x_302_, 2, lean_box(0));
lean_closure_set(v___x_302_, 3, v_toZero_296_);
v___x_303_ = lean_alloc_closure((void*)(lp_mathlib_Matrix_add___aux__1), 8, 4);
lean_closure_set(v___x_303_, 0, lean_box(0));
lean_closure_set(v___x_303_, 1, lean_box(0));
lean_closure_set(v___x_303_, 2, lean_box(0));
lean_closure_set(v___x_303_, 3, v_toAdd_297_);
v___f_304_ = lean_alloc_closure((void*)(lp_mathlib_NSMul_toSMul___redArg___lam__0), 3, 1);
lean_closure_set(v___f_304_, 0, v_toNSMul_298_);
v___f_305_ = lean_alloc_closure((void*)(lp_mathlib_Matrix_smul___redArg___lam__0), 5, 1);
lean_closure_set(v___f_305_, 0, v___f_304_);
v___f_306_ = lean_alloc_closure((void*)(lp_mathlib_NSMul_toSMul___redArg___lam__0), 3, 1);
lean_closure_set(v___f_306_, 0, v___f_305_);
v___f_307_ = lean_alloc_closure((void*)(lp_mathlib_NSMul_toSMul___redArg___lam__0), 3, 1);
lean_closure_set(v___f_307_, 0, v___f_306_);
v___f_308_ = lean_alloc_closure((void*)(lp_mathlib_NSMul_toSMul___redArg___lam__0), 3, 1);
lean_closure_set(v___f_308_, 0, v___f_307_);
if (v_isShared_301_ == 0)
{
lean_ctor_set(v___x_300_, 2, v___f_308_);
lean_ctor_set(v___x_300_, 1, v___x_303_);
lean_ctor_set(v___x_300_, 0, v___x_302_);
v___x_310_ = v___x_300_;
goto v_reusejp_309_;
}
else
{
lean_object* v_reuseFailAlloc_311_; 
v_reuseFailAlloc_311_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v_reuseFailAlloc_311_, 0, v___x_302_);
lean_ctor_set(v_reuseFailAlloc_311_, 1, v___x_303_);
lean_ctor_set(v_reuseFailAlloc_311_, 2, v___f_308_);
v___x_310_ = v_reuseFailAlloc_311_;
goto v_reusejp_309_;
}
v_reusejp_309_:
{
return v___x_310_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Matrix_addMonoid(lean_object* v_m_314_, lean_object* v_n_315_, lean_object* v_00_u03b1_316_, lean_object* v_inst_317_){
_start:
{
lean_object* v___x_318_; 
v___x_318_ = lp_mathlib_Matrix_addMonoid___redArg(v_inst_317_);
return v___x_318_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Matrix_addCommMonoid___redArg(lean_object* v_inst_319_){
_start:
{
lean_object* v___x_320_; 
v___x_320_ = lp_mathlib_Matrix_addMonoid___redArg(v_inst_319_);
return v___x_320_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Matrix_addCommMonoid(lean_object* v_m_321_, lean_object* v_n_322_, lean_object* v_00_u03b1_323_, lean_object* v_inst_324_){
_start:
{
lean_object* v___x_325_; 
v___x_325_ = lp_mathlib_Matrix_addMonoid___redArg(v_inst_324_);
return v___x_325_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Matrix_neg___aux__1___redArg(lean_object* v_inst_326_, lean_object* v_f_327_, lean_object* v_i_328_, lean_object* v_a_329_){
_start:
{
lean_object* v___x_330_; lean_object* v___x_331_; 
v___x_330_ = lean_apply_2(v_f_327_, v_i_328_, v_a_329_);
v___x_331_ = lean_apply_1(v_inst_326_, v___x_330_);
return v___x_331_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Matrix_neg___aux__1(lean_object* v_m_332_, lean_object* v_n_333_, lean_object* v_00_u03b1_334_, lean_object* v_inst_335_, lean_object* v_f_336_, lean_object* v_i_337_, lean_object* v_a_338_){
_start:
{
lean_object* v___x_339_; lean_object* v___x_340_; 
v___x_339_ = lean_apply_2(v_f_336_, v_i_337_, v_a_338_);
v___x_340_ = lean_apply_1(v_inst_335_, v___x_339_);
return v___x_340_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Matrix_neg___redArg(lean_object* v_inst_341_){
_start:
{
lean_object* v___x_342_; 
v___x_342_ = lean_alloc_closure((void*)(lp_mathlib_Matrix_neg___aux__1), 7, 4);
lean_closure_set(v___x_342_, 0, lean_box(0));
lean_closure_set(v___x_342_, 1, lean_box(0));
lean_closure_set(v___x_342_, 2, lean_box(0));
lean_closure_set(v___x_342_, 3, v_inst_341_);
return v___x_342_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Matrix_neg(lean_object* v_m_343_, lean_object* v_n_344_, lean_object* v_00_u03b1_345_, lean_object* v_inst_346_){
_start:
{
lean_object* v___x_347_; 
v___x_347_ = lean_alloc_closure((void*)(lp_mathlib_Matrix_neg___aux__1), 7, 4);
lean_closure_set(v___x_347_, 0, lean_box(0));
lean_closure_set(v___x_347_, 1, lean_box(0));
lean_closure_set(v___x_347_, 2, lean_box(0));
lean_closure_set(v___x_347_, 3, v_inst_346_);
return v___x_347_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Matrix_involutiveNeg___redArg(lean_object* v_inst_348_){
_start:
{
lean_object* v___x_349_; 
v___x_349_ = lean_alloc_closure((void*)(lp_mathlib_Matrix_neg___aux__1), 7, 4);
lean_closure_set(v___x_349_, 0, lean_box(0));
lean_closure_set(v___x_349_, 1, lean_box(0));
lean_closure_set(v___x_349_, 2, lean_box(0));
lean_closure_set(v___x_349_, 3, v_inst_348_);
return v___x_349_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Matrix_involutiveNeg(lean_object* v_m_350_, lean_object* v_n_351_, lean_object* v_00_u03b1_352_, lean_object* v_inst_353_){
_start:
{
lean_object* v___x_354_; 
v___x_354_ = lean_alloc_closure((void*)(lp_mathlib_Matrix_neg___aux__1), 7, 4);
lean_closure_set(v___x_354_, 0, lean_box(0));
lean_closure_set(v___x_354_, 1, lean_box(0));
lean_closure_set(v___x_354_, 2, lean_box(0));
lean_closure_set(v___x_354_, 3, v_inst_353_);
return v___x_354_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Matrix_sub___aux__1___redArg(lean_object* v_inst_355_, lean_object* v_f_356_, lean_object* v_g_357_, lean_object* v_i_358_, lean_object* v_a_359_){
_start:
{
lean_object* v___x_360_; lean_object* v___x_361_; lean_object* v___x_362_; 
lean_inc(v_a_359_);
lean_inc(v_i_358_);
v___x_360_ = lean_apply_2(v_f_356_, v_i_358_, v_a_359_);
v___x_361_ = lean_apply_2(v_g_357_, v_i_358_, v_a_359_);
v___x_362_ = lean_apply_2(v_inst_355_, v___x_360_, v___x_361_);
return v___x_362_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Matrix_sub___aux__1(lean_object* v_m_363_, lean_object* v_n_364_, lean_object* v_00_u03b1_365_, lean_object* v_inst_366_, lean_object* v_f_367_, lean_object* v_g_368_, lean_object* v_i_369_, lean_object* v_a_370_){
_start:
{
lean_object* v___x_371_; lean_object* v___x_372_; lean_object* v___x_373_; 
lean_inc(v_a_370_);
lean_inc(v_i_369_);
v___x_371_ = lean_apply_2(v_f_367_, v_i_369_, v_a_370_);
v___x_372_ = lean_apply_2(v_g_368_, v_i_369_, v_a_370_);
v___x_373_ = lean_apply_2(v_inst_366_, v___x_371_, v___x_372_);
return v___x_373_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Matrix_sub___redArg(lean_object* v_inst_374_){
_start:
{
lean_object* v___x_375_; 
v___x_375_ = lean_alloc_closure((void*)(lp_mathlib_Matrix_sub___aux__1), 8, 4);
lean_closure_set(v___x_375_, 0, lean_box(0));
lean_closure_set(v___x_375_, 1, lean_box(0));
lean_closure_set(v___x_375_, 2, lean_box(0));
lean_closure_set(v___x_375_, 3, v_inst_374_);
return v___x_375_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Matrix_sub(lean_object* v_m_376_, lean_object* v_n_377_, lean_object* v_00_u03b1_378_, lean_object* v_inst_379_){
_start:
{
lean_object* v___x_380_; 
v___x_380_ = lean_alloc_closure((void*)(lp_mathlib_Matrix_sub___aux__1), 8, 4);
lean_closure_set(v___x_380_, 0, lean_box(0));
lean_closure_set(v___x_380_, 1, lean_box(0));
lean_closure_set(v___x_380_, 2, lean_box(0));
lean_closure_set(v___x_380_, 3, v_inst_379_);
return v___x_380_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Matrix_addGroup___redArg(lean_object* v_inst_381_){
_start:
{
lean_object* v_toAddMonoid_382_; lean_object* v_toSub_383_; lean_object* v_toZSMul_384_; lean_object* v___x_385_; lean_object* v___x_386_; lean_object* v___x_388_; uint8_t v_isShared_389_; uint8_t v_isSharedCheck_401_; 
v_toAddMonoid_382_ = lean_ctor_get(v_inst_381_, 0);
v_toSub_383_ = lean_ctor_get(v_inst_381_, 2);
lean_inc(v_toSub_383_);
v_toZSMul_384_ = lean_ctor_get(v_inst_381_, 3);
lean_inc(v_toZSMul_384_);
lean_inc_ref(v_toAddMonoid_382_);
v___x_385_ = lp_mathlib_Matrix_addMonoid___redArg(v_toAddMonoid_382_);
v___x_386_ = lp_mathlib_SubNegZeroMonoid_toNegZeroClass___redArg(v_inst_381_);
v_isSharedCheck_401_ = !lean_is_exclusive(v_inst_381_);
if (v_isSharedCheck_401_ == 0)
{
lean_object* v_unused_402_; lean_object* v_unused_403_; lean_object* v_unused_404_; lean_object* v_unused_405_; 
v_unused_402_ = lean_ctor_get(v_inst_381_, 3);
lean_dec(v_unused_402_);
v_unused_403_ = lean_ctor_get(v_inst_381_, 2);
lean_dec(v_unused_403_);
v_unused_404_ = lean_ctor_get(v_inst_381_, 1);
lean_dec(v_unused_404_);
v_unused_405_ = lean_ctor_get(v_inst_381_, 0);
lean_dec(v_unused_405_);
v___x_388_ = v_inst_381_;
v_isShared_389_ = v_isSharedCheck_401_;
goto v_resetjp_387_;
}
else
{
lean_dec(v_inst_381_);
v___x_388_ = lean_box(0);
v_isShared_389_ = v_isSharedCheck_401_;
goto v_resetjp_387_;
}
v_resetjp_387_:
{
lean_object* v_toNeg_390_; lean_object* v___x_391_; lean_object* v___x_392_; lean_object* v___f_393_; lean_object* v___f_394_; lean_object* v___f_395_; lean_object* v___f_396_; lean_object* v___f_397_; lean_object* v___x_399_; 
v_toNeg_390_ = lean_ctor_get(v___x_386_, 1);
lean_inc(v_toNeg_390_);
lean_dec_ref(v___x_386_);
v___x_391_ = lean_alloc_closure((void*)(lp_mathlib_Matrix_neg___aux__1), 7, 4);
lean_closure_set(v___x_391_, 0, lean_box(0));
lean_closure_set(v___x_391_, 1, lean_box(0));
lean_closure_set(v___x_391_, 2, lean_box(0));
lean_closure_set(v___x_391_, 3, v_toNeg_390_);
v___x_392_ = lean_alloc_closure((void*)(lp_mathlib_Matrix_sub___aux__1), 8, 4);
lean_closure_set(v___x_392_, 0, lean_box(0));
lean_closure_set(v___x_392_, 1, lean_box(0));
lean_closure_set(v___x_392_, 2, lean_box(0));
lean_closure_set(v___x_392_, 3, v_toSub_383_);
v___f_393_ = lean_alloc_closure((void*)(lp_mathlib_ZSMul_toSMul___redArg___lam__0), 3, 1);
lean_closure_set(v___f_393_, 0, v_toZSMul_384_);
v___f_394_ = lean_alloc_closure((void*)(lp_mathlib_Matrix_smul___redArg___lam__0), 5, 1);
lean_closure_set(v___f_394_, 0, v___f_393_);
v___f_395_ = lean_alloc_closure((void*)(lp_mathlib_ZSMul_toSMul___redArg___lam__0), 3, 1);
lean_closure_set(v___f_395_, 0, v___f_394_);
v___f_396_ = lean_alloc_closure((void*)(lp_mathlib_ZSMul_toSMul___redArg___lam__0), 3, 1);
lean_closure_set(v___f_396_, 0, v___f_395_);
v___f_397_ = lean_alloc_closure((void*)(lp_mathlib_ZSMul_toSMul___redArg___lam__0), 3, 1);
lean_closure_set(v___f_397_, 0, v___f_396_);
if (v_isShared_389_ == 0)
{
lean_ctor_set(v___x_388_, 3, v___f_397_);
lean_ctor_set(v___x_388_, 2, v___x_392_);
lean_ctor_set(v___x_388_, 1, v___x_391_);
lean_ctor_set(v___x_388_, 0, v___x_385_);
v___x_399_ = v___x_388_;
goto v_reusejp_398_;
}
else
{
lean_object* v_reuseFailAlloc_400_; 
v_reuseFailAlloc_400_ = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(v_reuseFailAlloc_400_, 0, v___x_385_);
lean_ctor_set(v_reuseFailAlloc_400_, 1, v___x_391_);
lean_ctor_set(v_reuseFailAlloc_400_, 2, v___x_392_);
lean_ctor_set(v_reuseFailAlloc_400_, 3, v___f_397_);
v___x_399_ = v_reuseFailAlloc_400_;
goto v_reusejp_398_;
}
v_reusejp_398_:
{
return v___x_399_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Matrix_addGroup(lean_object* v_m_406_, lean_object* v_n_407_, lean_object* v_00_u03b1_408_, lean_object* v_inst_409_){
_start:
{
lean_object* v___x_410_; 
v___x_410_ = lp_mathlib_Matrix_addGroup___redArg(v_inst_409_);
return v___x_410_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Matrix_addCommGroup___redArg(lean_object* v_inst_411_){
_start:
{
lean_object* v___x_412_; 
v___x_412_ = lp_mathlib_Matrix_addGroup___redArg(v_inst_411_);
return v___x_412_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Matrix_addCommGroup(lean_object* v_m_413_, lean_object* v_n_414_, lean_object* v_00_u03b1_415_, lean_object* v_inst_416_){
_start:
{
lean_object* v___x_417_; 
v___x_417_ = lp_mathlib_Matrix_addGroup___redArg(v_inst_416_);
return v___x_417_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Matrix_unique___redArg(lean_object* v_inst_418_){
_start:
{
lean_object* v___x_419_; 
v___x_419_ = lean_alloc_closure((void*)(lp_mathlib_Matrix_inhabited___aux__1___boxed), 6, 4);
lean_closure_set(v___x_419_, 0, lean_box(0));
lean_closure_set(v___x_419_, 1, lean_box(0));
lean_closure_set(v___x_419_, 2, lean_box(0));
lean_closure_set(v___x_419_, 3, v_inst_418_);
return v___x_419_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Matrix_unique(lean_object* v_m_420_, lean_object* v_n_421_, lean_object* v_00_u03b1_422_, lean_object* v_inst_423_){
_start:
{
lean_object* v___x_424_; 
v___x_424_ = lean_alloc_closure((void*)(lp_mathlib_Matrix_inhabited___aux__1___boxed), 6, 4);
lean_closure_set(v___x_424_, 0, lean_box(0));
lean_closure_set(v___x_424_, 1, lean_box(0));
lean_closure_set(v___x_424_, 2, lean_box(0));
lean_closure_set(v___x_424_, 3, v_inst_423_);
return v___x_424_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Matrix_mulAction___redArg(lean_object* v_inst_425_){
_start:
{
lean_object* v___f_426_; 
v___f_426_ = lean_alloc_closure((void*)(lp_mathlib_Matrix_smul___redArg___lam__0), 5, 1);
lean_closure_set(v___f_426_, 0, v_inst_425_);
return v___f_426_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Matrix_mulAction(lean_object* v_m_427_, lean_object* v_n_428_, lean_object* v_R_429_, lean_object* v_00_u03b1_430_, lean_object* v_inst_431_, lean_object* v_inst_432_){
_start:
{
lean_object* v___f_433_; 
v___f_433_ = lean_alloc_closure((void*)(lp_mathlib_Matrix_smul___redArg___lam__0), 5, 1);
lean_closure_set(v___f_433_, 0, v_inst_432_);
return v___f_433_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Matrix_mulAction___boxed(lean_object* v_m_434_, lean_object* v_n_435_, lean_object* v_R_436_, lean_object* v_00_u03b1_437_, lean_object* v_inst_438_, lean_object* v_inst_439_){
_start:
{
lean_object* v_res_440_; 
v_res_440_ = lp_mathlib_Matrix_mulAction(v_m_434_, v_n_435_, v_R_436_, v_00_u03b1_437_, v_inst_438_, v_inst_439_);
lean_dec_ref(v_inst_438_);
return v_res_440_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Matrix_distribMulAction___redArg(lean_object* v_inst_441_){
_start:
{
lean_object* v___f_442_; 
v___f_442_ = lean_alloc_closure((void*)(lp_mathlib_Matrix_smul___redArg___lam__0), 5, 1);
lean_closure_set(v___f_442_, 0, v_inst_441_);
return v___f_442_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Matrix_distribMulAction(lean_object* v_m_443_, lean_object* v_n_444_, lean_object* v_R_445_, lean_object* v_00_u03b1_446_, lean_object* v_inst_447_, lean_object* v_inst_448_, lean_object* v_inst_449_){
_start:
{
lean_object* v___f_450_; 
v___f_450_ = lean_alloc_closure((void*)(lp_mathlib_Matrix_smul___redArg___lam__0), 5, 1);
lean_closure_set(v___f_450_, 0, v_inst_449_);
return v___f_450_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Matrix_distribMulAction___boxed(lean_object* v_m_451_, lean_object* v_n_452_, lean_object* v_R_453_, lean_object* v_00_u03b1_454_, lean_object* v_inst_455_, lean_object* v_inst_456_, lean_object* v_inst_457_){
_start:
{
lean_object* v_res_458_; 
v_res_458_ = lp_mathlib_Matrix_distribMulAction(v_m_451_, v_n_452_, v_R_453_, v_00_u03b1_454_, v_inst_455_, v_inst_456_, v_inst_457_);
lean_dec_ref(v_inst_456_);
lean_dec_ref(v_inst_455_);
return v_res_458_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Matrix_module___redArg(lean_object* v_inst_459_){
_start:
{
lean_object* v___f_460_; 
v___f_460_ = lean_alloc_closure((void*)(lp_mathlib_Matrix_smul___redArg___lam__0), 5, 1);
lean_closure_set(v___f_460_, 0, v_inst_459_);
return v___f_460_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Matrix_module(lean_object* v_m_461_, lean_object* v_n_462_, lean_object* v_R_463_, lean_object* v_00_u03b1_464_, lean_object* v_inst_465_, lean_object* v_inst_466_, lean_object* v_inst_467_){
_start:
{
lean_object* v___f_468_; 
v___f_468_ = lean_alloc_closure((void*)(lp_mathlib_Matrix_smul___redArg___lam__0), 5, 1);
lean_closure_set(v___f_468_, 0, v_inst_467_);
return v___f_468_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Matrix_module___boxed(lean_object* v_m_469_, lean_object* v_n_470_, lean_object* v_R_471_, lean_object* v_00_u03b1_472_, lean_object* v_inst_473_, lean_object* v_inst_474_, lean_object* v_inst_475_){
_start:
{
lean_object* v_res_476_; 
v_res_476_ = lp_mathlib_Matrix_module(v_m_469_, v_n_470_, v_R_471_, v_00_u03b1_472_, v_inst_473_, v_inst_474_, v_inst_475_);
lean_dec_ref(v_inst_474_);
lean_dec_ref(v_inst_473_);
return v_res_476_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Matrix_instAddCommMagma___redArg(lean_object* v_inst_477_){
_start:
{
lean_object* v___x_478_; 
v___x_478_ = lean_alloc_closure((void*)(lp_mathlib_Matrix_add___aux__1), 8, 4);
lean_closure_set(v___x_478_, 0, lean_box(0));
lean_closure_set(v___x_478_, 1, lean_box(0));
lean_closure_set(v___x_478_, 2, lean_box(0));
lean_closure_set(v___x_478_, 3, v_inst_477_);
return v___x_478_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Matrix_instAddCommMagma(lean_object* v_m_479_, lean_object* v_n_480_, lean_object* v_00_u03b1_481_, lean_object* v_inst_482_){
_start:
{
lean_object* v___x_483_; 
v___x_483_ = lean_alloc_closure((void*)(lp_mathlib_Matrix_add___aux__1), 8, 4);
lean_closure_set(v___x_483_, 0, lean_box(0));
lean_closure_set(v___x_483_, 1, lean_box(0));
lean_closure_set(v___x_483_, 2, lean_box(0));
lean_closure_set(v___x_483_, 3, v_inst_482_);
return v___x_483_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Matrix_instAddLeftCancelSemigroup___redArg(lean_object* v_inst_484_){
_start:
{
lean_object* v___x_485_; 
v___x_485_ = lean_alloc_closure((void*)(lp_mathlib_Matrix_add___aux__1), 8, 4);
lean_closure_set(v___x_485_, 0, lean_box(0));
lean_closure_set(v___x_485_, 1, lean_box(0));
lean_closure_set(v___x_485_, 2, lean_box(0));
lean_closure_set(v___x_485_, 3, v_inst_484_);
return v___x_485_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Matrix_instAddLeftCancelSemigroup(lean_object* v_m_486_, lean_object* v_n_487_, lean_object* v_00_u03b1_488_, lean_object* v_inst_489_){
_start:
{
lean_object* v___x_490_; 
v___x_490_ = lean_alloc_closure((void*)(lp_mathlib_Matrix_add___aux__1), 8, 4);
lean_closure_set(v___x_490_, 0, lean_box(0));
lean_closure_set(v___x_490_, 1, lean_box(0));
lean_closure_set(v___x_490_, 2, lean_box(0));
lean_closure_set(v___x_490_, 3, v_inst_489_);
return v___x_490_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Matrix_instAddRightCancelSemigroup___redArg(lean_object* v_inst_491_){
_start:
{
lean_object* v___x_492_; 
v___x_492_ = lean_alloc_closure((void*)(lp_mathlib_Matrix_add___aux__1), 8, 4);
lean_closure_set(v___x_492_, 0, lean_box(0));
lean_closure_set(v___x_492_, 1, lean_box(0));
lean_closure_set(v___x_492_, 2, lean_box(0));
lean_closure_set(v___x_492_, 3, v_inst_491_);
return v___x_492_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Matrix_instAddRightCancelSemigroup(lean_object* v_m_493_, lean_object* v_n_494_, lean_object* v_00_u03b1_495_, lean_object* v_inst_496_){
_start:
{
lean_object* v___x_497_; 
v___x_497_ = lean_alloc_closure((void*)(lp_mathlib_Matrix_add___aux__1), 8, 4);
lean_closure_set(v___x_497_, 0, lean_box(0));
lean_closure_set(v___x_497_, 1, lean_box(0));
lean_closure_set(v___x_497_, 2, lean_box(0));
lean_closure_set(v___x_497_, 3, v_inst_496_);
return v___x_497_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Matrix_instAddLeftCancelMonoid___redArg(lean_object* v_inst_498_){
_start:
{
lean_object* v___x_499_; 
v___x_499_ = lp_mathlib_Matrix_addMonoid___redArg(v_inst_498_);
return v___x_499_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Matrix_instAddLeftCancelMonoid(lean_object* v_m_500_, lean_object* v_n_501_, lean_object* v_00_u03b1_502_, lean_object* v_inst_503_){
_start:
{
lean_object* v___x_504_; 
v___x_504_ = lp_mathlib_Matrix_addMonoid___redArg(v_inst_503_);
return v___x_504_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Matrix_instAddRightCancelMonoid___redArg(lean_object* v_inst_505_){
_start:
{
lean_object* v___x_506_; 
v___x_506_ = lp_mathlib_Matrix_addMonoid___redArg(v_inst_505_);
return v___x_506_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Matrix_instAddRightCancelMonoid(lean_object* v_m_507_, lean_object* v_n_508_, lean_object* v_00_u03b1_509_, lean_object* v_inst_510_){
_start:
{
lean_object* v___x_511_; 
v___x_511_ = lp_mathlib_Matrix_addMonoid___redArg(v_inst_510_);
return v___x_511_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Matrix_instAddCancelMonoid___redArg(lean_object* v_inst_512_){
_start:
{
lean_object* v___x_513_; 
v___x_513_ = lp_mathlib_Matrix_addMonoid___redArg(v_inst_512_);
return v___x_513_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Matrix_instAddCancelMonoid(lean_object* v_m_514_, lean_object* v_n_515_, lean_object* v_00_u03b1_516_, lean_object* v_inst_517_){
_start:
{
lean_object* v___x_518_; 
v___x_518_ = lp_mathlib_Matrix_addMonoid___redArg(v_inst_517_);
return v___x_518_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Matrix_instAddCancelCommMonoid___redArg(lean_object* v_inst_519_){
_start:
{
lean_object* v___x_520_; 
v___x_520_ = lp_mathlib_Matrix_addMonoid___redArg(v_inst_519_);
return v___x_520_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Matrix_instAddCancelCommMonoid(lean_object* v_m_521_, lean_object* v_n_522_, lean_object* v_00_u03b1_523_, lean_object* v_inst_524_){
_start:
{
lean_object* v___x_525_; 
v___x_525_ = lp_mathlib_Matrix_addMonoid___redArg(v_inst_524_);
return v___x_525_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Matrix_ofAddEquiv(lean_object* v_m_526_, lean_object* v_n_527_, lean_object* v_00_u03b1_528_, lean_object* v_inst_529_){
_start:
{
lean_object* v___x_530_; 
v___x_530_ = lean_obj_once(&lp_mathlib_Matrix_of___closed__0, &lp_mathlib_Matrix_of___closed__0_once, _init_lp_mathlib_Matrix_of___closed__0);
return v___x_530_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Matrix_ofAddEquiv___boxed(lean_object* v_m_531_, lean_object* v_n_532_, lean_object* v_00_u03b1_533_, lean_object* v_inst_534_){
_start:
{
lean_object* v_res_535_; 
v_res_535_ = lp_mathlib_Matrix_ofAddEquiv(v_m_531_, v_n_532_, v_00_u03b1_533_, v_inst_534_);
lean_dec(v_inst_534_);
return v_res_535_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Matrix_submatrix___redArg___lam__0(lean_object* v_r_536_, lean_object* v_c_537_, lean_object* v_A_538_, lean_object* v_i_539_, lean_object* v_j_540_){
_start:
{
lean_object* v___x_541_; lean_object* v___x_542_; lean_object* v___x_543_; 
v___x_541_ = lean_apply_1(v_r_536_, v_i_539_);
v___x_542_ = lean_apply_1(v_c_537_, v_j_540_);
v___x_543_ = lean_apply_2(v_A_538_, v___x_541_, v___x_542_);
return v___x_543_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Matrix_submatrix___redArg(lean_object* v_A_544_, lean_object* v_r_545_, lean_object* v_c_546_, lean_object* v_a_547_, lean_object* v_a_548_){
_start:
{
lean_object* v___x_549_; lean_object* v_toFun_550_; lean_object* v___f_551_; lean_object* v___x_552_; 
v___x_549_ = lean_obj_once(&lp_mathlib_Matrix_of___closed__0, &lp_mathlib_Matrix_of___closed__0_once, _init_lp_mathlib_Matrix_of___closed__0);
v_toFun_550_ = lean_ctor_get(v___x_549_, 0);
v___f_551_ = lean_alloc_closure((void*)(lp_mathlib_Matrix_submatrix___redArg___lam__0), 5, 3);
lean_closure_set(v___f_551_, 0, v_r_545_);
lean_closure_set(v___f_551_, 1, v_c_546_);
lean_closure_set(v___f_551_, 2, v_A_544_);
lean_inc(v_toFun_550_);
v___x_552_ = lean_apply_3(v_toFun_550_, v___f_551_, v_a_547_, v_a_548_);
return v___x_552_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Matrix_submatrix(lean_object* v_l_553_, lean_object* v_m_554_, lean_object* v_n_555_, lean_object* v_o_556_, lean_object* v_00_u03b1_557_, lean_object* v_A_558_, lean_object* v_r_559_, lean_object* v_c_560_, lean_object* v_a_561_, lean_object* v_a_562_){
_start:
{
lean_object* v___x_563_; 
v___x_563_ = lp_mathlib_Matrix_submatrix___redArg(v_A_558_, v_r_559_, v_c_560_, v_a_561_, v_a_562_);
return v___x_563_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Matrix_reindex___redArg___lam__0(lean_object* v___x_564_, lean_object* v___y_565_){
_start:
{
lean_object* v_toFun_566_; lean_object* v___x_567_; 
v_toFun_566_ = lean_ctor_get(v___x_564_, 0);
lean_inc(v_toFun_566_);
lean_dec_ref(v___x_564_);
v___x_567_ = lean_apply_1(v_toFun_566_, v___y_565_);
return v___x_567_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Matrix_reindex___redArg___lam__2(lean_object* v_e_u2098_568_, lean_object* v_e_u2099_569_, lean_object* v_M_570_, lean_object* v___y_571_, lean_object* v___y_572_){
_start:
{
lean_object* v___x_573_; lean_object* v___f_574_; lean_object* v___x_575_; lean_object* v___f_576_; lean_object* v___x_577_; 
v___x_573_ = lp_mathlib_Equiv_symm___redArg(v_e_u2098_568_);
v___f_574_ = lean_alloc_closure((void*)(lp_mathlib_Matrix_reindex___redArg___lam__0), 2, 1);
lean_closure_set(v___f_574_, 0, v___x_573_);
v___x_575_ = lp_mathlib_Equiv_symm___redArg(v_e_u2099_569_);
v___f_576_ = lean_alloc_closure((void*)(lp_mathlib_Matrix_reindex___redArg___lam__0), 2, 1);
lean_closure_set(v___f_576_, 0, v___x_575_);
v___x_577_ = lp_mathlib_Matrix_submatrix___redArg(v_M_570_, v___f_574_, v___f_576_, v___y_571_, v___y_572_);
return v___x_577_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Matrix_reindex___redArg___lam__1(lean_object* v_e_u2098_578_, lean_object* v___y_579_){
_start:
{
lean_object* v_toFun_580_; lean_object* v___x_581_; 
v_toFun_580_ = lean_ctor_get(v_e_u2098_578_, 0);
lean_inc(v_toFun_580_);
lean_dec_ref(v_e_u2098_578_);
v___x_581_ = lean_apply_1(v_toFun_580_, v___y_579_);
return v___x_581_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Matrix_reindex___redArg___lam__3(lean_object* v_e_u2099_582_, lean_object* v___y_583_){
_start:
{
lean_object* v_toFun_584_; lean_object* v___x_585_; 
v_toFun_584_ = lean_ctor_get(v_e_u2099_582_, 0);
lean_inc(v_toFun_584_);
lean_dec_ref(v_e_u2099_582_);
v___x_585_ = lean_apply_1(v_toFun_584_, v___y_583_);
return v___x_585_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Matrix_reindex___redArg___lam__4(lean_object* v___f_586_, lean_object* v___f_587_, lean_object* v_M_588_, lean_object* v___y_589_, lean_object* v___y_590_){
_start:
{
lean_object* v___x_591_; 
v___x_591_ = lp_mathlib_Matrix_submatrix___redArg(v_M_588_, v___f_586_, v___f_587_, v___y_589_, v___y_590_);
return v___x_591_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Matrix_reindex___redArg(lean_object* v_e_u2098_592_, lean_object* v_e_u2099_593_){
_start:
{
lean_object* v___f_594_; lean_object* v___f_595_; lean_object* v___f_596_; lean_object* v___f_597_; lean_object* v___x_598_; 
lean_inc_ref(v_e_u2099_593_);
lean_inc_ref(v_e_u2098_592_);
v___f_594_ = lean_alloc_closure((void*)(lp_mathlib_Matrix_reindex___redArg___lam__2), 5, 2);
lean_closure_set(v___f_594_, 0, v_e_u2098_592_);
lean_closure_set(v___f_594_, 1, v_e_u2099_593_);
v___f_595_ = lean_alloc_closure((void*)(lp_mathlib_Matrix_reindex___redArg___lam__1), 2, 1);
lean_closure_set(v___f_595_, 0, v_e_u2098_592_);
v___f_596_ = lean_alloc_closure((void*)(lp_mathlib_Matrix_reindex___redArg___lam__3), 2, 1);
lean_closure_set(v___f_596_, 0, v_e_u2099_593_);
v___f_597_ = lean_alloc_closure((void*)(lp_mathlib_Matrix_reindex___redArg___lam__4), 5, 2);
lean_closure_set(v___f_597_, 0, v___f_595_);
lean_closure_set(v___f_597_, 1, v___f_596_);
v___x_598_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_598_, 0, v___f_594_);
lean_ctor_set(v___x_598_, 1, v___f_597_);
return v___x_598_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Matrix_reindex(lean_object* v_l_599_, lean_object* v_m_600_, lean_object* v_n_601_, lean_object* v_o_602_, lean_object* v_00_u03b1_603_, lean_object* v_e_u2098_604_, lean_object* v_e_u2099_605_){
_start:
{
lean_object* v___x_606_; 
v___x_606_ = lp_mathlib_Matrix_reindex___redArg(v_e_u2098_604_, v_e_u2099_605_);
return v___x_606_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Matrix_subLeft___redArg(lean_object* v_l_608_, lean_object* v_r_609_, lean_object* v_A_610_, lean_object* v_a_611_, lean_object* v_a_612_){
_start:
{
lean_object* v___x_613_; lean_object* v___x_614_; lean_object* v___x_615_; 
v___x_613_ = ((lean_object*)(lp_mathlib_Matrix_subLeft___redArg___closed__0));
v___x_614_ = lean_alloc_closure((void*)(l_Fin_castAdd___boxed), 3, 2);
lean_closure_set(v___x_614_, 0, v_l_608_);
lean_closure_set(v___x_614_, 1, v_r_609_);
v___x_615_ = lp_mathlib_Matrix_submatrix___redArg(v_A_610_, v___x_613_, v___x_614_, v_a_611_, v_a_612_);
return v___x_615_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Matrix_subLeft(lean_object* v_00_u03b1_616_, lean_object* v_m_617_, lean_object* v_l_618_, lean_object* v_r_619_, lean_object* v_A_620_, lean_object* v_a_621_, lean_object* v_a_622_){
_start:
{
lean_object* v___x_623_; lean_object* v___x_624_; lean_object* v___x_625_; 
v___x_623_ = ((lean_object*)(lp_mathlib_Matrix_subLeft___redArg___closed__0));
v___x_624_ = lean_alloc_closure((void*)(l_Fin_castAdd___boxed), 3, 2);
lean_closure_set(v___x_624_, 0, v_l_618_);
lean_closure_set(v___x_624_, 1, v_r_619_);
v___x_625_ = lp_mathlib_Matrix_submatrix___redArg(v_A_620_, v___x_623_, v___x_624_, v_a_621_, v_a_622_);
return v___x_625_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Matrix_subLeft___boxed(lean_object* v_00_u03b1_626_, lean_object* v_m_627_, lean_object* v_l_628_, lean_object* v_r_629_, lean_object* v_A_630_, lean_object* v_a_631_, lean_object* v_a_632_){
_start:
{
lean_object* v_res_633_; 
v_res_633_ = lp_mathlib_Matrix_subLeft(v_00_u03b1_626_, v_m_627_, v_l_628_, v_r_629_, v_A_630_, v_a_631_, v_a_632_);
lean_dec(v_m_627_);
return v_res_633_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Matrix_subRight___redArg(lean_object* v_l_634_, lean_object* v_r_635_, lean_object* v_A_636_, lean_object* v_a_637_, lean_object* v_a_638_){
_start:
{
lean_object* v___x_639_; lean_object* v___x_640_; lean_object* v___x_641_; 
v___x_639_ = ((lean_object*)(lp_mathlib_Matrix_subLeft___redArg___closed__0));
v___x_640_ = lean_alloc_closure((void*)(l_Fin_natAdd___boxed), 3, 2);
lean_closure_set(v___x_640_, 0, v_r_635_);
lean_closure_set(v___x_640_, 1, v_l_634_);
v___x_641_ = lp_mathlib_Matrix_submatrix___redArg(v_A_636_, v___x_639_, v___x_640_, v_a_637_, v_a_638_);
return v___x_641_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Matrix_subRight(lean_object* v_00_u03b1_642_, lean_object* v_m_643_, lean_object* v_l_644_, lean_object* v_r_645_, lean_object* v_A_646_, lean_object* v_a_647_, lean_object* v_a_648_){
_start:
{
lean_object* v___x_649_; lean_object* v___x_650_; lean_object* v___x_651_; 
v___x_649_ = ((lean_object*)(lp_mathlib_Matrix_subLeft___redArg___closed__0));
v___x_650_ = lean_alloc_closure((void*)(l_Fin_natAdd___boxed), 3, 2);
lean_closure_set(v___x_650_, 0, v_r_645_);
lean_closure_set(v___x_650_, 1, v_l_644_);
v___x_651_ = lp_mathlib_Matrix_submatrix___redArg(v_A_646_, v___x_649_, v___x_650_, v_a_647_, v_a_648_);
return v___x_651_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Matrix_subRight___boxed(lean_object* v_00_u03b1_652_, lean_object* v_m_653_, lean_object* v_l_654_, lean_object* v_r_655_, lean_object* v_A_656_, lean_object* v_a_657_, lean_object* v_a_658_){
_start:
{
lean_object* v_res_659_; 
v_res_659_ = lp_mathlib_Matrix_subRight(v_00_u03b1_652_, v_m_653_, v_l_654_, v_r_655_, v_A_656_, v_a_657_, v_a_658_);
lean_dec(v_m_653_);
return v_res_659_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Matrix_subUp___redArg(lean_object* v_d_660_, lean_object* v_u_661_, lean_object* v_A_662_, lean_object* v_a_663_, lean_object* v_a_664_){
_start:
{
lean_object* v___x_665_; lean_object* v___x_666_; lean_object* v___x_667_; 
v___x_665_ = lean_alloc_closure((void*)(l_Fin_castAdd___boxed), 3, 2);
lean_closure_set(v___x_665_, 0, v_u_661_);
lean_closure_set(v___x_665_, 1, v_d_660_);
v___x_666_ = ((lean_object*)(lp_mathlib_Matrix_subLeft___redArg___closed__0));
v___x_667_ = lp_mathlib_Matrix_submatrix___redArg(v_A_662_, v___x_665_, v___x_666_, v_a_663_, v_a_664_);
return v___x_667_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Matrix_subUp(lean_object* v_00_u03b1_668_, lean_object* v_d_669_, lean_object* v_u_670_, lean_object* v_n_671_, lean_object* v_A_672_, lean_object* v_a_673_, lean_object* v_a_674_){
_start:
{
lean_object* v___x_675_; lean_object* v___x_676_; lean_object* v___x_677_; 
v___x_675_ = lean_alloc_closure((void*)(l_Fin_castAdd___boxed), 3, 2);
lean_closure_set(v___x_675_, 0, v_u_670_);
lean_closure_set(v___x_675_, 1, v_d_669_);
v___x_676_ = ((lean_object*)(lp_mathlib_Matrix_subLeft___redArg___closed__0));
v___x_677_ = lp_mathlib_Matrix_submatrix___redArg(v_A_672_, v___x_675_, v___x_676_, v_a_673_, v_a_674_);
return v___x_677_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Matrix_subUp___boxed(lean_object* v_00_u03b1_678_, lean_object* v_d_679_, lean_object* v_u_680_, lean_object* v_n_681_, lean_object* v_A_682_, lean_object* v_a_683_, lean_object* v_a_684_){
_start:
{
lean_object* v_res_685_; 
v_res_685_ = lp_mathlib_Matrix_subUp(v_00_u03b1_678_, v_d_679_, v_u_680_, v_n_681_, v_A_682_, v_a_683_, v_a_684_);
lean_dec(v_n_681_);
return v_res_685_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Matrix_subDown___redArg(lean_object* v_d_686_, lean_object* v_u_687_, lean_object* v_A_688_, lean_object* v_a_689_, lean_object* v_a_690_){
_start:
{
lean_object* v___x_691_; lean_object* v___x_692_; lean_object* v___x_693_; 
v___x_691_ = lean_alloc_closure((void*)(l_Fin_natAdd___boxed), 3, 2);
lean_closure_set(v___x_691_, 0, v_d_686_);
lean_closure_set(v___x_691_, 1, v_u_687_);
v___x_692_ = ((lean_object*)(lp_mathlib_Matrix_subLeft___redArg___closed__0));
v___x_693_ = lp_mathlib_Matrix_submatrix___redArg(v_A_688_, v___x_691_, v___x_692_, v_a_689_, v_a_690_);
return v___x_693_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Matrix_subDown(lean_object* v_00_u03b1_694_, lean_object* v_d_695_, lean_object* v_u_696_, lean_object* v_n_697_, lean_object* v_A_698_, lean_object* v_a_699_, lean_object* v_a_700_){
_start:
{
lean_object* v___x_701_; lean_object* v___x_702_; lean_object* v___x_703_; 
v___x_701_ = lean_alloc_closure((void*)(l_Fin_natAdd___boxed), 3, 2);
lean_closure_set(v___x_701_, 0, v_d_695_);
lean_closure_set(v___x_701_, 1, v_u_696_);
v___x_702_ = ((lean_object*)(lp_mathlib_Matrix_subLeft___redArg___closed__0));
v___x_703_ = lp_mathlib_Matrix_submatrix___redArg(v_A_698_, v___x_701_, v___x_702_, v_a_699_, v_a_700_);
return v___x_703_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Matrix_subDown___boxed(lean_object* v_00_u03b1_704_, lean_object* v_d_705_, lean_object* v_u_706_, lean_object* v_n_707_, lean_object* v_A_708_, lean_object* v_a_709_, lean_object* v_a_710_){
_start:
{
lean_object* v_res_711_; 
v_res_711_ = lp_mathlib_Matrix_subDown(v_00_u03b1_704_, v_d_705_, v_u_706_, v_n_707_, v_A_708_, v_a_709_, v_a_710_);
lean_dec(v_n_707_);
return v_res_711_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Matrix_subUpRight___redArg(lean_object* v_d_712_, lean_object* v_u_713_, lean_object* v_l_714_, lean_object* v_r_715_, lean_object* v_A_716_, lean_object* v_a_717_, lean_object* v_a_718_){
_start:
{
lean_object* v___x_719_; lean_object* v___x_720_; lean_object* v___x_721_; lean_object* v___x_722_; lean_object* v___x_723_; 
v___x_719_ = lean_nat_add(v_u_713_, v_d_712_);
v___x_720_ = lean_alloc_closure((void*)(lp_mathlib_Matrix_subRight___boxed), 7, 5);
lean_closure_set(v___x_720_, 0, lean_box(0));
lean_closure_set(v___x_720_, 1, v___x_719_);
lean_closure_set(v___x_720_, 2, v_l_714_);
lean_closure_set(v___x_720_, 3, v_r_715_);
lean_closure_set(v___x_720_, 4, v_A_716_);
v___x_721_ = lean_alloc_closure((void*)(l_Fin_castAdd___boxed), 3, 2);
lean_closure_set(v___x_721_, 0, v_u_713_);
lean_closure_set(v___x_721_, 1, v_d_712_);
v___x_722_ = ((lean_object*)(lp_mathlib_Matrix_subLeft___redArg___closed__0));
v___x_723_ = lp_mathlib_Matrix_submatrix___redArg(v___x_720_, v___x_721_, v___x_722_, v_a_717_, v_a_718_);
return v___x_723_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Matrix_subUpRight(lean_object* v_00_u03b1_724_, lean_object* v_d_725_, lean_object* v_u_726_, lean_object* v_l_727_, lean_object* v_r_728_, lean_object* v_A_729_, lean_object* v_a_730_, lean_object* v_a_731_){
_start:
{
lean_object* v___x_732_; lean_object* v___x_733_; lean_object* v___x_734_; lean_object* v___x_735_; lean_object* v___x_736_; 
v___x_732_ = lean_nat_add(v_u_726_, v_d_725_);
v___x_733_ = lean_alloc_closure((void*)(lp_mathlib_Matrix_subRight___boxed), 7, 5);
lean_closure_set(v___x_733_, 0, lean_box(0));
lean_closure_set(v___x_733_, 1, v___x_732_);
lean_closure_set(v___x_733_, 2, v_l_727_);
lean_closure_set(v___x_733_, 3, v_r_728_);
lean_closure_set(v___x_733_, 4, v_A_729_);
v___x_734_ = lean_alloc_closure((void*)(l_Fin_castAdd___boxed), 3, 2);
lean_closure_set(v___x_734_, 0, v_u_726_);
lean_closure_set(v___x_734_, 1, v_d_725_);
v___x_735_ = ((lean_object*)(lp_mathlib_Matrix_subLeft___redArg___closed__0));
v___x_736_ = lp_mathlib_Matrix_submatrix___redArg(v___x_733_, v___x_734_, v___x_735_, v_a_730_, v_a_731_);
return v___x_736_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Matrix_subDownRight___redArg(lean_object* v_d_737_, lean_object* v_u_738_, lean_object* v_l_739_, lean_object* v_r_740_, lean_object* v_A_741_, lean_object* v_a_742_, lean_object* v_a_743_){
_start:
{
lean_object* v___x_744_; lean_object* v___x_745_; lean_object* v___x_746_; lean_object* v___x_747_; lean_object* v___x_748_; 
v___x_744_ = lean_nat_add(v_u_738_, v_d_737_);
v___x_745_ = lean_alloc_closure((void*)(lp_mathlib_Matrix_subRight___boxed), 7, 5);
lean_closure_set(v___x_745_, 0, lean_box(0));
lean_closure_set(v___x_745_, 1, v___x_744_);
lean_closure_set(v___x_745_, 2, v_l_739_);
lean_closure_set(v___x_745_, 3, v_r_740_);
lean_closure_set(v___x_745_, 4, v_A_741_);
v___x_746_ = lean_alloc_closure((void*)(l_Fin_natAdd___boxed), 3, 2);
lean_closure_set(v___x_746_, 0, v_d_737_);
lean_closure_set(v___x_746_, 1, v_u_738_);
v___x_747_ = ((lean_object*)(lp_mathlib_Matrix_subLeft___redArg___closed__0));
v___x_748_ = lp_mathlib_Matrix_submatrix___redArg(v___x_745_, v___x_746_, v___x_747_, v_a_742_, v_a_743_);
return v___x_748_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Matrix_subDownRight(lean_object* v_00_u03b1_749_, lean_object* v_d_750_, lean_object* v_u_751_, lean_object* v_l_752_, lean_object* v_r_753_, lean_object* v_A_754_, lean_object* v_a_755_, lean_object* v_a_756_){
_start:
{
lean_object* v___x_757_; lean_object* v___x_758_; lean_object* v___x_759_; lean_object* v___x_760_; lean_object* v___x_761_; 
v___x_757_ = lean_nat_add(v_u_751_, v_d_750_);
v___x_758_ = lean_alloc_closure((void*)(lp_mathlib_Matrix_subRight___boxed), 7, 5);
lean_closure_set(v___x_758_, 0, lean_box(0));
lean_closure_set(v___x_758_, 1, v___x_757_);
lean_closure_set(v___x_758_, 2, v_l_752_);
lean_closure_set(v___x_758_, 3, v_r_753_);
lean_closure_set(v___x_758_, 4, v_A_754_);
v___x_759_ = lean_alloc_closure((void*)(l_Fin_natAdd___boxed), 3, 2);
lean_closure_set(v___x_759_, 0, v_d_750_);
lean_closure_set(v___x_759_, 1, v_u_751_);
v___x_760_ = ((lean_object*)(lp_mathlib_Matrix_subLeft___redArg___closed__0));
v___x_761_ = lp_mathlib_Matrix_submatrix___redArg(v___x_758_, v___x_759_, v___x_760_, v_a_755_, v_a_756_);
return v___x_761_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Matrix_subUpLeft___redArg(lean_object* v_d_762_, lean_object* v_u_763_, lean_object* v_l_764_, lean_object* v_r_765_, lean_object* v_A_766_, lean_object* v_a_767_, lean_object* v_a_768_){
_start:
{
lean_object* v___x_769_; lean_object* v___x_770_; lean_object* v___x_771_; lean_object* v___x_772_; lean_object* v___x_773_; 
v___x_769_ = lean_nat_add(v_u_763_, v_d_762_);
v___x_770_ = lean_alloc_closure((void*)(lp_mathlib_Matrix_subLeft___boxed), 7, 5);
lean_closure_set(v___x_770_, 0, lean_box(0));
lean_closure_set(v___x_770_, 1, v___x_769_);
lean_closure_set(v___x_770_, 2, v_l_764_);
lean_closure_set(v___x_770_, 3, v_r_765_);
lean_closure_set(v___x_770_, 4, v_A_766_);
v___x_771_ = lean_alloc_closure((void*)(l_Fin_castAdd___boxed), 3, 2);
lean_closure_set(v___x_771_, 0, v_u_763_);
lean_closure_set(v___x_771_, 1, v_d_762_);
v___x_772_ = ((lean_object*)(lp_mathlib_Matrix_subLeft___redArg___closed__0));
v___x_773_ = lp_mathlib_Matrix_submatrix___redArg(v___x_770_, v___x_771_, v___x_772_, v_a_767_, v_a_768_);
return v___x_773_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Matrix_subUpLeft(lean_object* v_00_u03b1_774_, lean_object* v_d_775_, lean_object* v_u_776_, lean_object* v_l_777_, lean_object* v_r_778_, lean_object* v_A_779_, lean_object* v_a_780_, lean_object* v_a_781_){
_start:
{
lean_object* v___x_782_; lean_object* v___x_783_; lean_object* v___x_784_; lean_object* v___x_785_; lean_object* v___x_786_; 
v___x_782_ = lean_nat_add(v_u_776_, v_d_775_);
v___x_783_ = lean_alloc_closure((void*)(lp_mathlib_Matrix_subLeft___boxed), 7, 5);
lean_closure_set(v___x_783_, 0, lean_box(0));
lean_closure_set(v___x_783_, 1, v___x_782_);
lean_closure_set(v___x_783_, 2, v_l_777_);
lean_closure_set(v___x_783_, 3, v_r_778_);
lean_closure_set(v___x_783_, 4, v_A_779_);
v___x_784_ = lean_alloc_closure((void*)(l_Fin_castAdd___boxed), 3, 2);
lean_closure_set(v___x_784_, 0, v_u_776_);
lean_closure_set(v___x_784_, 1, v_d_775_);
v___x_785_ = ((lean_object*)(lp_mathlib_Matrix_subLeft___redArg___closed__0));
v___x_786_ = lp_mathlib_Matrix_submatrix___redArg(v___x_783_, v___x_784_, v___x_785_, v_a_780_, v_a_781_);
return v___x_786_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Matrix_subDownLeft___redArg(lean_object* v_d_787_, lean_object* v_u_788_, lean_object* v_l_789_, lean_object* v_r_790_, lean_object* v_A_791_, lean_object* v_a_792_, lean_object* v_a_793_){
_start:
{
lean_object* v___x_794_; lean_object* v___x_795_; lean_object* v___x_796_; lean_object* v___x_797_; lean_object* v___x_798_; 
v___x_794_ = lean_nat_add(v_u_788_, v_d_787_);
v___x_795_ = lean_alloc_closure((void*)(lp_mathlib_Matrix_subLeft___boxed), 7, 5);
lean_closure_set(v___x_795_, 0, lean_box(0));
lean_closure_set(v___x_795_, 1, v___x_794_);
lean_closure_set(v___x_795_, 2, v_l_789_);
lean_closure_set(v___x_795_, 3, v_r_790_);
lean_closure_set(v___x_795_, 4, v_A_791_);
v___x_796_ = lean_alloc_closure((void*)(l_Fin_natAdd___boxed), 3, 2);
lean_closure_set(v___x_796_, 0, v_d_787_);
lean_closure_set(v___x_796_, 1, v_u_788_);
v___x_797_ = ((lean_object*)(lp_mathlib_Matrix_subLeft___redArg___closed__0));
v___x_798_ = lp_mathlib_Matrix_submatrix___redArg(v___x_795_, v___x_796_, v___x_797_, v_a_792_, v_a_793_);
return v___x_798_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Matrix_subDownLeft(lean_object* v_00_u03b1_799_, lean_object* v_d_800_, lean_object* v_u_801_, lean_object* v_l_802_, lean_object* v_r_803_, lean_object* v_A_804_, lean_object* v_a_805_, lean_object* v_a_806_){
_start:
{
lean_object* v___x_807_; lean_object* v___x_808_; lean_object* v___x_809_; lean_object* v___x_810_; lean_object* v___x_811_; 
v___x_807_ = lean_nat_add(v_u_801_, v_d_800_);
v___x_808_ = lean_alloc_closure((void*)(lp_mathlib_Matrix_subLeft___boxed), 7, 5);
lean_closure_set(v___x_808_, 0, lean_box(0));
lean_closure_set(v___x_808_, 1, v___x_807_);
lean_closure_set(v___x_808_, 2, v_l_802_);
lean_closure_set(v___x_808_, 3, v_r_803_);
lean_closure_set(v___x_808_, 4, v_A_804_);
v___x_809_ = lean_alloc_closure((void*)(l_Fin_natAdd___boxed), 3, 2);
lean_closure_set(v___x_809_, 0, v_d_800_);
lean_closure_set(v___x_809_, 1, v_u_801_);
v___x_810_ = ((lean_object*)(lp_mathlib_Matrix_subLeft___redArg___closed__0));
v___x_811_ = lp_mathlib_Matrix_submatrix___redArg(v___x_808_, v___x_809_, v___x_810_, v_a_805_, v_a_806_);
return v___x_811_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Matrix_row___redArg(lean_object* v_A_812_, lean_object* v_a_813_, lean_object* v_a_814_){
_start:
{
lean_object* v___x_815_; 
v___x_815_ = lean_apply_2(v_A_812_, v_a_813_, v_a_814_);
return v___x_815_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Matrix_row(lean_object* v_m_816_, lean_object* v_n_817_, lean_object* v_00_u03b1_818_, lean_object* v_A_819_, lean_object* v_a_820_, lean_object* v_a_821_){
_start:
{
lean_object* v___x_822_; 
v___x_822_ = lean_apply_2(v_A_819_, v_a_820_, v_a_821_);
return v___x_822_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Matrix_col___redArg(lean_object* v_A_823_, lean_object* v_a_824_, lean_object* v_a_825_){
_start:
{
lean_object* v___x_826_; 
v___x_826_ = lp_mathlib_Matrix_transpose___redArg(v_A_823_, v_a_824_, v_a_825_);
return v___x_826_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Matrix_col(lean_object* v_m_827_, lean_object* v_n_828_, lean_object* v_00_u03b1_829_, lean_object* v_A_830_, lean_object* v_a_831_, lean_object* v_a_832_){
_start:
{
lean_object* v___x_833_; 
v___x_833_ = lp_mathlib_Matrix_transpose___redArg(v_A_830_, v_a_831_, v_a_832_);
return v___x_833_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_batteries_Batteries_Data_Fin_Lemmas(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Module_Pi(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Basic_Nontrivial_Basic(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Tactic_CrossRefAttribute(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Tactic_Attr_Core(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_LinearAlgebra_Matrix_Defs(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_batteries_Batteries_Data_Fin_Lemmas(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Module_Pi(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Basic_Nontrivial_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Tactic_CrossRefAttribute(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Tactic_Attr_Core(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_LinearAlgebra_Matrix_Defs(uint8_t builtin) {
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
lean_object* initialize_batteries_Batteries_Data_Fin_Lemmas(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Module_Pi(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Basic_Nontrivial_Basic(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Tactic_CrossRefAttribute(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Tactic_Attr_Core(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_LinearAlgebra_Matrix_Defs(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_batteries_Batteries_Data_Fin_Lemmas(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Module_Pi(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Basic_Nontrivial_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Tactic_CrossRefAttribute(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Tactic_Attr_Core(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_LinearAlgebra_Matrix_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_LinearAlgebra_Matrix_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_LinearAlgebra_Matrix_Defs(builtin);
}
#ifdef __cplusplus
}
#endif
