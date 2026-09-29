// Lean compiler output
// Module: Mathlib.LinearAlgebra.TensorProduct.Defs
// Imports: public import Init public meta import Init public import Mathlib.Algebra.Module.Submodule.Bilinear public import Mathlib.GroupTheory.Congruence.Hom public import Mathlib.Tactic.NormNum.Basic
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
lean_object* l_Lean_Name_mkStr2(lean_object*, lean_object*);
lean_object* lp_mathlib_FreeAddMonoid_instAddCancelMonoid(lean_object*);
lean_object* l_Lean_Name_mkStr1(lean_object*);
lean_object* lp_mathlib_AddCon_addMonoid___redArg(lean_object*, lean_object*);
lean_object* lp_mathlib_FreeAddMonoid_lift___redArg(lean_object*);
lean_object* lp_mathlib_AddMonoid_toAddZeroClass___redArg(lean_object*);
lean_object* lp_mathlib_FreeAddMonoid_of___redArg(lean_object*);
lean_object* lp_mathlib_AddCon_mk_x27___redArg(lean_object*, lean_object*);
lean_object* lp_mathlib_Con_lift___redArg___lam__0(lean_object*, lean_object*);
uint8_t l_Lean_Syntax_isOfKind(lean_object*, lean_object*);
lean_object* l_Lean_Syntax_getArg(lean_object*, lean_object*);
lean_object* l_Lean_SourceInfo_fromRef(lean_object*, uint8_t);
lean_object* l_Lean_Name_mkStr4(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_String_toRawSubstring_x27(lean_object*);
lean_object* l_Lean_addMacroScope(lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Syntax_node1(lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Syntax_node2(lean_object*, lean_object*, lean_object*, lean_object*);
uint8_t l_Lean_Syntax_matchesNull(lean_object*, lean_object*);
lean_object* l_Lean_replaceRef(lean_object*, lean_object*);
lean_object* l_Lean_Syntax_node5(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lean_string_length(lean_object*);
lean_object* lp_mathlib_AddZeroClass_toAddZero___redArg(lean_object*);
lean_object* l_Std_instToFormatFormat___lam__0___boxed(lean_object*);
lean_object* lp_mathlib_Equiv_refl(lean_object*);
lean_object* l_List_mapTR_loop___redArg(lean_object*, lean_object*, lean_object*);
uint8_t lean_nat_dec_lt(lean_object*, lean_object*);
lean_object* l_Std_Format_fill(lean_object*);
lean_object* lean_nat_to_int(lean_object*);
lean_object* l_Std_Format_joinSep___redArg(lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Syntax_node3(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_instMulActionNatOfAddMonoid___redArg(lean_object*);
lean_object* lp_mathlib_NSMul_toSMul___redArg___lam__0(lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_LinearMap_mk_u2082_x27_u209b_u2097___redArg___lam__0(lean_object*, lean_object*, lean_object*);
static lean_once_cell_t lp_mathlib_instZeroTensorProduct___aux__1___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_instZeroTensorProduct___aux__1___closed__0;
static lean_once_cell_t lp_mathlib_instZeroTensorProduct___aux__1___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_instZeroTensorProduct___aux__1___closed__1;
static lean_once_cell_t lp_mathlib_instZeroTensorProduct___aux__1___closed__2_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_instZeroTensorProduct___aux__1___closed__2;
LEAN_EXPORT lean_object* lp_mathlib_instZeroTensorProduct___aux__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instZeroTensorProduct___aux__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instZeroTensorProduct(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instZeroTensorProduct___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instAddTensorProduct___aux__1___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instAddTensorProduct___aux__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instAddTensorProduct___aux__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instAddTensorProduct___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instAddTensorProduct(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instAddZeroClassTensorProduct___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instAddZeroClassTensorProduct(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instAddSemigroupTensorProduct___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instAddSemigroupTensorProduct(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_TensorProduct_term___u2297___00__closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 14, .m_capacity = 14, .m_length = 13, .m_data = "TensorProduct"};
static const lean_object* lp_mathlib_TensorProduct_term___u2297___00__closed__0 = (const lean_object*)&lp_mathlib_TensorProduct_term___u2297___00__closed__0_value;
static const lean_string_object lp_mathlib_TensorProduct_term___u2297___00__closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 7, .m_data = "term_⊗_"};
static const lean_object* lp_mathlib_TensorProduct_term___u2297___00__closed__1 = (const lean_object*)&lp_mathlib_TensorProduct_term___u2297___00__closed__1_value;
static const lean_ctor_object lp_mathlib_TensorProduct_term___u2297___00__closed__2_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_TensorProduct_term___u2297___00__closed__0_value),LEAN_SCALAR_PTR_LITERAL(224, 202, 230, 33, 181, 133, 17, 186)}};
static const lean_ctor_object lp_mathlib_TensorProduct_term___u2297___00__closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_TensorProduct_term___u2297___00__closed__2_value_aux_0),((lean_object*)&lp_mathlib_TensorProduct_term___u2297___00__closed__1_value),LEAN_SCALAR_PTR_LITERAL(71, 87, 201, 198, 255, 65, 8, 88)}};
static const lean_object* lp_mathlib_TensorProduct_term___u2297___00__closed__2 = (const lean_object*)&lp_mathlib_TensorProduct_term___u2297___00__closed__2_value;
static const lean_string_object lp_mathlib_TensorProduct_term___u2297___00__closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "andthen"};
static const lean_object* lp_mathlib_TensorProduct_term___u2297___00__closed__3 = (const lean_object*)&lp_mathlib_TensorProduct_term___u2297___00__closed__3_value;
static const lean_ctor_object lp_mathlib_TensorProduct_term___u2297___00__closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_TensorProduct_term___u2297___00__closed__3_value),LEAN_SCALAR_PTR_LITERAL(40, 255, 78, 30, 143, 119, 117, 174)}};
static const lean_object* lp_mathlib_TensorProduct_term___u2297___00__closed__4 = (const lean_object*)&lp_mathlib_TensorProduct_term___u2297___00__closed__4_value;
static const lean_string_object lp_mathlib_TensorProduct_term___u2297___00__closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 3, .m_data = " ⊗ "};
static const lean_object* lp_mathlib_TensorProduct_term___u2297___00__closed__5 = (const lean_object*)&lp_mathlib_TensorProduct_term___u2297___00__closed__5_value;
static const lean_ctor_object lp_mathlib_TensorProduct_term___u2297___00__closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 5}, .m_objs = {((lean_object*)&lp_mathlib_TensorProduct_term___u2297___00__closed__5_value)}};
static const lean_object* lp_mathlib_TensorProduct_term___u2297___00__closed__6 = (const lean_object*)&lp_mathlib_TensorProduct_term___u2297___00__closed__6_value;
static const lean_string_object lp_mathlib_TensorProduct_term___u2297___00__closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "term"};
static const lean_object* lp_mathlib_TensorProduct_term___u2297___00__closed__7 = (const lean_object*)&lp_mathlib_TensorProduct_term___u2297___00__closed__7_value;
static const lean_ctor_object lp_mathlib_TensorProduct_term___u2297___00__closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_TensorProduct_term___u2297___00__closed__7_value),LEAN_SCALAR_PTR_LITERAL(187, 230, 181, 162, 253, 146, 122, 119)}};
static const lean_object* lp_mathlib_TensorProduct_term___u2297___00__closed__8 = (const lean_object*)&lp_mathlib_TensorProduct_term___u2297___00__closed__8_value;
static const lean_ctor_object lp_mathlib_TensorProduct_term___u2297___00__closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 7}, .m_objs = {((lean_object*)&lp_mathlib_TensorProduct_term___u2297___00__closed__8_value),((lean_object*)(((size_t)(101) << 1) | 1))}};
static const lean_object* lp_mathlib_TensorProduct_term___u2297___00__closed__9 = (const lean_object*)&lp_mathlib_TensorProduct_term___u2297___00__closed__9_value;
static const lean_ctor_object lp_mathlib_TensorProduct_term___u2297___00__closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib_TensorProduct_term___u2297___00__closed__4_value),((lean_object*)&lp_mathlib_TensorProduct_term___u2297___00__closed__6_value),((lean_object*)&lp_mathlib_TensorProduct_term___u2297___00__closed__9_value)}};
static const lean_object* lp_mathlib_TensorProduct_term___u2297___00__closed__10 = (const lean_object*)&lp_mathlib_TensorProduct_term___u2297___00__closed__10_value;
static const lean_ctor_object lp_mathlib_TensorProduct_term___u2297___00__closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*4 + 0, .m_other = 4, .m_tag = 4}, .m_objs = {((lean_object*)&lp_mathlib_TensorProduct_term___u2297___00__closed__2_value),((lean_object*)(((size_t)(100) << 1) | 1)),((lean_object*)(((size_t)(100) << 1) | 1)),((lean_object*)&lp_mathlib_TensorProduct_term___u2297___00__closed__10_value)}};
static const lean_object* lp_mathlib_TensorProduct_term___u2297___00__closed__11 = (const lean_object*)&lp_mathlib_TensorProduct_term___u2297___00__closed__11_value;
LEAN_EXPORT const lean_object* lp_mathlib_TensorProduct_term___u2297__ = (const lean_object*)&lp_mathlib_TensorProduct_term___u2297___00__closed__11_value;
static const lean_string_object lp_mathlib_TensorProduct___aux__Mathlib__LinearAlgebra__TensorProduct__Defs______macroRules__TensorProduct__term___u2297____1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "Lean"};
static const lean_object* lp_mathlib_TensorProduct___aux__Mathlib__LinearAlgebra__TensorProduct__Defs______macroRules__TensorProduct__term___u2297____1___closed__0 = (const lean_object*)&lp_mathlib_TensorProduct___aux__Mathlib__LinearAlgebra__TensorProduct__Defs______macroRules__TensorProduct__term___u2297____1___closed__0_value;
static const lean_string_object lp_mathlib_TensorProduct___aux__Mathlib__LinearAlgebra__TensorProduct__Defs______macroRules__TensorProduct__term___u2297____1___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "Parser"};
static const lean_object* lp_mathlib_TensorProduct___aux__Mathlib__LinearAlgebra__TensorProduct__Defs______macroRules__TensorProduct__term___u2297____1___closed__1 = (const lean_object*)&lp_mathlib_TensorProduct___aux__Mathlib__LinearAlgebra__TensorProduct__Defs______macroRules__TensorProduct__term___u2297____1___closed__1_value;
static const lean_string_object lp_mathlib_TensorProduct___aux__Mathlib__LinearAlgebra__TensorProduct__Defs______macroRules__TensorProduct__term___u2297____1___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "Term"};
static const lean_object* lp_mathlib_TensorProduct___aux__Mathlib__LinearAlgebra__TensorProduct__Defs______macroRules__TensorProduct__term___u2297____1___closed__2 = (const lean_object*)&lp_mathlib_TensorProduct___aux__Mathlib__LinearAlgebra__TensorProduct__Defs______macroRules__TensorProduct__term___u2297____1___closed__2_value;
static const lean_string_object lp_mathlib_TensorProduct___aux__Mathlib__LinearAlgebra__TensorProduct__Defs______macroRules__TensorProduct__term___u2297____1___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = "app"};
static const lean_object* lp_mathlib_TensorProduct___aux__Mathlib__LinearAlgebra__TensorProduct__Defs______macroRules__TensorProduct__term___u2297____1___closed__3 = (const lean_object*)&lp_mathlib_TensorProduct___aux__Mathlib__LinearAlgebra__TensorProduct__Defs______macroRules__TensorProduct__term___u2297____1___closed__3_value;
static const lean_ctor_object lp_mathlib_TensorProduct___aux__Mathlib__LinearAlgebra__TensorProduct__Defs______macroRules__TensorProduct__term___u2297____1___closed__4_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_TensorProduct___aux__Mathlib__LinearAlgebra__TensorProduct__Defs______macroRules__TensorProduct__term___u2297____1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib_TensorProduct___aux__Mathlib__LinearAlgebra__TensorProduct__Defs______macroRules__TensorProduct__term___u2297____1___closed__4_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_TensorProduct___aux__Mathlib__LinearAlgebra__TensorProduct__Defs______macroRules__TensorProduct__term___u2297____1___closed__4_value_aux_0),((lean_object*)&lp_mathlib_TensorProduct___aux__Mathlib__LinearAlgebra__TensorProduct__Defs______macroRules__TensorProduct__term___u2297____1___closed__1_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib_TensorProduct___aux__Mathlib__LinearAlgebra__TensorProduct__Defs______macroRules__TensorProduct__term___u2297____1___closed__4_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_TensorProduct___aux__Mathlib__LinearAlgebra__TensorProduct__Defs______macroRules__TensorProduct__term___u2297____1___closed__4_value_aux_1),((lean_object*)&lp_mathlib_TensorProduct___aux__Mathlib__LinearAlgebra__TensorProduct__Defs______macroRules__TensorProduct__term___u2297____1___closed__2_value),LEAN_SCALAR_PTR_LITERAL(75, 170, 162, 138, 136, 204, 251, 229)}};
static const lean_ctor_object lp_mathlib_TensorProduct___aux__Mathlib__LinearAlgebra__TensorProduct__Defs______macroRules__TensorProduct__term___u2297____1___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_TensorProduct___aux__Mathlib__LinearAlgebra__TensorProduct__Defs______macroRules__TensorProduct__term___u2297____1___closed__4_value_aux_2),((lean_object*)&lp_mathlib_TensorProduct___aux__Mathlib__LinearAlgebra__TensorProduct__Defs______macroRules__TensorProduct__term___u2297____1___closed__3_value),LEAN_SCALAR_PTR_LITERAL(69, 118, 10, 41, 220, 156, 243, 179)}};
static const lean_object* lp_mathlib_TensorProduct___aux__Mathlib__LinearAlgebra__TensorProduct__Defs______macroRules__TensorProduct__term___u2297____1___closed__4 = (const lean_object*)&lp_mathlib_TensorProduct___aux__Mathlib__LinearAlgebra__TensorProduct__Defs______macroRules__TensorProduct__term___u2297____1___closed__4_value;
static lean_once_cell_t lp_mathlib_TensorProduct___aux__Mathlib__LinearAlgebra__TensorProduct__Defs______macroRules__TensorProduct__term___u2297____1___closed__5_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_TensorProduct___aux__Mathlib__LinearAlgebra__TensorProduct__Defs______macroRules__TensorProduct__term___u2297____1___closed__5;
static const lean_ctor_object lp_mathlib_TensorProduct___aux__Mathlib__LinearAlgebra__TensorProduct__Defs______macroRules__TensorProduct__term___u2297____1___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_TensorProduct_term___u2297___00__closed__0_value),LEAN_SCALAR_PTR_LITERAL(224, 202, 230, 33, 181, 133, 17, 186)}};
static const lean_object* lp_mathlib_TensorProduct___aux__Mathlib__LinearAlgebra__TensorProduct__Defs______macroRules__TensorProduct__term___u2297____1___closed__6 = (const lean_object*)&lp_mathlib_TensorProduct___aux__Mathlib__LinearAlgebra__TensorProduct__Defs______macroRules__TensorProduct__term___u2297____1___closed__6_value;
static const lean_ctor_object lp_mathlib_TensorProduct___aux__Mathlib__LinearAlgebra__TensorProduct__Defs______macroRules__TensorProduct__term___u2297____1___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_TensorProduct___aux__Mathlib__LinearAlgebra__TensorProduct__Defs______macroRules__TensorProduct__term___u2297____1___closed__6_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_TensorProduct___aux__Mathlib__LinearAlgebra__TensorProduct__Defs______macroRules__TensorProduct__term___u2297____1___closed__7 = (const lean_object*)&lp_mathlib_TensorProduct___aux__Mathlib__LinearAlgebra__TensorProduct__Defs______macroRules__TensorProduct__term___u2297____1___closed__7_value;
static const lean_ctor_object lp_mathlib_TensorProduct___aux__Mathlib__LinearAlgebra__TensorProduct__Defs______macroRules__TensorProduct__term___u2297____1___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_TensorProduct___aux__Mathlib__LinearAlgebra__TensorProduct__Defs______macroRules__TensorProduct__term___u2297____1___closed__6_value)}};
static const lean_object* lp_mathlib_TensorProduct___aux__Mathlib__LinearAlgebra__TensorProduct__Defs______macroRules__TensorProduct__term___u2297____1___closed__8 = (const lean_object*)&lp_mathlib_TensorProduct___aux__Mathlib__LinearAlgebra__TensorProduct__Defs______macroRules__TensorProduct__term___u2297____1___closed__8_value;
static const lean_ctor_object lp_mathlib_TensorProduct___aux__Mathlib__LinearAlgebra__TensorProduct__Defs______macroRules__TensorProduct__term___u2297____1___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_TensorProduct___aux__Mathlib__LinearAlgebra__TensorProduct__Defs______macroRules__TensorProduct__term___u2297____1___closed__8_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_TensorProduct___aux__Mathlib__LinearAlgebra__TensorProduct__Defs______macroRules__TensorProduct__term___u2297____1___closed__9 = (const lean_object*)&lp_mathlib_TensorProduct___aux__Mathlib__LinearAlgebra__TensorProduct__Defs______macroRules__TensorProduct__term___u2297____1___closed__9_value;
static const lean_ctor_object lp_mathlib_TensorProduct___aux__Mathlib__LinearAlgebra__TensorProduct__Defs______macroRules__TensorProduct__term___u2297____1___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_TensorProduct___aux__Mathlib__LinearAlgebra__TensorProduct__Defs______macroRules__TensorProduct__term___u2297____1___closed__7_value),((lean_object*)&lp_mathlib_TensorProduct___aux__Mathlib__LinearAlgebra__TensorProduct__Defs______macroRules__TensorProduct__term___u2297____1___closed__9_value)}};
static const lean_object* lp_mathlib_TensorProduct___aux__Mathlib__LinearAlgebra__TensorProduct__Defs______macroRules__TensorProduct__term___u2297____1___closed__10 = (const lean_object*)&lp_mathlib_TensorProduct___aux__Mathlib__LinearAlgebra__TensorProduct__Defs______macroRules__TensorProduct__term___u2297____1___closed__10_value;
static const lean_string_object lp_mathlib_TensorProduct___aux__Mathlib__LinearAlgebra__TensorProduct__Defs______macroRules__TensorProduct__term___u2297____1___closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "null"};
static const lean_object* lp_mathlib_TensorProduct___aux__Mathlib__LinearAlgebra__TensorProduct__Defs______macroRules__TensorProduct__term___u2297____1___closed__11 = (const lean_object*)&lp_mathlib_TensorProduct___aux__Mathlib__LinearAlgebra__TensorProduct__Defs______macroRules__TensorProduct__term___u2297____1___closed__11_value;
static const lean_ctor_object lp_mathlib_TensorProduct___aux__Mathlib__LinearAlgebra__TensorProduct__Defs______macroRules__TensorProduct__term___u2297____1___closed__12_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_TensorProduct___aux__Mathlib__LinearAlgebra__TensorProduct__Defs______macroRules__TensorProduct__term___u2297____1___closed__11_value),LEAN_SCALAR_PTR_LITERAL(24, 58, 49, 223, 146, 207, 197, 136)}};
static const lean_object* lp_mathlib_TensorProduct___aux__Mathlib__LinearAlgebra__TensorProduct__Defs______macroRules__TensorProduct__term___u2297____1___closed__12 = (const lean_object*)&lp_mathlib_TensorProduct___aux__Mathlib__LinearAlgebra__TensorProduct__Defs______macroRules__TensorProduct__term___u2297____1___closed__12_value;
static const lean_string_object lp_mathlib_TensorProduct___aux__Mathlib__LinearAlgebra__TensorProduct__Defs______macroRules__TensorProduct__term___u2297____1___closed__13_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "hole"};
static const lean_object* lp_mathlib_TensorProduct___aux__Mathlib__LinearAlgebra__TensorProduct__Defs______macroRules__TensorProduct__term___u2297____1___closed__13 = (const lean_object*)&lp_mathlib_TensorProduct___aux__Mathlib__LinearAlgebra__TensorProduct__Defs______macroRules__TensorProduct__term___u2297____1___closed__13_value;
static const lean_ctor_object lp_mathlib_TensorProduct___aux__Mathlib__LinearAlgebra__TensorProduct__Defs______macroRules__TensorProduct__term___u2297____1___closed__14_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_TensorProduct___aux__Mathlib__LinearAlgebra__TensorProduct__Defs______macroRules__TensorProduct__term___u2297____1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib_TensorProduct___aux__Mathlib__LinearAlgebra__TensorProduct__Defs______macroRules__TensorProduct__term___u2297____1___closed__14_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_TensorProduct___aux__Mathlib__LinearAlgebra__TensorProduct__Defs______macroRules__TensorProduct__term___u2297____1___closed__14_value_aux_0),((lean_object*)&lp_mathlib_TensorProduct___aux__Mathlib__LinearAlgebra__TensorProduct__Defs______macroRules__TensorProduct__term___u2297____1___closed__1_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib_TensorProduct___aux__Mathlib__LinearAlgebra__TensorProduct__Defs______macroRules__TensorProduct__term___u2297____1___closed__14_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_TensorProduct___aux__Mathlib__LinearAlgebra__TensorProduct__Defs______macroRules__TensorProduct__term___u2297____1___closed__14_value_aux_1),((lean_object*)&lp_mathlib_TensorProduct___aux__Mathlib__LinearAlgebra__TensorProduct__Defs______macroRules__TensorProduct__term___u2297____1___closed__2_value),LEAN_SCALAR_PTR_LITERAL(75, 170, 162, 138, 136, 204, 251, 229)}};
static const lean_ctor_object lp_mathlib_TensorProduct___aux__Mathlib__LinearAlgebra__TensorProduct__Defs______macroRules__TensorProduct__term___u2297____1___closed__14_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_TensorProduct___aux__Mathlib__LinearAlgebra__TensorProduct__Defs______macroRules__TensorProduct__term___u2297____1___closed__14_value_aux_2),((lean_object*)&lp_mathlib_TensorProduct___aux__Mathlib__LinearAlgebra__TensorProduct__Defs______macroRules__TensorProduct__term___u2297____1___closed__13_value),LEAN_SCALAR_PTR_LITERAL(135, 134, 219, 115, 97, 130, 74, 55)}};
static const lean_object* lp_mathlib_TensorProduct___aux__Mathlib__LinearAlgebra__TensorProduct__Defs______macroRules__TensorProduct__term___u2297____1___closed__14 = (const lean_object*)&lp_mathlib_TensorProduct___aux__Mathlib__LinearAlgebra__TensorProduct__Defs______macroRules__TensorProduct__term___u2297____1___closed__14_value;
static const lean_string_object lp_mathlib_TensorProduct___aux__Mathlib__LinearAlgebra__TensorProduct__Defs______macroRules__TensorProduct__term___u2297____1___closed__15_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = "_"};
static const lean_object* lp_mathlib_TensorProduct___aux__Mathlib__LinearAlgebra__TensorProduct__Defs______macroRules__TensorProduct__term___u2297____1___closed__15 = (const lean_object*)&lp_mathlib_TensorProduct___aux__Mathlib__LinearAlgebra__TensorProduct__Defs______macroRules__TensorProduct__term___u2297____1___closed__15_value;
LEAN_EXPORT lean_object* lp_mathlib_TensorProduct___aux__Mathlib__LinearAlgebra__TensorProduct__Defs______macroRules__TensorProduct__term___u2297____1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_TensorProduct___aux__Mathlib__LinearAlgebra__TensorProduct__Defs______macroRules__TensorProduct__term___u2297____1___boxed(lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_TensorProduct_term___u2297_x5b___x5d___00__closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 13, .m_capacity = 13, .m_length = 10, .m_data = "term_⊗[_]_"};
static const lean_object* lp_mathlib_TensorProduct_term___u2297_x5b___x5d___00__closed__0 = (const lean_object*)&lp_mathlib_TensorProduct_term___u2297_x5b___x5d___00__closed__0_value;
static const lean_ctor_object lp_mathlib_TensorProduct_term___u2297_x5b___x5d___00__closed__1_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_TensorProduct_term___u2297___00__closed__0_value),LEAN_SCALAR_PTR_LITERAL(224, 202, 230, 33, 181, 133, 17, 186)}};
static const lean_ctor_object lp_mathlib_TensorProduct_term___u2297_x5b___x5d___00__closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_TensorProduct_term___u2297_x5b___x5d___00__closed__1_value_aux_0),((lean_object*)&lp_mathlib_TensorProduct_term___u2297_x5b___x5d___00__closed__0_value),LEAN_SCALAR_PTR_LITERAL(66, 68, 222, 189, 126, 193, 83, 51)}};
static const lean_object* lp_mathlib_TensorProduct_term___u2297_x5b___x5d___00__closed__1 = (const lean_object*)&lp_mathlib_TensorProduct_term___u2297_x5b___x5d___00__closed__1_value;
static const lean_string_object lp_mathlib_TensorProduct_term___u2297_x5b___x5d___00__closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 3, .m_data = " ⊗["};
static const lean_object* lp_mathlib_TensorProduct_term___u2297_x5b___x5d___00__closed__2 = (const lean_object*)&lp_mathlib_TensorProduct_term___u2297_x5b___x5d___00__closed__2_value;
static const lean_ctor_object lp_mathlib_TensorProduct_term___u2297_x5b___x5d___00__closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 5}, .m_objs = {((lean_object*)&lp_mathlib_TensorProduct_term___u2297_x5b___x5d___00__closed__2_value)}};
static const lean_object* lp_mathlib_TensorProduct_term___u2297_x5b___x5d___00__closed__3 = (const lean_object*)&lp_mathlib_TensorProduct_term___u2297_x5b___x5d___00__closed__3_value;
static const lean_ctor_object lp_mathlib_TensorProduct_term___u2297_x5b___x5d___00__closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 7}, .m_objs = {((lean_object*)&lp_mathlib_TensorProduct_term___u2297___00__closed__8_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_TensorProduct_term___u2297_x5b___x5d___00__closed__4 = (const lean_object*)&lp_mathlib_TensorProduct_term___u2297_x5b___x5d___00__closed__4_value;
static const lean_ctor_object lp_mathlib_TensorProduct_term___u2297_x5b___x5d___00__closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib_TensorProduct_term___u2297___00__closed__4_value),((lean_object*)&lp_mathlib_TensorProduct_term___u2297_x5b___x5d___00__closed__3_value),((lean_object*)&lp_mathlib_TensorProduct_term___u2297_x5b___x5d___00__closed__4_value)}};
static const lean_object* lp_mathlib_TensorProduct_term___u2297_x5b___x5d___00__closed__5 = (const lean_object*)&lp_mathlib_TensorProduct_term___u2297_x5b___x5d___00__closed__5_value;
static const lean_string_object lp_mathlib_TensorProduct_term___u2297_x5b___x5d___00__closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 3, .m_capacity = 3, .m_length = 2, .m_data = "] "};
static const lean_object* lp_mathlib_TensorProduct_term___u2297_x5b___x5d___00__closed__6 = (const lean_object*)&lp_mathlib_TensorProduct_term___u2297_x5b___x5d___00__closed__6_value;
static const lean_ctor_object lp_mathlib_TensorProduct_term___u2297_x5b___x5d___00__closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 5}, .m_objs = {((lean_object*)&lp_mathlib_TensorProduct_term___u2297_x5b___x5d___00__closed__6_value)}};
static const lean_object* lp_mathlib_TensorProduct_term___u2297_x5b___x5d___00__closed__7 = (const lean_object*)&lp_mathlib_TensorProduct_term___u2297_x5b___x5d___00__closed__7_value;
static const lean_ctor_object lp_mathlib_TensorProduct_term___u2297_x5b___x5d___00__closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib_TensorProduct_term___u2297___00__closed__4_value),((lean_object*)&lp_mathlib_TensorProduct_term___u2297_x5b___x5d___00__closed__5_value),((lean_object*)&lp_mathlib_TensorProduct_term___u2297_x5b___x5d___00__closed__7_value)}};
static const lean_object* lp_mathlib_TensorProduct_term___u2297_x5b___x5d___00__closed__8 = (const lean_object*)&lp_mathlib_TensorProduct_term___u2297_x5b___x5d___00__closed__8_value;
static const lean_ctor_object lp_mathlib_TensorProduct_term___u2297_x5b___x5d___00__closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib_TensorProduct_term___u2297___00__closed__4_value),((lean_object*)&lp_mathlib_TensorProduct_term___u2297_x5b___x5d___00__closed__8_value),((lean_object*)&lp_mathlib_TensorProduct_term___u2297___00__closed__9_value)}};
static const lean_object* lp_mathlib_TensorProduct_term___u2297_x5b___x5d___00__closed__9 = (const lean_object*)&lp_mathlib_TensorProduct_term___u2297_x5b___x5d___00__closed__9_value;
static const lean_ctor_object lp_mathlib_TensorProduct_term___u2297_x5b___x5d___00__closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*4 + 0, .m_other = 4, .m_tag = 4}, .m_objs = {((lean_object*)&lp_mathlib_TensorProduct_term___u2297_x5b___x5d___00__closed__1_value),((lean_object*)(((size_t)(100) << 1) | 1)),((lean_object*)(((size_t)(100) << 1) | 1)),((lean_object*)&lp_mathlib_TensorProduct_term___u2297_x5b___x5d___00__closed__9_value)}};
static const lean_object* lp_mathlib_TensorProduct_term___u2297_x5b___x5d___00__closed__10 = (const lean_object*)&lp_mathlib_TensorProduct_term___u2297_x5b___x5d___00__closed__10_value;
LEAN_EXPORT const lean_object* lp_mathlib_TensorProduct_term___u2297_x5b___x5d__ = (const lean_object*)&lp_mathlib_TensorProduct_term___u2297_x5b___x5d___00__closed__10_value;
LEAN_EXPORT lean_object* lp_mathlib_TensorProduct___aux__Mathlib__LinearAlgebra__TensorProduct__Defs______macroRules__TensorProduct__term___u2297_x5b___x5d____1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_TensorProduct___aux__Mathlib__LinearAlgebra__TensorProduct__Defs______macroRules__TensorProduct__term___u2297_x5b___x5d____1___boxed(lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_TensorProduct___aux__Mathlib__LinearAlgebra__TensorProduct__Defs______unexpand__TensorProduct__1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "ident"};
static const lean_object* lp_mathlib_TensorProduct___aux__Mathlib__LinearAlgebra__TensorProduct__Defs______unexpand__TensorProduct__1___closed__0 = (const lean_object*)&lp_mathlib_TensorProduct___aux__Mathlib__LinearAlgebra__TensorProduct__Defs______unexpand__TensorProduct__1___closed__0_value;
static const lean_ctor_object lp_mathlib_TensorProduct___aux__Mathlib__LinearAlgebra__TensorProduct__Defs______unexpand__TensorProduct__1___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_TensorProduct___aux__Mathlib__LinearAlgebra__TensorProduct__Defs______unexpand__TensorProduct__1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(52, 159, 208, 51, 14, 60, 6, 71)}};
static const lean_object* lp_mathlib_TensorProduct___aux__Mathlib__LinearAlgebra__TensorProduct__Defs______unexpand__TensorProduct__1___closed__1 = (const lean_object*)&lp_mathlib_TensorProduct___aux__Mathlib__LinearAlgebra__TensorProduct__Defs______unexpand__TensorProduct__1___closed__1_value;
LEAN_EXPORT lean_object* lp_mathlib_TensorProduct___aux__Mathlib__LinearAlgebra__TensorProduct__Defs______unexpand__TensorProduct__1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_TensorProduct___aux__Mathlib__LinearAlgebra__TensorProduct__Defs______unexpand__TensorProduct__1___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_TensorProduct_addCommSemigroup___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_TensorProduct_addCommSemigroup(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_TensorProduct_instInhabited(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_TensorProduct_instInhabited___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_TensorProduct_tmul___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_TensorProduct_tmul(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_TensorProduct_tmul___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_TensorProduct_term___u2297_u209c___00__closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 13, .m_capacity = 13, .m_length = 8, .m_data = "term_⊗ₜ_"};
static const lean_object* lp_mathlib_TensorProduct_term___u2297_u209c___00__closed__0 = (const lean_object*)&lp_mathlib_TensorProduct_term___u2297_u209c___00__closed__0_value;
static const lean_ctor_object lp_mathlib_TensorProduct_term___u2297_u209c___00__closed__1_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_TensorProduct_term___u2297___00__closed__0_value),LEAN_SCALAR_PTR_LITERAL(224, 202, 230, 33, 181, 133, 17, 186)}};
static const lean_ctor_object lp_mathlib_TensorProduct_term___u2297_u209c___00__closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_TensorProduct_term___u2297_u209c___00__closed__1_value_aux_0),((lean_object*)&lp_mathlib_TensorProduct_term___u2297_u209c___00__closed__0_value),LEAN_SCALAR_PTR_LITERAL(133, 154, 50, 102, 24, 222, 53, 153)}};
static const lean_object* lp_mathlib_TensorProduct_term___u2297_u209c___00__closed__1 = (const lean_object*)&lp_mathlib_TensorProduct_term___u2297_u209c___00__closed__1_value;
static const lean_string_object lp_mathlib_TensorProduct_term___u2297_u209c___00__closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 4, .m_data = " ⊗ₜ "};
static const lean_object* lp_mathlib_TensorProduct_term___u2297_u209c___00__closed__2 = (const lean_object*)&lp_mathlib_TensorProduct_term___u2297_u209c___00__closed__2_value;
static const lean_ctor_object lp_mathlib_TensorProduct_term___u2297_u209c___00__closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 5}, .m_objs = {((lean_object*)&lp_mathlib_TensorProduct_term___u2297_u209c___00__closed__2_value)}};
static const lean_object* lp_mathlib_TensorProduct_term___u2297_u209c___00__closed__3 = (const lean_object*)&lp_mathlib_TensorProduct_term___u2297_u209c___00__closed__3_value;
static const lean_ctor_object lp_mathlib_TensorProduct_term___u2297_u209c___00__closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib_TensorProduct_term___u2297___00__closed__4_value),((lean_object*)&lp_mathlib_TensorProduct_term___u2297_u209c___00__closed__3_value),((lean_object*)&lp_mathlib_TensorProduct_term___u2297___00__closed__9_value)}};
static const lean_object* lp_mathlib_TensorProduct_term___u2297_u209c___00__closed__4 = (const lean_object*)&lp_mathlib_TensorProduct_term___u2297_u209c___00__closed__4_value;
static const lean_ctor_object lp_mathlib_TensorProduct_term___u2297_u209c___00__closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*4 + 0, .m_other = 4, .m_tag = 4}, .m_objs = {((lean_object*)&lp_mathlib_TensorProduct_term___u2297_u209c___00__closed__1_value),((lean_object*)(((size_t)(100) << 1) | 1)),((lean_object*)(((size_t)(100) << 1) | 1)),((lean_object*)&lp_mathlib_TensorProduct_term___u2297_u209c___00__closed__4_value)}};
static const lean_object* lp_mathlib_TensorProduct_term___u2297_u209c___00__closed__5 = (const lean_object*)&lp_mathlib_TensorProduct_term___u2297_u209c___00__closed__5_value;
LEAN_EXPORT const lean_object* lp_mathlib_TensorProduct_term___u2297_u209c__ = (const lean_object*)&lp_mathlib_TensorProduct_term___u2297_u209c___00__closed__5_value;
static const lean_string_object lp_mathlib_TensorProduct___aux__Mathlib__LinearAlgebra__TensorProduct__Defs______macroRules__TensorProduct__term___u2297_u209c____1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "tmul"};
static const lean_object* lp_mathlib_TensorProduct___aux__Mathlib__LinearAlgebra__TensorProduct__Defs______macroRules__TensorProduct__term___u2297_u209c____1___closed__0 = (const lean_object*)&lp_mathlib_TensorProduct___aux__Mathlib__LinearAlgebra__TensorProduct__Defs______macroRules__TensorProduct__term___u2297_u209c____1___closed__0_value;
static lean_once_cell_t lp_mathlib_TensorProduct___aux__Mathlib__LinearAlgebra__TensorProduct__Defs______macroRules__TensorProduct__term___u2297_u209c____1___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_TensorProduct___aux__Mathlib__LinearAlgebra__TensorProduct__Defs______macroRules__TensorProduct__term___u2297_u209c____1___closed__1;
static const lean_ctor_object lp_mathlib_TensorProduct___aux__Mathlib__LinearAlgebra__TensorProduct__Defs______macroRules__TensorProduct__term___u2297_u209c____1___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_TensorProduct___aux__Mathlib__LinearAlgebra__TensorProduct__Defs______macroRules__TensorProduct__term___u2297_u209c____1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(254, 68, 102, 118, 210, 239, 53, 130)}};
static const lean_object* lp_mathlib_TensorProduct___aux__Mathlib__LinearAlgebra__TensorProduct__Defs______macroRules__TensorProduct__term___u2297_u209c____1___closed__2 = (const lean_object*)&lp_mathlib_TensorProduct___aux__Mathlib__LinearAlgebra__TensorProduct__Defs______macroRules__TensorProduct__term___u2297_u209c____1___closed__2_value;
static const lean_ctor_object lp_mathlib_TensorProduct___aux__Mathlib__LinearAlgebra__TensorProduct__Defs______macroRules__TensorProduct__term___u2297_u209c____1___closed__3_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_TensorProduct_term___u2297___00__closed__0_value),LEAN_SCALAR_PTR_LITERAL(224, 202, 230, 33, 181, 133, 17, 186)}};
static const lean_ctor_object lp_mathlib_TensorProduct___aux__Mathlib__LinearAlgebra__TensorProduct__Defs______macroRules__TensorProduct__term___u2297_u209c____1___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_TensorProduct___aux__Mathlib__LinearAlgebra__TensorProduct__Defs______macroRules__TensorProduct__term___u2297_u209c____1___closed__3_value_aux_0),((lean_object*)&lp_mathlib_TensorProduct___aux__Mathlib__LinearAlgebra__TensorProduct__Defs______macroRules__TensorProduct__term___u2297_u209c____1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(161, 106, 70, 55, 205, 89, 100, 199)}};
static const lean_object* lp_mathlib_TensorProduct___aux__Mathlib__LinearAlgebra__TensorProduct__Defs______macroRules__TensorProduct__term___u2297_u209c____1___closed__3 = (const lean_object*)&lp_mathlib_TensorProduct___aux__Mathlib__LinearAlgebra__TensorProduct__Defs______macroRules__TensorProduct__term___u2297_u209c____1___closed__3_value;
static const lean_ctor_object lp_mathlib_TensorProduct___aux__Mathlib__LinearAlgebra__TensorProduct__Defs______macroRules__TensorProduct__term___u2297_u209c____1___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_TensorProduct___aux__Mathlib__LinearAlgebra__TensorProduct__Defs______macroRules__TensorProduct__term___u2297_u209c____1___closed__3_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_TensorProduct___aux__Mathlib__LinearAlgebra__TensorProduct__Defs______macroRules__TensorProduct__term___u2297_u209c____1___closed__4 = (const lean_object*)&lp_mathlib_TensorProduct___aux__Mathlib__LinearAlgebra__TensorProduct__Defs______macroRules__TensorProduct__term___u2297_u209c____1___closed__4_value;
static const lean_ctor_object lp_mathlib_TensorProduct___aux__Mathlib__LinearAlgebra__TensorProduct__Defs______macroRules__TensorProduct__term___u2297_u209c____1___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_TensorProduct___aux__Mathlib__LinearAlgebra__TensorProduct__Defs______macroRules__TensorProduct__term___u2297_u209c____1___closed__4_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_TensorProduct___aux__Mathlib__LinearAlgebra__TensorProduct__Defs______macroRules__TensorProduct__term___u2297_u209c____1___closed__5 = (const lean_object*)&lp_mathlib_TensorProduct___aux__Mathlib__LinearAlgebra__TensorProduct__Defs______macroRules__TensorProduct__term___u2297_u209c____1___closed__5_value;
LEAN_EXPORT lean_object* lp_mathlib_TensorProduct___aux__Mathlib__LinearAlgebra__TensorProduct__Defs______macroRules__TensorProduct__term___u2297_u209c____1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_TensorProduct___aux__Mathlib__LinearAlgebra__TensorProduct__Defs______macroRules__TensorProduct__term___u2297_u209c____1___boxed(lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_TensorProduct_term___u2297_u209c_x5b___x5d___00__closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 16, .m_capacity = 16, .m_length = 11, .m_data = "term_⊗ₜ[_]_"};
static const lean_object* lp_mathlib_TensorProduct_term___u2297_u209c_x5b___x5d___00__closed__0 = (const lean_object*)&lp_mathlib_TensorProduct_term___u2297_u209c_x5b___x5d___00__closed__0_value;
static const lean_ctor_object lp_mathlib_TensorProduct_term___u2297_u209c_x5b___x5d___00__closed__1_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_TensorProduct_term___u2297___00__closed__0_value),LEAN_SCALAR_PTR_LITERAL(224, 202, 230, 33, 181, 133, 17, 186)}};
static const lean_ctor_object lp_mathlib_TensorProduct_term___u2297_u209c_x5b___x5d___00__closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_TensorProduct_term___u2297_u209c_x5b___x5d___00__closed__1_value_aux_0),((lean_object*)&lp_mathlib_TensorProduct_term___u2297_u209c_x5b___x5d___00__closed__0_value),LEAN_SCALAR_PTR_LITERAL(252, 41, 140, 1, 13, 62, 145, 239)}};
static const lean_object* lp_mathlib_TensorProduct_term___u2297_u209c_x5b___x5d___00__closed__1 = (const lean_object*)&lp_mathlib_TensorProduct_term___u2297_u209c_x5b___x5d___00__closed__1_value;
static const lean_string_object lp_mathlib_TensorProduct_term___u2297_u209c_x5b___x5d___00__closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 4, .m_data = " ⊗ₜ["};
static const lean_object* lp_mathlib_TensorProduct_term___u2297_u209c_x5b___x5d___00__closed__2 = (const lean_object*)&lp_mathlib_TensorProduct_term___u2297_u209c_x5b___x5d___00__closed__2_value;
static const lean_ctor_object lp_mathlib_TensorProduct_term___u2297_u209c_x5b___x5d___00__closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 5}, .m_objs = {((lean_object*)&lp_mathlib_TensorProduct_term___u2297_u209c_x5b___x5d___00__closed__2_value)}};
static const lean_object* lp_mathlib_TensorProduct_term___u2297_u209c_x5b___x5d___00__closed__3 = (const lean_object*)&lp_mathlib_TensorProduct_term___u2297_u209c_x5b___x5d___00__closed__3_value;
static const lean_ctor_object lp_mathlib_TensorProduct_term___u2297_u209c_x5b___x5d___00__closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib_TensorProduct_term___u2297___00__closed__4_value),((lean_object*)&lp_mathlib_TensorProduct_term___u2297_u209c_x5b___x5d___00__closed__3_value),((lean_object*)&lp_mathlib_TensorProduct_term___u2297_x5b___x5d___00__closed__4_value)}};
static const lean_object* lp_mathlib_TensorProduct_term___u2297_u209c_x5b___x5d___00__closed__4 = (const lean_object*)&lp_mathlib_TensorProduct_term___u2297_u209c_x5b___x5d___00__closed__4_value;
static const lean_ctor_object lp_mathlib_TensorProduct_term___u2297_u209c_x5b___x5d___00__closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib_TensorProduct_term___u2297___00__closed__4_value),((lean_object*)&lp_mathlib_TensorProduct_term___u2297_u209c_x5b___x5d___00__closed__4_value),((lean_object*)&lp_mathlib_TensorProduct_term___u2297_x5b___x5d___00__closed__7_value)}};
static const lean_object* lp_mathlib_TensorProduct_term___u2297_u209c_x5b___x5d___00__closed__5 = (const lean_object*)&lp_mathlib_TensorProduct_term___u2297_u209c_x5b___x5d___00__closed__5_value;
static const lean_ctor_object lp_mathlib_TensorProduct_term___u2297_u209c_x5b___x5d___00__closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib_TensorProduct_term___u2297___00__closed__4_value),((lean_object*)&lp_mathlib_TensorProduct_term___u2297_u209c_x5b___x5d___00__closed__5_value),((lean_object*)&lp_mathlib_TensorProduct_term___u2297___00__closed__9_value)}};
static const lean_object* lp_mathlib_TensorProduct_term___u2297_u209c_x5b___x5d___00__closed__6 = (const lean_object*)&lp_mathlib_TensorProduct_term___u2297_u209c_x5b___x5d___00__closed__6_value;
static const lean_ctor_object lp_mathlib_TensorProduct_term___u2297_u209c_x5b___x5d___00__closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*4 + 0, .m_other = 4, .m_tag = 4}, .m_objs = {((lean_object*)&lp_mathlib_TensorProduct_term___u2297_u209c_x5b___x5d___00__closed__1_value),((lean_object*)(((size_t)(100) << 1) | 1)),((lean_object*)(((size_t)(100) << 1) | 1)),((lean_object*)&lp_mathlib_TensorProduct_term___u2297_u209c_x5b___x5d___00__closed__6_value)}};
static const lean_object* lp_mathlib_TensorProduct_term___u2297_u209c_x5b___x5d___00__closed__7 = (const lean_object*)&lp_mathlib_TensorProduct_term___u2297_u209c_x5b___x5d___00__closed__7_value;
LEAN_EXPORT const lean_object* lp_mathlib_TensorProduct_term___u2297_u209c_x5b___x5d__ = (const lean_object*)&lp_mathlib_TensorProduct_term___u2297_u209c_x5b___x5d___00__closed__7_value;
LEAN_EXPORT lean_object* lp_mathlib_TensorProduct___aux__Mathlib__LinearAlgebra__TensorProduct__Defs______macroRules__TensorProduct__term___u2297_u209c_x5b___x5d____1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_TensorProduct___aux__Mathlib__LinearAlgebra__TensorProduct__Defs______macroRules__TensorProduct__term___u2297_u209c_x5b___x5d____1___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_TensorProduct___aux__Mathlib__LinearAlgebra__TensorProduct__Defs______unexpand__TensorProduct__tmul__1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_TensorProduct___aux__Mathlib__LinearAlgebra__TensorProduct__Defs______unexpand__TensorProduct__tmul__1___boxed(lean_object*, lean_object*, lean_object*);
static const lean_ctor_object lp_mathlib_TensorProduct_instRepr___redArg___lam__0___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_mathlib_TensorProduct_term___u2297_u209c___00__closed__2_value)}};
static const lean_object* lp_mathlib_TensorProduct_instRepr___redArg___lam__0___closed__0 = (const lean_object*)&lp_mathlib_TensorProduct_instRepr___redArg___lam__0___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_TensorProduct_instRepr___redArg___lam__0(lean_object*, lean_object*, lean_object*);
static lean_once_cell_t lp_mathlib_TensorProduct_instRepr___redArg___lam__1___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_TensorProduct_instRepr___redArg___lam__1___closed__0;
static const lean_string_object lp_mathlib_TensorProduct_instRepr___redArg___lam__1___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = "0"};
static const lean_object* lp_mathlib_TensorProduct_instRepr___redArg___lam__1___closed__1 = (const lean_object*)&lp_mathlib_TensorProduct_instRepr___redArg___lam__1___closed__1_value;
static const lean_ctor_object lp_mathlib_TensorProduct_instRepr___redArg___lam__1___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_mathlib_TensorProduct_instRepr___redArg___lam__1___closed__1_value)}};
static const lean_object* lp_mathlib_TensorProduct_instRepr___redArg___lam__1___closed__2 = (const lean_object*)&lp_mathlib_TensorProduct_instRepr___redArg___lam__1___closed__2_value;
static const lean_string_object lp_mathlib_TensorProduct_instRepr___redArg___lam__1___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = "("};
static const lean_object* lp_mathlib_TensorProduct_instRepr___redArg___lam__1___closed__3 = (const lean_object*)&lp_mathlib_TensorProduct_instRepr___redArg___lam__1___closed__3_value;
static const lean_string_object lp_mathlib_TensorProduct_instRepr___redArg___lam__1___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = ")"};
static const lean_object* lp_mathlib_TensorProduct_instRepr___redArg___lam__1___closed__4 = (const lean_object*)&lp_mathlib_TensorProduct_instRepr___redArg___lam__1___closed__4_value;
static lean_once_cell_t lp_mathlib_TensorProduct_instRepr___redArg___lam__1___closed__5_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_TensorProduct_instRepr___redArg___lam__1___closed__5;
static lean_once_cell_t lp_mathlib_TensorProduct_instRepr___redArg___lam__1___closed__6_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_TensorProduct_instRepr___redArg___lam__1___closed__6;
static const lean_ctor_object lp_mathlib_TensorProduct_instRepr___redArg___lam__1___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_mathlib_TensorProduct_instRepr___redArg___lam__1___closed__3_value)}};
static const lean_object* lp_mathlib_TensorProduct_instRepr___redArg___lam__1___closed__7 = (const lean_object*)&lp_mathlib_TensorProduct_instRepr___redArg___lam__1___closed__7_value;
static const lean_ctor_object lp_mathlib_TensorProduct_instRepr___redArg___lam__1___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_mathlib_TensorProduct_instRepr___redArg___lam__1___closed__4_value)}};
static const lean_object* lp_mathlib_TensorProduct_instRepr___redArg___lam__1___closed__8 = (const lean_object*)&lp_mathlib_TensorProduct_instRepr___redArg___lam__1___closed__8_value;
static const lean_string_object lp_mathlib_TensorProduct_instRepr___redArg___lam__1___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 3, .m_capacity = 3, .m_length = 2, .m_data = " +"};
static const lean_object* lp_mathlib_TensorProduct_instRepr___redArg___lam__1___closed__9 = (const lean_object*)&lp_mathlib_TensorProduct_instRepr___redArg___lam__1___closed__9_value;
static const lean_ctor_object lp_mathlib_TensorProduct_instRepr___redArg___lam__1___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_mathlib_TensorProduct_instRepr___redArg___lam__1___closed__9_value)}};
static const lean_object* lp_mathlib_TensorProduct_instRepr___redArg___lam__1___closed__10 = (const lean_object*)&lp_mathlib_TensorProduct_instRepr___redArg___lam__1___closed__10_value;
static const lean_ctor_object lp_mathlib_TensorProduct_instRepr___redArg___lam__1___closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 5}, .m_objs = {((lean_object*)&lp_mathlib_TensorProduct_instRepr___redArg___lam__1___closed__10_value),((lean_object*)(((size_t)(1) << 1) | 1))}};
static const lean_object* lp_mathlib_TensorProduct_instRepr___redArg___lam__1___closed__11 = (const lean_object*)&lp_mathlib_TensorProduct_instRepr___redArg___lam__1___closed__11_value;
LEAN_EXPORT lean_object* lp_mathlib_TensorProduct_instRepr___redArg___lam__1(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_TensorProduct_instRepr___redArg___lam__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_TensorProduct_instRepr___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Std_instToFormatFormat___lam__0___boxed, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_TensorProduct_instRepr___redArg___closed__0 = (const lean_object*)&lp_mathlib_TensorProduct_instRepr___redArg___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_TensorProduct_instRepr___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_TensorProduct_instRepr(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_TensorProduct_instRepr___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_TensorProduct_uniqueLeft(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_TensorProduct_uniqueLeft___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_TensorProduct_uniqueRight(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_TensorProduct_uniqueRight___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_once_cell_t lp_mathlib___private_Mathlib_LinearAlgebra_TensorProduct_Defs_0__TensorProduct_addMonoidWithWrongNSMul___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_LinearAlgebra_TensorProduct_Defs_0__TensorProduct_addMonoidWithWrongNSMul___closed__0;
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_LinearAlgebra_TensorProduct_Defs_0__TensorProduct_addMonoidWithWrongNSMul(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_LinearAlgebra_TensorProduct_Defs_0__TensorProduct_addMonoidWithWrongNSMul___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_TensorProduct_SMul_aux___redArg___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_TensorProduct_SMul_aux___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_TensorProduct_SMul_aux___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_TensorProduct_SMul_aux(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_TensorProduct_SMul_aux___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_TensorProduct_leftHasSMul___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_TensorProduct_leftHasSMul___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_TensorProduct_leftHasSMul___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_TensorProduct_leftHasSMul(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_TensorProduct_leftHasSMul___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_TensorProduct_instSMul___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_TensorProduct_instSMul(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_TensorProduct_addMonoid___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_TensorProduct_addMonoid(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_TensorProduct_addCommMonoid___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_TensorProduct_addCommMonoid(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_TensorProduct_leftDistribMulAction___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_TensorProduct_leftDistribMulAction(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_TensorProduct_leftDistribMulAction___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_TensorProduct_instDistribMulAction___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_TensorProduct_instDistribMulAction(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_TensorProduct_leftModule___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_TensorProduct_leftModule(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_TensorProduct_leftModule___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_TensorProduct_instModule___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_TensorProduct_instModule(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_TensorProduct_mk___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_TensorProduct_tmul___redArg, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_TensorProduct_mk___closed__0 = (const lean_object*)&lp_mathlib_TensorProduct_mk___closed__0_value;
static const lean_closure_object lp_mathlib_TensorProduct_mk___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*1, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_LinearMap_mk_u2082_x27_u209b_u2097___redArg___lam__0, .m_arity = 3, .m_num_fixed = 1, .m_objs = {((lean_object*)&lp_mathlib_TensorProduct_mk___closed__0_value)} };
static const lean_object* lp_mathlib_TensorProduct_mk___closed__1 = (const lean_object*)&lp_mathlib_TensorProduct_mk___closed__1_value;
LEAN_EXPORT lean_object* lp_mathlib_TensorProduct_mk(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_TensorProduct_mk___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_object* _init_lp_mathlib_instZeroTensorProduct___aux__1___closed__0(void){
_start:
{
lean_object* v___x_1_; 
v___x_1_ = lp_mathlib_FreeAddMonoid_instAddCancelMonoid(lean_box(0));
return v___x_1_;
}
}
static lean_object* _init_lp_mathlib_instZeroTensorProduct___aux__1___closed__1(void){
_start:
{
lean_object* v___x_2_; lean_object* v___x_3_; 
v___x_2_ = lean_obj_once(&lp_mathlib_instZeroTensorProduct___aux__1___closed__0, &lp_mathlib_instZeroTensorProduct___aux__1___closed__0_once, _init_lp_mathlib_instZeroTensorProduct___aux__1___closed__0);
v___x_3_ = lp_mathlib_AddMonoid_toAddZeroClass___redArg(v___x_2_);
return v___x_3_;
}
}
static lean_object* _init_lp_mathlib_instZeroTensorProduct___aux__1___closed__2(void){
_start:
{
lean_object* v___x_4_; lean_object* v___x_5_; 
v___x_4_ = lean_obj_once(&lp_mathlib_instZeroTensorProduct___aux__1___closed__1, &lp_mathlib_instZeroTensorProduct___aux__1___closed__1_once, _init_lp_mathlib_instZeroTensorProduct___aux__1___closed__1);
v___x_5_ = lp_mathlib_AddZeroClass_toAddZero___redArg(v___x_4_);
return v___x_5_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instZeroTensorProduct___aux__1(lean_object* v_R_6_, lean_object* v_inst_7_, lean_object* v_M_8_, lean_object* v_N_9_, lean_object* v_inst_10_, lean_object* v_inst_11_, lean_object* v_inst_12_, lean_object* v_inst_13_){
_start:
{
lean_object* v___x_14_; lean_object* v_toZero_15_; 
v___x_14_ = lean_obj_once(&lp_mathlib_instZeroTensorProduct___aux__1___closed__2, &lp_mathlib_instZeroTensorProduct___aux__1___closed__2_once, _init_lp_mathlib_instZeroTensorProduct___aux__1___closed__2);
v_toZero_15_ = lean_ctor_get(v___x_14_, 0);
lean_inc(v_toZero_15_);
return v_toZero_15_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instZeroTensorProduct___aux__1___boxed(lean_object* v_R_16_, lean_object* v_inst_17_, lean_object* v_M_18_, lean_object* v_N_19_, lean_object* v_inst_20_, lean_object* v_inst_21_, lean_object* v_inst_22_, lean_object* v_inst_23_){
_start:
{
lean_object* v_res_24_; 
v_res_24_ = lp_mathlib_instZeroTensorProduct___aux__1(v_R_16_, v_inst_17_, v_M_18_, v_N_19_, v_inst_20_, v_inst_21_, v_inst_22_, v_inst_23_);
lean_dec(v_inst_23_);
lean_dec(v_inst_22_);
lean_dec_ref(v_inst_21_);
lean_dec_ref(v_inst_20_);
lean_dec_ref(v_inst_17_);
return v_res_24_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instZeroTensorProduct(lean_object* v_R_25_, lean_object* v_inst_26_, lean_object* v_M_27_, lean_object* v_N_28_, lean_object* v_inst_29_, lean_object* v_inst_30_, lean_object* v_inst_31_, lean_object* v_inst_32_){
_start:
{
lean_object* v___x_33_; lean_object* v_toZero_34_; 
v___x_33_ = lean_obj_once(&lp_mathlib_instZeroTensorProduct___aux__1___closed__2, &lp_mathlib_instZeroTensorProduct___aux__1___closed__2_once, _init_lp_mathlib_instZeroTensorProduct___aux__1___closed__2);
v_toZero_34_ = lean_ctor_get(v___x_33_, 0);
lean_inc(v_toZero_34_);
return v_toZero_34_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instZeroTensorProduct___boxed(lean_object* v_R_35_, lean_object* v_inst_36_, lean_object* v_M_37_, lean_object* v_N_38_, lean_object* v_inst_39_, lean_object* v_inst_40_, lean_object* v_inst_41_, lean_object* v_inst_42_){
_start:
{
lean_object* v_res_43_; 
v_res_43_ = lp_mathlib_instZeroTensorProduct(v_R_35_, v_inst_36_, v_M_37_, v_N_38_, v_inst_39_, v_inst_40_, v_inst_41_, v_inst_42_);
lean_dec(v_inst_42_);
lean_dec(v_inst_41_);
lean_dec_ref(v_inst_40_);
lean_dec_ref(v_inst_39_);
lean_dec_ref(v_inst_36_);
return v_res_43_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instAddTensorProduct___aux__1___redArg(lean_object* v_a_44_, lean_object* v_a_45_){
_start:
{
lean_object* v___x_46_; lean_object* v_toAdd_47_; lean_object* v___x_48_; 
v___x_46_ = lean_obj_once(&lp_mathlib_instZeroTensorProduct___aux__1___closed__0, &lp_mathlib_instZeroTensorProduct___aux__1___closed__0_once, _init_lp_mathlib_instZeroTensorProduct___aux__1___closed__0);
v_toAdd_47_ = lean_ctor_get(v___x_46_, 1);
lean_inc(v_toAdd_47_);
v___x_48_ = lean_apply_2(v_toAdd_47_, v_a_44_, v_a_45_);
return v___x_48_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instAddTensorProduct___aux__1(lean_object* v_R_49_, lean_object* v_inst_50_, lean_object* v_M_51_, lean_object* v_N_52_, lean_object* v_inst_53_, lean_object* v_inst_54_, lean_object* v_inst_55_, lean_object* v_inst_56_, lean_object* v_a_57_, lean_object* v_a_58_){
_start:
{
lean_object* v___x_59_; lean_object* v_toAdd_60_; lean_object* v___x_61_; 
v___x_59_ = lean_obj_once(&lp_mathlib_instZeroTensorProduct___aux__1___closed__0, &lp_mathlib_instZeroTensorProduct___aux__1___closed__0_once, _init_lp_mathlib_instZeroTensorProduct___aux__1___closed__0);
v_toAdd_60_ = lean_ctor_get(v___x_59_, 1);
lean_inc(v_toAdd_60_);
v___x_61_ = lean_apply_2(v_toAdd_60_, v_a_57_, v_a_58_);
return v___x_61_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instAddTensorProduct___aux__1___boxed(lean_object* v_R_62_, lean_object* v_inst_63_, lean_object* v_M_64_, lean_object* v_N_65_, lean_object* v_inst_66_, lean_object* v_inst_67_, lean_object* v_inst_68_, lean_object* v_inst_69_, lean_object* v_a_70_, lean_object* v_a_71_){
_start:
{
lean_object* v_res_72_; 
v_res_72_ = lp_mathlib_instAddTensorProduct___aux__1(v_R_62_, v_inst_63_, v_M_64_, v_N_65_, v_inst_66_, v_inst_67_, v_inst_68_, v_inst_69_, v_a_70_, v_a_71_);
lean_dec(v_inst_69_);
lean_dec(v_inst_68_);
lean_dec_ref(v_inst_67_);
lean_dec_ref(v_inst_66_);
lean_dec_ref(v_inst_63_);
return v_res_72_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instAddTensorProduct___redArg(lean_object* v_inst_73_, lean_object* v_inst_74_, lean_object* v_inst_75_, lean_object* v_inst_76_, lean_object* v_inst_77_){
_start:
{
lean_object* v___x_78_; 
v___x_78_ = lean_alloc_closure((void*)(lp_mathlib_instAddTensorProduct___aux__1___boxed), 10, 8);
lean_closure_set(v___x_78_, 0, lean_box(0));
lean_closure_set(v___x_78_, 1, v_inst_73_);
lean_closure_set(v___x_78_, 2, lean_box(0));
lean_closure_set(v___x_78_, 3, lean_box(0));
lean_closure_set(v___x_78_, 4, v_inst_74_);
lean_closure_set(v___x_78_, 5, v_inst_75_);
lean_closure_set(v___x_78_, 6, v_inst_76_);
lean_closure_set(v___x_78_, 7, v_inst_77_);
return v___x_78_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instAddTensorProduct(lean_object* v_R_79_, lean_object* v_inst_80_, lean_object* v_M_81_, lean_object* v_N_82_, lean_object* v_inst_83_, lean_object* v_inst_84_, lean_object* v_inst_85_, lean_object* v_inst_86_){
_start:
{
lean_object* v___x_87_; 
v___x_87_ = lean_alloc_closure((void*)(lp_mathlib_instAddTensorProduct___aux__1___boxed), 10, 8);
lean_closure_set(v___x_87_, 0, lean_box(0));
lean_closure_set(v___x_87_, 1, v_inst_80_);
lean_closure_set(v___x_87_, 2, lean_box(0));
lean_closure_set(v___x_87_, 3, lean_box(0));
lean_closure_set(v___x_87_, 4, v_inst_83_);
lean_closure_set(v___x_87_, 5, v_inst_84_);
lean_closure_set(v___x_87_, 6, v_inst_85_);
lean_closure_set(v___x_87_, 7, v_inst_86_);
return v___x_87_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instAddZeroClassTensorProduct___redArg(lean_object* v_inst_88_, lean_object* v_inst_89_, lean_object* v_inst_90_, lean_object* v_inst_91_, lean_object* v_inst_92_){
_start:
{
lean_object* v___x_93_; lean_object* v___x_94_; lean_object* v___x_95_; 
v___x_93_ = lp_mathlib_instZeroTensorProduct(lean_box(0), v_inst_88_, lean_box(0), lean_box(0), v_inst_89_, v_inst_90_, v_inst_91_, v_inst_92_);
v___x_94_ = lean_alloc_closure((void*)(lp_mathlib_instAddTensorProduct___aux__1___boxed), 10, 8);
lean_closure_set(v___x_94_, 0, lean_box(0));
lean_closure_set(v___x_94_, 1, v_inst_88_);
lean_closure_set(v___x_94_, 2, lean_box(0));
lean_closure_set(v___x_94_, 3, lean_box(0));
lean_closure_set(v___x_94_, 4, v_inst_89_);
lean_closure_set(v___x_94_, 5, v_inst_90_);
lean_closure_set(v___x_94_, 6, v_inst_91_);
lean_closure_set(v___x_94_, 7, v_inst_92_);
v___x_95_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_95_, 0, v___x_93_);
lean_ctor_set(v___x_95_, 1, v___x_94_);
return v___x_95_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instAddZeroClassTensorProduct(lean_object* v_R_96_, lean_object* v_inst_97_, lean_object* v_M_98_, lean_object* v_N_99_, lean_object* v_inst_100_, lean_object* v_inst_101_, lean_object* v_inst_102_, lean_object* v_inst_103_){
_start:
{
lean_object* v___x_104_; 
v___x_104_ = lp_mathlib_instAddZeroClassTensorProduct___redArg(v_inst_97_, v_inst_100_, v_inst_101_, v_inst_102_, v_inst_103_);
return v___x_104_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instAddSemigroupTensorProduct___redArg(lean_object* v_inst_105_, lean_object* v_inst_106_, lean_object* v_inst_107_, lean_object* v_inst_108_, lean_object* v_inst_109_){
_start:
{
lean_object* v___x_110_; 
v___x_110_ = lean_alloc_closure((void*)(lp_mathlib_instAddTensorProduct___aux__1___boxed), 10, 8);
lean_closure_set(v___x_110_, 0, lean_box(0));
lean_closure_set(v___x_110_, 1, v_inst_105_);
lean_closure_set(v___x_110_, 2, lean_box(0));
lean_closure_set(v___x_110_, 3, lean_box(0));
lean_closure_set(v___x_110_, 4, v_inst_106_);
lean_closure_set(v___x_110_, 5, v_inst_107_);
lean_closure_set(v___x_110_, 6, v_inst_108_);
lean_closure_set(v___x_110_, 7, v_inst_109_);
return v___x_110_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instAddSemigroupTensorProduct(lean_object* v_R_111_, lean_object* v_inst_112_, lean_object* v_M_113_, lean_object* v_N_114_, lean_object* v_inst_115_, lean_object* v_inst_116_, lean_object* v_inst_117_, lean_object* v_inst_118_){
_start:
{
lean_object* v___x_119_; 
v___x_119_ = lean_alloc_closure((void*)(lp_mathlib_instAddTensorProduct___aux__1___boxed), 10, 8);
lean_closure_set(v___x_119_, 0, lean_box(0));
lean_closure_set(v___x_119_, 1, v_inst_112_);
lean_closure_set(v___x_119_, 2, lean_box(0));
lean_closure_set(v___x_119_, 3, lean_box(0));
lean_closure_set(v___x_119_, 4, v_inst_115_);
lean_closure_set(v___x_119_, 5, v_inst_116_);
lean_closure_set(v___x_119_, 6, v_inst_117_);
lean_closure_set(v___x_119_, 7, v_inst_118_);
return v___x_119_;
}
}
static lean_object* _init_lp_mathlib_TensorProduct___aux__Mathlib__LinearAlgebra__TensorProduct__Defs______macroRules__TensorProduct__term___u2297____1___closed__5(void){
_start:
{
lean_object* v___x_155_; lean_object* v___x_156_; 
v___x_155_ = ((lean_object*)(lp_mathlib_TensorProduct_term___u2297___00__closed__0));
v___x_156_ = l_String_toRawSubstring_x27(v___x_155_);
return v___x_156_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_TensorProduct___aux__Mathlib__LinearAlgebra__TensorProduct__Defs______macroRules__TensorProduct__term___u2297____1(lean_object* v_x_180_, lean_object* v_a_181_, lean_object* v_a_182_){
_start:
{
lean_object* v___x_183_; uint8_t v___x_184_; 
v___x_183_ = ((lean_object*)(lp_mathlib_TensorProduct_term___u2297___00__closed__2));
lean_inc(v_x_180_);
v___x_184_ = l_Lean_Syntax_isOfKind(v_x_180_, v___x_183_);
if (v___x_184_ == 0)
{
lean_object* v___x_185_; lean_object* v___x_186_; 
lean_dec(v_x_180_);
v___x_185_ = lean_box(1);
v___x_186_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_186_, 0, v___x_185_);
lean_ctor_set(v___x_186_, 1, v_a_182_);
return v___x_186_;
}
else
{
lean_object* v_quotContext_187_; lean_object* v_currMacroScope_188_; lean_object* v_ref_189_; lean_object* v___x_190_; lean_object* v___x_191_; lean_object* v___x_192_; lean_object* v___x_193_; uint8_t v___x_194_; lean_object* v___x_195_; lean_object* v___x_196_; lean_object* v___x_197_; lean_object* v___x_198_; lean_object* v___x_199_; lean_object* v___x_200_; lean_object* v___x_201_; lean_object* v___x_202_; lean_object* v___x_203_; lean_object* v___x_204_; lean_object* v___x_205_; lean_object* v___x_206_; lean_object* v___x_207_; lean_object* v___x_208_; lean_object* v___x_209_; lean_object* v___x_210_; lean_object* v___x_211_; 
v_quotContext_187_ = lean_ctor_get(v_a_181_, 1);
v_currMacroScope_188_ = lean_ctor_get(v_a_181_, 2);
v_ref_189_ = lean_ctor_get(v_a_181_, 5);
v___x_190_ = lean_unsigned_to_nat(0u);
v___x_191_ = l_Lean_Syntax_getArg(v_x_180_, v___x_190_);
v___x_192_ = lean_unsigned_to_nat(2u);
v___x_193_ = l_Lean_Syntax_getArg(v_x_180_, v___x_192_);
lean_dec(v_x_180_);
v___x_194_ = 0;
v___x_195_ = l_Lean_SourceInfo_fromRef(v_ref_189_, v___x_194_);
v___x_196_ = ((lean_object*)(lp_mathlib_TensorProduct___aux__Mathlib__LinearAlgebra__TensorProduct__Defs______macroRules__TensorProduct__term___u2297____1___closed__4));
v___x_197_ = lean_obj_once(&lp_mathlib_TensorProduct___aux__Mathlib__LinearAlgebra__TensorProduct__Defs______macroRules__TensorProduct__term___u2297____1___closed__5, &lp_mathlib_TensorProduct___aux__Mathlib__LinearAlgebra__TensorProduct__Defs______macroRules__TensorProduct__term___u2297____1___closed__5_once, _init_lp_mathlib_TensorProduct___aux__Mathlib__LinearAlgebra__TensorProduct__Defs______macroRules__TensorProduct__term___u2297____1___closed__5);
v___x_198_ = ((lean_object*)(lp_mathlib_TensorProduct___aux__Mathlib__LinearAlgebra__TensorProduct__Defs______macroRules__TensorProduct__term___u2297____1___closed__6));
lean_inc(v_currMacroScope_188_);
lean_inc(v_quotContext_187_);
v___x_199_ = l_Lean_addMacroScope(v_quotContext_187_, v___x_198_, v_currMacroScope_188_);
v___x_200_ = ((lean_object*)(lp_mathlib_TensorProduct___aux__Mathlib__LinearAlgebra__TensorProduct__Defs______macroRules__TensorProduct__term___u2297____1___closed__10));
lean_inc_n(v___x_195_, 6);
v___x_201_ = lean_alloc_ctor(3, 4, 0);
lean_ctor_set(v___x_201_, 0, v___x_195_);
lean_ctor_set(v___x_201_, 1, v___x_197_);
lean_ctor_set(v___x_201_, 2, v___x_199_);
lean_ctor_set(v___x_201_, 3, v___x_200_);
v___x_202_ = ((lean_object*)(lp_mathlib_TensorProduct___aux__Mathlib__LinearAlgebra__TensorProduct__Defs______macroRules__TensorProduct__term___u2297____1___closed__12));
v___x_203_ = ((lean_object*)(lp_mathlib_TensorProduct___aux__Mathlib__LinearAlgebra__TensorProduct__Defs______macroRules__TensorProduct__term___u2297____1___closed__14));
v___x_204_ = ((lean_object*)(lp_mathlib_TensorProduct___aux__Mathlib__LinearAlgebra__TensorProduct__Defs______macroRules__TensorProduct__term___u2297____1___closed__15));
v___x_205_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_205_, 0, v___x_195_);
lean_ctor_set(v___x_205_, 1, v___x_204_);
v___x_206_ = l_Lean_Syntax_node1(v___x_195_, v___x_203_, v___x_205_);
v___x_207_ = l_Lean_Syntax_node1(v___x_195_, v___x_202_, v___x_206_);
v___x_208_ = l_Lean_Syntax_node2(v___x_195_, v___x_196_, v___x_201_, v___x_207_);
v___x_209_ = l_Lean_Syntax_node2(v___x_195_, v___x_202_, v___x_191_, v___x_193_);
v___x_210_ = l_Lean_Syntax_node2(v___x_195_, v___x_196_, v___x_208_, v___x_209_);
v___x_211_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_211_, 0, v___x_210_);
lean_ctor_set(v___x_211_, 1, v_a_182_);
return v___x_211_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_TensorProduct___aux__Mathlib__LinearAlgebra__TensorProduct__Defs______macroRules__TensorProduct__term___u2297____1___boxed(lean_object* v_x_212_, lean_object* v_a_213_, lean_object* v_a_214_){
_start:
{
lean_object* v_res_215_; 
v_res_215_ = lp_mathlib_TensorProduct___aux__Mathlib__LinearAlgebra__TensorProduct__Defs______macroRules__TensorProduct__term___u2297____1(v_x_212_, v_a_213_, v_a_214_);
lean_dec_ref(v_a_213_);
return v_res_215_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_TensorProduct___aux__Mathlib__LinearAlgebra__TensorProduct__Defs______macroRules__TensorProduct__term___u2297_x5b___x5d____1(lean_object* v_x_246_, lean_object* v_a_247_, lean_object* v_a_248_){
_start:
{
lean_object* v___x_249_; uint8_t v___x_250_; 
v___x_249_ = ((lean_object*)(lp_mathlib_TensorProduct_term___u2297_x5b___x5d___00__closed__1));
lean_inc(v_x_246_);
v___x_250_ = l_Lean_Syntax_isOfKind(v_x_246_, v___x_249_);
if (v___x_250_ == 0)
{
lean_object* v___x_251_; lean_object* v___x_252_; 
lean_dec(v_x_246_);
v___x_251_ = lean_box(1);
v___x_252_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_252_, 0, v___x_251_);
lean_ctor_set(v___x_252_, 1, v_a_248_);
return v___x_252_;
}
else
{
lean_object* v_quotContext_253_; lean_object* v_currMacroScope_254_; lean_object* v_ref_255_; lean_object* v___x_256_; lean_object* v___x_257_; lean_object* v___x_258_; lean_object* v___x_259_; lean_object* v___x_260_; lean_object* v___x_261_; uint8_t v___x_262_; lean_object* v___x_263_; lean_object* v___x_264_; lean_object* v___x_265_; lean_object* v___x_266_; lean_object* v___x_267_; lean_object* v___x_268_; lean_object* v___x_269_; lean_object* v___x_270_; lean_object* v___x_271_; lean_object* v___x_272_; lean_object* v___x_273_; 
v_quotContext_253_ = lean_ctor_get(v_a_247_, 1);
v_currMacroScope_254_ = lean_ctor_get(v_a_247_, 2);
v_ref_255_ = lean_ctor_get(v_a_247_, 5);
v___x_256_ = lean_unsigned_to_nat(0u);
v___x_257_ = l_Lean_Syntax_getArg(v_x_246_, v___x_256_);
v___x_258_ = lean_unsigned_to_nat(2u);
v___x_259_ = l_Lean_Syntax_getArg(v_x_246_, v___x_258_);
v___x_260_ = lean_unsigned_to_nat(4u);
v___x_261_ = l_Lean_Syntax_getArg(v_x_246_, v___x_260_);
lean_dec(v_x_246_);
v___x_262_ = 0;
v___x_263_ = l_Lean_SourceInfo_fromRef(v_ref_255_, v___x_262_);
v___x_264_ = ((lean_object*)(lp_mathlib_TensorProduct___aux__Mathlib__LinearAlgebra__TensorProduct__Defs______macroRules__TensorProduct__term___u2297____1___closed__4));
v___x_265_ = lean_obj_once(&lp_mathlib_TensorProduct___aux__Mathlib__LinearAlgebra__TensorProduct__Defs______macroRules__TensorProduct__term___u2297____1___closed__5, &lp_mathlib_TensorProduct___aux__Mathlib__LinearAlgebra__TensorProduct__Defs______macroRules__TensorProduct__term___u2297____1___closed__5_once, _init_lp_mathlib_TensorProduct___aux__Mathlib__LinearAlgebra__TensorProduct__Defs______macroRules__TensorProduct__term___u2297____1___closed__5);
v___x_266_ = ((lean_object*)(lp_mathlib_TensorProduct___aux__Mathlib__LinearAlgebra__TensorProduct__Defs______macroRules__TensorProduct__term___u2297____1___closed__6));
lean_inc(v_currMacroScope_254_);
lean_inc(v_quotContext_253_);
v___x_267_ = l_Lean_addMacroScope(v_quotContext_253_, v___x_266_, v_currMacroScope_254_);
v___x_268_ = ((lean_object*)(lp_mathlib_TensorProduct___aux__Mathlib__LinearAlgebra__TensorProduct__Defs______macroRules__TensorProduct__term___u2297____1___closed__10));
lean_inc_n(v___x_263_, 2);
v___x_269_ = lean_alloc_ctor(3, 4, 0);
lean_ctor_set(v___x_269_, 0, v___x_263_);
lean_ctor_set(v___x_269_, 1, v___x_265_);
lean_ctor_set(v___x_269_, 2, v___x_267_);
lean_ctor_set(v___x_269_, 3, v___x_268_);
v___x_270_ = ((lean_object*)(lp_mathlib_TensorProduct___aux__Mathlib__LinearAlgebra__TensorProduct__Defs______macroRules__TensorProduct__term___u2297____1___closed__12));
v___x_271_ = l_Lean_Syntax_node3(v___x_263_, v___x_270_, v___x_259_, v___x_257_, v___x_261_);
v___x_272_ = l_Lean_Syntax_node2(v___x_263_, v___x_264_, v___x_269_, v___x_271_);
v___x_273_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_273_, 0, v___x_272_);
lean_ctor_set(v___x_273_, 1, v_a_248_);
return v___x_273_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_TensorProduct___aux__Mathlib__LinearAlgebra__TensorProduct__Defs______macroRules__TensorProduct__term___u2297_x5b___x5d____1___boxed(lean_object* v_x_274_, lean_object* v_a_275_, lean_object* v_a_276_){
_start:
{
lean_object* v_res_277_; 
v_res_277_ = lp_mathlib_TensorProduct___aux__Mathlib__LinearAlgebra__TensorProduct__Defs______macroRules__TensorProduct__term___u2297_x5b___x5d____1(v_x_274_, v_a_275_, v_a_276_);
lean_dec_ref(v_a_275_);
return v_res_277_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_TensorProduct___aux__Mathlib__LinearAlgebra__TensorProduct__Defs______unexpand__TensorProduct__1(lean_object* v_x_281_, lean_object* v_a_282_, lean_object* v_a_283_){
_start:
{
lean_object* v___x_284_; uint8_t v___x_285_; 
v___x_284_ = ((lean_object*)(lp_mathlib_TensorProduct___aux__Mathlib__LinearAlgebra__TensorProduct__Defs______macroRules__TensorProduct__term___u2297____1___closed__4));
lean_inc(v_x_281_);
v___x_285_ = l_Lean_Syntax_isOfKind(v_x_281_, v___x_284_);
if (v___x_285_ == 0)
{
lean_object* v___x_286_; lean_object* v___x_287_; 
lean_dec(v_x_281_);
v___x_286_ = lean_box(0);
v___x_287_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_287_, 0, v___x_286_);
lean_ctor_set(v___x_287_, 1, v_a_283_);
return v___x_287_;
}
else
{
lean_object* v___x_288_; lean_object* v___x_289_; lean_object* v___x_290_; uint8_t v___x_291_; 
v___x_288_ = lean_unsigned_to_nat(0u);
v___x_289_ = l_Lean_Syntax_getArg(v_x_281_, v___x_288_);
v___x_290_ = ((lean_object*)(lp_mathlib_TensorProduct___aux__Mathlib__LinearAlgebra__TensorProduct__Defs______unexpand__TensorProduct__1___closed__1));
lean_inc(v___x_289_);
v___x_291_ = l_Lean_Syntax_isOfKind(v___x_289_, v___x_290_);
if (v___x_291_ == 0)
{
lean_object* v___x_292_; lean_object* v___x_293_; 
lean_dec(v___x_289_);
lean_dec(v_x_281_);
v___x_292_ = lean_box(0);
v___x_293_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_293_, 0, v___x_292_);
lean_ctor_set(v___x_293_, 1, v_a_283_);
return v___x_293_;
}
else
{
lean_object* v___x_294_; lean_object* v___x_295_; lean_object* v___x_296_; uint8_t v___x_297_; 
v___x_294_ = lean_unsigned_to_nat(1u);
v___x_295_ = l_Lean_Syntax_getArg(v_x_281_, v___x_294_);
lean_dec(v_x_281_);
v___x_296_ = lean_unsigned_to_nat(3u);
lean_inc(v___x_295_);
v___x_297_ = l_Lean_Syntax_matchesNull(v___x_295_, v___x_296_);
if (v___x_297_ == 0)
{
lean_object* v___x_298_; lean_object* v___x_299_; 
lean_dec(v___x_295_);
lean_dec(v___x_289_);
v___x_298_ = lean_box(0);
v___x_299_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_299_, 0, v___x_298_);
lean_ctor_set(v___x_299_, 1, v_a_283_);
return v___x_299_;
}
else
{
lean_object* v___x_300_; lean_object* v___x_301_; lean_object* v___x_302_; lean_object* v___x_303_; lean_object* v_ref_304_; uint8_t v___x_305_; lean_object* v___x_306_; lean_object* v___x_307_; lean_object* v___x_308_; lean_object* v___x_309_; lean_object* v___x_310_; lean_object* v___x_311_; lean_object* v___x_312_; lean_object* v___x_313_; 
v___x_300_ = l_Lean_Syntax_getArg(v___x_295_, v___x_288_);
v___x_301_ = l_Lean_Syntax_getArg(v___x_295_, v___x_294_);
v___x_302_ = lean_unsigned_to_nat(2u);
v___x_303_ = l_Lean_Syntax_getArg(v___x_295_, v___x_302_);
lean_dec(v___x_295_);
v_ref_304_ = l_Lean_replaceRef(v___x_289_, v_a_282_);
lean_dec(v___x_289_);
v___x_305_ = 0;
v___x_306_ = l_Lean_SourceInfo_fromRef(v_ref_304_, v___x_305_);
lean_dec(v_ref_304_);
v___x_307_ = ((lean_object*)(lp_mathlib_TensorProduct_term___u2297_x5b___x5d___00__closed__1));
v___x_308_ = ((lean_object*)(lp_mathlib_TensorProduct_term___u2297_x5b___x5d___00__closed__2));
lean_inc_n(v___x_306_, 2);
v___x_309_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_309_, 0, v___x_306_);
lean_ctor_set(v___x_309_, 1, v___x_308_);
v___x_310_ = ((lean_object*)(lp_mathlib_TensorProduct_term___u2297_x5b___x5d___00__closed__6));
v___x_311_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_311_, 0, v___x_306_);
lean_ctor_set(v___x_311_, 1, v___x_310_);
v___x_312_ = l_Lean_Syntax_node5(v___x_306_, v___x_307_, v___x_301_, v___x_309_, v___x_300_, v___x_311_, v___x_303_);
v___x_313_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_313_, 0, v___x_312_);
lean_ctor_set(v___x_313_, 1, v_a_283_);
return v___x_313_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_TensorProduct___aux__Mathlib__LinearAlgebra__TensorProduct__Defs______unexpand__TensorProduct__1___boxed(lean_object* v_x_314_, lean_object* v_a_315_, lean_object* v_a_316_){
_start:
{
lean_object* v_res_317_; 
v_res_317_ = lp_mathlib_TensorProduct___aux__Mathlib__LinearAlgebra__TensorProduct__Defs______unexpand__TensorProduct__1(v_x_314_, v_a_315_, v_a_316_);
lean_dec(v_a_315_);
return v_res_317_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_TensorProduct_addCommSemigroup___redArg(lean_object* v_inst_318_, lean_object* v_inst_319_, lean_object* v_inst_320_, lean_object* v_inst_321_, lean_object* v_inst_322_){
_start:
{
lean_object* v___x_323_; 
v___x_323_ = lean_alloc_closure((void*)(lp_mathlib_instAddTensorProduct___aux__1___boxed), 10, 8);
lean_closure_set(v___x_323_, 0, lean_box(0));
lean_closure_set(v___x_323_, 1, v_inst_318_);
lean_closure_set(v___x_323_, 2, lean_box(0));
lean_closure_set(v___x_323_, 3, lean_box(0));
lean_closure_set(v___x_323_, 4, v_inst_319_);
lean_closure_set(v___x_323_, 5, v_inst_320_);
lean_closure_set(v___x_323_, 6, v_inst_321_);
lean_closure_set(v___x_323_, 7, v_inst_322_);
return v___x_323_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_TensorProduct_addCommSemigroup(lean_object* v_R_324_, lean_object* v_inst_325_, lean_object* v_M_326_, lean_object* v_N_327_, lean_object* v_inst_328_, lean_object* v_inst_329_, lean_object* v_inst_330_, lean_object* v_inst_331_){
_start:
{
lean_object* v___x_332_; 
v___x_332_ = lean_alloc_closure((void*)(lp_mathlib_instAddTensorProduct___aux__1___boxed), 10, 8);
lean_closure_set(v___x_332_, 0, lean_box(0));
lean_closure_set(v___x_332_, 1, v_inst_325_);
lean_closure_set(v___x_332_, 2, lean_box(0));
lean_closure_set(v___x_332_, 3, lean_box(0));
lean_closure_set(v___x_332_, 4, v_inst_328_);
lean_closure_set(v___x_332_, 5, v_inst_329_);
lean_closure_set(v___x_332_, 6, v_inst_330_);
lean_closure_set(v___x_332_, 7, v_inst_331_);
return v___x_332_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_TensorProduct_instInhabited(lean_object* v_R_333_, lean_object* v_inst_334_, lean_object* v_M_335_, lean_object* v_N_336_, lean_object* v_inst_337_, lean_object* v_inst_338_, lean_object* v_inst_339_, lean_object* v_inst_340_){
_start:
{
lean_object* v___x_341_; lean_object* v_toZero_342_; 
v___x_341_ = lean_obj_once(&lp_mathlib_instZeroTensorProduct___aux__1___closed__2, &lp_mathlib_instZeroTensorProduct___aux__1___closed__2_once, _init_lp_mathlib_instZeroTensorProduct___aux__1___closed__2);
v_toZero_342_ = lean_ctor_get(v___x_341_, 0);
lean_inc(v_toZero_342_);
return v_toZero_342_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_TensorProduct_instInhabited___boxed(lean_object* v_R_343_, lean_object* v_inst_344_, lean_object* v_M_345_, lean_object* v_N_346_, lean_object* v_inst_347_, lean_object* v_inst_348_, lean_object* v_inst_349_, lean_object* v_inst_350_){
_start:
{
lean_object* v_res_351_; 
v_res_351_ = lp_mathlib_TensorProduct_instInhabited(v_R_343_, v_inst_344_, v_M_345_, v_N_346_, v_inst_347_, v_inst_348_, v_inst_349_, v_inst_350_);
lean_dec(v_inst_350_);
lean_dec(v_inst_349_);
lean_dec_ref(v_inst_348_);
lean_dec_ref(v_inst_347_);
lean_dec_ref(v_inst_344_);
return v_res_351_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_TensorProduct_tmul___redArg(lean_object* v_m_352_, lean_object* v_n_353_){
_start:
{
lean_object* v___x_354_; lean_object* v___x_355_; lean_object* v___x_356_; lean_object* v___x_357_; lean_object* v___x_27__overap_358_; lean_object* v___x_359_; 
v___x_354_ = lean_obj_once(&lp_mathlib_instZeroTensorProduct___aux__1___closed__1, &lp_mathlib_instZeroTensorProduct___aux__1___closed__1_once, _init_lp_mathlib_instZeroTensorProduct___aux__1___closed__1);
v___x_355_ = lean_box(0);
v___x_356_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_356_, 0, v_m_352_);
lean_ctor_set(v___x_356_, 1, v_n_353_);
v___x_357_ = lp_mathlib_FreeAddMonoid_of___redArg(v___x_356_);
v___x_27__overap_358_ = lp_mathlib_AddCon_mk_x27___redArg(v___x_354_, v___x_355_);
v___x_359_ = lean_apply_1(v___x_27__overap_358_, v___x_357_);
return v___x_359_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_TensorProduct_tmul(lean_object* v_R_360_, lean_object* v_inst_361_, lean_object* v_M_362_, lean_object* v_N_363_, lean_object* v_inst_364_, lean_object* v_inst_365_, lean_object* v_inst_366_, lean_object* v_inst_367_, lean_object* v_m_368_, lean_object* v_n_369_){
_start:
{
lean_object* v___x_370_; 
v___x_370_ = lp_mathlib_TensorProduct_tmul___redArg(v_m_368_, v_n_369_);
return v___x_370_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_TensorProduct_tmul___boxed(lean_object* v_R_371_, lean_object* v_inst_372_, lean_object* v_M_373_, lean_object* v_N_374_, lean_object* v_inst_375_, lean_object* v_inst_376_, lean_object* v_inst_377_, lean_object* v_inst_378_, lean_object* v_m_379_, lean_object* v_n_380_){
_start:
{
lean_object* v_res_381_; 
v_res_381_ = lp_mathlib_TensorProduct_tmul(v_R_371_, v_inst_372_, v_M_373_, v_N_374_, v_inst_375_, v_inst_376_, v_inst_377_, v_inst_378_, v_m_379_, v_n_380_);
lean_dec(v_inst_378_);
lean_dec(v_inst_377_);
lean_dec_ref(v_inst_376_);
lean_dec_ref(v_inst_375_);
lean_dec_ref(v_inst_372_);
return v_res_381_;
}
}
static lean_object* _init_lp_mathlib_TensorProduct___aux__Mathlib__LinearAlgebra__TensorProduct__Defs______macroRules__TensorProduct__term___u2297_u209c____1___closed__1(void){
_start:
{
lean_object* v___x_399_; lean_object* v___x_400_; 
v___x_399_ = ((lean_object*)(lp_mathlib_TensorProduct___aux__Mathlib__LinearAlgebra__TensorProduct__Defs______macroRules__TensorProduct__term___u2297_u209c____1___closed__0));
v___x_400_ = l_String_toRawSubstring_x27(v___x_399_);
return v___x_400_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_TensorProduct___aux__Mathlib__LinearAlgebra__TensorProduct__Defs______macroRules__TensorProduct__term___u2297_u209c____1(lean_object* v_x_412_, lean_object* v_a_413_, lean_object* v_a_414_){
_start:
{
lean_object* v___x_415_; uint8_t v___x_416_; 
v___x_415_ = ((lean_object*)(lp_mathlib_TensorProduct_term___u2297_u209c___00__closed__1));
lean_inc(v_x_412_);
v___x_416_ = l_Lean_Syntax_isOfKind(v_x_412_, v___x_415_);
if (v___x_416_ == 0)
{
lean_object* v___x_417_; lean_object* v___x_418_; 
lean_dec(v_x_412_);
v___x_417_ = lean_box(1);
v___x_418_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_418_, 0, v___x_417_);
lean_ctor_set(v___x_418_, 1, v_a_414_);
return v___x_418_;
}
else
{
lean_object* v_quotContext_419_; lean_object* v_currMacroScope_420_; lean_object* v_ref_421_; lean_object* v___x_422_; lean_object* v___x_423_; lean_object* v___x_424_; lean_object* v___x_425_; uint8_t v___x_426_; lean_object* v___x_427_; lean_object* v___x_428_; lean_object* v___x_429_; lean_object* v___x_430_; lean_object* v___x_431_; lean_object* v___x_432_; lean_object* v___x_433_; lean_object* v___x_434_; lean_object* v___x_435_; lean_object* v___x_436_; lean_object* v___x_437_; lean_object* v___x_438_; lean_object* v___x_439_; lean_object* v___x_440_; lean_object* v___x_441_; lean_object* v___x_442_; lean_object* v___x_443_; 
v_quotContext_419_ = lean_ctor_get(v_a_413_, 1);
v_currMacroScope_420_ = lean_ctor_get(v_a_413_, 2);
v_ref_421_ = lean_ctor_get(v_a_413_, 5);
v___x_422_ = lean_unsigned_to_nat(0u);
v___x_423_ = l_Lean_Syntax_getArg(v_x_412_, v___x_422_);
v___x_424_ = lean_unsigned_to_nat(2u);
v___x_425_ = l_Lean_Syntax_getArg(v_x_412_, v___x_424_);
lean_dec(v_x_412_);
v___x_426_ = 0;
v___x_427_ = l_Lean_SourceInfo_fromRef(v_ref_421_, v___x_426_);
v___x_428_ = ((lean_object*)(lp_mathlib_TensorProduct___aux__Mathlib__LinearAlgebra__TensorProduct__Defs______macroRules__TensorProduct__term___u2297____1___closed__4));
v___x_429_ = lean_obj_once(&lp_mathlib_TensorProduct___aux__Mathlib__LinearAlgebra__TensorProduct__Defs______macroRules__TensorProduct__term___u2297_u209c____1___closed__1, &lp_mathlib_TensorProduct___aux__Mathlib__LinearAlgebra__TensorProduct__Defs______macroRules__TensorProduct__term___u2297_u209c____1___closed__1_once, _init_lp_mathlib_TensorProduct___aux__Mathlib__LinearAlgebra__TensorProduct__Defs______macroRules__TensorProduct__term___u2297_u209c____1___closed__1);
v___x_430_ = ((lean_object*)(lp_mathlib_TensorProduct___aux__Mathlib__LinearAlgebra__TensorProduct__Defs______macroRules__TensorProduct__term___u2297_u209c____1___closed__2));
lean_inc(v_currMacroScope_420_);
lean_inc(v_quotContext_419_);
v___x_431_ = l_Lean_addMacroScope(v_quotContext_419_, v___x_430_, v_currMacroScope_420_);
v___x_432_ = ((lean_object*)(lp_mathlib_TensorProduct___aux__Mathlib__LinearAlgebra__TensorProduct__Defs______macroRules__TensorProduct__term___u2297_u209c____1___closed__5));
lean_inc_n(v___x_427_, 6);
v___x_433_ = lean_alloc_ctor(3, 4, 0);
lean_ctor_set(v___x_433_, 0, v___x_427_);
lean_ctor_set(v___x_433_, 1, v___x_429_);
lean_ctor_set(v___x_433_, 2, v___x_431_);
lean_ctor_set(v___x_433_, 3, v___x_432_);
v___x_434_ = ((lean_object*)(lp_mathlib_TensorProduct___aux__Mathlib__LinearAlgebra__TensorProduct__Defs______macroRules__TensorProduct__term___u2297____1___closed__12));
v___x_435_ = ((lean_object*)(lp_mathlib_TensorProduct___aux__Mathlib__LinearAlgebra__TensorProduct__Defs______macroRules__TensorProduct__term___u2297____1___closed__14));
v___x_436_ = ((lean_object*)(lp_mathlib_TensorProduct___aux__Mathlib__LinearAlgebra__TensorProduct__Defs______macroRules__TensorProduct__term___u2297____1___closed__15));
v___x_437_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_437_, 0, v___x_427_);
lean_ctor_set(v___x_437_, 1, v___x_436_);
v___x_438_ = l_Lean_Syntax_node1(v___x_427_, v___x_435_, v___x_437_);
v___x_439_ = l_Lean_Syntax_node1(v___x_427_, v___x_434_, v___x_438_);
v___x_440_ = l_Lean_Syntax_node2(v___x_427_, v___x_428_, v___x_433_, v___x_439_);
v___x_441_ = l_Lean_Syntax_node2(v___x_427_, v___x_434_, v___x_423_, v___x_425_);
v___x_442_ = l_Lean_Syntax_node2(v___x_427_, v___x_428_, v___x_440_, v___x_441_);
v___x_443_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_443_, 0, v___x_442_);
lean_ctor_set(v___x_443_, 1, v_a_414_);
return v___x_443_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_TensorProduct___aux__Mathlib__LinearAlgebra__TensorProduct__Defs______macroRules__TensorProduct__term___u2297_u209c____1___boxed(lean_object* v_x_444_, lean_object* v_a_445_, lean_object* v_a_446_){
_start:
{
lean_object* v_res_447_; 
v_res_447_ = lp_mathlib_TensorProduct___aux__Mathlib__LinearAlgebra__TensorProduct__Defs______macroRules__TensorProduct__term___u2297_u209c____1(v_x_444_, v_a_445_, v_a_446_);
lean_dec_ref(v_a_445_);
return v_res_447_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_TensorProduct___aux__Mathlib__LinearAlgebra__TensorProduct__Defs______macroRules__TensorProduct__term___u2297_u209c_x5b___x5d____1(lean_object* v_x_472_, lean_object* v_a_473_, lean_object* v_a_474_){
_start:
{
lean_object* v___x_475_; uint8_t v___x_476_; 
v___x_475_ = ((lean_object*)(lp_mathlib_TensorProduct_term___u2297_u209c_x5b___x5d___00__closed__1));
lean_inc(v_x_472_);
v___x_476_ = l_Lean_Syntax_isOfKind(v_x_472_, v___x_475_);
if (v___x_476_ == 0)
{
lean_object* v___x_477_; lean_object* v___x_478_; 
lean_dec(v_x_472_);
v___x_477_ = lean_box(1);
v___x_478_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_478_, 0, v___x_477_);
lean_ctor_set(v___x_478_, 1, v_a_474_);
return v___x_478_;
}
else
{
lean_object* v_quotContext_479_; lean_object* v_currMacroScope_480_; lean_object* v_ref_481_; lean_object* v___x_482_; lean_object* v___x_483_; lean_object* v___x_484_; lean_object* v___x_485_; lean_object* v___x_486_; lean_object* v___x_487_; uint8_t v___x_488_; lean_object* v___x_489_; lean_object* v___x_490_; lean_object* v___x_491_; lean_object* v___x_492_; lean_object* v___x_493_; lean_object* v___x_494_; lean_object* v___x_495_; lean_object* v___x_496_; lean_object* v___x_497_; lean_object* v___x_498_; lean_object* v___x_499_; 
v_quotContext_479_ = lean_ctor_get(v_a_473_, 1);
v_currMacroScope_480_ = lean_ctor_get(v_a_473_, 2);
v_ref_481_ = lean_ctor_get(v_a_473_, 5);
v___x_482_ = lean_unsigned_to_nat(0u);
v___x_483_ = l_Lean_Syntax_getArg(v_x_472_, v___x_482_);
v___x_484_ = lean_unsigned_to_nat(2u);
v___x_485_ = l_Lean_Syntax_getArg(v_x_472_, v___x_484_);
v___x_486_ = lean_unsigned_to_nat(4u);
v___x_487_ = l_Lean_Syntax_getArg(v_x_472_, v___x_486_);
lean_dec(v_x_472_);
v___x_488_ = 0;
v___x_489_ = l_Lean_SourceInfo_fromRef(v_ref_481_, v___x_488_);
v___x_490_ = ((lean_object*)(lp_mathlib_TensorProduct___aux__Mathlib__LinearAlgebra__TensorProduct__Defs______macroRules__TensorProduct__term___u2297____1___closed__4));
v___x_491_ = lean_obj_once(&lp_mathlib_TensorProduct___aux__Mathlib__LinearAlgebra__TensorProduct__Defs______macroRules__TensorProduct__term___u2297_u209c____1___closed__1, &lp_mathlib_TensorProduct___aux__Mathlib__LinearAlgebra__TensorProduct__Defs______macroRules__TensorProduct__term___u2297_u209c____1___closed__1_once, _init_lp_mathlib_TensorProduct___aux__Mathlib__LinearAlgebra__TensorProduct__Defs______macroRules__TensorProduct__term___u2297_u209c____1___closed__1);
v___x_492_ = ((lean_object*)(lp_mathlib_TensorProduct___aux__Mathlib__LinearAlgebra__TensorProduct__Defs______macroRules__TensorProduct__term___u2297_u209c____1___closed__2));
lean_inc(v_currMacroScope_480_);
lean_inc(v_quotContext_479_);
v___x_493_ = l_Lean_addMacroScope(v_quotContext_479_, v___x_492_, v_currMacroScope_480_);
v___x_494_ = ((lean_object*)(lp_mathlib_TensorProduct___aux__Mathlib__LinearAlgebra__TensorProduct__Defs______macroRules__TensorProduct__term___u2297_u209c____1___closed__5));
lean_inc_n(v___x_489_, 2);
v___x_495_ = lean_alloc_ctor(3, 4, 0);
lean_ctor_set(v___x_495_, 0, v___x_489_);
lean_ctor_set(v___x_495_, 1, v___x_491_);
lean_ctor_set(v___x_495_, 2, v___x_493_);
lean_ctor_set(v___x_495_, 3, v___x_494_);
v___x_496_ = ((lean_object*)(lp_mathlib_TensorProduct___aux__Mathlib__LinearAlgebra__TensorProduct__Defs______macroRules__TensorProduct__term___u2297____1___closed__12));
v___x_497_ = l_Lean_Syntax_node3(v___x_489_, v___x_496_, v___x_485_, v___x_483_, v___x_487_);
v___x_498_ = l_Lean_Syntax_node2(v___x_489_, v___x_490_, v___x_495_, v___x_497_);
v___x_499_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_499_, 0, v___x_498_);
lean_ctor_set(v___x_499_, 1, v_a_474_);
return v___x_499_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_TensorProduct___aux__Mathlib__LinearAlgebra__TensorProduct__Defs______macroRules__TensorProduct__term___u2297_u209c_x5b___x5d____1___boxed(lean_object* v_x_500_, lean_object* v_a_501_, lean_object* v_a_502_){
_start:
{
lean_object* v_res_503_; 
v_res_503_ = lp_mathlib_TensorProduct___aux__Mathlib__LinearAlgebra__TensorProduct__Defs______macroRules__TensorProduct__term___u2297_u209c_x5b___x5d____1(v_x_500_, v_a_501_, v_a_502_);
lean_dec_ref(v_a_501_);
return v_res_503_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_TensorProduct___aux__Mathlib__LinearAlgebra__TensorProduct__Defs______unexpand__TensorProduct__tmul__1(lean_object* v_x_504_, lean_object* v_a_505_, lean_object* v_a_506_){
_start:
{
lean_object* v___x_507_; uint8_t v___x_508_; 
v___x_507_ = ((lean_object*)(lp_mathlib_TensorProduct___aux__Mathlib__LinearAlgebra__TensorProduct__Defs______macroRules__TensorProduct__term___u2297____1___closed__4));
lean_inc(v_x_504_);
v___x_508_ = l_Lean_Syntax_isOfKind(v_x_504_, v___x_507_);
if (v___x_508_ == 0)
{
lean_object* v___x_509_; lean_object* v___x_510_; 
lean_dec(v_x_504_);
v___x_509_ = lean_box(0);
v___x_510_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_510_, 0, v___x_509_);
lean_ctor_set(v___x_510_, 1, v_a_506_);
return v___x_510_;
}
else
{
lean_object* v___x_511_; lean_object* v___x_512_; lean_object* v___x_513_; uint8_t v___x_514_; 
v___x_511_ = lean_unsigned_to_nat(0u);
v___x_512_ = l_Lean_Syntax_getArg(v_x_504_, v___x_511_);
v___x_513_ = ((lean_object*)(lp_mathlib_TensorProduct___aux__Mathlib__LinearAlgebra__TensorProduct__Defs______unexpand__TensorProduct__1___closed__1));
lean_inc(v___x_512_);
v___x_514_ = l_Lean_Syntax_isOfKind(v___x_512_, v___x_513_);
if (v___x_514_ == 0)
{
lean_object* v___x_515_; lean_object* v___x_516_; 
lean_dec(v___x_512_);
lean_dec(v_x_504_);
v___x_515_ = lean_box(0);
v___x_516_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_516_, 0, v___x_515_);
lean_ctor_set(v___x_516_, 1, v_a_506_);
return v___x_516_;
}
else
{
lean_object* v___x_517_; lean_object* v___x_518_; lean_object* v___x_519_; uint8_t v___x_520_; 
v___x_517_ = lean_unsigned_to_nat(1u);
v___x_518_ = l_Lean_Syntax_getArg(v_x_504_, v___x_517_);
lean_dec(v_x_504_);
v___x_519_ = lean_unsigned_to_nat(3u);
lean_inc(v___x_518_);
v___x_520_ = l_Lean_Syntax_matchesNull(v___x_518_, v___x_519_);
if (v___x_520_ == 0)
{
lean_object* v___x_521_; lean_object* v___x_522_; 
lean_dec(v___x_518_);
lean_dec(v___x_512_);
v___x_521_ = lean_box(0);
v___x_522_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_522_, 0, v___x_521_);
lean_ctor_set(v___x_522_, 1, v_a_506_);
return v___x_522_;
}
else
{
lean_object* v___x_523_; lean_object* v___x_524_; lean_object* v___x_525_; lean_object* v___x_526_; lean_object* v_ref_527_; uint8_t v___x_528_; lean_object* v___x_529_; lean_object* v___x_530_; lean_object* v___x_531_; lean_object* v___x_532_; lean_object* v___x_533_; lean_object* v___x_534_; lean_object* v___x_535_; lean_object* v___x_536_; 
v___x_523_ = l_Lean_Syntax_getArg(v___x_518_, v___x_511_);
v___x_524_ = l_Lean_Syntax_getArg(v___x_518_, v___x_517_);
v___x_525_ = lean_unsigned_to_nat(2u);
v___x_526_ = l_Lean_Syntax_getArg(v___x_518_, v___x_525_);
lean_dec(v___x_518_);
v_ref_527_ = l_Lean_replaceRef(v___x_512_, v_a_505_);
lean_dec(v___x_512_);
v___x_528_ = 0;
v___x_529_ = l_Lean_SourceInfo_fromRef(v_ref_527_, v___x_528_);
lean_dec(v_ref_527_);
v___x_530_ = ((lean_object*)(lp_mathlib_TensorProduct_term___u2297_u209c_x5b___x5d___00__closed__1));
v___x_531_ = ((lean_object*)(lp_mathlib_TensorProduct_term___u2297_u209c_x5b___x5d___00__closed__2));
lean_inc_n(v___x_529_, 2);
v___x_532_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_532_, 0, v___x_529_);
lean_ctor_set(v___x_532_, 1, v___x_531_);
v___x_533_ = ((lean_object*)(lp_mathlib_TensorProduct_term___u2297_x5b___x5d___00__closed__6));
v___x_534_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_534_, 0, v___x_529_);
lean_ctor_set(v___x_534_, 1, v___x_533_);
v___x_535_ = l_Lean_Syntax_node5(v___x_529_, v___x_530_, v___x_524_, v___x_532_, v___x_523_, v___x_534_, v___x_526_);
v___x_536_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_536_, 0, v___x_535_);
lean_ctor_set(v___x_536_, 1, v_a_506_);
return v___x_536_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_TensorProduct___aux__Mathlib__LinearAlgebra__TensorProduct__Defs______unexpand__TensorProduct__tmul__1___boxed(lean_object* v_x_537_, lean_object* v_a_538_, lean_object* v_a_539_){
_start:
{
lean_object* v_res_540_; 
v_res_540_ = lp_mathlib_TensorProduct___aux__Mathlib__LinearAlgebra__TensorProduct__Defs______unexpand__TensorProduct__tmul__1(v_x_537_, v_a_538_, v_a_539_);
lean_dec(v_a_538_);
return v_res_540_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_TensorProduct_instRepr___redArg___lam__0(lean_object* v_inst_543_, lean_object* v_inst_544_, lean_object* v_x_545_){
_start:
{
lean_object* v_fst_546_; lean_object* v_snd_547_; lean_object* v___x_549_; uint8_t v_isShared_550_; uint8_t v_isSharedCheck_562_; 
v_fst_546_ = lean_ctor_get(v_x_545_, 0);
v_snd_547_ = lean_ctor_get(v_x_545_, 1);
v_isSharedCheck_562_ = !lean_is_exclusive(v_x_545_);
if (v_isSharedCheck_562_ == 0)
{
v___x_549_ = v_x_545_;
v_isShared_550_ = v_isSharedCheck_562_;
goto v_resetjp_548_;
}
else
{
lean_inc(v_snd_547_);
lean_inc(v_fst_546_);
lean_dec(v_x_545_);
v___x_549_ = lean_box(0);
v_isShared_550_ = v_isSharedCheck_562_;
goto v_resetjp_548_;
}
v_resetjp_548_:
{
lean_object* v___x_551_; lean_object* v___x_552_; lean_object* v___x_553_; lean_object* v___x_555_; 
v___x_551_ = lean_unsigned_to_nat(100u);
v___x_552_ = lean_apply_2(v_inst_543_, v_fst_546_, v___x_551_);
v___x_553_ = ((lean_object*)(lp_mathlib_TensorProduct_instRepr___redArg___lam__0___closed__0));
if (v_isShared_550_ == 0)
{
lean_ctor_set_tag(v___x_549_, 5);
lean_ctor_set(v___x_549_, 1, v___x_553_);
lean_ctor_set(v___x_549_, 0, v___x_552_);
v___x_555_ = v___x_549_;
goto v_reusejp_554_;
}
else
{
lean_object* v_reuseFailAlloc_561_; 
v_reuseFailAlloc_561_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v_reuseFailAlloc_561_, 0, v___x_552_);
lean_ctor_set(v_reuseFailAlloc_561_, 1, v___x_553_);
v___x_555_ = v_reuseFailAlloc_561_;
goto v_reusejp_554_;
}
v_reusejp_554_:
{
lean_object* v___x_556_; lean_object* v___x_557_; lean_object* v___x_558_; uint8_t v___x_559_; lean_object* v___x_560_; 
v___x_556_ = lean_unsigned_to_nat(101u);
v___x_557_ = lean_apply_2(v_inst_544_, v_snd_547_, v___x_556_);
v___x_558_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_558_, 0, v___x_555_);
lean_ctor_set(v___x_558_, 1, v___x_557_);
v___x_559_ = 0;
v___x_560_ = lean_alloc_ctor(6, 1, 1);
lean_ctor_set(v___x_560_, 0, v___x_558_);
lean_ctor_set_uint8(v___x_560_, sizeof(void*)*1, v___x_559_);
return v___x_560_;
}
}
}
}
static lean_object* _init_lp_mathlib_TensorProduct_instRepr___redArg___lam__1___closed__0(void){
_start:
{
lean_object* v___x_563_; 
v___x_563_ = lp_mathlib_Equiv_refl(lean_box(0));
return v___x_563_;
}
}
static lean_object* _init_lp_mathlib_TensorProduct_instRepr___redArg___lam__1___closed__5(void){
_start:
{
lean_object* v___x_569_; lean_object* v___x_570_; 
v___x_569_ = ((lean_object*)(lp_mathlib_TensorProduct_instRepr___redArg___lam__1___closed__3));
v___x_570_ = lean_string_length(v___x_569_);
return v___x_570_;
}
}
static lean_object* _init_lp_mathlib_TensorProduct_instRepr___redArg___lam__1___closed__6(void){
_start:
{
lean_object* v___x_571_; lean_object* v___x_572_; 
v___x_571_ = lean_obj_once(&lp_mathlib_TensorProduct_instRepr___redArg___lam__1___closed__5, &lp_mathlib_TensorProduct_instRepr___redArg___lam__1___closed__5_once, _init_lp_mathlib_TensorProduct_instRepr___redArg___lam__1___closed__5);
v___x_572_ = lean_nat_to_int(v___x_571_);
return v___x_572_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_TensorProduct_instRepr___redArg___lam__1(lean_object* v___f_583_, lean_object* v___f_584_, lean_object* v_mn_585_, lean_object* v_p_586_){
_start:
{
lean_object* v___x_587_; lean_object* v_toFun_588_; lean_object* v___x_589_; lean_object* v___x_590_; lean_object* v_parts_591_; 
v___x_587_ = lean_obj_once(&lp_mathlib_TensorProduct_instRepr___redArg___lam__1___closed__0, &lp_mathlib_TensorProduct_instRepr___redArg___lam__1___closed__0_once, _init_lp_mathlib_TensorProduct_instRepr___redArg___lam__1___closed__0);
v_toFun_588_ = lean_ctor_get(v___x_587_, 0);
lean_inc(v_toFun_588_);
v___x_589_ = lean_apply_1(v_toFun_588_, v_mn_585_);
v___x_590_ = lean_box(0);
v_parts_591_ = l_List_mapTR_loop___redArg(v___f_583_, v___x_589_, v___x_590_);
if (lean_obj_tag(v_parts_591_) == 0)
{
lean_object* v___x_592_; 
lean_dec_ref(v___f_584_);
v___x_592_ = ((lean_object*)(lp_mathlib_TensorProduct_instRepr___redArg___lam__1___closed__2));
return v___x_592_;
}
else
{
lean_object* v_tail_593_; 
v_tail_593_ = lean_ctor_get(v_parts_591_, 1);
lean_inc(v_tail_593_);
if (lean_obj_tag(v_tail_593_) == 0)
{
lean_object* v_head_594_; lean_object* v___x_596_; uint8_t v_isShared_597_; uint8_t v_isSharedCheck_610_; 
lean_dec_ref(v___f_584_);
v_head_594_ = lean_ctor_get(v_parts_591_, 0);
v_isSharedCheck_610_ = !lean_is_exclusive(v_parts_591_);
if (v_isSharedCheck_610_ == 0)
{
lean_object* v_unused_611_; 
v_unused_611_ = lean_ctor_get(v_parts_591_, 1);
lean_dec(v_unused_611_);
v___x_596_ = v_parts_591_;
v_isShared_597_ = v_isSharedCheck_610_;
goto v_resetjp_595_;
}
else
{
lean_inc(v_head_594_);
lean_dec(v_parts_591_);
v___x_596_ = lean_box(0);
v_isShared_597_ = v_isSharedCheck_610_;
goto v_resetjp_595_;
}
v_resetjp_595_:
{
lean_object* v___x_598_; uint8_t v___x_599_; 
v___x_598_ = lean_unsigned_to_nat(100u);
v___x_599_ = lean_nat_dec_lt(v___x_598_, v_p_586_);
if (v___x_599_ == 0)
{
lean_object* v___x_600_; 
lean_del_object(v___x_596_);
v___x_600_ = l_Std_Format_fill(v_head_594_);
return v___x_600_;
}
else
{
lean_object* v___x_601_; lean_object* v___x_602_; lean_object* v___x_604_; 
v___x_601_ = lean_obj_once(&lp_mathlib_TensorProduct_instRepr___redArg___lam__1___closed__6, &lp_mathlib_TensorProduct_instRepr___redArg___lam__1___closed__6_once, _init_lp_mathlib_TensorProduct_instRepr___redArg___lam__1___closed__6);
v___x_602_ = ((lean_object*)(lp_mathlib_TensorProduct_instRepr___redArg___lam__1___closed__7));
if (v_isShared_597_ == 0)
{
lean_ctor_set_tag(v___x_596_, 5);
lean_ctor_set(v___x_596_, 1, v_head_594_);
lean_ctor_set(v___x_596_, 0, v___x_602_);
v___x_604_ = v___x_596_;
goto v_reusejp_603_;
}
else
{
lean_object* v_reuseFailAlloc_609_; 
v_reuseFailAlloc_609_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v_reuseFailAlloc_609_, 0, v___x_602_);
lean_ctor_set(v_reuseFailAlloc_609_, 1, v_head_594_);
v___x_604_ = v_reuseFailAlloc_609_;
goto v_reusejp_603_;
}
v_reusejp_603_:
{
lean_object* v___x_605_; lean_object* v___x_606_; lean_object* v___x_607_; lean_object* v___x_608_; 
v___x_605_ = ((lean_object*)(lp_mathlib_TensorProduct_instRepr___redArg___lam__1___closed__8));
v___x_606_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_606_, 0, v___x_604_);
lean_ctor_set(v___x_606_, 1, v___x_605_);
v___x_607_ = lean_alloc_ctor(4, 2, 0);
lean_ctor_set(v___x_607_, 0, v___x_601_);
lean_ctor_set(v___x_607_, 1, v___x_606_);
v___x_608_ = l_Std_Format_fill(v___x_607_);
return v___x_608_;
}
}
}
}
else
{
lean_object* v___x_612_; uint8_t v___x_613_; lean_object* v___x_614_; lean_object* v___x_615_; 
lean_dec(v_tail_593_);
v___x_612_ = lean_unsigned_to_nat(65u);
v___x_613_ = lean_nat_dec_lt(v___x_612_, v_p_586_);
v___x_614_ = ((lean_object*)(lp_mathlib_TensorProduct_instRepr___redArg___lam__1___closed__11));
v___x_615_ = l_Std_Format_joinSep___redArg(v___f_584_, v_parts_591_, v___x_614_);
if (v___x_613_ == 0)
{
lean_object* v___x_616_; 
v___x_616_ = l_Std_Format_fill(v___x_615_);
return v___x_616_;
}
else
{
lean_object* v___x_617_; lean_object* v___x_618_; lean_object* v___x_619_; lean_object* v___x_620_; lean_object* v___x_621_; lean_object* v___x_622_; lean_object* v___x_623_; 
v___x_617_ = lean_obj_once(&lp_mathlib_TensorProduct_instRepr___redArg___lam__1___closed__6, &lp_mathlib_TensorProduct_instRepr___redArg___lam__1___closed__6_once, _init_lp_mathlib_TensorProduct_instRepr___redArg___lam__1___closed__6);
v___x_618_ = ((lean_object*)(lp_mathlib_TensorProduct_instRepr___redArg___lam__1___closed__7));
v___x_619_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_619_, 0, v___x_618_);
lean_ctor_set(v___x_619_, 1, v___x_615_);
v___x_620_ = ((lean_object*)(lp_mathlib_TensorProduct_instRepr___redArg___lam__1___closed__8));
v___x_621_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_621_, 0, v___x_619_);
lean_ctor_set(v___x_621_, 1, v___x_620_);
v___x_622_ = lean_alloc_ctor(4, 2, 0);
lean_ctor_set(v___x_622_, 0, v___x_617_);
lean_ctor_set(v___x_622_, 1, v___x_621_);
v___x_623_ = l_Std_Format_fill(v___x_622_);
return v___x_623_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_TensorProduct_instRepr___redArg___lam__1___boxed(lean_object* v___f_624_, lean_object* v___f_625_, lean_object* v_mn_626_, lean_object* v_p_627_){
_start:
{
lean_object* v_res_628_; 
v_res_628_ = lp_mathlib_TensorProduct_instRepr___redArg___lam__1(v___f_624_, v___f_625_, v_mn_626_, v_p_627_);
lean_dec(v_p_627_);
return v_res_628_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_TensorProduct_instRepr___redArg(lean_object* v_inst_630_, lean_object* v_inst_631_){
_start:
{
lean_object* v___f_632_; lean_object* v___f_633_; lean_object* v___f_634_; 
v___f_632_ = lean_alloc_closure((void*)(lp_mathlib_TensorProduct_instRepr___redArg___lam__0), 3, 2);
lean_closure_set(v___f_632_, 0, v_inst_630_);
lean_closure_set(v___f_632_, 1, v_inst_631_);
v___f_633_ = ((lean_object*)(lp_mathlib_TensorProduct_instRepr___redArg___closed__0));
v___f_634_ = lean_alloc_closure((void*)(lp_mathlib_TensorProduct_instRepr___redArg___lam__1___boxed), 4, 2);
lean_closure_set(v___f_634_, 0, v___f_632_);
lean_closure_set(v___f_634_, 1, v___f_633_);
return v___f_634_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_TensorProduct_instRepr(lean_object* v_R_635_, lean_object* v_inst_636_, lean_object* v_M_637_, lean_object* v_N_638_, lean_object* v_inst_639_, lean_object* v_inst_640_, lean_object* v_inst_641_, lean_object* v_inst_642_, lean_object* v_inst_643_, lean_object* v_inst_644_){
_start:
{
lean_object* v___x_645_; 
v___x_645_ = lp_mathlib_TensorProduct_instRepr___redArg(v_inst_643_, v_inst_644_);
return v___x_645_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_TensorProduct_instRepr___boxed(lean_object* v_R_646_, lean_object* v_inst_647_, lean_object* v_M_648_, lean_object* v_N_649_, lean_object* v_inst_650_, lean_object* v_inst_651_, lean_object* v_inst_652_, lean_object* v_inst_653_, lean_object* v_inst_654_, lean_object* v_inst_655_){
_start:
{
lean_object* v_res_656_; 
v_res_656_ = lp_mathlib_TensorProduct_instRepr(v_R_646_, v_inst_647_, v_M_648_, v_N_649_, v_inst_650_, v_inst_651_, v_inst_652_, v_inst_653_, v_inst_654_, v_inst_655_);
lean_dec(v_inst_653_);
lean_dec(v_inst_652_);
lean_dec_ref(v_inst_651_);
lean_dec_ref(v_inst_650_);
lean_dec_ref(v_inst_647_);
return v_res_656_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_TensorProduct_uniqueLeft(lean_object* v_R_657_, lean_object* v_inst_658_, lean_object* v_M_659_, lean_object* v_N_660_, lean_object* v_inst_661_, lean_object* v_inst_662_, lean_object* v_inst_663_, lean_object* v_inst_664_, lean_object* v_inst_665_){
_start:
{
lean_object* v___x_666_; lean_object* v_toZero_667_; 
v___x_666_ = lean_obj_once(&lp_mathlib_instZeroTensorProduct___aux__1___closed__2, &lp_mathlib_instZeroTensorProduct___aux__1___closed__2_once, _init_lp_mathlib_instZeroTensorProduct___aux__1___closed__2);
v_toZero_667_ = lean_ctor_get(v___x_666_, 0);
lean_inc(v_toZero_667_);
return v_toZero_667_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_TensorProduct_uniqueLeft___boxed(lean_object* v_R_668_, lean_object* v_inst_669_, lean_object* v_M_670_, lean_object* v_N_671_, lean_object* v_inst_672_, lean_object* v_inst_673_, lean_object* v_inst_674_, lean_object* v_inst_675_, lean_object* v_inst_676_){
_start:
{
lean_object* v_res_677_; 
v_res_677_ = lp_mathlib_TensorProduct_uniqueLeft(v_R_668_, v_inst_669_, v_M_670_, v_N_671_, v_inst_672_, v_inst_673_, v_inst_674_, v_inst_675_, v_inst_676_);
lean_dec(v_inst_675_);
lean_dec(v_inst_674_);
lean_dec_ref(v_inst_673_);
lean_dec_ref(v_inst_672_);
lean_dec_ref(v_inst_669_);
return v_res_677_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_TensorProduct_uniqueRight(lean_object* v_R_678_, lean_object* v_inst_679_, lean_object* v_M_680_, lean_object* v_N_681_, lean_object* v_inst_682_, lean_object* v_inst_683_, lean_object* v_inst_684_, lean_object* v_inst_685_, lean_object* v_inst_686_){
_start:
{
lean_object* v___x_687_; lean_object* v_toZero_688_; 
v___x_687_ = lean_obj_once(&lp_mathlib_instZeroTensorProduct___aux__1___closed__2, &lp_mathlib_instZeroTensorProduct___aux__1___closed__2_once, _init_lp_mathlib_instZeroTensorProduct___aux__1___closed__2);
v_toZero_688_ = lean_ctor_get(v___x_687_, 0);
lean_inc(v_toZero_688_);
return v_toZero_688_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_TensorProduct_uniqueRight___boxed(lean_object* v_R_689_, lean_object* v_inst_690_, lean_object* v_M_691_, lean_object* v_N_692_, lean_object* v_inst_693_, lean_object* v_inst_694_, lean_object* v_inst_695_, lean_object* v_inst_696_, lean_object* v_inst_697_){
_start:
{
lean_object* v_res_698_; 
v_res_698_ = lp_mathlib_TensorProduct_uniqueRight(v_R_689_, v_inst_690_, v_M_691_, v_N_692_, v_inst_693_, v_inst_694_, v_inst_695_, v_inst_696_, v_inst_697_);
lean_dec(v_inst_696_);
lean_dec(v_inst_695_);
lean_dec_ref(v_inst_694_);
lean_dec_ref(v_inst_693_);
lean_dec_ref(v_inst_690_);
return v_res_698_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_LinearAlgebra_TensorProduct_Defs_0__TensorProduct_addMonoidWithWrongNSMul___closed__0(void){
_start:
{
lean_object* v___x_699_; lean_object* v___x_700_; lean_object* v___x_701_; 
v___x_699_ = lean_box(0);
v___x_700_ = lean_obj_once(&lp_mathlib_instZeroTensorProduct___aux__1___closed__0, &lp_mathlib_instZeroTensorProduct___aux__1___closed__0_once, _init_lp_mathlib_instZeroTensorProduct___aux__1___closed__0);
v___x_701_ = lp_mathlib_AddCon_addMonoid___redArg(v___x_700_, v___x_699_);
return v___x_701_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_LinearAlgebra_TensorProduct_Defs_0__TensorProduct_addMonoidWithWrongNSMul(lean_object* v_R_702_, lean_object* v_inst_703_, lean_object* v_M_704_, lean_object* v_N_705_, lean_object* v_inst_706_, lean_object* v_inst_707_, lean_object* v_inst_708_, lean_object* v_inst_709_){
_start:
{
lean_object* v___x_710_; 
v___x_710_ = lean_obj_once(&lp_mathlib___private_Mathlib_LinearAlgebra_TensorProduct_Defs_0__TensorProduct_addMonoidWithWrongNSMul___closed__0, &lp_mathlib___private_Mathlib_LinearAlgebra_TensorProduct_Defs_0__TensorProduct_addMonoidWithWrongNSMul___closed__0_once, _init_lp_mathlib___private_Mathlib_LinearAlgebra_TensorProduct_Defs_0__TensorProduct_addMonoidWithWrongNSMul___closed__0);
return v___x_710_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_LinearAlgebra_TensorProduct_Defs_0__TensorProduct_addMonoidWithWrongNSMul___boxed(lean_object* v_R_711_, lean_object* v_inst_712_, lean_object* v_M_713_, lean_object* v_N_714_, lean_object* v_inst_715_, lean_object* v_inst_716_, lean_object* v_inst_717_, lean_object* v_inst_718_){
_start:
{
lean_object* v_res_719_; 
v_res_719_ = lp_mathlib___private_Mathlib_LinearAlgebra_TensorProduct_Defs_0__TensorProduct_addMonoidWithWrongNSMul(v_R_711_, v_inst_712_, v_M_713_, v_N_714_, v_inst_715_, v_inst_716_, v_inst_717_, v_inst_718_);
lean_dec(v_inst_718_);
lean_dec(v_inst_717_);
lean_dec_ref(v_inst_716_);
lean_dec_ref(v_inst_715_);
lean_dec_ref(v_inst_712_);
return v_res_719_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_TensorProduct_SMul_aux___redArg___lam__0(lean_object* v_inst_720_, lean_object* v_r_721_, lean_object* v_p_722_){
_start:
{
lean_object* v_fst_723_; lean_object* v_snd_724_; lean_object* v___x_725_; lean_object* v___x_726_; 
v_fst_723_ = lean_ctor_get(v_p_722_, 0);
lean_inc(v_fst_723_);
v_snd_724_ = lean_ctor_get(v_p_722_, 1);
lean_inc(v_snd_724_);
lean_dec_ref(v_p_722_);
v___x_725_ = lean_apply_2(v_inst_720_, v_r_721_, v_fst_723_);
v___x_726_ = lp_mathlib_TensorProduct_tmul___redArg(v___x_725_, v_snd_724_);
return v___x_726_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_TensorProduct_SMul_aux___redArg(lean_object* v_inst_727_, lean_object* v_inst_728_, lean_object* v_inst_729_, lean_object* v_inst_730_, lean_object* v_inst_731_, lean_object* v_inst_732_, lean_object* v_r_733_){
_start:
{
lean_object* v___x_734_; lean_object* v___x_735_; lean_object* v_toFun_736_; lean_object* v___f_737_; lean_object* v___x_738_; 
v___x_734_ = lp_mathlib___private_Mathlib_LinearAlgebra_TensorProduct_Defs_0__TensorProduct_addMonoidWithWrongNSMul(lean_box(0), v_inst_727_, lean_box(0), lean_box(0), v_inst_728_, v_inst_729_, v_inst_730_, v_inst_731_);
v___x_735_ = lp_mathlib_FreeAddMonoid_lift___redArg(v___x_734_);
v_toFun_736_ = lean_ctor_get(v___x_735_, 0);
lean_inc(v_toFun_736_);
lean_dec_ref(v___x_735_);
v___f_737_ = lean_alloc_closure((void*)(lp_mathlib_TensorProduct_SMul_aux___redArg___lam__0), 3, 2);
lean_closure_set(v___f_737_, 0, v_inst_732_);
lean_closure_set(v___f_737_, 1, v_r_733_);
v___x_738_ = lean_apply_1(v_toFun_736_, v___f_737_);
return v___x_738_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_TensorProduct_SMul_aux___redArg___boxed(lean_object* v_inst_739_, lean_object* v_inst_740_, lean_object* v_inst_741_, lean_object* v_inst_742_, lean_object* v_inst_743_, lean_object* v_inst_744_, lean_object* v_r_745_){
_start:
{
lean_object* v_res_746_; 
v_res_746_ = lp_mathlib_TensorProduct_SMul_aux___redArg(v_inst_739_, v_inst_740_, v_inst_741_, v_inst_742_, v_inst_743_, v_inst_744_, v_r_745_);
lean_dec(v_inst_743_);
lean_dec(v_inst_742_);
lean_dec_ref(v_inst_741_);
lean_dec_ref(v_inst_740_);
lean_dec_ref(v_inst_739_);
return v_res_746_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_TensorProduct_SMul_aux(lean_object* v_R_747_, lean_object* v_inst_748_, lean_object* v_M_749_, lean_object* v_N_750_, lean_object* v_inst_751_, lean_object* v_inst_752_, lean_object* v_inst_753_, lean_object* v_inst_754_, lean_object* v_R_x27_755_, lean_object* v_inst_756_, lean_object* v_r_757_){
_start:
{
lean_object* v___x_758_; 
v___x_758_ = lp_mathlib_TensorProduct_SMul_aux___redArg(v_inst_748_, v_inst_751_, v_inst_752_, v_inst_753_, v_inst_754_, v_inst_756_, v_r_757_);
return v___x_758_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_TensorProduct_SMul_aux___boxed(lean_object* v_R_759_, lean_object* v_inst_760_, lean_object* v_M_761_, lean_object* v_N_762_, lean_object* v_inst_763_, lean_object* v_inst_764_, lean_object* v_inst_765_, lean_object* v_inst_766_, lean_object* v_R_x27_767_, lean_object* v_inst_768_, lean_object* v_r_769_){
_start:
{
lean_object* v_res_770_; 
v_res_770_ = lp_mathlib_TensorProduct_SMul_aux(v_R_759_, v_inst_760_, v_M_761_, v_N_762_, v_inst_763_, v_inst_764_, v_inst_765_, v_inst_766_, v_R_x27_767_, v_inst_768_, v_r_769_);
lean_dec(v_inst_766_);
lean_dec(v_inst_765_);
lean_dec_ref(v_inst_764_);
lean_dec_ref(v_inst_763_);
lean_dec_ref(v_inst_760_);
return v_res_770_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_TensorProduct_leftHasSMul___redArg___lam__0(lean_object* v_inst_771_, lean_object* v_inst_772_, lean_object* v_inst_773_, lean_object* v_inst_774_, lean_object* v_inst_775_, lean_object* v_inst_776_, lean_object* v_r_777_, lean_object* v___y_778_){
_start:
{
lean_object* v___x_779_; lean_object* v___x_780_; 
v___x_779_ = lp_mathlib_TensorProduct_SMul_aux___redArg(v_inst_771_, v_inst_772_, v_inst_773_, v_inst_774_, v_inst_775_, v_inst_776_, v_r_777_);
v___x_780_ = lp_mathlib_Con_lift___redArg___lam__0(v___x_779_, v___y_778_);
return v___x_780_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_TensorProduct_leftHasSMul___redArg___lam__0___boxed(lean_object* v_inst_781_, lean_object* v_inst_782_, lean_object* v_inst_783_, lean_object* v_inst_784_, lean_object* v_inst_785_, lean_object* v_inst_786_, lean_object* v_r_787_, lean_object* v___y_788_){
_start:
{
lean_object* v_res_789_; 
v_res_789_ = lp_mathlib_TensorProduct_leftHasSMul___redArg___lam__0(v_inst_781_, v_inst_782_, v_inst_783_, v_inst_784_, v_inst_785_, v_inst_786_, v_r_787_, v___y_788_);
lean_dec(v_inst_785_);
lean_dec(v_inst_784_);
lean_dec_ref(v_inst_783_);
lean_dec_ref(v_inst_782_);
lean_dec_ref(v_inst_781_);
return v_res_789_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_TensorProduct_leftHasSMul___redArg(lean_object* v_inst_790_, lean_object* v_inst_791_, lean_object* v_inst_792_, lean_object* v_inst_793_, lean_object* v_inst_794_, lean_object* v_inst_795_){
_start:
{
lean_object* v___f_796_; 
v___f_796_ = lean_alloc_closure((void*)(lp_mathlib_TensorProduct_leftHasSMul___redArg___lam__0___boxed), 8, 6);
lean_closure_set(v___f_796_, 0, v_inst_790_);
lean_closure_set(v___f_796_, 1, v_inst_791_);
lean_closure_set(v___f_796_, 2, v_inst_792_);
lean_closure_set(v___f_796_, 3, v_inst_794_);
lean_closure_set(v___f_796_, 4, v_inst_795_);
lean_closure_set(v___f_796_, 5, v_inst_793_);
return v___f_796_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_TensorProduct_leftHasSMul(lean_object* v_R_797_, lean_object* v_R_x27_798_, lean_object* v_inst_799_, lean_object* v_inst_800_, lean_object* v_M_801_, lean_object* v_N_802_, lean_object* v_inst_803_, lean_object* v_inst_804_, lean_object* v_inst_805_, lean_object* v_inst_806_, lean_object* v_inst_807_, lean_object* v_inst_808_){
_start:
{
lean_object* v___f_809_; 
v___f_809_ = lean_alloc_closure((void*)(lp_mathlib_TensorProduct_leftHasSMul___redArg___lam__0___boxed), 8, 6);
lean_closure_set(v___f_809_, 0, v_inst_799_);
lean_closure_set(v___f_809_, 1, v_inst_803_);
lean_closure_set(v___f_809_, 2, v_inst_804_);
lean_closure_set(v___f_809_, 3, v_inst_806_);
lean_closure_set(v___f_809_, 4, v_inst_807_);
lean_closure_set(v___f_809_, 5, v_inst_805_);
return v___f_809_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_TensorProduct_leftHasSMul___boxed(lean_object* v_R_810_, lean_object* v_R_x27_811_, lean_object* v_inst_812_, lean_object* v_inst_813_, lean_object* v_M_814_, lean_object* v_N_815_, lean_object* v_inst_816_, lean_object* v_inst_817_, lean_object* v_inst_818_, lean_object* v_inst_819_, lean_object* v_inst_820_, lean_object* v_inst_821_){
_start:
{
lean_object* v_res_822_; 
v_res_822_ = lp_mathlib_TensorProduct_leftHasSMul(v_R_810_, v_R_x27_811_, v_inst_812_, v_inst_813_, v_M_814_, v_N_815_, v_inst_816_, v_inst_817_, v_inst_818_, v_inst_819_, v_inst_820_, v_inst_821_);
lean_dec_ref(v_inst_813_);
return v_res_822_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_TensorProduct_instSMul___redArg(lean_object* v_inst_823_, lean_object* v_inst_824_, lean_object* v_inst_825_, lean_object* v_inst_826_, lean_object* v_inst_827_){
_start:
{
lean_object* v___f_828_; 
lean_inc(v_inst_826_);
v___f_828_ = lean_alloc_closure((void*)(lp_mathlib_TensorProduct_leftHasSMul___redArg___lam__0___boxed), 8, 6);
lean_closure_set(v___f_828_, 0, v_inst_823_);
lean_closure_set(v___f_828_, 1, v_inst_824_);
lean_closure_set(v___f_828_, 2, v_inst_825_);
lean_closure_set(v___f_828_, 3, v_inst_826_);
lean_closure_set(v___f_828_, 4, v_inst_827_);
lean_closure_set(v___f_828_, 5, v_inst_826_);
return v___f_828_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_TensorProduct_instSMul(lean_object* v_R_829_, lean_object* v_inst_830_, lean_object* v_M_831_, lean_object* v_N_832_, lean_object* v_inst_833_, lean_object* v_inst_834_, lean_object* v_inst_835_, lean_object* v_inst_836_){
_start:
{
lean_object* v___f_837_; 
lean_inc(v_inst_835_);
v___f_837_ = lean_alloc_closure((void*)(lp_mathlib_TensorProduct_leftHasSMul___redArg___lam__0___boxed), 8, 6);
lean_closure_set(v___f_837_, 0, v_inst_830_);
lean_closure_set(v___f_837_, 1, v_inst_833_);
lean_closure_set(v___f_837_, 2, v_inst_834_);
lean_closure_set(v___f_837_, 3, v_inst_835_);
lean_closure_set(v___f_837_, 4, v_inst_836_);
lean_closure_set(v___f_837_, 5, v_inst_835_);
return v___f_837_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_TensorProduct_addMonoid___redArg(lean_object* v_inst_838_, lean_object* v_inst_839_, lean_object* v_inst_840_, lean_object* v_inst_841_, lean_object* v_inst_842_){
_start:
{
lean_object* v___x_843_; lean_object* v___x_844_; lean_object* v___x_845_; lean_object* v___f_846_; lean_object* v___f_847_; lean_object* v___x_848_; 
v___x_843_ = lp_mathlib_instZeroTensorProduct(lean_box(0), v_inst_838_, lean_box(0), lean_box(0), v_inst_839_, v_inst_840_, v_inst_841_, v_inst_842_);
lean_inc(v_inst_842_);
lean_inc(v_inst_841_);
lean_inc_ref(v_inst_840_);
lean_inc_ref_n(v_inst_839_, 2);
lean_inc_ref(v_inst_838_);
v___x_844_ = lean_alloc_closure((void*)(lp_mathlib_instAddTensorProduct___aux__1___boxed), 10, 8);
lean_closure_set(v___x_844_, 0, lean_box(0));
lean_closure_set(v___x_844_, 1, v_inst_838_);
lean_closure_set(v___x_844_, 2, lean_box(0));
lean_closure_set(v___x_844_, 3, lean_box(0));
lean_closure_set(v___x_844_, 4, v_inst_839_);
lean_closure_set(v___x_844_, 5, v_inst_840_);
lean_closure_set(v___x_844_, 6, v_inst_841_);
lean_closure_set(v___x_844_, 7, v_inst_842_);
v___x_845_ = lp_mathlib_instMulActionNatOfAddMonoid___redArg(v_inst_839_);
v___f_846_ = lean_alloc_closure((void*)(lp_mathlib_TensorProduct_leftHasSMul___redArg___lam__0___boxed), 8, 6);
lean_closure_set(v___f_846_, 0, v_inst_838_);
lean_closure_set(v___f_846_, 1, v_inst_839_);
lean_closure_set(v___f_846_, 2, v_inst_840_);
lean_closure_set(v___f_846_, 3, v_inst_841_);
lean_closure_set(v___f_846_, 4, v_inst_842_);
lean_closure_set(v___f_846_, 5, v___x_845_);
v___f_847_ = lean_alloc_closure((void*)(lp_mathlib_NSMul_toSMul___redArg___lam__0), 3, 1);
lean_closure_set(v___f_847_, 0, v___f_846_);
v___x_848_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_848_, 0, v___x_843_);
lean_ctor_set(v___x_848_, 1, v___x_844_);
lean_ctor_set(v___x_848_, 2, v___f_847_);
return v___x_848_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_TensorProduct_addMonoid(lean_object* v_R_849_, lean_object* v_inst_850_, lean_object* v_M_851_, lean_object* v_N_852_, lean_object* v_inst_853_, lean_object* v_inst_854_, lean_object* v_inst_855_, lean_object* v_inst_856_){
_start:
{
lean_object* v___x_857_; 
v___x_857_ = lp_mathlib_TensorProduct_addMonoid___redArg(v_inst_850_, v_inst_853_, v_inst_854_, v_inst_855_, v_inst_856_);
return v___x_857_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_TensorProduct_addCommMonoid___redArg(lean_object* v_inst_858_, lean_object* v_inst_859_, lean_object* v_inst_860_, lean_object* v_inst_861_, lean_object* v_inst_862_){
_start:
{
lean_object* v___x_863_; 
v___x_863_ = lp_mathlib_TensorProduct_addMonoid___redArg(v_inst_858_, v_inst_859_, v_inst_860_, v_inst_861_, v_inst_862_);
return v___x_863_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_TensorProduct_addCommMonoid(lean_object* v_R_864_, lean_object* v_inst_865_, lean_object* v_M_866_, lean_object* v_N_867_, lean_object* v_inst_868_, lean_object* v_inst_869_, lean_object* v_inst_870_, lean_object* v_inst_871_){
_start:
{
lean_object* v___x_872_; 
v___x_872_ = lp_mathlib_TensorProduct_addMonoid___redArg(v_inst_865_, v_inst_868_, v_inst_869_, v_inst_870_, v_inst_871_);
return v___x_872_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_TensorProduct_leftDistribMulAction___redArg(lean_object* v_inst_873_, lean_object* v_inst_874_, lean_object* v_inst_875_, lean_object* v_inst_876_, lean_object* v_inst_877_, lean_object* v_inst_878_){
_start:
{
lean_object* v___f_879_; 
v___f_879_ = lean_alloc_closure((void*)(lp_mathlib_TensorProduct_leftHasSMul___redArg___lam__0___boxed), 8, 6);
lean_closure_set(v___f_879_, 0, v_inst_873_);
lean_closure_set(v___f_879_, 1, v_inst_874_);
lean_closure_set(v___f_879_, 2, v_inst_875_);
lean_closure_set(v___f_879_, 3, v_inst_877_);
lean_closure_set(v___f_879_, 4, v_inst_878_);
lean_closure_set(v___f_879_, 5, v_inst_876_);
return v___f_879_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_TensorProduct_leftDistribMulAction(lean_object* v_R_880_, lean_object* v_R_x27_881_, lean_object* v_inst_882_, lean_object* v_inst_883_, lean_object* v_M_884_, lean_object* v_N_885_, lean_object* v_inst_886_, lean_object* v_inst_887_, lean_object* v_inst_888_, lean_object* v_inst_889_, lean_object* v_inst_890_, lean_object* v_inst_891_){
_start:
{
lean_object* v___f_892_; 
v___f_892_ = lean_alloc_closure((void*)(lp_mathlib_TensorProduct_leftHasSMul___redArg___lam__0___boxed), 8, 6);
lean_closure_set(v___f_892_, 0, v_inst_882_);
lean_closure_set(v___f_892_, 1, v_inst_886_);
lean_closure_set(v___f_892_, 2, v_inst_887_);
lean_closure_set(v___f_892_, 3, v_inst_889_);
lean_closure_set(v___f_892_, 4, v_inst_890_);
lean_closure_set(v___f_892_, 5, v_inst_888_);
return v___f_892_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_TensorProduct_leftDistribMulAction___boxed(lean_object* v_R_893_, lean_object* v_R_x27_894_, lean_object* v_inst_895_, lean_object* v_inst_896_, lean_object* v_M_897_, lean_object* v_N_898_, lean_object* v_inst_899_, lean_object* v_inst_900_, lean_object* v_inst_901_, lean_object* v_inst_902_, lean_object* v_inst_903_, lean_object* v_inst_904_){
_start:
{
lean_object* v_res_905_; 
v_res_905_ = lp_mathlib_TensorProduct_leftDistribMulAction(v_R_893_, v_R_x27_894_, v_inst_895_, v_inst_896_, v_M_897_, v_N_898_, v_inst_899_, v_inst_900_, v_inst_901_, v_inst_902_, v_inst_903_, v_inst_904_);
lean_dec_ref(v_inst_896_);
return v_res_905_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_TensorProduct_instDistribMulAction___redArg(lean_object* v_inst_906_, lean_object* v_inst_907_, lean_object* v_inst_908_, lean_object* v_inst_909_, lean_object* v_inst_910_){
_start:
{
lean_object* v___f_911_; 
lean_inc(v_inst_909_);
v___f_911_ = lean_alloc_closure((void*)(lp_mathlib_TensorProduct_leftHasSMul___redArg___lam__0___boxed), 8, 6);
lean_closure_set(v___f_911_, 0, v_inst_906_);
lean_closure_set(v___f_911_, 1, v_inst_907_);
lean_closure_set(v___f_911_, 2, v_inst_908_);
lean_closure_set(v___f_911_, 3, v_inst_909_);
lean_closure_set(v___f_911_, 4, v_inst_910_);
lean_closure_set(v___f_911_, 5, v_inst_909_);
return v___f_911_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_TensorProduct_instDistribMulAction(lean_object* v_R_912_, lean_object* v_inst_913_, lean_object* v_M_914_, lean_object* v_N_915_, lean_object* v_inst_916_, lean_object* v_inst_917_, lean_object* v_inst_918_, lean_object* v_inst_919_){
_start:
{
lean_object* v___f_920_; 
lean_inc(v_inst_918_);
v___f_920_ = lean_alloc_closure((void*)(lp_mathlib_TensorProduct_leftHasSMul___redArg___lam__0___boxed), 8, 6);
lean_closure_set(v___f_920_, 0, v_inst_913_);
lean_closure_set(v___f_920_, 1, v_inst_916_);
lean_closure_set(v___f_920_, 2, v_inst_917_);
lean_closure_set(v___f_920_, 3, v_inst_918_);
lean_closure_set(v___f_920_, 4, v_inst_919_);
lean_closure_set(v___f_920_, 5, v_inst_918_);
return v___f_920_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_TensorProduct_leftModule___redArg(lean_object* v_inst_921_, lean_object* v_inst_922_, lean_object* v_inst_923_, lean_object* v_inst_924_, lean_object* v_inst_925_, lean_object* v_inst_926_){
_start:
{
lean_object* v___f_927_; 
v___f_927_ = lean_alloc_closure((void*)(lp_mathlib_TensorProduct_leftHasSMul___redArg___lam__0___boxed), 8, 6);
lean_closure_set(v___f_927_, 0, v_inst_921_);
lean_closure_set(v___f_927_, 1, v_inst_922_);
lean_closure_set(v___f_927_, 2, v_inst_923_);
lean_closure_set(v___f_927_, 3, v_inst_925_);
lean_closure_set(v___f_927_, 4, v_inst_926_);
lean_closure_set(v___f_927_, 5, v_inst_924_);
return v___f_927_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_TensorProduct_leftModule(lean_object* v_R_928_, lean_object* v_R_x27_x27_929_, lean_object* v_inst_930_, lean_object* v_inst_931_, lean_object* v_M_932_, lean_object* v_N_933_, lean_object* v_inst_934_, lean_object* v_inst_935_, lean_object* v_inst_936_, lean_object* v_inst_937_, lean_object* v_inst_938_, lean_object* v_inst_939_){
_start:
{
lean_object* v___f_940_; 
v___f_940_ = lean_alloc_closure((void*)(lp_mathlib_TensorProduct_leftHasSMul___redArg___lam__0___boxed), 8, 6);
lean_closure_set(v___f_940_, 0, v_inst_930_);
lean_closure_set(v___f_940_, 1, v_inst_934_);
lean_closure_set(v___f_940_, 2, v_inst_935_);
lean_closure_set(v___f_940_, 3, v_inst_937_);
lean_closure_set(v___f_940_, 4, v_inst_938_);
lean_closure_set(v___f_940_, 5, v_inst_936_);
return v___f_940_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_TensorProduct_leftModule___boxed(lean_object* v_R_941_, lean_object* v_R_x27_x27_942_, lean_object* v_inst_943_, lean_object* v_inst_944_, lean_object* v_M_945_, lean_object* v_N_946_, lean_object* v_inst_947_, lean_object* v_inst_948_, lean_object* v_inst_949_, lean_object* v_inst_950_, lean_object* v_inst_951_, lean_object* v_inst_952_){
_start:
{
lean_object* v_res_953_; 
v_res_953_ = lp_mathlib_TensorProduct_leftModule(v_R_941_, v_R_x27_x27_942_, v_inst_943_, v_inst_944_, v_M_945_, v_N_946_, v_inst_947_, v_inst_948_, v_inst_949_, v_inst_950_, v_inst_951_, v_inst_952_);
lean_dec_ref(v_inst_944_);
return v_res_953_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_TensorProduct_instModule___redArg(lean_object* v_inst_954_, lean_object* v_inst_955_, lean_object* v_inst_956_, lean_object* v_inst_957_, lean_object* v_inst_958_){
_start:
{
lean_object* v___f_959_; 
lean_inc(v_inst_957_);
v___f_959_ = lean_alloc_closure((void*)(lp_mathlib_TensorProduct_leftHasSMul___redArg___lam__0___boxed), 8, 6);
lean_closure_set(v___f_959_, 0, v_inst_954_);
lean_closure_set(v___f_959_, 1, v_inst_955_);
lean_closure_set(v___f_959_, 2, v_inst_956_);
lean_closure_set(v___f_959_, 3, v_inst_957_);
lean_closure_set(v___f_959_, 4, v_inst_958_);
lean_closure_set(v___f_959_, 5, v_inst_957_);
return v___f_959_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_TensorProduct_instModule(lean_object* v_R_960_, lean_object* v_inst_961_, lean_object* v_M_962_, lean_object* v_N_963_, lean_object* v_inst_964_, lean_object* v_inst_965_, lean_object* v_inst_966_, lean_object* v_inst_967_){
_start:
{
lean_object* v___f_968_; 
lean_inc(v_inst_966_);
v___f_968_ = lean_alloc_closure((void*)(lp_mathlib_TensorProduct_leftHasSMul___redArg___lam__0___boxed), 8, 6);
lean_closure_set(v___f_968_, 0, v_inst_961_);
lean_closure_set(v___f_968_, 1, v_inst_964_);
lean_closure_set(v___f_968_, 2, v_inst_965_);
lean_closure_set(v___f_968_, 3, v_inst_966_);
lean_closure_set(v___f_968_, 4, v_inst_967_);
lean_closure_set(v___f_968_, 5, v_inst_966_);
return v___f_968_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_TensorProduct_mk(lean_object* v_R_972_, lean_object* v_inst_973_, lean_object* v_M_974_, lean_object* v_N_975_, lean_object* v_inst_976_, lean_object* v_inst_977_, lean_object* v_inst_978_, lean_object* v_inst_979_){
_start:
{
lean_object* v___f_980_; 
v___f_980_ = ((lean_object*)(lp_mathlib_TensorProduct_mk___closed__1));
return v___f_980_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_TensorProduct_mk___boxed(lean_object* v_R_981_, lean_object* v_inst_982_, lean_object* v_M_983_, lean_object* v_N_984_, lean_object* v_inst_985_, lean_object* v_inst_986_, lean_object* v_inst_987_, lean_object* v_inst_988_){
_start:
{
lean_object* v_res_989_; 
v_res_989_ = lp_mathlib_TensorProduct_mk(v_R_981_, v_inst_982_, v_M_983_, v_N_984_, v_inst_985_, v_inst_986_, v_inst_987_, v_inst_988_);
lean_dec(v_inst_988_);
lean_dec(v_inst_987_);
lean_dec_ref(v_inst_986_);
lean_dec_ref(v_inst_985_);
lean_dec_ref(v_inst_982_);
return v_res_989_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Module_Submodule_Bilinear(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_GroupTheory_Congruence_Hom(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Tactic_NormNum_Basic(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_LinearAlgebra_TensorProduct_Defs(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Module_Submodule_Bilinear(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_GroupTheory_Congruence_Hom(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Tactic_NormNum_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_LinearAlgebra_TensorProduct_Defs(uint8_t builtin) {
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
lean_object* initialize_mathlib_Mathlib_Algebra_Module_Submodule_Bilinear(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_GroupTheory_Congruence_Hom(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Tactic_NormNum_Basic(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_LinearAlgebra_TensorProduct_Defs(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Module_Submodule_Bilinear(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_GroupTheory_Congruence_Hom(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Tactic_NormNum_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_LinearAlgebra_TensorProduct_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_LinearAlgebra_TensorProduct_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_LinearAlgebra_TensorProduct_Defs(builtin);
}
#ifdef __cplusplus
}
#endif
