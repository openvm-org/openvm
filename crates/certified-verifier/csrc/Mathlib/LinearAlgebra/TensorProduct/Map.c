// Lean compiler output
// Module: Mathlib.LinearAlgebra.TensorProduct.Map
// Imports: public import Init public meta import Init public import Mathlib.LinearAlgebra.TensorProduct.Basic public import Mathlib.Algebra.Module.Shrink
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
lean_object* l_Lean_Name_mkStr3(lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Name_mkStr2(lean_object*, lean_object*);
lean_object* lp_mathlib_NonUnitalRingHom_id___lam__0___boxed(lean_object*);
lean_object* lp_mathlib_TensorProduct_addMonoid___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_TensorProduct_leftHasSMul___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_LinearMap_addMonoid___redArg(lean_object*);
lean_object* lp_mathlib_LinearMap_instSMul___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_LinearMap_llcomp___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_TensorProduct_mk(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_LinearMap_flip___redArg(lean_object*);
lean_object* lp_mathlib_LinearMap_comp___redArg___lam__0(lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_TensorProduct_liftAux___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_LinearEquiv_refl(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_LinearEquiv_symm___redArg(lean_object*);
lean_object* lp_mathlib_LinearMap_compl_u2082___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_LinearEquiv_ofLinearMap___redArg(lean_object*, lean_object*);
lean_object* lp_mathlib_SMulMemClass_subtype___lam__0___boxed(lean_object*);
lean_object* lp_mathlib_AddSubmonoidClass_toAddMonoid___redArg(lean_object*);
lean_object* lp_mathlib_Submodule_smul___redArg___lam__0(lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_LinearMap_mk_u2082_x27_u209b_u2097___redArg___lam__0(lean_object*, lean_object*, lean_object*);
uint8_t l_Lean_Syntax_isOfKind(lean_object*, lean_object*);
lean_object* l_Lean_Syntax_getArg(lean_object*, lean_object*);
lean_object* l_Lean_SourceInfo_fromRef(lean_object*, uint8_t);
lean_object* l_Lean_Name_mkStr4(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_String_toRawSubstring_x27(lean_object*);
lean_object* l_Lean_addMacroScope(lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Name_mkStr1(lean_object*);
lean_object* l_Lean_Syntax_node2(lean_object*, lean_object*, lean_object*, lean_object*);
uint8_t l_Lean_Syntax_matchesNull(lean_object*, lean_object*);
lean_object* l_Lean_replaceRef(lean_object*, lean_object*);
lean_object* l_Lean_Syntax_node3(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_LinearMap_id___lam__0___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_TensorProduct_map___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_TensorProduct_map(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_TensorProduct_map___boxed(lean_object**);
static const lean_string_object lp_mathlib_RingTheory_LinearMap_term___u2297_u2098___00__closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 11, .m_capacity = 11, .m_length = 10, .m_data = "RingTheory"};
static const lean_object* lp_mathlib_RingTheory_LinearMap_term___u2297_u2098___00__closed__0 = (const lean_object*)&lp_mathlib_RingTheory_LinearMap_term___u2297_u2098___00__closed__0_value;
static const lean_string_object lp_mathlib_RingTheory_LinearMap_term___u2297_u2098___00__closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 9, .m_data = "LinearMap"};
static const lean_object* lp_mathlib_RingTheory_LinearMap_term___u2297_u2098___00__closed__1 = (const lean_object*)&lp_mathlib_RingTheory_LinearMap_term___u2297_u2098___00__closed__1_value;
static const lean_string_object lp_mathlib_RingTheory_LinearMap_term___u2297_u2098___00__closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 13, .m_capacity = 13, .m_length = 8, .m_data = "term_⊗ₘ_"};
static const lean_object* lp_mathlib_RingTheory_LinearMap_term___u2297_u2098___00__closed__2 = (const lean_object*)&lp_mathlib_RingTheory_LinearMap_term___u2297_u2098___00__closed__2_value;
static const lean_ctor_object lp_mathlib_RingTheory_LinearMap_term___u2297_u2098___00__closed__3_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_RingTheory_LinearMap_term___u2297_u2098___00__closed__0_value),LEAN_SCALAR_PTR_LITERAL(204, 50, 210, 176, 233, 167, 74, 91)}};
static const lean_ctor_object lp_mathlib_RingTheory_LinearMap_term___u2297_u2098___00__closed__3_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_RingTheory_LinearMap_term___u2297_u2098___00__closed__3_value_aux_0),((lean_object*)&lp_mathlib_RingTheory_LinearMap_term___u2297_u2098___00__closed__1_value),LEAN_SCALAR_PTR_LITERAL(118, 100, 27, 238, 183, 36, 185, 13)}};
static const lean_ctor_object lp_mathlib_RingTheory_LinearMap_term___u2297_u2098___00__closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_RingTheory_LinearMap_term___u2297_u2098___00__closed__3_value_aux_1),((lean_object*)&lp_mathlib_RingTheory_LinearMap_term___u2297_u2098___00__closed__2_value),LEAN_SCALAR_PTR_LITERAL(198, 91, 146, 109, 97, 221, 215, 158)}};
static const lean_object* lp_mathlib_RingTheory_LinearMap_term___u2297_u2098___00__closed__3 = (const lean_object*)&lp_mathlib_RingTheory_LinearMap_term___u2297_u2098___00__closed__3_value;
static const lean_string_object lp_mathlib_RingTheory_LinearMap_term___u2297_u2098___00__closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "andthen"};
static const lean_object* lp_mathlib_RingTheory_LinearMap_term___u2297_u2098___00__closed__4 = (const lean_object*)&lp_mathlib_RingTheory_LinearMap_term___u2297_u2098___00__closed__4_value;
static const lean_ctor_object lp_mathlib_RingTheory_LinearMap_term___u2297_u2098___00__closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_RingTheory_LinearMap_term___u2297_u2098___00__closed__4_value),LEAN_SCALAR_PTR_LITERAL(40, 255, 78, 30, 143, 119, 117, 174)}};
static const lean_object* lp_mathlib_RingTheory_LinearMap_term___u2297_u2098___00__closed__5 = (const lean_object*)&lp_mathlib_RingTheory_LinearMap_term___u2297_u2098___00__closed__5_value;
static const lean_string_object lp_mathlib_RingTheory_LinearMap_term___u2297_u2098___00__closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 4, .m_data = " ⊗ₘ "};
static const lean_object* lp_mathlib_RingTheory_LinearMap_term___u2297_u2098___00__closed__6 = (const lean_object*)&lp_mathlib_RingTheory_LinearMap_term___u2297_u2098___00__closed__6_value;
static const lean_ctor_object lp_mathlib_RingTheory_LinearMap_term___u2297_u2098___00__closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 5}, .m_objs = {((lean_object*)&lp_mathlib_RingTheory_LinearMap_term___u2297_u2098___00__closed__6_value)}};
static const lean_object* lp_mathlib_RingTheory_LinearMap_term___u2297_u2098___00__closed__7 = (const lean_object*)&lp_mathlib_RingTheory_LinearMap_term___u2297_u2098___00__closed__7_value;
static const lean_string_object lp_mathlib_RingTheory_LinearMap_term___u2297_u2098___00__closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "term"};
static const lean_object* lp_mathlib_RingTheory_LinearMap_term___u2297_u2098___00__closed__8 = (const lean_object*)&lp_mathlib_RingTheory_LinearMap_term___u2297_u2098___00__closed__8_value;
static const lean_ctor_object lp_mathlib_RingTheory_LinearMap_term___u2297_u2098___00__closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_RingTheory_LinearMap_term___u2297_u2098___00__closed__8_value),LEAN_SCALAR_PTR_LITERAL(187, 230, 181, 162, 253, 146, 122, 119)}};
static const lean_object* lp_mathlib_RingTheory_LinearMap_term___u2297_u2098___00__closed__9 = (const lean_object*)&lp_mathlib_RingTheory_LinearMap_term___u2297_u2098___00__closed__9_value;
static const lean_ctor_object lp_mathlib_RingTheory_LinearMap_term___u2297_u2098___00__closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 7}, .m_objs = {((lean_object*)&lp_mathlib_RingTheory_LinearMap_term___u2297_u2098___00__closed__9_value),((lean_object*)(((size_t)(71) << 1) | 1))}};
static const lean_object* lp_mathlib_RingTheory_LinearMap_term___u2297_u2098___00__closed__10 = (const lean_object*)&lp_mathlib_RingTheory_LinearMap_term___u2297_u2098___00__closed__10_value;
static const lean_ctor_object lp_mathlib_RingTheory_LinearMap_term___u2297_u2098___00__closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib_RingTheory_LinearMap_term___u2297_u2098___00__closed__5_value),((lean_object*)&lp_mathlib_RingTheory_LinearMap_term___u2297_u2098___00__closed__7_value),((lean_object*)&lp_mathlib_RingTheory_LinearMap_term___u2297_u2098___00__closed__10_value)}};
static const lean_object* lp_mathlib_RingTheory_LinearMap_term___u2297_u2098___00__closed__11 = (const lean_object*)&lp_mathlib_RingTheory_LinearMap_term___u2297_u2098___00__closed__11_value;
static const lean_ctor_object lp_mathlib_RingTheory_LinearMap_term___u2297_u2098___00__closed__12_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*4 + 0, .m_other = 4, .m_tag = 4}, .m_objs = {((lean_object*)&lp_mathlib_RingTheory_LinearMap_term___u2297_u2098___00__closed__3_value),((lean_object*)(((size_t)(70) << 1) | 1)),((lean_object*)(((size_t)(71) << 1) | 1)),((lean_object*)&lp_mathlib_RingTheory_LinearMap_term___u2297_u2098___00__closed__11_value)}};
static const lean_object* lp_mathlib_RingTheory_LinearMap_term___u2297_u2098___00__closed__12 = (const lean_object*)&lp_mathlib_RingTheory_LinearMap_term___u2297_u2098___00__closed__12_value;
LEAN_EXPORT const lean_object* lp_mathlib_RingTheory_LinearMap_term___u2297_u2098__ = (const lean_object*)&lp_mathlib_RingTheory_LinearMap_term___u2297_u2098___00__closed__12_value;
static const lean_string_object lp_mathlib_RingTheory_LinearMap___aux__Mathlib__LinearAlgebra__TensorProduct__Map______macroRules__RingTheory__LinearMap__term___u2297_u2098____1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "Lean"};
static const lean_object* lp_mathlib_RingTheory_LinearMap___aux__Mathlib__LinearAlgebra__TensorProduct__Map______macroRules__RingTheory__LinearMap__term___u2297_u2098____1___closed__0 = (const lean_object*)&lp_mathlib_RingTheory_LinearMap___aux__Mathlib__LinearAlgebra__TensorProduct__Map______macroRules__RingTheory__LinearMap__term___u2297_u2098____1___closed__0_value;
static const lean_string_object lp_mathlib_RingTheory_LinearMap___aux__Mathlib__LinearAlgebra__TensorProduct__Map______macroRules__RingTheory__LinearMap__term___u2297_u2098____1___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "Parser"};
static const lean_object* lp_mathlib_RingTheory_LinearMap___aux__Mathlib__LinearAlgebra__TensorProduct__Map______macroRules__RingTheory__LinearMap__term___u2297_u2098____1___closed__1 = (const lean_object*)&lp_mathlib_RingTheory_LinearMap___aux__Mathlib__LinearAlgebra__TensorProduct__Map______macroRules__RingTheory__LinearMap__term___u2297_u2098____1___closed__1_value;
static const lean_string_object lp_mathlib_RingTheory_LinearMap___aux__Mathlib__LinearAlgebra__TensorProduct__Map______macroRules__RingTheory__LinearMap__term___u2297_u2098____1___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "Term"};
static const lean_object* lp_mathlib_RingTheory_LinearMap___aux__Mathlib__LinearAlgebra__TensorProduct__Map______macroRules__RingTheory__LinearMap__term___u2297_u2098____1___closed__2 = (const lean_object*)&lp_mathlib_RingTheory_LinearMap___aux__Mathlib__LinearAlgebra__TensorProduct__Map______macroRules__RingTheory__LinearMap__term___u2297_u2098____1___closed__2_value;
static const lean_string_object lp_mathlib_RingTheory_LinearMap___aux__Mathlib__LinearAlgebra__TensorProduct__Map______macroRules__RingTheory__LinearMap__term___u2297_u2098____1___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = "app"};
static const lean_object* lp_mathlib_RingTheory_LinearMap___aux__Mathlib__LinearAlgebra__TensorProduct__Map______macroRules__RingTheory__LinearMap__term___u2297_u2098____1___closed__3 = (const lean_object*)&lp_mathlib_RingTheory_LinearMap___aux__Mathlib__LinearAlgebra__TensorProduct__Map______macroRules__RingTheory__LinearMap__term___u2297_u2098____1___closed__3_value;
static const lean_ctor_object lp_mathlib_RingTheory_LinearMap___aux__Mathlib__LinearAlgebra__TensorProduct__Map______macroRules__RingTheory__LinearMap__term___u2297_u2098____1___closed__4_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_RingTheory_LinearMap___aux__Mathlib__LinearAlgebra__TensorProduct__Map______macroRules__RingTheory__LinearMap__term___u2297_u2098____1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib_RingTheory_LinearMap___aux__Mathlib__LinearAlgebra__TensorProduct__Map______macroRules__RingTheory__LinearMap__term___u2297_u2098____1___closed__4_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_RingTheory_LinearMap___aux__Mathlib__LinearAlgebra__TensorProduct__Map______macroRules__RingTheory__LinearMap__term___u2297_u2098____1___closed__4_value_aux_0),((lean_object*)&lp_mathlib_RingTheory_LinearMap___aux__Mathlib__LinearAlgebra__TensorProduct__Map______macroRules__RingTheory__LinearMap__term___u2297_u2098____1___closed__1_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib_RingTheory_LinearMap___aux__Mathlib__LinearAlgebra__TensorProduct__Map______macroRules__RingTheory__LinearMap__term___u2297_u2098____1___closed__4_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_RingTheory_LinearMap___aux__Mathlib__LinearAlgebra__TensorProduct__Map______macroRules__RingTheory__LinearMap__term___u2297_u2098____1___closed__4_value_aux_1),((lean_object*)&lp_mathlib_RingTheory_LinearMap___aux__Mathlib__LinearAlgebra__TensorProduct__Map______macroRules__RingTheory__LinearMap__term___u2297_u2098____1___closed__2_value),LEAN_SCALAR_PTR_LITERAL(75, 170, 162, 138, 136, 204, 251, 229)}};
static const lean_ctor_object lp_mathlib_RingTheory_LinearMap___aux__Mathlib__LinearAlgebra__TensorProduct__Map______macroRules__RingTheory__LinearMap__term___u2297_u2098____1___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_RingTheory_LinearMap___aux__Mathlib__LinearAlgebra__TensorProduct__Map______macroRules__RingTheory__LinearMap__term___u2297_u2098____1___closed__4_value_aux_2),((lean_object*)&lp_mathlib_RingTheory_LinearMap___aux__Mathlib__LinearAlgebra__TensorProduct__Map______macroRules__RingTheory__LinearMap__term___u2297_u2098____1___closed__3_value),LEAN_SCALAR_PTR_LITERAL(69, 118, 10, 41, 220, 156, 243, 179)}};
static const lean_object* lp_mathlib_RingTheory_LinearMap___aux__Mathlib__LinearAlgebra__TensorProduct__Map______macroRules__RingTheory__LinearMap__term___u2297_u2098____1___closed__4 = (const lean_object*)&lp_mathlib_RingTheory_LinearMap___aux__Mathlib__LinearAlgebra__TensorProduct__Map______macroRules__RingTheory__LinearMap__term___u2297_u2098____1___closed__4_value;
static const lean_string_object lp_mathlib_RingTheory_LinearMap___aux__Mathlib__LinearAlgebra__TensorProduct__Map______macroRules__RingTheory__LinearMap__term___u2297_u2098____1___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 18, .m_capacity = 18, .m_length = 17, .m_data = "TensorProduct.map"};
static const lean_object* lp_mathlib_RingTheory_LinearMap___aux__Mathlib__LinearAlgebra__TensorProduct__Map______macroRules__RingTheory__LinearMap__term___u2297_u2098____1___closed__5 = (const lean_object*)&lp_mathlib_RingTheory_LinearMap___aux__Mathlib__LinearAlgebra__TensorProduct__Map______macroRules__RingTheory__LinearMap__term___u2297_u2098____1___closed__5_value;
static lean_once_cell_t lp_mathlib_RingTheory_LinearMap___aux__Mathlib__LinearAlgebra__TensorProduct__Map______macroRules__RingTheory__LinearMap__term___u2297_u2098____1___closed__6_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_RingTheory_LinearMap___aux__Mathlib__LinearAlgebra__TensorProduct__Map______macroRules__RingTheory__LinearMap__term___u2297_u2098____1___closed__6;
static const lean_string_object lp_mathlib_RingTheory_LinearMap___aux__Mathlib__LinearAlgebra__TensorProduct__Map______macroRules__RingTheory__LinearMap__term___u2297_u2098____1___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 14, .m_capacity = 14, .m_length = 13, .m_data = "TensorProduct"};
static const lean_object* lp_mathlib_RingTheory_LinearMap___aux__Mathlib__LinearAlgebra__TensorProduct__Map______macroRules__RingTheory__LinearMap__term___u2297_u2098____1___closed__7 = (const lean_object*)&lp_mathlib_RingTheory_LinearMap___aux__Mathlib__LinearAlgebra__TensorProduct__Map______macroRules__RingTheory__LinearMap__term___u2297_u2098____1___closed__7_value;
static const lean_string_object lp_mathlib_RingTheory_LinearMap___aux__Mathlib__LinearAlgebra__TensorProduct__Map______macroRules__RingTheory__LinearMap__term___u2297_u2098____1___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = "map"};
static const lean_object* lp_mathlib_RingTheory_LinearMap___aux__Mathlib__LinearAlgebra__TensorProduct__Map______macroRules__RingTheory__LinearMap__term___u2297_u2098____1___closed__8 = (const lean_object*)&lp_mathlib_RingTheory_LinearMap___aux__Mathlib__LinearAlgebra__TensorProduct__Map______macroRules__RingTheory__LinearMap__term___u2297_u2098____1___closed__8_value;
static const lean_ctor_object lp_mathlib_RingTheory_LinearMap___aux__Mathlib__LinearAlgebra__TensorProduct__Map______macroRules__RingTheory__LinearMap__term___u2297_u2098____1___closed__9_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_RingTheory_LinearMap___aux__Mathlib__LinearAlgebra__TensorProduct__Map______macroRules__RingTheory__LinearMap__term___u2297_u2098____1___closed__7_value),LEAN_SCALAR_PTR_LITERAL(224, 202, 230, 33, 181, 133, 17, 186)}};
static const lean_ctor_object lp_mathlib_RingTheory_LinearMap___aux__Mathlib__LinearAlgebra__TensorProduct__Map______macroRules__RingTheory__LinearMap__term___u2297_u2098____1___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_RingTheory_LinearMap___aux__Mathlib__LinearAlgebra__TensorProduct__Map______macroRules__RingTheory__LinearMap__term___u2297_u2098____1___closed__9_value_aux_0),((lean_object*)&lp_mathlib_RingTheory_LinearMap___aux__Mathlib__LinearAlgebra__TensorProduct__Map______macroRules__RingTheory__LinearMap__term___u2297_u2098____1___closed__8_value),LEAN_SCALAR_PTR_LITERAL(147, 70, 65, 217, 199, 185, 115, 111)}};
static const lean_object* lp_mathlib_RingTheory_LinearMap___aux__Mathlib__LinearAlgebra__TensorProduct__Map______macroRules__RingTheory__LinearMap__term___u2297_u2098____1___closed__9 = (const lean_object*)&lp_mathlib_RingTheory_LinearMap___aux__Mathlib__LinearAlgebra__TensorProduct__Map______macroRules__RingTheory__LinearMap__term___u2297_u2098____1___closed__9_value;
static const lean_ctor_object lp_mathlib_RingTheory_LinearMap___aux__Mathlib__LinearAlgebra__TensorProduct__Map______macroRules__RingTheory__LinearMap__term___u2297_u2098____1___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_RingTheory_LinearMap___aux__Mathlib__LinearAlgebra__TensorProduct__Map______macroRules__RingTheory__LinearMap__term___u2297_u2098____1___closed__9_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_RingTheory_LinearMap___aux__Mathlib__LinearAlgebra__TensorProduct__Map______macroRules__RingTheory__LinearMap__term___u2297_u2098____1___closed__10 = (const lean_object*)&lp_mathlib_RingTheory_LinearMap___aux__Mathlib__LinearAlgebra__TensorProduct__Map______macroRules__RingTheory__LinearMap__term___u2297_u2098____1___closed__10_value;
static const lean_ctor_object lp_mathlib_RingTheory_LinearMap___aux__Mathlib__LinearAlgebra__TensorProduct__Map______macroRules__RingTheory__LinearMap__term___u2297_u2098____1___closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_RingTheory_LinearMap___aux__Mathlib__LinearAlgebra__TensorProduct__Map______macroRules__RingTheory__LinearMap__term___u2297_u2098____1___closed__10_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_RingTheory_LinearMap___aux__Mathlib__LinearAlgebra__TensorProduct__Map______macroRules__RingTheory__LinearMap__term___u2297_u2098____1___closed__11 = (const lean_object*)&lp_mathlib_RingTheory_LinearMap___aux__Mathlib__LinearAlgebra__TensorProduct__Map______macroRules__RingTheory__LinearMap__term___u2297_u2098____1___closed__11_value;
static const lean_string_object lp_mathlib_RingTheory_LinearMap___aux__Mathlib__LinearAlgebra__TensorProduct__Map______macroRules__RingTheory__LinearMap__term___u2297_u2098____1___closed__12_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "null"};
static const lean_object* lp_mathlib_RingTheory_LinearMap___aux__Mathlib__LinearAlgebra__TensorProduct__Map______macroRules__RingTheory__LinearMap__term___u2297_u2098____1___closed__12 = (const lean_object*)&lp_mathlib_RingTheory_LinearMap___aux__Mathlib__LinearAlgebra__TensorProduct__Map______macroRules__RingTheory__LinearMap__term___u2297_u2098____1___closed__12_value;
static const lean_ctor_object lp_mathlib_RingTheory_LinearMap___aux__Mathlib__LinearAlgebra__TensorProduct__Map______macroRules__RingTheory__LinearMap__term___u2297_u2098____1___closed__13_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_RingTheory_LinearMap___aux__Mathlib__LinearAlgebra__TensorProduct__Map______macroRules__RingTheory__LinearMap__term___u2297_u2098____1___closed__12_value),LEAN_SCALAR_PTR_LITERAL(24, 58, 49, 223, 146, 207, 197, 136)}};
static const lean_object* lp_mathlib_RingTheory_LinearMap___aux__Mathlib__LinearAlgebra__TensorProduct__Map______macroRules__RingTheory__LinearMap__term___u2297_u2098____1___closed__13 = (const lean_object*)&lp_mathlib_RingTheory_LinearMap___aux__Mathlib__LinearAlgebra__TensorProduct__Map______macroRules__RingTheory__LinearMap__term___u2297_u2098____1___closed__13_value;
LEAN_EXPORT lean_object* lp_mathlib_RingTheory_LinearMap___aux__Mathlib__LinearAlgebra__TensorProduct__Map______macroRules__RingTheory__LinearMap__term___u2297_u2098____1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RingTheory_LinearMap___aux__Mathlib__LinearAlgebra__TensorProduct__Map______macroRules__RingTheory__LinearMap__term___u2297_u2098____1___boxed(lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_RingTheory_LinearMap___aux__Mathlib__LinearAlgebra__TensorProduct__Map______unexpand__TensorProduct__map__1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "ident"};
static const lean_object* lp_mathlib_RingTheory_LinearMap___aux__Mathlib__LinearAlgebra__TensorProduct__Map______unexpand__TensorProduct__map__1___closed__0 = (const lean_object*)&lp_mathlib_RingTheory_LinearMap___aux__Mathlib__LinearAlgebra__TensorProduct__Map______unexpand__TensorProduct__map__1___closed__0_value;
static const lean_ctor_object lp_mathlib_RingTheory_LinearMap___aux__Mathlib__LinearAlgebra__TensorProduct__Map______unexpand__TensorProduct__map__1___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_RingTheory_LinearMap___aux__Mathlib__LinearAlgebra__TensorProduct__Map______unexpand__TensorProduct__map__1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(52, 159, 208, 51, 14, 60, 6, 71)}};
static const lean_object* lp_mathlib_RingTheory_LinearMap___aux__Mathlib__LinearAlgebra__TensorProduct__Map______unexpand__TensorProduct__map__1___closed__1 = (const lean_object*)&lp_mathlib_RingTheory_LinearMap___aux__Mathlib__LinearAlgebra__TensorProduct__Map______unexpand__TensorProduct__map__1___closed__1_value;
LEAN_EXPORT lean_object* lp_mathlib_RingTheory_LinearMap___aux__Mathlib__LinearAlgebra__TensorProduct__Map______unexpand__TensorProduct__map__1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RingTheory_LinearMap___aux__Mathlib__LinearAlgebra__TensorProduct__Map______unexpand__TensorProduct__map__1___boxed(lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_TensorProduct_mapIncl___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_NonUnitalRingHom_id___lam__0___boxed, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_TensorProduct_mapIncl___redArg___closed__0 = (const lean_object*)&lp_mathlib_TensorProduct_mapIncl___redArg___closed__0_value;
static const lean_closure_object lp_mathlib_TensorProduct_mapIncl___redArg___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_SMulMemClass_subtype___lam__0___boxed, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_TensorProduct_mapIncl___redArg___closed__1 = (const lean_object*)&lp_mathlib_TensorProduct_mapIncl___redArg___closed__1_value;
LEAN_EXPORT lean_object* lp_mathlib_TensorProduct_mapIncl___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_TensorProduct_mapIncl(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_TensorProduct_mapBilinear___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_TensorProduct_mapBilinear(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_TensorProduct_mapBilinear___boxed(lean_object**);
LEAN_EXPORT lean_object* lp_mathlib_TensorProduct_lTensorHomToHomLTensor___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_TensorProduct_lTensorHomToHomLTensor(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_TensorProduct_rTensorHomToHomRTensor___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_TensorProduct_rTensorHomToHomRTensor(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_TensorProduct_homTensorHomMap___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_TensorProduct_homTensorHomMap(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_TensorProduct_homTensorHomMap___boxed(lean_object**);
LEAN_EXPORT lean_object* lp_mathlib_TensorProduct_map_u2082___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_TensorProduct_map_u2082___redArg___boxed(lean_object**);
LEAN_EXPORT lean_object* lp_mathlib_TensorProduct_map_u2082(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_TensorProduct_map_u2082___boxed(lean_object**);
LEAN_EXPORT lean_object* lp_mathlib_TensorProduct_congr___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_TensorProduct_congr(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_TensorProduct_congr___boxed(lean_object**);
static const lean_closure_object lp_mathlib_LinearMap_lTensor___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_LinearMap_id___lam__0___boxed, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_LinearMap_lTensor___redArg___closed__0 = (const lean_object*)&lp_mathlib_LinearMap_lTensor___redArg___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_LinearMap_lTensor___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LinearMap_lTensor(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LinearMap_rTensor___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LinearMap_rTensor(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LinearMap_rTensor___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LinearMap_lTensorHom___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LinearMap_lTensorHom(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LinearMap_rTensorHom___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LinearMap_rTensorHom___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LinearMap_rTensorHom(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LinearMap_rTensorHom___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LinearEquiv_lTensor___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LinearEquiv_lTensor(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LinearEquiv_rTensor___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LinearEquiv_rTensor(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_TensorProduct_map___redArg(lean_object* v_inst_1_, lean_object* v_inst_2_, lean_object* v_00_u03c3_u2081_u2082_3_, lean_object* v_inst_4_, lean_object* v_inst_5_, lean_object* v_inst_6_, lean_object* v_inst_7_, lean_object* v_inst_8_, lean_object* v_inst_9_, lean_object* v_f_10_, lean_object* v_g_11_){
_start:
{
lean_object* v___x_12_; lean_object* v___f_13_; lean_object* v___x_14_; lean_object* v___f_15_; lean_object* v___f_16_; lean_object* v___x_17_; 
lean_inc_n(v_inst_9_, 2);
lean_inc_n(v_inst_8_, 3);
lean_inc_ref_n(v_inst_6_, 2);
lean_inc_ref_n(v_inst_5_, 2);
lean_inc_ref_n(v_inst_2_, 2);
v___x_12_ = lp_mathlib_TensorProduct_addMonoid___redArg(v_inst_2_, v_inst_5_, v_inst_6_, v_inst_8_, v_inst_9_);
v___f_13_ = lean_alloc_closure((void*)(lp_mathlib_TensorProduct_leftHasSMul___redArg___lam__0___boxed), 8, 6);
lean_closure_set(v___f_13_, 0, v_inst_2_);
lean_closure_set(v___f_13_, 1, v_inst_5_);
lean_closure_set(v___f_13_, 2, v_inst_6_);
lean_closure_set(v___f_13_, 3, v_inst_8_);
lean_closure_set(v___f_13_, 4, v_inst_9_);
lean_closure_set(v___f_13_, 5, v_inst_8_);
v___x_14_ = lp_mathlib_TensorProduct_mk(lean_box(0), v_inst_2_, lean_box(0), lean_box(0), v_inst_5_, v_inst_6_, v_inst_8_, v_inst_9_);
lean_dec(v_inst_9_);
lean_dec(v_inst_8_);
lean_dec_ref(v_inst_6_);
lean_dec_ref(v_inst_5_);
v___f_15_ = lean_alloc_closure((void*)(lp_mathlib_LinearMap_compl_u2082___redArg___lam__0), 4, 2);
lean_closure_set(v___f_15_, 0, v___x_14_);
lean_closure_set(v___f_15_, 1, v_g_11_);
v___f_16_ = lean_alloc_closure((void*)(lp_mathlib_LinearMap_comp___redArg___lam__0), 3, 2);
lean_closure_set(v___f_16_, 0, v_f_10_);
lean_closure_set(v___f_16_, 1, v___f_15_);
v___x_17_ = lp_mathlib_TensorProduct_liftAux___redArg(v_inst_1_, v_inst_2_, v_00_u03c3_u2081_u2082_3_, v_inst_4_, v___x_12_, v_inst_7_, v___f_13_, v___f_16_);
return v___x_17_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_TensorProduct_map(lean_object* v_R_18_, lean_object* v_R_u2082_19_, lean_object* v_inst_20_, lean_object* v_inst_21_, lean_object* v_00_u03c3_u2081_u2082_22_, lean_object* v_M_23_, lean_object* v_N_24_, lean_object* v_M_u2082_25_, lean_object* v_N_u2082_26_, lean_object* v_inst_27_, lean_object* v_inst_28_, lean_object* v_inst_29_, lean_object* v_inst_30_, lean_object* v_inst_31_, lean_object* v_inst_32_, lean_object* v_inst_33_, lean_object* v_inst_34_, lean_object* v_f_35_, lean_object* v_g_36_){
_start:
{
lean_object* v___x_37_; 
v___x_37_ = lp_mathlib_TensorProduct_map___redArg(v_inst_20_, v_inst_21_, v_00_u03c3_u2081_u2082_22_, v_inst_28_, v_inst_29_, v_inst_30_, v_inst_32_, v_inst_33_, v_inst_34_, v_f_35_, v_g_36_);
return v___x_37_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_TensorProduct_map___boxed(lean_object** _args){
lean_object* v_R_38_ = _args[0];
lean_object* v_R_u2082_39_ = _args[1];
lean_object* v_inst_40_ = _args[2];
lean_object* v_inst_41_ = _args[3];
lean_object* v_00_u03c3_u2081_u2082_42_ = _args[4];
lean_object* v_M_43_ = _args[5];
lean_object* v_N_44_ = _args[6];
lean_object* v_M_u2082_45_ = _args[7];
lean_object* v_N_u2082_46_ = _args[8];
lean_object* v_inst_47_ = _args[9];
lean_object* v_inst_48_ = _args[10];
lean_object* v_inst_49_ = _args[11];
lean_object* v_inst_50_ = _args[12];
lean_object* v_inst_51_ = _args[13];
lean_object* v_inst_52_ = _args[14];
lean_object* v_inst_53_ = _args[15];
lean_object* v_inst_54_ = _args[16];
lean_object* v_f_55_ = _args[17];
lean_object* v_g_56_ = _args[18];
_start:
{
lean_object* v_res_57_; 
v_res_57_ = lp_mathlib_TensorProduct_map(v_R_38_, v_R_u2082_39_, v_inst_40_, v_inst_41_, v_00_u03c3_u2081_u2082_42_, v_M_43_, v_N_44_, v_M_u2082_45_, v_N_u2082_46_, v_inst_47_, v_inst_48_, v_inst_49_, v_inst_50_, v_inst_51_, v_inst_52_, v_inst_53_, v_inst_54_, v_f_55_, v_g_56_);
lean_dec(v_inst_51_);
lean_dec_ref(v_inst_47_);
return v_res_57_;
}
}
static lean_object* _init_lp_mathlib_RingTheory_LinearMap___aux__Mathlib__LinearAlgebra__TensorProduct__Map______macroRules__RingTheory__LinearMap__term___u2297_u2098____1___closed__6(void){
_start:
{
lean_object* v___x_97_; lean_object* v___x_98_; 
v___x_97_ = ((lean_object*)(lp_mathlib_RingTheory_LinearMap___aux__Mathlib__LinearAlgebra__TensorProduct__Map______macroRules__RingTheory__LinearMap__term___u2297_u2098____1___closed__5));
v___x_98_ = l_String_toRawSubstring_x27(v___x_97_);
return v___x_98_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RingTheory_LinearMap___aux__Mathlib__LinearAlgebra__TensorProduct__Map______macroRules__RingTheory__LinearMap__term___u2297_u2098____1(lean_object* v_x_113_, lean_object* v_a_114_, lean_object* v_a_115_){
_start:
{
lean_object* v___x_116_; uint8_t v___x_117_; 
v___x_116_ = ((lean_object*)(lp_mathlib_RingTheory_LinearMap_term___u2297_u2098___00__closed__3));
lean_inc(v_x_113_);
v___x_117_ = l_Lean_Syntax_isOfKind(v_x_113_, v___x_116_);
if (v___x_117_ == 0)
{
lean_object* v___x_118_; lean_object* v___x_119_; 
lean_dec(v_x_113_);
v___x_118_ = lean_box(1);
v___x_119_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_119_, 0, v___x_118_);
lean_ctor_set(v___x_119_, 1, v_a_115_);
return v___x_119_;
}
else
{
lean_object* v_quotContext_120_; lean_object* v_currMacroScope_121_; lean_object* v_ref_122_; lean_object* v___x_123_; lean_object* v___x_124_; lean_object* v___x_125_; lean_object* v___x_126_; uint8_t v___x_127_; lean_object* v___x_128_; lean_object* v___x_129_; lean_object* v___x_130_; lean_object* v___x_131_; lean_object* v___x_132_; lean_object* v___x_133_; lean_object* v___x_134_; lean_object* v___x_135_; lean_object* v___x_136_; lean_object* v___x_137_; lean_object* v___x_138_; 
v_quotContext_120_ = lean_ctor_get(v_a_114_, 1);
v_currMacroScope_121_ = lean_ctor_get(v_a_114_, 2);
v_ref_122_ = lean_ctor_get(v_a_114_, 5);
v___x_123_ = lean_unsigned_to_nat(0u);
v___x_124_ = l_Lean_Syntax_getArg(v_x_113_, v___x_123_);
v___x_125_ = lean_unsigned_to_nat(2u);
v___x_126_ = l_Lean_Syntax_getArg(v_x_113_, v___x_125_);
lean_dec(v_x_113_);
v___x_127_ = 0;
v___x_128_ = l_Lean_SourceInfo_fromRef(v_ref_122_, v___x_127_);
v___x_129_ = ((lean_object*)(lp_mathlib_RingTheory_LinearMap___aux__Mathlib__LinearAlgebra__TensorProduct__Map______macroRules__RingTheory__LinearMap__term___u2297_u2098____1___closed__4));
v___x_130_ = lean_obj_once(&lp_mathlib_RingTheory_LinearMap___aux__Mathlib__LinearAlgebra__TensorProduct__Map______macroRules__RingTheory__LinearMap__term___u2297_u2098____1___closed__6, &lp_mathlib_RingTheory_LinearMap___aux__Mathlib__LinearAlgebra__TensorProduct__Map______macroRules__RingTheory__LinearMap__term___u2297_u2098____1___closed__6_once, _init_lp_mathlib_RingTheory_LinearMap___aux__Mathlib__LinearAlgebra__TensorProduct__Map______macroRules__RingTheory__LinearMap__term___u2297_u2098____1___closed__6);
v___x_131_ = ((lean_object*)(lp_mathlib_RingTheory_LinearMap___aux__Mathlib__LinearAlgebra__TensorProduct__Map______macroRules__RingTheory__LinearMap__term___u2297_u2098____1___closed__9));
lean_inc(v_currMacroScope_121_);
lean_inc(v_quotContext_120_);
v___x_132_ = l_Lean_addMacroScope(v_quotContext_120_, v___x_131_, v_currMacroScope_121_);
v___x_133_ = ((lean_object*)(lp_mathlib_RingTheory_LinearMap___aux__Mathlib__LinearAlgebra__TensorProduct__Map______macroRules__RingTheory__LinearMap__term___u2297_u2098____1___closed__11));
lean_inc_n(v___x_128_, 2);
v___x_134_ = lean_alloc_ctor(3, 4, 0);
lean_ctor_set(v___x_134_, 0, v___x_128_);
lean_ctor_set(v___x_134_, 1, v___x_130_);
lean_ctor_set(v___x_134_, 2, v___x_132_);
lean_ctor_set(v___x_134_, 3, v___x_133_);
v___x_135_ = ((lean_object*)(lp_mathlib_RingTheory_LinearMap___aux__Mathlib__LinearAlgebra__TensorProduct__Map______macroRules__RingTheory__LinearMap__term___u2297_u2098____1___closed__13));
v___x_136_ = l_Lean_Syntax_node2(v___x_128_, v___x_135_, v___x_124_, v___x_126_);
v___x_137_ = l_Lean_Syntax_node2(v___x_128_, v___x_129_, v___x_134_, v___x_136_);
v___x_138_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_138_, 0, v___x_137_);
lean_ctor_set(v___x_138_, 1, v_a_115_);
return v___x_138_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_RingTheory_LinearMap___aux__Mathlib__LinearAlgebra__TensorProduct__Map______macroRules__RingTheory__LinearMap__term___u2297_u2098____1___boxed(lean_object* v_x_139_, lean_object* v_a_140_, lean_object* v_a_141_){
_start:
{
lean_object* v_res_142_; 
v_res_142_ = lp_mathlib_RingTheory_LinearMap___aux__Mathlib__LinearAlgebra__TensorProduct__Map______macroRules__RingTheory__LinearMap__term___u2297_u2098____1(v_x_139_, v_a_140_, v_a_141_);
lean_dec_ref(v_a_140_);
return v_res_142_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RingTheory_LinearMap___aux__Mathlib__LinearAlgebra__TensorProduct__Map______unexpand__TensorProduct__map__1(lean_object* v_x_146_, lean_object* v_a_147_, lean_object* v_a_148_){
_start:
{
lean_object* v___x_149_; uint8_t v___x_150_; 
v___x_149_ = ((lean_object*)(lp_mathlib_RingTheory_LinearMap___aux__Mathlib__LinearAlgebra__TensorProduct__Map______macroRules__RingTheory__LinearMap__term___u2297_u2098____1___closed__4));
lean_inc(v_x_146_);
v___x_150_ = l_Lean_Syntax_isOfKind(v_x_146_, v___x_149_);
if (v___x_150_ == 0)
{
lean_object* v___x_151_; lean_object* v___x_152_; 
lean_dec(v_x_146_);
v___x_151_ = lean_box(0);
v___x_152_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_152_, 0, v___x_151_);
lean_ctor_set(v___x_152_, 1, v_a_148_);
return v___x_152_;
}
else
{
lean_object* v___x_153_; lean_object* v___x_154_; lean_object* v___x_155_; uint8_t v___x_156_; 
v___x_153_ = lean_unsigned_to_nat(0u);
v___x_154_ = l_Lean_Syntax_getArg(v_x_146_, v___x_153_);
v___x_155_ = ((lean_object*)(lp_mathlib_RingTheory_LinearMap___aux__Mathlib__LinearAlgebra__TensorProduct__Map______unexpand__TensorProduct__map__1___closed__1));
lean_inc(v___x_154_);
v___x_156_ = l_Lean_Syntax_isOfKind(v___x_154_, v___x_155_);
if (v___x_156_ == 0)
{
lean_object* v___x_157_; lean_object* v___x_158_; 
lean_dec(v___x_154_);
lean_dec(v_x_146_);
v___x_157_ = lean_box(0);
v___x_158_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_158_, 0, v___x_157_);
lean_ctor_set(v___x_158_, 1, v_a_148_);
return v___x_158_;
}
else
{
lean_object* v___x_159_; lean_object* v___x_160_; lean_object* v___x_161_; uint8_t v___x_162_; 
v___x_159_ = lean_unsigned_to_nat(1u);
v___x_160_ = l_Lean_Syntax_getArg(v_x_146_, v___x_159_);
lean_dec(v_x_146_);
v___x_161_ = lean_unsigned_to_nat(2u);
lean_inc(v___x_160_);
v___x_162_ = l_Lean_Syntax_matchesNull(v___x_160_, v___x_161_);
if (v___x_162_ == 0)
{
lean_object* v___x_163_; lean_object* v___x_164_; 
lean_dec(v___x_160_);
lean_dec(v___x_154_);
v___x_163_ = lean_box(0);
v___x_164_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_164_, 0, v___x_163_);
lean_ctor_set(v___x_164_, 1, v_a_148_);
return v___x_164_;
}
else
{
lean_object* v___x_165_; lean_object* v___x_166_; lean_object* v_ref_167_; uint8_t v___x_168_; lean_object* v___x_169_; lean_object* v___x_170_; lean_object* v___x_171_; lean_object* v___x_172_; lean_object* v___x_173_; lean_object* v___x_174_; 
v___x_165_ = l_Lean_Syntax_getArg(v___x_160_, v___x_153_);
v___x_166_ = l_Lean_Syntax_getArg(v___x_160_, v___x_159_);
lean_dec(v___x_160_);
v_ref_167_ = l_Lean_replaceRef(v___x_154_, v_a_147_);
lean_dec(v___x_154_);
v___x_168_ = 0;
v___x_169_ = l_Lean_SourceInfo_fromRef(v_ref_167_, v___x_168_);
lean_dec(v_ref_167_);
v___x_170_ = ((lean_object*)(lp_mathlib_RingTheory_LinearMap_term___u2297_u2098___00__closed__3));
v___x_171_ = ((lean_object*)(lp_mathlib_RingTheory_LinearMap_term___u2297_u2098___00__closed__6));
lean_inc(v___x_169_);
v___x_172_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_172_, 0, v___x_169_);
lean_ctor_set(v___x_172_, 1, v___x_171_);
v___x_173_ = l_Lean_Syntax_node3(v___x_169_, v___x_170_, v___x_165_, v___x_172_, v___x_166_);
v___x_174_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_174_, 0, v___x_173_);
lean_ctor_set(v___x_174_, 1, v_a_148_);
return v___x_174_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_RingTheory_LinearMap___aux__Mathlib__LinearAlgebra__TensorProduct__Map______unexpand__TensorProduct__map__1___boxed(lean_object* v_x_175_, lean_object* v_a_176_, lean_object* v_a_177_){
_start:
{
lean_object* v_res_178_; 
v_res_178_ = lp_mathlib_RingTheory_LinearMap___aux__Mathlib__LinearAlgebra__TensorProduct__Map______unexpand__TensorProduct__map__1(v_x_175_, v_a_176_, v_a_177_);
lean_dec(v_a_176_);
return v_res_178_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_TensorProduct_mapIncl___redArg(lean_object* v_inst_181_, lean_object* v_inst_182_, lean_object* v_inst_183_, lean_object* v_inst_184_, lean_object* v_inst_185_){
_start:
{
lean_object* v___f_186_; lean_object* v___x_187_; lean_object* v___f_188_; lean_object* v___f_189_; lean_object* v___x_190_; 
v___f_186_ = ((lean_object*)(lp_mathlib_TensorProduct_mapIncl___redArg___closed__0));
lean_inc_ref(v_inst_183_);
v___x_187_ = lp_mathlib_AddSubmonoidClass_toAddMonoid___redArg(v_inst_183_);
lean_inc(v_inst_185_);
v___f_188_ = lean_alloc_closure((void*)(lp_mathlib_Submodule_smul___redArg___lam__0), 3, 1);
lean_closure_set(v___f_188_, 0, v_inst_185_);
v___f_189_ = ((lean_object*)(lp_mathlib_TensorProduct_mapIncl___redArg___closed__1));
lean_inc_ref(v_inst_181_);
v___x_190_ = lp_mathlib_TensorProduct_map___redArg(v_inst_181_, v_inst_181_, v___f_186_, v___x_187_, v_inst_182_, v_inst_183_, v___f_188_, v_inst_184_, v_inst_185_, v___f_189_, v___f_189_);
return v___x_190_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_TensorProduct_mapIncl(lean_object* v_R_191_, lean_object* v_inst_192_, lean_object* v_P_193_, lean_object* v_Q_194_, lean_object* v_inst_195_, lean_object* v_inst_196_, lean_object* v_inst_197_, lean_object* v_inst_198_, lean_object* v_p_199_, lean_object* v_q_200_){
_start:
{
lean_object* v___f_201_; lean_object* v___x_202_; lean_object* v___f_203_; lean_object* v___f_204_; lean_object* v___x_205_; 
v___f_201_ = ((lean_object*)(lp_mathlib_TensorProduct_mapIncl___redArg___closed__0));
lean_inc_ref(v_inst_196_);
v___x_202_ = lp_mathlib_AddSubmonoidClass_toAddMonoid___redArg(v_inst_196_);
lean_inc(v_inst_198_);
v___f_203_ = lean_alloc_closure((void*)(lp_mathlib_Submodule_smul___redArg___lam__0), 3, 1);
lean_closure_set(v___f_203_, 0, v_inst_198_);
v___f_204_ = ((lean_object*)(lp_mathlib_TensorProduct_mapIncl___redArg___closed__1));
lean_inc_ref(v_inst_192_);
v___x_205_ = lp_mathlib_TensorProduct_map___redArg(v_inst_192_, v_inst_192_, v___f_201_, v___x_202_, v_inst_195_, v_inst_196_, v___f_203_, v_inst_197_, v_inst_198_, v___f_204_, v___f_204_);
return v___x_205_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_TensorProduct_mapBilinear___redArg(lean_object* v_inst_206_, lean_object* v_inst_207_, lean_object* v_00_u03c3_u2081_u2082_208_, lean_object* v_inst_209_, lean_object* v_inst_210_, lean_object* v_inst_211_, lean_object* v_inst_212_, lean_object* v_inst_213_, lean_object* v_inst_214_, lean_object* v_inst_215_, lean_object* v_inst_216_){
_start:
{
lean_object* v___x_217_; lean_object* v___f_218_; 
v___x_217_ = lean_alloc_closure((void*)(lp_mathlib_TensorProduct_map___boxed), 19, 17);
lean_closure_set(v___x_217_, 0, lean_box(0));
lean_closure_set(v___x_217_, 1, lean_box(0));
lean_closure_set(v___x_217_, 2, v_inst_206_);
lean_closure_set(v___x_217_, 3, v_inst_207_);
lean_closure_set(v___x_217_, 4, v_00_u03c3_u2081_u2082_208_);
lean_closure_set(v___x_217_, 5, lean_box(0));
lean_closure_set(v___x_217_, 6, lean_box(0));
lean_closure_set(v___x_217_, 7, lean_box(0));
lean_closure_set(v___x_217_, 8, lean_box(0));
lean_closure_set(v___x_217_, 9, v_inst_209_);
lean_closure_set(v___x_217_, 10, v_inst_210_);
lean_closure_set(v___x_217_, 11, v_inst_211_);
lean_closure_set(v___x_217_, 12, v_inst_212_);
lean_closure_set(v___x_217_, 13, v_inst_213_);
lean_closure_set(v___x_217_, 14, v_inst_214_);
lean_closure_set(v___x_217_, 15, v_inst_215_);
lean_closure_set(v___x_217_, 16, v_inst_216_);
v___f_218_ = lean_alloc_closure((void*)(lp_mathlib_LinearMap_mk_u2082_x27_u209b_u2097___redArg___lam__0), 3, 1);
lean_closure_set(v___f_218_, 0, v___x_217_);
return v___f_218_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_TensorProduct_mapBilinear(lean_object* v_R_219_, lean_object* v_R_u2082_220_, lean_object* v_inst_221_, lean_object* v_inst_222_, lean_object* v_00_u03c3_u2081_u2082_223_, lean_object* v_M_224_, lean_object* v_N_225_, lean_object* v_M_u2082_226_, lean_object* v_N_u2082_227_, lean_object* v_inst_228_, lean_object* v_inst_229_, lean_object* v_inst_230_, lean_object* v_inst_231_, lean_object* v_inst_232_, lean_object* v_inst_233_, lean_object* v_inst_234_, lean_object* v_inst_235_){
_start:
{
lean_object* v___x_236_; 
v___x_236_ = lp_mathlib_TensorProduct_mapBilinear___redArg(v_inst_221_, v_inst_222_, v_00_u03c3_u2081_u2082_223_, v_inst_228_, v_inst_229_, v_inst_230_, v_inst_231_, v_inst_232_, v_inst_233_, v_inst_234_, v_inst_235_);
return v___x_236_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_TensorProduct_mapBilinear___boxed(lean_object** _args){
lean_object* v_R_237_ = _args[0];
lean_object* v_R_u2082_238_ = _args[1];
lean_object* v_inst_239_ = _args[2];
lean_object* v_inst_240_ = _args[3];
lean_object* v_00_u03c3_u2081_u2082_241_ = _args[4];
lean_object* v_M_242_ = _args[5];
lean_object* v_N_243_ = _args[6];
lean_object* v_M_u2082_244_ = _args[7];
lean_object* v_N_u2082_245_ = _args[8];
lean_object* v_inst_246_ = _args[9];
lean_object* v_inst_247_ = _args[10];
lean_object* v_inst_248_ = _args[11];
lean_object* v_inst_249_ = _args[12];
lean_object* v_inst_250_ = _args[13];
lean_object* v_inst_251_ = _args[14];
lean_object* v_inst_252_ = _args[15];
lean_object* v_inst_253_ = _args[16];
_start:
{
lean_object* v_res_254_; 
v_res_254_ = lp_mathlib_TensorProduct_mapBilinear(v_R_237_, v_R_u2082_238_, v_inst_239_, v_inst_240_, v_00_u03c3_u2081_u2082_241_, v_M_242_, v_N_243_, v_M_u2082_244_, v_N_u2082_245_, v_inst_246_, v_inst_247_, v_inst_248_, v_inst_249_, v_inst_250_, v_inst_251_, v_inst_252_, v_inst_253_);
return v_res_254_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_TensorProduct_lTensorHomToHomLTensor___redArg(lean_object* v_inst_255_, lean_object* v_inst_256_, lean_object* v_00_u03c3_u2081_u2082_257_, lean_object* v_inst_258_, lean_object* v_inst_259_, lean_object* v_inst_260_, lean_object* v_inst_261_, lean_object* v_inst_262_, lean_object* v_inst_263_){
_start:
{
lean_object* v___f_264_; lean_object* v___x_265_; lean_object* v___x_266_; lean_object* v___f_267_; lean_object* v___x_268_; lean_object* v___f_269_; lean_object* v___f_270_; lean_object* v___x_271_; lean_object* v___x_272_; lean_object* v___f_273_; lean_object* v___x_274_; 
v___f_264_ = ((lean_object*)(lp_mathlib_TensorProduct_mapIncl___redArg___closed__0));
lean_inc_ref_n(v_inst_260_, 4);
v___x_265_ = lp_mathlib_LinearMap_addMonoid___redArg(v_inst_260_);
lean_inc_n(v_inst_262_, 4);
lean_inc_n(v_inst_261_, 3);
lean_inc_ref_n(v_inst_259_, 2);
lean_inc_ref_n(v_inst_256_, 5);
v___x_266_ = lp_mathlib_TensorProduct_addMonoid___redArg(v_inst_256_, v_inst_259_, v_inst_260_, v_inst_261_, v_inst_262_);
v___f_267_ = lean_alloc_closure((void*)(lp_mathlib_TensorProduct_leftHasSMul___redArg___lam__0___boxed), 8, 6);
lean_closure_set(v___f_267_, 0, v_inst_256_);
lean_closure_set(v___f_267_, 1, v_inst_259_);
lean_closure_set(v___f_267_, 2, v_inst_260_);
lean_closure_set(v___f_267_, 3, v_inst_261_);
lean_closure_set(v___f_267_, 4, v_inst_262_);
lean_closure_set(v___f_267_, 5, v_inst_261_);
lean_inc_ref(v___x_266_);
v___x_268_ = lp_mathlib_LinearMap_addMonoid___redArg(v___x_266_);
v___f_269_ = lean_alloc_closure((void*)(lp_mathlib_LinearMap_instSMul___redArg___lam__0), 4, 1);
lean_closure_set(v___f_269_, 0, v_inst_262_);
lean_inc_ref(v___f_267_);
v___f_270_ = lean_alloc_closure((void*)(lp_mathlib_LinearMap_instSMul___redArg___lam__0), 4, 1);
lean_closure_set(v___f_270_, 0, v___f_267_);
lean_inc(v_00_u03c3_u2081_u2082_257_);
v___x_271_ = lp_mathlib_LinearMap_llcomp___redArg(v_inst_255_, v_inst_256_, v_inst_256_, v_inst_258_, v_inst_260_, v___x_266_, v_inst_263_, v_inst_262_, v___f_267_, v_00_u03c3_u2081_u2082_257_, v_00_u03c3_u2081_u2082_257_, v___f_264_);
v___x_272_ = lp_mathlib_TensorProduct_mk(lean_box(0), v_inst_256_, lean_box(0), lean_box(0), v_inst_259_, v_inst_260_, v_inst_261_, v_inst_262_);
lean_dec(v_inst_262_);
lean_dec(v_inst_261_);
lean_dec_ref(v_inst_260_);
lean_dec_ref(v_inst_259_);
v___f_273_ = lean_alloc_closure((void*)(lp_mathlib_LinearMap_comp___redArg___lam__0), 3, 2);
lean_closure_set(v___f_273_, 0, v___x_272_);
lean_closure_set(v___f_273_, 1, v___x_271_);
v___x_274_ = lp_mathlib_TensorProduct_liftAux___redArg(v_inst_256_, v_inst_256_, v___f_264_, v___x_265_, v___x_268_, v___f_269_, v___f_270_, v___f_273_);
return v___x_274_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_TensorProduct_lTensorHomToHomLTensor(lean_object* v_R_275_, lean_object* v_R_u2082_276_, lean_object* v_inst_277_, lean_object* v_inst_278_, lean_object* v_00_u03c3_u2081_u2082_279_, lean_object* v_P_280_, lean_object* v_M_u2082_281_, lean_object* v_N_u2082_282_, lean_object* v_inst_283_, lean_object* v_inst_284_, lean_object* v_inst_285_, lean_object* v_inst_286_, lean_object* v_inst_287_, lean_object* v_inst_288_){
_start:
{
lean_object* v___x_289_; 
v___x_289_ = lp_mathlib_TensorProduct_lTensorHomToHomLTensor___redArg(v_inst_277_, v_inst_278_, v_00_u03c3_u2081_u2082_279_, v_inst_283_, v_inst_284_, v_inst_285_, v_inst_286_, v_inst_287_, v_inst_288_);
return v___x_289_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_TensorProduct_rTensorHomToHomRTensor___redArg(lean_object* v_inst_290_, lean_object* v_inst_291_, lean_object* v_00_u03c3_u2081_u2082_292_, lean_object* v_inst_293_, lean_object* v_inst_294_, lean_object* v_inst_295_, lean_object* v_inst_296_, lean_object* v_inst_297_, lean_object* v_inst_298_){
_start:
{
lean_object* v___f_299_; lean_object* v___x_300_; lean_object* v___f_301_; lean_object* v___x_302_; lean_object* v___f_303_; lean_object* v___x_304_; lean_object* v___x_305_; lean_object* v___x_306_; lean_object* v___f_307_; lean_object* v___x_308_; lean_object* v___x_309_; 
v___f_299_ = ((lean_object*)(lp_mathlib_TensorProduct_mapIncl___redArg___closed__0));
lean_inc_n(v_inst_297_, 2);
lean_inc_n(v_inst_296_, 4);
lean_inc_ref_n(v_inst_295_, 2);
lean_inc_ref_n(v_inst_294_, 3);
lean_inc_ref_n(v_inst_291_, 5);
v___x_300_ = lp_mathlib_TensorProduct_addMonoid___redArg(v_inst_291_, v_inst_294_, v_inst_295_, v_inst_296_, v_inst_297_);
v___f_301_ = lean_alloc_closure((void*)(lp_mathlib_TensorProduct_leftHasSMul___redArg___lam__0___boxed), 8, 6);
lean_closure_set(v___f_301_, 0, v_inst_291_);
lean_closure_set(v___f_301_, 1, v_inst_294_);
lean_closure_set(v___f_301_, 2, v_inst_295_);
lean_closure_set(v___f_301_, 3, v_inst_296_);
lean_closure_set(v___f_301_, 4, v_inst_297_);
lean_closure_set(v___f_301_, 5, v_inst_296_);
lean_inc_ref(v___x_300_);
v___x_302_ = lp_mathlib_LinearMap_addMonoid___redArg(v___x_300_);
lean_inc_ref(v___f_301_);
v___f_303_ = lean_alloc_closure((void*)(lp_mathlib_LinearMap_instSMul___redArg___lam__0), 4, 1);
lean_closure_set(v___f_303_, 0, v___f_301_);
lean_inc(v_00_u03c3_u2081_u2082_292_);
v___x_304_ = lp_mathlib_LinearMap_llcomp___redArg(v_inst_290_, v_inst_291_, v_inst_291_, v_inst_293_, v_inst_294_, v___x_300_, v_inst_298_, v_inst_296_, v___f_301_, v_00_u03c3_u2081_u2082_292_, v_00_u03c3_u2081_u2082_292_, v___f_299_);
v___x_305_ = lp_mathlib_TensorProduct_mk(lean_box(0), v_inst_291_, lean_box(0), lean_box(0), v_inst_294_, v_inst_295_, v_inst_296_, v_inst_297_);
lean_dec(v_inst_296_);
lean_dec_ref(v_inst_294_);
v___x_306_ = lp_mathlib_LinearMap_flip___redArg(v___x_305_);
v___f_307_ = lean_alloc_closure((void*)(lp_mathlib_LinearMap_comp___redArg___lam__0), 3, 2);
lean_closure_set(v___f_307_, 0, v___x_306_);
lean_closure_set(v___f_307_, 1, v___x_304_);
v___x_308_ = lp_mathlib_LinearMap_flip___redArg(v___f_307_);
v___x_309_ = lp_mathlib_TensorProduct_liftAux___redArg(v_inst_291_, v_inst_291_, v___f_299_, v_inst_295_, v___x_302_, v_inst_297_, v___f_303_, v___x_308_);
return v___x_309_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_TensorProduct_rTensorHomToHomRTensor(lean_object* v_R_310_, lean_object* v_R_u2082_311_, lean_object* v_inst_312_, lean_object* v_inst_313_, lean_object* v_00_u03c3_u2081_u2082_314_, lean_object* v_P_315_, lean_object* v_M_u2082_316_, lean_object* v_N_u2082_317_, lean_object* v_inst_318_, lean_object* v_inst_319_, lean_object* v_inst_320_, lean_object* v_inst_321_, lean_object* v_inst_322_, lean_object* v_inst_323_){
_start:
{
lean_object* v___x_324_; 
v___x_324_ = lp_mathlib_TensorProduct_rTensorHomToHomRTensor___redArg(v_inst_312_, v_inst_313_, v_00_u03c3_u2081_u2082_314_, v_inst_318_, v_inst_319_, v_inst_320_, v_inst_321_, v_inst_322_, v_inst_323_);
return v___x_324_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_TensorProduct_homTensorHomMap___redArg(lean_object* v_inst_325_, lean_object* v_inst_326_, lean_object* v_00_u03c3_u2081_u2082_327_, lean_object* v_inst_328_, lean_object* v_inst_329_, lean_object* v_inst_330_, lean_object* v_inst_331_, lean_object* v_inst_332_, lean_object* v_inst_333_, lean_object* v_inst_334_, lean_object* v_inst_335_){
_start:
{
lean_object* v___f_336_; lean_object* v___x_337_; lean_object* v___x_338_; lean_object* v___f_339_; lean_object* v___x_340_; lean_object* v___f_341_; lean_object* v___f_342_; lean_object* v___x_343_; lean_object* v___x_344_; 
v___f_336_ = ((lean_object*)(lp_mathlib_TensorProduct_mapIncl___redArg___closed__0));
lean_inc_ref_n(v_inst_331_, 3);
v___x_337_ = lp_mathlib_LinearMap_addMonoid___redArg(v_inst_331_);
lean_inc_n(v_inst_335_, 3);
lean_inc_n(v_inst_334_, 3);
lean_inc_ref_n(v_inst_330_, 2);
lean_inc_ref_n(v_inst_326_, 4);
v___x_338_ = lp_mathlib_TensorProduct_addMonoid___redArg(v_inst_326_, v_inst_330_, v_inst_331_, v_inst_334_, v_inst_335_);
v___f_339_ = lean_alloc_closure((void*)(lp_mathlib_TensorProduct_leftHasSMul___redArg___lam__0___boxed), 8, 6);
lean_closure_set(v___f_339_, 0, v_inst_326_);
lean_closure_set(v___f_339_, 1, v_inst_330_);
lean_closure_set(v___f_339_, 2, v_inst_331_);
lean_closure_set(v___f_339_, 3, v_inst_334_);
lean_closure_set(v___f_339_, 4, v_inst_335_);
lean_closure_set(v___f_339_, 5, v_inst_334_);
v___x_340_ = lp_mathlib_LinearMap_addMonoid___redArg(v___x_338_);
v___f_341_ = lean_alloc_closure((void*)(lp_mathlib_LinearMap_instSMul___redArg___lam__0), 4, 1);
lean_closure_set(v___f_341_, 0, v_inst_335_);
v___f_342_ = lean_alloc_closure((void*)(lp_mathlib_LinearMap_instSMul___redArg___lam__0), 4, 1);
lean_closure_set(v___f_342_, 0, v___f_339_);
v___x_343_ = lp_mathlib_TensorProduct_mapBilinear___redArg(v_inst_325_, v_inst_326_, v_00_u03c3_u2081_u2082_327_, v_inst_328_, v_inst_329_, v_inst_330_, v_inst_331_, v_inst_332_, v_inst_333_, v_inst_334_, v_inst_335_);
v___x_344_ = lp_mathlib_TensorProduct_liftAux___redArg(v_inst_326_, v_inst_326_, v___f_336_, v___x_337_, v___x_340_, v___f_341_, v___f_342_, v___x_343_);
return v___x_344_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_TensorProduct_homTensorHomMap(lean_object* v_R_345_, lean_object* v_R_u2082_346_, lean_object* v_inst_347_, lean_object* v_inst_348_, lean_object* v_00_u03c3_u2081_u2082_349_, lean_object* v_M_350_, lean_object* v_N_351_, lean_object* v_M_u2082_352_, lean_object* v_N_u2082_353_, lean_object* v_inst_354_, lean_object* v_inst_355_, lean_object* v_inst_356_, lean_object* v_inst_357_, lean_object* v_inst_358_, lean_object* v_inst_359_, lean_object* v_inst_360_, lean_object* v_inst_361_){
_start:
{
lean_object* v___x_362_; 
v___x_362_ = lp_mathlib_TensorProduct_homTensorHomMap___redArg(v_inst_347_, v_inst_348_, v_00_u03c3_u2081_u2082_349_, v_inst_354_, v_inst_355_, v_inst_356_, v_inst_357_, v_inst_358_, v_inst_359_, v_inst_360_, v_inst_361_);
return v___x_362_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_TensorProduct_homTensorHomMap___boxed(lean_object** _args){
lean_object* v_R_363_ = _args[0];
lean_object* v_R_u2082_364_ = _args[1];
lean_object* v_inst_365_ = _args[2];
lean_object* v_inst_366_ = _args[3];
lean_object* v_00_u03c3_u2081_u2082_367_ = _args[4];
lean_object* v_M_368_ = _args[5];
lean_object* v_N_369_ = _args[6];
lean_object* v_M_u2082_370_ = _args[7];
lean_object* v_N_u2082_371_ = _args[8];
lean_object* v_inst_372_ = _args[9];
lean_object* v_inst_373_ = _args[10];
lean_object* v_inst_374_ = _args[11];
lean_object* v_inst_375_ = _args[12];
lean_object* v_inst_376_ = _args[13];
lean_object* v_inst_377_ = _args[14];
lean_object* v_inst_378_ = _args[15];
lean_object* v_inst_379_ = _args[16];
_start:
{
lean_object* v_res_380_; 
v_res_380_ = lp_mathlib_TensorProduct_homTensorHomMap(v_R_363_, v_R_u2082_364_, v_inst_365_, v_inst_366_, v_00_u03c3_u2081_u2082_367_, v_M_368_, v_N_369_, v_M_u2082_370_, v_N_u2082_371_, v_inst_372_, v_inst_373_, v_inst_374_, v_inst_375_, v_inst_376_, v_inst_377_, v_inst_378_, v_inst_379_);
return v_res_380_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_TensorProduct_map_u2082___redArg(lean_object* v_inst_381_, lean_object* v_inst_382_, lean_object* v_inst_383_, lean_object* v_00_u03c3_u2082_u2083_384_, lean_object* v_00_u03c3_u2081_u2083_385_, lean_object* v_inst_386_, lean_object* v_inst_387_, lean_object* v_inst_388_, lean_object* v_inst_389_, lean_object* v_inst_390_, lean_object* v_inst_391_, lean_object* v_inst_392_, lean_object* v_inst_393_, lean_object* v_inst_394_, lean_object* v_inst_395_, lean_object* v_f_396_, lean_object* v_g_397_){
_start:
{
lean_object* v___x_398_; lean_object* v___x_399_; lean_object* v___f_400_; lean_object* v___f_401_; lean_object* v___x_402_; lean_object* v___x_403_; lean_object* v___f_404_; 
lean_inc_ref(v_inst_389_);
v___x_398_ = lp_mathlib_LinearMap_addMonoid___redArg(v_inst_389_);
lean_inc_ref(v_inst_390_);
v___x_399_ = lp_mathlib_LinearMap_addMonoid___redArg(v_inst_390_);
lean_inc(v_inst_394_);
v___f_400_ = lean_alloc_closure((void*)(lp_mathlib_LinearMap_instSMul___redArg___lam__0), 4, 1);
lean_closure_set(v___f_400_, 0, v_inst_394_);
lean_inc(v_inst_395_);
v___f_401_ = lean_alloc_closure((void*)(lp_mathlib_LinearMap_instSMul___redArg___lam__0), 4, 1);
lean_closure_set(v___f_401_, 0, v_inst_395_);
lean_inc_ref(v_inst_383_);
v___x_402_ = lp_mathlib_TensorProduct_homTensorHomMap___redArg(v_inst_382_, v_inst_383_, v_00_u03c3_u2082_u2083_384_, v_inst_387_, v_inst_388_, v_inst_389_, v_inst_390_, v_inst_392_, v_inst_393_, v_inst_394_, v_inst_395_);
v___x_403_ = lp_mathlib_TensorProduct_map___redArg(v_inst_381_, v_inst_383_, v_00_u03c3_u2081_u2083_385_, v_inst_386_, v___x_398_, v___x_399_, v_inst_391_, v___f_400_, v___f_401_, v_f_396_, v_g_397_);
v___f_404_ = lean_alloc_closure((void*)(lp_mathlib_LinearMap_comp___redArg___lam__0), 3, 2);
lean_closure_set(v___f_404_, 0, v___x_403_);
lean_closure_set(v___f_404_, 1, v___x_402_);
return v___f_404_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_TensorProduct_map_u2082___redArg___boxed(lean_object** _args){
lean_object* v_inst_405_ = _args[0];
lean_object* v_inst_406_ = _args[1];
lean_object* v_inst_407_ = _args[2];
lean_object* v_00_u03c3_u2082_u2083_408_ = _args[3];
lean_object* v_00_u03c3_u2081_u2083_409_ = _args[4];
lean_object* v_inst_410_ = _args[5];
lean_object* v_inst_411_ = _args[6];
lean_object* v_inst_412_ = _args[7];
lean_object* v_inst_413_ = _args[8];
lean_object* v_inst_414_ = _args[9];
lean_object* v_inst_415_ = _args[10];
lean_object* v_inst_416_ = _args[11];
lean_object* v_inst_417_ = _args[12];
lean_object* v_inst_418_ = _args[13];
lean_object* v_inst_419_ = _args[14];
lean_object* v_f_420_ = _args[15];
lean_object* v_g_421_ = _args[16];
_start:
{
lean_object* v_res_422_; 
v_res_422_ = lp_mathlib_TensorProduct_map_u2082___redArg(v_inst_405_, v_inst_406_, v_inst_407_, v_00_u03c3_u2082_u2083_408_, v_00_u03c3_u2081_u2083_409_, v_inst_410_, v_inst_411_, v_inst_412_, v_inst_413_, v_inst_414_, v_inst_415_, v_inst_416_, v_inst_417_, v_inst_418_, v_inst_419_, v_f_420_, v_g_421_);
return v_res_422_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_TensorProduct_map_u2082(lean_object* v_R_423_, lean_object* v_R_u2082_424_, lean_object* v_R_u2083_425_, lean_object* v_inst_426_, lean_object* v_inst_427_, lean_object* v_inst_428_, lean_object* v_00_u03c3_u2082_u2083_429_, lean_object* v_00_u03c3_u2081_u2083_430_, lean_object* v_M_431_, lean_object* v_N_432_, lean_object* v_M_u2082_433_, lean_object* v_M_u2083_434_, lean_object* v_N_u2082_435_, lean_object* v_N_u2083_436_, lean_object* v_inst_437_, lean_object* v_inst_438_, lean_object* v_inst_439_, lean_object* v_inst_440_, lean_object* v_inst_441_, lean_object* v_inst_442_, lean_object* v_inst_443_, lean_object* v_inst_444_, lean_object* v_inst_445_, lean_object* v_inst_446_, lean_object* v_inst_447_, lean_object* v_inst_448_, lean_object* v_f_449_, lean_object* v_g_450_){
_start:
{
lean_object* v___x_451_; 
v___x_451_ = lp_mathlib_TensorProduct_map_u2082___redArg(v_inst_426_, v_inst_427_, v_inst_428_, v_00_u03c3_u2082_u2083_429_, v_00_u03c3_u2081_u2083_430_, v_inst_438_, v_inst_439_, v_inst_440_, v_inst_441_, v_inst_442_, v_inst_444_, v_inst_445_, v_inst_446_, v_inst_447_, v_inst_448_, v_f_449_, v_g_450_);
return v___x_451_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_TensorProduct_map_u2082___boxed(lean_object** _args){
lean_object* v_R_452_ = _args[0];
lean_object* v_R_u2082_453_ = _args[1];
lean_object* v_R_u2083_454_ = _args[2];
lean_object* v_inst_455_ = _args[3];
lean_object* v_inst_456_ = _args[4];
lean_object* v_inst_457_ = _args[5];
lean_object* v_00_u03c3_u2082_u2083_458_ = _args[6];
lean_object* v_00_u03c3_u2081_u2083_459_ = _args[7];
lean_object* v_M_460_ = _args[8];
lean_object* v_N_461_ = _args[9];
lean_object* v_M_u2082_462_ = _args[10];
lean_object* v_M_u2083_463_ = _args[11];
lean_object* v_N_u2082_464_ = _args[12];
lean_object* v_N_u2083_465_ = _args[13];
lean_object* v_inst_466_ = _args[14];
lean_object* v_inst_467_ = _args[15];
lean_object* v_inst_468_ = _args[16];
lean_object* v_inst_469_ = _args[17];
lean_object* v_inst_470_ = _args[18];
lean_object* v_inst_471_ = _args[19];
lean_object* v_inst_472_ = _args[20];
lean_object* v_inst_473_ = _args[21];
lean_object* v_inst_474_ = _args[22];
lean_object* v_inst_475_ = _args[23];
lean_object* v_inst_476_ = _args[24];
lean_object* v_inst_477_ = _args[25];
lean_object* v_f_478_ = _args[26];
lean_object* v_g_479_ = _args[27];
_start:
{
lean_object* v_res_480_; 
v_res_480_ = lp_mathlib_TensorProduct_map_u2082(v_R_452_, v_R_u2082_453_, v_R_u2083_454_, v_inst_455_, v_inst_456_, v_inst_457_, v_00_u03c3_u2082_u2083_458_, v_00_u03c3_u2081_u2083_459_, v_M_460_, v_N_461_, v_M_u2082_462_, v_M_u2083_463_, v_N_u2082_464_, v_N_u2083_465_, v_inst_466_, v_inst_467_, v_inst_468_, v_inst_469_, v_inst_470_, v_inst_471_, v_inst_472_, v_inst_473_, v_inst_474_, v_inst_475_, v_inst_476_, v_inst_477_, v_f_478_, v_g_479_);
lean_dec(v_inst_472_);
lean_dec_ref(v_inst_466_);
return v_res_480_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_TensorProduct_congr___redArg(lean_object* v_inst_481_, lean_object* v_inst_482_, lean_object* v_00_u03c3_u2081_u2082_483_, lean_object* v_inst_484_, lean_object* v_inst_485_, lean_object* v_inst_486_, lean_object* v_inst_487_, lean_object* v_inst_488_, lean_object* v_inst_489_, lean_object* v_inst_490_, lean_object* v_inst_491_, lean_object* v_00_u03c3_u2082_u2081_492_, lean_object* v_f_493_, lean_object* v_g_494_){
_start:
{
lean_object* v_toLinearMap_495_; lean_object* v_toLinearMap_496_; lean_object* v___x_497_; lean_object* v_toLinearMap_498_; lean_object* v___x_499_; lean_object* v_toLinearMap_500_; lean_object* v___x_501_; lean_object* v___x_502_; lean_object* v___x_503_; 
v_toLinearMap_495_ = lean_ctor_get(v_f_493_, 0);
lean_inc(v_toLinearMap_495_);
v_toLinearMap_496_ = lean_ctor_get(v_g_494_, 0);
lean_inc(v_toLinearMap_496_);
v___x_497_ = lp_mathlib_LinearEquiv_symm___redArg(v_f_493_);
v_toLinearMap_498_ = lean_ctor_get(v___x_497_, 0);
lean_inc(v_toLinearMap_498_);
lean_dec_ref(v___x_497_);
v___x_499_ = lp_mathlib_LinearEquiv_symm___redArg(v_g_494_);
v_toLinearMap_500_ = lean_ctor_get(v___x_499_, 0);
lean_inc(v_toLinearMap_500_);
lean_dec_ref(v___x_499_);
lean_inc(v_inst_491_);
lean_inc(v_inst_489_);
lean_inc_ref(v_inst_487_);
lean_inc_ref(v_inst_485_);
lean_inc_ref(v_inst_482_);
lean_inc_ref(v_inst_481_);
v___x_501_ = lp_mathlib_TensorProduct_map___redArg(v_inst_481_, v_inst_482_, v_00_u03c3_u2081_u2082_483_, v_inst_485_, v_inst_486_, v_inst_487_, v_inst_489_, v_inst_490_, v_inst_491_, v_toLinearMap_495_, v_toLinearMap_496_);
v___x_502_ = lp_mathlib_TensorProduct_map___redArg(v_inst_482_, v_inst_481_, v_00_u03c3_u2082_u2081_492_, v_inst_487_, v_inst_484_, v_inst_485_, v_inst_491_, v_inst_488_, v_inst_489_, v_toLinearMap_498_, v_toLinearMap_500_);
v___x_503_ = lp_mathlib_LinearEquiv_ofLinearMap___redArg(v___x_501_, v___x_502_);
return v___x_503_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_TensorProduct_congr(lean_object* v_R_504_, lean_object* v_R_u2082_505_, lean_object* v_inst_506_, lean_object* v_inst_507_, lean_object* v_00_u03c3_u2081_u2082_508_, lean_object* v_M_509_, lean_object* v_N_510_, lean_object* v_M_u2082_511_, lean_object* v_N_u2082_512_, lean_object* v_inst_513_, lean_object* v_inst_514_, lean_object* v_inst_515_, lean_object* v_inst_516_, lean_object* v_inst_517_, lean_object* v_inst_518_, lean_object* v_inst_519_, lean_object* v_inst_520_, lean_object* v_00_u03c3_u2082_u2081_521_, lean_object* v_inst_522_, lean_object* v_inst_523_, lean_object* v_f_524_, lean_object* v_g_525_){
_start:
{
lean_object* v___x_526_; 
v___x_526_ = lp_mathlib_TensorProduct_congr___redArg(v_inst_506_, v_inst_507_, v_00_u03c3_u2081_u2082_508_, v_inst_513_, v_inst_514_, v_inst_515_, v_inst_516_, v_inst_517_, v_inst_518_, v_inst_519_, v_inst_520_, v_00_u03c3_u2082_u2081_521_, v_f_524_, v_g_525_);
return v___x_526_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_TensorProduct_congr___boxed(lean_object** _args){
lean_object* v_R_527_ = _args[0];
lean_object* v_R_u2082_528_ = _args[1];
lean_object* v_inst_529_ = _args[2];
lean_object* v_inst_530_ = _args[3];
lean_object* v_00_u03c3_u2081_u2082_531_ = _args[4];
lean_object* v_M_532_ = _args[5];
lean_object* v_N_533_ = _args[6];
lean_object* v_M_u2082_534_ = _args[7];
lean_object* v_N_u2082_535_ = _args[8];
lean_object* v_inst_536_ = _args[9];
lean_object* v_inst_537_ = _args[10];
lean_object* v_inst_538_ = _args[11];
lean_object* v_inst_539_ = _args[12];
lean_object* v_inst_540_ = _args[13];
lean_object* v_inst_541_ = _args[14];
lean_object* v_inst_542_ = _args[15];
lean_object* v_inst_543_ = _args[16];
lean_object* v_00_u03c3_u2082_u2081_544_ = _args[17];
lean_object* v_inst_545_ = _args[18];
lean_object* v_inst_546_ = _args[19];
lean_object* v_f_547_ = _args[20];
lean_object* v_g_548_ = _args[21];
_start:
{
lean_object* v_res_549_; 
v_res_549_ = lp_mathlib_TensorProduct_congr(v_R_527_, v_R_u2082_528_, v_inst_529_, v_inst_530_, v_00_u03c3_u2081_u2082_531_, v_M_532_, v_N_533_, v_M_u2082_534_, v_N_u2082_535_, v_inst_536_, v_inst_537_, v_inst_538_, v_inst_539_, v_inst_540_, v_inst_541_, v_inst_542_, v_inst_543_, v_00_u03c3_u2082_u2081_544_, v_inst_545_, v_inst_546_, v_f_547_, v_g_548_);
return v_res_549_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LinearMap_lTensor___redArg(lean_object* v_inst_551_, lean_object* v_inst_552_, lean_object* v_inst_553_, lean_object* v_inst_554_, lean_object* v_inst_555_, lean_object* v_inst_556_, lean_object* v_inst_557_, lean_object* v_f_558_){
_start:
{
lean_object* v___f_559_; lean_object* v___f_560_; lean_object* v___x_561_; 
v___f_559_ = ((lean_object*)(lp_mathlib_TensorProduct_mapIncl___redArg___closed__0));
v___f_560_ = ((lean_object*)(lp_mathlib_LinearMap_lTensor___redArg___closed__0));
lean_inc_ref(v_inst_551_);
v___x_561_ = lp_mathlib_TensorProduct_map___redArg(v_inst_551_, v_inst_551_, v___f_559_, v_inst_553_, v_inst_552_, v_inst_554_, v_inst_556_, v_inst_555_, v_inst_557_, v___f_560_, v_f_558_);
return v___x_561_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LinearMap_lTensor(lean_object* v_R_562_, lean_object* v_inst_563_, lean_object* v_M_564_, lean_object* v_N_565_, lean_object* v_P_566_, lean_object* v_inst_567_, lean_object* v_inst_568_, lean_object* v_inst_569_, lean_object* v_inst_570_, lean_object* v_inst_571_, lean_object* v_inst_572_, lean_object* v_f_573_){
_start:
{
lean_object* v___x_574_; 
v___x_574_ = lp_mathlib_LinearMap_lTensor___redArg(v_inst_563_, v_inst_567_, v_inst_568_, v_inst_569_, v_inst_570_, v_inst_571_, v_inst_572_, v_f_573_);
return v___x_574_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LinearMap_rTensor___redArg(lean_object* v_inst_575_, lean_object* v_inst_576_, lean_object* v_inst_577_, lean_object* v_inst_578_, lean_object* v_inst_579_, lean_object* v_f_580_){
_start:
{
lean_object* v___f_581_; lean_object* v___f_582_; lean_object* v___x_583_; 
v___f_581_ = ((lean_object*)(lp_mathlib_TensorProduct_mapIncl___redArg___closed__0));
v___f_582_ = ((lean_object*)(lp_mathlib_LinearMap_lTensor___redArg___closed__0));
lean_inc(v_inst_578_);
lean_inc_ref(v_inst_576_);
lean_inc_ref(v_inst_575_);
v___x_583_ = lp_mathlib_TensorProduct_map___redArg(v_inst_575_, v_inst_575_, v___f_581_, v_inst_576_, v_inst_577_, v_inst_576_, v_inst_578_, v_inst_579_, v_inst_578_, v_f_580_, v___f_582_);
return v___x_583_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LinearMap_rTensor(lean_object* v_R_584_, lean_object* v_inst_585_, lean_object* v_M_586_, lean_object* v_N_587_, lean_object* v_P_588_, lean_object* v_inst_589_, lean_object* v_inst_590_, lean_object* v_inst_591_, lean_object* v_inst_592_, lean_object* v_inst_593_, lean_object* v_inst_594_, lean_object* v_f_595_){
_start:
{
lean_object* v___x_596_; 
v___x_596_ = lp_mathlib_LinearMap_rTensor___redArg(v_inst_585_, v_inst_589_, v_inst_591_, v_inst_592_, v_inst_594_, v_f_595_);
return v___x_596_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LinearMap_rTensor___boxed(lean_object* v_R_597_, lean_object* v_inst_598_, lean_object* v_M_599_, lean_object* v_N_600_, lean_object* v_P_601_, lean_object* v_inst_602_, lean_object* v_inst_603_, lean_object* v_inst_604_, lean_object* v_inst_605_, lean_object* v_inst_606_, lean_object* v_inst_607_, lean_object* v_f_608_){
_start:
{
lean_object* v_res_609_; 
v_res_609_ = lp_mathlib_LinearMap_rTensor(v_R_597_, v_inst_598_, v_M_599_, v_N_600_, v_P_601_, v_inst_602_, v_inst_603_, v_inst_604_, v_inst_605_, v_inst_606_, v_inst_607_, v_f_608_);
lean_dec(v_inst_606_);
lean_dec_ref(v_inst_603_);
return v_res_609_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LinearMap_lTensorHom___redArg(lean_object* v_inst_610_, lean_object* v_inst_611_, lean_object* v_inst_612_, lean_object* v_inst_613_, lean_object* v_inst_614_, lean_object* v_inst_615_, lean_object* v_inst_616_){
_start:
{
lean_object* v___x_617_; 
v___x_617_ = lean_alloc_closure((void*)(lp_mathlib_LinearMap_lTensor), 12, 11);
lean_closure_set(v___x_617_, 0, lean_box(0));
lean_closure_set(v___x_617_, 1, v_inst_610_);
lean_closure_set(v___x_617_, 2, lean_box(0));
lean_closure_set(v___x_617_, 3, lean_box(0));
lean_closure_set(v___x_617_, 4, lean_box(0));
lean_closure_set(v___x_617_, 5, v_inst_611_);
lean_closure_set(v___x_617_, 6, v_inst_612_);
lean_closure_set(v___x_617_, 7, v_inst_613_);
lean_closure_set(v___x_617_, 8, v_inst_614_);
lean_closure_set(v___x_617_, 9, v_inst_615_);
lean_closure_set(v___x_617_, 10, v_inst_616_);
return v___x_617_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LinearMap_lTensorHom(lean_object* v_R_618_, lean_object* v_inst_619_, lean_object* v_M_620_, lean_object* v_N_621_, lean_object* v_P_622_, lean_object* v_inst_623_, lean_object* v_inst_624_, lean_object* v_inst_625_, lean_object* v_inst_626_, lean_object* v_inst_627_, lean_object* v_inst_628_){
_start:
{
lean_object* v___x_629_; 
v___x_629_ = lean_alloc_closure((void*)(lp_mathlib_LinearMap_lTensor), 12, 11);
lean_closure_set(v___x_629_, 0, lean_box(0));
lean_closure_set(v___x_629_, 1, v_inst_619_);
lean_closure_set(v___x_629_, 2, lean_box(0));
lean_closure_set(v___x_629_, 3, lean_box(0));
lean_closure_set(v___x_629_, 4, lean_box(0));
lean_closure_set(v___x_629_, 5, v_inst_623_);
lean_closure_set(v___x_629_, 6, v_inst_624_);
lean_closure_set(v___x_629_, 7, v_inst_625_);
lean_closure_set(v___x_629_, 8, v_inst_626_);
lean_closure_set(v___x_629_, 9, v_inst_627_);
lean_closure_set(v___x_629_, 10, v_inst_628_);
return v___x_629_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LinearMap_rTensorHom___redArg___lam__0(lean_object* v_inst_630_, lean_object* v_inst_631_, lean_object* v_inst_632_, lean_object* v_inst_633_, lean_object* v_inst_634_, lean_object* v_f_635_, lean_object* v___y_636_){
_start:
{
lean_object* v___x_55__overap_637_; lean_object* v___x_638_; 
v___x_55__overap_637_ = lp_mathlib_LinearMap_rTensor___redArg(v_inst_630_, v_inst_631_, v_inst_632_, v_inst_633_, v_inst_634_, v_f_635_);
v___x_638_ = lean_apply_1(v___x_55__overap_637_, v___y_636_);
return v___x_638_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LinearMap_rTensorHom___redArg(lean_object* v_inst_639_, lean_object* v_inst_640_, lean_object* v_inst_641_, lean_object* v_inst_642_, lean_object* v_inst_643_){
_start:
{
lean_object* v___f_644_; 
v___f_644_ = lean_alloc_closure((void*)(lp_mathlib_LinearMap_rTensorHom___redArg___lam__0), 7, 5);
lean_closure_set(v___f_644_, 0, v_inst_639_);
lean_closure_set(v___f_644_, 1, v_inst_640_);
lean_closure_set(v___f_644_, 2, v_inst_641_);
lean_closure_set(v___f_644_, 3, v_inst_642_);
lean_closure_set(v___f_644_, 4, v_inst_643_);
return v___f_644_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LinearMap_rTensorHom(lean_object* v_R_645_, lean_object* v_inst_646_, lean_object* v_M_647_, lean_object* v_N_648_, lean_object* v_P_649_, lean_object* v_inst_650_, lean_object* v_inst_651_, lean_object* v_inst_652_, lean_object* v_inst_653_, lean_object* v_inst_654_, lean_object* v_inst_655_){
_start:
{
lean_object* v___f_656_; 
v___f_656_ = lean_alloc_closure((void*)(lp_mathlib_LinearMap_rTensorHom___redArg___lam__0), 7, 5);
lean_closure_set(v___f_656_, 0, v_inst_646_);
lean_closure_set(v___f_656_, 1, v_inst_650_);
lean_closure_set(v___f_656_, 2, v_inst_652_);
lean_closure_set(v___f_656_, 3, v_inst_653_);
lean_closure_set(v___f_656_, 4, v_inst_655_);
return v___f_656_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LinearMap_rTensorHom___boxed(lean_object* v_R_657_, lean_object* v_inst_658_, lean_object* v_M_659_, lean_object* v_N_660_, lean_object* v_P_661_, lean_object* v_inst_662_, lean_object* v_inst_663_, lean_object* v_inst_664_, lean_object* v_inst_665_, lean_object* v_inst_666_, lean_object* v_inst_667_){
_start:
{
lean_object* v_res_668_; 
v_res_668_ = lp_mathlib_LinearMap_rTensorHom(v_R_657_, v_inst_658_, v_M_659_, v_N_660_, v_P_661_, v_inst_662_, v_inst_663_, v_inst_664_, v_inst_665_, v_inst_666_, v_inst_667_);
lean_dec(v_inst_666_);
lean_dec_ref(v_inst_663_);
return v_res_668_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LinearEquiv_lTensor___redArg(lean_object* v_inst_669_, lean_object* v_inst_670_, lean_object* v_inst_671_, lean_object* v_inst_672_, lean_object* v_inst_673_, lean_object* v_inst_674_, lean_object* v_inst_675_, lean_object* v_f_676_){
_start:
{
lean_object* v___f_677_; lean_object* v___x_678_; lean_object* v___x_679_; 
v___f_677_ = ((lean_object*)(lp_mathlib_TensorProduct_mapIncl___redArg___closed__0));
v___x_678_ = lp_mathlib_LinearEquiv_refl(lean_box(0), lean_box(0), v_inst_669_, v_inst_670_, v_inst_673_);
lean_inc(v_inst_673_);
lean_inc_ref(v_inst_670_);
lean_inc_ref(v_inst_669_);
v___x_679_ = lp_mathlib_TensorProduct_congr___redArg(v_inst_669_, v_inst_669_, v___f_677_, v_inst_670_, v_inst_671_, v_inst_670_, v_inst_672_, v_inst_673_, v_inst_674_, v_inst_673_, v_inst_675_, v___f_677_, v___x_678_, v_f_676_);
return v___x_679_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LinearEquiv_lTensor(lean_object* v_R_680_, lean_object* v_inst_681_, lean_object* v_M_682_, lean_object* v_N_683_, lean_object* v_P_684_, lean_object* v_inst_685_, lean_object* v_inst_686_, lean_object* v_inst_687_, lean_object* v_inst_688_, lean_object* v_inst_689_, lean_object* v_inst_690_, lean_object* v_f_691_){
_start:
{
lean_object* v___x_692_; 
v___x_692_ = lp_mathlib_LinearEquiv_lTensor___redArg(v_inst_681_, v_inst_685_, v_inst_686_, v_inst_687_, v_inst_688_, v_inst_689_, v_inst_690_, v_f_691_);
return v___x_692_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LinearEquiv_rTensor___redArg(lean_object* v_inst_693_, lean_object* v_inst_694_, lean_object* v_inst_695_, lean_object* v_inst_696_, lean_object* v_inst_697_, lean_object* v_inst_698_, lean_object* v_inst_699_, lean_object* v_f_700_){
_start:
{
lean_object* v___f_701_; lean_object* v___x_702_; lean_object* v___x_703_; 
v___f_701_ = ((lean_object*)(lp_mathlib_TensorProduct_mapIncl___redArg___closed__0));
v___x_702_ = lp_mathlib_LinearEquiv_refl(lean_box(0), lean_box(0), v_inst_693_, v_inst_694_, v_inst_697_);
lean_inc(v_inst_697_);
lean_inc_ref(v_inst_694_);
lean_inc_ref(v_inst_693_);
v___x_703_ = lp_mathlib_TensorProduct_congr___redArg(v_inst_693_, v_inst_693_, v___f_701_, v_inst_695_, v_inst_694_, v_inst_696_, v_inst_694_, v_inst_698_, v_inst_697_, v_inst_699_, v_inst_697_, v___f_701_, v_f_700_, v___x_702_);
return v___x_703_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LinearEquiv_rTensor(lean_object* v_R_704_, lean_object* v_inst_705_, lean_object* v_M_706_, lean_object* v_N_707_, lean_object* v_P_708_, lean_object* v_inst_709_, lean_object* v_inst_710_, lean_object* v_inst_711_, lean_object* v_inst_712_, lean_object* v_inst_713_, lean_object* v_inst_714_, lean_object* v_f_715_){
_start:
{
lean_object* v___x_716_; 
v___x_716_ = lp_mathlib_LinearEquiv_rTensor___redArg(v_inst_705_, v_inst_709_, v_inst_710_, v_inst_711_, v_inst_712_, v_inst_713_, v_inst_714_, v_f_715_);
return v___x_716_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_LinearAlgebra_TensorProduct_Basic(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Module_Shrink(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_LinearAlgebra_TensorProduct_Map(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_LinearAlgebra_TensorProduct_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Module_Shrink(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_LinearAlgebra_TensorProduct_Map(uint8_t builtin) {
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
lean_object* initialize_mathlib_Mathlib_LinearAlgebra_TensorProduct_Basic(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Module_Shrink(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_LinearAlgebra_TensorProduct_Map(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_LinearAlgebra_TensorProduct_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Module_Shrink(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_LinearAlgebra_TensorProduct_Map(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_LinearAlgebra_TensorProduct_Map(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_LinearAlgebra_TensorProduct_Map(builtin);
}
#ifdef __cplusplus
}
#endif
