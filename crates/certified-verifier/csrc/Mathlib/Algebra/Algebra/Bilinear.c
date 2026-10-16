// Lean compiler output
// Module: Mathlib.Algebra.Algebra.Bilinear
// Imports: public import Init public meta import Init public import Mathlib.Algebra.Algebra.NonUnitalHom public import Mathlib.LinearAlgebra.TensorProduct.Map
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
uint8_t l_Lean_Syntax_isOfKind(lean_object*, lean_object*);
lean_object* l_Lean_Syntax_getArg(lean_object*, lean_object*);
uint8_t l_Lean_Syntax_matchesNull(lean_object*, lean_object*);
lean_object* l_Lean_replaceRef(lean_object*, lean_object*);
lean_object* l_Lean_SourceInfo_fromRef(lean_object*, uint8_t);
lean_object* l_Lean_Name_mkStr3(lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Syntax_node3(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_NonUnitalNonAssocSemiring_toDistrib___redArg(lean_object*);
lean_object* lp_mathlib_LinearMap_mk_u2082_x27_u209b_u2097___redArg___lam__0(lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Semiring_toNonUnitalSemiring___redArg(lean_object*);
lean_object* lp_mathlib_NonUnitalRingHom_id___lam__0___boxed(lean_object*);
lean_object* lp_mathlib_TensorProduct_liftAux___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_String_toRawSubstring_x27(lean_object*);
lean_object* l_Lean_Name_mkStr2(lean_object*, lean_object*);
lean_object* l_Lean_addMacroScope(lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Syntax_node1(lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Syntax_node2(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Semiring_toNonAssocSemiring___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LinearMap_mul___redArg___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LinearMap_mul___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LinearMap_mul(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LinearMap_mul___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_LinearMap_mul_x27___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_NonUnitalRingHom_id___lam__0___boxed, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_LinearMap_mul_x27___redArg___closed__0 = (const lean_object*)&lp_mathlib_LinearMap_mul_x27___redArg___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_LinearMap_mul_x27___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LinearMap_mul_x27(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_RingTheory_LinearMap_term_u03bc___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 11, .m_capacity = 11, .m_length = 10, .m_data = "RingTheory"};
static const lean_object* lp_mathlib_RingTheory_LinearMap_term_u03bc___closed__0 = (const lean_object*)&lp_mathlib_RingTheory_LinearMap_term_u03bc___closed__0_value;
static const lean_string_object lp_mathlib_RingTheory_LinearMap_term_u03bc___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 9, .m_data = "LinearMap"};
static const lean_object* lp_mathlib_RingTheory_LinearMap_term_u03bc___closed__1 = (const lean_object*)&lp_mathlib_RingTheory_LinearMap_term_u03bc___closed__1_value;
static const lean_string_object lp_mathlib_RingTheory_LinearMap_term_u03bc___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 5, .m_data = "termμ"};
static const lean_object* lp_mathlib_RingTheory_LinearMap_term_u03bc___closed__2 = (const lean_object*)&lp_mathlib_RingTheory_LinearMap_term_u03bc___closed__2_value;
static const lean_ctor_object lp_mathlib_RingTheory_LinearMap_term_u03bc___closed__3_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_RingTheory_LinearMap_term_u03bc___closed__0_value),LEAN_SCALAR_PTR_LITERAL(204, 50, 210, 176, 233, 167, 74, 91)}};
static const lean_ctor_object lp_mathlib_RingTheory_LinearMap_term_u03bc___closed__3_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_RingTheory_LinearMap_term_u03bc___closed__3_value_aux_0),((lean_object*)&lp_mathlib_RingTheory_LinearMap_term_u03bc___closed__1_value),LEAN_SCALAR_PTR_LITERAL(118, 100, 27, 238, 183, 36, 185, 13)}};
static const lean_ctor_object lp_mathlib_RingTheory_LinearMap_term_u03bc___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_RingTheory_LinearMap_term_u03bc___closed__3_value_aux_1),((lean_object*)&lp_mathlib_RingTheory_LinearMap_term_u03bc___closed__2_value),LEAN_SCALAR_PTR_LITERAL(191, 74, 26, 188, 74, 69, 18, 117)}};
static const lean_object* lp_mathlib_RingTheory_LinearMap_term_u03bc___closed__3 = (const lean_object*)&lp_mathlib_RingTheory_LinearMap_term_u03bc___closed__3_value;
static const lean_string_object lp_mathlib_RingTheory_LinearMap_term_u03bc___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 3, .m_capacity = 3, .m_length = 1, .m_data = "μ"};
static const lean_object* lp_mathlib_RingTheory_LinearMap_term_u03bc___closed__4 = (const lean_object*)&lp_mathlib_RingTheory_LinearMap_term_u03bc___closed__4_value;
static const lean_ctor_object lp_mathlib_RingTheory_LinearMap_term_u03bc___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 5}, .m_objs = {((lean_object*)&lp_mathlib_RingTheory_LinearMap_term_u03bc___closed__4_value)}};
static const lean_object* lp_mathlib_RingTheory_LinearMap_term_u03bc___closed__5 = (const lean_object*)&lp_mathlib_RingTheory_LinearMap_term_u03bc___closed__5_value;
static const lean_ctor_object lp_mathlib_RingTheory_LinearMap_term_u03bc___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 3}, .m_objs = {((lean_object*)&lp_mathlib_RingTheory_LinearMap_term_u03bc___closed__3_value),((lean_object*)(((size_t)(1024) << 1) | 1)),((lean_object*)&lp_mathlib_RingTheory_LinearMap_term_u03bc___closed__5_value)}};
static const lean_object* lp_mathlib_RingTheory_LinearMap_term_u03bc___closed__6 = (const lean_object*)&lp_mathlib_RingTheory_LinearMap_term_u03bc___closed__6_value;
LEAN_EXPORT const lean_object* lp_mathlib_RingTheory_LinearMap_term_u03bc = (const lean_object*)&lp_mathlib_RingTheory_LinearMap_term_u03bc___closed__6_value;
static const lean_string_object lp_mathlib_RingTheory_LinearMap___aux__Mathlib__Algebra__Algebra__Bilinear______macroRules__RingTheory__LinearMap__term_u03bc__1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "Lean"};
static const lean_object* lp_mathlib_RingTheory_LinearMap___aux__Mathlib__Algebra__Algebra__Bilinear______macroRules__RingTheory__LinearMap__term_u03bc__1___closed__0 = (const lean_object*)&lp_mathlib_RingTheory_LinearMap___aux__Mathlib__Algebra__Algebra__Bilinear______macroRules__RingTheory__LinearMap__term_u03bc__1___closed__0_value;
static const lean_string_object lp_mathlib_RingTheory_LinearMap___aux__Mathlib__Algebra__Algebra__Bilinear______macroRules__RingTheory__LinearMap__term_u03bc__1___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "Parser"};
static const lean_object* lp_mathlib_RingTheory_LinearMap___aux__Mathlib__Algebra__Algebra__Bilinear______macroRules__RingTheory__LinearMap__term_u03bc__1___closed__1 = (const lean_object*)&lp_mathlib_RingTheory_LinearMap___aux__Mathlib__Algebra__Algebra__Bilinear______macroRules__RingTheory__LinearMap__term_u03bc__1___closed__1_value;
static const lean_string_object lp_mathlib_RingTheory_LinearMap___aux__Mathlib__Algebra__Algebra__Bilinear______macroRules__RingTheory__LinearMap__term_u03bc__1___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "Term"};
static const lean_object* lp_mathlib_RingTheory_LinearMap___aux__Mathlib__Algebra__Algebra__Bilinear______macroRules__RingTheory__LinearMap__term_u03bc__1___closed__2 = (const lean_object*)&lp_mathlib_RingTheory_LinearMap___aux__Mathlib__Algebra__Algebra__Bilinear______macroRules__RingTheory__LinearMap__term_u03bc__1___closed__2_value;
static const lean_string_object lp_mathlib_RingTheory_LinearMap___aux__Mathlib__Algebra__Algebra__Bilinear______macroRules__RingTheory__LinearMap__term_u03bc__1___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = "app"};
static const lean_object* lp_mathlib_RingTheory_LinearMap___aux__Mathlib__Algebra__Algebra__Bilinear______macroRules__RingTheory__LinearMap__term_u03bc__1___closed__3 = (const lean_object*)&lp_mathlib_RingTheory_LinearMap___aux__Mathlib__Algebra__Algebra__Bilinear______macroRules__RingTheory__LinearMap__term_u03bc__1___closed__3_value;
static const lean_ctor_object lp_mathlib_RingTheory_LinearMap___aux__Mathlib__Algebra__Algebra__Bilinear______macroRules__RingTheory__LinearMap__term_u03bc__1___closed__4_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_RingTheory_LinearMap___aux__Mathlib__Algebra__Algebra__Bilinear______macroRules__RingTheory__LinearMap__term_u03bc__1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib_RingTheory_LinearMap___aux__Mathlib__Algebra__Algebra__Bilinear______macroRules__RingTheory__LinearMap__term_u03bc__1___closed__4_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_RingTheory_LinearMap___aux__Mathlib__Algebra__Algebra__Bilinear______macroRules__RingTheory__LinearMap__term_u03bc__1___closed__4_value_aux_0),((lean_object*)&lp_mathlib_RingTheory_LinearMap___aux__Mathlib__Algebra__Algebra__Bilinear______macroRules__RingTheory__LinearMap__term_u03bc__1___closed__1_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib_RingTheory_LinearMap___aux__Mathlib__Algebra__Algebra__Bilinear______macroRules__RingTheory__LinearMap__term_u03bc__1___closed__4_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_RingTheory_LinearMap___aux__Mathlib__Algebra__Algebra__Bilinear______macroRules__RingTheory__LinearMap__term_u03bc__1___closed__4_value_aux_1),((lean_object*)&lp_mathlib_RingTheory_LinearMap___aux__Mathlib__Algebra__Algebra__Bilinear______macroRules__RingTheory__LinearMap__term_u03bc__1___closed__2_value),LEAN_SCALAR_PTR_LITERAL(75, 170, 162, 138, 136, 204, 251, 229)}};
static const lean_ctor_object lp_mathlib_RingTheory_LinearMap___aux__Mathlib__Algebra__Algebra__Bilinear______macroRules__RingTheory__LinearMap__term_u03bc__1___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_RingTheory_LinearMap___aux__Mathlib__Algebra__Algebra__Bilinear______macroRules__RingTheory__LinearMap__term_u03bc__1___closed__4_value_aux_2),((lean_object*)&lp_mathlib_RingTheory_LinearMap___aux__Mathlib__Algebra__Algebra__Bilinear______macroRules__RingTheory__LinearMap__term_u03bc__1___closed__3_value),LEAN_SCALAR_PTR_LITERAL(69, 118, 10, 41, 220, 156, 243, 179)}};
static const lean_object* lp_mathlib_RingTheory_LinearMap___aux__Mathlib__Algebra__Algebra__Bilinear______macroRules__RingTheory__LinearMap__term_u03bc__1___closed__4 = (const lean_object*)&lp_mathlib_RingTheory_LinearMap___aux__Mathlib__Algebra__Algebra__Bilinear______macroRules__RingTheory__LinearMap__term_u03bc__1___closed__4_value;
static const lean_string_object lp_mathlib_RingTheory_LinearMap___aux__Mathlib__Algebra__Algebra__Bilinear______macroRules__RingTheory__LinearMap__term_u03bc__1___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 15, .m_capacity = 15, .m_length = 14, .m_data = "LinearMap.mul'"};
static const lean_object* lp_mathlib_RingTheory_LinearMap___aux__Mathlib__Algebra__Algebra__Bilinear______macroRules__RingTheory__LinearMap__term_u03bc__1___closed__5 = (const lean_object*)&lp_mathlib_RingTheory_LinearMap___aux__Mathlib__Algebra__Algebra__Bilinear______macroRules__RingTheory__LinearMap__term_u03bc__1___closed__5_value;
static lean_once_cell_t lp_mathlib_RingTheory_LinearMap___aux__Mathlib__Algebra__Algebra__Bilinear______macroRules__RingTheory__LinearMap__term_u03bc__1___closed__6_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_RingTheory_LinearMap___aux__Mathlib__Algebra__Algebra__Bilinear______macroRules__RingTheory__LinearMap__term_u03bc__1___closed__6;
static const lean_string_object lp_mathlib_RingTheory_LinearMap___aux__Mathlib__Algebra__Algebra__Bilinear______macroRules__RingTheory__LinearMap__term_u03bc__1___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "mul'"};
static const lean_object* lp_mathlib_RingTheory_LinearMap___aux__Mathlib__Algebra__Algebra__Bilinear______macroRules__RingTheory__LinearMap__term_u03bc__1___closed__7 = (const lean_object*)&lp_mathlib_RingTheory_LinearMap___aux__Mathlib__Algebra__Algebra__Bilinear______macroRules__RingTheory__LinearMap__term_u03bc__1___closed__7_value;
static const lean_ctor_object lp_mathlib_RingTheory_LinearMap___aux__Mathlib__Algebra__Algebra__Bilinear______macroRules__RingTheory__LinearMap__term_u03bc__1___closed__8_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_RingTheory_LinearMap_term_u03bc___closed__1_value),LEAN_SCALAR_PTR_LITERAL(29, 55, 59, 137, 47, 1, 37, 113)}};
static const lean_ctor_object lp_mathlib_RingTheory_LinearMap___aux__Mathlib__Algebra__Algebra__Bilinear______macroRules__RingTheory__LinearMap__term_u03bc__1___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_RingTheory_LinearMap___aux__Mathlib__Algebra__Algebra__Bilinear______macroRules__RingTheory__LinearMap__term_u03bc__1___closed__8_value_aux_0),((lean_object*)&lp_mathlib_RingTheory_LinearMap___aux__Mathlib__Algebra__Algebra__Bilinear______macroRules__RingTheory__LinearMap__term_u03bc__1___closed__7_value),LEAN_SCALAR_PTR_LITERAL(99, 229, 230, 224, 41, 200, 41, 204)}};
static const lean_object* lp_mathlib_RingTheory_LinearMap___aux__Mathlib__Algebra__Algebra__Bilinear______macroRules__RingTheory__LinearMap__term_u03bc__1___closed__8 = (const lean_object*)&lp_mathlib_RingTheory_LinearMap___aux__Mathlib__Algebra__Algebra__Bilinear______macroRules__RingTheory__LinearMap__term_u03bc__1___closed__8_value;
static const lean_ctor_object lp_mathlib_RingTheory_LinearMap___aux__Mathlib__Algebra__Algebra__Bilinear______macroRules__RingTheory__LinearMap__term_u03bc__1___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_RingTheory_LinearMap___aux__Mathlib__Algebra__Algebra__Bilinear______macroRules__RingTheory__LinearMap__term_u03bc__1___closed__8_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_RingTheory_LinearMap___aux__Mathlib__Algebra__Algebra__Bilinear______macroRules__RingTheory__LinearMap__term_u03bc__1___closed__9 = (const lean_object*)&lp_mathlib_RingTheory_LinearMap___aux__Mathlib__Algebra__Algebra__Bilinear______macroRules__RingTheory__LinearMap__term_u03bc__1___closed__9_value;
static const lean_ctor_object lp_mathlib_RingTheory_LinearMap___aux__Mathlib__Algebra__Algebra__Bilinear______macroRules__RingTheory__LinearMap__term_u03bc__1___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_RingTheory_LinearMap___aux__Mathlib__Algebra__Algebra__Bilinear______macroRules__RingTheory__LinearMap__term_u03bc__1___closed__9_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_RingTheory_LinearMap___aux__Mathlib__Algebra__Algebra__Bilinear______macroRules__RingTheory__LinearMap__term_u03bc__1___closed__10 = (const lean_object*)&lp_mathlib_RingTheory_LinearMap___aux__Mathlib__Algebra__Algebra__Bilinear______macroRules__RingTheory__LinearMap__term_u03bc__1___closed__10_value;
static const lean_string_object lp_mathlib_RingTheory_LinearMap___aux__Mathlib__Algebra__Algebra__Bilinear______macroRules__RingTheory__LinearMap__term_u03bc__1___closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "null"};
static const lean_object* lp_mathlib_RingTheory_LinearMap___aux__Mathlib__Algebra__Algebra__Bilinear______macroRules__RingTheory__LinearMap__term_u03bc__1___closed__11 = (const lean_object*)&lp_mathlib_RingTheory_LinearMap___aux__Mathlib__Algebra__Algebra__Bilinear______macroRules__RingTheory__LinearMap__term_u03bc__1___closed__11_value;
static const lean_ctor_object lp_mathlib_RingTheory_LinearMap___aux__Mathlib__Algebra__Algebra__Bilinear______macroRules__RingTheory__LinearMap__term_u03bc__1___closed__12_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_RingTheory_LinearMap___aux__Mathlib__Algebra__Algebra__Bilinear______macroRules__RingTheory__LinearMap__term_u03bc__1___closed__11_value),LEAN_SCALAR_PTR_LITERAL(24, 58, 49, 223, 146, 207, 197, 136)}};
static const lean_object* lp_mathlib_RingTheory_LinearMap___aux__Mathlib__Algebra__Algebra__Bilinear______macroRules__RingTheory__LinearMap__term_u03bc__1___closed__12 = (const lean_object*)&lp_mathlib_RingTheory_LinearMap___aux__Mathlib__Algebra__Algebra__Bilinear______macroRules__RingTheory__LinearMap__term_u03bc__1___closed__12_value;
static const lean_string_object lp_mathlib_RingTheory_LinearMap___aux__Mathlib__Algebra__Algebra__Bilinear______macroRules__RingTheory__LinearMap__term_u03bc__1___closed__13_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "hole"};
static const lean_object* lp_mathlib_RingTheory_LinearMap___aux__Mathlib__Algebra__Algebra__Bilinear______macroRules__RingTheory__LinearMap__term_u03bc__1___closed__13 = (const lean_object*)&lp_mathlib_RingTheory_LinearMap___aux__Mathlib__Algebra__Algebra__Bilinear______macroRules__RingTheory__LinearMap__term_u03bc__1___closed__13_value;
static const lean_ctor_object lp_mathlib_RingTheory_LinearMap___aux__Mathlib__Algebra__Algebra__Bilinear______macroRules__RingTheory__LinearMap__term_u03bc__1___closed__14_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_RingTheory_LinearMap___aux__Mathlib__Algebra__Algebra__Bilinear______macroRules__RingTheory__LinearMap__term_u03bc__1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib_RingTheory_LinearMap___aux__Mathlib__Algebra__Algebra__Bilinear______macroRules__RingTheory__LinearMap__term_u03bc__1___closed__14_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_RingTheory_LinearMap___aux__Mathlib__Algebra__Algebra__Bilinear______macroRules__RingTheory__LinearMap__term_u03bc__1___closed__14_value_aux_0),((lean_object*)&lp_mathlib_RingTheory_LinearMap___aux__Mathlib__Algebra__Algebra__Bilinear______macroRules__RingTheory__LinearMap__term_u03bc__1___closed__1_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib_RingTheory_LinearMap___aux__Mathlib__Algebra__Algebra__Bilinear______macroRules__RingTheory__LinearMap__term_u03bc__1___closed__14_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_RingTheory_LinearMap___aux__Mathlib__Algebra__Algebra__Bilinear______macroRules__RingTheory__LinearMap__term_u03bc__1___closed__14_value_aux_1),((lean_object*)&lp_mathlib_RingTheory_LinearMap___aux__Mathlib__Algebra__Algebra__Bilinear______macroRules__RingTheory__LinearMap__term_u03bc__1___closed__2_value),LEAN_SCALAR_PTR_LITERAL(75, 170, 162, 138, 136, 204, 251, 229)}};
static const lean_ctor_object lp_mathlib_RingTheory_LinearMap___aux__Mathlib__Algebra__Algebra__Bilinear______macroRules__RingTheory__LinearMap__term_u03bc__1___closed__14_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_RingTheory_LinearMap___aux__Mathlib__Algebra__Algebra__Bilinear______macroRules__RingTheory__LinearMap__term_u03bc__1___closed__14_value_aux_2),((lean_object*)&lp_mathlib_RingTheory_LinearMap___aux__Mathlib__Algebra__Algebra__Bilinear______macroRules__RingTheory__LinearMap__term_u03bc__1___closed__13_value),LEAN_SCALAR_PTR_LITERAL(135, 134, 219, 115, 97, 130, 74, 55)}};
static const lean_object* lp_mathlib_RingTheory_LinearMap___aux__Mathlib__Algebra__Algebra__Bilinear______macroRules__RingTheory__LinearMap__term_u03bc__1___closed__14 = (const lean_object*)&lp_mathlib_RingTheory_LinearMap___aux__Mathlib__Algebra__Algebra__Bilinear______macroRules__RingTheory__LinearMap__term_u03bc__1___closed__14_value;
static const lean_string_object lp_mathlib_RingTheory_LinearMap___aux__Mathlib__Algebra__Algebra__Bilinear______macroRules__RingTheory__LinearMap__term_u03bc__1___closed__15_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = "_"};
static const lean_object* lp_mathlib_RingTheory_LinearMap___aux__Mathlib__Algebra__Algebra__Bilinear______macroRules__RingTheory__LinearMap__term_u03bc__1___closed__15 = (const lean_object*)&lp_mathlib_RingTheory_LinearMap___aux__Mathlib__Algebra__Algebra__Bilinear______macroRules__RingTheory__LinearMap__term_u03bc__1___closed__15_value;
LEAN_EXPORT lean_object* lp_mathlib_RingTheory_LinearMap___aux__Mathlib__Algebra__Algebra__Bilinear______macroRules__RingTheory__LinearMap__term_u03bc__1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RingTheory_LinearMap___aux__Mathlib__Algebra__Algebra__Bilinear______macroRules__RingTheory__LinearMap__term_u03bc__1___boxed(lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_RingTheory_LinearMap___aux__Mathlib__Algebra__Algebra__Bilinear______unexpand__LinearMap__mul_x27__1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "ident"};
static const lean_object* lp_mathlib_RingTheory_LinearMap___aux__Mathlib__Algebra__Algebra__Bilinear______unexpand__LinearMap__mul_x27__1___closed__0 = (const lean_object*)&lp_mathlib_RingTheory_LinearMap___aux__Mathlib__Algebra__Algebra__Bilinear______unexpand__LinearMap__mul_x27__1___closed__0_value;
static const lean_ctor_object lp_mathlib_RingTheory_LinearMap___aux__Mathlib__Algebra__Algebra__Bilinear______unexpand__LinearMap__mul_x27__1___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_RingTheory_LinearMap___aux__Mathlib__Algebra__Algebra__Bilinear______unexpand__LinearMap__mul_x27__1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(52, 159, 208, 51, 14, 60, 6, 71)}};
static const lean_object* lp_mathlib_RingTheory_LinearMap___aux__Mathlib__Algebra__Algebra__Bilinear______unexpand__LinearMap__mul_x27__1___closed__1 = (const lean_object*)&lp_mathlib_RingTheory_LinearMap___aux__Mathlib__Algebra__Algebra__Bilinear______unexpand__LinearMap__mul_x27__1___closed__1_value;
LEAN_EXPORT lean_object* lp_mathlib_RingTheory_LinearMap___aux__Mathlib__Algebra__Algebra__Bilinear______unexpand__LinearMap__mul_x27__1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RingTheory_LinearMap___aux__Mathlib__Algebra__Algebra__Bilinear______unexpand__LinearMap__mul_x27__1___boxed(lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_RingTheory_LinearMap_term_u03bc_x5b___x5d___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 8, .m_data = "termμ[_]"};
static const lean_object* lp_mathlib_RingTheory_LinearMap_term_u03bc_x5b___x5d___closed__0 = (const lean_object*)&lp_mathlib_RingTheory_LinearMap_term_u03bc_x5b___x5d___closed__0_value;
static const lean_ctor_object lp_mathlib_RingTheory_LinearMap_term_u03bc_x5b___x5d___closed__1_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_RingTheory_LinearMap_term_u03bc___closed__0_value),LEAN_SCALAR_PTR_LITERAL(204, 50, 210, 176, 233, 167, 74, 91)}};
static const lean_ctor_object lp_mathlib_RingTheory_LinearMap_term_u03bc_x5b___x5d___closed__1_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_RingTheory_LinearMap_term_u03bc_x5b___x5d___closed__1_value_aux_0),((lean_object*)&lp_mathlib_RingTheory_LinearMap_term_u03bc___closed__1_value),LEAN_SCALAR_PTR_LITERAL(118, 100, 27, 238, 183, 36, 185, 13)}};
static const lean_ctor_object lp_mathlib_RingTheory_LinearMap_term_u03bc_x5b___x5d___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_RingTheory_LinearMap_term_u03bc_x5b___x5d___closed__1_value_aux_1),((lean_object*)&lp_mathlib_RingTheory_LinearMap_term_u03bc_x5b___x5d___closed__0_value),LEAN_SCALAR_PTR_LITERAL(152, 119, 80, 39, 150, 53, 24, 226)}};
static const lean_object* lp_mathlib_RingTheory_LinearMap_term_u03bc_x5b___x5d___closed__1 = (const lean_object*)&lp_mathlib_RingTheory_LinearMap_term_u03bc_x5b___x5d___closed__1_value;
static const lean_string_object lp_mathlib_RingTheory_LinearMap_term_u03bc_x5b___x5d___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "andthen"};
static const lean_object* lp_mathlib_RingTheory_LinearMap_term_u03bc_x5b___x5d___closed__2 = (const lean_object*)&lp_mathlib_RingTheory_LinearMap_term_u03bc_x5b___x5d___closed__2_value;
static const lean_ctor_object lp_mathlib_RingTheory_LinearMap_term_u03bc_x5b___x5d___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_RingTheory_LinearMap_term_u03bc_x5b___x5d___closed__2_value),LEAN_SCALAR_PTR_LITERAL(40, 255, 78, 30, 143, 119, 117, 174)}};
static const lean_object* lp_mathlib_RingTheory_LinearMap_term_u03bc_x5b___x5d___closed__3 = (const lean_object*)&lp_mathlib_RingTheory_LinearMap_term_u03bc_x5b___x5d___closed__3_value;
static const lean_string_object lp_mathlib_RingTheory_LinearMap_term_u03bc_x5b___x5d___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 2, .m_data = "μ["};
static const lean_object* lp_mathlib_RingTheory_LinearMap_term_u03bc_x5b___x5d___closed__4 = (const lean_object*)&lp_mathlib_RingTheory_LinearMap_term_u03bc_x5b___x5d___closed__4_value;
static const lean_ctor_object lp_mathlib_RingTheory_LinearMap_term_u03bc_x5b___x5d___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 5}, .m_objs = {((lean_object*)&lp_mathlib_RingTheory_LinearMap_term_u03bc_x5b___x5d___closed__4_value)}};
static const lean_object* lp_mathlib_RingTheory_LinearMap_term_u03bc_x5b___x5d___closed__5 = (const lean_object*)&lp_mathlib_RingTheory_LinearMap_term_u03bc_x5b___x5d___closed__5_value;
static const lean_string_object lp_mathlib_RingTheory_LinearMap_term_u03bc_x5b___x5d___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "term"};
static const lean_object* lp_mathlib_RingTheory_LinearMap_term_u03bc_x5b___x5d___closed__6 = (const lean_object*)&lp_mathlib_RingTheory_LinearMap_term_u03bc_x5b___x5d___closed__6_value;
static const lean_ctor_object lp_mathlib_RingTheory_LinearMap_term_u03bc_x5b___x5d___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_RingTheory_LinearMap_term_u03bc_x5b___x5d___closed__6_value),LEAN_SCALAR_PTR_LITERAL(187, 230, 181, 162, 253, 146, 122, 119)}};
static const lean_object* lp_mathlib_RingTheory_LinearMap_term_u03bc_x5b___x5d___closed__7 = (const lean_object*)&lp_mathlib_RingTheory_LinearMap_term_u03bc_x5b___x5d___closed__7_value;
static const lean_ctor_object lp_mathlib_RingTheory_LinearMap_term_u03bc_x5b___x5d___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 7}, .m_objs = {((lean_object*)&lp_mathlib_RingTheory_LinearMap_term_u03bc_x5b___x5d___closed__7_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_RingTheory_LinearMap_term_u03bc_x5b___x5d___closed__8 = (const lean_object*)&lp_mathlib_RingTheory_LinearMap_term_u03bc_x5b___x5d___closed__8_value;
static const lean_ctor_object lp_mathlib_RingTheory_LinearMap_term_u03bc_x5b___x5d___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib_RingTheory_LinearMap_term_u03bc_x5b___x5d___closed__3_value),((lean_object*)&lp_mathlib_RingTheory_LinearMap_term_u03bc_x5b___x5d___closed__5_value),((lean_object*)&lp_mathlib_RingTheory_LinearMap_term_u03bc_x5b___x5d___closed__8_value)}};
static const lean_object* lp_mathlib_RingTheory_LinearMap_term_u03bc_x5b___x5d___closed__9 = (const lean_object*)&lp_mathlib_RingTheory_LinearMap_term_u03bc_x5b___x5d___closed__9_value;
static const lean_string_object lp_mathlib_RingTheory_LinearMap_term_u03bc_x5b___x5d___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = "]"};
static const lean_object* lp_mathlib_RingTheory_LinearMap_term_u03bc_x5b___x5d___closed__10 = (const lean_object*)&lp_mathlib_RingTheory_LinearMap_term_u03bc_x5b___x5d___closed__10_value;
static const lean_ctor_object lp_mathlib_RingTheory_LinearMap_term_u03bc_x5b___x5d___closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 5}, .m_objs = {((lean_object*)&lp_mathlib_RingTheory_LinearMap_term_u03bc_x5b___x5d___closed__10_value)}};
static const lean_object* lp_mathlib_RingTheory_LinearMap_term_u03bc_x5b___x5d___closed__11 = (const lean_object*)&lp_mathlib_RingTheory_LinearMap_term_u03bc_x5b___x5d___closed__11_value;
static const lean_ctor_object lp_mathlib_RingTheory_LinearMap_term_u03bc_x5b___x5d___closed__12_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib_RingTheory_LinearMap_term_u03bc_x5b___x5d___closed__3_value),((lean_object*)&lp_mathlib_RingTheory_LinearMap_term_u03bc_x5b___x5d___closed__9_value),((lean_object*)&lp_mathlib_RingTheory_LinearMap_term_u03bc_x5b___x5d___closed__11_value)}};
static const lean_object* lp_mathlib_RingTheory_LinearMap_term_u03bc_x5b___x5d___closed__12 = (const lean_object*)&lp_mathlib_RingTheory_LinearMap_term_u03bc_x5b___x5d___closed__12_value;
static const lean_ctor_object lp_mathlib_RingTheory_LinearMap_term_u03bc_x5b___x5d___closed__13_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 3}, .m_objs = {((lean_object*)&lp_mathlib_RingTheory_LinearMap_term_u03bc_x5b___x5d___closed__1_value),((lean_object*)(((size_t)(1024) << 1) | 1)),((lean_object*)&lp_mathlib_RingTheory_LinearMap_term_u03bc_x5b___x5d___closed__12_value)}};
static const lean_object* lp_mathlib_RingTheory_LinearMap_term_u03bc_x5b___x5d___closed__13 = (const lean_object*)&lp_mathlib_RingTheory_LinearMap_term_u03bc_x5b___x5d___closed__13_value;
LEAN_EXPORT const lean_object* lp_mathlib_RingTheory_LinearMap_term_u03bc_x5b___x5d = (const lean_object*)&lp_mathlib_RingTheory_LinearMap_term_u03bc_x5b___x5d___closed__13_value;
LEAN_EXPORT lean_object* lp_mathlib_RingTheory_LinearMap___aux__Mathlib__Algebra__Algebra__Bilinear______macroRules__RingTheory__LinearMap__term_u03bc_x5b___x5d__1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RingTheory_LinearMap___aux__Mathlib__Algebra__Algebra__Bilinear______macroRules__RingTheory__LinearMap__term_u03bc_x5b___x5d__1___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RingTheory_LinearMap___aux__Mathlib__Algebra__Algebra__Bilinear______unexpand__LinearMap__mul_x27__2(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RingTheory_LinearMap___aux__Mathlib__Algebra__Algebra__Bilinear______unexpand__LinearMap__mul_x27__2___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalAlgHom_lmul___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalAlgHom_lmul(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalAlgHom_lmul___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Algebra_lmul___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Algebra_lmul___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Algebra_lmul(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Algebra_lmul___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LinearMap_mul_x27_x27___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LinearMap_mul_x27_x27(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LinearMap_mul___redArg___lam__0(lean_object* v_toMul_1_, lean_object* v_x1_2_, lean_object* v_x2_3_){
_start:
{
lean_object* v___x_4_; 
v___x_4_ = lean_apply_2(v_toMul_1_, v_x1_2_, v_x2_3_);
return v___x_4_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LinearMap_mul___redArg(lean_object* v_inst_5_){
_start:
{
lean_object* v___x_6_; lean_object* v_toMul_7_; lean_object* v___f_8_; lean_object* v___f_9_; 
v___x_6_ = lp_mathlib_NonUnitalNonAssocSemiring_toDistrib___redArg(v_inst_5_);
v_toMul_7_ = lean_ctor_get(v___x_6_, 0);
lean_inc(v_toMul_7_);
lean_dec_ref(v___x_6_);
v___f_8_ = lean_alloc_closure((void*)(lp_mathlib_LinearMap_mul___redArg___lam__0), 3, 1);
lean_closure_set(v___f_8_, 0, v_toMul_7_);
v___f_9_ = lean_alloc_closure((void*)(lp_mathlib_LinearMap_mk_u2082_x27_u209b_u2097___redArg___lam__0), 3, 1);
lean_closure_set(v___f_9_, 0, v___f_8_);
return v___f_9_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LinearMap_mul(lean_object* v_R_10_, lean_object* v_A_11_, lean_object* v_inst_12_, lean_object* v_inst_13_, lean_object* v_inst_14_, lean_object* v_inst_15_, lean_object* v_inst_16_){
_start:
{
lean_object* v___x_17_; 
v___x_17_ = lp_mathlib_LinearMap_mul___redArg(v_inst_13_);
return v___x_17_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LinearMap_mul___boxed(lean_object* v_R_18_, lean_object* v_A_19_, lean_object* v_inst_20_, lean_object* v_inst_21_, lean_object* v_inst_22_, lean_object* v_inst_23_, lean_object* v_inst_24_){
_start:
{
lean_object* v_res_25_; 
v_res_25_ = lp_mathlib_LinearMap_mul(v_R_18_, v_A_19_, v_inst_20_, v_inst_21_, v_inst_22_, v_inst_23_, v_inst_24_);
lean_dec(v_inst_22_);
lean_dec_ref(v_inst_20_);
return v_res_25_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LinearMap_mul_x27___redArg(lean_object* v_inst_27_, lean_object* v_inst_28_, lean_object* v_inst_29_){
_start:
{
lean_object* v_toAddCommMonoid_30_; lean_object* v___f_31_; lean_object* v___x_32_; lean_object* v___x_33_; 
v_toAddCommMonoid_30_ = lean_ctor_get(v_inst_28_, 0);
lean_inc_ref_n(v_toAddCommMonoid_30_, 2);
v___f_31_ = ((lean_object*)(lp_mathlib_LinearMap_mul_x27___redArg___closed__0));
v___x_32_ = lp_mathlib_LinearMap_mul___redArg(v_inst_28_);
lean_inc(v_inst_29_);
lean_inc_ref(v_inst_27_);
v___x_33_ = lp_mathlib_TensorProduct_liftAux___redArg(v_inst_27_, v_inst_27_, v___f_31_, v_toAddCommMonoid_30_, v_toAddCommMonoid_30_, v_inst_29_, v_inst_29_, v___x_32_);
return v___x_33_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LinearMap_mul_x27(lean_object* v_R_34_, lean_object* v_A_35_, lean_object* v_inst_36_, lean_object* v_inst_37_, lean_object* v_inst_38_, lean_object* v_inst_39_, lean_object* v_inst_40_){
_start:
{
lean_object* v___x_41_; 
v___x_41_ = lp_mathlib_LinearMap_mul_x27___redArg(v_inst_36_, v_inst_37_, v_inst_38_);
return v___x_41_;
}
}
static lean_object* _init_lp_mathlib_RingTheory_LinearMap___aux__Mathlib__Algebra__Algebra__Bilinear______macroRules__RingTheory__LinearMap__term_u03bc__1___closed__6(void){
_start:
{
lean_object* v___x_67_; lean_object* v___x_68_; 
v___x_67_ = ((lean_object*)(lp_mathlib_RingTheory_LinearMap___aux__Mathlib__Algebra__Algebra__Bilinear______macroRules__RingTheory__LinearMap__term_u03bc__1___closed__5));
v___x_68_ = l_String_toRawSubstring_x27(v___x_67_);
return v___x_68_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RingTheory_LinearMap___aux__Mathlib__Algebra__Algebra__Bilinear______macroRules__RingTheory__LinearMap__term_u03bc__1(lean_object* v_x_89_, lean_object* v_a_90_, lean_object* v_a_91_){
_start:
{
lean_object* v___x_92_; uint8_t v___x_93_; 
v___x_92_ = ((lean_object*)(lp_mathlib_RingTheory_LinearMap_term_u03bc___closed__3));
v___x_93_ = l_Lean_Syntax_isOfKind(v_x_89_, v___x_92_);
if (v___x_93_ == 0)
{
lean_object* v___x_94_; lean_object* v___x_95_; 
v___x_94_ = lean_box(1);
v___x_95_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_95_, 0, v___x_94_);
lean_ctor_set(v___x_95_, 1, v_a_91_);
return v___x_95_;
}
else
{
lean_object* v_quotContext_96_; lean_object* v_currMacroScope_97_; lean_object* v_ref_98_; uint8_t v___x_99_; lean_object* v___x_100_; lean_object* v___x_101_; lean_object* v___x_102_; lean_object* v___x_103_; lean_object* v___x_104_; lean_object* v___x_105_; lean_object* v___x_106_; lean_object* v___x_107_; lean_object* v___x_108_; lean_object* v___x_109_; lean_object* v___x_110_; lean_object* v___x_111_; lean_object* v___x_112_; lean_object* v___x_113_; lean_object* v___x_114_; 
v_quotContext_96_ = lean_ctor_get(v_a_90_, 1);
v_currMacroScope_97_ = lean_ctor_get(v_a_90_, 2);
v_ref_98_ = lean_ctor_get(v_a_90_, 5);
v___x_99_ = 0;
v___x_100_ = l_Lean_SourceInfo_fromRef(v_ref_98_, v___x_99_);
v___x_101_ = ((lean_object*)(lp_mathlib_RingTheory_LinearMap___aux__Mathlib__Algebra__Algebra__Bilinear______macroRules__RingTheory__LinearMap__term_u03bc__1___closed__4));
v___x_102_ = lean_obj_once(&lp_mathlib_RingTheory_LinearMap___aux__Mathlib__Algebra__Algebra__Bilinear______macroRules__RingTheory__LinearMap__term_u03bc__1___closed__6, &lp_mathlib_RingTheory_LinearMap___aux__Mathlib__Algebra__Algebra__Bilinear______macroRules__RingTheory__LinearMap__term_u03bc__1___closed__6_once, _init_lp_mathlib_RingTheory_LinearMap___aux__Mathlib__Algebra__Algebra__Bilinear______macroRules__RingTheory__LinearMap__term_u03bc__1___closed__6);
v___x_103_ = ((lean_object*)(lp_mathlib_RingTheory_LinearMap___aux__Mathlib__Algebra__Algebra__Bilinear______macroRules__RingTheory__LinearMap__term_u03bc__1___closed__8));
lean_inc(v_currMacroScope_97_);
lean_inc(v_quotContext_96_);
v___x_104_ = l_Lean_addMacroScope(v_quotContext_96_, v___x_103_, v_currMacroScope_97_);
v___x_105_ = ((lean_object*)(lp_mathlib_RingTheory_LinearMap___aux__Mathlib__Algebra__Algebra__Bilinear______macroRules__RingTheory__LinearMap__term_u03bc__1___closed__10));
lean_inc_n(v___x_100_, 4);
v___x_106_ = lean_alloc_ctor(3, 4, 0);
lean_ctor_set(v___x_106_, 0, v___x_100_);
lean_ctor_set(v___x_106_, 1, v___x_102_);
lean_ctor_set(v___x_106_, 2, v___x_104_);
lean_ctor_set(v___x_106_, 3, v___x_105_);
v___x_107_ = ((lean_object*)(lp_mathlib_RingTheory_LinearMap___aux__Mathlib__Algebra__Algebra__Bilinear______macroRules__RingTheory__LinearMap__term_u03bc__1___closed__12));
v___x_108_ = ((lean_object*)(lp_mathlib_RingTheory_LinearMap___aux__Mathlib__Algebra__Algebra__Bilinear______macroRules__RingTheory__LinearMap__term_u03bc__1___closed__14));
v___x_109_ = ((lean_object*)(lp_mathlib_RingTheory_LinearMap___aux__Mathlib__Algebra__Algebra__Bilinear______macroRules__RingTheory__LinearMap__term_u03bc__1___closed__15));
v___x_110_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_110_, 0, v___x_100_);
lean_ctor_set(v___x_110_, 1, v___x_109_);
v___x_111_ = l_Lean_Syntax_node1(v___x_100_, v___x_108_, v___x_110_);
lean_inc(v___x_111_);
v___x_112_ = l_Lean_Syntax_node2(v___x_100_, v___x_107_, v___x_111_, v___x_111_);
v___x_113_ = l_Lean_Syntax_node2(v___x_100_, v___x_101_, v___x_106_, v___x_112_);
v___x_114_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_114_, 0, v___x_113_);
lean_ctor_set(v___x_114_, 1, v_a_91_);
return v___x_114_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_RingTheory_LinearMap___aux__Mathlib__Algebra__Algebra__Bilinear______macroRules__RingTheory__LinearMap__term_u03bc__1___boxed(lean_object* v_x_115_, lean_object* v_a_116_, lean_object* v_a_117_){
_start:
{
lean_object* v_res_118_; 
v_res_118_ = lp_mathlib_RingTheory_LinearMap___aux__Mathlib__Algebra__Algebra__Bilinear______macroRules__RingTheory__LinearMap__term_u03bc__1(v_x_115_, v_a_116_, v_a_117_);
lean_dec_ref(v_a_116_);
return v_res_118_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RingTheory_LinearMap___aux__Mathlib__Algebra__Algebra__Bilinear______unexpand__LinearMap__mul_x27__1(lean_object* v_x_122_, lean_object* v_a_123_, lean_object* v_a_124_){
_start:
{
lean_object* v___x_125_; uint8_t v___x_126_; 
v___x_125_ = ((lean_object*)(lp_mathlib_RingTheory_LinearMap___aux__Mathlib__Algebra__Algebra__Bilinear______macroRules__RingTheory__LinearMap__term_u03bc__1___closed__4));
lean_inc(v_x_122_);
v___x_126_ = l_Lean_Syntax_isOfKind(v_x_122_, v___x_125_);
if (v___x_126_ == 0)
{
lean_object* v___x_127_; lean_object* v___x_128_; 
lean_dec(v_x_122_);
v___x_127_ = lean_box(0);
v___x_128_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_128_, 0, v___x_127_);
lean_ctor_set(v___x_128_, 1, v_a_124_);
return v___x_128_;
}
else
{
lean_object* v___x_129_; lean_object* v___x_130_; lean_object* v___x_131_; uint8_t v___x_132_; 
v___x_129_ = lean_unsigned_to_nat(0u);
v___x_130_ = l_Lean_Syntax_getArg(v_x_122_, v___x_129_);
v___x_131_ = ((lean_object*)(lp_mathlib_RingTheory_LinearMap___aux__Mathlib__Algebra__Algebra__Bilinear______unexpand__LinearMap__mul_x27__1___closed__1));
lean_inc(v___x_130_);
v___x_132_ = l_Lean_Syntax_isOfKind(v___x_130_, v___x_131_);
if (v___x_132_ == 0)
{
lean_object* v___x_133_; lean_object* v___x_134_; 
lean_dec(v___x_130_);
lean_dec(v_x_122_);
v___x_133_ = lean_box(0);
v___x_134_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_134_, 0, v___x_133_);
lean_ctor_set(v___x_134_, 1, v_a_124_);
return v___x_134_;
}
else
{
lean_object* v___x_135_; lean_object* v___x_136_; lean_object* v___x_137_; uint8_t v___x_138_; 
v___x_135_ = lean_unsigned_to_nat(1u);
v___x_136_ = l_Lean_Syntax_getArg(v_x_122_, v___x_135_);
lean_dec(v_x_122_);
v___x_137_ = lean_unsigned_to_nat(2u);
lean_inc(v___x_136_);
v___x_138_ = l_Lean_Syntax_matchesNull(v___x_136_, v___x_137_);
if (v___x_138_ == 0)
{
lean_object* v___x_139_; lean_object* v___x_140_; 
lean_dec(v___x_136_);
lean_dec(v___x_130_);
v___x_139_ = lean_box(0);
v___x_140_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_140_, 0, v___x_139_);
lean_ctor_set(v___x_140_, 1, v_a_124_);
return v___x_140_;
}
else
{
lean_object* v___x_141_; lean_object* v___x_142_; uint8_t v___x_143_; 
v___x_141_ = l_Lean_Syntax_getArg(v___x_136_, v___x_129_);
v___x_142_ = ((lean_object*)(lp_mathlib_RingTheory_LinearMap___aux__Mathlib__Algebra__Algebra__Bilinear______macroRules__RingTheory__LinearMap__term_u03bc__1___closed__14));
v___x_143_ = l_Lean_Syntax_isOfKind(v___x_141_, v___x_142_);
if (v___x_143_ == 0)
{
lean_object* v___x_144_; lean_object* v___x_145_; 
lean_dec(v___x_136_);
lean_dec(v___x_130_);
v___x_144_ = lean_box(0);
v___x_145_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_145_, 0, v___x_144_);
lean_ctor_set(v___x_145_, 1, v_a_124_);
return v___x_145_;
}
else
{
lean_object* v___x_146_; uint8_t v___x_147_; 
v___x_146_ = l_Lean_Syntax_getArg(v___x_136_, v___x_135_);
lean_dec(v___x_136_);
v___x_147_ = l_Lean_Syntax_isOfKind(v___x_146_, v___x_142_);
if (v___x_147_ == 0)
{
lean_object* v___x_148_; lean_object* v___x_149_; 
lean_dec(v___x_130_);
v___x_148_ = lean_box(0);
v___x_149_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_149_, 0, v___x_148_);
lean_ctor_set(v___x_149_, 1, v_a_124_);
return v___x_149_;
}
else
{
lean_object* v_ref_150_; uint8_t v___x_151_; lean_object* v___x_152_; lean_object* v___x_153_; lean_object* v___x_154_; lean_object* v___x_155_; lean_object* v___x_156_; lean_object* v___x_157_; 
v_ref_150_ = l_Lean_replaceRef(v___x_130_, v_a_123_);
lean_dec(v___x_130_);
v___x_151_ = 0;
v___x_152_ = l_Lean_SourceInfo_fromRef(v_ref_150_, v___x_151_);
lean_dec(v_ref_150_);
v___x_153_ = ((lean_object*)(lp_mathlib_RingTheory_LinearMap_term_u03bc___closed__3));
v___x_154_ = ((lean_object*)(lp_mathlib_RingTheory_LinearMap_term_u03bc___closed__4));
lean_inc(v___x_152_);
v___x_155_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_155_, 0, v___x_152_);
lean_ctor_set(v___x_155_, 1, v___x_154_);
v___x_156_ = l_Lean_Syntax_node1(v___x_152_, v___x_153_, v___x_155_);
v___x_157_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_157_, 0, v___x_156_);
lean_ctor_set(v___x_157_, 1, v_a_124_);
return v___x_157_;
}
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_RingTheory_LinearMap___aux__Mathlib__Algebra__Algebra__Bilinear______unexpand__LinearMap__mul_x27__1___boxed(lean_object* v_x_158_, lean_object* v_a_159_, lean_object* v_a_160_){
_start:
{
lean_object* v_res_161_; 
v_res_161_ = lp_mathlib_RingTheory_LinearMap___aux__Mathlib__Algebra__Algebra__Bilinear______unexpand__LinearMap__mul_x27__1(v_x_158_, v_a_159_, v_a_160_);
lean_dec(v_a_159_);
return v_res_161_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RingTheory_LinearMap___aux__Mathlib__Algebra__Algebra__Bilinear______macroRules__RingTheory__LinearMap__term_u03bc_x5b___x5d__1(lean_object* v_x_195_, lean_object* v_a_196_, lean_object* v_a_197_){
_start:
{
lean_object* v___x_198_; uint8_t v___x_199_; 
v___x_198_ = ((lean_object*)(lp_mathlib_RingTheory_LinearMap_term_u03bc_x5b___x5d___closed__1));
lean_inc(v_x_195_);
v___x_199_ = l_Lean_Syntax_isOfKind(v_x_195_, v___x_198_);
if (v___x_199_ == 0)
{
lean_object* v___x_200_; lean_object* v___x_201_; 
lean_dec(v_x_195_);
v___x_200_ = lean_box(1);
v___x_201_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_201_, 0, v___x_200_);
lean_ctor_set(v___x_201_, 1, v_a_197_);
return v___x_201_;
}
else
{
lean_object* v_quotContext_202_; lean_object* v_currMacroScope_203_; lean_object* v_ref_204_; lean_object* v___x_205_; lean_object* v___x_206_; uint8_t v___x_207_; lean_object* v___x_208_; lean_object* v___x_209_; lean_object* v___x_210_; lean_object* v___x_211_; lean_object* v___x_212_; lean_object* v___x_213_; lean_object* v___x_214_; lean_object* v___x_215_; lean_object* v___x_216_; lean_object* v___x_217_; lean_object* v___x_218_; lean_object* v___x_219_; lean_object* v___x_220_; lean_object* v___x_221_; lean_object* v___x_222_; 
v_quotContext_202_ = lean_ctor_get(v_a_196_, 1);
v_currMacroScope_203_ = lean_ctor_get(v_a_196_, 2);
v_ref_204_ = lean_ctor_get(v_a_196_, 5);
v___x_205_ = lean_unsigned_to_nat(1u);
v___x_206_ = l_Lean_Syntax_getArg(v_x_195_, v___x_205_);
lean_dec(v_x_195_);
v___x_207_ = 0;
v___x_208_ = l_Lean_SourceInfo_fromRef(v_ref_204_, v___x_207_);
v___x_209_ = ((lean_object*)(lp_mathlib_RingTheory_LinearMap___aux__Mathlib__Algebra__Algebra__Bilinear______macroRules__RingTheory__LinearMap__term_u03bc__1___closed__4));
v___x_210_ = lean_obj_once(&lp_mathlib_RingTheory_LinearMap___aux__Mathlib__Algebra__Algebra__Bilinear______macroRules__RingTheory__LinearMap__term_u03bc__1___closed__6, &lp_mathlib_RingTheory_LinearMap___aux__Mathlib__Algebra__Algebra__Bilinear______macroRules__RingTheory__LinearMap__term_u03bc__1___closed__6_once, _init_lp_mathlib_RingTheory_LinearMap___aux__Mathlib__Algebra__Algebra__Bilinear______macroRules__RingTheory__LinearMap__term_u03bc__1___closed__6);
v___x_211_ = ((lean_object*)(lp_mathlib_RingTheory_LinearMap___aux__Mathlib__Algebra__Algebra__Bilinear______macroRules__RingTheory__LinearMap__term_u03bc__1___closed__8));
lean_inc(v_currMacroScope_203_);
lean_inc(v_quotContext_202_);
v___x_212_ = l_Lean_addMacroScope(v_quotContext_202_, v___x_211_, v_currMacroScope_203_);
v___x_213_ = ((lean_object*)(lp_mathlib_RingTheory_LinearMap___aux__Mathlib__Algebra__Algebra__Bilinear______macroRules__RingTheory__LinearMap__term_u03bc__1___closed__10));
lean_inc_n(v___x_208_, 4);
v___x_214_ = lean_alloc_ctor(3, 4, 0);
lean_ctor_set(v___x_214_, 0, v___x_208_);
lean_ctor_set(v___x_214_, 1, v___x_210_);
lean_ctor_set(v___x_214_, 2, v___x_212_);
lean_ctor_set(v___x_214_, 3, v___x_213_);
v___x_215_ = ((lean_object*)(lp_mathlib_RingTheory_LinearMap___aux__Mathlib__Algebra__Algebra__Bilinear______macroRules__RingTheory__LinearMap__term_u03bc__1___closed__12));
v___x_216_ = ((lean_object*)(lp_mathlib_RingTheory_LinearMap___aux__Mathlib__Algebra__Algebra__Bilinear______macroRules__RingTheory__LinearMap__term_u03bc__1___closed__14));
v___x_217_ = ((lean_object*)(lp_mathlib_RingTheory_LinearMap___aux__Mathlib__Algebra__Algebra__Bilinear______macroRules__RingTheory__LinearMap__term_u03bc__1___closed__15));
v___x_218_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_218_, 0, v___x_208_);
lean_ctor_set(v___x_218_, 1, v___x_217_);
v___x_219_ = l_Lean_Syntax_node1(v___x_208_, v___x_216_, v___x_218_);
v___x_220_ = l_Lean_Syntax_node2(v___x_208_, v___x_215_, v___x_206_, v___x_219_);
v___x_221_ = l_Lean_Syntax_node2(v___x_208_, v___x_209_, v___x_214_, v___x_220_);
v___x_222_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_222_, 0, v___x_221_);
lean_ctor_set(v___x_222_, 1, v_a_197_);
return v___x_222_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_RingTheory_LinearMap___aux__Mathlib__Algebra__Algebra__Bilinear______macroRules__RingTheory__LinearMap__term_u03bc_x5b___x5d__1___boxed(lean_object* v_x_223_, lean_object* v_a_224_, lean_object* v_a_225_){
_start:
{
lean_object* v_res_226_; 
v_res_226_ = lp_mathlib_RingTheory_LinearMap___aux__Mathlib__Algebra__Algebra__Bilinear______macroRules__RingTheory__LinearMap__term_u03bc_x5b___x5d__1(v_x_223_, v_a_224_, v_a_225_);
lean_dec_ref(v_a_224_);
return v_res_226_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RingTheory_LinearMap___aux__Mathlib__Algebra__Algebra__Bilinear______unexpand__LinearMap__mul_x27__2(lean_object* v_x_227_, lean_object* v_a_228_, lean_object* v_a_229_){
_start:
{
lean_object* v___x_230_; uint8_t v___x_231_; 
v___x_230_ = ((lean_object*)(lp_mathlib_RingTheory_LinearMap___aux__Mathlib__Algebra__Algebra__Bilinear______macroRules__RingTheory__LinearMap__term_u03bc__1___closed__4));
lean_inc(v_x_227_);
v___x_231_ = l_Lean_Syntax_isOfKind(v_x_227_, v___x_230_);
if (v___x_231_ == 0)
{
lean_object* v___x_232_; lean_object* v___x_233_; 
lean_dec(v_x_227_);
v___x_232_ = lean_box(0);
v___x_233_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_233_, 0, v___x_232_);
lean_ctor_set(v___x_233_, 1, v_a_229_);
return v___x_233_;
}
else
{
lean_object* v___x_234_; lean_object* v___x_235_; lean_object* v___x_236_; uint8_t v___x_237_; 
v___x_234_ = lean_unsigned_to_nat(0u);
v___x_235_ = l_Lean_Syntax_getArg(v_x_227_, v___x_234_);
v___x_236_ = ((lean_object*)(lp_mathlib_RingTheory_LinearMap___aux__Mathlib__Algebra__Algebra__Bilinear______unexpand__LinearMap__mul_x27__1___closed__1));
lean_inc(v___x_235_);
v___x_237_ = l_Lean_Syntax_isOfKind(v___x_235_, v___x_236_);
if (v___x_237_ == 0)
{
lean_object* v___x_238_; lean_object* v___x_239_; 
lean_dec(v___x_235_);
lean_dec(v_x_227_);
v___x_238_ = lean_box(0);
v___x_239_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_239_, 0, v___x_238_);
lean_ctor_set(v___x_239_, 1, v_a_229_);
return v___x_239_;
}
else
{
lean_object* v___x_240_; lean_object* v___x_241_; lean_object* v___x_242_; uint8_t v___x_243_; 
v___x_240_ = lean_unsigned_to_nat(1u);
v___x_241_ = l_Lean_Syntax_getArg(v_x_227_, v___x_240_);
lean_dec(v_x_227_);
v___x_242_ = lean_unsigned_to_nat(2u);
lean_inc(v___x_241_);
v___x_243_ = l_Lean_Syntax_matchesNull(v___x_241_, v___x_242_);
if (v___x_243_ == 0)
{
lean_object* v___x_244_; lean_object* v___x_245_; 
lean_dec(v___x_241_);
lean_dec(v___x_235_);
v___x_244_ = lean_box(0);
v___x_245_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_245_, 0, v___x_244_);
lean_ctor_set(v___x_245_, 1, v_a_229_);
return v___x_245_;
}
else
{
lean_object* v___x_246_; lean_object* v___x_247_; uint8_t v___x_248_; 
v___x_246_ = l_Lean_Syntax_getArg(v___x_241_, v___x_240_);
v___x_247_ = ((lean_object*)(lp_mathlib_RingTheory_LinearMap___aux__Mathlib__Algebra__Algebra__Bilinear______macroRules__RingTheory__LinearMap__term_u03bc__1___closed__14));
v___x_248_ = l_Lean_Syntax_isOfKind(v___x_246_, v___x_247_);
if (v___x_248_ == 0)
{
lean_object* v___x_249_; lean_object* v___x_250_; 
lean_dec(v___x_241_);
lean_dec(v___x_235_);
v___x_249_ = lean_box(0);
v___x_250_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_250_, 0, v___x_249_);
lean_ctor_set(v___x_250_, 1, v_a_229_);
return v___x_250_;
}
else
{
lean_object* v___x_251_; lean_object* v_ref_252_; uint8_t v___x_253_; lean_object* v___x_254_; lean_object* v___x_255_; lean_object* v___x_256_; lean_object* v___x_257_; lean_object* v___x_258_; lean_object* v___x_259_; lean_object* v___x_260_; lean_object* v___x_261_; 
v___x_251_ = l_Lean_Syntax_getArg(v___x_241_, v___x_234_);
lean_dec(v___x_241_);
v_ref_252_ = l_Lean_replaceRef(v___x_235_, v_a_228_);
lean_dec(v___x_235_);
v___x_253_ = 0;
v___x_254_ = l_Lean_SourceInfo_fromRef(v_ref_252_, v___x_253_);
lean_dec(v_ref_252_);
v___x_255_ = ((lean_object*)(lp_mathlib_RingTheory_LinearMap_term_u03bc_x5b___x5d___closed__1));
v___x_256_ = ((lean_object*)(lp_mathlib_RingTheory_LinearMap_term_u03bc_x5b___x5d___closed__4));
lean_inc_n(v___x_254_, 2);
v___x_257_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_257_, 0, v___x_254_);
lean_ctor_set(v___x_257_, 1, v___x_256_);
v___x_258_ = ((lean_object*)(lp_mathlib_RingTheory_LinearMap_term_u03bc_x5b___x5d___closed__10));
v___x_259_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_259_, 0, v___x_254_);
lean_ctor_set(v___x_259_, 1, v___x_258_);
v___x_260_ = l_Lean_Syntax_node3(v___x_254_, v___x_255_, v___x_257_, v___x_251_, v___x_259_);
v___x_261_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_261_, 0, v___x_260_);
lean_ctor_set(v___x_261_, 1, v_a_229_);
return v___x_261_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_RingTheory_LinearMap___aux__Mathlib__Algebra__Algebra__Bilinear______unexpand__LinearMap__mul_x27__2___boxed(lean_object* v_x_262_, lean_object* v_a_263_, lean_object* v_a_264_){
_start:
{
lean_object* v_res_265_; 
v_res_265_ = lp_mathlib_RingTheory_LinearMap___aux__Mathlib__Algebra__Algebra__Bilinear______unexpand__LinearMap__mul_x27__2(v_x_262_, v_a_263_, v_a_264_);
lean_dec(v_a_263_);
return v_res_265_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalAlgHom_lmul___redArg(lean_object* v_inst_266_){
_start:
{
lean_object* v___x_267_; 
v___x_267_ = lp_mathlib_LinearMap_mul___redArg(v_inst_266_);
return v___x_267_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalAlgHom_lmul(lean_object* v_R_268_, lean_object* v_A_269_, lean_object* v_inst_270_, lean_object* v_inst_271_, lean_object* v_inst_272_, lean_object* v_inst_273_, lean_object* v_inst_274_){
_start:
{
lean_object* v___x_275_; 
v___x_275_ = lp_mathlib_LinearMap_mul___redArg(v_inst_271_);
return v___x_275_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalAlgHom_lmul___boxed(lean_object* v_R_276_, lean_object* v_A_277_, lean_object* v_inst_278_, lean_object* v_inst_279_, lean_object* v_inst_280_, lean_object* v_inst_281_, lean_object* v_inst_282_){
_start:
{
lean_object* v_res_283_; 
v_res_283_ = lp_mathlib_NonUnitalAlgHom_lmul(v_R_276_, v_A_277_, v_inst_278_, v_inst_279_, v_inst_280_, v_inst_281_, v_inst_282_);
lean_dec(v_inst_280_);
lean_dec_ref(v_inst_278_);
return v_res_283_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Algebra_lmul___redArg(lean_object* v_inst_284_){
_start:
{
lean_object* v___x_285_; lean_object* v___x_286_; 
v___x_285_ = lp_mathlib_Semiring_toNonUnitalSemiring___redArg(v_inst_284_);
v___x_286_ = lp_mathlib_LinearMap_mul___redArg(v___x_285_);
return v___x_286_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Algebra_lmul___redArg___boxed(lean_object* v_inst_287_){
_start:
{
lean_object* v_res_288_; 
v_res_288_ = lp_mathlib_Algebra_lmul___redArg(v_inst_287_);
lean_dec_ref(v_inst_287_);
return v_res_288_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Algebra_lmul(lean_object* v_R_289_, lean_object* v_A_290_, lean_object* v_inst_291_, lean_object* v_inst_292_, lean_object* v_inst_293_){
_start:
{
lean_object* v___x_294_; 
v___x_294_ = lp_mathlib_Algebra_lmul___redArg(v_inst_292_);
return v___x_294_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Algebra_lmul___boxed(lean_object* v_R_295_, lean_object* v_A_296_, lean_object* v_inst_297_, lean_object* v_inst_298_, lean_object* v_inst_299_){
_start:
{
lean_object* v_res_300_; 
v_res_300_ = lp_mathlib_Algebra_lmul(v_R_295_, v_A_296_, v_inst_297_, v_inst_298_, v_inst_299_);
lean_dec_ref(v_inst_299_);
lean_dec_ref(v_inst_298_);
lean_dec_ref(v_inst_297_);
return v_res_300_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LinearMap_mul_x27_x27___redArg(lean_object* v_inst_301_, lean_object* v_inst_302_, lean_object* v_inst_303_){
_start:
{
lean_object* v___x_304_; lean_object* v_toNonUnitalNonAssocSemiring_305_; lean_object* v_toSMul_306_; lean_object* v___x_307_; 
v___x_304_ = lp_mathlib_Semiring_toNonAssocSemiring___redArg(v_inst_302_);
v_toNonUnitalNonAssocSemiring_305_ = lean_ctor_get(v___x_304_, 0);
lean_inc_ref(v_toNonUnitalNonAssocSemiring_305_);
lean_dec_ref(v___x_304_);
v_toSMul_306_ = lean_ctor_get(v_inst_303_, 0);
lean_inc(v_toSMul_306_);
lean_dec_ref(v_inst_303_);
v___x_307_ = lp_mathlib_LinearMap_mul_x27___redArg(v_inst_301_, v_toNonUnitalNonAssocSemiring_305_, v_toSMul_306_);
return v___x_307_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LinearMap_mul_x27_x27(lean_object* v_R_308_, lean_object* v_A_309_, lean_object* v_inst_310_, lean_object* v_inst_311_, lean_object* v_inst_312_){
_start:
{
lean_object* v___x_313_; 
v___x_313_ = lp_mathlib_LinearMap_mul_x27_x27___redArg(v_inst_310_, v_inst_311_, v_inst_312_);
return v___x_313_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Algebra_NonUnitalHom(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_LinearAlgebra_TensorProduct_Map(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Algebra_Bilinear(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Algebra_NonUnitalHom(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_LinearAlgebra_TensorProduct_Map(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_Algebra_Algebra_Bilinear(uint8_t builtin) {
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
lean_object* initialize_mathlib_Mathlib_Algebra_Algebra_NonUnitalHom(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_LinearAlgebra_TensorProduct_Map(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Algebra_Algebra_Bilinear(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Algebra_NonUnitalHom(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_LinearAlgebra_TensorProduct_Map(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Algebra_Bilinear(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_Algebra_Algebra_Bilinear(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_Algebra_Algebra_Bilinear(builtin);
}
#ifdef __cplusplus
}
#endif
