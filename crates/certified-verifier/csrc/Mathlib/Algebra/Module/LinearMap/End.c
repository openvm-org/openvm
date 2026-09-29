// Lean compiler output
// Module: Mathlib.Algebra.Module.LinearMap.End
// Imports: public import Init public meta import Init public import Mathlib.Algebra.Group.Center public import Mathlib.Algebra.Module.Equiv.Opposite public import Mathlib.Algebra.Module.Torsion.Free
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
lean_object* lean_mk_empty_array_with_capacity(lean_object*);
lean_object* lean_array_push(lean_object*, lean_object*);
lean_object* l_Lean_Name_mkStr4(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_mkAtom(lean_object*);
lean_object* lp_mathlib_LinearMap_id___lam__0(lean_object*);
lean_object* lp_mathlib_LinearMap_addMonoid___redArg(lean_object*);
lean_object* lp_mathlib_LinearMap_id___lam__0___boxed(lean_object*);
lean_object* lp_mathlib_LinearMap_comp___redArg___lam__0(lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_npowBinRecAuto___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_MulOpposite_instSemiring___redArg(lean_object*);
lean_object* lp_mathlib_Semiring_toNonAssocSemiring___redArg(lean_object*);
lean_object* lp_mathlib_NonAssocSemiring_toAddCommMonoidWithOne___redArg(lean_object*);
lean_object* lp_mathlib_Semiring_toOppositeModule___redArg(lean_object*);
lean_object* lp_mathlib_instDistribSMul___redArg(lean_object*);
lean_object* lp_mathlib_LinearMap_addCommGroup___redArg(lean_object*);
lean_object* lp_mathlib_Semiring_toModule___redArg(lean_object*);
static const lean_closure_object lp_mathlib_Module_End_instOne___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_LinearMap_id___lam__0___boxed, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Module_End_instOne___closed__0 = (const lean_object*)&lp_mathlib_Module_End_instOne___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_Module_End_instOne(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Module_End_instOne___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Module_End_instMul___lam__0(lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_Module_End_instMul___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Module_End_instMul___lam__0, .m_arity = 3, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Module_End_instMul___closed__0 = (const lean_object*)&lp_mathlib_Module_End_instMul___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_Module_End_instMul(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Module_End_instMul___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_Module_End_instMonoid___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*3, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_npowBinRecAuto___boxed, .m_arity = 5, .m_num_fixed = 3, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Module_End_instMul___closed__0_value),((lean_object*)&lp_mathlib_Module_End_instOne___closed__0_value)} };
static const lean_object* lp_mathlib_Module_End_instMonoid___closed__0 = (const lean_object*)&lp_mathlib_Module_End_instMonoid___closed__0_value;
static const lean_ctor_object lp_mathlib_Module_End_instMonoid___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_Module_End_instOne___closed__0_value),((lean_object*)&lp_mathlib_Module_End_instMul___closed__0_value),((lean_object*)&lp_mathlib_Module_End_instMonoid___closed__0_value)}};
static const lean_object* lp_mathlib_Module_End_instMonoid___closed__1 = (const lean_object*)&lp_mathlib_Module_End_instMonoid___closed__1_value;
LEAN_EXPORT lean_object* lp_mathlib_Module_End_instMonoid(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Module_End_instMonoid___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Module_End_instSemiring___redArg___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Module_End_instSemiring___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Module_End_instSemiring___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Module_End_instSemiring(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Module_End_instSemiring___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Module_End_instRing___redArg___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Module_End_instRing___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Module_End_instRing___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Module_End_instRing(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Module_End_instRing___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Module_End_smulLeft___redArg___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Module_End_smulLeft___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Module_End_smulLeft(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Module_End_smulLeft___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Module_End_smulLeft__eq___auto__1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "Lean"};
static const lean_object* lp_mathlib_Module_End_smulLeft__eq___auto__1___closed__0 = (const lean_object*)&lp_mathlib_Module_End_smulLeft__eq___auto__1___closed__0_value;
static const lean_string_object lp_mathlib_Module_End_smulLeft__eq___auto__1___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "Parser"};
static const lean_object* lp_mathlib_Module_End_smulLeft__eq___auto__1___closed__1 = (const lean_object*)&lp_mathlib_Module_End_smulLeft__eq___auto__1___closed__1_value;
static const lean_string_object lp_mathlib_Module_End_smulLeft__eq___auto__1___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "Tactic"};
static const lean_object* lp_mathlib_Module_End_smulLeft__eq___auto__1___closed__2 = (const lean_object*)&lp_mathlib_Module_End_smulLeft__eq___auto__1___closed__2_value;
static const lean_string_object lp_mathlib_Module_End_smulLeft__eq___auto__1___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 9, .m_data = "tacticSeq"};
static const lean_object* lp_mathlib_Module_End_smulLeft__eq___auto__1___closed__3 = (const lean_object*)&lp_mathlib_Module_End_smulLeft__eq___auto__1___closed__3_value;
static const lean_ctor_object lp_mathlib_Module_End_smulLeft__eq___auto__1___closed__4_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Module_End_smulLeft__eq___auto__1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib_Module_End_smulLeft__eq___auto__1___closed__4_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Module_End_smulLeft__eq___auto__1___closed__4_value_aux_0),((lean_object*)&lp_mathlib_Module_End_smulLeft__eq___auto__1___closed__1_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib_Module_End_smulLeft__eq___auto__1___closed__4_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Module_End_smulLeft__eq___auto__1___closed__4_value_aux_1),((lean_object*)&lp_mathlib_Module_End_smulLeft__eq___auto__1___closed__2_value),LEAN_SCALAR_PTR_LITERAL(166, 58, 35, 182, 187, 130, 147, 254)}};
static const lean_ctor_object lp_mathlib_Module_End_smulLeft__eq___auto__1___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Module_End_smulLeft__eq___auto__1___closed__4_value_aux_2),((lean_object*)&lp_mathlib_Module_End_smulLeft__eq___auto__1___closed__3_value),LEAN_SCALAR_PTR_LITERAL(212, 140, 85, 215, 241, 69, 7, 118)}};
static const lean_object* lp_mathlib_Module_End_smulLeft__eq___auto__1___closed__4 = (const lean_object*)&lp_mathlib_Module_End_smulLeft__eq___auto__1___closed__4_value;
static const lean_array_object lp_mathlib_Module_End_smulLeft__eq___auto__1___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_array_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 246}, .m_size = 0, .m_capacity = 0, .m_data = {}};
static const lean_object* lp_mathlib_Module_End_smulLeft__eq___auto__1___closed__5 = (const lean_object*)&lp_mathlib_Module_End_smulLeft__eq___auto__1___closed__5_value;
static const lean_string_object lp_mathlib_Module_End_smulLeft__eq___auto__1___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 19, .m_capacity = 19, .m_length = 18, .m_data = "tacticSeq1Indented"};
static const lean_object* lp_mathlib_Module_End_smulLeft__eq___auto__1___closed__6 = (const lean_object*)&lp_mathlib_Module_End_smulLeft__eq___auto__1___closed__6_value;
static const lean_ctor_object lp_mathlib_Module_End_smulLeft__eq___auto__1___closed__7_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Module_End_smulLeft__eq___auto__1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib_Module_End_smulLeft__eq___auto__1___closed__7_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Module_End_smulLeft__eq___auto__1___closed__7_value_aux_0),((lean_object*)&lp_mathlib_Module_End_smulLeft__eq___auto__1___closed__1_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib_Module_End_smulLeft__eq___auto__1___closed__7_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Module_End_smulLeft__eq___auto__1___closed__7_value_aux_1),((lean_object*)&lp_mathlib_Module_End_smulLeft__eq___auto__1___closed__2_value),LEAN_SCALAR_PTR_LITERAL(166, 58, 35, 182, 187, 130, 147, 254)}};
static const lean_ctor_object lp_mathlib_Module_End_smulLeft__eq___auto__1___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Module_End_smulLeft__eq___auto__1___closed__7_value_aux_2),((lean_object*)&lp_mathlib_Module_End_smulLeft__eq___auto__1___closed__6_value),LEAN_SCALAR_PTR_LITERAL(223, 90, 160, 238, 133, 180, 23, 239)}};
static const lean_object* lp_mathlib_Module_End_smulLeft__eq___auto__1___closed__7 = (const lean_object*)&lp_mathlib_Module_End_smulLeft__eq___auto__1___closed__7_value;
static const lean_string_object lp_mathlib_Module_End_smulLeft__eq___auto__1___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "null"};
static const lean_object* lp_mathlib_Module_End_smulLeft__eq___auto__1___closed__8 = (const lean_object*)&lp_mathlib_Module_End_smulLeft__eq___auto__1___closed__8_value;
static const lean_ctor_object lp_mathlib_Module_End_smulLeft__eq___auto__1___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Module_End_smulLeft__eq___auto__1___closed__8_value),LEAN_SCALAR_PTR_LITERAL(24, 58, 49, 223, 146, 207, 197, 136)}};
static const lean_object* lp_mathlib_Module_End_smulLeft__eq___auto__1___closed__9 = (const lean_object*)&lp_mathlib_Module_End_smulLeft__eq___auto__1___closed__9_value;
static const lean_string_object lp_mathlib_Module_End_smulLeft__eq___auto__1___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "simp"};
static const lean_object* lp_mathlib_Module_End_smulLeft__eq___auto__1___closed__10 = (const lean_object*)&lp_mathlib_Module_End_smulLeft__eq___auto__1___closed__10_value;
static const lean_ctor_object lp_mathlib_Module_End_smulLeft__eq___auto__1___closed__11_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Module_End_smulLeft__eq___auto__1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib_Module_End_smulLeft__eq___auto__1___closed__11_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Module_End_smulLeft__eq___auto__1___closed__11_value_aux_0),((lean_object*)&lp_mathlib_Module_End_smulLeft__eq___auto__1___closed__1_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib_Module_End_smulLeft__eq___auto__1___closed__11_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Module_End_smulLeft__eq___auto__1___closed__11_value_aux_1),((lean_object*)&lp_mathlib_Module_End_smulLeft__eq___auto__1___closed__2_value),LEAN_SCALAR_PTR_LITERAL(166, 58, 35, 182, 187, 130, 147, 254)}};
static const lean_ctor_object lp_mathlib_Module_End_smulLeft__eq___auto__1___closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Module_End_smulLeft__eq___auto__1___closed__11_value_aux_2),((lean_object*)&lp_mathlib_Module_End_smulLeft__eq___auto__1___closed__10_value),LEAN_SCALAR_PTR_LITERAL(50, 13, 241, 145, 67, 153, 105, 177)}};
static const lean_object* lp_mathlib_Module_End_smulLeft__eq___auto__1___closed__11 = (const lean_object*)&lp_mathlib_Module_End_smulLeft__eq___auto__1___closed__11_value;
static lean_once_cell_t lp_mathlib_Module_End_smulLeft__eq___auto__1___closed__12_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Module_End_smulLeft__eq___auto__1___closed__12;
static lean_once_cell_t lp_mathlib_Module_End_smulLeft__eq___auto__1___closed__13_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Module_End_smulLeft__eq___auto__1___closed__13;
static const lean_string_object lp_mathlib_Module_End_smulLeft__eq___auto__1___closed__14_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 9, .m_data = "optConfig"};
static const lean_object* lp_mathlib_Module_End_smulLeft__eq___auto__1___closed__14 = (const lean_object*)&lp_mathlib_Module_End_smulLeft__eq___auto__1___closed__14_value;
static const lean_ctor_object lp_mathlib_Module_End_smulLeft__eq___auto__1___closed__15_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Module_End_smulLeft__eq___auto__1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib_Module_End_smulLeft__eq___auto__1___closed__15_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Module_End_smulLeft__eq___auto__1___closed__15_value_aux_0),((lean_object*)&lp_mathlib_Module_End_smulLeft__eq___auto__1___closed__1_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib_Module_End_smulLeft__eq___auto__1___closed__15_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Module_End_smulLeft__eq___auto__1___closed__15_value_aux_1),((lean_object*)&lp_mathlib_Module_End_smulLeft__eq___auto__1___closed__2_value),LEAN_SCALAR_PTR_LITERAL(166, 58, 35, 182, 187, 130, 147, 254)}};
static const lean_ctor_object lp_mathlib_Module_End_smulLeft__eq___auto__1___closed__15_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Module_End_smulLeft__eq___auto__1___closed__15_value_aux_2),((lean_object*)&lp_mathlib_Module_End_smulLeft__eq___auto__1___closed__14_value),LEAN_SCALAR_PTR_LITERAL(137, 208, 10, 74, 108, 50, 106, 48)}};
static const lean_object* lp_mathlib_Module_End_smulLeft__eq___auto__1___closed__15 = (const lean_object*)&lp_mathlib_Module_End_smulLeft__eq___auto__1___closed__15_value;
static const lean_ctor_object lp_mathlib_Module_End_smulLeft__eq___auto__1___closed__16_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(2) << 1) | 1)),((lean_object*)&lp_mathlib_Module_End_smulLeft__eq___auto__1___closed__9_value),((lean_object*)&lp_mathlib_Module_End_smulLeft__eq___auto__1___closed__5_value)}};
static const lean_object* lp_mathlib_Module_End_smulLeft__eq___auto__1___closed__16 = (const lean_object*)&lp_mathlib_Module_End_smulLeft__eq___auto__1___closed__16_value;
static lean_once_cell_t lp_mathlib_Module_End_smulLeft__eq___auto__1___closed__17_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Module_End_smulLeft__eq___auto__1___closed__17;
static lean_once_cell_t lp_mathlib_Module_End_smulLeft__eq___auto__1___closed__18_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Module_End_smulLeft__eq___auto__1___closed__18;
static lean_once_cell_t lp_mathlib_Module_End_smulLeft__eq___auto__1___closed__19_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Module_End_smulLeft__eq___auto__1___closed__19;
static lean_once_cell_t lp_mathlib_Module_End_smulLeft__eq___auto__1___closed__20_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Module_End_smulLeft__eq___auto__1___closed__20;
static lean_once_cell_t lp_mathlib_Module_End_smulLeft__eq___auto__1___closed__21_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Module_End_smulLeft__eq___auto__1___closed__21;
static lean_once_cell_t lp_mathlib_Module_End_smulLeft__eq___auto__1___closed__22_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Module_End_smulLeft__eq___auto__1___closed__22;
static lean_once_cell_t lp_mathlib_Module_End_smulLeft__eq___auto__1___closed__23_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Module_End_smulLeft__eq___auto__1___closed__23;
static lean_once_cell_t lp_mathlib_Module_End_smulLeft__eq___auto__1___closed__24_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Module_End_smulLeft__eq___auto__1___closed__24;
static lean_once_cell_t lp_mathlib_Module_End_smulLeft__eq___auto__1___closed__25_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Module_End_smulLeft__eq___auto__1___closed__25;
static lean_once_cell_t lp_mathlib_Module_End_smulLeft__eq___auto__1___closed__26_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Module_End_smulLeft__eq___auto__1___closed__26;
static lean_once_cell_t lp_mathlib_Module_End_smulLeft__eq___auto__1___closed__27_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Module_End_smulLeft__eq___auto__1___closed__27;
static lean_once_cell_t lp_mathlib_Module_End_smulLeft__eq___auto__1___closed__28_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Module_End_smulLeft__eq___auto__1___closed__28;
static lean_once_cell_t lp_mathlib_Module_End_smulLeft__eq___auto__1___closed__29_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Module_End_smulLeft__eq___auto__1___closed__29;
static lean_once_cell_t lp_mathlib_Module_End_smulLeft__eq___auto__1___closed__30_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Module_End_smulLeft__eq___auto__1___closed__30;
LEAN_EXPORT lean_object* lp_mathlib_Module_End_smulLeft__eq___auto__1;
LEAN_EXPORT lean_object* lp_mathlib_Module_End_applyModule___lam__0(lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_Module_End_applyModule___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Module_End_applyModule___lam__0, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Module_End_applyModule___closed__0 = (const lean_object*)&lp_mathlib_Module_End_applyModule___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_Module_End_applyModule(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Module_End_applyModule___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DistribSMul_toLinearMap___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DistribSMul_toLinearMap(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DistribSMul_toLinearMap___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DistribMulAction_toModuleEnd___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DistribMulAction_toModuleEnd(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DistribMulAction_toModuleEnd___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Module_toModuleEnd___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Module_toModuleEnd(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Module_toModuleEnd___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RingEquiv_moduleEndSelf___redArg___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RingEquiv_moduleEndSelf___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RingEquiv_moduleEndSelf(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RingEquiv_moduleEndSelfOp___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RingEquiv_moduleEndSelfOp(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LinearMap_smulRight___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LinearMap_smulRight___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LinearMap_smulRight(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LinearMap_smulRight___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LinearMap_apply_u2097_x27___lam__0(lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_LinearMap_apply_u2097_x27___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_LinearMap_apply_u2097_x27___lam__0, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_LinearMap_apply_u2097_x27___closed__0 = (const lean_object*)&lp_mathlib_LinearMap_apply_u2097_x27___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_LinearMap_apply_u2097_x27(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LinearMap_apply_u2097_x27___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LinearMap_compRight___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LinearMap_compRight(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LinearMap_compRight___boxed(lean_object**);
LEAN_EXPORT lean_object* lp_mathlib_LinearMap_apply_u2097(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LinearMap_apply_u2097___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LinearMap_smulRight_u2097___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LinearMap_smulRight_u2097___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LinearMap_smulRight_u2097(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LinearMap_smulRight_u2097___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Module_End_instOne(lean_object* v_R_2_, lean_object* v_M_3_, lean_object* v_inst_4_, lean_object* v_inst_5_, lean_object* v_inst_6_){
_start:
{
lean_object* v___f_7_; 
v___f_7_ = ((lean_object*)(lp_mathlib_Module_End_instOne___closed__0));
return v___f_7_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Module_End_instOne___boxed(lean_object* v_R_8_, lean_object* v_M_9_, lean_object* v_inst_10_, lean_object* v_inst_11_, lean_object* v_inst_12_){
_start:
{
lean_object* v_res_13_; 
v_res_13_ = lp_mathlib_Module_End_instOne(v_R_8_, v_M_9_, v_inst_10_, v_inst_11_, v_inst_12_);
lean_dec(v_inst_12_);
lean_dec_ref(v_inst_11_);
lean_dec_ref(v_inst_10_);
return v_res_13_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Module_End_instMul___lam__0(lean_object* v_f_14_, lean_object* v_g_15_, lean_object* v___y_16_){
_start:
{
lean_object* v___x_17_; 
v___x_17_ = lp_mathlib_LinearMap_comp___redArg___lam__0(v_g_15_, v_f_14_, v___y_16_);
return v___x_17_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Module_End_instMul(lean_object* v_R_19_, lean_object* v_M_20_, lean_object* v_inst_21_, lean_object* v_inst_22_, lean_object* v_inst_23_){
_start:
{
lean_object* v___f_24_; 
v___f_24_ = ((lean_object*)(lp_mathlib_Module_End_instMul___closed__0));
return v___f_24_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Module_End_instMul___boxed(lean_object* v_R_25_, lean_object* v_M_26_, lean_object* v_inst_27_, lean_object* v_inst_28_, lean_object* v_inst_29_){
_start:
{
lean_object* v_res_30_; 
v_res_30_ = lp_mathlib_Module_End_instMul(v_R_25_, v_M_26_, v_inst_27_, v_inst_28_, v_inst_29_);
lean_dec(v_inst_29_);
lean_dec_ref(v_inst_28_);
lean_dec_ref(v_inst_27_);
return v_res_30_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Module_End_instMonoid(lean_object* v_R_38_, lean_object* v_M_39_, lean_object* v_inst_40_, lean_object* v_inst_41_, lean_object* v_inst_42_){
_start:
{
lean_object* v___x_43_; 
v___x_43_ = ((lean_object*)(lp_mathlib_Module_End_instMonoid___closed__1));
return v___x_43_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Module_End_instMonoid___boxed(lean_object* v_R_44_, lean_object* v_M_45_, lean_object* v_inst_46_, lean_object* v_inst_47_, lean_object* v_inst_48_){
_start:
{
lean_object* v_res_49_; 
v_res_49_ = lp_mathlib_Module_End_instMonoid(v_R_44_, v_M_45_, v_inst_46_, v_inst_47_, v_inst_48_);
lean_dec(v_inst_48_);
lean_dec_ref(v_inst_47_);
lean_dec_ref(v_inst_46_);
return v_res_49_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Module_End_instSemiring___redArg___lam__0(lean_object* v_inst_50_, lean_object* v_n_51_, lean_object* v___y_52_){
_start:
{
lean_object* v_toNSMul_53_; lean_object* v___x_54_; lean_object* v___x_55_; 
v_toNSMul_53_ = lean_ctor_get(v_inst_50_, 2);
lean_inc(v_toNSMul_53_);
lean_dec_ref(v_inst_50_);
v___x_54_ = lp_mathlib_LinearMap_id___lam__0(v___y_52_);
v___x_55_ = lean_apply_2(v_toNSMul_53_, v_n_51_, v___x_54_);
return v___x_55_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Module_End_instSemiring___redArg___lam__0___boxed(lean_object* v_inst_56_, lean_object* v_n_57_, lean_object* v___y_58_){
_start:
{
lean_object* v_res_59_; 
v_res_59_ = lp_mathlib_Module_End_instSemiring___redArg___lam__0(v_inst_56_, v_n_57_, v___y_58_);
lean_dec(v___y_58_);
return v_res_59_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Module_End_instSemiring___redArg(lean_object* v_inst_60_){
_start:
{
lean_object* v___f_61_; lean_object* v___x_62_; lean_object* v___x_63_; lean_object* v___x_64_; 
lean_inc_ref(v_inst_60_);
v___f_61_ = lean_alloc_closure((void*)(lp_mathlib_Module_End_instSemiring___redArg___lam__0___boxed), 3, 1);
lean_closure_set(v___f_61_, 0, v_inst_60_);
v___x_62_ = lp_mathlib_LinearMap_addMonoid___redArg(v_inst_60_);
v___x_63_ = ((lean_object*)(lp_mathlib_Module_End_instMonoid___closed__1));
v___x_64_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_64_, 0, v___x_62_);
lean_ctor_set(v___x_64_, 1, v___x_63_);
lean_ctor_set(v___x_64_, 2, v___f_61_);
return v___x_64_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Module_End_instSemiring(lean_object* v_R_65_, lean_object* v_M_66_, lean_object* v_inst_67_, lean_object* v_inst_68_, lean_object* v_inst_69_){
_start:
{
lean_object* v___x_70_; 
v___x_70_ = lp_mathlib_Module_End_instSemiring___redArg(v_inst_68_);
return v___x_70_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Module_End_instSemiring___boxed(lean_object* v_R_71_, lean_object* v_M_72_, lean_object* v_inst_73_, lean_object* v_inst_74_, lean_object* v_inst_75_){
_start:
{
lean_object* v_res_76_; 
v_res_76_ = lp_mathlib_Module_End_instSemiring(v_R_71_, v_M_72_, v_inst_73_, v_inst_74_, v_inst_75_);
lean_dec(v_inst_75_);
lean_dec_ref(v_inst_73_);
return v_res_76_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Module_End_instRing___redArg___lam__0(lean_object* v_toZSMul_77_, lean_object* v_z_78_, lean_object* v___y_79_){
_start:
{
lean_object* v___x_80_; lean_object* v___x_81_; 
v___x_80_ = lp_mathlib_LinearMap_id___lam__0(v___y_79_);
v___x_81_ = lean_apply_2(v_toZSMul_77_, v_z_78_, v___x_80_);
return v___x_81_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Module_End_instRing___redArg___lam__0___boxed(lean_object* v_toZSMul_82_, lean_object* v_z_83_, lean_object* v___y_84_){
_start:
{
lean_object* v_res_85_; 
v_res_85_ = lp_mathlib_Module_End_instRing___redArg___lam__0(v_toZSMul_82_, v_z_83_, v___y_84_);
lean_dec(v___y_84_);
return v_res_85_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Module_End_instRing___redArg(lean_object* v_inst_86_){
_start:
{
lean_object* v_toAddMonoid_87_; lean_object* v_toZSMul_88_; lean_object* v___x_89_; lean_object* v___x_90_; lean_object* v_toNeg_91_; lean_object* v_toSub_92_; lean_object* v_toZSMul_93_; lean_object* v___f_94_; lean_object* v___x_95_; 
v_toAddMonoid_87_ = lean_ctor_get(v_inst_86_, 0);
v_toZSMul_88_ = lean_ctor_get(v_inst_86_, 3);
lean_inc(v_toZSMul_88_);
lean_inc_ref(v_toAddMonoid_87_);
v___x_89_ = lp_mathlib_Module_End_instSemiring___redArg(v_toAddMonoid_87_);
v___x_90_ = lp_mathlib_LinearMap_addCommGroup___redArg(v_inst_86_);
v_toNeg_91_ = lean_ctor_get(v___x_90_, 1);
lean_inc(v_toNeg_91_);
v_toSub_92_ = lean_ctor_get(v___x_90_, 2);
lean_inc(v_toSub_92_);
v_toZSMul_93_ = lean_ctor_get(v___x_90_, 3);
lean_inc(v_toZSMul_93_);
lean_dec_ref(v___x_90_);
v___f_94_ = lean_alloc_closure((void*)(lp_mathlib_Module_End_instRing___redArg___lam__0___boxed), 3, 1);
lean_closure_set(v___f_94_, 0, v_toZSMul_88_);
v___x_95_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v___x_95_, 0, v___x_89_);
lean_ctor_set(v___x_95_, 1, v_toNeg_91_);
lean_ctor_set(v___x_95_, 2, v_toSub_92_);
lean_ctor_set(v___x_95_, 3, v_toZSMul_93_);
lean_ctor_set(v___x_95_, 4, v___f_94_);
return v___x_95_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Module_End_instRing(lean_object* v_R_96_, lean_object* v_N_u2081_97_, lean_object* v_inst_98_, lean_object* v_inst_99_, lean_object* v_inst_100_){
_start:
{
lean_object* v___x_101_; 
v___x_101_ = lp_mathlib_Module_End_instRing___redArg(v_inst_99_);
return v___x_101_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Module_End_instRing___boxed(lean_object* v_R_102_, lean_object* v_N_u2081_103_, lean_object* v_inst_104_, lean_object* v_inst_105_, lean_object* v_inst_106_){
_start:
{
lean_object* v_res_107_; 
v_res_107_ = lp_mathlib_Module_End_instRing(v_R_102_, v_N_u2081_103_, v_inst_104_, v_inst_105_, v_inst_106_);
lean_dec(v_inst_106_);
lean_dec_ref(v_inst_104_);
return v_res_107_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Module_End_smulLeft___redArg___lam__0(lean_object* v_inst_108_, lean_object* v_00_u03b1_109_, lean_object* v_x_110_){
_start:
{
lean_object* v___x_111_; 
v___x_111_ = lean_apply_2(v_inst_108_, v_00_u03b1_109_, v_x_110_);
return v___x_111_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Module_End_smulLeft___redArg(lean_object* v_inst_112_, lean_object* v_00_u03b1_113_){
_start:
{
lean_object* v___f_114_; 
v___f_114_ = lean_alloc_closure((void*)(lp_mathlib_Module_End_smulLeft___redArg___lam__0), 3, 2);
lean_closure_set(v___f_114_, 0, v_inst_112_);
lean_closure_set(v___f_114_, 1, v_00_u03b1_113_);
return v___f_114_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Module_End_smulLeft(lean_object* v_R_115_, lean_object* v_M_116_, lean_object* v_inst_117_, lean_object* v_inst_118_, lean_object* v_inst_119_, lean_object* v_00_u03b1_120_, lean_object* v_h_u03b1_121_){
_start:
{
lean_object* v___f_122_; 
v___f_122_ = lean_alloc_closure((void*)(lp_mathlib_Module_End_smulLeft___redArg___lam__0), 3, 2);
lean_closure_set(v___f_122_, 0, v_inst_119_);
lean_closure_set(v___f_122_, 1, v_00_u03b1_120_);
return v___f_122_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Module_End_smulLeft___boxed(lean_object* v_R_123_, lean_object* v_M_124_, lean_object* v_inst_125_, lean_object* v_inst_126_, lean_object* v_inst_127_, lean_object* v_00_u03b1_128_, lean_object* v_h_u03b1_129_){
_start:
{
lean_object* v_res_130_; 
v_res_130_ = lp_mathlib_Module_End_smulLeft(v_R_123_, v_M_124_, v_inst_125_, v_inst_126_, v_inst_127_, v_00_u03b1_128_, v_h_u03b1_129_);
lean_dec_ref(v_inst_126_);
lean_dec_ref(v_inst_125_);
return v_res_130_;
}
}
static lean_object* _init_lp_mathlib_Module_End_smulLeft__eq___auto__1___closed__12(void){
_start:
{
lean_object* v___x_157_; lean_object* v___x_158_; 
v___x_157_ = ((lean_object*)(lp_mathlib_Module_End_smulLeft__eq___auto__1___closed__10));
v___x_158_ = l_Lean_mkAtom(v___x_157_);
return v___x_158_;
}
}
static lean_object* _init_lp_mathlib_Module_End_smulLeft__eq___auto__1___closed__13(void){
_start:
{
lean_object* v___x_159_; lean_object* v___x_160_; lean_object* v___x_161_; 
v___x_159_ = lean_obj_once(&lp_mathlib_Module_End_smulLeft__eq___auto__1___closed__12, &lp_mathlib_Module_End_smulLeft__eq___auto__1___closed__12_once, _init_lp_mathlib_Module_End_smulLeft__eq___auto__1___closed__12);
v___x_160_ = ((lean_object*)(lp_mathlib_Module_End_smulLeft__eq___auto__1___closed__5));
v___x_161_ = lean_array_push(v___x_160_, v___x_159_);
return v___x_161_;
}
}
static lean_object* _init_lp_mathlib_Module_End_smulLeft__eq___auto__1___closed__17(void){
_start:
{
lean_object* v___x_172_; lean_object* v___x_173_; lean_object* v___x_174_; 
v___x_172_ = ((lean_object*)(lp_mathlib_Module_End_smulLeft__eq___auto__1___closed__16));
v___x_173_ = ((lean_object*)(lp_mathlib_Module_End_smulLeft__eq___auto__1___closed__5));
v___x_174_ = lean_array_push(v___x_173_, v___x_172_);
return v___x_174_;
}
}
static lean_object* _init_lp_mathlib_Module_End_smulLeft__eq___auto__1___closed__18(void){
_start:
{
lean_object* v___x_175_; lean_object* v___x_176_; lean_object* v___x_177_; lean_object* v___x_178_; 
v___x_175_ = lean_obj_once(&lp_mathlib_Module_End_smulLeft__eq___auto__1___closed__17, &lp_mathlib_Module_End_smulLeft__eq___auto__1___closed__17_once, _init_lp_mathlib_Module_End_smulLeft__eq___auto__1___closed__17);
v___x_176_ = ((lean_object*)(lp_mathlib_Module_End_smulLeft__eq___auto__1___closed__15));
v___x_177_ = lean_box(2);
v___x_178_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_178_, 0, v___x_177_);
lean_ctor_set(v___x_178_, 1, v___x_176_);
lean_ctor_set(v___x_178_, 2, v___x_175_);
return v___x_178_;
}
}
static lean_object* _init_lp_mathlib_Module_End_smulLeft__eq___auto__1___closed__19(void){
_start:
{
lean_object* v___x_179_; lean_object* v___x_180_; lean_object* v___x_181_; 
v___x_179_ = lean_obj_once(&lp_mathlib_Module_End_smulLeft__eq___auto__1___closed__18, &lp_mathlib_Module_End_smulLeft__eq___auto__1___closed__18_once, _init_lp_mathlib_Module_End_smulLeft__eq___auto__1___closed__18);
v___x_180_ = lean_obj_once(&lp_mathlib_Module_End_smulLeft__eq___auto__1___closed__13, &lp_mathlib_Module_End_smulLeft__eq___auto__1___closed__13_once, _init_lp_mathlib_Module_End_smulLeft__eq___auto__1___closed__13);
v___x_181_ = lean_array_push(v___x_180_, v___x_179_);
return v___x_181_;
}
}
static lean_object* _init_lp_mathlib_Module_End_smulLeft__eq___auto__1___closed__20(void){
_start:
{
lean_object* v___x_182_; lean_object* v___x_183_; lean_object* v___x_184_; 
v___x_182_ = ((lean_object*)(lp_mathlib_Module_End_smulLeft__eq___auto__1___closed__16));
v___x_183_ = lean_obj_once(&lp_mathlib_Module_End_smulLeft__eq___auto__1___closed__19, &lp_mathlib_Module_End_smulLeft__eq___auto__1___closed__19_once, _init_lp_mathlib_Module_End_smulLeft__eq___auto__1___closed__19);
v___x_184_ = lean_array_push(v___x_183_, v___x_182_);
return v___x_184_;
}
}
static lean_object* _init_lp_mathlib_Module_End_smulLeft__eq___auto__1___closed__21(void){
_start:
{
lean_object* v___x_185_; lean_object* v___x_186_; lean_object* v___x_187_; 
v___x_185_ = ((lean_object*)(lp_mathlib_Module_End_smulLeft__eq___auto__1___closed__16));
v___x_186_ = lean_obj_once(&lp_mathlib_Module_End_smulLeft__eq___auto__1___closed__20, &lp_mathlib_Module_End_smulLeft__eq___auto__1___closed__20_once, _init_lp_mathlib_Module_End_smulLeft__eq___auto__1___closed__20);
v___x_187_ = lean_array_push(v___x_186_, v___x_185_);
return v___x_187_;
}
}
static lean_object* _init_lp_mathlib_Module_End_smulLeft__eq___auto__1___closed__22(void){
_start:
{
lean_object* v___x_188_; lean_object* v___x_189_; lean_object* v___x_190_; 
v___x_188_ = ((lean_object*)(lp_mathlib_Module_End_smulLeft__eq___auto__1___closed__16));
v___x_189_ = lean_obj_once(&lp_mathlib_Module_End_smulLeft__eq___auto__1___closed__21, &lp_mathlib_Module_End_smulLeft__eq___auto__1___closed__21_once, _init_lp_mathlib_Module_End_smulLeft__eq___auto__1___closed__21);
v___x_190_ = lean_array_push(v___x_189_, v___x_188_);
return v___x_190_;
}
}
static lean_object* _init_lp_mathlib_Module_End_smulLeft__eq___auto__1___closed__23(void){
_start:
{
lean_object* v___x_191_; lean_object* v___x_192_; lean_object* v___x_193_; 
v___x_191_ = ((lean_object*)(lp_mathlib_Module_End_smulLeft__eq___auto__1___closed__16));
v___x_192_ = lean_obj_once(&lp_mathlib_Module_End_smulLeft__eq___auto__1___closed__22, &lp_mathlib_Module_End_smulLeft__eq___auto__1___closed__22_once, _init_lp_mathlib_Module_End_smulLeft__eq___auto__1___closed__22);
v___x_193_ = lean_array_push(v___x_192_, v___x_191_);
return v___x_193_;
}
}
static lean_object* _init_lp_mathlib_Module_End_smulLeft__eq___auto__1___closed__24(void){
_start:
{
lean_object* v___x_194_; lean_object* v___x_195_; lean_object* v___x_196_; lean_object* v___x_197_; 
v___x_194_ = lean_obj_once(&lp_mathlib_Module_End_smulLeft__eq___auto__1___closed__23, &lp_mathlib_Module_End_smulLeft__eq___auto__1___closed__23_once, _init_lp_mathlib_Module_End_smulLeft__eq___auto__1___closed__23);
v___x_195_ = ((lean_object*)(lp_mathlib_Module_End_smulLeft__eq___auto__1___closed__11));
v___x_196_ = lean_box(2);
v___x_197_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_197_, 0, v___x_196_);
lean_ctor_set(v___x_197_, 1, v___x_195_);
lean_ctor_set(v___x_197_, 2, v___x_194_);
return v___x_197_;
}
}
static lean_object* _init_lp_mathlib_Module_End_smulLeft__eq___auto__1___closed__25(void){
_start:
{
lean_object* v___x_198_; lean_object* v___x_199_; lean_object* v___x_200_; 
v___x_198_ = lean_obj_once(&lp_mathlib_Module_End_smulLeft__eq___auto__1___closed__24, &lp_mathlib_Module_End_smulLeft__eq___auto__1___closed__24_once, _init_lp_mathlib_Module_End_smulLeft__eq___auto__1___closed__24);
v___x_199_ = ((lean_object*)(lp_mathlib_Module_End_smulLeft__eq___auto__1___closed__5));
v___x_200_ = lean_array_push(v___x_199_, v___x_198_);
return v___x_200_;
}
}
static lean_object* _init_lp_mathlib_Module_End_smulLeft__eq___auto__1___closed__26(void){
_start:
{
lean_object* v___x_201_; lean_object* v___x_202_; lean_object* v___x_203_; lean_object* v___x_204_; 
v___x_201_ = lean_obj_once(&lp_mathlib_Module_End_smulLeft__eq___auto__1___closed__25, &lp_mathlib_Module_End_smulLeft__eq___auto__1___closed__25_once, _init_lp_mathlib_Module_End_smulLeft__eq___auto__1___closed__25);
v___x_202_ = ((lean_object*)(lp_mathlib_Module_End_smulLeft__eq___auto__1___closed__9));
v___x_203_ = lean_box(2);
v___x_204_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_204_, 0, v___x_203_);
lean_ctor_set(v___x_204_, 1, v___x_202_);
lean_ctor_set(v___x_204_, 2, v___x_201_);
return v___x_204_;
}
}
static lean_object* _init_lp_mathlib_Module_End_smulLeft__eq___auto__1___closed__27(void){
_start:
{
lean_object* v___x_205_; lean_object* v___x_206_; lean_object* v___x_207_; 
v___x_205_ = lean_obj_once(&lp_mathlib_Module_End_smulLeft__eq___auto__1___closed__26, &lp_mathlib_Module_End_smulLeft__eq___auto__1___closed__26_once, _init_lp_mathlib_Module_End_smulLeft__eq___auto__1___closed__26);
v___x_206_ = ((lean_object*)(lp_mathlib_Module_End_smulLeft__eq___auto__1___closed__5));
v___x_207_ = lean_array_push(v___x_206_, v___x_205_);
return v___x_207_;
}
}
static lean_object* _init_lp_mathlib_Module_End_smulLeft__eq___auto__1___closed__28(void){
_start:
{
lean_object* v___x_208_; lean_object* v___x_209_; lean_object* v___x_210_; lean_object* v___x_211_; 
v___x_208_ = lean_obj_once(&lp_mathlib_Module_End_smulLeft__eq___auto__1___closed__27, &lp_mathlib_Module_End_smulLeft__eq___auto__1___closed__27_once, _init_lp_mathlib_Module_End_smulLeft__eq___auto__1___closed__27);
v___x_209_ = ((lean_object*)(lp_mathlib_Module_End_smulLeft__eq___auto__1___closed__7));
v___x_210_ = lean_box(2);
v___x_211_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_211_, 0, v___x_210_);
lean_ctor_set(v___x_211_, 1, v___x_209_);
lean_ctor_set(v___x_211_, 2, v___x_208_);
return v___x_211_;
}
}
static lean_object* _init_lp_mathlib_Module_End_smulLeft__eq___auto__1___closed__29(void){
_start:
{
lean_object* v___x_212_; lean_object* v___x_213_; lean_object* v___x_214_; 
v___x_212_ = lean_obj_once(&lp_mathlib_Module_End_smulLeft__eq___auto__1___closed__28, &lp_mathlib_Module_End_smulLeft__eq___auto__1___closed__28_once, _init_lp_mathlib_Module_End_smulLeft__eq___auto__1___closed__28);
v___x_213_ = ((lean_object*)(lp_mathlib_Module_End_smulLeft__eq___auto__1___closed__5));
v___x_214_ = lean_array_push(v___x_213_, v___x_212_);
return v___x_214_;
}
}
static lean_object* _init_lp_mathlib_Module_End_smulLeft__eq___auto__1___closed__30(void){
_start:
{
lean_object* v___x_215_; lean_object* v___x_216_; lean_object* v___x_217_; lean_object* v___x_218_; 
v___x_215_ = lean_obj_once(&lp_mathlib_Module_End_smulLeft__eq___auto__1___closed__29, &lp_mathlib_Module_End_smulLeft__eq___auto__1___closed__29_once, _init_lp_mathlib_Module_End_smulLeft__eq___auto__1___closed__29);
v___x_216_ = ((lean_object*)(lp_mathlib_Module_End_smulLeft__eq___auto__1___closed__4));
v___x_217_ = lean_box(2);
v___x_218_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_218_, 0, v___x_217_);
lean_ctor_set(v___x_218_, 1, v___x_216_);
lean_ctor_set(v___x_218_, 2, v___x_215_);
return v___x_218_;
}
}
static lean_object* _init_lp_mathlib_Module_End_smulLeft__eq___auto__1(void){
_start:
{
lean_object* v___x_219_; 
v___x_219_ = lean_obj_once(&lp_mathlib_Module_End_smulLeft__eq___auto__1___closed__30, &lp_mathlib_Module_End_smulLeft__eq___auto__1___closed__30_once, _init_lp_mathlib_Module_End_smulLeft__eq___auto__1___closed__30);
return v___x_219_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Module_End_applyModule___lam__0(lean_object* v_x1_220_, lean_object* v_x2_221_){
_start:
{
lean_object* v___x_222_; 
v___x_222_ = lean_apply_1(v_x1_220_, v_x2_221_);
return v___x_222_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Module_End_applyModule(lean_object* v_R_224_, lean_object* v_M_225_, lean_object* v_inst_226_, lean_object* v_inst_227_, lean_object* v_inst_228_){
_start:
{
lean_object* v___f_229_; 
v___f_229_ = ((lean_object*)(lp_mathlib_Module_End_applyModule___closed__0));
return v___f_229_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Module_End_applyModule___boxed(lean_object* v_R_230_, lean_object* v_M_231_, lean_object* v_inst_232_, lean_object* v_inst_233_, lean_object* v_inst_234_){
_start:
{
lean_object* v_res_235_; 
v_res_235_ = lp_mathlib_Module_End_applyModule(v_R_230_, v_M_231_, v_inst_232_, v_inst_233_, v_inst_234_);
lean_dec(v_inst_234_);
lean_dec_ref(v_inst_233_);
lean_dec_ref(v_inst_232_);
return v_res_235_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DistribSMul_toLinearMap___redArg(lean_object* v_inst_236_, lean_object* v_s_237_){
_start:
{
lean_object* v___x_238_; 
v___x_238_ = lean_apply_1(v_inst_236_, v_s_237_);
return v___x_238_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DistribSMul_toLinearMap(lean_object* v_R_239_, lean_object* v_S_240_, lean_object* v_M_241_, lean_object* v_inst_242_, lean_object* v_inst_243_, lean_object* v_inst_244_, lean_object* v_inst_245_, lean_object* v_inst_246_, lean_object* v_s_247_){
_start:
{
lean_object* v___x_248_; 
v___x_248_ = lean_apply_1(v_inst_245_, v_s_247_);
return v___x_248_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DistribSMul_toLinearMap___boxed(lean_object* v_R_249_, lean_object* v_S_250_, lean_object* v_M_251_, lean_object* v_inst_252_, lean_object* v_inst_253_, lean_object* v_inst_254_, lean_object* v_inst_255_, lean_object* v_inst_256_, lean_object* v_s_257_){
_start:
{
lean_object* v_res_258_; 
v_res_258_ = lp_mathlib_DistribSMul_toLinearMap(v_R_249_, v_S_250_, v_M_251_, v_inst_252_, v_inst_253_, v_inst_254_, v_inst_255_, v_inst_256_, v_s_257_);
lean_dec(v_inst_254_);
lean_dec_ref(v_inst_253_);
lean_dec_ref(v_inst_252_);
return v_res_258_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DistribMulAction_toModuleEnd___redArg(lean_object* v_inst_259_, lean_object* v_inst_260_, lean_object* v_inst_261_, lean_object* v_inst_262_){
_start:
{
lean_object* v___x_263_; 
v___x_263_ = lean_alloc_closure((void*)(lp_mathlib_DistribSMul_toLinearMap___boxed), 9, 8);
lean_closure_set(v___x_263_, 0, lean_box(0));
lean_closure_set(v___x_263_, 1, lean_box(0));
lean_closure_set(v___x_263_, 2, lean_box(0));
lean_closure_set(v___x_263_, 3, v_inst_259_);
lean_closure_set(v___x_263_, 4, v_inst_260_);
lean_closure_set(v___x_263_, 5, v_inst_261_);
lean_closure_set(v___x_263_, 6, v_inst_262_);
lean_closure_set(v___x_263_, 7, lean_box(0));
return v___x_263_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DistribMulAction_toModuleEnd(lean_object* v_R_264_, lean_object* v_S_265_, lean_object* v_M_266_, lean_object* v_inst_267_, lean_object* v_inst_268_, lean_object* v_inst_269_, lean_object* v_inst_270_, lean_object* v_inst_271_, lean_object* v_inst_272_){
_start:
{
lean_object* v___x_273_; 
v___x_273_ = lean_alloc_closure((void*)(lp_mathlib_DistribSMul_toLinearMap___boxed), 9, 8);
lean_closure_set(v___x_273_, 0, lean_box(0));
lean_closure_set(v___x_273_, 1, lean_box(0));
lean_closure_set(v___x_273_, 2, lean_box(0));
lean_closure_set(v___x_273_, 3, v_inst_267_);
lean_closure_set(v___x_273_, 4, v_inst_268_);
lean_closure_set(v___x_273_, 5, v_inst_269_);
lean_closure_set(v___x_273_, 6, v_inst_271_);
lean_closure_set(v___x_273_, 7, lean_box(0));
return v___x_273_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DistribMulAction_toModuleEnd___boxed(lean_object* v_R_274_, lean_object* v_S_275_, lean_object* v_M_276_, lean_object* v_inst_277_, lean_object* v_inst_278_, lean_object* v_inst_279_, lean_object* v_inst_280_, lean_object* v_inst_281_, lean_object* v_inst_282_){
_start:
{
lean_object* v_res_283_; 
v_res_283_ = lp_mathlib_DistribMulAction_toModuleEnd(v_R_274_, v_S_275_, v_M_276_, v_inst_277_, v_inst_278_, v_inst_279_, v_inst_280_, v_inst_281_, v_inst_282_);
lean_dec_ref(v_inst_280_);
return v_res_283_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Module_toModuleEnd___redArg(lean_object* v_inst_284_, lean_object* v_inst_285_, lean_object* v_inst_286_, lean_object* v_inst_287_){
_start:
{
lean_object* v___x_288_; 
v___x_288_ = lean_alloc_closure((void*)(lp_mathlib_DistribSMul_toLinearMap___boxed), 9, 8);
lean_closure_set(v___x_288_, 0, lean_box(0));
lean_closure_set(v___x_288_, 1, lean_box(0));
lean_closure_set(v___x_288_, 2, lean_box(0));
lean_closure_set(v___x_288_, 3, v_inst_284_);
lean_closure_set(v___x_288_, 4, v_inst_285_);
lean_closure_set(v___x_288_, 5, v_inst_286_);
lean_closure_set(v___x_288_, 6, v_inst_287_);
lean_closure_set(v___x_288_, 7, lean_box(0));
return v___x_288_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Module_toModuleEnd(lean_object* v_R_289_, lean_object* v_S_290_, lean_object* v_M_291_, lean_object* v_inst_292_, lean_object* v_inst_293_, lean_object* v_inst_294_, lean_object* v_inst_295_, lean_object* v_inst_296_, lean_object* v_inst_297_){
_start:
{
lean_object* v___x_298_; 
v___x_298_ = lean_alloc_closure((void*)(lp_mathlib_DistribSMul_toLinearMap___boxed), 9, 8);
lean_closure_set(v___x_298_, 0, lean_box(0));
lean_closure_set(v___x_298_, 1, lean_box(0));
lean_closure_set(v___x_298_, 2, lean_box(0));
lean_closure_set(v___x_298_, 3, v_inst_292_);
lean_closure_set(v___x_298_, 4, v_inst_293_);
lean_closure_set(v___x_298_, 5, v_inst_294_);
lean_closure_set(v___x_298_, 6, v_inst_296_);
lean_closure_set(v___x_298_, 7, lean_box(0));
return v___x_298_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Module_toModuleEnd___boxed(lean_object* v_R_299_, lean_object* v_S_300_, lean_object* v_M_301_, lean_object* v_inst_302_, lean_object* v_inst_303_, lean_object* v_inst_304_, lean_object* v_inst_305_, lean_object* v_inst_306_, lean_object* v_inst_307_){
_start:
{
lean_object* v_res_308_; 
v_res_308_ = lp_mathlib_Module_toModuleEnd(v_R_299_, v_S_300_, v_M_301_, v_inst_302_, v_inst_303_, v_inst_304_, v_inst_305_, v_inst_306_, v_inst_307_);
lean_dec_ref(v_inst_305_);
return v_res_308_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RingEquiv_moduleEndSelf___redArg___lam__0(lean_object* v_toOne_309_, lean_object* v_f_310_){
_start:
{
lean_object* v___x_311_; 
v___x_311_ = lean_apply_1(v_f_310_, v_toOne_309_);
return v___x_311_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RingEquiv_moduleEndSelf___redArg(lean_object* v_inst_312_){
_start:
{
lean_object* v_toAddCommMonoid_313_; lean_object* v___x_314_; lean_object* v___x_315_; lean_object* v_toOne_316_; lean_object* v___x_317_; lean_object* v___x_318_; lean_object* v___x_319_; lean_object* v___f_320_; lean_object* v___x_321_; 
v_toAddCommMonoid_313_ = lean_ctor_get(v_inst_312_, 0);
lean_inc_ref(v_toAddCommMonoid_313_);
lean_inc_ref(v_inst_312_);
v___x_314_ = lp_mathlib_Semiring_toNonAssocSemiring___redArg(v_inst_312_);
v___x_315_ = lp_mathlib_NonAssocSemiring_toAddCommMonoidWithOne___redArg(v___x_314_);
v_toOne_316_ = lean_ctor_get(v___x_315_, 2);
lean_inc(v_toOne_316_);
lean_dec_ref(v___x_315_);
v___x_317_ = lp_mathlib_Semiring_toModule___redArg(v_inst_312_);
v___x_318_ = lp_mathlib_Semiring_toOppositeModule___redArg(v_inst_312_);
v___x_319_ = lean_alloc_closure((void*)(lp_mathlib_DistribSMul_toLinearMap___boxed), 9, 8);
lean_closure_set(v___x_319_, 0, lean_box(0));
lean_closure_set(v___x_319_, 1, lean_box(0));
lean_closure_set(v___x_319_, 2, lean_box(0));
lean_closure_set(v___x_319_, 3, v_inst_312_);
lean_closure_set(v___x_319_, 4, v_toAddCommMonoid_313_);
lean_closure_set(v___x_319_, 5, v___x_317_);
lean_closure_set(v___x_319_, 6, v___x_318_);
lean_closure_set(v___x_319_, 7, lean_box(0));
v___f_320_ = lean_alloc_closure((void*)(lp_mathlib_RingEquiv_moduleEndSelf___redArg___lam__0), 2, 1);
lean_closure_set(v___f_320_, 0, v_toOne_316_);
v___x_321_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_321_, 0, v___x_319_);
lean_ctor_set(v___x_321_, 1, v___f_320_);
return v___x_321_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RingEquiv_moduleEndSelf(lean_object* v_R_322_, lean_object* v_inst_323_){
_start:
{
lean_object* v___x_324_; 
v___x_324_ = lp_mathlib_RingEquiv_moduleEndSelf___redArg(v_inst_323_);
return v___x_324_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RingEquiv_moduleEndSelfOp___redArg(lean_object* v_inst_325_){
_start:
{
lean_object* v___x_326_; lean_object* v_toAddCommMonoid_327_; lean_object* v___x_328_; lean_object* v_toNonUnitalNonAssocSemiring_329_; lean_object* v___x_330_; lean_object* v_toOne_331_; lean_object* v___x_332_; lean_object* v___x_333_; lean_object* v___x_334_; lean_object* v___f_335_; lean_object* v___x_336_; 
lean_inc_ref_n(v_inst_325_, 2);
v___x_326_ = lp_mathlib_MulOpposite_instSemiring___redArg(v_inst_325_);
v_toAddCommMonoid_327_ = lean_ctor_get(v_inst_325_, 0);
lean_inc_ref(v_toAddCommMonoid_327_);
v___x_328_ = lp_mathlib_Semiring_toNonAssocSemiring___redArg(v_inst_325_);
v_toNonUnitalNonAssocSemiring_329_ = lean_ctor_get(v___x_328_, 0);
lean_inc_ref(v_toNonUnitalNonAssocSemiring_329_);
v___x_330_ = lp_mathlib_NonAssocSemiring_toAddCommMonoidWithOne___redArg(v___x_328_);
v_toOne_331_ = lean_ctor_get(v___x_330_, 2);
lean_inc(v_toOne_331_);
lean_dec_ref(v___x_330_);
v___x_332_ = lp_mathlib_Semiring_toOppositeModule___redArg(v_inst_325_);
lean_dec_ref(v_inst_325_);
v___x_333_ = lp_mathlib_instDistribSMul___redArg(v_toNonUnitalNonAssocSemiring_329_);
v___x_334_ = lean_alloc_closure((void*)(lp_mathlib_DistribSMul_toLinearMap___boxed), 9, 8);
lean_closure_set(v___x_334_, 0, lean_box(0));
lean_closure_set(v___x_334_, 1, lean_box(0));
lean_closure_set(v___x_334_, 2, lean_box(0));
lean_closure_set(v___x_334_, 3, v___x_326_);
lean_closure_set(v___x_334_, 4, v_toAddCommMonoid_327_);
lean_closure_set(v___x_334_, 5, v___x_332_);
lean_closure_set(v___x_334_, 6, v___x_333_);
lean_closure_set(v___x_334_, 7, lean_box(0));
v___f_335_ = lean_alloc_closure((void*)(lp_mathlib_RingEquiv_moduleEndSelf___redArg___lam__0), 2, 1);
lean_closure_set(v___f_335_, 0, v_toOne_331_);
v___x_336_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_336_, 0, v___x_334_);
lean_ctor_set(v___x_336_, 1, v___f_335_);
return v___x_336_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RingEquiv_moduleEndSelfOp(lean_object* v_R_337_, lean_object* v_inst_338_){
_start:
{
lean_object* v___x_339_; 
v___x_339_ = lp_mathlib_RingEquiv_moduleEndSelfOp___redArg(v_inst_338_);
return v___x_339_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LinearMap_smulRight___redArg___lam__0(lean_object* v_f_340_, lean_object* v_inst_341_, lean_object* v_x_342_, lean_object* v_b_343_){
_start:
{
lean_object* v___x_344_; lean_object* v___x_345_; 
v___x_344_ = lean_apply_1(v_f_340_, v_b_343_);
v___x_345_ = lean_apply_2(v_inst_341_, v___x_344_, v_x_342_);
return v___x_345_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LinearMap_smulRight___redArg(lean_object* v_inst_346_, lean_object* v_f_347_, lean_object* v_x_348_){
_start:
{
lean_object* v___f_349_; 
v___f_349_ = lean_alloc_closure((void*)(lp_mathlib_LinearMap_smulRight___redArg___lam__0), 4, 3);
lean_closure_set(v___f_349_, 0, v_f_347_);
lean_closure_set(v___f_349_, 1, v_inst_346_);
lean_closure_set(v___f_349_, 2, v_x_348_);
return v___f_349_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LinearMap_smulRight(lean_object* v_R_350_, lean_object* v_S_351_, lean_object* v_M_352_, lean_object* v_M_u2081_353_, lean_object* v_inst_354_, lean_object* v_inst_355_, lean_object* v_inst_356_, lean_object* v_inst_357_, lean_object* v_inst_358_, lean_object* v_inst_359_, lean_object* v_inst_360_, lean_object* v_inst_361_, lean_object* v_inst_362_, lean_object* v_f_363_, lean_object* v_x_364_){
_start:
{
lean_object* v___f_365_; 
v___f_365_ = lean_alloc_closure((void*)(lp_mathlib_LinearMap_smulRight___redArg___lam__0), 4, 3);
lean_closure_set(v___f_365_, 0, v_f_363_);
lean_closure_set(v___f_365_, 1, v_inst_361_);
lean_closure_set(v___f_365_, 2, v_x_364_);
return v___f_365_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LinearMap_smulRight___boxed(lean_object* v_R_366_, lean_object* v_S_367_, lean_object* v_M_368_, lean_object* v_M_u2081_369_, lean_object* v_inst_370_, lean_object* v_inst_371_, lean_object* v_inst_372_, lean_object* v_inst_373_, lean_object* v_inst_374_, lean_object* v_inst_375_, lean_object* v_inst_376_, lean_object* v_inst_377_, lean_object* v_inst_378_, lean_object* v_f_379_, lean_object* v_x_380_){
_start:
{
lean_object* v_res_381_; 
v_res_381_ = lp_mathlib_LinearMap_smulRight(v_R_366_, v_S_367_, v_M_368_, v_M_u2081_369_, v_inst_370_, v_inst_371_, v_inst_372_, v_inst_373_, v_inst_374_, v_inst_375_, v_inst_376_, v_inst_377_, v_inst_378_, v_f_379_, v_x_380_);
lean_dec(v_inst_376_);
lean_dec_ref(v_inst_375_);
lean_dec(v_inst_374_);
lean_dec(v_inst_373_);
lean_dec_ref(v_inst_372_);
lean_dec_ref(v_inst_371_);
lean_dec_ref(v_inst_370_);
return v_res_381_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LinearMap_apply_u2097_x27___lam__0(lean_object* v_v_382_, lean_object* v___y_383_){
_start:
{
lean_object* v___x_384_; 
v___x_384_ = lean_apply_1(v___y_383_, v_v_382_);
return v___x_384_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LinearMap_apply_u2097_x27(lean_object* v_R_386_, lean_object* v_S_387_, lean_object* v_M_388_, lean_object* v_M_u2082_389_, lean_object* v_inst_390_, lean_object* v_inst_391_, lean_object* v_inst_392_, lean_object* v_inst_393_, lean_object* v_inst_394_, lean_object* v_inst_395_, lean_object* v_inst_396_, lean_object* v_inst_397_){
_start:
{
lean_object* v___f_398_; 
v___f_398_ = ((lean_object*)(lp_mathlib_LinearMap_apply_u2097_x27___closed__0));
return v___f_398_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LinearMap_apply_u2097_x27___boxed(lean_object* v_R_399_, lean_object* v_S_400_, lean_object* v_M_401_, lean_object* v_M_u2082_402_, lean_object* v_inst_403_, lean_object* v_inst_404_, lean_object* v_inst_405_, lean_object* v_inst_406_, lean_object* v_inst_407_, lean_object* v_inst_408_, lean_object* v_inst_409_, lean_object* v_inst_410_){
_start:
{
lean_object* v_res_411_; 
v_res_411_ = lp_mathlib_LinearMap_apply_u2097_x27(v_R_399_, v_S_400_, v_M_401_, v_M_u2082_402_, v_inst_403_, v_inst_404_, v_inst_405_, v_inst_406_, v_inst_407_, v_inst_408_, v_inst_409_, v_inst_410_);
lean_dec(v_inst_409_);
lean_dec(v_inst_408_);
lean_dec(v_inst_407_);
lean_dec_ref(v_inst_406_);
lean_dec_ref(v_inst_405_);
lean_dec_ref(v_inst_404_);
lean_dec_ref(v_inst_403_);
return v_res_411_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LinearMap_compRight___redArg(lean_object* v_f_412_){
_start:
{
lean_object* v___f_413_; 
v___f_413_ = lean_alloc_closure((void*)(lp_mathlib_Module_End_instMul___lam__0), 3, 1);
lean_closure_set(v___f_413_, 0, v_f_412_);
return v___f_413_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LinearMap_compRight(lean_object* v_R_414_, lean_object* v_S_415_, lean_object* v_M_416_, lean_object* v_M_u2081_417_, lean_object* v_M_u2082_418_, lean_object* v_inst_419_, lean_object* v_inst_420_, lean_object* v_inst_421_, lean_object* v_inst_422_, lean_object* v_inst_423_, lean_object* v_inst_424_, lean_object* v_inst_425_, lean_object* v_inst_426_, lean_object* v_inst_427_, lean_object* v_inst_428_, lean_object* v_inst_429_, lean_object* v_inst_430_, lean_object* v_inst_431_, lean_object* v_f_432_){
_start:
{
lean_object* v___f_433_; 
v___f_433_ = lean_alloc_closure((void*)(lp_mathlib_Module_End_instMul___lam__0), 3, 1);
lean_closure_set(v___f_433_, 0, v_f_432_);
return v___f_433_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LinearMap_compRight___boxed(lean_object** _args){
lean_object* v_R_434_ = _args[0];
lean_object* v_S_435_ = _args[1];
lean_object* v_M_436_ = _args[2];
lean_object* v_M_u2081_437_ = _args[3];
lean_object* v_M_u2082_438_ = _args[4];
lean_object* v_inst_439_ = _args[5];
lean_object* v_inst_440_ = _args[6];
lean_object* v_inst_441_ = _args[7];
lean_object* v_inst_442_ = _args[8];
lean_object* v_inst_443_ = _args[9];
lean_object* v_inst_444_ = _args[10];
lean_object* v_inst_445_ = _args[11];
lean_object* v_inst_446_ = _args[12];
lean_object* v_inst_447_ = _args[13];
lean_object* v_inst_448_ = _args[14];
lean_object* v_inst_449_ = _args[15];
lean_object* v_inst_450_ = _args[16];
lean_object* v_inst_451_ = _args[17];
lean_object* v_f_452_ = _args[18];
_start:
{
lean_object* v_res_453_; 
v_res_453_ = lp_mathlib_LinearMap_compRight(v_R_434_, v_S_435_, v_M_436_, v_M_u2081_437_, v_M_u2082_438_, v_inst_439_, v_inst_440_, v_inst_441_, v_inst_442_, v_inst_443_, v_inst_444_, v_inst_445_, v_inst_446_, v_inst_447_, v_inst_448_, v_inst_449_, v_inst_450_, v_inst_451_, v_f_452_);
lean_dec(v_inst_448_);
lean_dec(v_inst_447_);
lean_dec(v_inst_446_);
lean_dec(v_inst_445_);
lean_dec(v_inst_444_);
lean_dec_ref(v_inst_443_);
lean_dec_ref(v_inst_442_);
lean_dec_ref(v_inst_441_);
lean_dec_ref(v_inst_440_);
lean_dec_ref(v_inst_439_);
return v_res_453_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LinearMap_apply_u2097(lean_object* v_R_454_, lean_object* v_M_455_, lean_object* v_M_u2082_456_, lean_object* v_inst_457_, lean_object* v_inst_458_, lean_object* v_inst_459_, lean_object* v_inst_460_, lean_object* v_inst_461_){
_start:
{
lean_object* v___f_462_; 
v___f_462_ = ((lean_object*)(lp_mathlib_LinearMap_apply_u2097_x27___closed__0));
return v___f_462_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LinearMap_apply_u2097___boxed(lean_object* v_R_463_, lean_object* v_M_464_, lean_object* v_M_u2082_465_, lean_object* v_inst_466_, lean_object* v_inst_467_, lean_object* v_inst_468_, lean_object* v_inst_469_, lean_object* v_inst_470_){
_start:
{
lean_object* v_res_471_; 
v_res_471_ = lp_mathlib_LinearMap_apply_u2097(v_R_463_, v_M_464_, v_M_u2082_465_, v_inst_466_, v_inst_467_, v_inst_468_, v_inst_469_, v_inst_470_);
lean_dec(v_inst_470_);
lean_dec(v_inst_469_);
lean_dec_ref(v_inst_468_);
lean_dec_ref(v_inst_467_);
lean_dec_ref(v_inst_466_);
return v_res_471_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LinearMap_smulRight_u2097___redArg___lam__0(lean_object* v_inst_472_, lean_object* v_f_473_, lean_object* v___y_474_, lean_object* v___y_475_){
_start:
{
lean_object* v___x_476_; 
v___x_476_ = lp_mathlib_LinearMap_smulRight___redArg___lam__0(v_f_473_, v_inst_472_, v___y_474_, v___y_475_);
return v___x_476_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LinearMap_smulRight_u2097___redArg(lean_object* v_inst_477_){
_start:
{
lean_object* v___f_478_; 
v___f_478_ = lean_alloc_closure((void*)(lp_mathlib_LinearMap_smulRight_u2097___redArg___lam__0), 4, 1);
lean_closure_set(v___f_478_, 0, v_inst_477_);
return v___f_478_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LinearMap_smulRight_u2097(lean_object* v_R_479_, lean_object* v_M_480_, lean_object* v_M_u2082_481_, lean_object* v_inst_482_, lean_object* v_inst_483_, lean_object* v_inst_484_, lean_object* v_inst_485_, lean_object* v_inst_486_){
_start:
{
lean_object* v___f_487_; 
v___f_487_ = lean_alloc_closure((void*)(lp_mathlib_LinearMap_smulRight_u2097___redArg___lam__0), 4, 1);
lean_closure_set(v___f_487_, 0, v_inst_485_);
return v___f_487_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LinearMap_smulRight_u2097___boxed(lean_object* v_R_488_, lean_object* v_M_489_, lean_object* v_M_u2082_490_, lean_object* v_inst_491_, lean_object* v_inst_492_, lean_object* v_inst_493_, lean_object* v_inst_494_, lean_object* v_inst_495_){
_start:
{
lean_object* v_res_496_; 
v_res_496_ = lp_mathlib_LinearMap_smulRight_u2097(v_R_488_, v_M_489_, v_M_u2082_490_, v_inst_491_, v_inst_492_, v_inst_493_, v_inst_494_, v_inst_495_);
lean_dec(v_inst_495_);
lean_dec_ref(v_inst_493_);
lean_dec_ref(v_inst_492_);
lean_dec_ref(v_inst_491_);
return v_res_496_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Group_Center(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Module_Equiv_Opposite(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Module_Torsion_Free(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Module_LinearMap_End(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Group_Center(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Module_Equiv_Opposite(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Module_Torsion_Free(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_Algebra_Module_LinearMap_End(uint8_t builtin) {
lean_object * res;
if (_G_meta_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_meta_initialized = true;
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
lp_mathlib_Module_End_smulLeft__eq___auto__1 = _init_lp_mathlib_Module_End_smulLeft__eq___auto__1();
lean_mark_persistent(lp_mathlib_Module_End_smulLeft__eq___auto__1);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Group_Center(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Module_Equiv_Opposite(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Module_Torsion_Free(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Algebra_Module_LinearMap_End(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Group_Center(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Module_Equiv_Opposite(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Module_Torsion_Free(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Module_LinearMap_End(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_Algebra_Module_LinearMap_End(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_Algebra_Module_LinearMap_End(builtin);
}
#ifdef __cplusplus
}
#endif
