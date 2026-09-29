// Lean compiler output
// Module: Mathlib.Order.Closure
// Imports: public import Init public meta import Init public import Mathlib.Data.Set.BooleanAlgebra public import Mathlib.Data.SetLike.Basic public import Mathlib.Order.Hom.Basic
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
lean_object* l_id___boxed(lean_object*, lean_object*);
lean_object* lp_mathlib_OrderIso_conj___redArg(lean_object*);
lean_object* lp_mathlib_Equiv_symm___redArg(lean_object*);
lean_object* l_Lean_Name_mkStr4(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lean_mk_empty_array_with_capacity(lean_object*);
lean_object* l_Lean_Name_mkStr1(lean_object*);
lean_object* l_Lean_mkAtom(lean_object*);
lean_object* lean_array_push(lean_object*, lean_object*);
lean_object* lp_mathlib_CompleteLattice_toCompleteSemilatticeInf___redArg(lean_object*);
static const lean_string_object lp_mathlib_ClosureOperator_isClosed__iff___autoParam___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "Lean"};
static const lean_object* lp_mathlib_ClosureOperator_isClosed__iff___autoParam___closed__0 = (const lean_object*)&lp_mathlib_ClosureOperator_isClosed__iff___autoParam___closed__0_value;
static const lean_string_object lp_mathlib_ClosureOperator_isClosed__iff___autoParam___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "Parser"};
static const lean_object* lp_mathlib_ClosureOperator_isClosed__iff___autoParam___closed__1 = (const lean_object*)&lp_mathlib_ClosureOperator_isClosed__iff___autoParam___closed__1_value;
static const lean_string_object lp_mathlib_ClosureOperator_isClosed__iff___autoParam___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "Tactic"};
static const lean_object* lp_mathlib_ClosureOperator_isClosed__iff___autoParam___closed__2 = (const lean_object*)&lp_mathlib_ClosureOperator_isClosed__iff___autoParam___closed__2_value;
static const lean_string_object lp_mathlib_ClosureOperator_isClosed__iff___autoParam___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 9, .m_data = "tacticSeq"};
static const lean_object* lp_mathlib_ClosureOperator_isClosed__iff___autoParam___closed__3 = (const lean_object*)&lp_mathlib_ClosureOperator_isClosed__iff___autoParam___closed__3_value;
static const lean_ctor_object lp_mathlib_ClosureOperator_isClosed__iff___autoParam___closed__4_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_ClosureOperator_isClosed__iff___autoParam___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib_ClosureOperator_isClosed__iff___autoParam___closed__4_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_ClosureOperator_isClosed__iff___autoParam___closed__4_value_aux_0),((lean_object*)&lp_mathlib_ClosureOperator_isClosed__iff___autoParam___closed__1_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib_ClosureOperator_isClosed__iff___autoParam___closed__4_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_ClosureOperator_isClosed__iff___autoParam___closed__4_value_aux_1),((lean_object*)&lp_mathlib_ClosureOperator_isClosed__iff___autoParam___closed__2_value),LEAN_SCALAR_PTR_LITERAL(166, 58, 35, 182, 187, 130, 147, 254)}};
static const lean_ctor_object lp_mathlib_ClosureOperator_isClosed__iff___autoParam___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_ClosureOperator_isClosed__iff___autoParam___closed__4_value_aux_2),((lean_object*)&lp_mathlib_ClosureOperator_isClosed__iff___autoParam___closed__3_value),LEAN_SCALAR_PTR_LITERAL(212, 140, 85, 215, 241, 69, 7, 118)}};
static const lean_object* lp_mathlib_ClosureOperator_isClosed__iff___autoParam___closed__4 = (const lean_object*)&lp_mathlib_ClosureOperator_isClosed__iff___autoParam___closed__4_value;
static const lean_array_object lp_mathlib_ClosureOperator_isClosed__iff___autoParam___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_array_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 246}, .m_size = 0, .m_capacity = 0, .m_data = {}};
static const lean_object* lp_mathlib_ClosureOperator_isClosed__iff___autoParam___closed__5 = (const lean_object*)&lp_mathlib_ClosureOperator_isClosed__iff___autoParam___closed__5_value;
static const lean_string_object lp_mathlib_ClosureOperator_isClosed__iff___autoParam___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 19, .m_capacity = 19, .m_length = 18, .m_data = "tacticSeq1Indented"};
static const lean_object* lp_mathlib_ClosureOperator_isClosed__iff___autoParam___closed__6 = (const lean_object*)&lp_mathlib_ClosureOperator_isClosed__iff___autoParam___closed__6_value;
static const lean_ctor_object lp_mathlib_ClosureOperator_isClosed__iff___autoParam___closed__7_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_ClosureOperator_isClosed__iff___autoParam___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib_ClosureOperator_isClosed__iff___autoParam___closed__7_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_ClosureOperator_isClosed__iff___autoParam___closed__7_value_aux_0),((lean_object*)&lp_mathlib_ClosureOperator_isClosed__iff___autoParam___closed__1_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib_ClosureOperator_isClosed__iff___autoParam___closed__7_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_ClosureOperator_isClosed__iff___autoParam___closed__7_value_aux_1),((lean_object*)&lp_mathlib_ClosureOperator_isClosed__iff___autoParam___closed__2_value),LEAN_SCALAR_PTR_LITERAL(166, 58, 35, 182, 187, 130, 147, 254)}};
static const lean_ctor_object lp_mathlib_ClosureOperator_isClosed__iff___autoParam___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_ClosureOperator_isClosed__iff___autoParam___closed__7_value_aux_2),((lean_object*)&lp_mathlib_ClosureOperator_isClosed__iff___autoParam___closed__6_value),LEAN_SCALAR_PTR_LITERAL(223, 90, 160, 238, 133, 180, 23, 239)}};
static const lean_object* lp_mathlib_ClosureOperator_isClosed__iff___autoParam___closed__7 = (const lean_object*)&lp_mathlib_ClosureOperator_isClosed__iff___autoParam___closed__7_value;
static const lean_string_object lp_mathlib_ClosureOperator_isClosed__iff___autoParam___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "null"};
static const lean_object* lp_mathlib_ClosureOperator_isClosed__iff___autoParam___closed__8 = (const lean_object*)&lp_mathlib_ClosureOperator_isClosed__iff___autoParam___closed__8_value;
static const lean_ctor_object lp_mathlib_ClosureOperator_isClosed__iff___autoParam___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_ClosureOperator_isClosed__iff___autoParam___closed__8_value),LEAN_SCALAR_PTR_LITERAL(24, 58, 49, 223, 146, 207, 197, 136)}};
static const lean_object* lp_mathlib_ClosureOperator_isClosed__iff___autoParam___closed__9 = (const lean_object*)&lp_mathlib_ClosureOperator_isClosed__iff___autoParam___closed__9_value;
static const lean_string_object lp_mathlib_ClosureOperator_isClosed__iff___autoParam___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "Aesop"};
static const lean_object* lp_mathlib_ClosureOperator_isClosed__iff___autoParam___closed__10 = (const lean_object*)&lp_mathlib_ClosureOperator_isClosed__iff___autoParam___closed__10_value;
static const lean_string_object lp_mathlib_ClosureOperator_isClosed__iff___autoParam___closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "Frontend"};
static const lean_object* lp_mathlib_ClosureOperator_isClosed__iff___autoParam___closed__11 = (const lean_object*)&lp_mathlib_ClosureOperator_isClosed__iff___autoParam___closed__11_value;
static const lean_string_object lp_mathlib_ClosureOperator_isClosed__iff___autoParam___closed__12_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 12, .m_capacity = 12, .m_length = 11, .m_data = "aesopTactic"};
static const lean_object* lp_mathlib_ClosureOperator_isClosed__iff___autoParam___closed__12 = (const lean_object*)&lp_mathlib_ClosureOperator_isClosed__iff___autoParam___closed__12_value;
static const lean_ctor_object lp_mathlib_ClosureOperator_isClosed__iff___autoParam___closed__13_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_ClosureOperator_isClosed__iff___autoParam___closed__10_value),LEAN_SCALAR_PTR_LITERAL(97, 226, 101, 135, 78, 117, 164, 248)}};
static const lean_ctor_object lp_mathlib_ClosureOperator_isClosed__iff___autoParam___closed__13_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_ClosureOperator_isClosed__iff___autoParam___closed__13_value_aux_0),((lean_object*)&lp_mathlib_ClosureOperator_isClosed__iff___autoParam___closed__11_value),LEAN_SCALAR_PTR_LITERAL(212, 61, 86, 210, 90, 71, 197, 0)}};
static const lean_ctor_object lp_mathlib_ClosureOperator_isClosed__iff___autoParam___closed__13_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_ClosureOperator_isClosed__iff___autoParam___closed__13_value_aux_1),((lean_object*)&lp_mathlib_ClosureOperator_isClosed__iff___autoParam___closed__1_value),LEAN_SCALAR_PTR_LITERAL(13, 69, 189, 207, 58, 198, 186, 160)}};
static const lean_ctor_object lp_mathlib_ClosureOperator_isClosed__iff___autoParam___closed__13_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_ClosureOperator_isClosed__iff___autoParam___closed__13_value_aux_2),((lean_object*)&lp_mathlib_ClosureOperator_isClosed__iff___autoParam___closed__12_value),LEAN_SCALAR_PTR_LITERAL(54, 142, 162, 195, 161, 101, 248, 175)}};
static const lean_object* lp_mathlib_ClosureOperator_isClosed__iff___autoParam___closed__13 = (const lean_object*)&lp_mathlib_ClosureOperator_isClosed__iff___autoParam___closed__13_value;
static const lean_string_object lp_mathlib_ClosureOperator_isClosed__iff___autoParam___closed__14_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "aesop"};
static const lean_object* lp_mathlib_ClosureOperator_isClosed__iff___autoParam___closed__14 = (const lean_object*)&lp_mathlib_ClosureOperator_isClosed__iff___autoParam___closed__14_value;
static lean_once_cell_t lp_mathlib_ClosureOperator_isClosed__iff___autoParam___closed__15_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_ClosureOperator_isClosed__iff___autoParam___closed__15;
static lean_once_cell_t lp_mathlib_ClosureOperator_isClosed__iff___autoParam___closed__16_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_ClosureOperator_isClosed__iff___autoParam___closed__16;
static const lean_ctor_object lp_mathlib_ClosureOperator_isClosed__iff___autoParam___closed__17_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(2) << 1) | 1)),((lean_object*)&lp_mathlib_ClosureOperator_isClosed__iff___autoParam___closed__9_value),((lean_object*)&lp_mathlib_ClosureOperator_isClosed__iff___autoParam___closed__5_value)}};
static const lean_object* lp_mathlib_ClosureOperator_isClosed__iff___autoParam___closed__17 = (const lean_object*)&lp_mathlib_ClosureOperator_isClosed__iff___autoParam___closed__17_value;
static lean_once_cell_t lp_mathlib_ClosureOperator_isClosed__iff___autoParam___closed__18_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_ClosureOperator_isClosed__iff___autoParam___closed__18;
static lean_once_cell_t lp_mathlib_ClosureOperator_isClosed__iff___autoParam___closed__19_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_ClosureOperator_isClosed__iff___autoParam___closed__19;
static lean_once_cell_t lp_mathlib_ClosureOperator_isClosed__iff___autoParam___closed__20_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_ClosureOperator_isClosed__iff___autoParam___closed__20;
static lean_once_cell_t lp_mathlib_ClosureOperator_isClosed__iff___autoParam___closed__21_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_ClosureOperator_isClosed__iff___autoParam___closed__21;
static lean_once_cell_t lp_mathlib_ClosureOperator_isClosed__iff___autoParam___closed__22_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_ClosureOperator_isClosed__iff___autoParam___closed__22;
static lean_once_cell_t lp_mathlib_ClosureOperator_isClosed__iff___autoParam___closed__23_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_ClosureOperator_isClosed__iff___autoParam___closed__23;
static lean_once_cell_t lp_mathlib_ClosureOperator_isClosed__iff___autoParam___closed__24_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_ClosureOperator_isClosed__iff___autoParam___closed__24;
static lean_once_cell_t lp_mathlib_ClosureOperator_isClosed__iff___autoParam___closed__25_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_ClosureOperator_isClosed__iff___autoParam___closed__25;
LEAN_EXPORT lean_object* lp_mathlib_ClosureOperator_isClosed__iff___autoParam;
LEAN_EXPORT lean_object* lp_mathlib_ClosureOperator_conjBy___redArg___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ClosureOperator_conjBy___redArg___lam__1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ClosureOperator_conjBy___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ClosureOperator_conjBy(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ClosureOperator_conjBy___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_ClosureOperator_id___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*1, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_id___boxed, .m_arity = 2, .m_num_fixed = 1, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1))} };
static const lean_object* lp_mathlib_ClosureOperator_id___closed__0 = (const lean_object*)&lp_mathlib_ClosureOperator_id___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_ClosureOperator_id(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ClosureOperator_id___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ClosureOperator_instInhabited(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ClosureOperator_instInhabited___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ClosureOperator_toCloseds___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ClosureOperator_toCloseds(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ClosureOperator_toCloseds___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ClosureOperator_mk_x27___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ClosureOperator_mk_x27___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ClosureOperator_mk_x27(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ClosureOperator_mk_x27___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ClosureOperator_mk_u2082___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ClosureOperator_mk_u2082___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ClosureOperator_mk_u2082(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ClosureOperator_mk_u2082___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ClosureOperator_ofPred___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ClosureOperator_ofPred___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ClosureOperator_ofPred(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ClosureOperator_ofPred___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ClosureOperator_ofCompletePred___redArg___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ClosureOperator_ofCompletePred___redArg___lam__0___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ClosureOperator_ofCompletePred___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ClosureOperator_ofCompletePred(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderIso_equivClosureOperator___redArg___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderIso_equivClosureOperator___redArg___lam__1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderIso_equivClosureOperator___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderIso_equivClosureOperator(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderIso_equivClosureOperator___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LowerAdjoint_id___lam__0(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LowerAdjoint_id___lam__0___boxed(lean_object*);
static const lean_closure_object lp_mathlib_LowerAdjoint_id___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_LowerAdjoint_id___lam__0___boxed, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_LowerAdjoint_id___closed__0 = (const lean_object*)&lp_mathlib_LowerAdjoint_id___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_LowerAdjoint_id(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LowerAdjoint_id___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LowerAdjoint_instInhabitedId(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LowerAdjoint_instInhabitedId___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LowerAdjoint_instCoeFunForall___lam__0(lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_LowerAdjoint_instCoeFunForall___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_LowerAdjoint_instCoeFunForall___lam__0, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_LowerAdjoint_instCoeFunForall___closed__0 = (const lean_object*)&lp_mathlib_LowerAdjoint_instCoeFunForall___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_LowerAdjoint_instCoeFunForall(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LowerAdjoint_instCoeFunForall___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LowerAdjoint_closureOperator___redArg___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LowerAdjoint_closureOperator___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LowerAdjoint_closureOperator(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LowerAdjoint_closureOperator___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LowerAdjoint_toClosed___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LowerAdjoint_toClosed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LowerAdjoint_toClosed___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_GaloisConnection_lowerAdjoint___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_GaloisConnection_lowerAdjoint___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_GaloisConnection_lowerAdjoint(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_GaloisConnection_lowerAdjoint___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_GaloisConnection_closureOperator___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_GaloisConnection_closureOperator(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_GaloisConnection_closureOperator___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ClosureOperator_gi___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ClosureOperator_gi___lam__0___boxed(lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_ClosureOperator_gi___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_ClosureOperator_gi___lam__0___boxed, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_ClosureOperator_gi___closed__0 = (const lean_object*)&lp_mathlib_ClosureOperator_gi___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_ClosureOperator_gi(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ClosureOperator_gi___boxed(lean_object*, lean_object*, lean_object*);
static lean_object* _init_lp_mathlib_ClosureOperator_isClosed__iff___autoParam___closed__15(void){
_start:
{
lean_object* v___x_30_; lean_object* v___x_31_; 
v___x_30_ = ((lean_object*)(lp_mathlib_ClosureOperator_isClosed__iff___autoParam___closed__14));
v___x_31_ = l_Lean_mkAtom(v___x_30_);
return v___x_31_;
}
}
static lean_object* _init_lp_mathlib_ClosureOperator_isClosed__iff___autoParam___closed__16(void){
_start:
{
lean_object* v___x_32_; lean_object* v___x_33_; lean_object* v___x_34_; 
v___x_32_ = lean_obj_once(&lp_mathlib_ClosureOperator_isClosed__iff___autoParam___closed__15, &lp_mathlib_ClosureOperator_isClosed__iff___autoParam___closed__15_once, _init_lp_mathlib_ClosureOperator_isClosed__iff___autoParam___closed__15);
v___x_33_ = ((lean_object*)(lp_mathlib_ClosureOperator_isClosed__iff___autoParam___closed__5));
v___x_34_ = lean_array_push(v___x_33_, v___x_32_);
return v___x_34_;
}
}
static lean_object* _init_lp_mathlib_ClosureOperator_isClosed__iff___autoParam___closed__18(void){
_start:
{
lean_object* v___x_39_; lean_object* v___x_40_; lean_object* v___x_41_; 
v___x_39_ = ((lean_object*)(lp_mathlib_ClosureOperator_isClosed__iff___autoParam___closed__17));
v___x_40_ = lean_obj_once(&lp_mathlib_ClosureOperator_isClosed__iff___autoParam___closed__16, &lp_mathlib_ClosureOperator_isClosed__iff___autoParam___closed__16_once, _init_lp_mathlib_ClosureOperator_isClosed__iff___autoParam___closed__16);
v___x_41_ = lean_array_push(v___x_40_, v___x_39_);
return v___x_41_;
}
}
static lean_object* _init_lp_mathlib_ClosureOperator_isClosed__iff___autoParam___closed__19(void){
_start:
{
lean_object* v___x_42_; lean_object* v___x_43_; lean_object* v___x_44_; lean_object* v___x_45_; 
v___x_42_ = lean_obj_once(&lp_mathlib_ClosureOperator_isClosed__iff___autoParam___closed__18, &lp_mathlib_ClosureOperator_isClosed__iff___autoParam___closed__18_once, _init_lp_mathlib_ClosureOperator_isClosed__iff___autoParam___closed__18);
v___x_43_ = ((lean_object*)(lp_mathlib_ClosureOperator_isClosed__iff___autoParam___closed__13));
v___x_44_ = lean_box(2);
v___x_45_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_45_, 0, v___x_44_);
lean_ctor_set(v___x_45_, 1, v___x_43_);
lean_ctor_set(v___x_45_, 2, v___x_42_);
return v___x_45_;
}
}
static lean_object* _init_lp_mathlib_ClosureOperator_isClosed__iff___autoParam___closed__20(void){
_start:
{
lean_object* v___x_46_; lean_object* v___x_47_; lean_object* v___x_48_; 
v___x_46_ = lean_obj_once(&lp_mathlib_ClosureOperator_isClosed__iff___autoParam___closed__19, &lp_mathlib_ClosureOperator_isClosed__iff___autoParam___closed__19_once, _init_lp_mathlib_ClosureOperator_isClosed__iff___autoParam___closed__19);
v___x_47_ = ((lean_object*)(lp_mathlib_ClosureOperator_isClosed__iff___autoParam___closed__5));
v___x_48_ = lean_array_push(v___x_47_, v___x_46_);
return v___x_48_;
}
}
static lean_object* _init_lp_mathlib_ClosureOperator_isClosed__iff___autoParam___closed__21(void){
_start:
{
lean_object* v___x_49_; lean_object* v___x_50_; lean_object* v___x_51_; lean_object* v___x_52_; 
v___x_49_ = lean_obj_once(&lp_mathlib_ClosureOperator_isClosed__iff___autoParam___closed__20, &lp_mathlib_ClosureOperator_isClosed__iff___autoParam___closed__20_once, _init_lp_mathlib_ClosureOperator_isClosed__iff___autoParam___closed__20);
v___x_50_ = ((lean_object*)(lp_mathlib_ClosureOperator_isClosed__iff___autoParam___closed__9));
v___x_51_ = lean_box(2);
v___x_52_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_52_, 0, v___x_51_);
lean_ctor_set(v___x_52_, 1, v___x_50_);
lean_ctor_set(v___x_52_, 2, v___x_49_);
return v___x_52_;
}
}
static lean_object* _init_lp_mathlib_ClosureOperator_isClosed__iff___autoParam___closed__22(void){
_start:
{
lean_object* v___x_53_; lean_object* v___x_54_; lean_object* v___x_55_; 
v___x_53_ = lean_obj_once(&lp_mathlib_ClosureOperator_isClosed__iff___autoParam___closed__21, &lp_mathlib_ClosureOperator_isClosed__iff___autoParam___closed__21_once, _init_lp_mathlib_ClosureOperator_isClosed__iff___autoParam___closed__21);
v___x_54_ = ((lean_object*)(lp_mathlib_ClosureOperator_isClosed__iff___autoParam___closed__5));
v___x_55_ = lean_array_push(v___x_54_, v___x_53_);
return v___x_55_;
}
}
static lean_object* _init_lp_mathlib_ClosureOperator_isClosed__iff___autoParam___closed__23(void){
_start:
{
lean_object* v___x_56_; lean_object* v___x_57_; lean_object* v___x_58_; lean_object* v___x_59_; 
v___x_56_ = lean_obj_once(&lp_mathlib_ClosureOperator_isClosed__iff___autoParam___closed__22, &lp_mathlib_ClosureOperator_isClosed__iff___autoParam___closed__22_once, _init_lp_mathlib_ClosureOperator_isClosed__iff___autoParam___closed__22);
v___x_57_ = ((lean_object*)(lp_mathlib_ClosureOperator_isClosed__iff___autoParam___closed__7));
v___x_58_ = lean_box(2);
v___x_59_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_59_, 0, v___x_58_);
lean_ctor_set(v___x_59_, 1, v___x_57_);
lean_ctor_set(v___x_59_, 2, v___x_56_);
return v___x_59_;
}
}
static lean_object* _init_lp_mathlib_ClosureOperator_isClosed__iff___autoParam___closed__24(void){
_start:
{
lean_object* v___x_60_; lean_object* v___x_61_; lean_object* v___x_62_; 
v___x_60_ = lean_obj_once(&lp_mathlib_ClosureOperator_isClosed__iff___autoParam___closed__23, &lp_mathlib_ClosureOperator_isClosed__iff___autoParam___closed__23_once, _init_lp_mathlib_ClosureOperator_isClosed__iff___autoParam___closed__23);
v___x_61_ = ((lean_object*)(lp_mathlib_ClosureOperator_isClosed__iff___autoParam___closed__5));
v___x_62_ = lean_array_push(v___x_61_, v___x_60_);
return v___x_62_;
}
}
static lean_object* _init_lp_mathlib_ClosureOperator_isClosed__iff___autoParam___closed__25(void){
_start:
{
lean_object* v___x_63_; lean_object* v___x_64_; lean_object* v___x_65_; lean_object* v___x_66_; 
v___x_63_ = lean_obj_once(&lp_mathlib_ClosureOperator_isClosed__iff___autoParam___closed__24, &lp_mathlib_ClosureOperator_isClosed__iff___autoParam___closed__24_once, _init_lp_mathlib_ClosureOperator_isClosed__iff___autoParam___closed__24);
v___x_64_ = ((lean_object*)(lp_mathlib_ClosureOperator_isClosed__iff___autoParam___closed__4));
v___x_65_ = lean_box(2);
v___x_66_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_66_, 0, v___x_65_);
lean_ctor_set(v___x_66_, 1, v___x_64_);
lean_ctor_set(v___x_66_, 2, v___x_63_);
return v___x_66_;
}
}
static lean_object* _init_lp_mathlib_ClosureOperator_isClosed__iff___autoParam(void){
_start:
{
lean_object* v___x_67_; 
v___x_67_ = lean_obj_once(&lp_mathlib_ClosureOperator_isClosed__iff___autoParam___closed__25, &lp_mathlib_ClosureOperator_isClosed__iff___autoParam___closed__25_once, _init_lp_mathlib_ClosureOperator_isClosed__iff___autoParam___closed__25);
return v___x_67_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ClosureOperator_conjBy___redArg___lam__0(lean_object* v_c_68_, lean_object* v___y_69_){
_start:
{
lean_object* v___x_70_; 
v___x_70_ = lean_apply_1(v_c_68_, v___y_69_);
return v___x_70_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ClosureOperator_conjBy___redArg___lam__1(lean_object* v_toFun_71_, lean_object* v___f_72_, lean_object* v___y_73_){
_start:
{
lean_object* v___x_74_; 
v___x_74_ = lean_apply_2(v_toFun_71_, v___f_72_, v___y_73_);
return v___x_74_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ClosureOperator_conjBy___redArg(lean_object* v_c_75_, lean_object* v_e_76_){
_start:
{
lean_object* v___x_77_; lean_object* v_toFun_78_; lean_object* v___f_79_; lean_object* v___f_80_; 
v___x_77_ = lp_mathlib_OrderIso_conj___redArg(v_e_76_);
v_toFun_78_ = lean_ctor_get(v___x_77_, 0);
lean_inc(v_toFun_78_);
lean_dec_ref(v___x_77_);
v___f_79_ = lean_alloc_closure((void*)(lp_mathlib_ClosureOperator_conjBy___redArg___lam__0), 2, 1);
lean_closure_set(v___f_79_, 0, v_c_75_);
v___f_80_ = lean_alloc_closure((void*)(lp_mathlib_ClosureOperator_conjBy___redArg___lam__1), 3, 2);
lean_closure_set(v___f_80_, 0, v_toFun_78_);
lean_closure_set(v___f_80_, 1, v___f_79_);
return v___f_80_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ClosureOperator_conjBy(lean_object* v_00_u03b1_81_, lean_object* v_00_u03b2_82_, lean_object* v_inst_83_, lean_object* v_inst_84_, lean_object* v_c_85_, lean_object* v_e_86_){
_start:
{
lean_object* v___x_87_; 
v___x_87_ = lp_mathlib_ClosureOperator_conjBy___redArg(v_c_85_, v_e_86_);
return v___x_87_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ClosureOperator_conjBy___boxed(lean_object* v_00_u03b1_88_, lean_object* v_00_u03b2_89_, lean_object* v_inst_90_, lean_object* v_inst_91_, lean_object* v_c_92_, lean_object* v_e_93_){
_start:
{
lean_object* v_res_94_; 
v_res_94_ = lp_mathlib_ClosureOperator_conjBy(v_00_u03b1_88_, v_00_u03b2_89_, v_inst_90_, v_inst_91_, v_c_92_, v_e_93_);
lean_dec_ref(v_inst_91_);
lean_dec_ref(v_inst_90_);
return v_res_94_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ClosureOperator_id(lean_object* v_00_u03b1_96_, lean_object* v_inst_97_){
_start:
{
lean_object* v___x_98_; 
v___x_98_ = ((lean_object*)(lp_mathlib_ClosureOperator_id___closed__0));
return v___x_98_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ClosureOperator_id___boxed(lean_object* v_00_u03b1_99_, lean_object* v_inst_100_){
_start:
{
lean_object* v_res_101_; 
v_res_101_ = lp_mathlib_ClosureOperator_id(v_00_u03b1_99_, v_inst_100_);
lean_dec_ref(v_inst_100_);
return v_res_101_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ClosureOperator_instInhabited(lean_object* v_00_u03b1_102_, lean_object* v_inst_103_){
_start:
{
lean_object* v___x_104_; 
v___x_104_ = ((lean_object*)(lp_mathlib_ClosureOperator_id___closed__0));
return v___x_104_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ClosureOperator_instInhabited___boxed(lean_object* v_00_u03b1_105_, lean_object* v_inst_106_){
_start:
{
lean_object* v_res_107_; 
v_res_107_ = lp_mathlib_ClosureOperator_instInhabited(v_00_u03b1_105_, v_inst_106_);
lean_dec_ref(v_inst_106_);
return v_res_107_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ClosureOperator_toCloseds___redArg(lean_object* v_c_108_, lean_object* v_x_109_){
_start:
{
lean_object* v___x_110_; 
v___x_110_ = lean_apply_1(v_c_108_, v_x_109_);
return v___x_110_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ClosureOperator_toCloseds(lean_object* v_00_u03b1_111_, lean_object* v_inst_112_, lean_object* v_c_113_, lean_object* v_x_114_){
_start:
{
lean_object* v___x_115_; 
v___x_115_ = lean_apply_1(v_c_113_, v_x_114_);
return v___x_115_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ClosureOperator_toCloseds___boxed(lean_object* v_00_u03b1_116_, lean_object* v_inst_117_, lean_object* v_c_118_, lean_object* v_x_119_){
_start:
{
lean_object* v_res_120_; 
v_res_120_ = lp_mathlib_ClosureOperator_toCloseds(v_00_u03b1_116_, v_inst_117_, v_c_118_, v_x_119_);
lean_dec_ref(v_inst_117_);
return v_res_120_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ClosureOperator_mk_x27___redArg(lean_object* v_f_121_){
_start:
{
lean_inc(v_f_121_);
return v_f_121_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ClosureOperator_mk_x27___redArg___boxed(lean_object* v_f_122_){
_start:
{
lean_object* v_res_123_; 
v_res_123_ = lp_mathlib_ClosureOperator_mk_x27___redArg(v_f_122_);
lean_dec(v_f_122_);
return v_res_123_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ClosureOperator_mk_x27(lean_object* v_00_u03b1_124_, lean_object* v_inst_125_, lean_object* v_f_126_, lean_object* v_hf_u2081_127_, lean_object* v_hf_u2082_128_, lean_object* v_hf_u2083_129_){
_start:
{
lean_inc(v_f_126_);
return v_f_126_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ClosureOperator_mk_x27___boxed(lean_object* v_00_u03b1_130_, lean_object* v_inst_131_, lean_object* v_f_132_, lean_object* v_hf_u2081_133_, lean_object* v_hf_u2082_134_, lean_object* v_hf_u2083_135_){
_start:
{
lean_object* v_res_136_; 
v_res_136_ = lp_mathlib_ClosureOperator_mk_x27(v_00_u03b1_130_, v_inst_131_, v_f_132_, v_hf_u2081_133_, v_hf_u2082_134_, v_hf_u2083_135_);
lean_dec(v_f_132_);
lean_dec_ref(v_inst_131_);
return v_res_136_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ClosureOperator_mk_u2082___redArg(lean_object* v_f_137_){
_start:
{
lean_inc(v_f_137_);
return v_f_137_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ClosureOperator_mk_u2082___redArg___boxed(lean_object* v_f_138_){
_start:
{
lean_object* v_res_139_; 
v_res_139_ = lp_mathlib_ClosureOperator_mk_u2082___redArg(v_f_138_);
lean_dec(v_f_138_);
return v_res_139_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ClosureOperator_mk_u2082(lean_object* v_00_u03b1_140_, lean_object* v_inst_141_, lean_object* v_f_142_, lean_object* v_hf_143_, lean_object* v_hmin_144_){
_start:
{
lean_inc(v_f_142_);
return v_f_142_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ClosureOperator_mk_u2082___boxed(lean_object* v_00_u03b1_145_, lean_object* v_inst_146_, lean_object* v_f_147_, lean_object* v_hf_148_, lean_object* v_hmin_149_){
_start:
{
lean_object* v_res_150_; 
v_res_150_ = lp_mathlib_ClosureOperator_mk_u2082(v_00_u03b1_145_, v_inst_146_, v_f_147_, v_hf_148_, v_hmin_149_);
lean_dec(v_f_147_);
lean_dec_ref(v_inst_146_);
return v_res_150_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ClosureOperator_ofPred___redArg(lean_object* v_f_151_){
_start:
{
lean_inc(v_f_151_);
return v_f_151_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ClosureOperator_ofPred___redArg___boxed(lean_object* v_f_152_){
_start:
{
lean_object* v_res_153_; 
v_res_153_ = lp_mathlib_ClosureOperator_ofPred___redArg(v_f_152_);
lean_dec(v_f_152_);
return v_res_153_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ClosureOperator_ofPred(lean_object* v_00_u03b1_154_, lean_object* v_inst_155_, lean_object* v_f_156_, lean_object* v_p_157_, lean_object* v_hf_158_, lean_object* v_hfp_159_, lean_object* v_hmin_160_){
_start:
{
lean_inc(v_f_156_);
return v_f_156_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ClosureOperator_ofPred___boxed(lean_object* v_00_u03b1_161_, lean_object* v_inst_162_, lean_object* v_f_163_, lean_object* v_p_164_, lean_object* v_hf_165_, lean_object* v_hfp_166_, lean_object* v_hmin_167_){
_start:
{
lean_object* v_res_168_; 
v_res_168_ = lp_mathlib_ClosureOperator_ofPred(v_00_u03b1_161_, v_inst_162_, v_f_163_, v_p_164_, v_hf_165_, v_hfp_166_, v_hmin_167_);
lean_dec(v_f_163_);
lean_dec_ref(v_inst_162_);
return v_res_168_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ClosureOperator_ofCompletePred___redArg___lam__0(lean_object* v_toInfSet_169_, lean_object* v_a_170_){
_start:
{
lean_object* v___x_171_; 
v___x_171_ = lean_apply_1(v_toInfSet_169_, lean_box(0));
return v___x_171_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ClosureOperator_ofCompletePred___redArg___lam__0___boxed(lean_object* v_toInfSet_172_, lean_object* v_a_173_){
_start:
{
lean_object* v_res_174_; 
v_res_174_ = lp_mathlib_ClosureOperator_ofCompletePred___redArg___lam__0(v_toInfSet_172_, v_a_173_);
lean_dec(v_a_173_);
return v_res_174_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ClosureOperator_ofCompletePred___redArg(lean_object* v_inst_175_){
_start:
{
lean_object* v___x_176_; lean_object* v_toInfSet_177_; lean_object* v___f_178_; 
v___x_176_ = lp_mathlib_CompleteLattice_toCompleteSemilatticeInf___redArg(v_inst_175_);
v_toInfSet_177_ = lean_ctor_get(v___x_176_, 1);
lean_inc(v_toInfSet_177_);
lean_dec_ref(v___x_176_);
v___f_178_ = lean_alloc_closure((void*)(lp_mathlib_ClosureOperator_ofCompletePred___redArg___lam__0___boxed), 2, 1);
lean_closure_set(v___f_178_, 0, v_toInfSet_177_);
return v___f_178_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ClosureOperator_ofCompletePred(lean_object* v_00_u03b1_179_, lean_object* v_inst_180_, lean_object* v_p_181_, lean_object* v_hsinf_182_){
_start:
{
lean_object* v___x_183_; 
v___x_183_ = lp_mathlib_ClosureOperator_ofCompletePred___redArg(v_inst_180_);
return v___x_183_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderIso_equivClosureOperator___redArg___lam__0(lean_object* v_e_184_, lean_object* v_c_185_, lean_object* v___y_186_){
_start:
{
lean_object* v___x_22__overap_187_; lean_object* v___x_188_; 
v___x_22__overap_187_ = lp_mathlib_ClosureOperator_conjBy___redArg(v_c_185_, v_e_184_);
v___x_188_ = lean_apply_1(v___x_22__overap_187_, v___y_186_);
return v___x_188_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderIso_equivClosureOperator___redArg___lam__1(lean_object* v_e_189_, lean_object* v_c_190_, lean_object* v___y_191_){
_start:
{
lean_object* v___x_192_; lean_object* v___x_25__overap_193_; lean_object* v___x_194_; 
v___x_192_ = lp_mathlib_Equiv_symm___redArg(v_e_189_);
v___x_25__overap_193_ = lp_mathlib_ClosureOperator_conjBy___redArg(v_c_190_, v___x_192_);
v___x_194_ = lean_apply_1(v___x_25__overap_193_, v___y_191_);
return v___x_194_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderIso_equivClosureOperator___redArg(lean_object* v_e_195_){
_start:
{
lean_object* v___f_196_; lean_object* v___f_197_; lean_object* v___x_198_; 
lean_inc_ref(v_e_195_);
v___f_196_ = lean_alloc_closure((void*)(lp_mathlib_OrderIso_equivClosureOperator___redArg___lam__0), 3, 1);
lean_closure_set(v___f_196_, 0, v_e_195_);
v___f_197_ = lean_alloc_closure((void*)(lp_mathlib_OrderIso_equivClosureOperator___redArg___lam__1), 3, 1);
lean_closure_set(v___f_197_, 0, v_e_195_);
v___x_198_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_198_, 0, v___f_196_);
lean_ctor_set(v___x_198_, 1, v___f_197_);
return v___x_198_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderIso_equivClosureOperator(lean_object* v_00_u03b1_199_, lean_object* v_00_u03b2_200_, lean_object* v_inst_201_, lean_object* v_inst_202_, lean_object* v_e_203_){
_start:
{
lean_object* v___x_204_; 
v___x_204_ = lp_mathlib_OrderIso_equivClosureOperator___redArg(v_e_203_);
return v___x_204_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderIso_equivClosureOperator___boxed(lean_object* v_00_u03b1_205_, lean_object* v_00_u03b2_206_, lean_object* v_inst_207_, lean_object* v_inst_208_, lean_object* v_e_209_){
_start:
{
lean_object* v_res_210_; 
v_res_210_ = lp_mathlib_OrderIso_equivClosureOperator(v_00_u03b1_205_, v_00_u03b2_206_, v_inst_207_, v_inst_208_, v_e_209_);
lean_dec_ref(v_inst_208_);
lean_dec_ref(v_inst_207_);
return v_res_210_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LowerAdjoint_id___lam__0(lean_object* v_x_211_){
_start:
{
lean_inc(v_x_211_);
return v_x_211_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LowerAdjoint_id___lam__0___boxed(lean_object* v_x_212_){
_start:
{
lean_object* v_res_213_; 
v_res_213_ = lp_mathlib_LowerAdjoint_id___lam__0(v_x_212_);
lean_dec(v_x_212_);
return v_res_213_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LowerAdjoint_id(lean_object* v_00_u03b1_215_, lean_object* v_inst_216_){
_start:
{
lean_object* v___f_217_; 
v___f_217_ = ((lean_object*)(lp_mathlib_LowerAdjoint_id___closed__0));
return v___f_217_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LowerAdjoint_id___boxed(lean_object* v_00_u03b1_218_, lean_object* v_inst_219_){
_start:
{
lean_object* v_res_220_; 
v_res_220_ = lp_mathlib_LowerAdjoint_id(v_00_u03b1_218_, v_inst_219_);
lean_dec_ref(v_inst_219_);
return v_res_220_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LowerAdjoint_instInhabitedId(lean_object* v_00_u03b1_221_, lean_object* v_inst_222_){
_start:
{
lean_object* v___f_223_; 
v___f_223_ = ((lean_object*)(lp_mathlib_LowerAdjoint_id___closed__0));
return v___f_223_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LowerAdjoint_instInhabitedId___boxed(lean_object* v_00_u03b1_224_, lean_object* v_inst_225_){
_start:
{
lean_object* v_res_226_; 
v_res_226_ = lp_mathlib_LowerAdjoint_instInhabitedId(v_00_u03b1_224_, v_inst_225_);
lean_dec_ref(v_inst_225_);
return v_res_226_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LowerAdjoint_instCoeFunForall___lam__0(lean_object* v_self_227_, lean_object* v___y_228_){
_start:
{
lean_object* v___x_229_; 
v___x_229_ = lean_apply_1(v_self_227_, v___y_228_);
return v___x_229_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LowerAdjoint_instCoeFunForall(lean_object* v_00_u03b1_231_, lean_object* v_00_u03b2_232_, lean_object* v_inst_233_, lean_object* v_inst_234_, lean_object* v_u_235_){
_start:
{
lean_object* v___f_236_; 
v___f_236_ = ((lean_object*)(lp_mathlib_LowerAdjoint_instCoeFunForall___closed__0));
return v___f_236_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LowerAdjoint_instCoeFunForall___boxed(lean_object* v_00_u03b1_237_, lean_object* v_00_u03b2_238_, lean_object* v_inst_239_, lean_object* v_inst_240_, lean_object* v_u_241_){
_start:
{
lean_object* v_res_242_; 
v_res_242_ = lp_mathlib_LowerAdjoint_instCoeFunForall(v_00_u03b1_237_, v_00_u03b2_238_, v_inst_239_, v_inst_240_, v_u_241_);
lean_dec(v_u_241_);
lean_dec_ref(v_inst_240_);
lean_dec_ref(v_inst_239_);
return v_res_242_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LowerAdjoint_closureOperator___redArg___lam__0(lean_object* v_l_243_, lean_object* v_u_244_, lean_object* v_x_245_){
_start:
{
lean_object* v___x_246_; lean_object* v___x_247_; 
v___x_246_ = lean_apply_1(v_l_243_, v_x_245_);
v___x_247_ = lean_apply_1(v_u_244_, v___x_246_);
return v___x_247_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LowerAdjoint_closureOperator___redArg(lean_object* v_u_248_, lean_object* v_l_249_){
_start:
{
lean_object* v___f_250_; 
v___f_250_ = lean_alloc_closure((void*)(lp_mathlib_LowerAdjoint_closureOperator___redArg___lam__0), 3, 2);
lean_closure_set(v___f_250_, 0, v_l_249_);
lean_closure_set(v___f_250_, 1, v_u_248_);
return v___f_250_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LowerAdjoint_closureOperator(lean_object* v_00_u03b1_251_, lean_object* v_00_u03b2_252_, lean_object* v_inst_253_, lean_object* v_inst_254_, lean_object* v_u_255_, lean_object* v_l_256_){
_start:
{
lean_object* v___f_257_; 
v___f_257_ = lean_alloc_closure((void*)(lp_mathlib_LowerAdjoint_closureOperator___redArg___lam__0), 3, 2);
lean_closure_set(v___f_257_, 0, v_l_256_);
lean_closure_set(v___f_257_, 1, v_u_255_);
return v___f_257_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LowerAdjoint_closureOperator___boxed(lean_object* v_00_u03b1_258_, lean_object* v_00_u03b2_259_, lean_object* v_inst_260_, lean_object* v_inst_261_, lean_object* v_u_262_, lean_object* v_l_263_){
_start:
{
lean_object* v_res_264_; 
v_res_264_ = lp_mathlib_LowerAdjoint_closureOperator(v_00_u03b1_258_, v_00_u03b2_259_, v_inst_260_, v_inst_261_, v_u_262_, v_l_263_);
lean_dec_ref(v_inst_261_);
lean_dec_ref(v_inst_260_);
return v_res_264_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LowerAdjoint_toClosed___redArg(lean_object* v_u_265_, lean_object* v_l_266_, lean_object* v_x_267_){
_start:
{
lean_object* v___x_268_; lean_object* v___x_269_; 
v___x_268_ = lean_apply_1(v_l_266_, v_x_267_);
v___x_269_ = lean_apply_1(v_u_265_, v___x_268_);
return v___x_269_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LowerAdjoint_toClosed(lean_object* v_00_u03b1_270_, lean_object* v_00_u03b2_271_, lean_object* v_inst_272_, lean_object* v_inst_273_, lean_object* v_u_274_, lean_object* v_l_275_, lean_object* v_x_276_){
_start:
{
lean_object* v___x_277_; 
v___x_277_ = lp_mathlib_LowerAdjoint_toClosed___redArg(v_u_274_, v_l_275_, v_x_276_);
return v___x_277_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LowerAdjoint_toClosed___boxed(lean_object* v_00_u03b1_278_, lean_object* v_00_u03b2_279_, lean_object* v_inst_280_, lean_object* v_inst_281_, lean_object* v_u_282_, lean_object* v_l_283_, lean_object* v_x_284_){
_start:
{
lean_object* v_res_285_; 
v_res_285_ = lp_mathlib_LowerAdjoint_toClosed(v_00_u03b1_278_, v_00_u03b2_279_, v_inst_280_, v_inst_281_, v_u_282_, v_l_283_, v_x_284_);
lean_dec_ref(v_inst_281_);
lean_dec_ref(v_inst_280_);
return v_res_285_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_GaloisConnection_lowerAdjoint___redArg(lean_object* v_l_286_){
_start:
{
lean_inc(v_l_286_);
return v_l_286_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_GaloisConnection_lowerAdjoint___redArg___boxed(lean_object* v_l_287_){
_start:
{
lean_object* v_res_288_; 
v_res_288_ = lp_mathlib_GaloisConnection_lowerAdjoint___redArg(v_l_287_);
lean_dec(v_l_287_);
return v_res_288_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_GaloisConnection_lowerAdjoint(lean_object* v_00_u03b1_289_, lean_object* v_00_u03b2_290_, lean_object* v_inst_291_, lean_object* v_inst_292_, lean_object* v_l_293_, lean_object* v_u_294_, lean_object* v_gc_295_){
_start:
{
lean_inc(v_l_293_);
return v_l_293_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_GaloisConnection_lowerAdjoint___boxed(lean_object* v_00_u03b1_296_, lean_object* v_00_u03b2_297_, lean_object* v_inst_298_, lean_object* v_inst_299_, lean_object* v_l_300_, lean_object* v_u_301_, lean_object* v_gc_302_){
_start:
{
lean_object* v_res_303_; 
v_res_303_ = lp_mathlib_GaloisConnection_lowerAdjoint(v_00_u03b1_296_, v_00_u03b2_297_, v_inst_298_, v_inst_299_, v_l_300_, v_u_301_, v_gc_302_);
lean_dec(v_u_301_);
lean_dec(v_l_300_);
lean_dec_ref(v_inst_299_);
lean_dec_ref(v_inst_298_);
return v_res_303_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_GaloisConnection_closureOperator___redArg(lean_object* v_l_304_, lean_object* v_u_305_){
_start:
{
lean_object* v___f_306_; 
v___f_306_ = lean_alloc_closure((void*)(lp_mathlib_LowerAdjoint_closureOperator___redArg___lam__0), 3, 2);
lean_closure_set(v___f_306_, 0, v_l_304_);
lean_closure_set(v___f_306_, 1, v_u_305_);
return v___f_306_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_GaloisConnection_closureOperator(lean_object* v_00_u03b1_307_, lean_object* v_00_u03b2_308_, lean_object* v_inst_309_, lean_object* v_inst_310_, lean_object* v_l_311_, lean_object* v_u_312_, lean_object* v_gc_313_){
_start:
{
lean_object* v___f_314_; 
v___f_314_ = lean_alloc_closure((void*)(lp_mathlib_LowerAdjoint_closureOperator___redArg___lam__0), 3, 2);
lean_closure_set(v___f_314_, 0, v_l_311_);
lean_closure_set(v___f_314_, 1, v_u_312_);
return v___f_314_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_GaloisConnection_closureOperator___boxed(lean_object* v_00_u03b1_315_, lean_object* v_00_u03b2_316_, lean_object* v_inst_317_, lean_object* v_inst_318_, lean_object* v_l_319_, lean_object* v_u_320_, lean_object* v_gc_321_){
_start:
{
lean_object* v_res_322_; 
v_res_322_ = lp_mathlib_GaloisConnection_closureOperator(v_00_u03b1_315_, v_00_u03b2_316_, v_inst_317_, v_inst_318_, v_l_319_, v_u_320_, v_gc_321_);
lean_dec_ref(v_inst_318_);
lean_dec_ref(v_inst_317_);
return v_res_322_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ClosureOperator_gi___lam__0(lean_object* v_x_323_, lean_object* v_hx_324_){
_start:
{
lean_inc(v_x_323_);
return v_x_323_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ClosureOperator_gi___lam__0___boxed(lean_object* v_x_325_, lean_object* v_hx_326_){
_start:
{
lean_object* v_res_327_; 
v_res_327_ = lp_mathlib_ClosureOperator_gi___lam__0(v_x_325_, v_hx_326_);
lean_dec(v_x_325_);
return v_res_327_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ClosureOperator_gi(lean_object* v_00_u03b1_329_, lean_object* v_inst_330_, lean_object* v_c_331_){
_start:
{
lean_object* v___f_332_; 
v___f_332_ = ((lean_object*)(lp_mathlib_ClosureOperator_gi___closed__0));
return v___f_332_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ClosureOperator_gi___boxed(lean_object* v_00_u03b1_333_, lean_object* v_inst_334_, lean_object* v_c_335_){
_start:
{
lean_object* v_res_336_; 
v_res_336_ = lp_mathlib_ClosureOperator_gi(v_00_u03b1_333_, v_inst_334_, v_c_335_);
lean_dec(v_c_335_);
lean_dec_ref(v_inst_334_);
return v_res_336_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Data_Set_BooleanAlgebra(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Data_SetLike_Basic(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Order_Hom_Basic(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_Order_Closure(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Data_Set_BooleanAlgebra(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Data_SetLike_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Order_Hom_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_Order_Closure(uint8_t builtin) {
lean_object * res;
if (_G_meta_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_meta_initialized = true;
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
lp_mathlib_ClosureOperator_isClosed__iff___autoParam = _init_lp_mathlib_ClosureOperator_isClosed__iff___autoParam();
lean_mark_persistent(lp_mathlib_ClosureOperator_isClosed__iff___autoParam);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Data_Set_BooleanAlgebra(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Data_SetLike_Basic(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Order_Hom_Basic(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Order_Closure(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_Set_BooleanAlgebra(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_SetLike_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Order_Hom_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Order_Closure(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_Order_Closure(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_Order_Closure(builtin);
}
#ifdef __cplusplus
}
#endif
