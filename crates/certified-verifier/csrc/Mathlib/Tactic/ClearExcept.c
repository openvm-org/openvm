// Lean compiler output
// Module: Mathlib.Tactic.ClearExcept
// Imports: public import Init public meta import Init public import Mathlib.Init public meta import Lean.Elab.Tactic.ElabTerm
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
uint8_t lean_usize_dec_eq(size_t, size_t);
lean_object* lean_array_uget_borrowed(lean_object*, size_t);
uint8_t l_Lean_instBEqFVarId_beq(lean_object*, lean_object*);
size_t lean_usize_add(size_t, size_t);
lean_object* l_Lean_Name_mkStr1(lean_object*);
lean_object* lean_array_get_size(lean_object*);
uint8_t lean_nat_dec_lt(lean_object*, lean_object*);
size_t lean_usize_of_nat(lean_object*);
extern lean_object* l_Lean_Elab_unsupportedSyntaxExceptionId;
uint8_t lean_usize_dec_lt(size_t, size_t);
lean_object* l_Lean_LocalDecl_fvarId(lean_object*);
lean_object* l_Lean_LocalDecl_type(lean_object*);
lean_object* l_Lean_Meta_isClass_x3f(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lean_array_push(lean_object*, lean_object*);
uint8_t l_Lean_LocalDecl_isAuxDecl(lean_object*);
size_t lean_array_size(lean_object*);
lean_object* l_Lean_Elab_Tactic_getMainGoal___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lean_mk_empty_array_with_capacity(lean_object*);
lean_object* l_Lean_MVarId_tryClearMany(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Elab_Tactic_replaceMainGoal___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Name_mkStr4(lean_object*, lean_object*, lean_object*, lean_object*);
uint8_t l_Lean_Syntax_isOfKind(lean_object*, lean_object*);
lean_object* l_Lean_Syntax_getArg(lean_object*, lean_object*);
lean_object* l_Lean_Syntax_getArgs(lean_object*);
lean_object* l_Lean_Elab_Tactic_getFVarIds(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Elab_Tactic_withMainContext___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib___private_Init_Data_Array_Basic_0__Array_anyMUnsafe_any___at___00Array_contains___at___00Lean_Elab_Tactic_getVarsToClear_spec__0_spec__0(lean_object*, lean_object*, size_t, size_t);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_anyMUnsafe_any___at___00Array_contains___at___00Lean_Elab_Tactic_getVarsToClear_spec__0_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Array_contains___at___00Lean_Elab_Tactic_getVarsToClear_spec__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Array_contains___at___00Lean_Elab_Tactic_getVarsToClear_spec__0___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00Lean_Elab_Tactic_getVarsToClear_spec__1_spec__3_spec__6(lean_object*, lean_object*, size_t, size_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00Lean_Elab_Tactic_getVarsToClear_spec__1_spec__3_spec__6___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00Lean_Elab_Tactic_getVarsToClear_spec__1_spec__3(lean_object*, lean_object*, size_t, size_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00Lean_Elab_Tactic_getVarsToClear_spec__1_spec__3___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00Lean_Elab_Tactic_getVarsToClear_spec__1_spec__2_spec__4_spec__5(lean_object*, lean_object*, size_t, size_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00Lean_Elab_Tactic_getVarsToClear_spec__1_spec__2_spec__4_spec__5___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00Lean_Elab_Tactic_getVarsToClear_spec__1_spec__2_spec__4(lean_object*, lean_object*, size_t, size_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00Lean_Elab_Tactic_getVarsToClear_spec__1_spec__2_spec__4___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00Lean_Elab_Tactic_getVarsToClear_spec__1_spec__2(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00Lean_Elab_Tactic_getVarsToClear_spec__1_spec__2_spec__3(lean_object*, lean_object*, lean_object*, size_t, size_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00Lean_Elab_Tactic_getVarsToClear_spec__1_spec__2_spec__3___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00Lean_Elab_Tactic_getVarsToClear_spec__1_spec__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentArray_forIn___at___00Lean_Elab_Tactic_getVarsToClear_spec__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentArray_forIn___at___00Lean_Elab_Tactic_getVarsToClear_spec__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_array_object lp_mathlib_Lean_Elab_Tactic_getVarsToClear___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_array_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 246}, .m_size = 0, .m_capacity = 0, .m_data = {}};
static const lean_object* lp_mathlib_Lean_Elab_Tactic_getVarsToClear___closed__0 = (const lean_object*)&lp_mathlib_Lean_Elab_Tactic_getVarsToClear___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_Tactic_getVarsToClear(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_Tactic_getVarsToClear___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_Tactic_clearExcept(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_Tactic_clearExcept___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Lean_Elab_Tactic_clearExceptTactic___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "Lean"};
static const lean_object* lp_mathlib_Lean_Elab_Tactic_clearExceptTactic___closed__0 = (const lean_object*)&lp_mathlib_Lean_Elab_Tactic_clearExceptTactic___closed__0_value;
static const lean_string_object lp_mathlib_Lean_Elab_Tactic_clearExceptTactic___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "Elab"};
static const lean_object* lp_mathlib_Lean_Elab_Tactic_clearExceptTactic___closed__1 = (const lean_object*)&lp_mathlib_Lean_Elab_Tactic_clearExceptTactic___closed__1_value;
static const lean_string_object lp_mathlib_Lean_Elab_Tactic_clearExceptTactic___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "Tactic"};
static const lean_object* lp_mathlib_Lean_Elab_Tactic_clearExceptTactic___closed__2 = (const lean_object*)&lp_mathlib_Lean_Elab_Tactic_clearExceptTactic___closed__2_value;
static const lean_string_object lp_mathlib_Lean_Elab_Tactic_clearExceptTactic___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 18, .m_capacity = 18, .m_length = 17, .m_data = "clearExceptTactic"};
static const lean_object* lp_mathlib_Lean_Elab_Tactic_clearExceptTactic___closed__3 = (const lean_object*)&lp_mathlib_Lean_Elab_Tactic_clearExceptTactic___closed__3_value;
static const lean_ctor_object lp_mathlib_Lean_Elab_Tactic_clearExceptTactic___closed__4_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Lean_Elab_Tactic_clearExceptTactic___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib_Lean_Elab_Tactic_clearExceptTactic___closed__4_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Lean_Elab_Tactic_clearExceptTactic___closed__4_value_aux_0),((lean_object*)&lp_mathlib_Lean_Elab_Tactic_clearExceptTactic___closed__1_value),LEAN_SCALAR_PTR_LITERAL(52, 247, 248, 201, 92, 23, 188, 159)}};
static const lean_ctor_object lp_mathlib_Lean_Elab_Tactic_clearExceptTactic___closed__4_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Lean_Elab_Tactic_clearExceptTactic___closed__4_value_aux_1),((lean_object*)&lp_mathlib_Lean_Elab_Tactic_clearExceptTactic___closed__2_value),LEAN_SCALAR_PTR_LITERAL(161, 230, 229, 85, 182, 144, 182, 176)}};
static const lean_ctor_object lp_mathlib_Lean_Elab_Tactic_clearExceptTactic___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Lean_Elab_Tactic_clearExceptTactic___closed__4_value_aux_2),((lean_object*)&lp_mathlib_Lean_Elab_Tactic_clearExceptTactic___closed__3_value),LEAN_SCALAR_PTR_LITERAL(24, 78, 116, 41, 162, 48, 98, 85)}};
static const lean_object* lp_mathlib_Lean_Elab_Tactic_clearExceptTactic___closed__4 = (const lean_object*)&lp_mathlib_Lean_Elab_Tactic_clearExceptTactic___closed__4_value;
static const lean_string_object lp_mathlib_Lean_Elab_Tactic_clearExceptTactic___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "andthen"};
static const lean_object* lp_mathlib_Lean_Elab_Tactic_clearExceptTactic___closed__5 = (const lean_object*)&lp_mathlib_Lean_Elab_Tactic_clearExceptTactic___closed__5_value;
static const lean_ctor_object lp_mathlib_Lean_Elab_Tactic_clearExceptTactic___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Lean_Elab_Tactic_clearExceptTactic___closed__5_value),LEAN_SCALAR_PTR_LITERAL(40, 255, 78, 30, 143, 119, 117, 174)}};
static const lean_object* lp_mathlib_Lean_Elab_Tactic_clearExceptTactic___closed__6 = (const lean_object*)&lp_mathlib_Lean_Elab_Tactic_clearExceptTactic___closed__6_value;
static const lean_string_object lp_mathlib_Lean_Elab_Tactic_clearExceptTactic___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "clear "};
static const lean_object* lp_mathlib_Lean_Elab_Tactic_clearExceptTactic___closed__7 = (const lean_object*)&lp_mathlib_Lean_Elab_Tactic_clearExceptTactic___closed__7_value;
static const lean_ctor_object lp_mathlib_Lean_Elab_Tactic_clearExceptTactic___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 8, .m_other = 1, .m_tag = 6}, .m_objs = {((lean_object*)&lp_mathlib_Lean_Elab_Tactic_clearExceptTactic___closed__7_value),LEAN_SCALAR_PTR_LITERAL(0, 0, 0, 0, 0, 0, 0, 0)}};
static const lean_object* lp_mathlib_Lean_Elab_Tactic_clearExceptTactic___closed__8 = (const lean_object*)&lp_mathlib_Lean_Elab_Tactic_clearExceptTactic___closed__8_value;
static const lean_string_object lp_mathlib_Lean_Elab_Tactic_clearExceptTactic___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = "*"};
static const lean_object* lp_mathlib_Lean_Elab_Tactic_clearExceptTactic___closed__9 = (const lean_object*)&lp_mathlib_Lean_Elab_Tactic_clearExceptTactic___closed__9_value;
static const lean_ctor_object lp_mathlib_Lean_Elab_Tactic_clearExceptTactic___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 5}, .m_objs = {((lean_object*)&lp_mathlib_Lean_Elab_Tactic_clearExceptTactic___closed__9_value)}};
static const lean_object* lp_mathlib_Lean_Elab_Tactic_clearExceptTactic___closed__10 = (const lean_object*)&lp_mathlib_Lean_Elab_Tactic_clearExceptTactic___closed__10_value;
static const lean_ctor_object lp_mathlib_Lean_Elab_Tactic_clearExceptTactic___closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib_Lean_Elab_Tactic_clearExceptTactic___closed__6_value),((lean_object*)&lp_mathlib_Lean_Elab_Tactic_clearExceptTactic___closed__8_value),((lean_object*)&lp_mathlib_Lean_Elab_Tactic_clearExceptTactic___closed__10_value)}};
static const lean_object* lp_mathlib_Lean_Elab_Tactic_clearExceptTactic___closed__11 = (const lean_object*)&lp_mathlib_Lean_Elab_Tactic_clearExceptTactic___closed__11_value;
static const lean_string_object lp_mathlib_Lean_Elab_Tactic_clearExceptTactic___closed__12_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 3, .m_capacity = 3, .m_length = 2, .m_data = " -"};
static const lean_object* lp_mathlib_Lean_Elab_Tactic_clearExceptTactic___closed__12 = (const lean_object*)&lp_mathlib_Lean_Elab_Tactic_clearExceptTactic___closed__12_value;
static const lean_ctor_object lp_mathlib_Lean_Elab_Tactic_clearExceptTactic___closed__13_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 5}, .m_objs = {((lean_object*)&lp_mathlib_Lean_Elab_Tactic_clearExceptTactic___closed__12_value)}};
static const lean_object* lp_mathlib_Lean_Elab_Tactic_clearExceptTactic___closed__13 = (const lean_object*)&lp_mathlib_Lean_Elab_Tactic_clearExceptTactic___closed__13_value;
static const lean_ctor_object lp_mathlib_Lean_Elab_Tactic_clearExceptTactic___closed__14_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib_Lean_Elab_Tactic_clearExceptTactic___closed__6_value),((lean_object*)&lp_mathlib_Lean_Elab_Tactic_clearExceptTactic___closed__11_value),((lean_object*)&lp_mathlib_Lean_Elab_Tactic_clearExceptTactic___closed__13_value)}};
static const lean_object* lp_mathlib_Lean_Elab_Tactic_clearExceptTactic___closed__14 = (const lean_object*)&lp_mathlib_Lean_Elab_Tactic_clearExceptTactic___closed__14_value;
static const lean_string_object lp_mathlib_Lean_Elab_Tactic_clearExceptTactic___closed__15_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "many"};
static const lean_object* lp_mathlib_Lean_Elab_Tactic_clearExceptTactic___closed__15 = (const lean_object*)&lp_mathlib_Lean_Elab_Tactic_clearExceptTactic___closed__15_value;
static const lean_ctor_object lp_mathlib_Lean_Elab_Tactic_clearExceptTactic___closed__16_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Lean_Elab_Tactic_clearExceptTactic___closed__15_value),LEAN_SCALAR_PTR_LITERAL(41, 35, 40, 86, 189, 97, 244, 31)}};
static const lean_object* lp_mathlib_Lean_Elab_Tactic_clearExceptTactic___closed__16 = (const lean_object*)&lp_mathlib_Lean_Elab_Tactic_clearExceptTactic___closed__16_value;
static const lean_string_object lp_mathlib_Lean_Elab_Tactic_clearExceptTactic___closed__17_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "ppSpace"};
static const lean_object* lp_mathlib_Lean_Elab_Tactic_clearExceptTactic___closed__17 = (const lean_object*)&lp_mathlib_Lean_Elab_Tactic_clearExceptTactic___closed__17_value;
static const lean_ctor_object lp_mathlib_Lean_Elab_Tactic_clearExceptTactic___closed__18_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Lean_Elab_Tactic_clearExceptTactic___closed__17_value),LEAN_SCALAR_PTR_LITERAL(207, 47, 58, 43, 30, 240, 125, 246)}};
static const lean_object* lp_mathlib_Lean_Elab_Tactic_clearExceptTactic___closed__18 = (const lean_object*)&lp_mathlib_Lean_Elab_Tactic_clearExceptTactic___closed__18_value;
static const lean_ctor_object lp_mathlib_Lean_Elab_Tactic_clearExceptTactic___closed__19_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_Lean_Elab_Tactic_clearExceptTactic___closed__18_value)}};
static const lean_object* lp_mathlib_Lean_Elab_Tactic_clearExceptTactic___closed__19 = (const lean_object*)&lp_mathlib_Lean_Elab_Tactic_clearExceptTactic___closed__19_value;
static const lean_string_object lp_mathlib_Lean_Elab_Tactic_clearExceptTactic___closed__20_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "colGt"};
static const lean_object* lp_mathlib_Lean_Elab_Tactic_clearExceptTactic___closed__20 = (const lean_object*)&lp_mathlib_Lean_Elab_Tactic_clearExceptTactic___closed__20_value;
static const lean_ctor_object lp_mathlib_Lean_Elab_Tactic_clearExceptTactic___closed__21_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Lean_Elab_Tactic_clearExceptTactic___closed__20_value),LEAN_SCALAR_PTR_LITERAL(185, 236, 32, 153, 169, 213, 53, 244)}};
static const lean_object* lp_mathlib_Lean_Elab_Tactic_clearExceptTactic___closed__21 = (const lean_object*)&lp_mathlib_Lean_Elab_Tactic_clearExceptTactic___closed__21_value;
static const lean_ctor_object lp_mathlib_Lean_Elab_Tactic_clearExceptTactic___closed__22_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_Lean_Elab_Tactic_clearExceptTactic___closed__21_value)}};
static const lean_object* lp_mathlib_Lean_Elab_Tactic_clearExceptTactic___closed__22 = (const lean_object*)&lp_mathlib_Lean_Elab_Tactic_clearExceptTactic___closed__22_value;
static const lean_ctor_object lp_mathlib_Lean_Elab_Tactic_clearExceptTactic___closed__23_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib_Lean_Elab_Tactic_clearExceptTactic___closed__6_value),((lean_object*)&lp_mathlib_Lean_Elab_Tactic_clearExceptTactic___closed__19_value),((lean_object*)&lp_mathlib_Lean_Elab_Tactic_clearExceptTactic___closed__22_value)}};
static const lean_object* lp_mathlib_Lean_Elab_Tactic_clearExceptTactic___closed__23 = (const lean_object*)&lp_mathlib_Lean_Elab_Tactic_clearExceptTactic___closed__23_value;
static const lean_string_object lp_mathlib_Lean_Elab_Tactic_clearExceptTactic___closed__24_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "ident"};
static const lean_object* lp_mathlib_Lean_Elab_Tactic_clearExceptTactic___closed__24 = (const lean_object*)&lp_mathlib_Lean_Elab_Tactic_clearExceptTactic___closed__24_value;
static const lean_ctor_object lp_mathlib_Lean_Elab_Tactic_clearExceptTactic___closed__25_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Lean_Elab_Tactic_clearExceptTactic___closed__24_value),LEAN_SCALAR_PTR_LITERAL(52, 159, 208, 51, 14, 60, 6, 71)}};
static const lean_object* lp_mathlib_Lean_Elab_Tactic_clearExceptTactic___closed__25 = (const lean_object*)&lp_mathlib_Lean_Elab_Tactic_clearExceptTactic___closed__25_value;
static const lean_ctor_object lp_mathlib_Lean_Elab_Tactic_clearExceptTactic___closed__26_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_Lean_Elab_Tactic_clearExceptTactic___closed__25_value)}};
static const lean_object* lp_mathlib_Lean_Elab_Tactic_clearExceptTactic___closed__26 = (const lean_object*)&lp_mathlib_Lean_Elab_Tactic_clearExceptTactic___closed__26_value;
static const lean_ctor_object lp_mathlib_Lean_Elab_Tactic_clearExceptTactic___closed__27_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib_Lean_Elab_Tactic_clearExceptTactic___closed__6_value),((lean_object*)&lp_mathlib_Lean_Elab_Tactic_clearExceptTactic___closed__23_value),((lean_object*)&lp_mathlib_Lean_Elab_Tactic_clearExceptTactic___closed__26_value)}};
static const lean_object* lp_mathlib_Lean_Elab_Tactic_clearExceptTactic___closed__27 = (const lean_object*)&lp_mathlib_Lean_Elab_Tactic_clearExceptTactic___closed__27_value;
static const lean_ctor_object lp_mathlib_Lean_Elab_Tactic_clearExceptTactic___closed__28_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Lean_Elab_Tactic_clearExceptTactic___closed__16_value),((lean_object*)&lp_mathlib_Lean_Elab_Tactic_clearExceptTactic___closed__27_value)}};
static const lean_object* lp_mathlib_Lean_Elab_Tactic_clearExceptTactic___closed__28 = (const lean_object*)&lp_mathlib_Lean_Elab_Tactic_clearExceptTactic___closed__28_value;
static const lean_ctor_object lp_mathlib_Lean_Elab_Tactic_clearExceptTactic___closed__29_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib_Lean_Elab_Tactic_clearExceptTactic___closed__6_value),((lean_object*)&lp_mathlib_Lean_Elab_Tactic_clearExceptTactic___closed__14_value),((lean_object*)&lp_mathlib_Lean_Elab_Tactic_clearExceptTactic___closed__28_value)}};
static const lean_object* lp_mathlib_Lean_Elab_Tactic_clearExceptTactic___closed__29 = (const lean_object*)&lp_mathlib_Lean_Elab_Tactic_clearExceptTactic___closed__29_value;
static const lean_ctor_object lp_mathlib_Lean_Elab_Tactic_clearExceptTactic___closed__30_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 3}, .m_objs = {((lean_object*)&lp_mathlib_Lean_Elab_Tactic_clearExceptTactic___closed__4_value),((lean_object*)(((size_t)(1022) << 1) | 1)),((lean_object*)&lp_mathlib_Lean_Elab_Tactic_clearExceptTactic___closed__29_value)}};
static const lean_object* lp_mathlib_Lean_Elab_Tactic_clearExceptTactic___closed__30 = (const lean_object*)&lp_mathlib_Lean_Elab_Tactic_clearExceptTactic___closed__30_value;
LEAN_EXPORT const lean_object* lp_mathlib_Lean_Elab_Tactic_clearExceptTactic = (const lean_object*)&lp_mathlib_Lean_Elab_Tactic_clearExceptTactic___closed__30_value;
static lean_once_cell_t lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Lean_Elab_Tactic___aux__Mathlib__Tactic__ClearExcept______elabRules__Lean__Elab__Tactic__clearExceptTactic__1_spec__0___redArg___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Lean_Elab_Tactic___aux__Mathlib__Tactic__ClearExcept______elabRules__Lean__Elab__Tactic__clearExceptTactic__1_spec__0___redArg___closed__0;
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Lean_Elab_Tactic___aux__Mathlib__Tactic__ClearExcept______elabRules__Lean__Elab__Tactic__clearExceptTactic__1_spec__0___redArg();
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Lean_Elab_Tactic___aux__Mathlib__Tactic__ClearExcept______elabRules__Lean__Elab__Tactic__clearExceptTactic__1_spec__0___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Lean_Elab_Tactic___aux__Mathlib__Tactic__ClearExcept______elabRules__Lean__Elab__Tactic__clearExceptTactic__1_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Lean_Elab_Tactic___aux__Mathlib__Tactic__ClearExcept______elabRules__Lean__Elab__Tactic__clearExceptTactic__1_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_Tactic___aux__Mathlib__Tactic__ClearExcept______elabRules__Lean__Elab__Tactic__clearExceptTactic__1___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_Tactic___aux__Mathlib__Tactic__ClearExcept______elabRules__Lean__Elab__Tactic__clearExceptTactic__1___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_Tactic___aux__Mathlib__Tactic__ClearExcept______elabRules__Lean__Elab__Tactic__clearExceptTactic__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_Tactic___aux__Mathlib__Tactic__ClearExcept______elabRules__Lean__Elab__Tactic__clearExceptTactic__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib___private_Init_Data_Array_Basic_0__Array_anyMUnsafe_any___at___00Array_contains___at___00Lean_Elab_Tactic_getVarsToClear_spec__0_spec__0(lean_object* v_a_1_, lean_object* v_as_2_, size_t v_i_3_, size_t v_stop_4_){
_start:
{
uint8_t v___x_5_; 
v___x_5_ = lean_usize_dec_eq(v_i_3_, v_stop_4_);
if (v___x_5_ == 0)
{
lean_object* v___x_6_; uint8_t v___x_7_; 
v___x_6_ = lean_array_uget_borrowed(v_as_2_, v_i_3_);
v___x_7_ = l_Lean_instBEqFVarId_beq(v_a_1_, v___x_6_);
if (v___x_7_ == 0)
{
size_t v___x_8_; size_t v___x_9_; 
v___x_8_ = ((size_t)1ULL);
v___x_9_ = lean_usize_add(v_i_3_, v___x_8_);
v_i_3_ = v___x_9_;
goto _start;
}
else
{
return v___x_7_;
}
}
else
{
uint8_t v___x_11_; 
v___x_11_ = 0;
return v___x_11_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_anyMUnsafe_any___at___00Array_contains___at___00Lean_Elab_Tactic_getVarsToClear_spec__0_spec__0___boxed(lean_object* v_a_12_, lean_object* v_as_13_, lean_object* v_i_14_, lean_object* v_stop_15_){
_start:
{
size_t v_i_boxed_16_; size_t v_stop_boxed_17_; uint8_t v_res_18_; lean_object* v_r_19_; 
v_i_boxed_16_ = lean_unbox_usize(v_i_14_);
lean_dec(v_i_14_);
v_stop_boxed_17_ = lean_unbox_usize(v_stop_15_);
lean_dec(v_stop_15_);
v_res_18_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_anyMUnsafe_any___at___00Array_contains___at___00Lean_Elab_Tactic_getVarsToClear_spec__0_spec__0(v_a_12_, v_as_13_, v_i_boxed_16_, v_stop_boxed_17_);
lean_dec_ref(v_as_13_);
lean_dec(v_a_12_);
v_r_19_ = lean_box(v_res_18_);
return v_r_19_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Array_contains___at___00Lean_Elab_Tactic_getVarsToClear_spec__0(lean_object* v_as_20_, lean_object* v_a_21_){
_start:
{
lean_object* v___x_22_; lean_object* v___x_23_; uint8_t v___x_24_; 
v___x_22_ = lean_unsigned_to_nat(0u);
v___x_23_ = lean_array_get_size(v_as_20_);
v___x_24_ = lean_nat_dec_lt(v___x_22_, v___x_23_);
if (v___x_24_ == 0)
{
return v___x_24_;
}
else
{
if (v___x_24_ == 0)
{
return v___x_24_;
}
else
{
size_t v___x_25_; size_t v___x_26_; uint8_t v___x_27_; 
v___x_25_ = ((size_t)0ULL);
v___x_26_ = lean_usize_of_nat(v___x_23_);
v___x_27_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_anyMUnsafe_any___at___00Array_contains___at___00Lean_Elab_Tactic_getVarsToClear_spec__0_spec__0(v_a_21_, v_as_20_, v___x_25_, v___x_26_);
return v___x_27_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Array_contains___at___00Lean_Elab_Tactic_getVarsToClear_spec__0___boxed(lean_object* v_as_28_, lean_object* v_a_29_){
_start:
{
uint8_t v_res_30_; lean_object* v_r_31_; 
v_res_30_ = lp_mathlib_Array_contains___at___00Lean_Elab_Tactic_getVarsToClear_spec__0(v_as_28_, v_a_29_);
lean_dec(v_a_29_);
lean_dec_ref(v_as_28_);
v_r_31_ = lean_box(v_res_30_);
return v_r_31_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00Lean_Elab_Tactic_getVarsToClear_spec__1_spec__3_spec__6(lean_object* v_preserve_32_, lean_object* v_as_33_, size_t v_sz_34_, size_t v_i_35_, lean_object* v_b_36_, lean_object* v___y_37_, lean_object* v___y_38_, lean_object* v___y_39_, lean_object* v___y_40_){
_start:
{
uint8_t v___x_42_; 
v___x_42_ = lean_usize_dec_lt(v_i_35_, v_sz_34_);
if (v___x_42_ == 0)
{
lean_object* v___x_43_; 
v___x_43_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_43_, 0, v_b_36_);
return v___x_43_;
}
else
{
lean_object* v_snd_44_; lean_object* v___x_46_; uint8_t v_isShared_47_; uint8_t v_isSharedCheck_76_; 
v_snd_44_ = lean_ctor_get(v_b_36_, 1);
v_isSharedCheck_76_ = !lean_is_exclusive(v_b_36_);
if (v_isSharedCheck_76_ == 0)
{
lean_object* v_unused_77_; 
v_unused_77_ = lean_ctor_get(v_b_36_, 0);
lean_dec(v_unused_77_);
v___x_46_ = v_b_36_;
v_isShared_47_ = v_isSharedCheck_76_;
goto v_resetjp_45_;
}
else
{
lean_inc(v_snd_44_);
lean_dec(v_b_36_);
v___x_46_ = lean_box(0);
v_isShared_47_ = v_isSharedCheck_76_;
goto v_resetjp_45_;
}
v_resetjp_45_:
{
lean_object* v___x_48_; lean_object* v_a_50_; lean_object* v_a_57_; 
v___x_48_ = lean_box(0);
v_a_57_ = lean_array_uget_borrowed(v_as_33_, v_i_35_);
if (lean_obj_tag(v_a_57_) == 0)
{
v_a_50_ = v_snd_44_;
goto v___jp_49_;
}
else
{
lean_object* v_val_58_; lean_object* v___x_59_; uint8_t v___y_61_; uint8_t v___x_74_; 
v_val_58_ = lean_ctor_get(v_a_57_, 0);
v___x_59_ = l_Lean_LocalDecl_fvarId(v_val_58_);
v___x_74_ = lp_mathlib_Array_contains___at___00Lean_Elab_Tactic_getVarsToClear_spec__0(v_preserve_32_, v___x_59_);
if (v___x_74_ == 0)
{
uint8_t v___x_75_; 
v___x_75_ = l_Lean_LocalDecl_isAuxDecl(v_val_58_);
v___y_61_ = v___x_75_;
goto v___jp_60_;
}
else
{
v___y_61_ = v___x_74_;
goto v___jp_60_;
}
v___jp_60_:
{
if (v___y_61_ == 0)
{
lean_object* v___x_62_; lean_object* v___x_63_; 
v___x_62_ = l_Lean_LocalDecl_type(v_val_58_);
v___x_63_ = l_Lean_Meta_isClass_x3f(v___x_62_, v___y_37_, v___y_38_, v___y_39_, v___y_40_);
if (lean_obj_tag(v___x_63_) == 0)
{
lean_object* v_a_64_; 
v_a_64_ = lean_ctor_get(v___x_63_, 0);
lean_inc(v_a_64_);
lean_dec_ref_known(v___x_63_, 1);
if (lean_obj_tag(v_a_64_) == 0)
{
lean_object* v___x_65_; 
v___x_65_ = lean_array_push(v_snd_44_, v___x_59_);
v_a_50_ = v___x_65_;
goto v___jp_49_;
}
else
{
lean_dec(v_a_64_);
lean_dec(v___x_59_);
v_a_50_ = v_snd_44_;
goto v___jp_49_;
}
}
else
{
lean_object* v_a_66_; lean_object* v___x_68_; uint8_t v_isShared_69_; uint8_t v_isSharedCheck_73_; 
lean_dec(v___x_59_);
lean_del_object(v___x_46_);
lean_dec(v_snd_44_);
v_a_66_ = lean_ctor_get(v___x_63_, 0);
v_isSharedCheck_73_ = !lean_is_exclusive(v___x_63_);
if (v_isSharedCheck_73_ == 0)
{
v___x_68_ = v___x_63_;
v_isShared_69_ = v_isSharedCheck_73_;
goto v_resetjp_67_;
}
else
{
lean_inc(v_a_66_);
lean_dec(v___x_63_);
v___x_68_ = lean_box(0);
v_isShared_69_ = v_isSharedCheck_73_;
goto v_resetjp_67_;
}
v_resetjp_67_:
{
lean_object* v___x_71_; 
if (v_isShared_69_ == 0)
{
v___x_71_ = v___x_68_;
goto v_reusejp_70_;
}
else
{
lean_object* v_reuseFailAlloc_72_; 
v_reuseFailAlloc_72_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_72_, 0, v_a_66_);
v___x_71_ = v_reuseFailAlloc_72_;
goto v_reusejp_70_;
}
v_reusejp_70_:
{
return v___x_71_;
}
}
}
}
else
{
lean_dec(v___x_59_);
v_a_50_ = v_snd_44_;
goto v___jp_49_;
}
}
}
v___jp_49_:
{
lean_object* v___x_52_; 
if (v_isShared_47_ == 0)
{
lean_ctor_set(v___x_46_, 1, v_a_50_);
lean_ctor_set(v___x_46_, 0, v___x_48_);
v___x_52_ = v___x_46_;
goto v_reusejp_51_;
}
else
{
lean_object* v_reuseFailAlloc_56_; 
v_reuseFailAlloc_56_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_56_, 0, v___x_48_);
lean_ctor_set(v_reuseFailAlloc_56_, 1, v_a_50_);
v___x_52_ = v_reuseFailAlloc_56_;
goto v_reusejp_51_;
}
v_reusejp_51_:
{
size_t v___x_53_; size_t v___x_54_; 
v___x_53_ = ((size_t)1ULL);
v___x_54_ = lean_usize_add(v_i_35_, v___x_53_);
v_i_35_ = v___x_54_;
v_b_36_ = v___x_52_;
goto _start;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00Lean_Elab_Tactic_getVarsToClear_spec__1_spec__3_spec__6___boxed(lean_object* v_preserve_78_, lean_object* v_as_79_, lean_object* v_sz_80_, lean_object* v_i_81_, lean_object* v_b_82_, lean_object* v___y_83_, lean_object* v___y_84_, lean_object* v___y_85_, lean_object* v___y_86_, lean_object* v___y_87_){
_start:
{
size_t v_sz_boxed_88_; size_t v_i_boxed_89_; lean_object* v_res_90_; 
v_sz_boxed_88_ = lean_unbox_usize(v_sz_80_);
lean_dec(v_sz_80_);
v_i_boxed_89_ = lean_unbox_usize(v_i_81_);
lean_dec(v_i_81_);
v_res_90_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00Lean_Elab_Tactic_getVarsToClear_spec__1_spec__3_spec__6(v_preserve_78_, v_as_79_, v_sz_boxed_88_, v_i_boxed_89_, v_b_82_, v___y_83_, v___y_84_, v___y_85_, v___y_86_);
lean_dec(v___y_86_);
lean_dec_ref(v___y_85_);
lean_dec(v___y_84_);
lean_dec_ref(v___y_83_);
lean_dec_ref(v_as_79_);
lean_dec_ref(v_preserve_78_);
return v_res_90_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00Lean_Elab_Tactic_getVarsToClear_spec__1_spec__3(lean_object* v_preserve_91_, lean_object* v_as_92_, size_t v_sz_93_, size_t v_i_94_, lean_object* v_b_95_, lean_object* v___y_96_, lean_object* v___y_97_, lean_object* v___y_98_, lean_object* v___y_99_){
_start:
{
uint8_t v___x_101_; 
v___x_101_ = lean_usize_dec_lt(v_i_94_, v_sz_93_);
if (v___x_101_ == 0)
{
lean_object* v___x_102_; 
v___x_102_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_102_, 0, v_b_95_);
return v___x_102_;
}
else
{
lean_object* v_snd_103_; lean_object* v___x_105_; uint8_t v_isShared_106_; uint8_t v_isSharedCheck_135_; 
v_snd_103_ = lean_ctor_get(v_b_95_, 1);
v_isSharedCheck_135_ = !lean_is_exclusive(v_b_95_);
if (v_isSharedCheck_135_ == 0)
{
lean_object* v_unused_136_; 
v_unused_136_ = lean_ctor_get(v_b_95_, 0);
lean_dec(v_unused_136_);
v___x_105_ = v_b_95_;
v_isShared_106_ = v_isSharedCheck_135_;
goto v_resetjp_104_;
}
else
{
lean_inc(v_snd_103_);
lean_dec(v_b_95_);
v___x_105_ = lean_box(0);
v_isShared_106_ = v_isSharedCheck_135_;
goto v_resetjp_104_;
}
v_resetjp_104_:
{
lean_object* v___x_107_; lean_object* v_a_109_; lean_object* v_a_116_; 
v___x_107_ = lean_box(0);
v_a_116_ = lean_array_uget_borrowed(v_as_92_, v_i_94_);
if (lean_obj_tag(v_a_116_) == 0)
{
v_a_109_ = v_snd_103_;
goto v___jp_108_;
}
else
{
lean_object* v_val_117_; lean_object* v___x_118_; uint8_t v___y_120_; uint8_t v___x_133_; 
v_val_117_ = lean_ctor_get(v_a_116_, 0);
v___x_118_ = l_Lean_LocalDecl_fvarId(v_val_117_);
v___x_133_ = lp_mathlib_Array_contains___at___00Lean_Elab_Tactic_getVarsToClear_spec__0(v_preserve_91_, v___x_118_);
if (v___x_133_ == 0)
{
uint8_t v___x_134_; 
v___x_134_ = l_Lean_LocalDecl_isAuxDecl(v_val_117_);
v___y_120_ = v___x_134_;
goto v___jp_119_;
}
else
{
v___y_120_ = v___x_133_;
goto v___jp_119_;
}
v___jp_119_:
{
if (v___y_120_ == 0)
{
lean_object* v___x_121_; lean_object* v___x_122_; 
v___x_121_ = l_Lean_LocalDecl_type(v_val_117_);
v___x_122_ = l_Lean_Meta_isClass_x3f(v___x_121_, v___y_96_, v___y_97_, v___y_98_, v___y_99_);
if (lean_obj_tag(v___x_122_) == 0)
{
lean_object* v_a_123_; 
v_a_123_ = lean_ctor_get(v___x_122_, 0);
lean_inc(v_a_123_);
lean_dec_ref_known(v___x_122_, 1);
if (lean_obj_tag(v_a_123_) == 0)
{
lean_object* v___x_124_; 
v___x_124_ = lean_array_push(v_snd_103_, v___x_118_);
v_a_109_ = v___x_124_;
goto v___jp_108_;
}
else
{
lean_dec(v_a_123_);
lean_dec(v___x_118_);
v_a_109_ = v_snd_103_;
goto v___jp_108_;
}
}
else
{
lean_object* v_a_125_; lean_object* v___x_127_; uint8_t v_isShared_128_; uint8_t v_isSharedCheck_132_; 
lean_dec(v___x_118_);
lean_del_object(v___x_105_);
lean_dec(v_snd_103_);
v_a_125_ = lean_ctor_get(v___x_122_, 0);
v_isSharedCheck_132_ = !lean_is_exclusive(v___x_122_);
if (v_isSharedCheck_132_ == 0)
{
v___x_127_ = v___x_122_;
v_isShared_128_ = v_isSharedCheck_132_;
goto v_resetjp_126_;
}
else
{
lean_inc(v_a_125_);
lean_dec(v___x_122_);
v___x_127_ = lean_box(0);
v_isShared_128_ = v_isSharedCheck_132_;
goto v_resetjp_126_;
}
v_resetjp_126_:
{
lean_object* v___x_130_; 
if (v_isShared_128_ == 0)
{
v___x_130_ = v___x_127_;
goto v_reusejp_129_;
}
else
{
lean_object* v_reuseFailAlloc_131_; 
v_reuseFailAlloc_131_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_131_, 0, v_a_125_);
v___x_130_ = v_reuseFailAlloc_131_;
goto v_reusejp_129_;
}
v_reusejp_129_:
{
return v___x_130_;
}
}
}
}
else
{
lean_dec(v___x_118_);
v_a_109_ = v_snd_103_;
goto v___jp_108_;
}
}
}
v___jp_108_:
{
lean_object* v___x_111_; 
if (v_isShared_106_ == 0)
{
lean_ctor_set(v___x_105_, 1, v_a_109_);
lean_ctor_set(v___x_105_, 0, v___x_107_);
v___x_111_ = v___x_105_;
goto v_reusejp_110_;
}
else
{
lean_object* v_reuseFailAlloc_115_; 
v_reuseFailAlloc_115_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_115_, 0, v___x_107_);
lean_ctor_set(v_reuseFailAlloc_115_, 1, v_a_109_);
v___x_111_ = v_reuseFailAlloc_115_;
goto v_reusejp_110_;
}
v_reusejp_110_:
{
size_t v___x_112_; size_t v___x_113_; lean_object* v___x_114_; 
v___x_112_ = ((size_t)1ULL);
v___x_113_ = lean_usize_add(v_i_94_, v___x_112_);
v___x_114_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00Lean_Elab_Tactic_getVarsToClear_spec__1_spec__3_spec__6(v_preserve_91_, v_as_92_, v_sz_93_, v___x_113_, v___x_111_, v___y_96_, v___y_97_, v___y_98_, v___y_99_);
return v___x_114_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00Lean_Elab_Tactic_getVarsToClear_spec__1_spec__3___boxed(lean_object* v_preserve_137_, lean_object* v_as_138_, lean_object* v_sz_139_, lean_object* v_i_140_, lean_object* v_b_141_, lean_object* v___y_142_, lean_object* v___y_143_, lean_object* v___y_144_, lean_object* v___y_145_, lean_object* v___y_146_){
_start:
{
size_t v_sz_boxed_147_; size_t v_i_boxed_148_; lean_object* v_res_149_; 
v_sz_boxed_147_ = lean_unbox_usize(v_sz_139_);
lean_dec(v_sz_139_);
v_i_boxed_148_ = lean_unbox_usize(v_i_140_);
lean_dec(v_i_140_);
v_res_149_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00Lean_Elab_Tactic_getVarsToClear_spec__1_spec__3(v_preserve_137_, v_as_138_, v_sz_boxed_147_, v_i_boxed_148_, v_b_141_, v___y_142_, v___y_143_, v___y_144_, v___y_145_);
lean_dec(v___y_145_);
lean_dec_ref(v___y_144_);
lean_dec(v___y_143_);
lean_dec_ref(v___y_142_);
lean_dec_ref(v_as_138_);
lean_dec_ref(v_preserve_137_);
return v_res_149_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00Lean_Elab_Tactic_getVarsToClear_spec__1_spec__2_spec__4_spec__5(lean_object* v_preserve_150_, lean_object* v_as_151_, size_t v_sz_152_, size_t v_i_153_, lean_object* v_b_154_, lean_object* v___y_155_, lean_object* v___y_156_, lean_object* v___y_157_, lean_object* v___y_158_){
_start:
{
uint8_t v___x_160_; 
v___x_160_ = lean_usize_dec_lt(v_i_153_, v_sz_152_);
if (v___x_160_ == 0)
{
lean_object* v___x_161_; 
v___x_161_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_161_, 0, v_b_154_);
return v___x_161_;
}
else
{
lean_object* v_snd_162_; lean_object* v___x_164_; uint8_t v_isShared_165_; uint8_t v_isSharedCheck_194_; 
v_snd_162_ = lean_ctor_get(v_b_154_, 1);
v_isSharedCheck_194_ = !lean_is_exclusive(v_b_154_);
if (v_isSharedCheck_194_ == 0)
{
lean_object* v_unused_195_; 
v_unused_195_ = lean_ctor_get(v_b_154_, 0);
lean_dec(v_unused_195_);
v___x_164_ = v_b_154_;
v_isShared_165_ = v_isSharedCheck_194_;
goto v_resetjp_163_;
}
else
{
lean_inc(v_snd_162_);
lean_dec(v_b_154_);
v___x_164_ = lean_box(0);
v_isShared_165_ = v_isSharedCheck_194_;
goto v_resetjp_163_;
}
v_resetjp_163_:
{
lean_object* v___x_166_; lean_object* v_a_168_; lean_object* v_a_175_; 
v___x_166_ = lean_box(0);
v_a_175_ = lean_array_uget_borrowed(v_as_151_, v_i_153_);
if (lean_obj_tag(v_a_175_) == 0)
{
v_a_168_ = v_snd_162_;
goto v___jp_167_;
}
else
{
lean_object* v_val_176_; lean_object* v___x_177_; uint8_t v___y_179_; uint8_t v___x_192_; 
v_val_176_ = lean_ctor_get(v_a_175_, 0);
v___x_177_ = l_Lean_LocalDecl_fvarId(v_val_176_);
v___x_192_ = lp_mathlib_Array_contains___at___00Lean_Elab_Tactic_getVarsToClear_spec__0(v_preserve_150_, v___x_177_);
if (v___x_192_ == 0)
{
uint8_t v___x_193_; 
v___x_193_ = l_Lean_LocalDecl_isAuxDecl(v_val_176_);
v___y_179_ = v___x_193_;
goto v___jp_178_;
}
else
{
v___y_179_ = v___x_192_;
goto v___jp_178_;
}
v___jp_178_:
{
if (v___y_179_ == 0)
{
lean_object* v___x_180_; lean_object* v___x_181_; 
v___x_180_ = l_Lean_LocalDecl_type(v_val_176_);
v___x_181_ = l_Lean_Meta_isClass_x3f(v___x_180_, v___y_155_, v___y_156_, v___y_157_, v___y_158_);
if (lean_obj_tag(v___x_181_) == 0)
{
lean_object* v_a_182_; 
v_a_182_ = lean_ctor_get(v___x_181_, 0);
lean_inc(v_a_182_);
lean_dec_ref_known(v___x_181_, 1);
if (lean_obj_tag(v_a_182_) == 0)
{
lean_object* v___x_183_; 
v___x_183_ = lean_array_push(v_snd_162_, v___x_177_);
v_a_168_ = v___x_183_;
goto v___jp_167_;
}
else
{
lean_dec(v_a_182_);
lean_dec(v___x_177_);
v_a_168_ = v_snd_162_;
goto v___jp_167_;
}
}
else
{
lean_object* v_a_184_; lean_object* v___x_186_; uint8_t v_isShared_187_; uint8_t v_isSharedCheck_191_; 
lean_dec(v___x_177_);
lean_del_object(v___x_164_);
lean_dec(v_snd_162_);
v_a_184_ = lean_ctor_get(v___x_181_, 0);
v_isSharedCheck_191_ = !lean_is_exclusive(v___x_181_);
if (v_isSharedCheck_191_ == 0)
{
v___x_186_ = v___x_181_;
v_isShared_187_ = v_isSharedCheck_191_;
goto v_resetjp_185_;
}
else
{
lean_inc(v_a_184_);
lean_dec(v___x_181_);
v___x_186_ = lean_box(0);
v_isShared_187_ = v_isSharedCheck_191_;
goto v_resetjp_185_;
}
v_resetjp_185_:
{
lean_object* v___x_189_; 
if (v_isShared_187_ == 0)
{
v___x_189_ = v___x_186_;
goto v_reusejp_188_;
}
else
{
lean_object* v_reuseFailAlloc_190_; 
v_reuseFailAlloc_190_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_190_, 0, v_a_184_);
v___x_189_ = v_reuseFailAlloc_190_;
goto v_reusejp_188_;
}
v_reusejp_188_:
{
return v___x_189_;
}
}
}
}
else
{
lean_dec(v___x_177_);
v_a_168_ = v_snd_162_;
goto v___jp_167_;
}
}
}
v___jp_167_:
{
lean_object* v___x_170_; 
if (v_isShared_165_ == 0)
{
lean_ctor_set(v___x_164_, 1, v_a_168_);
lean_ctor_set(v___x_164_, 0, v___x_166_);
v___x_170_ = v___x_164_;
goto v_reusejp_169_;
}
else
{
lean_object* v_reuseFailAlloc_174_; 
v_reuseFailAlloc_174_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_174_, 0, v___x_166_);
lean_ctor_set(v_reuseFailAlloc_174_, 1, v_a_168_);
v___x_170_ = v_reuseFailAlloc_174_;
goto v_reusejp_169_;
}
v_reusejp_169_:
{
size_t v___x_171_; size_t v___x_172_; 
v___x_171_ = ((size_t)1ULL);
v___x_172_ = lean_usize_add(v_i_153_, v___x_171_);
v_i_153_ = v___x_172_;
v_b_154_ = v___x_170_;
goto _start;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00Lean_Elab_Tactic_getVarsToClear_spec__1_spec__2_spec__4_spec__5___boxed(lean_object* v_preserve_196_, lean_object* v_as_197_, lean_object* v_sz_198_, lean_object* v_i_199_, lean_object* v_b_200_, lean_object* v___y_201_, lean_object* v___y_202_, lean_object* v___y_203_, lean_object* v___y_204_, lean_object* v___y_205_){
_start:
{
size_t v_sz_boxed_206_; size_t v_i_boxed_207_; lean_object* v_res_208_; 
v_sz_boxed_206_ = lean_unbox_usize(v_sz_198_);
lean_dec(v_sz_198_);
v_i_boxed_207_ = lean_unbox_usize(v_i_199_);
lean_dec(v_i_199_);
v_res_208_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00Lean_Elab_Tactic_getVarsToClear_spec__1_spec__2_spec__4_spec__5(v_preserve_196_, v_as_197_, v_sz_boxed_206_, v_i_boxed_207_, v_b_200_, v___y_201_, v___y_202_, v___y_203_, v___y_204_);
lean_dec(v___y_204_);
lean_dec_ref(v___y_203_);
lean_dec(v___y_202_);
lean_dec_ref(v___y_201_);
lean_dec_ref(v_as_197_);
lean_dec_ref(v_preserve_196_);
return v_res_208_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00Lean_Elab_Tactic_getVarsToClear_spec__1_spec__2_spec__4(lean_object* v_preserve_209_, lean_object* v_as_210_, size_t v_sz_211_, size_t v_i_212_, lean_object* v_b_213_, lean_object* v___y_214_, lean_object* v___y_215_, lean_object* v___y_216_, lean_object* v___y_217_){
_start:
{
uint8_t v___x_219_; 
v___x_219_ = lean_usize_dec_lt(v_i_212_, v_sz_211_);
if (v___x_219_ == 0)
{
lean_object* v___x_220_; 
v___x_220_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_220_, 0, v_b_213_);
return v___x_220_;
}
else
{
lean_object* v_snd_221_; lean_object* v___x_223_; uint8_t v_isShared_224_; uint8_t v_isSharedCheck_253_; 
v_snd_221_ = lean_ctor_get(v_b_213_, 1);
v_isSharedCheck_253_ = !lean_is_exclusive(v_b_213_);
if (v_isSharedCheck_253_ == 0)
{
lean_object* v_unused_254_; 
v_unused_254_ = lean_ctor_get(v_b_213_, 0);
lean_dec(v_unused_254_);
v___x_223_ = v_b_213_;
v_isShared_224_ = v_isSharedCheck_253_;
goto v_resetjp_222_;
}
else
{
lean_inc(v_snd_221_);
lean_dec(v_b_213_);
v___x_223_ = lean_box(0);
v_isShared_224_ = v_isSharedCheck_253_;
goto v_resetjp_222_;
}
v_resetjp_222_:
{
lean_object* v___x_225_; lean_object* v_a_227_; lean_object* v_a_234_; 
v___x_225_ = lean_box(0);
v_a_234_ = lean_array_uget_borrowed(v_as_210_, v_i_212_);
if (lean_obj_tag(v_a_234_) == 0)
{
v_a_227_ = v_snd_221_;
goto v___jp_226_;
}
else
{
lean_object* v_val_235_; lean_object* v___x_236_; uint8_t v___y_238_; uint8_t v___x_251_; 
v_val_235_ = lean_ctor_get(v_a_234_, 0);
v___x_236_ = l_Lean_LocalDecl_fvarId(v_val_235_);
v___x_251_ = lp_mathlib_Array_contains___at___00Lean_Elab_Tactic_getVarsToClear_spec__0(v_preserve_209_, v___x_236_);
if (v___x_251_ == 0)
{
uint8_t v___x_252_; 
v___x_252_ = l_Lean_LocalDecl_isAuxDecl(v_val_235_);
v___y_238_ = v___x_252_;
goto v___jp_237_;
}
else
{
v___y_238_ = v___x_251_;
goto v___jp_237_;
}
v___jp_237_:
{
if (v___y_238_ == 0)
{
lean_object* v___x_239_; lean_object* v___x_240_; 
v___x_239_ = l_Lean_LocalDecl_type(v_val_235_);
v___x_240_ = l_Lean_Meta_isClass_x3f(v___x_239_, v___y_214_, v___y_215_, v___y_216_, v___y_217_);
if (lean_obj_tag(v___x_240_) == 0)
{
lean_object* v_a_241_; 
v_a_241_ = lean_ctor_get(v___x_240_, 0);
lean_inc(v_a_241_);
lean_dec_ref_known(v___x_240_, 1);
if (lean_obj_tag(v_a_241_) == 0)
{
lean_object* v___x_242_; 
v___x_242_ = lean_array_push(v_snd_221_, v___x_236_);
v_a_227_ = v___x_242_;
goto v___jp_226_;
}
else
{
lean_dec(v_a_241_);
lean_dec(v___x_236_);
v_a_227_ = v_snd_221_;
goto v___jp_226_;
}
}
else
{
lean_object* v_a_243_; lean_object* v___x_245_; uint8_t v_isShared_246_; uint8_t v_isSharedCheck_250_; 
lean_dec(v___x_236_);
lean_del_object(v___x_223_);
lean_dec(v_snd_221_);
v_a_243_ = lean_ctor_get(v___x_240_, 0);
v_isSharedCheck_250_ = !lean_is_exclusive(v___x_240_);
if (v_isSharedCheck_250_ == 0)
{
v___x_245_ = v___x_240_;
v_isShared_246_ = v_isSharedCheck_250_;
goto v_resetjp_244_;
}
else
{
lean_inc(v_a_243_);
lean_dec(v___x_240_);
v___x_245_ = lean_box(0);
v_isShared_246_ = v_isSharedCheck_250_;
goto v_resetjp_244_;
}
v_resetjp_244_:
{
lean_object* v___x_248_; 
if (v_isShared_246_ == 0)
{
v___x_248_ = v___x_245_;
goto v_reusejp_247_;
}
else
{
lean_object* v_reuseFailAlloc_249_; 
v_reuseFailAlloc_249_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_249_, 0, v_a_243_);
v___x_248_ = v_reuseFailAlloc_249_;
goto v_reusejp_247_;
}
v_reusejp_247_:
{
return v___x_248_;
}
}
}
}
else
{
lean_dec(v___x_236_);
v_a_227_ = v_snd_221_;
goto v___jp_226_;
}
}
}
v___jp_226_:
{
lean_object* v___x_229_; 
if (v_isShared_224_ == 0)
{
lean_ctor_set(v___x_223_, 1, v_a_227_);
lean_ctor_set(v___x_223_, 0, v___x_225_);
v___x_229_ = v___x_223_;
goto v_reusejp_228_;
}
else
{
lean_object* v_reuseFailAlloc_233_; 
v_reuseFailAlloc_233_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_233_, 0, v___x_225_);
lean_ctor_set(v_reuseFailAlloc_233_, 1, v_a_227_);
v___x_229_ = v_reuseFailAlloc_233_;
goto v_reusejp_228_;
}
v_reusejp_228_:
{
size_t v___x_230_; size_t v___x_231_; lean_object* v___x_232_; 
v___x_230_ = ((size_t)1ULL);
v___x_231_ = lean_usize_add(v_i_212_, v___x_230_);
v___x_232_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00Lean_Elab_Tactic_getVarsToClear_spec__1_spec__2_spec__4_spec__5(v_preserve_209_, v_as_210_, v_sz_211_, v___x_231_, v___x_229_, v___y_214_, v___y_215_, v___y_216_, v___y_217_);
return v___x_232_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00Lean_Elab_Tactic_getVarsToClear_spec__1_spec__2_spec__4___boxed(lean_object* v_preserve_255_, lean_object* v_as_256_, lean_object* v_sz_257_, lean_object* v_i_258_, lean_object* v_b_259_, lean_object* v___y_260_, lean_object* v___y_261_, lean_object* v___y_262_, lean_object* v___y_263_, lean_object* v___y_264_){
_start:
{
size_t v_sz_boxed_265_; size_t v_i_boxed_266_; lean_object* v_res_267_; 
v_sz_boxed_265_ = lean_unbox_usize(v_sz_257_);
lean_dec(v_sz_257_);
v_i_boxed_266_ = lean_unbox_usize(v_i_258_);
lean_dec(v_i_258_);
v_res_267_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00Lean_Elab_Tactic_getVarsToClear_spec__1_spec__2_spec__4(v_preserve_255_, v_as_256_, v_sz_boxed_265_, v_i_boxed_266_, v_b_259_, v___y_260_, v___y_261_, v___y_262_, v___y_263_);
lean_dec(v___y_263_);
lean_dec_ref(v___y_262_);
lean_dec(v___y_261_);
lean_dec_ref(v___y_260_);
lean_dec_ref(v_as_256_);
lean_dec_ref(v_preserve_255_);
return v_res_267_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00Lean_Elab_Tactic_getVarsToClear_spec__1_spec__2(lean_object* v_init_268_, lean_object* v_preserve_269_, lean_object* v_n_270_, lean_object* v_b_271_, lean_object* v___y_272_, lean_object* v___y_273_, lean_object* v___y_274_, lean_object* v___y_275_){
_start:
{
if (lean_obj_tag(v_n_270_) == 0)
{
lean_object* v_cs_277_; lean_object* v___x_278_; lean_object* v___x_279_; size_t v_sz_280_; size_t v___x_281_; lean_object* v___x_282_; 
v_cs_277_ = lean_ctor_get(v_n_270_, 0);
v___x_278_ = lean_box(0);
v___x_279_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_279_, 0, v___x_278_);
lean_ctor_set(v___x_279_, 1, v_b_271_);
v_sz_280_ = lean_array_size(v_cs_277_);
v___x_281_ = ((size_t)0ULL);
v___x_282_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00Lean_Elab_Tactic_getVarsToClear_spec__1_spec__2_spec__3(v_init_268_, v_preserve_269_, v_cs_277_, v_sz_280_, v___x_281_, v___x_279_, v___y_272_, v___y_273_, v___y_274_, v___y_275_);
if (lean_obj_tag(v___x_282_) == 0)
{
lean_object* v_a_283_; lean_object* v___x_285_; uint8_t v_isShared_286_; uint8_t v_isSharedCheck_297_; 
v_a_283_ = lean_ctor_get(v___x_282_, 0);
v_isSharedCheck_297_ = !lean_is_exclusive(v___x_282_);
if (v_isSharedCheck_297_ == 0)
{
v___x_285_ = v___x_282_;
v_isShared_286_ = v_isSharedCheck_297_;
goto v_resetjp_284_;
}
else
{
lean_inc(v_a_283_);
lean_dec(v___x_282_);
v___x_285_ = lean_box(0);
v_isShared_286_ = v_isSharedCheck_297_;
goto v_resetjp_284_;
}
v_resetjp_284_:
{
lean_object* v_fst_287_; 
v_fst_287_ = lean_ctor_get(v_a_283_, 0);
if (lean_obj_tag(v_fst_287_) == 0)
{
lean_object* v_snd_288_; lean_object* v___x_289_; lean_object* v___x_291_; 
v_snd_288_ = lean_ctor_get(v_a_283_, 1);
lean_inc(v_snd_288_);
lean_dec(v_a_283_);
v___x_289_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_289_, 0, v_snd_288_);
if (v_isShared_286_ == 0)
{
lean_ctor_set(v___x_285_, 0, v___x_289_);
v___x_291_ = v___x_285_;
goto v_reusejp_290_;
}
else
{
lean_object* v_reuseFailAlloc_292_; 
v_reuseFailAlloc_292_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_292_, 0, v___x_289_);
v___x_291_ = v_reuseFailAlloc_292_;
goto v_reusejp_290_;
}
v_reusejp_290_:
{
return v___x_291_;
}
}
else
{
lean_object* v_val_293_; lean_object* v___x_295_; 
lean_inc_ref(v_fst_287_);
lean_dec(v_a_283_);
v_val_293_ = lean_ctor_get(v_fst_287_, 0);
lean_inc(v_val_293_);
lean_dec_ref_known(v_fst_287_, 1);
if (v_isShared_286_ == 0)
{
lean_ctor_set(v___x_285_, 0, v_val_293_);
v___x_295_ = v___x_285_;
goto v_reusejp_294_;
}
else
{
lean_object* v_reuseFailAlloc_296_; 
v_reuseFailAlloc_296_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_296_, 0, v_val_293_);
v___x_295_ = v_reuseFailAlloc_296_;
goto v_reusejp_294_;
}
v_reusejp_294_:
{
return v___x_295_;
}
}
}
}
else
{
lean_object* v_a_298_; lean_object* v___x_300_; uint8_t v_isShared_301_; uint8_t v_isSharedCheck_305_; 
v_a_298_ = lean_ctor_get(v___x_282_, 0);
v_isSharedCheck_305_ = !lean_is_exclusive(v___x_282_);
if (v_isSharedCheck_305_ == 0)
{
v___x_300_ = v___x_282_;
v_isShared_301_ = v_isSharedCheck_305_;
goto v_resetjp_299_;
}
else
{
lean_inc(v_a_298_);
lean_dec(v___x_282_);
v___x_300_ = lean_box(0);
v_isShared_301_ = v_isSharedCheck_305_;
goto v_resetjp_299_;
}
v_resetjp_299_:
{
lean_object* v___x_303_; 
if (v_isShared_301_ == 0)
{
v___x_303_ = v___x_300_;
goto v_reusejp_302_;
}
else
{
lean_object* v_reuseFailAlloc_304_; 
v_reuseFailAlloc_304_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_304_, 0, v_a_298_);
v___x_303_ = v_reuseFailAlloc_304_;
goto v_reusejp_302_;
}
v_reusejp_302_:
{
return v___x_303_;
}
}
}
}
else
{
lean_object* v_vs_306_; lean_object* v___x_307_; lean_object* v___x_308_; size_t v_sz_309_; size_t v___x_310_; lean_object* v___x_311_; 
v_vs_306_ = lean_ctor_get(v_n_270_, 0);
v___x_307_ = lean_box(0);
v___x_308_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_308_, 0, v___x_307_);
lean_ctor_set(v___x_308_, 1, v_b_271_);
v_sz_309_ = lean_array_size(v_vs_306_);
v___x_310_ = ((size_t)0ULL);
v___x_311_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00Lean_Elab_Tactic_getVarsToClear_spec__1_spec__2_spec__4(v_preserve_269_, v_vs_306_, v_sz_309_, v___x_310_, v___x_308_, v___y_272_, v___y_273_, v___y_274_, v___y_275_);
if (lean_obj_tag(v___x_311_) == 0)
{
lean_object* v_a_312_; lean_object* v___x_314_; uint8_t v_isShared_315_; uint8_t v_isSharedCheck_326_; 
v_a_312_ = lean_ctor_get(v___x_311_, 0);
v_isSharedCheck_326_ = !lean_is_exclusive(v___x_311_);
if (v_isSharedCheck_326_ == 0)
{
v___x_314_ = v___x_311_;
v_isShared_315_ = v_isSharedCheck_326_;
goto v_resetjp_313_;
}
else
{
lean_inc(v_a_312_);
lean_dec(v___x_311_);
v___x_314_ = lean_box(0);
v_isShared_315_ = v_isSharedCheck_326_;
goto v_resetjp_313_;
}
v_resetjp_313_:
{
lean_object* v_fst_316_; 
v_fst_316_ = lean_ctor_get(v_a_312_, 0);
if (lean_obj_tag(v_fst_316_) == 0)
{
lean_object* v_snd_317_; lean_object* v___x_318_; lean_object* v___x_320_; 
v_snd_317_ = lean_ctor_get(v_a_312_, 1);
lean_inc(v_snd_317_);
lean_dec(v_a_312_);
v___x_318_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_318_, 0, v_snd_317_);
if (v_isShared_315_ == 0)
{
lean_ctor_set(v___x_314_, 0, v___x_318_);
v___x_320_ = v___x_314_;
goto v_reusejp_319_;
}
else
{
lean_object* v_reuseFailAlloc_321_; 
v_reuseFailAlloc_321_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_321_, 0, v___x_318_);
v___x_320_ = v_reuseFailAlloc_321_;
goto v_reusejp_319_;
}
v_reusejp_319_:
{
return v___x_320_;
}
}
else
{
lean_object* v_val_322_; lean_object* v___x_324_; 
lean_inc_ref(v_fst_316_);
lean_dec(v_a_312_);
v_val_322_ = lean_ctor_get(v_fst_316_, 0);
lean_inc(v_val_322_);
lean_dec_ref_known(v_fst_316_, 1);
if (v_isShared_315_ == 0)
{
lean_ctor_set(v___x_314_, 0, v_val_322_);
v___x_324_ = v___x_314_;
goto v_reusejp_323_;
}
else
{
lean_object* v_reuseFailAlloc_325_; 
v_reuseFailAlloc_325_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_325_, 0, v_val_322_);
v___x_324_ = v_reuseFailAlloc_325_;
goto v_reusejp_323_;
}
v_reusejp_323_:
{
return v___x_324_;
}
}
}
}
else
{
lean_object* v_a_327_; lean_object* v___x_329_; uint8_t v_isShared_330_; uint8_t v_isSharedCheck_334_; 
v_a_327_ = lean_ctor_get(v___x_311_, 0);
v_isSharedCheck_334_ = !lean_is_exclusive(v___x_311_);
if (v_isSharedCheck_334_ == 0)
{
v___x_329_ = v___x_311_;
v_isShared_330_ = v_isSharedCheck_334_;
goto v_resetjp_328_;
}
else
{
lean_inc(v_a_327_);
lean_dec(v___x_311_);
v___x_329_ = lean_box(0);
v_isShared_330_ = v_isSharedCheck_334_;
goto v_resetjp_328_;
}
v_resetjp_328_:
{
lean_object* v___x_332_; 
if (v_isShared_330_ == 0)
{
v___x_332_ = v___x_329_;
goto v_reusejp_331_;
}
else
{
lean_object* v_reuseFailAlloc_333_; 
v_reuseFailAlloc_333_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_333_, 0, v_a_327_);
v___x_332_ = v_reuseFailAlloc_333_;
goto v_reusejp_331_;
}
v_reusejp_331_:
{
return v___x_332_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00Lean_Elab_Tactic_getVarsToClear_spec__1_spec__2_spec__3(lean_object* v_init_335_, lean_object* v_preserve_336_, lean_object* v_as_337_, size_t v_sz_338_, size_t v_i_339_, lean_object* v_b_340_, lean_object* v___y_341_, lean_object* v___y_342_, lean_object* v___y_343_, lean_object* v___y_344_){
_start:
{
uint8_t v___x_346_; 
v___x_346_ = lean_usize_dec_lt(v_i_339_, v_sz_338_);
if (v___x_346_ == 0)
{
lean_object* v___x_347_; 
v___x_347_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_347_, 0, v_b_340_);
return v___x_347_;
}
else
{
lean_object* v_snd_348_; lean_object* v___x_350_; uint8_t v_isShared_351_; uint8_t v_isSharedCheck_382_; 
v_snd_348_ = lean_ctor_get(v_b_340_, 1);
v_isSharedCheck_382_ = !lean_is_exclusive(v_b_340_);
if (v_isSharedCheck_382_ == 0)
{
lean_object* v_unused_383_; 
v_unused_383_ = lean_ctor_get(v_b_340_, 0);
lean_dec(v_unused_383_);
v___x_350_ = v_b_340_;
v_isShared_351_ = v_isSharedCheck_382_;
goto v_resetjp_349_;
}
else
{
lean_inc(v_snd_348_);
lean_dec(v_b_340_);
v___x_350_ = lean_box(0);
v_isShared_351_ = v_isSharedCheck_382_;
goto v_resetjp_349_;
}
v_resetjp_349_:
{
lean_object* v_a_352_; lean_object* v___x_353_; 
v_a_352_ = lean_array_uget_borrowed(v_as_337_, v_i_339_);
lean_inc(v_snd_348_);
v___x_353_ = lp_mathlib_Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00Lean_Elab_Tactic_getVarsToClear_spec__1_spec__2(v_init_335_, v_preserve_336_, v_a_352_, v_snd_348_, v___y_341_, v___y_342_, v___y_343_, v___y_344_);
if (lean_obj_tag(v___x_353_) == 0)
{
lean_object* v_a_354_; lean_object* v___x_356_; uint8_t v_isShared_357_; uint8_t v_isSharedCheck_373_; 
v_a_354_ = lean_ctor_get(v___x_353_, 0);
v_isSharedCheck_373_ = !lean_is_exclusive(v___x_353_);
if (v_isSharedCheck_373_ == 0)
{
v___x_356_ = v___x_353_;
v_isShared_357_ = v_isSharedCheck_373_;
goto v_resetjp_355_;
}
else
{
lean_inc(v_a_354_);
lean_dec(v___x_353_);
v___x_356_ = lean_box(0);
v_isShared_357_ = v_isSharedCheck_373_;
goto v_resetjp_355_;
}
v_resetjp_355_:
{
if (lean_obj_tag(v_a_354_) == 0)
{
lean_object* v___x_358_; lean_object* v___x_360_; 
v___x_358_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_358_, 0, v_a_354_);
if (v_isShared_351_ == 0)
{
lean_ctor_set(v___x_350_, 0, v___x_358_);
v___x_360_ = v___x_350_;
goto v_reusejp_359_;
}
else
{
lean_object* v_reuseFailAlloc_364_; 
v_reuseFailAlloc_364_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_364_, 0, v___x_358_);
lean_ctor_set(v_reuseFailAlloc_364_, 1, v_snd_348_);
v___x_360_ = v_reuseFailAlloc_364_;
goto v_reusejp_359_;
}
v_reusejp_359_:
{
lean_object* v___x_362_; 
if (v_isShared_357_ == 0)
{
lean_ctor_set(v___x_356_, 0, v___x_360_);
v___x_362_ = v___x_356_;
goto v_reusejp_361_;
}
else
{
lean_object* v_reuseFailAlloc_363_; 
v_reuseFailAlloc_363_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_363_, 0, v___x_360_);
v___x_362_ = v_reuseFailAlloc_363_;
goto v_reusejp_361_;
}
v_reusejp_361_:
{
return v___x_362_;
}
}
}
else
{
lean_object* v_a_365_; lean_object* v___x_366_; lean_object* v___x_368_; 
lean_del_object(v___x_356_);
lean_dec(v_snd_348_);
v_a_365_ = lean_ctor_get(v_a_354_, 0);
lean_inc(v_a_365_);
lean_dec_ref_known(v_a_354_, 1);
v___x_366_ = lean_box(0);
if (v_isShared_351_ == 0)
{
lean_ctor_set(v___x_350_, 1, v_a_365_);
lean_ctor_set(v___x_350_, 0, v___x_366_);
v___x_368_ = v___x_350_;
goto v_reusejp_367_;
}
else
{
lean_object* v_reuseFailAlloc_372_; 
v_reuseFailAlloc_372_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_372_, 0, v___x_366_);
lean_ctor_set(v_reuseFailAlloc_372_, 1, v_a_365_);
v___x_368_ = v_reuseFailAlloc_372_;
goto v_reusejp_367_;
}
v_reusejp_367_:
{
size_t v___x_369_; size_t v___x_370_; 
v___x_369_ = ((size_t)1ULL);
v___x_370_ = lean_usize_add(v_i_339_, v___x_369_);
v_i_339_ = v___x_370_;
v_b_340_ = v___x_368_;
goto _start;
}
}
}
}
else
{
lean_object* v_a_374_; lean_object* v___x_376_; uint8_t v_isShared_377_; uint8_t v_isSharedCheck_381_; 
lean_del_object(v___x_350_);
lean_dec(v_snd_348_);
v_a_374_ = lean_ctor_get(v___x_353_, 0);
v_isSharedCheck_381_ = !lean_is_exclusive(v___x_353_);
if (v_isSharedCheck_381_ == 0)
{
v___x_376_ = v___x_353_;
v_isShared_377_ = v_isSharedCheck_381_;
goto v_resetjp_375_;
}
else
{
lean_inc(v_a_374_);
lean_dec(v___x_353_);
v___x_376_ = lean_box(0);
v_isShared_377_ = v_isSharedCheck_381_;
goto v_resetjp_375_;
}
v_resetjp_375_:
{
lean_object* v___x_379_; 
if (v_isShared_377_ == 0)
{
v___x_379_ = v___x_376_;
goto v_reusejp_378_;
}
else
{
lean_object* v_reuseFailAlloc_380_; 
v_reuseFailAlloc_380_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_380_, 0, v_a_374_);
v___x_379_ = v_reuseFailAlloc_380_;
goto v_reusejp_378_;
}
v_reusejp_378_:
{
return v___x_379_;
}
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00Lean_Elab_Tactic_getVarsToClear_spec__1_spec__2_spec__3___boxed(lean_object* v_init_384_, lean_object* v_preserve_385_, lean_object* v_as_386_, lean_object* v_sz_387_, lean_object* v_i_388_, lean_object* v_b_389_, lean_object* v___y_390_, lean_object* v___y_391_, lean_object* v___y_392_, lean_object* v___y_393_, lean_object* v___y_394_){
_start:
{
size_t v_sz_boxed_395_; size_t v_i_boxed_396_; lean_object* v_res_397_; 
v_sz_boxed_395_ = lean_unbox_usize(v_sz_387_);
lean_dec(v_sz_387_);
v_i_boxed_396_ = lean_unbox_usize(v_i_388_);
lean_dec(v_i_388_);
v_res_397_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00Lean_Elab_Tactic_getVarsToClear_spec__1_spec__2_spec__3(v_init_384_, v_preserve_385_, v_as_386_, v_sz_boxed_395_, v_i_boxed_396_, v_b_389_, v___y_390_, v___y_391_, v___y_392_, v___y_393_);
lean_dec(v___y_393_);
lean_dec_ref(v___y_392_);
lean_dec(v___y_391_);
lean_dec_ref(v___y_390_);
lean_dec_ref(v_as_386_);
lean_dec_ref(v_preserve_385_);
lean_dec_ref(v_init_384_);
return v_res_397_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00Lean_Elab_Tactic_getVarsToClear_spec__1_spec__2___boxed(lean_object* v_init_398_, lean_object* v_preserve_399_, lean_object* v_n_400_, lean_object* v_b_401_, lean_object* v___y_402_, lean_object* v___y_403_, lean_object* v___y_404_, lean_object* v___y_405_, lean_object* v___y_406_){
_start:
{
lean_object* v_res_407_; 
v_res_407_ = lp_mathlib_Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00Lean_Elab_Tactic_getVarsToClear_spec__1_spec__2(v_init_398_, v_preserve_399_, v_n_400_, v_b_401_, v___y_402_, v___y_403_, v___y_404_, v___y_405_);
lean_dec(v___y_405_);
lean_dec_ref(v___y_404_);
lean_dec(v___y_403_);
lean_dec_ref(v___y_402_);
lean_dec_ref(v_n_400_);
lean_dec_ref(v_preserve_399_);
lean_dec_ref(v_init_398_);
return v_res_407_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentArray_forIn___at___00Lean_Elab_Tactic_getVarsToClear_spec__1(lean_object* v_preserve_408_, lean_object* v_t_409_, lean_object* v_init_410_, lean_object* v___y_411_, lean_object* v___y_412_, lean_object* v___y_413_, lean_object* v___y_414_){
_start:
{
lean_object* v_root_416_; lean_object* v_tail_417_; lean_object* v___x_418_; 
v_root_416_ = lean_ctor_get(v_t_409_, 0);
v_tail_417_ = lean_ctor_get(v_t_409_, 1);
lean_inc_ref(v_init_410_);
v___x_418_ = lp_mathlib_Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00Lean_Elab_Tactic_getVarsToClear_spec__1_spec__2(v_init_410_, v_preserve_408_, v_root_416_, v_init_410_, v___y_411_, v___y_412_, v___y_413_, v___y_414_);
lean_dec_ref(v_init_410_);
if (lean_obj_tag(v___x_418_) == 0)
{
lean_object* v_a_419_; lean_object* v___x_421_; uint8_t v_isShared_422_; uint8_t v_isSharedCheck_455_; 
v_a_419_ = lean_ctor_get(v___x_418_, 0);
v_isSharedCheck_455_ = !lean_is_exclusive(v___x_418_);
if (v_isSharedCheck_455_ == 0)
{
v___x_421_ = v___x_418_;
v_isShared_422_ = v_isSharedCheck_455_;
goto v_resetjp_420_;
}
else
{
lean_inc(v_a_419_);
lean_dec(v___x_418_);
v___x_421_ = lean_box(0);
v_isShared_422_ = v_isSharedCheck_455_;
goto v_resetjp_420_;
}
v_resetjp_420_:
{
if (lean_obj_tag(v_a_419_) == 0)
{
lean_object* v_a_423_; lean_object* v___x_425_; 
v_a_423_ = lean_ctor_get(v_a_419_, 0);
lean_inc(v_a_423_);
lean_dec_ref_known(v_a_419_, 1);
if (v_isShared_422_ == 0)
{
lean_ctor_set(v___x_421_, 0, v_a_423_);
v___x_425_ = v___x_421_;
goto v_reusejp_424_;
}
else
{
lean_object* v_reuseFailAlloc_426_; 
v_reuseFailAlloc_426_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_426_, 0, v_a_423_);
v___x_425_ = v_reuseFailAlloc_426_;
goto v_reusejp_424_;
}
v_reusejp_424_:
{
return v___x_425_;
}
}
else
{
lean_object* v_a_427_; lean_object* v___x_428_; lean_object* v___x_429_; size_t v_sz_430_; size_t v___x_431_; lean_object* v___x_432_; 
lean_del_object(v___x_421_);
v_a_427_ = lean_ctor_get(v_a_419_, 0);
lean_inc(v_a_427_);
lean_dec_ref_known(v_a_419_, 1);
v___x_428_ = lean_box(0);
v___x_429_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_429_, 0, v___x_428_);
lean_ctor_set(v___x_429_, 1, v_a_427_);
v_sz_430_ = lean_array_size(v_tail_417_);
v___x_431_ = ((size_t)0ULL);
v___x_432_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00Lean_Elab_Tactic_getVarsToClear_spec__1_spec__3(v_preserve_408_, v_tail_417_, v_sz_430_, v___x_431_, v___x_429_, v___y_411_, v___y_412_, v___y_413_, v___y_414_);
if (lean_obj_tag(v___x_432_) == 0)
{
lean_object* v_a_433_; lean_object* v___x_435_; uint8_t v_isShared_436_; uint8_t v_isSharedCheck_446_; 
v_a_433_ = lean_ctor_get(v___x_432_, 0);
v_isSharedCheck_446_ = !lean_is_exclusive(v___x_432_);
if (v_isSharedCheck_446_ == 0)
{
v___x_435_ = v___x_432_;
v_isShared_436_ = v_isSharedCheck_446_;
goto v_resetjp_434_;
}
else
{
lean_inc(v_a_433_);
lean_dec(v___x_432_);
v___x_435_ = lean_box(0);
v_isShared_436_ = v_isSharedCheck_446_;
goto v_resetjp_434_;
}
v_resetjp_434_:
{
lean_object* v_fst_437_; 
v_fst_437_ = lean_ctor_get(v_a_433_, 0);
if (lean_obj_tag(v_fst_437_) == 0)
{
lean_object* v_snd_438_; lean_object* v___x_440_; 
v_snd_438_ = lean_ctor_get(v_a_433_, 1);
lean_inc(v_snd_438_);
lean_dec(v_a_433_);
if (v_isShared_436_ == 0)
{
lean_ctor_set(v___x_435_, 0, v_snd_438_);
v___x_440_ = v___x_435_;
goto v_reusejp_439_;
}
else
{
lean_object* v_reuseFailAlloc_441_; 
v_reuseFailAlloc_441_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_441_, 0, v_snd_438_);
v___x_440_ = v_reuseFailAlloc_441_;
goto v_reusejp_439_;
}
v_reusejp_439_:
{
return v___x_440_;
}
}
else
{
lean_object* v_val_442_; lean_object* v___x_444_; 
lean_inc_ref(v_fst_437_);
lean_dec(v_a_433_);
v_val_442_ = lean_ctor_get(v_fst_437_, 0);
lean_inc(v_val_442_);
lean_dec_ref_known(v_fst_437_, 1);
if (v_isShared_436_ == 0)
{
lean_ctor_set(v___x_435_, 0, v_val_442_);
v___x_444_ = v___x_435_;
goto v_reusejp_443_;
}
else
{
lean_object* v_reuseFailAlloc_445_; 
v_reuseFailAlloc_445_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_445_, 0, v_val_442_);
v___x_444_ = v_reuseFailAlloc_445_;
goto v_reusejp_443_;
}
v_reusejp_443_:
{
return v___x_444_;
}
}
}
}
else
{
lean_object* v_a_447_; lean_object* v___x_449_; uint8_t v_isShared_450_; uint8_t v_isSharedCheck_454_; 
v_a_447_ = lean_ctor_get(v___x_432_, 0);
v_isSharedCheck_454_ = !lean_is_exclusive(v___x_432_);
if (v_isSharedCheck_454_ == 0)
{
v___x_449_ = v___x_432_;
v_isShared_450_ = v_isSharedCheck_454_;
goto v_resetjp_448_;
}
else
{
lean_inc(v_a_447_);
lean_dec(v___x_432_);
v___x_449_ = lean_box(0);
v_isShared_450_ = v_isSharedCheck_454_;
goto v_resetjp_448_;
}
v_resetjp_448_:
{
lean_object* v___x_452_; 
if (v_isShared_450_ == 0)
{
v___x_452_ = v___x_449_;
goto v_reusejp_451_;
}
else
{
lean_object* v_reuseFailAlloc_453_; 
v_reuseFailAlloc_453_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_453_, 0, v_a_447_);
v___x_452_ = v_reuseFailAlloc_453_;
goto v_reusejp_451_;
}
v_reusejp_451_:
{
return v___x_452_;
}
}
}
}
}
}
else
{
lean_object* v_a_456_; lean_object* v___x_458_; uint8_t v_isShared_459_; uint8_t v_isSharedCheck_463_; 
v_a_456_ = lean_ctor_get(v___x_418_, 0);
v_isSharedCheck_463_ = !lean_is_exclusive(v___x_418_);
if (v_isSharedCheck_463_ == 0)
{
v___x_458_ = v___x_418_;
v_isShared_459_ = v_isSharedCheck_463_;
goto v_resetjp_457_;
}
else
{
lean_inc(v_a_456_);
lean_dec(v___x_418_);
v___x_458_ = lean_box(0);
v_isShared_459_ = v_isSharedCheck_463_;
goto v_resetjp_457_;
}
v_resetjp_457_:
{
lean_object* v___x_461_; 
if (v_isShared_459_ == 0)
{
v___x_461_ = v___x_458_;
goto v_reusejp_460_;
}
else
{
lean_object* v_reuseFailAlloc_462_; 
v_reuseFailAlloc_462_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_462_, 0, v_a_456_);
v___x_461_ = v_reuseFailAlloc_462_;
goto v_reusejp_460_;
}
v_reusejp_460_:
{
return v___x_461_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentArray_forIn___at___00Lean_Elab_Tactic_getVarsToClear_spec__1___boxed(lean_object* v_preserve_464_, lean_object* v_t_465_, lean_object* v_init_466_, lean_object* v___y_467_, lean_object* v___y_468_, lean_object* v___y_469_, lean_object* v___y_470_, lean_object* v___y_471_){
_start:
{
lean_object* v_res_472_; 
v_res_472_ = lp_mathlib_Lean_PersistentArray_forIn___at___00Lean_Elab_Tactic_getVarsToClear_spec__1(v_preserve_464_, v_t_465_, v_init_466_, v___y_467_, v___y_468_, v___y_469_, v___y_470_);
lean_dec(v___y_470_);
lean_dec_ref(v___y_469_);
lean_dec(v___y_468_);
lean_dec_ref(v___y_467_);
lean_dec_ref(v_t_465_);
lean_dec_ref(v_preserve_464_);
return v_res_472_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_Tactic_getVarsToClear(lean_object* v_preserve_475_, lean_object* v_a_476_, lean_object* v_a_477_, lean_object* v_a_478_, lean_object* v_a_479_){
_start:
{
lean_object* v_lctx_481_; lean_object* v_decls_482_; lean_object* v_toClear_483_; lean_object* v___x_484_; 
v_lctx_481_ = lean_ctor_get(v_a_476_, 2);
v_decls_482_ = lean_ctor_get(v_lctx_481_, 1);
v_toClear_483_ = ((lean_object*)(lp_mathlib_Lean_Elab_Tactic_getVarsToClear___closed__0));
v___x_484_ = lp_mathlib_Lean_PersistentArray_forIn___at___00Lean_Elab_Tactic_getVarsToClear_spec__1(v_preserve_475_, v_decls_482_, v_toClear_483_, v_a_476_, v_a_477_, v_a_478_, v_a_479_);
return v___x_484_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_Tactic_getVarsToClear___boxed(lean_object* v_preserve_485_, lean_object* v_a_486_, lean_object* v_a_487_, lean_object* v_a_488_, lean_object* v_a_489_, lean_object* v_a_490_){
_start:
{
lean_object* v_res_491_; 
v_res_491_ = lp_mathlib_Lean_Elab_Tactic_getVarsToClear(v_preserve_485_, v_a_486_, v_a_487_, v_a_488_, v_a_489_);
lean_dec(v_a_489_);
lean_dec_ref(v_a_488_);
lean_dec(v_a_487_);
lean_dec_ref(v_a_486_);
lean_dec_ref(v_preserve_485_);
return v_res_491_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_Tactic_clearExcept(lean_object* v_preserve_492_, lean_object* v_goal_493_, lean_object* v_a_494_, lean_object* v_a_495_, lean_object* v_a_496_, lean_object* v_a_497_){
_start:
{
lean_object* v___x_499_; 
v___x_499_ = lp_mathlib_Lean_Elab_Tactic_getVarsToClear(v_preserve_492_, v_a_494_, v_a_495_, v_a_496_, v_a_497_);
if (lean_obj_tag(v___x_499_) == 0)
{
lean_object* v_a_500_; lean_object* v___x_501_; 
v_a_500_ = lean_ctor_get(v___x_499_, 0);
lean_inc(v_a_500_);
lean_dec_ref_known(v___x_499_, 1);
v___x_501_ = l_Lean_MVarId_tryClearMany(v_goal_493_, v_a_500_, v_a_494_, v_a_495_, v_a_496_, v_a_497_);
lean_dec(v_a_500_);
return v___x_501_;
}
else
{
lean_object* v_a_502_; lean_object* v___x_504_; uint8_t v_isShared_505_; uint8_t v_isSharedCheck_509_; 
lean_dec(v_goal_493_);
v_a_502_ = lean_ctor_get(v___x_499_, 0);
v_isSharedCheck_509_ = !lean_is_exclusive(v___x_499_);
if (v_isSharedCheck_509_ == 0)
{
v___x_504_ = v___x_499_;
v_isShared_505_ = v_isSharedCheck_509_;
goto v_resetjp_503_;
}
else
{
lean_inc(v_a_502_);
lean_dec(v___x_499_);
v___x_504_ = lean_box(0);
v_isShared_505_ = v_isSharedCheck_509_;
goto v_resetjp_503_;
}
v_resetjp_503_:
{
lean_object* v___x_507_; 
if (v_isShared_505_ == 0)
{
v___x_507_ = v___x_504_;
goto v_reusejp_506_;
}
else
{
lean_object* v_reuseFailAlloc_508_; 
v_reuseFailAlloc_508_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_508_, 0, v_a_502_);
v___x_507_ = v_reuseFailAlloc_508_;
goto v_reusejp_506_;
}
v_reusejp_506_:
{
return v___x_507_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_Tactic_clearExcept___boxed(lean_object* v_preserve_510_, lean_object* v_goal_511_, lean_object* v_a_512_, lean_object* v_a_513_, lean_object* v_a_514_, lean_object* v_a_515_, lean_object* v_a_516_){
_start:
{
lean_object* v_res_517_; 
v_res_517_ = lp_mathlib_Lean_Elab_Tactic_clearExcept(v_preserve_510_, v_goal_511_, v_a_512_, v_a_513_, v_a_514_, v_a_515_);
lean_dec(v_a_515_);
lean_dec_ref(v_a_514_);
lean_dec(v_a_513_);
lean_dec_ref(v_a_512_);
lean_dec_ref(v_preserve_510_);
return v_res_517_;
}
}
static lean_object* _init_lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Lean_Elab_Tactic___aux__Mathlib__Tactic__ClearExcept______elabRules__Lean__Elab__Tactic__clearExceptTactic__1_spec__0___redArg___closed__0(void){
_start:
{
lean_object* v___x_586_; lean_object* v___x_587_; lean_object* v___x_588_; 
v___x_586_ = lean_box(0);
v___x_587_ = l_Lean_Elab_unsupportedSyntaxExceptionId;
v___x_588_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_588_, 0, v___x_587_);
lean_ctor_set(v___x_588_, 1, v___x_586_);
return v___x_588_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Lean_Elab_Tactic___aux__Mathlib__Tactic__ClearExcept______elabRules__Lean__Elab__Tactic__clearExceptTactic__1_spec__0___redArg(){
_start:
{
lean_object* v___x_590_; lean_object* v___x_591_; 
v___x_590_ = lean_obj_once(&lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Lean_Elab_Tactic___aux__Mathlib__Tactic__ClearExcept______elabRules__Lean__Elab__Tactic__clearExceptTactic__1_spec__0___redArg___closed__0, &lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Lean_Elab_Tactic___aux__Mathlib__Tactic__ClearExcept______elabRules__Lean__Elab__Tactic__clearExceptTactic__1_spec__0___redArg___closed__0_once, _init_lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Lean_Elab_Tactic___aux__Mathlib__Tactic__ClearExcept______elabRules__Lean__Elab__Tactic__clearExceptTactic__1_spec__0___redArg___closed__0);
v___x_591_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_591_, 0, v___x_590_);
return v___x_591_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Lean_Elab_Tactic___aux__Mathlib__Tactic__ClearExcept______elabRules__Lean__Elab__Tactic__clearExceptTactic__1_spec__0___redArg___boxed(lean_object* v___y_592_){
_start:
{
lean_object* v_res_593_; 
v_res_593_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Lean_Elab_Tactic___aux__Mathlib__Tactic__ClearExcept______elabRules__Lean__Elab__Tactic__clearExceptTactic__1_spec__0___redArg();
return v_res_593_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Lean_Elab_Tactic___aux__Mathlib__Tactic__ClearExcept______elabRules__Lean__Elab__Tactic__clearExceptTactic__1_spec__0(lean_object* v_00_u03b1_594_, lean_object* v___y_595_, lean_object* v___y_596_, lean_object* v___y_597_, lean_object* v___y_598_, lean_object* v___y_599_, lean_object* v___y_600_, lean_object* v___y_601_, lean_object* v___y_602_){
_start:
{
lean_object* v___x_604_; 
v___x_604_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Lean_Elab_Tactic___aux__Mathlib__Tactic__ClearExcept______elabRules__Lean__Elab__Tactic__clearExceptTactic__1_spec__0___redArg();
return v___x_604_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Lean_Elab_Tactic___aux__Mathlib__Tactic__ClearExcept______elabRules__Lean__Elab__Tactic__clearExceptTactic__1_spec__0___boxed(lean_object* v_00_u03b1_605_, lean_object* v___y_606_, lean_object* v___y_607_, lean_object* v___y_608_, lean_object* v___y_609_, lean_object* v___y_610_, lean_object* v___y_611_, lean_object* v___y_612_, lean_object* v___y_613_, lean_object* v___y_614_){
_start:
{
lean_object* v_res_615_; 
v_res_615_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Lean_Elab_Tactic___aux__Mathlib__Tactic__ClearExcept______elabRules__Lean__Elab__Tactic__clearExceptTactic__1_spec__0(v_00_u03b1_605_, v___y_606_, v___y_607_, v___y_608_, v___y_609_, v___y_610_, v___y_611_, v___y_612_, v___y_613_);
lean_dec(v___y_613_);
lean_dec_ref(v___y_612_);
lean_dec(v___y_611_);
lean_dec_ref(v___y_610_);
lean_dec(v___y_609_);
lean_dec_ref(v___y_608_);
lean_dec(v___y_607_);
lean_dec_ref(v___y_606_);
return v_res_615_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_Tactic___aux__Mathlib__Tactic__ClearExcept______elabRules__Lean__Elab__Tactic__clearExceptTactic__1___lam__0(lean_object* v_a_616_, lean_object* v___y_617_, lean_object* v___y_618_, lean_object* v___y_619_, lean_object* v___y_620_, lean_object* v___y_621_, lean_object* v___y_622_, lean_object* v___y_623_, lean_object* v___y_624_){
_start:
{
lean_object* v___x_626_; 
v___x_626_ = l_Lean_Elab_Tactic_getMainGoal___redArg(v___y_618_, v___y_621_, v___y_622_, v___y_623_, v___y_624_);
if (lean_obj_tag(v___x_626_) == 0)
{
lean_object* v_a_627_; lean_object* v___x_628_; 
v_a_627_ = lean_ctor_get(v___x_626_, 0);
lean_inc(v_a_627_);
lean_dec_ref_known(v___x_626_, 1);
v___x_628_ = lp_mathlib_Lean_Elab_Tactic_clearExcept(v_a_616_, v_a_627_, v___y_621_, v___y_622_, v___y_623_, v___y_624_);
if (lean_obj_tag(v___x_628_) == 0)
{
lean_object* v_a_629_; lean_object* v___x_630_; lean_object* v___x_631_; lean_object* v___x_632_; 
v_a_629_ = lean_ctor_get(v___x_628_, 0);
lean_inc(v_a_629_);
lean_dec_ref_known(v___x_628_, 1);
v___x_630_ = lean_box(0);
v___x_631_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_631_, 0, v_a_629_);
lean_ctor_set(v___x_631_, 1, v___x_630_);
v___x_632_ = l_Lean_Elab_Tactic_replaceMainGoal___redArg(v___x_631_, v___y_618_, v___y_621_, v___y_622_, v___y_623_, v___y_624_);
return v___x_632_;
}
else
{
lean_object* v_a_633_; lean_object* v___x_635_; uint8_t v_isShared_636_; uint8_t v_isSharedCheck_640_; 
v_a_633_ = lean_ctor_get(v___x_628_, 0);
v_isSharedCheck_640_ = !lean_is_exclusive(v___x_628_);
if (v_isSharedCheck_640_ == 0)
{
v___x_635_ = v___x_628_;
v_isShared_636_ = v_isSharedCheck_640_;
goto v_resetjp_634_;
}
else
{
lean_inc(v_a_633_);
lean_dec(v___x_628_);
v___x_635_ = lean_box(0);
v_isShared_636_ = v_isSharedCheck_640_;
goto v_resetjp_634_;
}
v_resetjp_634_:
{
lean_object* v___x_638_; 
if (v_isShared_636_ == 0)
{
v___x_638_ = v___x_635_;
goto v_reusejp_637_;
}
else
{
lean_object* v_reuseFailAlloc_639_; 
v_reuseFailAlloc_639_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_639_, 0, v_a_633_);
v___x_638_ = v_reuseFailAlloc_639_;
goto v_reusejp_637_;
}
v_reusejp_637_:
{
return v___x_638_;
}
}
}
}
else
{
lean_object* v_a_641_; lean_object* v___x_643_; uint8_t v_isShared_644_; uint8_t v_isSharedCheck_648_; 
v_a_641_ = lean_ctor_get(v___x_626_, 0);
v_isSharedCheck_648_ = !lean_is_exclusive(v___x_626_);
if (v_isSharedCheck_648_ == 0)
{
v___x_643_ = v___x_626_;
v_isShared_644_ = v_isSharedCheck_648_;
goto v_resetjp_642_;
}
else
{
lean_inc(v_a_641_);
lean_dec(v___x_626_);
v___x_643_ = lean_box(0);
v_isShared_644_ = v_isSharedCheck_648_;
goto v_resetjp_642_;
}
v_resetjp_642_:
{
lean_object* v___x_646_; 
if (v_isShared_644_ == 0)
{
v___x_646_ = v___x_643_;
goto v_reusejp_645_;
}
else
{
lean_object* v_reuseFailAlloc_647_; 
v_reuseFailAlloc_647_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_647_, 0, v_a_641_);
v___x_646_ = v_reuseFailAlloc_647_;
goto v_reusejp_645_;
}
v_reusejp_645_:
{
return v___x_646_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_Tactic___aux__Mathlib__Tactic__ClearExcept______elabRules__Lean__Elab__Tactic__clearExceptTactic__1___lam__0___boxed(lean_object* v_a_649_, lean_object* v___y_650_, lean_object* v___y_651_, lean_object* v___y_652_, lean_object* v___y_653_, lean_object* v___y_654_, lean_object* v___y_655_, lean_object* v___y_656_, lean_object* v___y_657_, lean_object* v___y_658_){
_start:
{
lean_object* v_res_659_; 
v_res_659_ = lp_mathlib_Lean_Elab_Tactic___aux__Mathlib__Tactic__ClearExcept______elabRules__Lean__Elab__Tactic__clearExceptTactic__1___lam__0(v_a_649_, v___y_650_, v___y_651_, v___y_652_, v___y_653_, v___y_654_, v___y_655_, v___y_656_, v___y_657_);
lean_dec(v___y_657_);
lean_dec_ref(v___y_656_);
lean_dec(v___y_655_);
lean_dec_ref(v___y_654_);
lean_dec(v___y_653_);
lean_dec_ref(v___y_652_);
lean_dec(v___y_651_);
lean_dec_ref(v___y_650_);
lean_dec_ref(v_a_649_);
return v_res_659_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_Tactic___aux__Mathlib__Tactic__ClearExcept______elabRules__Lean__Elab__Tactic__clearExceptTactic__1(lean_object* v_x_660_, lean_object* v_a_661_, lean_object* v_a_662_, lean_object* v_a_663_, lean_object* v_a_664_, lean_object* v_a_665_, lean_object* v_a_666_, lean_object* v_a_667_, lean_object* v_a_668_){
_start:
{
lean_object* v___x_670_; uint8_t v___x_671_; 
v___x_670_ = ((lean_object*)(lp_mathlib_Lean_Elab_Tactic_clearExceptTactic___closed__4));
lean_inc(v_x_660_);
v___x_671_ = l_Lean_Syntax_isOfKind(v_x_660_, v___x_670_);
if (v___x_671_ == 0)
{
lean_object* v___x_672_; 
lean_dec(v_x_660_);
v___x_672_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Lean_Elab_Tactic___aux__Mathlib__Tactic__ClearExcept______elabRules__Lean__Elab__Tactic__clearExceptTactic__1_spec__0___redArg();
return v___x_672_;
}
else
{
lean_object* v___x_673_; lean_object* v___x_674_; lean_object* v_hs_675_; lean_object* v___x_676_; 
v___x_673_ = lean_unsigned_to_nat(3u);
v___x_674_ = l_Lean_Syntax_getArg(v_x_660_, v___x_673_);
lean_dec(v_x_660_);
v_hs_675_ = l_Lean_Syntax_getArgs(v___x_674_);
lean_dec(v___x_674_);
v___x_676_ = l_Lean_Elab_Tactic_getFVarIds(v_hs_675_, v_a_661_, v_a_662_, v_a_663_, v_a_664_, v_a_665_, v_a_666_, v_a_667_, v_a_668_);
if (lean_obj_tag(v___x_676_) == 0)
{
lean_object* v_a_677_; lean_object* v___f_678_; lean_object* v___x_679_; 
v_a_677_ = lean_ctor_get(v___x_676_, 0);
lean_inc(v_a_677_);
lean_dec_ref_known(v___x_676_, 1);
v___f_678_ = lean_alloc_closure((void*)(lp_mathlib_Lean_Elab_Tactic___aux__Mathlib__Tactic__ClearExcept______elabRules__Lean__Elab__Tactic__clearExceptTactic__1___lam__0___boxed), 10, 1);
lean_closure_set(v___f_678_, 0, v_a_677_);
v___x_679_ = l_Lean_Elab_Tactic_withMainContext___redArg(v___f_678_, v_a_661_, v_a_662_, v_a_663_, v_a_664_, v_a_665_, v_a_666_, v_a_667_, v_a_668_);
return v___x_679_;
}
else
{
lean_object* v_a_680_; lean_object* v___x_682_; uint8_t v_isShared_683_; uint8_t v_isSharedCheck_687_; 
v_a_680_ = lean_ctor_get(v___x_676_, 0);
v_isSharedCheck_687_ = !lean_is_exclusive(v___x_676_);
if (v_isSharedCheck_687_ == 0)
{
v___x_682_ = v___x_676_;
v_isShared_683_ = v_isSharedCheck_687_;
goto v_resetjp_681_;
}
else
{
lean_inc(v_a_680_);
lean_dec(v___x_676_);
v___x_682_ = lean_box(0);
v_isShared_683_ = v_isSharedCheck_687_;
goto v_resetjp_681_;
}
v_resetjp_681_:
{
lean_object* v___x_685_; 
if (v_isShared_683_ == 0)
{
v___x_685_ = v___x_682_;
goto v_reusejp_684_;
}
else
{
lean_object* v_reuseFailAlloc_686_; 
v_reuseFailAlloc_686_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_686_, 0, v_a_680_);
v___x_685_ = v_reuseFailAlloc_686_;
goto v_reusejp_684_;
}
v_reusejp_684_:
{
return v___x_685_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_Tactic___aux__Mathlib__Tactic__ClearExcept______elabRules__Lean__Elab__Tactic__clearExceptTactic__1___boxed(lean_object* v_x_688_, lean_object* v_a_689_, lean_object* v_a_690_, lean_object* v_a_691_, lean_object* v_a_692_, lean_object* v_a_693_, lean_object* v_a_694_, lean_object* v_a_695_, lean_object* v_a_696_, lean_object* v_a_697_){
_start:
{
lean_object* v_res_698_; 
v_res_698_ = lp_mathlib_Lean_Elab_Tactic___aux__Mathlib__Tactic__ClearExcept______elabRules__Lean__Elab__Tactic__clearExceptTactic__1(v_x_688_, v_a_689_, v_a_690_, v_a_691_, v_a_692_, v_a_693_, v_a_694_, v_a_695_, v_a_696_);
lean_dec(v_a_696_);
lean_dec_ref(v_a_695_);
lean_dec(v_a_694_);
lean_dec_ref(v_a_693_);
lean_dec(v_a_692_);
lean_dec_ref(v_a_691_);
lean_dec(v_a_690_);
lean_dec_ref(v_a_689_);
return v_res_698_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Init(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_Tactic_ClearExcept(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_Lean_Elab_Tactic_ElabTerm(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_Tactic_ClearExcept(uint8_t builtin) {
lean_object * res;
if (_G_meta_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_meta_initialized = true;
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_Lean_Elab_Tactic_ElabTerm(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Init(uint8_t builtin);
lean_object* initialize_Lean_Elab_Tactic_ElabTerm(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Tactic_ClearExcept(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Lean_Elab_Tactic_ElabTerm(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Tactic_ClearExcept(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_Tactic_ClearExcept(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_Tactic_ClearExcept(builtin);
}
#ifdef __cplusplus
}
#endif
