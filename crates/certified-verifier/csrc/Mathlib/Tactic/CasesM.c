// Lean compiler output
// Module: Mathlib.Tactic.CasesM
// Imports: public import Init public meta import Init public import Mathlib.Init public meta import Lean.Elab.Tactic.Conv.Pattern
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
uint8_t l_Lean_Expr_hasMVar(lean_object*);
lean_object* lean_st_ref_get(lean_object*);
lean_object* l_Lean_instantiateMVarsCore(lean_object*, lean_object*);
lean_object* lean_st_ref_take(lean_object*);
lean_object* lean_st_ref_set(lean_object*, lean_object*);
lean_object* l_Lean_Name_mkStr1(lean_object*);
lean_object* l_Lean_Name_mkStr3(lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Elab_Term_withoutErrToSorryImp___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_MVarId_getType(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lean_array_push(lean_object*, lean_object*);
lean_object* l_Lean_MVarId_constructor(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l___private_Lean_Meta_Basic_0__Lean_Meta_withMVarContextImp(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
size_t lean_array_size(lean_object*);
uint8_t lean_usize_dec_lt(size_t, size_t);
lean_object* lean_array_uget_borrowed(lean_object*, size_t);
lean_object* l_Lean_Elab_realizeGlobalConstNoOverloadWithInfo(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lean_array_uset(lean_object*, size_t, lean_object*);
size_t lean_usize_add(size_t, size_t);
lean_object* l_Lean_Elab_Tactic_getMainGoal___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Expr_headBeta(lean_object*);
lean_object* l_Lean_Expr_getAppFn(lean_object*);
lean_object* lean_array_get_size(lean_object*);
uint8_t lean_nat_dec_lt(lean_object*, lean_object*);
size_t lean_usize_of_nat(lean_object*);
uint8_t lean_usize_dec_eq(size_t, size_t);
uint8_t lean_name_eq(lean_object*, lean_object*);
lean_object* lean_mk_empty_array_with_capacity(lean_object*);
lean_object* lean_array_uget(lean_object*, size_t);
uint8_t lean_nat_dec_eq(lean_object*, lean_object*);
lean_object* lean_array_fget_borrowed(lean_object*, lean_object*);
lean_object* l_Lean_LocalDecl_userName(lean_object*);
lean_object* l_Lean_MVarId_rename(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
uint8_t l_Lean_LocalDecl_isImplementationDetail(lean_object*);
lean_object* l_Lean_LocalDecl_type(lean_object*);
lean_object* l_Lean_Meta_saveState___redArg(lean_object*, lean_object*);
lean_object* l_Lean_LocalDecl_fvarId(lean_object*);
lean_object* l_Lean_MVarId_cases(lean_object*, lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Meta_SavedState_restore___redArg(lean_object*, lean_object*, lean_object*);
lean_object* lean_array_to_list(lean_object*);
uint8_t l_Lean_instBEqMVarId_beq(lean_object*, lean_object*);
lean_object* l_Lean_stringToMessageData(lean_object*);
lean_object* l_Lean_Elab_Tactic_replaceMainGoal___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Elab_Tactic_withMainContext___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Elab_Term_elabTerm(lean_object*, lean_object*, uint8_t, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_replaceRef(lean_object*, lean_object*);
lean_object* l_Lean_Meta_abstractMVars(lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Elab_Term_withoutModifyingElabMetaStateWithInfo___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
uint8_t l_Lean_Syntax_isOfKind(lean_object*, lean_object*);
extern lean_object* l_Lean_Elab_unsupportedSyntaxExceptionId;
lean_object* l_Lean_Syntax_getArg(lean_object*, lean_object*);
lean_object* l_Lean_Syntax_getArgs(lean_object*);
lean_object* l_Lean_Syntax_getOptional_x3f(lean_object*);
lean_object* l_Lean_Elab_Tactic_Conv_matchPattern_x3f(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Syntax_TSepArray_getElems___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_withContext___at___00__private_Mathlib_Tactic_CasesM_0__Lean_MVarId_casesMatching_go_spec__2___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_withContext___at___00__private_Mathlib_Tactic_CasesM_0__Lean_MVarId_casesMatching_go_spec__2___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_withContext___at___00__private_Mathlib_Tactic_CasesM_0__Lean_MVarId_casesMatching_go_spec__2(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_withContext___at___00__private_Mathlib_Tactic_CasesM_0__Lean_MVarId_casesMatching_go_spec__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_CasesM_0__Lean_MVarId_casesMatching_go___lam__0___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_CasesM_0__Lean_MVarId_casesMatching_go___lam__0___closed__0 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_CasesM_0__Lean_MVarId_casesMatching_go___lam__0___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00__private_Mathlib_Tactic_CasesM_0__Lean_MVarId_casesMatching_go_spec__1_spec__1_spec__3(lean_object*, uint8_t, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, size_t, size_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_CasesM_0__Lean_MVarId_casesMatching_go_spec__0(uint8_t, lean_object*, uint8_t, lean_object*, lean_object*, size_t, size_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_ctor_object lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00__private_Mathlib_Tactic_CasesM_0__Lean_MVarId_casesMatching_go_spec__1_spec__1_spec__4___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00__private_Mathlib_Tactic_CasesM_0__Lean_MVarId_casesMatching_go_spec__1_spec__1_spec__4___closed__0 = (const lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00__private_Mathlib_Tactic_CasesM_0__Lean_MVarId_casesMatching_go_spec__1_spec__1_spec__4___closed__0_value;
static const lean_array_object lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00__private_Mathlib_Tactic_CasesM_0__Lean_MVarId_casesMatching_go_spec__1_spec__2___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_array_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 246}, .m_size = 0, .m_capacity = 0, .m_data = {}};
static const lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00__private_Mathlib_Tactic_CasesM_0__Lean_MVarId_casesMatching_go_spec__1_spec__2___closed__1 = (const lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00__private_Mathlib_Tactic_CasesM_0__Lean_MVarId_casesMatching_go_spec__1_spec__2___closed__1_value;
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00__private_Mathlib_Tactic_CasesM_0__Lean_MVarId_casesMatching_go_spec__1_spec__1_spec__4_spec__5(uint8_t, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, size_t, size_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00__private_Mathlib_Tactic_CasesM_0__Lean_MVarId_casesMatching_go_spec__1_spec__1_spec__4(uint8_t, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, size_t, size_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00__private_Mathlib_Tactic_CasesM_0__Lean_MVarId_casesMatching_go_spec__1_spec__1(lean_object*, uint8_t, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_ctor_object lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00__private_Mathlib_Tactic_CasesM_0__Lean_MVarId_casesMatching_go_spec__1_spec__2___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00__private_Mathlib_Tactic_CasesM_0__Lean_MVarId_casesMatching_go_spec__1_spec__2___closed__0 = (const lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00__private_Mathlib_Tactic_CasesM_0__Lean_MVarId_casesMatching_go_spec__1_spec__2___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00__private_Mathlib_Tactic_CasesM_0__Lean_MVarId_casesMatching_go_spec__1_spec__2_spec__6(uint8_t, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, size_t, size_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00__private_Mathlib_Tactic_CasesM_0__Lean_MVarId_casesMatching_go_spec__1_spec__2(uint8_t, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, size_t, size_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentArray_forIn___at___00__private_Mathlib_Tactic_CasesM_0__Lean_MVarId_casesMatching_go_spec__1(uint8_t, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_CasesM_0__Lean_MVarId_casesMatching_go___lam__0(uint8_t, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_CasesM_0__Lean_MVarId_casesMatching_go___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_CasesM_0__Lean_MVarId_casesMatching_go(lean_object*, uint8_t, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_CasesM_0__Lean_MVarId_casesMatching_go___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00__private_Mathlib_Tactic_CasesM_0__Lean_MVarId_casesMatching_go_spec__1_spec__1_spec__3___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentArray_forIn___at___00__private_Mathlib_Tactic_CasesM_0__Lean_MVarId_casesMatching_go_spec__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00__private_Mathlib_Tactic_CasesM_0__Lean_MVarId_casesMatching_go_spec__1_spec__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_CasesM_0__Lean_MVarId_casesMatching_go_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00__private_Mathlib_Tactic_CasesM_0__Lean_MVarId_casesMatching_go_spec__1_spec__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00__private_Mathlib_Tactic_CasesM_0__Lean_MVarId_casesMatching_go_spec__1_spec__2_spec__6___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00__private_Mathlib_Tactic_CasesM_0__Lean_MVarId_casesMatching_go_spec__1_spec__1_spec__4___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00__private_Mathlib_Tactic_CasesM_0__Lean_MVarId_casesMatching_go_spec__1_spec__1_spec__4_spec__5___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextFull___at___00Lean_throwError___at___00Lean_MVarId_casesMatching_spec__1_spec__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextFull___at___00Lean_throwError___at___00Lean_MVarId_casesMatching_spec__1_spec__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Lean_MVarId_casesMatching_spec__1___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Lean_MVarId_casesMatching_spec__1___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_List_beq___at___00Lean_MVarId_casesMatching_spec__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_beq___at___00Lean_MVarId_casesMatching_spec__0___boxed(lean_object*, lean_object*);
static const lean_array_object lp_mathlib_Lean_MVarId_casesMatching___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_array_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 246}, .m_size = 0, .m_capacity = 0, .m_data = {}};
static const lean_object* lp_mathlib_Lean_MVarId_casesMatching___closed__0 = (const lean_object*)&lp_mathlib_Lean_MVarId_casesMatching___closed__0_value;
static const lean_string_object lp_mathlib_Lean_MVarId_casesMatching___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "no match"};
static const lean_object* lp_mathlib_Lean_MVarId_casesMatching___closed__1 = (const lean_object*)&lp_mathlib_Lean_MVarId_casesMatching___closed__1_value;
static lean_once_cell_t lp_mathlib_Lean_MVarId_casesMatching___closed__2_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_MVarId_casesMatching___closed__2;
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_casesMatching(lean_object*, uint8_t, uint8_t, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_casesMatching___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Lean_MVarId_casesMatching_spec__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Lean_MVarId_casesMatching_spec__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib___private_Init_Data_Array_Basic_0__Array_anyMUnsafe_any___at___00Array_contains___at___00Lean_MVarId_casesType_spec__0_spec__0(lean_object*, lean_object*, size_t, size_t);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_anyMUnsafe_any___at___00Array_contains___at___00Lean_MVarId_casesType_spec__0_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Array_contains___at___00Lean_MVarId_casesType_spec__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Array_contains___at___00Lean_MVarId_casesType_spec__0___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_casesType___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_casesType___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_casesType(lean_object*, uint8_t, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_casesType___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_Term_withoutErrToSorry___at___00Mathlib_Tactic_elabPatterns_spec__1___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_Term_withoutErrToSorry___at___00Mathlib_Tactic_elabPatterns_spec__1___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_Term_withoutErrToSorry___at___00Mathlib_Tactic_elabPatterns_spec__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_Term_withoutErrToSorry___at___00Mathlib_Tactic_elabPatterns_spec__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Mathlib_Tactic_elabPatterns_spec__0___lam__0(lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Mathlib_Tactic_elabPatterns_spec__0___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Mathlib_Tactic_elabPatterns_spec__0(size_t, size_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Mathlib_Tactic_elabPatterns_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_elabPatterns___boxed__const__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*0 + sizeof(size_t)*1, .m_other = 0, .m_tag = 0}, .m_objs = {(lean_object*)(size_t)(0ULL)}};
LEAN_EXPORT const lean_object* lp_mathlib_Mathlib_Tactic_elabPatterns___boxed__const__1 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_elabPatterns___boxed__const__1_value;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_elabPatterns(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_elabPatterns___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00Mathlib_Tactic_matchPatterns_spec__0___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00Mathlib_Tactic_matchPatterns_spec__0___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00Mathlib_Tactic_matchPatterns_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00Mathlib_Tactic_matchPatterns_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_anyMUnsafe_any___at___00Mathlib_Tactic_matchPatterns_spec__1(lean_object*, lean_object*, size_t, size_t, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_anyMUnsafe_any___at___00Mathlib_Tactic_matchPatterns_spec__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_matchPatterns(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_matchPatterns___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_elabCasesM___lam__0(lean_object*, uint8_t, uint8_t, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_elabCasesM___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_elabCasesM(lean_object*, uint8_t, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_elabCasesM___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Mathlib_Tactic_casesM___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "Mathlib"};
static const lean_object* lp_mathlib_Mathlib_Tactic_casesM___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_casesM___closed__0_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_casesM___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "Tactic"};
static const lean_object* lp_mathlib_Mathlib_Tactic_casesM___closed__1 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_casesM___closed__1_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_casesM___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "casesM"};
static const lean_object* lp_mathlib_Mathlib_Tactic_casesM___closed__2 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_casesM___closed__2_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_casesM___closed__3_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_casesM___closed__0_value),LEAN_SCALAR_PTR_LITERAL(118, 213, 161, 2, 73, 184, 31, 228)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_casesM___closed__3_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_casesM___closed__3_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic_casesM___closed__1_value),LEAN_SCALAR_PTR_LITERAL(139, 222, 98, 232, 116, 132, 69, 249)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_casesM___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_casesM___closed__3_value_aux_1),((lean_object*)&lp_mathlib_Mathlib_Tactic_casesM___closed__2_value),LEAN_SCALAR_PTR_LITERAL(74, 27, 80, 108, 1, 40, 176, 17)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_casesM___closed__3 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_casesM___closed__3_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_casesM___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "andthen"};
static const lean_object* lp_mathlib_Mathlib_Tactic_casesM___closed__4 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_casesM___closed__4_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_casesM___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_casesM___closed__4_value),LEAN_SCALAR_PTR_LITERAL(40, 255, 78, 30, 143, 119, 117, 174)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_casesM___closed__5 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_casesM___closed__5_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_casesM___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "casesm"};
static const lean_object* lp_mathlib_Mathlib_Tactic_casesM___closed__6 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_casesM___closed__6_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_casesM___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 8, .m_other = 1, .m_tag = 6}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_casesM___closed__6_value),LEAN_SCALAR_PTR_LITERAL(0, 0, 0, 0, 0, 0, 0, 0)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_casesM___closed__7 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_casesM___closed__7_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_casesM___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "optional"};
static const lean_object* lp_mathlib_Mathlib_Tactic_casesM___closed__8 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_casesM___closed__8_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_casesM___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_casesM___closed__8_value),LEAN_SCALAR_PTR_LITERAL(233, 141, 154, 50, 143, 135, 42, 252)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_casesM___closed__9 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_casesM___closed__9_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_casesM___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = "*"};
static const lean_object* lp_mathlib_Mathlib_Tactic_casesM___closed__10 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_casesM___closed__10_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_casesM___closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 5}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_casesM___closed__10_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_casesM___closed__11 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_casesM___closed__11_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_casesM___closed__12_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_casesM___closed__9_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_casesM___closed__11_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_casesM___closed__12 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_casesM___closed__12_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_casesM___closed__13_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_casesM___closed__5_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_casesM___closed__7_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_casesM___closed__12_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_casesM___closed__13 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_casesM___closed__13_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_casesM___closed__14_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "group"};
static const lean_object* lp_mathlib_Mathlib_Tactic_casesM___closed__14 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_casesM___closed__14_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_casesM___closed__15_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_casesM___closed__14_value),LEAN_SCALAR_PTR_LITERAL(206, 113, 20, 57, 188, 177, 187, 30)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_casesM___closed__15 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_casesM___closed__15_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_casesM___closed__16_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "ppSpace"};
static const lean_object* lp_mathlib_Mathlib_Tactic_casesM___closed__16 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_casesM___closed__16_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_casesM___closed__17_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_casesM___closed__16_value),LEAN_SCALAR_PTR_LITERAL(207, 47, 58, 43, 30, 240, 125, 246)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_casesM___closed__17 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_casesM___closed__17_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_casesM___closed__18_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_casesM___closed__17_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_casesM___closed__18 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_casesM___closed__18_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_casesM___closed__19_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_casesM___closed__15_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_casesM___closed__18_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_casesM___closed__19 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_casesM___closed__19_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_casesM___closed__20_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_casesM___closed__5_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_casesM___closed__13_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_casesM___closed__19_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_casesM___closed__20 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_casesM___closed__20_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_casesM___closed__21_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "term"};
static const lean_object* lp_mathlib_Mathlib_Tactic_casesM___closed__21 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_casesM___closed__21_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_casesM___closed__22_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_casesM___closed__21_value),LEAN_SCALAR_PTR_LITERAL(187, 230, 181, 162, 253, 146, 122, 119)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_casesM___closed__22 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_casesM___closed__22_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_casesM___closed__23_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 7}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_casesM___closed__22_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_Mathlib_Tactic_casesM___closed__23 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_casesM___closed__23_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_casesM___closed__24_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = ","};
static const lean_object* lp_mathlib_Mathlib_Tactic_casesM___closed__24 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_casesM___closed__24_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_casesM___closed__25_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 3, .m_capacity = 3, .m_length = 2, .m_data = ", "};
static const lean_object* lp_mathlib_Mathlib_Tactic_casesM___closed__25 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_casesM___closed__25_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_casesM___closed__26_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 5}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_casesM___closed__25_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_casesM___closed__26 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_casesM___closed__26_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_casesM___closed__27_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 8, .m_other = 3, .m_tag = 11}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_casesM___closed__23_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_casesM___closed__24_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_casesM___closed__26_value),LEAN_SCALAR_PTR_LITERAL(0, 0, 0, 0, 0, 0, 0, 0)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_casesM___closed__27 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_casesM___closed__27_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_casesM___closed__28_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_casesM___closed__5_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_casesM___closed__20_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_casesM___closed__27_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_casesM___closed__28 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_casesM___closed__28_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_casesM___closed__29_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 3}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_casesM___closed__3_value),((lean_object*)(((size_t)(1022) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_casesM___closed__28_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_casesM___closed__29 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_casesM___closed__29_value;
LEAN_EXPORT const lean_object* lp_mathlib_Mathlib_Tactic_casesM = (const lean_object*)&lp_mathlib_Mathlib_Tactic_casesM___closed__29_value;
static lean_once_cell_t lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic___aux__Mathlib__Tactic__CasesM______elabRules__Mathlib__Tactic__casesM__1_spec__0___redArg___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic___aux__Mathlib__Tactic__CasesM______elabRules__Mathlib__Tactic__casesM__1_spec__0___redArg___closed__0;
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic___aux__Mathlib__Tactic__CasesM______elabRules__Mathlib__Tactic__casesM__1_spec__0___redArg();
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic___aux__Mathlib__Tactic__CasesM______elabRules__Mathlib__Tactic__casesM__1_spec__0___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic___aux__Mathlib__Tactic__CasesM______elabRules__Mathlib__Tactic__casesM__1_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic___aux__Mathlib__Tactic__CasesM______elabRules__Mathlib__Tactic__casesM__1_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__CasesM______elabRules__Mathlib__Tactic__casesM__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__CasesM______elabRules__Mathlib__Tactic__casesM__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Mathlib_Tactic_casesm_x21___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "casesm!"};
static const lean_object* lp_mathlib_Mathlib_Tactic_casesm_x21___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_casesm_x21___closed__0_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_casesm_x21___closed__1_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_casesM___closed__0_value),LEAN_SCALAR_PTR_LITERAL(118, 213, 161, 2, 73, 184, 31, 228)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_casesm_x21___closed__1_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_casesm_x21___closed__1_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic_casesM___closed__1_value),LEAN_SCALAR_PTR_LITERAL(139, 222, 98, 232, 116, 132, 69, 249)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_casesm_x21___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_casesm_x21___closed__1_value_aux_1),((lean_object*)&lp_mathlib_Mathlib_Tactic_casesm_x21___closed__0_value),LEAN_SCALAR_PTR_LITERAL(128, 93, 151, 244, 37, 106, 63, 21)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_casesm_x21___closed__1 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_casesm_x21___closed__1_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_casesm_x21___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 8, .m_other = 1, .m_tag = 6}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_casesm_x21___closed__0_value),LEAN_SCALAR_PTR_LITERAL(0, 0, 0, 0, 0, 0, 0, 0)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_casesm_x21___closed__2 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_casesm_x21___closed__2_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_casesm_x21___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_casesM___closed__5_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_casesm_x21___closed__2_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_casesM___closed__12_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_casesm_x21___closed__3 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_casesm_x21___closed__3_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_casesm_x21___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_casesM___closed__5_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_casesm_x21___closed__3_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_casesM___closed__19_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_casesm_x21___closed__4 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_casesm_x21___closed__4_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_casesm_x21___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_casesM___closed__5_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_casesm_x21___closed__4_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_casesM___closed__27_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_casesm_x21___closed__5 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_casesm_x21___closed__5_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_casesm_x21___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 3}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_casesm_x21___closed__1_value),((lean_object*)(((size_t)(1022) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_casesm_x21___closed__5_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_casesm_x21___closed__6 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_casesm_x21___closed__6_value;
LEAN_EXPORT const lean_object* lp_mathlib_Mathlib_Tactic_casesm_x21 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_casesm_x21___closed__6_value;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__CasesM______elabRules__Mathlib__Tactic__casesm_x21__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__CasesM______elabRules__Mathlib__Tactic__casesm_x21__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_elabCasesType___lam__0(lean_object*, uint8_t, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_elabCasesType___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Mathlib_Tactic_elabCasesType_spec__0___redArg(size_t, size_t, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Mathlib_Tactic_elabCasesType_spec__0___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_elabCasesType(lean_object*, uint8_t, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_elabCasesType___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Mathlib_Tactic_elabCasesType_spec__0(size_t, size_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Mathlib_Tactic_elabCasesType_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Mathlib_Tactic_casesType___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 9, .m_data = "casesType"};
static const lean_object* lp_mathlib_Mathlib_Tactic_casesType___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_casesType___closed__0_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_casesType___closed__1_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_casesM___closed__0_value),LEAN_SCALAR_PTR_LITERAL(118, 213, 161, 2, 73, 184, 31, 228)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_casesType___closed__1_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_casesType___closed__1_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic_casesM___closed__1_value),LEAN_SCALAR_PTR_LITERAL(139, 222, 98, 232, 116, 132, 69, 249)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_casesType___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_casesType___closed__1_value_aux_1),((lean_object*)&lp_mathlib_Mathlib_Tactic_casesType___closed__0_value),LEAN_SCALAR_PTR_LITERAL(150, 232, 147, 132, 178, 241, 201, 120)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_casesType___closed__1 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_casesType___closed__1_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_casesType___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 11, .m_capacity = 11, .m_length = 10, .m_data = "cases_type"};
static const lean_object* lp_mathlib_Mathlib_Tactic_casesType___closed__2 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_casesType___closed__2_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_casesType___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 8, .m_other = 1, .m_tag = 6}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_casesType___closed__2_value),LEAN_SCALAR_PTR_LITERAL(0, 0, 0, 0, 0, 0, 0, 0)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_casesType___closed__3 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_casesType___closed__3_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_casesType___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_casesM___closed__5_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_casesType___closed__3_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_casesM___closed__12_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_casesType___closed__4 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_casesType___closed__4_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_casesType___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "many1"};
static const lean_object* lp_mathlib_Mathlib_Tactic_casesType___closed__5 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_casesType___closed__5_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_casesType___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_casesType___closed__5_value),LEAN_SCALAR_PTR_LITERAL(55, 136, 52, 6, 12, 19, 78, 239)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_casesType___closed__6 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_casesType___closed__6_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_casesType___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "colGt"};
static const lean_object* lp_mathlib_Mathlib_Tactic_casesType___closed__7 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_casesType___closed__7_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_casesType___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_casesType___closed__7_value),LEAN_SCALAR_PTR_LITERAL(185, 236, 32, 153, 169, 213, 53, 244)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_casesType___closed__8 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_casesType___closed__8_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_casesType___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_casesType___closed__8_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_casesType___closed__9 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_casesType___closed__9_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_casesType___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_casesM___closed__5_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_casesM___closed__18_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_casesType___closed__9_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_casesType___closed__10 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_casesType___closed__10_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_casesType___closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "ident"};
static const lean_object* lp_mathlib_Mathlib_Tactic_casesType___closed__11 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_casesType___closed__11_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_casesType___closed__12_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_casesType___closed__11_value),LEAN_SCALAR_PTR_LITERAL(52, 159, 208, 51, 14, 60, 6, 71)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_casesType___closed__12 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_casesType___closed__12_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_casesType___closed__13_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_casesType___closed__12_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_casesType___closed__13 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_casesType___closed__13_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_casesType___closed__14_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_casesM___closed__5_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_casesType___closed__10_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_casesType___closed__13_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_casesType___closed__14 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_casesType___closed__14_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_casesType___closed__15_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_casesType___closed__6_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_casesType___closed__14_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_casesType___closed__15 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_casesType___closed__15_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_casesType___closed__16_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_casesM___closed__5_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_casesType___closed__4_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_casesType___closed__15_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_casesType___closed__16 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_casesType___closed__16_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_casesType___closed__17_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 3}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_casesType___closed__1_value),((lean_object*)(((size_t)(1022) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_casesType___closed__16_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_casesType___closed__17 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_casesType___closed__17_value;
LEAN_EXPORT const lean_object* lp_mathlib_Mathlib_Tactic_casesType = (const lean_object*)&lp_mathlib_Mathlib_Tactic_casesType___closed__17_value;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__CasesM______elabRules__Mathlib__Tactic__casesType__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__CasesM______elabRules__Mathlib__Tactic__casesType__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Mathlib_Tactic_casesType_x21___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 11, .m_capacity = 11, .m_length = 10, .m_data = "casesType!"};
static const lean_object* lp_mathlib_Mathlib_Tactic_casesType_x21___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_casesType_x21___closed__0_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_casesType_x21___closed__1_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_casesM___closed__0_value),LEAN_SCALAR_PTR_LITERAL(118, 213, 161, 2, 73, 184, 31, 228)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_casesType_x21___closed__1_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_casesType_x21___closed__1_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic_casesM___closed__1_value),LEAN_SCALAR_PTR_LITERAL(139, 222, 98, 232, 116, 132, 69, 249)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_casesType_x21___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_casesType_x21___closed__1_value_aux_1),((lean_object*)&lp_mathlib_Mathlib_Tactic_casesType_x21___closed__0_value),LEAN_SCALAR_PTR_LITERAL(184, 189, 34, 139, 29, 237, 203, 113)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_casesType_x21___closed__1 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_casesType_x21___closed__1_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_casesType_x21___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 12, .m_capacity = 12, .m_length = 11, .m_data = "cases_type!"};
static const lean_object* lp_mathlib_Mathlib_Tactic_casesType_x21___closed__2 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_casesType_x21___closed__2_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_casesType_x21___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 8, .m_other = 1, .m_tag = 6}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_casesType_x21___closed__2_value),LEAN_SCALAR_PTR_LITERAL(0, 0, 0, 0, 0, 0, 0, 0)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_casesType_x21___closed__3 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_casesType_x21___closed__3_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_casesType_x21___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_casesM___closed__5_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_casesType_x21___closed__3_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_casesM___closed__12_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_casesType_x21___closed__4 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_casesType_x21___closed__4_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_casesType_x21___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_casesM___closed__5_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_casesType_x21___closed__4_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_casesType___closed__15_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_casesType_x21___closed__5 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_casesType_x21___closed__5_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_casesType_x21___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 3}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_casesType_x21___closed__1_value),((lean_object*)(((size_t)(1022) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_casesType_x21___closed__5_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_casesType_x21___closed__6 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_casesType_x21___closed__6_value;
LEAN_EXPORT const lean_object* lp_mathlib_Mathlib_Tactic_casesType_x21 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_casesType_x21___closed__6_value;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__CasesM______elabRules__Mathlib__Tactic__casesType_x21__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__CasesM______elabRules__Mathlib__Tactic__casesType_x21__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_forIn_x27_loop___at___00__private_Mathlib_Tactic_CasesM_0__Mathlib_Tactic_constructorMatching_go_spec__0___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_CasesM_0__Mathlib_Tactic_constructorMatching_go___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_CasesM_0__Mathlib_Tactic_constructorMatching_go___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_CasesM_0__Mathlib_Tactic_constructorMatching_go(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_CasesM_0__Mathlib_Tactic_constructorMatching_go___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_forIn_x27_loop___at___00__private_Mathlib_Tactic_CasesM_0__Mathlib_Tactic_constructorMatching_go_spec__0___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_forIn_x27_loop___at___00__private_Mathlib_Tactic_CasesM_0__Mathlib_Tactic_constructorMatching_go_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_forIn_x27_loop___at___00__private_Mathlib_Tactic_CasesM_0__Mathlib_Tactic_constructorMatching_go_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_constructorMatching___lam__0(lean_object*, lean_object*, uint8_t, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_constructorMatching___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_constructorMatching(lean_object*, lean_object*, uint8_t, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_constructorMatching___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Mathlib_Tactic_constructorM___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 13, .m_capacity = 13, .m_length = 12, .m_data = "constructorM"};
static const lean_object* lp_mathlib_Mathlib_Tactic_constructorM___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_constructorM___closed__0_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_constructorM___closed__1_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_casesM___closed__0_value),LEAN_SCALAR_PTR_LITERAL(118, 213, 161, 2, 73, 184, 31, 228)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_constructorM___closed__1_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_constructorM___closed__1_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic_casesM___closed__1_value),LEAN_SCALAR_PTR_LITERAL(139, 222, 98, 232, 116, 132, 69, 249)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_constructorM___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_constructorM___closed__1_value_aux_1),((lean_object*)&lp_mathlib_Mathlib_Tactic_constructorM___closed__0_value),LEAN_SCALAR_PTR_LITERAL(197, 74, 144, 117, 43, 80, 197, 224)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_constructorM___closed__1 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_constructorM___closed__1_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_constructorM___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 13, .m_capacity = 13, .m_length = 12, .m_data = "constructorm"};
static const lean_object* lp_mathlib_Mathlib_Tactic_constructorM___closed__2 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_constructorM___closed__2_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_constructorM___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 8, .m_other = 1, .m_tag = 6}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_constructorM___closed__2_value),LEAN_SCALAR_PTR_LITERAL(0, 0, 0, 0, 0, 0, 0, 0)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_constructorM___closed__3 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_constructorM___closed__3_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_constructorM___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_casesM___closed__5_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_constructorM___closed__3_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_casesM___closed__12_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_constructorM___closed__4 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_constructorM___closed__4_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_constructorM___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_casesM___closed__5_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_constructorM___closed__4_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_casesM___closed__19_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_constructorM___closed__5 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_constructorM___closed__5_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_constructorM___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_casesM___closed__5_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_constructorM___closed__5_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_casesM___closed__27_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_constructorM___closed__6 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_constructorM___closed__6_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_constructorM___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 3}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_constructorM___closed__1_value),((lean_object*)(((size_t)(1022) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_constructorM___closed__6_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_constructorM___closed__7 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_constructorM___closed__7_value;
LEAN_EXPORT const lean_object* lp_mathlib_Mathlib_Tactic_constructorM = (const lean_object*)&lp_mathlib_Mathlib_Tactic_constructorM___closed__7_value;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__CasesM______elabRules__Mathlib__Tactic__constructorM__1___lam__0(lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__CasesM______elabRules__Mathlib__Tactic__constructorM__1___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__CasesM______elabRules__Mathlib__Tactic__constructorM__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__CasesM______elabRules__Mathlib__Tactic__constructorM__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_withContext___at___00__private_Mathlib_Tactic_CasesM_0__Lean_MVarId_casesMatching_go_spec__2___redArg(lean_object* v_mvarId_1_, lean_object* v_x_2_, lean_object* v___y_3_, lean_object* v___y_4_, lean_object* v___y_5_, lean_object* v___y_6_){
_start:
{
lean_object* v___x_8_; 
v___x_8_ = l___private_Lean_Meta_Basic_0__Lean_Meta_withMVarContextImp(lean_box(0), v_mvarId_1_, v_x_2_, v___y_3_, v___y_4_, v___y_5_, v___y_6_);
if (lean_obj_tag(v___x_8_) == 0)
{
lean_object* v_a_9_; lean_object* v___x_11_; uint8_t v_isShared_12_; uint8_t v_isSharedCheck_16_; 
v_a_9_ = lean_ctor_get(v___x_8_, 0);
v_isSharedCheck_16_ = !lean_is_exclusive(v___x_8_);
if (v_isSharedCheck_16_ == 0)
{
v___x_11_ = v___x_8_;
v_isShared_12_ = v_isSharedCheck_16_;
goto v_resetjp_10_;
}
else
{
lean_inc(v_a_9_);
lean_dec(v___x_8_);
v___x_11_ = lean_box(0);
v_isShared_12_ = v_isSharedCheck_16_;
goto v_resetjp_10_;
}
v_resetjp_10_:
{
lean_object* v___x_14_; 
if (v_isShared_12_ == 0)
{
v___x_14_ = v___x_11_;
goto v_reusejp_13_;
}
else
{
lean_object* v_reuseFailAlloc_15_; 
v_reuseFailAlloc_15_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_15_, 0, v_a_9_);
v___x_14_ = v_reuseFailAlloc_15_;
goto v_reusejp_13_;
}
v_reusejp_13_:
{
return v___x_14_;
}
}
}
else
{
lean_object* v_a_17_; lean_object* v___x_19_; uint8_t v_isShared_20_; uint8_t v_isSharedCheck_24_; 
v_a_17_ = lean_ctor_get(v___x_8_, 0);
v_isSharedCheck_24_ = !lean_is_exclusive(v___x_8_);
if (v_isSharedCheck_24_ == 0)
{
v___x_19_ = v___x_8_;
v_isShared_20_ = v_isSharedCheck_24_;
goto v_resetjp_18_;
}
else
{
lean_inc(v_a_17_);
lean_dec(v___x_8_);
v___x_19_ = lean_box(0);
v_isShared_20_ = v_isSharedCheck_24_;
goto v_resetjp_18_;
}
v_resetjp_18_:
{
lean_object* v___x_22_; 
if (v_isShared_20_ == 0)
{
v___x_22_ = v___x_19_;
goto v_reusejp_21_;
}
else
{
lean_object* v_reuseFailAlloc_23_; 
v_reuseFailAlloc_23_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_23_, 0, v_a_17_);
v___x_22_ = v_reuseFailAlloc_23_;
goto v_reusejp_21_;
}
v_reusejp_21_:
{
return v___x_22_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_withContext___at___00__private_Mathlib_Tactic_CasesM_0__Lean_MVarId_casesMatching_go_spec__2___redArg___boxed(lean_object* v_mvarId_25_, lean_object* v_x_26_, lean_object* v___y_27_, lean_object* v___y_28_, lean_object* v___y_29_, lean_object* v___y_30_, lean_object* v___y_31_){
_start:
{
lean_object* v_res_32_; 
v_res_32_ = lp_mathlib_Lean_MVarId_withContext___at___00__private_Mathlib_Tactic_CasesM_0__Lean_MVarId_casesMatching_go_spec__2___redArg(v_mvarId_25_, v_x_26_, v___y_27_, v___y_28_, v___y_29_, v___y_30_);
lean_dec(v___y_30_);
lean_dec_ref(v___y_29_);
lean_dec(v___y_28_);
lean_dec_ref(v___y_27_);
return v_res_32_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_withContext___at___00__private_Mathlib_Tactic_CasesM_0__Lean_MVarId_casesMatching_go_spec__2(lean_object* v_00_u03b1_33_, lean_object* v_mvarId_34_, lean_object* v_x_35_, lean_object* v___y_36_, lean_object* v___y_37_, lean_object* v___y_38_, lean_object* v___y_39_){
_start:
{
lean_object* v___x_41_; 
v___x_41_ = lp_mathlib_Lean_MVarId_withContext___at___00__private_Mathlib_Tactic_CasesM_0__Lean_MVarId_casesMatching_go_spec__2___redArg(v_mvarId_34_, v_x_35_, v___y_36_, v___y_37_, v___y_38_, v___y_39_);
return v___x_41_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_withContext___at___00__private_Mathlib_Tactic_CasesM_0__Lean_MVarId_casesMatching_go_spec__2___boxed(lean_object* v_00_u03b1_42_, lean_object* v_mvarId_43_, lean_object* v_x_44_, lean_object* v___y_45_, lean_object* v___y_46_, lean_object* v___y_47_, lean_object* v___y_48_, lean_object* v___y_49_){
_start:
{
lean_object* v_res_50_; 
v_res_50_ = lp_mathlib_Lean_MVarId_withContext___at___00__private_Mathlib_Tactic_CasesM_0__Lean_MVarId_casesMatching_go_spec__2(v_00_u03b1_42_, v_mvarId_43_, v_x_44_, v___y_45_, v___y_46_, v___y_47_, v___y_48_);
lean_dec(v___y_48_);
lean_dec_ref(v___y_47_);
lean_dec(v___y_46_);
lean_dec_ref(v___y_45_);
return v_res_50_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00__private_Mathlib_Tactic_CasesM_0__Lean_MVarId_casesMatching_go_spec__1_spec__1_spec__3(lean_object* v_init_54_, uint8_t v_recursive_55_, lean_object* v_matcher_56_, uint8_t v_allowSplit_57_, lean_object* v_acc_58_, lean_object* v_g_59_, lean_object* v_as_60_, size_t v_sz_61_, size_t v_i_62_, lean_object* v_b_63_, lean_object* v___y_64_, lean_object* v___y_65_, lean_object* v___y_66_, lean_object* v___y_67_){
_start:
{
uint8_t v___x_69_; 
v___x_69_ = lean_usize_dec_lt(v_i_62_, v_sz_61_);
if (v___x_69_ == 0)
{
lean_object* v___x_70_; 
lean_dec(v_g_59_);
lean_dec_ref(v_acc_58_);
lean_dec_ref(v_matcher_56_);
v___x_70_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_70_, 0, v_b_63_);
return v___x_70_;
}
else
{
lean_object* v_snd_71_; lean_object* v___x_73_; uint8_t v_isShared_74_; uint8_t v_isSharedCheck_105_; 
v_snd_71_ = lean_ctor_get(v_b_63_, 1);
v_isSharedCheck_105_ = !lean_is_exclusive(v_b_63_);
if (v_isSharedCheck_105_ == 0)
{
lean_object* v_unused_106_; 
v_unused_106_ = lean_ctor_get(v_b_63_, 0);
lean_dec(v_unused_106_);
v___x_73_ = v_b_63_;
v_isShared_74_ = v_isSharedCheck_105_;
goto v_resetjp_72_;
}
else
{
lean_inc(v_snd_71_);
lean_dec(v_b_63_);
v___x_73_ = lean_box(0);
v_isShared_74_ = v_isSharedCheck_105_;
goto v_resetjp_72_;
}
v_resetjp_72_:
{
lean_object* v_a_75_; lean_object* v___x_76_; 
v_a_75_ = lean_array_uget_borrowed(v_as_60_, v_i_62_);
lean_inc(v_snd_71_);
lean_inc(v_g_59_);
lean_inc_ref(v_acc_58_);
lean_inc_ref(v_matcher_56_);
v___x_76_ = lp_mathlib_Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00__private_Mathlib_Tactic_CasesM_0__Lean_MVarId_casesMatching_go_spec__1_spec__1(v_init_54_, v_recursive_55_, v_matcher_56_, v_allowSplit_57_, v_acc_58_, v_g_59_, v_a_75_, v_snd_71_, v___y_64_, v___y_65_, v___y_66_, v___y_67_);
if (lean_obj_tag(v___x_76_) == 0)
{
lean_object* v_a_77_; lean_object* v___x_79_; uint8_t v_isShared_80_; uint8_t v_isSharedCheck_96_; 
v_a_77_ = lean_ctor_get(v___x_76_, 0);
v_isSharedCheck_96_ = !lean_is_exclusive(v___x_76_);
if (v_isSharedCheck_96_ == 0)
{
v___x_79_ = v___x_76_;
v_isShared_80_ = v_isSharedCheck_96_;
goto v_resetjp_78_;
}
else
{
lean_inc(v_a_77_);
lean_dec(v___x_76_);
v___x_79_ = lean_box(0);
v_isShared_80_ = v_isSharedCheck_96_;
goto v_resetjp_78_;
}
v_resetjp_78_:
{
if (lean_obj_tag(v_a_77_) == 0)
{
lean_object* v___x_81_; lean_object* v___x_83_; 
lean_dec(v_g_59_);
lean_dec_ref(v_acc_58_);
lean_dec_ref(v_matcher_56_);
v___x_81_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_81_, 0, v_a_77_);
if (v_isShared_74_ == 0)
{
lean_ctor_set(v___x_73_, 0, v___x_81_);
v___x_83_ = v___x_73_;
goto v_reusejp_82_;
}
else
{
lean_object* v_reuseFailAlloc_87_; 
v_reuseFailAlloc_87_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_87_, 0, v___x_81_);
lean_ctor_set(v_reuseFailAlloc_87_, 1, v_snd_71_);
v___x_83_ = v_reuseFailAlloc_87_;
goto v_reusejp_82_;
}
v_reusejp_82_:
{
lean_object* v___x_85_; 
if (v_isShared_80_ == 0)
{
lean_ctor_set(v___x_79_, 0, v___x_83_);
v___x_85_ = v___x_79_;
goto v_reusejp_84_;
}
else
{
lean_object* v_reuseFailAlloc_86_; 
v_reuseFailAlloc_86_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_86_, 0, v___x_83_);
v___x_85_ = v_reuseFailAlloc_86_;
goto v_reusejp_84_;
}
v_reusejp_84_:
{
return v___x_85_;
}
}
}
else
{
lean_object* v_a_88_; lean_object* v___x_89_; lean_object* v___x_91_; 
lean_del_object(v___x_79_);
lean_dec(v_snd_71_);
v_a_88_ = lean_ctor_get(v_a_77_, 0);
lean_inc(v_a_88_);
lean_dec_ref_known(v_a_77_, 1);
v___x_89_ = lean_box(0);
if (v_isShared_74_ == 0)
{
lean_ctor_set(v___x_73_, 1, v_a_88_);
lean_ctor_set(v___x_73_, 0, v___x_89_);
v___x_91_ = v___x_73_;
goto v_reusejp_90_;
}
else
{
lean_object* v_reuseFailAlloc_95_; 
v_reuseFailAlloc_95_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_95_, 0, v___x_89_);
lean_ctor_set(v_reuseFailAlloc_95_, 1, v_a_88_);
v___x_91_ = v_reuseFailAlloc_95_;
goto v_reusejp_90_;
}
v_reusejp_90_:
{
size_t v___x_92_; size_t v___x_93_; 
v___x_92_ = ((size_t)1ULL);
v___x_93_ = lean_usize_add(v_i_62_, v___x_92_);
v_i_62_ = v___x_93_;
v_b_63_ = v___x_91_;
goto _start;
}
}
}
}
else
{
lean_object* v_a_97_; lean_object* v___x_99_; uint8_t v_isShared_100_; uint8_t v_isSharedCheck_104_; 
lean_del_object(v___x_73_);
lean_dec(v_snd_71_);
lean_dec(v_g_59_);
lean_dec_ref(v_acc_58_);
lean_dec_ref(v_matcher_56_);
v_a_97_ = lean_ctor_get(v___x_76_, 0);
v_isSharedCheck_104_ = !lean_is_exclusive(v___x_76_);
if (v_isSharedCheck_104_ == 0)
{
v___x_99_ = v___x_76_;
v_isShared_100_ = v_isSharedCheck_104_;
goto v_resetjp_98_;
}
else
{
lean_inc(v_a_97_);
lean_dec(v___x_76_);
v___x_99_ = lean_box(0);
v_isShared_100_ = v_isSharedCheck_104_;
goto v_resetjp_98_;
}
v_resetjp_98_:
{
lean_object* v___x_102_; 
if (v_isShared_100_ == 0)
{
v___x_102_ = v___x_99_;
goto v_reusejp_101_;
}
else
{
lean_object* v_reuseFailAlloc_103_; 
v_reuseFailAlloc_103_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_103_, 0, v_a_97_);
v___x_102_ = v_reuseFailAlloc_103_;
goto v_reusejp_101_;
}
v_reusejp_101_:
{
return v___x_102_;
}
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_CasesM_0__Lean_MVarId_casesMatching_go_spec__0(uint8_t v_recursive_107_, lean_object* v_matcher_108_, uint8_t v_allowSplit_109_, lean_object* v_val_110_, lean_object* v_as_111_, size_t v_sz_112_, size_t v_i_113_, lean_object* v_b_114_, lean_object* v___y_115_, lean_object* v___y_116_, lean_object* v___y_117_, lean_object* v___y_118_){
_start:
{
lean_object* v_a_121_; lean_object* v_g_126_; lean_object* v___y_127_; lean_object* v___y_128_; lean_object* v___y_129_; lean_object* v___y_130_; uint8_t v___x_134_; 
v___x_134_ = lean_usize_dec_lt(v_i_113_, v_sz_112_);
if (v___x_134_ == 0)
{
lean_object* v___x_135_; 
lean_dec_ref(v_matcher_108_);
v___x_135_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_135_, 0, v_b_114_);
return v___x_135_;
}
else
{
lean_object* v_a_136_; lean_object* v___y_138_; lean_object* v___y_139_; lean_object* v___y_140_; lean_object* v___y_141_; lean_object* v_toInductionSubgoal_144_; lean_object* v_mvarId_145_; lean_object* v_fields_146_; lean_object* v___x_147_; lean_object* v___x_148_; uint8_t v___x_149_; 
v_a_136_ = lean_array_uget_borrowed(v_as_111_, v_i_113_);
v_toInductionSubgoal_144_ = lean_ctor_get(v_a_136_, 0);
v_mvarId_145_ = lean_ctor_get(v_toInductionSubgoal_144_, 0);
v_fields_146_ = lean_ctor_get(v_toInductionSubgoal_144_, 1);
v___x_147_ = lean_array_get_size(v_fields_146_);
v___x_148_ = lean_unsigned_to_nat(1u);
v___x_149_ = lean_nat_dec_eq(v___x_147_, v___x_148_);
if (v___x_149_ == 0)
{
v___y_138_ = v___y_115_;
v___y_139_ = v___y_116_;
v___y_140_ = v___y_117_;
v___y_141_ = v___y_118_;
goto v___jp_137_;
}
else
{
lean_object* v___x_150_; lean_object* v___x_151_; 
v___x_150_ = lean_unsigned_to_nat(0u);
v___x_151_ = lean_array_fget_borrowed(v_fields_146_, v___x_150_);
if (lean_obj_tag(v___x_151_) == 1)
{
lean_object* v_fvarId_152_; lean_object* v___x_153_; lean_object* v___x_154_; 
v_fvarId_152_ = lean_ctor_get(v___x_151_, 0);
v___x_153_ = l_Lean_LocalDecl_userName(v_val_110_);
lean_inc(v_fvarId_152_);
lean_inc(v_mvarId_145_);
v___x_154_ = l_Lean_MVarId_rename(v_mvarId_145_, v_fvarId_152_, v___x_153_, v___y_115_, v___y_116_, v___y_117_, v___y_118_);
if (lean_obj_tag(v___x_154_) == 0)
{
lean_object* v_a_155_; 
v_a_155_ = lean_ctor_get(v___x_154_, 0);
lean_inc(v_a_155_);
lean_dec_ref_known(v___x_154_, 1);
v_g_126_ = v_a_155_;
v___y_127_ = v___y_115_;
v___y_128_ = v___y_116_;
v___y_129_ = v___y_117_;
v___y_130_ = v___y_118_;
goto v___jp_125_;
}
else
{
lean_object* v_a_156_; lean_object* v___x_158_; uint8_t v_isShared_159_; uint8_t v_isSharedCheck_163_; 
lean_dec_ref(v_b_114_);
lean_dec_ref(v_matcher_108_);
v_a_156_ = lean_ctor_get(v___x_154_, 0);
v_isSharedCheck_163_ = !lean_is_exclusive(v___x_154_);
if (v_isSharedCheck_163_ == 0)
{
v___x_158_ = v___x_154_;
v_isShared_159_ = v_isSharedCheck_163_;
goto v_resetjp_157_;
}
else
{
lean_inc(v_a_156_);
lean_dec(v___x_154_);
v___x_158_ = lean_box(0);
v_isShared_159_ = v_isSharedCheck_163_;
goto v_resetjp_157_;
}
v_resetjp_157_:
{
lean_object* v___x_161_; 
if (v_isShared_159_ == 0)
{
v___x_161_ = v___x_158_;
goto v_reusejp_160_;
}
else
{
lean_object* v_reuseFailAlloc_162_; 
v_reuseFailAlloc_162_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_162_, 0, v_a_156_);
v___x_161_ = v_reuseFailAlloc_162_;
goto v_reusejp_160_;
}
v_reusejp_160_:
{
return v___x_161_;
}
}
}
}
else
{
v___y_138_ = v___y_115_;
v___y_139_ = v___y_116_;
v___y_140_ = v___y_117_;
v___y_141_ = v___y_118_;
goto v___jp_137_;
}
}
v___jp_137_:
{
lean_object* v_toInductionSubgoal_142_; lean_object* v_mvarId_143_; 
v_toInductionSubgoal_142_ = lean_ctor_get(v_a_136_, 0);
v_mvarId_143_ = lean_ctor_get(v_toInductionSubgoal_142_, 0);
lean_inc(v_mvarId_143_);
v_g_126_ = v_mvarId_143_;
v___y_127_ = v___y_138_;
v___y_128_ = v___y_139_;
v___y_129_ = v___y_140_;
v___y_130_ = v___y_141_;
goto v___jp_125_;
}
}
v___jp_120_:
{
size_t v___x_122_; size_t v___x_123_; 
v___x_122_ = ((size_t)1ULL);
v___x_123_ = lean_usize_add(v_i_113_, v___x_122_);
v_i_113_ = v___x_123_;
v_b_114_ = v_a_121_;
goto _start;
}
v___jp_125_:
{
if (v_recursive_107_ == 0)
{
lean_object* v___x_131_; 
v___x_131_ = lean_array_push(v_b_114_, v_g_126_);
v_a_121_ = v___x_131_;
goto v___jp_120_;
}
else
{
lean_object* v___x_132_; 
lean_inc_ref(v_matcher_108_);
v___x_132_ = lp_mathlib___private_Mathlib_Tactic_CasesM_0__Lean_MVarId_casesMatching_go(v_matcher_108_, v_recursive_107_, v_allowSplit_109_, v_g_126_, v_b_114_, v___y_127_, v___y_128_, v___y_129_, v___y_130_);
if (lean_obj_tag(v___x_132_) == 0)
{
lean_object* v_a_133_; 
v_a_133_ = lean_ctor_get(v___x_132_, 0);
lean_inc(v_a_133_);
lean_dec_ref_known(v___x_132_, 1);
v_a_121_ = v_a_133_;
goto v___jp_120_;
}
else
{
lean_dec_ref(v_matcher_108_);
return v___x_132_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00__private_Mathlib_Tactic_CasesM_0__Lean_MVarId_casesMatching_go_spec__1_spec__1_spec__4_spec__5(uint8_t v_recursive_169_, lean_object* v_matcher_170_, uint8_t v_allowSplit_171_, lean_object* v_acc_172_, lean_object* v_g_173_, lean_object* v_as_174_, size_t v_sz_175_, size_t v_i_176_, lean_object* v_b_177_, lean_object* v___y_178_, lean_object* v___y_179_, lean_object* v___y_180_, lean_object* v___y_181_){
_start:
{
uint8_t v___x_183_; 
v___x_183_ = lean_usize_dec_lt(v_i_176_, v_sz_175_);
if (v___x_183_ == 0)
{
lean_object* v___x_184_; 
lean_dec(v_g_173_);
lean_dec_ref(v_acc_172_);
lean_dec_ref(v_matcher_170_);
v___x_184_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_184_, 0, v_b_177_);
return v___x_184_;
}
else
{
lean_object* v_snd_185_; lean_object* v___x_187_; uint8_t v_isShared_188_; uint8_t v_isSharedCheck_303_; 
v_snd_185_ = lean_ctor_get(v_b_177_, 1);
v_isSharedCheck_303_ = !lean_is_exclusive(v_b_177_);
if (v_isSharedCheck_303_ == 0)
{
lean_object* v_unused_304_; 
v_unused_304_ = lean_ctor_get(v_b_177_, 0);
lean_dec(v_unused_304_);
v___x_187_ = v_b_177_;
v_isShared_188_ = v_isSharedCheck_303_;
goto v_resetjp_186_;
}
else
{
lean_inc(v_snd_185_);
lean_dec(v_b_177_);
v___x_187_ = lean_box(0);
v_isShared_188_ = v_isSharedCheck_303_;
goto v_resetjp_186_;
}
v_resetjp_186_:
{
lean_object* v___x_189_; lean_object* v_a_191_; lean_object* v_a_198_; 
v___x_189_ = lean_box(0);
v_a_198_ = lean_array_uget(v_as_174_, v_i_176_);
if (lean_obj_tag(v_a_198_) == 0)
{
v_a_191_ = v_snd_185_;
goto v___jp_190_;
}
else
{
lean_object* v_val_199_; lean_object* v___x_201_; uint8_t v_isShared_202_; uint8_t v_isSharedCheck_302_; 
v_val_199_ = lean_ctor_get(v_a_198_, 0);
v_isSharedCheck_302_ = !lean_is_exclusive(v_a_198_);
if (v_isSharedCheck_302_ == 0)
{
v___x_201_ = v_a_198_;
v_isShared_202_ = v_isSharedCheck_302_;
goto v_resetjp_200_;
}
else
{
lean_inc(v_val_199_);
lean_dec(v_a_198_);
v___x_201_ = lean_box(0);
v_isShared_202_ = v_isSharedCheck_302_;
goto v_resetjp_200_;
}
v_resetjp_200_:
{
lean_object* v___x_203_; lean_object* v_subgoals_205_; lean_object* v___y_206_; lean_object* v___y_207_; lean_object* v___y_208_; lean_object* v___y_209_; lean_object* v___x_236_; uint8_t v___x_237_; 
v___x_203_ = lean_box(0);
v___x_236_ = ((lean_object*)(lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00__private_Mathlib_Tactic_CasesM_0__Lean_MVarId_casesMatching_go_spec__1_spec__1_spec__4___closed__0));
v___x_237_ = l_Lean_LocalDecl_isImplementationDetail(v_val_199_);
if (v___x_237_ == 0)
{
lean_object* v___x_238_; lean_object* v___x_239_; 
v___x_238_ = l_Lean_LocalDecl_type(v_val_199_);
lean_inc_ref(v_matcher_170_);
lean_inc(v___y_181_);
lean_inc_ref(v___y_180_);
lean_inc(v___y_179_);
lean_inc_ref(v___y_178_);
v___x_239_ = lean_apply_6(v_matcher_170_, v___x_238_, v___y_178_, v___y_179_, v___y_180_, v___y_181_, lean_box(0));
if (lean_obj_tag(v___x_239_) == 0)
{
lean_object* v_a_240_; uint8_t v___x_241_; 
v_a_240_ = lean_ctor_get(v___x_239_, 0);
lean_inc(v_a_240_);
lean_dec_ref_known(v___x_239_, 1);
v___x_241_ = lean_unbox(v_a_240_);
if (v___x_241_ == 0)
{
lean_dec(v_a_240_);
lean_del_object(v___x_201_);
lean_dec(v_val_199_);
lean_dec(v_snd_185_);
v_a_191_ = v___x_236_;
goto v___jp_190_;
}
else
{
if (v_allowSplit_171_ == 0)
{
lean_object* v___x_242_; 
v___x_242_ = l_Lean_Meta_saveState___redArg(v___y_179_, v___y_181_);
if (lean_obj_tag(v___x_242_) == 0)
{
lean_object* v_a_243_; lean_object* v___x_244_; lean_object* v___x_245_; lean_object* v___x_246_; lean_object* v___x_247_; lean_object* v___x_248_; uint8_t v___x_249_; lean_object* v___x_250_; lean_object* v___x_251_; lean_object* v___x_252_; lean_object* v___x_253_; 
v_a_243_ = lean_ctor_get(v___x_242_, 0);
lean_inc(v_a_243_);
lean_dec_ref_known(v___x_242_, 1);
v___x_244_ = l_Lean_LocalDecl_fvarId(v_val_199_);
v___x_245_ = l_Lean_LocalDecl_userName(v_val_199_);
v___x_246_ = lean_box(0);
v___x_247_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_247_, 0, v___x_245_);
lean_ctor_set(v___x_247_, 1, v___x_246_);
v___x_248_ = lean_alloc_ctor(0, 1, 1);
lean_ctor_set(v___x_248_, 0, v___x_247_);
v___x_249_ = lean_unbox(v_a_240_);
lean_dec(v_a_240_);
lean_ctor_set_uint8(v___x_248_, sizeof(void*)*1, v___x_249_);
v___x_250_ = lean_unsigned_to_nat(1u);
v___x_251_ = lean_mk_empty_array_with_capacity(v___x_250_);
v___x_252_ = lean_array_push(v___x_251_, v___x_248_);
lean_inc(v_g_173_);
v___x_253_ = l_Lean_MVarId_cases(v_g_173_, v___x_244_, v___x_252_, v_allowSplit_171_, v___x_189_, v___y_178_, v___y_179_, v___y_180_, v___y_181_);
if (lean_obj_tag(v___x_253_) == 0)
{
lean_object* v_a_254_; lean_object* v___x_255_; uint8_t v___x_256_; 
v_a_254_ = lean_ctor_get(v___x_253_, 0);
lean_inc(v_a_254_);
lean_dec_ref_known(v___x_253_, 1);
v___x_255_ = lean_array_get_size(v_a_254_);
v___x_256_ = lean_nat_dec_lt(v___x_250_, v___x_255_);
if (v___x_256_ == 0)
{
lean_dec(v_a_243_);
lean_del_object(v___x_187_);
lean_dec(v_g_173_);
v_subgoals_205_ = v_a_254_;
v___y_206_ = v___y_178_;
v___y_207_ = v___y_179_;
v___y_208_ = v___y_180_;
v___y_209_ = v___y_181_;
goto v___jp_204_;
}
else
{
lean_object* v___x_257_; 
lean_dec(v_a_254_);
lean_del_object(v___x_201_);
lean_dec(v_val_199_);
lean_dec(v_snd_185_);
v___x_257_ = l_Lean_Meta_SavedState_restore___redArg(v_a_243_, v___y_179_, v___y_181_);
lean_dec(v_a_243_);
if (lean_obj_tag(v___x_257_) == 0)
{
lean_dec_ref_known(v___x_257_, 1);
v_a_191_ = v___x_236_;
goto v___jp_190_;
}
else
{
lean_object* v_a_258_; lean_object* v___x_260_; uint8_t v_isShared_261_; uint8_t v_isSharedCheck_265_; 
lean_del_object(v___x_187_);
lean_dec(v_g_173_);
lean_dec_ref(v_acc_172_);
lean_dec_ref(v_matcher_170_);
v_a_258_ = lean_ctor_get(v___x_257_, 0);
v_isSharedCheck_265_ = !lean_is_exclusive(v___x_257_);
if (v_isSharedCheck_265_ == 0)
{
v___x_260_ = v___x_257_;
v_isShared_261_ = v_isSharedCheck_265_;
goto v_resetjp_259_;
}
else
{
lean_inc(v_a_258_);
lean_dec(v___x_257_);
v___x_260_ = lean_box(0);
v_isShared_261_ = v_isSharedCheck_265_;
goto v_resetjp_259_;
}
v_resetjp_259_:
{
lean_object* v___x_263_; 
if (v_isShared_261_ == 0)
{
v___x_263_ = v___x_260_;
goto v_reusejp_262_;
}
else
{
lean_object* v_reuseFailAlloc_264_; 
v_reuseFailAlloc_264_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_264_, 0, v_a_258_);
v___x_263_ = v_reuseFailAlloc_264_;
goto v_reusejp_262_;
}
v_reusejp_262_:
{
return v___x_263_;
}
}
}
}
}
else
{
lean_object* v_a_266_; lean_object* v___x_268_; uint8_t v_isShared_269_; uint8_t v_isSharedCheck_273_; 
lean_dec(v_a_243_);
lean_del_object(v___x_201_);
lean_dec(v_val_199_);
lean_del_object(v___x_187_);
lean_dec(v_snd_185_);
lean_dec(v_g_173_);
lean_dec_ref(v_acc_172_);
lean_dec_ref(v_matcher_170_);
v_a_266_ = lean_ctor_get(v___x_253_, 0);
v_isSharedCheck_273_ = !lean_is_exclusive(v___x_253_);
if (v_isSharedCheck_273_ == 0)
{
v___x_268_ = v___x_253_;
v_isShared_269_ = v_isSharedCheck_273_;
goto v_resetjp_267_;
}
else
{
lean_inc(v_a_266_);
lean_dec(v___x_253_);
v___x_268_ = lean_box(0);
v_isShared_269_ = v_isSharedCheck_273_;
goto v_resetjp_267_;
}
v_resetjp_267_:
{
lean_object* v___x_271_; 
if (v_isShared_269_ == 0)
{
v___x_271_ = v___x_268_;
goto v_reusejp_270_;
}
else
{
lean_object* v_reuseFailAlloc_272_; 
v_reuseFailAlloc_272_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_272_, 0, v_a_266_);
v___x_271_ = v_reuseFailAlloc_272_;
goto v_reusejp_270_;
}
v_reusejp_270_:
{
return v___x_271_;
}
}
}
}
else
{
lean_object* v_a_274_; lean_object* v___x_276_; uint8_t v_isShared_277_; uint8_t v_isSharedCheck_281_; 
lean_dec(v_a_240_);
lean_del_object(v___x_201_);
lean_dec(v_val_199_);
lean_del_object(v___x_187_);
lean_dec(v_snd_185_);
lean_dec(v_g_173_);
lean_dec_ref(v_acc_172_);
lean_dec_ref(v_matcher_170_);
v_a_274_ = lean_ctor_get(v___x_242_, 0);
v_isSharedCheck_281_ = !lean_is_exclusive(v___x_242_);
if (v_isSharedCheck_281_ == 0)
{
v___x_276_ = v___x_242_;
v_isShared_277_ = v_isSharedCheck_281_;
goto v_resetjp_275_;
}
else
{
lean_inc(v_a_274_);
lean_dec(v___x_242_);
v___x_276_ = lean_box(0);
v_isShared_277_ = v_isSharedCheck_281_;
goto v_resetjp_275_;
}
v_resetjp_275_:
{
lean_object* v___x_279_; 
if (v_isShared_277_ == 0)
{
v___x_279_ = v___x_276_;
goto v_reusejp_278_;
}
else
{
lean_object* v_reuseFailAlloc_280_; 
v_reuseFailAlloc_280_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_280_, 0, v_a_274_);
v___x_279_ = v_reuseFailAlloc_280_;
goto v_reusejp_278_;
}
v_reusejp_278_:
{
return v___x_279_;
}
}
}
}
else
{
lean_object* v___x_282_; lean_object* v___x_283_; lean_object* v___x_284_; 
lean_dec(v_a_240_);
lean_del_object(v___x_187_);
v___x_282_ = l_Lean_LocalDecl_fvarId(v_val_199_);
v___x_283_ = ((lean_object*)(lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00__private_Mathlib_Tactic_CasesM_0__Lean_MVarId_casesMatching_go_spec__1_spec__2___closed__1));
v___x_284_ = l_Lean_MVarId_cases(v_g_173_, v___x_282_, v___x_283_, v___x_237_, v___x_189_, v___y_178_, v___y_179_, v___y_180_, v___y_181_);
if (lean_obj_tag(v___x_284_) == 0)
{
lean_object* v_a_285_; 
v_a_285_ = lean_ctor_get(v___x_284_, 0);
lean_inc(v_a_285_);
lean_dec_ref_known(v___x_284_, 1);
v_subgoals_205_ = v_a_285_;
v___y_206_ = v___y_178_;
v___y_207_ = v___y_179_;
v___y_208_ = v___y_180_;
v___y_209_ = v___y_181_;
goto v___jp_204_;
}
else
{
lean_object* v_a_286_; lean_object* v___x_288_; uint8_t v_isShared_289_; uint8_t v_isSharedCheck_293_; 
lean_del_object(v___x_201_);
lean_dec(v_val_199_);
lean_dec(v_snd_185_);
lean_dec_ref(v_acc_172_);
lean_dec_ref(v_matcher_170_);
v_a_286_ = lean_ctor_get(v___x_284_, 0);
v_isSharedCheck_293_ = !lean_is_exclusive(v___x_284_);
if (v_isSharedCheck_293_ == 0)
{
v___x_288_ = v___x_284_;
v_isShared_289_ = v_isSharedCheck_293_;
goto v_resetjp_287_;
}
else
{
lean_inc(v_a_286_);
lean_dec(v___x_284_);
v___x_288_ = lean_box(0);
v_isShared_289_ = v_isSharedCheck_293_;
goto v_resetjp_287_;
}
v_resetjp_287_:
{
lean_object* v___x_291_; 
if (v_isShared_289_ == 0)
{
v___x_291_ = v___x_288_;
goto v_reusejp_290_;
}
else
{
lean_object* v_reuseFailAlloc_292_; 
v_reuseFailAlloc_292_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_292_, 0, v_a_286_);
v___x_291_ = v_reuseFailAlloc_292_;
goto v_reusejp_290_;
}
v_reusejp_290_:
{
return v___x_291_;
}
}
}
}
}
}
else
{
lean_object* v_a_294_; lean_object* v___x_296_; uint8_t v_isShared_297_; uint8_t v_isSharedCheck_301_; 
lean_del_object(v___x_201_);
lean_dec(v_val_199_);
lean_del_object(v___x_187_);
lean_dec(v_snd_185_);
lean_dec(v_g_173_);
lean_dec_ref(v_acc_172_);
lean_dec_ref(v_matcher_170_);
v_a_294_ = lean_ctor_get(v___x_239_, 0);
v_isSharedCheck_301_ = !lean_is_exclusive(v___x_239_);
if (v_isSharedCheck_301_ == 0)
{
v___x_296_ = v___x_239_;
v_isShared_297_ = v_isSharedCheck_301_;
goto v_resetjp_295_;
}
else
{
lean_inc(v_a_294_);
lean_dec(v___x_239_);
v___x_296_ = lean_box(0);
v_isShared_297_ = v_isSharedCheck_301_;
goto v_resetjp_295_;
}
v_resetjp_295_:
{
lean_object* v___x_299_; 
if (v_isShared_297_ == 0)
{
v___x_299_ = v___x_296_;
goto v_reusejp_298_;
}
else
{
lean_object* v_reuseFailAlloc_300_; 
v_reuseFailAlloc_300_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_300_, 0, v_a_294_);
v___x_299_ = v_reuseFailAlloc_300_;
goto v_reusejp_298_;
}
v_reusejp_298_:
{
return v___x_299_;
}
}
}
}
else
{
lean_del_object(v___x_201_);
lean_dec(v_val_199_);
lean_dec(v_snd_185_);
v_a_191_ = v___x_236_;
goto v___jp_190_;
}
v___jp_204_:
{
size_t v_sz_210_; size_t v___x_211_; lean_object* v___x_212_; 
v_sz_210_ = lean_array_size(v_subgoals_205_);
v___x_211_ = ((size_t)0ULL);
v___x_212_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_CasesM_0__Lean_MVarId_casesMatching_go_spec__0(v_recursive_169_, v_matcher_170_, v_allowSplit_171_, v_val_199_, v_subgoals_205_, v_sz_210_, v___x_211_, v_acc_172_, v___y_206_, v___y_207_, v___y_208_, v___y_209_);
lean_dec_ref(v_subgoals_205_);
lean_dec(v_val_199_);
if (lean_obj_tag(v___x_212_) == 0)
{
lean_object* v_a_213_; lean_object* v___x_215_; uint8_t v_isShared_216_; uint8_t v_isSharedCheck_227_; 
v_a_213_ = lean_ctor_get(v___x_212_, 0);
v_isSharedCheck_227_ = !lean_is_exclusive(v___x_212_);
if (v_isSharedCheck_227_ == 0)
{
v___x_215_ = v___x_212_;
v_isShared_216_ = v_isSharedCheck_227_;
goto v_resetjp_214_;
}
else
{
lean_inc(v_a_213_);
lean_dec(v___x_212_);
v___x_215_ = lean_box(0);
v_isShared_216_ = v_isSharedCheck_227_;
goto v_resetjp_214_;
}
v_resetjp_214_:
{
lean_object* v___x_218_; 
if (v_isShared_202_ == 0)
{
lean_ctor_set(v___x_201_, 0, v_a_213_);
v___x_218_ = v___x_201_;
goto v_reusejp_217_;
}
else
{
lean_object* v_reuseFailAlloc_226_; 
v_reuseFailAlloc_226_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_226_, 0, v_a_213_);
v___x_218_ = v_reuseFailAlloc_226_;
goto v_reusejp_217_;
}
v_reusejp_217_:
{
lean_object* v___x_219_; lean_object* v___x_220_; lean_object* v___x_221_; lean_object* v___x_222_; lean_object* v___x_224_; 
v___x_219_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_219_, 0, v___x_218_);
lean_ctor_set(v___x_219_, 1, v___x_203_);
v___x_220_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_220_, 0, v___x_219_);
v___x_221_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_221_, 0, v___x_220_);
v___x_222_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_222_, 0, v___x_221_);
lean_ctor_set(v___x_222_, 1, v_snd_185_);
if (v_isShared_216_ == 0)
{
lean_ctor_set(v___x_215_, 0, v___x_222_);
v___x_224_ = v___x_215_;
goto v_reusejp_223_;
}
else
{
lean_object* v_reuseFailAlloc_225_; 
v_reuseFailAlloc_225_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_225_, 0, v___x_222_);
v___x_224_ = v_reuseFailAlloc_225_;
goto v_reusejp_223_;
}
v_reusejp_223_:
{
return v___x_224_;
}
}
}
}
else
{
lean_object* v_a_228_; lean_object* v___x_230_; uint8_t v_isShared_231_; uint8_t v_isSharedCheck_235_; 
lean_del_object(v___x_201_);
lean_dec(v_snd_185_);
v_a_228_ = lean_ctor_get(v___x_212_, 0);
v_isSharedCheck_235_ = !lean_is_exclusive(v___x_212_);
if (v_isSharedCheck_235_ == 0)
{
v___x_230_ = v___x_212_;
v_isShared_231_ = v_isSharedCheck_235_;
goto v_resetjp_229_;
}
else
{
lean_inc(v_a_228_);
lean_dec(v___x_212_);
v___x_230_ = lean_box(0);
v_isShared_231_ = v_isSharedCheck_235_;
goto v_resetjp_229_;
}
v_resetjp_229_:
{
lean_object* v___x_233_; 
if (v_isShared_231_ == 0)
{
v___x_233_ = v___x_230_;
goto v_reusejp_232_;
}
else
{
lean_object* v_reuseFailAlloc_234_; 
v_reuseFailAlloc_234_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_234_, 0, v_a_228_);
v___x_233_ = v_reuseFailAlloc_234_;
goto v_reusejp_232_;
}
v_reusejp_232_:
{
return v___x_233_;
}
}
}
}
}
}
v___jp_190_:
{
lean_object* v___x_193_; 
if (v_isShared_188_ == 0)
{
lean_ctor_set(v___x_187_, 1, v_a_191_);
lean_ctor_set(v___x_187_, 0, v___x_189_);
v___x_193_ = v___x_187_;
goto v_reusejp_192_;
}
else
{
lean_object* v_reuseFailAlloc_197_; 
v_reuseFailAlloc_197_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_197_, 0, v___x_189_);
lean_ctor_set(v_reuseFailAlloc_197_, 1, v_a_191_);
v___x_193_ = v_reuseFailAlloc_197_;
goto v_reusejp_192_;
}
v_reusejp_192_:
{
size_t v___x_194_; size_t v___x_195_; 
v___x_194_ = ((size_t)1ULL);
v___x_195_ = lean_usize_add(v_i_176_, v___x_194_);
v_i_176_ = v___x_195_;
v_b_177_ = v___x_193_;
goto _start;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00__private_Mathlib_Tactic_CasesM_0__Lean_MVarId_casesMatching_go_spec__1_spec__1_spec__4(uint8_t v_recursive_305_, lean_object* v_matcher_306_, uint8_t v_allowSplit_307_, lean_object* v_acc_308_, lean_object* v_g_309_, lean_object* v_as_310_, size_t v_sz_311_, size_t v_i_312_, lean_object* v_b_313_, lean_object* v___y_314_, lean_object* v___y_315_, lean_object* v___y_316_, lean_object* v___y_317_){
_start:
{
uint8_t v___x_319_; 
v___x_319_ = lean_usize_dec_lt(v_i_312_, v_sz_311_);
if (v___x_319_ == 0)
{
lean_object* v___x_320_; 
lean_dec(v_g_309_);
lean_dec_ref(v_acc_308_);
lean_dec_ref(v_matcher_306_);
v___x_320_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_320_, 0, v_b_313_);
return v___x_320_;
}
else
{
lean_object* v_snd_321_; lean_object* v___x_323_; uint8_t v_isShared_324_; uint8_t v_isSharedCheck_439_; 
v_snd_321_ = lean_ctor_get(v_b_313_, 1);
v_isSharedCheck_439_ = !lean_is_exclusive(v_b_313_);
if (v_isSharedCheck_439_ == 0)
{
lean_object* v_unused_440_; 
v_unused_440_ = lean_ctor_get(v_b_313_, 0);
lean_dec(v_unused_440_);
v___x_323_ = v_b_313_;
v_isShared_324_ = v_isSharedCheck_439_;
goto v_resetjp_322_;
}
else
{
lean_inc(v_snd_321_);
lean_dec(v_b_313_);
v___x_323_ = lean_box(0);
v_isShared_324_ = v_isSharedCheck_439_;
goto v_resetjp_322_;
}
v_resetjp_322_:
{
lean_object* v___x_325_; lean_object* v_a_327_; lean_object* v_a_334_; 
v___x_325_ = lean_box(0);
v_a_334_ = lean_array_uget(v_as_310_, v_i_312_);
if (lean_obj_tag(v_a_334_) == 0)
{
v_a_327_ = v_snd_321_;
goto v___jp_326_;
}
else
{
lean_object* v_val_335_; lean_object* v___x_337_; uint8_t v_isShared_338_; uint8_t v_isSharedCheck_438_; 
v_val_335_ = lean_ctor_get(v_a_334_, 0);
v_isSharedCheck_438_ = !lean_is_exclusive(v_a_334_);
if (v_isSharedCheck_438_ == 0)
{
v___x_337_ = v_a_334_;
v_isShared_338_ = v_isSharedCheck_438_;
goto v_resetjp_336_;
}
else
{
lean_inc(v_val_335_);
lean_dec(v_a_334_);
v___x_337_ = lean_box(0);
v_isShared_338_ = v_isSharedCheck_438_;
goto v_resetjp_336_;
}
v_resetjp_336_:
{
lean_object* v___x_339_; lean_object* v_subgoals_341_; lean_object* v___y_342_; lean_object* v___y_343_; lean_object* v___y_344_; lean_object* v___y_345_; lean_object* v___x_372_; uint8_t v___x_373_; 
v___x_339_ = lean_box(0);
v___x_372_ = ((lean_object*)(lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00__private_Mathlib_Tactic_CasesM_0__Lean_MVarId_casesMatching_go_spec__1_spec__1_spec__4___closed__0));
v___x_373_ = l_Lean_LocalDecl_isImplementationDetail(v_val_335_);
if (v___x_373_ == 0)
{
lean_object* v___x_374_; lean_object* v___x_375_; 
v___x_374_ = l_Lean_LocalDecl_type(v_val_335_);
lean_inc_ref(v_matcher_306_);
lean_inc(v___y_317_);
lean_inc_ref(v___y_316_);
lean_inc(v___y_315_);
lean_inc_ref(v___y_314_);
v___x_375_ = lean_apply_6(v_matcher_306_, v___x_374_, v___y_314_, v___y_315_, v___y_316_, v___y_317_, lean_box(0));
if (lean_obj_tag(v___x_375_) == 0)
{
lean_object* v_a_376_; uint8_t v___x_377_; 
v_a_376_ = lean_ctor_get(v___x_375_, 0);
lean_inc(v_a_376_);
lean_dec_ref_known(v___x_375_, 1);
v___x_377_ = lean_unbox(v_a_376_);
if (v___x_377_ == 0)
{
lean_dec(v_a_376_);
lean_del_object(v___x_337_);
lean_dec(v_val_335_);
lean_dec(v_snd_321_);
v_a_327_ = v___x_372_;
goto v___jp_326_;
}
else
{
if (v_allowSplit_307_ == 0)
{
lean_object* v___x_378_; 
v___x_378_ = l_Lean_Meta_saveState___redArg(v___y_315_, v___y_317_);
if (lean_obj_tag(v___x_378_) == 0)
{
lean_object* v_a_379_; lean_object* v___x_380_; lean_object* v___x_381_; lean_object* v___x_382_; lean_object* v___x_383_; lean_object* v___x_384_; uint8_t v___x_385_; lean_object* v___x_386_; lean_object* v___x_387_; lean_object* v___x_388_; lean_object* v___x_389_; 
v_a_379_ = lean_ctor_get(v___x_378_, 0);
lean_inc(v_a_379_);
lean_dec_ref_known(v___x_378_, 1);
v___x_380_ = l_Lean_LocalDecl_fvarId(v_val_335_);
v___x_381_ = l_Lean_LocalDecl_userName(v_val_335_);
v___x_382_ = lean_box(0);
v___x_383_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_383_, 0, v___x_381_);
lean_ctor_set(v___x_383_, 1, v___x_382_);
v___x_384_ = lean_alloc_ctor(0, 1, 1);
lean_ctor_set(v___x_384_, 0, v___x_383_);
v___x_385_ = lean_unbox(v_a_376_);
lean_dec(v_a_376_);
lean_ctor_set_uint8(v___x_384_, sizeof(void*)*1, v___x_385_);
v___x_386_ = lean_unsigned_to_nat(1u);
v___x_387_ = lean_mk_empty_array_with_capacity(v___x_386_);
v___x_388_ = lean_array_push(v___x_387_, v___x_384_);
lean_inc(v_g_309_);
v___x_389_ = l_Lean_MVarId_cases(v_g_309_, v___x_380_, v___x_388_, v_allowSplit_307_, v___x_325_, v___y_314_, v___y_315_, v___y_316_, v___y_317_);
if (lean_obj_tag(v___x_389_) == 0)
{
lean_object* v_a_390_; lean_object* v___x_391_; uint8_t v___x_392_; 
v_a_390_ = lean_ctor_get(v___x_389_, 0);
lean_inc(v_a_390_);
lean_dec_ref_known(v___x_389_, 1);
v___x_391_ = lean_array_get_size(v_a_390_);
v___x_392_ = lean_nat_dec_lt(v___x_386_, v___x_391_);
if (v___x_392_ == 0)
{
lean_dec(v_a_379_);
lean_del_object(v___x_323_);
lean_dec(v_g_309_);
v_subgoals_341_ = v_a_390_;
v___y_342_ = v___y_314_;
v___y_343_ = v___y_315_;
v___y_344_ = v___y_316_;
v___y_345_ = v___y_317_;
goto v___jp_340_;
}
else
{
lean_object* v___x_393_; 
lean_dec(v_a_390_);
lean_del_object(v___x_337_);
lean_dec(v_val_335_);
lean_dec(v_snd_321_);
v___x_393_ = l_Lean_Meta_SavedState_restore___redArg(v_a_379_, v___y_315_, v___y_317_);
lean_dec(v_a_379_);
if (lean_obj_tag(v___x_393_) == 0)
{
lean_dec_ref_known(v___x_393_, 1);
v_a_327_ = v___x_372_;
goto v___jp_326_;
}
else
{
lean_object* v_a_394_; lean_object* v___x_396_; uint8_t v_isShared_397_; uint8_t v_isSharedCheck_401_; 
lean_del_object(v___x_323_);
lean_dec(v_g_309_);
lean_dec_ref(v_acc_308_);
lean_dec_ref(v_matcher_306_);
v_a_394_ = lean_ctor_get(v___x_393_, 0);
v_isSharedCheck_401_ = !lean_is_exclusive(v___x_393_);
if (v_isSharedCheck_401_ == 0)
{
v___x_396_ = v___x_393_;
v_isShared_397_ = v_isSharedCheck_401_;
goto v_resetjp_395_;
}
else
{
lean_inc(v_a_394_);
lean_dec(v___x_393_);
v___x_396_ = lean_box(0);
v_isShared_397_ = v_isSharedCheck_401_;
goto v_resetjp_395_;
}
v_resetjp_395_:
{
lean_object* v___x_399_; 
if (v_isShared_397_ == 0)
{
v___x_399_ = v___x_396_;
goto v_reusejp_398_;
}
else
{
lean_object* v_reuseFailAlloc_400_; 
v_reuseFailAlloc_400_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_400_, 0, v_a_394_);
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
}
else
{
lean_object* v_a_402_; lean_object* v___x_404_; uint8_t v_isShared_405_; uint8_t v_isSharedCheck_409_; 
lean_dec(v_a_379_);
lean_del_object(v___x_337_);
lean_dec(v_val_335_);
lean_del_object(v___x_323_);
lean_dec(v_snd_321_);
lean_dec(v_g_309_);
lean_dec_ref(v_acc_308_);
lean_dec_ref(v_matcher_306_);
v_a_402_ = lean_ctor_get(v___x_389_, 0);
v_isSharedCheck_409_ = !lean_is_exclusive(v___x_389_);
if (v_isSharedCheck_409_ == 0)
{
v___x_404_ = v___x_389_;
v_isShared_405_ = v_isSharedCheck_409_;
goto v_resetjp_403_;
}
else
{
lean_inc(v_a_402_);
lean_dec(v___x_389_);
v___x_404_ = lean_box(0);
v_isShared_405_ = v_isSharedCheck_409_;
goto v_resetjp_403_;
}
v_resetjp_403_:
{
lean_object* v___x_407_; 
if (v_isShared_405_ == 0)
{
v___x_407_ = v___x_404_;
goto v_reusejp_406_;
}
else
{
lean_object* v_reuseFailAlloc_408_; 
v_reuseFailAlloc_408_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_408_, 0, v_a_402_);
v___x_407_ = v_reuseFailAlloc_408_;
goto v_reusejp_406_;
}
v_reusejp_406_:
{
return v___x_407_;
}
}
}
}
else
{
lean_object* v_a_410_; lean_object* v___x_412_; uint8_t v_isShared_413_; uint8_t v_isSharedCheck_417_; 
lean_dec(v_a_376_);
lean_del_object(v___x_337_);
lean_dec(v_val_335_);
lean_del_object(v___x_323_);
lean_dec(v_snd_321_);
lean_dec(v_g_309_);
lean_dec_ref(v_acc_308_);
lean_dec_ref(v_matcher_306_);
v_a_410_ = lean_ctor_get(v___x_378_, 0);
v_isSharedCheck_417_ = !lean_is_exclusive(v___x_378_);
if (v_isSharedCheck_417_ == 0)
{
v___x_412_ = v___x_378_;
v_isShared_413_ = v_isSharedCheck_417_;
goto v_resetjp_411_;
}
else
{
lean_inc(v_a_410_);
lean_dec(v___x_378_);
v___x_412_ = lean_box(0);
v_isShared_413_ = v_isSharedCheck_417_;
goto v_resetjp_411_;
}
v_resetjp_411_:
{
lean_object* v___x_415_; 
if (v_isShared_413_ == 0)
{
v___x_415_ = v___x_412_;
goto v_reusejp_414_;
}
else
{
lean_object* v_reuseFailAlloc_416_; 
v_reuseFailAlloc_416_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_416_, 0, v_a_410_);
v___x_415_ = v_reuseFailAlloc_416_;
goto v_reusejp_414_;
}
v_reusejp_414_:
{
return v___x_415_;
}
}
}
}
else
{
lean_object* v___x_418_; lean_object* v___x_419_; lean_object* v___x_420_; 
lean_dec(v_a_376_);
lean_del_object(v___x_323_);
v___x_418_ = l_Lean_LocalDecl_fvarId(v_val_335_);
v___x_419_ = ((lean_object*)(lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00__private_Mathlib_Tactic_CasesM_0__Lean_MVarId_casesMatching_go_spec__1_spec__2___closed__1));
v___x_420_ = l_Lean_MVarId_cases(v_g_309_, v___x_418_, v___x_419_, v___x_373_, v___x_325_, v___y_314_, v___y_315_, v___y_316_, v___y_317_);
if (lean_obj_tag(v___x_420_) == 0)
{
lean_object* v_a_421_; 
v_a_421_ = lean_ctor_get(v___x_420_, 0);
lean_inc(v_a_421_);
lean_dec_ref_known(v___x_420_, 1);
v_subgoals_341_ = v_a_421_;
v___y_342_ = v___y_314_;
v___y_343_ = v___y_315_;
v___y_344_ = v___y_316_;
v___y_345_ = v___y_317_;
goto v___jp_340_;
}
else
{
lean_object* v_a_422_; lean_object* v___x_424_; uint8_t v_isShared_425_; uint8_t v_isSharedCheck_429_; 
lean_del_object(v___x_337_);
lean_dec(v_val_335_);
lean_dec(v_snd_321_);
lean_dec_ref(v_acc_308_);
lean_dec_ref(v_matcher_306_);
v_a_422_ = lean_ctor_get(v___x_420_, 0);
v_isSharedCheck_429_ = !lean_is_exclusive(v___x_420_);
if (v_isSharedCheck_429_ == 0)
{
v___x_424_ = v___x_420_;
v_isShared_425_ = v_isSharedCheck_429_;
goto v_resetjp_423_;
}
else
{
lean_inc(v_a_422_);
lean_dec(v___x_420_);
v___x_424_ = lean_box(0);
v_isShared_425_ = v_isSharedCheck_429_;
goto v_resetjp_423_;
}
v_resetjp_423_:
{
lean_object* v___x_427_; 
if (v_isShared_425_ == 0)
{
v___x_427_ = v___x_424_;
goto v_reusejp_426_;
}
else
{
lean_object* v_reuseFailAlloc_428_; 
v_reuseFailAlloc_428_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_428_, 0, v_a_422_);
v___x_427_ = v_reuseFailAlloc_428_;
goto v_reusejp_426_;
}
v_reusejp_426_:
{
return v___x_427_;
}
}
}
}
}
}
else
{
lean_object* v_a_430_; lean_object* v___x_432_; uint8_t v_isShared_433_; uint8_t v_isSharedCheck_437_; 
lean_del_object(v___x_337_);
lean_dec(v_val_335_);
lean_del_object(v___x_323_);
lean_dec(v_snd_321_);
lean_dec(v_g_309_);
lean_dec_ref(v_acc_308_);
lean_dec_ref(v_matcher_306_);
v_a_430_ = lean_ctor_get(v___x_375_, 0);
v_isSharedCheck_437_ = !lean_is_exclusive(v___x_375_);
if (v_isSharedCheck_437_ == 0)
{
v___x_432_ = v___x_375_;
v_isShared_433_ = v_isSharedCheck_437_;
goto v_resetjp_431_;
}
else
{
lean_inc(v_a_430_);
lean_dec(v___x_375_);
v___x_432_ = lean_box(0);
v_isShared_433_ = v_isSharedCheck_437_;
goto v_resetjp_431_;
}
v_resetjp_431_:
{
lean_object* v___x_435_; 
if (v_isShared_433_ == 0)
{
v___x_435_ = v___x_432_;
goto v_reusejp_434_;
}
else
{
lean_object* v_reuseFailAlloc_436_; 
v_reuseFailAlloc_436_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_436_, 0, v_a_430_);
v___x_435_ = v_reuseFailAlloc_436_;
goto v_reusejp_434_;
}
v_reusejp_434_:
{
return v___x_435_;
}
}
}
}
else
{
lean_del_object(v___x_337_);
lean_dec(v_val_335_);
lean_dec(v_snd_321_);
v_a_327_ = v___x_372_;
goto v___jp_326_;
}
v___jp_340_:
{
size_t v_sz_346_; size_t v___x_347_; lean_object* v___x_348_; 
v_sz_346_ = lean_array_size(v_subgoals_341_);
v___x_347_ = ((size_t)0ULL);
v___x_348_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_CasesM_0__Lean_MVarId_casesMatching_go_spec__0(v_recursive_305_, v_matcher_306_, v_allowSplit_307_, v_val_335_, v_subgoals_341_, v_sz_346_, v___x_347_, v_acc_308_, v___y_342_, v___y_343_, v___y_344_, v___y_345_);
lean_dec_ref(v_subgoals_341_);
lean_dec(v_val_335_);
if (lean_obj_tag(v___x_348_) == 0)
{
lean_object* v_a_349_; lean_object* v___x_351_; uint8_t v_isShared_352_; uint8_t v_isSharedCheck_363_; 
v_a_349_ = lean_ctor_get(v___x_348_, 0);
v_isSharedCheck_363_ = !lean_is_exclusive(v___x_348_);
if (v_isSharedCheck_363_ == 0)
{
v___x_351_ = v___x_348_;
v_isShared_352_ = v_isSharedCheck_363_;
goto v_resetjp_350_;
}
else
{
lean_inc(v_a_349_);
lean_dec(v___x_348_);
v___x_351_ = lean_box(0);
v_isShared_352_ = v_isSharedCheck_363_;
goto v_resetjp_350_;
}
v_resetjp_350_:
{
lean_object* v___x_354_; 
if (v_isShared_338_ == 0)
{
lean_ctor_set(v___x_337_, 0, v_a_349_);
v___x_354_ = v___x_337_;
goto v_reusejp_353_;
}
else
{
lean_object* v_reuseFailAlloc_362_; 
v_reuseFailAlloc_362_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_362_, 0, v_a_349_);
v___x_354_ = v_reuseFailAlloc_362_;
goto v_reusejp_353_;
}
v_reusejp_353_:
{
lean_object* v___x_355_; lean_object* v___x_356_; lean_object* v___x_357_; lean_object* v___x_358_; lean_object* v___x_360_; 
v___x_355_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_355_, 0, v___x_354_);
lean_ctor_set(v___x_355_, 1, v___x_339_);
v___x_356_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_356_, 0, v___x_355_);
v___x_357_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_357_, 0, v___x_356_);
v___x_358_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_358_, 0, v___x_357_);
lean_ctor_set(v___x_358_, 1, v_snd_321_);
if (v_isShared_352_ == 0)
{
lean_ctor_set(v___x_351_, 0, v___x_358_);
v___x_360_ = v___x_351_;
goto v_reusejp_359_;
}
else
{
lean_object* v_reuseFailAlloc_361_; 
v_reuseFailAlloc_361_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_361_, 0, v___x_358_);
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
else
{
lean_object* v_a_364_; lean_object* v___x_366_; uint8_t v_isShared_367_; uint8_t v_isSharedCheck_371_; 
lean_del_object(v___x_337_);
lean_dec(v_snd_321_);
v_a_364_ = lean_ctor_get(v___x_348_, 0);
v_isSharedCheck_371_ = !lean_is_exclusive(v___x_348_);
if (v_isSharedCheck_371_ == 0)
{
v___x_366_ = v___x_348_;
v_isShared_367_ = v_isSharedCheck_371_;
goto v_resetjp_365_;
}
else
{
lean_inc(v_a_364_);
lean_dec(v___x_348_);
v___x_366_ = lean_box(0);
v_isShared_367_ = v_isSharedCheck_371_;
goto v_resetjp_365_;
}
v_resetjp_365_:
{
lean_object* v___x_369_; 
if (v_isShared_367_ == 0)
{
v___x_369_ = v___x_366_;
goto v_reusejp_368_;
}
else
{
lean_object* v_reuseFailAlloc_370_; 
v_reuseFailAlloc_370_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_370_, 0, v_a_364_);
v___x_369_ = v_reuseFailAlloc_370_;
goto v_reusejp_368_;
}
v_reusejp_368_:
{
return v___x_369_;
}
}
}
}
}
}
v___jp_326_:
{
lean_object* v___x_329_; 
if (v_isShared_324_ == 0)
{
lean_ctor_set(v___x_323_, 1, v_a_327_);
lean_ctor_set(v___x_323_, 0, v___x_325_);
v___x_329_ = v___x_323_;
goto v_reusejp_328_;
}
else
{
lean_object* v_reuseFailAlloc_333_; 
v_reuseFailAlloc_333_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_333_, 0, v___x_325_);
lean_ctor_set(v_reuseFailAlloc_333_, 1, v_a_327_);
v___x_329_ = v_reuseFailAlloc_333_;
goto v_reusejp_328_;
}
v_reusejp_328_:
{
size_t v___x_330_; size_t v___x_331_; lean_object* v___x_332_; 
v___x_330_ = ((size_t)1ULL);
v___x_331_ = lean_usize_add(v_i_312_, v___x_330_);
v___x_332_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00__private_Mathlib_Tactic_CasesM_0__Lean_MVarId_casesMatching_go_spec__1_spec__1_spec__4_spec__5(v_recursive_305_, v_matcher_306_, v_allowSplit_307_, v_acc_308_, v_g_309_, v_as_310_, v_sz_311_, v___x_331_, v___x_329_, v___y_314_, v___y_315_, v___y_316_, v___y_317_);
return v___x_332_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00__private_Mathlib_Tactic_CasesM_0__Lean_MVarId_casesMatching_go_spec__1_spec__1(lean_object* v_init_441_, uint8_t v_recursive_442_, lean_object* v_matcher_443_, uint8_t v_allowSplit_444_, lean_object* v_acc_445_, lean_object* v_g_446_, lean_object* v_n_447_, lean_object* v_b_448_, lean_object* v___y_449_, lean_object* v___y_450_, lean_object* v___y_451_, lean_object* v___y_452_){
_start:
{
if (lean_obj_tag(v_n_447_) == 0)
{
lean_object* v_cs_454_; lean_object* v___x_455_; lean_object* v___x_456_; size_t v_sz_457_; size_t v___x_458_; lean_object* v___x_459_; 
v_cs_454_ = lean_ctor_get(v_n_447_, 0);
v___x_455_ = lean_box(0);
v___x_456_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_456_, 0, v___x_455_);
lean_ctor_set(v___x_456_, 1, v_b_448_);
v_sz_457_ = lean_array_size(v_cs_454_);
v___x_458_ = ((size_t)0ULL);
v___x_459_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00__private_Mathlib_Tactic_CasesM_0__Lean_MVarId_casesMatching_go_spec__1_spec__1_spec__3(v_init_441_, v_recursive_442_, v_matcher_443_, v_allowSplit_444_, v_acc_445_, v_g_446_, v_cs_454_, v_sz_457_, v___x_458_, v___x_456_, v___y_449_, v___y_450_, v___y_451_, v___y_452_);
if (lean_obj_tag(v___x_459_) == 0)
{
lean_object* v_a_460_; lean_object* v___x_462_; uint8_t v_isShared_463_; uint8_t v_isSharedCheck_474_; 
v_a_460_ = lean_ctor_get(v___x_459_, 0);
v_isSharedCheck_474_ = !lean_is_exclusive(v___x_459_);
if (v_isSharedCheck_474_ == 0)
{
v___x_462_ = v___x_459_;
v_isShared_463_ = v_isSharedCheck_474_;
goto v_resetjp_461_;
}
else
{
lean_inc(v_a_460_);
lean_dec(v___x_459_);
v___x_462_ = lean_box(0);
v_isShared_463_ = v_isSharedCheck_474_;
goto v_resetjp_461_;
}
v_resetjp_461_:
{
lean_object* v_fst_464_; 
v_fst_464_ = lean_ctor_get(v_a_460_, 0);
if (lean_obj_tag(v_fst_464_) == 0)
{
lean_object* v_snd_465_; lean_object* v___x_466_; lean_object* v___x_468_; 
v_snd_465_ = lean_ctor_get(v_a_460_, 1);
lean_inc(v_snd_465_);
lean_dec(v_a_460_);
v___x_466_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_466_, 0, v_snd_465_);
if (v_isShared_463_ == 0)
{
lean_ctor_set(v___x_462_, 0, v___x_466_);
v___x_468_ = v___x_462_;
goto v_reusejp_467_;
}
else
{
lean_object* v_reuseFailAlloc_469_; 
v_reuseFailAlloc_469_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_469_, 0, v___x_466_);
v___x_468_ = v_reuseFailAlloc_469_;
goto v_reusejp_467_;
}
v_reusejp_467_:
{
return v___x_468_;
}
}
else
{
lean_object* v_val_470_; lean_object* v___x_472_; 
lean_inc_ref(v_fst_464_);
lean_dec(v_a_460_);
v_val_470_ = lean_ctor_get(v_fst_464_, 0);
lean_inc(v_val_470_);
lean_dec_ref_known(v_fst_464_, 1);
if (v_isShared_463_ == 0)
{
lean_ctor_set(v___x_462_, 0, v_val_470_);
v___x_472_ = v___x_462_;
goto v_reusejp_471_;
}
else
{
lean_object* v_reuseFailAlloc_473_; 
v_reuseFailAlloc_473_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_473_, 0, v_val_470_);
v___x_472_ = v_reuseFailAlloc_473_;
goto v_reusejp_471_;
}
v_reusejp_471_:
{
return v___x_472_;
}
}
}
}
else
{
lean_object* v_a_475_; lean_object* v___x_477_; uint8_t v_isShared_478_; uint8_t v_isSharedCheck_482_; 
v_a_475_ = lean_ctor_get(v___x_459_, 0);
v_isSharedCheck_482_ = !lean_is_exclusive(v___x_459_);
if (v_isSharedCheck_482_ == 0)
{
v___x_477_ = v___x_459_;
v_isShared_478_ = v_isSharedCheck_482_;
goto v_resetjp_476_;
}
else
{
lean_inc(v_a_475_);
lean_dec(v___x_459_);
v___x_477_ = lean_box(0);
v_isShared_478_ = v_isSharedCheck_482_;
goto v_resetjp_476_;
}
v_resetjp_476_:
{
lean_object* v___x_480_; 
if (v_isShared_478_ == 0)
{
v___x_480_ = v___x_477_;
goto v_reusejp_479_;
}
else
{
lean_object* v_reuseFailAlloc_481_; 
v_reuseFailAlloc_481_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_481_, 0, v_a_475_);
v___x_480_ = v_reuseFailAlloc_481_;
goto v_reusejp_479_;
}
v_reusejp_479_:
{
return v___x_480_;
}
}
}
}
else
{
lean_object* v_vs_483_; lean_object* v___x_484_; lean_object* v___x_485_; size_t v_sz_486_; size_t v___x_487_; lean_object* v___x_488_; 
v_vs_483_ = lean_ctor_get(v_n_447_, 0);
v___x_484_ = lean_box(0);
v___x_485_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_485_, 0, v___x_484_);
lean_ctor_set(v___x_485_, 1, v_b_448_);
v_sz_486_ = lean_array_size(v_vs_483_);
v___x_487_ = ((size_t)0ULL);
v___x_488_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00__private_Mathlib_Tactic_CasesM_0__Lean_MVarId_casesMatching_go_spec__1_spec__1_spec__4(v_recursive_442_, v_matcher_443_, v_allowSplit_444_, v_acc_445_, v_g_446_, v_vs_483_, v_sz_486_, v___x_487_, v___x_485_, v___y_449_, v___y_450_, v___y_451_, v___y_452_);
if (lean_obj_tag(v___x_488_) == 0)
{
lean_object* v_a_489_; lean_object* v___x_491_; uint8_t v_isShared_492_; uint8_t v_isSharedCheck_503_; 
v_a_489_ = lean_ctor_get(v___x_488_, 0);
v_isSharedCheck_503_ = !lean_is_exclusive(v___x_488_);
if (v_isSharedCheck_503_ == 0)
{
v___x_491_ = v___x_488_;
v_isShared_492_ = v_isSharedCheck_503_;
goto v_resetjp_490_;
}
else
{
lean_inc(v_a_489_);
lean_dec(v___x_488_);
v___x_491_ = lean_box(0);
v_isShared_492_ = v_isSharedCheck_503_;
goto v_resetjp_490_;
}
v_resetjp_490_:
{
lean_object* v_fst_493_; 
v_fst_493_ = lean_ctor_get(v_a_489_, 0);
if (lean_obj_tag(v_fst_493_) == 0)
{
lean_object* v_snd_494_; lean_object* v___x_495_; lean_object* v___x_497_; 
v_snd_494_ = lean_ctor_get(v_a_489_, 1);
lean_inc(v_snd_494_);
lean_dec(v_a_489_);
v___x_495_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_495_, 0, v_snd_494_);
if (v_isShared_492_ == 0)
{
lean_ctor_set(v___x_491_, 0, v___x_495_);
v___x_497_ = v___x_491_;
goto v_reusejp_496_;
}
else
{
lean_object* v_reuseFailAlloc_498_; 
v_reuseFailAlloc_498_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_498_, 0, v___x_495_);
v___x_497_ = v_reuseFailAlloc_498_;
goto v_reusejp_496_;
}
v_reusejp_496_:
{
return v___x_497_;
}
}
else
{
lean_object* v_val_499_; lean_object* v___x_501_; 
lean_inc_ref(v_fst_493_);
lean_dec(v_a_489_);
v_val_499_ = lean_ctor_get(v_fst_493_, 0);
lean_inc(v_val_499_);
lean_dec_ref_known(v_fst_493_, 1);
if (v_isShared_492_ == 0)
{
lean_ctor_set(v___x_491_, 0, v_val_499_);
v___x_501_ = v___x_491_;
goto v_reusejp_500_;
}
else
{
lean_object* v_reuseFailAlloc_502_; 
v_reuseFailAlloc_502_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_502_, 0, v_val_499_);
v___x_501_ = v_reuseFailAlloc_502_;
goto v_reusejp_500_;
}
v_reusejp_500_:
{
return v___x_501_;
}
}
}
}
else
{
lean_object* v_a_504_; lean_object* v___x_506_; uint8_t v_isShared_507_; uint8_t v_isSharedCheck_511_; 
v_a_504_ = lean_ctor_get(v___x_488_, 0);
v_isSharedCheck_511_ = !lean_is_exclusive(v___x_488_);
if (v_isSharedCheck_511_ == 0)
{
v___x_506_ = v___x_488_;
v_isShared_507_ = v_isSharedCheck_511_;
goto v_resetjp_505_;
}
else
{
lean_inc(v_a_504_);
lean_dec(v___x_488_);
v___x_506_ = lean_box(0);
v_isShared_507_ = v_isSharedCheck_511_;
goto v_resetjp_505_;
}
v_resetjp_505_:
{
lean_object* v___x_509_; 
if (v_isShared_507_ == 0)
{
v___x_509_ = v___x_506_;
goto v_reusejp_508_;
}
else
{
lean_object* v_reuseFailAlloc_510_; 
v_reuseFailAlloc_510_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_510_, 0, v_a_504_);
v___x_509_ = v_reuseFailAlloc_510_;
goto v_reusejp_508_;
}
v_reusejp_508_:
{
return v___x_509_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00__private_Mathlib_Tactic_CasesM_0__Lean_MVarId_casesMatching_go_spec__1_spec__2_spec__6(uint8_t v_recursive_515_, lean_object* v_matcher_516_, uint8_t v_allowSplit_517_, lean_object* v_acc_518_, lean_object* v_g_519_, lean_object* v_as_520_, size_t v_sz_521_, size_t v_i_522_, lean_object* v_b_523_, lean_object* v___y_524_, lean_object* v___y_525_, lean_object* v___y_526_, lean_object* v___y_527_){
_start:
{
uint8_t v___x_529_; 
v___x_529_ = lean_usize_dec_lt(v_i_522_, v_sz_521_);
if (v___x_529_ == 0)
{
lean_object* v___x_530_; 
lean_dec(v_g_519_);
lean_dec_ref(v_acc_518_);
lean_dec_ref(v_matcher_516_);
v___x_530_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_530_, 0, v_b_523_);
return v___x_530_;
}
else
{
lean_object* v_snd_531_; lean_object* v___x_533_; uint8_t v_isShared_534_; uint8_t v_isSharedCheck_648_; 
v_snd_531_ = lean_ctor_get(v_b_523_, 1);
v_isSharedCheck_648_ = !lean_is_exclusive(v_b_523_);
if (v_isSharedCheck_648_ == 0)
{
lean_object* v_unused_649_; 
v_unused_649_ = lean_ctor_get(v_b_523_, 0);
lean_dec(v_unused_649_);
v___x_533_ = v_b_523_;
v_isShared_534_ = v_isSharedCheck_648_;
goto v_resetjp_532_;
}
else
{
lean_inc(v_snd_531_);
lean_dec(v_b_523_);
v___x_533_ = lean_box(0);
v_isShared_534_ = v_isSharedCheck_648_;
goto v_resetjp_532_;
}
v_resetjp_532_:
{
lean_object* v___x_535_; lean_object* v_a_537_; lean_object* v_a_544_; 
v___x_535_ = lean_box(0);
v_a_544_ = lean_array_uget(v_as_520_, v_i_522_);
if (lean_obj_tag(v_a_544_) == 0)
{
v_a_537_ = v_snd_531_;
goto v___jp_536_;
}
else
{
lean_object* v_val_545_; lean_object* v___x_547_; uint8_t v_isShared_548_; uint8_t v_isSharedCheck_647_; 
v_val_545_ = lean_ctor_get(v_a_544_, 0);
v_isSharedCheck_647_ = !lean_is_exclusive(v_a_544_);
if (v_isSharedCheck_647_ == 0)
{
v___x_547_ = v_a_544_;
v_isShared_548_ = v_isSharedCheck_647_;
goto v_resetjp_546_;
}
else
{
lean_inc(v_val_545_);
lean_dec(v_a_544_);
v___x_547_ = lean_box(0);
v_isShared_548_ = v_isSharedCheck_647_;
goto v_resetjp_546_;
}
v_resetjp_546_:
{
lean_object* v___x_549_; lean_object* v_subgoals_551_; lean_object* v___y_552_; lean_object* v___y_553_; lean_object* v___y_554_; lean_object* v___y_555_; lean_object* v___x_581_; uint8_t v___x_582_; 
v___x_549_ = lean_box(0);
v___x_581_ = ((lean_object*)(lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00__private_Mathlib_Tactic_CasesM_0__Lean_MVarId_casesMatching_go_spec__1_spec__2___closed__0));
v___x_582_ = l_Lean_LocalDecl_isImplementationDetail(v_val_545_);
if (v___x_582_ == 0)
{
lean_object* v___x_583_; lean_object* v___x_584_; 
v___x_583_ = l_Lean_LocalDecl_type(v_val_545_);
lean_inc_ref(v_matcher_516_);
lean_inc(v___y_527_);
lean_inc_ref(v___y_526_);
lean_inc(v___y_525_);
lean_inc_ref(v___y_524_);
v___x_584_ = lean_apply_6(v_matcher_516_, v___x_583_, v___y_524_, v___y_525_, v___y_526_, v___y_527_, lean_box(0));
if (lean_obj_tag(v___x_584_) == 0)
{
lean_object* v_a_585_; uint8_t v___x_586_; 
v_a_585_ = lean_ctor_get(v___x_584_, 0);
lean_inc(v_a_585_);
lean_dec_ref_known(v___x_584_, 1);
v___x_586_ = lean_unbox(v_a_585_);
if (v___x_586_ == 0)
{
lean_dec(v_a_585_);
lean_del_object(v___x_547_);
lean_dec(v_val_545_);
lean_dec(v_snd_531_);
v_a_537_ = v___x_581_;
goto v___jp_536_;
}
else
{
if (v_allowSplit_517_ == 0)
{
lean_object* v___x_587_; 
v___x_587_ = l_Lean_Meta_saveState___redArg(v___y_525_, v___y_527_);
if (lean_obj_tag(v___x_587_) == 0)
{
lean_object* v_a_588_; lean_object* v___x_589_; lean_object* v___x_590_; lean_object* v___x_591_; lean_object* v___x_592_; lean_object* v___x_593_; uint8_t v___x_594_; lean_object* v___x_595_; lean_object* v___x_596_; lean_object* v___x_597_; lean_object* v___x_598_; 
v_a_588_ = lean_ctor_get(v___x_587_, 0);
lean_inc(v_a_588_);
lean_dec_ref_known(v___x_587_, 1);
v___x_589_ = l_Lean_LocalDecl_fvarId(v_val_545_);
v___x_590_ = l_Lean_LocalDecl_userName(v_val_545_);
v___x_591_ = lean_box(0);
v___x_592_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_592_, 0, v___x_590_);
lean_ctor_set(v___x_592_, 1, v___x_591_);
v___x_593_ = lean_alloc_ctor(0, 1, 1);
lean_ctor_set(v___x_593_, 0, v___x_592_);
v___x_594_ = lean_unbox(v_a_585_);
lean_dec(v_a_585_);
lean_ctor_set_uint8(v___x_593_, sizeof(void*)*1, v___x_594_);
v___x_595_ = lean_unsigned_to_nat(1u);
v___x_596_ = lean_mk_empty_array_with_capacity(v___x_595_);
v___x_597_ = lean_array_push(v___x_596_, v___x_593_);
lean_inc(v_g_519_);
v___x_598_ = l_Lean_MVarId_cases(v_g_519_, v___x_589_, v___x_597_, v_allowSplit_517_, v___x_535_, v___y_524_, v___y_525_, v___y_526_, v___y_527_);
if (lean_obj_tag(v___x_598_) == 0)
{
lean_object* v_a_599_; lean_object* v___x_600_; uint8_t v___x_601_; 
v_a_599_ = lean_ctor_get(v___x_598_, 0);
lean_inc(v_a_599_);
lean_dec_ref_known(v___x_598_, 1);
v___x_600_ = lean_array_get_size(v_a_599_);
v___x_601_ = lean_nat_dec_lt(v___x_595_, v___x_600_);
if (v___x_601_ == 0)
{
lean_dec(v_a_588_);
lean_del_object(v___x_533_);
lean_dec(v_g_519_);
v_subgoals_551_ = v_a_599_;
v___y_552_ = v___y_524_;
v___y_553_ = v___y_525_;
v___y_554_ = v___y_526_;
v___y_555_ = v___y_527_;
goto v___jp_550_;
}
else
{
lean_object* v___x_602_; 
lean_dec(v_a_599_);
lean_del_object(v___x_547_);
lean_dec(v_val_545_);
lean_dec(v_snd_531_);
v___x_602_ = l_Lean_Meta_SavedState_restore___redArg(v_a_588_, v___y_525_, v___y_527_);
lean_dec(v_a_588_);
if (lean_obj_tag(v___x_602_) == 0)
{
lean_dec_ref_known(v___x_602_, 1);
v_a_537_ = v___x_581_;
goto v___jp_536_;
}
else
{
lean_object* v_a_603_; lean_object* v___x_605_; uint8_t v_isShared_606_; uint8_t v_isSharedCheck_610_; 
lean_del_object(v___x_533_);
lean_dec(v_g_519_);
lean_dec_ref(v_acc_518_);
lean_dec_ref(v_matcher_516_);
v_a_603_ = lean_ctor_get(v___x_602_, 0);
v_isSharedCheck_610_ = !lean_is_exclusive(v___x_602_);
if (v_isSharedCheck_610_ == 0)
{
v___x_605_ = v___x_602_;
v_isShared_606_ = v_isSharedCheck_610_;
goto v_resetjp_604_;
}
else
{
lean_inc(v_a_603_);
lean_dec(v___x_602_);
v___x_605_ = lean_box(0);
v_isShared_606_ = v_isSharedCheck_610_;
goto v_resetjp_604_;
}
v_resetjp_604_:
{
lean_object* v___x_608_; 
if (v_isShared_606_ == 0)
{
v___x_608_ = v___x_605_;
goto v_reusejp_607_;
}
else
{
lean_object* v_reuseFailAlloc_609_; 
v_reuseFailAlloc_609_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_609_, 0, v_a_603_);
v___x_608_ = v_reuseFailAlloc_609_;
goto v_reusejp_607_;
}
v_reusejp_607_:
{
return v___x_608_;
}
}
}
}
}
else
{
lean_object* v_a_611_; lean_object* v___x_613_; uint8_t v_isShared_614_; uint8_t v_isSharedCheck_618_; 
lean_dec(v_a_588_);
lean_del_object(v___x_547_);
lean_dec(v_val_545_);
lean_del_object(v___x_533_);
lean_dec(v_snd_531_);
lean_dec(v_g_519_);
lean_dec_ref(v_acc_518_);
lean_dec_ref(v_matcher_516_);
v_a_611_ = lean_ctor_get(v___x_598_, 0);
v_isSharedCheck_618_ = !lean_is_exclusive(v___x_598_);
if (v_isSharedCheck_618_ == 0)
{
v___x_613_ = v___x_598_;
v_isShared_614_ = v_isSharedCheck_618_;
goto v_resetjp_612_;
}
else
{
lean_inc(v_a_611_);
lean_dec(v___x_598_);
v___x_613_ = lean_box(0);
v_isShared_614_ = v_isSharedCheck_618_;
goto v_resetjp_612_;
}
v_resetjp_612_:
{
lean_object* v___x_616_; 
if (v_isShared_614_ == 0)
{
v___x_616_ = v___x_613_;
goto v_reusejp_615_;
}
else
{
lean_object* v_reuseFailAlloc_617_; 
v_reuseFailAlloc_617_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_617_, 0, v_a_611_);
v___x_616_ = v_reuseFailAlloc_617_;
goto v_reusejp_615_;
}
v_reusejp_615_:
{
return v___x_616_;
}
}
}
}
else
{
lean_object* v_a_619_; lean_object* v___x_621_; uint8_t v_isShared_622_; uint8_t v_isSharedCheck_626_; 
lean_dec(v_a_585_);
lean_del_object(v___x_547_);
lean_dec(v_val_545_);
lean_del_object(v___x_533_);
lean_dec(v_snd_531_);
lean_dec(v_g_519_);
lean_dec_ref(v_acc_518_);
lean_dec_ref(v_matcher_516_);
v_a_619_ = lean_ctor_get(v___x_587_, 0);
v_isSharedCheck_626_ = !lean_is_exclusive(v___x_587_);
if (v_isSharedCheck_626_ == 0)
{
v___x_621_ = v___x_587_;
v_isShared_622_ = v_isSharedCheck_626_;
goto v_resetjp_620_;
}
else
{
lean_inc(v_a_619_);
lean_dec(v___x_587_);
v___x_621_ = lean_box(0);
v_isShared_622_ = v_isSharedCheck_626_;
goto v_resetjp_620_;
}
v_resetjp_620_:
{
lean_object* v___x_624_; 
if (v_isShared_622_ == 0)
{
v___x_624_ = v___x_621_;
goto v_reusejp_623_;
}
else
{
lean_object* v_reuseFailAlloc_625_; 
v_reuseFailAlloc_625_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_625_, 0, v_a_619_);
v___x_624_ = v_reuseFailAlloc_625_;
goto v_reusejp_623_;
}
v_reusejp_623_:
{
return v___x_624_;
}
}
}
}
else
{
lean_object* v___x_627_; lean_object* v___x_628_; lean_object* v___x_629_; 
lean_dec(v_a_585_);
lean_del_object(v___x_533_);
v___x_627_ = l_Lean_LocalDecl_fvarId(v_val_545_);
v___x_628_ = ((lean_object*)(lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00__private_Mathlib_Tactic_CasesM_0__Lean_MVarId_casesMatching_go_spec__1_spec__2___closed__1));
v___x_629_ = l_Lean_MVarId_cases(v_g_519_, v___x_627_, v___x_628_, v___x_582_, v___x_535_, v___y_524_, v___y_525_, v___y_526_, v___y_527_);
if (lean_obj_tag(v___x_629_) == 0)
{
lean_object* v_a_630_; 
v_a_630_ = lean_ctor_get(v___x_629_, 0);
lean_inc(v_a_630_);
lean_dec_ref_known(v___x_629_, 1);
v_subgoals_551_ = v_a_630_;
v___y_552_ = v___y_524_;
v___y_553_ = v___y_525_;
v___y_554_ = v___y_526_;
v___y_555_ = v___y_527_;
goto v___jp_550_;
}
else
{
lean_object* v_a_631_; lean_object* v___x_633_; uint8_t v_isShared_634_; uint8_t v_isSharedCheck_638_; 
lean_del_object(v___x_547_);
lean_dec(v_val_545_);
lean_dec(v_snd_531_);
lean_dec_ref(v_acc_518_);
lean_dec_ref(v_matcher_516_);
v_a_631_ = lean_ctor_get(v___x_629_, 0);
v_isSharedCheck_638_ = !lean_is_exclusive(v___x_629_);
if (v_isSharedCheck_638_ == 0)
{
v___x_633_ = v___x_629_;
v_isShared_634_ = v_isSharedCheck_638_;
goto v_resetjp_632_;
}
else
{
lean_inc(v_a_631_);
lean_dec(v___x_629_);
v___x_633_ = lean_box(0);
v_isShared_634_ = v_isSharedCheck_638_;
goto v_resetjp_632_;
}
v_resetjp_632_:
{
lean_object* v___x_636_; 
if (v_isShared_634_ == 0)
{
v___x_636_ = v___x_633_;
goto v_reusejp_635_;
}
else
{
lean_object* v_reuseFailAlloc_637_; 
v_reuseFailAlloc_637_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_637_, 0, v_a_631_);
v___x_636_ = v_reuseFailAlloc_637_;
goto v_reusejp_635_;
}
v_reusejp_635_:
{
return v___x_636_;
}
}
}
}
}
}
else
{
lean_object* v_a_639_; lean_object* v___x_641_; uint8_t v_isShared_642_; uint8_t v_isSharedCheck_646_; 
lean_del_object(v___x_547_);
lean_dec(v_val_545_);
lean_del_object(v___x_533_);
lean_dec(v_snd_531_);
lean_dec(v_g_519_);
lean_dec_ref(v_acc_518_);
lean_dec_ref(v_matcher_516_);
v_a_639_ = lean_ctor_get(v___x_584_, 0);
v_isSharedCheck_646_ = !lean_is_exclusive(v___x_584_);
if (v_isSharedCheck_646_ == 0)
{
v___x_641_ = v___x_584_;
v_isShared_642_ = v_isSharedCheck_646_;
goto v_resetjp_640_;
}
else
{
lean_inc(v_a_639_);
lean_dec(v___x_584_);
v___x_641_ = lean_box(0);
v_isShared_642_ = v_isSharedCheck_646_;
goto v_resetjp_640_;
}
v_resetjp_640_:
{
lean_object* v___x_644_; 
if (v_isShared_642_ == 0)
{
v___x_644_ = v___x_641_;
goto v_reusejp_643_;
}
else
{
lean_object* v_reuseFailAlloc_645_; 
v_reuseFailAlloc_645_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_645_, 0, v_a_639_);
v___x_644_ = v_reuseFailAlloc_645_;
goto v_reusejp_643_;
}
v_reusejp_643_:
{
return v___x_644_;
}
}
}
}
else
{
lean_del_object(v___x_547_);
lean_dec(v_val_545_);
lean_dec(v_snd_531_);
v_a_537_ = v___x_581_;
goto v___jp_536_;
}
v___jp_550_:
{
size_t v_sz_556_; size_t v___x_557_; lean_object* v___x_558_; 
v_sz_556_ = lean_array_size(v_subgoals_551_);
v___x_557_ = ((size_t)0ULL);
v___x_558_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_CasesM_0__Lean_MVarId_casesMatching_go_spec__0(v_recursive_515_, v_matcher_516_, v_allowSplit_517_, v_val_545_, v_subgoals_551_, v_sz_556_, v___x_557_, v_acc_518_, v___y_552_, v___y_553_, v___y_554_, v___y_555_);
lean_dec_ref(v_subgoals_551_);
lean_dec(v_val_545_);
if (lean_obj_tag(v___x_558_) == 0)
{
lean_object* v_a_559_; lean_object* v___x_561_; uint8_t v_isShared_562_; uint8_t v_isSharedCheck_572_; 
v_a_559_ = lean_ctor_get(v___x_558_, 0);
v_isSharedCheck_572_ = !lean_is_exclusive(v___x_558_);
if (v_isSharedCheck_572_ == 0)
{
v___x_561_ = v___x_558_;
v_isShared_562_ = v_isSharedCheck_572_;
goto v_resetjp_560_;
}
else
{
lean_inc(v_a_559_);
lean_dec(v___x_558_);
v___x_561_ = lean_box(0);
v_isShared_562_ = v_isSharedCheck_572_;
goto v_resetjp_560_;
}
v_resetjp_560_:
{
lean_object* v___x_564_; 
if (v_isShared_548_ == 0)
{
lean_ctor_set(v___x_547_, 0, v_a_559_);
v___x_564_ = v___x_547_;
goto v_reusejp_563_;
}
else
{
lean_object* v_reuseFailAlloc_571_; 
v_reuseFailAlloc_571_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_571_, 0, v_a_559_);
v___x_564_ = v_reuseFailAlloc_571_;
goto v_reusejp_563_;
}
v_reusejp_563_:
{
lean_object* v___x_565_; lean_object* v___x_566_; lean_object* v___x_567_; lean_object* v___x_569_; 
v___x_565_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_565_, 0, v___x_564_);
lean_ctor_set(v___x_565_, 1, v___x_549_);
v___x_566_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_566_, 0, v___x_565_);
v___x_567_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_567_, 0, v___x_566_);
lean_ctor_set(v___x_567_, 1, v_snd_531_);
if (v_isShared_562_ == 0)
{
lean_ctor_set(v___x_561_, 0, v___x_567_);
v___x_569_ = v___x_561_;
goto v_reusejp_568_;
}
else
{
lean_object* v_reuseFailAlloc_570_; 
v_reuseFailAlloc_570_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_570_, 0, v___x_567_);
v___x_569_ = v_reuseFailAlloc_570_;
goto v_reusejp_568_;
}
v_reusejp_568_:
{
return v___x_569_;
}
}
}
}
else
{
lean_object* v_a_573_; lean_object* v___x_575_; uint8_t v_isShared_576_; uint8_t v_isSharedCheck_580_; 
lean_del_object(v___x_547_);
lean_dec(v_snd_531_);
v_a_573_ = lean_ctor_get(v___x_558_, 0);
v_isSharedCheck_580_ = !lean_is_exclusive(v___x_558_);
if (v_isSharedCheck_580_ == 0)
{
v___x_575_ = v___x_558_;
v_isShared_576_ = v_isSharedCheck_580_;
goto v_resetjp_574_;
}
else
{
lean_inc(v_a_573_);
lean_dec(v___x_558_);
v___x_575_ = lean_box(0);
v_isShared_576_ = v_isSharedCheck_580_;
goto v_resetjp_574_;
}
v_resetjp_574_:
{
lean_object* v___x_578_; 
if (v_isShared_576_ == 0)
{
v___x_578_ = v___x_575_;
goto v_reusejp_577_;
}
else
{
lean_object* v_reuseFailAlloc_579_; 
v_reuseFailAlloc_579_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_579_, 0, v_a_573_);
v___x_578_ = v_reuseFailAlloc_579_;
goto v_reusejp_577_;
}
v_reusejp_577_:
{
return v___x_578_;
}
}
}
}
}
}
v___jp_536_:
{
lean_object* v___x_539_; 
if (v_isShared_534_ == 0)
{
lean_ctor_set(v___x_533_, 1, v_a_537_);
lean_ctor_set(v___x_533_, 0, v___x_535_);
v___x_539_ = v___x_533_;
goto v_reusejp_538_;
}
else
{
lean_object* v_reuseFailAlloc_543_; 
v_reuseFailAlloc_543_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_543_, 0, v___x_535_);
lean_ctor_set(v_reuseFailAlloc_543_, 1, v_a_537_);
v___x_539_ = v_reuseFailAlloc_543_;
goto v_reusejp_538_;
}
v_reusejp_538_:
{
size_t v___x_540_; size_t v___x_541_; 
v___x_540_ = ((size_t)1ULL);
v___x_541_ = lean_usize_add(v_i_522_, v___x_540_);
v_i_522_ = v___x_541_;
v_b_523_ = v___x_539_;
goto _start;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00__private_Mathlib_Tactic_CasesM_0__Lean_MVarId_casesMatching_go_spec__1_spec__2(uint8_t v_recursive_650_, lean_object* v_matcher_651_, uint8_t v_allowSplit_652_, lean_object* v_acc_653_, lean_object* v_g_654_, lean_object* v_as_655_, size_t v_sz_656_, size_t v_i_657_, lean_object* v_b_658_, lean_object* v___y_659_, lean_object* v___y_660_, lean_object* v___y_661_, lean_object* v___y_662_){
_start:
{
uint8_t v___x_664_; 
v___x_664_ = lean_usize_dec_lt(v_i_657_, v_sz_656_);
if (v___x_664_ == 0)
{
lean_object* v___x_665_; 
lean_dec(v_g_654_);
lean_dec_ref(v_acc_653_);
lean_dec_ref(v_matcher_651_);
v___x_665_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_665_, 0, v_b_658_);
return v___x_665_;
}
else
{
lean_object* v_snd_666_; lean_object* v___x_668_; uint8_t v_isShared_669_; uint8_t v_isSharedCheck_783_; 
v_snd_666_ = lean_ctor_get(v_b_658_, 1);
v_isSharedCheck_783_ = !lean_is_exclusive(v_b_658_);
if (v_isSharedCheck_783_ == 0)
{
lean_object* v_unused_784_; 
v_unused_784_ = lean_ctor_get(v_b_658_, 0);
lean_dec(v_unused_784_);
v___x_668_ = v_b_658_;
v_isShared_669_ = v_isSharedCheck_783_;
goto v_resetjp_667_;
}
else
{
lean_inc(v_snd_666_);
lean_dec(v_b_658_);
v___x_668_ = lean_box(0);
v_isShared_669_ = v_isSharedCheck_783_;
goto v_resetjp_667_;
}
v_resetjp_667_:
{
lean_object* v___x_670_; lean_object* v_a_672_; lean_object* v_a_679_; 
v___x_670_ = lean_box(0);
v_a_679_ = lean_array_uget(v_as_655_, v_i_657_);
if (lean_obj_tag(v_a_679_) == 0)
{
v_a_672_ = v_snd_666_;
goto v___jp_671_;
}
else
{
lean_object* v_val_680_; lean_object* v___x_682_; uint8_t v_isShared_683_; uint8_t v_isSharedCheck_782_; 
v_val_680_ = lean_ctor_get(v_a_679_, 0);
v_isSharedCheck_782_ = !lean_is_exclusive(v_a_679_);
if (v_isSharedCheck_782_ == 0)
{
v___x_682_ = v_a_679_;
v_isShared_683_ = v_isSharedCheck_782_;
goto v_resetjp_681_;
}
else
{
lean_inc(v_val_680_);
lean_dec(v_a_679_);
v___x_682_ = lean_box(0);
v_isShared_683_ = v_isSharedCheck_782_;
goto v_resetjp_681_;
}
v_resetjp_681_:
{
lean_object* v___x_684_; lean_object* v_subgoals_686_; lean_object* v___y_687_; lean_object* v___y_688_; lean_object* v___y_689_; lean_object* v___y_690_; lean_object* v___x_716_; uint8_t v___x_717_; 
v___x_684_ = lean_box(0);
v___x_716_ = ((lean_object*)(lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00__private_Mathlib_Tactic_CasesM_0__Lean_MVarId_casesMatching_go_spec__1_spec__2___closed__0));
v___x_717_ = l_Lean_LocalDecl_isImplementationDetail(v_val_680_);
if (v___x_717_ == 0)
{
lean_object* v___x_718_; lean_object* v___x_719_; 
v___x_718_ = l_Lean_LocalDecl_type(v_val_680_);
lean_inc_ref(v_matcher_651_);
lean_inc(v___y_662_);
lean_inc_ref(v___y_661_);
lean_inc(v___y_660_);
lean_inc_ref(v___y_659_);
v___x_719_ = lean_apply_6(v_matcher_651_, v___x_718_, v___y_659_, v___y_660_, v___y_661_, v___y_662_, lean_box(0));
if (lean_obj_tag(v___x_719_) == 0)
{
lean_object* v_a_720_; uint8_t v___x_721_; 
v_a_720_ = lean_ctor_get(v___x_719_, 0);
lean_inc(v_a_720_);
lean_dec_ref_known(v___x_719_, 1);
v___x_721_ = lean_unbox(v_a_720_);
if (v___x_721_ == 0)
{
lean_dec(v_a_720_);
lean_del_object(v___x_682_);
lean_dec(v_val_680_);
lean_dec(v_snd_666_);
v_a_672_ = v___x_716_;
goto v___jp_671_;
}
else
{
if (v_allowSplit_652_ == 0)
{
lean_object* v___x_722_; 
v___x_722_ = l_Lean_Meta_saveState___redArg(v___y_660_, v___y_662_);
if (lean_obj_tag(v___x_722_) == 0)
{
lean_object* v_a_723_; lean_object* v___x_724_; lean_object* v___x_725_; lean_object* v___x_726_; lean_object* v___x_727_; lean_object* v___x_728_; uint8_t v___x_729_; lean_object* v___x_730_; lean_object* v___x_731_; lean_object* v___x_732_; lean_object* v___x_733_; 
v_a_723_ = lean_ctor_get(v___x_722_, 0);
lean_inc(v_a_723_);
lean_dec_ref_known(v___x_722_, 1);
v___x_724_ = l_Lean_LocalDecl_fvarId(v_val_680_);
v___x_725_ = l_Lean_LocalDecl_userName(v_val_680_);
v___x_726_ = lean_box(0);
v___x_727_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_727_, 0, v___x_725_);
lean_ctor_set(v___x_727_, 1, v___x_726_);
v___x_728_ = lean_alloc_ctor(0, 1, 1);
lean_ctor_set(v___x_728_, 0, v___x_727_);
v___x_729_ = lean_unbox(v_a_720_);
lean_dec(v_a_720_);
lean_ctor_set_uint8(v___x_728_, sizeof(void*)*1, v___x_729_);
v___x_730_ = lean_unsigned_to_nat(1u);
v___x_731_ = lean_mk_empty_array_with_capacity(v___x_730_);
v___x_732_ = lean_array_push(v___x_731_, v___x_728_);
lean_inc(v_g_654_);
v___x_733_ = l_Lean_MVarId_cases(v_g_654_, v___x_724_, v___x_732_, v_allowSplit_652_, v___x_670_, v___y_659_, v___y_660_, v___y_661_, v___y_662_);
if (lean_obj_tag(v___x_733_) == 0)
{
lean_object* v_a_734_; lean_object* v___x_735_; uint8_t v___x_736_; 
v_a_734_ = lean_ctor_get(v___x_733_, 0);
lean_inc(v_a_734_);
lean_dec_ref_known(v___x_733_, 1);
v___x_735_ = lean_array_get_size(v_a_734_);
v___x_736_ = lean_nat_dec_lt(v___x_730_, v___x_735_);
if (v___x_736_ == 0)
{
lean_dec(v_a_723_);
lean_del_object(v___x_668_);
lean_dec(v_g_654_);
v_subgoals_686_ = v_a_734_;
v___y_687_ = v___y_659_;
v___y_688_ = v___y_660_;
v___y_689_ = v___y_661_;
v___y_690_ = v___y_662_;
goto v___jp_685_;
}
else
{
lean_object* v___x_737_; 
lean_dec(v_a_734_);
lean_del_object(v___x_682_);
lean_dec(v_val_680_);
lean_dec(v_snd_666_);
v___x_737_ = l_Lean_Meta_SavedState_restore___redArg(v_a_723_, v___y_660_, v___y_662_);
lean_dec(v_a_723_);
if (lean_obj_tag(v___x_737_) == 0)
{
lean_dec_ref_known(v___x_737_, 1);
v_a_672_ = v___x_716_;
goto v___jp_671_;
}
else
{
lean_object* v_a_738_; lean_object* v___x_740_; uint8_t v_isShared_741_; uint8_t v_isSharedCheck_745_; 
lean_del_object(v___x_668_);
lean_dec(v_g_654_);
lean_dec_ref(v_acc_653_);
lean_dec_ref(v_matcher_651_);
v_a_738_ = lean_ctor_get(v___x_737_, 0);
v_isSharedCheck_745_ = !lean_is_exclusive(v___x_737_);
if (v_isSharedCheck_745_ == 0)
{
v___x_740_ = v___x_737_;
v_isShared_741_ = v_isSharedCheck_745_;
goto v_resetjp_739_;
}
else
{
lean_inc(v_a_738_);
lean_dec(v___x_737_);
v___x_740_ = lean_box(0);
v_isShared_741_ = v_isSharedCheck_745_;
goto v_resetjp_739_;
}
v_resetjp_739_:
{
lean_object* v___x_743_; 
if (v_isShared_741_ == 0)
{
v___x_743_ = v___x_740_;
goto v_reusejp_742_;
}
else
{
lean_object* v_reuseFailAlloc_744_; 
v_reuseFailAlloc_744_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_744_, 0, v_a_738_);
v___x_743_ = v_reuseFailAlloc_744_;
goto v_reusejp_742_;
}
v_reusejp_742_:
{
return v___x_743_;
}
}
}
}
}
else
{
lean_object* v_a_746_; lean_object* v___x_748_; uint8_t v_isShared_749_; uint8_t v_isSharedCheck_753_; 
lean_dec(v_a_723_);
lean_del_object(v___x_682_);
lean_dec(v_val_680_);
lean_del_object(v___x_668_);
lean_dec(v_snd_666_);
lean_dec(v_g_654_);
lean_dec_ref(v_acc_653_);
lean_dec_ref(v_matcher_651_);
v_a_746_ = lean_ctor_get(v___x_733_, 0);
v_isSharedCheck_753_ = !lean_is_exclusive(v___x_733_);
if (v_isSharedCheck_753_ == 0)
{
v___x_748_ = v___x_733_;
v_isShared_749_ = v_isSharedCheck_753_;
goto v_resetjp_747_;
}
else
{
lean_inc(v_a_746_);
lean_dec(v___x_733_);
v___x_748_ = lean_box(0);
v_isShared_749_ = v_isSharedCheck_753_;
goto v_resetjp_747_;
}
v_resetjp_747_:
{
lean_object* v___x_751_; 
if (v_isShared_749_ == 0)
{
v___x_751_ = v___x_748_;
goto v_reusejp_750_;
}
else
{
lean_object* v_reuseFailAlloc_752_; 
v_reuseFailAlloc_752_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_752_, 0, v_a_746_);
v___x_751_ = v_reuseFailAlloc_752_;
goto v_reusejp_750_;
}
v_reusejp_750_:
{
return v___x_751_;
}
}
}
}
else
{
lean_object* v_a_754_; lean_object* v___x_756_; uint8_t v_isShared_757_; uint8_t v_isSharedCheck_761_; 
lean_dec(v_a_720_);
lean_del_object(v___x_682_);
lean_dec(v_val_680_);
lean_del_object(v___x_668_);
lean_dec(v_snd_666_);
lean_dec(v_g_654_);
lean_dec_ref(v_acc_653_);
lean_dec_ref(v_matcher_651_);
v_a_754_ = lean_ctor_get(v___x_722_, 0);
v_isSharedCheck_761_ = !lean_is_exclusive(v___x_722_);
if (v_isSharedCheck_761_ == 0)
{
v___x_756_ = v___x_722_;
v_isShared_757_ = v_isSharedCheck_761_;
goto v_resetjp_755_;
}
else
{
lean_inc(v_a_754_);
lean_dec(v___x_722_);
v___x_756_ = lean_box(0);
v_isShared_757_ = v_isSharedCheck_761_;
goto v_resetjp_755_;
}
v_resetjp_755_:
{
lean_object* v___x_759_; 
if (v_isShared_757_ == 0)
{
v___x_759_ = v___x_756_;
goto v_reusejp_758_;
}
else
{
lean_object* v_reuseFailAlloc_760_; 
v_reuseFailAlloc_760_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_760_, 0, v_a_754_);
v___x_759_ = v_reuseFailAlloc_760_;
goto v_reusejp_758_;
}
v_reusejp_758_:
{
return v___x_759_;
}
}
}
}
else
{
lean_object* v___x_762_; lean_object* v___x_763_; lean_object* v___x_764_; 
lean_dec(v_a_720_);
lean_del_object(v___x_668_);
v___x_762_ = l_Lean_LocalDecl_fvarId(v_val_680_);
v___x_763_ = ((lean_object*)(lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00__private_Mathlib_Tactic_CasesM_0__Lean_MVarId_casesMatching_go_spec__1_spec__2___closed__1));
v___x_764_ = l_Lean_MVarId_cases(v_g_654_, v___x_762_, v___x_763_, v___x_717_, v___x_670_, v___y_659_, v___y_660_, v___y_661_, v___y_662_);
if (lean_obj_tag(v___x_764_) == 0)
{
lean_object* v_a_765_; 
v_a_765_ = lean_ctor_get(v___x_764_, 0);
lean_inc(v_a_765_);
lean_dec_ref_known(v___x_764_, 1);
v_subgoals_686_ = v_a_765_;
v___y_687_ = v___y_659_;
v___y_688_ = v___y_660_;
v___y_689_ = v___y_661_;
v___y_690_ = v___y_662_;
goto v___jp_685_;
}
else
{
lean_object* v_a_766_; lean_object* v___x_768_; uint8_t v_isShared_769_; uint8_t v_isSharedCheck_773_; 
lean_del_object(v___x_682_);
lean_dec(v_val_680_);
lean_dec(v_snd_666_);
lean_dec_ref(v_acc_653_);
lean_dec_ref(v_matcher_651_);
v_a_766_ = lean_ctor_get(v___x_764_, 0);
v_isSharedCheck_773_ = !lean_is_exclusive(v___x_764_);
if (v_isSharedCheck_773_ == 0)
{
v___x_768_ = v___x_764_;
v_isShared_769_ = v_isSharedCheck_773_;
goto v_resetjp_767_;
}
else
{
lean_inc(v_a_766_);
lean_dec(v___x_764_);
v___x_768_ = lean_box(0);
v_isShared_769_ = v_isSharedCheck_773_;
goto v_resetjp_767_;
}
v_resetjp_767_:
{
lean_object* v___x_771_; 
if (v_isShared_769_ == 0)
{
v___x_771_ = v___x_768_;
goto v_reusejp_770_;
}
else
{
lean_object* v_reuseFailAlloc_772_; 
v_reuseFailAlloc_772_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_772_, 0, v_a_766_);
v___x_771_ = v_reuseFailAlloc_772_;
goto v_reusejp_770_;
}
v_reusejp_770_:
{
return v___x_771_;
}
}
}
}
}
}
else
{
lean_object* v_a_774_; lean_object* v___x_776_; uint8_t v_isShared_777_; uint8_t v_isSharedCheck_781_; 
lean_del_object(v___x_682_);
lean_dec(v_val_680_);
lean_del_object(v___x_668_);
lean_dec(v_snd_666_);
lean_dec(v_g_654_);
lean_dec_ref(v_acc_653_);
lean_dec_ref(v_matcher_651_);
v_a_774_ = lean_ctor_get(v___x_719_, 0);
v_isSharedCheck_781_ = !lean_is_exclusive(v___x_719_);
if (v_isSharedCheck_781_ == 0)
{
v___x_776_ = v___x_719_;
v_isShared_777_ = v_isSharedCheck_781_;
goto v_resetjp_775_;
}
else
{
lean_inc(v_a_774_);
lean_dec(v___x_719_);
v___x_776_ = lean_box(0);
v_isShared_777_ = v_isSharedCheck_781_;
goto v_resetjp_775_;
}
v_resetjp_775_:
{
lean_object* v___x_779_; 
if (v_isShared_777_ == 0)
{
v___x_779_ = v___x_776_;
goto v_reusejp_778_;
}
else
{
lean_object* v_reuseFailAlloc_780_; 
v_reuseFailAlloc_780_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_780_, 0, v_a_774_);
v___x_779_ = v_reuseFailAlloc_780_;
goto v_reusejp_778_;
}
v_reusejp_778_:
{
return v___x_779_;
}
}
}
}
else
{
lean_del_object(v___x_682_);
lean_dec(v_val_680_);
lean_dec(v_snd_666_);
v_a_672_ = v___x_716_;
goto v___jp_671_;
}
v___jp_685_:
{
size_t v_sz_691_; size_t v___x_692_; lean_object* v___x_693_; 
v_sz_691_ = lean_array_size(v_subgoals_686_);
v___x_692_ = ((size_t)0ULL);
v___x_693_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_CasesM_0__Lean_MVarId_casesMatching_go_spec__0(v_recursive_650_, v_matcher_651_, v_allowSplit_652_, v_val_680_, v_subgoals_686_, v_sz_691_, v___x_692_, v_acc_653_, v___y_687_, v___y_688_, v___y_689_, v___y_690_);
lean_dec_ref(v_subgoals_686_);
lean_dec(v_val_680_);
if (lean_obj_tag(v___x_693_) == 0)
{
lean_object* v_a_694_; lean_object* v___x_696_; uint8_t v_isShared_697_; uint8_t v_isSharedCheck_707_; 
v_a_694_ = lean_ctor_get(v___x_693_, 0);
v_isSharedCheck_707_ = !lean_is_exclusive(v___x_693_);
if (v_isSharedCheck_707_ == 0)
{
v___x_696_ = v___x_693_;
v_isShared_697_ = v_isSharedCheck_707_;
goto v_resetjp_695_;
}
else
{
lean_inc(v_a_694_);
lean_dec(v___x_693_);
v___x_696_ = lean_box(0);
v_isShared_697_ = v_isSharedCheck_707_;
goto v_resetjp_695_;
}
v_resetjp_695_:
{
lean_object* v___x_699_; 
if (v_isShared_683_ == 0)
{
lean_ctor_set(v___x_682_, 0, v_a_694_);
v___x_699_ = v___x_682_;
goto v_reusejp_698_;
}
else
{
lean_object* v_reuseFailAlloc_706_; 
v_reuseFailAlloc_706_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_706_, 0, v_a_694_);
v___x_699_ = v_reuseFailAlloc_706_;
goto v_reusejp_698_;
}
v_reusejp_698_:
{
lean_object* v___x_700_; lean_object* v___x_701_; lean_object* v___x_702_; lean_object* v___x_704_; 
v___x_700_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_700_, 0, v___x_699_);
lean_ctor_set(v___x_700_, 1, v___x_684_);
v___x_701_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_701_, 0, v___x_700_);
v___x_702_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_702_, 0, v___x_701_);
lean_ctor_set(v___x_702_, 1, v_snd_666_);
if (v_isShared_697_ == 0)
{
lean_ctor_set(v___x_696_, 0, v___x_702_);
v___x_704_ = v___x_696_;
goto v_reusejp_703_;
}
else
{
lean_object* v_reuseFailAlloc_705_; 
v_reuseFailAlloc_705_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_705_, 0, v___x_702_);
v___x_704_ = v_reuseFailAlloc_705_;
goto v_reusejp_703_;
}
v_reusejp_703_:
{
return v___x_704_;
}
}
}
}
else
{
lean_object* v_a_708_; lean_object* v___x_710_; uint8_t v_isShared_711_; uint8_t v_isSharedCheck_715_; 
lean_del_object(v___x_682_);
lean_dec(v_snd_666_);
v_a_708_ = lean_ctor_get(v___x_693_, 0);
v_isSharedCheck_715_ = !lean_is_exclusive(v___x_693_);
if (v_isSharedCheck_715_ == 0)
{
v___x_710_ = v___x_693_;
v_isShared_711_ = v_isSharedCheck_715_;
goto v_resetjp_709_;
}
else
{
lean_inc(v_a_708_);
lean_dec(v___x_693_);
v___x_710_ = lean_box(0);
v_isShared_711_ = v_isSharedCheck_715_;
goto v_resetjp_709_;
}
v_resetjp_709_:
{
lean_object* v___x_713_; 
if (v_isShared_711_ == 0)
{
v___x_713_ = v___x_710_;
goto v_reusejp_712_;
}
else
{
lean_object* v_reuseFailAlloc_714_; 
v_reuseFailAlloc_714_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_714_, 0, v_a_708_);
v___x_713_ = v_reuseFailAlloc_714_;
goto v_reusejp_712_;
}
v_reusejp_712_:
{
return v___x_713_;
}
}
}
}
}
}
v___jp_671_:
{
lean_object* v___x_674_; 
if (v_isShared_669_ == 0)
{
lean_ctor_set(v___x_668_, 1, v_a_672_);
lean_ctor_set(v___x_668_, 0, v___x_670_);
v___x_674_ = v___x_668_;
goto v_reusejp_673_;
}
else
{
lean_object* v_reuseFailAlloc_678_; 
v_reuseFailAlloc_678_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_678_, 0, v___x_670_);
lean_ctor_set(v_reuseFailAlloc_678_, 1, v_a_672_);
v___x_674_ = v_reuseFailAlloc_678_;
goto v_reusejp_673_;
}
v_reusejp_673_:
{
size_t v___x_675_; size_t v___x_676_; lean_object* v___x_677_; 
v___x_675_ = ((size_t)1ULL);
v___x_676_ = lean_usize_add(v_i_657_, v___x_675_);
v___x_677_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00__private_Mathlib_Tactic_CasesM_0__Lean_MVarId_casesMatching_go_spec__1_spec__2_spec__6(v_recursive_650_, v_matcher_651_, v_allowSplit_652_, v_acc_653_, v_g_654_, v_as_655_, v_sz_656_, v___x_676_, v___x_674_, v___y_659_, v___y_660_, v___y_661_, v___y_662_);
return v___x_677_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentArray_forIn___at___00__private_Mathlib_Tactic_CasesM_0__Lean_MVarId_casesMatching_go_spec__1(uint8_t v_recursive_785_, lean_object* v_matcher_786_, uint8_t v_allowSplit_787_, lean_object* v_acc_788_, lean_object* v_g_789_, lean_object* v_t_790_, lean_object* v_init_791_, lean_object* v___y_792_, lean_object* v___y_793_, lean_object* v___y_794_, lean_object* v___y_795_){
_start:
{
lean_object* v_root_797_; lean_object* v_tail_798_; lean_object* v___x_799_; 
v_root_797_ = lean_ctor_get(v_t_790_, 0);
v_tail_798_ = lean_ctor_get(v_t_790_, 1);
lean_inc(v_g_789_);
lean_inc_ref(v_acc_788_);
lean_inc_ref(v_matcher_786_);
lean_inc_ref(v_init_791_);
v___x_799_ = lp_mathlib_Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00__private_Mathlib_Tactic_CasesM_0__Lean_MVarId_casesMatching_go_spec__1_spec__1(v_init_791_, v_recursive_785_, v_matcher_786_, v_allowSplit_787_, v_acc_788_, v_g_789_, v_root_797_, v_init_791_, v___y_792_, v___y_793_, v___y_794_, v___y_795_);
lean_dec_ref(v_init_791_);
if (lean_obj_tag(v___x_799_) == 0)
{
lean_object* v_a_800_; lean_object* v___x_802_; uint8_t v_isShared_803_; uint8_t v_isSharedCheck_836_; 
v_a_800_ = lean_ctor_get(v___x_799_, 0);
v_isSharedCheck_836_ = !lean_is_exclusive(v___x_799_);
if (v_isSharedCheck_836_ == 0)
{
v___x_802_ = v___x_799_;
v_isShared_803_ = v_isSharedCheck_836_;
goto v_resetjp_801_;
}
else
{
lean_inc(v_a_800_);
lean_dec(v___x_799_);
v___x_802_ = lean_box(0);
v_isShared_803_ = v_isSharedCheck_836_;
goto v_resetjp_801_;
}
v_resetjp_801_:
{
if (lean_obj_tag(v_a_800_) == 0)
{
lean_object* v_a_804_; lean_object* v___x_806_; 
lean_dec(v_g_789_);
lean_dec_ref(v_acc_788_);
lean_dec_ref(v_matcher_786_);
v_a_804_ = lean_ctor_get(v_a_800_, 0);
lean_inc(v_a_804_);
lean_dec_ref_known(v_a_800_, 1);
if (v_isShared_803_ == 0)
{
lean_ctor_set(v___x_802_, 0, v_a_804_);
v___x_806_ = v___x_802_;
goto v_reusejp_805_;
}
else
{
lean_object* v_reuseFailAlloc_807_; 
v_reuseFailAlloc_807_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_807_, 0, v_a_804_);
v___x_806_ = v_reuseFailAlloc_807_;
goto v_reusejp_805_;
}
v_reusejp_805_:
{
return v___x_806_;
}
}
else
{
lean_object* v_a_808_; lean_object* v___x_809_; lean_object* v___x_810_; size_t v_sz_811_; size_t v___x_812_; lean_object* v___x_813_; 
lean_del_object(v___x_802_);
v_a_808_ = lean_ctor_get(v_a_800_, 0);
lean_inc(v_a_808_);
lean_dec_ref_known(v_a_800_, 1);
v___x_809_ = lean_box(0);
v___x_810_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_810_, 0, v___x_809_);
lean_ctor_set(v___x_810_, 1, v_a_808_);
v_sz_811_ = lean_array_size(v_tail_798_);
v___x_812_ = ((size_t)0ULL);
v___x_813_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00__private_Mathlib_Tactic_CasesM_0__Lean_MVarId_casesMatching_go_spec__1_spec__2(v_recursive_785_, v_matcher_786_, v_allowSplit_787_, v_acc_788_, v_g_789_, v_tail_798_, v_sz_811_, v___x_812_, v___x_810_, v___y_792_, v___y_793_, v___y_794_, v___y_795_);
if (lean_obj_tag(v___x_813_) == 0)
{
lean_object* v_a_814_; lean_object* v___x_816_; uint8_t v_isShared_817_; uint8_t v_isSharedCheck_827_; 
v_a_814_ = lean_ctor_get(v___x_813_, 0);
v_isSharedCheck_827_ = !lean_is_exclusive(v___x_813_);
if (v_isSharedCheck_827_ == 0)
{
v___x_816_ = v___x_813_;
v_isShared_817_ = v_isSharedCheck_827_;
goto v_resetjp_815_;
}
else
{
lean_inc(v_a_814_);
lean_dec(v___x_813_);
v___x_816_ = lean_box(0);
v_isShared_817_ = v_isSharedCheck_827_;
goto v_resetjp_815_;
}
v_resetjp_815_:
{
lean_object* v_fst_818_; 
v_fst_818_ = lean_ctor_get(v_a_814_, 0);
if (lean_obj_tag(v_fst_818_) == 0)
{
lean_object* v_snd_819_; lean_object* v___x_821_; 
v_snd_819_ = lean_ctor_get(v_a_814_, 1);
lean_inc(v_snd_819_);
lean_dec(v_a_814_);
if (v_isShared_817_ == 0)
{
lean_ctor_set(v___x_816_, 0, v_snd_819_);
v___x_821_ = v___x_816_;
goto v_reusejp_820_;
}
else
{
lean_object* v_reuseFailAlloc_822_; 
v_reuseFailAlloc_822_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_822_, 0, v_snd_819_);
v___x_821_ = v_reuseFailAlloc_822_;
goto v_reusejp_820_;
}
v_reusejp_820_:
{
return v___x_821_;
}
}
else
{
lean_object* v_val_823_; lean_object* v___x_825_; 
lean_inc_ref(v_fst_818_);
lean_dec(v_a_814_);
v_val_823_ = lean_ctor_get(v_fst_818_, 0);
lean_inc(v_val_823_);
lean_dec_ref_known(v_fst_818_, 1);
if (v_isShared_817_ == 0)
{
lean_ctor_set(v___x_816_, 0, v_val_823_);
v___x_825_ = v___x_816_;
goto v_reusejp_824_;
}
else
{
lean_object* v_reuseFailAlloc_826_; 
v_reuseFailAlloc_826_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_826_, 0, v_val_823_);
v___x_825_ = v_reuseFailAlloc_826_;
goto v_reusejp_824_;
}
v_reusejp_824_:
{
return v___x_825_;
}
}
}
}
else
{
lean_object* v_a_828_; lean_object* v___x_830_; uint8_t v_isShared_831_; uint8_t v_isSharedCheck_835_; 
v_a_828_ = lean_ctor_get(v___x_813_, 0);
v_isSharedCheck_835_ = !lean_is_exclusive(v___x_813_);
if (v_isSharedCheck_835_ == 0)
{
v___x_830_ = v___x_813_;
v_isShared_831_ = v_isSharedCheck_835_;
goto v_resetjp_829_;
}
else
{
lean_inc(v_a_828_);
lean_dec(v___x_813_);
v___x_830_ = lean_box(0);
v_isShared_831_ = v_isSharedCheck_835_;
goto v_resetjp_829_;
}
v_resetjp_829_:
{
lean_object* v___x_833_; 
if (v_isShared_831_ == 0)
{
v___x_833_ = v___x_830_;
goto v_reusejp_832_;
}
else
{
lean_object* v_reuseFailAlloc_834_; 
v_reuseFailAlloc_834_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_834_, 0, v_a_828_);
v___x_833_ = v_reuseFailAlloc_834_;
goto v_reusejp_832_;
}
v_reusejp_832_:
{
return v___x_833_;
}
}
}
}
}
}
else
{
lean_object* v_a_837_; lean_object* v___x_839_; uint8_t v_isShared_840_; uint8_t v_isSharedCheck_844_; 
lean_dec(v_g_789_);
lean_dec_ref(v_acc_788_);
lean_dec_ref(v_matcher_786_);
v_a_837_ = lean_ctor_get(v___x_799_, 0);
v_isSharedCheck_844_ = !lean_is_exclusive(v___x_799_);
if (v_isSharedCheck_844_ == 0)
{
v___x_839_ = v___x_799_;
v_isShared_840_ = v_isSharedCheck_844_;
goto v_resetjp_838_;
}
else
{
lean_inc(v_a_837_);
lean_dec(v___x_799_);
v___x_839_ = lean_box(0);
v_isShared_840_ = v_isSharedCheck_844_;
goto v_resetjp_838_;
}
v_resetjp_838_:
{
lean_object* v___x_842_; 
if (v_isShared_840_ == 0)
{
v___x_842_ = v___x_839_;
goto v_reusejp_841_;
}
else
{
lean_object* v_reuseFailAlloc_843_; 
v_reuseFailAlloc_843_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_843_, 0, v_a_837_);
v___x_842_ = v_reuseFailAlloc_843_;
goto v_reusejp_841_;
}
v_reusejp_841_:
{
return v___x_842_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_CasesM_0__Lean_MVarId_casesMatching_go___lam__0(uint8_t v_recursive_845_, lean_object* v_matcher_846_, uint8_t v_allowSplit_847_, lean_object* v_acc_848_, lean_object* v_g_849_, lean_object* v___y_850_, lean_object* v___y_851_, lean_object* v___y_852_, lean_object* v___y_853_){
_start:
{
lean_object* v_lctx_855_; lean_object* v_decls_856_; lean_object* v___x_857_; lean_object* v___x_858_; 
v_lctx_855_ = lean_ctor_get(v___y_850_, 2);
v_decls_856_ = lean_ctor_get(v_lctx_855_, 1);
v___x_857_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_CasesM_0__Lean_MVarId_casesMatching_go___lam__0___closed__0));
lean_inc(v_g_849_);
lean_inc_ref(v_acc_848_);
v___x_858_ = lp_mathlib_Lean_PersistentArray_forIn___at___00__private_Mathlib_Tactic_CasesM_0__Lean_MVarId_casesMatching_go_spec__1(v_recursive_845_, v_matcher_846_, v_allowSplit_847_, v_acc_848_, v_g_849_, v_decls_856_, v___x_857_, v___y_850_, v___y_851_, v___y_852_, v___y_853_);
if (lean_obj_tag(v___x_858_) == 0)
{
lean_object* v_a_859_; lean_object* v___x_861_; uint8_t v_isShared_862_; uint8_t v_isSharedCheck_872_; 
v_a_859_ = lean_ctor_get(v___x_858_, 0);
v_isSharedCheck_872_ = !lean_is_exclusive(v___x_858_);
if (v_isSharedCheck_872_ == 0)
{
v___x_861_ = v___x_858_;
v_isShared_862_ = v_isSharedCheck_872_;
goto v_resetjp_860_;
}
else
{
lean_inc(v_a_859_);
lean_dec(v___x_858_);
v___x_861_ = lean_box(0);
v_isShared_862_ = v_isSharedCheck_872_;
goto v_resetjp_860_;
}
v_resetjp_860_:
{
lean_object* v_fst_863_; 
v_fst_863_ = lean_ctor_get(v_a_859_, 0);
lean_inc(v_fst_863_);
lean_dec(v_a_859_);
if (lean_obj_tag(v_fst_863_) == 0)
{
lean_object* v___x_864_; lean_object* v___x_866_; 
v___x_864_ = lean_array_push(v_acc_848_, v_g_849_);
if (v_isShared_862_ == 0)
{
lean_ctor_set(v___x_861_, 0, v___x_864_);
v___x_866_ = v___x_861_;
goto v_reusejp_865_;
}
else
{
lean_object* v_reuseFailAlloc_867_; 
v_reuseFailAlloc_867_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_867_, 0, v___x_864_);
v___x_866_ = v_reuseFailAlloc_867_;
goto v_reusejp_865_;
}
v_reusejp_865_:
{
return v___x_866_;
}
}
else
{
lean_object* v_val_868_; lean_object* v___x_870_; 
lean_dec(v_g_849_);
lean_dec_ref(v_acc_848_);
v_val_868_ = lean_ctor_get(v_fst_863_, 0);
lean_inc(v_val_868_);
lean_dec_ref_known(v_fst_863_, 1);
if (v_isShared_862_ == 0)
{
lean_ctor_set(v___x_861_, 0, v_val_868_);
v___x_870_ = v___x_861_;
goto v_reusejp_869_;
}
else
{
lean_object* v_reuseFailAlloc_871_; 
v_reuseFailAlloc_871_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_871_, 0, v_val_868_);
v___x_870_ = v_reuseFailAlloc_871_;
goto v_reusejp_869_;
}
v_reusejp_869_:
{
return v___x_870_;
}
}
}
}
else
{
lean_object* v_a_873_; lean_object* v___x_875_; uint8_t v_isShared_876_; uint8_t v_isSharedCheck_880_; 
lean_dec(v_g_849_);
lean_dec_ref(v_acc_848_);
v_a_873_ = lean_ctor_get(v___x_858_, 0);
v_isSharedCheck_880_ = !lean_is_exclusive(v___x_858_);
if (v_isSharedCheck_880_ == 0)
{
v___x_875_ = v___x_858_;
v_isShared_876_ = v_isSharedCheck_880_;
goto v_resetjp_874_;
}
else
{
lean_inc(v_a_873_);
lean_dec(v___x_858_);
v___x_875_ = lean_box(0);
v_isShared_876_ = v_isSharedCheck_880_;
goto v_resetjp_874_;
}
v_resetjp_874_:
{
lean_object* v___x_878_; 
if (v_isShared_876_ == 0)
{
v___x_878_ = v___x_875_;
goto v_reusejp_877_;
}
else
{
lean_object* v_reuseFailAlloc_879_; 
v_reuseFailAlloc_879_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_879_, 0, v_a_873_);
v___x_878_ = v_reuseFailAlloc_879_;
goto v_reusejp_877_;
}
v_reusejp_877_:
{
return v___x_878_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_CasesM_0__Lean_MVarId_casesMatching_go___lam__0___boxed(lean_object* v_recursive_881_, lean_object* v_matcher_882_, lean_object* v_allowSplit_883_, lean_object* v_acc_884_, lean_object* v_g_885_, lean_object* v___y_886_, lean_object* v___y_887_, lean_object* v___y_888_, lean_object* v___y_889_, lean_object* v___y_890_){
_start:
{
uint8_t v_recursive_boxed_891_; uint8_t v_allowSplit_boxed_892_; lean_object* v_res_893_; 
v_recursive_boxed_891_ = lean_unbox(v_recursive_881_);
v_allowSplit_boxed_892_ = lean_unbox(v_allowSplit_883_);
v_res_893_ = lp_mathlib___private_Mathlib_Tactic_CasesM_0__Lean_MVarId_casesMatching_go___lam__0(v_recursive_boxed_891_, v_matcher_882_, v_allowSplit_boxed_892_, v_acc_884_, v_g_885_, v___y_886_, v___y_887_, v___y_888_, v___y_889_);
lean_dec(v___y_889_);
lean_dec_ref(v___y_888_);
lean_dec(v___y_887_);
lean_dec_ref(v___y_886_);
return v_res_893_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_CasesM_0__Lean_MVarId_casesMatching_go(lean_object* v_matcher_894_, uint8_t v_recursive_895_, uint8_t v_allowSplit_896_, lean_object* v_g_897_, lean_object* v_acc_898_, lean_object* v_a_899_, lean_object* v_a_900_, lean_object* v_a_901_, lean_object* v_a_902_){
_start:
{
lean_object* v___x_904_; lean_object* v___x_905_; lean_object* v___f_906_; lean_object* v___x_907_; 
v___x_904_ = lean_box(v_recursive_895_);
v___x_905_ = lean_box(v_allowSplit_896_);
lean_inc(v_g_897_);
v___f_906_ = lean_alloc_closure((void*)(lp_mathlib___private_Mathlib_Tactic_CasesM_0__Lean_MVarId_casesMatching_go___lam__0___boxed), 10, 5);
lean_closure_set(v___f_906_, 0, v___x_904_);
lean_closure_set(v___f_906_, 1, v_matcher_894_);
lean_closure_set(v___f_906_, 2, v___x_905_);
lean_closure_set(v___f_906_, 3, v_acc_898_);
lean_closure_set(v___f_906_, 4, v_g_897_);
v___x_907_ = lp_mathlib_Lean_MVarId_withContext___at___00__private_Mathlib_Tactic_CasesM_0__Lean_MVarId_casesMatching_go_spec__2___redArg(v_g_897_, v___f_906_, v_a_899_, v_a_900_, v_a_901_, v_a_902_);
return v___x_907_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_CasesM_0__Lean_MVarId_casesMatching_go___boxed(lean_object* v_matcher_908_, lean_object* v_recursive_909_, lean_object* v_allowSplit_910_, lean_object* v_g_911_, lean_object* v_acc_912_, lean_object* v_a_913_, lean_object* v_a_914_, lean_object* v_a_915_, lean_object* v_a_916_, lean_object* v_a_917_){
_start:
{
uint8_t v_recursive_boxed_918_; uint8_t v_allowSplit_boxed_919_; lean_object* v_res_920_; 
v_recursive_boxed_918_ = lean_unbox(v_recursive_909_);
v_allowSplit_boxed_919_ = lean_unbox(v_allowSplit_910_);
v_res_920_ = lp_mathlib___private_Mathlib_Tactic_CasesM_0__Lean_MVarId_casesMatching_go(v_matcher_908_, v_recursive_boxed_918_, v_allowSplit_boxed_919_, v_g_911_, v_acc_912_, v_a_913_, v_a_914_, v_a_915_, v_a_916_);
lean_dec(v_a_916_);
lean_dec_ref(v_a_915_);
lean_dec(v_a_914_);
lean_dec_ref(v_a_913_);
return v_res_920_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00__private_Mathlib_Tactic_CasesM_0__Lean_MVarId_casesMatching_go_spec__1_spec__1_spec__3___boxed(lean_object* v_init_921_, lean_object* v_recursive_922_, lean_object* v_matcher_923_, lean_object* v_allowSplit_924_, lean_object* v_acc_925_, lean_object* v_g_926_, lean_object* v_as_927_, lean_object* v_sz_928_, lean_object* v_i_929_, lean_object* v_b_930_, lean_object* v___y_931_, lean_object* v___y_932_, lean_object* v___y_933_, lean_object* v___y_934_, lean_object* v___y_935_){
_start:
{
uint8_t v_recursive_boxed_936_; uint8_t v_allowSplit_boxed_937_; size_t v_sz_boxed_938_; size_t v_i_boxed_939_; lean_object* v_res_940_; 
v_recursive_boxed_936_ = lean_unbox(v_recursive_922_);
v_allowSplit_boxed_937_ = lean_unbox(v_allowSplit_924_);
v_sz_boxed_938_ = lean_unbox_usize(v_sz_928_);
lean_dec(v_sz_928_);
v_i_boxed_939_ = lean_unbox_usize(v_i_929_);
lean_dec(v_i_929_);
v_res_940_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00__private_Mathlib_Tactic_CasesM_0__Lean_MVarId_casesMatching_go_spec__1_spec__1_spec__3(v_init_921_, v_recursive_boxed_936_, v_matcher_923_, v_allowSplit_boxed_937_, v_acc_925_, v_g_926_, v_as_927_, v_sz_boxed_938_, v_i_boxed_939_, v_b_930_, v___y_931_, v___y_932_, v___y_933_, v___y_934_);
lean_dec(v___y_934_);
lean_dec_ref(v___y_933_);
lean_dec(v___y_932_);
lean_dec_ref(v___y_931_);
lean_dec_ref(v_as_927_);
lean_dec_ref(v_init_921_);
return v_res_940_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentArray_forIn___at___00__private_Mathlib_Tactic_CasesM_0__Lean_MVarId_casesMatching_go_spec__1___boxed(lean_object* v_recursive_941_, lean_object* v_matcher_942_, lean_object* v_allowSplit_943_, lean_object* v_acc_944_, lean_object* v_g_945_, lean_object* v_t_946_, lean_object* v_init_947_, lean_object* v___y_948_, lean_object* v___y_949_, lean_object* v___y_950_, lean_object* v___y_951_, lean_object* v___y_952_){
_start:
{
uint8_t v_recursive_boxed_953_; uint8_t v_allowSplit_boxed_954_; lean_object* v_res_955_; 
v_recursive_boxed_953_ = lean_unbox(v_recursive_941_);
v_allowSplit_boxed_954_ = lean_unbox(v_allowSplit_943_);
v_res_955_ = lp_mathlib_Lean_PersistentArray_forIn___at___00__private_Mathlib_Tactic_CasesM_0__Lean_MVarId_casesMatching_go_spec__1(v_recursive_boxed_953_, v_matcher_942_, v_allowSplit_boxed_954_, v_acc_944_, v_g_945_, v_t_946_, v_init_947_, v___y_948_, v___y_949_, v___y_950_, v___y_951_);
lean_dec(v___y_951_);
lean_dec_ref(v___y_950_);
lean_dec(v___y_949_);
lean_dec_ref(v___y_948_);
lean_dec_ref(v_t_946_);
return v_res_955_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00__private_Mathlib_Tactic_CasesM_0__Lean_MVarId_casesMatching_go_spec__1_spec__1___boxed(lean_object* v_init_956_, lean_object* v_recursive_957_, lean_object* v_matcher_958_, lean_object* v_allowSplit_959_, lean_object* v_acc_960_, lean_object* v_g_961_, lean_object* v_n_962_, lean_object* v_b_963_, lean_object* v___y_964_, lean_object* v___y_965_, lean_object* v___y_966_, lean_object* v___y_967_, lean_object* v___y_968_){
_start:
{
uint8_t v_recursive_boxed_969_; uint8_t v_allowSplit_boxed_970_; lean_object* v_res_971_; 
v_recursive_boxed_969_ = lean_unbox(v_recursive_957_);
v_allowSplit_boxed_970_ = lean_unbox(v_allowSplit_959_);
v_res_971_ = lp_mathlib_Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00__private_Mathlib_Tactic_CasesM_0__Lean_MVarId_casesMatching_go_spec__1_spec__1(v_init_956_, v_recursive_boxed_969_, v_matcher_958_, v_allowSplit_boxed_970_, v_acc_960_, v_g_961_, v_n_962_, v_b_963_, v___y_964_, v___y_965_, v___y_966_, v___y_967_);
lean_dec(v___y_967_);
lean_dec_ref(v___y_966_);
lean_dec(v___y_965_);
lean_dec_ref(v___y_964_);
lean_dec_ref(v_n_962_);
lean_dec_ref(v_init_956_);
return v_res_971_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_CasesM_0__Lean_MVarId_casesMatching_go_spec__0___boxed(lean_object* v_recursive_972_, lean_object* v_matcher_973_, lean_object* v_allowSplit_974_, lean_object* v_val_975_, lean_object* v_as_976_, lean_object* v_sz_977_, lean_object* v_i_978_, lean_object* v_b_979_, lean_object* v___y_980_, lean_object* v___y_981_, lean_object* v___y_982_, lean_object* v___y_983_, lean_object* v___y_984_){
_start:
{
uint8_t v_recursive_boxed_985_; uint8_t v_allowSplit_boxed_986_; size_t v_sz_boxed_987_; size_t v_i_boxed_988_; lean_object* v_res_989_; 
v_recursive_boxed_985_ = lean_unbox(v_recursive_972_);
v_allowSplit_boxed_986_ = lean_unbox(v_allowSplit_974_);
v_sz_boxed_987_ = lean_unbox_usize(v_sz_977_);
lean_dec(v_sz_977_);
v_i_boxed_988_ = lean_unbox_usize(v_i_978_);
lean_dec(v_i_978_);
v_res_989_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_CasesM_0__Lean_MVarId_casesMatching_go_spec__0(v_recursive_boxed_985_, v_matcher_973_, v_allowSplit_boxed_986_, v_val_975_, v_as_976_, v_sz_boxed_987_, v_i_boxed_988_, v_b_979_, v___y_980_, v___y_981_, v___y_982_, v___y_983_);
lean_dec(v___y_983_);
lean_dec_ref(v___y_982_);
lean_dec(v___y_981_);
lean_dec_ref(v___y_980_);
lean_dec_ref(v_as_976_);
lean_dec_ref(v_val_975_);
return v_res_989_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00__private_Mathlib_Tactic_CasesM_0__Lean_MVarId_casesMatching_go_spec__1_spec__2___boxed(lean_object* v_recursive_990_, lean_object* v_matcher_991_, lean_object* v_allowSplit_992_, lean_object* v_acc_993_, lean_object* v_g_994_, lean_object* v_as_995_, lean_object* v_sz_996_, lean_object* v_i_997_, lean_object* v_b_998_, lean_object* v___y_999_, lean_object* v___y_1000_, lean_object* v___y_1001_, lean_object* v___y_1002_, lean_object* v___y_1003_){
_start:
{
uint8_t v_recursive_boxed_1004_; uint8_t v_allowSplit_boxed_1005_; size_t v_sz_boxed_1006_; size_t v_i_boxed_1007_; lean_object* v_res_1008_; 
v_recursive_boxed_1004_ = lean_unbox(v_recursive_990_);
v_allowSplit_boxed_1005_ = lean_unbox(v_allowSplit_992_);
v_sz_boxed_1006_ = lean_unbox_usize(v_sz_996_);
lean_dec(v_sz_996_);
v_i_boxed_1007_ = lean_unbox_usize(v_i_997_);
lean_dec(v_i_997_);
v_res_1008_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00__private_Mathlib_Tactic_CasesM_0__Lean_MVarId_casesMatching_go_spec__1_spec__2(v_recursive_boxed_1004_, v_matcher_991_, v_allowSplit_boxed_1005_, v_acc_993_, v_g_994_, v_as_995_, v_sz_boxed_1006_, v_i_boxed_1007_, v_b_998_, v___y_999_, v___y_1000_, v___y_1001_, v___y_1002_);
lean_dec(v___y_1002_);
lean_dec_ref(v___y_1001_);
lean_dec(v___y_1000_);
lean_dec_ref(v___y_999_);
lean_dec_ref(v_as_995_);
return v_res_1008_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00__private_Mathlib_Tactic_CasesM_0__Lean_MVarId_casesMatching_go_spec__1_spec__2_spec__6___boxed(lean_object* v_recursive_1009_, lean_object* v_matcher_1010_, lean_object* v_allowSplit_1011_, lean_object* v_acc_1012_, lean_object* v_g_1013_, lean_object* v_as_1014_, lean_object* v_sz_1015_, lean_object* v_i_1016_, lean_object* v_b_1017_, lean_object* v___y_1018_, lean_object* v___y_1019_, lean_object* v___y_1020_, lean_object* v___y_1021_, lean_object* v___y_1022_){
_start:
{
uint8_t v_recursive_boxed_1023_; uint8_t v_allowSplit_boxed_1024_; size_t v_sz_boxed_1025_; size_t v_i_boxed_1026_; lean_object* v_res_1027_; 
v_recursive_boxed_1023_ = lean_unbox(v_recursive_1009_);
v_allowSplit_boxed_1024_ = lean_unbox(v_allowSplit_1011_);
v_sz_boxed_1025_ = lean_unbox_usize(v_sz_1015_);
lean_dec(v_sz_1015_);
v_i_boxed_1026_ = lean_unbox_usize(v_i_1016_);
lean_dec(v_i_1016_);
v_res_1027_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00__private_Mathlib_Tactic_CasesM_0__Lean_MVarId_casesMatching_go_spec__1_spec__2_spec__6(v_recursive_boxed_1023_, v_matcher_1010_, v_allowSplit_boxed_1024_, v_acc_1012_, v_g_1013_, v_as_1014_, v_sz_boxed_1025_, v_i_boxed_1026_, v_b_1017_, v___y_1018_, v___y_1019_, v___y_1020_, v___y_1021_);
lean_dec(v___y_1021_);
lean_dec_ref(v___y_1020_);
lean_dec(v___y_1019_);
lean_dec_ref(v___y_1018_);
lean_dec_ref(v_as_1014_);
return v_res_1027_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00__private_Mathlib_Tactic_CasesM_0__Lean_MVarId_casesMatching_go_spec__1_spec__1_spec__4___boxed(lean_object* v_recursive_1028_, lean_object* v_matcher_1029_, lean_object* v_allowSplit_1030_, lean_object* v_acc_1031_, lean_object* v_g_1032_, lean_object* v_as_1033_, lean_object* v_sz_1034_, lean_object* v_i_1035_, lean_object* v_b_1036_, lean_object* v___y_1037_, lean_object* v___y_1038_, lean_object* v___y_1039_, lean_object* v___y_1040_, lean_object* v___y_1041_){
_start:
{
uint8_t v_recursive_boxed_1042_; uint8_t v_allowSplit_boxed_1043_; size_t v_sz_boxed_1044_; size_t v_i_boxed_1045_; lean_object* v_res_1046_; 
v_recursive_boxed_1042_ = lean_unbox(v_recursive_1028_);
v_allowSplit_boxed_1043_ = lean_unbox(v_allowSplit_1030_);
v_sz_boxed_1044_ = lean_unbox_usize(v_sz_1034_);
lean_dec(v_sz_1034_);
v_i_boxed_1045_ = lean_unbox_usize(v_i_1035_);
lean_dec(v_i_1035_);
v_res_1046_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00__private_Mathlib_Tactic_CasesM_0__Lean_MVarId_casesMatching_go_spec__1_spec__1_spec__4(v_recursive_boxed_1042_, v_matcher_1029_, v_allowSplit_boxed_1043_, v_acc_1031_, v_g_1032_, v_as_1033_, v_sz_boxed_1044_, v_i_boxed_1045_, v_b_1036_, v___y_1037_, v___y_1038_, v___y_1039_, v___y_1040_);
lean_dec(v___y_1040_);
lean_dec_ref(v___y_1039_);
lean_dec(v___y_1038_);
lean_dec_ref(v___y_1037_);
lean_dec_ref(v_as_1033_);
return v_res_1046_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00__private_Mathlib_Tactic_CasesM_0__Lean_MVarId_casesMatching_go_spec__1_spec__1_spec__4_spec__5___boxed(lean_object* v_recursive_1047_, lean_object* v_matcher_1048_, lean_object* v_allowSplit_1049_, lean_object* v_acc_1050_, lean_object* v_g_1051_, lean_object* v_as_1052_, lean_object* v_sz_1053_, lean_object* v_i_1054_, lean_object* v_b_1055_, lean_object* v___y_1056_, lean_object* v___y_1057_, lean_object* v___y_1058_, lean_object* v___y_1059_, lean_object* v___y_1060_){
_start:
{
uint8_t v_recursive_boxed_1061_; uint8_t v_allowSplit_boxed_1062_; size_t v_sz_boxed_1063_; size_t v_i_boxed_1064_; lean_object* v_res_1065_; 
v_recursive_boxed_1061_ = lean_unbox(v_recursive_1047_);
v_allowSplit_boxed_1062_ = lean_unbox(v_allowSplit_1049_);
v_sz_boxed_1063_ = lean_unbox_usize(v_sz_1053_);
lean_dec(v_sz_1053_);
v_i_boxed_1064_ = lean_unbox_usize(v_i_1054_);
lean_dec(v_i_1054_);
v_res_1065_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00__private_Mathlib_Tactic_CasesM_0__Lean_MVarId_casesMatching_go_spec__1_spec__1_spec__4_spec__5(v_recursive_boxed_1061_, v_matcher_1048_, v_allowSplit_boxed_1062_, v_acc_1050_, v_g_1051_, v_as_1052_, v_sz_boxed_1063_, v_i_boxed_1064_, v_b_1055_, v___y_1056_, v___y_1057_, v___y_1058_, v___y_1059_);
lean_dec(v___y_1059_);
lean_dec_ref(v___y_1058_);
lean_dec(v___y_1057_);
lean_dec_ref(v___y_1056_);
lean_dec_ref(v_as_1052_);
return v_res_1065_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextFull___at___00Lean_throwError___at___00Lean_MVarId_casesMatching_spec__1_spec__1(lean_object* v_msgData_1066_, lean_object* v___y_1067_, lean_object* v___y_1068_, lean_object* v___y_1069_, lean_object* v___y_1070_){
_start:
{
lean_object* v___x_1072_; lean_object* v_env_1073_; lean_object* v___x_1074_; lean_object* v_mctx_1075_; lean_object* v_lctx_1076_; lean_object* v_options_1077_; lean_object* v___x_1078_; lean_object* v___x_1079_; lean_object* v___x_1080_; 
v___x_1072_ = lean_st_ref_get(v___y_1070_);
v_env_1073_ = lean_ctor_get(v___x_1072_, 0);
lean_inc_ref(v_env_1073_);
lean_dec(v___x_1072_);
v___x_1074_ = lean_st_ref_get(v___y_1068_);
v_mctx_1075_ = lean_ctor_get(v___x_1074_, 0);
lean_inc_ref(v_mctx_1075_);
lean_dec(v___x_1074_);
v_lctx_1076_ = lean_ctor_get(v___y_1067_, 2);
v_options_1077_ = lean_ctor_get(v___y_1069_, 2);
lean_inc_ref(v_options_1077_);
lean_inc_ref(v_lctx_1076_);
v___x_1078_ = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(v___x_1078_, 0, v_env_1073_);
lean_ctor_set(v___x_1078_, 1, v_mctx_1075_);
lean_ctor_set(v___x_1078_, 2, v_lctx_1076_);
lean_ctor_set(v___x_1078_, 3, v_options_1077_);
v___x_1079_ = lean_alloc_ctor(3, 2, 0);
lean_ctor_set(v___x_1079_, 0, v___x_1078_);
lean_ctor_set(v___x_1079_, 1, v_msgData_1066_);
v___x_1080_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1080_, 0, v___x_1079_);
return v___x_1080_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextFull___at___00Lean_throwError___at___00Lean_MVarId_casesMatching_spec__1_spec__1___boxed(lean_object* v_msgData_1081_, lean_object* v___y_1082_, lean_object* v___y_1083_, lean_object* v___y_1084_, lean_object* v___y_1085_, lean_object* v___y_1086_){
_start:
{
lean_object* v_res_1087_; 
v_res_1087_ = lp_mathlib_Lean_addMessageContextFull___at___00Lean_throwError___at___00Lean_MVarId_casesMatching_spec__1_spec__1(v_msgData_1081_, v___y_1082_, v___y_1083_, v___y_1084_, v___y_1085_);
lean_dec(v___y_1085_);
lean_dec_ref(v___y_1084_);
lean_dec(v___y_1083_);
lean_dec_ref(v___y_1082_);
return v_res_1087_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Lean_MVarId_casesMatching_spec__1___redArg(lean_object* v_msg_1088_, lean_object* v___y_1089_, lean_object* v___y_1090_, lean_object* v___y_1091_, lean_object* v___y_1092_){
_start:
{
lean_object* v_ref_1094_; lean_object* v___x_1095_; lean_object* v_a_1096_; lean_object* v___x_1098_; uint8_t v_isShared_1099_; uint8_t v_isSharedCheck_1104_; 
v_ref_1094_ = lean_ctor_get(v___y_1091_, 5);
v___x_1095_ = lp_mathlib_Lean_addMessageContextFull___at___00Lean_throwError___at___00Lean_MVarId_casesMatching_spec__1_spec__1(v_msg_1088_, v___y_1089_, v___y_1090_, v___y_1091_, v___y_1092_);
v_a_1096_ = lean_ctor_get(v___x_1095_, 0);
v_isSharedCheck_1104_ = !lean_is_exclusive(v___x_1095_);
if (v_isSharedCheck_1104_ == 0)
{
v___x_1098_ = v___x_1095_;
v_isShared_1099_ = v_isSharedCheck_1104_;
goto v_resetjp_1097_;
}
else
{
lean_inc(v_a_1096_);
lean_dec(v___x_1095_);
v___x_1098_ = lean_box(0);
v_isShared_1099_ = v_isSharedCheck_1104_;
goto v_resetjp_1097_;
}
v_resetjp_1097_:
{
lean_object* v___x_1100_; lean_object* v___x_1102_; 
lean_inc(v_ref_1094_);
v___x_1100_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1100_, 0, v_ref_1094_);
lean_ctor_set(v___x_1100_, 1, v_a_1096_);
if (v_isShared_1099_ == 0)
{
lean_ctor_set_tag(v___x_1098_, 1);
lean_ctor_set(v___x_1098_, 0, v___x_1100_);
v___x_1102_ = v___x_1098_;
goto v_reusejp_1101_;
}
else
{
lean_object* v_reuseFailAlloc_1103_; 
v_reuseFailAlloc_1103_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1103_, 0, v___x_1100_);
v___x_1102_ = v_reuseFailAlloc_1103_;
goto v_reusejp_1101_;
}
v_reusejp_1101_:
{
return v___x_1102_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Lean_MVarId_casesMatching_spec__1___redArg___boxed(lean_object* v_msg_1105_, lean_object* v___y_1106_, lean_object* v___y_1107_, lean_object* v___y_1108_, lean_object* v___y_1109_, lean_object* v___y_1110_){
_start:
{
lean_object* v_res_1111_; 
v_res_1111_ = lp_mathlib_Lean_throwError___at___00Lean_MVarId_casesMatching_spec__1___redArg(v_msg_1105_, v___y_1106_, v___y_1107_, v___y_1108_, v___y_1109_);
lean_dec(v___y_1109_);
lean_dec_ref(v___y_1108_);
lean_dec(v___y_1107_);
lean_dec_ref(v___y_1106_);
return v_res_1111_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_List_beq___at___00Lean_MVarId_casesMatching_spec__0(lean_object* v_x_1112_, lean_object* v_x_1113_){
_start:
{
if (lean_obj_tag(v_x_1112_) == 0)
{
if (lean_obj_tag(v_x_1113_) == 0)
{
uint8_t v___x_1114_; 
v___x_1114_ = 1;
return v___x_1114_;
}
else
{
uint8_t v___x_1115_; 
v___x_1115_ = 0;
return v___x_1115_;
}
}
else
{
if (lean_obj_tag(v_x_1113_) == 0)
{
uint8_t v___x_1116_; 
v___x_1116_ = 0;
return v___x_1116_;
}
else
{
lean_object* v_head_1117_; lean_object* v_tail_1118_; lean_object* v_head_1119_; lean_object* v_tail_1120_; uint8_t v___x_1121_; 
v_head_1117_ = lean_ctor_get(v_x_1112_, 0);
v_tail_1118_ = lean_ctor_get(v_x_1112_, 1);
v_head_1119_ = lean_ctor_get(v_x_1113_, 0);
v_tail_1120_ = lean_ctor_get(v_x_1113_, 1);
v___x_1121_ = l_Lean_instBEqMVarId_beq(v_head_1117_, v_head_1119_);
if (v___x_1121_ == 0)
{
return v___x_1121_;
}
else
{
v_x_1112_ = v_tail_1118_;
v_x_1113_ = v_tail_1120_;
goto _start;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_beq___at___00Lean_MVarId_casesMatching_spec__0___boxed(lean_object* v_x_1123_, lean_object* v_x_1124_){
_start:
{
uint8_t v_res_1125_; lean_object* v_r_1126_; 
v_res_1125_ = lp_mathlib_List_beq___at___00Lean_MVarId_casesMatching_spec__0(v_x_1123_, v_x_1124_);
lean_dec(v_x_1124_);
lean_dec(v_x_1123_);
v_r_1126_ = lean_box(v_res_1125_);
return v_r_1126_;
}
}
static lean_object* _init_lp_mathlib_Lean_MVarId_casesMatching___closed__2(void){
_start:
{
lean_object* v___x_1130_; lean_object* v___x_1131_; 
v___x_1130_ = ((lean_object*)(lp_mathlib_Lean_MVarId_casesMatching___closed__1));
v___x_1131_ = l_Lean_stringToMessageData(v___x_1130_);
return v___x_1131_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_casesMatching(lean_object* v_matcher_1132_, uint8_t v_recursive_1133_, uint8_t v_allowSplit_1134_, uint8_t v_throwOnNoMatch_1135_, lean_object* v_g_1136_, lean_object* v_a_1137_, lean_object* v_a_1138_, lean_object* v_a_1139_, lean_object* v_a_1140_){
_start:
{
lean_object* v___x_1142_; lean_object* v___x_1143_; 
v___x_1142_ = ((lean_object*)(lp_mathlib_Lean_MVarId_casesMatching___closed__0));
lean_inc(v_g_1136_);
v___x_1143_ = lp_mathlib___private_Mathlib_Tactic_CasesM_0__Lean_MVarId_casesMatching_go(v_matcher_1132_, v_recursive_1133_, v_allowSplit_1134_, v_g_1136_, v___x_1142_, v_a_1137_, v_a_1138_, v_a_1139_, v_a_1140_);
if (lean_obj_tag(v___x_1143_) == 0)
{
lean_object* v_a_1144_; lean_object* v___x_1146_; uint8_t v_isShared_1147_; uint8_t v_isSharedCheck_1160_; 
v_a_1144_ = lean_ctor_get(v___x_1143_, 0);
v_isSharedCheck_1160_ = !lean_is_exclusive(v___x_1143_);
if (v_isSharedCheck_1160_ == 0)
{
v___x_1146_ = v___x_1143_;
v_isShared_1147_ = v_isSharedCheck_1160_;
goto v_resetjp_1145_;
}
else
{
lean_inc(v_a_1144_);
lean_dec(v___x_1143_);
v___x_1146_ = lean_box(0);
v_isShared_1147_ = v_isSharedCheck_1160_;
goto v_resetjp_1145_;
}
v_resetjp_1145_:
{
lean_object* v___x_1148_; 
v___x_1148_ = lean_array_to_list(v_a_1144_);
if (v_throwOnNoMatch_1135_ == 0)
{
lean_object* v___x_1150_; 
lean_dec(v_g_1136_);
if (v_isShared_1147_ == 0)
{
lean_ctor_set(v___x_1146_, 0, v___x_1148_);
v___x_1150_ = v___x_1146_;
goto v_reusejp_1149_;
}
else
{
lean_object* v_reuseFailAlloc_1151_; 
v_reuseFailAlloc_1151_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1151_, 0, v___x_1148_);
v___x_1150_ = v_reuseFailAlloc_1151_;
goto v_reusejp_1149_;
}
v_reusejp_1149_:
{
return v___x_1150_;
}
}
else
{
lean_object* v___x_1152_; lean_object* v___x_1153_; uint8_t v___x_1154_; 
v___x_1152_ = lean_box(0);
v___x_1153_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_1153_, 0, v_g_1136_);
lean_ctor_set(v___x_1153_, 1, v___x_1152_);
v___x_1154_ = lp_mathlib_List_beq___at___00Lean_MVarId_casesMatching_spec__0(v___x_1148_, v___x_1153_);
lean_dec_ref_known(v___x_1153_, 2);
if (v___x_1154_ == 0)
{
lean_object* v___x_1156_; 
if (v_isShared_1147_ == 0)
{
lean_ctor_set(v___x_1146_, 0, v___x_1148_);
v___x_1156_ = v___x_1146_;
goto v_reusejp_1155_;
}
else
{
lean_object* v_reuseFailAlloc_1157_; 
v_reuseFailAlloc_1157_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1157_, 0, v___x_1148_);
v___x_1156_ = v_reuseFailAlloc_1157_;
goto v_reusejp_1155_;
}
v_reusejp_1155_:
{
return v___x_1156_;
}
}
else
{
lean_object* v___x_1158_; lean_object* v___x_1159_; 
lean_dec(v___x_1148_);
lean_del_object(v___x_1146_);
v___x_1158_ = lean_obj_once(&lp_mathlib_Lean_MVarId_casesMatching___closed__2, &lp_mathlib_Lean_MVarId_casesMatching___closed__2_once, _init_lp_mathlib_Lean_MVarId_casesMatching___closed__2);
v___x_1159_ = lp_mathlib_Lean_throwError___at___00Lean_MVarId_casesMatching_spec__1___redArg(v___x_1158_, v_a_1137_, v_a_1138_, v_a_1139_, v_a_1140_);
return v___x_1159_;
}
}
}
}
else
{
lean_object* v_a_1161_; lean_object* v___x_1163_; uint8_t v_isShared_1164_; uint8_t v_isSharedCheck_1168_; 
lean_dec(v_g_1136_);
v_a_1161_ = lean_ctor_get(v___x_1143_, 0);
v_isSharedCheck_1168_ = !lean_is_exclusive(v___x_1143_);
if (v_isSharedCheck_1168_ == 0)
{
v___x_1163_ = v___x_1143_;
v_isShared_1164_ = v_isSharedCheck_1168_;
goto v_resetjp_1162_;
}
else
{
lean_inc(v_a_1161_);
lean_dec(v___x_1143_);
v___x_1163_ = lean_box(0);
v_isShared_1164_ = v_isSharedCheck_1168_;
goto v_resetjp_1162_;
}
v_resetjp_1162_:
{
lean_object* v___x_1166_; 
if (v_isShared_1164_ == 0)
{
v___x_1166_ = v___x_1163_;
goto v_reusejp_1165_;
}
else
{
lean_object* v_reuseFailAlloc_1167_; 
v_reuseFailAlloc_1167_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1167_, 0, v_a_1161_);
v___x_1166_ = v_reuseFailAlloc_1167_;
goto v_reusejp_1165_;
}
v_reusejp_1165_:
{
return v___x_1166_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_casesMatching___boxed(lean_object* v_matcher_1169_, lean_object* v_recursive_1170_, lean_object* v_allowSplit_1171_, lean_object* v_throwOnNoMatch_1172_, lean_object* v_g_1173_, lean_object* v_a_1174_, lean_object* v_a_1175_, lean_object* v_a_1176_, lean_object* v_a_1177_, lean_object* v_a_1178_){
_start:
{
uint8_t v_recursive_boxed_1179_; uint8_t v_allowSplit_boxed_1180_; uint8_t v_throwOnNoMatch_boxed_1181_; lean_object* v_res_1182_; 
v_recursive_boxed_1179_ = lean_unbox(v_recursive_1170_);
v_allowSplit_boxed_1180_ = lean_unbox(v_allowSplit_1171_);
v_throwOnNoMatch_boxed_1181_ = lean_unbox(v_throwOnNoMatch_1172_);
v_res_1182_ = lp_mathlib_Lean_MVarId_casesMatching(v_matcher_1169_, v_recursive_boxed_1179_, v_allowSplit_boxed_1180_, v_throwOnNoMatch_boxed_1181_, v_g_1173_, v_a_1174_, v_a_1175_, v_a_1176_, v_a_1177_);
lean_dec(v_a_1177_);
lean_dec_ref(v_a_1176_);
lean_dec(v_a_1175_);
lean_dec_ref(v_a_1174_);
return v_res_1182_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Lean_MVarId_casesMatching_spec__1(lean_object* v_00_u03b1_1183_, lean_object* v_msg_1184_, lean_object* v___y_1185_, lean_object* v___y_1186_, lean_object* v___y_1187_, lean_object* v___y_1188_){
_start:
{
lean_object* v___x_1190_; 
v___x_1190_ = lp_mathlib_Lean_throwError___at___00Lean_MVarId_casesMatching_spec__1___redArg(v_msg_1184_, v___y_1185_, v___y_1186_, v___y_1187_, v___y_1188_);
return v___x_1190_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Lean_MVarId_casesMatching_spec__1___boxed(lean_object* v_00_u03b1_1191_, lean_object* v_msg_1192_, lean_object* v___y_1193_, lean_object* v___y_1194_, lean_object* v___y_1195_, lean_object* v___y_1196_, lean_object* v___y_1197_){
_start:
{
lean_object* v_res_1198_; 
v_res_1198_ = lp_mathlib_Lean_throwError___at___00Lean_MVarId_casesMatching_spec__1(v_00_u03b1_1191_, v_msg_1192_, v___y_1193_, v___y_1194_, v___y_1195_, v___y_1196_);
lean_dec(v___y_1196_);
lean_dec_ref(v___y_1195_);
lean_dec(v___y_1194_);
lean_dec_ref(v___y_1193_);
return v_res_1198_;
}
}
LEAN_EXPORT uint8_t lp_mathlib___private_Init_Data_Array_Basic_0__Array_anyMUnsafe_any___at___00Array_contains___at___00Lean_MVarId_casesType_spec__0_spec__0(lean_object* v_a_1199_, lean_object* v_as_1200_, size_t v_i_1201_, size_t v_stop_1202_){
_start:
{
uint8_t v___x_1203_; 
v___x_1203_ = lean_usize_dec_eq(v_i_1201_, v_stop_1202_);
if (v___x_1203_ == 0)
{
lean_object* v___x_1204_; uint8_t v___x_1205_; 
v___x_1204_ = lean_array_uget_borrowed(v_as_1200_, v_i_1201_);
v___x_1205_ = lean_name_eq(v_a_1199_, v___x_1204_);
if (v___x_1205_ == 0)
{
size_t v___x_1206_; size_t v___x_1207_; 
v___x_1206_ = ((size_t)1ULL);
v___x_1207_ = lean_usize_add(v_i_1201_, v___x_1206_);
v_i_1201_ = v___x_1207_;
goto _start;
}
else
{
return v___x_1205_;
}
}
else
{
uint8_t v___x_1209_; 
v___x_1209_ = 0;
return v___x_1209_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_anyMUnsafe_any___at___00Array_contains___at___00Lean_MVarId_casesType_spec__0_spec__0___boxed(lean_object* v_a_1210_, lean_object* v_as_1211_, lean_object* v_i_1212_, lean_object* v_stop_1213_){
_start:
{
size_t v_i_boxed_1214_; size_t v_stop_boxed_1215_; uint8_t v_res_1216_; lean_object* v_r_1217_; 
v_i_boxed_1214_ = lean_unbox_usize(v_i_1212_);
lean_dec(v_i_1212_);
v_stop_boxed_1215_ = lean_unbox_usize(v_stop_1213_);
lean_dec(v_stop_1213_);
v_res_1216_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_anyMUnsafe_any___at___00Array_contains___at___00Lean_MVarId_casesType_spec__0_spec__0(v_a_1210_, v_as_1211_, v_i_boxed_1214_, v_stop_boxed_1215_);
lean_dec_ref(v_as_1211_);
lean_dec(v_a_1210_);
v_r_1217_ = lean_box(v_res_1216_);
return v_r_1217_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Array_contains___at___00Lean_MVarId_casesType_spec__0(lean_object* v_as_1218_, lean_object* v_a_1219_){
_start:
{
lean_object* v___x_1220_; lean_object* v___x_1221_; uint8_t v___x_1222_; 
v___x_1220_ = lean_unsigned_to_nat(0u);
v___x_1221_ = lean_array_get_size(v_as_1218_);
v___x_1222_ = lean_nat_dec_lt(v___x_1220_, v___x_1221_);
if (v___x_1222_ == 0)
{
return v___x_1222_;
}
else
{
if (v___x_1222_ == 0)
{
return v___x_1222_;
}
else
{
size_t v___x_1223_; size_t v___x_1224_; uint8_t v___x_1225_; 
v___x_1223_ = ((size_t)0ULL);
v___x_1224_ = lean_usize_of_nat(v___x_1221_);
v___x_1225_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_anyMUnsafe_any___at___00Array_contains___at___00Lean_MVarId_casesType_spec__0_spec__0(v_a_1219_, v_as_1218_, v___x_1223_, v___x_1224_);
return v___x_1225_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Array_contains___at___00Lean_MVarId_casesType_spec__0___boxed(lean_object* v_as_1226_, lean_object* v_a_1227_){
_start:
{
uint8_t v_res_1228_; lean_object* v_r_1229_; 
v_res_1228_ = lp_mathlib_Array_contains___at___00Lean_MVarId_casesType_spec__0(v_as_1226_, v_a_1227_);
lean_dec(v_a_1227_);
lean_dec_ref(v_as_1226_);
v_r_1229_ = lean_box(v_res_1228_);
return v_r_1229_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_casesType___lam__0(lean_object* v_heads_1230_, lean_object* v_ty_1231_, lean_object* v___y_1232_, lean_object* v___y_1233_, lean_object* v___y_1234_, lean_object* v___y_1235_){
_start:
{
lean_object* v___x_1237_; lean_object* v___x_1238_; 
v___x_1237_ = l_Lean_Expr_headBeta(v_ty_1231_);
v___x_1238_ = l_Lean_Expr_getAppFn(v___x_1237_);
lean_dec_ref(v___x_1237_);
if (lean_obj_tag(v___x_1238_) == 4)
{
lean_object* v_declName_1239_; uint8_t v___x_1240_; lean_object* v___x_1241_; lean_object* v___x_1242_; 
v_declName_1239_ = lean_ctor_get(v___x_1238_, 0);
lean_inc(v_declName_1239_);
lean_dec_ref_known(v___x_1238_, 2);
v___x_1240_ = lp_mathlib_Array_contains___at___00Lean_MVarId_casesType_spec__0(v_heads_1230_, v_declName_1239_);
lean_dec(v_declName_1239_);
v___x_1241_ = lean_box(v___x_1240_);
v___x_1242_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1242_, 0, v___x_1241_);
return v___x_1242_;
}
else
{
uint8_t v___x_1243_; lean_object* v___x_1244_; lean_object* v___x_1245_; 
lean_dec_ref(v___x_1238_);
v___x_1243_ = 0;
v___x_1244_ = lean_box(v___x_1243_);
v___x_1245_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1245_, 0, v___x_1244_);
return v___x_1245_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_casesType___lam__0___boxed(lean_object* v_heads_1246_, lean_object* v_ty_1247_, lean_object* v___y_1248_, lean_object* v___y_1249_, lean_object* v___y_1250_, lean_object* v___y_1251_, lean_object* v___y_1252_){
_start:
{
lean_object* v_res_1253_; 
v_res_1253_ = lp_mathlib_Lean_MVarId_casesType___lam__0(v_heads_1246_, v_ty_1247_, v___y_1248_, v___y_1249_, v___y_1250_, v___y_1251_);
lean_dec(v___y_1251_);
lean_dec_ref(v___y_1250_);
lean_dec(v___y_1249_);
lean_dec_ref(v___y_1248_);
lean_dec_ref(v_heads_1246_);
return v_res_1253_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_casesType(lean_object* v_heads_1254_, uint8_t v_recursive_1255_, uint8_t v_allowSplit_1256_, lean_object* v_g_1257_, lean_object* v_a_1258_, lean_object* v_a_1259_, lean_object* v_a_1260_, lean_object* v_a_1261_){
_start:
{
lean_object* v_matcher_1263_; uint8_t v___x_1264_; lean_object* v___x_1265_; 
v_matcher_1263_ = lean_alloc_closure((void*)(lp_mathlib_Lean_MVarId_casesType___lam__0___boxed), 7, 1);
lean_closure_set(v_matcher_1263_, 0, v_heads_1254_);
v___x_1264_ = 1;
v___x_1265_ = lp_mathlib_Lean_MVarId_casesMatching(v_matcher_1263_, v_recursive_1255_, v_allowSplit_1256_, v___x_1264_, v_g_1257_, v_a_1258_, v_a_1259_, v_a_1260_, v_a_1261_);
return v___x_1265_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_casesType___boxed(lean_object* v_heads_1266_, lean_object* v_recursive_1267_, lean_object* v_allowSplit_1268_, lean_object* v_g_1269_, lean_object* v_a_1270_, lean_object* v_a_1271_, lean_object* v_a_1272_, lean_object* v_a_1273_, lean_object* v_a_1274_){
_start:
{
uint8_t v_recursive_boxed_1275_; uint8_t v_allowSplit_boxed_1276_; lean_object* v_res_1277_; 
v_recursive_boxed_1275_ = lean_unbox(v_recursive_1267_);
v_allowSplit_boxed_1276_ = lean_unbox(v_allowSplit_1268_);
v_res_1277_ = lp_mathlib_Lean_MVarId_casesType(v_heads_1266_, v_recursive_boxed_1275_, v_allowSplit_boxed_1276_, v_g_1269_, v_a_1270_, v_a_1271_, v_a_1272_, v_a_1273_);
lean_dec(v_a_1273_);
lean_dec_ref(v_a_1272_);
lean_dec(v_a_1271_);
lean_dec_ref(v_a_1270_);
return v_res_1277_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_Term_withoutErrToSorry___at___00Mathlib_Tactic_elabPatterns_spec__1___redArg(lean_object* v_a_1278_, lean_object* v___y_1279_, lean_object* v___y_1280_, lean_object* v___y_1281_, lean_object* v___y_1282_, lean_object* v___y_1283_, lean_object* v___y_1284_){
_start:
{
lean_object* v___x_1286_; 
v___x_1286_ = l_Lean_Elab_Term_withoutErrToSorryImp___redArg(v_a_1278_, v___y_1279_, v___y_1280_, v___y_1281_, v___y_1282_, v___y_1283_, v___y_1284_);
return v___x_1286_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_Term_withoutErrToSorry___at___00Mathlib_Tactic_elabPatterns_spec__1___redArg___boxed(lean_object* v_a_1287_, lean_object* v___y_1288_, lean_object* v___y_1289_, lean_object* v___y_1290_, lean_object* v___y_1291_, lean_object* v___y_1292_, lean_object* v___y_1293_, lean_object* v___y_1294_){
_start:
{
lean_object* v_res_1295_; 
v_res_1295_ = lp_mathlib_Lean_Elab_Term_withoutErrToSorry___at___00Mathlib_Tactic_elabPatterns_spec__1___redArg(v_a_1287_, v___y_1288_, v___y_1289_, v___y_1290_, v___y_1291_, v___y_1292_, v___y_1293_);
lean_dec(v___y_1293_);
lean_dec_ref(v___y_1292_);
lean_dec(v___y_1291_);
lean_dec_ref(v___y_1290_);
lean_dec(v___y_1289_);
lean_dec_ref(v___y_1288_);
return v_res_1295_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_Term_withoutErrToSorry___at___00Mathlib_Tactic_elabPatterns_spec__1(lean_object* v_00_u03b1_1296_, lean_object* v_a_1297_, lean_object* v___y_1298_, lean_object* v___y_1299_, lean_object* v___y_1300_, lean_object* v___y_1301_, lean_object* v___y_1302_, lean_object* v___y_1303_){
_start:
{
lean_object* v___x_1305_; 
v___x_1305_ = l_Lean_Elab_Term_withoutErrToSorryImp___redArg(v_a_1297_, v___y_1298_, v___y_1299_, v___y_1300_, v___y_1301_, v___y_1302_, v___y_1303_);
return v___x_1305_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_Term_withoutErrToSorry___at___00Mathlib_Tactic_elabPatterns_spec__1___boxed(lean_object* v_00_u03b1_1306_, lean_object* v_a_1307_, lean_object* v___y_1308_, lean_object* v___y_1309_, lean_object* v___y_1310_, lean_object* v___y_1311_, lean_object* v___y_1312_, lean_object* v___y_1313_, lean_object* v___y_1314_){
_start:
{
lean_object* v_res_1315_; 
v_res_1315_ = lp_mathlib_Lean_Elab_Term_withoutErrToSorry___at___00Mathlib_Tactic_elabPatterns_spec__1(v_00_u03b1_1306_, v_a_1307_, v___y_1308_, v___y_1309_, v___y_1310_, v___y_1311_, v___y_1312_, v___y_1313_);
lean_dec(v___y_1313_);
lean_dec_ref(v___y_1312_);
lean_dec(v___y_1311_);
lean_dec_ref(v___y_1310_);
lean_dec(v___y_1309_);
lean_dec_ref(v___y_1308_);
return v_res_1315_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Mathlib_Tactic_elabPatterns_spec__0___lam__0(lean_object* v_v_1316_, lean_object* v___x_1317_, uint8_t v___x_1318_, lean_object* v___y_1319_, lean_object* v___y_1320_, lean_object* v___y_1321_, lean_object* v___y_1322_, lean_object* v___y_1323_, lean_object* v___y_1324_){
_start:
{
lean_object* v___x_1326_; 
lean_inc(v_v_1316_);
v___x_1326_ = l_Lean_Elab_Term_elabTerm(v_v_1316_, v___x_1317_, v___x_1318_, v___x_1318_, v___y_1319_, v___y_1320_, v___y_1321_, v___y_1322_, v___y_1323_, v___y_1324_);
if (lean_obj_tag(v___x_1326_) == 0)
{
lean_object* v_a_1327_; lean_object* v_fileName_1328_; lean_object* v_fileMap_1329_; lean_object* v_options_1330_; lean_object* v_currRecDepth_1331_; lean_object* v_maxRecDepth_1332_; lean_object* v_ref_1333_; lean_object* v_currNamespace_1334_; lean_object* v_openDecls_1335_; lean_object* v_initHeartbeats_1336_; lean_object* v_maxHeartbeats_1337_; lean_object* v_quotContext_1338_; lean_object* v_currMacroScope_1339_; uint8_t v_diag_1340_; lean_object* v_cancelTk_x3f_1341_; uint8_t v_suppressElabErrors_1342_; lean_object* v_inheritedTraceOptions_1343_; lean_object* v___x_1345_; uint8_t v_isShared_1346_; uint8_t v_isSharedCheck_1352_; 
v_a_1327_ = lean_ctor_get(v___x_1326_, 0);
lean_inc(v_a_1327_);
lean_dec_ref_known(v___x_1326_, 1);
v_fileName_1328_ = lean_ctor_get(v___y_1323_, 0);
v_fileMap_1329_ = lean_ctor_get(v___y_1323_, 1);
v_options_1330_ = lean_ctor_get(v___y_1323_, 2);
v_currRecDepth_1331_ = lean_ctor_get(v___y_1323_, 3);
v_maxRecDepth_1332_ = lean_ctor_get(v___y_1323_, 4);
v_ref_1333_ = lean_ctor_get(v___y_1323_, 5);
v_currNamespace_1334_ = lean_ctor_get(v___y_1323_, 6);
v_openDecls_1335_ = lean_ctor_get(v___y_1323_, 7);
v_initHeartbeats_1336_ = lean_ctor_get(v___y_1323_, 8);
v_maxHeartbeats_1337_ = lean_ctor_get(v___y_1323_, 9);
v_quotContext_1338_ = lean_ctor_get(v___y_1323_, 10);
v_currMacroScope_1339_ = lean_ctor_get(v___y_1323_, 11);
v_diag_1340_ = lean_ctor_get_uint8(v___y_1323_, sizeof(void*)*14);
v_cancelTk_x3f_1341_ = lean_ctor_get(v___y_1323_, 12);
v_suppressElabErrors_1342_ = lean_ctor_get_uint8(v___y_1323_, sizeof(void*)*14 + 1);
v_inheritedTraceOptions_1343_ = lean_ctor_get(v___y_1323_, 13);
v_isSharedCheck_1352_ = !lean_is_exclusive(v___y_1323_);
if (v_isSharedCheck_1352_ == 0)
{
v___x_1345_ = v___y_1323_;
v_isShared_1346_ = v_isSharedCheck_1352_;
goto v_resetjp_1344_;
}
else
{
lean_inc(v_inheritedTraceOptions_1343_);
lean_inc(v_cancelTk_x3f_1341_);
lean_inc(v_currMacroScope_1339_);
lean_inc(v_quotContext_1338_);
lean_inc(v_maxHeartbeats_1337_);
lean_inc(v_initHeartbeats_1336_);
lean_inc(v_openDecls_1335_);
lean_inc(v_currNamespace_1334_);
lean_inc(v_ref_1333_);
lean_inc(v_maxRecDepth_1332_);
lean_inc(v_currRecDepth_1331_);
lean_inc(v_options_1330_);
lean_inc(v_fileMap_1329_);
lean_inc(v_fileName_1328_);
lean_dec(v___y_1323_);
v___x_1345_ = lean_box(0);
v_isShared_1346_ = v_isSharedCheck_1352_;
goto v_resetjp_1344_;
}
v_resetjp_1344_:
{
lean_object* v_ref_1347_; lean_object* v___x_1349_; 
v_ref_1347_ = l_Lean_replaceRef(v_v_1316_, v_ref_1333_);
lean_dec(v_ref_1333_);
lean_dec(v_v_1316_);
if (v_isShared_1346_ == 0)
{
lean_ctor_set(v___x_1345_, 5, v_ref_1347_);
v___x_1349_ = v___x_1345_;
goto v_reusejp_1348_;
}
else
{
lean_object* v_reuseFailAlloc_1351_; 
v_reuseFailAlloc_1351_ = lean_alloc_ctor(0, 14, 2);
lean_ctor_set(v_reuseFailAlloc_1351_, 0, v_fileName_1328_);
lean_ctor_set(v_reuseFailAlloc_1351_, 1, v_fileMap_1329_);
lean_ctor_set(v_reuseFailAlloc_1351_, 2, v_options_1330_);
lean_ctor_set(v_reuseFailAlloc_1351_, 3, v_currRecDepth_1331_);
lean_ctor_set(v_reuseFailAlloc_1351_, 4, v_maxRecDepth_1332_);
lean_ctor_set(v_reuseFailAlloc_1351_, 5, v_ref_1347_);
lean_ctor_set(v_reuseFailAlloc_1351_, 6, v_currNamespace_1334_);
lean_ctor_set(v_reuseFailAlloc_1351_, 7, v_openDecls_1335_);
lean_ctor_set(v_reuseFailAlloc_1351_, 8, v_initHeartbeats_1336_);
lean_ctor_set(v_reuseFailAlloc_1351_, 9, v_maxHeartbeats_1337_);
lean_ctor_set(v_reuseFailAlloc_1351_, 10, v_quotContext_1338_);
lean_ctor_set(v_reuseFailAlloc_1351_, 11, v_currMacroScope_1339_);
lean_ctor_set(v_reuseFailAlloc_1351_, 12, v_cancelTk_x3f_1341_);
lean_ctor_set(v_reuseFailAlloc_1351_, 13, v_inheritedTraceOptions_1343_);
lean_ctor_set_uint8(v_reuseFailAlloc_1351_, sizeof(void*)*14, v_diag_1340_);
lean_ctor_set_uint8(v_reuseFailAlloc_1351_, sizeof(void*)*14 + 1, v_suppressElabErrors_1342_);
v___x_1349_ = v_reuseFailAlloc_1351_;
goto v_reusejp_1348_;
}
v_reusejp_1348_:
{
lean_object* v___x_1350_; 
v___x_1350_ = l_Lean_Meta_abstractMVars(v_a_1327_, v___x_1318_, v___y_1321_, v___y_1322_, v___x_1349_, v___y_1324_);
lean_dec_ref(v___x_1349_);
return v___x_1350_;
}
}
}
else
{
lean_object* v_a_1353_; lean_object* v___x_1355_; uint8_t v_isShared_1356_; uint8_t v_isSharedCheck_1360_; 
lean_dec_ref(v___y_1323_);
lean_dec(v_v_1316_);
v_a_1353_ = lean_ctor_get(v___x_1326_, 0);
v_isSharedCheck_1360_ = !lean_is_exclusive(v___x_1326_);
if (v_isSharedCheck_1360_ == 0)
{
v___x_1355_ = v___x_1326_;
v_isShared_1356_ = v_isSharedCheck_1360_;
goto v_resetjp_1354_;
}
else
{
lean_inc(v_a_1353_);
lean_dec(v___x_1326_);
v___x_1355_ = lean_box(0);
v_isShared_1356_ = v_isSharedCheck_1360_;
goto v_resetjp_1354_;
}
v_resetjp_1354_:
{
lean_object* v___x_1358_; 
if (v_isShared_1356_ == 0)
{
v___x_1358_ = v___x_1355_;
goto v_reusejp_1357_;
}
else
{
lean_object* v_reuseFailAlloc_1359_; 
v_reuseFailAlloc_1359_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1359_, 0, v_a_1353_);
v___x_1358_ = v_reuseFailAlloc_1359_;
goto v_reusejp_1357_;
}
v_reusejp_1357_:
{
return v___x_1358_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Mathlib_Tactic_elabPatterns_spec__0___lam__0___boxed(lean_object* v_v_1361_, lean_object* v___x_1362_, lean_object* v___x_1363_, lean_object* v___y_1364_, lean_object* v___y_1365_, lean_object* v___y_1366_, lean_object* v___y_1367_, lean_object* v___y_1368_, lean_object* v___y_1369_, lean_object* v___y_1370_){
_start:
{
uint8_t v___x_1720__boxed_1371_; lean_object* v_res_1372_; 
v___x_1720__boxed_1371_ = lean_unbox(v___x_1363_);
v_res_1372_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Mathlib_Tactic_elabPatterns_spec__0___lam__0(v_v_1361_, v___x_1362_, v___x_1720__boxed_1371_, v___y_1364_, v___y_1365_, v___y_1366_, v___y_1367_, v___y_1368_, v___y_1369_);
lean_dec(v___y_1369_);
lean_dec(v___y_1367_);
lean_dec_ref(v___y_1366_);
lean_dec(v___y_1365_);
lean_dec_ref(v___y_1364_);
return v_res_1372_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Mathlib_Tactic_elabPatterns_spec__0(size_t v_sz_1373_, size_t v_i_1374_, lean_object* v_bs_1375_, lean_object* v___y_1376_, lean_object* v___y_1377_, lean_object* v___y_1378_, lean_object* v___y_1379_, lean_object* v___y_1380_, lean_object* v___y_1381_){
_start:
{
uint8_t v___x_1383_; 
v___x_1383_ = lean_usize_dec_lt(v_i_1374_, v_sz_1373_);
if (v___x_1383_ == 0)
{
lean_object* v___x_1384_; 
v___x_1384_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1384_, 0, v_bs_1375_);
return v___x_1384_;
}
else
{
lean_object* v_v_1385_; lean_object* v___x_1386_; lean_object* v___x_1387_; lean_object* v___f_1388_; lean_object* v___x_1389_; 
v_v_1385_ = lean_array_uget_borrowed(v_bs_1375_, v_i_1374_);
v___x_1386_ = lean_box(0);
v___x_1387_ = lean_box(v___x_1383_);
lean_inc(v_v_1385_);
v___f_1388_ = lean_alloc_closure((void*)(lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Mathlib_Tactic_elabPatterns_spec__0___lam__0___boxed), 10, 3);
lean_closure_set(v___f_1388_, 0, v_v_1385_);
lean_closure_set(v___f_1388_, 1, v___x_1386_);
lean_closure_set(v___f_1388_, 2, v___x_1387_);
v___x_1389_ = l_Lean_Elab_Term_withoutModifyingElabMetaStateWithInfo___redArg(v___f_1388_, v___y_1376_, v___y_1377_, v___y_1378_, v___y_1379_, v___y_1380_, v___y_1381_);
if (lean_obj_tag(v___x_1389_) == 0)
{
lean_object* v_a_1390_; lean_object* v___x_1391_; lean_object* v_bs_x27_1392_; size_t v___x_1393_; size_t v___x_1394_; lean_object* v___x_1395_; 
v_a_1390_ = lean_ctor_get(v___x_1389_, 0);
lean_inc(v_a_1390_);
lean_dec_ref_known(v___x_1389_, 1);
v___x_1391_ = lean_unsigned_to_nat(0u);
v_bs_x27_1392_ = lean_array_uset(v_bs_1375_, v_i_1374_, v___x_1391_);
v___x_1393_ = ((size_t)1ULL);
v___x_1394_ = lean_usize_add(v_i_1374_, v___x_1393_);
v___x_1395_ = lean_array_uset(v_bs_x27_1392_, v_i_1374_, v_a_1390_);
v_i_1374_ = v___x_1394_;
v_bs_1375_ = v___x_1395_;
goto _start;
}
else
{
lean_object* v_a_1397_; lean_object* v___x_1399_; uint8_t v_isShared_1400_; uint8_t v_isSharedCheck_1404_; 
lean_dec_ref(v_bs_1375_);
v_a_1397_ = lean_ctor_get(v___x_1389_, 0);
v_isSharedCheck_1404_ = !lean_is_exclusive(v___x_1389_);
if (v_isSharedCheck_1404_ == 0)
{
v___x_1399_ = v___x_1389_;
v_isShared_1400_ = v_isSharedCheck_1404_;
goto v_resetjp_1398_;
}
else
{
lean_inc(v_a_1397_);
lean_dec(v___x_1389_);
v___x_1399_ = lean_box(0);
v_isShared_1400_ = v_isSharedCheck_1404_;
goto v_resetjp_1398_;
}
v_resetjp_1398_:
{
lean_object* v___x_1402_; 
if (v_isShared_1400_ == 0)
{
v___x_1402_ = v___x_1399_;
goto v_reusejp_1401_;
}
else
{
lean_object* v_reuseFailAlloc_1403_; 
v_reuseFailAlloc_1403_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1403_, 0, v_a_1397_);
v___x_1402_ = v_reuseFailAlloc_1403_;
goto v_reusejp_1401_;
}
v_reusejp_1401_:
{
return v___x_1402_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Mathlib_Tactic_elabPatterns_spec__0___boxed(lean_object* v_sz_1405_, lean_object* v_i_1406_, lean_object* v_bs_1407_, lean_object* v___y_1408_, lean_object* v___y_1409_, lean_object* v___y_1410_, lean_object* v___y_1411_, lean_object* v___y_1412_, lean_object* v___y_1413_, lean_object* v___y_1414_){
_start:
{
size_t v_sz_boxed_1415_; size_t v_i_boxed_1416_; lean_object* v_res_1417_; 
v_sz_boxed_1415_ = lean_unbox_usize(v_sz_1405_);
lean_dec(v_sz_1405_);
v_i_boxed_1416_ = lean_unbox_usize(v_i_1406_);
lean_dec(v_i_1406_);
v_res_1417_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Mathlib_Tactic_elabPatterns_spec__0(v_sz_boxed_1415_, v_i_boxed_1416_, v_bs_1407_, v___y_1408_, v___y_1409_, v___y_1410_, v___y_1411_, v___y_1412_, v___y_1413_);
lean_dec(v___y_1413_);
lean_dec_ref(v___y_1412_);
lean_dec(v___y_1411_);
lean_dec_ref(v___y_1410_);
lean_dec(v___y_1409_);
lean_dec_ref(v___y_1408_);
return v_res_1417_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_elabPatterns(lean_object* v_pats_1420_, lean_object* v_a_1421_, lean_object* v_a_1422_, lean_object* v_a_1423_, lean_object* v_a_1424_, lean_object* v_a_1425_, lean_object* v_a_1426_){
_start:
{
lean_object* v_declName_x3f_1428_; lean_object* v_macroStack_1429_; uint8_t v_mayPostpone_1430_; uint8_t v_errToSorry_1431_; lean_object* v_autoBoundImplicitContext_1432_; lean_object* v_autoBoundImplicitForbidden_1433_; lean_object* v_sectionVars_1434_; lean_object* v_sectionFVars_1435_; uint8_t v_implicitLambda_1436_; uint8_t v_heedElabAsElim_1437_; uint8_t v_isNoncomputableSection_1438_; uint8_t v_isMetaSection_1439_; uint8_t v_inPattern_1440_; lean_object* v_tacSnap_x3f_1441_; uint8_t v_saveRecAppSyntax_1442_; uint8_t v_holesAsSyntheticOpaque_1443_; uint8_t v_checkDeprecated_1444_; lean_object* v_fixedTermElabs_1445_; size_t v_sz_1446_; lean_object* v___x_1447_; lean_object* v___x_1448_; lean_object* v___x_1449_; uint8_t v___x_1450_; lean_object* v___x_1451_; lean_object* v___x_1452_; 
v_declName_x3f_1428_ = lean_ctor_get(v_a_1421_, 0);
v_macroStack_1429_ = lean_ctor_get(v_a_1421_, 1);
v_mayPostpone_1430_ = lean_ctor_get_uint8(v_a_1421_, sizeof(void*)*8);
v_errToSorry_1431_ = lean_ctor_get_uint8(v_a_1421_, sizeof(void*)*8 + 1);
v_autoBoundImplicitContext_1432_ = lean_ctor_get(v_a_1421_, 2);
v_autoBoundImplicitForbidden_1433_ = lean_ctor_get(v_a_1421_, 3);
v_sectionVars_1434_ = lean_ctor_get(v_a_1421_, 4);
v_sectionFVars_1435_ = lean_ctor_get(v_a_1421_, 5);
v_implicitLambda_1436_ = lean_ctor_get_uint8(v_a_1421_, sizeof(void*)*8 + 2);
v_heedElabAsElim_1437_ = lean_ctor_get_uint8(v_a_1421_, sizeof(void*)*8 + 3);
v_isNoncomputableSection_1438_ = lean_ctor_get_uint8(v_a_1421_, sizeof(void*)*8 + 4);
v_isMetaSection_1439_ = lean_ctor_get_uint8(v_a_1421_, sizeof(void*)*8 + 5);
v_inPattern_1440_ = lean_ctor_get_uint8(v_a_1421_, sizeof(void*)*8 + 7);
v_tacSnap_x3f_1441_ = lean_ctor_get(v_a_1421_, 6);
v_saveRecAppSyntax_1442_ = lean_ctor_get_uint8(v_a_1421_, sizeof(void*)*8 + 8);
v_holesAsSyntheticOpaque_1443_ = lean_ctor_get_uint8(v_a_1421_, sizeof(void*)*8 + 9);
v_checkDeprecated_1444_ = lean_ctor_get_uint8(v_a_1421_, sizeof(void*)*8 + 10);
v_fixedTermElabs_1445_ = lean_ctor_get(v_a_1421_, 7);
v_sz_1446_ = lean_array_size(v_pats_1420_);
v___x_1447_ = lean_box_usize(v_sz_1446_);
v___x_1448_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_elabPatterns___boxed__const__1));
v___x_1449_ = lean_alloc_closure((void*)(lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Mathlib_Tactic_elabPatterns_spec__0___boxed), 10, 3);
lean_closure_set(v___x_1449_, 0, v___x_1447_);
lean_closure_set(v___x_1449_, 1, v___x_1448_);
lean_closure_set(v___x_1449_, 2, v_pats_1420_);
v___x_1450_ = 1;
lean_inc_ref(v_fixedTermElabs_1445_);
lean_inc(v_tacSnap_x3f_1441_);
lean_inc(v_sectionFVars_1435_);
lean_inc(v_sectionVars_1434_);
lean_inc_ref(v_autoBoundImplicitForbidden_1433_);
lean_inc(v_autoBoundImplicitContext_1432_);
lean_inc(v_macroStack_1429_);
lean_inc(v_declName_x3f_1428_);
v___x_1451_ = lean_alloc_ctor(0, 8, 11);
lean_ctor_set(v___x_1451_, 0, v_declName_x3f_1428_);
lean_ctor_set(v___x_1451_, 1, v_macroStack_1429_);
lean_ctor_set(v___x_1451_, 2, v_autoBoundImplicitContext_1432_);
lean_ctor_set(v___x_1451_, 3, v_autoBoundImplicitForbidden_1433_);
lean_ctor_set(v___x_1451_, 4, v_sectionVars_1434_);
lean_ctor_set(v___x_1451_, 5, v_sectionFVars_1435_);
lean_ctor_set(v___x_1451_, 6, v_tacSnap_x3f_1441_);
lean_ctor_set(v___x_1451_, 7, v_fixedTermElabs_1445_);
lean_ctor_set_uint8(v___x_1451_, sizeof(void*)*8, v_mayPostpone_1430_);
lean_ctor_set_uint8(v___x_1451_, sizeof(void*)*8 + 1, v_errToSorry_1431_);
lean_ctor_set_uint8(v___x_1451_, sizeof(void*)*8 + 2, v_implicitLambda_1436_);
lean_ctor_set_uint8(v___x_1451_, sizeof(void*)*8 + 3, v_heedElabAsElim_1437_);
lean_ctor_set_uint8(v___x_1451_, sizeof(void*)*8 + 4, v_isNoncomputableSection_1438_);
lean_ctor_set_uint8(v___x_1451_, sizeof(void*)*8 + 5, v_isMetaSection_1439_);
lean_ctor_set_uint8(v___x_1451_, sizeof(void*)*8 + 6, v___x_1450_);
lean_ctor_set_uint8(v___x_1451_, sizeof(void*)*8 + 7, v_inPattern_1440_);
lean_ctor_set_uint8(v___x_1451_, sizeof(void*)*8 + 8, v_saveRecAppSyntax_1442_);
lean_ctor_set_uint8(v___x_1451_, sizeof(void*)*8 + 9, v_holesAsSyntheticOpaque_1443_);
lean_ctor_set_uint8(v___x_1451_, sizeof(void*)*8 + 10, v_checkDeprecated_1444_);
v___x_1452_ = l_Lean_Elab_Term_withoutErrToSorryImp___redArg(v___x_1449_, v___x_1451_, v_a_1422_, v_a_1423_, v_a_1424_, v_a_1425_, v_a_1426_);
lean_dec_ref_known(v___x_1451_, 8);
return v___x_1452_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_elabPatterns___boxed(lean_object* v_pats_1453_, lean_object* v_a_1454_, lean_object* v_a_1455_, lean_object* v_a_1456_, lean_object* v_a_1457_, lean_object* v_a_1458_, lean_object* v_a_1459_, lean_object* v_a_1460_){
_start:
{
lean_object* v_res_1461_; 
v_res_1461_ = lp_mathlib_Mathlib_Tactic_elabPatterns(v_pats_1453_, v_a_1454_, v_a_1455_, v_a_1456_, v_a_1457_, v_a_1458_, v_a_1459_);
lean_dec(v_a_1459_);
lean_dec_ref(v_a_1458_);
lean_dec(v_a_1457_);
lean_dec_ref(v_a_1456_);
lean_dec(v_a_1455_);
lean_dec_ref(v_a_1454_);
return v_res_1461_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00Mathlib_Tactic_matchPatterns_spec__0___redArg(lean_object* v_e_1462_, lean_object* v___y_1463_){
_start:
{
uint8_t v___x_1465_; 
v___x_1465_ = l_Lean_Expr_hasMVar(v_e_1462_);
if (v___x_1465_ == 0)
{
lean_object* v___x_1466_; 
v___x_1466_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1466_, 0, v_e_1462_);
return v___x_1466_;
}
else
{
lean_object* v___x_1467_; lean_object* v_mctx_1468_; lean_object* v___x_1469_; lean_object* v_fst_1470_; lean_object* v_snd_1471_; lean_object* v___x_1472_; lean_object* v_cache_1473_; lean_object* v_zetaDeltaFVarIds_1474_; lean_object* v_postponed_1475_; lean_object* v_diag_1476_; lean_object* v___x_1478_; uint8_t v_isShared_1479_; uint8_t v_isSharedCheck_1485_; 
v___x_1467_ = lean_st_ref_get(v___y_1463_);
v_mctx_1468_ = lean_ctor_get(v___x_1467_, 0);
lean_inc_ref(v_mctx_1468_);
lean_dec(v___x_1467_);
v___x_1469_ = l_Lean_instantiateMVarsCore(v_mctx_1468_, v_e_1462_);
v_fst_1470_ = lean_ctor_get(v___x_1469_, 0);
lean_inc(v_fst_1470_);
v_snd_1471_ = lean_ctor_get(v___x_1469_, 1);
lean_inc(v_snd_1471_);
lean_dec_ref(v___x_1469_);
v___x_1472_ = lean_st_ref_take(v___y_1463_);
v_cache_1473_ = lean_ctor_get(v___x_1472_, 1);
v_zetaDeltaFVarIds_1474_ = lean_ctor_get(v___x_1472_, 2);
v_postponed_1475_ = lean_ctor_get(v___x_1472_, 3);
v_diag_1476_ = lean_ctor_get(v___x_1472_, 4);
v_isSharedCheck_1485_ = !lean_is_exclusive(v___x_1472_);
if (v_isSharedCheck_1485_ == 0)
{
lean_object* v_unused_1486_; 
v_unused_1486_ = lean_ctor_get(v___x_1472_, 0);
lean_dec(v_unused_1486_);
v___x_1478_ = v___x_1472_;
v_isShared_1479_ = v_isSharedCheck_1485_;
goto v_resetjp_1477_;
}
else
{
lean_inc(v_diag_1476_);
lean_inc(v_postponed_1475_);
lean_inc(v_zetaDeltaFVarIds_1474_);
lean_inc(v_cache_1473_);
lean_dec(v___x_1472_);
v___x_1478_ = lean_box(0);
v_isShared_1479_ = v_isSharedCheck_1485_;
goto v_resetjp_1477_;
}
v_resetjp_1477_:
{
lean_object* v___x_1481_; 
if (v_isShared_1479_ == 0)
{
lean_ctor_set(v___x_1478_, 0, v_snd_1471_);
v___x_1481_ = v___x_1478_;
goto v_reusejp_1480_;
}
else
{
lean_object* v_reuseFailAlloc_1484_; 
v_reuseFailAlloc_1484_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_1484_, 0, v_snd_1471_);
lean_ctor_set(v_reuseFailAlloc_1484_, 1, v_cache_1473_);
lean_ctor_set(v_reuseFailAlloc_1484_, 2, v_zetaDeltaFVarIds_1474_);
lean_ctor_set(v_reuseFailAlloc_1484_, 3, v_postponed_1475_);
lean_ctor_set(v_reuseFailAlloc_1484_, 4, v_diag_1476_);
v___x_1481_ = v_reuseFailAlloc_1484_;
goto v_reusejp_1480_;
}
v_reusejp_1480_:
{
lean_object* v___x_1482_; lean_object* v___x_1483_; 
v___x_1482_ = lean_st_ref_set(v___y_1463_, v___x_1481_);
v___x_1483_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1483_, 0, v_fst_1470_);
return v___x_1483_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00Mathlib_Tactic_matchPatterns_spec__0___redArg___boxed(lean_object* v_e_1487_, lean_object* v___y_1488_, lean_object* v___y_1489_){
_start:
{
lean_object* v_res_1490_; 
v_res_1490_ = lp_mathlib_Lean_instantiateMVars___at___00Mathlib_Tactic_matchPatterns_spec__0___redArg(v_e_1487_, v___y_1488_);
lean_dec(v___y_1488_);
return v_res_1490_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00Mathlib_Tactic_matchPatterns_spec__0(lean_object* v_e_1491_, lean_object* v___y_1492_, lean_object* v___y_1493_, lean_object* v___y_1494_, lean_object* v___y_1495_){
_start:
{
lean_object* v___x_1497_; 
v___x_1497_ = lp_mathlib_Lean_instantiateMVars___at___00Mathlib_Tactic_matchPatterns_spec__0___redArg(v_e_1491_, v___y_1493_);
return v___x_1497_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00Mathlib_Tactic_matchPatterns_spec__0___boxed(lean_object* v_e_1498_, lean_object* v___y_1499_, lean_object* v___y_1500_, lean_object* v___y_1501_, lean_object* v___y_1502_, lean_object* v___y_1503_){
_start:
{
lean_object* v_res_1504_; 
v_res_1504_ = lp_mathlib_Lean_instantiateMVars___at___00Mathlib_Tactic_matchPatterns_spec__0(v_e_1498_, v___y_1499_, v___y_1500_, v___y_1501_, v___y_1502_);
lean_dec(v___y_1502_);
lean_dec_ref(v___y_1501_);
lean_dec(v___y_1500_);
lean_dec_ref(v___y_1499_);
return v_res_1504_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_anyMUnsafe_any___at___00Mathlib_Tactic_matchPatterns_spec__1(lean_object* v_a_1505_, lean_object* v_as_1506_, size_t v_i_1507_, size_t v_stop_1508_, lean_object* v___y_1509_, lean_object* v___y_1510_, lean_object* v___y_1511_, lean_object* v___y_1512_){
_start:
{
uint8_t v___x_1514_; 
v___x_1514_ = lean_usize_dec_eq(v_i_1507_, v_stop_1508_);
if (v___x_1514_ == 0)
{
lean_object* v___x_1515_; lean_object* v___x_1516_; 
v___x_1515_ = lean_array_uget_borrowed(v_as_1506_, v_i_1507_);
lean_inc_ref(v_a_1505_);
lean_inc(v___x_1515_);
v___x_1516_ = l_Lean_Elab_Tactic_Conv_matchPattern_x3f(v___x_1515_, v_a_1505_, v___y_1509_, v___y_1510_, v___y_1511_, v___y_1512_);
if (lean_obj_tag(v___x_1516_) == 0)
{
lean_object* v_a_1517_; lean_object* v___x_1519_; uint8_t v_isShared_1520_; uint8_t v_isSharedCheck_1544_; 
v_a_1517_ = lean_ctor_get(v___x_1516_, 0);
v_isSharedCheck_1544_ = !lean_is_exclusive(v___x_1516_);
if (v_isSharedCheck_1544_ == 0)
{
v___x_1519_ = v___x_1516_;
v_isShared_1520_ = v_isSharedCheck_1544_;
goto v_resetjp_1518_;
}
else
{
lean_inc(v_a_1517_);
lean_dec(v___x_1516_);
v___x_1519_ = lean_box(0);
v_isShared_1520_ = v_isSharedCheck_1544_;
goto v_resetjp_1518_;
}
v_resetjp_1518_:
{
uint8_t v___x_1521_; uint8_t v_a_1523_; 
v___x_1521_ = 1;
if (lean_obj_tag(v_a_1517_) == 1)
{
lean_object* v_val_1531_; lean_object* v___x_1533_; uint8_t v_isShared_1534_; uint8_t v_isSharedCheck_1543_; 
v_val_1531_ = lean_ctor_get(v_a_1517_, 0);
v_isSharedCheck_1543_ = !lean_is_exclusive(v_a_1517_);
if (v_isSharedCheck_1543_ == 0)
{
v___x_1533_ = v_a_1517_;
v_isShared_1534_ = v_isSharedCheck_1543_;
goto v_resetjp_1532_;
}
else
{
lean_inc(v_val_1531_);
lean_dec(v_a_1517_);
v___x_1533_ = lean_box(0);
v_isShared_1534_ = v_isSharedCheck_1543_;
goto v_resetjp_1532_;
}
v_resetjp_1532_:
{
lean_object* v_snd_1535_; lean_object* v___x_1536_; lean_object* v___x_1537_; uint8_t v___x_1538_; 
v_snd_1535_ = lean_ctor_get(v_val_1531_, 1);
lean_inc(v_snd_1535_);
lean_dec(v_val_1531_);
v___x_1536_ = lean_array_get_size(v_snd_1535_);
lean_dec(v_snd_1535_);
v___x_1537_ = lean_unsigned_to_nat(0u);
v___x_1538_ = lean_nat_dec_eq(v___x_1536_, v___x_1537_);
if (v___x_1538_ == 0)
{
lean_del_object(v___x_1533_);
v_a_1523_ = v___x_1514_;
goto v___jp_1522_;
}
else
{
lean_object* v___x_1539_; lean_object* v___x_1541_; 
lean_del_object(v___x_1519_);
lean_dec_ref(v_a_1505_);
v___x_1539_ = lean_box(v___x_1521_);
if (v_isShared_1534_ == 0)
{
lean_ctor_set_tag(v___x_1533_, 0);
lean_ctor_set(v___x_1533_, 0, v___x_1539_);
v___x_1541_ = v___x_1533_;
goto v_reusejp_1540_;
}
else
{
lean_object* v_reuseFailAlloc_1542_; 
v_reuseFailAlloc_1542_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1542_, 0, v___x_1539_);
v___x_1541_ = v_reuseFailAlloc_1542_;
goto v_reusejp_1540_;
}
v_reusejp_1540_:
{
return v___x_1541_;
}
}
}
}
else
{
lean_dec(v_a_1517_);
v_a_1523_ = v___x_1514_;
goto v___jp_1522_;
}
v___jp_1522_:
{
if (v_a_1523_ == 0)
{
size_t v___x_1524_; size_t v___x_1525_; 
lean_del_object(v___x_1519_);
v___x_1524_ = ((size_t)1ULL);
v___x_1525_ = lean_usize_add(v_i_1507_, v___x_1524_);
v_i_1507_ = v___x_1525_;
goto _start;
}
else
{
lean_object* v___x_1527_; lean_object* v___x_1529_; 
lean_dec_ref(v_a_1505_);
v___x_1527_ = lean_box(v___x_1521_);
if (v_isShared_1520_ == 0)
{
lean_ctor_set(v___x_1519_, 0, v___x_1527_);
v___x_1529_ = v___x_1519_;
goto v_reusejp_1528_;
}
else
{
lean_object* v_reuseFailAlloc_1530_; 
v_reuseFailAlloc_1530_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1530_, 0, v___x_1527_);
v___x_1529_ = v_reuseFailAlloc_1530_;
goto v_reusejp_1528_;
}
v_reusejp_1528_:
{
return v___x_1529_;
}
}
}
}
}
else
{
lean_object* v_a_1545_; lean_object* v___x_1547_; uint8_t v_isShared_1548_; uint8_t v_isSharedCheck_1552_; 
lean_dec_ref(v_a_1505_);
v_a_1545_ = lean_ctor_get(v___x_1516_, 0);
v_isSharedCheck_1552_ = !lean_is_exclusive(v___x_1516_);
if (v_isSharedCheck_1552_ == 0)
{
v___x_1547_ = v___x_1516_;
v_isShared_1548_ = v_isSharedCheck_1552_;
goto v_resetjp_1546_;
}
else
{
lean_inc(v_a_1545_);
lean_dec(v___x_1516_);
v___x_1547_ = lean_box(0);
v_isShared_1548_ = v_isSharedCheck_1552_;
goto v_resetjp_1546_;
}
v_resetjp_1546_:
{
lean_object* v___x_1550_; 
if (v_isShared_1548_ == 0)
{
v___x_1550_ = v___x_1547_;
goto v_reusejp_1549_;
}
else
{
lean_object* v_reuseFailAlloc_1551_; 
v_reuseFailAlloc_1551_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1551_, 0, v_a_1545_);
v___x_1550_ = v_reuseFailAlloc_1551_;
goto v_reusejp_1549_;
}
v_reusejp_1549_:
{
return v___x_1550_;
}
}
}
}
else
{
uint8_t v___x_1553_; lean_object* v___x_1554_; lean_object* v___x_1555_; 
lean_dec_ref(v_a_1505_);
v___x_1553_ = 0;
v___x_1554_ = lean_box(v___x_1553_);
v___x_1555_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1555_, 0, v___x_1554_);
return v___x_1555_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_anyMUnsafe_any___at___00Mathlib_Tactic_matchPatterns_spec__1___boxed(lean_object* v_a_1556_, lean_object* v_as_1557_, lean_object* v_i_1558_, lean_object* v_stop_1559_, lean_object* v___y_1560_, lean_object* v___y_1561_, lean_object* v___y_1562_, lean_object* v___y_1563_, lean_object* v___y_1564_){
_start:
{
size_t v_i_boxed_1565_; size_t v_stop_boxed_1566_; lean_object* v_res_1567_; 
v_i_boxed_1565_ = lean_unbox_usize(v_i_1558_);
lean_dec(v_i_1558_);
v_stop_boxed_1566_ = lean_unbox_usize(v_stop_1559_);
lean_dec(v_stop_1559_);
v_res_1567_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_anyMUnsafe_any___at___00Mathlib_Tactic_matchPatterns_spec__1(v_a_1556_, v_as_1557_, v_i_boxed_1565_, v_stop_boxed_1566_, v___y_1560_, v___y_1561_, v___y_1562_, v___y_1563_);
lean_dec(v___y_1563_);
lean_dec_ref(v___y_1562_);
lean_dec(v___y_1561_);
lean_dec_ref(v___y_1560_);
lean_dec_ref(v_as_1557_);
return v_res_1567_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_matchPatterns(lean_object* v_pats_1568_, lean_object* v_e_1569_, lean_object* v_a_1570_, lean_object* v_a_1571_, lean_object* v_a_1572_, lean_object* v_a_1573_){
_start:
{
lean_object* v___x_1575_; lean_object* v_a_1576_; lean_object* v___x_1578_; uint8_t v_isShared_1579_; uint8_t v_isSharedCheck_1594_; 
v___x_1575_ = lp_mathlib_Lean_instantiateMVars___at___00Mathlib_Tactic_matchPatterns_spec__0___redArg(v_e_1569_, v_a_1571_);
v_a_1576_ = lean_ctor_get(v___x_1575_, 0);
v_isSharedCheck_1594_ = !lean_is_exclusive(v___x_1575_);
if (v_isSharedCheck_1594_ == 0)
{
v___x_1578_ = v___x_1575_;
v_isShared_1579_ = v_isSharedCheck_1594_;
goto v_resetjp_1577_;
}
else
{
lean_inc(v_a_1576_);
lean_dec(v___x_1575_);
v___x_1578_ = lean_box(0);
v_isShared_1579_ = v_isSharedCheck_1594_;
goto v_resetjp_1577_;
}
v_resetjp_1577_:
{
lean_object* v___x_1580_; lean_object* v___x_1581_; uint8_t v___x_1582_; 
v___x_1580_ = lean_unsigned_to_nat(0u);
v___x_1581_ = lean_array_get_size(v_pats_1568_);
v___x_1582_ = lean_nat_dec_lt(v___x_1580_, v___x_1581_);
if (v___x_1582_ == 0)
{
lean_object* v___x_1583_; lean_object* v___x_1585_; 
lean_dec(v_a_1576_);
v___x_1583_ = lean_box(v___x_1582_);
if (v_isShared_1579_ == 0)
{
lean_ctor_set(v___x_1578_, 0, v___x_1583_);
v___x_1585_ = v___x_1578_;
goto v_reusejp_1584_;
}
else
{
lean_object* v_reuseFailAlloc_1586_; 
v_reuseFailAlloc_1586_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1586_, 0, v___x_1583_);
v___x_1585_ = v_reuseFailAlloc_1586_;
goto v_reusejp_1584_;
}
v_reusejp_1584_:
{
return v___x_1585_;
}
}
else
{
if (v___x_1582_ == 0)
{
lean_object* v___x_1587_; lean_object* v___x_1589_; 
lean_dec(v_a_1576_);
v___x_1587_ = lean_box(v___x_1582_);
if (v_isShared_1579_ == 0)
{
lean_ctor_set(v___x_1578_, 0, v___x_1587_);
v___x_1589_ = v___x_1578_;
goto v_reusejp_1588_;
}
else
{
lean_object* v_reuseFailAlloc_1590_; 
v_reuseFailAlloc_1590_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1590_, 0, v___x_1587_);
v___x_1589_ = v_reuseFailAlloc_1590_;
goto v_reusejp_1588_;
}
v_reusejp_1588_:
{
return v___x_1589_;
}
}
else
{
size_t v___x_1591_; size_t v___x_1592_; lean_object* v___x_1593_; 
lean_del_object(v___x_1578_);
v___x_1591_ = ((size_t)0ULL);
v___x_1592_ = lean_usize_of_nat(v___x_1581_);
v___x_1593_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_anyMUnsafe_any___at___00Mathlib_Tactic_matchPatterns_spec__1(v_a_1576_, v_pats_1568_, v___x_1591_, v___x_1592_, v_a_1570_, v_a_1571_, v_a_1572_, v_a_1573_);
return v___x_1593_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_matchPatterns___boxed(lean_object* v_pats_1595_, lean_object* v_e_1596_, lean_object* v_a_1597_, lean_object* v_a_1598_, lean_object* v_a_1599_, lean_object* v_a_1600_, lean_object* v_a_1601_){
_start:
{
lean_object* v_res_1602_; 
v_res_1602_ = lp_mathlib_Mathlib_Tactic_matchPatterns(v_pats_1595_, v_e_1596_, v_a_1597_, v_a_1598_, v_a_1599_, v_a_1600_);
lean_dec(v_a_1600_);
lean_dec_ref(v_a_1599_);
lean_dec(v_a_1598_);
lean_dec_ref(v_a_1597_);
lean_dec_ref(v_pats_1595_);
return v_res_1602_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_elabCasesM___lam__0(lean_object* v___x_1603_, uint8_t v_recursive_1604_, uint8_t v_allowSplit_1605_, uint8_t v___x_1606_, lean_object* v___y_1607_, lean_object* v___y_1608_, lean_object* v___y_1609_, lean_object* v___y_1610_, lean_object* v___y_1611_, lean_object* v___y_1612_, lean_object* v___y_1613_, lean_object* v___y_1614_){
_start:
{
lean_object* v___x_1616_; 
v___x_1616_ = l_Lean_Elab_Tactic_getMainGoal___redArg(v___y_1608_, v___y_1611_, v___y_1612_, v___y_1613_, v___y_1614_);
if (lean_obj_tag(v___x_1616_) == 0)
{
lean_object* v_a_1617_; lean_object* v___x_1618_; 
v_a_1617_ = lean_ctor_get(v___x_1616_, 0);
lean_inc(v_a_1617_);
lean_dec_ref_known(v___x_1616_, 1);
v___x_1618_ = lp_mathlib_Lean_MVarId_casesMatching(v___x_1603_, v_recursive_1604_, v_allowSplit_1605_, v___x_1606_, v_a_1617_, v___y_1611_, v___y_1612_, v___y_1613_, v___y_1614_);
if (lean_obj_tag(v___x_1618_) == 0)
{
lean_object* v_a_1619_; lean_object* v___x_1620_; 
v_a_1619_ = lean_ctor_get(v___x_1618_, 0);
lean_inc(v_a_1619_);
lean_dec_ref_known(v___x_1618_, 1);
v___x_1620_ = l_Lean_Elab_Tactic_replaceMainGoal___redArg(v_a_1619_, v___y_1608_, v___y_1611_, v___y_1612_, v___y_1613_, v___y_1614_);
if (lean_obj_tag(v___x_1620_) == 0)
{
lean_object* v___x_1622_; uint8_t v_isShared_1623_; uint8_t v_isSharedCheck_1628_; 
v_isSharedCheck_1628_ = !lean_is_exclusive(v___x_1620_);
if (v_isSharedCheck_1628_ == 0)
{
lean_object* v_unused_1629_; 
v_unused_1629_ = lean_ctor_get(v___x_1620_, 0);
lean_dec(v_unused_1629_);
v___x_1622_ = v___x_1620_;
v_isShared_1623_ = v_isSharedCheck_1628_;
goto v_resetjp_1621_;
}
else
{
lean_dec(v___x_1620_);
v___x_1622_ = lean_box(0);
v_isShared_1623_ = v_isSharedCheck_1628_;
goto v_resetjp_1621_;
}
v_resetjp_1621_:
{
lean_object* v___x_1624_; lean_object* v___x_1626_; 
v___x_1624_ = lean_box(0);
if (v_isShared_1623_ == 0)
{
lean_ctor_set(v___x_1622_, 0, v___x_1624_);
v___x_1626_ = v___x_1622_;
goto v_reusejp_1625_;
}
else
{
lean_object* v_reuseFailAlloc_1627_; 
v_reuseFailAlloc_1627_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1627_, 0, v___x_1624_);
v___x_1626_ = v_reuseFailAlloc_1627_;
goto v_reusejp_1625_;
}
v_reusejp_1625_:
{
return v___x_1626_;
}
}
}
else
{
return v___x_1620_;
}
}
else
{
lean_object* v_a_1630_; lean_object* v___x_1632_; uint8_t v_isShared_1633_; uint8_t v_isSharedCheck_1637_; 
v_a_1630_ = lean_ctor_get(v___x_1618_, 0);
v_isSharedCheck_1637_ = !lean_is_exclusive(v___x_1618_);
if (v_isSharedCheck_1637_ == 0)
{
v___x_1632_ = v___x_1618_;
v_isShared_1633_ = v_isSharedCheck_1637_;
goto v_resetjp_1631_;
}
else
{
lean_inc(v_a_1630_);
lean_dec(v___x_1618_);
v___x_1632_ = lean_box(0);
v_isShared_1633_ = v_isSharedCheck_1637_;
goto v_resetjp_1631_;
}
v_resetjp_1631_:
{
lean_object* v___x_1635_; 
if (v_isShared_1633_ == 0)
{
v___x_1635_ = v___x_1632_;
goto v_reusejp_1634_;
}
else
{
lean_object* v_reuseFailAlloc_1636_; 
v_reuseFailAlloc_1636_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1636_, 0, v_a_1630_);
v___x_1635_ = v_reuseFailAlloc_1636_;
goto v_reusejp_1634_;
}
v_reusejp_1634_:
{
return v___x_1635_;
}
}
}
}
else
{
lean_object* v_a_1638_; lean_object* v___x_1640_; uint8_t v_isShared_1641_; uint8_t v_isSharedCheck_1645_; 
lean_dec_ref(v___x_1603_);
v_a_1638_ = lean_ctor_get(v___x_1616_, 0);
v_isSharedCheck_1645_ = !lean_is_exclusive(v___x_1616_);
if (v_isSharedCheck_1645_ == 0)
{
v___x_1640_ = v___x_1616_;
v_isShared_1641_ = v_isSharedCheck_1645_;
goto v_resetjp_1639_;
}
else
{
lean_inc(v_a_1638_);
lean_dec(v___x_1616_);
v___x_1640_ = lean_box(0);
v_isShared_1641_ = v_isSharedCheck_1645_;
goto v_resetjp_1639_;
}
v_resetjp_1639_:
{
lean_object* v___x_1643_; 
if (v_isShared_1641_ == 0)
{
v___x_1643_ = v___x_1640_;
goto v_reusejp_1642_;
}
else
{
lean_object* v_reuseFailAlloc_1644_; 
v_reuseFailAlloc_1644_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1644_, 0, v_a_1638_);
v___x_1643_ = v_reuseFailAlloc_1644_;
goto v_reusejp_1642_;
}
v_reusejp_1642_:
{
return v___x_1643_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_elabCasesM___lam__0___boxed(lean_object* v___x_1646_, lean_object* v_recursive_1647_, lean_object* v_allowSplit_1648_, lean_object* v___x_1649_, lean_object* v___y_1650_, lean_object* v___y_1651_, lean_object* v___y_1652_, lean_object* v___y_1653_, lean_object* v___y_1654_, lean_object* v___y_1655_, lean_object* v___y_1656_, lean_object* v___y_1657_, lean_object* v___y_1658_){
_start:
{
uint8_t v_recursive_boxed_1659_; uint8_t v_allowSplit_boxed_1660_; uint8_t v___x_220__boxed_1661_; lean_object* v_res_1662_; 
v_recursive_boxed_1659_ = lean_unbox(v_recursive_1647_);
v_allowSplit_boxed_1660_ = lean_unbox(v_allowSplit_1648_);
v___x_220__boxed_1661_ = lean_unbox(v___x_1649_);
v_res_1662_ = lp_mathlib_Mathlib_Tactic_elabCasesM___lam__0(v___x_1646_, v_recursive_boxed_1659_, v_allowSplit_boxed_1660_, v___x_220__boxed_1661_, v___y_1650_, v___y_1651_, v___y_1652_, v___y_1653_, v___y_1654_, v___y_1655_, v___y_1656_, v___y_1657_);
lean_dec(v___y_1657_);
lean_dec_ref(v___y_1656_);
lean_dec(v___y_1655_);
lean_dec_ref(v___y_1654_);
lean_dec(v___y_1653_);
lean_dec_ref(v___y_1652_);
lean_dec(v___y_1651_);
lean_dec_ref(v___y_1650_);
return v_res_1662_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_elabCasesM(lean_object* v_pats_1663_, uint8_t v_recursive_1664_, uint8_t v_allowSplit_1665_, lean_object* v_a_1666_, lean_object* v_a_1667_, lean_object* v_a_1668_, lean_object* v_a_1669_, lean_object* v_a_1670_, lean_object* v_a_1671_, lean_object* v_a_1672_, lean_object* v_a_1673_){
_start:
{
lean_object* v___x_1675_; 
v___x_1675_ = lp_mathlib_Mathlib_Tactic_elabPatterns(v_pats_1663_, v_a_1668_, v_a_1669_, v_a_1670_, v_a_1671_, v_a_1672_, v_a_1673_);
if (lean_obj_tag(v___x_1675_) == 0)
{
lean_object* v_a_1676_; lean_object* v___x_1677_; uint8_t v___x_1678_; lean_object* v___x_1679_; lean_object* v___x_1680_; lean_object* v___x_1681_; lean_object* v___f_1682_; lean_object* v___x_1683_; 
v_a_1676_ = lean_ctor_get(v___x_1675_, 0);
lean_inc(v_a_1676_);
lean_dec_ref_known(v___x_1675_, 1);
v___x_1677_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_Tactic_matchPatterns___boxed), 7, 1);
lean_closure_set(v___x_1677_, 0, v_a_1676_);
v___x_1678_ = 1;
v___x_1679_ = lean_box(v_recursive_1664_);
v___x_1680_ = lean_box(v_allowSplit_1665_);
v___x_1681_ = lean_box(v___x_1678_);
v___f_1682_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_Tactic_elabCasesM___lam__0___boxed), 13, 4);
lean_closure_set(v___f_1682_, 0, v___x_1677_);
lean_closure_set(v___f_1682_, 1, v___x_1679_);
lean_closure_set(v___f_1682_, 2, v___x_1680_);
lean_closure_set(v___f_1682_, 3, v___x_1681_);
v___x_1683_ = l_Lean_Elab_Tactic_withMainContext___redArg(v___f_1682_, v_a_1666_, v_a_1667_, v_a_1668_, v_a_1669_, v_a_1670_, v_a_1671_, v_a_1672_, v_a_1673_);
return v___x_1683_;
}
else
{
lean_object* v_a_1684_; lean_object* v___x_1686_; uint8_t v_isShared_1687_; uint8_t v_isSharedCheck_1691_; 
v_a_1684_ = lean_ctor_get(v___x_1675_, 0);
v_isSharedCheck_1691_ = !lean_is_exclusive(v___x_1675_);
if (v_isSharedCheck_1691_ == 0)
{
v___x_1686_ = v___x_1675_;
v_isShared_1687_ = v_isSharedCheck_1691_;
goto v_resetjp_1685_;
}
else
{
lean_inc(v_a_1684_);
lean_dec(v___x_1675_);
v___x_1686_ = lean_box(0);
v_isShared_1687_ = v_isSharedCheck_1691_;
goto v_resetjp_1685_;
}
v_resetjp_1685_:
{
lean_object* v___x_1689_; 
if (v_isShared_1687_ == 0)
{
v___x_1689_ = v___x_1686_;
goto v_reusejp_1688_;
}
else
{
lean_object* v_reuseFailAlloc_1690_; 
v_reuseFailAlloc_1690_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1690_, 0, v_a_1684_);
v___x_1689_ = v_reuseFailAlloc_1690_;
goto v_reusejp_1688_;
}
v_reusejp_1688_:
{
return v___x_1689_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_elabCasesM___boxed(lean_object* v_pats_1692_, lean_object* v_recursive_1693_, lean_object* v_allowSplit_1694_, lean_object* v_a_1695_, lean_object* v_a_1696_, lean_object* v_a_1697_, lean_object* v_a_1698_, lean_object* v_a_1699_, lean_object* v_a_1700_, lean_object* v_a_1701_, lean_object* v_a_1702_, lean_object* v_a_1703_){
_start:
{
uint8_t v_recursive_boxed_1704_; uint8_t v_allowSplit_boxed_1705_; lean_object* v_res_1706_; 
v_recursive_boxed_1704_ = lean_unbox(v_recursive_1693_);
v_allowSplit_boxed_1705_ = lean_unbox(v_allowSplit_1694_);
v_res_1706_ = lp_mathlib_Mathlib_Tactic_elabCasesM(v_pats_1692_, v_recursive_boxed_1704_, v_allowSplit_boxed_1705_, v_a_1695_, v_a_1696_, v_a_1697_, v_a_1698_, v_a_1699_, v_a_1700_, v_a_1701_, v_a_1702_);
lean_dec(v_a_1702_);
lean_dec_ref(v_a_1701_);
lean_dec(v_a_1700_);
lean_dec_ref(v_a_1699_);
lean_dec(v_a_1698_);
lean_dec_ref(v_a_1697_);
lean_dec(v_a_1696_);
lean_dec_ref(v_a_1695_);
return v_res_1706_;
}
}
static lean_object* _init_lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic___aux__Mathlib__Tactic__CasesM______elabRules__Mathlib__Tactic__casesM__1_spec__0___redArg___closed__0(void){
_start:
{
lean_object* v___x_1773_; lean_object* v___x_1774_; lean_object* v___x_1775_; 
v___x_1773_ = lean_box(0);
v___x_1774_ = l_Lean_Elab_unsupportedSyntaxExceptionId;
v___x_1775_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_1775_, 0, v___x_1774_);
lean_ctor_set(v___x_1775_, 1, v___x_1773_);
return v___x_1775_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic___aux__Mathlib__Tactic__CasesM______elabRules__Mathlib__Tactic__casesM__1_spec__0___redArg(){
_start:
{
lean_object* v___x_1777_; lean_object* v___x_1778_; 
v___x_1777_ = lean_obj_once(&lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic___aux__Mathlib__Tactic__CasesM______elabRules__Mathlib__Tactic__casesM__1_spec__0___redArg___closed__0, &lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic___aux__Mathlib__Tactic__CasesM______elabRules__Mathlib__Tactic__casesM__1_spec__0___redArg___closed__0_once, _init_lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic___aux__Mathlib__Tactic__CasesM______elabRules__Mathlib__Tactic__casesM__1_spec__0___redArg___closed__0);
v___x_1778_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_1778_, 0, v___x_1777_);
return v___x_1778_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic___aux__Mathlib__Tactic__CasesM______elabRules__Mathlib__Tactic__casesM__1_spec__0___redArg___boxed(lean_object* v___y_1779_){
_start:
{
lean_object* v_res_1780_; 
v_res_1780_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic___aux__Mathlib__Tactic__CasesM______elabRules__Mathlib__Tactic__casesM__1_spec__0___redArg();
return v_res_1780_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic___aux__Mathlib__Tactic__CasesM______elabRules__Mathlib__Tactic__casesM__1_spec__0(lean_object* v_00_u03b1_1781_, lean_object* v___y_1782_, lean_object* v___y_1783_, lean_object* v___y_1784_, lean_object* v___y_1785_, lean_object* v___y_1786_, lean_object* v___y_1787_, lean_object* v___y_1788_, lean_object* v___y_1789_){
_start:
{
lean_object* v___x_1791_; 
v___x_1791_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic___aux__Mathlib__Tactic__CasesM______elabRules__Mathlib__Tactic__casesM__1_spec__0___redArg();
return v___x_1791_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic___aux__Mathlib__Tactic__CasesM______elabRules__Mathlib__Tactic__casesM__1_spec__0___boxed(lean_object* v_00_u03b1_1792_, lean_object* v___y_1793_, lean_object* v___y_1794_, lean_object* v___y_1795_, lean_object* v___y_1796_, lean_object* v___y_1797_, lean_object* v___y_1798_, lean_object* v___y_1799_, lean_object* v___y_1800_, lean_object* v___y_1801_){
_start:
{
lean_object* v_res_1802_; 
v_res_1802_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic___aux__Mathlib__Tactic__CasesM______elabRules__Mathlib__Tactic__casesM__1_spec__0(v_00_u03b1_1792_, v___y_1793_, v___y_1794_, v___y_1795_, v___y_1796_, v___y_1797_, v___y_1798_, v___y_1799_, v___y_1800_);
lean_dec(v___y_1800_);
lean_dec_ref(v___y_1799_);
lean_dec(v___y_1798_);
lean_dec_ref(v___y_1797_);
lean_dec(v___y_1796_);
lean_dec_ref(v___y_1795_);
lean_dec(v___y_1794_);
lean_dec_ref(v___y_1793_);
return v_res_1802_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__CasesM______elabRules__Mathlib__Tactic__casesM__1(lean_object* v_x_1803_, lean_object* v_a_1804_, lean_object* v_a_1805_, lean_object* v_a_1806_, lean_object* v_a_1807_, lean_object* v_a_1808_, lean_object* v_a_1809_, lean_object* v_a_1810_, lean_object* v_a_1811_){
_start:
{
lean_object* v___x_1813_; uint8_t v___x_1814_; 
v___x_1813_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_casesM___closed__3));
lean_inc(v_x_1803_);
v___x_1814_ = l_Lean_Syntax_isOfKind(v_x_1803_, v___x_1813_);
if (v___x_1814_ == 0)
{
lean_object* v___x_1815_; 
lean_dec(v_x_1803_);
v___x_1815_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic___aux__Mathlib__Tactic__CasesM______elabRules__Mathlib__Tactic__casesM__1_spec__0___redArg();
return v___x_1815_;
}
else
{
lean_object* v___x_1816_; lean_object* v___x_1817_; lean_object* v___x_1818_; lean_object* v___x_1819_; lean_object* v_pats_1820_; lean_object* v___x_1821_; 
v___x_1816_ = lean_unsigned_to_nat(1u);
v___x_1817_ = l_Lean_Syntax_getArg(v_x_1803_, v___x_1816_);
v___x_1818_ = lean_unsigned_to_nat(3u);
v___x_1819_ = l_Lean_Syntax_getArg(v_x_1803_, v___x_1818_);
lean_dec(v_x_1803_);
v_pats_1820_ = l_Lean_Syntax_getArgs(v___x_1819_);
lean_dec(v___x_1819_);
v___x_1821_ = l_Lean_Syntax_getOptional_x3f(v___x_1817_);
lean_dec(v___x_1817_);
if (lean_obj_tag(v___x_1821_) == 0)
{
lean_object* v___x_1822_; uint8_t v___x_1823_; lean_object* v___x_1824_; 
v___x_1822_ = l_Lean_Syntax_TSepArray_getElems___redArg(v_pats_1820_);
lean_dec_ref(v_pats_1820_);
v___x_1823_ = 0;
v___x_1824_ = lp_mathlib_Mathlib_Tactic_elabCasesM(v___x_1822_, v___x_1823_, v___x_1814_, v_a_1804_, v_a_1805_, v_a_1806_, v_a_1807_, v_a_1808_, v_a_1809_, v_a_1810_, v_a_1811_);
return v___x_1824_;
}
else
{
lean_object* v___x_1825_; lean_object* v___x_1826_; 
lean_dec_ref_known(v___x_1821_, 1);
v___x_1825_ = l_Lean_Syntax_TSepArray_getElems___redArg(v_pats_1820_);
lean_dec_ref(v_pats_1820_);
v___x_1826_ = lp_mathlib_Mathlib_Tactic_elabCasesM(v___x_1825_, v___x_1814_, v___x_1814_, v_a_1804_, v_a_1805_, v_a_1806_, v_a_1807_, v_a_1808_, v_a_1809_, v_a_1810_, v_a_1811_);
return v___x_1826_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__CasesM______elabRules__Mathlib__Tactic__casesM__1___boxed(lean_object* v_x_1827_, lean_object* v_a_1828_, lean_object* v_a_1829_, lean_object* v_a_1830_, lean_object* v_a_1831_, lean_object* v_a_1832_, lean_object* v_a_1833_, lean_object* v_a_1834_, lean_object* v_a_1835_, lean_object* v_a_1836_){
_start:
{
lean_object* v_res_1837_; 
v_res_1837_ = lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__CasesM______elabRules__Mathlib__Tactic__casesM__1(v_x_1827_, v_a_1828_, v_a_1829_, v_a_1830_, v_a_1831_, v_a_1832_, v_a_1833_, v_a_1834_, v_a_1835_);
lean_dec(v_a_1835_);
lean_dec_ref(v_a_1834_);
lean_dec(v_a_1833_);
lean_dec_ref(v_a_1832_);
lean_dec(v_a_1831_);
lean_dec_ref(v_a_1830_);
lean_dec(v_a_1829_);
lean_dec_ref(v_a_1828_);
return v_res_1837_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__CasesM______elabRules__Mathlib__Tactic__casesm_x21__1(lean_object* v_x_1863_, lean_object* v_a_1864_, lean_object* v_a_1865_, lean_object* v_a_1866_, lean_object* v_a_1867_, lean_object* v_a_1868_, lean_object* v_a_1869_, lean_object* v_a_1870_, lean_object* v_a_1871_){
_start:
{
lean_object* v___y_1874_; uint8_t v___y_1875_; lean_object* v___x_1878_; uint8_t v___x_1879_; 
v___x_1878_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_casesm_x21___closed__1));
lean_inc(v_x_1863_);
v___x_1879_ = l_Lean_Syntax_isOfKind(v_x_1863_, v___x_1878_);
if (v___x_1879_ == 0)
{
lean_object* v___x_1880_; 
lean_dec(v_x_1863_);
v___x_1880_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic___aux__Mathlib__Tactic__CasesM______elabRules__Mathlib__Tactic__casesM__1_spec__0___redArg();
return v___x_1880_;
}
else
{
lean_object* v___x_1881_; lean_object* v___x_1882_; lean_object* v___x_1883_; lean_object* v___x_1884_; lean_object* v_pats_1885_; lean_object* v___x_1886_; 
v___x_1881_ = lean_unsigned_to_nat(1u);
v___x_1882_ = l_Lean_Syntax_getArg(v_x_1863_, v___x_1881_);
v___x_1883_ = lean_unsigned_to_nat(3u);
v___x_1884_ = l_Lean_Syntax_getArg(v_x_1863_, v___x_1883_);
lean_dec(v_x_1863_);
v_pats_1885_ = l_Lean_Syntax_getArgs(v___x_1884_);
lean_dec(v___x_1884_);
v___x_1886_ = l_Lean_Syntax_getOptional_x3f(v___x_1882_);
lean_dec(v___x_1882_);
if (lean_obj_tag(v___x_1886_) == 0)
{
lean_object* v___x_1887_; uint8_t v___x_1888_; 
v___x_1887_ = l_Lean_Syntax_TSepArray_getElems___redArg(v_pats_1885_);
lean_dec_ref(v_pats_1885_);
v___x_1888_ = 0;
v___y_1874_ = v___x_1887_;
v___y_1875_ = v___x_1888_;
goto v___jp_1873_;
}
else
{
lean_object* v___x_1889_; 
lean_dec_ref_known(v___x_1886_, 1);
v___x_1889_ = l_Lean_Syntax_TSepArray_getElems___redArg(v_pats_1885_);
lean_dec_ref(v_pats_1885_);
v___y_1874_ = v___x_1889_;
v___y_1875_ = v___x_1879_;
goto v___jp_1873_;
}
}
v___jp_1873_:
{
uint8_t v___x_1876_; lean_object* v___x_1877_; 
v___x_1876_ = 0;
v___x_1877_ = lp_mathlib_Mathlib_Tactic_elabCasesM(v___y_1874_, v___y_1875_, v___x_1876_, v_a_1864_, v_a_1865_, v_a_1866_, v_a_1867_, v_a_1868_, v_a_1869_, v_a_1870_, v_a_1871_);
return v___x_1877_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__CasesM______elabRules__Mathlib__Tactic__casesm_x21__1___boxed(lean_object* v_x_1890_, lean_object* v_a_1891_, lean_object* v_a_1892_, lean_object* v_a_1893_, lean_object* v_a_1894_, lean_object* v_a_1895_, lean_object* v_a_1896_, lean_object* v_a_1897_, lean_object* v_a_1898_, lean_object* v_a_1899_){
_start:
{
lean_object* v_res_1900_; 
v_res_1900_ = lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__CasesM______elabRules__Mathlib__Tactic__casesm_x21__1(v_x_1890_, v_a_1891_, v_a_1892_, v_a_1893_, v_a_1894_, v_a_1895_, v_a_1896_, v_a_1897_, v_a_1898_);
lean_dec(v_a_1898_);
lean_dec_ref(v_a_1897_);
lean_dec(v_a_1896_);
lean_dec_ref(v_a_1895_);
lean_dec(v_a_1894_);
lean_dec_ref(v_a_1893_);
lean_dec(v_a_1892_);
lean_dec_ref(v_a_1891_);
return v_res_1900_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_elabCasesType___lam__0(lean_object* v_a_1901_, uint8_t v_recursive_1902_, uint8_t v_allowSplit_1903_, lean_object* v___y_1904_, lean_object* v___y_1905_, lean_object* v___y_1906_, lean_object* v___y_1907_, lean_object* v___y_1908_, lean_object* v___y_1909_, lean_object* v___y_1910_, lean_object* v___y_1911_){
_start:
{
lean_object* v___x_1913_; 
v___x_1913_ = l_Lean_Elab_Tactic_getMainGoal___redArg(v___y_1905_, v___y_1908_, v___y_1909_, v___y_1910_, v___y_1911_);
if (lean_obj_tag(v___x_1913_) == 0)
{
lean_object* v_a_1914_; lean_object* v___x_1915_; 
v_a_1914_ = lean_ctor_get(v___x_1913_, 0);
lean_inc(v_a_1914_);
lean_dec_ref_known(v___x_1913_, 1);
v___x_1915_ = lp_mathlib_Lean_MVarId_casesType(v_a_1901_, v_recursive_1902_, v_allowSplit_1903_, v_a_1914_, v___y_1908_, v___y_1909_, v___y_1910_, v___y_1911_);
if (lean_obj_tag(v___x_1915_) == 0)
{
lean_object* v_a_1916_; lean_object* v___x_1917_; 
v_a_1916_ = lean_ctor_get(v___x_1915_, 0);
lean_inc(v_a_1916_);
lean_dec_ref_known(v___x_1915_, 1);
v___x_1917_ = l_Lean_Elab_Tactic_replaceMainGoal___redArg(v_a_1916_, v___y_1905_, v___y_1908_, v___y_1909_, v___y_1910_, v___y_1911_);
if (lean_obj_tag(v___x_1917_) == 0)
{
lean_object* v___x_1919_; uint8_t v_isShared_1920_; uint8_t v_isSharedCheck_1925_; 
v_isSharedCheck_1925_ = !lean_is_exclusive(v___x_1917_);
if (v_isSharedCheck_1925_ == 0)
{
lean_object* v_unused_1926_; 
v_unused_1926_ = lean_ctor_get(v___x_1917_, 0);
lean_dec(v_unused_1926_);
v___x_1919_ = v___x_1917_;
v_isShared_1920_ = v_isSharedCheck_1925_;
goto v_resetjp_1918_;
}
else
{
lean_dec(v___x_1917_);
v___x_1919_ = lean_box(0);
v_isShared_1920_ = v_isSharedCheck_1925_;
goto v_resetjp_1918_;
}
v_resetjp_1918_:
{
lean_object* v___x_1921_; lean_object* v___x_1923_; 
v___x_1921_ = lean_box(0);
if (v_isShared_1920_ == 0)
{
lean_ctor_set(v___x_1919_, 0, v___x_1921_);
v___x_1923_ = v___x_1919_;
goto v_reusejp_1922_;
}
else
{
lean_object* v_reuseFailAlloc_1924_; 
v_reuseFailAlloc_1924_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1924_, 0, v___x_1921_);
v___x_1923_ = v_reuseFailAlloc_1924_;
goto v_reusejp_1922_;
}
v_reusejp_1922_:
{
return v___x_1923_;
}
}
}
else
{
return v___x_1917_;
}
}
else
{
lean_object* v_a_1927_; lean_object* v___x_1929_; uint8_t v_isShared_1930_; uint8_t v_isSharedCheck_1934_; 
v_a_1927_ = lean_ctor_get(v___x_1915_, 0);
v_isSharedCheck_1934_ = !lean_is_exclusive(v___x_1915_);
if (v_isSharedCheck_1934_ == 0)
{
v___x_1929_ = v___x_1915_;
v_isShared_1930_ = v_isSharedCheck_1934_;
goto v_resetjp_1928_;
}
else
{
lean_inc(v_a_1927_);
lean_dec(v___x_1915_);
v___x_1929_ = lean_box(0);
v_isShared_1930_ = v_isSharedCheck_1934_;
goto v_resetjp_1928_;
}
v_resetjp_1928_:
{
lean_object* v___x_1932_; 
if (v_isShared_1930_ == 0)
{
v___x_1932_ = v___x_1929_;
goto v_reusejp_1931_;
}
else
{
lean_object* v_reuseFailAlloc_1933_; 
v_reuseFailAlloc_1933_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1933_, 0, v_a_1927_);
v___x_1932_ = v_reuseFailAlloc_1933_;
goto v_reusejp_1931_;
}
v_reusejp_1931_:
{
return v___x_1932_;
}
}
}
}
else
{
lean_object* v_a_1935_; lean_object* v___x_1937_; uint8_t v_isShared_1938_; uint8_t v_isSharedCheck_1942_; 
lean_dec_ref(v_a_1901_);
v_a_1935_ = lean_ctor_get(v___x_1913_, 0);
v_isSharedCheck_1942_ = !lean_is_exclusive(v___x_1913_);
if (v_isSharedCheck_1942_ == 0)
{
v___x_1937_ = v___x_1913_;
v_isShared_1938_ = v_isSharedCheck_1942_;
goto v_resetjp_1936_;
}
else
{
lean_inc(v_a_1935_);
lean_dec(v___x_1913_);
v___x_1937_ = lean_box(0);
v_isShared_1938_ = v_isSharedCheck_1942_;
goto v_resetjp_1936_;
}
v_resetjp_1936_:
{
lean_object* v___x_1940_; 
if (v_isShared_1938_ == 0)
{
v___x_1940_ = v___x_1937_;
goto v_reusejp_1939_;
}
else
{
lean_object* v_reuseFailAlloc_1941_; 
v_reuseFailAlloc_1941_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1941_, 0, v_a_1935_);
v___x_1940_ = v_reuseFailAlloc_1941_;
goto v_reusejp_1939_;
}
v_reusejp_1939_:
{
return v___x_1940_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_elabCasesType___lam__0___boxed(lean_object* v_a_1943_, lean_object* v_recursive_1944_, lean_object* v_allowSplit_1945_, lean_object* v___y_1946_, lean_object* v___y_1947_, lean_object* v___y_1948_, lean_object* v___y_1949_, lean_object* v___y_1950_, lean_object* v___y_1951_, lean_object* v___y_1952_, lean_object* v___y_1953_, lean_object* v___y_1954_){
_start:
{
uint8_t v_recursive_boxed_1955_; uint8_t v_allowSplit_boxed_1956_; lean_object* v_res_1957_; 
v_recursive_boxed_1955_ = lean_unbox(v_recursive_1944_);
v_allowSplit_boxed_1956_ = lean_unbox(v_allowSplit_1945_);
v_res_1957_ = lp_mathlib_Mathlib_Tactic_elabCasesType___lam__0(v_a_1943_, v_recursive_boxed_1955_, v_allowSplit_boxed_1956_, v___y_1946_, v___y_1947_, v___y_1948_, v___y_1949_, v___y_1950_, v___y_1951_, v___y_1952_, v___y_1953_);
lean_dec(v___y_1953_);
lean_dec_ref(v___y_1952_);
lean_dec(v___y_1951_);
lean_dec_ref(v___y_1950_);
lean_dec(v___y_1949_);
lean_dec_ref(v___y_1948_);
lean_dec(v___y_1947_);
lean_dec_ref(v___y_1946_);
return v_res_1957_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Mathlib_Tactic_elabCasesType_spec__0___redArg(size_t v_sz_1958_, size_t v_i_1959_, lean_object* v_bs_1960_, lean_object* v___y_1961_, lean_object* v___y_1962_){
_start:
{
uint8_t v___x_1964_; 
v___x_1964_ = lean_usize_dec_lt(v_i_1959_, v_sz_1958_);
if (v___x_1964_ == 0)
{
lean_object* v___x_1965_; 
v___x_1965_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1965_, 0, v_bs_1960_);
return v___x_1965_;
}
else
{
lean_object* v_v_1966_; lean_object* v___x_1967_; lean_object* v___x_1968_; 
v_v_1966_ = lean_array_uget_borrowed(v_bs_1960_, v_i_1959_);
v___x_1967_ = lean_box(0);
lean_inc(v_v_1966_);
v___x_1968_ = l_Lean_Elab_realizeGlobalConstNoOverloadWithInfo(v_v_1966_, v___x_1967_, v___y_1961_, v___y_1962_);
if (lean_obj_tag(v___x_1968_) == 0)
{
lean_object* v_a_1969_; lean_object* v___x_1970_; lean_object* v_bs_x27_1971_; size_t v___x_1972_; size_t v___x_1973_; lean_object* v___x_1974_; 
v_a_1969_ = lean_ctor_get(v___x_1968_, 0);
lean_inc(v_a_1969_);
lean_dec_ref_known(v___x_1968_, 1);
v___x_1970_ = lean_unsigned_to_nat(0u);
v_bs_x27_1971_ = lean_array_uset(v_bs_1960_, v_i_1959_, v___x_1970_);
v___x_1972_ = ((size_t)1ULL);
v___x_1973_ = lean_usize_add(v_i_1959_, v___x_1972_);
v___x_1974_ = lean_array_uset(v_bs_x27_1971_, v_i_1959_, v_a_1969_);
v_i_1959_ = v___x_1973_;
v_bs_1960_ = v___x_1974_;
goto _start;
}
else
{
lean_object* v_a_1976_; lean_object* v___x_1978_; uint8_t v_isShared_1979_; uint8_t v_isSharedCheck_1983_; 
lean_dec_ref(v_bs_1960_);
v_a_1976_ = lean_ctor_get(v___x_1968_, 0);
v_isSharedCheck_1983_ = !lean_is_exclusive(v___x_1968_);
if (v_isSharedCheck_1983_ == 0)
{
v___x_1978_ = v___x_1968_;
v_isShared_1979_ = v_isSharedCheck_1983_;
goto v_resetjp_1977_;
}
else
{
lean_inc(v_a_1976_);
lean_dec(v___x_1968_);
v___x_1978_ = lean_box(0);
v_isShared_1979_ = v_isSharedCheck_1983_;
goto v_resetjp_1977_;
}
v_resetjp_1977_:
{
lean_object* v___x_1981_; 
if (v_isShared_1979_ == 0)
{
v___x_1981_ = v___x_1978_;
goto v_reusejp_1980_;
}
else
{
lean_object* v_reuseFailAlloc_1982_; 
v_reuseFailAlloc_1982_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1982_, 0, v_a_1976_);
v___x_1981_ = v_reuseFailAlloc_1982_;
goto v_reusejp_1980_;
}
v_reusejp_1980_:
{
return v___x_1981_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Mathlib_Tactic_elabCasesType_spec__0___redArg___boxed(lean_object* v_sz_1984_, lean_object* v_i_1985_, lean_object* v_bs_1986_, lean_object* v___y_1987_, lean_object* v___y_1988_, lean_object* v___y_1989_){
_start:
{
size_t v_sz_boxed_1990_; size_t v_i_boxed_1991_; lean_object* v_res_1992_; 
v_sz_boxed_1990_ = lean_unbox_usize(v_sz_1984_);
lean_dec(v_sz_1984_);
v_i_boxed_1991_ = lean_unbox_usize(v_i_1985_);
lean_dec(v_i_1985_);
v_res_1992_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Mathlib_Tactic_elabCasesType_spec__0___redArg(v_sz_boxed_1990_, v_i_boxed_1991_, v_bs_1986_, v___y_1987_, v___y_1988_);
lean_dec(v___y_1988_);
lean_dec_ref(v___y_1987_);
return v_res_1992_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_elabCasesType(lean_object* v_heads_1993_, uint8_t v_recursive_1994_, uint8_t v_allowSplit_1995_, lean_object* v_a_1996_, lean_object* v_a_1997_, lean_object* v_a_1998_, lean_object* v_a_1999_, lean_object* v_a_2000_, lean_object* v_a_2001_, lean_object* v_a_2002_, lean_object* v_a_2003_){
_start:
{
size_t v_sz_2005_; size_t v___x_2006_; lean_object* v___x_2007_; 
v_sz_2005_ = lean_array_size(v_heads_1993_);
v___x_2006_ = ((size_t)0ULL);
v___x_2007_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Mathlib_Tactic_elabCasesType_spec__0___redArg(v_sz_2005_, v___x_2006_, v_heads_1993_, v_a_2002_, v_a_2003_);
if (lean_obj_tag(v___x_2007_) == 0)
{
lean_object* v_a_2008_; lean_object* v___x_2009_; lean_object* v___x_2010_; lean_object* v___f_2011_; lean_object* v___x_2012_; 
v_a_2008_ = lean_ctor_get(v___x_2007_, 0);
lean_inc(v_a_2008_);
lean_dec_ref_known(v___x_2007_, 1);
v___x_2009_ = lean_box(v_recursive_1994_);
v___x_2010_ = lean_box(v_allowSplit_1995_);
v___f_2011_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_Tactic_elabCasesType___lam__0___boxed), 12, 3);
lean_closure_set(v___f_2011_, 0, v_a_2008_);
lean_closure_set(v___f_2011_, 1, v___x_2009_);
lean_closure_set(v___f_2011_, 2, v___x_2010_);
v___x_2012_ = l_Lean_Elab_Tactic_withMainContext___redArg(v___f_2011_, v_a_1996_, v_a_1997_, v_a_1998_, v_a_1999_, v_a_2000_, v_a_2001_, v_a_2002_, v_a_2003_);
return v___x_2012_;
}
else
{
lean_object* v_a_2013_; lean_object* v___x_2015_; uint8_t v_isShared_2016_; uint8_t v_isSharedCheck_2020_; 
v_a_2013_ = lean_ctor_get(v___x_2007_, 0);
v_isSharedCheck_2020_ = !lean_is_exclusive(v___x_2007_);
if (v_isSharedCheck_2020_ == 0)
{
v___x_2015_ = v___x_2007_;
v_isShared_2016_ = v_isSharedCheck_2020_;
goto v_resetjp_2014_;
}
else
{
lean_inc(v_a_2013_);
lean_dec(v___x_2007_);
v___x_2015_ = lean_box(0);
v_isShared_2016_ = v_isSharedCheck_2020_;
goto v_resetjp_2014_;
}
v_resetjp_2014_:
{
lean_object* v___x_2018_; 
if (v_isShared_2016_ == 0)
{
v___x_2018_ = v___x_2015_;
goto v_reusejp_2017_;
}
else
{
lean_object* v_reuseFailAlloc_2019_; 
v_reuseFailAlloc_2019_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2019_, 0, v_a_2013_);
v___x_2018_ = v_reuseFailAlloc_2019_;
goto v_reusejp_2017_;
}
v_reusejp_2017_:
{
return v___x_2018_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_elabCasesType___boxed(lean_object* v_heads_2021_, lean_object* v_recursive_2022_, lean_object* v_allowSplit_2023_, lean_object* v_a_2024_, lean_object* v_a_2025_, lean_object* v_a_2026_, lean_object* v_a_2027_, lean_object* v_a_2028_, lean_object* v_a_2029_, lean_object* v_a_2030_, lean_object* v_a_2031_, lean_object* v_a_2032_){
_start:
{
uint8_t v_recursive_boxed_2033_; uint8_t v_allowSplit_boxed_2034_; lean_object* v_res_2035_; 
v_recursive_boxed_2033_ = lean_unbox(v_recursive_2022_);
v_allowSplit_boxed_2034_ = lean_unbox(v_allowSplit_2023_);
v_res_2035_ = lp_mathlib_Mathlib_Tactic_elabCasesType(v_heads_2021_, v_recursive_boxed_2033_, v_allowSplit_boxed_2034_, v_a_2024_, v_a_2025_, v_a_2026_, v_a_2027_, v_a_2028_, v_a_2029_, v_a_2030_, v_a_2031_);
lean_dec(v_a_2031_);
lean_dec_ref(v_a_2030_);
lean_dec(v_a_2029_);
lean_dec_ref(v_a_2028_);
lean_dec(v_a_2027_);
lean_dec_ref(v_a_2026_);
lean_dec(v_a_2025_);
lean_dec_ref(v_a_2024_);
return v_res_2035_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Mathlib_Tactic_elabCasesType_spec__0(size_t v_sz_2036_, size_t v_i_2037_, lean_object* v_bs_2038_, lean_object* v___y_2039_, lean_object* v___y_2040_, lean_object* v___y_2041_, lean_object* v___y_2042_, lean_object* v___y_2043_, lean_object* v___y_2044_, lean_object* v___y_2045_, lean_object* v___y_2046_){
_start:
{
lean_object* v___x_2048_; 
v___x_2048_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Mathlib_Tactic_elabCasesType_spec__0___redArg(v_sz_2036_, v_i_2037_, v_bs_2038_, v___y_2045_, v___y_2046_);
return v___x_2048_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Mathlib_Tactic_elabCasesType_spec__0___boxed(lean_object* v_sz_2049_, lean_object* v_i_2050_, lean_object* v_bs_2051_, lean_object* v___y_2052_, lean_object* v___y_2053_, lean_object* v___y_2054_, lean_object* v___y_2055_, lean_object* v___y_2056_, lean_object* v___y_2057_, lean_object* v___y_2058_, lean_object* v___y_2059_, lean_object* v___y_2060_){
_start:
{
size_t v_sz_boxed_2061_; size_t v_i_boxed_2062_; lean_object* v_res_2063_; 
v_sz_boxed_2061_ = lean_unbox_usize(v_sz_2049_);
lean_dec(v_sz_2049_);
v_i_boxed_2062_ = lean_unbox_usize(v_i_2050_);
lean_dec(v_i_2050_);
v_res_2063_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Mathlib_Tactic_elabCasesType_spec__0(v_sz_boxed_2061_, v_i_boxed_2062_, v_bs_2051_, v___y_2052_, v___y_2053_, v___y_2054_, v___y_2055_, v___y_2056_, v___y_2057_, v___y_2058_, v___y_2059_);
lean_dec(v___y_2059_);
lean_dec_ref(v___y_2058_);
lean_dec(v___y_2057_);
lean_dec_ref(v___y_2056_);
lean_dec(v___y_2055_);
lean_dec_ref(v___y_2054_);
lean_dec(v___y_2053_);
lean_dec_ref(v___y_2052_);
return v_res_2063_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__CasesM______elabRules__Mathlib__Tactic__casesType__1(lean_object* v_x_2110_, lean_object* v_a_2111_, lean_object* v_a_2112_, lean_object* v_a_2113_, lean_object* v_a_2114_, lean_object* v_a_2115_, lean_object* v_a_2116_, lean_object* v_a_2117_, lean_object* v_a_2118_){
_start:
{
lean_object* v___x_2120_; uint8_t v___x_2121_; 
v___x_2120_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_casesType___closed__1));
lean_inc(v_x_2110_);
v___x_2121_ = l_Lean_Syntax_isOfKind(v_x_2110_, v___x_2120_);
if (v___x_2121_ == 0)
{
lean_object* v___x_2122_; 
lean_dec(v_x_2110_);
v___x_2122_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic___aux__Mathlib__Tactic__CasesM______elabRules__Mathlib__Tactic__casesM__1_spec__0___redArg();
return v___x_2122_;
}
else
{
lean_object* v___x_2123_; lean_object* v___x_2124_; lean_object* v___x_2125_; lean_object* v___x_2126_; lean_object* v_heads_2127_; lean_object* v___x_2128_; 
v___x_2123_ = lean_unsigned_to_nat(1u);
v___x_2124_ = l_Lean_Syntax_getArg(v_x_2110_, v___x_2123_);
v___x_2125_ = lean_unsigned_to_nat(2u);
v___x_2126_ = l_Lean_Syntax_getArg(v_x_2110_, v___x_2125_);
lean_dec(v_x_2110_);
v_heads_2127_ = l_Lean_Syntax_getArgs(v___x_2126_);
lean_dec(v___x_2126_);
v___x_2128_ = l_Lean_Syntax_getOptional_x3f(v___x_2124_);
lean_dec(v___x_2124_);
if (lean_obj_tag(v___x_2128_) == 0)
{
uint8_t v___x_2129_; lean_object* v___x_2130_; 
v___x_2129_ = 0;
v___x_2130_ = lp_mathlib_Mathlib_Tactic_elabCasesType(v_heads_2127_, v___x_2129_, v___x_2121_, v_a_2111_, v_a_2112_, v_a_2113_, v_a_2114_, v_a_2115_, v_a_2116_, v_a_2117_, v_a_2118_);
return v___x_2130_;
}
else
{
lean_object* v___x_2131_; 
lean_dec_ref_known(v___x_2128_, 1);
v___x_2131_ = lp_mathlib_Mathlib_Tactic_elabCasesType(v_heads_2127_, v___x_2121_, v___x_2121_, v_a_2111_, v_a_2112_, v_a_2113_, v_a_2114_, v_a_2115_, v_a_2116_, v_a_2117_, v_a_2118_);
return v___x_2131_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__CasesM______elabRules__Mathlib__Tactic__casesType__1___boxed(lean_object* v_x_2132_, lean_object* v_a_2133_, lean_object* v_a_2134_, lean_object* v_a_2135_, lean_object* v_a_2136_, lean_object* v_a_2137_, lean_object* v_a_2138_, lean_object* v_a_2139_, lean_object* v_a_2140_, lean_object* v_a_2141_){
_start:
{
lean_object* v_res_2142_; 
v_res_2142_ = lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__CasesM______elabRules__Mathlib__Tactic__casesType__1(v_x_2132_, v_a_2133_, v_a_2134_, v_a_2135_, v_a_2136_, v_a_2137_, v_a_2138_, v_a_2139_, v_a_2140_);
lean_dec(v_a_2140_);
lean_dec_ref(v_a_2139_);
lean_dec(v_a_2138_);
lean_dec_ref(v_a_2137_);
lean_dec(v_a_2136_);
lean_dec_ref(v_a_2135_);
lean_dec(v_a_2134_);
lean_dec_ref(v_a_2133_);
return v_res_2142_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__CasesM______elabRules__Mathlib__Tactic__casesType_x21__1(lean_object* v_x_2165_, lean_object* v_a_2166_, lean_object* v_a_2167_, lean_object* v_a_2168_, lean_object* v_a_2169_, lean_object* v_a_2170_, lean_object* v_a_2171_, lean_object* v_a_2172_, lean_object* v_a_2173_){
_start:
{
lean_object* v___x_2175_; uint8_t v___x_2176_; 
v___x_2175_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_casesType_x21___closed__1));
lean_inc(v_x_2165_);
v___x_2176_ = l_Lean_Syntax_isOfKind(v_x_2165_, v___x_2175_);
if (v___x_2176_ == 0)
{
lean_object* v___x_2177_; 
lean_dec(v_x_2165_);
v___x_2177_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic___aux__Mathlib__Tactic__CasesM______elabRules__Mathlib__Tactic__casesM__1_spec__0___redArg();
return v___x_2177_;
}
else
{
lean_object* v___x_2178_; lean_object* v___x_2179_; lean_object* v___x_2180_; lean_object* v___x_2181_; lean_object* v_heads_2182_; uint8_t v___y_2184_; lean_object* v___x_2187_; 
v___x_2178_ = lean_unsigned_to_nat(1u);
v___x_2179_ = l_Lean_Syntax_getArg(v_x_2165_, v___x_2178_);
v___x_2180_ = lean_unsigned_to_nat(2u);
v___x_2181_ = l_Lean_Syntax_getArg(v_x_2165_, v___x_2180_);
lean_dec(v_x_2165_);
v_heads_2182_ = l_Lean_Syntax_getArgs(v___x_2181_);
lean_dec(v___x_2181_);
v___x_2187_ = l_Lean_Syntax_getOptional_x3f(v___x_2179_);
lean_dec(v___x_2179_);
if (lean_obj_tag(v___x_2187_) == 0)
{
uint8_t v___x_2188_; 
v___x_2188_ = 0;
v___y_2184_ = v___x_2188_;
goto v___jp_2183_;
}
else
{
lean_dec_ref_known(v___x_2187_, 1);
v___y_2184_ = v___x_2176_;
goto v___jp_2183_;
}
v___jp_2183_:
{
uint8_t v___x_2185_; lean_object* v___x_2186_; 
v___x_2185_ = 0;
v___x_2186_ = lp_mathlib_Mathlib_Tactic_elabCasesType(v_heads_2182_, v___y_2184_, v___x_2185_, v_a_2166_, v_a_2167_, v_a_2168_, v_a_2169_, v_a_2170_, v_a_2171_, v_a_2172_, v_a_2173_);
return v___x_2186_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__CasesM______elabRules__Mathlib__Tactic__casesType_x21__1___boxed(lean_object* v_x_2189_, lean_object* v_a_2190_, lean_object* v_a_2191_, lean_object* v_a_2192_, lean_object* v_a_2193_, lean_object* v_a_2194_, lean_object* v_a_2195_, lean_object* v_a_2196_, lean_object* v_a_2197_, lean_object* v_a_2198_){
_start:
{
lean_object* v_res_2199_; 
v_res_2199_ = lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__CasesM______elabRules__Mathlib__Tactic__casesType_x21__1(v_x_2189_, v_a_2190_, v_a_2191_, v_a_2192_, v_a_2193_, v_a_2194_, v_a_2195_, v_a_2196_, v_a_2197_);
lean_dec(v_a_2197_);
lean_dec_ref(v_a_2196_);
lean_dec(v_a_2195_);
lean_dec_ref(v_a_2194_);
lean_dec(v_a_2193_);
lean_dec_ref(v_a_2192_);
lean_dec(v_a_2191_);
lean_dec_ref(v_a_2190_);
return v_res_2199_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_forIn_x27_loop___at___00__private_Mathlib_Tactic_CasesM_0__Mathlib_Tactic_constructorMatching_go_spec__0___redArg(lean_object* v_matcher_2200_, lean_object* v_as_x27_2201_, lean_object* v_b_2202_, lean_object* v___y_2203_, lean_object* v___y_2204_, lean_object* v___y_2205_, lean_object* v___y_2206_){
_start:
{
if (lean_obj_tag(v_as_x27_2201_) == 0)
{
lean_object* v___x_2208_; 
lean_dec_ref(v_matcher_2200_);
v___x_2208_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_2208_, 0, v_b_2202_);
return v___x_2208_;
}
else
{
lean_object* v_head_2209_; lean_object* v_tail_2210_; lean_object* v___x_2211_; 
v_head_2209_ = lean_ctor_get(v_as_x27_2201_, 0);
v_tail_2210_ = lean_ctor_get(v_as_x27_2201_, 1);
lean_inc(v_head_2209_);
lean_inc_ref(v_matcher_2200_);
v___x_2211_ = lp_mathlib___private_Mathlib_Tactic_CasesM_0__Mathlib_Tactic_constructorMatching_go(v_matcher_2200_, v_head_2209_, v_b_2202_, v___y_2203_, v___y_2204_, v___y_2205_, v___y_2206_);
if (lean_obj_tag(v___x_2211_) == 0)
{
lean_object* v_a_2212_; 
v_a_2212_ = lean_ctor_get(v___x_2211_, 0);
lean_inc(v_a_2212_);
lean_dec_ref_known(v___x_2211_, 1);
v_as_x27_2201_ = v_tail_2210_;
v_b_2202_ = v_a_2212_;
goto _start;
}
else
{
lean_dec_ref(v_matcher_2200_);
return v___x_2211_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_CasesM_0__Mathlib_Tactic_constructorMatching_go___lam__0(lean_object* v_g_2214_, lean_object* v_matcher_2215_, lean_object* v_acc_2216_, lean_object* v___y_2217_, lean_object* v___y_2218_, lean_object* v___y_2219_, lean_object* v___y_2220_){
_start:
{
lean_object* v___x_2222_; 
lean_inc(v_g_2214_);
v___x_2222_ = l_Lean_MVarId_getType(v_g_2214_, v___y_2217_, v___y_2218_, v___y_2219_, v___y_2220_);
if (lean_obj_tag(v___x_2222_) == 0)
{
lean_object* v_a_2223_; lean_object* v___x_2224_; 
v_a_2223_ = lean_ctor_get(v___x_2222_, 0);
lean_inc(v_a_2223_);
lean_dec_ref_known(v___x_2222_, 1);
lean_inc_ref(v_matcher_2215_);
lean_inc(v___y_2220_);
lean_inc_ref(v___y_2219_);
lean_inc(v___y_2218_);
lean_inc_ref(v___y_2217_);
v___x_2224_ = lean_apply_6(v_matcher_2215_, v_a_2223_, v___y_2217_, v___y_2218_, v___y_2219_, v___y_2220_, lean_box(0));
if (lean_obj_tag(v___x_2224_) == 0)
{
lean_object* v_a_2225_; lean_object* v___x_2227_; uint8_t v_isShared_2228_; uint8_t v_isSharedCheck_2250_; 
v_a_2225_ = lean_ctor_get(v___x_2224_, 0);
v_isSharedCheck_2250_ = !lean_is_exclusive(v___x_2224_);
if (v_isSharedCheck_2250_ == 0)
{
v___x_2227_ = v___x_2224_;
v_isShared_2228_ = v_isSharedCheck_2250_;
goto v_resetjp_2226_;
}
else
{
lean_inc(v_a_2225_);
lean_dec(v___x_2224_);
v___x_2227_ = lean_box(0);
v_isShared_2228_ = v_isSharedCheck_2250_;
goto v_resetjp_2226_;
}
v_resetjp_2226_:
{
uint8_t v___x_2229_; 
v___x_2229_ = lean_unbox(v_a_2225_);
if (v___x_2229_ == 0)
{
lean_object* v___x_2230_; lean_object* v___x_2232_; 
lean_dec(v_a_2225_);
lean_dec(v___y_2220_);
lean_dec_ref(v___y_2219_);
lean_dec(v___y_2218_);
lean_dec_ref(v___y_2217_);
lean_dec_ref(v_matcher_2215_);
v___x_2230_ = lean_array_push(v_acc_2216_, v_g_2214_);
if (v_isShared_2228_ == 0)
{
lean_ctor_set(v___x_2227_, 0, v___x_2230_);
v___x_2232_ = v___x_2227_;
goto v_reusejp_2231_;
}
else
{
lean_object* v_reuseFailAlloc_2233_; 
v_reuseFailAlloc_2233_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2233_, 0, v___x_2230_);
v___x_2232_ = v_reuseFailAlloc_2233_;
goto v_reusejp_2231_;
}
v_reusejp_2231_:
{
return v___x_2232_;
}
}
else
{
uint8_t v___x_2234_; uint8_t v___x_2235_; lean_object* v___x_2236_; uint8_t v___x_2237_; uint8_t v___x_2238_; lean_object* v___x_2239_; 
lean_del_object(v___x_2227_);
v___x_2234_ = 0;
v___x_2235_ = 0;
v___x_2236_ = lean_alloc_ctor(0, 0, 4);
lean_ctor_set_uint8(v___x_2236_, 0, v___x_2234_);
v___x_2237_ = lean_unbox(v_a_2225_);
lean_ctor_set_uint8(v___x_2236_, 1, v___x_2237_);
lean_ctor_set_uint8(v___x_2236_, 2, v___x_2235_);
v___x_2238_ = lean_unbox(v_a_2225_);
lean_dec(v_a_2225_);
lean_ctor_set_uint8(v___x_2236_, 3, v___x_2238_);
v___x_2239_ = l_Lean_MVarId_constructor(v_g_2214_, v___x_2236_, v___y_2217_, v___y_2218_, v___y_2219_, v___y_2220_);
if (lean_obj_tag(v___x_2239_) == 0)
{
lean_object* v_a_2240_; lean_object* v___x_2241_; 
v_a_2240_ = lean_ctor_get(v___x_2239_, 0);
lean_inc(v_a_2240_);
lean_dec_ref_known(v___x_2239_, 1);
v___x_2241_ = lp_mathlib_List_forIn_x27_loop___at___00__private_Mathlib_Tactic_CasesM_0__Mathlib_Tactic_constructorMatching_go_spec__0___redArg(v_matcher_2215_, v_a_2240_, v_acc_2216_, v___y_2217_, v___y_2218_, v___y_2219_, v___y_2220_);
lean_dec(v___y_2220_);
lean_dec_ref(v___y_2219_);
lean_dec(v___y_2218_);
lean_dec_ref(v___y_2217_);
lean_dec(v_a_2240_);
return v___x_2241_;
}
else
{
lean_object* v_a_2242_; lean_object* v___x_2244_; uint8_t v_isShared_2245_; uint8_t v_isSharedCheck_2249_; 
lean_dec(v___y_2220_);
lean_dec_ref(v___y_2219_);
lean_dec(v___y_2218_);
lean_dec_ref(v___y_2217_);
lean_dec_ref(v_acc_2216_);
lean_dec_ref(v_matcher_2215_);
v_a_2242_ = lean_ctor_get(v___x_2239_, 0);
v_isSharedCheck_2249_ = !lean_is_exclusive(v___x_2239_);
if (v_isSharedCheck_2249_ == 0)
{
v___x_2244_ = v___x_2239_;
v_isShared_2245_ = v_isSharedCheck_2249_;
goto v_resetjp_2243_;
}
else
{
lean_inc(v_a_2242_);
lean_dec(v___x_2239_);
v___x_2244_ = lean_box(0);
v_isShared_2245_ = v_isSharedCheck_2249_;
goto v_resetjp_2243_;
}
v_resetjp_2243_:
{
lean_object* v___x_2247_; 
if (v_isShared_2245_ == 0)
{
v___x_2247_ = v___x_2244_;
goto v_reusejp_2246_;
}
else
{
lean_object* v_reuseFailAlloc_2248_; 
v_reuseFailAlloc_2248_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2248_, 0, v_a_2242_);
v___x_2247_ = v_reuseFailAlloc_2248_;
goto v_reusejp_2246_;
}
v_reusejp_2246_:
{
return v___x_2247_;
}
}
}
}
}
}
else
{
lean_object* v_a_2251_; lean_object* v___x_2253_; uint8_t v_isShared_2254_; uint8_t v_isSharedCheck_2258_; 
lean_dec(v___y_2220_);
lean_dec_ref(v___y_2219_);
lean_dec(v___y_2218_);
lean_dec_ref(v___y_2217_);
lean_dec_ref(v_acc_2216_);
lean_dec_ref(v_matcher_2215_);
lean_dec(v_g_2214_);
v_a_2251_ = lean_ctor_get(v___x_2224_, 0);
v_isSharedCheck_2258_ = !lean_is_exclusive(v___x_2224_);
if (v_isSharedCheck_2258_ == 0)
{
v___x_2253_ = v___x_2224_;
v_isShared_2254_ = v_isSharedCheck_2258_;
goto v_resetjp_2252_;
}
else
{
lean_inc(v_a_2251_);
lean_dec(v___x_2224_);
v___x_2253_ = lean_box(0);
v_isShared_2254_ = v_isSharedCheck_2258_;
goto v_resetjp_2252_;
}
v_resetjp_2252_:
{
lean_object* v___x_2256_; 
if (v_isShared_2254_ == 0)
{
v___x_2256_ = v___x_2253_;
goto v_reusejp_2255_;
}
else
{
lean_object* v_reuseFailAlloc_2257_; 
v_reuseFailAlloc_2257_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2257_, 0, v_a_2251_);
v___x_2256_ = v_reuseFailAlloc_2257_;
goto v_reusejp_2255_;
}
v_reusejp_2255_:
{
return v___x_2256_;
}
}
}
}
else
{
lean_object* v_a_2259_; lean_object* v___x_2261_; uint8_t v_isShared_2262_; uint8_t v_isSharedCheck_2266_; 
lean_dec(v___y_2220_);
lean_dec_ref(v___y_2219_);
lean_dec(v___y_2218_);
lean_dec_ref(v___y_2217_);
lean_dec_ref(v_acc_2216_);
lean_dec_ref(v_matcher_2215_);
lean_dec(v_g_2214_);
v_a_2259_ = lean_ctor_get(v___x_2222_, 0);
v_isSharedCheck_2266_ = !lean_is_exclusive(v___x_2222_);
if (v_isSharedCheck_2266_ == 0)
{
v___x_2261_ = v___x_2222_;
v_isShared_2262_ = v_isSharedCheck_2266_;
goto v_resetjp_2260_;
}
else
{
lean_inc(v_a_2259_);
lean_dec(v___x_2222_);
v___x_2261_ = lean_box(0);
v_isShared_2262_ = v_isSharedCheck_2266_;
goto v_resetjp_2260_;
}
v_resetjp_2260_:
{
lean_object* v___x_2264_; 
if (v_isShared_2262_ == 0)
{
v___x_2264_ = v___x_2261_;
goto v_reusejp_2263_;
}
else
{
lean_object* v_reuseFailAlloc_2265_; 
v_reuseFailAlloc_2265_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2265_, 0, v_a_2259_);
v___x_2264_ = v_reuseFailAlloc_2265_;
goto v_reusejp_2263_;
}
v_reusejp_2263_:
{
return v___x_2264_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_CasesM_0__Mathlib_Tactic_constructorMatching_go___lam__0___boxed(lean_object* v_g_2267_, lean_object* v_matcher_2268_, lean_object* v_acc_2269_, lean_object* v___y_2270_, lean_object* v___y_2271_, lean_object* v___y_2272_, lean_object* v___y_2273_, lean_object* v___y_2274_){
_start:
{
lean_object* v_res_2275_; 
v_res_2275_ = lp_mathlib___private_Mathlib_Tactic_CasesM_0__Mathlib_Tactic_constructorMatching_go___lam__0(v_g_2267_, v_matcher_2268_, v_acc_2269_, v___y_2270_, v___y_2271_, v___y_2272_, v___y_2273_);
return v_res_2275_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_CasesM_0__Mathlib_Tactic_constructorMatching_go(lean_object* v_matcher_2276_, lean_object* v_g_2277_, lean_object* v_acc_2278_, lean_object* v_a_2279_, lean_object* v_a_2280_, lean_object* v_a_2281_, lean_object* v_a_2282_){
_start:
{
lean_object* v___f_2284_; lean_object* v___x_2285_; 
lean_inc(v_g_2277_);
v___f_2284_ = lean_alloc_closure((void*)(lp_mathlib___private_Mathlib_Tactic_CasesM_0__Mathlib_Tactic_constructorMatching_go___lam__0___boxed), 8, 3);
lean_closure_set(v___f_2284_, 0, v_g_2277_);
lean_closure_set(v___f_2284_, 1, v_matcher_2276_);
lean_closure_set(v___f_2284_, 2, v_acc_2278_);
v___x_2285_ = lp_mathlib_Lean_MVarId_withContext___at___00__private_Mathlib_Tactic_CasesM_0__Lean_MVarId_casesMatching_go_spec__2___redArg(v_g_2277_, v___f_2284_, v_a_2279_, v_a_2280_, v_a_2281_, v_a_2282_);
return v___x_2285_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_CasesM_0__Mathlib_Tactic_constructorMatching_go___boxed(lean_object* v_matcher_2286_, lean_object* v_g_2287_, lean_object* v_acc_2288_, lean_object* v_a_2289_, lean_object* v_a_2290_, lean_object* v_a_2291_, lean_object* v_a_2292_, lean_object* v_a_2293_){
_start:
{
lean_object* v_res_2294_; 
v_res_2294_ = lp_mathlib___private_Mathlib_Tactic_CasesM_0__Mathlib_Tactic_constructorMatching_go(v_matcher_2286_, v_g_2287_, v_acc_2288_, v_a_2289_, v_a_2290_, v_a_2291_, v_a_2292_);
lean_dec(v_a_2292_);
lean_dec_ref(v_a_2291_);
lean_dec(v_a_2290_);
lean_dec_ref(v_a_2289_);
return v_res_2294_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_forIn_x27_loop___at___00__private_Mathlib_Tactic_CasesM_0__Mathlib_Tactic_constructorMatching_go_spec__0___redArg___boxed(lean_object* v_matcher_2295_, lean_object* v_as_x27_2296_, lean_object* v_b_2297_, lean_object* v___y_2298_, lean_object* v___y_2299_, lean_object* v___y_2300_, lean_object* v___y_2301_, lean_object* v___y_2302_){
_start:
{
lean_object* v_res_2303_; 
v_res_2303_ = lp_mathlib_List_forIn_x27_loop___at___00__private_Mathlib_Tactic_CasesM_0__Mathlib_Tactic_constructorMatching_go_spec__0___redArg(v_matcher_2295_, v_as_x27_2296_, v_b_2297_, v___y_2298_, v___y_2299_, v___y_2300_, v___y_2301_);
lean_dec(v___y_2301_);
lean_dec_ref(v___y_2300_);
lean_dec(v___y_2299_);
lean_dec_ref(v___y_2298_);
lean_dec(v_as_x27_2296_);
return v_res_2303_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_forIn_x27_loop___at___00__private_Mathlib_Tactic_CasesM_0__Mathlib_Tactic_constructorMatching_go_spec__0(lean_object* v_matcher_2304_, lean_object* v_as_2305_, lean_object* v_as_x27_2306_, lean_object* v_b_2307_, lean_object* v_a_2308_, lean_object* v___y_2309_, lean_object* v___y_2310_, lean_object* v___y_2311_, lean_object* v___y_2312_){
_start:
{
lean_object* v___x_2314_; 
v___x_2314_ = lp_mathlib_List_forIn_x27_loop___at___00__private_Mathlib_Tactic_CasesM_0__Mathlib_Tactic_constructorMatching_go_spec__0___redArg(v_matcher_2304_, v_as_x27_2306_, v_b_2307_, v___y_2309_, v___y_2310_, v___y_2311_, v___y_2312_);
return v___x_2314_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_forIn_x27_loop___at___00__private_Mathlib_Tactic_CasesM_0__Mathlib_Tactic_constructorMatching_go_spec__0___boxed(lean_object* v_matcher_2315_, lean_object* v_as_2316_, lean_object* v_as_x27_2317_, lean_object* v_b_2318_, lean_object* v_a_2319_, lean_object* v___y_2320_, lean_object* v___y_2321_, lean_object* v___y_2322_, lean_object* v___y_2323_, lean_object* v___y_2324_){
_start:
{
lean_object* v_res_2325_; 
v_res_2325_ = lp_mathlib_List_forIn_x27_loop___at___00__private_Mathlib_Tactic_CasesM_0__Mathlib_Tactic_constructorMatching_go_spec__0(v_matcher_2315_, v_as_2316_, v_as_x27_2317_, v_b_2318_, v_a_2319_, v___y_2320_, v___y_2321_, v___y_2322_, v___y_2323_);
lean_dec(v___y_2323_);
lean_dec_ref(v___y_2322_);
lean_dec(v___y_2321_);
lean_dec_ref(v___y_2320_);
lean_dec(v_as_x27_2317_);
lean_dec(v_as_2316_);
return v_res_2325_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_constructorMatching___lam__0(lean_object* v_g_2326_, lean_object* v_matcher_2327_, uint8_t v___x_2328_, uint8_t v_recursive_2329_, lean_object* v___y_2330_, lean_object* v___y_2331_, lean_object* v___y_2332_, lean_object* v___y_2333_){
_start:
{
lean_object* v___x_2335_; 
lean_inc(v_g_2326_);
v___x_2335_ = l_Lean_MVarId_getType(v_g_2326_, v___y_2330_, v___y_2331_, v___y_2332_, v___y_2333_);
if (lean_obj_tag(v___x_2335_) == 0)
{
lean_object* v_a_2336_; lean_object* v___x_2337_; 
v_a_2336_ = lean_ctor_get(v___x_2335_, 0);
lean_inc(v_a_2336_);
lean_dec_ref_known(v___x_2335_, 1);
lean_inc(v___y_2333_);
lean_inc_ref(v___y_2332_);
lean_inc(v___y_2331_);
lean_inc_ref(v___y_2330_);
v___x_2337_ = lean_apply_6(v_matcher_2327_, v_a_2336_, v___y_2330_, v___y_2331_, v___y_2332_, v___y_2333_, lean_box(0));
if (lean_obj_tag(v___x_2337_) == 0)
{
lean_object* v_a_2338_; lean_object* v___x_2340_; uint8_t v_isShared_2341_; uint8_t v_isSharedCheck_2351_; 
v_a_2338_ = lean_ctor_get(v___x_2337_, 0);
v_isSharedCheck_2351_ = !lean_is_exclusive(v___x_2337_);
if (v_isSharedCheck_2351_ == 0)
{
v___x_2340_ = v___x_2337_;
v_isShared_2341_ = v_isSharedCheck_2351_;
goto v_resetjp_2339_;
}
else
{
lean_inc(v_a_2338_);
lean_dec(v___x_2337_);
v___x_2340_ = lean_box(0);
v_isShared_2341_ = v_isSharedCheck_2351_;
goto v_resetjp_2339_;
}
v_resetjp_2339_:
{
uint8_t v___x_2342_; 
v___x_2342_ = lean_unbox(v_a_2338_);
lean_dec(v_a_2338_);
if (v___x_2342_ == 0)
{
lean_object* v___x_2343_; lean_object* v___x_2344_; lean_object* v___x_2346_; 
lean_dec(v___y_2333_);
lean_dec_ref(v___y_2332_);
lean_dec(v___y_2331_);
lean_dec_ref(v___y_2330_);
v___x_2343_ = lean_box(0);
v___x_2344_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_2344_, 0, v_g_2326_);
lean_ctor_set(v___x_2344_, 1, v___x_2343_);
if (v_isShared_2341_ == 0)
{
lean_ctor_set(v___x_2340_, 0, v___x_2344_);
v___x_2346_ = v___x_2340_;
goto v_reusejp_2345_;
}
else
{
lean_object* v_reuseFailAlloc_2347_; 
v_reuseFailAlloc_2347_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2347_, 0, v___x_2344_);
v___x_2346_ = v_reuseFailAlloc_2347_;
goto v_reusejp_2345_;
}
v_reusejp_2345_:
{
return v___x_2346_;
}
}
else
{
uint8_t v___x_2348_; lean_object* v___x_2349_; lean_object* v___x_2350_; 
lean_del_object(v___x_2340_);
v___x_2348_ = 0;
v___x_2349_ = lean_alloc_ctor(0, 0, 4);
lean_ctor_set_uint8(v___x_2349_, 0, v___x_2348_);
lean_ctor_set_uint8(v___x_2349_, 1, v___x_2328_);
lean_ctor_set_uint8(v___x_2349_, 2, v_recursive_2329_);
lean_ctor_set_uint8(v___x_2349_, 3, v___x_2328_);
v___x_2350_ = l_Lean_MVarId_constructor(v_g_2326_, v___x_2349_, v___y_2330_, v___y_2331_, v___y_2332_, v___y_2333_);
lean_dec(v___y_2333_);
lean_dec_ref(v___y_2332_);
lean_dec(v___y_2331_);
lean_dec_ref(v___y_2330_);
return v___x_2350_;
}
}
}
else
{
lean_object* v_a_2352_; lean_object* v___x_2354_; uint8_t v_isShared_2355_; uint8_t v_isSharedCheck_2359_; 
lean_dec(v___y_2333_);
lean_dec_ref(v___y_2332_);
lean_dec(v___y_2331_);
lean_dec_ref(v___y_2330_);
lean_dec(v_g_2326_);
v_a_2352_ = lean_ctor_get(v___x_2337_, 0);
v_isSharedCheck_2359_ = !lean_is_exclusive(v___x_2337_);
if (v_isSharedCheck_2359_ == 0)
{
v___x_2354_ = v___x_2337_;
v_isShared_2355_ = v_isSharedCheck_2359_;
goto v_resetjp_2353_;
}
else
{
lean_inc(v_a_2352_);
lean_dec(v___x_2337_);
v___x_2354_ = lean_box(0);
v_isShared_2355_ = v_isSharedCheck_2359_;
goto v_resetjp_2353_;
}
v_resetjp_2353_:
{
lean_object* v___x_2357_; 
if (v_isShared_2355_ == 0)
{
v___x_2357_ = v___x_2354_;
goto v_reusejp_2356_;
}
else
{
lean_object* v_reuseFailAlloc_2358_; 
v_reuseFailAlloc_2358_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2358_, 0, v_a_2352_);
v___x_2357_ = v_reuseFailAlloc_2358_;
goto v_reusejp_2356_;
}
v_reusejp_2356_:
{
return v___x_2357_;
}
}
}
}
else
{
lean_object* v_a_2360_; lean_object* v___x_2362_; uint8_t v_isShared_2363_; uint8_t v_isSharedCheck_2367_; 
lean_dec(v___y_2333_);
lean_dec_ref(v___y_2332_);
lean_dec(v___y_2331_);
lean_dec_ref(v___y_2330_);
lean_dec_ref(v_matcher_2327_);
lean_dec(v_g_2326_);
v_a_2360_ = lean_ctor_get(v___x_2335_, 0);
v_isSharedCheck_2367_ = !lean_is_exclusive(v___x_2335_);
if (v_isSharedCheck_2367_ == 0)
{
v___x_2362_ = v___x_2335_;
v_isShared_2363_ = v_isSharedCheck_2367_;
goto v_resetjp_2361_;
}
else
{
lean_inc(v_a_2360_);
lean_dec(v___x_2335_);
v___x_2362_ = lean_box(0);
v_isShared_2363_ = v_isSharedCheck_2367_;
goto v_resetjp_2361_;
}
v_resetjp_2361_:
{
lean_object* v___x_2365_; 
if (v_isShared_2363_ == 0)
{
v___x_2365_ = v___x_2362_;
goto v_reusejp_2364_;
}
else
{
lean_object* v_reuseFailAlloc_2366_; 
v_reuseFailAlloc_2366_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2366_, 0, v_a_2360_);
v___x_2365_ = v_reuseFailAlloc_2366_;
goto v_reusejp_2364_;
}
v_reusejp_2364_:
{
return v___x_2365_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_constructorMatching___lam__0___boxed(lean_object* v_g_2368_, lean_object* v_matcher_2369_, lean_object* v___x_2370_, lean_object* v_recursive_2371_, lean_object* v___y_2372_, lean_object* v___y_2373_, lean_object* v___y_2374_, lean_object* v___y_2375_, lean_object* v___y_2376_){
_start:
{
uint8_t v___x_932__boxed_2377_; uint8_t v_recursive_boxed_2378_; lean_object* v_res_2379_; 
v___x_932__boxed_2377_ = lean_unbox(v___x_2370_);
v_recursive_boxed_2378_ = lean_unbox(v_recursive_2371_);
v_res_2379_ = lp_mathlib_Mathlib_Tactic_constructorMatching___lam__0(v_g_2368_, v_matcher_2369_, v___x_932__boxed_2377_, v_recursive_boxed_2378_, v___y_2372_, v___y_2373_, v___y_2374_, v___y_2375_);
return v_res_2379_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_constructorMatching(lean_object* v_g_2380_, lean_object* v_matcher_2381_, uint8_t v_recursive_2382_, uint8_t v_throwOnNoMatch_2383_, lean_object* v_a_2384_, lean_object* v_a_2385_, lean_object* v_a_2386_, lean_object* v_a_2387_){
_start:
{
lean_object* v___y_2390_; lean_object* v_a_2391_; 
if (v_recursive_2382_ == 0)
{
uint8_t v___x_2397_; lean_object* v___x_2398_; lean_object* v___x_2399_; lean_object* v___f_2400_; lean_object* v___x_2401_; 
v___x_2397_ = 1;
v___x_2398_ = lean_box(v___x_2397_);
v___x_2399_ = lean_box(v_recursive_2382_);
lean_inc_n(v_g_2380_, 2);
v___f_2400_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_Tactic_constructorMatching___lam__0___boxed), 9, 4);
lean_closure_set(v___f_2400_, 0, v_g_2380_);
lean_closure_set(v___f_2400_, 1, v_matcher_2381_);
lean_closure_set(v___f_2400_, 2, v___x_2398_);
lean_closure_set(v___f_2400_, 3, v___x_2399_);
v___x_2401_ = lp_mathlib_Lean_MVarId_withContext___at___00__private_Mathlib_Tactic_CasesM_0__Lean_MVarId_casesMatching_go_spec__2___redArg(v_g_2380_, v___f_2400_, v_a_2384_, v_a_2385_, v_a_2386_, v_a_2387_);
if (lean_obj_tag(v___x_2401_) == 0)
{
lean_object* v_a_2402_; 
v_a_2402_ = lean_ctor_get(v___x_2401_, 0);
lean_inc(v_a_2402_);
v___y_2390_ = v___x_2401_;
v_a_2391_ = v_a_2402_;
goto v___jp_2389_;
}
else
{
lean_dec(v_g_2380_);
return v___x_2401_;
}
}
else
{
lean_object* v___x_2403_; lean_object* v___x_2404_; 
v___x_2403_ = ((lean_object*)(lp_mathlib_Lean_MVarId_casesMatching___closed__0));
lean_inc(v_g_2380_);
v___x_2404_ = lp_mathlib___private_Mathlib_Tactic_CasesM_0__Mathlib_Tactic_constructorMatching_go(v_matcher_2381_, v_g_2380_, v___x_2403_, v_a_2384_, v_a_2385_, v_a_2386_, v_a_2387_);
if (lean_obj_tag(v___x_2404_) == 0)
{
lean_object* v_a_2405_; lean_object* v___x_2407_; uint8_t v_isShared_2408_; uint8_t v_isSharedCheck_2413_; 
v_a_2405_ = lean_ctor_get(v___x_2404_, 0);
v_isSharedCheck_2413_ = !lean_is_exclusive(v___x_2404_);
if (v_isSharedCheck_2413_ == 0)
{
v___x_2407_ = v___x_2404_;
v_isShared_2408_ = v_isSharedCheck_2413_;
goto v_resetjp_2406_;
}
else
{
lean_inc(v_a_2405_);
lean_dec(v___x_2404_);
v___x_2407_ = lean_box(0);
v_isShared_2408_ = v_isSharedCheck_2413_;
goto v_resetjp_2406_;
}
v_resetjp_2406_:
{
lean_object* v___x_2409_; lean_object* v___x_2411_; 
v___x_2409_ = lean_array_to_list(v_a_2405_);
lean_inc(v___x_2409_);
if (v_isShared_2408_ == 0)
{
lean_ctor_set(v___x_2407_, 0, v___x_2409_);
v___x_2411_ = v___x_2407_;
goto v_reusejp_2410_;
}
else
{
lean_object* v_reuseFailAlloc_2412_; 
v_reuseFailAlloc_2412_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2412_, 0, v___x_2409_);
v___x_2411_ = v_reuseFailAlloc_2412_;
goto v_reusejp_2410_;
}
v_reusejp_2410_:
{
v___y_2390_ = v___x_2411_;
v_a_2391_ = v___x_2409_;
goto v___jp_2389_;
}
}
}
else
{
lean_object* v_a_2414_; lean_object* v___x_2416_; uint8_t v_isShared_2417_; uint8_t v_isSharedCheck_2421_; 
lean_dec(v_g_2380_);
v_a_2414_ = lean_ctor_get(v___x_2404_, 0);
v_isSharedCheck_2421_ = !lean_is_exclusive(v___x_2404_);
if (v_isSharedCheck_2421_ == 0)
{
v___x_2416_ = v___x_2404_;
v_isShared_2417_ = v_isSharedCheck_2421_;
goto v_resetjp_2415_;
}
else
{
lean_inc(v_a_2414_);
lean_dec(v___x_2404_);
v___x_2416_ = lean_box(0);
v_isShared_2417_ = v_isSharedCheck_2421_;
goto v_resetjp_2415_;
}
v_resetjp_2415_:
{
lean_object* v___x_2419_; 
if (v_isShared_2417_ == 0)
{
v___x_2419_ = v___x_2416_;
goto v_reusejp_2418_;
}
else
{
lean_object* v_reuseFailAlloc_2420_; 
v_reuseFailAlloc_2420_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2420_, 0, v_a_2414_);
v___x_2419_ = v_reuseFailAlloc_2420_;
goto v_reusejp_2418_;
}
v_reusejp_2418_:
{
return v___x_2419_;
}
}
}
}
v___jp_2389_:
{
if (v_throwOnNoMatch_2383_ == 0)
{
lean_dec(v_a_2391_);
lean_dec(v_g_2380_);
return v___y_2390_;
}
else
{
lean_object* v___x_2392_; lean_object* v___x_2393_; uint8_t v___x_2394_; 
v___x_2392_ = lean_box(0);
v___x_2393_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_2393_, 0, v_g_2380_);
lean_ctor_set(v___x_2393_, 1, v___x_2392_);
v___x_2394_ = lp_mathlib_List_beq___at___00Lean_MVarId_casesMatching_spec__0(v___x_2393_, v_a_2391_);
lean_dec(v_a_2391_);
lean_dec_ref_known(v___x_2393_, 2);
if (v___x_2394_ == 0)
{
return v___y_2390_;
}
else
{
lean_object* v___x_2395_; lean_object* v___x_2396_; 
lean_dec_ref(v___y_2390_);
v___x_2395_ = lean_obj_once(&lp_mathlib_Lean_MVarId_casesMatching___closed__2, &lp_mathlib_Lean_MVarId_casesMatching___closed__2_once, _init_lp_mathlib_Lean_MVarId_casesMatching___closed__2);
v___x_2396_ = lp_mathlib_Lean_throwError___at___00Lean_MVarId_casesMatching_spec__1___redArg(v___x_2395_, v_a_2384_, v_a_2385_, v_a_2386_, v_a_2387_);
return v___x_2396_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_constructorMatching___boxed(lean_object* v_g_2422_, lean_object* v_matcher_2423_, lean_object* v_recursive_2424_, lean_object* v_throwOnNoMatch_2425_, lean_object* v_a_2426_, lean_object* v_a_2427_, lean_object* v_a_2428_, lean_object* v_a_2429_, lean_object* v_a_2430_){
_start:
{
uint8_t v_recursive_boxed_2431_; uint8_t v_throwOnNoMatch_boxed_2432_; lean_object* v_res_2433_; 
v_recursive_boxed_2431_ = lean_unbox(v_recursive_2424_);
v_throwOnNoMatch_boxed_2432_ = lean_unbox(v_throwOnNoMatch_2425_);
v_res_2433_ = lp_mathlib_Mathlib_Tactic_constructorMatching(v_g_2422_, v_matcher_2423_, v_recursive_boxed_2431_, v_throwOnNoMatch_boxed_2432_, v_a_2426_, v_a_2427_, v_a_2428_, v_a_2429_);
lean_dec(v_a_2429_);
lean_dec_ref(v_a_2428_);
lean_dec(v_a_2427_);
lean_dec_ref(v_a_2426_);
return v_res_2433_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__CasesM______elabRules__Mathlib__Tactic__constructorM__1___lam__0(lean_object* v_a_2460_, lean_object* v___y_2461_, uint8_t v___x_2462_, lean_object* v___y_2463_, lean_object* v___y_2464_, lean_object* v___y_2465_, lean_object* v___y_2466_, lean_object* v___y_2467_, lean_object* v___y_2468_, lean_object* v___y_2469_, lean_object* v___y_2470_){
_start:
{
lean_object* v___y_2473_; lean_object* v___x_2493_; 
v___x_2493_ = l_Lean_Elab_Tactic_getMainGoal___redArg(v___y_2464_, v___y_2467_, v___y_2468_, v___y_2469_, v___y_2470_);
if (lean_obj_tag(v___x_2493_) == 0)
{
lean_object* v_a_2494_; lean_object* v___x_2495_; 
v_a_2494_ = lean_ctor_get(v___x_2493_, 0);
lean_inc(v_a_2494_);
lean_dec_ref_known(v___x_2493_, 1);
v___x_2495_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_Tactic_matchPatterns___boxed), 7, 1);
lean_closure_set(v___x_2495_, 0, v_a_2460_);
if (lean_obj_tag(v___y_2461_) == 0)
{
uint8_t v___x_2496_; lean_object* v___x_2497_; 
v___x_2496_ = 0;
v___x_2497_ = lp_mathlib_Mathlib_Tactic_constructorMatching(v_a_2494_, v___x_2495_, v___x_2496_, v___x_2462_, v___y_2467_, v___y_2468_, v___y_2469_, v___y_2470_);
v___y_2473_ = v___x_2497_;
goto v___jp_2472_;
}
else
{
lean_object* v___x_2498_; 
v___x_2498_ = lp_mathlib_Mathlib_Tactic_constructorMatching(v_a_2494_, v___x_2495_, v___x_2462_, v___x_2462_, v___y_2467_, v___y_2468_, v___y_2469_, v___y_2470_);
v___y_2473_ = v___x_2498_;
goto v___jp_2472_;
}
}
else
{
lean_object* v_a_2499_; lean_object* v___x_2501_; uint8_t v_isShared_2502_; uint8_t v_isSharedCheck_2506_; 
lean_dec_ref(v_a_2460_);
v_a_2499_ = lean_ctor_get(v___x_2493_, 0);
v_isSharedCheck_2506_ = !lean_is_exclusive(v___x_2493_);
if (v_isSharedCheck_2506_ == 0)
{
v___x_2501_ = v___x_2493_;
v_isShared_2502_ = v_isSharedCheck_2506_;
goto v_resetjp_2500_;
}
else
{
lean_inc(v_a_2499_);
lean_dec(v___x_2493_);
v___x_2501_ = lean_box(0);
v_isShared_2502_ = v_isSharedCheck_2506_;
goto v_resetjp_2500_;
}
v_resetjp_2500_:
{
lean_object* v___x_2504_; 
if (v_isShared_2502_ == 0)
{
v___x_2504_ = v___x_2501_;
goto v_reusejp_2503_;
}
else
{
lean_object* v_reuseFailAlloc_2505_; 
v_reuseFailAlloc_2505_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2505_, 0, v_a_2499_);
v___x_2504_ = v_reuseFailAlloc_2505_;
goto v_reusejp_2503_;
}
v_reusejp_2503_:
{
return v___x_2504_;
}
}
}
v___jp_2472_:
{
if (lean_obj_tag(v___y_2473_) == 0)
{
lean_object* v_a_2474_; lean_object* v___x_2475_; 
v_a_2474_ = lean_ctor_get(v___y_2473_, 0);
lean_inc(v_a_2474_);
lean_dec_ref_known(v___y_2473_, 1);
v___x_2475_ = l_Lean_Elab_Tactic_replaceMainGoal___redArg(v_a_2474_, v___y_2464_, v___y_2467_, v___y_2468_, v___y_2469_, v___y_2470_);
if (lean_obj_tag(v___x_2475_) == 0)
{
lean_object* v___x_2477_; uint8_t v_isShared_2478_; uint8_t v_isSharedCheck_2483_; 
v_isSharedCheck_2483_ = !lean_is_exclusive(v___x_2475_);
if (v_isSharedCheck_2483_ == 0)
{
lean_object* v_unused_2484_; 
v_unused_2484_ = lean_ctor_get(v___x_2475_, 0);
lean_dec(v_unused_2484_);
v___x_2477_ = v___x_2475_;
v_isShared_2478_ = v_isSharedCheck_2483_;
goto v_resetjp_2476_;
}
else
{
lean_dec(v___x_2475_);
v___x_2477_ = lean_box(0);
v_isShared_2478_ = v_isSharedCheck_2483_;
goto v_resetjp_2476_;
}
v_resetjp_2476_:
{
lean_object* v___x_2479_; lean_object* v___x_2481_; 
v___x_2479_ = lean_box(0);
if (v_isShared_2478_ == 0)
{
lean_ctor_set(v___x_2477_, 0, v___x_2479_);
v___x_2481_ = v___x_2477_;
goto v_reusejp_2480_;
}
else
{
lean_object* v_reuseFailAlloc_2482_; 
v_reuseFailAlloc_2482_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2482_, 0, v___x_2479_);
v___x_2481_ = v_reuseFailAlloc_2482_;
goto v_reusejp_2480_;
}
v_reusejp_2480_:
{
return v___x_2481_;
}
}
}
else
{
return v___x_2475_;
}
}
else
{
lean_object* v_a_2485_; lean_object* v___x_2487_; uint8_t v_isShared_2488_; uint8_t v_isSharedCheck_2492_; 
v_a_2485_ = lean_ctor_get(v___y_2473_, 0);
v_isSharedCheck_2492_ = !lean_is_exclusive(v___y_2473_);
if (v_isSharedCheck_2492_ == 0)
{
v___x_2487_ = v___y_2473_;
v_isShared_2488_ = v_isSharedCheck_2492_;
goto v_resetjp_2486_;
}
else
{
lean_inc(v_a_2485_);
lean_dec(v___y_2473_);
v___x_2487_ = lean_box(0);
v_isShared_2488_ = v_isSharedCheck_2492_;
goto v_resetjp_2486_;
}
v_resetjp_2486_:
{
lean_object* v___x_2490_; 
if (v_isShared_2488_ == 0)
{
v___x_2490_ = v___x_2487_;
goto v_reusejp_2489_;
}
else
{
lean_object* v_reuseFailAlloc_2491_; 
v_reuseFailAlloc_2491_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2491_, 0, v_a_2485_);
v___x_2490_ = v_reuseFailAlloc_2491_;
goto v_reusejp_2489_;
}
v_reusejp_2489_:
{
return v___x_2490_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__CasesM______elabRules__Mathlib__Tactic__constructorM__1___lam__0___boxed(lean_object* v_a_2507_, lean_object* v___y_2508_, lean_object* v___x_2509_, lean_object* v___y_2510_, lean_object* v___y_2511_, lean_object* v___y_2512_, lean_object* v___y_2513_, lean_object* v___y_2514_, lean_object* v___y_2515_, lean_object* v___y_2516_, lean_object* v___y_2517_, lean_object* v___y_2518_){
_start:
{
uint8_t v___x_661__boxed_2519_; lean_object* v_res_2520_; 
v___x_661__boxed_2519_ = lean_unbox(v___x_2509_);
v_res_2520_ = lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__CasesM______elabRules__Mathlib__Tactic__constructorM__1___lam__0(v_a_2507_, v___y_2508_, v___x_661__boxed_2519_, v___y_2510_, v___y_2511_, v___y_2512_, v___y_2513_, v___y_2514_, v___y_2515_, v___y_2516_, v___y_2517_);
lean_dec(v___y_2517_);
lean_dec_ref(v___y_2516_);
lean_dec(v___y_2515_);
lean_dec_ref(v___y_2514_);
lean_dec(v___y_2513_);
lean_dec_ref(v___y_2512_);
lean_dec(v___y_2511_);
lean_dec_ref(v___y_2510_);
lean_dec(v___y_2508_);
return v_res_2520_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__CasesM______elabRules__Mathlib__Tactic__constructorM__1(lean_object* v_x_2521_, lean_object* v_a_2522_, lean_object* v_a_2523_, lean_object* v_a_2524_, lean_object* v_a_2525_, lean_object* v_a_2526_, lean_object* v_a_2527_, lean_object* v_a_2528_, lean_object* v_a_2529_){
_start:
{
lean_object* v___x_2531_; uint8_t v___x_2532_; 
v___x_2531_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_constructorM___closed__1));
lean_inc(v_x_2521_);
v___x_2532_ = l_Lean_Syntax_isOfKind(v_x_2521_, v___x_2531_);
if (v___x_2532_ == 0)
{
lean_object* v___x_2533_; 
lean_dec(v_x_2521_);
v___x_2533_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic___aux__Mathlib__Tactic__CasesM______elabRules__Mathlib__Tactic__casesM__1_spec__0___redArg();
return v___x_2533_;
}
else
{
lean_object* v___x_2534_; lean_object* v___x_2535_; lean_object* v___x_2536_; lean_object* v___x_2537_; lean_object* v_pats_2538_; lean_object* v___y_2540_; lean_object* v___x_2555_; 
v___x_2534_ = lean_unsigned_to_nat(1u);
v___x_2535_ = l_Lean_Syntax_getArg(v_x_2521_, v___x_2534_);
v___x_2536_ = lean_unsigned_to_nat(3u);
v___x_2537_ = l_Lean_Syntax_getArg(v_x_2521_, v___x_2536_);
lean_dec(v_x_2521_);
v_pats_2538_ = l_Lean_Syntax_getArgs(v___x_2537_);
lean_dec(v___x_2537_);
v___x_2555_ = l_Lean_Syntax_getOptional_x3f(v___x_2535_);
lean_dec(v___x_2535_);
if (lean_obj_tag(v___x_2555_) == 0)
{
lean_object* v___x_2556_; 
v___x_2556_ = lean_box(0);
v___y_2540_ = v___x_2556_;
goto v___jp_2539_;
}
else
{
lean_object* v_val_2557_; lean_object* v___x_2559_; uint8_t v_isShared_2560_; uint8_t v_isSharedCheck_2564_; 
v_val_2557_ = lean_ctor_get(v___x_2555_, 0);
v_isSharedCheck_2564_ = !lean_is_exclusive(v___x_2555_);
if (v_isSharedCheck_2564_ == 0)
{
v___x_2559_ = v___x_2555_;
v_isShared_2560_ = v_isSharedCheck_2564_;
goto v_resetjp_2558_;
}
else
{
lean_inc(v_val_2557_);
lean_dec(v___x_2555_);
v___x_2559_ = lean_box(0);
v_isShared_2560_ = v_isSharedCheck_2564_;
goto v_resetjp_2558_;
}
v_resetjp_2558_:
{
lean_object* v___x_2562_; 
if (v_isShared_2560_ == 0)
{
v___x_2562_ = v___x_2559_;
goto v_reusejp_2561_;
}
else
{
lean_object* v_reuseFailAlloc_2563_; 
v_reuseFailAlloc_2563_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2563_, 0, v_val_2557_);
v___x_2562_ = v_reuseFailAlloc_2563_;
goto v_reusejp_2561_;
}
v_reusejp_2561_:
{
v___y_2540_ = v___x_2562_;
goto v___jp_2539_;
}
}
}
v___jp_2539_:
{
lean_object* v___x_2541_; lean_object* v___x_2542_; 
v___x_2541_ = l_Lean_Syntax_TSepArray_getElems___redArg(v_pats_2538_);
lean_dec_ref(v_pats_2538_);
v___x_2542_ = lp_mathlib_Mathlib_Tactic_elabPatterns(v___x_2541_, v_a_2524_, v_a_2525_, v_a_2526_, v_a_2527_, v_a_2528_, v_a_2529_);
if (lean_obj_tag(v___x_2542_) == 0)
{
lean_object* v_a_2543_; lean_object* v___x_2544_; lean_object* v___f_2545_; lean_object* v___x_2546_; 
v_a_2543_ = lean_ctor_get(v___x_2542_, 0);
lean_inc(v_a_2543_);
lean_dec_ref_known(v___x_2542_, 1);
v___x_2544_ = lean_box(v___x_2532_);
v___f_2545_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__CasesM______elabRules__Mathlib__Tactic__constructorM__1___lam__0___boxed), 12, 3);
lean_closure_set(v___f_2545_, 0, v_a_2543_);
lean_closure_set(v___f_2545_, 1, v___y_2540_);
lean_closure_set(v___f_2545_, 2, v___x_2544_);
v___x_2546_ = l_Lean_Elab_Tactic_withMainContext___redArg(v___f_2545_, v_a_2522_, v_a_2523_, v_a_2524_, v_a_2525_, v_a_2526_, v_a_2527_, v_a_2528_, v_a_2529_);
return v___x_2546_;
}
else
{
lean_object* v_a_2547_; lean_object* v___x_2549_; uint8_t v_isShared_2550_; uint8_t v_isSharedCheck_2554_; 
lean_dec(v___y_2540_);
v_a_2547_ = lean_ctor_get(v___x_2542_, 0);
v_isSharedCheck_2554_ = !lean_is_exclusive(v___x_2542_);
if (v_isSharedCheck_2554_ == 0)
{
v___x_2549_ = v___x_2542_;
v_isShared_2550_ = v_isSharedCheck_2554_;
goto v_resetjp_2548_;
}
else
{
lean_inc(v_a_2547_);
lean_dec(v___x_2542_);
v___x_2549_ = lean_box(0);
v_isShared_2550_ = v_isSharedCheck_2554_;
goto v_resetjp_2548_;
}
v_resetjp_2548_:
{
lean_object* v___x_2552_; 
if (v_isShared_2550_ == 0)
{
v___x_2552_ = v___x_2549_;
goto v_reusejp_2551_;
}
else
{
lean_object* v_reuseFailAlloc_2553_; 
v_reuseFailAlloc_2553_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2553_, 0, v_a_2547_);
v___x_2552_ = v_reuseFailAlloc_2553_;
goto v_reusejp_2551_;
}
v_reusejp_2551_:
{
return v___x_2552_;
}
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__CasesM______elabRules__Mathlib__Tactic__constructorM__1___boxed(lean_object* v_x_2565_, lean_object* v_a_2566_, lean_object* v_a_2567_, lean_object* v_a_2568_, lean_object* v_a_2569_, lean_object* v_a_2570_, lean_object* v_a_2571_, lean_object* v_a_2572_, lean_object* v_a_2573_, lean_object* v_a_2574_){
_start:
{
lean_object* v_res_2575_; 
v_res_2575_ = lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__CasesM______elabRules__Mathlib__Tactic__constructorM__1(v_x_2565_, v_a_2566_, v_a_2567_, v_a_2568_, v_a_2569_, v_a_2570_, v_a_2571_, v_a_2572_, v_a_2573_);
lean_dec(v_a_2573_);
lean_dec_ref(v_a_2572_);
lean_dec(v_a_2571_);
lean_dec_ref(v_a_2570_);
lean_dec(v_a_2569_);
lean_dec_ref(v_a_2568_);
lean_dec(v_a_2567_);
lean_dec_ref(v_a_2566_);
return v_res_2575_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Init(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_Tactic_CasesM(uint8_t builtin) {
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
lean_object* runtime_initialize_Lean_Elab_Tactic_Conv_Pattern(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_Tactic_CasesM(uint8_t builtin) {
lean_object * res;
if (_G_meta_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_meta_initialized = true;
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_Lean_Elab_Tactic_Conv_Pattern(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Init(uint8_t builtin);
lean_object* initialize_Lean_Elab_Tactic_Conv_Pattern(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Tactic_CasesM(uint8_t builtin) {
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
res = initialize_Lean_Elab_Tactic_Conv_Pattern(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Tactic_CasesM(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_Tactic_CasesM(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_Tactic_CasesM(builtin);
}
#ifdef __cplusplus
}
#endif
